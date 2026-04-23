import os
from pathlib import Path
import numpy as np
import requests
import json
from multiprocessing import cpu_count
from functools import partial

import hydra
import pyrootutils
import polars as pl
from tqdm import tqdm
import concurrent.futures
import orjson

from datasets import (
    Image,
    Value,
    concatenate_datasets,
    load_dataset,
)

from planktonzilla.utils.logger import get_pylogger

# =============================================================================
# CONFIGURATION
# =============================================================================
root = pyrootutils.setup_root(
    search_from=".",  
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

logger = get_pylogger(__name__)
num_proc = int(cpu_count() / 2)


# =============================================================================
# METADATA RETRIEVAL FUNCTIONS
# =============================================================================

def retrieve_whoi_metadata(bin_id, session=None):
    """Retrieve metadata from WHOI API for a given bin_id."""
    api_url = f"https://ifcb-data.whoi.edu/api/bin/{bin_id}"
    hdr_url = f"https://ifcb-data.whoi.edu/mvco/{bin_id}.hdr"

    requester = session or requests

    info = {
        "Latitude": np.nan,
        "Longitude": np.nan,
        "Depth": np.nan,
        "Temperature": np.nan,
        "Humidity": np.nan,
        "BinID": str(bin_id),
    }

    try:
        # ---------- JSON metadata ----------
        r = requester.get(api_url, timeout=10)
        if r.ok:
            data = r.json()
            info["Latitude"] = data.get("lat")
            info["Longitude"] = data.get("lng")
            info["Depth"] = data.get("depth")

        # ---------- HDR metadata ----------
        r = requester.get(hdr_url, timeout=10)
        if r.ok:
            lines = r.text.splitlines()

            for idx, line in enumerate(lines):
                if "Temp Humidity" in line and idx + 1 < len(lines):
                    headers = line.replace('"', '').split()
                    values = lines[idx + 1].replace('"', '').split(",")

                    if len(values) < len(headers):
                        values = lines[idx + 1].split()

                    mapping = dict(zip(headers, values))
                    info["Temperature"] = mapping.get("Temp")
                    info["Humidity"] = mapping.get("Humidity")
                    break

        # ---------- Fast float cast ----------
        for k in ("Latitude", "Longitude", "Depth", "Temperature", "Humidity"):
            v = info[k]
            info[k] = float(v) if v not in (None, "", np.nan) else np.nan

    except Exception:
        pass

    return info


def retrieve_ecotaxa_metadata(obj_id, session=None):
    """Retrieve metadata from EcoTaxa API for a given object_id."""
    api_url = f"https://ecotaxa.obs-vlfr.fr/api/object/{obj_id}"

    info = {
        "Depth_max": np.nan,
        "Depth_min": np.nan,
        "Latitude": np.nan,
        "Longitude": np.nan,
        "ObjID": str(obj_id),
    }

    requester = session if session else requests

    try:
        response = requester.get(api_url, timeout=10)
        if response.status_code != 200:
            return info

        data = response.json()

        for src, dst in [
            ("depth_max", "Depth_max"),
            ("depth_min", "Depth_min"),
            ("latitude", "Latitude"),
            ("longitude", "Longitude"),
        ]:
            val = data.get(src)
            info[dst] = float(val) if val is not None else np.nan

    except (requests.RequestException, ValueError, TypeError):
        pass

    return info


# =============================================================================
# REDEFINER CLASSES FOR TAXONOMY AND METADATA PROCESSING
# =============================================================================

class RedefineDataset:
    """Base class for redefining datasets with taxonomy and metadata."""
    
    def __init__(self, csv_taxonomies_path):
        self.csv_tax = pl.read_csv(csv_taxonomies_path).fill_null("")

        # Taxonomy columns
        self.taxonomy_cols = [
            "Kingdom", "Phylum", "Class",
            "Order", "Family", "Genus", "Species"
        ]
        
        # Extra columns (including new ones: root_class, qualifier)
        self.extra_cols = ["proposed_label", "plankton", "root_class", "qualifier"]
        self.all_cols = self.taxonomy_cols + self.extra_cols

        # Create lookup dictionary from CSV
        keys = zip(self.csv_tax["Dataset"], self.csv_tax["Raw_Labels"])
        values = self.csv_tax.select(self.all_cols).to_dicts()
        self.lookup = dict(zip(keys, values))

        # Metadata column names that will be extracted from JSON
        self.metadata_cols_final = [
            "Latitude", "Humidity", "Temperature", "Longitude",
            "ObjID", "Depth_max", "Depth_min"
        ]

    def _add_metadata(self, processed_ds):
        """Add metadata to dataset. Must be implemented by subclasses."""
        raise NotImplementedError()

    def _flatten_metadata(self, processed_ds):
        """
        Convert metadata JSON string to flattened columns.
        Integrates the transformation logic from update_dataset.ipynb
        """
        def extract_metadata_fields(example):
            # Parse JSON metadata
            try:
                md = orjson.loads(example["metadata"]) if example["metadata"] else {}
            except Exception:
                md = {}
            
            # Initialize all metadata columns
            for col in self.metadata_cols_final:
                example[col] = None
            
            # Extract ObjID (prioritize ObjID, fallback to BinID)
            obj_val = md.get("ObjID") if md.get("ObjID") is not None else md.get("BinID")
            example["ObjID"] = str(obj_val) if obj_val not in (None, "") else None
            
            # Extract Depth
            depth_val = md.get("Depth")
            if depth_val not in (None, ""):
                example["Depth_max"] = np.float32(depth_val)
                example["Depth_min"] = np.float32(depth_val)
            else:
                d_max = md.get("Depth_max")
                d_min = md.get("Depth_min")
                example["Depth_max"] = np.float32(d_max) if d_max not in (None, "") else None
                example["Depth_min"] = np.float32(d_min) if d_min not in (None, "") else None
            
            # Extract other numeric metadata
            for col in ["Latitude", "Humidity", "Temperature", "Longitude"]:
                val = md.get(col)
                example[col] = np.float32(val) if val not in (None, "") else None
            
            return example
        
        # Apply flattening
        processed_ds = processed_ds.map(
            extract_metadata_fields,
            desc="Flattening metadata from JSON",
            num_proc=num_proc,
        )
        
        # Remove original metadata column
        processed_ds = processed_ds.remove_columns("metadata")
        
        return processed_ds

    def redefine(self, hf_dataset, dataset_name, num_proc):
        """Main method to redefine dataset with taxonomy and metadata."""
        
        ds_list = []
        n_splits = len(hf_dataset)

        for split in hf_dataset.keys():
            ds = hf_dataset[split]
            class_names = ds.features["label"].names

            ds = ds.cast_column("image", Image(decode=False))

            def process_row(example):
                label_str = class_names[example["label"]]
                full_path = example["image"]["path"]

                parts = full_path.split(os.sep)
                short_path = (
                    "/" + "/".join(parts[-3:])
                    if n_splits >= 2
                    else "/" + "/".join(parts[-2:])
                )

                # Lookup taxonomy from CSV
                tax_data = self.lookup.get(
                    (dataset_name, label_str),
                    {col: None for col in self.all_cols},
                )

                return {
                    "dataset": dataset_name,
                    "original_label": label_str,
                    "original_path": short_path,
                    **tax_data,
                }

            print(f"Processing {split} split...")

            processed_ds = ds.map(
                process_row,
                desc="Taxonomy mapping",
                num_proc=num_proc,
            )

            # Add metadata (specific to dataset type)
            processed_ds = self._add_metadata(processed_ds)

            # Flatten metadata from JSON to independent columns
            processed_ds = self._flatten_metadata(processed_ds)

            # Remove original label column
            if "label" in processed_ds.column_names:
                processed_ds = processed_ds.remove_columns("label")

            # Decode images
            processed_ds = processed_ds.cast_column("image", Image(decode=True))

            ds_list.append(processed_ds)

        return concatenate_datasets(ds_list)


class EcoTaxaRedefiner(RedefineDataset):
    """Redefiner for EcoTaxa-sourced datasets (e.g., flowcamnet, uvp6net, zooscan)."""

    def _add_metadata(self, processed_ds):
        """Retrieve metadata from EcoTaxa API."""
        
        ids = [
            path.split("/")[-1].split(".")[0]
            for path in processed_ds["original_path"]
        ]

        # Parallel retrieval from EcoTaxa API
        with requests.Session() as session:
            func = partial(retrieve_ecotaxa_metadata, session=session)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
                raw = list(tqdm(executor.map(func, ids), total=len(ids), desc="Retrieving EcoTaxa metadata"))

        # Normalize metadata dicts to strings
        def normalize_metadata(md: dict | None) -> dict:
            if not md:
                return {}
            return {str(k): str(v) for k, v in md.items() if v is not None}

        metadata = [normalize_metadata(r) for r in raw]

        # Add as JSON string column
        processed_ds = processed_ds.add_column("metadata", metadata)
        
        # Serialize to JSON string
        def to_json(example):
            return {"metadata": json.dumps(example["metadata"])}
        
        processed_ds = processed_ds.map(to_json, desc="Serializing metadata", num_proc=num_proc)
        
        # Cast to string type
        features = processed_ds.features.copy()
        features["metadata"] = Value("string")
        processed_ds = processed_ds.cast(features)
        
        return processed_ds


class NoMetadataRedefiner(RedefineDataset):
    """Redefiner for datasets without external metadata (e.g., lensless, medplanktonset, zoolake)."""

    def _add_metadata(self, processed_ds):
        """Add empty metadata column."""
        n = len(processed_ds)
        processed_ds = processed_ds.add_column("metadata", [{}] * n)
        
        # Serialize to JSON string
        def to_json(example):
            return {"metadata": json.dumps(example["metadata"])}
        
        processed_ds = processed_ds.map(to_json, desc="Serializing metadata", num_proc=num_proc)
        
        # Cast to string type
        features = processed_ds.features.copy()
        features["metadata"] = Value("string")
        processed_ds = processed_ds.cast(features)
        
        return processed_ds


class WHOIRedefiner(RedefineDataset):
    """Redefiner for WHOI-sourced datasets."""

    def _add_metadata(self, processed_ds):
        """Retrieve metadata from WHOI API based on bin_id."""
        
        def extract_bin_id(example):
            fname = example["original_path"].split("/")[-1]
            parts = fname.split(".")[0].split("_")[:-1]
            return {"bin_id": "_".join(parts)}

        processed_ds = processed_ds.map(
            extract_bin_id,
            desc="Extracting WHOI bin_id"
        )

        bin_ids = np.unique(processed_ds["bin_id"])
        print(f"{len(bin_ids)} unique bin_ids found")

        bin_id_lookup = {}

        # Parallel retrieval from WHOI API
        with requests.Session() as session:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
                futures = {
                    executor.submit(retrieve_whoi_metadata, bin_id, session): bin_id
                    for bin_id in bin_ids
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Retrieving WHOI metadata"
                ):
                    bin_id = futures[future]
                    try:
                        raw = future.result()
                        bin_id_lookup[bin_id] = {
                            str(k): str(v)
                            for k, v in raw.items()
                            if v is not None
                        }
                    except Exception:
                        bin_id_lookup[bin_id] = {}

        def add_metadata(example):
            return {"metadata": bin_id_lookup.get(example["bin_id"], {})}

        processed_ds = processed_ds.map(
            add_metadata,
            desc="Attaching WHOI metadata"
        )

        processed_ds = processed_ds.remove_columns("bin_id")
        
        # Serialize to JSON string
        def to_json(example):
            return {"metadata": json.dumps(example["metadata"])}
        
        processed_ds = processed_ds.map(to_json, desc="Serializing metadata", num_proc=num_proc)
        
        # Cast to string type
        features = processed_ds.features.copy()
        features["metadata"] = Value("string")
        processed_ds = processed_ds.cast(features)
        
        return processed_ds


class JediRedefiner(RedefineDataset):
    """Redefiner for JEDI Oceans dataset with fixed metadata."""
    
    def __init__(self, csv_taxonomies_path: str):
        super().__init__(csv_taxonomies_path)
        
        # Static metadata for JEDI dataset
        self.metadata = {
            "Latitude": "34.682718",
            "Longitude": "139.444779",
            "Depth_min": "20",
            "Depth_max": "20",
        }

    def _add_metadata(self, processed_ds):
        """Add fixed metadata for JEDI dataset."""
        metadata_column = [self.metadata] * len(processed_ds)
        processed_ds = processed_ds.add_column("metadata", metadata_column)
        
        # Serialize to JSON string
        def to_json(example):
            return {"metadata": json.dumps(example["metadata"])}
        
        processed_ds = processed_ds.map(to_json, desc="Serializing metadata", num_proc=num_proc)
        
        # Cast to string type
        features = processed_ds.features.copy()
        features["metadata"] = Value("string")
        processed_ds = processed_ds.cast(features)
        
        return processed_ds


# =============================================================================
# MAIN FUNCTION AND DATASET CONFIGURATION
# =============================================================================

def main():
    """
    Generate imagefolders for all datasets and create HuggingFace datasets
    with taxonomies and metadata.
    
    Note: The Zoolake and SYKE Zooscan 2024 datasets have anti-bot protection
    on their download URLs. You must manually download the .zip files and 
    provide their file paths:
    
    - Zoolake: https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip
    - SYKE Zooscan 2024: https://etsin.fairdata.fi/dataset/6fa42787-9772-41a5-a6fc-0dde489ed908/data
    """

    # =================================================================
    # CONFIGURATION PATHS
    # =================================================================
    DATA_ROOT = Path("/lustre/fsn1/projects/rech/tec/uod68bo/data").resolve()
    
    # Path to taxonomy CSV
    taxo_csv_path = "/lustre/fswork/projects/rech/tec/uod68bo/am/planktonzilla/notebooks/planktonzilla_taxo.csv"

    # Paths for manually downloaded datasets (comment out if files are not available)
    # path_zip_jedi = DATA_ROOT / "CPICS_Validated.zip"
    # path_zip_zoolake = DATA_ROOT / "zoolake_data.zip"
    # path_zip_syze_zooscan2024 = DATA_ROOT / "SYKE-plankton_ZooScan_2024.zip"

    # For now, these are undefined. Uncomment above and use them in overrides below
    path_zip_jedi = None
    path_zip_zoolake = None
    path_zip_syze_zooscan2024 = None
    
    datasets_configs = {
        "isiisnet": {
            "overrides": [
                "dataset_import=isiisnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "whoi": {
            "overrides": [
                "dataset_import=whoi-plankton",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": WHOIRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "flowcamnet": {
            "overrides": [
                "dataset_import=flowcamnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        # "jedi_oceans_cpics": {
        #     "overrides": [
        #         "dataset_import=jedi",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={path_zip_jedi}"
        #     ],
        #     "redefiner": JediRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },

        "lensless": {
            "overrides": [
                "dataset_import=lensless",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "medplanktonset": {
            "overrides": [
                "dataset_import=medplanktonset",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        # "sykezooscan2024": {
        #     "overrides": [
        #         "dataset_import=sykezooscan2024",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={path_zip_syze_zooscan2024}"
        #     ],
        #     "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },

        "uvp6net": {
            "overrides": [
                "dataset_import=uvp6net",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "zoocamnet": {
            "overrides": [
                "dataset_import=zoocamnet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        # "zoolake": {
        #     "overrides": [
        #         "dataset_import=zoolake",
        #         "dataset_import.cleanup_after_processing=True",
        #         "dataset_import.push_to_hub=False",
        #         f"dataset_import.data_dir={DATA_ROOT}",
        #         f"dataset_import.manual_download_local_file_names={path_zip_zoolake}"
        #     ],
        #     "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        # },

        "zooscan": {
            "overrides": [
                "dataset_import=zooscannet",
                "dataset_import.cleanup_after_processing=True",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "planktonset1.0": {
            "overrides": [
                "dataset_import=planktonset1",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "syke_ifcb_2022": {
            "overrides": [
                "dataset_import=syke_ifcb_2022",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "planktoscope": {
            "overrides": [
                "dataset_import=planktoscope",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },

        "global_uvp5": {
            "overrides": [
                "dataset_import=global_uvp5net",
                "dataset_import.cleanup_after_processing=False",
                "dataset_import.push_to_hub=False",
                f"dataset_import.data_dir={DATA_ROOT}",
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path=taxo_csv_path),
        },
    }

    ds = []

    with hydra.initialize(version_base="1.3", config_path="../configs"):

        for dataset_name, ds_cfg in datasets_configs.items():
            print(f"\n=== Dataset: {dataset_name} ===")

            cfg = hydra.compose(
                config_name="import_dataset",
                overrides=ds_cfg["overrides"]
            )

            dataset_importer = hydra.utils.instantiate(cfg.dataset_import)
            imagefolder_dir = Path(dataset_importer.imagefolder_dir)

            has_content = imagefolder_dir.exists() and bool(os.listdir(imagefolder_dir))

            if has_content:
                num_items = len(os.listdir(imagefolder_dir))
                print(f" Using existing imagefolder with {num_items} categories at:")
                print(f" {imagefolder_dir}")
            else:
                print(f" Building imagefolder from raw data...")
                dataset_importer.import_dataset()

            # Determine data files for HuggingFace dataset
            split_aliases = {
                "train": ["train"],
                "validation": ["validation", "val"],
                "test": ["test"],
            }

            data_files = {}

            for canonical_split, aliases in split_aliases.items():
                for alias in aliases:
                    split_path = root / alias
                    if split_path.exists():
                        data_files[canonical_split] = str(split_path / "*/[!._]*")
                        break

            # Fallback: no splits
            if not data_files:
                data_files = {
                    "train": str(dataset_importer.imagefolder_dir / "*/*[!._]*")
                }

            # Load dataset with HuggingFace imagefolder loader
            print(f"Loading dataset with imagefolder loader...")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
            )

            # Redefine with taxonomy and metadata
            print(f"Adding taxonomy and metadata...")
            dataset = ds_cfg["redefiner"].redefine(
                hf_dataset=dataset,
                dataset_name=dataset_name,
                num_proc=num_proc
            )

            # Save individual dataset
            # output_path = DATA_ROOT / f"{dataset_name}_hf"
            # print(f"Saving dataset to: {output_path}")
            # dataset.save_to_disk(output_path)

            ds.append(dataset)

    # Concatenate all datasets
    ds = concatenate_datasets(ds)
    
    # Save concatenated dataset
    output_path = DATA_ROOT / "planktonzilla_others"
    print(f"Saving concatenated dataset to: {output_path}")
    ds.save_to_disk(output_path)
    
    print(f"\n Process completed successfully!")


if __name__ == "__main__":
    main()
