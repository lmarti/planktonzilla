import os
import sys
import shutil
import subprocess
from pathlib import Path
from shutil import rmtree
import numpy as np
import requests
import time
import json


import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pyrootutils
import polars as pl
from tqdm import tqdm
import concurrent.futures

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk
)

from planktonzilla.utils.logger import get_pylogger
from planktonzilla.dataset_import.dataset_importer import (
    DatasetImporter,
    is_dir_empty,
    is_valid_image_file,
)

import json
from datasets import Value

from multiprocessing import cpu_count
num_proc = min(cpu_count(), 32)


root = pyrootutils.setup_root(
    search_from=".",  
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

logger = get_pylogger(__name__)

# ============= GENERATING HF DATASETS WITH METADATA ============= #


def cast_metadata_json(ds):
    # 1. Convert dict → JSON string
    def to_json(example):
        return {"metadata": json.dumps(example["metadata"])}

    ds = ds.map(to_json, desc="Serializing metadata")

    # 2. Cast feature
    features = ds.features.copy()
    features["metadata"] = Value("string")
    return ds.cast(features)


def retrieve_whoi_metadata(bin_id, session=None):
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


class RedefineDataset:
    def __init__(self, csv_taxonomies_path):

        self.csv_tax = pl.read_csv(csv_taxonomies_path).fill_null("")

        self.taxonomy_cols = [
            "Kingdom", "Phylum", "Class",
            "Order", "Family", "Genus", "Species"
        ]
        self.extra_cols = ["proposed_label", "plankton", "living"]
        self.all_cols = self.taxonomy_cols + self.extra_cols

        keys = zip(self.csv_tax["Dataset"], self.csv_tax["Raw_Labels"])
        values = self.csv_tax.select(self.all_cols).to_dicts()
        self.lookup = dict(zip(keys, values))

    def _add_metadata(self, processed_ds):
        raise NotImplementedError()

    def redefine(self, hf_dataset, dataset_name, num_proc):

        ds_list = []
        n_splits = len(hf_dataset)

        for split in hf_dataset.keys():
            ds = hf_dataset[split]
            class_names = ds.features["label"].names

            # ⬅️ critical optimization
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

            processed_ds = self._add_metadata(processed_ds)

            if "label" in processed_ds.column_names:
                processed_ds = processed_ds.remove_columns("label")

            processed_ds = processed_ds.cast_column("image", Image(decode=True))

            ds_list.append(processed_ds)

        return concatenate_datasets(ds_list)
    

class EcoTaxaRedefiner(RedefineDataset):

    def _add_metadata(self, processed_ds):

        ids = [
            path.split("/")[-1].split(".")[0]
            for path in processed_ds["original_path"]
        ]

        from functools import partial
        import concurrent.futures

            
        with requests.Session() as session:
            func = partial(retrieve_ecotaxa_metadata, session=session)
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                raw = list(tqdm(executor.map(func, ids), total=len(ids)))


        def normalize_metadata(md: dict | None) -> dict:
            if not md:
                return {}
            return {str(k): str(v) for k, v in md.items() if v is not None}
            
        metadata = [normalize_metadata(r) for r in raw]

        processed_ds = processed_ds.add_column("metadata", metadata)
        return cast_metadata_json(processed_ds)

class NoMetadataRedefiner(RedefineDataset):

    def _add_metadata(self, processed_ds):
        n = len(processed_ds)
        processed_ds = processed_ds.add_column("metadata", [{}] * n)
        return cast_metadata_json(processed_ds)


class WHOIRedefiner(RedefineDataset):

    def _add_metadata(self, processed_ds):

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

        with requests.Session() as session:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
            print()
            return {"metadata": bin_id_lookup.get(example["bin_id"], {})}

        processed_ds = processed_ds.map(
            add_metadata,
            desc="Attaching WHOI metadata"
        )

        processed_ds = processed_ds.remove_columns("bin_id")
        return cast_metadata_json(processed_ds)


class JediRedefiner(RedefineDataset):
    def __init__(self, csv_taxonomies_path: str):
        super().__init__(csv_taxonomies_path)

        self.metadata = {
            "Latitude": "34.682718",
            "Longitude": "139.444779",
            "Depth_min": "20",
            "Depth_max": "20",
        }

    def _add_metadata(self, processed_ds):
        metadata_column = [self.metadata] * len(processed_ds)
        processed_ds = processed_ds.add_column("metadata", metadata_column)

        return cast_metadata_json(processed_ds)

# ============= GENERATING HF DATASETS WITH METADATA ============= #


def main():

    # Note: the Zoolake and SYKE Zooscan 2024 URL has an anti-bot protection, so it cannot be downloaded using Python libraries. 
    # Therefore, you must manually download the .zip file, add it to the repository, and provide the file path.
    # zoolake url: https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip
    # SYKE Zooscan 2024 url: https://etsin.fairdata.fi/dataset/6fa42787-9772-41a5-a6fc-0dde489ed908/data

    DATA_ROOT = Path("/lustre/fsn1/projects/rech/tec/uod68bo/data").resolve()

    taxo_csv_path = DATA_ROOT / "planktonzilla_taxonomy.csv"

    datasets_configs = {
        "planktonset1": {
            "overrides": [
                "dataset_import=planktonset1",
                f"dataset_import.data_dir={DATA_ROOT}",
                "dataset_import.force_download=false",
                "dataset_import.push_to_hub=false"
            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path = taxo_csv_path)

        },

        "syke_ifcb_2022": {
            "overrides": [
                "dataset_import=syke_ifcb_2022",
                f"dataset_import.data_dir={DATA_ROOT}",
                "dataset_import.force_download=false",
                "dataset_import.push_to_hub=false"

            ],
            "redefiner": NoMetadataRedefiner(csv_taxonomies_path = taxo_csv_path)

        },

        "planktoscope": {
            "overrides": [
                "dataset_import=planktoscope",
                f"dataset_import.data_dir={DATA_ROOT}",
                "dataset_import.force_download=false",
                "dataset_import.push_to_hub=false"
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path = taxo_csv_path)

        },

        "global_uvp5net": {
            "overrides": [
                "dataset_import=global_uvp5net",
                f"dataset_import.data_dir={DATA_ROOT}",
                "dataset_import.force_download=false",
                "dataset_import.push_to_hub=false"
            ],
            "redefiner": EcoTaxaRedefiner(csv_taxonomies_path = taxo_csv_path)

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
            dataset_importer.import_dataset()

            dataset = load_dataset("imagefolder", data_dir=dataset_importer.imagefolder_dir)

            dataset = ds_cfg["redefiner"].redefine(
                hf_dataset=dataset,
                dataset_name=dataset_name,
                num_proc=num_proc
            )

            ds.append(dataset)

    ds = concatenate_datasets(ds)
    ds.save_to_disk(DATA_ROOT / "planktonzilla_others")

if __name__ == "__main__":
    main()