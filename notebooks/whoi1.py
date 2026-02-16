import os
import json
import time
from functools import partial
import concurrent.futures
from multiprocessing import cpu_count

import time
import numpy as np
import requests
import polars as pl
import pyrootutils
from tqdm import tqdm

# Hugging Face Datasets
from datasets import (
    Dataset,
    Image,
    Value,
    concatenate_datasets,
    load_dataset,
)


num_proc = min(cpu_count(), 32)


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
        # Replicar la MISMA referencia (no copiar dicts)
        metadata_column = [self.metadata] * len(processed_ds)
        processed_ds = processed_ds.add_column("metadata", metadata_column)

        return cast_metadata_json(processed_ds)


taxo_csv_path = "/lustre/fswork/projects/rech/tec/uod68bo/am/planktonzilla/notebooks/planktonzilla_taxonomy_final.csv"

datasets_names = {
    "3e2d20a947449a1e51bbdcc8fe371773b8c35e4e8479019551539bf5f7823eb5/2014": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    "56f94ecd289ddb36f87494d978aef99790c5650f770d01f89ead558e5963771b/2013": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    "408bc6812e35a3d2e90e8fb57c85f539e0eaf4500dff5c1ce0e8a8074b528718/2012": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"f80c0b3d369767185088799304629f77b5030743a43d91b4b3264971b9c4cc61/2011": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"da13f05092bb13b9d367fbae02398afb8a3a80ea37814789cc9512f590cd5476/2010": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"0e296dd67c83e0cb68e0c2cf2b201582af4e3583100e919e8264d76e7a8a7467/2009": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"11ddb677b01b8b8740bae905030809d38d2970694e53266793d459b46cd5e10f/2008": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"d98acf51510e492812af496f03d5cf7eca360240e3c31ffb143095481a1435f4/2007": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    #"60311988dbc3b51977e35672ac7c7162ac52b620eba2e241fc344ae23d011255/2006": ["whoi", WHOIRedefiner(csv_taxonomies_path = taxo_csv_path)],
    }

def main():
    for imgfolder_name, (dataset_name, r) in datasets_names.items():
        print(f"DATASET ====== {dataset_name}")

        t0 = time.perf_counter()

        dataset = load_dataset("imagefolder", data_dir=f"/lustre/fsn1/projects/rech/tec/uod68bo/data/whoiplanktondatasetimporter_raw_download_old/extracted/{imgfolder_name}", num_proc=num_proc)
        dataset = r.redefine(hf_dataset=dataset,dataset_name=dataset_name, num_proc=num_proc)

        year = imgfolder_name.split("/")[-1]
        dataset.save_to_disk(f"/lustre/fsn1/projects/rech/tec/uod68bo/data/whoi_{year}_hf")
        elapsed_min = (time.perf_counter() - t0) / 60
        print(f"Done — tiempo: {elapsed_min:.2f} minutos\n")

if __name__ == "__main__":
    main()
