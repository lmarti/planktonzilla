# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:50:45 2026

@author: equil

MODIFICATION FROM PLANKTONZILLA_FULL DATASET DIRECTLY
"""
# ============= IMPORTATIONS ============= #

import os # accès au système de fichiers (dossiers, chemins, etc.)
import sys # accès à l’environnement Python (arguments, chemins Python)
import shutil # copier / déplacer / supprimer des dossiers
import subprocess # exécuter des commandes système

from pathlib import Path # gestion moderne des chemins (remplace souvent os.path)
from shutil import rmtree # fonction qui supprime un dossier entier récursivement
import numpy as np
import math
import requests # Permet d'interroger des API scientifiques (EcoTaxa, WHOI)
import time
import json
import orjson

# Configuration Hydra
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import pyrootutils # permet de trouver automatiquement la racine du projet
import polars as pl # dataframe rapide (plus rapide que pandas)
from tqdm import tqdm # barre de progression
import concurrent.futures # exécution parallèle (threads)

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
) # librairie Hugging Face Datasets sert à manipuler datasets ML, gérer images

from planktonzilla.utils.logger import get_pylogger # crée un logger (INFO ERROR WARRNING)
from planktonzilla.dataset_import.dataset_importer import (
    DatasetImporter, # télécharge et prépare les datasets
    is_dir_empty, # vérifie si dossier vide
    is_valid_image_file, # vérifie si une image est corrompue"
)

from multiprocessing import cpu_count



num_proc = min(cpu_count(), 8)

root = pyrootutils.setup_root(
    search_from=".",  
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
) # pour trouver la racine du repo, ajouter les chemins python, charger .env

logger = get_pylogger(__name__) # crée un loger __name__ qui correspond au modèle actuel


# ============= GENERATING HF DATASETS WITH METADATA ============= #


def cast_metadata_json(ds): # "metadata" : dict → string JSON => "metadata": '{"lat": 10, "lon": 20}'
    # 1. Convert dict → JSON string
    def to_json(example): 
        return {"metadata": json.dumps(example["metadata"])} 

    ds = ds.map(to_json, desc="Serializing metadata")

    # 2. Cast feature
    features = ds.features.copy()
    features["metadata"] = Value("string")
    return ds.cast(features)


class ProcessDataset:
    def __init__(self, csv_taxonomies_path):

        self.csv_tax = pl.read_csv(csv_taxonomies_path, separator=";").fill_null("")

        self.taxonomy_cols = ["image", "dataset", "original_label", "original_path", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
        self.extra_cols = ["proposed_label", "plankton", "root_class"]
        self.meta_cols = ["Latitude", "Humidity", "BinID", "Depth_max", "Depth", "Temperature", "ObjID", "Depth_min", "Longitude", "Timestamp"]
        self.all_cols = self.taxonomy_cols + self.extra_cols + ["metadata"] + self.meta_cols

        keys = zip(self.csv_tax["Dataset"], self.csv_tax["Raw_Labels"])
        values =  self.csv_tax.select(self.extra_cols).to_dicts()
        self.lookup = dict(zip(keys, values)) # ignore taxonomy columns in lookup

    def retrieve_ecotaxa_metadata(self, obj_id, session=None):
        if obj_id is None:
            return {}
        api_url = f"https://ecotaxa.obs-vlfr.fr/api/object/{obj_id}"

        info = {
            "Depth_max": None,
            "Depth_min": None,
            "Latitude": None,
            "Longitude": None,
            "ObjID": str(obj_id),
            "Timestamp": "",
        } # initialise metadata

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
                ("objdate", "Timestamp"), # check names of the classes
                ("objtime", "Timestamp"),]:

                val = data.get(src)
                if dst not in ["objdate", "objtime"]: # avoid taking the float of a string object
                    info[dst] = np.float32(val) if val is not None else None
                else :
                    info[dst] += " " + val if val is not None else None

        except (requests.RequestException, ValueError, TypeError):
            pass

        return info
    
    def retrieve_whoi_metadata(self, bin_id, session=None):
        api_url = f"https://ifcb-data.whoi.edu/api/bin/{bin_id}"
        hdr_url = f"https://ifcb-data.whoi.edu/mvco/{bin_id}.hdr"

        requester = session or requests

        info = {
            "Latitude": None,
            "Longitude": None,
            "Depth": None,
            "Temperature": None,
            "BinID": str(bin_id),
            "Timestamp": None,
        }

        try:
            # ---------- JSON metadata ----------
            r = requester.get(api_url, timeout=10)
            if r.ok:
                data = r.json()
                info["Latitude"] = data.get("lat") # check names of the classes
                info["Longitude"] = data.get("lng")
                info["Depth"] = data.get("depth")
                info["Temperature"] = data.get("temperature")
                info["Timestamp"] = data.get("timestamp_iso")

            # ---------- HDR metadata ----------
            r = requester.get(hdr_url, timeout=10)
            if r.ok:
                lines = r.text.splitlines()

                for idx, line in enumerate(lines):
                    if "Temp" in line and idx + 1 < len(lines):
                        headers = line.replace('"', '').split()
                        values = lines[idx + 1].replace('"', '').split(",")

                        if len(values) < len(headers):
                            values = lines[idx + 1].split()

                        mapping = dict(zip(headers, values))
                        info["Temperature"] = mapping.get("Temp")
                        break

            # ---------- Fast float cast ----------
            for k in ("Latitude", "Longitude", "Depth", "Temperature"):
                v = info[k]
                if v is None or v == "" or (isinstance(v, np.float32) and math.isnan(v)):
                    info[k] = None
                else:
                    info[k] = np.float32(v)

        except Exception as e:
            print(f"WHOI metadata error: {e}")

        return info

    def _add_metadata(self, ds):
        ecotaxa_indices = []
        ecotaxa_ids = []
        whoi_indices = []
        whoi_ids = []
        for i, meta_str in enumerate(ds["metadata"]):
            md = orjson.loads(meta_str)
            obj_id = md.get("ObjID")
            bin_id = md.get("BinID")

            # Ecotaxa case
            if obj_id not in (None, ""):
                ecotaxa_indices.append(i)
                ecotaxa_ids.append(obj_id)
            # Whoi case
            elif bin_id not in (None, ""):
                whoi_indices.append(i)
                whoi_ids.append(bin_id)

        from functools import partial
        import concurrent.futures
        
        # Ecotaxa API
        ecotaxa_lookup = {}
        with requests.Session() as session:
            func = partial(self.retrieve_ecotaxa_metadata, session=session)
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor: # or maybe 16
                results = list(tqdm(executor.map(func, ecotaxa_ids), total=len(ecotaxa_ids)))
        for obj_id, md in zip(ecotaxa_ids, results):
            ecotaxa_lookup[obj_id] = md

        # WHOI API
        whoi_lookup = {}
        with requests.Session() as session:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self.retrieve_whoi_metadata, bin_id, session): bin_id
                    for bin_id in whoi_ids
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="WHOI"
                ):
                    bin_id = futures[future]
                    try:
                        whoi_lookup[bin_id] = future.result()
                    except Exception:
                        whoi_lookup[bin_id] = {}



        def normalize_metadata(md: dict | None) -> dict:
            if not md:
                return {}
            return {str(k): str(v) for k, v in md.items() if v is not None}
            
        existing_metadata = ds["metadata"]

        metadata = [
            dict(m) if isinstance(m, dict) else {}
            for m in existing_metadata
        ]

        # Merge EcoTaxa
        for idx, obj_id in zip(ecotaxa_indices, ecotaxa_ids):
            md = ecotaxa_lookup.get(obj_id)
            if md:
                metadata[idx].update(normalize_metadata(md))

        # Merge WHOI
        for idx, bin_id in zip(whoi_indices, whoi_ids):
            md = whoi_lookup.get(bin_id)
            if md:
                metadata[idx].update(normalize_metadata(md))

        ds = ds.remove_columns("metadata")
        ds = ds.add_column("metadata", metadata)

        return cast_metadata_json(ds)
    
    def retrieve_COXid_metadata(self, processed_ds):
        raise NotImplementedError()
    


    def redefine_dataset(self, ds, num_proc): # transforms dataset
        
        # critical optimization
        ds = ds.cast_column("image", Image(decode=False)) # doesnt decode images

        #add more metadata : Timestamp, depth, latitude, longitude,..
        ds = self._add_metadata(ds)
        print("added metadata after requests API : ", ds["metadata"][0], end="\n")
        print()

        def process_row(example):
            
            # update root_class, proposed label and plankton columns
            dataset_name = example["dataset"]
            label_str = example["original_label"]
            data = self.lookup.get((dataset_name, label_str)) # data = {'proposed_label': 'fiber', 'plankton': False, 'root_class': 'detritus'}
            example["root_class"] = data.get("root_class") if data != None else None 
            example["plankton"] = data.get("plankton") if data != None else None 
            example["proposed_label"] = data.get("proposed_label") if data != None else None

            # if data == None : # means its in planktonzilla but not in the csv ??
                # print(dataset_name, label_str, data, type(data))
            
            # extract metadata into columns
            md = orjson.loads(example["metadata"])
            for col in self.meta_cols : # ["Latitude", "Humidity", "BinID", "Depth_max", "Depth", "Temperature", "ObjID", "Depth_min", "Longitude", "Timestamp"]
                val = md.get(col)
                if col not in ["Depth", "Timestamp"] :
                    if (col == "ObjID" or col == "BinID") : # have a single column for the ObjID and BinID
                        val = md.get("ObjID") if md.get("ObjID") is not None else md.get("BinID")
                        example["ObjID"]= val if val not in (None, "") else None
                    else:    
                        example[col]= np.float32(val) if val not in (None, "") else None # transform every numeric value into float32
                
                elif col=="Depth": # copy the 'Depth' value into the Depth min and max columns
                    example["Depth_max"]= np.float32(val) if val not in (None, "") else None
                    example["Depth_min"]= np.float32(val) if val not in (None, "") else None

                else : # if col=="Timestamp"
                    example[col]= val if val not in (None, "") else None # leave Timestamp as a string

            return example
            '''
            "qualifier":"",
            "cox_gene_id ": "",
            "wikipedia_id": "",
            '''
        processed_ds = ds.map(process_row, desc="Columns mapping",num_proc=num_proc, load_from_cache_file=False,) # keep_in_memory=True for small datasets
        processed_ds = processed_ds.remove_columns(["living", "metadata"]) # ["Latitude",  "Longitude", "Humidity", "Depth_min", "Depth_max", "Temperature", "ObjID", "Timestamp"]

        processed_ds = processed_ds.cast_column("image", Image(decode=True))
        return processed_ds
    

# ============= GENERATING HF DATASETS WITH METADATA ============= #


def main():

    # DATA_ROOT = Path("planktonzilla/notebooks").resolve()
    base_path = Path.home() # / "group_storage_sophia/saguilera/Labels"
    print(base_path)
    #file1 = base_path / "eKOI_taxonomy_labels.parquet"
    #file2 = base_path / "MetaCOXI_taxonomy_labels.parquet"
    #df = pl.read_parquet(file1)
    #print(df.head())

    dataset = load_dataset(
        "project-oceania/planktonzilla_full",
        split="train") # streaming=True # creates IterableDataset != normal dataset so num_proc is not supported
    dataset = dataset.shuffle(seed=42)
    ds = dataset.select(range(50_333))

    # Print a sample of the intitial dataset
    '''
    for idx, example in enumerate(ds) :
        if idx%1000== 0:
            print(example, end="\n")
    print()
    '''

    # Process dataset
    csv_taxonomies_path = "planktonzilla_taxonomy_v3.csv"
    processor = ProcessDataset(csv_taxonomies_path)
    modified_ds= processor.redefine_dataset(ds, num_proc=num_proc)


    # Print a sample of the final dataset
    print("Rearranged dataset")
    for idx, example in enumerate(modified_ds) :
        #if example['proposed_label'] == None:
            #print(example)
            #print()
        if example['Timestamp'] != None :
            #print("Final example : ", example, end="\n")
            if example['dataset']!= 'whoi': # all the whoi examples have timestamps but no other datasets
                print("not just whoi datasets have timestamp", example['dataset'])
        if example['dataset']== 'whoi' and example['Timestamp'] == None:
            print("some whoi datasets dont have the timestamp")
    # modified_ds.save_to_disk(DATA_ROOT / "planktonzilla_full_modified")

if __name__ == "__main__":
    main()