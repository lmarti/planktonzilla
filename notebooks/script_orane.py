##### Scripts for the .CSV handling

from pathlib import Path
import numpy as np
import polars as pl
from planktonzilla.dataset_import.dataset_importer import (
    DatasetImporter, # télécharge et prépare les datasets
    is_dir_empty, # vérifie si dossier vide
    is_valid_image_file, # vérifie si une image est corrompue"
)

def main():
        ##### Script to create a v3 .csv from Alan's .csv (v2) + mine (v1) - by adding to Alan's .csv the information I modified : ["proposed_label", "plankton", "root_class"] #####
        
        csv_tax_old = pl.read_csv("planktonzilla_taxonomy_v1.csv", separator=";").fill_null("")
        csv_tax_new = pl.read_csv("planktonzilla_taxonomy_v2.csv", separator=";").fill_null("")

        cols = ["Dataset", "Raw_Labels", "proposed_label", "plankton", "root_class"]
        csv_tax_new = csv_tax_new.join(csv_tax_old.select(cols),on=["Dataset", "Raw_Labels"],how="left")
        # csv_tax_new.write_csv("planktonzilla_taxonomy_v-.csv", separator=";")

        ##### Script to check what has been modified by Alan for the 'plankton' and 'living' columns #####
        # csv_tax_old = pl.read_csv("planktonzilla_taxonomy_v1.csv", separator=";").fill_null("")
        # csv_tax_new = pl.read_csv("planktonzilla_taxonomy_v2.csv", separator=";").fill_null("")

        # extra_cols = ["proposed_label", "plankton", "living", "root_class"]

        # keys_old = zip(csv_tax_old["Dataset"], csv_tax_old["Raw_Labels"])
        # values_old =  csv_tax_old.select(extra_cols).to_dicts()
        # lookup_old = dict(zip(keys_old, values_old)) # ie : ('zooscan', 'larvae_Annelida'): {'proposed_label': 'annelida', 'plankton': True, 'root_class': 'living'}

        
        # keys_new = list(zip(csv_tax_new["Dataset"], csv_tax_new["Raw_Labels"]))
        # for idx, row in enumerate(keys_new) : # going thru the new .csv
        #         if ((row[0] != 'global_uvp5' and row[0] != 'syke_ifcb_2022') and row[0] != 'planktonset1.0'): # avoid comparing the columns Alan added because I didn't modify those
        #             if csv_tax_new["plankton"][idx] != lookup_old[row]["plankton"]:
        #                 print("Id:", row, "v2 Alan", csv_tax_new["plankton"][idx], "v1 Orane", lookup_old[row]["plankton"] )
        #             if csv_tax_new["living"][idx] != lookup_old[row]["living"]:
        #                 print("Id:", row, "v2 Alan", csv_tax_new["living"][idx], "v1 Orane", lookup_old[row]["living"] )

        ##### Script to update Alan's CSV by fetching the information I modified : ["proposed_label", "plankton", "root_class"] #####
        # # Method doesn't work with polars since it doesn't accept affections
        # csv_tax_new = csv_tax_new.with_columns(pl.lit("").alias("root_class"))
        # extra_cols = ["proposed_label", "plankton", "root_class"]

        # keys_old = zip(csv_tax_old["Dataset"], csv_tax_old["Raw_Labels"])
        # values_old =  csv_tax_old.select(extra_cols).to_dicts()
        # lookup_old = dict(zip(keys_old, values_old)) # ie : ('zooscan', 'larvae_Annelida'): {'proposed_label': 'annelida', 'plankton': True, 'root_class': 'living'}

        # keys_new = list(zip(csv_tax_new["Dataset"], csv_tax_new["Raw_Labels"]))
        # # print(keys_new[0], list(lookup_old.keys())[0])
        
        # for idx, row in enumerate(keys_new) : # going thru the new .csv
        #     if row in lookup_old : # examples in common between both .csv
        #         csv_tax_new["proposed_label"][idx] = lookup_old[row]["proposed_label"]
        #         csv_tax_new["plankton"][idx] = lookup_old[row]["plankton"]
        #         csv_tax_new["root_class"][idx] = lookup_old[row]["root_class"]
            
        # print(sum(1 for row in lookup_old if row in keys_new)) # = 1060
        # print(sum(1 for row in lookup_old )) # = 1060, means that every row in lookup_old is in keys_new

        # csv_tax_new.write_csv("planktonzilla_taxonomy_v-.csv", separator=";")
        


if __name__ == "__main__":
    main()