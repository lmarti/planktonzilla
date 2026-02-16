import sys
import os

root_path = os.path.abspath("..")
sys.path.append(root_path)

from planktonzilla.dataset_import.dataset_importer import *


from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset, Value
from pathlib import Path


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() == "true"
    return False


def normalize_example(example):
    example["plankton"] = to_bool(example["plankton"])
    example["living"]   = to_bool(example["living"])
    return example



BASE_DIR = Path("/lustre/fsn1/projects/rech/tec/uod68bo/data")

# Detecta automáticamente todos los datasets *_hf
hf_paths = sorted(p for p in BASE_DIR.glob("*_hf") if p.is_dir())

datasets = []

for p in hf_paths:
    d = load_from_disk(p)

    d = d.map(
        normalize_example,
        num_proc=16,
        desc="Normalizing bool columns",
    )

    datasets.append(d)

datasets = concatenate_datasets(datasets)

datasets = datasets.map(
    lambda x: {
        "proposed_label": x["proposed_label"] if x["proposed_label"] is not None else ""
    },
    num_proc=16
)


def report_dataset_content_custom(huggingface_dataset: Dataset | DatasetDict) -> str:
    def report_split(dataset: Dataset, split_name: str) -> str:
        class_names, class_counts = np.unique(dataset["proposed_label"], return_counts=True)

        content = []
        for class_idx, class_name in enumerate(class_names):
            content += [f"{class_idx}: {class_name}"]

        plt.simple_bar(content, class_counts.astype(int), title=f"Label histogram for {split_name} split ", width=83)
        plt.show()

        return strip_ansi_codes(plt.build())

    if isinstance(huggingface_dataset, DatasetDict):
        split_reports = []
        split_reports = [
            f"**Samples per class for split `{split}`**\n ```{report_split(huggingface_dataset[split], split)}```\n"
            for split in huggingface_dataset
        ]
        return "\n".join(split_reports)
    else:
        return report_split(huggingface_dataset) + "\n"


import planktonzilla.dataset_import.dataset_importer as importer_module

# Reemplazamos la función original por la personalizada en el contexto del módulo
importer_module.report_dataset_content = report_dataset_content_custom



@dataclass
class PlanktonzillaDatasetImporter(DatasetImporter):
    def import_dataset(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self._push_to_hub()


from huggingface_hub.utils import get_token
hf_token = get_token()


hf_dataset = DatasetDict({
    "train": datasets,
})


importer = PlanktonzillaDatasetImporter(
    data_dir="",
    hf_token =hf_token,
    hf_private=False,
    push_to_hub=True,
    hf_dataset_name="planktonzilla_full",
    hf_org_name="project-oceania",
    human_readable_name="Plankton Dataset",
    description = "A dataset composed of all publicly available, labeled plankton datasets",
    source_url="https://huggingface.co/datasets/project-oceania/planktonzilla_full",
    license="cc-by-4.0",
    citation_bibtex="""@misc{planktonzilla_full,
        author       = {Inria Chile},
        title        = {Planktonzilla dataset},
        month        = jan,
        year         = 2026,
        version      = {1.0.0},
        doi          = {TBD},
        url          = {https://huggingface.co/datasets/project-oceania/planktonzilla_full},
    }""",
)



def main():
    importer.import_dataset(hf_dataset)
    print("DONE")
if __name__ == "__main__":
    main()
