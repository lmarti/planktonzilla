"""
(c) Inria
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import os
from functools import partial
import torch.nn as nn
import hydra
import numpy as np
import torch
from evaluate import combine, load
from huggingface_hub import DatasetCard, login
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments, set_seed

from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from pytorch_ood.utils import OODMetrics

from tqdm import tqdm

from planktonzilla.dataset import DatasetWrapper
from planktonzilla.utils.hydra import (
    get_metric_value,
    task_wrapper,
)
from planktonzilla.utils.logger import get_pylogger
import pandas as pd
log = get_pylogger(__name__)

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass



def set_unknown(batch):
    batch["label"] = [-1] * len(batch["label"])
    return batch


def transform_to_tuple(batch):
    # Extraemos las imágenes y las etiquetas del batch de diccionarios
    images = torch.stack([x["image"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return images, labels


def validate_environment():
    """Check and log important external service environment variables.

    Warns when Hugging Face hub or tracking services are likely unavailable
    and logs presence of common environment variables such as `HF_TOKEN`,
    `WANDB_API_KEY` and `MLFLOW_TRACKING_URI`.
    """

    if "HF_HUB_OFFLINE" in os.environ and os.environ["HF_HUB_OFFLINE"] == "1":
        log.warning("⚠️ Environment variable HF_HUB_OFFLINE=1. Hugging Face hub will be offline.")
    else:
        if "HF_TOKEN" in os.environ:
            log.info("✅ HF_TOKEN environment variable is set.")
            try:
                login(new_session=False, write_permission=True)
                log.info("✅ Login to Hugging Face hub verified.")
            except ValueError as e:
                log.error(f"🛑 Login to Hugging Face hub failed: {e}.")
            except ImportError:  # If running in a notebook but ipywidgets is not installed.
                log.error("🛑 Running in a notebook but ipywidgets is not installed.")
        else:
            log.warning("⚠️ HF_TOKEN environment variable is not set. Access to private models and datasets will be limited.")

    if "WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] == "offline":
        log.warning("⚠️ Environment variable WANDB_MODE=offline. WandB will be offline. Remember to sync results later on.")
    elif "WANDB_API_KEY" in os.environ:
        log.info("✅ WANDB_API_KEY environment variable is set.")
    else:
        log.warning("⚠️ WANDB_API_KEY environment variable is not set. WandB logging will be disabled.")

    if "MLFLOW_TRACKING_URI" in os.environ:
        log.info("✅ MLFLOW_TRACKING_URI environment variable is set.")
        if "MLFLOW_TRACKING_USERNAME" in os.environ:
            log.info("✅ MLFLOW_TRACKING_USERNAME environment variable is set.")
        if "MLFLOW_TRACKING_PASSWORD" in os.environ:
            log.info("✅ MLFLOW_TRACKING_PASSWORD environment variable is set.")
    else:
        log.warning("⚠️ MLFLOW_TRACKING_URI environment variable is not set, if mlflow is enabled will log to local folder.")



@task_wrapper
def train_ood(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the ood detectors. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random

    validate_environment()

    if cfg.get("seed"):
        set_seed(cfg.seed, cfg.get("deterministic", False))

    # set proper matmul precision
    # hydra.utils.instantiate(cfg.torch_matmul_precision)

    log.info(f"Instantiating dataset wrapper for «{cfg.dataset.name}».")
    dataset_id: DatasetWrapper = hydra.utils.instantiate(cfg.dataset)
    dataset_ood: DatasetWrapper = hydra.utils.instantiate(cfg.dataset_ood)

    log.info("Instantiating data augmentation(s).")
    augmentation = hydra.utils.instantiate(cfg.augmentation)

    log.info(f"Preparing data splits for «{cfg.dataset.name}».")
    dataset_id.prepare_datasets(augmentation)
    dataset_ood.prepare_datasets(augmentation)

    dataset_card = DatasetCard.load(cfg.dataset.name)
    log.info(
        f"Dataset «{cfg.dataset.name}» {dataset_card.data.dataset_info.get('dataset_name', '')} <https://huggingface.co/datasets/{cfg.dataset.name}>."
    )

    log.info(f"Instantiating base model «{cfg.model._args_[0]}».")

    model: AutoModelForImageClassification = hydra.utils.instantiate(
        cfg.model,
        id2label=dataset_id.id2label,
        label2id=dataset_id.label2id,
        num_labels=len(dataset_id.label2id),
        _convert_="all",
    )

    # timm only have head or fc (?)

    model= model.timm_model

    try:
        last_linear = model.head
        model.head = nn.Identity()
    except:
        last_linear= model.fc
        model.fc = nn.Identity()

    ds_id = concatenate_datasets([dataset_id.training_dataset, dataset_id.validation_dataset])
    ds_ood = concatenate_datasets([
                                dataset_ood.training_dataset,
                                dataset_ood.validation_dataset,
                                dataset_ood.test_dataset
                            ])



    ds_ood = ds_ood.map(set_unknown, batched=True, num_proc=16)    
    ds_test = concatenate_datasets([ds_ood, dataset_id.test_dataset])

    def apply_transforms(examples):
        # Sobrescribimos la columna "image" con la versión transformada
        # .convert("RGB") asegura que no haya errores con imágenes en escala de grises o RGBA
        examples["image"] = [dataset_id.transform(img.convert("RGB")) for img in examples["image"]]
        return examples


    ds_id.set_transform(apply_transforms)
    ds_test.set_transform(apply_transforms)


    dataloader_train = DataLoader(ds_id, batch_size=128, num_workers=4,collate_fn=transform_to_tuple)
    dataloader_test = DataLoader(ds_test, batch_size=128, num_workers=4,collate_fn=transform_to_tuple)

    log.info("Instantiating detectors...")
    detectors = []

    results = []

    for detector_name, detector_cfg in cfg.ood_detectors.items():
        if "_target_" in detector_cfg:
            log.info(f"Instantiating {detector_name}...")
            
            # Preparamos los argumentos extra en un diccionario normal de Python
            extra_args = {"model": model}

            # Si es ViM, agregamos los pesos y bias a los argumentos extra
            # NO modificamos detector_cfg
            if detector_name == "vim": # Ojo: asegúrate que la key en el yaml sea exactamente 'vim'
                extra_args["w"] = last_linear.weight
                extra_args["b"] = last_linear.bias

            # Pasamos extra_args desempaquetados (**extra_args)
            # Hydra fusionará estos argumentos con los que ya vienen en el YAML
            detector = hydra.utils.instantiate(detector_cfg, **extra_args)

            log.info("Fitting detector...")

            detector.fit(dataloader_train)
            log.info("Evaluating detector...")

            with torch.no_grad():
                metrics = OODMetrics()
                for x, y in tqdm(dataloader_test):
                    metrics.update(detector(x), y)

            r = {"Detector": detector_name, "Dataset": "test"}
            r.update(metrics.compute())
            results.append(r)

            metrics.buffer.save(f"{cfg.paths.output_dir}/{detector_name}.pt")

    # calculate mean scores over all datasets, use percent
    df = pd.DataFrame(results)
    mean_scores = (
        df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
    )

    print(df)

    print("")
    print(mean_scores)

    return 0, 0


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train_ood.yaml")
def main(cfg: DictConfig) -> float | None:
    # train the model
    metric_dict, _ = train_ood(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()
