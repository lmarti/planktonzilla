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
import time

log = get_pylogger(__name__)


try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

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
    shared_path_list = [cfg.paths.output_dir if dist.get_rank() == 0 else None]

    # 2. Broadcast the list from Rank 0 to everyone else
    # This physically sends the string over the network to Ranks 1, 2, and 3
    dist.broadcast_object_list(shared_path_list, src=0)

    # 3. Extract the perfectly synced path!
    synced_output_dir = shared_path_list[0]
    
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
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

        
    ds_id = concatenate_datasets([dataset_id.training_dataset, dataset_id.validation_dataset])
    ds_ood = concatenate_datasets([
                                dataset_ood.training_dataset,
                                dataset_ood.validation_dataset,
                                dataset_ood.test_dataset
                            ])
    


    ds_ood = ds_ood.map(set_unknown, batched=True, num_proc=16)
    ds_ood = ds_ood.cast(dataset_id.training_dataset.features)

    ds_test = concatenate_datasets([ds_ood, dataset_id.test_dataset])

    print(f"Largo dataset test full:{len(ds_test)}")
    def apply_transforms(examples):
        # Sobrescribimos la columna "image" con la versión transformada
        # .convert("RGB") asegura que no haya errores con imágenes en escala de grises o RGBA
        examples["image"] = [dataset_id.transform(img.convert("RGB")) for img in examples["image"]]
        return examples


    ds_id.set_transform(apply_transforms)
    ds_test.set_transform(apply_transforms)

    from torch.utils.data.distributed import DistributedSampler
    test_sampler = DistributedSampler(ds_test, shuffle=False)
    train_sampler = DistributedSampler(ds_id,shuffle=False)
    dataloader_train = DataLoader(
        ds_id,
        batch_size=16,
        shuffle=False,
        sampler = train_sampler,
        collate_fn=transform_to_tuple,
    )

    dataloader_test = DataLoader(
        ds_test,
        batch_size=16,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=transform_to_tuple,
    )
    log.info("Instantiating detectors...")

    def load_and_cast_param(detector, param_name, loaded_value, fallback_device):
        """
        Inspects the internal attribute's type, casts the loaded value to match, 
        and applies .to(device) only if the object supports it.
        """
        # Get the current internal attribute (might be None on first initialization)
        current_attr = getattr(detector, param_name, None)
        
        # 1. Inspect the type and cast (only if it already exists and isn't None)
        if current_attr is not None:
            if isinstance(current_attr, torch.Tensor):
                # Cast to Tensor and ensure the data types match exactly
                loaded_value = torch.as_tensor(loaded_value, dtype=current_attr.dtype)
            elif isinstance(current_attr, np.ndarray):
                # Cast to NumPy array
                loaded_value = np.array(loaded_value)
            else:
                # Cast to standard Python types (float, int, etc.)
                attr_type = type(current_attr)
                try:
                    loaded_value = attr_type(loaded_value)
                except (ValueError, TypeError):
                    pass # If casting fails, keep the original loaded value
                    
        # 2. Check if the type can receive .to()
        if hasattr(loaded_value, "to"):
            # Use the attribute's current device if it has one, otherwise use the loop's device
            target_device = getattr(current_attr, "device", fallback_device)
            loaded_value = loaded_value.to(target_device)
            
        # 3. Assign the safely casted and moved value back to the detector
        setattr(detector, param_name, loaded_value)
    results = []
    t0 = time.time()
    ##################### Extracción de features paralelizada ########################
    log.info(f"Rank {dist.get_rank()}: Extracting local features...")

    local_z = []
    local_y = []

    model.eval()
    with torch.no_grad():
        # Ensure dataloader_train is using a DistributedSampler!
        for x, y in tqdm(dataloader_train, desc=f"Extracting (Rank {dist.get_rank()})"):
            x = x.to(device)
            
            # Ensure model(x) returns the raw feature embeddings, not logits
            features = model(x) 
            
            local_z.append(features)
            local_y.append(y.to(device))
    
    local_z = torch.cat(local_z, dim=0)
    local_y = torch.cat(local_y, dim=0)

    log.info(f"Rank {dist.get_rank()}: Gathering features across all nodes...")
    
    world_size = dist.get_world_size()
    gathered_z = [torch.zeros_like(local_z) for _ in range(world_size)]
    gathered_y = [torch.zeros_like(local_y) for _ in range(world_size)]

    # Sync across all GPUs and Nodes
    dist.all_gather(gathered_z, local_z)
    dist.all_gather(gathered_y, local_y)

    # Combine and IMMEDIATELY move to CPU memory. 
    # 2 Million features will likely crash GPU VRAM if kept there.
    global_z = torch.cat(gathered_z, dim=0).cpu().double()
    global_y = torch.cat(gathered_y, dim=0).cpu()
    

    log.info(f"Global features gathered! Shape: {global_z.shape}")

    for detector_name, detector_cfg in cfg.ood_detectors.items():
        log.info(f"Rank {dist.get_rank()}: Fitting {detector_name}...")
        
        # Instantiate the standard pytorch-ood detector from Hydra
        extra_args = {"model": model}
        if "vim" == detector_name:
            extra_args["w"] = last_linear.weight
            extra_args["b"] = last_linear.bias
        detector = hydra.utils.instantiate(detector_cfg, **extra_args)
        
        
        # 🛑 THE MAGIC: Pass the pre-computed features directly!
        # device="cpu" ensures the math operations don't blow up your GPU memory
        detector.fit_features(global_z, global_y)
        
        # Save the mathematically fitted parameters
        if dist.get_rank() == 0:
            log.info(f"Rank 0: Saving {detector_name} parameters...")
            if detector_name == "mahalanobis":
                torch.save({"mu": detector.mu, "precision": detector.precision}, f"{synced_output_dir}/mahalanobis_parameters.pt")
            elif detector_name == "vim":
                torch.save({"alpha": detector.alpha, "principal_subspace": detector.principal_subspace}, f"{synced_output_dir}/vim_parameters.pt")

    # Force everyone to wait until Rank 0 finishes saving all files
    dist.barrier()
    log.info(f"Rank {dist.get_rank()}: All fitting complete. Ready for evaluation!")


    for detector_name, detector_cfg in cfg.ood_detectors.items():
        # EVERY GPU instantiates a fresh, empty detector
        extra_args = {"model": model}
        if "vim" == detector_name:
            extra_args["w"] = last_linear.weight
            extra_args["b"] = last_linear.bias
        detector = hydra.utils.instantiate(detector_cfg, **extra_args)
        
        # EVERY GPU loads the parameters from the hard drive
        log.info(f"Rank {dist.get_rank()} loading parameters for {detector_name}...")
        
        if detector_name == "mahalanobis":
            d = torch.load(f"{synced_output_dir}/mahalanobis_parameters.pt", map_location="cpu", weights_only=False)
            if "mu" in d and d["mu"] is not None:
                load_and_cast_param(detector, "mu", d["mu"], device)
            if "precision" in d and d["precision"] is not None:
                load_and_cast_param(detector, "precision", d["precision"], device)
                
        elif detector_name == "vim":
            d = torch.load(f"{synced_output_dir}/vim_parameters.pt", map_location="cpu", weights_only=False)
            if "alpha" in d and d["alpha"] is not None:
                load_and_cast_param(detector, "alpha", d["alpha"], device)
            if "principal_subspace" in d and d["principal_subspace"] is not None:
                load_and_cast_param(detector, "principal_subspace", d["principal_subspace"], device)

        dist.barrier()
        log.info(f"Evaluating detector on Rank {dist.get_rank()}...")
        
        # 1. Grab the indices assigned to this GPU and push them to the device
        gpu_indices = torch.tensor(list(test_sampler), dtype=torch.long, device=device)
        local_scores = []
        local_labels = []

        with torch.no_grad():
            for x, y in tqdm(dataloader_test, desc=f"Evaluating {detector_name}"):
                x = x.to(device)
                y = y.to(device)
                local_scores.append(detector(x))
                local_labels.append(y)

        # Concatenate local results
        local_scores = torch.cat(local_scores, dim=0).to(device)
        local_labels = torch.cat(local_labels, dim=0).to(device)
        gpu_indices = gpu_indices.to(device)

        # ==========================================
        # 🚀 THE IN-MEMORY "GLOBAL LIST" GATHERING
        # ==========================================
        world_size = dist.get_world_size()
        
        # Create empty lists of tensors to hold the incoming data from all GPUs.
        # These placeholders MUST be exactly the same shape and dtype as the local tensors.
        gathered_scores = [torch.zeros_like(local_scores) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
        gathered_idxs = [torch.zeros_like(gpu_indices) for _ in range(world_size)]
        
        # Fire the collective operation! 
        # This magically fills the empty lists with data from Rank 0, Rank 1, etc.
        dist.all_gather(gathered_scores, local_scores)
        dist.all_gather(gathered_labels, local_labels)
        dist.all_gather(gathered_idxs, gpu_indices)

        # ==========================================
        # 📊 RANK 0 PROCESSES THE GATHERED DATA
        # ==========================================
        if dist.get_rank() == 0:
            log.info("Stitching in-memory results together...")
            
            # Concatenate the gathered lists into massive, unordered tensors
            # We move them to CPU here so metric calculation doesn't hog GPU memory
            global_scores = torch.cat(gathered_scores, dim=0).cpu()
            global_labels = torch.cat(gathered_labels, dim=0).cpu()
            global_idxs = torch.cat(gathered_idxs, dim=0).cpu()
            
            # 🛑 Apply the exact same Magic Fix to reconstruct dataset order
            true_length = len(ds_test)
            
            perfect_scores = torch.zeros((true_length, *global_scores.shape[1:]), dtype=global_scores.dtype)
            perfect_labels = torch.zeros((true_length, *global_labels.shape[1:]), dtype=global_labels.dtype)
            
            perfect_scores[global_idxs] = global_scores
            perfect_labels[global_idxs] = global_labels
            
            # Calculate metrics
            metrics = OODMetrics()
            metrics.update(perfect_scores, perfect_labels)

            r = {"Detector": detector_name, "Dataset": "test"}
            r.update(metrics.compute())
            results.append(r)
            
            # Only save the FINAL combined buffer to disk
            metrics.buffer.save(f"{synced_output_dir}/{detector_name}_FINAL.pt")
        
        # Keep GPUs synchronized before moving to the next task
        dist.barrier()

    # calculate mean scores over all datasets, use percent
    if dist.get_rank() == 0:
        df = pd.DataFrame(results)
        mean_scores = df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
        print(df)
        print(mean_scores)
        print(f"Elapsed time: {time.time() - t0:.2f} seconds")

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
