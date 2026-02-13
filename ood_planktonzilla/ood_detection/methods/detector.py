"""OOD detection method setup and application utilities."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_ood.detector import (
    EnergyBased, MaxLogit, KLMatching, ReAct, 
    Mahalanobis, ViM, MaxSoftmax
)
from typing import Union, Optional
from .custom_detectors import CustomEnergyBased, CustomMaxLogit, CustomReAct, CustomMaxSoftmax,CustomKLMatching, CustomMahalanobis, CustomViM


CALL_DICT = {
    "MSP": CustomMaxSoftmax,
    "KL-Matching": CustomKLMatching,
    "Energy": CustomEnergyBased,
    "MaxLogit": CustomMaxLogit,
    "React": CustomReAct,
    "Mahalanobis": CustomMahalanobis,
    "ViM": CustomViM,
}


def _create_method(model, dict_method: dict):
    """
    Instantiate an OOD detection method without fitting or loading parameters.

    Args:
        model: The model to use for OOD detection.
        dict_method (dict): Dictionary with 'name' and 'cfg' keys.

    Returns:
        OOD detector instance, or None if method not found.
    """
    assert dict_method.get("name") and isinstance(dict_method.get("cfg"), dict)
    cls = CALL_DICT.get(dict_method["name"])
    if cls is None:
        return None
    return cls(model, **dict_method["cfg"])


def fit_detection_method(model, dict_method: dict, fit_dataloader, device,
                         save_fitted_params: Optional[str] = None):
    """
    Instantiate an OOD detection method, fit it, and optionally save parameters.

    Args:
        model: The model to use for OOD detection.
        dict_method (dict): Dictionary with 'name' and 'cfg' keys.
        fit_dataloader: DataLoader for fitting the OOD detector.
        device: Device to use for computation.
        save_fitted_params: Optional directory path to save fitted parameters.

    Returns:
        Fitted OOD detector instance, or None if method not found.
    """
    method = _create_method(model, dict_method)
    if method is not None:
        method.fit(fit_dataloader, device=device)
        if save_fitted_params:
            method.save_fitted_parameters(save_fitted_params)
    return method


def load_detection_method(model, dict_method: dict, load_fitted_params: str):
    """
    Instantiate an OOD detection method and load pre-fitted parameters.

    Args:
        model: The model to use for OOD detection.
        dict_method (dict): Dictionary with 'name' and 'cfg' keys.
        load_fitted_params: Directory path to load fitted parameters from.

    Returns:
        OOD detector instance with loaded parameters, or None if method not found.
    """
    method = _create_method(model, dict_method)
    if method is not None:
        method.load_fitted_parameters(load_fitted_params)
    return method


def apply_ood_method(dataloader: DataLoader, method_obj, device: torch.device, method_name: str):
    """
    Apply an OOD detection method to a dataset and collect scores.
    
    This function runs inference on the entire dataset using the provided
    OOD detection method and returns the scores and labels.
    
    Args:
        dataloader (DataLoader): DataLoader containing the data to evaluate.
        method_obj: The OOD detector instance.
        device (torch.device): Device to use for computation.
        method_name (str): Name of the method (for progress bar display).
        
    Returns:
        tuple: (scores, labels)
            - scores: Tensor of OOD scores for all samples
            - labels: Tensor of ground truth labels for all samples
    """
    global_labels, global_scores = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Applying {method_name}"):
            x, y = batch
            inputs = x.to(device)
            scores = method_obj.predict(inputs)
            global_labels.append(y.cpu())
            global_scores.append(scores.cpu())
    
    return torch.cat(global_scores), torch.cat(global_labels)
