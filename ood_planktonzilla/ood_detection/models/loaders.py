"""Model loading utilities for different backbone architectures."""

import copy
import torch.nn as nn
import timm
from transformers import AutoModelForImageClassification


def load_model(backbone: str, n_classes: int):
    """
    Load a pretrained model for the specified backbone architecture.
    
    Args:
        backbone (str): Name of the backbone architecture. Supported values:
                       'beit-base', 'resnet18', 'vit-base', 'timm-eva02-large-m38m'
        n_classes (int): Number of output classes for classification.
        
    Returns:
        nn.Module: The loaded pretrained model.
        
    Raises:
        ValueError: If the backbone is not supported.
    """
    if backbone == "beit-base":
        model = timm.create_model(
            'beit_base_patch16_384.in22k_ft_in22k_in1k',
            pretrained=True,
            num_classes=n_classes
        )
    elif backbone == "resnet18":
        model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18')
        model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=512, out_features=n_classes)
        )
    elif backbone == "vit-base":
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=n_classes
        )
    elif backbone == "timm-eva02-large-m38m":
        model = AutoModelForImageClassification.from_pretrained(
            "project-oceania/timm-eva02-large-m38m-ft-planktonzilla"
        )
        model = model.timm_model
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model


def get_features_extractor(backbone: str, model):
    """
    Extract the feature extraction layers from a model.
    
    This creates a deep copy of the model and replaces the classification
    head with an Identity layer, effectively turning it into a feature extractor.
    
    Args:
        backbone (str): Name of the backbone architecture.
        model: The original model (should be a TorchModel wrapper).
        
    Returns:
        nn.Module: Feature extractor model.
        
    Raises:
        ValueError: If the backbone is not supported.
    """
    # Important: Deepcopy ensures we don't mess up the original model
    if backbone == "beit-base":
        features = copy.deepcopy(model.model)
        features.head = nn.Identity()
    elif backbone == "resnet18":
        features = copy.deepcopy(model)
        features.model.classifier = nn.Flatten(start_dim=1, end_dim=-1)
    elif backbone == "vit-base":
        features = copy.deepcopy(model)
        features.model.classifier = nn.Identity()
    elif backbone == "timm-eva02-large-m38m":
        features = copy.deepcopy(model)
        features.model.head = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return features


def get_head_layer(backbone: str, model):
    """
    Retrieve the classification head layer from a model.
    
    This retrieves the actual layer object from the UNWRAPPED model,
    which is needed for some OOD detection methods like ViM and ReAct.
    
    Args:
        backbone (str): Name of the backbone architecture.
        model: The model (should be a TorchModel wrapper).
        
    Returns:
        nn.Module: The classification head layer.
        
    Raises:
        ValueError: If the backbone is not supported.
    """
    if backbone == "beit-base":
        return model.model.head
    elif backbone == "resnet18":
        return model.model.classifier[1]
    elif backbone == "vit-base":
        return model.model.classifier
    elif backbone == "timm-eva02-large-m38m":
        return model.model.head
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
