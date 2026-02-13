"""Image preprocessing configurations for different model backbones."""

import torchvision.transforms as F


def get_preprocessing(backbone: str):
    """
    Get the appropriate preprocessing transforms for a given model backbone.
    
    Args:
        backbone (str): Name of the model backbone.
        
    Returns:
        torchvision.transforms.Compose: Composed preprocessing transforms.
        
    Raises:
        ValueError: If the backbone is not supported.
    """
    preprocessing = {
        "beit-base": F.Compose([
            F.Resize((384, 384)),
            F.ToTensor()
        ]),
        "resnet18": F.Compose([
            F.Resize((224, 224)),
            F.ToTensor()
        ]),
        "vit-base": F.Compose([
            F.Resize((224, 224)),
            F.ToTensor()
        ]),
        "timm-eva02-large-m38m": F.Compose([
            F.ToTensor(),
            F.Resize((448, 448), antialias=True),
            F.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
        ]),
    }
    
    if backbone not in preprocessing:
        raise ValueError(f"Unsupported backbone: {backbone}. "
                        f"Supported backbones: {list(preprocessing.keys())}")
    
    return preprocessing[backbone]
