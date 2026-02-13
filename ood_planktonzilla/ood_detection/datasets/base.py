"""Base dataset classes and transformations."""

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OODLabelTransform:
    """
    Transform for marking OOD (Out-of-Distribution) samples with a special label.
    
    Args:
        value (int): The label value to assign to OOD samples. Default is -1.
    """
    
    def __init__(self, value=-1):
        self.value = value
    
    def __call__(self, label):
        """Convert any label to the OOD label value."""
        return torch.tensor(self.value, dtype=torch.long)


class HFImageDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace image datasets.
    
    This class wraps HuggingFace datasets and applies transformations
    to both images and labels for use with PyTorch DataLoader.
    
    Args:
        hf_dataset: A HuggingFace dataset with 'image' and 'label' fields.
        img_transform: Transform to apply to images. If None, uses default
                      resize to 224x224 and conversion to tensor.
        label_transform: Optional transform to apply to labels (e.g., for OOD).
    """
    
    def __init__(self, hf_dataset, img_transform=None, label_transform=None):
        self.data = hf_dataset
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve.
            
        Returns:
            tuple: (image_tensor, label_tensor)
        """
        item = self.data[idx]
        image = item['image'].convert("RGB")  # Ensure it's in RGB format
        image = self.img_transform(image)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        if self.label_transform is not None:
            label = self.label_transform(label)
        
        return image, label
