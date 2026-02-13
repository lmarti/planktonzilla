"""Advanced dataset loading utilities with config support."""

from typing import Optional, List
from datasets import load_dataset, load_from_disk
import datasets


def load_dataset_from_config(
    source: str,
    split: str = "train",
    classes: Optional[List[str]] = None,
    samples_per_class: Optional[int] = None,
    logger=None
):
    """
    Load a dataset from HuggingFace or local path with optional filtering.
    
    This function provides a flexible interface for loading datasets with
    optional class filtering and balanced sampling.
    
    Args:
        source (str): HuggingFace dataset identifier or local path.
        split (str): Dataset split to load. Default is 'train'.
        classes (list, optional): List of class names to keep. If None, keeps all.
        samples_per_class (int, optional): Number of samples per class. If None, keeps all.
        logger (ExperimentLogger, optional): Logger instance for timing and messages.
        
    Returns:
        tuple: (dataset, num_labels)
            - dataset: Loaded and filtered HuggingFace dataset
            - num_labels: Number of unique labels in the dataset
    """
    # Load the dataset
    if logger:
        logger.start_timer(f"loading_{source}")
    try:
        ds = load_dataset(source, split=split, num_proc=4)
    except:
        ds = load_from_disk(source)
    
    # If classes are specified, filter by those classes
    if classes is not None:
        hf_labels = ds.features['label']
        
        # Create name to int mapping
        if hasattr(hf_labels, 'names'):
            name2int = {name: hf_labels.str2int(name) for name in hf_labels.names}
            class_indices = [name2int[c] for c in classes if c in name2int]
        else:
            # If labels are already integers, assume classes are the indices
            class_indices = classes
        
        # Filter by specified classes
        ds = ds.filter(lambda x: x['label'] in class_indices)
        
        # Remap labels to contiguous range
        present_labels = sorted(list(set(ds['label'])))
        label_map = {old: new for new, old in enumerate(present_labels)}
        ds = ds.map(lambda x: {'label': label_map[x['label']]})
        
        # Update features
        num_labels = len(present_labels)
        new_features = ds.features.copy()
        new_features["label"] = datasets.ClassLabel(num_classes=num_labels)
        ds = ds.cast(new_features)
    
    # If samples_per_class is specified, create balanced subset
    if samples_per_class is not None:
        label_counts = {}
        indices_to_keep = []
        
        for idx, label in enumerate(ds["label"]):
            if label_counts.get(label, 0) < samples_per_class:
                indices_to_keep.append(idx)
                label_counts[label] = label_counts.get(label, 0) + 1
        
        ds = ds.select(indices_to_keep)

    if logger:
        logger.end_timer(f"loading_{source}")
    return ds

