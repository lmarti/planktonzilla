"""Dataset generation and filtering utilities."""

from datasets import load_dataset
import datasets


def generate_id_subset_planktonzilla(samples_per_class, split="test"):
    """
    Create a balanced subset of the PlanktonZilla dataset with a specific
    number of samples per class.
    
    This function efficiently creates a subset by only iterating through
    the label column rather than loading entire samples.
    
    Args:
        samples_per_class (int): Number of samples to select per class.
        split (str): Dataset split to use ('train', 'test', etc.). Default is 'test'.
        
    Returns:
        tuple: (subset_dataset, num_labels)
            - subset_dataset: HuggingFace dataset with selected samples
            - num_labels: Number of unique labels in the subset
    """
    ds = load_dataset(
        "project-oceania/planktonzilla_only_plankton",
        split=split,
        num_proc=4
    )
    
    # Trackers for our indices
    label_counts = {}
    indices_to_keep = []
    
    # Iterate ONLY through the 'label' column.
    # This is incredibly fast because it ignores the rest of the data.
    for idx, label in enumerate(ds["label"]):
        # If we haven't reached our limit for this specific label...
        if label_counts.get(label, 0) < samples_per_class:
            indices_to_keep.append(idx)  # Save the row index
            label_counts[label] = label_counts.get(label, 0) + 1
    
    # Create the new subset dataset
    subset_ds = ds.select(indices_to_keep)
    num_labels = len(set(subset_ds['label']))
    
    return subset_ds, num_labels


def generate_planktonzilla_full(split="train"):
    """
    Load the complete PlanktonZilla dataset without filtering.
    
    Args:
        split (str): Dataset split to use ('train', 'test', etc.). Default is 'train'.
        
    Returns:
        tuple: (dataset, num_labels)
            - dataset: Full HuggingFace dataset
            - num_labels: Number of unique labels in the dataset
    """
    ds_fit = load_dataset(
        "project-oceania/planktonzilla_only_plankton",
        split=split,
        num_proc=4
    )
    num_labels = len(set(ds_fit['label']))
    return ds_fit, num_labels


def get_splits_zoolake(hf_zoolake="project-oceania/zoolake", 
                       id_classes=None, 
                       ood_classes=None):
    """
    Split the ZooLake dataset into ID (in-distribution) and OOD (out-of-distribution) subsets.
    
    This function filters the dataset based on provided class lists, remaps ID labels
    to a contiguous range starting from 0, and returns both ID and OOD datasets.
    
    Args:
        hf_zoolake (str): HuggingFace dataset identifier for ZooLake.
        id_classes (list): List of class names to include as ID. If None, uses default.
        ood_classes (list): List of class names to include as OOD. If None, uses default.
        
    Returns:
        tuple: (ds_id_only, ds_ood_only, num_labels)
            - ds_id_only: Filtered and remapped ID dataset
            - ds_ood_only: Filtered OOD dataset
            - num_labels: Number of classes in the ID dataset
    """
    # Default ID classes
    if id_classes is None:
        id_classes = [
            'aphanizomenon', 'asplanchna', 'asterionella', 'bosmina', 'brachionus',
            'ceratium', 'chaoborus', 'conochilus', 'cyclops', 'daphnia',
            'diaphanosoma', 'diatom_chain', 'dinobryon', 'eudiaptomus', 'filament',
            'fragilaria', 'kellicottia', 'keratella_cochlearis',
            'keratella_quadrata', 'leptodora', 'nauplius', 'paradileptus',
            'polyarthra', 'rotifers', 'synchaeta', 'trichocerca', 'uroglena',
            'unknown_plankton',
        ]
    
    # Default OOD classes
    if ood_classes is None:
        ood_classes = [
            'dirt', 'fish', 'hydra', 'unknown', 
            'copepod_skins', 'daphnia_skins', 'maybe_cyano'
        ]
    
    ds_full = load_dataset(hf_zoolake, split="train")
    hf_labels = ds_full.features['label']
    name2int = {name: hf_labels.str2int(name) for name in hf_labels.names}
    
    id_indices = [name2int[c] for c in id_classes if c in name2int]
    near_ood_indices = [name2int[c] for c in ood_classes if c in name2int]
    
    # 1. Filter ID
    ds_id_only = ds_full.filter(lambda x: x['label'] in id_indices)
    
    # 2. Remap Labels to contiguous range [0, num_classes-1]
    present_labels = sorted(list(set(ds_id_only['label'])))
    final_label_map = {old: new for new, old in enumerate(present_labels)}
    ds_id_only = ds_id_only.map(lambda x: {'label': final_label_map[x['label']]})
    
    # 3. Update Features
    num_labels = len(present_labels)
    new_features = ds_id_only.features.copy()
    new_features["label"] = datasets.ClassLabel(num_classes=num_labels)
    ds_id_only = ds_id_only.cast(new_features)
    
    # 4. Filter OOD
    ds_ood_only = ds_full.filter(lambda x: x['label'] in near_ood_indices)
    
    return ds_id_only, ds_ood_only, num_labels
