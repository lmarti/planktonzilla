# Using the Config-Based Runner

The config-based runner provides a flexible, YAML-driven approach to running OOD detection experiments.

## Quick Start

1. **Edit the config file** (or create your own):
   ```bash
   # Edit configs/example_config.yaml to set your parameters
   ```

2. **Run the experiment**:
   ```bash
   conda activate ood_env
   cd experiments
   python run_from_config.py --config example_config
   ```

   Note: Just provide the config name without path or `.yaml` extension!

## Configuration Format

### Model Configuration
```yaml
model:
  backbone: "resnet18"  # Model architecture
  device: "cuda"        # Device to use
```

### Dataset Configuration
Each dataset section (id, ood, fit) supports:
- `source`: HuggingFace dataset path or local path
- `split`: Dataset split (train, test, validation)
- `classes`: (Optional) List of class names to keep
- `samples_per_class`: (Optional) Number of samples per class

**Example - Keep all classes:**
```yaml
dataset:
  id:
    source: "project-oceania/planktonzilla_only_plankton"
    split: "test"
    classes: null  # Keep all classes
    samples_per_class: 10
```

**Example - Filter specific classes:**
```yaml
dataset:
  ood:
    source: "project-oceania/zoolake"
    split: "train"
    classes:  # Keep only these classes
      - dirt
      - fish
      - hydra
    samples_per_class: 50
```

### DataLoader Configuration
```yaml
dataloader:
  batch_size: 64
  num_workers: 4
  pin_memory: true
```

### OOD Methods Configuration
```yaml
ood_methods:
  - name: "Energy"
    config:
      t: 1.0
  - name: "Mahalanobis"
    config:
      eps: 0.0
```

### Experiment Configuration
```yaml
experiment:
  seed: 42
  name: "my_experiment"
  output_dir: "ood_results"
  save: true  # Save results to disk
```

### Advanced Configuration
```yaml
advanced:
  auto_scale_batch_size: true  # Auto-scale for multi-GPU
  memory_efficient: false
  mixed_precision: false
```

## Examples

### Example 1: Simple ResNet-18 Experiment
```yaml
model:
  backbone: "resnet18"
  device: "cuda"

dataset:
  id:
    source: "my-dataset/plankton"
    split: "test"
    classes: null
    samples_per_class: 100
  ood:
    source: "my-dataset/ood-samples"
    split: "test"
    classes: null
    samples_per_class: 100
  fit:
    source: "my-dataset/plankton"
    split: "train"
    classes: null
    samples_per_class: 200
```

### Example 2: Class Filtering
```yaml
dataset:
  id:
    source: "project-oceania/planktonzilla_only_plankton"
    split: "test"
    classes:  # Only use these classes
      - copepod
      - diatom
      - chaetoceros
    samples_per_class: 50
```

### Example 3: Local Dataset
```yaml
dataset:
  id:
    source: "/path/to/local/dataset"  # Local path
    split: "test"
    classes: null
    samples_per_class: null  # Use all samples
```

## Command Line Usage

**Default config:**
```bash
python run_from_config.py
# Uses configs/example_config.yaml by default
```

**Custom config:**
```bash
python run_from_config.py --config my_experiment
# Looks for configs/my_experiment.yaml
```

**With conda environment:**
```bash
conda run -n ood_env python run_from_config.py --config example_config
```

**Create your own config:**
```bash
# Copy the example
cp ../configs/example_config.yaml ../configs/my_experiment.yaml
# Edit it
# Run it
python run_from_config.py --config my_experiment
```

## Notes

- Config files must be in the `configs/` directory
- Only provide the config name (without path or `.yaml` extension)
- The script automatically looks in `../configs/<name>.yaml`
- If `classes` is `null`, all classes from the source will be used
- If `samples_per_class` is `null`, all available samples will be used
- Batch size is automatically scaled when using multiple GPUs (if `auto_scale_batch_size: true`)
- Results are saved to `ood_results/<timestamp>/` when `save: true`
