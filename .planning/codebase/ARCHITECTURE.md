<!-- refreshed: 2026-05-12 -->
# Architecture

**Analysis Date:** 2026-05-12

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER ENTRY (CLI / SLURM / pytest)                    │
│  poetry run pz_train  ·  poetry run pz_import_dataset                         │
│  scripts/train.sh  ·  scripts/train_clip.sh  ·  scripts/push_dataset.sh       │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │  CLI overrides
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    HYDRA CONFIG COMPOSITION  (`configs/`)                     │
│  train.yaml  ─►  model/, dataset/, training_arguments/, augmentation/,        │
│                  custom_loss/, tracking/, peft/, paths/, hydra/, extras/,     │
│                  experiment/, hparams_search/, debug/, local/                 │
│  import_dataset.yaml ─►  dataset_import/, paths/, tracking/, hydra/, extras/  │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │  DictConfig
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            APPLICATION ENTRY POINTS  (`planktonzilla/`)                       │
│  ┌──────────────────────┐    ┌─────────────────────────────────────────┐     │
│  │ `planktonzilla/      │    │ `planktonzilla/dataset_import/          │     │
│  │  train.py:main`      │    │  import_dataset.py:main`                │     │
│  │  @hydra.main         │    │  @hydra.main                            │     │
│  └─────────┬────────────┘    └─────────────────┬───────────────────────┘     │
└────────────┼──────────────────────────────────┼──────────────────────────────┘
             │ hydra.utils.instantiate           │ hydra.utils.instantiate
             ▼                                   ▼
┌──────────────────────────────────────┐  ┌─────────────────────────────────────┐
│   TRAINING PIPELINE                  │  │  DATASET IMPORT PIPELINE             │
│   `planktonzilla/train.py:train`     │  │  `planktonzilla/dataset_import/`     │
│                                      │  │  `dataset_importer.DatasetImporter`  │
│  1. validate_environment()           │  │  1. _download_and_extract            │
│  2. DatasetWrapper                   │  │  2. _prepare_imagefolder (subclass)  │
│     (`planktonzilla/dataset.py`)     │  │  3. load `imagefolder` HF dataset    │
│  3. augmentation (torchvision.v2)    │  │  4. _push_to_hub                     │
│  4. model: HF AutoModel              │  │  5. update_dataset_metadata          │
│     OR ClipClassifier                │  │     (DataCard + mean/std)            │
│     (`planktonzilla/clip_model.py`)  │  │  6. cleanup                          │
│  5. optional PEFT/LoRA adapters      │  └─────────────────┬───────────────────┘
│  6. TrainingArguments                │                    │
│  7. custom_loss (subclass of         │                    │
│     `loss.AbstractHFLoss`)           │                    │
│  8. HuggingFace Trainer              │                    │
└────────────────┬─────────────────────┘                    │
                 │                                          │
                 ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL STORES & TRACKERS                                │
│  HuggingFace Hub (datasets, models, dataset cards)                            │
│  Weights & Biases (`wandb/`)  ·  MLflow  ·  Trackio                           │
│  Hydra outputs: `logs/{task_name}/runs/{date_time}_{slurm_id}/`               │
│  Local data cache: `data/{importer_class_lower}_imagefolder/`,                │
│                    `data/{importer_class_lower}_raw_download/`                │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `train.main` (Hydra entry) | Compose train config, call `train(cfg)`, return optimized metric | `planktonzilla/train.py` |
| `train.train` | Orchestrate dataset → model → loss → Trainer → fit/eval/push | `planktonzilla/train.py` |
| `train.compute_metrics` | sklearn-based accuracy/F1/precision/recall over Trainer eval batches | `planktonzilla/train.py` |
| `train.validate_environment` | Probe `HF_TOKEN`, `WANDB_*`, `MLFLOW_*` and login to HF | `planktonzilla/train.py` |
| `DatasetWrapper` | HF `load_dataset` wrapper: split creation, label maps, `with_transform` | `planktonzilla/dataset.py` |
| `augment_and_transform_batch` | Per-batch transform + augmentation hook applied via `with_transform` | `planktonzilla/dataset.py` |
| `compute_mean_and_std_dev` | Per-channel normalization stats for dataset cards | `planktonzilla/dataset.py` |
| `ClipClassifier` | Wrap an `open_clip` visual tower with a fresh `nn.Linear` head, exposing HF `ImageClassifierOutput` | `planktonzilla/clip_model.py` |
| `AbstractHFLoss` (+ subclasses) | Loss API consuming `ImageClassifierOutputWithNoAttention.logits` and target labels | `planktonzilla/loss.py` |
| `DatasetImporter` (+ subclasses) | Download → extract → ImageFolder layout → `load_dataset("imagefolder")` → push HF | `planktonzilla/dataset_import/dataset_importer.py` |
| `import_dataset.main` (Hydra entry) | Compose import config, dispatch action: `import` / `update-metadata` / `show` | `planktonzilla/dataset_import/import_dataset.py` |
| `task_wrapper` | Decorator: timing log, exception capture, exec_time.log, close loggers | `planktonzilla/utils/hydra.py` |
| `extras` / `print_config_tree` / `enforce_tags` | Pre-task utilities driven by `cfg.extras.*` | `planktonzilla/utils/hydra.py`, `planktonzilla/utils/rich_utils.py` |
| `get_pylogger` | Project-wide stdlib logger factory | `planktonzilla/utils/logger.py` |
| `open_clip` (vendored) | Vision encoders for CLIP backbones (`open_clip.create_model_and_transforms`) | `open_clip/src/open_clip/` |
| `open_clip_train` (vendored) | Pretraining loop for CLIP (used out-of-package by `scripts/train_clip.sh`) | `open_clip/src/open_clip_train/` |

## Pattern Overview

**Overall:** Hydra-composed, config-driven ML pipeline framework wrapping HuggingFace Transformers `Trainer`, with a parallel dataset-importer pipeline. Two distinct entry points (`pz_train`, `pz_import_dataset`) each rooted in `pyrootutils.setup_root` and decorated with `@hydra.main`.

**Key Characteristics:**
- **Hydra-first composition.** Every runtime concern (model, dataset, loss, augmentation, trainer args, tracking, PEFT, paths) is a YAML group under `configs/`. Code instantiates components from the merged `DictConfig` via `hydra.utils.instantiate(cfg.<group>, _convert_="all")`.
- **HuggingFace as the backbone.** Datasets flow through `datasets.load_dataset`, models default to `transformers.AutoModelForImageClassification.from_pretrained`, training is delegated to `transformers.Trainer`, artifacts can `push_to_hub` to `huggingface.co/project-oceania`.
- **Vendored OpenCLIP coexists with the package.** `open_clip/` is a copy of the upstream library used as an importable Python source tree (added to `PYTHONPATH` in `scripts/train_clip.sh`). It is consumed in two modes: (a) `open_clip.create_model_and_transforms` from `planktonzilla.clip_model.ClipClassifier` for fine-tuning, (b) `python -m open_clip_train.main` for full CLIP pretraining on shards (Jean Zay SLURM).
- **Reproducibility.** Hydra writes per-run output dirs (`logs/{task_name}/runs/{date}_{time}_{slurm}/`) containing `config_tree.log`, `tags.log`, `exec_time.log`. `transformers.set_seed(cfg.seed)` plus `data_seed` in `TrainingArguments`. W&B/MLflow/Trackio configured exclusively via env vars set inside `train.train`.
- **Imbalanced-class focus.** A first-class `custom_loss` config group with six losses (CE, Focal, LDAM, MaxMargin, Asymmetric, RAL, BalancedMetaSoftmax) all consuming HF `ImageClassifierOutputWithNoAttention.logits`. `cls_num_list` is auto-injected from `DatasetWrapper.cls_num_list` via a try/except instantiate fallback.

## Layers

**Configuration layer:**
- Purpose: Compose runtime experiment from orthogonal YAML groups.
- Location: `configs/`
- Contains: Top-level `train.yaml`, `import_dataset.yaml` and group dirs (`model/`, `dataset/`, `dataset_import/`, `training_arguments/`, `augmentation/`, `custom_loss/`, `tracking/`, `peft/`, `paths/`, `hydra/`, `extras/`, `experiment/`, `hparams_search/`, `debug/`, `local/`).
- Depends on: Nothing at runtime; resolved by Hydra at app start.
- Used by: `planktonzilla/train.py`, `planktonzilla/dataset_import/import_dataset.py`.

**Application / orchestration layer:**
- Purpose: Hydra entry points wiring config to domain logic.
- Location: `planktonzilla/train.py`, `planktonzilla/dataset_import/import_dataset.py`.
- Contains: `@hydra.main` decorated `main()` and `@task_wrapper`-decorated business function (`train`, `import_dataset`).
- Depends on: Domain layer + utils.
- Used by: Console scripts in `pyproject.toml` (`pz_train`, `pz_import_dataset`).

**Domain layer:**
- Purpose: Encapsulate ML primitives.
- Location: `planktonzilla/dataset.py`, `planktonzilla/clip_model.py`, `planktonzilla/loss.py`, `planktonzilla/dataset_import/dataset_importer.py`.
- Contains: `DatasetWrapper`, `ClipClassifier`, `AbstractHFLoss` family, `DatasetImporter` family.
- Depends on: HuggingFace `transformers` + `datasets`, `torchvision.transforms.v2`, vendored `open_clip`, `huggingface_hub`.
- Used by: Application layer.

**Utility layer:**
- Purpose: Hydra/runtime helpers.
- Location: `planktonzilla/utils/`.
- Contains: `hydra.py` (`task_wrapper`, `extras`, `get_metric_value`, `close_loggers`), `rich_utils.py` (`print_config_tree`, `enforce_tags`), `logger.py` (`get_pylogger`).
- Depends on: `hydra-core`, `omegaconf`, `rich`, optionally `wandb`.
- Used by: Application layer.

**Vendored OpenCLIP layer:**
- Purpose: Pretrained CLIP visual backbones + standalone CLIP pretraining loop.
- Location: `open_clip/src/open_clip/` (library), `open_clip/src/open_clip_train/` (pretraining loop).
- Contains: Model factory (`factory.py`), config JSONs (`open_clip/src/open_clip/model_configs/*.json`), `model.py`, `transformer.py`, `timm_model.py`, `coca_model.py`, etc.
- Depends on: `torch`, `timm`, `huggingface_hub`, `webdataset`.
- Used by: `planktonzilla/clip_model.py:ClipClassifier` (fine-tune path) and `scripts/train_clip.sh` (`-m open_clip_train.main` for full pretraining on `webdataset` tar shards).

**Notebooks / one-off scripts layer:**
- Purpose: Exploratory dataset assembly, evaluation, hub publishing.
- Location: `notebooks/`.
- Contains: `gen_planktonzilla.py`, `gen_planktonzilla_ood.py`, `add_planktonzilla.py`, `push_planktonzilla.py`, `save_planktonzilla_for_clip.py`, `metrics_*.ipynb`, `fix_taxo.ipynb`, `gen_datasets.ipynb`, `load_models.ipynb`.
- Depends on: Domain layer (`planktonzilla.dataset_import.dataset_importer`, `planktonzilla.utils.logger`), `datasets`, `polars`/`pandas`, `joblib`.
- Used by: Manual / SLURM scripts (`scripts/save_plankt.sh`, `scripts/save_plankt_plus.sh`, `scripts/push_planktonzilla.sh`).

**Outputs / runtime artifact layer:**
- Purpose: Persist run metadata, models, metrics, telemetry.
- Location: `logs/`, `wandb/`, `data/`.
- Contains: `logs/{train,train_node,import_dataset}/runs/<timestamp>/{config_tree.log,tags.log,exec_time.log,checkpoints,...}`, `wandb/run-<timestamp>-<id>/{files,logs,tmp}`, `data/<importer>_imagefolder/`, `data/<importer>_raw_download/`.

## Data Flow

### Dataset Import Pipeline (`pz_import_dataset`)

1. CLI: `poetry run pz_import_dataset dataset_import=isiisnet action=import` → `planktonzilla/dataset_import/import_dataset.py:main` (`@hydra.main` with `config_name="import_dataset.yaml"`).
2. Hydra composes `configs/import_dataset.yaml` (defaults: `paths`, `extras`, `hydra`, `dataset_import`, `tracking`).
3. `task_wrapper` runs `extras(cfg)` (warnings/tags/print).
4. `hydra.utils.instantiate(cfg.dataset_import)` → concrete `DatasetImporter` subclass (e.g. `ISIISNetDatasetImporter` from `configs/dataset_import/isiisnet.yaml`, `_target_: planktonzilla.dataset_import.dataset_importer.ISIISNetDatasetImporter`).
5. Action dispatch in `import_dataset.import_dataset` (`planktonzilla/dataset_import/import_dataset.py:45-55`):
   - `import` → `dataset_importer.import_dataset()`:
     - `_download_and_extract` via `datasets.download.DownloadManager` into `data/{importer_class_lower}_raw_download/`.
     - `_prepare_imagefolder` (subclass-specific: copies/renames/unzips into `data/{importer_class_lower}_imagefolder/<class>/<file>` and optionally split subfolders `train/val/test`).
     - `load_dataset("imagefolder", data_files=..., name=hf_dataset_name)` → in-memory HF dataset.
     - `_push_to_hub` → `hf_dataset.push_to_hub(...)` with retries.
     - `update_dataset_metadata` → builds Markdown report (label histograms via `plotext.simple_bar`), per-channel mean/std (`compute_mean_and_std_dev`), pushes a `DatasetCard` rendered from `DATACARD_TEMPLATE`.
     - `cleanup` if `cleanup_after_processing=True`.
   - `update-metadata` → only refreshes the dataset card.
   - `show` → prints builder info + card markdown via Rich.
6. Output: HF dataset published as `project-oceania/<hf_dataset_name>`; local raw + imagefolder retained under `data/`.

### Training Pipeline (`pz_train`)

1. CLI: `poetry run pz_train dataset=isiisnet model=resnet18 custom_loss=focal training_arguments.num_train_epochs=10` → `planktonzilla/train.py:main` (`@hydra.main` with `config_name="train.yaml"`).
2. `pyrootutils.setup_root` sets `PROJECT_ROOT` env, adds repo to `sys.path`, loads `.env`.
3. `torch.multiprocessing.set_sharing_strategy("file_system")` (`planktonzilla/train.py:14-16`).
4. Hydra composes `configs/train.yaml` (defaults: `model=resnet18.yaml`, `dataset=lensless.yaml`, `training_arguments=default.yaml`, `paths`, `extras`, `hydra`, `augmentation=default.yaml`, `tracking=default.yaml`, `custom_loss=default.yaml`, `peft=default.yaml`, optional `experiment`, `hparams_search`, `local`, `debug`).
5. `train.train` (decorated by `task_wrapper`):
   - `validate_environment()` logs presence of `HF_TOKEN`/`WANDB_API_KEY`/`MLFLOW_TRACKING_URI`, calls `huggingface_hub.login`.
   - `set_seed(cfg.seed, cfg.get("deterministic"))`.
   - `DatasetWrapper` instantiated from `cfg.dataset` → `prepare_datasets(augmentation)`:
     - `datasets.load_dataset(self.name, streaming=...)` (e.g. `project-oceania/isiisnet`).
     - Builds `id2label`/`label2id` from `train.features["label"].names`.
     - If `test_split_name` missing in dataset dict → `train_test_split(test_split, stratify_by_column="label", seed=split_seed)`. Same logic for validation split.
     - Computes `cls_num_list = np.unique(train["label"], return_counts=True)[1]`.
     - Attaches `with_transform(partial(augment_and_transform_batch, transform=self.transform, augmentation=...))` to train (with augmentation) and val/test (without).
   - Loads `DatasetCard` for the HF dataset and logs it.
   - Instantiates the model via `hydra.utils.instantiate(cfg.model, id2label=..., label2id=..., num_labels=..., _convert_="all")`. Try/except fallback to `ClipClassifier` (passing `num_features=cfg.num_features`) when `AutoModelForImageClassification.from_pretrained` fails (`planktonzilla/train.py:158-175`).
   - If `cfg.peft` truthy → iterate adapters and `model.add_adapter(adapter, adapter_name=name)` (LoRA via `peft.LoraConfig`).
   - If `cfg.freeze_backbone` → unfreeze only params containing `"classifier"` or `"head"`.
   - `TrainingArguments` instantiated from `cfg.training_arguments`. If `cfg.model_push_to_hub`, fabricate `hub_model_id = "<org>/<prefix>_<model>_<dataset>"` (then immediately overrides `push_to_hub=False` — see CONCERNS in code at `planktonzilla/train.py:196-211`). If `cfg.resume_from_ckpt_path` → `training_args.resume_from_checkpoint`.
   - Custom loss (`planktonzilla/train.py:217-226`): instantiate `cfg.custom_loss.custom_loss` (or `cfg.custom_loss`); first try without args, fallback with `cls_num_list=dataset_wrapper.cls_num_list`; wrap in `partial(loss_instance.forward)`.
   - Tracking env-vars set per `cfg.tracking.use_{wandb,mlflow,trackio}`. `training_args.report_to` set to the resulting list. `run_name = "<model>__<dataset>"`.
   - `Trainer(model, args, train_dataset, eval_dataset=val, compute_metrics=compute_metrics, compute_loss_func=custom_loss)` constructed.
   - `trainer.train()` → `train_metrics`. `trainer.evaluate(val, metric_key_prefix="val")`. `trainer.evaluate(test, metric_key_prefix="test")`.
   - Optional `trainer.push_to_hub(dataset=..., license="mit")`.
   - Returns `metric_dict`, `object_dict`.
6. `main` extracts `cfg.optimized_metric` via `get_metric_value` and returns the float (Hydra/Optuna sweepers consume this).
7. Outputs: HF Trainer checkpoints under `cfg.training_arguments.output_dir = ${paths.output_dir} = ${hydra:runtime.output_dir} = logs/train/runs/<timestamp>_<slurm>/`; W&B run dirs under `wandb/run-<timestamp>-<id>/`; logs `config_tree.log`, `tags.log`, `exec_time.log` colocated.

### Inference / Evaluation

There is no dedicated inference module. Evaluation is performed in-process by `Trainer.evaluate(...)` inside `train.train`. Post-hoc analysis happens in `notebooks/metrics_clip.ipynb` and `notebooks/metrics_paper.ipynb`, and dataset/model assembly for downstream evaluation in `notebooks/gen_planktonzilla.py`, `notebooks/gen_planktonzilla_ood.py`, `notebooks/save_planktonzilla_for_clip.py`, `notebooks/push_planktonzilla.py`, `notebooks/load_models.ipynb`.

### CLIP Pretraining Pipeline (vendored `open_clip_train`)

1. SLURM submission: `sbatch scripts/train_clip.sh`.
2. The script `cd $WORK/am/open_clip` and exports `PYTHONPATH=$WORK/planktonzilla/open_clip/src:$PYTHONPATH` (`scripts/train_clip.sh:20`).
3. `srun torchrun --nproc_per_node=4 --nnodes=16 ... -m open_clip_train.main --train-data ".../shard_{00000..01771}.tar" --dataset-type webdataset --model EVA02-L-14 ...` (`scripts/train_clip.sh:36-58`).
4. Shards are produced earlier by `notebooks/save_planktonzilla_for_clip.py:export_to_tar_shards` from a HuggingFace `DatasetDict` (image + text label per sample).
5. Outputs are managed by `open_clip_train.main` (independent of `planktonzilla/`).

**State Management:**
- No global mutable state in `planktonzilla/` beyond two safe one-shots: `OmegaConf.register_new_resolver("eval", eval)` (idempotent try/except) in `train.py` and `import_dataset.py`, and `torch.multiprocessing.set_sharing_strategy("file_system")` at `train.py` import time.
- Per-run state lives in `cfg` (`DictConfig`), `dataset_wrapper.dataset` (HF `DatasetDict`), `Trainer` instance, and on-disk `output_dir`.

## Key Abstractions

**`DatasetWrapper` (`planktonzilla/dataset.py:96`):**
- Purpose: Single contract between Hydra config and HF Trainer for tabular metadata (splits, label maps, transforms, class counts).
- Pattern: `@dataclass` with `__post_init__`, instantiated by Hydra (`_target_: planktonzilla.dataset.DatasetWrapper`); `prepare_datasets(augmentation)` mutates `self.dataset` lazily.

**`AbstractHFLoss` (`planktonzilla/loss.py:13`):**
- Purpose: Common contract `forward(output: ImageClassifierOutputWithNoAttention, target, **kwargs) -> Tensor` so any loss is plug-compatible with HuggingFace `Trainer.compute_loss_func`.
- Subclasses: `FocalLoss`, `LDAMLoss`, `MaximumMarginLoss`, `AsymmetricLoss`, `RobustAsymmetricLoss`, `BalancedMetaSoftmaxLoss`, `CrossEntropyLossHF`. Each is instantiable via `_target_` in `configs/custom_loss/*.yaml`.

**`DatasetImporter` (`planktonzilla/dataset_import/dataset_importer.py:163`):**
- Purpose: Template-method base for ETL: subclasses override `_prepare_imagefolder` and (optionally) `_download_and_extract`. Common `import_dataset()` orchestrates download, prepare, load, push, cleanup.
- Subclasses: `LenslessDatasetImporter`, `ZooLakeDatasetImporter`, `ZooScanNetDatasetImporter`, `WHOIPlanktonDatasetImporter`, `JEDISystemsOceansCPICSDatasetImporter`, `UVP6NetDatasetImporter`, `ZooCAMNetDatasetImporter`, `FlowCAMNetDatasetImporter`, `ISIISNetDatasetImporter`, `PlanktoScopeDatasetImporter`, `GlobalUVP5NetDatasetImporter`, `PlanktonSet1DatasetImporter`, `SYKEIFCB2022DatasetImporter`, `SYKEZooScan2024DatasetImporter`.
- Each subclass paired with a YAML at `configs/dataset_import/<dataset>.yaml` (e.g. `configs/dataset_import/isiisnet.yaml`).

**`ClipClassifier` (`planktonzilla/clip_model.py:5`):**
- Purpose: Adapter from `open_clip` visual towers to HF classification interface. Exposes `forward(pixel_values, labels=None, ...) -> ImageClassifierOutput` so HF `Trainer` is unaware of the CLIP origin.
- Pattern: Two paths — ViT-style towers (uses `model.proj = None` then wraps in `nn.Sequential([visual, nn.Linear(num_features, num_labels)])`); timm-backed towers (replaces `trunk.head` with a fresh `nn.Linear`).

**Custom-loss registry pattern:**
- Indirection through Hydra `_target_`. `train.train` first tries `instantiate(cfg_loss)`, on `Exception` falls back to `instantiate(cfg_loss, cls_num_list=dataset_wrapper.cls_num_list)` to support distribution-aware losses (`planktonzilla/train.py:217-226`).

**`task_wrapper` (`planktonzilla/utils/hydra.py:22`):**
- Purpose: Cross-cutting decorator providing extras/timing/exception-logging/log-closing for any Hydra task; required by `train.train` and `import_dataset.import_dataset`.

## Entry Points

**Console scripts (`pyproject.toml` `[project.scripts]`):**
- `pz_train` → `planktonzilla.train:main` (active).
- `pz_import_dataset` → `planktonzilla.dataset_import.import_dataset:main` (active).
- `pz_prepare_train` → `planktonzilla.prepare_train:main` (declared but **module not present** — see CONCERNS).
- `pz_push_model` → `planktonzilla.push_model:main` (declared but **module not present** — see CONCERNS).

**Module main blocks:**
- `planktonzilla/train.py:316-317` — direct `python -m planktonzilla.train`.
- `planktonzilla/dataset_import/import_dataset.py:70-71` — direct `python -m planktonzilla.dataset_import.import_dataset`.
- `planktonzilla/utils/rich_utils.py:130-135` — local debug entry that composes `train.yaml` and prints the config tree.

**SLURM/Bash scripts (`scripts/`):**
- `scripts/train.sh` — multi-node `torchrun` for `pz_train` on Jean Zay H100 (8 nodes × 4 GPUs).
- `scripts/train_clip.sh` — multi-node `torchrun` for `python -m open_clip_train.main` on `webdataset` tar shards (16 nodes × 4 GPUs).
- `scripts/push_dataset.sh` — `srun pz_import_dataset action=import dataset_import.data_dir=...` on Jean Zay `compil` partition.
- `scripts/save_plankt.sh` / `scripts/save_plankt_plus.sh` — `srun python notebooks/save_planktonzilla2.py` / `notebooks/add_planktonzilla.py` (filenames in scripts assume the notebook variant referenced by the SLURM jobs).
- `scripts/push_planktonzilla.sh` — `srun python push_planktonzilla2.py` for hub publishing.

**Tests (`tests/`):**
- `tests/test_train.py` parametrizes Hydra `compose("train", overrides=[...])` over `dataset_names × model_names × losses` and runs `train(cfg)` end-to-end with `training_arguments=test_minirun` (`max_steps=2`).
- `tests/test_datasets.py` parametrizes over `dataset_names`, asserts `DatasetWrapper` instantiation and (CI-skipped) `prepare_datasets`.

## Architectural Constraints

- **Threading:** Default PyTorch threading. `torch.multiprocessing.set_sharing_strategy("file_system")` is forced at `planktonzilla/train.py` import time. Trainer uses `dataloader_num_workers=4`, `dataloader_persistent_workers=true`, `dataloader_pin_memory=true` (`configs/training_arguments/default.yaml`).
- **Distributed training:** All multi-GPU/multi-node training is run through `torchrun` from `scripts/train.sh` and `scripts/train_clip.sh`. `ddp_find_unused_parameters=false` in default training args.
- **Global state:** Two import-time globals in `planktonzilla/train.py` (sharing strategy, `eval` resolver). `pyrootutils.setup_root` mutates `sys.path` and sets `PROJECT_ROOT` env on import of `train.py`, `import_dataset.py`, and the test files.
- **Circular imports:** None observed. `dataset_import/dataset_importer.py` depends on `planktonzilla.dataset.compute_mean_and_std_dev` (one-way).
- **Vendored library boundary:** `open_clip/` is NOT a Poetry-managed dependency; it lives in-tree and must either be on `PYTHONPATH` (the `train_clip.sh` path) or be importable as `open_clip` from the active env. `planktonzilla.clip_model.ClipClassifier` does `import open_clip` and depends on `open_clip.create_model_and_transforms`.
- **Hydra `_target_` indirection:** All component swapping happens via YAML — code never directly imports concrete model/loss/importer classes by name (`planktonzilla/train.py:158-175,217-224`, `planktonzilla/dataset_import/import_dataset.py:43`).
- **Auto-injection of `cls_num_list`:** Distribution-aware losses (`LDAMLoss`, `MaximumMarginLoss`, `BalancedMetaSoftmaxLoss`) require the dataset's class counts. The framework detects the need via try/except, not via signature inspection (`planktonzilla/train.py:217-226`).
- **Splits assumed from `label` column.** `DatasetWrapper.prepare_datasets` calls `train_test_split(..., stratify_by_column="label")` — non-`label` label columns will not stratify correctly even though `label_column_name` is configurable.

## Anti-Patterns

### Bare `except` swallowing the model factory failure

**What happens:** `planktonzilla/train.py:158-175` wraps `AutoModelForImageClassification.from_pretrained` in `try: ... except: ...` and on *any* exception switches to `ClipClassifier`. There is no `except Exception` and no logging of the underlying failure.
**Why it's wrong:** A typo in `cfg.model._args_[0]` or a transient HF Hub error silently routes execution into the CLIP path, often producing a confusing downstream `KeyError`/`AttributeError`. The user has no signal of the original failure.
**Do this instead:** Branch on `cfg.model._target_` (e.g. `if cfg.model._target_.endswith("ClipClassifier"): ...`) or catch a specific exception class and re-raise unexpected ones; always `log.exception(...)`.

### Bare `except` to detect distribution-aware losses

**What happens:** `planktonzilla/train.py:217-224` instantiates `cfg.custom_loss` once; on *any* exception, retries with `cls_num_list=...`. Real bugs in a loss `__init__` get masked as "must be a distribution-aware loss".
**Do this instead:** Inspect `inspect.signature(cls).parameters` for `cls_num_list`, or expose an explicit `requires_class_counts: bool` flag in the loss config.

### Silent `push_to_hub` toggle

**What happens:** When `cfg.model_push_to_hub` is true, `planktonzilla/train.py:196-211` constructs `hub_model_id` and immediately sets `training_args.push_to_hub = False`, then later (`planktonzilla/train.py:292-295`) calls `trainer.push_to_hub(...)` manually.
**Why it's wrong:** Reading the config alone, the `hub_strategy: end` setting (`configs/training_arguments/default.yaml:47`) suggests Trainer-driven pushes, but the code disables that path. A future contributor flipping `push_to_hub` in the YAML expects it to take effect; it won't.
**Do this instead:** Use one mechanism (Trainer-managed pushes via `hub_strategy=end` and `push_to_hub=True`) and remove the manual `trainer.push_to_hub(...)` call, OR keep the manual path and remove `hub_strategy` from training args to avoid contradiction.

### Sibling-instantiation of `DatasetCard` requires hub access at training start

**What happens:** `planktonzilla/train.py:151-153` calls `DatasetCard.load(cfg.dataset.name)` immediately after `prepare_datasets`. If `HF_HUB_OFFLINE=1` (set by `scripts/train.sh:23`) and the card isn't cached, this raises before training starts.
**Do this instead:** Wrap the card load in a try/except and degrade gracefully when offline.

### Vendored `open_clip` outside dependency management

**What happens:** `open_clip/` lives in-tree and is not declared in `pyproject.toml`. `scripts/train_clip.sh` injects it via `PYTHONPATH`; `planktonzilla/clip_model.py` does `import open_clip` assuming env-installed availability.
**Why it's wrong:** Drift between vendored copy and runtime can produce hard-to-diagnose issues, and contributors aren't sure which copy is being used.
**Do this instead:** Either pin a version of `open_clip-torch` in `pyproject.toml` and delete `open_clip/`, or expose the vendored path explicitly in `pyproject.toml` (`packages = [{ include = "open_clip", from = "open_clip/src" }]`) so import resolution is deterministic.

## Error Handling

**Strategy:** Logging-first, fail-loud at the orchestration layer.

**Patterns:**
- `task_wrapper` (`planktonzilla/utils/hydra.py:35-50`) wraps each Hydra task: catches all exceptions, calls `log.exception("")` (saves the traceback to the per-run `.log`), then `raise`s. Always writes `exec_time.log` and calls `close_loggers()` (which `wandb.finish()`s if a run is open).
- `validate_environment` (`planktonzilla/train.py:53-90`) emits warnings (not errors) when optional integrations are missing.
- `DatasetImporter._push_to_hub` retries `push_to_hub` `push_to_hub_retries` times (default 10) with exception logging.
- `is_valid_image_file` (`planktonzilla/dataset_import/dataset_importer.py:151-159`) silently drops corrupt files when `check_image_file_integrity=true`.
- Two bare `except:` clauses are documented under Anti-Patterns above.

## Cross-Cutting Concerns

**Logging:** Stdlib `logging` via `planktonzilla.utils.logger.get_pylogger(__name__)`. Hydra installs `colorlog` formatters by default (`configs/hydra/default.yaml`). Per-run logs land in `${paths.log_dir}/{task_name}/runs/<timestamp>/`.

**Validation:** OmegaConf interpolation enforces presence (`???` sentinels in `configs/dataset/default.yaml`, `configs/model/default.yaml`, `configs/custom_loss/ldam.yaml`). `DatasetImporter._validate` enforces `hf_token` and `hf_dataset_name` when `push_to_hub=True`. No JSON/Pydantic schema validation.

**Authentication:** `huggingface_hub.login(new_session=False, write_permission=True)` from `validate_environment`, gated on `HF_TOKEN` env var. W&B/MLflow auth via env vars (`WANDB_API_KEY`, `MLFLOW_TRACKING_USERNAME`/`PASSWORD`).

**Tracking:** Configured by `configs/tracking/default.yaml` (`use_wandb`, `use_mlflow`, `use_trackio`). Activation flips happen by mutating `os.environ` inside `train.train` (`planktonzilla/train.py:228-251`), then assigning `training_args.report_to`. Run names are deterministic: `<model>__<dataset>` (e.g. `microsoft_resnet-18__project-oceania_isiisnet`).

**Reproducibility:** `seed: ${dataset.split_seed}` in `configs/train.yaml` (default 42); `transformers.set_seed(seed, deterministic)`; `data_seed=42` in `TrainingArguments`. Per-run output dir captures `config_tree.log`, `tags.log`, full Hydra `.hydra/` snapshot.

**Reproducibility infra:**
- Configs serialized into per-run output dir by Hydra.
- W&B run dirs (`wandb/run-<timestamp>-<id>/`) hold full system metadata, file snapshots, and metric history.
- HuggingFace `DatasetCard` updated by importers includes per-channel mean/std and class histograms (`DATACARD_TEMPLATE` in `planktonzilla/dataset_import/dataset_importer.py:43-76`).
- Hub model id includes both model and dataset (`<prefix>_<model_path>_<dataset_path>`), all `/` replaced with `_`.

---

*Architecture analysis: 2026-05-12*
