# Codebase Structure

**Analysis Date:** 2026-05-12

## Directory Layout

```
planktonzilla/                                  # repo root
├── planktonzilla/                              # main Python package (~1.1k LOC)
│   ├── __init__.py                             # empty
│   ├── train.py                                # Hydra entry: pz_train (317 lines)
│   ├── dataset.py                              # DatasetWrapper + transforms (195 lines)
│   ├── clip_model.py                           # ClipClassifier (open_clip → HF) (59 lines)
│   ├── loss.py                                 # AbstractHFLoss + 7 losses (391 lines)
│   ├── dataset_import/
│   │   ├── import_dataset.py                   # Hydra entry: pz_import_dataset
│   │   ├── dataset_importer.py                 # DatasetImporter base + 14 subclasses (777 lines)
│   │   └── public_data/
│   │       ├── lensless_dataset.zip            # vendored small public dataset
│   │       └── README.md
│   └── utils/
│       ├── __init__.py
│       ├── hydra.py                            # task_wrapper, extras, get_metric_value
│       ├── logger.py                           # get_pylogger
│       └── rich_utils.py                       # print_config_tree, enforce_tags
├── open_clip/                                  # vendored OpenCLIP source tree (NOT a Poetry dep)
│   ├── LICENSE
│   └── src/
│       ├── open_clip/                          # library: model.py, factory.py, transformer.py, ...
│       │   └── model_configs/                  # *.json model configs (ViT-*, EVA-*, ViTamin-*, RN50*, etc.)
│       └── open_clip_train/                    # standalone CLIP pretraining loop (main.py, train.py, data.py, ...)
├── configs/                                    # Hydra config root (composed at runtime)
│   ├── train.yaml                              # top-level train config (defaults list)
│   ├── import_dataset.yaml                     # top-level import config
│   ├── augmentation/                           # default, autoaugment, randaugment, trivialaugment, manual, clip_aug
│   ├── custom_loss/                            # default, focal, ldam, asymmetric, ral, max_margin, balanced_meta_softmax
│   ├── dataset/                                # cifar10, isiisnet, lensless, planktonzilla, whoi-plankton, zoolake, ...
│   ├── dataset_import/                         # one YAML per importable dataset (isiisnet, flowcamnet, uvp6net, ...)
│   ├── debug/                                  # default, fdr, limit, overfit, profiler
│   ├── experiment/                             # base_<dataset>.yaml presets, default.yaml, test.yaml
│   ├── extras/default.yaml                     # warnings/tags/print toggles
│   ├── hparams_search/optuna.yaml              # Optuna sweeper integration
│   ├── hydra/
│   │   ├── default.yaml                        # output dir pattern + colorlog
│   │   ├── help/planktonzilla-help.yaml        # custom --help banner
│   │   └── launcher/                           # local_submitit, jeanzay_submitit_h100mono, grid5000_submitit
│   ├── local/.gitkeep                          # opt-in user-local overrides
│   ├── model/                                  # resnet18, vit-base, beit-base, eva02-large-clip-..., timm-*, ...
│   ├── paths/default.yaml                      # root_dir/data_dir/log_dir/output_dir interpolations
│   ├── peft/                                   # default (null), vit-base.yaml (LoRA targets)
│   ├── tracking/default.yaml                   # wandb/mlflow/trackio toggles + env names
│   └── training_arguments/                     # default, experimental, many, test_minirun, accelerate/ (empty)
├── scripts/                                    # SLURM/bash entry helpers (NOT Python)
│   ├── train.sh                                # multi-node torchrun → pz_train (Jean Zay H100)
│   ├── train_clip.sh                           # multi-node torchrun → -m open_clip_train.main
│   ├── push_dataset.sh                         # srun pz_import_dataset action=import
│   ├── save_plankt.sh                          # srun python notebooks/save_planktonzilla2.py
│   ├── save_plankt_plus.sh                     # srun python notebooks/add_planktonzilla.py
│   └── push_planktonzilla.sh                   # srun python notebooks/push_planktonzilla2.py
├── notebooks/                                  # exploration, dataset assembly, evaluation
│   ├── gen_planktonzilla.py                    # build the planktonzilla aggregate dataset
│   ├── gen_planktonzilla_ood.py                # OOD variant
│   ├── add_planktonzilla.py                    # incremental add to aggregate
│   ├── push_planktonzilla.py                   # publish OOD dataset to HF Hub
│   ├── save_planktonzilla_for_clip.py          # export DatasetDict → webdataset tar shards
│   ├── load_models.ipynb                       # quick model loader sanity checks
│   ├── metrics_clip.ipynb                      # CLIP eval / metrics (1.6 MB)
│   ├── metrics_paper.ipynb                     # paper-figure metrics
│   ├── fix_taxo.ipynb                          # taxonomy cleanup
│   └── gen_datasets.ipynb                      # dataset generation experiments
├── tests/                                      # pytest suite
│   ├── __init__.py
│   ├── conftest.py                             # `hydra_conf_path` fixture → "./../configs"
│   ├── shared.py                               # model_names, dataset_names, skip_in_github_ci
│   ├── test_datasets.py                        # DatasetWrapper instantiation + (skipped on CI) prepare_datasets
│   └── test_train.py                           # Full train(cfg) smoke matrix (model × dataset × loss)
├── data/                                       # local raw + prepared dataset caches (gitignored except .gitkeep)
│   ├── isiisnetdatasetimporter_raw_download/   # DownloadManager cache
│   └── isiisnetdatasetimporter_imagefolder/    # class-folder layout consumed by `imagefolder` loader
├── logs/                                       # Hydra per-run output trees
│   ├── train/runs/<YYYY-MM-DD>_<HH-MM-SS>_<slurm|local>/
│   ├── train/multiruns/<...>/<job_num>/
│   ├── train_node/runs/...                     # legacy/aux task name
│   ├── import_dataset/runs/...
│   └── wandb/                                  # wandb mirror (alongside top-level `wandb/`)
├── wandb/                                      # W&B run dirs: run-<timestamp>-<id>/{files,logs,tmp}, debug-internal.log, debug.log
├── docs/
│   └── images/planktonzilla-logo.gif           # README assets
├── .devcontainer/
│   ├── Dockerfile                              # CUDA 12.5 + Ubuntu 22.04 + Python 3.11 + Poetry
│   ├── devcontainer.json
│   └── setup_env.sh                            # writes a .env with git author info
├── .github/
│   └── copilot-instructions.md                 # AI-agent guide, mirrors much of this doc
├── .gsd, .gsd-id, .planning/                   # GSD workflow state
├── pyproject.toml                              # Poetry project + console scripts + ruff/pytest config
├── poetry.lock
├── poetry.toml                                 # Poetry settings (in-project venv etc.)
├── README.md                                   # main entry doc with quick start
├── LICENSE                                     # MIT
├── train.log                                   # last-run training log (root-level)
├── import_dataset.log                          # last-run import log
└── .gitignore
```

## Directory Purposes

**`planktonzilla/`:**
- Purpose: The single Python package distributed as `planktonzilla` (per `pyproject.toml`).
- Contains: Two Hydra `@hydra.main` entry modules (`train.py`, `dataset_import/import_dataset.py`), domain modules (`dataset.py`, `clip_model.py`, `loss.py`, `dataset_import/dataset_importer.py`), and `utils/`.
- Key files: `planktonzilla/train.py`, `planktonzilla/dataset.py`, `planktonzilla/loss.py`, `planktonzilla/clip_model.py`, `planktonzilla/dataset_import/dataset_importer.py`.

**`planktonzilla/dataset_import/`:**
- Purpose: ETL pipeline for plankton public datasets → HuggingFace Hub.
- Contains: `dataset_importer.py` with `DatasetImporter` base + 14 subclasses (one per dataset), `import_dataset.py` Hydra entry, `public_data/` with the lensless dataset zip embedded for offline use.

**`planktonzilla/utils/`:**
- Purpose: Cross-cutting helpers used by both Hydra entries.
- Key files: `planktonzilla/utils/hydra.py` (`task_wrapper`, `extras`, `get_metric_value`, `close_loggers`), `planktonzilla/utils/logger.py` (`get_pylogger`), `planktonzilla/utils/rich_utils.py` (`print_config_tree`, `enforce_tags`).

**`open_clip/`:**
- Purpose: Vendored copy of [OpenCLIP](https://github.com/mlfoundations/open_clip). Provides `open_clip` (library) and `open_clip_train` (pretraining loop on webdataset shards). Used by `planktonzilla/clip_model.py:ClipClassifier` for visual towers and by `scripts/train_clip.sh` for full pretraining via `-m open_clip_train.main`.
- Contains: `open_clip/src/open_clip/`, `open_clip/src/open_clip_train/`, model JSON configs under `open_clip/src/open_clip/model_configs/`.
- Note: NOT declared in `pyproject.toml`. See ARCHITECTURE.md → Anti-Patterns.

**`configs/`:**
- Purpose: Hydra hierarchical configuration. Each subdirectory is a "config group" that can be selected via CLI (`pz_train model=resnet18`).
- Top-level orchestrators: `configs/train.yaml`, `configs/import_dataset.yaml`.
- Per-group dirs: `augmentation/`, `custom_loss/`, `dataset/`, `dataset_import/`, `debug/`, `experiment/`, `extras/`, `hparams_search/`, `hydra/`, `local/`, `model/`, `paths/`, `peft/`, `tracking/`, `training_arguments/`.

**`scripts/`:**
- Purpose: SLURM batch and helper bash scripts for HPC environments (Jean Zay H100, Grid5000). Not invoked by tests.
- Key files: `scripts/train.sh`, `scripts/train_clip.sh`, `scripts/push_dataset.sh`.

**`notebooks/`:**
- Purpose: Exploratory dataset construction, model loading, evaluation, paper-figure metrics. Some `.py` files here are SLURM-launched scripts, not interactive notebooks (`gen_planktonzilla.py`, `add_planktonzilla.py`, `push_planktonzilla.py`, `save_planktonzilla_for_clip.py`, `gen_planktonzilla_ood.py`).
- Key files: `notebooks/metrics_clip.ipynb`, `notebooks/metrics_paper.ipynb`, `notebooks/gen_planktonzilla.py`.

**`tests/`:**
- Purpose: Pytest suite — instantiation tests (always run) and full smoke training (skipped in GitHub CI via `skip_in_github_ci`).
- Key files: `tests/test_train.py`, `tests/test_datasets.py`, `tests/conftest.py`, `tests/shared.py`.

**`data/`:**
- Purpose: Local cache of raw and prepared dataset trees. Each `DatasetImporter` subclass writes to `data/<class_name_lower>_raw_download/` and `data/<class_name_lower>_imagefolder/` (`planktonzilla/dataset_import/dataset_importer.py:212-213`).
- Generated: Yes (via `pz_import_dataset action=import`).
- Committed: No (gitignored).

**`logs/`:**
- Purpose: Hydra per-run output trees. Pattern: `${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}_${oc.env:SLURM_JOB_ID,local}` (`configs/hydra/default.yaml`).
- Contains: `config_tree.log` (rendered Hydra config), `tags.log`, `exec_time.log`, Hydra's `.hydra/` snapshot, Trainer checkpoints, eval predictions.
- Subdirs: `train/`, `train_node/` (legacy), `import_dataset/`, `wandb/` (mirror).

**`wandb/`:**
- Purpose: Local W&B run files (used in `WANDB_MODE=offline`, set by `scripts/train.sh:23`). Each run lives under `wandb/run-<YYYYMMDD_HHMMSS>-<id>/{files,logs,tmp}`.

**`docs/`:**
- Purpose: README assets (currently only `docs/images/planktonzilla-logo.gif`). No generated docs site.

**`.devcontainer/`:**
- Purpose: VS Code dev container based on `nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04` with Python 3.11, Poetry, zsh, oh-my-zsh, NVIDIA cutlass.

**`.github/`:**
- Purpose: Currently only houses `.github/copilot-instructions.md` (AI-agent project rules). No GitHub Actions workflows present.

## Key File Locations

**Read these first to understand the project:**
1. `README.md` — quick start, supported datasets, loss list.
2. `pyproject.toml` — dependency truth + `[project.scripts]` console entries.
3. `planktonzilla/train.py` — full training pipeline.
4. `configs/train.yaml` — defaults list shows the composition surface.
5. `planktonzilla/dataset.py` — `DatasetWrapper` contract.
6. `planktonzilla/loss.py` — `AbstractHFLoss` + concrete losses.
7. `planktonzilla/dataset_import/dataset_importer.py` — ETL template-method base.
8. `.github/copilot-instructions.md` — concise AI-agent contributor rules.

**Entry points:**
- `planktonzilla/train.py` — `pz_train` console script (`@hydra.main` config_name=`train.yaml`).
- `planktonzilla/dataset_import/import_dataset.py` — `pz_import_dataset` console script (`@hydra.main` config_name=`import_dataset.yaml`).
- `pyproject.toml` `[project.scripts]` — also declares `pz_prepare_train` (→ `planktonzilla.prepare_train:main`, **module not present**) and `pz_push_model` (→ `planktonzilla.push_model:main`, **module not present**).

**Configuration:**
- `configs/train.yaml` — training defaults list and global toggles.
- `configs/import_dataset.yaml` — import defaults list and `action` selector.
- `configs/paths/default.yaml` — path interpolations (`root_dir`, `data_dir`, `log_dir`, `output_dir`).
- `configs/hydra/default.yaml` — output-dir pattern with timestamp + SLURM ID.
- `configs/hydra/launcher/jeanzay_submitit_h100mono.yaml` — submitit launcher for Jean Zay H100.

**Core logic:**
- `planktonzilla/train.py:117-302` — `train(cfg)` orchestration.
- `planktonzilla/train.py:305-313` — `main(cfg)` Hydra entry.
- `planktonzilla/dataset.py:96-195` — `DatasetWrapper.prepare_datasets`.
- `planktonzilla/dataset_import/dataset_importer.py:163-411` — `DatasetImporter` base + `import_dataset()` orchestration.
- `planktonzilla/loss.py:13-24` — `AbstractHFLoss` contract.
- `planktonzilla/clip_model.py:5-59` — `ClipClassifier` adapter.

**Testing:**
- `tests/conftest.py` — `hydra_conf_path` fixture pointing to `../configs`.
- `tests/shared.py` — `model_names = ["resnet18"]`, `dataset_names = ["lensless"]`, `skip_in_github_ci` decorator.
- `tests/test_train.py` — full Hydra-composed `train(cfg)` smoke matrix.
- `tests/test_datasets.py` — `DatasetWrapper` instantiation parametrized on `dataset_names`.

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` — e.g. `dataset_importer.py`, `rich_utils.py`, `clip_model.py`.
- Hydra config files: `lower_snake_or_kebab.yaml`, often hyphenated to mirror HF model IDs — e.g. `vit-base-clip-224-openai.yaml`, `eva02-large-clip-224-2b-s4b-b131k.yaml`, `timm-eva02-large-m38m.yaml`.
- Test files: `test_<area>.py` — e.g. `test_train.py`, `test_datasets.py`.
- SLURM/Bash scripts: `<verb>_<noun>.sh` — e.g. `train.sh`, `train_clip.sh`, `push_dataset.sh`, `push_planktonzilla.sh`, `save_plankt.sh`, `save_plankt_plus.sh`.
- Notebooks: `<verb>_<noun>.ipynb` — `metrics_clip.ipynb`, `metrics_paper.ipynb`, `fix_taxo.ipynb`, `gen_datasets.ipynb`, `load_models.ipynb`. Some `.py` files in `notebooks/` are SLURM-driven scripts not Jupyter exports.
- Log files: `<task>.log` at repo root for the most recent CLI run; per-run trees under `logs/<task>/runs/<timestamp>_<slurm>/`.

**Directories:**
- Hydra config groups are pluralised conceptual nouns: `model/`, `dataset/`, `augmentation/`, `tracking/`. Where the group has a single representative the dir still exists (`paths/`, `extras/`, `hparams_search/`).
- Local data caches: `data/<importer_class_name_lowercased>_<raw_download|imagefolder>/` (`planktonzilla/dataset_import/dataset_importer.py:212-213`).
- Hydra output runs: `logs/<task_name>/runs/<YYYY-MM-DD>_<HH-MM-SS>_<SLURM_JOB_ID|local>/`.
- W&B runs: `wandb/run-<YYYYMMDD_HHMMSS>-<8char_id>/`.

**Code style (per `pyproject.toml [tool.ruff]`):**
- `line-length = 128`.
- Lints enabled: `F`, `E`, `W`, `I`, `N`, `NPY`, `PERF`, `FURB`, `PD`, `RUF`.
- Ignored: `E402`, `N806`. Notebooks excluded from ruff.
- Functions/variables: `snake_case`. Classes: `PascalCase` (e.g. `DatasetWrapper`, `ClipClassifier`, `AbstractHFLoss`, `ISIISNetDatasetImporter`).

**Hydra header conventions:**
- Group YAMLs that override the global config use `# @package _global_` (see `configs/model/default.yaml:1`, `configs/custom_loss/default.yaml:1`, `configs/augmentation/default.yaml:1`, `configs/experiment/default.yaml:1`, `configs/debug/default.yaml:1`, `configs/hparams_search/optuna.yaml:1`).
- Dataset YAMLs do NOT use `_global_`; they nest under `dataset:` automatically (see `configs/dataset/default.yaml`).
- Each config group has a `default.yaml` providing the baseline; specialised YAMLs `defaults: [default.yaml]` and override fields.
- HuggingFace `_target_` strings are the canonical instantiation handle: `transformers.AutoModelForImageClassification.from_pretrained`, `planktonzilla.clip_model.ClipClassifier`, `planktonzilla.loss.FocalLoss`, `transformers.TrainingArguments`, `torchvision.transforms.v2.Compose`, `peft.LoraConfig`, `planktonzilla.dataset_import.dataset_importer.<Subclass>`.
- Required-but-unset values use the OmegaConf sentinel `???` (e.g. `configs/dataset/default.yaml`, `configs/model/default.yaml`, `configs/custom_loss/ldam.yaml`).

## Where to Add New Code

**New plankton dataset (downloadable):**
1. Add a subclass to `planktonzilla/dataset_import/dataset_importer.py` overriding `_prepare_imagefolder` (and optionally `_download_and_extract`).
2. Create `configs/dataset_import/<dataset>.yaml` with `_target_: planktonzilla.dataset_import.dataset_importer.<NewClass>`, `hf_dataset_name`, `download_uris`, `human_readable_name`, citations.
3. Once published to HF Hub, add `configs/dataset/<dataset>.yaml` (extending `default.yaml`) for training-time access — set `name: project-oceania/<dataset>` and a `transform` (typically `ToTensor → Resize → Normalize` with dataset-specific mean/std).
4. Run `poetry run pz_import_dataset dataset_import=<dataset> action=import`.
5. Smoke train: `poetry run pz_train dataset=<dataset> training_arguments=test_minirun`.

**New model architecture:**
- Pure HuggingFace: add `configs/model/<name>.yaml` extending `configs/model/default.yaml` with `model._args_: [<hf_repo_id>]` and `img_size`.
- Pure CLIP: extend `configs/model/default_clip.yaml`, set `model._args_` to `[<open_clip_arch>, <open_clip_pretrained_tag>]`, set `num_features`. Example: `configs/model/eva02-large-clip-224-2b-s4b-b131k.yaml`.
- Custom architecture: add a new module under `planktonzilla/` exposing an HF-compatible `forward(pixel_values, labels=None, ...) -> ImageClassifierOutput` (mirror `planktonzilla/clip_model.py:ClipClassifier`); reference it from a new `configs/model/<name>.yaml` via `_target_`.

**New loss function:**
1. Implement subclass of `AbstractHFLoss` in `planktonzilla/loss.py` with `forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs) -> Tensor`.
2. Add `configs/custom_loss/<name>.yaml` with `# @package _global_` header, `custom_loss._target_: planktonzilla.loss.<NewLoss>` and any constructor args.
3. If the loss requires `cls_num_list`, declare it as a parameter (auto-injected by `train.train` via try/except, see `planktonzilla/train.py:217-224`). Use `???` sentinel in the YAML to make the dependency explicit (mirror `configs/custom_loss/ldam.yaml`).

**New augmentation strategy:**
- Add `configs/augmentation/<name>.yaml` with `# @package _global_` and `augmentation` defined as a `torchvision.transforms.v2.Compose` (or any callable). The training pipeline applies it AFTER the dataset-level `transform` (`planktonzilla/dataset.py:18-38`).

**New PEFT (LoRA) recipe:**
- Add `configs/peft/<model>.yaml` with one or more named adapter blocks (`<adapter_name>: { _target_: peft.LoraConfig, ... }`). Mirror `configs/peft/vit-base.yaml`. Activated by passing `peft=<model>` on the CLI.

**New experiment preset:**
- Add `configs/experiment/<name>.yaml` with `# @package _global_` and an `override /<group>: <option>` defaults list (mirror `configs/experiment/base_zoolake.yaml`).

**New tests:**
- Place under `tests/` as `test_<area>.py`. Use the `hydra_conf_path` fixture from `tests/conftest.py` and `Hydra compose(config_name="train", overrides=[...])`. Decorate any heavy test with `@skip_in_github_ci` from `tests/shared.py`.

**Utilities (logging, Hydra, Rich):**
- Place under `planktonzilla/utils/`. Always obtain a logger with `get_pylogger(__name__)`.

## Special Directories

**`data/`:**
- Purpose: Local raw + prepared dataset cache, written by `DatasetImporter.import_dataset()`. Folder names are `<DatasetImporterClassName>.lower() + "_raw_download"` and `..._imagefolder`.
- Generated: Yes.
- Committed: No (gitignored; large).

**`logs/`:**
- Purpose: Hydra per-task per-run output trees.
- Generated: Yes (every `pz_train`/`pz_import_dataset` run).
- Committed: No.

**`wandb/`:**
- Purpose: Local W&B artifacts (offline-mode-friendly).
- Generated: Yes.
- Committed: No.

**`open_clip/`:**
- Purpose: Vendored library source (NOT a pip dependency).
- Generated: No (manually vendored).
- Committed: Yes.

**`planktonzilla/dataset_import/public_data/`:**
- Purpose: Holds `lensless_dataset.zip` so the lensless dataset can be reproduced offline (`LenslessDatasetImporter._download_and_extract` reads it via `public_data.__path__[0]`, `planktonzilla/dataset_import/dataset_importer.py:414-427`).
- Generated: No.
- Committed: Yes.

**`configs/local/`:**
- Purpose: Opt-in machine-/user-specific overrides loaded via `defaults: [optional local: default.yaml]` in `configs/train.yaml`.
- Generated: No.
- Committed: Only `.gitkeep`; user files excluded.

**`configs/training_arguments/accelerate/`:**
- Purpose: Empty. Reserved/legacy folder (no YAML files yet).
- Generated: No.
- Committed: Empty directory only.

**`.planning/`:**
- Purpose: GSD workflow state (codebase maps written here).
- Generated: Yes (by `/gsd-map-codebase`).
- Committed: Tracked but managed by GSD tooling.

---

*Structure analysis: 2026-05-12*
