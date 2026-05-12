# Coding Conventions

**Analysis Date:** 2026-05-12

## Naming Patterns

**Files:**
- Source modules: `snake_case.py` (e.g. `clip_model.py`, `dataset_importer.py`, `import_dataset.py`).
- Subpackages are flat snake_case directories with an `__init__.py` (e.g. `planktonzilla/utils/`, `planktonzilla/dataset_import/`).
- Test files: `test_*.py` (e.g. `tests/test_train.py`, `tests/test_datasets.py`). Pytest's default discovery is used (no custom `python_files` pattern set in `pyproject.toml`).
- Hydra YAML configs: lowercase with underscores or hyphens, organized by group under `configs/<group>/<name>.yaml` (e.g. `configs/training_arguments/test_minirun.yaml`, `configs/custom_loss/focal.yaml`).
- Shell scripts: `snake_case.sh` under `scripts/` (e.g. `scripts/train.sh`).

**Functions:**
- `snake_case` everywhere (`compute_metrics`, `validate_environment`, `compute_mean_and_std_dev`, `get_pylogger`, `task_wrapper`, `print_config_tree`).
- Private helpers are prefixed with `_` (e.g. `DatasetImporter._validate`, `_download_and_extract`, `_prepare_imagefolder`, `_push_to_hub`).

**Variables:**
- `snake_case` for locals and module-level (`dataset_wrapper`, `train_metrics`, `report_to`, `cls_num_list`).
- Loop short names (`x`, `f`, `z`, `dm`) are accepted in tight numeric/file-IO blocks; ruff `N806` (variable-in-function-should-be-lowercase) is **explicitly disabled** in `pyproject.toml` (`lint.ignore = ["E402", "N806"]`), so capitalised tensor names like `F.cross_entropy` callers using `X`, `LR` etc. are tolerated.
- Constants on dataclass subclasses use `UPPER_SNAKE_CASE` and are typed as `Final` / `ClassVar` (e.g. `LenslessDatasetImporter.DATASET_FILENAME: Final[str]`, `ZooLakeDatasetImporter.SPLIT_NAMES: ClassVar[Dict[str, str]]`, `GlobalUVP5NetDatasetImporter.OBJECTS_URL`).

**Types / Classes:**
- `PascalCase` (`DatasetWrapper`, `ClipClassifier`, `AbstractHFLoss`, `FocalLoss`, `LDAMLoss`, `DatasetImporter`, `ISIISNetDatasetImporter`).
- Loss classes inherit from `AbstractHFLoss(nn.Module)` (`planktonzilla/loss.py`), so all losses share the signature `forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs)`.
- Dataset importers inherit from `DatasetImporter` (`planktonzilla/dataset_import/dataset_importer.py`) and override `_prepare_imagefolder` (and sometimes `_download_and_extract`).

## Code Style

**Formatting:**
- **Ruff** is the only formatter/linter (no Black, isort, or mypy configured). The project copilot guide (`.github/copilot-instructions.md`) tells contributors to run `poetry run ruff format` and `poetry run ruff check`.
- `[tool.ruff]` in `pyproject.toml`:
  - `line-length = 128` (much wider than the PEP 8 default of 79).
  - `exclude = ["notebooks/"]` — Jupyter notebooks are not linted.

**Linting (`pyproject.toml [tool.ruff.lint]`):**
- `select = ["F", "E", "W", "I", "N", "NPY", "PERF", "FURB", "PD", "RUF"]`
  - `F` Pyflakes, `E`/`W` pycodestyle, `I` isort-style import ordering, `N` pep8-naming,
    `NPY` NumPy-specific, `PERF` Perflint, `FURB` refurb modernizations, `PD` pandas-vet,
    `RUF` Ruff-specific rules.
- `ignore = ["E402", "N806"]`:
  - `E402` (module-level import not at top) is disabled because `planktonzilla/train.py`, `planktonzilla/dataset_import/import_dataset.py` and `tests/test_train.py` deliberately call `pyrootutils.setup_root(...)` **before** the rest of their imports.
  - `N806` (uppercase variable in function) is disabled to match the ML/torch convention (`F.cross_entropy`, capital tensor names, etc.).
- Per-file `# noqa` is used only where strictly needed: `import torch.nn.functional as F  # noqa: N812` in `planktonzilla/loss.py:8`.

**No mypy / no pre-commit / no black / no isort:**
- No `.pre-commit-config.yaml`, no `.editorconfig`, no `mypy.ini`, no `setup.cfg`, no `[tool.mypy]`, `[tool.black]` or `[tool.isort]` blocks anywhere. Type checking is opportunistic, not enforced.
- `.vscode/settings.json` only configures pytest discovery; it does not pin a formatter.

## Import Organization

Ordering matches Ruff's `I` (isort) convention. A typical example from `planktonzilla/train.py`:

```python
"""(c) Inria"""

# 1) Bootstrap project root BEFORE other imports (so E402 is intentionally ignored).
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# 2) Stdlib
import os
from functools import partial

# 3) Third-party scientific / framework
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments, set_seed

# 4) First-party
from planktonzilla.clip_model import ClipClassifier
from planktonzilla.dataset import DatasetWrapper
from planktonzilla.utils.hydra import get_metric_value, task_wrapper
from planktonzilla.utils.logger import get_pylogger
```

**Path Aliases:**
- None. Absolute imports rooted at `planktonzilla.` are used everywhere; `pyrootutils.setup_root(..., pythonpath=True)` adds the repo root to `sys.path` so this works whether scripts are run via `poetry run pz_train` or `python planktonzilla/train.py`.
- `planktonzilla/train.py` currently imports `numpy as np` twice (lines 21 and 30) — a real duplicate import that ruff `F811` would catch if it weren't shadowed by `select` order; treat this as a known smell, not the convention.

## Configuration Conventions

**Hydra-first.** All runtime configuration lives under `configs/` and is composed by **Hydra 1.3+** (`hydra-core`, with `hydra-colorlog` and `hydra-submitit-launcher`).

- Top-level entry configs: `configs/train.yaml`, `configs/import_dataset.yaml`.
- Group directories: `configs/{model,dataset,training_arguments,paths,extras,hydra,augmentation,tracking,custom_loss,peft,experiment,hparams_search,debug,local}/`.
- Entry points use the canonical pattern:
  ```python
  @hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
  def main(cfg: DictConfig) -> float | None: ...
  ```
- `OmegaConf.register_new_resolver("eval", eval)` is registered (best-effort, wrapped in `try/except ValueError`) at import time in both `planktonzilla/train.py` and `planktonzilla/dataset_import/import_dataset.py` to allow `${eval:...}` interpolations.
- An optional `local` config (`configs/local/default.yaml`) is loaded with `optional local: default.yaml` for per-machine overrides; it is **not** committed by convention (excluded from version control).
- New experiments are added as YAML files under `configs/experiment/` rather than by editing defaults — see `.github/copilot-instructions.md`.

**CLI conventions:**
- No `argparse`, `click`, or `typer` anywhere in `planktonzilla/`. CLIs are exposed exclusively through Hydra-decorated `main()` functions.
- Console scripts are declared in `pyproject.toml [project.scripts]`:
  - `pz_import_dataset = "planktonzilla.dataset_import.import_dataset:main"`
  - `pz_prepare_train  = "planktonzilla.prepare_train:main"`  *(declared but `planktonzilla/prepare_train.py` is missing — see CONCERNS.md)*
  - `pz_train          = "planktonzilla.train:main"`
  - `pz_push_model     = "planktonzilla.push_model:main"`     *(declared but `planktonzilla/push_model.py` is missing — see CONCERNS.md)*
- Preferred invocation is via Poetry, e.g. `poetry run pz_train dataset=isiisnet model=resnet18 training_arguments.num_train_epochs=5`.

## Type Hints

- Type hints are **partial and inconsistent**, but present on most public function signatures.
- Modern PEP 604 syntax is used (`float | None`, `tuple[dict, dict]`, `list[int]`, `str | list[str]`) in newer files (`planktonzilla/train.py`, `planktonzilla/loss.py`).
- Older code in `planktonzilla/dataset_import/` still uses `typing.Optional`, `typing.Union`, `typing.Dict`, `typing.Final`, `typing.ClassVar` (e.g. `dataset_importer.py:15`).
- Function bodies are largely untyped; type hints are concentrated on the signature and on `@dataclass` fields.
- `Callable` is imported from `typing` in `planktonzilla/dataset.py` and from `collections.abc` in `planktonzilla/utils/hydra.py` — the codebase has not standardized on one.
- No `mypy`, `pyright`, or `ty` config; types are documentation, not gates.

## Docstring Style

- Module headers: every first-party `.py` file starts with a 3-line copyright docstring:
  ```python
  """
  (c) Inria
  """
  ```
- Function/class docstrings: short triple-quoted descriptions, optionally followed by Google-style `Args:` / `Returns:` sections. Examples:
  - `planktonzilla/train.py:118` (`train(cfg)`) — Google-style with `Args:` and `Returns:`.
  - `planktonzilla/dataset.py:41` (`compute_mean_and_std_dev`) — Google-style.
  - `planktonzilla/utils/rich_utils.py:25` (`print_config_tree`) — Google-style.
- Loss classes use a free-form *Source:* / *Note:* convention citing the original paper and the reference implementation (see `planktonzilla/loss.py:30`, `:85`, `:130`, `:236`, `:293`).
- Many internal helpers (especially in `planktonzilla/dataset_import/dataset_importer.py`) have **no docstring at all**; the dataclass fields are the only documentation.
- `pytest` is configured with `--doctest-modules` (`pyproject.toml [tool.pytest.ini_options]`), which means docstrings can technically be executed — but no module currently ships runnable doctests. New docstrings should not include `>>>` blocks unless they're verified to pass.

## Error Handling

**Patterns observed:**
- `raise ValueError(...)` for caller misuse (e.g. `DatasetImporter._validate` checks `push_to_hub`/`hf_token` consistency at `dataset_importer.py:201`).
- `raise RuntimeError(...)` for unexpected runtime/file-system failures (`dataset_importer.py:346`, `:665`, `:694`).
- `raise NotImplementedError(...)` on abstract hooks (`AbstractHFLoss.forward`, `DatasetImporter._prepare_imagefolder`).
- `raise Exception(...)` is used in one place (`utils/hydra.py:195`, `get_metric_value`) — too generic; prefer `KeyError` or a dedicated subclass when extending.
- `try/except` blocks are typically narrow (`ValueError`, `ImportError`, `OSError`, `FileNotFoundError`, `IOError`, `SyntaxError`).
- **Bare `except:` is used in `planktonzilla/train.py:167` and `planktonzilla/clip_model.py:39`** as a "fallback to alternative branch" pattern. Avoid extending this — see CONCERNS.md.
- `task_wrapper` (`utils/hydra.py:22`) wraps the whole training flow in a `try/except Exception` that calls `log.exception("")` so the stack trace is persisted to the per-run `.log` file, then re-raises.

## Logging

**How it is configured:**
- The application uses the **standard library `logging` module**, accessed through `planktonzilla.utils.logger.get_pylogger(__name__)` (`planktonzilla/utils/logger.py`). That helper is just `logging.getLogger(name)` — no custom formatter, level, or handler is set in Python code.
- Formatting and level come from **Hydra's `colorlog` plugin**: `configs/hydra/default.yaml` does
  ```yaml
  defaults:
    - override hydra_logging: colorlog
    - override job_logging: colorlog
  ```
  This is what produces the colored, structured lines seen in `train.log` (e.g. `[2026-01-01 15:15:17,210][planktonzilla.utils.hydra][INFO] - …`).
- W&B / MLflow / trackio are **not** the source of `train.log`; they are toggled via env vars set in `train.py` (`WANDB_*`, `MLFLOW_*`, `TRACKIO_*`) and only attach when the corresponding `cfg.tracking.use_*` flag is true.
- Per-run output goes under `${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}_${oc.env:SLURM_JOB_ID,local}` (see `configs/hydra/default.yaml`); multi-run sweeps use `multiruns/...`.

**Patterns:**
- Module-level logger: `log = get_pylogger(__name__)` at the top of every executable module. (`planktonzilla/dataset.py` and `planktonzilla/dataset_import/dataset_importer.py` use the name `logger` instead of `log` — minor inconsistency.)
- f-strings are used directly in log calls (`log.info(f"Instantiating ...")`) — convenient but not lazy. Acceptable here because tracing lifecycle is the main use.
- Status messages are heavily emoji-decorated (`✅`, `⚠️`, `🛑`) in `validate_environment` (`planktonzilla/train.py:53`).
- `log.exception("")` (with an empty message) is used inside `task_wrapper` to dump the full traceback into the per-run `.log` file before re-raising.

## Comments

**When to Comment:**
- Comments tend to mark *why*: explanations of Hydra-multirun gotchas, links to upstream docs, references to original papers.
- Large blocks of commented-out code remain (PyTorch Lightning callbacks/loggers in `planktonzilla/utils/hydra.py:96-131`, the alternative `compute_metrics` in `planktonzilla/train.py:93-99`). These should be deleted rather than left commented out — see CONCERNS.md.

**JSDoc / type-style annotations:**
- N/A (Python). Use Google-style docstrings as shown above.

## Function Design

- Module entry points (`main`, `train`, `import_dataset`) are large, linear procedures (300+ lines). Heavy use of `hydra.utils.instantiate(cfg.<group>)` rather than direct construction.
- Decorator pattern: long-running tasks are wrapped with `@task_wrapper` (defined in `planktonzilla/utils/hydra.py:22`) which manages timing, exception logging, and logger shutdown.
- Helper functions (`compute_metrics`, `compute_mean_and_std_dev`, `augment_and_transform_batch`) are small and pure-ish.
- Default-argument values are often `None` rather than sentinel objects; `__post_init__` does the real validation/normalization.

## Module Design

**Exports:**
- Implicit (no `__all__`); subpackage `__init__.py` files contain only the copyright docstring.
- Cross-module references go through fully qualified imports (`from planktonzilla.utils.hydra import task_wrapper`); barrel re-exports are not used.

**Barrel Files:**
- Not used. Each subpackage's `__init__.py` is intentionally empty (apart from the `(c) Inria` docstring).

## Editor / IDE Configuration

- `.vscode/settings.json` enables pytest, disables unittest. No formatter / linter pinned in VS Code config.
- `.devcontainer/` provides a Docker-based dev environment: `postStartCommand` runs `poetry install`; `remoteEnv` forwards `WANDB_API_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `HF_TOKEN` from the host.
- `poetry.toml` sets `[virtualenvs] in-project = true` so `.venv/` is created at the repo root.
- No `.pre-commit-config.yaml` and no `.editorconfig`. Style enforcement is opt-in via `poetry run ruff check` / `ruff format`.

---

*Convention analysis: 2026-05-12*
