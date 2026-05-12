# Testing Patterns

**Analysis Date:** 2026-05-12

## Test Framework

**Runner:**
- **pytest** (declared in `pyproject.toml` under `[dependency-groups] dev = ["pytest", "notebook", "ipywidgets", "pytest-cov", "ipykernel"]`).
- Pytest 8.x / 9.x are both seen in the cached bytecode (`tests/__pycache__/*-pytest-8.4.1.pyc`, `*-pytest-8.4.2.pyc`, `*-pytest-9.0.2.pyc`), so tests are expected to work across both major versions.
- Configured under `[tool.pytest.ini_options]` in `pyproject.toml`:
  ```toml
  addopts = "--doctest-modules --log-cli-format='%(asctime)s [%(levelname)7s] %(message)s (%(filename)s:%(lineno)s)' --log-cli-date-format='%Y-%m-%d %H:%M:%S'"
  junit_family = "xunit2"
  testpaths = ["tests"]
  log_cli = true
  log_cli_level = "INFO"
  norecursedirs = ".venv"
  ```
  Notable consequences:
  - `--doctest-modules` means **every importable module is scanned for doctests**, including production code under `planktonzilla/`. No module currently ships runnable doctests, but adding any `>>>` block to a docstring will be executed by the suite.
  - `log_cli = true` + `log_cli_level = "INFO"` makes pytest stream the standard-library logger (i.e. anything emitted via `planktonzilla.utils.logger.get_pylogger`) to the terminal during test runs.
  - `junit_family = "xunit2"` is set for CI compatibility, but no `--junitxml` path is wired into `addopts`; CI must pass that explicitly.
  - `testpaths = ["tests"]` and `norecursedirs = ".venv"` keep discovery scoped.

**Assertion Library:**
- Plain `assert` statements (pytest's rewritten asserts). No `unittest.TestCase`, no `pytest-mock`, no `hypothesis`.

**Run Commands:**
```bash
poetry run pytest                                  # run the full suite (tests/ + module doctests)
poetry run pytest tests/test_datasets.py           # one file
poetry run pytest tests/test_train.py::test_training -k "lensless and resnet18"  # one parametrized case
poetry run pytest --cov=planktonzilla --cov-report=term-missing                  # coverage (pytest-cov is in dev deps)
GITHUB_ACTIONS=true poetry run pytest              # exercise the CI skip path locally
```

The repo's contributor guide (`.github/copilot-instructions.md`) explicitly tells agents to run `poetry run pytest` before proposing changes.

## Test File Organization

**Location:**
- All tests live in a single top-level `tests/` directory (no co-located tests next to source).
- Files at this date:
  ```
  tests/
  ├── __init__.py        # only contains `"""(c) Inria"""`
  ├── conftest.py        # 1 fixture: hydra_conf_path
  ├── shared.py          # parametrize lists + skip_in_github_ci decorator
  ├── test_datasets.py   # dataset instantiation + prepare_datasets
  └── test_train.py      # end-to-end training smoke + custom-loss matrix
  ```
- The `__pycache__` directory shows two stale test modules that no longer exist on disk: `test_models.cpython-313-pytest-8.4.1.pyc` and `test_import_dataset.cpython-313-pytest-8.4.1.pyc`. These were deleted from source but their compiled artifacts remain — see CONCERNS.md.

**Naming:**
- `test_*.py` files, `test_*` functions. Pytest defaults; no overrides in `pyproject.toml`.

**Structure:**
```
tests/
  conftest.py     # shared pytest fixtures (project-wide)
  shared.py       # plain Python module imported via `from .shared import ...`
  test_*.py       # one file per high-level concern (datasets, training)
```

## Test Structure

**Shared bootstrap pattern.** Every test module that needs Hydra also runs `pyrootutils.setup_root(...)` at the top so the project root is on `sys.path` and `.env` is loaded:

```python
# tests/test_train.py
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from planktonzilla.train import train
from .shared import dataset_names, model_names, skip_in_github_ci
```

**Suite Organization:** flat module-level test functions, no test classes. Example from `tests/test_datasets.py`:

```python
@pytest.mark.parametrize("dataset_name", dataset_names)
def test_dataset_instantiation(hydra_conf_path: Path, dataset_name: str):
    with initialize(version_base=None, config_path=hydra_conf_path):
        cfg: DictConfig = compose(
            config_name="train",
            overrides=[f"dataset={dataset_name}",
                       "extras.print_config=False",
                       "extras.enforce_tags=False"],
        )
        dataset: DatasetWrapper = instantiate(cfg.dataset)
        assert dataset
```

**Patterns:**
- Setup: `hydra.initialize(version_base=None, config_path=hydra_conf_path)` + `hydra.compose(config_name="train", overrides=[...])`. The `extras.print_config=False` and `extras.enforce_tags=False` overrides are applied in every test to suppress the interactive tag prompt and the Rich config tree.
- Teardown: handled by the `with initialize(...)` and `with TemporaryDirectory()` context managers — no explicit teardown.
- Assertion: minimal, mostly truthiness (`assert dataset`, `assert metric_dict`). The suite checks "does training complete and return metrics" rather than asserting exact metric values.

**Parametrization** is the central pattern — driven by lists in `tests/shared.py`:

```python
# tests/shared.py
model_names = ["resnet18"]
dataset_names = ["lensless"]      # whoi-plankton is commented out
```

and in `tests/test_train.py`:

```python
losses = ["asymmetric", "balanced_meta_softmax", "focal", "ldam", "max_margin", "ral"]

@skip_in_github_ci
@pytest.mark.parametrize("dataset_name", dataset_names)
@pytest.mark.parametrize("model_name", model_names)
@pytest.mark.parametrize("custom_loss", losses)
def test_training_custom_losses(...):
```

This produces the cartesian product `1 dataset × 1 model × 6 losses = 6 runs` for the loss matrix, and `1 × 1 = 1` run for the baseline `test_training`.

## Mocking

- **No mocking framework is used.** No `unittest.mock`, no `pytest-mock`, no `monkeypatch`, no `responses`, no fakes.
- The training tests do not mock external services; they sidestep them by setting config flags:
  ```python
  cfg.training_arguments.output_dir = tmp_dir
  cfg.paths.output_dir = tmp_dir
  cfg.tracking.use_wandb = False
  cfg.tracking.use_mlflow = False
  cfg.tracking.use_trackio = False
  cfg.model_push_to_hub = False
  ```
  i.e. real HuggingFace Hub access is still required to pull the dataset and base model, but no logging/pushing happens.
- The `lensless` dataset is intentionally chosen because the zip ships in-tree (`planktonzilla/dataset_import/public_data/lensless_dataset.zip`), so most test runs do not need the network.

**What to Mock:**
- Currently nothing. New tests that touch HuggingFace Hub, W&B, MLflow or trackio should prefer `monkeypatch.setenv` + `cfg.tracking.use_*=False` over real network calls.

**What NOT to Mock:**
- Hydra config composition itself — tests deliberately exercise the real Hydra graph to catch config drift.

## Fixtures

- One project-wide fixture in `tests/conftest.py`:
  ```python
  @pytest.fixture()
  def hydra_conf_path():
      return "./../configs"
  ```
  This path is **relative to the test file directory**, which is what `hydra.initialize(config_path=...)` requires.
- No factory functions, no `tmp_path` usage (tests use the stdlib `tempfile.TemporaryDirectory()` context manager directly), no autouse fixtures, no session-scoped fixtures.

## Helpers / Skip Logic

`tests/shared.py` defines `skip_in_github_ci`, applied to any test that is too heavy for GitHub Actions:

```python
def skip_in_github_ci(func):
    @wraps(func)
    def run_if_not_gh_ci(*args, **kwargs):
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("Test skipped on CI to avoid disk full/time-out errors.")
        return func(*args, **kwargs)
    return run_if_not_gh_ci
```

It is currently applied to:
- `tests/test_datasets.py::test_dataset_prepare_datasets`
- `tests/test_train.py::test_training`
- `tests/test_train.py::test_training_custom_losses`

`test_dataset_instantiation` is the **only test that runs unconditionally on GitHub CI** (and even that requires Hydra configs and importable transformers/timm, so it's not zero-network).

## Coverage

- **`pytest-cov` is declared as a dev dependency** in `pyproject.toml`'s `[dependency-groups] dev`, but no `[tool.coverage]` block is configured and no `--cov` flag lives in `addopts`.
- No coverage threshold is enforced; no `.coveragerc`, no Codecov config.
- To produce a report locally:
  ```bash
  poetry run pytest --cov=planktonzilla --cov-report=term-missing
  poetry run pytest --cov=planktonzilla --cov-report=html        # writes htmlcov/
  ```

## Test Types

**Unit Tests:**
- `tests/test_datasets.py::test_dataset_instantiation` — pure config + dataclass instantiation, no I/O. The closest thing to a real unit test in the repo.

**Integration Tests:**
- `tests/test_datasets.py::test_dataset_prepare_datasets` — calls `DatasetWrapper.prepare_datasets(None)`, which invokes `datasets.load_dataset`. Real HuggingFace Hub access, skipped on CI.
- `tests/test_train.py::test_training` — full Hydra compose + `train(cfg)` smoke run with `training_arguments=test_minirun` (`max_steps: 2`, `per_device_train_batch_size: 2`). End-to-end "does it return metrics" assertion.
- `tests/test_train.py::test_training_custom_losses` — same, parametrized across all 6 custom-loss configs.

**E2E Tests:**
- Not separated. The `test_training*` tests *are* E2E in spirit but they're marked CI-skipped, so they only run on developer machines / GPU boxes with HuggingFace credentials.

## Common Patterns

**Async Testing:**
- None. No `pytest-asyncio` and no `async def` test functions, despite `aiohttp` being used inside `DatasetImporter._download_and_extract`.

**Error Testing:**
- None. The suite contains zero `pytest.raises(...)` blocks. Defensive `raise ValueError`/`RuntimeError` paths in `planktonzilla/dataset_import/dataset_importer.py` (`_validate`, `_prepare_imagefolder` not implemented, missing `objects.tsv.gz`) are entirely untested.

**Temp directories:**
- Use `tempfile.TemporaryDirectory()` rather than the pytest `tmp_path`/`tmp_path_factory` fixtures. Both are valid; for new tests, `tmp_path` is slightly more idiomatic and gives nicer per-test cleanup semantics.

## Honest Assessment of Coverage Gaps

**What is tested:**
- DatasetWrapper instantiation through Hydra (`tests/test_datasets.py`).
- Full training loop end-to-end on a tiny `max_steps=2` budget (`tests/test_train.py`), with all 6 custom losses exercised at least once.
- The Hydra config graph itself (every `compose(config_name="train", ...)` would fail loudly if the YAML defaults under `configs/` got out of sync).

**What is NOT tested (high-priority gaps):**
1. **Dataset importers.** The 12+ subclasses in `planktonzilla/dataset_import/dataset_importer.py` (`LenslessDatasetImporter`, `ZooLakeDatasetImporter`, `ZooScanNetDatasetImporter`, `WHOIPlanktonDatasetImporter`, `JEDISystemsOceansCPICSDatasetImporter`, `UVP6NetDatasetImporter`, `ZooCAMNetDatasetImporter`, `FlowCAMNetDatasetImporter`, `ISIISNetDatasetImporter`, `PlanktoScopeDatasetImporter`, `GlobalUVP5NetDatasetImporter`, `PlanktonSet1DatasetImporter`, `SYKEIFCB2022DatasetImporter`, `SYKEZooScan2024DatasetImporter`) have no tests. The compiled `tests/__pycache__/test_import_dataset.cpython-313-pytest-8.4.1.pyc` suggests there *was* a `test_import_dataset.py` that has since been removed.
2. **Loss math.** `planktonzilla/loss.py` (FocalLoss, LDAMLoss, MaximumMarginLoss, AsymmetricLoss, RobustAsymmetricLoss, BalancedMetaSoftmaxLoss, CrossEntropyLossHF) is exercised only via `test_training_custom_losses`, which checks training completes — not numerical correctness.
3. **`ClipClassifier`** in `planktonzilla/clip_model.py` has no test (the `try/except` model-instantiation fallback in `train.py:158` toward `ClipClassifier` is never hit in CI).
4. **Utilities.** `planktonzilla/utils/hydra.py` (`task_wrapper`, `extras`, `log_hyperparameters`, `get_metric_value`, `close_loggers`) and `planktonzilla/utils/rich_utils.py` (`print_config_tree`, `enforce_tags`) are untested in isolation.
5. **CLI entrypoints.** `pz_train`, `pz_import_dataset` console scripts are not invoked by any test. The two declared scripts `pz_prepare_train` and `pz_push_model` reference modules that don't exist on disk (`planktonzilla/prepare_train.py`, `planktonzilla/push_model.py`) — a smoke test would catch this immediately.
6. **Failure paths.** Zero `pytest.raises(...)` calls; the explicit `raise ValueError`/`RuntimeError` in `DatasetImporter._validate` and `GlobalUVP5NetDatasetImporter._prepare_imagefolder` are uncovered.
7. **The `--doctest-modules` flag** is on but no module ships doctests, so it currently buys nothing and slightly slows the suite.

**Practical implication for new work:**
- For pure-Python additions (loss functions, metrics, helpers), prefer adding focused unit tests in a new `tests/test_<area>.py` module rather than extending the existing E2E `test_training` parametrize lists — the smoke runs cost real GPU/data time.
- Tests that need dataset access should use the in-tree `lensless` dataset (`planktonzilla/dataset_import/public_data/lensless_dataset.zip`) or `monkeypatch` the `datasets.load_dataset` call.
- New tests that should run on GitHub Actions must avoid HuggingFace Hub network calls; otherwise wrap them with `@skip_in_github_ci` from `tests/shared.py`.

---

*Testing analysis: 2026-05-12*
