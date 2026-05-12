# Codebase Concerns

**Analysis Date:** 2026-05-12

## Severity Legend

- **High** — actively breaks reproducibility, training, or trust in results; or creates a security/operational risk that needs prompt attention.
- **Medium** — meaningful debt that will bite during the next refactor, port, or new collaborator onboarding.
- **Low** — cosmetic / minor / cleanup; safe to defer.

## Tech Debt

### [High] Vendored `open_clip` is unpinned, undeclared, and has no upstream link

- **Files:** `open_clip/`, `open_clip/src/open_clip/version.py` (`__version__ = '4.0.0.dev0'`), `open_clip/src/open_clip_train/`, `pyproject.toml`, `planktonzilla/clip_model.py:2`
- **Issue:** A full copy of `open_clip` (24 modules under `open_clip/src/open_clip/` plus the `open_clip_train` training harness) is committed at the repo root and self-identifies as `4.0.0.dev0` (a development snapshot, not any released tag). It is **not declared as a dependency** anywhere in `pyproject.toml` or `poetry.lock` (only `torchvision`, `transformers`, `timm`, etc. are pinned), yet `planktonzilla/clip_model.py:2` does a hard `import open_clip`. There is no `README`, `VENDORED.md`, `git_sha.txt`, or commit pointer in `open_clip/` indicating which upstream commit/tag was vendored, nor any patch/diff file describing local modifications.
- **Impact:**
  - The package will only import when `open_clip/src/` happens to be on `PYTHONPATH`. `scripts/train_clip.sh:20` literally hardcodes `export PYTHONPATH=/home/acontreras/planktonzilla/open_clip/src:$PYTHONPATH` — a path on a single user's account on Jean Zay. Anyone else reproducing CLIP training silently gets the wrong (or no) `open_clip`.
  - Cannot tell whether local edits were made versus upstream — drift detection is impossible.
  - Security/freshness: a `4.0.0.dev0` snapshot will not receive upstream bug or model-config fixes.
- **Fix approach (in order of effort):**
  1. Short term — record the upstream provenance: add `open_clip/UPSTREAM.md` with the git commit SHA + URL of `mlfoundations/open_clip` it was copied from, and run `git diff` against that SHA so any local patches are explicit.
  2. Medium term — make the import path deterministic: either ship `open_clip` as a Poetry path dependency (`open-clip = { path = "open_clip", develop = true }`) or replace the vendored tree with a normal pinned dep (`open-clip-torch = "X.Y.Z"`) if no local modifications are required.
  3. Remove `export PYTHONPATH=/home/acontreras/...` from `scripts/train_clip.sh`.

### [High] `pyproject.toml` declares a non-existent `transformers` major version

- **Files:** `pyproject.toml:25` (`transformers = {version = "^5.3.0", extras = ["sentencepiece"]}`), `poetry.lock:5874-5876` (resolves to `transformers 4.57.3`)
- **Issue:** `^5.3.0` requires `>=5.3.0,<6.0.0` but no `transformers` 5.x exists on PyPI; the lock file ended up at `4.57.3`. Either the lock is stale relative to `pyproject.toml`, or the constraint is wrong.
- **Impact:** A clean `poetry lock --no-update && poetry install` from scratch will fail to satisfy the constraint and refuse to install. New contributors / CI runners cannot bootstrap the env.
- **Fix approach:** Pin to the actually-supported range, e.g. `transformers = {version = "^4.57.0", extras = ["sentencepiece"]}`, then `poetry lock`. Confirm the same for `wandb = "^0.22.2"` (lock is `0.22.3`, OK) and `datasets = "^4.4.0"` (lock is `4.4.2`, OK).

### [High] Bare `except:` swallows control-flow errors at model selection

- **Files:** `planktonzilla/train.py:159-176` (the `try: AutoModelForImageClassification ... except: ClipClassifier ...` block) and `planktonzilla/clip_model.py:33-44` (`try: _ = self.model.proj ... except: ...`)
- **Issue:** A naked `except:` (no exception class) catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`, and any silent typo (`AttributeError`, `ImportError`) and proceeds to a fallback path. In `train.py` it determines whether the model is HF-AutoModel vs `ClipClassifier`; in `clip_model.py` it decides between `nn.Sequential` and `model.trunk` branching. A wrong branch produces a model that may train but predict garbage.
- **Impact:** Hard-to-diagnose silent failures (model architecture chosen by exception path), e.g. a Hydra typo in `cfg.model._args_` would route to `ClipClassifier` instead of erroring.
- **Fix approach:** Catch the specific exceptions Hydra/transformers can raise (`hydra.errors.InstantiationException`, `OSError`, `ValueError`) and log at `WARNING` which fallback was taken, e.g.
  ```python
  except (hydra.errors.InstantiationException, OSError) as e:
      log.warning(f"Falling back to ClipClassifier: {e}")
      model = hydra.utils.instantiate(...)
  ```

### [High] Custom `FocalLoss` uses deprecated PyTorch APIs that will warn / break

- **Files:** `planktonzilla/loss.py:63` (`logpt = F.log_softmax(logits)` — missing `dim=` argument), `planktonzilla/loss.py:66, 72` (`Variable(...)` — autograd `Variable` was deprecated in PyTorch 0.4 ~2018)
- **Issue:**
  - `F.log_softmax(logits)` without `dim=` triggers a `UserWarning` in current PyTorch and may default-pick a different dim across versions.
  - `torch.autograd.Variable` is a deprecated no-op alias for `Tensor`.
- **Impact:** Spammy warnings in `train.log`; risk of silent semantic change if PyTorch removes the warning fallback.
- **Fix approach:** `logpt = F.log_softmax(logits, dim=-1)`; replace `Variable(x)` with `x.detach()` or just `x`.

### [Medium] Dead/commented-out code throughout core modules

- **Files:**
  - `planktonzilla/train.py:90-97` — old `compute_metrics` left commented above the new one.
  - `planktonzilla/train.py:139` — commented `hydra.utils.instantiate(cfg.torch_matmul_precision)`.
  - `planktonzilla/train.py:207` — commented `# training_args.hub_token = cfg.hf_token`.
  - `planktonzilla/dataset.py:24-31` — old `transform`/`image_processor` block left as comments inside the active `augment_and_transform_batch`.
  - `planktonzilla/dataset.py:147` — commented `self.test_data = load_dataset("vendimia50/ct_metadataset", ...)`.
  - `planktonzilla/utils/hydra.py:104-150` — large `instantiate_callbacks` / `instantiate_loggers` blocks commented out (Lightning leftovers).
  - `planktonzilla/utils/logger.py:11-22` — commented rank-zero decorator block.
  - `planktonzilla/utils/rich_utils.py:14-16` — commented Lightning imports.
- **Impact:** Obscures intent, lengthens code review, and rot increases the longer it stays.
- **Fix approach:** Delete; use `git log` / `git blame` if anything ever needs to come back.

### [Medium] `experimental.yaml` is mostly a giant comment-only reference file

- **Files:** `configs/training_arguments/experimental.yaml` (~280 lines, almost all commented out)
- **Issue:** Functions as inline copy of `transformers.TrainingArguments` docstring rather than a runnable config.
- **Fix approach:** Either rename to `_REFERENCE.md` outside the Hydra search path, or delete and link to the `transformers` docs from `README.md`.

### [Medium] Duplicate / superseded `compute_metrics` implementations

- **Files:** `planktonzilla/train.py:90-97` (commented HF-evaluate version) vs `planktonzilla/train.py:99-111` (active sklearn version)
- **Issue:** The older `evaluate.combine([...])` implementation is preserved as a comment, but `evaluate` is still required at the top of the file (`from evaluate import combine, load`, line 32) yet never used.
- **Fix approach:** Drop the commented block, drop the unused `evaluate` import, and consider whether `evaluate` should remain in `pyproject.toml` (currently `evaluate = "^0.4.6"`).

### [Medium] Empty `accelerate/` config directory

- **Files:** `configs/training_arguments/accelerate/` (empty dir, tracked via `.DS_Store`-only context)
- **Issue:** Empty config dir suggests an aborted/half-finished migration to `accelerate`.
- **Fix approach:** Delete the directory or land the intended configs.

## Operational Fragility (ML-Specific)

### [High] Hardcoded Jean Zay (`/lustre/...`) paths leak across notebooks and scripts

- **Files:**
  - `notebooks/add_planktonzilla.py:354` — `DATA_ROOT = Path("/lustre/fsn1/projects/rech/tec/uod68bo/data").resolve()`
  - `notebooks/gen_planktonzilla.py:466,469` — same `/lustre/fsn1/...` paths plus `taxo_csv_path = "/lustre/fswork/.../planktonzilla/notebooks/planktonzilla_taxo.csv"`
  - `notebooks/gen_planktonzilla_ood.py:110` — `dataset.save_to_disk(f"/lustre/fsn1/projects/rech/tec/uod68bo/data/planktonzilla_ood")`
  - `notebooks/push_planktonzilla.py:13` — `load_from_disk("/lustre/fsn1/projects/rech/tec/uod68bo/data/planktonzilla_ood")`
  - `notebooks/save_planktonzilla_for_clip.py:76,78` — same pattern
  - `scripts/push_dataset.sh:14` — `dataset_import.data_dir=/lustre/fsn1/projects/rech/tec/uod68bo/data`
  - `scripts/train_clip.sh:20,43,44` — `PYTHONPATH=/home/acontreras/planktonzilla/open_clip/src` and `--train-data /lustre/fsn1/.../shards/...`
  - `notebooks/metrics_paper.ipynb` — many `/lustre/fswork/projects/rech/tec/uod68bo/am/open_clip/tutorials/*.npz` references (lines 958, 1138-1144, 1242, 1245, 1390-1409, 1989, 2074-2077, 2155-2158, 2244-2275)
- **Impact:** Anything that produces canonical artifacts (the `planktonzilla` HF dataset itself, OOD splits, WebDataset shards, npz prediction dumps) is bound to one user's home dir on one specific HPC. Reproducible only on Jean Zay, by `uod68bo`/`acontreras`.
- **Fix approach:**
  1. Promote `DATA_ROOT` and `OPEN_CLIP_SRC` to env vars (`PLANKT_DATA_ROOT`, `OPEN_CLIP_SRC`) read once at the top of each script/notebook with `os.environ.get(..., "./data")`.
  2. For shell scripts, use `${DATA_ROOT:?DATA_ROOT must be set}` rather than literal `/lustre/...`.
  3. Reference these vars from `configs/paths/default.yaml` so Hydra runs are portable.

### [Medium] No deterministic dataloader workers; `worker_init_fn` is never set

- **Files:** `configs/training_arguments/default.yaml:22-24` (`dataloader_num_workers: 4`, `dataloader_persistent_workers: true`), `planktonzilla/train.py:251-258` (Trainer instantiated with no `data_collator`/`worker_init_fn` overrides)
- **Issue:** Each worker process inherits a forked RNG state; without a `worker_init_fn` (e.g. `lambda wid: np.random.seed(seed + wid)`) the augmentation pipeline (`configs/augmentation/*.yaml` → torchvision v2 transforms) is not deterministically seeded across workers. `set_seed(cfg.seed, deterministic=False)` in `planktonzilla/train.py:137` covers torch/python/numpy in the main process but not per-worker NumPy RNG.
- **Impact:** Identical config + git SHA still gives non-bit-identical training runs whenever augmentation is enabled — papers/benchmarks become hard to defend.
- **Fix approach:**
  - Set `cfg.deterministic: true` in `configs/train.yaml:55` for the reproducibility-critical experiments (it currently ships `false`).
  - Pass a `worker_init_fn` to the Trainer's dataloader (the easiest path is to subclass and override `get_train_dataloader`/`get_eval_dataloader`).
  - Document the `seed`/`split_seed`/`data_seed` interaction (currently `cfg.seed = ${dataset.split_seed}` from `configs/train.yaml:54` and `seed: 42` is also hardcoded under `configs/training_arguments/default.yaml:60`, which is confusing — two seeds with the same name in different scopes).

### [Medium] Run provenance is not tied to git SHA / config snapshot

- **Files:** `planktonzilla/train.py` (whole file), `configs/hydra/default.yaml:9-13` (run dir uses `${oc.env:SLURM_JOB_ID,local}`), `configs/tracking/default.yaml`
- **Issue:** Hydra writes the resolved config to `${log_dir}/${task_name}/runs/<timestamp>_<slurm_id>/.hydra/`, which is great, but:
  - The git SHA / dirty state is **not** logged anywhere (no `git rev-parse HEAD` step in `task_wrapper`, no `wandb.config["git_commit"]`).
  - The W&B run name is set to `model.name_or_path + "__" + dataset.name` at `planktonzilla/train.py:248` — multiple runs of the same model/dataset overwrite each other in the W&B "name" filter (W&B distinguishes by `id` only).
  - `cfg.tracking.wandb_log_model` defaults to `"false"` (`configs/tracking/default.yaml:5`) so checkpoints are not even archived to the W&B side.
- **Impact:** When a paper figure cites `wandb_run=abc123`, you cannot reliably ask "which commit produced that?".
- **Fix approach:** Add a few lines in `validate_environment()` (`planktonzilla/train.py:51`) to compute `subprocess.check_output(["git", "rev-parse", "HEAD"])`, append to `cfg.tags`, and call `wandb.config.update({"git_commit": sha, "git_dirty": <bool>})` after Trainer init.

### [Medium] Notebook → script drift: dataset construction lives only in `notebooks/`

- **Files:**
  - `notebooks/gen_planktonzilla.py` (717 lines) — the actual `planktonzilla` HF dataset is built here.
  - `notebooks/gen_planktonzilla_ood.py` (115 lines) — OOD split.
  - `notebooks/save_planktonzilla_for_clip.py` (82 lines) — converts to WebDataset shards consumed by `scripts/train_clip.sh`.
  - `notebooks/add_planktonzilla.py` (433 lines) — augments the HF dataset.
  - `notebooks/push_planktonzilla.py` (60 lines) — pushes to HF hub.
- **Issue:** None of these are exposed as `[project.scripts]` entry points (only `pz_import_dataset`, `pz_prepare_train`, `pz_train`, `pz_push_model` are — see `pyproject.toml:73-76`, but `pz_prepare_train` and `pz_push_model` don't actually exist in the package and will fail with `ModuleNotFoundError`). They're also `.py` files inside `notebooks/`, which is excluded from `ruff` (`pyproject.toml:65`), so they receive zero static checks.
- **Impact:** The pipeline that produces the project's flagship dataset is only runnable by reading shell scripts, finding which `.py` they invoke, and replicating the user's CWD. CI cannot smoke-test it.
- **Fix approach:**
  1. Promote each `notebooks/*.py` to `planktonzilla/data_pipeline/<name>.py` and register them under `[project.scripts]`.
  2. Drop them from the ruff exclude.
  3. Verify `pz_prepare_train` / `pz_push_model` either exist or remove the dangling entries from `pyproject.toml:74-76`.

### [Medium] `cleanup_after_processing: false` plus no on-disk size checks in dataset importers

- **Files:** `configs/dataset_import/default.yaml:13` (`cleanup_after_processing: false`), `planktonzilla/dataset_import/dataset_importer.py:314-326`
- **Issue:** Default leaves the raw downloads (e.g. WHOI ~3.5M images, ~tens of GB; ISIIS 408k objects) on disk indefinitely after the imagefolder is built. There is no `--max-disk` guard or `df`-style check.
- **Impact:** Easy to fill up shared scratch on Jean Zay or a dev laptop; `data/isiisnetdatasetimporter_raw_download/` will sit forever until manually purged.
- **Fix approach:** Set `cleanup_after_processing: true` as the new default (or have `import_dataset.py` log a `WARNING` showing `du -sh self.raw_dir` after import).

### [Low] `compute_mean_and_std_dev` returns from out-of-scope variable

- **Files:** `planktonzilla/dataset.py:88-93`
- **Issue:** The trailing `if/elif` block uses `image_array` (the loop variable from the previous `for item in huggingface_dataset` loop). If the dataset is empty, `image_array` is undefined and the function raises `UnboundLocalError`.
- **Fix approach:** Track the channel count once at the start of the loop and store it in a local; assert non-empty dataset.

## Performance

### [Medium] `compute_mean_and_std_dev` iterates the full HF dataset in pure Python

- **Files:** `planktonzilla/dataset.py:36-93`
- **Issue:** Loops over every `item` in a (potentially multi-million-image) `Dataset`, calling `np.array(image)` per row. No batching, no `Dataset.map(num_proc=...)`, no streaming sample. For WHOI (3.5M images) this is single-process, single-threaded, ~hours of wall time per dataset card refresh.
- **Impact:** Calling `update_dataset_metadata()` (which invokes this) on `pz_import_dataset action=update-metadata` blocks the dataset-card pipeline for hours.
- **Fix approach:** Either subsample (`huggingface_dataset.shuffle(seed=...).select(range(N))` with N=10k) for the card statistics, or use `dataset.map(..., batched=True, num_proc=cpu_count())` with a per-batch reducer.

### [Low] Notebook `.ipynb` files are huge (1.6 MB) due to embedded plot images

- **Files:** `notebooks/metrics_clip.ipynb` (1.6 MB), `notebooks/metrics_paper.ipynb` (90 KB), `notebooks/fix_taxo.ipynb` (20 KB)
- **Issue:** Embedded base64-encoded matplotlib output bloats the notebook and inflates the git history. Diffs are unreviewable.
- **Fix approach:** Run `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before committing, or add a `pre-commit` hook (`nbstripout`).

## Security & Secrets

### [None found] Source tree, configs, and notebooks are clean

- **Scanned:** `planktonzilla/`, `scripts/`, `configs/`, `pyproject.toml`, `notebooks/*.py`, `notebooks/*.ipynb`
- **Result:** No `WANDB_API_KEY=...`, `HF_TOKEN=...`, `hf_xxx`, `sk-xxx`, or `AKIA...` literals. All references go through `${oc.env:HF_TOKEN, null}` (`configs/dataset_import/default.yaml:16`) or `os.environ["WANDB_*"]` (`planktonzilla/train.py:232-236`). The `huggingface_hub.utils.get_token()` call in `notebooks/push_planktonzilla.py:21` reads the user's local HF cache and is safe to commit.

### [Low] `.devcontainer/.env` exists locally; verify before sharing the workspace

- **Files:** `.devcontainer/.env` (164 bytes, **gitignored** via `.gitignore:148` and confirmed in `git status --ignored`); `.devcontainer/setup_env.sh` writes `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL` here.
- **Status:** Currently safe — the file is ignored. The script only writes git identity (no API keys), so the present contents are not sensitive. Flag for awareness only: if anyone later starts dropping `WANDB_API_KEY` or `HF_TOKEN` here, they remain ignored, which is the correct behaviour.
- **Action:** No action required. (Do not read or commit the file.)

### [Low] `huggingface_hub.login(write_permission=True)` runs unconditionally when `HF_TOKEN` is set

- **Files:** `planktonzilla/train.py:67-73`
- **Issue:** Every training launch performs a write-scoped login, even for read-only `do_train: true; model_push_to_hub: false` runs.
- **Impact:** A leaked `HF_TOKEN` from a CI runner or shared SLURM node has unnecessarily broad scope.
- **Fix approach:** Only call `login(write_permission=True)` when `cfg.model_push_to_hub` is true; otherwise `login(write_permission=False)` (or skip the call and rely on `HF_TOKEN` being read from env by `huggingface_hub` automatically).

## Fragile Areas

### [High] Hard coupling between `planktonzilla.clip_model.ClipClassifier` and the vendored `open_clip` internals

- **Files:** `planktonzilla/clip_model.py:32-46`
- **Issue:** The class probes `self.model.proj` (a private ViT attribute), assigns `self.model.proj = None` to drop the projection, and on `AttributeError` falls back to `self.model = self.model.trunk` and overwrites `self.model.head` — both `trunk` and `head` are timm-internals, not `open_clip` public API. Any refactor in either lib silently breaks classification head construction.
- **Impact:** Upgrading `open_clip` or `timm` (currently `^1.0.20`) is a hand-grenade.
- **Fix approach:** Replace the heuristic with explicit branching by model name (`if name.startswith(("ViT-", "EVA"))`) or use `open_clip.get_model_config()` to detect arch. Add a unit test in `tests/test_train.py` that round-trips at least one ViT and one timm-backed CLIP through `ClipClassifier`.

### [Medium] `train.py` swallows missing `cfg.peft` typo via duck-typing on `cfg.get("peft")`

- **Files:** `planktonzilla/train.py:189-194`
- **Issue:** `if cfg.get("peft")` is true for both a populated dict **and** the placeholder dict `{"_target_": null, ...}` from `configs/peft/default.yaml` (which yields a Hydra DictConfig of `null` keys). Iterating with `for adapter_name in cfg.peft:` then either does nothing silently or, if a malformed entry slips in, crashes mid-training.
- **Fix approach:** Validate `cfg.peft` is a non-empty dict whose values each contain a `_target_` before calling `instantiate`.

### [Medium] Train/val/test splits are recomputed from scratch on every run

- **Files:** `planktonzilla/dataset.py:152-169`
- **Issue:** When `validation`/`test` splits don't already exist on the loaded HF dataset, `prepare_datasets()` rebuilds them via `train_test_split(stratify_by_column="label", seed=split_seed)` on each invocation. This is deterministic given the seed, but the dataset version on the hub can change underneath (datasets are pulled by `name`, not by revision SHA) — causing silent split drift.
- **Fix approach:** Pin the dataset revision: `load_dataset(self.name, revision=self.revision, streaming=self.streaming)` and surface `revision` in `configs/dataset/default.yaml`.

### [Low] `prepare_datasets` will load **all training labels into memory** to compute `cls_num_list`

- **Files:** `planktonzilla/dataset.py:171` (`np.unique(self.dataset["train"]["label"], return_counts=True)`)
- **Issue:** `dataset["train"]["label"]` materializes the full label column. Fine for current datasets but won't scale to streaming or to multi-million-row WebDatasets.
- **Fix approach:** Use `Counter` over `dataset["train"].to_iterable_dataset()` or precompute `cls_num_list` once and store on the dataset card.

## Project Hygiene

### [Low] `train.log` and `import_dataset.log` exist at repo root

- **Files:** `train.log` (96 KB), `import_dataset.log` (108 KB)
- **Status:** **Already gitignored** — `git check-ignore` matches them via `.gitignore:59 (*.log)` and `git status --ignored` lists them as Ignored. They will not be accidentally committed.
- **Recommendation:** Move them under `logs/` (which is already ignored via `.gitignore:210`) so the repo root stays clean. The current Python logger writes them to CWD.

### [Low] `wandb/` directory present locally but properly ignored

- **Files:** `wandb/` (416 KB, 19 run subdirs, contains symlinks `debug.log -> run-.../logs/debug.log`)
- **Status:** **Gitignored** (`.gitignore` final block, confirmed via `git status --ignored`). Not in `git ls-files` (`git ls-files | grep wandb` → empty). Safe.
- **Recommendation:** None required, but `cfg.tracking.wandb_dir` defaults to `${paths.log_dir}` (`configs/tracking/default.yaml:6`), so all *new* runs already land under `logs/wandb/`. The 19 stale runs at repo root are leftovers from before that change — `rm -rf wandb/` to clean up.

### [Low] `data/` and `logs/` are properly ignored, but contain large local artifacts

- **Files:** `logs/` (2.5 GB), `data/` (small now, will grow), both gitignored.
- **Status:** Safe; just confirm no one ever runs `git add -f`.

### [Low] No CI workflow

- **Files:** `.github/` is **gitignored** (`.gitignore:223`); only contains `copilot-instructions.md` locally.
- **Issue:** A repo without `.github/workflows/` means tests never run on PRs — `tests/test_train.py` and `tests/test_datasets.py` are gated by `skip_in_github_ci` (`tests/shared.py:19-25`) anyway, but even `test_dataset_instantiation` (which is *not* skipped) is not exercised on push.
- **Fix approach:** Add a minimal `.github/workflows/test.yml` that runs `poetry install && poetry run pytest tests/test_datasets.py::test_dataset_instantiation`. Remove `.github/` from `.gitignore` (or scope the ignore to specific files).

### [Low] `planktonzilla/__pycache__/` shows up in `git status --ignored` despite being inside an ignored package — verify nothing slipped in

- **Files:** `planktonzilla/__pycache__/`, `planktonzilla/dataset_import/__pycache__/`, `planktonzilla/utils/__pycache__/`, `tests/__pycache__/`
- **Status:** All correctly ignored.

## Test Coverage Gaps

### [High] `test_train.py` and the heavy `test_dataset_prepare_datasets` are skipped on CI

- **Files:** `tests/shared.py:19-25` (`skip_in_github_ci` decorator), `tests/test_train.py:24, 56` (both training tests gated), `tests/test_datasets.py:38` (`test_dataset_prepare_datasets` gated)
- **Issue:** The only test that ever runs in CI is `test_dataset_instantiation` (instantiates a `DatasetWrapper`, doesn't load anything). That's barely a smoke test.
- **Impact:** Regressions in `train.py`, `clip_model.py`, all 7 `loss.py` losses, and dataset preparation can land unnoticed.
- **Fix approach:** Add a tiny synthetic in-memory HF dataset (10 images, 2 classes) and run `test_training[resnet18, synthetic]` + `test_training_custom_losses` against it in CI. The current "skip if `GITHUB_ACTIONS == true`" is overly broad.

### [Medium] No test exercises `ClipClassifier`, `compute_metrics`, or any custom loss directly

- **Files:** `tests/test_train.py:13` (`model_names = ["resnet18"]` only), `tests/shared.py:11-15`
- **Issue:** None of the `default_clip` / `eva02-large-clip` / `vit-base-clip` model configs, nor any of the 6 custom losses, have a unit test independent of the full Trainer.
- **Fix approach:** Add `tests/test_losses.py` that instantiates each loss with dummy `(logits, labels)` and checks gradient flow + finite scalar.

### [Medium] No test exercises `dataset_importer.py` (777 lines, the largest file in the repo)

- **Files:** `planktonzilla/dataset_import/dataset_importer.py` — `DatasetImporter`, `LenslessDatasetImporter`, `ZooLakeDatasetImporter` (~10 importers total), all untested.
- **Fix approach:** At minimum, mock `DownloadManager` and assert `_prepare_imagefolder` produces the expected directory layout for one importer.

## Dependencies at Risk

### [Medium] `peft = "^0.18.1"` and `accelerate = "^1.10.1"` are pinned at floors that have known major-version churn

- **Files:** `pyproject.toml:30, 36`
- **Issue:** PEFT and Accelerate both ship breaking changes in minor versions historically; with `^` Poetry will allow 0.18.x → 0.99.x and 1.10.x → 1.99.x respectively (Poetry's caret on `0.x.y` allows `>=0.x.y, <0.(x+1).0`, but for `1.10.x` it allows `<2.0.0`).
- **Fix approach:** Tighten to `~0.18.1` (allow only `0.18.x`) for `peft`, evaluate before bumping.

### [Medium] `deepspeed` is commented out in `pyproject.toml`

- **Files:** `pyproject.toml:32` — `# deepspeed = "^0.18.0"`
- **Issue:** Comment suggests it was tried and removed. If any SLURM script (or `configs/training_arguments/many.yaml`) references DeepSpeed offload, it will fail at runtime.
- **Action:** Either re-enable or grep configs and remove any DeepSpeed-only options.

## Missing Critical Features

### [Medium] No `[project.scripts]` entry actually exists for `pz_prepare_train` and `pz_push_model`

- **Files:** `pyproject.toml:74-75` declare `pz_prepare_train = 'planktonzilla.prepare_train:main'` and `pz_push_model = 'planktonzilla.push_model:main'`, but the package has no `prepare_train.py` or `push_model.py` (verify via `ls planktonzilla/*.py` → only `clip_model.py`, `dataset.py`, `loss.py`, `train.py`).
- **Impact:** `poetry run pz_prepare_train` and `pz_push_model` raise `ModuleNotFoundError` at first invocation.
- **Fix approach:** Either implement the modules or remove the entries.

### [Low] No `README.md` section / `docs/` page describes the W&B project, HF org, or how to obtain `HF_TOKEN`

- **Files:** `README.md` (10 KB, present), `docs/` (1 subdir)
- **Action:** Document the `HF_TOKEN`, `WANDB_API_KEY`, and `WANDB_MODE=offline` ⇒ `wandb sync` workflow.

---

*Concerns audit: 2026-05-12*
