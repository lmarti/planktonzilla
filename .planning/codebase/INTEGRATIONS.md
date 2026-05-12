# External Integrations

**Analysis Date:** 2026-05-12

## APIs & External Services

**Model & Dataset Hub:**
- **Hugging Face Hub** (`https://huggingface.co/project-oceania`) — primary registry for datasets, trained checkpoints and dataset cards.
  - Client: `huggingface_hub` `0.36.0` (`planktonzilla/train.py:33` → `from huggingface_hub import DatasetCard, login`).
  - Auth: env var `HF_TOKEN` (validated in `planktonzilla/train.py:53-74`; passed through devcontainer in `.devcontainer/devcontainer.json:77`).
  - Offline mode: env var `HF_HUB_OFFLINE=1` (set in SLURM scripts `scripts/train.sh:21`, `scripts/train_clip.sh:18`, and Hydra launcher `configs/hydra/launcher/jeanzay_submitit_h100mono.yaml:46`).
  - Org used by the project: `project-oceania` (`configs/dataset_import/default.yaml:15`, `configs/train.yaml:49`).
  - Dataset name template: `{hf_org_name}/{hf_dataset_name}` (`planktonzilla/dataset.py:151`, importer template `planktonzilla/dataset_import/dataset_importer.py:43-76`).
  - Push pattern: trained models pushed via `transformers.Trainer.push_to_hub(...)` in `planktonzilla/train.py:294`. Repo names follow `{org}/{prefix}_{model}_{dataset}`, e.g. `project-oceania/pz_microsoft_resnet-18_project-oceania_isiisnet`. `model_push_as_private: true` by default.

**Vision backbones (downloaded transparently from HF Hub at instantiate time):**
- `microsoft/resnet-18` (`configs/model/resnet18.yaml`)
- `microsoft/beit-base-patch16-224` (`configs/model/beit-base.yaml`)
- `google/vit-base-patch16-224` (`configs/model/vit-base.yaml`)
- `timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (`configs/model/timm-eva02-large-m38m.yaml`)
- `timm/eva_giant_patch14_336.clip_ft_in1k` (`configs/model/timm-evagiant-m30m-336.yaml`)
- `timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k` (`configs/model/timm-beitv2_large_patch16_224.yaml`)
- `timm/convnextv2_huge.fcmae_ft_in22k_in1k_384` (`configs/model/timm-convnextv-huge-384.yaml`)
- `timm/deit3_huge_patch14_224.fb_in22k_ft_in1k` (`configs/model/timm-deit3-huge-224.yaml`)
- `vit_base_patch16_clip_224.openai` via `transformers` (`configs/model/vit-base-clip-224-openai.yaml`) and via `timm` (`configs/model/timm-vit-base-16-clip-openai.yaml`).
- `EVA02-L-14` + `merged2b_s4b_b131k` via vendored `open_clip` (`configs/model/eva02-large-clip-224-2b-s4b-b131k.yaml`, instantiated through `planktonzilla.clip_model.ClipClassifier`).

**Experiment tracking services:**
- **Weights & Biases (wandb.ai)**
  - Client: `wandb` `0.22.3`.
  - Project: `planktonzilla-turbo`, entity: `oceania-plankton` (`configs/tracking/default.yaml:1-7`).
  - Auth: env var `WANDB_API_KEY` (forwarded through devcontainer `.devcontainer/devcontainer.json:71-72`; checked in `planktonzilla/train.py:76-81`).
  - Online/offline: env var `WANDB_MODE` (`offline` enforced on Jean Zay in `scripts/train.sh:20`, `scripts/train_clip.sh:17`).
  - Local artifacts: `wandb/` directory at repo root (20 historical run folders, e.g. `run-20260102_001950-9c7gr94x/`).
- **MLflow** (optional, code paths exist)
  - Tracking URI: env var `MLFLOW_TRACKING_URI` (devcontainer `.devcontainer/devcontainer.json:75`); falls back to `file:${paths.log_dir}/mlflow` (`configs/tracking/default.yaml:11-13`).
  - Optional auth: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (probed in `planktonzilla/train.py:84-88`).
  - Experiment name: defaults to repo basename (`MLFLOW_EXPERIMENT_NAME` env in devcontainer, `mlflow_experiment_name: vendimia50` in config).
  - Toggled by `cfg.tracking.use_mlflow`. **Not installed** by `pyproject.toml` / `poetry.lock` — runtime install required if enabled.
- **Trackio** (HF-based tracking)
  - Client: `trackio` `0.5.3`.
  - Logs run metadata to HF dataset `project-oceania/pz_experiments` (`configs/tracking/default.yaml:18`).
  - Local dir: `${paths.log_dir}/trackio`.

## Data Storage

**Datasets:**
- All canonical training datasets live on Hugging Face Hub under `project-oceania/*` (e.g. `project-oceania/isiisnet`, `project-oceania/planktonzilla_only_plankton`, `project-oceania/whoi-plankton`, `project-oceania/lensless`, `project-oceania/zoolake`, `project-oceania/uvp6net`, `project-oceania/jedi_oceans`, `project-oceania/flowcamnet`, `project-oceania/zooscannet`).
- Loaded via `datasets.load_dataset(self.name, streaming=self.streaming)` in `planktonzilla/dataset.py:151`.
- Local working copies under `data/`:
  - `data/isiisnetdatasetimporter_imagefolder/` — 32 plankton class subfolders (`Acantharea`, `Annelida`, `Appendicularia`, `Aulacanthidae`, `Bacillariophyceae`, `Chaetognatha`, `Cnidaria`, …).
  - `data/isiisnetdatasetimporter_raw_download/` — staged raw download from upstream (sha-named tar payload).
  - The `data/` directory is gitignored (`.gitignore:222`).

**Cluster data path:**
- Lustre paths used in production training scripts:
  - `/lustre/fsn1/projects/rech/tec/uod68bo/data/shards/train/shard_{00000..01771}.tar` and `validation/shard_{00000..00590}.tar` (`scripts/train_clip.sh:37-38`).
  - `/lustre/fsn1/projects/rech/tec/uod68bo/data` for `pz_import_dataset` runs (`scripts/push_dataset.sh:14`).
- Sharded TAR layout consumed via `webdataset` from `open_clip_train`.

**File Storage:**
- Local filesystem only, organised by Hydra into time-stamped run dirs:
  - `logs/${task_name}/runs/${YYYY-MM-DD}_${HH-MM-SS}_${SLURM_JOB_ID}/` (single run)
  - `logs/${task_name}/multiruns/.../` (Hydra sweeps)
  - Defined in `configs/hydra/default.yaml:9-13`.
- `logs/` directory currently contains: `import_dataset/`, `train/`, `train_node/`, `wandb/`.
- No object storage / S3 / GCS integration in the codebase.

**Caching:**
- Hugging Face datasets cache on disk (default `~/.cache/huggingface/`), with `dataset_import/default.yaml` honouring `force_download`, `resume_download`, `force_imagefolder_preparation`.
- Pip cache persisted across container rebuilds via the `remote-pip-cache-${USER}` Docker volume (`.devcontainer/devcontainer.json:47`).

## Authentication & Identity

- No application-level auth (no users, no sessions — this is an ML library/CLI).
- External-service auth only:
  - `HF_TOKEN` → Hugging Face Hub.
  - `WANDB_API_KEY` → Weights & Biases.
  - Optional `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.
  - Git author identity propagated into the container via `.devcontainer/setup_env.sh` (writes `.devcontainer/.env` with `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL`).
- Container `postCreateCommand` (`.devcontainer/devcontainer.json:64`) clones `https://github.com/Inria-Chile/${localWorkspaceFolderBasename}` if no `.git` exists — uses host SSH/HTTPS creds.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry / Rollbar / OpenTelemetry integration.

**Logs:**
- File logs: `train.log`, `import_dataset.log` at repo root (each ≈100 KB; example messages format `[YYYY-MM-DD HH:MM:SS][module][LEVEL] - message`).
- Per-run Hydra logs under `logs/${task_name}/runs/.../`.
- pytest log_cli enabled (`pyproject.toml:75-76`) so test runs print structured `%(asctime)s [%(levelname)7s] %(message)s` lines.
- Library-level loggers via `planktonzilla.utils.logger.get_pylogger` (`planktonzilla/utils/logger.py`) used throughout `planktonzilla/`.
- Structured rich console output via `rich` (`planktonzilla/utils/rich_utils.py` prints config tree at run start, controlled by `configs/extras/default.yaml:print_config`).

## CI/CD & Deployment

**Hosting:**
- Models distributed through Hugging Face Hub (`project-oceania` org).
- Compute hosted on:
  - **Jean Zay** (IDRIS, France) — H100 GPU partition `gpu_p6`, account `tec@h100`, qos `qos_gpu_h100-t3` / `qos_gpu_h100-dev` (`configs/hydra/launcher/jeanzay_submitit_h100mono.yaml`, `scripts/train.sh`, `scripts/train_clip.sh`, `scripts/push_dataset.sh`).
  - **Grid'5000** (placeholder launcher: `configs/hydra/launcher/grid5000_submitit.yaml`).
  - Local SLURM stub: `configs/hydra/launcher/local_submitit.yaml`.

**CI Pipeline:**
- None. `.github/` only contains `copilot-instructions.md` — no `.github/workflows/` directory.
- The test suite contains a CI guard (`skip_in_github_ci` in `tests/shared.py`) reserved for future GitHub Actions adoption.

## Environment Configuration

**Required env vars (presence checked at runtime):**
| Variable | Used by | Required when |
|----------|---------|---------------|
| `HF_TOKEN` | `huggingface_hub.login` (`planktonzilla/train.py:67`) | Pulling/pushing private HF datasets/models |
| `HF_HUB_OFFLINE` | conditional skip (`planktonzilla/train.py:61-63`) | Air-gapped clusters |
| `WANDB_API_KEY` | implicit by `wandb` SDK | `cfg.tracking.use_wandb=true` |
| `WANDB_MODE` | wandb online/offline | Set to `offline` on Jean Zay |
| `MLFLOW_TRACKING_URI` | env-overrides config (`planktonzilla/train.py:241`) | `cfg.tracking.use_mlflow=true` |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | basic auth for remote MLflow | If MLflow server is auth-protected |
| `PROJECT_ROOT` | `configs/paths/default.yaml:4` | Always (set by `pyrootutils`) |
| `SLURM_JOB_ID` | Hydra run-dir naming (`configs/hydra/default.yaml:10-12`) | Optional (defaults to `local`) |
| `CUTLASS_PATH` | NVIDIA cutlass headers (`.devcontainer/Dockerfile:72`) | Only if a kernel needs it |
| `TORCH_DISTRIBUTED_TIMEOUT`, `NCCL_TIMEOUT`, `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_ASYNC_ERROR_HANDLING` | distributed CLIP training (`scripts/train_clip.sh:21-24`) | Multi-node CLIP runs |

**Secrets location:**
- No secret values in repo. Tokens injected via host environment and forwarded through `.devcontainer/devcontainer.json` `remoteEnv` (`HF_TOKEN`, `WANDB_API_KEY`, `MLFLOW_TRACKING_URI`).
- `.devcontainer/.env` contains only Git author metadata (autogenerated by `setup_env.sh`).
- `.gitignore` excludes `.env`, `.env.*`, `.venv/`, `data/`, `logs/`, `wandb/` cache, `poetry.lock` (entry overridden — lock IS committed).

## Webhooks & Callbacks

**Incoming:** None — no HTTP server.

**Outgoing:**
- Synchronous metric posts to W&B / MLflow / Trackio during training (driven by `transformers.Trainer.report_to`, set in `planktonzilla/train.py:251`).
- Synchronous push to Hugging Face Hub at end of training when `cfg.model_push_to_hub=true` (`planktonzilla/train.py:292-295`).

## Vendored / Forked Libraries

- **`open_clip`** vendored under `open_clip/src/open_clip/` and `open_clip/src/open_clip_train/` (≈ 8.5 kLOC of Python plus `bpe_simple_vocab_16e6.txt.gz` and 145 model configs in `open_clip/src/open_clip/model_configs/`).
  - Internal version: `4.0.0.dev0` (`open_clip/src/open_clip/version.py`).
  - Upstream: `https://github.com/mlfoundations/open_clip` (MIT licensed; copyright Gabriel Ilharco et al., see `open_clip/LICENSE`).
  - **Not declared as a pip dependency.** Imported by adding `open_clip/src` to `PYTHONPATH` (e.g. `scripts/train_clip.sh:14`: `export PYTHONPATH=/home/acontreras/planktonzilla/open_clip/src:$PYTHONPATH`).
  - Consumed by `planktonzilla/clip_model.py:2` (`import open_clip`) which wraps `open_clip.create_model_and_transforms(...)` into a `transformers`-compatible `ClipClassifier`.
  - Likely vendored to allow custom training (`open_clip_train.main`) on the project's own `webdataset`-format shards and to apply local patches without depending on PyPI release cadence.

## Dataset Sources (external URLs declared in importer configs)

| Dataset | Source URL | License | Config |
|---------|------------|---------|--------|
| ISIISNet | `https://www.seanoe.org/data/00908/101950/` (download `.tar`) | CC-BY-NC-4.0 | `configs/dataset_import/isiisnet.yaml` |
| WHOI-Plankton | `https://darchive.mblwhoilibrary.org/bitstreams/...` (9 archives) | MIT | `configs/dataset_import/whoi-plankton.yaml` |
| ZooLake | `https://opendata.eawag.ch/dataset/.../data.zip` | CC-BY-4.0 | `configs/dataset_import/zoolake.yaml` |
| Lensless | `https://ibm.ent.box.com/v/PlanktonData` | CC-BY-4.0 | `configs/dataset_import/lensless.yaml` |
| FlowCamNet | declared in `configs/dataset_import/flowcamnet.yaml` | — | `configs/dataset_import/flowcamnet.yaml` |
| Global UVP5Net | declared in `configs/dataset_import/global_uvp5net.yaml` | — | `configs/dataset_import/global_uvp5net.yaml` |
| JEDI Oceans CPICS | declared in `configs/dataset_import/jedi_oceans_cpics.yaml` | — | `configs/dataset_import/jedi_oceans_cpics.yaml` |
| MedPlanktonSet | declared in `configs/dataset_import/medplanktonset.yaml` | — | `configs/dataset_import/medplanktonset.yaml` |
| PlanktonSet1 | declared in `configs/dataset_import/planktonset1.yaml` | — | `configs/dataset_import/planktonset1.yaml` |
| PlanktoScope | declared in `configs/dataset_import/planktoscope.yaml` | — | `configs/dataset_import/planktoscope.yaml` |
| SYKE-IFCB-2022 | declared in `configs/dataset_import/syke_ifcb_2022.yaml` | — | `configs/dataset_import/syke_ifcb_2022.yaml` |
| SYKE ZooScan 2024 | declared in `configs/dataset_import/sykezooscan2024.yaml` | — | `configs/dataset_import/sykezooscan2024.yaml` |
| UVP6Net | declared in `configs/dataset_import/uvp6net.yaml` | — | `configs/dataset_import/uvp6net.yaml` |
| ZooCamNet | declared in `configs/dataset_import/zoocamnet.yaml` | — | `configs/dataset_import/zoocamnet.yaml` |
| ZooScanNet | declared in `configs/dataset_import/zooscannet.yaml` | — | `configs/dataset_import/zooscannet.yaml` |

All importer configs share `configs/dataset_import/default.yaml` (sets `hf_org_name: project-oceania`, `hf_token: ${oc.env:HF_TOKEN, null}`).

## Pre-trained Pretraining Tags (vendored open_clip)

- `merged2b_s4b_b131k` — pretraining tag used with `EVA02-L-14` (`configs/model/eva02-large-clip-224-2b-s4b-b131k.yaml:9`). Resolved via `open_clip.pretrained.get_pretrained_url(...)` in `open_clip/src/open_clip/pretrained.py`.
- All other pretraining tags listed in `open_clip/src/open_clip/pretrained.py` are reachable but only the EVA02-L-14 + merged2b combo is wired into a `configs/model/*.yaml`.

## Notebook Tooling

- Jupyter `notebook` `7.5.1`, `ipykernel` `7.1.0`, `ipywidgets` `8.1.8` available via the `dev` dependency group.
- `notebooks/` contains a mix of `.ipynb` (interactive) and `.py` files run as standalone scripts (e.g. via `python save_planktonzilla2.py` from `scripts/save_plankt.sh`).
- No papermill/nbconvert orchestration detected; notebooks are run manually.
- Ruff lint excludes `notebooks/` (`pyproject.toml:67-69`).

---

*Integration audit: 2026-05-12*
