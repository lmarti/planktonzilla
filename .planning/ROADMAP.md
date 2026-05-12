# Roadmap: planktonzilla

**Created:** 2026-05-12
**Granularity:** coarse (5 phases — at the upper edge of the coarse band, but each phase has a clean, verifiable unit of completion)
**Mode:** interactive
**Project mode:** standard (Horizontal Layers)

**Core Value:** Curated, citable pre-trained plankton-classification models on Hugging Face Hub, each with a real model card, backed by a repo a stranger can credibly load a model from.

**Sequencing principles applied:**
- REL-06 honored: CLIP-backed models ship FIRST (Phase 3), pure-HF models follow (Phase 4)
- HARD-04 (clean-env smoke test) is satisfied in Phase 2, before any REL-* publish in Phase 3+
- LAUNCH-* is the LAST phase — no public announcement until working URLs exist for every artifact
- Phase 4 runs the pure-HF publish wave AND the Gradio Space in parallel, both gated on Phase 3's spike having proven the publishing template + producing one live model

## Phases

- [ ] **Phase 1: Foundation** — Freeze the inputs (winning checkpoint manifest + label-mapping freeze) so every downstream phase knows what is being shipped
- [ ] **Phase 2: Hardening (load-path only)** — Bump `transformers` 5.x / `huggingface_hub` 1.x, pin `open-clip-torch` from PyPI, build the standalone `clip_classifier_hub/` shim, and prove the clean-env load path on Docker
- [ ] **Phase 3: CLIP-first publish** — End-to-end one-CLIP-model spike (proves the entire publishing template), then scale to the remaining 3 CLIP models on the Hub with full structured model cards and reproducible eval
- [ ] **Phase 4: Pure-HF publish + Gradio Space** — Mechanically scale the publish template to the 3 pure-HF models, build the Gradio Space, and create the OcéanIA HF Collection (parallelizable)
- [ ] **Phase 5: Documentation + Launch** — README rewrite covering the 4 headline use cases, committed clean-env smoke test, HF org-level README, GitHub release tag, pre-launch PITFALLS checklist pass, and announcement copy

## Phase Details

### Phase 1: Foundation
**Goal**: Freeze the curated inputs to the release — every downstream phase depends on knowing which checkpoints are being shipped, which architecture they are, what labels they emit, and which dataset license is being declared.
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02
**Success Criteria** (what must be TRUE):
  1. A reviewer can open `.planning/release-manifest.yaml` and see, for each of the 7 datasets, the physically-retrievable checkpoint location, its architecture (`pure-hf` | `clip`), the source dataset's HF dataset ID, and the dataset's license declared verbatim from upstream (or `other` with a pointer note if the upstream license is unknown)
  2. Every checkpoint listed in the manifest can actually be loaded from its declared location (no entries pointing at cleared SLURM scratch, no entries pointing at non-existent HF private repos)
  3. For each checkpoint, a frozen `id2label` / `label2id` mapping is recorded that matches the trained model's classification head index-by-index against the source dataset's existing class list
  4. The manifest unambiguously partitions the 7 models into the CLIP wave (Phase 3) and the pure-HF wave (Phase 4), so downstream phases can sequence themselves without re-litigating the architecture question
**Plans**: TBD

### Phase 2: Hardening (load-path only)
**Goal**: Make the "load me from a clean env" snippet physically possible — fix the dependency drift, pin `open-clip-torch` from PyPI, build the standalone CLIP shim, and prove the clean-env load path works in Docker before any model card promises it.
**Depends on**: Phase 1 (need the manifest's CLIP-vs-pure-HF partition to know whether the shim is exercised)
**Requirements**: HARD-01, HARD-02, HARD-03, HARD-04
**Success Criteria** (what must be TRUE):
  1. `pyproject.toml` declares `transformers >=5.0,<6` and `huggingface_hub >=1.0,<2`; `poetry lock` resolves cleanly; imports of `Trainer`, `AutoModelForImageClassification`, `ModelCard`, `ModelCardData`, `EvalResult`, and `register_for_auto_class` all succeed in the resolved environment
  2. `pyproject.toml` declares a pinned `open-clip-torch` version range that is API-compatible with the vendored `open_clip 4.0.0.dev0`; `scripts/train_clip.sh` no longer hardcodes `/home/acontreras/...`
  3. The standalone `clip_classifier_hub/` directory contains a `PreTrainedConfig` + `PreTrainedModel` pair that imports only `transformers`, `torch`, and `open_clip` — and explicitly does NOT import anything from the `planktonzilla` package
  4. In a fresh Docker container running `python:3.11`, after only `pip install transformers huggingface_hub open-clip-torch torch Pillow`, a 6-line `AutoModelForImageClassification.from_pretrained(repo_id, trust_remote_code=True, revision=<sha>)` snippet returns a working model for both a pure-HF test repo AND a CLIP-shim test repo
**Plans**: TBD

### Phase 3: CLIP-first publish
**Goal**: Prove the entire publishing template end-to-end on ONE CLIP model (highest-risk path first), then mechanically scale to the remaining CLIP models so the headline artifacts are live on the Hub with real, reproducible eval numbers and defensible model cards.
**Depends on**: Phase 2 (clean-env load path must be smoke-tested before any "load me" snippet is committed to a model card)
**Requirements**: REL-01, REL-02, REL-03, REL-04, REL-06
**Success Criteria** (what must be TRUE):
  1. Each of the 4 CLIP-backed models exists at `huggingface.co/project-oceania/<repo-name>` with: `model.safetensors` weights, `config.json` containing `auto_map` pointing at the shipped `modeling_clip_classifier.py` + `configuration_clip_classifier.py`, `preprocessor_config.json` matching training-time mean/std/size, and a `README.md` model card
  2. A `pip install`-only Python user (no `planktonzilla` clone, no `git+...` install) can call `AutoModelForImageClassification.from_pretrained("project-oceania/<repo>", trust_remote_code=True, revision=<sha>)` for any of the 4 CLIP models and get a working model whose `config.id2label[0]` returns a real species name (not `LABEL_0`)
  3. Every CLIP model's card is built via `huggingface_hub.ModelCard` + `ModelCardData` + `EvalResult` (NOT hand-rolled YAML, NOT the `Trainer.push_to_hub` autostub) and contains: dataset card cross-link (verbatim, no edits), base-model link, license declared verbatim from the source dataset, training-config snapshot reference (commit SHA + W&B run URL), citation BibTeX (model + dataset + base model), intended-use, limitations, bias/risks, and the plankton-domain caveats from `PITFALLS.md` (cross-instrument transfer, taxonomy non-standardization, OOD/background absence, cruise leakage)
  4. The headline metrics in every CLIP model card (top-1 accuracy + macro-F1 minimum) come from `release/eval_model.py` running against the published checkpoint and the dataset's held-out split — a reviewer downloading the model can re-run the script and reproduce the numbers within ±0.5pp; the script itself is committed and citable
**Plans**: TBD

### Phase 4: Pure-HF publish + Gradio Space
**Goal**: Scale the proven publishing template to the remaining pure-HF models AND ship the live Gradio Space + the OcéanIA HF Collection — these run in parallel because the Space dev only needs ONE model live (which Phase 3 provides).
**Depends on**: Phase 3 (template proven; at least one model is live so the Space can be developed against it)
**Requirements**: REL-05, DEMO-01, DEMO-02, DEMO-03, DEMO-04
**Success Criteria** (what must be TRUE):
  1. Each of the 3 remaining pure-HF models is live at `huggingface.co/project-oceania/<repo-name>` with the same model-card structural standard as the Phase 3 CLIP cards (the same Phase 3 success criterion #2 — clean-env loadable — applies, minus `trust_remote_code=True` for these repos), and the model-card metrics come from the same `release/eval_model.py` reproducible-eval script
  2. A visitor to `huggingface.co/spaces/project-oceania/planktonzilla-demo` can drag-and-drop a single image, pick any of the 7 shipped models from a Gradio dropdown, and see top-K class probabilities — with the model picker labeling each option by instrument (e.g. "ISIIS — In-Situ Shadowgraph") and the UI footer carrying a license + intended-use disclaimer (especially for CC-BY-NC-derived models)
  3. The Space has stable cold-start behavior: models lazy-load via `@lru_cache` per `repo_id`, `preload_from_hub` in the Space `README.md` YAML warms the disk cache at build time, and `requirements.txt` exactly pins all versions (matching Phase 2's `transformers` / `huggingface_hub` / `open-clip-torch` pins) so a "factory rebuild" produces byte-identical predictions on a fixed test image
  4. A single OcéanIA HF Collection page at `huggingface.co/collections/project-oceania/...` groups all 7 shipped models + linked datasets + the Space under one shareable URL with per-item notes
**Plans**: TBD
**UI hint**: yes

### Phase 5: Documentation + Launch
**Goal**: Turn the published artifacts into a discoverable, citable, and announceable release — README rewrite, committed smoke test, HF org-level page, GitHub release tag, pre-launch checklist pass, and announcement copy.
**Depends on**: Phases 3 and 4 (every documentation surface here links to live artifacts; the pre-launch checklist verifies them; the announcement names them)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, LAUNCH-01, LAUNCH-02
**Success Criteria** (what must be TRUE):
  1. A first-time visitor to `github.com/Inria-Chile/deep_plankton` reads a README that walks them through, in order: (1) loading a published planktonzilla model in 6 lines from a clean env, (2) trying the live Spaces demo (live URL, not "coming soon"), (3) retraining a published model on their own data, (4) importing a new dataset into the planktonzilla pipeline — with the existing training-framework content demoted to "Advanced Usage"
  2. The "load a published model" snippet in the README is verified by a committed `tests/test_clean_env_load.sh` script (run-on-demand, not in default CI) that re-runs the Phase 2 Docker smoke test against every shipped model — a developer can execute it locally and see all 7 models load and emit a real species-name prediction
  3. The HF org page at `huggingface.co/project-oceania` shows an org-level README with project description, link to the OcéanIA collection (Phase 4), link back to the GitHub repo, citation, and contact — and a GitHub release is tagged (`v1.0.0` or `v0.2.0`) with release notes summarizing which models shipped, the collection link, the demo link, and what is deferred to v1.1
  4. The pre-launch must-pass checklist from `PITFALLS.md` passes: every model-card YAML validates, every model-card inference snippet runs in a clean env, every `license:` field matches its source dataset's license verbatim, and no SOTA / "production-ready" / "best-in-class" claims appear anywhere
  5. Announcement copy (one-paragraph blurb + key links) exists with the plankton-domain disclaimers from `PITFALLS.md` (cross-instrument transfer, taxonomy specifics) and Inria/OcéanIA distribution channels identified — ready to send the moment the user signs off
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/0 | Not started | - |
| 2. Hardening (load-path only) | 0/0 | Not started | - |
| 3. CLIP-first publish | 0/0 | Not started | - |
| 4. Pure-HF publish + Gradio Space | 0/0 | Not started | - |
| 5. Documentation + Launch | 0/0 | Not started | - |

## Coverage

- v1 requirements: 22 total
- Mapped to phases: 22 ✓
- Orphaned: 0
- Duplicates: 0

| Requirement | Phase |
| ----------- | ----- |
| FOUND-01    | 1     |
| FOUND-02    | 1     |
| HARD-01     | 2     |
| HARD-02     | 2     |
| HARD-03     | 2     |
| HARD-04     | 2     |
| REL-01      | 3     |
| REL-02      | 3     |
| REL-03      | 3     |
| REL-04      | 3     |
| REL-05      | 4     |
| REL-06      | 3     |
| DEMO-01     | 4     |
| DEMO-02     | 4     |
| DEMO-03     | 4     |
| DEMO-04     | 4     |
| DOC-01      | 5     |
| DOC-02      | 5     |
| DOC-03      | 5     |
| DOC-04      | 5     |
| LAUNCH-01   | 5     |
| LAUNCH-02   | 5     |

**Sequencing checks:**
- ✓ REL-06 (CLIP-first) — CLIP models live in Phase 3; pure-HF wave deferred to Phase 4
- ✓ HARD-04 (clean-env smoke test) — satisfied in Phase 2 before any REL-* in Phase 3
- ✓ LAUNCH-01..02 — both in Phase 5 (the final phase); no launch-coordination work in earlier phases
- ✓ DEMO-* parallelizable with REL-* pure-HF wave — both in Phase 4, both unblocked by Phase 3's spike

## Out-of-Scope Reminders (NOT in any phase)

These are explicit non-goals for v1 (per `REQUIREMENTS.md` Out of Scope table) — no phase touches them:

- Dataset-side work (license audit, dataset card edits, taxonomy harmonization, contacting upstream dataset authors)
- Training-side hardening beyond HARD-01..04 (bare `except:`, `F.log_softmax(logits)` deprecation, `skip_in_github_ci`, full vendoring policy decision for `open_clip`)
- New training campaigns
- Saliency / Grad-CAM in the demo (deferred to v1.1)
- Pip / PyPI release
- Pretty docs site (MkDocs / Sphinx / GitHub Pages)
- Per-dataset benchmark paper / leaderboard write-up
- Cross-dataset taxonomic harmonization
- Inference API / "Use this model" widget on Hub (auto-disabled for `trust_remote_code=True` models — Space replaces it)
- HF "Benchmark" registration with `.eval_results/` format (legacy `model-index` is sufficient)
- Multi-language model cards (FR/EN)

---
*Roadmap created: 2026-05-12*
