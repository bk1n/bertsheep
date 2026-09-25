# bertsheep — TODO

`README.md` is the spec. This lists what it asks for that `src/bertsheep/` does
not do yet, at the level of "what needs building", not how.

## URGENT

- [x] Run `benchmark.py` on the real EGFR frame — seconds/epoch and peak VRAM on the 4060 decide the Optuna trial budget and whether cloud compute is needed
    - [x] benchmark.py: add benchmark of train loss vs test loss w/ autocast on + off (are speed up improvements costing performance)
- [x] data.py: 
    - [x] add parquet caching of data_path target data frames; if fetched again, use parquet rather than loading full CSV; save to out/.cache/
    - [x] add mutation argument to filter frame by mutation; use "wildtype" as wildtype arg; post-caching to parquet
- [x] eda.py: on the preprocessed EGFR frame — label distribution, scaffold/Butina cluster sizes, singleton fraction, realised split ratios
    - [x] Fix eda.py; update to latest splitting methods
    - [x] Read `MIN_CLUSTER_SIZE` (eda.py, = 10) off the split figures and pick it deliberately — the dropped share is drawn on them now, and benchmark.py hardcodes its own copy of the same number
    - [x] Open EDA questions:
        - [x] do we group mutations together or stratify analyses by mutation?
- [x] Ensure appropriate epoch-level logging of necessary elements: per epoch states (to fetch embeddings later), losses, etc.
    - [x] model.py: Build method to get embedding from the .pt model state_dict's weights
    - [x] Attention maps for Q2 are not stored; recompute from checkpoints (`output_attentions=True`) when the RSA/UMAP work needs them
- [x] benchmark.py
    - [x] Add benchmarking of autocast bf16 vs none - does it degrade performance?
- [x] data.py
    - [x] Add warning when mutation=None, stating that mutations are grouped together and may affect splitting validity
- [ ] model.py
    - [x] Restore `Model._seed` — lost in the `b236c05` auto-merge, which kept every reference to it (`MODEL_SEED`, the `seed` argument, the call at `model.py:200`, `generator=` on the train loader) but took the other branch's side of the hunk holding the definition. `Model(...)` raises `AttributeError` before it builds the network, so nothing downstream runs. The method is at `1d42f3a:src/bertsheep/model.py:259`
    - [x] Replace `valid_predictions.csv` with saved split indices — `splits.parquet` dumps all three frames per run, duplicating the preprocessed data across the 360-run grid, and the predictions CSV only ever covers the best epoch's valid split. Save the positional indices plus a method to recover the split frames from them; predictions for any split at any epoch then come from a checkpoint forward pass, which is seconds on the 4060 and the path `embeddings()` already takes
    - [x] Checkpoint every 4 epochs rather than every one — 40 MB per file is ~3.2 GB per run and ~200 GB across the 60 fine-tuned runs (frozen runs write none). `evaluate()` now restores the best epoch from memory, so thinning no longer has to keep the running best on disk
    - [x] Replace the every-4th thinning with what the figures use: every fine-tuned run writes `init.pt` + `best.pt` (the two ends RSA compares — the best epoch previously reached disk only when it landed on the interval), and only `GIF_SEED`'s two runs (in + out) keep every epoch, for the GIF. ~11 GB worst case across the 60 runs instead of ~50 GB, and the GIF gets a frame per epoch
    - [x] Record `min_cluster_size` in `config.json` — it is a split setting like `train_size` and `split_seed`, and it is the one that decides which molecules are dropped entirely. With index-only splits the frame cannot be rebuilt without it, nor without `data_path` and `mutation`, which are also unrecorded
    - [x] Implement early stopping
- [x] tuning.py
    - [x] Implement Optuna hyperparameter optimisation w/ TPESampler
    - [x] Don't include fine-tuning hyperparameters in the search yet; e.g. LLRD, reinit_n, these will be left for Q2.1 (if we have time)
    - [x] Implement complementary tuning for pretrained model
    - [x] Implement basic hyperparameter optim for XGBoost model
- [x] experiment.py
    - [x] standalone experiment() method, convert Data (target, mutation), Splitter(method, distribution), Model() from strings into actual instantiations ready to fit + run
    - [x] add tests to ensure methods output consistently everytime
    - [x] implement a suite of experiments that answer questions in README.md
    - [x] don't implement scaffold splits; Butina splits are working much better than BM scaffolds for this; doesn't need to be comprehensive
    - [x] Command-line runner: `uv run bertsheep <target>` preprocesses, tunes (resuming the SQLite study) and runs the missing grid cells; `--arms`/`--distributions`/`--seeds` narrow it for smoke runs
    - [x] `MUTATION` defaults to wildtype (11k labels on EGFR); the tuning study has to be run on the same frame, via `Experiment.tune()`
    - [x] Add caching for distance matrix on target + mutation status; saves remaking Tanimoto distance matrix every run
- [ ] results.py: results visualiser, driven off the results CSV's `run_dir` column (not a glob of `out/models/`, which also holds tuning-trial and legacy runs). Figures are headed for `results.md`/`README.md`, so they need a tracked home outside the gitignored `out/`
    - [ ] Loss curves: 1x2 grid (in / out distribution), coloured by arm, every seed's per-epoch loss as points with a LOESS trend per arm

## Question 1 — does fine-tuning help?

- [ ] Results aggregation into one tidy frame, then the four-panel figure (in/out-of-distribution R2 boxplots + fine-tuned R2 curves per epoch)

## Question 2 — latent space during fine-tuning

- [x] `Results.q2()`: aligned-UMAP GIF (distribution x embedding/middle/last layer, plus a loss column with a moving cursor) and a static filmstrip of its key frames, each coloured by affinity and by Butina cluster, over a seeded fifth of each split (`SUBSAMPLE`). Coordinates cache to `out/latent/<run>.npy`
- [ ] RSA line graphs, (a) per embedding/encoder layer and (b) per attention layer, in vs out of distribution. Attention maps are recomputed from `init.pt`/`best.pt` with `output_attentions=True`; the GIF leaves attention to this figure

## Question 2.1 — fine-tuning methods

- [ ] Compare LLRD and top-layer reinit. Both are out of the Question 1 search and default off in `Model` (`llrd_decay=1.0`, `reinit_n=0`), so the fine-tuned arm is plain AdamW. The ranges the search used before they came out: `llrd_decay` 0.7–1.0, `reinit_n` 0–1. `reinit_n` is close to binary because ChemBERTa-10M-MTR has three encoder layers. The few-sample reinit trick discards the quarter of the encoder nearest the head (Zhang et al. 2021, 6 of BERT-large's 24 layers), and a third is already past that. Reinitialising all three would throw pretraining away rather than being a fine-tuning setting

## Housekeeping

- [ ] Rotate the W&B API key still in git history
- [ ] Untrack the 12 committed `.pyc` files
- [ ] Make the BindingDB download reproducible (currently rsync from a local Windows path)
- [x] Checkpoint retention — `epoch{NNN}.pt` is written every epoch with no cap (now `init.pt` + `best.pt`, and every epoch only on `GIF_SEED`'s trajectory runs)
- [x] `Eda` UMAP split plots call the removed `Chemist.split_groups`; point them at `Splitters`
- [ ] `eda.py` is the only module in `src/` with no type hints — the methods touched by the `Splitters` port have them, the rest do not
- [ ] `eda.py`, `model.py` and `benchmark.py` each end in a `__main__` block with the target and the dump path as literals, which is the legacy pattern this repo is undoing
