# bertsheep — TODO

`README.md` is the spec. This lists what it asks for that `src/bertsheep/` does
not do yet, at the level of "what needs building", not how.

## Before any real run

- [ ] Run `benchmark.py` on the real EGFR frame — seconds/epoch and peak VRAM on the 4060 decide the Optuna trial budget and whether cloud compute is needed
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
    - [ ] Restore `Model._seed` — lost in the `b236c05` auto-merge, which kept every reference to it (`MODEL_SEED`, the `seed` argument, the call at `model.py:200`, `generator=` on the train loader) but took the other branch's side of the hunk holding the definition. `Model(...)` raises `AttributeError` before it builds the network, so nothing downstream runs. The method is at `1d42f3a:src/bertsheep/model.py:259`
    - [ ] Replace `valid_predictions.csv` with saved split indices — `splits.parquet` dumps all three frames per run, duplicating the preprocessed data across the 360-run grid, and the predictions CSV only ever covers the best epoch's valid split. Save the positional indices plus a method to recover the split frames from them; predictions for any split at any epoch then come from a checkpoint forward pass, which is seconds on the 4060 and the path `embeddings()` already takes
    - [x] Checkpoint every 4 epochs rather than every one — 40 MB per file is ~3.2 GB per run and ~1 TB across the grid. Early stopping does not land on a multiple of 4, so `evaluate()` will ask for a `best_epoch` checkpoint that was never written: keep epoch -1 and the running best as well as the every-fourth ones
    - [ ] Record `min_cluster_size` in `config.json` — it is a split setting like `train_size` and `split_seed`, and it is the one that decides which molecules are dropped entirely. With index-only splits the frame cannot be rebuilt without it, nor without `data_path` and `mutation`, which are also unrecorded
    - [ ] Implement early stopping
- [ ] tuning.py
    - [ ] Implement Optuna hyperparameter optimisation w/ TPESampler
- [ ] experiment.py
    - [ ] standalone experiment() method, convert Data (target, mutation), Splitter(method, distribution), Model() from strings into actual instantiations ready to fit + run
    - [ ] add tests to ensure methods output consistently everytime
    - [ ] implement a suite of experiments that answer questions in README.md
    - [ ] don't implement scaffold splits; Butina splits are working much better than BM scaffolds for this; doesn't need to be comprehensive

## Question 1 — does fine-tuning help?

- [ ] Three model arms scored on identical splits: XGBoost fingerprint baseline, pre-trained (frozen encoder + trained head), fine-tuned
- [ ] Optuna (TPESampler) hyperparameter search, once per seed, tuned on train + test, reported on held-out valid
- [ ] Experiment runner: arm x split method (scaffold, Butina) x distribution (in, out) x 30 seeds, one results row per run — 360 runs. Fingerprint splitting is gone: its greedy deal is deterministic, so its 30 repeats would have resampled the model rather than the chemistry, and its boxplot would not have been measuring what the other two were
- [ ] Results aggregation into one tidy frame, then the four-panel figure (in/out-of-distribution R2 boxplots + fine-tuned R2 curves per epoch)

## Question 2 — how does fine-tuning change the latent space?

- [ ] Define RSA (pre-trained vs fine-tuned representational similarity per layer)
- [ ] Capture embedding and attention-layer activations per fine-tuning epoch
- [ ] RSA line plots per layer, in vs out-of-distribution
- [ ] Aligned-UMAP GIF (first/middle/last layer x in/out), one per splitting strategy

## Housekeeping

- [ ] Rotate the W&B API key still in git history
- [ ] Untrack the 12 committed `.pyc` files
- [ ] Make the BindingDB download reproducible (currently rsync from a local Windows path)
- [x] Checkpoint retention — `epoch{NNN}.pt` is written every epoch with no cap (now `init.pt`, every `CHECKPOINT_EVERY`-th epoch and the running best only: at most `num_epochs / 4 + 2` files a run)
- [x] `Eda` UMAP split plots call the removed `Chemist.split_groups`; point them at `Splitters`
- [ ] `eda.py` is the only module in `src/` with no type hints — the methods touched by the `Splitters` port have them, the rest do not
- [ ] `eda.py`, `model.py` and `benchmark.py` each end in a `__main__` block with the target and the dump path as literals, which is the legacy pattern this repo is undoing
