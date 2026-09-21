# bertsheep — TODO

`README.md` is the spec. This lists what it asks for that `src/bertsheep/` does
not do yet, at the level of "what needs building", not how.

## Before any real run

- [ ] Run `benchmark.py` on the real EGFR frame — seconds/epoch and peak VRAM on the 4060 decide the Optuna trial budget and whether cloud compute is needed
- [x] data.py: 
    - [x] add parquet caching of data_path target data frames; if fetched again, use parquet rather than loading full CSV; save to out/.cache/
    - [x] add mutation argument to filter frame by mutation; use "wildtype" as wildtype arg; post-caching to parquet
- [ ] eda.py: on the preprocessed EGFR frame — label distribution, scaffold/Butina cluster sizes, singleton fraction, realised split ratios
    - [x] Fix eda.py; update to latest splitting methods
    - [ ] Read `MIN_CLUSTER_SIZE` (eda.py, = 10) off the split figures and pick it deliberately — the dropped share is drawn on them now, and benchmark.py hardcodes its own copy of the same number
    - [ ] Open EDA questions:
        - [ ] do we group mutations together or stratify analyses by mutation?
- [ ] Ensure appropriate epoch-level logging of necessary elements: per epoch states (to fetch embeddings later), losses, etc.
    - [ ] model.py: Build method to get embedding from the .pt model state_dict's weights

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
- [ ] Checkpoint retention — `epoch{NNN}.pt` is written every epoch with no cap
- [x] `Eda` UMAP split plots call the removed `Chemist.split_groups`; point them at `Splitters`
- [ ] `eda.py` is the only module in `src/` with no type hints — the methods touched by the `Splitters` port have them, the rest do not
- [ ] `eda.py`, `model.py` and `benchmark.py` each end in a `__main__` block with the target and the dump path as literals, which is the legacy pattern this repo is undoing
