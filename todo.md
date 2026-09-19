# bertsheep — TODO

`README.md` is the spec. This lists what it asks for that `src/bertsheep/` does
not do yet, at the level of "what needs building", not how.

## Before any real run

- [ ] Run `benchmark.py` on the real EGFR frame — seconds/epoch and peak VRAM on the 4060 decide the Optuna trial budget and whether cloud compute is needed
- [ ] EDA on the preprocessed EGFR frame — label distribution, scaffold/Butina cluster sizes, singleton fraction, realised split ratios
- [ ] Settle the open data questions that EDA answers: affinity cutoff (`MAX_KI_NM` is unused), and whether splits must group by ligand across mutations

## Question 1 — does fine-tuning help?

- [ ] Three model arms scored on identical splits: XGBoost fingerprint baseline, pre-trained (frozen encoder + trained head), fine-tuned
- [ ] Optuna (TPESampler) hyperparameter search, once per seed, tuned on train + test, reported on held-out valid
- [ ] Experiment runner: arm x split method (scaffold, Butina, fingerprint) x distribution (in, out) x 30 seeds, one results row per run
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
- [ ] `Eda` UMAP split plots call the removed `Chemist.split_groups`; point them at `Splitters`
