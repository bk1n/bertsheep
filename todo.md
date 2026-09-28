# bertsheep — TODO

`README.md` is the spec. This lists what it asks for that `src/bertsheep/` does
not do yet, at the level of "what needs building", not how.

## Question 2 — latent space during fine-tuning

- [x] `Results.q2()`: aligned-UMAP GIF (distribution x embedding/middle/last layer, plus a loss column with a moving cursor) and a static filmstrip of its key frames, each coloured by affinity and by Butina cluster, over a seeded fifth of each split (`SUBSAMPLE`). Coordinates cache to `out/latent/<run>.npy`
- [ ] RSA line graphs, (a) per embedding/encoder layer and (b) per attention layer, in vs out of distribution. Attention maps are recomputed from `init.pt`/`best.pt` with `output_attentions=True`; the GIF leaves attention to this figure

## Housekeeping 

- [ ] Rotate the W&B API key still in git history (gone from the tree since 85439be, but `f8f1626` is pushed to `origin/main`, so the key must be revoked on wandb.ai; rewriting history is optional once it is dead)
- [x] Untrack the 12 committed `.pyc` files (deleted in 85439be; `__pycache__` is gitignored)
- [x] Make the BindingDB download reproducible (currently rsync from a local Windows path) (not possible; added to README)
- [x] Remove the fine-tuning methods arm from model fitting + results - no longer used
- [x] Spelling + grammar check on README.md
- [x] Ensure all units are in -ln IC50 (nM), use RMSE throughout (figures, README and console output report RMSE; MSE stays as the training loss, the history.csv loss columns and the Optuna objective, whose persisted study would otherwise mix units; `valid_r2` is still recorded)
- [ ] Swap train/test/validation for train/validation/test 
- [x] Add/complete a Data section in README detailing EGFR selection + filtering etc
- [ ] Build a basic linear mixed effects model w/ seed as random effect for RMSE; extract p-values from fixed effects comparisons
- [ ] To the boxplots, add points coloured/filled by the median tanimoto similarity of training+validation sets
- [ ] GitHub actions + pytest on every push plus a badge
- [ ] If we switched targets, mutations, etcetera - would this repo function as expected?
- [x] Checkpoint retention — `epoch{NNN}.pt` is written every epoch with no cap (now `init.pt` + `best.pt`, and every epoch only on `GIF_SEED`'s trajectory runs)
- [x] `Eda` UMAP split plots call the removed `Chemist.split_groups`; point them at `Splitters`
- [ ] `eda.py` is the only module in `src/` with no type hints — the methods touched by the `Splitters` port have them, the rest do not
- [ ] `eda.py`, `model.py` and `benchmark.py` each end in a `__main__` block with the target and the dump path as literals, which is the legacy pattern this repo is undoing
