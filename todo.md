# bertsheep — TODO

## Compute strategy (DDP → cloud / local GPU)

- [ ] Decide compute target: previous multi-GPU DDP setup is no longer available and any cloud spend is self-funded
- [ ] Make the code device-agnostic — single-GPU run should not require `torchrun`/DDP (`run_ddp.py` is the only entry point today)
- [ ] Benchmark one epoch on the local RTX 4060 before committing to paid compute
- [ ] If going cloud: compare on-demand vs spot (Vast.ai / RunPod / Lambda) and set a hard budget cap
- [ ] Add checkpoint resume so a preempted spot instance doesn't lose the run (ties into the `saveCheckpoint` fix below)
- [ ] Cloud runs need the data reproducible from source — blocked on the missing preprocessing script below

## Security

- [ ] Rotate the W&B API key (it's committed, so it's in git history)
- [ ] `modules/wandb_log.py` — read key from `WANDB_API_KEY` env var instead of hardcoding

## Repo hygiene

- [ ] Add `.gitignore` (`__pycache__/`, `*.pyc`, `models/`, `wandb/`)
- [ ] Untrack the 12 committed `.pyc` files, incl. stale `modules/__pycache__/model.cpython-310.pyc`
- [ ] Add `requirements.txt` / env file (torch, transformers, datasets, rdkit, umap-learn, hdbscan, bertviz, wandb, scikit-learn, scipy, seaborn)
- [ ] Pick one import convention (`from modules import ...` everywhere, run from repo root) — `run_ddp.py` currently can't import `model.py`
- [ ] Move hardcoded config out of `__main__` blocks into CLI args or a config file
- [ ] De-duplicate the hyperparameter dict — it exists in both `model.py` and `run_ddp.py` and has already drifted

## Data download + preprocessing

- [ ] Write the missing script that turns raw BindingDB into `bindingDB_<TARGET>_HUMAN_<UNIPROT>.csv` (never committed — data is currently unreproducible)
- [ ] Add a download step for the BindingDB source dump
- [ ] Output contract: one CSV per target with `SMILES` and `Ki` (nM) columns
- [ ] Targets referenced in code: DRD2 P14416, OPRK P41145, OPRM P35372, JAK2 O60674, CAH2 P00918
- [ ] Replace `/data/ben/bindingDB/processed_data/` absolute paths with a configurable data root
- [ ] Document the label transform (`-np.log(Ki)`) — decide whether to use standard pKi (`-log10` of molar)

## `model.py` — `Model.train()` / train loop

- [ ] Add `self.model.train()` at the top of each epoch and `self.model.eval()` before the test loop
- [ ] Split the loop into `_train_epoch()` / `_evaluate()` so mode is set per-operation, not by call order
- [ ] Rename `Model.train()` → `fit()` to stop it colliding with `nn.Module.train()`
- [ ] Remove the `and False` that hard-disables early stopping (line ~216)
- [ ] Compute `test_loss` over the whole test set, not just the last batch (line ~215)
- [ ] Consider `torch.inference_mode()` in place of `no_grad()`

## `model.py` — `Model.__init__` / checkpointing

- [ ] Add `self.model.eval()` at the end of `__init__` so inference-only subclasses default to eval
- [ ] Break up the ~75-line `__init__`; move config unpacking into a dataclass
- [ ] `saveCheckpoint` — track best score instead of overwriting on any r2 > 0.6

## DDP (`model.py` + `run_ddp.py`)

- [ ] Gather preds/labels across ranks before computing metrics — rank 0 currently logs only its own shard
- [ ] Switch backend from `gloo` to `nccl` for GPU training
- [ ] Call `sampler.set_epoch(epoch)` each epoch
- [ ] Replace the bare `except` + `input('Press Enter to Retry')` in `run_ddp.py` (blocks unattended runs)

## `modules/data.py`

- [ ] `getTrainTestSplit` — use `elif` and add `else: raise` so a bad method name fails loudly
- [ ] `generate_scaffolds` — drop the discarded `.apply()` on line 65 (computes scaffolds twice)
- [ ] Move the `alternateProteinSplit` paths out of the function body into config
- [ ] Clean up the `__index_level_0__` / `index` column left by the dedup concat
- [ ] Decide on the commented-out `MinMaxScaler` — currently dead but the model head assumes it
- [ ] `generate_scaffold` — switch `MurckoScaffoldSmiles` to `includeChirality=False` (DeepChem default): stereoisomers currently get distinct scaffolds, so an enantiomer pair can straddle train/test with near-identical Ki
- [ ] `generate_scaffold` — handle the `''` scaffold returned for acyclic molecules; they all share one bucket that sorts near the front and goes wholesale into train
- [ ] `generate_scaffold` — no try/except, unlike `can()` on line 32; one RDKit failure kills the run

## Splitting + evaluation methodology

- [ ] Make the split three-way (train/valid/test) — currently two-way, and `test_df` doubles as validation
- [ ] Select on validation, not test: `saveCheckpoint` fires on test r2 (~line 229) and the early stopper consumes `test_loss` (~line 218), while `run_ddp.py` sweeps hyperparameters over the same set — the scaffold split's OOD estimate is spent on model selection (ties into the `saveCheckpoint` and early-stopping fixes above)
- [ ] Add seeded/balanced scaffold-split replicates — the split is a single deterministic partition, so results are a point estimate with no variance, and scaffold-split scores are noisy
- [ ] Report which split method produced each number in the README/figures — random vs scaffold vs fingerprint are not comparable

## `modules/models.py`

- [ ] `CustomClassifier` — three separate `nn.Linear` layers instead of reusing `self.dense` three times
- [ ] Remove the final `torch.tanh` (clamps output to [-1,1], labels are unbounded) or restore label scaling

## `modules/visualisation.py`

- [ ] `getAttWeights` — add `eval()` and `no_grad()`; committed attention heatmaps were made with dropout on
- [ ] `get_hidden_states` — add `eval()`; latent-space figures were made with dropout on
- [ ] Regenerate `figures/attention_heatmaps/` and `figures/latent_space/` after the above
- [ ] Stop subclassing `Model` for plotting — take a trained model as an argument instead
- [ ] Fix save paths: code writes `./data/figures/`, repo has `./figures/`
UMAP: figure out how to generate updated/aligned UMAPs
Robust method for considering cluster consistency/overlap between cluster + 

## `modules/clustering.py`

- [ ] `plotClusterMap` — use `self.getFingerPrints`, not the module-level `c`
- [ ] Fix `./data/figures/` save paths (tsne, umap, heatmaps)
- [ ] Make plot titles a parameter instead of commenting/uncommenting them between runs

## Testing

- [ ] Add tests for the three split methods (no leakage between train/test)
- [ ] Add a smoke test that trains a few steps on a tiny fixture CSV
