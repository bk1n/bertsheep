# bertsheep

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state: mid-rewrite

This repo is being converted from a 2023 research project (transferred from GitLab) into a clean, reproducible one. Three things are deliberately out of sync — do not treat any of them as a bug to "fix" opportunistically:

- `README.md` describes the **intended** experimental design (five kinase targets, XGBoost baseline, Optuna nested CV, in- vs out-of-distribution scaffold splits, aligned-UMAP latent-space figures). Almost none of it is implemented in the new code yet. It is a spec, not a description.
- Root-level `model.py`, `run_ddp.py` and `modules/` are the **legacy** codebase. They are kept as reference for the rewrite, are not imported by `src/bertsheep/`, and are not expected to run as-is (hardcoded `/data/ben/...` paths, a committed W&B key, broken imports). Read them for prior art; don't extend them.
- `figures/`, `model_preds/` and `data/target_counts.csv` are outputs from the legacy runs.

`todo.md` is the migration backlog and the closest thing to a source of truth for known defects (legacy bugs, methodology problems, compute plans). Check it before proposing work — most obvious issues are already logged there, often with the reasoning. Keep it updated as items are done.

New work goes in `src/bertsheep/`.

## Commands

The project uses `uv` (Python 3.14, `uv_build` backend, src layout).

```bash
uv sync                      # create/refresh .venv from uv.lock
uv run bertsheep             # console script -> bertsheep:main (currently a stub)
uv run python -c "..."       # run anything against the project env
uv add <pkg>                 # add a dependency (updates pyproject.toml + uv.lock)
./fetch.sh                   # rsync the BindingDB dump from Windows into ./data
```

There is no test suite, linter config, or CLI entry point beyond the stub yet. `ipykernel` is a dependency: exploratory work is expected to happen in a notebook/REPL against the installed package.

Typical interactive use, since `Data` has no public runner yet:

```python
from bertsheep.data import Data
from bertsheep.eda import Eda
d = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", "EGFR")
df = d._preprocess()
Eda(df, "EGFR").label_histogram()
```

## Data

`data/` is gitignored and is **not** reproducible from the repo — the raw BindingDB dump (`data/BindingDB_All_202609_tsv/BindingDB_All.tsv`, ~8.4 GB, 640 columns) is rsynced from a Windows path given by `DATA_PATH` in `.env`, via WSL (`wslpath` + `rsync`). Both `fetch.sh` and `Data._fetch_data` do this; they are duplicates of each other.

Because the source is large, `Data._load` reads it in chunks (`CHUNK_SIZE`) with `usecols=list(COLUMNS)` and `dtype=str`, filtering each chunk down to the target before concatenating. Any change to the load path must preserve that — never read the TSV whole.

## Architecture (`src/bertsheep/`)

- **`data.py`** — `Data` owns the raw-dump → training-frame pipeline: `_load` → `_filter` → `_deduplicate` → `_canonicalise` → `_transform_labels`, orchestrated by `_preprocess`. Each stage is a small method taking and returning a dataframe, so stages can be run and inspected individually. Module-level constants (`LABEL`, `MAX_SMILES_LENGTH`, `CHUNK_SIZE`, `COLUMNS`, `TARGET`) hold everything configurable; `TARGET` maps a short name to `(UniProt entry name, primary ID)` and is currently `EGFR` only, whereas the README and legacy code assume five targets.
- **`eda.py`** — `Eda` wraps a preprocessed `(smiles, labels)` frame and writes figures to `figures/<target>_<name>.png`. Plots are thin wrappers over pandas/matplotlib built-ins; see "Use the library" below.
- **`chemistry.py`** — empty placeholder for the scaffold/fingerprint/splitting logic being ported from `modules/data.py` (Murcko scaffolds, Butina/fingerprint clustering, split methods).

Label semantics are in flux and are the thing most likely to trip you up: the label column is selected by the `LABEL` constant and is currently **IC50** (nM) transformed with `-np.log`, i.e. natural log, not standard pKi (`-log10` of molar). The README talks about Ki. `MAX_KI_NM` exists but the affinity cutoff is currently not applied. Don't silently change any of this — it affects every number downstream.

## Legacy architecture (root + `modules/`), for porting reference

`model.py` defines `Model`, the orchestrator that wires together the pieces in `modules/`: `models.py` (HF ChemBERTa wrappers, custom regression head, activation hooks), `data.py` (`DataManager`: preprocessing, scaffold/fingerprint/random splits, HF `datasets` + `DataLoader` construction), `optimisers.py` (plain AdamW and layerwise-LR-decay variants), `loss.py`, `earlystop.py`, `wandb_log.py`. `visualisation.py` subclasses `Model` to reach the trained network for attention/latent-space figures, and `clustering.py` provides fingerprints, Tanimoto distance, t-SNE/UMAP/HDBSCAN.

`run_ddp.py` is the only entry point: it sweeps a hyperparameter dict and `mp.spawn`s `main_ddp` across GPUs. The multi-GPU machine is gone, so the rewrite targets a single local GPU (RTX 4060) or paid cloud — device-agnostic code, no `torchrun` requirement.

Note that `torch`, `transformers`, `datasets`, `wandb`, `umap-learn` and `hdbscan` are **not** in `pyproject.toml`. The new package is deliberately data/cheminformatics-only so far; adding the deep-learning stack back is a deliberate step, not a fix to slip into an unrelated change.

## Use the library, don't reimplement it

Always reach for what the installed packages already do before writing your own version of it. Prefer the shortest call that gets there.

- Plotting: `pandas` `.hist()` / `.plot()`, or `matplotlib` `ax.hist()`. Don't hand-roll binning, styling helpers, or save wrappers.
- Binning: `bins="auto"` — numpy already picks a sensible rule. Don't implement Freedman-Diaconis or Sturges by hand.
- Stats and summaries: `df.describe()`, `.value_counts()`, `.groupby()`.
- Chemistry: `rdkit`. Splits and metrics: `scikit-learn`.

If a task looks like it needs 100 lines, check whether a package already does it in 5. It usually does.

## Conventions

- British spellings in identifiers, matching the existing code and the author: `canonicalise`, `optimisers`, `visualisation`, `colour`.
- In `src/`, pipeline stages are single-underscore methods on a class, each with a short docstring explaining *why* the step is shaped that way (e.g. why chunked, why median-collapsed), not just what it does. Comments in this codebase carry non-obvious reasoning — keep that density.
- Tunables go in module-level `UPPER_CASE` constants, not inline literals or `__main__` blocks (the legacy code did the latter, and `todo.md` tracks undoing it).
- Keep methods short and obvious; no defensive branching for cases that haven't come up.
- Docstrings say why, not what — the code already says what.
