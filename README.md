# bertsheep
[![tests](https://github.com/bk1n/bertsheep/actions/workflows/tests.yml/badge.svg)](https://github.com/bk1n/bertsheep/actions/workflows/tests.yml)

Bidirectional encoder representations from transformers (BERT) for binding affinity (ba 🐑) prediction.
Fine-tunes the pre-trained language model ([ChemBERTa](https://arxiv.org/abs/2010.09885)) on [binding affinity](https://www.bindingdb.org/) of compounds to a single target and evaluates generalisation performance.

Part of my work at Accenture Labs, refactored with Claude Code.

**[See the results](https://bk1n.github.io/bertsheep/)**, with figures, data and methods.

## Running
Run the tests: `uv run pytest`\
Train the model:
```
uv run bertsheep TARGET [--mutation MUTATION] [--train] [--results]
                        [--arms ARM ...] [--distributions {in,out} ...] [--seeds N]
```

e.g. the following will fine-tune a model on EGFR (wildtype) and generate results:
```
uv run bertsheep EGFR --train --results
```

e.g. whilst the following will do the same for BRAF with V600E mutation:
```
uv run bertsheep BRAF --mutation V600E --train --results
```

Four targets, including EGFR, are currently registered for model training:

| Command | Ligands | Butina clusters of 10+ | Example mutation |
|---|---|---|---|
| EGFR | 11,056 | 219 | L858R,T790M |
| JAK2 | 10,804 | 158 |  |
| BRAF | 3,048 | 37 | V600E |
| LRRK2 | 1,458 | 28 | G2019S |

## Setup
Download and unzip `BindingDB_All_202609_tsv` from BindingDB into `data/`.
The target frame is cached to `out/.cache/` after the first scan of the dump, so the first run takes a few minutes.
