# bertsheep
Bi-directional encoder representations from transformers (BERT) for binding affinity (ba 🐑) prediction.

Fine-tunes the pre-trained language model ([ChemBERTa](https://arxiv.org/abs/2010.09885)) on binding affinity of compounds to a single target and evaluates generalisation performance.

We are interested in the most profiled kinase target in [bindingDB](https://www.bindingdb.org/): EGFR.

## Question 1

> "Does fine-tuning improve binding affinity prediction to targets?"

1. Evaluate in-distribution performance of models.\
    The performance delta shows us whether the model is simply mapping scaffolds -> binding affinity, or actually learning features within scaffolds that predict binding affinity.

2. Evaluate out-of-distribution performance of models.\
    The performance delta shows us whether the model is learning features that generalise to different scaffolds.

## Question 2

> "How does fine-tuning alter the models latent space during training?"

Hypothesis: if the model is simply re-learning rough scaffolds, then fine-tuning will not alter the layers substantially.

How do the **embedding** layers differ between pre-trained and fine-tuned models?

## Setup

### Model

Model: [DeepChem/ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR)

Pre-trained: frozen model + trained regression head
Fine-tuned: unfrozen model + trained regression head

### Splitting Strategies

For both tests, we split compounds in two ways:
1. Bemis-Murcko scaffolds (less stringent)
2. Butina-split clusters (more stringent)

See [here](https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#scaffoldsplitter) for more detail.

A third strategy, fingerprint splitting, was specified and then dropped. The
greedy Tanimoto deal is deterministic: one set of molecules gives one split,
whatever the seed. Its 30 repeats would therefore have resampled the model and
not the chemistry held out, so its spread would have measured something
different from the other two strategies' while sitting in the same figure
beside them. Dropping it also returns a third of the run budget, which is the
binding constraint on a 4060. Butina clustering already covers holding out
whole regions of chemical space.

### Cross-validation

Due to computational limitations (this is trained locally on an RTX 4060), we will sample run Optuna w/ TPESampler once per seed, optimising hyperparameters on train + test data, with held-out evaluation set, and share these hyperparameters across runs.

In-distribution: for both Bemis-Murcko and Butina splits, we use stratified splitting by "scaffolds" to ensure equal representation of scaffolds.

Out-of-distribution: for both, group shuffled splits ensure some scaffolds are held-out.

Groups smaller than a minimum size are dropped from every split: a cluster with one member cannot be represented on both sides of an in-distribution split, and is not a series an out-of-distribution one can learn from.

All experiments are repeated 30 times. A repeat is a new seed on both the splitter and the model, so head initialisation and batch order are resampled alongside the chemistry held out.

## Figures
### Question 1
Multi-panel:\
(a) in-distribution boxplot: x: model (baseline, pre-trained, fine-tuned), y: R2, color: scaffold-split strategy\
(b) in-distribution loss-curve for fine-tuned model: x: epoch, y: R2, color: scaffold-split strategy\
(c) out-of-distribution boxplot: x: model (baseline, pre-trained, fine-tuned), y: R2, color: scaffold-split strategy\
(d) out-of-distribution loss-curve for fine-tuned model: x: epoch, y: R2, color: scaffold-split strategy

## Question 2
Multi-panel:\
(a) line-graph: x: embedding layer, y: RSA, color: in vs out-of-distribution\
(b) line-graph: x: attention layer, y: RSA, color: in vs out-of-distribution

Multi-panel GIF (2x3):\
x: first layer, middle layer, bottom layer\
y: in-distribution, out-of-distribution

Two separate figures: one for each scaffold-splitting strategy.

> GIF creation:
> 1. Capture each (a) embedding layer and (b) attention layer at each fine-tuning epoch.
> 2. Visualise each w/ aligned UMAP.
> 3. Convert to GIF.

