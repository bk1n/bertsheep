# bertsheep
Bi-directional encoder representations from transformers (BERT) for binding affinity (ba 🐑) prediction.

Fine-tunes the pre-trained language model ([ChemBERTa](https://arxiv.org/abs/2010.09885)) on binding affinity of compounds to a single target and evaluates generalisation performance.

Part of my work at Accenture Labs.

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

### Question 2.1

How do fine-tuning methods (e.g. reinit_n and LLRD) affect the embedding layers?\
Future focus (if we have time to implement)

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

Due to computational limitations (this is trained locally on an RTX 4060), we run Optuna w/ TPESampler once, on the first seed's split, optimising hyperparameters on train + test data with the evaluation set held out. The remaining seeds reuse those hyperparameters.

In-distribution: for both Bemis-Murcko and Butina splits, we use stratified splitting by "scaffolds" to ensure equal representation of scaffolds.

Out-of-distribution: for both, group shuffled splits ensure some scaffolds are held-out.

Groups smaller than a minimum size are dropped from every split: a cluster with one member cannot be represented on both sides of an in-distribution split, and is not a series an out-of-distribution one can learn from.

All experiments are repeated 30 times. A repeat is a new seed on both the splitter and the model, so head initialisation and batch order are resampled alongside the chemistry held out.

## Figures
### Question 1
Multi-panel:\
(a) boxplot: x: model (baseline, pre-trained, fine-tuned), y: R2, color: in vs out-of-distribution\
(b) R2-curve for fine-tuned model: x: epoch, y: R2, color: in vs out-of-distribution

## Question 2
Multi-panel:\
(a) line-graph: x: embedding layer, y: RSA, color: in vs out-of-distribution\
(b) line-graph: x: attention layer, y: RSA, color: in vs out-of-distribution

Multi-panel GIF (2x3):\
x: first layer, middle layer, bottom layer\
y: in-distribution, out-of-distribution

> GIF creation:
> 1. Capture each (a) embedding layer and (b) attention layer at each fine-tuning epoch.
> 2. Visualise each w/ aligned UMAP.
> 3. Convert to GIF.

### Results Interpretation

Training compounds move with validation compounds.
In both in-distribution + out-of-distribution, training error is low. 
We can see from in-distribution UMAP, that (a) the latent space is reorganised around affinity, and (b) the validation molecules move in latent space with their training molecules. In out-of-distribution, we can see that (a) the latent space is **still** reorganised around affinity, but (b) training molecules are moving a lot whilst the validation molecules are remaining in similar locations. 
As the out-of-distribution clusters had no training molecules, nothing is causing these to move in affinity space, suggesting the model is struggling to generalise to out-of-distribution compounds.

This also shows us that the models are capable of predicting in-distribution affinity.
Question: can the model predict in-distribution affinity better than simple cluster-mean? If so, it suggests the model is learning more than basic Butina clusters, learning something about chemical space that generalises across in-distribution compounds. If no, the model is simply relearning Butina-like clusters in the data and mapping those to affinity.

