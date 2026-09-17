# bertsheep
Bi-directional encoder representations from transformers (BERT) for binding affinity (ba 🐑) prediction.

Fine-tunes the pre-trained language model ([ChemBERTa](https://arxiv.org/abs/2010.09885)) on binding affinity of compounds to a single target and evaluates generalisation performance.

We select the top five most profiled _kinase_ targets from [bindingDB](https://www.bindingdb.org/).

## Question 1

> "Does fine-tuning improve binding affinity prediction to targets?"

1. Evaluate in-distribution performance of models.\
    The performance delta shows us whether the model is simply mapping scaffolds -> binding affinity, or actually learning features within scaffolds that predict binding affinity.

2. Evaluate out-of-distribution performance of models.\
    The performance delta shows us whether the model is learning features that generalise to different scaffolds.

## Question 2

> "How does fine-tuning alter the models latent space during training?"

Hypothesis: if the model is simply re-learning rough scaffolds, then fine-tuning will not alter the layers substantially.

1. How do the **embedding** layers differ between pre-trained and fine-tuned models?
2. How do the **attention** layers differ between pre-trained and fine-tuned models?

## Setup

### Models

Baseline: XGBoost on molecular fingerprints\
Pre-trained: [DeepChem/ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR)\
Fine-tuned: [DeepChem/ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR) + bindingDB

### Cross-validation

Inner loop: Optuna w/ TPESampler\
Outer loop: Evaluation w/ held-out eval set  

In-distribution: stratified train/test/eval splits by scaffolds, ensuring equal representation of scaffolds between them; repeated $s$ times.\
Out-of-distribution: train/test/eval leave-one-out-scaffold split for each $s$

Where $s$ is the number of scaffolds.

### Splitting Strategies

For both tests, we define "scaffolds" in two ways:
1. Bemis-Murcko scaffolds (less stringent; https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#scaffoldsplitter)
2. Butina-split clusters (more stringent; https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#butinasplitter)

Scaffolds 

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

