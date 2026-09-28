# bertsheep
Bi-directional encoder representations from transformers (BERT) for binding affinity (ba 🐑) prediction.
Fine-tunes the pre-trained language model ([ChemBERTa](https://arxiv.org/abs/2010.09885)) on [binding affinity](https://www.bindingdb.org/) of compounds to a single target (EGFR) and evaluates generalisation performance. 

Part of my work at Accenture Labs, refactored with Claude Code.

## Running
Run the tests: `uv run pytest`
Train the models: `uv run bertsheep EGFR`
Visualise the results: `uv run python -m bertsheep.results <path>`

## Setup
### Data
Download and unzip the `BindingDB_All_202609_tsv` from bindingDB into `data/` -> `data/BindingDB_All_202609_tsv/BindingDB_All.tsv`

### Model
Model: [DeepChem/ChemBERTa-10M-MTR](https://huggingface.co/DeepChem/ChemBERTa-10M-MTR)

Pre-trained: frozen model + trained regression head
Fine-tuned: unfrozen model + trained regression head

### Splitting Strategies

Compounds are split into Butina clusters (a relatively stringent clustering method).

### Cross-validation

!!! Not cross validation really

Trained locally on an RTX 4060 with Optuna + TPESampler; hyperparameters are found on seed zero and shared across runs (due to computational limitations).

In-distribution: for Butina splits, we use stratified splitting by clusters to ensure equal representation of clusters.
Out-of-distribution: group shuffled splits ensure some clusters are held-out.

### Training

Trained locally on an RTX 4060. Made possible with approx. 2x speedup with autocast bf16.

## Results

EGFR wild type, Butina clusters (minimum size 10), 30 seeds per arm and distribution. A seed resamples both the split and the model, and every arm sees the same split for a given seed, so arms are compared seed by seed.

### Optimal hyperparameters per-arm

Tuned with Optuna (TPE) on seed 0's split, scored on test, validation never read. Trials: XGBoost 100, pretrained 50, fine-tuned 50.

| Arm | Distribution | lr | weight decay | warmup | dropout |
|---|---|---|---|---|---|
| Pretrained | in | 7.6e-3 | 0.018 | 0.042 | 0.097 |
| Pretrained | out | 9.3e-3 | 0.131 | 0.074 | 0.138 |
| Fine-tuned | in | 2.4e-4 | 0.150 | 0.054 | 0.050 |
| Fine-tuned | out | 4.9e-4 | 0.224 | 0.026 | 0.065 |

| XGBoost | max depth | learning rate | subsample | colsample | min child weight | L2 |
|---|---|---|---|---|---|---|
| in | 10 | 0.015 | 0.96 | 0.25 | 1.1 | 1.23 |
| out | 10 | 0.070 | 0.92 | 0.85 | 6.6 | 0.005 |


### Model performance

![Question 1: validation R² per arm (left) and loss curves (right), in- and out-of-distribution](figures/EGFR-wildtype_q1.png)

Means over 30 seeds; RMSE in log10 units.

**ChemBERTa learns more than cluster identity.** 
In-distribution performance of all models was elevated compared to the cluster mean, suggesting the models are successfully learning features of the compounds that predict binding to EGFR. 

**XGBoost on fingerprints beats pre-trained and fine-tuned ChemBERTa.** 
In-distribution and out-of-distribution performance is higher for XGBoost trained on fingerprints versus ChemBERTa trained on SMILES. Other authors have observed [this](https://www.nature.com/articles/s41467-023-41948-6) and many reasons for weaker performance exist, including negative transfer and limited dataset size.

**Fine-tuning exhibits similar performance to pre-trained model.**
Fine-tuning shows similar performance to the pre-trained model with a small advantage in-distribution (fine-tuned beats pre-trained 25/30 seeds). 
This is potentially due to the MTR pre-training method, which regresses ~200 RDKit descriptors, already learning affinity-relevant descriptors in the frozen latent space.

**Out-of-distribution, which clusters are held out matters more than which model is used.** The seeds are strongly correlated across arms. Seed 24 is the worst seed for both XGBoost (−0.06) and fine-tuning (−0.07), and seed 17 is the best for XGBoost, pretrained and fine-tuned (0.57, 0.47, 0.50). The spread across seeds (SD ≈ 0.13 R²) is about twice the gap between the best and worst model. The validation set also ranges from 708 to 2,250 molecules because clusters vary in size. A single held-out split would give a misleading ranking of these models.

Wide performance range out-of-distribution suggests that some clusters may be easier to predict than others; the performance tracks across arms by seed (i.e. some seeds are easier to predict than others out-of-distribution).

### Latent space with fine-tuning

![Question 2: aligned UMAP of the fine-tuned model's latent space over training, coloured by affinity](figures/EGFR-wildtype_q2_label.gif)
![Question 2: aligned UMAP of the fine-tuned model's latent space over training, coloured by cluster](figures/EGFR-wildtype_q2_cluster.gif)

- **The embedding layer barely moves.** In both distributions its layout is nearly the same at the pretrained weights and at the last epoch.
Fine-tuning changes the encoder layers, not the token embeddings.
- **The last encoder layer reorganises around affinity.** In-distribution, weak binders (dark) collect on one side of encoder layer 3 and potent ones (yellow/orange) on the other by the selected epoch. The same starts to happen out-of-distribution.
- **In-distribution, validation molecules move with their training neighbours.** In the above figures, clusters are coloured if they are part of the validation set. During in-distribution training, these clusters are associated with training compounds and so the valdiation clusters are carried along into the affinity-sorted layout.
- **Out-of-distribution, the training molecules rearrange while the held-out molecules stay close to where they started.** 
During out-of-distribution training, the training molecules are still re-organised in the last layer to an affinity-sorted layout. In contrast, the validation clusters remain relatively static, moving little and not following the training molecules. The model is not able to learn generalisable features that transfer to these clusters. This highlights the poor performance on out-of-distribution compounds and weak generalisation.