from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

import bertsheep.model as bm
from bertsheep.model import Model
from bertsheep.splitters import Splitters

# Model's hyperparameters are constructor arguments rather than module
# constants, so the values every test runs against live here: MODEL_LINK is
# both what the fixture fine-tunes and what the skip guard looks for in the
# Hugging Face cache, which keeps the two from drifting apart.
MODEL_LINK = "DeepChem/ChemBERTa-10M-MTR"
BATCH_SIZE = 8
NUM_EPOCHS = 3

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(MODEL_LINK, "config.json"), str),
    reason=f"{MODEL_LINK} is not in the local Hugging Face cache",
)

# Forty ring-bearing molecules, from single rings up to the EGFR inhibitors, so
# the tokeniser sees a realistic spread of lengths. With BATCH_SIZE 8 the 70%
# train split is four batches, enough for the scheduler to have steps to count.
# They carry about ten Bemis-Murcko scaffolds between them, so a scaffold split
# has groups to stratify on rather than forty singletons.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc2[nH]cc(CCN(C)C)c2c1", "c1ccc2c(c1)cc[nH]2", "Cc1ccccc1",
    "c1ccc(cc1)C(=O)O", "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "c1ccc2ncccc2c1",
    "C1CCNCC1", "C1COCCN1", "c1cnc2ccccc2n1",
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "CS(=O)(=O)CCNCc1ccc(o1)-c1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1",
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(n1)-c1cccnc1",
    "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1",
    "c1ccc(cc1)-c1ccccc1", "O=C(O)c1ccccc1O", "c1ccsc1", "c1ccoc1", "c1cc[nH]c1",
    "c1ncc[nH]1", "C1CCCCC1", "OC1CCCCC1", "c1ccc2ccccc2c1", "Clc1ccccc1",
    "NC(=O)c1ccc(F)cc1", "CC(C)(C)c1ccccc1", "COc1ccccc1", "CN1CCN(CC1)c1ccccc1",
    "O=C1CCCN1", "c1ccc2[nH]ncc2c1", "Nc1ncnc2[nH]cnc12", "O=c1cc[nH]c(=O)[nH]1",
    "CC1=CC(=O)C=CC1=O", "c1ccc(Nc2ncccn2)cc1",
]


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Copy every tensor in a module's state dict, so later training steps can be
    compared against it.

    Parameters
    ----------
    module : torch.nn.Module
        Module to copy.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached copies keyed by state-dict name.
    """
    return {k: v.clone() for k, v in module.state_dict().items()}


def _unchanged(module: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    """
    Check whether a module still holds exactly the tensors in a snapshot.

    Parameters
    ----------
    module : torch.nn.Module
        Module to compare, with the same structure as the snapshot's source.
    snapshot : dict[str, torch.Tensor]
        Output of _snapshot().

    Returns
    -------
    bool
        True if every tensor is bitwise equal.
    """
    return all(torch.equal(v, snapshot[k]) for k, v in module.state_dict().items())


@pytest.fixture
def frame() -> pd.DataFrame:
    """A tiny training frame in Data's output contract, with seeded labels."""
    labels = np.random.default_rng(0).normal(6, 1, len(SMILES))
    return pd.DataFrame({"smiles": SMILES, "mutations": "", "labels": labels})


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: pd.DataFrame) -> Model:
    """
    A Model on CPU with small batches and few epochs, writing into tmp_path so
    runs never touch out/ or collide with each other. The splitter is built over
    the fixture frame's own SMILES, in frame order, because it deals positional
    indices that Model applies to that frame with iloc.
    """
    monkeypatch.setattr(bm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(bm, "DEVICE", torch.device("cpu"))
    torch.manual_seed(0)
    return Model(
        frame,
        "TEST",
        Splitters(frame["smiles"], "scaffold", "in"),
        model_link=MODEL_LINK,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )


def test_parameter_groups_cover_each_parameter_once(model: Model) -> None:
    """
    Every parameter is in exactly one group, only biases and LayerNorm weights
    skip weight decay, and each depth trains at llrd_decay x the one above.
    """
    groups = model._parameter_groups()
    grouped = [id(p) for g in groups for p in g["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.model.parameters())

    names = {id(p): n for n, p in model.model.named_parameters()}
    for g in groups:
        undecayed = all(any(k in names[id(p)] for k in bm.NO_DECAY) for p in g["params"])
        assert undecayed == (g["weight_decay"] == 0.0)

    # Groups come in (decayed, undecayed) pairs per depth, head first.
    lrs = [g["lr"] for g in groups[::2]]
    assert lrs[0] == model.lr
    assert np.allclose(np.divide(lrs[1:], lrs[:-1]), model.llrd_decay)


def test_reinit_resets_only_the_top_layers(model: Model) -> None:
    """
    The top n encoder layers get fresh weights; lower layers and embeddings keep
    their pretrained ones.
    """
    layers = model.model.roberta.encoder.layer
    before = [_snapshot(layer) for layer in layers]
    embeddings = _snapshot(model.model.roberta.embeddings)
    model._reinit_layers(1)
    assert not _unchanged(layers[-1], before[-1])
    assert all(_unchanged(layer, snap) for layer, snap in zip(layers[:-1], before[:-1]))
    assert _unchanged(model.model.roberta.embeddings, embeddings)


def test_reinit_of_zero_layers_changes_nothing(model: Model) -> None:
    """n=0 is a no-op, not a reset of the whole encoder."""
    before = _snapshot(model.model)
    model._reinit_layers(0)
    assert _unchanged(model.model, before)


def test_scheduler_steps_once_per_batch(model: Model) -> None:
    """One epoch advances the scheduler by the number of training batches."""
    model._train_epoch(0)
    assert model.scheduler.last_epoch == len(model.train_loader)


def test_lr_warms_up_from_zero_and_decays_to_zero(model: Model) -> None:
    """LR starts at zero, peaks early, and falls to zero by the last epoch."""
    lrs = [model.scheduler.get_last_lr()[0]]
    for epoch in range(model.num_epochs):
        model._train_epoch(epoch)
        lrs.append(model.scheduler.get_last_lr()[0])
    assert lrs[0] == 0.0
    assert lrs[1] == max(lrs)
    assert (np.diff(lrs[1:]) < 0).all()
    assert lrs[-1] == pytest.approx(0.0)


def test_train_epoch_updates_weights_in_train_mode(model: Model) -> None:
    """A training epoch moves the weights and leaves dropout switched on."""
    before = _snapshot(model.model)
    loss = model._train_epoch(0)
    assert np.isfinite(loss)
    assert model.model.training
    assert not _unchanged(model.model, before)


def test_scoring_a_loader_leaves_the_weights_alone(model: Model) -> None:
    """
    Scoring runs in eval mode, leaves the weights alone, and returns one
    prediction per row, aligned with the split's labels, with loss as their MSE.
    """
    model.model.train()
    before = _snapshot(model.model)
    loss, preds, labels = model._score(model.valid_loader)
    assert not model.model.training
    assert _unchanged(model.model, before)
    assert len(preds) == len(model.valid_df)
    assert np.allclose(labels, model.valid_df["labels"])
    assert loss == pytest.approx(np.mean((preds - labels) ** 2), rel=1e-5)


def test_fit_stops_early_and_checkpoints_the_best_epoch(
    monkeypatch: pytest.MonkeyPatch, model: Model
) -> None:
    """
    With a scripted test loss, fit() stops patience epochs after the best one,
    writes one history row and one checkpoint per epoch run, and keeps the best
    epoch's weights.
    """
    monkeypatch.setattr(model, "num_epochs", 10)
    monkeypatch.setattr(model, "patience", 3)
    # Early stopping and best.pt key off the test loader, so that is the loss
    # worth scripting; valid's is held flat to prove it does not drive either.
    test_losses = iter([3.0, 2.0, 2.5, 2.6, 2.7, 1.0])
    labels = model.valid_df["labels"].to_numpy()
    monkeypatch.setattr(model, "_train_epoch", lambda epoch: 0.0)
    monkeypatch.setattr(
        model,
        "_score",
        lambda loader: (
            next(test_losses) if loader is model.test_loader else 9.0, labels, labels
        ),
    )

    history = model.fit()
    assert history["epoch"].tolist() == [0, 1, 2, 3, 4]
    assert len(pd.read_csv(model.model_dir / bm.HISTORY)) == 5
    assert sorted(p.name for p in model.checkpoint_dir.glob("epoch*.pt")) == [
        f"epoch{epoch:03d}.pt" for epoch in range(5)
    ]
    checkpoint = torch.load(model.checkpoint_dir / bm.CHECKPOINT)
    assert checkpoint["epoch"] == 1
    assert checkpoint["loss"]["test"] == 2.0


def test_fit_smoke(monkeypatch: pytest.MonkeyPatch, model: Model) -> None:
    """Two real epochs run end to end and leave a history and a checkpoint."""
    monkeypatch.setattr(model, "num_epochs", 2)
    history = model.fit()
    assert len(history) == 2
    metrics = ["train_loss", "test_loss", "valid_loss",
               "test_rmse", "test_r2", "valid_rmse", "valid_r2"]
    assert np.isfinite(history[metrics]).all(axis=None)
    assert (model.checkpoint_dir / bm.CHECKPOINT).exists()


def test_checkpoint_round_trip_restores_weights(model: Model) -> None:
    """Loading a checkpoint undoes later training and returns its epoch."""
    before = _snapshot(model.model)
    model._save_checkpoint(bm.CHECKPOINT, 7, {"train": 0.5, "test": 0.5, "valid": 0.5})
    model._train_epoch(0)
    assert model.load_checkpoint(model.checkpoint_dir / bm.CHECKPOINT) == 7
    assert _unchanged(model.model, before)


def test_evaluate_scores_the_checkpoint_not_the_last_weights(model: Model) -> None:
    """
    evaluate() predicts with the checkpointed weights even after training has
    moved on, and writes those predictions alongside the run. It reads the valid
    split, which is the one held out of selection.
    """
    model._save_checkpoint(bm.CHECKPOINT, 0, {"train": 1.0, "test": 1.0, "valid": 1.0})
    _, expected, _ = model._score(model.valid_loader)
    model._train_epoch(0)
    metrics = model.evaluate()
    assert set(metrics) == {"rmse", "r2"}
    written = pd.read_csv(model.model_dir / bm.PREDICTIONS)
    assert np.allclose(written["preds"], expected, atol=1e-6)
    assert np.allclose(written["labels"], model.valid_df["labels"])
