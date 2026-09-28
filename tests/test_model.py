import json
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
    frame = pd.DataFrame({"smiles": SMILES, "mutations": "", "labels": labels})
    # Data._preprocess records its source here, and Model writes it to config.json
    frame.attrs = {"data_path": "test.tsv", "mutation": None}
    return frame


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


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: pd.DataFrame) -> Model:
    """
    The same small CPU Model with its encoder frozen and checkpointing off, as
    the pre-trained arm runs it.
    """
    monkeypatch.setattr(bm, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(bm, "DEVICE", torch.device("cpu"))
    return Model(
        frame,
        "TEST",
        Splitters(frame["smiles"], "scaffold", "in"),
        model_link=MODEL_LINK,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        checkpoint=False,
        freeze=True,
    )


def test_parameter_groups_cover_each_parameter_once(model: Model) -> None:
    """
    Every parameter is in exactly one group, only biases and LayerNorm weights
    skip weight decay, and each depth trains at llrd_decay x the one above.
    The default of 1.0 would pass the depth check whatever the groups' order,
    so a real decay is set first; the groups read it when they are built.
    """
    model.llrd_decay = 0.9
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


def test_fit_stops_early_and_keeps_the_best_epoch(
    monkeypatch: pytest.MonkeyPatch, model: Model
) -> None:
    """
    With a scripted valid loss, fit() stops patience epochs after the best one,
    writes one history row per epoch run, keeps the best epoch's weights in
    memory, and writes only the starting and best weights to disk.
    """
    monkeypatch.setattr(model, "num_epochs", 10)
    monkeypatch.setattr(model, "patience", 3)
    # Early stopping and selection key off the valid loader, so that is the loss
    # worth scripting; test's is held flat to prove it does not drive either.
    # The first loss is epoch -1's, which is scored but never selected.
    valid_losses = iter([0.5, 3.0, 2.0, 2.5, 2.6, 2.7, 1.0])
    labels = model.test_df["labels"].to_numpy()
    monkeypatch.setattr(model, "_train_epoch", lambda epoch: 0.0)
    monkeypatch.setattr(
        model,
        "_score",
        lambda loader: (
            next(valid_losses) if loader is model.valid_loader else 9.0, labels, labels
        ),
    )

    history = model.fit()
    assert history["epoch"].tolist() == [-1, 0, 1, 2, 3, 4]
    assert len(pd.read_csv(model.model_dir / bm.HISTORY)) == 6
    assert sorted(p.name for p in model.checkpoint_dir.glob("*.pt")) == [
        bm.BEST_CHECKPOINT, bm.INIT_CHECKPOINT
    ]
    assert model.best_epoch == 1
    assert model.best_state is not None
    best = torch.load(model.checkpoint_dir / bm.BEST_CHECKPOINT)
    assert best["epoch"] == 1
    assert best["loss"]["valid"] == 2.0


def test_trajectory_run_checkpoints_every_epoch(
    monkeypatch: pytest.MonkeyPatch, model: Model
) -> None:
    """A trajectory run writes every epoch's weights as well as the two ends."""
    monkeypatch.setattr(model, "num_epochs", 3)
    monkeypatch.setattr(model, "trajectory", True)
    monkeypatch.setattr(model, "_train_epoch", lambda epoch: 0.0)
    model.fit()
    assert sorted(p.name for p in model.checkpoint_dir.glob("*.pt")) == [
        bm.BEST_CHECKPOINT, "epoch000.pt", "epoch001.pt", "epoch002.pt",
        bm.INIT_CHECKPOINT,
    ]


def test_fit_smoke(monkeypatch: pytest.MonkeyPatch, model: Model) -> None:
    """Two real epochs run end to end and leave a history and checkpoints."""
    monkeypatch.setattr(model, "num_epochs", 2)
    history = model.fit()
    assert len(history) == 3  # epoch -1 and two trained epochs
    metrics = ["train_loss", "valid_loss", "test_loss",
               "valid_rmse", "valid_r2", "test_rmse", "test_r2"]
    assert np.isfinite(history.loc[history["epoch"] >= 0, metrics]).all(axis=None)
    assert model._checkpoint_path(-1).exists()
    assert (model.checkpoint_dir / bm.BEST_CHECKPOINT).exists()


def test_checkpoint_round_trip_restores_weights(model: Model) -> None:
    """Loading a checkpoint undoes later training and returns its epoch."""
    before = _snapshot(model.model)
    # _save_checkpoint takes its losses from the epoch's history row
    model.history.append({"epoch": 7, "train_loss": 0.5, "valid_loss": 0.5,
                          "test_loss": 0.5})
    model._save_checkpoint(7, model.model.state_dict(), model._checkpoint_path(7))
    model._train_epoch(0)
    assert model.load_checkpoint(model._checkpoint_path(7)) == 7
    assert _unchanged(model.model, before)


def test_evaluate_scores_the_best_weights_not_the_last(model: Model) -> None:
    """
    evaluate() scores the best epoch's weights even after training has moved
    on. It reads the test split, which is the one held out of selection.
    """
    model.best_epoch, model.best_state = 0, _snapshot(model.model)
    _, preds, labels = model._score(model.test_loader)
    expected = model._metrics(preds, labels)
    model._train_epoch(0)
    metrics = model.evaluate()
    assert metrics.keys() == expected.keys()
    assert np.allclose(list(metrics.values()), list(expected.values()), atol=1e-6)


def test_split_frames_recovers_the_trained_splits(model: Model, frame: pd.DataFrame) -> None:
    """
    The run directory holds only split positions, and split_frames() turns them
    back into exactly the frames the run trained, selected and scored on.
    """
    for recovered, used in zip(bm.split_frames(frame, model.model_dir),
                               (model.train_df, model.valid_df, model.test_df)):
        pd.testing.assert_frame_equal(recovered, used)


def test_frozen_epoch_trains_the_head_only(frozen: Model) -> None:
    """
    A frozen epoch leaves every encoder weight where it was, moves the head,
    and keeps the encoder's dropout off while the head's is on.
    """
    encoder = _snapshot(frozen.model.roberta)
    head = _snapshot(frozen.model.classifier)
    frozen._train_epoch(0)
    assert _unchanged(frozen.model.roberta, encoder)
    assert not _unchanged(frozen.model.classifier, head)
    assert not frozen.model.roberta.training
    assert frozen.model.classifier.training


def test_fit_without_checkpoints_still_evaluates(
    monkeypatch: pytest.MonkeyPatch, frozen: Model
) -> None:
    """With checkpointing off nothing reaches disk, and evaluate() still runs."""
    monkeypatch.setattr(frozen, "num_epochs", 2)
    frozen.fit()
    assert not list(frozen.checkpoint_dir.glob("*.pt"))
    assert np.isfinite(frozen.evaluate()["rmse"])


def test_config_records_what_rebuilds_the_split(model: Model) -> None:
    """
    config.json carries the source dump, mutation and min_cluster_size, which
    together with splits.parquet are what rebuild the frames a run saw.
    """
    config = json.loads((model.model_dir / bm.CONFIG).read_text())
    assert config["data_path"] == "test.tsv"
    assert config["mutation"] is None
    assert config["min_cluster_size"] == model.splitter.min_cluster_size
