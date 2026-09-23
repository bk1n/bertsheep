from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bertsheep.experiment as be
from bertsheep.experiment import ARMS, Experiment
from bertsheep.splitters import DISTRIBUTIONS, Splitters

# Ten ring systems, each carried by four alkyl homologues, so Butina has
# clusters to group and both distributions have something to hold out.
CORES = ("c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "C1CCNCC1", "C1CCNC1",
         "c1ccco1", "c1ccsc1", "c1cc2ccccc2[nH]1", "c1ccc2ccccc2n1", "C1CCOCC1")
ALKYLS = ("C", "CC", "CCC", "CCCC")
SMILES = [alkyl + core for core in CORES for alkyl in ALKYLS]


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Experiment:
    """
    An Experiment over the homologue series, writing its results into tmp_path.
    min_cluster_size 1 keeps every molecule, since forty is already few.
    """
    monkeypatch.setattr(be, "EXPERIMENT_DIR", tmp_path)
    labels = np.random.default_rng(0).normal(6, 1, len(SMILES))
    frame = pd.DataFrame({"smiles": SMILES, "labels": labels})
    frame.attrs = {"data_path": "test.tsv", "mutation": None}
    return Experiment(frame, "TEST", mutation=None, min_cluster_size=1)


def _fake_transformer(splitter: Splitters, freeze: bool) -> dict:
    """
    Stand-in for Experiment._transformer that returns at once.

    Parameters
    ----------
    splitter : Splitters
        Ignored.
    freeze : bool
        Ignored.

    Returns
    -------
    dict
        A result with the keys the real method returns.
    """
    return {"valid_rmse": 1.0, "valid_r2": 0.0, "best_epoch": 0, "run_dir": None}


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_same_seed_gives_the_same_split(experiment: Experiment,
                                        distribution: str) -> None:
    """A rebuilt splitter with the same seed deals the same rows."""
    first = experiment._splitter(distribution, 3).split()
    second = experiment._splitter(distribution, 3).split()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_arms_are_scored_on_the_same_split(
    monkeypatch: pytest.MonkeyPatch, experiment: Experiment
) -> None:
    """Every arm of a (distribution, seed) sees identical train/test/valid rows."""
    seen = []

    def record(splitter: Splitters, *args, **kwargs) -> dict:
        seen.append(splitter.split())
        return _fake_transformer(splitter, False)

    monkeypatch.setattr(experiment, "_baseline", record)
    monkeypatch.setattr(experiment, "_transformer", record)
    experiment.grid(distributions=["out"], seeds=[0])
    assert len(seen) == len(ARMS)
    for split in seen[1:]:
        assert all(np.array_equal(a, b) for a, b in zip(seen[0], split))


def test_baseline_is_repeatable(experiment: Experiment) -> None:
    """The same split and seed give the same baseline score."""
    splitter = experiment._splitter("in", 0)
    assert experiment._baseline(splitter) == experiment._baseline(splitter)


def test_grid_resumes_rather_than_repeating(
    monkeypatch: pytest.MonkeyPatch, experiment: Experiment
) -> None:
    """
    A second call runs only the combinations the results file does not have,
    so extending the seeds adds rows and repeating a call adds none.
    """
    monkeypatch.setattr(experiment, "_transformer", _fake_transformer)
    per_seed = len(ARMS) * len(DISTRIBUTIONS)
    assert len(experiment.grid(seeds=range(2))) == 2 * per_seed
    assert len(experiment.grid(seeds=range(2))) == 2 * per_seed
    results = experiment.grid(seeds=range(3))
    assert len(results) == 3 * per_seed
    assert not results.duplicated(be.KEY).any()


@pytest.mark.parametrize("seed", [5, be.GIF_SEED])
@pytest.mark.parametrize("arm", ["pretrained", "finetuned"])
def test_transformer_arms_differ_only_in_freezing(
    monkeypatch: pytest.MonkeyPatch, experiment: Experiment, arm: str, seed: int
) -> None:
    """
    Both transformer arms get the tuned parameters and the replicate seed; the
    frozen one also drops reinit_n and skips checkpoints. Only GIF_SEED keeps
    every epoch.
    """
    built = {}

    def fake_model(df: pd.DataFrame, target: str, splitter: Splitters,
                   **kwargs) -> SimpleNamespace:
        built.update(kwargs)
        return SimpleNamespace(fit=lambda: None,
                               evaluate=lambda: {"rmse": 1.0, "r2": 0.0},
                               best_epoch=0, model_dir=Path("run"))

    monkeypatch.setattr(be, "Model", fake_model)
    monkeypatch.setattr(experiment, "_params",
                        lambda distribution: {"lr": 1e-4, "reinit_n": 1})
    experiment.run(arm, "in", seed)

    frozen = arm == "pretrained"
    assert built["seed"] == seed
    assert built["lr"] == 1e-4
    assert built["freeze"] is frozen
    assert built["checkpoint"] is not frozen
    assert built["trajectory"] is (seed == be.GIF_SEED)
    assert ("reinit_n" in built) is not frozen
