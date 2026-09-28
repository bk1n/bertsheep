from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bertsheep.results as br
from bertsheep.results import Results
from bertsheep.splitters import DISTRIBUTIONS

EPOCHS = 10


@pytest.fixture
def results_path(tmp_path: Path) -> Path:
    """
    A results file shaped like Experiment.grid()'s: a cluster-mean and a
    baseline row with no run_dir and a pretrained and a fine-tuned run per distribution, each with a
    history.csv that starts at epoch -1 with no train loss, as Model.fit()'s does.
    """
    rng = np.random.default_rng(0)
    rows = []
    for distribution in DISTRIBUTIONS:
        rows.append({"arm": "cluster_mean", "distribution": distribution, "seed": 0,
                     "test_rmse": 2.0, "test_r2": 0.3, "run_dir": None})
        rows.append({"arm": "baseline", "distribution": distribution, "seed": 0,
                     "test_rmse": 1.5, "test_r2": 0.6, "run_dir": None})
        for arm in ("pretrained", "finetuned"):
            run_dir = tmp_path / f"{arm}-{distribution}"
            run_dir.mkdir()
            epochs = np.arange(-1, EPOCHS)
            pd.DataFrame({
                "epoch": epochs,
                "train_loss": np.r_[np.nan, rng.uniform(1, 5, EPOCHS)],
                "valid_loss": rng.uniform(1, 5, EPOCHS + 1),
                "test_loss": rng.uniform(1, 5, EPOCHS + 1),
            }).to_csv(run_dir / "history.csv", index=False)
            rows.append({"arm": arm, "distribution": distribution, "seed": 0,
                         "test_rmse": 1.0, "test_r2": 0.7, "run_dir": str(run_dir)})
    path = tmp_path / "EGFR-wildtype.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_histories_are_long_and_complete(results_path: Path) -> None:
    histories = Results(results_path)._histories()
    assert list(histories.columns) == ["arm", "distribution", "seed", "epoch",
                                       "split", "rmse"]
    assert not histories["rmse"].isna().any()
    assert set(histories["split"]) == {"train", "valid", "test"}
    assert set(histories["arm"]) == {"pretrained", "finetuned"}
    # 4 runs x (EPOCHS + 1 valid and test losses each + EPOCHS train losses)
    assert len(histories) == 4 * (3 * EPOCHS + 2)


def test_surviving_drops_epochs_most_seeds_never_reached(
        results_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ten seeds, seed i stopping after epoch i: epoch e is reached by 10 - e
    # seeds, so only epochs 0 and 1 keep the 90% needed.
    histories = pd.DataFrame([
        {"arm": "pretrained", "distribution": "in", "seed": seed, "epoch": epoch,
         "split": "test", "rmse": 1.0}
        for seed in range(10) for epoch in range(seed + 1)
    ])
    monkeypatch.setattr(Results, "_histories", lambda self: histories)
    monkeypatch.setattr(br, "SURVIVOR_SHARE", 0.9)
    assert set(Results(results_path)._surviving()["epoch"]) == {0, 1}


def test_q1_writes_figure(results_path: Path, tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(br, "FIGURE_DIR", tmp_path / "figures")
    results = Results(results_path)
    # Stands in for the cached property, which needs the dump and real splits.
    results.similarity = pd.Series(
        [0.7, 0.4], index=pd.MultiIndex.from_tuples([("in", 0), ("out", 0)]))
    path = results.q1()
    assert path == tmp_path / "figures" / "EGFR-wildtype_q1.png"
    assert path.stat().st_size > 0


def test_similarity_is_median_nearest_train_neighbour(results_path: Path,
                                                      tmp_path: Path) -> None:
    # Molecules 0-2 are train and 3-4 test. 3's nearest train molecule is 0.2
    # away and 4's is 0.6 away, so the similarities are 0.8 and 0.4, median 0.6.
    distances = np.ones((5, 5))
    distances[3, [0, 1, 2]] = [0.5, 0.2, 0.9]
    distances[4, [0, 1, 2]] = [0.6, 0.7, 0.8]
    for distribution in DISTRIBUTIONS:
        pd.DataFrame({"row": [0, 1, 2, 3, 4],
                      "split": ["train"] * 3 + ["test"] * 2}).to_parquet(
            tmp_path / f"pretrained-{distribution}" / br.SPLITS, index=False)
    results = Results(results_path)
    results.distances = distances  # stands in for the cached matrix
    np.testing.assert_allclose(results.similarity.loc[("in", 0)], 0.6)


@pytest.fixture
def trajectories(results_path: Path) -> list[br.Trajectory]:
    """
    One fake trajectory per distribution over the fixture's fine-tuned
    histories: random coordinates for every GIF layer and epoch, so the Q2
    figures can be drawn without ChemBERTa or aligned UMAP.
    """
    rng = np.random.default_rng(0)
    df = pd.read_csv(results_path)
    runs = df[df["arm"] == "finetuned"].set_index("distribution")
    n = 30
    molecules = pd.DataFrame({
        "split": np.repeat(["train", "valid", "test"], n // 3),
        "labels": rng.normal(size=n),
        "cluster": rng.integers(0, 50, n),
    })
    return [
        br.Trajectory(distribution, molecules,
                      rng.normal(size=(len(br.GIF_LAYERS), EPOCHS + 1, n, 2)),
                      pd.read_csv(Path(runs.loc[distribution, "run_dir"]) / "history.csv"),
                      best_epoch=3)
        for distribution in DISTRIBUTIONS
    ]


def test_at_interpolates_between_epochs_and_holds_after_the_last(
        trajectories: list[br.Trajectory], results_path: Path) -> None:
    results, trajectory = Results(results_path), trajectories[0]
    coords = trajectory.coords[0]
    np.testing.assert_allclose(results._at(trajectory, 0, -1), coords[0])
    np.testing.assert_allclose(results._at(trajectory, 0, 0.5),
                               (coords[1] + coords[2]) / 2)
    np.testing.assert_allclose(results._at(trajectory, 0, EPOCHS + 5), coords[-1])


def test_status_marks_start_selection_and_stop(
        trajectories: list[br.Trajectory], results_path: Path) -> None:
    results, trajectory = Results(results_path), trajectories[0]
    assert "pretrained" in results._status(trajectory, -1)
    assert results._status(trajectory, 3.5).endswith("epoch 3 (selected)")
    assert "stopped after epoch 9" in results._status(trajectory, EPOCHS - 1)


@pytest.mark.parametrize("colour", br.COLOURINGS)
def test_q2_writes_gif_and_filmstrip(
        trajectories: list[br.Trajectory], results_path: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, colour: str) -> None:
    monkeypatch.setattr(br, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(br, "HOLD_FRAMES", 1)
    monkeypatch.setattr(br, "TWEEN_FRAMES", 1)
    monkeypatch.setattr(br, "DPI", 50)
    results = Results(results_path)
    results.trajectories = trajectories  # stands in for the cached property
    for path in (results.q2_gif(colour), results.q2_filmstrip(colour)):
        assert path.parent == tmp_path / "figures"
        assert path.stat().st_size > 0
