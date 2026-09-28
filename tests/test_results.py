from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bertsheep.results as br
from bertsheep.results import Results
from bertsheep.splitters import DISTRIBUTIONS

EPOCHS = 10
SEEDS = 5  # enough for the mixed model behind q1()'s brackets to fit


@pytest.fixture
def results_path(tmp_path: Path) -> Path:
    """
    A results file shaped like Experiment.grid()'s, over SEEDS seeds: a
    cluster-mean and a baseline row with no run_dir, and a pretrained and a
    fine-tuned run per distribution, each with a history.csv that starts at
    epoch -1 with no train loss, as Model.fit()'s does.
    """
    rng = np.random.default_rng(0)
    rows = []
    for distribution in DISTRIBUTIONS:
        for seed in range(SEEDS):
            for arm, rmse in (("cluster_mean", 2.0), ("baseline", 1.5)):
                rows.append({"arm": arm, "distribution": distribution, "seed": seed,
                             "test_rmse": rmse + rng.normal(0, 0.1),
                             "test_r2": 0.5, "run_dir": None})
            for arm in ("pretrained", "finetuned"):
                run_dir = tmp_path / f"{arm}-{distribution}-{seed}"
                run_dir.mkdir()
                pd.DataFrame({
                    "epoch": np.arange(-1, EPOCHS),
                    "train_loss": np.r_[np.nan, rng.uniform(1, 5, EPOCHS)],
                    "test_loss": rng.uniform(1, 5, EPOCHS + 1),
                    "valid_loss": rng.uniform(1, 5, EPOCHS + 1),
                }).to_csv(run_dir / "history.csv", index=False)
                rows.append({"arm": arm, "distribution": distribution, "seed": seed,
                             "test_rmse": 1.0 + rng.normal(0, 0.1), "test_r2": 0.7,
                             "run_dir": str(run_dir)})
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
    # 4 runs a seed x (EPOCHS + 1 valid and test losses each + EPOCHS train losses)
    assert len(histories) == 4 * SEEDS * (3 * EPOCHS + 2)


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
    path = Results(results_path).q1()
    assert path == tmp_path / "figures" / "EGFR-wildtype_q1.png"
    assert path.stat().st_size > 0


def test_comparisons_find_a_real_gap_and_not_a_null_one(tmp_path: Path) -> None:
    # Per-seed difficulty shared by every arm, as the grid's splits give; the
    # baseline 0.2 better than pretrained, and fine-tuned a copy of pretrained
    # so the null pair is null in the sample too, not just in expectation.
    rng = np.random.default_rng(0)
    offsets = {"cluster_mean": 1.0, "baseline": 0.0, "pretrained": 0.2, "finetuned": 0.2}
    rows = [
        {"arm": arm, "distribution": distribution, "seed": seed,
         "test_rmse": 1.5 + offset + difficulty + rng.normal(0, 0.05)}
        for distribution in DISTRIBUTIONS
        for seed, difficulty in enumerate(rng.normal(0, 0.3, 30))
        for arm, offset in offsets.items()
    ]
    df = pd.DataFrame(rows)
    df.loc[df["arm"] == "finetuned", "test_rmse"] = df.loc[
        df["arm"] == "pretrained", "test_rmse"].to_numpy()
    path = tmp_path / "EGFR-wildtype.csv"
    df.to_csv(path, index=False)
    comparisons = Results(path).comparisons()
    assert len(comparisons) == len(DISTRIBUTIONS) * 6
    assert (comparisons["p_adjusted"] >= comparisons["p"]).all()
    for distribution in DISTRIBUTIONS:
        pairs = comparisons.loc[distribution]
        assert pairs.loc[("baseline", "pretrained"), "difference"] == pytest.approx(-0.2, abs=0.05)
        assert pairs.loc[("baseline", "pretrained"), "p_adjusted"] < 0.05
        assert pairs.loc[("pretrained", "finetuned"), "p_adjusted"] > 0.05


@pytest.fixture
def trajectories(results_path: Path) -> list[br.Trajectory]:
    """
    One fake trajectory per distribution over the fixture's fine-tuned
    histories: random coordinates for every GIF layer and epoch, so the Q2
    figures can be drawn without ChemBERTa or aligned UMAP.
    """
    rng = np.random.default_rng(0)
    df = pd.read_csv(results_path)
    runs = df[(df["arm"] == "finetuned") & (df["seed"] == 0)].set_index("distribution")
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
