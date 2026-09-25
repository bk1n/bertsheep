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
                     "valid_rmse": 2.0, "valid_r2": 0.3, "run_dir": None})
        rows.append({"arm": "baseline", "distribution": distribution, "seed": 0,
                     "valid_rmse": 1.5, "valid_r2": 0.6, "run_dir": None})
        for arm in ("pretrained", "finetuned"):
            run_dir = tmp_path / f"{arm}-{distribution}"
            run_dir.mkdir()
            epochs = np.arange(-1, EPOCHS)
            pd.DataFrame({
                "epoch": epochs,
                "train_loss": np.r_[np.nan, rng.uniform(1, 5, EPOCHS)],
                "test_loss": rng.uniform(1, 5, EPOCHS + 1),
                "valid_loss": rng.uniform(1, 5, EPOCHS + 1),
            }).to_csv(run_dir / "history.csv", index=False)
            rows.append({"arm": arm, "distribution": distribution, "seed": 0,
                         "valid_rmse": 1.0, "valid_r2": 0.7, "run_dir": str(run_dir)})
    path = tmp_path / "EGFR-wildtype.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_histories_are_long_and_complete(results_path: Path) -> None:
    histories = Results(results_path)._histories()
    assert list(histories.columns) == ["arm", "distribution", "seed", "epoch",
                                       "split", "loss"]
    assert not histories["loss"].isna().any()
    assert set(histories["split"]) == {"train", "test", "valid"}
    assert set(histories["arm"]) == {"pretrained", "finetuned"}
    # 4 runs x (EPOCHS + 1 test and valid losses each + EPOCHS train losses)
    assert len(histories) == 4 * (3 * EPOCHS + 2)


def test_surviving_drops_epochs_most_seeds_never_reached(
        results_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ten seeds, seed i stopping after epoch i: epoch e is reached by 10 - e
    # seeds, so only epochs 0 and 1 keep the 90% needed.
    histories = pd.DataFrame([
        {"arm": "pretrained", "distribution": "in", "seed": seed, "epoch": epoch,
         "split": "test", "loss": 1.0}
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
