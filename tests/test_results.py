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
    A results file shaped like Experiment.grid()'s: a baseline row with no
    run_dir and a pretrained and a fine-tuned run per distribution, each with a
    history.csv that starts at epoch -1 with no train loss, as Model.fit()'s does.
    """
    rng = np.random.default_rng(0)
    rows = []
    for distribution in DISTRIBUTIONS:
        rows.append({"arm": "baseline", "distribution": distribution, "seed": 0,
                     "valid_rmse": 1.5, "run_dir": None})
        for arm in ("pretrained", "finetuned"):
            run_dir = tmp_path / f"{arm}-{distribution}"
            run_dir.mkdir()
            epochs = np.arange(-1, EPOCHS)
            pd.DataFrame({
                "epoch": epochs,
                "train_loss": np.r_[np.nan, rng.uniform(1, 5, EPOCHS)],
                "test_loss": rng.uniform(1, 5, EPOCHS + 1),
            }).to_csv(run_dir / "history.csv", index=False)
            rows.append({"arm": arm, "distribution": distribution, "seed": 0,
                         "valid_rmse": 1.0, "run_dir": str(run_dir)})
    path = tmp_path / "EGFR-wildtype.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_histories_are_long_and_complete(results_path: Path) -> None:
    histories = Results(results_path)._histories()
    assert list(histories.columns) == ["arm", "distribution", "seed", "epoch",
                                       "split", "loss"]
    assert not histories["loss"].isna().any()
    assert set(histories["split"]) == {"train", "test"}
    assert "baseline" not in set(histories["arm"])
    # 4 runs x (EPOCHS + 1 test losses + EPOCHS train losses)
    assert len(histories) == 4 * (2 * EPOCHS + 1)


def test_loss_curves_writes_figure(results_path: Path, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(br, "FIGURE_DIR", tmp_path / "figures")
    path = Results(results_path).loss_curves()
    assert path == tmp_path / "figures" / "EGFR-wildtype_loss_curves.png"
    assert path.stat().st_size > 0
