import sys
from pathlib import Path

import pytest

import bertsheep.cli as bc
from bertsheep.experiment import results_path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Stand-ins for Data, Experiment and Results in the CLI that record what was
    built and called, in order, so the dispatch is tested without training or
    drawing anything.
    """
    calls = []

    class FakeData:
        def __init__(self, data_path: Path, target: str, mutation: str) -> None:
            calls.append("data")

        def _preprocess(self) -> None:
            return None

    class FakeExperiment:
        results_path = Path("results.csv")

        def __init__(self, df: None, target: str, mutation: str) -> None:
            calls.append("experiment")

        def tune(self, arms: list[str], distributions: list[str]) -> None:
            calls.append("tune")

        def grid(self, arms: list[str], distributions: list[str],
                 seeds: range) -> None:
            calls.append("grid")

    class FakeResults:
        def __init__(self, path: Path) -> None:
            calls.append(f"results {path}")

        def comparisons(self) -> None:
            calls.append("comparisons")

        def q1(self) -> None:
            calls.append("q1")

        def q2(self) -> list[Path]:
            calls.append("q2")
            return []

    monkeypatch.setattr(bc, "Data", FakeData)
    monkeypatch.setattr(bc, "Experiment", FakeExperiment)
    monkeypatch.setattr(bc, "Results", FakeResults)
    return calls


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    """
    Call the CLI's main as `bertsheep EGFR <args>`.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set sys.argv.
    *args : str
        Flags after the target.
    """
    monkeypatch.setattr(sys, "argv", ["bertsheep", "EGFR", *args])
    bc.main()


def test_neither_flag_is_an_error(monkeypatch: pytest.MonkeyPatch,
                                  calls: list[str]) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch)
    assert calls == []


def test_results_alone_skips_training(monkeypatch: pytest.MonkeyPatch,
                                      calls: list[str]) -> None:
    _run(monkeypatch, "--results", "--mutation", "L858R")
    assert calls == [f"results {results_path('EGFR', 'L858R')}",
                     "comparisons", "q1", "q2"]


def test_train_runs_before_results(monkeypatch: pytest.MonkeyPatch,
                                   calls: list[str]) -> None:
    _run(monkeypatch, "--results", "--train")
    assert calls == ["data", "experiment", "tune", "grid",
                     f"results {results_path('EGFR', 'wildtype')}",
                     "comparisons", "q1", "q2"]


def test_results_path_names_pooled_frame() -> None:
    assert results_path("TEST", None).name == "TEST-pooled.csv"
