import inspect
import json

import optuna
import pytest

import bertsheep.tuning as bt
from bertsheep.experiment import METHOD, TUNED_ARMS, Experiment
from bertsheep.model import Model
from bertsheep.tuning import SEARCH_SPACES, Tuner, best_params_path, study_name


def _record(arm: str, distribution: str = "in") -> dict:
    """
    Read back the best-parameters file a TEST study wrote.

    Parameters
    ----------
    arm : str
        Arm the study tuned.
    distribution : str
        Distribution it was tuned for.

    Returns
    -------
    dict
        The file's contents.
    """
    path = best_params_path(study_name("TEST", arm, METHOD, distribution))
    return json.loads(path.read_text())


def test_every_arm_has_a_space_and_a_budget() -> None:
    """The experiment's arms, the search spaces and the trial budgets agree."""
    assert set(SEARCH_SPACES) == set(TUNED_ARMS) == set(bt.N_TRIALS)


def test_fine_tuning_methods_are_left_at_plain_defaults() -> None:
    """
    LLRD and top-layer reinit are not searched, and Model's defaults for them
    are plain fine-tuning: uniform LR over depth and no layer reset.
    """
    assert not any({"llrd_decay", "reinit_n"} & set(space)
                   for space in SEARCH_SPACES.values())
    defaults = inspect.signature(Model).parameters
    assert defaults["llrd_decay"].default == 1.0
    assert defaults["reinit_n"].default == 0


@pytest.mark.parametrize("arm", TUNED_ARMS)
def test_search_space_draws_within_bounds(experiment: Experiment, arm: str) -> None:
    """
    A trial draws every parameter in its arm's space, inside the bounds, and
    integer-valued where the bounds are integers.
    """
    tuner = Tuner(experiment.df, "TEST", experiment._splitter("in", 0), arm)
    params = tuner._search_space(optuna.create_study().ask())
    assert params.keys() == SEARCH_SPACES[arm].keys()
    for name, (low, high, _) in SEARCH_SPACES[arm].items():
        assert low <= params[name] <= high
        assert isinstance(params[name], int) is isinstance(low, int)


def test_baseline_study_writes_resumes_and_is_read_back(
    experiment: Experiment,
) -> None:
    """
    An XGBoost study writes its winner under its arm's name, a second call to
    the same total runs no more trials, and the matrix fits from what it wrote.
    """
    splitter = experiment._splitter("in", 0)
    for _ in range(2):
        Tuner(experiment.df, "TEST", splitter, "baseline",
              fingerprints=experiment.fingerprints, n_trials=3).optimise()
    record = _record("baseline")
    assert record["trials"] == 3
    assert record["params"].keys() == SEARCH_SPACES["baseline"].keys()
    assert experiment._params("baseline", "in") == record["params"]
    assert experiment._baseline(splitter)["test_rmse"] > 0


def test_tune_keys_each_arm_separately(
    monkeypatch: pytest.MonkeyPatch, experiment: Experiment
) -> None:
    """
    tune() runs one study per arm asked for, each written to its own file, so
    narrowing the arms narrows the tuning and the arms never overwrite each
    other. The transformer fits are stubbed with a score read off the LR.
    """
    monkeypatch.setattr(bt, "N_TRIALS", dict.fromkeys(TUNED_ARMS, 2))
    monkeypatch.setattr(Tuner, "_transformer_loss",
                        lambda self, trial, params: params["lr"])

    experiment.tune(arms=["pretrained"], distributions=["in"])
    assert [p.stem for p in bt.TUNING_DIR.glob("*.json")] == [
        study_name("TEST", "pretrained", METHOD, "in")
    ]

    experiment.tune(distributions=["in"])
    for arm in TUNED_ARMS:
        record = _record(arm)
        assert record["study"] == study_name("TEST", arm, METHOD, "in")
        assert record["params"].keys() == SEARCH_SPACES[arm].keys()
    low, high, _ = SEARCH_SPACES["pretrained"]["lr"]
    assert low <= _record("pretrained")["params"]["lr"] <= high
