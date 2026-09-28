import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from bertsheep.baseline import fit_baseline
from bertsheep.model import Model
from bertsheep.splitters import Splitters

TUNING_DIR = Path("out/tuning")
STUDY_DB = "studies.db"
BEST_PARAMS = "{}.json"

# Trials each arm's study runs to. The transformer arms search four dimensions,
# where 50 gives TPE enough to model the space after its random start; six
# would want nearer 75. XGBoost searches six and a trial
# is seconds rather than GPU minutes, so it can afford more.
N_TRIALS = {"baseline": 100, "pretrained": 50, "finetuned": 50}
TUNER_SEED = 0

# Trials TPE samples at random before it has enough observations to model the
# space, and which the pruner needs before it has a median to compare against.
STARTUP_TRIALS = 10
# Epochs a trial runs before it can be pruned. The LR is still warming up over
# the first few, so a trial stopped there would be judged on the schedule it
# shares with every other trial rather than on its own hyperparameters.
WARMUP_EPOCHS = 10

# Search spaces, as {name: (low, high, log)}. Integer bounds draw integers.
# Anything not listed runs at Model's default, or for the baseline at
# XGB_PARAMS and then XGBoost's own.
#
# Fine-tuning. Plain AdamW over the whole network: llrd_decay and reinit_n are
# fine-tuning *methods*, which Question 2.1 compares, so they stay at Model's
# defaults (1.0 and 0, i.e. neither) rather than letting the arm that answers
# Question 1 win on them. `batch_size` is fixed rather than searched: it sets
# the number of optimiser steps in an epoch, so varying it would make the
# per-epoch losses the pruner compares mean different amounts of training
# between trials, and it is the parameter the 4060 timings that sized N_TRIALS
# were measured at. `num_epochs` is fixed for a sharper reason -- it sets
# total_steps for the LR schedule, so searching it would reshape the decay
# curve rather than just lengthen the budget. LR is drawn on a log scale
# because it matters by order of magnitude rather than by increment. Dropout
# is searched rather than inherited: the checkpoint's 0.144 was tuned for
# multi-task pretraining over millions of molecules, and weight decay is a
# poor substitute for it -- dropout does most of the regularising in a
# transformer.
FINETUNED_SPACE = {
    "lr": (1e-5, 1e-3, True),
    "weight_decay": (0.0, 0.3, False),
    "warmup_ratio": (0.0, 0.2, False),
    "dropout": (0.05, 0.30, False),
}
# Frozen encoder, trained head. The same four settings are the only ones that
# reach a head-only fit -- dropout through the head alone, since Model keeps a
# frozen encoder in eval mode -- but the LR range moves up a decade: there are
# no pretrained weights for a large step to wreck, and a randomly initialised
# head on fixed features wants the larger steps a linear probe takes.
PRETRAINED_SPACE = FINETUNED_SPACE | {"lr": (1e-4, 1e-2, True)}
# XGBoost on 1024 sparse fingerprint bits. Column subsampling reaches low
# because most bits are uninformative for any one split; min_child_weight and
# reg_lambda are the regularisers that matter on ~8k training rows, and both
# act by order of magnitude. The number of trees is left to early stopping.
BASELINE_SPACE = {
    "max_depth": (3, 10, False),
    "learning_rate": (0.01, 0.3, True),
    "subsample": (0.5, 1.0, False),
    "colsample_bytree": (0.2, 1.0, False),
    "min_child_weight": (1.0, 20.0, True),
    "reg_lambda": (1e-3, 10.0, True),
}
SEARCH_SPACES = {
    "baseline": BASELINE_SPACE,
    "pretrained": PRETRAINED_SPACE,
    "finetuned": FINETUNED_SPACE,
}


def study_name(target: str, mutation: str | None, arm: str, method: str,
               distribution: str) -> str:
    """
    Name a study, and the best-parameters file it writes, for everything its
    parameters are specific to: the frame they were tuned on, the arm they
    configure and the split they were tuned against, since one variant's or
    one split's winners are not another's. The variant has to be in the name
    because studies resume by name: without it, a mutant's run would load the
    wild-type study, find its trials already run, and fit on its winners.

    Parameters
    ----------
    target : str
        Short target name.
    mutation : str | None
        Variant the frame was selected for, see Data; None for the pooled
        frame, named as the results file and distance cache name it.
    arm : str
        Key of SEARCH_SPACES.
    method : str
        Split method, see Splitters.
    distribution : str
        'in' or 'out'.

    Returns
    -------
    str
        Study name, e.g. 'EGFR-wildtype-finetuned-butina-out'.
    """
    return f"{target}-{mutation or 'pooled'}-{arm}-{method}-{distribution}"


def best_params_path(study: str) -> Path:
    """
    Where a study's winning parameters are written by Tuner and read back
    from by the experiment runner, so the two cannot disagree on it.

    Parameters
    ----------
    study : str
        Study name, see study_name.

    Returns
    -------
    Path
        JSON file under TUNING_DIR.
    """
    return TUNING_DIR / BEST_PARAMS.format(study)


class Tuner:
    """
    Search one arm's hyperparameters for one target and one split with
    Optuna's TPE sampler, scoring trials on valid and never reading test.

    The study's product is a set of *parameters*, not a trained model: the
    experiment matrix refits with them across its replicate seeds, and those
    fits are what report on test. Keeping the search out of that path means
    re-running the matrix does not re-run the search, and it keeps the held-out
    split genuinely held out -- valid is already doing double duty here, picking
    the epoch (or boosting round) within a trial and the trial within the
    study. Each arm gets its own study, so each is compared at its own best
    rather than at settings tuned for another.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns, whose
        `attrs["mutation"]` names the study, so a study cannot be named for a
        variant other than the one its trials were fitted on.
    target : str
        Short target name, used in the study name and passed through to Model.
    splitter : Splitters
        Splitter every trial takes its split from. One instance, built once by
        the caller: holding the split fixed is what makes the trials'
        scores comparable, and it saves rebuilding the fingerprints and
        distance matrix per trial.
    arm : str
        Which model to tune, a key of SEARCH_SPACES: 'baseline' (XGBoost),
        'pretrained' (frozen encoder) or 'finetuned'.
    fingerprints : np.ndarray | None
        (n, bits) fingerprints over `df`, in frame order. They are the
        baseline's features, so only 'baseline' reads them.
    n_trials : int | None
        Total trials the study should reach, counting any it has already run.
        None takes the arm's N_TRIALS.
    seed : int
        Seed for the sampler and for every trial's model, so a repeated search
        over the same frame proposes the same configurations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        splitter: Splitters,
        arm: str,
        fingerprints: np.ndarray | None = None,
        n_trials: int | None = None,
        seed: int = TUNER_SEED,
    ) -> None:
        self.df = df
        self.target = target
        self.splitter = splitter
        self.arm = arm
        self.fingerprints = fingerprints
        self.n_trials = N_TRIALS[arm] if n_trials is None else n_trials
        self.seed = seed
        self.study_name = study_name(target, df.attrs["mutation"], arm,
                                     splitter.method, splitter.distribution)
        TUNING_DIR.mkdir(parents=True, exist_ok=True)

    def _search_space(self, trial: optuna.Trial) -> dict[str, float | int]:
        """
        Draw one trial's hyperparameters from the arm's space in
        SEARCH_SPACES, as integers where both bounds are integers.

        Parameters
        ----------
        trial : optuna.Trial
            Trial to draw from.

        Returns
        -------
        dict[str, float | int]
            Keyword arguments for Model, or XGBoost parameters for the baseline.
        """
        return {
            name: (trial.suggest_int if isinstance(low, int) else trial.suggest_float)(
                name, low, high, log=log
            )
            for name, (low, high, log) in SEARCH_SPACES[self.arm].items()
        }

    def _baseline_loss(self, params: dict[str, float | int]) -> float:
        """
        Fit XGBoost with one hyperparameter set and score it on valid. XGBoost
        reports no per-epoch losses to Optuna, so these trials are never
        pruned; at seconds apiece there is little to save.

        Parameters
        ----------
        params : dict[str, float | int]
            XGBoost parameters drawn for the trial.

        Returns
        -------
        float
            Valid MSE at the boosting round early stopping selected, in the
            same units as the transformer arms' valid loss.
        """
        train, valid, _ = self.splitter.split()
        labels = self.df["labels"].to_numpy()
        model = fit_baseline(self.fingerprints, labels, train, valid, self.seed, params)
        return mean_squared_error(labels[valid], model.predict(self.fingerprints[valid]))

    def _transformer_loss(self, trial: optuna.Trial,
                          params: dict[str, float | int]) -> float:
        """
        Fit ChemBERTa, frozen or fine-tuned as the arm says, with one
        hyperparameter set and score it on valid, stopping part way if the
        trial trails the ones before it.

        Checkpointing is off: a search of this size would write hundreds of
        gigabytes of weights it would then discard, since only the parameters
        are kept. The run directory, config.json and history.csv are still
        written, so a finished study leaves a readable curve per trial.

        Parameters
        ----------
        trial : optuna.Trial
            Trial the epoch losses are reported to.
        params : dict[str, float | int]
            Model arguments drawn for the trial.

        Returns
        -------
        float
            Lowest valid loss the fit reached, which is the epoch that fit()
            would have selected.
        """
        def report(epoch: int, valid_loss: float) -> None:
            """
            Pass the epoch's valid loss to Optuna and stop the fit if the trial
            is already trailing the ones before it.

            Parameters
            ----------
            epoch : int
                Epoch just finished.
            valid_loss : float
                That epoch's valid loss.
            """
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

        model = Model(
            self.df, self.target, self.splitter, seed=self.seed,
            checkpoint=False, freeze=self.arm == "pretrained", **params,
        )
        return model.fit(callback=report)["valid_loss"].min()

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Score one trial's hyperparameters with the arm's model.

        Parameters
        ----------
        trial : optuna.Trial
            Trial supplying the hyperparameters.

        Returns
        -------
        float
            Valid MSE of the fit.
        """
        params = self._search_space(trial)
        if self.arm == "baseline":
            return self._baseline_loss(params)
        return self._transformer_loss(trial, params)

    def _save(self, study: optuna.Study) -> None:
        """
        Write the winning parameters where the experiment runner can read them,
        keyed by the arm and split they were tuned for.

        Parameters
        ----------
        study : optuna.Study
            Finished study.
        """
        record = {
            "study": self.study_name,
            "trials": len(study.trials),
            "best_value": study.best_value,
            "params": study.best_params,
        }
        path = best_params_path(self.study_name)
        path.write_text(json.dumps(record, indent=2))
        print(f"-- Best valid loss {study.best_value:.4f} -- written to {path}")

    def optimise(self) -> dict[str, float | int]:
        """
        Run the study to `n_trials` and record its winner.

        Trials are held in SQLite rather than in memory so a search that dies
        part way -- and this is hours of GPU time -- resumes where it stopped
        rather than starting over; `n_trials` is therefore the total to reach,
        not a number to add on each call.

        Returns
        -------
        dict[str, float | int]
            The best trial's parameters, also written to out/tuning/.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{TUNING_DIR / STUDY_DB}",
            direction="minimize",
            sampler=TPESampler(seed=self.seed, n_startup_trials=STARTUP_TRIALS),
            pruner=MedianPruner(
                n_startup_trials=STARTUP_TRIALS, n_warmup_steps=WARMUP_EPOCHS
            ),
            load_if_exists=True,
        )
        remaining = max(0, self.n_trials - len(study.trials))
        print(f"-- Tuning {self.study_name} -- {remaining} of {self.n_trials} "
              f"trials to run")
        study.optimize(self._objective, n_trials=remaining)
        self._save(study)
        return study.best_params
