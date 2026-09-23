import json
from pathlib import Path

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from bertsheep.model import Model
from bertsheep.splitters import Splitters

TUNING_DIR = Path("out/tuning")
STUDY_DB = "studies.db"
BEST_PARAMS = "{}.json"

N_TRIALS = 50
TUNER_SEED = 0

# Trials TPE samples at random before it has enough observations to model the
# space, and which the pruner needs before it has a median to compare against.
STARTUP_TRIALS = 10
# Epochs a trial runs before it can be pruned. The LR is still warming up over
# the first few, so a trial stopped there would be judged on the schedule it
# shares with every other trial rather than on its own hyperparameters.
WARMUP_EPOCHS = 10

# Search space. `batch_size` is fixed rather than searched: it sets the number
# of optimiser steps in an epoch, so varying it would make the per-epoch losses
# the pruner compares mean different amounts of training between trials, and it
# is the parameter the 4060 timings that sized N_TRIALS were measured at.
# `num_epochs` is fixed for a sharper reason -- it sets total_steps for the LR
# schedule, so searching it would reshape the decay curve rather than just
# lengthen the budget.
LR_RANGE = (1e-5, 1e-3)
LLRD_DECAY_RANGE = (0.7, 1.0)
WEIGHT_DECAY_RANGE = (0.0, 0.3)
WARMUP_RATIO_RANGE = (0.0, 0.2)
# ChemBERTa-10M-MTR has three encoder layers, so this is nearly binary: the
# few-sample reinit trick discards the quarter of the encoder nearest the head
# (Zhang et al. 2021, 6 layers of BERT-large's 24), and a third is already
# past that. Reinitialising all three would be pretraining thrown away rather
# than a fine-tuning setting, and a "fine-tuned" arm that won that way would
# not answer the question the arm exists to answer.
REINIT_N_RANGE = (0, 1)


class Tuner:
    """
    Search fine-tuning hyperparameters for one target and one split with
    Optuna's TPE sampler, scoring trials on test and never reading valid.

    The study's product is a set of *parameters*, not a trained model: the
    experiment matrix refits with them across its replicate seeds, and those
    fits are what report on valid. Keeping the search out of that path means
    re-running the matrix does not re-run the search, and it keeps the held-out
    split genuinely held out -- test is already doing double duty here, picking
    the epoch within a trial and the trial within the study.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns.
    target : str
        Short target name, used in the study name and passed through to Model.
    splitter : Splitters
        Splitter every trial takes its split from. One instance, built once by
        the caller: holding the split fixed is what makes the trials'
        scores comparable, and it saves rebuilding the fingerprints and
        distance matrix per trial.
    n_trials : int
        Total trials the study should reach, counting any it has already run.
    seed : int
        Seed for the sampler and for every trial's Model, so a repeated search
        over the same frame proposes the same configurations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        splitter: Splitters,
        n_trials: int = N_TRIALS,
        seed: int = TUNER_SEED,
    ) -> None:
        self.df = df
        self.target = target
        self.splitter = splitter
        self.n_trials = n_trials
        self.seed = seed
        # Named for what the parameters will be reused on: hyperparameters
        # tuned against one split method are not the ones the other wants.
        self.study_name = f"{target}-{splitter.method}-{splitter.distribution}"
        TUNING_DIR.mkdir(parents=True, exist_ok=True)

    def _search_space(self, trial: optuna.Trial) -> dict[str, float | int]:
        """
        Draw one trial's hyperparameters, as keyword arguments for Model.

        Parameters
        ----------
        trial : optuna.Trial
            Trial to draw from.

        Returns
        -------
        dict[str, float | int]
            Model constructor arguments. LR is drawn on a log scale because it
            matters by order of magnitude rather than by increment.
        """
        return {
            "lr": trial.suggest_float("lr", *LR_RANGE, log=True),
            "llrd_decay": trial.suggest_float("llrd_decay", *LLRD_DECAY_RANGE),
            "weight_decay": trial.suggest_float(
                "weight_decay", *WEIGHT_DECAY_RANGE
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *WARMUP_RATIO_RANGE
            ),
            "reinit_n": trial.suggest_int("reinit_n", *REINIT_N_RANGE),
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Fit one hyperparameter set and score it on test.

        Checkpointing is off: a search of this size would write hundreds of
        gigabytes of weights it would then discard, since only the parameters
        are kept. The run directory, config.json and history.csv are still
        written, so a finished study leaves a readable curve per trial.

        Parameters
        ----------
        trial : optuna.Trial
            Trial supplying the hyperparameters.

        Returns
        -------
        float
            Lowest test loss the fit reached, which is the epoch that fit()
            would have selected.
        """
        def report(epoch: int, test_loss: float) -> None:
            """
            Pass the epoch's test loss to Optuna and stop the fit if the trial
            is already trailing the ones before it.

            Parameters
            ----------
            epoch : int
                Epoch just finished.
            test_loss : float
                That epoch's test loss.
            """
            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

        model = Model(
            self.df, self.target, self.splitter, seed=self.seed,
            checkpoint=False, **self._search_space(trial),
        )
        return model.fit(callback=report)["test_loss"].min()

    def _save(self, study: optuna.Study) -> None:
        """
        Write the winning parameters where the experiment runner can read them,
        keyed by the configuration they were tuned for.

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
        path = TUNING_DIR / BEST_PARAMS.format(self.study_name)
        path.write_text(json.dumps(record, indent=2))
        print(f"-- Best test loss {study.best_value:.4f} -- written to {path}")

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
            The best trial's Model arguments, also written to out/tuning/.
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
