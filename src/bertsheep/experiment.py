import argparse
import json
import time
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from bertsheep.baseline import fit_baseline
from bertsheep.chemistry import Chemist
from bertsheep.data import CACHE_DIR, DUMP_PATH, WILD_TYPE, Data
from bertsheep.model import Model
from bertsheep.splitters import DISTRIBUTIONS, Splitters
from bertsheep.tuning import Tuner, best_params_path, study_name

EXPERIMENT_DIR = Path("out/experiments")
RESULTS = "{target}-{mutation}.csv"
KEY = ["arm", "distribution", "seed"]  # one results row per combination

# Butina only; Bemis-Murcko scaffold splits are not part of the matrix.
METHOD = "butina"
# "mean" predicts the train mean: the floor every model has to clear. R2 = 0
# is not that floor, since it scores against the valid set's own mean, which
# no model could know -- least of all out of distribution.
# "cluster_mean" predicts each molecule's Butina cluster's train mean: what
# knowing the series alone is worth. In distribution every valid cluster is in
# train, so it is the floor a model has to clear to have learnt chemistry
# within a series; out of distribution no valid cluster is, so it falls back to
# the train mean everywhere and scores exactly as "mean" does.
ARMS = ("mean", "cluster_mean", "baseline", "pretrained", "finetuned")
UNTUNED = {"mean", "cluster_mean"}  # arms with no hyperparameters to search
TUNED_ARMS = tuple(arm for arm in ARMS if arm not in UNTUNED)
N_REPEATS = 30  # seeds 0..N-1, each a new split and a new model initialisation
# The search runs once, on the first replicate's split; every seed reuses it.
TUNING_SEED = 0
# The replicate whose fine-tuned runs keep every epoch's weights, for the
# latent-space GIF. One run per distribution is all the animation draws on;
# the other seeds keep only the starting and best weights RSA compares.
GIF_SEED = 0
MIN_CLUSTER_SIZE = 10
MUTATION = WILD_TYPE


class Experiment:
    """
    Run the arm x distribution x seed matrix that answers README's Question 1
    for one target frame, and write one results row per run.

    Every arm -- the label-only and cluster-mean references, an XGBoost
    fingerprint baseline, a frozen pretrained encoder with a trained head, and
    a fine-tuned encoder -- is scored on the same split for every
    (distribution, seed), so their differences can be read in pairs, seed by
    seed. The fingerprints and Tanimoto distance matrix are computed once here
    and shared by every splitter, since they depend on the molecules and not on
    the seed. The matrix is also cached under out/.cache/, so a restarted or
    repeated run on the same frame loads it rather than spending a minute
    rebuilding it.

    Rows are appended to out/experiments/<target>-<mutation>.csv as each run
    finishes, and grid() skips combinations already there, so a matrix that
    stops part way resumes rather than restarting.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns, as returned by
        Data._preprocess.
    target : str
        Short target name, used for the tuned parameters and the results file.
    mutation : str | None
        Variant the frame was selected for, recorded on every row.
    min_cluster_size : int
        Butina clusters smaller than this are left out of every split.
    """

    def __init__(self, df: pd.DataFrame, target: str,
                 mutation: str | None = MUTATION,
                 min_cluster_size: int = MIN_CLUSTER_SIZE) -> None:
        self.df = df
        self.target = target
        self.mutation = mutation
        self.min_cluster_size = min_cluster_size
        variant = mutation or "pooled"
        chemist = Chemist()
        self.fingerprints = chemist.fingerprints(df["smiles"])
        self.distances = chemist.cached_tanimoto(
            self.fingerprints, CACHE_DIR, f"{target}-{variant}"
        )
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        self.results_path = EXPERIMENT_DIR / RESULTS.format(
            target=target, mutation=variant
        )

    def _splitter(self, distribution: str, seed: int) -> Splitters:
        """
        Build the split for one replicate from the shared distance matrix, so
        a new seed costs a Butina clustering, not a new matrix.

        Parameters
        ----------
        distribution : str
            'in' or 'out', see Splitters.
        seed : int
            Replicate seed.

        Returns
        -------
        Splitters
            Splitter over this frame's SMILES.
        """
        return Splitters(
            self.df["smiles"], METHOD, distribution, seed=seed,
            min_cluster_size=self.min_cluster_size, distances=self.distances,
        )

    def tune(self, arms: Iterable[str] = ARMS,
             distributions: Iterable[str] = DISTRIBUTIONS) -> None:
        """
        Run each arm's hyperparameter search for each distribution on the
        first replicate's split. Tuning is split-specific because what an
        out-of-distribution fit needs to regularise against differs from an
        in-distribution one, and arm-specific because a frozen encoder, a
        fine-tuned one and a tree ensemble want different settings; tuning
        once rather than per seed is what the 4060's budget allows.

        Parameters
        ----------
        arms : Iterable[str]
            Arms to tune.
        distributions : Iterable[str]
            Distributions to tune for.
        """
        arms = [arm for arm in arms if arm in TUNED_ARMS]
        for distribution in distributions:
            splitter = self._splitter(distribution, TUNING_SEED)
            for arm in arms:
                Tuner(self.df, self.target, splitter, arm,
                      fingerprints=self.fingerprints).optimise()

    def _params(self, arm: str, distribution: str) -> dict[str, float | int]:
        """
        An arm's tuned parameters for a distribution, as written by Tuner.
        Read rather than defaulted: a matrix run on untuned defaults would
        look exactly like a tuned one in the results.

        Parameters
        ----------
        arm : str
            One of ARMS.
        distribution : str
            Distribution the parameters were tuned for.

        Returns
        -------
        dict[str, float | int]
            Model keyword arguments, or XGBoost parameters for the baseline.
        """
        path = best_params_path(study_name(self.target, arm, METHOD, distribution))
        return json.loads(path.read_text())["params"]

    def _mean(self, splitter: Splitters) -> dict[str, float | str | None]:
        """
        Predict the train mean for every valid molecule: what a model scores
        from the label distribution alone, with no chemistry.

        Parameters
        ----------
        splitter : Splitters
            Split to fit and score on.

        Returns
        -------
        dict[str, float | str | None]
            Valid RMSE and R2, and no epoch or run directory.
        """
        train, _, valid = splitter.split()
        X, y = self.fingerprints, self.df["labels"].to_numpy()
        preds = DummyRegressor().fit(X[train], y[train]).predict(X[valid])
        return {
            "valid_rmse": root_mean_squared_error(y[valid], preds),
            "valid_r2": r2_score(y[valid], preds),
            "best_epoch": None,
            "run_dir": None,
        }

    def _cluster_mean(self, splitter: Splitters) -> dict[str, float | str | None]:
        """
        Predict each valid molecule's cluster mean over train, or the train
        mean where its cluster has no train molecules: what a model scores
        from knowing which series a molecule belongs to, with no chemistry
        inside the series.

        Parameters
        ----------
        splitter : Splitters
            Split to fit and score on; its clusters are the series.

        Returns
        -------
        dict[str, float | str | None]
            Valid RMSE and R2, and no epoch or run directory.
        """
        train, _, valid = splitter.split()
        y = self.df["labels"].to_numpy()
        means = pd.Series(y[train]).groupby(splitter.clusters[train]).mean()
        preds = pd.Series(splitter.clusters[valid]).map(means).fillna(y[train].mean())
        return {
            "valid_rmse": root_mean_squared_error(y[valid], preds),
            "valid_r2": r2_score(y[valid], preds),
            "best_epoch": None,
            "run_dir": None,
        }

    def _baseline(self, splitter: Splitters) -> dict[str, float | str | None]:
        """
        Fit XGBoost with its tuned parameters and score it on valid, once, at
        the boosting round early stopping on test selected.

        Parameters
        ----------
        splitter : Splitters
            Split to fit and score on; its seed also seeds the subsampling.

        Returns
        -------
        dict[str, float | str | None]
            Valid RMSE and R2, the best boosting round and no run directory.
        """
        train, test, valid = splitter.split()
        X, y = self.fingerprints, self.df["labels"].to_numpy()
        model = fit_baseline(X, y, train, test, splitter.seed,
                             self._params("baseline", splitter.distribution))
        preds = model.predict(X[valid])
        return {
            "valid_rmse": root_mean_squared_error(y[valid], preds),
            "valid_r2": r2_score(y[valid], preds),
            "best_epoch": model.best_iteration,
            "run_dir": None,
        }

    def _transformer(self, splitter: Splitters,
                     arm: str) -> dict[str, float | str | None]:
        """
        Fit and score ChemBERTa on the split, frozen or fine-tuned, with the
        hyperparameters tuned for that arm. Only fine-tuned runs write
        checkpoints: a frozen encoder's weights are the published ones at
        every epoch. Of those, only GIF_SEED's keep every epoch rather than
        just the ends.

        Parameters
        ----------
        splitter : Splitters
            Split to fit and score on; its seed also seeds the model.
        arm : str
            'pretrained' to train the head only, or 'finetuned'.

        Returns
        -------
        dict[str, float | str | None]
            Valid RMSE and R2, the selected epoch and the run directory.
        """
        freeze = arm == "pretrained"
        params = self._params(arm, splitter.distribution)
        model = Model(self.df, self.target, splitter, seed=splitter.seed,
                      checkpoint=not freeze,
                      trajectory=splitter.seed == GIF_SEED,
                      freeze=freeze, **params)
        model.fit()
        metrics = model.evaluate()
        return {
            "valid_rmse": metrics["rmse"],
            "valid_r2": metrics["r2"],
            "best_epoch": model.best_epoch,
            "run_dir": str(model.model_dir),
        }

    def _run(self, arm: str, splitter: Splitters) -> dict:
        """
        Fit and score one arm on a split and append its row to the results
        file straight away, so a crash loses at most the run in progress.

        Parameters
        ----------
        arm : str
            One of ARMS.
        splitter : Splitters
            Split to fit and score on.

        Returns
        -------
        dict
            The results row: what was run, the realised split sizes, the
            valid scores and the wall time.
        """
        start = time.time()
        if arm == "mean":
            result = self._mean(splitter)
        elif arm == "cluster_mean":
            result = self._cluster_mean(splitter)
        elif arm == "baseline":
            result = self._baseline(splitter)
        else:
            result = self._transformer(splitter, arm)
        train, test, valid = splitter.split()
        row = {
            "target": self.target, "mutation": self.mutation, "method": METHOD,
            "distribution": splitter.distribution, "arm": arm,
            "seed": splitter.seed, "min_cluster_size": self.min_cluster_size,
            "n_train": len(train), "n_test": len(test), "n_valid": len(valid),
            **result, "seconds": time.time() - start,
        }
        pd.DataFrame([row]).to_csv(self.results_path, mode="a", index=False,
                                   header=not self.results_path.exists())
        return row

    def run(self, arm: str, distribution: str, seed: int) -> dict:
        """
        One cell of the matrix on its own.

        Parameters
        ----------
        arm : str
            One of ARMS.
        distribution : str
            'in' or 'out'.
        seed : int
            Replicate seed, for both the split and the model.

        Returns
        -------
        dict
            The results row, also appended to the results file.
        """
        return self._run(arm, self._splitter(distribution, seed))

    def _done(self) -> set[tuple]:
        """
        Combinations already in the results file.

        Returns
        -------
        set[tuple]
            (arm, distribution, seed) of every finished run.
        """
        if not self.results_path.exists():
            return set()
        return set(pd.read_csv(self.results_path)[KEY].itertuples(index=False, name=None))

    def grid(self, arms: Iterable[str] = ARMS,
             distributions: Iterable[str] = DISTRIBUTIONS,
             seeds: Iterable[int] = range(N_REPEATS)) -> pd.DataFrame:
        """
        Run every missing combination. Arms are the innermost loop so each
        (distribution, seed) builds one splitter and hands it to every arm:
        the arms are compared on the same rows by construction.

        Parameters
        ----------
        arms : Iterable[str]
            Arms to run.
        distributions : Iterable[str]
            Distributions to run.
        seeds : Iterable[int]
            Replicate seeds to run.

        Returns
        -------
        pd.DataFrame
            Every row in the results file, including those from earlier calls.
        """
        done, arms = self._done(), list(arms)
        for distribution, seed in product(distributions, seeds):
            todo = [arm for arm in arms if (arm, distribution, seed) not in done]
            if todo:
                splitter = self._splitter(distribution, seed)
                for arm in todo:
                    self._run(arm, splitter)
        return pd.read_csv(self.results_path)


def experiment(data_path: str | Path, target: str, arm: str, distribution: str,
               seed: int, mutation: str | None = MUTATION) -> dict:
    """
    Run one cell of the matrix from plain strings, preprocessing the target
    from the dump (or its parquet cache) first. For many cells, build one
    Experiment and call grid(): this pays for the distance matrix every call.

    Parameters
    ----------
    data_path : str | Path
        Path to the raw BindingDB TSV dump.
    target : str
        Short target name, a key of data.TARGET.
    arm : str
        One of ARMS.
    distribution : str
        'in' or 'out'.
    seed : int
        Replicate seed, for both the split and the model.
    mutation : str | None
        Variant to keep, see Data.

    Returns
    -------
    dict
        The results row, also appended to the results file.
    """
    df = Data(data_path, target, mutation)._preprocess()
    return Experiment(df, target, mutation).run(arm, distribution, seed)


def main() -> None:
    """
    Command-line entry point, `uv run bertsheep <target>`: preprocess the
    target, tune any (arm, distribution) whose study is short of its trials,
    then run every missing cell of the matrix. Both stages resume, so the same
    command restarts a run that died part way; the narrowing flags are for a
    smoke run whose rows then count towards the full grid, and they narrow the
    tuning too, since each arm reads only its own study.
    """
    parser = argparse.ArgumentParser(prog="bertsheep", description=main.__doc__)
    parser.add_argument("target", help="short target name, a key of data.TARGET")
    parser.add_argument("--mutation", default=MUTATION,
                        help=f"variant to keep (default: {MUTATION})")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS,
                        default=DISTRIBUTIONS)
    parser.add_argument("--seeds", type=int, default=N_REPEATS,
                        help=f"run seeds 0..N-1 (default: {N_REPEATS})")
    args = parser.parse_args()

    df = Data(DUMP_PATH, args.target, args.mutation)._preprocess()
    exp = Experiment(df, args.target, args.mutation)
    exp.tune(args.arms, args.distributions)
    exp.grid(args.arms, args.distributions, range(args.seeds))
    print(f"-- Results in {exp.results_path}")
