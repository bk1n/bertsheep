import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bertsheep.model import Model
from bertsheep.splitters import Splitters

BENCHMARK_DIR = Path("out/benchmark")
TIMED_EPOCHS = 3  # the first is discarded as warmup
# Epoch cost does not depend on which split is cut: the frame is the same size
# and only the row assignment changes, so one representative split is timed.
METHOD, DISTRIBUTION = "scaffold", "out"
MIN_CLUSTER_SIZE = 10


class Benchmark:
    """
    Measure what one fit costs on this machine, so the experiment matrix is
    sized from timings rather than from guesses.

    Produces a single row: the shape the representative split actually lands
    on, how much of the target is one-off chemistry, and how long an epoch
    takes. Training is timed through Model's epoch and scoring
    steps rather than through fit(), because fit() writes a checkpoint every
    epoch and would put disk I/O inside the measurement.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns, as returned by
        Data._preprocess.
    target : str
        Short target name, passed through to Model.
    epochs : int
        Epochs to time; the first is discarded as warmup.
    """

    def __init__(self, df: pd.DataFrame, target: str,
                 epochs: int = TIMED_EPOCHS) -> None:
        self.df = df
        self.target = target
        self.epochs = epochs

    def dataset(self) -> dict[str, float]:
        """
        Size and spread of the frame every fit trains on. The distinct ligand
        count sits beside the row count because deduplication keys on
        (smiles, mutations): a ligand measured against several constructs is
        several rows, and those rows can be dealt to opposite sides of a split.
        The gap between the two counts is how much room that leaves.

        Returns
        -------
        dict[str, float]
            Row count, distinct ligand count, and label mean and standard
            deviation.
        """
        return {
            "rows": len(self.df),
            "ligands": self.df["smiles"].nunique(),
            "label_mean": self.df["labels"].mean(),
            "label_std": self.df["labels"].std(),
        }

    def _split_diagnostics(self, model: Model) -> dict[str, float]:
        """
        What the split a Model was built on actually produced, rather than what
        it was asked for. Neither cut is obliged to honour its requested sizes:
        GroupShuffleSplit's are fractions of clusters, not of molecules, and
        StratifiedKFold's are quantised to 1/n_splits. The realised fractions
        are the ones results have to be reported against.

        The singleton fraction is the share of the target that is one-off
        chemistry. It bounds how honest an in-distribution split can be, since
        a cluster with one member cannot be represented on both sides of one.

        Read off the Model's own split frames rather than by splitting again:
        the fingerprint deal is quadratic in the number of molecules, so a
        second call to get the same deterministic answer is not free.

        Parameters
        ----------
        model : Model
            Model whose splitter and split frames are being described.

        Returns
        -------
        dict[str, float]
            Realised fraction of molecules per split, the cluster count, and
            the fraction of molecules alone in their cluster. The cluster
            columns are NaN for the fingerprint split, which does not group.
        """
        splitter = model.splitter
        clusters = getattr(splitter, "clusters", None)
        # Kept molecules only, so the counts agree with the split fractions.
        sizes = None if clusters is None else pd.Series(clusters[splitter.rows]).value_counts()
        return {
            "train": len(model.train_df) / splitter.n,
            "test": len(model.test_df) / splitter.n,
            "valid": len(model.valid_df) / splitter.n,
            "clusters": np.nan if sizes is None else len(sizes),
            "singletons": np.nan if sizes is None else sizes.eq(1).sum() / splitter.n,
        }

    def _time_training(self, model: Model) -> dict[str, float]:
        """
        Seconds per epoch and peak GPU memory for one Model, projected out to a
        full run at the Model's own epoch ceiling. The first epoch is discarded
        because it pays for CUDA context setup and the first allocation of every
        buffer, which the epochs after it do not.

        The projection is an upper bound: it assumes every epoch is spent,
        whereas early stopping usually ends a real run sooner.

        Parameters
        ----------
        model : Model
            Model to time. It is left trained for a few epochs, so it should be
            discarded afterwards rather than fitted.

        Returns
        -------
        dict[str, float]
            Mean seconds per epoch, peak allocated VRAM in GiB (NaN on CPU),
            and the projected minutes for a full fit.
        """
        cuda = model.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        times = []
        for epoch in range(self.epochs):
            start = time.time()
            model._train_epoch(epoch)
            model._score(model.test_loader)
            times.append(time.time() - start)

        seconds = np.mean(times[1:])
        return {
            "seconds_per_epoch": seconds,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if cuda else np.nan,
            "projected_fit_minutes": seconds * model.num_epochs / 60,
        }

    def run(self) -> pd.DataFrame:
        """
        Benchmark the representative split end to end and write the row to
        out/benchmark/<timestamp>.csv: build a Model on the split, describe
        what the split landed on, then time training. The dataset statistics are
        carried on the row so a saved run is self-describing: a timing only
        means something next to the frame it was measured on.

        Returns
        -------
        pd.DataFrame
            The single measurement row, as written to the CSV.
        """
        stats = self.dataset()
        print(f"-- {self.target}: {stats['rows']} rows, {stats['ligands']} ligands, "
              f"labels {stats['label_mean']:.2f} +/- {stats['label_std']:.2f}")

        print(f"-- Benchmarking {METHOD} ({DISTRIBUTION}-distribution)")
        splitter = Splitters(
            self.df["smiles"], METHOD, DISTRIBUTION,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
        model = Model(self.df, self.target, splitter)
        results = pd.DataFrame([
            {"method": METHOD, "distribution": DISTRIBUTION}
            | self._split_diagnostics(model) | self._time_training(model) | stats
        ])

        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        path = BENCHMARK_DIR / f"{datetime.now():%Y-%m-%d-%H%M%S}.csv"
        results.to_csv(path, index=False)
        print(results.T.to_string(header=False))
        print(f"-- Written to {path}")
        return results


if __name__ == "__main__":
    from bertsheep.data import Data

    target = "EGFR"
    start = time.time()
    df = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", target)._preprocess()
    print(f"-- Preprocessed in {time.time() - start:.0f}s")
    Benchmark(df, target).run()
