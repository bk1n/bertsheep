import sys
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from bertsheep import model as model_module
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

    Two benchmarks, run separately. epoch_time() times one fit at the
    module's default precision, to size the trial budget. autocast() times
    and scores the same fit with bf16 autocast off and on, to check the
    speed-up does not cost accuracy. Each row carries the shape the
    representative split actually lands on, how much of the target is
    one-off chemistry, how long an epoch takes, and where the losses stand
    after the timed epochs. Training is timed through Model's epoch and
    scoring steps rather than through fit(), because fit() writes
    checkpoints and would put disk I/O inside the measurement.

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

        Read off the Model's own split frames rather than by splitting again,
        so the numbers describe the split that was actually trained on.

        Parameters
        ----------
        model : Model
            Model whose splitter and split frames are being described.

        Returns
        -------
        dict[str, float]
            Realised fraction of molecules per split, the cluster count, and
            the fraction of molecules alone in their cluster.
        """
        splitter = model.splitter
        # Kept molecules only, so the counts agree with the split fractions.
        sizes = pd.Series(splitter.clusters[splitter.rows]).value_counts()
        return {
            "train": len(model.train_df) / splitter.n,
            "test": len(model.test_df) / splitter.n,
            "valid": len(model.valid_df) / splitter.n,
            "clusters": len(sizes),
            "singletons": sizes.eq(1).sum() / splitter.n,
        }

    def _time_training(self, model: Model) -> dict[str, float]:
        """
        Seconds per epoch and peak GPU memory for one Model, projected out to a
        full run at the Model's own epoch ceiling, plus the train and test
        losses after the last timed epoch. The first epoch is discarded from
        the timing because it pays for CUDA context setup and the first
        allocation of every buffer, which the epochs after it do not.

        The projection is an upper bound: it assumes every epoch is spent,
        whereas early stopping usually ends a real run sooner. The losses are
        a few epochs into an 80-epoch schedule, still inside warmup at the
        default ratio, so they compare the two precisions early in training
        rather than at convergence.

        Parameters
        ----------
        model : Model
            Model to time. It is left trained for a few epochs, so it should be
            discarded afterwards rather than fitted.

        Returns
        -------
        dict[str, float]
            Mean seconds per epoch, peak allocated VRAM in GiB (NaN on CPU),
            the projected minutes for a full fit, and the final epoch's train
            loss, test loss and test R2.
        """
        cuda = model.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        times = []
        for epoch in range(self.epochs):
            start = time.time()
            train_loss = model._train_epoch(epoch)
            test_loss, preds, labels = model._score(model.test_loader)
            times.append(time.time() - start)

        seconds = np.mean(times[1:])
        return {
            "seconds_per_epoch": seconds,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if cuda else np.nan,
            "projected_fit_minutes": seconds * model.num_epochs / 60,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_r2": model._metrics(preds, labels)["r2"],
        }

    def _arm(self, splitter: Splitters, autocast: bool) -> dict[str, float]:
        """
        Build a fresh Model on the split with bf16 autocast forced on or off,
        then describe the split and time training. `AUTOCAST` is patched on
        the model module because model.autocast() reads it at call time; the
        patch is undone on return so the module default is left as found. The
        Model is local so its weights are freed before the next arm measures
        peak memory.

        Parameters
        ----------
        splitter : Splitters
            Splitter shared by every arm, so all arms train on the same rows.
        autocast : bool
            Whether the forward passes run under bf16 autocast.

        Returns
        -------
        dict[str, float]
            The arm's precision setting, split diagnostics and timings.
        """
        with mock.patch.object(model_module, "AUTOCAST", autocast):
            model = Model(self.df, self.target, splitter)
            return ({"autocast": autocast}
                    | self._split_diagnostics(model) | self._time_training(model))

    def _benchmark(self, autocasts: tuple[bool, ...], name: str) -> pd.DataFrame:
        """
        Run one arm per autocast setting on the representative split and write
        the rows to out/benchmark/<name>-<timestamp>.csv. Every arm shares one
        splitter and the model seed, so the rows differ only in precision. The
        dataset statistics are carried on every row so a saved run is
        self-describing: a timing only means something next to the frame it
        was measured on.

        Parameters
        ----------
        autocasts : tuple[bool, ...]
            Autocast setting for each arm, one row each.
        name : str
            Benchmark name, used as the CSV filename prefix.

        Returns
        -------
        pd.DataFrame
            One measurement row per arm, as written to the CSV.
        """
        stats = self.dataset()
        print(f"-- {self.target}: {stats['rows']} rows, {stats['ligands']} ligands, "
              f"labels {stats['label_mean']:.2f} +/- {stats['label_std']:.2f}")

        print(f"-- Benchmarking {METHOD} ({DISTRIBUTION}-distribution)")
        splitter = Splitters(
            self.df["smiles"], METHOD, DISTRIBUTION,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
        results = pd.DataFrame([
            {"method": METHOD, "distribution": DISTRIBUTION}
            | self._arm(splitter, autocast) | stats
            for autocast in autocasts
        ])

        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        path = BENCHMARK_DIR / f"{name}-{datetime.now():%Y-%m-%d-%H%M%S}.csv"
        results.to_csv(path, index=False)
        print(results.T.to_string(header=False))
        print(f"-- Written to {path}")
        return results

    def epoch_time(self) -> pd.DataFrame:
        """
        Seconds per epoch, peak VRAM and projected fit time at the precision
        real runs use, which is what the trial budget is sized from.

        Returns
        -------
        pd.DataFrame
            The single measurement row, as written to the CSV.
        """
        return self._benchmark((model_module.AUTOCAST,), "epoch-time")

    def autocast(self) -> pd.DataFrame:
        """
        The same fit in fp32 and under bf16 autocast, so the speed and memory
        saved can be read next to any change in train loss, test loss and test
        R2. On CPU autocast is a no-op, so the two rows differ only by timing
        noise.

        Returns
        -------
        pd.DataFrame
            One row for fp32 and one for bf16, as written to the CSV.
        """
        return self._benchmark((False, True), "autocast")

if __name__ == "__main__":
    from bertsheep.data import Data

    target = "EGFR"
    start = time.time()
    df = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", target)._preprocess()
    print(f"-- Preprocessed in {time.time() - start:.0f}s")
    # epoch_time (default) or autocast, e.g. `python -m bertsheep.benchmark autocast`
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "epoch_time"
    epochs = 50 if benchmark == 'autocast' else TIMED_EPOCHS
    getattr(Benchmark(df, target, epochs), benchmark)()
