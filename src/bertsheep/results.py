from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from bertsheep.splitters import DISTRIBUTIONS

# figures/ rather than out/: these are headed for results.md and README.md, so
# they have to be tracked, and out/ is gitignored.
FIGURE_DIR = Path("figures")
DPI = 600
FIGSIZE = (12, 5)
# Share of the points each LOESS fit weighs: wide enough to smooth over seeds,
# narrow enough to keep the elbow of the early epochs.
LOESS_FRAC = 0.3
POINT_SIZE = 5
# Low because a panel holds every seed's every epoch: ~5k points per arm at 30.
POINT_ALPHA = 0.2
# Grey for the baseline, which is a reference line rather than a curve; blue and
# orange are the colourblind-safe pair eda.py settles on.
ARM_COLOURS = {"baseline": "grey", "pretrained": "tab:blue",
               "finetuned": "tab:orange"}
SPLIT_STYLES = {"test": "-", "train": "--"}
# The model loss is MSE on the -ln IC50 (nM) label, see eda.LABEL_AXIS.
LOSS_AXIS = "MSE, -ln IC50 (nM)"


class Results:
    """
    Figures over one experiment's results file, as written by
    Experiment.grid(), for results.md and README.md.

    Parameters
    ----------
    path : Path
        out/experiments/<target>-<mutation>.csv. Taken as a path rather than a
        frame so the figure names always come from the file they describe.
    """

    def __init__(self, path: Path) -> None:
        self.df = pd.read_csv(path)
        self.name = path.stem

    def _histories(self) -> pd.DataFrame:
        """
        Every transformer run's per-epoch train and test loss, one row per
        (run, epoch, split). Driven off the results file's run_dir rather than a
        glob of out/models/, which also holds tuning-trial and legacy runs; the
        baseline has no run_dir and no epochs, so it drops out here.

        Returns
        -------
        pd.DataFrame
            Columns arm, distribution, seed, epoch, split and loss. Epoch -1 has
            a test loss only, since nothing was trained before it.
        """
        runs = self.df.dropna(subset="run_dir")
        histories = pd.concat(
            pd.read_csv(Path(run.run_dir) / "history.csv").assign(
                arm=run.arm, distribution=run.distribution, seed=run.seed)
            for run in runs.itertuples()
        )
        return histories.melt(
            id_vars=["arm", "distribution", "seed", "epoch"],
            value_vars=["train_loss", "test_loss"], var_name="split",
            value_name="loss",
        ).dropna().assign(split=lambda d: d["split"].str.removesuffix("_loss"))

    def loss_curves(self) -> Path:
        """
        Train and test loss per epoch for the two transformer arms, one panel
        per distribution: every seed's value as a point, with a LOESS trend per
        arm and split. The baseline counts boosting rounds, not epochs, so it
        is a horizontal line at its mean valid MSE rather than a curve.

        Early stopping ends seeds at different epochs, so each trend's tail is
        fitted only on the seeds that were still improving there -- read the
        late epochs as the survivors' curve, not the arm's.

        Returns
        -------
        Path
            The saved figure.
        """
        histories = self._histories()
        baseline = self.df[self.df["arm"] == "baseline"]
        fig, axes = plt.subplots(1, len(DISTRIBUTIONS), figsize=FIGSIZE,
                                 sharey=True)
        for ax, distribution in zip(axes, DISTRIBUTIONS):
            panel = histories[histories["distribution"] == distribution]
            for (arm, split), group in panel.groupby(["arm", "split"]):
                colour = ARM_COLOURS[arm]
                ax.scatter(group["epoch"], group["loss"], s=POINT_SIZE,
                           alpha=POINT_ALPHA, color=colour, linewidths=0)
                ax.plot(*lowess(group["loss"], group["epoch"], frac=LOESS_FRAC).T,
                        color=colour, ls=SPLIT_STYLES[split],
                        label=f"{arm} {split}")
            valid_mse = (baseline.loc[baseline["distribution"] == distribution,
                                      "valid_rmse"] ** 2).mean()
            ax.axhline(valid_mse, color=ARM_COLOURS["baseline"], ls=":",
                       label="baseline valid")
            ax.set(title=f"{distribution}-distribution", xlabel="epoch")
        axes[0].set_ylabel(LOSS_AXIS)
        axes[-1].legend()
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_loss_curves.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return path
