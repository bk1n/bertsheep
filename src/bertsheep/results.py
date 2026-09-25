import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess

from bertsheep.splitters import DISTRIBUTIONS

# figures/ rather than out/: these are headed for results.md and README.md, so
# they have to be tracked, and out/ is gitignored.
FIGURE_DIR = Path("figures")
DPI = 600
# Share of the points each LOESS fit weighs: wide enough to smooth over seeds,
# narrow enough to keep the elbow of the early epochs.
LOESS_FRAC = 0.3
POINT_SIZE = 5
# Low because a panel holds every seed's every epoch: ~5k points per arm at 30.
POINT_ALPHA = 0.2
# Greys for the references, which are lines rather than curves, the lighter for
# the one with no chemistry in it; blue and orange are the colourblind-safe
# pair eda.py settles on.
ARM_COLOURS = {"cluster_mean": "lightgrey", "baseline": "grey",
               "pretrained": "tab:blue", "finetuned": "tab:orange"}
# The baseline is XGBoost on Morgan fingerprints; the last two are ChemBERTa.
ARM_LABELS = {"cluster_mean": "Cluster mean", "baseline": "XGBoost + FPS",
              "pretrained": "Pretrained", "finetuned": "Fine-tuned"}
# Arms scored once on valid rather than per epoch, drawn as horizontal lines.
REFERENCE_ARMS = ("cluster_mean", "baseline")
SPLIT_LABELS = {"train": "train", "test": "test", "valid": "validation"}
# Dotted for valid, so the references -- also valid scores -- take dash-dot.
SPLIT_STYLES = {"train": "--", "test": "-", "valid": ":"}
# The model loss is MSE on the -ln IC50 (nM) label, see eda.LABEL_AXIS.
LOSS_AXIS = "MSE, -ln IC50 (nM)"
# Early stopping ends seeds at different epochs, so an epoch is only plotted
# while at least this share of an arm's seeds are still training; past it the
# curve would be the few longest runs, not the arm.
SURVIVOR_SHARE = 0.3
Q1_FIGSIZE = (12, 8)
# Boxes need less room than curves over tens of epochs, but enough that four
# arm names fit under them.
Q1_WIDTH_RATIOS = (2, 3)


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
        Every transformer run's per-epoch train, test and valid loss, one row per
        (run, epoch, split). Driven off the results file's run_dir rather than a
        glob of out/models/, which also holds tuning-trial and legacy runs; the
        baseline has no run_dir and no epochs, so it drops out here.

        Returns
        -------
        pd.DataFrame
            Columns arm, distribution, seed, epoch, split and loss. Epoch -1 has
            test and valid losses only, since nothing was trained before it.
        """
        runs = self.df.dropna(subset="run_dir")
        histories = pd.concat(
            pd.read_csv(Path(run.run_dir) / "history.csv").assign(
                arm=run.arm, distribution=run.distribution, seed=run.seed)
            for run in runs.itertuples()
        )
        return histories.melt(
            id_vars=["arm", "distribution", "seed", "epoch"],
            value_vars=["train_loss", "test_loss", "valid_loss"], var_name="split",
            value_name="loss",
        ).dropna().assign(split=lambda d: d["split"].str.removesuffix("_loss"))

    def _surviving(self) -> pd.DataFrame:
        """
        The histories cut, per arm and distribution, at the first epoch fewer
        than SURVIVOR_SHARE of the seeds reach. A seed that stopped early has
        no rows after its last epoch, so the survivors thin out with epoch and
        this keeps a prefix of each curve.

        Returns
        -------
        pd.DataFrame
            As _histories(), less the epochs too few seeds reached.
        """
        histories = self._histories()
        runs = histories.groupby(["arm", "distribution"])["seed"].transform("nunique")
        alive = histories.groupby(["arm", "distribution", "epoch"])["seed"].transform(
            "nunique")
        return histories[alive >= SURVIVOR_SHARE * runs]

    def _loss_panel(self, ax: plt.Axes, histories: pd.DataFrame,
                    distribution: str) -> None:
        """
        Train, test and valid loss per epoch for the two transformer arms on one
        distribution: every seed's value as a point, with a LOESS trend per arm
        and split. The reference arms have no epochs -- the baseline counts
        boosting rounds, the cluster mean fits nothing -- so each is a
        horizontal line at its mean valid MSE rather than a curve.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        histories : pd.DataFrame
            From _surviving(), taken as an argument so the history files are
            read once per figure rather than once per panel.
        distribution : str
            One of DISTRIBUTIONS.
        """
        panel = histories[histories["distribution"] == distribution]
        for (arm, split), group in panel.groupby(["arm", "split"]):
            colour = ARM_COLOURS[arm]
            ax.scatter(group["epoch"], group["loss"], s=POINT_SIZE,
                       alpha=POINT_ALPHA, color=colour, linewidths=0)
            ax.plot(*lowess(group["loss"], group["epoch"], frac=LOESS_FRAC).T,
                    color=colour, ls=SPLIT_STYLES[split], label=f"{ARM_LABELS[arm]} {SPLIT_LABELS[split]}")
        for arm in REFERENCE_ARMS:
            scores = self.df[(self.df["arm"] == arm)
                             & (self.df["distribution"] == distribution)]
            ax.axhline((scores["valid_rmse"] ** 2).mean(), color=ARM_COLOURS[arm],
                       ls="-.", label=f"{ARM_LABELS[arm]} {SPLIT_LABELS['valid']}")
        ax.set(title=f"{distribution}-distribution", xlabel="Epoch",
               ylabel=LOSS_AXIS)
        # Epochs are whole; left alone, the out panel's short range gets halves.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _r2_panel(self, ax: plt.Axes, distribution: str) -> None:
        """
        Valid R2 per arm over the seeds on one distribution, each box in its
        arm's colour so it reads against the loss curves. Valid is the one
        split no arm's model selection read, so it is the only fair ground for
        comparing them.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        distribution : str
            One of DISTRIBUTIONS.
        """
        arms = list(ARM_COLOURS)
        scores = self.df[self.df["distribution"] == distribution]
        boxes = ax.boxplot(
            [scores.loc[scores["arm"] == arm, "valid_r2"] for arm in arms],
            tick_labels=[ARM_LABELS[arm] for arm in arms], patch_artist=True, medianprops={"color": "black"},
        )
        for box, arm in zip(boxes["boxes"], arms):
            box.set_facecolor(ARM_COLOURS[arm])
        ax.set(title=f"{distribution}-distribution", ylabel="Validation Set R²")

    def q1(self) -> Path:
        """
        README's Question 1 figure: one row per distribution, valid R2 boxes on
        the left and loss curves on the right. Each column shares its y axis, so
        the in/out gap -- how much of an arm's score depends on having seen the
        scaffold series -- reads straight down it.

        Returns
        -------
        Path
            The saved figure.
        """
        histories = self._surviving()
        fig, axes = plt.subplots(len(DISTRIBUTIONS), 2, figsize=Q1_FIGSIZE,
                                 sharey="col", width_ratios=Q1_WIDTH_RATIOS,
                                 layout="constrained")
        for (r2_ax, loss_ax), distribution in zip(axes, DISTRIBUTIONS):
            self._r2_panel(r2_ax, distribution)
            self._loss_panel(loss_ax, histories, distribution)
        axes[0, 1].legend()
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q1.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return path


if __name__ == "__main__":
    # The results file is an argument rather than a literal, so this is not
    # the hardcoded-target __main__ pattern todo.md is undoing elsewhere.
    parser = argparse.ArgumentParser(description=Results.__doc__)
    parser.add_argument("results", type=Path,
                        help="out/experiments/<target>-<mutation>.csv")
    results = Results(parser.parse_args().results)
    print(results.q1())
