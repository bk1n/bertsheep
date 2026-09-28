import argparse
import json
from collections.abc import Callable
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import umap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

from bertsheep.chemistry import Chemist
from bertsheep.data import CACHE_DIR, Data
from bertsheep.eda import (
    GREY,
    LABEL_AXIS,
    LABEL_CLIP,
    LABEL_CMAP,
    TOP_CLUSTERS,
    UMAP_MIN_DIST,
    UMAP_NEIGHBOURS,
    UMAP_SEED,
)
from bertsheep.experiment import GIF_SEED
from bertsheep.model import (
    CONFIG,
    EPOCH_CHECKPOINT,
    HISTORY,
    INIT_CHECKPOINT,
    SPLIT_NAMES,
    embeddings,
    split_frames,
)
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
# Every score is RMSE, in the label's own units (see eda.LABEL_AXIS), so the
# boxes and curves read on one scale. The model trains on MSE; history.csv's
# losses are split-wide MSE, so their square root is that split's RMSE.
RMSE_AXIS = f"RMSE, {LABEL_AXIS}"
# Early stopping ends seeds at different epochs, so an epoch is only plotted
# while at least this share of an arm's seeds are still training; past it the
# curve would be the few longest runs, not the arm.
SURVIVOR_SHARE = 0.3
Q1_FIGSIZE = (12, 8)
# Boxes need less room than curves over tens of epochs, but enough that four
# arm names fit under them.
Q1_WIDTH_RATIOS = (2, 3)
# Arms tested against each other, in box order. "mean" is left out: every model
# beats it by a mile, and out-of-distribution it scores exactly as the cluster
# mean does, so it adds tests without adding information -- and, sharing the
# mixed model's residual variance, it would move the others' standard errors.
COMPARED_ARMS = ("cluster_mean", "baseline", "pretrained", "finetuned")
# Holm keeps Bonferroni's family-wise error rate and is never less powerful.
P_ADJUST = "holm"
SIGNIFICANCE_BINS = [0, 0.001, 0.01, 0.05, 1]
SIGNIFICANCE_LABELS = ["***", "**", "*", "ns"]
# Gap between stacked brackets, as a share of the valid RMSE's range over both
# distributions: the box rows share a y axis, so one step suits both.
BRACKET_STEP = 0.08

# Aligned UMAP coordinates per GIF run. Cached because embedding every
# checkpoint and aligning the epochs takes minutes, and restyling a figure
# should not repeat it. Keyed on the run alone: delete the file after changing
# GIF_LAYERS, SUBSAMPLE, LATENT_METRIC or the UMAP constants.
LATENT_DIR = Path("out/latent")
# Hidden-state slices drawn: the embedding layer, the middle and the last of
# ChemBERTa-10M-MTR's three encoder layers -- README's first, middle, last.
GIF_LAYERS = (0, 2, 3)
LAYER_LABELS = {0: "Embedding layer", 1: "Encoder layer 1",
                2: "Encoder layer 2", 3: "Encoder layer 3"}
# Share of each split's molecules embedded and aligned. Aligned UMAP's cost
# grows with molecules x epochs, ~1.5 h a panel at the full ~8.8k; taken per
# split so the train/test/valid proportions are the run's own.
SUBSAMPLE = 0.2
SUBSAMPLE_SEED = 0
# Transformer embeddings differ more in direction than in length.
LATENT_METRIC = "cosine"
# Frames interpolated per epoch. Aligned UMAP keeps each molecule's identity
# across epochs, so the in-betweens are real paths the eye can follow rather
# than ~9k points jumping at once.
TWEEN_FRAMES = 4
HOLD_FRAMES = 12  # repeats of the first and last frame, PillowWriter's fps being fixed
GIF_FPS = 10
GIF_DPI = 100  # the GIF is embedded in README.md, so kept to a few MB
GIF_FIGSIZE = (17, 8)
GIF_WIDTH_RATIOS = (1, 1, 1, 1.2)
# Valid molecules were never trained on, so they are drawn larger and opaque
# over the faint train and test molecules: generalisation reads as the held-out
# points landing among training points of their colour.
VALID_POINT_SIZE = 10
VALID_ALPHA = 0.8
# Filmstrip columns besides the best and last epochs, as shares of the run.
FILMSTRIP_FRACTIONS = (0, 0.25, 0.5)
FILMSTRIP_FIGSIZE = (14, 16)
# Butina's cluster IDs are ranked by size, so ID n is the n-th largest and
# indexes this map directly; every ID from TOP_CLUSTERS on shares the grey.
CLUSTER_CMAP = ListedColormap(
    [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors][:TOP_CLUSTERS]
    + [GREY]
)
COLOURINGS = {"label": "affinity",
              "cluster": f"Butina cluster (top {TOP_CLUSTERS}, grey = other)"}


class Trajectory(NamedTuple):
    """
    One GIF run's latent space over fine-tuning.

    Parameters
    ----------
    distribution : str
        One of DISTRIBUTIONS.
    molecules : pd.DataFrame
        Every molecule the run was split over, with `split`, `labels` and
        `cluster` columns, in the order of `coords`' molecule axis.
    coords : np.ndarray
        Aligned UMAP coordinates, shape (len(GIF_LAYERS), epochs, molecules, 2).
    history : pd.DataFrame
        The run's history.csv, one row per epoch from -1.
    best_epoch : int
        The epoch the run selected on test loss.
    """

    distribution: str
    molecules: pd.DataFrame
    coords: np.ndarray
    history: pd.DataFrame
    best_epoch: int


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
        Every transformer run's per-epoch train, test and valid RMSE, one row
        per (run, epoch, split). Driven off the results file's run_dir rather than a
        glob of out/models/, which also holds tuning-trial and legacy runs; the
        baseline has no run_dir and no epochs, so it drops out here.

        Returns
        -------
        pd.DataFrame
            Columns arm, distribution, seed, epoch, split and rmse. Epoch -1 has
            test and valid scores only, since nothing was trained before it.
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
            value_name="rmse",
        ).dropna().assign(split=lambda d: d["split"].str.removesuffix("_loss"),
                          rmse=lambda d: np.sqrt(d["rmse"]))

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
        Train, test and valid RMSE per epoch for the two transformer arms on one
        distribution: every seed's value as a point, with a LOESS trend per arm
        and split. The reference arms have no epochs -- the baseline counts
        boosting rounds, the cluster mean fits nothing -- so each is a
        horizontal line at its mean valid RMSE rather than a curve.

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
            ax.scatter(group["epoch"], group["rmse"], s=POINT_SIZE,
                       alpha=POINT_ALPHA, color=colour, linewidths=0)
            ax.plot(*lowess(group["rmse"], group["epoch"], frac=LOESS_FRAC).T,
                    color=colour, ls=SPLIT_STYLES[split], label=f"{ARM_LABELS[arm]} {SPLIT_LABELS[split]}")
        for arm in REFERENCE_ARMS:
            scores = self.df[(self.df["arm"] == arm)
                             & (self.df["distribution"] == distribution)]
            ax.axhline(scores["valid_rmse"].mean(), color=ARM_COLOURS[arm],
                       ls="-.", label=f"{ARM_LABELS[arm]} {SPLIT_LABELS['valid']}")
        ax.set(title=f"{distribution}-distribution", xlabel="Epoch",
               ylabel=RMSE_AXIS)
        # Epochs are whole; left alone, the out panel's short range gets halves.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _pairwise(self, distribution: str) -> pd.DataFrame:
        """
        Every pair of COMPARED_ARMS tested on one distribution's valid RMSE,
        with a linear mixed model that gives each seed a random intercept.
        Every arm sees the same split for a given seed, and some splits are
        harder than others for every arm, so the seed is a block. Each seed's
        intercept takes out that shared difficulty before the arms are compared.

        The arms are coded as cell means (no intercept), so each fixed effect is
        an arm's mean RMSE and a pair's contrast is one row of +1/-1. The
        contrasts go through t_test by hand: MixedLM's t_test takes fixed-effect
        columns only, and the built-in t_test_pairwise also counts the seed
        variance as a column, so it cannot run on MixedLM.
        In-distribution the seed variance is close to zero, and statsmodels warns
        that the estimate is on the boundary. That is expected, and the
        contrasts are still valid.

        Parameters
        ----------
        distribution : str
            One of DISTRIBUTIONS.

        Returns
        -------
        pd.DataFrame
            One row per pair, arm_a before arm_b in COMPARED_ARMS order.
            difference is arm_a's RMSE minus arm_b's, so a negative value means
            arm_a scores better. The p-values are Wald z tests, with p_adjusted
            corrected for this distribution's pairs.
        """
        scores = self.df[(self.df["distribution"] == distribution)
                         & self.df["arm"].isin(COMPARED_ARMS)]
        scores = scores.assign(arm=pd.Categorical(scores["arm"], COMPARED_ARMS))
        fit = smf.mixedlm("valid_rmse ~ 0 + arm", scores, groups=scores["seed"]).fit()
        pairs = pd.DataFrame(combinations(COMPARED_ARMS, 2), columns=["arm_a", "arm_b"])
        means = pd.DataFrame(np.eye(len(COMPARED_ARMS)), index=COMPARED_ARMS)
        tests = fit.t_test(means.loc[pairs["arm_a"]].to_numpy()
                           - means.loc[pairs["arm_b"]].to_numpy()).summary_frame()
        return pairs.assign(
            distribution=distribution,
            difference=tests["coef"].to_numpy(),
            std_err=tests["std err"].to_numpy(),
            ci_low=tests["Conf. Int. Low"].to_numpy(),
            ci_high=tests["Conf. Int. Upp."].to_numpy(),
            p=tests["P>|z|"].to_numpy(),
            p_adjusted=multipletests(tests["P>|z|"], method=P_ADJUST)[1],
        )

    def comparisons(self) -> pd.DataFrame:
        """
        The pairwise arm tests for every distribution. Each distribution gets
        its own mixed model: seeds are not shared between them (each has its
        own splitter), and out-of-distribution scores vary about ten times more,
        so one pooled residual variance would suit neither.

        Returns
        -------
        pd.DataFrame
            As _pairwise(), indexed by distribution, arm_a and arm_b.
        """
        return pd.concat(self._pairwise(d) for d in DISTRIBUTIONS).set_index(
            ["distribution", "arm_a", "arm_b"])

    def _brackets(self, ax: plt.Axes, pairs: pd.DataFrame, top: float) -> None:
        """
        Put a bracket over each pair of boxes, labelled with the stars for its
        adjusted p-value. Brackets are stacked with the shortest lowest, so a
        bracket never cuts through a longer one above it.

        Parameters
        ----------
        ax : plt.Axes
            The box panel.
        pairs : pd.DataFrame
            One distribution's rows of comparisons(), indexed by arm_a and arm_b.
        top : float
            The panel's highest score, which the first bracket sits above.
        """
        compared = self.df.loc[self.df["arm"].isin(COMPARED_ARMS), "valid_rmse"]
        step = BRACKET_STEP * np.ptp(compared)
        # boxplot's default positions: 1 for the first box, 2 for the next.
        pairs = pairs.assign(
            left=[COMPARED_ARMS.index(a) + 1 for a, _ in pairs.index],
            right=[COMPARED_ARMS.index(b) + 1 for _, b in pairs.index],
            stars=pd.cut(pairs["p_adjusted"], SIGNIFICANCE_BINS,
                         labels=SIGNIFICANCE_LABELS, include_lowest=True),
        ).assign(span=lambda d: d["right"] - d["left"]).sort_values(["span", "left"])
        for level, pair in enumerate(pairs.itertuples(), start=1):
            y = top + level * step
            ax.plot([pair.left, pair.left, pair.right, pair.right],
                    [y - step / 4, y, y, y - step / 4], color="black", lw=1)
            ax.text((pair.left + pair.right) / 2, y, pair.stars, ha="center",
                    va="bottom", fontsize="small")

    def _rmse_panel(self, ax: plt.Axes, distribution: str,
                    comparisons: pd.DataFrame) -> None:
        """
        Valid RMSE per arm over the seeds on one distribution, each box in its
        arm's colour so it reads against the loss curves. Valid is the one
        split no arm's model selection read, so it is the only fair ground for
        comparing them. Brackets above the boxes give each pair's adjusted
        significance.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        distribution : str
            One of DISTRIBUTIONS.
        comparisons : pd.DataFrame
            From comparisons(), passed in so the mixed models are fitted once
            per figure rather than once per panel.
        """
        scores = self.df[(self.df["distribution"] == distribution)
                         & self.df["arm"].isin(COMPARED_ARMS)]
        boxes = ax.boxplot(
            [scores.loc[scores["arm"] == arm, "valid_rmse"] for arm in COMPARED_ARMS],
            tick_labels=[ARM_LABELS[arm] for arm in COMPARED_ARMS], patch_artist=True,
            medianprops={"color": "black"},
        )
        for box, arm in zip(boxes["boxes"], COMPARED_ARMS):
            box.set_facecolor(ARM_COLOURS[arm])
        self._brackets(ax, comparisons.loc[distribution], scores["valid_rmse"].max())
        ax.set(title=f"{distribution}-distribution", ylabel=f"Validation {RMSE_AXIS}")

    def q1(self) -> Path:
        """
        README's Question 1 figure: one row per distribution, valid RMSE boxes
        on the left, with pairwise significance brackets, and RMSE curves on the
        right. Each column shares its y axis, so the in/out gap -- how much of an
        arm's score depends on having seen the scaffold series -- reads straight
        down it.

        Returns
        -------
        Path
            The saved figure.
        """
        histories = self._surviving()
        comparisons = self.comparisons()
        fig, axes = plt.subplots(len(DISTRIBUTIONS), 2, figsize=Q1_FIGSIZE,
                                 sharey="col", width_ratios=Q1_WIDTH_RATIOS,
                                 layout="constrained")
        for (box_ax, curve_ax), distribution in zip(axes, DISTRIBUTIONS):
            self._rmse_panel(box_ax, distribution, comparisons)
            self._loss_panel(curve_ax, histories, distribution)
        axes[0, 1].legend()
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q1.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    def _frame(self, config: dict) -> pd.DataFrame:
        """
        The preprocessed frame the GIF runs were split over, rebuilt from the
        data path, target and mutation their config.json records, with each
        molecule's Butina cluster. The clusters come from the same cached
        distance matrix Experiment splits with, so they are the groups the
        out-of-distribution split held out.

        Parameters
        ----------
        config : dict
            A GIF run's config.json.

        Returns
        -------
        pd.DataFrame
            Data._preprocess()'s frame plus a `cluster` column; 0 is the largest.
        """
        frame = Data(config["data_path"], config["target"],
                     config["mutation"])._preprocess()
        chemist = Chemist()
        distances = chemist.cached_tanimoto(
            chemist.fingerprints(frame["smiles"]), CACHE_DIR,
            f"{config['target']}-{config['mutation'] or 'pooled'}",
        )
        return frame.assign(cluster=chemist.butina_clusters(distances))

    def _coordinates(self, run_dir: Path, smiles: pd.Series,
                     epochs: np.ndarray) -> np.ndarray:
        """
        Aligned UMAP of every epoch's embeddings, one alignment per layer. The
        same molecules are embedded at every epoch, so each is related to
        itself in the next, and the alignment keeps the layout still wherever
        fine-tuning did not move a molecule -- what moves is what changed.

        Parameters
        ----------
        run_dir : Path
            A run that kept every epoch's checkpoint.
        smiles : pd.Series
            Molecules to embed.
        epochs : np.ndarray
            The run's epochs from -1, one checkpoint each.

        Returns
        -------
        np.ndarray
            Shape (len(GIF_LAYERS), len(epochs), len(smiles), 2).
        """
        path = LATENT_DIR / f"{run_dir.name}.npy"
        if path.exists():
            return np.load(path)
        model_link = json.loads((run_dir / CONFIG).read_text())["model_link"]
        checkpoints = [
            run_dir / "checkpoints" / (INIT_CHECKPOINT if epoch < 0
                                       else EPOCH_CHECKPOINT.format(epoch))
            for epoch in epochs
        ]
        # (epochs, layers, molecules, hidden) -> (layers, epochs, molecules, hidden)
        vectors = np.stack([embeddings(checkpoint, smiles, model_link)[list(GIF_LAYERS)]
                            for checkpoint in checkpoints], axis=1)
        identity = {i: i for i in range(len(smiles))}
        coords = np.stack([
            umap.AlignedUMAP(
                n_neighbors=UMAP_NEIGHBOURS, min_dist=UMAP_MIN_DIST,
                metric=LATENT_METRIC, random_state=UMAP_SEED,
            ).fit(list(layer), relations=[identity] * (len(epochs) - 1)).embeddings_
            for layer in vectors
        ])
        LATENT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, coords)
        return coords

    @cached_property
    def trajectories(self) -> list[Trajectory]:
        """
        The latent-space trajectories of GIF_SEED's fine-tuned runs, the only
        runs that keep a checkpoint per epoch, in DISTRIBUTIONS order, over a
        SUBSAMPLE of each split. The sample is seeded, so it is the same
        molecules the cached coordinates were fitted on. Cached because every
        Question 2 figure draws from them.

        Returns
        -------
        list[Trajectory]
            One per distribution.
        """
        runs = self.df[(self.df["arm"] == "finetuned") & (self.df["seed"] == GIF_SEED)]
        runs = runs.set_index("distribution").loc[list(DISTRIBUTIONS)]
        run_dirs = [Path(run_dir) for run_dir in runs["run_dir"]]
        frame = self._frame(json.loads((run_dirs[0] / CONFIG).read_text()))
        trajectories = []
        for (distribution, run), run_dir in zip(runs.iterrows(), run_dirs):
            molecules = pd.concat(
                split.sample(frac=SUBSAMPLE, random_state=SUBSAMPLE_SEED).assign(split=name)
                for name, split in zip(SPLIT_NAMES, split_frames(frame, run_dir))
            )
            history = pd.read_csv(run_dir / HISTORY)
            coords = self._coordinates(run_dir, molecules["smiles"],
                                       history["epoch"].to_numpy())
            trajectories.append(Trajectory(distribution, molecules, coords,
                                           history, int(run["best_epoch"])))
        return trajectories

    def _label_norm(self) -> Normalize:
        """
        One affinity colour scale for every panel, clipped at LABEL_CLIP as in
        eda.py so a few extreme labels do not wash out the rest.

        Returns
        -------
        Normalize
            Maps -ln IC50 onto LABEL_CMAP.
        """
        labels = pd.concat(t.molecules["labels"] for t in self.trajectories)
        return Normalize(*labels.quantile([LABEL_CLIP, 1 - LABEL_CLIP]))

    def _colours(self, trajectory: Trajectory, colour: str) -> np.ndarray:
        """
        Each molecule's colour, worked out once so a frame only moves points.

        Parameters
        ----------
        trajectory : Trajectory
            Run whose molecules to colour.
        colour : str
            A key of COLOURINGS.

        Returns
        -------
        np.ndarray
            RGBA per molecule, shape (molecules, 4).
        """
        if colour == "label":
            return plt.get_cmap(LABEL_CMAP)(self._label_norm()(trajectory.molecules["labels"]))
        return CLUSTER_CMAP(np.minimum(trajectory.molecules["cluster"], TOP_CLUSTERS))

    def _at(self, trajectory: Trajectory, layer: int, epoch: float) -> np.ndarray:
        """
        Coordinates at a fractional epoch, linearly between the two epochs
        either side, and held at the last epoch once the run has stopped.

        Parameters
        ----------
        trajectory : Trajectory
            Run to read.
        layer : int
            Position in GIF_LAYERS.
        epoch : float
            Epoch from -1; the fraction is the way to the next one.

        Returns
        -------
        np.ndarray
            Shape (molecules, 2).
        """
        coords = trajectory.coords[layer]
        step = min(epoch + 1, len(coords) - 1)  # epoch -1 is index 0
        low = int(step)
        high = min(low + 1, len(coords) - 1)
        return coords[low] + (step - low) * (coords[high] - coords[low])

    def _latent_panel(self, ax: plt.Axes, trajectory: Trajectory, layer: int,
                      colours: np.ndarray) -> Callable[[float], None]:
        """
        One layer's latent space for one run, with the limits fixed over every
        epoch so the frame holds still while the points move. UMAP's axes carry
        no units, so their ticks are dropped.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        trajectory : Trajectory
            Run to draw.
        layer : int
            Position in GIF_LAYERS.
        colours : np.ndarray
            From _colours().

        Returns
        -------
        Callable[[float], None]
            Moves the points to a (fractional) epoch.
        """
        valid = (trajectory.molecules["split"] == "valid").to_numpy()
        context = ax.scatter(*trajectory.coords[layer, 0, ~valid].T, s=POINT_SIZE,
                             c=colours[~valid], alpha=POINT_ALPHA, linewidths=0)
        held_out = ax.scatter(*trajectory.coords[layer, 0, valid].T,
                              s=VALID_POINT_SIZE, c=colours[valid],
                              alpha=VALID_ALPHA, linewidths=0)
        ax.update_datalim(trajectory.coords[layer].reshape(-1, 2))
        ax.autoscale_view()
        ax.set(xticks=[], yticks=[])

        def move(epoch: float) -> None:
            coords = self._at(trajectory, layer, epoch)
            context.set_offsets(coords[~valid])
            held_out.set_offsets(coords[valid])

        return move

    def _history_panel(self, ax: plt.Axes,
                       trajectory: Trajectory) -> Callable[[float], None]:
        """
        The run's train, test and valid RMSE, styled as in q1(), with the
        selected epoch starred on the test curve it was selected on. A cursor
        tracks the frame's epoch, so each frame of the latent space can be read
        against where training was.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        trajectory : Trajectory
            Run to draw.

        Returns
        -------
        Callable[[float], None]
            Moves the cursor, and the title's epoch, to a (fractional) epoch.
        """
        history = trajectory.history
        for split, label in SPLIT_LABELS.items():
            ax.plot(history["epoch"], np.sqrt(history[f"{split}_loss"]),
                    color=ARM_COLOURS["finetuned"], ls=SPLIT_STYLES[split],
                    label=label)
        best = np.sqrt(history.set_index("epoch").loc[trajectory.best_epoch, "test_loss"])
        ax.plot(trajectory.best_epoch, best, "*", color="black", ms=10,
                label="selected epoch")
        ax.set(xlabel="Epoch", ylabel=RMSE_AXIS)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        cursor = ax.axvline(-1, color="black", lw=1)
        last = history["epoch"].max()

        def move(epoch: float) -> None:
            epoch = min(epoch, last)
            cursor.set_xdata([epoch, epoch])
            ax.set_title(self._status(trajectory, epoch))

        return move

    def _status(self, trajectory: Trajectory, epoch: float) -> str:
        """
        Where a run is at a frame, for its loss panel's title: before training,
        at an epoch, or stopped by early stopping and holding its last epoch.

        Parameters
        ----------
        trajectory : Trajectory
            Run the title is for.
        epoch : float
            The frame's epoch, clamped to the run's last.

        Returns
        -------
        str
            Title text.
        """
        last = trajectory.history["epoch"].max()
        if epoch < 0:
            status = "pretrained, before fine-tuning"
        elif epoch == last:
            status = f"stopped after epoch {last}"
        else:
            status = f"epoch {int(epoch)}"
        if int(epoch) == trajectory.best_epoch:
            status += " (selected)"
        return f"{trajectory.distribution}-distribution: {status}"

    def _key(self, fig: Figure, axes: np.ndarray, colour: str) -> None:
        """
        Say what colour and size mean. Affinity gets a colourbar; clusters get
        none, since their colours are an identity rather than a key -- what
        the figure shows is whether one cluster stays in one place.

        Parameters
        ----------
        fig : Figure
            Figure to label.
        axes : np.ndarray
            The latent-space axes, which the colourbar is set beside.
        colour : str
            A key of COLOURINGS.
        """
        if colour == "label":
            fig.colorbar(ScalarMappable(self._label_norm(), LABEL_CMAP), ax=axes,
                         label=LABEL_AXIS, extend="both", shrink=0.8)
        fig.suptitle(f"ChemBERTa latent space over fine-tuning (seed {GIF_SEED}), "
                     f"coloured by {COLOURINGS[colour]}; large points are the "
                     f"validation set")

    def q2_gif(self, colour: str = "label") -> Path:
        """
        README's Question 2 animation: rows are distributions, columns the
        embedding, middle and last layers, plus the loss curve. It starts on
        the pretrained weights and plays through fine-tuning, so the change
        each layer undergoes -- and which layers barely move -- is the motion.
        A run that stops early holds its last epoch while the other plays on.

        Parameters
        ----------
        colour : str
            A key of COLOURINGS.

        Returns
        -------
        Path
            The saved GIF.
        """
        fig, axes = plt.subplots(len(DISTRIBUTIONS), len(GIF_LAYERS) + 1,
                                 figsize=GIF_FIGSIZE, width_ratios=GIF_WIDTH_RATIOS,
                                 layout="constrained")
        # Only the losses share a scale; each latent panel is its own UMAP.
        axes[1, -1].sharey(axes[0, -1])
        moves = []
        for row, trajectory in zip(axes, self.trajectories):
            colours = self._colours(trajectory, colour)
            for i, ax in enumerate(row[:-1]):
                moves.append(self._latent_panel(ax, trajectory, i, colours))
            moves.append(self._history_panel(row[-1], trajectory))
            row[0].set_ylabel(f"{trajectory.distribution}-distribution")
        for ax, layer in zip(axes[0], GIF_LAYERS):
            ax.set_title(LAYER_LABELS[layer])
        axes[0, -1].legend(fontsize="small")
        self._key(fig, axes[:, :-1], colour)

        last = max(t.history["epoch"].max() for t in self.trajectories)
        epochs = np.r_[np.full(HOLD_FRAMES, -1.0),
                       np.linspace(-1, last, (last + 1) * TWEEN_FRAMES + 1),
                       np.full(HOLD_FRAMES, float(last))]
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q2_{colour}.gif"
        FuncAnimation(fig, lambda epoch: [move(epoch) for move in moves],
                      frames=epochs).save(path, writer=PillowWriter(fps=GIF_FPS),
                                          dpi=GIF_DPI)
        plt.close(fig)
        return path

    def q2_filmstrip(self, colour: str = "label") -> Path:
        """
        The animation's key frames side by side, for print and results.md:
        one row per distribution and layer, and columns for the pretrained
        weights, fixed shares of the run, and the selected and last epochs.
        The columns are shares rather than epoch numbers because the runs stop
        at different epochs.

        Parameters
        ----------
        colour : str
            A key of COLOURINGS.

        Returns
        -------
        Path
            The saved figure.
        """
        fig, axes = plt.subplots(len(self.trajectories) * len(GIF_LAYERS),
                                 len(FILMSTRIP_FRACTIONS) + 2,
                                 figsize=FILMSTRIP_FIGSIZE, layout="constrained")
        rows = iter(axes)
        for trajectory in self.trajectories:
            colours = self._colours(trajectory, colour)
            last = trajectory.history["epoch"].max()
            epochs = [round(f * (last + 1)) - 1 for f in FILMSTRIP_FRACTIONS]
            titles = [f"epoch {epoch}" if epoch >= 0 else "pretrained"
                      for epoch in epochs]
            epochs += [trajectory.best_epoch, last]
            titles += [f"epoch {trajectory.best_epoch} (selected)",
                       f"epoch {last} (last)"]
            for i, layer in enumerate(GIF_LAYERS):
                row = next(rows)
                for ax, epoch, title in zip(row, epochs, titles):
                    self._latent_panel(ax, trajectory, i, colours)(epoch)
                    ax.set_title(title, fontsize="small")
                row[0].set_ylabel(f"{trajectory.distribution}-distribution\n"
                                  f"{LAYER_LABELS[layer]}", fontsize="small")
        self._key(fig, axes, colour)
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q2_{colour}_filmstrip.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    def q2(self) -> list[Path]:
        """
        Every Question 2 figure: the animation and its filmstrip, coloured by
        affinity and by Butina cluster. One trajectory set serves all four.

        Returns
        -------
        list[Path]
            The saved figures.
        """
        return [figure(colour) for colour in COLOURINGS
                for figure in (self.q2_gif, self.q2_filmstrip)]


if __name__ == "__main__":
    # The results file is an argument rather than a literal, so this is not
    # the hardcoded-target __main__ pattern todo.md is undoing elsewhere.
    parser = argparse.ArgumentParser(description=Results.__doc__)
    parser.add_argument("results", type=Path,
                        help="out/experiments/<target>-<mutation>.csv")
    results = Results(parser.parse_args().results)
    print(results.comparisons())
    print(results.q1())
    print(*results.q2(), sep="\n")
