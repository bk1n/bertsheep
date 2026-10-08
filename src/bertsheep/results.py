import json
from collections.abc import Callable
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import imageio_ffmpeg
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba_array
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

from bertsheep.chemistry import Chemist
from bertsheep.data import CACHE_DIR, Data
from bertsheep.eda import (
    LABEL_AXIS,
    LABEL_CLIP,
    LABEL_CMAP,
    SPLIT_COLOURS,
    UMAP_MIN_DIST,
    UMAP_NEIGHBOURS,
    UMAP_SEED,
)
from bertsheep.experiment import VIDEO_SEED
from bertsheep.model import (
    CONFIG,
    EPOCH_CHECKPOINT,
    HISTORY,
    INIT_CHECKPOINT,
    SPLIT_NAMES,
    SPLITS,
    embeddings,
    split_frames,
)
from bertsheep.splitters import DISTRIBUTIONS

# docs/figures/ rather than out/: these are embedded in the report GitHub Pages
# serves from docs/, so they have to be tracked, and out/ is gitignored.
FIGURE_DIR = Path("docs/figures")
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
# Arms scored once on test rather than per epoch, drawn as horizontal lines.
REFERENCE_ARMS = ("cluster_mean", "baseline")
SPLIT_LABELS = {"train": "train", "valid": "validation", "test": "test"}
# Dotted for test, so the references -- also test scores -- take dash-dot.
SPLIT_STYLES = {"train": "--", "valid": "-", "test": ":"}
# Every score is RMSE, in the label's own units (see eda.LABEL_AXIS), so the
# boxes and curves read on one scale. The model trains on MSE; history.csv's
# losses are split-wide MSE, so their square root is that split's RMSE.
RMSE_AXIS = f"RMSE, {LABEL_AXIS}"
# Early stopping ends seeds at different epochs, so an epoch is only plotted
# while at least this share of an arm's seeds are still training; past it the
# curve would be the few longest runs, not the arm.
SURVIVOR_SHARE = 0.3
Q1_FIGSIZE = (16, 7.5)
SIMILARITY_AXIS = "Median Tanimoto similarity (test vs train)"
SEED_POINT_SIZE = 20
# Arms tested against each other, in box order. "mean" is left out: every model
# beats it by a mile, and out-of-distribution it scores exactly as the cluster
# mean does, so it adds tests to the Holm family without adding information.
COMPARED_ARMS = ("cluster_mean", "baseline", "pretrained", "finetuned")
CONFIDENCE = 0.95
# Holm keeps Bonferroni's family-wise error rate and is never less powerful.
P_ADJUST = "holm"
SIGNIFICANCE_BINS = [0, 0.001, 0.01, 0.05, 1]
SIGNIFICANCE_LABELS = ["***", "**", "*", "ns"]
# Gap between stacked brackets, as a share of the test RMSE's range over both
# distributions: the box rows share a y axis, so one step suits both.
BRACKET_STEP = 0.2

# Aligned UMAP coordinates per video run. Cached because embedding every
# checkpoint and aligning the epochs takes minutes, and restyling a figure
# should not repeat it. Keyed on the run alone: delete the file after changing
# VIDEO_LAYERS, SUBSAMPLE, LATENT_METRIC or the UMAP constants.
LATENT_DIR = Path("out/latent")
# Hidden-state slices drawn: the embedding layer, the middle and the last of
# ChemBERTa-10M-MTR's three encoder layers -- the report's first, middle, last.
VIDEO_LAYERS = (1, 2, 3)
LAYER_LABELS = {0: "Embedding layer", 1: "Encoder layer 1",
                2: "Encoder layer 2", 3: "Encoder layer 3"}
# Share of each split's molecules embedded and aligned. Aligned UMAP's cost
# grows with molecules x epochs, ~1.5 h a panel at the full ~8.8k; taken per
# split so the train/valid/test proportions are the run's own.
SUBSAMPLE = 0.2
SUBSAMPLE_SEED = 0
# Transformer embeddings differ more in direction than in length.
LATENT_METRIC = "cosine"
# Frames interpolated per epoch. Aligned UMAP keeps each molecule's identity
# across epochs, so the in-betweens are real paths the eye can follow rather
# than ~9k points jumping at once.
TWEEN_FRAMES = 4
HOLD_FRAMES = 12  # repeats of the first and last frame, so a looping video pauses on both
VIDEO_FPS = 10
VIDEO_DPI = 200  # served by GitHub Pages, so kept to a few MB
# The pip-installed binary, so the videos need no system ffmpeg.
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_FIGSIZE = (17, 8)
VIDEO_WIDTH_RATIOS = (1, 1, 1, 1.2)
# In the affinity row, test molecules were never trained on, so they are drawn
# larger and opaque over the faint train and valid molecules: generalisation
# reads as the held-out points landing among training points of their colour.
TEST_POINT_SIZE = 10
TEST_ALPHA = 0.8
# The split row draws every molecule at one alpha, in a seeded shuffle:
# in-distribution the test set is spread among train, so test drawn on top -- or
# any split drawn last -- would hide the others, and the splits could not be
# seen moving together. Marker area falls with a split's size, so each split
# puts about the same ink on the page: the smallest gets SPLIT_MAX_SIZE, and
# train, ~5-8x the others, would otherwise swamp them by numbers alone.
SPLIT_ALPHA = 0.6
SPLIT_MAX_SIZE = 16
SHUFFLE_SEED = 0
# A video's rows, top to bottom: the same molecules at the same coordinates,
# so a point's affinity above can be read against its split below.
COLOURINGS = {"label": "affinity", "split": "split"}


class Trajectory(NamedTuple):
    """
    One video run's latent space over fine-tuning.

    Parameters
    ----------
    distribution : str
        One of DISTRIBUTIONS.
    molecules : pd.DataFrame
        The embedded molecules, with `split` and `labels` columns, in the
        order of `coords`' molecule axis.
    coords : np.ndarray
        Aligned UMAP coordinates, shape (len(VIDEO_LAYERS), epochs, molecules, 2).
    history : pd.DataFrame
        The run's history.csv, one row per epoch from -1.
    best_epoch : int
        The epoch the run selected on valid loss.
    """

    distribution: str
    molecules: pd.DataFrame
    coords: np.ndarray
    history: pd.DataFrame
    best_epoch: int


class Results:
    """
    Figures over one experiment's results file, as written by
    Experiment.grid(), for the report in docs/index.html.

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
        Every transformer run's per-epoch train, valid and test RMSE, one row
        per (run, epoch, split). Driven off the results file's run_dir rather than a
        glob of out/models/, which also holds tuning-trial and legacy runs; the
        baseline has no run_dir and no epochs, so it drops out here.

        Returns
        -------
        pd.DataFrame
            Columns arm, distribution, seed, epoch, split and rmse. Epoch -1 has
            valid and test scores only, since nothing was trained before it.
        """
        runs = self.df.dropna(subset="run_dir")
        histories = pd.concat(
            pd.read_csv(Path(run.run_dir) / "history.csv").assign(
                arm=run.arm, distribution=run.distribution, seed=run.seed)
            for run in runs.itertuples()
        )
        return histories.melt(
            id_vars=["arm", "distribution", "seed", "epoch"],
            value_vars=["train_loss", "valid_loss", "test_loss"], var_name="split",
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
        Train, valid and test RMSE per epoch for the two transformer arms on one
        distribution: every seed's value as a point, with a LOESS trend per arm
        and split. The reference arms have no epochs -- the baseline counts
        boosting rounds, the cluster mean fits nothing -- so each is a
        horizontal line at its mean test RMSE rather than a curve.

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
            ax.axhline(scores["test_rmse"].mean(), color=ARM_COLOURS[arm],
                       ls="-.", label=f"{ARM_LABELS[arm]} {SPLIT_LABELS['test']}")
        ax.set(title=f"{distribution}-distribution", xlabel="Epoch",
               ylabel=RMSE_AXIS)
        # Epochs are whole; left alone, the out panel's short range gets halves.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _pairwise(self, distribution: str) -> pd.DataFrame:
        """
        Every pair of COMPARED_ARMS tested on one distribution's test RMSE with
        Nadeau and Bengio's (2003) corrected resampled t-test. Every arm sees
        the same split for a given seed, so each pair is compared on its
        per-seed differences, which takes out how hard that seed's split is.

        The seeds are J resplits of one dataset, not J datasets: their training
        sets overlap, so their differences are positively correlated, and the
        plain paired t-test's variance, sigma^2 / J, understates how much the
        mean difference would move on new data. Nadeau and Bengio replace it
        with (1 / J + n2 / n1) sigma^2, n1 and n2 being the training and test
        sizes (n1 + n2 = n), and compare against t with J - 1 degrees of
        freedom. Validation feeds every arm's model selection, so it counts
        towards n1. The out-of-distribution splits vary n2 by seed, so n2 / n1
        is averaged over the seeds.

        Parameters
        ----------
        distribution : str
            One of DISTRIBUTIONS.

        Returns
        -------
        pd.DataFrame
            One row per pair, arm_a before arm_b in COMPARED_ARMS order.
            difference is arm_a's RMSE minus arm_b's, so a negative value means
            arm_a scores better. ci_low and ci_high bound it at CONFIDENCE, and
            p_adjusted is p corrected for this distribution's pairs.
        """
        scores = self.df[(self.df["distribution"] == distribution)
                         & self.df["arm"].isin(COMPARED_ARMS)]
        # Only seeds every arm has finished can be paired.
        wide = scores.pivot(index="seed", columns="arm", values="test_rmse").dropna()
        sizes = scores.groupby("seed")[["n_train", "n_valid", "n_test"]].first().loc[wide.index]
        ratio = (sizes["n_test"] / (sizes["n_train"] + sizes["n_valid"])).mean()
        pairs = pd.DataFrame(combinations(COMPARED_ARMS, 2), columns=["arm_a", "arm_b"])
        differences = wide[pairs["arm_a"]].to_numpy() - wide[pairs["arm_b"]].to_numpy()
        seeds = len(differences)
        difference = differences.mean(axis=0)
        std_err = np.sqrt((1 / seeds + ratio) * differences.var(axis=0, ddof=1))
        ci_low, ci_high = stats.t.interval(CONFIDENCE, seeds - 1, loc=difference,
                                           scale=std_err)
        p = 2 * stats.t.sf(np.abs(difference / std_err), seeds - 1)
        return pairs.assign(
            distribution=distribution,
            difference=difference,
            std_err=std_err,
            ci_low=ci_low,
            ci_high=ci_high,
            p=p,
            p_adjusted=multipletests(p, method=P_ADJUST)[1],
        )

    def comparisons(self) -> pd.DataFrame:
        """
        The pairwise arm tests for every distribution, tested and corrected
        for multiple comparisons separately: seeds are not shared between them,
        since each has its own splitter.

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
        compared = self.df.loc[self.df["arm"].isin(COMPARED_ARMS), "test_rmse"]
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
        Test RMSE per arm over the seeds on one distribution, each box in its
        arm's colour so it reads against the loss curves. Test is the one
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
            From comparisons(), passed in so the tests run once per figure
            rather than once per panel.
        """
        scores = self.df[(self.df["distribution"] == distribution)
                         & self.df["arm"].isin(COMPARED_ARMS)]
        groups = [scores[scores["arm"] == arm] for arm in COMPARED_ARMS]
        boxes = ax.boxplot(
            [group["test_rmse"] for group in groups],
            tick_labels=[ARM_LABELS[arm] for arm in COMPARED_ARMS], patch_artist=True,
            medianprops={"color": "black"},
        )
        for box, arm in zip(boxes["boxes"], COMPARED_ARMS):
            box.set_facecolor(ARM_COLOURS[arm])
        self._brackets(ax, comparisons.loc[distribution], scores["test_rmse"].max())
        ax.set(title=f"{distribution}-distribution", ylabel=f"Test {RMSE_AXIS}")

    def q1(self) -> Path:
        """
        The report's Question 1 figure: one row per distribution, with test
        RMSE boxes, test RMSE against split similarity, and RMSE curves. The
        box and curve columns share their y axes, so the in/out gap -- how much
        of an arm's score depends on having seen the scaffold series -- reads
        straight down them. The similarity column does not: in- and
        out-of-distribution similarities barely overlap, so a common scale
        would squash each cloud into a corner.

        Returns
        -------
        Path
            The saved figure.
        """
        histories = self._surviving()
        comparisons = self.comparisons()
        scores = self.df[self.df["arm"].isin(COMPARED_ARMS)].join(
            self.similarity.rename("similarity"), on=["distribution", "seed"])
        fig, axes = plt.subplots(len(DISTRIBUTIONS), 3, figsize=Q1_FIGSIZE,
                                 layout="constrained")
        axes[1, 0].sharey(axes[0, 0])
        axes[1, 2].sharey(axes[0, 2])
        for (box_ax, similarity_ax, curve_ax), distribution in zip(axes, DISTRIBUTIONS):
            self._rmse_panel(box_ax, distribution, comparisons)
            self._similarity_panel(similarity_ax, scores, distribution)
            self._loss_panel(curve_ax, histories, distribution)
        # On the out panel, whose bottom left is empty; the in panel has no gap.
        axes[1, 1].legend(fontsize="small", loc="lower left")
        axes[0, 2].legend(fontsize="small")
        # Placed as in Eda._umap_splits: x at the y-label's left edge, y on the title
        # line. Only the top row, as the letters name columns, not panels.
        fig.align_ylabels(axes)
        for ax, letter in zip(axes[0], "abc"):
            ax.annotate(letter, (0, 1),
                        xycoords=(ax.yaxis.label, "axes fraction"),
                        xytext=(0, plt.rcParams["axes.titlepad"]),
                        textcoords="offset points", va="baseline",
                        fontsize="large", fontweight="bold")
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q1.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    def _similarity_panel(self, ax: plt.Axes, scores: pd.DataFrame,
                          distribution: str) -> None:
        """
        Each seed's test RMSE per arm against its split's similarity, with a
        least-squares line per arm. A seed's split is shared by every arm, so
        lines that fall together show the split, not the model, setting the
        score.

        Parameters
        ----------
        ax : plt.Axes
            Axes to draw on.
        scores : pd.DataFrame
            COMPARED_ARMS' rows of the results with a `similarity` column.
        distribution : str
            One of DISTRIBUTIONS.
        """
        panel = scores[scores["distribution"] == distribution]
        for arm in COMPARED_ARMS:
            group = panel[panel["arm"] == arm]
            ax.scatter(group["similarity"], group["test_rmse"], s=SEED_POINT_SIZE,
                       color=ARM_COLOURS[arm], edgecolors="black", linewidths=0.5,
                       label=ARM_LABELS[arm], zorder=3)
            fit = np.polyfit(group["similarity"], group["test_rmse"], 1)
            x = np.sort(group["similarity"])
            ax.plot(x, np.polyval(fit, x), color=ARM_COLOURS[arm])
        ax.set(title=f"{distribution}-distribution", xlabel=SIMILARITY_AXIS,
               ylabel=f"Test {RMSE_AXIS}")

    @cached_property
    def config(self) -> dict:
        """
        The first transformer run's config.json. Every run in a results file
        was split over the same frame, so any one of them records the data
        path, target and mutation that rebuild it.

        Returns
        -------
        dict
            The run's config.
        """
        run_dir = Path(self.df["run_dir"].dropna().iloc[0])
        return json.loads((run_dir / CONFIG).read_text())

    @cached_property
    def frame(self) -> pd.DataFrame:
        """
        The preprocessed frame every run was split over, in the order the
        splits' row positions index.

        Returns
        -------
        pd.DataFrame
            Data._preprocess()'s frame.
        """
        return Data(self.config["data_path"], self.config["target"],
                    self.config["mutation"])._preprocess()

    @cached_property
    def distances(self) -> np.ndarray:
        """
        Tanimoto distances over the frame: the same cached matrix Experiment
        splits with, so reading it here costs a load rather than a rebuild.

        Returns
        -------
        np.ndarray
            (n, n) distance matrix in the frame's order.
        """
        chemist = Chemist()
        return chemist.cached_tanimoto(
            chemist.fingerprints(self.frame["smiles"]), CACHE_DIR,
            f"{self.config['target']}-{self.config['mutation'] or 'pooled'}",
        )

    @cached_property
    def similarity(self) -> pd.Series:
        """
        How close each seed's test set sits to its train set: every test
        molecule's Tanimoto similarity to its nearest train molecule, medianed
        over test. Nearest neighbour rather than all pairs because what helps
        a model is having seen one close analogue; the median over all pairs
        sits at ~0.15 on ECFP4 whatever the split. Read from the pretrained runs'
        saved splits, since every seed has one and each arm on a seed shares
        its split.

        Returns
        -------
        pd.Series
            Median nearest-neighbour similarity, indexed by (distribution, seed).
        """
        runs = self.df[self.df["arm"] == "pretrained"].set_index(
            ["distribution", "seed"])["run_dir"]

        def median_nearest(run_dir: str) -> float:
            splits = pd.read_parquet(Path(run_dir) / SPLITS).groupby("split")["row"]
            train, test = splits.get_group("train"), splits.get_group("test")
            return np.median(1 - self.distances[np.ix_(test, train)].min(axis=1))

        return runs.map(median_nearest)

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
            Shape (len(VIDEO_LAYERS), len(epochs), len(smiles), 2).
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
        vectors = np.stack([embeddings(checkpoint, smiles, model_link)[list(VIDEO_LAYERS)]
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
        The latent-space trajectories of VIDEO_SEED's fine-tuned runs, the only
        runs that keep a checkpoint per epoch, in DISTRIBUTIONS order, over a
        SUBSAMPLE of each split. The sample is seeded, so it is the same
        molecules the cached coordinates were fitted on. Cached because every
        Question 2 figure draws from them.

        Returns
        -------
        list[Trajectory]
            One per distribution.
        """
        runs = self.df[(self.df["arm"] == "finetuned") & (self.df["seed"] == VIDEO_SEED)]
        runs = runs.set_index("distribution").loc[list(DISTRIBUTIONS)]
        run_dirs = [Path(run_dir) for run_dir in runs["run_dir"]]
        trajectories = []
        for (distribution, run), run_dir in zip(runs.iterrows(), run_dirs):
            molecules = pd.concat(
                split.sample(frac=SUBSAMPLE, random_state=SUBSAMPLE_SEED).assign(split=name)
                for name, split in zip(SPLIT_NAMES, split_frames(self.frame, run_dir))
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

    def _style(self, trajectory: Trajectory,
               colour: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Each molecule's draw order, colour and size, worked out once so a frame
        only moves points. The affinity row draws test last, larger and
        opaque; the split row draws every molecule the same size in a shuffled
        order, with marker area inversely proportional to its split's size.

        Parameters
        ----------
        trajectory : Trajectory
            Run whose molecules to style.
        colour : str
            A key of COLOURINGS.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Molecule indices in draw order, RGBA per molecule with its alpha,
            shape (molecules, 4), and marker size per molecule.
        """
        if colour == "split":
            splits = trajectory.molecules["split"]
            counts = splits.map(splits.value_counts())
            rgba = to_rgba_array(splits.map(SPLIT_COLOURS), alpha=SPLIT_ALPHA)
            order = np.random.default_rng(SHUFFLE_SEED).permutation(len(rgba))
            return order, rgba, (SPLIT_MAX_SIZE * counts.min() / counts).to_numpy()
        test = (trajectory.molecules["split"] == "test").to_numpy()
        rgba = plt.get_cmap(LABEL_CMAP)(self._label_norm()(trajectory.molecules["labels"]))
        rgba[:, 3] = np.where(test, TEST_ALPHA, POINT_ALPHA)
        order = np.argsort(test, kind="stable")
        return order, rgba, np.where(test, TEST_POINT_SIZE, POINT_SIZE)

    def _at(self, trajectory: Trajectory, layer: int, epoch: float) -> np.ndarray:
        """
        Coordinates at a fractional epoch, linearly between the two epochs
        either side, and held at the last epoch once the run has stopped.

        Parameters
        ----------
        trajectory : Trajectory
            Run to read.
        layer : int
            Position in VIDEO_LAYERS.
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
                      style: tuple[np.ndarray, np.ndarray, np.ndarray]
                      ) -> Callable[[float], None]:
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
            Position in VIDEO_LAYERS.
        style : tuple[np.ndarray, np.ndarray, np.ndarray]
            From _style().

        Returns
        -------
        Callable[[float], None]
            Moves the points to a (fractional) epoch.
        """
        order, rgba, sizes = style
        points = ax.scatter(*trajectory.coords[layer, 0, order].T, s=sizes[order],
                            c=rgba[order], linewidths=0)
        ax.update_datalim(trajectory.coords[layer].reshape(-1, 2))
        ax.autoscale_view()
        ax.set(xticks=[], yticks=[])

        def move(epoch: float) -> None:
            points.set_offsets(self._at(trajectory, layer, epoch)[order])

        return move

    def _history_panel(self, ax: plt.Axes,
                       trajectory: Trajectory) -> Callable[[float], None]:
        """
        The run's train, valid and test RMSE, styled as in q1(), with the
        selected epoch starred on the valid curve it was selected on. A cursor
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
        best = np.sqrt(history.set_index("epoch").loc[trajectory.best_epoch, "valid_loss"])
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

    def _key(self, fig: Figure, axes: np.ndarray, trajectory: Trajectory) -> None:
        """
        Say what each row's colours and the point sizes mean: a colourbar
        beside the affinity row, and the split legend below the figure, as in
        Eda._umap_splits, so it does not crowd the colourbar's margin.

        Parameters
        ----------
        fig : Figure
            Figure to label.
        axes : np.ndarray
            The latent-space axes, one row per COLOURINGS key.
        trajectory : Trajectory
            Run the figure draws, named in the title.
        """
        fig.colorbar(ScalarMappable(self._label_norm(), LABEL_CMAP), ax=axes[0],
                     label=LABEL_AXIS, extend="both", shrink=0.8)
        fig.legend(handles=[Patch(color=SPLIT_COLOURS[split], label=label)
                            for split, label in SPLIT_LABELS.items()],
                   loc="outside lower center", ncols=len(SPLIT_LABELS))
        fig.suptitle(f"ChemBERTa latent space over fine-tuning, "
                     f"{trajectory.distribution}-distribution (seed {VIDEO_SEED}); "
                     f"large points in the affinity row are the test set")

    def q2_video(self, trajectory: Trajectory) -> Path:
        """
        The report's Question 2 animation for one distribution: columns are
        the embedding, middle and last layers, rows the same layout coloured
        by affinity and by split, with the loss curve down the right. It
        starts on the pretrained weights and plays through fine-tuning, so the
        change each layer undergoes -- and which layers barely move -- is the
        motion. Every video runs to the longest run's last epoch, holding its
        own last epoch if it stopped earlier, so the report's synced players
        stay on the same epoch.

        Parameters
        ----------
        trajectory : Trajectory
            Run to draw.

        Returns
        -------
        Path
            The saved MP4.
        """
        fig = plt.figure(figsize=VIDEO_FIGSIZE, layout="constrained")
        grid = fig.add_gridspec(len(COLOURINGS), len(VIDEO_LAYERS) + 1,
                                width_ratios=VIDEO_WIDTH_RATIOS)
        axes = np.array([[fig.add_subplot(grid[row, column])
                          for column in range(len(VIDEO_LAYERS))]
                         for row in range(len(COLOURINGS))])
        loss_ax = fig.add_subplot(grid[:, -1])
        moves = [self._history_panel(loss_ax, trajectory)]
        for row, colour in zip(axes, COLOURINGS):
            style = self._style(trajectory, colour)
            for i, ax in enumerate(row):
                moves.append(self._latent_panel(ax, trajectory, i, style))
            row[0].set_ylabel(f"Coloured by {COLOURINGS[colour]}")
        for ax, layer in zip(axes[0], VIDEO_LAYERS):
            ax.set_title(LAYER_LABELS[layer])
        loss_ax.legend(fontsize="small")
        self._key(fig, axes, trajectory)

        last = max(t.history["epoch"].max() for t in self.trajectories)
        epochs = np.r_[np.full(HOLD_FRAMES, -1.0),
                       np.linspace(-1, last, (last + 1) * TWEEN_FRAMES + 1),
                       np.full(HOLD_FRAMES, float(last))]
        FIGURE_DIR.mkdir(exist_ok=True)
        path = FIGURE_DIR / f"{self.name}_q2_{trajectory.distribution}.mp4"
        FuncAnimation(fig, lambda epoch: [move(epoch) for move in moves],
                      frames=epochs).save(path, writer=FFMpegWriter(fps=VIDEO_FPS),
                                          dpi=VIDEO_DPI)
        plt.close(fig)
        return path

    def q2(self) -> list[Path]:
        """
        Every Question 2 figure: one animation per distribution. The affinity
        scale is shared, so a colour means the same in both.

        Returns
        -------
        list[Path]
            The saved figures, in DISTRIBUTIONS order.
        """
        return [self.q2_video(trajectory) for trajectory in self.trajectories]
