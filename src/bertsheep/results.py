import argparse
import json
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess

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
# Grey for the baseline, which is a reference line rather than a curve; blue and
# orange are the colourblind-safe pair eda.py settles on.
ARM_COLOURS = {"baseline": "grey", "pretrained": "tab:blue",
               "finetuned": "tab:orange"}
# The baseline is XGBoost on Morgan fingerprints; the other two are ChemBERTa.
ARM_LABELS = {"baseline": "XGBoost + FPS", "pretrained": "Pretrained",
              "finetuned": "Fine-tuned"}
SPLIT_LABELS = {"train": "train", "test": "test", "valid": "validation"}
# Dotted for valid, so the baseline -- also a valid score -- takes dash-dot.
SPLIT_STYLES = {"train": "--", "test": "-", "valid": ":"}
# The model loss is MSE on the -ln IC50 (nM) label, see eda.LABEL_AXIS.
LOSS_AXIS = "MSE, -ln IC50 (nM)"
# Early stopping ends seeds at different epochs, so an epoch is only plotted
# while at least this share of an arm's seeds are still training; past it the
# curve would be the few longest runs, not the arm.
SURVIVOR_SHARE = 0.3
Q1_FIGSIZE = (12, 8)
# Boxes need less room than curves over tens of epochs.
Q1_WIDTH_RATIOS = (1, 2)

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
        and split. The baseline counts boosting rounds, not epochs, so it is a
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
        baseline = self.df[(self.df["arm"] == "baseline")
                           & (self.df["distribution"] == distribution)]
        ax.axhline((baseline["valid_rmse"] ** 2).mean(),
                   color=ARM_COLOURS["baseline"], ls="-.", label=f"{ARM_LABELS['baseline']} {SPLIT_LABELS['valid']}")
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
        The run's train, test and valid loss, styled as in q1(), with the
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
            ax.plot(history["epoch"], history[f"{split}_loss"],
                    color=ARM_COLOURS["finetuned"], ls=SPLIT_STYLES[split],
                    label=label)
        best = history.set_index("epoch").loc[trajectory.best_epoch, "test_loss"]
        ax.plot(trajectory.best_epoch, best, "*", color="black", ms=10,
                label="selected epoch")
        ax.set(xlabel="Epoch", ylabel=LOSS_AXIS)
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
    print(results.q1())
    print(*results.q2(), sep="\n")
