from functools import cache, cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import umap as umap_
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from bertsheep.chemistry import BUTINA_CUTOFF, Chemist
from bertsheep.splitters import SPLIT_SEED, TEST_SPLIT, TRAIN_SPLIT

FIGURE_DIR = Path("out/eda")
UMAP_NEIGHBOURS = 15
UMAP_MIN_DIST = 0.4
UMAP_SEED = 444
# PCA rather than UMAP's spectral default: the fingerprint k-NN graph breaks into
# components whose eigensolve fails, and the fallback is a random layout.
UMAP_INIT = "pca"
TOP_SCAFFOLDS = 30  # tab20 + tab20b supply 40 distinct hues, so 30 is safe
TOP_VARIANTS = 30
TOP_CLUSTERS = 40  # same palette ceiling as TOP_SCAFFOLDS
# Tanimoto distances swept, tight to loose; BUTINA_CUTOFF sits inside the range.
BUTINA_CUTOFFS = (0.2, 0.3, 0.4, 0.5, 0.6)
# Where a framework stops being one-off chemistry and starts being a series a
# split can hold out as a block.
MIN_SCAFFOLD_SIZE = 10
POINT_SIZE = 5
POINT_ALPHA = 0.2
GREY = "lightgrey"
SIZE_CMAP = "viridis"
LABEL_CMAP = "plasma"
# Tail fraction the affinity colour scale is clipped to at each end.
LABEL_CLIP = 0.02
BOX_COLOUR = "tab:blue"
DPI = 300
BOX_FIGSIZE = (12, 6)
# The label column is -ln(IC50 in nM); every figure names the axis the same way.
LABEL_AXIS = "log IC50 (nM)"
# Seeds per row of the split figures: enough to see which structure the split
# repeats and which is luck of the draw.
SPLIT_REPLICATES = 3
SPLIT_COLOURS = {"train": GREY, "valid": "tab:blue", "test": "tab:red"}
SPLIT_FIGSIZE = (15, 9)
OTHER = "other"
WILD_TYPE = "wild type"

class Eda():
    """
    Plots summary views of a preprocessed target dataframe (smiles, labels).
    """
    def __init__(self, df, target, figure_dir=FIGURE_DIR):
        self.df = df
        self.target = target
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def scaffolds(self):
        """
        Bemis-Murcko scaffold per row. Cached because three figures want it and
        the RDKit parse over every ligand is what each of them actually costs.
        """
        return pd.Series(Chemist().scaffold_clusters(self.df["smiles"])[1],
                         index=self.df.index)

    @cached_property
    def variants(self):
        """
        Construct per row, with the empty mutation string spelled out. Kept
        separate from scaffolds so a frame without mutations still plots.
        """
        return self.df["mutations"].replace("", WILD_TYPE)

    def label_histogram(self, column="labels", bins="auto"):
        """
        Histogram of the label distribution.
        """
        _, ax = plt.subplots()
        self.df[column].hist(bins=bins, ax=ax)
        ax.set(
            title=f"{self.target} {LABEL_AXIS} distribution (n={len(self.df)})",
            xlabel=LABEL_AXIS,
            ylabel="ligands",
        )
        ax.figure.savefig(self.figure_dir / f"{self.target}_{column}_histogram.png", dpi=DPI)
        return ax

    @cached_property
    def ligands(self):
        """
        One row per molecule, its label collapsed across constructs. _deduplicate
        keys on (smiles, mutations), so a ligand measured against several
        constructs arrives as several identical rows; embedding them all stacks
        duplicates at distance zero, which is what collapses the eigengap UMAP's
        spectral init needs.
        """
        frame = self.df.drop_duplicates("smiles")
        medians = self.df.groupby("smiles")["labels"].median()
        return frame.assign(labels=frame["smiles"].map(medians))

    @cached_property
    def ligand_scaffolds(self):
        """
        Scaffold per unique ligand, i.e. the scaffolds as the map sees them --
        counts here are molecules, not measurements.
        """
        return self.scaffolds.loc[self.ligands.index]

    @cached_property
    def distances(self):
        """
        Tanimoto distance matrix over the unique ligands. Cached apart from the
        clusters because it doesn't depend on the cutoff and is most of their
        cost, so a sweep pays for it once. ~1.3 GB resident at 12.8k ligands.
        """
        chemist = Chemist()
        return chemist.pairwise_tanimoto(chemist.fingerprints(self.ligands["smiles"]))

    @cache
    def ligand_butina(self, cutoff=BUTINA_CUTOFF):
        """
        Butina cluster per unique ligand. Clustered over molecules, not rows: a
        ligand measured against three constructs would otherwise be three points
        at distance zero, a ready-made cluster Butina would happily centre on.
        Named rather than numbered so they behave like scaffold SMILES as
        categories and tick labels.
        
        cutoff: butina cutoff, controls number of clusters generated
        """
        clusters = Chemist().butina_clusters(self.distances, cutoff=cutoff)
        return pd.Series(clusters, index=self.ligands.index).map("cluster {}".format)

    @cache
    def butina(self, cutoff=BUTINA_CUTOFF):
        """
        Butina cluster per row, for the figures that count measurements the way
        the scaffold ones do.
        """
        lookup = pd.Series(self.ligand_butina(cutoff).values, index=self.ligands["smiles"])
        return self.df["smiles"].map(lookup)

    @cached_property
    def embedding(self):
        """
        The chemical space every umap_* figure recolours. Cached because the fit
        is the whole cost and the colouring is free.
        """
        return self._embed(self.ligands)

    def _embed(self, frame):
        """
        ECFP4 bits reduced to 2D. Jaccard distance on a bit vector *is* Tanimoto,
        so UMAP's compiled metric replaces the hand-rolled callable the legacy
        code passed (which made every distance a Python call).
        """
        embedding = umap_.UMAP(
            n_neighbors=UMAP_NEIGHBOURS,
            min_dist=UMAP_MIN_DIST,
            metric="jaccard",
            init=UMAP_INIT,
            random_state=UMAP_SEED,
        ).fit_transform(Chemist().fingerprints(frame["smiles"]))
        return pd.DataFrame(embedding, columns=["umap1", "umap2"], index=frame.index)

    def _save_umap(self, ax, name, n_ligands, subtitle):
        """
        Every chemical space figure carries the same axes and the same ligand
        count, so the recolourings can be read as one map.
        """
        ax.set(
            title=f"{self.target} chemical space (ECFP4, n={n_ligands} unique ligands)\n"
                  f"{subtitle}",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )
        ax.figure.savefig(self.figure_dir / f"{self.target}_umap_{name}.png",
                          bbox_inches="tight", dpi=DPI)
        return ax

    def _scatter_groups(self, ax, coords, groups, n_groups):
        """
        One colour per largest group, grey for the rest. The colours are an
        identity, not a key: what the figure shows is whether one group lands in
        one place.
        """
        top = groups.value_counts().head(n_groups).index
        # Ordered categorical so groupby yields OTHER first and it is drawn under
        # the highlighted groups rather than over them.
        group = pd.Series(
            pd.Categorical(groups.where(groups.isin(top), OTHER),
                           categories=[OTHER, *top]),
            index=groups.index,
        )
        palette = [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors]
        colours = dict(zip(top, palette))
        for name, points in coords.groupby(group, observed=True):
            ax.scatter(points["umap1"], points["umap2"],
                       s=POINT_SIZE, alpha=POINT_ALPHA, color=colours.get(name, GREY))
        return (~groups.isin(top)).mean()

    def _umap_groups(self, groups, name, noun, n_groups):
        _, ax = plt.subplots()
        grey = self._scatter_groups(ax, self.embedding, groups, n_groups)
        return self._save_umap(
            ax, name, len(self.embedding),
            f"coloured by {noun} (top {n_groups}; grey = other, {grey:.0%} of ligands)",
        )

    def _umap_group_size(self, groups, name, noun):
        sizes = groups.map(groups.value_counts())
        order = sizes.sort_values().index  # the crowded groups drawn last, on top
        _, ax = plt.subplots()
        points = ax.scatter(
            self.embedding.loc[order, "umap1"], self.embedding.loc[order, "umap2"],
            c=sizes.loc[order], s=POINT_SIZE, alpha=POINT_ALPHA,
            cmap=SIZE_CMAP, norm=LogNorm(),
        )
        # Keyed off the norm and cmap, not the points, so POINT_ALPHA -- there for
        # overplotting -- doesn't fade the colour bar with them.
        ax.figure.colorbar(ScalarMappable(points.norm, points.cmap), ax=ax,
                           label=f"ligands sharing the {noun}")
        singletons = (sizes == 1).mean()
        return self._save_umap(
            ax, name, len(self.embedding),
            f"coloured by {noun} size ({singletons:.0%} of ligands are the only "
            f"member of theirs)",
        )

    def umap_scaffolds(self, n_scaffolds=TOP_SCAFFOLDS):
        """
        Chemical space coloured by Bemis-Murcko scaffold. Only the n_scaffolds
        most common frameworks get a colour -- there are thousands, and the tail
        is singletons -- so most of the map is deliberately grey; umap_scaffold_size
        is the view that says what that grey is made of.
        """
        return self._umap_groups(
            self.ligand_scaffolds, "scaffolds", "Bemis-Murcko scaffold", n_scaffolds
        )

    def umap_scaffold_size(self):
        """
        The same map with every point coloured by how many ligands share its
        framework, so nothing is grey. This is the picture a scaffold split is
        made of: if the large series sit in a few dense pockets and the
        singletons spread everywhere, holding out whole scaffolds moves the test
        set into a different region of chemical space, not just a different
        sample of the same one.
        """
        return self._umap_group_size(self.ligand_scaffolds, "scaffold_size", "scaffold")

    def umap_butina(self, cutoff=BUTINA_CUTOFF, n_clusters=TOP_CLUSTERS):
        """
        The same map coloured by Butina cluster. Butina groups whole molecules
        by Tanimoto similarity to a centroid, so where it disagrees with the
        scaffold view is the point: analogues across a ring swap merge into one
        cluster, and a scaffold whose decorations diverge splits into several.
        UMAP is built on the same Tanimoto distance, so clusters should come out
        as compact islands here -- a scattered one is a cutoff problem.
        """
        return self._umap_groups(self.ligand_butina(cutoff), f"butina_{cutoff:.2f}",
                                 f"Butina {cutoff} cluster", n_clusters)

    def umap_butina_size(self, cutoff=BUTINA_CUTOFF):
        """
        The map coloured by Butina cluster size -- the fingerprint-split
        counterpart of umap_scaffold_size, and where the two splits' notions of
        "novel chemistry" can be compared region by region.
        """
        return self._umap_group_size(self.ligand_butina(cutoff), f"butina_size_{cutoff:.2f}",
                                     f"Butina {cutoff} cluster")

    def umap_labels(self, column="labels"):
        """
        The same map coloured by affinity, median across constructs. If potency
        tracks region rather than scattering through it, a random split hands the
        model test answers it can reach by similarity alone; if it scatters, the
        structure-activity relationship is steep and the scaffold split is hard.
        """
        order = self.ligands[column].sort_values().index  # the potent drawn on top
        # Clipped to the central percentiles: the untrimmed range runs to roughly
        # -18, so a handful of inactives otherwise take the whole colour scale and
        # leave the real measurements in one flat band.
        low, high = self.ligands[column].quantile([LABEL_CLIP, 1 - LABEL_CLIP])
        _, ax = plt.subplots()
        points = ax.scatter(
            self.embedding.loc[order, "umap1"], self.embedding.loc[order, "umap2"],
            c=self.ligands.loc[order, column], s=POINT_SIZE, alpha=POINT_ALPHA,
            cmap=LABEL_CMAP, vmin=low, vmax=high,
        )
        ax.figure.colorbar(ScalarMappable(points.norm, points.cmap), ax=ax,
                           label=LABEL_AXIS, extend="both")
        return self._save_umap(
            ax, column, len(self.embedding), f"coloured by {LABEL_AXIS}",
        )

    def _umap_splits(self, groups, name, noun):
        """
        The map coloured by where each ligand lands, in-distribution on top and
        out-of-distribution below, one seed per column. What the two rows should
        show: in-distribution test points scattered through every island, with
        train alongside them; out-of-distribution test points arriving as whole
        islands that train never touches.
        """
        chemist = Chemist()
        fig, axes = plt.subplots(2, SPLIT_REPLICATES, figsize=SPLIT_FIGSIZE,
                                 sharex=True, sharey=True)
        for row, stratify in zip(axes, (True, False)):
            for ax, seed in zip(row, range(SPLIT_SEED, SPLIT_SEED + SPLIT_REPLICATES)):
                indices = chemist.split_groups(groups, TRAIN_SPLIT, TEST_SPLIT,
                                               seed=seed, stratify=stratify)
                assigned = pd.Series(pd.NA, index=self.embedding.index, dtype=object)
                for split, index in zip(SPLIT_COLOURS, indices):
                    assigned.iloc[index] = split
                # Drawn in a random order, not split by split: a set drawn last
                # covers the others wherever they overlap, and an in-distribution
                # split, which overlaps everywhere, would look like all test.
                points = self.embedding.sample(frac=1, random_state=seed)
                ax.scatter(points["umap1"], points["umap2"], s=POINT_SIZE,
                           alpha=POINT_ALPHA,
                           c=assigned.loc[points.index].map(SPLIT_COLOURS))
                ax.set_title(f"seed {seed}: " + " / ".join(
                    f"{split} {len(index) / len(groups):.0%}"
                    for split, index in zip(SPLIT_COLOURS, indices)
                ), fontsize="small")
        axes[0, 0].set_ylabel("in-distribution (stratified by group)\nUMAP 2")
        axes[1, 0].set_ylabel("out-of-distribution (whole groups held out)\nUMAP 2")
        for ax in axes[1]:
            ax.set_xlabel("UMAP 1")
        fig.legend(handles=[Patch(color=colour, label=split)
                            for split, colour in SPLIT_COLOURS.items()],
                   loc="upper right")
        fig.suptitle(f"{self.target} {noun} splits over chemical space "
                     f"(ECFP4 UMAP, n={len(groups)} unique ligands)")
        fig.savefig(self.figure_dir / f"{self.target}_umap_splits_{name}.png",
                    bbox_inches="tight", dpi=DPI)
        return axes

    def umap_scaffold_splits(self):
        """
        Scaffold splits on the shared map. Singleton scaffolds are most of the
        groups, so the out-of-distribution test set is mostly one-off chemistry
        and can still sit right beside its training analogues.
        """
        return self._umap_splits(self.ligand_scaffolds, "scaffold", "Bemis-Murcko scaffold")

    def umap_butina_splits(self, cutoff=BUTINA_CUTOFF):
        """
        Butina splits on the shared map -- the stricter counterpart, since a
        cluster takes a ligand's near analogues out with it whatever their
        scaffold.
        """
        return self._umap_splits(self.ligand_butina(cutoff), f"butina_{cutoff:.2f}",
                                 f"Butina {cutoff} cluster")

    def _grouped_boxplot(self, groups, name, n_groups, column, labelsize):
        """
        Label spread within the n_groups largest groups, ordered by median so the
        level and the spread read off the same axis. Ticks carry the group size:
        a box drawn from three points and one from three thousand look identical
        otherwise.
        """
        counts = groups.value_counts()
        frame = self.df.assign(**{name: groups})
        frame = frame[frame[name].isin(counts.head(n_groups).index)]
        # Ordered categorical, because pandas otherwise sorts the boxes by name.
        order = frame.groupby(name, observed=True)[column].median().sort_values().index
        frame[name] = pd.Categorical(frame[name], categories=order, ordered=True)

        _, ax = plt.subplots(figsize=BOX_FIGSIZE)
        frame.boxplot(
            column=column, by=name, ax=ax, rot=90, patch_artist=True,
            boxprops={"facecolor": BOX_COLOUR}, medianprops={"color": "black"},
        )
        ax.figure.suptitle("")  # pandas adds its own "Boxplot grouped by ..."
        ax.set_xticks(ax.get_xticks(), [
            f"{t.get_text()}\n(n={counts[t.get_text()]})" for t in ax.get_xticklabels()
        ])
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.set(xlabel="", ylabel=LABEL_AXIS)
        return ax

    def _coverage(self, groups, name, noun, n_groups):
        counts = groups.value_counts()
        singletons = (counts == 1).mean()
        # 1-based so the first point is "the largest group", not "no groups".
        coverage = (counts.cumsum() / counts.sum()).set_axis(range(1, len(counts) + 1))
        _, ax = plt.subplots()
        coverage.plot(ax=ax)
        # Mark the groups the UMAP colours and the boxplot shows, so the three
        # figures can be read against one another.
        marked = coverage.iloc[min(n_groups, len(coverage)) - 1]
        ax.axvline(n_groups, color="tab:red", linestyle="--", linewidth=1)
        ax.annotate(f"top {n_groups} = {marked:.0%} of ligands",
                    xy=(n_groups, marked), xytext=(6, -12),
                    textcoords="offset points", color="tab:red", fontsize="small")
        ax.set(
            title=f"{self.target} {noun} coverage "
                  f"({len(counts)} {noun}s, {singletons:.0%} singletons)",
            xlabel=f"{noun}s, largest first",
            ylabel="cumulative share of ligands",
        )
        ax.figure.savefig(self.figure_dir / f"{self.target}_{name}_coverage.png",
                          bbox_inches="tight", dpi=DPI)
        return ax

    def scaffold_coverage(self, n_scaffolds=TOP_SCAFFOLDS):
        """
        Cumulative share of ligands covered by the scaffolds, largest first.
        A curve that saturates in a few dozen steps means a scaffold split has
        almost no blocks to build a test set from; a long flat tail of
        singletons means the opposite, and that most test blocks hold one
        molecule.
        """
        return self._coverage(self.scaffolds, "scaffold", "scaffold", n_scaffolds)

    def butina_coverage(self, cutoff=BUTINA_CUTOFF, n_clusters=TOP_CLUSTERS):
        """
        The same curve over Butina clusters. At a fixed cutoff the cluster count
        is a choice, not a property of the data, so this is the figure to read
        BUTINA_CUTOFF against: fewer, larger clusters make a coarser split with
        fewer test blocks.
        """
        return self._coverage(self.butina(cutoff), f"butina_{cutoff:.2f}",
                              f"Butina {cutoff} cluster", n_clusters)

    def label_by_scaffold(self, column="labels", n_scaffolds=TOP_SCAFFOLDS):
        """
        Label spread inside each of the largest scaffolds. If a series is flat,
        the model can score well by recognising the framework alone, so a
        scaffold split is the only honest test -- and if the series sit at
        different levels, a random split leaks that level into the test set.
        """
        ax = self._grouped_boxplot(
            self.scaffolds, "scaffold", n_scaffolds, column, "xx-small"
        )
        ax.set_title(f"{self.target} {LABEL_AXIS} by scaffold (top {n_scaffolds})")
        ax.figure.savefig(self.figure_dir / f"{self.target}_{column}_by_scaffold.png",
                          bbox_inches="tight", dpi=DPI)
        return ax

    def label_by_butina(self, column="labels", cutoff=BUTINA_CUTOFF, n_clusters=TOP_CLUSTERS):
        """
        Label spread inside each of the largest Butina clusters. Membership is
        bounded by similarity to the centroid, so the spread in a box is roughly
        the activity-cliff range at that cutoff: tight boxes mean near neighbours
        share an affinity, wide ones mean they don't.
        """
        ax = self._grouped_boxplot(self.butina(cutoff), "butina", n_clusters, column, "x-small")
        ax.set_title(f"{self.target} {LABEL_AXIS} by Butina {cutoff} cluster (top {n_clusters})")
        ax.figure.savefig(self.figure_dir / f"{self.target}_{column}_by_butina_{cutoff:.2f}.png",
                          bbox_inches="tight", dpi=DPI)
        return ax

    def label_by_variant(self, column="labels", n_variants=TOP_VARIANTS):
        """
        Label spread per construct, tick-labelled with the ligand count. Pooling
        variants into one model assumes they are interchangeable; wild type
        dominates the counts and the resistance mutants are measured against a
        different ligand set, so any gap here is a confound, not a signal.
        """
        ax = self._grouped_boxplot(
            self.variants, "variant", n_variants, column, "x-small"
        )
        ax.set_title(f"{self.target} {LABEL_AXIS} by construct (top {n_variants})")
        ax.figure.savefig(self.figure_dir / f"{self.target}_{column}_by_variant.png",
                          bbox_inches="tight", dpi=DPI)
        return ax


if __name__ == '__main__':
    from bertsheep.data import Data

    target = "EGFR"
    data = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", target)
    v = Eda(data._preprocess(), target)

    # visualise IC50
    v.label_histogram()
    v.umap_labels()
    
    # BM scaffolds
    v.umap_scaffolds()
    v.umap_scaffold_size()
    v.scaffold_coverage()
    v.label_by_scaffold()
    
    # Butina split
    v.umap_butina()
    v.umap_butina_size()
    v.butina_coverage()
    v.label_by_butina()

    # train/valid/test splits
    v.umap_scaffold_splits()
    v.umap_butina_splits()
        
    # mutations
    v.label_by_variant()
