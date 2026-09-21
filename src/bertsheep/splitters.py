from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

from bertsheep.chemistry import Chemist

SPLIT_SEED = 444
TRAIN_SPLIT, TEST_SPLIT = 0.7, 0.15  # valid takes the remainder
SPLIT_METHODS = ("scaffold", "butina")
DISTRIBUTIONS = ("in", "out")


class Splitters:
    """
    Build train/test/valid splits that place the held-out sets a chosen
    structural distance from the training set -- interleaved with it, or as far
    from it as the chemistry allows.

    The choice of method and distribution is fixed at construction because it
    is the experiment: the same model on an in- and an out-of-distribution
    split of the same target is the comparison being made. Fixing it here also
    means the structural work each method needs -- fingerprints, distances,
    cluster IDs -- is done once, then reused by both cuts of the three-way
    split.

    Splits are returned as positional indices into `smiles` rather than as
    frames, so a caller can take whatever it is carrying alongside the SMILES
    with the same indices.

    Parameters
    ----------
    smiles : Iterable[str]
        SMILES strings, all parseable by RDKit.
    method : str
        What the molecules are grouped by, one of SPLIT_METHODS.
    distribution : str
        Which side of that structure the held-out sets fall on: 'in' to put
        them inside the training set's chemical space, 'out' to drive them out
        of it.
    train_size : float
        Fraction of molecules in train.
    test_size : float
        Fraction of molecules in test; valid takes the remainder.
    seed : int
        Seed for the shuffle, so a replicate is a new seed rather than a new
        code path.
    min_cluster_size : int
        Molecules in a cluster smaller than this are left out of every split,
        which trims the one-off chemistry that neither an in-distribution split
        can represent on both sides nor an out-of-distribution one can learn
        from. 1 keeps everything.
    distances : np.ndarray | None
        Precomputed (n, n) Tanimoto distance matrix over `smiles`, in that
        order. Only 'butina' needs one, and it is the memory ceiling at ~1.3 GB
        for 12.8k ligands, so a caller already holding the matrix can hand it
        over rather than pay for it again per seed. None computes it if the
        method calls for it.
    """

    def __init__(
        self,
        smiles: Iterable[str],
        method: str = "scaffold",
        distribution: str = "in",
        train_size: float = TRAIN_SPLIT,
        test_size: float = TEST_SPLIT,
        seed: int = SPLIT_SEED,
        min_cluster_size: int = 1,
        distances: np.ndarray | None = None,
    ) -> None:
        if method not in SPLIT_METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {SPLIT_METHODS}")
        if distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"unknown distribution {distribution!r}; choose from {DISTRIBUTIONS}"
            )
        if not 0 < train_size + test_size < 1:
            raise ValueError("train_size + test_size must leave a validation set")
        self.method = method
        self.distribution = distribution
        self.train_size, self.test_size = train_size, test_size
        self.seed = seed

        smiles = list(smiles)
        # A matrix of the wrong size would otherwise cluster some other set of
        # molecules and hand back indices that quietly mean nothing here.
        if distances is not None and distances.shape != (len(smiles), len(smiles)):
            raise ValueError(
                f"distances is {distances.shape}, not "
                f"({len(smiles)}, {len(smiles)}); it must cover these molecules"
            )
        chemist = Chemist()
        if method == "scaffold":
            self.clusters = chemist.scaffold_clusters(smiles)[0]
        else:
            if distances is None:
                distances = chemist.pairwise_tanimoto(chemist.fingerprints(smiles))
            self.clusters = chemist.butina_clusters(distances)
        # Cluster IDs are contiguous from 0, so bincount is the size table.
        sizes = np.bincount(self.clusters)
        self.rows = np.flatnonzero(sizes[self.clusters] >= min_cluster_size)
        self.n = len(self.rows)

    def split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Turn the configured two-way split into a three-way one by running it
        twice: once to cut train off the whole set, then again on what is left
        to cut test from valid. Splitting the remainder rather than dealing
        three ways at once means the valid/test boundary is drawn by the same
        rule as the train/test one, so the held-out valid split is no easier
        than the test set selection runs on.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Positional train, test and valid indices into the SMILES. Molecules
            dropped by `min_cluster_size` appear in none of them.
        """
        train, rest = self._cluster_split(self.rows, self.train_size)
        # The second cut is of the remainder, so test's share has to be
        # rescaled out of it -- 15% of everything is half of the 30% left.
        test, valid = self._cluster_split(rest, self.test_size / (1 - self.train_size))
        sizes = " / ".join(
            f"{len(split)} {name}"
            for name, split in (("train", train), ("valid", valid), ("test", test))
        )
        print(f"-- Split ({self.method}, {self.distribution}-distribution): {sizes}")
        return train, test, valid

    def _cluster_split(self, rows: np.ndarray,
                       train_size: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split molecules that have been grouped by structure, i.e. the scaffold
        or Butina cluster IDs from Chemist. Both grouping methods return the
        same contract, so either can be handed straight here.

        distribution='in' keeps every cluster represented on both sides, so a
        test molecule is chemistry whose neighbours the model has seen.
        StratifiedKFold rather than StratifiedShuffleSplit because most
        scaffolds are singletons: the shuffle splitters refuse a class they
        cannot place on both sides, while the fold splitters warn and deal it
        to one. The number of folds is what sets the ratio, so train_size is
        honoured to the nearest 1/n_splits -- 0.7 gives three folds, 67/33.

        distribution='out' holds whole clusters out with GroupShuffleSplit, so
        no test scaffold appears in train at all. Its train_size is a fraction
        of clusters, not of molecules, so a target carried by a few dominant
        scaffolds will not land near it.

        sklearn splits on row count rather than on the data itself, so X is a
        placeholder of the right length; the clusters are the y stratified on
        and the groups held out, and no labels are needed to build a split.

        Parameters
        ----------
        rows : np.ndarray
            Positional indices of the molecules to split.
        train_size : float
            Fraction of molecules ('in') or of clusters ('out') in the first
            split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Positional indices of the first and second split.
        """
        clusters = self.clusters[rows]
        X = np.zeros(len(rows))
        if self.distribution == "out":
            folds = GroupShuffleSplit(
                n_splits=1, train_size=train_size, random_state=self.seed
            ).split(X, groups=clusters)
        else:
            folds = StratifiedKFold(
                n_splits=round(1 / (1 - train_size)), shuffle=True,
                random_state=self.seed,
            ).split(X, y=clusters)
        first, second = next(folds)
        return rows[first], rows[second]
