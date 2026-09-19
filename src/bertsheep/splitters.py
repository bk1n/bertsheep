from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

from bertsheep.chemistry import Chemist

SPLIT_SEED = 444
TRAIN_SPLIT, TEST_SPLIT = 0.7, 0.15  # valid takes the remainder
SPLIT_METHODS = ("scaffold", "butina", "fingerprint")
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
        What the molecules are split on, one of SPLIT_METHODS.
    distribution : str
        Which side of that structure the held-out sets fall on: 'in' to put
        them inside the training set's chemical space, 'out' to drive them out
        of it.
    train_size : float
        Fraction of molecules in train.
    test_size : float
        Fraction of molecules in test; valid takes the remainder.
    seed : int
        Seed for the methods that shuffle, so a replicate is a new seed rather
        than a new code path. The fingerprint deal is deterministic and ignores
        it.
    min_cluster_size : int
        Molecules in a cluster smaller than this are left out of every split,
        which trims the one-off chemistry that neither an in-distribution split
        can represent on both sides nor an out-of-distribution one can learn
        from. 1 keeps everything. Only the grouping methods have clusters to
        measure, so anything above 1 with 'fingerprint' is an error.
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
    ) -> None:
        if method not in SPLIT_METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {SPLIT_METHODS}")
        if distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"unknown distribution {distribution!r}; choose from {DISTRIBUTIONS}"
            )
        if not 0 < train_size + test_size < 1:
            raise ValueError("train_size + test_size must leave a validation set")
        if method == "fingerprint" and min_cluster_size > 1:
            raise ValueError("min_cluster_size needs clusters; fingerprint has none")
        self.method = method
        self.distribution = distribution
        self.train_size, self.test_size = train_size, test_size
        self.seed = seed

        smiles = list(smiles)
        self.rows = np.arange(len(smiles))
        chemist = Chemist()
        # Only what this method needs: the distance matrix is the expensive
        # part and is the memory ceiling, ~1.3 GB at 12.8k ligands.
        if method in ("butina", "fingerprint"):
            self.distances = chemist.pairwise_tanimoto(chemist.fingerprints(smiles))
        if method == "scaffold":
            self.clusters = chemist.scaffold_clusters(smiles)[0]
        elif method == "butina":
            self.clusters = chemist.butina_clusters(self.distances)
        if method != "fingerprint":
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
        train, rest = self._cut(self.rows, self.train_size)
        # The second cut is of the remainder, so test's share has to be
        # rescaled out of it -- 15% of everything is half of the 30% left.
        test, valid = self._cut(rest, self.test_size / (1 - self.train_size))
        sizes = " / ".join(
            f"{len(split)} {name}"
            for name, split in (("train", train), ("valid", valid), ("test", test))
        )
        print(f"-- Split ({self.method}, {self.distribution}-distribution): {sizes}")
        return train, test, valid

    def _cut(self, rows: np.ndarray, train_size: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split a subset of the molecules two ways by the configured method. Every
        method takes and returns positional indices into the whole set, so the
        second cut of a three-way split is the same call over the remainder of
        the first and needs no index bookkeeping from the caller.

        Parameters
        ----------
        rows : np.ndarray
            Positional indices of the molecules to split.
        train_size : float
            Fraction of `rows` in the first of the two splits.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Positional indices of the first and second split.
        """
        methods = {
            "scaffold": self._cluster_split,
            "butina": self._cluster_split,
            "fingerprint": self._fingerprint_split,
        }
        return methods[self.method](rows, train_size)

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

    def _fingerprint_split(self, rows: np.ndarray,
                           train_size: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split molecules on ECFP4 Tanimoto distance without grouping them first.
        There are no clusters to keep whole here, so unlike a scaffold or Butina
        split the ratio lands on train_size exactly rather than on whatever the
        group sizes happen to allow.

        distribution='out' is the standard fingerprint split: each molecule is
        dealt to the split it is *least* similar to the other of, so the two
        end up as far apart in chemical space as this data allows, and the test
        set measures generalisation to chemistry the model has not seen.

        distribution='in' reverses the choice -- the molecule most similar to
        the other split is the one that moves -- so near-neighbours are spread
        across both splits and the test set sits inside the training set's
        chemical space. It is the upper bound the 'out' split is read against.

        The deal is greedy and deterministic: it starts from the first molecule
        and breaks ties by position, so a replicate comes from shuffling the
        input, not from the seed.

        Parameters
        ----------
        rows : np.ndarray
            Positional indices of the molecules to split.
        train_size : float
            Fraction of `rows` in the first split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Positional indices of the first and second split.
        """
        first, second = self._deal(
            self.distances[np.ix_(rows, rows)], train_size,
            furthest=self.distribution == "out",
        )
        return rows[first], rows[second]

    def _deal(self, distances: np.ndarray, train_size: float,
              furthest: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Deal molecules one at a time into two splits, each going to whichever
        split is furthest behind its quota. Greedy rather than optimal: the
        assignment that actually maximises the distance between two sets is
        combinatorial, and one pass gets close enough for a split.

        Works off the whole distance matrix rather than recomputing similarity
        per molecule, so each step is one vectorised pass instead of a Python
        loop over the unassigned.

        Parameters
        ----------
        distances : np.ndarray
            (n, n) Tanimoto distance matrix, from Chemist.pairwise_tanimoto().
        train_size : float
            Fraction of molecules in the first split.
        furthest : bool
            Deal the molecule furthest from the other split, rather than the
            nearest to it.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Positional indices into `distances` of the first and second split.
        """
        n = len(distances)
        quota = (n * train_size, n * (1 - train_size))
        # Distance from each molecule to the nearest member of each split. A
        # molecule is maximally distant from an empty split, and 1.0 is the
        # largest a Tanimoto distance gets -- keeping it finite means the
        # sentinel below is the only -inf, whichever direction we are picking.
        nearest = np.ones((2, n))
        unassigned = np.ones(n, dtype=bool)
        splits = np.empty(n, dtype=int)
        counts = [0, 0]
        sign = 1 if furthest else -1
        while unassigned.any():
            split = 0 if counts[0] / quota[0] <= counts[1] / quota[1] else 1
            # Scored on distance to the *other* split: what makes this split a
            # good home is being far from (or close to) everything already in
            # the one it is being kept away from.
            scores = np.where(unassigned, sign * nearest[1 - split], -np.inf)
            molecule = np.argmax(scores)
            splits[molecule] = split
            counts[split] += 1
            unassigned[molecule] = False
            nearest[split] = np.minimum(nearest[split], distances[molecule])
        return np.flatnonzero(splits == 0), np.flatnonzero(splits == 1)
