import numpy as np
import pytest

from bertsheep.chemistry import Chemist
from bertsheep.splitters import Splitters

# Ten ring systems, each carried by four alkyl homologues: every molecule shares
# its Bemis-Murcko scaffold with exactly three others, so the cluster sizes are
# known by construction and a fraction of the clusters is the same fraction of
# the molecules. That is what lets a split's size and its treatment of scaffolds
# be asserted against the same numbers.
CORES = ("c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "C1CCNCC1", "C1CCNC1",
         "c1ccco1", "c1ccsc1", "c1cc2ccccc2[nH]1", "c1ccc2ccccc2n1", "C1CCOCC1")
ALKYLS = ("C", "CC", "CCC", "CCCC")
SMILES = [alkyl + core for core in CORES for alkyl in ALKYLS]
CLUSTER_SIZE = len(ALKYLS)
ROWS = np.arange(len(SMILES))

TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
# The fingerprint split deals one molecule at a time, so it lands on the size
# asked for to within a molecule. A cluster split can only land to within a
# cluster: 'out' moves whole clusters, and 'in' quantises the ratio into folds.
MOLECULE_TOLERANCE = 1 / len(SMILES)
CLUSTER_TOLERANCE = CLUSTER_SIZE / len(SMILES)
# Random splits to read the fingerprint deal against: 'out' has to put the test
# set further from train than chance does, and 'in' closer.
RANDOM_REPLICATES = 20
RANDOM_SEED = 0


def splitter(method: str, distribution: str = "in", seed: int = 11) -> Splitters:
    """
    A Splitters over the fixture molecules, configured the way a test needs it.

    Parameters
    ----------
    method : str
        Split method under test.
    distribution : str
        'in' or 'out'.
    seed : int
        Seed for the methods that shuffle.

    Returns
    -------
    Splitters
        Splitters ready to split the fixture molecules.
    """
    return Splitters(SMILES, method, distribution, TRAIN_SIZE, TEST_SIZE, seed)


@pytest.fixture
def chemist() -> Chemist:
    """A Chemist on the default ECFP4 settings."""
    return Chemist()


@pytest.fixture
def clusters(chemist: Chemist) -> np.ndarray:
    """Scaffold cluster ID per molecule: ten clusters of four."""
    return chemist.scaffold_clusters(SMILES)[0]


@pytest.fixture
def distances(chemist: Chemist) -> np.ndarray:
    """Tanimoto distances between every pair of molecules."""
    return chemist.pairwise_tanimoto(chemist.fingerprints(SMILES))


@pytest.fixture(params=["in", "out"])
def distribution(request) -> str:
    """Both distributions, for the contracts that hold either way."""
    return request.param


@pytest.fixture(params=["scaffold", "butina", "fingerprint"])
def method(request) -> str:
    """Every split method, for the contracts they all share."""
    return request.param


@pytest.fixture(params=["scaffold", "fingerprint"])
def structural_method(request) -> str:
    """The two methods a scaffold assertion can be made against."""
    return request.param


def nearest_train_distance(distances: np.ndarray, train: np.ndarray,
                           test: np.ndarray) -> float:
    """
    Mean Tanimoto distance from a test molecule to the closest training
    molecule: how far the test set sits from what the model has seen, which is
    the quantity a fingerprint split exists to push up or down.

    Parameters
    ----------
    distances : np.ndarray
        (n, n) Tanimoto distance matrix over all molecules.
    train : np.ndarray
        Positional indices of the training molecules.
    test : np.ndarray
        Positional indices of the test molecules.

    Returns
    -------
    float
        Mean over test molecules of the distance to their nearest train
        neighbour.
    """
    return distances[np.ix_(test, train)].min(axis=1).mean()


def chance_distance(distances: np.ndarray) -> float:
    """
    The same distance for a random split of the same size, averaged over
    replicates: the baseline both fingerprint distributions are read against,
    since a split that is no further from train than chance has not done
    anything.

    Parameters
    ----------
    distances : np.ndarray
        (n, n) Tanimoto distance matrix over all molecules.

    Returns
    -------
    float
        Mean nearest-train distance over RANDOM_REPLICATES random splits.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    cut = round(len(SMILES) * TRAIN_SIZE)
    shuffles = (rng.permutation(len(SMILES)) for _ in range(RANDOM_REPLICATES))
    return np.mean([nearest_train_distance(distances, s[:cut], s[cut:])
                    for s in shuffles])


def test_two_way_split_is_a_partition(method: str, distribution: str) -> None:
    """Any method, either distribution: every molecule lands in exactly one side."""
    train, test = splitter(method, distribution)._cut(ROWS, TRAIN_SIZE)
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(SMILES)))


def test_fingerprint_split_lands_on_the_requested_size(distribution: str) -> None:
    """Dealing by quota hits train_size to within one molecule."""
    train, _ = splitter("fingerprint", distribution)._cut(ROWS, TRAIN_SIZE)
    assert len(train) / len(SMILES) == pytest.approx(TRAIN_SIZE, abs=MOLECULE_TOLERANCE)


def test_cluster_split_lands_on_the_requested_size(distribution: str) -> None:
    """A cluster split hits train_size to within one cluster, clusters being indivisible."""
    train, _ = splitter("scaffold", distribution)._cut(ROWS, TRAIN_SIZE)
    assert len(train) / len(SMILES) == pytest.approx(TRAIN_SIZE, abs=CLUSTER_TOLERANCE)


def test_fingerprint_split_out_is_further_from_train_than_chance(
    distances: np.ndarray
) -> None:
    """'out' deals the least similar molecule away, so its test set beats a random one."""
    train, test = splitter("fingerprint", "out")._cut(ROWS, TRAIN_SIZE)
    assert nearest_train_distance(distances, train, test) > chance_distance(distances)


def test_fingerprint_split_in_is_closer_to_train_than_chance(
    distances: np.ndarray
) -> None:
    """'in' deals the most similar molecule away, so its test set sits inside train's space."""
    train, test = splitter("fingerprint", "in")._cut(ROWS, TRAIN_SIZE)
    assert nearest_train_distance(distances, train, test) < chance_distance(distances)


def test_fingerprint_split_out_is_further_than_in(distances: np.ndarray) -> None:
    """The two distributions bracket chance: 'out' is the hard split, 'in' the easy one."""
    apart = splitter("fingerprint", "out")._cut(ROWS, TRAIN_SIZE)
    together = splitter("fingerprint", "in")._cut(ROWS, TRAIN_SIZE)
    assert (nearest_train_distance(distances, *apart)
            > nearest_train_distance(distances, *together))


def test_out_keeps_scaffolds_whole(structural_method: str,
                                   clusters: np.ndarray) -> None:
    """No test scaffold appears in train at all, which is the point of the 'out' split."""
    train, test = splitter(structural_method, "out")._cut(ROWS, TRAIN_SIZE)
    assert not set(clusters[test]) & set(clusters[train])


def test_in_shares_its_scaffolds_with_train(structural_method: str,
                                            clusters: np.ndarray) -> None:
    """Every test scaffold is one train has seen, which is the point of the 'in' split."""
    train, test = splitter(structural_method, "in")._cut(ROWS, TRAIN_SIZE)
    assert set(clusters[test]) <= set(clusters[train])


def test_cluster_split_is_seeded(distribution: str) -> None:
    """Two seeds give two splits, so a replicate is a new seed rather than a new method."""
    first = splitter("scaffold", distribution, seed=1)._cut(ROWS, TRAIN_SIZE)[1]
    second = splitter("scaffold", distribution, seed=2)._cut(ROWS, TRAIN_SIZE)[1]
    assert not np.array_equal(first, second)


def test_fingerprint_split_is_deterministic(distribution: str) -> None:
    """The deal ignores the seed: same molecules in, same split out."""
    first = splitter("fingerprint", distribution, seed=1)._cut(ROWS, TRAIN_SIZE)[1]
    second = splitter("fingerprint", distribution, seed=2)._cut(ROWS, TRAIN_SIZE)[1]
    assert np.array_equal(first, second)


def test_three_way_split_is_a_partition(method: str, distribution: str) -> None:
    """Splitting the remainder a second time still places every molecule exactly once."""
    splits = splitter(method, distribution).split()
    assert sorted(np.concatenate(splits).tolist()) == list(range(len(SMILES)))


def test_three_way_split_sizes(method: str, distribution: str) -> None:
    """Train, test and valid come out the sizes asked for, to within a cluster."""
    train, test, valid = splitter(method, distribution).split()
    sizes = [len(split) / len(SMILES) for split in (train, test, valid)]
    expected = [TRAIN_SIZE, TEST_SIZE, 1 - TRAIN_SIZE - TEST_SIZE]
    assert sizes == pytest.approx(expected, abs=CLUSTER_TOLERANCE)


def test_three_way_out_puts_each_cluster_in_one_split(clusters: np.ndarray) -> None:
    """The second cut keeps the first's guarantee: no scaffold is shared by two splits."""
    splits = splitter("scaffold", "out").split()
    train, test, valid = (set(clusters[split]) for split in splits)
    assert not (train & test or train & valid or test & valid)


def test_three_way_in_keeps_test_and_valid_clusters_in_train(
    structural_method: str, clusters: np.ndarray
) -> None:
    """Neither held-out set brings a scaffold train has not seen, under either method."""
    train, test, valid = splitter(structural_method, "in").split()
    assert set(clusters[test]) <= set(clusters[train])
    assert set(clusters[valid]) <= set(clusters[train])


def test_unknown_method_raises() -> None:
    """A mistyped method fails at construction, before any chemistry is computed."""
    with pytest.raises(ValueError, match="unknown method"):
        splitter("scaffolds")


def test_unknown_distribution_raises() -> None:
    """A mistyped distribution fails at construction, not silently as one of the two."""
    with pytest.raises(ValueError, match="unknown distribution"):
        splitter("scaffold", "sideways")


def test_sizes_must_leave_a_validation_set() -> None:
    """Train and test that fill the set leave nothing to select on, so they are refused."""
    with pytest.raises(ValueError, match="validation set"):
        Splitters(SMILES, "scaffold", "in", 0.9, 0.1)


def test_min_cluster_size_drops_small_clusters_from_every_split() -> None:
    """A one-off scaffold is left out of all three splits; the rest are untouched."""
    smiles = [*SMILES, "C1CCC2CCCCC2C1"]  # the only member of its scaffold
    kept = np.concatenate(
        Splitters(smiles, "scaffold", "out", min_cluster_size=2).split()
    )
    assert len(smiles) - 1 not in kept
    assert len(kept) == len(SMILES)


def test_min_cluster_size_needs_clusters() -> None:
    """The fingerprint split has no clusters to measure, so a cut-off is refused."""
    with pytest.raises(ValueError, match="needs clusters"):
        Splitters(SMILES, "fingerprint", min_cluster_size=2)
