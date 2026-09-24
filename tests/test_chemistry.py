from pathlib import Path

import numpy as np
import pytest

from bertsheep.chemistry import Chemist
from bertsheep.eda import BUTINA_CUTOFFS

# Three toluene homologues reduce to one benzene scaffold, so the largest group
# is known by construction; naphthalene and piperidine are one each.
BENZENES = ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"]
NAPHTHALENE = "c1ccc2ccccc2c1"
PIPERIDINE = "C1CCNCC1"
SMILES = [*BENZENES, NAPHTHALENE, PIPERIDINE]
BENZENE_SCAFFOLD = "c1ccccc1"


@pytest.fixture
def chemist() -> Chemist:
    """A Chemist on the default ECFP4 settings."""
    return Chemist()


@pytest.fixture(params=["scaffold", "butina"])
def clusters(request, chemist: Chemist) -> np.ndarray:
    """Cluster IDs for SMILES from each method, to assert one shared contract."""
    if request.param == "scaffold":
        return chemist.scaffold_clusters(SMILES)[0]
    return chemist.butina_clusters(chemist.pairwise_tanimoto(chemist.fingerprints(SMILES)))


def test_clusters_are_one_id_per_molecule(clusters: np.ndarray) -> None:
    """Both methods return a 1-D integer array as long as the input."""
    assert isinstance(clusters, np.ndarray)
    assert clusters.shape == (len(SMILES),)
    assert np.issubdtype(clusters.dtype, np.integer)


def test_cluster_ids_are_contiguous_from_zero(clusters: np.ndarray) -> None:
    """IDs label groups 0..k-1 with no gaps, so they index a palette directly."""
    assert set(clusters.tolist()) == set(range(clusters.max() + 1))


def test_cluster_zero_is_the_largest_group(clusters: np.ndarray) -> None:
    """ID 0 is the biggest group under either method, so an ID means one thing."""
    sizes = np.bincount(clusters)
    assert sizes[0] == sizes.max()
    assert (np.diff(sizes) <= 0).all()


def test_scaffold_clusters_group_by_framework(chemist: Chemist) -> None:
    """Molecules sharing a Bemis-Murcko framework share an ID, and only those."""
    ids, _ = chemist.scaffold_clusters(SMILES)
    assert len(set(ids[:3])) == 1
    assert len(set(ids.tolist())) == 3


def test_scaffold_clusters_return_the_scaffold_smiles(chemist: Chemist) -> None:
    """The second element is the scaffold per molecule, aligned with the IDs."""
    ids, scaffolds = chemist.scaffold_clusters(SMILES)
    assert len(scaffolds) == len(ids)
    assert scaffolds[:3] == [BENZENE_SCAFFOLD] * 3
    assert scaffolds[3] == NAPHTHALENE


def test_scaffold_clusters_keep_unscaffoldable_molecules(chemist: Chemist) -> None:
    """
    An acyclic molecule ('') and one RDKit cannot parse (None) still get a row,
    so the IDs stay aligned with the input frame rather than silently shortening.
    """
    ids, scaffolds = chemist.scaffold_clusters([*BENZENES, "CCCC", "not a molecule"])
    assert len(ids) == len(scaffolds) == 5
    assert scaffolds[3] == ""
    assert scaffolds[4] is None


def test_butina_splits_and_merges_with_the_cutoff(chemist: Chemist) -> None:
    """A tight cutoff isolates every molecule; a loose one collapses them to one."""
    distances = chemist.pairwise_tanimoto(chemist.fingerprints(SMILES))
    assert chemist.butina_clusters(distances, cutoff=0.0).max() == len(SMILES) - 1
    assert chemist.butina_clusters(distances, cutoff=1.0).max() == 0


def test_butina_clusters_identical_molecules_together(chemist: Chemist) -> None:
    """
    The same molecule written two ways is one cluster at any cutoff. This is the
    regression test for reading the input as similarity: inverted, two molecules
    at distance 0 look maximally dissimilar and are split apart.
    """
    smiles = ["Cc1ccccc1", "c1ccccc1C", NAPHTHALENE, PIPERIDINE]
    clusters = chemist.butina_clusters(
        chemist.pairwise_tanimoto(chemist.fingerprints(smiles)), cutoff=0.1
    )
    assert clusters[0] == clusters[1]
    assert len(set(clusters.tolist())) == 3


def test_pairwise_tanimoto_is_a_distance_matrix(chemist: Chemist) -> None:
    """Symmetric, zero on the diagonal, and in [0, 1] -- distance, not similarity."""
    distances = chemist.pairwise_tanimoto(chemist.fingerprints(SMILES))
    assert distances.shape == (len(SMILES), len(SMILES))
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0)
    assert ((distances >= 0) & (distances <= 1)).all()
    # Two benzene homologues are nearer each other than either is to piperidine.
    assert distances[1, 2] < distances[1, 4]


def test_fingerprints_are_one_bit_vector_per_molecule(chemist: Chemist) -> None:
    """Fingerprints stack into an (n, FP_BITS) array the distance metric can take."""
    fps = chemist.fingerprints(SMILES)
    assert fps.shape == (len(SMILES), 1024)
    assert set(np.unique(fps).tolist()) <= {0, 1}


def test_cached_tanimoto_hit_returns_the_saved_matrix(
        chemist: Chemist, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call with the same fingerprints reads the file, not a rebuild."""
    fps = chemist.fingerprints(SMILES)
    first = chemist.cached_tanimoto(fps, tmp_path, "TEST-wildtype")
    assert np.array_equal(first, chemist.pairwise_tanimoto(fps).astype(np.float32))

    def rebuild(fps: np.ndarray) -> np.ndarray:
        """Stand-in for pairwise_tanimoto that fails if the cache is missed."""
        raise AssertionError("cache missed")

    monkeypatch.setattr(chemist, "pairwise_tanimoto", rebuild)
    second = chemist.cached_tanimoto(fps, tmp_path, "TEST-wildtype")
    assert np.array_equal(first, second)
    assert second.dtype == np.float32
    assert len(list(tmp_path.glob("*.npy"))) == 1


@pytest.mark.parametrize("changed, fp_bits", [
    (SMILES[::-1], 1024),  # same molecules, different row order
    (SMILES[:-1], 1024),  # preprocessing dropped a molecule
    (SMILES, 2048),  # different fingerprint settings
])
def test_cached_tanimoto_misses_on_a_changed_frame(
        tmp_path: Path, changed: list[str], fp_bits: int) -> None:
    """
    A frame that differs in rows, order or fingerprint settings gets its own
    matrix under the same name, never the one saved for the original.
    """
    original = Chemist()
    original.cached_tanimoto(original.fingerprints(SMILES), tmp_path, "TEST-wildtype")
    chemist = Chemist(fp_bits=fp_bits)
    fps = chemist.fingerprints(changed)
    cached = chemist.cached_tanimoto(fps, tmp_path, "TEST-wildtype")
    assert np.array_equal(cached, chemist.pairwise_tanimoto(fps).astype(np.float32))
    assert len(list(tmp_path.glob("*.npy"))) == 2


@pytest.mark.parametrize("cutoff", BUTINA_CUTOFFS)
def test_cached_float32_matrix_gives_the_same_clusters(
        chemist: Chemist, tmp_path: Path, cutoff: float) -> None:
    """The float32 cache clusters exactly as the float64 matrix at every cutoff."""
    fps = chemist.fingerprints(SMILES)
    assert np.array_equal(
        chemist.butina_clusters(chemist.cached_tanimoto(fps, tmp_path, "TEST"), cutoff),
        chemist.butina_clusters(chemist.pairwise_tanimoto(fps), cutoff),
    )
