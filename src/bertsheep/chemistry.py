from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina
from sklearn.metrics import pairwise_distances

FP_BITS = 1024
FP_RADIUS = 2  # radius 2 over 1024 bits is ECFP4
BUTINA_CUTOFF = 0.5  # Tanimoto distance


class Chemist:
    """
    Turn SMILES into the structural representations the splits and figures
    need -- fingerprints, Tanimoto distances, and the two ways of grouping
    molecules by structure.

    Both grouping methods return the same thing, a cluster ID per molecule, so
    a split or a figure can take either without knowing which it was handed.

    Parameters
    ----------
    fp_bits : int
        Length of the Morgan fingerprint bit vector.
    fp_radius : int
        Morgan radius; 2 gives ECFP4.
    """

    def __init__(self, fp_bits: int = FP_BITS, fp_radius: int = FP_RADIUS) -> None:
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_bits
        )

    def fingerprints(self, smiles: Iterable[str]) -> np.ndarray:
        """
        ECFP4 bit vectors as a numpy array.

        Parameters
        ----------
        smiles : Iterable[str]
            SMILES strings, all parseable by RDKit.

        Returns
        -------
        np.ndarray
            (n, fp_bits) uint8 array, one row per molecule.
        """
        return np.array([
            self.generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles
        ])

    def pairwise_tanimoto(self, fps: np.ndarray) -> np.ndarray:
        """
        Square matrix of pairwise Tanimoto distances. On bit vectors Tanimoto is
        the Jaccard index, so sklearn computes the whole matrix in one call --
        no reason to loop over BulkTanimotoSimilarity and mirror the triangle by
        hand. The bits go in as bool because the boolean metrics are the ones
        that read them as set membership rather than as numbers.

        Parameters
        ----------
        fps : np.ndarray
            (n, fp_bits) bit vectors, from fingerprints().

        Returns
        -------
        np.ndarray
            (n, n) symmetric distance matrix, 0 on the diagonal.
        """
        return pairwise_distances(fps.astype(bool), metric="jaccard")

    def scaffold_clusters(self, smiles: Iterable[str]) -> tuple[np.ndarray, list[str | None]]:
        """
        Group molecules by Bemis-Murcko scaffold. Chirality is excluded so
        enantiomers share a scaffold -- otherwise a pair with near-identical
        affinity can straddle a scaffold split. The scaffold SMILES come back
        alongside the IDs because the figures label groups with them, and
        recomputing the parse to get them is most of the cost of the method.

        Two falsy scaffolds are possible: '' for an acyclic molecule, which has
        no ring system to reduce to, and None where RDKit cannot parse the
        SMILES. Both keep their place in the output -- dropping them would
        misalign the IDs against the frame they came from -- and both group
        together like any other key, so a caller that must reject them (a
        scaffold split) can test for them. A frame from Data._preprocess
        contains neither, but this is also the method reached for interactively
        against raw BindingDB SMILES. That is also why the parse goes through
        MolFromSmiles instead of handing the string straight to
        MurckoScaffoldSmiles, which raises on a molecule it cannot read.

        Parameters
        ----------
        smiles : Iterable[str]
            SMILES strings.

        Returns
        -------
        tuple[np.ndarray, list[str | None]]
            Cluster ID per molecule, and the scaffold SMILES it was grouped by:
            '' if acyclic, None if unparseable.
        """
        mols = (Chem.MolFromSmiles(s) for s in smiles)
        scaffolds = [
            None if mol is None else MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            for mol in mols
        ]
        return self._cluster_ids(scaffolds), scaffolds

    def butina_clusters(self, distances: np.ndarray,
                        cutoff: float = BUTINA_CUTOFF) -> np.ndarray:
        """
        Group molecules by Butina clustering of their Tanimoto distances. Takes
        the matrix rather than the SMILES because it is the expensive part and
        does not depend on the cutoff, so a sweep over cutoffs pays for it once.

        ClusterData reads the matrix as distances, not similarities: handing it
        similarities silently inverts every neighbourhood, and identical
        molecules -- distance 0 -- come out in different clusters.

        Parameters
        ----------
        distances : np.ndarray
            (n, n) Tanimoto distance matrix, from pairwise_tanimoto().
        cutoff : float
            Tanimoto distance within which molecules are neighbours.

        Returns
        -------
        np.ndarray
            Cluster ID per molecule; 0 is the largest cluster.
        """
        n = len(distances)
        clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
        keys = np.empty(n, dtype=int)
        for index, cluster in enumerate(clusters):
            keys[list(cluster)] = index
        return self._cluster_ids(keys)

    def _cluster_ids(self, keys: Sequence[str | int | None]) -> np.ndarray:
        """
        Number a grouping key, whatever its type, as contiguous cluster IDs with
        0 the largest group. Both grouping methods end here, so an ID means the
        same thing whichever produced it: the two are comparable, and a figure
        can colour either by ID without a lookup.

        factorize rather than a dict keyed on the scaffolds themselves, because
        None is a key here and does not compare equal to itself once pandas has
        made it NaN; use_na_sentinel keeps it a group rather than a -1 gap.

        Parameters
        ----------
        keys : Sequence[str | int | None]
            Grouping key per molecule, e.g. scaffold SMILES or Butina cluster.

        Returns
        -------
        np.ndarray
            Cluster ID per molecule, 0 to n_clusters - 1.
        """
        codes = pd.Series(keys).factorize(use_na_sentinel=False)[0]
        # Codes ordered by falling group size, inverted to give each code its rank.
        largest_first = np.argsort(-np.bincount(codes), kind="stable")
        return np.argsort(largest_first)[codes]
