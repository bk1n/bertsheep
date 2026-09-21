import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rdkit import Chem

from bertsheep.chemistry import Chemist

MAX_KI_NM = 100_000  # unused while the affinity cutoff is dropped
MAX_SMILES_LENGTH = 128
CHUNK_SIZE = 200_000
LABEL = "ic50"  # which affinity column becomes the training label
CACHE_DIR = Path("out/.cache")
WILD_TYPE = "wildtype"  # mutation argument selecting the unmutated construct

# Target Name carries the construct in brackets, e.g.
#   "Epidermal growth factor receptor [1-745,751-1210,T790M]"
# BindingDB's TSV spec documents the ranges as the residues the construct
# *includes*, and "L692P"-style tokens as point mutations, so a gap between two
# ranges is absent sequence. Quoted peptides ('NPG') are NOT documented; they are
# read as insertions because they always sit between two contiguous ranges and
# reproduce the known EGFR exon-20 insertions.
# https://www.bindingdb.org/rwd/bind/chemsearch/marvin/BindingDB-TSV-Format.pdf
POINT_MUTATION = re.compile(r"[A-Z]\d+[A-Z]")
RESIDUE_RANGE = re.compile(r"(\d+)-(\d+)")
INSERTION = re.compile(r"'([A-Z]+)'")

# BindingDB_All.tsv is 640 columns wide: 40 fixed, then a 12-column block per
# target chain, repeated 50 times. These are the only ones we need.
COLUMNS = {
    "Ligand SMILES": "smiles",
    "Target Name": "target_name",
    "Ki (nM)": "ki",
    "IC50 (nM)" : "ic50",
    "Number of Protein Chains in Target (>1 implies a multichain complex)": "n_chains",
    "UniProt (SwissProt) Entry Name of Target Chain 1": "entry_name",
    "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
}

TARGET = {
    "EGFR": ("EGFR_HUMAN", "P00533")
}

class Data():
    """
    Loads binding affinity data from BindingDB and fetches data.

    Parameters
    ----------
    data_path : str | Path
        Path to the raw BindingDB TSV dump.
    target : str
        Short target name, a key of TARGET.
    mutation : str | None
        Variant to keep: WILD_TYPE for the unmutated construct, a mutations
        string as produced by _parse_annotation (e.g. "L858R,T790M") for one
        mutant, or None to keep every variant.
    """
    def __init__(self, data_path: str | Path, target: str, mutation: str | None = None) -> None:
        if target not in TARGET:
            raise KeyError(f"unknown target {target!r}; choose from {sorted(TARGET)}")
        self.data_path = Path(data_path)
        self.target = target
        self.mutation = mutation
        self.entry_name, self.uniprot_id = TARGET[target]
        self.cache_path = CACHE_DIR / f"{self.data_path.stem}_{target}.parquet"

    def _fetch_data(self):
        load_dotenv()
        windows_path = os.environ["DATA_PATH"]
        src = subprocess.run(
            ["wslpath", "-u", windows_path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        subprocess.run(
            ["rsync", "-ah", "--info=progress2", "--stats", f"{src}/", f"{self.data_path}/"],
            check=True,
        )
        print(f"Data fetched from {windows_path} on {datetime.now()}")

    def _canonicalise_smiles(self, smiles):
        """
        Canonicalises a single SMILES string, returning None if RDKit cannot parse it.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, True)
        except Exception:
            return None

    def _load(self):
        """
        Loads BindingDB data into pandas dataframe, keeping only the selected
        target. Read in chunks so the 8.4GB source is never fully resident.
        """
        chunks = pd.read_table(
            self.data_path,
            usecols=list(COLUMNS),
            dtype=str,
            quoting=csv.QUOTE_NONE,
            on_bad_lines="warn",
            chunksize=CHUNK_SIZE,
        )
        df = pd.concat(
            (self._select_target(chunk.rename(columns=COLUMNS)) for chunk in chunks),
            ignore_index=True,
        )
        print(f"-- {len(df)} rows for {self.target} ({TARGET[self.target]})")
        return df

    def _load_cached(self) -> pd.DataFrame:
        """
        Returns the target's raw rows from the parquet cache, building the
        cache with _load on first use. Scanning the full TSV takes minutes;
        the cached target frame reads in well under a second. The cache holds
        every variant, unfiltered, so one file serves all mutation arguments
        and every downstream stage still runs on each call.

        The cache is keyed on the dump's file name and the target only: it is
        not invalidated if the TSV is replaced in place, so delete CACHE_DIR
        after refetching the same dump.

        Returns
        -------
        pd.DataFrame
            Target rows with smiles, target_name, ki and ic50 columns.
        """
        if self.cache_path.exists():
            print(f"-- Reading cached {self.cache_path}")
            return pd.read_parquet(self.cache_path)
        df = self._load()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        return df

    def _select_target(self, df):
        """
        Keeps single-chain rows measured against the selected target. Vectorised:
        three column-wide equality comparisons, no per-row string building.
        """
        mask = (
            (df["uniprot_id"] == self.uniprot_id)
            & (df["entry_name"] == self.entry_name)
            & (df["n_chains"] == "1")
        )
        return df.loc[mask, ["smiles", "target_name", "ki", "ic50"]]

    def _filter(self, df):
        """
        Promotes the chosen affinity column to "labels" and coerces it to numeric,
        dropping rows with no value or a censored one (">400000", "<0.5"). Then
        drops SMILES too long to tokenise.
        """
        df = df.assign(labels=pd.to_numeric(df[LABEL], errors="coerce"))
        df = df[df["labels"] > 0]  # drops blanks, censored values, and IC50 == 0
        return df[df["smiles"].str.len() < MAX_SMILES_LENGTH]

    def _parse_annotation(self, name):
        """
        Reduces the bracketed construct on Target Name to its mutations:
            "...[1-745,751-1210,T790M]" -> "T790M,del746-750"
            "...[1-770,'NPG',771-1210]" -> "insNPG"
            "Epidermal growth factor receptor" -> ""   (wild type)
        Point mutations and insertions are read directly; deletions are inferred
        from gaps between consecutive residue ranges. A range that merely starts
        late or stops early is a construct boundary, not a mutation, so only
        internal gaps count -- BindingDB itself calls such truncations deletions,
        but a kinase-domain construct binds ATP-site ligands like the full protein.
        """
        bracket = re.search(r"\[(.*)\]", name)
        if not bracket:
            return ""
        found, ranges = [], []
        for token in (t.strip() for t in bracket.group(1).split(",")):
            if POINT_MUTATION.fullmatch(token):
                found.append(token)
            elif RESIDUE_RANGE.fullmatch(token):
                start, end = RESIDUE_RANGE.fullmatch(token).groups()
                ranges.append((int(start), int(end)))
            elif INSERTION.fullmatch(token):
                found.append("ins" + INSERTION.fullmatch(token).group(1))
        ranges.sort()
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            if start > end + 1:
                found.append(f"del{end + 1}-{start - 1}")
        return ",".join(sorted(found))

    def _extract_mutations(self, df):
        """
        Adds a "mutations" column, empty for wild type. Parsed over the distinct
        Target Name strings only (EGFR has 58) and mapped back, so the regex work
        is done tens of times rather than tens of thousands.
        """
        names = df["target_name"].fillna("")
        lookup = {name: self._parse_annotation(name) for name in names.unique()}
        return df.assign(mutations=names.map(lookup))

    def _select_mutation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keeps only the variant named by the mutation argument. Matching is
        exact on the sorted, comma-joined mutations string, so a double mutant
        does not also pull in its single mutants. An argument matching no rows
        raises rather than returning an empty frame, since a misspelt or
        misordered variant would otherwise surface much later as a crash in
        splitting or training.

        Parameters
        ----------
        df : pd.DataFrame
            Frame with a mutations column, "" for wild type.

        Returns
        -------
        pd.DataFrame
            Rows for the selected variant, or df unchanged if mutation is None.
        """
        if self.mutation is None:
            return df
        wanted = "" if self.mutation == WILD_TYPE else self.mutation
        df = df[df["mutations"] == wanted]
        if df.empty:
            raise ValueError(f"no rows for mutation {self.mutation!r}")
        print(f"-- {len(df)} rows for mutation {self.mutation!r}")
        return df

    def _deduplicate(self, df):
        """
        Collapses repeated measurements of the same ligand against the same
        variant to their median IC50. Wild type and each mutant stay separate.
        """
        return df.groupby(["smiles", "mutations"], as_index=False)["labels"].median()

    def _canonicalise(self, df):
        """
        Canonicalises the SMILES column and drops rows RDKit could not parse.
        """
        df = df.copy()
        df["smiles"] = df["smiles"].apply(self._canonicalise_smiles)
        return df[df["smiles"].notna()]

    def _transform_labels(self, df):
        """
        Converts IC50 (nM) to the -log scale the model is trained on.
        """
        df = df.copy()
        df["labels"] = -np.log(df["labels"])
        return df

    def _drop_acyclic(self, df):
        """
        Drops molecules with no ring system. Their Bemis-Murcko scaffold is the
        empty string, so every one of them lands in a single scaffold group --
        and that group is normally big enough to swamp whichever side of a
        scaffold split it falls on, which is how a scaffold split quietly stops
        being one. They are removed rather than grouped around because a ligand
        with no ring is not a kinase inhibitor chemotype to begin with: solvents,
        salts and small fragments, not compounds the model should be scored on.

        Tested on the scaffold itself rather than on a ring count, so the test
        is the same computation the scaffold split later groups on: whatever
        survives this is guaranteed to have a scaffold to be grouped by.

        Runs after canonicalisation, so every SMILES here parses, and before
        deduplication, so the scaffold perception is done on the smaller frame.
        """
        # Chemist gives '' for an acyclic molecule and None for one it cannot
        # parse. Both are falsy and neither can carry a scaffold split.
        #
        # Not Mol.GetRingInfo().NumRings(): with torch imported before rdkit
        # that intermittently raises "RingInfo not initialized" on a freshly
        # parsed molecule, which is the import order every entry point here
        # uses. MurckoScaffoldSmiles does its own ring perception and is stable
        # under it.
        scaffolds = Chemist().scaffold_clusters(df["smiles"])[1]
        cyclic = np.array([bool(scaffold) for scaffold in scaffolds])
        print(f"-- Dropped {(~cyclic).sum()} acyclic molecules")
        return df[cyclic]

    def _preprocess(self):
        """
        Runs load (cached) -> filter -> mutations -> select mutation ->
        canonicalise -> drop acyclic -> deduplicate -> transform.
        Canonicalising before deduplicating means the same molecule written
        two ways collapses into one group. Selecting the mutation before
        canonicalising keeps the RDKit work to the rows actually kept.
        """
        df = self._load_cached()
        df = self._filter(df)
        df = self._extract_mutations(df)
        df = self._select_mutation(df)
        df = self._canonicalise(df)
        df = self._drop_acyclic(df)
        df = self._deduplicate(df)
        df = self._transform_labels(df)
        print(f"-- Training on a total dataset of {len(df)} labels")
        return df
