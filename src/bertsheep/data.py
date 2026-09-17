import csv
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rdkit import Chem

MAX_KI_NM = 100_000
MAX_SMILES_LENGTH = 128
CHUNK_SIZE = 200_000

# BindingDB_All.tsv is 640 columns wide: 40 fixed, then a 12-column block per
# target chain, repeated 50 times. These are the only ones we need.
COLUMNS = {
    "Ligand SMILES": "smiles",
    "Ki (nM)": "labels",
    "Number of Protein Chains in Target (>1 implies a multichain complex)": "n_chains",
    "UniProt (SwissProt) Entry Name of Target Chain 1": "entry_name",
    "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id",
}

TARGET = {
    "EGFR": ("EGFR_HUMAN", "P00533")
}

class Data():
    """
    Loads binding affinity data from BindingDB and fetches data
    """
    def __init__(self, data_path, target):
        if target not in TARGET:
            raise KeyError(f"unknown target {target!r}; choose from {sorted(TARGET)}")
        self.data_path = data_path
        self.target = target
        self.entry_name, self.uniprot_id = TARGET[target]

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

    def _select_target(self, df):
        """
        Keeps single-chain rows measured against the selected target. Vectorised:
        two column-wide equality comparisons, no per-row string building.
        """
        mask = (
            (df["uniprot_id"] == self.uniprot_id)
            & (df["entry_name"] == self.entry_name)
            & (df["n_chains"] == "1")
        )
        return df.loc[mask, ["smiles", "labels"]]

    def _filter(self, df):
        """
        Drops weak binders (Ki >= 100uM) and SMILES too long to tokenise.
        """
        df = df[df["labels"] < float(MAX_KI_NM)]
        return df[df["smiles"].str.len() < MAX_SMILES_LENGTH]

    def _deduplicate(self, df):
        """
        Collapses repeated measurements of the same ligand to their median Ki.
        """
        return df.groupby("smiles", as_index=False)["labels"].median()

    def _canonicalise(self, df):
        """
        Canonicalises the SMILES column and drops rows RDKit could not parse.
        """
        df = df.copy()
        df["smiles"] = df["smiles"].apply(self._canonicalise_smiles)
        return df[df["smiles"].notna()]

    def _transform_labels(self, df):
        """
        Converts Ki (nM) to the -log scale the model is trained on.
        """
        df = df.copy()
        df["labels"] = -np.log(df["labels"])
        return df

    def _preprocess(self):
        """
        Runs load -> filter -> deduplicate -> canonicalise -> transform.
        """
        df = self._load()
        df = self._filter(df)
        df = self._deduplicate(df)
        df = self._canonicalise(df)
        df = self._transform_labels(df)
        print(f"-- Training on a total dataset of {len(df)} labels")
        return df
