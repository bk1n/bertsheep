import os

# huggingface_hub reads this once at import, so it has to be set before any
# test module imports transformers. Tests use the locally cached model and must
# never hit the network.
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bertsheep.experiment as be
import bertsheep.tuning as bt
from bertsheep.experiment import Experiment

# Ten ring systems, each carried by four alkyl homologues, so Butina has
# clusters to group and both distributions have something to hold out.
CORES = ("c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "C1CCNCC1", "C1CCNC1",
         "c1ccco1", "c1ccsc1", "c1cc2ccccc2[nH]1", "c1ccc2ccccc2n1", "C1CCOCC1")
ALKYLS = ("C", "CC", "CCC", "CCCC")
SMILES = [alkyl + core for core in CORES for alkyl in ALKYLS]


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Experiment:
    """
    An Experiment over the homologue series, writing its results and tuned
    parameters into tmp_path. min_cluster_size 1 keeps every molecule, since
    forty is already few.
    """
    monkeypatch.setattr(be, "EXPERIMENT_DIR", tmp_path / "experiments")
    monkeypatch.setattr(bt, "TUNING_DIR", tmp_path / "tuning")
    labels = np.random.default_rng(0).normal(6, 1, len(SMILES))
    frame = pd.DataFrame({"smiles": SMILES, "labels": labels})
    frame.attrs = {"data_path": "test.tsv", "mutation": None}
    return Experiment(frame, "TEST", mutation=None, min_cluster_size=1)
