import json
import random
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging,
)

from bertsheep.splitters import Splitters

MODEL_DIR = Path("out/models")
INIT_CHECKPOINT = "init.pt"  # epoch -1: the weights fine-tuning starts from
EPOCH_CHECKPOINT = "epoch{:03d}.pt"
CONFIG = "config.json"
SPLITS = "splits.parquet"
SPLIT_NAMES = ("train", "test", "valid")  # the order Splitters.split() deals in
HISTORY = "history.csv"

# Tokeniser truncation length. Data.MAX_SMILES_LENGTH caps SMILES at 128
# *characters*, which is a looser bound than 128 tokens, so truncation is rare.
MAX_TOKENS = 128
EMBEDDING_BATCH_SIZE = 256  # inference only, so larger than training fits

MAX_GRAD_NORM = 1.0
NO_DECAY = ("bias", "LayerNorm.weight")

MIN_DELTA = 0.0  # improvement in test loss that resets early stopping
LOG_EVERY = 5  # batches between training-loss lines

# Default run seed. A replicate is a new seed here as well as on the splitter:
# without it, head initialisation and batch order vary run to run and their
# spread is indistinguishable from the spread across splits.
MODEL_SEED = 0

LOSS_FN = torch.nn.MSELoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bf16 rather than fp16: it keeps fp32's exponent range, so no loss scaling.
# Autocast keeps the weights and optimiser state in fp32; only the forward
# pass runs in bf16, and only on CUDA.
AUTOCAST = True


def autocast() -> torch.autocast:
    """
    Mixed-precision context for a forward pass, shared by training, scoring
    and embedding extraction so every number comes from the same arithmetic
    the weights were trained with.

    Returns
    -------
    torch.autocast
        bf16 autocast on CUDA when `AUTOCAST` is set, otherwise a no-op.
    """
    return torch.autocast(
        DEVICE.type, dtype=torch.bfloat16,
        enabled=AUTOCAST and DEVICE.type == "cuda",
    )


@torch.inference_mode()
def embeddings(checkpoint: Path, smiles: Iterable[str],
               model_link: str) -> np.ndarray:
    """
    Per-layer molecule embeddings from a checkpoint's weights, for comparing
    the latent space across epochs. A module-level function rather than a
    Model method because it needs only a run directory and the preprocessed
    frame: config.json gives `model_link` and split_frames() gives the SMILES,
    so no splitter has to be rebuilt to read finished runs.

    Each layer is mean-pooled over the attention mask rather than read at the
    `<s>` token, which RoBERTa only trains through the classification head and
    so carries little in the intermediate layers.

    Parameters
    ----------
    checkpoint : Path
        Checkpoint written by Model, e.g. `init.pt` or `epoch007.pt`.
    smiles : Iterable[str]
        Molecules to embed, typically the `smiles` column of a frame from
        split_frames().
    model_link : str
        Hugging Face ID the run was fine-tuned from, which fixes the
        architecture and tokeniser the weights load into.

    Returns
    -------
    np.ndarray
        Shape (n_layers + 1, n_molecules, hidden), float32, in input order.
        Slice 0 is the embedding layer, the rest are the encoder layers.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_link)
    model = RobertaForSequenceClassification.from_pretrained(model_link, num_labels=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.to(DEVICE).eval()

    encoded = tokenizer(
        list(smiles), padding=True, truncation=True,
        max_length=MAX_TOKENS, return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoded["input_ids"], encoded["attention_mask"]),
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    pooled = []
    for input_ids, mask in loader:
        input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
        with autocast():
            out = model(input_ids=input_ids, attention_mask=mask,
                        output_hidden_states=True)
        hidden = torch.stack(out.hidden_states).float()  # (layers, batch, seq, dim)
        weights = mask.unsqueeze(-1).float()  # (batch, seq, 1), broadcast over layers
        pooled.append(((hidden * weights).sum(2) / weights.sum(1)).cpu())
    return torch.cat(pooled, dim=1).numpy()


def split_frames(
    df: pd.DataFrame, model_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rebuild a run's train, test and valid frames from the positional indices
    in its splits.parquet. A run stores indices rather than frames so the
    preprocessed data is not copied into every directory of the grid; the
    frames, and predictions for any split at any epoch via a checkpoint
    forward pass, are recovered from the one preprocessed frame instead.
    Model builds its own splits through here too, so a recovered split is the
    one that was trained on by construction rather than by agreement.

    The indices are taken with iloc because Splitters deals positions -- a
    preprocessed frame carries the raw dump's row numbers as its index, which
    .loc would read as labels.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame the run was split over, in the same order.
    model_dir : Path
        Run directory holding splits.parquet.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, test and valid frames, rows in the order the splitter dealt them.
    """
    splits = pd.read_parquet(model_dir / SPLITS)
    train, test, valid = (
        df.iloc[splits.loc[splits["split"] == name, "row"]] for name in SPLIT_NAMES
    )
    return train, test, valid


class Model():
    """
    Fine-tunes ChemBERTa as a regressor on a preprocessed (smiles, labels) frame.
    Hyperparameters are constructor arguments rather than module constants so a
    sweep can build one Model per trial without mutating shared state.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns, as returned by
        Data._preprocess, whose `attrs` carry the `data_path` and `mutation`
        it was built from into config.json.
    target : str
        Short target name, used in the run directory name.
    splitter : Splitters
        Splitter the three-way split is taken from, built by the caller over
        this frame's SMILES. It is passed in rather than chosen here because
        the split is the experiment rather than a hyperparameter, and because a
        splitter does its structural work -- fingerprints, distance matrix,
        cluster IDs -- once at construction, so a sweep hands the same one to
        every Model instead of rebuilding it per trial. Its indices are
        positional into the SMILES it was built from, so `df` has to be that
        frame, in that order.
    seed : int
        Seed for head initialisation, dropout and batch order. Independent of
        the splitter's seed so the split can be held fixed while the model is
        resampled, or the reverse, which is what separates variance due to the
        chemistry held out from variance due to the fit.
    reinit_n : int
        Number of top encoder layers to re-initialise before fine-tuning.
    model_link : str
        Hugging Face ID of the pretrained ChemBERTa.
    lr : float
        Learning rate of the head; lower layers are scaled by llrd_decay.
    batch_size : int
        Batch size for every split.
    num_epochs : int
        Upper bound on epochs; early stopping usually ends training sooner.
    warmup_ratio : float
        Fraction of total steps spent in linear warmup, before linear decay to
        zero, stepped per batch. This is the standard BERT fine-tuning schedule
        and the one that matters here: AdamW's second-moment estimate is
        meaningless for the first few dozen steps, and the randomly initialised
        regression head sends large gradients back into the pretrained encoder,
        so a cold start at full LR is what wrecks small-dataset fine-tunes. The
        legacy alternatives were stepped per *epoch* (LinearLR over num_epochs,
        ExponentialLR), which gives no warmup at all.
    weight_decay : float
        AdamW weight decay, applied to every parameter except NO_DECAY ones.
    llrd_decay : float
        Layerwise LR decay: each layer below the head trains at llrd_decay x
        the rate of the one above it. Lower layers hold general SMILES grammar
        and want to move less than the head. 1.0 gives uniform AdamW.
    dropout : float | None
        Overrides the checkpoint's `hidden_dropout_prob`, which also governs
        the classification head while its own `classifier_dropout` is unset.
        None keeps whatever the checkpoint was published with -- ChemBERTa's
        0.144 was tuned for multi-task pretraining over millions of molecules,
        which is a different overfitting regime from a 17k-row downstream fit,
        so it is worth searching rather than inheriting.
    patience : int
        Epochs without test-loss improvement before early stopping.
    checkpoint : bool
        Write the weights every epoch. A hyperparameter search wants this off:
        the weights are ~40 MB an epoch and a search keeps only the winning
        *parameters*, refitting from them afterwards. evaluate() restores the
        best epoch from memory rather than from disk, so a run fitted with
        this off can still be evaluated.
    freeze : bool
        Train the regression head only, with the encoder's weights fixed at
        their pretrained values -- the pre-trained arm of the comparison. It
        shares the loop, schedule, early stopping and selection with the
        fine-tuned arm, so the two differ in which weights move and nothing
        else. Expects reinit_n = 0, since a re-initialised layer that cannot
        train is just noise in front of the head.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        splitter: Splitters,
        seed: int = MODEL_SEED,
        reinit_n: int = 0,
        model_link: str = "DeepChem/ChemBERTa-10M-MTR",
        lr: float = 6.9e-5,
        batch_size: int = 128,
        num_epochs: int = 80,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        llrd_decay: float = 0.9,
        dropout: float | None = None,
        patience: int = 5,
        checkpoint: bool = True,
        freeze: bool = False,
    ) -> None:
        logging.set_verbosity_error()
        self.target = target
        self.splitter = splitter
        self.source = df.attrs
        self.seed = seed
        # Before the network exists, so the head's initialisation is seeded too.
        self.generator = self._seed(seed)
        self.reinit_n = reinit_n
        self.model_link = model_link
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.llrd_decay = llrd_decay
        self.dropout = dropout
        self.patience = patience
        self.checkpoint = checkpoint
        self.freeze = freeze
        self.device = DEVICE
        self.model_dir, self.checkpoint_dir = self._create_model_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(model_link)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_link, num_labels=1,
            **({} if dropout is None else {"hidden_dropout_prob": dropout}),
        ).to(self.device)
        if reinit_n > 0:
            self._reinit_layers(reinit_n)
        if freeze:
            # With no encoder parameter requiring grad, autograd builds no graph
            # through it, so a frozen epoch skips the encoder's backward pass.
            self.model.roberta.requires_grad_(False)

        self._save_run_record()
        self.train_df, self.test_df, self.valid_df = split_frames(df, self.model_dir)
        self._init_head_bias()
        self.train_loader = self._dataloader(self.train_df, shuffle=True)
        self.test_loader = self._dataloader(self.test_df, shuffle=False)
        self.valid_loader = self._dataloader(self.valid_df, shuffle=False)

        self.loss_fn = LOSS_FN()
        self.optimiser = torch.optim.AdamW(self._parameter_groups(), lr=self.lr)
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=int(self.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        self.history = []
        self.best_epoch = None  # set by fit(), read by evaluate()
        self.best_state = None  # best epoch's weights on CPU, for evaluate()
        # Leave the model in eval mode: dropout, normalise batch off by default
        self.model.eval()

    def _create_model_dir(self) -> tuple[Path, Path]:
        """
        One directory per run, named for the target, the split it was scored
        on, whether the encoder was frozen and the start time, with a
        checkpoints/ subdirectory. Nothing is overwritten between runs, so a
        sweep leaves one comparable history.csv per configuration -- and the
        split is in the name because a number from one split method is not
        comparable to a number from another.

        Returns
        -------
        tuple[Path, Path]
            The run directory and its checkpoints/ subdirectory.
        """
        model_dir = MODEL_DIR / (f"{self.target}-{self.splitter.method}-"
                                 f"{self.splitter.distribution}-"
                                 f"{'frozen' if self.freeze else 'finetuned'}-"
                                 f"{datetime.now():%Y-%m-%d-%H%M%S}")
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return model_dir, checkpoint_dir

    def _reinit_layers(self, n: int) -> None:
        """
        Re-initialise the top n encoder layers with the model's own init scheme.
        Top, not bottom: this is the Zhang et al. (2021) few-sample fine-tuning
        trick, where the layers nearest the head are the ones worth discarding.

        transformers flags every parameter loaded from a checkpoint with
        `_is_hf_initialized`, and its init functions skip flagged parameters,
        so the flag is cleared first or `_init_weights` leaves them untouched.
        The slice counts up from the bottom so that n=0 selects no layers;
        `layers[-0:]` would select all of them.

        Parameters
        ----------
        n : int
            Number of encoder layers to re-initialise, counted from the top.
        """
        layers = self.model.roberta.encoder.layer
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p._is_hf_initialized = False
            layer.apply(self.model._init_weights)
        print(f"-- Re-initialised top {n} encoder layers")

    def _init_head_bias(self) -> None:
        """
        Start the regression head predicting the training set's mean label
        instead of zero.

        The head is randomly initialised with a zero bias, so an untrained
        model predicts about 0 against labels averaging -3.9 (-ln IC50). Over
        half the initial loss is then pure intercept error, and it is that
        offset -- not the chemistry -- which dominates the gradients reaching
        the encoder during warmup, when the schedule is least able to absorb
        them. Seeding the bias makes the first update about structure, and
        leaves epoch -1 scoring near R2 = 0, which is the honest reference for
        the pre-fine-tuning latent space rather than the -1.4 a zero bias gives.
        """
        with torch.no_grad():
            self.model.classifier.out_proj.bias.fill_(self.train_df["labels"].mean())

    def _parameter_groups(self) -> list[dict]:
        """
        AdamW groups, decayed by depth (see llrd_decay). Bias and LayerNorm
        weights are excluded from weight decay as they are in every BERT recipe:
        decaying a normalisation scale just fights the normalisation.

        Returns
        -------
        list[dict]
            One decayed and one undecayed group per depth, head first, each
            with its own `lr` and `weight_decay`.
        """
        depthwise = [
            self.model.classifier,
            *reversed(self.model.roberta.encoder.layer),
            self.model.roberta.embeddings,
        ]
        groups = []
        for depth, module in enumerate(depthwise):
            named = list(module.named_parameters())
            lr = self.lr * self.llrd_decay ** depth
            groups.append({
                "params": [p for n, p in named if not any(k in n for k in NO_DECAY)],
                "lr": lr, "weight_decay": self.weight_decay,
            })
            groups.append({
                "params": [p for n, p in named if any(k in n for k in NO_DECAY)],
                "lr": lr, "weight_decay": 0.0,
            })
        return groups

    def _save_run_record(self) -> None:
        """
        Write what the run was, before it is trained, so a finished run
        directory can be analysed without the objects that built it.
        config.json holds the hyperparameters and split settings behind every
        number in history.csv. splits.parquet holds the split membership
        itself, as positional indices into the preprocessed frame, because the
        splitter's min_cluster_size filter drops molecules that the settings
        alone would not recover, and embeddings() across epochs has to see
        exactly the same molecules. split_frames() turns it back into frames.
        data_path, mutation and min_cluster_size are recorded because together
        they decide which molecules the split was drawn from: the first two
        rebuild the preprocessed frame, the last drops whole clusters from it.
        """
        config = {
            "target": self.target, "model_link": self.model_link,
            "reinit_n": self.reinit_n, "lr": self.lr,
            "batch_size": self.batch_size, "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio, "weight_decay": self.weight_decay,
            "llrd_decay": self.llrd_decay, "dropout": self.dropout,
            "patience": self.patience, "freeze": self.freeze,
            "split_method": self.splitter.method,
            "distribution": self.splitter.distribution,
            "train_size": self.splitter.train_size,
            "test_size": self.splitter.test_size,
            "split_seed": self.splitter.seed,
            "min_cluster_size": self.splitter.min_cluster_size,
            "data_path": self.source["data_path"],
            "mutation": self.source["mutation"],
            "device": str(self.device),
        }
        (self.model_dir / CONFIG).write_text(json.dumps(config, indent=2))
        pd.concat([
            pd.DataFrame({"row": index, "split": name})
            for name, index in zip(SPLIT_NAMES, self.splitter.split())
        ]).to_parquet(self.model_dir / SPLITS, index=False)

    def _seed(self, seed: int) -> torch.Generator:
        """
        Seed every generator a run draws on -- Python, numpy and torch on both
        devices -- and return the one the training DataLoader shuffles with.

        This makes a run repeatable without forcing deterministic kernels,
        which cost throughput and raise on ops that have no deterministic
        implementation. cuDNN is still free to autotune, so two runs of the
        same seed on GPU can differ in the last decimal places.

        Parameters
        ----------
        seed : int
            Seed applied to every generator.

        Returns
        -------
        torch.Generator
            Generator for the training DataLoader's shuffle.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return torch.Generator().manual_seed(seed)

    def _dataloader(self, frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        """
        Tokenise a split in one pass into a TensorDataset. The whole split is
        small enough to encode up front, which removes the per-batch collator and
        pads once to the longest sequence in the split rather than to MAX_TOKENS.
        Memory is pinned only on CUDA, where it speeds host-to-device copies;
        on CPU it does nothing but warn.

        Parameters
        ----------
        frame : pd.DataFrame
            One split, with `smiles` and `labels` columns.
        shuffle : bool
            Reshuffle every epoch; True for train only.

        Returns
        -------
        DataLoader
            Batches of (input_ids, attention_mask, labels), labels shaped (n, 1).
        """
        encoded = self.tokenizer(
            list(frame["smiles"]), padding=True, truncation=True,
            max_length=MAX_TOKENS, return_tensors="pt",
        )
        labels = torch.tensor(frame["labels"].to_numpy(), dtype=torch.float32)
        dataset = TensorDataset(
            encoded["input_ids"], encoded["attention_mask"], labels.unsqueeze(1)
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            pin_memory=self.device.type == "cuda", generator=self.generator,
        )

    def _train_epoch(self, epoch: int) -> float:
        """
        One pass over the training set. Sets train() itself so dropout state
        follows the operation rather than the order the methods happen to be
        called in. A frozen encoder is put back in eval mode: with its dropout
        on, the head would see differently noised features every epoch from
        weights that never change, and the frozen arm would no longer be a
        fixed pretrained representation with a head trained on it.

        Parameters
        ----------
        epoch : int
            Epoch number, for the progress lines only.

        Returns
        -------
        float
            Sample-weighted mean training loss over the epoch.
        """
        self.model.train()
        if self.freeze:
            self.model.roberta.eval()
        total = 0.0
        for i, (input_ids, mask, labels) in enumerate(self.train_loader):
            input_ids, mask, labels = (
                input_ids.to(self.device), mask.to(self.device), labels.to(self.device)
            )
            with autocast():
                preds = self.model(input_ids=input_ids, attention_mask=mask).logits
                loss = self.loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimiser.step()
            self.scheduler.step()  # per batch: the warmup is measured in steps
            self.optimiser.zero_grad()

            total += loss.item() * len(labels)
            if (i + 1) % LOG_EVERY == 0:
                print(f"-- Epoch: {epoch} -- Iter: {i + 1} -- Loss: {loss.item():.4f}")
        return total / len(self.train_loader.dataset)

    @torch.inference_mode()
    def _score(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Loss and predictions over the *whole* loader, not the last batch.

        Parameters
        ----------
        loader : DataLoader
            Any of the run's split loaders.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray]
            Loss, predictions and labels. The arrays are numpy so the metrics
            and any downstream plotting are off-device.
        """
        self.model.eval()
        preds, labels = [], []
        for input_ids, mask, y in loader:
            with autocast():
                out = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=mask.to(self.device),
                )
            preds.append(out.logits.float().cpu())
            labels.append(y)
        preds, labels = torch.cat(preds), torch.cat(labels)
        loss = self.loss_fn(preds, labels).item()
        return loss, preds.squeeze(1).numpy(), labels.squeeze(1).numpy()

    def _metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Score one split's predictions.

        Parameters
        ----------
        preds : np.ndarray
            Predicted labels.
        labels : np.ndarray
            True labels, in the same order.

        Returns
        -------
        dict[str, float]
            RMSE, in label units (-ln IC50), and R2, which is what the legacy
            runs reported.
        """
        return {
            "rmse": root_mean_squared_error(labels, preds),
            "r2": r2_score(labels, preds),
        }

    def _checkpoint_path(self, epoch: int) -> Path:
        """
        Where the weights for an epoch live, with epoch -1 standing for the
        weights before any training step.

        Parameters
        ----------
        epoch : int
            Epoch number, or -1 for the starting weights.

        Returns
        -------
        Path
            Checkpoint file within the run's checkpoints/ directory.
        """
        name = INIT_CHECKPOINT if epoch < 0 else EPOCH_CHECKPOINT.format(epoch)
        return self.checkpoint_dir / name

    def _save_checkpoint(self, epoch: int, loss: dict[str, float]) -> None:
        """
        Write the weights and the epoch's losses to a checkpoint. Optimiser and
        scheduler state are deliberately left out: the checkpoint is for scoring
        and figures, not for resuming a run.

        Parameters
        ----------
        epoch : int
            Epoch the weights were taken at, or -1 for the starting weights.
        loss : dict[str, float]
            Train, test and valid loss.
        """
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model": self.model.state_dict(),
        }, self._checkpoint_path(epoch))

    def load_checkpoint(self, path: Path) -> int:
        """
        Restore the weights from a checkpoint written by _save_checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint file.

        Returns
        -------
        int
            Epoch the checkpoint was saved at.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        return state["epoch"]

    def _record_epoch(self, epoch: int, train_loss: float, start: float) -> float:
        """
        Score test and valid, then log the epoch: a history row, a progress
        line and, unless the run has checkpointing off, a checkpoint.
        history.csv is rewritten every epoch so a run that crashes or is
        interrupted still leaves a history alongside its checkpoints.

        Parameters
        ----------
        epoch : int
            Epoch just finished, or -1 for the starting weights.
        train_loss : float
            Mean training loss over the epoch; NaN for epoch -1, which has
            no training pass.
        start : float
            time.time() at the start of the epoch, so the scoring pass is
            counted in `seconds`.

        Returns
        -------
        float
            Test loss, which drives model selection and early stopping.
        """
        test_loss, test_preds, test_labels = self._score(self.test_loader)
        valid_loss, valid_preds, valid_labels = self._score(self.valid_loader)
        test_metrics = self._metrics(test_preds, test_labels)
        valid_metrics = self._metrics(valid_preds, valid_labels)
        elapsed = time.time() - start
        # Both metric dicts carry the same keys, so they are prefixed
        # rather than merged -- one column per split, not the last one in.
        self.history.append({
            "epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
            "valid_loss": valid_loss, "lr": self.scheduler.get_last_lr()[0],
            "seconds": elapsed,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        })
        pd.DataFrame(self.history).to_csv(self.model_dir / HISTORY, index=False)
        print(f"-- Epoch: {epoch} -- Train: {train_loss:.4f} "
              f"-- Test: {test_loss:.4f} (R2 {test_metrics['r2']:.3f}) "
              f"-- Valid: {valid_loss:.4f} (R2 {valid_metrics['r2']:.3f}) "
              f"-- {elapsed:.1f}s")
        if self.checkpoint:
            self._save_checkpoint(
                epoch, {"train": train_loss, "test": test_loss, "valid": valid_loss}
            )
        return test_loss

    def fit(self, callback: Callable[[int, float], None] | None = None
            ) -> pd.DataFrame:
        """
        Trains until test loss stops improving -- test is the selection set
        here and valid is held back for evaluate(). Named fit() rather than
        train() so it doesn't shadow nn.Module.train(), which sets dropout mode.

        The starting weights are scored and checkpointed as epoch -1 first, so
        the latent-space comparison has its pre-fine-tuning reference (which
        differs from the published weights whenever reinit_n > 0) and the R2
        curve has its starting point. Epoch -1 is never a selection candidate.
        Every epoch is checkpointed unless the run was built with checkpoint
        off. The best epoch's weights are also kept in memory, on CPU, so
        evaluate() never depends on which epochs happened to reach disk.

        Parameters
        ----------
        callback : Callable[[int, float], None] | None
            Called with the epoch and its test loss after each training epoch,
            for a hyperparameter search to watch a fit in progress. Raising
            from it abandons the fit, which is how a search abandons a trial
            that is already trailing; the history written so far survives on
            disk. Epoch -1 is not reported, having had no training pass.

        Returns
        -------
        pd.DataFrame
            One row per epoch run, from -1 -- losses, metrics per split, LR
            and seconds -- also written to history.csv in the run directory.
        """
        print(f"""-- Training {self.model_link} on {self.target} ({self.device})
              ---- Epochs: {self.num_epochs} (patience {self.patience})
              ---- Batch size: {self.batch_size}
              ---- LR: {self.lr} (LLRD {self.llrd_decay}, warmup {self.warmup_ratio:.0%})
              ---- Loss: {self.loss_fn}
              ---- Split: {self.splitter.method} ({self.splitter.distribution}-distribution)
              ---- Reinit layers: {self.reinit_n}
              ---- Frozen encoder: {self.freeze}
              ---- Output: {self.model_dir}""")
        self._record_epoch(-1, np.nan, time.time())
        best, stale = np.inf, 0
        for epoch in range(self.num_epochs):
            start = time.time()
            train_loss = self._train_epoch(epoch)
            test_loss = self._record_epoch(epoch, train_loss, start)
            if callback is not None:
                callback(epoch, test_loss)

            if test_loss < best - MIN_DELTA:
                best, stale, self.best_epoch = test_loss, 0, epoch
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    print(f"-- Early stop: no improvement on {best:.4f} in {self.patience} epochs")
                    break
        return pd.DataFrame(self.history)

    def evaluate(self) -> dict[str, float]:
        """
        Final read of the held-out valid split, from the best epoch's
        weights rather than the last epoch's. Kept out of fit() because it
        should be run once, after any hyperparameter search is finished --
        calling it inside the loop is how the legacy code spent its held-out
        set on model selection.

        Returns
        -------
        dict[str, float]
            RMSE and R2 on the valid split. Predictions are not saved; any
            split at any epoch is recovered by a checkpoint forward pass over
            split_frames().
        """
        self.model.load_state_dict(self.best_state)
        loss, preds, labels = self._score(self.valid_loader)
        metrics = self._metrics(preds, labels)
        print(f"-- Valid (weights from epoch {self.best_epoch}) -- Loss: {loss:.4f} "
              f"-- RMSE: {metrics['rmse']:.3f} -- R2: {metrics['r2']:.3f}")
        return metrics


if __name__ == '__main__':
    from bertsheep.data import Data

    target = "EGFR"
    df = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", target)._preprocess()
    m = Model(df, target, Splitters(df["smiles"], "scaffold", "in"))
    m.fit()
    m.evaluate()
