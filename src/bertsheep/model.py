import time
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
CHECKPOINT = "best.pt"
HISTORY = "history.csv"
PREDICTIONS = "valid_predictions.csv"

# Tokeniser truncation length. Data.MAX_SMILES_LENGTH caps SMILES at 128
# *characters*, which is a looser bound than 128 tokens, so truncation is rare.
MAX_TOKENS = 128

MAX_GRAD_NORM = 1.0
NO_DECAY = ("bias", "LayerNorm.weight")

MIN_DELTA = 0.0  # improvement in test loss that resets early stopping
LOG_EVERY = 5  # batches between training-loss lines

LOSS_FN = torch.nn.MSELoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bf16 rather than fp16: it keeps fp32's exponent range, so no loss scaling.
# Autocast keeps the weights and optimiser state in fp32; only the forward
# pass runs in bf16, and only on CUDA.
AUTOCAST = True


class Model():
    """
    Fine-tunes ChemBERTa as a regressor on a preprocessed (smiles, labels) frame.
    Hyperparameters are constructor arguments rather than module constants so a
    sweep can build one Model per trial without mutating shared state.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns.
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
    patience : int
        Epochs without test-loss improvement before early stopping.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        splitter: Splitters,
        reinit_n: int = 0,
        model_link: str = "DeepChem/ChemBERTa-10M-MTR",
        lr: float = 6.9e-5,
        batch_size: int = 128,
        num_epochs: int = 80,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        llrd_decay: float = 0.9,
        patience: int = 5,
    ) -> None:
        logging.set_verbosity_error()
        self.target = target
        self.splitter = splitter
        self.reinit_n = reinit_n
        self.model_link = model_link
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.llrd_decay = llrd_decay
        self.patience = patience
        self.device = DEVICE
        self.model_dir, self.checkpoint_dir = self._create_model_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(model_link)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_link, num_labels=1
        ).to(self.device)
        if reinit_n > 0:
            self._reinit_layers(reinit_n)

        self.train_df, self.test_df, self.valid_df = self._split(df)
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
        # Leave the model in eval mode: dropout, normalise batch off by default
        self.model.eval()

    def _create_model_dir(self) -> tuple[Path, Path]:
        """
        One directory per run, named for the target, the split it was scored
        on and the start time, with a checkpoints/ subdirectory. Nothing is
        overwritten between runs, so a sweep leaves one comparable history.csv
        per configuration -- and the split is in the name because a number from
        one split method is not comparable to a number from another.

        Returns
        -------
        tuple[Path, Path]
            The run directory and its checkpoints/ subdirectory.
        """
        model_dir = MODEL_DIR / (f"{self.target}-{self.splitter.method}-"
                                 f"{self.splitter.distribution}-"
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

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Cut the frame three ways with the run's splitter.

        Splitters deals positional indices, so the frame is taken with iloc --
        a preprocessed frame carries the raw dump's row numbers as its index,
        which .loc would read as labels.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed frame with `smiles` and `labels` columns, in the order
            the splitter was built over.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, test and valid frames.
        """
        train, test, valid = (df.iloc[index] for index in self.splitter.split())
        return train, test, valid

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
            pin_memory=self.device.type == "cuda",
        )

    def _autocast(self) -> torch.autocast:
        """
        Mixed-precision context for a forward pass, so training and scoring
        share one precision policy and the reported metrics come from the same
        arithmetic the weights were trained with.

        Returns
        -------
        torch.autocast
            bf16 autocast on CUDA when `AUTOCAST` is set, otherwise a no-op.
        """
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16,
            enabled=AUTOCAST and self.device.type == "cuda",
        )

    def _train_epoch(self, epoch: int) -> float:
        """
        One pass over the training set. Sets train() itself so dropout state
        follows the operation rather than the order the methods happen to be
        called in.

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
        total = 0.0
        for i, (input_ids, mask, labels) in enumerate(self.train_loader):
            input_ids, mask, labels = (
                input_ids.to(self.device), mask.to(self.device), labels.to(self.device)
            )
            with self._autocast():
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
            with self._autocast():
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

    def _save_checkpoint(self, name: str, epoch: int,
                         loss: dict[str, float]) -> None:
        """
        Write the weights and the epoch's losses to a checkpoint. Optimiser and
        scheduler state are deliberately left out: the checkpoint is for scoring
        and figures, not for resuming a run.

        Parameters
        ----------
        name : str
            Filename within the run's checkpoints/ directory. The caller names
            it, so fit() can keep a per-epoch history and overwrite one
            best-so-far file from the same method.
        epoch : int
            Epoch the weights were taken at.
        loss : dict[str, float]
            Train, test and valid loss.
        """
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model": self.model.state_dict(),
        }, self.checkpoint_dir / name)

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

    def fit(self) -> pd.DataFrame:
        """
        Trains until test loss stops improving -- test is the selection set
        here and valid is held back for evaluate(). Named fit() rather than
        train() so it doesn't shadow nn.Module.train(), which sets dropout mode.
        One improvement check drives both the checkpoint and the early stopper,
        so the saved weights are always the ones the stopper stopped on.

        Returns
        -------
        pd.DataFrame
            One row per epoch run -- losses, metrics per split, LR and seconds
            -- also written to history.csv in the run directory.
        """
        print(f"""-- Training {self.model_link} on {self.target} ({self.device})
              ---- Epochs: {self.num_epochs} (patience {self.patience})
              ---- Batch size: {self.batch_size}
              ---- LR: {self.lr} (LLRD {self.llrd_decay}, warmup {self.warmup_ratio:.0%})
              ---- Loss: {self.loss_fn}
              ---- Split: {self.splitter.method} ({self.splitter.distribution}-distribution)
              ---- Reinit layers: {self.reinit_n}
              ---- Output: {self.model_dir}""")
        best, stale = np.inf, 0
        for epoch in range(self.num_epochs):
            start = time.time()
            train_loss = self._train_epoch(epoch)
            test_loss, test_preds, test_labels = self._score(self.test_loader)
            valid_loss, valid_preds, valid_labels = self._score(self.valid_loader)
            loss = {'train': train_loss, 'test': test_loss, 'valid': valid_loss}
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
            print(f"-- Epoch: {epoch} -- Train: {train_loss:.4f} "
                  f"-- Test: {test_loss:.4f} (R2 {test_metrics['r2']:.3f}) "
                  f"-- Valid: {valid_loss:.4f} (R2 {valid_metrics['r2']:.3f}) "
                  f"-- {elapsed:.1f}s")

            self._save_checkpoint(f'epoch{epoch:03d}.pt', epoch, loss)

            if test_loss < best - MIN_DELTA:
                best, stale = test_loss, 0
                self._save_checkpoint('best.pt', epoch, loss)
            else:
                stale += 1
                if stale >= self.patience:
                    print(f"-- Early stop: no improvement on {best:.4f} in {self.patience} epochs")
                    break

        history = pd.DataFrame(self.history)
        history.to_csv(self.model_dir / HISTORY, index=False)
        return history

    def evaluate(self) -> dict[str, float]:
        """
        Final read of the held-out valid split, from the best checkpoint rather
        than the last epoch. Kept out of fit() because it should be run once,
        after any hyperparameter search is finished -- calling it inside the loop
        is how the legacy code spent its held-out set on model selection.

        Returns
        -------
        dict[str, float]
            RMSE and R2 on the valid split; the predictions behind them are
            written to valid_predictions.csv in the run directory.
        """
        epoch = self.load_checkpoint(self.checkpoint_dir / CHECKPOINT)
        loss, preds, labels = self._score(self.valid_loader)
        metrics = self._metrics(preds, labels)
        print(f"-- Valid (checkpoint from epoch {epoch}) -- Loss: {loss:.4f} "
              f"-- RMSE: {metrics['rmse']:.3f} -- R2: {metrics['r2']:.3f}")
        pd.DataFrame({"labels": labels, "preds": preds}).to_csv(
            self.model_dir / PREDICTIONS, index=False
        )
        return metrics


if __name__ == '__main__':
    from bertsheep.data import Data

    target = "EGFR"
    df = Data("data/BindingDB_All_202609_tsv/BindingDB_All.tsv", target)._preprocess()
    m = Model(df, target, Splitters(df["smiles"], "scaffold", "in"))
    m.fit()
    m.evaluate()
