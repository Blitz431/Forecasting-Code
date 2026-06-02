"""LSTM and GRU recurrent neural network models for Phase 4 ML Engine.

Both models accept a standard 2D feature matrix (n_samples, n_features) and
internally reshape it into fixed-length sequences for the recurrent layers.
GPU acceleration uses the 3060 Ti via CUDA when available.

Classes
-------
LSTMModel — Long Short-Term Memory regressor
GRUModel  — Gated Recurrent Unit regressor (lighter, often faster)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ml.base import MLModel, _write_sig, _verify_sig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _get_device() -> "torch.device":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.debug(f"PyTorch device: {torch.cuda.get_device_name(0)}")
    else:
        logger.debug("PyTorch device: CPU")
    return device


# ------------------------------------------------------------------ #
# Neural network architecture
# ------------------------------------------------------------------ #


class _RecurrentNet:
    """Internal PyTorch module wrapper shared by LSTM and GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str,  # "lstm" or "gru"
    ) -> None:
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
                self_.rnn = rnn_cls(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self_.dropout = nn.Dropout(dropout)
                self_.fc = nn.Linear(hidden_size, 1)

            def forward(self_, x):
                # x: (batch, seq_len, input_size)
                out, _ = self_.rnn(x)
                out = self_.dropout(out[:, -1, :])   # last time step
                return self_.fc(out).squeeze(-1)

        self.net = _Net()

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


# ------------------------------------------------------------------ #
# Shared training/prediction logic
# ------------------------------------------------------------------ #


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Reshape 2D (n, features) into 3D (n-seq_len+1, seq_len, features)."""
    n, f = X.shape
    if n < seq_len:
        raise ValueError(f"Not enough samples ({n}) for seq_len={seq_len}")
    seqs = np.stack([X[i: i + seq_len] for i in range(n - seq_len + 1)], axis=0)
    return seqs.astype(np.float32)


class _BaseRecurrentModel(MLModel):
    """Shared training loop for LSTM and GRU."""

    def __init__(
        self,
        cell_type: str,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
    ) -> None:
        self.cell_type = cell_type
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self._net: _RecurrentNet | None = None
        self._device = None
        self._input_size: int | None = None
        self._scaler_X = None
        self._scaler_y = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        device = _get_device()
        self._device = device

        # Scale features and target
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_sc = self._scaler_X.fit_transform(X).astype(np.float32)
        y_sc = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        # Build sequences
        X_seq = _make_sequences(X_sc, self.seq_len)    # (n-sl+1, sl, features)
        y_seq = y_sc[self.seq_len - 1:]                 # aligned targets

        n_samples, sl, n_features = X_seq.shape
        self._input_size = n_features

        # Build model
        self._net = _RecurrentNet(n_features, self.hidden_size, self.num_layers,
                                  self.dropout, self.cell_type).to(device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(X_seq).to(device)
        y_t = torch.from_numpy(y_seq).to(device)

        # Train/val split (last 10%)
        val_size = max(1, int(n_samples * 0.1))
        X_tr, X_val = X_t[:-val_size], X_t[-val_size:]
        y_tr, y_val = y_t[:-val_size], y_t[-val_size:]

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._net.train_mode()
            # Mini-batch SGD
            perm = torch.randperm(len(X_tr))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i: i + self.batch_size]
                xb, yb = X_tr[idx], y_tr[idx]
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._net.eval_mode()
            with torch.no_grad():
                val_pred = self._net(X_val)
                val_loss = loss_fn(val_pred, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug(
                        f"{self.name} early stopping at epoch {epoch + 1} "
                        f"(val_loss={best_val_loss:.6f})"
                    )
                    break

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        logger.debug(
            f"{self.name} trained: {epoch + 1} epochs, "
            f"best_val_loss={best_val_loss:.6f}, device={device}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError(f"{self.name}.train() must be called before predict()")

        X_sc = self._scaler_X.transform(X).astype(np.float32)
        X_seq = _make_sequences(X_sc, self.seq_len)

        self._net.eval_mode()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            preds_sc = self._net(X_t).cpu().numpy()

        # Inverse-scale predictions
        preds = self._scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()

        # Pad the first (seq_len-1) predictions with NaN so output length == input length
        pad = np.full(self.seq_len - 1, np.nan)
        return np.concatenate([pad, preds]).astype(np.float64)

    def save(self, path: Path) -> None:
        import torch

        if self._net is None:
            raise RuntimeError("Model not trained")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": {
                    "cell_type": self.cell_type,
                    "seq_len": self.seq_len,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "input_size": self._input_size,
                },
                "scaler_X": self._scaler_X,
                "scaler_y": self._scaler_y,
            },
            str(path),
        )
        _write_sig(path)
        logger.debug(f"{self.name} saved to {path}")


# ------------------------------------------------------------------ #
# Public model classes
# ------------------------------------------------------------------ #


class LSTMModel(_BaseRecurrentModel):
    """LSTM regressor with GPU acceleration.

    Args:
        seq_len: Number of time steps in each input sequence (default 20 ≈ 1 month).
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between layers (only applied if num_layers > 1).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        patience: Early stopping patience (epochs without val improvement).
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> None:
        super().__init__(
            cell_type="lstm",
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )

    @property
    def name(self) -> str:
        return "LSTM"


class GRUModel(_BaseRecurrentModel):
    """GRU regressor with GPU acceleration.

    Lighter than LSTM (fewer parameters) and often converges faster while
    matching or exceeding LSTM accuracy on financial time series.

    Args:
        seq_len: Number of time steps in each input sequence.
        hidden_size: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between layers.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        patience: Early stopping patience.
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> None:
        super().__init__(
            cell_type="gru",
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )

    @property
    def name(self) -> str:
        return "GRU"
