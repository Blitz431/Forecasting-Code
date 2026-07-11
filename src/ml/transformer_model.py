from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.ml.base import MLModel, _write_sig, _verify_sig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Transformer-based time series regressor (TFT-style) — multi-head self-attention, GPU-accelerated for Phase 4.

Connections:
  - src/ml/base.py: subclasses MLModel, uses _write_sig/_verify_sig for model integrity
  - src/ml/runner.py: instantiated when deep_learning=True in run_ticker()

In:  (X_train, y_train) numpy arrays (reshaped to sequences internally)
Out: MLResult with predictions and RMSE/MAE/MAPE; PyTorch .pt model saved with HMAC signature
"""


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
# Architecture
# ------------------------------------------------------------------ #


def _build_transformer_net(input_size: int, d_model: int, nhead: int,
                            num_layers: int, dropout: float, dim_feedforward: int):
    """Build and return a PyTorch transformer encoder + regression head."""
    import torch
    import torch.nn as nn
    import math

    class _PositionalEncoding(nn.Module):
        def __init__(self_, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self_.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self_.register_buffer("pe", pe)

        def forward(self_, x):
            x = x + self_.pe[:, : x.size(1), :]
            return self_.dropout(x)

    class _TransformerNet(nn.Module):
        def __init__(self_):
            super().__init__()
            self_.input_proj = nn.Linear(input_size, d_model)
            self_.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability
            )
            self_.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self_.fc = nn.Linear(d_model, 1)

        def forward(self_, x):
            # x: (batch, seq_len, input_size)
            x = self_.input_proj(x)
            x = self_.pos_enc(x)
            x = self_.encoder(x)
            x = x.mean(dim=1)   # mean pool over time
            return self_.fc(x).squeeze(-1)

    return _TransformerNet()


# ------------------------------------------------------------------ #
# Public model class
# ------------------------------------------------------------------ #


class TransformerModel(MLModel):
    """Transformer encoder regressor for financial time series.

    Args:
        seq_len: Sequence length (number of days to look back).
        d_model: Internal embedding dimension. Must be divisible by nhead.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of the position-wise FFN sublayer.
        dropout: Dropout probability in attention and FFN layers.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        patience: Early stopping patience.
    """

    def __init__(
        self,
        seq_len: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> None:
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self._net = None
        self._device = None
        self._input_size: int | None = None
        self._scaler_X = None
        self._scaler_y = None

    @property
    def name(self) -> str:
        return "Transformer"

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        n, f = X.shape
        if n < self.seq_len:
            raise ValueError(f"[{self.name}] Need ≥{self.seq_len} samples, got {n}")
        return np.stack([X[i: i + self.seq_len] for i in range(n - self.seq_len + 1)],
                        axis=0).astype(np.float32)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        device = _get_device()
        self._device = device
        n_features = X.shape[1]
        self._input_size = n_features

        # Ensure d_model is divisible by nhead
        d_model = self.d_model
        if d_model % self.nhead != 0:
            d_model = (d_model // self.nhead + 1) * self.nhead
            logger.debug(f"Adjusted d_model to {d_model} (divisible by nhead={self.nhead})")

        # Scale
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_sc = self._scaler_X.fit_transform(X).astype(np.float32)
        y_sc = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        # Sequences
        X_seq = self._make_sequences(X_sc)        # (n-sl+1, sl, features)
        y_seq = y_sc[self.seq_len - 1:]

        # Build model
        self._net = _build_transformer_net(
            n_features, d_model, self.nhead, self.num_layers,
            self.dropout, self.dim_feedforward
        ).to(device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        loss_fn = nn.HuberLoss(delta=0.1)

        X_t = torch.from_numpy(X_seq).to(device)
        y_t = torch.from_numpy(y_seq).to(device)

        val_size = max(1, int(len(X_t) * 0.1))
        X_tr, X_val = X_t[:-val_size], X_t[-val_size:]
        y_tr, y_val = y_t[:-val_size], y_t[-val_size:]

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i: i + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._net(X_tr[idx]), y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._net(X_val), y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug(
                        f"Transformer early stopping at epoch {epoch + 1} "
                        f"(val_loss={best_val_loss:.6f})"
                    )
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        logger.debug(
            f"Transformer trained: {epoch + 1} epochs, "
            f"best_val_loss={best_val_loss:.6f}, device={device}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("TransformerModel.train() must be called before predict()")

        X_sc = self._scaler_X.transform(X).astype(np.float32)
        X_seq = self._make_sequences(X_sc)

        self._net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(self._device)
            preds_sc = self._net(X_t).cpu().numpy()

        preds = self._scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()

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
                    "seq_len": self.seq_len,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "input_size": self._input_size,
                },
                "scaler_X": self._scaler_X,
                "scaler_y": self._scaler_y,
            },
            str(path),
        )
        _write_sig(path)
        logger.debug(f"Transformer saved to {path}")
