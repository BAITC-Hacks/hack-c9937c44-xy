"""Joint two-turbine Transformer using history and issue-time weather forecasts.

The caller must enforce temporal cutoffs and forecast issue times. This module
never downloads weather, selects a holdout, or substitutes observed future data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ModelConfig:
    history_features: int
    weather_features: int
    num_turbines: int = 2
    lookback: int = 72
    horizon: int = 48
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "history_features", "weather_features", "num_turbines", "lookback", "horizon",
            "hidden_size", "num_heads", "num_layers", "epochs", "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon not in (24, 48):
            raise ValueError("horizon must be 24 or 48 hours")
        if self.hidden_size % 2 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be even and divisible by num_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


def _array(value: np.ndarray, name: str) -> np.ndarray:
    """Own contiguous float32 input and reject missing/overflowing features."""
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    with np.errstate(over="ignore"):
        array = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} cannot be represented as finite float32")
    return array


class _Standardizer:
    """Feature statistics fit exclusively to the supplied training sequences."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = mean
        self.scale = scale

    @classmethod
    def fit(cls, values: np.ndarray) -> _Standardizer:
        flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            result = ((values.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Standardized features contain nonfinite values")
        return np.ascontiguousarray(result)

    def state(self) -> dict[str, torch.Tensor]:
        # Tensor-only numerical state can be loaded with weights_only=True.
        return {"mean": torch.from_numpy(self.mean.copy()), "scale": torch.from_numpy(self.scale.copy())}

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor], features: int) -> _Standardizer:
        mean = state["mean"].cpu().numpy().astype(np.float64)
        scale = state["scale"].cpu().numpy().astype(np.float64)
        if mean.shape != (features,) or scale.shape != (features,):
            raise ValueError("Checkpoint normalization dimensions do not match ModelConfig")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Checkpoint contains invalid normalization statistics")
        return cls(mean, scale)


class _ForecastTransformer(nn.Module):
    """Non-autoregressive encoder/decoder; all future inputs are known forecasts.

    Per-turbine features are concatenated by the data pipeline, allowing shared
    attention to learn interactions. This is a Transformer baseline, not a TFT.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.history_projection = nn.Sequential(nn.Linear(config.history_features, width), nn.LayerNorm(width))
        self.weather_projection = nn.Sequential(nn.Linear(config.weather_features, width), nn.LayerNorm(width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=config.num_heads, dim_feedforward=4 * width,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=config.num_heads, dim_feedforward=4 * width,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers, norm=nn.LayerNorm(width))
        self.output = nn.Sequential(nn.Linear(width, config.num_turbines), nn.Sigmoid())
        self.dropout = nn.Dropout(config.dropout)
        self.lookback = config.lookback
        positions = torch.arange(config.lookback + config.horizon, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(1, config.lookback + config.horizon, width)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positions", encoding)

    def forward(self, history: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        history_tokens = self.dropout(self.history_projection(history) + self.positions[:, :self.lookback])
        memory = self.encoder(history_tokens)
        forecast_tokens = self.dropout(self.weather_projection(weather) + self.positions[:, self.lookback:])
        # No target values enter the decoder. The full issued forecast is known
        # at the origin, so attention among all forecast hours is permissible.
        return self.output(self.decoder(forecast_tokens, memory))


class ModelAgent:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("device must be 'cpu', 'cuda', or a CUDA device such as 'cuda:0'")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable. Install CUDA-enabled PyTorch or explicitly select device='cpu'.")
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {self.device.index} is unavailable")
        self._seed()
        self.model = _ForecastTransformer(config).to(self.device)
        self.history_scaler: _Standardizer | None = None
        self.weather_scaler: _Standardizer | None = None
        self.fitted = False

    def _seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _validate_inputs(self, history: np.ndarray, weather: np.ndarray) -> None:
        config = self.config
        if history.ndim != 3 or history.shape[1:] != (config.lookback, config.history_features):
            raise ValueError(f"history must have shape [N, {config.lookback}, {config.history_features}]")
        if weather.ndim != 3 or weather.shape[1:] != (config.horizon, config.weather_features):
            raise ValueError(f"weather must have shape [N, {config.horizon}, {config.weather_features}]")
        if history.shape[0] == 0 or history.shape[0] != weather.shape[0]:
            raise ValueError("history and weather must contain the same nonzero number of samples")

    def fit(self, history: np.ndarray, weather: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        """Refit from scratch using only caller-approved, pre-origin samples.

        There is intentionally no random validation split of overlapping windows.
        Fit fixed epochs; evaluate independently on a chronological backtest.
        Weather for EACH training sample must be a forecast issued by its origin.
        """
        history = _array(history, "history")
        weather = _array(weather, "weather")
        targets = _array(targets, "targets")
        self._validate_inputs(history, weather)
        expected = (history.shape[0], self.config.horizon, self.config.num_turbines)
        if targets.shape != expected:
            raise ValueError(f"targets must have shape {expected}")
        if ((targets < 0) | (targets > 1)).any():
            raise ValueError("targets must be normalized to [0, 1] before training")

        self.fitted = False
        self._seed()
        self.model = _ForecastTransformer(self.config).to(self.device)
        self.history_scaler = _Standardizer.fit(history)
        self.weather_scaler = _Standardizer.fit(weather)
        dataset = TensorDataset(
            torch.from_numpy(self.history_scaler.transform(history)),
            torch.from_numpy(self.weather_scaler.transform(weather)),
            torch.from_numpy(targets),
        )
        on_cuda = self.device.type == "cuda"
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0,
            pin_memory=on_cuda, generator=torch.Generator().manual_seed(self.config.seed),
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=on_cuda)
        criterion = nn.MSELoss()
        losses: list[float] = []
        for _ in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            for history_batch, weather_batch, target_batch in loader:
                history_batch = history_batch.to(self.device, non_blocking=on_cuda)
                weather_batch = weather_batch.to(self.device, non_blocking=on_cuda)
                target_batch = target_batch.to(self.device, non_blocking=on_cuda)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=on_cuda):
                    predictions = self.model(history_batch, weather_batch)
                    loss = criterion(predictions, target_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu()) * history_batch.shape[0]
            losses.append(total_loss / len(dataset))
        self.model.eval()
        self.fitted = True
        return {
            "training_mse": losses[-1], "epoch_training_mse": losses,
            "samples": len(dataset), "epochs": self.config.epochs,
            "device": str(self.device), "mixed_precision": on_cuda,
        }

    def predict(self, history: np.ndarray, weather: np.ndarray) -> np.ndarray:
        """Return normalized [H, turbines] or [B, H, turbines] predictions."""
        if not self.fitted or self.history_scaler is None or self.weather_scaler is None:
            raise RuntimeError("Call fit() or load() before predict()")
        history = _array(history, "history")
        weather = _array(weather, "weather")
        unbatched = history.ndim == 2 and weather.ndim == 2
        if unbatched:
            history, weather = history[None, ...], weather[None, ...]
        self._validate_inputs(history, weather)
        history_tensor = torch.from_numpy(self.history_scaler.transform(history))
        weather_tensor = torch.from_numpy(self.weather_scaler.transform(weather))
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(history_tensor), self.config.batch_size):
                stop = start + self.config.batch_size
                batch_history = history_tensor[start:stop].to(self.device)
                batch_weather = weather_tensor[start:stop].to(self.device)
                # Float32 inference gives consistent checkpoint round trips.
                result = self.model(batch_history, batch_weather).float().cpu().numpy()
                if not np.isfinite(result).all():
                    raise FloatingPointError("Model produced nonfinite predictions")
                outputs.append(result)
        prediction = np.concatenate(outputs, axis=0)
        return prediction[0] if unbatched else prediction

    def save(self, path: str | Path) -> None:
        """Atomically persist weights, configuration, and training-only scalers."""
        if not self.fitted or self.history_scaler is None or self.weather_scaler is None:
            raise RuntimeError("Cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1, "config": asdict(self.config),
            "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "history_scaler": self.history_scaler.state(),
            "weather_scaler": self.weather_scaler.state(),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(checkpoint, handle)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> ModelAgent:
        """Load own tensor-only checkpoint; optionally override execution device."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported model checkpoint format")
        config = dict(checkpoint["config"])
        if device is not None:
            config["device"] = device
        agent = cls(ModelConfig(**config))
        agent.history_scaler = _Standardizer.from_state(checkpoint["history_scaler"], agent.config.history_features)
        agent.weather_scaler = _Standardizer.from_state(checkpoint["weather_scaler"], agent.config.weather_features)
        state = checkpoint["model_state"]
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        agent.model.load_state_dict(state, strict=True)
        agent.model.eval()
        agent.fitted = True
        return agent
