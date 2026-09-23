"""Joint two-turbine Transformer using history and issue-time weather forecasts.

Public train_model/predict wrappers check temporal provenance before execution.
The lower-level ModelAgent API assumes those checks were performed by its caller.
This module never downloads weather or substitutes observed future data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Sequence

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
    scaler_backend: str = "auto"

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
        if self.scaler_backend not in {"auto", "cuml", "numpy"}:
            raise ValueError("scaler_backend must be 'auto', 'cuml', or 'numpy'")


@dataclass(frozen=True)
class TrainingData:
    """Dense hourly windows with provenance supplied by the data agent.

    history[i] is ordered from origins[i]-lookback through origins[i]-1h;
    weather[i] and targets[i] start at origins[i]. Availability is the latest
    upper bound across all turbines/features, per [sample, forecast hour].
    All metadata timestamps must be timezone-aware datetime objects.
    """

    history: np.ndarray
    weather: np.ndarray
    targets: np.ndarray
    origins: Sequence[datetime]
    current_date: datetime
    available_at_upper_bound: np.ndarray


@dataclass(frozen=True)
class ForecastData:
    """One dense hourly history window and its issued [H, W] weather forecast.

    The final history row describes context_origin-1h (current_date-1h when
    omitted). A stale context_origin can be used when recent SCADA is missing.
    Weather always starts at current_date; availability bounds have H entries.
    """

    history: np.ndarray
    weather: np.ndarray
    current_date: datetime
    available_at_upper_bound: Sequence[datetime]
    context_origin: datetime | None = None


def _utc_timestamp(value: datetime, name: str, *, hourly: bool = False) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    result = value.astimezone(timezone.utc)
    if hourly and (result.minute or result.second or result.microsecond):
        raise ValueError(f"{name} must be aligned to a UTC hour")
    return result


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

    def __init__(
        self, mean: np.ndarray, scale: np.ndarray,
        backend: str = "numpy", device_index: int | None = None,
    ) -> None:
        self.mean = mean
        self.scale = scale
        self.backend = backend
        self.device_index = device_index

    @classmethod
    def fit(
        cls, values: np.ndarray, backend: str = "numpy", device_index: int | None = None,
    ) -> _Standardizer:
        flattened = values.reshape(-1, values.shape[-1])
        if backend == "cuml":
            try:
                import cupy as cp
                from cuml.preprocessing import StandardScaler
            except ImportError as error:
                raise RuntimeError(
                    "CUDA scaling requires CuPy and RAPIDS cuML. Install matching RAPIDS/CUDA "
                    "packages or explicitly set scaler_backend='numpy'."
                ) from error
            with cp.cuda.Device(device_index):
                scaler = StandardScaler().fit(cp.asarray(flattened, dtype=cp.float64))
                mean = cp.asnumpy(cp.asarray(scaler.mean_)).astype(np.float64)
                scale = cp.asnumpy(cp.asarray(scaler.scale_)).astype(np.float64)
        else:
            flattened = flattened.astype(np.float64)
            mean = flattened.mean(axis=0)
            scale = flattened.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale, backend, device_index)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.backend == "cuml":
            try:
                import cupy as cp
            except ImportError as error:
                raise RuntimeError("CUDA normalization requires CuPy") from error
            with cp.cuda.Device(self.device_index):
                result = cp.asnumpy(
                    ((cp.asarray(values, dtype=cp.float64) - cp.asarray(self.mean)) / cp.asarray(self.scale))
                    .astype(cp.float32)
                )
        else:
            with np.errstate(over="ignore"):
                result = ((values.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Standardized features contain nonfinite values")
        return np.ascontiguousarray(result)

    def state(self) -> dict[str, torch.Tensor]:
        # Tensor-only numerical state can be loaded with weights_only=True.
        return {"mean": torch.from_numpy(self.mean.copy()), "scale": torch.from_numpy(self.scale.copy())}

    @classmethod
    def from_state(
        cls, state: dict[str, torch.Tensor], features: int,
        backend: str = "numpy", device_index: int | None = None,
    ) -> _Standardizer:
        mean = state["mean"].cpu().numpy().astype(np.float64)
        scale = state["scale"].cpu().numpy().astype(np.float64)
        if mean.shape != (features,) or scale.shape != (features,):
            raise ValueError("Checkpoint normalization dimensions do not match ModelConfig")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Checkpoint contains invalid normalization statistics")
        return cls(mean, scale, backend, device_index)


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
        self.scaler_backend = (
            "cuml" if self.device.type == "cuda" else "numpy"
        ) if config.scaler_backend == "auto" else config.scaler_backend
        if self.scaler_backend == "cuml" and self.device.type != "cuda":
            raise ValueError("scaler_backend='cuml' requires a CUDA device")
        self._seed()
        self.model = _ForecastTransformer(config).to(self.device)
        self.history_scaler: _Standardizer | None = None
        self.weather_scaler: _Standardizer | None = None
        self.fitted = False
        self.training_metrics: dict[str, Any] = {}
        self.training_cutoff: datetime | None = None

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
        self.training_cutoff = None
        self.training_metrics = {}
        self._seed()
        self.model = _ForecastTransformer(self.config).to(self.device)
        self.history_scaler = _Standardizer.fit(history, self.scaler_backend, self.device.index)
        self.weather_scaler = _Standardizer.fit(weather, self.scaler_backend, self.device.index)
        # cuML fits feature statistics and CuPy scales on GPU. Host tensors remain
        # the DataLoader boundary, with pinned batches copied to CUDA as needed.
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
        self.training_metrics = {
            "training_mse": losses[-1], "epoch_training_mse": losses,
            "samples": len(dataset), "epochs": self.config.epochs,
            "device": str(self.device), "mixed_precision": on_cuda,
            "scaler_backend": self.scaler_backend,
        }
        return dict(self.training_metrics)

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
            "training_cutoff": self.training_cutoff.isoformat() if self.training_cutoff is not None else None,
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
            if torch.device(device).type == "cpu" and config.get("scaler_backend") == "cuml":
                config["scaler_backend"] = "numpy"
        agent = cls(ModelConfig(**config))
        agent.history_scaler = _Standardizer.from_state(
            checkpoint["history_scaler"], agent.config.history_features, agent.scaler_backend, agent.device.index,
        )
        agent.weather_scaler = _Standardizer.from_state(
            checkpoint["weather_scaler"], agent.config.weather_features, agent.scaler_backend, agent.device.index,
        )
        state = checkpoint["model_state"]
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        agent.model.load_state_dict(state, strict=True)
        agent.model.eval()
        agent.fitted = True
        if checkpoint.get("training_cutoff") is not None:
            agent.training_cutoff = _utc_timestamp(
                datetime.fromisoformat(checkpoint["training_cutoff"]), "checkpoint training_cutoff", hourly=True,
            )
        return agent


def train_model(train_data: TrainingData, config: ModelConfig | None = None) -> ModelAgent:
    """Train on completed pre-cutoff targets and forecasts available at each origin.

    Defaults use CUDA and cuML. Supply ModelConfig(device='cpu', ...) explicitly
    for development. The data agent is responsible for the documented ordering
    of dense history/target rows and genuine weather-forecast provenance.
    """
    cutoff = _utc_timestamp(train_data.current_date, "current_date", hourly=True)
    history = np.asarray(train_data.history)
    weather = np.asarray(train_data.weather)
    targets = np.asarray(train_data.targets)
    if history.ndim != 3 or weather.ndim != 3 or targets.ndim != 3:
        raise ValueError("Training history, weather, and targets must each be three-dimensional")
    samples, horizon = weather.shape[:2]
    if samples == 0 or history.shape[0] != samples or targets.shape[:2] != (samples, horizon):
        raise ValueError("Training history, weather, and targets must have matching nonzero samples/horizon")
    if len(train_data.origins) != samples:
        raise ValueError("origins must contain one timestamp per training sample")
    available = np.asarray(train_data.available_at_upper_bound, dtype=object)
    if available.shape != (samples, horizon):
        raise ValueError("available_at_upper_bound must have shape [N, H]")
    for sample, value in enumerate(train_data.origins):
        origin = _utc_timestamp(value, f"origins[{sample}]", hourly=True)
        # A target stamped origin+H-1h is complete only at origin+H.
        if origin + timedelta(hours=horizon) > cutoff:
            raise ValueError("Training targets must be strictly before current_date and fully observed")
        for lead, bound in enumerate(available[sample]):
            if _utc_timestamp(bound, f"available_at_upper_bound[{sample},{lead}]") > origin:
                raise ValueError("Training weather forecast was not available at its sample origin")
    if config is None:
        config = ModelConfig(
            history_features=history.shape[2], weather_features=weather.shape[2],
            num_turbines=targets.shape[2], lookback=history.shape[1], horizon=horizon,
        )
    agent = ModelAgent(config)
    agent.fit(history, weather, targets)
    agent.training_cutoff = cutoff
    return agent


def predict(model: ModelAgent, forecast_data: ForecastData) -> np.ndarray:
    """Return [H, turbines] power, rejecting unavailable weather/backdated models."""
    current = _utc_timestamp(forecast_data.current_date, "current_date", hourly=True)
    context = current if forecast_data.context_origin is None else _utc_timestamp(
        forecast_data.context_origin, "context_origin", hourly=True,
    )
    if context > current:
        raise ValueError("History context_origin cannot be later than current_date")
    if model.training_cutoff is None:
        raise ValueError("Model training provenance is missing; use train_model() before public predict()")
    if model.training_cutoff > current:
        raise ValueError("Model was trained after the prediction current_date")
    history = np.asarray(forecast_data.history)
    weather = np.asarray(forecast_data.weather)
    if history.ndim != 2 or weather.ndim != 2:
        raise ValueError("ForecastData must contain one two-dimensional history/weather window")
    available = np.asarray(forecast_data.available_at_upper_bound, dtype=object)
    if available.shape != (model.config.horizon,):
        raise ValueError("available_at_upper_bound must contain H timestamps")
    for lead, bound in enumerate(available):
        if _utc_timestamp(bound, f"available_at_upper_bound[{lead}]") > current:
            raise ValueError("Weather forecast was not available at current_date")
    return model.predict(history, weather)
