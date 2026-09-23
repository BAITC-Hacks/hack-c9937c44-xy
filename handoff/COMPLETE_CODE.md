# Complete WPF Python source

Snapshot of the working tree. See README.md and HANDOFF.md for setup and data limitations.

## data_agent.py

```python
"""Causal hourly SCADA features and archived, fixed-lead GFS forecasts.

The stitched Historical Forecast endpoint is deliberately not used: its latest
values can contain runs issued after a simulated forecast origin. Previous Runs
provides 24*N-hour lead products. Their inferred issue/availability bounds are
recorded explicitly; they are not claimed to be exact model run timestamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_FILENAMES = (
    "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 1.csv",
    "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv",
)
HISTORY_FEATURES = (
    "power", "wind_speed", "temperature", "hour_sin", "hour_cos", "doy_sin",
    "doy_cos", "power_roll_6", "power_roll_24", "wind_roll_6",
)
WEATHER_FEATURES = (
    "wind_speed", "temperature", "pressure", "air_density", "hour_sin",
    "hour_cos", "doy_sin", "doy_cos",
)
CSV_COLUMNS = {
    "Статистическое время": "timestamp",
    "Средняя скорость ветра(m/s)": "wind_speed",
    "Нормализованная активная мощность": "power",
    "Средняя температура окружающей среды(°C)": "temperature",
}
FORECAST_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "gfs_global"
FORECAST_DOCUMENTATION = "https://open-meteo.com/en/docs/previous-runs-api"
_WEATHER_VARIABLES = {
    "wind_speed": ("wind_speed_10m", "m/s"),
    "temperature": ("temperature_2m", "°C"),
    "pressure": ("surface_pressure", "hPa"),
}


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("current_date must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _add_cycles(frame: Any, index: pd.DatetimeIndex, xp: Any = np) -> None:
    """Use UTC clock features consistently in training and prediction."""
    hours = xp.asarray(index.hour.to_numpy(), dtype="float64")
    days = xp.asarray(index.dayofyear.to_numpy() - 1, dtype="float64")
    year_lengths = xp.asarray(np.where(index.is_leap_year, 366.0, 365.0))
    frame["hour_sin"] = xp.sin(2 * xp.pi * hours / 24.0)
    frame["hour_cos"] = xp.cos(2 * xp.pi * hours / 24.0)
    frame["doy_sin"] = xp.sin(2 * xp.pi * days / year_lengths)
    frame["doy_cos"] = xp.cos(2 * xp.pi * days / year_lengths)


class DataAgent:
    """GPU processing by default; pandas is an explicit CPU/test backend.

    CSV timestamps are interpreted in data_timezone, then converted to UTC.
    A raw timestamp labels the beginning of its ten-minute measurement interval.
    Hourly means require six unique, valid readings; missing bins remain missing.
    The caller must confirm that timestamp convention against the data owner.
    """

    def __init__(
        self,
        backend: str = "cudf",
        data_timezone: str = "UTC",
        cache_dir: Path | str = Path("data/weather"),
        publication_lag_hours: float = 8,
    ) -> None:
        if backend not in {"cudf", "pandas"}:
            raise ValueError("backend must be 'cudf' or 'pandas'")
        if not math.isfinite(publication_lag_hours) or publication_lag_hours < 0:
            raise ValueError("publication_lag_hours must be finite and nonnegative")
        # Validate the explicitly selected source timezone immediately.
        pd.Timestamp("2026-01-01").tz_localize(data_timezone)
        self.backend = backend
        self.data_timezone = data_timezone
        self.cache_dir = Path(cache_dir)
        self.publication_lag_hours = float(publication_lag_hours)
        self._http_session: Any = None

    def _engine(self) -> tuple[Any, Any]:
        if self.backend == "pandas":
            return pd, np
        try:
            import cudf
            import cupy
        except ImportError as exc:
            raise RuntimeError(
                "The cudf backend requires NVIDIA RAPIDS on Linux/WSL2 with CUDA; "
                "use backend='pandas' explicitly for CPU smoke tests."
            ) from exc
        return cudf, cupy

    def load_history(self, csv_path: Path | str, current_date: Any) -> pd.DataFrame:
        """Read only observations available strictly before the forecast origin.

        Grouping, completeness checks, means and causal rolling run on the chosen
        backend. IANA timezone normalization occurs on the CPU at the boundary.
        Nothing fills gaps or incorporates the target hour into a past feature.
        """
        cutoff = _utc_timestamp(current_date)
        engine, xp = self._engine()
        raw = engine.read_csv(
            str(csv_path), usecols=list(CSV_COLUMNS),
            dtype={column: "str" for column in CSV_COLUMNS},
        ).rename(columns=CSV_COLUMNS)
        source_times = raw["timestamp"].to_pandas() if self.backend == "cudf" else raw["timestamp"]
        times = pd.DatetimeIndex(pd.to_datetime(source_times, format="mixed", errors="raise"))
        if times.hasnans:
            raise ValueError("CSV contains missing timestamps")
        if times.tz is None:
            times = times.tz_localize(self.data_timezone, ambiguous="raise", nonexistent="raise")
        utc_times = times.tz_convert("UTC").tz_localize(None)
        raw["timestamp"] = engine.Series(utc_times.to_numpy(dtype="datetime64[ns]"))
        cutoff_naive = cutoff.tz_localize(None)
        # Filter BEFORE numeric conversion, aggregation, rolling or statistics.
        raw = raw.loc[raw["timestamp"] < cutoff_naive.to_datetime64()].copy()
        if raw.empty:
            raise ValueError("No historical observations precede current_date")
        values = ["power", "wind_speed", "temperature"]
        for column in values:
            raw[column] = engine.to_numeric(raw[column], errors="coerce").astype("float64")
            finite = (raw[column] != float("inf")) & (raw[column] != float("-inf"))
            raw[column] = raw[column].where(finite, np.nan)
        # Exact duplicates are harmless; conflicting readings must be resolved.
        raw = raw.drop_duplicates(subset=["timestamp", *values])
        if bool(raw["timestamp"].duplicated().any()):
            raise ValueError("Conflicting duplicate SCADA timestamps")
        stamps = raw["timestamp"].astype("int64")
        if bool((stamps % (10 * 60 * 1_000_000_000) != 0).any()):
            raise ValueError("SCADA timestamps must lie on a 10-minute grid")
        raw["hour"] = raw["timestamp"].dt.floor("h")
        grouped = raw.groupby("hour", sort=True)
        hourly = grouped[values].mean()
        counts = grouped[values].count()
        complete = (grouped.size() == 6) & (counts.min(axis=1) == 6)
        for column in values:
            hourly[column] = hourly[column].where(complete, np.nan)
        first_hour = pd.Timestamp(raw["hour"].min())
        last_hour = cutoff_naive.floor("h") - pd.Timedelta(hours=1)
        if first_hour > last_hour:
            raise ValueError("No fully elapsed historical hour precedes current_date")
        full_index = pd.date_range(first_hour, last_hour, freq="h", name="timestamp")
        hourly = hourly.reindex(engine.Index(full_index))
        # Reindex first: row windows now represent elapsed hours across gaps.
        hourly["power_roll_6"] = hourly["power"].rolling(6, min_periods=6).mean()
        hourly["power_roll_24"] = hourly["power"].rolling(24, min_periods=24).mean()
        hourly["wind_roll_6"] = hourly["wind_speed"].rolling(6, min_periods=6).mean()
        _add_cycles(hourly, full_index, xp)
        result = hourly[list(HISTORY_FEATURES)]
        if self.backend == "cudf":
            result = result.to_pandas()
        result.index = pd.DatetimeIndex(result.index).tz_localize("UTC").rename("timestamp")
        result.attrs.update(
            cutoff=cutoff.isoformat(), data_timezone=self.data_timezone,
            backend=self.backend, aggregation="six unique 10-minute samples per left-labeled hour",
        )
        return result

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError as exc:
            raise RuntimeError("Install requests to fetch archived forecasts") from exc
        if self._http_session is None:
            self._http_session = requests.Session()
            retry = Retry(
                total=3, connect=3, read=3, backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
            )
            self._http_session.mount("https://", HTTPAdapter(max_retries=retry))
        response = self._http_session.get(FORECAST_ENDPOINT, params=params, timeout=(10, 60))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("error"):
            raise ValueError("Archived forecast API returned an error or unexpected schema")
        return payload

    def _decode_forecast(
        self, payload: dict[str, Any], index: pd.DatetimeIndex,
        lead_days: np.ndarray, cutoff: pd.Timestamp,
    ) -> pd.DataFrame:
        if not isinstance(payload, dict) or payload.get("error") or payload.get("utc_offset_seconds") != 0:
            raise ValueError("Forecast response must be successful and use UTC")
        try:
            hourly, units = payload["hourly"], payload["hourly_units"]
            times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True, errors="raise"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid hourly forecast schema or timestamps") from exc
        if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Forecast times must be valid, unique and increasing")
        positions = times.get_indexer(index)
        if (positions < 0).any():
            raise ValueError("Archived forecast does not cover every requested hour")
        result = pd.DataFrame(index=index)
        for output_name, (variable, expected_unit) in _WEATHER_VARIABLES.items():
            output = np.empty(len(index), dtype=np.float64)
            for day in np.unique(lead_days):
                key = f"{variable}_previous_day{day}"
                if units.get(key) != expected_unit:
                    raise ValueError(f"Missing or unexpected units for {key}: expected {expected_unit}")
                column = hourly.get(key)
                if not isinstance(column, list) or len(column) != len(times):
                    raise ValueError(f"Missing or incomplete archived forecast variable: {key}")
                mask = lead_days == day
                try:
                    output[mask] = np.asarray([column[i] for i in positions[mask]], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Nonnumeric archived forecast values: {key}") from exc
            if not np.isfinite(output).all():
                raise ValueError(f"Archived forecast has missing/nonfinite values: {output_name}")
            result[output_name] = output
        if (result.wind_speed < 0).any() or (result.pressure <= 0).any() or (result.temperature <= -273.15).any():
            raise ValueError("Archived forecast contains physically impossible weather")
        result["air_density"] = result.pressure * 100.0 / (287.05 * (result.temperature + 273.15))
        _add_cycles(result, index)
        result = result[list(WEATHER_FEATURES)]
        result["lead_days"] = lead_days
        result["issued_at_upper_bound"] = index - pd.to_timedelta(lead_days, unit="D")
        result["available_at_upper_bound"] = result.issued_at_upper_bound + pd.Timedelta(hours=self.publication_lag_hours)
        if (result.available_at_upper_bound > cutoff).any():
            raise ValueError("Forecast availability would exceed the simulation cutoff")
        result.attrs.update(
            source=FORECAST_ENDPOINT, model=FORECAST_MODEL, cutoff=cutoff.isoformat(),
            publication_lag_hours=self.publication_lag_hours,
            provenance="fixed-lead archive; inferred conservative bounds, not exact run timestamps",
            documentation=FORECAST_DOCUMENTATION, wind_height_m=10,
        )
        return result

    def fetch_forecast(
        self, latitude: float, longitude: float, current_date: Any, horizon: int = 48,
    ) -> pd.DataFrame:
        """Return [T, T+horizon) covariates under a fixed-lead availability policy.

        N = ceil((valid_time - T + publication_lag) / 24h), at least one.
        The 8h default is a conservative operational assumption, not a provider
        guarantee of historical publication time. No reanalysis/live fallback.
        Availability is conditional on the archive's documented lead convention
        and this publication-lag assumption, not verified issue-time evidence.
        """
        cutoff = _utc_timestamp(current_date)
        if cutoff != cutoff.floor("h"):
            raise ValueError("Forecast current_date must be aligned to an hour")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 48:
            raise ValueError("horizon must be an integer from 1 to 48")
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("latitude must lie in [-90, 90]")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("longitude must lie in [-180, 180]")
        index = pd.date_range(cutoff, periods=horizon, freq="h", name="timestamp")
        lead_days = np.maximum(1, np.ceil((np.arange(horizon) + self.publication_lag_hours) / 24)).astype(int)
        if lead_days.max() > 7:
            raise ValueError("Publication lag requires unavailable archive offsets beyond day 7")
        days = sorted(set(lead_days.tolist()))
        params = {
            "latitude": latitude, "longitude": longitude, "models": FORECAST_MODEL,
            "start_date": index[0].strftime("%Y-%m-%d"),
            "end_date": index[-1].strftime("%Y-%m-%d"), "timezone": "UTC",
            "wind_speed_unit": "ms", "temperature_unit": "celsius",
            "hourly": ",".join(f"{variable}_previous_day{day}" for variable, _ in _WEATHER_VARIABLES.values() for day in days),
        }
        identity = {
            "schema_version": 1, "endpoint": FORECAST_ENDPOINT, "params": params,
            "cutoff": cutoff.isoformat(), "horizon": horizon,
            "publication_lag_hours": self.publication_lag_hours,
            "provenance_policy": "fixed-lead-upper-bound-v1",
        }
        cache_key = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict) or envelope.get("identity") != identity:
                raise ValueError("Forecast cache provenance does not match the request")
            payload = envelope.get("response")
            digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
            if digest != envelope.get("response_sha256"):
                raise ValueError("Forecast cache checksum mismatch")
            return self._decode_forecast(payload, index, lead_days, cutoff)
        payload = self._request_json(params)
        result = self._decode_forecast(payload, index, lead_days, cutoff)
        envelope = {
            "identity": identity, "response": payload,
            "response_sha256": hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=f".{cache_key}-", suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_canonical_json(envelope))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return result


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    current_date: Any,
    *,
    horizon: int = 48,
    cache_dir: Path | str = Path("data/weather"),
    publication_lag_hours: float = 8,
    agent: DataAgent | None = None,
) -> pd.DataFrame:
    """Fetch archived forecast features for an explicitly timezone-aware origin.

    Uses Open-Meteo Previous Runs products from the historical forecast archive.
    The stitched Historical Forecast series cannot reconstruct forecasts known
    at an earlier origin. Each returned row includes inferred issue and
    availability upper bounds; these depend on the documented fixed-lead
    convention and the chosen publication lag, not exact publication records.

    Weather responses are small CPU tables. SCADA ingestion and rolling feature
    processing use ``DataAgent(backend="cudf")`` by default on NVIDIA systems.
    For repeated calls, pass ``agent`` to reuse its HTTP session. When supplied,
    that client's cache directory and publication lag apply; the corresponding
    helper arguments configure only newly created clients.
    """
    if agent is not None:
        return agent.fetch_forecast(latitude, longitude, current_date, horizon=horizon)
    agent = DataAgent(
        cache_dir=cache_dir, publication_lag_hours=publication_lag_hours,
    )
    try:
        return agent.fetch_forecast(latitude, longitude, current_date, horizon=horizon)
    finally:
        if agent._http_session is not None:
            agent._http_session.close()
```

## model_agent.py

```python
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
```

## validator_agent.py

```python
"""Physical safety checks shared by the simulator and an eventual serving API."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicsLimits:
    cut_in_ms: float = 3.0
    cut_out_ms: float = 25.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.cut_in_ms) and np.isfinite(self.cut_out_ms)):
            raise ValueError("Wind speed limits must be finite.")
        if not 0 <= self.cut_in_ms < self.cut_out_ms:
            raise ValueError("Require 0 <= cut-in < cut-out.")


def validate_power(
    predicted_power: np.ndarray,
    wind_speed_ms: np.ndarray,
    limits: PhysicsLimits | None = None,
    *,
    apply_wind_limits: bool = True,
) -> np.ndarray:
    """Bound power to [0, 1]; apply shutdowns only for comparable hub-height wind.

    Arrays must have the same nonempty [hours, turbines] shape. Invalid numeric
    inputs fail explicitly: NaN forecasts must never become plausible output.
    The exact cut-in/cut-out boundaries remain eligible for generation.
    """
    limits = limits or PhysicsLimits()
    power = np.asarray(predicted_power, dtype=np.float64)
    wind = np.asarray(wind_speed_ms, dtype=np.float64)
    if power.ndim != 2 or min(power.shape) == 0 or power.shape != wind.shape:
        raise ValueError("Power and wind require identical nonempty [hours, turbines] shapes.")
    if not np.isfinite(power).all() or not np.isfinite(wind).all():
        raise ValueError("Power and wind must contain only finite numbers.")
    if (wind < 0).any():
        raise ValueError("Wind speed cannot be negative.")
    result = np.clip(power, 0.0, 1.0)
    if apply_wind_limits:
        result[(wind < limits.cut_in_ms) | (wind > limits.cut_out_ms)] = 0.0
    return result.astype(np.float32)


def modulus_flow_features(topography: Any, boundary_conditions: Any) -> Any:
    """Extension point for a calibrated NVIDIA Modulus flow model, not a mock.

    Requires terrain, turbine geometry and validated boundary conditions before
    physics-derived features can be claimed or used in training.
    """
    raise NotImplementedError("A calibrated Modulus flow model and terrain inputs are required.")
```

## main_simulation.py

```python
"""Daily, strictly chronological wind-power backtest; offline UI demo included.

All array timestamps denote the beginning of an hour. A simulation origin T may
use observations with timestamps < T and only weather available as of T. The
default expanding policy retrains daily and requires new observed SCADA. The
optional frozen policy reuses the first model/context when later SCADA is absent.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from validator_agent import validate_power


@dataclass(frozen=True)
class Turbine:
    turbine_id: str
    latitude: float
    longitude: float


TURBINES = (Turbine("turbine_1", 51.04, 71.46), Turbine("turbine_2", 51.05, 71.45))
HISTORY_FEATURES = (
    "power", "wind_speed", "temperature", "hour_sin", "hour_cos", "doy_sin",
    "doy_cos", "power_roll_6", "power_roll_24", "wind_roll_6",
)
WEATHER_FEATURES = (
    "wind_speed", "temperature", "pressure", "air_density", "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
)


@dataclass
class TrainingBatch:
    history: np.ndarray
    weather: np.ndarray
    targets: np.ndarray
    origins: list[pd.Timestamp]
    available_at_upper_bound: np.ndarray
    latest_target: pd.Timestamp
    weather_audits: list[dict]
    skipped_incomplete_samples: int


def utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required.")
    return timestamp.tz_convert("UTC")


def simulation_origins(start: str, end: str, simulation_timezone: str = "UTC") -> list[pd.Timestamp]:
    """Produce local daily issue times, represented in UTC, including both dates."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.tzinfo is not None or last.tzinfo is not None:
        raise ValueError("--start and --end must be local dates without a timezone.")
    if pd.isna(first) or pd.isna(last) or first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError("Require midnight dates and start <= end.")
    origins = []
    current_date, last_date = first.to_pydatetime(), last.to_pydatetime()
    while current_date <= last_date:
        origins.append(pd.Timestamp(current_date).tz_localize(simulation_timezone).tz_convert("UTC"))
        current_date += timedelta(days=1)
    return origins


def hour_index(origin: pd.Timestamp, periods: int, offset_hours: int = 0) -> pd.DatetimeIndex:
    return pd.date_range(
        utc_timestamp(origin) + pd.Timedelta(hours=offset_hours),
        periods=periods, freq="h", name="timestamp",
    )


def _check_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{label}: require a timezone-aware DatetimeIndex.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps must be unique and sorted.")
    if not (frame.index == frame.index.floor("h")).all():
        raise ValueError(f"{label}: timestamps must align with whole hours.")


def _observations(
    histories: Sequence[pd.DataFrame], current_date: pd.Timestamp,
) -> None:
    if len(histories) != len(TURBINES):
        raise ValueError("Exactly two turbine histories are required.")
    for number, frame in enumerate(histories, start=1):
        _check_index(frame, f"Turbine {number}")
        if not frame.empty and frame.index.max() >= current_date:
            raise ValueError("Observation leakage: history contains timestamps >= simulation origin.")


def aligned_weather(
    forecasts: Sequence[pd.DataFrame], origin: pd.Timestamp, horizon: int,
    weather_features: Sequence[str] = WEATHER_FEATURES,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Check complete forecasts and conservative publication bounds before use."""
    origin = utc_timestamp(origin)
    expected = hour_index(origin, horizon)
    if len(forecasts) != len(TURBINES):
        raise ValueError("Exactly two weather forecasts are required.")
    arrays, winds, audits = [], [], []
    for turbine, frame in zip(TURBINES, forecasts):
        _check_index(frame, f"Weather {turbine.turbine_id}")
        if not frame.index.tz_convert("UTC").equals(expected):
            raise ValueError("Weather timestamps must exactly cover the requested forecast horizon.")
        values = frame.loc[:, list(weather_features)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Weather forecasts cannot contain gaps or nonfinite values.")
        issued = pd.to_datetime(frame["issued_at_upper_bound"], utc=True, errors="raise")
        available = pd.to_datetime(frame["available_at_upper_bound"], utc=True, errors="raise")
        if issued.isna().any() or available.isna().any():
            raise ValueError("Weather publication metadata cannot be missing.")
        if (issued > available).any() or (available > origin).any():
            raise ValueError("Weather leakage: forecast publication bound exceeds its simulation origin.")
        wind = frame["wind_speed"].to_numpy(dtype=np.float32)
        if (wind < 0).any():
            raise ValueError("Forecast wind speed cannot be negative.")
        arrays.append(values)
        winds.append(wind)
        audits.append({
            "turbine_id": turbine.turbine_id,
            "issued_at_upper_bound": issued.max().isoformat(),
            "available_at_upper_bound": available.max().isoformat(),
            "lead_days": sorted(pd.unique(frame["lead_days"]).astype(int).tolist()),
        })
    return np.concatenate(arrays, axis=-1), np.stack(winds, axis=-1), audits


def build_training_arrays(
    histories: Sequence[pd.DataFrame],
    forecast_loader: Callable[[pd.Timestamp], Sequence[pd.DataFrame]],
    current_date: pd.Timestamp,
    *,
    lookback: int = 72,
    horizon: int = 48,
    train_days: int = 90,
    min_samples: int = 14,
    history_features: Sequence[str] = HISTORY_FEATURES,
    weather_features: Sequence[str] = WEATHER_FEATURES,
) -> TrainingBatch:
    """Make daily samples; every label hour and context hour must be strictly < T.

    The future weather input for each historical origin s is fetched as of s.
    Incomplete observational windows are excluded, never filled/interpolated.
    Missing archived forecasts are a hard error rather than a reanalysis fallback.
    """
    current_date = utc_timestamp(current_date)
    if min(lookback, horizon, train_days, min_samples) < 1:
        raise ValueError("Window lengths and minimum sample count must be positive.")
    _observations(histories, current_date)
    history_arrays, weather_arrays, labels, origins, audits, availability_bounds = [], [], [], [], [], []
    skipped = 0
    candidates = pd.date_range(
        current_date - pd.Timedelta(days=train_days), current_date, freq="D", inclusive="left",
    )
    for origin in candidates:
        label_times = hour_index(origin, horizon)
        if label_times[-1] >= current_date:
            continue
        context_times = hour_index(origin, lookback, -lookback)
        context = np.concatenate([
            frame.reindex(context_times).loc[:, list(history_features)].to_numpy(dtype=np.float32)
            for frame in histories
        ], axis=-1)
        target = np.stack([
            frame.reindex(label_times)["power"].to_numpy(dtype=np.float32) for frame in histories
        ], axis=-1)
        if not np.isfinite(context).all() or not np.isfinite(target).all():
            skipped += 1
            continue
        if ((target < 0) | (target > 1)).any():
            raise ValueError("Training targets must be normalized power in [0, 1].")
        weather, _, weather_audit = aligned_weather(
            forecast_loader(origin), origin, horizon, weather_features,
        )
        history_arrays.append(context)
        weather_arrays.append(weather)
        labels.append(target)
        origins.append(origin)
        audits.append({"origin": origin.isoformat(), "turbines": weather_audit})
        # A conservative maximum over both turbines and all hours bounds every
        # forecast value in this sample; retain it for the model's own checks.
        bound = max(pd.Timestamp(item["available_at_upper_bound"]) for item in weather_audit)
        availability_bounds.append([bound.to_pydatetime()] * horizon)
    if len(origins) < min_samples:
        raise ValueError(
            f"Only {len(origins)} complete training samples; require {min_samples}. "
            "Check input timezone, hourly completeness and history length."
        )
    return TrainingBatch(
        history=np.stack(history_arrays), weather=np.stack(weather_arrays), targets=np.stack(labels),
        origins=origins, latest_target=origins[-1] + pd.Timedelta(hours=horizon - 1),
        available_at_upper_bound=np.asarray(availability_bounds, dtype=object),
        weather_audits=audits, skipped_incomplete_samples=skipped,
    )


def _json_default(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False, newline="",
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_daily_submission(
    output_dir: Path, origin: pd.Timestamp, power: np.ndarray,
    wind: np.ndarray, audit: dict, *, apply_wind_limits: bool = True,
) -> Path:
    """Write one complete horizon per origin, retaining March hours on February 28."""
    origin = utc_timestamp(origin)
    bounded = validate_power(power, wind, apply_wind_limits=apply_wind_limits)
    if bounded.shape[1] != len(TURBINES):
        raise ValueError("Submission requires two turbines.")
    valid_times = hour_index(origin, bounded.shape[0])
    rows = [
        {
            "forecast_origin": origin.isoformat(), "valid_time": valid_time.isoformat(),
            "turbine_id": turbine.turbine_id, "power_normalized": float(bounded[hour, index]),
            "wind_speed_ms": float(wind[hour, index]),
        }
        for hour, valid_time in enumerate(valid_times)
        for index, turbine in enumerate(TURBINES)
    ]
    stem = f"forecast_{origin.strftime('%Y%m%dT%H%MZ')}"
    csv_path, audit_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.json"
    if csv_path.exists() and not audit_path.exists():
        raise FileExistsError(f"Refusing to replace an unaudited existing file: {csv_path}")
    if audit_path.exists():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        if previous.get("mode") != audit.get("mode"):
            raise ValueError("Demo and real backtest outputs must use separate directories.")
    audit = {
        **audit, "forecast_origin": origin.isoformat(), "horizon_hours": len(valid_times),
        "last_valid_time": valid_times[-1].isoformat(), "rows": len(rows),
        "physical_validation": {
            "power_range": [0, 1], "cut_in_ms": 3 if apply_wind_limits else None,
            "cut_out_ms": 25 if apply_wind_limits else None,
            "wind_limits_applied": apply_wind_limits,
            "strict_wind_thresholds": apply_wind_limits,
            "forced_zero_count": int(((wind < 3) | (wind > 25)).sum()) if apply_wind_limits else 0,
        },
    }
    # Each artifact is atomically replaced; the sidecar contains its own complete provenance.
    _atomic_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    _atomic_text(csv_path, pd.DataFrame(rows).to_csv(index=False, float_format="%.7f"))
    return csv_path


SUBMISSION_COLUMNS = (
    "forecast_origin", "valid_time", "turbine_id", "power_normalized", "wind_speed_ms",
)


def write_cumulative_submission(path: Path, daily_paths: Sequence[Path]) -> int:
    """Atomically replace the cumulative CSV using only this run's daily paths.

    Overlapping valid times from different forecast origins are separate rows.
    Rebuilding avoids duplicate append rows on reruns and ignores stale files.
    """
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    rows = 0
    for daily_path in daily_paths:
        with daily_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
                raise ValueError(f"Unexpected submission columns: {daily_path}")
            for row in reader:
                writer.writerow(row)
                rows += 1
    _atomic_text(path, stream.getvalue())
    return rows


def synthetic_forecast(origin: pd.Timestamp, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic illustrative data for UI development; no model/weather API used."""
    hours = hour_index(origin, horizon).asi8.astype(np.float64) / 3.6e12
    wind = np.stack([
        8.5 + 3.8 * np.sin(hours / 10.0 + index * 0.2)
        + 1.8 * np.cos(hours / 37.0) + index * 0.25
        for index in range(len(TURBINES))
    ], axis=-1).astype(np.float32)
    power = np.clip((wind**3 - 3**3) / (12**3 - 3**3), 0, 1)
    return validate_power(power, wind), wind


def run(args: argparse.Namespace) -> None:
    if args.lookback < 1 or args.train_days < 1 or args.min_train_samples < 1:
        raise ValueError("History windows and sample counts must be positive.")
    origins = simulation_origins(args.start, args.end, args.simulation_timezone)
    mode = "synthetic-demo" if args.demo else "historical-backtest"
    output_dir = Path(args.output or ("outputs/demo" if args.demo else "outputs/backtest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("mode") != mode:
            raise ValueError("Demo and real backtest outputs must use separate directories.")
    manifest = {
        "mode": mode, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "submission_complete": False,
        "submission_path": "submission.csv", "submission_rows": 0,
        "completed_origins": [], "daily_files": [],
        "simulation_timezone": args.simulation_timezone, "source_data_timezone": args.data_timezone,
        "start": args.start, "end": args.end, "horizon_hours": args.horizon,
        "history_policy": args.history_policy if not args.demo else None,
        "turbines": [asdict(turbine) for turbine in TURBINES],
        "synthetic_notice": "Illustrative synthetic data; not trained-model predictions." if args.demo else None,
    }
    def save_manifest() -> None:
        _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n")

    save_manifest()
    daily_paths: list[Path] = []
    try:
        # Reset cumulative output for this invocation; old daily files are not
        # discovered by glob and cannot enter the current run's submission.
        write_cumulative_submission(output_dir / "submission.csv", daily_paths)
        if not args.demo:
            if not args.data_timezone:
                raise ValueError("Real execution requires --data-timezone for the source CSV timestamps.")
            # Lazy imports keep --help and --demo independent of CUDA/PyTorch.
            from data_agent import DATASET_FILENAMES, DataAgent, fetch_weather_forecast
            from model_agent import ForecastData, ModelConfig, TrainingData, predict, train_model

            input_paths = (
                Path(args.turbine_1 or (args.data_dir / DATASET_FILENAMES[0])),
                Path(args.turbine_2 or (args.data_dir / DATASET_FILENAMES[1])),
            )
            data_agent = DataAgent(
                backend=args.backend, data_timezone=args.data_timezone,
                cache_dir=Path(args.cache_dir), publication_lag_hours=args.publication_lag_hours,
            )
            config = ModelConfig(
                history_features=len(HISTORY_FEATURES) * len(TURBINES),
                weather_features=len(WEATHER_FEATURES) * len(TURBINES),
                num_turbines=len(TURBINES), lookback=args.lookback, horizon=args.horizon,
                epochs=args.epochs, batch_size=args.batch_size, device=args.device, seed=args.seed,
            )
            manifest["model_config"] = asdict(config)
            manifest["input_files"] = [str(path.resolve()) for path in input_paths]
            manifest["publication_lag_hours_assumption"] = args.publication_lag_hours
            save_manifest()

            def forecasts_at(sample_origin: pd.Timestamp) -> list[pd.DataFrame]:
                return [
                    fetch_weather_forecast(
                        turbine.latitude, turbine.longitude, current_date=sample_origin,
                        horizon=args.horizon, agent=data_agent,
                    ) for turbine in TURBINES
                ]

        frozen_state = None
        for origin in origins:
            if args.demo:
                power, wind = synthetic_forecast(origin, args.horizon)
                audit = {
                    "mode": mode, "trained_model": False, "weather_source": "synthetic",
                    "notice": "Offline illustrative data. Not a competition prediction or model evaluation.",
                }
            else:
                context_origin = origins[0] if args.history_policy == "frozen" else origin
                if frozen_state is not None:
                    histories, batch, context, model, checkpoint = frozen_state
                else:
                    histories = [
                        data_agent.load_history(path, current_date=context_origin) for path in input_paths
                    ]
                    context_times = hour_index(context_origin, args.lookback, -args.lookback)
                    context = np.concatenate([
                        history.reindex(context_times).loc[:, list(HISTORY_FEATURES)].to_numpy(dtype=np.float32)
                        for history in histories
                    ], axis=-1)
                    if not np.isfinite(context).all():
                        latest = [str(history["power"].dropna().index.max()) for history in histories]
                        raise ValueError(
                            f"Incomplete inference history at {context_origin}. Expanding daily training "
                            f"requires observed SCADA through {context_times[-1]}; latest measurements: "
                            f"{latest}. The supplied files stop in January; February observations are "
                            "required from February 2 onward. Gaps cannot be synthesized or filled. "
                            "Use --history-policy frozen only for an explicitly frozen baseline."
                        )
                    batch = build_training_arrays(
                        histories, forecasts_at, context_origin, lookback=args.lookback, horizon=args.horizon,
                        train_days=args.train_days, min_samples=args.min_train_samples,
                    )
                    model = train_model(TrainingData(
                        history=batch.history, weather=batch.weather, targets=batch.targets,
                        origins=[sample.to_pydatetime() for sample in batch.origins],
                        current_date=context_origin.to_pydatetime(),
                        available_at_upper_bound=batch.available_at_upper_bound,
                    ), config=config)
                    checkpoint = output_dir / "checkpoints" / f"model_{context_origin.strftime('%Y%m%dT%H%MZ')}.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    model.save(checkpoint)
                    if args.history_policy == "frozen":
                        frozen_state = (histories, batch, context, model, checkpoint)
                weather, wind, forecast_audit = aligned_weather(forecasts_at(origin), origin, args.horizon)
                bound = max(pd.Timestamp(item["available_at_upper_bound"]) for item in forecast_audit)
                power = predict(model, ForecastData(
                    history=context, weather=weather, current_date=origin.to_pydatetime(),
                    available_at_upper_bound=[bound.to_pydatetime()] * args.horizon,
                    context_origin=context_origin.to_pydatetime(),
                ))
                audit = {
                    "mode": mode, "trained_model": True,
                    "history_policy": args.history_policy,
                    "observation_cutoff_exclusive": context_origin.isoformat(),
                    "context_origin": context_origin.isoformat(),
                    "context_age_hours": (origin - context_origin).total_seconds() / 3600,
                    "history_policy_notice": (
                        "Frozen-context baseline: SCADA and model fixed at first simulation origin. "
                        "Weather updates daily; increasingly stale context differs from training conditions."
                        if args.history_policy == "frozen" else
                        "Expanding history: full daily retraining using only prior observed SCADA."
                    ),
                    "latest_observation": [history["power"].dropna().index.max().isoformat() for history in histories],
                    "weather_source": "Open-Meteo previous-runs forecast API / gfs_global",
                    "wind_height_m": 10,
                    "publication_lag_hours_assumption": args.publication_lag_hours,
                    "weather_availability_bounds_checked": True, "inference_weather": forecast_audit,
                    "training": {
                        "samples": len(batch.origins), "first_origin": batch.origins[0].isoformat(),
                        "last_origin": batch.origins[-1].isoformat(),
                        "latest_target_time": batch.latest_target.isoformat(),
                        "targets_strictly_before_observation_cutoff": True,
                        "skipped_incomplete_samples": batch.skipped_incomplete_samples,
                        "forecast_provenance": batch.weather_audits, "metrics": model.training_metrics,
                    },
                    "checkpoint": str(checkpoint.relative_to(output_dir)),
                }
                if args.history_policy == "expanding":
                    del model
            path = write_daily_submission(output_dir, origin, power, wind, audit, apply_wind_limits=args.demo)
            daily_paths.append(path)
            manifest["submission_rows"] = write_cumulative_submission(output_dir / "submission.csv", daily_paths)
            manifest["completed_origins"].append(origin.isoformat())
            manifest["daily_files"].append(path.name)
            save_manifest()
            print(f"[{mode}] {origin.isoformat()} -> {path}", flush=True)
        manifest["status"] = "completed"
        manifest["submission_complete"] = True
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["submission_complete"] = False
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Offline synthetic demo; does not train a model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing the two exact participant CSV filenames.")
    parser.add_argument("--turbine-1", type=Path, help="Override the turbine 1 CSV under --data-dir.")
    parser.add_argument("--turbine-2", type=Path, help="Override the turbine 2 CSV under --data-dir.")
    parser.add_argument("--data-timezone", help="IANA timezone of source CSV timestamps; confirm with owner.")
    parser.add_argument("--simulation-timezone", default="UTC", help="Timezone of daily forecast origin (default UTC).")
    parser.add_argument(
        "--history-policy", choices=("frozen", "expanding"), default="expanding",
        help="Expanding retrains daily (default), requiring observed SCADA; frozen reuses the first model/context.",
    )
    parser.add_argument("--start", default="2026-02-01")
    parser.add_argument("--end", default="2026-02-28")
    parser.add_argument("--horizon", type=int, choices=(24, 48), default=48)
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--min-train-samples", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("cudf", "pandas"), default="cudf")
    parser.add_argument("--device", default="cuda", help="PyTorch device; no automatic CUDA fallback.")
    parser.add_argument("--publication-lag-hours", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/weather")
    parser.add_argument("--output", type=Path, help="Default: outputs/demo or outputs/backtest depending on mode.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.demo and not args.data_timezone:
        parser.error("real mode requires --data-timezone; CSVs default to the exact filenames in --data-dir")
    run(args)


if __name__ == "__main__":
    main()
```

## scientific_report.py

```python
"""Reproducible scientific figures and PDF/HTML reports from audited hourly forecasts.

This module only describes forecasts and optionally scores later observations.
It neither trains models nor substitutes observations into forecast features.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from standards_profile import wind_standards_profile

IDS = ("turbine_1", "turbine_2")
COLORS = ("#177565", "#35649a")
FIELDS = ("forecast_origin", "valid_time", "turbine_id", "power_normalized", "wind_speed_ms")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
         "axes.labelsize": 10, "axes.linewidth": .8, "pdf.fonttype": 42,
         "svg.fonttype": "none", "savefig.facecolor": "white"}


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include an explicit UTC offset")
    return result.astimezone(timezone.utc)


@dataclass
class Forecast:
    origin: datetime
    times: list[datetime]
    power: np.ndarray
    wind: np.ndarray
    audit: dict

    @property
    def demo(self) -> bool:
        return self.audit["mode"] == "synthetic-demo"


def validate_forecast(records: list[dict], audit: dict, horizon: int | None = None) -> Forecast:
    """Reject partial, mixed-origin, duplicated or nonfinite forecast artifacts."""
    if audit.get("mode") not in ("synthetic-demo", "historical-backtest"):
        raise ValueError("Unsupported forecast audit mode")
    if audit["mode"] == "historical-backtest" and audit.get("trained_model") is not True:
        raise ValueError("Model forecasts must have a trained-model audit")
    full_horizon = audit.get("horizon_hours")
    if type(full_horizon) is not int or full_horizon not in (24, 48):
        raise ValueError("Forecast horizon must be 24 or 48 hours")
    horizon = full_horizon if horizon is None else horizon
    if type(horizon) is not int or horizon not in (24, 48) or horizon > full_horizon:
        raise ValueError("Requested horizon exceeds the available forecast")
    origin = timestamp(audit["forecast_origin"])
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError("Forecast origin must align to an hour")
    if len(records) != full_horizon * 2:
        raise ValueError("Exactly two turbine values are required per forecast hour")
    power = np.full((full_horizon, 2), np.nan)
    wind = np.full_like(power, np.nan)
    for row in records:
        if set(row) != set(FIELDS) or timestamp(row["forecast_origin"]) != origin:
            raise ValueError("Unexpected columns or mixed forecast origins")
        if row["turbine_id"] not in IDS:
            raise ValueError("Unknown turbine")
        lead = (timestamp(row["valid_time"]) - origin).total_seconds() / 3600
        if not lead.is_integer() or not 0 <= lead < full_horizon:
            raise ValueError("Forecast timestamp is outside the hourly horizon")
        i, j = int(lead), IDS.index(row["turbine_id"])
        p, v = float(row["power_normalized"]), float(row["wind_speed_ms"])
        if not np.isfinite([p, v]).all() or not 0 <= p <= 1 or v < 0:
            raise ValueError("Power must be finite in [0,1]; wind must be finite and nonnegative")
        if np.isfinite(power[i, j]):
            raise ValueError("Duplicate forecast turbine/hour")
        power[i, j], wind[i, j] = p, v
    if not np.isfinite(power).all():
        raise ValueError("Missing forecast turbine/hour")
    return Forecast(origin, [origin + timedelta(hours=i) for i in range(horizon)],
                    power[:horizon], wind[:horizon], audit)


def load_forecast(path: Path, horizon: int | None = None) -> Forecast:
    audit = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with path.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    return validate_forecast(records, audit, horizon)


def load_observations(path: Path, forecast: Forecast) -> np.ndarray:
    """Match later hourly normalized observations by UTC timestamp and turbine.

    Missing hours stay NaN. No interpolation or filling of validation targets.
    Input schema: valid_time,turbine_id,power_normalized.
    """
    if forecast.demo:
        raise ValueError("Synthetic forecasts cannot be scored against observed SCADA")
    observed = np.full_like(forecast.power, np.nan)
    lookup = {time: i for i, time in enumerate(forecast.times)}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != {"valid_time", "turbine_id", "power_normalized"}:
            raise ValueError("Unexpected observation columns")
        for row in reader:
            time, turbine = timestamp(row["valid_time"]), row["turbine_id"]
            if turbine not in IDS or time.minute or time.second or time.microsecond:
                raise ValueError("Observations must identify a turbine and an aligned UTC hour")
            key = (time, turbine)
            if key in seen:
                raise ValueError("Duplicate observed turbine/hour")
            seen.add(key)
            value = float(row["power_normalized"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Observed power must be finite in [0,1]")
            if time in lookup:
                observed[lookup[time], IDS.index(turbine)] = value
    return observed


def statistics(forecast: Forecast, observed: np.ndarray | None = None) -> list[dict]:
    if observed is not None:
        if observed.shape != forecast.power.shape or np.isinf(observed).any():
            raise ValueError("Observation matrix must match forecast shape and contain no infinities")
        finite = observed[np.isfinite(observed)]
        if ((finite < 0) | (finite > 1)).any() or forecast.demo:
            raise ValueError("Cannot score these observations")
    result = []
    for j, turbine in enumerate(IDS):
        p, wind = forecast.power[:, j], forecast.wind[:, j]
        row = {"turbine_id": turbine, "n_forecast": len(p), "mean_power_pu": float(p.mean()),
               "peak_power_pu": float(p.max()), "mean_wind_ms": float(wind.mean()),
               "max_abs_ramp_pu_per_hour": float(np.abs(np.diff(p)).max()),
               "integrated_power_pu_h": float(p.sum()), "n_observed": 0,
               "mae_pu": None, "rmse_pu": None, "bias_pu": None}
        if observed is not None:
            mask = np.isfinite(observed[:, j])
            error = p[mask] - observed[mask, j]
            row["n_observed"] = int(mask.sum())
            if len(error):
                row.update(mae_pu=float(np.abs(error).mean()), rmse_pu=float(np.sqrt(np.mean(error**2))),
                           bias_pu=float(error.mean()))
        result.append(row)
    return result


def forecast_csv(forecast: Forecast) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(FIELDS)
    for i, time in enumerate(forecast.times):
        for j, turbine in enumerate(IDS):
            writer.writerow([forecast.origin.isoformat(), time.isoformat(), turbine,
                             forecast.power[i, j], forecast.wind[i, j]])
    return stream.getvalue()


def grid(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#bdc4c9", linewidth=.65)
    ax.grid(which="minor", color="#e6e9eb", linewidth=.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="in", which="both", top=True, right=True)


def time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M", tz=timezone.utc))


def figures(forecast: Forecast, observed: np.ndarray | None):
    """Yield original, data-derived figures; no manufacturer curves are fabricated."""
    t, p, v = forecast.times, forecast.power, forecast.wind
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), sharex=True)
    for j in range(2):
        style = "-" if j == 0 else "--"
        axes[0].plot(t, p[:, j], style, color=COLORS[j], label=f"T{j+1} · прогноз", linewidth=1.6)
        axes[1].plot(t, v[:, j], style, color=COLORS[j], label=f"T{j+1}", linewidth=1.4)
        if observed is not None:
            axes[0].plot(t, observed[:, j], ".", color=COLORS[j], label=f"T{j+1} · факт")
    grid(axes[0], "", "Нормализованная мощность, p.u.")
    grid(axes[1], "Время действия прогноза, UTC", "Скорость ветра, м/с")
    axes[0].set_ylim(-.03, 1.06)
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].legend(fontsize=9)
    time_axis(axes[1])
    yield fig, "trajectory", "Почасовые траектории мощности и ветра", "Каждая точка соответствует одному часу. Разрывы фактических измерений не заполняются."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.scatter(v[:, j], p[:, j], s=30, marker="o" if j == 0 else "^", facecolors="none",
                   edgecolors=COLORS[j], linewidths=1.1, label=f"T{j+1} · {len(t)} прогнозных точек")
    if forecast.audit.get("physical_validation", {}).get("wind_limits_applied", forecast.demo):
        for threshold, label in ((3, "Включение 3 м/с"), (25, "Отключение 25 м/с")):
            ax.axvline(threshold, color="#9b7546", linestyle=":", linewidth=1, label=label)
    grid(ax, "Прогноз скорости ветра, м/с", "Прогноз мощности, p.u.")
    ax.set_ylim(-.03, 1.06)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    yield fig, "power_wind", "Рабочие точки: мощность — скорость ветра", "Зависимость двух прогнозируемых величин; не паспортная и не измеренная кривая мощности. Высота ветра указана в методике."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.step(np.arange(1, len(t) + 1) / len(t) * 100, np.sort(p[:, j])[::-1],
                where="post", color=COLORS[j], linestyle="-" if j == 0 else "--", label=f"T{j+1}")
    grid(ax, "Доля прогнозных часов с мощностью не ниже указанной, %", "Нормализованная мощность, p.u.")
    ax.set_xlim(0, 100)
    ax.set_ylim(-.03, 1.06)
    ax.legend()
    yield fig, "duration", "Кривые обеспеченности прогнозной мощности", "Мощность отсортирована по убыванию внутри выбранного горизонта. Это характеристика прогноза, а не годовая статистика."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.plot(t[1:], np.diff(p[:, j]), "o-" if j == 0 else "^--", color=COLORS[j],
                markersize=3, linewidth=1.2, label=f"T{j+1}")
    ax.axhline(0, color="#343f46", linewidth=.8)
    grid(ax, "Время окончания часового интервала, UTC", "Изменение мощности ΔP/Δt, p.u./ч")
    time_axis(ax)
    ax.legend()
    yield fig, "ramps", "Почасовые изменения мощности", "ΔP = P(t) − P(t−1), Δt = 1 ч. Положительные значения означают рост прогнозной мощности."

    if observed is not None and np.isfinite(observed).any():
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), sharex=True, sharey=True)
        stats = statistics(forecast, observed)
        for j, ax in enumerate(axes):
            mask = np.isfinite(observed[:, j])
            ax.plot([0, 1], [0, 1], "k--", linewidth=.8, label="Идеальное совпадение")
            ax.scatter(observed[mask, j], p[mask, j], facecolors="none", edgecolors=COLORS[j], s=35)
            ax.set_aspect("equal")
            grid(ax, "Фактическая мощность, p.u.", "Прогнозная мощность, p.u.")
            ax.set_xlim(-.03, 1.03)
            ax.set_ylim(-.03, 1.03)
            row = stats[j]
            label = f"T{j+1} · n = {row['n_observed']}"
            if row["n_observed"]:
                label += f"\nMAE = {row['mae_pu']:.4f}; RMSE = {row['rmse_pu']:.4f} p.u."
            ax.set_title(label, fontsize=11)
        yield fig, "validation", "Сопоставление прогноза с измерениями", "Используются только совпавшие UTC-часы и турбины. Измерения применяются после прогноза исключительно для оценки."

        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))
        for j in range(2):
            error = p[:, j] - observed[:, j]
            axes[0].plot(t, error, "o-" if j == 0 else "^--", color=COLORS[j], markersize=3, label=f"T{j+1}")
            finite = error[np.isfinite(error)]
            if len(finite):
                axes[1].hist(finite, bins=np.linspace(-1, 1, 21), histtype="step", color=COLORS[j], label=f"T{j+1}")
        axes[0].axhline(0, color="#343f46", linewidth=.8)
        grid(axes[0], "Время действия прогноза, UTC", "Ошибка Pпрогноз − Pфакт, p.u.")
        time_axis(axes[0])
        grid(axes[1], "Ошибка прогноза, p.u.", "Число совпавших часов")
        for ax in axes:
            ax.legend(fontsize=9)
        yield fig, "residuals", "Временная структура и распределение ошибок", "Положительная ошибка означает завышение прогноза. Пропуски измерений исключены; выборка ограничена выбранным горизонтом."


def make_report(forecast: Forecast, output_root: Path, observed: np.ndarray | None = None) -> Path:
    stats = statistics(forecast, observed)
    payload = forecast_csv(forecast)
    canonical = payload + json.dumps(forecast.audit, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if observed is not None:
        canonical += json.dumps(np.where(np.isfinite(observed), observed, -1).tolist())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    standards = wind_standards_profile()
    report_digest = hashlib.sha256((canonical + json.dumps(standards, sort_keys=True, ensure_ascii=False)).encode("utf-8")).hexdigest()
    report_id = f"forecast_{forecast.origin:%Y%m%dT%H%MZ}_{len(forecast.times)}h_{report_digest[:12]}"
    destination = output_root / report_id
    destination.mkdir(parents=True, exist_ok=True)
    observed_n = sum(row["n_observed"] for row in stats)
    status = "СИНТЕТИЧЕСКИЙ ПРИМЕР" if forecast.demo else "ПРОГНОЗ МОДЕЛИ"
    wind_height = forecast.audit.get("wind_height_m")
    wind_text = f"{wind_height} м" if wind_height is not None else "не указана в аудите"
    notes = [
        "Выбранный горизонт содержит полные почасовые пары T1 и T2. Часовой пояс отчёта — UTC.",
        "Мощность задана в p.u. Пересчёт в МВт и МВт·ч невозможен без подтверждённой базы нормировки и номинальной мощности.",
        f"Высота ветра: {wind_text}. Диаграмма P(v) описывает прогнозные рабочие точки, а не паспортную характеристику.",
        "Наблюдения используются только для последующей оценки. Никакой подстановки в признаки модели здесь нет.",
        "MAE = mean(|Pпрогноз − Pфакт|); RMSE = sqrt(mean((Pпрогноз − Pфакт)²)); bias = mean(Pпрогноз − Pфакт).",
        "Доверительные интервалы, аэродинамический КПД и карты компрессора не рассчитываются: необходимых данных и калибровки нет.",
    ]
    if forecast.demo:
        notes.insert(0, "Все значения синтетические. Этот документ демонстрирует формат, а не точность модели или свойства реального оборудования.")
    if not observed_n:
        notes.append("Фактические значения для оценки не предоставлены или не совпали по времени. Ошибки и метрики качества недоступны.")
    bounds = forecast.audit.get("physical_validation", {})
    limits_applied = bounds.get("wind_limits_applied", forecast.demo)
    notes.append("Пороги 3/25 м/с применены источником прогноза." if limits_applied
                 else "Пороги 3/25 м/с не применены источником; ветер на 10 м не равен ветру на высоте ступицы.")
    metadata = {"schema_version": 2, "report_id": report_id, "source_sha256": digest, "report_sha256": report_digest,
                "generated_at": datetime.now(timezone.utc).isoformat(), "mode": forecast.audit["mode"],
                "forecast_origin": forecast.origin.isoformat(), "horizon_hours": len(forecast.times),
                "n_observed": observed_n, "statistics": stats, "methodology": notes, "audit": forecast.audit,
                "standards_profile": standards, "standards_file": "standards.json",
                "references": [{"title": "NREL: Fundamentals of Wind Energy", "url": "https://www.nrel.gov/docs/fy23osti/84501.pdf"}]
                              + [{"title": item["designation"], "url": item["source"]} for item in standards["selected"]],
                "figures": [], "pdf": "report.pdf", "html": "report.html", "bundle": "report_bundle.zip"}
    provenance = [
        ("Выпуск прогноза", forecast.origin.strftime("%d.%m.%Y %H:%M UTC")),
        ("Горизонт", f"{len(forecast.times)} ч · {len(forecast.times)*2} турбино-часов"),
        ("Источник погоды", forecast.audit.get("weather_source", "не указан")),
        ("Политика истории", forecast.audit.get("history_policy", "не применимо")),
        ("Наблюдения до (исключая)", forecast.audit.get("observation_cutoff_exclusive", "не указано")),
        ("Лаг доступности погоды", str(forecast.audit.get("publication_lag_hours_assumption", "не указан")) + " ч (допущение)"),
        ("Checkpoint", forecast.audit.get("checkpoint", "не применимо")),
        ("SHA-256 источника", digest),
    ]
    svg_sections = []
    with plt.rc_context(STYLE), PdfPages(destination / "report.pdf", metadata={"Title": "ALEM WIND — научный отчёт", "Author": "ALEM WIND", "Subject": status}) as pdf:
        cover = plt.figure(figsize=(11.7, 8.3))
        cover.text(.075, .92, "ALEM WIND / НАУЧНЫЙ ОТЧЁТ", fontsize=20, weight="bold")
        cover.text(.075, .87, status, fontsize=11, color=COLORS[0])
        y = .81
        for key, value in provenance:
            line = f"{key}: {value}"
            for wrapped in textwrap.wrap(line, 100):
                cover.text(.075, y, wrapped, fontsize=9)
                y -= .026
        columns = ["Турбина", "n", "Средняя P\np.u.", "Пик P\np.u.", "Средний v\nм/с", "max |ΔP|\np.u./ч", "Факт n", "MAE\np.u.", "RMSE\np.u.", "Bias\np.u."]
        table_rows = [[f"T{j+1}", s["n_forecast"], f"{s['mean_power_pu']:.4f}", f"{s['peak_power_pu']:.4f}", f"{s['mean_wind_ms']:.2f}", f"{s['max_abs_ramp_pu_per_hour']:.4f}", s["n_observed"], "—" if s["mae_pu"] is None else f"{s['mae_pu']:.4f}", "—" if s["rmse_pu"] is None else f"{s['rmse_pu']:.4f}", "—" if s["bias_pu"] is None else f"{s['bias_pu']:.4f}"] for j, s in enumerate(stats)]
        ax = cover.add_axes([.075, .30, .85, .18])
        ax.axis("off")
        table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        cover.text(.075, .24, "Оценка: " + (f"{observed_n} совпавших турбино-часов; неполная выборка возможна." if observed_n else "фактические данные недоступны; точность не оценена."), fontsize=10)
        cover.text(.075, .16, textwrap.fill(notes[0], 105), fontsize=9)
        cover.text(.075, .06, f"ID: {report_id} · значения округлены только для отображения", fontsize=8, color="#56636d")
        pdf.savefig(cover)
        plt.close(cover)
        for number, (fig, name, title, caption) in enumerate(figures(forecast, observed), 1):
            fig.suptitle(f"Рисунок {number}. {title}", x=.075, ha="left", y=.97, fontsize=15, weight="bold")
            fig.text(.075, .92, f"{status} · выпуск {forecast.origin:%d.%m.%Y %H:%M} UTC · горизонт {len(forecast.times)} ч", fontsize=9, color="#52606b")
            fig.tight_layout(rect=(.025, .13, .98, .89), h_pad=2)
            fig.text(.075, .045, textwrap.fill(caption, 115), fontsize=9)
            fig.savefig(destination / f"{name}.svg")
            fig.savefig(destination / f"{name}.png", dpi=300)
            pdf.savefig(fig)
            plt.close(fig)
            metadata["figures"].append({"id": name, "title": title, "caption": caption, "svg": f"{name}.svg", "png": f"{name}.png"})
            svg = (destination / f"{name}.svg").read_text(encoding="utf-8")
            svg = svg[svg.index("<svg"):]
            svg_sections.append(f'<figure>{svg}<figcaption>Рисунок {number}. {html.escape(caption)}</figcaption></figure>')
        method = plt.figure(figsize=(11.7, 8.3))
        method.text(.075, .91, "Методика, происхождение и ограничения", fontsize=18, weight="bold")
        y = .84
        for i, note in enumerate(notes, 1):
            lines = textwrap.wrap(f"{i}. {note}", 112)
            method.text(.075, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.6)
            y -= .03 * len(lines) + .019
        method.text(.075, .10, "Справочный материал: NREL, Fundamentals of Wind Energy\nhttps://www.nrel.gov/docs/fy23osti/84501.pdf", fontsize=9)
        method.text(.075, .045, "Аудит источника и машинные значения приложены в metadata.json и forecast.csv.", fontsize=9)
        pdf.savefig(method)
        plt.close(method)
        normative = plt.figure(figsize=(11.7, 8.3))
        normative.text(.075, .92, "Нормативная основа ВЭС", fontsize=18, weight="bold")
        normative.text(.075, .87, standards["statement"], fontsize=10, color="#78591f")
        y = .80
        for item in standards["selected"]:
            normative.text(.075, y, item["designation"], fontsize=12, weight="bold")
            lines = textwrap.wrap(item["application"] + " " + item["implementation"], 114)
            y -= .035
            normative.text(.075, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.5)
            y -= .025 * len(lines) + .035
        normative.text(.075, y, "Что требуется для дальнейшего применения", fontsize=12, weight="bold")
        y -= .04
        for item in standards["open_items"]:
            lines = textwrap.wrap("• " + item, 114)
            normative.text(.075, y, "\n".join(lines), fontsize=9, va="top", linespacing=1.4)
            y -= .025 * len(lines) + .015
        normative.text(.075, .075, "Области применения и ссылки проверены 23.09.2026. Перечень источников и статусы — в standards.json.", fontsize=8)
        normative.text(.075, .04, "Объект: ветроустановки. Классы и обозначения паровых турбин к этим данным не применяются.", fontsize=8)
        pdf.savefig(normative)
        plt.close(normative)
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in table_rows)
    provenance_html = "".join(f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>" for k, v in provenance)
    standards_html = "".join(
        f'<li><a href="{html.escape(item["source"], quote=True)}">{html.escape(item["designation"])}</a>: '
        f'{html.escape(item["application"])} {html.escape(item["implementation"])}</li>'
        for item in standards["selected"]
    )
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALEM WIND — научный отчёт</title>
<style>body{{font:15px/1.6 Arial,sans-serif;color:#172630;margin:40px auto;padding:0 24px;max-width:1100px}}h1{{font-size:30px}}.status{{color:#177565;font-weight:bold}}dl{{display:grid;grid-template-columns:240px 1fr;gap:8px}}dt{{font-weight:bold}}dd{{margin:0;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #bac4ca;padding:10px}}th{{background:#f0f3f4}}figure{{margin:40px 0}}svg{{width:100%;height:auto}}figcaption{{font-size:13px;color:#475963}}li{{margin:10px 0}}@media print{{@page{{size:A4 landscape;margin:14mm}}body{{margin:0;max-width:none}}figure{{break-before:page;break-inside:avoid}}a.download{{display:none}}}}</style>
<h1>ALEM WIND / Научный отчёт</h1><p class="status">{status}</p><p><a class="download" href="report.pdf">PDF</a> · <a class="download" href="report_bundle.zip">Все файлы и графики</a></p><dl>{provenance_html}</dl>
<table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{cells}</tbody></table>
<p>Фактических совпавших турбино-часов: {observed_n}. {'Точность не оценена.' if not observed_n else 'Метрики рассчитаны только по совпавшим значениям.'}</p>
{''.join(svg_sections)}<h2>Методика и ограничения</h2><ol>{''.join(f'<li>{html.escape(n)}</li>' for n in notes)}</ol>
<h2>Нормативная основа ВЭС</h2><p>{html.escape(standards['statement'])}</p><ul>{standards_html}</ul>
<p>Для дальнейшего применения:</p><ol>{''.join(f'<li>{html.escape(n)}</li>' for n in standards['open_items'])}</ol>
<p>Справочный материал: <a href="https://www.nrel.gov/docs/fy23osti/84501.pdf">NREL: Fundamentals of Wind Energy</a>.</p></html>'''
    (destination / "report.html").write_text(document, encoding="utf-8")
    (destination / "forecast.csv").write_text(payload, encoding="utf-8")
    if observed is not None:
        with (destination / "observations.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["valid_time", "turbine_id", "power_normalized"])
            for i, time in enumerate(forecast.times):
                for j, turbine in enumerate(IDS):
                    if np.isfinite(observed[i, j]):
                        writer.writerow([time.isoformat(), turbine, observed[i, j]])
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "standards.json").write_text(json.dumps(standards, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    names = ["report.pdf", "report.html", "forecast.csv", "metadata.json", "standards.json"]
    names += [figure[extension] for figure in metadata["figures"] for extension in ("svg", "png")]
    if observed is not None:
        names.append("observations.csv")
    with zipfile.ZipFile(destination / "report_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        sums = []
        for name in names:
            content = (destination / name).read_bytes()
            archive.writestr(name, content)
            sums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
        archive.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True, help="Daily CSV with adjacent JSON audit")
    parser.add_argument("--horizon", type=int, choices=(24, 48))
    parser.add_argument("--observations", type=Path, help="UTC-hourly observed normalized power CSV")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()
    forecast = load_forecast(args.forecast, args.horizon)
    observed = load_observations(args.observations, forecast) if args.observations else None
    print(make_report(forecast, args.output, observed))


if __name__ == "__main__":
    main()
```

## serve_preview.py

```python
"""Local dashboard server with an on-demand Matplotlib report endpoint.

Run from any directory: python serve_preview.py --port 8767
The service is bound to localhost and serves only the UI and output artifacts.
"""
from __future__ import annotations

import argparse
from datetime import date
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from scientific_report import load_forecast, load_observations, make_report, validate_forecast

ROOT = Path(__file__).resolve().parent


def report_input(payload: dict, root: Path = ROOT):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    source, day, horizon = payload.get("source"), payload.get("date"), payload.get("horizon")
    if source not in ("synthetic", "demo", "backtest"):
        raise ValueError("Unknown source")
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("Invalid issue date")
    date.fromisoformat(day)
    if type(horizon) is not int or horizon not in (24, 48):
        raise ValueError("Expected a 24h or 48h horizon")
    if source == "synthetic":
        origin = f"{day}T00:00:00+00:00"
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != horizon:
            raise ValueError("Incomplete browser demo")
        records = [{"forecast_origin": origin, "valid_time": row["time"], "turbine_id": f"turbine_{j}",
                    "power_normalized": row[f"power{j}"], "wind_speed_ms": row[f"wind{j}"]}
                   for row in rows for j in (1, 2)]
        audit = {"mode": "synthetic-demo", "trained_model": False, "forecast_origin": origin,
                 "horizon_hours": horizon, "weather_source": "browser synthetic scenario",
                 "physical_validation": {"wind_limits_applied": True}}
        forecast = validate_forecast(records, audit)
    else:
        path = root / "outputs" / source / f"forecast_{day.replace('-', '')}T0000Z.csv"
        forecast = load_forecast(path, horizon)
        expected = "synthetic-demo" if source == "demo" else "historical-backtest"
        if forecast.audit["mode"] != expected or forecast.origin.date().isoformat() != day:
            raise ValueError("Forecast audit does not match requested source/date")
    observed_path = root / "outputs" / "observations.csv"
    observed = load_observations(observed_path, forecast) if source == "backtest" and observed_path.exists() else None
    return source, forecast, observed


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, status: int, value: dict):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/preview/")
            self.end_headers()
            return None
        if path == "/preview/":
            path += "index.html"
        relative = Path(path.lstrip("/"))
        resolved = (ROOT / relative).resolve()
        allowed = (
            (path.startswith("/preview/") and resolved.suffix in (".html", ".js", ".css"))
            or (re.match(r"^/outputs/(demo|backtest)/forecast_\d{8}T\d{4}Z\.(csv|json)$", path) is not None)
            or (path.startswith("/outputs/reports/") and resolved.suffix in (".pdf", ".html", ".svg", ".png", ".json", ".zip", ".csv"))
        )
        if not allowed or ".." in relative.parts or not resolved.is_relative_to(ROOT) or not resolved.is_file():
            self.send_error(404)
            return None
        self.path = path
        return super().send_head()

    def do_POST(self):
        if self.path != "/api/report":
            self.json_response(404, {"error": "Unknown endpoint"})
            return
        port = self.server.server_port
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in [f"http://{host}" for host in hosts]):
            self.json_response(403, {"error": "Local same-origin requests only"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128_000 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json, maximum 128 KB")
            payload = json.loads(self.rfile.read(length))
            source, forecast, observed = report_input(payload)
            folder = make_report(forecast, ROOT / "outputs" / "reports" / source, observed)
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.json_response(200, {"base_url": "/" + folder.relative_to(ROOT).as_posix() + "/", "report": metadata})
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.json_response(422, {"error": f"Проверьте исходные данные: {exc}"})
        except FileNotFoundError:
            self.json_response(404, {"error": "CSV или JSON-аудит прогноза не найден. Сначала выполните расчёт."})
        except Exception as exc:
            self.log_error("Report generation failed: %s", type(exc).__name__)
            self.json_response(500, {"error": "Не удалось создать отчёт. Проверьте сервер и зависимости отчётов."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ALEM WIND: http://127.0.0.1:{args.port}/preview/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
```

## standards_profile.py

```python
"""Reviewed scope of standards for ALEM WIND, not a conformity certificate.

The project forecasts wind-turbine power. Measurement standards do not certify
forecast accuracy, and selecting a report standard does not complete its layout.
"""
from __future__ import annotations


def wind_standards_profile() -> dict:
    """Return a fresh, versioned profile with explicitly limited implementation status."""
    return {
        "profile_version": "wind-references-2026-09-23-v1",
        "reviewed_on": "2026-09-23",
        "equipment": "wind_turbine",
        "report_purpose": "hourly_power_forecast_analysis",
        "conformity_claim": False,
        "conformity_status": "not_assessed",
        "statement": "Подобраны нормативные ориентиры для ВЭС. Полное соответствие стандартам не установлено.",
        "selected": [
            {
                "designation": "ГОСТ 7.32-2017",
                "title": "Отчёт о научно-исследовательской работе. Структура и правила оформления",
                "role": "research_report_format",
                "status": "reference_selected_layout_not_fully_implemented",
                "application": "Основа для структуры и оформления отчёта о НИР.",
                "implementation": "Текущий PDF — аналитический отчёт. Полная вёрстка и реквизиты отчёта о НИР ещё не реализованы и не проверены.",
                "source": "https://protect.gost.ru/gost/details/7d280e43-7036-4a69-8e6e-d15867028343",
                "kazakhstan_catalog_status": "listed_as_active",
                "kazakhstan_source": "https://new-shop.ksm.kz/catalog/?PAGEN_1=346&arrFilter_pf%5BCATEGORY%5D=&page=360&set_filter=Y&view=cards",
            },
            {
                "designation": "IEC 61400-12-1:2022 + COR1:2025",
                "title": "Измерение энергетических характеристик ветроустановок",
                "role": "power_performance_measurements",
                "status": "reference_selected_measurements_not_performed",
                "application": "Методический ориентир для будущих измерений характеристики отдельной ВЭУ и оценки неопределённости.",
                "implementation": "Испытания по IEC не выполнялись. Прогнозные точки P(v) и MAE/RMSE модели не заменяют измеренную кривую и бюджет неопределённости.",
                "source": "https://webstore.iec.ch/en/publication/68499",
                "corrigendum_source": "https://webstore.iec.ch/en/publication/106400",
                "kazakhstan_adoption": "not_verified",
            },
        ],
        "related_reference": {
            "designation": "ГОСТ Р 54418.12.1-2011 (МЭК 61400-12-1:2005)",
            "role": "russian_national_reference_only",
            "note": "Российский национальный документ на основе редакции IEC 2005 года; не подменяет IEC 2022 и не назначен обязательным для Казахстана.",
            "source": "https://protect.gost.ru/gost/details/15493e9c-75f3-4c7d-80fa-042f908fe2b3",
        },
        "excluded": [
            {
                "designation": "ГОСТ 3618-2016",
                "reason": "Область — стационарные паровые турбины до 50 МВт; оборудование проекта — ветроустановки.",
                "source": "https://protect.gost.ru/gost/details/07e4e8d9-b4b7-4059-9140-50dfb72bfbe3",
            },
            {
                "designation": "ГОСТ 24278-2016",
                "reason": "Область — стационарные паротурбинные установки ТЭС 50–1600 МВт; к ВЭС проекта не относится.",
                "source": "https://protect.gost.ru/gost/details/06e66c55-66ea-4d7c-83c3-e20cd141a1a0",
            },
        ],
        "open_items": [
            "Подтвердить применяемую в Казахстане редакцию стандарта испытаний и требования заказчика.",
            "Подготовить полный шаблон НИР по ГОСТ 7.32: структуру, вёрстку и организационные реквизиты; выполнить нормоконтроль.",
            "Для измерительной программы получить паспорт ВЭУ, номинальную мощность, геометрию и высоту ступицы.",
            "Получить измерения ветра и мощности, сведения о средствах измерений, калибровке, площадке и влияющих условиях.",
            "Разработать и выполнить программу испытаний с оценкой неопределённости по полному тексту выбранной редакции IEC.",
        ],
    }
```
