# Complete WPF Python source

Commit: `bbdbb61303b05f42366b70214a6a0d4622948c6b`

See README.md and HANDOFF.md for environment setup, weather availability assumptions, and the missing February SCADA limitation.

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
) -> np.ndarray:
    """Return a copy bounded to [0, 1], with shutdowns outside [3, 25] m/s.

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
    wind: np.ndarray, audit: dict,
) -> Path:
    """Write one complete horizon per origin, retaining March hours on February 28."""
    origin = utc_timestamp(origin)
    bounded = validate_power(power, wind)
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
            "power_range": [0, 1], "cut_in_ms": 3, "cut_out_ms": 25,
            "strict_wind_thresholds": True,
            "forced_zero_count": int(((wind < 3) | (wind > 25)).sum()),
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
                power = validate_power(predict(model, ForecastData(
                    history=context, weather=weather, current_date=origin.to_pydatetime(),
                    available_at_upper_bound=[bound.to_pydatetime()] * args.horizon,
                    context_origin=context_origin.to_pydatetime(),
                )), wind)
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
            path = write_daily_submission(output_dir, origin, power, wind, audit)
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
