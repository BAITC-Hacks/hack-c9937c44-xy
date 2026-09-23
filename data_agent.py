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
