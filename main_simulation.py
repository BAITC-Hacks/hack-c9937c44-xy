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
            manifest["training_window"] = {"train_days": args.train_days, "min_train_samples": args.min_train_samples}
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
