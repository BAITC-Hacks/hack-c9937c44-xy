"""Score historical forecasts against later SCADA, with two simple baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent import DataAgent
from main import TURBINES, _atomic_text, hour_index, utc_timestamp


def evaluate(forecast_dir: Path, sources: list[Path], data_timezone: str) -> pd.DataFrame:
    manifest = json.loads((forecast_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "historical-backtest":
        raise ValueError("Only historical-backtest forecasts can be evaluated")
    if manifest.get("source_data_timezone") != data_timezone:
        raise ValueError("Evaluation timezone must match the forecast run")
    if len(sources) != len(TURBINES):
        raise ValueError("Exactly two turbine CSVs are required")
    paths = sorted(forecast_dir.glob("forecast_*.csv"))
    if not paths:
        raise ValueError("No forecast CSVs found")

    forecasts = []
    origins = set()
    horizon = manifest["horizon_hours"]
    if horizon not in (24, 48):
        raise ValueError("Forecast horizon must be 24 or 48 hours")
    for path in paths:
        audit = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if audit.get("mode") != "historical-backtest" or audit.get("trained_model") is not True:
            raise ValueError(f"Not a trained historical forecast: {path}")
        origin = utc_timestamp(audit["forecast_origin"])
        if origin in origins:
            raise ValueError(f"Duplicate forecast origin: {origin}")
        origins.add(origin)
        frame = pd.read_csv(path)
        expected_columns = {"forecast_origin", "valid_time", "turbine_id", "power_normalized", "wind_speed_ms"}
        if set(frame.columns) != expected_columns:
            raise ValueError(f"Unexpected forecast columns: {path}")
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"], utc=True, errors="raise")
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="raise")
        frame["power_normalized"] = pd.to_numeric(frame["power_normalized"], errors="raise")
        frame["wind_speed_ms"] = pd.to_numeric(frame["wind_speed_ms"], errors="raise")
        if not frame.forecast_origin.eq(origin).all() or len(frame) != horizon * len(TURBINES):
            raise ValueError(f"Incomplete or mixed forecast origin: {path}")
        for turbine in TURBINES:
            times = frame.loc[frame.turbine_id == turbine.turbine_id, "valid_time"]
            if not times.is_unique or set(times) != set(hour_index(origin, horizon)):
                raise ValueError(f"Incomplete forecast hours for {turbine.turbine_id}: {path}")
        power = frame.power_normalized.to_numpy()
        wind = frame.wind_speed_ms.to_numpy()
        if not np.isfinite(power).all() or not np.isfinite(wind).all() or (power < 0).any() or (power > 1).any() or (wind < 0).any():
            raise ValueError(f"Invalid forecast values: {path}")
        forecasts.append(frame)

    cutoff = max(origins) + pd.Timedelta(hours=horizon)
    agent = DataAgent(backend="pandas", data_timezone=data_timezone)
    histories = [agent.load_history(path, cutoff) for path in sources]
    rows = []
    for frame in forecasts:
        origin = frame.forecast_origin.iloc[0]
        for turbine, history in zip(TURBINES, histories):
            previous_hour = origin - pd.Timedelta(hours=1)
            previous_power = history.power.get(previous_hour, np.nan)
            if not np.isfinite(previous_power):
                continue
            part = frame.loc[frame.turbine_id == turbine.turbine_id].copy()
            part["observed"] = history.power.reindex(pd.DatetimeIndex(part.valid_time)).to_numpy()
            part = part.loc[part.observed.notna()]
            part["persistence"] = previous_power
            wind = part.wind_speed_ms.to_numpy()
            part["power_curve"] = np.where(
                (wind < 3) | (wind > 25), 0,
                np.clip((wind**3 - 3**3) / (12**3 - 3**3), 0, 1),
            )
            part["lead_hour"] = ((part.valid_time - origin) / pd.Timedelta(hours=1)).astype(int) + 1
            rows.append(part)
    if not rows:
        raise ValueError("No observed forecast hours with a complete prior-hour persistence reference")
    scored = pd.concat(rows, ignore_index=True)
    if ((scored.observed < 0) | (scored.observed > 1)).any():
        raise ValueError("Observed power must be normalized to [0, 1]")
    result = []
    for (turbine_id, lead_hour), group in scored.groupby(["turbine_id", "lead_hour"], sort=True):
        row = {"turbine_id": turbine_id, "lead_hour": lead_hour, "n": len(group)}
        for name, column in (("model", "power_normalized"), ("persistence", "persistence"), ("power_curve", "power_curve")):
            error = group[column].to_numpy() - group.observed.to_numpy()
            row[f"{name}_mae"] = float(np.mean(np.abs(error)))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result.append(row)
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True, help="Directory of real forecast CSV/JSON files")
    parser.add_argument("--turbine-1", type=Path, required=True)
    parser.add_argument("--turbine-2", type=Path, required=True)
    parser.add_argument("--data-timezone", required=True, help="Confirmed IANA timezone of source CSV timestamps")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.csv"))
    args = parser.parse_args()
    result = evaluate(args.forecasts, [args.turbine_1, args.turbine_2], args.data_timezone)
    _atomic_text(args.output, result.to_csv(index=False, float_format="%.7f"))
    print(f"Scored {int(result.n.sum())} turbine-hours across {len(result)} turbine/lead groups -> {args.output}")


if __name__ == "__main__":
    main()
