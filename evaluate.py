"""Score historical forecasts against later SCADA, with two simple baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent import DataAgent
from main_simulation import TURBINES, _atomic_text, hour_index, utc_timestamp


def comparison_hours(forecast_dir: Path, sources: list[Path], data_timezone: str) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((forecast_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "historical-backtest":
        raise ValueError("Only historical-backtest forecasts can be evaluated")
    if manifest.get("source_data_timezone") != data_timezone:
        raise ValueError("Evaluation timezone must match the forecast run")
    if manifest.get("status") != "completed" or manifest.get("submission_complete") is not True:
        raise ValueError("Only completed runs can be evaluated; rerun legacy or incomplete forecasts")
    if len(sources) != len(TURBINES):
        raise ValueError("Exactly two turbine CSVs are required")
    names = manifest.get("daily_files")
    if not isinstance(names, list) or not names:
        raise ValueError("Completed run must list its daily forecast files")
    if any(not isinstance(name, str) or Path(name).name != name
           or "/" in name or "\\" in name
           or not name.startswith("forecast_") or not name.endswith(".csv") for name in names):
        raise ValueError("Daily forecast entries must be local CSV filenames")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate daily forecast file in manifest")
    paths = [forecast_dir / name for name in names]
    completed = manifest.get("completed_origins")
    if not isinstance(completed, list) or len(completed) != len(paths):
        raise ValueError("Completed origins must match daily forecast files")
    expected_origins = {utc_timestamp(value) for value in completed}
    if len(expected_origins) != len(paths):
        raise ValueError("Duplicate completed origin in manifest")

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

    if origins != expected_origins:
        raise ValueError("Forecast origins do not match the completed run manifest")
    cutoff = max(origins) + pd.Timedelta(hours=horizon)
    agent = DataAgent(backend="pandas", data_timezone=data_timezone)
    histories = [agent.load_history(path, cutoff) for path in sources]
    rows = []
    for frame in forecasts:
        origin = frame.forecast_origin.iloc[0]
        for turbine, history in zip(TURBINES, histories):
            previous_hour = origin - pd.Timedelta(hours=1)
            previous_power = history.power.get(previous_hour, np.nan)
            part = frame.loc[frame.turbine_id == turbine.turbine_id].copy()
            part["observed"] = history.power.reindex(pd.DatetimeIndex(part.valid_time)).to_numpy()
            part["observed_wind_ms"] = history.wind_speed.reindex(pd.DatetimeIndex(part.valid_time)).to_numpy()
            part["persistence"] = previous_power
            wind = part.wind_speed_ms.to_numpy()
            part["power_curve"] = np.where(
                (wind < 3) | (wind > 25), 0,
                np.clip((wind**3 - 3**3) / (12**3 - 3**3), 0, 1),
            )
            part["lead_hour"] = ((part.valid_time - origin) / pd.Timedelta(hours=1)).astype(int) + 1
            part["eligible"] = part.observed.notna() & part.persistence.notna()
            rows.append(part)
    scored = pd.concat(rows, ignore_index=True)
    if ((scored.observed < 0) | (scored.observed > 1)).any():
        raise ValueError("Observed power must be normalized to [0, 1]")
    if ((scored.persistence < 0) | (scored.persistence > 1)).any():
        raise ValueError("Persistence power must be normalized to [0, 1]")
    return manifest, scored


def metrics(group: pd.DataFrame) -> dict:
    paired = group.loc[group.eligible]
    row = {
        "n": len(paired), "expected": len(group), "excluded": len(group) - len(paired),
        "missing_observed": int(group.observed.isna().sum()),
        "missing_persistence": int(group.persistence.isna().sum()),
        "origins": int(group.forecast_origin.nunique()),
        "first_origin": group.forecast_origin.min().isoformat(),
        "last_origin": group.forecast_origin.max().isoformat(),
    }
    for name, column in (("model", "power_normalized"), ("persistence", "persistence"), ("power_curve", "power_curve")):
        error = paired[column].to_numpy() - paired.observed.to_numpy()
        row[f"{name}_mae"] = float(np.mean(np.abs(error))) if len(error) else None
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2))) if len(error) else None
    if len(paired):
        row["forecast_wind_mean_ms"] = float(paired.wind_speed_ms.mean())
        row["observed_wind_mean_ms"] = float(paired.observed_wind_ms.mean())
    return row


def evaluate(forecast_dir: Path, sources: list[Path], data_timezone: str) -> pd.DataFrame:
    """Keep the original per-lead CSV contract for existing callers."""
    _, hours = comparison_hours(forecast_dir, sources, data_timezone)
    scored = hours.loc[hours.eligible]
    if scored.empty:
        raise ValueError("No observed forecast hours with a complete prior-hour persistence reference")
    result = []
    for (turbine_id, lead_hour), group in scored.groupby(["turbine_id", "lead_hour"], sort=True):
        row = {"turbine_id": turbine_id, "lead_hour": lead_hour, "n": len(group)}
        row.update({key: value for key, value in metrics(group).items() if key.endswith(("_mae", "_rmse"))})
        result.append(row)
    return pd.DataFrame(result)


def evaluation_report(manifest: dict, hours: pd.DataFrame, holdout_start: str | None = None) -> dict:
    """Split whole forecast horizons, purging origins that cross the boundary."""
    hours = hours.copy()
    hours["split"] = "all"
    boundary = utc_timestamp(holdout_start) if holdout_start else None
    if boundary is not None:
        if pd.isna(boundary) or boundary != boundary.floor("h"):
            raise ValueError("Holdout start must be a valid whole-hour timestamp")
        ends = hours.forecast_origin + pd.Timedelta(hours=manifest["horizon_hours"])
        hours["split"] = np.where(hours.forecast_origin >= boundary, "holdout",
                                  np.where(ends <= boundary, "development", "purged"))
        if not {"development", "holdout"}.issubset(set(hours.split)):
            raise ValueError("Holdout boundary must leave complete development and holdout origins")
    summaries, leads, daily = [], [], []
    for (split, turbine), group in hours.groupby(["split", "turbine_id"], sort=True):
        summaries.append({"split": split, "turbine_id": turbine, **metrics(group)})
    for (split, turbine, lead), group in hours.groupby(["split", "turbine_id", "lead_hour"], sort=True):
        leads.append({"split": split, "turbine_id": turbine, "lead_hour": int(lead), **metrics(group)})
    for origin, group in hours.groupby("forecast_origin", sort=True):
        for horizon in (24, 48):
            if horizon > manifest["horizon_hours"]:
                continue
            for turbine in ("both", *(t.turbine_id for t in TURBINES)):
                selected = group.loc[group.lead_hour <= horizon]
                if turbine != "both":
                    selected = selected.loc[selected.turbine_id == turbine]
                daily.append({"forecast_origin": origin.isoformat(), "horizon_hours": horizon,
                              "turbine_id": turbine, "split": group.split.iloc[0], **metrics(selected)})
    for column in ("forecast_origin", "valid_time"):
        hours[column] = hours[column].map(lambda value: value.isoformat())
    return {
        "schema_version": 1, "run_created_at": manifest.get("created_at"),
        "source_data_timezone": manifest["source_data_timezone"],
        "horizon_hours": manifest["horizon_hours"], "completed_origins": manifest["completed_origins"],
        "holdout_start": boundary.isoformat() if boundary is not None else None,
        "protocol": "Fixed configuration; daily retraining on observations before each origin. Crossing horizons are purged. Overlapping forecast hours are separate cases, not independent observations.",
        "assumptions": [
            f"Часовой пояс CSV {manifest['source_data_timezone']} — допущение; требует подтверждения.",
            "Координаты, высоты и задержка SCADA не подтверждены; метка считается началом интервала.",
            f"Доступность архивного прогноза: допущение задержки {manifest.get('publication_lag_hours_assumption', 8)} ч.",
            "Кубическая кривая на ветре 10 м не калибрована. Метрики предварительные.",
        ],
        "summary": summaries, "by_lead": leads, "daily_metrics": daily,
        "rows": json.loads(hours.to_json(orient="records", double_precision=15)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True, help="Directory of real forecast CSV/JSON files")
    parser.add_argument("--turbine-1", type=Path, required=True)
    parser.add_argument("--turbine-2", type=Path, required=True)
    parser.add_argument("--data-timezone", required=True, help="Assumed IANA timezone of source CSV timestamps")
    parser.add_argument("--holdout-start", help="Timezone-aware first holdout origin; crossing horizons are purged")
    parser.add_argument("--output", type=Path, help="Default: evaluation.csv beside run.json, ready for the preview")
    args = parser.parse_args()
    args.output = args.output or args.forecasts / "evaluation.csv"
    manifest, hours = comparison_hours(args.forecasts, [args.turbine_1, args.turbine_2], args.data_timezone)
    report = evaluation_report(manifest, hours, args.holdout_start)
    result = pd.DataFrame(report["by_lead"])
    _atomic_text(args.output, result.to_csv(index=False, float_format="%.7f"))
    _atomic_text(args.output.with_suffix(".json"), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic_text(args.output.with_name(args.output.stem + "-hours.csv"), pd.DataFrame(report["rows"]).to_csv(index=False, float_format="%.7f"))
    print(f"Scored {int(result.n.sum())}/{len(hours)} paired turbine-hours -> {args.output}")
    print(pd.DataFrame(report["summary"])[["split", "turbine_id", "n", "expected", "model_mae", "model_rmse", "persistence_mae", "persistence_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
