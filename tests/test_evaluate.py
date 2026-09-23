"""One chronological scoring check with a missing target hour."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_agent import CSV_COLUMNS
from evaluate import comparison_hours, evaluate, evaluation_report


class EvaluationTests(unittest.TestCase):
    def test_scores_only_later_complete_hours_and_uses_past_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecasts = root / "forecasts"
            forecasts.mkdir()
            origin = pd.Timestamp("2026-01-30T00:00:00Z")
            (forecasts / "run.json").write_text(json.dumps({
                "mode": "historical-backtest", "horizon_hours": 24,
                "source_data_timezone": "UTC",
                "status": "completed", "submission_complete": True,
                "daily_files": ["forecast_20260130T0000Z.csv"],
                "completed_origins": [origin.isoformat()],
            }))
            stem = forecasts / "forecast_20260130T0000Z"
            stem.with_suffix(".json").write_text(json.dumps({
                "mode": "historical-backtest", "trained_model": True,
                "forecast_origin": origin.isoformat(),
            }))
            records = []
            for hour in range(24):
                for turbine in ("turbine_1", "turbine_2"):
                    records.append({
                        "forecast_origin": origin.isoformat(),
                        "valid_time": (origin + pd.Timedelta(hours=hour)).isoformat(),
                        "turbine_id": turbine, "power_normalized": 0.5,
                        "wind_speed_ms": 10,
                    })
            pd.DataFrame(records).to_csv(stem.with_suffix(".csv"), index=False)
            # Unlisted leftovers from a previous run must never enter scoring.
            (forecasts / "forecast_20260129T0000Z.csv").write_text("stale,invalid\n")
            times = pd.date_range(origin - pd.Timedelta(hours=1), periods=25 * 6, freq="10min")
            power = np.where(times < origin, 0.2, 0.8)
            for turbine in ("turbine_1", "turbine_2"):
                frame = pd.DataFrame({
                    "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
                    "wind_speed": 10, "power": power, "temperature": 10,
                })
                frame = frame.loc[~((times == origin + pd.Timedelta(hours=5, minutes=10)))]
                frame.rename(columns={value: key for key, value in CSV_COLUMNS.items()}).to_csv(root / f"{turbine}.csv", index=False)
            result = evaluate(forecasts, [root / "turbine_1.csv", root / "turbine_2.csv"], "UTC")
            self.assertEqual(result.n.sum(), 46)
            self.assertNotIn(6, result.lead_hour.to_list())
            self.assertTrue(np.allclose(result.model_mae, 0.3))
            self.assertTrue(np.allclose(result.persistence_mae, 0.6))
            manifest, hours = comparison_hours(forecasts, [root / "turbine_1.csv", root / "turbine_2.csv"], "UTC")
            report = evaluation_report(manifest, hours)
            self.assertEqual(len(report["rows"]), 48)
            self.assertEqual(report["summary"][0]["missing_observed"], 1)
            self.assertEqual(report["summary"][0]["n"], 23)
            missing = [r for r in report["by_lead"] if r["lead_hour"] == 6]
            self.assertTrue(all(r["n"] == 0 and r["model_mae"] is None for r in missing))
            # A missing reference excludes the entire turbine-origin for ALL methods.
            source = root / "turbine_1.csv"
            original_source = source.read_text()
            pd.read_csv(source).iloc[1:].to_csv(source, index=False)
            _, hours = comparison_hours(forecasts, [source, root / "turbine_2.csv"], "UTC")
            missing_reference = evaluation_report(manifest, hours)["summary"][0]
            self.assertEqual(missing_reference["n"], 0)
            self.assertEqual(missing_reference["excluded"], 24)
            self.assertEqual(missing_reference["missing_persistence"], 24)
            self.assertIsNone(missing_reference["model_rmse"])
            json.dumps(evaluation_report(manifest, hours), allow_nan=False)

            # 48h horizons around a split: Jan 28 is purged, Jan 27 ends at cutoff.
            full_horizon = pd.concat([hours, hours.assign(valid_time=hours.valid_time + pd.Timedelta(days=1), lead_hour=hours.lead_hour + 24)])
            overlapping = pd.concat([full_horizon.assign(
                forecast_origin=full_horizon.forecast_origin + pd.Timedelta(days=offset),
                valid_time=full_horizon.valid_time + pd.Timedelta(days=offset),
            ) for offset in (-3, -2, -1)], ignore_index=True)
            manifest["horizon_hours"] = 48
            split_report = evaluation_report(manifest, overlapping, "2026-01-29T00:00:00Z")
            split_times = {name: {r["valid_time"] for r in split_report["rows"] if r["split"] == name}
                           for name in ("development", "holdout", "purged")}
            self.assertTrue(all(split_times.values()))
            self.assertFalse(split_times["development"] & split_times["holdout"])
            self.assertTrue(all(r["split"] == "purged" for r in split_report["rows"]
                                if r["forecast_origin"].startswith("2026-01-28")))
            with self.assertRaisesRegex(ValueError, "leave complete"):
                evaluation_report(manifest, overlapping, "2026-02-01T00:00:00Z")
            source.write_text(original_source)

            # Later truth may not have arrived yet, as with the supplied CSVs.
            for turbine in ("turbine_1", "turbine_2"):
                source = root / f"{turbine}.csv"
                pd.read_csv(source).iloc[:6].to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "No observed forecast hours"):
                evaluate(forecasts, [root / "turbine_1.csv", root / "turbine_2.csv"], "UTC")

    def test_rejects_incomplete_run_before_loading_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({
                "mode": "historical-backtest", "source_data_timezone": "UTC",
                "status": "failed", "submission_complete": False,
            }))
            with self.assertRaisesRegex(ValueError, "Only completed runs"):
                evaluate(root, [root / "t1.csv", root / "t2.csv"], "UTC")

    def test_rejects_demo_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(json.dumps({"mode": "synthetic-demo"}))
            with self.assertRaisesRegex(ValueError, "historical-backtest"):
                evaluate(root, [root / "t1.csv", root / "t2.csv"], "UTC")


if __name__ == "__main__":
    unittest.main()
