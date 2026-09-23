import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from main import (
    HISTORY_FEATURES, WEATHER_FEATURES, aligned_weather, build_parser, build_training_arrays, hour_index,
    run, simulation_origins, write_daily_submission,
)


def fake_weather(origin, horizon):
    index = hour_index(origin, horizon)
    return [pd.DataFrame({
        "wind_speed": np.full(horizon, 8 + turbine),
        "issued_at_upper_bound": origin - pd.Timedelta(hours=12),
        "available_at_upper_bound": origin - pd.Timedelta(hours=4),
        "lead_days": np.full(horizon, 2),
    }, index=index) for turbine in range(2)]


class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2026-02-01", tz="UTC")
        self.history = [pd.DataFrame(
            {"power": np.full(8 * 24, 0.2 + turbine * 0.1)},
            index=hour_index(self.origin, 8 * 24, -8 * 24),
        ) for turbine in range(2)]

    def test_all_labels_are_strictly_before_current_origin(self):
        requested_origins = []

        def loader(origin):
            requested_origins.append(origin)
            return fake_weather(origin, 48)

        batch = build_training_arrays(
            self.history, loader, self.origin, lookback=2, horizon=48,
            train_days=5, min_samples=1, history_features=("power",),
            weather_features=("wind_speed",),
        )
        self.assertEqual(batch.history.shape, (4, 2, 2))
        self.assertEqual(batch.weather.shape, (4, 48, 2))
        self.assertEqual(batch.targets.shape, (4, 48, 2))
        self.assertEqual(batch.latest_target, self.origin - pd.Timedelta(hours=1))
        self.assertEqual(batch.origins, requested_origins)
        self.assertTrue(all(origin + pd.Timedelta(hours=47) < self.origin for origin in batch.origins))
        self.assertTrue(all(origin < self.origin for origin in requested_origins))

    def test_future_observations_are_rejected(self):
        self.history[0].loc[self.origin, "power"] = 0.7
        with self.assertRaisesRegex(ValueError, "Observation leakage"):
            build_training_arrays(self.history, lambda s: fake_weather(s, 48), self.origin)

    def test_incomplete_history_is_excluded_without_gap_filling(self):
        missing_hour = self.origin - pd.Timedelta(days=5, hours=1)
        self.history[0] = self.history[0].drop(missing_hour)
        batch = build_training_arrays(
            self.history, lambda s: fake_weather(s, 48), self.origin,
            lookback=2, horizon=48, train_days=5, min_samples=1,
            history_features=("power",), weather_features=("wind_speed",),
        )
        self.assertEqual(batch.skipped_incomplete_samples, 1)
        self.assertEqual(len(batch.origins), 3)
        self.assertTrue(np.isfinite(batch.history).all())

    def test_future_weather_publication_is_rejected(self):
        weather = fake_weather(self.origin, 24)
        weather[1]["available_at_upper_bound"] = self.origin + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "Weather leakage"):
            aligned_weather(weather, self.origin, 24, ("wind_speed",))

    def test_missing_weather_hour_is_rejected(self):
        weather = fake_weather(self.origin, 24)
        weather[0] = weather[0].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            aligned_weather(weather, self.origin, 24, ("wind_speed",))

    def test_simulation_uses_explicit_local_day_boundaries(self):
        origins = simulation_origins("2026-02-01", "2026-02-28", "Asia/Almaty")
        self.assertEqual(len(origins), 28)
        self.assertEqual(origins[0], pd.Timestamp("2026-01-31T19:00:00Z"))


class DemoOutputTests(unittest.TestCase):
    def test_full_february_demo_retains_march_forecast_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["--demo", "--output", directory])
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            path = Path(directory)
            self.assertEqual(len(list(path.glob("forecast_*.csv"))), 28)
            self.assertEqual(len(list(path.glob("forecast_*.json"))), 28)
            last = pd.read_csv(path / "forecast_20260228T0000Z.csv")
            self.assertEqual(len(last), 96)
            self.assertEqual(pd.to_datetime(last.valid_time, utc=True).max(), pd.Timestamp("2026-03-01T23:00:00Z"))
            self.assertTrue(last.power_normalized.between(0, 1).all())
            audit = json.loads((path / "forecast_20260228T0000Z.json").read_text())
            self.assertEqual(audit["mode"], "synthetic-demo")
            self.assertFalse(audit["trained_model"])

    def test_demo_cannot_overwrite_real_artifacts(self):
        origin = pd.Timestamp("2026-02-01", tz="UTC")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            power, wind = np.full((24, 2), 0.5), np.full((24, 2), 8.0)
            csv_path = write_daily_submission(path, origin, power, wind, {"mode": "historical-backtest"})
            original = csv_path.read_text()
            with self.assertRaisesRegex(ValueError, "separate directories"):
                write_daily_submission(path, origin, power, wind, {"mode": "synthetic-demo"})
            self.assertEqual(csv_path.read_text(), original)


class HistoryPolicyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.first_origin = pd.Timestamp("2026-02-01T00:00:00Z")
        self.history_requests = []
        self.weather_requests = []
        self.fit_calls = []
        self.predict_calls = []
        owner = self

        @dataclass
        class FakeConfig:
            history_features: int
            weather_features: int
            num_turbines: int
            lookback: int
            horizon: int
            epochs: int
            batch_size: int
            device: str
            seed: int

        class FakeDataAgent:
            def __init__(self, **kwargs):
                pass

            def load_history(self, path, current_date):
                owner.history_requests.append(current_date)
                # Fixture intentionally has no February observations, like the supplied CSVs.
                index = hour_index(owner.first_origin, 8 * 24, -8 * 24)
                frame = pd.DataFrame({feature: np.full(len(index), 0.4) for feature in HISTORY_FEATURES}, index=index)
                return frame.loc[frame.index < current_date]

            def fetch_forecast(self, latitude, longitude, current_date, horizon):
                owner.weather_requests.append(current_date)
                forecast = fake_weather(current_date, horizon)[0]
                for feature in WEATHER_FEATURES:
                    if feature not in forecast:
                        forecast[feature] = 0.1
                forecast["wind_speed"] = 8 + (current_date - owner.first_origin).days / 100
                return forecast

        class FakeModelAgent:
            def __init__(self, config):
                self.config = config

            def fit(self, history, weather, targets):
                owner.fit_calls.append((history.copy(), weather.copy(), targets.copy()))
                return {"training_mse": 0.1}

            def predict(self, history, weather):
                owner.predict_calls.append((history.copy(), weather.copy()))
                return np.full((self.config.horizon, 2), 0.5)

            def save(self, path):
                path.write_text("mock checkpoint", encoding="utf-8")

        self.modules = {
            "data_agent": SimpleNamespace(DataAgent=FakeDataAgent),
            "model_agent": SimpleNamespace(ModelAgent=FakeModelAgent, ModelConfig=FakeConfig),
        }

    def arguments(self, directory, policy):
        return build_parser().parse_args([
            "--turbine-1", "fixture1.csv", "--turbine-2", "fixture2.csv", "--data-timezone", "UTC",
            "--output", directory, "--history-policy", policy, "--end", "2026-02-03",
            "--horizon", "24", "--lookback", "2", "--train-days", "3", "--min-train-samples", "2",
        ])

    def test_frozen_context_locks_training_but_updates_daily_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", self.modules), contextlib.redirect_stdout(io.StringIO()):
                run(self.arguments(directory, "frozen"))
            self.assertEqual(self.history_requests, [self.first_origin, self.first_origin])
            self.assertEqual(len(self.fit_calls), 1)
            self.assertEqual(len(self.predict_calls), 3)
            np.testing.assert_array_equal(self.predict_calls[0][0], self.predict_calls[-1][0])
            self.assertNotEqual(self.predict_calls[0][1][0, 0], self.predict_calls[-1][1][0, 0])
            self.assertIn(self.first_origin + pd.Timedelta(days=2), self.weather_requests)
            path = Path(directory)
            self.assertEqual(len(list((path / "checkpoints").glob("*.pt"))), 1)
            audit = json.loads((path / "forecast_20260203T0000Z.json").read_text())
            self.assertEqual(audit["history_policy"], "frozen")
            self.assertEqual(audit["context_origin"], self.first_origin.isoformat())
            self.assertEqual(audit["observation_cutoff_exclusive"], self.first_origin.isoformat())
            self.assertEqual(audit["context_age_hours"], 48)
            self.assertEqual(audit["training"]["latest_target_time"], "2026-01-31T23:00:00+00:00")

    def test_expanding_requires_new_observed_scada(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", self.modules), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Incomplete inference history"):
                    run(self.arguments(directory, "expanding"))
            self.assertEqual(len(list(Path(directory).glob("forecast_*.csv"))), 1)
            self.assertEqual(self.history_requests[-1], self.first_origin + pd.Timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
