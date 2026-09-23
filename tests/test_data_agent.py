"""CPU contract tests: cutoff isolation, real hourly gaps and forecast provenance."""

import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from data_agent import (
    CSV_COLUMNS, DataAgent, HISTORY_FEATURES, WEATHER_FEATURES,
    fetch_weather_forecast,
)


class DataAgentTests(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path(__file__).resolve().parent
        self.directory = self.fixture_root / f"data-agent-fixture-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.clean_fixture)
        self.agent = DataAgent(backend="pandas", cache_dir=self.directory / "weather")
        self.cutoff = pd.Timestamp("2026-02-01", tz="UTC")

    def clean_fixture(self):
        # Validate the resolved deletion boundary before recursive cleanup.
        if self.directory.resolve().parent != self.fixture_root:
            raise RuntimeError("Fixture cleanup path escaped tests directory")
        shutil.rmtree(self.directory)

    def write_history(self, times, power=None, path="history.csv"):
        times = pd.DatetimeIndex(times)
        frame = pd.DataFrame({
            "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed": np.full(len(times), 8.0),
            "power": np.full(len(times), 0.4) if power is None else power,
            "temperature": np.full(len(times), 10.0),
        })
        frame = frame.rename(columns={value: key for key, value in CSV_COLUMNS.items()})
        csv_path = self.directory / path
        frame.to_csv(csv_path, index=False)
        return csv_path

    def payload(self, start=None, hours=48):
        times = pd.date_range(self.cutoff if start is None else start, periods=hours, freq="h")
        hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
        units = {"time": "iso8601"}
        for day in (1, 2, 3):
            for name, unit, base in (
                ("wind_speed_10m", "m/s", 5.0),
                ("temperature_2m", "°C", 10.0),
                ("surface_pressure", "hPa", 1000.0),
            ):
                key = f"{name}_previous_day{day}"
                hourly[key] = [base + day] * len(times)
                units[key] = unit
        return {"utc_offset_seconds": 0, "hourly": hourly, "hourly_units": units}

    def test_history_filters_future_before_numeric_conversion(self):
        times = pd.date_range("2026-01-30", periods=3 * 24 * 6, freq="10min")
        power = np.full(len(times), 0.4, dtype=object)
        path = self.write_history(times, power)
        baseline = self.agent.load_history(path, self.cutoff)
        power[times >= self.cutoff.tz_localize(None)] = "future nonsense"
        changed = self.agent.load_history(self.write_history(times, power), self.cutoff)
        pd.testing.assert_frame_equal(baseline, changed)
        self.assertEqual(len(baseline), 48)
        self.assertEqual(tuple(baseline.columns), HISTORY_FEATURES)
        self.assertEqual(baseline.index[-1], self.cutoff - pd.Timedelta(hours=1))
        self.assertEqual(str(baseline.index.tz), "UTC")
        self.assertAlmostEqual(baseline.iloc[-1].power_roll_24, 0.4)

    def test_hour_is_equal_weight_mean_of_six_unique_readings(self):
        times = pd.date_range("2026-01-31T23:00", periods=6, freq="10min")
        times = times.append(times[-1:])
        result = self.agent.load_history(self.write_history(times, [0, .1, .2, .3, .4, .5, .5]), self.cutoff)
        self.assertAlmostEqual(result.iloc[0].power, .25)

    def test_conflicting_duplicate_fails(self):
        times = pd.date_range("2026-01-31T23:00", periods=6, freq="10min")
        path = self.write_history(times.append(times[-1:]), [0, .1, .2, .3, .4, .5, .9])
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.agent.load_history(path, self.cutoff)

    def test_missing_hours_and_incomplete_hours_are_not_filled(self):
        times = pd.date_range("2026-01-31", periods=24 * 6, freq="10min")
        times = times[(times.hour != 12) & ~((times.hour == 13) & (times.minute == 20))]
        result = self.agent.load_history(self.write_history(times), self.cutoff)
        self.assertEqual(len(result), 24)
        self.assertTrue(result.loc["2026-01-31T12:00:00Z", "power":"temperature"].isna().all())
        self.assertTrue(result.loc["2026-01-31T13:00:00Z", "power":"temperature"].isna().all())
        self.assertTrue(np.isnan(result.loc["2026-01-31T18:00:00Z", "power_roll_6"]))
        self.assertAlmostEqual(result.loc["2026-01-31T19:00:00Z", "power_roll_6"], .4)

    def test_partial_cutoff_hour_is_excluded(self):
        times = pd.date_range("2026-01-31T23:00", periods=12, freq="10min")
        result = self.agent.load_history(self.write_history(times), self.cutoff + pd.Timedelta(minutes=30))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index[-1], self.cutoff - pd.Timedelta(hours=1))

    def test_explicit_source_timezone_converts_to_utc(self):
        times = pd.date_range("2026-02-01T04:00", periods=6, freq="10min")
        agent = DataAgent(backend="pandas", data_timezone="Asia/Almaty")
        result = agent.load_history(self.write_history(times), self.cutoff)
        self.assertEqual(result.index[0], pd.Timestamp("2026-01-31T23:00:00Z"))

    def test_naive_cutoff_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.agent.fetch_forecast(51.04, 71.46, "2026-02-01")

    def test_forecast_selects_only_available_fixed_lead_products(self):
        with patch.object(self.agent, "_request_json", return_value=self.payload()) as request:
            result = self.agent.fetch_forecast(51.04, 71.46, self.cutoff)
        self.assertEqual(tuple(result.columns[:len(WEATHER_FEATURES)]), WEATHER_FEATURES)
        self.assertTrue((result.available_at_upper_bound <= self.cutoff).all())
        self.assertEqual(result.iloc[16].lead_days, 1)
        self.assertEqual(result.iloc[17].lead_days, 2)
        self.assertEqual(result.iloc[40].lead_days, 2)
        self.assertEqual(result.iloc[41].lead_days, 3)
        self.assertEqual(result.iloc[17].wind_speed, 7.0)
        self.assertAlmostEqual(result.iloc[0].air_density, 1001 * 100 / (287.05 * (11 + 273.15)))
        params = request.call_args.args[0]
        self.assertEqual(params["models"], "gfs_global")
        self.assertEqual(params["wind_speed_unit"], "ms")
        self.assertNotIn("previous_day0", params["hourly"])

    def test_function_entrypoint_preserves_availability_policy_and_cache(self):
        cache_dir = self.directory / "function-weather"
        with patch.object(DataAgent, "_request_json", return_value=self.payload()) as request:
            first = fetch_weather_forecast(
                51.04, 71.46, self.cutoff, horizon=24,
                cache_dir=cache_dir, publication_lag_hours=32,
            )
            second = fetch_weather_forecast(
                51.04, 71.46, self.cutoff, horizon=24,
                cache_dir=cache_dir, publication_lag_hours=32,
            )
        request.assert_called_once()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(first.attrs["publication_lag_hours"], 32)
        self.assertEqual(first.iloc[0].lead_days, 2)
        self.assertEqual(first.iloc[-1].lead_days, 3)
        self.assertTrue((first.available_at_upper_bound <= self.cutoff).all())
        self.assertEqual(request.call_args.args[0]["latitude"], 51.04)
        self.assertEqual(request.call_args.args[0]["longitude"], 71.46)

    def test_function_entrypoint_reuses_supplied_client_configuration(self):
        self.agent.publication_lag_hours = 32
        self.agent._http_session = Mock()
        with patch.object(self.agent, "_request_json", return_value=self.payload()):
            result = fetch_weather_forecast(
                51.04, 71.46, self.cutoff, horizon=24, agent=self.agent,
            )
        self.assertEqual(result.attrs["publication_lag_hours"], 32)
        self.assertEqual(result.iloc[-1].lead_days, 3)
        self.assertTrue(self.agent.cache_dir.exists())
        self.agent._http_session.close.assert_not_called()

    def test_cache_roundtrip_and_provenance_validation(self):
        with patch.object(self.agent, "_request_json", return_value=self.payload()) as request:
            first = self.agent.fetch_forecast(51.04, 71.46, self.cutoff)
            second = self.agent.fetch_forecast(51.04, 71.46, self.cutoff)
        request.assert_called_once()
        pd.testing.assert_frame_equal(first, second)
        cache_path = next(self.agent.cache_dir.glob("*.json"))
        envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        envelope["identity"]["cutoff"] = "2099-01-01T00:00:00+00:00"
        cache_path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.agent.fetch_forecast(51.04, 71.46, self.cutoff)

    def test_cache_checksum_validation(self):
        with patch.object(self.agent, "_request_json", return_value=self.payload()):
            self.agent.fetch_forecast(51.04, 71.46, self.cutoff)
        cache_path = next(self.agent.cache_dir.glob("*.json"))
        envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        envelope["response"]["hourly"]["wind_speed_10m_previous_day1"][0] = 999
        cache_path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.agent.fetch_forecast(51.04, 71.46, self.cutoff)

    def test_bad_forecasts_fail_closed_without_cache(self):
        cases = []
        missing_hour = self.payload(hours=47)
        cases.append(missing_hour)
        wrong_units = self.payload()
        wrong_units["hourly_units"]["wind_speed_10m_previous_day1"] = "km/h"
        cases.append(wrong_units)
        null_value = self.payload()
        null_value["hourly"]["temperature_2m_previous_day1"][0] = None
        cases.append(null_value)
        duplicate = self.payload()
        duplicate["hourly"]["time"][1] = duplicate["hourly"]["time"][0]
        cases.append(duplicate)
        non_utc = self.payload()
        non_utc["utc_offset_seconds"] = 18000
        cases.append(non_utc)
        for payload in cases:
            with self.subTest(payload=payload):
                with patch.object(self.agent, "_request_json", return_value=copy.deepcopy(payload)):
                    with self.assertRaises(ValueError):
                        self.agent.fetch_forecast(51.04, 71.46, self.cutoff)
        self.assertFalse(self.agent.cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
