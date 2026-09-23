from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from scientific_report import figures, forecast_csv, load_observations, make_report, statistics, validate_forecast
from serve_preview import report_input


def fixture(demo=False, horizon=24):
    origin = datetime(2026, 2, 1, tzinfo=timezone.utc)
    audit = {"mode": "synthetic-demo" if demo else "historical-backtest", "trained_model": not demo,
             "forecast_origin": origin.isoformat(), "horizon_hours": horizon,
             "wind_height_m": 10, "physical_validation": {"wind_limits_applied": demo}}
    records = [{"forecast_origin": origin.isoformat(), "valid_time": (origin + timedelta(hours=i)).isoformat(),
                "turbine_id": f"turbine_{j}", "power_normalized": .4 + .1 * (j - 1), "wind_speed_ms": 8 + j}
               for i in range(horizon) for j in (1, 2)]
    return records, audit


class ReportTests(unittest.TestCase):
    def test_rejects_duplicate_partial_mixed_and_nonfinite(self):
        for change in ("duplicate", "partial", "origin", "nan", "wind", "power", "naive"):
            with self.subTest(change=change):
                rows, audit = fixture()
                if change == "duplicate": rows[1] = rows[0].copy()
                if change == "partial": rows.pop()
                if change == "origin": rows[0]["forecast_origin"] = "2026-01-01T00:00:00Z"
                if change == "nan": rows[0]["power_normalized"] = float("nan")
                if change == "wind": rows[0]["wind_speed_ms"] = -1
                if change == "power": rows[0]["power_normalized"] = 1.1
                if change == "naive": rows[0]["valid_time"] = "2026-02-01T00:00:00"
                with self.assertRaises(ValueError): validate_forecast(rows, audit)

    def test_crop_validates_entire_source_first(self):
        rows, audit = fixture(horizon=48)
        data = validate_forecast(rows, audit, horizon=24)
        self.assertEqual(data.power.shape, (24, 2))
        rows[-1] = rows[-2].copy()
        with self.assertRaises(ValueError): validate_forecast(rows, audit, horizon=24)

    def test_missing_observations_never_become_zero_errors(self):
        data = validate_forecast(*fixture())
        observed = np.full_like(data.power, np.nan)
        observed[0, 0], observed[3, 0] = .2, .6
        stats = statistics(data, observed)
        self.assertEqual(stats[0]["n_observed"], 2)
        self.assertAlmostEqual(stats[0]["mae_pu"], .2)
        self.assertAlmostEqual(stats[0]["rmse_pu"], .2)
        self.assertAlmostEqual(stats[0]["bias_pu"], 0)
        self.assertIsNone(stats[1]["mae_pu"])
        self.assertIsNone(statistics(data)[0]["rmse_pu"])

    def test_observations_align_by_utc_not_row_position(self):
        data = validate_forecast(*fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.csv"
            path.write_text("valid_time,turbine_id,power_normalized\n2026-02-01T06:00:00+05:00,turbine_2,0.9\n", encoding="utf-8")
            values = load_observations(path, data)
            self.assertEqual(values[1, 1], .9)
            self.assertEqual(np.isfinite(values).sum(), 1)
            with path.open("a", encoding="utf-8") as file:
                file.write("2026-02-01T01:00:00Z,turbine_2,0.9\n")
            with self.assertRaises(ValueError): load_observations(path, data)

    def test_demo_cannot_be_scored(self):
        data = validate_forecast(*fixture(demo=True))
        with self.assertRaises(ValueError): statistics(data, np.zeros_like(data.power))

    def test_error_figures_require_matched_observations(self):
        import matplotlib.pyplot as plt
        data = validate_forecast(*fixture())
        observed = np.full_like(data.power, np.nan)
        generated = list(figures(data, observed))
        self.assertEqual(len(generated), 4)
        for fig, *_ in generated: plt.close(fig)
        observed[3, 1] = .2
        generated = list(figures(data, observed))
        self.assertEqual([entry[1] for entry in generated][-2:], ["validation", "residuals"])
        for fig, *_ in generated: plt.close(fig)

    def test_server_reads_model_from_disk_not_client_values(self):
        data = validate_forecast(*fixture())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "outputs" / "backtest"
            folder.mkdir(parents=True)
            path = folder / "forecast_20260201T0000Z.csv"
            path.write_text(forecast_csv(data), encoding="utf-8")
            path.with_suffix(".json").write_text(json.dumps(data.audit), encoding="utf-8")
            source, loaded, observed = report_input({"source": "backtest", "date": "2026-02-01", "horizon": 24, "rows": [{"power1": .99}]}, root)
            self.assertEqual(source, "backtest")
            self.assertEqual(loaded.power[0, 0], .4)
            self.assertIsNone(observed)

    def test_server_rejects_path_injection_and_mismatched_mode(self):
        with self.assertRaises(ValueError):
            report_input({"source": "../", "date": "2026-02-01", "horizon": 24})
        with self.assertRaises(ValueError):
            report_input({"source": "demo", "date": "../../test", "horizon": 24})
        rows, audit = fixture()
        audit["trained_model"] = False
        with self.assertRaises(ValueError): validate_forecast(rows, audit)

    def test_report_exports_real_vector_figures_and_verifiable_bundle(self):
        data = validate_forecast(*fixture(demo=True))
        with tempfile.TemporaryDirectory() as directory:
            report = make_report(data, Path(directory))
            metadata = json.loads((report / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["figures"]), 4)
            self.assertEqual(metadata["n_observed"], 0)
            self.assertIsNone(metadata["statistics"][0]["mae_pu"])
            self.assertTrue((report / "report.pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("<svg", (report / "trajectory.svg").read_text(encoding="utf-8"))
            self.assertIn("СИНТЕТИЧЕСКИЙ ПРИМЕР", (report / "report.html").read_text(encoding="utf-8"))
            with zipfile.ZipFile(report / "report_bundle.zip") as archive:
                for line in archive.read("SHA256SUMS.txt").decode().splitlines():
                    expected, name = line.split("  ", 1)
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
