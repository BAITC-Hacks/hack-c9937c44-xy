from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from web_app import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "preview").mkdir()
        (self.root / "preview/index.html").write_text("ALEM WIND")
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        origin = datetime(2026, 1, 29, tzinfo=timezone.utc)
        self.payload = {"source": "synthetic", "date": "2026-01-29", "horizon": 24,
                        "rows": [{"time": (origin + timedelta(hours=i)).isoformat(), "power1": .5,
                                  "power2": .5, "wind1": 8, "wind2": 8} for i in range(24)],
                        "economics": {"capacity_1_mw": 2, "capacity_2_mw": 2, "plan_mw": 3,
                                      "tariff_kzt_kwh": 25, "tariff_note": "demo"}}

    def test_health_dashboard_and_private_file_boundaries(self):
        self.assertEqual(self.client.get("/healthz").json, {"status": "ok"})
        self.assertEqual(self.client.get("/").location, "/preview/?run=rolling-january-gpu")
        with self.client.get("/preview/") as response:
            self.assertEqual(response.data, b"ALEM WIND")
        (self.root / ".env").write_text("not-public")
        (self.root / "preview/leak.html").symlink_to(self.root.parent / "outside.html")
        for path in ("/.env", "/web_app.py", "/preview/../.env", "/outputs/backtest/checkpoints/model.pt", "/preview/leak.html"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_economics_works_over_https_host_without_localhost_restriction(self):
        response = self.client.post("/api/economics", json=self.payload, base_url="https://alem.example",
                                    headers={"Origin": "https://alem.example"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["totals"]["lost_energy_revenue_kzt"], 600000)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertFalse(response.json["order_placed"])

    def test_api_rejects_cross_origin_bad_json_and_large_requests(self):
        self.assertEqual(self.client.post("/api/economics", json=self.payload, headers={"Origin": "https://other.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/economics", data="{}").status_code, 415)
        self.assertEqual(self.client.post("/api/economics", data="{", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/api/economics", json={"large": "a" * 128000}).status_code, 413)
        self.assertEqual(self.client.post("/api/economics", json=[]).status_code, 422)

    def test_report_lock_prevents_parallel_matplotlib_and_recovers_after_error(self):
        def fail_report(*args):
            busy = self.client.post("/api/report", json=self.payload)
            self.assertEqual(busy.status_code, 429)
            self.assertEqual(busy.headers["Retry-After"], "5")
            raise RuntimeError("failed render")
        with patch("web_app.make_report", side_effect=fail_report), self.assertLogs(self.app.logger, level="ERROR"):
            self.assertEqual(self.client.post("/api/report", json=self.payload).status_code, 500)
        folder = self.root / "outputs/reports/synthetic/example"
        folder.mkdir(parents=True)
        (folder / "metadata.json").write_text(json.dumps({"pdf": "report.pdf"}))
        with patch("web_app.make_report", return_value=folder):
            response = self.client.post("/api/report", json=self.payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["base_url"], "/outputs/reports/synthetic/example/")


if __name__ == "__main__":
    unittest.main()
