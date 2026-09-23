"""Local dashboard server with an on-demand Matplotlib report endpoint.

Run from any directory: python serve_preview.py --port 8767
The service is bound to localhost and serves only the UI and output artifacts.
"""
from __future__ import annotations

import argparse
from datetime import date
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from scientific_report import load_forecast, load_observations, make_report, timestamp, validate_forecast
from economics import calculate_economics

ROOT = Path(__file__).resolve().parent


def report_input(payload: dict, root: Path = ROOT):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    source, day, horizon = payload.get("source"), payload.get("date"), payload.get("horizon")
    if not isinstance(source, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError("Unknown source")
    if source == "synthetic" or payload.get("origin") is None:
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Invalid issue date")
        date.fromisoformat(day)
        origin = timestamp(f"{day}T00:00:00+00:00")
    else:
        origin = timestamp(payload["origin"])
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError("Issue time must align to an hour")
    if type(horizon) is not int or horizon not in (24, 48):
        raise ValueError("Expected a 24h or 48h horizon")
    if source == "synthetic":
        origin = origin.isoformat()
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != horizon:
            raise ValueError("Incomplete browser demo")
        records = [{"forecast_origin": origin, "valid_time": row["time"], "turbine_id": f"turbine_{j}",
                    "power_normalized": row[f"power{j}"], "wind_speed_ms": row[f"wind{j}"]}
                   for row in rows for j in (1, 2)]
        audit = {"mode": "synthetic-demo", "trained_model": False, "forecast_origin": origin,
                 "horizon_hours": horizon, "weather_source": "browser synthetic scenario",
                 "physical_validation": {"wind_limits_applied": True}}
        forecast = validate_forecast(records, audit)
    else:
        path = root / "outputs" / source / f"forecast_{origin:%Y%m%dT%H%MZ}.csv"
        if not path.resolve().is_relative_to((root / "outputs").resolve()):
            raise ValueError("Forecast must be inside outputs")
        forecast = load_forecast(path, horizon)
        expected = "synthetic-demo" if source == "demo" else "historical-backtest"
        if forecast.audit["mode"] != expected or forecast.origin != origin:
            raise ValueError("Forecast audit does not match requested source/date")
    observed_path = root / "outputs" / "observations.csv"
    observed = load_observations(observed_path, forecast) if not forecast.demo and observed_path.exists() else None
    return source, forecast, observed


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, status: int, value: dict):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/preview/")
            self.end_headers()
            return None
        if path == "/preview/":
            path += "index.html"
        relative = Path(path.lstrip("/"))
        resolved = (ROOT / relative).resolve()
        allowed = (
            (path.startswith("/preview/") and resolved.suffix in (".html", ".js", ".css"))
            or (re.fullmatch(r"/outputs/[A-Za-z0-9_-]+/(forecast_\d{8}T\d{4}Z\.(csv|json)|run\.json|evaluation\.(csv|json)|evaluation-hours\.csv)", path) is not None)
            or (path.startswith("/outputs/reports/") and resolved.suffix in (".pdf", ".html", ".svg", ".png", ".json", ".zip", ".csv"))
        )
        if not allowed or ".." in relative.parts or not resolved.is_relative_to(ROOT) or not resolved.is_file():
            self.send_error(404)
            return None
        self.path = path
        return super().send_head()

    def do_POST(self):
        if self.path not in ("/api/report", "/api/economics"):
            self.json_response(404, {"error": "Unknown endpoint"})
            return
        port = self.server.server_port
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in [f"http://{host}" for host in hosts]):
            self.json_response(403, {"error": "Local same-origin requests only"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128_000 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json, maximum 128 KB")
            payload = json.loads(self.rfile.read(length))
            source, forecast, observed = report_input(payload)
            if self.path == "/api/economics":
                self.json_response(200, calculate_economics(forecast, payload.get("economics")))
                return
            folder = make_report(forecast, ROOT / "outputs" / "reports" / source, observed, payload.get("economics"))
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            self.json_response(200, {"base_url": "/" + folder.relative_to(ROOT).as_posix() + "/", "report": metadata})
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.json_response(422, {"error": f"Проверьте исходные данные: {exc}"})
        except FileNotFoundError:
            self.json_response(404, {"error": "CSV или JSON-аудит прогноза не найден. Сначала выполните расчёт."})
        except Exception as exc:
            self.log_error("Report generation failed: %s", type(exc).__name__)
            self.json_response(500, {"error": "Не удалось создать отчёт. Проверьте сервер и зависимости отчётов."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ALEM WIND: http://127.0.0.1:{args.port}/preview/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
