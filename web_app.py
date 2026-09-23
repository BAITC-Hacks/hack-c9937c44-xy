"""Public MVP entry point: gunicorn --config gunicorn.conf.py web_app:app."""
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, redirect, request, send_file
from werkzeug.exceptions import HTTPException

from economics import calculate_economics
from scientific_report import make_report
from serve_preview import ROOT, public_file, report_input


def create_app(root=ROOT):
    root = Path(root).resolve()
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 128_000
    # ponytail: one report worker; use a job queue before adding Gunicorn workers.
    report_lock = Lock()

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def home():
        return redirect("/preview/?run=rolling-january-gpu")

    @app.get("/preview/")
    def dashboard():
        return send_file(root / "preview/index.html")

    @app.get("/<path:name>")
    def artifact(name):
        path = public_file("/" + name, root)
        if path is None:
            abort(404)
        return send_file(path)

    @app.post("/api/<operation>")
    def api(operation):
        if operation not in ("report", "economics"):
            abort(404)
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or parsed.netloc != request.host:
                abort(403, "Запрос должен быть с этого сайта.")
        if not request.is_json:
            abort(415, "Ожидается JSON.")
        payload = request.get_json()
        try:
            source, forecast, observed = report_input(payload, root)
            if operation == "economics":
                return jsonify(calculate_economics(forecast, payload.get("economics")))
            if not report_lock.acquire(blocking=False):
                return jsonify(error="Формируется другой отчёт. Повторите через несколько секунд."), 429, {"Retry-After": "5"}
            try:
                folder = make_report(forecast, root / "outputs/reports" / source, observed, payload.get("economics"))
            finally:
                report_lock.release()
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            return jsonify(base_url="/" + folder.relative_to(root).as_posix() + "/", report=metadata)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return jsonify(error=f"Проверьте исходные данные: {exc}"), 422
        except FileNotFoundError:
            return jsonify(error="Прогноз для выбранного выпуска не найден."), 404
        except Exception:
            app.logger.exception("Report generation failed")
            return jsonify(error="Не удалось сформировать отчёт. Повторите позже."), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    return app


app = create_app()
