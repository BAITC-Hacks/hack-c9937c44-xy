import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

base = sys.argv[1].rstrip("/")
repository = Path(sys.argv[2])
evidence = Path(sys.argv[3])


def fetch(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(base + path, data=data, headers={"Content-Type": "application/json", "Origin": base})
    with urlopen(request, timeout=120) as response:
        assert response.status == 200
        return response.read()


assert json.loads(fetch("/healthz")) == {"status": "ok"}
assert b"ALEM WIND" in fetch("/")
for name in ("app.js", "evaluation.js", "report-ui.js", "economics-ui.js", "styles.css", "economics.css",
             "map-ui.js", "map-ui.css", "assets/maplibre-gl.js", "assets/maplibre-gl.css"):
    assert fetch("/preview/" + name)
sources = ["backtest", "replay", "rolling-january", "gpu-check", "rolling-january-gpu", "demo"]
for source in sources:
    assert json.loads(fetch(f"/outputs/{source}/run.json"))
for path in ("/.env", "/.git/config", "/web_app.py", "/data/raw/", "/preview/../web_app.py"):
    try:
        fetch(path)
    except HTTPError as error:
        assert error.code == 404, (path, error.code)
    else:
        raise AssertionError(path)
print("PASS: health, dashboard assets, six run manifests and private-path boundaries", flush=True)

example = json.loads((repository / "outputs/evidence/economics-example.json").read_text())
payload = {"source": "rolling-january-gpu", "origin": "2026-01-29T00:00:00+00:00", "horizon": 24, "economics": example["inputs"]}
economics = json.loads(fetch("/api/economics", payload))
assert economics["totals"] == example["totals"]
assert economics["procurement"] == example["procurement"]
print("PASS: GPU forecast economics matches committed reference exactly", flush=True)
start = time.monotonic()
report = json.loads(fetch("/api/report", payload))
elapsed = time.monotonic() - start
assert report["report"]["n_observed"] == 48
assert len(report["report"]["figures"]) == 6
assert report["report"]["economics"]["totals"] == example["totals"]
pdf = fetch(report["base_url"] + report["report"]["pdf"])
assert pdf.startswith(b"%PDF")
bundle = fetch(report["base_url"] + report["report"]["bundle"])
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    assert archive.testzip() is None
    for line in archive.read("SHA256SUMS.txt").decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
print(f"PASS: PDF and ZIP generated in {elapsed:.2f}s; 6 figures, 48 observations, all ZIP hashes match", flush=True)
result = {"checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_url": base,
          "health": "ok", "private_paths": "404", "run_manifests": sources,
          "economics_matches_reference": True, "report_seconds": round(elapsed, 2),
          "report_base_url": report["base_url"], "n_observed": 48, "figures": 6,
          "pdf_bytes": len(pdf), "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
          "zip_bytes": len(bundle), "zip_sha256": hashlib.sha256(bundle).hexdigest(), "zip_hashes_verified": True,
          "source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                            for name in ("Dockerfile", ".dockerignore", "web_app.py", "serve_preview.py", "gunicorn.conf.py", "requirements-web.txt", "compose.yaml", "render.yaml")}}
evidence.write_text(json.dumps(result, indent=2) + "\n")
