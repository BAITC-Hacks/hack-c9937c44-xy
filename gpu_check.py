"""Offline January GPU check: SCADA parity, same-weight inference, then training."""

import argparse
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data_agent import DATASET_FILENAMES, DataAgent
from evaluate import comparison_hours, evaluation_report
from main_simulation import HISTORY_FEATURES, TURBINES, aligned_weather, build_parser, hour_index, run
from model_agent import ForecastData, ModelAgent, predict


def check(output: Path, cpu_rehearsal: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    device, backend = ("cpu", "pandas") if cpu_rehearsal else ("cuda", "cudf")
    report = {"status": "running", "device": device, "gpu_tested": False,
              "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    try:
        if not cpu_rehearsal:
            if not torch.cuda.is_available():
                raise RuntimeError("NVIDIA CUDA unavailable; run on a Linux GPU instance. No CPU fallback.")
            import cudf
            import cuml
            import cupy as cp

            report.update(gpu=torch.cuda.get_device_name(0), cudf=cudf.__version__,
                          cuml=cuml.__version__, cupy=cp.__version__)
            assert int(cp.arange(4).sum().get()) == 6
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
        print(json.dumps(report, indent=2), flush=True)
        origin = pd.Timestamp("2026-01-30T00:00:00Z")
        sources = [Path("data/raw") / name for name in DATASET_FILENAMES]
        reference_agent = DataAgent(backend="pandas", data_timezone="UTC")
        target_agent = DataAgent(backend=backend, data_timezone="UTC")
        reference = [reference_agent.load_history(path, origin) for path in sources]
        actual = [target_agent.load_history(path, origin) for path in sources]
        for left, right in zip(reference, actual):
            pd.testing.assert_index_equal(left.index, right.index)
            assert list(left.columns) == list(right.columns)
            np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-6, atol=1e-7, equal_nan=True)
        report["scada_parity"] = "passed"
        print(f"PASS: pandas/{backend} hourly features and missing-value masks match", flush=True)
        times = hour_index(origin, 72, -72)
        contexts = [np.concatenate([h.reindex(times)[list(HISTORY_FEATURES)].to_numpy(dtype=np.float32)
                                   for h in histories], axis=-1) for histories in (reference, actual)]
        weather, _, audits = aligned_weather([
            reference_agent.fetch_forecast(t.latitude, t.longitude, origin, 24) for t in TURBINES
        ], origin, 24)
        bound = max(pd.Timestamp(item["available_at_upper_bound"]) for item in audits)
        checkpoint = Path("outputs/backtest/checkpoints/model_20260130T0000Z.pt")
        predictions = []
        for target, context in zip(("cpu", device), contexts):
            model = ModelAgent.load(checkpoint, device=target)
            if target == "cuda":
                assert next(model.model.parameters()).is_cuda and model.scaler_backend == "cuml"
            predictions.append(predict(model, ForecastData(
                history=context, weather=weather, current_date=origin.to_pydatetime(),
                available_at_upper_bound=[bound.to_pydatetime()] * 24,
            )))
            del model
        np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-4, atol=1e-4)
        report["inference_max_abs_difference"] = float(np.max(np.abs(predictions[0] - predictions[1])))
        report["inference_tolerance"] = {"rtol": 1e-4, "atol": 1e-4}
        print(f"PASS: same-checkpoint CPU/{device} inference; max difference={report['inference_max_abs_difference']:.8g}", flush=True)
        args = build_parser().parse_args([
            "--data-dir", "data/raw", "--data-timezone", "UTC", "--history-policy", "expanding",
            "--backend", backend, "--device", device, "--start", "2026-01-30", "--end", "2026-01-30",
            "--horizon", "24", "--lookback", "72", "--train-days", "14", "--min-train-samples", "14",
            "--epochs", "10", "--seed", "42", "--output", str(output),
        ])
        started = time.perf_counter()
        run(args)
        if not cpu_rehearsal:
            torch.cuda.synchronize()
            report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
            assert report["peak_cuda_memory_bytes"] > 0
        report["training_pipeline_seconds"] = time.perf_counter() - started
        audit = json.loads((output / "forecast_20260130T0000Z.json").read_text())
        assert audit["training"]["metrics"]["device"] == device
        assert audit["training"]["metrics"]["scaler_backend"] == ("numpy" if cpu_rehearsal else "cuml")
        manifest, hours = comparison_hours(output, sources, "UTC")
        assert manifest["status"] == "completed" and manifest["submission_rows"] == 48
        evaluation = evaluation_report(manifest, hours)
        (output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False))
        pd.DataFrame(evaluation["by_lead"]).to_csv(output / "evaluation.csv", index=False)
        pd.DataFrame(evaluation["rows"]).to_csv(output / "evaluation-hours.csv", index=False)
        report["evaluation"] = evaluation["summary"]
        report["gpu_tested"] = not cpu_rehearsal
        report["status"] = "cpu-rehearsal-passed" if cpu_rehearsal else "gpu-passed"
        report["notice"] = "Same weights are compared numerically; independent GPU training uses mixed precision and need not equal CPU training. UTC/coordinates/heights remain assumptions."
        print(f"PASS: {device} training and evaluation completed, 48 rows", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu-check"), help="A new output directory; existing results are never overwritten")
    parser.add_argument("--cpu-rehearsal", action="store_true", help="Validate this runner locally; does NOT constitute a GPU check")
    options = parser.parse_args()
    with patch("requests.sessions.Session.request", side_effect=RuntimeError("GPU check must use bundled weather cache")):
        check(options.output, options.cpu_rehearsal)
