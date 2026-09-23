"""Verify committed evidence offline; optionally re-score against original SCADA CSVs."""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    check(result.tzinfo is not None, "Naive timestamp in evidence")
    return result


def key(row):
    return row["forecast_origin"], row["valid_time"], row["turbine_id"]


def verify_run(folder):
    run = read_json(folder / "run.json")
    check(run["status"] == "completed" and run["submission_complete"], f"Incomplete run: {folder}")
    all_rows = []
    for name, origin_text in zip(run["daily_files"], run["completed_origins"], strict=True):
        check(Path(name).name == name, "Invalid daily filename")
        path = folder / name
        audit = read_json(path.with_suffix(".json"))
        origin = stamp(origin_text)
        check(stamp(audit["forecast_origin"]) == origin, "Audit origin mismatch")
        check(audit["trained_model"] and audit["mode"] == "historical-backtest", "Synthetic result")
        check(stamp(audit["training"]["latest_target_time"]) < origin, "Future training targets")
        check(all(stamp(t) < origin for t in audit["latest_observation"]), "Future observation")
        for weather in audit["inference_weather"]:
            check(stamp(weather["available_at_upper_bound"]) <= origin, "Future inference weather")
        for sample in audit["training"]["forecast_provenance"]:
            for weather in sample["turbines"]:
                check(stamp(weather["available_at_upper_bound"]) <= stamp(sample["origin"]), "Future training weather")
        checkpoint = (folder / audit["checkpoint"]).resolve()
        check(checkpoint.is_relative_to(folder.resolve()) and checkpoint.is_file(), "Missing checkpoint")
        rows = read_csv(path)
        expected = {(origin, origin + timedelta(hours=h), turbine["turbine_id"])
                    for h in range(run["horizon_hours"]) for turbine in run["turbines"]}
        actual = {(stamp(r["forecast_origin"]), stamp(r["valid_time"]), r["turbine_id"]) for r in rows}
        check(len(rows) == len(expected) and actual == expected, "Missing or duplicate forecast hours")
        check(all(0 <= float(r["power_normalized"]) <= 1 for r in rows), "Invalid power")
        check(audit["training"]["metrics"]["device"] == run["model_config"]["device"], "Wrong training device")
        all_rows.extend(rows)
    check(len(all_rows) == run["submission_rows"], "Submission count mismatch")
    check(all_rows == read_csv(folder / "submission.csv"), "Submission differs from daily CSVs")
    report_path = folder / "evaluation.json"
    if report_path.exists():
        report = read_json(report_path)
        check(report["run_created_at"] == run["created_at"], "Stale evaluation")
        indexed = {key(r): r for r in report["rows"]}
        check(len(indexed) == len(all_rows) == len(report["rows"]), "Evaluation coverage mismatch")
        for row in all_rows:
            check(key(row) in indexed, "Missing evaluated row")
            for column in ("power_normalized", "wind_speed_ms"):
                check(abs(float(row[column]) - indexed[key(row)][column]) < 1e-7, "Evaluation differs from forecast")
        boundary = stamp(report["holdout_start"]) if report["holdout_start"] else None
        for row in report["rows"]:
            origin = stamp(row["forecast_origin"])
            split = "all" if boundary is None else "holdout" if origin >= boundary else (
                "development" if origin + timedelta(hours=run["horizon_hours"]) <= boundary else "purged")
            check(row["split"] == split, "Incorrect chronological split")
            check(row["eligible"] == (row["observed"] is not None and row["persistence"] is not None), "Incorrect missing-value mask")
        for summary in report["summary"]:
            group = [r for r in report["rows"] if (r["split"], r["turbine_id"]) == (summary["split"], summary["turbine_id"])]
            paired = [r for r in group if r["eligible"]]
            check(len(group) == summary["expected"] and len(paired) == summary["n"], "Incorrect sample size")
            for method, column in (("model", "power_normalized"), ("persistence", "persistence"), ("power_curve", "power_curve")):
                errors = [r[column] - r["observed"] for r in paired]
                values = {"mae": sum(map(abs, errors)) / len(errors), "rmse": math.sqrt(sum(e * e for e in errors) / len(errors))}
                for metric, value in values.items():
                    check(math.isclose(value, summary[f"{method}_{metric}"], abs_tol=1e-12), f"Wrong {method} {metric}")
    print(f"PASS: {folder.name}: {len(run['completed_origins'])} origins, {len(all_rows)} rows; artifacts, timing bounds and metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Original SCADA directory; requires project Python dependencies")
    args = parser.parse_args()
    evidence = read_json(ROOT / "outputs/evidence/manifest.json")
    for name, expected in {**evidence["files"], **evidence["runtime_sources"]}.items():
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT), "Evidence path outside repository")
        check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"SHA256 mismatch: {name}")
    print(f"PASS: {len(evidence['files'])} evidence files and {len(evidence['runtime_sources'])} executed source files match SHA256")
    for name in evidence["runs"]:
        verify_run(ROOT / "outputs" / name)
    gpu = read_json(ROOT / "outputs/gpu-check/verification.json")
    check(gpu["status"] == "gpu-passed" and gpu["gpu_tested"], "GPU check did not pass")
    check(gpu["scada_parity"] == "passed" and gpu["inference_max_abs_difference"] <= 1e-4, "GPU parity failed")
    if args.data_dir:
        import pandas as pd
        from evaluate import comparison_hours, evaluation_report
        sources = []
        for name, expected in evidence["input_files"].items():
            path = args.data_dir / name
            check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Different input CSV: {name}")
            sources.append(path)
        for name in evidence["runs"]:
            folder = ROOT / "outputs" / name
            if not (folder / "evaluation.json").exists():
                continue
            saved = read_json(folder / "evaluation.json")
            run, hours = comparison_hours(folder, sources, "UTC")
            actual = evaluation_report(run, hours, saved["holdout_start"])
            pd.testing.assert_frame_equal(pd.DataFrame(actual["rows"]), pd.DataFrame(saved["rows"]), check_exact=False, atol=1e-12, rtol=1e-12)
            pd.testing.assert_frame_equal(pd.DataFrame(actual["summary"]), pd.DataFrame(saved["summary"]), check_exact=False, atol=1e-12, rtol=1e-12)
        print("PASS: original SCADA hashes, observed values, persistence and metrics independently re-scored")
    else:
        print("Original SCADA not checked; use --data-dir to verify observations against the source CSVs.")
    print("PASS: saved GPU parity evidence verified; this command does not run CUDA or attest historical weather publication times.")


if __name__ == "__main__":
    main()
