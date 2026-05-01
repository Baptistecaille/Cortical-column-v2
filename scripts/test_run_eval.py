"""
Integration tests for scripts/run_eval.py orchestrator (Task 4 will make them pass).
run: python -m pytest scripts/test_run_eval.py -v --timeout=120
"""
import sys
import os
import json
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMALL_MODEL_ARGS = [
    "--n_columns", "2",
    "--n_sdr", "256",
    "--w", "10",
    "--n_minicolumns", "64",
    "--k_active", "10",
    "--n_grid_modules", "3",
    "--grid_periods", "3,5,7",
]


def _run_eval(extra_args, report_path):
    return subprocess.run(
        [sys.executable, "scripts/run_eval.py"]
        + SMALL_MODEL_ARGS
        + ["--n_samples", "10", "--n_bench_train", "20", "--n_bench_test", "10",
           "--n_rot_samples", "5", "--output", report_path]
        + extra_args,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


def test_run_eval_produces_json_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.json")
        result = _run_eval([], report_path)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(report_path)
        with open(report_path) as f:
            data = json.load(f)
        assert "unsupervised" in data
        assert "benchmark" in data


def test_run_eval_metrics_in_valid_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.json")
        _run_eval([], report_path)
        with open(report_path) as f:
            data = json.load(f)
        for k, v in data["unsupervised"].items():
            if not isinstance(v, float) or v != v:   # skip nan
                continue
            assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of [0,1]"
        for k, v in data["benchmark"].items():
            if isinstance(v, float) and v == v:
                assert 0.0 <= v <= 1.0, f"Benchmark metric {k}={v} out of [0,1]"


def test_run_eval_with_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "model.pt")
        # Quick train to get a checkpoint
        subprocess.run(
            [sys.executable, "scripts/train.py",
             "--n_images", "50", "--n_epochs", "1",
             "--n_columns", "2", "--n_sdr", "256", "--w", "10",
             "--n_minicolumns", "64", "--k_active", "10",
             "--n_grid_modules", "3",
             "--save_path", checkpoint],
            capture_output=True, text=True, cwd=ROOT, timeout=60,
        )
        assert os.path.exists(checkpoint), "Checkpoint not created by train.py"

        report_path = os.path.join(tmpdir, "report.json")
        result = _run_eval(["--checkpoint", checkpoint], report_path)
        assert result.returncode == 0, result.stderr
        with open(report_path) as f:
            data = json.load(f)
        assert data["checkpoint"] == checkpoint
