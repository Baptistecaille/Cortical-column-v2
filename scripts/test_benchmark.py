# scripts/test_benchmark.py
"""
Tests for new BenchmarkRunner methods (Task 2 will make them pass).
run: python -m pytest scripts/test_benchmark.py -v
"""
import sys
import os
import json

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from column import CorticalColumn
from eval.benchmark import BenchmarkRunner


def _make_small_model():
    return CorticalColumn(
        n_columns=2,
        input_dim=784,
        n_sdr=256,
        w=10,
        n_minicolumns=64,
        k_active=10,
        n_grid_modules=3,
        grid_periods=[3, 5, 7],
        consensus_threshold=1.0,
    )


def test_run_mnist_linear_probe_returns_float():
    model = _make_small_model()
    runner = BenchmarkRunner(model, device="cpu")
    acc = runner.run_mnist_linear_probe(n_train=50, n_test=20, n_epochs=5)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_run_mnist_rotation_benchmark_returns_float():
    model = _make_small_model()
    runner = BenchmarkRunner(model, device="cpu")
    score = runner.run_mnist_rotation_benchmark(n_samples=5, n_views=3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_save_report_writes_json(tmp_path):
    model = _make_small_model()
    runner = BenchmarkRunner(model, device="cpu")
    results = {"lin_probe": 0.42, "rotation_inv": 0.31}
    out = str(tmp_path / "bench.json")
    runner.save_report(results, out)
    with open(out) as f:
        data = json.load(f)
    assert data["lin_probe"] == pytest.approx(0.42)
    assert "rotation_inv" in data


def test_rotation_invariance_uses_correct_result_key():
    """
    Regression: run_rotation_invariance must NOT crash with KeyError("grid_code").
    The correct key is result["all_grid_codes"][0].
    """
    model = _make_small_model()
    runner = BenchmarkRunner(model, device="cpu")
    # Build a minimal views dataloader: 1 sample, 2 views
    img = torch.rand(784)
    views = [img, img + 0.01]
    loader = [([v.unsqueeze(0) for v in views], torch.tensor(0))]
    # Should not raise KeyError
    score = runner.run_rotation_invariance(loader, n_views=2)
    assert isinstance(score, float)
