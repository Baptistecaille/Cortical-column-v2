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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from column import CorticalColumn
from eval.benchmark import BenchmarkRunner
from eval.unsupervised_eval import prediction_success_rate


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


def test_run_multiview_rotation_benchmark_returns_float():
    model = _make_small_model()
    runner = BenchmarkRunner(model, device="cpu")
    score = runner.run_multiview_rotation_benchmark(n_samples=5, n_views=3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_run_multiview_rotation_benchmark_uses_sequential_delta_velocity(monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            self.transform = transform

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), 0

    class SpyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.reset_count = 0
            self.velocities = []

        def eval(self):
            return self

        def reset(self):
            self.reset_count += 1

        def step(self, s_t, v_t, train=False):
            self.velocities.append(v_t.detach().cpu().clone())
            idx = len(self.velocities) - 1
            sdr = torch.eye(4)[idx]
            grid = torch.tensor([1.0, 0.0])
            return {"sdr": sdr, "all_grid_codes": [grid]}

    import torchvision

    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)

    model = SpyModel()
    runner = BenchmarkRunner(model, device="cpu")
    score = runner.run_multiview_rotation_benchmark(
        n_samples=1,
        n_views=4,
        velocity_scale=0.1,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
        ]
    )
    assert model.reset_count == 1
    assert torch.allclose(torch.stack(model.velocities), expected, atol=1e-6)
    assert score == pytest.approx(0.5)


def test_prediction_success_rate_lift_uses_predicted_sparsity_as_chance_level():
    predicted = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    observed = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

    stats = prediction_success_rate(predicted, observed)

    assert stats["overlap"] == pytest.approx(1.0)
    assert stats["chance_level"] == pytest.approx(0.5)
    assert stats["overlap_lift"] == pytest.approx(2.0)


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


def test_cortical_column_step_returns_all_grid_codes_not_grid_code():
    """Regression: step() exposes 'all_grid_codes', not 'grid_code'."""
    model = _make_small_model()
    s_t = torch.rand(784)
    v_t = torch.zeros(2)
    result = model.step(s_t, v_t, train=False)
    assert "all_grid_codes" in result, "Missing 'all_grid_codes' key in step() output"
    assert "grid_code" not in result, "'grid_code' key must not exist in step() output"
    assert isinstance(result["all_grid_codes"][0], torch.Tensor)
