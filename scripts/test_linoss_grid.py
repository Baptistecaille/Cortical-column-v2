"""
Tests pour l'intégration LinOSS dans GridCellNetwork (Rusch 2024).
Ces tests doivent ÉCHOUER avant l'implémentation (Phase 3.2).

GridCellNetwork(use_linoss=True) : l'intégrateur Euler est remplacé par
une couche LinOSSLayer. Les invariants I4.1/I4.2/I4.3 doivent rester verts.

Réf : Rusch & Rus 2024 arXiv 2410.03943 (ICLR 2025 Oral).
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid_cell import GridCellNetwork


def test_linoss_phases_stay_in_range():
    """I4.1 : φ_k ∈ [0, 2π)² avec LinOSS sur 100 pas."""
    gc = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11], use_linoss=True)
    for _ in range(100):
        v = torch.randn(2) * 0.5
        phases = gc.integrate(v)
        assert ((phases >= 0) & (phases < 2 * math.pi)).all(), (
            f"I4.1 violé : min={phases.min():.4f}, max={phases.max():.4f}"
        )


def test_linoss_get_code_dimension():
    """I4.2 : get_code() retourne 4·n_modules avec LinOSS."""
    gc = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11], use_linoss=True)
    gc.integrate(torch.randn(2))
    code = gc.get_code()
    assert code.shape == (16,), f"I4.2 violé : shape={code.shape}, attendu (16,)"


def test_linoss_get_code_cossin_range():
    """I4.2 : valeurs de get_code() dans [-1, 1] (cos/sin)."""
    gc = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11], use_linoss=True)
    gc.integrate(torch.randn(2))
    code = gc.get_code()
    assert ((code >= -1.0) & (code <= 1.0)).all(), (
        f"I4.2 violé : code hors [-1,1] : min={code.min():.4f}, max={code.max():.4f}"
    )


def test_linoss_anchor_modifies_phases():
    """I4.3 : anchor() modifie les phases avec LinOSS (correction pondérée 1/λ_k)."""
    gc = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11], use_linoss=True)
    gc.integrate(torch.randn(2) * 0.5)
    target = torch.zeros(4, 2)
    phases_before = gc.phases.clone()
    gc.anchor(target, confidence=1.0)
    assert not torch.allclose(phases_before, gc.phases), \
        "I4.3 violé : anchor() n'a pas modifié les phases"


def test_linoss_is_learnable_module():
    """LinOSS ajoute des paramètres apprenables à GridCellNetwork."""
    gc_euler  = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11])
    gc_linoss = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11], use_linoss=True)
    n_params_euler  = sum(p.numel() for p in gc_euler.parameters())
    n_params_linoss = sum(p.numel() for p in gc_linoss.parameters())
    assert n_params_linoss > n_params_euler, (
        f"LinOSS doit ajouter des paramètres : euler={n_params_euler}, linoss={n_params_linoss}"
    )


def test_euler_fallback_still_works():
    """use_linoss=False (défaut) : comportement Euler inchangé."""
    gc = GridCellNetwork(n_modules=4, periods=[3, 5, 7, 11])
    for _ in range(10):
        phases = gc.integrate(torch.randn(2) * 0.1)
        assert ((phases >= 0) & (phases < 2 * math.pi)).all()


if __name__ == "__main__":
    tests = [
        test_linoss_phases_stay_in_range,
        test_linoss_get_code_dimension,
        test_linoss_get_code_cossin_range,
        test_linoss_anchor_modifies_phases,
        test_linoss_is_learnable_module,
        test_euler_fallback_still_works,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"  PASS : {t.__name__}")
            n_pass += 1
        except (AssertionError, AttributeError, TypeError) as e:
            print(f"  FAIL : {t.__name__} — {e}")
    print(f"\n{n_pass}/{len(tests)} tests passés.")
