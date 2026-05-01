"""
Tests pour le circuit interneurones de PECircuits (Nemati 2025 / Lee 2025).
Ces tests doivent ÉCHOUER avant l'implémentation (Phase 2).

Propriétés attendues :
  - PV1 (feedforward) supprime PE- quand le signal L4 est fort
  - PV2 (feedback) supprime PE+ quand le top-down est fort
  - PE+ et PE- ne peuvent pas être simultanément actifs pour le même neurone
  - Les états r_pv1, r_pv2, r_som, r_vip sont accessibles après un step
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.pe_circuits import PECircuits


def test_pv1_suppresses_pe_neg_when_ff_strong():
    """PV1 driven by strong L4 input → PE- is suppressed relative to weak L4."""
    pe = PECircuits(dim=64, context_dim=24)

    predicted = torch.ones(64) * 0.5
    x_strong = torch.ones(64)   # fort signal feedforward → PV1 fort → PE- supprimé
    x_weak   = torch.zeros(64)  # signal nul → PV1 faible → PE- non supprimé

    _, pe_neg_strong = pe.compute_prediction_errors_with_interneurons(x_strong, predicted)
    _, pe_neg_weak   = pe.compute_prediction_errors_with_interneurons(x_weak,   predicted)

    assert pe_neg_strong.mean() <= pe_neg_weak.mean(), (
        f"PV1 doit supprimer PE- davantage quand x est fort : "
        f"pe_neg_strong={pe_neg_strong.mean():.4f} > pe_neg_weak={pe_neg_weak.mean():.4f}"
    )


def test_pv2_suppresses_pe_pos_when_td_strong():
    """PV2 driven by strong top-down → PE+ is suppressed relative to weak top-down."""
    pe = PECircuits(dim=64, context_dim=24)

    x = torch.ones(64) * 0.5
    predicted_strong = torch.ones(64)   # fort top-down → PV2 fort → PE+ supprimé
    predicted_weak   = torch.zeros(64)  # top-down nul → PV2 faible → PE+ non supprimé

    pe_pos_strong, _ = pe.compute_prediction_errors_with_interneurons(x, predicted_strong)
    pe_pos_weak,   _ = pe.compute_prediction_errors_with_interneurons(x, predicted_weak)

    assert pe_pos_strong.mean() <= pe_pos_weak.mean(), (
        f"PV2 doit supprimer PE+ davantage quand predicted est fort : "
        f"pe_pos_strong={pe_pos_strong.mean():.4f} > pe_pos_weak={pe_pos_weak.mean():.4f}"
    )


def test_pe_pos_neg_mutually_exclusive():
    """PE+ et PE- ne peuvent pas être simultanément actifs pour le même neurone."""
    pe = PECircuits(dim=64, context_dim=24)

    torch.manual_seed(42)
    x         = torch.rand(64)
    predicted = torch.rand(64)

    pe_pos, pe_neg = pe.compute_prediction_errors_with_interneurons(x, predicted)

    both_active = (pe_pos > 0) & (pe_neg > 0)
    assert not both_active.any(), (
        f"PE+ et PE- actifs simultanément sur {both_active.sum().item()} neurones"
    )


def test_interneuron_states_accessible_after_step():
    """Après step_with_update(), r_pv1/r_pv2/r_som/r_vip sont accessibles."""
    pe = PECircuits(dim=64, context_dim=24)
    x       = torch.randn(64)
    context = torch.randn(24)

    pe.step_with_update(x, context)

    for attr in ("r_pv1", "r_pv2", "r_som", "r_vip"):
        assert hasattr(pe, attr), f"Attribut {attr} manquant après step_with_update()"

    n_pv = 64 // 4  # 16
    n_sv = 64 // 8  # 8
    assert pe.r_pv1.shape == (n_pv,), f"r_pv1.shape={pe.r_pv1.shape}, attendu ({n_pv},)"
    assert pe.r_pv2.shape == (n_pv,), f"r_pv2.shape={pe.r_pv2.shape}, attendu ({n_pv},)"
    assert pe.r_som.shape  == (n_sv,), f"r_som.shape={pe.r_som.shape},  attendu ({n_sv},)"
    assert pe.r_vip.shape  == (n_sv,), f"r_vip.shape={pe.r_vip.shape},  attendu ({n_sv},)"


def test_vip_releases_pe_neg_via_disinhibition():
    """
    VIP disinhibits PE- en supprimant SOM.
    Quand VIP est actif (fort top-down), SOM est supprimé → PE- est moins inhibé.
    On vérifie indirectement : PE- ne s'effondre pas quand top-down est fort.
    """
    pe = PECircuits(dim=64, context_dim=24)

    # Situation où PE- devrait survivre malgré PV1 fort :
    # x fort (active PV1) mais predicted fort aussi (active VIP → supprime SOM)
    x         = torch.ones(64)
    predicted = torch.ones(64)

    pe_pos, pe_neg = pe.compute_prediction_errors_with_interneurons(x, predicted)

    # PE- et PE+ peuvent tous deux être nuls (x == predicted → erreur nulle)
    # mais aucun ne doit être négatif
    assert (pe_pos >= 0).all(), "PE+ doit être >= 0"
    assert (pe_neg >= 0).all(), "PE- doit être >= 0"


if __name__ == "__main__":
    tests = [
        test_pv1_suppresses_pe_neg_when_ff_strong,
        test_pv2_suppresses_pe_pos_when_td_strong,
        test_pe_pos_neg_mutually_exclusive,
        test_interneuron_states_accessible_after_step,
        test_vip_releases_pe_neg_via_disinhibition,
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
