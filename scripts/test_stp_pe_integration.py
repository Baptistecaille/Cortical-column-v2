"""
Tests pour l'intégration iSTP dans PECircuits (Waitzmann 2024).
Ces tests doivent ÉCHOUER avant l'implémentation (Phase 3.1).

Propriété clé : PV→E STD uniquement.
Quand predicted > x, PE- est actif. PV1 (driven by x) l'inhibe.
Après déplétion STD (activité feedforward soutenue), r_pv1_eff diminue
→ l'inhibition PV1→PE- s'affaiblit → PE- augmente.

Réf : Waitzmann 2024 PNAS e2311040121, Fig. 3F.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions.pe_circuits import PECircuits


def test_pv1_inhibition_depresses_with_stp():
    """
    PV1→PE- inhibition doit s'affaiblir après activité feedforward soutenue (STD).

    Setup :
    - Pendant la déplétion : x_strong=1.0 (fort signal L4 → PV1 déclenche)
    - Mesure PE- avec predicted=0.8, x_probe=0.3 (predicted > x → PE- actif)
    - PV1 est driven by x_probe → inhibe PE-
    - Après 30 pas de x_strong : ressources PV1 épuisées (STD)
    - PE- doit être plus élevé après déplétion (inhibition affaiblie)
    """
    torch.manual_seed(0)
    pe = PECircuits(dim=64, context_dim=24, use_stp=True)

    x_probe   = torch.full((64,), 0.3)   # feedforward modéré (active PV1 doucement)
    predicted = torch.full((64,), 0.8)   # top-down fort → predicted > x → PE- actif

    # Mesure initiale (ressources PV1 fraîches, inhibition maximale)
    _, pe_neg_initial = pe.compute_prediction_errors_with_interneurons(x_probe, predicted)

    # Déplétion STD : 30 pas avec forte activité feedforward → épuise les ressources PV1
    x_strong     = torch.ones(64)
    context_zero = torch.zeros(24)
    for _ in range(30):
        pe.step_with_update(x_strong, context_zero)

    # Après déplétion STD : PV1 inhibe moins → PE- doit être plus élevé
    _, pe_neg_depleted = pe.compute_prediction_errors_with_interneurons(x_probe, predicted)

    assert pe_neg_depleted.mean() > pe_neg_initial.mean(), (
        f"PE- doit augmenter après dépression STD de PV1 : "
        f"initial={pe_neg_initial.mean():.4f}, "
        f"après déplétion={pe_neg_depleted.mean():.4f}"
    )


def test_stp_resources_recover_after_silence():
    """
    Après silence (pas de spikes), les ressources STD de PV1 récupèrent.
    Propriété Tsodyks-Markram : dx/dt = (1-x)/τ_d quand spike=0 → x → 1.
    """
    torch.manual_seed(0)
    pe = PECircuits(dim=64, context_dim=24, use_stp=True)

    x_probe   = torch.full((64,), 0.3)
    predicted = torch.full((64,), 0.8)
    x_strong  = torch.ones(64)
    x_silent  = torch.zeros(64)
    context_zero = torch.zeros(24)

    # Épuiser les ressources
    for _ in range(30):
        pe.step_with_update(x_strong, context_zero)

    _, pe_neg_depleted = pe.compute_prediction_errors_with_interneurons(x_probe, predicted)

    # 100 pas de silence → récupération STD (x → 1)
    for _ in range(100):
        pe.step_with_update(x_silent, context_zero)

    _, pe_neg_recovered = pe.compute_prediction_errors_with_interneurons(x_probe, predicted)

    # Après récupération : inhibition PV1 restaurée → PE- doit baisser
    assert pe_neg_recovered.mean() < pe_neg_depleted.mean(), (
        f"PE- doit diminuer après récupération STD : "
        f"déplété={pe_neg_depleted.mean():.4f}, "
        f"récupéré={pe_neg_recovered.mean():.4f}"
    )


def test_use_stp_false_disables_stp():
    """use_stp=False (défaut) → pas d'attribut pv1_stp actif."""
    pe_no_stp = PECircuits(dim=64, context_dim=24, use_stp=False)
    pe_stp    = PECircuits(dim=64, context_dim=24, use_stp=True)

    has_stp = hasattr(pe_no_stp, "pv1_stp") and pe_no_stp.pv1_stp is not None
    assert not has_stp, "use_stp=False ne doit pas créer pv1_stp actif"

    assert hasattr(pe_stp, "pv1_stp") and pe_stp.pv1_stp is not None, \
        "use_stp=True doit créer pv1_stp"


if __name__ == "__main__":
    tests = [
        test_pv1_inhibition_depresses_with_stp,
        test_stp_resources_recover_after_silence,
        test_use_stp_false_disables_stp,
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
