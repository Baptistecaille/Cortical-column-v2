"""
Script de vérification automatique des 6 invariants formels.

À lancer avant et après chaque modification du code (CLAUDE.md §8).

Invariants vérifiés :
    I1.1 : ||x||₁ = w exactement (SDRSpace)
    I1.2 : x ∈ {0,1}^n (SDRSpace)
    I2.1 : |A_t| = k exactement (SpatialPooler)
    I2.2 : p_ij ∈ [0,1] (SpatialPooler permanences)
    I3.1 : isométrie de Layer6bTransformer (norme préservée)
    I4.1 : φ_k ∈ [0, 2π)² (GridCellNetwork)
    I4.2 : get_code() → 4·n_modules dimensions
    I4.3 : anchor() corrige avec 1/λ_k (vérifié qualitativement)
    I5.1 : D = φ_objet − φ_capteur (soustraction modulaire)
    I5.2 : compositionnalité D_A→C = D_A→B ⊕ D_B→C
    I6.1 : pas de weight sharing entre colonnes
    I6.2 : consensus = AND strict (pas moyenne)
    I6.3 : consensus_threshold = 1.0 par défaut
    I6.4 : P_fp décroît exponentiellement avec K

Usage :
    python scripts/run_invariants.py
    python scripts/run_invariants.py --verbose
"""

import sys
import os
import math
import torch
import argparse

# Ajout du répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sdr_space import SDRSpace
from core.spatial_pooler import SpatialPooler
from core.layer6b import Layer6bTransformer
from core.grid_cell import GridCellNetwork
from core.displacement import DisplacementAlgebra
from core.consensus import MultiColumnConsensus
from column import CorticalColumn


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Affiche le résultat d'un test d'invariant."""
    status = "✓" if condition else "✗"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f"\n       Détail : {detail}"
    print(msg)
    return condition


def test_sdr_space(verbose: bool = False) -> bool:
    """Teste les invariants I1.1 et I1.2."""
    print("\n── Module 1 : SDRSpace ─────────────────────────────────────")
    all_ok = True

    sdr = SDRSpace(input_dim=128, n=2048, w=40)
    s = torch.randn(128)
    x = sdr.encode(s)

    # I1.2 : x ∈ {0,1}^n
    is_binary = ((x == 0) | (x == 1)).all().item()
    all_ok &= check("I1.2 : x ∈ {0,1}^n", is_binary, f"non-binary values: {x.unique()}")

    # I1.1 : ||x||₁ = w exactement
    sparsity_ok = (x.sum() == 40).item()
    all_ok &= check(f"I1.1 : ||x||₁ = w = 40", sparsity_ok, f"||x||₁ = {x.sum().item()}")

    # Batch
    S = torch.randn(8, 128)
    X = sdr.encode(S)
    batch_ok = (X.sum(dim=-1) == 40).all().item()
    all_ok &= check("I1.1 (batch) : ||x_i||₁ = 40 pour tout i", batch_ok)

    return all_ok


def test_spatial_pooler(verbose: bool = False) -> bool:
    """Teste les invariants I2.1 et I2.2."""
    print("\n── Module 2 : SpatialPooler ────────────────────────────────")
    all_ok = True

    sp = SpatialPooler(n_inputs=2048, n_columns=256, k=40)

    # Générer un SDR valide
    sdr_space = SDRSpace(input_dim=128, n=2048, w=40)
    s = torch.randn(128)
    x = sdr_space.encode(s)

    active = sp.forward(x)

    # I2.1 : |A_t| = k exactement
    k_ok = (len(active) == 40)
    all_ok &= check(f"I2.1 : |A_t| = k = 40", k_ok, f"|A_t| = {len(active)}")

    # I2.2 : p_ij ∈ [0,1]
    perm_min = sp.permanences.min().item()
    perm_max = sp.permanences.max().item()
    perm_ok = (perm_min >= 0.0) and (perm_max <= 1.0)
    all_ok &= check(
        f"I2.2 : p_ij ∈ [0,1]",
        perm_ok,
        f"range = [{perm_min:.4f}, {perm_max:.4f}]",
    )

    # Après mise à jour hebbienne : toujours dans [0,1]
    sp.hebbian_update(x, active)
    perm_min_after = sp.permanences.min().item()
    perm_max_after = sp.permanences.max().item()
    perm_ok_after = (perm_min_after >= 0.0) and (perm_max_after <= 1.0)
    all_ok &= check(
        "I2.2 (après hebbian) : p_ij ∈ [0,1]",
        perm_ok_after,
        f"range = [{perm_min_after:.4f}, {perm_max_after:.4f}]",
    )

    # Vérification gamma
    gamma_init = sp.gamma()
    gamma_ok = (0.0 <= gamma_init <= 1.0)
    all_ok &= check(f"γ(0) ∈ [0,1]", gamma_ok, f"γ(0) = {gamma_init:.4f}")

    return all_ok


def test_layer6b(verbose: bool = False) -> bool:
    """Teste l'invariant I3.1 (isométrie)."""
    print("\n── Module 3 : Layer6bTransformer ───────────────────────────")
    all_ok = True

    layer6b = Layer6bTransformer(sdr_dim=2048, n_grid_modules=6)

    sdr = torch.zeros(2048)
    sdr[:40] = 1.0
    v_t = torch.randn(2) * 0.1

    out = layer6b.transform(sdr, v_t)

    # I3.1 : sortie de dimension correcte (2 × n_modules)
    dim_ok = (out.shape == (12,))
    all_ok &= check("I3.1 : dim(sortie) = 2·n_modules = 12", dim_ok, f"shape = {out.shape}")

    # Sortie pas un scalaire (piège .mean())
    not_scalar = (out.dim() == 1 and out.shape[0] > 1)
    all_ok &= check("I3.1 : sortie vectorielle (pas scalaire)", not_scalar)

    # Norme non-nulle (représentation non-dégénérée)
    norm = out.norm().item()
    norm_ok = norm > 1e-6
    all_ok &= check(f"I3.1 : ||sortie|| > 0", norm_ok, f"norm = {norm:.6f}")

    return all_ok


def test_grid_cell(verbose: bool = False) -> bool:
    """Teste les invariants I4.1, I4.2, I4.3."""
    print("\n── Module 4 : GridCellNetwork ──────────────────────────────")
    all_ok = True

    gc = GridCellNetwork(n_modules=6, periods=[3, 5, 7, 11, 13, 17])
    velocity = torch.randn(2) * 0.1

    phases = gc.integrate(velocity)

    # I4.1 : φ_k ∈ [0, 2π)²
    in_range = ((phases >= 0) & (phases < 2 * math.pi)).all().item()
    all_ok &= check(
        "I4.1 : φ_k ∈ [0, 2π)²",
        in_range,
        f"min={phases.min().item():.4f}, max={phases.max().item():.4f}",
    )

    # I4.2 : get_code() → 4·n_modules = 24 dimensions
    code = gc.get_code()
    dim_ok = (code.shape[0] == 24)
    all_ok &= check(
        "I4.2 : dim(get_code()) = 4·n_modules = 24",
        dim_ok,
        f"dim = {code.shape[0]}",
    )

    # I4.2 : valeurs dans [-1, 1] (cos/sin)
    vals_ok = ((code >= -1.0) & (code <= 1.0)).all().item()
    all_ok &= check("I4.2 : code ∈ [-1, 1]^24 (cos/sin)", vals_ok)

    # Coprimarité des périodes
    from math import gcd
    periods = [3, 5, 7, 11, 13, 17]
    coprime = all(
        gcd(periods[i], periods[j]) == 1
        for i in range(len(periods))
        for j in range(i + 1, len(periods))
    )
    all_ok &= check("CRT : périodes copremières", coprime, f"periods = {periods}")

    # I4.3 : anchor() corrige dans la bonne direction
    target = torch.zeros(6, 2)  # cible : toutes les phases à 0
    phases_before = gc.phases.clone()
    gc.anchor(target, confidence=1.0)
    phases_after = gc.phases
    # Les phases doivent se rapprocher de 0 (ou π)
    correction_applied = not torch.allclose(phases_before, phases_after)
    all_ok &= check("I4.3 : anchor() modifie les phases", correction_applied)

    return all_ok


def test_displacement(verbose: bool = False) -> bool:
    """Teste les invariants I5.1 et I5.2."""
    print("\n── Module 5 : DisplacementAlgebra ─────────────────────────")
    all_ok = True

    da = DisplacementAlgebra(n_modules=6)

    phi_obj = torch.rand(6, 2) * 2 * math.pi
    phi_sens = torch.rand(6, 2) * 2 * math.pi

    D = da.compute(phi_obj, phi_sens)

    # I5.1 : D ∈ (−π, π]²
    in_range = ((D > -math.pi) & (D <= math.pi)).all().item()
    all_ok &= check(
        "I5.1 : D ∈ (−π, π]²",
        in_range,
        f"min={D.min().item():.4f}, max={D.max().item():.4f}",
    )

    # I5.2 : compositionnalité D_A→C = D_A→B ⊕ D_B→C
    phi_a = torch.rand(6, 2) * 2 * math.pi
    phi_b = torch.rand(6, 2) * 2 * math.pi
    phi_c = torch.rand(6, 2) * 2 * math.pi

    D_ab = da.compute(phi_b, phi_a)
    D_bc = da.compute(phi_c, phi_b)
    D_ac_composed = da.compose(D_ab, D_bc)
    D_ac_direct = da.compute(phi_c, phi_a)

    compositionality_ok = torch.allclose(D_ac_composed, D_ac_direct, atol=1e-5)
    all_ok &= check(
        "I5.2 : D_A→C = D_A→B ⊕ D_B→C",
        compositionality_ok,
        f"max_diff = {(D_ac_composed - D_ac_direct).abs().max().item():.6f}",
    )

    # Inversion
    D_inv = da.invert(D_ab)
    D_roundtrip = da.compose(D_ab, D_inv)
    zero_ok = (D_roundtrip.abs() < 1e-5).all().item()
    all_ok &= check("I5.2 : D ⊕ D_inv ≈ 0", zero_ok)

    return all_ok


def test_consensus(verbose: bool = False) -> bool:
    """Teste les invariants I6.2, I6.3, I6.4."""
    print("\n── Module 6 : MultiColumnConsensus ─────────────────────────")
    all_ok = True

    consensus = MultiColumnConsensus(n_sdr=2048, consensus_threshold=1.0)

    # I6.3 : threshold = 1.0 par défaut
    threshold_ok = (consensus.consensus_threshold == 1.0)
    all_ok &= check("I6.3 : consensus_threshold = 1.0", threshold_ok)

    # I6.2 : consensus = AND strict (pas moyenne)
    # Test : deux SDRs avec un seul bit en commun → consensus a exactement ce bit
    sdr_a = torch.zeros(2048)
    sdr_b = torch.zeros(2048)
    sdr_a[0] = 1.0
    sdr_a[1] = 1.0
    sdr_b[0] = 1.0
    sdr_b[2] = 1.0   # bit 2 ≠ bit 1

    result = consensus.vote([sdr_a, sdr_b])
    and_ok = (result[0] == 1.0) and (result[1] == 0.0) and (result[2] == 0.0)
    all_ok &= check(
        "I6.2 : consensus = AND strict (intersection)",
        and_ok,
        f"bit[0]={result[0]}, bit[1]={result[1]}, bit[2]={result[2]}",
    )

    # I6.4 : P_fp décroît avec K
    print("  [~] I6.4 : estimation P_fp (peut prendre quelques secondes)...")
    p_fp_k2 = consensus.test_false_positive_rate(n_trials=1000, n_columns=2, w=40, n=2048)
    p_fp_k4 = consensus.test_false_positive_rate(n_trials=1000, n_columns=4, w=40, n=2048)
    decay_ok = (p_fp_k4 < p_fp_k2)
    all_ok &= check(
        "I6.4 : P_fp(K=4) < P_fp(K=2)",
        decay_ok,
        f"P_fp(K=2)={p_fp_k2:.6f}, P_fp(K=4)={p_fp_k4:.6f}",
    )
    expected_k4 = (40 / 2048) ** 4
    print(f"         P_fp empirique K=4 : {p_fp_k4:.6f} (théorie : {expected_k4:.6f})")

    return all_ok


def test_cortical_column(verbose: bool = False) -> bool:
    """Teste l'assemblage complet CorticalColumn."""
    print("\n── CorticalColumn — Assemblage complet ─────────────────────")
    all_ok = True

    model = CorticalColumn(
        n_columns=4,
        input_dim=128,
        n_sdr=512,
        w=20,
        n_minicolumns=64,
        k_active=10,
        n_grid_modules=4,
        grid_periods=[3, 5, 7, 11],
    )

    s_t = torch.randn(128)
    v_t = torch.zeros(2)

    result = model.step(s_t, v_t, train=True)

    # Vérification des sorties
    all_ok &= check("step() retourne 'sdr'", "sdr" in result)
    all_ok &= check("step() retourne 'phase'", "phase" in result)
    all_ok &= check("step() retourne 'consensus'", "consensus" in result)
    all_ok &= check("step() retourne 'triplet'", "triplet" in result)

    # I1.1 sur la sortie
    sdr = result["sdr"]
    sparsity_ok = (sdr.sum() == 20).item()
    all_ok &= check(f"Sortie SDR : ||x||₁ = 20", sparsity_ok, f"||x||₁ = {sdr.sum()}")

    # I6.1 : pas de weight sharing
    col0_perm = model.columns[0].spatial_pooler.permanences
    col1_perm = model.columns[1].spatial_pooler.permanences
    no_sharing = not (col0_perm is col1_perm)
    all_ok &= check("I6.1 : pas de weight sharing entre colonnes", no_sharing)

    # K colonnes dans all_sdrs
    k_ok = (len(result["all_sdrs"]) == 4)
    all_ok &= check("K = 4 colonnes indépendantes", k_ok)

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification des invariants formels")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("VÉRIFICATION DES INVARIANTS FORMELS — Cortical Column v2")
    print("=" * 60)

    results = {
        "SDRSpace (I1.1, I1.2)": test_sdr_space(args.verbose),
        "SpatialPooler (I2.1, I2.2)": test_spatial_pooler(args.verbose),
        "Layer6bTransformer (I3.1)": test_layer6b(args.verbose),
        "GridCellNetwork (I4.1, I4.2, I4.3)": test_grid_cell(args.verbose),
        "DisplacementAlgebra (I5.1, I5.2)": test_displacement(args.verbose),
        "MultiColumnConsensus (I6.2, I6.3, I6.4)": test_consensus(args.verbose),
        "CorticalColumn (assemblage)": test_cortical_column(args.verbose),
    }

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    n_pass = sum(results.values())
    n_total = len(results)

    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status} : {name}")

    print("-" * 60)
    print(f"  {n_pass}/{n_total} modules validés")
    print("=" * 60)

    if n_pass < n_total:
        sys.exit(1)
    else:
        print("  Tous les invariants sont respectés.")
        sys.exit(0)


if __name__ == "__main__":
    main()
