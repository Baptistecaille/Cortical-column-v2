# Critères d'évaluation des modules — Backlog des invariants

## Invariants formels par module

### Module 1 — SDRSpace

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I1.1 | `||x||₁ == w` | `assert (x.sum() == w).all()` | Strict : 0 tolérance |
| I1.2 | `x ∈ {0,1}^n` | `assert ((x==0) | (x==1)).all()` | Strict |

**Test de régression :**
```python
sdr = SDRSpace(n=2048, w=40)
x = sdr.encode(torch.randn(128))
assert x.sum() == 40
assert ((x == 0) | (x == 1)).all()
```

---

### Module 2 — SpatialPooler

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I2.1 | `|A_t| == k` exactement | `assert len(active) == k` | Strict |
| I2.2 | `p_ij ∈ [0,1]` | `assert (p >= 0).all() and (p <= 1).all()` | Strict |
| I2.3 | Convergence vers {0,1} | `frac_near_zero + frac_near_one → 1` | > 0.9 après 10k pas |

**Métriques de suivi :**
- `duty_cycle.mean()` doit être proche de `k/n_columns`
- `gamma(t)` doit décroître de 1.0 à 0.0 entre `m` et `m+τ`
- Pas de neurones silencieux (duty_cycle jamais 0 grâce au boost)

**Test d'annealing :**
```python
sp = SpatialPooler(n_inputs=2048, n_columns=256, k=40, newborn_steps=100, tau_decay=500)
# t < 100 → gamma == 1.0
# 100 < t < 600 → gamma décroît linéairement
# t > 600 → gamma == 0.0
```

---

### Module 3 — Layer6bTransformer

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I3.1 | Isométrie (norme préservée) | `|‖out‖ - ‖in‖| < ε` | ε = 0.1 (LayerNorm ~isométrie) |
| — | Sortie vectorielle (pas scalaire) | `out.shape == (2*n_modules,)` | Strict |
| — | Boucle TC/hT/NRT unique | Pas de double intégration | Inspection code |

**Test de non-régression (piège .mean()) :**
```python
layer = Layer6bTransformer(sdr_dim=2048, n_grid_modules=6)
out = layer.transform(sdr, v)
assert out.shape == (12,)  # PAS out.shape == () ou (1,)
```

---

### Module 4 — GridCellNetwork

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I4.1 | `φ_k ∈ [0, 2π)²` | `assert (phases >= 0).all() and (phases < 2π).all()` | Strict |
| I4.2 | `get_code()` → 4·n_modules dim | `assert code.shape == (4*n_modules,)` | Strict |
| I4.2 | Valeurs cos/sin | `assert (code >= -1).all() and (code <= 1).all()` | Strict |
| I4.3 | Correction 1/λ_k | `phases_haute_res corrigées plus que basse_res` | Qualitatif |
| CRT | Périodes copremières | `gcd(λ_i, λ_j) == 1` pour tous i≠j | Strict |

**Test de capacité CRT :**
```python
gc = GridCellNetwork(periods=[3, 5, 7, 11])
assert gc.position_capacity() == 3*5*7*11  # = 1155 positions uniques
```

---

### Module 5 — DisplacementAlgebra

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I5.1 | `D ∈ (−π, π]²` | `assert (D > -π).all() and (D <= π).all()` | Strict |
| I5.2 | Compositionnalité | `compose(D_ab, D_bc) ≈ compute(φ_c, φ_a)` | atol=1e-5 |
| — | Inversion | `compose(D, invert(D)) ≈ 0` | atol=1e-5 |

**Test benchmark CLEVR :**
La distance entre déplacements doit être discriminante pour les relations
spatiales (gauche/droite < 1.0, gauche/devant > 2.0).

---

### Module 6 — MultiColumnConsensus

| Invariant | Critère | Test | Seuil |
|-----------|---------|------|-------|
| I6.1 | Pas de weight sharing | `col_i.sp.permanences is not col_j.sp.permanences` | Strict |
| I6.2 | AND strict (pas moyenne) | `vote([a,b]) == a AND b` | Strict (0/1) |
| I6.3 | threshold = 1.0 par défaut | `consensus.consensus_threshold == 1.0` | Strict |
| I6.4 | P_fp décroît avec K | `P_fp(K=4) < P_fp(K=2)` | Strict |
| I6.4 | Valeur empirique K=4 | `P_fp ≈ (w/n)^4 ≈ 0.001` | ± ordre de grandeur |

**Test anti-régression (piège de la moyenne) :**
```python
# Ce test DOIT échouer si on utilise mean() > 0.35 au lieu de AND
sdr_a = zeros(2048); sdr_a[0] = 1
sdr_b = zeros(2048); sdr_b[1] = 1
consensus = vote([sdr_a, sdr_b])
assert consensus.sum() == 0  # aucun bit en commun → consensus vide
```

---

## Checklist avant chaque PR

- [ ] `python scripts/run_invariants.py` → tous PASS
- [ ] Aucun `autograd` dans `spatial_pooler.py`
- [ ] `W_23` / permanences initialisées avec `Uniforme(0.3, 0.7)` (pas randn)
- [ ] Sigma = 0.8 pour grille 4×4 (pas 2.0)
- [ ] `get_code()` retourne 4·n_modules dimensions (pas 2)
- [ ] `consensus_threshold = 1.0` (pas 0.35)
- [ ] `theta_ca = 15.0` dans pe_circuits.py (pas 0.15)
- [ ] PAC calculé sur axe temporel (pas spatial)
- [ ] STP STD : u = U constant (pas de mise à jour de u)

---

*Mis à jour : 2026-04-21*
