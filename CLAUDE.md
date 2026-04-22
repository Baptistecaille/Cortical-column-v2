# CLAUDE.md — Cortical Column World Model (v2)

> Lis ce fichier en entier avant de toucher au code.
> Il contient le contexte, le pseudo-algorithme, et les conventions de structure.

---

## 1. Contexte du projet

Baptiste est doctorant en neurosciences computationnelles. L'objectif de la thèse est
de construire un **world model** grounded dans la biologie du néocortex, en
s'appuyant sur l'architecture des **colonnes corticales** (Thousand Brains Theory).

### Objectif scientifique

Démontrer qu'une architecture basée sur des colonnes corticales indépendantes peut :
1. Apprendre des représentations spatiales **invariantes** (ego → allo) sans labels
2. Agréger les prédictions inter-colonnes via un **consensus** (vote, pas moyenne)
3. Faire de l'**intégration de chemin** sur un tore de phases (grid cells)
4. Généraliser à la reconnaissance multi-vue, au raisonnement compositionnel,
   et à la navigation (avantage attendu sur JEPA)

### Cadre théorique (hiérarchie des références)

| Rôle | Papier |
|------|--------|
| Justification théorique ("pourquoi") | LeCun/Dupoux/Malik 2026 (arXiv 2603.15381) — Système A/B/M |
| Pseudoalgorithme minicolonne | Lan et al. 2022 (PC-SNN) |
| Simulation multiscale | Alibaba Orangutan 2024 |
| Implémentation spatiale ("comment") | Thousand Brains Theory, Rao-Ballard, Thalamic Routing |
| Erreur d'encodage signée | Lee 2025 (PE+/PE−) |
| Plasticité synaptique | Waitzmann 2024 (iSTP : PV STD, SST STF) |
| ODE stable | Rusch 2024 (LinOSS) |
| Mémoire associative | Kozachkov 2025 (Astrocyte) |
| Signal d'erreur continu | Max 2025 (GLE) |
| Vote compositionnel | Clay-Leadholm 2024 (Monty/CMP) |

### Comparatif benchmark visé

Le modèle sera comparé à I-JEPA, V-JEPA 2, et Point-JEPA sur :
- CO3Dv2 (invariance rotation — avantage TBT)
- CLEVR (compositionnalité — avantage TBT)
- Something-Something v2 (motion — terrain V-JEPA)
- Ego4D spatial memory (navigation — avantage TBT exclusif)
- ImageNet-1K (baseline représentation)

---

## 2. Architecture des 6 modules (formalisation mathématique)

L'architecture est définie mathématiquement dans `Formalisation_Mille_Cerveaux.pdf`.
Chaque module a des **invariants formels** que le code doit respecter.

### Module 1 — `SDRSpace`
**Rôle :** Encodeur sensoriel → Sparse Distributed Representation binaire  
**Invariant I1.1 :** `||x||₁ = w` exactement (parcimonie *dure*, pas soft)  
**Invariant I1.2 :** `x ∈ {0,1}^n`  
**Paramètres typiques :** `n=2048, w=40` (ratio ~2%)  
**Piège :** Ne jamais seuiller un vecteur flottant — utiliser `top-k` strict + binarisation

### Module 2 — `SpatialPooler`
**Rôle :** Apprentissage hebbien non-supervisé des permanences synaptiques  
**Invariant I2.1 :** Exactement `k` colonnes actives à chaque pas  
**Invariant I2.2 :** `p_ij ∈ [0,1]` (permanences bornées)  
**Invariant I2.3 :** Convergence vers `{0,1}` (Thm 2.2)  
**Annealing (§7.2) :** `γ(t)` schedule linéaire par morceaux — newborn stage `m` itérations,
puis décroissance sur `τ_decay`. Deux effets : boost `b_i*(t) = 1 + γ(t)·(b_i−1)`,
dépression `δ⁻(t) = δ⁻·γ(t)`  
**Règle :** Pas d'autograd — tout est `@torch.no_grad()` ou `.data`  
**Piège :** L'érosion des permanences à long terme est résolue par l'annealing, pas par `δ⁻` élevé

### Module 3 — `Layer6bTransformer`
**Rôle :** Transformation ego → allocentrique (L6b comme rotateur de référentiel)  
**Invariant I3.1 :** La transformation est une isométrie (norme préservée)  
**Connexion biologique :** L6b projette vers le thalamus (TC) et reçoit le feedback
thalamique (hT/NRT) — ne pas supprimer cette boucle  
**Piège :** Réduire L6b à un `.mean()` détruit toute la structure spatiale — sortie doit
rester un vecteur de phase 2D par module, pas un scalaire

### Module 4 — `GridCellNetwork`
**Rôle :** Intégration de chemin sur tore 2D `𝕋²`  
**Invariant I4.1 :** `φ_k ∈ [0, 2π)²` pour tout module k (mise à jour `mod 2π`)  
**Invariant I4.2 :** `get_code()` produit `[cos(φ₀), sin(φ₀), cos(φ₁), sin(φ₁)]` par module
→ dimension totale `4·n_modules` (pas `2·n_modules`)  
**Invariant I4.3 :** Correction `anchor()` pondérée par `1/λ_k` (pas uniforme)  
**Piège CRT :** Les périodes `λ_k` doivent être **copremières** entre elles pour que la
résolution de position soit sans ambiguïté sur `∏λ_k` unités  
**Pourquoi le tore :** La périodicité du signal de grid cell impose `ℝ²/Λ ≅ 𝕋²` — confirmé
empiriquement par Gardner et al. 2022 (homologie persistante sur enregistrements de rat)

### Module 5 — `DisplacementAlgebra`
**Rôle :** Représentation relationnelle `⟨SDR_parent, SDR_subobject, D_subobject⟩`  
**Invariant I5.1 :** `D = φ_objet − φ_capteur` (soustraction modulaire sur 𝕋²)  
**Invariant I5.2 :** Compositionnalité — `D_A→C = D_A→B ⊕ D_B→C`  
**Usage :** Encoder les relations spatiales objet-partie (ex. tasse/logo)  
**Benchmark direct :** CLEVR tasks (relations gauche/droite/devant/derrière)

### Module 6 — `MultiColumnConsensus`
**Rôle :** Agrégation des votes de K colonnes indépendantes  
**Invariant I6.1 :** Pas de weight sharing entre colonnes (chaque colonne spécialisée)  
**Invariant I6.2 :** Consensus = **intersection** (AND logique), pas moyenne  
**Invariant I6.3 :** `vote_fraction >= consensus_threshold` avec `threshold=1.0` par défaut  
**Invariant I6.4 :** Décroissance exponentielle du taux de faux positifs avec K
(Thm 6.3) — validé : K=4 colonnes → P_fp ≈ 0.001  
**Piège :** `(sdr_stack * gamma_stack).mean(dim=0) > 0.35` est une moyenne pondérée,
pas un AND — cela viole I6.2 et brise la garantie du Thm 6.3

### Classe d'assemblage — `CorticalColumn`
```python
class CorticalColumn:
    modules: [SDRSpace, SpatialPooler, Layer6bTransformer,
              GridCellNetwork, DisplacementAlgebra, MultiColumnConsensus]

    def step(self, sensory_input, velocity) -> dict:
        sdr       = self.sdr_space.encode(sensory_input)         # I1.1, I1.2
        active    = self.spatial_pooler.forward(sdr)              # I2.1
        allo      = self.layer6b.transform(active, velocity)      # I3.1
        phase     = self.grid_cell.integrate(allo, velocity)      # I4.1, I4.2
        disp      = self.displacement.compute(phase, active)      # I5.1
        consensus = self.consensus.vote([active, ...])            # I6.2
        self.spatial_pooler.hebbian_update(sdr, active)           # I2.3
        return {"sdr": sdr, "phase": phase, "consensus": consensus}
```

---

## 3. Pseudo-algorithme global

```
ENTRÉE : séquence sensorielle S = {s_t}, vecteurs de vitesse v_t
SORTIE : représentations allocentriques, consensus inter-colonnes

INITIALISATION :
  Pour chaque colonne c ∈ {1..K} (indépendantes, pas de weight sharing) :
    permanences p_ij ~ Uniforme(0.3, 0.7)
    phase φ_k ~ Uniforme(0, 2π)² pour chaque module k
    t_step ← 0

BOUCLE PRINCIPALE (pour chaque pas t) :
  ┌─ ENCODAGE (L4 / SDRSpace) ──────────────────────────────────────────
  │  x_t ← top_k(W_enc · s_t, w)     # top-k strict, binarisation
  │  Invariant : ||x_t||₁ = w exactement
  │
  ├─ SÉLECTION DE COLONNES (L2/3 / SpatialPooler) ──────────────────────
  │  γ(t) ← annealing_factor(t, m, τ_decay)   # § 7.2
  │  b_i* ← 1 + γ(t) · (exp(β(μ_voisins − d_i)) − 1)
  │  overlap_i ← b_i* · (p_i · x_t)          # boost corrigé
  │  A_t ← top_k(overlap_i, k)               # k colonnes gagnantes
  │  Invariant : |A_t| = k
  │
  ├─ TRANSFORMATION EGO→ALLO (L6b / Layer6bTransformer) ────────────────
  │  Pour chaque module grille k :
  │    φ_k ← (φ_k + R_k · v_t) mod 2π       # rotation sur 𝕋²
  │    code_k ← [cos(φ_k,0), sin(φ_k,0), cos(φ_k,1), sin(φ_k,1)]
  │  grid_code ← concat(code_k, k=1..n_mod)  # dim = 4·n_modules
  │  Boucle thalamique (TC/hT/NRT) mise à jour *une seule fois*
  │
  ├─ INTÉGRATION DE CHEMIN (GridCellNetwork) ────────────────────────────
  │  anchor() : correction pondérée par 1/λ_k si repère visuel détecté
  │  Invariant : périodes λ_k copremières (CRT)
  │
  ├─ ALGÈBRE DE DÉPLACEMENT (DisplacementAlgebra) ──────────────────────
  │  D_t ← φ_objet − φ_capteur (mod 2π)     # soustraction modulaire
  │  triplet ← ⟨SDR_parent, SDR_objet, D_t⟩
  │
  ├─ APPRENTISSAGE HEBBIEN (SpatialPooler, @no_grad) ───────────────────
  │  Pour chaque colonne i ∈ A_t :
  │    p_ij += δ⁺   si x_t,j = 1  (potentiation)
  │    p_ij -= δ⁻·γ(t) si x_t,j = 0  (dépression annealed)
  │    p_ij ← clamp(p_ij, 0, 1)
  │  Mettre à jour duty cycle EMA
  │  t_step += 1
  │
  └─ CONSENSUS INTER-COLONNES (MultiColumnConsensus) ───────────────────
     Pour K colonnes indépendantes :
       votes ← [SDR_colonne_c pour c ∈ 1..K]
       consensus ← AND(votes)   # intersection stricte, pas moyenne
       P_fp décroît exponentiellement avec K (Thm 6.3)

ÉVALUATION UNSUPERVISED (6 métriques) :
  ε      : reconstruction error
  sparsity : L2/3 sparsity (seuil sur t_max)
  var_red  : vote variance reduction
  lin_prob : linear probing accuracy
  nmi      : k-means NMI clustering
  SI       : column specialization index
```

---

## 4. Structure de code

```
cortical_column/
│
├── CLAUDE.md                    ← ce fichier
│
├── core/
│   ├── __init__.py
│   ├── sdr_space.py             ← SDRSpace (Module 1)
│   │     # SDRSpace.encode(x) → {0,1}^n, ||·||₁=w strict
│   │     # Pas de seuillage flottant : top-k + binarisation uniquement
│   │
│   ├── spatial_pooler.py        ← SpatialPooler (Module 2)
│   │     # forward(), hebbian_update(), update_duty_cycle()
│   │     # gamma(), t_step registré comme buffer (pas Parameter)
│   │     # Aucun autograd — tout @torch.no_grad() ou .data
│   │
│   ├── layer6b.py               ← Layer6bTransformer (Module 3)
│   │     # transform(active, velocity) → vecteur allo
│   │     # Boucle TC/hT/NRT intégrée ici (une seule mise à jour)
│   │     # Sortie : vecteur de phase 2D par module, jamais .mean()
│   │
│   ├── grid_cell.py             ← GridCellNetwork (Module 4)
│   │     # integrate(allo_v, dt) → φ mise à jour sur 𝕋²
│   │     # get_code() → 4·n_modules dimensions (cos/sin par axe)
│   │     # anchor(repere) → correction pondérée 1/λ_k
│   │     # Périodes λ_k : vérifier coprimarité au __init__
│   │
│   ├── displacement.py          ← DisplacementAlgebra (Module 5)
│   │     # compute(φ_objet, φ_capteur) → D modulaire
│   │     # compose(D1, D2) → D1 ⊕ D2
│   │
│   └── consensus.py             ← MultiColumnConsensus (Module 6)
│         # vote(list[SDR]) → intersection (AND), pas moyenne
│         # consensus_threshold=1.0 par défaut (ET strict)
│         # test_false_positive_rate() pour vérifier Thm 6.3
│
├── column.py                    ← CorticalColumn (assemblage)
│     # __init__(n_columns=K, ...) — K colonnes INDÉPENDANTES
│     # step(s_t, v_t) → dict{sdr, phase, consensus, triplet}
│     # Pas de weight sharing entre colonnes
│
├── extensions/                  ← enrichissements biologiques (optionnels)
│   ├── stp_synapse.py           # Tsodyks-Markram (Waitzmann 2024)
│   │     # STD : u=U constant, seul x évolue (PV→E)
│   │     # STF : u évolue (SST→PC)
│   ├── astrocyte.py             # Kozachkov 2025
│   ├── pe_circuits.py           # PE+/PE− signés (Lee 2025)
│   │     # r_Ls_pos et r_Ls_neg séparés — PRIORITÉ 1
│   ├── linoss.py                # LinOSS ODE stable (Rusch 2024)
│   └── pac_detector.py          # PAC temporelle (buffer roulant 100 pas)
│         # FFT sur axe temporel, PAS spatial — éviter confusion domaines
│
├── eval/
│   ├── unsupervised_eval.py     ← pipeline 6 métriques
│   │     # ε, sparsity, var_red, lin_prob, nmi, SI
│   └── benchmark.py            ← comparaison JEPA
│         # CO3Dv2, CLEVR, Something-Something v2, Ego4D, ImageNet-1K
│
├── scripts/
│   ├── train.py                 ← entraînement non-supervisé
│   └── run_invariants.py        ← vérification automatique des 6 invariants
│
└── docs/
    ├── Formalisation_Mille_Cerveaux.pdf   ← référence math primaire
    ├── roadmap-implementation-v3.md       ← 4 phases, 7 lacunes
    └── criteres_evaluation_modules.md     ← backlog des invariants à tester
```

---

## 5. Conventions de code

### Typage et docstrings
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Sélectionne les k colonnes actives par boost homéostatique.

    Invariant I2.1 : |A_t| = k exactement.
    Réf. math : §2.3, Déf. 2.1.

    Args:
        x: SDR binaire, shape (n,), ||x||₁ = w

    Returns:
        active: indices des k colonnes gagnantes, shape (k,)
    """
```

- Commentaires **en français** pour les explications métier
- Références `§X.Y` pointant vers `Formalisation_Mille_Cerveaux.pdf`
- Invariants dans chaque docstring qui les concernent
- Type hints sur toutes les signatures publiques

### Règles PyTorch
- `SpatialPooler` : **aucun autograd** — `.data`, `@torch.no_grad()`, jamais `nn.Parameter` pour les permanences
- `Layer6bTransformer` et `MultiColumnConsensus` : autograd autorisé
- `register_buffer()` pour tout état interne (t_step, duty_cycle, permanences)
- Pas de boucle Python sur les timesteps — vectoriser avec `cumsum`/`argmin`

### Initialisation des poids
- `W_23` (L4→L2/3) : `torch.abs(torch.randn(...)) * 0.3` — **excitateur uniquement**
  (glutamatergique, viole les invariants si négatif)
- Permanences : `Uniforme(0.3, 0.7)`, jamais `torch.randn`
- Sigma inter-colonnes : `0.8` pour grille 4×4 (pas `2.0` — trop large)

---

## 6. Pièges critiques (bugs résolus)

| # | Symptôme | Cause | Fix |
|---|----------|-------|-----|
| B1 | Neurones silencieux, métriques pathologiques | `W_23` initialisé avec `torch.randn` (valeurs négatives) | `torch.abs(...) * 0.3` + clamp ≥ 0 |
| B2 | Sparsité hors spec | Seuil off-by-one sur `t_max` | Corriger le seuil |
| B3 | Sigma trop large sur grille 4×4 | `sigma=2.0` → tout le voisinage actif | `sigma=0.8` |
| B4 | Neurones silencieux dans Hebbian | `t_L23 == t_max−1` non masqués | Masque avant ΔW |
| B5 | Boucle Python lente sur IF | Timestep loop en Python | Vectoriser avec cumsum/argmin |
| B6 | STP STD : `u` mal mis à jour | Équation de facilitation utilisée en mode STD | En STD : `u = U` constant |
| B7 | `get_code()` phase incohérente | `cos(φ[:,0])` et `sin(φ[:,1])` mélangés | 4 composantes par module |
| B8 | `anchor()` correction uniforme | Scalaire ajouté à tous les modules | Pondérer par `1/λ_k` |
| B9 | Double intégration TC | ODE général + boucle thalamique | Séparer les 12 pop. corticales |
| B10 | Consensus = moyenne | `mean() > 0.35` viole Thm 6.3 | AND strict (`vote_fraction >= 1.0`) |
| B11 | PAC spatiale ≠ temporelle | FFT sur vecteur spatial 128D | Buffer roulant 100 pas, FFT temporelle |
| B12 | `theta_ca` saturé | `0.15` → `tanh` ≡ 1 pour toute erreur | `theta_ca = 15.0 Hz` |

---

## 7. Ordre d'implémentation recommandé

```
Phase 1 — Core (valider les 6 invariants) :
  SDRSpace → SpatialPooler → Layer6bTransformer
  → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

Phase 2 — Erreur signée (PRIORITÉ 1 — change la nature du calcul) :
  pe_circuits.py : scinder r_Ls en r_Ls_pos / r_Ls_neg (Lee 2025)

Phase 3 — Richesse biologique (améliorations de performance) :
  stp_synapse.py (iSTP) → astrocyte.py → linoss.py

Phase 4 — Évaluation et benchmark :
  unsupervised_eval.py (6 métriques) → benchmark.py (JEPA comparison)
```

**Ne pas sauter Phase 1 pour aller à Phase 3** — la v3 de `cortical_column.py`
a fait cette erreur et a abouti à un modèle biophysique sans ancrage mathématique.

---

## 8. Fichiers à lire avant de modifier le code

1. `docs/Formalisation_Mille_Cerveaux.pdf` — définitions et théorèmes de référence
2. `docs/roadmap-implementation-v3.md` — priorités et lacunes identifiées
3. `eval/unsupervised_eval.py` — comprendre les 6 métriques avant d'optimiser
4. `scripts/run_invariants.py` — lancer avant et après chaque modification

---

*Dernière mise à jour : 2026-04-21*
