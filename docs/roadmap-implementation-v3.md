# Roadmap d'implémentation v3 — Cortical Column World Model

## 4 phases, 7 lacunes identifiées

---

## Phase 1 — Core (validation des 6 invariants)

**Objectif :** Valider l'architecture de base avant toute extension biologique.

| Module | Fichier | Statut | Invariants |
|--------|---------|--------|-----------|
| SDRSpace | `core/sdr_space.py` | ✓ Implémenté | I1.1, I1.2 |
| SpatialPooler | `core/spatial_pooler.py` | ✓ Implémenté | I2.1, I2.2, I2.3 |
| Layer6bTransformer | `core/layer6b.py` | ✓ Implémenté | I3.1 |
| GridCellNetwork | `core/grid_cell.py` | ✓ Implémenté | I4.1, I4.2, I4.3 |
| DisplacementAlgebra | `core/displacement.py` | ✓ Implémenté | I5.1, I5.2 |
| MultiColumnConsensus | `core/consensus.py` | ✓ Implémenté | I6.1–I6.4 |
| CorticalColumn | `column.py` | ✓ Implémenté | Assemblage |

**Validation :** `python scripts/run_invariants.py`

---

## Phase 2 — Erreur signée (PRIORITÉ 1)

**Objectif :** Scinder r_Ls en r_Ls_pos / r_Ls_neg (Lee 2025).
Change la nature du calcul — ne pas sauter.

| Tâche | Fichier | Statut |
|-------|---------|--------|
| PE+ population (r_Ls_pos) | `extensions/pe_circuits.py` | ✓ Implémenté |
| PE− population (r_Ls_neg) | `extensions/pe_circuits.py` | ✓ Implémenté |
| Intégration dans CorticalColumn | `column.py` | ○ À faire |
| Test theta_ca = 15.0 Hz (bug B12) | `extensions/pe_circuits.py` | ✓ Protégé |

---

## Phase 3 — Richesse biologique

**Objectif :** Extensions améliorant les performances (après Phase 1 validée).

| Module | Fichier | Statut | Priorité |
|--------|---------|--------|----------|
| iSTP (PV STD, SST STF) | `extensions/stp_synapse.py` | ✓ Implémenté | 2 |
| Astrocyte (mémoire associative) | `extensions/astrocyte.py` | ✓ Implémenté | 3 |
| LinOSS ODE stable | `extensions/linoss.py` | ✓ Implémenté | 3 |
| PAC temporel | `extensions/pac_detector.py` | ✓ Implémenté | 4 |

---

## Phase 4 — Évaluation et benchmark

**Objectif :** Comparaison avec I-JEPA, V-JEPA 2, Point-JEPA.

| Tâche | Fichier | Statut |
|-------|---------|--------|
| Pipeline 6 métriques | `eval/unsupervised_eval.py` | ✓ Implémenté |
| Benchmark JEPA | `eval/benchmark.py` | ✓ Squelette |
| Intégration CO3Dv2 | `eval/benchmark.py` | ○ À faire |
| Intégration CLEVR | `eval/benchmark.py` | ○ À faire |
| Intégration SSv2 | `eval/benchmark.py` | ○ À faire |
| Intégration Ego4D | `eval/benchmark.py` | ○ À faire |
| Intégration ImageNet-1K | `eval/benchmark.py` | ○ À faire |

---

## 7 Lacunes identifiées

1. **Intégration PE circuits dans CorticalColumn** — pe_circuits.py est implémenté
   mais pas encore connecté au step() principal. Connexion requise pour Phase 2.

2. **Dataloaders pour benchmarks** — eval/benchmark.py a le squelette mais
   les dataloaders CO3Dv2/CLEVR/SSv2/Ego4D/ImageNet sont à implémenter selon
   la disponibilité locale des données.

3. **Entraînement L6b via autograd** — Layer6bTransformer est entraînable via
   autograd mais le script train.py n'optimise pas encore ces poids (seul
   l'apprentissage hebbien SpatialPooler est actif).

4. **Validation expérimentale Thm 6.3** — test_false_positive_rate() fournit
   une estimation Monte-Carlo mais le test analytique du Thm 6.3 reste à faire.

5. **Sauvegarde/chargement du modèle** — Pas de checkpoint dans train.py.
   À ajouter pour les runs longs.

6. **Multi-GPU** — Chaque colonne est indépendante → parallélisation naturelle
   sur plusieurs GPU non implémentée.

7. **Visualisation des phases grid cell** — Pas de visualisation des trajectoires
   de phases sur le tore 𝕋². Utile pour le debugging de l'intégration de chemin.

---

*Mis à jour : 2026-04-21*
