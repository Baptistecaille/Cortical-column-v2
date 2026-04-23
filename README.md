# Cortical Column World Model (v2)

Implémentation d’un **world model non supervisé** inspiré de la **Thousand Brains Theory** et des **colonnes corticales**.

Le projet vise un modèle biologiquement grounded capable de :
- apprendre des représentations spatiales invariantes (ego → allo),
- intégrer le mouvement sur un tore de phases (grid cells),
- encoder des relations compositionnelles objet-partie,
- agréger plusieurs colonnes indépendantes via un consensus.

---

## 1) Objectif scientifique

Ce dépôt sert à explorer une architecture corticale modulaire où chaque colonne apprend localement, puis contribue à une décision collective.

Axes principaux :
- **encodage sparse strict** (SDR binaire à parcimonie fixe),
- **apprentissage hebbien** (sans autograd dans le SpatialPooler),
- **transformation de référentiel** (Layer6b + boucle thalamique),
- **intégration de chemin** (GridCellNetwork sur 𝕋²),
- **algèbre de déplacement** (relations spatiales composables),
- **consensus inter-colonnes**.

Références de conception et invariants : `CLAUDE.md`.

---

## 2) Architecture (6 modules + assemblage)

### Module 1 — `SDRSpace` (`core/sdr_space.py`)
- Encodage capteur → SDR binaire.
- Invariants :
  - `||x||₁ = w` exactement,
  - `x ∈ {0,1}^n`.

### Module 2 — `SpatialPooler` (`core/spatial_pooler.py`)
- Sélection de minicolonnes actives + plasticité hebbienne.
- Invariants :
  - exactement `k` colonnes actives,
  - permanences dans `[0,1]`,
  - convergence vers états saturés.
- **Pas d’autograd** pour les permanences.

### Module 3 — `Layer6bTransformer` (`core/layer6b.py`)
- Transformation ego → allocentrique.
- Boucle thalamique TC/hT/NRT.
- Sortie vectorielle de phase (pas de réduction scalaire).

### Module 4 — `GridCellNetwork` (`core/grid_cell.py`)
- Intégration de chemin sur tore 2D.
- Phases bornées modulo `2π`.
- `get_code()` retourne `4 × n_modules` composantes cos/sin.
- `anchor()` corrige par pondération en `1/λ_k`.
- Vérification des périodes copremières.

### Module 5 — `DisplacementAlgebra` (`core/displacement.py`)
- Déplacements modulaires `D = φ_objet − φ_capteur`.
- Composition : `D_A→C = D_A→B ⊕ D_B→C`.
- Triplets relationnels pour objet/sous-objet.

### Module 6 — `MultiColumnConsensus` (`core/consensus.py`)
- Agrégation de `K` colonnes.
- Consensus calculé par `vote_fraction >= threshold` (seuil par défaut `1.0`).
- Outils de stats et estimation Monte Carlo du taux de faux positifs.

### Assemblage — `CorticalColumn` (`column.py`)
- Crée `K` colonnes indépendantes (`SingleColumn`) + consensus.
- API principale :
  - `step(s_t, v_t, train=True)` : mode séquentiel (avec phases/grid code),
  - `step_batch(s_batch, train=True)` : mode batch optimisé (MNIST statique),
  - `reset()` : réinitialisation épisode.

---

## 3) Arborescence

```text
Cortical-column-v2/
├── CLAUDE.md
├── column.py
├── core/
│   ├── sdr_space.py
│   ├── spatial_pooler.py
│   ├── layer6b.py
│   ├── grid_cell.py
│   ├── displacement.py
│   └── consensus.py
├── extensions/
│   ├── pe_circuits.py
│   ├── stp_synapse.py
│   ├── astrocyte.py
│   ├── linoss.py
│   └── pac_detector.py
├── eval/
│   ├── unsupervised_eval.py
│   └── benchmark.py
├── scripts/
│   ├── run_invariants.py
│   ├── train.py
│   ├── test_mnist.py
│   └── test_multiview.py
└── docs/
    ├── Formalisation_Mille_Cerveaux.pdf
    ├── roadmap-implementation-v3.md
    └── criteres_evaluation_modules.md
```

---

## 4) Prérequis

- Python 3.10+ (recommandé)
- PyTorch
- torchvision
- scikit-learn (pour NMI / linear probing avancé)
- numpy (utilisé notamment dans PAC)

Exemple d’installation :

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision scikit-learn numpy
```

> Le dépôt ne fournit pas actuellement de `requirements.txt` / `pyproject.toml`.

---

## 5) Démarrage rapide

### Vérifier les invariants formels

```bash
python scripts/run_invariants.py
python scripts/run_invariants.py --verbose
```

### Entraînement non supervisé (MNIST multi-vues)

```bash
python scripts/train.py
python scripts/train.py --n_images 5000 --n_epochs 3 --n_views 5
```

### Test MNIST (pipeline batch, perf CPU/GPU)

```bash
python scripts/test_mnist.py
python scripts/test_mnist.py --n_train 10000 --n_epochs 3 --batch_size 128
python scripts/test_mnist.py --device cpu
```

### Test multi-vues avec intégration de chemin

```bash
python scripts/test_multiview.py
python scripts/test_multiview.py --n_images 2000 --n_views 5 --n_epochs 3
python scripts/test_multiview.py --enable_vote --alpha_divergence 3.0
```

---

## 6) Interface minimale (exemple Python)

```python
import torch
from column import CorticalColumn

model = CorticalColumn(
    n_columns=4,
    input_dim=784,
    n_sdr=2048,
    w=40,
    n_minicolumns=256,
    k_active=40,
    n_grid_modules=6,
    grid_periods=[3, 5, 7, 11, 13, 17],
)

s_t = torch.randn(784)
v_t = torch.zeros(2)
out = model.step(s_t, v_t, train=False)

print(out.keys())
# sdr, sdr_predicted, predicted_consensus, phase, consensus, triplet, ...
```

---

## 7) Évaluation

### `eval/unsupervised_eval.py`
Pipeline d’évaluation non supervisée incluant notamment :
- erreur de reconstruction (`epsilon`),
- violation de parcimonie (`sparsity_violation_rate`),
- réduction de variance par consensus (`var_red`),
- linear probing (`lin_prob`),
- NMI (`nmi`),
- spécialisation de colonnes (`SI`),
- succès prédictif (`pred_success_rate`).

### `eval/benchmark.py`
Squelette de benchmark contre familles JEPA sur :
- CO3Dv2
- CLEVR
- Something-Something v2
- Ego4D
- ImageNet-1K

Les dataloaders spécifiques restent à brancher selon les données disponibles localement.

---

## 8) Extensions biologiques (`extensions/`)

- `pe_circuits.py` : circuits d’erreur signée PE+/PE− (priorité haute).
- `stp_synapse.py` : plasticité court-terme STD/STF.
- `astrocyte.py` : mémoire associative modulée astrocytairement.
- `linoss.py` : dynamique ODE stable type LinOSS.
- `pac_detector.py` : détection PAC temporelle (FFT sur axe temps).

---

## 9) Conventions importantes

- Respect strict des invariants décrits dans `CLAUDE.md`.
- Commentaires/docstrings métier en français.
- États internes persistants via buffers (`register_buffer`) quand pertinent.
- Pour `SpatialPooler`, règles hebbiennes sans autograd.

---

## 10) Roadmap actuelle

Voir `docs/roadmap-implementation-v3.md` :
- Phase 1 core : implémentée et testable,
- Phase 2 PE circuits : implémentée mais intégration encore à consolider selon la roadmap,
- Phase 3 extensions biologiques : implémentées,
- Phase 4 benchmark JEPA : squelette prêt, intégrations datasets à finaliser.

---

## 11) Dépannage rapide

- **`ModuleNotFoundError: No module named 'torch'`**
  - Installez PyTorch dans l’environnement Python actif.
- **Consensus trop faible / vide**
  - Vérifier les hyperparamètres (`w`, `k_active`, `consensus_threshold`) et l’entraînement multi-vues.
- **Benchmarks indisponibles**
  - Vérifier l’accès local aux datasets ciblés et adapter les dataloaders de `eval/benchmark.py`.

---

## 12) Références internes

- Spécification projet : `CLAUDE.md`
- Formalisation mathématique : `docs/Formalisation_Mille_Cerveaux.pdf`
- Backlog invariants : `docs/criteres_evaluation_modules.md`
- Script de vérification : `scripts/run_invariants.py`

