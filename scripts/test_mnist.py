"""
Test du CorticalColumn World Model sur MNIST.

Pipeline :
    1. Chargement MNIST (28×28 = 784 dims)
    2. Entraînement hebbien non-supervisé (train set)
    3. Extraction des représentations (SDR + grid code)
    4. Évaluation : 6 métriques non-supervisées + linear probing
    5. Affichage du rapport

Les vitesses sont nulles pour MNIST (images statiques, pas d'intégration
de chemin spatiale). Le réseau apprend néanmoins des représentations
allocentriques via la transformation L6b.

Usage :
    python scripts/test_mnist.py
    python scripts/test_mnist.py --n_train 5000 --n_columns 4 --verbose
"""

import sys
import os
import math
import torch
import argparse
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from column import CorticalColumn
from eval.unsupervised_eval import UnsupervisedEvaluator, linear_probing_accuracy, compute_nmi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Chargement MNIST ──────────────────────────────────────────────────────────

def load_mnist(
    n_train: int = 10_000,
    n_test: int = 2_000,
    data_dir: str = "./data",
) -> tuple:
    """
    Charge MNIST et retourne des tenseurs (inputs, labels).

    Les images sont normalisées dans [0, 1] et aplaties en vecteurs 784D.

    Args:
        n_train: nombre d'exemples d'entraînement
        n_test:  nombre d'exemples de test
        data_dir: dossier de téléchargement

    Returns:
        (X_train, y_train, X_test, y_test)
        X : shape (N, 784), y : shape (N,)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),   # aplatissement 28×28 → 784
    ])

    train_ds = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Sous-ensemble
    train_ds = Subset(train_ds, range(min(n_train, len(train_ds))))
    test_ds = Subset(test_ds, range(min(n_test, len(test_ds))))

    def ds_to_tensors(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        Xs, ys = [], []
        for x, y in loader:
            Xs.append(x)
            ys.append(y)
        return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)

    X_train, y_train = ds_to_tensors(train_ds)
    X_test, y_test = ds_to_tensors(test_ds)

    return X_train, y_train, X_test, y_test


# ── Entraînement hebbien ──────────────────────────────────────────────────────

def train_hebbian(
    model: CorticalColumn,
    X_train: torch.Tensor,
    n_epochs: int = 1,
    log_every: int = 500,
) -> None:
    """
    Entraîne le SpatialPooler via la règle hebbienne sur MNIST.

    Les vitesses sont nulles (images statiques).
    Layer6b et GridCells s'adaptent via leur propre dynamique interne.

    Args:
        model:     CorticalColumn
        X_train:   images MNIST aplaties, shape (N, 784)
        n_epochs:  passes sur le dataset
        log_every: fréquence de log (en images)
    """
    N = X_train.shape[0]
    v_zero = torch.zeros(2)
    model.reset()

    for epoch in range(n_epochs):
        # Permutation aléatoire à chaque epoch
        perm = torch.randperm(N)

        for i, idx in enumerate(perm):
            model.step(X_train[idx], v_zero, train=True)

            if (i + 1) % log_every == 0:
                gamma = model.columns[0].spatial_pooler.gamma()
                perm_stats = model.columns[0].spatial_pooler.permanence_stats()
                logger.info(
                    f"Epoch {epoch+1} | Image {i+1:5d}/{N} | "
                    f"γ={gamma:.4f} | "
                    f"p̄={perm_stats['mean']:.3f} | "
                    f"p_conn={perm_stats['frac_connected']:.3f} | "
                    f"p→1={perm_stats['frac_near_one']:.3f}"
                )


# ── Extraction des représentations ───────────────────────────────────────────

@torch.no_grad()
def extract_representations(
    model: CorticalColumn,
    X: torch.Tensor,
    use_grid_code: bool = True,
) -> torch.Tensor:
    """
    Extrait les représentations du modèle pour un jeu de données.

    Représentation = concat(SDR_colonne0, grid_code_colonne0).

    Args:
        model:         CorticalColumn
        X:             images, shape (N, 784)
        use_grid_code: si True, concatène le grid code au SDR

    Returns:
        reprs: shape (N, n_sdr + 4·n_modules) ou (N, n_sdr)
    """
    model.reset()
    v_zero = torch.zeros(2)
    reprs = []

    for i in range(X.shape[0]):
        result = model.step(X[i], v_zero, train=False)
        sdr = result["sdr"].float()

        if use_grid_code:
            gc = result["all_grid_codes"][0].float()
            rep = torch.cat([sdr, gc])
        else:
            rep = sdr

        reprs.append(rep)

    return torch.stack(reprs, dim=0)  # (N, dim)


# ── Rapport ───────────────────────────────────────────────────────────────────

def print_report(metrics: dict, elapsed: float) -> None:
    """Affiche le rapport d'évaluation formaté."""
    print("\n" + "=" * 58)
    print("  RÉSULTATS — CorticalColumn on MNIST")
    print("=" * 58)

    labels = {
        "epsilon":                  ("ε  (erreur reconstruction)", "↓", "{:.4f}"),
        "sparsity_violation_rate":  ("Violation sparsité I1.1",    "=0", "{:.4f}"),
        "var_red":                  ("var_red (bénéfice vote)",     "↑", "{:.4f}"),
        "lin_prob_sdr":             ("Linear probe (SDR seul)",     "↑", "{:.2%}"),
        "lin_prob_full":            ("Linear probe (SDR+GridCode)", "↑", "{:.2%}"),
        "nmi":                      ("NMI k-means (10 clusters)",   "↑", "{:.4f}"),
        "SI":                       ("SI (spécialisation col.)",    "↑", "{:.4f}"),
    }

    for key, (name, direction, fmt) in labels.items():
        val = metrics.get(key, float("nan"))
        formatted = fmt.format(val) if not math.isnan(val) else "N/A"
        print(f"  {name:<38} {direction:>2}  {formatted:>8}")

    print("-" * 58)
    print(f"  Temps total : {elapsed:.1f}s")
    print("=" * 58 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test CorticalColumn sur MNIST")
    parser.add_argument("--n_train", type=int, default=5000,
                        help="Exemples d'entraînement (défaut 5000)")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Exemples de test (défaut 1000)")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Epochs d'entraînement hebbien (défaut 1)")
    parser.add_argument("--n_columns", type=int, default=4,
                        help="Nombre de colonnes K (défaut 4)")
    parser.add_argument("--n_sdr", type=int, default=2048,
                        help="Dimension SDR (défaut 2048)")
    parser.add_argument("--w", type=int, default=40,
                        help="Parcimonie SDR (défaut 40)")
    parser.add_argument("--n_minicolumns", type=int, default=256,
                        help="Minicolonnes par colonne, carré parfait (défaut 256=16²)")
    parser.add_argument("--k_active", type=int, default=40,
                        help="Colonnes actives (défaut 40)")
    parser.add_argument("--n_grid_modules", type=int, default=6,
                        help="Modules grid cells (défaut 6)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dossier de données MNIST")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    t0 = time.time()

    # ── 1. Chargement MNIST ──────────────────────────────────────────────
    logger.info(f"Chargement MNIST ({args.n_train} train, {args.n_test} test)...")
    X_train, y_train, X_test, y_test = load_mnist(
        n_train=args.n_train,
        n_test=args.n_test,
        data_dir=args.data_dir,
    )
    logger.info(f"  Train : {X_train.shape}, Test : {X_test.shape}")

    # ── 2. Construction du modèle ────────────────────────────────────────
    grid_periods = [3, 5, 7, 11, 13, 17][:args.n_grid_modules]
    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,           # 28×28 MNIST
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=grid_periods,
        consensus_threshold=1.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Modèle : {args.n_columns} colonnes | SDR {args.n_sdr}D w={args.w} | "
        f"Grid {4*args.n_grid_modules}D | {n_params:,} paramètres"
    )

    # ── 3. Entraînement hebbien ──────────────────────────────────────────
    logger.info(f"Entraînement hebbien ({args.n_epochs} epoch(s))...")
    train_hebbian(
        model, X_train,
        n_epochs=args.n_epochs,
        log_every=500 if args.verbose else 2000,
    )
    t_train = time.time()
    logger.info(f"  Entraînement terminé en {t_train - t0:.1f}s")

    # ── 4. Évaluation non-supervisée (6 métriques) ───────────────────────
    logger.info("Évaluation des 6 métriques non-supervisées...")
    evaluator = UnsupervisedEvaluator(model, expected_w=args.w, n_classes=10)

    v_zeros_test = torch.zeros(args.n_test, 2)
    with torch.no_grad():
        base_metrics = evaluator.evaluate(X_test, v_zeros_test, labels=y_test)

    metrics = {
        "epsilon":               base_metrics["epsilon"],
        "sparsity_violation_rate": base_metrics["sparsity_violation_rate"],
        "var_red":               base_metrics["var_red"],
        "nmi":                   base_metrics.get("nmi", float("nan")),
        "SI":                    base_metrics["SI"],
    }

    # ── 5. Linear probing ────────────────────────────────────────────────
    logger.info("Extraction des représentations pour linear probing...")

    # SDR seul
    reprs_train_sdr = extract_representations(model, X_train, use_grid_code=False)
    reprs_test_sdr = extract_representations(model, X_test, use_grid_code=False)

    # SDR + grid code
    reprs_train_full = extract_representations(model, X_train, use_grid_code=True)
    reprs_test_full = extract_representations(model, X_test, use_grid_code=True)

    logger.info(f"  Dim représentation SDR  : {reprs_train_sdr.shape[1]}")
    logger.info(f"  Dim représentation full : {reprs_train_full.shape[1]}")

    logger.info("Linear probing (SDR seul)...")
    acc_sdr = linear_probing_accuracy(
        torch.cat([reprs_train_sdr, reprs_test_sdr], dim=0),
        torch.cat([y_train[:args.n_train], y_test], dim=0),
        n_classes=10,
        n_epochs=100,
    )

    logger.info("Linear probing (SDR + GridCode)...")
    acc_full = linear_probing_accuracy(
        torch.cat([reprs_train_full, reprs_test_full], dim=0),
        torch.cat([y_train[:args.n_train], y_test], dim=0),
        n_classes=10,
        n_epochs=100,
    )

    metrics["lin_prob_sdr"] = acc_sdr
    metrics["lin_prob_full"] = acc_full

    # ── 6. Rapport ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_report(metrics, elapsed)

    # Statistiques des permanences (santé du SpatialPooler)
    if args.verbose:
        print("── Permanences SpatialPooler (colonne 0) ──")
        stats = model.columns[0].spatial_pooler.permanence_stats()
        for k, v in stats.items():
            print(f"  {k:<25} {v:.4f}")
        print()


if __name__ == "__main__":
    main()
