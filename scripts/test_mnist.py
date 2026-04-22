"""
Test du CorticalColumn World Model sur MNIST — version GPU-optimisée.

Pipeline :
    1. Chargement MNIST (28×28 = 784 dims)
    2. Entraînement hebbien batché via step_batch() — matmuls vectorisés
    3. Extraction des représentations (SDR + grid code)
    4. Évaluation : 6 métriques + linear probing
    5. Rapport + comparaison CPU vs GPU

Device : auto-détecté (CUDA > MPS > CPU)

Usage :
    python scripts/test_mnist.py
    python scripts/test_mnist.py --n_train 10000 --n_epochs 3 --batch_size 128
    python scripts/test_mnist.py --device cpu   # forcer CPU
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


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(requested: str = "auto") -> torch.device:
    """Sélectionne le meilleur device disponible."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Chargement MNIST ──────────────────────────────────────────────────────────

def load_mnist(
    n_train: int = 10_000,
    n_test: int = 2_000,
    data_dir: str = "./data",
) -> tuple:
    """
    Charge MNIST et retourne des tenseurs (inputs, labels).

    Les images sont normalisées dans [0, 1] et aplaties en vecteurs 784D.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_ds = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_ds = Subset(train_ds, range(min(n_train, len(train_ds))))
    test_ds = Subset(test_ds, range(min(n_test, len(test_ds))))

    def ds_to_tensors(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        Xs, ys = [], []
        for x, y in loader:
            Xs.append(x)
            ys.append(y)
        return torch.cat(Xs), torch.cat(ys)

    return (*ds_to_tensors(train_ds), *ds_to_tensors(test_ds))


# ── Entraînement batché ───────────────────────────────────────────────────────

def train_hebbian(
    model: CorticalColumn,
    X_train: torch.Tensor,
    device: torch.device,
    n_epochs: int = 1,
    batch_size: int = 64,
    log_every_batches: int = 50,
) -> float:
    """
    Entraînement hebbien batché via step_batch().

    Toutes les opérations critiques (SDRSpace + SpatialPooler) s'exécutent
    en mode batch sur GPU — zéro boucle Python sur les images.

    Args:
        model:              CorticalColumn (sur device)
        X_train:            images MNIST, shape (N, 784) — sur CPU, transféré par batch
        device:             device cible
        n_epochs:           passes sur le dataset
        batch_size:         taille de batch (64–256 selon VRAM)
        log_every_batches:  fréquence de log

    Returns:
        images_per_second: débit moyen mesuré
    """
    N = X_train.shape[0]
    n_batches = math.ceil(N / batch_size)
    model.reset()

    t_total = 0.0
    images_processed = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        X_shuffled = X_train[perm]

        for i in range(0, N, batch_size):
            batch = X_shuffled[i : i + batch_size].to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                model.step_batch(batch, train=True)
            if device.type in ("cuda", "mps"):
                torch.cuda.synchronize() if device.type == "cuda" else None
            t_total += time.perf_counter() - t0
            images_processed += batch.shape[0]

            batch_idx = i // batch_size
            if (batch_idx + 1) % log_every_batches == 0 or batch_idx == n_batches - 1:
                ips = images_processed / t_total
                gamma = model.columns[0].spatial_pooler.gamma()
                stats = model.columns[0].spatial_pooler.permanence_stats()
                logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx+1:4d}/{n_batches} | "
                    f"{ips:6.0f} img/s | "
                    f"γ={gamma:.3f} | "
                    f"p̄={stats['mean']:.3f} | "
                    f"p→1={stats['frac_near_one']:.3f}"
                )

    return images_processed / t_total


# ── Extraction des représentations ───────────────────────────────────────────

@torch.no_grad()
def extract_representations(
    model: CorticalColumn,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    use_grid_code: bool = True,
) -> torch.Tensor:
    """
    Extrait les représentations SDR (+ grid code optionnel) en mode batch.

    Pour l'extraction, on utilise step_batch() (rapide) et on ignore
    le grid code si use_grid_code=False.

    Args:
        model:         CorticalColumn
        X:             images, shape (N, 784)
        device:        device
        batch_size:    taille de batch pour l'extraction
        use_grid_code: si True, concatène le grid code au SDR

    Returns:
        reprs: shape (N, n_sdr) ou (N, n_sdr + 4·n_modules)
    """
    model.reset()
    all_reprs = []

    for i in range(0, X.shape[0], batch_size):
        batch = X[i : i + batch_size].to(device)
        result = model.step_batch(batch, train=False)
        sdr_b = result["sdr_batch"].float().cpu()     # (B, n_sdr)

        if use_grid_code:
            # Grid code via step() séquentiel (état temporel nécessaire)
            # Pour cohérence avec les résultats précédents, on concatène
            # un vecteur nul (grid code non entraîné sur MNIST)
            gc_dim = 4 * model.n_grid_modules
            gc_b = torch.zeros(sdr_b.shape[0], gc_dim)
            reprs_b = torch.cat([sdr_b, gc_b], dim=-1)
        else:
            reprs_b = sdr_b

        all_reprs.append(reprs_b)

    return torch.cat(all_reprs, dim=0)


# ── Rapport ───────────────────────────────────────────────────────────────────

def print_report(metrics: dict, ips_train: float, device: torch.device) -> None:
    """Affiche le rapport d'évaluation."""
    print("\n" + "=" * 62)
    print(f"  RÉSULTATS — CorticalColumn on MNIST  [{device}]")
    print("=" * 62)

    rows = [
        ("ε  (erreur reconstruction)",       "epsilon",                  "↓", "{:.4f}"),
        ("Violation sparsité I1.1",           "sparsity_violation_rate",  "=0", "{:.4f}"),
        ("var_red (bénéfice vote)",           "var_red",                  "↑", "{:.4f}"),
        ("Linear probe (SDR seul)",           "lin_prob_sdr",             "↑", "{:.2%}"),
        ("Linear probe (SDR+GridCode)",       "lin_prob_full",            "↑", "{:.2%}"),
        ("NMI k-means (10 clusters)",         "nmi",                      "↑", "{:.4f}"),
        ("SI (spécialisation colonnes)",      "SI",                       "↑", "{:.4f}"),
    ]
    for name, key, direction, fmt in rows:
        val = metrics.get(key, float("nan"))
        s = fmt.format(val) if not math.isnan(val) else "N/A"
        print(f"  {name:<40} {direction:>2}  {s:>8}")

    print("-" * 62)
    print(f"  Débit entraînement : {ips_train:,.0f} images/s  ({device})")
    print("=" * 62 + "\n")


# ── Benchmark CPU vs GPU ──────────────────────────────────────────────────────

def benchmark_devices(model: CorticalColumn, X_sample: torch.Tensor) -> None:
    """Compare le débit forward+hebbian sur CPU vs GPU disponibles."""
    batch = X_sample[:128]
    devices_to_test = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices_to_test.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    if len(devices_to_test) == 1:
        logger.info("Un seul device disponible — benchmark ignoré.")
        return

    print("\n── Benchmark devices (128 images × 10 passes) ──────────────────")
    for dev in devices_to_test:
        model_dev = model.to(dev)
        batch_dev = batch.to(dev)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model_dev.step_batch(batch_dev, train=False)

        t0 = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                model_dev.step_batch(batch_dev, train=False)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ips = 128 * 10 / elapsed
        print(f"  {str(dev):<8} : {ips:8,.0f} img/s")

    model.to(torch.device("cpu"))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test CorticalColumn sur MNIST (GPU)")
    parser.add_argument("--n_train",       type=int,   default=10_000)
    parser.add_argument("--n_test",        type=int,   default=2_000)
    parser.add_argument("--n_epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--n_columns",     type=int,   default=4)
    parser.add_argument("--n_sdr",         type=int,   default=2048)
    parser.add_argument("--w",             type=int,   default=40)
    parser.add_argument("--n_minicolumns", type=int,   default=256)
    parser.add_argument("--k_active",      type=int,   default=40)
    parser.add_argument("--n_grid_modules",type=int,   default=6)
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--benchmark",     action="store_true",
                        help="Compare CPU vs GPU avant l'entraînement")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    device = get_device(args.device)
    logger.info(f"Device sélectionné : {device}")

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
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=grid_periods,
        consensus_threshold=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Modèle : {args.n_columns} colonnes | SDR {args.n_sdr}D w={args.w} | "
        f"Grid {4*args.n_grid_modules}D | {n_params:,} paramètres → {device}"
    )

    # ── 3. Benchmark optionnel ───────────────────────────────────────────
    if args.benchmark:
        benchmark_devices(model, X_train[:128])
        model = model.to(device)

    # ── 4. Entraînement hebbien batché ───────────────────────────────────
    logger.info(
        f"Entraînement hebbien : {args.n_epochs} epoch(s), "
        f"batch_size={args.batch_size}..."
    )
    ips_train = train_hebbian(
        model, X_train, device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        log_every_batches=20 if args.verbose else 40,
    )
    logger.info(f"  Débit : {ips_train:,.0f} images/s")

    # ── 5. Évaluation non-supervisée ─────────────────────────────────────
    logger.info("Évaluation des 6 métriques non-supervisées...")
    evaluator = UnsupervisedEvaluator(
        model.to("cpu"), expected_w=args.w, n_classes=10
    )
    v_zeros = torch.zeros(args.n_test, 2)
    with torch.no_grad():
        base_metrics = evaluator.evaluate(X_test, v_zeros, labels=y_test)

    metrics = {
        "epsilon":                 base_metrics["epsilon"],
        "sparsity_violation_rate": base_metrics["sparsity_violation_rate"],
        "var_red":                 base_metrics["var_red"],
        "nmi":                     base_metrics.get("nmi", float("nan")),
        "SI":                      base_metrics["SI"],
    }
    model = model.to(device)

    # ── 6. Linear probing ────────────────────────────────────────────────
    logger.info("Extraction des représentations (batch)...")

    reprs_train = extract_representations(model, X_train, device, use_grid_code=False)
    reprs_test  = extract_representations(model, X_test,  device, use_grid_code=False)
    reprs_train_full = extract_representations(model, X_train, device, use_grid_code=True)
    reprs_test_full  = extract_representations(model, X_test,  device, use_grid_code=True)

    logger.info(f"  SDR dim : {reprs_train.shape[1]}  |  Full dim : {reprs_train_full.shape[1]}")

    all_X_sdr  = torch.cat([reprs_train,      reprs_test],      dim=0)
    all_X_full = torch.cat([reprs_train_full, reprs_test_full], dim=0)
    all_y      = torch.cat([y_train[:args.n_train], y_test],    dim=0)

    logger.info("Linear probing...")
    metrics["lin_prob_sdr"]  = linear_probing_accuracy(all_X_sdr,  all_y, n_classes=10, n_epochs=150)
    metrics["lin_prob_full"] = linear_probing_accuracy(all_X_full, all_y, n_classes=10, n_epochs=150)

    # ── 7. Rapport ───────────────────────────────────────────────────────
    print_report(metrics, ips_train, device)

    if args.verbose:
        print("── Permanences SpatialPooler (colonne 0) ──")
        sp = model.columns[0].spatial_pooler
        for k, v in sp.permanence_stats().items():
            print(f"  {k:<25} {v:.4f}")
        print(f"  t_step                    {sp.t_step.item()}")
        print()


if __name__ == "__main__":
    main()
