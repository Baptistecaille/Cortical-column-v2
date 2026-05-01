"""
Pipeline d'évaluation complète — Phase 4.

Charge un checkpoint (ou utilise des poids aléatoires), exécute
l'évaluation non-supervisée (7 métriques) et les benchmarks proxies MNIST,
puis écrit un rapport JSON et affiche un tableau console.

Usage :
    python scripts/run_eval.py [--checkpoint model.pt] [options]
"""
import sys
import os
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation complète Phase 4 — CorticalColumn World Model"
    )
    # ── Architecture ──────────────────────────────────────────────────────
    parser.add_argument("--n_columns",      type=int,   default=4)
    parser.add_argument("--n_sdr",          type=int,   default=2048)
    parser.add_argument("--w",              type=int,   default=40)
    parser.add_argument("--n_minicolumns",  type=int,   default=256)
    parser.add_argument("--k_active",       type=int,   default=40)
    parser.add_argument("--n_grid_modules", type=int,   default=6)
    parser.add_argument("--grid_periods",   type=str,   default=None,
                        help="Périodes λ_k séparées par virgules ex: 3,5,7,11,13,17")
    # ── Évaluation non-supervisée ─────────────────────────────────────────
    parser.add_argument("--n_samples",      type=int,   default=200,
                        help="Échantillons MNIST pour l'éval non-supervisée")
    # ── Benchmark proxies ─────────────────────────────────────────────────
    parser.add_argument("--n_bench_train",  type=int,   default=500)
    parser.add_argument("--n_bench_test",   type=int,   default=200)
    parser.add_argument("--n_rot_samples",  type=int,   default=100)
    # ── I/O ───────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--output",         type=str,
                        default="./eval_outputs/eval_report_phase4.json")
    parser.add_argument("--data_dir",       type=str,   default="./data")
    parser.add_argument("--device",         type=str,   default="cpu")

    args = parser.parse_args()

    from column import CorticalColumn
    from eval.unsupervised_eval import UnsupervisedEvaluator
    from eval.benchmark import BenchmarkRunner
    import torchvision
    import torchvision.transforms as T

    device = torch.device(args.device)

    periods = (
        [int(p) for p in args.grid_periods.split(",")]
        if args.grid_periods
        else [3, 5, 7, 11, 13, 17][: args.n_grid_modules]
    )

    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=periods,
        consensus_threshold=1.0,
    ).to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERREUR] Checkpoint introuvable : {args.checkpoint}")
            sys.exit(1)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        print(f"[OK] Checkpoint chargé : {args.checkpoint}")
    else:
        print("[INFO] Aucun checkpoint — modèle non entraîné (poids aléatoires)")

    # ── Données MNIST test ────────────────────────────────────────────────
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
    ds = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    n = min(args.n_samples, len(ds))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, range(n)),
        batch_size=n,
        shuffle=False,
        num_workers=0,
    )
    imgs_raw, labels = next(iter(loader))
    imgs = imgs_raw.to(device)
    v_zeros = torch.zeros(n, 2, device=device)

    # ── 1 — Évaluation non-supervisée ────────────────────────────────────
    print("\n[1/2] Évaluation non-supervisée (7 métriques)...")
    evaluator = UnsupervisedEvaluator(model, expected_w=args.w, n_classes=10)
    unsup_metrics = evaluator.evaluate(imgs, v_zeros, labels)

    print("  Métriques :")
    for k, v in unsup_metrics.items():
        if isinstance(v, float) and v == v:
            print(f"    {k:35s} = {v:.4f}")
        else:
            print(f"    {k:35s} = {v}")

    # ── 2 — Benchmarks proxies MNIST ─────────────────────────────────────
    print("\n[2/2] Benchmarks proxies MNIST...")
    runner = BenchmarkRunner(model, device=str(device))

    lin_probe_acc = runner.run_mnist_linear_probe(
        n_train=args.n_bench_train,
        n_test=args.n_bench_test,
        data_dir=args.data_dir,
    )
    print(f"  [proxy ImageNet-1K] Linear probe accuracy : {lin_probe_acc:.4f}")

    rot_inv_score = runner.run_mnist_rotation_benchmark(
        n_samples=args.n_rot_samples,
        data_dir=args.data_dir,
    )
    print(f"  [proxy CO3Dv2]     Rotation invariance   : {rot_inv_score:.4f}")

    # ── Rapport ───────────────────────────────────────────────────────────
    report = {
        "checkpoint": args.checkpoint or "random_weights",
        "n_samples": args.n_samples,
        "unsupervised": unsup_metrics,
        "benchmark": {
            "mnist_linear_probe_acc": lin_probe_acc,
            "mnist_rotation_invariance": rot_inv_score,
        },
    }

    out_path = Path(args.output)
    runner.save_report(report, str(out_path))

    # ── Tableau console ───────────────────────────────────────────────────
    all_metrics = {**unsup_metrics, **report["benchmark"]}
    print("\n" + "=" * 62)
    print("RAPPORT FINAL — Phase 4 : CorticalColumn World Model")
    print("=" * 62)
    print(f"  {'Métrique':<38} {'Valeur':>10}")
    print("-" * 62)
    for k, v in all_metrics.items():
        if isinstance(v, float) and v == v:
            print(f"  {k:<38} {v:>10.4f}")
        elif isinstance(v, float):
            print(f"  {k:<38} {'nan':>10}")
    print("=" * 62)
    print(f"\n[OK] Rapport → {out_path}\n")


if __name__ == "__main__":
    main()
