"""
Sweep du seuil de consensus — expérience d'évaluation.

Évalue l'impact du consensus_threshold sur les métriques clés en mutant
l'instance chargée sans retraining, pour 4 valeurs : [1.0, 0.75, 0.5, 0.25].

Métriques collectées pour chaque seuil :
    consensus_empty_rate         : fraction de pas où le consensus est vide
    consensus_mean_active        : nombre moyen de bits actifs dans le consensus
    epsilon                      : erreur de reconstruction SDR ∈ [0, 1]
    mean_pairwise_column_overlap : overlap Jaccard moyen entre colonnes
                                   (calculé AVANT consensus — doit être invariant)
    lin_prob                     : précision du sondage linéaire ∈ [0, 1]

Invariant de sanité :
    mean_pairwise_column_overlap NE DOIT PAS changer entre les seuils,
    car il est calculé sur les SDRs individuels des colonnes, avant le vote.
    Toute variation indique un bug dans le pipeline d'évaluation.

Design intent :
    Ce script est une EXPÉRIENCE D'ÉVAL — il ne modifie jamais le code
    source de consensus.py. La mutation se fait uniquement sur l'instance
    modèle chargée. Les invariants I6.2 et I6.3 restent documentés dans
    consensus.py comme invariants de design.

Réf. math : §6.3, Thm 6.3 — Formalisation_Mille_Cerveaux.pdf

Usage :
    python scripts/sweep_consensus_threshold.py
    python scripts/sweep_consensus_threshold.py --checkpoint eval_outputs/cortical_column_phase4_v2.pt
    python scripts/sweep_consensus_threshold.py --n_samples 500 --device cuda
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from column import CorticalColumn
from eval.unsupervised_eval import UnsupervisedEvaluator


# Métriques extraites par le sweep (sous-ensemble de celles de l'évaluateur)
_SWEEP_METRICS = [
    "consensus_empty_rate",
    "consensus_mean_active",
    "epsilon",
    "mean_pairwise_column_overlap",
    "lin_prob",
]

_THRESHOLDS = [1.0, 0.75, 0.5, 0.25]


def _build_model(args, device: torch.device) -> CorticalColumn:
    """Instancie CorticalColumn avec la config de référence."""
    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11, 13, 17][: args.n_grid_modules],
        consensus_threshold=1.0,
    ).to(device)
    return model


def _load_checkpoint(model: CorticalColumn, checkpoint_path: str, device: torch.device) -> None:
    """Charge les poids depuis un checkpoint PyTorch."""
    if not os.path.isfile(checkpoint_path):
        print(f"[AVERTISSEMENT] Checkpoint introuvable : {checkpoint_path}")
        print("[INFO] Évaluation avec poids aléatoires (modèle non entraîné)")
        return
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"[OK] Checkpoint chargé : {checkpoint_path}")


def _load_mnist(n_samples: int, data_dir: str, device: torch.device):
    """Charge n_samples images MNIST (test set) et leurs étiquettes."""
    ds = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True,
        transform=T.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=n_samples, shuffle=False, num_workers=0
    )
    imgs_raw, labels = next(iter(loader))
    imgs = imgs_raw.view(imgs_raw.shape[0], -1).to(device)
    n = min(n_samples, imgs.shape[0])
    return imgs[:n], labels[:n].to(device)


def _evaluate_one_threshold(
    model: CorticalColumn,
    threshold: float,
    inputs: torch.Tensor,
    velocities: torch.Tensor,
    labels: torch.Tensor,
    expected_w: int,
) -> Dict[str, float]:
    """
    Évalue le modèle avec un seuil donné, en mutant l'instance.

    La mutation est réversible : l'appelant restaure le threshold original.
    """
    model.consensus.consensus_threshold = threshold
    evaluator = UnsupervisedEvaluator(model, expected_w=expected_w, n_classes=10)
    metrics = evaluator.evaluate(inputs, velocities, labels)
    return {k: metrics[k] for k in _SWEEP_METRICS}


def _check_overlap_stability(
    results: Dict[float, Dict[str, float]],
    tol: float = 1e-6,
) -> bool:
    """
    Vérifie que mean_pairwise_column_overlap est invariant entre les seuils.

    Retourne True si la valeur est stable (variation < tol), False sinon.
    Les résultats sont calculés AVANT le consensus — toute variation signale
    un bug dans le pipeline d'évaluation (le threshold ne devrait jamais
    affecter les SDRs individuels des colonnes).
    """
    overlaps = [v["mean_pairwise_column_overlap"] for v in results.values()]
    variation = max(overlaps) - min(overlaps)
    return variation < tol, variation


def _print_table(results: Dict[float, Dict[str, float]]) -> None:
    """Affiche un tableau récapitulatif aligné dans le terminal."""
    col_w = 12
    header_w = 32

    # En-tête
    header = f"{'Métrique':<{header_w}}"
    for thr in _THRESHOLDS:
        header += f"{'thr='+str(thr):>{col_w}}"
    print("\n" + "=" * (header_w + col_w * len(_THRESHOLDS)))
    print(header)
    print("-" * (header_w + col_w * len(_THRESHOLDS)))

    for metric in _SWEEP_METRICS:
        row = f"{metric:<{header_w}}"
        for thr in _THRESHOLDS:
            val = results[thr][metric]
            if isinstance(val, float) and not (val != val):  # nan guard
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{'nan':>{col_w}}"
        print(row)

    print("=" * (header_w + col_w * len(_THRESHOLDS)))


def _sanity_check(results: Dict[float, Dict[str, float]]) -> None:
    """Affiche les résultats des vérifications de sanité."""
    stable, variation = _check_overlap_stability(results)
    print("\n[Sanité] mean_pairwise_column_overlap invariant entre thresholds :", end=" ")
    if stable:
        print(f"OK (variation = {variation:.2e})")
    else:
        print(
            f"ECHEC — variation = {variation:.6f}\n"
            "  Ce champ est calculé AVANT le consensus et ne doit pas varier.\n"
            "  Vérifier UnsupervisedEvaluator.evaluate() : la boucle model.reset()\n"
            "  est-elle appelée avant chaque seuil ?"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep du seuil de consensus — expérience d'évaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="eval_outputs/cortical_column_phase4_v2.pt",
        help="Chemin vers le checkpoint CorticalColumn (.pt)",
    )
    parser.add_argument("--n_samples",      type=int, default=200)
    parser.add_argument("--n_columns",      type=int, default=4)
    parser.add_argument("--n_sdr",          type=int, default=2048)
    parser.add_argument("--w",              type=int, default=40)
    parser.add_argument("--n_minicolumns",  type=int, default=256)
    parser.add_argument("--k_active",       type=int, default=40)
    parser.add_argument("--n_grid_modules", type=int, default=6)
    parser.add_argument("--data_dir",       type=str, default="./data")
    parser.add_argument("--output_dir",     type=str, default="./eval_outputs")
    parser.add_argument("--device",         type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = _build_model(args, device)
    _load_checkpoint(model, args.checkpoint, device)

    # ── Données MNIST ─────────────────────────────────────────────────────────
    print(f"\n[Données] Chargement de {args.n_samples} images MNIST (test set)...")
    imgs, labels = _load_mnist(args.n_samples, args.data_dir, device)
    v_zeros = torch.zeros(imgs.shape[0], 2, device=device)
    print(f"[OK] {imgs.shape[0]} images chargées, {labels.unique().numel()} classes")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    original_threshold = model.consensus.consensus_threshold
    results: Dict[float, Dict[str, float]] = {}

    print(f"\n[Sweep] Évaluation sur {len(_THRESHOLDS)} seuils : {_THRESHOLDS}")
    try:
        for thr in _THRESHOLDS:
            print(f"  threshold = {thr:.2f}...", end=" ", flush=True)
            row = _evaluate_one_threshold(
                model, thr, imgs, v_zeros, labels, expected_w=args.w
            )
            results[thr] = row
            print(
                f"ε={row['epsilon']:.4f}  "
                f"empty={row['consensus_empty_rate']:.3f}  "
                f"active={row['consensus_mean_active']:.1f}"
            )
    finally:
        # Restauration garantie même en cas d'exception
        model.consensus.consensus_threshold = original_threshold

    # ── Affichage tableau ─────────────────────────────────────────────────────
    _print_table(results)
    _sanity_check(results)

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    out_path = out_dir / "sweep_consensus_results.json"
    payload = {
        "config": {
            "checkpoint":    args.checkpoint,
            "n_samples":     args.n_samples,
            "n_columns":     args.n_columns,
            "n_sdr":         args.n_sdr,
            "w":             args.w,
            "n_minicolumns": args.n_minicolumns,
            "k_active":      args.k_active,
            "n_grid_modules":args.n_grid_modules,
            "thresholds":    _THRESHOLDS,
        },
        "results": {str(k): v for k, v in results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    main()
