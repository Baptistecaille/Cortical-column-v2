"""
Script d'entraînement non-supervisé — CorticalColumn World Model.

Pipeline (CLAUDE.md §3, Pseudo-algorithme global) :
    Phase 1 — Core (valider les 6 invariants) :
        SDRSpace → SpatialPooler → Layer6bTransformer
        → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

    Ne pas sauter Phase 1 pour aller à Phase 3.

Aligné sur test_multiview.py :
    - MNIST réel (784D) à la place de la séquence synthétique
    - Épisodes multi-vues avec reset() entre images
    - Métriques : cross_view_overlap (Jaccard vue0/vue1) + pred_success_rate
    - tau_decay calibré sur le nombre total de steps
    - Les paramètres par défaut correspondent à ceux de test_multiview.py

Usage :
    python scripts/train.py
    python scripts/train.py --n_images 5000 --n_epochs 3 --n_views 5
    python scripts/train.py --help
"""

import sys
import os
import math
import torch
import argparse
import logging
import time
import subprocess
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset

from column import CorticalColumn
from eval.unsupervised_eval import batch_prediction_success_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_mnist(n_images: int = 5000, data_dir: str = "./data"):
    """Charge MNIST, retourne (images 1×28×28, labels)."""
    ds = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    ds = Subset(ds, range(min(n_images, len(ds))))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
    return torch.cat(imgs), torch.cat(labels)


def generate_episode(
    image: torch.Tensor,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    velocity_scale: float = 10.0,
):
    """Génère n_views vues d'une image avec retour à l'origine."""
    views, velocities = [], []

    views.append(image.squeeze(0).view(-1))
    velocities.append(torch.zeros(2))
    prev_pos = torch.zeros(2)

    for _ in range(n_views - 2):
        dx = torch.empty(1).uniform_(-max_translation, max_translation).item()
        dy = torch.empty(1).uniform_(-max_translation, max_translation).item()
        angle = torch.empty(1).uniform_(-max_rotation, max_rotation).item()
        curr_pos = torch.tensor([dx, dy])
        view = TF.affine(image, angle=angle, translate=[int(dx), int(dy)],
                         scale=1.0, shear=0).squeeze(0).view(-1)
        views.append(view)
        velocities.append((curr_pos - prev_pos) / velocity_scale)
        prev_pos = curr_pos

    # Vue de retour à l'origine
    views.append(image.squeeze(0).view(-1))
    velocities.append((torch.zeros(2) - prev_pos) / velocity_scale)

    return torch.stack(views), torch.stack(velocities)


def train(
    model: CorticalColumn,
    images: torch.Tensor,
    n_epochs: int = 2,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    anchor_confidence: float = 0.8,
    eval_every: int = 500,
    device: str = "cpu",
    lambda_mix: float = 0.5,
    use_surprise_annealing: bool = True,
) -> dict:
    """
    Boucle d'entraînement multi-vues sur MNIST avec annealing surprise-driven.

    Métriques loguées :
        cross_view_overlap : Jaccard SDR_vue0 / SDR_vue1 (même image)
        pred_success_rate  : % de prédictions correctes (W_pred)
        p_conn             : fraction de permanences connectées
        gamma              : γ effectif (mélange schedule + surprise)
        surprise           : ε_t moyen sur l'épisode courant

    Nouveau paramètre — annealing surprise-driven (Rao & Ballard 1999) :
        use_surprise_annealing : active la modulation de γ par la surprise
        lambda_mix             : poids du schedule temporel vs surprise
                                 λ=1.0 → schedule pur (comportement v1)
                                 λ=0.0 → surprise pure
                                 λ=0.5 → équilibre (défaut)

    Pipeline par épisode :
        1. Générer n_views vues de l'image
        2. Pour chaque vue v :
            a. model.step() avec gamma_override calculé depuis la surprise
               du pas PRÉCÉDENT (convention causale : on ne connaît pas
               encore la surprise courante au moment de l'update hebbien)
            b. Récupérer result["surprise"] pour le pas suivant
        3. Anchor grid cells au retour à l'origine

    Returns:
        history dict
    """
    model = model.to(device)
    N = images.shape[0]

    history = {
        "step": [],
        "cross_view_overlap": [],
        "pred_success_rate": [],
        "gamma": [],
        "gamma_surprise_ema": [],
        "surprise": [],
        "p_conn": [],
    }

    _overlap_window: List[float] = []
    _pred_window: List[float] = []
    _surprise_window: List[float] = []
    images_processed = 0
    t0 = time.perf_counter()

    # Surprise du pas précédent (initialisée à 1.0 = surprise maximale).
    # Au démarrage, W_pred est non calibré → on suppose surprise totale.
    # Sera écrasée dès le premier model.step().
    prev_surprise: float = 1.0

    for epoch in range(n_epochs):
        perm = torch.randperm(N)

        for idx_in_epoch, img_idx in enumerate(perm):
            img = images[img_idx]   # (1, 28, 28)

            views, velocities = generate_episode(
                img, n_views=n_views,
                max_translation=max_translation,
                max_rotation=max_rotation,
            )
            views = views.to(device)
            velocities = velocities.to(device)

            # Reset entre épisodes
            model.reset()
            initial_phases = [col.grid_cell.phases.clone() for col in model.columns]

            sdrs_episode = []
            sdrs_predicted_episode = []
            surprises_episode = []

            for v in range(views.shape[0]):
                # ── Calcul du γ effectif depuis la surprise du pas PRÉCÉDENT ──
                # Convention causale : on utilise ε_{t-1} pour modérer l'update
                # hebbien au pas t, car ε_t n'est connu qu'après l'observation.
                # Pendant le newborn stage, gamma_surprise() retourne 1.0
                # inconditionnellement (W_pred non calibré).
                if use_surprise_annealing:
                    sp = model.columns[0].spatial_pooler
                    gamma_eff = sp.gamma_surprise(
                        surprise=prev_surprise,
                        lambda_mix=lambda_mix,
                    )
                else:
                    gamma_eff = None  # schedule temporel pur

                result = model.step(
                    views[v], velocities[v],
                    train=True,
                    gamma_override=gamma_eff,
                )
                sdrs_episode.append(result["sdr"].cpu())
                sdrs_predicted_episode.append(result["sdr_predicted"].cpu())
                surprises_episode.append(result["surprise"])

                # Mémorisation de la surprise courante pour le pas suivant
                prev_surprise = result["surprise"]

            # Anchor au retour à l'origine
            if anchor_confidence > 0:
                for c, col in enumerate(model.columns):
                    col.grid_cell.anchor(initial_phases[c], confidence=anchor_confidence)

            images_processed += 1

            # ── Métriques périodiques ─────────────────────────────────────
            if images_processed % eval_every == 0:
                # cross_view_overlap : Jaccard vue0 vs vue1
                if len(sdrs_episode) >= 2:
                    s0 = sdrs_episode[0].float()
                    s1 = sdrs_episode[1].float()
                    inter = (s0 * s1).sum().item()
                    union = ((s0 + s1) > 0).float().sum().item()
                    overlap = inter / union if union > 0 else 0.0
                else:
                    overlap = 0.0

                _overlap_window.append(overlap)
                if len(_overlap_window) > 100:
                    _overlap_window.pop(0)
                overlap_smooth = sum(_overlap_window) / len(_overlap_window)

                pred_stats = batch_prediction_success_rate(
                    sdrs_predicted_episode, sdrs_episode
                )
                _pred_window.append(pred_stats["pred_success_rate"])
                if len(_pred_window) > 100:
                    _pred_window.pop(0)
                pred_smooth = sum(_pred_window) / len(_pred_window)

                # Surprise moyenne sur l'épisode (fenêtre glissante 100 épisodes)
                ep_surprise = sum(surprises_episode) / max(len(surprises_episode), 1)
                _surprise_window.append(ep_surprise)
                if len(_surprise_window) > 100:
                    _surprise_window.pop(0)
                surprise_smooth = sum(_surprise_window) / len(_surprise_window)

                sp = model.columns[0].spatial_pooler
                gamma_temporal = sp.gamma()
                gamma_eff_log = sp.gamma_surprise(prev_surprise, lambda_mix) \
                    if use_surprise_annealing else gamma_temporal
                p_stats = sp.permanence_stats()
                elapsed = time.perf_counter() - t0
                ips = images_processed / elapsed

                history["step"].append(images_processed)
                history["cross_view_overlap"].append(overlap_smooth)
                history["pred_success_rate"].append(pred_smooth)
                history["gamma"].append(gamma_eff_log)
                history["gamma_surprise_ema"].append(gamma_temporal)
                history["surprise"].append(surprise_smooth)
                history["p_conn"].append(p_stats["frac_connected"])

                logger.info(
                    f"Epoch {epoch+1} | {idx_in_epoch+1:5d}/{N} | "
                    f"{ips:4.0f} img/s | "
                    f"γ_eff={gamma_eff_log:.3f} | "
                    f"γ_sched={gamma_temporal:.3f} | "
                    f"ε={surprise_smooth:.3f} | "
                    f"p_conn={p_stats['frac_connected']:.2f} | "
                    f"p̄={p_stats['mean']:.3f} | "
                    f"overlap={overlap_smooth:.3f} | "
                    f"pred={pred_smooth*100:.1f}%"
                )

    logger.info("Entraînement terminé.")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement non-supervisé du CorticalColumn World Model"
    )
    parser.add_argument("--n_images",       type=int,   default=3000)
    parser.add_argument("--n_epochs",       type=int,   default=2)
    parser.add_argument("--n_views",        type=int,   default=5)
    parser.add_argument("--n_columns",      type=int,   default=4)
    parser.add_argument("--n_sdr",          type=int,   default=2048)
    parser.add_argument("--w",              type=int,   default=40)
    parser.add_argument("--n_minicolumns",  type=int,   default=256)
    parser.add_argument("--k_active",       type=int,   default=40)
    parser.add_argument("--n_grid_modules", type=int,   default=6)
    parser.add_argument("--max_translation",type=float, default=4.0)
    parser.add_argument("--max_rotation",   type=float, default=15.0)
    parser.add_argument("--anchor_conf",    type=float, default=0.8)
    parser.add_argument("--eval_every",     type=int,   default=500)
    parser.add_argument("--device",         type=str,   default="cpu")
    parser.add_argument("--data_dir",       type=str,   default="./data")
    parser.add_argument("--verbose", "-v",  action="store_true")
    parser.add_argument(
        "--lambda_mix", type=float, default=0.5,
        help="Poids du schedule temporel vs surprise dans γ_effectif. "
             "0.0=surprise pure, 1.0=schedule pur, 0.5=équilibre (défaut)."
    )
    parser.add_argument(
        "--no_surprise_annealing", action="store_true",
        help="Désactive l'annealing surprise-driven — revient au schedule pur."
    )
    args = parser.parse_args()

    # tau_decay calibré sur le total de steps (comme test_multiview.py)
    newborn_steps = 1000
    total_steps = args.n_epochs * args.n_images * args.n_views
    tau_decay = max(1, total_steps - newborn_steps)

    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11, 13, 17][:args.n_grid_modules],
        consensus_threshold=1.0,
        sp_kwargs={
            "newborn_steps":      newborn_steps,
            "tau_decay":          tau_decay,
            # Équilibre δ+/δ- recalibré pour n_sdr=2048, w=40, potential_pct=0.5 :
            # n_pool = 0.5 × 2048 = 1024 bits par colonne
            # bits actifs attendus dans le pool ≈ 40 × 1024/2048 = 20
            # ratio d'équilibre = (1024 - 20) / 20 ≈ 50
            # → δ+ / δ- ≥ 50 pour que p̄ reste stable
            # δ+ = 0.015, δ- = 0.015/50 = 0.0003
            "delta_plus":         0.015,
            "delta_minus":        0.0003,
            "delta_minus_floor":  0.00003,
        },
    )

    logger.info(f"Modèle : {args.n_columns} colonnes × {args.n_minicolumns} minicolonnes")
    logger.info(f"SDR : n={args.n_sdr}, w={args.w} ({100*args.w/args.n_sdr:.1f}%)")
    logger.info(f"γ schedule : 1.0 → 0.2 sur {total_steps} steps (τ={tau_decay})")

    # Vérification des invariants
    logger.info("Lancement de la vérification des invariants...")
    invariants_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_invariants.py",
    )
    subprocess.run([sys.executable, invariants_script], check=True)

    # Chargement MNIST
    logger.info(f"Chargement MNIST ({args.n_images} images)...")
    images, labels = load_mnist(args.n_images, args.data_dir)

    use_surprise = not args.no_surprise_annealing
    logger.info(
        f"Annealing surprise-driven : {'activé' if use_surprise else 'désactivé'} | "
        f"λ_mix={args.lambda_mix:.2f}"
    )

    # Entraînement
    history = train(
        model=model,
        images=images,
        n_epochs=args.n_epochs,
        n_views=args.n_views,
        max_translation=args.max_translation,
        max_rotation=args.max_rotation,
        anchor_confidence=args.anchor_conf,
        eval_every=args.eval_every,
        device=args.device,
        lambda_mix=args.lambda_mix,
        use_surprise_annealing=use_surprise,
    )

    if history["cross_view_overlap"]:
        logger.info(f"Overlap final  = {history['cross_view_overlap'][-1]:.3f}")
        logger.info(f"Pred final     = {history['pred_success_rate'][-1]*100:.1f}%")


if __name__ == "__main__":
    main()
