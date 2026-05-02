"""
Script d'entraînement non-supervisé — CorticalColumn World Model.

Optimisations CUDA / parallélisation :
    - Batch B épisodes indépendants en parallèle via step_parallel()
      (chaque colonne traite B images simultanément avec ops GPU batched)
    - DataLoader pin_memory + num_workers pour transfert CPU→GPU asynchrone
    - torch.backends.cudnn.benchmark pour auto-tuning des kernels CUDA
    - torch.compile() optionnel (PyTorch 2.0+) — JIT TorchInductor
    - Génération d'épisodes vectorisée (torchvision batché)
    - Sauvegarde checkpoint --save_path

Pipeline (CLAUDE.md §3) :
    Phase 1 — Core (valider les 6 invariants) :
        SDRSpace → SpatialPooler → Layer6bTransformer
        → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

    Ne pas sauter Phase 1 pour aller à Phase 3.

Usage :
    python scripts/train.py
    python scripts/train.py --device cuda --batch_size 32 --n_images 5000
    python scripts/train.py --compile --save_path model.pt
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
from typing import List, Optional, Tuple

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


# ─── Chargement données ───────────────────────────────────────────────────────

def load_mnist(n_images: int = 5000, data_dir: str = "./data") -> Tuple[torch.Tensor, torch.Tensor]:
    """Charge MNIST, retourne (images (N,1,28,28), labels (N,))."""
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


# ─── Génération d'épisodes ────────────────────────────────────────────────────

def generate_episode(
    image: torch.Tensor,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    velocity_scale: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Génère n_views vues avec retour à l'origine (une image).

    Returns:
        views:      (n_views, 784) float
        velocities: (n_views, 2)  float
    """
    views, velocities = [], []
    views.append(image.squeeze(0).view(-1))
    velocities.append(torch.zeros(2))
    prev_pos = torch.zeros(2)

    for _ in range(n_views - 2):
        dx    = torch.empty(1).uniform_(-max_translation, max_translation).item()
        dy    = torch.empty(1).uniform_(-max_translation, max_translation).item()
        angle = torch.empty(1).uniform_(-max_rotation, max_rotation).item()
        curr_pos = torch.tensor([dx, dy])
        view = TF.affine(image, angle=angle, translate=[int(dx), int(dy)],
                         scale=1.0, shear=0).squeeze(0).view(-1)
        views.append(view)
        velocities.append((curr_pos - prev_pos) / velocity_scale)
        prev_pos = curr_pos

    views.append(image.squeeze(0).view(-1))
    velocities.append((torch.zeros(2) - prev_pos) / velocity_scale)

    return torch.stack(views), torch.stack(velocities)


def generate_episodes_batch(
    images_batch: torch.Tensor,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    velocity_scale: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Génère un lot de B épisodes en parallèle (boucle Python légère sur B).

    Args:
        images_batch: (B, 1, 28, 28) — batch d'images MNIST

    Returns:
        views_batch:  (B, n_views, 784)
        vel_batch:    (B, n_views, 2)
    """
    all_views, all_vels = [], []
    for b in range(images_batch.shape[0]):
        v, vel = generate_episode(
            images_batch[b], n_views, max_translation, max_rotation, velocity_scale
        )
        all_views.append(v)
        all_vels.append(vel)
    return torch.stack(all_views), torch.stack(all_vels)


# ─── Boucle d'entraînement batché ────────────────────────────────────────────

def train(
    model: CorticalColumn,
    images: torch.Tensor,          # (N, 1, 28, 28)
    n_epochs: int = 2,
    batch_size: int = 1,           # B épisodes en parallèle
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    anchor_confidence: float = 0.8,
    eval_every: int = 500,
    device: str = "cpu",
    lambda_mix: float = 0.5,
    use_surprise_annealing: bool = True,
    save_path: Optional[str] = None,
) -> dict:
    """
    Boucle d'entraînement multi-vues batché.

    Avec batch_size > 1, utilise step_parallel() qui vectorise les B images
    sur GPU simultanément. Avec batch_size=1, retombe sur le mode séquentiel
    (comportement identique à l'ancienne version).

    Métriques loguées :
        cross_view_overlap  : Jaccard SDR vue0/vue1
        pred_success_rate   : % de prédictions correctes (W_pred)
        p_conn              : fraction de permanences connectées
        gamma               : γ effectif
        surprise            : ε_t moyen sur le lot

    Returns:
        history dict
    """
    dev = torch.device(device)
    model = model.to(dev)
    N = images.shape[0]

    # ── Optimisations CUDA ──────────────────────────────────────────────────
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    history = {
        "step": [], "cross_view_overlap": [], "pred_success_rate": [],
        "gamma": [], "gamma_surprise_ema": [], "surprise": [], "p_conn": [],
        "mean_pairwise_column_overlap": [],
        # Diagnostics CMP proximaux (A/B test enable_vote)
        "cmp_jaccard_active_vs_vote": [],
        "cmp_pressure": [],
        "cmp_vote_stability": [],
    }

    _overlap_window:       List[float] = []
    _pred_window:          List[float] = []
    _surprise_window:      List[float] = []
    _pairwise_window:      List[float] = []
    _cmp_jaccard_window:   List[float] = []
    _cmp_pressure_window:  List[float] = []
    _cmp_stability_window: List[float] = []
    last_all_sdrs:         List[torch.Tensor] = []
    images_processed = 0
    t0 = time.perf_counter()
    prev_surprise: float = 1.0

    # DataLoader CPU-side avec pin_memory pour transfert asynchrone
    use_pin = (dev.type == "cuda")
    ds_full = torch.utils.data.TensorDataset(images)
    n_workers = min(4, os.cpu_count() or 1) if dev.type == "cuda" else 0
    loader = DataLoader(
        ds_full,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=use_pin,
        drop_last=True,
        prefetch_factor=2 if n_workers > 0 else None,
        persistent_workers=(n_workers > 0),
    )

    for epoch in range(n_epochs):
        for (batch_imgs,) in loader:
            B = batch_imgs.shape[0]   # peut être < batch_size au dernier batch

            # Génération des épisodes (CPU) + transfert GPU non-bloquant
            views_batch, vel_batch = generate_episodes_batch(
                batch_imgs, n_views, max_translation, max_rotation
            )
            views_batch = views_batch.to(dev, non_blocking=use_pin)  # (B, n_views, 784)
            vel_batch   = vel_batch.to(dev, non_blocking=use_pin)    # (B, n_views, 2)

            # ── Initialisation des états (B épisodes) ──────────────────────
            state_batch = model.make_batch_state(B, dev)
            # Mémoriser les phases initiales pour l'ancrage de retour
            initial_phases = [state_batch[c]["phases"].clone() for c in range(model.n_columns)]

            sdrs_episode:           List[torch.Tensor] = []
            sdrs_predicted_episode: List[torch.Tensor] = []
            surprises_episode:      List[float]        = []
            # Métriques CMP par vue — collectées uniquement en mode B=1
            # (step_parallel ne calcule pas ces diagnostics)
            cmp_jaccard_episode:   List[float] = []
            cmp_pressure_episode:  List[float] = []
            cmp_stability_episode: List[float] = []

            # ── Boucle sur les vues ─────────────────────────────────────────
            for v in range(n_views):
                # Calcul du γ effectif depuis la surprise du pas PRÉCÉDENT
                if use_surprise_annealing:
                    sp = model.columns[0].spatial_pooler
                    gamma_eff = sp.gamma_surprise(prev_surprise, lambda_mix)
                else:
                    gamma_eff = None

                if B == 1:
                    # Mode scalaire (B=1) — utilise l'interface step() standard
                    # pour éviter la surcouche batch quand elle n'est pas utile
                    model_result = model.step(
                        views_batch[0, v], vel_batch[0, v],
                        train=True, gamma_override=gamma_eff,
                    )
                    # Ré-encapsuler pour cohérence de l'interface
                    results = {
                        "sdr":           model_result["sdr"].unsqueeze(0),
                        "sdr_predicted": model_result["sdr_predicted"].unsqueeze(0),
                        "surprise":      torch.tensor([model_result["surprise"]], device=dev),
                        "all_sdrs":      model_result["all_sdrs"],   # liste K × (n_sdr,)
                    }
                    # Collecte des diagnostics CMP (disponibles uniquement via step())
                    jac = model_result.get("cmp_jaccard_active_vs_vote", float("nan"))
                    prs = model_result.get("cmp_pressure", float("nan"))
                    sta = model_result.get("cmp_vote_stability", float("nan"))
                    if not math.isnan(jac):
                        cmp_jaccard_episode.append(jac)
                    if not math.isnan(prs):
                        cmp_pressure_episode.append(prs)
                    if not math.isnan(sta):
                        cmp_stability_episode.append(sta)
                    # Mettre à jour l'état (déjà géré en interne par step())
                else:
                    results, state_batch = model.step_parallel(
                        views_batch[:, v, :], vel_batch[:, v, :],
                        state_batch, train=True, gamma_override=gamma_eff,
                    )

                last_all_sdrs = results["all_sdrs"]   # mis à jour à chaque vue

                sdrs_episode.append(results["sdr"].cpu())
                sdrs_predicted_episode.append(results["sdr_predicted"].cpu())
                ep_surprise = float(results["surprise"].mean().item())
                surprises_episode.append(ep_surprise)
                prev_surprise = ep_surprise

            # ── Ancrage au retour à l'origine ──────────────────────────────
            if anchor_confidence > 0:
                if B == 1:
                    for c, col in enumerate(model.columns):
                        col.grid_cell.anchor(
                            initial_phases[c][0], confidence=anchor_confidence
                        )
                else:
                    model.anchor_batch(state_batch, initial_phases, anchor_confidence)

            images_processed += B

            # ── Métriques périodiques ───────────────────────────────────────
            if images_processed % eval_every < B:
                # cross_view_overlap : Jaccard vue0 vs vue1 (moyenné sur B)
                if len(sdrs_episode) >= 2:
                    s0 = sdrs_episode[0].float()    # (B, n_sdr)
                    s1 = sdrs_episode[1].float()    # (B, n_sdr)
                    inter  = (s0 * s1).sum(dim=-1)                            # (B,)
                    union  = ((s0 + s1) > 0).float().sum(dim=-1).clamp(min=1) # (B,)
                    overlap = float((inter / union).mean().item())
                else:
                    overlap = 0.0

                _overlap_window.append(overlap)
                if len(_overlap_window) > 100: _overlap_window.pop(0)
                overlap_smooth = sum(_overlap_window) / len(_overlap_window)

                # pred_success_rate (sur le premier échantillon du lot)
                preds_flat = [p[0] for p in sdrs_predicted_episode]
                obs_flat   = [s[0] for s in sdrs_episode]
                pred_stats = batch_prediction_success_rate(preds_flat, obs_flat)
                _pred_window.append(pred_stats["pred_success_rate"])
                if len(_pred_window) > 100: _pred_window.pop(0)
                pred_smooth = sum(_pred_window) / len(_pred_window)

                ep_surp = sum(surprises_episode) / max(len(surprises_episode), 1)
                _surprise_window.append(ep_surp)
                if len(_surprise_window) > 100: _surprise_window.pop(0)
                surprise_smooth = sum(_surprise_window) / len(_surprise_window)

                sp = model.columns[0].spatial_pooler
                gamma_temporal = sp.gamma()
                gamma_eff_log  = (
                    sp.gamma_surprise(prev_surprise, lambda_mix)
                    if use_surprise_annealing else gamma_temporal
                )
                p_stats = sp.permanence_stats()
                elapsed = time.perf_counter() - t0
                ips = images_processed / elapsed

                # mean_pairwise_column_overlap : Jaccard entre toutes les paires
                # de colonnes, calculé sur last_all_sdrs (dernière vue du lot).
                # Pour B==1 : liste K × (n_sdr,) ; pour B>1 : liste K × (B, n_sdr).
                # Invariant : ne doit pas dépendre du consensus_threshold (voir sweep).
                if last_all_sdrs and len(last_all_sdrs) >= 2:
                    K_c = len(last_all_sdrs)
                    pair_overlaps = []
                    for ci in range(K_c):
                        for cj in range(ci + 1, K_c):
                            a = last_all_sdrs[ci].float()
                            b = last_all_sdrs[cj].float()
                            if a.dim() == 1:
                                a, b = a.unsqueeze(0), b.unsqueeze(0)
                            inter = (a * b).sum(dim=-1)
                            union = ((a + b) > 0).float().sum(dim=-1).clamp(min=1.0)
                            pair_overlaps.append(float((inter / union).mean().item()))
                    col_overlap = sum(pair_overlaps) / len(pair_overlaps)
                else:
                    col_overlap = 0.0

                _pairwise_window.append(col_overlap)
                if len(_pairwise_window) > 100: _pairwise_window.pop(0)
                col_overlap_smooth = sum(_pairwise_window) / len(_pairwise_window)

                # ── Métriques CMP proximales ──────────────────────────────
                # Agrégation de l'épisode courant dans les fenêtres glissantes
                if cmp_jaccard_episode:
                    _cmp_jaccard_window.append(
                        sum(cmp_jaccard_episode) / len(cmp_jaccard_episode)
                    )
                    if len(_cmp_jaccard_window) > 50:
                        _cmp_jaccard_window.pop(0)
                if cmp_pressure_episode:
                    _cmp_pressure_window.append(
                        sum(cmp_pressure_episode) / len(cmp_pressure_episode)
                    )
                    if len(_cmp_pressure_window) > 50:
                        _cmp_pressure_window.pop(0)
                if cmp_stability_episode:
                    _cmp_stability_window.append(
                        sum(cmp_stability_episode) / len(cmp_stability_episode)
                    )
                    if len(_cmp_stability_window) > 50:
                        _cmp_stability_window.pop(0)

                cmp_jac_smooth = (
                    sum(_cmp_jaccard_window) / len(_cmp_jaccard_window)
                    if _cmp_jaccard_window else float("nan")
                )
                cmp_prs_smooth = (
                    sum(_cmp_pressure_window) / len(_cmp_pressure_window)
                    if _cmp_pressure_window else float("nan")
                )
                cmp_sta_smooth = (
                    sum(_cmp_stability_window) / len(_cmp_stability_window)
                    if _cmp_stability_window else float("nan")
                )

                history["step"].append(images_processed)
                history["cross_view_overlap"].append(overlap_smooth)
                history["pred_success_rate"].append(pred_smooth)
                history["gamma"].append(gamma_eff_log)
                history["gamma_surprise_ema"].append(gamma_temporal)
                history["surprise"].append(surprise_smooth)
                history["p_conn"].append(p_stats["frac_connected"])
                history["mean_pairwise_column_overlap"].append(col_overlap_smooth)
                history["cmp_jaccard_active_vs_vote"].append(cmp_jac_smooth)
                history["cmp_pressure"].append(cmp_prs_smooth)
                history["cmp_vote_stability"].append(cmp_sta_smooth)

                cmp_log = (
                    f" | jac={cmp_jac_smooth:.3f}"
                    f" | prs={cmp_prs_smooth:.3f}"
                    f" | stab={cmp_sta_smooth:.3f}"
                    if not math.isnan(cmp_jac_smooth) else ""
                )
                logger.info(
                    f"Epoch {epoch+1} | {images_processed:6d}/{N*n_epochs} | "
                    f"{ips:5.0f} img/s | B={B} | "
                    f"γ={gamma_eff_log:.3f} | "
                    f"ε={surprise_smooth:.3f} | "
                    f"p_conn={p_stats['frac_connected']:.2f} | "
                    f"overlap={overlap_smooth:.3f} | "
                    f"col_overlap={col_overlap_smooth:.4f} | "
                    f"pred={pred_smooth*100:.1f}%"
                    f"{cmp_log}"
                )

    logger.info("Entraînement terminé.")

    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Modèle sauvegardé → {save_path}")

    return history


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement non-supervisé du CorticalColumn World Model"
    )
    parser.add_argument("--n_images",        type=int,   default=3000)
    parser.add_argument("--n_epochs",        type=int,   default=2)
    parser.add_argument("--n_views",         type=int,   default=5)
    parser.add_argument("--n_columns",       type=int,   default=4)
    parser.add_argument("--n_sdr",           type=int,   default=2048)
    parser.add_argument("--w",               type=int,   default=40)
    parser.add_argument("--n_minicolumns",   type=int,   default=256)
    parser.add_argument("--k_active",        type=int,   default=40)
    parser.add_argument("--n_grid_modules",  type=int,   default=6)
    parser.add_argument("--grid_periods",    type=str,   default=None,
                        help="Périodes λ_k séparées par virgules ex: 3,5,7,11,13,17")
    parser.add_argument("--max_translation", type=float, default=4.0)
    parser.add_argument("--max_rotation",    type=float, default=15.0)
    parser.add_argument("--anchor_conf",     type=float, default=0.8)
    parser.add_argument("--eval_every",      type=int,   default=500)
    parser.add_argument("--device",          type=str,   default="cpu")
    parser.add_argument("--data_dir",        type=str,   default="./data")
    parser.add_argument("--verbose", "-v",   action="store_true")

    # ── Parallélisation / CUDA ─────────────────────────────────────────────
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help=(
            "Nombre d'épisodes traités en parallèle par step_parallel(). "
            "1 = mode séquentiel (comportement original). "
            "32–64 recommandé sur GPU CUDA pour saturer les ALUs."
        ),
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile le modèle avec torch.compile() (PyTorch 2.0+ requis). "
             "Gain ~20–50%% sur CUDA après la phase de warm-up.",
    )

    # ── Annealing ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--lambda_mix", type=float, default=0.5,
        help="Poids schedule temporel vs surprise dans γ_effectif. "
             "0.0=surprise pure, 1.0=schedule pur, 0.5=équilibre (défaut).",
    )
    parser.add_argument(
        "--no_surprise_annealing", action="store_true",
        help="Désactive l'annealing surprise-driven — revient au schedule pur.",
    )

    # ── Vote inter-colonnes (Phase 2 CMP, Clay-Leadholm 2024) ─────────────
    parser.add_argument(
        "--enable_vote", action="store_true",
        help=(
            "Active le vote inter-colonnes hebbian_update_targeted (Phase 2 CMP). "
            "Désactivé par défaut — à activer pour forcer la convergence inter-colonnes "
            "et observer la montée de mean_pairwise_column_overlap dans les logs."
        ),
    )
    parser.add_argument(
        "--alpha_divergence", type=float, default=3.0,
        help=(
            "Pénalité de dépression sur les bits divergents du vote inter-colonnes. "
            "Ignoré si --enable_vote est absent. Défaut : 3.0."
        ),
    )

    # ── Sauvegarde ────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="Chemin de sauvegarde du modèle après entraînement (ex: model.pt).",
    )

    args = parser.parse_args()

    # tau_decay calibré sur le total de steps (comme test_multiview.py)
    newborn_steps = 1000
    total_steps   = args.n_epochs * args.n_images * args.n_views
    tau_decay     = max(1, total_steps - newborn_steps)

    grid_periods_list = (
        [int(p) for p in args.grid_periods.split(",")]
        if args.grid_periods
        else [3, 5, 7, 11, 13, 17][:args.n_grid_modules]
    )
    if len(grid_periods_list) != args.n_grid_modules:
        parser.error(
            f"--grid_periods a {len(grid_periods_list)} éléments mais --n_grid_modules={args.n_grid_modules}"
        )

    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=grid_periods_list,
        consensus_threshold=1.0,
        enable_vote=args.enable_vote,
        alpha_divergence=args.alpha_divergence,
        sp_kwargs={
            "newborn_steps":     newborn_steps,
            "tau_decay":         tau_decay,
            "delta_plus":        0.015,
            "delta_minus":       0.0003,
            "delta_minus_floor": 0.00003,
        },
    )

    # ── torch.compile (PyTorch 2.0+) ────────────────────────────────────────
    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile() activé — warm-up au premier batch")
        except Exception as e:
            logger.warning(f"torch.compile() indisponible ({e}) — mode eager")

    logger.info(f"Modèle : {args.n_columns} colonnes × {args.n_minicolumns} minicolonnes")
    logger.info(f"SDR : n={args.n_sdr}, w={args.w} ({100*args.w/args.n_sdr:.1f}%)")
    logger.info(f"γ schedule : 1.0 → 0.2 sur {total_steps} steps (τ={tau_decay})")
    logger.info(f"Batch size : {args.batch_size} épisodes parallèles | device : {args.device}")

    # Vérification des invariants
    logger.info("Lancement de la vérification des invariants...")
    invariants_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_invariants.py"
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

    history = train(
        model=model,
        images=images,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_views=args.n_views,
        max_translation=args.max_translation,
        max_rotation=args.max_rotation,
        anchor_confidence=args.anchor_conf,
        eval_every=args.eval_every,
        device=args.device,
        lambda_mix=args.lambda_mix,
        use_surprise_annealing=use_surprise,
        save_path=args.save_path,
    )

    if history["cross_view_overlap"]:
        logger.info(f"Overlap final  = {history['cross_view_overlap'][-1]:.3f}")
        logger.info(f"Pred final     = {history['pred_success_rate'][-1]*100:.1f}%")


if __name__ == "__main__":
    main()
