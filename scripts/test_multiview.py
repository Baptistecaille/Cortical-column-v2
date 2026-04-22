"""
Test Option B — Séquences multi-vues avec intégration de chemin.

Motivation :
    Avec des images statiques (v=0), le consensus AND reste vide parce que
    les K colonnes restent dans des partitions indépendantes de l'espace SDR.
    En présentant chaque image comme une SÉQUENCE de vues (translations/rotations)
    avec des vecteurs de vitesse non-nuls, les grid cells intègrent le chemin
    et anchor() re-synchronise les phases au retour à l'origine.
    Les SpatialPoolers des K colonnes, soumis aux mêmes trajectoires,
    commencent à partager des colonnes actives → ε descend.

Structure d'un épisode :
    Vue 0 (originale, v=0)   → phase_0 établie, ancre mémorisée
    Vue 1 (translation δ₁)   → v₁ = δ₁,      phase intégrée
    Vue 2 (translation δ₂)   → v₂ = δ₂ − δ₁, phase intégrée
    ...
    Vue N (retour à 0)        → vN = −δN₋₁,   phase ≈ phase_0 → anchor()

Métriques spécifiques multi-vue :
    cross_view_overlap : Jaccard moyen entre SDRs de la même image vues différemment
    grid_consistency   : cohérence des grid codes à la même position physique
    anchor_error       : écart angulaire moyen après anchor() vs phase initiale
    epsilon            : erreur de consensus (doit descendre vs mode statique)

Usage :
    python scripts/test_multiview.py
    python scripts/test_multiview.py --n_images 2000 --n_views 5 --n_epochs 3
"""

import sys
import os
import math
import torch
import argparse
import logging
import time
from typing import List, Tuple, NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset

from column import CorticalColumn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Épisode multi-vue ─────────────────────────────────────────────────────────

class Episode(NamedTuple):
    """Séquence de vues d'une même image avec vecteurs de vitesse associés."""
    views:      torch.Tensor   # (N_views, 784) — vues aplaties
    velocities: torch.Tensor   # (N_views, 2)   — vitesse pour atteindre la vue i
    positions:  torch.Tensor   # (N_views, 2)   — position physique (pixels)
    label:      int


def generate_episode(
    image: torch.Tensor,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    velocity_scale: float = 10.0,
    return_to_origin: bool = True,
) -> Episode:
    """
    Génère une séquence de vues d'une image avec trajectoire spatiale.

    La vue 0 est l'image originale (position 0,0). Les vues suivantes
    appliquent des translations et rotations aléatoires. Si return_to_origin=True,
    la dernière vue revient à (0,0) pour déclencher l'anchor().

    Args:
        image:            tenseur (1, 28, 28) de l'image originale
        n_views:          nombre total de vues (incluant la vue de retour)
        max_translation:  translation maximale en pixels
        max_rotation:     rotation maximale en degrés
        velocity_scale:   facteur de normalisation des vitesses
        return_to_origin: si True, la dernière vue revient à l'origine

    Returns:
        Episode avec views (N, 784), velocities (N, 2), positions (N, 2)
    """
    views = []
    velocities = []
    positions = []

    # Vue 0 : image originale, v=0, position=(0,0)
    views.append(image.squeeze(0).view(-1))   # (784,)
    velocities.append(torch.zeros(2))
    positions.append(torch.zeros(2))

    prev_pos = torch.zeros(2)

    # Vues intermédiaires (1 à n_views-2 ou n_views-1)
    n_intermediate = (n_views - 2) if return_to_origin else (n_views - 1)

    for _ in range(n_intermediate):
        # Nouvelle position aléatoire
        dx = torch.empty(1).uniform_(-max_translation, max_translation).item()
        dy = torch.empty(1).uniform_(-max_translation, max_translation).item()
        angle = torch.empty(1).uniform_(-max_rotation, max_rotation).item()
        curr_pos = torch.tensor([dx, dy])

        # Transformation affine de l'image
        view = TF.affine(
            image,
            angle=angle,
            translate=[int(dx), int(dy)],
            scale=1.0,
            shear=0,
        ).squeeze(0).view(-1)   # (784,)
        views.append(view)

        # Vitesse = déplacement depuis la vue précédente, normalisé
        vel = (curr_pos - prev_pos) / velocity_scale
        velocities.append(vel)
        positions.append(curr_pos)
        prev_pos = curr_pos

    # Vue de retour à l'origine (déclenche anchor)
    if return_to_origin and n_views > 1:
        views.append(image.squeeze(0).view(-1))   # même image qu'au début
        vel_return = (torch.zeros(2) - prev_pos) / velocity_scale
        velocities.append(vel_return)
        positions.append(torch.zeros(2))

    return Episode(
        views=torch.stack(views, dim=0),           # (N, 784)
        velocities=torch.stack(velocities, dim=0), # (N, 2)
        positions=torch.stack(positions, dim=0),   # (N, 2)
        label=-1,
    )


# ── Chargement et préparation ─────────────────────────────────────────────────

def load_mnist_raw(
    n_images: int = 5000,
    data_dir: str = "./data",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Charge MNIST et retourne (images 1×28×28, labels)."""
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


# ── Entraînement séquentiel multi-vue ────────────────────────────────────────

def train_multiview(
    model: CorticalColumn,
    images: torch.Tensor,
    device: torch.device,
    n_epochs: int = 1,
    n_views: int = 5,
    max_translation: float = 4.0,
    max_rotation: float = 15.0,
    anchor_confidence: float = 0.8,
    log_every: int = 200,
) -> dict:
    """
    Entraînement hebbien sur séquences multi-vues.

    Pour chaque image :
        1. Générer un épisode (n_views vues avec vitesses)
        2. Présenter les vues séquentiellement via step()
        3. Après la vue de retour, déclencher anchor() sur toutes les colonnes
        4. Mesurer la cohérence des SDRs à travers les vues

    Args:
        model:              CorticalColumn
        images:             tenseur (N, 1, 28, 28)
        device:             device cible
        n_epochs:           passes sur le dataset
        n_views:            nombre de vues par épisode
        max_translation:    amplitude des translations (pixels)
        max_rotation:       amplitude des rotations (degrés)
        anchor_confidence:  confiance de l'ancrage (0=aucune, 1=totale)
        log_every:          fréquence de log (en images)

    Returns:
        dict avec historique des métriques d'entraînement
    """
    N = images.shape[0]
    model.to(device)

    history = {
        "cross_view_overlap": [],
        "grid_consistency": [],
        "anchor_error_deg": [],
        "step": [],
    }

    t0 = time.perf_counter()
    images_processed = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(N)

        for idx_in_epoch, img_idx in enumerate(perm):
            img = images[img_idx]   # (1, 28, 28)

            # Génération de l'épisode
            episode = generate_episode(
                img,
                n_views=n_views,
                max_translation=max_translation,
                max_rotation=max_rotation,
                return_to_origin=True,
            )

            # Réinitialisation de l'état temporel pour chaque épisode
            model.reset()

            # Phase initiale (mémorisée pour l'anchor et la mesure de cohérence)
            initial_phases = [
                col.grid_cell.phases.clone()
                for col in model.columns
            ]

            sdrs_in_episode = []
            phases_at_origin = []   # phases quand position ≈ (0,0)

            # ── Présentation séquentielle des vues ───────────────────────
            for view_i in range(episode.views.shape[0]):
                s_t = episode.views[view_i].to(device)
                v_t = episode.velocities[view_i].to(device)

                result = model.step(s_t, v_t, train=True)
                sdrs_in_episode.append(result["sdr"].cpu())

                # Enregistrer les phases à la vue de retour (position ≈ 0)
                is_return = (view_i == episode.views.shape[0] - 1)
                if is_return or view_i == 0:
                    phases_at_origin.append(
                        model.columns[0].grid_cell.phases.clone().cpu()
                    )

                # ── Anchor au retour à l'origine ─────────────────────────
                if is_return and anchor_confidence > 0:
                    for c, col in enumerate(model.columns):
                        col.grid_cell.anchor(
                            initial_phases[c].to(device),
                            confidence=anchor_confidence,
                        )

            images_processed += 1

            # ── Métriques (calculées périodiquement) ─────────────────────
            if (idx_in_epoch + 1) % log_every == 0:
                # Overlap inter-vues (Jaccard entre vue 0 et vue 1)
                if len(sdrs_in_episode) >= 2:
                    s0, s1 = sdrs_in_episode[0].float(), sdrs_in_episode[1].float()
                    inter = (s0 * s1).sum().item()
                    union = ((s0 + s1) > 0).float().sum().item()
                    overlap = inter / union if union > 0 else 0.0
                else:
                    overlap = 0.0

                # Cohérence des grid codes entre vue 0 et vue retour
                if len(phases_at_origin) >= 2:
                    p0 = phases_at_origin[0]
                    p_ret = phases_at_origin[-1]
                    err_rad = ((p_ret - p0 + math.pi) % (2 * math.pi) - math.pi).abs().mean().item()
                    err_deg = math.degrees(err_rad)
                    cos_sim = math.cos(err_rad)
                else:
                    err_deg = 180.0
                    cos_sim = 0.0

                elapsed = time.perf_counter() - t0
                ips = images_processed / elapsed
                gamma = model.columns[0].spatial_pooler.gamma()
                stats = model.columns[0].spatial_pooler.permanence_stats()

                history["cross_view_overlap"].append(overlap)
                history["grid_consistency"].append(cos_sim)
                history["anchor_error_deg"].append(err_deg)
                history["step"].append(images_processed)

                logger.info(
                    f"Epoch {epoch+1} | Image {idx_in_epoch+1:5d}/{N} | "
                    f"{ips:5.0f} img/s | "
                    f"γ={gamma:.3f} | "
                    f"p̄={stats['mean']:.3f} | "
                    f"view_overlap={overlap:.3f} | "
                    f"grid_err={err_deg:.1f}° | "
                    f"grid_cos={cos_sim:.3f}"
                )

    return history


# ── Évaluation multi-vue ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_multiview(
    model: CorticalColumn,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_eval: int = 500,
    n_views: int = 5,
) -> dict:
    """
    Évalue la qualité des représentations multi-vues.

    Métriques :
        cross_view_overlap  : Jaccard moyen entre vue 0 et vue 1 (même objet)
        cross_digit_overlap : Jaccard moyen entre vues de digits DIFFÉRENTS
        discrimination      : cross_view / cross_digit (> 1 = signal discriminant)
        grid_path_error     : erreur angulaire après intégration de chemin (°)
        anchor_correction   : réduction d'erreur après anchor() (%)
        consensus_sparsity  : fraction de bits actifs dans le consensus AND

    Args:
        model:   CorticalColumn
        images:  (N, 1, 28, 28)
        labels:  (N,)
        n_eval:  nombre d'images évaluées

    Returns:
        dict des métriques
    """
    model.to(device)
    n_eval = min(n_eval, images.shape[0])

    same_overlaps = []
    diff_overlaps = []
    path_errors_before = []
    path_errors_after = []
    consensus_densities = []

    for i in range(n_eval):
        img = images[i]
        ep = generate_episode(img, n_views=n_views, return_to_origin=True)

        model.reset()
        initial_phases = [col.grid_cell.phases.clone() for col in model.columns]

        sdrs_at_views = []

        # Présentation séquentielle
        for v in range(ep.views.shape[0]):
            s_t = ep.views[v].to(device)
            vel = ep.velocities[v].to(device)
            result = model.step(s_t, vel, train=False)
            sdrs_at_views.append(result["all_sdrs"])   # list of K SDRs

        # ── Métriques de consensus (avant anchor) ────────────────────────
        if len(sdrs_at_views) >= 2:
            # SDRs vue 0 vs vue 1 (même objet, vue différente)
            sdr0 = sdrs_at_views[0][0].float()
            sdr1 = sdrs_at_views[1][0].float()
            inter = (sdr0 * sdr1).sum().item()
            union = ((sdr0 + sdr1) > 0).float().sum().item()
            same_overlaps.append(inter / union if union > 0 else 0.0)

            # Consensus AND à travers toutes les vues (même colonne)
            sdr_stack = torch.stack([
                sdrs_at_views[v][0].float() for v in range(len(sdrs_at_views))
            ], dim=0)   # (N_views, n_sdr)
            consensus = (sdr_stack.mean(0) >= 1.0).float()
            consensus_densities.append(consensus.mean().item())

        # ── Erreur d'intégration de chemin (avant anchor) ────────────────
        phase_end = model.columns[0].grid_cell.phases.clone()
        err_before = (
            (phase_end - initial_phases[0].to(device) + math.pi)
            % (2 * math.pi) - math.pi
        ).abs().mean().item()
        path_errors_before.append(math.degrees(err_before))

        # ── Anchor et erreur après ────────────────────────────────────────
        for c, col in enumerate(model.columns):
            col.grid_cell.anchor(initial_phases[c].to(device), confidence=1.0)
        phase_after = model.columns[0].grid_cell.phases.clone()
        err_after = (
            (phase_after - initial_phases[0].to(device) + math.pi)
            % (2 * math.pi) - math.pi
        ).abs().mean().item()
        path_errors_after.append(math.degrees(err_after))

        # ── Overlap inter-digit (fond de bruit) ──────────────────────────
        j = (i + 1) % n_eval
        ep_j = generate_episode(images[j], n_views=1, return_to_origin=False)
        model.reset()
        result_j = model.step(ep_j.views[0].to(device), torch.zeros(2, device=device), train=False)
        sdr_j = result_j["sdr"].float()
        inter_diff = (sdr0 * sdr_j).sum().item()
        union_diff = ((sdr0 + sdr_j) > 0).float().sum().item()
        diff_overlaps.append(inter_diff / union_diff if union_diff > 0 else 0.0)

    mean_same   = sum(same_overlaps) / len(same_overlaps) if same_overlaps else 0.0
    mean_diff   = sum(diff_overlaps) / len(diff_overlaps) if diff_overlaps else 0.0
    mean_err_b  = sum(path_errors_before) / len(path_errors_before) if path_errors_before else 180.0
    mean_err_a  = sum(path_errors_after) / len(path_errors_after) if path_errors_after else 180.0
    mean_cons   = sum(consensus_densities) / len(consensus_densities) if consensus_densities else 0.0

    anchor_reduction = (
        (mean_err_b - mean_err_a) / mean_err_b * 100
        if mean_err_b > 0 else 0.0
    )
    discrimination = mean_same / mean_diff if mean_diff > 0 else float("inf")

    return {
        "cross_view_overlap":   mean_same,
        "cross_digit_overlap":  mean_diff,
        "discrimination":       discrimination,
        "grid_path_error_deg":  mean_err_b,
        "anchor_error_deg":     mean_err_a,
        "anchor_correction_pct": anchor_reduction,
        "consensus_density":    mean_cons,
    }


# ── Rapport ───────────────────────────────────────────────────────────────────

def print_report(
    metrics_before: dict,
    metrics_after: dict,
    history: dict,
    device: torch.device,
) -> None:
    """Affiche le rapport comparatif avant/après entraînement multi-vue."""
    print("\n" + "=" * 66)
    print(f"  RÉSULTATS — Multi-Vue MNIST  [{device}]")
    print("=" * 66)
    print(f"  {'Métrique':<38} {'Avant':>8} {'Après':>8} {'Δ':>7}")
    print("-" * 66)

    rows = [
        ("SDR overlap (même image, vue diff.)",  "cross_view_overlap",   "{:.3f}"),
        ("SDR overlap (digits différents)",       "cross_digit_overlap",   "{:.3f}"),
        ("Discrimination (same/diff)",            "discrimination",        "{:.2f}"),
        ("Erreur chemin avant anchor (°)",        "grid_path_error_deg",   "{:.1f}°"),
        ("Erreur chemin après anchor (°)",        "anchor_error_deg",      "{:.1f}°"),
        ("Réduction anchor (%)",                  "anchor_correction_pct", "{:.1f}%"),
        ("Densité consensus AND",                 "consensus_density",     "{:.5f}"),
    ]

    for name, key, fmt in rows:
        b = metrics_before.get(key, float("nan"))
        a = metrics_after.get(key, float("nan"))
        bs = fmt.format(b) if not math.isnan(b) else "N/A"
        as_ = fmt.format(a) if not math.isnan(a) else "N/A"
        try:
            d = a - b
            ds = f"{d:+.3f}"
        except Exception:
            ds = "—"
        print(f"  {name:<38} {bs:>8} {as_:>8} {ds:>7}")

    print("-" * 66)

    if history["cross_view_overlap"]:
        final_overlap = history["cross_view_overlap"][-1]
        final_err = history["anchor_error_deg"][-1]
        print(f"  Overlap final (entraînement) : {final_overlap:.3f}")
        print(f"  Erreur anchor finale         : {final_err:.1f}°")

    print("=" * 66 + "\n")

    # Interprétation
    b_disc = metrics_before.get("discrimination", 1.0)
    a_disc = metrics_after.get("discrimination", 1.0)
    if a_disc > b_disc * 1.05:
        print("  ✓ La discrimination s'améliore — les représentations deviennent")
        print("    plus stables à travers les vues qu'entre digits différents.")
    else:
        print("  ~ Discrimination stable — l'entraînement multi-vue n'a pas encore")
        print("    forcé la convergence inter-colonnes. Essayer plus d'epochs ou")
        print("    activer la Phase 2 (PE circuits) pour le feedback L6b→SDR.")

    a_err = metrics_after.get("anchor_error_deg", 180.0)
    if a_err < 10.0:
        print(f"  ✓ Anchor efficace : erreur résiduelle {a_err:.1f}° < 10°.")
    else:
        print(f"  ~ Anchor : erreur résiduelle {a_err:.1f}° — intégration de chemin")
        print("    à améliorer (W_integrator de GridCellNetwork non entraîné).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test multi-vue sur MNIST avec intégration de chemin"
    )
    parser.add_argument("--n_images",      type=int,   default=3000,
                        help="Nombre d'images d'entraînement")
    parser.add_argument("--n_eval",        type=int,   default=300,
                        help="Nombre d'images pour l'évaluation")
    parser.add_argument("--n_epochs",      type=int,   default=2,
                        help="Epochs d'entraînement")
    parser.add_argument("--n_views",       type=int,   default=5,
                        help="Vues par épisode (inclut vue de retour)")
    parser.add_argument("--n_columns",     type=int,   default=4)
    parser.add_argument("--n_sdr",         type=int,   default=2048)
    parser.add_argument("--w",             type=int,   default=40)
    parser.add_argument("--n_minicolumns", type=int,   default=256)
    parser.add_argument("--k_active",      type=int,   default=40)
    parser.add_argument("--n_grid_modules",type=int,   default=6)
    parser.add_argument("--max_translation",type=float, default=4.0)
    parser.add_argument("--max_rotation",  type=float, default=15.0)
    parser.add_argument("--anchor_conf",   type=float, default=0.8,
                        help="Confiance de l'anchor (0=désactivé, 1=correction totale)")
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    device = get_device(args.device)
    logger.info(f"Device : {device}")

    # ── Chargement ───────────────────────────────────────────────────────
    logger.info(f"Chargement MNIST ({args.n_images} images)...")
    images, labels = load_mnist_raw(args.n_images, args.data_dir)
    logger.info(f"  {images.shape}  labels={labels.unique().tolist()}")

    # ── Modèle ───────────────────────────────────────────────────────────
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Modèle : {args.n_columns} colonnes | "
        f"SDR {args.n_sdr}D w={args.w} | "
        f"Grid {4*args.n_grid_modules}D | "
        f"{n_params:,} params"
    )
    logger.info(
        f"Épisodes : {args.n_views} vues | "
        f"translation ±{args.max_translation}px | "
        f"rotation ±{args.max_rotation}° | "
        f"anchor={args.anchor_conf}"
    )

    # ── Évaluation AVANT entraînement ────────────────────────────────────
    logger.info("Évaluation avant entraînement...")
    eval_imgs = images[:args.n_eval]
    eval_lbls = labels[:args.n_eval]
    metrics_before = evaluate_multiview(
        model, eval_imgs, eval_lbls, device,
        n_eval=args.n_eval, n_views=args.n_views,
    )
    logger.info(
        f"  view_overlap={metrics_before['cross_view_overlap']:.3f} | "
        f"discrimination={metrics_before['discrimination']:.2f} | "
        f"path_err={metrics_before['grid_path_error_deg']:.1f}°"
    )

    # ── Entraînement multi-vue ────────────────────────────────────────────
    logger.info(
        f"Entraînement multi-vue : {args.n_epochs} epoch(s) × "
        f"{args.n_images} images × {args.n_views} vues..."
    )
    t0 = time.perf_counter()
    history = train_multiview(
        model, images, device,
        n_epochs=args.n_epochs,
        n_views=args.n_views,
        max_translation=args.max_translation,
        max_rotation=args.max_rotation,
        anchor_confidence=args.anchor_conf,
        log_every=200 if args.verbose else 500,
    )
    elapsed = time.perf_counter() - t0
    total_steps = args.n_images * args.n_epochs
    logger.info(
        f"  Terminé en {elapsed:.1f}s | "
        f"{total_steps / elapsed:.0f} épisodes/s | "
        f"{total_steps * args.n_views / elapsed:.0f} vues/s"
    )

    # ── Évaluation APRÈS entraînement ────────────────────────────────────
    logger.info("Évaluation après entraînement...")
    metrics_after = evaluate_multiview(
        model, eval_imgs, eval_lbls, device,
        n_eval=args.n_eval, n_views=args.n_views,
    )
    logger.info(
        f"  view_overlap={metrics_after['cross_view_overlap']:.3f} | "
        f"discrimination={metrics_after['discrimination']:.2f} | "
        f"path_err={metrics_after['grid_path_error_deg']:.1f}°"
    )

    # ── Permanences ──────────────────────────────────────────────────────
    stats = model.columns[0].spatial_pooler.permanence_stats()
    logger.info(
        f"Permanences : p̄={stats['mean']:.3f} | "
        f"p_conn={stats['frac_connected']:.3f} | "
        f"p→1={stats['frac_near_one']:.3f}"
    )

    # ── Rapport ──────────────────────────────────────────────────────────
    print_report(metrics_before, metrics_after, history, device)


if __name__ == "__main__":
    main()
