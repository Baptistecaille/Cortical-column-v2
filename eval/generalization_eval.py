"""
Évaluation de généralisation OOD (Out-Of-Distribution).

Transformations appliquées après entraînement (modèle gelé) :
    - Flou gaussien : kernel 3×3, 5×5
    - Rotation      : 15°, 45°, 90°
    - Occlusion     : 25%, 50% des patches masqués

Pour chaque transformation :
    - Dégradation de la linear probing accuracy vs baseline propre
    - Dégradation de la MSE de reconstruction vs baseline propre
    - Robustness Score = moyenne des scores relatifs sur toutes les intensités

Robustness Score ∈ [0, 1] :
    RS = 1 → aucune dégradation (idéal)
    RS = 0 → échec total

Réf. : §8 — Formalisation_Mille_Cerveaux.pdf
"""

import sys
import os
import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import torchvision.transforms.functional as TF
    _TF_OK = True
except ImportError:
    _TF_OK = False


# ─── Transforms ──────────────────────────────────────────────────────────────

def _apply_gaussian_blur(
    imgs_flat: torch.Tensor,
    kernel_size: int,
    img_shape: Tuple[int, int] = (28, 28),
) -> torch.Tensor:
    """
    Flou gaussien sur un batch d'images aplaties.

    Args:
        imgs_flat: (N, H×W) — images MNIST aplaties
        kernel_size: taille du noyau (impair)
        img_shape: (H, W) de l'image

    Returns:
        (N, H×W) — images floutées aplaties
    """
    H, W = img_shape
    imgs_4d = imgs_flat.view(-1, 1, H, W)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # formule OpenCV
    blurred = TF.gaussian_blur(imgs_4d, kernel_size=[kernel_size, kernel_size],
                               sigma=[sigma, sigma])
    return blurred.view(-1, H * W)


def _apply_rotation(
    imgs_flat: torch.Tensor,
    angle_deg: float,
    img_shape: Tuple[int, int] = (28, 28),
) -> torch.Tensor:
    """
    Rotation d'un batch d'images aplaties.

    Args:
        imgs_flat: (N, H×W)
        angle_deg: angle en degrés (sens antihoraire)
        img_shape: (H, W)

    Returns:
        (N, H×W) — images pivotées aplaties
    """
    H, W = img_shape
    imgs_4d = imgs_flat.view(-1, 1, H, W)
    rotated = TF.rotate(imgs_4d, angle=angle_deg)
    return rotated.view(-1, H * W)


def _apply_occlusion(
    imgs_flat: torch.Tensor,
    occlude_ratio: float,
    img_shape: Tuple[int, int] = (28, 28),
    n_patches_h: int = 4,
    n_patches_w: int = 4,
    seed: int = 99,
) -> torch.Tensor:
    """
    Occlusion aléatoire par masquage de patches.

    Args:
        imgs_flat:     (N, H×W)
        occlude_ratio: fraction des patches à masquer
        img_shape:     (H, W)
        seed:          graine pour reproductibilité inter-samples

    Returns:
        (N, H×W) — images occultées aplaties
    """
    from eval.prediction_eval import _build_patch_indices
    H, W = img_shape
    patch_indices = _build_patch_indices(H, W, n_patches_h, n_patches_w)
    n_patches = len(patch_indices)
    n_mask = max(1, int(np.ceil(occlude_ratio * n_patches)))

    torch.manual_seed(seed)
    perm = torch.randperm(n_patches)[:n_mask]
    mask_idx = torch.cat([patch_indices[k.item()] for k in perm])

    result = imgs_flat.clone()
    result[:, mask_idx] = 0.0
    return result


# ─── Extraction de représentations ───────────────────────────────────────────

@torch.no_grad()
def _extract_representations(
    model,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extrait les représentations SDR pour un batch d'images.

    Chaque image est traitée indépendamment (reset entre images).

    Returns:
        sdrs: shape (N, n_sdr), float
    """
    v_zero = torch.zeros(2, device=device)
    sdrs = []
    for i in range(images.shape[0]):
        model.reset()
        result = model.step(images[i], v_zero, train=False)
        sdrs.append(result["sdr"].float().cpu())
    return torch.stack(sdrs, dim=0)


@torch.no_grad()
def _compute_recon_mse(
    model,
    images: torch.Tensor,
    W_enc: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Calcule la MSE de reconstruction pixel moyennée sur le batch.

    s_recon = W_enc^T @ sdr
    MSE = mean_i(||s_recon_i - s_i||²)
    """
    from eval.prediction_eval import _reconstruct_pixels
    v_zero = torch.zeros(2, device=device)
    total_mse = 0.0
    N = images.shape[0]
    for i in range(N):
        s = images[i]
        model.reset()
        result = model.step(s, v_zero, train=False)
        s_recon = _reconstruct_pixels(result["sdr"], W_enc)
        total_mse += F.mse_loss(s_recon, s).item()
    return total_mse / N


def _linear_probing_accuracy(
    reprs: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    n_epochs: int = 30,
    lr: float = 0.01,
) -> float:
    """
    Sondage linéaire rapide (sous-ensemble des epochs de unsupervised_eval).
    """
    import torch.nn as nn
    N, dim = reprs.shape
    classifier = nn.Linear(dim, n_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    split = max(1, int(0.8 * N))
    X_tr, y_tr = reprs[:split].detach(), labels[:split]
    X_te, y_te = reprs[split:].detach(), labels[split:]

    with torch.enable_grad():
        for _ in range(n_epochs):
            optimizer.zero_grad()
            F.cross_entropy(classifier(X_tr), y_tr).backward()
            optimizer.step()

    with torch.no_grad():
        acc = (classifier(X_te).argmax(-1) == y_te).float().mean().item()
    return float(acc)


# ─── OOD Evaluator ───────────────────────────────────────────────────────────

class OODEvaluator:
    """
    Évaluateur de généralisation OOD sur transformations contrôlées.

    Pour chaque transformation T et intensité i :
        1. Applique T_i aux images de test
        2. Extrait les représentations SDR (modèle gelé)
        3. Mesure linear probing accuracy et MSE de reconstruction
        4. Calcule le Robustness Score relatif à la baseline propre

    Robustness Score d'un transform :
        RS_acc = mean_i(acc_i / acc_clean)
        RS_mse = mean_i(mse_clean / mse_i)   ← inversé (plus petit = mieux)

    Args:
        model:      CorticalColumn instance (gelé)
        img_shape:  (H, W) de l'image (défaut 28×28)
        n_classes:  nombre de classes (défaut 10 pour MNIST)
    """

    # Définition des transformations et intensités
    TRANSFORMS = {
        "blur_3x3":     ("blur",     3),
        "blur_5x5":     ("blur",     5),
        "rotation_15":  ("rotation", 15),
        "rotation_45":  ("rotation", 45),
        "rotation_90":  ("rotation", 90),
        "occlusion_25": ("occlusion", 0.25),
        "occlusion_50": ("occlusion", 0.50),
    }

    # Seuils de sanité attendus
    SANITY = {
        "rotation_90":  {"rs_max": 0.5,  "label": "rotation 90° fort"},
        "blur_3x3":     {"rs_min": 0.7,  "label": "flou 3×3 léger"},
        "occlusion_25": {"rs_min": 0.55, "label": "occlusion 25% (vote colonnes)"},
    }

    def __init__(
        self,
        model,
        img_shape: Tuple[int, int] = (28, 28),
        n_classes: int = 10,
    ) -> None:
        self.model = model
        self.img_shape = img_shape
        self.n_classes = n_classes

    def _apply_transform(
        self,
        imgs: torch.Tensor,
        transform_type: str,
        intensity,
    ) -> torch.Tensor:
        if transform_type == "blur":
            return _apply_gaussian_blur(imgs, int(intensity), self.img_shape)
        elif transform_type == "rotation":
            return _apply_rotation(imgs, float(intensity), self.img_shape)
        elif transform_type == "occlusion":
            return _apply_occlusion(imgs, float(intensity), self.img_shape)
        else:
            raise ValueError(f"Transform inconnu : {transform_type}")

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 100,
    ) -> Dict:
        """
        Évalue toutes les transformations OOD.

        Args:
            images:    images aplaties, shape (N, input_dim)
            labels:    étiquettes entières, shape (N,)
            n_samples: nombre d'images utilisées

        Returns:
            dict avec clés : "clean_baseline" + une entrée par transformation
        """
        n_samples = min(n_samples, images.shape[0])
        imgs = images[:n_samples]
        lbls = labels[:n_samples]
        device = imgs.device

        W_enc = self.model.columns[0].sdr_space.W_enc.weight.detach()

        # ── Baseline propre ──────────────────────────────────────────────
        print("  Baseline propre...")
        reprs_clean = _extract_representations(self.model, imgs, device)
        acc_clean   = _linear_probing_accuracy(reprs_clean, lbls, self.n_classes)
        mse_clean   = _compute_recon_mse(self.model, imgs, W_enc, device)

        results: Dict = {
            "clean_baseline": {
                "linear_probing_acc": acc_clean,
                "recon_mse":          mse_clean,
            }
        }

        # ── Chaque transformation ────────────────────────────────────────
        for key, (t_type, intensity) in self.TRANSFORMS.items():
            print(f"  {key}...")
            imgs_t = self._apply_transform(imgs, t_type, intensity)
            reprs_t = _extract_representations(self.model, imgs_t, device)
            acc_t   = _linear_probing_accuracy(reprs_t, lbls, self.n_classes)
            mse_t   = _compute_recon_mse(self.model, imgs_t, W_enc, device)

            # Robustness Score
            rs_acc = acc_t / max(acc_clean, 1e-8)
            rs_mse = mse_clean / max(mse_t, 1e-8)
            # Moyenne des deux indicateurs → RS global ∈ [0, 1]
            rs = float(np.clip((rs_acc + rs_mse) / 2.0, 0.0, 1.0))

            results[key] = {
                "linear_probing_acc": float(acc_t),
                "recon_mse":          float(mse_t),
                "robustness_score":   rs,
            }

        return results

    def check_sanity(self, results: Dict) -> List[str]:
        """Retourne la liste des warnings de sanité."""
        warnings_out = []
        for key, thresholds in self.SANITY.items():
            if key not in results:
                continue
            rs = results[key].get("robustness_score", float("nan"))
            label = thresholds["label"]

            if "rs_max" in thresholds and rs > thresholds["rs_max"]:
                warnings_out.append(
                    f"[WARN] OOD {key} : RS={rs:.3f} > seuil max "
                    f"{thresholds['rs_max']} — dégradation insuffisante "
                    f"pour {label}"
                )
            if "rs_min" in thresholds and rs < thresholds["rs_min"]:
                warnings_out.append(
                    f"[WARN] OOD {key} : RS={rs:.3f} < seuil min "
                    f"{thresholds['rs_min']} — robustesse insuffisante "
                    f"pour {label}"
                )
        return warnings_out


# ─── Plots ────────────────────────────────────────────────────────────────────

def _plot_ood_metric(
    results: Dict,
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: str,
    higher_is_better: bool = True,
) -> None:
    """Génère une courbe de dégradation OOD pour une métrique donnée."""
    if not _MPL_OK:
        return

    # Groupement par famille de transformations
    families = {
        "Flou gaussien": ["blur_3x3", "blur_5x5"],
        "Rotation":      ["rotation_15", "rotation_45", "rotation_90"],
        "Occlusion":     ["occlusion_25", "occlusion_50"],
    }
    x_labels = {
        "blur_3x3": "3×3", "blur_5x5": "5×5",
        "rotation_15": "15°", "rotation_45": "45°", "rotation_90": "90°",
        "occlusion_25": "25%", "occlusion_50": "50%",
    }
    colors = {"Flou gaussien": "#1f77b4", "Rotation": "#ff7f0e", "Occlusion": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_val = results["clean_baseline"].get(metric_key, None)
    if baseline_val is not None:
        ax.axhline(baseline_val, color="grey", linestyle=":", label="Baseline propre")

    for family, keys in families.items():
        vals = [results[k][metric_key] for k in keys if k in results]
        xlabels = [x_labels[k] for k in keys if k in results]
        if vals:
            ax.plot(range(len(vals)), vals, "o-", color=colors[family], label=family)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(xlabels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] OOD → {output_path}")


def plot_ood_accuracy(results: Dict, output_path: str) -> None:
    """Courbe de dégradation de l'accuracy OOD."""
    _plot_ood_metric(
        results,
        metric_key="linear_probing_acc",
        ylabel="Linear probing accuracy",
        title="OOD — Dégradation de la précision",
        output_path=output_path,
        higher_is_better=True,
    )


def plot_ood_reconstruction(results: Dict, output_path: str) -> None:
    """Courbe de dégradation de la MSE OOD."""
    _plot_ood_metric(
        results,
        metric_key="recon_mse",
        ylabel="MSE de reconstruction",
        title="OOD — Dégradation de la reconstruction",
        output_path=output_path,
        higher_is_better=False,
    )


def plot_ood_robustness(results: Dict, output_path: str) -> None:
    """Barplot des Robustness Scores par transformation."""
    if not _MPL_OK:
        return

    keys = [k for k in results if k != "clean_baseline"]
    scores = [results[k].get("robustness_score", 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if s >= 0.7 else "#ff7f0e" if s >= 0.5 else "#d62728"
              for s in scores]
    ax.bar(range(len(keys)), scores, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.axhline(0.7, color="grey", linestyle="--", label="Seuil robustesse (0.7)")
    ax.axhline(0.5, color="grey", linestyle=":", label="Seuil dégradation (0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Robustness Score")
    ax.set_title("OOD — Robustness Scores par transformation")
    ax.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] OOD Robustness → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation OOD — généralisation hors distribution"
    )
    parser.add_argument("--dataset",    type=str, default="mnist")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--n_columns",  type=int, default=4)
    parser.add_argument("--n_sdr",      type=int, default=2048)
    parser.add_argument("--w",          type=int, default=40)
    parser.add_argument("--n_grid_modules", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from column import CorticalColumn
    import torchvision
    import torchvision.transforms as T

    device = torch.device(args.device)

    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11, 13, 17][:args.n_grid_modules],
    ).to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERREUR] Checkpoint introuvable : {args.checkpoint}")
            print("[INFO] Lancement avec un modèle non entraîné (poids aléatoires)")
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"[OK] Checkpoint chargé : {args.checkpoint}")

    ds = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=T.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.n_samples, shuffle=False, num_workers=0
    )
    imgs_raw, labels = next(iter(loader))
    imgs = imgs_raw.view(imgs_raw.shape[0], -1).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[OOD] Évaluation généralisation...")
    evaluator = OODEvaluator(model, n_classes=10)
    results = evaluator.evaluate(imgs, labels, n_samples=args.n_samples)

    for w in evaluator.check_sanity(results):
        print(w)

    plot_ood_accuracy(results, str(out_dir / "ood_accuracy_degradation.png"))
    plot_ood_reconstruction(results, str(out_dir / "ood_reconstruction_degradation.png"))
    plot_ood_robustness(results, str(out_dir / "ood_robustness_scores.png"))

    report_path = out_dir / "ood_report.json"
    with open(report_path, "w") as f:
        json.dump({"ood": results}, f, indent=2)
    print(f"\n[OK] Rapport → {report_path}")


if __name__ == "__main__":
    main()
