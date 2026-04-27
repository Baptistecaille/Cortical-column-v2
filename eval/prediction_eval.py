"""
Évaluation prédictive — Protocoles A et B.

Protocole A : Masquage spatial (MAE-style)
    - Masque k% des patches d'entrée (grille 4×4 = 16 patches)
    - Mesure la MSE de reconstruction sur les patches masqués
      via décodage linéaire inverse : s_recon = W_enc^T @ sdr
    - Compare le modèle à une baseline (mean pixel imputation)
    - Courbe : mask_ratio → MSE

Protocole B : Convergence interne (predictive coding dynamics)
    - Itérations répétées sur le même input fixe (modèle gelé)
    - Mesure ε_step = ||sdr_predicted - sdr||² / n_sdr à chaque pas
    - Courbe de convergence + scalaire : steps_to_90pct_convergence

Réf. : §8, Protocoles d'évaluation — Formalisation_Mille_Cerveaux.pdf
"""

import sys
import os
import argparse
import json
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


# ─── Utilitaires patches ──────────────────────────────────────────────────────

def _build_patch_indices(
    img_height: int = 28,
    img_width: int = 28,
    n_patches_h: int = 4,
    n_patches_w: int = 4,
) -> List[torch.Tensor]:
    """
    Retourne les indices de pixels (1D dans le vecteur aplati) pour chaque
    patch d'une grille n_patches_h × n_patches_w sur une image img_height × img_width.

    Returns:
        Liste de n_patches_h*n_patches_w tenseurs d'indices entiers
    """
    patch_h = img_height // n_patches_h
    patch_w = img_width // n_patches_w
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            rows = torch.arange(i * patch_h, (i + 1) * patch_h)
            cols = torch.arange(j * patch_w, (j + 1) * patch_w)
            idx = (rows.unsqueeze(1) * img_width + cols.unsqueeze(0)).reshape(-1)
            patches.append(idx)
    return patches


def _reconstruct_pixels(
    sdr: torch.Tensor,
    W_enc_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruction pixel approximée depuis un SDR binaire.

    s_recon ≈ W_enc^T @ sdr  (décodage linéaire inverse, sparse coding standard)

    Args:
        sdr:          SDR binaire, shape (n,)
        W_enc_weight: matrice d'encodage, shape (n, input_dim)

    Returns:
        s_recon: shape (input_dim,)
    """
    return W_enc_weight.T @ sdr.float()


# ─── Protocole A ─────────────────────────────────────────────────────────────

class PredictionProtocolA:
    """
    Protocole A : évaluation MAE-style par masquage spatial.

    Pour chaque ratio de masquage mask_ratio ∈ {0.1, 0.3, 0.5, 0.7} :
        1. Masque ceil(mask_ratio × n_patches) patches aléatoires (→ zéros)
        2. Passe l'image masquée dans le modèle → SDR
        3. Reconstruit les pixels via W_enc^T @ sdr
        4. Mesure MSE uniquement sur les pixels des patches masqués
        5. Baseline : imputation par la moyenne pixel (sans modèle)

    Robustesse attendue : MSE_modèle < MSE_baseline pour tout ratio.

    Args:
        model:       CorticalColumn instance
        img_shape:   (H, W) de l'image (défaut : 28×28 MNIST)
        n_patches_h: rangées de patches (défaut : 4)
        n_patches_w: colonnes de patches (défaut : 4)
    """

    def __init__(
        self,
        model,
        img_shape: Tuple[int, int] = (28, 28),
        n_patches_h: int = 4,
        n_patches_w: int = 4,
    ) -> None:
        self.model = model
        self.img_shape = img_shape
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.patch_indices = _build_patch_indices(*img_shape, n_patches_h, n_patches_w)
        self.n_patches = len(self.patch_indices)

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        mask_ratios: Optional[List[float]] = None,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Dict:
        """
        Évalue la MSE de reconstruction pour différents ratios de masquage.

        Args:
            images:      images aplaties, shape (N, input_dim)
            mask_ratios: ratios testés (défaut [0.1, 0.3, 0.5, 0.7])
            n_samples:   nombre d'images (capé à N)
            seed:        graine aléatoire

        Returns:
            {mask_ratios, mse_model, mse_baseline}
        """
        if mask_ratios is None:
            mask_ratios = [0.1, 0.3, 0.5, 0.7]

        torch.manual_seed(seed)
        n_samples = min(n_samples, images.shape[0])
        imgs = images[:n_samples]
        device = imgs.device

        # W_enc de la première colonne — toutes les colonnes partagent la même
        # structure mais pas les mêmes poids ; on utilise la col. 0 pour
        # la reconstruction (représentative du comportement moyen).
        W_enc = self.model.columns[0].sdr_space.W_enc.weight.detach()

        # Moyenne pixel sur l'ensemble — baseline d'imputation
        pixel_mean = imgs.mean(dim=0)  # (input_dim,)

        v_zero = torch.zeros(2, device=device)

        mse_model_list: List[float] = []
        mse_baseline_list: List[float] = []

        for ratio in mask_ratios:
            n_mask = max(1, int(np.ceil(ratio * self.n_patches)))
            model_mses: List[float] = []
            baseline_mses: List[float] = []

            for i in range(n_samples):
                s_clean = imgs[i]

                # Référence : reconstruction depuis image propre
                self.model.reset()
                ref = self.model.step(s_clean, v_zero, train=False)
                s_recon_clean = _reconstruct_pixels(ref["sdr"], W_enc)

                # Patches masqués aléatoires
                perm = torch.randperm(self.n_patches, device=device)[:n_mask]
                mask_idx = torch.cat([
                    self.patch_indices[k.item()].to(device) for k in perm
                ])

                # Image masquée (zéros)
                s_masked = s_clean.clone()
                s_masked[mask_idx] = 0.0

                # Forward modèle sur image masquée
                self.model.reset()
                out = self.model.step(s_masked, v_zero, train=False)
                s_recon_masked = _reconstruct_pixels(out["sdr"], W_enc)

                # MSE sur pixels masqués uniquement
                mse_m = F.mse_loss(
                    s_recon_masked[mask_idx], s_recon_clean[mask_idx]
                ).item()
                model_mses.append(mse_m)

                # Baseline : MSE entre imputation par moyenne et reconstruction propre
                baseline_mses.append(
                    F.mse_loss(
                        pixel_mean[mask_idx], s_clean[mask_idx]
                    ).item()
                )

            mse_model_list.append(float(np.mean(model_mses)))
            mse_baseline_list.append(float(np.mean(baseline_mses)))

        return {
            "mask_ratios": mask_ratios,
            "mse_model":   mse_model_list,
            "mse_baseline": mse_baseline_list,
        }


# ─── Protocole B ─────────────────────────────────────────────────────────────

class PredictionProtocolB:
    """
    Protocole B : convergence interne du predictive coding.

    Lancer n_steps passes itératives sur le même input fixe (modèle gelé) :
        - Pas 0 : état initial (reset)
        - Pas k : prev_grid_code = grid_code_{k-1} → prédiction plus précise
        - ε_k = ||sdr_predicted_k - sdr||² / n_sdr  (erreur de prédiction MSE)

    Convergence attendue : ε décroissant, plateau < 50% de ε_0.
    Métrique scalaire : steps_to_90pct_convergence = premier k tel que
        ε_k ≤ ε_0 + 0.1 × (ε_final − ε_0)   (90% de la réduction totale)

    Args:
        model:   CorticalColumn instance
        n_steps: nombre de passes itératives (défaut : 10)
    """

    def __init__(self, model, n_steps: int = 10) -> None:
        self.model = model
        self.n_steps = n_steps

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        n_samples: int = 50,
        seed: int = 42,
    ) -> Dict:
        """
        Mesure la courbe de convergence ε_step pour chaque image.

        Args:
            images:    images aplaties, shape (N, input_dim)
            n_samples: nombre d'images (capé à N)
            seed:      graine aléatoire

        Returns:
            {steps, l6_error, steps_to_90pct_convergence}
            steps:     [0, 1, ..., n_steps-1]
            l6_error:  ε moyen sur les n_samples images, shape (n_steps,)
            steps_to_90pct_convergence: int (−1 si non atteint)
        """
        torch.manual_seed(seed)
        n_samples = min(n_samples, images.shape[0])
        imgs = images[:n_samples]
        device = imgs.device
        v_zero = torch.zeros(2, device=device)
        n_sdr = self.model.n_sdr

        # Matrice d'erreurs : (n_samples, n_steps)
        errors = torch.zeros(n_samples, self.n_steps)

        for i in range(n_samples):
            s_t = imgs[i]
            # Reset complet → prev_grid_code = 0 (prior uniforme)
            self.model.reset()

            for step_k in range(self.n_steps):
                # Pas sans apprentissage — l'état prev_grid_code est mis à jour
                # naturellement à la fin de chaque step()
                result = self.model.step(s_t, v_zero, train=False)

                sdr_pred = result["sdr_predicted"].float()
                sdr_obs  = result["sdr"].float()
                # ε_k = MSE normalisée sur les bits SDR
                errors[i, step_k] = F.mse_loss(sdr_pred, sdr_obs).item()

        # Moyenne et écart-type sur les échantillons
        mean_errors = errors.mean(dim=0).tolist()  # (n_steps,)

        # Scalaire : steps_to_90pct_convergence
        # Convention : 90% de la réduction totale ε_0 → ε_final
        eps_0     = mean_errors[0]
        eps_final = mean_errors[-1]
        reduction = eps_0 - eps_final
        steps_to_90 = -1

        if reduction > 1e-8:
            target = eps_0 - 0.9 * reduction
            for k, e in enumerate(mean_errors):
                if e <= target:
                    steps_to_90 = k
                    break

        return {
            "steps":   list(range(self.n_steps)),
            "l6_error": mean_errors,
            "steps_to_90pct_convergence": steps_to_90,
        }


# ─── Sanity checks ───────────────────────────────────────────────────────────

def _check_protocol_a(results: Dict) -> List[str]:
    """Vérifie les conditions de sanité du Protocole A, retourne les warnings."""
    warnings_out = []
    mse_m = results["mse_model"]
    mse_b = results["mse_baseline"]

    # MSE doit croître avec le masquage
    for i in range(len(mse_m) - 1):
        if mse_m[i] > mse_m[i + 1] + 1e-6:
            warnings_out.append(
                f"[WARN] Protocole A : MSE_modèle décroît entre "
                f"mask_ratio={results['mask_ratios'][i]} et "
                f"mask_ratio={results['mask_ratios'][i+1]} — "
                "pas de dégradation gracieuse"
            )

    # Le modèle doit être meilleur que la baseline
    for r, mm, mb in zip(results["mask_ratios"], mse_m, mse_b):
        if mm > mb:
            warnings_out.append(
                f"[WARN] Protocole A : MSE_modèle ({mm:.4f}) > MSE_baseline "
                f"({mb:.4f}) à mask_ratio={r} — le modèle est moins bon que "
                "l'imputation par la moyenne"
            )
    return warnings_out


def _check_protocol_b(results: Dict) -> List[str]:
    """Vérifie les conditions de sanité du Protocole B."""
    warnings_out = []
    errors = results["l6_error"]

    if len(errors) < 2:
        return warnings_out

    eps_0 = errors[0]
    eps_final = errors[-1]

    # Erreur doit décroître
    if eps_final > eps_0 * 1.05:
        warnings_out.append(
            f"[WARN] Protocole B : ε_final ({eps_final:.4f}) > ε_0 ({eps_0:.4f}) "
            "— pas de convergence détectée"
        )

    # Plateau doit être < 50% de l'erreur initiale
    if eps_final > 0.5 * eps_0:
        warnings_out.append(
            f"[WARN] Protocole B : ε_final ({eps_final:.4f}) > 0.5 × ε_0 "
            f"({0.5 * eps_0:.4f}) — convergence insuffisante (plateau trop élevé)"
        )

    return warnings_out


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_protocol_a(results: Dict, output_path: str) -> None:
    """Génère la courbe MSE vs mask_ratio (Protocole A)."""
    if not _MPL_OK:
        print("[INFO] matplotlib non disponible — graphique non généré")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ratios = results["mask_ratios"]
    ax.plot(ratios, results["mse_model"],   "o-", color="#1f77b4", label="Modèle")
    ax.plot(ratios, results["mse_baseline"], "s--", color="#d62728", label="Baseline (mean imputation)")
    ax.set_xlabel("Ratio de masquage")
    ax.set_ylabel("MSE de reconstruction")
    ax.set_title("Protocole A — Dégradation par masquage spatial")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Protocole A → {output_path}")


def plot_protocol_b(results: Dict, output_path: str) -> None:
    """Génère la courbe ε_L6 vs inference step (Protocole B)."""
    if not _MPL_OK:
        print("[INFO] matplotlib non disponible — graphique non généré")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    steps = results["steps"]
    errors = results["l6_error"]
    ax.plot(steps, errors, "o-", color="#2ca02c")

    s90 = results["steps_to_90pct_convergence"]
    if s90 >= 0 and s90 < len(steps):
        ax.axvline(s90, color="#ff7f0e", linestyle="--",
                   label=f"90% convergence @ step {s90}")
        ax.legend()

    ax.set_xlabel("Pas d'inférence")
    ax.set_ylabel("ε — erreur de prédiction L6→L4 (MSE normalisée)")
    ax.set_title("Protocole B — Convergence prédictive interne")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Protocole B → {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation prédictive — Protocoles A et B"
    )
    parser.add_argument("--dataset",   type=str, default="mnist")
    parser.add_argument("--data_dir",  type=str, default="./data")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Nombre d'images pour les évaluations")
    parser.add_argument("--n_steps_b", type=int, default=10,
                        help="Nombre de pas d'inférence (Protocole B)")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--n_columns", type=int, default=4)
    parser.add_argument("--n_sdr",     type=int, default=2048)
    parser.add_argument("--w",         type=int, default=40)
    parser.add_argument("--n_grid_modules", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers un checkpoint PyTorch (.pt)")
    parser.add_argument("--device",    type=str, default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from column import CorticalColumn

    device = torch.device(args.device)

    # Chargement du modèle
    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11, 13, 17][:args.n_grid_modules],
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[OK] Checkpoint chargé : {args.checkpoint}")
    else:
        print("[INFO] Aucun checkpoint — modèle non entraîné (test de structure)")

    # Chargement des images
    import torchvision
    import torchvision.transforms as T
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
    report = {}

    # ── Protocole A ──────────────────────────────────────────────────────────
    print("\n[Protocole A] Masquage spatial...")
    proto_a = PredictionProtocolA(model)
    res_a = proto_a.evaluate(imgs, n_samples=args.n_samples)
    report["prediction_A"] = res_a

    for w in _check_protocol_a(res_a):
        print(w)

    plot_protocol_a(res_a, str(out_dir / "prediction_A_curve.png"))

    # ── Protocole B ──────────────────────────────────────────────────────────
    print("\n[Protocole B] Convergence prédictive...")
    proto_b = PredictionProtocolB(model, n_steps=args.n_steps_b)
    res_b = proto_b.evaluate(imgs, n_samples=min(args.n_samples, 50))
    report["prediction_B"] = res_b

    for w in _check_protocol_b(res_b):
        print(w)

    plot_protocol_b(res_b, str(out_dir / "prediction_B_convergence.png"))

    # ── Rapport JSON ─────────────────────────────────────────────────────────
    report_path = out_dir / "prediction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Rapport → {report_path}")


if __name__ == "__main__":
    main()
