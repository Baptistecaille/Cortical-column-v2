"""
Module 4 — GridCellNetwork
Intégration de chemin sur tore 2D 𝕋².

Invariants :
    I4.1 : φ_k ∈ [0, 2π)² pour tout module k (mise à jour mod 2π)
    I4.2 : get_code() produit [cos(φ₀), sin(φ₀), cos(φ₁), sin(φ₁)] par module
           → dimension totale 4·n_modules (pas 2·n_modules)
    I4.3 : Correction anchor() pondérée par 1/λ_k (pas uniforme)

Règle d'intégration (pseudo-algo CLAUDE.md §3) :
    φ_k ← (φ_k + R_k · v_t) mod 2π
    où v_t est la VITESSE 2D directement (pas l'allo_phase de Layer6b).
    R_k est une matrice de rotation 2×2 par module, scalée par 2π/λ_k.

Ancienne implémentation (bug) :
    W_integrator apprenait à mapper allo_phase (12D) → delta_phi.
    Initialisé orthogonalement, il produisait ~90° d'erreur aléatoire.
    Fix : intégration directe de la vitesse via R_k analytique.

Propriété de retour à l'origine :
    Si ∑v_t = 0 sur un épisode (retour au point de départ),
    alors ∑Δφ_k = R_k · ∑v_t = 0 → les phases reviennent exactement
    à leur valeur initiale. L'anchor() ne corrige que la dérive de
    bruit numérique (proche de 0).

Réf. math : §4 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple
from math import gcd
from functools import reduce


def _are_coprime(periods: List[int]) -> bool:
    """Vérifie que les périodes sont copremières deux à deux (CRT)."""
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            if gcd(periods[i], periods[j]) != 1:
                return False
    return True


class GridCellNetwork(nn.Module):
    """
    Réseau de cellules de grille pour l'intégration de chemin sur 𝕋².

    Chaque module k maintient une phase 2D φ_k ∈ [0, 2π)²
    mise à jour directement par la vitesse 2D : φ_k ← φ_k + R_k · v_t.

    R_k = (2π / λ_k) · I₂  : rotation isotropique scalée par la période.
    Les modules à petite période (λ=3) ont une plus grande résolution spatiale
    — ils détectent les petits déplacements avec plus de précision.

    Args:
        n_modules:  nombre de modules de grid cells
        periods:    liste d'entiers λ_k (périodes spatiales, copremières)
        velocity_scale: facteur de normalisation des vitesses entrantes.
                    Doit correspondre au velocity_scale utilisé lors de la
                    génération des épisodes (défaut 10.0 → v ∈ [-0.4, 0.4]).
    """

    DEFAULT_PERIODS = [3, 5, 7, 11, 13, 17, 19, 23]

    def __init__(
        self,
        n_modules: int = 6,
        periods: Optional[List[int]] = None,
        velocity_scale: float = 10.0,
        use_linoss: bool = False,
    ) -> None:
        super().__init__()

        self.n_modules = n_modules
        self.velocity_scale = velocity_scale

        if periods is None:
            periods = self.DEFAULT_PERIODS[:n_modules]
        assert len(periods) == n_modules
        assert _are_coprime(periods), (
            f"Les périodes {periods} ne sont pas copremières — viole le CRT (§4.3)"
        )
        self.periods = periods

        lambda_k = torch.tensor(periods, dtype=torch.float)
        self.register_buffer("lambda_k", lambda_k)

        # ── Phases φ_k ∈ [0, 2π)² ────────────────────────────────────────
        phases = torch.rand(n_modules, 2) * 2 * math.pi
        self.register_buffer("phases", phases)

        # ── Matrices de rotation R_k ──────────────────────────────────────
        # R_k = (2π / λ_k) · I₂ : chaque module intègre la vitesse à sa
        # propre résolution spatiale. Pas de paramètre appris — analytique.
        # Shape : (n_modules,) — scalaire par module (isotropique)
        scales = (2 * math.pi / lambda_k)   # (n_modules,)
        self.register_buffer("R_k_scale", scales)

        # ── LinOSS path integration (Rusch & Rus 2024) — optionnel ──────────
        # Remplace l'intégration Euler R_k * v_t par un ODE stable oscillatoire.
        # Entrée : vitesse 2D ; sortie : delta_phi (n_modules, 2).
        # I4.1 préservé par mod 2π appliqué après la sortie LinOSS.
        self.use_linoss = use_linoss
        if use_linoss:
            from extensions.linoss import LinOSSLayer
            self.linoss = LinOSSLayer(
                input_dim=2,
                state_dim=4 * n_modules,
                output_dim=2 * n_modules,
                dt=1.0,
            )
        else:
            self.linoss = None

    # ── Intégration de chemin (vitesse directe) ───────────────────────────

    def integrate(
        self,
        velocity: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Intègre la vitesse 2D pour mettre à jour les phases sur 𝕋².

        Mode Euler (use_linoss=False, défaut) :
            δφ_k = R_k_scale[k] · v_t · dt

        Mode LinOSS (use_linoss=True, Rusch & Rus 2024 arXiv 2410.03943) :
            δφ = linoss(v_t).reshape(n_modules, 2), clamped to (-π, π)
            Invariant I4.1 préservé par mod 2π.

        Args:
            velocity: vecteur vitesse 2D, shape (2,)
            dt:       pas de temps (mode Euler uniquement)

        Returns:
            phases: shape (n_modules, 2)
        """
        if self.use_linoss and self.linoss is not None:
            delta_phi_flat = self.linoss(velocity.float())
            delta_phi = delta_phi_flat.view(self.n_modules, 2).clamp(-math.pi, math.pi)
            new_phases = (self.phases + dt * delta_phi) % (2 * math.pi)
            self.phases.data = new_phases.detach()
        else:
            with torch.no_grad():
                delta_phi = (
                    self.R_k_scale.unsqueeze(1) * velocity.float().unsqueeze(0)
                )
                new_phases = (self.phases + dt * delta_phi) % (2 * math.pi)
                self.phases.data = new_phases
        return self.phases   # (n_modules, 2)

    # ── Code de position ──────────────────────────────────────────────────

    @torch.no_grad()
    def get_code(self) -> torch.Tensor:
        """
        Code de position : [cos(φ_k,0), sin(φ_k,0), cos(φ_k,1), sin(φ_k,1)]
        par module. Dimension = 4 × n_modules. Invariant I4.2.
        """
        phi_0 = self.phases[:, 0]
        phi_1 = self.phases[:, 1]
        code = torch.stack(
            [torch.cos(phi_0), torch.sin(phi_0),
             torch.cos(phi_1), torch.sin(phi_1)],
            dim=-1,
        ).reshape(4 * self.n_modules)
        return code

    # ── Intégration batché ────────────────────────────────────────────────

    @torch.no_grad()
    def integrate_batch(
        self,
        vel_batch: torch.Tensor,
        phases_batch: torch.Tensor,
        dt: float = 1.0,
    ) -> "Tuple[torch.Tensor, torch.Tensor]":
        """
        Intègre la vitesse pour B épisodes indépendants en parallèle.

        Contrairement à integrate(), les phases sont passées et retournées
        explicitement — aucune mutation du buffer self.phases.
        Cela permet de maintenir B états de phases indépendants.

        Invariant I4.1 : φ_k ∈ [0, 2π)² après mod 2π.
        Invariant I4.2 : code retourné = [cos φ₀, sin φ₀, cos φ₁, sin φ₁]
                         par module → dimension 4·n_modules.

        Args:
            vel_batch:    vitesses 2D, shape (B, 2)
            phases_batch: phases courantes, shape (B, n_modules, 2)
            dt:           pas de temps

        Returns:
            new_phases: shape (B, n_modules, 2)
            grid_codes: shape (B, 4 * n_modules)
        """
        # delta_phi[b, k, :] = R_k_scale[k] * vel_batch[b, :]
        # R_k_scale : (n_modules,) → (1, n_modules, 1) pour broadcast
        delta_phi = (
            self.R_k_scale.view(1, self.n_modules, 1)
            * vel_batch.float().unsqueeze(1)
        )                                                    # (B, n_modules, 2)

        new_phases = (phases_batch + dt * delta_phi) % (2 * math.pi)  # (B, n_modules, 2)

        # Code cos/sin : (B, 4*n_modules)
        phi_0 = new_phases[:, :, 0]                         # (B, n_modules)
        phi_1 = new_phases[:, :, 1]                         # (B, n_modules)
        grid_codes = torch.stack(
            [torch.cos(phi_0), torch.sin(phi_0),
             torch.cos(phi_1), torch.sin(phi_1)],
            dim=-1,
        ).reshape(vel_batch.shape[0], 4 * self.n_modules)   # (B, 4*n_modules)

        return new_phases, grid_codes

    # ── Ancrage ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def anchor(
        self,
        landmark_phases: torch.Tensor,
        confidence: float = 1.0,
    ) -> None:
        """
        Corrige les phases par ancrage sur un repère visuel.

        Pondéré par 1/λ_k (I4.3) — les modules haute résolution
        (petite période) reçoivent une correction plus forte.

        Avec l'intégration directe de la vitesse, l'erreur résiduelle
        sur un épisode bien formé est proche de 0 (dérive numérique).
        anchor() corrige néanmoins pour les épisodes avec bruit ou
        erreurs d'arrondi dans les translations.

        Réf. math : §4.5, Éq. 4.8.

        Args:
            landmark_phases: phases cibles, shape (n_modules, 2)
            confidence:      confiance dans le repère ∈ [0, 1]
        """
        if landmark_phases.shape != self.phases.shape:
            raise ValueError(
                f"landmark_phases.shape={landmark_phases.shape} ≠ "
                f"phases.shape={self.phases.shape}"
            )

        weights = (1.0 / self.lambda_k).unsqueeze(-1)   # (n_modules, 1)
        weights = weights / weights.sum()

        delta = (landmark_phases - self.phases + math.pi) % (2 * math.pi) - math.pi
        correction = confidence * weights * delta
        self.phases.data = (self.phases + correction) % (2 * math.pi)

    @torch.no_grad()
    def anchor_batch(
        self,
        phases_batch: torch.Tensor,
        landmark_batch: torch.Tensor,
        confidence: float = 1.0,
    ) -> torch.Tensor:
        """
        Ancrage pour B épisodes en parallèle — version fonctionnelle (no buffer).

        Invariant I4.3 : correction pondérée par 1/λ_k (pas uniforme).

        Args:
            phases_batch:   phases courantes, shape (B, n_modules, 2)
            landmark_batch: phases cibles,    shape (B, n_modules, 2)
            confidence:     confiance dans le repère ∈ [0, 1]

        Returns:
            new_phases: shape (B, n_modules, 2)
        """
        # Poids 1/λ_k normalisés : (1, n_modules, 1) pour broadcast sur B et axe 2D
        weights = (1.0 / self.lambda_k).unsqueeze(-1)   # (n_modules, 1)
        weights = (weights / weights.sum()).view(1, self.n_modules, 1)  # (1, n_mod, 1)

        delta = (landmark_batch - phases_batch + math.pi) % (2 * math.pi) - math.pi
        correction = confidence * weights * delta
        return (phases_batch + correction) % (2 * math.pi)

    # ── Utilitaires ───────────────────────────────────────────────────────

    @torch.no_grad()
    def reset(self) -> None:
        """Réinitialise les phases aléatoirement (nouvel épisode)."""
        self.phases.data = torch.rand_like(self.phases) * 2 * math.pi

    @torch.no_grad()
    def reset_to(self, phases: torch.Tensor) -> None:
        """Fixe les phases à une valeur donnée (reprise d'épisode)."""
        self.phases.data = phases.to(self.phases.device).clone()

    def position_capacity(self) -> int:
        """Capacité de résolution = ∏λ_k (Thm CRT)."""
        return reduce(lambda a, b: a * b, self.periods)

    def path_error_deg(self, target_phases: torch.Tensor) -> float:
        """Erreur angulaire moyenne entre phases courantes et cibles (°)."""
        diff = (self.phases.cpu() - target_phases + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(diff.abs().mean().item())

    def extra_repr(self) -> str:
        return (
            f"n_modules={self.n_modules}, periods={self.periods}, "
            f"capacity={self.position_capacity()}, "
            f"velocity_scale={self.velocity_scale}, "
            f"use_linoss={self.use_linoss}"
        )
