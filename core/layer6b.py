"""
Module 3 — Layer6bTransformer
Transformation ego → allocentrique (L6b comme rotateur de référentiel).

Invariants :
    I3.1 : La transformation est une isométrie (norme préservée)

Biologie :
    L6b projette vers le thalamus (TC) et reçoit le feedback thalamique
    (hT/NRT) — la boucle TC/hT/NRT est intégrée ici, une seule mise à jour
    par pas.

Piège critique :
    Ne JAMAIS réduire L6b à un .mean() — la sortie doit rester un vecteur
    de phase 2D par module, pas un scalaire.

Réf. math : §3, Déf. 3.1 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
from typing import Tuple


class ThalamicLoop(nn.Module):
    """
    Boucle thalamique TC/hT/NRT (Layer 6b → Thalamus → Layer 4).

    Modélise le circuit :
        TC (Thalamo-Cortical) : reçoit la sortie L6b
        hT  (hyperpolarisation thalamique) : modulation inhibitrice
        NRT (Nucleus Reticularis Thalami) : contrôle de gain

    La dynamique est une RNN linéaire à une couche avec contrainte d'isométrie.

    Args:
        dim: dimension du vecteur de phase (= 2 × n_grid_modules)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        # TC : projection L6b → état thalamique
        self.W_tc = nn.Linear(dim, dim, bias=False)
        # hT : feedback thalamique → L6b
        self.W_ht = nn.Linear(dim, dim, bias=False)
        # NRT : contrôle de gain (scalaire par dimension)
        self.g_nrt = nn.Parameter(torch.ones(dim))

        # Initialisation orthogonale → isométrie (I3.1)
        nn.init.orthogonal_(self.W_tc.weight)
        nn.init.orthogonal_(self.W_ht.weight)

        # État thalamique interne (buffer)
        self.register_buffer("h_tc", torch.zeros(dim))

    def forward(self, z_l6b: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour de la boucle thalamique (une seule fois par pas).

        Args:
            z_l6b: sortie L6b, shape (dim,)

        Returns:
            feedback: signal thalamique retourné à L4, shape (dim,)
        """
        # TC : intégration de la sortie L6b
        h_new = torch.tanh(self.W_tc(z_l6b) + self.W_ht(self.h_tc))

        # NRT : modulation de gain (sigmoïde pour borner ∈ [0,1])
        gain = torch.sigmoid(self.g_nrt)

        # Mise à jour de l'état thalamique
        self.h_tc.data = h_new

        # Signal de feedback pondéré par le gain NRT
        feedback = gain * h_new
        return feedback

    def reset(self) -> None:
        """Réinitialise l'état thalamique (nouveau stimulus)."""
        self.h_tc.data.zero_()


class Layer6bTransformer(nn.Module):
    """
    Transformateur de référentiel ego → allocentrique via L6b.

    La transformation intègre :
    1. Une rotation dans l'espace des phases guidée par la vitesse ego.
    2. Une correction thalamique (boucle TC/hT/NRT).

    La sortie est un vecteur de phase 2D par module de grid cell —
    jamais un scalaire ou une moyenne.

    Args:
        sdr_dim:        dimension du SDR d'entrée (= SpatialPooler.n_columns)
        n_grid_modules: nombre de modules de grid cells
        hidden_dim:     dimension de la couche cachée de transformation
    """

    def __init__(
        self,
        sdr_dim: int,
        n_grid_modules: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.sdr_dim = sdr_dim
        self.n_grid_modules = n_grid_modules
        self.phase_dim = 2 * n_grid_modules   # vecteur de phase 2D par module
        self.hidden_dim = hidden_dim

        # ── Encodage SDR → espace de phases ─────────────────────────────
        # Sortie : n_grid_modules vecteurs 2D (cos θ, sin θ)
        self.encoder = nn.Sequential(
            nn.Linear(sdr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.phase_dim),
        )

        # ── Rotation ego → allo guidée par la vitesse ────────────────────
        # La vitesse est encodée comme un angle de rotation dans 𝕋²
        # W_rot : (n_grid_modules × 2, 2) — rotation 2D par module
        self.W_rot = nn.Parameter(
            torch.eye(2).unsqueeze(0).expand(n_grid_modules, -1, -1).clone()
        )
        self.velocity_encoder = nn.Linear(2, n_grid_modules, bias=False)

        # ── Boucle thalamique ─────────────────────────────────────────────
        self.thalamic_loop = ThalamicLoop(dim=self.phase_dim)

        # ── Projection sur la sphère de norme constante (I3.1) ───────────
        # La sortie est renormalisée pour conserver la norme du latent
        # avant correction thalamique.
        self.norm_eps = 1e-8

    def _rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Construit des matrices de rotation 2D à partir d'angles.

        Args:
            theta: angles de rotation, shape (n_grid_modules,)

        Returns:
            R: matrices de rotation 2D, shape (n_grid_modules, 2, 2)
        """
        cos_t = torch.cos(theta)  # (n_modules,)
        sin_t = torch.sin(theta)  # (n_modules,)
        R = torch.stack(
            [
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t, cos_t], dim=-1),
            ],
            dim=-2,
        )  # (n_modules, 2, 2)
        return R

    def transform(
        self,
        active_sdr: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transforme la représentation SDR ego en vecteur de phase allocentrique.

        Invariant I3.1 : la transformation est une isométrie.
        Sortie : vecteur de phase 2D par module (dim = 2 × n_grid_modules).

        Réf. math : §3.2, Éq. 3.4.

        Args:
            active_sdr: SDR binaire ou indices actifs sous forme dense,
                        shape (sdr_dim,)
            velocity:   vecteur de vitesse egocentrique, shape (2,)

        Returns:
            allo_phase: vecteur allocentrique, shape (2 × n_grid_modules,)
        """
        # Encodage SDR → espace de phases ego
        z_ego = self.encoder(active_sdr.float())  # (phase_dim,)
        z_ego = z_ego.reshape(self.n_grid_modules, 2)  # (n_mod, 2)

        # Calcul des angles de rotation induits par la vitesse
        theta = self.velocity_encoder(velocity.float())  # (n_modules,)
        R = self._rotation_matrix(theta)  # (n_modules, 2, 2)

        # Rotation ego → allo : z_allo_k = R_k · z_ego_k
        z_allo = torch.einsum("mij,mj->mi", R, z_ego)  # (n_modules, 2)
        z_allo_flat = z_allo.reshape(self.phase_dim)    # (phase_dim,)

        # Boucle thalamique : une seule mise à jour (pas de double intégration)
        feedback = self.thalamic_loop(z_allo_flat)  # (phase_dim,)

        # Intégration du feedback thalamique
        z_allo_flat = z_allo_flat + 0.1 * feedback   # gain thalamique modéré

        # Renormalisation : la norme finale doit rester celle du latent
        # après rotation, qui est identique à celle de l'encodage ego.
        target_norm = z_ego.norm().item()
        current_norm = z_allo_flat.norm().item()
        if current_norm > self.norm_eps and target_norm > self.norm_eps:
            z_allo_flat = z_allo_flat * (target_norm / current_norm)
        elif target_norm <= self.norm_eps:
            z_allo_flat = torch.zeros_like(z_allo_flat)

        return z_allo_flat  # (2 × n_grid_modules,) — JAMAIS .mean()

    def forward(
        self,
        active_sdr: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Alias de transform() pour compatibilité nn.Module."""
        return self.transform(active_sdr, velocity)

    def reset_thalamic_state(self) -> None:
        """Réinitialise la boucle thalamique (nouveau stimulus/épisode)."""
        self.thalamic_loop.reset()

    def extra_repr(self) -> str:
        return (
            f"sdr_dim={self.sdr_dim}, n_grid_modules={self.n_grid_modules}, "
            f"phase_dim={self.phase_dim}, hidden_dim={self.hidden_dim}"
        )
