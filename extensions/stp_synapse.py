"""
Extension — iSTP (Improved Short-Term Plasticity) — Waitzmann 2024
Plasticité synaptique à court terme : PV→STD, SST→STF.

Biologie :
    - STD (Short-Term Depression) : PV→E synapses
        u = U constant (taux de libération fixe)
        x évolue : dx/dt = (1−x)/τ_d − u·x·δ(spike)
    - STF (Short-Term Facilitation) : SST→PC synapses
        u évolue : du/dt = (U−u)/τ_f + U·(1−u)·δ(spike)
        x évolue : dx/dt = (1−x)/τ_d − u·x·δ(spike)

Piège B6 : en mode STD, u = U constant (ne pas utiliser l'équation STF).

Réf. : Waitzmann 2024 (iSTP) — Formalisation_Mille_Cerveaux.pdf §ext.1
"""

import torch
import torch.nn as nn
from typing import Literal


class STPSynapse(nn.Module):
    """
    Synapse avec plasticité à court terme (Tsodyks-Markram modifié).

    Deux modes :
        - "STD" : dépression synaptique (PV→E) — u constant
        - "STF" : facilitation synaptique (SST→PC) — u évolue

    Args:
        n_synapses: nombre de synapses modélisées
        mode:       "STD" (dépression, PV) ou "STF" (facilitation, SST)
        U:          probabilité de libération au repos ∈ (0, 1)
        tau_d:      constante de récupération des ressources (ms)
        tau_f:      constante de facilitation (ms, STF uniquement)
        dt:         pas de temps (ms)
    """

    def __init__(
        self,
        n_synapses: int,
        mode: Literal["STD", "STF"] = "STD",
        U: float = 0.5,
        tau_d: float = 200.0,
        tau_f: float = 600.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__()

        assert mode in ("STD", "STF"), f"mode doit être 'STD' ou 'STF', pas '{mode}'"
        assert 0.0 < U < 1.0, f"U={U} doit être dans (0, 1)"

        self.n_synapses = n_synapses
        self.mode = mode
        self.U = U
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.dt = dt

        # ── Variables dynamiques ──────────────────────────────────────────
        # x : fraction de ressources disponibles ∈ [0, 1]
        self.register_buffer("x", torch.ones(n_synapses))

        if mode == "STD":
            # u = U constant en STD (bug B6 : ne pas faire évoluer u en STD)
            self.u_fixed = U
        else:
            # STF : u évolue, initialisation à U
            self.register_buffer("u", torch.full((n_synapses,), U))

    @torch.no_grad()
    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        """
        Calcule la sortie synaptique pour un vecteur de spikes.

        Sortie : efficacité synaptique = u · x · spike

        Args:
            spike: vecteur de spikes binaires ou taux, shape (n_synapses,)

        Returns:
            efficacy: efficacité synaptique, shape (n_synapses,)
        """
        if self.mode == "STD":
            return self._step_std(spike)
        else:
            return self._step_stf(spike)

    @torch.no_grad()
    def _step_std(self, spike: torch.Tensor) -> torch.Tensor:
        """
        Étape STD (dépression) — u = U constant (Piège B6).

        Dynamique :
            dx/dt = (1−x)/τ_d − u·x·spike
            u = U  (constant en STD)

        Réf. : Waitzmann 2024, Éq. 2.

        Args:
            spike: shape (n_synapses,)

        Returns:
            efficacy: u · x · spike, shape (n_synapses,)
        """
        u = self.U  # constant en STD — bug B6 : ne pas faire évoluer u

        # Calcul de l'efficacité avant mise à jour (ordre important)
        efficacy = u * self.x * spike

        # Mise à jour des ressources (Euler)
        dx = (1.0 - self.x) / self.tau_d - u * self.x * spike
        self.x.data = (self.x + self.dt * dx).clamp(0.0, 1.0)

        return efficacy

    @torch.no_grad()
    def _step_stf(self, spike: torch.Tensor) -> torch.Tensor:
        """
        Étape STF (facilitation) — u évolue.

        Dynamique :
            du/dt = (U−u)/τ_f + U·(1−u)·spike
            dx/dt = (1−x)/τ_d − u·x·spike

        Réf. : Waitzmann 2024, Éq. 3.

        Args:
            spike: shape (n_synapses,)

        Returns:
            efficacy: u · x · spike, shape (n_synapses,)
        """
        # Mise à jour de u (facilitation) avant le spike
        du = (self.U - self.u) / self.tau_f + self.U * (1.0 - self.u) * spike
        self.u.data = (self.u + self.dt * du).clamp(0.0, 1.0)

        # Calcul de l'efficacité
        efficacy = self.u * self.x * spike

        # Mise à jour des ressources
        dx = (1.0 - self.x) / self.tau_d - self.u * self.x * spike
        self.x.data = (self.x + self.dt * dx).clamp(0.0, 1.0)

        return efficacy

    @torch.no_grad()
    def reset(self) -> None:
        """Réinitialise les variables dynamiques (nouvel épisode)."""
        self.x.data.fill_(1.0)
        if self.mode == "STF":
            self.u.data.fill_(self.U)

    def extra_repr(self) -> str:
        return (
            f"n_synapses={self.n_synapses}, mode={self.mode}, "
            f"U={self.U}, τ_d={self.tau_d}, τ_f={self.tau_f}"
        )
