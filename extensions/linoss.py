"""
Extension — LinOSS (Linear Oscillatory State Space) — Rusch 2024
ODE stable pour la dynamique neuronale.

Motivations :
    Remplace les RNN classiques (gradient vanishing/exploding) par un
    système d'oscillateurs linéaires stables. Garantit la stabilité par
    construction via des contraintes sur le spectre de l'opérateur.

Architecture :
    dx/dt = A·x + B·u    (système d'état linéaire)
    y     = C·x + D·u    (sortie)
    où A est paramétrisé pour garantir Re(λ(A)) < 0 (stabilité asymptotique)

Discrétisation :
    Utilise le schéma de Bilinear (Tustin) pour la discrétisation exacte.

Réf. : Rusch 2024 (LinOSS) — Formalisation_Mille_Cerveaux.pdf §ext.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinOSSLayer(nn.Module):
    """
    Couche d'oscillateurs linéaires stables (Rusch 2024).

    La matrice A est paramétrisée comme A = −Λ + iΩ en forme diagonale
    complexe, garantissant la stabilité (Re(λ) = −Λ < 0).

    Args:
        input_dim:  dimension de l'entrée u
        state_dim:  dimension de l'espace d'état x (doit être pair pour
                    représentation complexe paires re/im)
        output_dim: dimension de la sortie y
        dt:         pas de temps de discrétisation
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        dt: float = 1.0,
    ) -> None:
        super().__init__()

        assert state_dim % 2 == 0, "state_dim doit être pair (paires oscillateurs)"
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.dt = dt
        self.n_oscillators = state_dim // 2

        # ── Paramètres de l'opérateur A (paires complexes) ───────────────
        # Λ : partie réelle négative (amortissement) — log_lambda pour garantir > 0
        self.log_lambda = nn.Parameter(torch.zeros(self.n_oscillators))
        # Ω : partie imaginaire (fréquence d'oscillation)
        self.omega = nn.Parameter(torch.randn(self.n_oscillators) * 0.1)

        # ── Matrices d'entrée/sortie ──────────────────────────────────────
        self.B_re = nn.Parameter(torch.randn(self.n_oscillators, input_dim) * 0.01)
        self.B_im = nn.Parameter(torch.randn(self.n_oscillators, input_dim) * 0.01)
        self.C = nn.Linear(state_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=False)

        # ── État interne ──────────────────────────────────────────────────
        self.register_buffer("x_re", torch.zeros(self.n_oscillators))
        self.register_buffer("x_im", torch.zeros(self.n_oscillators))

    def _compute_A_discrete(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule la matrice de transition discrète via Bilinear (Tustin).

        Pour un système A = diag(−Λ + iΩ), la discrétisation bilinéaire donne :
            A_d = (I + dt/2 · A) · (I − dt/2 · A)⁻¹

        Pour des oscillateurs 2D découplés :
            A_d_k = [a_rr, -a_ri; a_ri, a_rr] avec
            a_rr = (1 − dt·Λ/2) / ((1+dt·Λ/2)² + (dt·Ω/2)²)

        Returns:
            A_rr: partie diagonale réelle, shape (n_oscillators,)
            A_ri: partie hors-diagonale, shape (n_oscillators,)
        """
        lam = torch.exp(self.log_lambda)  # Λ > 0 garanti
        omega = self.omega
        h = self.dt

        denom = (1 + h * lam / 2) ** 2 + (h * omega / 2) ** 2
        A_rr = (1 - (h * lam / 2) ** 2 - (h * omega / 2) ** 2) / denom
        A_ri = h * omega / denom

        return A_rr, A_ri

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Effectue un pas de l'ODE linéaire oscillatoire.

        Args:
            u: entrée, shape (input_dim,) ou (batch, input_dim)

        Returns:
            y: sortie, shape (output_dim,) ou (batch, output_dim)
        """
        batched = u.dim() == 2
        if not batched:
            u = u.unsqueeze(0)
        batch_size = u.shape[0]

        A_rr, A_ri = self._compute_A_discrete()

        # Expansion pour le batch
        x_re = self.x_re.unsqueeze(0).expand(batch_size, -1)  # (B, n_osc)
        x_im = self.x_im.unsqueeze(0).expand(batch_size, -1)

        # Calcul de Bu : (B, n_osc)
        Bu_re = u @ self.B_re.T
        Bu_im = u @ self.B_im.T

        # Intégration bilinéaire
        x_re_new = A_rr * x_re - A_ri * x_im + Bu_re
        x_im_new = A_ri * x_re + A_rr * x_im + Bu_im

        # Mise à jour de l'état (premier élément du batch pour le buffer)
        self.x_re.data = x_re_new[0].detach()
        self.x_im.data = x_im_new[0].detach()

        # Concaténation re/im pour la sortie
        x_concat = torch.cat([x_re_new, x_im_new], dim=-1)  # (B, state_dim)

        # Sortie linéaire
        y = self.C(x_concat) + self.D(u)

        if not batched:
            y = y.squeeze(0)
        return y

    def reset(self) -> None:
        """Réinitialise l'état de l'ODE."""
        self.x_re.data.zero_()
        self.x_im.data.zero_()

    def stability_check(self) -> bool:
        """
        Vérifie que le spectre de A satisfait la condition de stabilité.

        Tous les eigenvalues doivent avoir une partie réelle < 0.

        Returns:
            True si le système est stable
        """
        lam = torch.exp(self.log_lambda)
        return (lam > 0).all().item()

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, state_dim={self.state_dim}, "
            f"output_dim={self.output_dim}, n_oscillators={self.n_oscillators}, "
            f"dt={self.dt}"
        )
