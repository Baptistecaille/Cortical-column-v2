"""
Extension — Astrocyte (Mémoire associative) — Kozachkov 2025
Réseau de mémoire associative astrocytaire.

Biologie :
    Les astrocytes modulent la transmission synaptique via la libération de
    gliotransmetteurs (glutamate, D-sérine) — implémentés ici comme un
    réseau de Hopfield avec mise à jour astrocytaire.

Architecture :
    - Réseau de Hopfield classique (mémoire associative)
    - Modulation astrocytaire : ajustement lent du seuil d'activation
    - Interaction tripartite : pré-synaptique, post-synaptique, astrocyte

Réf. : Kozachkov 2025 — Formalisation_Mille_Cerveaux.pdf §ext.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AstrocyteMemory(nn.Module):
    """
    Réseau de mémoire associative à modulation astrocytaire.

    Le réseau de Hopfield stocke des patterns et les récupère par
    descente d'énergie. L'astrocyte ajuste dynamiquement les seuils
    d'activation selon l'historique d'activité synaptique.

    Args:
        dim:           dimension de l'espace de représentation
        n_patterns:    nombre maximum de patterns mémorisés
        tau_astro:     constante de temps de la dynamique astrocytaire
        alpha_astro:   gain de modulation astrocytaire
        beta_hopfield: inverse de la température (sharpness du recall)
    """

    def __init__(
        self,
        dim: int,
        n_patterns: int = 100,
        tau_astro: float = 50.0,
        alpha_astro: float = 0.1,
        beta_hopfield: float = 1.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_patterns = n_patterns
        self.tau_astro = tau_astro
        self.alpha_astro = alpha_astro
        self.beta_hopfield = beta_hopfield

        # ── Matrice de poids de Hopfield ──────────────────────────────────
        # W = somme des patterns mémorisés (Hebb rule)
        self.register_buffer("W", torch.zeros(dim, dim))

        # ── État astrocytaire (seuils adaptatifs) ─────────────────────────
        # θ_astro : seuil par dimension, modifié lentement par l'astrocyte
        self.register_buffer("theta_astro", torch.zeros(dim))

        # ── Compteur de patterns stockés ──────────────────────────────────
        self.register_buffer("n_stored", torch.tensor(0, dtype=torch.long))

        # ── Historique d'activité synaptique (pour l'astrocyte) ───────────
        self.register_buffer("activity_trace", torch.zeros(dim))

    @torch.no_grad()
    def store(self, pattern: torch.Tensor) -> None:
        """
        Mémorise un pattern via la règle de Hebb.

        W ← W + pattern ⊗ pattern / n_patterns

        Args:
            pattern: vecteur ∈ {−1, +1}^dim ou ∈ [0,1]^dim, shape (dim,)
        """
        if self.n_stored >= self.n_patterns:
            # Oubli du pattern le plus ancien (approx. : réduction du gain)
            self.W.data *= (self.n_patterns - 1) / self.n_patterns

        # Normalisation du pattern en {−1, +1}
        p = pattern.sign() if pattern.abs().max() > 0 else pattern
        self.W.data += torch.outer(p, p) / self.dim
        self.W.data.fill_diagonal_(0.0)  # Pas d'auto-connexions
        self.n_stored.data += 1

    @torch.no_grad()
    def recall(
        self,
        query: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Récupère le pattern mémorisé le plus proche par itération de Hopfield.

        Avec modulation astrocytaire : le seuil θ_astro modifie le bassin
        d'attraction pour favoriser les patterns récemment activés.

        Args:
            query:   vecteur d'entrée partiel ou bruité, shape (dim,)
            n_steps: nombre d'itérations de relaxation

        Returns:
            recalled: pattern récupéré, shape (dim,)
        """
        state = query.sign()

        for _ in range(n_steps):
            # Champ local avec modulation astrocytaire
            h = self.W @ state - self.theta_astro
            state = torch.tanh(self.beta_hopfield * h).sign()

        return state

    @torch.no_grad()
    def update_astrocyte(self, pre_activity: torch.Tensor, dt: float = 1.0) -> None:
        """
        Met à jour l'état astrocytaire selon l'activité pré-synaptique.

        Dynamique : dθ/dt = (−θ + α·trace) / τ_astro

        L'astrocyte abaisse le seuil pour les patterns fréquemment activés
        (facilitation de la récupération mémorielle).

        Args:
            pre_activity: activité pré-synaptique, shape (dim,)
            dt:           pas de temps
        """
        # Trace d'activité (EMA)
        self.activity_trace.data = (
            0.9 * self.activity_trace + 0.1 * pre_activity.abs()
        )

        # Dynamique astrocytaire
        d_theta = (-self.theta_astro + self.alpha_astro * self.activity_trace) / self.tau_astro
        self.theta_astro.data = self.theta_astro + dt * d_theta

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'énergie de Hopfield d'un état.

        E = −½ · sᵀ · W · s + θᵀ · s

        Doit décroître à chaque itération de recall().

        Args:
            state: état du réseau, shape (dim,)

        Returns:
            energy: scalaire
        """
        return -0.5 * (state @ self.W @ state) + self.theta_astro @ state

    def reset(self) -> None:
        """Efface la mémoire et réinitialise l'état astrocytaire."""
        self.W.data.zero_()
        self.theta_astro.data.zero_()
        self.activity_trace.data.zero_()
        self.n_stored.data.zero_()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_patterns={self.n_patterns}, "
            f"n_stored={self.n_stored.item()}, "
            f"tau_astro={self.tau_astro}"
        )
