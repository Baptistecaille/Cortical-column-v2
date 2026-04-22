"""
Module 4 — GridCellNetwork
Intégration de chemin sur tore 2D 𝕋².

Invariants :
    I4.1 : φ_k ∈ [0, 2π)² pour tout module k (mise à jour mod 2π)
    I4.2 : get_code() produit [cos(φ₀), sin(φ₀), cos(φ₁), sin(φ₁)] par module
           → dimension totale 4·n_modules (pas 2·n_modules)
    I4.3 : Correction anchor() pondérée par 1/λ_k (pas uniforme)

Théorème CRT :
    Les périodes λ_k doivent être copremières pour une résolution sans ambiguïté
    sur ∏λ_k unités (§4.3, Thm 4.1).

Pourquoi le tore :
    La périodicité du signal de grid cell impose ℝ²/Λ ≅ 𝕋² — confirmé
    empiriquement par Gardner et al. 2022 (homologie persistante sur
    enregistrements de rat).

Réf. math : §4 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List
from math import gcd
from functools import reduce


def _are_coprime(periods: List[int]) -> bool:
    """
    Vérifie que les périodes sont copremières deux à deux (CRT).

    Args:
        periods: liste d'entiers représentant les périodes λ_k

    Returns:
        True si toutes les paires sont copremières
    """
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            if gcd(periods[i], periods[j]) != 1:
                return False
    return True


class GridCellNetwork(nn.Module):
    """
    Réseau de cellules de grille pour l'intégration de chemin sur 𝕋².

    Chaque module k maintient une phase 2D φ_k ∈ [0, 2π)²
    mise à jour par intégration de la vitesse allocentrique.

    Args:
        n_modules:       nombre de modules de grid cells
        periods:         liste d'entiers λ_k (périodes spatiales, copremières)
        phase_dim_in:    dimension du vecteur de phase allocentrique entrant
                         (= Layer6bTransformer.phase_dim = 2 × n_modules)
    """

    # Périodes canoniques copremières (suites de nombres premiers)
    DEFAULT_PERIODS = [3, 5, 7, 11, 13, 17, 19, 23]

    def __init__(
        self,
        n_modules: int = 6,
        periods: Optional[List[int]] = None,
        phase_dim_in: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.n_modules = n_modules

        # Périodes λ_k : copremières (Thm CRT)
        if periods is None:
            periods = self.DEFAULT_PERIODS[:n_modules]
        assert len(periods) == n_modules, (
            f"len(periods)={len(periods)} ≠ n_modules={n_modules}"
        )
        # Vérification coprimarité — piège CRT
        assert _are_coprime(periods), (
            f"Les périodes {periods} ne sont pas copremières — viole le CRT (§4.3)"
        )
        self.periods = periods

        # Phase λ_k comme buffer (float, en degrés d'arclength)
        lambda_k = torch.tensor(periods, dtype=torch.float)
        self.register_buffer("lambda_k", lambda_k)

        # ── Phases φ_k ∈ [0, 2π)² ────────────────────────────────────────
        # Shape : (n_modules, 2) — une phase 2D par module
        phases = torch.rand(n_modules, 2) * 2 * math.pi
        self.register_buffer("phases", phases)

        # ── Matrice d'intégration ─────────────────────────────────────────
        # Transforme le vecteur allo en incrément de phase par module
        phase_dim_in = phase_dim_in or (2 * n_modules)
        self.W_integrator = nn.Linear(phase_dim_in, n_modules * 2, bias=False)

        # Initialisation orthogonale pour stabilité
        nn.init.orthogonal_(self.W_integrator.weight)

    # ── Intégration de chemin ─────────────────────────────────────────────

    def integrate(
        self,
        allo_phase: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Intègre la vitesse allocentrique pour mettre à jour les phases.

        Pour chaque module k :
            φ_k ← (φ_k + R_k · v_allo · dt) mod 2π

        Invariant I4.1 : φ_k ∈ [0, 2π)² après mod 2π.

        Réf. math : §4.2, Éq. 4.3.

        Args:
            allo_phase: vecteur allocentrique de Layer6b, shape (2·n_modules,)
            dt:         pas de temps (typiquement 1.0)

        Returns:
            phases: phases mises à jour, shape (n_modules, 2)
        """
        # Calcul des incréments de phase via la matrice d'intégration
        delta_phi = self.W_integrator(allo_phase.float())   # (n_modules × 2,)
        delta_phi = delta_phi.reshape(self.n_modules, 2)    # (n_modules, 2)

        # Mise à jour sur le tore 𝕋² — mod 2π strict (I4.1)
        new_phases = (self.phases + dt * delta_phi) % (2 * math.pi)
        self.phases.data = new_phases

        return self.phases  # (n_modules, 2)

    # ── Code de position ──────────────────────────────────────────────────

    def get_code(self) -> torch.Tensor:
        """
        Retourne le code de position via cos/sin de chaque axe de phase.

        Pour chaque module k :
            code_k = [cos(φ_k,0), sin(φ_k,0), cos(φ_k,1), sin(φ_k,1)]

        Invariant I4.2 : dimension totale = 4 × n_modules (pas 2 × n_modules).

        Réf. math : §4.4, Éq. 4.6.

        Returns:
            code: vecteur de position, shape (4 × n_modules,)
        """
        # φ_k,0 et φ_k,1 : les deux axes de chaque module
        phi_0 = self.phases[:, 0]  # (n_modules,)
        phi_1 = self.phases[:, 1]  # (n_modules,)

        # 4 composantes par module — I4.2
        code = torch.stack(
            [
                torch.cos(phi_0),
                torch.sin(phi_0),
                torch.cos(phi_1),
                torch.sin(phi_1),
            ],
            dim=-1,
        ).reshape(4 * self.n_modules)  # (4·n_modules,)

        return code

    # ── Ancrage ───────────────────────────────────────────────────────────

    def anchor(
        self,
        landmark_phases: torch.Tensor,
        confidence: float = 1.0,
    ) -> None:
        """
        Corrige les phases par ancrage sur un repère visuel.

        La correction est pondérée par 1/λ_k — les modules à haute résolution
        (petite période) ont une correction plus forte.

        Invariant I4.3 : correction pondérée par 1/λ_k (pas uniforme).

        Réf. math : §4.5, Éq. 4.8.

        Args:
            landmark_phases: phases cibles de référence, shape (n_modules, 2)
            confidence:      confiance dans le repère (0=aucune, 1=totale)
        """
        if landmark_phases.shape != self.phases.shape:
            raise ValueError(
                f"landmark_phases.shape={landmark_phases.shape} ≠ "
                f"phases.shape={self.phases.shape}"
            )

        # Poids de correction : 1/λ_k (I4.3) — modules haute-résolution prioritaires
        weights = (1.0 / self.lambda_k).unsqueeze(-1)  # (n_modules, 1)
        weights = weights / weights.sum()               # normalisation

        # Erreur angulaire (sur le tore)
        delta = (landmark_phases - self.phases + math.pi) % (2 * math.pi) - math.pi

        # Correction pondérée et ancrée par la confiance
        correction = confidence * weights * delta
        corrected = (self.phases + correction) % (2 * math.pi)
        self.phases.data = corrected

    # ── Utilitaires ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Réinitialise les phases à des valeurs aléatoires (nouvel épisode)."""
        self.phases.data = torch.rand_like(self.phases) * 2 * math.pi

    def position_capacity(self) -> int:
        """
        Retourne la capacité de résolution de position (produit des périodes).

        Grâce au CRT, la position est unique sur ∏λ_k unités.
        """
        return reduce(lambda a, b: a * b, self.periods)

    def extra_repr(self) -> str:
        return (
            f"n_modules={self.n_modules}, periods={self.periods}, "
            f"capacity={self.position_capacity()}"
        )
