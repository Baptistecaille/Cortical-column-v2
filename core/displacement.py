"""
Module 5 — DisplacementAlgebra
Représentation relationnelle ⟨SDR_parent, SDR_subobject, D_subobject⟩.

Invariants :
    I5.1 : D = φ_objet − φ_capteur (soustraction modulaire sur 𝕋²)
    I5.2 : Compositionnalité — D_A→C = D_A→B ⊕ D_B→C

Usage :
    Encoder les relations spatiales objet-partie (ex. tasse/logo).
    Benchmark direct : CLEVR tasks (relations gauche/droite/devant/derrière).

Réf. math : §5, Déf. 5.1 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
import math
from typing import NamedTuple, Optional


class DisplacementTriplet(NamedTuple):
    """
    Triplet représentant une relation spatiale objet-partie.

    Champs :
        sdr_parent:    SDR de l'objet parent (contexte),  shape (n,)
        sdr_subobject: SDR du sous-objet (partie),        shape (n,)
        displacement:  déplacement D = φ_objet − φ_capteur sur 𝕋²,
                       shape (n_modules, 2)
    """

    sdr_parent: torch.Tensor
    sdr_subobject: torch.Tensor
    displacement: torch.Tensor


class DisplacementAlgebra(nn.Module):
    """
    Algèbre de déplacement sur le tore 𝕋².

    Calcule et compose les déplacements entre référentiels :
        D = φ_objet − φ_capteur  (soustraction modulaire, I5.1)
        D_A→C = D_A→B ⊕ D_B→C  (compositionnalité, I5.2)

    Args:
        n_modules: nombre de modules de grid cells (doit correspondre
                   à GridCellNetwork.n_modules)
    """

    def __init__(self, n_modules: int = 6) -> None:
        super().__init__()
        self.n_modules = n_modules

    def compute(
        self,
        phi_object: torch.Tensor,
        phi_sensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule le déplacement D = φ_objet − φ_capteur sur 𝕋².

        La soustraction modulaire garantit D ∈ (−π, π]² par module.

        Invariant I5.1 : D = φ_objet − φ_capteur (mod 2π, centré).

        Réf. math : §5.2, Éq. 5.1.

        Args:
            phi_object: phase allocentrique de l'objet,   shape (n_modules, 2)
            phi_sensor: phase allocentrique du capteur,   shape (n_modules, 2)

        Returns:
            D: déplacement modulaire, shape (n_modules, 2), D ∈ (−π, π]²
        """
        # Soustraction modulaire sur 𝕋² : résultat centré dans (−π, π]
        D = (phi_object - phi_sensor + math.pi) % (2 * math.pi) - math.pi
        return D  # (n_modules, 2)

    def compose(
        self,
        D_ab: torch.Tensor,
        D_bc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose deux déplacements : D_A→C = D_A→B ⊕ D_B→C.

        L'opération ⊕ est l'addition modulaire sur 𝕋².

        Invariant I5.2 : compositionnalité des déplacements.

        Réf. math : §5.3, Éq. 5.3.

        Args:
            D_ab: déplacement A→B, shape (n_modules, 2)
            D_bc: déplacement B→C, shape (n_modules, 2)

        Returns:
            D_ac: déplacement A→C, shape (n_modules, 2), D ∈ (−π, π]²
        """
        # Addition modulaire centrée dans (−π, π]
        D_ac = (D_ab + D_bc + math.pi) % (2 * math.pi) - math.pi
        return D_ac

    def invert(self, D: torch.Tensor) -> torch.Tensor:
        """
        Calcule le déplacement inverse : D_inv = −D (mod 2π).

        Réf. math : §5.3, Éq. 5.4.

        Args:
            D: déplacement, shape (n_modules, 2)

        Returns:
            D_inv: déplacement inverse, shape (n_modules, 2)
        """
        D_inv = (-D + math.pi) % (2 * math.pi) - math.pi
        return D_inv

    def make_triplet(
        self,
        sdr_parent: torch.Tensor,
        sdr_subobject: torch.Tensor,
        phi_object: torch.Tensor,
        phi_sensor: torch.Tensor,
    ) -> DisplacementTriplet:
        """
        Construit un triplet relationnel complet.

        Réf. math : §5.4, Déf. 5.2.

        Args:
            sdr_parent:    SDR de l'objet parent,   shape (n,)
            sdr_subobject: SDR du sous-objet,        shape (n,)
            phi_object:    phase de l'objet,         shape (n_modules, 2)
            phi_sensor:    phase du capteur,         shape (n_modules, 2)

        Returns:
            triplet: DisplacementTriplet
        """
        D = self.compute(phi_object, phi_sensor)
        return DisplacementTriplet(
            sdr_parent=sdr_parent,
            sdr_subobject=sdr_subobject,
            displacement=D,
        )

    def distance(
        self,
        D1: torch.Tensor,
        D2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distance angulaire entre deux déplacements sur 𝕋².

        Utile pour la reconnaissance de la relation spatiale
        (benchmark CLEVR).

        Args:
            D1: déplacement 1, shape (n_modules, 2)
            D2: déplacement 2, shape (n_modules, 2)

        Returns:
            dist: distance scalaire (somme des distances angulaires)
        """
        # Distance angulaire sur le tore : |D1 − D2|_tore
        diff = (D1 - D2 + math.pi) % (2 * math.pi) - math.pi
        dist = diff.abs().sum()
        return dist

    def forward(
        self,
        phi_object: torch.Tensor,
        phi_sensor: torch.Tensor,
    ) -> torch.Tensor:
        """Alias de compute() pour compatibilité nn.Module."""
        return self.compute(phi_object, phi_sensor)

    def extra_repr(self) -> str:
        return f"n_modules={self.n_modules}"
