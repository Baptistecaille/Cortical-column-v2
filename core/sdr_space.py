"""
Module 1 — SDRSpace
Encodeur sensoriel → Sparse Distributed Representation binaire.

Invariants :
    I1.1 : ||x||₁ = w exactement (parcimonie dure, pas soft)
    I1.2 : x ∈ {0,1}^n

Réf. math : §1, Déf. 1.1 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
from typing import Optional


class SDRSpace(nn.Module):
    """
    Encodeur sensoriel produisant un SDR binaire à parcimonie stricte.

    L'encodage repose sur un top-k strict suivi d'une binarisation —
    jamais un seuillage flottant (qui violerait I1.1).

    Args:
        input_dim:  dimension du vecteur sensoriel d'entrée
        n:          dimension du SDR de sortie (typiquement 2048)
        w:          nombre exact de bits actifs (typiquement 40, ~2%)
    """

    def __init__(self, input_dim: int, n: int = 2048, w: int = 40) -> None:
        super().__init__()
        assert w < n, f"w={w} doit être strictement inférieur à n={n}"
        assert w > 0, "w doit être > 0"

        self.input_dim = input_dim
        self.n = n
        self.w = w

        # Matrice de projection linéaire — état appris par autograd
        # W_enc : (n, input_dim)
        self.W_enc = nn.Linear(input_dim, n, bias=True)

        # Initialisation : valeurs positives (glutamatergique, Convention §5)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="relu")
            self.W_enc.weight.data = torch.abs(self.W_enc.weight.data)
            nn.init.zeros_(self.W_enc.bias)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """
        Encode un vecteur sensoriel en SDR binaire.

        Invariant I1.1 : ||x||₁ = w exactement.
        Invariant I1.2 : x ∈ {0,1}^n.

        Réf. math : §1.2, Éq. 1.1.

        Args:
            s: vecteur sensoriel, shape (input_dim,) ou (batch, input_dim)

        Returns:
            x: SDR binaire, shape (..., n), ||x||₁ = w par ligne
        """
        batched = s.dim() == 2
        if not batched:
            s = s.unsqueeze(0)

        # Projection linéaire
        logits = self.W_enc(s)  # (batch, n)

        # Top-k strict + binarisation — JAMAIS de seuillage flottant
        _, top_indices = torch.topk(logits, self.w, dim=-1, sorted=False)

        x = torch.zeros_like(logits)
        x.scatter_(-1, top_indices, 1.0)

        # Vérification invariant I1.1 (en mode debug)
        assert (x.sum(dim=-1) == self.w).all(), (
            f"Invariant I1.1 violé : ||x||₁ ≠ {self.w}"
        )

        if not batched:
            x = x.squeeze(0)

        return x  # {0,1}^n, ||·||₁ = w

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Alias de encode() pour compatibilité nn.Module."""
        return self.encode(s)

    @torch.no_grad()
    def pe_update(self, delta_W: torch.Tensor) -> None:
        """
        Applique une mise à jour PE-modulée à W_enc.

        Clamp ≥ 0 pour respecter la contrainte glutamatergique (§5, Bug B1) :
        les poids excitateurs ne peuvent pas devenir inhibiteurs.

        Réf. : Lee 2025, Éq. 6 ; §5 — Initialisation des poids.

        Args:
            delta_W: incrément de poids, shape (n, input_dim) = W_enc.weight.shape
        """
        self.W_enc.weight.data.add_(delta_W).clamp_(min=0.0)

    def sparsity(self) -> float:
        """Retourne le ratio de parcimonie w/n."""
        return self.w / self.n

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, n={self.n}, w={self.w}, sparsity={self.sparsity():.2%}"
