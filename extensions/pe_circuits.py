"""
Extension — PE Circuits (Prediction Error Circuits) — PRIORITÉ 1
Erreur de prédiction signée PE+/PE− (Lee 2025).

Cette extension est marquée PRIORITÉ 1 car elle change la nature du calcul :
au lieu d'une erreur non-signée scalaire, les deux populations PE+/PE−
permettent une correction de gradient biologique asymétrique.

Biologie :
    - PE+ (r_Ls_pos) : erreur de prédiction positive — activation quand
      l'entrée dépasse la prédiction (neurones pyramidaux L2/3 superficiels)
    - PE− (r_Ls_neg) : erreur de prédiction négative — activation quand
      la prédiction dépasse l'entrée (interneurones SST/PV)

Les deux populations sont SÉPARÉES — ne jamais les fusionner en une seule
variable r_Ls (cela éliminerait l'asymétrie du signal).

Réf. : Lee 2025, §2 — Formalisation_Mille_Cerveaux.pdf §6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PECircuits(nn.Module):
    """
    Circuits d'erreur de prédiction signée (Lee 2025).

    Calcule séparément :
        r_Ls_pos = ReLU(x − prediction)    # PE+ : surplus sensoriel
        r_Ls_neg = ReLU(prediction − x)    # PE− : déficit sensoriel

    Ces deux signaux pilotent des mises à jour asymétriques des
    poids synaptiques — analogues aux trajectoires LTP/LTD biologiques.

    Args:
        dim:           dimension de l'espace de représentation
        tau_pos:       constante de temps de la population PE+
        tau_neg:       constante de temps de la population PE−
        theta_ca:      seuil calcique d'activation (Lee 2025 : 15.0 Hz)
                       ATTENTION : ne pas utiliser 0.15 (→ tanh ≡ 1)
    """

    def __init__(
        self,
        dim: int,
        tau_pos: float = 10.0,
        tau_neg: float = 10.0,
        theta_ca: float = 15.0,    # Bug B12 : doit être 15.0, pas 0.15
    ) -> None:
        super().__init__()

        self.dim = dim
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.theta_ca = theta_ca

        # Réseaux de prédiction (top-down)
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        # États internes des populations PE+/PE−
        self.register_buffer("r_Ls_pos", torch.zeros(dim))
        self.register_buffer("r_Ls_neg", torch.zeros(dim))

    def compute_prediction_errors(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les erreurs de prédiction PE+ et PE− séparément.

        r_Ls_pos = ReLU(x − prediction)   # surplus sensoriel
        r_Ls_neg = ReLU(prediction − x)   # déficit sensoriel

        Les deux populations sont SÉPARÉES (PRIORITÉ 1 — ne pas fusionner).

        Réf. : Lee 2025, Éq. 3.

        Args:
            x:       représentation sensorielle courante, shape (dim,)
            context: vecteur de contexte top-down, shape (dim,)

        Returns:
            r_Ls_pos: erreur positive, shape (dim,)
            r_Ls_neg: erreur négative, shape (dim,)
        """
        # Prédiction top-down basée sur le contexte
        prediction = self.predictor(context)   # (dim,)

        # Erreurs signées asymétriques
        error = x - prediction

        r_pos = F.relu(error)    # PE+ : surplus (x > prediction)
        r_neg = F.relu(-error)   # PE− : déficit (prediction > x)

        return r_pos, r_neg

    def step(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Met à jour les états des populations PE+/PE− avec dynamique temporelle.

        Filtre passe-bas τ × dr/dt = −r + PE(x, pred) — discrétisé en Euler.

        Réf. : Lee 2025, Éq. 4.

        Args:
            x:       représentation sensorielle, shape (dim,)
            context: contexte top-down, shape (dim,)
            dt:      pas de temps

        Returns:
            r_Ls_pos: état PE+, shape (dim,)
            r_Ls_neg: état PE−, shape (dim,)
        """
        pe_pos, pe_neg = self.compute_prediction_errors(x, context)

        # Dynamique temporelle : filtre passe-bas
        alpha_pos = dt / self.tau_pos
        alpha_neg = dt / self.tau_neg

        self.r_Ls_pos.data = (1 - alpha_pos) * self.r_Ls_pos + alpha_pos * pe_pos
        self.r_Ls_neg.data = (1 - alpha_neg) * self.r_Ls_neg + alpha_neg * pe_neg

        return self.r_Ls_pos, self.r_Ls_neg

    def calcium_gate(self, r: torch.Tensor) -> torch.Tensor:
        """
        Porte calcique sigmoidale bornant l'activation.

        Utilise theta_ca = 15.0 Hz (pas 0.15 — bug B12 dans CLAUDE.md).
        Pour theta_ca = 0.15, tanh(r/0.15) ≡ 1 pour toute erreur non-nulle.

        Args:
            r: taux d'activité, shape (dim,)

        Returns:
            gated: taux pondéré par la porte calcique, shape (dim,)
        """
        return torch.tanh(r / self.theta_ca) * r

    def modulated_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """
        Mise à jour synaptique asymétrique guidée par PE+/PE−.

        ΔW = η · (PE+ − PE−) ⊗ pre    (analogie LTP − LTD)

        Args:
            weights:       matrice de poids, shape (dim_post, dim_pre)
            pre:           activation pré-synaptique, shape (dim_pre,)
            learning_rate: taux d'apprentissage η

        Returns:
            delta_W: incrément de poids, shape (dim_post, dim_pre)
        """
        # Erreur nette (asymétrique)
        error_net = self.calcium_gate(self.r_Ls_pos) - self.calcium_gate(self.r_Ls_neg)

        # Règle hebbienne signée : ΔW = η · error ⊗ pre
        delta_W = learning_rate * torch.outer(error_net, pre)
        return delta_W

    def reset(self) -> None:
        """Réinitialise les états des populations PE+/PE−."""
        self.r_Ls_pos.data.zero_()
        self.r_Ls_neg.data.zero_()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, tau_pos={self.tau_pos}, tau_neg={self.tau_neg}, "
            f"theta_ca={self.theta_ca}"
        )
