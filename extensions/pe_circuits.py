"""
Extension — PE Circuits (Prediction Error Circuits) — PRIORITÉ 1
Erreur de prédiction signée PE+/PE− (Lee 2025).

Cette extension est marquée PRIORITÉ 1 car elle change la nature du calcul :
au lieu d'une erreur non-signée scalaire, les deux populations PE+/PE−
permettent une correction de gradient biologique asymétrique.

Biologie :
    - PE+ (r_Ls_pos) : erreur positive — activation quand l'entrée
      dépasse la prédiction (neurones pyramidaux L2/3 superficiels)
    - PE− (r_Ls_neg) : erreur négative — activation quand la prédiction
      dépasse l'entrée (interneurones SST/PV)

Connexion à W_enc (PRIORITÉ 1) :
    Le prédicteur apprend top-down : grid_code (position) → SDR attendu.
    L'erreur signée (PE+ − PE−) module la mise à jour de W_enc par
    la règle de Hebb asymétrique :
        ΔW_enc = η · calcium_gate(PE+ − PE−) ⊗ s_t
    W_enc clampé ≥ 0 (glutamatergique — §5, Bug B1).

Le prédicteur lui-même est mis à jour par la règle delta (sans autograd) :
    ΔW_pred = lr_pred · (sdr − predicted) ⊗ grid_code

Réf. : Lee 2025, §2 — Formalisation_Mille_Cerveaux.pdf §6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PECircuits(nn.Module):
    """
    Circuits d'erreur de prédiction signée (Lee 2025).

    Le prédicteur mappe context_dim → dim (typiquement grid_code → SDR)
    et est mis à jour par règle delta (@no_grad, sans autograd).

    PE+/PE− sont calculés et maintenues séparément — ne jamais les fusionner.

    Args:
        dim:         dimension de la représentation cible (n_sdr)
        context_dim: dimension du contexte top-down (4·n_modules pour grid_code).
                     Si None, utilise dim (contexte = même espace que cible).
        tau_pos:     constante de temps de la population PE+ (filtrage)
        tau_neg:     constante de temps de la population PE−
        theta_ca:    seuil calcique (Lee 2025 : 15.0 Hz). Bug B12 : ne pas
                     utiliser 0.15 → tanh(r/0.15) ≡ 1 pour toute erreur.
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        tau_pos: float = 10.0,
        tau_neg: float = 10.0,
        theta_ca: float = 15.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.context_dim = context_dim if context_dim is not None else dim
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.theta_ca = theta_ca

        # Prédicteur top-down : context → SDR attendu
        # Stocké comme nn.Linear pour compatibilité device/state_dict,
        # mais mis à jour exclusivement par règle delta (@no_grad).
        self.predictor = nn.Linear(self.context_dim, dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.predictor.weight)
            nn.init.zeros_(self.predictor.bias)

        # États des populations PE+/PE− (filtres passe-bas)
        self.register_buffer("r_Ls_pos", torch.zeros(dim))
        self.register_buffer("r_Ls_neg", torch.zeros(dim))

    # ── Prédiction et erreurs ─────────────────────────────────────────────

    @torch.no_grad()
    def compute_prediction_errors(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule PE+ et PE− depuis le prédicteur top-down.

        r_Ls_pos = ReLU(x − predicted)   # surplus sensoriel (LTP)
        r_Ls_neg = ReLU(predicted − x)   # déficit sensoriel (LTD)

        Réf. : Lee 2025, Éq. 3.

        Args:
            x:       SDR courant, shape (dim,)
            context: vecteur de contexte top-down, shape (context_dim,)

        Returns:
            (r_pos, r_neg): erreurs PE+/PE−, chacune shape (dim,)
        """
        predicted = self.predictor(context)
        error = x - predicted
        return F.relu(error), F.relu(-error)

    # ── Pas complet : prédiction + mise à jour prédicteur + états PE ─────

    @torch.no_grad()
    def step_with_update(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        lr_pred: float = 0.01,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Un pas complet des circuits PE (règle delta + filtre temporel).

        Ordre :
          1. Forward du prédicteur → erreur signée
          2. Mise à jour du prédicteur par règle delta (@no_grad)
          3. Mise à jour des états PE+/PE− par filtre passe-bas

        La règle delta (Widrow-Hoff) est la forme Hebbienne de l'apprentissage
        supervisé sans autograd :
            ΔW_pred = lr_pred · error ⊗ context
            Δb_pred = lr_pred · error

        Args:
            x:        SDR courant, shape (dim,)
            context:  contexte top-down (grid_code), shape (context_dim,)
            lr_pred:  taux d'apprentissage du prédicteur
            dt:       pas de temps

        Returns:
            (r_Ls_pos, r_Ls_neg): états PE filtés, chacun shape (dim,)
        """
        predicted = self.predictor(context)        # (dim,)
        error = x - predicted                       # (dim,) — erreur signée

        pe_pos = F.relu(error)                      # surplus  → LTP
        pe_neg = F.relu(-error)                     # déficit  → LTD

        # Mise à jour du prédicteur — règle delta (sans autograd)
        self.predictor.weight.data.add_(
            lr_pred * torch.outer(error, context)
        )
        self.predictor.bias.data.add_(lr_pred * error)

        # Filtre passe-bas sur les états PE+/PE−
        alpha_pos = dt / self.tau_pos
        alpha_neg = dt / self.tau_neg
        self.r_Ls_pos.data = (
            (1.0 - alpha_pos) * self.r_Ls_pos + alpha_pos * pe_pos
        )
        self.r_Ls_neg.data = (
            (1.0 - alpha_neg) * self.r_Ls_neg + alpha_neg * pe_neg
        )

        return self.r_Ls_pos, self.r_Ls_neg

    # ── Porte calcique ────────────────────────────────────────────────────

    @torch.no_grad()
    def calcium_gate(self, r: torch.Tensor) -> torch.Tensor:
        """
        Porte calcique sigmoidale bornant l'activation.

        Utilise theta_ca = 15.0 Hz (pas 0.15 — bug B12).
        Pour theta_ca = 0.15 : tanh(r/0.15) ≡ 1 pour toute erreur non-nulle.

        Réf. : Lee 2025, Éq. 5.

        Args:
            r: taux d'activité, shape (dim,)

        Returns:
            gated: taux modulé, shape (dim,)
        """
        return torch.tanh(r / self.theta_ca) * r

    # ── Mise à jour de W_enc (PE-modulée) ────────────────────────────────

    @torch.no_grad()
    def modulated_update(
        self,
        pre: torch.Tensor,
        learning_rate: float = 0.001,
    ) -> torch.Tensor:
        """
        Calcule ΔW pour W_enc via l'erreur PE signée.

        ΔW_enc = η · (calcium_gate(PE+) − calcium_gate(PE−)) ⊗ s_t

        La règle est asymétrique : PE+ potentialise les synapses actives
        (LTP biologique), PE− les déprime (LTD). L'asymétrie est préservée
        par les deux populations séparées — ne jamais les fusionner.

        Réf. : Lee 2025, Éq. 6 ; Formalisation_Mille_Cerveaux.pdf §6.3.

        Args:
            pre:           vecteur pré-synaptique (s_t brut), shape (input_dim,)
            learning_rate: η — taux d'apprentissage PE→W_enc

        Returns:
            delta_W: incrément, shape (dim, input_dim) = shape de W_enc.weight
        """
        error_net = (
            self.calcium_gate(self.r_Ls_pos)
            - self.calcium_gate(self.r_Ls_neg)
        )
        return learning_rate * torch.outer(error_net, pre)

    # ── Compatibilité (ancien step, sans mise à jour prédicteur) ─────────

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Met à jour les états PE sans toucher au prédicteur.
        Utiliser step_with_update() dans le pipeline d'entraînement.
        """
        pe_pos, pe_neg = self.compute_prediction_errors(x, context)
        alpha_pos = dt / self.tau_pos
        alpha_neg = dt / self.tau_neg
        self.r_Ls_pos.data = (
            (1.0 - alpha_pos) * self.r_Ls_pos + alpha_pos * pe_pos
        )
        self.r_Ls_neg.data = (
            (1.0 - alpha_neg) * self.r_Ls_neg + alpha_neg * pe_neg
        )
        return self.r_Ls_pos, self.r_Ls_neg

    # ── Utilitaires ───────────────────────────────────────────────────────

    @torch.no_grad()
    def reset(self) -> None:
        """Réinitialise les états PE+/PE− (nouvel épisode)."""
        self.r_Ls_pos.data.zero_()
        self.r_Ls_neg.data.zero_()

    def pe_magnitude(self) -> dict:
        """Statistiques des signaux PE courants (diagnostic)."""
        return {
            "pe_pos_mean": self.r_Ls_pos.mean().item(),
            "pe_neg_mean": self.r_Ls_neg.mean().item(),
            "pe_pos_max":  self.r_Ls_pos.max().item(),
            "pe_neg_max":  self.r_Ls_neg.max().item(),
            "pe_net_mean": (self.r_Ls_pos - self.r_Ls_neg).mean().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, context_dim={self.context_dim}, "
            f"tau_pos={self.tau_pos}, tau_neg={self.tau_neg}, "
            f"theta_ca={self.theta_ca}"
        )
