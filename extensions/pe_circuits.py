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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING


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
        # Initialisation Xavier (pas zéro) : W=0 → predicted=0 → PE+=sdr,
        # PE-=0 toujours → signal d'erreur asymétrique dès t=0, mais le
        # filtre tau=10 lisse tout à ~0 → W_pred ne bouge jamais.
        # Xavier donne une prédiction initiale non nulle et variée → PE+/PE-
        # équilibrés → signal d'apprentissage effectif dès les premiers pas.
        self.predictor = nn.Linear(self.context_dim, dim, bias=True)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.predictor.weight)
            nn.init.zeros_(self.predictor.bias)

        # États des populations PE+/PE− (filtres passe-bas)
        self.register_buffer("r_Ls_pos", torch.zeros(dim))
        self.register_buffer("r_Ls_neg", torch.zeros(dim))

        # ── Populations interneurones (Nemati 2025) — poids FIXES ────────────
        n_pv = max(1, dim // 4)
        n_sv = max(1, dim // 8)
        self.n_pv = n_pv
        self.n_sv = n_sv

        self.register_buffer("r_pv1", torch.zeros(n_pv))
        self.register_buffer("r_pv2", torch.zeros(n_pv))
        self.register_buffer("r_som", torch.zeros(n_sv))
        self.register_buffer("r_vip", torch.zeros(n_sv))

        scale_pv = 1.0 / math.sqrt(dim)
        scale_sv = 1.0 / math.sqrt(dim)

        self.register_buffer("_W_pv1_ff",     torch.rand(n_pv, dim) * scale_pv)
        self.register_buffer("_W_pv2_td",     torch.rand(n_pv, dim) * scale_pv)
        self.register_buffer("_W_vip_td",     torch.rand(n_sv, dim) * scale_sv)
        self.register_buffer("_W_som_in",     torch.rand(n_sv, dim) * scale_sv)
        self.register_buffer("_W_vip_som",    torch.rand(n_sv, n_sv) * (1.0 / math.sqrt(max(1, n_sv))))
        self.register_buffer("_W_inh_pe_pos", torch.rand(dim, n_pv) * scale_pv)
        self.register_buffer("_W_inh_pe_neg", torch.rand(dim, n_pv) * scale_pv)
        self.register_buffer("_W_som_dend",   torch.rand(dim, n_sv) * scale_sv)

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

    @torch.no_grad()
    def compute_prediction_errors_with_interneurons(
        self,
        x: torch.Tensor,
        predicted: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule PE+ et PE− avec modulation par le circuit interneurones.

        Circuit (Nemati 2025 bioRxiv 2025.11.01.686040) :
            PV1 ← feedforward x        → supprime PE- soma
            PV2 ← top-down predicted   → supprime PE+ soma
            VIP ← top-down predicted   → supprime SOM (disinhibition)
            SOM ← x                    → supprime dendrite PE- (équilibre)
            PE+ = ReLU(x − predicted − W_inh_pe_pos @ r_pv2)
            PE− = ReLU(predicted − x  − W_inh_pe_neg @ r_pv1 − W_som_dend @ r_som)

        Args:
            x:         activité L4 (feedforward), shape (dim,)
            predicted: prédiction top-down,        shape (dim,)

        Returns:
            (pe_pos, pe_neg): erreurs signées, chacune shape (dim,)
        """
        r_pv1 = F.relu(self._W_pv1_ff @ x)
        r_pv2 = F.relu(self._W_pv2_td @ predicted)
        r_vip = F.relu(self._W_vip_td @ predicted)
        r_som = F.relu(self._W_som_in @ x - self._W_vip_som @ r_vip)

        pe_pos = F.relu(x - predicted - self._W_inh_pe_pos @ r_pv2)
        pe_neg = F.relu(
            predicted - x
            - self._W_inh_pe_neg @ r_pv1
            - self._W_som_dend  @ r_som
        )

        self.r_pv1.data.copy_(r_pv1)
        self.r_pv2.data.copy_(r_pv2)
        self.r_som.data.copy_(r_som)
        self.r_vip.data.copy_(r_vip)

        return pe_pos, pe_neg

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

        pe_pos, pe_neg = self.compute_prediction_errors_with_interneurons(
            x, predicted
        )

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

    # ── Versions batchées (B épisodes indépendants en parallèle) ─────────

    @torch.no_grad()
    def step_with_update_batch(
        self,
        x_batch: torch.Tensor,
        context_batch: torch.Tensor,
        r_pos_batch: torch.Tensor,
        r_neg_batch: torch.Tensor,
        lr_pred: float = 0.01,
        dt: float = 1.0,
    ) -> "Tuple[torch.Tensor, torch.Tensor]":
        """
        Pas PE pour B échantillons indépendants — états PE passés explicitement.

        Le prédicteur est mis à jour par la règle delta moyennée sur le batch
        (équivalent à traiter B images séquentiellement avec le même lr).
        Les états PE r_pos/r_neg sont retournés sans muter les buffers.

        Args:
            x_batch:       SDRs courants, shape (B, dim)
            context_batch: grid_codes top-down, shape (B, context_dim)
            r_pos_batch:   état PE+ courant, shape (B, dim)
            r_neg_batch:   état PE− courant, shape (B, dim)
            lr_pred:       taux d'apprentissage du prédicteur
            dt:            pas de temps

        Returns:
            new_r_pos: shape (B, dim)
            new_r_neg: shape (B, dim)
        """
        predicted = self.predictor(context_batch)            # (B, dim)
        error     = x_batch - predicted                       # (B, dim)
        # Circuit interneurones batché (B épisodes)
        r_pv1_b = F.relu(x_batch @ self._W_pv1_ff.T)         # (B, n_pv)
        r_pv2_b = F.relu(predicted @ self._W_pv2_td.T)       # (B, n_pv)
        r_vip_b = F.relu(predicted @ self._W_vip_td.T)       # (B, n_sv)
        r_som_b = F.relu(
            F.relu(x_batch @ self._W_som_in.T)
            - r_vip_b @ self._W_vip_som.T
        )                                                     # (B, n_sv)
        pe_pos = F.relu(
            x_batch - predicted - r_pv2_b @ self._W_inh_pe_pos.T
        )                                                     # (B, dim)
        pe_neg = F.relu(
            predicted - x_batch
            - r_pv1_b @ self._W_inh_pe_neg.T
            - r_som_b @ self._W_som_dend.T
        )                                                     # (B, dim)

        # Mise à jour du prédicteur — règle delta moyennée sur le batch
        # ΔW = lr · (1/B) · erreurᵀ @ contexte  (produit externe moyenné)
        B = x_batch.shape[0]
        self.predictor.weight.data.add_(
            lr_pred * (error.T @ context_batch) / B
        )
        self.predictor.bias.data.add_(lr_pred * error.mean(dim=0))

        # Filtre passe-bas sur les états PE+/PE− — sans mutation du buffer
        alpha_pos = dt / self.tau_pos
        alpha_neg = dt / self.tau_neg
        new_r_pos = (1.0 - alpha_pos) * r_pos_batch + alpha_pos * pe_pos
        new_r_neg = (1.0 - alpha_neg) * r_neg_batch + alpha_neg * pe_neg

        return new_r_pos, new_r_neg

    @torch.no_grad()
    def modulated_update_batch(
        self,
        pre_batch: torch.Tensor,
        r_pos_batch: torch.Tensor,
        r_neg_batch: torch.Tensor,
        learning_rate: float = 0.001,
    ) -> torch.Tensor:
        """
        ΔW_enc batché via l'erreur PE signée, moyennée sur B.

        ΔW = η · (1/B) · (Ca(PE+) − Ca(PE−))ᵀ @ pre_batch

        Réf. : Lee 2025, Éq. 6.

        Args:
            pre_batch:    vecteurs pré-synaptiques (s_t), shape (B, input_dim)
            r_pos_batch:  états PE+, shape (B, dim)
            r_neg_batch:  états PE−, shape (B, dim)
            learning_rate: η

        Returns:
            delta_W: shape (dim, input_dim) = shape de W_enc.weight
        """
        error_net = (
            self.calcium_gate(r_pos_batch)
            - self.calcium_gate(r_neg_batch)
        )                                                     # (B, dim)
        return learning_rate * (error_net.T @ pre_batch) / pre_batch.shape[0]

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
        self.r_pv1.data.zero_()
        self.r_pv2.data.zero_()
        self.r_som.data.zero_()
        self.r_vip.data.zero_()

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
            f"n_pv={self.n_pv}, n_sv={self.n_sv}, "
            f"tau_pos={self.tau_pos}, tau_neg={self.tau_neg}, "
            f"theta_ca={self.theta_ca}"
        )
