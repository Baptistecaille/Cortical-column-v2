"""
Module 2 — SpatialPooler
Apprentissage hebbien non-supervisé des permanences synaptiques (L2/3).

Invariants :
    I2.1 : Exactement k colonnes actives à chaque pas
    I2.2 : p_ij ∈ [0,1] (permanences bornées)
    I2.3 : Convergence vers {0,1} (Thm 2.2)

Annealing (§7.2) :
    γ(t) schedule linéaire par morceaux avec plancher :
        - Newborn stage : t < m         → γ = 1.0
        - Décroissance  : m ≤ t < m+τ   → γ = gamma_floor + (1 - gamma_floor) · (1 - (t-m)/τ)
        - Mature        : t ≥ m+τ        → γ = gamma_floor  (jamais en dessous)
    Effets :
        - Boost    : b_i*(t) = 1 + γ(t)·(b_i − 1)
        - Dépression : δ⁻(t) = δ_floor + (δ⁻ − δ_floor)·γ(t)
    Pourquoi un plancher γ=0.2 :
        Thousand Brains — l'apprentissage hebbien reste actif en continu.
        Sans plancher, γ→0 gèle p̄ et empêche la correction de dérive.

GPU :
    Toutes les opérations critiques (forward_batch, hebbian_update_batch)
    sont vectorisées — zéro boucle Python, compatibles MPS/CUDA.
    L'initialisation du potential pool utilise argsort au lieu d'une boucle.

Règle absolue : aucun autograd — tout @torch.no_grad() ou .data
Réf. math : §2, Déf. 2.1, Thm 2.2 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
import math
import warnings


class SpatialPooler(nn.Module):
    """
    Sélecteur homeöstatique de k colonnes actives avec apprentissage hebbien.

    Les permanences synaptiques sont stockées comme buffer (pas Parameter)
    et mises à jour exclusivement par la règle hebbienne sans autograd.

    Interface simple  : forward(x)               → active (k,)
                        hebbian_update(x, active) → None
    Interface batched : forward_batch(x_batch)   → active_batch (B, k)
                        hebbian_update_batch(...)  → None  [GPU-optimisé]

    Args:
        n_inputs:       dimension du SDR d'entrée (= SDRSpace.n)
        n_columns:      nombre de colonnes minicolumnes (neurones L2/3)
        k:              nombre exact de colonnes actives par pas (I2.1)
        delta_plus:          incrément hebbien (potentiation)
        delta_minus:         décrément hebbien à γ=1. Ratio δ⁺/δ⁻ ≈ 21 pour
                             équilibre avec potential_pct=0.5, s=0.0195.
        delta_minus_floor:   plancher de dépression à γ=gamma_floor.
                             δ⁻(t) = δ_floor + (δ⁻ − δ_floor)·γ(t)
        gamma_floor:         plancher du schedule γ — γ ne descend jamais
                             en dessous (défaut 0.2, plasticité résiduelle).
        connected_perm:      seuil de connexion synaptique (défaut 0.20)
        potential_pct:  fraction des inputs dans le potential pool de chaque
                        colonne (0.75 par défaut). Résout le déséquilibre
                        δ+/δ− : sans pool, 98% des bits reçoivent δ− et les
                        permanences s'effondrent.
        newborn_steps:  durée du newborn stage m (annealing, en steps)
        tau_decay:      durée de la décroissance γ (en steps).
                        Pour couvrir N epochs : tau = N_steps_total - newborn_steps.
                        Ex : 3 epochs × 2000 img × 5 vues → tau = 30000 - 1000 = 29000.
        sigma:          largeur du voisinage homeöstatique (grille 4×4 → 0.8)
        beta:           force du boost homeöstatique
    """

    def __init__(
        self,
        n_inputs: int,
        n_columns: int,
        k: int = 40,
        delta_plus: float = 0.015,
        delta_minus: float = 0.000134,
        delta_minus_floor: float = 0.0000268,
        gamma_floor: float = 0.2,
        connected_perm: float = 0.20,
        potential_pct: float = 0.5,
        newborn_steps: int = 1000,
        tau_decay: int = 5000,
        sigma: float = 0.8,
        beta: float = 3.0,
    ) -> None:
        super().__init__()

        self.n_inputs = n_inputs
        self.n_columns = n_columns
        self.k = k
        self.delta_plus = delta_plus
        self.delta_minus = delta_minus
        self.delta_minus_floor = delta_minus_floor
        self.gamma_floor = gamma_floor
        self.connected_perm = connected_perm
        self.potential_pct = potential_pct
        self.newborn_steps = newborn_steps
        self.tau_decay = tau_decay
        self.sigma = sigma
        self.beta = beta

        # ── Potential pool (vectorisé, sans boucle Python) ────────────────
        # Chaque colonne n'a des synapses candidates que sur un sous-ensemble
        # de potential_pct × n_inputs bits. Résout l'effondrement des
        # permanences dû au déséquilibre δ+/δ− avec des SDRs creux.
        n_potential = max(1, int(potential_pct * n_inputs))
        # argsort d'un bruit aléatoire : permutation indépendante par ligne
        noise = torch.rand(n_columns, n_inputs)
        top_idx = torch.argsort(noise, dim=1)[:, :n_potential]   # (C, n_potential)
        potential_mask = torch.zeros(n_columns, n_inputs, dtype=torch.bool)
        potential_mask.scatter_(1, top_idx, True)
        self.register_buffer("potential_mask", potential_mask)
        # Masque float précompilé pour les matmuls (évite le cast répété)
        self.register_buffer("potential_mask_f", potential_mask.float())

        # ── Permanences synaptiques ──────────────────────────────────────
        # p_ij ∈ [0,1], shape (n_columns, n_inputs)
        # Uniforme(0.10, 0.30) centré sur connected_perm=0.20 :
        #   - 50% des synapses démarrent connectées (> 0.20)
        #   - 50% démarrent sub-threshold
        # Permet à la potentiation hebbienne de différencier les synapses
        # actives (qui franchissent 0.20) des inactives (qui restent dessous).
        # Bug résolu : uniform(0.20, 0.24) → tout connecté dès t=0 → p_conn=1.0
        permanences = torch.zeros(n_columns, n_inputs)
        permanences[potential_mask] = torch.empty(
            potential_mask.sum().item()
        ).uniform_(0.10, 0.30)
        self.register_buffer("permanences", permanences)

        # ── Vérification de l'équilibre δ+/δ− ────────────────────────────
        n_pool = int(potential_pct * n_inputs)
        w_effective = max(1, int(k * n_inputs / n_columns))
        equilibrium_ratio = (n_pool - w_effective) / max(w_effective, 1)
        actual_ratio = delta_plus / max(delta_minus, 1e-9)
        if actual_ratio < equilibrium_ratio * 0.5:
            warnings.warn(
                f"SpatialPooler : déséquilibre δ+/δ− probable. "
                f"Ratio actuel δ+/δ−={actual_ratio:.1f}, équilibre requis ≈{equilibrium_ratio:.1f}. "
                f"Suggestion : delta_minus ≤ {delta_plus / equilibrium_ratio:.4f}",
                UserWarning,
                stacklevel=2,
            )

        # ── Duty cycles (EMA de l'activité par colonne) ──────────────────
        duty_cycle = torch.full((n_columns,), fill_value=k / n_columns)
        self.register_buffer("duty_cycle", duty_cycle)

        # ── Compteur de pas (buffer, pas Parameter) ───────────────────────
        self.register_buffer("t_step", torch.tensor(0, dtype=torch.long))

        # ── Topologie des colonnes pour boost homeöstatique ───────────────
        side = int(math.isqrt(n_columns))
        assert side * side == n_columns, (
            f"n_columns={n_columns} doit être un carré parfait pour la topologie 2D"
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(side, dtype=torch.float),
                torch.arange(side, dtype=torch.float),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(n_columns, 2)
        self.register_buffer("coords", coords)

        diff = coords.unsqueeze(0) - coords.unsqueeze(1)   # (C, C, 2)
        dist = diff.norm(dim=-1)                            # (C, C)
        self.register_buffer("col_dist", dist)

        # Poids gaussiens précompilés (ne changent pas)
        weights = torch.exp(-0.5 * (dist / sigma) ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        self.register_buffer("boost_weights", weights)

    # ── Annealing ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _gamma_tensor(self) -> torch.Tensor:
        """
        Schedule γ(t) linéaire par morceaux avec plancher gamma_floor.

        - t < m         : γ = 1.0
        - m ≤ t < m+τ   : γ = gamma_floor + (1 - gamma_floor) · (1 - (t-m)/τ)
        - t ≥ m+τ        : γ = gamma_floor  (plasticité résiduelle permanente)

        Jamais en dessous de gamma_floor (défaut 0.2).
        """
        t = self.t_step.to(dtype=torch.float32)
        m = t.new_tensor(float(self.newborn_steps))
        tau = t.new_tensor(float(self.tau_decay))
        tau_safe = torch.clamp(tau, min=1.0)
        floor = t.new_tensor(float(self.gamma_floor))

        # Fraction de décroissance dans [0, 1]
        frac = torch.clamp((t - m) / tau_safe, min=0.0, max=1.0)
        # γ décroît de 1.0 vers gamma_floor
        gamma_decay = floor + (1.0 - floor) * (1.0 - frac)

        return torch.where(
            t < m,
            t.new_tensor(1.0),
            gamma_decay,
        )

    @torch.no_grad()
    def gamma(self) -> float:
        """γ(t) ∈ [0,1], schedule linéaire par morceaux. Réf. §7.2."""
        return float(self._gamma_tensor().item())

    # ── Boost homeöstatique ───────────────────────────────────────────────

    @torch.no_grad()
    def _compute_boost(self) -> torch.Tensor:
        """
        Boost b_i*(t) = 1 + γ(t)·(exp(β·(μ_voisins − d_i)) − 1).

        Réf. math : §2.4, Éq. 2.7.

        Returns:
            boost: shape (n_columns,)
        """
        mu = self.boost_weights @ self.duty_cycle          # (C,)
        b_base = torch.exp(self.beta * (mu - self.duty_cycle))
        return 1.0 + self._gamma_tensor() * (b_base - 1.0)

    # ── Forward — interface simple (1 sample) ────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sélectionne les k colonnes actives. Invariant I2.1 : |A_t| = k.

        Args:
            x: SDR binaire, shape (n_inputs,)

        Returns:
            active: indices, shape (k,)
        """
        return self.forward_batch(x.unsqueeze(0)).squeeze(0)

    # ── Forward — interface batché (GPU-optimisé) ────────────────────────

    @torch.no_grad()
    def forward_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Sélectionne les k colonnes actives pour un batch de SDRs.

        Invariant I2.1 : |A_t| = k exactement par ligne.

        Implémentation : overlap = x_batch @ connected.T → (B, C)
        Zéro boucle Python — compatible MPS/CUDA.

        Réf. math : §2.3, Déf. 2.1.

        Args:
            x_batch: SDRs binaires, shape (B, n_inputs)

        Returns:
            active_batch: indices des k colonnes, shape (B, k)
        """
        # Matrice de connectivité : (C, n_inputs) — seuil + pool
        connected_f = (
            (self.permanences >= self.connected_perm) & self.potential_mask
        ).float()

        # Overlap boosté : (B, C)
        overlap = x_batch.float() @ connected_f.T
        boost = self._compute_boost()                     # (C,)
        boosted = overlap * boost.unsqueeze(0)            # (B, C)

        # Top-k par ligne — I2.1 garanti
        _, active_batch = torch.topk(boosted, self.k, dim=-1, sorted=False)
        return active_batch   # (B, k)

    # ── Hebbian — interface simple (1 sample) ────────────────────────────

    @torch.no_grad()
    def hebbian_update(self, x: torch.Tensor, active: torch.Tensor) -> None:
        """
        Met à jour les permanences hebbienement (1 sample).

        Délègue à hebbian_update_batch() pour cohérence.

        Args:
            x:      SDR binaire, shape (n_inputs,)
            active: indices actifs, shape (k,)
        """
        self.hebbian_update_batch(x.unsqueeze(0), active.unsqueeze(0))

    # ── Hebbian — interface batché (GPU-optimisé) ────────────────────────

    @torch.no_grad()
    def hebbian_update_batch(
        self,
        x_batch: torch.Tensor,
        active_batch: torch.Tensor,
    ) -> None:
        """
        Met à jour les permanences sur un batch — zéro boucle Python.

        Formulation matricielle (GPU-friendly) :
            active_oh[b,c] = 1  si colonne c active pour l'image b
            n_active_cj   = active_oh.T @ x_batch     (C, n)
            n_active_c    = active_oh.sum(0)           (C,)
            Δp[c,j]       = n_active_cj·δ+ − (n_c − n_cj)·δ−(t)
            Δp masqué par potential_mask, puis clamp [0,1]

        Invariant I2.2 : p_ij ∈ [0,1] après clamp.
        Réf. math : §2.5, Éq. 2.9.

        Args:
            x_batch:      SDRs binaires, shape (B, n_inputs)
            active_batch: indices actifs, shape (B, k)
        """
        B = x_batch.shape[0]
        g = self._gamma_tensor()
        # δ⁻(t) = δ_floor + (δ⁻ − δ_floor) · γ(t)
        # Plancher non nul pour éviter la dérive de p̄ quand γ→0 (§7.2).
        # Sans ce plancher, la dépression s'annule et les permanences dérivent
        # vers 1.0 car la potentiation continue sans contrepoids.
        delta_minus_t = self.delta_minus_floor + (
            self.delta_minus - self.delta_minus_floor
        ) * g

        # One-hot des colonnes actives : (B, C)
        active_oh = torch.zeros(
            B, self.n_columns,
            dtype=torch.float,
            device=x_batch.device,
        )
        active_oh.scatter_(1, active_batch, 1.0)

        # Accumulation des co-activations : (C, n_inputs)
        # n_active_cj[c,j] = nombre de fois où col c active ET bit j = 1
        n_active_cj = active_oh.T @ x_batch.float()      # matmul GPU-friendly

        # Nombre total d'activations par colonne : (C,)
        n_active_c = active_oh.sum(0)                     # (C,)

        # Δp = potentiation − dépression, masqué par le potential pool
        delta = (
            n_active_cj * self.delta_plus
            - (n_active_c.unsqueeze(1) - n_active_cj) * delta_minus_t
        ) * self.potential_mask_f                         # (C, n_inputs)

        # Mise à jour et clamp I2.2
        self.permanences.data.add_(delta).clamp_(0.0, 1.0)

        # Duty cycle EMA (fenêtre 1000 pas per-sample)
        alpha = min(B / 1000.0, 0.5)
        mean_activity = active_oh.mean(0)                 # (C,)
        self.duty_cycle.data.mul_(1.0 - alpha).add_(alpha * mean_activity)

        # Incrément compteur de pas
        self.t_step.data += B

    # ── Hebbian ciblé — vote inter-colonnes (Phase 2) ───────────────────

    @torch.no_grad()
    def hebbian_update_targeted(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        vote_sdr: torch.Tensor,
        alpha_divergence: float = 3.0,
    ) -> None:
        """
        Mise à jour hebbienne modulée par le vote inter-colonnes (Phase 2).

        Principe CMP (Monty/Thousand Brains) :
            Le vote agrégé des K colonnes sert de cible pour la colonne courante.
            Les bits actifs qui sont dans le vote sont potentialisés normalement.
            Les bits actifs qui divergent du vote sont déprimés plus fortement
            (facteur alpha_divergence), ce qui pousse les colonnes à converger.

        Règle :
            active ∩ vote  → Δp = +δ⁺               (accord — LTP normal)
            active - vote  → Δp = -α · δ⁻(t)         (désaccord — LTD renforcé)
            vote - active  → Δp = 0                   (le boost s'en charge)

        Invariant I6.1 respecté : les permanences restent propres à chaque
        colonne, seule la force de la dépression est modulée.
        Invariant I2.2 : p_ij ∈ [0,1] après clamp.

        Args:
            x:                SDR binaire, shape (n_inputs,)
            active:           indices actifs SP, shape (k,)
            vote_sdr:         SDR de vote agrégé, shape (n_inputs,) — binaire
            alpha_divergence: facteur multiplicatif de dépression sur les bits
                              divergents (défaut 3.0). Trop fort → collapse.
        """
        g = self._gamma_tensor()
        delta_minus_t = self.delta_minus_floor + (
            self.delta_minus - self.delta_minus_floor
        ) * g

        # One-hot de la colonne active : (n_columns,)
        active_oh = torch.zeros(
            self.n_columns, dtype=torch.float, device=x.device
        )
        active_oh.scatter_(0, active, 1.0)           # (n_columns,)

        # Masques accord / désaccord sur les bits d'entrée
        # active_dense[j] = 1 si le bit j est actif dans le SDR d'entrée
        active_dense = torch.zeros(self.n_inputs, dtype=torch.float, device=x.device)
        active_dense[active] = 1.0                   # reconstruction dense (k bits)

        vote_f = vote_sdr.float()                    # (n_inputs,) binaire

        # Pour chaque minicolonne c active, calculer les deltas différenciés.
        # n_active_cj[c,j] = active_oh[c] * x[j]  (co-activations)
        n_active_cj = torch.outer(active_oh, x.float())    # (n_columns, n_inputs)
        n_active_c  = active_oh                             # (n_columns,)

        # Δp standard (accord avec vote ou bits inactifs non-vote)
        delta_std = (
            n_active_cj * self.delta_plus
            - (n_active_c.unsqueeze(1) - n_active_cj) * delta_minus_t
        )

        # Pénalité supplémentaire sur les bits actifs qui divergent du vote :
        # diverge[j] = 1 si bit j actif dans x MAIS absent du vote
        diverge = (x.float() - vote_f).clamp(min=0.0)  # (n_inputs,) ∈ {0,1}
        # Appliquée uniquement aux minicolonnes actives
        delta_diverge = (
            -alpha_divergence * delta_minus_t
            * n_active_c.unsqueeze(1)                  # (n_columns, 1) — actives seult
            * diverge.unsqueeze(0)                     # (1, n_inputs)
        )

        delta = (delta_std + delta_diverge) * self.potential_mask_f
        self.permanences.data.add_(delta).clamp_(0.0, 1.0)

        # Duty cycle EMA
        alpha_ema = 1.0 / 1000.0
        self.duty_cycle.data.mul_(1.0 - alpha_ema).add_(alpha_ema * active_oh)

        # Incrément compteur
        self.t_step.data += 1

    # ── Utilitaires ───────────────────────────────────────────────────────

    def permanence_stats(self) -> dict:
        """Statistiques sur les permanences dans le potential pool."""
        p = self.permanences[self.potential_mask]
        return {
            "min": p.min().item(),
            "max": p.max().item(),
            "mean": p.mean().item(),
            "frac_connected": (p >= self.connected_perm).float().mean().item(),
            "frac_near_zero": (p < 0.1).float().mean().item(),
            "frac_near_one": (p > 0.9).float().mean().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, n_columns={self.n_columns}, k={self.k}, "
            f"potential_pct={self.potential_pct}, connected_perm={self.connected_perm}, "
            f"δ⁺={self.delta_plus}, δ⁻={self.delta_minus}, δ⁻_floor={self.delta_minus_floor}, "
            f"γ_floor={self.gamma_floor}, m={self.newborn_steps}, τ={self.tau_decay}"
        )
