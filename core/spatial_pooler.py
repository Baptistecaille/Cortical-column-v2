"""
Module 2 — SpatialPooler
Apprentissage hebbien non-supervisé des permanences synaptiques (L2/3).

Invariants :
    I2.1 : Exactement k colonnes actives à chaque pas
    I2.2 : p_ij ∈ [0,1] (permanences bornées)
    I2.3 : Convergence vers {0,1} (Thm 2.2)

Annealing (§7.2) :
    γ(t) schedule linéaire par morceaux :
        - Newborn stage : t < m   → γ = 1.0
        - Décroissance  : t < m+τ → γ = 1 - (t-m)/τ
        - Mature        : t ≥ m+τ → γ = 0.0
    Effets :
        - Boost    : b_i*(t) = 1 + γ(t)·(b_i − 1)
        - Dépression : δ⁻(t) = δ⁻·γ(t)

Règle absolue : aucun autograd — tout @torch.no_grad() ou .data
Réf. math : §2, Déf. 2.1, Thm 2.2 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class SpatialPooler(nn.Module):
    """
    Sélecteur homeöstatique de k colonnes actives avec apprentissage hebbien.

    Les permanences synaptiques sont stockées comme buffer (pas Parameter)
    et mises à jour exclusivement par la règle hebbienne sans autograd.

    Args:
        n_inputs:       dimension du SDR d'entrée (= SDRSpace.n)
        n_columns:      nombre de colonnes minicolumnes (neurones L2/3)
        k:              nombre exact de colonnes actives par pas (I2.1)
        delta_plus:     incrément hebbien (potentiation)
        delta_minus:    décrément hebbien (dépression, annealed)
        connected_perm: seuil de connexion synaptique (typiquement 0.5)
        potential_pct:  fraction des inputs dans le potential pool de chaque
                        colonne (0.75 par défaut). Résout le déséquilibre
                        δ+/δ− : sans pool, 98% des bits reçoivent δ− et les
                        permanences s'effondrent.
        newborn_steps:  durée du newborn stage m (annealing)
        tau_decay:      durée de la décroissance γ
        sigma:          largeur du voisinage homeöstatique (grille 4×4 → 0.8)
        beta:           force du boost homeöstatique
    """

    def __init__(
        self,
        n_inputs: int,
        n_columns: int,
        k: int = 40,
        delta_plus: float = 0.05,
        delta_minus: float = 0.005,
        connected_perm: float = 0.5,
        potential_pct: float = 0.75,
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
        self.connected_perm = connected_perm
        self.potential_pct = potential_pct
        self.newborn_steps = newborn_steps
        self.tau_decay = tau_decay
        self.sigma = sigma
        self.beta = beta

        # ── Potential pool ────────────────────────────────────────────────
        # Masque booléen fixé à l'init : chaque colonne n'a des synapses
        # candidats que sur un sous-ensemble de potential_pct × n_inputs bits.
        # Sans pool, δ− s'applique à 98% des bits sur MNIST (w/n≈2%),
        # ce qui écrase les permanences vers 0.
        n_potential = max(1, int(potential_pct * n_inputs))
        potential_mask = torch.zeros(n_columns, n_inputs, dtype=torch.bool)
        for i in range(n_columns):
            idx = torch.randperm(n_inputs)[:n_potential]
            potential_mask[i, idx] = True
        self.register_buffer("potential_mask", potential_mask)

        # ── Permanences synaptiques ──────────────────────────────────────
        # p_ij ∈ [0,1], shape (n_columns, n_inputs)
        # Uniforme(0.3, 0.7) — jamais torch.randn (I2.2)
        # Les permanences hors potential pool restent à 0 (non utilisées).
        permanences = torch.zeros(n_columns, n_inputs)
        permanences[potential_mask] = torch.empty(
            potential_mask.sum().item()
        ).uniform_(0.3, 0.7)
        self.register_buffer("permanences", permanences)

        # ── Vérification de l'équilibre δ+/δ− ────────────────────────────
        # Condition d'équilibre : w_pool·δ+ ≈ (n_pool - w_pool)·δ−
        # où w_pool ≈ k (colonnes actives, approximation au premier ordre)
        # Si violée, les permanences s'effondrent vers 0.
        n_pool = int(potential_pct * n_inputs)
        w_effective = max(1, int(k * n_inputs / n_columns))  # bits actifs par colonne
        equilibrium_ratio = (n_pool - w_effective) / max(w_effective, 1)
        actual_ratio = delta_plus / max(delta_minus, 1e-9)
        if actual_ratio < equilibrium_ratio * 0.5:
            import warnings
            warnings.warn(
                f"SpatialPooler : déséquilibre δ+/δ− probable. "
                f"Ratio actuel δ+/δ−={actual_ratio:.1f}, équilibre requis ≈{equilibrium_ratio:.1f}. "
                f"Les permanences risquent de s'effondrer. "
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
        # Grille 2D : sqrt(n_columns) × sqrt(n_columns)
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
        ).reshape(n_columns, 2)  # (n_columns, 2)
        self.register_buffer("coords", coords)

        # Distance inter-colonnes précompilée : (n_columns, n_columns)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (C,C,2)
        dist = diff.norm(dim=-1)                          # (C,C)
        self.register_buffer("col_dist", dist)

    # ── Annealing ────────────────────────────────────────────────────────

    @torch.no_grad()
    def gamma(self) -> float:
        """
        Facteur d'annealing γ(t) ∈ [0,1], schedule linéaire par morceaux.

        Réf. math : §7.2, Éq. 7.3.

        Returns:
            γ(t) : float dans [0,1]
        """
        t = self.t_step.item()
        m = self.newborn_steps
        tau = self.tau_decay

        if t < m:
            return 1.0
        elif t < m + tau:
            return 1.0 - (t - m) / tau
        else:
            return 0.0

    # ── Boost homeöstatique ───────────────────────────────────────────────

    @torch.no_grad()
    def _compute_boost(self) -> torch.Tensor:
        """
        Calcule le facteur de boost b_i*(t) = 1 + γ(t)·(b_i − 1).

        b_i est le boost homeöstatique basé sur la densité locale :
            b_i = exp(β · (μ_voisins − d_i))
        où d_i est le duty cycle de la colonne i et μ_voisins est
        le duty cycle moyen du voisinage gaussien.

        Réf. math : §2.4, Éq. 2.7.

        Returns:
            boost: facteur de boost par colonne, shape (n_columns,)
        """
        g = self.gamma()

        # Poids gaussien des voisins : (n_columns, n_columns)
        weights = torch.exp(-0.5 * (self.col_dist / self.sigma) ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Duty cycle moyen des voisins
        mu_neighbors = weights @ self.duty_cycle  # (n_columns,)

        # Boost de base
        b_base = torch.exp(self.beta * (mu_neighbors - self.duty_cycle))

        # Boost annealed : b_i*(t) = 1 + γ(t)·(b_i − 1)
        boost = 1.0 + g * (b_base - 1.0)
        return boost

    # ── Forward ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sélectionne les k colonnes actives par overlap boosté.

        Invariant I2.1 : |A_t| = k exactement.

        Réf. math : §2.3, Déf. 2.1.

        Args:
            x: SDR binaire, shape (n_inputs,), ||x||₁ = w

        Returns:
            active: indices des k colonnes gagnantes, shape (k,)
        """
        # Overlap : synapses connectées (dans pool ET permanence > seuil) × SDR
        connected = (self.permanences >= self.connected_perm) & self.potential_mask
        overlap = connected.float() @ x  # (C,)

        # Boost homeöstatique
        boost = self._compute_boost()  # (C,)
        boosted_overlap = boost * overlap  # (C,)

        # Top-k strict — I2.1 garanti
        _, active = torch.topk(boosted_overlap, self.k, sorted=False)
        return active  # shape (k,)

    # ── Apprentissage hebbien ─────────────────────────────────────────────

    @torch.no_grad()
    def hebbian_update(self, x: torch.Tensor, active: torch.Tensor) -> None:
        """
        Met à jour les permanences synaptiques selon la règle hebbienne.

        Pour chaque colonne i ∈ A_t :
            p_ij += δ⁺          si x_j = 1  (potentiation)
            p_ij -= δ⁻·γ(t)     si x_j = 0  (dépression annealed)
        Puis : p_ij ← clamp(p_ij, 0, 1)     (I2.2)

        Invariant I2.2 : p_ij ∈ [0,1] après clamp.
        Réf. math : §2.5, Éq. 2.9.

        Args:
            x:      SDR binaire, shape (n_inputs,)
            active: indices des colonnes actives, shape (k,)
        """
        g = self.gamma()
        delta_minus_t = self.delta_minus * g  # dépression annealed

        # Masque des bits actifs / inactifs dans le SDR
        x_on = x.bool()   # (n_inputs,) — bits = 1
        x_off = ~x_on     # (n_inputs,) — bits = 0

        # Mise à jour uniquement pour les colonnes actives
        perm_active = self.permanences.data[active]        # (k, n_inputs)
        pool_active = self.potential_mask[active]           # (k, n_inputs)

        # Potentiation : bits SDR actifs ET dans le potential pool
        perm_active[pool_active & x_on.unsqueeze(0).expand_as(pool_active)] += self.delta_plus

        # Dépression : bits SDR inactifs ET dans le potential pool
        perm_active[pool_active & x_off.unsqueeze(0).expand_as(pool_active)] -= delta_minus_t

        # Clamp I2.2 — uniquement dans le pool (hors pool reste à 0)
        perm_active.clamp_(0.0, 1.0)
        self.permanences.data[active] = perm_active

        # Mise à jour du duty cycle EMA
        self._update_duty_cycle(active)

        # Incrément du compteur de pas
        self.t_step.data += 1

    @torch.no_grad()
    def _update_duty_cycle(self, active: torch.Tensor) -> None:
        """
        Met à jour le duty cycle EMA de chaque colonne.

        EMA avec fenêtre glissante de 1000 pas (§2.6).

        Args:
            active: indices des colonnes actives au pas courant, shape (k,)
        """
        alpha = 1.0 / 1000.0
        activity = torch.zeros(self.n_columns, device=self.duty_cycle.device)
        activity[active] = 1.0
        self.duty_cycle.data = (1.0 - alpha) * self.duty_cycle + alpha * activity

    # ── Utilitaires ───────────────────────────────────────────────────────

    def permanence_stats(self) -> dict:
        """Retourne des statistiques sur les permanences (dans le potential pool uniquement)."""
        p = self.permanences[self.potential_mask]   # 1D, seulement les synapses candidates
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
            f"potential_pct={self.potential_pct}, "
            f"δ⁺={self.delta_plus}, δ⁻={self.delta_minus}, "
            f"m={self.newborn_steps}, τ={self.tau_decay}"
        )
