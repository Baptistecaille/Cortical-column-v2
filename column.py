"""
CorticalColumn — Classe d'assemblage des 6 modules.

Pseudo-algorithme global (CLAUDE.md §3) :
    ENTRÉE  : séquence sensorielle S = {s_t}, vecteurs de vitesse v_t
    SORTIE  : représentations allocentriques, consensus inter-colonnes

Conventions :
    - K colonnes INDÉPENDANTES (pas de weight sharing — I6.1)
    - step() exécute exactement un pas du pseudo-algorithme
    - L'ordre des modules est respecté : SDR → SP → L6b → GC → DA → CC

Réf. math : §7, Algorithme 7.1 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math

from core.sdr_space import SDRSpace
from core.spatial_pooler import SpatialPooler
from core.layer6b import Layer6bTransformer
from core.grid_cell import GridCellNetwork
from core.displacement import DisplacementAlgebra, DisplacementTriplet
from core.consensus import MultiColumnConsensus


class SingleColumn(nn.Module):
    """
    Une colonne corticale indépendante avec ses 5 modules propres.

    Chaque colonne possède ses propres permanences, grid cells et
    transformateur L6b — pas de partage de poids avec les autres colonnes.

    Args:
        input_dim:      dimension sensorielle brute
        n_sdr:          dimension du SDR (SDRSpace.n)
        w:              parcimonie du SDR (SDRSpace.w)
        n_minicolumns:  nombre de minicolonnes (SpatialPooler.n_columns)
        k_active:       colonnes actives dans SpatialPooler (I2.1)
        n_grid_modules: nombre de modules de grid cells (GridCellNetwork)
        grid_periods:   périodes λ_k (copremières, CRT)
        sp_kwargs:      arguments supplémentaires pour SpatialPooler
    """

    def __init__(
        self,
        input_dim: int,
        n_sdr: int = 2048,
        w: int = 40,
        n_minicolumns: int = 256,   # 16×16 grille
        k_active: int = 40,
        n_grid_modules: int = 6,
        grid_periods: Optional[List[int]] = None,
        sp_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.n_sdr = n_sdr
        self.n_grid_modules = n_grid_modules

        sp_kwargs = sp_kwargs or {}

        # Module 1 : encodeur sensoriel
        self.sdr_space = SDRSpace(input_dim=input_dim, n=n_sdr, w=w)

        # Module 2 : sélecteur homeöstatique
        self.spatial_pooler = SpatialPooler(
            n_inputs=n_sdr,
            n_columns=n_minicolumns,
            k=k_active,
            **sp_kwargs,
        )

        # Module 3 : transformateur ego→allo (L6b)
        self.layer6b = Layer6bTransformer(
            sdr_dim=n_sdr,
            n_grid_modules=n_grid_modules,
        )

        # Module 4 : intégration de chemin (grid cells)
        self.grid_cell = GridCellNetwork(
            n_modules=n_grid_modules,
            periods=grid_periods,
            phase_dim_in=2 * n_grid_modules,
        )

    def step(
        self,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        train: bool = True,
    ) -> dict:
        """
        Exécute un pas de traitement pour cette colonne.

        Args:
            s_t:   stimulus sensoriel, shape (input_dim,)
            v_t:   vecteur de vitesse egocentrique, shape (2,)
            train: si True, effectue la mise à jour hebbienne

        Returns:
            dict avec :
                sdr:        SDR binaire, shape (n_sdr,)
                active:     indices actifs, shape (k_active,)
                allo_phase: vecteur allocentrique L6b, shape (2·n_modules,)
                phase:      phases grid cell, shape (n_modules, 2)
                grid_code:  code de position, shape (4·n_modules,)
        """
        # Module 1 — Encodage SDR (I1.1, I1.2)
        sdr = self.sdr_space.encode(s_t)                    # {0,1}^n_sdr

        # Module 2 — Sélection de colonnes (I2.1)
        active = self.spatial_pooler.forward(sdr)           # (k_active,)

        # Reconstruction dense du SDR actif (pour L6b)
        sdr_active_dense = torch.zeros(self.n_sdr, device=sdr.device)
        sdr_active_dense[active] = 1.0

        # Module 3 — Transformation ego→allo (I3.1)
        allo_phase = self.layer6b.transform(sdr_active_dense, v_t)  # (2·n_mod,)

        # Module 4 — Intégration de chemin (I4.1, I4.2)
        phase = self.grid_cell.integrate(allo_phase)        # (n_modules, 2)
        grid_code = self.grid_cell.get_code()               # (4·n_modules,)

        # Apprentissage hebbien (I2.3) — @no_grad interne
        if train:
            self.spatial_pooler.hebbian_update(sdr, active)

        return {
            "sdr": sdr,
            "active": active,
            "allo_phase": allo_phase,
            "phase": phase,
            "grid_code": grid_code,
        }


class CorticalColumn(nn.Module):
    """
    Ensemble de K colonnes corticales indépendantes avec consensus.

    Chaque colonne est une instance de SingleColumn avec ses propres
    paramètres — pas de weight sharing (I6.1).

    Le consensus est calculé par intersection (AND) des SDRs (I6.2).

    Args:
        n_columns:      nombre de colonnes K (typiquement 4–8)
        input_dim:      dimension sensorielle brute
        n_sdr:          dimension du SDR par colonne
        w:              parcimonie du SDR
        n_minicolumns:  nombre de minicolonnes par colonne
        k_active:       colonnes actives par pas
        n_grid_modules: modules de grid cells par colonne
        grid_periods:   périodes λ_k
        consensus_threshold: seuil de vote (1.0 = AND strict, I6.3)
        sp_kwargs:      arguments supplémentaires pour SpatialPooler
    """

    def __init__(
        self,
        n_columns: int = 4,
        input_dim: int = 784,       # ex. MNIST 28×28
        n_sdr: int = 2048,
        w: int = 40,
        n_minicolumns: int = 256,
        k_active: int = 40,
        n_grid_modules: int = 6,
        grid_periods: Optional[List[int]] = None,
        consensus_threshold: float = 1.0,
        sp_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.n_columns = n_columns
        self.n_sdr = n_sdr
        self.n_grid_modules = n_grid_modules

        # K colonnes INDÉPENDANTES — pas de weight sharing (I6.1)
        self.columns = nn.ModuleList([
            SingleColumn(
                input_dim=input_dim,
                n_sdr=n_sdr,
                w=w,
                n_minicolumns=n_minicolumns,
                k_active=k_active,
                n_grid_modules=n_grid_modules,
                grid_periods=grid_periods,
                sp_kwargs=sp_kwargs,
            )
            for _ in range(n_columns)
        ])

        # Module 5 : algèbre de déplacement
        self.displacement = DisplacementAlgebra(n_modules=n_grid_modules)

        # Module 6 : consensus inter-colonnes (AND strict)
        self.consensus = MultiColumnConsensus(
            n_sdr=n_sdr,
            consensus_threshold=consensus_threshold,
        )

    def step(
        self,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        train: bool = True,
    ) -> dict:
        """
        Exécute un pas complet du pseudo-algorithme pour toutes les colonnes.

        Ordre des modules (§3, Algorithme 7.1) :
            SDRSpace → SpatialPooler → Layer6bTransformer
            → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

        Args:
            s_t:   stimulus sensoriel, shape (input_dim,)
            v_t:   vecteur de vitesse egocentrique, shape (2,)
            train: si True, effectue l'apprentissage hebbien

        Returns:
            dict avec :
                sdr:        SDR de la première colonne (repr.), shape (n_sdr,)
                phase:      phases de la première colonne, shape (n_modules, 2)
                consensus:  SDR de consensus (AND), shape (n_sdr,)
                triplet:    triplet de déplacement (première colonne)
                all_sdrs:   liste des K SDRs, chacun shape (n_sdr,)
                all_phases: liste des K phases, chacun shape (n_modules, 2)
                vote_stats: statistiques du vote de consensus
        """
        all_sdrs = []
        all_phases = []
        all_grid_codes = []
        all_allo_phases = []

        # ── Étapes 1–4 pour chaque colonne ───────────────────────────────
        for col in self.columns:
            result = col.step(s_t, v_t, train=train)
            all_sdrs.append(result["sdr"])
            all_phases.append(result["phase"])
            all_grid_codes.append(result["grid_code"])
            all_allo_phases.append(result["allo_phase"])

        # ── Module 5 — Algèbre de déplacement ────────────────────────────
        # Calculé sur la première colonne (référence)
        phi_sensor = torch.zeros(self.n_grid_modules, 2, device=s_t.device)
        triplet = self.displacement.make_triplet(
            sdr_parent=all_sdrs[0],
            sdr_subobject=all_sdrs[0],
            phi_object=all_phases[0],
            phi_sensor=phi_sensor,
        )

        # ── Module 6 — Consensus inter-colonnes (AND strict) ─────────────
        vote_stats = self.consensus.vote_with_stats(all_sdrs)
        consensus_sdr = vote_stats["consensus"]

        return {
            "sdr": all_sdrs[0],
            "phase": all_phases[0],
            "consensus": consensus_sdr,
            "triplet": triplet,
            "all_sdrs": all_sdrs,
            "all_phases": all_phases,
            "all_grid_codes": all_grid_codes,
            "vote_stats": vote_stats,
        }

    def step_batch(
        self,
        s_batch: torch.Tensor,
        train: bool = True,
    ) -> dict:
        """
        Entraînement batché — version GPU-optimisée de step().

        Exécute SDRSpace + SpatialPooler en mode batch (matmuls vectorisés),
        puis calcule le consensus par batch. Layer6b et GridCell (état séquentiel)
        sont omis ici — utilisés uniquement lors de l'évaluation via step().

        Adapté aux images statiques (MNIST) où la vitesse est nulle et
        l'intégration de chemin n'est pas nécessaire à l'entraînement.

        Args:
            s_batch: stimuli sensoriels, shape (B, input_dim)
            train:   si True, effectue l'apprentissage hebbien

        Returns:
            dict avec :
                sdr_batch:       SDRs col. 0, shape (B, n_sdr)
                consensus_batch: SDR de consensus AND, shape (B, n_sdr)
                all_sdr_batches: liste des K tenseurs (B, n_sdr)
        """
        B = s_batch.shape[0]
        all_sdr_batches = []

        for col in self.columns:
            # Module 1 — SDRSpace (batch natif)
            sdr_batch = col.sdr_space.encode(s_batch)          # (B, n_sdr)

            # Module 2 — SpatialPooler (batch, GPU-friendly)
            active_batch = col.spatial_pooler.forward_batch(sdr_batch)  # (B, k)

            if train:
                col.spatial_pooler.hebbian_update_batch(sdr_batch, active_batch)

            all_sdr_batches.append(sdr_batch)

        # Module 6 — Consensus AND par batch
        # sdr_stack : (K, B, n_sdr) → vote_fraction : (B, n_sdr)
        sdr_stack = torch.stack(all_sdr_batches, dim=0).float()
        vote_fraction = sdr_stack.mean(dim=0)
        consensus_batch = (
            vote_fraction >= self.consensus.consensus_threshold
        ).float()

        return {
            "sdr_batch": all_sdr_batches[0],
            "consensus_batch": consensus_batch,
            "all_sdr_batches": all_sdr_batches,
        }

    def reset(self) -> None:
        """Réinitialise l'état interne de toutes les colonnes (nouvel épisode)."""
        for col in self.columns:
            col.grid_cell.reset()
            col.layer6b.reset_thalamic_state()

    def extra_repr(self) -> str:
        return (
            f"n_columns={self.n_columns}, n_sdr={self.n_sdr}, "
            f"n_grid_modules={self.n_grid_modules}"
        )
