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
from extensions.pe_circuits import PECircuits


class SingleColumn(nn.Module):
    """
    Une colonne corticale indépendante avec ses 5 modules propres.

    Chaque colonne possède ses propres permanences SP, grid cells et
    transformateur L6b — pas de partage de poids au sens de I6.1.

    Note sur SDRSpace (encodeur L4) :
        Chaque colonne possède son propre SDRSpace par défaut pour que
        le consensus inter-colonnes observe une vraie diversité de vues.
        Biologiquement, cela revient à des projections L4 spécialisées
        par colonne plutôt qu'à un encodeur strictement partagé.

    Prédiction causale (predictive coding) :
        À chaque pas t, AVANT d'encoder s_t, la colonne génère un SDR prédit
        à partir du grid_code mémorisé au pas précédent :
            sdr_predicted = binarize_topk(W_pred · grid_code_{t-1}, w)
        Le prédicteur W_pred est celui de pe_circuits (Lee 2025).

    Args:
        input_dim:        dimension sensorielle brute
        n_sdr:            dimension du SDR (SDRSpace.n)
        w:                parcimonie du SDR (SDRSpace.w)
        n_minicolumns:    nombre de minicolonnes (SpatialPooler.n_columns)
        k_active:         colonnes actives dans SpatialPooler (I2.1)
        n_grid_modules:   nombre de modules de grid cells (GridCellNetwork)
        grid_periods:     périodes λ_k (copremières, CRT)
        shared_sdr_space: SDRSpace partagé (None = créer un SDRSpace propre)
        sp_kwargs:        arguments supplémentaires pour SpatialPooler
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
        pe_lr: float = 0.001,
        pe_lr_pred: float = 0.01,
        shared_sdr_space=None,
        sp_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.n_sdr = n_sdr
        self.n_grid_modules = n_grid_modules
        self.pe_lr = pe_lr
        self.pe_lr_pred = pe_lr_pred

        sp_kwargs = sp_kwargs or {}

        # Module 1 : encodeur sensoriel (partagé si fourni, sinon propre)
        if shared_sdr_space is not None:
            self.sdr_space = shared_sdr_space
        else:
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
        )

        # Module PE : erreur de prédiction signée (Lee 2025 — PRIORITÉ 1)
        # Prédicteur top-down : grid_code (4·n_modules) → SDR attendu (n_sdr)
        self.pe_circuits = PECircuits(
            dim=n_sdr,
            context_dim=4 * n_grid_modules,
        )

        # Buffer mémorisant le grid_code du pas précédent.
        # Utilisé pour générer la prédiction causale sdr_predicted au pas t
        # AVANT de recevoir s_t (convention predictive coding Rao-Ballard 1999).
        # Initialisé à zéro → prédiction nulle au premier pas (normal).
        self.register_buffer(
            "prev_grid_code",
            torch.zeros(4 * n_grid_modules),
        )

    @torch.no_grad()
    def _make_sdr_predicted(self) -> torch.Tensor:
        """
        Génère le SDR prédit pour le pas courant depuis le grid_code mémorisé.

        Utilise le prédicteur top-down de PECircuits (W_pred déjà entraîné) :
            logits      = W_pred · prev_grid_code + b_pred   (linéaire)
            sdr_pred    = top-w strict(logits) → {0,1}^n_sdr (binarisation dure)

        Convention causale : prev_grid_code est le grid_code du pas t-1,
        mis à jour EN FIN de step(). La prédiction est donc disponible
        AVANT de recevoir s_t, conformément au schéma Rao-Ballard (1999)
        et à la projection L6b→L4 descendante.

        Returns:
            sdr_predicted: SDR prédit, shape (n_sdr,), binaire, ||·||₁ = w
        """
        logits = self.pe_circuits.predictor(self.prev_grid_code)  # (n_sdr,)
        _, top_idx = torch.topk(logits, self.spatial_pooler.k, sorted=False)
        sdr_predicted = torch.zeros(self.n_sdr, device=self.prev_grid_code.device)
        sdr_predicted[top_idx] = 1.0
        return sdr_predicted

    def step(
        self,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        train: bool = True,
        gamma_override: float | None = None,
        l6_init: torch.Tensor | None = None,
    ) -> dict:
        """
        Exécute un pas de traitement pour cette colonne.

        Ordre causal (predictive coding) :
            0. Prédiction top-down depuis grid_code_{t-1}  ← AVANT s_t
            1. Encodage SDR de s_t
            2. Sélection de colonnes (SpatialPooler)
            3. Transformation ego→allo (L6b)
            4. Intégration de chemin (GridCell) → grid_code_t
            5. Apprentissage hebbien + PE circuits (train uniquement)
            6. Mémorisation de grid_code_t → prev_grid_code  ← EN FIN

        Args:
            s_t:            stimulus sensoriel, shape (input_dim,)
            v_t:            vecteur de vitesse egocentrique, shape (2,)
            train:          si True, effectue la mise à jour hebbienne
            gamma_override: γ effectif calculé depuis gamma_surprise() dans
                            train.py. Si None → schedule temporel pur.
                            Permet de moduler la plasticité par la surprise
                            de prédiction (Rao & Ballard 1999).
            l6_init:        grid_code initial pour l'inférence itérative
                            (Protocole B). Si None, utilise prev_grid_code
                            courant (comportement normal). Shape (4·n_modules,).

        Returns:
            dict avec :
                sdr:           SDR binaire observé,  shape (n_sdr,)
                sdr_predicted: SDR prédit (top-down), shape (n_sdr,)
                active:        indices actifs, shape (k_active,)
                allo_phase:    vecteur allocentrique L6b, shape (2·n_modules,)
                phase:         phases grid cell, shape (n_modules, 2)
                grid_code:     code de position, shape (4·n_modules,)
                surprise:      erreur de prédiction Jaccard ∈ [0,1]
        """
        # Initialisation explicite de prev_grid_code pour inférence itérative
        if l6_init is not None:
            self.prev_grid_code.data.copy_(l6_init.detach())

        # ── Étape 0 — Prédiction causale (AVANT encodage de s_t) ─────────
        # W_pred a été mis à jour au pas précédent → prédiction informée
        sdr_predicted = self._make_sdr_predicted()          # {0,1}^n_sdr

        # Module 1 — Encodage SDR (I1.1, I1.2)
        sdr = self.sdr_space.encode(s_t)                    # {0,1}^n_sdr

        # Module 2 — Sélection de colonnes (I2.1)
        active = self.spatial_pooler.forward(sdr)           # (k_active,)

        # Reconstruction dense du SDR actif (pour L6b)
        sdr_active_dense = torch.zeros(self.n_sdr, device=sdr.device)
        sdr_active_dense[active] = 1.0

        # Module 3 — Transformation ego→allo (I3.1)
        allo_phase = self.layer6b.transform(sdr_active_dense, v_t)  # (2·n_mod,)

        # Module 4 — Intégration de chemin via vitesse directe (I4.1, I4.2)
        prev_phase = self.grid_cell.phases.clone()
        phase = self.grid_cell.integrate(v_t)               # (n_modules, 2)
        grid_code = self.grid_cell.get_code()               # (4·n_modules,)

        # ── Calcul de la surprise (Jaccard complement SDR_pred vs SDR_obs) ─
        # ε_t = 1 - |sdr_predicted ∩ sdr| / |sdr_predicted ∪ sdr|
        # ε = 0 : prédiction parfaite → plasticité minimale (γ_floor)
        # ε = 1 : prédiction nulle   → plasticité maximale (γ = 1.0)
        # Calcul @no_grad, pas de coût autograd.
        with torch.no_grad():
            inter = (sdr_predicted * sdr).sum()
            union = ((sdr_predicted + sdr) > 0).float().sum().clamp(min=1.0)
            surprise = float(1.0 - inter / union)

        if train:
            # Apprentissage hebbien SP (I2.3) — @no_grad interne.
            # gamma_override injecté ici si fourni par CorticalColumn.step()
            # (calculé depuis spatial_pooler.gamma_surprise(surprise) dans train.py).
            self.spatial_pooler.hebbian_update(sdr, active, gamma_override=gamma_override)

            # PE circuits (Lee 2025 — PRIORITÉ 1)
            # 1. Mise à jour du prédicteur top-down + états PE+/PE−
            self.pe_circuits.step_with_update(
                sdr.float(),
                grid_code,
                lr_pred=self.pe_lr_pred,
            )
            # 2. Mise à jour de W_enc via l'erreur PE signée
            delta_W = self.pe_circuits.modulated_update(
                s_t.float(),
                learning_rate=self.pe_lr,
            )
            self.sdr_space.pe_update(delta_W)

        # ── Étape 6 — Mémorisation du grid_code pour la prédiction au pas t+1
        # Fait EN FIN de step(), après l'apprentissage PE, pour que W_pred
        # soit à jour avant d'être utilisé par _make_sdr_predicted() au pas suivant.
        self.prev_grid_code.data.copy_(grid_code.detach())

        return {
            "sdr":           sdr,
            "sdr_predicted": sdr_predicted,   # prédiction causale (t-1 → t)
            "active":        active,
            "allo_phase":    allo_phase,
            "prev_phase":    prev_phase,
            "phase":         phase,
            "grid_code":     grid_code,
            "surprise":      surprise,         # ε_t ∈ [0,1] — erreur Jaccard
        }


    @torch.no_grad()
    def step_batch_external(
        self,
        s_batch: torch.Tensor,
        v_batch: torch.Tensor,
        ext_state: dict,
        train: bool = True,
        gamma_override: Optional[float] = None,
    ) -> tuple:
        """
        Traitement de B épisodes indépendants en parallèle via état externe.

        Toute la pipeline SDR→SP→L6b→GC→PE est vectorisée sur B.
        L'état externe (prev_grid_code, h_tc, phases, r_pos, r_neg) est passé
        explicitement et retourné mis à jour — aucune mutation de buffer.
        Cela permet de maintenir B flux d'états indépendants avec un seul jeu
        de poids partagés (parcimonie GPU maximale).

        Convention de parcimonie pour la mise à jour hebbienne :
            hebbian_update_batch accumule les gradients de B images dans une
            seule mise à jour de permanence (équivalent à B pas séquentiels).

        Args:
            s_batch:         stimuli, shape (B, input_dim)
            v_batch:         vitesses, shape (B, 2)
            ext_state:       dict avec clés :
                               prev_grid_code (B, 4·n_modules)
                               h_tc           (B, 2·n_modules)
                               phases         (B, n_modules, 2)
                               r_pos          (B, n_sdr)
                               r_neg          (B, n_sdr)
            train:           si True, effectue les mises à jour hebbienne et PE
            gamma_override:  γ effectif depuis gamma_surprise() (None = schedule pur)

        Returns:
            (results_batch, new_ext_state)
            results_batch:  dict avec sdr (B,n_sdr), sdr_predicted (B,n_sdr),
                            active (B,k), allo_phase (B,phase_dim),
                            prev_phases (B,n_mod,2), phases (B,n_mod,2),
                            grid_code (B,4·n_mod), surprise (B,)
            new_ext_state:  dict mis à jour (même structure qu'ext_state)
        """
        B = s_batch.shape[0]
        device = s_batch.device

        # ── Étape 0 : prédiction causale depuis prev_grid_code ────────────
        logits = self.pe_circuits.predictor(ext_state["prev_grid_code"])   # (B, n_sdr)
        _, top_idx = torch.topk(logits, self.spatial_pooler.k, dim=-1, sorted=False)
        sdr_predicted = torch.zeros(B, self.n_sdr, device=device)
        sdr_predicted.scatter_(1, top_idx, 1.0)                            # (B, n_sdr)

        # ── Module 1 : encodage SDR ──────────────────────────────────────
        sdr = self.sdr_space.encode(s_batch)                               # (B, n_sdr)

        # ── Module 2 : sélection de colonnes ────────────────────────────
        active = self.spatial_pooler.forward_batch(sdr)                    # (B, k)

        # SDR actif dense (B, n_sdr)
        sdr_active_dense = torch.zeros(B, self.n_sdr, device=device)
        sdr_active_dense.scatter_(1, active, 1.0)

        # ── Module 3 : transformation ego→allo ──────────────────────────
        allo_phase, new_h_tc = self.layer6b.transform_batch(
            sdr_active_dense, v_batch, ext_state["h_tc"]
        )                                                                   # (B, phase_dim)

        # ── Module 4 : intégration de chemin ────────────────────────────
        prev_phases = ext_state["phases"]
        new_phases, grid_code = self.grid_cell.integrate_batch(
            v_batch, ext_state["phases"]
        )                                                                   # (B, n_mod, 2), (B, 4·n_mod)

        # ── Surprise : Jaccard complement, par échantillon ───────────────
        inter    = (sdr_predicted * sdr).sum(dim=-1)                       # (B,)
        union    = ((sdr_predicted + sdr) > 0).float().sum(dim=-1).clamp(min=1.0)
        surprise = 1.0 - inter / union                                     # (B,)

        # ── Apprentissage ────────────────────────────────────────────────
        if train:
            self.spatial_pooler.hebbian_update_batch(
                sdr, active, gamma_override=gamma_override
            )
            new_r_pos, new_r_neg = self.pe_circuits.step_with_update_batch(
                sdr.float(), grid_code,
                ext_state["r_pos"], ext_state["r_neg"],
                lr_pred=self.pe_lr_pred,
            )
            delta_W = self.pe_circuits.modulated_update_batch(
                s_batch.float(), new_r_pos, new_r_neg,
                learning_rate=self.pe_lr,
            )
            self.sdr_space.pe_update(delta_W)
        else:
            new_r_pos = ext_state["r_pos"]
            new_r_neg = ext_state["r_neg"]

        new_state = {
            "prev_grid_code": grid_code,
            "h_tc":           new_h_tc,
            "phases":         new_phases,
            "r_pos":          new_r_pos,
            "r_neg":          new_r_neg,
        }
        results = {
            "sdr":           sdr,
            "sdr_predicted": sdr_predicted,
            "active":        active,
            "allo_phase":    allo_phase,
            "prev_phases":   prev_phases,
            "phases":        new_phases,
            "grid_code":     grid_code,
            "surprise":      surprise,
        }
        return results, new_state


class CorticalColumn(nn.Module):
    """
    Ensemble de K colonnes corticales indépendantes avec consensus.

    Chaque colonne est une instance de SingleColumn avec ses propres
    paramètres — pas de weight sharing (I6.1).

    Le consensus est calculé par intersection (AND) des SDRs (I6.2).
    Le vote inter-colonnes (Phase 2, CMP) module l'apprentissage hebbien
    pour forcer la convergence des représentations sans weight sharing.

    Args:
        n_columns:        nombre de colonnes K (typiquement 4–8)
        input_dim:        dimension sensorielle brute
        n_sdr:            dimension du SDR par colonne
        w:                parcimonie du SDR
        n_minicolumns:    nombre de minicolonnes par colonne
        k_active:         colonnes actives par pas
        n_grid_modules:   modules de grid cells par colonne
        grid_periods:     périodes λ_k
        consensus_threshold: seuil de vote (1.0 = AND strict, I6.3)
        enable_vote:      active le vote inter-colonnes (Phase 2 CMP)
        alpha_divergence: pénalité de dépression sur les bits divergents du vote
        sp_kwargs:        arguments supplémentaires pour SpatialPooler
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
        enable_vote: bool = False,
        alpha_divergence: float = 3.0,
        pe_lr: float = 0.001,
        pe_lr_pred: float = 0.01,
        sp_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.n_columns = n_columns
        self.n_sdr = n_sdr
        self.n_grid_modules = n_grid_modules
        self.w = w
        self.enable_vote = enable_vote
        self.alpha_divergence = alpha_divergence
        # État glissant pour cmp_vote_stability (Jaccard consécutif)
        self._prev_vote_sdr: Optional[torch.Tensor] = None

        # K colonnes INDÉPENDANTES (SDRSpace, permanences SP, L6b, GridCell) — I6.1
        self.columns = nn.ModuleList([
            SingleColumn(
                input_dim=input_dim,
                n_sdr=n_sdr,
                w=w,
                n_minicolumns=n_minicolumns,
                k_active=k_active,
                n_grid_modules=n_grid_modules,
                grid_periods=grid_periods,
                pe_lr=pe_lr,
                pe_lr_pred=pe_lr_pred,
                shared_sdr_space=None,
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

    @torch.no_grad()
    def _cross_column_vote(
        self,
        all_sdrs: List[torch.Tensor],
        all_grid_codes: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Vote inter-colonnes pondéré par la cohérence de pose (Phase 2 CMP).

        Protocole :
            1. Calcule le grid_code moyen (référence de pose)
            2. Pondère chaque colonne par cos_sim(grid_code_c, grid_code_mean)
            3. Somme pondérée des SDRs → top-w strict → SDR binaire de vote

        La pondération par cosinus sur le grid_code (vecteur cos/sin 24D)
        donne plus de poids aux colonnes dont la pose est cohérente avec
        le consensus spatial courant.

        Args:
            all_sdrs:        liste de K tenseurs (n_sdr,) — SDRs binaires
            all_grid_codes:  liste de K tenseurs (4·n_modules,) — codes de pose

        Returns:
            vote_sdr: SDR binaire de vote, shape (n_sdr,), ||vote||₁ = w
        """
        K = len(all_sdrs)
        device = all_sdrs[0].device

        # Grid codes empilés : (K, 4·n_modules)
        gc_stack = torch.stack(all_grid_codes, dim=0).float()   # (K, D_gc)

        # Pose de référence = moyenne normalisée
        gc_mean = gc_stack.mean(dim=0)                           # (D_gc,)
        gc_mean_norm = gc_mean / (gc_mean.norm() + 1e-8)

        # Similarité cosinus de chaque colonne avec la pose moyenne
        gc_norms = gc_stack.norm(dim=1, keepdim=True) + 1e-8    # (K, 1)
        gc_normalized = gc_stack / gc_norms                     # (K, D_gc)
        cos_weights = (gc_normalized @ gc_mean_norm).clamp(min=0.0)  # (K,)

        # Si toutes les poses sont orthogonales (cos=0), poids uniformes
        if cos_weights.sum() < 1e-8:
            cos_weights = torch.ones(K, device=device)
        cos_weights = cos_weights / cos_weights.sum()           # normalisation

        # Somme pondérée des SDRs : (n_sdr,)
        sdr_stack = torch.stack(all_sdrs, dim=0).float()        # (K, n_sdr)
        vote_scores = (cos_weights.unsqueeze(1) * sdr_stack).sum(dim=0)  # (n_sdr,)

        # Top-w strict → SDR binaire (parcimonie dure, invariant I1.1)
        _, top_idx = torch.topk(vote_scores, self.w, sorted=False)
        vote_sdr = torch.zeros(self.n_sdr, device=device)
        vote_sdr[top_idx] = 1.0

        return vote_sdr

    def step(
        self,
        s_t: torch.Tensor,
        v_t: torch.Tensor,
        train: bool = True,
        gamma_override: float | None = None,
        l6_init: torch.Tensor | list | None = None,
    ) -> dict:
        """
        Exécute un pas complet du pseudo-algorithme pour toutes les colonnes.

        Ordre des modules (§3, Algorithme 7.1) :
            SDRSpace → SpatialPooler → Layer6bTransformer
            → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

        Args:
            s_t:            stimulus sensoriel, shape (input_dim,)
            v_t:            vecteur de vitesse egocentrique, shape (2,)
            train:          si True, effectue l'apprentissage hebbien
            gamma_override: γ effectif depuis gamma_surprise(), calculé dans
                            train.py à partir de la surprise du pas précédent.
                            Propagé à chaque SingleColumn.step() → SpatialPooler.
                            None → schedule temporel pur (rétro-compatible).
            l6_init:        grid_code initial pour l'inférence itérative
                            (Protocole B évaluation prédictive). Si tenseur
                            1D, diffusé à toutes les colonnes. Si liste de K
                            tenseurs, un par colonne. None = comportement normal.

        Returns:
            dict avec :
                sdr:        SDR de la première colonne (repr.), shape (n_sdr,)
                phase:      phases de la première colonne, shape (n_modules, 2)
                consensus:  SDR de consensus (AND), shape (n_sdr,)
                triplet:    triplet de déplacement (première colonne)
                all_sdrs:   liste des K SDRs, chacun shape (n_sdr,)
                all_phases: liste des K phases, chacun shape (n_modules, 2)
                vote_stats: statistiques du vote de consensus
                surprise:   ε_t moyen sur les K colonnes ∈ [0,1]
        """
        all_sdrs = []
        all_phases = []
        all_prev_phases = []
        all_grid_codes = []
        all_allo_phases = []
        all_actives = []
        all_sdrs_predicted = []   # prédictions causales (une par colonne)
        all_surprises = []        # ε_t par colonne (moyenne ensuite)

        # ── Étapes 0–4 pour chaque colonne ───────────────────────────────
        for c_idx, col in enumerate(self.columns):
            # Résolution du l6_init par colonne
            if l6_init is None:
                col_l6_init = None
            elif isinstance(l6_init, list):
                col_l6_init = l6_init[c_idx] if c_idx < len(l6_init) else None
            else:
                col_l6_init = l6_init  # même tenseur diffusé à toutes les colonnes

            result = col.step(
                s_t, v_t, train=train,
                gamma_override=gamma_override,
                l6_init=col_l6_init,
            )
            all_sdrs.append(result["sdr"])
            all_sdrs_predicted.append(result["sdr_predicted"])
            all_phases.append(result["phase"])
            all_prev_phases.append(result["prev_phase"])
            all_grid_codes.append(result["grid_code"])
            all_allo_phases.append(result["allo_phase"])
            all_actives.append(result["active"])
            all_surprises.append(result["surprise"])

        # ── Vote inter-colonnes (Phase 2 CMP) ────────────────────────────
        # Calcul systématique pour disposer d'une baseline passive même
        # quand enable_vote=False — sans cela, l'A/B test n'est pas interprétable.
        vote_sdr = None
        if self.n_columns > 1:
            vote_sdr = self._cross_column_vote(all_sdrs, all_grid_codes)
            if train and self.enable_vote:
                # Mise à jour hebbienne ciblée : col.step(train=True) a déjà
                # incrémenté t_step. On applique l'update ciblé en PLUS, avec
                # pression de dépression uniquement sur les bits divergents
                # (alpha_divergence, §2 CMP Clay-Leadholm 2024).
                for c, col in enumerate(self.columns):
                    col.spatial_pooler.hebbian_update_targeted(
                        x=all_sdrs[c],
                        active=all_actives[c],
                        vote_sdr=vote_sdr,
                        alpha_divergence=self.alpha_divergence,
                    )

        # ── Métriques CMP proximales (diagnostics A/B test enable_vote) ──
        # Calculées dans les deux conditions (enable_vote=True/False).
        cmp_jaccard_list: List[float] = []
        cmp_pressure_list: List[float] = []
        if vote_sdr is not None:
            for sdr_c in all_sdrs:
                s = sdr_c.float()
                v = vote_sdr.float()
                inter = (s * v).sum().item()
                union = ((s + v) > 0).float().sum().item()
                # cmp_jaccard_active_vs_vote : Jaccard(sdr_c, vote_sdr)
                cmp_jaccard_list.append(inter / union if union > 0 else 0.0)
                # cmp_pressure : fraction de bits actifs déprimés à α·δ⁻
                n_active = s.sum().item()
                cmp_pressure_list.append(
                    1.0 - inter / n_active if n_active > 0 else 0.0
                )

        cmp_jaccard = (
            sum(cmp_jaccard_list) / len(cmp_jaccard_list) if cmp_jaccard_list else float("nan")
        )
        cmp_pressure = (
            sum(cmp_pressure_list) / len(cmp_pressure_list) if cmp_pressure_list else float("nan")
        )

        # cmp_vote_stability : Jaccard(vote_sdr_t, vote_sdr_{t-1})
        if vote_sdr is not None and self._prev_vote_sdr is not None:
            v_curr = vote_sdr.float()
            v_prev = self._prev_vote_sdr.float()
            inter_v = (v_curr * v_prev).sum().item()
            union_v = ((v_curr + v_prev) > 0).float().sum().item()
            cmp_vote_stability = inter_v / union_v if union_v > 0 else 0.0
        else:
            cmp_vote_stability = float("nan")

        # Mise à jour du vote précédent (détaché du graphe, pas de gradient)
        self._prev_vote_sdr = vote_sdr.clone() if vote_sdr is not None else None

        # ── Module 5 — Algèbre de déplacement ────────────────────────────
        # Déplacement courant = phase après intégration vs phase précédente.
        # Cela fournit un triplet non trivial même sans annotation d'objet.
        phi_sensor = all_prev_phases[0]
        triplet = self.displacement.make_triplet(
            sdr_parent=all_sdrs_predicted[0],
            sdr_subobject=all_sdrs[0],
            phi_object=all_phases[0],
            phi_sensor=phi_sensor,
        )

        # ── Module 6 — Consensus inter-colonnes (AND strict) ─────────────
        vote_stats = self.consensus.vote_with_stats(all_sdrs)
        consensus_sdr = vote_stats["consensus"]

        # Prédiction de consensus : AND strict des SDRs prédits par chaque colonne.
        # Interprétation : seuls les bits sur lesquels TOUTES les colonnes
        # s'accordent sont considérés comme prédits avec confiance.
        predicted_consensus = self.consensus.vote_with_stats(
            all_sdrs_predicted
        )["consensus"]

        # Surprise moyenne sur les K colonnes (signal de plasticité global)
        mean_surprise = float(sum(all_surprises) / max(len(all_surprises), 1))

        return {
            "sdr":              all_sdrs[0],
            "sdr_predicted":    all_sdrs_predicted[0],          # col. 0, prédiction causale
            "predicted_consensus": predicted_consensus,          # AND sur K prédictions
            "phase":            all_phases[0],
            "consensus":        consensus_sdr,
            "triplet":          triplet,
            "all_sdrs":         all_sdrs,
            "all_sdrs_predicted": all_sdrs_predicted,
            "all_phases":       all_phases,
            "all_grid_codes":   all_grid_codes,
            "vote_stats":              vote_stats,
            "vote_sdr":                vote_sdr,            # None si n_columns==1
            "surprise":                mean_surprise,        # ε_t moyen ∈ [0,1]
            # Diagnostics CMP proximaux — présents quelle que soit enable_vote
            "cmp_jaccard_active_vs_vote": cmp_jaccard,      # Jaccard(sdr_c, vote_sdr) ↑ si CMP converge
            "cmp_pressure":            cmp_pressure,         # bits déprimés / w ↓ si CMP converge
            "cmp_vote_stability":      cmp_vote_stability,   # Jaccard(vote_t, vote_{t-1}) ↑ si attracteur stable
        }

    def make_batch_state(self, B: int, device: torch.device) -> List[dict]:
        """
        Alloue les états externes initiaux pour B épisodes indépendants.

        Appeler avant chaque nouveau lot d'épisodes (équivalent à reset() pour
        le mode séquentiel). Les phases sont initialisées aléatoirement sur 𝕋²
        comme dans GridCellNetwork.reset().

        Args:
            B:      taille de lot (nombre d'épisodes indépendants)
            device: device cible (cpu/cuda)

        Returns:
            Liste de K dicts, un par colonne, chacun avec les tenseurs
            (B, ...) pour prev_grid_code, h_tc, phases, r_pos, r_neg.
        """
        states = []
        for col in self.columns:
            phase_dim = col.layer6b.phase_dim          # 2 * n_grid_modules
            n_modules = col.grid_cell.n_modules
            ctx_dim   = 4 * col.n_grid_modules

            states.append({
                "prev_grid_code": torch.zeros(B, ctx_dim,           device=device),
                "h_tc":           torch.zeros(B, phase_dim,         device=device),
                "phases":         torch.rand (B, n_modules, 2,      device=device) * 2 * math.pi,
                "r_pos":          torch.zeros(B, col.n_sdr,         device=device),
                "r_neg":          torch.zeros(B, col.n_sdr,         device=device),
            })
        return states

    @torch.no_grad()
    def step_parallel(
        self,
        s_batch: torch.Tensor,
        v_batch: torch.Tensor,
        state_batch: List[dict],
        train: bool = True,
        gamma_override: Optional[float] = None,
    ) -> tuple:
        """
        Traitement de B épisodes indépendants × K colonnes en parallèle.

        Chaque colonne traite les B images avec step_batch_external() (vectorisé).
        Les K colonnes sont itérées séquentiellement (poids différents), mais
        chaque colonne sature le GPU sur B échantillons simultanément.

        Remplace CorticalColumn.step() pour l'entraînement batché.

        Args:
            s_batch:       stimuli, shape (B, input_dim)
            v_batch:       vitesses, shape (B, 2)
            state_batch:   liste de K dicts d'états externes (B, ...)
            train:         si True, mises à jour hebbienne + PE
            gamma_override: γ effectif (None = schedule pur)

        Returns:
            (results, new_state_batch)
            results contient :
                sdr       (B, n_sdr)  — première colonne
                consensus (B, n_sdr)  — AND strict
                all_sdrs  list K × (B, n_sdr)
                all_grid_codes list K × (B, 4·n_mod)
                surprise  (B,)        — moyenne sur K colonnes
        """
        all_sdrs:          List[torch.Tensor] = []
        all_sdrs_predicted: List[torch.Tensor] = []
        all_grid_codes:    List[torch.Tensor] = []
        all_phases:        List[torch.Tensor] = []
        all_surprises:     List[torch.Tensor] = []
        new_states:        List[dict] = []

        for c_idx, col in enumerate(self.columns):
            res, new_st = col.step_batch_external(
                s_batch, v_batch, state_batch[c_idx],
                train=train, gamma_override=gamma_override,
            )
            all_sdrs.append(res["sdr"])
            all_sdrs_predicted.append(res["sdr_predicted"])
            all_grid_codes.append(res["grid_code"])
            all_phases.append(res["phases"])
            all_surprises.append(res["surprise"])
            new_states.append(new_st)

        # ── Consensus AND strict (I6.2, I6.3) ────────────────────────────
        # sdr_stack : (K, B, n_sdr) → vote_fraction : (B, n_sdr)
        sdr_stack    = torch.stack(all_sdrs, dim=0).float()               # (K, B, n_sdr)
        vote_fraction = sdr_stack.mean(dim=0)                              # (B, n_sdr)
        consensus    = (vote_fraction >= self.consensus.consensus_threshold).float()

        # ── Prédiction consensus AND strict sur les SDRs prédits ─────────
        pred_stack    = torch.stack(all_sdrs_predicted, dim=0).float()
        pred_consensus = (pred_stack.mean(dim=0) >= self.consensus.consensus_threshold).float()

        # ── Surprise moyenne K colonnes ───────────────────────────────────
        mean_surprise = torch.stack(all_surprises, dim=0).mean(dim=0)     # (B,)

        results = {
            "sdr":              all_sdrs[0],
            "sdr_predicted":    all_sdrs_predicted[0],
            "predicted_consensus": pred_consensus,
            "consensus":        consensus,
            "all_sdrs":         all_sdrs,
            "all_sdrs_predicted": all_sdrs_predicted,
            "all_grid_codes":   all_grid_codes,
            "all_phases":       all_phases,
            "surprise":         mean_surprise,
        }
        return results, new_states

    @torch.no_grad()
    def anchor_batch(
        self,
        state_batch: List[dict],
        initial_phases_batch: List[torch.Tensor],
        confidence: float = 0.8,
    ) -> None:
        """
        Applique l'ancrage de retour à l'origine sur les B états en parallèle.

        Modifie state_batch["phases"] en place pour chaque colonne.

        Args:
            state_batch:          K dicts d'états courants
            initial_phases_batch: K tenseurs (B, n_modules, 2) des phases initiales
            confidence:           confiance dans le repère ∈ [0, 1]
        """
        for c_idx, col in enumerate(self.columns):
            corrected = col.grid_cell.anchor_batch(
                state_batch[c_idx]["phases"],
                initial_phases_batch[c_idx],
                confidence=confidence,
            )
            state_batch[c_idx]["phases"] = corrected

    def step_batch(
        self,
        s_batch: torch.Tensor,
        v_batch: Optional[torch.Tensor] = None,
        train: bool = True,
        reset_each_sample: bool = True,
    ) -> dict:
        """
        Entraînement batché — wrapper pratique au-dessus de step().

        Chaque élément du batch est traité comme un échantillon indépendant.
        Par défaut, l'état interne est réinitialisé avant chaque sample afin
        d'exécuter aussi Layer6b, GridCell, DisplacementAlgebra et les PE
        circuits sur le chemin complet du modèle.

        Args:
            s_batch:          stimuli sensoriels, shape (B, input_dim)
            v_batch:          vitesses, shape (B, 2). Si None, zéros.
            train:            si True, effectue l'apprentissage hebbien
            reset_each_sample:si True, chaque sample démarre d'un état propre

        Returns:
            dict avec :
                sdr_batch:          SDRs col. 0, shape (B, n_sdr)
                sdr_predicted_batch: prédictions col. 0, shape (B, n_sdr)
                phase_batch:        phases col. 0, shape (B, n_modules, 2)
                grid_code_batch:    grid codes col. 0, shape (B, 4·n_modules)
                consensus_batch:    consensus AND, shape (B, n_sdr)
                all_sdr_batches:    liste des K tenseurs (B, n_sdr)
                all_grid_code_batches: liste des K tenseurs (B, 4·n_modules)
        """
        B = s_batch.shape[0]
        if v_batch is None:
            v_batch = torch.zeros(B, 2, device=s_batch.device, dtype=s_batch.dtype)
        if v_batch.shape != (B, 2):
            raise ValueError(
                f"v_batch.shape={tuple(v_batch.shape)} doit être {(B, 2)}"
            )

        sample_results = []
        for i in range(B):
            if reset_each_sample:
                self.reset()
            sample_results.append(
                self.step(s_batch[i], v_batch[i], train=train)
            )

        sdr_batch = torch.stack(
            [r["sdr"] for r in sample_results], dim=0
        )
        sdr_predicted_batch = torch.stack(
            [r["sdr_predicted"] for r in sample_results], dim=0
        )
        phase_batch = torch.stack(
            [r["phase"] for r in sample_results], dim=0
        )
        consensus_batch = torch.stack(
            [r["consensus"] for r in sample_results], dim=0
        )
        grid_code_batch = torch.stack(
            [r["all_grid_codes"][0] for r in sample_results], dim=0
        )

        all_sdr_batches = [
            torch.stack([r["all_sdrs"][c] for r in sample_results], dim=0)
            for c in range(self.n_columns)
        ]
        all_grid_code_batches = [
            torch.stack([r["all_grid_codes"][c] for r in sample_results], dim=0)
            for c in range(self.n_columns)
        ]

        return {
            "sdr_batch": sdr_batch,
            "sdr_predicted_batch": sdr_predicted_batch,
            "phase_batch": phase_batch,
            "grid_code_batch": grid_code_batch,
            "consensus_batch": consensus_batch,
            "all_sdr_batches": all_sdr_batches,
            "all_grid_code_batches": all_grid_code_batches,
        }

    def reset(self) -> None:
        """Réinitialise l'état interne de toutes les colonnes (nouvel épisode).

        Inclut le buffer de prédiction prev_grid_code : après un reset,
        la première prédiction est nulle (prior uniforme sur la position) —
        comportement correct car le modèle ne sait pas encore où il se trouve.
        """
        self._prev_vote_sdr = None
        for col in self.columns:
            col.grid_cell.reset()
            col.layer6b.reset_thalamic_state()
            col.pe_circuits.reset()
            col.prev_grid_code.data.zero_()   # prédiction nulle au démarrage

    def extra_repr(self) -> str:
        return (
            f"n_columns={self.n_columns}, n_sdr={self.n_sdr}, "
            f"n_grid_modules={self.n_grid_modules}"
        )
