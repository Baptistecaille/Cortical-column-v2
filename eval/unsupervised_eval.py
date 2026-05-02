"""
Pipeline d'évaluation non-supervisée — 7 métriques.

Métriques :
    ε            : erreur de reconstruction (fidélité SDR)
    sparsity     : parcimonie L2/3 (doit respecter le seuil sur t_max)
    var_red      : réduction de variance par le vote inter-colonnes
    lin_prob     : précision de sondage linéaire (linear probing)
    nmi          : NMI k-means sur les représentations
    SI           : indice de spécialisation des colonnes
    pred_success : taux de succès des prédictions (predictive coding)

Ces 7 métriques correspondent à l'évaluation non-supervisée définie dans
CLAUDE.md §3 et le pseudo-algorithme global.

Flags CLI :
    --eval-prediction  active les Protocoles A et B (prediction_eval.py)
    --eval-ood         active l'évaluation OOD (generalization_eval.py)
    --eval-all         active tous les protocoles (équivalent --eval-prediction --eval-ood)

Réf. math : §8 — Formalisation_Mille_Cerveaux.pdf
"""

import sys
import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


def reconstruction_error(
    sdr_pred: torch.Tensor,
    sdr_true: torch.Tensor,
) -> float:
    """
    Calcule l'erreur de reconstruction ε entre deux SDRs.

    ε = 1 − |SDR_pred ∩ SDR_true| / |SDR_true|
      = fraction de bits manquants dans la prédiction

    Réf. math : §8.1, Éq. 8.1.

    Args:
        sdr_pred: SDR prédit,  shape (n,), binaire
        sdr_true: SDR cible,   shape (n,), binaire

    Returns:
        ε ∈ [0, 1] (0 = reconstruction parfaite)
    """
    intersection = (sdr_pred * sdr_true).sum()
    n_true = sdr_true.sum()
    if n_true == 0:
        return 0.0
    return float(1.0 - intersection / n_true)


def sdr_sparsity(sdr: torch.Tensor, expected_w: int) -> dict:
    """
    Vérifie la parcimonie du SDR par rapport à la spécification.

    Args:
        sdr:        SDR binaire, shape (n,)
        expected_w: nombre de bits actifs attendus (w de SDRSpace)

    Returns:
        dict avec :
            actual_w:     bits actifs réels
            target_w:     w spécifié
            ratio:        actual_w / target_w (1.0 = correct)
            is_valid:     True si actual_w == expected_w
    """
    actual_w = int(sdr.sum().item())
    return {
        "actual_w": actual_w,
        "target_w": expected_w,
        "ratio": actual_w / max(expected_w, 1),
        "is_valid": actual_w == expected_w,
    }


def vote_variance_reduction(
    sdrs: List[torch.Tensor],
    consensus: torch.Tensor,
) -> float:
    """
    Mesure la réduction de variance apportée par le vote inter-colonnes.

    var_red = 1 − Var(consensus) / mean(Var(sdr_k))

    Une réduction proche de 1 indique que le consensus est plus stable
    que les SDRs individuels (bénéfice du vote multi-colonnes).

    Réf. math : §8.3, Éq. 8.4.

    Args:
        sdrs:      liste de K SDRs individuels, chacun shape (n,)
        consensus: SDR de consensus, shape (n,)

    Returns:
        var_red ∈ [0, 1] (1 = réduction totale, 0 = aucun bénéfice)
    """
    sdr_stack = torch.stack(sdrs, dim=0).float()  # (K, n)
    var_individual = sdr_stack.var(dim=0).mean().item()

    var_consensus = consensus.float().var().item()

    if var_individual == 0:
        return 1.0
    return float(1.0 - var_consensus / var_individual)


def linear_probing_accuracy(
    representations: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    n_epochs: int = 50,
    lr: float = 0.01,
) -> float:
    """
    Évalue la qualité des représentations par sondage linéaire.

    Entraîne un classifieur linéaire sur les représentations SDR/phase
    et retourne la précision de classification.

    Réf. math : §8.4.

    Args:
        representations: tenseur de représentations, shape (N, dim)
        labels:          étiquettes entières, shape (N,)
        n_classes:       nombre de classes
        n_epochs:        epochs d'entraînement du sondage linéaire
        lr:              taux d'apprentissage

    Returns:
        accuracy ∈ [0, 1]
    """
    N, dim = representations.shape

    # Classifieur linéaire simple
    classifier = nn.Linear(dim, n_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Entraînement (80% train, 20% test)
    split = int(0.8 * N)
    X_train = representations[:split].detach()
    y_train = labels[:split]
    X_test = representations[split:].detach()
    y_test = labels[split:]

    # torch.enable_grad() nécessaire si appelé depuis un contexte @no_grad
    with torch.enable_grad():
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = classifier(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = classifier(X_test).argmax(dim=-1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy


def compute_nmi(
    representations: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = None,
    n_init: int = 10,
) -> float:
    """
    Calcule le NMI (Normalized Mutual Information) entre le clustering
    k-means des représentations et les étiquettes réelles.

    Réf. math : §8.5.

    Args:
        representations: tenseur, shape (N, dim)
        labels:          étiquettes entières, shape (N,)
        n_clusters:      nombre de clusters k-means (défaut = nb classes)
        n_init:          répétitions k-means

    Returns:
        NMI ∈ [0, 1]
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        raise ImportError(
            "sklearn requis pour compute_nmi : pip install scikit-learn"
        )

    n_clusters = n_clusters or int(labels.max().item()) + 1
    X = representations.detach().float().numpy()
    y = labels.numpy()

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    nmi = normalized_mutual_info_score(y, cluster_labels)
    return float(nmi)


def prediction_success_rate(
    sdr_predicted: torch.Tensor,
    sdr_observed: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Calcule le taux de succès de la prédiction du prochain SDR.

    La prédiction est considérée comme un succès si le chevauchement
    (overlap) entre le SDR prédit et le SDR réellement observé
    dépasse le seuil `threshold`.

    overlap = |SDR_pred ∩ SDR_obs| / w
    succès  = overlap >= threshold

    Connexion biologique : correspond au signal de prédiction descendant
    (L6 → L4) dans le cadre du predictive coding (Rao-Ballard 1999).
    Un overlap élevé signifie que la colonne anticipait correctement
    l'entrée sensorielle avant de la recevoir.

    Réf. math : §8.7, Éq. 8.7.

    Args:
        sdr_predicted: SDR prédit pour le prochain pas, shape (n,), binaire
        sdr_observed:  SDR réellement observé,          shape (n,), binaire
        threshold:     seuil d'overlap pour considérer la prédiction
                       comme un succès (défaut 0.5 = 50% des bits corrects)

    Returns:
        dict avec :
            overlap:  fraction de bits prédits corrects ∈ [0, 1]
            success:  True si overlap >= threshold
            precision: |pred ∩ obs| / |pred| (évite les fausses alarmes)
            recall:    |pred ∩ obs| / |obs|  (= overlap si |pred|=|obs|=w)
            chance_level: E[overlap] sous prédiction aléatoire = |pred| / n
            overlap_lift: overlap / chance_level
            bits_correct: |pred ∩ obs|
    """
    w_obs = sdr_observed.sum().item()
    w_pred = sdr_predicted.sum().item()
    n_sdr = sdr_observed.numel()

    # overlap = |pred ∩ obs| / |obs|. Sous prédiction aléatoire, chaque bit
    # observé a une probabilité |pred| / n d'être couvert par le SDR prédit.
    chance_level = float(w_pred / n_sdr) if n_sdr > 0 else 40.0 / 2048.0

    if w_obs == 0 or w_pred == 0:
        return {
            "overlap": 0.0,
            "success": False,
            "precision": 0.0,
            "recall": 0.0,
            "chance_level": chance_level,
            "overlap_lift": 0.0,
            "bits_correct": 0,
        }

    intersection = (sdr_predicted * sdr_observed).sum().item()
    overlap = intersection / w_obs
    precision = intersection / w_pred
    recall = intersection / w_obs
    overlap_lift = overlap / max(chance_level, 1e-8)

    return {
        "overlap": float(overlap),
        "success": overlap >= threshold,
        "precision": float(precision),
        "recall": float(recall),
        "chance_level": chance_level,
        "overlap_lift": float(overlap_lift),
        "bits_correct": int(intersection),
    }


def batch_prediction_success_rate(
    sdrs_predicted: List[torch.Tensor],
    sdrs_observed: List[torch.Tensor],
    threshold: float = 0.5,
) -> dict:
    """
    Calcule le taux de succès agrégé sur une séquence de prédictions.

    Pour une séquence de N pas, compare chaque SDR prédit au pas t
    avec le SDR observé au pas t+1 (convention causale standard).

    Args:
        sdrs_predicted: liste de N SDRs prédits, chacun shape (n,)
        sdrs_observed:  liste de N SDRs observés, chacun shape (n,)
        threshold:      seuil de succès (défaut 0.5)

    Returns:
        dict avec :
            pred_success_rate : fraction de prédictions réussies ∈ [0, 1]
            mean_overlap      : overlap moyen sur tous les pas
            mean_precision    : précision moyenne
            mean_recall       : rappel moyen
            mean_overlap_lift : lift moyen vs niveau de chance
            mean_bits_correct : nombre moyen de bits correctement prédits
            success_rate_at_* : taux de pas dépassant 5%, 10%, 20% d'overlap
    """
    assert len(sdrs_predicted) == len(sdrs_observed), (
        "sdrs_predicted et sdrs_observed doivent avoir la même longueur"
    )

    results = [
        prediction_success_rate(p, o, threshold)
        for p, o in zip(sdrs_predicted, sdrs_observed)
    ]

    n = len(results)
    return {
        "pred_success_rate":      float(sum(r["success"] for r in results) / n),
        "mean_overlap":           float(sum(r["overlap"] for r in results) / n),
        "mean_precision":         float(sum(r["precision"] for r in results) / n),
        "mean_recall":            float(sum(r["recall"] for r in results) / n),
        "mean_overlap_lift":      float(sum(r["overlap_lift"] for r in results) / n),
        "mean_bits_correct":      float(sum(r["bits_correct"] for r in results) / n),
        "success_rate_at_5pct":   float(sum(r["overlap"] >= 0.05 for r in results) / n),
        "success_rate_at_10pct":  float(sum(r["overlap"] >= 0.10 for r in results) / n),
        "success_rate_at_20pct":  float(sum(r["overlap"] >= 0.20 for r in results) / n),
    }


def column_specialization_index(
    sdrs_per_column: List[List[torch.Tensor]],
) -> float:
    """
    Calcule l'indice de spécialisation des colonnes (SI).

    SI mesure à quel point chaque colonne a développé des représentations
    distinctes — un SI élevé indique une bonne spécialisation (I6.1).

    SI = 1 − mean_overlap_inter / mean_overlap_intra
    où overlap = |SDR_i ∩ SDR_j| / |SDR_i ∪ SDR_j| (Jaccard)

    Réf. math : §8.6.

    Args:
        sdrs_per_column: liste de K listes, chacune contenant N SDRs
                         de shape (n,) pour une colonne donnée

    Returns:
        SI ∈ [0, 1] (1 = spécialisation maximale, 0 = aucune)
    """
    K = len(sdrs_per_column)
    if K < 2:
        return 0.0

    def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
        inter = (a * b).sum().item()
        union = ((a + b) > 0).float().sum().item()
        return inter / union if union > 0 else 0.0

    # Overlap intra-colonne : similarité entre SDRs de la MÊME colonne
    intra_overlaps = []
    for col_sdrs in sdrs_per_column:
        N = len(col_sdrs)
        if N < 2:
            continue
        for i in range(min(N, 10)):
            for j in range(i + 1, min(N, 10)):
                intra_overlaps.append(jaccard(col_sdrs[i], col_sdrs[j]))

    # Overlap inter-colonne : similarité entre SDRs de COLONNES DIFFÉRENTES
    inter_overlaps = []
    col_means = [
        torch.stack(sdrs, dim=0).float().mean(dim=0)
        for sdrs in sdrs_per_column
    ]
    for i in range(K):
        for j in range(i + 1, K):
            inter_overlaps.append(jaccard(col_means[i], col_means[j]))

    mean_intra = sum(intra_overlaps) / len(intra_overlaps) if intra_overlaps else 0.0
    mean_inter = sum(inter_overlaps) / len(inter_overlaps) if inter_overlaps else 0.0

    si = 1.0 - (mean_inter / max(mean_intra, 1e-8))
    return float(max(0.0, si))


class UnsupervisedEvaluator:
    """
    Pipeline d'évaluation complet — 7 métriques non-supervisées.

    Métriques retournées par evaluate() :
        epsilon              : erreur de reconstruction SDR ∈ [0, 1]
        sparsity_violation_rate : fraction de pas violant I1.1 ∈ [0, 1]
        var_red              : réduction de variance du consensus ∈ [0, 1]
        lin_prob             : précision du sondage linéaire ∈ [0, 1]
        nmi                  : NMI k-means ∈ [0, 1]
        SI                   : indice de spécialisation ∈ [0, 1]
        pred_success_rate    : taux de prédictions réussies ∈ [0, 1]
        pred_mean_overlap    : overlap moyen prédit/observé ∈ [0, 1]
        pred_precision       : précision des prédictions ∈ [0, 1]
        pred_recall          : rappel des prédictions ∈ [0, 1]

    Note sur pred_success_rate :
        Si result["sdr_predicted"] est exposé par CorticalColumn.step(),
        on compare le SDR prédit explicite au SDR observé — c'est la vraie
        métrique predictive coding (Rao-Ballard).
        Sinon, on utilise SDR(t) comme proxy pour SDR(t+1) — mesure de
        cohérence séquentielle minimale.

    Usage :
        evaluator = UnsupervisedEvaluator(model, expected_w=40)
        metrics = evaluator.evaluate(inputs, velocities, labels)
    """

    def __init__(
        self,
        model,  # CorticalColumn
        expected_w: int = 40,
        n_classes: Optional[int] = None,
    ) -> None:
        self.model = model
        self.expected_w = expected_w
        self.n_classes = n_classes

    @torch.no_grad()
    def evaluate(
        self,
        inputs: torch.Tensor,
        velocities: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un ensemble de stimuli.

        Args:
            inputs:     stimuli sensoriels, shape (N, input_dim)
            velocities: vecteurs de vitesse, shape (N, 2)
            labels:     étiquettes optionnelles pour lin_prob et nmi

        Returns:
            dict avec les 6 métriques : ε, sparsity, var_red, lin_prob, nmi, SI
        """
        N = inputs.shape[0]
        all_sdrs = []           # (N, n_sdr) — première colonne
        all_grid_codes = []     # (N, 4·n_mod) — première colonne
        per_column_sdrs = [[] for _ in range(self.model.n_columns)]
        reconstruction_errors = []
        variance_reductions = []
        sparsity_violations = 0

        # Prédiction : SDR prédit au pas t comparé au SDR observé au pas t+1
        # (convention causale : la colonne prédit avant de recevoir l'entrée)
        sdrs_predicted_seq: List[torch.Tensor] = []
        sdrs_observed_seq: List[torch.Tensor] = []

        # Diagnostics consensus
        consensus_n_active_list: List[int] = []
        pairwise_overlap_list: List[float] = []

        self.model.reset()

        for t in range(N):
            result = self.model.step(inputs[t], velocities[t], train=False)

            sdr = result["sdr"]
            consensus = result["consensus"]

            # ε — erreur de reconstruction (SDR vs consensus)
            reconstruction_errors.append(reconstruction_error(consensus, sdr))

            # sparsity — vérification invariant I1.1
            stats = sdr_sparsity(sdr, self.expected_w)
            if not stats["is_valid"]:
                sparsity_violations += 1

            # var_red — réduction de variance du vote
            variance_reductions.append(
                vote_variance_reduction(result["all_sdrs"], consensus)
            )

            # pred_success — collecte prédiction si disponible dans result
            # result["sdr_predicted"] est le SDR que le modèle prédit
            # pour le prochain pas (calculé par L6b avant réception de t+1)
            if "sdr_predicted" in result:
                sdrs_predicted_seq.append(result["sdr_predicted"])
                sdrs_observed_seq.append(sdr)

            # Collecte des représentations
            all_sdrs.append(sdr)
            all_grid_codes.append(result["all_grid_codes"][0])

            for c, col_sdr in enumerate(result["all_sdrs"]):
                per_column_sdrs[c].append(col_sdr)

            # consensus diagnostics — nombre de bits actifs au pas t
            consensus_n_active_list.append(int(consensus.sum().item()))

            # overlap pairwise (Jaccard) entre toutes les paires de SDRs de colonnes
            sdrs_t = result["all_sdrs"]
            K_t = len(sdrs_t)
            pair_overlaps_t = []
            for ci in range(K_t):
                for cj in range(ci + 1, K_t):
                    a = sdrs_t[ci].float()
                    b = sdrs_t[cj].float()
                    inter = (a * b).sum().item()
                    union = ((a + b) > 0).float().sum().item()
                    pair_overlaps_t.append(inter / union if union > 0 else 0.0)
            pairwise_overlap_list.append(
                sum(pair_overlaps_t) / len(pair_overlaps_t) if pair_overlaps_t else 0.0
            )

        metrics = {
            "epsilon": float(sum(reconstruction_errors) / N),
            "sparsity_violation_rate": float(sparsity_violations / N),
            "var_red": float(sum(variance_reductions) / N),
            "consensus_mean_active": float(sum(consensus_n_active_list) / N),
            "consensus_empty_rate": float(
                sum(1 for v in consensus_n_active_list if v == 0) / N
            ),
            "mean_pairwise_column_overlap": float(sum(pairwise_overlap_list) / N),
        }

        # pred_success — taux de succès des prédictions (predictive coding)
        # Si le modèle ne fournit pas encore "sdr_predicted", on aligne
        # les SDRs consécutifs pour mesurer la cohérence temporelle :
        # dans quelle mesure le SDR au pas t prédit le SDR au pas t+1.
        if sdrs_predicted_seq:
            # Cas nominal : le modèle expose ses prédictions explicites
            pred_stats = batch_prediction_success_rate(
                sdrs_predicted_seq, sdrs_observed_seq
            )
        elif len(all_sdrs) >= 2:
            # Cas dégradé : on utilise SDR(t) comme proxy de prédiction
            # pour SDR(t+1) — mesure la cohérence séquentielle minimale
            pred_stats = batch_prediction_success_rate(
                all_sdrs[:-1], all_sdrs[1:]
            )
        else:
            pred_stats = {
                "pred_success_rate":    float("nan"),
                "mean_overlap":         float("nan"),
                "mean_precision":       float("nan"),
                "mean_recall":          float("nan"),
                "mean_overlap_lift":    float("nan"),
                "mean_bits_correct":    float("nan"),
                "success_rate_at_5pct": float("nan"),
                "success_rate_at_10pct": float("nan"),
                "success_rate_at_20pct": float("nan"),
            }

        metrics["pred_success_rate"] = pred_stats["pred_success_rate"]
        metrics["pred_mean_overlap"] = pred_stats["mean_overlap"]
        metrics["pred_precision"] = pred_stats["mean_precision"]
        metrics["pred_recall"] = pred_stats["mean_recall"]
        metrics["mean_overlap_lift"] = pred_stats.get("mean_overlap_lift", float("nan"))
        metrics["mean_bits_correct"] = pred_stats.get("mean_bits_correct", float("nan"))
        metrics["pred_success_rate_at_5pct"] = pred_stats.get("success_rate_at_5pct", float("nan"))
        metrics["pred_success_rate_at_10pct"] = pred_stats.get("success_rate_at_10pct", float("nan"))
        metrics["pred_success_rate_at_20pct"] = pred_stats.get("success_rate_at_20pct", float("nan"))

        # lin_prob et nmi — nécessitent des étiquettes
        if labels is not None:
            reprs = torch.stack(all_sdrs, dim=0).float()
            n_classes = self.n_classes or int(labels.max().item()) + 1

            metrics["lin_prob"] = linear_probing_accuracy(
                reprs, labels, n_classes=n_classes
            )

            if reprs.shape[0] >= 10:
                metrics["nmi"] = compute_nmi(reprs, labels, n_clusters=n_classes)
            else:
                metrics["nmi"] = float("nan")
        else:
            metrics["lin_prob"] = float("nan")
            metrics["nmi"] = float("nan")

        # SI — indice de spécialisation
        metrics["SI"] = column_specialization_index(per_column_sdrs)

        return metrics


def sweep_consensus_threshold(
    model,
    inputs: torch.Tensor,
    velocities: torch.Tensor,
    thresholds: List[float] = [1.0, 0.75, 0.5, 0.25],
) -> Dict[float, Dict[str, float]]:
    """
    Évalue evaluate() avec plusieurs seuils de consensus.

    Patche temporairement model.consensus.consensus_threshold pour chaque
    seuil, puis restaure la valeur originale. Permet de vérifier la
    décroissance de P_fp avec K (Thm 6.3) et l'impact du seuil sur
    epsilon et var_red.

    Réf. math : §6.3 (Thm 6.3 — décroissance exponentielle P_fp avec K).

    Args:
        model:      CorticalColumn
        inputs:     stimuli sensoriels, shape (N, input_dim)
        velocities: vecteurs de vitesse, shape (N, 2)
        thresholds: liste de seuils à évaluer (défaut [1.0, 0.75, 0.5, 0.25])

    Returns:
        dict { threshold: {"epsilon": ..., "consensus_mean_active": ..., "var_red": ...} }
    """
    original = model.consensus.consensus_threshold
    evaluator = UnsupervisedEvaluator(model)
    results: Dict[float, Dict[str, float]] = {}
    try:
        for t in thresholds:
            model.consensus.consensus_threshold = t
            m = evaluator.evaluate(inputs, velocities)
            results[t] = {
                "epsilon": m["epsilon"],
                "consensus_mean_active": m["consensus_mean_active"],
                "var_red": m["var_red"],
            }
    finally:
        model.consensus.consensus_threshold = original
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Point d'entrée CLI pour l'évaluation complète.

    Flags :
        --eval-prediction  : active Protocoles A et B
        --eval-ood         : active l'évaluation OOD
        --eval-all         : active tous les protocoles
    """
    parser = argparse.ArgumentParser(
        description="Évaluation non-supervisée du CorticalColumn World Model"
    )
    parser.add_argument("--dataset",    type=str, default="mnist")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--n_samples",  type=int, default=100,
                        help="Nombre d'images pour l'évaluation non-supervisée")
    parser.add_argument("--n_columns",  type=int, default=4)
    parser.add_argument("--n_sdr",      type=int, default=2048)
    parser.add_argument("--w",          type=int, default=40)
    parser.add_argument("--n_minicolumns", type=int, default=256)
    parser.add_argument("--k_active",   type=int, default=40)
    parser.add_argument("--n_grid_modules", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint PyTorch (.pt) — si absent, modèle non entraîné")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--device",     type=str, default="cpu")

    # ── Flags d'activation des protocoles ────────────────────────────────────
    parser.add_argument("--eval-prediction", dest="eval_prediction",
                        action="store_true",
                        help="Active les Protocoles A et B (prédiction)")
    parser.add_argument("--eval-ood", dest="eval_ood",
                        action="store_true",
                        help="Active l'évaluation OOD")
    parser.add_argument("--eval-all", dest="eval_all",
                        action="store_true",
                        help="Active tous les protocoles (--eval-prediction + --eval-ood)")

    # ── Paramètres spécifiques aux protocoles ─────────────────────────────────
    parser.add_argument("--n_steps_b",  type=int, default=10,
                        help="Pas d'inférence pour le Protocole B")
    parser.add_argument("--n_samples_b", type=int, default=50,
                        help="Échantillons pour le Protocole B (peut être < n_samples)")

    args = parser.parse_args()

    # Résolution des flags composites
    run_prediction = args.eval_prediction or args.eval_all
    run_ood        = args.eval_ood        or args.eval_all

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from column import CorticalColumn
    import torchvision
    import torchvision.transforms as T

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Construction du modèle ────────────────────────────────────────────────
    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=784,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11, 13, 17][:args.n_grid_modules],
        consensus_threshold=1.0,
    ).to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"[ERREUR] Checkpoint introuvable : {args.checkpoint}")
            print("[INFO] Lancement avec un modèle non entraîné (poids aléatoires)")
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"[OK] Checkpoint chargé : {args.checkpoint}")
    else:
        print("[INFO] Aucun checkpoint — modèle non entraîné")

    # ── Chargement des images ─────────────────────────────────────────────────
    ds = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=T.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.n_samples, shuffle=False, num_workers=0
    )
    imgs_raw, labels = next(iter(loader))
    imgs = imgs_raw.view(imgs_raw.shape[0], -1).to(device)
    n = min(args.n_samples, imgs.shape[0])
    imgs, labels = imgs[:n], labels[:n]

    v_zeros = torch.zeros(n, 2, device=device)

    # ── Évaluation non-supervisée (7 métriques) ───────────────────────────────
    print("\n[Unsupervised] Évaluation 7 métriques...")
    evaluator = UnsupervisedEvaluator(model, expected_w=args.w, n_classes=10)
    metrics = evaluator.evaluate(imgs, v_zeros, labels)
    print("  Métriques :")
    for k, v in metrics.items():
        print(f"    {k:30s} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")

    report: Dict = {"unsupervised": metrics}

    # ── Protocoles prédictifs (A + B) ─────────────────────────────────────────
    if run_prediction:
        print("\n[Protocole A + B] Évaluation prédictive...")
        from eval.prediction_eval import (
            PredictionProtocolA, PredictionProtocolB,
            _check_protocol_a, _check_protocol_b,
            plot_protocol_a, plot_protocol_b,
        )

        proto_a = PredictionProtocolA(model)
        res_a = proto_a.evaluate(imgs, n_samples=n)
        report["prediction_A"] = res_a
        for w in _check_protocol_a(res_a):
            print(w)
        plot_protocol_a(res_a, str(out_dir / "prediction_A_curve.png"))

        proto_b = PredictionProtocolB(model, n_steps=args.n_steps_b)
        n_b = min(args.n_samples_b, n)
        res_b = proto_b.evaluate(imgs[:n_b], n_samples=n_b)
        report["prediction_B"] = res_b
        for w in _check_protocol_b(res_b):
            print(w)
        plot_protocol_b(res_b, str(out_dir / "prediction_B_convergence.png"))

    # ── Évaluation OOD ────────────────────────────────────────────────────────
    if run_ood:
        print("\n[OOD] Évaluation généralisation hors distribution...")
        from eval.generalization_eval import (
            OODEvaluator,
            plot_ood_accuracy, plot_ood_reconstruction, plot_ood_robustness,
        )

        ood_eval = OODEvaluator(model, n_classes=10)
        ood_results = ood_eval.evaluate(imgs, labels, n_samples=n)
        report["ood"] = ood_results

        for w in ood_eval.check_sanity(ood_results):
            print(w)

        plot_ood_accuracy(ood_results,
                          str(out_dir / "ood_accuracy_degradation.png"))
        plot_ood_reconstruction(ood_results,
                                str(out_dir / "ood_reconstruction_degradation.png"))
        plot_ood_robustness(ood_results,
                            str(out_dir / "ood_robustness_scores.png"))

    # ── Rapport JSON global ───────────────────────────────────────────────────
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Rapport complet → {report_path}")

    if not run_prediction and not run_ood:
        print("\n[INFO] Pour activer les protocoles prédictifs et OOD :")
        print("  --eval-prediction   Protocoles A + B")
        print("  --eval-ood          Généralisation OOD")
        print("  --eval-all          Tous les protocoles")


if __name__ == "__main__":
    main()
