"""
Pipeline d'évaluation non-supervisée — 6 métriques.

Métriques :
    ε        : erreur de reconstruction (fidélité SDR)
    sparsity : parcimonie L2/3 (doit respecter le seuil sur t_max)
    var_red  : réduction de variance par le vote inter-colonnes
    lin_prob : précision de sondage linéaire (linear probing)
    nmi      : NMI k-means sur les représentations
    SI       : indice de spécialisation des colonnes

Ces 6 métriques correspondent à l'évaluation non-supervisée définie dans
CLAUDE.md §3 et le pseudo-algorithme global.

Réf. math : §8 — Formalisation_Mille_Cerveaux.pdf
"""

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
    Pipeline d'évaluation complet — 6 métriques non-supervisées.

    Usage :
        evaluator = UnsupervisedEvaluator(model, expected_w=40)
        metrics = evaluator.evaluate(dataloader)
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

            # Collecte des représentations
            all_sdrs.append(sdr)
            all_grid_codes.append(result["all_grid_codes"][0])

            for c, col_sdr in enumerate(result["all_sdrs"]):
                per_column_sdrs[c].append(col_sdr)

        metrics = {
            "epsilon": float(sum(reconstruction_errors) / N),
            "sparsity_violation_rate": float(sparsity_violations / N),
            "var_red": float(sum(variance_reductions) / N),
        }

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
