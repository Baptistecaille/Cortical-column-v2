"""
Pipeline de benchmark — Comparaison avec JEPA (I-JEPA, V-JEPA 2, Point-JEPA).

Datasets cibles (CLAUDE.md §1) :
    - CO3Dv2             : invariance rotation (avantage TBT)
    - CLEVR              : compositionnalité (avantage TBT)
    - Something-Something v2 : motion (terrain V-JEPA)
    - Ego4D spatial memory   : navigation (avantage TBT exclusif)
    - ImageNet-1K            : baseline représentation

Ce fichier fournit le squelette d'évaluation benchmark.
Les dataloaders spécifiques à chaque dataset doivent être fournis
par l'utilisateur selon la disponibilité locale des données.

Réf. math : §9 — Formalisation_Mille_Cerveaux.pdf
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration d'un benchmark."""
    name: str
    dataset_path: str
    metric_name: str
    higher_is_better: bool = True
    description: str = ""


# Benchmarks définis dans CLAUDE.md §1
BENCHMARKS = [
    BenchmarkConfig(
        name="CO3Dv2",
        dataset_path="data/co3dv2",
        metric_name="rotation_invariance_acc",
        description="Invariance rotation — avantage TBT sur JEPA",
    ),
    BenchmarkConfig(
        name="CLEVR",
        dataset_path="data/clevr",
        metric_name="compositional_accuracy",
        description="Compositionnalité spatiale — avantage TBT",
    ),
    BenchmarkConfig(
        name="SomethingSomethingV2",
        dataset_path="data/ssv2",
        metric_name="top1_accuracy",
        description="Motion understanding — terrain V-JEPA",
    ),
    BenchmarkConfig(
        name="Ego4D",
        dataset_path="data/ego4d",
        metric_name="spatial_recall_at_1",
        description="Navigation spatiale — avantage TBT exclusif",
    ),
    BenchmarkConfig(
        name="ImageNet1K",
        dataset_path="data/imagenet",
        metric_name="top1_accuracy",
        description="Baseline représentation visuelle",
    ),
]


class BenchmarkRunner:
    """
    Exécuteur de benchmarks pour le CorticalColumn vs JEPA.

    Usage :
        runner = BenchmarkRunner(cortical_model)
        results = runner.run_clevr(dataloader)
        runner.print_comparison(results)
    """

    def __init__(
        self,
        model,  # CorticalColumn
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract_representations(
        self,
        dataloader: Any,
        max_samples: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Extrait les représentations du modèle sur un dataloader.

        Args:
            dataloader: itérable yielding (input, label) ou (input, velocity, label)
            max_samples: nombre maximum d'échantillons

        Returns:
            dict avec 'representations' (N, dim), 'labels' (N,)
        """
        self.model.eval()
        self.model.reset()

        all_reprs = []
        all_labels = []
        count = 0

        for batch in dataloader:
            if count >= max_samples:
                break

            # Format : (input, label) ou (input, velocity, label)
            if len(batch) == 2:
                inputs, labels = batch
                velocities = torch.zeros(inputs.shape[0], 2, device=self.device)
            else:
                inputs, velocities, labels = batch

            inputs = inputs.to(self.device)
            velocities = velocities.to(self.device)

            for i in range(inputs.shape[0]):
                if count >= max_samples:
                    break
                result = self.model.step(inputs[i], velocities[i], train=False)

                # Concaténation SDR + grid code comme représentation
                repr_vec = torch.cat([
                    result["sdr"].float(),
                    result["all_grid_codes"][0].float(),
                ])
                all_reprs.append(repr_vec)
                all_labels.append(labels[i] if labels.dim() == 1 else labels[i].item())
                count += 1

        return {
            "representations": torch.stack(all_reprs, dim=0),
            "labels": torch.tensor(all_labels),
        }

    def run_linear_probe(
        self,
        train_reprs: torch.Tensor,
        train_labels: torch.Tensor,
        test_reprs: torch.Tensor,
        test_labels: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """
        Évalue les représentations par sondage linéaire (protocole standard JEPA).

        Args:
            train_reprs: représentations d'entraînement, shape (N_train, dim)
            train_labels: étiquettes d'entraînement, shape (N_train,)
            test_reprs: représentations de test, shape (N_test, dim)
            test_labels: étiquettes de test, shape (N_test,)
            n_epochs: epochs du sondage linéaire
            lr: taux d'apprentissage

        Returns:
            top1_accuracy ∈ [0, 1]
        """
        import torch.nn.functional as F

        n_classes = int(max(train_labels.max().item(), test_labels.max().item())) + 1
        dim = train_reprs.shape[-1]

        classifier = nn.Linear(dim, n_classes).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

        train_reprs = train_reprs.detach().to(self.device)
        train_labels = train_labels.long().to(self.device)

        for epoch in range(n_epochs):
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(train_reprs)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            test_reprs = test_reprs.to(self.device)
            test_labels = test_labels.long().to(self.device)
            preds = classifier(test_reprs).argmax(dim=-1)
            accuracy = (preds == test_labels).float().mean().item()

        return accuracy

    def run_rotation_invariance(
        self,
        dataloader_views: Any,
        n_views: int = 4,
    ) -> float:
        """
        Évalue l'invariance à la rotation (benchmark CO3Dv2).

        Pour chaque objet, génère n_views rotations et mesure la
        similarité des représentations entre vues.

        Args:
            dataloader_views: yield (views_list, label), views_list = [view_i]
            n_views: nombre de vues par objet

        Returns:
            invariance_score ∈ [0, 1] (1 = représentations identiques)
        """
        self.model.eval()
        scores = []

        for views, label in dataloader_views:
            view_reprs = []
            for view in views[:n_views]:
                self.model.reset()
                v_t = torch.zeros(2, device=self.device)
                result = self.model.step(
                    view.to(self.device), v_t, train=False
                )
                view_reprs.append(result["grid_code"].float())

            if len(view_reprs) < 2:
                continue

            # Similarité cosinus entre toutes les paires de vues
            pairs = 0
            total_sim = 0.0
            for i in range(len(view_reprs)):
                for j in range(i + 1, len(view_reprs)):
                    sim = torch.nn.functional.cosine_similarity(
                        view_reprs[i].unsqueeze(0),
                        view_reprs[j].unsqueeze(0),
                    ).item()
                    total_sim += sim
                    pairs += 1

            if pairs > 0:
                scores.append(total_sim / pairs)

        return float(sum(scores) / len(scores)) if scores else 0.0

    def print_comparison(
        self,
        results: Dict[str, float],
        jepa_baselines: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Affiche un tableau comparatif CorticalColumn vs JEPA.

        Args:
            results:         résultats du CorticalColumn
            jepa_baselines:  résultats JEPA de référence (optionnel)
        """
        print("\n" + "=" * 60)
        print("BENCHMARK — CorticalColumn vs JEPA")
        print("=" * 60)

        header = f"{'Métrique':<30} {'CorticalColumn':>15}"
        if jepa_baselines:
            header += f" {'JEPA':>10} {'Δ':>8}"
        print(header)
        print("-" * 60)

        for metric, value in results.items():
            row = f"{metric:<30} {value:>15.4f}"
            if jepa_baselines and metric in jepa_baselines:
                jepa_val = jepa_baselines[metric]
                delta = value - jepa_val
                sign = "+" if delta >= 0 else ""
                row += f" {jepa_val:>10.4f} {sign}{delta:>7.4f}"
            print(row)

        print("=" * 60 + "\n")
