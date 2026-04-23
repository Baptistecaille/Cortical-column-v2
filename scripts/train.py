"""
Script d'entraînement non-supervisé — CorticalColumn World Model.

Pipeline (CLAUDE.md §3, Pseudo-algorithme global) :
    Phase 1 — Core (valider les 6 invariants) :
        SDRSpace → SpatialPooler → Layer6bTransformer
        → GridCellNetwork → DisplacementAlgebra → MultiColumnConsensus

    Ne pas sauter Phase 1 pour aller à Phase 3.

Usage :
    python scripts/train.py --n_steps 10000 --n_columns 4
    python scripts/train.py --help
"""

import sys
import os
import math
import torch
import argparse
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from column import CorticalColumn
from eval.unsupervised_eval import UnsupervisedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_synthetic_sequence(
    n_steps: int,
    input_dim: int,
    velocity_scale: float = 0.1,
    device: str = "cpu",
) -> tuple:
    """
    Génère une séquence sensorielle synthétique avec intégration de chemin.

    La séquence simule un agent se déplaçant dans un environnement 2D simple :
    - Les stimuli sont des patches gaussiens sur une grille
    - Les vitesses suivent une marche aléatoire avec inertie

    Args:
        n_steps:        nombre de pas de temps
        input_dim:      dimension sensorielle
        velocity_scale: amplitude des vitesses
        device:         dispositif PyTorch

    Returns:
        inputs:     tenseur (n_steps, input_dim)
        velocities: tenseur (n_steps, 2)
    """
    inputs = []
    velocities = []
    phase = torch.linspace(0, 2 * math.pi, input_dim)

    # Position initiale
    pos = torch.zeros(2)
    vel = torch.randn(2) * velocity_scale

    for t in range(n_steps):
        # Mise à jour de la position avec inertie
        vel = 0.9 * vel + 0.1 * torch.randn(2) * velocity_scale
        pos = pos + vel

        # Stimulus : projection sinusoïdale de la position (grille spatiale)
        step_term = t * 0.01
        stimulus = (
            0.5 * torch.sin(phase * pos[0] + step_term)
            + 0.5 * torch.cos(phase * pos[1] + step_term)
        )
        # Bruit sensoriel
        stimulus = stimulus + 0.1 * torch.randn(input_dim)

        inputs.append(stimulus)
        velocities.append(vel.clone())

    return (
        torch.stack(inputs, dim=0).to(device),
        torch.stack(velocities, dim=0).to(device),
    )


def train(
    model: CorticalColumn,
    n_steps: int = 10_000,
    input_dim: int = 128,
    eval_every: int = 1000,
    device: str = "cpu",
    log_level: str = "INFO",
) -> dict:
    """
    Boucle d'entraînement non-supervisée principale.

    L'apprentissage est entièrement hebbien — pas d'optimiseur autograd
    pour SpatialPooler. Les autres modules (Layer6b, etc.) sont figés
    en Phase 1.

    Args:
        model:      CorticalColumn à entraîner
        n_steps:    nombre de pas d'entraînement
        input_dim:  dimension sensorielle
        eval_every: fréquence d'évaluation (en pas)
        device:     dispositif PyTorch
        log_level:  niveau de logging

    Returns:
        history: dict avec les métriques au cours du temps
    """
    logger.setLevel(getattr(logging, log_level))
    model = model.to(device)
    model.reset()
    evaluator = UnsupervisedEvaluator(model, expected_w=model.columns[0].sdr_space.w)

    logger.info(f"Démarrage de l'entraînement : {n_steps} pas, {model.n_columns} colonnes")

    # Génération de la séquence synthétique
    logger.info("Génération de la séquence sensorielle synthétique...")
    inputs, velocities = generate_synthetic_sequence(n_steps, input_dim, device=device)

    history = {
        "step": [],
        "epsilon": [],
        "var_red": [],
        "sparsity_violation_rate": [],
        "gamma": [],
    }

    for t in range(n_steps):
        s_t = inputs[t]
        v_t = velocities[t]

        # Pas d'entraînement (apprentissage hebbien interne)
        result = model.step(s_t, v_t, train=True)

        # Évaluation périodique
        if (t + 1) % eval_every == 0 or t == 0:
            t_eval_start = max(0, t - 200)
            t_eval_end = min(n_steps, t + 1)

            with torch.no_grad():
                metrics = evaluator.evaluate(
                    inputs[t_eval_start:t_eval_end],
                    velocities[t_eval_start:t_eval_end],
                    labels=None,
                )

            gamma = model.columns[0].spatial_pooler.gamma()

            history["step"].append(t + 1)
            history["epsilon"].append(metrics["epsilon"])
            history["var_red"].append(metrics["var_red"])
            history["sparsity_violation_rate"].append(metrics["sparsity_violation_rate"])
            history["gamma"].append(gamma)

            logger.info(
                f"Pas {t+1:6d}/{n_steps} | "
                f"ε={metrics['epsilon']:.4f} | "
                f"var_red={metrics['var_red']:.4f} | "
                f"γ={gamma:.4f} | "
                f"sparsity_ok={metrics['sparsity_violation_rate'] < 0.05}"
            )

    logger.info("Entraînement terminé.")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement non-supervisé du CorticalColumn World Model"
    )
    parser.add_argument("--n_steps", type=int, default=5000, help="Nombre de pas")
    parser.add_argument("--n_columns", type=int, default=4, help="Nombre de colonnes K")
    parser.add_argument("--input_dim", type=int, default=128, help="Dimension sensorielle")
    parser.add_argument("--n_sdr", type=int, default=512, help="Dimension SDR")
    parser.add_argument("--w", type=int, default=20, help="Parcimonie SDR")
    parser.add_argument("--n_minicolumns", type=int, default=64, help="Minicolonnes (carré parfait)")
    parser.add_argument("--k_active", type=int, default=10, help="Colonnes actives")
    parser.add_argument("--n_grid_modules", type=int, default=4, help="Modules grid cells")
    parser.add_argument("--eval_every", type=int, default=500, help="Fréquence d'évaluation")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou cuda")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Construction du modèle
    model = CorticalColumn(
        n_columns=args.n_columns,
        input_dim=args.input_dim,
        n_sdr=args.n_sdr,
        w=args.w,
        n_minicolumns=args.n_minicolumns,
        k_active=args.k_active,
        n_grid_modules=args.n_grid_modules,
        grid_periods=[3, 5, 7, 11][:args.n_grid_modules],
        consensus_threshold=1.0,
    )

    logger.info(f"Modèle : {model.n_columns} colonnes × {args.n_minicolumns} minicolonnes")
    logger.info(f"SDR : n={args.n_sdr}, w={args.w} ({100*args.w/args.n_sdr:.1f}%)")
    logger.info(f"Grid cells : {args.n_grid_modules} modules, code={4*args.n_grid_modules}D")

    # Vérification des invariants avant entraînement
    logger.info("Lancement de la vérification des invariants...")
    os.system(f"python {os.path.dirname(os.path.abspath(__file__))}/run_invariants.py")

    # Entraînement
    history = train(
        model=model,
        n_steps=args.n_steps,
        input_dim=args.input_dim,
        eval_every=args.eval_every,
        device=args.device,
        log_level="DEBUG" if args.verbose else "INFO",
    )

    logger.info(f"Métrique finale ε = {history['epsilon'][-1]:.4f}")
    logger.info(f"Variance reduction finale = {history['var_red'][-1]:.4f}")


if __name__ == "__main__":
    main()
