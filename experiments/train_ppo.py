
from __future__ import annotations

import argparse
import os
import sys

# Make `src` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.ppo_controller import PPOController
from src.env.consensus_env import ConsensusEnv


def main():
    parser = argparse.ArgumentParser(description="Train a PPO controller for ConsensusEnv.")
    parser.add_argument(
        "--total-timesteps", type=int, default=50_000,
        help="Total environment timesteps for training (default: 50000)"
    )
    parser.add_argument(
        "--output", type=str, default="models/ppo_consensus",
        help="Path to save the trained model (without extension)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="PPO rollout buffer size in steps"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient for exploration"
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Print progress every N episodes"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Training PPO controller on ConsensusEnv")
    print("=" * 60)
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  learning_rate:  {args.learning_rate}")
    print(f"  n_steps:        {args.n_steps}")
    print(f"  gamma:          {args.gamma}")
    print(f"  ent_coef:       {args.ent_coef}")
    print(f"  seed:           {args.seed}")
    print(f"  output:         {args.output}")
    print("=" * 60)
    print()

    committee_sizes = (3, 5, 7, 9)
    env = ConsensusEnv(
        committee_sizes=committee_sizes,
        max_byzantine_fraction=0.30,
        requests_per_window=10,
        safety_penalty=10.0,
        rng_seed=args.seed,
    )

    controller = PPOController(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=args.seed,
    )

    print("Starting training...")
    controller.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
    )

    controller.save(args.output)
    print(f"\nModel saved to {args.output}.zip")
    print()
    print("Next steps:")
    print(f"  Evaluate against baselines:")
    print(f"    python experiments/evaluate.py --ppo-model {args.output}")
    print(f"  This will produce results/ppo_vs_baseline.png")


if __name__ == "__main__":
    main()
