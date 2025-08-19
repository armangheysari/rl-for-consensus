
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.agents.ppo_controller import RewardLoggerCallback
from src.env.consensus_env_advanced import ConsensusEnvAdvanced


class ConsensusFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO controller on ConsensusEnvAdvanced."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=50_000,
        help="Total environment timesteps for training (default: 50000)"
    )
    parser.add_argument(
        "--output", type=str, default="models/ppo_consensus_advanced",
        help="Path to save the trained model (without extension)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--ent-coef", type=float, default=0.08,
        help="Entropy coefficient (higher = more exploration)"
    )
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument(
        "--attacks", nargs="*", default=None,
        help="List of attack types to attach: collusion equivocation timing"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Minibatch size for PPO updates"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="Number of epochs per PPO update"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Training PPO controller on ConsensusEnvAdvanced (v2: deep net)")
    print("=" * 70)
    print(f"  total_timesteps:  {args.total_timesteps}")
    print(f"  learning_rate:     {args.learning_rate}")
    print(f"  n_steps:            {args.n_steps}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  n_epochs:           {args.n_epochs}")
    print(f"  gamma:              {args.gamma}")
    print(f"  ent_coef:           {args.ent_coef} (increased for exploration)")
    print(f"  seed:               {args.seed}")
    print(f"  output:             {args.output}")
    print(f"  attack_types:       {args.attacks or 'none (default Byzantine behavior)'}")
    print(f"  action_space:       MultiDiscrete([4, 4])  (committee_size, view_change_timeout)")
    print(f"  observation_space:  Box(0, 1, (8,))  (enriched with moving averages)")
    print(f"  network:            [128, 128, 64] ReLU (deeper than default [64, 64])")
    print(f"  reward:             variable safety penalty (scales with violation magnitude)")
    print("=" * 70)
    print()

    env = ConsensusEnvAdvanced(
        committee_sizes=(3, 5, 7, 9),
        view_change_timeouts=(5, 10, 15, 20),
        max_byzantine_fraction=0.30,
        requests_per_window=10,
        safety_penalty_base=2.0,
        view_change_penalty=0.02,
        attack_types=args.attacks,
        rng_seed=args.seed,
    )

    vec_env = DummyVecEnv([lambda: env])

    # Custom policy kwargs: deeper network + the new feature extractor
    policy_kwargs = {
        "features_extractor_class": ConsensusFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        # The policy head and value head will use the default MLP on top
        # of the 64-dim features.
        "net_arch": [64],  # one hidden layer of 64 in the policy/value heads
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=args.seed,
    )

    print("Starting training...")
    callback = RewardLoggerCallback(log_interval=args.log_interval, verbose=1)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to {output_path}.zip")
    print()
    print("Next steps:")
    print(f"  Evaluate:")
    print(f"    python experiments/evaluate_advanced.py --ppo-model {output_path}")


if __name__ == "__main__":
    main()
