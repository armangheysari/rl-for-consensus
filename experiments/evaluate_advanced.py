
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.consensus_env_advanced import ConsensusEnvAdvanced


def run_static(env_factory, policy_fn, num_episodes, rng_seed):
    throughputs: dict[float, list[float]] = defaultdict(list)
    env = env_factory()
    obs, info = env.reset(seed=rng_seed)
    for _ in range(num_episodes):
        byz = info.get("byzantine_fraction", 0.0)
        action = policy_fn(byz)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        key = round(stats.byzantine_fraction, 2)
        throughputs[key].append(stats.commit_rate)
        if terminated or truncated:
            obs, info = env.reset()
    return throughputs


def run_ppo(env_factory, model_path, num_episodes, rng_seed):
    throughputs: dict[float, list[float]] = defaultdict(list)
    env = env_factory()
    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env)
    obs, info = env.reset(seed=rng_seed)
    for _ in range(num_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        key = round(stats.byzantine_fraction, 2)
        throughputs[key].append(stats.commit_rate)
        if terminated or truncated:
            obs, info = env.reset()
    return throughputs


def aggregate(throughputs):
    keys = sorted(throughputs.keys())
    means = [float(np.mean(throughputs[k])) for k in keys]
    stds = [float(np.std(throughputs[k])) for k in keys]
    return keys, means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the plot. If not given, derives from --ppo-model and --attacks."
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Path to save the summary. If not given, derives from --ppo-model and --attacks."
    )
    parser.add_argument(
        "--attacks", nargs="*", default=None,
        help="Optional attack types to enable"
    )
    args = parser.parse_args()

    # Derive output paths if not given
    model_basename = os.path.basename(args.ppo_model)
    attacks_suffix = "_attacks" if args.attacks else "_noattacks"
    default_output = f"results/{model_basename}{attacks_suffix}.png"
    default_summary = f"results/{model_basename}{attacks_suffix}_summary.txt"
    output_path = args.output or default_output
    summary_path = args.summary or default_summary

    committee_sizes = (3, 5, 7, 9)
    view_change_timeouts = (5, 10, 15, 20)

    def env_factory():
        return ConsensusEnvAdvanced(
            committee_sizes=committee_sizes,
            view_change_timeouts=view_change_timeouts,
            max_byzantine_fraction=0.30,
            requests_per_window=10,
            safety_penalty_base=2.0,
            view_change_penalty=0.02,
            attack_types=args.attacks,
            rng_seed=args.seed,
        )

    # Static policies
    policies = {
        "smallest_aggressive": lambda byz: np.array([0, 0]),  # n=3, vct=5
        "largest_conservative": lambda byz: np.array([3, 3]),  # n=9, vct=20
        "oracle": lambda byz: (
            # Smallest committee that tolerates the fault
            np.array([
                next((i for i, n in enumerate(committee_sizes)
                      if n >= 3 * int(byz * n) + 1), len(committee_sizes) - 1),
                2  # middle VCT
            ])
        ),
    }

    results = {}
    print("Running static baselines...")
    for name, fn in policies.items():
        results[name] = run_static(env_factory, fn, args.episodes, args.seed)
        print(f"  {name}: {sum(len(v) for v in results[name].values())} windows")

    print(f"\nRunning PPO controller from {args.ppo_model}.zip ...")
    results["ppo"] = run_ppo(env_factory, args.ppo_model, args.episodes, args.seed)
    print(f"  ppo: {sum(len(v) for v in results['ppo'].values())} windows")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = {
        "smallest_aggressive": "#ef4444",
        "largest_conservative": "#3b82f6",
        "oracle": "#10b981",
        "ppo": "#a855f7",
    }
    labels = {
        "smallest_aggressive": "smallest (n=3, vct=5)",
        "largest_conservative": "largest (n=9, vct=20)",
        "oracle": "oracle (smallest safe + mid VCT)",
        "ppo": "PPO controller (learned, deep net)",
    }
    markers = {
        "smallest_aggressive": "o",
        "largest_conservative": "s",
        "oracle": "^",
        "ppo": "D",
    }
    for name in ("smallest_aggressive", "largest_conservative", "oracle", "ppo"):
        keys, means, stds = aggregate(results[name])
        if not keys:
            continue
        ax.errorbar(
            keys, means, yerr=stds,
            label=labels[name], color=colors[name],
            marker=markers[name], capsize=3, linewidth=2, markersize=7,
        )
    attacks_str = f" + {args.attacks}" if args.attacks else ""
    ax.set_xlabel("Byzantine fault fraction", fontsize=12)
    ax.set_ylabel("Commit rate (throughput)", fontsize=12)
    ax.set_title(f"PBFT multi-action control (v2: deep net + variable penalty){attacks_str}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

    # Summary
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("PPO (advanced v2) vs static baselines — summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Episodes per policy: {args.episodes}\n")
        f.write(f"RNG seed: {args.seed}\n")
        f.write(f"Committee sizes: {committee_sizes}\n")
        f.write(f"View-change timeouts: {view_change_timeouts}\n")
        f.write(f"Max byzantine fraction: 0.30\n")
        f.write(f"Attack types: {args.attacks or 'none (default Byzantine)'}\n")
        f.write(f"Network: [128, 128, 64] ReLU (custom feature extractor)\n")
        f.write(f"Reward: variable safety penalty (scales with violation magnitude)\n\n")
        f.write("Mean commit rate by byzantine fraction:\n")
        cols = ("smallest_aggressive", "largest_conservative", "oracle", "ppo")
        f.write(f"{'byz':>6} | " + " | ".join(f"{c:>22}" for c in cols) + "\n")
        f.write("-" * 110 + "\n")
        all_keys = sorted(set().union(*[set(results[m].keys()) for m in cols]))
        for k in all_keys:
            row = f"{k:>6.2f} | "
            for m in cols:
                vals = results[m].get(k, [])
                mean = float(np.mean(vals)) if vals else 0.0
                row += f"{mean:>22.3f} | "
            f.write(row + "\n")
        f.write("\nOverall mean commit rate (across all byzantine fractions):\n")
        for m in cols:
            all_vals = [v for vals in results[m].values() for v in vals]
            mean = float(np.mean(all_vals)) if all_vals else 0.0
            f.write(f"  {m:>22}: {mean:.3f}\n")
    print(f"Summary saved to {summary_path}")
    print()
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
