
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

from src.agents.ppo_controller import PPOController
from src.agents.static_policy import StaticPolicy
from src.env.consensus_env import ConsensusEnv


def run_static_policy(
    env: ConsensusEnv,
    policy: StaticPolicy,
    num_episodes: int,
    rng_seed: int,
) -> dict[float, list[float]]:
    throughputs: dict[float, list[float]] = defaultdict(list)
    obs, info = env.reset(seed=rng_seed)
    for _ in range(num_episodes):
        byz = info.get("byzantine_fraction", 0.0) if policy.mode == "oracle" else None
        action = policy.act(byz)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        key = round(stats.byzantine_fraction, 2)
        throughputs[key].append(stats.commit_rate)
        if terminated or truncated:
            obs, info = env.reset()
    return throughputs


def run_ppo_policy(
    env: ConsensusEnv,
    controller: PPOController,
    num_episodes: int,
    rng_seed: int,
) -> dict[float, list[float]]:
    throughputs: dict[float, list[float]] = defaultdict(list)
    obs, info = env.reset(seed=rng_seed)
    for _ in range(num_episodes):
        action = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        key = round(stats.byzantine_fraction, 2)
        throughputs[key].append(stats.commit_rate)
        if terminated or truncated:
            obs, info = env.reset()
    return throughputs


def aggregate(
    throughputs: dict[float, list[float]],
) -> tuple[list[float], list[float], list[float]]:
    keys = sorted(throughputs.keys())
    means = [float(np.mean(throughputs[k])) for k in keys]
    stds = [float(np.std(throughputs[k])) for k in keys]
    return keys, means, stds


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs static baselines.")
    parser.add_argument(
        "--ppo-model", type=str, required=True,
        help="Path to the trained PPO model (without .zip extension)"
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ppo_vs_baseline.png")
    parser.add_argument("--summary", type=str, default="results/ppo_summary.txt")
    args = parser.parse_args()

    committee_sizes = (3, 5, 7, 9)

    # Build one env per policy so the RNG state does not leak between them.
    results: dict[str, dict[float, list[float]]] = {}

    print("Running static baselines...")
    for mode in ("smallest", "largest", "oracle"):
        env = ConsensusEnv(
            committee_sizes=committee_sizes,
            max_byzantine_fraction=0.30,
            requests_per_window=10,
            safety_penalty=10.0,
            rng_seed=args.seed,
        )
        policy = StaticPolicy(committee_sizes=committee_sizes, mode=mode)
        env.reset(seed=args.seed)
        results[mode] = run_static_policy(env, policy, args.episodes, args.seed)
        print(f"  {mode}: {sum(len(v) for v in results[mode].values())} windows")

    print(f"\nRunning PPO controller from {args.ppo_model}.zip ...")
    env = ConsensusEnv(
        committee_sizes=committee_sizes,
        max_byzantine_fraction=0.30,
        requests_per_window=10,
        safety_penalty=10.0,
        rng_seed=args.seed,
    )
    controller = PPOController.load(args.ppo_model, env)
    env.reset(seed=args.seed)
    results["ppo"] = run_ppo_policy(env, controller, args.episodes, args.seed)
    print(f"  ppo: {sum(len(v) for v in results['ppo'].values())} windows")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = {
        "smallest": "#ef4444",
        "largest": "#3b82f6",
        "oracle": "#10b981",
        "ppo": "#a855f7",
    }
    labels = {
        "smallest": "smallest committee (n=3, static)",
        "largest":  "largest committee (n=9, static)",
        "oracle":  "oracle (smallest safe, upper bound)",
        "ppo":     "PPO controller (learned)",
    }
    markers = {"smallest": "o", "largest": "s", "oracle": "^", "ppo": "D"}
    for mode in ("smallest", "largest", "oracle", "ppo"):
        keys, means, stds = aggregate(results[mode])
        if not keys:
            continue
        ax.errorbar(
            keys,
            means,
            yerr=stds,
            label=labels[mode],
            color=colors[mode],
            marker=markers[mode],
            capsize=3,
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("Byzantine fault fraction", fontsize=12)
    ax.set_ylabel("Commit rate (throughput)", fontsize=12)
    ax.set_title("PBFT committee-size control: PPO vs static baselines", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")

    # ---- Text summary ----
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("PPO vs static baselines — summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Episodes per policy: {args.episodes}\n")
        f.write(f"RNG seed: {args.seed}\n")
        f.write(f"Committee sizes: {committee_sizes}\n")
        f.write(f"Max byzantine fraction: 0.30\n\n")
        f.write("Mean commit rate by byzantine fraction:\n")
        f.write(f"{'byz':>6} | {'smallest':>10} | {'largest':>10} | {'oracle':>10} | {'ppo':>10}\n")
        f.write("-" * 60 + "\n")
        all_keys = sorted(
            set().union(*[set(results[m].keys()) for m in results])
        )
        for k in all_keys:
            row = f"{k:>6.2f} | "
            for m in ("smallest", "largest", "oracle", "ppo"):
                vals = results[m].get(k, [])
                mean = float(np.mean(vals)) if vals else 0.0
                row += f"{mean:>10.3f} | "
            f.write(row + "\n")
        f.write("\n")
        # Overall stats
        f.write("Overall mean commit rate (across all byzantine fractions):\n")
        for m in ("smallest", "largest", "oracle", "ppo"):
            all_vals = [v for vals in results[m].values() for v in vals]
            mean = float(np.mean(all_vals)) if all_vals else 0.0
            f.write(f"  {m:>10}: {mean:.3f}\n")
    print(f"Summary saved to {args.summary}")
    print()
    # Also print summary to stdout
    with open(args.summary) as f:
        print(f.read())


if __name__ == "__main__":
    main()
