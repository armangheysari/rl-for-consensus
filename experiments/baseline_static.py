
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np

# Make `src` importable when running this script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.static_policy import StaticPolicy
from src.env.consensus_env import ConsensusEnv


def run_policy(
    env: ConsensusEnv,
    policy: StaticPolicy,
    num_episodes: int,
    rng_seed: int,
) -> dict[float, list[float]]:
    throughputs_by_byz: dict[float, list[float]] = defaultdict(list)
    obs, info = env.reset(seed=rng_seed)
    for _ in range(num_episodes):
        # In a real online setting the agent does not know the current
        # Byzantine fraction, but the oracle baseline needs it. We pass it
        # from the info dict for the oracle only.
        byz = info.get("byzantine_fraction", 0.0) if policy.mode == "oracle" else None
        action = policy.act(byz)
        obs, reward, terminated, truncated, info = env.step(action)
        stats = info["stats"]
        # Round the byzantine fraction to 2 decimal places for grouping
        key = round(stats.byzantine_fraction, 2)
        throughputs_by_byz[key].append(stats.commit_rate)
        if terminated or truncated:
            obs, info = env.reset()
    return throughputs_by_byz


def aggregate(throughputs_by_byz: dict[float, list[float]]) -> tuple[list[float], list[float], list[float]]:
    keys = sorted(throughputs_by_byz.keys())
    means = [float(np.mean(throughputs_by_byz[k])) for k in keys]
    stds = [float(np.std(throughputs_by_byz[k])) for k in keys]
    return keys, means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/baseline.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    committee_sizes = (3, 5, 7, 9)
    env = ConsensusEnv(
        committee_sizes=committee_sizes,
        max_byzantine_fraction=0.30,
        requests_per_window=10,
        safety_penalty=10.0,
        rng_seed=args.seed,
    )

    results = {}
    for mode in ("smallest", "largest", "oracle"):
        policy = StaticPolicy(committee_sizes=committee_sizes, mode=mode)
        env.reset(seed=args.seed)
        results[mode] = run_policy(env, policy, args.episodes, args.seed)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = {"smallest": "#ef4444", "largest": "#3b82f6", "oracle": "#10b981"}
    labels = {
        "smallest": "smallest committee (n=3)",
        "largest": "largest committee (n=9)",
        "oracle": "oracle (smallest safe)",
    }
    for mode in ("smallest", "largest", "oracle"):
        keys, means, stds = aggregate(results[mode])
        if not keys:
            continue
        ax.errorbar(
            keys,
            means,
            yerr=stds,
            label=labels[mode],
            color=colors[mode],
            marker="o",
            capsize=3,
        )
    ax.set_xlabel("Byzantine fraction")
    ax.set_ylabel("Commit rate (throughput)")
    ax.set_title("Static policies vs fault rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    # Also print a quick summary table to stdout
    print("\n=== Summary (mean commit rate) ===")
    print(f"{'byz':>6} | " + " | ".join(f"{m:>12}" for m in ("smallest", "largest", "oracle")))
    all_keys = sorted(set().union(*[set(results[m].keys()) for m in results]))
    for k in all_keys:
        row = f"{k:>6.2f} | "
        for m in ("smallest", "largest", "oracle"):
            vals = results[m].get(k, [])
            mean = float(np.mean(vals)) if vals else 0.0
            row += f"{mean:>12.3f} | "
        print(row)


if __name__ == "__main__":
    main()
