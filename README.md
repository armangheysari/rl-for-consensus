# rl-for-consensus

A reinforcement learning agent that tunes PBFT parameters (committee size,
view-change timeout, reconfiguration threshold) in response to observed
fault patterns. The simulator implements the three-phase pre-prepare /
prepare / commit protocol from Castro & Liskov (OSDI 1999) with view
changes and Byzantine workers.

The work builds on my MSc thesis on Byzantine fault-tolerant SoC design
and two patents on per-core PBFT consensus. The longer research arc is
in the blog post at [armangheysari.github.io](https://armangheysari.github.io/#blog).

## Layout

* `src/pbft/`, PBFT simulator. Each node runs the three-phase commit
  protocol with view changes. Configurable fraction of Byzantine peers.
* `src/env/`, Gymnasium environment wrapping the simulator. The agent
  observes commit latency and fault counters, picks committee
  configurations. Reward is throughput under a hard safety constraint.
* `src/agents/`, PPO controller (Stable Baselines3) and static baseline
  policies for comparison.
* `experiments/`, training and evaluation entry points.

## The problem

PBFT has three knobs operators set statically: committee size `n`,
view-change timeout `T_v`, and reconfiguration threshold `R`. The right
values depend on the fault pattern. Under a low fault rate, larger
committees waste throughput. Under a high fault rate, smaller committees
violate the `n >= 3f + 1` safety bound. The operator cannot know the
fault pattern in advance, and the optimal setting changes as the
adversary adapts.

The standard BFT answer is to pick conservative values and live with
the throughput hit. The RL framing asks whether a learned controller
can do better, and do so safely.

## Status

Three phases done.

* PBFT simulator with view changes (Castro & Liskov, OSDI 1999).
* Gymnasium environment with single-action (committee size) and
  multi-action (committee size + view-change timeout) variants.
* Three static baselines: smallest, largest, oracle.
* PPO controller trained with Stable Baselines3.
* Three Byzantine attack patterns: collusion, equivocation, timing.
* Evaluation scripts producing comparison plots.

## Results

Phase 2, single-action (committee size only). 40k PPO timesteps on CPU.

| Policy | Mean commit rate | Notes |
|---|---|---|
| smallest (always n=3) | 0.66 | Fails under any Byzantine fault |
| largest (always n=9) | 0.81 | Safe but slow from O(n^2) message overhead |
| oracle (smallest safe) | 0.66 | Upper bound, requires knowing f |
| PPO controller | 0.90 | Adapts committee size to fault rate |

Phase 3 v2, multi-action with attacks enabled (collusion, equivocation,
timing). Variable safety penalty scaling with violation magnitude, 8D
observation with moving averages, custom [128, 128, 64] feature
extractor.

| Policy | Without attacks | With attacks |
|---|---|---|
| smallest_aggressive (n=3, vct=5) | 0.66 | 0.64 |
| largest_conservative (n=9, vct=20) | 0.81 | 0.80 |
| oracle (smallest safe + mid VCT) | 0.66 | 0.64 |
| PPO controller (v2) | 0.81 | 0.85 |

The v2 reward signal is positive and stable (mean window reward around
0.2, versus v1 at -4 to -6). The moving-average features let the agent
detect persistent adversaries and pick larger committees proactively.

Plots: `results/ppo_vs_baseline.png` (Phase 2),
`results/ppo_consensus_advanced_v2_noattacks.png` and
`results/ppo_consensus_attacks_v2_attacks.png` (Phase 3).

## Requirements

Python 3.10+, gymnasium >= 0.29, stable-baselines3 >= 2.2, torch >= 2.1,
numpy, matplotlib.

## Quickstart

```bash
pip install -r requirements.txt

# Tests
python3 tests/test_simulator.py
python3 tests/test_env.py
python3 tests/test_attacks.py
python3 tests/test_env_advanced.py

# Phase 2: train PPO on the basic env
python3 experiments/train_ppo.py --total-timesteps 40000 --output models/ppo_consensus
python3 experiments/evaluate.py --ppo-model models/ppo_consensus

# Phase 3: train PPO on the advanced env with attacks
python3 experiments/train_ppo_advanced.py --total-timesteps 30000 \
    --output models/ppo_consensus_attacks_v2 \
    --attacks collusion equivocation timing
python3 experiments/evaluate_advanced.py --ppo-model models/ppo_consensus_attacks_v2 \
    --attacks collusion equivocation timing
```

## License

MIT. See `LICENSE`.

## References

* L. Lamport, R. Shostak, M. Pease. The Byzantine Generals Problem.
  ACM TOPLAS, 1982.
* M. Castro, B. Liskov. Practical Byzantine Fault Tolerance. OSDI, 1999.

## Author

Arman Gheysari, [armangheysari.github.io](https://armangheysari.github.io)
