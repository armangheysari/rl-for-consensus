
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..pbft.attacks import (
    ATTACK_REGISTRY,
    ByzantineAttack,
    CollusionAttack,
    EquivocationAttack,
    TimingAttack,
)
from ..pbft.simulator import PBFTSimulator, EpisodeStats
from .consensus_env import ConsensusEnv


# Default action space dimensions
DEFAULT_COMMITTEE_SIZES: tuple[int, ...] = (3, 5, 7, 9)
DEFAULT_VIEW_CHANGE_TIMEOUTS: tuple[int, ...] = (5, 10, 15, 20)

# History window for moving averages
HISTORY_WINDOW: int = 10


class ConsensusEnvAdvanced(ConsensusEnv):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        committee_sizes: tuple[int, ...] = DEFAULT_COMMITTEE_SIZES,
        view_change_timeouts: tuple[int, ...] = DEFAULT_VIEW_CHANGE_TIMEOUTS,
        max_byzantine_fraction: float = 0.30,
        requests_per_window: int = 10,
        safety_penalty_base: float = 2.0,
        view_change_penalty: float = 0.02,
        attack_types: list[str] | None = None,
        rng_seed: int | None = None,
    ):
        if attack_types is not None:
            for atk in attack_types:
                if atk not in ATTACK_REGISTRY:
                    raise ValueError(
                        f"unknown attack type: {atk}. Valid: {list(ATTACK_REGISTRY.keys())}"
                    )
        if max_byzantine_fraction >= 1.0 / 3.0:
            raise ValueError(
                "max_byzantine_fraction must be < 1/3 for PBFT safety; "
                f"got {max_byzantine_fraction}"
            )
        self.committee_sizes = committee_sizes
        self.view_change_timeouts = view_change_timeouts
        self.max_byzantine_fraction = max_byzantine_fraction
        self.requests_per_window = requests_per_window
        self.safety_penalty_base = safety_penalty_base
        self.view_change_penalty = view_change_penalty
        self.attack_types = attack_types
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed

        # Multi-discrete action space: (committee_size_idx, view_change_timeout_idx)
        self.action_space = spaces.MultiDiscrete(
            [len(committee_sizes), len(view_change_timeouts)]
        )
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )
        self._last_stats: EpisodeStats | None = None
        self._window_index: int = 0
        self._current_simulator: PBFTSimulator | None = None

        # History for moving averages
        self._byz_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
        self._latency_history: deque[float] = deque(maxlen=HISTORY_WINDOW)
        self._last_had_violation: int = 0

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rng_seed = seed
        self._window_index = 0
        self._last_stats = None
        self._byz_history.clear()
        self._latency_history.clear()
        self._last_had_violation = 0
        obs = np.zeros(8, dtype=np.float32)
        info = {"window": 0}
        return obs, info

    def step(self, action):
        # Handle both single-action (legacy) and multi-discrete
        if isinstance(action, (list, tuple, np.ndarray)):
            committee_idx = int(action[0])
            vct_idx = int(action[1])
        else:
            committee_idx = int(action)
            vct_idx = len(self.view_change_timeouts) // 2

        committee_size = self.committee_sizes[committee_idx]
        view_change_timeout = self.view_change_timeouts[vct_idx]

        # Sample byzantine fraction
        byzantine_fraction = float(self.rng.uniform(0.0, self.max_byzantine_fraction))
        n = committee_size
        actual_byz_count = int(self.rng.binomial(n=n, p=byzantine_fraction))
        max_safe_f = (n - 1) // 3
        # Magnitude of safety violation: how many Byzantine workers exceed the
        # safe bound. 0 means safe; >0 means unsafe.
        violation_magnitude = max(0, actual_byz_count - max_safe_f)
        is_unsafe = violation_magnitude > 0

        byzantine_indices: set[int] = set()
        if actual_byz_count > 0:
            byzantine_indices = set(
                self.rng.choice(n, size=actual_byz_count, replace=False).tolist()
            )

        # Attach attacks to Byzantine nodes if configured
        node_attacks: dict[int, ByzantineAttack] | None = None
        if self.attack_types and byzantine_indices:
            node_attacks = {}
            for node_id in byzantine_indices:
                atk_name = self.attack_types[
                    int(self.rng.integers(0, len(self.attack_types)))
                ]
                atk_class = ATTACK_REGISTRY[atk_name]
                if atk_name == "equivocation":
                    node_attacks[node_id] = atk_class(corrupt_probability=0.5)
                elif atk_name == "timing":
                    node_attacks[node_id] = atk_class(delay_probability=0.5)
                else:
                    node_attacks[node_id] = atk_class()

        f_for_simulator = min(actual_byz_count, max_safe_f)
        if n < 3 * f_for_simulator + 1:
            f_for_simulator = max(0, (n - 1) // 3)

        try:
            sim = PBFTSimulator(
                n=n,
                f=f_for_simulator,
                byzantine_indices=byzantine_indices,
                delay=1,
                drop_rate=0.0,
                view_change_timeout=view_change_timeout,
                node_attacks=node_attacks,
                rng_seed=int(self.rng.integers(0, 2**31 - 1)),
            )
        except ValueError:
            sim = None

        if sim is not None:
            stats = sim.run_episode(
                num_requests=self.requests_per_window,
                max_steps_per_request=view_change_timeout * 2,
                max_total_steps=1000,
            )
        else:
            stats = EpisodeStats(
                requests_total=self.requests_per_window,
                requests_committed=0,
                view_changes=0,
                safety_violations=1,
                commit_latencies=[],
                steps_total=0,
                byzantine_fraction=byzantine_fraction,
                committee_size=n,
            )

        # Update history
        self._byz_history.append(byzantine_fraction)
        self._latency_history.append(stats.avg_commit_latency)
        self._last_had_violation = 1 if (is_unsafe or stats.safety_violations > 0) else 0

        # Record violation magnitude in stats
        if is_unsafe:
            # Scale safety violations by the magnitude — this is the key
            # change from the previous version. A small overshoot gets a
            # small penalty; a large overshoot gets a large penalty.
            stats.safety_violations += violation_magnitude

        self._last_stats = stats
        self._window_index += 1
        obs = self._stats_to_obs(stats)
        reward = self._compute_reward(stats, violation_magnitude)
        terminated = self._window_index >= 200
        truncated = False
        info = {
            "window": self._window_index,
            "committee_size": n,
            "view_change_timeout": view_change_timeout,
            "byzantine_fraction": byzantine_fraction,
            "byzantine_count": actual_byz_count,
            "was_unsafe": is_unsafe,
            "violation_magnitude": violation_magnitude,
            "max_safe_f": max_safe_f,
            "attack_types_used": list(node_attacks.values()) if node_attacks else [],
            "stats": stats,
        }
        return obs, float(reward), terminated, truncated, info

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _stats_to_obs(self, stats: EpisodeStats) -> np.ndarray:
        # Base features
        avg_lat = min(stats.avg_commit_latency / 50.0, 1.0)
        vcs = min(stats.view_changes / 10.0, 1.0)
        commit_rate = max(0.0, min(1.0, stats.commit_rate))
        byz = max(0.0, min(1.0, stats.byzantine_fraction))
        violations = min(stats.safety_violations / 5.0, 1.0)
        # Moving averages
        moving_avg_byz = (
            float(np.mean(self._byz_history)) if self._byz_history else 0.0
        )
        moving_avg_lat = (
            min(float(np.mean(self._latency_history)) / 50.0, 1.0)
            if self._latency_history
            else 0.0
        )
        last_violation = float(self._last_had_violation)
        return np.array(
            [
                avg_lat,
                vcs,
                commit_rate,
                byz,
                violations,
                moving_avg_byz,
                moving_avg_lat,
                last_violation,
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self, stats: EpisodeStats, violation_magnitude: int = 0
    ) -> float:
        # Efficiency: penalize larger committees (PBFT message complexity is O(n^2))
        n = stats.committee_size
        n_max = max(self.committee_sizes)
        alpha = 0.30  # overhead weight
        efficiency = 1.0 - alpha * (n / n_max) ** 2
        throughput = stats.commit_rate * max(0.0, efficiency)
        # Variable safety penalty
        safety_penalty = self.safety_penalty_base * violation_magnitude
        # View-change cost
        vc_cost = self.view_change_penalty * stats.view_changes
        return float(throughput - safety_penalty - vc_cost)
