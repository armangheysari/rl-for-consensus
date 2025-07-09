
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..pbft.simulator import PBFTSimulator, EpisodeStats


# Valid committee sizes — all must satisfy n >= 3f+1 for the supported f range.
DEFAULT_COMMITTEE_SIZES: tuple[int, ...] = (3, 5, 7, 9)


class ConsensusEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        committee_sizes: tuple[int, ...] = DEFAULT_COMMITTEE_SIZES,
        max_byzantine_fraction: float = 0.30,
        requests_per_window: int = 10,
        safety_penalty: float = 10.0,
        rng_seed: int | None = None,
    ):
        super().__init__()
        if max_byzantine_fraction >= 1.0 / 3.0:
            raise ValueError(
                "max_byzantine_fraction must be < 1/3 for PBFT safety; "
                f"got {max_byzantine_fraction}"
            )
        self.committee_sizes = committee_sizes
        self.max_byzantine_fraction = max_byzantine_fraction
        self.requests_per_window = requests_per_window
        self.safety_penalty = safety_penalty
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed

        self.action_space = spaces.Discrete(len(committee_sizes))
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )
        # Internal state
        self._last_stats: EpisodeStats | None = None
        self._window_index: int = 0
        self._current_simulator: PBFTSimulator | None = None

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rng_seed = seed
        self._window_index = 0
        self._last_stats = None
        obs = np.zeros(5, dtype=np.float32)
        info = {"window": 0}
        return obs, info

    def step(self, action: int):
        committee_size = self.committee_sizes[int(action)]
        byzantine_fraction = float(self.rng.uniform(0.0, self.max_byzantine_fraction))
        n = committee_size
        # Stochastic committee selection: each member is Byzantine with
        # probability byz_frac. This is more realistic than floor(byz*n)
        # for small committees.
        actual_byz_count = int(self.rng.binomial(n=n, p=byzantine_fraction))
        # The PBFT protocol is safe only when n >= 3f+1, where f is the
        # actual Byzantine count. If actual_byz_count exceeds (n-1)/3, the
        # committee is unsafe — it cannot guarantee agreement.
        max_safe_f = (n - 1) // 3
        is_unsafe = actual_byz_count > max_safe_f
        # Pick which nodes are Byzantine.
        byzantine_indices: set[int] = set()
        if actual_byz_count > 0:
            byzantine_indices = set(
                self.rng.choice(n, size=actual_byz_count, replace=False).tolist()
            )
        # For the simulator, we set f to min(actual_byz_count, max_safe_f).
        # If actual_byz_count > max_safe_f, the simulator will run with an
        # unsafe configuration (too many Byzantine workers for the committee
        # size), and we expect safety violations.
        f_for_simulator = min(actual_byz_count, max_safe_f)
        # Cap f at 0 if the simulator would otherwise reject it
        if n < 3 * f_for_simulator + 1:
            f_for_simulator = max(0, (n - 1) // 3)
        try:
            sim = PBFTSimulator(
                n=n,
                f=f_for_simulator,
                byzantine_indices=byzantine_indices,
                delay=1,
                drop_rate=0.0,
                view_change_timeout=10,
                rng_seed=int(self.rng.integers(0, 2**31 - 1)),
            )
        except ValueError:
            sim = None
        if sim is not None:
            stats = sim.run_episode(
                num_requests=self.requests_per_window,
                max_steps_per_request=30,
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
        # If the agent's choice was unsafe, record it
        if is_unsafe:
            stats.safety_violations += 1
        self._last_stats = stats
        self._window_index += 1
        obs = self._stats_to_obs(stats)
        reward = self._compute_reward(stats)
        # Episode terminates after a fixed number of windows (200 by default)
        terminated = self._window_index >= 200
        truncated = False
        info = {
            "window": self._window_index,
            "committee_size": n,
            "byzantine_fraction": byzantine_fraction,
            "byzantine_count": actual_byz_count,
            "was_unsafe": is_unsafe,
            "max_safe_f": max_safe_f,
            "stats": stats,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._last_stats is None:
            print("No stats yet.")
            return
        s = self._last_stats
        print(
            f"window={self._window_index} "
            f"n={s.committee_size} "
            f"byz={s.byzantine_fraction:.2f} "
            f"committed={s.requests_committed}/{s.requests_total} "
            f"vcs={s.view_changes} "
            f"avg_lat={s.avg_commit_latency:.2f} "
            f"viol={s.safety_violations}"
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _stats_to_obs(self, stats: EpisodeStats) -> np.ndarray:
        avg_lat = min(stats.avg_commit_latency / 50.0, 1.0)
        vcs = min(stats.view_changes / 10.0, 1.0)
        commit_rate = max(0.0, min(1.0, stats.commit_rate))
        byz = max(0.0, min(1.0, stats.byzantine_fraction))
        violations = min(stats.safety_violations / 5.0, 1.0)
        return np.array(
            [avg_lat, vcs, commit_rate, byz, violations],
            dtype=np.float32,
        )

    def _compute_reward(self, stats: EpisodeStats) -> float:
        # Efficiency: penalize larger committees (PBFT message complexity is O(n^2))
        n = stats.committee_size
        n_max = 9  # max committee size in the default action space
        alpha = 0.30  # weight on the overhead penalty
        efficiency = 1.0 - alpha * (n / n_max) ** 2
        throughput = stats.commit_rate * max(0.0, efficiency)
        return float(throughput - self.safety_penalty * stats.safety_violations)
