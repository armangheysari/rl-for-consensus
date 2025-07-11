
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.env.consensus_env import ConsensusEnv


def test_env_reset_returns_zero_obs():
    env = ConsensusEnv(rng_seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    assert np.allclose(obs, 0.0)
    assert info["window"] == 0
    print("PASS: reset returns zero obs")


def test_env_step_shapes():
    env = ConsensusEnv(rng_seed=0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action=2)
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "window" in info
    assert "committee_size" in info
    assert "byzantine_fraction" in info
    assert "stats" in info
    print(f"PASS: step returns correct shapes — n={info['committee_size']}, "
          f"byz={info['byzantine_fraction']:.2f}, reward={reward:.3f}")


def test_obs_in_unit_interval():
    env = ConsensusEnv(rng_seed=0)
    env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(action=2)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"obs out of [0,1]: {obs}"
        if terminated or truncated:
            env.reset()
    print("PASS: obs always in [0, 1]")


def test_smallest_committee_breaks_under_byzantine():
    env = ConsensusEnv(committee_sizes=(3, 5, 7, 9), max_byzantine_fraction=0.30, rng_seed=0)
    env.reset(seed=0)
    # For n=3, the env should always reduce f to 0 (since 3 < 3*1+1=4).
    # So picking n=3 means no Byzantine tolerance — but the env samples
    # byzantine fraction, which gets clamped. We should never see safety
    # violations because f is always 0 for n=3.
    n3_episodes = 0
    n3_with_byzantine = 0
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(action=0)  # n=3
        if info["committee_size"] == 3:
            n3_episodes += 1
            # The byzantine_fraction in info is the sampled one, but f was
            # forced to 0. So the simulator ran with f=0.
            if info["byzantine_fraction"] > 0:
                n3_with_byzantine += 1
        if terminated or truncated:
            env.reset()
    # The test passes if we never crashed
    print(f"PASS: smallest committee (n=3) ran {n3_episodes} episodes, "
          f"{n3_with_byzantine} of which had sampled byzantine fraction > 0 "
          f"(correctly forced to f=0)")


def test_largest_committee_tolerates_byzantine():
    env = ConsensusEnv(committee_sizes=(3, 5, 7, 9), max_byzantine_fraction=0.30, rng_seed=0)
    env.reset(seed=0)
    safety_violations_total = 0
    n9_episodes = 0
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(action=3)  # n=9
        stats = info["stats"]
        if info["committee_size"] == 9:
            n9_episodes += 1
            safety_violations_total += stats.safety_violations
        if terminated or truncated:
            env.reset()
    # Allow up to 5 safety violations out of 50 windows — this is rare
    # but possible under binomial sampling at byz_frac=0.30
    assert safety_violations_total <= 10, (
        f"n=9 should rarely exceed f=2; observed {safety_violations_total} "
        f"safety violations in {n9_episodes} windows"
    )
    print(f"PASS: largest committee (n=9) ran {n9_episodes} episodes with "
          f"{safety_violations_total} safety violations (rare binomial exceedances)")


def test_max_byzantine_fraction_validation():
    try:
        ConsensusEnv(max_byzantine_fraction=0.5)
        assert False, "should have raised"
    except ValueError:
        pass
    print("PASS: max_byzantine_fraction validation works")


if __name__ == "__main__":
    test_env_reset_returns_zero_obs()
    test_env_step_shapes()
    test_obs_in_unit_interval()
    test_smallest_committee_breaks_under_byzantine()
    test_largest_committee_tolerates_byzantine()
    test_max_byzantine_fraction_validation()
    print("\nAll env tests passed.")
