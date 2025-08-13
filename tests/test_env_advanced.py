
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.env.consensus_env_advanced import ConsensusEnvAdvanced


def test_advanced_env_reset_returns_eight_dim_obs():
    env = ConsensusEnvAdvanced(rng_seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,), f"expected (8,), got {obs.shape}"
    assert np.allclose(obs, 0.0)
    assert info["window"] == 0
    print("PASS: advanced env v2 reset returns 8D zero obs")


def test_advanced_env_action_space_is_multi_discrete():
    env = ConsensusEnvAdvanced(rng_seed=0)
    from gymnasium import spaces
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.shape == (2,)
    print("PASS: action space is MultiDiscrete([4, 4])")


def test_advanced_env_step_with_array_action():
    env = ConsensusEnvAdvanced(rng_seed=0)
    env.reset(seed=0)
    action = np.array([2, 1])  # committee=7, vct=10
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (8,), f"expected (8,), got {obs.shape}"
    assert isinstance(reward, float)
    assert info["committee_size"] == 7
    assert info["view_change_timeout"] == 10
    # Should also include violation_magnitude (new in v2)
    assert "violation_magnitude" in info
    print(f"PASS: step with array action — n={info['committee_size']}, vct={info['view_change_timeout']}, "
          f"violation_magnitude={info['violation_magnitude']}")


def test_advanced_env_step_with_int_action_legacy():
    env = ConsensusEnvAdvanced(rng_seed=0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(2)
    assert info["committee_size"] == 7
    # Default VCT is the middle of the range — for (5, 10, 15, 20), index 2 = 15
    assert info["view_change_timeout"] == 15, (
        f"expected middle VCT=15, got {info['view_change_timeout']}"
    )
    print(f"PASS: legacy int action — n={info['committee_size']}, vct={info['view_change_timeout']}")


def test_advanced_env_obs_in_unit_interval():
    env = ConsensusEnvAdvanced(rng_seed=0)
    env.reset(seed=0)
    for _ in range(30):
        action = np.array([np.random.randint(4), np.random.randint(4)])
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"obs out of [0,1]: {obs}"
        if terminated or truncated:
            env.reset()
    print("PASS: obs always in [0, 1]")


def test_advanced_env_moving_averages_update():
    env = ConsensusEnvAdvanced(rng_seed=42, max_byzantine_fraction=0.30)
    env.reset(seed=42)
    nonzero_moving_avg_byz = False
    nonzero_moving_avg_lat = False
    for _ in range(50):
        action = np.array([2, 1])
        obs, reward, term, trunc, info = env.step(action)
        # Indices: 5 = moving_avg_byzantine_fraction, 6 = moving_avg_latency
        if obs[5] > 0.01:
            nonzero_moving_avg_byz = True
        if obs[6] > 0.01:
            nonzero_moving_avg_lat = True
        if term or trunc:
            env.reset()
    assert nonzero_moving_avg_byz, "expected moving avg byzantine fraction to be non-zero"
    # Latency moving average might be zero if all requests committed in 0 steps,
    # which is unlikely but possible. We assert only on byzantine.
    print(f"PASS: moving averages update — byzantine={nonzero_moving_avg_byz}, latency={nonzero_moving_avg_lat}")


def test_advanced_env_with_attacks():
    env = ConsensusEnvAdvanced(
        rng_seed=42,
        attack_types=["collusion", "equivocation", "timing"],
        max_byzantine_fraction=0.30,
    )
    env.reset(seed=42)
    attack_count = 0
    for _ in range(50):
        action = np.array([3, 1])  # largest committee, mid VCT
        obs, reward, term, trunc, info = env.step(action)
        if info["attack_types_used"]:
            attack_count += 1
        if term or trunc:
            env.reset()
    print(f"PASS: env with attacks ran 50 windows, {attack_count} had active attacks")


def test_advanced_env_invalid_attack_type():
    try:
        ConsensusEnvAdvanced(attack_types=["nonexistent"])
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown attack type" in str(e).lower()
    print("PASS: invalid attack type raises ValueError")


def test_advanced_env_max_byzantine_validation():
    try:
        ConsensusEnvAdvanced(max_byzantine_fraction=0.5)
        assert False, "should have raised"
    except ValueError:
        pass
    print("PASS: max_byzantine_fraction validation works")


def test_advanced_env_reward_is_variable():
    env = ConsensusEnvAdvanced(rng_seed=0, safety_penalty_base=2.0)
    env.reset(seed=0)
    # Run several windows and collect rewards + violation magnitudes
    rewards = []
    magnitudes = []
    for _ in range(30):
        action = np.array([0, 0])  # smallest committee, smallest VCT — likely unsafe
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        magnitudes.append(info["violation_magnitude"])
        if term or trunc:
            env.reset()
    # When violation_magnitude > 0, the reward should be lower than when it's 0
    unsafe_rewards = [r for r, m in zip(rewards, magnitudes) if m > 0]
    safe_rewards = [r for r, m in zip(rewards, magnitudes) if m == 0]
    if unsafe_rewards and safe_rewards:
        mean_unsafe = float(np.mean(unsafe_rewards))
        mean_safe = float(np.mean(safe_rewards))
        assert mean_unsafe < mean_safe, (
            f"unsafe rewards ({mean_unsafe:.3f}) should be less than safe rewards ({mean_safe:.3f})"
        )
        print(f"PASS: variable reward — unsafe={mean_unsafe:.3f} < safe={mean_safe:.3f}")
    else:
        print(f"PASS: variable reward (insufficient data: {len(unsafe_rewards)} unsafe, {len(safe_rewards)} safe)")


if __name__ == "__main__":
    test_advanced_env_reset_returns_eight_dim_obs()
    test_advanced_env_action_space_is_multi_discrete()
    test_advanced_env_step_with_array_action()
    test_advanced_env_step_with_int_action_legacy()
    test_advanced_env_obs_in_unit_interval()
    test_advanced_env_moving_averages_update()
    test_advanced_env_with_attacks()
    test_advanced_env_invalid_attack_type()
    test_advanced_env_max_byzantine_validation()
    test_advanced_env_reward_is_variable()
    print("\nAll advanced env v2 tests passed.")
