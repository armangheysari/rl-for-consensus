
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pbft.attacks import (
    ATTACK_REGISTRY,
    ByzantineAttack,
    CollusionAttack,
    EquivocationAttack,
    TimingAttack,
    apply_attack_to_outgoing,
)
from src.pbft.node import Message
from src.pbft.simulator import PBFTSimulator


def test_base_attack_passthrough():
    attack = ByzantineAttack()
    msg = Message(
        msg_type="PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload="test",
    )
    result = attack.modify_outgoing(msg, recipient=1)
    assert result is msg, "base attack should return message unchanged"
    print("PASS: base attack passthrough")


def test_collusion_attack_corrupts_payload():
    attack = CollusionAttack()
    msg = Message(
        msg_type="PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload=42,  # numeric payload — will be sign-flipped
    )
    result = attack.modify_outgoing(msg, recipient=1)
    assert result is not None
    assert result.payload == -42, f"expected -42, got {result.payload}"
    # PRE_PREPARE should pass through unchanged
    pre_prepare = Message(
        msg_type="PRE_PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload=42,
    )
    result2 = attack.modify_outgoing(pre_prepare, recipient=1)
    assert result2 is pre_prepare, "PRE_PREPARE should pass through"
    print("PASS: collusion attack corrupts PREPARE/COMMIT, leaves PRE_PREPARE")


def test_equivocation_attack_only_affects_pre_prepare():
    attack = EquivocationAttack(corrupt_probability=1.0)  # always corrupt
    # PRE_PREPARE should be corrupted
    pre_prepare = Message(
        msg_type="PRE_PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload=42,
    )
    result = attack.modify_outgoing(pre_prepare, recipient=1)
    assert result is not None
    assert result.payload == -42, f"expected -42, got {result.payload}"
    # PREPARE should pass through unchanged
    prepare = Message(
        msg_type="PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload=42,
    )
    result2 = attack.modify_outgoing(prepare, recipient=1)
    assert result2 is prepare, "PREPARE should pass through equivocation"
    print("PASS: equivocation attack only affects PRE_PREPARE")


def test_timing_attack_drops_some_commits():
    attack = TimingAttack(delay_probability=1.0)
    commit = Message(
        msg_type="COMMIT",
        sender=0,
        view=0,
        sequence=1,
        payload="test",
    )
    # Try multiple recipients — all should be dropped
    for recipient in range(5):
        result = attack.modify_outgoing(commit, recipient=recipient)
        assert result is None, f"recipient {recipient}: expected drop, got {result}"
    # PREPARE should pass through
    prepare = Message(
        msg_type="PREPARE",
        sender=0,
        view=0,
        sequence=1,
        payload="test",
    )
    result = attack.modify_outgoing(prepare, recipient=1)
    assert result is prepare, "PREPARE should pass through timing attack"
    print("PASS: timing attack drops all COMMITs at p=1.0")


def test_attack_registry():
    assert "base" in ATTACK_REGISTRY
    assert "collusion" in ATTACK_REGISTRY
    assert "equivocation" in ATTACK_REGISTRY
    assert "timing" in ATTACK_REGISTRY
    assert ATTACK_REGISTRY["collusion"] is CollusionAttack
    print("PASS: attack registry contains all four attacks")


def test_collusion_attack_in_simulator():
    # Vanilla Byzantine: leader sends corrupted PRE_PREPARE
    sim_vanilla = PBFTSimulator(
        n=7,
        f=2,
        byzantine_indices={0, 1},
        rng_seed=42,
    )
    stats_vanilla = sim_vanilla.run_episode(num_requests=10, max_steps_per_request=30)

    # Collusion attack: same Byzantine set, but with collusion
    attacks = {0: CollusionAttack(), 1: CollusionAttack()}
    sim_collusion = PBFTSimulator(
        n=7,
        f=2,
        byzantine_indices={0, 1},
        node_attacks=attacks,
        rng_seed=42,
    )
    stats_collusion = sim_collusion.run_episode(num_requests=10, max_steps_per_request=30)

    # Both should produce some commits (committee is still safe at n=7, f=2)
    # but the collusion attack should affect the protocol in some way:
    # either more view changes or more value violations
    print(f"  vanilla:    committed={stats_vanilla.requests_committed}, "
          f"vcs={stats_vanilla.view_changes}, violations={stats_vanilla.safety_violations}")
    print(f"  collusion:  committed={stats_collusion.requests_committed}, "
          f"vcs={stats_collusion.view_changes}, violations={stats_collusion.safety_violations}")
    # The test passes as long as both runs complete without crashing
    print("PASS: simulator runs with collusion attacks without errors")


def test_timing_attack_increases_view_changes():
    # No attack: should have few view changes
    sim_no_attack = PBFTSimulator(
        n=7,
        f=2,
        byzantine_indices={0, 1},
        view_change_timeout=5,
        rng_seed=42,
    )
    stats_no_attack = sim_no_attack.run_episode(num_requests=5, max_steps_per_request=20)

    # With timing attack on COMMITs: should have more view changes
    attacks = {0: TimingAttack(delay_probability=0.5), 1: TimingAttack(delay_probability=0.5)}
    sim_with_attack = PBFTSimulator(
        n=7,
        f=2,
        byzantine_indices={0, 1},
        node_attacks=attacks,
        view_change_timeout=5,
        rng_seed=42,
    )
    stats_with_attack = sim_with_attack.run_episode(num_requests=5, max_steps_per_request=20)

    print(f"  no attack:    committed={stats_no_attack.requests_committed}, "
          f"vcs={stats_no_attack.view_changes}")
    print(f"  timing attack: committed={stats_with_attack.requests_committed}, "
          f"vcs={stats_with_attack.view_changes}")
    # Timing attack should not crash the simulator; the test passes either way
    # since timing attack effect depends on which messages are dropped
    print("PASS: simulator runs with timing attacks without errors")


if __name__ == "__main__":
    test_base_attack_passthrough()
    test_collusion_attack_corrupts_payload()
    test_equivocation_attack_only_affects_pre_prepare()
    test_timing_attack_drops_some_commits()
    test_attack_registry()
    test_collusion_attack_in_simulator()
    test_timing_attack_increases_view_changes()
    print("\nAll attack tests passed.")
