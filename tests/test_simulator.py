
from __future__ import annotations

import os
import sys

# Make `src` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pbft.simulator import PBFTSimulator, EpisodeStats


def test_no_byzantine_commits_everything():
    sim = PBFTSimulator(n=4, f=1, byzantine_indices=set(), rng_seed=42)
    stats = sim.run_episode(num_requests=10, max_steps_per_request=30)
    assert stats.requests_total == 10
    assert stats.requests_committed == 10, f"expected 10 commits, got {stats.requests_committed}"
    assert stats.safety_violations == 0
    assert stats.view_changes == 0
    print(f"PASS: no-byzantine — committed {stats.requests_committed}/{stats.requests_total}, "
          f"avg latency {stats.avg_commit_latency:.2f}")


def test_byzantine_leader_commits_corrupted_value():
    sim = PBFTSimulator(n=4, f=1, byzantine_indices={0}, rng_seed=42)
    stats = sim.run_episode(num_requests=10, max_steps_per_request=30)
    # All 10 requests should commit (PBFT liveness under 1 Byzantine out of 4)
    # but the committed payloads should be corrupted versions of the requests
    assert stats.requests_committed > 0, "expected some commits despite Byzantine leader"
    # Collect what each honest node committed
    honest_payloads_per_seq: dict[int, set[str]] = {}
    for node in sim.nodes:
        if node.is_byzantine:
            continue
        for seq, step, payload in node.committed_log:
            honest_payloads_per_seq.setdefault(seq, set()).add(repr(payload))
    # For each seq, check if the committed payload differs from the original
    # request: the simulator uses payloads like "req-1", "req-2", ...
    value_violations = 0
    for seq, payloads in honest_payloads_per_seq.items():
        original = repr(f"req-{seq}")
        for p in payloads:
            if p != original:
                value_violations += 1
                break
    # We expect at least some value violations (Byzantine leader corrupted
    # the payload before commit)
    assert value_violations > 0, (
        f"expected at least one value violation under Byzantine leader; "
        f"committed payloads: {honest_payloads_per_seq}"
    )
    print(f"PASS: byzantine-leader — committed {stats.requests_committed}/{stats.requests_total}, "
          f"value_violations={value_violations}, "
          f"view_changes={stats.view_changes}, safety_violations={stats.safety_violations}")


def test_view_change_recovery():
    # Use a Byzantine leader that drops all its messages
    sim = PBFTSimulator(
        n=4,
        f=1,
        byzantine_indices={0},
        delay=1,
        drop_rate=1.0,  # Byzantine leader's messages are all dropped
        view_change_timeout=5,
        rng_seed=42,
    )
    stats = sim.run_episode(num_requests=5, max_steps_per_request=20)
    # With the leader's messages all dropped, the committee should not commit
    # until a view change elects node 1 as the new leader
    assert stats.view_changes > 0, (
        "expected view changes when leader is unresponsive"
    )
    print(f"PASS: view-change-recovery — committed {stats.requests_committed}/{stats.requests_total}, "
          f"view_changes={stats.view_changes}")


def test_n_must_satisfy_3f_plus_1():
    try:
        PBFTSimulator(n=2, f=1)
        assert False, "should have raised"
    except ValueError:
        pass
    print("PASS: n >= 3f+1 enforced")


def test_stats_fields_populated():
    sim = PBFTSimulator(n=7, f=2, byzantine_indices={0, 1}, rng_seed=0)
    stats = sim.run_episode(num_requests=5)
    assert stats.committee_size == 7
    assert stats.requests_total == 5
    assert 0.0 <= stats.byzantine_fraction <= 1.0
    assert stats.steps_total > 0
    print(f"PASS: stats populated — n={stats.committee_size}, "
          f"byz_frac={stats.byzantine_fraction:.2f}, "
          f"throughput={stats.throughput:.3f}")


def test_reproducibility():
    sim1 = PBFTSimulator(n=4, f=1, byzantine_indices={0}, rng_seed=123)
    s1 = sim1.run_episode(num_requests=10)
    sim2 = PBFTSimulator(n=4, f=1, byzantine_indices={0}, rng_seed=123)
    s2 = sim2.run_episode(num_requests=10)
    assert s1.requests_committed == s2.requests_committed
    assert s1.view_changes == s2.view_changes
    assert s1.avg_commit_latency == s2.avg_commit_latency
    print(f"PASS: reproducible — committed={s1.requests_committed}, vcs={s1.view_changes}")


if __name__ == "__main__":
    test_n_must_satisfy_3f_plus_1()
    test_no_byzantine_commits_everything()
    test_byzantine_leader_commits_corrupted_value()
    test_view_change_recovery()
    test_stats_fields_populated()
    test_reproducibility()
    print("\nAll tests passed.")
