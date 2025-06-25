
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .attacks import ByzantineAttack
from .network import Network
from .node import PBFTNode, Phase


@dataclass
class EpisodeStats:
    requests_total: int = 0
    requests_committed: int = 0
    view_changes: int = 0
    safety_violations: int = 0
    commit_latencies: list[int] = field(default_factory=list)
    steps_total: int = 0
    byzantine_fraction: float = 0.0
    committee_size: int = 0

    @property
    def avg_commit_latency(self) -> float:
        if not self.commit_latencies:
            return 0.0
        return sum(self.commit_latencies) / len(self.commit_latencies)

    @property
    def throughput(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.requests_committed / self.steps_total

    @property
    def commit_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        if self.safety_violations > 0:
            return 0.0
        return self.requests_committed / self.requests_total


class PBFTSimulator:

    def __init__(
        self,
        n: int,
        f: int,
        byzantine_indices: set[int] | None = None,
        delay: int = 1,
        drop_rate: float = 0.0,
        view_change_timeout: int = 10,
        node_attacks: dict[int, ByzantineAttack] | None = None,
        rng_seed: int | None = None,
    ):
        if n < 3 * f + 1:
            raise ValueError(f"PBFT requires n >= 3f+1; got n={n}, f={f}")
        self.n = n
        self.f = f
        self.byzantine_indices = set(byzantine_indices or set())
        self.delay = delay
        self.drop_rate = drop_rate
        self.view_change_timeout = view_change_timeout
        self.rng = random.Random(rng_seed)
        self.nodes: list[PBFTNode] = [
            PBFTNode(node_id=i, n=n, f=f, is_byzantine=(i in self.byzantine_indices))
            for i in range(n)
        ]
        self.network = Network(
            nodes=self.nodes,
            delay=delay,
            byzantine_workers=self.byzantine_indices,
            drop_rate=drop_rate,
            node_attacks=node_attacks,
            rng=self.rng,
        )

    # ----------------------------------------------------------------
    # Episode runner
    # ----------------------------------------------------------------

    def run_episode(
        self,
        num_requests: int = 20,
        max_steps_per_request: int = 30,
        max_total_steps: int = 1000,
    ) -> EpisodeStats:
        stats = EpisodeStats(
            requests_total=num_requests,
            committee_size=self.n,
            byzantine_fraction=len(self.byzantine_indices) / self.n if self.n else 0.0,
        )
        self.network.reset()
        # Per-request tracking
        request_step_started: dict[int, int] = {}
        # The current leader's view of the request payload
        seq_counter = 0
        for _ in range(num_requests):
            seq_counter += 1
            request_payload = f"req-{seq_counter}"
            start_step = self.network.current_step
            request_step_started[seq_counter] = start_step

            # Leader multicasts PRE_PREPARE
            leader = self.nodes[self.nodes[0].leader]
            pre_prepare_msgs = leader.start_request(seq_counter, request_payload)
            self.network.broadcast(pre_prepare_msgs, sender=leader.node_id)

            # Drive the protocol until commit or timeout
            committed = False
            steps_for_this_request = 0
            while not committed and steps_for_this_request < max_steps_per_request:
                self._drive_one_step(seq_counter, stats)
                steps_for_this_request += 1
                committed = self._any_committed(seq_counter)
                # If not committed and we are near the view-change timeout,
                # trigger a view change
                if not committed and steps_for_this_request >= self.view_change_timeout:
                    self._force_view_change(stats)
                    # Re-issue the request as the new leader
                    new_leader = self.nodes[self.nodes[0].leader]
                    pre_prepare_msgs = new_leader.start_request(seq_counter, request_payload)
                    self.network.broadcast(pre_prepare_msgs, sender=new_leader.node_id)
                    steps_for_this_request = 0
                if self.network.current_step - start_step > max_total_steps:
                    break

            if committed:
                # find the earliest commit
                earliest = self._earliest_commit_step(seq_counter)
                if earliest is not None:
                    stats.commit_latencies.append(earliest - start_step)
                stats.requests_committed += 1
            # Check safety: did honest nodes commit to different payloads?
            if self._safety_violation(seq_counter):
                stats.safety_violations += 1
            stats.steps_total = self.network.current_step
            if self.network.current_step > max_total_steps:
                break
        return stats

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _drive_one_step(self, current_seq: int, stats: EpisodeStats) -> None:
        due = self.network.step()
        for msg, recipient in due:
            node = self.nodes[recipient]
            outgoing = node.handle(msg, current_step=self.network.current_step)
            if outgoing:
                self.network.broadcast(outgoing, sender=node.node_id)

    def _any_committed(self, seq: int) -> bool:
        for node in self.nodes:
            if node.is_byzantine:
                continue
            st = node.state.get(seq)
            if st and st.phase == Phase.COMMITTED:
                return True
        return False

    def _earliest_commit_step(self, seq: int) -> int | None:
        steps = [
            st.committed_at_step
            for node in self.nodes
            if not node.is_byzantine
            for st in [node.state.get(seq)]
            if st and st.phase == Phase.COMMITTED and st.committed_at_step > 0
        ]
        return min(steps) if steps else None

    def _safety_violation(self, seq: int) -> bool:
        committed_payloads: set = set()
        for node in self.nodes:
            if node.is_byzantine:
                continue
            for s, step, payload in node.committed_log:
                if s == seq:
                    committed_payloads.add(repr(payload))
        return len(committed_payloads) > 1

    def _force_view_change(self, stats: EpisodeStats) -> None:
        for node in self.nodes:
            if node.is_byzantine:
                continue
            vc = node.start_view_change(self.network.current_step)
            self.network.broadcast(vc, sender=node.node_id)
        stats.view_changes += 1
