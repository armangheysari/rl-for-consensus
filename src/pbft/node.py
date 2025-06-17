
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(Enum):
    IDLE = 0
    PRE_PREPARE = 1
    PREPARE = 2
    COMMIT = 3
    COMMITTED = 4
    VIEW_CHANGE = 5


@dataclass
class Message:
    msg_type: str
    sender: int
    view: int
    sequence: int
    payload: Any = None
    signature: str = ""

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.msg_type}|{self.sender}|{self.view}|{self.sequence}|{self.payload}".encode())
        return h.hexdigest()


@dataclass
class NodeState:
    phase: Phase = Phase.IDLE
    prepares: set[int] = field(default_factory=set)
    commits: set[int] = field(default_factory=set)
    request_payload: Any = None
    pre_prepare_sender: int = -1
    committed_at_step: int = -1


class PBFTNode:

    def __init__(self, node_id: int, n: int, f: int, is_byzantine: bool = False):
        if n < 3 * f + 1:
            raise ValueError(f"PBFT requires n >= 3f+1; got n={n}, f={f}")
        self.node_id = node_id
        self.n = n
        self.f = f
        self.is_byzantine = is_byzantine
        self.view = 0
        # leader for view v is v % n
        self.state: dict[int, NodeState] = {}
        self.committed_log: list[tuple[int, int, Any]] = []
        # bookkeeping
        self.view_changes_started: int = 0

    @property
    def leader(self) -> int:
        return self.view % self.n

    @property
    def is_leader(self) -> bool:
        return self.node_id == self.leader

    def _quorum(self) -> int:
        return 2 * self.f + 1

    def _state_for(self, seq: int) -> NodeState:
        if seq not in self.state:
            self.state[seq] = NodeState()
        return self.state[seq]

    # ----------------------------------------------------------------
    # Message production
    # ----------------------------------------------------------------

    def start_request(self, seq: int, payload: Any) -> list[Message]:
        if not self.is_leader:
            return []
        # Initialize the leader's own state for this sequence, then emit a
        # PREPARE on its own behalf. This matches the standard PBFT behavior
        # where the leader's PREPARE is implicit and counted in the quorum.
        st = self._state_for(seq)
        if st.phase != Phase.IDLE:
            return []
        if self.is_byzantine:
            pre_prepare = self._byzantine_pre_prepare(seq, payload)[0]
        else:
            pre_prepare = Message(
                msg_type="PRE_PREPARE",
                sender=self.node_id,
                view=self.view,
                sequence=seq,
                payload=payload,
            )
        # Move leader into PREPARE phase, with its own PREPARE already counted
        st.phase = Phase.PREPARE
        st.request_payload = pre_prepare.payload
        st.pre_prepare_sender = self.node_id
        st.prepares.add(self.node_id)
        # Broadcast PRE_PREPARE to other nodes
        outgoing = [pre_prepare]
        # Also broadcast the leader's own PREPARE
        leader_prepare = Message(
            msg_type="PREPARE",
            sender=self.node_id,
            view=self.view,
            sequence=seq,
            payload=pre_prepare.payload,
        )
        outgoing.append(leader_prepare)
        return outgoing

    def _byzantine_pre_prepare(self, seq: int, payload: Any) -> list[Message]:
        corrupted = _corrupt_payload(payload)
        msg = Message(
            msg_type="PRE_PREPARE",
            sender=self.node_id,
            view=self.view,
            sequence=seq,
            payload=corrupted,
        )
        return [msg]

    def handle(self, msg: Message, current_step: int) -> list[Message]:
        if msg.view < self.view:
            # stale message from an old view; drop
            return []
        if msg.view > self.view:
            # Future-view message: ignore unless it is a NEW_VIEW that
            # legitimately advances our view. We handle that in
            # `handle_new_view`.
            return []

        if msg.msg_type == "PRE_PREPARE":
            return self._handle_pre_prepare(msg, current_step)
        if msg.msg_type == "PREPARE":
            return self._handle_prepare(msg, current_step)
        if msg.msg_type == "COMMIT":
            return self._handle_commit(msg, current_step)
        return []

    def _handle_pre_prepare(self, msg: Message, current_step: int) -> list[Message]:
        # Only the leader for the current view should send PRE_PREPARE.
        if msg.sender != self.leader:
            return []
        st = self._state_for(msg.sequence)
        if st.phase != Phase.IDLE:
            # already processing this sequence; ignore duplicate
            return []
        st.phase = Phase.PRE_PREPARE
        st.request_payload = msg.payload
        st.pre_prepare_sender = msg.sender
        # broadcast PREPARE
        prepare = Message(
            msg_type="PREPARE",
            sender=self.node_id,
            view=self.view,
            sequence=msg.sequence,
            payload=msg.payload,
        )
        st.phase = Phase.PREPARE
        return [prepare]

    def _handle_prepare(self, msg: Message, current_step: int) -> list[Message]:
        st = self._state_for(msg.sequence)
        if st.phase not in (Phase.PREPARE, Phase.COMMIT):
            return []
        st.prepares.add(msg.sender)
        # Enter COMMIT once we have 2f+1 PREPAREs (including the leader's
        # implicit one — here we approximate by counting all distinct
        # senders we've seen).
        if st.phase == Phase.PREPARE and len(st.prepares) >= self._quorum():
            st.phase = Phase.COMMIT
            commit = Message(
                msg_type="COMMIT",
                sender=self.node_id,
                view=self.view,
                sequence=msg.sequence,
                payload=st.request_payload,
            )
            return [commit]
        return []

    def _handle_commit(self, msg: Message, current_step: int) -> list[Message]:
        st = self._state_for(msg.sequence)
        if st.phase != Phase.COMMIT:
            return []
        st.commits.add(msg.sender)
        if len(st.commits) >= self._quorum() and st.phase != Phase.COMMITTED:
            st.phase = Phase.COMMITTED
            st.committed_at_step = current_step
            self.committed_log.append(
                (msg.sequence, current_step, st.request_payload)
            )
        return []

    # ----------------------------------------------------------------
    # View change (simplified)
    # ----------------------------------------------------------------

    def start_view_change(self, current_step: int) -> list[Message]:
        self.view += 1
        self.state.clear()
        self.view_changes_started += 1
        vc = Message(
            msg_type="VIEW_CHANGE",
            sender=self.node_id,
            view=self.view,
            sequence=-1,
            payload=None,
        )
        return [vc]

    def handle_new_view(self, msg: Message) -> bool:
        if msg.msg_type != "NEW_VIEW":
            return False
        if msg.view <= self.view:
            return False
        self.view = msg.view
        self.state.clear()
        return True

    def reset(self) -> None:
        self.view = 0
        self.state.clear()
        self.committed_log.clear()
        self.view_changes_started = 0


def _corrupt_payload(payload: Any) -> Any:
    if isinstance(payload, (int, float)):
        return -payload
    if isinstance(payload, str):
        return payload + "::byz"
    if isinstance(payload, dict):
        return {k: _corrupt_payload(v) for k, v in payload.items()}
    return payload
