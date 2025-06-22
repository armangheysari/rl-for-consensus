
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .attacks import ByzantineAttack, apply_attack_to_outgoing
from .node import Message, PBFTNode


class Network:

    def __init__(
        self,
        nodes: list[PBFTNode],
        delay: int = 1,
        byzantine_workers: set[int] | None = None,
        drop_rate: float = 0.0,
        node_attacks: dict[int, ByzantineAttack] | None = None,
        rng: random.Random | None = None,
    ):
        self.nodes = nodes
        self.delay = max(1, delay)
        self.byzantine_workers = byzantine_workers or set()
        self.drop_rate = max(0.0, min(1.0, drop_rate))
        self.node_attacks = node_attacks or {}
        self.rng = rng or random.Random()
        # in-flight messages keyed by delivery step
        self._in_flight: dict[int, list[tuple[Message, int]]] = defaultdict(list)
        self.current_step = 0

    def reset(self) -> None:
        self._in_flight.clear()
        self.current_step = 0
        for node in self.nodes:
            node.reset()

    def broadcast(self, msgs: Iterable[Message], sender: int) -> None:
        for msg in msgs:
            for recipient in range(len(self.nodes)):
                if recipient == sender:
                    continue
                # Apply attack if one is attached to this sender
                if sender in self.node_attacks:
                    attack = self.node_attacks[sender]
                    modified = apply_attack_to_outgoing(
                        msg, recipient, attack, rng_state=self.rng
                    )
                    if modified is None:
                        # Attack dropped the message
                        continue
                    msg_to_send = modified
                elif sender in self.byzantine_workers:
                    # Default Byzantine behavior: silent drop with drop_rate
                    if self._should_drop():
                        continue
                    msg_to_send = msg
                else:
                    msg_to_send = msg
                delivery_step = self.current_step + self.delay
                self._in_flight[delivery_step].append((msg_to_send, recipient))

    def step(self) -> list[tuple[Message, int]]:
        self.current_step += 1
        due = self._in_flight.pop(self.current_step, [])
        return due

    def _should_drop(self) -> bool:
        return self.rng.random() < self.drop_rate

    def inject_byzantine_workers(self, indices: set[int]) -> None:
        self.byzantine_workers = set(indices)

    def attach_attack(self, node_id: int, attack: ByzantineAttack) -> None:
        self.node_attacks[node_id] = attack

    def detach_attack(self, node_id: int) -> None:
        self.node_attacks.pop(node_id, None)
