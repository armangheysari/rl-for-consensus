
from __future__ import annotations

from typing import Any

from .node import Message, PBFTNode, _corrupt_payload


class ByzantineAttack:

    name: str = "base"

    def modify_outgoing(
        self,
        msg: Message,
        recipient: int,
        rng_state: Any = None,
    ) -> Message | None:
        return msg


class CollusionAttack(ByzantineAttack):

    name = "collusion"

    def modify_outgoing(
        self,
        msg: Message,
        recipient: int,
        rng_state: Any = None,
    ) -> Message | None:
        if msg.msg_type in ("PREPARE", "COMMIT"):
            # Replace the payload with a corrupted version. All colluding
            # workers send the same corrupted value, so the corrupted
            # payload is identical across workers.
            corrupted = _corrupt_payload(msg.payload)
            return Message(
                msg_type=msg.msg_type,
                sender=msg.sender,
                view=msg.view,
                sequence=msg.sequence,
                payload=corrupted,
                signature=msg.signature,
            )
        return msg


class EquivocationAttack(ByzantineAttack):

    name = "equivocation"

    def __init__(self, corrupt_probability: float = 0.5):
        if not 0.0 <= corrupt_probability <= 1.0:
            raise ValueError("corrupt_probability must be in [0, 1]")
        self.corrupt_probability = corrupt_probability

    def modify_outgoing(
        self,
        msg: Message,
        recipient: int,
        rng_state: Any = None,
    ) -> Message | None:
        if msg.msg_type != "PRE_PREPARE":
            return msg
        # Use the RNG state to decide whether to corrupt for this recipient.
        # We use a hash of (msg.sequence, recipient) so the decision is
        # deterministic given the inputs, but appears random across peers.
        import hashlib

        h = hashlib.sha256()
        h.update(f"{msg.sequence}|{recipient}|{id(rng_state)}".encode())
        digest_int = int(h.hexdigest(), 16)
        if (digest_int % 1000) / 1000.0 < self.corrupt_probability:
            corrupted = _corrupt_payload(msg.payload)
            return Message(
                msg_type=msg.msg_type,
                sender=msg.sender,
                view=msg.view,
                sequence=msg.sequence,
                payload=corrupted,
                signature=msg.signature,
            )
        return msg


class TimingAttack(ByzantineAttack):

    name = "timing"

    def __init__(self, delay_probability: float = 0.5):
        if not 0.0 <= delay_probability <= 1.0:
            raise ValueError("delay_probability must be in [0, 1]")
        self.delay_probability = delay_probability

    def modify_outgoing(
        self,
        msg: Message,
        recipient: int,
        rng_state: Any = None,
    ) -> Message | None:
        if msg.msg_type != "COMMIT":
            return msg
        import hashlib

        h = hashlib.sha256()
        h.update(f"{msg.sequence}|{recipient}|{msg.sender}".encode())
        digest_int = int(h.hexdigest(), 16)
        if (digest_int % 1000) / 1000.0 < self.delay_probability:
            # Drop the COMMIT — simulates a delay beyond the view-change
            # timeout.
            return None
        return msg


# A registry for easy lookup by name
ATTACK_REGISTRY: dict[str, type[ByzantineAttack]] = {
    "base": ByzantineAttack,
    "collusion": CollusionAttack,
    "equivocation": EquivocationAttack,
    "timing": TimingAttack,
}


def apply_attack_to_outgoing(
    msg: Message,
    recipient: int,
    attack: ByzantineAttack | None,
    rng_state: Any = None,
) -> Message | None:
    if attack is None:
        return msg
    return attack.modify_outgoing(msg, recipient, rng_state)
