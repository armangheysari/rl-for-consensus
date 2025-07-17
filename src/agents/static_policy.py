
from __future__ import annotations

from typing import Sequence


class StaticPolicy:

    def __init__(self, committee_sizes: Sequence[int], mode: str = "largest"):
        if mode not in ("smallest", "largest", "oracle"):
            raise ValueError(f"unknown mode: {mode}")
        self.committee_sizes = list(committee_sizes)
        self.mode = mode

    def act(self, observed_byzantine_fraction: float | None = None) -> int:
        if self.mode == "smallest":
            return 0
        if self.mode == "largest":
            return len(self.committee_sizes) - 1
        if self.mode == "oracle":
            if observed_byzantine_fraction is None:
                raise ValueError("oracle mode requires observed_byzantine_fraction")
            # Smallest n such that n >= 3f+1, where f = floor(byz * n)
            # We iterate over available committee sizes and pick the smallest
            # that tolerates the observed fault fraction.
            for idx, n in enumerate(self.committee_sizes):
                f = int(observed_byzantine_fraction * n)
                if n >= 3 * f + 1:
                    return idx
            return len(self.committee_sizes) - 1
        return 0
