"""Fixed-size buffer of recent self-play samples (see selfplay.py's
`Sample`), sampled uniformly at random for training batches."""

from __future__ import annotations

import random
from collections import deque

from .selfplay import Sample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._samples: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def add_game(self, samples: list[Sample]) -> None:
        self._samples.extend(samples)

    def sample(self, batch_size: int, rng: random.Random) -> list[Sample]:
        n = min(batch_size, len(self._samples))
        return rng.sample(self._samples, n)
