from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[List[int]]):
    """Simple P*K batch sampler.

    Each batch consists of P classes, K samples per class (batch_size = P*K).
    This is useful for in-batch class statistics.

    Notes:
    - Samples with replacement within a class if needed.
    - Class selection is uniform over classes present in the dataset labels.
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        k: int,
        length_before_new_iter: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(None)
        if labels.dim() != 1:
            raise ValueError("labels must be a 1D tensor")
        if batch_size % k != 0:
            raise ValueError(f"batch_size must be divisible by k. Got batch_size={batch_size}, k={k}")

        self.labels = labels.to(torch.long).cpu()
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.p = int(batch_size // k)
        self.seed = int(seed)

        # Build index lists per class
        by_class: Dict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.labels.tolist()):
            by_class[int(y)].append(int(idx))
        self.by_class = dict(by_class)
        self.classes: List[int] = sorted(self.by_class.keys())

        if not self.classes:
            raise ValueError("No classes found in labels")

        n = int(labels.numel())
        if length_before_new_iter is None:
            self.length_before_new_iter = (n // self.batch_size) * self.batch_size
        else:
            self.length_before_new_iter = int(length_before_new_iter)

    def __len__(self) -> int:
        return max(1, int(self.length_before_new_iter // self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        num_batches = len(self)

        for _ in range(num_batches):
            chosen_classes = rng.sample(self.classes, k=min(self.p, len(self.classes)))
            # If #classes < P, pad with repeats
            while len(chosen_classes) < self.p:
                chosen_classes.append(rng.choice(self.classes))

            batch: List[int] = []
            for c in chosen_classes:
                pool: Sequence[int] = self.by_class[c]
                if len(pool) >= self.k:
                    batch.extend(rng.sample(pool, k=self.k))
                else:
                    # sample with replacement
                    batch.extend([rng.choice(pool) for _ in range(self.k)])

            rng.shuffle(batch)
            yield batch
