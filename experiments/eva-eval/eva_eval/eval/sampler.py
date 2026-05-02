from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable


def stratified_indices(question_types: Iterable[str], total: int, seed: int = 42) -> list[int]:
    """Return up to `total` row indices, stratified by question_type to preserve
    proportions. Deterministic given the same seed."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, qt in enumerate(question_types):
        by_type[qt].append(i)

    types_sorted = sorted(by_type)
    n_types = len(types_sorted)
    if n_types == 0 or total <= 0:
        return []

    base = total // n_types
    leftover = total - base * n_types

    rng = random.Random(seed)
    chosen: list[int] = []
    for k, qt in enumerate(types_sorted):
        idxs = by_type[qt][:]
        rng.shuffle(idxs)
        size = base + (1 if k < leftover else 0)
        size = min(size, len(idxs))
        chosen.extend(idxs[:size])

    chosen.sort()
    return chosen
