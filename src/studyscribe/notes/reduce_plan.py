"""Grouping for the hierarchical reduce.

When the map outputs of a lecture do not fit one reduce request, consecutive outputs are reduced
in groups first (partial reduces), and the final reduce works on the partial notes; more levels
if needed. The groups are as few as possible and balanced: among the groupings with the fewest
groups, the one whose largest group is smallest, so no partial reduce gets much more text than
the others. Sizes are estimated input tokens; the budget is what one request may hold for them.
"""

from __future__ import annotations

from collections.abc import Sequence


class ReduceTooLarge(ValueError):
    """One output alone is larger than the budget: no grouping can help."""


def fits(sizes: Sequence[int], budget: int) -> bool:
    return sum(sizes) <= budget


def plan_groups(sizes: Sequence[int], budget: int) -> list[list[int]]:
    """Consecutive groups of indexes into `sizes`, each summing to at most `budget`. One group
    when everything fits."""
    if budget <= 0:
        raise ValueError("the budget must be positive")
    if not sizes:
        return []
    if max(sizes) > budget:
        raise ReduceTooLarge(f"an input of {max(sizes):,} tokens exceeds the budget of "
                             f"{budget:,}")
    if fits(sizes, budget):
        return [list(range(len(sizes)))]
    # Greedy filling gives the fewest groups; then the smallest limit that still needs no more
    # groups than that makes the largest group as small as it can be.
    count = len(_greedy(sizes, budget))
    low, high = max(sizes), budget
    while low < high:
        middle = (low + high) // 2
        if len(_greedy(sizes, middle)) <= count:
            high = middle
        else:
            low = middle + 1
    return _greedy(sizes, low)


def _greedy(sizes: Sequence[int], limit: int) -> list[list[int]]:
    groups: list[list[int]] = [[]]
    total = 0
    for i, size in enumerate(sizes):
        if groups[-1] and total + size > limit:
            groups.append([])
            total = 0
        groups[-1].append(i)
        total += size
    return groups

