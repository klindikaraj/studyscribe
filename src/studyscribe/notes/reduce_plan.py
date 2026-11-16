"""Grouping for the hierarchical reduce.

When the map outputs of a lecture do not fit one reduce request, consecutive outputs are reduced
in groups first (partial reduces), and the final reduce works on the partial notes; more levels
if needed. The groups are as few as possible and balanced: among the groupings with the fewest
groups, the one whose largest group is smallest, so no partial reduce gets much more text than
the others. Sizes are input tokens, counted on the real outputs (not estimated); the budget is
what one request may hold for them.

A second, optional limit bounds each group's weight: the pipeline passes the parts' speech and
the largest speech whose partial reduce is expected to answer within the output cap, so a group
is never asked for more than one answer can hold. A part heavier than that alone is passed on
as it is (a group of one is not a request).
"""

from __future__ import annotations

from collections.abc import Sequence


class ReduceTooLarge(ValueError):
    """One output alone is larger than the budget: no grouping can help."""


def fits(sizes: Sequence[int], budget: int) -> bool:
    return sum(sizes) <= budget


def plan_groups(sizes: Sequence[int], budget: int, *, weights: Sequence[int] | None = None,
                weight_limit: int | None = None) -> list[list[int]]:
    """Consecutive groups of indexes into `sizes`, each summing to at most `budget` (and, of
    more than one part, to at most `weight_limit` of `weights`). One group when the sizes fit:
    that is the final reduce, whose size the weights do not change."""
    if budget <= 0:
        raise ValueError("the budget must be positive")
    if weights is not None and len(weights) != len(sizes):
        raise ValueError("one weight per size")
    if not sizes:
        return []
    if max(sizes) > budget:
        raise ReduceTooLarge(f"an input of {max(sizes):,} tokens exceeds the budget of "
                             f"{budget:,}")
    if fits(sizes, budget):
        return [list(range(len(sizes)))]
    # Greedy filling gives the fewest groups; then the smallest limit that still needs no more
    # groups than that makes the largest group as small as it can be.
    w = list(weights) if weights is not None else [0] * len(sizes)
    w_limit = weight_limit if weight_limit is not None else sum(w)
    count = len(_greedy(sizes, budget, w, w_limit))
    low, high = max(sizes), budget
    while low < high:
        middle = (low + high) // 2
        if len(_greedy(sizes, middle, w, w_limit)) <= count:
            high = middle
        else:
            low = middle + 1
    return _greedy(sizes, low, w, w_limit)


def _greedy(sizes: Sequence[int], limit: int, weights: Sequence[int],
            weight_limit: int) -> list[list[int]]:
    """Fill each group while both sums stay within their limits (fewest groups: with
    consecutive parts and additive limits, filling greedily is optimal)."""
    groups: list[list[int]] = [[]]
    total = weight = 0
    for i, size in enumerate(sizes):
        if groups[-1] and (total + size > limit or weight + weights[i] > weight_limit):
            groups.append([])
            total = weight = 0
        groups[-1].append(i)
        total += size
        weight += weights[i]
    return groups

