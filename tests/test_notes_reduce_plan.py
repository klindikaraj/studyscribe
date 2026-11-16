import itertools

import pytest

from studyscribe.notes.reduce_plan import ReduceTooLarge, fits, plan_groups


def sums(sizes, groups):
    return [sum(sizes[i] for i in g) for g in groups]


def test_everything_that_fits_is_one_group():
    assert plan_groups([3, 4, 5], 12) == [[0, 1, 2]]
    assert fits([3, 4, 5], 12) and not fits([3, 4, 5], 11)


def test_empty_and_invalid():
    assert plan_groups([], 10) == []
    with pytest.raises(ValueError):
        plan_groups([1], 0)
    with pytest.raises(ReduceTooLarge, match="exceeds the budget"):
        plan_groups([5, 17, 3], 16)


def test_three_hour_lecture_like_the_plan():
    # 14 map outputs of about 2.6k tokens each, budget 16k: 3 groups, none above 16k
    sizes = [2600, 2500, 2700, 2400, 2600, 2550, 2650, 2500, 2600, 2700, 2450, 2600, 2550, 2600]
    groups = plan_groups(sizes, 16_000)
    assert len(groups) == 3
    assert [i for g in groups for i in g] == list(range(14))
    assert max(sums(sizes, groups)) <= 16_000


def test_groups_are_as_few_as_possible_and_the_largest_is_minimal():
    sizes = [9, 1, 1, 1, 1, 1, 1, 9]
    budget = 12
    groups = plan_groups(sizes, budget)
    # brute force over every split into the same number of consecutive groups
    best = min(
        max(sums(sizes, [list(range(a, b)) for a, b in zip((0, *cuts), (*cuts, len(sizes)))]))
        for cuts in itertools.combinations(range(1, len(sizes)), len(groups) - 1))
    assert len(groups) == 2
    assert max(sums(sizes, groups)) == best


@pytest.mark.parametrize("sizes, budget", [
    ([5, 5, 5, 5, 5], 11), ([1] * 30, 7), ([8, 1, 8, 1, 8], 9), ([3, 9, 3, 9, 3, 9], 12),
])
def test_groups_cover_all_consecutively_within_budget(sizes, budget):
    groups = plan_groups(sizes, budget)
    assert [i for g in groups for i in g] == list(range(len(sizes)))
    assert all(g for g in groups) and max(sums(sizes, groups)) <= budget



def test_a_weight_limit_makes_more_groups():
    sizes = [100] * 6
    assert plan_groups(sizes, 300) == [[0, 1, 2], [3, 4, 5]]
    # weights: speech; at most 20 per group, so at most two parts of 10
    groups = plan_groups(sizes, 300, weights=[10] * 6, weight_limit=20)
    assert groups == [[0, 1], [2, 3], [4, 5]]


def test_a_part_heavier_than_the_weight_limit_stays_alone():
    groups = plan_groups([100, 100, 100, 100], 250, weights=[5, 50, 5, 5], weight_limit=20)
    assert groups == [[0], [1], [2, 3]]


def test_weights_do_not_matter_when_everything_fits_one_request():
    assert plan_groups([100, 100], 300, weights=[50, 50], weight_limit=20) == [[0, 1]]


def test_weights_must_match_sizes():
    with pytest.raises(ValueError, match="one weight per size"):
        plan_groups([1, 2], 10, weights=[1], weight_limit=5)
