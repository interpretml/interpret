from math import ceil, floor
from interpret.glassbox._ebm._utils import (
    make_bag,
    convert_categorical_to_continuous,
    convert_to_cuts,
    convert_to_intervals,
    _create_proportional_tensor,
    deduplicate_bins,
)

import numpy as np
import pytest


def test_deduplicate_bins():
    bins = [
        [{"a": 1, "b": 2}, {"a": 2, "b": 1}, {"b": 2, "a": 1}, {"b": 2, "a": 1}],
        [
            np.array([1, 2, 3], dtype=np.float64),
            np.array([1, 3, 2], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
        ],
    ]

    deduplicate_bins(bins)

    assert len(bins[0]) == 3
    assert id(bins[0][0]) != id(bins[0][1])
    assert id(bins[0][0]) == id(bins[0][2])
    assert id(bins[0][1]) != id(bins[0][2])

    assert len(bins[1]) == 3
    assert id(bins[1][0]) != id(bins[1][1])
    assert id(bins[1][0]) == id(bins[1][2])
    assert id(bins[1][1]) != id(bins[1][2])


def test_conversion_cut_intervals():
    """Minimal test with roundtrip."""
    # cuts -> intervals -> cuts
    for cuts, intervals in [
        ([1, 2], [(float("-inf"), 1.0), (1.0, 2.0), (2.0, float("inf"))]),
        ([], [(float("-inf"), float("inf"))])
    ]:
        to_interval = convert_to_intervals(cuts)
        assert to_interval == intervals
        cut_rountrip = convert_to_cuts(to_interval)
        assert cut_rountrip == cuts


@pytest.mark.skip(reason="make_bag test needs to be updated")
def test_make_bag():
    # TODO: write this test
    bag = make_bag(y, test_size=0.25, rng=1, is_stratified=False)


@pytest.mark.skip(reason="make_bag test needs to be updated")
def test_make_bag_stratified():
    # TODO: write this test
    bag = make_bag(y, test_size=0.25, rng=1, is_stratified=True)


def test_convert_categorical_to_continuous_easy():
    cuts, mapping, old_min, old_max = convert_categorical_to_continuous(
        {"10": 1, "20": 2, "30": 3}
    )
    assert len(cuts) == 2
    assert cuts[0] == 15
    assert cuts[1] == 25
    assert mapping == [[0], [1], [2], [3], [4]]
    assert old_min == 10
    assert old_max == 30


def test_convert_categorical_to_continuous_overlap():
    cuts, mapping, old_min, old_max = convert_categorical_to_continuous(
        {"10": 1, "+5": 1, "40": 4, "abc": 2, "20": 2, "25": 1, "30": 3, "35": 3}
    )
    assert len(cuts) == 2
    assert cuts[0] == 27.5
    assert cuts[1] == 37.5
    assert mapping == [[0], [1, 2], [3], [4], [5]]
    assert old_min == 5
    assert old_max == 40


def test_convert_categorical_to_continuous_identical():
    cuts, mapping, old_min, old_max = convert_categorical_to_continuous(
        {"10": 1, "+20": 2, "  20  ": 3, "30": 4}
    )
    assert len(cuts) == 2
    assert cuts[0] == 15
    assert cuts[1] == 25
    assert mapping == [[0], [1], [2, 3], [4], [5]]
    assert old_min == 10
    assert old_max == 30


def test_create_proportional_tensor():
    axis_weights = [np.array([1, 2], np.float64), np.array([5, 15, 7], np.float64)]
    tensor = _create_proportional_tensor(axis_weights)
    # geometric mean is 9, so each should sum to that
    expected = np.array(
        [[0.55555556, 1.66666667, 0.77777778], [1.11111111, 3.33333333, 1.55555556]]
    )
    assert np.allclose(tensor, expected)
