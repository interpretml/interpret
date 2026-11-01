# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
from interpret.utils import SPOT_GreedySubsetSelection


def _reference_greedy(costs, marginal, count):
    """Choose each candidate by directly evaluating the transport objective."""
    marginal = marginal / marginal.sum()
    selected = []
    remaining = list(range(len(costs)))
    for _ in range(count):
        objectives = [
            np.min(costs[[*selected, candidate]], axis=0) @ marginal
            for candidate in remaining
        ]
        selected.append(remaining.pop(int(np.argmin(objectives))))
    assignments = np.argmin(costs[selected], axis=0)
    weights = np.bincount(assignments, weights=marginal, minlength=count)
    return np.asarray(selected), weights


@pytest.mark.parametrize("scale", [1.0, 1_000_000.0, 1_000_000_000.0, float(2**64)])
def test_single_prototype_minimizes_weighted_cost(scale):
    costs = np.array([[3.0, 4.0], [2.0, 3.0]]) * scale
    marginal = np.array([0.5, 0.5])
    indices, weights = SPOT_GreedySubsetSelection(costs, marginal, 1)
    np.testing.assert_array_equal(indices, [np.argmin(costs @ marginal)])
    np.testing.assert_allclose(weights, [1.0])


@pytest.mark.parametrize("scale", [1.0, 1_000_000.0, 1_000_000_000.0, float(2**64)])
def test_weights_follow_nearest_selected_prototypes(scale):
    costs = np.array([[2.0, 5.0], [5.0, 2.0]]) * scale
    indices, weights = SPOT_GreedySubsetSelection(costs, np.array([0.5, 0.5]), 2)
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(weights, [0.5, 0.5])


@pytest.mark.parametrize("scale", [1.0, 1_000_000.0, 1_000_000_000.0, float(2**64)])
@pytest.mark.parametrize("seed", [0, 5])
def test_matches_direct_greedy_objective_at_different_scales(seed, scale):
    costs = np.random.default_rng(seed).integers(1, 20, size=(5, 4)) * scale
    marginal = np.array([1.0, 1.0, 2.0, 4.0])
    original_costs = costs.copy()
    original_marginal = marginal.copy()
    expected_indices, expected_weights = _reference_greedy(costs, marginal, 3)
    indices, weights = SPOT_GreedySubsetSelection(costs, marginal, 3)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(weights, expected_weights)
    np.testing.assert_array_equal(costs, original_costs)
    np.testing.assert_array_equal(marginal, original_marginal)
