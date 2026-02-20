# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest

from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils._native import Native

# Skip if the native lib is not available (common in source checkouts without build artifacts)
try:
    Native._get_ebm_lib_path()
except Exception:  # pragma: no cover - just gating environments without libebm
    pytest.skip("libebm shared library not available", allow_module_level=True)


def test_callback_iteration_is_monotonic():
    """Ensure callback receives strictly increasing iteration indexes even when no progress is made."""

    X = np.array([[0], [1], [0], [1]], dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    iterations = []

    def cb(bag_idx, iteration_idx, made_progress, metric):
        iterations.append(iteration_idx)
        # stop early to keep test fast; plenty of callback invocations happen before the first boost
        return len(iterations) >= 15

    ebm = ExplainableBoostingClassifier(
        interactions=0,
        max_rounds=1,
        cyclic_progress=0.1,  # forces several no-progress iterations up front
        outer_bags=1,
        max_bins=2,
        max_leaves=2,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=1,
        callback=cb,
    )

    ebm.fit(X, y)

    assert iterations == sorted(set(iterations))
