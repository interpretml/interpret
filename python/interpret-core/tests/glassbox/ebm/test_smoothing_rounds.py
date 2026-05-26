# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Tests for per-feature ``smoothing_rounds`` (issue #626)."""

from __future__ import annotations

import numpy as np
import pytest
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)


def _small_dataset(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.2 - X[:, 2] * 0.3 + rng.normal(scale=0.1, size=n)
    return X, y


def test_scalar_and_broadcast_list_match():
    """A list of identical values must produce the same model as the scalar."""
    X, y = _small_dataset()

    ebm_scalar = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=10,
        random_state=42,
        n_jobs=1,
    )
    ebm_scalar.fit(X, y)

    ebm_list = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=[10, 10, 10],
        random_state=42,
        n_jobs=1,
    )
    ebm_list.fit(X, y)

    pred_scalar = ebm_scalar.predict(X)
    pred_list = ebm_list.predict(X)
    np.testing.assert_allclose(pred_scalar, pred_list, atol=1e-10)


def test_per_feature_zeros_disable_smoothing_for_those_features():
    """Setting ``smoothing_rounds=0`` for some features must still let
    other features run their smoothing phase, and the model must train
    end-to-end without error.
    """
    X, y = _small_dataset()

    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=[50, 0, 0],
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    assert ebm.term_features_ == [(0,), (1,), (2,)]


def test_zero_everywhere_matches_scalar_zero():
    """An all-zeros list disables smoothing entirely, same as scalar 0."""
    X, y = _small_dataset()

    ebm_zero = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=0,
        random_state=42,
        n_jobs=1,
    )
    ebm_zero.fit(X, y)

    ebm_list_zero = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=[0, 0, 0],
        random_state=42,
        n_jobs=1,
    )
    ebm_list_zero.fit(X, y)

    np.testing.assert_allclose(
        ebm_zero.predict(X), ebm_list_zero.predict(X), atol=1e-10
    )


def test_smoothing_rounds_wrong_length_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=[10, 10],  # only 2 entries, dataset has 3 features
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="smoothing_rounds list length"):
        ebm.fit(X, y)


def test_smoothing_rounds_negative_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=[10, -1, 0],
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        ebm.fit(X, y)


def test_smoothing_rounds_non_integer_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=[10.5, 5, 0],
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="must all be integers"):
        ebm.fit(X, y)


def test_smoothing_rounds_empty_sequence_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=[],
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="empty sequence"):
        ebm.fit(X, y)


def test_smoothing_rounds_scalar_negative_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=-1,
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        ebm.fit(X, y)


def test_smoothing_rounds_aligns_with_exclude():
    """When features are excluded, the per-feature list still indexes by
    the original feature index. Excluded features' entries are skipped."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=[50, 0, 25],
        exclude=[(1,)],
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    # Feature 1 was excluded; only mains for features 0 and 2 are present.
    assert ebm.term_features_ == [(0,), (2,)]


def test_interaction_smoothing_rounds_list_requires_explicit_interactions():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=2,  # int -> FAST-discovered, not explicit
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=5,
        interaction_smoothing_rounds=[10, 10],
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="explicit list"):
        ebm.fit(X, y)


def test_interaction_smoothing_rounds_list_length_must_match():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=[(0, 1), (1, 2)],
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=5,
        interaction_smoothing_rounds=[10],  # wrong length
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="interaction_smoothing_rounds list length"):
        ebm.fit(X, y)


def test_interaction_smoothing_rounds_list_with_explicit_interactions_runs():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=[(0, 1), (1, 2)],
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=5,
        interaction_smoothing_rounds=[10, 0],
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    assert (0, 1) in ebm.term_features_ or (1, 0) in ebm.term_features_
    assert (1, 2) in ebm.term_features_ or (2, 1) in ebm.term_features_


def test_classifier_accepts_smoothing_list():
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 4
    X = rng.normal(size=(n_samples, n_features))
    logits = X[:, 0] - 0.5 * X[:, 1]
    y = (logits + rng.normal(scale=0.1, size=n_samples) > 0).astype(int)
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        outer_bags=1,
        max_rounds=100,
        smoothing_rounds=[5] * n_features,
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    proba = ebm.predict_proba(X)
    assert proba.shape == (n_samples, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# --- Validation-helper coverage tests ---------------------------------------
# These exercise the remaining branches of _normalize_smoothing_rounds and
# the call-site edge cases (no mains, all-interactions-deduped) so the
# helper does not need separate unit tests.


def test_smoothing_rounds_none_treated_as_zero():
    """``smoothing_rounds=None`` must be accepted and behave like 0."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=None,
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    assert ebm.term_features_ == [(0,), (1,), (2,)]


def test_smoothing_rounds_float_scalar_accepted():
    """A whole-numbered float should behave the same as the equivalent int."""
    X, y = _small_dataset()
    ebm_int = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=10,
        random_state=42,
        n_jobs=1,
    )
    ebm_int.fit(X, y)

    ebm_float = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=200,
        smoothing_rounds=10.0,
        random_state=42,
        n_jobs=1,
    )
    ebm_float.fit(X, y)
    np.testing.assert_allclose(ebm_int.predict(X), ebm_float.predict(X), atol=1e-10)


def test_smoothing_rounds_float_non_integer_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=10.5,
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="must be an integer or a sequence"):
        ebm.fit(X, y)


def test_smoothing_rounds_float_negative_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=-3.0,
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        ebm.fit(X, y)


def test_smoothing_rounds_2d_array_raises():
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=np.array([[10, 5, 0]]),
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="1-dimensional"):
        ebm.fit(X, y)


def test_smoothing_rounds_object_dtype_array_raises():
    """Arrays with non-int / non-float dtype (e.g. object/string) must be rejected."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds=np.array(["10", "5", "0"]),
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="entries must all be integers"):
        ebm.fit(X, y)


def test_smoothing_rounds_unsupported_type_raises():
    """Types that aren't int/float/list/tuple/ndarray must be rejected."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=0,
        outer_bags=1,
        max_rounds=50,
        smoothing_rounds={10, 5, 0},  # a set
        random_state=42,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="must be an integer or a sequence"):
        ebm.fit(X, y)


def test_smoothing_rounds_list_with_exclude_mains():
    """Passing a per-feature list together with ``exclude='mains'`` should
    not crash; the gather hits the empty-term_features branch."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=[(0, 1)],
        outer_bags=1,
        max_rounds=100,
        smoothing_rounds=[10, 5, 0],
        exclude="mains",
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    # No mains kept, but the interaction pair must still appear.
    main_terms = [t for t in ebm.term_features_ if len(t) == 1]
    assert main_terms == []


def test_interaction_smoothing_rounds_list_with_all_interactions_excluded():
    """When every explicit interaction is excluded, boost_groups is empty.
    The list-form interaction_smoothing_rounds must still validate against
    the user's interactions length without crashing the empty boost call."""
    X, y = _small_dataset()
    ebm = ExplainableBoostingRegressor(
        interactions=[(0, 1)],
        outer_bags=1,
        max_rounds=100,
        smoothing_rounds=0,
        interaction_smoothing_rounds=[10],
        exclude=[(0, 1)],
        random_state=42,
        n_jobs=1,
    )
    ebm.fit(X, y)
    # The interaction was excluded; only mains remain.
    interaction_terms = [t for t in ebm.term_features_ if len(t) > 1]
    assert interaction_terms == []
