# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Regression tests for issue #635: callback API uses keyword-only args."""

import numpy as np

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret.utils import make_synthetic


class RecordingCallback:
    """Picklable callback that records all invocations.

    Uses n_jobs=1 in tests so that state is shared in-process.
    """

    def __init__(self):
        self.records = []

    def __call__(self, *, bag, stage, step, term, metric):
        self.records.append((bag, stage, step, term, metric))
        return False


class StopAfterCallback:
    """Picklable callback that stops training after N calls."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.call_count = 0

    def __call__(self, *, bag, stage, step, term, metric):
        self.call_count += 1
        return self.call_count >= self.stop_after


def test_callback_no_repeated_steps_classifier():
    """Verify the callback receives strictly increasing step values.

    Before the fix, the callback was invoked on every internal loop
    iteration — including non-progressing cycles — which caused
    the same step value to be reported multiple times.
    """
    cb = RecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    steps_by_bag = {}
    for bag, stage, step, _, _ in cb.records:
        steps_by_bag.setdefault(bag, []).append((stage, step))

    for bag, steps in steps_by_bag.items():
        for i in range(1, len(steps)):
            assert steps[i] > steps[i - 1], (
                f"Bag {bag}: (stage, step) went from {steps[i - 1]} to "
                f"{steps[i]} (expected strictly increasing)"
            )


def test_callback_no_repeated_steps_regressor():
    """Same test as above but for ExplainableBoostingRegressor."""
    cb = RecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=None, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingRegressor(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    steps_by_bag = {}
    for bag, stage, step, _, _ in cb.records:
        steps_by_bag.setdefault(bag, []).append((stage, step))

    for bag, steps in steps_by_bag.items():
        for i in range(1, len(steps)):
            assert steps[i] > steps[i - 1], (
                f"Bag {bag}: (stage, step) went from {steps[i - 1]} to "
                f"{steps[i]} (expected strictly increasing)"
            )


def test_callback_receives_term_index():
    """Verify the callback receives a valid term index."""
    cb = RecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    for i, (_, _, _, term, _) in enumerate(cb.records):
        assert isinstance(term, (int, np.integer)), (
            f"term at call {i} should be an int, got {type(term)}"
        )
        assert term >= 0, f"term at call {i} should be non-negative, got {term}"


def test_callback_early_termination():
    """Verify the callback can still terminate training early."""
    cb = StopAfterCallback(stop_after=5)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=5000,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert cb.call_count == cb.stop_after, (
        f"Expected callback to be called exactly {cb.stop_after} times "
        f"before stopping, but was called {cb.call_count} times"
    )

    # The model should still be valid after early stopping
    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


def test_callback_receives_valid_metrics():
    """Verify the callback receives valid (finite) metric values."""
    cb = RecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    for i, (_, _, _, _, metric) in enumerate(cb.records):
        assert np.isfinite(metric), f"Metric at step {i} is not finite: {metric}"


def test_callback_keyword_only_signature():
    """Verify the callback is invoked with keyword-only arguments.

    This test ensures that the callback cannot be invoked with positional
    arguments, which is the core API change in this PR.
    """

    class KeywordOnlyCallback:
        def __init__(self):
            self.called = False

        def __call__(self, *, bag, stage, step, term, metric):
            self.called = True
            # Verify all args were passed as keywords by checking they exist
            assert isinstance(bag, int)
            assert isinstance(step, int)
            assert isinstance(term, (int, np.integer))
            assert isinstance(metric, float)
            return True  # stop immediately

    cb = KeywordOnlyCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=cb,
    )
    ebm.fit(X, y)

    assert cb.called, "Keyword-only callback should have been invoked"
