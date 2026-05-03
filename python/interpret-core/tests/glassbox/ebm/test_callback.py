# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Regression tests for issue #635: callback API uses keyword-only args."""

import numpy as np
import pytest

from interpret.glassbox import _ebm
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


class ExamRecordingCallback:
    """Picklable callback that records all examined term gains."""

    def __init__(self):
        self.records = []

    def __call__(self, *, bag, stage, step, term, gain):
        self.records.append((bag, stage, step, term, gain))
        return False


class StopAfterExamCallback:
    """Picklable callback that stops training after N examination calls."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.call_count = 0

    def __call__(self, *, bag, stage, step, term, gain):
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


def test_fit_without_callback_still_trains():
    """Verify the no-callback training path still works."""
    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=200
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=None,
    )
    ebm.fit(X, y)

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


def test_exam_callback_receives_valid_gains():
    """Verify the examination callback receives finite gain values."""
    cb = ExamRecordingCallback()

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

    assert len(cb.records) > 0, "Exam callback should have been invoked at least once"

    for i, (_, _, _, term, gain) in enumerate(cb.records):
        assert isinstance(term, (int, np.integer)), (
            f"term at call {i} should be an int, got {type(term)}"
        )
        assert np.isfinite(gain), f"Gain at step {i} is not finite: {gain}"


def test_callback_tuple_support_calls_both_callbacks():
    """Verify tuple callbacks dispatch both progress and examination hooks."""
    progress_cb = RecordingCallback()
    exam_cb = ExamRecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=500
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=50,
        n_jobs=1,
        callback=(exam_cb, progress_cb),
    )
    ebm.fit(X, y)

    assert len(progress_cb.records) > 0, "Progress callback should have been invoked"
    assert len(exam_cb.records) > 0, "Exam callback should have been invoked"


def test_exam_callback_early_termination():
    """Verify the examination callback can terminate training early."""
    cb = StopAfterExamCallback(stop_after=5)

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
        f"Expected exam callback to be called exactly {cb.stop_after} times "
        f"before stopping, but was called {cb.call_count} times"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


@pytest.mark.parametrize(
    "callback, message",
    [
        ((RecordingCallback(), RecordingCallback()), "more than one progress callback"),
        (
            (ExamRecordingCallback(), ExamRecordingCallback()),
            "more than one examination callback",
        ),
        (tuple(), "cannot be empty"),
    ],
)
def test_callback_tuple_validation_errors(callback, message):
    """Verify tuple callback validation errors are raised clearly."""
    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=callback,
    )

    with pytest.raises(ValueError, match=message):
        ebm.fit(X, y)


def test_callback_signature_requires_metric_or_gain():
    """Verify callbacks are classified by metric/gain keyword names."""

    class InvalidCallback:
        def __call__(self, *, bag, stage, step, term):
            return False

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=InvalidCallback(),
    )

    with pytest.raises(ValueError, match="either the progress signature"):
        ebm.fit(X, y)


def test_callback_must_be_callable():
    """Verify non-callable callback values are rejected."""
    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=1,
    )

    with pytest.raises(ValueError, match="callable or a tuple of callables"):
        ebm.fit(X, y)


def test_callback_signature_must_be_inspectable(monkeypatch):
    """Verify callbacks with uninspectable signatures are rejected."""

    class ValidProgressCallback:
        def __call__(self, *, bag, stage, step, term, metric):
            return False

    def raise_type_error(_):
        raise TypeError("boom")

    monkeypatch.setattr(_ebm.inspect, "signature", raise_type_error)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=ValidProgressCallback(),
    )

    with pytest.raises(ValueError, match="inspectable signature"):
        ebm.fit(X, y)


def test_callback_missing_required_parameters():
    """Verify callbacks missing required keyword names are rejected."""

    class MissingTermCallback:
        def __call__(self, *, bag, stage, step, metric):
            return False

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=MissingTermCallback(),
    )

    with pytest.raises(ValueError, match="missing required parameters"):
        ebm.fit(X, y)


def test_callback_must_accept_keyword_arguments():
    """Verify positional-only callback signatures are rejected."""

    class PositionalOnlyCallback:
        def __call__(self, bag, stage, step, term, metric, /):
            return False

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=PositionalOnlyCallback(),
    )

    with pytest.raises(ValueError, match="callable with keyword arguments"):
        ebm.fit(X, y)


def test_callback_tuple_rejects_more_than_two_callbacks():
    """Verify callback tuples longer than two items are rejected."""
    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callback=(
            RecordingCallback(),
            ExamRecordingCallback(),
            RecordingCallback(),
        ),
    )

    with pytest.raises(ValueError, match="at most one progress callback"):
        ebm.fit(X, y)
