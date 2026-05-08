# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Regression tests for issue #635: callback API uses keyword-only args."""

import numpy as np
import pytest

from interpret.glassbox import (
    CallbackAction,
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
        # falling off the end == None == CallbackAction.CONTINUE


class StopAfterCallback:
    """Picklable callback that stops all training after N calls."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.call_count = 0

    def __call__(self, *, bag, stage, step, term, metric):
        self.call_count += 1
        if self.call_count >= self.stop_after:
            return CallbackAction.STOP_ALL
        return CallbackAction.CONTINUE


class ExamRecordingCallback:
    """Picklable callback that records all examined term gains."""

    def __init__(self):
        self.records = []

    def __call__(self, *, bag, stage, step, term, gain):
        self.records.append((bag, stage, step, term, gain))
        # falling off the end == None == CallbackAction.CONTINUE


class InteractionRecordingCallback:
    """Picklable callback that records all interaction-detection invocations."""

    def __init__(self):
        self.records = []

    def __call__(self, *, bag, term, strength):
        self.records.append((bag, term, strength))
        # falling off the end == None == CallbackAction.CONTINUE


class StopAfterExamCallback:
    """Picklable callback that stops all training after N exam calls."""

    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.call_count = 0

    def __call__(self, *, bag, stage, step, term, gain):
        self.call_count += 1
        if self.call_count >= self.stop_after:
            return CallbackAction.STOP_ALL
        return None


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
        callbacks=cb,
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
        callbacks=cb,
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
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    for i, (_, _, _, term, _) in enumerate(cb.records):
        assert isinstance(term, (int, np.integer)), (
            f"term at call {i} should be an int, got {type(term)}"
        )
        assert term >= 0, f"term at call {i} should be non-negative, got {term}"


def test_callback_early_termination():
    """Verify ``CallbackAction.STOP_ALL`` terminates training immediately."""
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
        callbacks=cb,
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
        callbacks=cb,
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
            return CallbackAction.STOP_ALL  # stop immediately

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
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert cb.called, "Keyword-only callback should have been invoked"


@pytest.mark.parametrize("callback", [None, tuple()])
def test_fit_without_callback_still_trains(callback):
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
        callbacks=callback,
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
        callbacks=cb,
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
        callbacks=(exam_cb, progress_cb),
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
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert cb.call_count == cb.stop_after, (
        f"Expected exam callback to be called exactly {cb.stop_after} times "
        f"before stopping, but was called {cb.call_count} times"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


@pytest.mark.parametrize(
    "callbacks, message",
    [
        ((RecordingCallback(), RecordingCallback()), "more than one boost callback"),
        (
            (ExamRecordingCallback(), ExamRecordingCallback()),
            "more than one examine callback",
        ),
        (
            (InteractionRecordingCallback(), InteractionRecordingCallback()),
            "more than one interaction callback",
        ),
    ],
)
def test_callback_tuple_validation_errors(callbacks, message):
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
        callbacks=callbacks,
    )

    with pytest.raises(ValueError, match=message):
        ebm.fit(X, y)


def test_callback_signature_requires_metric_or_gain():
    """Verify callbacks are classified by metric/gain keyword names."""

    class InvalidCallback:
        def __call__(self, *, bag, stage, step, term):
            return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=InvalidCallback(),
    )

    with pytest.raises(ValueError, match="does not match any known"):
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
        callbacks=1,
    )

    with pytest.raises((TypeError, ValueError)):
        ebm.fit(X, y)


def test_callback_allows_trailing_kwargs_progress():
    """A progress callback with trailing ``**kwargs`` is accepted.

    This lets users opt in to forward-compatible signatures: the library
    can add new canonical kwargs in the future without breaking callbacks
    that capture them via ``**kwargs``.
    """
    invocations = []

    def cb(*, bag, stage, step, term, metric, **kwargs):
        invocations.append(metric)
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_allows_trailing_kwargs_exam():
    """An exam callback with trailing ``**kwargs`` is accepted."""
    invocations = []

    def cb(*, bag, stage, step, term, gain, **kwargs):
        invocations.append(gain)
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_allows_defaulted_extra_parameters():
    """Extra parameters are accepted as long as they have default values."""
    invocations = []

    def cb(*, bag, stage, step, term, metric, foo=None, bar=42):
        invocations.append(metric)
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_allows_defaulted_extras_and_trailing_kwargs():
    """Defaulted extras and trailing ``**kwargs`` may be combined."""
    invocations = []

    def cb(*, bag, stage, step, term, gain, foo=None, **kwargs):
        invocations.append(gain)
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_bare_kwargs_shortcut_rejected():
    """A bare ``**kwargs`` callback (no canonical params) is rejected.

    The canonical parameters must be declared by name so the user has an
    explicit, stable contract for what the callback receives.
    """

    def cb(**kwargs):
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )

    with pytest.raises(ValueError, match="does not match any known"):
        ebm.fit(X, y)


def test_callback_extra_param_without_default_rejected():
    """Extra parameters without defaults remain rejected."""

    def cb(*, bag, stage, step, term, metric, extra):
        return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )

    with pytest.raises(ValueError, match="without a default value"):
        ebm.fit(X, y)


def test_callback_signature_must_be_inspectable():
    """Verify callbacks with uninspectable signatures are rejected."""

    class UninspectableCallback:
        @property
        def __signature__(self):
            raise TypeError("uninspectable")

        def __call__(self, *, bag, stage, step, term, metric):
            return None

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=UninspectableCallback(),
    )

    with pytest.raises(ValueError, match="inspectable signature"):
        ebm.fit(X, y)


# ---------------------------------------------------------------------------
# CallbackAction (3-state return contract)
# ---------------------------------------------------------------------------


class StopCurrentOnFirstCallback:
    """Picklable callback that stops the targeted bag's current step on each call."""

    def __init__(self, bag_to_stop):
        self.bag_to_stop = bag_to_stop
        self.records = []

    def __call__(self, *, bag, stage, step, term, metric):
        self.records.append((bag, stage, step))
        if bag == self.bag_to_stop:
            return CallbackAction.STOP_CURRENT
        return None


def test_callback_stop_current_only_ends_current_step():
    """``STOP_CURRENT`` ends the current boosting step but allows later steps.

    EBM training proceeds in big steps (mains, then interactions). When a
    callback returns ``STOP_CURRENT`` during the mains step for a bag, that
    bag advances directly to the interactions step rather than continuing
    to boost mains. Other bags are unaffected.
    """
    cb = StopCurrentOnFirstCallback(bag_to_stop=0)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=2,
        max_rounds=50,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    bag0_calls = [(stage, step) for bag, stage, step in cb.records if bag == 0]
    bag1_calls = [(stage, step) for bag, stage, step in cb.records if bag == 1]

    # Bag 0 stops on the first callback of every stage it enters; the number
    # of calls equals the number of stages. Bag 1 trains every stage fully,
    # so it receives many more calls.
    assert 1 <= len(bag0_calls) <= 4, (
        f"bag 0 should stop on the first call of each stage, got {bag0_calls}"
    )
    assert len(bag1_calls) > len(bag0_calls), (
        f"bag 1 should have continued training, got {len(bag1_calls)} calls"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


def test_callback_string_return_values_accepted():
    """Returning the string equivalents of CallbackAction members works."""
    call_count = {"n": 0}

    def cb(*, bag, stage, step, term, metric):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return "stop_all"
        return "continue"

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=5000,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert call_count["n"] == 3


def test_callback_invalid_return_value_raises_typeerror():
    """Returning a value that is neither None nor a CallbackAction is rejected."""

    def cb(*, bag, stage, step, term, metric):
        return True  # booleans are no longer valid return values

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )

    with pytest.raises(TypeError, match="callback returned"):
        ebm.fit(X, y)


def test_callback_invalid_string_return_value_raises_typeerror():
    """Returning an unrecognized string is rejected with a clear error."""

    def cb(*, bag, stage, step, term, metric):
        return "halt"

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )

    with pytest.raises(TypeError, match="'halt'"):
        ebm.fit(X, y)


# ---------------------------------------------------------------------------
# Interaction-detection callback
# ---------------------------------------------------------------------------


def test_interaction_callback_receives_valid_strengths():
    """Verify the interaction callback receives the expected kwargs for each pair."""
    cb = InteractionRecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "interaction callback should have been invoked"

    for bag, term, strength in cb.records:
        assert bag == 0
        assert isinstance(term, tuple), f"term should be a tuple, got {type(term)}"
        assert len(term) == 2, f"interaction detection examines pairs, got term={term}"
        assert all(isinstance(i, (int, np.integer)) for i in term)
        assert np.isfinite(strength), f"strength {strength} is not finite"


def test_interaction_callback_can_combine_with_others():
    """All three callback types can be supplied together."""
    boost_cb = RecordingCallback()
    exam_cb = ExamRecordingCallback()
    interaction_cb = InteractionRecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks=(boost_cb, exam_cb, interaction_cb),
    )
    ebm.fit(X, y)

    assert len(boost_cb.records) > 0
    assert len(exam_cb.records) > 0
    assert len(interaction_cb.records) > 0


def test_interaction_callback_stop_current_only_ends_current_bag():
    """``STOP_CURRENT`` from interaction callback ends detection for that bag only.

    Other bags continue ranking pairs normally; interaction boosting
    afterwards still runs on whatever each bag accumulated.
    """

    class StopCurrentInteractionCallback:
        def __init__(self, bag_to_stop):
            self.bag_to_stop = bag_to_stop
            self.records = []

        def __call__(self, *, bag, term, strength):
            self.records.append(bag)
            if bag == self.bag_to_stop:
                return CallbackAction.STOP_CURRENT
            return None

    cb = StopCurrentInteractionCallback(bag_to_stop=0)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=2,
        max_rounds=10,
        n_jobs=1,
        callbacks=cb,
    )
    ebm.fit(X, y)

    bag0_calls = [b for b in cb.records if b == 0]
    bag1_calls = [b for b in cb.records if b == 1]

    # Bag 0 stops on the very first pair; bag 1 evaluates all pairs.
    assert len(bag0_calls) == 1, (
        f"bag 0 should stop on the first interaction call, got {len(bag0_calls)}"
    )
    assert len(bag1_calls) > 1, (
        f"bag 1 should evaluate all pairs, got {len(bag1_calls)}"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


def test_interaction_callback_stop_all_skips_interaction_boosting():
    """``STOP_ALL`` from interaction callback aborts all detection and skips boosting.

    With ``STOP_ALL`` from the interaction callback, the second-stage boost
    callback (which would otherwise be invoked during interaction boosting)
    must not be invoked.
    """

    class StopAllOnFirstInteraction:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, bag, term, strength):
            self.calls += 1
            return CallbackAction.STOP_ALL

    interaction_cb = StopAllOnFirstInteraction()
    boost_cb = RecordingCallback()

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=300
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=2,
        max_rounds=10,
        n_jobs=1,
        callbacks=(interaction_cb, boost_cb),
    )
    ebm.fit(X, y)

    # Boost callback should only have been invoked during stage 0 (mains).
    # Interaction stage (stage 1) must be skipped entirely after STOP_ALL.
    stages_seen = {stage for _, stage, _, _, _ in boost_cb.records}
    assert 1 not in stages_seen, (
        f"interaction stage should be skipped after STOP_ALL, got stages {stages_seen}"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)
