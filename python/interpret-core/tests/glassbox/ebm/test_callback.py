# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Tests for explicit callback registration and keyword-based invocation."""

from functools import partial, update_wrapper
from types import MappingProxyType
from unittest.mock import Mock

import numpy as np
import pytest

from interpret.glassbox import (
    CallbackAction,
    EBMModel,
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret.glassbox._ebm import _normalize_callbacks
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Callback should have been invoked at least once"

    for i, (_, _, _, _, metric) in enumerate(cb.records):
        assert np.isfinite(metric), f"Metric at step {i} is not finite: {metric}"


def test_callback_keyword_only_signature():
    """Verify callbacks are invoked with keywords rather than positional arguments."""

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
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert cb.called, "Keyword-only callback should have been invoked"


@pytest.mark.parametrize("callbacks", [None, {}])
def test_fit_without_callback_still_trains(callbacks):
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
        callbacks=callbacks,
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
        callbacks={"examine": cb},
    )
    ebm.fit(X, y)

    assert len(cb.records) > 0, "Exam callback should have been invoked at least once"

    for i, (_, _, _, term, gain) in enumerate(cb.records):
        assert isinstance(term, (int, np.integer)), (
            f"term at call {i} should be an int, got {type(term)}"
        )
        assert np.isfinite(gain), f"Gain at step {i} is not finite: {gain}"


def test_callback_mapping_calls_both_callbacks():
    """Verify mapping keys select both boost and examination hooks."""
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
        callbacks={"examine": exam_cb, "boost": progress_cb},
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
        callbacks={"examine": cb},
    )
    ebm.fit(X, y)

    assert cb.call_count == cb.stop_after, (
        f"Expected exam callback to be called exactly {cb.stop_after} times "
        f"before stopping, but was called {cb.call_count} times"
    )

    predictions = ebm.predict(X)
    assert len(predictions) == len(y)


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType])
def test_callback_mapping_preserves_callables(mapping_type):
    exam_cb = ExamRecordingCallback()
    boost_cb = RecordingCallback()
    interaction_cb = InteractionRecordingCallback()
    callbacks = mapping_type(
        {"interaction": interaction_cb, "boost": boost_cb, "examine": exam_cb}
    )

    assert _normalize_callbacks(callbacks) == (exam_cb, boost_cb, interaction_cb)
    assert dict(callbacks) == {
        "interaction": interaction_cb,
        "boost": boost_cb,
        "examine": exam_cb,
    }


@pytest.mark.parametrize(
    "callbacks",
    [
        RecordingCallback(),
        [RecordingCallback()],
        (RecordingCallback(),),
        [],
        (),
        1,
        "boost",
    ],
)
def test_callbacks_must_be_a_mapping(callbacks):
    with pytest.raises(ValueError, match="callbacks must be a mapping"):
        _normalize_callbacks(callbacks)


@pytest.mark.parametrize("name", ["progress", "Boost", "", 1])
def test_callback_names_are_validated(name):
    with pytest.raises(ValueError, match="Unknown callback name"):
        _normalize_callbacks({name: RecordingCallback()})


@pytest.mark.parametrize("name", ["examine", "boost", "interaction"])
@pytest.mark.parametrize("callback", [None, 1, [RecordingCallback()]])
def test_callback_values_must_be_callable(name, callback):
    with pytest.raises(ValueError, match="must be callable"):
        _normalize_callbacks({name: callback})


def test_callback_missing_keyword_argument_raises_typeerror():
    """Callback signatures must still accept the supplied keyword arguments."""

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
        callbacks={"boost": InvalidCallback()},
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'metric'"):
        ebm.fit(X, y)


@pytest.mark.parametrize(
    ("model_type", "classes", "objective"),
    [
        (EBMModel, None, "rmse"),
        (ExplainableBoostingClassifier, 2, "log_loss"),
        (ExplainableBoostingRegressor, None, "rmse"),
    ],
)
def test_callback_allows_selected_kwargs_boost(model_type, classes, objective):
    """A callback can declare only the arguments it needs and capture the rest."""
    invocations = []

    def cb(*, metric, **_kwargs):
        invocations.append(metric)

    X, y, names, types = make_synthetic(
        seed=42, classes=classes, output_type="float", n_samples=100
    )

    ebm = model_type(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
        objective=objective,
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_allows_selected_kwargs_exam():
    """An examine callback can capture unused keyword arguments."""
    invocations = []

    def cb(*, gain, **_kwargs):
        invocations.append(gain)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"examine": cb},
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


def test_callback_registration_ignores_other_event_parameter_names():
    """Parameters from other events do not affect explicit event selection."""
    invocations = []

    def cb(*, metric, gain=None, strength=None, **_kwargs):
        assert gain is None
        assert strength is None
        invocations.append(metric)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
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
        callbacks={"examine": cb},
    )
    ebm.fit(X, y)

    assert len(invocations) > 0


@pytest.mark.parametrize(
    ("name", "expected_keywords"),
    [
        ("examine", {"bag", "stage", "step", "term", "gain"}),
        ("boost", {"bag", "stage", "step", "term", "metric"}),
        ("interaction", {"bag", "term", "strength"}),
    ],
)
def test_callback_accepts_bare_kwargs(name, expected_keywords):
    """The mapping key alone selects the event and its keyword arguments."""
    invocations = []

    def cb(**kwargs):
        invocations.append(kwargs)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={name: cb},
    )
    ebm.fit(X, y)

    assert invocations
    assert all(set(kwargs) == expected_keywords for kwargs in invocations)


def test_callback_missing_extra_argument_raises_typeerror():
    """Unbound required arguments fail when the callback is invoked."""

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
        callbacks={"boost": cb},
    )

    with pytest.raises(TypeError, match="required keyword-only argument: 'extra'"):
        ebm.fit(X, y)


def test_callback_signature_is_not_inspected():
    """Callbacks do not need to expose an inspectable signature."""

    class UninspectableCallback(RecordingCallback):
        @property
        def __signature__(self):
            pytest.fail("Callback signatures must not be inspected.")

    cb = UninspectableCallback()
    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )

    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert cb.records


@pytest.mark.parametrize("preserve_metadata", [False, True])
def test_callback_partial_binds_state(preserve_metadata):
    invocations = []

    def record(records, *, metric, **_kwargs):
        records.append(metric)

    cb = partial(record, invocations)
    if preserve_metadata:
        update_wrapper(cb, record)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )
    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert invocations


def test_callback_accepts_forwarding_wrapper():
    invocations = []

    def record(*, metric, **_kwargs):
        invocations.append(metric)

    def cb(*args, **kwargs):
        return record(*args, **kwargs)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )
    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert invocations


@pytest.mark.parametrize("use_mock", [False, True])
def test_callback_accepts_builtin_and_mock(use_mock):
    last_event = {}
    cb = Mock(wraps=last_event.update) if use_mock else last_event.update

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )
    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"boost": cb},
    )
    ebm.fit(X, y)

    assert set(last_event) == {"bag", "stage", "step", "term", "metric"}


def test_callback_can_handle_multiple_events():
    invocations = []

    def cb(**kwargs):
        invocations.append(kwargs)

    X, y, names, types = make_synthetic(
        seed=42, classes=2, output_type="float", n_samples=100
    )
    ebm = ExplainableBoostingClassifier(
        names,
        types,
        outer_bags=1,
        max_rounds=10,
        n_jobs=1,
        callbacks={"examine": cb, "boost": cb, "interaction": cb},
    )
    ebm.fit(X, y)

    assert any("gain" in kwargs for kwargs in invocations)
    assert any("metric" in kwargs for kwargs in invocations)
    assert any("strength" in kwargs for kwargs in invocations)


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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"boost": cb},
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
        callbacks={"interaction": cb},
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
        callbacks={
            "boost": boost_cb,
            "examine": exam_cb,
            "interaction": interaction_cb,
        },
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
        callbacks={"interaction": cb},
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
        callbacks={"interaction": interaction_cb, "boost": boost_cb},
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
