# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import inspect
import warnings

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    merge_ebms,
)
from interpret.utils import make_synthetic
from sklearn.base import clone as sklearn_clone
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from ...tutils import (
    iris_classification,
    smoke_test_explanations,
)


def valid_ebm(ebm):
    assert ebm.term_features_[0] == (0,)

    for term_scores in ebm.term_scores_:
        all_finite = np.isfinite(term_scores).all()
        assert all_finite


def test_merge_ebms():
    # TODO: improve this test by checking the merged ebms for validity.
    #       Right now the merged ebms fail the check for valid_ebm.
    #       The failure might be related to the warning we're getting
    #       about the scalar divide in the merge_ebms line:
    #       "percentage.append((new_high - new_low) / (old_high - old_low))"

    X, y, names, _ = make_synthetic(classes=2, missing=True, output_type="str")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Missing values detected.*")
        warnings.filterwarnings("ignore", "Dropping term.*")
        warnings.filterwarnings(
            "ignore",
            "Interactions with 3 or more terms are not graphed in global explanations.*",
        )

        random_state = 1
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=random_state
        )
        X_train[:, 3] = ""
        X_train[0, 3] = "-1"
        X_train[1, 3] = "-1."
        X_train[2, 3] = "-1.0"
        # make this categorical to merge with the continuous feature in the other ebms
        X_train[3, 3] = "me"
        X_train[4, 3] = "you"

        ebm1 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=10,
            max_interaction_bins=5,
            interactions=[(8, 3, 0)],
        )
        ebm1.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.40, random_state=random_state
        )
        ebm2 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=11,
            max_interaction_bins=4,
            interactions=[(8, 2), (7, 3), (1, 2)],
        )
        ebm2.fit(X_train, y_train)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.60, random_state=random_state
        )
        ebm3 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=12,
            max_interaction_bins=3,
            interactions=[(1, 2), (2, 8)],
        )
        ebm3.fit(X_train, y_train)

        merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
        # valid_ebm(merged_ebm1)
        global_exp = merged_ebm1.explain_global()
        local_exp = merged_ebm1.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.10, random_state=random_state
        )
        ebm4 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=13,
            max_interaction_bins=8,
            interactions=2,
        )
        ebm4.fit(X_train, y_train)

        merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
        # valid_ebm(merged_ebm2)
        global_exp = merged_ebm2.explain_global()
        local_exp = merged_ebm2.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)

        random_state += 10
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.50, random_state=random_state
        )
        ebm5 = ExplainableBoostingClassifier(
            names,
            random_state=random_state,
            max_bins=14,
            max_interaction_bins=8,
            interactions=2,
        )
        ebm5.fit(X_train, y_train)

        merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
        # valid_ebm(merged_ebm3)
        global_exp = merged_ebm3.explain_global()
        local_exp = merged_ebm3.explain_local(X[:5, :], y[:5])
        smoke_test_explanations(global_exp, local_exp, 6000)


def test_merge_ebms_multiclass():
    data = iris_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    random_state = 1
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    ebm1 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=10,
    )
    ebm1.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.40, random_state=random_state
    )
    ebm2 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=11,
    )
    ebm2.fit(X_train, y_train)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.60, random_state=random_state
    )
    ebm3 = ExplainableBoostingClassifier(
        random_state=random_state,
        interactions=0,
        max_bins=12,
    )
    ebm3.fit(X_train, y_train)

    merged_ebm1 = merge_ebms([ebm1, ebm2, ebm3])
    valid_ebm(merged_ebm1)
    global_exp = merged_ebm1.explain_global()
    local_exp = merged_ebm1.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.10, random_state=random_state
    )
    ebm4 = ExplainableBoostingClassifier(
        random_state=random_state, interactions=0, max_bins=13
    )
    ebm4.fit(X_train, y_train)

    merged_ebm2 = merge_ebms([merged_ebm1, ebm4])
    valid_ebm(merged_ebm2)
    global_exp = merged_ebm2.explain_global()
    local_exp = merged_ebm2.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)

    random_state += 10
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.50, random_state=random_state
    )
    ebm5 = ExplainableBoostingClassifier(
        random_state=random_state, interactions=0, max_bins=14
    )
    ebm5.fit(X_train, y_train)

    merged_ebm3 = merge_ebms([ebm5, merged_ebm2])
    valid_ebm(merged_ebm3)
    global_exp = merged_ebm3.explain_global()
    local_exp = merged_ebm3.explain_local(X_te, y_te)
    smoke_test_explanations(global_exp, local_exp, 6000)


# ---------------------------------------------------------------------------
# Tests for Issue #576: merge_ebms produces broken classifiers
#
# The merge_ebms function creates EBM objects using __new__(), which skips
# __init__() and leaves all hyperparameters missing. This causes repr(),
# get_params(), and sklearn.base.clone() to fail with AttributeError.
# The following tests verify that the fix correctly initializes all
# hyperparameters on the merged model.
# ---------------------------------------------------------------------------


def _create_fitted_classifier_pair():
    """Create a pair of fitted EBM classifiers for merge testing.

    Uses a small dataset with minimal training to keep tests fast while
    still producing valid fitted models that can be merged.

    Returns:
        A tuple of (classifier_one, classifier_two, feature_data, target_data)
        where both classifiers are fitted and ready for merging.
    """
    iris = load_iris()
    feature_data = iris.data
    target_data = iris.target

    classifier_one = ExplainableBoostingClassifier(
        interactions=0, outer_bags=2, max_rounds=50, random_state=42
    )
    classifier_one.fit(feature_data, target_data)

    classifier_two = ExplainableBoostingClassifier(
        interactions=0, outer_bags=2, max_rounds=50, random_state=99
    )
    classifier_two.fit(feature_data, target_data)

    return classifier_one, classifier_two, feature_data, target_data


def _create_fitted_regressor_pair():
    """Create a pair of fitted EBM regressors for merge testing.

    Uses the diabetes dataset with minimal training to keep tests fast.

    Returns:
        A tuple of (regressor_one, regressor_two, feature_data, target_data)
        where both regressors are fitted and ready for merging.
    """
    diabetes = load_diabetes()
    feature_data = diabetes.data
    target_data = diabetes.target

    regressor_one = ExplainableBoostingRegressor(
        interactions=0, outer_bags=2, max_rounds=50, random_state=42
    )
    regressor_one.fit(feature_data, target_data)

    regressor_two = ExplainableBoostingRegressor(
        interactions=0, outer_bags=2, max_rounds=50, random_state=99
    )
    regressor_two.fit(feature_data, target_data)

    return regressor_one, regressor_two, feature_data, target_data


def test_merge_ebms_repr():
    """Test that repr() works on merged classifiers (exact bug from Issue #576).

    Before the fix, calling repr() on a merged EBM would raise:
        AttributeError: 'ExplainableBoostingClassifier' object has no
        attribute 'cyclic_progress'

    This test reproduces the exact failure scenario from the issue report.
    """
    classifier_one, classifier_two, _, _ = _create_fitted_classifier_pair()

    merged_classifier = merge_ebms([classifier_one, classifier_two])

    # This is the exact operation that was crashing before the fix.
    # repr() calls get_params() internally, which reads self.<param>
    # for every parameter defined in __init__().
    representation = repr(merged_classifier)

    assert isinstance(representation, str)
    assert "EBMClassifier" in representation
    assert len(representation) > 0


def test_merge_ebms_get_params():
    """Test that get_params() returns a valid parameter dictionary.

    The scikit-learn estimator contract requires that get_params() returns
    a dictionary of all constructor parameters. This fails when __init__()
    is bypassed because the parameter attributes don't exist.
    """
    classifier_one, classifier_two, _, _ = _create_fitted_classifier_pair()

    merged_classifier = merge_ebms([classifier_one, classifier_two])
    merged_parameters = merged_classifier.get_params(deep=False)

    assert isinstance(merged_parameters, dict)
    assert len(merged_parameters) > 0

    # Verify that the parameters match those from the first source model
    source_parameters = classifier_one.get_params(deep=False)
    for parameter_name, source_value in source_parameters.items():
        if parameter_name == "callback":
            # callback should be explicitly set to None on merged models
            assert merged_parameters[parameter_name] is None
        else:
            assert parameter_name in merged_parameters, (
                f"Parameter '{parameter_name}' is missing from merged model"
            )
            assert merged_parameters[parameter_name] == source_value, (
                f"Parameter '{parameter_name}' has value "
                f"'{merged_parameters[parameter_name]}' but expected '{source_value}'"
            )


def test_merge_ebms_sklearn_clone():
    """Test that sklearn.base.clone() works on merged models.

    clone() is a core scikit-learn operation used in cross-validation,
    grid search, and pipelines. It calls get_params() and then creates
    a new instance with those parameters. This test verifies the full
    round-trip works correctly.
    """
    classifier_one, classifier_two, _, _ = _create_fitted_classifier_pair()

    merged_classifier = merge_ebms([classifier_one, classifier_two])

    # clone() should not raise any exceptions
    cloned_classifier = sklearn_clone(merged_classifier)

    assert type(cloned_classifier) is type(merged_classifier)

    # The cloned model should have the same parameters
    merged_parameters = merged_classifier.get_params(deep=False)
    cloned_parameters = cloned_classifier.get_params(deep=False)
    for parameter_name in merged_parameters:
        if parameter_name == "callback":
            continue
        assert parameter_name in cloned_parameters


def test_merge_ebms_has_all_hyperparameters():
    """Test that ALL hyperparameters from __init__ exist on the merged model.

    This test introspects the __init__ signature to get the complete list
    of expected parameters and verifies each one exists as an attribute
    on the merged model. This guards against future parameters being
    added to __init__ without being handled by the merge function.
    """
    classifier_one, classifier_two, _, _ = _create_fitted_classifier_pair()

    merged_classifier = merge_ebms([classifier_one, classifier_two])

    # Get the parameter names from the __init__ signature (excluding 'self')
    init_signature = inspect.signature(ExplainableBoostingClassifier.__init__)
    expected_parameter_names = [
        name for name in init_signature.parameters if name != "self"
    ]

    for parameter_name in expected_parameter_names:
        assert hasattr(merged_classifier, parameter_name), (
            f"Merged classifier is missing hyperparameter '{parameter_name}'. "
            f"This causes repr(), get_params(), and clone() to fail."
        )


def test_merge_ebms_callback_is_none():
    """Test that the callback parameter is explicitly set to None.

    Callbacks are per-training-session callables. A merged model is not
    being trained, so retaining a stale callback reference from one of
    the source models would be misleading and potentially dangerous
    (e.g., the callback might hold references to training state).
    """

    def training_callback(*, bag, step, term, metric):
        return False  # continue training

    classifier_one = ExplainableBoostingClassifier(
        interactions=0,
        outer_bags=2,
        max_rounds=50,
        random_state=42,
        callback=training_callback,
    )
    iris_data = load_iris()
    classifier_one.fit(iris_data.data, iris_data.target)

    classifier_two = ExplainableBoostingClassifier(
        interactions=0, outer_bags=2, max_rounds=50, random_state=99
    )
    classifier_two.fit(iris_data.data, iris_data.target)

    merged_classifier = merge_ebms([classifier_one, classifier_two])

    # Even though classifier_one had a callback, the merged model should not
    assert merged_classifier.callback is None
    assert merged_classifier.get_params()["callback"] is None


def test_merge_ebms_regressor_repr():
    """Test that repr() also works correctly for merged regressors.

    The fix must handle both EBMClassifier and EBMRegressor since they
    have different __init__ signatures (regressor has no 'classes_' etc.).
    """
    regressor_one, regressor_two, feature_data, _ = _create_fitted_regressor_pair()

    merged_regressor = merge_ebms([regressor_one, regressor_two])

    # repr() should work without raising AttributeError
    representation = repr(merged_regressor)
    assert isinstance(representation, str)
    assert "EBMRegressor" in representation

    # get_params() should return valid parameters
    merged_parameters = merged_regressor.get_params(deep=False)
    assert isinstance(merged_parameters, dict)
    assert len(merged_parameters) > 0

    # clone() should also work
    cloned_regressor = sklearn_clone(merged_regressor)
    assert type(cloned_regressor) is type(merged_regressor)

    # Predictions should still work after the fix
    predictions = merged_regressor.predict(feature_data)
    assert len(predictions) == len(feature_data)
    assert np.isfinite(predictions).all()


def test_merge_ebms_predictions_unchanged():
    """Test that the fix does not alter the merged model's predictions.

    Adding hyperparameters to the merged model is purely metadata — it
    should not change any prediction behavior. This test verifies that
    the merged model produces valid, finite predictions.
    """
    classifier_one, classifier_two, feature_data, _ = _create_fitted_classifier_pair()

    merged_classifier = merge_ebms([classifier_one, classifier_two])

    predictions = merged_classifier.predict(feature_data)
    assert len(predictions) == len(feature_data)

    probabilities = merged_classifier.predict_proba(feature_data)
    assert probabilities.shape[0] == len(feature_data)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
