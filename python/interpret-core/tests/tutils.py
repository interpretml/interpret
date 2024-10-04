# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


import dash.development.base_component as dash_base
import numpy as np
import pandas as pd
from interpret.blackbox import (
    PartialDependence,
)
from interpret.data import ClassHistogram, Marginal
from interpret.glassbox import (
    ClassificationTree,
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    LinearRegression,
    LogisticRegression,
    RegressionTree,
)

# from ..blackbox import PermutationImportance
from interpret.perf import PR, ROC, RegressionPerf
from pandas.core.generic import NDFrame
from plotly import graph_objs as go
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split


def get_all_explainers():
    # True means run on classification.  False means run on regression
    data_explainer_classes = [(ClassHistogram, None), (Marginal, None)]
    perf_explainer_classes = [(ROC, True), (PR, True), (RegressionPerf, False)]
    model_explainer_classes = [
        (ClassificationTree, True),
        (RegressionTree, False),
        (LogisticRegression, True),
        (LinearRegression, False),
        (ExplainableBoostingClassifier, True),
        (ExplainableBoostingRegressor, False),
    ]
    specific_explainer_classes = [
        # (TreeInterpreter, True),  # andosa/treeinterpreter no longer maintained
        # (TreeInterpreter, False),  # andosa/treeinterpreter no longer maintained
        # TODO: Turn this back on after TreeSHAP works on numpy 2.0
        # https://github.com/shap/shap/pull/3704
        # (ShapTree, True),
        # (ShapTree, False),
    ]
    blackbox_explainer_classes = [
        # (LimeTabular, True),  # lime no longer maintained
        # (LimeTabular, False),  # lime no longer maintained
        # TODO: Turn this back on after SHAP works on numpy 2.0
        # https://github.com/shap/shap/pull/3704
        # (ShapKernel, True),
        # (ShapKernel, False),
        # (MorrisSensitivity, True),
        # (MorrisSensitivity, False),
        (PartialDependence, True),
        (PartialDependence, False),
        # PermutationImportance
    ]
    all_explainers = []
    all_explainers.extend(model_explainer_classes)
    all_explainers.extend(specific_explainer_classes)
    all_explainers.extend(blackbox_explainer_classes)
    all_explainers.extend(data_explainer_classes)
    all_explainers.extend(perf_explainer_classes)

    # if sys.version_info[0] <= 3 and sys.version_info[1] < 10:
    #     # skope-rules doesn't work in python 3.10 as of March 2023,
    #     # although there was a PR accepted to change this a few weeks ago
    #     # so the version after 1.0.1 should work
    #     all_explainers.append((DecisionListClassifier, True))

    return all_explainers


def synthetic_regression():
    dataset = _synthetic("regression")
    return dataset


def synthetic_classification():
    dataset = _synthetic("classification")
    return dataset


def synthetic_multiclass():
    dataset = _synthetic("multiclass")
    return dataset


def _synthetic(mode="regression"):
    n_rows = 400
    X_df = pd.DataFrame(np.random.randn(n_rows, 4), columns=list("ABCD"))
    if mode == "classification":
        y_df = pd.DataFrame(
            np.random.randint(2, size=n_rows), columns=list("Y")
        ).squeeze()
    elif mode == "multiclass":
        y_df = pd.DataFrame(
            np.random.randint(3, size=n_rows), columns=list("Y")
        ).squeeze()
    else:
        y_df = pd.DataFrame(np.random.randn(n_rows, 1), columns=list("Y")).squeeze()
    X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(
        X_df, y_df, test_size=0.20, random_state=1
    )

    dataset = {
        "full": {"X": X_df, "y": y_df},
        "train": {"X": X_df_train, "y": y_df_train},
        "test": {"X": X_df_test, "y": y_df_test},
    }

    return dataset


def iris_classification():
    from sklearn.datasets import load_iris

    iris = load_iris()

    X_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    y_df = pd.DataFrame(data=iris["target"], columns=["target"])

    X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(
        X_df, y_df, test_size=0.20, random_state=1
    )

    dataset = {
        "full": {"X": X_df, "y": y_df},
        "train": {"X": X_df_train, "y": y_df_train},
        "test": {"X": X_df_test, "y": y_df_test},
    }

    return dataset


def valid_predict(explainer, X):
    all_valid = True

    predictions = explainer.predict(X)
    all_finite = np.isfinite(predictions).all()

    all_valid &= all_finite
    all_valid &= isinstance(predictions, np.ndarray)
    all_valid &= predictions.ndim == 1

    return all_valid


def valid_predict_proba(explainer, X):
    all_valid = True

    predictions = explainer.predict_proba(X)
    all_finite = np.isfinite(predictions).all()
    within_bounds = (predictions >= 0.0).all() and (predictions <= 1.0).all()

    all_valid &= all_finite
    all_valid &= within_bounds
    all_valid &= isinstance(predictions, np.ndarray)
    all_valid &= predictions.ndim == 2

    return all_valid


def assert_valid_model_explainer(explainer, X):
    assert valid_predict(explainer, X)

    if is_classifier(explainer):
        assert valid_predict_proba(explainer, X)


def valid_visualization(obj):
    if obj is None or isinstance(obj, NDFrame) or isinstance(obj, str) or isinstance(obj, go.Figure) or isinstance(obj, dash_base.Component):
        return True
    return False


def valid_data_dict(data_dict):
    if data_dict is None:
        return True

    if not isinstance(data_dict, dict):
        return False

    return True


def valid_internal_obj(obj):
    if obj is None:
        return True

    overall = obj.get("overall", False)
    specific = obj.get("specific", False)
    if overall is False:
        return False
    if specific is False:
        return False

    if overall is not None and not isinstance(overall, dict):
        return False
    if specific is not None and not isinstance(specific, list):
        return False

    if isinstance(specific, list):
        for item in specific:
            if not isinstance(item, dict):
                return False

    return True


def assert_valid_explanation(explanation):
    assert valid_internal_obj(explanation._internal_obj)
    assert valid_data_dict(explanation.data())
    assert valid_visualization(explanation.visualize())

    try:
        _ = explanation.data(0)
        has_specific = True
    except Exception:
        has_specific = False

    if has_specific:
        assert valid_data_dict(explanation.data(0))
        assert valid_visualization(explanation.visualize(0))


def smoke_test_explanations(global_exp, local_exp, port):
    from interpret import preserve, set_show_addr, show, shutdown_show_server

    set_show_addr(("127.0.0.1", port))

    # Smoke test: should run without crashing.
    preserve(global_exp)
    preserve(local_exp)
    show(global_exp)
    show(local_exp)

    # Check all features for global (including interactions).
    for selector_key in global_exp.selector[global_exp.selector.columns[0]]:
        preserve(global_exp, selector_key)

    shutdown_show_server()
