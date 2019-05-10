# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import dash.development.base_component as dash_base
from pandas.core.generic import NDFrame
from plotly import graph_objs as go
from sklearn.base import is_classifier


def synthetic_regression():
    dataset = _synthetic("regression")
    return dataset


def synthetic_classification():
    dataset = _synthetic("classification")
    return dataset


def _synthetic(mode="regression"):
    n_rows = 100
    X_df = pd.DataFrame(np.random.randn(n_rows, 4), columns=list("ABCD"))
    if mode == "classification":
        y_df = pd.DataFrame(np.random.randint(2, size=n_rows), columns=list("Y"))
    else:
        y_df = pd.DataFrame(np.random.randn(n_rows, 1), columns=list("Y"))
    X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(
        X_df, y_df, test_size=0.20, random_state=1
    )

    dataset = {
        "full": {"X": X_df, "y": y_df},
        "train": {"X": X_df_train, "y": y_df_train},
        "test": {"X": X_df_test, "y": y_df_test},
    }

    return dataset


def boston_regression():
    return None


def adult_classification():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
    ).sample(frac=0.01, random_state=42)
    df.columns = [
        "Age",
        "WorkClass",
        "fnlwgt",
        "Education",
        "EducationNum",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "Income",
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols].values
    y_df = df[label].values
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
    if obj is None:
        return True
    elif isinstance(obj, NDFrame):
        return True
    elif isinstance(obj, str):
        return True
    elif isinstance(obj, go.Figure):
        return True
    elif isinstance(obj, dash_base.Component):
        return True
    else:
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
