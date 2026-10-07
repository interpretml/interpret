# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pandas as pd
from interpret.blackbox import PartialDependence
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def test_partial_dependence_with_string_categories():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.choice(["20-29", "30-39", "40-49"], n),
            "size": rng.normal(size=n),
        }
    )
    y = ((X["age"] == "30-39") ^ (X["size"] > 0)).astype(int)
    model = make_pipeline(
        ColumnTransformer(
            [("onehot", OneHotEncoder(), ["age"])], remainder="passthrough"
        ),
        LogisticRegression(),
    ).fit(X, y)

    explanation = PartialDependence(model, X).explain_global()

    age = explanation.data(0)
    assert list(age["names"]) == ["20-29", "30-39", "40-49"]
    expected = [
        model.predict_proba(X.assign(age=value))[:, 1].mean() for value in age["names"]
    ]
    np.testing.assert_allclose(age["scores"], expected)

    size = explanation.data(1)
    assert np.asarray(size["names"]).dtype == np.float64
    expected = [
        model.predict_proba(X.assign(size=value))[:, 1].mean()
        for value in size["names"]
    ]
    np.testing.assert_allclose(size["scores"], expected)
