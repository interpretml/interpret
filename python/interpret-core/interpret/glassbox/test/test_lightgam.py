# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..lightgam import LightGAMClassifier, LightGAMRegressor
from ...test.utils import (
    adult_classification,
    boston_regression
)


def test_lightgam_boston():
    data = boston_regression()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    from sklearn.metrics import mean_squared_error

    learner = LightGAMRegressor(random_state=42)
    learner.fit(X_train, y_train)

    local_exp = learner.explain_local(X_test.iloc[:5])
    local_exp.visualize(0)

    global_exp = learner.explain_global()
    global_exp.visualize()

    global_exp = learner.explain_global()
    global_exp.visualize(0)

    print(mean_squared_error(y_test, learner.predict(X_test)))
    assert mean_squared_error(y_test, learner.predict(X_test)) > 0


def test_lightgam_adult():
    data = adult_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    from lightgbm import LGBMClassifier
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    # learner = LightGAMClassifier()
    # is_cat = np.array([dt.kind == 'O' for dt in X_train.dtypes])
    # cat_cols = X_train.columns.values[is_cat]
    # if cat_cols is not None:
    #     for col in cat_cols:
    #         X_train[col] = pd.Categorical(X_train[col])
    #         X_test[col] = pd.Categorical(X_test[col])

    learner = LightGAMClassifier(random_state=42)
    learner.fit(X_train, y_train)

    local_exp = learner.explain_local(X_test.iloc[:5])
    local_exp.visualize(0)

    global_exp = learner.explain_global()
    global_exp.visualize()

    global_exp = learner.explain_global()
    global_exp.visualize(0)

    assert roc_auc_score(y_test, learner.predict_proba(X_test)[:, 1]) > 0.9

