# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..lightgam import LightGAMClassifier, LightGAMRegressor
from ...test.utils import (
    adult_classification,
    boston_regression,
    iris_classification
)


def test_lightgam_boston():
    from sklearn.metrics import mean_squared_error

    data = boston_regression()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

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
    from sklearn.metrics import roc_auc_score

    data = adult_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    learner = LightGAMClassifier(random_state=42)
    learner.fit(X_train, y_train)

    local_exp = learner.explain_local(X_test.iloc[:5])
    local_exp.visualize(0)

    global_exp = learner.explain_global()
    global_exp.visualize()

    global_exp = learner.explain_global()
    global_exp.visualize(0)

    print(roc_auc_score(y_test, learner.predict_proba(X_test)[:, 1]))
    assert roc_auc_score(y_test, learner.predict_proba(X_test)[:, 1]) > 0.9


def test_lightgam_iris():
    from sklearn.metrics import accuracy_score

    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    learner = LightGAMClassifier(random_state=42)
    learner.fit(X_train, y_train)

    local_exp = learner.explain_local(X_test.iloc[:5])
    local_exp.visualize(0)

    global_exp = learner.explain_global()
    global_exp.visualize()

    global_exp = learner.explain_global()
    global_exp.visualize(0)

    assert accuracy_score(y_train, learner.predict(X_train)) > 0.9
    assert accuracy_score(y_test, learner.predict(X_test)) > 0.9
