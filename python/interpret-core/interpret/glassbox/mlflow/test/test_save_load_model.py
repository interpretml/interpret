# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import json
import os

import pytest

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.linear_model import Lasso as SKLinear

from interpret.glassbox.linear import LogisticRegression, LinearRegression
from interpret.glassbox.mlflow import load_model, log_model


@pytest.fixture()
def glassbox_model():
    boston = load_boston()
    return LinearRegression(feature_names=boston.feature_names, random_state=1)


@pytest.fixture()
def model():
    return SKLinear(random_state=1)


def test_linear_regression_save_load(glassbox_model, model):
    boston = load_boston()
    X, y = boston.data, boston.target

    model.fit(X, y)
    glassbox_model.fit(X, y)

    save_location = "save_location"
    log_model(save_location, glassbox_model)


    import mlflow
    glassbox_model_loaded = load_model("runs:/{}/{}".format(mlflow.active_run().info.run_id, save_location))

    name = "name"
    explanation_glassbox_data = glassbox_model.explain_global(name).data(-1)["mli"]
    explanation_glassbox_data_loaded = glassbox_model_loaded.explain_global(name).data(-1)["mli"]
    assert explanation_glassbox_data == explanation_glassbox_data_loaded
