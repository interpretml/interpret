# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret.utils import make_synthetic
from interpret.develop import get_option, set_option
from interpret.utils._native import Native

from ...tutils import toy_regression, toy_binary, toy_multiclass


def test_identical_ebm():
    original = get_option("acceleration")
    set_option("acceleration", 0)

    fingerprint = 1.0
    seed = 0
    for n_classes in range(Native.Task_Regression, 4):
        if n_classes < 2 and n_classes != Native.Task_Regression:
            continue

        classes = None if n_classes == Native.Task_Regression else n_classes

        for iteration in range(2):
            test_type = (
                "regression"
                if n_classes == Native.Task_Regression
                else str(n_classes) + " classes"
            )
            print(f"Exact test for {test_type}, iteration {iteration}.")

            if n_classes == -2:
                X, y, names, types = toy_regression()
            elif n_classes == 2:
                X, y, names, types = toy_binary()
            elif n_classes == 3:
                X, y, names, types = toy_multiclass()
            else:
                raise Exception(f"unsupported number of classes {n_classes}")

            ebm_type = (
                ExplainableBoostingRegressor
                if n_classes == Native.Task_Regression
                else ExplainableBoostingClassifier
            )
            ebm = ebm_type(names, types, random_state=seed)
            ebm.fit(X, y)

            pred = (
                ebm.predict(X)
                if n_classes == Native.Task_Regression
                else ebm.predict_proba(X)
            )
            fingerprint *= sum(pred.flat)  # do not use numpy since it can use SIMD.

            seed += 1

    expected = 3.293830243001898e+19

    assert fingerprint == expected

    set_option("acceleration", original)
