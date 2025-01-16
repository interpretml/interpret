# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret.utils import make_synthetic
from interpret.develop import get_option, set_option
from interpret.utils._native import Native


def test_identical_classification():
    original = get_option("acceleration")
    set_option("acceleration", 0)

    total = 0.0
    seed = 0
    for n_classes in range(Native.Task_Regression, 4):
        if n_classes < 2 and n_classes != Native.Task_Regression:
            continue

        classes = None if n_classes == Native.Task_Regression else n_classes

        for iteration in range(2):
            X, y, names, types = make_synthetic(
                seed=seed, classes=classes, output_type="float", n_samples=257
            )

            ebm_type = (
                ExplainableBoostingClassifier
                if 0 <= n_classes
                else ExplainableBoostingRegressor
            )
            ebm = ebm_type(names, types, random_state=seed, max_rounds=10)
            ebm.fit(X, y)

            pred = ebm._predict_score(X)
            total += sum(pred.flat)

            seed += 1

    expected = 604.4169336846871

    if total != expected:
        assert total == expected

    set_option("acceleration", original)
