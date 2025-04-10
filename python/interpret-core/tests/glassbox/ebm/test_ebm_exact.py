# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret.utils import make_synthetic
from interpret.develop import get_option, set_option
from interpret.utils._native import Native


def test_identical_ebm():
    original = get_option("acceleration")
    set_option("acceleration", 0)

    interactions = []

    fingerprint = 1.0
    seed = 0
    n_rounds = 0
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
            X, y, names, types = make_synthetic(
                seed=seed,
                classes=classes,
                output_type="float",
                n_samples=257 + iteration,
            )

            ebm_type = (
                ExplainableBoostingClassifier
                if 0 <= n_classes
                else ExplainableBoostingRegressor
            )
            ebm = ebm_type(names, types, random_state=seed)

            if n_classes < 0:
                # TODO: remove the limitations below force a more complex comparison

                # currently we get floating point last bit differences for regression
                # if we do too many rounds.  This happens even if interactions=0 and
                # if "early_stopping_rounds=0, max_rounds=30001".
                ebm.max_rounds = 10
            else:
                # TODO: remove the limitations below force a more complex comparison

                # leaving interactions enabled leads to fairly large differences in the 2nd most
                # siginficant digit (at least sometimes), so probably different splits are
                # being chosen in rare cases
                ebm.interactions = 0

                # leaving early stopping enabled leads to very small float differences
                # in the least significant bit, so probably it's a floating point
                # handling difference in the assembly code.
                ebm.early_stopping_rounds = 0

            ebm.fit(X, y)

            n_rounds += sum(ebm.best_iteration_.ravel())

            interactions.append(ebm.term_features_[ebm.n_features_in_ :])

            pred = ebm._predict_score(X)
            fingerprint *= sum(pred.flat)  # do not use numpy since it can use SIMD.

            seed += 1

    expected = 8.231780387209846e16

    print(n_rounds)
    print(interactions)
    assert fingerprint == expected

    set_option("acceleration", original)
