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
    set_option("n_intercept_rounds_initial", 0)
    set_option("n_intercept_rounds_final", 0)

    interactions = []

    fingerprint = 1.0
    model_fingerprint = 1.0
    seed = 10000
    n_rounds = 0
    for n_classes in range(Native.Task_Regression, 4):
        if n_classes < 2 and n_classes != Native.Task_Regression:
            continue



        if n_classes != 2:
            continue



        classes = None if n_classes == Native.Task_Regression else n_classes

        for iteration in range(1):
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
            ebm = ebm_type(names, types, random_state=seed, max_rounds=30001, early_stopping_rounds=0, interactions=0)
            ebm.fit(X, y)

            if isinstance(ebm.intercept_, float):
                if ebm.intercept_ != 0:
                    model_fingerprint *= ebm.intercept_
            else:
                for score in ebm.intercept_:
                    if score != 0:
                        model_fingerprint *= score
            for term_scores in ebm.term_scores_:
                for score in term_scores.ravel():
                    if score != 0:
                        model_fingerprint *= score
                    while 1e100 <= abs(model_fingerprint):
                        model_fingerprint *= 0.5
                    while abs(model_fingerprint) <= 1e-100:
                        model_fingerprint *= 2.0

            n_rounds += sum(ebm.best_iteration_.ravel())

            interactions.append(ebm.term_features_[ebm.n_features_in_ :])

            pred = ebm._predict_score(X)
            fingerprint *= sum(pred.flat)  # do not use numpy since it can use SIMD.

            seed += 1

    #expected = -276897159.85349244
    #expected = 241.3978094140813  # regression
    expected = 1125.5058112124595 # binary
    expected_model = -277224116080027.38

    print(interactions)

    # if model_fingerprint != expected_model:
    #     assert model_fingerprint == expected_model

    if fingerprint != expected:
        assert fingerprint == str(expected) + " " + str(n_rounds)

    set_option("acceleration", original)
