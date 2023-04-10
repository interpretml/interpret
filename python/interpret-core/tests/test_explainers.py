# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from interpret.test.utils import synthetic_classification, get_all_explainers
from interpret.test.utils import assert_valid_explanation, assert_valid_model_explainer

# from interpret.blackbox import PermutationImportance

import pytest


@pytest.mark.slow
def test_spec_synthetic():
    all_explainers = get_all_explainers()
    # use the same dataset for both regression and classification
    data = synthetic_classification()

    binary_model = RandomForestClassifier()
    binary_model.fit(data["train"]["X"], data["train"]["y"])

    regression_model = RandomForestRegressor()
    regression_model.fit(data["train"]["X"], data["train"]["y"])

    for explainer_class, is_classification in all_explainers:
        # if explainer_class == PermutationImportance:  # TODO should true labels be passed in the constructor here?
        #     explainer = explainer_class(binary_model, data["train"]["X"], data["train"]["y"])
        if explainer_class.explainer_type == "blackbox":
            if is_classification:
                explainer = explainer_class(binary_model, data["train"]["X"])
            else:
                explainer = explainer_class(regression_model, data["train"]["X"])
        elif explainer_class.explainer_type == "model":
            explainer = explainer_class()
            explainer.fit(data["train"]["X"], data["train"]["y"])
            assert_valid_model_explainer(explainer, data["test"]["X"].head())
        elif explainer_class.explainer_type == "specific":
            if is_classification:
                explainer = explainer_class(binary_model, data["train"]["X"])
            else:
                explainer = explainer_class(regression_model, data["train"]["X"])
        elif explainer_class.explainer_type == "data":
            explainer = explainer_class()
        elif explainer_class.explainer_type == "perf":
            if is_classification:
                explainer = explainer_class(binary_model)
            else:
                explainer = explainer_class(regression_model)
        else:
            raise Exception("Not supported explainer type.")

        if "local" in explainer.available_explanations:
            # With labels
            explanation = explainer.explain_local(
                data["test"]["X"].head(), data["test"]["y"].head()
            )
            assert_valid_explanation(explanation)

            # Without labels
            explanation = explainer.explain_local(data["test"]["X"])
            assert_valid_explanation(explanation)

        if "global" in explainer.available_explanations:
            explanation = explainer.explain_global()
            assert_valid_explanation(explanation)

        if "data" in explainer.available_explanations:
            explanation = explainer.explain_data(data["train"]["X"], data["train"]["y"])
            assert_valid_explanation(explanation)

        if "perf" in explainer.available_explanations:
            explanation = explainer.explain_perf(data["test"]["X"], data["test"]["y"])
            assert_valid_explanation(explanation)
