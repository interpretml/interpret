# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from sklearn.ensemble import RandomForestClassifier

from .utils import synthetic_classification, get_all_explainers
from .utils import assert_valid_explanation, assert_valid_model_explainer

# from ..blackbox import PermutationImportance

from ..glassbox import LogisticRegression

import pytest


# TODO: Generalize specific models (currently only testing trees)
@pytest.mark.slow
def test_spec_synthetic():
    all_explainers = get_all_explainers()
    data = synthetic_classification()

    blackbox = LogisticRegression()
    blackbox.fit(data["train"]["X"], data["train"]["y"])
    tree = RandomForestClassifier()
    tree.fit(data["train"]["X"], data["train"]["y"])

    predict_fn = lambda x: blackbox.predict_proba(x)  # noqa: E731

    for explainer_class in all_explainers:
        # if explainer_class == PermutationImportance:  # TODO should true labels be passed in the constructor here?
        #     explainer = explainer_class(predict_fn, data["train"]["X"], data["train"]["y"])
        if explainer_class.explainer_type == "blackbox":
            explainer = explainer_class(predict_fn, data["train"]["X"])
        elif explainer_class.explainer_type == "model":
            explainer = explainer_class()
            explainer.fit(data["train"]["X"], data["train"]["y"])
            assert_valid_model_explainer(explainer, data["test"]["X"].head())
        elif explainer_class.explainer_type == "specific":
            explainer = explainer_class(tree, data["train"]["X"])
        elif explainer_class.explainer_type == "data":
            explainer = explainer_class()
        elif explainer_class.explainer_type == "perf":
            explainer = explainer_class(predict_fn)
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
