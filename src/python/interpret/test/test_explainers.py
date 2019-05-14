# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..data import ClassHistogram
from ..perf import ROC, RegressionPerf

from ..blackbox import LimeTabular
from ..blackbox import ShapKernel
from ..blackbox import MorrisSensitivity
from ..blackbox import PartialDependence

from ..glassbox import LogisticRegression, LinearRegression
from ..glassbox import ClassificationTree, RegressionTree
from ..glassbox import DecisionListClassifier
from ..glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from .utils import synthetic_classification
from .utils import assert_valid_explanation, assert_valid_model_explainer


def test_spec_synthetic():
    data_explainer_classes = [ClassHistogram]
    perf_explainer_classes = [ROC, RegressionPerf]
    model_explainer_classes = [
        ClassificationTree,
        DecisionListClassifier,
        LogisticRegression,
        ExplainableBoostingClassifier,
        RegressionTree,
        LinearRegression,
        ExplainableBoostingRegressor,
    ]
    blackbox_explainer_classes = [
        LimeTabular,
        ShapKernel,
        MorrisSensitivity,
        PartialDependence,
    ]
    all_explainers = []
    all_explainers.extend(model_explainer_classes)
    all_explainers.extend(blackbox_explainer_classes)
    all_explainers.extend(data_explainer_classes)
    all_explainers.extend(perf_explainer_classes)

    data = synthetic_classification()
    blackbox = LogisticRegression()
    blackbox.fit(data["train"]["X"], data["train"]["y"])

    predict_fn = lambda x: blackbox.predict_proba(x)  # noqa: E731

    for explainer_class in all_explainers:
        if explainer_class.explainer_type == "blackbox":
            explainer = explainer_class(predict_fn, data["train"]["X"])
        elif explainer_class.explainer_type == "model":
            explainer = explainer_class()

            explainer.fit(data["train"]["X"], data["train"]["y"])
            assert_valid_model_explainer(explainer, data["test"]["X"].head())
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
