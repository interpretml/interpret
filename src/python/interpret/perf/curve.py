# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data, gen_name_from_class, unify_predict_fn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from ..visual.plot import plot_performance_curve

import numpy as np


class PR(ExplainerMixin):
    available_explanations = ["perf"]
    explainer_type = "perf"

    def __init__(self, predict_fn, feature_names=None, feature_types=None, **kwargs):
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.feature_names = feature_names
        self.feature_types = feature_types

    def explain_perf(self, X, y, name=None):
        if name is None:
            name = gen_name_from_class(self)

        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        predict_fn = unify_predict_fn(self.predict_fn, X)
        scores = predict_fn(X)

        precision, recall, thresh = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)

        abs_residuals = np.abs(y - scores)
        counts, values = np.histogram(abs_residuals, bins="doane")

        overall_dict = {
            "type": "perf_curve",
            "density": {"names": values, "scores": counts},
            "scores": scores,
            "x_values": recall,
            "y_values": precision,
            "threshold": thresh,
            "auc": ap,
        }
        internal_obj = {"overall": overall_dict, "specific": None}

        return PRExplanation(
            "perf",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
        )


class ROC(ExplainerMixin):
    available_explanations = ["perf"]
    explainer_type = "perf"

    def __init__(self, predict_fn, feature_names=None, feature_types=None, **kwargs):
        self.predict_fn = predict_fn
        self.kwargs = kwargs
        self.feature_names = feature_names
        self.feature_types = feature_types

    def explain_perf(self, X, y, name=None):
        if name is None:
            name = gen_name_from_class(self)

        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )
        predict_fn = unify_predict_fn(self.predict_fn, X)
        scores = predict_fn(X)

        fpr, tpr, thresh = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)

        abs_residuals = np.abs(y - scores)
        counts, values = np.histogram(abs_residuals, bins="doane")

        overall_dict = {
            "type": "perf_curve",
            "density": {"names": values, "scores": counts},
            "scores": scores,
            "x_values": fpr,
            "y_values": tpr,
            "threshold": thresh,
            "auc": roc_auc,
        }
        internal_obj = {"overall": overall_dict, "specific": None}

        return ROCExplanation(
            "perf",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
        )


class ROCExplanation(ExplanationMixin):
    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        if key is None:
            return self._internal_obj["overall"]
        return None

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        return plot_performance_curve(
            data_dict,
            xtitle="FPR",
            ytitle="TPR",
            baseline=True,
            title="ROC Curve: " + self.name,
            auc_prefix="AUC",
        )


class PRExplanation(ExplanationMixin):
    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        if key is None:
            return self._internal_obj["overall"]
        return None

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        return plot_performance_curve(
            data_dict,
            xtitle="Recall",
            ytitle="Precision",
            baseline=False,
            title="PR Curve: " + self.name,
            auc_prefix="Average Precision",
        )
