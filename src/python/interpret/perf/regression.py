# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data, gen_name_from_class, unify_predict_fn
from ..visual.plot import plot_density

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class RegressionPerf(ExplainerMixin):
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

        mse = mean_squared_error(y, scores)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, scores)
        r2 = r2_score(y, scores)
        residuals = y - scores

        # abs_residuals = np.abs(y - scores)
        counts, values = np.histogram(residuals, bins="doane")

        overall_dict = {
            "type": "perf_curve",
            "density": {"names": values, "scores": counts},
            "scores": scores,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "residuals": residuals,
        }
        internal_obj = {"overall": overall_dict, "specific": None}

        return RegressionExplanation(
            "perf",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
        )


class RegressionExplanation(ExplanationMixin):
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

        rmse = data_dict["rmse"]
        r2 = data_dict["r2"]

        title = "{0} <br> RMSE = {1:.2f}" + " | R<sup>2</sup> = {2:.2f}"
        title = title.format(self.name, rmse, r2)
        density_fig = plot_density(data_dict["density"], title=title)
        return density_fig
