# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import unify_predict_fn, unify_data
from ..utils import perf_dict, gen_name_from_class, gen_local_selector
import warnings

import shap


class ShapKernel(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs=None,
        n_jobs=1,
        **kwargs
    ):

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = unify_predict_fn(predict_fn, self.data)
        self.n_jobs = n_jobs

        if sampler is not None:  # pragma: no cover
            warnings.warn("Sampler interface not currently supported.")
        self.sampler = sampler
        self.explain_kwargs = explain_kwargs
        self.kwargs = kwargs

        self.shap = shap.KernelExplainer(self.predict_fn, data, **self.kwargs)

    def explain_local(self, X, y=None, name=None):
        if name is None:
            name = gen_name_from_class(self)
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        all_shap_values = self.shap.shap_values(X)
        predictions = self.predict_fn(X)

        data_dicts = []
        scores_list = all_shap_values
        perf_list = []
        for i, instance in enumerate(X):
            shap_values = all_shap_values[i]
            perf_dict_obj = perf_dict(y, predictions, i)

            perf_list.append(perf_dict_obj)

            data_dict = {
                "type": "univariate",
                "names": self.feature_names,
                "perf": perf_dict_obj,
                "scores": shap_values,
                "values": instance,
                "extra": {
                    "names": ["Base Value"],
                    "scores": [self.shap.expected_value],
                    "values": [1],
                },
            }
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts, "mli": [
            {
                "explanation_type": "local_feature_importance",
                "value": {
                    "scores": scores_list,
                    "intercept": self.shap.expected_value,
                    "perf": perf_list
                }
            }]
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {
                    "dataset_x": X,
                    "dataset_y": y
                }
            }
        )
        selector = gen_local_selector(X, y, predictions)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )
