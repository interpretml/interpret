# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation

from ..utils import gen_name_from_class, gen_local_selector
from ..utils import perf_dict
from ..utils import unify_data, unify_predict_fn
from lime.lime_tabular import LimeTabularExplainer
import warnings


# TODO: Make kwargs explicit.
class LimeTabular(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
        **kwargs
    ):

        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )
        self.predict_fn = unify_predict_fn(predict_fn, self.data)

        if sampler is not None:  # pragma: no cover
            warnings.warn("Sampler interface not currently supported.")

        self.sampler = sampler
        self.explain_kwargs = explain_kwargs

        self.kwargs = kwargs
        final_kwargs = {"mode": "regression"}
        if self.feature_names:
            final_kwargs["feature_names"] = self.feature_names
        final_kwargs.update(self.kwargs)

        self.lime = LimeTabularExplainer(self.data, **final_kwargs)

    def explain_local(self, X, y=None, name=None):
        if name is None:
            name = gen_name_from_class(self)
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        predictions = self.predict_fn(X)
        pred_fn = self.predict_fn

        data_dicts = []
        for i, instance in enumerate(X):
            lime_explanation = self.lime.explain_instance(
                instance, pred_fn, **self.explain_kwargs
            )

            names = []
            scores = []
            values = []
            feature_idx_imp_pairs = lime_explanation.as_map()[1]
            for feat_idx, imp in feature_idx_imp_pairs:
                names.append(self.feature_names[feat_idx])
                scores.append(imp)
                values.append(instance[feat_idx])
            intercept = lime_explanation.intercept[1]

            data_dict = {
                "type": "univariate",
                "names": names,
                "perf": perf_dict(y, predictions, i),
                "scores": scores,
                "values": values,
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}
        selector = gen_local_selector(X, y, predictions)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )
