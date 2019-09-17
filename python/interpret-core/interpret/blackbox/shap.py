# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
from ..utils.shap import shap_explain_local

from ..api.base import ExplainerMixin
from ..utils import unify_predict_fn, unify_data
import warnings


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

        import shap

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
        return shap_explain_local(self, X, y=y, name=name)
