# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..utils.shap import shap_explain_local
from sklearn.base import is_classifier

from ..api.base import ExplainerMixin
from ..utils import unify_predict_fn, unify_data


class ShapTree(ExplainerMixin):
    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(
        self,
        model,
        data,
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

        self.model = model
        self.is_classifier = is_classifier(self.model)
        if is_classifier:
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        self.predict_fn = unify_predict_fn(predict_fn, self.data)

        self.n_jobs = n_jobs

        self.explain_kwargs = explain_kwargs
        self.kwargs = kwargs

        self.shap = shap.TreeExplainer(model, data, **self.kwargs)

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        return shap_explain_local(
            self, X, y=y, name=name, is_classification=self.is_classifier
        )
