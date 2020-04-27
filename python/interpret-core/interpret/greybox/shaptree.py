# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..utils.shap import shap_explain_local
from sklearn.base import is_classifier

from ..api.base import ExplainerMixin
from ..utils import unify_predict_fn, unify_data


class ShapTree(ExplainerMixin):
    """ Exposes tree specific SHAP approximation, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/slundberg/shap
    """

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
        """ Initializes class.

        Args:
            model: A tree object that works with Tree SHAP.
            data: Data used to initialize SHAP with.
            sampler: Currently unused. Due for deprecation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            explain_kwargs: Currently unused. Due for deprecation.
            n_jobs: Number of jobs to run in parallel.
            **kwargs: Kwargs that will be sent to SHAP at initialization time.
        """
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
