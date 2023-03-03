# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
from ..utils.shap import shap_explain_local

from ..api.base import ExplainerMixin
import warnings

from ..utils._binning import (
    determine_min_cols,
    clean_X,
    determine_n_classes,
    unify_predict_fn2,
    unify_data2,
)


class ShapKernel(ExplainerMixin):
    """Exposes SHAP kernel explainer from shap package, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/slundberg/shap
    """

    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(self, model, data, feature_names=None, feature_types=None, **kwargs):
        """Initializes class.

        Args:
            model: model or prediction function of model (predict_proba for classification or predict for regression)
            data: Data used to initialize SHAP with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs that will be sent to SHAP at initialization time.
        """

        import shap

        min_cols = determine_min_cols(feature_names, feature_types)
        data, n_samples = clean_X(data, min_cols, None)

        predict_fn, self.n_classes = determine_n_classes(model, data, n_samples)

        self.predict_fn = unify_predict_fn2(self.n_classes, predict_fn, data)

        data, self.feature_names, self.feature_types = unify_data2(
            data, n_samples, feature_names, feature_types, False, 0
        )

        self.shap = shap.KernelExplainer(self.predict_fn, data, **kwargs)

    def explain_local(self, X, y=None, name=None, **kwargs):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
            **kwargs: Kwargs that will be sent to SHAP

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        return shap_explain_local(self, X, y, name, False, **kwargs)
