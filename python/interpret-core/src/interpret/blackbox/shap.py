# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
from ..utils.shap import shap_explain_local

from ..api.base import ExplainerMixin
import warnings

import numpy as np
from ..utils._binning import (
    preclean_X,
)
from ..utils._unify_predict import determine_classes, unify_predict_fn
from ..utils._unify_data import unify_data


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
            **kwargs: Kwargs that will be sent to shap.KernelExplainer
        """

        from shap import KernelExplainer

        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types

        data, n_samples = preclean_X(data, feature_names, feature_types)

        predict_fn, n_classes, _ = determine_classes(model, data, n_samples)
        if 3 <= n_classes:
            raise Exception("multiclass SHAP not supported")
        predict_fn = unify_predict_fn(predict_fn, data, 1 if n_classes == 2 else -1)

        data, self.feature_names_in_, self.feature_types_in_ = unify_data(
            data, n_samples, feature_names, feature_types, False, 0
        )

        # SHAP does not support string categoricals, and np.object_ is slower,
        # so convert to np.float64 until we implement some automatic categorical handling
        data = data.astype(np.float64, order="C", copy=False)

        self.shap_ = KernelExplainer(predict_fn, data, **kwargs)

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
