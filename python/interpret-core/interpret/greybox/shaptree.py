# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.utils.shap import shap_explain_local
from sklearn.base import is_classifier

from interpret.api.base import ExplainerMixin

import numpy as np
import copy
from ..utils._binning import (
    determine_min_cols,
    clean_X,
    determine_n_classes,
    unify_predict_fn2,
    unify_data2,
)

class ShapTree(ExplainerMixin):
    """Exposes tree specific SHAP approximation, in interpret API form.
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
        **kwargs
    ):
        """Initializes class.

        Args:
            model: A tree object that works with Tree SHAP.
            data: Data used to initialize SHAP with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs that will be sent to SHAP
        """
        import shap

        min_cols = determine_min_cols(feature_names, feature_types)
        data, n_samples = clean_X(data, min_cols, None)

        predict_fn, self.n_classes = determine_n_classes(model, data, n_samples)

        self.predict_fn = unify_predict_fn2(self.n_classes, predict_fn, data)

        data, self.feature_names, self.feature_types = unify_data2(
            data, n_samples, feature_names, feature_types, False, 0
        )

        # shap.TreeExplainer currently needs data to be np.float64
        data = data.astype(np.float64, order="C", copy=False)

        self.shap = shap.TreeExplainer(model, data, **kwargs)

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
        # NOTE: Check additivity is set to false by default as there is a problem with Mac OS that
        # doesn't always reach the specified precision.
        kwargs = copy.deepcopy(kwargs)
        kwargs["check_additivity"] = False
        return shap_explain_local(
            self,
            X,
            y,
            name,
            0 <= self.n_classes,
            **kwargs,
        )
