# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..utils._shap_common import shap_explain_local

from interpret.api.base import ExplainerMixin

import numpy as np
from ..utils._clean_x import preclean_X
from ..utils._unify_data import unify_data


class ShapTree(ExplainerMixin):
    """Exposes tree specific SHAP approximation, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/slundberg/shap
    """

    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(self, model, data, feature_names=None, feature_types=None, **kwargs):
        """Initializes class.

        Args:
            model: A tree object that works with Tree SHAP.
            data: Data used to initialize SHAP with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs that will be sent to shap.TreeExplainer
        """

        from shap import TreeExplainer

        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.feature_names_in_ = None
        self.feature_types_in_ = None

        if data is not None:
            # data can be None for some tree SHAP options

            data, n_samples = preclean_X(data, feature_names, feature_types)

            data, self.feature_names_in_, self.feature_types_in_ = unify_data(
                data, n_samples, feature_names, feature_types, False, 0
            )

            # SHAP does not support string categoricals, and np.object_ is slower,
            # so convert to np.float64 until we implement some automatic categorical handling
            data = data.astype(np.float64, order="C", copy=False)

        self.shap_ = TreeExplainer(model, data, **kwargs)

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
        new_kwargs = {"check_additivity": False}
        new_kwargs.update(kwargs)
        return shap_explain_local(
            self,
            X,
            y,
            name,
            True,
            **new_kwargs,
        )
