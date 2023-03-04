# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation

from ..utils import gen_name_from_class, gen_local_selector
from ..utils import gen_perf_dicts
import warnings

import numpy as np
from ..utils._binning import (
    determine_min_cols,
    clean_X,
    determine_n_classes,
    unify_predict_fn,
    unify_data2,
    clean_dimensions,
    typify_classification,
)


# TODO: Make kwargs explicit.
class LimeTabular(ExplainerMixin):
    """Exposes LIME tabular explainer from lime package, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/marcotcr/lime/blob/master/citation.bib
    """

    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(self, model, data, feature_names=None, feature_types=None, **kwargs):
        """Initializes class.

        Args:
            model: model or prediction function of model (predict_proba for classification or predict for regression)
            data: Data used to initialize LIME with.
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs that will be sent to lime
        """

        from lime.lime_tabular import LimeTabularExplainer

        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types

        min_cols = determine_min_cols(feature_names, feature_types)
        data, n_samples = clean_X(data, min_cols, None)

        predict_fn, n_classes = determine_n_classes(model, data, n_samples)
        if 3 <= n_classes:
            raise Exception("multiclass LIME not supported")
        predict_fn = unify_predict_fn(predict_fn, data, 1 if n_classes == 2 else -1)

        data, self.feature_names_in_, self.feature_types_in_ = unify_data2(
            data, n_samples, feature_names, feature_types, False, 0
        )

        # LIME does not support string categoricals, and np.object_ is slower,
        # so convert to np.float64 until we implement some automatic categorical handling
        data = data.astype(np.float64, order="C", copy=False)

        # rewrite these even if the user specified them
        kwargs = kwargs.copy()
        kwargs["mode"] = "regression"
        kwargs["feature_names"] = self.feature_names_in_

        self.lime_ = LimeTabularExplainer(data, **kwargs)

    def explain_local(self, X, y=None, name=None, **kwargs):
        """Generates local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
            **kwargs: Kwargs that will be sent to lime

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        if name is None:
            name = gen_name_from_class(self)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        min_cols = determine_min_cols(self.feature_names_in_, self.feature_types_in_)
        X, n_samples = clean_X(X, min_cols, n_samples)

        predict_fn, n_classes = determine_n_classes(self.model, X, n_samples)
        if 3 <= n_classes:
            raise Exception("multiclass LIME not supported")
        predict_fn = unify_predict_fn(predict_fn, X, 1 if n_classes == 2 else -1)

        X, _, _ = unify_data2(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        # LimeTabularExplainer does not support string categoricals, and np.object_ is slower,
        # so convert to np.float64 until we implement some automatic categorical handling
        X = X.astype(np.float64, order="C", copy=False)

        if y is not None:
            if 0 <= n_classes:
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        predictions = predict_fn(X)

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, False)
        for i, instance in enumerate(X):
            lime_explanation = self.lime_.explain_instance(
                instance, predict_fn, **kwargs
            )

            names = []
            scores = []
            values = []
            feature_idx_imp_pairs = lime_explanation.as_map()[1]
            for feat_idx, imp in feature_idx_imp_pairs:
                names.append(self.feature_names_in_[feat_idx])
                scores.append(imp)
                values.append(instance[feat_idx])
            intercept = lime_explanation.intercept[1]

            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

            scores_list.append(scores)
            perf_list.append(perf_dict_obj)

            data_dict = {
                "type": "univariate",
                "names": names,
                "perf": perf_dict_obj,
                "scores": scores,
                "values": values,
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            data_dicts.append(data_dict)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "local_feature_importance",
                    "value": {
                        "scores": scores_list,
                        "intercept": intercept,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )
        selector = gen_local_selector(data_dicts, is_classification=False)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )
