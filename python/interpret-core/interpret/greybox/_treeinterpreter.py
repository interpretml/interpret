# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils._explanation import gen_name_from_class, gen_perf_dicts, gen_local_selector

import numpy as np
from ..utils._clean_x import preclean_X
from ..utils._clean_simple import clean_dimensions, typify_classification

from ..utils._unify_predict import determine_classes, unify_predict_fn
from ..utils._unify_data import unify_data


class TreeInterpreter(ExplainerMixin):
    """Provides 'Tree Explainer' algorithm for specific sklearn trees.

    Wrapper around andosa/treeinterpreter github package.

    https://github.com/andosa/treeinterpreter

    Currently supports (copied from README.md):

    - DecisionTreeRegressor
    - DecisionTreeClassifier
    - ExtraTreeRegressor
    - ExtraTreeClassifier
    - RandomForestRegressor
    - RandomForestClassifier
    - ExtraTreesRegressor
    - ExtraTreesClassifier

    """

    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(
        self,
        model,
        data=None,
        feature_names=None,
        feature_types=None,
    ):
        """Initializes class.

        Args:
            model: A scikit-learn tree object
            data: mostly ignored. Only included for conformance to the greybox API
                  if data is provided though we use it to determine the feature names and types
            feature_names: List of feature names.
            feature_types: List of feature types.
        """

        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.feature_names_in_ = None
        self.feature_types_in_ = None

        if data is not None:
            # if the user provides data, we use it as a larger corpus than X
            data, n_samples = preclean_X(data, feature_names, feature_types)

            _, self.feature_names_in_, self.feature_types_in_ = unify_data(
                data, n_samples, feature_names, feature_types, False, 0
            )

    def explain_local(self, X, y=None, name=None, **kwargs):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
            **kwargs: Kwargs that will be sent to treeinterpreter

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        from treeinterpreter import treeinterpreter as ti

        if name is None:
            name = gen_name_from_class(self)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        feature_names = (
            self.feature_names
            if self.feature_names_in_ is None
            else self.feature_names_in_
        )
        feature_types = (
            self.feature_types
            if self.feature_types_in_ is None
            else self.feature_types_in_
        )

        X, n_samples = preclean_X(X, feature_names, feature_types, n_samples)

        predict_fn, n_classes, classes = determine_classes(self.model, X, n_samples)
        predict_fn = unify_predict_fn(predict_fn, X, -1)

        X, feature_names, feature_types = unify_data(
            X, n_samples, feature_names, feature_types, False, 0
        )

        is_classification = 0 <= n_classes
        if y is not None:
            if is_classification:
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        predictions = predict_fn(X)

        _, biases, contributions = ti.predict(self.model, X, **kwargs)

        data_dicts = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, is_classification, classes)
        for i, instance in enumerate(X):
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]
            data_dict["perf"] = perf_dict_obj
            perf_list.append(perf_dict_obj)

            # Names/scores
            data_dict["names"] = feature_names
            if n_classes == 2:
                data_dict["scores"] = contributions[i, :, 1]
            else:
                data_dict["scores"] = contributions[i, :]

            # Values
            data_dict["values"] = instance
            # TODO: Value 1 doesn't make sense for this bias, consider refactoring values to take None.
            bias = biases[0, 1] if n_classes == 2 else biases[0]
            data_dict["extra"] = {"names": ["Bias"], "scores": [bias], "values": [1]}
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}
        selector = gen_local_selector(data_dicts, is_classification=is_classification)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )
