# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier
from sklearn.linear_model import LinearRegression as SKLinear
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.utils.validation import check_is_fitted

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils._clean_simple import clean_dimensions, typify_classification
from ..utils._clean_x import preclean_X
from ..utils._explanation import (
    gen_global_selector,
    gen_local_selector,
    gen_name_from_class,
    gen_perf_dicts,
)
from ..utils._unify_data import unify_data


@dataclass
class LinearInputTags:
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = True
    categorical: bool = False
    string: bool = True
    dict: bool = True
    positive_only: bool = False
    allow_nan: bool = False
    pairwise: bool = False


@dataclass
class LinearTargetTags:
    required: bool = True
    one_d_labels: bool = True
    two_d_labels: bool = False
    positive_only: bool = False
    multi_output: bool = False
    single_output: bool = True


@dataclass
class LinearClassifierTags:
    poor_score: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class LinearRegressorTags:
    poor_score: bool = False


@dataclass
class LinearTags:
    estimator_type: Optional[str] = None
    target_tags: LinearTargetTags = field(default_factory=LinearTargetTags)
    transformer_tags: None = None
    classifier_tags: Optional[LinearClassifierTags] = None
    regressor_tags: Optional[LinearRegressorTags] = None
    array_api_support: bool = True
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: LinearInputTags = field(default_factory=LinearInputTags)


class BaseLinear(ExplainerMixin):
    """Base linear model.

    Currently wrapper around linear models in scikit-learn.

    https://github.com/scikit-learn/scikit-learn

    """

    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLinear, **kwargs
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            linear_class: A scikit-learn linear class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.linear_class = linear_class
        self.kwargs = kwargs

    @abstractmethod
    def _model(self):
        # This method should be overridden.
        return None

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            msg = "y must be 1 dimensional"
            raise ValueError(msg)
        if len(y) == 0:
            msg = "y cannot have 0 samples"
            raise ValueError(msg)

        if is_classifier(self):
            y = typify_classification(y)
        else:
            y = y.astype(np.float64, copy=False)

        X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))

        X, self.feature_names_in_, self.feature_types_in_ = unify_data(
            X, n_samples, self.feature_names, self.feature_types, False, 0
        )

        model = self._model()
        model.fit(X, y)

        self.n_features_in_ = len(self.feature_names_in_)
        if is_classifier(self):
            self.classes_ = model.classes_

        self.X_mins_ = np.min(X, axis=0)
        self.X_maxs_ = np.max(X, axis=0)
        self.categorical_uniq_ = {}

        for i, feature_type in enumerate(self.feature_types_in_):
            if feature_type in ("nominal", "ordinal"):
                self.categorical_uniq_[i] = sorted(set(X[:, i]))

        unique_val_counts = np.zeros(len(self.feature_names_in_), dtype=np.int64)
        for col_idx in range(len(self.feature_names_in_)):
            X_col = X[:, col_idx]
            unique_val_counts[col_idx] = len(np.unique(X_col))

        self.global_selector_ = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            unique_val_counts,
            None,
        )
        self.bin_counts_, self.bin_edges_ = _hist_per_column(X, self.feature_types_in_)

        self.has_fitted_ = True

        return self

    def predict(self, X):
        """Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        return self._model().predict(X)

    def explain_local(self, X, y=None, name=None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                msg = "y must be 1 dimensional"
                raise ValueError(msg)
            n_samples = len(y)

            if is_classifier(self):
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        X, n_samples = preclean_X(
            X, self.feature_names_in_, self.feature_types_in_, n_samples
        )

        if n_samples == 0:
            # TODO: we could probably handle this case
            msg = "X has zero samples"
            raise ValueError(msg)

        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        model = self._model()

        classes = None
        is_classification = is_classifier(self)
        intercept = model.intercept_
        coef = model.coef_
        if is_classification:
            classes = self.classes_
            predictions = self.predict_proba(X)
            if len(classes) == 2:
                predictions = predictions[:, 1]
                intercept = intercept[0]
                coef = coef[0]
        else:
            predictions = self.predict(X)

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, is_classification, classes)
        for i, instance in enumerate(X):
            scores = list(coef * instance)
            scores_list.append(scores)
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]
            data_dict["perf"] = perf_dict_obj
            perf_list.append(perf_dict_obj)

            # Names/scores
            data_dict["names"] = self.feature_names_in_
            data_dict["scores"] = scores

            # Values
            data_dict["values"] = instance

            data_dict["extra"] = {
                "names": ["Intercept"],
                "scores": [intercept],
                "values": [1],
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

        selector = gen_local_selector(data_dicts, is_classification=is_classification)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )

    def explain_global(self, name=None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """

        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        model = self._model()
        if is_classifier(self):
            intercept = model.intercept_[0]
            coef = model.coef_[0]
        else:
            intercept = model.intercept_
            coef = model.coef_

        overall_data_dict = {
            "names": self.feature_names_in_,
            "scores": list(coef),
            "extra": {"names": ["Intercept"], "scores": [intercept]},
        }

        specific_data_dicts = []
        for index, _feature in enumerate(self.feature_names_in_):
            feat_min = self.X_mins_[index]
            feat_max = self.X_maxs_[index]
            feat_coef = coef[index]

            feat_type = self.feature_types_in_[index]

            if feat_type == "continuous":
                # Generate x, y points to plot from coef for continuous features
                grid_points = np.linspace(feat_min, feat_max, 30)
            else:
                grid_points = np.array(self.categorical_uniq_[index])

            y_scores = feat_coef * grid_points

            data_dict = {
                "names": grid_points,
                "scores": y_scores,
                "density": {
                    "scores": self.bin_counts_[index],
                    "names": self.bin_edges_[index],
                },
            }

            specific_data_dicts.append(data_dict)

        internal_obj = {
            "overall": overall_data_dict,
            "specific": specific_data_dicts,
            "mli": [
                {
                    "explanation_type": "global_feature_importance",
                    "value": {"scores": list(coef), "intercept": intercept},
                }
            ],
        }
        return LinearExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=self.global_selector_,
        )

    def __sklearn_tags__(self):
        return LinearTags()


class LinearExplanation(FeatureValueExplanation):
    """Visualizes specifically for Linear methods."""

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        super().__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ..visual.plot import (
            get_explanation_index,
            get_sort_indexes,
            mli_plot_horizontal_bar,
            mli_sort_take,
            plot_horizontal_bar,
            sort_take,
        )

        if isinstance(key, tuple) and len(key) == 2:
            provider, key = key
            if (
                provider == "mli"
                and "mli" in self.data(-1)
                and self.explanation_type == "global"
            ):
                explanation_list = self.data(-1)["mli"]
                explanation_index = get_explanation_index(
                    explanation_list, "global_feature_importance"
                )
                scores = explanation_list[explanation_index]["value"]["scores"]
                sort_indexes = get_sort_indexes(
                    scores, sort_fn=lambda x: -abs(x), top_n=15
                )
                sorted_scores = mli_sort_take(
                    scores, sort_indexes, reverse_results=True
                )
                sorted_names = mli_sort_take(
                    self.feature_names, sort_indexes, reverse_results=True
                )
                return mli_plot_horizontal_bar(
                    sorted_scores,
                    sorted_names,
                    title="Overall Importance:<br>Coefficients",
                )
            # pragma: no cover
            msg = f"Visual provider {provider} not supported"
            raise RuntimeError(msg)
        data_dict = self.data(key)
        if data_dict is None:
            return None

        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            return plot_horizontal_bar(
                data_dict, title="Overall Importance:<br>Coefficients"
            )

        return super().visualize(key)


class LinearRegression(RegressorMixin, BaseLinear):
    """Linear regression.

    Currently wrapper around linear models in scikit-learn: https://github.com/scikit-learn/scikit-learn
    """

    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLinear, **kwargs
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            linear_class: A scikit-learn linear class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        super().__init__(feature_names, feature_types, linear_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.sk_model_ = self.linear_class(**self.kwargs)
        return super().fit(X, y)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = LinearRegressorTags()
        return tags


class LogisticRegression(ClassifierMixin, BaseLinear):
    """Logistic regression.

    Currently wrapper around linear models in scikit-learn: https://github.com/scikit-learn/scikit-learn
    """

    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLogistic, **kwargs
    ):
        """Initializes class.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            linear_class: A scikit-learn linear class.
            **kwargs: Kwargs pass to linear class at initialization time.
        """
        super().__init__(feature_names, feature_types, linear_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.sk_model_ = self.linear_class(**self.kwargs)
        return super().fit(X, y)

    def predict_proba(self, X):
        """Probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        X, _, _ = unify_data(
            X, n_samples, self.feature_names_in_, self.feature_types_in_, False, 0
        )

        return self._model().predict_proba(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = LinearClassifierTags()
        return tags


def _hist_per_column(arr, feature_types=None):
    counts = []
    bin_edges = []

    if feature_types is not None:
        for i, feat_type in enumerate(feature_types):
            if feat_type == "continuous":
                count, bin_edge = np.histogram(arr[:, i], bins="doane")
                counts.append(count)
                bin_edges.append(bin_edge)
            elif feat_type in ("nominal", "ordinal"):
                # Todo: check if this call
                bin_edge, count = np.unique(arr[:, i], return_counts=True)
                counts.append(count)
                bin_edges.append(bin_edge)
    else:
        for i in range(arr.shape[1]):
            count, bin_edge = np.histogram(arr[:, i], bins="doane")
            counts.append(count)
            bin_edges.append(bin_edge)
    return counts, bin_edges
