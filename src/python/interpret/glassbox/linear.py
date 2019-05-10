# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import gen_name_from_class, gen_global_selector, gen_local_selector
from ..utils import perf_dict, hist_per_column
from ..utils import unify_data
from ..visual.plot import sort_take, plot_horizontal_bar

from abc import abstractmethod
from sklearn.base import is_classifier
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.linear_model import Lasso as SKLinear


class BaseLinear:
    """ Logistic regression.

    Currently wrapper around LogisticRegression in scikit-learn.

    https://github.com/scikit-learn/scikit-learn

    """

    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLinear, **kwargs
    ):
        """ Initializes logistic regression.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.linear_class = linear_class
        self.kwargs = kwargs

    @abstractmethod
    def _model(self):
        # This method should be overriden
        return None

    def fit(self, X, y):
        """ Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        X, y, self.feature_names, self.feature_types = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        sk_model_ = self._model()
        sk_model_.fit(X, y)

        self.X_mins_ = np.min(X, axis=0)
        self.X_maxs_ = np.max(X, axis=0)
        self.categorical_uniq_ = {}

        for i, feature_type in enumerate(self.feature_types):
            if feature_type == "categorical":
                self.categorical_uniq_[i] = list(sorted(set(X[:, i])))

        self.global_selector = gen_global_selector(
            X, self.feature_names, self.feature_types, None
        )
        self.bin_counts_, self.bin_edges_ = hist_per_column(X, self.feature_types)
        return self

    def predict(self, X):
        """ Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict(X)

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        if name is None:
            name = gen_name_from_class(self)
        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        sk_model_ = self._model()

        if is_classifier(self):
            predictions = self.predict_proba(X)[:, 1]
            intercept = sk_model_.intercept_[0]
            coef = sk_model_.coef_[0]
        else:
            predictions = self.predict(X)
            intercept = sk_model_.intercept_
            coef = sk_model_.coef_

        data_dicts = []
        for i, instance in enumerate(X):
            scores = list(coef * instance)
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            data_dict["perf"] = perf_dict(y, predictions, i)

            # Names/scores
            data_dict["names"] = self.feature_names
            data_dict["scores"] = scores

            # Values
            data_dict["values"] = instance

            data_dict["extra"] = {
                "names": ["Intercept"],
                "scores": [intercept],
                "values": [1],
            }
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}

        selector = gen_local_selector(X, y, predictions)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )

    def explain_global(self, name=None):
        """ Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        sk_model_ = self._model()
        if is_classifier(self):
            intercept = sk_model_.intercept_[0]
            coef = sk_model_.coef_[0]
        else:
            intercept = sk_model_.intercept_
            coef = sk_model_.coef_

        overall_data_dict = {
            "names": self.feature_names,
            "scores": list(coef),
            "extra": {"names": ["Intercept"], "scores": [intercept]},
        }

        specific_data_dicts = []
        for index, feature in enumerate(self.feature_names):
            feat_min = self.X_mins_[index]
            feat_max = self.X_maxs_[index]
            feat_coef = coef[index]

            feat_type = self.feature_types[index]

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

        internal_obj = {"overall": overall_data_dict, "specific": specific_data_dicts}
        return LinearExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )


class LinearExplanation(FeatureValueExplanation):
    """ Visualizes specifically for Linear methods.
    """

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

        super(LinearExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        data_dict = self.data(key)
        if data_dict is None:
            return None

        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            figure = plot_horizontal_bar(
                data_dict, title="Overall Importance:<br>Coefficients"
            )
            return figure

        return super().visualize(key)


class LinearRegression(BaseLinear, RegressorMixin, ExplainerMixin):
    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLinear, **kwargs
    ):
        super().__init__(feature_names, feature_types, linear_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        self.sk_model_ = self.linear_class(**self.kwargs)
        return super().fit(X, y)


class LogisticRegression(BaseLinear, ClassifierMixin, ExplainerMixin):
    def __init__(
        self, feature_names=None, feature_types=None, linear_class=SKLinear, **kwargs
    ):
        super().__init__(feature_names, feature_types, linear_class, **kwargs)

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        self.sk_model_ = SKLogistic(**self.kwargs)
        return super().fit(X, y)

    def predict_proba(self, X):
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        return self._model().predict_proba(X)
