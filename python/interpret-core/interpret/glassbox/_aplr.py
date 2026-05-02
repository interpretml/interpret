# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license
from warnings import warn

import numpy as np

try:
    from pandas import DataFrame as _DataFrameType
    from pandas import Series as _SeriesType
except ImportError:

    class _DataFrameType:
        pass

    class _SeriesType:
        pass


from ..core.sklearn import (
    SKBaseEstimator,
    SKClassifierMixin,
    SKNotFittedError,
    SKRegressorMixin,
)
from ..core.base import LocalExplainer, GlobalExplainer
from ..core.templates import FeatureValueExplanation
from ..utils._clean_simple import clean_dimensions
from ..utils._explanation import (
    gen_global_selector,
    gen_local_selector,
    gen_name_from_class,
    gen_perf_dicts,
)

FloatVector = np.ndarray
FloatMatrix = np.ndarray
IntVector = np.ndarray
IntMatrix = np.ndarray


try:
    from aplr import APLRRegressor as APLRRegressorNative
except ImportError:

    class APLRRegressorNative:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "APLRRegressor requires the 'aplr' package. Please install it using 'pip install aplr'"
            )


class APLRRegressor(
    SKRegressorMixin,
    LocalExplainer,
    GlobalExplainer,
    SKBaseEstimator,
    APLRRegressorNative,
):
    """APLR Regressor."""

    def __init__(self, **kwargs):
        """Initializes class.

        Args:
            **kwargs: Kwargs passed to APLRRegressor at initialization time.
        """

        # TODO: add feature_names and feature_types to conform to glassbox API
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return APLRRegressorNative.get_params(self)

    def set_params(self, **params):
        APLRRegressorNative.set_params(self, **params)
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.target_tags.required = True
        return tags

    def predict(self, X):
        """Predicts target values."""
        if not hasattr(self, "n_features_in_"):
            raise SKNotFittedError(
                "This model has not been fitted yet. Call 'fit' first."
            )
        return super().predict(X)

    def fit(self, X, y, **kwargs):
        """Fits model."""
        X_names = kwargs.get("X_names")

        self.bin_counts_, self.bin_edges_ = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X, X_names=X_names)
        self.n_features_in_ = len(self.feature_names_in_)

        super().fit(
            X,
            y,
            **kwargs,
        )
        return self

    def explain_global(self, name: str | None = None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        overall_dict = {
            "names": self.get_unique_term_affiliations(),
            "scores": self.get_feature_importance(),
        }

        data_dicts = []
        feature_list = []
        density_list = []
        keep_idxs = []
        predictors_in_each_affiliation = (
            self.get_base_predictors_in_each_unique_term_affiliation()
        )
        unique_values = []
        for affiliation_index, affiliation in enumerate(
            self.get_unique_term_affiliations()
        ):
            shape = self.get_unique_term_affiliation_shape(affiliation)
            predictor_indexes_used = predictors_in_each_affiliation[affiliation_index]
            is_main_effect: bool = len(predictor_indexes_used) == 1
            is_two_way_interaction: bool = len(predictor_indexes_used) == 2
            if is_main_effect:
                density_dict = {
                    "names": self.bin_edges_[predictor_indexes_used[0]],
                    "scores": self.bin_counts_[predictor_indexes_used[0]],
                }
                feature_dict = {
                    "type": "univariate",
                    "feature_name": self.feature_names_in_[predictor_indexes_used[0]],
                    "names": shape[:, 0],
                    "scores": shape[:, 1],
                }
                data_dict = {
                    "type": "univariate",
                    "feature_name": self.feature_names_in_[predictor_indexes_used[0]],
                    "names": shape[:, 0],
                    "scores": shape[:, 1],
                    "density": density_dict,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)
                data_dicts.append(data_dict)
                keep_idxs.append(affiliation_index)
                unique_values.append(self.unique_values_in_[predictor_indexes_used[0]])
            elif is_two_way_interaction:
                feature_dict = {
                    "type": "interaction",
                    "feature_names": [
                        self.feature_names_in_[idx] for idx in predictor_indexes_used
                    ],
                    "left_names": shape[:, 0],
                    "right_names": shape[:, 1],
                    "scores": shape[:, 2],
                }
                data_dict = {
                    "type": "interaction",
                    "feature_names": [
                        self.feature_names_in_[idx] for idx in predictor_indexes_used
                    ],
                    "left_names": shape[:, 0],
                    "right_names": shape[:, 1],
                    "scores": shape[:, 2],
                }
                feature_list.append(feature_dict)
                density_list.append({})
                data_dicts.append(data_dict)
                keep_idxs.append(affiliation_index)
                unique_values.append(np.nan)
            else:  # pragma: no cover
                warn(
                    f"Dropping term {affiliation} from explanation "
                    "since we can't graph more than 2 dimensions."
                )
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "aplr_global",
                    "value": {"feature_list": feature_list},
                },
                {
                    "explanation_type": "density",
                    "value": {"density": density_list},
                },
            ],
        }
        term_names = [self.get_unique_term_affiliations()[i] for i in keep_idxs]
        term_types = [feature_dict["type"] for feature_dict in feature_list]
        selector = gen_global_selector(
            len(keep_idxs),
            term_names,
            term_types,
            unique_values,
            None,
        )
        return APLRExplanation(
            "global",
            internal_obj,
            feature_names=term_names,
            feature_types=term_types,
            name=name,
            selector=selector,
        )

    def explain_local(
        self, X: FloatMatrix, y: FloatVector | None = None, name: str | None = None
    ):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        pred = self.predict(X)
        term_names = self.get_unique_term_affiliations()
        explanations = self.calculate_local_feature_contribution(X)

        data_dicts = []
        perf_list = []

        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                msg = f"y must be 1 dimensional, but got {y.ndim} dimensions with shape {y.shape}"
                raise ValueError(msg)
            y = y.astype(np.float64, copy=False)
        X_values = create_values(X, explanations, term_names, self.feature_names_in_)
        perf_list = gen_perf_dicts(pred, y, False)

        for data, sample_scores, perf in zip(X_values, explanations, perf_list):
            values = ["" if np.isnan(val) else val for val in data.tolist()]
            data_dict = {
                "type": "univariate",
                "names": term_names,
                "scores": list(sample_scores),
                "values": values,
                "extra": {
                    "names": ["Intercept"],
                    "scores": [self.get_intercept()],
                    "values": [1],
                },
                "perf": perf,
            }
            data_dicts.append(data_dict)

        selector = gen_local_selector(data_dicts, is_classification=False)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
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

        term_types = [
            "interaction" if len(base_predictors) > 1 else "univariate"
            for base_predictors in self.get_base_predictors_in_each_unique_term_affiliation()
        ]
        return APLRExplanation(
            "local",
            internal_obj,
            feature_names=term_names,
            feature_types=term_types,
            name=gen_name_from_class(self) if name is None else name,
            selector=selector,
        )


def calculate_densities(X: FloatMatrix) -> tuple[list[list[int]], list[list[float]]]:
    bin_counts: list[list[int]] = []
    bin_edges: list[list[float]] = []
    for col in convert_to_numpy_matrix(X).T:
        counts_this_col, bin_edges_this_col = np.histogram(col, bins="doane")
        bin_counts.append(counts_this_col)
        bin_edges.append(bin_edges_this_col)
    return bin_counts, bin_edges


def convert_to_numpy_matrix(X: FloatMatrix) -> np.ndarray:
    try:
        from scipy import sparse as _sparse

        if _sparse.issparse(X):
            raise TypeError(
                "Sparse input is not supported. Please convert X to a dense array."
            )
    except ImportError:
        pass

    if isinstance(X, np.ndarray):
        if X.dtype == object:
            try:
                return X.astype(np.float64)
            except (ValueError, TypeError):
                msg = "argument must be a string or a real number"
                raise TypeError(msg)
        if not np.issubdtype(X.dtype, np.number):
            msg = f"If X is a numpy array, it must contain only numeric values, but got dtype '{X.dtype}'."
            raise TypeError(msg)
        return X.astype(np.float64, copy=False)
    if isinstance(X, _DataFrameType) and not X.empty:
        try:
            return X.to_numpy(np.float64)
        except (ValueError, TypeError) as e:
            msg = "If X is a pandas DataFrame, all columns must be numeric."
            raise TypeError(msg) from e
    if isinstance(X, list):
        try:
            return np.array(X, dtype=np.float64)
        except (ValueError, TypeError) as e:
            msg = "If X is a list, it must be a list of lists containing only numeric values."
            raise TypeError(msg) from e
    msg = f"X must be a numpy array, pandas DataFrame, or list of numeric lists, but got {type(X).__name__}."
    raise TypeError(msg)


def calculate_unique_values(X: FloatMatrix) -> list[int]:
    return [len(np.unique(col)) for col in convert_to_numpy_matrix(X).T]


def define_feature_names(X: FloatMatrix, X_names: list[str] | None = None) -> list[str]:
    if X_names is None or len(X_names) == 0:
        return [f"X{i + 1}" for i in range(convert_to_numpy_matrix(X).shape[1])]
    return list(X_names)


def create_values(
    X: np.ndarray,
    explanations: np.ndarray,
    term_names: list[str],
    feature_names: list[str],
) -> np.ndarray:
    X_values = np.full(shape=explanations.shape, fill_value=np.nan)
    for term_index, term_name in enumerate(term_names):
        if term_name in feature_names:
            feature_index = feature_names.index(term_name)
            X_values[:, term_index] = convert_to_numpy_matrix(X)[:, feature_index]
    return X_values


try:
    from aplr import APLRClassifier as APLRClassifierNative
except ImportError:

    class APLRClassifierNative:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "APLRClassifier requires the 'aplr' package. Please install it using 'pip install aplr'"
            )


class APLRClassifier(
    SKClassifierMixin,
    LocalExplainer,
    GlobalExplainer,
    SKBaseEstimator,
    APLRClassifierNative,
):
    """APLR Classifier."""

    def __init__(self, **kwargs):
        """Initializes class.

        Args:
            **kwargs: Kwargs passed to APLRClassifier at initialization time.
        """

        # TODO: add feature_names and feature_types to conform to glassbox API
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return APLRClassifierNative.get_params(self)

    def set_params(self, **params):
        APLRClassifierNative.set_params(self, **params)
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.target_tags.required = True
        return tags

    def predict(self, X):
        """Predicts class labels."""
        if not hasattr(self, "n_features_in_"):
            raise SKNotFittedError(
                "This model has not been fitted yet. Call 'fit' first."
            )
        str_preds = super().predict(X)
        return np.array(
            [self._str_to_label_[s] for s in str_preds], dtype=self.classes_.dtype
        )

    def predict_proba(self, X):
        """Predicts class probabilities."""
        if not hasattr(self, "n_features_in_"):
            raise SKNotFittedError(
                "This model has not been fitted yet. Call 'fit' first."
            )
        return self.predict_class_probabilities(X)

    def fit(self, X, y, **kwargs):
        """Fits model."""
        X_names = kwargs.get("X_names")

        self.bin_counts_, self.bin_edges_ = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X, X_names=X_names)
        self.n_features_in_ = len(self.feature_names_in_)

        y_arr = np.asarray(y)
        y_str = [str(val) for val in y_arr]

        super().fit(
            X,
            y_str,
            **kwargs,
        )

        categories = self.get_categories()
        unique_orig = {}
        for val, s in zip(y_arr, y_str):
            if s not in unique_orig:
                unique_orig[s] = val
        self.classes_ = np.array([unique_orig[c] for c in categories])
        self._str_to_label_ = {c: unique_orig[c] for c in categories}
        return self

    def explain_global(self, name: str | None = None):
        """Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """

        overall_dict = {
            "names": self.get_unique_term_affiliations(),
            "scores": self.get_feature_importance(),
        }

        data_dicts = []
        feature_list = []
        density_list = []
        unique_values = []
        categories = self.get_categories()
        for category in categories:
            model = self.get_logit_model(category)
            predictors_in_each_affiliation = (
                model.get_base_predictors_in_each_unique_term_affiliation()
            )
            for affiliation_index, affiliation in enumerate(
                model.get_unique_term_affiliations()
            ):
                shape = model.get_unique_term_affiliation_shape(affiliation)
                predictor_indexes_used = predictors_in_each_affiliation[
                    affiliation_index
                ]
                is_main_effect: bool = len(predictor_indexes_used) == 1
                is_two_way_interaction: bool = len(predictor_indexes_used) == 2
                if is_main_effect:
                    density_dict = {
                        "names": self.bin_edges_[predictor_indexes_used[0]],
                        "scores": self.bin_counts_[predictor_indexes_used[0]],
                    }
                    feature_dict = {
                        "type": "univariate",
                        "feature_name": self.feature_names_in_[
                            predictor_indexes_used[0]
                        ],
                        "term_name": f"class {category}: {affiliation}",
                        "names": shape[:, 0],
                        "scores": shape[:, 1],
                    }
                    data_dict = {
                        "type": "univariate",
                        "feature_name": self.feature_names_in_[
                            predictor_indexes_used[0]
                        ],
                        "names": shape[:, 0],
                        "scores": shape[:, 1],
                        "density": density_dict,
                    }
                    feature_list.append(feature_dict)
                    density_list.append(density_dict)
                    data_dicts.append(data_dict)
                    unique_values.append(
                        self.unique_values_in_[predictor_indexes_used[0]]
                    )
                elif is_two_way_interaction:
                    feature_dict = {
                        "type": "interaction",
                        "feature_names": [
                            self.feature_names_in_[idx]
                            for idx in predictor_indexes_used
                        ],
                        "term_name": f"class {category}: {affiliation}",
                        "left_names": shape[:, 0],
                        "right_names": shape[:, 1],
                        "scores": shape[:, 2],
                    }
                    data_dict = {
                        "type": "interaction",
                        "feature_names": [
                            self.feature_names_in_[idx]
                            for idx in predictor_indexes_used
                        ],
                        "left_names": shape[:, 0],
                        "right_names": shape[:, 1],
                        "scores": shape[:, 2],
                    }
                    feature_list.append(feature_dict)
                    density_list.append({})
                    data_dicts.append(data_dict)
                    unique_values.append(np.nan)
                else:  # pragma: no cover
                    warn(
                        f"Dropping term {affiliation} from explanation "
                        "since we can't graph more than 2 dimensions."
                    )
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "aplr_global",
                    "value": {"feature_list": feature_list},
                },
                {
                    "explanation_type": "density",
                    "value": {"density": density_list},
                },
            ],
        }
        term_names = [feature_dict["term_name"] for feature_dict in feature_list]
        term_types = [feature_dict["type"] for feature_dict in feature_list]
        selector = gen_global_selector(
            len(term_names),
            term_names,
            term_types,
            unique_values,
            None,
        )
        return APLRExplanation(
            "global",
            internal_obj,
            feature_names=term_names,
            feature_types=term_types,
            name=name,
            selector=selector,
        )

    def explain_local(
        self, X: FloatMatrix, y: FloatVector | None = None, name: str | None = None
    ):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        pred = APLRClassifierNative.predict(self, X)
        pred_proba = self.predict_class_probabilities(X)
        pred_max_prob = np.max(pred_proba, axis=1)
        term_names = self.get_unique_term_affiliations()
        explanations = self.calculate_local_feature_contribution(X)
        classes = self.get_categories()

        data_dicts = []
        perf_list = []

        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                msg = f"y must be 1 dimensional, but got {y.ndim} dimensions with shape {y.shape}"
                raise ValueError(msg)
            if not all(isinstance(val, str) for val in y):
                y = [str(val) for val in y]
        X_values = create_values(X, explanations, term_names, self.feature_names_in_)
        perf_list = []
        for i in range(len(pred)):
            di = {}
            di["is_classification"] = True
            di["actual"] = np.nan if y is None else classes.index(y[i])
            di["predicted"] = classes.index(pred[i])
            di["actual_score"] = (
                np.nan if y is None else pred_proba[i, classes.index(y[i])]
            )
            di["predicted_score"] = pred_max_prob[i]
            perf_list.append(di)

        for data, sample_scores, perf in zip(X_values, explanations, perf_list):
            model = self.get_logit_model(classes[perf["predicted"]])
            values = ["" if np.isnan(val) else val for val in data.tolist()]
            data_dict = {
                "type": "univariate",
                "names": term_names,
                "scores": list(sample_scores),
                "values": values,
                "extra": {
                    "names": ["Intercept"],
                    "scores": [model.get_intercept()],
                    "values": [1],
                },
                "perf": perf,
                "meta": {"label_names": classes},
            }
            data_dicts.append(data_dict)

        selector = gen_local_selector(data_dicts, is_classification=True)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
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

        term_types = [
            "interaction" if len(base_predictors) > 1 else "univariate"
            for base_predictors in self.get_base_predictors_in_each_unique_term_affiliation()
        ]
        return APLRExplanation(
            "local",
            internal_obj,
            feature_names=term_names,
            feature_types=term_types,
            name=gen_name_from_class(self) if name is None else name,
            selector=selector,
        )


class APLRExplanation(FeatureValueExplanation):
    """Visualizes specifically for APLR."""

    explanation_type = None

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
            plot_horizontal_bar,
            plot_line,
            plot_pairwise_heatmap,
            sort_take,
        )

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Overall global explanation
        if self.explanation_type == "global" and key is None:
            data_dict = sort_take(
                data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
            )
            title = "Global Term/Feature Importances"

            figure = plot_horizontal_bar(
                data_dict,
                title=title,
                start_zero=True,
                xtitle="Standard deviation of contribution to linear predictor",
            )

            figure._interpret_help_text = (
                "The term importances are the standard deviations "
                "of the contribution that each term makes to the linear predictor "
                "in the training dataset. For classification this is averaged "
                "over all of the underlying APLRRegressor logit models. "
                "The 15 most important terms are shown."
            )
            figure._interpret_help_link = ""

            return figure

        # Per term global explanation
        if self.explanation_type == "global":
            title = f"Term: {self.feature_names[key]} ({self.feature_types[key]})"

            if self.feature_types[key] == "univariate":
                xtitle = self.feature_names[key]
                figure = plot_line(data_dict, title=title, xtitle=xtitle)

            elif self.feature_types[key] == "interaction":
                xtitle = data_dict["feature_names"][0]
                ytitle = data_dict["feature_names"][1]
                figure = plot_pairwise_heatmap(
                    data_dict,
                    title=title,
                    xtitle=xtitle,
                    ytitle=ytitle,
                    transform_vals=False,
                )
            else:  # pragma: no cover
                msg = f"Unsupported configuration: explanation_type='{self.explanation_type}', feature_type='{self.feature_types[key]}'"
                raise ValueError(msg)

            figure._interpret_help_text = (
                "The contribution (score) of the term "
                f"{self.feature_names[key]} to the linear predictor."
            )

            return figure

        # Local explanation graph
        if self.explanation_type == "local":
            figure = super().visualize(key)
            figure.update_layout(
                title="Local Explanation (" + figure.layout.title.text + ")",
                xaxis_title="Contribution to linear predictor",
            )
            figure._interpret_help_text = (
                "A local explanation shows the breakdown of how much "
                "each term contributed to the linear predictor for a single sample. "
                "For regression, the predict method by default caps predictions to min/max "
                "of predictions/response in the training data. For observations that "
                "where capped, the predictions will not be consistent with the "
                "contributions to the linear predictor. "
                "For classification, this pertains to the logit APLRRegressor model for "
                "the category that corresponds to the predicted class of the sample. "
                "The 15 most important terms are shown."
            )

            return figure
        return None
