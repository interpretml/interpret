# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
import numpy as np
from typing import List, Tuple
from warnings import warn
import aplr
from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils._explanation import (
    gen_name_from_class,
    gen_global_selector,
    gen_local_selector,
    gen_perf_dicts,
)
from ..utils._clean_simple import clean_dimensions


FloatVector = List[float]
FloatMatrix = List[List[float]]
IntVector = List[int]
IntMatrix = List[List[int]]
StrVector = List[str]


class APLRRegressor(aplr.APLRRegressor, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def fit(
        self,
        X: FloatMatrix,
        y: FloatVector,
        sample_weight: FloatVector = [],
        X_names: StrVector = [],
        cv_observations: IntMatrix = [],
        prioritized_predictors_indexes: IntVector = [],
        monotonic_constraints: IntVector = [],
        group: FloatVector = [],
        interaction_constraints: List[List[int]] = [],
        other_data: FloatMatrix = [],
        predictor_learning_rates: FloatVector = [],
        predictor_penalties_for_non_linearity: FloatVector = [],
        predictor_penalties_for_interactions: FloatVector = [],
    ):
        self.bin_counts, self.bin_edges = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X_names, X)
        super().fit(
            X,
            y,
            sample_weight,
            X_names,
            cv_observations,
            prioritized_predictors_indexes,
            monotonic_constraints,
            group,
            interaction_constraints,
            other_data,
            predictor_learning_rates,
            predictor_penalties_for_non_linearity,
            predictor_penalties_for_interactions,
        )

    def explain_global(self, name: str = None):
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
                    "names": self.bin_edges[predictor_indexes_used[0]],
                    "scores": self.bin_counts[predictor_indexes_used[0]],
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
                {"explanation_type": "density", "value": {"density": density_list}},
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

    def explain_local(self, X: FloatMatrix, y: FloatVector = None, name: str = None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        pred = self.predict(X, True)
        term_names = self.get_unique_term_affiliations()
        explanations = self.calculate_local_feature_contribution(X)

        data_dicts = []
        perf_list = []

        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
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


def calculate_densities(X: FloatMatrix) -> Tuple[List[List[int]], List[List[float]]]:
    bin_counts: List[List[int]] = []
    bin_edges: List[List[float]] = []
    for col in X.T:
        counts_this_col, bin_edges_this_col = np.histogram(col, bins="doane")
        bin_counts.append(counts_this_col)
        bin_edges.append(bin_edges_this_col)
    return bin_counts, bin_edges


def calculate_unique_values(X: FloatMatrix) -> List[int]:
    unique_values_counts = [len(np.unique(col)) for col in X.T]
    return unique_values_counts


def define_feature_names(X_names: StrVector, X: FloatMatrix) -> StrVector:
    if len(X_names) == 0:
        names = [f"X{i+1}" for i in range(X.shape[1])]
        return names
    else:
        return X_names


def create_values(
    X: np.ndarray,
    explanations: np.ndarray,
    term_names: StrVector,
    feature_names: StrVector,
) -> np.ndarray:
    X_values = np.full(shape=explanations.shape, fill_value=np.nan)
    for term_index, term_name in enumerate(term_names):
        if term_name in feature_names:
            feature_index = feature_names.index(term_name)
            X_values[:, term_index] = X[:, feature_index]
    return X_values


class APLRClassifier(aplr.APLRClassifier, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def fit(
        self,
        X: FloatMatrix,
        y: StrVector,
        sample_weight: FloatVector = [],
        X_names: StrVector = [],
        cv_observations: IntMatrix = [],
        prioritized_predictors_indexes: IntVector = [],
        monotonic_constraints: IntVector = [],
        interaction_constraints: List[List[int]] = [],
        predictor_learning_rates: FloatVector = [],
        predictor_penalties_for_non_linearity: FloatVector = [],
        predictor_penalties_for_interactions: FloatVector = [],
    ):
        self.bin_counts, self.bin_edges = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X_names, X)

        if not all(isinstance(val, str) for val in y):
            y = [str(val) for val in y]

        super().fit(
            X,
            y,
            sample_weight,
            X_names,
            cv_observations,
            prioritized_predictors_indexes,
            monotonic_constraints,
            interaction_constraints,
            predictor_learning_rates,
            predictor_penalties_for_non_linearity,
            predictor_penalties_for_interactions,
        )

    def explain_global(self, name: str = None):
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
                shape = model.get_unique_term_affiliation_shape(
                    affiliation, max_rows_before_sampling=100000
                )
                predictor_indexes_used = predictors_in_each_affiliation[
                    affiliation_index
                ]
                is_main_effect: bool = len(predictor_indexes_used) == 1
                is_two_way_interaction: bool = len(predictor_indexes_used) == 2
                if is_main_effect:
                    density_dict = {
                        "names": self.bin_edges[predictor_indexes_used[0]],
                        "scores": self.bin_counts[predictor_indexes_used[0]],
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
                {"explanation_type": "density", "value": {"density": density_list}},
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

    def explain_local(self, X: FloatMatrix, y: FloatVector = None, name: str = None):
        """Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        ...


class APLRExplanation(FeatureValueExplanation):
    """Visualizes specifically for APLR."""

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
        super(APLRExplanation, self).__init__(
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
            plot_line,
            plot_pairwise_heatmap,
            plot_continuous_bar,
            plot_horizontal_bar,
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
            title = "Term: {0} ({1})".format(
                self.feature_names[key], self.feature_types[key]
            )

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
                raise Exception(
                    "Not supported configuration: {0}, {1}".format(
                        self.explanation_type, self.feature_types[key]
                    )
                )

            figure._interpret_help_text = (
                "The contribution (score) of the term "
                "{0} to the linear predictor.".format(self.feature_names[key])
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
                "For classification, this pertains to the logit APLRRegressor model for "
                "the category that corresponds to the predicted class of the sample. "
                "By default the predict method in APLR caps predictions to min/max "
                "of predictions/response in the training data. For observations that "
                "where capped, the predictions will not be consistent with the "
                "contributions to the linear predictor. "
                "The 15 most important terms are shown."
            )

            return figure
