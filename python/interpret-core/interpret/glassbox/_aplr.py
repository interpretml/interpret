# Copyright (c) 2024 The InterpretML Contributors
# Distributed under the MIT software license
from typing import Dict, List, Optional, Tuple
from warnings import warn
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
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


@dataclass
class APLRInputTags:
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = False
    categorical: bool = False
    string: bool = False
    dict: bool = False
    positive_only: bool = False
    allow_nan: bool = False
    pairwise: bool = False


@dataclass
class APLRTargetTags:
    required: bool = True
    one_d_labels: bool = True
    two_d_labels: bool = False
    positive_only: bool = False
    multi_output: bool = False
    single_output: bool = True


@dataclass
class APLRClassifierTags:
    poor_score: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class APLRRegressorTags:
    poor_score: bool = False


@dataclass
class APLRTags:
    estimator_type: Optional[str] = None
    target_tags: APLRTargetTags = field(default_factory=APLRTargetTags)
    transformer_tags: None = None
    classifier_tags: Optional[APLRClassifierTags] = None
    regressor_tags: Optional[APLRRegressorTags] = None
    array_api_support: bool = False
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: APLRInputTags = field(default_factory=APLRInputTags)


class APLRRegressor(RegressorMixin, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(self, **kwargs):
        """Initializes class.

        Args:
            **kwargs: Kwargs pass to APLRRegressor at initialization time.
        """

        # TODO: add feature_names and feature_types to conform to glassbox API

        from aplr import APLRRegressor

        self.model_ = APLRRegressor(**kwargs)

    def fit(
        self,
        X: FloatMatrix,
        y: FloatVector,
        sample_weight: FloatVector = np.empty(0),
        X_names: Optional[List[str]] = None,
        cv_observations: IntMatrix = np.empty([0, 0]),
        prioritized_predictors_indexes: Optional[List[int]] = None,
        monotonic_constraints: Optional[List[int]] = None,
        group: FloatVector = np.empty(0),
        interaction_constraints: Optional[List[List[int]]] = None,
        other_data: FloatMatrix = np.empty([0, 0]),
        predictor_learning_rates: Optional[List[float]] = None,
        predictor_penalties_for_non_linearity: Optional[List[float]] = None,
        predictor_penalties_for_interactions: Optional[List[float]] = None,
    ):
        if predictor_penalties_for_interactions is None:
            predictor_penalties_for_interactions = []
        if predictor_penalties_for_non_linearity is None:
            predictor_penalties_for_non_linearity = []
        if predictor_learning_rates is None:
            predictor_learning_rates = []
        if interaction_constraints is None:
            interaction_constraints = []
        if monotonic_constraints is None:
            monotonic_constraints = []
        if prioritized_predictors_indexes is None:
            prioritized_predictors_indexes = []
        if X_names is None:
            X_names = []
        self.bin_counts, self.bin_edges = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X_names, X)

        self.model_.fit(
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

    def predict(
        self, X: FloatMatrix, cap_predictions_to_minmax_in_training: bool = True
    ) -> FloatVector:
        return self.model_.predict(X, cap_predictions_to_minmax_in_training)

    def set_term_names(self, X_names: List[str]):
        self.model_.set_term_names(X_names)

    def calculate_feature_importance(
        self, X: FloatMatrix, sample_weight: FloatVector = np.empty(0)
    ) -> FloatVector:
        return self.model_.calculate_feature_importance(X, sample_weight)

    def calculate_term_importance(
        self, X: FloatMatrix, sample_weight: FloatVector = np.empty(0)
    ) -> FloatVector:
        return self.model_.calculate_term_importance(X, sample_weight)

    def calculate_local_feature_contribution(self, X: FloatMatrix) -> FloatMatrix:
        return self.model_.calculate_local_feature_contribution(X)

    def calculate_local_term_contribution(self, X: FloatMatrix) -> FloatMatrix:
        return self.model_.calculate_local_term_contribution(X)

    def calculate_local_contribution_from_selected_terms(
        self, X: FloatMatrix, predictor_indexes: List[int]
    ) -> FloatVector:
        return self.model_.calculate_local_contribution_from_selected_terms(
            X, predictor_indexes
        )

    def calculate_terms(self, X: FloatMatrix) -> FloatMatrix:
        return self.model_.calculate_terms(X)

    def get_term_names(self) -> List[str]:
        return self.model_.get_term_names()

    def get_term_affiliations(self) -> List[str]:
        return self.model_.get_term_affiliations()

    def get_unique_term_affiliations(self) -> List[str]:
        return self.model_.get_unique_term_affiliations()

    def get_base_predictors_in_each_unique_term_affiliation(self) -> List[List[int]]:
        return self.model_.get_base_predictors_in_each_unique_term_affiliation()

    def get_term_coefficients(self) -> FloatVector:
        return self.model_.get_term_coefficients()

    def get_validation_error_steps(self) -> FloatMatrix:
        return self.model_.get_validation_error_steps()

    def get_feature_importance(self) -> FloatVector:
        return self.model_.get_feature_importance()

    def get_term_importance(self) -> FloatVector:
        return self.model_.get_term_importance()

    def get_term_main_predictor_indexes(self) -> IntVector:
        return self.model_.get_term_main_predictor_indexes()

    def get_term_interaction_levels(self) -> IntVector:
        return self.model_.get_term_interaction_levels()

    def get_intercept(self) -> float:
        return self.model_.get_intercept()

    def get_optimal_m(self) -> int:
        return self.model_.get_optimal_m()

    def get_validation_tuning_metric(self) -> str:
        return self.model_.get_validation_tuning_metric()

    def get_main_effect_shape(self, predictor_index: int) -> Dict[float, float]:
        return self.model_.get_main_effect_shape(predictor_index)

    def get_unique_term_affiliation_shape(
        self, unique_term_affiliation: str, max_rows_before_sampling: int = 100000
    ) -> FloatMatrix:
        return self.model_.get_unique_term_affiliation_shape(
            unique_term_affiliation, max_rows_before_sampling
        )

    def get_cv_error(self) -> float:
        return self.model_.get_cv_error()

    def get_params(self, deep=True):
        return {
            "m": self.model_.m,
            "v": self.model_.v,
            "random_state": self.model_.random_state,
            "loss_function": self.model_.loss_function,
            "link_function": self.model_.link_function,
            "n_jobs": self.model_.n_jobs,
            "cv_folds": self.model_.cv_folds,
            "bins": self.model_.bins,
            "max_interaction_level": self.model_.max_interaction_level,
            "max_interactions": self.model_.max_interactions,
            "verbosity": self.model_.verbosity,
            "min_observations_in_split": self.model_.min_observations_in_split,
            "ineligible_boosting_steps_added": self.model_.ineligible_boosting_steps_added,
            "max_eligible_terms": self.model_.max_eligible_terms,
            "dispersion_parameter": self.model_.dispersion_parameter,
            "validation_tuning_metric": self.model_.validation_tuning_metric,
            "quantile": self.model_.quantile,
            "calculate_custom_validation_error_function": self.model_.calculate_custom_validation_error_function,
            "calculate_custom_loss_function": self.model_.calculate_custom_loss_function,
            "calculate_custom_negative_gradient_function": self.model_.calculate_custom_negative_gradient_function,
            "calculate_custom_transform_linear_predictor_to_predictions_function": self.model_.calculate_custom_transform_linear_predictor_to_predictions_function,
            "calculate_custom_differentiate_predictions_wrt_linear_predictor_function": self.model_.calculate_custom_differentiate_predictions_wrt_linear_predictor_function,
            "boosting_steps_before_interactions_are_allowed": self.model_.boosting_steps_before_interactions_are_allowed,
            "monotonic_constraints_ignore_interactions": self.model_.monotonic_constraints_ignore_interactions,
            "group_mse_by_prediction_bins": self.model_.group_mse_by_prediction_bins,
            "group_mse_cycle_min_obs_in_bin": self.model_.group_mse_cycle_min_obs_in_bin,
            "early_stopping_rounds": self.model_.early_stopping_rounds,
            "num_first_steps_with_linear_effects_only": self.model_.num_first_steps_with_linear_effects_only,
            "penalty_for_non_linearity": self.model_.penalty_for_non_linearity,
            "penalty_for_interactions": self.model_.penalty_for_interactions,
            "max_terms": self.model_.max_terms,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.model_, parameter, value)
        self.model_.__set_params_cpp()
        return self

    def explain_global(self, name: Optional[str] = None):
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

    def explain_local(
        self, X: FloatMatrix, y: FloatVector = None, name: Optional[str] = None
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
                msg = "y must be 1 dimensional"
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

    def __sklearn_tags__(self):
        tags = APLRTags()
        tags.estimator_type = "regressor"
        tags.regressor_tags = APLRRegressorTags()
        return tags


def calculate_densities(X: FloatMatrix) -> Tuple[List[List[int]], List[List[float]]]:
    bin_counts: List[List[int]] = []
    bin_edges: List[List[float]] = []
    for col in convert_to_numpy_matrix(X).T:
        counts_this_col, bin_edges_this_col = np.histogram(col, bins="doane")
        bin_counts.append(counts_this_col)
        bin_edges.append(bin_edges_this_col)
    return bin_counts, bin_edges


def convert_to_numpy_matrix(X: FloatMatrix) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, pd.DataFrame):
        return X.values
    if isinstance(X, list):
        return np.array(X)
    msg = (
        "X must either be a numpy matrix, a pandas dataframe or a list of float lists."
    )
    raise TypeError(msg)


def calculate_unique_values(X: FloatMatrix) -> List[int]:
    return [len(np.unique(col)) for col in convert_to_numpy_matrix(X).T]


def define_feature_names(X_names: List[str], X: FloatMatrix) -> List[str]:
    if len(X_names) == 0:
        return [f"X{i+1}" for i in range(convert_to_numpy_matrix(X).shape[1])]
    return list(X_names)


def create_values(
    X: np.ndarray,
    explanations: np.ndarray,
    term_names: List[str],
    feature_names: List[str],
) -> np.ndarray:
    X_values = np.full(shape=explanations.shape, fill_value=np.nan)
    for term_index, term_name in enumerate(term_names):
        if term_name in feature_names:
            feature_index = feature_names.index(term_name)
            X_values[:, term_index] = convert_to_numpy_matrix(X)[:, feature_index]
    return X_values


class APLRClassifier(ClassifierMixin, ExplainerMixin):
    available_explanations = ["local", "global"]
    explainer_type = "model"

    def __init__(self, **kwargs):
        """Initializes class.

        Args:
            **kwargs: Kwargs pass to APLRClassifier at initialization time.
        """

        # TODO: add feature_names and feature_types to conform to glassbox API

        from aplr import APLRClassifier

        self.model_ = APLRClassifier(**kwargs)

    def fit(
        self,
        X: FloatMatrix,
        y: List[str],
        sample_weight: FloatVector = np.empty(0),
        X_names: Optional[List[str]] = None,
        cv_observations: IntMatrix = np.empty([0, 0]),
        prioritized_predictors_indexes: Optional[List[int]] = None,
        monotonic_constraints: Optional[List[int]] = None,
        interaction_constraints: Optional[List[List[int]]] = None,
        predictor_learning_rates: Optional[List[float]] = None,
        predictor_penalties_for_non_linearity: Optional[List[float]] = None,
        predictor_penalties_for_interactions: Optional[List[float]] = None,
    ):
        if predictor_penalties_for_interactions is None:
            predictor_penalties_for_interactions = []
        if predictor_penalties_for_non_linearity is None:
            predictor_penalties_for_non_linearity = []
        if predictor_learning_rates is None:
            predictor_learning_rates = []
        if interaction_constraints is None:
            interaction_constraints = []
        if monotonic_constraints is None:
            monotonic_constraints = []
        if prioritized_predictors_indexes is None:
            prioritized_predictors_indexes = []
        if X_names is None:
            X_names = []
        self.bin_counts, self.bin_edges = calculate_densities(X)
        self.unique_values_in_ = calculate_unique_values(X)
        self.feature_names_in_ = define_feature_names(X_names, X)

        if not all(isinstance(val, str) for val in y):
            y = [str(val) for val in y]

        if isinstance(y, pd.Series):
            y = y.values

        self.model_.fit(
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
        self.classes_ = self.model_.classes_

    def predict_class_probabilities(
        self, X: FloatMatrix, cap_predictions_to_minmax_in_training: bool = False
    ) -> FloatMatrix:
        return self.model_.predict_class_probabilities(
            X, cap_predictions_to_minmax_in_training
        )

    def predict(
        self, X: FloatMatrix, cap_predictions_to_minmax_in_training: bool = False
    ) -> List[str]:
        return self.model_.predict(X, cap_predictions_to_minmax_in_training)

    def calculate_local_feature_contribution(self, X: FloatMatrix) -> FloatMatrix:
        return self.model_.calculate_local_feature_contribution(X)

    def get_categories(self) -> List[str]:
        return self.model_.get_categories()

    def get_logit_model(self, category: str) -> APLRRegressor:
        return self.model_.get_logit_model(category)

    def get_validation_error_steps(self) -> FloatMatrix:
        return self.model_.get_validation_error_steps()

    def get_cv_error(self) -> float:
        return self.model_.get_cv_error()

    def get_feature_importance(self) -> FloatVector:
        return self.model_.get_feature_importance()

    def get_unique_term_affiliations(self) -> List[str]:
        return self.model_.get_unique_term_affiliations()

    def get_base_predictors_in_each_unique_term_affiliation(self) -> List[List[int]]:
        return self.model_.get_base_predictors_in_each_unique_term_affiliation()

    def predict_proba(self, X: FloatMatrix) -> FloatMatrix:
        return self.model_.predict_class_probabilities(X)

    def get_params(self, deep=True):
        return {
            "m": self.model_.m,
            "v": self.model_.v,
            "random_state": self.model_.random_state,
            "n_jobs": self.model_.n_jobs,
            "cv_folds": self.model_.cv_folds,
            "bins": self.model_.bins,
            "verbosity": self.model_.verbosity,
            "max_interaction_level": self.model_.max_interaction_level,
            "max_interactions": self.model_.max_interactions,
            "min_observations_in_split": self.model_.min_observations_in_split,
            "ineligible_boosting_steps_added": self.model_.ineligible_boosting_steps_added,
            "max_eligible_terms": self.model_.max_eligible_terms,
            "boosting_steps_before_interactions_are_allowed": self.model_.boosting_steps_before_interactions_are_allowed,
            "monotonic_constraints_ignore_interactions": self.model_.monotonic_constraints_ignore_interactions,
            "early_stopping_rounds": self.model_.early_stopping_rounds,
            "num_first_steps_with_linear_effects_only": self.model_.num_first_steps_with_linear_effects_only,
            "penalty_for_non_linearity": self.model_.penalty_for_non_linearity,
            "penalty_for_interactions": self.model_.penalty_for_interactions,
            "max_terms": self.model_.max_terms,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.model_, parameter, value)
        self.model_.__set_params_cpp()
        return self

    def explain_global(self, name: Optional[str] = None):
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

    def explain_local(
        self, X: FloatMatrix, y: FloatVector = None, name: Optional[str] = None
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
                msg = "y must be 1 dimensional"
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

    def __sklearn_tags__(self):
        tags = APLRTags()
        tags.estimator_type = "classifier"
        tags.classifier_tags = APLRClassifierTags()
        return tags


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
                msg = f"Not supported configuration: {self.explanation_type}, {self.feature_types[key]}"
                raise Exception(msg)

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
