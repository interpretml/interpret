# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from ...utils import perf_dict
from .utils import EBMUtils
from .internal import NativeEBM
from ...utils import unify_data, autogen_schema
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...utils import JobLibProvider
from ...utils import gen_name_from_class, gen_global_selector, gen_local_selector
from ...visual.plot import plot_continuous_bar, plot_horizontal_bar, sort_take

import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import Counter

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split
from contextlib import closing
from itertools import combinations

import logging

log = logging.getLogger(__name__)


class EBMExplanation(FeatureValueExplanation):
    """ Visualizes specifically for EBM.
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

        super(EBMExplanation, self).__init__(
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
                data_dict,
                title="Overall Importance:<br>Mean Absolute Score",
                start_zero=True,
            )
            return figure

        if (
            self.explanation_type == "global"
            and self.feature_types[key] == "continuous"
        ):
            title = self.feature_names[key]
            figure = plot_continuous_bar(data_dict, title=title)
            return figure

        return super().visualize(key)


# TODO: More documentation in binning process to be explicit.
# TODO: Consider stripping this down to the bare minimum.
class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self,
        schema=None,
        cont_n_bins=255,
        missing_constant=0,
        unknown_constant=0,
        feature_names=None,
    ):
        """ Initializes EBM preprocessor.

        Args:
            schema: A dictionary that encapsulates column information,
                    such as type and domain.
            cont_n_bins: Max number of bins to process numeric features.
            missing_constant: Missing encoded as this constant.
            unknown_constant: Unknown encoded as this constant.
            feature_names: Feature names as list.
        """
        self.schema = schema
        self.cont_n_bins = cont_n_bins
        self.missing_constant = missing_constant
        self.unknown_constant = unknown_constant
        self.feature_names = feature_names

    def fit(self, X):
        """ Fits transformer to provided instances.

        Args:
            X: Numpy array for training instances.

        Returns:
            Itself.
        """
        # self.col_bin_counts_ = {}
        self.col_bin_edges_ = {}

        self.hist_counts_ = {}
        self.hist_edges_ = {}

        self.col_mapping_ = {}
        self.col_mapping_counts_ = {}

        self.col_n_bins_ = {}

        self.col_names_ = []
        self.col_types_ = []
        self.has_fitted_ = False

        # TODO: Remove this.
        if self.schema is not None:
            self.schema_ = self.schema
        else:
            self.schema_ = autogen_schema(X, feature_names=self.feature_names)

        self.schema_ = (
            self.schema
            if self.schema is not None
            else autogen_schema(X, feature_names=self.feature_names)
        )
        schema = self.schema_

        for col_idx in range(X.shape[1]):
            col_name = list(schema.keys())[col_idx]
            self.col_names_.append(col_name)

            col_info = schema[col_name]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            self.col_types_.append(col_info["type"])
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)

                uniq_vals = set(col_data[~np.isnan(col_data)])
                if len(uniq_vals) < self.cont_n_bins:
                    bins = list(sorted(uniq_vals))
                else:
                    bins = self.cont_n_bins

                _, bin_edges = np.histogram(col_data, bins=bins)

                hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                self.col_bin_edges_[col_idx] = bin_edges

                self.hist_edges_[col_idx] = hist_edges
                self.hist_counts_[col_idx] = hist_counts
                self.col_n_bins_[col_idx] = len(bin_edges)
            elif col_info["type"] == "ordinal":
                mapping = {val: indx for indx, val in enumerate(col_info["order"])}
                self.col_mapping_[col_idx] = mapping
                self.col_n_bins_[col_idx] = len(col_info["order"])
            elif col_info["type"] == "categorical":
                uniq_vals, counts = np.unique(col_data, return_counts=True)

                non_nan_index = ~np.isnan(counts)
                uniq_vals = uniq_vals[non_nan_index]
                counts = counts[non_nan_index]

                mapping = {val: indx for indx, val in enumerate(uniq_vals)}
                self.col_mapping_counts_[col_idx] = counts
                self.col_mapping_[col_idx] = mapping

                # TODO: Review NA as we don't support it yet.
                self.col_n_bins_[col_idx] = len(uniq_vals)

        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        schema = self.schema
        X_new = np.copy(X)
        for col_idx in range(X.shape[1]):
            col_info = schema[list(schema.keys())[col_idx]]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)
                bin_edges = self.col_bin_edges_[col_idx].copy()

                digitized = np.digitize(col_data, bin_edges, right=False)
                digitized[digitized == 0] = 1
                digitized -= 1

                # NOTE: NA handling done later.
                # digitized[np.isnan(col_data)] = self.missing_constant
                X_new[:, col_idx] = digitized
            elif col_info["type"] == "ordinal":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)
            elif col_info["type"] == "categorical":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)

        return X_new.astype(np.int64)

    def get_hist_counts(self, attribute_index):
        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.hist_counts_[attribute_index])
        elif col_type == "categorical":
            return list(self.col_mapping_counts_[attribute_index])
        else:
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_hist_edges(self, attribute_index):
        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.hist_edges_[attribute_index])
        elif col_type == "categorical":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        else:
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    # def get_bin_counts(self, attribute_index):
    #     col_type = self.col_types_[attribute_index]
    #     if col_type == 'continuous':
    #         return list(self.col_bin_counts_[attribute_index])
    #     elif col_type == 'categorical':
    #         return list(self.col_mapping_counts_[attribute_index])
    #     else:
    #         raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_bin_labels(self, attribute_index):
        """ Returns bin labels for a given attribute index.

        Args:
            attribute_index: An integer for attribute index.

        Returns:
            List of labels for bins.
        """

        col_type = self.col_types_[attribute_index]
        if col_type == "continuous":
            return list(self.col_bin_edges_[attribute_index])
        elif col_type == "ordinal":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        elif col_type == "categorical":
            map = self.col_mapping_[attribute_index]
            return list(map.keys())
        else:
            raise Exception("Unknown column type")


# TODO: Clean up
class BaseCoreEBM(BaseEstimator):
    """Internal use EBM."""

    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):

        # Arguments for data
        self.col_types = col_types
        self.col_n_bins = col_n_bins

        # Arguments for EBM beyond training a feature-step.
        self.interactions = interactions
        self.holdout_split = holdout_split
        self.data_n_episodes = data_n_episodes
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_run_length = early_stopping_run_length

        # Arguments for internal EBM.
        self.feature_step_n_inner_bags = feature_step_n_inner_bags
        self.learning_rate = learning_rate
        self.training_step_episodes = training_step_episodes
        self.max_tree_splits = max_tree_splits
        self.min_cases_for_splits = min_cases_for_splits

        # Arguments for overall
        self.random_state = random_state

    def fit(self, X, y):
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self.num_classes_ = len(self.classes_)
        else:
            self.num_classes_ = -1

        # Split data into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.holdout_split,
            random_state=self.random_state,
            stratify=y if is_classifier(self) else None,
        )
        # Define attributes
        self.attributes_ = EBMUtils.gen_attributes(self.col_types, self.col_n_bins)
        # Build EBM allocation code
        if is_classifier(self):
            model_type = "classification"
        else:
            model_type = "regression"

        self.intercept_ = 0
        self.attribute_sets_ = []
        self.attribute_set_models_ = []

        main_attr_indices = [[x] for x in range(len(self.attributes_))]
        main_attr_sets = EBMUtils.gen_attribute_sets(main_attr_indices)
        with closing(
            NativeEBM(
                self.attributes_,
                main_attr_sets,
                X_train,
                y_train,
                X_val,
                y_val,
                num_inner_bags=self.feature_step_n_inner_bags,
                num_classification_states=self.num_classes_,
                model_type=model_type,
                training_scores=None,
                validation_scores=None,
            )
        ) as native_ebm:
            # Train main effects
            self._fit_main(native_ebm, main_attr_sets)

            # Build interaction terms
            self.inter_indices_ = self._build_interactions(native_ebm)

        self.staged_fit_interactions(X, y, self.inter_indices_)

        return self

    def _build_interactions(self, native_ebm):
        if isinstance(self.interactions, int) and self.interactions != 0:
            log.debug("Estimating with FAST")
            interaction_scores = []
            interaction_indices = [
                x for x in combinations(range(len(self.col_types)), 2)
            ]
            for pair in interaction_indices:
                score = native_ebm.fast_interaction_score(pair)
                interaction_scores.append((pair, score))

            ranked_scores = list(
                sorted(interaction_scores, key=lambda x: x[1], reverse=True)
            )
            n_interactions = min(len(ranked_scores), self.interactions)

            inter_indices_ = [x[0] for x in ranked_scores[0:n_interactions]]
        elif isinstance(self.interactions, int) and self.interactions == 0:
            inter_indices_ = []
        elif isinstance(self.interactions, list):
            inter_indices_ = self.interactions
        else:
            raise RuntimeError("Argument 'interaction' has invalid value")

        return inter_indices_

    def _fit_main(self, native_ebm, main_attr_sets):
        log.debug("Train main effects")
        self.current_metric_ = self._cyclic_gradient_boost(
            native_ebm, main_attr_sets, "Main"
        )
        log.debug("Main Metric: {0}".format(self.current_metric_))
        for index, attr_set in enumerate(main_attr_sets):
            self.attribute_set_models_.append(native_ebm.get_best_model(index))
            self.attribute_sets_.append(attr_set)

        self.has_fitted_ = True

        return self

    def staged_fit_interactions(self, X, y, inter_indices=[]):
        check_is_fitted(self, "has_fitted_")

        log.debug("Train interactions")

        if len(inter_indices) == 0:
            return self

        # Split data into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.holdout_split,
            random_state=self.random_state,
            stratify=y if is_classifier(self) else None,
        )
        if is_classifier(self):
            model_type = "classification"
        else:
            model_type = "regression"

        # Discard initial interactions
        new_attribute_set_models = []
        new_attribute_sets = []
        for i, attribute_set in enumerate(self.attribute_sets_):
            if attribute_set["n_attributes"] != 1:
                continue
            new_attribute_set_models.append(self.attribute_set_models_[i])
            new_attribute_sets.append(self.attribute_sets_[i])
        self.attribute_set_models_ = new_attribute_set_models
        self.attribute_sets_ = new_attribute_sets

        # Fix main, train interactions
        training_scores = self.decision_function(X_train)
        validation_scores = self.decision_function(X_val)
        inter_attr_sets = EBMUtils.gen_attribute_sets(inter_indices)
        with closing(
            NativeEBM(
                self.attributes_,
                inter_attr_sets,
                X_train,
                y_train,
                X_val,
                y_val,
                num_inner_bags=self.feature_step_n_inner_bags,
                num_classification_states=self.num_classes_,
                model_type=model_type,
                training_scores=training_scores,
                validation_scores=validation_scores,
                random_state=self.random_state,
            )
        ) as native_ebm:
            log.debug("Train interactions")
            self.current_metric_ = self._cyclic_gradient_boost(
                native_ebm, inter_attr_sets, "Pair"
            )
            log.debug("Interaction Metric: {0}".format(self.current_metric_))

            for index, attr_set in enumerate(inter_attr_sets):
                self.attribute_set_models_.append(native_ebm.get_best_model(index))
                self.attribute_sets_.append(attr_set)

        return self

    def decision_function(self, X):
        check_is_fitted(self, "has_fitted_")

        return EBMUtils.decision_function(
            X, self.attribute_sets_, self.attribute_set_models_, 0
        )

    def _cyclic_gradient_boost(self, native_ebm, attribute_sets, name=None):

        no_change_run_length = 0
        curr_metric = np.inf
        min_metric = np.inf
        bp_metric = np.inf
        log.debug("Start boosting {0}".format(name))
        for data_episode_index in range(self.data_n_episodes):
            if data_episode_index % 10 == 0:
                log.debug("Sweep Index for {0}: {1}".format(name, data_episode_index))
                log.debug("Metric: {0}".format(curr_metric))

            if len(attribute_sets) == 0:
                log.debug("No sets to boost for {0}".format(name))

            log.debug("Start boosting {0}".format(name))
            for index, attribute_set in enumerate(attribute_sets):
                curr_metric = native_ebm.training_step(
                    index,
                    training_step_episodes=self.training_step_episodes,
                    learning_rate=self.learning_rate,
                    max_tree_splits=self.max_tree_splits,
                    min_cases_for_split=self.min_cases_for_splits,
                    training_weights=0,
                    validation_weights=0,
                )

            min_metric = min(curr_metric, min_metric)

            if no_change_run_length == 0:
                bp_metric = min_metric
            if curr_metric + self.early_stopping_tolerance < bp_metric:
                no_change_run_length = 0
            else:
                no_change_run_length += 1
            if no_change_run_length >= self.early_stopping_run_length:
                log.debug("Early break {0}: {1}".format(name, data_episode_index))
                break
        log.debug("End boosting {0}".format(name))

        return curr_metric


class CoreEBMClassifier(BaseCoreEBM, ClassifierMixin):
    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):
        super(CoreEBMClassifier, self).__init__(
            # Data
            col_types=col_types,
            col_n_bins=col_n_bins,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            random_state=random_state,
        )

    def predict_proba(self, X):
        check_is_fitted(self, "has_fitted_")
        return EBMUtils.classifier_predict_proba(X, self)

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        return EBMUtils.classifier_predict(X, self)


class CoreEBMRegressor(BaseCoreEBM, RegressorMixin):
    def __init__(
        self,
        # Data
        col_types=None,
        col_n_bins=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        random_state=42,
    ):
        super(CoreEBMRegressor, self).__init__(
            # Data
            col_types=col_types,
            col_n_bins=col_n_bins,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            random_state=random_state,
        )

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        return EBMUtils.regressor_predict(X, self)


class BaseEBM(BaseEstimator):
    """Client facing SK EBM."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):

        # Arguments for explainer
        self.feature_names = feature_names
        self.feature_types = feature_types

        # Arguments for data
        self.schema = schema

        # Arguments for ensemble
        self.n_estimators = n_estimators
        self.holdout_size = holdout_size
        self.scoring = scoring

        # Arguments for EBM beyond training a feature-step.
        self.interactions = interactions
        self.holdout_split = holdout_split
        self.data_n_episodes = data_n_episodes
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_run_length = early_stopping_run_length

        # Arguments for internal EBM.
        self.feature_step_n_inner_bags = feature_step_n_inner_bags
        self.learning_rate = learning_rate
        self.training_step_episodes = training_step_episodes
        self.max_tree_splits = max_tree_splits
        self.min_cases_for_splits = min_cases_for_splits

        # Arguments for overall
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X, y, self.feature_names, _ = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        # Build preprocessor
        self.schema_ = self.schema
        if self.schema_ is None:
            self.schema_ = autogen_schema(
                X, feature_names=self.feature_names, feature_types=self.feature_types
            )

        self.preprocessor_ = EBMPreprocessor(schema=self.schema_)
        self.preprocessor_.fit(X)

        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            proto_estimator = CoreEBMClassifier(
                # Data
                col_types=self.preprocessor_.col_types_,
                col_n_bins=self.preprocessor_.col_n_bins_,
                # Core
                interactions=self.interactions,
                holdout_split=self.holdout_split,
                data_n_episodes=self.data_n_episodes,
                early_stopping_tolerance=self.early_stopping_tolerance,
                early_stopping_run_length=self.early_stopping_run_length,
                # Native
                feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                learning_rate=self.learning_rate,
                training_step_episodes=self.training_step_episodes,
                max_tree_splits=self.max_tree_splits,
                min_cases_for_splits=self.min_cases_for_splits,
                # Overall
                random_state=self.random_state,
            )
        else:
            proto_estimator = CoreEBMRegressor(
                # Data
                col_types=self.preprocessor_.col_types_,
                col_n_bins=self.preprocessor_.col_n_bins_,
                # Core
                interactions=self.interactions,
                holdout_split=self.holdout_split,
                data_n_episodes=self.data_n_episodes,
                early_stopping_tolerance=self.early_stopping_tolerance,
                early_stopping_run_length=self.early_stopping_run_length,
                # Native
                feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                learning_rate=self.learning_rate,
                training_step_episodes=self.training_step_episodes,
                max_tree_splits=self.max_tree_splits,
                min_cases_for_splits=self.min_cases_for_splits,
                # Overall
                random_state=self.random_state,
            )

        # Train base models for main effects, pair detection.
        self.intercept_ = 0
        X_orig = X
        X = self.preprocessor_.transform(X)
        estimators = []
        for i in range(self.n_estimators):
            estimator = clone(proto_estimator)
            estimator.set_params(random_state=self.random_state + i)
            estimators.append(estimator)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        def train_model(estimator, X, y):
            return estimator.fit(X, y)

        train_model_args_iter = (
            (estimators[i], X, y) for i in range(self.n_estimators)
        )

        estimators = provider.parallel(train_model, train_model_args_iter)

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = self._select_merged_pairs(estimators, X, y)

            # Retrain interactions for base models
            def staged_fit_fn(estimator, X, y, inter_indices=[]):
                return estimator.staged_fit_interactions(X, y, inter_indices)

            staged_fit_args_iter = (
                (estimators[i], X, y, pair_indices) for i in range(self.n_estimators)
            )

            estimators = provider.parallel(staged_fit_fn, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
        else:
            raise RuntimeError("Argument 'interaction' has invalid value")

        # Average base models into one.
        self.attributes_ = EBMUtils.gen_attributes(
            self.preprocessor_.col_types_, self.preprocessor_.col_n_bins_
        )
        main_indices = [[x] for x in range(len(self.attributes_))]
        self.attribute_sets_ = EBMUtils.gen_attribute_sets(main_indices)
        self.attribute_sets_.extend(EBMUtils.gen_attribute_sets(pair_indices))

        # Merge estimators into one.
        self.attribute_set_models_ = []
        self.model_errors_ = []
        for index, _ in enumerate(self.attribute_sets_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.attribute_set_models_[index])
            averaged_model = np.average(np.array(log_odds_tensors), axis=0)

            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            self.attribute_set_models_.append(averaged_model)
            self.model_errors_.append(model_errors)

        # Extract feature names and feature types.
        self.feature_names = []
        self.feature_types = []
        for index, attribute_set in enumerate(self.attribute_sets_):
            feature_name = EBMUtils.gen_feature_name(
                attribute_set["attributes"], self.preprocessor_.col_names_
            )
            feature_type = EBMUtils.gen_feature_type(
                attribute_set["attributes"], self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_type)
            self.feature_names.append(feature_name)

        # Mean center graphs
        scores_gen = EBMUtils.scores_by_attrib_set(
            X, self.attribute_sets_, self.attribute_set_models_, []
        )
        self._attrib_set_model_means_ = []
        for set_idx, attribute_set, scores in scores_gen:
            score_mean = np.mean(scores)

            self.attribute_set_models_[set_idx] = (
                self.attribute_set_models_[set_idx] - score_mean
            )

            # Add mean center adjustment back to intercept
            self.intercept_ = self.intercept_ + score_mean
            self._attrib_set_model_means_.append(score_mean)

        # Generate overall importance
        scores_gen = EBMUtils.scores_by_attrib_set(
            X, self.attribute_sets_, self.attribute_set_models_, []
        )
        self.mean_abs_scores_ = []
        for set_idx, attribute_set, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            self.mean_abs_scores_.append(mean_abs_score)

        # Generate selector
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        self.has_fitted_ = True
        return self

    def _select_merged_pairs(self, estimators, X, y):
        # Select pairs from base models
        def score_fn(est, X, y, drop_indices):
            if is_classifier(est):
                prob = EBMUtils.classifier_predict_proba(X, estimator, drop_indices)
                return -1.0 * roc_auc_score(y, prob[:, 1])
            else:
                pred = EBMUtils.regressor_predict(X, estimator, drop_indices)
                return mean_squared_error(y, pred)

        pair_cum_rank = Counter()
        pair_freq = Counter()
        for index, estimator in enumerate(estimators):
            backward_impacts = []
            forward_impacts = []

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.holdout_split,
                random_state=estimator.random_state,
                stratify=y if is_classifier(self) else None,
            )
            base_forward_score = score_fn(
                estimator, X_val, y_val, estimator.inter_indices_
            )
            base_backward_score = score_fn(estimator, X_val, y_val, [])
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                pair_freq[pair] += 1
                backward_score = score_fn(
                    estimator, X_val, y_val, estimator.inter_indices_[pair_idx]
                )
                forward_score = score_fn(
                    estimator,
                    X_val,
                    y_val,
                    estimator.inter_indices_[:pair_idx]
                    + estimator.inter_indices_[pair_idx + 1 :],
                )
                backward_impact = backward_score - base_backward_score
                forward_impact = base_forward_score - forward_score

                backward_impacts.append(backward_impact)
                forward_impacts.append(forward_impact)

            # Average ranks
            backward_ranks = np.argsort(backward_impacts[::-1])
            forward_ranks = np.argsort(forward_impacts[::-1])
            pair_ranks = np.mean(np.array([backward_ranks, forward_ranks]), axis=0)

            # Add to cumulative rank for a pair across all models
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                pair_cum_rank[pair] += pair_ranks[pair_idx]

        # Calculate pair importance ranks
        pair_weighted_ranks = pair_cum_rank.copy()
        for pair, freq in pair_freq.items():
            # Calculate average rank
            pair_weighted_ranks[pair] /= freq
            # Reweight by frequency
            pair_weighted_ranks[pair] /= np.sqrt(freq)
        pair_weighted_ranks = sorted(pair_weighted_ranks.items(), key=lambda x: x[1])

        # Retrieve top K pairs
        pair_indices = [x[0] for x in pair_weighted_ranks[: self.interactions]]

        return pair_indices

    def decision_function(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        decision_scores = EBMUtils.decision_function(
            X, self.attribute_sets_, self.attribute_set_models_, self.intercept_
        )

        return decision_scores

    def explain_global(self, name=None):
        if name is None:
            name = gen_name_from_class(self)

        # Add per feature graph
        data_dicts = []
        for attribute_set_index, attribute_set in enumerate(self.attribute_sets_):
            model_graph = self.attribute_set_models_[attribute_set_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.model_errors_[attribute_set_index]
            attribute_indexes = self.attribute_sets_[attribute_set_index]["attributes"]

            if len(attribute_indexes) == 1:
                bin_labels = self.preprocessor_.get_bin_labels(attribute_indexes[0])
                # bin_counts = self.preprocessor_.get_bin_counts(
                #     attribute_indexes[0]
                # )
                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": list(model_graph),
                    "upper_bounds": list(model_graph + errors),
                    "lower_bounds": list(model_graph - errors),
                    "density": {
                        "names": self.preprocessor_.get_hist_edges(
                            attribute_indexes[0]
                        ),
                        "scores": self.preprocessor_.get_hist_counts(
                            attribute_indexes[0]
                        ),
                    },
                }
                data_dicts.append(data_dict)
            elif len(attribute_indexes) == 2:
                bin_labels_left = self.preprocessor_.get_bin_labels(
                    attribute_indexes[0]
                )
                bin_labels_right = self.preprocessor_.get_bin_labels(
                    attribute_indexes[1]
                )
                data_dict = {
                    "type": "pairwise",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                }
                data_dicts.append(data_dict)
            else:
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.mean_abs_scores_,
        }
        internal_obj = {"overall": overall_dict, "specific": data_dicts}

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def explain_local(self, X, y=None, name=None):
        # Produce feature value pairs for each instance.
        # Values are the model graph score per respective attribute set.
        if name is None:
            name = gen_name_from_class(self)

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)
        instances = self.preprocessor_.transform(X)
        scores_gen = EBMUtils.scores_by_attrib_set(
            instances, self.attribute_sets_, self.attribute_set_models_
        )

        n_rows = instances.shape[0]
        data_dicts = []
        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {
                    "names": ["Intercept"],
                    "scores": [self.intercept_],
                    "values": [1],
                },
            }
            data_dicts.append(data_dict)

        for set_idx, attribute_set, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if attribute_set["n_attributes"] == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, attribute_set["attributes"][0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        if is_classifier(self):
            scores = EBMUtils.classifier_predict_proba(instances, self)[:, 1]
        else:
            scores = EBMUtils.regressor_predict(instances, self)

        for row_idx in range(n_rows):
            data_dicts[row_idx]["perf"] = perf_dict(y, scores, row_idx)

        selector = gen_local_selector(instances, y, scores)

        internal_obj = {"overall": None, "specific": data_dicts}

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


class ExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):

        super(ExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Data
            schema=schema,
            # Ensemble
            n_estimators=n_estimators,
            holdout_size=holdout_size,
            scoring=scoring,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return EBMUtils.classifier_predict_proba(X, self)

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return EBMUtils.classifier_predict(X, self)


class ExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Data
        schema=None,
        # Ensemble
        n_estimators=16,
        holdout_size=0.15,
        scoring=None,
        # Core
        interactions=0,
        holdout_split=0.15,
        data_n_episodes=2000,
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):

        super(ExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Data
            schema=schema,
            # Ensemble
            n_estimators=n_estimators,
            holdout_size=holdout_size,
            scoring=scoring,
            # Core
            interactions=interactions,
            holdout_split=holdout_split,
            data_n_episodes=data_n_episodes,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_run_length=early_stopping_run_length,
            # Native
            feature_step_n_inner_bags=feature_step_n_inner_bags,
            learning_rate=learning_rate,
            training_step_episodes=training_step_episodes,
            max_tree_splits=max_tree_splits,
            min_cases_for_splits=min_cases_for_splits,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return EBMUtils.regressor_predict(X, self)
