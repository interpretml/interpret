# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from ...utils import perf_dict
from .utils import EBMUtils
from .internal import NativeHelper
from .postprocessing import multiclass_postprocess
from ...utils import unify_data, autogen_schema
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...provider.compute import JobLibProvider
from ...utils import gen_name_from_class, gen_global_selector, gen_local_selector

import numpy as np
from warnings import warn

from sklearn.base import is_classifier, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss, mean_squared_error
from collections import Counter

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
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
        from ...visual.plot import plot_continuous_bar, plot_horizontal_bar, sort_take

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Overall graph
        # TODO: Fix for multiclass classification
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

        # Continuous feature graph
        if (
            self.explanation_type == "global"
            and self.feature_types[key] == "continuous"
        ):
            title = self.feature_names[key]
            if (
                isinstance(data_dict["scores"], np.ndarray)
                and data_dict["scores"].ndim == 2
            ):
                figure = plot_continuous_bar(
                    data_dict, multiclass=True, show_error=False, title=title
                )
            else:
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
        max_n_bins=255,
        missing_constant=0,
        unknown_constant=0,
        feature_names=None,
        feature_types=None,
        binning_strategy="uniform",
    ):
        """ Initializes EBM preprocessor.

        Args:
            schema: A dictionary that encapsulates column information,
                    such as type and domain.
            max_n_bins: Max number of bins to process numeric features.
            missing_constant: Missing encoded as this constant.
            unknown_constant: Unknown encoded as this constant.
            feature_names: Feature names as list.
            feature_types: Feature types as list, for example "continuous" or "categorical".
            binning_strategy: Strategy to compute bins according to density if "quantile" or equidistant if "uniform".
        """
        self.schema = schema
        self.max_n_bins = max_n_bins
        self.missing_constant = missing_constant
        self.unknown_constant = unknown_constant
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.binning_strategy = binning_strategy

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

        self.schema_ = (
            self.schema
            if self.schema is not None
            else autogen_schema(
                X, feature_names=self.feature_names, feature_types=self.feature_types
            )
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
                if len(uniq_vals) < self.max_n_bins:
                    bins = list(sorted(uniq_vals))
                else:
                    if self.binning_strategy == "uniform":
                        bins = self.max_n_bins
                    elif self.binning_strategy == "quantile":
                        bins = np.unique(
                            np.quantile(
                                col_data, q=np.linspace(0, 1, self.max_n_bins + 1)
                            )
                        )
                    else:  # pragma: no cover
                        raise ValueError(
                            "Unknown binning_strategy: '{}'.".format(
                                self.binning_strategy
                            )
                        )

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

        schema = self.schema_
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

    def get_hist_counts(self, feature_index):
        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            return list(self.hist_counts_[feature_index])
        elif col_type == "categorical":
            return list(self.col_mapping_counts_[feature_index])
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_hist_edges(self, feature_index):
        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            return list(self.hist_edges_[feature_index])
        elif col_type == "categorical":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    # def get_bin_counts(self, feature_index):
    #     col_type = self.col_types_[feature_index]
    #     if col_type == 'continuous':
    #         return list(self.col_bin_counts_[feature_index])
    #     elif col_type == 'categorical':
    #         return list(self.col_mapping_counts_[feature_index])
    #     else:
    #         raise Exception("Cannot get counts for type: {0}".format(col_type))

    def get_bin_labels(self, feature_index):
        """ Returns bin labels for a given feature index.

        Args:
            feature_index: An integer for feature index.

        Returns:
            List of labels for bins.
        """

        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            return list(self.col_bin_edges_[feature_index])
        elif col_type == "ordinal":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        elif col_type == "categorical":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Unknown column type")


# TODO: Clean up
class BaseCoreEBM:
    """Internal use EBM."""

    # TODO PK decide if we should follow any kind of sklearn convention here with
    # our private class with respect to using trailing underscores

    # TODO PK do we really need all of these parameters??
    def __init__(
        self,
        model_type,
        # Data
        col_types,
        col_n_bins,
        # Core
        main_features,
        interactions,
        holdout_split,
        data_n_episodes,
        early_stopping_tolerance,
        early_stopping_run_length,
        # Native
        feature_step_n_inner_bags,
        learning_rate,
        boosting_step_episodes,
        max_tree_splits,
        min_cases_for_splits,
        # Overall
        random_state,
    ):

        self.model_type = model_type

        # Arguments for data
        self.col_types = col_types
        self.col_n_bins = col_n_bins

        # Arguments for EBM beyond training a feature-step.
        self.main_features = main_features
        self.interactions = interactions
        self.holdout_split = holdout_split
        self.data_n_episodes = data_n_episodes
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_run_length = early_stopping_run_length

        # Arguments for internal EBM.
        self.feature_step_n_inner_bags = feature_step_n_inner_bags
        self.learning_rate = learning_rate
        self.boosting_step_episodes = boosting_step_episodes
        self.max_tree_splits = max_tree_splits
        self.min_cases_for_splits = min_cases_for_splits

        # Arguments for overall
        self.random_state = random_state

    def fit_parallel(self, X, y, n_classes):
        self.n_classes_ = n_classes

        # Split data into train/val

        X_train, X_val, y_train, y_val = EBMUtils.ebm_train_test_split(
            X,
            y,
            test_size=self.holdout_split,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
        )

        # Define features
        self.features_ = EBMUtils.gen_features(self.col_types, self.col_n_bins)
        # Build EBM allocation code

        # scikit-learn returns an np.array for classification and
        # a single np.float64 for regression, so we do the same
        if self.model_type == "classification":
            self.intercept_ = np.zeros(
                EBMUtils.get_count_scores_c(self.n_classes_),
                dtype=np.float64,
                order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        if isinstance(self.main_features, str) and self.main_features == "all":
            main_feature_indices = [[x] for x in range(len(self.features_))]
        elif isinstance(self.main_features, list) and all(
            isinstance(x, int) for x in self.main_features
        ):
            main_feature_indices = [[x] for x in self.main_features]
        else:  # pragma: no cover
            raise RuntimeError("Argument 'main_attr' has invalid value")

        main_feature_combinations = EBMUtils.gen_feature_combinations(
            main_feature_indices
        )
        self.feature_combinations_ = []
        self.model_ = []

        # Train main effects
        self._fit_main(main_feature_combinations, X_train, y_train, X_val, y_val)

        # Build interaction terms, if required
        self.inter_indices_, self.inter_scores_ = self._build_interactions(
            X_train, y_train
        )

        self.inter_episode_idx_ = 0
        if len(self.inter_indices_) != 0:
            self._staged_fit_interactions(
                X_train, y_train, X_val, y_val, self.inter_indices_
            )

        return self

    def _fit_main(self, main_feature_combinations, X_train, y_train, X_val, y_val):
        log.info("Train main effects")

        (
            self.model_,
            self.current_metric_,
            self.main_episode_idx_,
        ) = NativeHelper.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features=self.features_,
            feature_combinations=main_feature_combinations,
            X_train=X_train,
            y_train=y_train,
            scores_train=None,
            X_val=X_val,
            y_val=y_val,
            scores_val=None,
            n_inner_bags=self.feature_step_n_inner_bags,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_tree_splits=self.max_tree_splits,
            min_cases_for_splits=self.min_cases_for_splits,
            boosting_step_episodes=self.boosting_step_episodes,
            data_n_episodes=self.data_n_episodes,
            early_stopping_tolerance=self.early_stopping_tolerance,
            early_stopping_run_length=self.early_stopping_run_length,
            name="Main",
        )

        self.feature_combinations_ = main_feature_combinations

        return

    def _build_interactions(self, X_train, y_train):
        if isinstance(self.interactions, int) and self.interactions != 0:
            log.info("Estimating with FAST")

            scores_train = EBMUtils.decision_function(
                X_train, self.feature_combinations_, self.model_, self.intercept_
            )

            iter_feature_combinations = combinations(range(len(self.col_types)), 2)

            final_indices, final_scores = NativeHelper.get_interactions(
                n_interactions=self.interactions,
                iter_feature_combinations=iter_feature_combinations,
                model_type=self.model_type,
                n_classes=self.n_classes_,
                features=self.features_,
                X=X_train,
                y=y_train,
                scores=scores_train,
            )
        elif isinstance(self.interactions, int) and self.interactions == 0:
            final_indices = []
            final_scores = []
        elif isinstance(self.interactions, list):
            final_indices = self.interactions
            final_scores = [None for _ in range(len(self.interactions))]
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        return final_indices, final_scores

    def _staged_fit_interactions(
        self, X_train, y_train, X_val, y_val, inter_indices=[]
    ):

        log.info("Training interactions")

        scores_train = EBMUtils.decision_function(
            X_train, self.feature_combinations_, self.model_, self.intercept_
        )
        scores_val = EBMUtils.decision_function(
            X_val, self.feature_combinations_, self.model_, self.intercept_
        )

        inter_feature_combinations = EBMUtils.gen_feature_combinations(inter_indices)

        (
            model_update,
            self.current_metric_,
            self.inter_episode_idx_,
        ) = NativeHelper.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features=self.features_,
            feature_combinations=inter_feature_combinations,
            X_train=X_train,
            y_train=y_train,
            scores_train=scores_train,
            X_val=X_val,
            y_val=y_val,
            scores_val=scores_val,
            n_inner_bags=self.feature_step_n_inner_bags,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_tree_splits=self.max_tree_splits,
            min_cases_for_splits=self.min_cases_for_splits,
            boosting_step_episodes=self.boosting_step_episodes,
            data_n_episodes=self.data_n_episodes,
            early_stopping_tolerance=self.early_stopping_tolerance,
            early_stopping_run_length=self.early_stopping_run_length,
            name="Pair",
        )

        self.model_.extend(model_update)
        self.feature_combinations_.extend(inter_feature_combinations)

        return

    def staged_fit_interactions_parallel(self, X, y, inter_indices=[]):

        log.info("Splitting train/test for interactions")

        # Split data into train/val
        # NOTE: ideally we would store the train/validation split in the
        #       remote processes, but joblib doesn't have a concept
        #       of keeping remote state, so we re-split our sets
        X_train, X_val, y_train, y_val = EBMUtils.ebm_train_test_split(
            X,
            y,
            test_size=self.holdout_split,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
        )

        self._staged_fit_interactions(X_train, y_train, X_val, y_val, inter_indices)
        return self


# TODO PK v.2 Should we be exposing data in this class, or use helper functions to return the values?
class BaseEBM(BaseEstimator):
    """Client facing SK EBM."""

    # Interface modeled after:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

    # According to scikit-learn convention, everything related to the dataset should be passed into the fit function, but
    # also according to convention the fit function should have only X, y, and instance weights.  For intelligibility
    # we need to include things like the names of features, so we need to break one of these rules.  We have chosen
    # to pass information about the dataset into the __init__ function because then it's possible to do the training
    # first and then later set things like the features names after training.  Also, people have become accustomed
    # to passing optional parameters into the __init__ function, but not the fit function, so we maintain that by
    # using __init__.  This is slightly inconcistent if the user passes in a pandas DataFrame which has feature column names,
    # but this still gives the user ultimate control since they can either keep the DataFrame names or pass in new ones to __init__
    # Lastly, scikit-learn probably doesn't include X, y, and weights in __init__ because those should be pickled given their
    # potential size.  We don't have that issue with our smaller extra dataset dependent parameters

    # TODO PK v.2 per above, we've decided to pass information related to the dataset in via __init__, but
    #             we need to decide then if we inlcude the trailing underscores for these variables, which include:
    #             feature_names, feature_types, schema, main_attr, interactions (for specific columns)
    #             per : https://scikit-learn.org/dev/developers/develop.html
    #             "Attributes that have been estimated from the data must always have a name ending with trailing underscore,
    #             for example the coefficients of some regression estimator would be stored in a coef_ attribute after fit has been called."

    def __init__(
        self,
        # Explainer
        # TODO PK v.2 feature_names is currently by feature_combination.  Perahps we need to make one per
        # feature as well, so would be called feature_names_by_feature and feature_names_by_feature_combination
        feature_names=None,
        # TODO PK v.2 look at how sklearn has thought about feature types -> https://github.com/scikit-learn/scikit-learn/pull/3346
        #      also look at lightGBM's categorical_feature parameter
        #      https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
        #
        # TODO PK v.2 feature_types is currently by feature_combination.  Perahps we need to make one per
        # feature as well, so would be called feature_types_by_feature and feature_types_by_feature_combination
        feature_types=None,
        # Data
        # TODO PK v.2 either add a bin_cuts parameter here, or add a preprocessor/transformer class input parameter here so that the user
        #             can specify a common algorithm for bin cutting to compare against other algorithms.
        #             PK -> I favor the simple solution of just passing in bin_cuts, since the user can then specify their own bin_cuts, which could be important
        #             for AUC or interpretability visualizations.  Anyone wanting to do specialized work in comparing our algorithm against others may want
        #             to precisely duplicate our binning procedure, but this is a very very small subset of our users, so they can just
        #             copy our internal bin cutting function -> we can make this easier by having a clean function just for bin cutting
        #             that other people can either call or copy if they want to do this specialized work of having exactly the same
        #             bins across two different ML algorithms.
        # TODO PK v.2 can we eliminate the schema parameter given that we also take feature_names and feature_types definitions in this interface?
        schema=None,
        # Ensemble
        n_estimators=16,
        # TODO PK v.2 holdout_size doesn't seem to do anything.  Eliminate.  holdout_split is used
        holdout_size=0.15,
        scoring=None,
        # Core
        # TODO PK v.2 change main_attr -> main_features (also look for anything with attr in it)
        main_attr="all",
        # TODO PK v.2 we should probably have two types of interaction terms.
        #             The first is either a number or array of numbres that indicates
        #             how many interactions at each dimension level(starting at two)
        #             This parameter should be part of __init__
        #             The second parameter would be a list of specific interaction sets
        #             that people may want to use
        #             both at the same time, and there isn't a good way to separate the two concepts
        #             without issues.  Also, the deserve to be in separate functions (init vs fit)
        # TODO PK v.2 change interactions to n_interactions which can either be a number for pairs
        #             or can be a list/tuple of integers which denote the number of interactions per dimension
        #             so (3,2,1) would mean 3 pairs, 2 tripples, 1 quadruple
        # TODO PK v.2 add specific_interactions list of interactions to include (n_interactions will not re-pick these).
        # Allow these to be in any order and don't sort that order, unlike the n_interactions parameter
        # TODO PK v.2 exclude -> exclude feature_combinations, either mains, or pairs or whatever.  This will take precedence over specific_interactions so anything there will be excluded
        interactions=0,
        # TODO PK v.2 use test_size instead of holdout_split, since sklearn does
        holdout_split=0.15,
        data_n_episodes=2000,
        # TODO PK v.2 eliminate early_stopping_tolerance (use zero for this!)
        early_stopping_tolerance=1e-5,
        early_stopping_run_length=50,
        # Native
        # TODO PK v.2 feature_step_n_inner_bags -> n_inner_bags
        feature_step_n_inner_bags=0,
        learning_rate=0.01,
        # TODO PK v.2 eliminate training_step_episodes (if not, rename to boosting_step_episodes)
        training_step_episodes=1,
        max_tree_splits=2,
        min_cases_for_splits=2,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Preprocessor
        binning_strategy="uniform",
    ):
        # TODO PK sanity check all our inputs

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
        self.main_attr = main_attr
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

        # Arguments for preprocessor
        self.binning_strategy = binning_strategy

    # NOTE: Consider refactoring later.
    def fit(self, X, y):  # noqa: C901
        # TODO PK we shouldn't expose our internal state until we are 100% sure that we succeeded
        #         so move everything to local variables until the end when we assign them to self.*

        # TODO PK we should do some basic checks here that X and y have the same dimensions and that
        #      they are well formed (look for NaNs, etc)

        # TODO PK handle calls where X.dim == 1.  This could occur if there was only 1 feature, or if
        #     there was only 1 instance?  We can differentiate either condition via y.dim and reshape
        #     AND add some tests for the X.dim == 1 scenario

        # TODO PK write an efficient striping converter for X that replaces unify_data for EBMs
        # algorithm: grap N columns and convert them to rows then process those by sending them to C
        X, y, self.feature_names, _ = unify_data(
            X, y, self.feature_names, self.feature_types
        )

        # Build preprocessor
        self.preprocessor_ = EBMPreprocessor(
            schema=self.schema,
            binning_strategy=self.binning_strategy,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
        )
        self.preprocessor_.fit(X)

        # TODO PK v.2 get rid of self.schema_ here since that's already included in self.preprocessor_.schema_
        self.schema_ = self.preprocessor_.schema_

        X_orig = X
        X = self.preprocessor_.transform(X)

        estimators = []
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            y = y.astype(np.int64, casting="unsafe", copy=False)
            n_classes = len(self.classes_)
            if n_classes > 2:  # pragma: no cover
                warn("Multiclass is still experimental. Subject to change per release.")
            if n_classes > 2 and self.interactions != 0:  # pragma: no cover
                raise RuntimeError(
                    "Multiclass with interactions currently not supported."
                )
            for i in range(self.n_estimators):
                estimator = BaseCoreEBM(
                    # Data
                    model_type="classification",
                    col_types=self.preprocessor_.col_types_,
                    col_n_bins=self.preprocessor_.col_n_bins_,
                    # Core
                    main_features=self.main_attr,
                    interactions=self.interactions,
                    holdout_split=self.holdout_split,
                    data_n_episodes=self.data_n_episodes,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_run_length=self.early_stopping_run_length,
                    # Native
                    feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                    learning_rate=self.learning_rate,
                    boosting_step_episodes=self.training_step_episodes,
                    max_tree_splits=self.max_tree_splits,
                    min_cases_for_splits=self.min_cases_for_splits,
                    # Overall
                    random_state=self.random_state + i,
                )
                estimators.append(estimator)
        else:
            n_classes = -1
            y = y.astype(np.float64, casting="unsafe", copy=False)
            for i in range(self.n_estimators):
                estimator = BaseCoreEBM(
                    # Data
                    model_type="regression",
                    col_types=self.preprocessor_.col_types_,
                    col_n_bins=self.preprocessor_.col_n_bins_,
                    # Core
                    main_features=self.main_attr,
                    interactions=self.interactions,
                    holdout_split=self.holdout_split,
                    data_n_episodes=self.data_n_episodes,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_run_length=self.early_stopping_run_length,
                    # Native
                    feature_step_n_inner_bags=self.feature_step_n_inner_bags,
                    learning_rate=self.learning_rate,
                    boosting_step_episodes=self.training_step_episodes,
                    max_tree_splits=self.max_tree_splits,
                    min_cases_for_splits=self.min_cases_for_splits,
                    # Overall
                    random_state=self.random_state + i,
                )
                estimators.append(estimator)

        # TODO PK v.2 eliminate self.n_classes_ for our public interface BaseEBM
        #             and let our consumers use len(self.classes_) like scikit
        #             our private BaseCoreEBM should continue to keep n_classes_
        self.n_classes_ = n_classes

        # Train base models for main effects, pair detection.

        # scikit-learn returns an np.array for classification and
        # a single float64 for regression, so we do the same
        if is_classifier(self):
            self.intercept_ = np.zeros(
                EBMUtils.get_count_scores_c(self.n_classes_),
                dtype=np.float64,
                order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        def train_model(estimator, X, y, n_classes):
            return estimator.fit_parallel(X, y, n_classes)

        train_model_args_iter = (
            (estimators[i], X, y, n_classes) for i in range(self.n_estimators)
        )

        estimators = provider.parallel(train_model, train_model_args_iter)

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = self._select_merged_pairs(estimators, X, y)

            for estimator in estimators:
                # Discard initial interactions
                new_model = []
                new_feature_combinations = []
                for i, feature_combination in enumerate(
                    estimator.feature_combinations_
                ):
                    if len(feature_combination["attributes"]) != 1:
                        continue
                    new_model.append(estimator.model_[i])
                    new_feature_combinations.append(estimator.feature_combinations_[i])
                estimator.model_ = new_model
                estimator.feature_combinations_ = new_feature_combinations
                estimator.inter_episode_idx_ = 0

            if len(pair_indices) != 0:
                # Retrain interactions for base models
                def staged_fit_fn(estimator, X, y, inter_indices=[]):
                    return estimator.staged_fit_interactions_parallel(
                        X, y, inter_indices
                    )

                staged_fit_args_iter = (
                    (estimators[i], X, y, pair_indices)
                    for i in range(self.n_estimators)
                )

                estimators = provider.parallel(staged_fit_fn, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        self.inter_indices_ = pair_indices

        X = np.ascontiguousarray(X.T)

        # Average base models into one.
        self.attributes_ = EBMUtils.gen_features(
            self.preprocessor_.col_types_, self.preprocessor_.col_n_bins_
        )
        if isinstance(self.main_attr, str) and self.main_attr == "all":
            main_indices = [[x] for x in range(len(self.attributes_))]
        elif isinstance(self.main_attr, list) and all(
            isinstance(x, int) for x in self.main_attr
        ):
            main_indices = [[x] for x in self.main_attr]
        else:  # pragma: no cover
            msg = "Argument 'main_attr' has invalid value (valid values are 'all'|list<int>): {}".format(
                self.main_attr
            )
            raise RuntimeError(msg)

        # TODO PK v.2 attribute_sets_ -> feature_combinations_
        self.attribute_sets_ = EBMUtils.gen_feature_combinations(main_indices)
        self.attribute_sets_.extend(EBMUtils.gen_feature_combinations(pair_indices))

        # Merge estimators into one.
        # TODO PK v.2 consider exposing our models as pandas.NDFrame
        # TODO PK v.2 attribute_set_models_ -> model_
        self.attribute_set_models_ = []
        self.model_errors_ = []
        for index, _ in enumerate(self.attribute_sets_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.model_[index])

            averaged_model = np.average(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            # TODO PK v.2 if we end up choosing to expand/contract by removing
            #             logits from multiclass models, averaged_model
            #             do it HERE AND apply post processing before returning

            self.attribute_set_models_.append(averaged_model)
            self.model_errors_.append(model_errors)

        # Get episode indexes for base estimators.
        self.main_episode_idxs_ = []
        # TODO PK v.2 inter_episode_idxs_ -> interaction_episode_idxs_
        #             (but does this need to be exposed at all)
        self.inter_episode_idxs_ = []
        for estimator in estimators:
            self.main_episode_idxs_.append(estimator.main_episode_idx_)
            self.inter_episode_idxs_.append(estimator.inter_episode_idx_)

        # Extract feature names and feature types.
        self.feature_names = []
        self.feature_types = []
        for index, feature_combination in enumerate(self.attribute_sets_):
            feature_name = EBMUtils.gen_feature_name(
                feature_combination["attributes"], self.preprocessor_.col_names_
            )
            feature_type = EBMUtils.gen_feature_type(
                feature_combination["attributes"], self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_type)
            self.feature_names.append(feature_name)

        if n_classes <= 2:
            # Mean center graphs - only for binary classification and regression
            scores_gen = EBMUtils.scores_by_feature_combination(
                X, self.attribute_sets_, self.attribute_set_models_
            )
            # TODO PK v.2 _attrib_set_model_means_ -> _model_means_
            # (or something else matching what this is being used for)
            # also look for anything with attrib inside of it
            self._attrib_set_model_means_ = []

            # TODO: Clean this up before release.
            for set_idx, feature_combination, scores in scores_gen:
                score_mean = np.mean(scores)

                self.attribute_set_models_[set_idx] = (
                    self.attribute_set_models_[set_idx] - score_mean
                )

                # Add mean center adjustment back to intercept
                self.intercept_ += score_mean
                self._attrib_set_model_means_.append(score_mean)
        else:
            # Postprocess model graphs for multiclass
            binned_predict_proba = lambda x: EBMUtils.classifier_predict_proba(
                x, self.attribute_sets_, self.attribute_set_models_, self.intercept_
            )

            postprocessed = multiclass_postprocess(
                X, self.attribute_set_models_, binned_predict_proba, self.feature_types
            )
            self.attribute_set_models_ = postprocessed["feature_graphs"]
            self.intercept_ = postprocessed["intercepts"]

        # Generate overall importance
        scores_gen = EBMUtils.scores_by_feature_combination(
            X, self.attribute_sets_, self.attribute_set_models_
        )
        self.mean_abs_scores_ = []
        for set_idx, feature_combination, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            self.mean_abs_scores_.append(mean_abs_score)

        # Generate selector
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        self.has_fitted_ = True
        return self

    def _select_merged_pairs(self, estimators, X, y):
        # TODO PK we really need to use purification before here because it's not really legal to elminate
        #         a feature combination unless it's average contribution value is zero, and for a pair that
        #         would mean that the intercepts for both features in the combination were zero, hense purified

        # Select pairs from base models
        def score_fn(model_type, X, y, feature_combinations, model, intercept):
            if model_type == "classification":
                prob = EBMUtils.classifier_predict_proba(
                    X, feature_combinations, model, intercept
                )
                return (
                    0 if len(y) == 0 else log_loss(y, prob)
                )  # use logloss to conform consistnetly and for multiclass
            elif model_type == "regression":
                pred = EBMUtils.regressor_predict(
                    X, feature_combinations, model, intercept
                )
                return 0 if len(y) == 0 else mean_squared_error(y, pred)
            else:  # pragma: no cover
                msg = "Unknown model_type: '{}'.".format(model_type)
                raise ValueError(msg)

        # TODO PK rename the "pair" variables in this function to "interaction" since that's more generalized

        # TODO PK sort the interaction tuples so that they have a unique ordering, otherwise
        #         when they get inserted into pair_cum_rank and pair_freq they could potentially have
        #         reversed ordering and then be duplicates
        #         ordering by increasing indexes is probably the most meaningful representation to the user

        pair_cum_rank = Counter()
        pair_freq = Counter()

        for index, estimator in enumerate(estimators):
            # TODO PK move the work done inside this loop to the original parallel threads so that this part can be done in parallel

            # TODO PK this algorithm in O(N^2) by the number of interactions.  Alternatively
            #         there is an O(N) algorithm where we generate the logits for the base forward and base backwards
            #         predictions, then we copy that entire array AND add or substract the one feature under consideration

            backward_impacts = []
            forward_impacts = []

            # TODO PK we can remove the is_train input to ebm_train_test_split once we've moved the pair scoring stuff
            #         to a background thread because we'll already have the validation split without re-splitting it
            _, X_val, _, y_val = EBMUtils.ebm_train_test_split(
                X,
                y,
                test_size=self.holdout_split,
                random_state=estimator.random_state,
                is_classification=is_classifier(self),
                is_train=False,
            )

            n_base_feature_combinations = len(estimator.feature_combinations_) - len(
                estimator.inter_indices_
            )

            base_forward_score = score_fn(
                estimator.model_type,
                X_val,
                y_val,
                estimator.feature_combinations_[:n_base_feature_combinations],
                estimator.model_[:n_base_feature_combinations],
                estimator.intercept_,
            )
            base_backward_score = score_fn(
                estimator.model_type,
                X_val,
                y_val,
                estimator.feature_combinations_,
                estimator.model_,
                estimator.intercept_,
            )
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                n_full_idx = n_base_feature_combinations + pair_idx

                pair_freq[pair] += 1

                backward_score = score_fn(
                    estimator.model_type,
                    X_val,
                    y_val,
                    estimator.feature_combinations_[:n_full_idx]
                    + estimator.feature_combinations_[n_full_idx + 1 :],
                    estimator.model_[:n_full_idx] + estimator.model_[n_full_idx + 1 :],
                    estimator.intercept_,
                )
                forward_score = score_fn(
                    estimator.model_type,
                    X_val,
                    y_val,
                    estimator.feature_combinations_[:n_base_feature_combinations]
                    + estimator.feature_combinations_[n_full_idx : n_full_idx + 1],
                    estimator.model_[:n_base_feature_combinations]
                    + estimator.model_[n_full_idx : n_full_idx + 1],
                    estimator.intercept_,
                )
                # for both regression (mean square error) and classification (log loss), higher values are bad, so
                # interactions with high positive values for backward_impact and forward_impact are good
                backward_impact = backward_score - base_backward_score
                forward_impact = base_forward_score - forward_score

                backward_impacts.append(backward_impact)
                forward_impacts.append(forward_impact)

            # Average ranks
            backward_ranks = np.argsort(backward_impacts)[::-1]
            forward_ranks = np.argsort(forward_impacts)[::-1]
            pair_ranks = np.mean(np.array([backward_ranks, forward_ranks]), axis=0)

            # Add to cumulative rank for a pair across all models
            for pair_idx, pair in enumerate(estimator.inter_indices_):
                pair_cum_rank[pair] += pair_ranks[pair_idx]

        # Calculate pair importance ranks
        # TODO PK this copy isn't required
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

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        decision_scores = EBMUtils.decision_function(
            X, self.attribute_sets_, self.attribute_set_models_, self.intercept_
        )

        # TODO PK v.2 these decision_scores are unexpanded.  We need to expand them
        return decision_scores

    def explain_global(self, name=None):
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_combination_index, feature_combination in enumerate(
            self.attribute_sets_
        ):
            errors = self.model_errors_[feature_combination_index]
            scores = self.attribute_set_models_[feature_combination_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_combination_index, feature_combination in enumerate(
            self.attribute_sets_
        ):
            model_graph = self.attribute_set_models_[feature_combination_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.model_errors_[feature_combination_index]
            feature_indexes = self.attribute_sets_[feature_combination_index][
                "attributes"
            ]

            if len(feature_indexes) == 1:
                bin_labels = self.preprocessor_.get_bin_labels(feature_indexes[0])
                # bin_counts = self.preprocessor_.get_bin_counts(
                #     feature_indexes[0]
                # )
                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": self.preprocessor_.get_hist_edges(feature_indexes[0]),
                    "scores": self.preprocessor_.get_hist_counts(feature_indexes[0]),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": self.preprocessor_.get_hist_edges(feature_indexes[0]),
                        "scores": self.preprocessor_.get_hist_counts(
                            feature_indexes[0]
                        ),
                    },
                }
                data_dicts.append(data_dict)
            elif len(feature_indexes) == 2:
                bin_labels_left = self.preprocessor_.get_bin_labels(feature_indexes[0])
                bin_labels_right = self.preprocessor_.get_bin_labels(feature_indexes[1])

                feature_dict = {
                    "type": "pairwise",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "pairwise",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                data_dicts.append(data_dict)
            else:  # pragma: no cover
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.mean_abs_scores_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

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
        # Values are the model graph score per respective feature combination.
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)
        instances = self.preprocessor_.transform(X)

        instances = np.ascontiguousarray(instances.T)

        scores_gen = EBMUtils.scores_by_feature_combination(
            instances, self.attribute_sets_, self.attribute_set_models_
        )

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        n_rows = instances.shape[1]
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

        for set_idx, feature_combination, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if len(feature_combination["attributes"]) == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, feature_combination["attributes"][0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        if is_classifier(self):
            scores = EBMUtils.classifier_predict_proba(
                instances,
                self.attribute_sets_,
                self.attribute_set_models_,
                self.intercept_,
            )[:, 1]
        else:
            scores = EBMUtils.regressor_predict(
                instances,
                self.attribute_sets_,
                self.attribute_set_models_,
                self.intercept_,
            )

        perf_list = []
        for row_idx in range(n_rows):
            perf = perf_dict(y, scores, row_idx)
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(y, scores)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": self.attribute_set_models_,
                        "intercept": self.intercept_,
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

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


class ExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    # TODO PK v.2 use underscores here like ClassifierMixin._estimator_type?
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
        main_attr="all",
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
        # Preprocessor
        binning_strategy="uniform",
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
            main_attr=main_attr,
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
            # Preprocessor
            binning_strategy=binning_strategy,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        prob = EBMUtils.classifier_predict_proba(
            X, self.attribute_sets_, self.attribute_set_models_, self.intercept_
        )
        return prob

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        return EBMUtils.classifier_predict(
            X,
            self.attribute_sets_,
            self.attribute_set_models_,
            self.intercept_,
            self.classes_,
        )


class ExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    # TODO PK v.2 use underscores here like RegressorMixin._estimator_type?
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
        main_attr="all",
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
        # Preprocessor
        binning_strategy="uniform",
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
            main_attr=main_attr,
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
            # Preprocessor
            binning_strategy=binning_strategy,
        )

    def predict(self, X):
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        return EBMUtils.regressor_predict(
            X, self.attribute_sets_, self.attribute_set_models_, self.intercept_
        )
