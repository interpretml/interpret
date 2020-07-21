# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from ...utils import gen_perf_dicts
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

from sklearn.base import is_classifier
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
    """ Visualizes specifically for EBM. """

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
        """ Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """
        super(EBMExplanation, self).__init__(
            explanation_type,
            internal_obj,
            feature_names=feature_names,
            feature_types=feature_types,
            name=name,
            selector=selector,
        )

    def visualize(self, key=None):
        """ Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from ...visual.plot import (
            plot_continuous_bar,
            plot_horizontal_bar,
            sort_take,
            is_multiclass_global_data_dict,
        )

        data_dict = self.data(key)
        if data_dict is None:
            return None

        # Overall graph
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
            if is_multiclass_global_data_dict(data_dict):
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
        self, feature_names=None, feature_types=None, max_bins=255, binning="quantile",
    ):
        """ Initializes EBM preprocessor.

        Args:
            feature_names: Feature names as list.
            feature_types: Feature types as list, for example "continuous" or "categorical".
            max_bins: Max number of bins to process numeric features.
            binning: Strategy to compute bins according to density if "quantile" or equidistant if "uniform".
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.binning = binning

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

        schema = autogen_schema(
            X, feature_names=self.feature_names, feature_types=self.feature_types
        )

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
                if len(uniq_vals) < self.max_bins:
                    bins = list(sorted(uniq_vals))
                else:
                    if self.binning == "uniform":
                        bins = self.max_bins
                    elif self.binning == "quantile":
                        bins = np.unique(
                            np.quantile(
                                col_data, q=np.linspace(0, 1, self.max_bins + 1)
                            )
                        )
                    else:  # pragma: no cover
                        raise ValueError("Unknown binning: '{}'.".format(self.binning))

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

        missing_constant = -1
        unknown_constant = -2

        X_new = np.copy(X)
        for col_idx in range(X.shape[1]):
            col_type = self.col_types_[col_idx]
            col_data = X[:, col_idx]

            if col_type == "continuous":
                col_data = col_data.astype(float)
                bin_edges = self.col_bin_edges_[col_idx].copy()

                digitized = np.digitize(col_data, bin_edges, right=False)
                digitized[digitized == 0] = 1
                digitized -= 1

                # NOTE: NA handling done later.
                # digitized[np.isnan(col_data)] = missing_constant
                X_new[:, col_idx] = digitized
            elif col_type == "ordinal":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)
            elif col_type == "categorical":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else unknown_constant
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
        validation_size,
        max_rounds,
        early_stopping_tolerance,
        early_stopping_rounds,
        # Native
        inner_bags,
        learning_rate,
        max_leaves,
        min_samples_leaf,
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
        self.validation_size = validation_size
        self.max_rounds = max_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_rounds = early_stopping_rounds

        # Arguments for internal EBM.
        self.inner_bags = inner_bags
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf

        # Arguments for overall
        self.random_state = random_state

    def fit_parallel(self, X, y, n_classes):
        self.n_classes_ = n_classes

        # Split data into train/val

        X_train, X_val, y_train, y_val = EBMUtils.ebm_train_test_split(
            X,
            y,
            test_size=self.validation_size,
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
            raise RuntimeError("Argument 'mains' has invalid value")

        self.feature_groups_ = []
        self.model_ = []

        # Train main effects
        self._fit_main(main_feature_indices, X_train, y_train, X_val, y_val)

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

    def _fit_main(self, main_feature_groups, X_train, y_train, X_val, y_val):
        log.info("Train main effects")

        (
            self.model_,
            self.current_metric_,
            self.main_episode_idx_,
        ) = NativeHelper.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features=self.features_,
            feature_combinations=main_feature_groups,
            X_train=X_train,
            y_train=y_train,
            scores_train=None,
            X_val=X_val,
            y_val=y_val,
            scores_val=None,
            n_inner_bags=self.inner_bags,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_leaves=self.max_leaves,
            min_samples_leaf=self.min_samples_leaf,
            max_rounds=self.max_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            early_stopping_rounds=self.early_stopping_rounds,
            name="Main",
        )

        self.feature_groups_ = main_feature_groups

        return

    def _build_interactions(self, X_train, y_train):
        if isinstance(self.interactions, int) and self.interactions != 0:
            log.info("Estimating with FAST")

            scores_train = EBMUtils.decision_function(
                X_train, self.feature_groups_, self.model_, self.intercept_
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
                min_samples_leaf=self.min_samples_leaf,
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
            X_train, self.feature_groups_, self.model_, self.intercept_
        )
        scores_val = EBMUtils.decision_function(
            X_val, self.feature_groups_, self.model_, self.intercept_
        )

        (
            model_update,
            self.current_metric_,
            self.inter_episode_idx_,
        ) = NativeHelper.cyclic_gradient_boost(
            model_type=self.model_type,
            n_classes=self.n_classes_,
            features=self.features_,
            feature_combinations=inter_indices,
            X_train=X_train,
            y_train=y_train,
            scores_train=scores_train,
            X_val=X_val,
            y_val=y_val,
            scores_val=scores_val,
            n_inner_bags=self.inner_bags,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            max_leaves=self.max_leaves,
            min_samples_leaf=self.min_samples_leaf,
            max_rounds=self.max_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
            early_stopping_rounds=self.early_stopping_rounds,
            name="Pair",
        )

        self.model_.extend(model_update)
        self.feature_groups_.extend(inter_indices)

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
            test_size=self.validation_size,
            random_state=self.random_state,
            is_classification=self.model_type == "classification",
        )

        self._staged_fit_interactions(X_train, y_train, X_val, y_val, inter_indices)
        return self


class BaseEBM(BaseEstimator):
    """Client facing SK EBM."""

    # Interface modeled after:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

    def __init__(
        self,
        # Explainer
        #
        # feature_names in scikit-learn convention should probably be passed in via the fit function.  Also,
        #   we can get feature_names via pandas dataframes, and those would only be known at fit time, so
        #   we need a version of feature_names_out_ with the underscore to indicate items set at fit time.
        #   Despite this, we need to recieve a list of feature_names here to be compatible with blackbox explainations
        #   where we still need to have feature_names, but we do not have a fit function since we explain existing
        #   models without fitting them ourselves.  To conform to a common explaination API we get the feature_names
        #   here.
        feature_names,
        # other packages LightGBM, CatBoost, Scikit-Learn (future) are using categorical specific ways to indicate
        #   feature_types.  The benefit to them is that they can accept multiple ways of specifying categoricals like:
        #   categorical = [true, false, true, true] OR categorical = [1, 4, 8] OR categorical = 'all'/'auto'/'none'
        #   We're choosing a different route because for visualization we want to be able to express multiple
        #   different types of data.  For example, if the user has data with strings of "low", "medium", "high"
        #   We want to keep both the ordinal nature of this feature and we wish to preserve the text for visualization
        #   scikit-learn callers can pre-convert these things to [0, 1, 2] in the correct order because they don't
        #   need to worry about visualizing the data afterwards, but for us we  need a way to specify the strings
        #   back anyways.  So we need some way to express both the categorical nature of features and the order
        #   mapping.  We can do this and more complicated conversions via:
        #   feature_types = ["categorical", ["low", "medium", "high"], "continuous", "time", "bool"]
        feature_types,
        # Data
        #
        # TODO PK v.3 add a bin_cuts parameter to allow the user to control binning
        # Ensemble
        outer_bags,
        inner_bags,
        # Core
        # TODO PK v.3 mains will be deprecated in the future in favor of "boosting_stage_plan"
        mains,
        # TODO PK v.2 we should probably have two types of interaction terms.
        #             The first is either a number or array of numbres that indicates
        #             how many interactions at each dimension level (starting at two)
        #             The second parameter would be a list of specific interaction sets
        #             that people may want to use.  There isn't a good way to separate the two concepts
        #             without issues.
        # TODO PK v.2 change interactions to n_interactions which can either be a number for pairs
        #             or can be a list/tuple of integers which denote the number of interactions per dimension
        #             so (3,2,1) would mean 3 pairs, 2 tripples, 1 quadruple
        # TODO PK v.2 add specific_interactions list of interactions to include (n_interactions will not re-pick these).
        #             Allow these to be in any order and don't sort that order, unlike the n_interactions parameter
        # TODO PK v.2 exclude -> exclude feature_combinations, either mains, or pairs or whatever.  This will take precedence over specific_interactions so anything there will be excluded
        interactions,
        validation_size,
        max_rounds,
        early_stopping_tolerance,
        early_stopping_rounds,
        # Native
        learning_rate,
        max_leaves,
        # Holte, R. C. (1993) "Very simple classification rules perform well on most commonly used datasets"
        # says use 6 as the minimum instances https://link.springer.com/content/pdf/10.1023/A:1022631118932.pdf
        # TODO PK try setting this (not here, but in our caller) to 6 and run tests to verify the best value.
        min_samples_leaf,
        # Overall
        n_jobs,
        random_state,
        # Preprocessor
        binning,
        max_bins,
    ):
        # TODO PK sanity check all our inputs

        # Arguments for explainer
        self.feature_names = feature_names
        self.feature_types = feature_types

        # Arguments for ensemble
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags

        # Arguments for EBM beyond training a feature-step.
        self.mains = mains
        self.interactions = interactions
        self.validation_size = validation_size
        self.max_rounds = max_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_rounds = early_stopping_rounds

        # Arguments for internal EBM.
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf

        # Arguments for overall
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Arguments for preprocessor
        self.binning = binning
        self.max_bins = max_bins

    # NOTE: Generally, we want to keep parameters in the __init__ function, since scikit-learn
    #       doesn't like parameters in the fit function, other than ones like weights that have
    #       the same length as the number of instances.  See:
    #       https://github.com/microsoft/LightGBM/issues/2628#issue-536116395
    #
    # NOTE: Consider refactoring later.
    def fit(self, X, y):  # noqa: C901
        """ Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
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
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            max_bins=self.max_bins,
            binning=self.binning,
        )
        self.preprocessor_.fit(X)

        X_orig = X
        X = self.preprocessor_.transform(X)

        estimators = []
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self._class_idx_ = {x: index for index, x in enumerate(self.classes_)}

            y = y.astype(np.int64, casting="unsafe", copy=False)
            n_classes = len(self.classes_)
            if n_classes > 2:  # pragma: no cover
                warn("Multiclass is still experimental. Subject to change per release.")
            if n_classes > 2 and self.interactions != 0:  # pragma: no cover
                raise RuntimeError(
                    "Multiclass with interactions currently not supported."
                )
            for i in range(self.outer_bags):
                estimator = BaseCoreEBM(
                    # Data
                    model_type="classification",
                    col_types=self.preprocessor_.col_types_,
                    col_n_bins=self.preprocessor_.col_n_bins_,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    max_leaves=self.max_leaves,
                    min_samples_leaf=self.min_samples_leaf,
                    # Overall
                    random_state=self.random_state + i,
                )
                estimators.append(estimator)
        else:
            n_classes = -1
            y = y.astype(np.float64, casting="unsafe", copy=False)
            for i in range(self.outer_bags):
                estimator = BaseCoreEBM(
                    # Data
                    model_type="regression",
                    col_types=self.preprocessor_.col_types_,
                    col_n_bins=self.preprocessor_.col_n_bins_,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    max_leaves=self.max_leaves,
                    min_samples_leaf=self.min_samples_leaf,
                    # Overall
                    random_state=self.random_state + i,
                )
                estimators.append(estimator)

        # Train base models for main effects, pair detection.

        # scikit-learn returns an np.array for classification and
        # a single float64 for regression, so we do the same
        if is_classifier(self):
            self.intercept_ = np.zeros(
                EBMUtils.get_count_scores_c(n_classes), dtype=np.float64, order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        def train_model(estimator, X, y, n_classes):
            return estimator.fit_parallel(X, y, n_classes)

        train_model_args_iter = (
            (estimators[i], X, y, n_classes) for i in range(self.outer_bags)
        )

        estimators = provider.parallel(train_model, train_model_args_iter)

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = self._select_merged_pairs(estimators, X, y)

            for estimator in estimators:
                # Discard initial interactions
                new_model = []
                new_feature_combinations = []
                for i, feature_combination in enumerate(estimator.feature_groups_):
                    if len(feature_combination) != 1:
                        continue
                    new_model.append(estimator.model_[i])
                    new_feature_combinations.append(estimator.feature_groups_[i])
                estimator.model_ = new_model
                estimator.feature_groups_ = new_feature_combinations
                estimator.inter_episode_idx_ = 0

            if len(pair_indices) != 0:
                # Retrain interactions for base models
                def staged_fit_fn(estimator, X, y, inter_indices=[]):
                    return estimator.staged_fit_interactions_parallel(
                        X, y, inter_indices
                    )

                staged_fit_args_iter = (
                    (estimators[i], X, y, pair_indices) for i in range(self.outer_bags)
                )

                estimators = provider.parallel(staged_fit_fn, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        X = np.ascontiguousarray(X.T)

        if isinstance(self.mains, str) and self.mains == "all":
            main_indices = [[x] for x in range(X.shape[0])]
        elif isinstance(self.mains, list) and all(
            isinstance(x, int) for x in self.mains
        ):
            main_indices = [[x] for x in self.mains]
        else:  # pragma: no cover
            msg = "Argument 'mains' has invalid value (valid values are 'all'|list<int>): {}".format(
                self.mains
            )
            raise RuntimeError(msg)

        self.feature_groups_ = main_indices + pair_indices

        # Merge estimators into one.
        self.additive_terms_ = []
        self.term_standard_deviations_ = []
        for index, _ in enumerate(self.feature_groups_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.model_[index])

            averaged_model = np.average(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            self.additive_terms_.append(averaged_model)
            self.term_standard_deviations_.append(model_errors)

        # Get episode indexes for base estimators.
        main_episode_idxs = []
        inter_episode_idxs = []
        for estimator in estimators:
            main_episode_idxs.append(estimator.main_episode_idx_)
            inter_episode_idxs.append(estimator.inter_episode_idx_)

        self.breakpoint_iteration_ = [main_episode_idxs]
        if len(pair_indices) != 0:
            self.breakpoint_iteration_.append(inter_episode_idxs)

        # Extract feature names and feature types.
        self.feature_names = []
        self.feature_types = []
        for index, feature_indices in enumerate(self.feature_groups_):
            feature_name = EBMUtils.gen_feature_name(
                feature_indices, self.preprocessor_.col_names_
            )
            feature_type = EBMUtils.gen_feature_type(
                feature_indices, self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_type)
            self.feature_names.append(feature_name)

        if n_classes <= 2:
            # Mean center graphs - only for binary classification and regression
            scores_gen = EBMUtils.scores_by_feature_combination(
                X, self.feature_groups_, self.additive_terms_
            )
            self._original_term_means_ = []

            for set_idx, _, scores in scores_gen:
                score_mean = np.mean(scores)

                self.additive_terms_[set_idx] = (
                    self.additive_terms_[set_idx] - score_mean
                )

                # Add mean center adjustment back to intercept
                self.intercept_ += score_mean
                self._original_term_means_.append(score_mean)
        else:
            # Postprocess model graphs for multiclass
            binned_predict_proba = lambda x: EBMUtils.classifier_predict_proba(
                x, self.feature_groups_, self.additive_terms_, self.intercept_
            )

            postprocessed = multiclass_postprocess(
                X, self.additive_terms_, binned_predict_proba, self.feature_types
            )
            self.additive_terms_ = postprocessed["feature_graphs"]
            self.intercept_ = postprocessed["intercepts"]

        # Generate overall importance
        scores_gen = EBMUtils.scores_by_feature_combination(
            X, self.feature_groups_, self.additive_terms_
        )
        self.feature_importances_ = []
        for set_idx, _, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            self.feature_importances_.append(mean_abs_score)

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
                test_size=self.validation_size,
                random_state=estimator.random_state,
                is_classification=is_classifier(self),
                is_train=False,
            )

            n_base_feature_combinations = len(estimator.feature_groups_) - len(
                estimator.inter_indices_
            )

            base_forward_score = score_fn(
                estimator.model_type,
                X_val,
                y_val,
                estimator.feature_groups_[:n_base_feature_combinations],
                estimator.model_[:n_base_feature_combinations],
                estimator.intercept_,
            )
            base_backward_score = score_fn(
                estimator.model_type,
                X_val,
                y_val,
                estimator.feature_groups_,
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
                    estimator.feature_groups_[:n_full_idx]
                    + estimator.feature_groups_[n_full_idx + 1 :],
                    estimator.model_[:n_full_idx] + estimator.model_[n_full_idx + 1 :],
                    estimator.intercept_,
                )
                forward_score = score_fn(
                    estimator.model_type,
                    X_val,
                    y_val,
                    estimator.feature_groups_[:n_base_feature_combinations]
                    + estimator.feature_groups_[n_full_idx : n_full_idx + 1],
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
        pair_indices = [list(x[0]) for x in pair_weighted_ranks[: self.interactions]]

        return pair_indices

    def decision_function(self, X):
        """ Predict scores from model before calling the link function.

            Args:
                X: Numpy array for instances.

            Returns:
                The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        decision_scores = EBMUtils.decision_function(
            X, self.feature_groups_, self.additive_terms_, self.intercept_
        )

        return decision_scores

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

        check_is_fitted(self, "has_fitted_")

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_combination_index, _ in enumerate(self.feature_groups_):
            errors = self.term_standard_deviations_[feature_combination_index]
            scores = self.additive_terms_[feature_combination_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_combination_index, feature_indexes in enumerate(
            self.feature_groups_
        ):
            model_graph = self.additive_terms_[feature_combination_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.term_standard_deviations_[feature_combination_index]

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
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
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
            "scores": self.feature_importances_,
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
        """ Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        # Produce feature value pairs for each instance.
        # Values are the model graph score per respective feature combination.
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types)

        # Transform y if classifier
        if is_classifier(self) and y is not None:
            y = np.array([self._class_idx_[el] for el in y])

        instances = self.preprocessor_.transform(X)

        instances = np.ascontiguousarray(instances.T)

        scores_gen = EBMUtils.scores_by_feature_combination(
            instances, self.feature_groups_, self.additive_terms_
        )

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        n_rows = instances.shape[1]
        data_dicts = []
        intercept = self.intercept_
        if not is_classifier(self) or len(self.classes_) <= 2:
            if isinstance(self.intercept_, np.ndarray) or isinstance(
                self.intercept_, list
            ):
                intercept = intercept[0]

        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            if is_classifier(self):
                data_dict["meta"] = {
                    "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                }
            data_dicts.append(data_dict)

        for set_idx, feature_combination, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if len(feature_combination) == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, feature_combination[0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        is_classification = is_classifier(self)
        if is_classification:
            scores = EBMUtils.classifier_predict_proba(
                instances, self.feature_groups_, self.additive_terms_, self.intercept_,
            )
        else:
            scores = EBMUtils.regressor_predict(
                instances, self.feature_groups_, self.additive_terms_, self.intercept_,
            )

        perf_list = []
        perf_dicts = gen_perf_dicts(scores, y, is_classification)
        for row_idx in range(n_rows):
            perf = None if perf_dicts is None else perf_dicts[row_idx]
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classification)

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": self.additive_terms_,
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
    """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog. """

    # TODO PK v.2 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=255,
        binning="quantile",
        # Stages
        mains="all",
        interactions=0,
        # Ensemble
        outer_bags=16,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=0,
        max_rounds=5000,
        # Trees
        max_leaves=3,
        min_samples_leaf=2,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            binning: Method to bin values for pre-processing. Choose "uniform" or "quantile".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(ExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            binning=binning,
            # Stages
            mains=mains,
            interactions=interactions,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            max_rounds=max_rounds,
            # Trees
            max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X):
        """ Probability estimates on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Probability estimate of instance for each class.
        """
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        prob = EBMUtils.classifier_predict_proba(
            X, self.feature_groups_, self.additive_terms_, self.intercept_
        )
        return prob

    def predict(self, X):
        """ Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        return EBMUtils.classifier_predict(
            X,
            self.feature_groups_,
            self.additive_terms_,
            self.intercept_,
            self.classes_,
        )


class ExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    """ Explainable Boosting Regressor. The arguments will change in a future release, watch the changelog. """

    # TODO PK v.2 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=255,
        binning="quantile",
        # Stages
        mains="all",
        interactions=0,
        # Ensemble
        outer_bags=16,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=0,
        max_rounds=5000,
        # Trees
        max_leaves=3,
        min_samples_leaf=2,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Regressor. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            binning: Method to bin values for pre-processing. Choose "uniform" or "quantile".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(ExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            binning=binning,
            # Stages
            mains=mains,
            interactions=interactions,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            max_rounds=max_rounds,
            # Trees
            max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict(self, X):
        """ Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """
        check_is_fitted(self, "has_fitted_")
        X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        X = np.ascontiguousarray(X.T)

        return EBMUtils.regressor_predict(
            X, self.feature_groups_, self.additive_terms_, self.intercept_
        )
