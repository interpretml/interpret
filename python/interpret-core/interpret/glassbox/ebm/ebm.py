# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from typing import DefaultDict

from interpret.provider.visualize import PreserveProvider
from ...utils import gen_perf_dicts
from .utils import DPUtils, EBMUtils, process_terms
from .bin import clean_X, clean_vector, construct_bins, bin_native_by_dimension, ebm_decision_function, ebm_decision_function_and_explain, make_boosting_weights, after_boosting, remove_last2, get_counts_and_weights, trim_tensor, unify_data2, eval_terms
from .internal import Native
from ...utils import unify_data, autogen_schema, unify_vector
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...provider.compute import JobLibProvider
from ...utils import gen_name_from_class, gen_global_selector, gen_global_selector2, gen_local_selector

import gc
import json
import math

import numpy as np
from warnings import warn

from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss, mean_squared_error
import heapq

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.utils.extmath import softmax
from itertools import combinations, groupby

import logging

_log = logging.getLogger(__name__)


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


def is_private(estimator):
    """Return True if the given estimator is a differentially private EBM estimator
    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a differentially private EBM estimator and False otherwise.
    """

    return isinstance(estimator, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor))

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

    # TODO: order these parameters the same as our public parameter list
    def __init__(
        self,
        # Explainer
        feature_names,
        feature_types,
        # Ensemble
        outer_bags,
        inner_bags,
        # Core
        mains, # TODO PK v.3 replace "mains" with a more flexible "exclude" parameter
        interactions,
        validation_size,
        max_rounds,
        early_stopping_tolerance,
        early_stopping_rounds,
        # Native
        learning_rate,
        # Holte, R. C. (1993) "Very simple classification rules perform well on most commonly used datasets"
        # says use 6 as the minimum samples https://link.springer.com/content/pdf/10.1023/A:1022631118932.pdf
        # TODO PK try setting this (not here, but in our caller) to 6 and run tests to verify the best value.
        min_samples_leaf,
        max_leaves,
        # Overall
        n_jobs,
        random_state,
        # Preprocessor
        binning,
        max_bins,
        max_interaction_bins,
        # Differential Privacy
        epsilon=None,
        delta=None,
        composition=None,
        bin_budget_frac=None,
        privacy_schema=None,
    ):
        # Arguments for explainer
        self.feature_names = feature_names
        self.feature_types = feature_types

        # Arguments for ensemble
        self.outer_bags = outer_bags
        if not is_private(self):
            self.inner_bags = inner_bags

        # Arguments for EBM beyond training a feature-step.
        self.mains = mains
        if not is_private(self):
            self.interactions = interactions
        self.validation_size = validation_size
        self.max_rounds = max_rounds
        if not is_private(self):
            self.early_stopping_tolerance = early_stopping_tolerance
            self.early_stopping_rounds = early_stopping_rounds

        # Arguments for internal EBM.
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaves = max_leaves

        # Arguments for overall
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Arguments for preprocessor
        self.binning = binning
        self.max_bins = max_bins
        if not is_private(self):
            self.max_interaction_bins = max_interaction_bins

        # Arguments for differential privacy
        if is_private(self):
            self.epsilon = epsilon
            self.delta = delta
            self.composition = composition
            self.bin_budget_frac = bin_budget_frac
            self.privacy_schema = privacy_schema

    def fit(self, X, y, sample_weight=None):  # noqa: C901
        """ Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.

        Returns:
            Itself.
        """

        # sometimes building EBMs takes a lot of memory, and sometimes the OS terminates big memory programs without
        # warning, so clear away as much previous cruft as possible before we allocate big chunks of memory
        gc.collect()

        X, n_samples = clean_X(X)
        if n_samples == 0:
            msg = "X has 0 samples"
            _log.error(msg)
            raise ValueError(msg)

        if is_classifier(self):
            y = clean_vector(y, True, "y")
            # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
            # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
            # in two separate runs, which would flip the ordering of the classes within our score tensors.
            classes, y = np.unique(y, return_inverse=True)
            n_classes = len(classes)
            if n_classes > 2:  # pragma: no cover
                if is_private(self):
                    raise ValueError("multiclass not supported in Differentially private EBMs")

                warn("Multiclass is still experimental. Subject to change per release.")

            class_idx = {x: index for index, x in enumerate(classes)}
        else:
            y = clean_vector(y, False, "y")
            min_target = y.min()
            max_target = y.max()
            n_classes = -1

        if n_samples != len(y):
            msg = f"X has {n_samples} samples and y has {len(y)} samples"
            _log.error(msg)
            raise ValueError(msg)

        if sample_weight is not None:
            sample_weight = clean_vector(sample_weight, False, "sample_weight")
            if n_samples != len(sample_weight):
                msg = f"X has {n_samples} samples and sample_weight has {len(sample_weight)} samples"
                _log.error(msg)
                raise ValueError(msg)

        # Privacy calculations
        noise_scale = None
        bin_eps_ = None
        bin_delta_ = None
        composition=None
        if is_private(self):
            DPUtils.validate_eps_delta(self.epsilon, self.delta)

            if not is_classifier(self):
                bounds = None if self.privacy_schema is None else self.privacy_schema.get('target', None)
                if bounds is None:
                    warn("Possible privacy violation: assuming min/max values for target are public info."
                            "Pass a privacy schema with known public target ranges to avoid this warning.")
                else:
                    min_target = bounds[0]
                    max_target = bounds[1]
                    if max_target < min_target:
                        raise ValueError(f"target minimum {min_target} must be smaller than maximum {max_target}")

                    y = np.clip(y, min_target, max_target)

            # Split epsilon, delta budget for binning and learning
            bin_eps_ = self.epsilon * self.bin_budget_frac
            training_eps_ = self.epsilon - bin_eps_
            bin_delta_ = self.delta / 2
            training_delta_ = self.delta / 2
            composition=self.composition

            # TODO: remove the + 1 for max_bins and max_interaction_bins.  It's just here to compare to the previous results!
            bin_levels = [self.max_bins + 1]
        else:
            # TODO: remove the + 1 for max_bins and max_interaction_bins.  It's just here to compare to the previous results!
            bin_levels = [self.max_bins + 1, self.max_interaction_bins + 1]

        binning_result = construct_bins(
            X=X,
            sample_weight=sample_weight,
            feature_names_given=self.feature_names, 
            feature_types_given=self.feature_types, 
            max_bins_leveled=bin_levels, 
            binning=self.binning, 
            min_samples_bin=1, 
            min_unique_continuous=3, 
            epsilon=bin_eps_, 
            delta=bin_delta_, 
            composition=composition,
            privacy_schema=getattr(self, 'privacy_schema', None)
        )
        feature_names_in = binning_result[0]
        feature_types_in = binning_result[1]
        bins = binning_result[2]
        term_bin_weights = binning_result[3]
        min_vals = binning_result[4]
        max_vals = binning_result[5]
        histogram_cuts = binning_result[6]
        histogram_counts = binning_result[7]
        unique_counts = binning_result[8]
        zero_counts = binning_result[9]

        n_features_in = len(feature_names_in)

        if is_private(self):
             # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            domain_size = 1 if is_classifier(self) else max_target - min_target
            max_weight = 1 if sample_weight is None else np.max(sample_weight)
            if self.composition == 'classic':
                noise_scale = DPUtils.calc_classic_noise_multi(
                    total_queries = self.max_rounds * n_features_in * self.outer_bags, 
                    target_epsilon = training_eps_, 
                    delta = training_delta_, 
                    sensitivity = domain_size * self.learning_rate * max_weight
                )
            elif self.composition == 'gdp':
                noise_scale = DPUtils.calc_gdp_noise_multi(
                    total_queries = self.max_rounds * n_features_in * self.outer_bags, 
                    target_epsilon = training_eps_, 
                    delta = training_delta_
                )
                noise_scale = noise_scale * domain_size * self.learning_rate * max_weight # Alg Line 17
            else:
                raise NotImplementedError(f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'.")

        dataset = bin_native_by_dimension(
            n_classes, 
            1,
            bins,
            X, 
            y, 
            sample_weight, 
            feature_names_in, 
            feature_types_in, 
        )

        bin_data_weights = None
        if is_private(self):
            bin_data_weights = make_boosting_weights(term_bin_weights)

        native = Native.get_native_singleton()

        provider = JobLibProvider(n_jobs=self.n_jobs)

        if isinstance(self.mains, str) and self.mains == "all":
            feature_groups = [(x,) for x in range(n_features_in)]
        else:
            feature_groups = [(int(x),) for x in self.mains]
              
        # Train main effects
        if is_private(self):
            update = Native.GenerateUpdateOptions_GradientSums | Native.GenerateUpdateOptions_RandomSplits
        else:
            update = Native.GenerateUpdateOptions_Default

        init_seed = EBMUtils.normalize_initial_random_seed(self.random_state)

        inner_bags = 0 if is_private(self) else self.inner_bags
        early_stopping_rounds = -1 if is_private(self) else self.early_stopping_rounds
        early_stopping_tolerance = -1 if is_private(self) else self.early_stopping_tolerance

        bags = []
        bagged_seed = init_seed
        for _ in range(self.outer_bags):
            bagged_seed=native.generate_random_number(bagged_seed, 1416147523)
            bags.append(EBMUtils.make_bag(y, self.validation_size, bagged_seed, is_classifier(self)))

        parallel_args = []
        bagged_seed = init_seed
        for idx in range(self.outer_bags):
            bagged_seed=native.generate_random_number(bagged_seed, 1416147523)
            parallel_args.append(
                (
                    dataset,
                    bags[idx],
                    None,
                    feature_groups,
                    inner_bags,
                    update,
                    self.learning_rate,
                    self.min_samples_leaf,
                    self.max_leaves,
                    early_stopping_rounds,
                    early_stopping_tolerance,
                    self.max_rounds,
                    noise_scale,
                    bin_data_weights,
                    bagged_seed,
                    None,
                )
            )

        gc.collect() # clean up before starting/forking new processes
        results = provider.parallel(EBMUtils.cyclic_gradient_boost, parallel_args)

        # let the garbage collector claim the dataset
        del parallel_args # parallel_args holds referecnes to dataset, so must be deleted
        del dataset
        gc.collect()

        breakpoint_iteration = [[]]
        models = []
        for model, bag_breakpoint_iteration in results:
            breakpoint_iteration[-1].append(bag_breakpoint_iteration)
            models.append(after_boosting(feature_groups, model, term_bin_weights))

        interactions = 0 if is_private(self) else self.interactions
        if n_classes > 2:
            if isinstance(interactions, int):
               if interactions != 0:
                    warn("Detected multiclass problem: forcing interactions to 0")
                    interactions = 0
            elif len(interactions) != 0:
                raise ValueError("interactions are not supported for multiclass")

        if isinstance(interactions, int) and 0 < interactions or not isinstance(interactions, int) and 0 < len(interactions):
            initial_intercept = np.zeros(Native.get_count_scores_c(n_classes), np.float64)

            scores_bags = []
            for model in models:
                # TODO: instead of going back to the original data in X, we 
                # could use the compressed and already binned data in dataset
                scores_bags.append(ebm_decision_function(
                    X, 
                    n_samples, 
                    feature_names_in, 
                    feature_types_in, 
                    bins, 
                    initial_intercept, 
                    model, 
                    feature_groups
                ))

            dataset = bin_native_by_dimension(
                n_classes, 
                2,
                bins,
                X, 
                y, 
                sample_weight, 
                feature_names_in, 
                feature_types_in, 
            )
            del y # we no longer need this, so allow the garbage collector to reclaim it

            if isinstance(interactions, int):
                _log.info("Estimating with FAST")

                parallel_args = []
                for idx in range(self.outer_bags):
                    parallel_args.append(
                        (
                            dataset,
                            bags[idx],
                            scores_bags[idx],
                            combinations(range(n_features_in), 2),
                            self.min_samples_leaf,
                            None,
                        )
                    )

                # TODO: for now we're using only 1 job because FAST isn't memory optimized.  After
                # the native code is done with compression of the data we can go back to using self.n_jobs
                provider2 = JobLibProvider(n_jobs=1) 
                gc.collect() # clean up before starting/forking new processes
                bagged_interaction_indices = provider2.parallel(EBMUtils.get_interactions, parallel_args)

                del parallel_args # this holds references to dataset, bags, and scores_bags which we want to gc later

                # Select merged pairs
                pair_ranks = {}
                for n, interaction_indices in enumerate(bagged_interaction_indices):
                    for rank, indices in enumerate(interaction_indices):
                        old_mean = pair_ranks.get(indices, 0)
                        pair_ranks[indices] = old_mean + ((rank - old_mean) / (n + 1))

                final_ranks = []
                total_interactions = 0
                for indices in pair_ranks:
                    heapq.heappush(final_ranks, (pair_ranks[indices], indices))
                    total_interactions += 1

                n_interactions = min(interactions, total_interactions)
                boost_groups = [heapq.heappop(final_ranks)[1] for _ in range(n_interactions)]
            else:
                # Check and remove duplicate interaction terms
                uniquifier = set()
                boost_groups = []
                max_dimensions = 0

                for feature_group in interactions:
                    # clean these up since we expose them publically inside self.feature_groups_ 
                    feature_group = tuple([int(feature_idx) for feature_idx in feature_group])

                    max_dimensions = max(max_dimensions, len(feature_group))
                    sorted_tuple = tuple(sorted(feature_group))
                    if sorted_tuple not in uniquifier:
                        uniquifier.add(sorted_tuple)
                        boost_groups.append(feature_group)

                # Warn the users that we have made change to the interactions list
                if len(boost_groups) != len(interactions):
                    warn("Detected duplicate interaction terms: removing duplicate interaction terms")

                if 2 < max_dimensions:
                    warn("Interactions with 3 or more terms are not graphed in global explanations. Local explanations are still available and exact.")


            parallel_args = []
            bagged_seed = init_seed
            for idx in range(self.outer_bags):
                bagged_seed=native.generate_random_number(bagged_seed, 1416147523)
                parallel_args.append(
                    (
                        dataset,
                        bags[idx],
                        scores_bags[idx],
                        boost_groups,
                        inner_bags,
                        update,
                        self.learning_rate,
                        self.min_samples_leaf,
                        self.max_leaves,
                        early_stopping_rounds,
                        early_stopping_tolerance,
                        self.max_rounds,
                        noise_scale,
                        bin_data_weights,
                        bagged_seed,
                        None,
                    )
                )

            gc.collect() # clean up before starting/forking new processes
            results = provider.parallel(EBMUtils.cyclic_gradient_boost, parallel_args)

            # allow the garbage collector to reclaim these big memory items
            del parallel_args # this holds references to dataset, scores_bags, and bags
            del dataset
            del scores_bags
            del bags
            gc.collect()

            breakpoint_iteration.append([])
            for idx in range(self.outer_bags):
                breakpoint_iteration[-1].append(results[idx][1])
                models[idx].extend(after_boosting(boost_groups, results[idx][0], term_bin_weights))

            feature_groups.extend(boost_groups)

        breakpoint_iteration = np.array(breakpoint_iteration, np.int64)

        bagged_additive_terms = (np.array([model[idx] for model in models]) for idx in range(len(feature_groups)))

        keys = ([len(feature_group)] + sorted(feature_group) for feature_group in feature_groups)
        sorted_items = sorted(zip(keys, feature_groups, bagged_additive_terms))
        feature_groups = [x[1] for x in sorted_items]
        bagged_additive_terms = [x[2] for x in sorted_items]

        if is_private(self):
            # for now we only support mains for DP models
            bin_weights = [term_bin_weights[feature_group[0]] for feature_group in feature_groups]
        else:
            bin_counts, bin_weights = get_counts_and_weights(
                X, 
                n_samples,
                sample_weight, 
                feature_names_in, 
                feature_types_in, 
                bins, 
                feature_groups
            )

        additive_terms, term_standard_deviations, intercept = process_terms(
            n_classes, 
            n_samples, 
            bagged_additive_terms, 
            bin_weights
        )

        if is_private(self):
            self.noise_scale_ = noise_scale
        else:
            # differentially private models would need to pay additional privacy budget to make
            # these public, but they are non-essential so we don't disclose them in the DP setting

            self.n_samples_ = n_samples

            # per-feature
            self.histogram_cuts_ = histogram_cuts
            self.histogram_counts_ = histogram_counts
            self.unique_counts_ = unique_counts
            self.zero_counts_ = zero_counts

            # per-feature group
            self.bin_counts_ = bin_counts # use bin_weights_ instead for DP models
        
        if 0 <= n_classes:
            self.classes_ = classes # required by scikit-learn
            self._class_idx_ = class_idx
        else:
            self.min_target_ = min_target
            self.max_target_ = max_target

        # per-feature
        self.n_features_in_ = n_features_in # required by scikit-learn
        self.feature_names_in_ = feature_names_in # scikit-learn specified name
        self.feature_types_in_ = feature_types_in
        self.bins_ = bins
        self.min_vals_ = min_vals
        self.max_vals_ = max_vals

        # per-feature group
        self.feature_groups_ = feature_groups
        self.bin_weights_ = bin_weights
        self.bagged_additive_terms_ = bagged_additive_terms
        self.additive_terms_ = additive_terms
        self.term_standard_deviations_ = term_standard_deviations

        # general
        self.intercept_ = intercept
        self.breakpoint_iteration_ = breakpoint_iteration
        self.has_fitted_ = True
        return self

    def _to_json(self):
        j = {}

        if is_classifier(self):
            j['model_type'] = "classification"
            j['classes'] = self.classes_.tolist()
        else:
            j['model_type'] = "regression"
            if hasattr(self, 'min_target_') and not math.isnan(self.min_target_):
                j['min_target'] = EBMUtils.jsonify_item(self.min_target_)
            if hasattr(self, 'max_target_') and not math.isnan(self.max_target_):
                j['max_target'] = EBMUtils.jsonify_item(self.max_target_)

        features = []
        for i in range(self.n_features_in_):
            feature = {}

            feature['name'] = self.feature_names_in_[i]
            feature_type = self.feature_types_in_[i]
            feature['type'] = feature_type

            if feature_type == 'continuous':
                cuts = []
                for bins in self.bins_[i]:
                    cuts.append(bins.tolist())
                feature['cuts'] = cuts
                if hasattr(self, 'min_vals_') and not math.isnan(self.min_vals_[i]):
                    feature['min'] = EBMUtils.jsonify_item(self.min_vals_[i])
                if hasattr(self, 'max_vals_') and not math.isnan(self.max_vals_[i]):
                    feature['max'] = EBMUtils.jsonify_item(self.max_vals_[i])
                if hasattr(self, 'histogram_cuts_') and self.histogram_cuts_[i] is not None:
                    feature['histogram_cuts'] = self.histogram_cuts_[i].tolist()
                if hasattr(self, 'histogram_counts_') and self.histogram_counts_[i] is not None:
                    feature['histogram_counts'] = self.histogram_counts_[i].tolist()
            else:
                categories = []
                for bins in self.bins_[i]:
                    leveled_categories = []
                    feature_categories = list(map(tuple, map(reversed, bins.items())))
                    feature_categories.sort() # groupby requires sorted data
                    for _, category_iter in groupby(feature_categories, lambda x: x[0]):
                        category_group = [category for _, category in category_iter]
                        if len(category_group) == 1:
                            leveled_categories.append(category_group[0])
                        else:
                            leveled_categories.append(category_group)
                    categories.append(leveled_categories)
                feature['categories'] = categories

            features.append(feature)
        j['features'] = features

        j['intercept'] = EBMUtils.jsonify_item(self.intercept_) if type(self.intercept_) is float else EBMUtils.jsonify_lists(self.intercept_.tolist())

        terms = []
        for i in range(len(self.feature_groups_)):
            term = {}
            term['features'] = [self.feature_names_in_[feature_idx] for feature_idx in self.feature_groups_[i]]
            term['scores'] = EBMUtils.jsonify_lists(self.additive_terms_[i].tolist())
            if hasattr(self, 'bin_counts_') and self.bin_counts_[i] is not None:
                term['bin_counts'] = self.bin_counts_[i].tolist()
            term['bin_weights'] = EBMUtils.jsonify_lists(self.bin_weights_[i].tolist())
            
            terms.append(term)
        j['terms'] = terms

        return json.dumps(j, allow_nan=False, indent=2)

    def decision_function(self, X):
        """ Predict scores from model before calling the link function.

            Args:
                X: Numpy array for samples.

            Returns:
                The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        return ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
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

        check_is_fitted(self, "has_fitted_")

        mod_counts = remove_last2(getattr(self, 'bin_counts_', self.bin_weights_), self.bin_weights_)
        mod_additive_terms = remove_last2(self.additive_terms_, self.bin_weights_)
        mod_term_standard_deviations = remove_last2(self.term_standard_deviations_, self.bin_weights_)
        for feature_group_idx, feature_group in enumerate(self.feature_groups_):
            mod_additive_terms[feature_group_idx] = trim_tensor(mod_additive_terms[feature_group_idx], trim_low=[True] * len(feature_group))
            mod_term_standard_deviations[feature_group_idx] = trim_tensor(mod_term_standard_deviations[feature_group_idx], trim_low=[True] * len(feature_group))
            mod_counts[feature_group_idx] = trim_tensor(mod_counts[feature_group_idx], trim_low=[True] * len(feature_group))

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_group_index, _ in enumerate(self.feature_groups_):
            errors = mod_term_standard_deviations[feature_group_index]
            scores = mod_additive_terms[feature_group_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        native = Native.get_native_singleton()

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        keep_idxs = []
        for feature_group_index, feature_indexes in enumerate(
            self.feature_groups_
        ):
            model_graph = mod_additive_terms[feature_group_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = mod_term_standard_deviations[feature_group_index]

            if len(feature_indexes) == 1:
                keep_idxs.append(feature_group_index)

                feature_index0 = feature_indexes[0]

                feature_bins = self.bins_[feature_index0][0]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append('DPOther')

                    names=bin_labels
                    densities = list(mod_counts[feature_group_index])
                else:
                    # continuous
                    min_val = self.min_vals_[feature_index0]
                    max_val = self.max_vals_[feature_index0]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_val, max_val = native.suggest_graph_bounds(feature_bins, min_val=min_val, max_val=max_val)

                    bin_labels = list(np.concatenate(([min_val], feature_bins, [max_val])))

                    if hasattr(self, 'histogram_cuts_') and hasattr(self, 'histogram_counts_'):
                        names = self.histogram_cuts_[feature_index0]
                        densities = list(self.histogram_counts_[feature_index0][1:-1])
                    else:
                        names = feature_bins
                        densities = list(mod_counts[feature_group_index])

                    names = list(np.concatenate(([min_val], names, [max_val])))

                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": names,
                    "scores": densities,
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
                        "names": names,
                        "scores": densities,
                    },
                }
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                    }

                data_dicts.append(data_dict)
            elif len(feature_indexes) == 2:
                keep_idxs.append(feature_group_index)

                bin_levels = self.bins_[feature_indexes[0]]
                feature_bins = bin_levels[1] if 1 < len(bin_levels) else bin_levels[0]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append('DPOther')
                else:
                    # continuous
                    min_val = self.min_vals_[feature_indexes[0]]
                    max_val = self.max_vals_[feature_indexes[0]]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_val, max_val = native.suggest_graph_bounds(feature_bins, min_val=min_val, max_val=max_val)

                    bin_labels = list(np.concatenate(([min_val], feature_bins, [max_val])))
                bin_labels_left = bin_labels


                bin_levels = self.bins_[feature_indexes[1]]
                feature_bins = bin_levels[1] if 1 < len(bin_levels) else bin_levels[0]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[1]:
                        bin_labels.append('DPOther')
                else:
                    # continuous
                    min_val = self.min_vals_[feature_indexes[1]]
                    max_val = self.max_vals_[feature_indexes[1]]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_val, max_val = native.suggest_graph_bounds(feature_bins, min_val=min_val, max_val=max_val)

                    bin_labels = list(np.concatenate(([min_val], feature_bins, [max_val])))
                bin_labels_right = bin_labels


                feature_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                data_dicts.append(data_dict)
            else:  # pragma: no cover
                warn(f"Dropping feature {self.term_names_[feature_group_index]} from explanation since we can't graph more than 2 dimensions.")

        overall_dict = {
            "type": "univariate",
            "names": [self.term_names_[i] for i in keep_idxs],
            "scores": [self.feature_importances_[i] for i in keep_idxs],
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
            feature_names=[self.term_names_[i] for i in keep_idxs],
            feature_types=['categorical' if x == 'nominal' or x == 'ordinal' else x for x in [self.term_types_[i] for i in keep_idxs]],
            name=name,
            selector=gen_global_selector2(getattr(self, 'n_samples_', None), self.n_features_in_, [self.term_names_[i] for i in keep_idxs], ['categorical' if x == 'nominal' or x == 'ordinal' else x for x in [self.term_types_[i] for i in keep_idxs]], getattr(self, 'unique_counts_', None), getattr(self, 'zero_counts_', None)),
        )

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """

        # Produce feature value pairs for each sample.
        # Values are the model graph score per respective feature group.

        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        if y is not None:
            if is_classifier(self):
                y = clean_vector(y, True, "y")
                y = np.array([self._class_idx_[el] for el in y], dtype=np.int64)
            else:
                y = clean_vector(y, False, "y")

            if n_samples != len(y):
                msg = f"X has {n_samples} samples and y has {len(y)} samples"
                _log.error(msg)
                raise ValueError(msg)

        data_dicts = []
        perf_list = []
        if n_samples == 0:
            X_unified = np.empty((0, len(self.feature_names_in_)), dtype=np.object_)
        else:
            X_unified, _, _ = unify_data2(X, n_samples, self.feature_names_in_, self.feature_types_in_, True)

            intercept = self.intercept_
            if not is_classifier(self) or len(self.classes_) <= 2:
                if isinstance(intercept, np.ndarray) or isinstance(intercept, list):
                    intercept = intercept[0]

            for _ in range(n_samples):
                data_dict = {
                    "type": "univariate",
                    "names": [None] * len(self.feature_groups_),
                    "scores": [None] * len(self.feature_groups_),
                    "values": [None] * len(self.feature_groups_),
                    "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
                }
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                    }
                data_dicts.append(data_dict)

            term_names = self.term_names_
            for set_idx, binned_data in eval_terms(X, n_samples, self.feature_names_in_, self.feature_types_in_, self.bins_, self.feature_groups_):
                scores = self.additive_terms_[set_idx][tuple(binned_data)]
                feature_group = self.feature_groups_[set_idx]
                for row_idx in range(n_samples):
                    feature_name = term_names[set_idx]
                    data_dicts[row_idx]["names"][set_idx] = feature_name
                    data_dicts[row_idx]["scores"][set_idx] = scores[row_idx]
                    if len(feature_group) == 1:
                        data_dicts[row_idx]["values"][set_idx] = X_unified[row_idx, feature_group[0]]
                    else:
                        data_dicts[row_idx]["values"][set_idx] = ""

            scores = ebm_decision_function(
                X, 
                n_samples, 
                self.feature_names_in_, 
                self.feature_types_in_, 
                self.bins_, 
                self.intercept_, 
                self.additive_terms_, 
                self.feature_groups_
            )

            if is_classifier(self):
                # Handle binary classification case -- softmax only works with 0s appended
                if scores.ndim == 1:
                    scores = np.c_[np.zeros(scores.shape), scores]

                scores = softmax(scores)

            perf_dicts = gen_perf_dicts(scores, y, is_classifier(self))
            for row_idx in range(n_samples):
                perf = None if perf_dicts is None else perf_dicts[row_idx]
                perf_list.append(perf)
                data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classifier(self))

        additive_terms = remove_last2(self.additive_terms_, self.bin_weights_)
        for feature_group_idx, feature_group in enumerate(self.feature_groups_):
            additive_terms[feature_group_idx] = trim_tensor(additive_terms[feature_group_idx], trim_low=[True] * len(feature_group))

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": additive_terms,
                        "intercept": self.intercept_,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X_unified, "dataset_y": y},
            }
        )

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.term_names_,
            feature_types=['categorical' if x == 'nominal' or x == 'ordinal' else x for x in self.term_types_],
            name=gen_name_from_class(self) if name is None else name,
            selector=selector,
        )

    @property
    def feature_importances_(self):
        feature_importances = np.empty(len(self.feature_groups_), np.float64)
        for i in range(len(self.feature_groups_)):
            mean_abs_score = np.abs(self.additive_terms_[i])
            if is_classifier(self) and 2 < len(self.classes_):
                mean_abs_score = np.average(mean_abs_score, axis=mean_abs_score.ndim - 1)
            mean_abs_score = np.average(mean_abs_score, weights=self.bin_weights_[i])
            feature_importances.itemset(i, mean_abs_score)
        return feature_importances

    @property
    def term_names_(self):
        return [EBMUtils.gen_feature_group_name(feature_idxs, self.feature_names_in_) for feature_idxs in self.feature_groups_]

    @property
    def term_types_(self):
        return [EBMUtils.gen_feature_group_type(feature_idxs, self.feature_types_in_) for feature_idxs in self.feature_groups_]


class ExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog. """

    # TODO PK v.3 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        binning="quantile",
        # Stages
        mains="all",
        interactions=10,
        # Ensemble
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile" or "quantile_humanized".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
                Interactions are forcefully set to 0 for multiclass problems.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(ExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
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
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict_proba(self, X):
        """ Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        log_odds_vector = ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    def predict(self, X):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        log_odds_vector = ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return self.classes_[np.argmax(log_odds_vector, axis=1)]

    def predict_and_contrib(self, X, output='probabilities'):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.
            output: Prediction type to output (i.e. one of 'probabilities', 'logits', 'labels')

        Returns:
            Predictions and local explanations for each sample.
        """

        check_is_fitted(self, "has_fitted_")

        allowed_outputs = ['probabilities', 'logits', 'labels']
        if output not in allowed_outputs:
            msg = "Argument 'output' has invalid value.  Got '{}', expected one of " 
            + repr(allowed_outputs)
            raise ValueError(msg.format(output))

        X, n_samples = clean_X(X)

        scores, explanations = ebm_decision_function_and_explain(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

        if output == 'probabilities':
            if scores.ndim == 1:
                scores= np.c_[np.zeros(scores.shape), scores]
            result = softmax(scores)
        elif output == 'labels':
            if scores.ndim == 1:
                scores = np.c_[np.zeros(scores.shape), scores]
            result = self.classes_[np.argmax(scores, axis=1)]
        else:
            result = scores

        return result, explanations

class ExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    """ Explainable Boosting Regressor. The arguments will change in a future release, watch the changelog. """

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        binning="quantile",
        # Stages
        mains="all",
        interactions=10,
        # Ensemble
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Regressor. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage on main effects.
            max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile", or "quantile_humanized".
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
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(ExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
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
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict(self, X):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        return ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

    def predict_and_contrib(self, X):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.

        Returns:
            Predictions and local explanations for each sample.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        return ebm_decision_function_and_explain(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )


class DPExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    """ Differentially Private Explainable Boosting Classifier."""

    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing DPEBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=32,
        binning="private",
        # Stages
        mains="all",
        # Ensemble
        outer_bags=1,
        # Boosting
        learning_rate=0.01,
        validation_size=0,
        max_rounds=300,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Differential Privacy
        epsilon=1,
        delta=1e-5,
        composition='gdp',
        bin_budget_frac=0.1,
        privacy_schema=None,
    ):
        """ Differentially Private Explainable Boosting Classifier. Note that many arguments are defaulted differently than regular EBMs.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            binning: Method to bin values for pre-processing. Choose "uniform" or "quantile".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            outer_bags: Number of outer bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            max_rounds: Number of rounds for boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
            epsilon: Total privacy budget to be spent across all rounds of training.
            delta: Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
            composition: composition.
            bin_budget_frac: Percentage of total epsilon budget to use for binning.
            privacy_schema: Dictionary specifying known min/max values of each feature and target. 
                If None, DP-EBM throws warning and uses data to calculate these values.
        """
        super(DPExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,    
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=None,
            binning=binning,
            # Stages
            mains=mains,
            interactions=0,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=0,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=-1,
            early_stopping_tolerance=-1,
            max_rounds=max_rounds,
            # Trees
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
            # Differential Privacy
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_schema=privacy_schema,
        )

    def predict_proba(self, X):
        """ Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        log_odds_vector = ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return softmax(log_odds_vector)

    def predict(self, X):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        log_odds_vector = ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return self.classes_[np.argmax(log_odds_vector, axis=1)]


class DPExplainableBoostingRegressor(BaseEBM, RegressorMixin, ExplainerMixin):
    """ Differentially Private Explainable Boosting Regressor."""

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing DPEBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=32,
        binning="private",
        # Stages
        mains="all",
        # Ensemble
        outer_bags=1,
        # Boosting
        learning_rate=0.01,
        validation_size=0,
        max_rounds=300,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Differential Privacy
        epsilon=1,
        delta=1e-5,
        composition='gdp',
        bin_budget_frac=0.1,
        privacy_schema=None,
    ):
        """ Differentially Private Explainable Boosting Regressor. Note that many arguments are defaulted differently than regular EBMs.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            binning: Method to bin values for pre-processing. Choose "uniform" or "quantile".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            outer_bags: Number of outer bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            max_rounds: Number of rounds for boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
            epsilon: Total privacy budget to be spent across all rounds of training.
            delta: Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
            composition: Method of tracking noise aggregation. Must be one of 'classic' or 'gdp'. 
            bin_budget_frac: Percentage of total epsilon budget to use for private binning.
            privacy_schema: Dictionary specifying known min/max values of each feature and target. 
                If None, DP-EBM throws warning and uses data to calculate these values.
        """
        super(DPExplainableBoostingRegressor, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=None,
            binning=binning,
            # Stages
            mains=mains,
            interactions=0,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=0,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=-1,
            early_stopping_tolerance=-1,
            max_rounds=max_rounds,
            # Trees
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
            # Differential Privacy
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_schema=privacy_schema,
        )

    def predict(self, X):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        X, n_samples = clean_X(X)

        return ebm_decision_function(
            X, 
            n_samples, 
            self.feature_names_in_, 
            self.feature_types_in_, 
            self.bins_, 
            self.intercept_, 
            self.additive_terms_, 
            self.feature_groups_
        )

