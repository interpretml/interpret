# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from typing import DefaultDict

from interpret.provider.visualize import PreserveProvider
from ...utils import gen_perf_dicts
from .utils import DPUtils, EBMUtils
from .utils import _process_terms, make_histogram_edges, _order_terms, _remove_unused_higher_bins, _deduplicate_bins, _generate_term_names, _generate_term_types
from .bin import clean_X, clean_vector, construct_bins, bin_native_by_dimension, ebm_decision_function, ebm_decision_function_and_explain, make_boosting_weights, after_boosting, remove_last2, get_counts_and_weights, trim_tensor, unify_data2, eval_terms
from .internal import Native
from ...utils import unify_data, autogen_schema, unify_vector
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...provider.compute import JobLibProvider
from ...utils import gen_name_from_class, gen_global_selector, gen_global_selector2, gen_local_selector

import json
from math import isnan

import numpy as np
from warnings import warn

from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import log_loss, mean_squared_error
import heapq
import operator

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

class EBMModel(BaseEstimator):
    """Base class for all EBMs"""

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

            if random_state is not None:
                warn(f"Privacy violation: using a fixed random_state of {random_state} will cause deterministic noise additions."
                        "This capability is only for debugging/testing. Set random_state to None to remove this warning.")

    def fit(self, X, y, sample_weight=None):  # noqa: C901
        """ Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.

        Returns:
            Itself.
        """

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
        is_differential_privacy = is_private(self)
        if is_differential_privacy:
            DPUtils.validate_eps_delta(self.epsilon, self.delta)

            if is_classifier(self):
                if n_classes > 2:  # pragma: no cover
                    raise ValueError("multiclass not supported in Differentially private EBMs.")
            else:
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
            bin_eps = self.epsilon * self.bin_budget_frac
            bin_delta = self.delta / 2
            composition = self.composition
            privacy_schema = self.privacy_schema

            bin_levels = [self.max_bins]
        else:
            bin_eps = None
            bin_delta = None
            composition = None
            privacy_schema = None

            bin_levels = [self.max_bins, self.max_interaction_bins]

        init_random_state = EBMUtils.normalize_initial_seed(self.random_state)

        binning_result = construct_bins(
            X=X,
            sample_weight=sample_weight,
            feature_names_given=self.feature_names, 
            feature_types_given=self.feature_types, 
            max_bins_leveled=bin_levels, 
            binning=self.binning, 
            min_samples_bin=1, 
            min_unique_continuous=3, 
            epsilon=bin_eps, 
            delta=bin_delta, 
            composition=composition,
            privacy_schema=privacy_schema,
            random_state=init_random_state,
        )
        feature_names_in = binning_result[0]
        feature_types_in = binning_result[1]
        bins = binning_result[2]
        main_bin_weights = binning_result[3]
        feature_bounds = binning_result[4]
        histogram_counts = binning_result[5]
        unique_val_counts = binning_result[6]
        zero_val_counts = binning_result[7]

        n_features_in = len(bins)

        if isinstance(self.mains, str) and self.mains == "all":
            term_features = [(x,) for x in range(n_features_in)]
        else:
            term_features = [(int(x),) for x in self.mains]

        if is_differential_privacy:
            # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            domain_size = 1 if is_classifier(self) else max_target - min_target
            max_weight = 1 if sample_weight is None else np.max(sample_weight)
            training_eps = self.epsilon - bin_eps
            training_delta = self.delta / 2
            if self.composition == 'classic':
                noise_scale = DPUtils.calc_classic_noise_multi(
                    total_queries = self.max_rounds * len(term_features) * self.outer_bags, 
                    target_epsilon = training_eps, 
                    delta = training_delta, 
                    sensitivity = domain_size * self.learning_rate * max_weight
                )
            elif self.composition == 'gdp':
                noise_scale = DPUtils.calc_gdp_noise_multi(
                    total_queries = self.max_rounds * len(term_features) * self.outer_bags, 
                    target_epsilon = training_eps, 
                    delta = training_delta
                )
                noise_scale *= domain_size * self.learning_rate * max_weight # Alg Line 17
            else:
                raise NotImplementedError(f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'.")

            bin_data_weights = make_boosting_weights(main_bin_weights)
            boosting_flags = Native.GenerateUpdateOptions_GradientSums | Native.GenerateUpdateOptions_RandomSplits
            inner_bags = 0
            early_stopping_rounds = -1
            early_stopping_tolerance = -1
            interactions = 0
        else:
            noise_scale = None
            bin_data_weights = None
            boosting_flags = Native.GenerateUpdateOptions_Default
            inner_bags = self.inner_bags
            early_stopping_rounds = self.early_stopping_rounds
            early_stopping_tolerance = self.early_stopping_tolerance
            interactions = self.interactions

        native = Native.get_native_singleton()
        bagged_random_state = init_random_state
        bag_weights = []
        bags = []
        for _ in range(self.outer_bags):
            bagged_random_state = native.generate_deterministic_seed(bagged_random_state, 886321150)
            bag = EBMUtils.make_bag(
                y,
                self.validation_size,
                bagged_random_state,
                is_classifier(self) and not is_differential_privacy
            )
            bags.append(bag)
            if bag is None:
                if sample_weight is None:
                    bag_weights.append(n_samples)
                else:
                    bag_weights.append(sample_weight.sum())
            else:
                keep = 0 < bag
                if sample_weight is None:
                    bag_weights.append(bag[keep].sum())
                else:
                    bag_weights.append((bag[keep] * sample_weight[keep]).sum())
        bag_weights = np.array(bag_weights, np.float64)

        provider = JobLibProvider(n_jobs=self.n_jobs)

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

        bagged_random_state = init_random_state
        parallel_args = []
        for idx in range(self.outer_bags):
            bagged_random_state = native.generate_deterministic_seed(bagged_random_state, 13098686)
            parallel_args.append(
                (
                    dataset,
                    bags[idx],
                    None,
                    term_features,
                    inner_bags,
                    boosting_flags,
                    self.learning_rate,
                    self.min_samples_leaf,
                    self.max_leaves,
                    early_stopping_rounds,
                    early_stopping_tolerance,
                    self.max_rounds,
                    noise_scale,
                    bin_data_weights,
                    bagged_random_state,
                    None,
                )
            )

        results = provider.parallel(EBMUtils.cyclic_gradient_boost, parallel_args)

        # let python reclaim the dataset memory via reference counting
        del parallel_args # parallel_args holds references to dataset, so must be deleted
        del dataset

        breakpoint_iteration = [[]]
        models = []
        for model, bag_breakpoint_iteration in results:
            breakpoint_iteration[-1].append(bag_breakpoint_iteration)
            models.append(after_boosting(term_features, model, main_bin_weights))

        if n_classes > 2:
            if isinstance(interactions, int):
                if interactions != 0:
                    warn("Detected multiclass problem. Forcing interactions to 0. Multiclass interactions work except for global visualizations, so the line below setting interactions to zero can be disabled if you know what you are doing.")
                    interactions = 0
            elif len(interactions) != 0:
                raise ValueError("Interactions are not supported for multiclass. Multiclass interactions work except for global visualizations, so this exception can be disabled if you know what you are doing.")

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
                    term_features
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
                from ...utils.fast import _get_ranked_interactions

                parallel_args = []
                for idx in range(self.outer_bags):
                    # TODO: the combinations below should be selected from the non-excluded features 
                    parallel_args.append(
                        (
                            dataset,
                            bags[idx],
                            scores_bags[idx],
                            combinations(range(n_features_in), 2),
                            Native.InteractionOptions_Default, 
                            self.min_samples_leaf,
                            None,
                        )
                    )

                # TODO: for now we're using only 1 job because FAST isn't memory optimized.  After
                # the native code is done with compression of the data we can go back to using self.n_jobs
                provider2 = JobLibProvider(n_jobs=1) 
                bagged_ranked_interaction = provider2.parallel(_get_ranked_interactions, parallel_args)

                # this holds references to dataset, bags, and scores_bags which we want python to reclaim later
                del parallel_args 

                # Select merged pairs
                pair_ranks = {}
                for n, interaction_strengths_and_indices in enumerate(bagged_ranked_interaction):
                    interaction_indices =  list(map(operator.itemgetter(1), interaction_strengths_and_indices))
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

                for feature_idxs in interactions:
                    # clean these up since we expose them publically inside self.term_features_ 
                    feature_idxs = tuple(map(int, feature_idxs))

                    max_dimensions = max(max_dimensions, len(feature_idxs))
                    sorted_tuple = tuple(sorted(feature_idxs))
                    if sorted_tuple not in uniquifier:
                        uniquifier.add(sorted_tuple)
                        boost_groups.append(feature_idxs)

                # Warn the users that we have made change to the interactions list
                if len(boost_groups) != len(interactions):
                    warn("Detected duplicate interaction terms: removing duplicate interaction terms")

                if 2 < max_dimensions:
                    warn("Interactions with 3 or more terms are not graphed in global explanations. Local explanations are still available and exact.")


            bagged_random_state = init_random_state
            parallel_args = []
            for idx in range(self.outer_bags):
                bagged_random_state = native.generate_deterministic_seed(bagged_random_state, 521040308)
                parallel_args.append(
                    (
                        dataset,
                        bags[idx],
                        scores_bags[idx],
                        boost_groups,
                        inner_bags,
                        boosting_flags,
                        self.learning_rate,
                        self.min_samples_leaf,
                        self.max_leaves,
                        early_stopping_rounds,
                        early_stopping_tolerance,
                        self.max_rounds,
                        noise_scale,
                        bin_data_weights,
                        bagged_random_state,
                        None,
                    )
                )

            results = provider.parallel(EBMUtils.cyclic_gradient_boost, parallel_args)

            # allow python to reclaim these big memory items via reference counting
            del parallel_args # this holds references to dataset, scores_bags, and bags
            del dataset
            del scores_bags

            breakpoint_iteration.append([])
            for idx in range(self.outer_bags):
                breakpoint_iteration[-1].append(results[idx][1])
                models[idx].extend(after_boosting(boost_groups, results[idx][0], main_bin_weights))

            term_features.extend(boost_groups)

        breakpoint_iteration = np.array(breakpoint_iteration, np.int64)

        _remove_unused_higher_bins(term_features, bins)
        # removing the higher order terms might allow us to eliminate some extra bins now that couldn't before
        _deduplicate_bins(bins)

        bagged_scores = (np.array([model[idx] for model in models], np.float64) for idx in range(len(term_features)))

        term_features, bagged_scores = _order_terms(term_features, bagged_scores)

        if is_differential_privacy:
            # for now we only support mains for DP models
            bin_weights = [main_bin_weights[feature_idxs[0]] for feature_idxs in term_features]
        else:
            bin_counts, bin_weights = get_counts_and_weights(
                X, 
                n_samples,
                sample_weight, 
                feature_names_in, 
                feature_types_in, 
                bins, 
                term_features
            )

        term_scores, term_standard_deviations, intercept, bagged_scores = _process_terms(
            n_classes, 
            n_samples, 
            bagged_scores, 
            bin_weights,
            bag_weights
        )

        term_names = _generate_term_names(feature_names_in, term_features)

        # dependent attributes (can be re-derrived after serialization)
        self.n_features_in_ = n_features_in # scikit-learn specified name
        self.term_names_ = term_names

        if is_differential_privacy:
            self.noise_scale_ = noise_scale
        else:
            # differentially private models would need to pay additional privacy budget to make
            # these public, but they are non-essential so we don't disclose them in the DP setting

            self.n_samples_ = n_samples

            # per-feature
            self.histogram_counts_ = histogram_counts
            self.unique_val_counts_ = unique_val_counts
            self.zero_val_counts_ = zero_val_counts

            # per-term
            self.bin_counts_ = bin_counts # use bin_weights_ instead for DP models
        
        if 0 <= n_classes:
            self.classes_ = classes # required by scikit-learn
            self._class_idx_ = class_idx
        else:
            self.min_target_ = min_target
            self.max_target_ = max_target

        # per-feature
        self.bins_ = bins
        self.feature_names_in_ = feature_names_in # scikit-learn specified name
        self.feature_types_in_ = feature_types_in
        self.feature_bounds_ = feature_bounds

        # per-term
        self.term_features_ = term_features
        self.bin_weights_ = bin_weights
        self.bagged_scores_ = bagged_scores
        self.term_scores_ = term_scores
        self.term_standard_deviations_ = term_standard_deviations

        # general
        self.intercept_ = intercept
        self.bag_weights_ = bag_weights
        self.breakpoint_iteration_ = breakpoint_iteration
        self.has_fitted_ = True

        return self

    def _to_inner_jsonable(self, properties='interpretable'):
        """ Converts the inner model to a JSONable representation.

        Args:
            properties: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSONable object
        """

        if properties == 'minimal':
            level = 0
        elif properties == 'interpretable':
            level = 1
        elif properties == 'mergeable':
            level = 2
        elif properties == 'all':
            level = 3
        else:
            msg = f"Unrecognized export properties: {properties}"
            _log.error(msg)
            raise ValueError(msg)

        j = {}

        # future-proof support for multi-output models
        outputs = []
        output = {}
        if is_classifier(self):
            if len(self.classes_) <= 2:
                # include 1 class classification in the binary classification category and use -inf in the intercept
                output['output_type'] = 'binary'
            else:
                # distinquish from binary classification so that we can support 'ordinal' classification someday
                # https://en.wikipedia.org/wiki/Ordinal_regression
                output['output_type'] = 'multinomial'
            output['classes'] = self.classes_.tolist()
            output['link_function'] = 'logit' # logistic is the inverse link function for logit
        else:
            output['output_type'] = 'regression'
            if 3 <= level:
                min_target = getattr(self, 'min_target_', None)
                if min_target is not None and not isnan(min_target):
                    output['min_target'] = EBMUtils.jsonify_item(min_target)
                max_target = getattr(self, 'max_target_', None)
                if max_target is not None and not isnan(max_target):
                    output['max_target'] = EBMUtils.jsonify_item(max_target)
            output['link_function'] = 'identity'
        outputs.append(output)
        j['outputs'] = outputs

        if type(self.intercept_) is float:
            # scikit-learn requires that we have a single float value as our intercept for compatibility with 
            # RegressorMixin, but in other scenarios where we want to support things like mulit-output it would be 
            # easier if the regression intercept were handled identically to classification, so put it in an array
            # for our JSON format to harmonize the cross-language representation
            j['intercept'] = [EBMUtils.jsonify_item(self.intercept_)]
        else:
            j['intercept'] = EBMUtils.jsonify_lists(self.intercept_.tolist())

        if 3 <= level:
            noise_scale = getattr(self, 'noise_scale_', None)
            if noise_scale is not None:
                j['noise_scale'] = EBMUtils.jsonify_item(noise_scale)
        if 1 <= level:
            n_samples = getattr(self, 'n_samples_', None)
            if n_samples is not None:
                j['num_samples'] = n_samples
        if 2 <= level:
            bag_weights = getattr(self, 'bag_weights_', None)
            if bag_weights is not None:
                j['bag_weights'] = EBMUtils.jsonify_lists(bag_weights.tolist())
        if 3 <= level:
            breakpoint_iteration = getattr(self, 'breakpoint_iteration_', None)
            if breakpoint_iteration is not None:
                j['breakpoint_iteration'] = breakpoint_iteration.tolist()

        if 3 <= level:
            j['implementation'] = 'python'
            params = {}

            # TODO: we need to clean up and validate our input parameters before putting them into JSON
            # if we were pass a numpy array instead of a list or a numpy type these would fail
            # for now we can just require that anything numpy as input is illegal

            if hasattr(self, 'feature_names'):
                params['feature_names'] = self.feature_names

            if hasattr(self, 'feature_types'):
                params['feature_types'] = self.feature_types

            if hasattr(self, 'outer_bags'):
                params['outer_bags'] = self.outer_bags

            if hasattr(self, 'inner_bags'):
                params['inner_bags'] = self.inner_bags

            if hasattr(self, 'mains'):
                params['mains'] = self.mains

            if hasattr(self, 'interactions'):
                params['interactions'] = self.interactions

            if hasattr(self, 'validation_size'):
                params['validation_size'] = self.validation_size

            if hasattr(self, 'max_rounds'):
                params['max_rounds'] = self.max_rounds

            if hasattr(self, 'early_stopping_tolerance'):
                params['early_stopping_tolerance'] = self.early_stopping_tolerance

            if hasattr(self, 'early_stopping_rounds'):
                params['early_stopping_rounds'] = self.early_stopping_rounds

            if hasattr(self, 'learning_rate'):
                params['learning_rate'] = self.learning_rate

            if hasattr(self, 'min_samples_leaf'):
                params['min_samples_leaf'] = self.min_samples_leaf

            if hasattr(self, 'max_leaves'):
                params['max_leaves'] = self.max_leaves

            if hasattr(self, 'n_jobs'):
                params['n_jobs'] = self.n_jobs

            if hasattr(self, 'random_state'):
                params['random_state'] = self.random_state

            if hasattr(self, 'binning'):
                params['binning'] = self.binning

            if hasattr(self, 'max_bins'):
                params['max_bins'] = self.max_bins

            if hasattr(self, 'max_interaction_bins'):
                params['max_interaction_bins'] = self.max_interaction_bins

            if hasattr(self, 'epsilon'):
                params['epsilon'] = self.epsilon

            if hasattr(self, 'delta'):
                params['delta'] = self.delta

            if hasattr(self, 'composition'):
                params['composition'] = self.composition

            if hasattr(self, 'bin_budget_frac'):
                params['bin_budget_frac'] = self.bin_budget_frac

            if hasattr(self, 'privacy_schema'):
                params['privacy_schema'] = self.privacy_schema

            j['implementation_params'] = params

        unique_val_counts = getattr(self, 'unique_val_counts_', None)
        zero_val_counts = getattr(self, 'zero_val_counts_', None)
        feature_bounds = getattr(self, 'feature_bounds_', None)
        histogram_counts = getattr(self, 'histogram_counts_', None)

        features = []
        for i in range(len(self.bins_)):
            feature = {}

            feature['name'] = self.feature_names_in_[i]
            feature_type = self.feature_types_in_[i]
            feature['type'] = feature_type

            if 1 <= level:
                if unique_val_counts is not None:
                    feature['num_unique_vals'] = int(unique_val_counts[i])
                if zero_val_counts is not None:
                    feature['num_zero_vals'] = int(zero_val_counts[i])

            if feature_type == 'continuous':
                cuts = []
                for bins in self.bins_[i]:
                    cuts.append(bins.tolist())
                feature['cuts'] = cuts
                if 1 <= level:
                    if feature_bounds is not None:
                        feature_min = feature_bounds[i, 0]
                        if not isnan(feature_min):
                            feature['min'] = EBMUtils.jsonify_item(feature_min)
                        feature_max = feature_bounds[i, 1]
                        if not isnan(feature_max):
                            feature['max'] = EBMUtils.jsonify_item(feature_max)
                    if histogram_counts is not None:
                        feature_histogram_counts = histogram_counts[i]
                        if feature_histogram_counts is not None:
                            feature['histogram_counts'] = feature_histogram_counts.tolist()
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

        standard_deviations_all = getattr(self, 'term_standard_deviations_', None)
        bagged_scores_all = getattr(self, 'bagged_scores_', None)
        bin_counts_all = getattr(self, 'bin_counts_', None)

        terms = []
        for term_idx in range(len(self.term_features_)):
            term = {}
            term['term_features'] = [self.feature_names_in_[feature_idx] for feature_idx in self.term_features_[term_idx]]
            term['scores'] = EBMUtils.jsonify_lists(self.term_scores_[term_idx].tolist())
            if 1 <= level:
                if standard_deviations_all is not None:
                   standard_deviations = standard_deviations_all[term_idx] 
                   if standard_deviations is not None:
                        term['standard_deviations'] = EBMUtils.jsonify_lists(standard_deviations.tolist())
            if 2 <= level:
                if bagged_scores_all is not None:
                   bagged_scores = bagged_scores_all[term_idx] 
                   if bagged_scores is not None:
                        term['bagged_scores'] = EBMUtils.jsonify_lists(bagged_scores.tolist())
            if 3 <= level:
                if bin_counts_all is not None:
                   bin_counts = bin_counts_all[term_idx] 
                   if bin_counts is not None:
                        term['bin_counts'] = bin_counts.tolist()
            if 1 <= level:
                term['bin_weights'] = EBMUtils.jsonify_lists(self.bin_weights_[term_idx].tolist())
            
            terms.append(term)
        j['terms'] = terms

        return j

    def _to_outer_jsonable(self, properties='interpretable'):
        """ Converts the outer model to a JSONable representation.

        Args:
            properties: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSONable object
        """

        # NOTES: When recording edits to the EBM within a single file, we should:
        #        1) Have the final EBM section first.  This allows people to diff two models and the diffs for 
        #           the current model (the most important information) will be at the top. If people are comparing a 
        #           non-edited model to an edited model then they will be comparing the non-edited model to the 
        #           current model, which is what we want. When people open the file they'll see the current model, 
        #           which will confuse people less.
        #        2) Have the initial model LAST.  This will help separate the final and inital model spacially.
        #           Someone examining the models won't accidentlly stray as easily from the current model into the 
        #           initial model while examining them. This also helps prevent the diffing tool from getting
        #           confused and diffing parts of the final model with parts of the initial model if there are
        #           substantial changes. Two final models that have the same initial model should then have a large
        #           unmodified section at the bottom, which the diffing tool should easily identify and keep
        #           together as one block since diffing tools look for longest unmodified sections of text
        #        3) The edits in the MIDDLE, starting from the LAST edit to the FIRST edit chronologically.
        #           If two models are derrived from the same initial model, then they will share a common initial
        #           block of text at the bottom of the file. If the two models share a few edits, then the shared edits
        #           will be at the bottom and will therefore form a larger block of unmodified text along with the
        #           initial model.  Since diff tools look for longest unmodified blocks, this will gobble up the initial 
        #           model and the initial edits together first, and thus leave the final models for comparison with
        #           eachother. All edits should have a bi-directional nature so someone could start
        #           from the final model and work backwards to the initial model, or vice versa. The overall file
        #           can then be viewed as a reverse chronological ordering from the final model back to its
        #           original/initial model.
        # - A non-edited EBM file should be saved with just the single JSON for the model and not an initial and 
        #   final model.  The only section should be marked with the tag "ebm" so that tools that read in EBMs
        #   Are compatible with both editied and non-edited files.  The tools will always look for the "ebm"
        #   section, which will be in both non-edited EBMs and edited EBMs at the top.
        # - The file would look like this for an edited EBMs:
        #   {
        #     "version": "1.0"
        #     "ebm": { FINAL_EBM_JSON }
        #     "edits": [
        #       { NEWEST_EDIT_JSON },
        #       { MID_EDITs_JSON },
        #       { OLDEST_EDIT_JSON }
        #     ]
        #     "initial_ebm": { INITIAL_EBM_JSON }
        #   }
        # - The file would look like this for an unedited EBMs:
        #   {
        #     "version": "1.0"
        #     "ebm": { EBM_JSON }
        #   }
        # - In python, we could contain these in attributes called "initial_ebm" which would contain a fully formed ebm
        #   and "edits", which would contain a list of the edits.  These fields wouldn't be present in a scikit-learn
        #   generated EBM, but would appear if the user edited the EBM, or if they loaded one that had edits.

        inner = self._to_inner_jsonable(properties)

        outer = {}
        outer['version'] = '1.0'
        outer['ebm'] = inner

        return outer

    def _to_json(self, properties='interpretable'):
        """ Converts the model to a JSON representation.

        Args:
            properties: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSON string
        """

        outer = self._to_outer_jsonable(properties)
        return json.dumps(outer, allow_nan=False, indent=2)

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
            self.term_scores_, 
            self.term_features_
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
        mod_term_scores = remove_last2(self.term_scores_, self.bin_weights_)
        mod_term_standard_deviations = remove_last2(self.term_standard_deviations_, self.bin_weights_)
        for term_idx, feature_idxs in enumerate(self.term_features_):
            mod_term_scores[term_idx] = trim_tensor(mod_term_scores[term_idx], trim_low=[True] * len(feature_idxs))
            mod_term_standard_deviations[term_idx] = trim_tensor(mod_term_standard_deviations[term_idx], trim_low=[True] * len(feature_idxs))
            mod_counts[term_idx] = trim_tensor(mod_counts[term_idx], trim_low=[True] * len(feature_idxs))

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for errors, scores in zip(mod_term_standard_deviations, mod_term_scores):
            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        term_names = self.term_names_
        term_types = _generate_term_types(self.feature_types_in_, self.term_features_)

        native = Native.get_native_singleton()

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        keep_idxs = []
        for term_idx, feature_idxs in enumerate(self.term_features_):
            model_graph = mod_term_scores[term_idx]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = mod_term_standard_deviations[term_idx]

            if len(feature_idxs) == 1:
                keep_idxs.append(term_idx)

                feature_index0 = feature_idxs[0]

                feature_bins = self.bins_[feature_index0][0]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append('DPOther')

                    names=bin_labels
                    densities = list(mod_counts[term_idx])
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, 'feature_bounds_', None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_index0, 0]
                        max_feature_val = feature_bounds[feature_index0, 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(feature_bins, min_feature_val, max_feature_val)
                    bin_labels = list(np.concatenate(([min_graph], feature_bins, [max_graph])))

                    histogram_edges = self.get_histogram_edges(feature_index0)
                    if histogram_edges is not None:
                        names = list(histogram_edges)
                        densities = list(self.histogram_counts_[feature_index0][1:-1])
                    else:
                        names = bin_labels
                        densities = list(mod_counts[term_idx])

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
            elif len(feature_idxs) == 2:
                keep_idxs.append(term_idx)

                bin_levels = self.bins_[feature_idxs[0]]
                feature_bins = bin_levels[min(len(feature_idxs), len(bin_levels)) - 1]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append('DPOther')
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, 'feature_bounds_', None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_idxs[0], 0]
                        max_feature_val = feature_bounds[feature_idxs[0], 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(feature_bins, min_feature_val, max_feature_val)
                    bin_labels = list(np.concatenate(([min_graph], feature_bins, [max_graph])))

                bin_labels_left = bin_labels

                bin_levels = self.bins_[feature_idxs[1]]
                feature_bins = bin_levels[min(len(feature_idxs), len(bin_levels)) - 1]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[1]:
                        bin_labels.append('DPOther')
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, 'feature_bounds_', None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_idxs[1], 0]
                        max_feature_val = feature_bounds[feature_idxs[1], 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(feature_bins, min_feature_val, max_feature_val)
                    bin_labels = list(np.concatenate(([min_graph], feature_bins, [max_graph])))

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
                warn(f"Dropping feature {term_names[term_idx]} from explanation since we can't graph more than 2 dimensions.")

        importances = self.get_importances()

        overall_dict = {
            "type": "univariate",
            "names": [term_names[i] for i in keep_idxs],
            "scores": [importances[i] for i in keep_idxs],
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
            feature_names=[term_names[i] for i in keep_idxs],
            feature_types=['categorical' if x == 'nominal' or x == 'ordinal' else x for x in [term_types[i] for i in keep_idxs]],
            name=name,
            selector=gen_global_selector2(getattr(self, 'n_samples_', None), self.n_features_in_, [term_names[i] for i in keep_idxs], ['categorical' if x == 'nominal' or x == 'ordinal' else x for x in [term_types[i] for i in keep_idxs]], getattr(self, 'unique_val_counts_', None), getattr(self, 'zero_val_counts_', None)),
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
        # Values are the model graph score per respective term.

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

        term_names = self.term_names_
        term_types = _generate_term_types(self.feature_types_in_, self.term_features_)

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
                    "names": [None] * len(self.term_features_),
                    "scores": [None] * len(self.term_features_),
                    "values": [None] * len(self.term_features_),
                    "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
                }
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                    }
                data_dicts.append(data_dict)

            for term_idx, bin_indexes in eval_terms(X, n_samples, self.feature_names_in_, self.feature_types_in_, self.bins_, self.term_features_):
                scores = self.term_scores_[term_idx][tuple(bin_indexes)]
                feature_idxs = self.term_features_[term_idx]
                for row_idx in range(n_samples):
                    term_name = term_names[term_idx]
                    data_dicts[row_idx]["names"][term_idx] = term_name
                    data_dicts[row_idx]["scores"][term_idx] = scores[row_idx]
                    if len(feature_idxs) == 1:
                        data_dicts[row_idx]["values"][term_idx] = X_unified[row_idx, feature_idxs[0]]
                    else:
                        data_dicts[row_idx]["values"][term_idx] = ""

            sample_scores = ebm_decision_function(
                X, 
                n_samples, 
                self.feature_names_in_, 
                self.feature_types_in_, 
                self.bins_, 
                self.intercept_, 
                self.term_scores_, 
                self.term_features_
            )

            if is_classifier(self):
                # Handle binary classification case -- softmax only works with 0s appended
                if sample_scores.ndim == 1:
                    sample_scores = np.c_[np.zeros(sample_scores.shape), sample_scores]

                sample_scores = softmax(sample_scores)

            perf_dicts = gen_perf_dicts(sample_scores, y, is_classifier(self))
            for row_idx in range(n_samples):
                perf = None if perf_dicts is None else perf_dicts[row_idx]
                perf_list.append(perf)
                data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classifier(self))

        term_scores = remove_last2(self.term_scores_, self.bin_weights_)
        for term_idx, feature_idxs in enumerate(self.term_features_):
            term_scores[term_idx] = trim_tensor(term_scores[term_idx], trim_low=[True] * len(feature_idxs))

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": term_scores,
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
            feature_names=term_names,
            feature_types=['categorical' if x == 'nominal' or x == 'ordinal' else x for x in term_types],
            name=gen_name_from_class(self) if name is None else name,
            selector=selector,
        )

    def get_histogram_edges(self, feature_idx):
        """ Provides the histogram edges used in the model

        Args:
            feature_idx: index of the feature to generate the histogram edges for

        Returns:
            An array of histogram edges
        """

        feature_bounds = getattr(self, 'feature_bounds_', None)
        if feature_bounds is not None:
            min_feature_val = feature_bounds[feature_idx, 0]
            max_feature_val = feature_bounds[feature_idx, 1]
            if not isnan(min_feature_val) and not isnan(max_feature_val):
                histogram_counts = getattr(self, 'histogram_counts_', None)
                if histogram_counts is not None:
                    histogram_bin_counts = histogram_counts[feature_idx]
                    if histogram_bin_counts is not None:
                        return make_histogram_edges(min_feature_val, max_feature_val, histogram_bin_counts)
        return None

    def get_importances(self, importance_type='avg_weight'):
        """ Provides the term importances

        Args:
            importance_type: the type of term importance requested ('avg_weight', 'min_max')

        Returns:
            An array term importances with one importance per additive term
        """

        if importance_type == 'avg_weight':
            importances = np.empty(len(self.term_features_), np.float64)
            for i in range(len(self.term_features_)):
                mean_abs_score = np.abs(self.term_scores_[i])
                if is_classifier(self) and 2 < len(self.classes_):
                    mean_abs_score = np.average(mean_abs_score, axis=-1)
                mean_abs_score = np.average(mean_abs_score, weights=self.bin_weights_[i])
                importances.itemset(i, mean_abs_score)
            return importances
        elif importance_type == 'min_max':
            return np.array([np.max(tensor) - np.min(tensor) for tensor in self.term_scores_], np.float64)
        else:
            raise ValueError(f"Unrecognized importance_type: {importance_type}")

class ExplainableBoostingClassifier(EBMModel, ClassifierMixin, ExplainerMixin):
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
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile", or "rounded_quantile". 'rounded_quantile' will round to as few decimals as possible while preserving the same bins as 'quantile'.
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
            self.term_scores_, 
            self.term_features_
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
            self.term_scores_, 
            self.term_features_
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
            self.term_scores_, 
            self.term_features_
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

class ExplainableBoostingRegressor(EBMModel, RegressorMixin, ExplainerMixin):
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
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile", or "rounded_quantile". 'rounded_quantile' will round to as few decimals as possible while preserving the same bins as 'quantile'.
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
            self.term_scores_, 
            self.term_features_
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
            self.term_scores_, 
            self.term_features_
        )


class DPExplainableBoostingClassifier(EBMModel, ClassifierMixin, ExplainerMixin):
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
        random_state=None,
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
            binning: Method to bin values for pre-processing. 'private' is the only legal option currently for DP.
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
            self.term_scores_, 
            self.term_features_
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
            self.term_scores_, 
            self.term_features_
        )

        # Handle binary classification case -- softmax only works with 0s appended
        if log_odds_vector.ndim == 1:
            log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]

        return self.classes_[np.argmax(log_odds_vector, axis=1)]


class DPExplainableBoostingRegressor(EBMModel, RegressorMixin, ExplainerMixin):
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
        random_state=None,
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
            binning: Method to bin values for pre-processing. 'private' is the only legal option currently for DP.
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
            self.term_scores_, 
            self.term_features_
        )

