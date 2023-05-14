# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


from typing import Optional, List, Tuple, Sequence, Dict, Mapping, Union

from itertools import count

from ...utils._explanation import gen_perf_dicts
from ._boost import boost
from ._utils import (
    make_bag,
    jsonify_item,
    jsonify_lists,
    process_terms,
    order_terms,
    remove_unused_higher_bins,
    deduplicate_bins,
    generate_term_names,
    generate_term_types,
)
from ...utils._histogram import make_all_histogram_edges
from ...utils._link import inv_link
from ...utils._seed import normalize_initial_seed
from ...utils._clean_x import preclean_X
from ...utils._clean_simple import (
    clean_dimensions,
    clean_init_score_and_X,
    typify_classification,
)

from ...utils._unify_data import unify_data

from ...utils._preprocessor import construct_bins
from ...utils._compressed_dataset import bin_native_by_dimension

from ._bin import (
    eval_terms,
    ebm_decision_function,
    ebm_decision_function_and_explain,
    make_bin_weights,
)
from ._tensor import (
    make_boosting_weights,
    after_boosting,
    remove_last,
    trim_tensor,
)
from ...utils._native import Native
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...provider import JobLibProvider
from ...utils._explanation import (
    gen_name_from_class,
    gen_global_selector,
    gen_local_selector,
)
from ...utils._rank_interactions import rank_interactions
from ...utils._privacy import (
    validate_eps_delta,
    calc_classic_noise_multi,
    calc_gdp_noise_multi,
)

import json
from math import isnan, ceil

import numpy as np
from warnings import warn

from sklearn.base import is_classifier  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from sklearn.isotonic import IsotonicRegression

import heapq
import operator

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)  # type: ignore
from itertools import combinations, groupby

import logging

_log = logging.getLogger(__name__)


class EBMExplanation(FeatureValueExplanation):
    """Visualizes specifically for EBM."""

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
        super(EBMExplanation, self).__init__(
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
        from ...visual.plot import (
            plot_continuous_bar,
            plot_horizontal_bar,
            sort_take,
            is_multiclass_global_data_dict,
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
                xtitle="Mean Absolute Score (Weighted)",
            )

            figure._interpret_help_text = (
                "The term importances are the mean absolute "
                "contribution (score) each term (feature or interaction) makes to predictions "
                "averaged across the training dataset. Contributions are weighted by the number "
                "of samples in each bin, and by the sample weights (if any). The 15 most "
                "important terms are shown."
            )
            figure._interpret_help_link = "https://github.com/interpretml/interpret/blob/develop/examples/python/EBM_Feature_Importances.ipynb"

            return figure

        # Per term global explanation
        if self.explanation_type == "global":
            title = "Term: {0} ({1})".format(
                self.feature_names[key], self.feature_types[key]
            )

            if self.feature_types[key] == "continuous":
                xtitle = self.feature_names[key]

                if is_multiclass_global_data_dict(data_dict):
                    figure = plot_continuous_bar(
                        data_dict,
                        multiclass=True,
                        show_error=False,
                        title=title,
                        xtitle=xtitle,
                    )
                else:
                    figure = plot_continuous_bar(data_dict, title=title, xtitle=xtitle)

            elif (
                self.feature_types[key] == "nominal"
                or self.feature_types[key] == "ordinal"
                or self.feature_types[key] == "interaction"
            ):
                figure = super().visualize(key, title)
                figure._interpret_help_text = (
                    "The contribution (score) of the term {0} to predictions "
                    "made by the model.".format(self.feature_names[key])
                )
            else:  # pragma: no cover
                raise Exception(
                    "Not supported configuration: {0}, {1}".format(
                        self.explanation_type, self.feature_types[key]
                    )
                )

            figure._interpret_help_text = (
                "The contribution (score) of the term "
                "{0} to predictions made by the model. For classification, "
                "scores are on a log scale (logits). For regression, scores are on the same "
                "scale as the outcome being predicted (e.g., dollars when predicting cost). "
                "Each graph is centered vertically such that average prediction on the train "
                "set is 0.".format(self.feature_names[key])
            )

            return figure

        # Local explanation graph
        if self.explanation_type == "local":
            figure = super().visualize(key)
            figure.update_layout(
                title="Local Explanation (" + figure.layout.title.text + ")",
                xaxis_title="Contribution to Prediction",
            )
            figure._interpret_help_text = (
                "A local explanation shows the breakdown of how much "
                "each term contributed to the prediction for a single sample. The intercept "
                "reflects the average case. In regression, the intercept is the average y-value "
                "of the train set (e.g., $5.51 if predicting cost). In classification, the "
                "intercept is the log of the base rate (e.g., -2.3 if the base rate is 10%). The "
                "15 most important terms are shown."
            )

            return figure


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

    return isinstance(
        estimator, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)
    )


def _clean_exclude(exclude, feature_map):
    ret = set()
    for term in exclude:
        if isinstance(term, int) or isinstance(term, float) or isinstance(term, str):
            term = (term,)

        cleaned = []
        for feature in term:
            if isinstance(feature, float):
                if not feature.is_integer():
                    msg = "exclude must contain integers or feature names"
                    _log.error(msg)
                    raise ValueError(msg)
                feature = int(feature)
            elif isinstance(feature, str):
                if feature not in feature_map:
                    msg = f"exclude item {feature} not in feature names"
                    _log.error(msg)
                    raise ValueError(msg)
                feature = feature_map[feature]
            elif not isinstance(feature, int):
                msg = f"unrecognized item type {type(feature)} in exclude"
                _log.error(msg)
                raise ValueError(msg)
            cleaned.append(feature)
        if len(set(cleaned)) != len(cleaned):
            msg = f"exclude contains duplicate features: {cleaned}"
            _log.error(msg)
            raise ValueError(msg)
        cleaned.sort()
        # allow duplicates in the exclude list
        ret.add(tuple(cleaned))
    return ret


class EBMModel(BaseEstimator):
    """Base class for all EBMs"""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        # Stages
        interactions=10,
        exclude=[],
        # Ensemble
        validation_size=0.15,
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        greediness=0.0,
        smoothing_rounds=0,
        max_rounds=5000,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        objective=None,
        # Overall
        n_jobs=-2,
        random_state=42,
        # Differential Privacy
        epsilon=1,
        delta=1e-5,
        composition="gdp",
        bin_budget_frac=0.1,
        privacy_bounds=None,
        privacy_target_min=None,
        privacy_target_max=None,
    ):
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.max_bins = max_bins
        if not is_private(self):
            self.max_interaction_bins = max_interaction_bins

        if not is_private(self):
            self.interactions = interactions
        self.exclude = exclude

        self.validation_size = validation_size
        self.outer_bags = outer_bags
        if not is_private(self):
            self.inner_bags = inner_bags

        self.learning_rate = learning_rate
        if not is_private(self):
            self.greediness = greediness
            self.smoothing_rounds = smoothing_rounds
        self.max_rounds = max_rounds
        if not is_private(self):
            self.early_stopping_rounds = early_stopping_rounds
            self.early_stopping_tolerance = early_stopping_tolerance

            self.min_samples_leaf = min_samples_leaf

        self.max_leaves = max_leaves
        self.objective = objective

        self.n_jobs = n_jobs
        self.random_state = random_state

        if is_private(self):
            # Arguments for differential privacy
            self.epsilon = epsilon
            self.delta = delta
            self.composition = composition
            self.bin_budget_frac = bin_budget_frac
            self.privacy_bounds = privacy_bounds
            self.privacy_target_min = privacy_target_min
            self.privacy_target_max = privacy_target_max

            if random_state is not None:
                warn(
                    f"Privacy violation: using a fixed random_state of {random_state} "
                    "will cause deterministic noise additions. This capability is only "
                    "for debugging/testing. Set random_state to None to remove this warning."
                )

    def fit(self, X, y, sample_weight=None, init_score=None):  # noqa: C901
        """Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Itself.
        """

        # with 64 bytes per tensor cell, a 2^20 tensor would be 1/16 gigabyte.
        max_cardinality = 1048576

        if not isinstance(self.outer_bags, int) and not self.outer_bags.is_integer():
            msg = "outer_bags must be an integer"
            _log.error(msg)
            raise ValueError(msg)
        elif self.outer_bags < 1:
            msg = "outer_bags must be 1 or greater. Did you mean to set: outer_bags=1, validation_size=0?"
            _log.error(msg)
            raise ValueError(msg)

        if not isinstance(self.validation_size, int) and not isinstance(
            self.validation_size, float
        ):
            msg = "validation_size must be an integer or float"
            _log.error(msg)
            raise ValueError(msg)
        elif self.validation_size <= 0:
            if self.validation_size < 0:
                msg = "validation_size cannot be negative"
                _log.error(msg)
                raise ValueError(msg)
            elif 1 < self.outer_bags:
                warn(
                    "If validation_size is 0, the outer_bags have no purpose. Set outer_bags=1 to remove this warning."
                )
        elif 1 <= self.validation_size:
            # validation_size equal to 1 or more is an exact number specification, so it must be an integer
            if (
                not isinstance(self.validation_size, int)
                and not self.validation_size.is_integer()
            ):
                msg = "If 1 <= validation_size, it is an exact count of samples, and must be an integer"
                _log.error(msg)
                raise ValueError(msg)

        if not isinstance(self.max_rounds, int) and not self.max_rounds.is_integer():
            msg = "max_rounds must be an integer"
            _log.error(msg)
            raise ValueError(msg)
        elif self.max_rounds < 0:
            # max_rounds == 0 means no boosting. This can be useful to just perform discretization
            msg = "max_rounds cannot be negative"
            _log.error(msg)
            raise ValueError(msg)

        if not is_private(self):
            if self.greediness < 0.0:
                msg = "greediness cannot be negative"
                _log.error(msg)
                raise ValueError(msg)
            elif 1.0 < self.greediness:
                msg = "greediness must be a percentage between 0.0 and 1.0 inclusive"
                _log.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(self.smoothing_rounds, int)
                and not self.smoothing_rounds.is_integer()
            ):
                msg = "smoothing_rounds must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            elif self.smoothing_rounds < 0:
                msg = "smoothing_rounds cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(self.inner_bags, int)
                and not self.inner_bags.is_integer()
            ):
                msg = "inner_bags must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            elif self.inner_bags < 0:
                # inner_bags == 0 turns off inner bagging
                msg = "inner_bags cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(self.early_stopping_rounds, int)
                and not self.early_stopping_rounds.is_integer()
            ):
                msg = "early_stopping_rounds must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            elif self.early_stopping_rounds < 0:
                # early_stopping_rounds == 0 means turn off early_stopping
                # early_stopping_rounds == 1 means check after the first round, etc
                msg = "early_stopping_rounds cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if not isinstance(self.early_stopping_tolerance, int) and not isinstance(
                self.early_stopping_tolerance, float
            ):
                msg = "early_stopping_tolerance must be a float"
                _log.error(msg)
                raise ValueError(msg)

        if not isinstance(self.learning_rate, int) and not isinstance(
            self.learning_rate, float
        ):
            msg = "learning_rate must be a float"
            _log.error(msg)
            raise ValueError(msg)
        elif self.learning_rate <= 0:
            msg = "learning_rate must be a positive number"
            _log.error(msg)
            raise ValueError(msg)

        # TODO: check the other inputs for common mistakes here

        y = clean_dimensions(y, "y")
        if y.ndim != 1:
            msg = "y must be 1 dimensional"
            _log.error(msg)
            raise ValueError(msg)
        if len(y) == 0:
            msg = "y cannot have 0 samples"
            _log.error(msg)
            raise ValueError(msg)

        objective = self.objective
        if is_classifier(self):
            y = typify_classification(y)
            # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
            # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
            # in two separate runs, which would flip the ordering of the classes within our score tensors.
            classes, y = np.unique(y, return_inverse=True)
            n_classes = len(classes)
            if objective is None or objective.isspace():
                objective = "log_loss"
        else:
            y = y.astype(np.float64, copy=False)
            min_target = y.min()
            max_target = y.max()
            n_classes = -1
            if objective is None or objective.isspace():
                objective = "rmse"

        if sample_weight is not None:
            sample_weight = clean_dimensions(sample_weight, "sample_weight")
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1 dimensional")
            if len(y) != len(sample_weight):
                msg = f"y has {len(y)} samples and sample_weight has {len(sample_weight)} samples"
                _log.error(msg)
                raise ValueError(msg)
            sample_weight = sample_weight.astype(np.float64, copy=False)

        is_differential_privacy = is_private(self)

        native = Native.get_native_singleton()
        link, link_param = native.determine_link(is_differential_privacy, objective)

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names, self.feature_types, len(y))
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                link,
                link_param,
                init_score,
                X,
                self.feature_names,
                self.feature_types,
                len(y),
            )

        # Privacy calculations
        if is_differential_privacy:
            validate_eps_delta(self.epsilon, self.delta)

            if is_classifier(self):
                if 2 < n_classes:  # pragma: no cover
                    raise ValueError(
                        "Multiclass not supported for Differentially Private EBMs."
                    )
            else:
                is_privacy_warning = False
                is_clipping = False

                if self.privacy_target_min is None or isnan(self.privacy_target_min):
                    is_privacy_warning = True
                else:
                    is_clipping = True
                    min_target = float(self.privacy_target_min)

                if self.privacy_target_max is None or isnan(self.privacy_target_max):
                    is_privacy_warning = True
                else:
                    is_clipping = True
                    max_target = float(self.privacy_target_max)

                # In theory privacy_target_min and privacy_target_max are not needed
                # in our interface since the caller could clip 'y' themselves, but
                # having it here is a check that the clipping was not overlooked.
                if is_privacy_warning:
                    warn(
                        "Possible privacy violation: assuming min/max values for "
                        "target are public info. Pass in privacy_target_min and "
                        "privacy_target_max with known public values to avoid "
                        "this warning."
                    )

                if is_clipping:
                    y = np.clip(y, min_target, max_target)

            # Split epsilon, delta budget for binning and learning
            bin_eps = self.epsilon * self.bin_budget_frac
            bin_delta = self.delta / 2
            composition = self.composition
            privacy_bounds = self.privacy_bounds
            binning = "private"
            # TODO: should we make this something higher?
            min_unique_continuous = 3

            bin_levels = [self.max_bins]
        else:
            bin_eps = None
            bin_delta = None
            composition = None
            privacy_bounds = None
            binning = "quantile"
            # TODO: bump this up to something like 10 again, but ONLY after we've standardized
            #       our code to turn 1 and 1.0 both into the categorical "1" AND we can handle
            #       categorical to continuous soft transitions.
            min_unique_continuous = 0

            bin_levels = [self.max_bins, self.max_interaction_bins]

        init_random_state = normalize_initial_seed(self.random_state)

        # after normalizing to a 32-bit signed integer, we pass the random_state into the EBMPreprocessor
        # exactly as passed to us. This means that we should get the same preprocessed data for the mains
        # if we create an EBMPreprocessor with the same seed.  For interactions, we increment by one
        # so it can be replicated without creating an EBM
        binning_result = construct_bins(
            X=X,
            y=y,
            sample_weight=sample_weight,
            feature_names_given=self.feature_names,
            feature_types_given=self.feature_types,
            max_bins_leveled=bin_levels,
            binning=binning,
            min_samples_bin=1,
            min_unique_continuous=min_unique_continuous,
            random_state=init_random_state,
            epsilon=bin_eps,
            delta=bin_delta,
            composition=composition,
            privacy_bounds=privacy_bounds,
        )
        feature_names_in = binning_result[0]
        feature_types_in = binning_result[1]
        bins = binning_result[2]
        main_bin_weights = binning_result[3]
        feature_bounds = binning_result[4]
        histogram_weights = binning_result[5]
        missing_val_counts = binning_result[6]
        unique_val_counts = binning_result[7]
        noise_scale_binning = binning_result[8]

        if np.count_nonzero(missing_val_counts):
            warn(
                "Missing values detected. Our visualizations do not currently display missing values. "
                "To retain the glassbox nature of the model you need to either set the missing values "
                "to an extreme value like -1000 that will be visible on the graphs, or manually "
                "examine the missing value score in ebm.term_scores_[term_index][0]"
            )

        n_features_in = len(bins)

        feature_map = dict(zip(feature_names_in, count()))

        exclude = self.exclude
        if exclude is None:
            exclude = set()
            term_features = [(x,) for x in range(n_features_in)]
        elif exclude == "mains":
            exclude = set()
            term_features = []
        else:
            exclude = _clean_exclude(exclude, feature_map)
            term_features = [(x,) for x in range(n_features_in) if (x,) not in exclude]

        if is_differential_privacy:
            # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            domain_size = 1 if is_classifier(self) else max_target - min_target
            max_weight = 1 if sample_weight is None else np.max(sample_weight)
            training_eps = self.epsilon - bin_eps
            training_delta = self.delta - bin_delta
            if self.composition == "classic":
                noise_scale_boosting = calc_classic_noise_multi(
                    total_queries=self.max_rounds
                    * len(term_features)
                    * self.outer_bags,
                    target_epsilon=training_eps,
                    delta=training_delta,
                    sensitivity=domain_size * self.learning_rate * max_weight,
                )
            elif self.composition == "gdp":
                noise_scale_boosting = calc_gdp_noise_multi(
                    total_queries=self.max_rounds
                    * len(term_features)
                    * self.outer_bags,
                    target_epsilon=training_eps,
                    delta=training_delta,
                )
                # Alg Line 17
                noise_scale_boosting *= domain_size * self.learning_rate * max_weight
            else:
                raise NotImplementedError(
                    f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'."
                )

            bin_data_weights = make_boosting_weights(main_bin_weights)
            boost_flags = (
                Native.BoostFlags_GradientSums | Native.BoostFlags_RandomSplits
            )
            inner_bags = 0
            greediness = 0.0
            smoothing_rounds = 0
            early_stopping_rounds = 0
            early_stopping_tolerance = 0
            min_samples_leaf = 0
            interactions = 0
        else:
            noise_scale_boosting = None
            bin_data_weights = None
            boost_flags = Native.BoostFlags_Default
            inner_bags = self.inner_bags
            greediness = self.greediness
            smoothing_rounds = self.smoothing_rounds
            early_stopping_rounds = self.early_stopping_rounds
            early_stopping_tolerance = self.early_stopping_tolerance
            min_samples_leaf = self.min_samples_leaf
            interactions = self.interactions

        rng = native.create_rng(init_random_state)
        # branch it so we have no correlation to the binning rng that uses the same seed
        rng = native.branch_rng(rng)
        used_seeds = set()
        rngs = []
        bag_weights = []
        bags = []
        for _ in range(self.outer_bags):
            while True:
                bagged_rng = native.branch_rng(rng)
                seed = native.generate_seed(bagged_rng)
                # we really really do not want identical bags. branch_rng is pretty good but it can lead to
                # collisions, so check with a 32-bit seed if we possibly have a collision and regenerate if so
                if seed not in used_seeds:
                    break
            # we do not need used_seeds if the rng is None, but it does not hurt anything
            used_seeds.add(seed)

            bag = make_bag(
                y,
                self.validation_size,
                bagged_rng,
                is_classifier(self) and not is_differential_privacy,
            )
            # we bag within the same proces, so bagged_rng will progress inside make_bag
            rngs.append(bagged_rng)
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

        if n_classes == 1:
            warn(
                "Only 1 class detected for classification. The model will predict 1.0 whenever predict_proba is called."
            )

            breakpoint_iteration = [[]]
            models = []
            for idx in range(self.outer_bags):
                breakpoint_iteration[-1].append(0)
                tensors = []
                for bin_levels in bins:
                    feature_bins = bin_levels[0]
                    if isinstance(feature_bins, dict):
                        # categorical feature
                        n_bins = (
                            2
                            if len(feature_bins) == 0
                            else max(feature_bins.values()) + 2
                        )
                    else:
                        # continuous feature
                        n_bins = len(feature_bins) + 3
                    tensor = np.full(n_bins, -np.inf, np.float64)
                    tensors.append(tensor)
                models.append(tensors)
        else:
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

            parallel_args = []
            for idx in range(self.outer_bags):
                early_stopping_rounds_local = early_stopping_rounds
                if bags[idx] is None or (0 <= bags[idx]).all():
                    # if there are no validation samples, turn off early stopping
                    # because the validation metric cannot improve each round
                    early_stopping_rounds_local = 0

                parallel_args.append(
                    (
                        dataset,
                        bags[idx],
                        init_score,
                        term_features,
                        inner_bags,
                        boost_flags,
                        self.learning_rate,
                        min_samples_leaf,
                        self.max_leaves,
                        greediness,
                        smoothing_rounds,
                        self.max_rounds,
                        early_stopping_rounds_local,
                        early_stopping_tolerance,
                        noise_scale_boosting,
                        bin_data_weights,
                        rngs[idx],
                        is_differential_privacy,
                        objective,
                        None,
                    )
                )

            results = provider.parallel(boost, parallel_args)

            # let python reclaim the dataset memory via reference counting
            del parallel_args  # parallel_args holds references to dataset, so must be deleted
            del dataset

            breakpoint_iteration = [[]]
            models = []
            rngs = []
            for model, bag_breakpoint_iteration, bagged_rng in results:
                breakpoint_iteration[-1].append(bag_breakpoint_iteration)
                models.append(after_boosting(term_features, model, main_bin_weights))
                # retrieve our rng state since this was used outside of our process
                rngs.append(bagged_rng)

            while True:  # this isn't for looping. Just for break statements to exit
                if interactions is None:
                    break

                if isinstance(interactions, int) or isinstance(interactions, float):
                    if interactions <= 0:
                        if interactions == 0:
                            break
                        msg = "interactions cannot be negative"
                        _log.error(msg)
                        raise ValueError(msg)

                    if interactions < 1.0:
                        interactions = int(ceil(n_features_in * interactions))
                    elif isinstance(interactions, float):
                        if not interactions.is_integer():
                            msg = "interactions above 1 cannot be a float percentage and need to be an int instead"
                            _log.error(msg)
                            raise ValueError(msg)
                        interactions = int(interactions)

                    if 2 < n_classes:
                        warn(
                            "Detected multiclass problem. Forcing interactions to 0. "
                            "Multiclass interactions work except for global "
                            "visualizations, so the break statement below that "
                            "disables multiclass interactions can be removed."
                        )
                        break

                    # at this point interactions will be a positive, nonzero integer
                else:
                    # interactions must be a list of the interactions
                    if len(interactions) == 0:
                        break

                    if 2 < n_classes:
                        raise ValueError(
                            "Interactions are not supported for multiclass. "
                            "Multiclass interactions work except for global "
                            "visualizations, so this exception can be disabled "
                            "if you know what you are doing."
                        )

                initial_intercept = np.zeros(
                    Native.get_count_scores_c(n_classes), np.float64
                )
                scores_bags = []
                for model in models:
                    # TODO: instead of going back to the original data in X, we
                    # could use the compressed and already binned data in dataset
                    scores_bags.append(
                        ebm_decision_function(
                            X,
                            n_samples,
                            feature_names_in,
                            feature_types_in,
                            bins,
                            initial_intercept,
                            model,
                            term_features,
                            init_score,
                        )
                    )

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
                del y  # we no longer need this, so allow the garbage collector to reclaim it

                if isinstance(interactions, int):
                    _log.info("Estimating with FAST")

                    parallel_args = []
                    for idx in range(self.outer_bags):
                        # TODO: the combinations below should be selected from the non-excluded features
                        parallel_args.append(
                            (
                                dataset,
                                bags[idx],
                                scores_bags[idx],
                                combinations(range(n_features_in), 2),
                                exclude,
                                Native.InteractionFlags_Default,
                                max_cardinality,
                                min_samples_leaf,
                                is_differential_privacy,
                                objective,
                                None,
                            )
                        )

                    bagged_ranked_interaction = provider.parallel(
                        rank_interactions, parallel_args
                    )

                    # this holds references to dataset, bags, and scores_bags which we want python to reclaim later
                    del parallel_args

                    # Select merged pairs
                    pair_ranks = {}
                    for n, interaction_strengths_and_indices in enumerate(
                        bagged_ranked_interaction
                    ):
                        interaction_indices = list(
                            map(
                                operator.itemgetter(1),
                                interaction_strengths_and_indices,
                            )
                        )
                        for rank, indices in enumerate(interaction_indices):
                            old_mean = pair_ranks.get(indices, 0)
                            pair_ranks[indices] = old_mean + (
                                (rank - old_mean) / (n + 1)
                            )

                    final_ranks = []
                    total_interactions = 0
                    for indices in pair_ranks:
                        heapq.heappush(final_ranks, (pair_ranks[indices], indices))
                        total_interactions += 1

                    n_interactions = min(interactions, total_interactions)
                    boost_groups = [
                        heapq.heappop(final_ranks)[1] for _ in range(n_interactions)
                    ]
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
                        if (
                            sorted_tuple not in uniquifier
                            and sorted_tuple not in exclude
                        ):
                            uniquifier.add(sorted_tuple)
                            boost_groups.append(feature_idxs)

                    # Warn the users that we have made change to the interactions list
                    if len(boost_groups) != len(interactions):
                        warn("Removed interaction terms")

                    if 2 < max_dimensions:
                        warn(
                            "Interactions with 3 or more terms are not graphed in "
                            "global explanations. Local explanations are still "
                            "available and exact."
                        )

                parallel_args = []
                for idx in range(self.outer_bags):
                    early_stopping_rounds_local = early_stopping_rounds
                    if bags[idx] is None or (0 <= bags[idx]).all():
                        # if there are no validation samples, turn off early stopping
                        # because the validation metric cannot improve each round
                        early_stopping_rounds_local = 0

                    parallel_args.append(
                        (
                            dataset,
                            bags[idx],
                            scores_bags[idx],
                            boost_groups,
                            inner_bags,
                            boost_flags,
                            self.learning_rate,
                            min_samples_leaf,
                            self.max_leaves,
                            greediness,
                            0,  # no smoothing rounds for interactions
                            self.max_rounds,
                            early_stopping_rounds_local,
                            early_stopping_tolerance,
                            noise_scale_boosting,
                            bin_data_weights,
                            rngs[idx],
                            is_differential_privacy,
                            objective,
                            None,
                        )
                    )

                results = provider.parallel(boost, parallel_args)

                # allow python to reclaim these big memory items via reference counting
                del parallel_args  # this holds references to dataset, scores_bags, and bags
                del dataset
                del scores_bags

                breakpoint_iteration.append([])
                for idx in range(self.outer_bags):
                    breakpoint_iteration[-1].append(results[idx][1])
                    models[idx].extend(
                        after_boosting(boost_groups, results[idx][0], main_bin_weights)
                    )
                    rngs[idx] = results[idx][2]

                term_features.extend(boost_groups)

                break  # do not loop!

        breakpoint_iteration = np.array(breakpoint_iteration, np.int64)

        remove_unused_higher_bins(term_features, bins)
        deduplicate_bins(bins)

        bagged_scores = (
            np.array([model[idx] for model in models], np.float64)
            for idx in range(len(term_features))
        )

        term_features, bagged_scores = order_terms(term_features, bagged_scores)

        if is_differential_privacy:
            # for now we only support mains for DP models
            bin_weights = [
                main_bin_weights[feature_idxs[0]] for feature_idxs in term_features
            ]
        else:
            histogram_edges = make_all_histogram_edges(
                feature_bounds, histogram_weights
            )
            bin_weights = make_bin_weights(
                X,
                n_samples,
                sample_weight,
                feature_names_in,
                feature_types_in,
                bins,
                term_features,
            )

        term_scores, standard_deviations, intercept, bagged_scores = process_terms(
            n_classes, bagged_scores, bin_weights, bag_weights
        )

        term_names = generate_term_names(feature_names_in, term_features)

        # dependent attributes (can be re-derrived after serialization)
        self.n_features_in_ = n_features_in  # scikit-learn specified name
        self.term_names_ = term_names

        if is_differential_privacy:
            self.noise_scale_binning_ = noise_scale_binning
            self.noise_scale_boosting_ = noise_scale_boosting
        else:
            # differentially private models would need to pay additional privacy budget to make
            # these public, but they are non-essential so we don't disclose them in the DP setting

            # dependent attribute (can be re-derrived after serialization from feature_bounds_)
            self.histogram_edges_ = histogram_edges

            # per-feature
            self.histogram_weights_ = histogram_weights
            self.unique_val_counts_ = unique_val_counts

        if 0 <= n_classes:
            self.classes_ = classes  # required by scikit-learn
        else:
            # we do not use these currently, but they indicate the domain for DP and
            # we could use them in the future to indicate on the graphs the target range
            self.min_target_ = min_target
            self.max_target_ = max_target

        # per-feature
        self.bins_ = bins
        self.feature_names_in_ = feature_names_in  # scikit-learn specified name
        self.feature_types_in_ = feature_types_in
        self.feature_bounds_ = feature_bounds

        # per-term
        self.term_features_ = term_features
        self.bin_weights_ = bin_weights
        self.bagged_scores_ = bagged_scores
        self.term_scores_ = term_scores
        self.standard_deviations_ = standard_deviations

        # general
        self.intercept_ = intercept
        self.link_ = link
        self.link_param_ = link_param
        self.bag_weights_ = bag_weights
        self.breakpoint_iteration_ = breakpoint_iteration
        self.has_fitted_ = True

        return self

    def _to_inner_jsonable(self, properties="all"):
        """Converts the inner model to a JSONable representation.

        Args:
            properties: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSONable object
        """

        check_is_fitted(self, "has_fitted_")

        if properties == "minimal":
            level = 0
        elif properties == "interpretable":
            level = 1
        elif properties == "mergeable":
            level = 2
        elif properties == "all":
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
            output["output_type"] = "classification"
            output["classes"] = self.classes_.tolist()
        else:
            output["output_type"] = "regression"
            if 3 <= level:
                min_target = getattr(self, "min_target_", None)
                if min_target is not None and not isnan(min_target):
                    output["min_target"] = jsonify_item(min_target)
                max_target = getattr(self, "max_target_", None)
                if max_target is not None and not isnan(max_target):
                    output["max_target"] = jsonify_item(max_target)

        output["link"] = self.link_
        output["link_param"] = jsonify_item(self.link_param_)

        outputs.append(output)
        j["outputs"] = outputs

        if type(self.intercept_) is float:
            # scikit-learn requires that we have a single float value as our intercept for compatibility with
            # RegressorMixin, but in other scenarios where we want to support things like multi-output it would be
            # easier if the regression intercept were handled identically to classification, so put it in an array
            # for our JSON format to harmonize the cross-language representation
            j["intercept"] = [jsonify_item(self.intercept_)]
        else:
            j["intercept"] = jsonify_lists(self.intercept_.tolist())

        if 3 <= level:
            noise_scale_binning = getattr(self, "noise_scale_binning_", None)
            if noise_scale_binning is not None:
                j["noise_scale_binning"] = jsonify_item(noise_scale_binning)
            noise_scale_boosting = getattr(self, "noise_scale_boosting_", None)
            if noise_scale_boosting is not None:
                j["noise_scale_boosting"] = jsonify_item(noise_scale_boosting)
        if 2 <= level:
            bag_weights = getattr(self, "bag_weights_", None)
            if bag_weights is not None:
                j["bag_weights"] = jsonify_lists(bag_weights.tolist())
        if 3 <= level:
            breakpoint_iteration = getattr(self, "breakpoint_iteration_", None)
            if breakpoint_iteration is not None:
                j["breakpoint_iteration"] = breakpoint_iteration.tolist()

        if 3 <= level:
            j["implementation"] = "python"
            params = {}

            # TODO: we need to clean up and validate our input parameters before putting them into JSON
            # if we were pass a numpy array instead of a list or a numpy type these would fail
            # for now we can just require that anything numpy as input is illegal

            if hasattr(self, "feature_names"):
                params["feature_names"] = self.feature_names

            if hasattr(self, "feature_types"):
                params["feature_types"] = self.feature_types

            if hasattr(self, "max_bins"):
                params["max_bins"] = self.max_bins

            if hasattr(self, "max_interaction_bins"):
                params["max_interaction_bins"] = self.max_interaction_bins

            if hasattr(self, "interactions"):
                params["interactions"] = self.interactions

            if hasattr(self, "exclude"):
                params["exclude"] = self.exclude

            if hasattr(self, "validation_size"):
                params["validation_size"] = self.validation_size

            if hasattr(self, "outer_bags"):
                params["outer_bags"] = self.outer_bags

            if hasattr(self, "inner_bags"):
                params["inner_bags"] = self.inner_bags

            if hasattr(self, "learning_rate"):
                params["learning_rate"] = self.learning_rate

            if hasattr(self, "greediness"):
                params["greediness"] = self.greediness

            if hasattr(self, "max_rounds"):
                params["max_rounds"] = self.max_rounds

            if hasattr(self, "early_stopping_rounds"):
                params["early_stopping_rounds"] = self.early_stopping_rounds

            if hasattr(self, "early_stopping_tolerance"):
                params["early_stopping_tolerance"] = self.early_stopping_tolerance

            if hasattr(self, "min_samples_leaf"):
                params["min_samples_leaf"] = self.min_samples_leaf

            if hasattr(self, "max_leaves"):
                params["max_leaves"] = self.max_leaves

            if hasattr(self, "n_jobs"):
                params["n_jobs"] = self.n_jobs

            if hasattr(self, "random_state"):
                params["random_state"] = self.random_state

            if hasattr(self, "epsilon"):
                params["epsilon"] = self.epsilon

            if hasattr(self, "delta"):
                params["delta"] = self.delta

            if hasattr(self, "composition"):
                params["composition"] = self.composition

            if hasattr(self, "bin_budget_frac"):
                params["bin_budget_frac"] = self.bin_budget_frac

            if hasattr(self, "privacy_bounds"):
                params["privacy_bounds"] = self.privacy_bounds

            if hasattr(self, "privacy_target_min"):
                params["privacy_target_min"] = self.privacy_target_min

            if hasattr(self, "privacy_target_max"):
                params["privacy_target_max"] = self.privacy_target_max

            j["implementation_params"] = params

        unique_val_counts = getattr(self, "unique_val_counts_", None)
        feature_bounds = getattr(self, "feature_bounds_", None)
        histogram_weights = getattr(self, "histogram_weights_", None)

        features = []
        for i in range(len(self.bins_)):
            feature = {}

            feature["name"] = self.feature_names_in_[i]
            feature["type"] = self.feature_types_in_[i]

            if 1 <= level:
                if unique_val_counts is not None:
                    feature["num_unique_vals"] = int(unique_val_counts[i])

            if isinstance(self.bins_[i][0], dict):
                categories = []
                for bins in self.bins_[i]:
                    leveled_categories = []
                    feature_categories = list(map(tuple, map(reversed, bins.items())))
                    feature_categories.sort()  # groupby requires sorted data
                    for _, category_iter in groupby(feature_categories, lambda x: x[0]):
                        category_group = [category for _, category in category_iter]
                        if len(category_group) == 1:
                            leveled_categories.append(category_group[0])
                        else:
                            leveled_categories.append(category_group)
                    categories.append(leveled_categories)
                feature["categories"] = categories
            else:
                cuts = []
                for bins in self.bins_[i]:
                    cuts.append(bins.tolist())
                feature["cuts"] = cuts
                if 1 <= level:
                    if feature_bounds is not None:
                        feature_min = feature_bounds[i, 0]
                        if not isnan(feature_min):
                            feature["min"] = jsonify_item(feature_min)
                        feature_max = feature_bounds[i, 1]
                        if not isnan(feature_max):
                            feature["max"] = jsonify_item(feature_max)
                    if histogram_weights is not None:
                        feature_histogram_weights = histogram_weights[i]
                        if feature_histogram_weights is not None:
                            feature[
                                "histogram_weights"
                            ] = feature_histogram_weights.tolist()

            features.append(feature)
        j["features"] = features

        standard_deviations_all = getattr(self, "standard_deviations_", None)
        bagged_scores_all = getattr(self, "bagged_scores_", None)

        terms = []
        for term_idx in range(len(self.term_features_)):
            term = {}
            term["term_features"] = [
                self.feature_names_in_[feature_idx]
                for feature_idx in self.term_features_[term_idx]
            ]
            term["scores"] = jsonify_lists(self.term_scores_[term_idx].tolist())
            if 1 <= level:
                if standard_deviations_all is not None:
                    standard_deviations = standard_deviations_all[term_idx]
                    if standard_deviations is not None:
                        term["standard_deviations"] = jsonify_lists(
                            standard_deviations.tolist()
                        )
            if 2 <= level:
                if bagged_scores_all is not None:
                    bagged_scores = bagged_scores_all[term_idx]
                    if bagged_scores is not None:
                        term["bagged_scores"] = jsonify_lists(bagged_scores.tolist())
            if 1 <= level:
                term["bin_weights"] = jsonify_lists(
                    self.bin_weights_[term_idx].tolist()
                )

            terms.append(term)
        j["terms"] = terms

        return j

    def _to_outer_jsonable(self, properties="all"):
        """Converts the outer model to a JSONable representation.

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
        outer["version"] = "1.0"
        outer["ebm"] = inner

        return outer

    def _to_json(self, properties="all"):
        """Converts the model to a JSON representation.

        Args:
            properties: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSON string
        """

        outer = self._to_outer_jsonable(properties)
        return json.dumps(outer, allow_nan=False, indent=2)

    def decision_function(self, X, init_score=None):
        """Predict scores from model before calling the link function.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        # TODO: handle the 1 class case here

        return ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

    def explain_global(self, name=None):
        """Provides global explanation for model.

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
        for scores, errors in zip(self.term_scores_, self.standard_deviations_):
            if errors is None:
                lower_bound = min(lower_bound, np.min(scores))
                upper_bound = max(upper_bound, np.max(scores))
            else:
                lower_bound = min(lower_bound, np.min(scores - errors))
                upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        mod_weights = remove_last(self.bin_weights_, self.bin_weights_)
        mod_term_scores = remove_last(self.term_scores_, self.bin_weights_)
        mod_standard_deviations = remove_last(
            self.standard_deviations_, self.bin_weights_
        )
        for term_idx, feature_idxs in enumerate(self.term_features_):
            mod_term_scores[term_idx] = trim_tensor(
                mod_term_scores[term_idx], trim_low=[True] * len(feature_idxs)
            )
            if mod_standard_deviations[term_idx] is not None:
                mod_standard_deviations[term_idx] = trim_tensor(
                    mod_standard_deviations[term_idx],
                    trim_low=[True] * len(feature_idxs),
                )
            if mod_weights[term_idx] is not None:
                mod_weights[term_idx] = trim_tensor(
                    mod_weights[term_idx], trim_low=[True] * len(feature_idxs)
                )

        term_names = self.term_names_
        term_types = generate_term_types(self.feature_types_in_, self.term_features_)

        native = Native.get_native_singleton()

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        keep_idxs = []
        for term_idx, feature_idxs in enumerate(self.term_features_):
            model_graph = mod_term_scores[term_idx]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = mod_standard_deviations[term_idx]

            if len(feature_idxs) == 1:
                keep_idxs.append(term_idx)

                feature_index0 = feature_idxs[0]

                feature_bins = self.bins_[feature_index0][0]
                if isinstance(feature_bins, dict):
                    # categorical

                    # TODO: this will fail if we have multiple categories in a bin
                    bin_labels = list(feature_bins.keys())

                    histogram_weights = getattr(self, "histogram_weights_", None)
                    if histogram_weights is not None:
                        histogram_weights = histogram_weights[feature_index0]

                    if histogram_weights is None:
                        histogram_weights = self.bin_weights_[term_idx]

                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append("DPOther")
                        histogram_weights = histogram_weights[1:]
                    else:
                        histogram_weights = histogram_weights[1:-1]

                    names = bin_labels
                    densities = list(histogram_weights)
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, "feature_bounds_", None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_index0, 0]
                        max_feature_val = feature_bounds[feature_index0, 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(
                        feature_bins, min_feature_val, max_feature_val
                    )
                    bin_labels = list(
                        np.concatenate(([min_graph], feature_bins, [max_graph]))
                    )

                    histogram_edges = getattr(self, "histogram_edges_", None)
                    if histogram_edges is not None:
                        histogram_edges = histogram_edges[feature_index0]
                    if histogram_edges is not None:
                        names = list(histogram_edges)
                        densities = list(self.histogram_weights_[feature_index0][1:-1])
                    else:
                        names = bin_labels
                        densities = list(mod_weights[term_idx])

                scores = list(model_graph)
                upper_bounds = None if errors is None else list(model_graph + errors)
                lower_bounds = None if errors is None else list(model_graph - errors)
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
                    "upper_bounds": None if errors is None else model_graph + errors,
                    "lower_bounds": None if errors is None else model_graph - errors,
                    "density": {
                        "names": names,
                        "scores": densities,
                    },
                }
                if is_classifier(self):
                    # Classes should be numpy array, convert to list.
                    data_dict["meta"] = {"label_names": self.classes_.tolist()}

                data_dicts.append(data_dict)
            elif len(feature_idxs) == 2:
                keep_idxs.append(term_idx)

                bin_levels = self.bins_[feature_idxs[0]]
                feature_bins = bin_levels[min(len(feature_idxs), len(bin_levels)) - 1]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[0]:
                        bin_labels.append("DPOther")
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, "feature_bounds_", None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_idxs[0], 0]
                        max_feature_val = feature_bounds[feature_idxs[0], 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(
                        feature_bins, min_feature_val, max_feature_val
                    )
                    bin_labels = list(
                        np.concatenate(([min_graph], feature_bins, [max_graph]))
                    )

                bin_labels_left = bin_labels

                bin_levels = self.bins_[feature_idxs[1]]
                feature_bins = bin_levels[min(len(feature_idxs), len(bin_levels)) - 1]
                if isinstance(feature_bins, dict):
                    # categorical
                    bin_labels = list(feature_bins.keys())
                    if len(bin_labels) != model_graph.shape[1]:
                        bin_labels.append("DPOther")
                else:
                    # continuous
                    min_feature_val = np.nan
                    max_feature_val = np.nan
                    feature_bounds = getattr(self, "feature_bounds_", None)
                    if feature_bounds is not None:
                        min_feature_val = feature_bounds[feature_idxs[1], 0]
                        max_feature_val = feature_bounds[feature_idxs[1], 1]

                    # this will have no effect in normal models, but will handle inconsistent editied models
                    min_graph, max_graph = native.suggest_graph_bounds(
                        feature_bins, min_feature_val, max_feature_val
                    )
                    bin_labels = list(
                        np.concatenate(([min_graph], feature_bins, [max_graph]))
                    )

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
                warn(
                    f"Dropping feature {term_names[term_idx]} from explanation "
                    "since we can't graph more than 2 dimensions."
                )

        importances = self.term_importances()

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
            feature_types=[term_types[i] for i in keep_idxs],
            name=name,
            selector=gen_global_selector(
                self.n_features_in_,
                [term_names[i] for i in keep_idxs],
                [term_types[i] for i in keep_idxs],
                getattr(self, "unique_val_counts_", None),
                None,
            ),
        )

    def explain_local(self, X, y=None, name=None, init_score=None):
        """Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """

        # Produce feature value pairs for each sample.
        # Values are the model graph score per respective term.

        check_is_fitted(self, "has_fitted_")

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

            if is_classifier(self):
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        if init_score is None:
            X, n_samples = preclean_X(
                X, self.feature_names_in_, self.feature_types_in_, n_samples
            )
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
                n_samples,
            )

        term_names = self.term_names_
        term_types = generate_term_types(self.feature_types_in_, self.term_features_)

        data_dicts = []
        perf_list = []
        if n_samples == 0:
            X_unified = np.empty((0, len(self.feature_names_in_)), dtype=np.object_)
        else:
            X_unified, _, _ = unify_data(
                X, n_samples, self.feature_names_in_, self.feature_types_in_, True
            )

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
                    "extra": {
                        "names": ["Intercept"],
                        "scores": [intercept],
                        "values": [1],
                    },
                }
                if is_classifier(self):
                    # Classes should be numpy array, convert to list.
                    data_dict["meta"] = {"label_names": self.classes_.tolist()}
                data_dicts.append(data_dict)

            for term_idx, bin_indexes in eval_terms(
                X,
                n_samples,
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.term_features_,
            ):
                scores = self.term_scores_[term_idx][tuple(bin_indexes)]
                feature_idxs = self.term_features_[term_idx]
                for row_idx in range(n_samples):
                    term_name = term_names[term_idx]
                    data_dicts[row_idx]["names"][term_idx] = term_name
                    data_dicts[row_idx]["scores"][term_idx] = scores[row_idx]
                    if len(feature_idxs) == 1:
                        data_dicts[row_idx]["values"][term_idx] = X_unified[
                            row_idx, feature_idxs[0]
                        ]
                    else:
                        data_dicts[row_idx]["values"][term_idx] = ""

            # TODO: handle the 1 class case here

            pred = ebm_decision_function(
                X,
                n_samples,
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.intercept_,
                self.term_scores_,
                self.term_features_,
                init_score,
            )
            n_classes = len(self.classes_) if is_classifier(self) else -1
            pred = inv_link(self.link_, self.link_param_, pred, n_classes)

            classes = self.classes_ if is_classifier(self) else None

            perf_dicts = gen_perf_dicts(pred, y, is_classifier(self), classes)
            for row_idx in range(n_samples):
                perf = None if perf_dicts is None else perf_dicts[row_idx]
                perf_list.append(perf)
                data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classifier(self))

        term_scores = remove_last(self.term_scores_, self.bin_weights_)
        for term_idx, feature_idxs in enumerate(self.term_features_):
            term_scores[term_idx] = trim_tensor(
                term_scores[term_idx], trim_low=[True] * len(feature_idxs)
            )

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
            feature_types=term_types,
            name=gen_name_from_class(self) if name is None else name,
            selector=selector,
        )

    def term_importances(self, importance_type="avg_weight"):
        """Provides the term importances

        Args:
            importance_type: the type of term importance requested ('avg_weight', 'min_max')

        Returns:
            An array term importances with one importance per additive term
        """

        check_is_fitted(self, "has_fitted_")

        if importance_type == "avg_weight":
            importances = np.empty(len(self.term_features_), np.float64)
            for i in range(len(self.term_features_)):
                if is_classifier(self):
                    mean_abs_score = (
                        0  # everything is useless if we're predicting 1 class
                    )
                    if 1 < len(self.classes_):
                        mean_abs_score = np.abs(self.term_scores_[i])
                        if 2 < len(self.classes_):
                            mean_abs_score = np.average(mean_abs_score, axis=-1)
                        mean_abs_score = np.average(
                            mean_abs_score, weights=self.bin_weights_[i]
                        )
                else:
                    mean_abs_score = np.abs(self.term_scores_[i])
                    mean_abs_score = np.average(
                        mean_abs_score, weights=self.bin_weights_[i]
                    )
                importances.itemset(i, mean_abs_score)
            return importances
        elif importance_type == "min_max":
            return np.array(
                [np.max(tensor) - np.min(tensor) for tensor in self.term_scores_],
                np.float64,
            )
        else:
            raise ValueError(f"Unrecognized importance_type: {importance_type}")

    def monotonize(self, term, increasing="auto"):
        """Adjusts a term to be monotone using isotonic regression.

        Args:
            term: Index or name of continuous univariate term to apply monotone constraints
            increasing: 'auto' or bool. 'auto' decides direction based on Spearman correlation estimate.

        Returns:
            Itself.
        """

        check_is_fitted(self, "has_fitted_")

        if is_classifier(self) and 2 < len(self.classes_):
            msg = "monotonize not supported for multiclass"
            _log.error(msg)
            raise ValueError(msg)

        if isinstance(term, str):
            term = self.term_names_.index(term)

        features = self.term_features_[term]
        if 2 <= len(features):
            msg = "monotonize only works on univariate feature terms"
            _log.error(msg)
            raise ValueError(msg)

        feature_idx = features[0]

        if self.feature_types_in_[feature_idx] not in ["continuous", "ordinal"]:
            msg = "monotonize only supported on ordered feature types"
            _log.error(msg)
            raise ValueError(msg)

        if increasing is None:
            increasing = "auto"
        elif increasing not in ["auto", True, False]:
            msg = "increasing must be 'auto', True, or False"
            _log.error(msg)
            raise ValueError(msg)

        # copy any fields we overwrite in case someone has a shalow copy of self
        term_scores = self.term_scores_.copy()
        scores = term_scores[feature_idx].copy()

        # the missing and unknown bins are not part of the continuous range
        y = scores[1:-1]
        x = np.arange(len(y), dtype=np.int64)

        weights = self.bin_weights_[feature_idx][1:-1]

        # Fit isotonic regression weighted by training data bin counts
        ir = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
        y = ir.fit_transform(x, y, sample_weight=weights)

        # re-center y. Throw away the intercept changes since the monotonize
        # operation shouldn't be allowed to change the overall model intercept
        y -= np.average(y, weights=weights)

        scores[1:-1] = y
        term_scores[feature_idx] = scores
        self.term_scores_ = term_scores

        bagged_scores = self.bagged_scores_.copy()
        standard_deviations = self.standard_deviations_.copy()

        # TODO: in the future we can apply monotonize to the individual outer bags in bagged_scores_
        #       and then re-compute standard_deviations_ and term_scores_ from the monotonized bagged scores.
        #       but first we need to do some testing to figure out if this gives a worse result than applying
        #       IsotonicRegression to the final model which should be more regularized
        bagged_scores[feature_idx] = None
        standard_deviations[feature_idx] = None

        self.bagged_scores_ = bagged_scores
        self.standard_deviations_ = standard_deviations

        return self


class ExplainableBoostingClassifier(EBMModel, ClassifierMixin, ExplainerMixin):
    """An Explainable Boosting Classifier

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. FeatureType can be:

            - `None`: Auto-detect
            - `'quantile'`: Continuous with equal density bins
            - `'rounded_quantile'`: Continuous with quantile bins, but the cut values are rounded when possible
            - `'uniform'`: Continuous with equal width bins
            - `'winsorized'`: Continuous with equal width bins, but the leftmost and rightmost cut are chosen by quantiles
            - `'continuous'`: Use the default binning for continuous features, which is 'quantile' currently
            - `[List of float]`: Continuous with specified cut values. Eg: [5.5, 8.75]
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]
            - `'ordinal'`: Ordinal categorical where the order is determined by sorting the feature strings
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names
    max_bins : int, default=256
        Max number of bins per feature for the main effects stage.
    max_interaction_bins : int, default=32
        Max number of bins per feature for interaction terms.
    interactions : int, float, or list of tuples of feature indices, default=10

        Interaction terms to be included in the model. Options are:

            - Integer (1 <= interactions): Count of interactions to be automatically selected
            - Percentage (interactions < 1.0): Determine the integer count of interactions by multiplying the number of features by this percentage
            - List of tuples: The tuples contain the indices of the features within the additive term
    exclude : 'mains' or list of tuples of feature indices|names, default=[]
        Features or terms to be excluded.
    validation_size : int or float, default=0.15

        Validation set size. Used for early stopping during boosting, and is needed to create outer bags.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Turns off early stopping. Outer bags have no utility. Error bounds will be eliminated
    outer_bags : int, default=8
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    inner_bags : int, default=0
        Number of inner bags. 0 turns off inner bagging.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    greediness : float, default=0.0
        Percentage of rounds where boosting is greedy instead of round-robin. Greedy rounds are intermixed with cyclic rounds.
    smoothing_rounds : int, default=0
        Number of initial highly regularized rounds to set the basic shape of the main effect feature graphs.
    max_rounds : int, default=5000
        Total number of boosting rounds with n_terms boosting steps per round.
    early_stopping_rounds : int, default=50
        Number of rounds with no improvement to trigger early stopping. 0 turns off
        early stopping and boosting will occur for exactly max_rounds.
    early_stopping_tolerance : float, default=1e-4
        Tolerance that dictates the smallest delta required to be considered an improvement.
    min_samples_leaf : int, default=2
        Minimum number of samples allowed in the leaves.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    objective : str, default="log_loss"
        The objective to optimize.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=42
        Random state. None uses device_random and generates non-repeatable sequences.

    Attributes
    ----------
    classes\\_ : array of bool, int, or unicode with shape ``(n_classes,)``
        The class labels.
    n_features_in\\_ : int
        Number of features.
    feature_names_in\\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    histogram_edges\\_ : List of None or array of float with shape ``(n_hist_edges,)``
        Per-feature list of the histogram edges. Categorical features contain None within the List
        at their feature index.
    histogram_weights\\_ : List of array of float with shape ``(n_hist_bins,)``
        Per-feature list of the total sample weights within each feature's histogram bins.
    unique_val_counts\\_ : array of int with shape ``(n_features,)``
        Per-feature count of the number of unique feature values.
    term_features\\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\\_ : List of str
        List of term names.
    bin_weights\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the total sample weights in each term's tensor bins.
    bagged_scores\\_ : List of array of float with shape ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the bagged model scores.
        The last dimension of length n_classes is dropped for binary classification.
    term_scores\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the model scores.
        The last dimension of length n_classes is dropped for binary classification.
    standard_deviations\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the standard deviations of the bagged model scores.
        The last dimension of length n_classes is dropped for binary classification.
    link\\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_classification", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    breakpoint_iteration\\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting rounds performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting rounds will be in breakpoint_iteration_[0],
        and the count of interaction boosting rounds will be in breakpoint_iteration_[1].
    intercept\\_ : array of float with shape ``(n_classes,)`` or ``(1,)``
        Intercept of the model. Binary classification is shape ``(1,)``, and multiclass is shape ``(n_classes,)``.
    """

    n_features_in_: int
    term_names_: List[str]
    bins_: List[Union[List[Dict[str, int]], List[np.ndarray]]]  # np.float64, 1D[cut]
    feature_names_in_: List[str]
    feature_types_in_: List[str]
    feature_bounds_: np.ndarray  # np.float64, 2D[feature, min_max]
    term_features_: List[Tuple[int, ...]]
    bin_weights_: List[np.ndarray]  # np.float64, [bin0...]
    bagged_scores_: List[np.ndarray]  # np.float64, [bag, bin0..., ?class]
    term_scores_: List[np.ndarray]  # np.float64, [bin0..., ?class]
    standard_deviations_: List[np.ndarray]  # np.float64, [bin0..., ?class]
    link_: str
    link_param_: float
    bag_weights_: np.ndarray  # np.float64, 1D[bag]
    breakpoint_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    histogram_edges_: List[Union[None, np.ndarray]]  # np.float64, 1D[hist_edge]
    histogram_weights_: List[np.ndarray]  # np.float64, 1D[hist_bin]
    unique_val_counts_: np.ndarray  # np.int64, 1D[feature]

    classes_: np.ndarray  # np.int64, np.bool_, or np.unicode_, 1D[class]
    intercept_: np.ndarray  # np.float64, 1D[class]

    # TODO PK v.3 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    # TODO: use Literal for the string types once everything is python 3.8
    def __init__(
        self,
        # Explainer
        feature_names: Optional[Sequence[Union[None, str]]] = None,
        feature_types: Optional[
            Sequence[Union[None, str, Sequence[str], Sequence[float]]]
        ] = None,
        # Preprocessor
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        # Stages
        interactions: Optional[
            Union[int, float, Sequence[Union[int, str, Sequence[Union[int, str]]]]]
        ] = 10,
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = [],
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0.15,
        outer_bags: int = 8,
        inner_bags: Optional[int] = 0,
        # Boosting
        learning_rate: float = 0.01,
        greediness: Optional[float] = 0.0,
        smoothing_rounds: Optional[int] = 0,
        max_rounds: Optional[int] = 5000,
        early_stopping_rounds: Optional[int] = 50,
        early_stopping_tolerance: Optional[float] = 1e-4,
        # Trees
        min_samples_leaf: Optional[int] = 2,
        max_leaves: int = 3,
        objective: str = "log_loss",
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = 42,
    ):
        super(ExplainableBoostingClassifier, self).__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greediness=greediness,
            smoothing_rounds=smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict_proba(self, X, init_score=None):
        """Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        if len(self.classes_) == 1:
            # if there is only one class then all probabilities are 100%
            return np.full((n_samples, 1), 1.0, np.float64)

        log_odds = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

        return inv_link(self.link_, self.link_param_, log_odds, len(self.classes_))

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        # TODO: handle the 1 class case here

        log_odds = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

        # TODO: for binary classification we could just look for values greater than zero instead of expanding
        if log_odds.ndim == 1:
            # Handle binary classification case -- softmax only works with 0s appended
            log_odds = np.c_[np.zeros(log_odds.shape), log_odds]

        return self.classes_[np.argmax(log_odds, axis=1)]

    def predict_and_contrib(self, X, output="probabilities", init_score=None):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.
            output: Prediction type to output (i.e. one of 'probabilities', 'labels', 'logits')
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predictions and local explanations for each sample.
        """

        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        # TODO: handle the 1 class case here

        scores, explanations = ebm_decision_function_and_explain(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

        if output == "probabilities":
            result = inv_link(self.link_, self.link_param_, scores, len(self.classes_))
        elif output == "labels":
            # TODO: for binary classification we could just look for values greater than zero instead of expanding
            if scores.ndim == 1:
                scores = np.c_[np.zeros(scores.shape), scores]
            result = self.classes_[np.argmax(scores, axis=1)]
        elif output == "logits":
            result = scores
        else:
            msg = f"Argument 'output' has invalid value. Got '{output}', expected 'probabilities', 'labels', or 'logits'"
            _log.error(msg)
            raise ValueError(msg)

        return result, explanations


class ExplainableBoostingRegressor(EBMModel, RegressorMixin, ExplainerMixin):
    """An Explainable Boosting Regressor

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. FeatureType can be:

            - `None`: Auto-detect
            - `'quantile'`: Continuous with equal density bins
            - `'rounded_quantile'`: Continuous with quantile bins, but the cut values are rounded when possible
            - `'uniform'`: Continuous with equal width bins
            - `'winsorized'`: Continuous with equal width bins, but the leftmost and rightmost cut are chosen by quantiles
            - `'continuous'`: Use the default binning for continuous features, which is 'quantile' currently
            - `[List of float]`: Continuous with specified cut values. Eg: [5.5, 8.75]
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]
            - `'ordinal'`: Ordinal categorical where the order is determined by sorting the feature strings
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names
    max_bins : int, default=256
        Max number of bins per feature for the main effects stage.
    max_interaction_bins : int, default=32
        Max number of bins per feature for interaction terms.
    interactions : int, float, or list of tuples of feature indices, default=10

        Interaction terms to be included in the model. Options are:

            - Integer (1 <= interactions): Count of interactions to be automatically selected
            - Percentage (interactions < 1.0): Determine the integer count of interactions by multiplying the number of features by this percentage
            - List of tuples: The tuples contain the indices of the features within the additive term
    exclude : 'mains' or list of tuples of feature indices|names, default=[]
        Features or terms to be excluded.
    validation_size : int or float, default=0.15

        Validation set size. Used for early stopping during boosting, and is needed to create outer bags.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Turns off early stopping. Outer bags have no utility. Error bounds will be eliminated
    outer_bags : int, default=8
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    inner_bags : int, default=0
        Number of inner bags. 0 turns off inner bagging.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    greediness : float, default=0.0
        Percentage of rounds where boosting is greedy instead of round-robin. Greedy rounds are intermixed with cyclic rounds.
    smoothing_rounds : int, default=0
        Number of initial highly regularized rounds to set the basic shape of the main effect feature graphs.
    max_rounds : int, default=5000
        Total number of boosting rounds with n_terms boosting steps per round.
    early_stopping_rounds : int, default=50
        Number of rounds with no improvement to trigger early stopping. 0 turns off
        early stopping and boosting will occur for exactly max_rounds.
    early_stopping_tolerance : float, default=1e-4
        Tolerance that dictates the smallest delta required to be considered an improvement.
    min_samples_leaf : int, default=2
        Minimum number of samples allowed in the leaves.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    objective : str, default="rmse"
        The objective to optimize. Options include: "rmse",
        "poisson_deviance", "tweedie_deviance:variance_power=1.5", "gamma_deviance",
        "pseudo_huber:delta=1.0", "rmse_log" (rmse with a log link function)
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=42
        Random state. None uses device_random and generates non-repeatable sequences.

    Attributes
    ----------
    n_features_in\\_ : int
        Number of features.
    feature_names_in\\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    histogram_edges\\_ : List of None or array of float with shape ``(n_hist_edges,)``
        Per-feature list of the histogram edges. Categorical features contain None within the List
        at their feature index.
    histogram_weights\\_ : List of array of float with shape ``(n_hist_bins,)``
        Per-feature list of the total sample weights within each feature's histogram bins.
    unique_val_counts\\_ : array of int with shape ``(n_features,)``
        Per-feature count of the number of unique feature values.
    term_features\\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\\_ : List of str
        List of term names.
    bin_weights\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the total sample weights in each term's tensor bins.
    bagged_scores\\_ : List of array of float with shape ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the bagged model scores.
    term_scores\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the model scores.
    standard_deviations\\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_regression", "power", "identity", "log", "inverse", "inverse_square", "sqrt"
    link_param\\_ : float
        Float value that can be used by the link function. The primary use is for the power link.
    bag_weights\\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    breakpoint_iteration\\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting rounds performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting rounds will be in breakpoint_iteration_[0],
        and the count of interaction boosting rounds will be in breakpoint_iteration_[1].
    intercept\\_ : float
        Intercept of the model.
    min_target\\_ : float
        The minimum value found in 'y'.
    max_target\\_ : float
        The maximum value found in 'y'.
    """

    n_features_in_: int
    term_names_: List[str]
    bins_: List[Union[List[Dict[str, int]], List[np.ndarray]]]  # np.float64, 1D[cut]
    feature_names_in_: List[str]
    feature_types_in_: List[str]
    feature_bounds_: np.ndarray  # np.float64, 2D[feature, min_max]
    term_features_: List[Tuple[int, ...]]
    bin_weights_: List[np.ndarray]  # np.float64, [bin0...]
    bagged_scores_: List[np.ndarray]  # np.float64, [bag, bin0...]
    term_scores_: List[np.ndarray]  # np.float64, [bin0...]
    standard_deviations_: List[np.ndarray]  # np.float64, [bin0...]
    link_: str
    link_param_: float
    bag_weights_: np.ndarray  # np.float64, 1D[bag]
    breakpoint_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    histogram_edges_: List[Union[None, np.ndarray]]  # np.float64, 1D[hist_edge]
    histogram_weights_: List[np.ndarray]  # np.float64, 1D[hist_bin]
    unique_val_counts_: np.ndarray  # np.int64, 1D[feature]

    intercept_: float
    min_target_: float
    max_target_: float

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names: Optional[Sequence[Union[None, str]]] = None,
        feature_types: Optional[
            Sequence[Union[None, str, Sequence[str], Sequence[float]]]
        ] = None,
        # Preprocessor
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        # Stages
        interactions: Optional[
            Union[int, float, Sequence[Union[int, str, Sequence[Union[int, str]]]]]
        ] = 10,
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = [],
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0.15,
        outer_bags: int = 8,
        inner_bags: Optional[int] = 0,
        # Boosting
        learning_rate: float = 0.01,
        greediness: Optional[float] = 0.0,
        smoothing_rounds: Optional[int] = 0,
        max_rounds: Optional[int] = 5000,
        early_stopping_rounds: Optional[int] = 50,
        early_stopping_tolerance: Optional[float] = 1e-4,
        # Trees
        min_samples_leaf: Optional[int] = 2,
        max_leaves: int = 3,
        objective: str = "rmse",
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = 42,
    ):
        super(ExplainableBoostingRegressor, self).__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            interactions=interactions,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            greediness=greediness,
            smoothing_rounds=smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        scores = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )
        return inv_link(self.link_, self.link_param_, scores, -1)

    def predict_and_contrib(self, X, init_score=None):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predictions and local explanations for each sample.
        """

        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        scores, explanations = ebm_decision_function_and_explain(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )
        return inv_link(self.link_, self.link_param_, scores, -1), explanations


class DPExplainableBoostingClassifier(EBMModel, ClassifierMixin, ExplainerMixin):
    """Differentially Private Explainable Boosting Classifier. Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `None`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'ordinal'`: Ordinal categorical where the order is determined by sorting the feature strings. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=[]
        Features to be excluded.
    validation_size : int or float, default=0

        Validation set size. A validation set is needed if outer bags or error bars are desired.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Outer bags have no utility and error bounds will be eliminated
    outer_bags : int, default=1
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_rounds : int, default=300
        Total number of boosting rounds with n_terms boosting steps per round.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=None
        Random state. None uses device_random and generates non-repeatable sequences.
        Should be set to 'None' for privacy, but can be set to an integer for testing and repeatability.
    epsilon: float, default=1.0
        Total privacy budget to be spent.
    delta: float, default=1e-5
        Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
    composition: {'gdp', 'classic'}, default='gdp'
        Method of tracking noise aggregation.
    bin_budget_frac: float, default=0.1
        Percentage of total epsilon budget to use for private binning.
    privacy_bounds: Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]], default=None
        Specifies known min/max values for each feature.
        If None, DP-EBM shows a warning and uses the data to determine these values.

    Attributes
    ----------
    classes\\_ : array of bool, int, or unicode with shape ``(2,)``
        The class labels. DPExplainableBoostingClassifier only supports binary classification, so there are 2 classes.
    n_features_in\\_ : int
        Number of features.
    feature_names_in\\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\\_ : List of str
        List of term names.
    bin_weights\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_classification", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    breakpoint_iteration\\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting rounds performed within each stage. Normally, the count of main effects
        boosting rounds will be in breakpoint_iteration_[0].
    intercept\\_ : array of float with shape ``(1,)``
        Intercept of the model.
    noise_scale_binning\\_ : float
        The noise scale during binning.
    noise_scale_boosting\\_ : float
        The noise scale during boosting.
    """

    n_features_in_: int
    term_names_: List[str]
    bins_: List[Union[List[Dict[str, int]], List[np.ndarray]]]  # np.float64, 1D[cut]
    feature_names_in_: List[str]
    feature_types_in_: List[str]
    feature_bounds_: np.ndarray  # np.float64, 2D[feature, min_max]
    term_features_: List[Tuple[int, ...]]
    bin_weights_: List[np.ndarray]  # np.float64, [bin]
    bagged_scores_: List[np.ndarray]  # np.float64, [bag, bin]
    term_scores_: List[np.ndarray]  # np.float64, [bin]
    standard_deviations_: List[np.ndarray]  # np.float64, [bin]
    link_: str
    link_param_: float
    bag_weights_: np.ndarray  # np.float64, 1D[bag]
    breakpoint_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    noise_scale_binning_: float
    noise_scale_boosting_: float

    classes_: np.ndarray  # np.int64, np.bool_, or np.unicode_, 1D[class]
    intercept_: np.ndarray  # np.float64, 1D[class]

    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing DPEBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names: Optional[Sequence[Union[None, str]]] = None,
        feature_types: Optional[
            Sequence[Union[None, str, Sequence[str], Sequence[float]]]
        ] = None,
        # Preprocessor
        max_bins: int = 32,
        # Stages
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = [],
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0,
        outer_bags: int = 1,
        # Boosting
        learning_rate: float = 0.01,
        max_rounds: Optional[int] = 300,
        # Trees
        max_leaves: int = 3,
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = None,
        # Differential Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        composition: str = "gdp",
        bin_budget_frac: float = 0.1,
        privacy_bounds: Optional[
            Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]]
        ] = None,
    ):
        super(DPExplainableBoostingClassifier, self).__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=None,
            interactions=0,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=0,
            learning_rate=learning_rate,
            greediness=0.0,
            smoothing_rounds=0,
            max_rounds=max_rounds,
            early_stopping_rounds=0,
            early_stopping_tolerance=0.0,
            min_samples_leaf=0,
            max_leaves=max_leaves,
            objective="log_loss",
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_bounds=privacy_bounds,
        )

    def predict_proba(self, X, init_score=None):
        """Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        if len(self.classes_) == 1:
            # if there is only one class then all probabilities are 100%
            return np.full((n_samples, 1), 1.0, np.float64)

        log_odds = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )
        return inv_link(self.link_, self.link_param_, log_odds, len(self.classes_))

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        # TODO: handle the 1 class case here

        log_odds = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

        # TODO: for binary classification we could just look for values greater than zero instead of expanding
        if log_odds.ndim == 1:
            # Handle binary classification case -- softmax only works with 0s appended
            log_odds = np.c_[np.zeros(log_odds.shape), log_odds]

        return self.classes_[np.argmax(log_odds, axis=1)]


class DPExplainableBoostingRegressor(EBMModel, RegressorMixin, ExplainerMixin):
    """Differentially Private Explainable Boosting Regressor. Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `None`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'ordinal'`: Ordinal categorical where the order is determined by sorting the feature strings. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=[]
        Features to be excluded.
    validation_size : int or float, default=0

        Validation set size. A validation set is needed if outer bags or error bars are desired.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Outer bags have no utility and error bounds will be eliminated
    outer_bags : int, default=1
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    learning_rate : float, default=0.01
        Learning rate for boosting.
    max_rounds : int, default=300
        Total number of boosting rounds with n_terms boosting steps per round.
    max_leaves : int, default=3
        Maximum number of leaves allowed in each tree.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=None
        Random state. None uses device_random and generates non-repeatable sequences.
        Should be set to 'None' for privacy, but can be set to an integer for testing and repeatability.
    epsilon: float, default=1.0
        Total privacy budget to be spent.
    delta: float, default=1e-5
        Additive component of differential privacy guarantee. Should be smaller than 1/n_training_samples.
    composition: {'gdp', 'classic'}, default='gdp'
        Method of tracking noise aggregation.
    bin_budget_frac: float, default=0.1
        Percentage of total epsilon budget to use for private binning.
    privacy_bounds: Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]], default=None
        Specifies known min/max values for each feature.
        If None, DP-EBM shows a warning and uses the data to determine these values.
    privacy_target_min: float, default=None
        Known target minimum. 'y' values will be clipped to this min.
        If None, DP-EBM shows a warning and uses the data to determine this value.
    privacy_target_max: float, default=None
        Known target maximum. 'y' values will be clipped to this max.
        If None, DP-EBM shows a warning and uses the data to determine this value.

    Attributes
    ----------
    n_features_in\\_ : int
        Number of features.
    feature_names_in\\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\\_ : List of str
        List of term names.
    bin_weights\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_regression", "power", "identity", "log", "inverse", "inverse_square", "sqrt"
    link_param\\_ : float
        Float value that can be used by the link function. The primary use is for the power link.
    bag_weights\\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    breakpoint_iteration\\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting rounds performed within each stage. Normally, the count of main effects
        boosting rounds will be in breakpoint_iteration_[0].
    intercept\\_ : float
        Intercept of the model.
    min_target\\_ : float
        The minimum value found in 'y', or privacy_target_min if provided.
    max_target\\_ : float
        The maximum value found in 'y', or privacy_target_max if provided.
    noise_scale_binning\\_ : float
        The noise scale during binning.
    noise_scale_boosting\\_ : float
        The noise scale during boosting.
    """

    n_features_in_: int
    term_names_: List[str]
    bins_: List[Union[List[Dict[str, int]], List[np.ndarray]]]  # np.float64, 1D[cut]
    feature_names_in_: List[str]
    feature_types_in_: List[str]
    feature_bounds_: np.ndarray  # np.float64, 2D[feature, min_max]
    term_features_: List[Tuple[int, ...]]
    bin_weights_: List[np.ndarray]  # np.float64, [bin0...]
    bagged_scores_: List[np.ndarray]  # np.float64, [bag, bin0..., ?class]
    term_scores_: List[np.ndarray]  # np.float64, [bin0..., ?class]
    standard_deviations_: List[np.ndarray]  # np.float64, [bin0..., ?class]
    link_: str
    link_param_: float
    bag_weights_: np.ndarray  # np.float64, 1D[bag]
    breakpoint_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    noise_scale_binning_: float
    noise_scale_boosting_: float

    intercept_: float
    min_target_: float
    max_target_: float

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing DPEBM regressor."""

    def __init__(
        self,
        # Explainer
        feature_names: Optional[Sequence[Union[None, str]]] = None,
        feature_types: Optional[
            Sequence[Union[None, str, Sequence[str], Sequence[float]]]
        ] = None,
        # Preprocessor
        max_bins: int = 32,
        # Stages
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = [],
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0,
        outer_bags: int = 1,
        # Boosting
        learning_rate: float = 0.01,
        max_rounds: Optional[int] = 300,
        # Trees
        max_leaves: int = 3,
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = None,
        # Differential Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        composition: str = "gdp",
        bin_budget_frac: float = 0.1,
        privacy_bounds: Optional[
            Union[np.ndarray, Mapping[Union[int, str], Tuple[float, float]]]
        ] = None,
        privacy_target_min: Optional[float] = None,
        privacy_target_max: Optional[float] = None,
    ):
        super(DPExplainableBoostingRegressor, self).__init__(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=max_bins,
            max_interaction_bins=None,
            interactions=0,
            exclude=exclude,
            validation_size=validation_size,
            outer_bags=outer_bags,
            inner_bags=0,
            learning_rate=learning_rate,
            greediness=0.0,
            smoothing_rounds=0,
            max_rounds=max_rounds,
            early_stopping_rounds=0,
            early_stopping_tolerance=0.0,
            min_samples_leaf=0,
            max_leaves=max_leaves,
            objective="rmse",
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_bounds=privacy_bounds,
            privacy_target_min=privacy_target_min,
            privacy_target_max=privacy_target_max,
        )

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")

        if init_score is None:
            X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)
        else:
            init_score, X, n_samples = clean_init_score_and_X(
                self.link_,
                self.link_param_,
                init_score,
                X,
                self.feature_names_in_,
                self.feature_types_in_,
            )

        scores = ebm_decision_function(
            X,
            n_samples,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
            init_score,
        )

        return inv_link(self.link_, self.link_param_, scores, -1)
