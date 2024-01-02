# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license


from typing import Optional, List, Tuple, Sequence, Dict, Mapping, Union
from copy import deepcopy

from itertools import count

import os
from ...utils._explanation import gen_perf_dicts
from ._boost import boost
from ._utils import (
    make_bag,
    process_terms,
    order_terms,
    remove_unused_higher_bins,
    deduplicate_bins,
    generate_term_names,
    generate_term_types,
)
from ._json import to_jsonable, UNTESTED_from_jsonable

from ...utils._misc import clean_index, clean_indexes
from ...utils._histogram import make_all_histogram_edges
from ...utils._link import link_func, inv_link
from ...utils._seed import normalize_seed
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
    ebm_predict_scores,
    ebm_eval_terms,
    make_bin_weights,
)
from ._tensor import remove_last, trim_tensor
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

from sklearn.base import is_classifier, is_regressor  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from sklearn.isotonic import IsotonicRegression

import heapq
import operator

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)  # type: ignore
from itertools import combinations

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

    def fit(self, X, y, sample_weight=None, bags=None, init_score=None):  # noqa: C901
        """Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.
            bags: Optional bag definitions. The first dimension should have length equal to the number of outer_bags.
                The second dimension should have length equal to the number of samples. The contents should be
                +1 for training, -1 for validation, and 0 if not included in the bag. Numbers other than 1 indicate
                how many times to include the sample in the training or validation sets.
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

        if bags is not None:
            if len(bags) != self.outer_bags:
                msg = f"bags has {len(bags)} bags and self.outer_bags is {self.outer_bags} bags"
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

        native = Native.get_native_singleton()

        objective = self.objective
        task = None
        if objective is not None:
            if len(objective.strip()) == 0:
                objective = None
            else:
                # "classification" or "regression"
                task = native.determine_task(objective)

        if is_classifier(self):
            if task is None:
                task = "classification"
            elif task != "classification":
                msg = f"classifier cannot have objective {self.objective}"
                _log.error(msg)
                raise ValueError(msg)

        if is_regressor(self):
            if task is None:
                task = "regression"
            elif task != "regression":
                msg = f"regressor cannot have objective {self.objective}"
                _log.error(msg)
                raise ValueError(msg)

        if task == "classification":
            y = typify_classification(y)
            # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
            # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
            # in two separate runs, which would flip the ordering of the classes within our score tensors.
            classes, y = np.unique(y, return_inverse=True)
            n_classes = len(classes)
            if objective is None:
                objective = "log_loss"
        elif task == "regression":
            y = y.astype(np.float64, copy=False)
            min_target = y.min()
            max_target = y.max()
            n_classes = Native.Task_Regression
            if objective is None:
                objective = "rmse"
        else:
            msg = f"Unrecognized objective {self.objective}"
            _log.error(msg)
            raise ValueError(msg)

        n_scores = Native.get_count_scores_c(n_classes)

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

        flags = (
            Native.LinkFlags_DifferentialPrivacy
            if is_differential_privacy
            else Native.LinkFlags_Default
        )
        link, link_param = native.determine_link(flags, objective, n_classes)

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

            if n_classes < 0:
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
            elif 2 != n_classes:  # pragma: no cover
                raise ValueError(
                    "Multiclass not supported for Differentially Private EBMs."
                )

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

        seed = normalize_seed(self.random_state)

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
            seed=seed,
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

        # branch so we have no correlation to the binning rng that uses the same seed
        rng = native.branch_rng(native.create_rng(seed))
        used_seeds = set()
        rngs = []
        internal_bags = []
        for idx in range(self.outer_bags):
            while True:
                bagged_rng = native.branch_rng(rng)
                check_seed = native.generate_seed(bagged_rng)
                # We do not want identical bags. branch_rng is pretty good at avoiding
                # collisions, but it is not a cryptographic RNG, so it is possible.
                # Check with a 32-bit seed if we have a collision and regenerate if so.
                if check_seed not in used_seeds:
                    break
            used_seeds.add(check_seed)

            if bags is None:
                bag = make_bag(
                    y,
                    self.validation_size,
                    bagged_rng,
                    0 <= n_classes and not is_differential_privacy,
                )
            else:
                bag = bags[idx]
                if not isinstance(bag, np.ndarray):
                    bag = np.array(bag)
                if bag.ndim != 1:
                    msg = "bags must be 2-dimensional"
                    _log.error(msg)
                    raise ValueError(msg)
                if len(y) != len(bag):
                    msg = f"y has {len(y)} samples and bags has {len(bag)} samples"
                    _log.error(msg)
                    raise ValueError(msg)
                if (127 < bag).any() or (bag < -128).any():
                    msg = "A value in bags is outside the valid range -128 to 127"
                    _log.error(msg)
                    raise ValueError(msg)
                bag = bag.astype(np.int8, copy=not bag.flags.c_contiguous)

            rngs.append(bagged_rng)
            internal_bags.append(bag)

        bag_weights = []
        for bag in internal_bags:
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
                del keep
        bag_weights = np.array(bag_weights, np.float64)

        if is_differential_privacy:
            # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            domain_size = 1 if 0 <= n_classes else max_target - min_target
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

            bin_data_weights = main_bin_weights
            term_boost_flags = (
                Native.TermBoostFlags_GradientSums | Native.TermBoostFlags_RandomSplits
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
            term_boost_flags = Native.TermBoostFlags_Default
            inner_bags = self.inner_bags
            greediness = self.greediness
            smoothing_rounds = self.smoothing_rounds
            early_stopping_rounds = self.early_stopping_rounds
            early_stopping_tolerance = self.early_stopping_tolerance
            min_samples_leaf = self.min_samples_leaf
            interactions = self.interactions

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
            bag = internal_bags[idx]
            if bag is None or (0 <= bag).all():
                # if there are no validation samples, turn off early stopping
                # because the validation metric cannot improve each round
                early_stopping_rounds_local = 0

            init_score_local = init_score
            if (
                init_score_local is not None
                and bag is not None
                and np.count_nonzero(bag) != len(bag)
            ):
                # TODO: instead of making these copies we should
                # put init_score into the native shared dataframe
                init_score_local = init_score_local[bag != 0]

            parallel_args.append(
                (
                    dataset,
                    bag,
                    init_score_local,
                    term_features,
                    inner_bags,
                    term_boost_flags,
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
                    Native.CreateBoosterFlags_DifferentialPrivacy
                    if is_differential_privacy
                    else Native.CreateBoosterFlags_Default,
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
        for exception, model, bag_breakpoint_iteration, bagged_rng in results:
            if exception is not None:
                raise exception
            breakpoint_iteration[-1].append(bag_breakpoint_iteration)
            models.append(model)
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
                        "Multiclass interactions only have local explanations. "
                        "They are not currently displayed in the global explanation "
                        "visualizations. Set interactions=0 to disable this warning. "
                        "If you still want multiclass interactions, this API accepts "
                        "a list, and the measure_interactions function can be used to "
                        "detect them."
                    )
                    break

                # at this point interactions will be a positive, nonzero integer
            else:
                # interactions must be a list of the interactions
                if len(interactions) == 0:
                    break

            initial_intercept = np.zeros(n_scores, np.float64)
            scores_bags = []
            for model, bag in zip(models, internal_bags):
                # TODO: instead of going back to the original data in X, we
                # could use the compressed and already binned data in dataset
                scores = ebm_predict_scores(
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
                if bag is not None and np.count_nonzero(bag) != len(bag):
                    scores = scores[bag != 0]
                scores_bags.append(scores)

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
                            internal_bags[idx],
                            scores_bags[idx],
                            combinations(range(n_features_in), 2),
                            exclude,
                            Native.CalcInteractionFlags_Default,
                            max_cardinality,
                            min_samples_leaf,
                            Native.CreateInteractionFlags_DifferentialPrivacy
                            if is_differential_privacy
                            else Native.CreateInteractionFlags_Default,
                            objective,
                            None,
                        )
                    )

                bagged_ranked_interaction = provider.parallel(
                    rank_interactions, parallel_args
                )

                # this holds references to dataset, internal_bags, and scores_bags which we want python to reclaim later
                del parallel_args

                # Select merged pairs
                pair_ranks = {}
                for n, interaction_strengths_and_indices in enumerate(
                    bagged_ranked_interaction
                ):
                    if isinstance(interaction_strengths_and_indices, Exception):
                        raise interaction_strengths_and_indices

                    interaction_indices = list(
                        map(
                            operator.itemgetter(1),
                            interaction_strengths_and_indices,
                        )
                    )
                    for rank, indices in enumerate(interaction_indices):
                        old_mean = pair_ranks.get(indices, 0)
                        pair_ranks[indices] = old_mean + ((rank - old_mean) / (n + 1))

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
                    if sorted_tuple not in uniquifier and sorted_tuple not in exclude:
                        uniquifier.add(sorted_tuple)
                        boost_groups.append(feature_idxs)

                if 2 < max_dimensions:
                    warn(
                        "Interactions with 3 or more terms are not graphed in "
                        "global explanations. Local explanations are still "
                        "available and exact."
                    )

            parallel_args = []
            for idx in range(self.outer_bags):
                early_stopping_rounds_local = early_stopping_rounds
                if internal_bags[idx] is None or (0 <= internal_bags[idx]).all():
                    # if there are no validation samples, turn off early stopping
                    # because the validation metric cannot improve each round
                    early_stopping_rounds_local = 0

                parallel_args.append(
                    (
                        dataset,
                        internal_bags[idx],
                        scores_bags[idx],
                        boost_groups,
                        inner_bags,
                        term_boost_flags,
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
                        Native.CreateBoosterFlags_DifferentialPrivacy
                        if is_differential_privacy
                        else Native.CreateBoosterFlags_Default,
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
                if results[idx][0] is not None:
                    raise results[idx][0]
                breakpoint_iteration[-1].append(results[idx][2])
                models[idx].extend(results[idx][1])
                rngs[idx] = results[idx][3]

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

        if n_scores == 1:
            bagged_intercept = np.zeros(self.outer_bags, np.float64)
        else:
            bagged_intercept = np.zeros((self.outer_bags, n_scores), np.float64)

        intercept, term_scores, standard_deviations = process_terms(
            bagged_intercept, bagged_scores, bin_weights, bag_weights
        )
        if n_classes < 0:
            # scikit-learn uses a float for regression, and a numpy array with 1 element for binary classification
            intercept = float(intercept[0])

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
        self.bagged_intercept_ = bagged_intercept
        self.link_ = link
        self.link_param_ = link_param
        self.bag_weights_ = bag_weights
        self.breakpoint_iteration_ = breakpoint_iteration
        self.has_fitted_ = True

        return self

    def to_jsonable(self, detail="all"):
        """Converts the model to a JSONable representation.

        Args:
            detail: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSONable object
        """

        check_is_fitted(self, "has_fitted_")

        return to_jsonable(self, detail)

    def to_json(self, file, detail="all", indent=2):
        """Exports the model to a JSON text file.

        Args:
            file: a path-like object (str or os.PathLike),
                or a file-like object implementing .write().
            detail: 'minimal', 'interpretable', 'mergeable', 'all'
            indent: If indent is a non-negative integer or string, then JSON array
                elements and object members will be pretty-printed with that indent
                level. An indent level of 0, negative, or "" will only insert newlines.
                None (the default) selects the most compact representation. Using a
                positive integer indent indents that many spaces per level. If indent
                is a string (such as "\t"), that string is used to indent each level.
        """

        check_is_fitted(self, "has_fitted_")

        if isinstance(file, (str, os.PathLike)):
            # file is a path-like object (str or os.PathLike)
            outer = to_jsonable(self, detail)
            with open(file, "w") as fp:
                json.dump(outer, fp, allow_nan=False, indent=indent)
        else:
            # file is a file-like object implementing .write()
            outer = to_jsonable(self, detail)
            json.dump(outer, file, allow_nan=False, indent=indent)

    def _from_jsonable(self, jsonable):
        """Converts a JSONable EBM representation into an EBM.

        Args:
            jsonable: the JSONable object

        Returns:
            Itself after de-JSONifying.
        """

        UNTESTED_from_jsonable(self, jsonable)
        return self

    def _from_json(self, file):
        """Loads from a JSON EBM file.

        Args:
            file: a path-like object (str or os.PathLike),
                or a file-like object implementing .read().

        Returns:
            Itself after loading.
        """

        if isinstance(file, (str, os.PathLike)):
            # file is a path-like object (str or os.PathLike)
            with open(file, "r") as fp:
                jsonable = json.load(fp)
            UNTESTED_from_jsonable(self, jsonable)
        else:
            # file is a file-like object implementing .read()
            jsonable = json.load(fp)
            UNTESTED_from_jsonable(self, jsonable)
        return self

    def to_excel(self, file):
        """Exports the model to an Excel workbook.

        Args:
            file: a path-like object (str or os.PathLike),
                or a file-like object implementing .write().
        """

        raise NotImplementedError()

    def _predict_score(self, X, init_score=None):
        """Predicts scores on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")

        init_score, X, n_samples = clean_init_score_and_X(
            self.link_,
            self.link_param_,
            init_score,
            X,
            self.feature_names_in_,
            self.feature_types_in_,
        )

        return ebm_predict_scores(
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

    def eval_terms(self, X):
        """The term scores returned will be identical to the local explanation values
        obtained by calling ebm.explain_local(X). Calling
        interpret.utils.inv_link(ebm.eval_terms(X).sum(axis=1) + ebm.intercept\\_, ebm.link\\_)
        is equivalent to calling ebm.predict(X) for regression or ebm.predict_proba(X) for classification.

        Args:
            X: Numpy array for samples.

        Returns:
            local explanation scores for each term of each sample.
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)

        n_scores = 1 if isinstance(self.intercept_, float) else len(self.intercept_)

        explanations = ebm_eval_terms(
            X,
            n_samples,
            n_scores,
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.term_scores_,
            self.term_features_,
        )

        return explanations

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
                if hasattr(self, "classes_"):
                    # Classes should be numpy array, convert to list.
                    data_dict["meta"] = {"label_names": self.classes_.tolist()}

                data_dicts.append(data_dict)
            elif len(feature_idxs) == 2:
                if hasattr(self, "classes_") and 2 != len(self.classes_):
                    warn(
                        f"Dropping term {term_names[term_idx]} from explanation "
                        "since we can't graph multinomial interactions."
                    )
                else:
                    keep_idxs.append(term_idx)

                    bin_levels = self.bins_[feature_idxs[0]]
                    feature_bins = bin_levels[
                        min(len(feature_idxs), len(bin_levels)) - 1
                    ]
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
                    feature_bins = bin_levels[
                        min(len(feature_idxs), len(bin_levels)) - 1
                    ]
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
                    f"Dropping term {term_names[term_idx]} from explanation "
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

        classes = getattr(self, "classes_", None)

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

            if classes is not None:
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        init_score, X, n_samples = clean_init_score_and_X(
            self.link_,
            self.link_param_,
            init_score,
            X,
            self.feature_names_in_,
            self.feature_types_in_,
            n_samples,
        )

        data_dicts = []
        perf_list = []
        if n_samples == 0:
            X_unified = np.empty((0, len(self.feature_names_in_)), dtype=np.object_)
        else:
            X_unified, _, _ = unify_data(
                X, n_samples, self.feature_names_in_, self.feature_types_in_, True
            )

            intercept = self.intercept_
            if classes is None or len(classes) == 2:
                if isinstance(intercept, np.ndarray) or isinstance(intercept, list):
                    intercept = intercept[0]

            n_scores = 1 if isinstance(self.intercept_, float) else len(self.intercept_)

            explanations = ebm_eval_terms(
                X,
                n_samples,
                n_scores,
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.term_scores_,
                self.term_features_,
            )
            scores = explanations.sum(axis=1) + intercept
            if init_score is not None:
                scores += init_score
            pred = inv_link(scores, self.link_, self.link_param_)

            perf_dicts = gen_perf_dicts(pred, y, classes is not None, classes)
            for row_idx in range(n_samples):
                perf = None if perf_dicts is None else perf_dicts[row_idx]
                perf_list.append(perf)

            for data, sample_scores, perf in zip(X_unified, explanations, perf_list):
                values = [
                    data[tfs[0]] if len(tfs) == 1 else "" for tfs in self.term_features_
                ]
                data_dict = {
                    "type": "univariate",
                    "names": list(self.term_names_),
                    "scores": list(sample_scores),
                    "values": values,
                    "extra": {
                        "names": ["Intercept"],
                        "scores": [intercept],
                        "values": [1],
                    },
                    "perf": perf,
                }
                if classes is not None:
                    # Classes should be numpy array, convert to list.
                    data_dict["meta"] = {"label_names": classes.tolist()}
                data_dicts.append(data_dict)

        selector = gen_local_selector(data_dicts, is_classification=classes is not None)

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

        term_types = generate_term_types(self.feature_types_in_, self.term_features_)
        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=list(self.term_names_),
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
                if hasattr(self, "classes_"):
                    # everything is useless if we're predicting 1 class
                    mean_abs_score = 0
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
            # TODO: handle mono-classification
            return np.array(
                [np.max(tensor) - np.min(tensor) for tensor in self.term_scores_],
                np.float64,
            )
        else:
            raise ValueError(f"Unrecognized importance_type: {importance_type}")

    def copy(self):
        """Makes a deepcopy of the EBM.

        Args:

        Returns:
            The new copy.
        """

        return deepcopy(self)

    def monotonize(self, term, increasing="auto", passthrough=0.0):
        """Adjusts a term to be monotone using isotonic regression. An important consideration
        is that this function only adjusts a single term and will not modify pairwise terms.
        When a feature needs to be globally monotonic, any pairwise terms that include the feature
        should be excluded from the model.

        Args:
            term: Index or name of the term to monotonize
            increasing: 'auto' or bool. 'auto' decides direction based on Spearman correlation estimate.
            passthrough: the process of monotonization can result in a change to the mean response
                of the model. If passthrough is set to 0.0 then the model's mean response to the
                training set will not change. If passthrough is set to 1.0 then any change to the
                mean response made by monotonization will be passed through to self.intercept\\_.
                Values between 0 and 1 will result in that percentage being passed through.

        Returns:
            Itself.
        """

        check_is_fitted(self, "has_fitted_")

        if hasattr(self, "classes_") and len(self.classes_) != 2:
            msg = "monotonize not supported for multiclass"
            _log.error(msg)
            raise ValueError(msg)

        term = clean_index(
            term,
            len(self.term_features_),
            getattr(self, "term_names_", None),
            "term",
            "self.term_names_",
        )

        features = self.term_features_[term]
        if 2 <= len(features):
            msg = "monotonize only works on univariate feature terms"
            _log.error(msg)
            raise ValueError(msg)

        if self.feature_types_in_[features[0]] not in ["continuous", "ordinal"]:
            msg = "monotonize only supported on ordered feature types"
            _log.error(msg)
            raise ValueError(msg)

        if increasing is None:
            increasing = "auto"
        elif increasing not in ["auto", True, False]:
            msg = "increasing must be 'auto', True, or False"
            _log.error(msg)
            raise ValueError(msg)

        if passthrough < 0.0 or 1.0 < passthrough:
            msg = "passthrough must be between 0.0 and 1.0 inclusive"
            _log.error(msg)
            raise ValueError(msg)

        # the missing and unknown bins are not part of the continuous range
        y = self.term_scores_[term][1:-1]
        x = np.arange(len(y), dtype=np.int64)

        all_weights = self.bin_weights_[term]
        weights = all_weights[1:-1]

        # this should normally be zero, except if there are missing or unknown values
        original_mean = np.average(y, weights=weights)

        # Fit isotonic regression weighted by training data bin counts
        ir = IsotonicRegression(increasing=increasing)
        y = ir.fit_transform(x, y, sample_weight=weights)

        result_mean = np.average(y, weights=weights)
        change = (original_mean - result_mean) * (1.0 - passthrough)
        y += change

        self.term_scores_[term][1:-1] = y

        if 0.0 < passthrough:
            mean = np.average(self.term_scores_[term], weights=all_weights)
            self.term_scores_[term] -= mean
            self.intercept_ += mean

        # TODO: in the future we can apply monotonize to the individual outer bags in bagged_scores_
        #       and then re-compute standard_deviations_ and term_scores_ from the monotonized bagged scores.
        #       but first we need to do some testing to figure out if this gives a worse result than applying
        #       IsotonicRegression to the final model which should be more regularized
        self.bagged_intercept_ = None
        self.bagged_scores_[term] = None
        self.standard_deviations_[term] = None

        return self

    def remove_terms(self, terms):
        """Removes terms (and their associated components) from a fitted EBM. Note
        that this will change the structure (i.e., by removing the specified
        indices) of the following components of ``self``: ``term_features_``,
        ``term_names_``, ``term_scores_``, ``bagged_scores_``,
        ``standard_deviations_``, and ``bin_weights_``.

        Args:
            terms: A list (or other enumerable object) of term names or indices or booleans.

        Returns:
            Itself.
        """
        check_is_fitted(self, "has_fitted_")

        # If terms contains term names, convert them to indices
        terms = clean_indexes(
            terms,
            len(self.term_features_),
            getattr(self, "term_names_", None),
            "terms",
            "self.term_names_",
        )

        def _remove_indices(x, idx):
            # Remove elements of a list based on provided index
            return [i for j, i in enumerate(x) if j not in idx]

        term_features = _remove_indices(self.term_features_, idx=terms)
        term_names = _remove_indices(self.term_names_, idx=terms)
        term_scores = _remove_indices(self.term_scores_, idx=terms)
        bagged_scores = _remove_indices(self.bagged_scores_, idx=terms)
        standard_deviations = _remove_indices(self.standard_deviations_, idx=terms)
        bin_weights = _remove_indices(self.bin_weights_, idx=terms)

        # Update components of self
        self.term_features_ = term_features
        self.term_names_ = term_names
        self.term_scores_ = term_scores
        self.bagged_scores_ = bagged_scores
        self.standard_deviations_ = standard_deviations
        self.bin_weights_ = bin_weights

        return self

    def remove_features(self, features):
        """Removes features (and their associated components) from a fitted EBM. Note
        that this will change the structure (i.e., by removing the specified
        indices) of the following components of ``self``: ``histogram_edges_``,
        ``histogram_weights_``, ``unique_val_counts_``, ``bins_``,
        ``feature_names_in_``, ``feature_types_in_``, and ``feature_bounds_``.
        Also, any terms that use the features being deleted will be deleted.
        The following attributes that the caller passed to the \\_\\_init\\_\\_ function are
        not modified: ``feature_names``, and ``feature_types``.

        Args:
            features: A list or enumerable of feature names or indices or
                booleans indicating which features to remove.

        Returns:
            Itself.
        """

        check_is_fitted(self, "has_fitted_")

        drop_features = clean_indexes(
            features,
            len(self.bins_),
            getattr(self, "feature_names_in_", None),
            "features",
            "self.feature_names_in_",
        )

        drop_terms = [
            term_idx
            for term_idx, feature_idxs in enumerate(self.term_features_)
            if any(feature_idx in drop_features for feature_idx in feature_idxs)
        ]
        self.remove_terms(drop_terms)

        self.histogram_edges_ = [
            v for i, v in enumerate(self.histogram_edges_) if i not in drop_features
        ]
        self.histogram_weights_ = [
            v for i, v in enumerate(self.histogram_weights_) if i not in drop_features
        ]
        self.bins_ = [v for i, v in enumerate(self.bins_) if i not in drop_features]
        self.feature_names_in_ = [
            v for i, v in enumerate(self.feature_names_in_) if i not in drop_features
        ]
        self.feature_types_in_ = [
            v for i, v in enumerate(self.feature_types_in_) if i not in drop_features
        ]

        drop_features = list(drop_features)
        self.unique_val_counts_ = np.delete(self.unique_val_counts_, drop_features)
        self.feature_bounds_ = np.delete(self.feature_bounds_, drop_features, axis=0)

        self.n_features_in_ = len(self.bins_)

        return self

    def sweep(self, terms=True, bins=True, features=False):
        """Purges unused elements from a fitted EBM.

        Args:
            terms: Boolean indicating if zeroed terms that do not affect the output
                should be purged from the model.
            bins: Boolean indicating if unused bin levels that do not affect the output
                should be purged from the model.
            features: Boolean indicating if features that are not used in any terms
                and therefore do not affect the output should be purged from the model.

        Returns:
            Itself.
        """

        check_is_fitted(self, "has_fitted_")

        if terms is True:
            terms = [
                i for i, v in enumerate(self.term_scores_) if np.count_nonzero(v) == 0
            ]
            self.remove_terms(terms)
        elif terms is not False:
            msg = "terms must be True or False"
            _log.error(msg)
            raise ValueError(msg)

        if bins is True:
            remove_unused_higher_bins(self.term_features_, self.bins_)
            deduplicate_bins(self.bins_)
        elif bins is not False:
            msg = "bins must be True or False"
            _log.error(msg)
            raise ValueError(msg)

        if features is True:
            features = np.ones(len(self.bins_), np.bool_)
            for term in self.term_features_:
                for feature_idx in term:
                    features.itemset(feature_idx, False)
            self.remove_features(features)
        elif features is not False:
            msg = "features must be True or False"
            _log.error(msg)
            raise ValueError(msg)

    def scale(self, term, factor):
        """Scale the individual term contribution by a constant factor. For
        example, you can nullify the contribution of specific terms by setting
        their corresponding weights to zero; this would cause the associated
        global explanations (e.g., variable importance) to also be zero. A
        couple of things are worth noting: 1) this method has no affect on the
        fitted intercept and users will have to change that attribute directly
        (if desired), and 2) reweighting specific term contributions will also
        reweight their related components in a similar manner (e.g., variable
        importance scores, standard deviations, etc.).

        Args:
            term: term index or name of the term to be scaled.
            factor: The amount to scale the term by.

        Returns:
            Itself.
        """

        check_is_fitted(self, "has_fitted_")

        term = clean_index(
            term,
            len(self.term_features_),
            getattr(self, "term_names_", None),
            "term",
            "self.term_names_",
        )

        self.term_scores_[term] *= factor
        self.bagged_scores_[term] *= factor
        self.standard_deviations_[term] *= factor

        return self

    def _multinomialize(self, passthrough=0.0):
        check_is_fitted(self, "has_fitted_")

        if passthrough < 0.0 or 1.0 < passthrough:
            msg = "passthrough must be between 0.0 and 1.0 inclusive"
            _log.error(msg)
            raise ValueError(msg)

        if self.link_ == "vlogit":
            multi_link = "mlogit"
            multi_param = np.nan
        else:
            msg = f"multinomialize can only be called on a OVR EBM classifier, but this classifier has link function {self.link_}."
            _log.error(msg)
            raise ValueError(msg)

        intercept_binary = self.intercept_.copy()

        # redo zero centering in-case the EBM has been unbalanced by editing
        terms = []
        for scores, w in zip(self.term_scores_, self.bin_weights_):
            mean = np.average(scores.reshape(-1, scores.shape[-1]), 0, w.flatten())
            intercept_binary += mean
            terms.append(scores - mean)

        prob = inv_link(intercept_binary, self.link_, self.link_param_)
        intercept_multi = link_func(prob, multi_link, multi_param)

        shift = np.zeros_like(intercept_multi)
        for i, w in enumerate(self.bin_weights_):
            prob = inv_link(terms[i] + intercept_binary, self.link_, self.link_param_)
            term = link_func(prob, multi_link, multi_param) - intercept_multi
            mean = np.average(term.reshape(-1, term.shape[-1]), 0, w.flatten())
            shift += mean
            terms[i] = term - mean

        intercept_multi += shift * passthrough

        self.term_scores_ = terms
        # TODO: do this per-bag in addition to the final scores:
        self.bagged_intercept_ = None
        self.bagged_scores_ = None
        self.standard_deviations_ = None

        self.link_ = multi_link
        self.link_param_ = multi_param
        self.intercept_ = intercept_multi

        return self

    def _ovrize(self, passthrough=0.0):
        check_is_fitted(self, "has_fitted_")

        if passthrough < 0.0 or 1.0 < passthrough:
            msg = "passthrough must be between 0.0 and 1.0 inclusive"
            _log.error(msg)
            raise ValueError(msg)

        if self.link_ == "mlogit":
            binary_link = "vlogit"
            binary_param = np.nan
        else:
            msg = f"ovrize can only be called on a multinomial EBM classifier, but this classifier has link function {self.link_}."
            _log.error(msg)
            raise ValueError(msg)

        intercept_multi = self.intercept_.copy()

        # redo zero centering in-case the EBM has been unbalanced by editing
        terms = []
        for scores, w in zip(self.term_scores_, self.bin_weights_):
            mean = np.average(scores.reshape(-1, scores.shape[-1]), 0, w.flatten())
            intercept_multi += mean
            terms.append(scores - mean)

        prob = inv_link(intercept_multi, self.link_, self.link_param_)
        intercept_binary = link_func(prob, binary_link, binary_param)

        shift = np.zeros_like(intercept_binary)
        for i, w in enumerate(self.bin_weights_):
            prob = inv_link(terms[i] + intercept_multi, self.link_, self.link_param_)
            term = link_func(prob, binary_link, binary_param) - intercept_binary
            mean = np.average(term.reshape(-1, term.shape[-1]), 0, w.flatten())
            shift += mean
            terms[i] = term - mean

        intercept_binary += shift * passthrough

        self.term_scores_ = terms
        # TODO: do this per-bag in addition to the final scores:
        self.bagged_intercept_ = None
        self.bagged_scores_ = None
        self.standard_deviations_ = None

        self.link_ = binary_link
        self.link_param_ = binary_param
        self.intercept_ = intercept_binary

        return self

    def _binarize(self, passthrough=0.0):
        check_is_fitted(self, "has_fitted_")

        if passthrough < 0.0 or 1.0 < passthrough:
            msg = "passthrough must be between 0.0 and 1.0 inclusive"
            _log.error(msg)
            raise ValueError(msg)

        original = self
        if original.link_ == "mlogit":
            original = self.copy()._ovrize(passthrough)

        if original.link_ == "vlogit":
            binary_link = "logit"
            binary_param = np.nan

            ebms = []
            for i in range(len(original.intercept_)):
                ebm = original.copy()
                ebm.classes_ = np.array([0, 1], np.int64)
                ebm.link_ = binary_link
                ebm.link_param_ = binary_param
                ebm.intercept_ = np.array([original.intercept_[i]], np.float64)

                # TODO: do this per-bag in addition to the final scores:
                ebm.bagged_intercept_ = None
                ebm.bagged_scores_ = None
                ebm.standard_deviations_ = None
                ebm.term_scores_ = [s[..., i] for s in original.term_scores_]

                ebms.append(ebm)

            return ebms
        else:
            msg = f"binarize can only be called on a multiclass EBM classifier, but this classifier has link function {self.link_}."
            _log.error(msg)
            raise ValueError(msg)


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
        "monoclassification", "custom_binary", "custom_ovr", "custom_multinomial",
        "mlogit", "vlogit", "logit", "probit", "cloglog", "loglog", "cauchit"
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
    bagged_intercept\\_ : array of float with shape ``(n_outer_bags, n_classes)`` or ``(n_outer_bags,)``
        Bagged intercept of the model. Binary classification is shape ``(n_outer_bags,)``, and multiclass is shape ``(n_outer_bags, n_classes)``.
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
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag], or 2D[bag, class]

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

        scores = self._predict_score(X, init_score)
        return inv_link(scores, self.link_, self.link_param_)

    def decision_function(self, X, init_score=None):
        """Predict scores from model before calling the link function.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.
        """

        return self._predict_score(X, init_score)

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """

        scores = self._predict_score(X, init_score)
        if scores.ndim == 1:
            # binary classification.  scikit-learn uses greater than semantics,
            # so score <= 0 means class_0, and 0 < score means class_1
            return self.classes_[(0 < scores).astype(np.int8)]
        elif scores.shape[1] == 0:
            # mono classification
            return np.full(len(scores), self.classes_[0], self.classes_.dtype)
        else:
            # multiclass
            return self.classes_[np.argmax(scores, axis=1)]


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
    bagged_intercept\\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
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
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]
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

        scores = self._predict_score(X, init_score)
        return inv_link(scores, self.link_, self.link_param_)


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
        "monoclassification", "custom_binary", "custom_ovr", "custom_multinomial",
        "mlogit", "vlogit", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    breakpoint_iteration\\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting rounds performed within each stage. Normally, the count of main effects
        boosting rounds will be in breakpoint_iteration_[0].
    intercept\\_ : array of float with shape ``(1,)``
        Intercept of the model.
    bagged_intercept\\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
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
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]

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

        scores = self._predict_score(X, init_score)
        return inv_link(scores, self.link_, self.link_param_)

    def decision_function(self, X, init_score=None):
        """Predict scores from model before calling the link function.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.
        """

        return self._predict_score(X, init_score)

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.
        """

        scores = self._predict_score(X, init_score)
        if scores.ndim == 1:
            # binary classification.  scikit-learn uses greater than semantics,
            # so score <= 0 means class_0, and 0 < score means class_1
            return self.classes_[(0 < scores).astype(np.int8)]
        elif scores.shape[1] == 0:
            # mono classification
            return np.full(len(scores), self.classes_[0], self.classes_.dtype)
        else:
            # multiclass
            return self.classes_[np.argmax(scores, axis=1)]


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
    bagged_intercept\\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
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
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]
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

        scores = self._predict_score(X, init_score)
        return inv_link(scores, self.link_, self.link_param_)
