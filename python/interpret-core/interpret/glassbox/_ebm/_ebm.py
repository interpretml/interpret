# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import heapq
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations, count, starmap
from math import ceil, isnan
from multiprocessing import shared_memory
from operator import itemgetter
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    is_classifier,
    is_regressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

from ... import develop
from ...api.base import ExplainerMixin
from ...api.templates import FeatureValueExplanation
from ...utils._clean_simple import (
    clean_dimensions,
    clean_X_and_init_score,
    typify_classification,
)
from ...utils._clean_x import preclean_X
from ...utils._compressed_dataset import bin_native_by_dimension
from ...utils._explanation import (
    gen_global_selector,
    gen_local_selector,
    gen_name_from_class,
    gen_perf_dicts,
)
from ...utils._histogram import make_all_histogram_edges
from ...utils._link import inv_link, link_func
from ...utils._measure_mem import total_bytes
from ...utils._misc import clean_index, clean_indexes
from ...utils._native import Native
from ...utils._preprocessor import construct_bins
from ...utils._privacy import (
    calc_classic_noise_multi,
    calc_gdp_noise_multi,
    validate_eps_delta,
)
from ...utils._rank_interactions import rank_interactions
from ...utils._seed import normalize_seed
from ...utils._shared_dataset import SharedDataset
from ...utils._unify_data import unify_data
from ._bin import (
    ebm_eval_terms,
    ebm_predict_scores,
    make_bin_weights,
)
from ._boost import boost
from ._json import UNTESTED_from_jsonable, to_jsonable
from ._tensor import remove_last, trim_tensor
from ._utils import (
    generate_term_names,
    generate_term_types,
    make_bag,
    order_terms,
    process_terms,
    remove_extra_bins,
)

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
        """Initialize class.

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
        """Provide interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.

        """
        from ...visual.plot import (
            is_multiclass_global_data_dict,
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
                xtitle="Mean Absolute Score (Weighted)",
            )

            figure._interpret_help_text = (
                "The term importances are the mean absolute "
                "contribution (score) each term (feature or interaction) makes to predictions "
                "averaged across the training dataset. Contributions are weighted by the number "
                "of samples in each bin, and by the sample weights (if any). The 15 most "
                "important terms are shown."
            )
            figure._interpret_help_link = "https://github.com/interpretml/interpret/blob/develop/docs/interpret/python/examples/group-importances.ipynb"

            return figure

        # Per term global explanation
        if self.explanation_type == "global":
            title = f"Term: {self.feature_names[key]} ({self.feature_types[key]})"

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
                    f"The contribution (score) of the term {self.feature_names[key]} to predictions "
                    "made by the model."
                )
            else:  # pragma: no cover
                msg = f"Not supported configuration: {self.explanation_type}, {self.feature_types[key]}"
                raise Exception(msg)

            figure._interpret_help_text = (
                "The contribution (score) of the term "
                f"{self.feature_names[key]} to predictions made by the model. For classification, "
                "scores are on a log scale (logits). For regression, scores are on the same "
                "scale as the outcome being predicted (e.g., dollars when predicting cost). "
                "Each graph is centered vertically such that average prediction on the train "
                "set is 0."
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
        msg = (
            f"`explainer_type has to be 'global' or 'local', got {self.explainer_type}."
        )
        raise NotImplementedError(msg)


def is_private(estimator):
    """Return True if the given estimator is a differentially private EBM estimator.

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
        if isinstance(term, (float, int, str)):
            term = (term,)  # noqa: PLW2901

        cleaned = []
        for feature in term:
            if isinstance(feature, float):
                if not feature.is_integer():
                    msg = "exclude must contain integers or feature names"
                    _log.error(msg)
                    raise ValueError(msg)
                feature = int(feature)  # noqa: PLW2901
            elif isinstance(feature, str):
                if feature not in feature_map:
                    msg = f"exclude item {feature} not in feature names"
                    _log.error(msg)
                    raise ValueError(msg)
                feature = feature_map[feature]  # noqa: PLW2901
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


@dataclass
class EbmInputTags:
    one_d_array: bool = False
    two_d_array: bool = True
    three_d_array: bool = False
    sparse: bool = True
    categorical: bool = True
    string: bool = True
    dict: bool = True
    positive_only: bool = False
    allow_nan: bool = True
    pairwise: bool = False


@dataclass
class EbmTargetTags:
    required: bool = True
    one_d_labels: bool = False
    two_d_labels: bool = False
    positive_only: bool = False
    multi_output: bool = False
    single_output: bool = True


@dataclass
class EbmClassifierTags:
    poor_score: bool = False
    multi_class: bool = True
    multi_label: bool = False


@dataclass
class EbmRegressorTags:
    poor_score: bool = False


@dataclass
class EbmTags:
    estimator_type: Optional[str] = None
    target_tags: EbmTargetTags = field(default_factory=EbmTargetTags)
    transformer_tags: None = None
    classifier_tags: Optional[EbmClassifierTags] = None
    regressor_tags: Optional[EbmRegressorTags] = None
    array_api_support: bool = True
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: EbmInputTags = field(default_factory=EbmInputTags)


def clean_interactions(interactions, n_features_in):
    if interactions is None:
        return 0

    if isinstance(interactions, (float, int)):
        if interactions <= 0:
            if interactions == 0:
                return 0
            msg = "interactions cannot be negative"
            _log.error(msg)
            raise ValueError(msg)

        if isinstance(interactions, float):
            if interactions < 1.0 or develop.get_option("allow_float_interactions"):
                interactions = int(ceil(n_features_in * interactions))
            else:
                msg = "interactions above 1 cannot be a float percentage and need to be an int instead"
                _log.error(msg)
                raise ValueError(msg)

        # at this point interactions will be a positive, nonzero integer
        return interactions
    elif isinstance(interactions, str):
        interactions = interactions.strip()

        if not interactions.lower().endswith("x"):
            raise ValueError(
                "If passing a string for interactions, it must end in an 'x' character."
            )

        interactions = interactions[:-1]
        try:
            interactions = float(interactions)
        except ValueError:
            raise ValueError(f"'{interactions}' is not a valid floating-point number.")

        if interactions <= 0.0:
            if interactions == 0.0:
                return 0
            msg = "interactions cannot be negative"
            _log.error(msg)
            raise ValueError(msg)

        interactions = int(ceil(n_features_in * interactions))

        # at this point interactions will be a positive, nonzero integer
        return interactions
    elif len(interactions) == 0:
        return 0
    else:
        # if it's a list then just return it
        return interactions


class EBMModel(ExplainerMixin, BaseEstimator):
    """Base class for all EBMs."""

    def __init__(
        self,
        # Explainer
        feature_names,
        feature_types,
        # Preprocessor
        max_bins,
        max_interaction_bins,
        # Stages
        interactions,
        exclude,
        # Ensemble
        validation_size,
        outer_bags,
        inner_bags,
        # Boosting
        learning_rate,
        greedy_ratio,
        cyclic_progress,
        smoothing_rounds,
        interaction_smoothing_rounds,
        max_rounds,
        early_stopping_rounds,
        early_stopping_tolerance,
        callback,
        # Trees
        min_samples_leaf,
        min_hessian,
        reg_alpha,
        reg_lambda,
        max_delta_step,
        gain_scale,
        min_cat_samples,
        cat_smooth,
        missing,
        max_leaves,
        monotone_constraints,
        objective,
        # Overall
        n_jobs,
        random_state,
        # Differential Privacy
        epsilon,
        delta,
        composition,
        bin_budget_frac,
        privacy_bounds,
        privacy_target_min,
        privacy_target_max,
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
            self.greedy_ratio = greedy_ratio
            self.cyclic_progress = cyclic_progress
            self.smoothing_rounds = smoothing_rounds
            self.interaction_smoothing_rounds = interaction_smoothing_rounds
        self.max_rounds = max_rounds
        if not is_private(self):
            self.early_stopping_rounds = early_stopping_rounds
            self.early_stopping_tolerance = early_stopping_tolerance
            self.callback = callback

            self.min_samples_leaf = min_samples_leaf
            self.min_hessian = min_hessian
            self.reg_alpha = reg_alpha
            self.reg_lambda = reg_lambda
            self.max_delta_step = max_delta_step
            self.gain_scale = gain_scale
            self.min_cat_samples = min_cat_samples
            self.cat_smooth = cat_smooth
            self.missing = missing

        self.max_leaves = max_leaves
        if not is_private(self):
            self.monotone_constraints = monotone_constraints
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

    def fit(self, X, y, sample_weight=None, bags=None, init_score=None):
        """Fit model to provided samples.

        Args:
            X: {array-like, sparse matrix} of shape (n_samples, n_features). Training data.
            y: array-like of shape (n_samples,). Target values.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.
            bags: Optional bag definitions. The first dimension should have length equal to the number of samples.
                The second dimension should have length equal to the number of outer_bags. The contents should be
                +1 for training, -1 for validation, and 0 if not included in the bag. Numbers other than 1 indicate
                how many times to include the sample in the training or validation sets.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Itself.

        """
        # with 64 bytes per tensor cell, a 2^20 tensor would be 1/16 gigabyte.
        max_cardinality = 1048576
        nominal_smoothing = True
        # In the future we might replace min_samples_leaf=4 with min_samples_bin=4 so
        # that we don't need to have the count when boosting or for interaction
        # detection. Benchmarking indicates switching these would decrease the accuracy
        # slightly, but it might be worth the speedup. Unfortunately, with outer bags
        # sometimes there will be 1 or 0 samples in some bags, so it isn't as good as
        # a boosting restriction. Another option would be to add tail restrictions
        # like min_samples_tail since when max_leaves is 2 the restriction only applies
        # to the tails anyways.
        min_samples_bin = 1

        if not isinstance(self.outer_bags, int) and not self.outer_bags.is_integer():
            msg = "outer_bags must be an integer"
            _log.error(msg)
            raise ValueError(msg)
        if self.outer_bags < 1:
            msg = "outer_bags must be 1 or greater. Did you mean to set: outer_bags=1, validation_size=0?"
            _log.error(msg)
            raise ValueError(msg)

        if (bags is not None) and bags.shape[1] != self.outer_bags:
            msg = f"bags has {bags.shape[1]} bags and self.outer_bags is {self.outer_bags} bags"
            _log.error(msg)
            raise ValueError(msg)

        if not isinstance(self.validation_size, int) and not isinstance(
            self.validation_size, float
        ):
            msg = "validation_size must be an integer or float"
            _log.error(msg)
            raise ValueError(msg)
        if self.validation_size <= 0:
            if self.validation_size < 0:
                msg = "validation_size cannot be negative"
                _log.error(msg)
                raise ValueError(msg)
            if self.outer_bags > 1:
                warn(
                    "If validation_size is 0, the outer_bags have no purpose. Set outer_bags=1 to remove this warning.",
                    stacklevel=1,
                )
        elif self.validation_size >= 1:
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
        if self.max_rounds < 0:
            # max_rounds == 0 means no boosting. This can be useful to just perform discretization
            msg = "max_rounds cannot be negative"
            _log.error(msg)
            raise ValueError(msg)

        if not is_private(self):
            if isnan(self.greedy_ratio):
                msg = "greedy_ratio cannot be NaN"
                _log.error(msg)
                raise ValueError(msg)
            if self.greedy_ratio < 0.0:
                msg = "greedy_ratio cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if self.cyclic_progress is True:
                cyclic_progress = 1.0
            elif self.cyclic_progress is False:
                cyclic_progress = 0.0
            elif isnan(self.cyclic_progress):
                msg = "cyclic_progress cannot be NaN"
                _log.error(msg)
                raise ValueError(msg)
            elif self.cyclic_progress < 0.0:
                msg = "cyclic_progress cannot be negative"
                _log.error(msg)
                raise ValueError(msg)
            elif self.cyclic_progress > 1.0:
                msg = "cyclic_progress cannot be above 1.0"
                _log.error(msg)
                raise ValueError(msg)
            else:
                cyclic_progress = float(self.cyclic_progress)

            if (
                not isinstance(self.smoothing_rounds, int)
                and not self.smoothing_rounds.is_integer()
            ):
                msg = "smoothing_rounds must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            if self.smoothing_rounds < 0:
                msg = "smoothing_rounds cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(self.interaction_smoothing_rounds, int)
                and not self.interaction_smoothing_rounds.is_integer()
            ):
                msg = "interaction_smoothing_rounds must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            if self.interaction_smoothing_rounds < 0:
                msg = "interaction_smoothing_rounds cannot be negative"
                _log.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(self.inner_bags, int)
                and not self.inner_bags.is_integer()
            ):
                msg = "inner_bags must be an integer"
                _log.error(msg)
                raise ValueError(msg)
            if self.inner_bags < 0:
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
            if self.early_stopping_rounds < 0:
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
        if self.learning_rate <= 0:
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
        n_classes = Native.Task_Unknown
        if objective is not None:
            if len(objective.strip()) == 0:
                objective = None
            else:
                n_classes = native.determine_task(objective)

        if is_classifier(self):
            if n_classes == Native.Task_Unknown:
                n_classes = Native.Task_GeneralClassification
            elif n_classes < Native.Task_GeneralClassification:
                msg = f"classifier cannot have objective {self.objective}"
                _log.error(msg)
                raise ValueError(msg)

        if is_regressor(self):
            if n_classes == Native.Task_Unknown:
                n_classes = Native.Task_Regression
            elif n_classes != Native.Task_Regression:
                msg = f"regressor cannot have objective {self.objective}"
                _log.error(msg)
                raise ValueError(msg)

        if Native.Task_GeneralClassification <= n_classes:
            y = typify_classification(y)
            # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
            # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
            # in two separate runs, which would flip the ordering of the classes within our score tensors.
            classes, y = np.unique(y, return_inverse=True)
            n_classes = len(classes)
            if objective is None:
                objective = "log_loss"
        elif n_classes == Native.Task_Regression:
            y = y.astype(np.float64, copy=False)
            min_target = y.min()
            max_target = y.max()
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
                msg = "sample_weight must be 1 dimensional"
                raise ValueError(msg)
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
        objective_code, _, link, link_param = native.determine_link(
            flags, objective, n_classes
        )

        X, n_samples, init_score = clean_X_and_init_score(
            X,
            init_score,
            self.feature_names,
            self.feature_types,
            link,
            link_param,
            len(y),
        )

        # Privacy calculations
        if is_differential_privacy:
            if self.random_state is not None:
                warn(
                    f"Privacy violation: using a fixed random_state of {self.random_state} "
                    "will cause deterministic noise additions. This capability is only "
                    "for debugging/testing. Set random_state to None to remove this warning.",
                    stacklevel=1,
                )

            validate_eps_delta(self.epsilon, self.delta)

            if n_classes < Native.Task_GeneralClassification:
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
                        "this warning.",
                        stacklevel=1,
                    )

                if is_clipping:
                    y = np.clip(y, min_target, max_target)
            elif n_classes >= Native.Task_MulticlassPlus:  # pragma: no cover
                msg = "Multiclass not supported for Differentially Private EBMs."
                raise ValueError(msg)

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
            min_samples_bin=min_samples_bin,
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
                "examine the missing value score in ebm.term_scores_[term_index][0]",
                stacklevel=1,
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
                    n_classes,
                    self.validation_size,
                    bagged_rng,
                    n_classes >= Native.Task_GeneralClassification
                    and not is_differential_privacy,
                )
            else:
                bag = bags[:, idx]
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
                if (bag > 127).any() or (bag < -128).any():
                    msg = "A value in bags is outside the valid range -128 to 127"
                    _log.error(msg)
                    raise ValueError(msg)
                bag = bag.astype(np.int8, "C", copy=False)

            rngs.append(bagged_rng)
            internal_bags.append(bag)

        bag_weights = []
        for bag in internal_bags:
            if bag is None:
                if sample_weight is None:
                    bag_weights.append(n_samples)
                else:
                    bag_weight = np.array(1, np.float64)
                    native.safe_sum(sample_weight, bag_weight, 0)
                    bag_weights.append(bag_weight.item())
            else:
                keep = bag > 0
                if sample_weight is None:
                    bag_weights.append(bag[keep].sum())
                else:
                    bag_weight = np.array(1, np.float64)
                    native.safe_sum(bag[keep] * sample_weight[keep], bag_weight, 0)
                    bag_weights.append(bag_weight.item())
                del keep
        bag_weights = np.array(bag_weights, np.float64)

        if is_differential_privacy:
            # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            domain_size = (
                1
                if n_classes >= Native.Task_GeneralClassification
                else max_target - min_target
            )
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
                msg = f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'."
                raise NotImplementedError(msg)

            bin_data_weights = main_bin_weights
            term_boost_flags = (
                Native.TermBoostFlags_GradientSums | Native.TermBoostFlags_RandomSplits
            )
            inner_bags = 0
            greedy_ratio = 0.0
            cyclic_progress = 1.0
            smoothing_rounds = 0
            interaction_smoothing_rounds = 0
            early_stopping_rounds = 0
            early_stopping_tolerance = 0.0
            callback = None
            min_samples_leaf = 0
            min_hessian = 0.0
            reg_alpha = 0.0
            reg_lambda = 0.0
            max_delta_step = 0.0
            gain_scale = 1.0
            min_cat_samples = 0
            cat_smooth = 0.0
            missing = "low"
            interactions = 0
            monotone_constraints = None
            n_intercept_rounds = 0
            interaction_flags = Native.CalcInteractionFlags_Default
        else:
            noise_scale_boosting = None
            bin_data_weights = None
            term_boost_flags = Native.TermBoostFlags_Default
            if develop.get_option("purify_boosting"):
                term_boost_flags |= Native.TermBoostFlags_PurifyUpdate
            if develop.get_option("boost_corners"):
                term_boost_flags |= Native.TermBoostFlags_Corners
            inner_bags = self.inner_bags
            greedy_ratio = self.greedy_ratio
            smoothing_rounds = self.smoothing_rounds
            interaction_smoothing_rounds = self.interaction_smoothing_rounds
            early_stopping_rounds = self.early_stopping_rounds
            early_stopping_tolerance = self.early_stopping_tolerance
            callback = self.callback
            min_samples_leaf = self.min_samples_leaf
            min_hessian = self.min_hessian
            reg_alpha = self.reg_alpha
            reg_lambda = self.reg_lambda
            max_delta_step = self.max_delta_step
            gain_scale = self.gain_scale
            min_cat_samples = self.min_cat_samples
            cat_smooth = self.cat_smooth
            missing = self.missing
            interactions = self.interactions
            monotone_constraints = self.monotone_constraints
            n_intercept_rounds = develop.get_option("n_intercept_rounds_initial")
            interaction_flags = Native.CalcInteractionFlags_Default
            if develop.get_option("full_interaction"):
                interaction_flags |= Native.CalcInteractionFlags_Full

        exclude_features = set()
        if monotone_constraints is not None:
            if len(monotone_constraints) != n_features_in:
                msg = f"There are {n_features_in} features, so monotone_constraints must have the same length, but it is instead {len(monotone_constraints)}."
                _log.error(msg)
                raise ValueError(msg)

            exclude_features = {i for i, v in enumerate(monotone_constraints) if v != 0}

        parallel = Parallel(n_jobs=self.n_jobs)

        bagged_intercept = np.zeros((self.outer_bags, n_scores), np.float64)
        if not is_differential_privacy:
            if objective_code == Native.Objective_MonoClassification:
                n_intercept_rounds = 0
            elif objective_code == Native.Objective_Rmse:
                # RMSE is very special and we can do closed form even with init_scores
                y_shifted = y if init_score is None else y - init_score
                n_intercept_rounds = 0

                for idx in range(self.outer_bags):
                    bag = internal_bags[idx]
                    sample_weight_local = sample_weight
                    y_local = y_shifted
                    if bag is not None:
                        include_samples = 0 < bag
                        y_local = y_local[include_samples]
                        if sample_weight_local is None:
                            sample_weight_local = bag[include_samples]
                        else:
                            sample_weight_local = (
                                sample_weight_local[include_samples]
                                * bag[include_samples]
                            )

                    bagged_intercept[idx, :] = native.flat_mean(
                        y_local,
                        None
                        if sample_weight_local is None
                        else np.asarray(sample_weight_local, np.float64),
                    )
            elif init_score is None:
                if (
                    objective_code == Native.Objective_LogLossBinary
                    or objective_code == Native.Objective_LogLossMulticlass
                ):
                    n_intercept_rounds = 0
                    for idx in range(self.outer_bags):
                        bag = internal_bags[idx]
                        sample_weight_local = sample_weight
                        y_local = y
                        if bag is not None:
                            include_samples = 0 < bag
                            y_local = y_local[include_samples]
                            if sample_weight_local is None:
                                sample_weight_local = bag[include_samples]
                            else:
                                sample_weight_local = (
                                    sample_weight_local[include_samples]
                                    * bag[include_samples]
                                )

                        probs = np.bincount(y_local, weights=sample_weight_local)
                        if sample_weight_local is None:
                            total = probs.sum()
                            probs = probs.astype(np.float64, copy=False)
                        else:
                            total = np.array(1, np.float64)
                            native.safe_sum(probs, total, 0)
                            total = total.item()

                        probs /= total
                        bagged_intercept[idx, :] = link_func(probs, link, link_param)

        with SharedDataset() as shared:
            bin_native_by_dimension(
                n_classes,
                1,
                bins,
                X,
                y,
                sample_weight,
                feature_names_in,
                feature_types_in,
                shared,
            )

            shm = None
            try:
                stop_flag = None
                shm_name = None
                if callback is not None:
                    shm = shared_memory.SharedMemory(create=True, size=1, name=None)
                    shm_name = shm.name
                    stop_flag = np.ndarray(1, dtype=np.bool_, buffer=shm.buf)
                    stop_flag[0] = False

                parallel_args = []
                for idx in range(self.outer_bags):
                    early_stopping_rounds_local = early_stopping_rounds
                    bag = internal_bags[idx]
                    if bag is None or (bag >= 0).all():
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
                            shm_name,
                            idx,
                            callback,
                            shared.name if shared.name is not None else shared.dataset,
                            n_intercept_rounds,
                            develop.get_option("intercept_learning_rate"),
                            bagged_intercept[idx],
                            bag,
                            init_score_local,
                            term_features,
                            inner_bags,
                            term_boost_flags,
                            self.learning_rate,
                            min_samples_leaf,
                            min_hessian,
                            reg_alpha,
                            reg_lambda,
                            max_delta_step,
                            gain_scale,
                            min_cat_samples,
                            cat_smooth,
                            missing,
                            self.max_leaves,
                            monotone_constraints,
                            greedy_ratio,
                            cyclic_progress,
                            smoothing_rounds,
                            nominal_smoothing,
                            self.max_rounds,
                            early_stopping_rounds_local,
                            early_stopping_tolerance,
                            noise_scale_boosting,
                            bin_data_weights,
                            rngs[idx],
                            (
                                Native.CreateBoosterFlags_DifferentialPrivacy
                                if is_differential_privacy
                                else Native.CreateBoosterFlags_Default
                            ),
                            objective,
                            develop.get_option("acceleration"),
                            None,
                            develop._develop_options,
                        )
                    )

                results = parallel(starmap(delayed(boost), parallel_args))

                # let python reclaim the dataset memory via reference counting
                # parallel_args holds references to dataset, so must be deleted
                del parallel_args

                best_iteration = [[]]
                models = []
                rngs = []
                for idx in range(self.outer_bags):
                    exception = results[idx][0]
                    intercept_update = results[idx][1]
                    model = results[idx][2]
                    bag_best_iteration = results[idx][3]
                    bagged_rng = results[idx][4]

                    if exception is not None:
                        raise exception
                    bagged_intercept[idx, :] = intercept_update
                    best_iteration[-1].append(bag_best_iteration)
                    models.append(model)
                    # retrieve our rng state since this was used outside of our process
                    rngs.append(bagged_rng)

                while True:  # this isn't for looping. Just for break statements to exit
                    if stop_flag is not None and stop_flag[0]:
                        break

                    interactions = clean_interactions(interactions, n_features_in)
                    if interactions == 0:  # works if interactions is a list
                        break

                    if n_classes >= Native.Task_MulticlassPlus:
                        warn(
                            "For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model."
                        )

                    # at this point we know we will be making a new one, so delete it now
                    shared.reset()

                    # we pass the bagged intercept into the rank_interactions and boost functions below, so do not include twice
                    initial_intercept = np.zeros(n_scores, np.float64)
                    scores_bags = []
                    for model, bag in zip(models, internal_bags):
                        # TODO: instead of going back to the original data in X, we
                        # could use the compressed and already binned data in dataset
                        scores = ebm_predict_scores(
                            X,
                            n_samples,
                            init_score,
                            feature_names_in,
                            feature_types_in,
                            bins,
                            initial_intercept,
                            model,
                            term_features,
                        )
                        if bag is not None and np.count_nonzero(bag) != len(bag):
                            scores = scores[bag != 0]
                        scores_bags.append(scores)

                    bin_native_by_dimension(
                        n_classes,
                        2,
                        bins,
                        X,
                        y,
                        sample_weight,
                        feature_names_in,
                        feature_types_in,
                        shared,
                    )

                    if isinstance(interactions, int):
                        _log.info("Estimating with FAST")

                        parallel_args = []
                        for idx in range(self.outer_bags):
                            # TODO: the combinations below should be selected from the non-excluded features
                            parallel_args.append(
                                (
                                    shm_name,
                                    idx,
                                    shared.name
                                    if shared.name is not None
                                    else shared.dataset,
                                    bagged_intercept[idx],
                                    internal_bags[idx],
                                    scores_bags[idx],
                                    combinations(range(n_features_in), 2),
                                    exclude,
                                    exclude_features,
                                    interaction_flags,
                                    max_cardinality,
                                    min_samples_leaf,
                                    min_hessian,
                                    reg_alpha,
                                    reg_lambda,
                                    max_delta_step,
                                    (
                                        Native.CreateInteractionFlags_DifferentialPrivacy
                                        if is_differential_privacy
                                        else Native.CreateInteractionFlags_Default
                                    ),
                                    objective,
                                    develop.get_option("acceleration"),
                                    None,
                                    0,
                                    develop._develop_options,
                                )
                            )

                        bagged_ranked_interaction = parallel(
                            starmap(delayed(rank_interactions), parallel_args)
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
                                    itemgetter(1),
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

                        for features in interactions:
                            feature_idxs = []
                            for feature in features:
                                if isinstance(feature, float):
                                    if not feature.is_integer():
                                        msg = f"interaction feature index {feature} is not an integer."
                                        _log.error(msg)
                                        raise ValueError(msg)
                                    feature = int(feature)  # noqa: PLW2901

                                if isinstance(feature, int):
                                    if feature < 0:
                                        feature_idx = len(feature_map) + feature
                                        if feature_idx < 0:
                                            msg = f"interaction feature index {feature} out of range of the features."
                                            _log.error(msg)
                                            raise ValueError(msg)
                                    elif feature < len(feature_map):
                                        feature_idx = feature
                                    else:
                                        msg = f"interaction feature index {feature} out of range of the features."
                                        _log.error(msg)
                                        raise ValueError(msg)
                                elif isinstance(feature, str):
                                    feature_idx = feature_map.get(feature)
                                    if feature_idx is None:
                                        msg = f'interaction feature "{feature}" not in the list of feature names.'
                                        _log.error(msg)
                                        raise ValueError(msg)
                                else:
                                    msg = f'interaction feature "{feature}" has unsupported type {type(feature)}.'
                                    _log.error(msg)
                                    raise ValueError(msg)
                                feature_idxs.append(feature_idx)
                            feature_idxs = tuple(feature_idxs)

                            max_dimensions = max(max_dimensions, len(feature_idxs))
                            sorted_tuple = tuple(sorted(feature_idxs))
                            if (
                                sorted_tuple not in uniquifier
                                and sorted_tuple not in exclude
                            ):
                                uniquifier.add(sorted_tuple)
                                boost_groups.append(feature_idxs)

                        if max_dimensions > 2:
                            warn(
                                "Interactions with 3 or more terms are not graphed in "
                                "global explanations. Local explanations are still "
                                "available and exact.",
                                stacklevel=1,
                            )

                    if stop_flag is not None and stop_flag[0]:
                        break

                    parallel_args = []
                    for idx in range(self.outer_bags):
                        early_stopping_rounds_local = early_stopping_rounds
                        if (
                            internal_bags[idx] is None
                            or (internal_bags[idx] >= 0).all()
                        ):
                            # if there are no validation samples, turn off early stopping
                            # because the validation metric cannot improve each round
                            early_stopping_rounds_local = 0

                        parallel_args.append(
                            (
                                shm_name,
                                idx,
                                callback,
                                shared.name
                                if shared.name is not None
                                else shared.dataset,
                                0,  # intercept should already be close for pairs
                                0.0,  # intercept should already be close for pairs
                                bagged_intercept[idx],
                                internal_bags[idx],
                                scores_bags[idx],
                                boost_groups,
                                inner_bags,
                                term_boost_flags,
                                self.learning_rate,
                                min_samples_leaf,
                                min_hessian,
                                reg_alpha,
                                reg_lambda,
                                max_delta_step,
                                gain_scale,
                                min_cat_samples,
                                cat_smooth,
                                missing,
                                self.max_leaves,
                                monotone_constraints,
                                greedy_ratio,
                                cyclic_progress,
                                interaction_smoothing_rounds,
                                nominal_smoothing,
                                self.max_rounds,
                                early_stopping_rounds_local,
                                early_stopping_tolerance,
                                noise_scale_boosting,
                                bin_data_weights,
                                rngs[idx],
                                (
                                    Native.CreateBoosterFlags_DifferentialPrivacy
                                    if is_differential_privacy
                                    else Native.CreateBoosterFlags_Default
                                ),
                                objective,
                                develop.get_option("acceleration"),
                                None,
                                develop._develop_options,
                            )
                        )

                    results = parallel(starmap(delayed(boost), parallel_args))

                    # allow python to reclaim these big memory items via reference counting
                    # this holds references to dataset, scores_bags, and bags
                    del parallel_args
                    del scores_bags

                    best_iteration.append([])

                    for idx in range(self.outer_bags):
                        exception = results[idx][0]
                        intercept_update = results[idx][1]
                        model = results[idx][2]
                        bag_best_iteration = results[idx][3]
                        bagged_rng = results[idx][4]

                        if exception is not None:
                            raise exception
                        bagged_intercept[idx, :] = intercept_update
                        best_iteration[-1].append(bag_best_iteration)
                        models[idx].extend(model)
                        rngs[idx] = bagged_rng

                    term_features.extend(boost_groups)

                    break  # do not loop!

            finally:
                if shm is not None:
                    shm.close()
                    shm.unlink()

            best_iteration = np.array(best_iteration, np.int64)

            remove_extra_bins(term_features, bins)

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

            if bagged_intercept.shape[1] == 1:
                bagged_intercept = bagged_intercept.ravel()

            intercept, term_scores, standard_deviations = process_terms(
                bagged_intercept, bagged_scores, bin_weights, bag_weights
            )

            if not is_differential_privacy:
                scores = ebm_predict_scores(
                    X,
                    n_samples,
                    init_score,
                    feature_names_in,
                    feature_types_in,
                    bins,
                    intercept,
                    term_scores,
                    term_features,
                )

                if objective_code == Native.Objective_MonoClassification:
                    pass
                elif objective_code == Native.Objective_Rmse:
                    correction = native.flat_mean(y - scores, sample_weight)
                    intercept += correction
                    bagged_intercept += correction
                else:
                    exception, intercept_change, _, _, rng = boost(
                        None,
                        0,
                        None,
                        shared.dataset,
                        develop.get_option("n_intercept_rounds_final"),
                        develop.get_option("intercept_learning_rate"),
                        np.zeros(n_scores, np.float64),
                        None,
                        scores,
                        [],
                        0,
                        term_boost_flags,
                        self.learning_rate,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        min_cat_samples,
                        cat_smooth,
                        missing,
                        1,
                        None,
                        greedy_ratio,
                        cyclic_progress,
                        0,
                        nominal_smoothing,
                        0,
                        0,
                        early_stopping_tolerance,
                        noise_scale_boosting,
                        bin_data_weights,
                        rng,
                        (
                            Native.CreateBoosterFlags_DifferentialPrivacy
                            if is_differential_privacy
                            else Native.CreateBoosterFlags_Default
                        ),
                        objective,
                        Native.AccelerationFlags_NONE,
                        None,
                        develop._develop_options,
                    )
                    if exception is not None:
                        raise exception

                    bagged_intercept += intercept_change
                    intercept += intercept_change

                    if bagged_intercept.ndim == 2:
                        # multiclass
                        # pick the class that we're going to zero
                        zero_index = np.argmax(intercept)
                        intercept -= intercept[zero_index]
                        bagged_intercept -= np.expand_dims(
                            bagged_intercept[..., zero_index], -1
                        )

        if n_classes < Native.Task_GeneralClassification:
            # scikit-learn requires intercept to be float for RegressorMixin, not numpy
            intercept = float(intercept[0])
        elif n_classes == Native.Task_MonoClassification:
            # for monoclassification, cells are either NaN or -inf
            intercept[~np.isnan(intercept)] = -np.inf
            bagged_intercept[~np.isnan(bagged_intercept)] = -np.inf
            bagged_scores = [
                np.where(np.isnan(s), np.nan, -np.inf) for s in bagged_scores
            ]
            term_scores = [np.where(np.isnan(s), np.nan, -np.inf) for s in term_scores]
            standard_deviations = [
                np.where(np.isnan(s), np.nan, np.inf) for s in standard_deviations
            ]

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

        if n_classes >= Native.Task_GeneralClassification:
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
        self.best_iteration_ = best_iteration
        self.has_fitted_ = True

        return self

    def estimate_mem(self, X, y=None, data_multiplier=0.0):
        """Estimate the amount of memory required during the call to the fit
            function. This estimate does not include the code or data memory used
            by the calling process, or the amount of code or non-EBM data memory
            allocated by any child processes started by joblib.  If these need to be
            included, it is recommended to fit a toy EBM model first with n_jobs
            and outer_bags set to the numbers that will be used in the final fit and
            add that value to the estimate provided by this function.

        Args:
            X (array-like or sparse matrix): Shape (n_samples, n_features). Training data.
            y (array-like, optional): Shape (n_samples,). Target values.
            data_multiplier (float): Controls inclusion of X's memory.
                If set to 0.0, this function only estimates the additional
                memory consumed by the fit function. If set to 1.0, it includes
                the memory allocated to X by the caller. Set above 1.0 to account
                for extra copies made before calling fit.

        Returns:
            int: Estimated memory usage in bytes.
                The estimate excludes the memory from the caller's copy of X,
                and the process's code or other unrelated data. The estimate
                is more accurate for larger datasets.
        """

        # TODO: include inner_bags
        # TODO: include sample weights

        n_bytes = total_bytes(X)
        if y is not None:
            n_bytes += total_bytes(y)

        n_bytes = int(n_bytes * data_multiplier)

        is_hessian = True
        if y is not None:
            y_id = id(y)
            n_classes = Native.Task_Unknown
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
            if objective is not None:
                if len(objective.strip()) == 0:
                    objective = None
                else:
                    n_classes = native.determine_task(objective)

            if is_classifier(self):
                if n_classes == Native.Task_Unknown:
                    n_classes = Native.Task_GeneralClassification
                elif n_classes < Native.Task_GeneralClassification:
                    msg = f"classifier cannot have objective {self.objective}"
                    _log.error(msg)
                    raise ValueError(msg)

            if is_regressor(self):
                if n_classes == Native.Task_Unknown:
                    n_classes = Native.Task_Regression
                elif n_classes != Native.Task_Regression:
                    msg = f"regressor cannot have objective {self.objective}"
                    _log.error(msg)
                    raise ValueError(msg)

            if Native.Task_GeneralClassification <= n_classes:
                y = typify_classification(y)
                # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
                # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
                # in two separate runs, which would flip the ordering of the classes within our score tensors.
                classes, y = np.unique(y, return_inverse=True)
                n_classes = len(classes)
            elif n_classes == Native.Task_Regression:
                y = y.astype(np.float64, copy=False)
            else:
                msg = f"Unrecognized objective {self.objective}"
                _log.error(msg)
                raise ValueError(msg)

            _, is_hessian, _, _ = native.determine_link(
                Native.LinkFlags_Default, objective, n_classes
            )

            if y_id != id(y):
                # in fit we'll also make a copy of y that cannot be deleted until the end
                n_bytes += total_bytes(y)

        n_samples = None if y is None else len(y)
        X_id = id(X)
        X, n_samples = preclean_X(
            X, self.feature_names, self.feature_types, n_samples, "y"
        )
        if X_id != id(X):
            # a copy was made, and we'll need to also do this on fit, so add the new memory too
            n_bytes += total_bytes(X)

        if y is None:
            n_classes = Native.Task_Regression
            # create a dummy y array (simulate regression)
            y = np.zeros(n_samples, dtype=np.float64)

        n_scores = Native.get_count_scores_c(n_classes)

        n_bytes_per_item = n_scores * n_samples * self.outer_bags * np.float32().nbytes

        # boosting requires 1 score, 1 gradient, and often 1 hessian per score, per sample, per outer_bag
        n_boost_sample_bytes = (3 if is_hessian else 2) * n_bytes_per_item

        # interaction detection requires 1 gradient, and often 1 hessian per score, per sample, per outer_bag
        n_interaction_detect_sample_bytes = (2 if is_hessian else 1) * n_bytes_per_item

        bin_levels = [self.max_bins, self.max_interaction_bins]

        binning_result = construct_bins(
            X=X,
            y=y,
            sample_weight=None,
            feature_names_given=self.feature_names,
            feature_types_given=self.feature_types,
            max_bins_leveled=bin_levels,
        )
        feature_names_in = binning_result[0]
        feature_types_in = binning_result[1]
        bins = binning_result[2]

        n_bytes_mains = bin_native_by_dimension(
            n_classes,
            1,
            bins,
            X,
            y,
            None,
            feature_names_in,
            feature_types_in,
            None,
        )

        # first calculate the number of cells in the mains for all features
        n_tensor_bytes = sum(
            (2 if len(x[0]) == 0 else max(x[0].values()) + 2)
            if isinstance(x[0], dict)
            else len(x[0]) + 3
            for x in bins
            if len(x) != 0
        )
        # We have 2 copies of the upate tensors in C++ (current and best) and we extract
        # one more in python for the update before tearning down the C++ data.
        n_tensor_bytes *= n_scores * self.outer_bags * 3 * np.float64().nbytes

        # One shared memory copy of the data mapped into all processes, plus a copy of
        # the test and train data for each outer bag. Assume all processes are started
        # at some point and are eating up memory.
        # When we cannot use shared memory the parent has a copy of the dataset and
        # all the children share one copy.
        max_bytes = (
            n_boost_sample_bytes
            + n_bytes_mains
            + n_bytes_mains
            + n_bytes_mains * self.outer_bags
            + n_tensor_bytes
        )

        n_features_in = len(bins)

        interactions = clean_interactions(self.interactions, n_features_in)
        if not isinstance(interactions, int):
            interactions = len(interactions)

        if interactions != 0:
            n_bytes_pairs = bin_native_by_dimension(
                n_classes,
                2,
                bins,
                X,
                y,
                None,
                feature_names_in,
                feature_types_in,
                None,
            )

            # INTERACTION DETECTION

            # Each outer bag makes a copy of the features. Only the training features
            # are kept for interaction detection, but don't estimate that for now.
            interaction_detection_bytes = (
                n_interaction_detect_sample_bytes
                + n_bytes_pairs
                + n_bytes_pairs
                + n_bytes_pairs * self.outer_bags
            )

            max_bytes = max(max_bytes, interaction_detection_bytes)

            # PAIR BOOSTING

            # find which bins will be selected for each feature when dealing with pairs
            bin_lengths = [x[0] if len(x) == 1 else x[1] for x in bins if len(x) != 0]
            bin_lengths = [
                (2 if len(x) == 0 else max(x.values()) + 2)
                if isinstance(x, dict)
                else len(x) + 3
                for x in bin_lengths
            ]
            bin_lengths.sort()
            # We don't know which features will be selected for interactions, so
            # use the 75th percentile bin length to estimate the number of bins
            n_bad_case_bins = bin_lengths[len(bin_lengths) // 4 * 3]

            # We have 2 copies of the update tensors in C++ (current and best) and we extract
            # one more in python for the update before tearning down the C++ data.
            interaction_boosting_bytes = (
                n_scores
                * n_bad_case_bins
                * n_bad_case_bins
                * interactions
                * 3
                * self.outer_bags
                * np.float64().nbytes
            )

            # We merge the interactions together to make a combined interaction
            # dataset, so if feature1 takes 4 bits and feature2 takes 10 bits
            # then the resulting data storage should take approx 14 bits in total,
            # so as a loose approximation we can add the bits in a pair, which means
            # roughtly multiply by 2.0 for pairs.  Multiply by another 2.0 just because
            # we might get unlucky and the pairs used are biased towards the ones
            # that have more bins.
            interaction_multiple = 4.0 * float(interactions) / float(n_features_in)
            interaction_boosting_bytes += (
                n_boost_sample_bytes
                + n_bytes_pairs
                + n_bytes_pairs
                + int(n_bytes_pairs * interaction_multiple * self.outer_bags)
            )

            max_bytes = max(max_bytes, interaction_boosting_bytes)

        return int(n_bytes + max_bytes)

    def to_jsonable(self, detail="all"):
        """Convert the model to a JSONable representation.

        Args:
            detail: 'minimal', 'interpretable', 'mergeable', 'all'

        Returns:
            JSONable object

        """
        check_is_fitted(self, "has_fitted_")

        return to_jsonable(self, detail)

    def to_json(self, file, detail="all", indent=2):
        r"""Export the model to a JSON text file.

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
        """Convert a JSONable EBM representation into an EBM.

        Args:
            jsonable: the JSONable object

        Returns:
            Itself after de-JSONifying.

        """
        UNTESTED_from_jsonable(self, jsonable)
        return self

    def _from_json(self, file):
        """Load from a JSON EBM file.

        Args:
            file: a path-like object (str or os.PathLike),
                or a file-like object implementing .read().

        Returns:
            Itself after loading.

        """
        if isinstance(file, (str, os.PathLike)):
            # file is a path-like object (str or os.PathLike)
            with open(file) as fp:
                jsonable = json.load(fp)
            UNTESTED_from_jsonable(self, jsonable)
        else:
            # file is a file-like object implementing .read()
            jsonable = json.load(fp)
            UNTESTED_from_jsonable(self, jsonable)
        return self

    def to_excel_exportable(self, file):
        """Converts the model to an Excel exportable representation.

        Args:

        Returns:
            An xlsxwriter.Workbook object with an Excel representation of the model.
            This Workbook can be modified and then exported as any xlsxwriter object
            for advanced usages when custom export is required.
        """

        check_is_fitted(self, "has_fitted_")

        from ._excel import UNTESTED_to_excel_exportable

        assert isinstance(self, ExplainableBoostingClassifier)

        workbook = UNTESTED_to_excel_exportable(self, file)

        return workbook

    def to_excel(self, file):
        """Exports the model to an Excel workbook.

        Args:
            file: a path-like object (str or os.PathLike),
                or a file-like object implementing .write().
        """

        check_is_fitted(self, "has_fitted_")

        workbook = self.to_excel_exportable(file)
        workbook.close()

    def _predict_score(self, X, init_score=None):
        """Predict scores on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.

        """
        check_is_fitted(self, "has_fitted_")

        return ebm_predict_scores(
            *clean_X_and_init_score(
                X,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.link_,
                self.link_param_,
            ),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
        )

    def eval_terms(self, X):
        r"""Term scores identical to the local explanation values obtained by calling ebm.explain_local(x).

        Calling interpret.utils.inv_link(ebm.eval_terms(X).sum(axis=1) + ebm.intercept\_, ebm.link\_)
        is equivalent to calling ebm.predict(X) for regression or ebm.predict_proba(X) for classification.

        Args:
            X: NumPy array for samples.

        Returns:
            local explanation scores for each term of each sample.

        """
        check_is_fitted(self, "has_fitted_")

        return ebm_eval_terms(
            *preclean_X(X, self.feature_names_in_, self.feature_types_in_),
            1 if isinstance(self.intercept_, float) else len(self.intercept_),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.term_scores_,
            self.term_features_,
        )

    def explain_global(self, name=None):
        """Provide global explanation for model.

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

                    # this will have no effect in normal models, but will handle inconsistent edited models
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
                    # Classes should be NumPy array, convert to list.
                    data_dict["meta"] = {"label_names": self.classes_.tolist()}

                data_dicts.append(data_dict)
            elif len(feature_idxs) == 2:
                if hasattr(self, "classes_") and len(self.classes_) > 2:
                    warn(
                        f"Dropping term {term_names[term_idx]} from explanation "
                        "since we can't graph multinomial interactions.",
                        stacklevel=1,
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

                        # this will have no effect in normal models, but will handle inconsistent edited models
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

                        # this will have no effect in normal models, but will handle inconsistent edited models
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
                    "since we can't graph more than 2 dimensions.",
                    stacklevel=1,
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
        """Provide local explanations for provided samples.

        Args:
            X: NumPy array for X to explain.
            y: NumPy vector for y to explain.
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
                msg = "y must be 1 dimensional"
                raise ValueError(msg)
            n_samples = len(y)

            if classes is not None:
                y = typify_classification(y)
            else:
                y = y.astype(np.float64, copy=False)

        X, n_samples, init_score = clean_X_and_init_score(
            X,
            init_score,
            self.feature_names_in_,
            self.feature_types_in_,
            self.link_,
            self.link_param_,
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
            if (classes is None or len(classes) <= 2) and isinstance(
                intercept, (list, np.ndarray)
            ):
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
                    # Classes should be NumPy array, convert to list.
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
        """Provide the term importances.

        Args:
            importance_type: the type of term importance requested ('avg_weight', 'min_max')

        Returns:
            An array term importances with one importance per additive term

        """
        check_is_fitted(self, "has_fitted_")

        n_terms = len(self.term_features_)
        if hasattr(self, "classes_") and len(self.classes_) <= 1:
            # everything is useless if we're predicting 1 class
            return np.zeros(n_terms, np.float64)

        if importance_type == "avg_weight":
            importances = np.empty(n_terms, np.float64)
            for i in range(n_terms):
                scores = np.abs(self.term_scores_[i])
                if hasattr(self, "classes_") and len(self.classes_) >= 3:
                    # for multiclass, average the abs class scores
                    scores = np.average(scores, axis=-1)
                importances[i] = np.average(scores, weights=self.bin_weights_[i])
            return importances
        if importance_type == "min_max":
            return np.array(
                [np.max(tensor) - np.min(tensor) for tensor in self.term_scores_],
                np.float64,
            )
        msg = f"Unrecognized importance_type: {importance_type}"
        raise ValueError(msg)

    def copy(self):
        """Make a deepcopy of the EBM.

        Returns:
            The new copy.

        """
        return deepcopy(self)

    def monotonize(self, term, increasing="auto", passthrough=0.0):
        r"""Adjust a term to be monotone using isotonic regression.

        An important consideration is that this function only adjusts a single term and will not modify pairwise terms.
        When a feature needs to be globally monotonic, any pairwise terms that include the feature
        should be excluded from the model.

        Args:
            term: Index or name of the term to monotonize
            increasing: 'auto' or bool. 'auto' decides direction based on Spearman correlation estimate.
            passthrough: the process of monotonization can result in a change to the mean response
                of the model. If passthrough is set to 0.0 then the model's mean response to the
                training set will not change. If passthrough is set to 1.0 then any change to the
                mean response made by monotonization will be passed through to self.intercept\_.
                Values between 0 and 1 will result in that percentage being passed through.

        Returns:
            Itself.

        """
        check_is_fitted(self, "has_fitted_")

        if hasattr(self, "classes_"):
            if len(self.classes_) == 1:
                # monoclassification is always monotonized
                return None
            if len(self.classes_) > 2:
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
        if len(features) >= 2:
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

        if passthrough < 0.0 or passthrough > 1.0:
            msg = "passthrough must be between 0.0 and 1.0 inclusive"
            _log.error(msg)
            raise ValueError(msg)

        # the missing and unseen bins are not part of the continuous range
        y = self.term_scores_[term][1:-1]
        x = np.arange(len(y), dtype=np.int64)

        all_weights = self.bin_weights_[term]
        weights = all_weights[1:-1]

        # this should normally be zero, except if there are missing or unseen values
        original_mean = np.average(y, weights=weights)

        # Fit isotonic regression weighted by training data bin counts
        ir = IsotonicRegression(increasing=increasing)
        y = ir.fit_transform(x, y, sample_weight=weights)

        result_mean = np.average(y, weights=weights)
        change = (original_mean - result_mean) * (1.0 - passthrough)
        y += change

        self.term_scores_[term][1:-1] = y

        if passthrough > 0.0:
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
        """Remove terms (and their associated components) from a fitted EBM.

        Note that this will change the structure (i.e., by removing the specified
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
        r"""Remove features (and their associated components) from a fitted EBM.

        Note that this will change the structure (i.e., by removing the specified
        indices) of the following components of ``self``: ``histogram_edges_``,
        ``histogram_weights_``, ``unique_val_counts_``, ``bins_``,
        ``feature_names_in_``, ``feature_types_in_``, and ``feature_bounds_``.
        Also, any terms that use the features being deleted will be deleted.
        The following attributes that the caller passed to the \_\_init\_\_ function are
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
        """Purge unused elements from a fitted EBM.

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
            remove_extra_bins(self.term_features_, self.bins_)
        elif bins is not False:
            msg = "bins must be True or False"
            _log.error(msg)
            raise ValueError(msg)

        if features is True:
            features = np.ones(len(self.bins_), np.bool_)
            for term in self.term_features_:
                for feature_idx in term:
                    features[feature_idx] = False
            self.remove_features(features)
        elif features is not False:
            msg = "features must be True or False"
            _log.error(msg)
            raise ValueError(msg)

        return self

    def scale(self, term, factor):
        """Scale the individual term contribution by a constant factor.

        For example, you can nullify the contribution of specific terms by setting
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

    def predict_with_uncertainty(self, X, init_score=None):
        """Gets raw scores and uncertainties from the bagged base models.
        Generates predictions by averaging outputs across all bagged models, and estimates
        uncertainty using the standard deviation of predictions across bags.

        Args:
            X: ndarray of shape (n_samples, n_features)
                The input samples to predict on.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            ndarray of shape (n_samples, 2)
                First column contains mean predictions
                Second column contains uncertainties
        """

        check_is_fitted(self, "has_fitted_")

        X, n_samples, init_score = clean_X_and_init_score(
            X,
            init_score,
            self.feature_names_in_,
            self.feature_types_in_,
            self.link_,
            self.link_param_,
        )

        preds_per_bag = np.empty((len(self.bagged_intercept_), n_samples), np.float64)

        for bag_index in range(len(self.bagged_intercept_)):
            preds_per_bag[bag_index, :] = ebm_predict_scores(
                X,
                n_samples,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.bagged_intercept_[bag_index],
                list(map(itemgetter(bag_index), self.bagged_scores_)),
                self.term_features_,
            )

        # TODO: would transposing preds_per_bag here improve speed?

        # Calculate mean predictions and uncertainties
        return np.c_[np.mean(preds_per_bag, axis=0), np.std(preds_per_bag, axis=0)]

    def _multinomialize(self, passthrough=0.0):
        check_is_fitted(self, "has_fitted_")

        if passthrough < 0.0 or passthrough > 1.0:
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
            mean = np.average(scores.reshape(-1, scores.shape[-1]), 0, w.ravel())
            intercept_binary += mean
            terms.append(scores - mean)

        prob = inv_link(intercept_binary, self.link_, self.link_param_)
        intercept_multi = link_func(prob, multi_link, multi_param)

        shift = np.zeros_like(intercept_multi)
        for i, w in enumerate(self.bin_weights_):
            prob = inv_link(terms[i] + intercept_binary, self.link_, self.link_param_)
            term = link_func(prob, multi_link, multi_param) - intercept_multi
            mean = np.average(term.reshape(-1, term.shape[-1]), 0, w.ravel())
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

        if passthrough < 0.0 or passthrough > 1.0:
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
            mean = np.average(scores.reshape(-1, scores.shape[-1]), 0, w.ravel())
            intercept_multi += mean
            terms.append(scores - mean)

        prob = inv_link(intercept_multi, self.link_, self.link_param_)
        intercept_binary = link_func(prob, binary_link, binary_param)

        shift = np.zeros_like(intercept_binary)
        for i, w in enumerate(self.bin_weights_):
            prob = inv_link(terms[i] + intercept_multi, self.link_, self.link_param_)
            term = link_func(prob, binary_link, binary_param) - intercept_binary
            mean = np.average(term.reshape(-1, term.shape[-1]), 0, w.ravel())
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

        if passthrough < 0.0 or passthrough > 1.0:
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
        msg = f"binarize can only be called on a multiclass EBM classifier, but this classifier has link function {self.link_}."
        _log.error(msg)
        raise ValueError(msg)

    def _more_tags(self):
        return {
            "allow_nan": True,
            "requires_y": True,
            "array_api_support": True,
            "X_types": [
                "2darray",
                "string",
                "sparse",
                "categorical",
                "dict",
                "1dlabels",
            ],
        }

    def __sklearn_tags__(self):
        return EbmTags()


class ExplainableBoostingClassifier(ClassifierMixin, EBMModel):
    r"""An Explainable Boosting Classifier.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. FeatureType can be:

            - `'auto'`: Auto-detect
            - `'quantile'`: Continuous with equal density bins
            - `'rounded_quantile'`: Continuous with quantile bins, but the cut values are rounded when possible
            - `'uniform'`: Continuous with equal width bins
            - `'winsorized'`: Continuous with equal width bins, but the leftmost and rightmost cut are chosen by quantiles
            - `'continuous'`: Use the default binning for continuous features, which is 'quantile' currently
            - `[List of float]`: Continuous with specified cut values. Eg: [5.5, 8.75]
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names
    max_bins : int, default=1024
        Max number of bins per feature for the main effects stage.
    max_interaction_bins : int, default=64
        Max number of bins per feature for interaction terms.
    interactions : int, float, str, or list of tuples of feature indices, default="3x"

        Interaction terms to be included in the model. Options are:

            - Integer (1 <= interactions): Count of interactions to be automatically selected
            - Percentage (interactions < 1.0): Determine the integer count of interactions by multiplying the number of features by this percentage
            - String with format (float + "x"): Determine the integer count of interactions by multiplying the number of features by the float value.
            - List of tuples: The tuples contain the indices of the features within each additive term. In addition to pairs,
              the interactions parameter accepts higher order interactions. It also accepts univariate terms which will cause
              the algorithm to boost the main terms at the same time as the interactions. When boosting mains at the same time
              as interactions, the exclude parameter should be set to 'mains' and currently max_bins needs to be equal to
              max_interaction_bins.
    exclude : 'mains' or list of tuples of feature indices|names, default=None
        Features or terms to be excluded.
    validation_size : int or float, default=0.15

        Validation set size. Used for early stopping during boosting, and is needed to create outer bags.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Turns off early stopping. Outer bags have no utility. Error bounds will be eliminated
    outer_bags : int, default=14
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    inner_bags : int, default=0
        Number of inner bags. 0 turns off inner bagging.
    learning_rate : float, default=0.015
        Learning rate for boosting.
    greedy_ratio : float, default=10.0
        The proportion of greedy boosting steps relative to cyclic boosting steps.
        A value of 0 disables greedy boosting, effectively turning it off.
    cyclic_progress : bool or float, default=False
        This parameter specifies the proportion of the boosting cycles that will
        actively contribute to improving the model's performance. It is expressed
        as a bool or float between 0 and 1, with the default set to True(1.0), meaning 100% of
        the cycles are expected to make forward progress. If forward progress is
        not achieved during a cycle, that cycle will not be wasted; instead,
        it will be used to update internal gain calculations related to how effective
        each feature is in predicting the target variable. Setting this parameter
        to a value less than 1.0 can be useful for preventing overfitting.
    smoothing_rounds : int, default=75
        Number of initial highly regularized rounds to set the basic shape of the main effect feature graphs.
    interaction_smoothing_rounds : int, default=75
        Number of initial highly regularized rounds to set the basic shape of the interaction effect feature graphs during fitting.
    max_rounds : int, default=50000
        Total number of boosting rounds with n_terms boosting steps per round.
    early_stopping_rounds : int, default=100
        Number of rounds with no improvement to trigger early stopping. 0 turns off
        early stopping and boosting will occur for exactly max_rounds.
    early_stopping_tolerance : float, default=1e-5
        Tolerance that dictates the smallest delta required to be considered an
        improvement which prevents the algorithm from early stopping.
        early_stopping_tolerance is expressed as a percentage of the early
        stopping metric. Negative values indicate that the individual
        models should be overfit before stopping.
        EBMs are a bagged ensemble of models. Setting the early_stopping_tolerance
        to zero (or even negative), allows learning to overfit each of the individual
        models a little, which can improve the accuracy of the ensemble as a whole.
        Overfitting each of the individual models reduces the bias of each model at
        the expense of increasing the variance (due to overfitting) of the individual
        models.  But averaging the models in the ensemble reduces variance without
        much change in bias.  Since the goal is to find the optimum bias-variance
        tradeoff for the ensemble of models --- not the individual models --- a small
        amount of overfitting of the individual models can improve the accuracy of
        the ensemble as a whole.
    callback : Optional[Callable[[int, int, bool, float], bool]], default=None
        A user-defined function that is invoked at the end of each boosting step to determine
        whether to terminate boosting or continue. If it returns True, the boosting loop is
        stopped immediately. By default, no callback is used and training proceeds according
        to the early stopping settings.  The callback function receives:
        (1) the bag index, (2) the number of boosting steps completed,
        (3) a boolean indicating whether progress was made in the current step, and (4) the current best score.
    min_samples_leaf : int, default=4
        Minimum number of samples allowed in the leaves.
    min_hessian : float, default=1e-4
        Minimum hessian required to consider a potential split valid.
    reg_alpha : float, default=0.0
        L1 regularization.
    reg_lambda : float, default=0.0
        L2 regularization.
    max_delta_step : float, default=0.0
        Used to limit the max output of tree leaves. <=0.0 means no constraint.
    gain_scale : float, default=5.0
        Scale factor to apply to nominal categoricals. A scale factor above 1.0 will cause the
        algorithm focus more on the nominal categoricals.
    min_cat_samples : int, default=10
        Minimum number of samples in order to treat a category separately. If lower than this threshold
        the category is combined with other categories that have low numbers of samples.
    cat_smooth : float, default=10.0
        Used for the categorical features. This can reduce the effect of noises in categorical features,
        especially for categories with limited data.
    missing: str, default="separate"

        Method for handling missing values during boosting. The placement of the missing value bin can influence
        the resulting model graphs. For example, placing the bin on the "low" side may cause missing values to
        affect lower bins, and vice versa. This parameter does not affect the final placement
        of the missing bin in the model (the missing bin will remain at index 0 in the term_scores\_ attribute).
        Possible values for missing are:

            - `'low'`: Place the missing bin on the left side of the graphs.
            - `'high'`: Place the missing bin on the right side of the graphs.
            - `'separate'`: Place the missing bin in its own leaf during each boosting step,
              effectively making it location-agnostic. This can lead to overfitting, especially
              when the proportion of missing values is small.
            - `'gain'`: Choose the best leaf for the missing value contribution at each boosting step, based on gain.
    max_leaves : int, default=2
        Maximum number of leaves allowed in each tree.
    monotone_constraints: list of int, default=None

        This parameter allows you to specify monotonic constraints for each feature's
        relationship with the target variable during model fitting. However, it is
        generally recommended to apply monotonic constraints post-fit using the
        `monotonize` function rather than setting them during the fitting process.
        This recommendation is based on the observation that, during fitting,
        the boosting algorithm may compensate for a monotone constraint on one
        feature by utilizing another correlated feature, potentially obscuring
        any monotonic violations.

        If you choose to define monotone constraints, `monotone_constraints`
        should be a list with a length equal to the number of features.
        Each element in the list corresponds to a feature and should take
        one of the following values:

            - 0: No monotonic constraint is imposed on the corresponding feature's partial response.
            - +1: The partial response of the corresponding feature should be monotonically increasing with respect to the target.
            - -1: The partial response of the corresponding feature should be monotonically decreasing with respect to the target.
    objective : str, default="log_loss"
        The objective to optimize.
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=42
        Random state. None uses device_random and generates non-repeatable sequences.

    Attributes
    ----------
    classes\_ : array of bool, int, or unicode with shape ``(n_classes,)``
        The class labels.
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    histogram_edges\_ : List of None or array of float with shape ``(n_hist_edges,)``
        Per-feature list of the histogram edges. Categorical features contain None within the List
        at their feature index.
    histogram_weights\_ : List of array of float with shape ``(n_hist_bins,)``
        Per-feature list of the total sample weights within each feature's histogram bins.
    unique_val_counts\_ : array of int with shape ``(n_features,)``
        Per-feature count of the number of unique feature values.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the total sample weights in each term's tensor bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the bagged model scores.
        The last dimension of length n_classes is dropped for binary classification.
    term_scores\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the model scores.
        The last dimension of length n_classes is dropped for binary classification.
    standard_deviations\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins, n_classes)`` or ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the standard deviations of the bagged model scores.
        The last dimension of length n_classes is dropped for binary classification.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "monoclassification", "custom_binary", "custom_ovr", "custom_multinomial",
        "mlogit", "vlogit", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0],
        and the count of interaction boosting iterations will be in best_iteration_[1].
    intercept\_ : array of float with shape ``(n_classes,)`` or ``(1,)``
        Intercept of the model. Binary classification is shape ``(1,)``, and multiclass is shape ``(n_classes,)``.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags, n_classes)`` or ``(n_outer_bags,)``
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
    best_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    histogram_edges_: List[Union[None, np.ndarray]]  # np.float64, 1D[hist_edge]
    histogram_weights_: List[np.ndarray]  # np.float64, 1D[hist_bin]
    unique_val_counts_: np.ndarray  # np.int64, 1D[feature]

    classes_: np.ndarray  # np.int64, np.bool_, or np.unicode_, 1D[class]
    intercept_: np.ndarray  # np.float64, 1D[class]
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag], or 2D[bag, class]

    # TODO PK v.3 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ("global", "local")
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
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        # Stages
        interactions: Optional[
            Union[int, float, str, Sequence[Union[int, str, Sequence[Union[int, str]]]]]
        ] = "3x",
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = None,
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0.15,
        outer_bags: int = 14,
        inner_bags: Optional[int] = 0,
        # Boosting
        learning_rate: float = 0.015,
        greedy_ratio: Optional[float] = 10.0,
        cyclic_progress: Union[bool, float, int] = False,  # noqa: PYI041
        smoothing_rounds: Optional[int] = 75,
        interaction_smoothing_rounds: Optional[int] = 75,
        max_rounds: Optional[int] = 50000,
        early_stopping_rounds: Optional[int] = 100,
        early_stopping_tolerance: Optional[float] = 1e-5,
        callback: Optional[Callable[[int, int, bool, float], bool]] = None,
        # Trees
        min_samples_leaf: Optional[int] = 4,
        min_hessian: Optional[float] = 1e-4,
        reg_alpha: Optional[float] = 0.0,
        reg_lambda: Optional[float] = 0.0,
        max_delta_step: Optional[float] = 0.0,
        gain_scale: Optional[float] = 5.0,
        min_cat_samples: Optional[int] = 10,
        cat_smooth: Optional[float] = 10.0,
        missing: str = "separate",
        max_leaves: int = 2,
        monotone_constraints: Optional[Sequence[int]] = None,
        objective: str = "log_loss",
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
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
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            callback=callback,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=max_delta_step,
            gain_scale=gain_scale,
            min_cat_samples=min_cat_samples,
            cat_smooth=cat_smooth,
            missing=missing,
            max_leaves=max_leaves,
            monotone_constraints=monotone_constraints,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=None,
            delta=None,
            composition=None,
            bin_budget_frac=None,
            privacy_bounds=None,
            privacy_target_min=None,
            privacy_target_max=None,
        )

    def predict_proba(self, X, init_score=None):
        """Probability estimates on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Probability estimate of sample for each class.

        """

        check_is_fitted(self, "has_fitted_")

        return inv_link(
            ebm_predict_scores(
                *clean_X_and_init_score(
                    X,
                    init_score,
                    self.feature_names_in_,
                    self.feature_types_in_,
                    self.link_,
                    self.link_param_,
                ),
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.intercept_,
                self.term_scores_,
                self.term_features_,
            ),
            self.link_,
            self.link_param_,
        )

    def decision_function(self, X, init_score=None):
        """Predict scores from model before calling the link function.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.

        """
        check_is_fitted(self, "has_fitted_")

        return ebm_predict_scores(
            *clean_X_and_init_score(
                X,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.link_,
                self.link_param_,
            ),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
        )

    def predict(self, X, init_score=None):
        """Predict on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.

        """
        check_is_fitted(self, "has_fitted_")

        scores = ebm_predict_scores(
            *clean_X_and_init_score(
                X,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.link_,
                self.link_param_,
            ),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
        )

        if scores.ndim == 1:
            # binary classification.  scikit-learn uses greater than semantics,
            # so score <= 0 means class_0, and 0 < score means class_1
            return self.classes_[(scores > 0).astype(np.int8)]
        # multiclass
        return self.classes_[np.argmax(scores, axis=1)]

    def reorder_classes(self, classes):
        """Re-order the class positions in a classification EBM.

        Args:
            classes: The new class order

        Returns:
            Itself.

        """
        check_is_fitted(self, "has_fitted_")

        classes = np.asarray(classes, dtype=self.classes_.dtype)

        if len(classes) != len(self.classes_):
            raise ValueError(
                "The EBM contains {len(self.classes_)} classes, but the 'classes' parameter contains {len(classes)} items."
            )

        mapping = dict(zip(self.classes_, count()))
        try:
            mapping = np.fromiter(
                map(mapping.__getitem__, classes),
                np.uint64,
                count=len(mapping),
            )
        except KeyError as e:
            raise ValueError(
                f"The 'classes' parameter contains a class '{e.args[0]}' not present in the EBM."
            ) from e

        if len(mapping) != len(set(mapping)):
            raise ValueError("The 'classes' parameter contains duplicates.")

        self.classes_ = self.classes_[mapping]

        if len(mapping) == 2:
            if mapping[0] == 1:
                np.negative(self.intercept_, out=self.intercept_)
                np.negative(self.bagged_intercept_, out=self.bagged_intercept_)
                for scores in self.bagged_scores_:
                    np.negative(scores, out=scores)
                for scores in self.term_scores_:
                    np.negative(scores, out=scores)
        elif 3 <= len(mapping):
            self.intercept_ = self.intercept_[mapping]
            self.bagged_intercept_ = self.bagged_intercept_[:, mapping]
            self.bagged_scores_ = [
                scores[..., mapping] for scores in self.bagged_scores_
            ]
            self.term_scores_ = [scores[..., mapping] for scores in self.term_scores_]
            self.standard_deviations_ = [
                scores[..., mapping] for scores in self.standard_deviations_
            ]

        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = EbmClassifierTags()
        return tags


class ExplainableBoostingRegressor(RegressorMixin, EBMModel):
    r"""An Explainable Boosting Regressor.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. FeatureType can be:

            - `'auto'`: Auto-detect
            - `'quantile'`: Continuous with equal density bins
            - `'rounded_quantile'`: Continuous with quantile bins, but the cut values are rounded when possible
            - `'uniform'`: Continuous with equal width bins
            - `'winsorized'`: Continuous with equal width bins, but the leftmost and rightmost cut are chosen by quantiles
            - `'continuous'`: Use the default binning for continuous features, which is 'quantile' currently
            - `[List of float]`: Continuous with specified cut values. Eg: [5.5, 8.75]
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names
    max_bins : int, default=1024
        Max number of bins per feature for the main effects stage.
    max_interaction_bins : int, default=64
        Max number of bins per feature for interaction terms.
    interactions : int, float, str, or list of tuples of feature indices, default="5x"

        Interaction terms to be included in the model. Options are:

            - Integer (1 <= interactions): Count of interactions to be automatically selected
            - Percentage (interactions < 1.0): Determine the integer count of interactions by multiplying the number of features by this percentage
            - String with format (float + "x"): Determine the integer count of interactions by multiplying the number of features by the float value.
            - List of tuples: The tuples contain the indices of the features within each additive term. In addition to pairs,
              the interactions parameter accepts higher order interactions. It also accepts univariate terms which will cause
              the algorithm to boost the main terms at the same time as the interactions. When boosting mains at the same time
              as interactions, the exclude parameter should be set to 'mains' and currently max_bins needs to be equal to
              max_interaction_bins.
    exclude : 'mains' or list of tuples of feature indices|names, default=None
        Features or terms to be excluded.
    validation_size : int or float, default=0.15

        Validation set size. Used for early stopping during boosting, and is needed to create outer bags.

            - Integer (1 <= validation_size): Count of samples to put in the validation sets
            - Percentage (validation_size < 1.0): Percentage of the data to put in the validation sets
            - 0: Turns off early stopping. Outer bags have no utility. Error bounds will be eliminated
    outer_bags : int, default=14
        Number of outer bags. Outer bags are used to generate error bounds and help with smoothing the graphs.
    inner_bags : int, default=0
        Number of inner bags. 0 turns off inner bagging.
    learning_rate : float, default=0.04
        Learning rate for boosting.
    greedy_ratio : float, default=10.0
        The proportion of greedy boosting steps relative to cyclic boosting steps.
        A value of 0 disables greedy boosting, effectively turning it off.
    cyclic_progress : bool or float, default=False
        This parameter specifies the proportion of the boosting cycles that will
        actively contribute to improving the model's performance. It is expressed
        as a bool or float between 0 and 1, with the default set to True(1.0), meaning 100% of
        the cycles are expected to make forward progress. If forward progress is
        not achieved during a cycle, that cycle will not be wasted; instead,
        it will be used to update internal gain calculations related to how effective
        each feature is in predicting the target variable. Setting this parameter
        to a value less than 1.0 can be useful for preventing overfitting.
    smoothing_rounds : int, default=500
        Number of initial highly regularized rounds to set the basic shape of the main effect feature graphs.
    interaction_smoothing_rounds : int, default=100
        Number of initial highly regularized rounds to set the basic shape of the interaction effect feature graphs during fitting.
    max_rounds : int, default=50000
        Total number of boosting rounds with n_terms boosting steps per round.
    early_stopping_rounds : int, default=100
        Number of rounds with no improvement to trigger early stopping. 0 turns off
        early stopping and boosting will occur for exactly max_rounds.
    early_stopping_tolerance : float, default=1e-5
        Tolerance that dictates the smallest delta required to be considered an
        improvement which prevents the algorithm from early stopping.
        early_stopping_tolerance is expressed as a percentage of the early
        stopping metric. Negative values indicate that the individual
        models should be overfit before stopping.
        EBMs are a bagged ensemble of models. Setting the early_stopping_tolerance
        to zero (or even negative), allows learning to overfit each of the individual
        models a little, which can improve the accuracy of the ensemble as a whole.
        Overfitting each of the individual models reduces the bias of each model at
        the expense of increasing the variance (due to overfitting) of the individual
        models.  But averaging the models in the ensemble reduces variance without
        much change in bias.  Since the goal is to find the optimum bias-variance
        tradeoff for the ensemble of models --- not the individual models --- a small
        amount of overfitting of the individual models can improve the accuracy of
        the ensemble as a whole.
    callback : Optional[Callable[[int, int, bool, float], bool]], default=None
        A user-defined function that is invoked at the end of each boosting step to determine
        whether to terminate boosting or continue. If it returns True, the boosting loop is
        stopped immediately. By default, no callback is used and training proceeds according
        to the early stopping settings.  The callback function receives:
        (1) the bag index, (2) the number of boosting steps completed,
        (3) a boolean indicating whether progress was made in the current step, and (4) the current best score.
    min_samples_leaf : int, default=4
        Minimum number of samples allowed in the leaves.
    min_hessian : float, default=0.0
        Minimum hessian required to consider a potential split valid.
    reg_alpha : float, default=0.0
        L1 regularization.
    reg_lambda : float, default=0.0
        L2 regularization.
    max_delta_step : float, default=0.0
        Used to limit the max output of tree leaves. <=0.0 means no constraint.
    gain_scale : float, default=5.0
        Scale factor to apply to nominal categoricals. A scale factor above 1.0 will cause the
        algorithm focus more on the nominal categoricals.
    min_cat_samples : int, default=10
        Minimum number of samples in order to treat a category separately. If lower than this threshold
        the category is combined with other categories that have low numbers of samples.
    cat_smooth : float, default=10.0
        Used for the categorical features. This can reduce the effect of noises in categorical features,
        especially for categories with limited data.
    missing: str, default="separate"

        Method for handling missing values during boosting. The placement of the missing value bin can influence
        the resulting model graphs. For example, placing the bin on the "low" side may cause missing values to
        affect lower bins, and vice versa. This parameter does not affect the final placement
        of the missing bin in the model (the missing bin will remain at index 0 in the term_scores\_ attribute).
        Possible values for missing are:

            - `'low'`: Place the missing bin on the left side of the graphs.
            - `'high'`: Place the missing bin on the right side of the graphs.
            - `'separate'`: Place the missing bin in its own leaf during each boosting step,
              effectively making it location-agnostic. This can lead to overfitting, especially
              when the proportion of missing values is small.
            - `'gain'`: Choose the best leaf for the missing value contribution at each boosting step, based on gain.
    max_leaves : int, default=2
        Maximum number of leaves allowed in each tree.
    monotone_constraints: list of int, default=None

        This parameter allows you to specify monotonic constraints for each feature's
        relationship with the target variable during model fitting. However, it is
        generally recommended to apply monotonic constraints post-fit using the
        `monotonize` function rather than setting them during the fitting process.
        This recommendation is based on the observation that, during fitting,
        the boosting algorithm may compensate for a monotone constraint on one
        feature by utilizing another correlated feature, potentially obscuring
        any monotonic violations.

        If you choose to define monotone constraints, `monotone_constraints`
        should be a list with a length equal to the number of features.
        Each element in the list corresponds to a feature and should take
        one of the following values:

            - 0: No monotonic constraint is imposed on the corresponding feature's partial response.
            - +1: The partial response of the corresponding feature should be monotonically increasing with respect to the target.
            - -1: The partial response of the corresponding feature should be monotonically decreasing with respect to the target.
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
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    histogram_edges\_ : List of None or array of float with shape ``(n_hist_edges,)``
        Per-feature list of the histogram edges. Categorical features contain None within the List
        at their feature index.
    histogram_weights\_ : List of array of float with shape ``(n_hist_bins,)``
        Per-feature list of the total sample weights within each feature's histogram bins.
    unique_val_counts\_ : array of int with shape ``(n_features,)``
        Per-feature count of the number of unique feature values.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the total sample weights in each term's tensor bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the bagged model scores.
    term_scores\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the model scores.
    standard_deviations\_ : List of array of float with shape ``(n_feature0_bins, ..., n_featureN_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_regression", "power", "identity", "log", "inverse", "inverse_square", "sqrt"
    link_param\_ : float
        Float value that can be used by the link function. The primary use is for the power link.
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0],
        and the count of interaction boosting iterations will be in best_iteration_[1].
    intercept\_ : float
        Intercept of the model.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
    min_target\_ : float
        The minimum value found in 'y'.
    max_target\_ : float
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
    best_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    histogram_edges_: List[Union[None, np.ndarray]]  # np.float64, 1D[hist_edge]
    histogram_weights_: List[np.ndarray]  # np.float64, 1D[hist_bin]
    unique_val_counts_: np.ndarray  # np.int64, 1D[feature]

    intercept_: float
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]
    min_target_: float
    max_target_: float

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ("global", "local")
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
        max_bins: int = 1024,
        max_interaction_bins: int = 64,
        # Stages
        interactions: Optional[
            Union[int, float, str, Sequence[Union[int, str, Sequence[Union[int, str]]]]]
        ] = "5x",
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = None,
        # Ensemble
        validation_size: Optional[Union[int, float]] = 0.15,
        outer_bags: int = 14,
        inner_bags: Optional[int] = 0,
        # Boosting
        learning_rate: float = 0.04,
        greedy_ratio: Optional[float] = 10.0,
        cyclic_progress: Union[bool, float, int] = False,  # noqa: PYI041
        smoothing_rounds: Optional[int] = 500,
        interaction_smoothing_rounds: Optional[int] = 100,
        max_rounds: Optional[int] = 50000,
        early_stopping_rounds: Optional[int] = 100,
        early_stopping_tolerance: Optional[float] = 1e-5,
        callback: Optional[Callable[[int, int, bool, float], bool]] = None,
        # Trees
        min_samples_leaf: Optional[int] = 4,
        min_hessian: Optional[float] = 0.0,
        reg_alpha: Optional[float] = 0.0,
        reg_lambda: Optional[float] = 0.0,
        max_delta_step: Optional[float] = 0.0,
        gain_scale: Optional[float] = 5.0,
        min_cat_samples: Optional[int] = 10,
        cat_smooth: Optional[float] = 10.0,
        missing: str = "separate",
        max_leaves: int = 2,
        monotone_constraints: Optional[Sequence[int]] = None,
        objective: str = "rmse",
        # Overall
        n_jobs: Optional[int] = -2,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
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
            greedy_ratio=greedy_ratio,
            cyclic_progress=cyclic_progress,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            callback=callback,
            min_samples_leaf=min_samples_leaf,
            min_hessian=min_hessian,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=max_delta_step,
            gain_scale=gain_scale,
            min_cat_samples=min_cat_samples,
            cat_smooth=cat_smooth,
            missing=missing,
            max_leaves=max_leaves,
            monotone_constraints=monotone_constraints,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=None,
            delta=None,
            composition=None,
            bin_budget_frac=None,
            privacy_bounds=None,
            privacy_target_min=None,
            privacy_target_max=None,
        )

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.

        """

        check_is_fitted(self, "has_fitted_")

        return inv_link(
            ebm_predict_scores(
                *clean_X_and_init_score(
                    X,
                    init_score,
                    self.feature_names_in_,
                    self.feature_types_in_,
                    self.link_,
                    self.link_param_,
                ),
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.intercept_,
                self.term_scores_,
                self.term_features_,
            ),
            self.link_,
            self.link_param_,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = EbmRegressorTags()
        return tags


class DPExplainableBoostingClassifier(ClassifierMixin, EBMModel):
    r"""Differentially Private Explainable Boosting Classifier.

    Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `'auto'`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=None
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
    classes\_ : array of bool, int, or unicode with shape ``(2,)``
        The class labels. DPExplainableBoostingClassifier only supports binary classification, so there are 2 classes.
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "monoclassification", "custom_binary", "custom_ovr", "custom_multinomial",
        "mlogit", "vlogit", "logit", "probit", "cloglog", "loglog", "cauchit"
    link_param\_ : float
        Float value that can be used by the link function. For classification it is only used by "custom_classification".
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0].
    intercept\_ : array of float with shape ``(1,)``
        Intercept of the model.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
    noise_scale_binning\_ : float
        The noise scale during binning.
    noise_scale_boosting\_ : float
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
    best_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    noise_scale_binning_: float
    noise_scale_boosting_: float

    classes_: np.ndarray  # np.int64, np.bool_, or np.unicode_, 1D[class]
    intercept_: np.ndarray  # np.float64, 1D[class]
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]

    available_explanations = ("global", "local")
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
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = None,
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
        super().__init__(
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
            greedy_ratio=0.0,
            cyclic_progress=True,
            smoothing_rounds=0,
            interaction_smoothing_rounds=0,
            max_rounds=max_rounds,
            early_stopping_rounds=0,
            early_stopping_tolerance=0.0,
            callback=None,
            min_samples_leaf=0,
            min_hessian=0.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            max_delta_step=0.0,
            gain_scale=1.0,
            min_cat_samples=0,
            cat_smooth=0.0,
            missing=None,
            max_leaves=max_leaves,
            monotone_constraints=None,
            objective="log_loss",
            n_jobs=n_jobs,
            random_state=random_state,
            epsilon=epsilon,
            delta=delta,
            composition=composition,
            bin_budget_frac=bin_budget_frac,
            privacy_bounds=privacy_bounds,
            privacy_target_min=None,
            privacy_target_max=None,
        )

    def predict_proba(self, X, init_score=None):
        """Probability estimates on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Probability estimate of sample for each class.

        """

        check_is_fitted(self, "has_fitted_")

        return inv_link(
            ebm_predict_scores(
                *clean_X_and_init_score(
                    X,
                    init_score,
                    self.feature_names_in_,
                    self.feature_types_in_,
                    self.link_,
                    self.link_param_,
                ),
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.intercept_,
                self.term_scores_,
                self.term_features_,
            ),
            self.link_,
            self.link_param_,
        )

    def decision_function(self, X, init_score=None):
        """Predict scores from model before calling the link function.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            The sum of the additive term contributions.

        """

        check_is_fitted(self, "has_fitted_")

        return ebm_predict_scores(
            *clean_X_and_init_score(
                X,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.link_,
                self.link_param_,
            ),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
        )

    def predict(self, X, init_score=None):
        """Predicts on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.

        """

        check_is_fitted(self, "has_fitted_")

        scores = ebm_predict_scores(
            *clean_X_and_init_score(
                X,
                init_score,
                self.feature_names_in_,
                self.feature_types_in_,
                self.link_,
                self.link_param_,
            ),
            self.feature_names_in_,
            self.feature_types_in_,
            self.bins_,
            self.intercept_,
            self.term_scores_,
            self.term_features_,
        )

        if scores.ndim == 1:
            # binary classification.  scikit-learn uses greater than semantics,
            # so score <= 0 means class_0, and 0 < score means class_1
            return self.classes_[(scores > 0).astype(np.int8)]
        # multiclass
        return self.classes_[np.argmax(scores, axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = EbmClassifierTags()
        return tags


class DPExplainableBoostingRegressor(RegressorMixin, EBMModel):
    r"""Differentially Private Explainable Boosting Regressor.

    Note that many arguments are defaulted differently than regular EBMs.

    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None

        List of feature types. For DP-EBMs, feature_types should be fully specified.
        The auto-detector, if used, examines the data and is not included in the privacy budget.
        If auto-detection is used, a privacy warning will be issued.
        FeatureType can be:

            - `'auto'`: Auto-detect (privacy budget is not respected!).
            - `'continuous'`: Use private continuous binning.
            - `[List of str]`: Ordinal categorical where the order has meaning. Eg: ["low", "medium", "high"]. Uses private categorical binning.
            - `'nominal'`: Categorical where the order has no meaning. Eg: country names. Uses private categorical binning.
    max_bins : int, default=32
        Max number of bins per feature.
    exclude : list of tuples of feature indices|names, default=None
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
    n_features_in\_ : int
        Number of features.
    feature_names_in\_ : List of str
        Resolved feature names. Names can come from feature_names, X, or be auto-generated.
    feature_types_in\_ : List of str
        Resolved feature types. Can be: 'continuous', 'nominal', or 'ordinal'.
    bins\_ : List[Union[List[Dict[str, int]], List[array of float with shape ``(n_cuts,)``]]]
        Per-feature list that defines how to bin each feature. Each feature in the list contains
        a list of binning resolutions. The first item in the binning resolution list is for binning
        main effect features. If there are more items in the binning resolution list, they define the
        binning for successive levels of resolutions. The item at index 1, if it exists, defines the
        binning for pairs. The last binning resolution defines the bins for all successive interaction levels.
        If the binning resolution list contains dictionaries, then the feature is either a 'nominal' or
        'ordinal' categorical. If the binning resolution list contains arrays, then the feature is 'continuous'
        and the arrays will contain float cut points that separate continuous values into bins.
    feature_bounds\_ : array of float with shape ``(n_features, 2)``
        min/max bounds for each feature. feature_bounds_[feature_index, 0] is the min value of the feature
        and feature_bounds_[feature_index, 1] is the max value of the feature. Categoricals have min & max
        values of NaN.
    term_features\_ : List of tuples of feature indices
        Additive terms used in the model and their component feature indices.
    term_names\_ : List of str
        List of term names.
    bin_weights\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the total sample weights in each term's bins.
    bagged_scores\_ : List of array of float with shape ``(n_outer_bags, n_bins)``
        Per-term list of the bagged model scores.
    term_scores\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the model scores.
    standard_deviations\_ : List of array of float with shape ``(n_bins)``
        Per-term list of the standard deviations of the bagged model scores.
    link\_ : str
        Link function used to convert the predictions or targets into linear space
        additive scores and vice versa via the inverse link. Possible values include:
        "custom_regression", "power", "identity", "log", "inverse", "inverse_square", "sqrt"
    link_param\_ : float
        Float value that can be used by the link function. The primary use is for the power link.
    bag_weights\_ : array of float with shape ``(n_outer_bags,)``
        Per-bag record of the total weight within each bag.
    best_iteration\_ : array of int with shape ``(n_stages, n_outer_bags)``
        The number of boosting iterations performed within each stage until either early stopping, or the max_rounds was reached.
        Normally, the count of main effects boosting iterations will be in best_iteration_[0].
    intercept\_ : float
        Intercept of the model.
    bagged_intercept\_ : array of float with shape ``(n_outer_bags,)``
        Bagged intercept of the model.
    min_target\_ : float
        The minimum value found in 'y', or privacy_target_min if provided.
    max_target\_ : float
        The maximum value found in 'y', or privacy_target_max if provided.
    noise_scale_binning\_ : float
        The noise scale during binning.
    noise_scale_boosting\_ : float
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
    best_iteration_: np.ndarray  # np.int64, 2D[stage, bag]

    noise_scale_binning_: float
    noise_scale_boosting_: float

    intercept_: float
    bagged_intercept_: np.ndarray  # np.float64, 1D[bag]
    min_target_: float
    max_target_: float

    # TODO PK v.3 use underscores here like RegressorMixin._estimator_type?
    available_explanations = ("global", "local")
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
        exclude: Optional[Sequence[Union[int, str, Sequence[Union[int, str]]]]] = None,
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
        super().__init__(
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
            greedy_ratio=0.0,
            cyclic_progress=True,
            smoothing_rounds=0,
            interaction_smoothing_rounds=0,
            max_rounds=max_rounds,
            early_stopping_rounds=0,
            early_stopping_tolerance=0.0,
            callback=None,
            min_samples_leaf=0,
            min_hessian=0.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            max_delta_step=0.0,
            gain_scale=1.0,
            min_cat_samples=0,
            cat_smooth=0.0,
            missing=None,
            max_leaves=max_leaves,
            monotone_constraints=None,
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
        """Predict on provided samples.

        Args:
            X: NumPy array for samples.
            init_score: Optional. Either a model that can generate scores or per-sample initialization score.
                If samples scores it should be the same length as X.

        Returns:
            Predicted class label per sample.

        """

        check_is_fitted(self, "has_fitted_")

        return inv_link(
            ebm_predict_scores(
                *clean_X_and_init_score(
                    X,
                    init_score,
                    self.feature_names_in_,
                    self.feature_types_in_,
                    self.link_,
                    self.link_param_,
                ),
                self.feature_names_in_,
                self.feature_types_in_,
                self.bins_,
                self.intercept_,
                self.term_scores_,
                self.term_features_,
            ),
            self.link_,
            self.link_param_,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = EbmRegressorTags()
        return tags
