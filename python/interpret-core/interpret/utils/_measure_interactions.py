# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

""" FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""

import logging

from itertools import count
import numpy as np
from itertools import combinations

from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor

from ._clean_x import preclean_X
from ._clean_simple import (
    clean_dimensions,
    typify_classification,
    clean_init_score_and_X,
)

from ._preprocessor import construct_bins
from ._native import Native
from ._compressed_dataset import bin_native_by_dimension

from ._rank_interactions import rank_interactions

_log = logging.getLogger(__name__)


def measure_interactions(
    X,
    y,
    interactions=None,
    init_score=None,
    sample_weight=None,
    feature_names=None,
    feature_types=None,
    max_interaction_bins=32,
    min_hessian=1e-3,
    objective=None,
):
    """Run the FAST algorithm and return the ranked interactions and their strengths as a dictionary.

    Args:
        X: Array of training samples
        y: Array of training targets
        interactions: Interactions to evaluate
            Either a list of tuples of feature indices, or an integer for the max number of pairs returned.
            None evaluates all pairwise interactions
        init_score: Either a model that can generate scores or per-sample initialization score.
            If samples scores it should be the same length as X and y.
        sample_weight: Optional array of weights per sample. Should be the same length as X and y
        feature_names: List of feature names
        feature_types: List of feature types, for example "continuous" or "nominal"
        max_interaction_bins: Max number of bins per interaction terms
        min_hessian: Minimum hessian required to consider a potential split valid
        objective: None (rmse or log_loss), "rmse" (regression default), "log_loss" (classification default),
            "poisson_deviance", "tweedie_deviance:variance_power=1.5", "gamma_deviance",
            "pseudo_huber:delta=1.0", "rmse_log" (rmse with a log link function)
    Returns:
        List containing a tuple of feature indices for the terms and interaction strengths,
            e.g. [((1, 2), 0.134), ((3, 7), 0.0842)].  Ordered by decreasing interaction strengths.
    """

    # with 64 bytes per tensor cell, a 2^20 tensor would be 1/16 gigabyte.
    max_cardinality = 1048576

    y = clean_dimensions(y, "y")
    if y.ndim != 1:
        msg = "y must be 1 dimensional"
        _log.error(msg)
        raise ValueError(msg)
    if len(y) == 0:
        msg = "y cannot have 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    is_differential_privacy = False

    flags = (
        Native.LinkFlags_DifferentialPrivacy
        if is_differential_privacy
        else Native.LinkFlags_Default
    )

    native = Native.get_native_singleton()

    task = None
    if objective is not None:
        if len(objective.strip()) == 0:
            objective = None
        else:
            # "classification" or "regression"
            task = native.determine_task(objective)

    classes = None
    link = None
    link_param = None
    if is_classifier(init_score):
        # all scikit-learn classification models need to expose self.classes_
        classes = init_score.classes_
        y = typify_classification(y)

        invert_classes = dict(zip(classes, count()))
        y = np.array([invert_classes[el] for el in y], dtype=np.int64)

        if task is None:
            task = "classification"
            link, link_param = native.determine_link(flags, "log_loss", len(classes))
        elif task == "classification":
            link, link_param = native.determine_link(flags, objective, len(classes))
        else:
            raise ValueError(
                f"init_score is a classifier, but the objective is: {objective}"
            )
    elif is_regressor(init_score):
        if task is None:
            task = "regression"
            link, link_param = native.determine_link(
                flags, "rmse", Native.Task_Regression
            )
        elif task == "regression":
            link, link_param = native.determine_link(
                flags, objective, Native.Task_Regression
            )
        else:
            raise ValueError(
                f"init_score is a regressor, but the objective is: {objective}"
            )
    elif task == "classification":
        y = typify_classification(y)
        # scikit-learn requires that the self.classes_ are sorted with np.unique, so rely on this
        classes, y = np.unique(y, return_inverse=True)
        link, link_param = native.determine_link(flags, objective, len(classes))

    init_score, X, n_samples = clean_init_score_and_X(
        link, link_param, init_score, X, feature_names, feature_types, len(y)
    )
    if init_score is not None and init_score.ndim == 2:
        # it must be multiclass, or mono-classification
        if task is None:
            task = "classification"
        elif task != "classification":
            raise ValueError(
                f"init_score has 2 dimensions so it is a multiclass model, but the objective is: {objective}"
            )

    if task is None:
        # type_of_target does not seem to like np.object_, so convert it to something that works
        try:
            y_discard = y.astype(dtype=np.float64, copy=False)
        except (TypeError, ValueError):
            y_discard = y.astype(dtype=np.unicode_, copy=False)

        target_type = type_of_target(y_discard)
        if target_type == "continuous":
            task = "regression"
        elif target_type == "binary":
            task = "classification"
        elif target_type == "multiclass":
            if init_score is not None:
                # type_of_target is guessing the model type. if init_score was multiclass then it
                # should have a 2nd dimension, but it does not, so the guess made by type_of_target was wrong.
                # The only other option is for it to be regression, so force that.
                task = "regression"
            else:
                task = "classification"
        else:
            raise ValueError("unrecognized target type in y")

    if task == "classification":
        if classes is None:
            y = typify_classification(y)
            # scikit-learn requires that the self.classes_ are sorted with np.unique, so rely on this
            classes, y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        if objective is None:
            objective = "log_loss"
    elif task == "regression":
        y = y.astype(np.float64, copy=False)
        n_classes = Native.Task_Regression
        if objective is None:
            objective = "rmse"
    else:
        msg = f"Unknown objective {objective}"
        _log.error(msg)
        raise ValueError(msg)

    if init_score is not None:
        if n_classes == 2 or n_classes < 0:
            if init_score.ndim != 1:
                raise ValueError(
                    "diagreement between the number of classes in y and in the init_score shape"
                )
        elif 3 <= n_classes:
            if init_score.ndim != 2 or init_score.shape[1] != n_classes:
                raise ValueError(
                    "diagreement between the number of classes in y and in the init_score shape"
                )
        else:  # 1 class
            # what the init_score should be for mono-classifiction is somewhat abiguous,
            # so allow either 0 or 1 (which means the dimension is eliminated)
            if init_score.ndim == 2 and 2 <= init_score.shape[1]:
                raise ValueError(
                    "diagreement between the number of classes in y and in the init_score shape"
                )
            init_score = None

    if sample_weight is not None:
        sample_weight = clean_dimensions(sample_weight, "sample_weight")
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be 1 dimensional")
        if len(y) != len(sample_weight):
            raise ValueError(
                f"y has {len(y)} samples and sample_weight has {len(sample_weight)} samples"
            )
        sample_weight = sample_weight.astype(np.float64, copy=False)

    binning_result = construct_bins(
        X=X,
        y=y,
        sample_weight=sample_weight,
        feature_names_given=feature_names,
        feature_types_given=feature_types,
        max_bins_leveled=[max_interaction_bins],
        binning="quantile",
        min_samples_bin=1,
        min_unique_continuous=0,
    )

    feature_names_in = binning_result[0]
    feature_types_in = binning_result[1]
    bins = binning_result[2]
    n_features_in = len(bins)

    dataset = bin_native_by_dimension(
        n_classes=n_classes,
        n_dimensions=2,
        bins=bins,
        X=X,
        y=y,
        sample_weight=sample_weight,
        feature_names_in=feature_names_in,
        feature_types_in=feature_types_in,
    )

    if isinstance(interactions, int):
        n_output_interactions = interactions
        iter_term_features = combinations(range(n_features_in), 2)
    elif interactions is None:
        n_output_interactions = 0
        iter_term_features = combinations(range(n_features_in), 2)
    else:
        n_output_interactions = 0
        iter_term_features = interactions

    # TODO: benchmarking indicates that using CalcInteractionFlags_Pure is
    # slightly worse than not using it when boosting pairs AFTER mains, but
    # I'm leaving it here for now until we can benchmark the scenario where
    # interaction detection is used without fitting any mains which someone
    # could do using this interaface currently, but is not possible in regular EBMs.
    ranked_interactions = rank_interactions(
        dataset=dataset,
        bag=None,
        init_scores=init_score,
        iter_term_features=iter_term_features,
        exclude=set(),
        calc_interaction_flags=Native.CalcInteractionFlags_Pure,
        max_cardinality=max_cardinality,
        min_hessian=min_hessian,
        create_interaction_flags=(
            Native.CreateInteractionFlags_DifferentialPrivacy
            if is_differential_privacy
            else Native.CreateInteractionFlags_Default
        ),
        objective=objective,
        experimental_params=None,
        n_output_interactions=n_output_interactions,
    )

    if isinstance(ranked_interactions, Exception):
        raise ranked_interactions

    return list(map(tuple, map(reversed, ranked_interactions)))
