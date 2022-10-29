# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

""" FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""

import numpy as np
import heapq
from itertools import combinations

from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor

from ._binning import determine_min_cols, clean_X, clean_dimensions, typify_classification, clean_init_score, construct_bins, bin_native_by_dimension
from ._native import Native, InteractionDetector

def _get_ranked_interactions(
        dataset,
        bag,
        scores,
        iter_term_features,
        interaction_flags,
        min_samples_leaf,
        experimental_params=None,
        n_output_interactions=0
    ):
    interaction_strengths = []
    with InteractionDetector(dataset, bag, scores, experimental_params) as interaction_detector:
        for feature_idxs in iter_term_features:
            strength = interaction_detector.calc_interaction_strength(
                feature_idxs, interaction_flags, min_samples_leaf,
            )
            item = (strength, feature_idxs)
            if(n_output_interactions <= 0):
                interaction_strengths.append(item)
            else:
                if len(interaction_strengths) == n_output_interactions:
                    heapq.heappushpop(interaction_strengths, item)
                else:
                    heapq.heappush(interaction_strengths, item)

    interaction_strengths.sort(reverse=True)
    return interaction_strengths

def measure_interactions(
        X,
        y,
        interactions=None,
        init_score=None,
        sample_weight=None,
        feature_names=None,
        feature_types=None,
        max_interaction_bins=32,
        binning='quantile',
        min_samples_leaf=2,
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
        binning: Method to bin values for pre-processing - "uniform", "quantile", or "rounded_quantile".
        min_samples_leaf: Minimum number of samples for tree splits used when calculating gain
        objective: 'regression' (RMSE) or 'classification' (log loss) or None for auto. More objectives to come
    Returns:
        List containing a tuple of feature indices for the terms and interaction strengths, 
            e.g. [((1, 2), 0.134), ((3, 7), 0.0842)].  Ordered by decreasing interaction strengths.
    """
        
    y = clean_dimensions(y, "y")
    if y.ndim != 1:
        msg = "y must be 1 dimensional"
        _log.error(msg)
        raise ValueError(msg)
    if len(y) == 0:
        msg = "y cannot have 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    is_classification = None
    if objective in ['classification']:
        is_classification = True
    elif objective in ['regression']:
        is_classification = False

    if is_classifier(init_score):
        if is_classification == False:
            raise ValueError("objective is for regresion but the init_score is a classification model")
        is_classification = True
    elif is_regressor(init_score):
        if is_classification == True:
            raise ValueError("objective is for classification but the init_score is a regression model")
        is_classification = False

    if init_score is not None:
        # use the uncleaned X since scikit-learn Estimators need the original data format
        init_score = clean_init_score(init_score, len(y), X)
        if init_score.ndim == 2:
            # it must be multiclass, or mono-classification
            if is_classification == False:
                raise ValueError("objective is for regresion but the init_score is for a multiclass model")
            is_classification = True

    if is_classification is None:
        # type_of_target does not seem to like np.object_, so convert it to something that works
        try:
            y_discard = y.astype(dtype=np.float64, copy=False)
        except (TypeError, ValueError):
            y_discard = y.astype(dtype=np.unicode_, copy=False)

        target_type = type_of_target(y_discard)
        if target_type == 'continuous':
            is_classification = False
        elif target_type == 'binary':
            is_classification = True
        elif target_type == 'multiclass':
            if init_score is not None:
                # type_of_target is guessing the model type. if init_score was multiclass then it
                # should have a 2nd dimension, but it does not, so the guess made by type_of_target was wrong. 
                # The only other option is for it to be regression, so force that.
                is_classification = False
            else:
                is_classification = True
        else:
            raise ValueError("unrecognized target type in y")

    if is_classification:
        y = typify_classification(y)
        classes, y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
    else:
        y = y.astype(np.float64, copy=False)
        n_classes = -1

    if init_score is not None:
        if n_classes == 2 or n_classes == -1:
            if init_score.ndim != 1:
                raise ValueError("diagreement between the number of classes in y and in the init_score shape")
        elif 3 <= n_classes:
            if init_score.ndim != 2 or init_score.shape[1] != n_classes:
                raise ValueError("diagreement between the number of classes in y and in the init_score shape")
        else: # 1 class
            # what the init_score should be for mono-classifiction is somewhat abiguous, 
            # so allow either 0 or 1 (which means the dimension is eliminated)
            if init_score.ndim == 2 and 2 <= init_score.shape[1]:
                raise ValueError("diagreement between the number of classes in y and in the init_score shape")
            init_score = None

    if sample_weight is not None:
        sample_weight = clean_dimensions(sample_weight, "sample_weight")
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be 1 dimensional")
        if len(y) != len(sample_weight):
            raise ValueError(f"y has {len(y)} samples and sample_weight has {len(sample_weight)} samples")
        sample_weight = sample_weight.astype(np.float64, copy=False)

    min_cols = determine_min_cols(feature_names, feature_types)
    X, n_samples = clean_X(X, min_cols, len(y))

    binning_result = construct_bins(
        X=X,
        y=y,
        sample_weight=sample_weight,
        feature_names_given=feature_names,
        feature_types_given=feature_types,
        max_bins_leveled=[max_interaction_bins],
        binning=binning
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
        feature_types_in=feature_types_in
    )

    if isinstance(interactions, int):
        n_output_interactions = interactions
        iter_term_features=combinations(range(n_features_in), 2)
    elif interactions is None:
        n_output_interactions = 0
        iter_term_features=combinations(range(n_features_in), 2)
    else:
        n_output_interactions = 0
        iter_term_features = interactions

    ranked_interactions = _get_ranked_interactions(
        dataset=dataset,
        bag=None,
        scores=init_score,
        iter_term_features=iter_term_features,
        interaction_flags=Native.InteractionFlags_Pure,
        min_samples_leaf=min_samples_leaf,
        experimental_params=None,
        n_output_interactions=n_output_interactions
    )

    return list(map(tuple, map(reversed, ranked_interactions)))
