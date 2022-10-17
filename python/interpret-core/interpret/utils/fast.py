# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

""" FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""

import numpy as np
from itertools import combinations

from sklearn.base import is_classifier
from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor

from ..glassbox.ebm.bin import clean_X, clean_vector, construct_bins, bin_native_by_dimension
from ._native import Native, InteractionDetector

def _prepare_X_y(X, y, is_classification):
    X, n_samples = clean_X(X)
    if n_samples == 0:
        raise ValueError("X has 0 samples")

    if is_classification:
        y = clean_vector(y, True, "y")
        classes, y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
    else:
        y = clean_vector(y, False, "y")
        n_classes = -1

    if n_samples != len(y):
        raise ValueError(f"X has {n_samples} samples and y has {len(y)} samples")

    return X, y, n_samples, n_classes

def _prepare_sample_weight(sample_weight, num_samples):
    sample_weight = clean_vector(sample_weight, False, "sample_weight")
    if num_samples != len(sample_weight):
        raise ValueError(f"X has {num_samples} samples and sample_weight has {len(sample_weight)} samples")
    return sample_weight

def _get_scores(X, init_scores, init_model):
    # TODO: remove this
    if init_model is not None:
        if is_classifier(init_model):
            scores = init_model.decision_function(X)
        else:
            # TODO Add link function to operate on predict's output when needed
            scores = init_model.predict(X)
    else:
        scores = init_scores

    if scores is not None:
        return clean_vector(scores, False, "scores")
    return None

def _get_ranked_interactions(
        dataset,
        bag,
        scores,
        iter_term_features,
        interaction_flags,
        min_samples_leaf,
        experimental_params=None,
        num_output_interactions=0
    ):
    interaction_strengths = []
    with InteractionDetector(dataset, bag, scores, experimental_params) as interaction_detector:
        for feature_idxs in iter_term_features:
            strength = interaction_detector.calc_interaction_strength(
                feature_idxs, interaction_flags, min_samples_leaf,
            )
            interaction_strengths.append((strength, feature_idxs))

    interaction_strengths.sort(reverse=True)

    num_interactions = len(interaction_strengths)
    if (num_output_interactions > 0):
        num_interactions = min(num_output_interactions, num_interactions)

    # TODO put this in a priority queue to reduce memory consumption which might be important for tripples
    return interaction_strengths[:num_interactions]

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
            Either a list of lists of feature indices, or an integer for the max number of pairs returned.
            None evaluates all pairwise interactions
        init_score: Either a model that can generate scores or per-sample initialization score. 
            If samples scores it should be the same length as X and y.
        sample_weight: Optional array of weights per sample. Should be the same length as X and y
        feature_names: List of feature names
        feature_types: List of feature types, for example "continuous" or "nominal"
        max_interaction_bins: Max number of bins per interaction terms
        binning: Method to bin values for pre-processing - "uniform", "quantile", or "rounded_quantile".
            'rounded_quantile' will round to as few decimals as possible while preserving the same bins as 'quantile'.
        min_samples_leaf: Minimum number of cases for tree splits used in boosting
        objective: 'regression' (RMSE) or 'classification' (log loss) or None for auto
    Returns:
        Dictionary with a pair of indices as keys and strengths as values, e.g. { (1, 2) : 0.134 }.
            Ordered by decreasing strengths
    """

    is_classification = None
    if objective in ['classification']:
        is_classification = True
    elif objective in ['regression']:
        is_classification = False

    if is_classifier(init_score):
        if is_classification == False:
            raise ValueError("objective is for regresion but the init_score is a classification model")
        is_classification = True
        try:
            init_score = np.array(init_score.decision_function(X), dtype=np.float64)
        except AttributeError:
            probs = np.array(init_score.predict_proba(X), dtype=np.float64)
            maxes = np.amax(probs, axis=1)
            init_score = np.log(probs / maxes[:,np.newaxis])
            if init_score.shape[1] == 2: # binary classification
                init_score = init_score[:,1] - init_score[:,0]
    elif is_regressor(init_score):
        if is_classification == True:
            raise ValueError("objective is for classification but the init_score is a regression model")
        is_classification = False
        init_score = np.array(init_score.predict(X), dtype=np.float64)
        init_score = clean_vector(init_score, False, "init_score")
        # TODO Add link function to operate on predict's output when needed
    elif init_score is not None:
        init_score = np.array(init_score, dtype=np.float64)

    if is_classification is None:
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            is_classification = True
        elif target_type in ['continuous']:
            is_classification = False
        else:
            raise ValueError("unrecognized target type in y")

    # num_classes is -1 for regression
    X, y, num_samples, num_classes = _prepare_X_y(X, y, is_classification)

    if init_score is not None:
        if is_classification:
            if num_classes == 2:
                # should be 1 dimensionable, and our clean_vector will force that
                init_score = clean_vector(init_score, False, "init_score")
            elif init_score.ndim != 2:
                raise ValueError("for multiclass init_score should have 2 dimensions")
            elif init_score.shape[1] != num_classes:
                raise ValueError("for multiclass init_score.shape[1] should be the number of classes")
        else:
            # clean_vector will fail if it is not one dimenionable
            init_score = clean_vector(init_score, False, "init_score")

    if sample_weight is not None:
        sample_weight = _prepare_sample_weight(sample_weight, num_samples)

    binning_result = construct_bins(
        X=X,
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
        n_classes=num_classes,
        n_dimensions=2,
        bins=bins,
        X=X,
        y=y,
        sample_weight=sample_weight,
        feature_names_in=feature_names_in,
        feature_types_in=feature_types_in
    )

    if isinstance(interactions, int):
        num_output_interactions = interactions
        iter_term_features=combinations(range(n_features_in), 2)
    elif interactions is None:
        num_output_interactions = 0
        iter_term_features=combinations(range(n_features_in), 2)
    else:
        num_output_interactions = 0
        iter_term_features = interactions

    ranked_interactions = _get_ranked_interactions(
        dataset=dataset,
        bag=None,
        scores=init_score,
        iter_term_features=iter_term_features,
        interaction_flags=Native.InteractionFlags_Pure,
        min_samples_leaf=min_samples_leaf,
        experimental_params=None,
        num_output_interactions=num_output_interactions
    )

    ranked_interactions_dict = {}
    for strength, indices in ranked_interactions:
        ranked_interactions_dict[indices] = strength

    return ranked_interactions_dict