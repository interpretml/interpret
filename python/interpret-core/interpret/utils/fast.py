""" FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""
import numpy as np
from itertools import combinations

from sklearn.base import is_classifier

from ..glassbox.ebm.bin import clean_X, clean_vector, construct_bins, bin_native_by_dimension
from ..glassbox.ebm.internal import Native, InteractionDetector

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

def measure_interactions(X,
         y,
         is_classification,
         sample_weight=None,
         feature_names=None,
         feature_types=None,
         num_output_interactions=0,
         max_interaction_bins=32,
         binning='quantile',
         min_samples_leaf=2,
         **kwargs
    ):
    """Run the FAST algorithm and return the ranked interactions and their strengths as a dictionary.

    Args:
        X (numpy array): Array for training samples
        y (numpy array): Array as training labels
        is_classification: True if the task is a classification task, False otherwise
        sample_weight (numpy array): Optional array of weights per sample. Should be the same length as X and y
        feature_names: List of feature names
        feature_types: List of feature types, for example "continuous" or "nominal"
        num_output_interactions: Number of ranked interactions returned by the function. Set 0 for all interactions.
        max_interaction_bins: Max number of bins per interaction terms
        binning: Method to bin values for pre-processing - "uniform", "quantile", or "rounded_quantile".
           'rounded_quantile' will round to as few decimals as possible while preserving the same bins as 'quantile'.
        min_samples_leaf: Minimum number of cases for tree splits used in boosting
    Returns:
        Dictionary with a pair of indices as keys and strengths as values, e.g. { (1, 2) : 0.134 }
    """
    if is_classification is None:
        raise ValueError("is_classification should be provided.")

    init_scores = kwargs.get("init_scores", None)
    init_model = kwargs.get("init_model", None)
    if init_model is not None:
        if is_classification != is_classifier(init_model):
            raise ValueError(f"is_classification is {is_classification} bu init_model's task type is {is_classifier(init_model)}.")

    scores = _get_scores(X, init_scores, init_model)

    # num_classes is -1 for regression
    X, y, num_samples, num_classes = _prepare_X_y(X, y, is_classification)

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

    ranked_interactions = _get_ranked_interactions(
        dataset=dataset,
        bag=None,
        scores=scores,
        iter_term_features=combinations(range(n_features_in), 2),
        interaction_flags=Native.InteractionFlags_Pure,
        min_samples_leaf=min_samples_leaf,
        experimental_params=None,
        num_output_interactions=num_output_interactions
    )

    ranked_interactions_dict = {}
    for strength, indices in ranked_interactions:
        ranked_interactions_dict[indices] = strength

    return ranked_interactions_dict