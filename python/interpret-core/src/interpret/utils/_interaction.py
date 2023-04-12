# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

""" FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""

from itertools import count
import numpy as np
import heapq
from itertools import combinations

from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor

from ._clean_x import preclean_X
from ._clean_simple import clean_dimensions, typify_classification, clean_init_score

from ._preprocessor import construct_bins
from ._native import Native, InteractionDetector
from ._compressed_dataset import bin_native_by_dimension


def rank_interactions(
    dataset,
    bag,
    init_scores,
    iter_term_features,
    exclude,
    interaction_flags,
    max_cardinality,
    min_samples_leaf,
    objective,
    experimental_params=None,
    n_output_interactions=0,
):
    interaction_strengths = []
    with InteractionDetector(
        dataset, bag, init_scores, objective, experimental_params
    ) as interaction_detector:
        for feature_idxs in iter_term_features:
            if tuple(sorted(feature_idxs)) in exclude:
                continue
            strength = interaction_detector.calc_interaction_strength(
                feature_idxs,
                interaction_flags,
                max_cardinality,
                min_samples_leaf,
            )
            item = (strength, feature_idxs)
            if n_output_interactions <= 0:
                interaction_strengths.append(item)
            else:
                if len(interaction_strengths) == n_output_interactions:
                    heapq.heappushpop(interaction_strengths, item)
                else:
                    heapq.heappush(interaction_strengths, item)

    interaction_strengths.sort(reverse=True)
    return interaction_strengths
