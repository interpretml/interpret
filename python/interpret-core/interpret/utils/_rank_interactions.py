# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""FAST - Interaction Detection

This module exposes a method called FAST [1] to measure and rank the strengths
of the interaction of all pairs of features in a dataset.

[1] https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf
"""

import heapq

from ._native import InteractionDetector
from .. import develop


def rank_interactions(
    dataset,
    intercept,
    bag,
    init_scores,
    iter_term_features,
    exclude,
    exclude_features,
    calc_interaction_flags,
    max_cardinality,
    min_samples_leaf,
    min_hessian,
    reg_alpha,
    reg_lambda,
    max_delta_step,
    create_interaction_flags,
    objective,
    acceleration,
    experimental_params,
    n_output_interactions,
    develop_options,
):
    try:
        develop._develop_options = develop_options  # restore these in this process
        interaction_strengths = []
        with InteractionDetector(
            dataset,
            intercept,
            bag,
            init_scores,
            create_interaction_flags,
            objective,
            acceleration,
            experimental_params,
        ) as interaction_detector:
            for feature_idxs in iter_term_features:
                if tuple(sorted(feature_idxs)) in exclude:
                    continue
                if any(i in exclude_features for i in feature_idxs):
                    continue

                strength = interaction_detector.calc_interaction_strength(
                    feature_idxs,
                    calc_interaction_flags,
                    max_cardinality,
                    min_samples_leaf,
                    min_hessian,
                    reg_alpha,
                    reg_lambda,
                    max_delta_step,
                )
                item = (strength, feature_idxs)
                if n_output_interactions <= 0:
                    interaction_strengths.append(item)
                elif len(interaction_strengths) == n_output_interactions:
                    heapq.heappushpop(interaction_strengths, item)
                else:
                    heapq.heappush(interaction_strengths, item)

        interaction_strengths.sort(reverse=True)
        return interaction_strengths
    except Exception as e:
        return e
