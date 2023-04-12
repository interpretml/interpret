# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from math import ceil, floor, isnan, isinf, exp, log
from ...utils._native import Native, Booster

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier
import numbers
import numpy as np
import warnings
from itertools import islice, count, chain

import heapq

import logging

_log = logging.getLogger(__name__)


def boost(
    dataset,
    bag,
    init_scores,
    term_features,
    n_inner_bags,
    boost_flags,
    learning_rate,
    min_samples_leaf,
    max_leaves,
    greediness,
    smoothing_rounds,
    max_rounds,
    early_stopping_rounds,
    early_stopping_tolerance,
    noise_scale,
    bin_weights,
    rng,
    objective,
    experimental_params=None,
):
    episode_index = 0
    with Booster(
        dataset,
        bag,
        init_scores,
        term_features,
        n_inner_bags,
        rng,
        objective,
        experimental_params,
    ) as booster:
        # the first round is alwasy cyclic since we need to get the initial gains
        greedy_portion = 0.0

        min_metric = np.inf
        no_change_run_length = 0
        bp_metric = np.inf
        _log.info("Start boosting")
        native = Native.get_native_singleton()

        for episode_index in range(max_rounds):
            if episode_index % 10 == 0:
                _log.debug("Sweep Index {0}".format(episode_index))
                _log.debug("Metric: {0}".format(min_metric))

            if greedy_portion < 1.0:
                # we're doing a cyclic round
                heap = []

            boost_flags_local = boost_flags
            if 0 < smoothing_rounds:
                # modify some of our parameters temporarily
                boost_flags_local |= (
                    Native.BoostFlags_DisableNewtonGain
                    | Native.BoostFlags_DisableNewtonUpdate
                    | Native.BoostFlags_RandomSplits
                )

            for term_idx in range(len(term_features)):
                if 1.0 <= greedy_portion:
                    # we're being greedy, so select something from our
                    # queue and overwrite the term_idx we'll work on
                    _, term_idx = heapq.heappop(heap)

                avg_gain = booster.generate_term_update(
                    term_idx=term_idx,
                    boost_flags=boost_flags_local,
                    learning_rate=learning_rate,
                    min_samples_leaf=min_samples_leaf,
                    max_leaves=max_leaves,
                )

                heapq.heappush(heap, (-avg_gain, term_idx))

                if noise_scale:  # Differentially private updates
                    splits = booster.get_term_update_splits()[0]

                    term_update_tensor = booster.get_term_update()
                    noisy_update_tensor = term_update_tensor.copy()

                    # Make splits iteration friendly
                    splits_iter = [0] + list(splits + 1) + [len(term_update_tensor)]

                    n_sections = len(splits_iter) - 1
                    noises = native.generate_gaussian_random(
                        rng, noise_scale, n_sections
                    )

                    # Loop through all random splits and add noise before updating
                    for f, s, noise in zip(splits_iter[:-1], splits_iter[1:], noises):
                        if s == 1:
                            # Skip cuts that fall on 0th (missing value) bin -- missing values not supported in DP
                            continue

                        noisy_update_tensor[f:s] = term_update_tensor[f:s] + noise

                        # Native code will be returning sums of residuals in slices, not averages.
                        # Compute noisy average by dividing noisy sum by noisy bin weights
                        region_weight = np.sum(bin_weights[term_idx][f:s])
                        noisy_update_tensor[f:s] = (
                            noisy_update_tensor[f:s] / region_weight
                        )

                    # Invert gradients before updates
                    noisy_update_tensor = -noisy_update_tensor
                    booster.set_term_update(term_idx, noisy_update_tensor)

                cur_metric = booster.apply_term_update()

                min_metric = min(cur_metric, min_metric)

            # TODO PK this early_stopping_tolerance is a little inconsistent
            #      since it triggers intermittently and only re-triggers if the
            #      threshold is re-passed, but not based on a smooth windowed set
            #      of checks.  We can do better by keeping a list of the last
            #      number of measurements to have a consistent window of values.
            #      If we only cared about the metric at the start and end of the epoch
            #      window a circular buffer would be best choice with O(1).
            if no_change_run_length == 0:
                bp_metric = min_metric

            # TODO: PK, I think this is a bug and the first iteration no_change_run_length
            # get incremented to 1, so if early_stopping_rounds is 1 it will always
            # exit on the first round? I haven't changed it since it's just going to affect 1
            # and changing it would change the results so I need to benchmark update it
            if min_metric + early_stopping_tolerance < bp_metric:
                no_change_run_length = 0
            else:
                no_change_run_length += 1

            if 1.0 <= greedy_portion:
                greedy_portion -= 1.0

            if 0 < smoothing_rounds:
                # disable early stopping progress during the smoothing rounds since
                # cuts are chosen randomly, which will lead to high variance on the
                # validation metric
                no_change_run_length = 0
                smoothing_rounds -= 1
            else:
                # do not progress into greedy rounds until we're done with the smoothing_rounds
                greedy_portion += greediness

            if (
                early_stopping_rounds > 0
                and no_change_run_length >= early_stopping_rounds
            ):
                break

        _log.info(
            "End boosting, Best Metric: {0}, Num Rounds: {1}".format(
                min_metric, episode_index
            )
        )

        if early_stopping_rounds > 0:
            model_update = booster.get_best_model()
        else:
            model_update = booster.get_current_model()

    return model_update, episode_index, rng
