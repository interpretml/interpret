# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ...utils._native import Native, Booster

import numpy as np

import heapq

import logging

_log = logging.getLogger(__name__)


def boost(
    dataset,
    bag,
    init_scores,
    term_features,
    n_inner_bags,
    term_boost_flags,
    learning_rate,
    min_samples_leaf,
    min_hessian,
    max_leaves,
    greediness,
    smoothing_rounds,
    max_rounds,
    early_stopping_rounds,
    early_stopping_tolerance,
    noise_scale,
    bin_weights,
    rng,
    create_booster_flags,
    objective,
    experimental_params=None,
):
    try:
        episode_index = 0
        with Booster(
            dataset,
            bag,
            init_scores,
            term_features,
            n_inner_bags,
            rng,
            create_booster_flags,
            objective,
            experimental_params,
        ) as booster:
            # the first round is alwasy cyclic since we need to get the initial gains
            greedy_portion = 0.0

            min_metric = np.inf
            min_prev_metric = np.inf
            circular = np.full(
                early_stopping_rounds * len(term_features), np.inf, np.float64
            )
            circular_index = 0

            _log.info("Start boosting")
            native = Native.get_native_singleton()

            for episode_index in range(max_rounds):
                if episode_index % 10 == 0:
                    _log.debug("Sweep Index {0}".format(episode_index))
                    _log.debug("Metric: {0}".format(min_metric))

                if greedy_portion < 1.0:
                    # we're doing a cyclic round
                    heap = []

                term_boost_flags_local = term_boost_flags
                if 0 < smoothing_rounds:
                    # modify some of our parameters temporarily
                    term_boost_flags_local |= Native.TermBoostFlags_RandomSplits

                for term_idx in range(len(term_features)):
                    if 1.0 <= greedy_portion:
                        # we're being greedy, so select something from our
                        # queue and overwrite the term_idx we'll work on
                        _, term_idx = heapq.heappop(heap)

                    avg_gain = booster.generate_term_update(
                        rng,
                        term_idx=term_idx,
                        term_boost_flags=term_boost_flags_local,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        min_hessian=min_hessian,
                        max_leaves=max_leaves,
                    )

                    heapq.heappush(heap, (-avg_gain, term_idx))

                    if noise_scale:  # Differentially private updates
                        splits = booster.get_term_update_splits()[0]

                        term_update_tensor = booster.get_term_update()
                        noisy_update_tensor = term_update_tensor.copy()

                        # Make splits iteration friendly
                        splits_iter = [0] + list(splits) + [len(term_update_tensor)]

                        n_sections = len(splits_iter) - 1
                        noises = native.generate_gaussian_random(
                            rng, noise_scale, n_sections
                        )

                        # Loop through all random splits and add noise before updating
                        for f, s, noise in zip(
                            splits_iter[:-1], splits_iter[1:], noises
                        ):
                            noisy_update_tensor[f:s] = term_update_tensor[f:s] + noise

                            # Native code will be returning sums of residuals in slices, not averages.
                            # Compute noisy average by dividing noisy sum by noisy bin weights
                            region_weight = np.sum(
                                bin_weights[term_features[term_idx][0]][f:s]
                            )
                            noisy_update_tensor[f:s] = (
                                noisy_update_tensor[f:s] / region_weight
                            )

                        # Invert gradients before updates
                        noisy_update_tensor = -noisy_update_tensor
                        booster.set_term_update(term_idx, noisy_update_tensor)

                    cur_metric = booster.apply_term_update()
                    # if early_stopping_tolerance is negative then keep accepting
                    # model updates as they get worse past the minimum. We might
                    # want to boost past the lowest because averaging the outer bags
                    # improves the model, so boosting past the minimum can yield
                    # a better overall model after averaging

                    modified_tolerance = (
                        min(abs(min_metric), abs(min_prev_metric))
                        * early_stopping_tolerance
                    )
                    if np.isinf(modified_tolerance):
                        modified_tolerance = 0.0

                    if cur_metric <= min_metric - min(0.0, modified_tolerance):
                        # TODO : change the C API to allow us to "commit" the current
                        # model into the best model instead of having the C layer
                        # decide that base on what it returned us
                        pass
                    min_metric = min(cur_metric, min_metric)

                    if early_stopping_rounds > 0 and smoothing_rounds <= 0:
                        # during smoothing, do not use early stopping because smoothing
                        # is using random cuts, which means gain is highly variable
                        toss = circular[circular_index]
                        circular[circular_index] = cur_metric
                        circular_index = (circular_index + 1) % len(circular)
                        min_prev_metric = min(toss, min_prev_metric)

                        if min_prev_metric - modified_tolerance <= circular.min():
                            circular_index = -1
                            break

                if circular_index < 0:
                    break

                if 1.0 <= greedy_portion:
                    greedy_portion -= 1.0

                if 0 < smoothing_rounds:
                    smoothing_rounds -= 1
                else:
                    # do not progress into greedy rounds until we're done with the smoothing_rounds
                    greedy_portion += greediness

            _log.info(
                "End boosting, Best Metric: {0}, Num Rounds: {1}".format(
                    min_metric, episode_index
                )
            )

            if early_stopping_rounds > 0:
                model_update = booster.get_best_model()
            else:
                model_update = booster.get_current_model()

        return None, model_update, episode_index, rng
    except Exception as e:
        return e, None, None, None
