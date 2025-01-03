# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import heapq
import logging

import numpy as np

from ... import develop
from ...utils._native import Booster, Native

_log = logging.getLogger(__name__)


def boost(
    dataset,
    intercept_rounds,
    intercept_learning_rate,
    intercept,
    bag,
    init_scores,
    term_features,
    n_inner_bags,
    term_boost_flags,
    learning_rate,
    min_samples_leaf,
    min_hessian,
    reg_alpha,
    reg_lambda,
    max_delta_step,
    min_cat_samples,
    cat_smooth,
    missing,
    max_leaves,
    monotone_constraints,
    greedy_ratio,
    cyclic_progress,
    smoothing_rounds,
    nominal_smoothing,
    max_rounds,
    early_stopping_rounds,
    early_stopping_tolerance,
    noise_scale,
    bin_weights,
    rng,
    create_booster_flags,
    objective,
    acceleration,
    experimental_params,
    develop_options,
):
    try:
        develop._develop_options = develop_options  # restore these in this process
        step_idx = 0

        _log.info("Start boosting")
        native = Native.get_native_singleton()

        with Booster(
            dataset,
            intercept,
            bag,
            init_scores,
            term_features,
            n_inner_bags,
            rng,
            create_booster_flags,
            objective,
            acceleration,
            experimental_params,
        ) as booster:
            for _ in range(intercept_rounds):
                booster.generate_term_update(
                    rng,
                    term_idx=-1,
                    term_boost_flags=term_boost_flags,
                    learning_rate=intercept_learning_rate,
                    min_samples_leaf=0,
                    min_hessian=0.0,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    max_delta_step=0.0,
                    min_cat_samples=min_cat_samples,
                    cat_smooth=cat_smooth,
                    max_cat_threshold=develop.get_option("max_cat_threshold"),
                    cat_include=develop.get_option("cat_include"),
                    max_leaves=1,
                    monotone_constraints=None,
                )
                intercept += booster.get_term_update()
                booster.apply_term_update()

            min_metric = np.inf
            min_prev_metric = np.inf
            circular = np.full(
                early_stopping_rounds * len(term_features), np.inf, np.float64
            )
            circular_idx = 0

            max_steps = max_rounds * len(term_features)

            # if greedy_ratio is set to +inf then set it to the max rounds.
            greedy_ratio = min(greedy_ratio, max_rounds)
            greedy_steps = int(np.ceil(greedy_ratio * len(term_features)))
            if greedy_steps <= 0:
                # if there are no greedy steps, then force progress on cyclic rounds
                cyclic_progress = 1.0
            cyclic_state = cyclic_progress

            state_idx = 0

            nominals = native.extract_nominals(dataset)
            random_cyclic_ordering = np.arange(len(term_features), dtype=np.int64)

            while step_idx < max_steps:
                if state_idx >= 0:
                    # cyclic
                    if state_idx == 0:
                        # starting a fresh cyclic round. Clear the priority queue
                        bestkey = None
                        heap = []
                        if (
                            step_idx == 0
                            and develop.get_option("randomize_initial_feature_order")
                            or develop.get_option("randomize_greedy_feature_order")
                            and greedy_steps > 0
                            or develop.get_option("randomize_feature_order")
                        ):
                            native.shuffle(rng, random_cyclic_ordering)

                    term_idx = random_cyclic_ordering[state_idx]

                    make_progress = False
                    if cyclic_state >= 1.0 or smoothing_rounds > 0:
                        # if cyclic_state is above 1.0 we make progress
                        step_idx += 1
                        make_progress = True
                else:
                    # greedy
                    make_progress = True
                    step_idx += 1
                    _, _, term_idx = heapq.heappop(heap)

                contains_nominals = any(nominals[i] for i in term_features[term_idx])

                term_boost_flags_local = term_boost_flags
                reg_lambda_local = reg_lambda
                min_samples_leaf_local = min_samples_leaf
                if contains_nominals:
                    reg_lambda_local += develop.get_option("cat_l2")

                    if develop.get_option("min_samples_leaf_nominal") is not None:
                        min_samples_leaf_local = develop.get_option(
                            "min_samples_leaf_nominal"
                        )

                if missing == "low":
                    term_boost_flags_local |= Native.TermBoostFlags_MissingLow
                elif missing == "high":
                    term_boost_flags_local |= Native.TermBoostFlags_MissingHigh
                elif missing == "separate":
                    term_boost_flags_local |= Native.TermBoostFlags_MissingSeparate
                elif missing != "gain":
                    msg = f"Unrecognized missing option {missing}."
                    raise Exception(msg)

                if smoothing_rounds > 0 and (
                    nominal_smoothing or not contains_nominals
                ):
                    # modify some of our parameters temporarily
                    term_boost_flags_local |= Native.TermBoostFlags_RandomSplits

                if bestkey is None or state_idx >= 0:
                    term_monotone = None
                    if monotone_constraints is not None:
                        term_monotone = np.array(
                            [monotone_constraints[i] for i in term_features[term_idx]],
                            dtype=np.int32,
                        )

                    avg_gain = booster.generate_term_update(
                        rng,
                        term_idx=term_idx,
                        term_boost_flags=term_boost_flags_local,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf_local,
                        min_hessian=min_hessian,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda_local,
                        max_delta_step=max_delta_step,
                        min_cat_samples=min_cat_samples,
                        cat_smooth=cat_smooth,
                        max_cat_threshold=develop.get_option("max_cat_threshold"),
                        cat_include=develop.get_option("cat_include"),
                        max_leaves=max_leaves,
                        monotone_constraints=term_monotone,
                    )

                    if contains_nominals and len(term_features[term_idx]) == 1:
                        # penalize nominals a bit because they benefit from sorting categories
                        avg_gain *= develop.get_option("cat_scale")

                    gainkey = (-avg_gain, native.generate_seed(rng), term_idx)
                    if not make_progress:
                        if bestkey is None or gainkey < bestkey:
                            bestkey = gainkey
                            cached_update = booster.get_term_update()
                else:
                    gainkey = bestkey
                    bestkey = None
                    assert term_idx == gainkey[2]  # heap and cached must agree
                    booster.set_term_update(term_idx, cached_update)

                heapq.heappush(heap, gainkey)

                if noise_scale:  # Differentially private updates
                    splits = booster.get_term_update_splits()[0]

                    term_update_tensor = booster.get_term_update()
                    noisy_update_tensor = term_update_tensor.copy()

                    # Make splits iteration friendly
                    splits_iter = [0, *list(splits), len(term_update_tensor)]

                    n_sections = len(splits_iter) - 1
                    noises = native.generate_gaussian_random(
                        rng, noise_scale, n_sections
                    )

                    # Loop through all random splits and add noise before updating
                    for f, s, noise in zip(splits_iter[:-1], splits_iter[1:], noises):
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

                if make_progress:
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
                    if np.isnan(modified_tolerance) or np.isinf(modified_tolerance):
                        modified_tolerance = 0.0

                    if cur_metric <= min_metric - min(0.0, modified_tolerance):
                        # TODO : change the C API to allow us to "commit" the current
                        # model into the best model instead of having the C layer
                        # decide that base on what it returned us

                        # TODO: step_idx gets turned reported publically as best_iteration
                        # although for now it is actually the number of boosting steps
                        # that were executed instead of how many were accepted until we
                        # stopped progressing once the best model was selected. For now
                        # we do not have that information, but once we move the decision
                        # point as to what the best model was from C++ to python we can
                        # change our reporting to be the best instead of the number of
                        # steps we took before stopping.
                        pass
                    min_metric = min(cur_metric, min_metric)

                    if len(circular) > 0 and smoothing_rounds <= 0:
                        # during smoothing, do not use early stopping because smoothing
                        # is using random cuts, which means gain is highly variable
                        toss = circular[circular_idx]
                        circular[circular_idx] = cur_metric
                        circular_idx = (circular_idx + 1) % len(circular)
                        min_prev_metric = min(toss, min_prev_metric)

                        if min_prev_metric - modified_tolerance <= circular.min():
                            break

                state_idx = state_idx + 1
                if len(term_features) <= state_idx:
                    if smoothing_rounds > 0:
                        state_idx = 0  # all smoothing rounds are cyclic rounds
                        smoothing_rounds -= 1
                    else:
                        state_idx = -greedy_steps
                        if cyclic_state >= 1.0:
                            cyclic_state -= 1.0
                        cyclic_state += cyclic_progress
            if len(circular) > 0:
                model_update = booster.get_best_model()
            else:
                model_update = booster.get_current_model()

        return None, intercept, model_update, step_idx, rng
    except Exception as e:
        return e, None, None, None, None
