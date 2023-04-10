# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

# TODO: Test EBMUtils

from math import ceil, floor, isnan, isinf, exp, log
from ...utils._native import Native, Booster
from ...utils._binning import _deduplicate_bins

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier
import numbers
import numpy as np
import warnings
from itertools import islice, count, chain

import heapq

from .postprocessing import multiclass_postprocess

import logging

_log = logging.getLogger(__name__)


def _zero_tensor(tensor, zero_low=None, zero_high=None):
    entire_tensor = [slice(None) for _ in range(tensor.ndim)]
    if zero_low is not None:
        for dimension_idx, is_zero in enumerate(zero_low):
            if is_zero:
                dim_slices = entire_tensor.copy()
                dim_slices[dimension_idx] = 0
                tensor[tuple(dim_slices)] = 0
    if zero_high is not None:
        for dimension_idx, is_zero in enumerate(zero_high):
            if is_zero:
                dim_slices = entire_tensor.copy()
                dim_slices[dimension_idx] = -1
                tensor[tuple(dim_slices)] = 0


def _restore_missing_value_zeros2(tensor, weights):
    n_dimensions = weights.ndim
    entire_tensor = [slice(None)] * n_dimensions
    lower = []
    higher = []
    for dimension_idx in range(n_dimensions):
        dim_slices = entire_tensor.copy()
        dim_slices[dimension_idx] = 0
        total_sum = np.sum(weights[tuple(dim_slices)])
        lower.append(True if total_sum == 0 else False)
        dim_slices[dimension_idx] = -1
        total_sum = np.sum(weights[tuple(dim_slices)])
        higher.append(True if total_sum == 0 else False)
    _zero_tensor(tensor, lower, higher)


def _weighted_std(a, axis, weights):
    average = np.average(a, axis, weights)
    variance = np.average((a - average) ** 2, axis, weights)
    return np.sqrt(variance)


def _convert_categorical_to_continuous(categories):
    # we do automagic detection of feature types by default, and sometimes a feature which
    # was really continuous might have most of it's data as one or two values.  An example would
    # be a feature that we have "0" and "1" in the training data, but "-0.1" and "3.1" are also
    # possible.  If during prediction we see a "3.1" we can magically convert our categories
    # into a continuous range with a cut point at 0.5.  Now "-0.1" goes into the [-inf, 0.5) bin
    # and 3.1 goes into the [0.5, +inf] bin.
    #
    # We can't convert a continuous feature that has cuts back into categoricals
    # since the categorical value could have been anything between the cuts that we know about.

    clusters = dict()
    non_float_idxs = set()

    old_min = np.nan
    old_max = np.nan
    for category, idx in categories.items():
        try:
            # this strips leading and trailing spaces
            val = float(category)
        except ValueError:
            non_float_idxs.add(idx)
            continue

        if isnan(val) or isinf(val):
            continue

        if isnan(old_min) or val < old_min:
            old_min = val
        if isnan(old_max) or old_max < val:
            old_max = val

        cluster_list = clusters.get(idx)
        if cluster_list is None:
            clusters[idx] = [val]
        else:
            cluster_list.append(val)

    # there's a super fringe case where two category strings map to the same bin, but
    # one of them is a float and the other is a non-float.  Normally, we'd include the
    # non-float categorical in the unknowns, but in this case we'd need to include
    # a part of a bin.  Handling this just adds too much complexity for the benefit
    # and you could argue that the evidence from the other models is indicating that
    # the string should be closer to zero of the weight from the floating point bin
    # so we take the simple route of putting all the weight into the float and none on the
    # non-float.  We still need to remove any indexes though that map to both a float
    # and a non-float, so this line handles that
    non_float_idxs = [idx for idx in non_float_idxs if idx not in clusters]
    non_float_idxs.append(max(categories.values()) + 1)

    if len(clusters) <= 1:
        return np.empty(0, np.float64)

    cluster_bounds = []
    for cluster_list in clusters.values():
        cluster_list.sort()
        cluster_bounds.append((cluster_list[0], cluster_list[-1]))

    # TODO: move everything below here into C++ to ensure cross language compatibility

    cluster_bounds.sort()

    cuts = []
    cluster_iter = iter(cluster_bounds)
    low = next(cluster_iter)[-1]
    for cluster in cluster_iter:
        high = cluster[0]
        if low < high:
            # if they are equal or if low is higher then we can't separate one cluster
            # from another, so we keep joining them until we can get clean separations

            half_diff = (high - low) / 2
            if isinf(half_diff):
                # first try to subtract then divide since that's more accurate but some float64
                # values will fail eg (max_float - min_float == +inf) so we need to try
                # a less accurate way of dividing first if we detect this.  Dividing
                # first will always succeed, even with the most extreme possible values of
                # max_float / 2 - min_float / 2
                half_diff = high / 2 - low / 2

            # floats have more precision the smaller they are,
            # so use the smaller number as the anchor
            if abs(low) <= abs(high):
                mid = low + half_diff
            else:
                mid = high - half_diff

            if mid <= low:
                # this can happen with very small half_diffs that underflow the add/subtract operation
                # if this happens the numbers must be very close together on the order of a float tick.
                # We use lower bound inclusive for our cut discretization, so make the mid == high
                mid = high

            cuts.append(mid)
        low = max(low, cluster[-1])
    cuts = np.array(cuts, np.float64)

    mapping = [[] for _ in range(len(cuts) + 3)]
    for old_idx, cluster_list in clusters.items():
        # all the items in a cluster should be binned into the same bins
        new_idx = np.searchsorted(cuts, cluster_list[:1], side="right")[0] + 1
        mapping[new_idx].append(old_idx)

    mapping[0].append(0)
    mapping[-1] = non_float_idxs

    return cuts, mapping, old_min, old_max


def _create_proportional_tensor(axis_weights):
    # take the per-feature weights and distribute them proportionally to each cell in a tensor

    axis_sums = [weights.sum() for weights in axis_weights]

    # Normally you'd expect each axis to sum to the total weight from the model,
    # so normally they should be identical.  We encourage model editing though, so they may
    # not be identical under some edits.  Also, if the model is a DP model then the weights are
    # probably different due to the noise contribution.  Let's take the geometic mean to compensate.
    total_weight = exp(sum(log(axis_sum) for axis_sum in axis_sums) / len(axis_sums))
    axis_percentages = [
        weights / axis_sum for weights, axis_sum in zip(axis_weights, axis_sums)
    ]

    shape = tuple(map(len, axis_percentages))
    n_cells = np.prod(shape)
    tensor = np.empty(n_cells, np.float64)

    # the last index items are next together in flat memory layout
    axis_percentages.reverse()

    for cell_idx in range(n_cells):
        remainder = cell_idx
        frac = 1.0
        for percentages in axis_percentages:
            bin_idx = remainder % len(percentages)
            remainder //= len(percentages)
            frac *= percentages[bin_idx]
        val = frac * total_weight
        tensor.itemset(cell_idx, val)
    return tensor.reshape(shape)


def _process_terms(n_classes, bagged_scores, bin_weights, bag_weights):
    term_scores = []
    standard_deviations = []
    new_bagged_scores = []
    for score_tensors, weights in zip(bagged_scores, bin_weights):
        # if the missing/unknown bin has zero weight then whatever number was generated via boosting is
        # effectively meaningless and can be ignored. Set the value to zero for interpretability reasons
        tensor_bags = []
        for tensor in score_tensors:
            tensor_copy = tensor.copy()
            if n_classes != 1:
                _restore_missing_value_zeros2(tensor_copy, weights)
            tensor_bags.append(tensor_copy)
        score_tensors = np.array(
            tensor_bags, np.float64
        )  # replace it to get stddev of 0 for weight of 0
        new_bagged_scores.append(score_tensors)

        # TODO PK: shouldn't we be zero centering each score tensor first before taking the standard deviation
        # It's possible to shift scores arbitary to the intercept, so we should be able to get any desired stddev

        # TODO PK: Generally if a bin for missing/unknown has zero weight then it means that the score should be
        # ignored since its value was set by boosting to be something useless based on adjacent bins. The normal
        # way we deal with this is to set a missing/unknown bin that has zero weight to zero AFTER centering.
        # This works well during fit, but there is an issue later if someone merges two models
        # where one model had no missing/unknown data but the other model being merged did. Since we only use and
        # keep a combined bin_weights field that contains the average of all bagged models, we no longer know that the
        # scores in some of the original bagged models should have been ignored/zeroed. When a model with no
        # missing/unknown data during fitting encounters missing/unknown data during prediction we chose a neutral
        # answer by returning a value (zero) that is the average accross the training data for a centered graph.
        # We wish to maintain that, but we do not want to have a separate attribute for the intercept of each
        # bagged model. We can handle this by using the following procedure:
        #   1) Center the models within each bag and generate a separate intercept for each bag
        #   2) Zero the missing/unknown bins if the weights are zero
        #   3) re-distribute the intercept back into the bagged models equitably. The missing/unknown bins
        #      will now be non-zero, but if during merging we loose the information that they should have been
        #      zeroed our missing/unknown bins will already be set to the value that will yield a neutral response

        if (bag_weights == bag_weights[0]).all():
            # if all the bags have the same total weight we can avoid some numeracy issues
            # by using a non-weighted standard deviation
            feature_term_scores = np.average(score_tensors, axis=0)
            term_scores.append(feature_term_scores)
            if n_classes == 1:
                standard_deviations.append(
                    np.zeros(feature_term_scores.shape, np.float64)
                )
            else:
                standard_deviations.append(np.std(score_tensors, axis=0))
        else:
            feature_term_scores = np.average(score_tensors, axis=0, weights=bag_weights)
            term_scores.append(feature_term_scores)
            if n_classes == 1:
                standard_deviations.append(
                    np.zeros(feature_term_scores.shape, dtype=np.float64)
                )
            else:
                standard_deviations.append(
                    _weighted_std(score_tensors, axis=0, weights=bag_weights)
                )

    if n_classes == 1:
        intercept = np.full(1, -np.inf, np.float64)
    else:
        intercept = np.zeros(Native.get_count_scores_c(n_classes), np.float64)

        if n_classes <= 2:
            for scores, weights in zip(term_scores, bin_weights):
                score_mean = np.average(scores, weights=weights)
                scores -= score_mean

                # Add mean center adjustment back to intercept
                intercept += score_mean
        else:
            # Postprocess model graphs for multiclass
            multiclass_postprocess(n_classes, term_scores, bin_weights, intercept)

        for scores, weights in zip(term_scores, bin_weights):
            # set these to zero again since zero-centering them causes the missing/unknown to shift away from zero
            _restore_missing_value_zeros2(scores, weights)

        if n_classes < 0:
            # scikit-learn uses a float for regression, and a numpy array with 1 element for binary classification
            intercept = float(intercept)

    return term_scores, standard_deviations, intercept, new_bagged_scores


def _generate_term_names(feature_names, term_features):
    return [" & ".join(feature_names[i] for i in grp) for grp in term_features]


def _generate_term_types(feature_types, term_features):
    return [
        feature_types[grp[0]] if len(grp) == 1 else "interaction"
        for grp in term_features
    ]


def _order_terms(term_features, *args):
    keys = (
        [len(feature_idxs)] + sorted(feature_idxs) for feature_idxs in term_features
    )
    sorted_items = sorted(zip(keys, term_features, *args))
    ret = tuple(list(x) for x in islice(zip(*sorted_items), 1, None))
    # in Python if only 1 item exists then the item is returned and not a tuple
    return ret if 2 <= len(ret) else ret[0]


def _remove_unused_higher_bins(term_features, bins):
    # many features are not used in pairs, so we can simplify the model
    # by removing the extra higher interaction level bins

    highest_levels = [0] * len(bins)
    for feature_idxs in term_features:
        for feature_idx in feature_idxs:
            highest_levels[feature_idx] = max(
                highest_levels[feature_idx], len(feature_idxs)
            )

    for bin_levels, max_level in zip(bins, highest_levels):
        del bin_levels[max_level:]


# TODO: Clean up
class EBMUtils:
    # NOTE: Interval / cut conversions are future work. Not registered for code coverage.
    @staticmethod
    def convert_to_intervals(cuts):  # pragma: no cover
        cuts = np.array(cuts, dtype=np.float64)

        if np.isnan(cuts).any():
            raise Exception("cuts cannot contain nan")

        if np.isinf(cuts).any():
            raise Exception("cuts cannot contain infinity")

        smaller = np.insert(cuts, 0, -np.inf)
        larger = np.append(cuts, np.inf)
        intervals = list(zip(smaller, larger))

        if any(x[1] <= x[0] for x in intervals):
            raise Exception("cuts must contain increasing values")

        return intervals

    @staticmethod
    def convert_to_cuts(intervals):  # pragma: no cover
        if len(intervals) == 0:
            raise Exception("intervals must have at least one interval")

        if any(len(x) != 2 for x in intervals):
            raise Exception("intervals must be a list of tuples")

        if intervals[0][0] != -np.inf:
            raise Exception("intervals must start from -inf")

        if intervals[-1][-1] != np.inf:
            raise Exception("intervals must end with inf")

        cuts = [x[0] for x in intervals[1:]]
        cuts_verify = [x[1] for x in intervals[:-1]]

        if np.isnan(cuts).any():
            raise Exception("intervals cannot contain NaN")

        if any(x[0] != x[1] for x in zip(cuts, cuts_verify)):
            raise Exception("intervals must contain adjacent sections")

        if any(higher <= lower for lower, higher in zip(cuts, cuts[1:])):
            raise Exception("intervals must contain increasing sections")

        return cuts

    @staticmethod
    def make_bag(y, test_size, rng, is_stratified):
        # all test/train splits should be done with this function to ensure that
        # if we re-generate the train/test splits that they are generated exactly
        # the same as before

        if test_size == 0:
            return None
        elif test_size > 0:
            n_samples = len(y)
            n_test_samples = 0

            if test_size >= 1:
                if test_size % 1:
                    raise Exception(
                        "If test_size >= 1, test_size should be a whole number."
                    )
                n_test_samples = test_size
            else:
                n_test_samples = ceil(n_samples * test_size)

            n_train_samples = n_samples - n_test_samples
            native = Native.get_native_singleton()

            # Adapt test size if too small relative to number of classes
            if is_stratified:
                n_classes = len(set(y))
                if n_test_samples < n_classes:  # pragma: no cover
                    warnings.warn(
                        "Too few samples per class, adapting test size to guarantee 1 sample per class."
                    )
                    n_test_samples = n_classes
                    n_train_samples = n_samples - n_test_samples

                return native.sample_without_replacement_stratified(
                    rng, n_classes, n_train_samples, n_test_samples, y
                )
            else:
                return native.sample_without_replacement(
                    rng, n_train_samples, n_test_samples
                )
        else:  # pragma: no cover
            raise Exception("test_size must be a positive numeric value.")

    @staticmethod
    def jsonify_lists(vals):
        if len(vals) != 0:
            if type(vals[0]) is float:
                for idx, val in enumerate(vals):
                    # JSON doesn't have NaN, or infinities, but javaScript has these, so use javaScript strings
                    if isnan(val):
                        vals[idx] = "NaN"  # this is what JavaScript outputs for 0/0
                    elif val == np.inf:
                        vals[
                            idx
                        ] = "Infinity"  # this is what JavaScript outputs for 1/0
                    elif val == -np.inf:
                        vals[
                            idx
                        ] = "-Infinity"  # this is what JavaScript outputs for -1/0
            else:
                for nested in vals:
                    EBMUtils.jsonify_lists(nested)
        return vals  # we modify in place, but return it just for easy access

    @staticmethod
    def jsonify_item(val):
        # JSON doesn't have NaN, or infinities, but javaScript has these, so use javaScript strings
        if isnan(val):
            val = "NaN"  # this is what JavaScript outputs for 0/0
        elif val == np.inf:
            val = "Infinity"  # this is what JavaScript outputs for 1/0
        elif val == -np.inf:
            val = "-Infinity"  # this is what JavaScript outputs for -1/0
        return val

    @staticmethod
    def cyclic_gradient_boost(
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
                        for f, s, noise in zip(
                            splits_iter[:-1], splits_iter[1:], noises
                        ):
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
