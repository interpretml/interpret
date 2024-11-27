# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
import warnings
from collections import defaultdict
from itertools import islice
from math import ceil, exp, isfinite, isinf, log

import numpy as np

from ... import develop
from ...utils._native import Native
from ...utils._purify import purify
from ._tensor import restore_missing_value_zeros

_log = logging.getLogger(__name__)


def _midpoint(low: float, high: float) -> float:
    """Return midpoint between `low` and `high` with high numerical accuracy."""
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
    mid = low + half_diff if abs(low) <= abs(high) else high - half_diff

    if mid <= low:
        # this can happen with very small half_diffs that underflow the add/subtract operation
        # if this happens the numbers must be very close together on the order of a float tick.
        # We use lower bound inclusive for our cut discretization, so make the mid == high
        mid = high
    return mid


def convert_categorical_to_continuous(categories):
    # we do automagic detection of feature types by default, and sometimes a feature which
    # was really continuous might have most of it's data as one or two values.  An example would
    # be a feature that we have "0" and "1" in the training data, but "-0.1" and "3.1" are also
    # possible.  If during prediction we see a "3.1" we can magically convert our categories
    # into a continuous range with a cut point at 0.5.  Now "-0.1" goes into the [-inf, 0.5) bin
    # and 3.1 goes into the [0.5, +inf] bin.
    #
    # We can't convert a continuous feature that has cuts back into categoricals
    # since the categorical value could have been anything between the cuts that we know about.

    clusters = defaultdict(list)
    non_float_idxs = set()

    old_min = +np.inf
    old_max = -np.inf
    for category, idx in categories.items():
        try:
            # this strips leading and trailing spaces
            val = float(category)
        except ValueError:
            non_float_idxs.add(idx)
            continue

        if not isfinite(val):
            continue

        old_min = min(old_min, val)
        old_max = max(old_max, val)

        clusters[idx].append(val)

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

    if len(clusters) == 0:
        return np.empty(0, np.float64), [[0], [], non_float_idxs], np.nan, np.nan

    cluster_bounds = sorted(
        (min(cluster_list), max(cluster_list)) for cluster_list in clusters.values()
    )

    # TODO: move everything below here into C++ to ensure cross language compatibility
    cuts = []
    _, low = cluster_bounds[0]
    for high, next_low in cluster_bounds[1:]:
        if low < high:
            # if they are equal or if low is higher then we can't separate one cluster
            # from another, so we keep joining them until we can get clean separations
            cuts.append(_midpoint(low, high))
        low = max(low, next_low)
    cuts = np.array(cuts, np.float64)

    mapping = [[] for _ in range(len(cuts) + 3)]
    for old_idx, cluster_list in clusters.items():
        # all the items in a cluster should be binned into the same bins
        new_idx = np.searchsorted(cuts, [min(cluster_list)], side="right")[0] + 1
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
        tensor[cell_idx] = val
    return tensor.reshape(shape)


def process_bag_terms(intercept, term_scores, bin_weights):
    for scores, weights in zip(term_scores, bin_weights):
        if develop.get_option("purify_result"):
            new_scores, add_impurities, add_intercept = purify(scores, weights)
            # TODO: benchmark if it is better to add new_impurities to the existing model scores,
            #       or better to discard them.  Discarding might be better if we assume the
            #       non-overfit benefit of the lower dimensional interactions has already been extracted.
            scores[:] = new_scores
            intercept += add_intercept
        elif scores.ndim == weights.ndim:
            temp_scores = scores.flatten()  # ndarray.flatten() makes a copy
            temp_weights = weights.flatten()  # ndarray.flatten() makes a copy

            ignored = ~np.isfinite(temp_scores)
            temp_scores[ignored] = 0.0
            temp_weights[ignored] = 0.0

            if temp_weights.sum() != 0:
                mean = np.average(temp_scores, 0, temp_weights)
                intercept += mean
                scores -= mean
        else:
            for i in range(scores.shape[-1]):
                temp_scores = scores[..., i].flatten()  # ndarray.flatten() makes a copy
                temp_weights = weights.flatten()  # ndarray.flatten() makes a copy

                ignored = ~np.isfinite(temp_scores)
                temp_scores[ignored] = 0.0
                temp_weights[ignored] = 0.0

                if temp_weights.sum() != 0:
                    mean = np.average(temp_scores, 0, temp_weights)
                    intercept[i] += mean
                    scores[..., i] -= mean

        # We could apply the algorithm proposed by Xuezhou Zhang here, however that algorithm doesn't work
        # for nominal categoricals since there is no concept of adjacency, so for nominal categoricals we
        # need some way to make the multiclass scores identifiable. Making the scores sum to zero, or alternatively
        # choosing to zero the class that has the highest intercept class score would work. Making the scores
        # sum to zero is less arbitrary than Xuezhou's algorithm when it comes to calculating feature/term
        # importance values, so if we use Xuezhou's algorithm then apply it when generating an explanation
        # instead of here which will make calculating importances faster.

        # if the missing/unknown bin has zero weight then whatever number was generated via boosting is
        # effectively meaningless and can be ignored. Set the value to zero for interpretability reasons

        restore_missing_value_zeros(scores, weights)


def process_terms(bagged_intercept, bagged_scores, bin_weights, bag_weights):
    native = Native.get_native_singleton()

    n_bags = len(bag_weights)
    n_terms = len(bin_weights)
    for bag_idx in range(n_bags):
        term_scores = [bagged_tensor[bag_idx] for bagged_tensor in bagged_scores]
        intercept = np.atleast_1d(bagged_intercept[bag_idx])
        process_bag_terms(intercept, term_scores, bin_weights)
        bagged_intercept[bag_idx] = intercept[0] if len(intercept) == 1 else intercept
        for term_idx in range(n_terms):
            bagged_scores[term_idx][bag_idx] = term_scores[term_idx]

    term_scores = []
    standard_deviations = []
    for scores in bagged_scores:
        averaged = native.safe_mean(scores, bag_weights)
        term_scores.append(averaged)
        stddevs = native.safe_stddev(scores, bag_weights)
        standard_deviations.append(stddevs)

    intercept = native.safe_mean(bagged_intercept, bag_weights)

    if bagged_intercept.ndim == 2:
        # multiclass
        # pick the class that we're going to zero
        zero_index = np.argmax(intercept)
        intercept -= intercept[zero_index]
        bagged_intercept -= np.expand_dims(bagged_intercept[..., zero_index], -1)

    return intercept, term_scores, standard_deviations


def generate_term_names(feature_names, term_features):
    return [" & ".join(feature_names[i] for i in grp) for grp in term_features]


def generate_term_types(feature_types, term_features):
    return [
        feature_types[grp[0]] if len(grp) == 1 else "interaction"
        for grp in term_features
    ]


def order_terms(term_features, *args):
    if len(term_features) == 0:
        # in Python if only 1 item exists then the item is returned and not a tuple
        if len(args) == 0:
            return []
        return tuple([] for _ in range(len(args) + 1))
    keys = (
        [len(feature_idxs), *sorted(feature_idxs)] for feature_idxs in term_features
    )
    sorted_items = sorted(zip(keys, term_features, *args))
    ret = tuple(list(x) for x in islice(zip(*sorted_items), 1, None))
    # in Python if only 1 item exists then the item is returned and not a tuple
    return ret if len(ret) >= 2 else ret[0]


def remove_unused_higher_bins(term_features, bins):
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


def deduplicate_bins(bins):
    # calling this function before calling score_terms allows score_terms to operate more efficiently since it'll
    # be able to avoid re-binning data for pairs that have already been processed in mains or other pairs since we
    # use the id of the bins to identify feature data that was previously binned

    uniques = {}
    for bin_levels in bins:
        highest_key = None
        highest_idx = -1
        for level_idx, feature_bins in enumerate(bin_levels):
            if isinstance(feature_bins, dict):
                key = frozenset(feature_bins.items())
            else:
                key = tuple(feature_bins)
            if key in uniques:
                bin_levels[level_idx] = uniques[key]
            else:
                uniques[key] = feature_bins

            if highest_key != key:
                highest_key = key
                highest_idx = level_idx
        del bin_levels[highest_idx + 1 :]


def convert_to_intervals(cuts):  # pragma: no cover
    cuts = np.array(cuts, dtype=np.float64)
    if cuts.size == 0:
        return [(-np.inf, np.inf)]

    if not np.isfinite(cuts).all():
        msg = "cuts must contain only finite numbers"
        raise Exception(msg)

    intervals = [(-np.inf, cuts[0]), *zip(cuts[:-1], cuts[1:]), (cuts[-1], np.inf)]

    if any(higher <= lower for (lower, higher) in intervals):
        msg = "cuts must contain increasing values"
        raise Exception(msg)

    return intervals


def convert_to_cuts(intervals):  # pragma: no cover
    if len(intervals) == 0:
        msg = "intervals must have at least one interval"
        raise Exception(msg)

    if any(len(x) != 2 for x in intervals):
        msg = "intervals must be a list of tuples"
        raise Exception(msg)

    if intervals[0][0] != -np.inf:
        msg = "intervals must start from -inf"
        raise Exception(msg)

    if intervals[-1][-1] != np.inf:
        msg = "intervals must end with inf"
        raise Exception(msg)

    cuts = [lower for (lower, _) in intervals[1:]]
    cuts_verify = [higher for (_, higher) in intervals[:-1]]

    if np.isnan(cuts).any():
        msg = "intervals cannot contain NaN"
        raise Exception(msg)

    if any(x[0] != x[1] for x in zip(cuts, cuts_verify)):
        msg = "intervals must contain adjacent sections"
        raise Exception(msg)

    if any(higher <= lower for lower, higher in zip(cuts, cuts[1:])):
        msg = "intervals must contain increasing sections"
        raise Exception(msg)

    return cuts


def make_bag(y, test_size, rng, is_stratified):
    # all test/train splits should be done with this function to ensure that
    # if we re-generate the train/test splits that they are generated exactly
    # the same as before

    if test_size < 0:  # pragma: no cover
        msg = "test_size must be a positive numeric value."
        raise Exception(msg)
    if test_size == 0:
        return None
    n_samples = len(y)
    n_test_samples = 0

    if test_size >= 1:
        if test_size % 1:
            msg = "If test_size >= 1, test_size should be a whole number."
            raise Exception(msg)
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
    return native.sample_without_replacement(rng, n_train_samples, n_test_samples)
