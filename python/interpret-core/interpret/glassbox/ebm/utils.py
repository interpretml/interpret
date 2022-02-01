# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# TODO: Test EBMUtils

from math import ceil, isnan
from .internal import Native, Booster, InteractionDetector

# from scipy.special import expit
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier
import numbers
import numpy as np
import warnings
import copy
import operator

from scipy.stats import norm
from scipy.optimize import root_scalar, brentq

from .postprocessing import multiclass_postprocess2

from itertools import count, chain

import logging

log = logging.getLogger(__name__)

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

def _restore_missing_value_zeros2(tensors, term_bin_weights):
    for tensor, weights in zip(tensors, term_bin_weights):
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
    average = np.average(a, axis , weights)
    variance = np.average((a - average)**2, axis , weights)
    return np.sqrt(variance)

def _process_terms(n_classes, n_samples, bagged_additive_terms, bin_weights, bag_weights=None):
    additive_terms = []
    term_standard_deviations = []
    for score_tensors in bagged_additive_terms:
        # TODO PK: shouldn't we be zero centering each score tensor first before taking the standard deviation
        # It's possible to shift scores arbitary to the intercept, so we should be able to get any desired stddev

        if bag_weights is None:
            additive_terms.append(np.average(score_tensors, axis=0))
            term_standard_deviations.append(np.std(score_tensors, axis=0))
        else:
            additive_terms.append(np.average(score_tensors, axis=0, weights=bag_weights))
            term_standard_deviations.append(_weighted_std(score_tensors, axis=0, weights=bag_weights))

    intercept = np.zeros(Native.get_count_scores_c(n_classes), np.float64)

    if n_classes <= 2:
        for idx in range(len(bagged_additive_terms)):
            score_mean = np.average(additive_terms[idx], weights=bin_weights[idx])
            additive_terms[idx] = (additive_terms[idx] - score_mean)

            # Add mean center adjustment back to intercept
            intercept += score_mean
    else:
        # Postprocess model graphs for multiclass
        multiclass_postprocess2(n_classes, n_samples, additive_terms, intercept, bin_weights)

    _restore_missing_value_zeros2(additive_terms, bin_weights)
    _restore_missing_value_zeros2(term_standard_deviations, bin_weights)

    if n_classes < 0:
        # scikit-learn uses a float for regression, and a numpy array with 1 element for binary classification
        intercept = float(intercept)

    return additive_terms, term_standard_deviations, intercept


def _deduplicate_bins(bins):
    # calling this function before calling score_terms allows score_terms to operate more efficiently since it'll
    # be able to avoid re-binning data for pairs that have already been processed in mains or other pairs since we 
    # use the id of the bins to identify feature data that was previously binned

    uniques = dict()
    for feature_idx in range(len(bins)):
        bin_levels = bins[feature_idx]
        highest_key = None
        for level_idx, feature_bins in enumerate(bin_levels):
            if isinstance(feature_bins, dict):
                key = frozenset(feature_bins.items())
            else:
                key = tuple(feature_bins)
            existing = uniques.get(key, None)
            if existing is None:
                uniques[key] = feature_bins
            else:
                bin_levels[level_idx] = existing

            if highest_key != key:
                highest_key = key
                highest_idx = level_idx
        del bin_levels[highest_idx + 1:]

def _harmonize_tensor(
    new_feature_idxs, 
    new_bins, 
    new_bounds, 
    old_feature_idxs, 
    old_bins, 
    old_bounds, 
    old_tensor, 
    is_proportional
):
    old_feature_idxs = list(old_feature_idxs)

    axes = []
    for feature_idx in new_feature_idxs:
        old_idx = old_feature_idxs.index(feature_idx)
        old_feature_idxs[old_idx] = -1 # in case we have duplicate feature idxs
        axes.append(old_idx)


    if len(axes) != old_tensor.ndim:
        # multiclass. The last dimension always stays put
        axes.append(len(axes))

    old_tensor = old_tensor.transpose(tuple(axes))

    lookups = []
    percentages = []
    for feature_idx in new_feature_idxs:
        old_bin_levels = old_bins[feature_idx]
        old_feature_bins = old_bin_levels[min(len(old_bin_levels), len(old_feature_idxs)) - 1]

        new_bin_levels = new_bins[feature_idx]
        new_feature_bins = new_bin_levels[min(len(new_bin_levels), len(new_feature_idxs)) - 1]


        if isinstance(old_feature_bins, dict):
            # categorical feature

            old_reversed = dict()
            for category, bin_idx in old_feature_bins.items():
                category_list = old_reversed.get(bin_idx)
                if category_list is None:
                    old_reversed[bin_idx] = [category]
                else:
                    category_list.append(category)

            new_reversed = dict()
            for category, bin_idx in new_feature_bins.items():
                category_list = new_reversed.get(bin_idx)
                if category_list is None:
                    new_reversed[bin_idx] = [category]
                else:
                    category_list.append(category)
            new_reversed = sorted(new_reversed.items())

            lookup = [0]
            percentage = [1.0]
            for _, new_categories in new_reversed:
                # if there are two items in new_categories then they should both resolve
                # to the same index in old_feature_bins otherwise they would have been
                # split into two categories
                old_bin_idx = old_feature_bins.get(new_categories[0], -1)
                if 0 <= old_bin_idx:
                    percentage.append(len(new_categories) / len(old_reversed[old_bin_idx]))
                else:
                    percentage.append(np.nan)
                    if not is_proportional:
                        old_bin_idx = len(old_reversed) + 1 # use the unknown value for scores
                lookup.append(old_bin_idx)
            percentage.append(1.0)
            lookup.append(len(old_reversed) + 1)
        else:
            # continuous feature

            lookup = list(np.searchsorted(old_feature_bins, new_feature_bins) + 1)
            lookup.append(len(old_feature_bins) + 1)

            percentage = [1.0]
            for new_idx_minus_one, old_idx in enumerate(lookup):
                if new_idx_minus_one == 0:
                    new_low = new_bounds[feature_idx, 0]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.  If -inf or +inf, change it to min/max for float
                else:
                    new_low = new_feature_bins[new_idx_minus_one - 1]

                if len(new_feature_bins) <= new_idx_minus_one:
                    new_high = new_bounds[feature_idx, 1]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.  If -inf or +inf, change it to min/max for float
                else:
                    new_high = new_feature_bins[new_idx_minus_one]


                if old_idx == 1:
                    old_low = old_bounds[feature_idx, 0]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.  If -inf or +inf, change it to min/max for float
                else:
                    old_low = old_feature_bins[old_idx - 2]

                if len(old_feature_bins) < old_idx:
                    old_high = old_bounds[feature_idx, 1]
                    # TODO: if nan OR out of bounds from the cuts, estimate it.  If -inf or +inf, change it to min/max for float
                else:
                    old_high = old_feature_bins[old_idx - 1]

                if old_high <= new_low or new_high <= old_low:
                    # if there are bins in the area above where the old data extended, then 
                    # we'll have zero contribution in the old data where these new bins are
                    # located
                    percentage.append(0)
                else:
                    if new_low < old_low:
                        # this can't happen except at the lowest bin where the new min can be
                        # lower than the old min.  In that case we know the old data
                        # had zero contribution between the new min to the old min.
                        new_low = old_low

                    if old_high < new_high:
                        # this can't happen except at the lowest bin where the new max can be
                        # higher than the old max.  In that case we know the old data
                        # had zero contribution between the new max to the old max.
                        new_high = old_high

                    percentage.append((new_high - new_low) / (old_high - old_low))

            percentage.append(1.0)
            lookup.insert(0, 0)
            lookup.append(len(old_feature_bins) + 2)

        lookups.append(lookup)
        percentages.append(percentage)

    new_shape = tuple(len(lookup) for lookup in lookups)
    n_cells = np.prod(new_shape)

    lookups.reverse()
    percentages.reverse()

    # now we need to inflate it
    new_tensor = np.empty(n_cells, np.float64)
    for cell_idx in range(n_cells):
        remainder = cell_idx
        old_reversed_bin_idxs = []
        frac = 1
        for lookup, percentage in zip(lookups, percentages):
            n_bins = len(lookup)
            new_bin_idx = remainder % n_bins
            remainder //= n_bins
            old_reversed_bin_idxs.append(lookup[new_bin_idx])
            frac *= percentage[new_bin_idx]

        if any(bin_idx < 0 for bin_idx in old_reversed_bin_idxs):
            val = 0 # categorical that exists in the new tensor but not the old one
        else:
            val = old_tensor[tuple(reversed(old_reversed_bin_idxs))]
            if is_proportional:
                val *= frac
        new_tensor.itemset(cell_idx, val)
    new_tensor = new_tensor.reshape(new_shape)
    return new_tensor

def merge_ebms(models):
    """ Merging multiple EBM models trained on the same dataset.
    Args:
        models: List of EBM models to be merged.
    Returns:
        An EBM model with averaged mean and standard deviation of input models.
    """


    # TODO: We're not currently moving over any of the __init__ parameters, but we should do that

    if len(models) < 2:  # pragma: no cover
        raise Exception("At least two models are required to merge.")

    if len(set(type(model) for model in models)) != 1:  # pragma: no cover
        # TODO: we might be able to relax this and merge for instance a DP classification model with a non-DP one
        raise Exception("All models should be the same type.")

    ebm = copy.deepcopy(models[0])

    type_name = type(ebm).__name__
    if type_name not in {'ExplainableBoostingClassifier', 'DPExplainableBoostingClassifier', 'ExplainableBoostingRegressor', 'DPExplainableBoostingRegressor'}:
        raise Exception(f"Unknown EBM model type {type_name}.")

    is_classifier = type_name == 'ExplainableBoostingClassifier' or type_name == 'DPExplainableBoostingClassifier'
    is_private = type_name == 'DPExplainableBoostingClassifier' or type_name == 'DPExplainableBoostingRegressor'

    if any(not getattr(model, 'has_fitted_', False) for model in models):  # pragma: no cover
        raise Exception("All models must be fitted.")

    if any(ebm.n_features_in_ != model.n_features_in_ for model in models):  # pragma: no cover
        raise Exception("All models should have the same number of features.")

    if any(ebm.feature_names_in_ != model.feature_names_in_ for model in models):  # pragma: no cover
        raise Exception("All models should have the same feature names.")

    if any(ebm.feature_types_in_ != model.feature_types_in_ for model in models):  # pragma: no cover
        raise Exception("All models should have the same feature types.")

    if ebm.n_features_in_ != len(ebm.feature_names_in_ ):  # pragma: no cover
        raise Exception("Bad model format.  Number of features doesn't match feature names.")

    if ebm.n_features_in_ != len(ebm.feature_types_in_ ):  # pragma: no cover
        raise Exception("Bad model format.  Number of features doesn't match feature types.")

    if is_private:
        if hasattr(ebm, 'noise_scale_'):
            del ebm.noise_scale_ # TODO ask Harsha if we can/should estimate this
    else:
        ebm.n_samples_ = sum(model.n_samples_ for model in models)

        # TODO: we could probably use the overall min/max to construct new bins (using the max number of histogram bins), 
        # and then approximate the counts by proportioning them
        if hasattr(ebm, 'histogram_cuts_'):
            del ebm.histogram_cuts_
        if hasattr(ebm, 'histogram_counts_'):
            del ebm.histogram_counts_

        if hasattr(ebm, 'unique_counts_'):
            del ebm.unique_counts_

        ebm.zero_counts_ = sum(model.zero_counts_ for model in models)

        # TODO: we could probably estimate these like we do with weights but with the added step
        # of afterwards re-integerizing the counts by putting residuals into the most likely bins
        if hasattr(ebm, 'bin_counts_'):
            del ebm.bin_counts_

    if is_classifier:
        if any(not np.array_equal(ebm.classes_, model.classes_) for model in models):  # pragma: no cover
            raise Exception("The target classes should be identical.")

        ebm._class_idx_ = {x: index for index, x in enumerate(ebm.classes_)}
        n_classes = len(ebm.classes_)
    else:
        ebm.min_target_ = min(model.min_target_ for model in models)
        ebm.max_target_ = max(model.max_target_ for model in models)
        n_classes = -1

    if any(ebm.n_features_in_ != model.feature_bounds_.shape[0] for model in models if getattr(model, 'feature_bounds_', None) is not None):  # pragma: no cover
        raise Exception("All feature_bounds_ should have the same length as the number of features.")

    min_vals = [model.feature_bounds_[:, 0] for model in models if getattr(model, 'feature_bounds_', None) is not None]
    max_vals = [model.feature_bounds_[:, 1] for model in models if getattr(model, 'feature_bounds_', None) is not None]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if hasattr(ebm, 'feature_bounds_'):
            del ebm.feature_bounds_
        if 0 < len(min_vals): # max_vals has the same len
            min_vals = np.nanmin(min_vals, axis=0)
            max_vals = np.nanmax(max_vals, axis=0)
            ebm.feature_bounds_ = np.array(list(zip(min_vals, max_vals)), np.float64)

    if hasattr(ebm, 'breakpoint_iteration_'):
        del ebm.breakpoint_iteration_

    if any(ebm.n_features_in_ != len(model.bins_) for model in models):  # pragma: no cover
        raise Exception("All bins_ should have the same length as the number of features.")

    new_bins = []
    for idx in range(ebm.n_features_in_):
        level_end = max(len(model.bins_[idx]) for model in models)
        new_leveled_bins = []
        for level_idx in range(level_end):
            bagged_bins = []
            for model in models:
                bin_levels = model.bins_[idx]
                bagged_bins.append(bin_levels[min(level_idx, len(bin_levels) - 1)])

            if len(set(type(bins) for bins in bagged_bins)) != 1:  # pragma: no cover
                raise Exception("All models should have the same feature types.")

            if isinstance(bagged_bins[0], dict):
                # categorical
                merged_keys = sorted(set(chain.from_iterable(bin.keys() for bin in bagged_bins)))
                # TODO: for now we just support alphabetical ordering in merged models, but
                # we could do all sort of special processing like trying to figure out if the original
                # ordering was by prevalence or alphabetical and then attempting to preserve that
                # order and also handling merged categories (where two categories map to a single score)
                merged_bins = dict(zip(merged_keys, count(1)))
            else:
                # continuous
                merged_bins = np.array(sorted(set(chain.from_iterable(bagged_bins))), np.float64)
            new_leveled_bins.append(merged_bins)
        new_bins.append(new_leveled_bins)
    _deduplicate_bins(new_bins)
    ebm.bins_ = new_bins

    bag_weights = []
    model_weights = []
    for model in models:
        avg_weight = np.average([tensor.sum() for tensor in model.bin_weights_])
        model_weights.append(avg_weight)

        n_outer_bags = -1
        if hasattr(model, 'bagged_additive_terms_'):
            if 0 < len(model.bagged_additive_terms_):
                n_outer_bags = len(model.bagged_additive_terms_[0])

        model_bag_weights = getattr(model, 'bag_weights_', None)
        if model_bag_weights is None:
            # this model wasn't the result of a merge, so get the total weight for the model
            # every feature group in a model should have the same weight, but perhaps the user edited
            # the model weights and they don't agree.  We handle these by taking the average
            model_bag_weights = [avg_weight] * n_outer_bags
        elif len(model_bag_weights) != n_outer_bags:
            raise Exception("self.bagged_weights_ should have the same length as n_outer_bags.")

        bag_weights.extend(model_bag_weights)
    # this attribute wasn't available in the original model since we can calculate it for non-merged
    # models, but once a model is merged we need to preserve it for future merging or other uses
    # of the ebm.bagged_additive_terms_ attribute
    ebm.bag_weights_ = bag_weights

    fg_dicts = []
    all_fg = set()
    for model in models:
        fg_sorted = [tuple(sorted(feature_idxs)) for feature_idxs in model.term_features_]
        fg_dicts.append(dict(zip(fg_sorted, count(0))))
        all_fg.update(fg_sorted)

    all_fg = list(all_fg)
    keys = ([len(feature_idxs)] + sorted(feature_idxs) for feature_idxs in all_fg)
    sorted_items = sorted(zip(keys, all_fg))
    sorted_fgs = [x[1] for x in sorted_items]
    # TODO: in the future we might at this point try and figure out the most 
    #       common feature ordering within the feature groups
    ebm.term_features_ = sorted_fgs


    ebm.bin_weights_ = []
    ebm.bagged_additive_terms_ = []
    for sorted_fg in sorted_fgs:
        # many times we'll have interaction mismatches where an interaction will be in one
        # model, but not the other.  We need to estimate the bin_weight_ tensors that would have been, 
        # so we'll use the feature groups that we do have to estimate the distribution of weight
        # and then scale it by the weights in each bag

        bin_weight_percentages = []
        for model, fg_dict, model_weight in zip(models, fg_dicts, model_weights):
            term_idx = fg_dict.get(sorted_fg)
            if term_idx is not None:
                fixed_tensor = _harmonize_tensor(
                    sorted_fg,
                    ebm.bins_, 
                    ebm.feature_bounds_,
                    model.term_features_[term_idx], 
                    model.bins_,
                    model.feature_bounds_,
                    model.bin_weights_[term_idx], 
                    True
                )
                bin_weight_percentages.append(fixed_tensor * model_weight)

        # use this when we don't have a feature group in a model as a reasonable 
        # set of guesses for the distribution of the weight of the model
        bin_weight_percentages = np.sum(bin_weight_percentages, axis=0)
        bin_weight_percentages = bin_weight_percentages / bin_weight_percentages.sum()

        additive_shape = bin_weight_percentages.shape
        if 2 < n_classes:
            additive_shape = tuple(list(additive_shape) + [n_classes])

        new_bin_weights = []
        new_bagged_additive_terms = []
        for model, fg_dict, model_weight in zip(models, fg_dicts, model_weights):
            n_outer_bags = -1
            if hasattr(model, 'bagged_additive_terms_'):
                if 0 < len(model.bagged_additive_terms_):
                    n_outer_bags = len(model.bagged_additive_terms_[0])

            term_idx = fg_dict.get(sorted_fg)
            if term_idx is None:
                new_bin_weights.append(model_weight * bin_weight_percentages)
                new_bagged_additive_terms.extend(n_outer_bags * [np.zeros(additive_shape, np.float64)])
            else:
                harmonized_bin_weights = _harmonize_tensor(
                    sorted_fg,
                    ebm.bins_, 
                    ebm.feature_bounds_,
                    model.term_features_[term_idx], 
                    model.bins_,
                    model.feature_bounds_,
                    model.bin_weights_[term_idx], 
                    True
                )
                new_bin_weights.append(harmonized_bin_weights)
                for bag_idx in range(n_outer_bags):
                    harmonized_bagged_additive_terms = _harmonize_tensor(
                        sorted_fg,
                        ebm.bins_, 
                        ebm.feature_bounds_,
                        model.term_features_[term_idx], 
                        model.bins_,
                        model.feature_bounds_,
                        model.bagged_additive_terms_[term_idx][bag_idx], 
                        False
                    )
                    new_bagged_additive_terms.append(harmonized_bagged_additive_terms)
        ebm.bin_weights_.append(np.sum(new_bin_weights, axis=0))
        ebm.bagged_additive_terms_.append(np.array(new_bagged_additive_terms, np.float64))

    ebm.additive_terms_, ebm.term_standard_deviations_, ebm.intercept_ = _process_terms(
        n_classes, 
        ebm.n_samples_, 
        ebm.bagged_additive_terms_, 
        ebm.bin_weights_,
        ebm.bag_weights_
    )

    return ebm

# TODO: Clean up
class EBMUtils:
    
    @staticmethod
    def normalize_initial_random_seed(seed):  # pragma: no cover
        # Some languages do not support 64-bit values.  Other languages do not support unsigned integers.
        # Almost all languages support signed 32-bit integers, so we standardize on that for our 
        # random number seed values.  If the caller passes us a number that doesn't fit into a 
        # 32-bit signed integer, we convert it.  This conversion doesn't need to generate completely 
        # uniform results provided they are reasonably uniform, since this is just the seed.
        # 
        # We use a simple conversion because we use the same method in multiple languages, 
        # and we need to keep the results identical between them, so simplicity is key.
        # 
        # The result of the modulo operator is not standardized accross languages for 
        # negative numbers, so take the negative before the modulo if the number is negative.
        # https://torstencurdt.com/tech/posts/modulo-of-negative-numbers

        if 2147483647 <= seed:
            return seed % 2147483647
        if seed <= -2147483647:
            return -((-seed) % 2147483647)
        return seed

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
    def make_bag(y, test_size, random_state, is_classification):
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
                    raise Exception("If test_size >= 1, test_size should be a whole number.")
                n_test_samples = test_size 
            else:
                n_test_samples = ceil(n_samples * test_size)

            n_train_samples = n_samples - n_test_samples
            native = Native.get_native_singleton()

            # Adapt test size if too small relative to number of classes
            if is_classification:
                y_uniq = len(set(y))
                if n_test_samples < y_uniq:  # pragma: no cover
                    warnings.warn(
                        "Too few samples per class, adapting test size to guarantee 1 sample per class."
                    )
                    n_test_samples = y_uniq
                    n_train_samples = n_samples - n_test_samples

                return native.stratified_sampling_without_replacement(
                    random_state,
                    y_uniq,
                    n_train_samples,
                    n_test_samples,
                    y
                )
            else:
                return native.sample_without_replacement(
                    random_state,
                    n_train_samples,
                    n_test_samples
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
                        vals[idx] = "NaN" # this is what JavaScript outputs for 0/0
                    elif val == np.inf:
                        vals[idx] = "Infinity" # this is what JavaScript outputs for 1/0
                    elif val == -np.inf:
                        vals[idx] = "-Infinity" # this is what JavaScript outputs for -1/0
            else:
                for nested in vals:
                    EBMUtils.jsonify_lists(nested)
        return vals # we modify in place, but return it just for easy access

    @staticmethod
    def jsonify_item(val):
        # JSON doesn't have NaN, or infinities, but javaScript has these, so use javaScript strings
        if isnan(val):
            val = "NaN" # this is what JavaScript outputs for 0/0
        elif val == np.inf:
            val = "Infinity" # this is what JavaScript outputs for 1/0
        elif val == -np.inf:
            val = "-Infinity" # this is what JavaScript outputs for -1/0
        return val

    @staticmethod
    def cyclic_gradient_boost(
        dataset,
        bag,
        scores,
        term_features,
        n_inner_bags,
        generate_update_options,
        learning_rate,
        min_samples_leaf,
        max_leaves,
        early_stopping_rounds,
        early_stopping_tolerance,
        max_rounds,
        noise_scale,
        bin_weights,
        random_state,
        optional_temp_params=None,
    ):
        min_metric = np.inf
        episode_index = 0
        with Booster(
            dataset,
            bag,
            scores,
            term_features,
            n_inner_bags,
            random_state,
            optional_temp_params,
        ) as booster:
            no_change_run_length = 0
            bp_metric = np.inf
            log.info("Start boosting")
            for episode_index in range(max_rounds):
                if episode_index % 10 == 0:
                    log.debug("Sweep Index {0}".format(episode_index))
                    log.debug("Metric: {0}".format(min_metric))

                for term_idx in range(len(term_features)):
                    gain = booster.generate_term_update(
                        term_idx=term_idx,
                        generate_update_options=generate_update_options,
                        learning_rate=learning_rate,
                        min_samples_leaf=min_samples_leaf,
                        max_leaves=max_leaves,
                    )

                    if noise_scale: # Differentially private updates
                        splits = booster.get_term_update_splits()[0]

                        term_update_tensor = booster.get_term_update_expanded()
                        noisy_update_tensor = term_update_tensor.copy()

                        splits_iter = [0] + list(splits + 1) + [len(term_update_tensor)] # Make splits iteration friendly
                        # Loop through all random splits and add noise before updating
                        for f, s in zip(splits_iter[:-1], splits_iter[1:]):
                            if s == 1: 
                                continue # Skip cuts that fall on 0th (missing value) bin -- missing values not supported in DP

                            noise = np.random.normal(0.0, noise_scale)
                            noisy_update_tensor[f:s] = term_update_tensor[f:s] + noise

                            # Native code will be returning sums of residuals in slices, not averages.
                            # Compute noisy average by dividing noisy sum by noisy bin weights
                            instance_weight = np.sum(bin_weights[term_idx][f:s])
                            noisy_update_tensor[f:s] = noisy_update_tensor[f:s] / instance_weight

                        noisy_update_tensor = noisy_update_tensor * -1 # Invert gradients before updates
                        booster.set_term_update_expanded(term_idx, noisy_update_tensor)


                    curr_metric = booster.apply_term_update()

                    min_metric = min(curr_metric, min_metric)

                # TODO PK this early_stopping_tolerance is a little inconsistent
                #      since it triggers intermittently and only re-triggers if the
                #      threshold is re-passed, but not based on a smooth windowed set
                #      of checks.  We can do better by keeping a list of the last
                #      number of measurements to have a consistent window of values.
                #      If we only cared about the metric at the start and end of the epoch
                #      window a circular buffer would be best choice with O(1).
                if no_change_run_length == 0:
                    bp_metric = min_metric
                if min_metric + early_stopping_tolerance < bp_metric:
                    no_change_run_length = 0
                else:
                    no_change_run_length += 1

                if (
                    early_stopping_rounds >= 0
                    and no_change_run_length >= early_stopping_rounds
                ):
                    break

            log.info(
                "End boosting, Best Metric: {0}, Num Rounds: {1}".format(
                    min_metric, episode_index
                )
            )

            # TODO: Add more ways to call alternative get_current_model
            # Use latest model if there are no instances in the (transposed) validation set 
            # or if training with privacy
            if bag is None or noise_scale is not None:
                model_update = booster.get_current_model()
            else:
                model_update = booster.get_best_model()

        return model_update, episode_index

    @staticmethod
    def get_interactions(
        dataset,
        bag,
        scores,
        iter_term_features,
        min_samples_leaf,
        optional_temp_params=None,
    ):
        interaction_scores = []
        with InteractionDetector(dataset, bag, scores, optional_temp_params) as interaction_detector:
            for feature_idxs in iter_term_features:
                score = interaction_detector.get_interaction_score(
                    feature_idxs, min_samples_leaf,
                )
                interaction_scores.append((score, feature_idxs))

        interaction_scores.sort(reverse=True)
        return list(map(operator.itemgetter(1), interaction_scores))


class DPUtils:

    @staticmethod
    def calc_classic_noise_multi(total_queries, target_epsilon, delta, sensitivity):
        variance = (8*total_queries*sensitivity**2 * np.log(np.exp(1) + target_epsilon / delta)) / target_epsilon ** 2
        return np.sqrt(variance)

    @staticmethod
    def calc_gdp_noise_multi(total_queries, target_epsilon, delta):
        ''' GDP analysis following Algorithm 2 in: https://arxiv.org/abs/2106.09680. 
        '''
        def f(mu, eps, delta):
            return DPUtils.delta_eps_mu(eps, mu) - delta

        final_mu = brentq(lambda x: f(x, target_epsilon, delta), 1e-5, 1000)
        sigma = np.sqrt(total_queries) / final_mu
        return sigma

    # General calculations, largely borrowed from tensorflow/privacy and presented in https://arxiv.org/abs/1911.11607
    @staticmethod
    def delta_eps_mu(eps, mu):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L44
        '''
        return norm.cdf(-eps/mu + mu/2) - np.exp(eps) * norm.cdf(-eps/mu - mu/2)

    @staticmethod
    def eps_from_mu(mu, delta):
        ''' Code adapted from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/gdp_accountant.py#L50
        '''
        def f(x):
            return DPUtils.delta_eps_mu(x, mu)-delta    
        return root_scalar(f, bracket=[0, 500], method='brentq').root

    @staticmethod
    def private_numeric_binning(col_data, sample_weight, noise_scale, max_bins, min_val, max_val):
        uniform_weights, uniform_edges = np.histogram(col_data, bins=max_bins*2, range=(min_val, max_val), weights=sample_weight)
        noisy_weights = uniform_weights + np.random.normal(0, noise_scale, size=uniform_weights.shape[0])
        
        # Postprocess to ensure realistic bin values (min=0)
        noisy_weights = np.clip(noisy_weights, 0, None)

        # TODO PK: check with Harsha, but we can probably alternate the taking of nibbles from both ends
        # so that the larger leftover bin tends to be in the center rather than on the right.

        # Greedily collapse bins until they meet or exceed target_weight threshold
        sample_weight_total = len(col_data) if sample_weight is None else np.sum(sample_weight)
        target_weight = sample_weight_total / max_bins
        bin_weights, bin_cuts = [0], [uniform_edges[0]]
        curr_weight = 0
        for index, right_edge in enumerate(uniform_edges[1:]):
            curr_weight += noisy_weights[index]
            if curr_weight >= target_weight:
                bin_cuts.append(right_edge)
                bin_weights.append(curr_weight)
                curr_weight = 0

        if len(bin_weights) == 1:
            # since we're adding unbounded random noise, it's possible that the total weight is less than the
            # threshold required for a single bin.  It could in theory even be negative.
            # clip to the target_weight.  If we had more than the target weight we'd have a bin

            bin_weights.append(target_weight)
            bin_cuts = np.empty(0, dtype=np.float64)
        else:
            # Ignore min/max value as part of cut definition
            bin_cuts = np.array(bin_cuts, dtype=np.float64)[1:-1]

            # All leftover datapoints get collapsed into final bin
            bin_weights[-1] += curr_weight

        return bin_cuts, bin_weights

    @staticmethod
    def private_categorical_binning(col_data, sample_weight, noise_scale, max_bins):
        # Initialize estimate
        col_data = col_data.astype('U')
        uniq_vals, uniq_idxs = np.unique(col_data, return_inverse=True)
        weights = np.bincount(uniq_idxs, weights=sample_weight, minlength=len(uniq_vals))

        weights = weights + np.random.normal(0, noise_scale, size=weights.shape[0])

        # Postprocess to ensure realistic bin values (min=0)
        weights = np.clip(weights, 0, None)

        # Collapse bins until target_weight is achieved.
        sample_weight_total = len(col_data) if sample_weight is None else np.sum(sample_weight)
        target_weight = sample_weight_total / max_bins
        small_bins = np.where(weights < target_weight)[0]
        if len(small_bins) > 0:
            other_weight = np.sum(weights[small_bins])
            mask = np.ones(weights.shape, dtype=bool)
            mask[small_bins] = False

            # Collapse all small bins into "DPOther"
            uniq_vals = np.append(uniq_vals[mask], "DPOther")
            weights = np.append(weights[mask], other_weight)

            if other_weight < target_weight:
                if len(weights) == 1:
                    # since we're adding unbounded random noise, it's possible that the total weight is less than the
                    # threshold required for a single bin.  It could in theory even be negative.
                    # clip to the target_weight
                    weights[0] = target_weight
                else:
                    # If "DPOther" bin is too small, absorb 1 more bin (guaranteed above threshold)
                    collapse_bin = np.argmin(weights[:-1])
                    mask = np.ones(weights.shape, dtype=bool)
                    mask[collapse_bin] = False

                    # Pack data into the final "DPOther" bin
                    weights[-1] += weights[collapse_bin]

                    # Delete absorbed bin
                    uniq_vals = uniq_vals[mask]
                    weights = weights[mask]

        return uniq_vals, weights

    @staticmethod
    def validate_eps_delta(eps, delta):
        if eps is None or eps <= 0 or delta is None or delta <= 0:
            raise ValueError(f"Epsilon: '{eps}' and delta: '{delta}' must be set to positive numbers")
