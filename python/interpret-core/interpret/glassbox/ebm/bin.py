# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from itertools import count
import numpy as np

import logging
_log = logging.getLogger(__name__)

from ...utils._native import Native
from ...utils._binning import unify_columns

_none_list = [None]
_none_ndarray = np.array(None)

def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # called under: predict

    # prior to calling this function, call _deduplicate_bins which will eliminate extra work in this function

    # this generator function returns data in whatever order it thinks is most efficient.  Normally for 
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # could be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    _log.info("eval_terms")

    requests = []
    waiting = dict()
    for term_idx, feature_idxs in enumerate(term_features):
        # the first len(feature_idxs) items hold the binned data that we get back as it arrives
        requirements = _none_list * (len(feature_idxs) + 1)
        requirements[-1] = term_idx
        for feature_idx in feature_idxs:
            bin_levels = bins[feature_idx]
            feature_bins = bin_levels[min(len(bin_levels), len(feature_idxs)) - 1]
            if isinstance(feature_bins, dict):
                # categorical feature
                request = (feature_idx, feature_bins)
                key = (feature_idx, id(feature_bins))
            else:
                # continuous feature
                request = (feature_idx, None)
                key = feature_idx
            waiting_list = waiting.get(key, None)
            if waiting_list is None:
                requests.append(request)
                waiting[key] = [requirements]
            else:
                waiting_list.append(requirements)

    native = Native.get_native_singleton()

    for (column_feature_idx, _), (_, X_col, column_categories, bad) in zip(requests, unify_columns(X, requests, feature_names_in, feature_types_in, None, True)):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if column_categories is None:
            # continuous feature

            if bad is not None:
                # TODO: we could pass out a bool array instead of objects for this function only
                bad = bad != _none_ndarray

            if not X_col.flags.c_contiguous:
                # we requrested this feature, so at some point we're going to call discretize, 
                # which requires contiguous memory
                X_col = X_col.copy()

            bin_levels = bins[column_feature_idx]
            max_level = len(bin_levels)
            binning_completed = _none_list * max_level
            for requirements in waiting[column_feature_idx]:
                if len(requirements) != 0:
                    term_idx = requirements[-1]
                    feature_idxs = term_features[term_idx]
                    is_done = True
                    for dimension_idx, term_feature_idx in enumerate(feature_idxs):
                        if term_feature_idx == column_feature_idx:
                            level_idx = min(max_level, len(feature_idxs)) - 1
                            bin_indexes = binning_completed[level_idx]
                            if bin_indexes is None:
                                cuts = bin_levels[level_idx]
                                bin_indexes = native.discretize(X_col, cuts)
                                if bad is not None:
                                    bin_indexes[bad] = -1
                                binning_completed[level_idx] = bin_indexes
                            requirements[dimension_idx] = bin_indexes
                        elif requirements[dimension_idx] is None:
                            is_done = False

                    if is_done:
                        yield term_idx, requirements[:-1]
                        # clear references so that the garbage collector can free them
                        requirements.clear()
        else:
            # categorical feature

            if bad is not None:
                # TODO: we could pass out a single bool (not an array) if these aren't continuous convertible
                pass # TODO: improve this handling

            for requirements in waiting[(column_feature_idx, id(column_categories))]:
                if len(requirements) != 0:
                    term_idx = requirements[-1]
                    feature_idxs = term_features[term_idx]
                    is_done = True
                    for dimension_idx, term_feature_idx in enumerate(feature_idxs):
                        if term_feature_idx == column_feature_idx:
                            # "term_categories is column_categories" since any term in the waiting_list must have
                            # one of it's elements match this (feature_idx, categories) index, and all items in this
                            # term need to have the same categories since they came from the same bin_level
                            requirements[dimension_idx] = X_col
                        elif requirements[dimension_idx] is None:
                            is_done = False

                    if is_done:
                        yield term_idx, requirements[:-1]
                        # clear references so that the garbage collector can free them
                        requirements.clear()

def ebm_decision_function(
    X, 
    n_samples, 
    feature_names_in, 
    feature_types_in, 
    bins, 
    intercept, 
    term_scores, 
    term_features
):
    if type(intercept) is float or len(intercept) == 1:
        sample_scores = np.full(n_samples, intercept, dtype=np.float64)
    else:
        sample_scores = np.full((n_samples, len(intercept)), intercept, dtype=np.float64)

    if 0 < n_samples:
        for term_idx, bin_indexes in eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
            sample_scores += term_scores[term_idx][tuple(bin_indexes)]

    return sample_scores

def ebm_decision_function_and_explain(
    X, 
    n_samples, 
    feature_names_in, 
    feature_types_in, 
    bins, 
    intercept, 
    term_scores, 
    term_features
):
    if type(intercept) is float or len(intercept) == 1:
        sample_scores = np.full(n_samples, intercept, dtype=np.float64)
        explanations = np.empty((n_samples, len(term_features)), dtype=np.float64)
    else:
        # TODO: add a test for multiclass calls to ebm_decision_function_and_explain
        sample_scores = np.full((n_samples, len(intercept)), intercept, dtype=np.float64)
        explanations = np.empty((n_samples, len(term_features), len(intercept)), dtype=np.float64)

    if 0 < n_samples:
        for term_idx, bin_indexes in eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
            scores = term_scores[term_idx][tuple(bin_indexes)]
            sample_scores += scores
            explanations[:, term_idx] = scores

    return sample_scores, explanations

def make_bin_weights(X, n_samples, sample_weight, feature_names_in, feature_types_in, bins, term_features):
    bin_weights = _none_list * len(term_features)
    for term_idx, bin_indexes in eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
        feature_idxs = term_features[term_idx]
        multiple = 1
        dimensions = []
        for dimension_idx in range(len(feature_idxs) - 1, -1, -1):
            feature_idx = feature_idxs[dimension_idx]
            bin_levels = bins[feature_idx]
            feature_bins = bin_levels[min(len(bin_levels), len(feature_idxs)) - 1]
            if isinstance(feature_bins, dict):
                # categorical feature
                n_bins = 2 if len(feature_bins) == 0 else max(feature_bins.values()) + 2
            else:
                # continuous feature
                n_bins = len(feature_bins) + 3

            dimensions.append(n_bins)
            dim_data = bin_indexes[dimension_idx]
            dim_data = np.where(dim_data < 0, n_bins - 1, dim_data)
            if multiple == 1:
                flat_indexes = dim_data
            else:
                flat_indexes += dim_data * multiple
            multiple *= n_bins
        dimensions = tuple(reversed(dimensions))

        if sample_weight is None:
            term_bin_weights = np.bincount(flat_indexes, minlength=multiple)
        else:
            term_bin_weights = np.bincount(flat_indexes, weights=sample_weight, minlength=multiple)
        term_bin_weights = term_bin_weights.astype(np.float64, copy=False)
        term_bin_weights = term_bin_weights.reshape(dimensions)
        bin_weights[term_idx] = term_bin_weights

    return bin_weights

def append_tensor(tensor, append_low=None, append_high=None):
    if append_low is None:
        if append_high is None:
            return tensor
        dim_slices = [slice(0, dim_len) for dim_len in tensor.shape]
        new_shape = [dim_len + int(is_high) for dim_len, is_high in zip(tensor.shape, append_high)]
    else:
        dim_slices = [slice(int(is_low), dim_len + int(is_low)) for dim_len, is_low in zip(tensor.shape, append_low)]
        if append_high is None:
            new_shape = [dim_len + int(is_low) for dim_len, is_low in zip(tensor.shape, append_low)]
        else:
            new_shape = [dim_len + int(is_low) + int(is_high) for dim_len, is_low, is_high in zip(tensor.shape, append_low, append_high)]

    if len(new_shape) != tensor.ndim:
        # multiclass
        new_shape.append(tensor.shape[-1])

    new_tensor = np.zeros(tuple(new_shape), dtype=tensor.dtype)
    new_tensor[tuple(dim_slices)] = tensor
    return new_tensor

def trim_tensor(tensor, trim_low=None, trim_high=None):
    if trim_low is None:
        if trim_high is None:
            return tensor
        dim_slices = [slice(0, -1 if is_high else None) for dim_len, is_high in zip(tensor.shape, trim_high)]
    else:
        if trim_high is None:
            dim_slices = [slice(int(is_low), None) for dim_len, is_low in zip(tensor.shape, trim_low)]
        else:
            dim_slices = [slice(int(is_low), -1 if is_high else None) for dim_len, is_low, is_high in zip(tensor.shape, trim_low, trim_high)]
    return tensor[tuple(dim_slices)]

def make_boosting_weights(term_bin_weights):
    # TODO: replace this function with a bool array that we generate in bin_native.. this function will crash
    # if there are samples with zero weights
    bin_data_weights = []
    for term_weights in term_bin_weights:
        if term_weights[-1] == 0:
            bin_data_weights.append(term_weights[:-1])
        else:
            bin_data_weights.append(term_weights)
    return bin_data_weights

def after_boosting(term_features, tensors, feature_bin_weights):
    # TODO: this isn't a problem today since any unnamed categories in the mains and the pairs are the same
    #       (they don't exist in the pairs today at all since DP-EBMs aren't pair enabled yet and we haven't
    #       made the option for them in regular EBMs), but when we eventually go that way then we'll
    #       need to examine the tensored term based bin weights to see what to do.  Alternatively, we could
    #       obtain this information from bin_native which would be cleaner since we only need it during boosting
    new_tensors=[]
    for term_idx, feature_idxs in enumerate(term_features):
        higher = [feature_bin_weights[feature_idx][-1] == 0 for feature_idx in feature_idxs]
        new_tensors.append(append_tensor(tensors[term_idx], None, higher))
    return new_tensors

def remove_last2(tensors, term_bin_weights):
    new_tensors=[]
    for idx, tensor, weights in zip(count(), tensors, term_bin_weights):
        n_dimensions = weights.ndim
        entire_tensor = [slice(None)] * n_dimensions
        higher = []
        for dimension_idx in range(n_dimensions):
            dim_slices = entire_tensor.copy()
            dim_slices[dimension_idx] = -1
            total_sum = np.sum(weights[tuple(dim_slices)])
            higher.append(True if total_sum == 0 else False)
        new_tensors.append(trim_tensor(tensor, None, higher))
    return new_tensors
