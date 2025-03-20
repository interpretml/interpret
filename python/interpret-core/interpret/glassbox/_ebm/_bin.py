# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from itertools import repeat
from operator import itemgetter, is_not

from ...utils._clean_x import unify_columns
from ...utils._native import Native

_log = logging.getLogger(__name__)


_none_list = [None]
_none_ndarray = np.array(None)
_repeat_none = repeat(None)
_itemgetter0 = itemgetter(0)
_itemgetter1 = itemgetter(1)
_slice_remove_last = slice(None, -1)


def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # called under: predict

    # prior to calling this function, call remove_extra_bins which will eliminate extra work in this function

    # This generator function returns data as the feature data within terms gets read.  Normally for
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # would be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    # Flatten the term_features array to make one entry per feature within each term
    # each item in the list contains placeholders for the binned array that we need
    # to complete the term. We fill these with None initially.  At the end of the array
    # is the term_idx. So it looks approximately like this:
    # eg: [[None, 0], [None, 1], [None, 2], [None, None, 3], [None, None, None, 4]]

    requests = []
    waiting = {}
    for term_idx, feature_idxs in enumerate(term_features):
        # the first len(feature_idxs) items hold the binned data that we get back as it arrives
        num_features = len(feature_idxs)
        requirements = _none_list * (num_features + 1)
        requirements[-1] = term_idx
        for feature_idx in feature_idxs:
            bin_levels = bins[feature_idx]
            feature_bins = bin_levels[min(len(bin_levels), num_features) - 1]
            if isinstance(feature_bins, dict):
                # categorical feature
                key = (feature_idx, id(feature_bins))
            else:
                # continuous feature
                feature_bins = None
                key = feature_idx
            waiting_list = waiting.get(key)
            if waiting_list is None:
                waiting[key] = [requirements]
                requests.append((feature_idx, feature_bins))
            else:
                waiting_list.append(requirements)

    request_feature_idxs = list(map(_itemgetter0, requests))

    native = Native.get_native_singleton()

    for column_feature_idx, (_, X_col, column_categories, bad) in zip(
        request_feature_idxs,
        unify_columns(
            X,
            request_feature_idxs,
            map(_itemgetter1, requests),
            feature_names_in,
            feature_types_in,
            None,
            True,
        ),
    ):
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
                term_idx = requirements[-1]
                feature_idxs = term_features[term_idx]
                level_idx = min(max_level, len(feature_idxs)) - 1
                bin_indexes = binning_completed[level_idx]
                if bin_indexes is None:
                    bin_indexes = native.discretize(X_col, bin_levels[level_idx])
                    if bad is not None:
                        bin_indexes[bad] = -1
                    binning_completed[level_idx] = bin_indexes
                for dimension_idx, term_feature_idx in enumerate(feature_idxs):
                    # TODO: consider making it illegal to duplicate features in terms
                    # then use: dimension_idx = feature_idxs.index(column_feature_idx)
                    if term_feature_idx == column_feature_idx:
                        requirements[dimension_idx] = bin_indexes

                if all(map(is_not, requirements, _repeat_none)):
                    yield term_idx, requirements[_slice_remove_last]
                    # clear references so that the garbage collector can free them
                    requirements.clear()
        else:
            # categorical feature

            # if bad is not None:
            #     # TODO: we could pass out a single bool (not an array) if these aren't continuous convertible
            #     pass  # TODO: improve this handling

            for requirements in waiting[(column_feature_idx, id(column_categories))]:
                term_idx = requirements[-1]
                for dimension_idx, term_feature_idx in enumerate(
                    term_features[term_idx]
                ):
                    # TODO: consider making it illegal to duplicate features in terms
                    # then use: dimension_idx = feature_idxs.index(column_feature_idx)
                    if term_feature_idx == column_feature_idx:
                        # "term_categories is column_categories" since any term in the waiting_list must have
                        # one of it's elements match this (feature_idx, categories) index, and all items in this
                        # term need to have the same categories since they came from the same bin_level
                        requirements[dimension_idx] = X_col

                if all(map(is_not, requirements, _repeat_none)):
                    yield term_idx, requirements[_slice_remove_last]
                    # clear references so that the garbage collector can free them
                    requirements.clear()


def ebm_predict_scores(
    X,
    n_samples,
    init_score,
    feature_names_in,
    feature_types_in,
    bins,
    intercept,
    term_scores,
    term_features,
):
    sample_scores = (
        np.full(
            n_samples
            if isinstance(intercept, float) or len(intercept) == 1
            else (n_samples, len(intercept)),
            intercept,
            dtype=np.float64,
        )
        if init_score is None
        else init_score + intercept
    )

    if n_samples > 0:
        for term_idx, bin_indexes in eval_terms(
            X, n_samples, feature_names_in, feature_types_in, bins, term_features
        ):
            sample_scores += term_scores[term_idx][tuple(bin_indexes)]

    return sample_scores


def ebm_eval_terms(
    X,
    n_samples,
    n_scores,
    feature_names_in,
    feature_types_in,
    bins,
    term_scores,
    term_features,
):
    explanations = np.empty(
        (n_samples, len(term_features))
        if n_scores == 1
        else (n_samples, len(term_features), n_scores),
        dtype=np.float64,
    )

    if n_samples > 0:
        for term_idx, bin_indexes in eval_terms(
            X, n_samples, feature_names_in, feature_types_in, bins, term_features
        ):
            explanations[:, term_idx] = term_scores[term_idx][tuple(bin_indexes)]

    return explanations


def make_bin_weights(
    X, n_samples, sample_weight, feature_names_in, feature_types_in, bins, term_features
):
    bin_weights = _none_list * len(term_features)
    for term_idx, bin_indexes in eval_terms(
        X, n_samples, feature_names_in, feature_types_in, bins, term_features
    ):
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
            term_bin_weights = np.bincount(
                flat_indexes, weights=sample_weight, minlength=multiple
            )
        term_bin_weights = term_bin_weights.astype(np.float64, copy=False)
        term_bin_weights = term_bin_weights.reshape(dimensions)
        bin_weights[term_idx] = term_bin_weights

    return bin_weights
