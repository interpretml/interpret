# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from itertools import repeat
from operator import itemgetter, is_not

from ...utils._clean_x import unify_columns, categorical_encode
from ...utils._native import Native

_log = logging.getLogger(__name__)


_none_list = [None]
_none_ndarray = np.array(None)
_repeat_none = repeat(None)
_slice_remove_last = slice(None, -1)


def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # prior to calling this function, call remove_extra_bins which will eliminate extra work in this function

    # This generator function returns data as the feature data within terms gets read.  Normally for
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # would be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    waiting = {}
    # term_features are guaranteed to be ordered by: num_features, [feature_idxes]
    # Which typically means that the mains are processed in order first
    # by feature_idx.
    for term_idx, feature_idxs in enumerate(term_features):
        # the first len(feature_idxs) items hold the binned data that we get back as it arrives
        num_features = len(feature_idxs)
        requirements = _none_list * (num_features + 1)
        requirements[-1] = term_idx
        for feature_idx in feature_idxs:
            waiting_list = waiting.get(feature_idx)
            if waiting_list is None:
                # rely on the guarantee that iterating over dict is by insertion order
                waiting[feature_idx] = [requirements]
            else:
                waiting_list.append(requirements)

    native = Native.get_native_singleton()

    for column_feature_idx, (
        _,
        X_col,
        bad,
        uniques,
        nonmissings,
    ) in zip(
        waiting.keys(),
        unify_columns(
            X, waiting.keys(), feature_names_in, feature_types_in, None, False, True
        ),
    ):
        if uniques is None:
            # continuous feature

            if n_samples != len(X_col):
                msg = "The columns of X are mismatched in the number of of samples"
                _log.error(msg)
                raise ValueError(msg)

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

            if nonmissings is None or nonmissings is False:
                if n_samples != len(X_col):
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)
            else:
                if n_samples != len(nonmissings):
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)

            bin_levels = bins[column_feature_idx]
            max_level = len(bin_levels)
            binning_completed = _none_list * max_level
            for requirements in waiting[column_feature_idx]:
                term_idx = requirements[-1]
                feature_idxs = term_features[term_idx]
                level_idx = min(max_level, len(feature_idxs)) - 1
                bin_indexes = binning_completed[level_idx]
                if bin_indexes is None:
                    bin_indexes, _ = categorical_encode(
                        uniques, X_col, nonmissings, bin_levels[level_idx]
                    )
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
            np.float64,
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
        np.float64,
        "F",
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
