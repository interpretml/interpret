# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from numpy import (
    arange as arange,
    array_equal as array_equal,
    concatenate as concatenate,
    empty as empty,
    float64 as float64,
    fromiter as fromiter,
    full as full,
    int64 as int64,
    zeros as zeros,
)
from itertools import repeat, chain, compress
from operator import itemgetter, is_not, attrgetter

from ...utils._clean_x import (
    unify_columns_schematized,
    categorical_encode,
)
from ...utils._native import Native

_log = logging.getLogger(__name__)

_is_contiguous = attrgetter("flags.c_contiguous")
_float_type_eq = np.dtype(np.float64).__eq__
_dtype = attrgetter("dtype")
_continuous_eq = "continuous".__eq__
_from_iterable = chain.from_iterable
_array_zero = np.zeros(1, np.int64)
_repeat_negativeone = repeat(-1)


def eval_terms(
    X,
    n_samples,
    feature_names_in,
    feature_types_in,
    bins,
    term_features,
    _tuple=tuple,
    _map=map,
    _min=min,
    _len=len,
    arange=arange,
    array_equal=array_equal,
    concatenate=concatenate,
    empty=empty,
    fromiter=fromiter,
    int64=int64,
    zeros=zeros,
    _repeat_negativeone=_repeat_negativeone,
    _array_zero=_array_zero,
):
    # Prior to calling this function, call remove_extra_bins which will
    # eliminate extra work in this function. The only place we need fast
    # performance here is when called from ebm_predict_scores or ebm_eval_terms.

    continuous_bins = list(
        _from_iterable(compress(bins, _map(_continuous_eq, feature_types_in)))
    )
    if not all(_map(_float_type_eq, _map(_dtype, continuous_bins))):
        raise ValueError(
            "All bins for continuous features must be of dtype np.float64."
        )
    if not all(_map(_is_contiguous, continuous_bins)):
        raise ValueError(
            "All bins for continuous features must be C-contiguous arrays."
        )

    get_col = unify_columns_schematized(
        X, n_samples, feature_names_in, feature_types_in
    )

    get_feature_type = feature_types_in.__getitem__

    cached_raw = {}
    cached_raw_get = cached_raw.get
    cached_raw_set = cached_raw.__setitem__
    cached_discretized = {}
    cached_discretized_get = cached_discretized.get
    cached_discretized_set = cached_discretized.__setitem__
    Discretize = Native.get_native_singleton()._unsafe.Discretize
    bins_getitem = bins.__getitem__
    for feature_idxs, num_features in zip(term_features, _map(_len, term_features)):
        term_discretized = []
        for feature_idx in feature_idxs:
            bin_levels = bins_getitem(feature_idx)
            bin_level = _min(_len(bin_levels), num_features)
            key = (feature_idx, bin_level)
            discretized = cached_discretized_get(key)
            if discretized is None:
                raw = cached_raw_get(feature_idx)
                if raw is None:
                    raw = get_col(feature_idx, get_feature_type(feature_idx))
                    cached_raw_set(feature_idx, raw)

                # these are the variables in raw
                # bad, X_col, uniques, nonmissings = raw

                uniques = raw[2]
                if uniques is None:
                    # continuous feature

                    cuts = bin_levels[bin_level - 1]
                    discretized = empty(n_samples, int64)

                    return_code = Discretize(
                        n_samples,
                        raw[1].ctypes.data,
                        cuts.shape[0],
                        cuts.ctypes.data,
                        discretized.ctypes.data,
                    )
                    if return_code:  # pragma: no cover
                        raise Native._get_native_exception(return_code, "Discretize")

                    bad = raw[0]
                    if bad is not None:
                        discretized[bad] = -1
                else:
                    # categorical feature
                    nonmissings = raw[3]
                    categories = bin_levels[bin_level - 1]

                    mapping = fromiter(
                        _map(categories.get, uniques, _repeat_negativeone),
                        int64,
                        uniques.shape[0],
                    )

                    n_cat = _len(categories)
                    if mapping.shape[0] <= n_cat:
                        if array_equal(
                            mapping, arange(1, mapping.shape[0] + 1, dtype=int64)
                        ):
                            # CategoricalDType can encode values as np.int8. We cannot allow an
                            # int8 to overflow when we add 1, so convert to int64 first, and we
                            # also need to make a copy here because we cache the raw data and
                            # re-use it for different binning levels on the same feature.

                            discretized = raw[1].astype(int64)
                            discretized += 1

                            if nonmissings is not None and nonmissings is not False:
                                discretized_tmp = zeros(nonmissings.shape[0], int64)
                                discretized_tmp[nonmissings] = discretized
                                discretized = discretized_tmp
                        else:
                            if nonmissings is None:
                                # discretized should be all positive if nonmissings is None
                                discretized = mapping[raw[1]]
                            elif nonmissings is False:
                                # missing values are -1 in discretized, so append 0 to the map, which is index -1
                                discretized = concatenate((mapping, _array_zero))[
                                    raw[1]
                                ]
                            else:
                                discretized = zeros(nonmissings.shape[0], int64)
                                discretized[nonmissings] = mapping[raw[1]]
                    else:
                        if array_equal(
                            mapping[:n_cat], arange(1, n_cat + 1, dtype=int64)
                        ):
                            # CategoricalDType can encode values as np.int8. We cannot allow an
                            # int8 to overflow when we add 1, so convert to int64 first, and we
                            # also need to make a copy here because we cache the raw data and
                            # re-use it for different binning levels on the same feature.

                            discretized = raw[1].astype(int64)
                            discretized += 1
                            discretized[n_cat < discretized] = -1

                            if nonmissings is not None and nonmissings is not False:
                                discretized_tmp = zeros(nonmissings.shape[0], int64)
                                discretized_tmp[nonmissings] = discretized
                                discretized = discretized_tmp
                        else:
                            if nonmissings is None:
                                # discretized should be all positive if nonmissings is None
                                discretized = mapping[raw[1]]
                            elif nonmissings is False:
                                # missing values are -1 in discretized, so append 0 to the map, which is index -1
                                discretized = concatenate((mapping, _array_zero))[
                                    raw[1]
                                ]
                            else:
                                discretized = zeros(nonmissings.shape[0], int64)
                                discretized[nonmissings] = mapping[raw[1]]

                cached_discretized_set(key, discretized)
            term_discretized.append(discretized)
        yield _tuple(term_discretized)


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
        full(
            n_samples
            if isinstance(intercept, float) or intercept.shape[0] == 1
            else (n_samples, intercept.shape[0]),
            intercept,
            float64,
        )
        if init_score is None
        else init_score + intercept
    )

    if n_samples > 0:
        sample_scores_iadd = sample_scores.__iadd__
        for scores, bin_indexes in zip(
            term_scores,
            eval_terms(
                X, n_samples, feature_names_in, feature_types_in, bins, term_features
            ),
        ):
            sample_scores_iadd(scores[bin_indexes])

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
        for term_idx, bin_indexes in enumerate(
            eval_terms(
                X, n_samples, feature_names_in, feature_types_in, bins, term_features
            )
        ):
            explanations[:, term_idx] = term_scores[term_idx][bin_indexes]

    return explanations


def make_bin_weights(
    X, n_samples, sample_weight, feature_names_in, feature_types_in, bins, term_features
):
    bin_weights = [None] * len(term_features)
    for term_idx, bin_indexes in enumerate(
        eval_terms(
            X, n_samples, feature_names_in, feature_types_in, bins, term_features
        )
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
