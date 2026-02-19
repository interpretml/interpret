# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np
from itertools import repeat, chain, compress
from operator import itemgetter, is_not, attrgetter

from ...utils._clean_x import unify_columns, categorical_encode
from ...utils._native import Native

_log = logging.getLogger(__name__)


def eval_terms(X, n_samples, feature_names_in, feature_types_in, bins, term_features):
    # Prior to calling this function, call remove_extra_bins which will
    # eliminate extra work in this function. The only place we need fast
    # performance here is when called from ebm_predict_scores or ebm_eval_terms.

    continuous_bins = list(
        chain.from_iterable(compress(bins, map("continuous".__eq__, feature_types_in)))
    )
    if not all(
        map(np.dtype(np.float64).__eq__, map(attrgetter("dtype"), continuous_bins))
    ):
        raise ValueError(
            "All bins for continuous features must be of dtype np.float64."
        )
    if not all(map(attrgetter("flags.c_contiguous"), continuous_bins)):
        raise ValueError(
            "All bins for continuous features must be C-contiguous arrays."
        )

    get_col = unify_columns(
        X, n_samples, feature_names_in, feature_types_in, None, True, True
    )

    cached_raw = {}
    cached_discretized = {}
    Discretize = Native.get_native_singleton()._unsafe.Discretize
    bins_getitem = bins.__getitem__
    for feature_idxs, num_features in zip(term_features, map(len, term_features)):
        term_discretized = []
        for feature_idx in feature_idxs:
            bin_levels = bins_getitem(feature_idx)
            bin_level = min(len(bin_levels), num_features)
            key = (feature_idx, bin_level)
            discretized = cached_discretized.get(key)
            if discretized is None:
                raw = cached_raw.get(feature_idx)
                if raw is None:
                    raw = get_col(feature_idx)
                    cached_raw[feature_idx] = raw

                # these are the variables in raw
                # _, nonmissings, uniques, X_col, bad = raw

                uniques = raw[2]
                if uniques is None:
                    # continuous feature

                    cuts = bin_levels[bin_level - 1]
                    discretized = np.empty(n_samples, np.int64)

                    return_code = Discretize(
                        n_samples,
                        raw[3].ctypes.data,
                        cuts.shape[0],
                        cuts.ctypes.data,
                        discretized.ctypes.data,
                    )
                    if return_code:  # pragma: no cover
                        raise Native._get_native_exception(return_code, "Discretize")

                    bad = raw[4]
                    if bad is not None:
                        discretized[bad] = -1
                else:
                    # categorical feature
                    discretized = categorical_encode(
                        uniques, raw[3], raw[1], bin_levels[bin_level - 1]
                    )

                cached_discretized[key] = discretized
            term_discretized.append(discretized)
        yield tuple(term_discretized)


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
            if isinstance(intercept, float) or intercept.shape[0] == 1
            else (n_samples, intercept.shape[0]),
            intercept,
            np.float64,
        )
        if init_score is None
        else init_score + intercept
    )

    if n_samples > 0:
        for scores, bin_indexes in zip(
            term_scores,
            eval_terms(
                X, n_samples, feature_names_in, feature_types_in, bins, term_features
            ),
        ):
            sample_scores += scores[bin_indexes]

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
