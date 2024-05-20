# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

import numpy as np

from ._native import Native
from ._clean_x import unify_columns

_log = logging.getLogger(__name__)
_none_ndarray = np.array(None)


def bin_native(
    n_classes,
    feature_idxs,
    bins_iter,
    X,
    y,
    sample_weight,
    feature_names_in,
    feature_types_in,
):
    # called under: fit

    # bin_native exists separately from bin_native_by_dimension so that we can eventually boost 
    # mains, pairs, and interactions together at the same time. In order to accomplish this though,
    # we want to be able to insert the same feature multiple times with different binning resolutions.  
    # The bins_iter parameter exists here in order to be able to pass in features multiple times with 
    # different bin resolutions.
    # TODO: implement the above and also a method of keeping track when the same feature appears multiple
    # times with different resolutions.  Today we have a 1:1 correspondence with feature indexes, so
    # feature #99 is located at the 99th column. If a feature is inserted mutliple times with different
    # resolutions we need a way to map feature indexes and the binning resultion to a column index.

    _log.info("Creating native dataset")

    n_samples = len(y)

    native = Native.get_native_singleton()

    responses = []
    requests = []
    for request in zip(feature_idxs, bins_iter):
        responses.append(request)
        if not isinstance(request[1], dict):
            # continuous feature.  Don't include the continuous definition
            request = (request[0], None)
        requests.append(request)

    n_weights = 0
    if sample_weight is not None:
        n_weights = 1
        if not sample_weight.flags.c_contiguous:
            # sample_weight could be a slice that has a stride.  We need contiguous for caling into C
            sample_weight = sample_weight.copy()

    if not y.flags.c_contiguous:
        # y could be a slice that has a stride.  We need contiguous for caling into C
        y = y.copy()

    n_bytes = native.measure_dataset_header(len(requests), n_weights, 1)
    for (feature_idx, feature_bins), (_, X_col, _, bad) in zip(
        responses,
        unify_columns(X, requests, feature_names_in, feature_types_in, None, False),
    ):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if not X_col.flags.c_contiguous:
            # X_col could be a slice that has a stride.  We need contiguous for caling into C
            X_col = X_col.copy()

        if isinstance(feature_bins, dict):
            # categorical feature
            n_bins = 2 if len(feature_bins) == 0 else (max(feature_bins.values()) + 2)
        else:
            # continuous feature
            X_col = native.discretize(X_col, feature_bins)
            n_bins = len(feature_bins) + 3

        if bad is not None:
            X_col[bad != _none_ndarray] = n_bins - 1

        n_bytes += native.measure_feature(
            n_bins,
            np.count_nonzero(X_col) != len(X_col),
            bad is not None,
            feature_types_in[feature_idx] == "nominal",
            X_col,
        )

    if sample_weight is not None:
        n_bytes += native.measure_weight(sample_weight)

    if y.dtype == np.float64:
        n_bytes += native.measure_regression_target(y)
    elif y.dtype == np.int64:
        n_bytes += native.measure_classification_target(n_classes, y)
    else:
        msg = "y must be either float64 or int64"
        _log.error(msg)
        raise ValueError(msg)

    dataset = np.empty(n_bytes, np.ubyte)  # joblib loky doesn't support RawArray

    native.fill_dataset_header(len(requests), n_weights, 1, dataset)

    for (feature_idx, feature_bins), (_, X_col, _, bad) in zip(
        responses,
        unify_columns(X, requests, feature_names_in, feature_types_in, None, False),
    ):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if not X_col.flags.c_contiguous:
            # X_col could be a slice that has a stride.  We need contiguous for caling into C
            X_col = X_col.copy()

        if isinstance(feature_bins, dict):
            # categorical feature
            n_bins = 2 if len(feature_bins) == 0 else (max(feature_bins.values()) + 2)
        else:
            # continuous feature
            X_col = native.discretize(X_col, feature_bins)
            n_bins = len(feature_bins) + 3

        if bad is not None:
            X_col[bad != _none_ndarray] = n_bins - 1

        native.fill_feature(
            n_bins,
            np.count_nonzero(X_col) != len(X_col),
            bad is not None,
            feature_types_in[feature_idx] == "nominal",
            X_col,
            dataset,
        )

    if sample_weight is not None:
        native.fill_weight(sample_weight, dataset)

    if y.dtype == np.float64:
        native.fill_regression_target(y, dataset)
    elif y.dtype == np.int64:
        native.fill_classification_target(n_classes, y, dataset)
    else:
        msg = "y must be either float64 or int64"
        _log.error(msg)
        raise ValueError(msg)

    return dataset


def bin_native_by_dimension(
    n_classes,
    n_dimensions,
    bins,
    X,
    y,
    sample_weight,
    feature_names_in,
    feature_types_in,
):
    # called under: fit

    feature_idxs = range(len(feature_names_in))
    bins_iter = []
    for feature_idx in feature_idxs:
        bin_levels = bins[feature_idx]
        feature_bins = bin_levels[min(len(bin_levels), n_dimensions) - 1]
        bins_iter.append(feature_bins)

    return bin_native(
        n_classes,
        feature_idxs,
        bins_iter,
        X,
        y,
        sample_weight,
        feature_names_in,
        feature_types_in,
    )
