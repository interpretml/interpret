# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from math import isnan

import numpy as np

from ._native import Native

_log = logging.getLogger(__name__)


def _make_histogram_edges(min_feature_val, max_feature_val, histogram_weights):
    native = Native.get_native_singleton()

    # the EBM model spec disallows subnormal values since they can be problems in computation
    # and serialization. We only use the min_feature_value and max_feature_value in the histogram edges as information
    # so this won't affect binning or reporting other than potentially visualization where subnormals can be ignored

    min_feature_val = native.clean_float(min_feature_val)
    max_feature_val = native.clean_float(max_feature_val)

    n_cuts = len(histogram_weights) - 3
    cuts = native.cut_uniform(
        np.array([min_feature_val, max_feature_val], np.float64), n_cuts
    )
    if len(cuts) != n_cuts:
        raise Exception(
            f"There are insufficient floating point values between min_feature_val={min_feature_val} to max_feature_val={max_feature_val} to make {n_cuts} cuts"
        )

    return np.concatenate(([min_feature_val], cuts, [max_feature_val]))


def make_all_histogram_edges(feature_bounds, histogram_weights):
    if feature_bounds is None:
        msg = "feature_bounds is None"
        _log.error(msg)
        raise ValueError(msg)

    if histogram_weights is None:
        msg = "histogram_weights is None"
        _log.error(msg)
        raise ValueError(msg)

    if len(feature_bounds) != len(histogram_weights):
        msg = "feature_bounds and histogram_weights have different lengths"
        _log.error(msg)
        raise ValueError(msg)

    ret = [None] * len(histogram_weights)
    for idx in range(len(ret)):
        min_feature_val = feature_bounds[idx, 0]
        max_feature_val = feature_bounds[idx, 1]

        if not isnan(min_feature_val) and not isnan(max_feature_val):
            histogram_bin_weights = histogram_weights[idx]
            if histogram_bin_weights is None:
                msg = "histogram_weights[idx] is None"
                _log.error(msg)
                raise ValueError(msg)

            ret[idx] = _make_histogram_edges(
                min_feature_val, max_feature_val, histogram_bin_weights
            )
    return ret
