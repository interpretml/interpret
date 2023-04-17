# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from itertools import count
import numpy as np

import logging

_log = logging.getLogger(__name__)


def _append_tensor(tensor, append_low=None, append_high=None):
    if append_low is None:
        if append_high is None:
            return tensor
        dim_slices = [slice(0, dim_len) for dim_len in tensor.shape]
        new_shape = [
            dim_len + int(is_high)
            for dim_len, is_high in zip(tensor.shape, append_high)
        ]
    else:
        dim_slices = [
            slice(int(is_low), dim_len + int(is_low))
            for dim_len, is_low in zip(tensor.shape, append_low)
        ]
        if append_high is None:
            new_shape = [
                dim_len + int(is_low)
                for dim_len, is_low in zip(tensor.shape, append_low)
            ]
        else:
            new_shape = [
                dim_len + int(is_low) + int(is_high)
                for dim_len, is_low, is_high in zip(
                    tensor.shape, append_low, append_high
                )
            ]

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
        dim_slices = [
            slice(0, -1 if is_high else None)
            for dim_len, is_high in zip(tensor.shape, trim_high)
        ]
    else:
        if trim_high is None:
            dim_slices = [
                slice(int(is_low), None)
                for dim_len, is_low in zip(tensor.shape, trim_low)
            ]
        else:
            dim_slices = [
                slice(int(is_low), -1 if is_high else None)
                for dim_len, is_low, is_high in zip(tensor.shape, trim_low, trim_high)
            ]
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
    new_tensors = []
    for term_idx, feature_idxs in enumerate(term_features):
        higher = [
            feature_bin_weights[feature_idx][-1] == 0 for feature_idx in feature_idxs
        ]
        new_tensors.append(_append_tensor(tensors[term_idx], None, higher))
    return new_tensors


def remove_last(tensors, term_bin_weights):
    new_tensors = []
    for idx, tensor, weights in zip(count(), tensors, term_bin_weights):
        if tensor is None:
            result = None
        elif weights is None:
            result = tensor
        else:
            n_dimensions = weights.ndim
            entire_tensor = [slice(None)] * n_dimensions
            higher = []
            for dimension_idx in range(n_dimensions):
                dim_slices = entire_tensor.copy()
                dim_slices[dimension_idx] = -1
                total_sum = np.sum(weights[tuple(dim_slices)])
                higher.append(True if total_sum == 0 else False)
            result = trim_tensor(tensor, None, higher)
        new_tensors.append(result)
    return new_tensors


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


def restore_missing_value_zeros(tensor, weights):
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
