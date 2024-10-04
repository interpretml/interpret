# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from itertools import count

import numpy as np

_log = logging.getLogger(__name__)


def trim_tensor(tensor, trim_low=None, trim_high=None):
    if trim_low is None:
        if trim_high is None:
            return tensor
        dim_slices = [
            slice(0, -1 if is_high else None)
            for dim_len, is_high in zip(tensor.shape, trim_high)
        ]
    elif trim_high is None:
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
