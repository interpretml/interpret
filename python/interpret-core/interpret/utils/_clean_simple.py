# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import numpy.ma as ma
from sklearn.base import is_classifier, is_regressor

import logging

_log = logging.getLogger(__name__)

try:
    import pandas as pd

    _pandas_installed = True
except ImportError:
    _pandas_installed = False

try:
    import scipy as sp

    _scipy_installed = True
except ImportError:
    _scipy_installed = False


_none_list = [None]
_none_ndarray = np.array(None)


def _remove_extra_dimensions(arr):
    shape = arr.shape
    if len(shape) == 0 or 0 in shape:
        # 0 dimensional items exist, but are weird/unexpected. len fails, shape is length 0.
        # arrays with dimension having 0 length exist, but cannot contain anything
        return np.empty(0, arr.dtype)

    shape = [x for x in arr.shape if x != 1]
    if len(shape) == 0:
        shape = (1,)

    # reshape returns a view
    return arr.reshape(shape)


def clean_dimensions(data, param_name):
    # called under: fit

    if data is None:
        msg = f"{param_name} cannot be None"
        _log.error(msg)
        raise ValueError(msg)

    while True:
        if isinstance(data, ma.masked_array):
            # do this before np.ndarray since ma.masked_array is a subclass of np.ndarray
            mask = data.mask
            if mask is not ma.nomask:
                if mask.any():
                    msg = f"{param_name} cannot contain missing values"
                    _log.error(msg)
                    raise ValueError(msg)
            data = data.data
        elif isinstance(data, np.ndarray):
            pass
        elif _pandas_installed and isinstance(data, pd.Series):
            if data.hasnans:
                # if hasnans is true then there is definetly a real missing value in there and not just a mask
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
            # can be a non-numpy datatype, but has enough conformance for us to work on it
            data = data.values
        elif _pandas_installed and isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                data = data.iloc[:, 0]
                if data.hasnans:
                    # if hasnans is true then there is definetly a real missing value in there and not just a mask
                    msg = f"{param_name} cannot contain missing values"
                    _log.error(msg)
                    raise ValueError(msg)
                # can be a non-numpy datatype, but has enough conformance for us to work on it
                data = data.values
            else:
                # can be a non-numpy datatype, but has enough conformance for us to work on it
                data = data.astype(np.object_, copy=False).values
        elif _scipy_installed and isinstance(data, sp.sparse.spmatrix):
            data = data.toarray()
        elif isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data, np.object_)
        elif callable(getattr(data, "__array__", None)):
            data = data.__array__()
        elif isinstance(data, str):
            # we have just 1 item, so re-pack it and return
            ret = np.empty(1, np.object_)
            ret.itemset(0, data)
            return ret
        else:
            try:
                data = list(data)
            except TypeError:
                # we have just 1 item, so re-pack it and return
                ret = np.empty(1, np.object_)
                ret.itemset(0, data)
                return ret
            data = np.array(data, np.object_)

        data = _remove_extra_dimensions(data)

        if data.shape[0] == 0:
            # data.ndim must be 1
            return data

        if data.dtype.type is not np.object_:
            if 3 <= data.ndim:
                msg = f"{param_name} cannot have 3rd dimension"
                _log.error(msg)
                raise TypeError(msg)
            if issubclass(data.dtype.type, np.floating) and np.isnan(data).any():
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
            return data

        if data.shape[0] != 1:
            break

        data = data[0]

    n_second_dim = None

    # check the interior items
    idx = 0
    n = len(data)
    while idx < n:
        item = data[idx]

        if isinstance(item, str):
            if n_second_dim is not None and n_second_dim != 1:
                msg = (
                    f"{param_name} is not consistent in length for the second dimension"
                )
                _log.error(msg)
                raise TypeError(msg)
            n_second_dim = 1
            idx = idx + 1
            continue

        # TODO: if we checked item for various types like numpy, and those types were not of type np.object_
        # then we could avoid iterating the list contents below, or if the list contained only list
        # then sometimes we would not have to re-convert data to a list, which we need to do below incase
        # item is an iterator and therefore must modify data[idx] to insert the list we created from the iterator

        try:
            item = list(item)
        except TypeError:
            if n_second_dim is not None and n_second_dim != 1:
                msg = (
                    f"{param_name} is not consistent in length for the second dimension"
                )
                _log.error(msg)
                raise TypeError(msg)
            n_second_dim = 1
            idx = idx + 1
            continue

        if data is not list:
            data = list(data)

        n_items = len(item)
        if n_items == 1:
            # keep iterating down into them until they hit a non-1 length
            data[idx] = item[0]
            continue

        if n_second_dim is not None and n_second_dim != n_items:
            msg = f"{param_name} is not consistent in length for the second dimension"
            _log.error(msg)
            raise TypeError(msg)
        n_second_dim = n_items

        # now check if any of the sub-items are iterable
        sub_idx = 0
        while sub_idx < n_items:
            subitem = item[sub_idx]

            if isinstance(subitem, str):
                sub_idx = sub_idx + 1
                continue

            try:
                subitem = list(subitem)
            except TypeError:
                sub_idx = sub_idx + 1
                continue

            if len(subitem) != 1:
                msg = f"{param_name} cannot have 3rd dimension"
                _log.error(msg)
                raise TypeError(msg)

            # keep iterating down into them until they hit a non-1 length
            item[sub_idx] = subitem[0]

        # if it was an iterable or we dug into any of the items, we need to replace it
        data[idx] = item
        idx = idx + 1

    if n_second_dim == 0:
        return np.empty(0, np.object_)

    data = np.array(data, np.object_, copy=False)  # in case it was converted to list

    if _pandas_installed:
        # pandas also has the pd.NA value that indicates missing.  If Pandas is available though
        # we can use it's function that checks for pd.NA, np.nan, and None
        if pd.isna(data).any():
            msg = f"{param_name} cannot contain missing values"
            _log.error(msg)
            raise ValueError(msg)
    else:
        # data != data is a check for nan that works even with mixed types, since nan != nan
        if (data == _none_ndarray).any() or (data != data).any():
            msg = f"{param_name} cannot contain missing values"
            _log.error(msg)
            raise ValueError(msg)

    return data


def typify_classification(vec):
    # Per scikit-learn, we need to accept y of list or numpy array that contains either strings or integers.
    # We want to serialize these models to/from JSON, and JSON allows us to differentiate between string
    # and integer types with just the JSON type, so that's nice.  JSON also allows boolean types,
    # and that seems like a type someone might pass us for binary classification, so accept bools too.
    # https://scikit-learn.org/stable/developers/develop.html

    if issubclass(vec.dtype.type, np.integer):
        # this also handles pandas Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype
        # JSON has a number datatype, so we can preserve this information in JSON!
        dtype = np.int64
    elif issubclass(vec.dtype.type, np.bool_):
        # this also handles pandas BooleanDtype
        # JSON has a boolean datatype, so we can preserve this information in JSON!
        dtype = np.bool_
    elif issubclass(vec.dtype.type, np.object_):
        types = set(map(type, vec))
        if all(
            one_type is int or issubclass(one_type, np.integer) for one_type in types
        ):
            # the vec.astype call below can fail if we're passed an unsigned np.uint64
            # array with big values, but we don't want to surprise anyone by converting to
            # strings in that special case, so throw if we're presented this unusual type
            dtype = np.int64
        elif all(
            one_type is bool or issubclass(one_type, np.bool_) for one_type in types
        ):
            dtype = np.bool_
        else:
            dtype = np.unicode_
    else:
        dtype = np.unicode_

    return vec.astype(dtype, copy=False)


def clean_init_score(init_score, n_samples, X_unclean):
    if is_classifier(init_score):
        # scikit-learn estimators either have predict_proba or decision_function
        # first try predict_proba since we can more reliably detect mono-classification
        try:
            probs = clean_dimensions(init_score.predict_proba(X_unclean), "init_score")
            if n_samples == 1:  # then the sample dimension would have been eliminated
                if probs.ndim != 1:
                    msg = "init_score.predict_proba(X) returned inconsistent number of dimensions"
                    _log.error(msg)
                    raise ValueError(msg)
                if probs.shape[0] <= 1:  # 0 or 1 means 1 class
                    # only 1 class to predict means perfect prediction, and no scores for EBMs
                    # do not check if probs are all one in case there is floating point noise
                    return np.empty((1, 0), np.float64)
                probs = probs.reshape([1] + probs.shape)
            else:
                if probs.shape[0] == 0:
                    # having any dimension as zero length probably means 1 class, so treat it that way
                    return np.empty((n_samples, 0), np.float64)
                if probs.shape[0] != n_samples:
                    msg = "init_score.predict_proba(X) returned inconsistent number of samples compared to y"
                    _log.error(msg)
                    raise ValueError(msg)
                if probs.ndim == 1:
                    # only 1 class to predict means perfect prediction, and no scores for EBMs
                    # do not check if probs are all one in case there is floating point noise
                    return np.empty((n_samples, 0), np.float64)
            probs = probs.astype(np.float64, copy=False)
            maxes = np.amax(probs, axis=1)
            with np.errstate(divide="ignore"):
                init_score = np.log(probs / maxes[:, np.newaxis])
            if init_score.shape[1] == 2:  # binary classification
                init_score = init_score[:, 1] - init_score[:, 0]
            return init_score
        except AttributeError:
            init_score = clean_dimensions(
                init_score.decision_function(X_unclean), "init_score"
            )
            if n_samples == 1:  # then the sample dimension would have been eliminated
                if init_score.ndim != 1:
                    msg = "init_score.decision_function(X) returned inconsistent number of dimensions"
                    _log.error(msg)
                    raise ValueError(msg)
                if init_score.shape[0] != 1:
                    init_score = init_score.reshape([1] + init_score.shape)
            else:
                if init_score.shape[0] == 0:
                    # must be a 1 class problem
                    return np.empty((n_samples, 0), np.float64)
                if init_score.shape[0] != n_samples:
                    msg = "init_score.decision_function(X) returned inconsistent number of samples compared to y"
                    _log.error(msg)
                    raise ValueError(msg)
            init_score = init_score.astype(np.float64, copy=False)
            return init_score
    elif is_regressor(init_score):
        init_score = clean_dimensions(init_score.predict(X_unclean), "init_score")
        if init_score.ndim != 1:
            msg = "init_score.predict(X) must have only 1 dimension"
            _log.error(msg)
            raise ValueError(msg)
        if init_score.shape[0] != n_samples:
            msg = "init_score.predict(X) returned inconsistent number of samples compared to y"
            _log.error(msg)
            raise ValueError(msg)
        init_score = init_score.astype(np.float64, copy=False)

        # TODO Add link function to operate on predict's output when needed

        return init_score

    init_score = clean_dimensions(init_score, "init_score")
    if n_samples == 1:  # then the sample dimension would have been eliminated
        if init_score.ndim != 1:
            msg = "init_score has an inconsistent number of samples compared to X"
            _log.error(msg)
            raise ValueError(msg)
        if init_score.shape[0] != 1:
            init_score = init_score.reshape([1] + init_score.shape)
    else:
        if init_score.shape[0] == 0:
            # must be a 1 class problem
            return np.empty((n_samples, 0), np.float64)
        if init_score.shape[0] != n_samples:
            msg = "init_score has an inconsistent number of samples compared to X"
            _log.error(msg)
            raise ValueError(msg)
    init_score = init_score.astype(np.float64, copy=False)
    return init_score
