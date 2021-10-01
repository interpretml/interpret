# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from collections import Counter
from itertools import count, repeat
from multiprocessing.sharedctypes import RawArray
import numpy as np
import numpy.ma as ma

from .internal import Native

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

_disallowed_types = frozenset([complex, list, tuple, range, bytes, bytearray, memoryview, set, frozenset, dict, Ellipsis, np.csingle, np.complex_, np.clongfloat, np.void])
_none_list = [None]
_none_ndarray = np.array(None)

def _densify_object_ndarray(X_col):
    # called under: fit or predict

    # numpy hierarchy of types
    # https://numpy.org/doc/stable/reference/arrays.scalars.html

    # TODO: add special case handling if there is only 1 sample to make that faster

    types = set(map(type, X_col))
    if len(types) == 1:
        if str in types:
            return X_col.astype(np.unicode_)
        elif bool in types:
            return X_col.astype(np.bool_)

    if all(one_type is int or issubclass(one_type, np.integer) for one_type in types):
        if all(issubclass(one_type, np.unsignedinteger) for one_type in types):
            if all(one_type is np.uint8 for one_type in types):
                return X_col.astype(np.uint8)
            types.discard(np.uint8)

            if all(one_type is np.uint16 for one_type in types):
                return X_col.astype(np.uint16)
            types.discard(np.uint16)

            if all(one_type is np.uint32 for one_type in types):
                return X_col.astype(np.uint32)

            return X_col.astype(np.uint64)

        if all(one_type is np.int8 for one_type in types):
            return X_col.astype(np.int8)
        types.discard(np.int8)

        if all(one_type is np.uint8 or one_type is np.int16 for one_type in types):
            return X_col.astype(np.int16)
        types.discard(np.uint8)
        types.discard(np.int16)

        if all(one_type is np.uint16 or one_type is np.int32 for one_type in types):
            return X_col.astype(np.int32)

        try:
            return X_col.astype(np.int64)
        except OverflowError:
            # we must have a big number that can only be represented by np.uint64 AND also signed integers mixed together
            # if we do X_col.astype(np.uint64), it will silently convert negative integers to unsigned!

            # TODO : should this be np.float64 with a check for big integers
            return X_col.astype(np.unicode_)

    if all(one_type is float or issubclass(one_type, np.floating) for one_type in types):
        if all(one_type is np.float16 for one_type in types):
            return X_col.astype(np.float16)
        types.discard(np.float16)

        if all(one_type is np.float32 for one_type in types):
            return X_col.astype(np.float32)

        return X_col.astype(np.float64)

    # TODO: also check for bool conversion since "False"/"True" strings don't later convert to 'continuous'
    is_float_conversion = False
    for one_type in types:
        if one_type is str:
            pass # str objects have __iter__, so special case this to allow
        elif one_type is int:
            pass # int objects use the default __str__ function, so special case this to allow
        elif one_type is float:
            is_float_conversion = True # force to np.float64 to guarantee consistent string formatting
        elif issubclass(one_type, np.generic):
            # numpy objects have __getitem__, so special case this to allow
            if one_type is np.float64:
                pass # np.float64 is what we convert to for floats, so no need to convert this
            elif issubclass(one_type, np.floating):
                is_float_conversion = True # force to np.float64 to ensure consistent string formatting of floats
        elif one_type in _disallowed_types:
            # list of python types primarily from: https://docs.python.org/3/library/stdtypes.html
            msg = f"X contains the disallowed type {one_type}"
            _log.error(msg)
            raise ValueError(msg)
        elif hasattr(one_type, '__iter__') or hasattr(one_type, '__getitem__'):
            # check for __iter__ and __getitem__ to filter out iterables
            # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
            msg = f"X contains the disallowed iterable type {one_type}"
            _log.error(msg)
            raise ValueError(msg)
        elif hasattr(one_type, '__contains__'):
            msg = f"X contains the disallowed set type {one_type}"
            _log.error(msg)
            raise ValueError(msg)
        elif one_type.__str__ is object.__str__:
            # if any object in our list uses the default object __str__ function then it'll
            # include the id(val) pointer in the string text, which isn't going to be useful as a categorical

            # use type(val) instead of val.__str__ to detect inherited __str__ functions per:
            # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object

            msg = f"X contains the type {one_type} which does not define a __str__ function"
            _log.error(msg)
            raise ValueError(msg)

    if is_float_conversion:
        # TODO: handle ints here too which need to be checked if they are larger than the safe int max value

        X_col = X_col.copy()
        places = np.fromiter((val_type is float or issubclass(val_type, np.floating) for val_type in map(type, X_col)), dtype=np.bool_, count=len(X_col))
        np.place(X_col, places, X_col[places].astype(np.float64))

    return X_col.astype(np.unicode_)

def _process_column_initial(X_col, nonmissings, processing, min_unique_continuous):
    # called under: fit

    if issubclass(X_col.dtype.type, np.floating):
        missings = np.isnan(X_col)
        if missings.any():
            nonmissings = ~missings
            X_col = X_col[nonmissings]
    elif X_col.dtype.type is np.object_:
        X_col = _densify_object_ndarray(X_col)

    uniques, indexes, counts = np.unique(X_col, return_inverse=True, return_counts=True)

    if issubclass(uniques.dtype.type, np.floating):
        floats = uniques.astype(np.float64, copy=False)
        uniques = floats.astype(np.unicode_)
    else:
        uniques = uniques.astype(np.unicode_, copy=False)
        try:
            # we rely here on there being a round trip format within this language from float64 to text to float64

            # TODO: does this work if there are spaces or bools?

            floats = uniques.astype(dtype=np.float64)  
        except ValueError:
            floats = None

    if min_unique_continuous is not None and floats is not None:
        # floats can have more than one string representation, so run unique again to check if we have 
        # min_unique_continuous unique float64s in binary representation
        if min_unique_continuous <= len(np.unique(floats)):
            floats = floats[indexes] # expand from the unique floats to expanded floats
            if nonmissings is not None:
                floats_tmp = np.full(len(nonmissings), np.nan, dtype=np.float64)
                np.place(floats_tmp, nonmissings, floats)
                floats = floats_tmp

            return floats, None

    if processing == 'nominal_prevalence':
        if floats is None:
            categories = [(-item[0], item[1]) for item in zip(counts, uniques)]
        else:
            categories = [(-item[0], item[1], item[2]) for item in zip(counts, floats, uniques)]
        categories.sort()
        categories = [x[-1] for x in categories]
    elif processing == 'nominal_prevalence_reversed':
        if floats is None:
            categories = [(-item[0], item[1]) for item in zip(counts, uniques)]
        else:
            categories = [(-item[0], item[1], item[2]) for item in zip(counts, floats, uniques)]
        categories.sort(reverse=True)
        categories = [x[-1] for x in categories]
    elif processing == 'nominal_alphabetical':
        categories = uniques.tolist()
        categories.sort()
    elif processing == 'nominal_alphabetical_reversed':
        categories = uniques.tolist()
        categories.sort(reverse=True)
    elif processing == 'nominal_numerical_strict':
        if floats is None:
            msg = f"could not sort nominal_numerical_strict type by numeric value"
            _log.error(msg)
            raise ValueError(msg)
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        categories.sort()
        categories = [x[1] for x in categories]
    elif processing == 'nominal_numerical_strict_reversed':
        if floats is None:
            msg = f"could not sort nominal_numerical_strict_reversed type by numeric value"
            _log.error(msg)
            raise ValueError(msg)
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        categories.sort(reverse=True)
        categories = [x[1] for x in categories]
    elif floats is not None:
        # 'nominal_numerical_permissive' or 'nominal_numerical_permissive_reversed'
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        is_reversed = processing == 'nominal_numerical_permissive_reversed'
        categories.sort(reverse=is_reversed)
        categories = [x[1] for x in categories]
    else:
        # default to same as 'nominal_alphabetical'
        categories = uniques.tolist()
        categories.sort()

    categories = dict(zip(categories, count(1)))
    mapping = np.fromiter((categories[val] for val in uniques), dtype=np.int64, count=len(uniques))
    encoded = mapping[indexes]

    if nonmissings is not None:
        encoded_tmp = np.zeros(len(nonmissings), dtype=np.int64)
        np.place(encoded_tmp, nonmissings, encoded)
        encoded = encoded_tmp

    return encoded, categories

def _encode_categorical_existing(X_col, nonmissings, categories):
    # called under: predict

    # TODO: add special case handling if there is only 1 sample to make that faster

    if issubclass(X_col.dtype.type, np.floating):
        missings = np.isnan(X_col)
        if missings.any():
            nonmissings = ~missings
            X_col = X_col[nonmissings]
    elif X_col.dtype.type is np.object_:
        X_col = _densify_object_ndarray(X_col)

    uniques, indexes = np.unique(X_col, return_inverse=True)

    if issubclass(X_col.dtype.type, np.floating):
        uniques = uniques.astype(np.float64, copy=False)
    uniques = uniques.astype(np.unicode_, copy=False)

    mapping = np.fromiter((categories.get(val, -1) for val in uniques), dtype=np.int64, count=len(uniques))
    encoded = mapping[indexes]

    if (mapping < 0).any():
        if nonmissings is not None:
            encoded_tmp = np.zeros(len(nonmissings), dtype=np.int64)
            np.place(encoded_tmp, nonmissings, encoded)
            bad = np.full(len(nonmissings), None, dtype=np.object_)
            np.place(bad, encoded_tmp < 0, uniques[indexes[encoded < 0]])
            encoded = encoded_tmp
        else:
            bad = np.full(len(encoded), None, dtype=np.object_)
            np.place(bad, unknowns, uniques[indexes[encoded < 0]])
    else:
        bad = None
        if nonmissings is not None:
            encoded_tmp = np.zeros(len(nonmissings), dtype=np.int64)
            np.place(encoded_tmp, nonmissings, encoded)
            encoded = encoded_tmp

    return encoded, bad

def _encode_pandas_categorical_initial(X_col, pd_categories, is_ordered, processing):
    # called under: fit

    if processing == 'nominal':
        if is_ordered:
            msg = "nominal type invalid for ordered pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
    elif processing == 'ordinal':
        if not is_ordered:
            msg = "ordinal type invalid for unordered pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
    elif processing is None or processing == 'auto':
        pass
    elif processing == 'nominal_prevalence' or processing == 'nominal_prevalence_reversed' or processing == 'nominal_alphabetical' or processing == 'nominal_alphabetical_reversed' or processing == 'nominal_numerical_strict' or processing == 'nominal_numerical_strict_reversed' or processing == 'nominal_numerical_permissive' or processing == 'nominal_numerical_permissive_reversed':
        # TODO: we could instead handle this by re-ordering the pandas pd_categories.  Someone might want to construct it quickly but then override the pd_categories
        msg = f"{processing} type invalid for pandas.CategoricalDtype"
        _log.error(msg)
        raise ValueError(msg)
    else:
        if isinstance(processing, str):
            # don't allow strings to get to the for loop below
            msg = f"{processing} type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)

        n_items = 0
        n_ordinals = 0
        n_continuous = 0
        try:
            for item in processing:
                n_items += 1
                if isinstance(item, str):
                    n_ordinals += 1
                elif isinstance(item, float) or isinstance(item, int) or isinstance(item, np.floating) or isinstance(item, np.integer):
                    n_continuous += 1
        except TypeError:
            msg = f"{processing} type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)

        if n_continuous == n_items:
            msg = "continuous type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
        elif n_ordinals == n_items:
            if not is_ordered:
                msg = "ordinal type invalid for unordered pandas.CategoricalDtype"
                _log.error(msg)
                raise ValueError(msg)

            # TODO: instead of throwing, we could match the ordinal values with the pandas pd_categories and
            # report the rest as bad items.  For now though, just assume it's bad to specify this
            msg = "cannot specify ordinal categories for a pandas.CategoricalDtype which already has categories"
            _log.error(msg)
            raise ValueError(msg)
        else:
            msg = f"{processing} type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)

    categories = dict(zip(pd_categories, count(1)))
    X_col = X_col.astype(dtype=np.int64, copy=False) # we'll need int64 for calling C++ anyways
    X_col = X_col + 1
    return X_col, categories

def _encode_pandas_categorical_existing(X_col, pd_categories, categories):
    # called under: predict

    # TODO: add special case handling if there is only 1 sample to make that faster

    mapping = np.fromiter((categories.get(val, -1) for val in pd_categories), dtype=np.int64, count=len(pd_categories))

    if len(mapping) <= len(categories):
        mapping_cmp = np.arange(1, len(mapping) + 1, dtype=np.int64)
        if np.array_equal(mapping, mapping_cmp):
            X_col = X_col.astype(dtype=np.int64, copy=False) # avoid overflows for np.int8
            X_col = X_col + 1
            return X_col, None
    else:
        mapping_cmp = np.arange(1, len(categories) + 1, dtype=np.int64)
        if np.array_equal(mapping[0:len(mapping_cmp)], mapping_cmp):
            unknowns = len(categories) <= X_col
            bad = np.full(len(X_col), None, dtype=np.object_)
            bad[unknowns] = pd_categories[X_col[unknowns]]
            X_col = X_col.astype(dtype=np.int64, copy=False) # avoid overflows for np.int8
            X_col = X_col + 1
            X_col[unknowns] = -1
            return X_col, bad

    mapping = np.insert(mapping, 0, 0)
    encoded = mapping[X_col + 1]

    bad = None
    unknowns = encoded < 0
    if unknowns.any():
        bad = np.full(len(X_col), None, dtype=np.object_)
        bad[unknowns] = pd_categories[X_col[unknowns]]

    return encoded, bad

def _process_continuous(X_col, nonmissings):
    # called under: fit or predict

    # TODO: add special case handling if there is only 1 sample to make that faster

    if issubclass(X_col.dtype.type, np.floating):
        X_col = X_col.astype(dtype=np.float64, copy=False)
        return X_col, None
    elif issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_:
        X_col = X_col.astype(dtype=np.float64)
        if nonmissings is not None:
            X_col_tmp = np.full(len(nonmissings), np.nan, dtype=np.float64)
            np.place(X_col_tmp, nonmissings, X_col)
            X_col = X_col_tmp

        return X_col, None
    else:
        # we either have an np.object_ or np.unicode_/np.str_
        try:
            floats = X_col.astype(dtype=np.float64)
            bad = None
        except (TypeError, ValueError):
            # we get a TypeError whenever we have an np.object_ array and numpy attempts to call float(), but the 
            # object doesn't have a __float__ function.  We get a ValueError when either a str object inside an 
            # np.object_ array or when an np.unicode_ array attempts to convert a string to a float and fails

            n_samples = len(X_col)
            bad = np.full(n_samples, None, dtype=np.object_)
            floats = np.zeros(n_samples, dtype=np.float64)
            for idx in range(n_samples):
                one_item_array = X_col[idx:idx + 1] # slice one item at a time keeping as an np.ndarray
                try:
                    # use .astype(..) instead of float(..) to ensure identical conversion results
                    floats[idx] = one_item_array.astype(dtype=np.float64)
                except TypeError:
                    # use .astype instead of str(one_item_array) here to ensure identical string categories
                    one_str_array = one_item_array.astype(dtype=np.unicode_)
                    try:
                        # use .astype(..) instead of float(..) to ensure identical conversion results
                        floats[idx] = one_str_array.astype(dtype=np.float64)
                    except ValueError:
                        bad.itemset(idx, one_str_array.item())
                except ValueError:
                    bad.itemset(idx, one_item_array.item())

            # bad.any() would fail to work if bad was allowed to be either None or False, but None
            # values in X_col should always be identified as missing by our caller, and False should be successfully 
            # converted to 0.0 above, so neither should end up in the bad array other than non-bad indicators
            bad = bad if bad.any() else None

        if nonmissings is not None:
            floats_tmp = np.full(len(nonmissings), np.nan, dtype=np.float64)
            np.place(floats_tmp, nonmissings, floats)
            floats = floats_tmp

            if bad is not None:
                bad_tmp = np.full(len(nonmissings), None, dtype=np.object_)
                np.place(bad_tmp, nonmissings, bad)
                bad = bad_tmp

        return floats, bad

def _process_ndarray(X_col, nonmissings, categories, processing, min_unique_continuous):
    if processing == 'continuous':
        # called under: fit or predict
        X_col, bad = _process_continuous(X_col, nonmissings)
        return 'continuous', X_col, None, bad
    elif processing == 'nominal':
        if categories is None:
            # called under: fit
            X_col, categories = _process_column_initial(X_col, nonmissings, None, None)
            return 'nominal', X_col, categories, None
        else:
            # called under: predict
            X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
            return 'nominal', X_col, categories, bad
    elif processing == 'ordinal':
        if categories is None:
            # called under: fit
            # It's an error since we need to also provide the ordinal definition during fit
            msg = "ordinal category definition missing for ordinal type"
            _log.error(msg)
            raise ValueError(msg)
        else:
            # called under: predict
            X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
            return 'ordinal', X_col, categories, bad
    elif processing is None or processing == 'auto':
        # called under: fit
        X_col, categories = _process_column_initial(X_col, nonmissings, None, min_unique_continuous)
        return 'continuous' if categories is None else 'nominal', X_col, categories, None
    elif processing == 'nominal_prevalence' or processing == 'nominal_prevalence_reversed' or processing == 'nominal_alphabetical' or processing == 'nominal_alphabetical_reversed' or processing == 'nominal_numerical_strict' or processing == 'nominal_numerical_strict_reversed' or processing == 'nominal_numerical_permissive' or processing == 'nominal_numerical_permissive_reversed':
        # called under: fit
        X_col, categories = _process_column_initial(X_col, nonmissings, processing, None)
        return 'nominal', X_col, categories, None
    elif processing == 'quantile' or processing == 'quantile_humanized' or processing == 'uniform' or processing == 'winsorized':
        # called under: fit
        X_col, bad = _process_continuous(X_col, nonmissings)
        return 'continuous', X_col, None, bad
    elif isinstance(processing, int):
        # called under: fit
        X_col, categories = _process_column_initial(X_col, nonmissings, None, processing)
        return 'continuous' if categories is None else 'nominal', X_col, categories, None
    elif processing == 'ignore':
        # called under: fit or predict
        X_col, categories = _process_column_initial(X_col, nonmissings, None, None)
        mapping = np.empty(len(categories) + 1, dtype=np.object_)
        mapping.itemset(0, None)
        for category, idx in categories.items():
            mapping.itemset(idx, category)
        bad = mapping[X_col]
        return 'ignore', None, None, bad
    elif isinstance(processing, str):
        # called under: fit

        # don't allow strings to get to the np.array conversion below
        msg = f"{processing} type invalid"
        _log.error(msg)
        raise ValueError(msg)
    else:
        # called under: fit

        n_items = 0
        n_ordinals = 0
        n_continuous = 0
        try:
            for item in processing:
                n_items += 1
                if isinstance(item, str):
                    n_ordinals += 1
                elif isinstance(item, float) or isinstance(item, int) or isinstance(item, np.floating) or isinstance(item, np.integer):
                    n_continuous += 1
        except TypeError:
            msg = f"{processing} type invalid"
            _log.error(msg)
            raise ValueError(msg)

        if n_continuous == n_items:
            # if n_items == 0 then it must be continuous since we can have zero cut points, but not zero ordinal categories
            X_col, bad = _process_continuous(X_col, nonmissings)
            return 'continuous', X_col, None, bad
        elif n_ordinals == n_items:
            categories = dict(zip(processing, count(1)))
            X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
            return 'ordinal', X_col, categories, bad
        else:
            msg = f"{processing} type invalid"
            _log.error(msg)
            raise ValueError(msg)

def _reshape_1D_if_possible(col):
    if col.ndim != 1:
        # ignore dimensions that have just 1 item and assume the intent was to give us 1D
        is_found = False
        for n_items in col.shape:
            if n_items != 1:
                if is_found:
                    msg = f"Cannot reshape to 1D. Original shape was {col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
                is_found = True
        col = col.reshape(-1)
    return col

def _process_numpy_column(X_col, categories, feature_type, min_unique_continuous):
    nonmissings = None

    if isinstance(X_col, ma.masked_array):
        mask = X_col.mask
        if mask is ma.nomask:
            X_col = X_col.data
        else:
            X_col = X_col.compressed()
            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
            if len(X_col) != len(mask):
                nonmissings = ~mask

    if X_col.dtype.type is np.object_:
        if _pandas_installed:
            # pandas also has the pd.NA value that indicates missing.  If Pandas is available though
            # we can use it's function that checks for pd.NA, np.nan, and None
            nonmissings2 = pd.notna(X_col)
        else:
            # X_col == X_col is a check for nan that works even with mixed types, since nan != nan
            nonmissings2 = np.logical_and(X_col != _none_ndarray, X_col == X_col)
        if not nonmissings2.all():
            X_col = X_col[nonmissings2]
            if nonmissings is None:
                nonmissings = nonmissings2
            else:
                # it's a little weird and possibly dangerous to place inside the array being read,
                # but algorithmically this is the fastest thing to do, and it seems to work..
                np.place(nonmissings, nonmissings, nonmissings2)

    return _process_ndarray(X_col, nonmissings, categories, feature_type, min_unique_continuous)

def _process_pandas_column(X_col, categories, feature_type, min_unique_continuous):
    if isinstance(X_col.dtype, pd.CategoricalDtype):
        # unlike other missing value types, we get back -1's for missing here, so no need to drop them
        X_col = X_col.values
        is_ordered = X_col.ordered
        pd_categories = X_col.categories.values.astype(dtype=np.unicode_, copy=False)
        X_col = X_col.codes

        if feature_type == 'ignore':
            pd_categories = pd_categories.astype(dtype=np.object_)
            pd_categories = np.insert(pd_categories, 0, None)
            bad = pd_categories[X_col + 1]
            return None, None, bad, 'ignore'
        else:
            if categories is None:
                # called under: fit
                X_col, categories = _encode_pandas_categorical_initial(X_col, pd_categories, is_ordered, feature_type)
                bad = None
            else:
                # called under: predict
                X_col, bad = _encode_pandas_categorical_existing(X_col, pd_categories, categories)

            return 'ordinal' if is_ordered else 'nominal', X_col, categories, bad
    elif issubclass(X_col.dtype.type, np.floating):
        X_col = X_col.values
        return _process_ndarray(X_col, None, categories, feature_type, min_unique_continuous)
    elif issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_ or X_col.dtype.type is np.unicode_ or X_col.dtype.type is np.object_:
        # this also handles Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype, and BooleanDtype
        nonmissings = None
        if X_col.hasnans:
            # if hasnans is true then there is definetly a real missing value in there and not just a mask
            nonmissings = X_col.notna().values
            X_col = X_col.dropna()
        X_col = X_col.values
        X_col = X_col.astype(dtype=X_col.dtype.type, copy=False)
        return _process_ndarray(X_col, nonmissings, categories, feature_type, min_unique_continuous)
    else:
        # TODO: implement pd.SparseDtype
        # TODO: implement pd.StringDtype both the numpy and arrow versions
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.StringDtype.html#pandas.StringDtype
        msg = f"{type(X_col.dtype)} not supported"
        _log.error(msg)
        raise ValueError(msg)

def _process_scipy_column(X_col, categories, feature_type, min_unique_continuous):
    X_col = X_col.toarray().reshape(-1)

    nonmissings = None
    if X_col.dtype.type is np.object_:
        if _pandas_installed:
            # pandas also has the pd.NA value that indicates missing.  If Pandas is available though
            # we can use it's function that checks for pd.NA, np.nan, and None
            nonmissings = pd.notna(X_col)
        else:
            # X_col == X_col is a check for nan that works even with mixed types, since nan != nan
            nonmissings = np.logical_and(X_col != _none_ndarray, X_col == X_col)

        if nonmissings.all():
            nonmissings = None
        else:
            X_col = X_col[nonmissings]

    return _process_ndarray(X_col, nonmissings, categories, feature_type, min_unique_continuous)

def _process_dict_column(X_col, categories, feature_type, min_unique_continuous):
    if isinstance(X_col, np.ndarray): # this includes ma.masked_array
        pass
    elif _pandas_installed and isinstance(X_col, pd.Series):
        return _process_pandas_column(X_col, categories, feature_type, min_unique_continuous)
    elif _pandas_installed and isinstance(X_col, pd.DataFrame):
        if X_col.shape[1] == 1:
            X_col = X_col.iloc[:, 0]
            return _process_pandas_column(X_col, categories, feature_type, min_unique_continuous)
        elif data.shape[0] == 1:
            X_col = X_col.astype(np.object_, copy=False).values.reshape(-1)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif _scipy_installed and isinstance(X_col, sp.sparse.spmatrix):
        if X_col.shape[1] == 1 or X_col.shape[0] == 1:
            return _process_scipy_column(X_col, categories, feature_type, min_unique_continuous)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(X_col, list) or isinstance(X_col, tuple):
        X_col = np.array(X_col, dtype=np.object_)
    elif isinstance(X_col, str):
        # don't allow strings to get to the np.array conversion below
        X_col_tmp = np.empty(shape=1, dtype=np.object_)
        X_col_tmp.itemset(0, X_col)
        X_col = X_col_tmp
    else:
        try:
            # we don't support iterables that get exhausted on their first examination.  This condition
            # should be detected though in clean_X where we get the length or bin_native where we check the
            # number of samples on the 2nd run through the generator
            X_col = list(X_col)
            X_col = np.array(X_col, dtype=np.object_)
        except TypeError:
            # if our item isn't iterable, assume it has just 1 item and we'll check below if that's consistent
            X_col_tmp = np.empty(shape=1, dtype=np.object_)
            X_col_tmp.itemset(0, X_col)
            X_col = X_col_tmp

    X_col = _reshape_1D_if_possible(X_col)
    return _process_numpy_column(X_col, categories, feature_type, min_unique_continuous)

def unify_columns(X, requests, feature_names_out, feature_types=None, min_unique_continuous=4, go_fast=False):
    # If the requests paramter contains a categories dictionary, then that same categories object is guaranteed to
    # be yielded back to the caller.  This guarantee can be used to rapidly identify which request is being 
    # yielded by using the id(categories) along with the feature_idx

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        if issubclass(X.dtype.type, np.complexfloating):
            msg = "X contains complex numbers, which are not a supported dtype"
            _log.error(msg)
            raise ValueError(msg)
        elif issubclass(X.dtype.type, np.void):
            msg = "X contains numpy.void data, which are not a supported dtype"
            _log.error(msg)
            raise ValueError(msg)

        # TODO: in the future special case this to make single samples faster at predict time

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        elif X.ndim != 2:
            msg = f"X cannot have {X.ndim} dimensions"
            _log.error(msg)
            raise ValueError(msg)

        n_cols = X.shape[1]
        col_map = None
        if n_cols != len(feature_names_out):
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
            # feature_types to not be None.  During predict time feature_types_out cannot be None, but we need 
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter((val != 'ignore' for val in feature_types), dtype=np.bool_, count=len(feature_types))
            if n_cols != keep_cols.sum():
                # called under: predict
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), dtype=np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

        # TODO: create a C++ transposer that takes the stride length between items, so we can pass in 1 for bytes
        # 2 for int16, 4 for int32, 8 for int64 and special case those sizes to be fast.  We can then also transpose
        # np.object_ and np.unicode by passing in whatever lengths those are, which we can get from numpy reliably
        # Inisde C++ we can use a templated function that takes the stride length or 0, so we'll get compiled
        # versions that specialize the 1,2,4,8 sizes, and use memcpy to make the cell copies.  memcpy is an
        # intrinsic that'll optimize down to avoid loops when possible, so that should give us fast results.
        #
        # For some reason numpy really sucks at transposing data and asfortranarray makes it slower, so let's do it ourselves.
        # Allocate an empty fortran array here in python and have C++ fill it.  Then we can keep all the
        # rest of the code below the same since it'll just be accessed internally more efficiently.
        #if go_fast and X.flags.c_contiguous:
        #    # called under: predict
        #    # during predict we don't care as much about memory consumption, so speed it by transposing everything
        #    X = np.asfortranarray(X)

        for feature_idx, categories in requests:
            col_idx = feature_idx if col_map is None else col_map[feature_idx]
            X_col = X[:, col_idx]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_out, X_col, categories, bad = _process_numpy_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_idx, feature_type_out, X_col, categories, bad
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        names_original = X.columns
        names_dict = dict(zip(map(str, names_original), count()))
        n_cols = len(names_original)
        if len(names_dict) != n_cols:
            # this can happen if for instance one column is "0" and annother is int(0)
            # Pandas also allows duplicate labels by default:
            # https://pandas.pydata.org/docs/user_guide/duplicates.html#duplicates-disallow
            # we can tollerate duplicate labels here, provided none of them are being used by our model
            for name, n_count in Counter(map(str, names_original)).items():
                if n_count != 1:
                    names_dict.remove(name)

        if feature_types is None:
            if any(feature_name_out not in names_dict for feature_name_out in feature_names_out):
               names_dict = None
        else:
            if any(feature_name_out not in names_dict for feature_name_out, feature_type in zip(feature_names_out, feature_types) if feature_type != 'ignore'):
               names_dict = None

        if names_dict is None:
            if n_cols == len(feature_names_out):
                names_dict = dict(zip(feature_names_out, count()))
            else:
                # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
                # feature_types to not be None.  During predict time feature_types_out cannot be None, but we need 
                # to check for legality on the dimensions of X
                names_dict = dict(zip((feature_name_out for feature_name_out, feature_type in zip(feature_names_out, feature_types) if feature_type != 'ignore'), count()))
                if n_cols != len(names_dict):
                    msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                    _log.error(msg)
                    raise ValueError(msg)

        # Pandas also sometimes uses a dense 2D ndarray instead of per column 1D ndarrays, which would benefit from 
        # transposing, but accessing the BlockManager is currently unsupported behavior. They are also planning to eliminate
        # the BlockManager in Pandas2, so not much benefit in special casing this while they move in that direction
        # https://uwekorn.com/2020/05/24/the-one-pandas-internal.html

        for feature_idx, categories in requests:
            col_idx = names_dict[feature_names_out[feature_idx]]
            X_col = X.iloc[:, col_idx]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_out, X_col, categories, bad = _process_pandas_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_idx, feature_type_out, X_col, categories, bad
    elif _scipy_installed and isinstance(X, sp.sparse.spmatrix):
        n_cols = X.shape[1]

        col_map = None
        if n_cols != len(feature_names_out):
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
            # feature_types to not be None.  During predict time feature_types_out cannot be None, but we need 
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter((val != 'ignore' for val in feature_types), dtype=np.bool_, count=len(feature_types))
            if n_cols != keep_cols.sum():
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), dtype=np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

        for feature_idx, categories in requests:
            col_idx = feature_idx if col_map is None else col_map[feature_idx]
            X_col = X.getcol(col_idx)
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_out, X_col, categories, bad = _process_scipy_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_idx, feature_type_out, X_col, categories, bad
    elif isinstance(X, dict):
        for feature_idx, categories in requests:
            X_col = X[feature_names_out[feature_idx]]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_out, X_col, categories, bad = _process_dict_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_idx, feature_type_out, X_col, categories, bad
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)

def unify_feature_names(X, feature_names=None, feature_types=None):
    # called under: fit

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        X_names = None
        n_cols = X.shape[0] if X.ndim == 1 else X.shape[1]
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        X_names = list(map(str, X.columns))
        n_cols = len(X_names)
    elif _scipy_installed and isinstance(X, sp.sparse.spmatrix):
        X_names = None
        n_cols = X.shape[1]
    elif isinstance(X, dict):
        X_names = list(map(str, X.keys()))
        # there is no natural order for dictionaries, but we want a consistent order, so sort them by string
        # python uses unicode code points for sorting, which is what we want for cross-language equivalent results
        X_names.sort()
        n_cols = len(X_names)
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)

    n_ignored = 0 if feature_types is None else feature_types.count('ignore')

    if feature_names is None:
        if feature_types is not None:
            if len(feature_types) != n_cols and len(feature_types) != n_cols + n_ignored:
                msg = f"There are {len(feature_types)} feature_types, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            n_cols = len(feature_types)

        feature_names_out = X_names
        if X_names is None:
            feature_names_out = []
            # this isn't used other than to indicate new names need to be created
            feature_types = ['ignore'] * n_cols 
    else:
        n_final = len(feature_names)
        if feature_types is not None:
            n_final = len(feature_types)
            if n_final != len(feature_names) and n_final != len(feature_names) + n_ignored:
                msg = f"There are {n_final} feature_types and {len(feature_names)} feature_names which is a mismatch"
                _log.error(msg)
                raise ValueError(msg)

        feature_names_out = list(map(str, feature_names))

        if X_names is None:
            # ok, need to use position indexing
            if n_final != n_cols and n_final != n_cols + n_ignored:
                msg = f"There are {n_final} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
        else:
            # we might be indexing by name
            names_used = feature_names_out
            if feature_types is not None and len(feature_names_out) == len(feature_types):
                names_used = [feature_name_out for feature_name_out, feature_type in zip(feature_names_out, feature_types) if feature_type != 'ignore']

            X_names_unique = set(name for name, n_count in Counter(X_names).items() if n_count == 1)
            if any(name not in X_names_unique for name in names_used):
                # ok, need to use position indexing
                if n_final != n_cols and n_final != n_cols + n_ignored:
                    msg = f"There are {n_final} features, but X has {n_cols} columns"
                    _log.error(msg)
                    raise ValueError(msg)

    if feature_types is not None:
        if len(feature_types) == len(feature_names_out):
            if len(feature_names_out) - n_ignored != len(set(feature_name_out for feature_name_out, feature_type in zip(feature_names_out, feature_types) if feature_type != 'ignore')):
                msg = "cannot have duplicate feature names"
                _log.error(msg)
                raise ValueError(msg)

            return feature_names_out

        names_set = set(feature_names_out)

        names = []
        names_idx = 0
        feature_idx = 0
        for feature_type in feature_types:
            if feature_type == 'ignore':
                while True:
                    # non-devs looking at our models will like 1 indexing better than 0 indexing
                    # give 4 digits to the number so that anything below 9999 gets sorted in the right order in string format
                    feature_idx += 1
                    name = f"feature_{feature_idx:04}"
                    if name not in names_set:
                        break
            else:
                name = feature_names_out[names_idx]
                names_idx += 1
            names.append(name)

        feature_names_out = names

    if len(feature_names_out) != len(set(feature_names_out)):
        msg = "cannot have duplicate feature names"
        _log.error(msg)
        raise ValueError(msg)

    return feature_names_out

def clean_vector(vec, param_name):
    # called under: fit

    if isinstance(vec, ma.masked_array):
        # do this before np.ndarray since ma.masked_array is a subclass of np.ndarray
        mask = vec.mask
        if mask is not ma.nomask:
            if mask.any():
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
        vec = vec.data
    elif isinstance(vec, np.ndarray):
        pass
    elif _pandas_installed and isinstance(vec, pd.Series):
        if vec.hasnans:
            # if hasnans is true then there is definetly a real missing value in there and not just a mask
            msg = f"{param_name} cannot contain missing values"
            _log.error(msg)
            raise ValueError(msg)
        vec = vec.values.astype(dtype=vec.dtype.type, copy=False)
    elif _pandas_installed and isinstance(vec, pd.DataFrame):
        if vec.shape[1] == 1:
            vec = vec.iloc[:, 0]
            if vec.hasnans:
                # if hasnans is true then there is definetly a real missing value in there and not just a mask
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
            vec = vec.values.astype(dtype=vec.dtype.type, copy=False)
        elif vec.shape[0] == 1:
            vec = vec.astype(np.object_, copy=False).values.reshape(-1)
        else:
            msg = f"{param_name} cannot be a multidimensional pandas.DataFrame"
            _log.error(msg)
            raise ValueError(msg)
    elif _scipy_installed and isinstance(vec, sp.sparse.spmatrix):
        if vec.shape[0] == 1 or vec.shape[1] == 1:
            vec = vec.toarray().reshape(-1)
        else:
            msg = f"{param_name} cannot be a multidimensional scipy.sparse.spmatrix"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(vec, list) or isinstance(vec, tuple):
        vec = np.array(vec, dtype=np.object_)
    elif isinstance(vec, str):
        msg = f"{param_name} cannot be a single object"
        _log.error(msg)
        raise ValueError(msg)
    else:
        try:
            vec = list(vec)
            vec = np.array(vec, dtype=np.object_)
        except TypeError:
            msg = f"{param_name} cannot be a single object"
            _log.error(msg)
            raise ValueError(msg)

    vec = _reshape_1D_if_possible(vec)

    if vec.dtype.type is np.object_:
        if _pandas_installed:
            # pandas also has the pd.NA value that indicates missing.  If Pandas is available though
            # we can use it's function that checks for pd.NA, np.nan, and None
            if pd.isna(vec).any():
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
        else:
            # vec != vec is a check for nan that works even with mixed types, since nan != nan
            if (vec == _none_ndarray).any() or (vec != vec).any():
                msg = f"{param_name} cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)
    elif issubclass(vec.dtype.type, np.floating):
        if np.isnan(vec).any():
            msg = f"{param_name} cannot contain missing values"
            _log.error(msg)
            raise ValueError(msg)
    elif issubclass(vec.dtype.type, np.void):
        msg = f"{param_name} cannot be dtype=numpy.void"
        _log.error(msg)
        raise ValueError(msg)

    return vec

def clean_X(X):
    # called under: fit or predict

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        return X, 1 if X.ndim == 1 else X.shape[0]
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        return X, X.shape[0]
    elif _scipy_installed and isinstance(X, sp.sparse.spmatrix):
        return X, X.shape[0]
    elif isinstance(X, dict):
        for val in X.values():
            # we don't support iterators for dict, so len should work
            return X, len(val)
        return X, -1
    elif isinstance(X, list) or isinstance(X, tuple):
        is_copied = False
    elif X is None:
        msg = "X cannot be a single None"
        _log.error(msg)
        raise ValueError(msg)
    elif isinstance(X, str):
        # str objects are iterable, so don't allow them to get to the list() conversion below
        msg = "X cannot be a single str"
        _log.error(msg)
        raise ValueError(msg)
    else:
        try:
            X = list(X)
            is_copied = True
        except TypeError:
            msg = "X must be an iterable"
            _log.error(msg)
            raise ValueError(msg)

    # for consistency with what the caller expects, we should mirror what np.array([[..], [..], .., [..]]) does
    # [1, 2, 3] is one sample with 3 features
    # [[1], [2], [3]] is three samples with 1 feature
    # [[1], [2], 3] is bug prone.  You could argue that it has to be a single sample since
    #   the 3 only makes sense in that context, but if the 2 value was removed it would change 
    #   from being a single sample with 3 features to being two samples with a single feature, 
    #   so force the user to have consistent inner lists/objects

    for idx in range(len(X)):
        sample = X[idx]
        if isinstance(sample, list) or isinstance(sample, tuple):
            pass
        elif isinstance(sample, ma.masked_array):
            # do this before np.ndarray since ma.masked_array is a subclass of np.ndarray
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = _reshape_1D_if_possible(sample.astype(np.object_, copy=False).filled(np.nan))
        elif isinstance(sample, np.ndarray):
            if sample.ndim == 1:
                pass
            else:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = _reshape_1D_if_possible(sample)
        elif _pandas_installed and isinstance(sample, pd.Series):
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = sample.astype(np.object_, copy=False).values
        elif _pandas_installed and isinstance(sample, pd.DataFrame):
            if sample.shape[0] == 1 or sample.shape[1] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.astype(np.object_, copy=False).values.reshape(-1)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif _scipy_installed and isinstance(sample, sp.sparse.spmatrix):
            if sample.shape[0] == 1 or sample.shape[1] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.toarray().reshape(-1)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif isinstance(sample, str):
            break # this only legal if we have one sample
        else:
            try:
                sample = list(sample)
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample
            except TypeError:
                break # this only legal if we have one sample

    # leave these as np.object_ for now and we'll try to densify per column where we're more likely to 
    # succeed in densification since columns should generally be a single type
    X = np.array(X, dtype=np.object_)
    return X, 1 if X.ndim == 1 else X.shape[0]

def _cut_continuous(native, X_col, processing, binning, bins, min_samples_bin):
    # called under: fit
      
    if processing != 'quantile' and processing != 'quantile_humanized' and processing != 'uniform' and processing != 'winsorized' and not isinstance(processing, list) and not isinstance(processing, np.ndarray):
        if isinstance(binning, list) or isinstance(binning, np.ndarray):
            msg = f"illegal binning type {binning}"
            _log.error(msg)
            raise ValueError(msg)
        processing = binning

    if processing == 'quantile':
        # one bin for missing, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 0, bins - 2)
    elif processing == 'quantile_humanized':
        # one bin for missing, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 1, bins - 2)
    elif processing == 'uniform':
        # one bin for missing, and # of cuts is one less again
        cuts = native.cut_uniform(X_col, bins - 2)
    elif processing == 'winsorized':
        # one bin for missing, and # of cuts is one less again
        cuts = native.cut_winsorized(X_col, bins - 2)
    elif isinstance(processing, np.ndarray):
        cuts = processing.astype(dtype=np.float64, copy=False)
    elif isinstance(processing, list):
        cuts = np.array(processing, dtype=np.float64, copy=False)
    else:
        msg = f"illegal binning type {processing}"
        _log.error(msg)
        raise ValueError(msg)

    return cuts

def bin_native(is_classification, feature_idxs, bins_in, X, y, w, feature_names, feature_types, binning='quantile', min_unique_continuous=4, min_samples_bin=1):
    # called under: fit

    _log.info("Creating native dataset")

    X, n_samples = clean_X(X)
    if n_samples <= 0:
        msg = "X has no samples to train on"
        _log.error(msg)
        raise ValueError(msg)

    y = clean_vector(y, "y")
    if n_samples != len(y):
        msg = f"X has {n_samples} samples and y has {len(y)} samples"
        _log.error(msg)
        raise ValueError(msg)

    if w is not None:
        w = clean_vector(w, "sample_weight")
        if n_samples != len(w):
            msg = f"X has {n_samples} samples and sample_weight has {len(w)} samples"
            _log.error(msg)
            raise ValueError(msg)
        w = w.astype(np.float64, copy=False)
    else:
        # TODO: eliminate this eventually
        w = np.ones_like(y, dtype=np.float64)

    if is_classification:
        if y.dtype == np.object_:
            y = y.astype(np.unicode_)

        uniques, indexes = np.unique(y, return_inverse=True)
        # we're assuming here that all homogenious numpy types generate unique strings
        uniques_text_orginal = uniques.astype(np.unicode_, copy=False)
        uniques_text = uniques_text_orginal.copy()

        # use pure alphabetical ordering for the classes.  It's tempting to sort by frequency first
        # but that could lead to a lot of bugs if the # of categories is close and we flip the ordering
        # in two separate runs, which would flip the ordering of the classes within our score tensors.
        uniques_text.sort()
        classes = dict(zip(uniques_text, count()))

        indexes_remap = np.fromiter((classes[val] for val in uniques_text_orginal), dtype=np.int64, count=len(uniques_text_orginal))
        y = indexes_remap[indexes]
    else:
        classes = None
        y = y.astype(np.float64, copy=False)

    feature_names_out = unify_feature_names(X, feature_names, feature_types)

    native = Native.get_native_singleton()
    n_bytes = native.size_data_set_header(len(feature_idxs), 1, 1)

    feature_types_out = _none_list * len(feature_names_out)
    bins_out = []

    for bins, (feature_idx, feature_type_out, X_col, categories, bad) in zip(bins_in, unify_columns(X, zip(feature_idxs, repeat(None)), feature_names_out, feature_types, min_unique_continuous, False)):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if bins < 2:
            raise ValueError(f"bins was {bins}, but must be 2 or higher. One bin for missing, and at least one more for the non-missing values.")

        feature_types_out[feature_idx] = feature_type_out
        feature_type = None if feature_types is None else feature_types[feature_idx]

        if categories is None:
            # continuous feature
            if bad is not None:
                msg = f"Feature {feature_names_out[feature_idx]} is indicated as continuous, but has non-numeric data"
                _log.error(msg)
                raise ValueError(msg)

            cuts = _cut_continuous(native, X_col, feature_type, binning, bins, min_samples_bin)
            X_col = native.discretize(X_col, cuts)
            bins_out.append(cuts)
            n_bins = len(cuts) + 2
        else:
            # categorical feature
            bins_out.append(categories)
            n_bins = len(categories) + 1
            if bad is not None:
                msg = f"Feature {feature_names_out[feature_idx]} has unrecognized ordinal values"
                _log.error(msg)
                raise ValueError(msg)

        n_bytes += native.size_feature(feature_type_out == 'nominal', n_bins, X_col)

    n_bytes += native.size_weight(w)
    if is_classification:
        n_bytes += native.size_classification_target(len(classes), y)
    else:
        n_bytes += native.size_regression_target(y)

    shared_dataset = RawArray('B', n_bytes)

    native.fill_data_set_header(len(feature_idxs), 1, 1, n_bytes, shared_dataset)

    for bins, (feature_idx, feature_type_out, X_col, categories, _) in zip(bins_out, unify_columns(X, zip(feature_idxs, repeat(None)), feature_names_out, feature_types, min_unique_continuous, False)):
        if n_samples != len(X_col):
            # re-check that that number of samples is identical since iterators can be used up by looking at them
            # this also protects us from badly behaved iterators from causing a segfault in C++ by returning an
            # unexpected number of items and thus a buffer overrun on the second pass through the data
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        feature_type = None if feature_types is None else feature_types[feature_idx]
        if categories is None:
            # continuous feature
            X_col = native.discretize(X_col, bins)
            n_bins = len(cuts) + 2
        else:
            # categorical feature
            n_bins = len(categories) + 1

        # TODO: we're writing these out currently in any order.  We need to include an integer indicating which
        # feature_idx we think a feature is in our higher language and we should use those when referring to features
        # accross the C++ interface

        # We're writing our feature data out in any random order that we get it.  This is fine in terms of performance
        # since the booster has a chance to re-order them again when it constructs the boosting specific dataframe.  
        # For interactions we'll be examining many combinations so the order in our C++ dataframe won't really matter.
        native.fill_feature(feature_type_out == 'nominal', n_bins, X_col, n_bytes, shared_dataset)

    native.fill_weight(w, n_bytes, shared_dataset)
    if is_classification:
        native.fill_classification_target(len(classes), y, n_bytes, shared_dataset)
    else:
        native.fill_regression_target(y, n_bytes, shared_dataset)

    return shared_dataset, feature_names_out, feature_types_out, bins_out, classes

def score_terms(X, feature_names_out, feature_types_out, terms):
    # called under: predict

    # prior to calling this function, call deduplicate_bins which will eliminate extra work in this function

    # this generator function returns data in whatever order it thinks is most efficient.  Normally for 
    # mains it returns them in order, but pairs will be returned as their data completes and they can
    # be mixed in with mains.  So, if we request data for [(0), (1), (2), (3), (4), (1, 3)] the return sequence
    # could be [(0), (1), (2), (3), (1, 3), (4)].  More complicated pair/triples return even more randomized ordering.
    # For additive models the results can be processed in any order, so this imposes no penalities on us.

    _log.info("score_terms")

    X, n_samples = clean_X(X)

    requests = []
    waiting = dict()
    for term in terms:
        features = term['features']
        # the last position holds the term object
        # the first len(features) items hold the binned data that we get back as it arrives
        # the middle len(features) items hold either "True" or None indicating if there are unknown categories we need to zero
        requirements = _none_list * (1 + 2 * len(features))
        requirements[-1] = term
        for feature_idx, feature_bins in zip(features, term['bins']):
            if isinstance(feature_bins, dict):
                # categorical feature
                request = (feature_idx, feature_bins)
                key = (feature_idx, id(feature_bins))
            else:
                # continuous feature
                request = (feature_idx, None)
                key = request
            waiting_list = waiting.get(key, None)
            if waiting_list is None:
                requests.append(request)
                waiting[key] = [requirements]
            else:
                waiting_list.append(requirements)

    native = Native.get_native_singleton()

    for column_feature_idx, _, X_col, column_categories, bad in unify_columns(X, requests, feature_names_out, feature_types_out, None, True):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if column_categories is None:
            # continuous feature

            if bad is not None:
                # TODO: we could pass out a bool array instead of objects for this function only
                bad = bad != _none_ndarray

            cuts_completed = dict()
            for requirements in waiting[(column_feature_idx, None)]:
                term = requirements[-1]
                if term is not None:
                    features = term['features']
                    is_done = True
                    for dimension_idx, term_feature_idx, cuts in zip(count(), features, term['bins']):
                        if term_feature_idx == column_feature_idx:
                            discretized = cuts_completed.get(id(cuts), None)
                            if discretized is None:
                                discretized = native.discretize(X_col, cuts)
                                if bad is not None:
                                    discretized[bad] = -1

                                cuts_completed[id(cuts)] = discretized
                            requirements[dimension_idx] = discretized
                            if bad is not None:
                                # indicate that we need to check for unknowns
                                requirements[len(features) + dimension_idx] = True
                        else:
                            if requirements[dimension_idx] is None:
                                is_done = False

                    if is_done:
                        # the requirements can contain features with both categoricals or continuous
                        binned_data = tuple(requirements[0:len(features)])
                        scores = term['scores'][binned_data]
                        for data, unknown_indicator in zip(binned_data, requirements[len(features):-1]):
                            if unknown_indicator:
                                scores[data < 0] = 0
                        requirements[:] = _none_list # clear references so that the garbage collector can free them
                        yield term, scores
        else:
            # categorical feature

            if bad is not None:
                # TODO: we could pass out a single bool (not an array) if these aren't continuous convertible
                pass # TODO: improve this handling

            for requirements in waiting[(column_feature_idx, id(column_categories))]:
                term = requirements[-1]
                if term is not None:
                    features = term['features']
                    is_done = True
                    for dimension_idx, term_feature_idx, term_categories in zip(count(), features, term['bins']):
                        if term_feature_idx == column_feature_idx and term_categories is column_categories:
                            requirements[dimension_idx] = X_col
                            if bad is not None:
                                # indicate that we need to check for unknowns
                                requirements[len(features) + dimension_idx] = True
                        else:
                            if requirements[dimension_idx] is None:
                                is_done = False

                    if is_done:
                        # the requirements can contain features with both categoricals or continuous
                        binned_data = tuple(requirements[0:len(features)])
                        scores = term['scores'][binned_data]
                        for data, unknown_indicator in zip(binned_data, requirements[len(features):-1]):
                            if unknown_indicator:
                                scores[data < 0] = 0
                        requirements[:] = _none_list # clear references so that the garbage collector can free them
                        yield term, scores

def deduplicate_bins(terms):
    # calling this function before calling score_terms allows score_terms to operate more efficiently since it'll
    # be able to avoid re-binning data for pairs that have already been processed in mains or other pairs since we 
    # use the id of the bins to identify feature data that was previously binned

    # TODO: use this function!

    uniques = dict()
    for term in terms:
        term_bins = term['bins']
        for idx, feature_bins in enumerate(term_bins):
            if isinstance(feature_bins, dict):
                key = frozenset(feature_bins.items())
            else:
                key = tuple(feature_bins)
            existing = uniques.get(key, None)
            if existing is None:
                uniques[key] = feature_bins
            else:
                term_bins[idx] = feature_bins

def unify_data2(X, y=None, feature_names=None, feature_types=None, missing_data_allowed=False):
    pass # TODO: do

