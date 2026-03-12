# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from warnings import warn
from collections import Counter
from itertools import count, repeat, compress
from operator import ne, eq

import numpy as np
from numpy import ma
from numpy import (
    array as np_array,
    array_equal,
    concatenate,
    result_type,
    zeros,
    empty,
    arange,
    fromiter,
    unique,
    place,
    isnan,
    logical_not,
    ascontiguousarray,
    full,
    where,
    ndarray,
    dtype,
    bool_,
    int64,
    float64,
    integer,
    str_,
    object_,
    floating,
    complexfloating,
    nan,
)

from numpy.ma import masked_array, nomask
from numpy.char import endswith, rstrip

_log = logging.getLogger(__name__)


class _ImpossibleType:
    pass


try:
    import pandas as pd

    _SeriesType = pd.Series
    _DataFrameType = pd.DataFrame
    _CategoricalDtype = pd.CategoricalDtype
    _StringDtype = pd.StringDtype

    # pandas also has the pd.NA value that indicates missing. If Pandas is
    # available we can use the pd.notna function that checks for
    # pd.NA, np.nan, math.nan, and None.  pd.notna is also faster than the
    # alternative (X_col == X_col) & (X_col != np.array(None)) below
    from pandas import notna as _notna
    from pandas import factorize as _factorize

except ImportError:
    _SeriesType = _ImpossibleType
    _DataFrameType = _ImpossibleType
    _CategoricalDtype = _ImpossibleType
    _StringDtype = _ImpossibleType

    def _factorize(values):
        uniques, indexes = unique(values, return_inverse=True)
        return indexes, uniques

    def _notna(values):
        nonmissings = values == values
        nonmissings &= values != _none_ndarray
        return nonmissings


try:
    import scipy as sp

    _sparray = sp.sparse.sparray
    _spmatrix = sp.sparse.spmatrix
    _dia_array = sp.sparse.dia_array
    _bsr_array = sp.sparse.bsr_array
    _coo_array = sp.sparse.coo_array
except ImportError:
    _sparray = _ImpossibleType
    _spmatrix = _ImpossibleType
    _dia_array = _ImpossibleType
    _bsr_array = _ImpossibleType
    _coo_array = _ImpossibleType


# BIG TODO LIST:
# - add support for a "ordinal_fast" and "nominal_fast".  We would accept these in feature_types as
#   a dict of (int/float -> string) for 'ordinal_fast', and (string -> int/float) for 'nominal_fast'
#   the we'd write our feature_types_in values as "ordinal_fast" and "nominal_fast" and we'd exepct
#   integers in whatever evaluation format we got.  This would allow us to accept a float64 numpy array
#   and have inside that nominal/ordinal/continuous/missing values that would be highly compressed.  Both of these
#   would have restriction in that the numbers would have to be contiguous (maybe allowing for compression??) and
#   would start from 1, with 0 as reserved for missing values.  A big issues is that with this encoding, the
#   system on which we do predict needs to also encode them as integers and they have no flexibility to change
#   that, except perhaps they could edit the model to change from 'nominal_fast' to 'nominal'
#   { "Canada" : 1, "Japan" : 2, "Seychelles" : 3} => string to int mapping -> nominals
#   { 1: "low", 2: "medium", 3: "high" } => int to object(string) mapping -> ordinals
#   We still record these as ["low", "medium", "high"] and ["Canada", "Japan", "Seychelles"] and we use the
#   feature type value to know that these are "ordinal_fast" and "nominal_fast"
#
# - if we recieve an unseen float64 value in a 'nominal' or 'ordinal', then check if all the categorical
#   value strings are convertible to float64.  If that's the case then find the mid-point between the categories
#   after they are converted to strings and create a pseudo-continuous value of the feature and figure out where
#   the previously unseen float64 should go.  WE do need to sort the category strings by float64, but we don't
#   to to compute the split points because we can just do a binary search against the categories after they are
#   converted to floats and then look at the distance between the upper and lower category and choose the one
#   that is closest, and choose the upper one if the distance is equal since then the cut would be on the value
#   and we use lower bound semantics (where the value gets into the upper bin if it's exactly the cut value)
#
# - we should create post-model modification routines so someone could construct an integer based
#   ordinal/categorical and build their model and evaluate it efficiently, BUT when they want
#   to view the model they can replace the "1", "2", "3" values with "low", "medium", "high" for graphing
#
# NOTES:
#   - other double to text and text to double:
#     https://github.com/google/double-conversion/blob/master/LICENSE -> BSD-3
#     https://stackoverflow.com/questions/28494758/how-does-javascript-print-0-1-with-such-accuracy -> https://www.cs.tufts.edu/~nr/cs257/archive/florian-loitsch/printf.pdf
#     https://github.com/juj/MathGeoLib/blob/master/src/Math/grisu3.c -> ?
#     https://github.com/dvidelabs/flatcc/blob/master/external/grisu3/grisu3_print.h -> Apache 2.0
#     https://github.com/dvidelabs/flatcc/tree/master/external/grisu3
#     https://www.ryanjuckett.com/printing-floating-point-numbers/
#     https://github.com/catboost/catboost/blob/ff34a3aadeb2e31e573519b4371a252ff5e5f209/contrib/python/numpy/py3/numpy/core/src/multiarray/dragon4.h
#     Apparently Numpy has a copy of Ryan Juckett's code liceded in MIT instead of Zlib license
#     YES!  ->   float to string in MIT license:
#     https://github.com/numpy/numpy/blob/3de252be1215c0f9bc0a2f5c3aebdd7ffc86e410/numpy/core/src/multiarray/dragon4.h
#     https://github.com/numpy/numpy/blob/3de252be1215c0f9bc0a2f5c3aebdd7ffc86e410/numpy/core/src/multiarray/dragon4.c
#   - Python uses the gold standard for float/string conversion: https://www.netlib.org/fp/dtoa.c
#     https://github.com/python/cpython/blob/main/Python/dtoa.c
#     This code outputs the shortest possible string that uses IEEE 754 "exact rounding" using bankers' rounding
#     which also guarantees rountrips precicely.  This is great for interpretability.  Unfortunatetly this means
#     that we'll need code in the other languages that generates the same strings and for converting back to floats.
#     Fortunately the python C++ code is available and we can use that to get the exact same conversions and make
#     that available in other languages to call into the C++ to harmonize floating point formats.
#     Python is our premier language and has poor performance if you try to do operations in loops, so we'll
#     force all the other platforms to conform to python specifications.


_disallowed_types = (
    complex,
    list,
    tuple,
    range,
    bytearray,
    memoryview,
    set,
    frozenset,
    dict,
    type(Ellipsis),
    np.void,
    np.complexfloating,
)
_none_ndarray = np.array(None)
_float_types = (float, np.floating)
_bool_types = (bool, np.bool_)
_all_int_types = (int, np.integer)
# np.str_ derrives from str and np.bytes_ derrives from bytes so no need to include
_strconv_types = (str, bytes, int, np.integer, np.datetime64, np.timedelta64)
_intboolpython_types = (int, bool, np.integer, np.bool_)
_float_int_bool_types = (np.floating, np.integer, np.bool_)
_float_int_bool_object_types = (np.floating, np.integer, np.bool_, np.object_)
_complex_void_types = (np.complexfloating, np.void)
_float_int_types = (float, int, np.floating, np.integer)
_list_tuple_types = (list, tuple)
_repeat_float_types = repeat(_float_types)
_repeat_bool_types = repeat(_bool_types)
_repeat_ignore = repeat("ignore")
_repeat_negativeone = repeat(-1)
_floatable = (float, int, bool, np.floating, np.integer, np.bool_)
_repeat_floatable = repeat(_floatable)
_array_zero = np.zeros(1, np.int64)
_stringable = (_CategoricalDtype, _StringDtype)
_slice_none = slice(None)
_repeat_str = repeat(str)
_not_one = (1).__ne__
_spmatrix_or_sparray = (_spmatrix, _sparray)
_hard_sparse = (_dia_array, _bsr_array, _coo_array)
_str_bytes_types = (str, bytes)
_str_bytes_types_and_prev = (
    str,
    bytes,
    float,
    int,
    bool,
    np.floating,
    np.integer,
    np.bool_,
)
_repeat_str_bytes = repeat(_str_bytes_types)


def _densify_categorical(X_col):
    # TODO: this function could be optimized more

    # numpy hierarchy of types
    # https://numpy.org/doc/stable/reference/arrays.scalars.html

    types = set(map(type, X_col))

    if all(issubclass(t, _float_types) for t in types):
        # strip trailing ".0" for floats so that floats and integers are the same for integer representable floats
        X_col = (X_col.astype(float64) + 0.0).astype(str_)
        wholes = endswith(X_col, ".0")
        if wholes.any():
            X_col[wholes] = rstrip(rstrip(X_col[wholes], "0"), ".")
        return X_col

    if all(issubclass(t, _bool_types) for t in types):
        return where(X_col.astype(bool_), "1", "0")

    if all(issubclass(t, _strconv_types) for t in types):
        return X_col.astype(str_)

    is_float = False
    is_bool = False
    for one_type in types:
        if issubclass(one_type, _float_types):
            is_float = True
        elif issubclass(one_type, _bool_types):
            is_bool = True
        elif issubclass(one_type, _strconv_types):
            # issubclass(, str) also works for np.str_
            # issubclass(, bytes) also works for np.bytes_
            # str/bytes objects have __iter__, so special case this to allow
            # int objects use the default __str__ function, so special case this to allow
            continue
        elif issubclass(one_type, _disallowed_types):
            # list of python types primarily from: https://docs.python.org/3/library/stdtypes.html
            msg = f"X contains the disallowed type {one_type}"
            _log.error(msg)
            raise TypeError(msg)
        elif hasattr(one_type, "__iter__") or hasattr(one_type, "__getitem__"):
            # check for __iter__ and __getitem__ to filter out iterables
            # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
            msg = f"X contains the disallowed iterable type {one_type}"
            _log.error(msg)
            raise TypeError(msg)
        elif hasattr(one_type, "__contains__"):
            msg = f"X contains the disallowed set type {one_type}"
            _log.error(msg)
            raise TypeError(msg)
        elif one_type.__str__ is object.__str__:
            # if any object in our list uses the default object __str__ function then it'll
            # include the id(val) pointer in the string text, which isn't going to be useful as a categorical
            # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object

            msg = f"X contains the type {one_type} which does not define a __str__ function"
            _log.error(msg)
            raise TypeError(msg)

    non = None
    if is_float:
        floatable = fromiter(
            map(issubclass, map(type, X_col), _repeat_float_types),
            bool_,
            X_col.shape[0],
        )

        floats = (X_col[floatable].astype(float64) + 0.0).astype(str_)
        wholes = endswith(floats, ".0")
        if wholes.any():
            floats[wholes] = rstrip(rstrip(floats[wholes], "0"), ".")
        non = ~floatable
        X_col = X_col[non]

    if is_bool:
        boolable = fromiter(
            map(issubclass, map(type, X_col), _repeat_bool_types),
            bool_,
            X_col.shape[0],
        )
        bools = where(X_col[boolable].astype(bool_), "1", "0")
        nonboolable = ~boolable
        X_col = X_col[nonboolable]

        if non is None:
            non = nonboolable
        else:
            # convert to positions in the original array
            tmp = zeros(non.shape[0], bool_)
            tmp[non] = boolable
            boolable = tmp

            tmp = zeros(non.shape[0], bool_)
            tmp[non] = nonboolable
            non = tmp

    # bytes and np.bytes_ are converted to strings cleanly when calling .astype(str_)
    X_col = X_col.astype(str_)

    if is_float or is_bool:
        types = [X_col.dtype]
        if is_float:
            types.append(floats.dtype)
        if is_bool:
            types.append(bools.dtype)

        X_col_tmp = empty(non.shape[0], result_type(*types))
        X_col_tmp[non] = X_col

        if is_float:
            X_col_tmp[floatable] = floats

        if is_bool:
            X_col_tmp[boolable] = bools

        X_col = X_col_tmp

    return X_col


def _densify_continuous(X_col):
    # TODO: this function could be optimized more

    # numpy hierarchy of types
    # https://numpy.org/doc/stable/reference/arrays.scalars.html

    types = set(map(type, X_col))

    if all(issubclass(t, _intboolpython_types) for t in types):
        try:
            # faster to use vectorized conversion after int64 conversion
            return X_col.astype(int64).astype(float64, "C"), None
        except OverflowError:
            # We must have a big number that can only be represented by np.uint64
            return X_col.astype(float64, "C"), None

    if all(issubclass(t, _floatable) for t in types):
        return X_col.astype(float64, "C"), None

    if all(issubclass(t, _str_bytes_types_and_prev) for t in types):
        # this also catches np.str_ and np.bytes_
        X_col = X_col.astype(str_)
        try:
            return X_col.astype(float64, "C"), None
        except ValueError:
            # ValueError occurs when a string could not be converted to a float
            return _process_continuous_strings(X_col, None)

    floatable = None
    if any(issubclass(t, _floatable) for t in types):
        # some floatable, but not all

        floatable = fromiter(
            map(issubclass, map(type, X_col), _repeat_floatable),
            bool_,
            X_col.shape[0],
        )

        floats = X_col[floatable].astype(float64)
        nonfloatable = ~floatable
        X_col = X_col[nonfloatable]

    stringable = None
    if any(issubclass(t, _str_bytes_types) for t in types):
        # some strings, but not all

        stringable = fromiter(
            map(issubclass, map(type, X_col), _repeat_str_bytes),
            bool_,
            X_col.shape[0],
        )

        strings = X_col[stringable].astype(str_)
        try:
            strings = strings.astype(float64, "C")
            strings_bad = None
        except ValueError:
            # ValueError occurs when a string could not be converted to a float
            strings, strings_bad = _process_continuous_strings(strings, None)

        nonstringable = ~stringable
        X_col = X_col[nonstringable]

    n = X_col.shape[0]
    conv_float = empty(n, float64)
    conv_fail = zeros(n, bool_)

    for i, x in enumerate(X_col):
        try:
            conv_float[i] = float(x)
        except:
            conv_fail[i] = True

    X_col = X_col[conv_fail].astype(str_)

    try:
        X_col = X_col.astype(float64, "C")
        bad = None
    except ValueError:
        # ValueError occurs when a string could not be converted to a float
        X_col, bad = _process_continuous_strings(X_col, None)

    if bad is not None:
        # there should be at least one bad one
        bad_tmp = zeros(conv_fail.shape[0], bool_)
        bad_tmp[conv_fail] = bad
        bad = bad_tmp

    conv_float[conv_fail] = X_col
    X_col = conv_float

    if stringable is not None:
        X_col_tmp = empty(stringable.shape[0], float64)
        X_col_tmp[nonstringable] = X_col
        X_col_tmp[stringable] = strings
        X_col = X_col_tmp
        if bad is not None or strings_bad is not None:
            bad_tmp = zeros(stringable.shape[0], bool_)
            if bad is not None:
                bad_tmp[nonstringable] = bad
            if strings_bad is not None:
                bad_tmp[stringable] = strings_bad
            bad = bad_tmp

    if floatable is not None:
        X_col_tmp = empty(floatable.shape[0], float64)
        X_col_tmp[floatable] = floats
        X_col_tmp[nonfloatable] = X_col
        X_col = X_col_tmp

        if bad is not None:
            bad_tmp = zeros(floatable.shape[0], bool_)
            bad_tmp[nonfloatable] = bad
            bad = bad_tmp

    return X_col, bad


def categorical_encode(uniques, indexes, nonmissings, categories):
    mapping = fromiter(
        map(categories.get, uniques, _repeat_negativeone), int64, uniques.shape[0]
    )

    n_cat = len(categories)
    if mapping.shape[0] <= n_cat:
        if array_equal(mapping, arange(1, mapping.shape[0] + 1, dtype=int64)):
            # CategoricalDType can encode values as np.int8. We cannot allow an
            # int8 to overflow when we add 1, so convert to int64 first, and we
            # also need to make a copy here because we cache the raw data and
            # re-use it for different binning levels on the same feature.

            indexes = indexes.astype(int64)
            indexes += 1

            if nonmissings is None or nonmissings is False:
                return indexes

            indexes_tmp = zeros(nonmissings.shape[0], int64)
            indexes_tmp[nonmissings] = indexes
            return indexes_tmp
    else:
        if array_equal(mapping[:n_cat], arange(1, n_cat + 1, dtype=int64)):
            # CategoricalDType can encode values as np.int8. We cannot allow an
            # int8 to overflow when we add 1, so convert to int64 first, and we
            # also need to make a copy here because we cache the raw data and
            # re-use it for different binning levels on the same feature.

            indexes = indexes.astype(int64)
            indexes += 1
            indexes[n_cat < indexes] = -1

            if nonmissings is None or nonmissings is False:
                return indexes

            indexes_tmp = zeros(nonmissings.shape[0], int64)
            indexes_tmp[nonmissings] = indexes
            return indexes_tmp

    if nonmissings is None:
        # indexes should be all positive if nonmissings is None
        return mapping[indexes]

    if nonmissings is False:
        # missing values are -1 in indexes, so append 0 to the map, which is index -1
        return concatenate((mapping, _array_zero))[indexes]

    indexes_tmp = zeros(nonmissings.shape[0], int64)
    indexes_tmp[nonmissings] = mapping[indexes]
    return indexes_tmp


def _process_column_initial_nonschematized(
    feature_idx, feature_type, min_unique_continuous, get_col_schematized
):
    nonmissings, uniques, indexes, _ = get_col_schematized(feature_idx, "nominal")
    if nonmissings is False:
        nonmissings = 0 <= indexes
        if nonmissings.all():
            nonmissings = None
        else:
            indexes = indexes[nonmissings]

    try:
        # we rely here on there being a round trip format within this language from float64 to text to float64

        # TODO: does this work if there are spaces or bools?

        floats = uniques.astype(np.float64)
    except ValueError:
        # ValueError occurs when a string could not be converted to a float
        floats = None

    if min_unique_continuous is not None and floats is not None:
        # floats can have more than one string representation, so run unique again to check if we have
        # min_unique_continuous unique float64s in binary representation
        if min_unique_continuous <= len(np.unique(floats)):
            _, _, floats, bad = get_col_schematized(feature_idx, "continuous")
            if bad is None:
                # the schematized code could have slight string to float differences
                # so check that both accept all the floats
                return None, None, floats

    # TODO: we need to move this re-ordering functionality to EBMPreprocessor.fit(...) and return a
    # np.unicode_ array here.  There are two issues with keeping it here
    #   1) If the user wants 'nominal_prevalence' in a DP model, then we need to order the prevalence
    #      by the publically visible noisy weights rather than the private non-noisy prevalences,
    #      but we don't have access to the noisy weights here.  We haven't documented 'nominal_prevalence'
    #      yet, so nobody should be using it yet, but before we make it public we need to solve this issue
    #   2) If we someday want to have an 'eval_set' that has a separate X_eval, then we'll need
    #      two iterators that operate on different X's.  If that happens then the categories dictionary
    #      needs to be synchronized, so we need access to all the possible categories which is not available
    #      here
    # Since we only really care about speed during predict time, and at predict time we already have a
    # categories dictionary, moving this to EBMPreprocessor.fit(...) won't cause any performance issues
    # but it's a bit more complicated.  Also, we need to think through how we handle categoricals from
    # pandas.  We can't return an np.unicode_ array there since then we'd loose the ordering that pandas
    # gives us, which at a minimum is required for ordinals, and is nice to preserve for nominals because
    # it gives the user an easy way to order the nominals on the graph and in the models (for model editing).
    #
    # Alternatively, if we decide to expose the integer bag definitions instead of having an eval_set then
    # we could probably just keep the ordering here and then re-order them again in
    # EBMPreprocessor.fit(...) for DP models.  If we destroy the information about prevalence and resort
    # by noisy prevalence then that would be ok.

    # TODO: add a callback function option here that allows the caller to sort, remove, combine
    if feature_type == "nominal_prevalence":
        counts = np.bincount(indexes)
        if floats is None:
            categories = [(-item[0], item[1]) for item in zip(counts, uniques)]
        else:
            categories = [
                (-item[0], item[1], item[2]) for item in zip(counts, floats, uniques)
            ]
        categories.sort()
        categories = [x[-1] for x in categories]
    elif feature_type != "nominal_alphabetical" and floats is not None:
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        categories.sort()
        categories = [x[1] for x in categories]
    else:
        categories = [item for item in uniques]
        categories.sort()

    mapping = np.fromiter(
        map(dict(zip(categories, count())).__getitem__, uniques),
        np.int64,
        len(uniques),
    )
    return nonmissings, np.array(categories, np.str_), mapping[indexes]


def _process_continuous_strings(X_col, nonmissings):
    # In theory, python, pandas, and numpy all use correct rounding and
    # should therefore convert strings to the same float64 values,
    # but there can be slight differences with things like allowing spaces
    # at the start and end of strings, or which order or methods are used
    # on objects when converting to floats. We want repetable conversion
    # methods that give the same results even if the data is presented
    # slightly differently. Since we want to accept vals.astype(np.float64)
    # for fast converstion elsewhere, we force all conversions as numpy
    # arrays of objects or strings.

    # TODO: attempt to optimize this by converting entire windows
    # within the data and progressively growing/shrinking the windows

    n_samples = X_col.shape[0]
    bad = zeros(n_samples, bool_)
    floats = zeros(n_samples, float64)
    for idx in range(n_samples):
        # slice one item at a time keeping as an ndarray for consistency
        one_item_array = X_col[idx : idx + 1]
        try:
            floats[idx] = one_item_array.astype(float64).item()
        except ValueError:
            bad[idx] = True

    if not bad.any():
        bad = None

    if nonmissings is None:
        return floats, bad

    floats_tmp = full(nonmissings.shape[0], nan, float64)
    floats_tmp[nonmissings] = floats

    if bad is None:
        return floats_tmp, None

    bad_tmp = zeros(nonmissings.shape[0], bool_)
    bad_tmp[nonmissings] = bad
    return floats_tmp, bad_tmp


def _process_arrayish_nonschematized(
    feature_idx, X_col, feature_type, min_unique_continuous, get_col_schematized
):
    if feature_type == "continuous":
        # called under: fit or predict
        return (
            "continuous",
            *get_col_schematized(feature_idx, "continuous"),
        )
    if feature_type == "nominal":
        if isinstance(X_col.dtype, _CategoricalDtype):
            return (feature_type, *get_col_schematized(feature_idx, "nominal"))
        return (
            feature_type,
            *_process_column_initial_nonschematized(
                feature_idx, None, None, get_col_schematized
            ),
            None,
        )
    if feature_type == "ordinal":
        if isinstance(X_col.dtype, _CategoricalDtype):
            return (
                feature_type,
                *get_col_schematized(feature_idx, "ordinal"),
            )

        warn(
            "During fitting you should usually specify the ordered strings instead of specifying 'ordinal' as the feature type. When 'ordinal' is specified then alphabetic ordering is used."
        )

        # if the caller passes "ordinal" during fit, the only order that makes sense is either
        # alphabetical or based on float values. Frequency doesn't make sense
        # if the caller would prefer an error, they can check feature_types themselves
        return (
            feature_type,
            *_process_column_initial_nonschematized(
                feature_idx, None, None, get_col_schematized
            ),
            None,
        )
    if feature_type is None or feature_type == "auto":
        if isinstance(X_col.dtype, _CategoricalDtype):
            feature_type = "ordinal" if X_col.array.ordered else "nominal"

            return (
                feature_type,
                *get_col_schematized(feature_idx, feature_type),
            )

        nonmissings, uniques, indexes = _process_column_initial_nonschematized(
            feature_idx, None, min_unique_continuous, get_col_schematized
        )
        return (
            "continuous" if uniques is None else "nominal",
            nonmissings,
            uniques,
            indexes,
            None,
        )
    if feature_type in ("nominal_prevalence", "nominal_alphabetical"):
        if isinstance(X_col.dtype, _CategoricalDtype):
            # TODO: add re-ordering here to support this
            msg = f"{feature_type} currently unsupported"
            _log.error(msg)
            raise ValueError(msg)

        return (
            "nominal",
            *_process_column_initial_nonschematized(
                feature_idx, feature_type, None, get_col_schematized
            ),
            None,
        )
    if feature_type in ("quantile", "rounded_quantile", "uniform", "winsorized"):
        return "continuous", *get_col_schematized(feature_idx, "continuous")
    if isinstance(feature_type, _all_int_types):
        if isinstance(X_col.dtype, _CategoricalDtype):
            # TODO: add support for specifying the threshold between continuous and nominal
            msg = "integer feature_types currently unsupported"
            _log.error(msg)
            raise ValueError(msg)

        nonmissings, uniques, indexes = _process_column_initial_nonschematized(
            feature_idx, None, feature_type, get_col_schematized
        )
        return (
            "continuous" if uniques is None else "nominal",
            nonmissings,
            uniques,
            indexes,
            None,
        )
    if isinstance(feature_type, (str, bytes)):
        # don't allow strings to get to the np.array conversion below
        # isinstance(, str) also works for np.str_
        msg = f"{feature_type} type invalid"
        _log.error(msg)
        raise ValueError(msg)

    n_items = 0
    n_ordinals = 0
    n_continuous = 0
    try:
        for item in feature_type:
            n_items += 1
            if isinstance(item, str):
                # isinstance(, str) also works for np.str_
                n_ordinals += 1
            elif isinstance(item, _float_int_types):
                n_continuous += 1
    except TypeError:
        msg = f"{feature_type} type invalid"
        _log.error(msg)
        raise TypeError(msg)

    if n_continuous == n_items:
        # if n_items == 0 then it must be continuous since we
        # can have zero cut points, but not zero ordinal categories

        return "continuous", *get_col_schematized(feature_idx, "continuous")
    if n_ordinals == n_items:
        if isinstance(X_col.dtype, _CategoricalDtype):
            # TODO: add support for specifying the order of ordinal features
            msg = "reordering ordinals unsupported for CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)

        nonmissings, uniques, indexes, _ = get_col_schematized(feature_idx, "ordinal")
        if nonmissings is False:
            nonmissings = 0 <= indexes
            if nonmissings.all():
                nonmissings = None
            else:
                indexes = indexes[nonmissings]

        try:
            feature_type_dict = dict(zip(feature_type, count()))

            if len(feature_type) != len(feature_type_dict):
                msg = "feature_types contains duplicate ordinal categories."
                _log.error(msg)
                raise Exception(msg)

            mapping = np.fromiter(
                map(feature_type_dict.__getitem__, uniques),
                np.int64,
                len(uniques),
            )
        except KeyError:
            # TODO: warn the user, but allow them to make unseen values

            msg = "X contains values outside of the ordinal set."
            _log.error(msg)
            raise Exception(msg)

        return (
            "ordinal",
            nonmissings,
            np.array(feature_type, np.str_),
            mapping[indexes],
            None,
        )
    msg = f"{feature_type} type invalid"
    _log.error(msg)
    raise TypeError(msg)


def _reshape_1D_if_possible(col):
    if col.ndim != 1:
        if col.ndim == 0:
            # 0 dimensional items exist, but are weird/unexpected. len fails, shape is length 0.
            return empty(0, col.dtype)

        # ignore dimensions that have just 1 item and assume the intent was to give us 1D
        is_found = False
        for n_items in col.shape:
            if n_items > 1:
                if is_found:
                    msg = f"Cannot reshape to 1D. Original shape was {col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
                is_found = True
        col = col.ravel()
    return col


def _process_numpy_column_nonschematized(
    feature_idx,
    X_col,
    feature_type,
    min_unique_continuous,
    get_col_schematized,
):
    if isinstance(X_col, ma.masked_array):
        nonmissings = X_col.mask
        if nonmissings is not ma.nomask:
            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
            if nonmissings.any():
                nonmissings = ~nonmissings
                X_col = X_col.compressed()
                if X_col.dtype.type is np.object_:
                    nonmissings2 = _notna(X_col)

                    if not nonmissings2.all():
                        X_col = X_col[nonmissings2]
                        np.place(nonmissings, nonmissings, nonmissings2)

                return _process_arrayish_nonschematized(
                    feature_idx,
                    X_col,
                    feature_type,
                    min_unique_continuous,
                    get_col_schematized,
                )
        X_col = X_col.data

    if X_col.dtype.type is np.object_:
        nonmissings = _notna(X_col)

        if not nonmissings.all():
            return _process_arrayish_nonschematized(
                feature_idx,
                X_col[nonmissings],
                feature_type,
                min_unique_continuous,
                get_col_schematized,
            )

    return _process_arrayish_nonschematized(
        feature_idx, X_col, feature_type, min_unique_continuous, get_col_schematized
    )


def _process_pandas_column_schematized(X_col, feature_type):
    if feature_type == "continuous":
        dt = X_col.dtype
        tt = dt.type
        if isinstance(dt, dtype):
            if tt is float64:
                return (
                    None,
                    None,
                    ascontiguousarray(X_col.values),
                    None,
                )
            elif issubclass(tt, _float_int_bool_types):
                return (
                    None,
                    None,
                    X_col.values.astype(float64, "C"),
                    None,
                )
            elif tt is object_:
                X_col = X_col.values
                nonmissings = _notna(X_col)
                if nonmissings.all():
                    return (
                        None,
                        None,
                        *_densify_continuous(X_col),
                    )
                else:
                    X_col = X_col[nonmissings]

                    X_col, bad = _densify_continuous(X_col)

                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                    X_col_tmp[nonmissings] = X_col
                    X_col = X_col_tmp

                    if bad is not None:
                        bad_tmp = zeros(nonmissings.shape[0], bool_)
                        bad_tmp[nonmissings] = bad
                        bad = bad_tmp

                    return (
                        None,
                        None,
                        X_col,
                        bad,
                    )

            # pandas never uses np.str_ or np.bytes_

            # fall through to the default handler
        elif issubclass(tt, _float_int_bool_types):
            # this handles Float64Dtype, Float32Dtype, Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype, and BooleanDtype

            return (
                None,
                None,
                X_col.array.to_numpy(float64),  # missing becomes nan for these types
                None,
            )
        elif isinstance(dt, _stringable):
            X_col = X_col.array
            nonmissings = X_col.isna()
            if nonmissings.any():
                nonmissings = ~nonmissings
                # convert to numpy str_ so that we can process strings
                # with the same float conversion code. Mostly IEEE 754 should
                # be identical, but there are edge cases with different conversions
                X_col = X_col[nonmissings].to_numpy(str_)

                try:
                    X_col = X_col.astype(float64, "C")
                except ValueError:
                    # ValueError occurs when a string could not be converted to a float
                    return (
                        None,
                        None,
                        *_process_continuous_strings(X_col, nonmissings),
                    )

                X_col_tmp = full(nonmissings.shape[0], nan, float64)
                X_col_tmp[nonmissings] = X_col
                X_col = X_col_tmp

                return (
                    None,
                    None,
                    X_col,
                    None,
                )
            else:
                # convert to numpy str_ so that we can process strings
                # with the same float conversion code. Mostly IEEE 754 should
                # be identical, but there are edge cases with different conversions
                X_col = X_col.to_numpy(str_)

                try:
                    # numpy, pandas, and python all have identical conversions (IEEE-754)
                    X_col = X_col.astype(float64, "C")
                except ValueError:
                    # ValueError occurs when a string could not be converted to a float
                    return (
                        None,
                        None,
                        *_process_continuous_strings(X_col, None),
                    )

                return (
                    None,
                    None,
                    X_col,
                    None,
                )

        # TODO: implement pd.SparseDtype
        msg = f"{type(dt)} not supported"
        _log.error(msg)
        raise TypeError(msg)

    # feature_type == "nominal" or feature_type == "ordinal"

    dt = X_col.dtype
    if isinstance(dt, _CategoricalDtype):
        # unlike other missing value types, we get back -1's for missing here, so no need to drop them
        X_col = X_col.array
        categories = X_col.categories
        tt = categories.dtype.type
        if issubclass(tt, floating):
            categories = (categories.to_numpy(float64) + 0.0).astype(str_)
            wholes = endswith(categories, ".0")
            if wholes.any():
                categories[wholes] = rstrip(rstrip(categories[wholes], "0"), ".")
        elif tt is bool_:
            categories = where(categories.to_numpy(bool_), "1", "0")
        else:
            categories = categories.to_numpy(str_)

        return (
            False,
            categories,
            X_col.codes,
            None,
        )
    elif isinstance(dt, _StringDtype):
        # factorize uses -1 for missing values
        indexes, uniques = _factorize(X_col.array)
        return (
            False,
            uniques.to_numpy(str_),
            indexes,
            None,
        )
    else:
        tt = dt.type
        if isinstance(dt, dtype):
            if tt is object_:
                X_col = X_col.values
                nonmissings = _notna(X_col)
                if nonmissings.all():
                    nonmissings = None
                else:
                    X_col = X_col[nonmissings]

                indexes, uniques = _factorize(_densify_categorical(X_col))
                return (
                    nonmissings,
                    uniques,
                    indexes,
                    None,
                )
            elif tt is float64:
                indexes, uniques = _factorize(X_col.values)

                uniques = (uniques + 0.0).astype(str_)
                wholes = endswith(uniques, ".0")
                if wholes.any():
                    uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                return (
                    False,
                    uniques,
                    indexes,
                    None,
                )
            elif issubclass(tt, floating):
                indexes, uniques = _factorize(X_col.values)

                uniques = (uniques.astype(float64) + 0.0).astype(str_)
                wholes = endswith(uniques, ".0")
                if wholes.any():
                    uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                return (
                    False,
                    uniques,
                    indexes,
                    None,
                )
            elif tt is bool_:
                indexes, uniques = _factorize(X_col.values)
                return None, where(uniques, "1", "0"), indexes, None
            elif issubclass(tt, integer):
                indexes, uniques = _factorize(X_col.values)
                return None, uniques.astype(str_), indexes, None

            # pandas never uses np.str_ or np.bytes_

            # fall through to the default handler
        elif issubclass(tt, floating):
            # this handles Float64Dtype, Float32Dtype

            indexes, uniques = _factorize(X_col.array)

            uniques = (uniques.to_numpy(float64) + 0.0).astype(str_)
            wholes = endswith(uniques, ".0")
            if wholes.any():
                uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

            return (
                False,
                uniques,
                indexes,
                None,
            )
        elif tt is bool_:
            # BooleanDtype
            indexes, uniques = _factorize(X_col.array)
            return (
                False,
                where(uniques.to_numpy(bool_), "1", "0"),
                indexes,
                None,
            )
        elif issubclass(tt, integer):
            # Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype
            indexes, uniques = _factorize(X_col.array)
            return (
                False,
                uniques.to_numpy(str_),
                indexes,
                None,
            )

        # TODO: implement pd.SparseDtype
        msg = f"{type(dt)} not supported"
        _log.error(msg)
        raise TypeError(msg)


def _process_pandas_column_nonschematized(
    feature_idx, X_col, feature_type, min_unique_continuous, get_col_schematized
):
    if isinstance(X_col.dtype, _CategoricalDtype):
        # unlike other missing value types, we get back -1's for missing here, so no need to drop them
        return _process_arrayish_nonschematized(
            feature_idx, X_col, feature_type, min_unique_continuous, get_col_schematized
        )
    elif isinstance(X_col.dtype, _StringDtype):
        return _process_arrayish_nonschematized(
            feature_idx,
            X_col.dropna().array,  # keep as pandas to preserve compact strings
            feature_type,
            min_unique_continuous,
            get_col_schematized,
        )
    elif issubclass(X_col.dtype.type, _float_int_bool_object_types):
        # this handles Float64Dtype, Float32Dtype, Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype, and BooleanDtype
        return _process_arrayish_nonschematized(
            feature_idx,
            X_col.dropna().to_numpy(),
            feature_type,
            min_unique_continuous,
            get_col_schematized,
        )

    # TODO: implement pd.SparseDtype
    msg = f"{type(dt)} not supported"
    _log.error(msg)
    raise TypeError(msg)


def _process_sparse_column_schematized(X_col, feature_type):
    X_col = X_col.toarray().ravel()

    if feature_type == "continuous":
        if X_col.dtype.type is float64:
            # force C contiguous here for a later call to native.discretize
            return None, None, ascontiguousarray(X_col), None
        try:
            # force C contiguous here for a later call to native.discretize
            return None, None, X_col.astype(float64, "C"), None
        except:  # object conversion can throw any exception in their __float__ or __str__
            return None, None, *_process_continuous_strings(X_col, None)

    # feature_type == "nominal" or feature_type == "ordinal"

    tt = X_col.dtype.type
    if issubclass(tt, floating):
        m = isnan(X_col)
        if m.any():
            logical_not(m, out=m)
            X_col = X_col[m]
            indexes, uniques = _factorize(X_col)

            uniques = (uniques.astype(float64, copy=False) + 0.0).astype(str_)
            wholes = endswith(uniques, ".0")
            if wholes.any():
                uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

            return m, uniques, indexes, None
        indexes, uniques = _factorize(X_col)

        uniques = (uniques.astype(float64, copy=False) + 0.0).astype(str_)
        wholes = endswith(uniques, ".0")
        if wholes.any():
            uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

        return None, uniques, indexes, None

    indexes, uniques = _factorize(X_col)

    if tt is bool_:
        return None, where(uniques, "1", "0"), indexes, None

    return None, uniques.astype(str_, copy=False), indexes, None


def _process_dict_column_nonschematized(
    feature_idx,
    X_col,
    feature_type,
    min_unique_continuous,
    get_col_schematized,
):
    if isinstance(X_col, np.ndarray):  # this includes ma.masked_array
        pass
    elif isinstance(X_col, _SeriesType):
        return _process_pandas_column_nonschematized(
            feature_idx,
            X_col,
            feature_type,
            min_unique_continuous,
            get_col_schematized,
        )
    elif isinstance(X_col, _DataFrameType):
        if X_col.shape[1] == 1:
            return _process_pandas_column_nonschematized(
                feature_idx,
                X_col.iloc[:, 0],
                feature_type,
                min_unique_continuous,
                get_col_schematized,
            )
        if X_col.shape[0] == 1:
            X_col = X_col.to_numpy(np.object_).ravel()
        elif X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(X_col, _spmatrix_or_sparray):
        if X_col.shape[1] == 1 or X_col.shape[0] == 1:
            return _process_arrayish_nonschematized(
                feature_idx,
                X_col.toarray().ravel(),
                feature_type,
                min_unique_continuous,
                get_col_schematized,
            )
        if X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(X_col, _list_tuple_types):
        X_col = np.array(X_col, np.object_)
    elif isinstance(X_col, (str, bytes)):
        # isinstance(, str) also works for np.str_

        # don't allow strings to get to the np.array conversion below
        X_col_tmp = np.empty(1, np.object_)
        X_col_tmp[0] = X_col
        X_col = X_col_tmp
    else:
        try:
            # TODO: we need to iterate though all the columns in preclean_X to handle iterable columns

            # we don't support iterables that get exhausted on their first examination.  This condition
            # should be detected though in preclean_X where we get the length or bin_native where we check the
            # number of samples on the 2nd run through the generator
            X_col = np.array(list(X_col), np.object_)
        except TypeError:
            # if our item isn't iterable, assume it has just 1 item and we'll check below if that's consistent
            X_col_tmp = np.empty(1, np.object_)
            X_col_tmp[0] = X_col
            X_col = X_col_tmp

    return _process_numpy_column_nonschematized(
        feature_idx,
        _reshape_1D_if_possible(X_col),
        feature_type,
        min_unique_continuous,
        get_col_schematized,
    )


def unify_columns_schematized(
    X,
    n_samples,
    feature_names_in,
    feature_types_ignore,
):
    # preclean_X is always called on X prior to calling this function
    #
    # feature_names_in and feature_types are cleaned up versions where there are no
    # duplicate names and feature_types can only consist of
    # "continous", "nominal", or "ordinal"

    if isinstance(X, ndarray):  # this includes ma.masked_array
        tt = X.dtype.type
        if issubclass(tt, _complex_void_types):
            msg = f"{X.dtype.type} type not supported."
            _log.error(msg)
            raise ValueError(msg)

        # TODO: in the future special case this to make single samples faster at predict time

        # TODO: I'm not sure that simply checking X.flags.c_contiguous handles all the situations that we'd want
        # to know about some data.  If we recieved a transposed array that was C ordered how would that look?
        # so read up on this more
        # https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
        # https://numpy.org/doc/stable/reference/arrays.interface.html
        # memoryview

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
        # if X.flags.c_contiguous:
        #    # during predict we don't care as much about memory consumption, so speed it by transposing everything
        #    X = np.asfortranarray(X)

        if len(feature_names_in) == X.shape[1]:
            if isinstance(X, masked_array):
                mask = X.mask
                X = X.data
                if mask is not nomask:
                    if tt is float64:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, feature_idx)
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    place(nonmissings, nonmissings, m)

                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    nonmissings,
                                    uniques,
                                    indexes,
                                    None,
                                )
                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    # force C contiguous here for a later call to native.discretize
                                    return (
                                        None,
                                        None,
                                        ascontiguousarray(X_col),
                                        None,
                                    )

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    indexes, uniques = _factorize(X_col)

                                    uniques = (uniques + 0.0).astype(str_)
                                    wholes = endswith(uniques, ".0")
                                    if wholes.any():
                                        uniques[wholes] = rstrip(
                                            rstrip(uniques[wholes], "0"), "."
                                        )

                                    return m, uniques, indexes, None

                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    None,
                                    uniques,
                                    indexes,
                                    None,
                                )

                    elif issubclass(tt, floating):

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, feature_idx)
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col = X_col.astype(float64)
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    place(nonmissings, nonmissings, m)
                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques.astype(float64) + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    nonmissings,
                                    uniques,
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    return (
                                        None,
                                        None,
                                        X_col.astype(float64, "C"),
                                        None,
                                    )

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    indexes, uniques = _factorize(X_col)

                                    uniques = (uniques.astype(float64) + 0.0).astype(
                                        str_
                                    )
                                    wholes = endswith(uniques, ".0")
                                    if wholes.any():
                                        uniques[wholes] = rstrip(
                                            rstrip(uniques[wholes], "0"), "."
                                        )

                                    return m, uniques, indexes, None

                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques.astype(float64) + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    None,
                                    uniques,
                                    indexes,
                                    None,
                                )

                    elif tt is object_:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, feature_idx)
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                nonmissings2 = _notna(X_col)

                                if not nonmissings2.all():
                                    X_col = X_col[nonmissings2]
                                    place(nonmissings, nonmissings, nonmissings2)
                            else:
                                X_col = X[index]
                                nonmissings = _notna(X_col)

                                if nonmissings.all():
                                    nonmissings = None
                                else:
                                    X_col = X_col[nonmissings]

                            if feature_type == "continuous":
                                X_col, bad = _densify_continuous(X_col)

                                if nonmissings is not None:
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    X_col = X_col_tmp

                                    if bad is not None:
                                        bad_tmp = zeros(nonmissings.shape[0], bool_)
                                        bad_tmp[nonmissings] = bad
                                        bad = bad_tmp

                                return (
                                    None,
                                    None,
                                    X_col,
                                    bad,
                                )

                            indexes, uniques = _factorize(_densify_categorical(X_col))
                            return (
                                nonmissings,
                                uniques,
                                indexes,
                                None,
                            )

                    elif tt is bool_:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, feature_idx)
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col = X_col.astype(float64)
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                indexes, uniques = _factorize(X_col)
                                return (
                                    nonmissings,
                                    where(uniques, "1", "0"),
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    return (
                                        None,
                                        None,
                                        X_col.astype(float64, "C"),
                                        None,
                                    )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    None,
                                    where(uniques, "1", "0"),
                                    indexes,
                                    None,
                                )

                    else:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, feature_idx)
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    try:
                                        X_col = X_col.astype(float64)
                                        X_col_tmp = full(
                                            nonmissings.shape[0], nan, float64
                                        )
                                        X_col_tmp[nonmissings] = X_col
                                        return None, None, X_col_tmp, None
                                    except:  # object conversion can throw any exception in their __float__ or __str__
                                        return (
                                            None,
                                            None,
                                            *_process_continuous_strings(
                                                X_col, nonmissings
                                            ),
                                        )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    nonmissings,
                                    uniques.astype(str_, copy=False),
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    try:
                                        return (
                                            None,
                                            None,
                                            X_col.astype(float64, "C"),
                                            None,
                                        )

                                    except:  # object conversion can throw any exception in their __float__ or __str__
                                        return (
                                            None,
                                            None,
                                            *_process_continuous_strings(X_col, None),
                                        )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    None,
                                    uniques.astype(str_, copy=False),
                                    indexes,
                                    None,
                                )

                    return internal

            if tt is float64:

                def internal(feature_idx, feature_type):
                    X_col = X[:, feature_idx]

                    if feature_type == "continuous":
                        # force C contiguous here for a later call to native.discretize
                        return None, None, ascontiguousarray(X_col), None

                    m = isnan(X_col)
                    if m.any():
                        logical_not(m, out=m)
                        X_col = X_col[m]
                        indexes, uniques = _factorize(X_col)

                        uniques = (uniques + 0.0).astype(str_)
                        wholes = endswith(uniques, ".0")
                        if wholes.any():
                            uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                        return m, uniques, indexes, None
                    indexes, uniques = _factorize(X_col)

                    uniques = (uniques + 0.0).astype(str_)
                    wholes = endswith(uniques, ".0")
                    if wholes.any():
                        uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                    return None, uniques, indexes, None

            elif issubclass(tt, floating):

                def internal(feature_idx, feature_type):
                    X_col = X[:, feature_idx]

                    if feature_type == "continuous":
                        # force C contiguous here for a later call to native.discretize
                        return None, None, X_col.astype(float64, "C"), None

                    m = isnan(X_col)
                    if m.any():
                        logical_not(m, out=m)
                        X_col = X_col[m]
                        indexes, uniques = _factorize(X_col)

                        uniques = (uniques.astype(float64) + 0.0).astype(str_)
                        wholes = endswith(uniques, ".0")
                        if wholes.any():
                            uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                        return m, uniques, indexes, None
                    indexes, uniques = _factorize(X_col)

                    uniques = (uniques.astype(float64) + 0.0).astype(str_)
                    wholes = endswith(uniques, ".0")
                    if wholes.any():
                        uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                    return None, uniques, indexes, None

            elif tt is object_:

                def internal(feature_idx, feature_type):
                    X_col = X[:, feature_idx]

                    nonmissings = _notna(X_col)
                    if nonmissings.all():
                        if feature_type == "continuous":
                            return (
                                None,
                                None,
                                *_densify_continuous(X_col),
                            )

                        indexes, uniques = _factorize(_densify_categorical(X_col))
                        return (
                            None,
                            uniques,
                            indexes,
                            None,
                        )

                    else:
                        if feature_type == "continuous":
                            X_col, bad = _densify_continuous(X_col[nonmissings])

                            X_col_tmp = full(nonmissings.shape[0], nan, float64)
                            X_col_tmp[nonmissings] = X_col
                            X_col = X_col_tmp

                            if bad is not None:
                                bad_tmp = zeros(nonmissings.shape[0], bool_)
                                bad_tmp[nonmissings] = bad
                                bad = bad_tmp

                            return (
                                None,
                                None,
                                X_col,
                                bad,
                            )

                        indexes, uniques = _factorize(
                            _densify_categorical(X_col[nonmissings])
                        )
                        return (
                            nonmissings,
                            uniques,
                            indexes,
                            None,
                        )

            elif tt is bool_:

                def internal(feature_idx, feature_type):
                    X_col = X[:, feature_idx]

                    if feature_type == "continuous":
                        return None, None, X_col.astype(float64, "C"), None

                    indexes, uniques = _factorize(X_col)
                    return None, where(uniques, "1", "0"), indexes, None

            else:

                def internal(feature_idx, feature_type):
                    X_col = X[:, feature_idx]

                    if feature_type == "continuous":
                        try:
                            # force C contiguous here for a later call to native.discretize
                            return None, None, X_col.astype(float64, "C"), None
                        except:  # object conversion can throw any exception in their __float__ or __str__
                            return (
                                None,
                                None,
                                *_process_continuous_strings(X_col, None),
                            )

                    indexes, uniques = _factorize(X_col)
                    return None, uniques.astype(str_, copy=False), indexes, None

            return internal
        else:
            keep_cols = fromiter(
                map("ignore".__ne__, feature_types_ignore),
                bool_,
                len(feature_types_ignore),
            )
            n_keep = keep_cols.sum()
            if n_keep != X.shape[1]:
                msg = f"The model has {len(feature_names_in)} features, but X has {X.shape[1]} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(keep_cols.shape[0], int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)
            if isinstance(X, masked_array):
                mask = X.mask
                X = X.data
                if mask is not nomask:
                    if tt is float64:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, col_map[feature_idx])
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    place(nonmissings, nonmissings, m)
                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    nonmissings,
                                    uniques,
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    # force C contiguous here for a later call to native.discretize
                                    return (
                                        None,
                                        None,
                                        ascontiguousarray(X_col),
                                        None,
                                    )

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    indexes, uniques = _factorize(X_col)

                                    uniques = (uniques + 0.0).astype(str_)
                                    wholes = endswith(uniques, ".0")
                                    if wholes.any():
                                        uniques[wholes] = rstrip(
                                            rstrip(uniques[wholes], "0"), "."
                                        )

                                    return m, uniques, indexes, None
                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    None,
                                    uniques,
                                    indexes,
                                    None,
                                )

                    elif issubclass(tt, floating):

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, col_map[feature_idx])
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col = X_col.astype(float64)
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    place(nonmissings, nonmissings, m)
                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques.astype(float64) + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    nonmissings,
                                    uniques,
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    return (
                                        None,
                                        None,
                                        X_col.astype(float64, "C"),
                                        None,
                                    )

                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    indexes, uniques = _factorize(X_col)

                                    uniques = (uniques.astype(float64) + 0.0).astype(
                                        str_
                                    )
                                    wholes = endswith(uniques, ".0")
                                    if wholes.any():
                                        uniques[wholes] = rstrip(
                                            rstrip(uniques[wholes], "0"), "."
                                        )

                                    return m, uniques, indexes, None
                                indexes, uniques = _factorize(X_col)

                                uniques = (uniques.astype(float64) + 0.0).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    None,
                                    uniques,
                                    indexes,
                                    None,
                                )

                    elif tt is object_:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, col_map[feature_idx])
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                nonmissings2 = _notna(X_col)

                                if not nonmissings2.all():
                                    X_col = X_col[nonmissings2]
                                    place(nonmissings, nonmissings, nonmissings2)
                            else:
                                X_col = X[index]

                                nonmissings = _notna(X_col)

                                if nonmissings.all():
                                    nonmissings = None
                                else:
                                    X_col = X_col[nonmissings]

                            if feature_type == "continuous":
                                X_col, bad = _densify_continuous(X_col)

                                if nonmissings is not None:
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    X_col = X_col_tmp

                                    if bad is not None:
                                        bad_tmp = zeros(nonmissings.shape[0], bool_)
                                        bad_tmp[nonmissings] = bad
                                        bad = bad_tmp

                                return (
                                    None,
                                    None,
                                    X_col,
                                    bad,
                                )

                            indexes, uniques = _factorize(_densify_categorical(X_col))
                            return (
                                nonmissings,
                                uniques,
                                indexes,
                                None,
                            )

                    elif tt is bool_:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, col_map[feature_idx])
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    X_col = X_col.astype(float64)
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None

                                indexes, uniques = _factorize(X_col)
                                return (
                                    nonmissings,
                                    where(uniques, "1", "0"),
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    return (
                                        None,
                                        None,
                                        X_col.astype(float64, "C"),
                                        None,
                                    )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    None,
                                    where(uniques, "1", "0"),
                                    indexes,
                                    None,
                                )

                    else:

                        def internal(feature_idx, feature_type):
                            index = (_slice_none, col_map[feature_idx])
                            nonmissings = mask[index]

                            # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                            if nonmissings.any():
                                nonmissings = ~nonmissings
                                X_col = X[index][nonmissings]

                                if feature_type == "continuous":
                                    try:
                                        X_col = X_col.astype(float64)
                                        X_col_tmp = full(
                                            nonmissings.shape[0], nan, float64
                                        )
                                        X_col_tmp[nonmissings] = X_col
                                        return None, None, X_col_tmp, None
                                    except:  # object conversion can throw any exception in their __float__ or __str__
                                        return (
                                            None,
                                            None,
                                            *_process_continuous_strings(
                                                X_col, nonmissings
                                            ),
                                        )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    nonmissings,
                                    uniques.astype(str_, copy=False),
                                    indexes,
                                    None,
                                )

                            else:
                                X_col = X[index]
                                if feature_type == "continuous":
                                    try:
                                        return (
                                            None,
                                            None,
                                            X_col.astype(float64, "C"),
                                            None,
                                        )
                                    except:  # object conversion can throw any exception in their __float__ or __str__
                                        return (
                                            None,
                                            None,
                                            *_process_continuous_strings(X_col, None),
                                        )

                                indexes, uniques = _factorize(X_col)
                                return (
                                    None,
                                    uniques.astype(str_, copy=False),
                                    indexes,
                                    None,
                                )

                    return internal

            if tt is float64:

                def internal(feature_idx, feature_type):
                    X_col = X[:, col_map[feature_idx]]

                    if feature_type == "continuous":
                        # force C contiguous here for a later call to native.discretize
                        return None, None, ascontiguousarray(X_col), None

                    m = isnan(X_col)
                    if m.any():
                        logical_not(m, out=m)
                        X_col = X_col[m]
                        indexes, uniques = _factorize(X_col)

                        uniques = (uniques + 0.0).astype(str_)
                        wholes = endswith(uniques, ".0")
                        if wholes.any():
                            uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                        return m, uniques, indexes, None
                    indexes, uniques = _factorize(X_col)

                    uniques = (uniques + 0.0).astype(str_)
                    wholes = endswith(uniques, ".0")
                    if wholes.any():
                        uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                    return None, uniques, indexes, None

            elif issubclass(tt, floating):

                def internal(feature_idx, feature_type):
                    X_col = X[:, col_map[feature_idx]]

                    if feature_type == "continuous":
                        # force C contiguous here for a later call to native.discretize
                        return None, None, X_col.astype(float64, "C"), None

                    m = isnan(X_col)
                    if m.any():
                        logical_not(m, out=m)
                        X_col = X_col[m]
                        indexes, uniques = _factorize(X_col)

                        uniques = (uniques.astype(float64) + 0.0).astype(str_)
                        wholes = endswith(uniques, ".0")
                        if wholes.any():
                            uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                        return m, uniques, indexes, None
                    indexes, uniques = _factorize(X_col)

                    uniques = (uniques.astype(float64) + 0.0).astype(str_)
                    wholes = endswith(uniques, ".0")
                    if wholes.any():
                        uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                    return None, uniques, indexes, None

            elif tt is object_:

                def internal(feature_idx, feature_type):
                    X_col = X[:, col_map[feature_idx]]

                    nonmissings = _notna(X_col)

                    if nonmissings.all():
                        if feature_type == "continuous":
                            return (
                                None,
                                None,
                                *_densify_continuous(X_col),
                            )

                        indexes, uniques = _factorize(_densify_categorical(X_col))
                        return (
                            None,
                            uniques,
                            indexes,
                            None,
                        )

                    else:
                        if feature_type == "continuous":
                            X_col, bad = _densify_continuous(X_col[nonmissings])

                            X_col_tmp = full(nonmissings.shape[0], nan, float64)
                            X_col_tmp[nonmissings] = X_col
                            X_col = X_col_tmp

                            if bad is not None:
                                bad_tmp = zeros(nonmissings.shape[0], bool_)
                                bad_tmp[nonmissings] = bad
                                bad = bad_tmp

                            return (
                                None,
                                None,
                                X_col,
                                bad,
                            )

                        indexes, uniques = _factorize(
                            _densify_categorical(X_col[nonmissings])
                        )
                        return (
                            nonmissings,
                            uniques,
                            indexes,
                            None,
                        )

            elif tt is bool_:

                def internal(feature_idx, feature_type):
                    X_col = X[:, col_map[feature_idx]]

                    if feature_type == "continuous":
                        # force C contiguous here for a later call to native.discretize
                        return None, None, X_col.astype(float64, "C"), None

                    indexes, uniques = _factorize(X_col)
                    return None, where(uniques, "1", "0"), indexes, None

            else:

                def internal(feature_idx, feature_type):
                    X_col = X[:, col_map[feature_idx]]

                    if feature_type == "continuous":
                        try:
                            # force C contiguous here for a later call to native.discretize
                            return None, None, X_col.astype(float64, "C"), None
                        except:  # object conversion can throw any exception in their __float__ or __str__
                            return (
                                None,
                                None,
                                *_process_continuous_strings(X_col, None),
                            )

                    indexes, uniques = _factorize(X_col)
                    return None, uniques.astype(str_, copy=False), indexes, None

            return internal
    elif isinstance(X, _DataFrameType):
        cols = X.columns
        mapping = dict(zip(map(str, cols), cols))
        n_cols = len(cols)
        if len(mapping) != n_cols:
            warn(
                "Columns with duplicate names detected. This can happen for example if there are columns '0' and 0."
            )

            # We can handle duplicate names if they are not being used by the model.
            counter = Counter(map(str, cols))
            for name in compress(counter.keys(), map(_not_one, counter.values())):
                del mapping[name]

        if all(map(mapping.__contains__, feature_names_in)) or all(
            map(
                mapping.__contains__,
                compress(
                    feature_names_in,
                    map("ignore".__ne__, feature_types_ignore),
                ),
            )
        ):
            # we can index by name, which is a lot faster in pandas

            if len(feature_names_in) < n_cols:
                # this warning isn't perfect with ignored features, but it is cheap
                warn("Extra columns present in X that are not used by the model.")

            def internal(feature_idx, feature_type):
                return _process_pandas_column_schematized(
                    X[mapping[feature_names_in[feature_idx]]],
                    feature_type,
                )

            return internal
        else:
            warn(
                "Pandas dataframe X does not contain all feature names. Falling back to positional columns."
            )

            if len(feature_names_in) == n_cols:
                X = X.iloc

                def internal(feature_idx, feature_type):
                    return _process_pandas_column_schematized(
                        X[:, feature_idx],
                        feature_type,
                    )

                return internal
            else:
                keep_cols = fromiter(
                    map("ignore".__ne__, feature_types_ignore),
                    bool_,
                    len(feature_types_ignore),
                )
                n_keep = keep_cols.sum()
                if n_keep != n_cols:
                    msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                    _log.error(msg)
                    raise ValueError(msg)
                col_map = empty(keep_cols.shape[0], int64)
                col_map[keep_cols] = arange(n_keep, dtype=int64)

                X = X.iloc

                def internal(feature_idx, feature_type):
                    return _process_pandas_column_schematized(
                        X[:, col_map[feature_idx]],
                        feature_type,
                    )

                return internal
    elif isinstance(X, _sparray):
        if isinstance(X, _hard_sparse):
            X = X.tocsc()

        n_cols = X.shape[1]
        if len(feature_names_in) == n_cols:

            def internal(feature_idx, feature_type):
                return _process_sparse_column_schematized(
                    X[:, (feature_idx,)],
                    feature_type,
                )

            return internal
        else:
            keep_cols = fromiter(
                map("ignore".__ne__, feature_types_ignore),
                bool_,
                len(feature_types_ignore),
            )
            n_keep = keep_cols.sum()
            if n_keep != n_cols:
                msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(len(feature_types_ignore), int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)

            def internal(feature_idx, feature_type):
                return _process_sparse_column_schematized(
                    X[:, (col_map[feature_idx],)],
                    feature_type,
                )

            return internal
    elif isinstance(X, _spmatrix):
        n_cols = X.shape[1]
        if len(feature_names_in) == n_cols:
            X_getcol = X.getcol

            def internal(feature_idx, feature_type):
                return _process_sparse_column_schematized(
                    X_getcol(feature_idx),
                    feature_type,
                )

            return internal
        else:
            keep_cols = fromiter(
                map("ignore".__ne__, feature_types_ignore),
                bool_,
                len(feature_types_ignore),
            )
            n_keep = keep_cols.sum()
            if n_keep != n_cols:
                msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(len(feature_types_ignore), int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)

            X_getcol = X.getcol

            def internal(feature_idx, feature_type):
                return _process_sparse_column_schematized(
                    X_getcol(col_map[feature_idx]),
                    feature_type,
                )

            return internal
    elif isinstance(X, _SeriesType):
        # TODO: handle as a single feature model
        msg = "X as pandas.Series is unsupported"
        _log.error(msg)
        raise ValueError(msg)
    elif isinstance(X, dict):

        def internal(feature_idx, feature_type):
            X_col = X[feature_names_in[feature_idx]]

            if isinstance(X_col, ndarray):  # this includes ma.masked_array
                pass
            elif isinstance(X_col, _SeriesType):
                # unlike other datasets, dict must be checked for content length
                if n_samples != X_col.shape[0]:
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)

                return _process_pandas_column_schematized(X_col, feature_type)
            elif isinstance(X_col, _DataFrameType):
                if X_col.shape[1] == 1:
                    # unlike other datasets, dict must be checked for content length
                    if n_samples != X_col.shape[0]:
                        msg = "The columns of X are mismatched in the number of of samples"
                        _log.error(msg)
                        raise ValueError(msg)

                    return _process_pandas_column_schematized(
                        X_col.iloc[:, 0], feature_type
                    )
                if X_col.shape[0] == 1:
                    X_col = X_col.to_numpy(object_).ravel()
                elif X_col.shape[1] == 0 or X_col.shape[0] == 0:
                    X_col = empty(0, object_)
                else:
                    msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
            elif isinstance(X_col, _spmatrix_or_sparray):
                if X_col.shape[1] == 1:
                    # unlike other datasets, dict must be checked for content length
                    if n_samples != X_col.shape[0]:
                        msg = "The columns of X are mismatched in the number of of samples"
                        _log.error(msg)
                        raise ValueError(msg)

                    return _process_sparse_column_schematized(X_col, feature_type)
                if X_col.shape[0] == 1:
                    # unlike other datasets, dict must be checked for content length
                    if n_samples != X_col.shape[1]:
                        msg = "The columns of X are mismatched in the number of of samples"
                        _log.error(msg)
                        raise ValueError(msg)

                    return _process_sparse_column_schematized(X_col, feature_type)
                if X_col.shape[1] == 0 or X_col.shape[0] == 0:
                    X_col = empty(0, object_)
                else:
                    msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
            elif isinstance(X_col, _list_tuple_types):
                X_col = np_array(X_col, object_)
            elif isinstance(X_col, _str_bytes_types):
                # isinstance(, str) also works for np.str_

                # don't allow strings to get to the np.array conversion below
                X_col_tmp = empty(1, object_)
                X_col_tmp[0] = X_col
                X_col = X_col_tmp
            else:
                try:
                    # TODO: we need to iterate though all the columns in preclean_X to handle iterable columns

                    # we don't support iterables that get exhausted on their first examination.  This condition
                    # should be detected though in preclean_X where we get the length or bin_native where we check the
                    # number of samples on the 2nd run through the generator
                    X_col = np_array(list(X_col), object_)
                except TypeError:
                    # if our item isn't iterable, assume it has just 1 item and we'll check below if that's consistent
                    X_col_tmp = empty(1, object_)
                    X_col_tmp[0] = X_col
                    X_col = X_col_tmp

            X_col = _reshape_1D_if_possible(X_col)

            # unlike other datasets, dict must be checked for content length
            if n_samples != X_col.shape[0]:
                msg = "The columns of X are mismatched in the number of of samples"
                _log.error(msg)
                raise ValueError(msg)

            tt = X_col.dtype.type
            if isinstance(X_col, masked_array):
                nonmissings = X_col.mask
                if nonmissings is not nomask:
                    # it's legal for a mask to exist and yet have all valid entries in the mask, so check for this
                    if nonmissings.any():
                        nonmissings = ~nonmissings
                        X_col = X_col.compressed()
                        if tt is object_:
                            nonmissings2 = _notna(X_col)

                            if not nonmissings2.all():
                                X_col = X_col[nonmissings2]
                                place(nonmissings, nonmissings, nonmissings2)

                            if feature_type == "continuous":
                                X_col, bad = _densify_continuous(X_col)

                                X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                X_col_tmp[nonmissings] = X_col
                                X_col = X_col_tmp

                                if bad is not None:
                                    bad_tmp = zeros(nonmissings.shape[0], bool_)
                                    bad_tmp[nonmissings] = bad
                                    bad = bad_tmp

                                return (
                                    None,
                                    None,
                                    X_col,
                                    bad,
                                )

                            # feature_type == "nominal" or feature_type == "ordinal"
                            indexes, uniques = _factorize(_densify_categorical(X_col))

                            return (
                                nonmissings,
                                uniques,
                                indexes,
                                None,
                            )
                        else:
                            if feature_type == "continuous":
                                if tt is float64:
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None
                                try:
                                    X_col = X_col.astype(float64)
                                    X_col_tmp = full(nonmissings.shape[0], nan, float64)
                                    X_col_tmp[nonmissings] = X_col
                                    return None, None, X_col_tmp, None
                                except:  # object conversion can throw any exception in their __float__ or __str__
                                    return (
                                        None,
                                        None,
                                        *_process_continuous_strings(
                                            X_col, nonmissings
                                        ),
                                    )

                            # feature_type == "nominal" or feature_type == "ordinal"

                            if issubclass(tt, floating):
                                m = isnan(X_col)
                                if m.any():
                                    logical_not(m, out=m)
                                    X_col = X_col[m]
                                    place(nonmissings, nonmissings, m)

                                indexes, uniques = _factorize(X_col)

                                uniques = (
                                    uniques.astype(float64, copy=False) + 0.0
                                ).astype(str_)
                                wholes = endswith(uniques, ".0")
                                if wholes.any():
                                    uniques[wholes] = rstrip(
                                        rstrip(uniques[wholes], "0"), "."
                                    )

                                return (
                                    nonmissings,
                                    uniques,
                                    indexes,
                                    None,
                                )

                            indexes, uniques = _factorize(X_col)

                            if tt is bool_:
                                return (
                                    nonmissings,
                                    where(uniques, "1", "0"),
                                    indexes,
                                    None,
                                )

                            return (
                                nonmissings,
                                uniques.astype(str_, copy=False),
                                indexes,
                                None,
                            )
                X_col = X_col.data

            if tt is object_:
                nonmissings = _notna(X_col)

                if nonmissings.all():
                    nonmissings = None
                else:
                    X_col = X_col[nonmissings]

                if feature_type == "continuous":
                    X_col, bad = _densify_continuous(X_col)

                    if nonmissings is not None:
                        X_col_tmp = full(nonmissings.shape[0], nan, float64)
                        X_col_tmp[nonmissings] = X_col
                        X_col = X_col_tmp

                        if bad is not None:
                            bad_tmp = zeros(nonmissings.shape[0], bool_)
                            bad_tmp[nonmissings] = bad
                            bad = bad_tmp

                    return (
                        None,
                        None,
                        X_col,
                        bad,
                    )

                # feature_type == "nominal" or feature_type == "ordinal"

                indexes, uniques = _factorize(_densify_categorical(X_col))

                return (
                    nonmissings,
                    uniques,
                    indexes,
                    None,
                )

            if feature_type == "continuous":
                if tt is float64:
                    # force C contiguous here for a later call to native.discretize
                    return None, None, ascontiguousarray(X_col), None
                try:
                    return None, None, X_col.astype(float64, "C"), None
                except:  # object conversion can throw any exception in their __float__ or __str__
                    return None, None, *_process_continuous_strings(X_col, None)

            # feature_type == "nominal" or feature_type == "ordinal"

            if issubclass(tt, floating):
                m = isnan(X_col)
                if m.any():
                    logical_not(m, out=m)
                    X_col = X_col[m]
                    indexes, uniques = _factorize(X_col)

                    uniques = (uniques.astype(float64, copy=False) + 0.0).astype(str_)
                    wholes = endswith(uniques, ".0")
                    if wholes.any():
                        uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                    return m, uniques, indexes, None
                indexes, uniques = _factorize(X_col)

                uniques = (uniques.astype(float64, copy=False) + 0.0).astype(str_)
                wholes = endswith(uniques, ".0")
                if wholes.any():
                    uniques[wholes] = rstrip(rstrip(uniques[wholes], "0"), ".")

                return None, uniques, indexes, None

            indexes, uniques = _factorize(X_col)

            if tt is bool_:
                return None, where(uniques, "1", "0"), indexes, None

            return None, uniques.astype(str_, copy=False), indexes, None

        return internal
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)


def unify_columns_nonschematized(
    X,
    n_samples,
    feature_names_in,
    feature_types,
    min_unique_continuous,
):
    # preclean_X is always called on X prior to calling this function

    # unify_feature_names is always called on feature_names_in prior to calling this function

    # feature_names_in is guranteed not to contain duplicate names because unify_feature_names checks this.

    # clean the feature_types since feature types can contain non-strings
    get_col_schematized = unify_columns_schematized(
        X,
        n_samples,
        feature_names_in,
        [x if x == "ignore" else "" for x in feature_types],
    )

    if isinstance(X, ndarray):  # this includes ma.masked_array
        if issubclass(X.dtype.type, _complex_void_types):
            msg = f"{X.dtype.type} type not supported."
            _log.error(msg)
            raise ValueError(msg)

        # TODO: I'm not sure that simply checking X.flags.c_contiguous handles all the situations that we'd want
        # to know about some data.  If we recieved a transposed array that was C ordered how would that look?
        # so read up on this more
        # https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
        # https://numpy.org/doc/stable/reference/arrays.interface.html
        # memoryview

        if len(feature_names_in) == X.shape[1]:
            col_map = arange(len(feature_names_in), dtype=int64)
        else:
            keep_cols = fromiter(
                map(ne, _repeat_ignore, feature_types),
                bool_,
                len(feature_types),
            )
            n_keep = keep_cols.sum()
            if n_keep != X.shape[1]:
                msg = f"The model has {len(feature_names_in)} features, but X has {X.shape[1]} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(keep_cols.shape[0], int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)

        def internal(feature_idx):
            return _process_numpy_column_nonschematized(
                feature_idx,
                X[:, col_map[feature_idx]],
                feature_types[feature_idx],
                min_unique_continuous,
                get_col_schematized,
            )

        return internal
    elif isinstance(X, _DataFrameType):
        cols = X.columns
        mapping = dict(zip(map(str, cols), cols))
        n_cols = len(cols)
        if len(mapping) != n_cols:
            warn(
                "Columns with duplicate names detected. This can happen for example if there are columns '0' and 0."
            )

            # We can handle duplicate names if they are not being used by the model.
            counter = Counter(map(str, cols))
            for name in compress(counter.keys(), map(_not_one, counter.values())):
                del mapping[name]

        good_names = compress(feature_names_in, map(ne, _repeat_ignore, feature_types))
        if all(map(mapping.__contains__, good_names)):
            # we can index by name, which is a lot faster in pandas

            if len(feature_names_in) < n_cols:
                # this warning isn't perfect with ignored features, but it is cheap
                warn("Extra columns present in X that are not used by the model.")

            def internal(feature_idx):
                return _process_pandas_column_nonschematized(
                    feature_idx,
                    X[mapping[feature_names_in[feature_idx]]],
                    feature_types[feature_idx],
                    min_unique_continuous,
                    get_col_schematized,
                )

            return internal
        else:
            X = X.iloc

            if len(feature_names_in) == n_cols:
                col_map = arange(len(feature_names_in), dtype=int64)
            else:
                keep_cols = fromiter(
                    map(ne, _repeat_ignore, feature_types),
                    bool_,
                    len(feature_types),
                )
                n_keep = keep_cols.sum()
                if n_keep != n_cols:
                    msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                    _log.error(msg)
                    raise ValueError(msg)
                col_map = empty(keep_cols.shape[0], int64)
                col_map[keep_cols] = arange(n_keep, dtype=int64)

            warn(
                "Pandas dataframe X does not contain all feature names. Falling back to positional columns."
            )

            def internal(feature_idx):
                return _process_pandas_column_nonschematized(
                    feature_idx,
                    X[:, col_map[feature_idx]],
                    feature_types[feature_idx],
                    min_unique_continuous,
                    get_col_schematized,
                )

            return internal
    elif isinstance(X, _sparray):
        if isinstance(X, _hard_sparse):
            X = X.tocsc()

        n_cols = X.shape[1]

        if len(feature_names_in) == n_cols:
            col_map = arange(len(feature_names_in), dtype=int64)
        else:
            keep_cols = fromiter(
                map(ne, _repeat_ignore, feature_types),
                bool_,
                len(feature_types),
            )
            n_keep = keep_cols.sum()
            if n_keep != n_cols:
                msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(len(feature_types), int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)

        def internal(feature_idx):
            return _process_arrayish_nonschematized(
                feature_idx,
                X[:, (col_map[feature_idx],)].toarray().ravel(),
                feature_types[feature_idx],
                min_unique_continuous,
                get_col_schematized,
            )

        return internal
    elif isinstance(X, _spmatrix):
        n_cols = X.shape[1]
        X_get = X.getcol

        if len(feature_names_in) == n_cols:
            col_map = arange(len(feature_names_in), dtype=int64)
        else:
            keep_cols = fromiter(
                map(ne, _repeat_ignore, feature_types),
                bool_,
                len(feature_types),
            )
            n_keep = keep_cols.sum()
            if n_keep != n_cols:
                msg = f"The model has {len(feature_names_in)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = empty(len(feature_types), int64)
            col_map[keep_cols] = arange(n_keep, dtype=int64)

        def internal(feature_idx):
            return _process_arrayish_nonschematized(
                feature_idx,
                X_get(col_map[feature_idx]).toarray().ravel(),
                feature_types[feature_idx],
                min_unique_continuous,
                get_col_schematized,
            )

        return internal
    elif isinstance(X, _SeriesType):
        # TODO: handle as a single feature model
        msg = "X as pandas.Series is unsupported"
        _log.error(msg)
        raise ValueError(msg)
    elif isinstance(X, dict):

        def internal(feature_idx):
            feature_type, nonmissings, uniques, X_col, bad = (
                _process_dict_column_nonschematized(
                    feature_idx,
                    X[feature_names_in[feature_idx]],
                    feature_types[feature_idx],
                    min_unique_continuous,
                    get_col_schematized,
                )
            )

            # unlike other datasets, dict must be checked for content length
            if nonmissings is None or nonmissings is False:
                if n_samples != X_col.shape[0]:
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)
            else:
                if n_samples != nonmissings.shape[0]:
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)

            return feature_type, nonmissings, uniques, X_col, bad

        return internal
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)


def _determine_min_cols(feature_names=None, feature_types=None):
    if feature_types is None:
        return None if feature_names is None else len(feature_names)
    n_ignored = sum(
        map(
            "ignore".__eq__,
            compress(feature_types, map(isinstance, feature_types, _repeat_str)),
        )
    )
    if (
        feature_names is None
        or len(feature_names) == len(feature_types)
        or len(feature_names) == len(feature_types) - n_ignored
    ):
        return len(feature_types) - n_ignored
    msg = f"feature_names has length {len(feature_names)} which does not match the length of feature_types {len(feature_types)}"
    _log.error(msg)
    raise ValueError(msg)


def unify_feature_names(X, feature_names=None, feature_types=None):
    # called under: fit

    # If feature_names and feature_types were the outputs of a fit function, then
    # this function is re-callable because it will return the same feature names
    # as previously generated.

    # Passing a None feature_name will cause us to use positional
    # ordering and generate a name, but None values in the X column names
    # will be stringified to "None".

    if isinstance(X, np.ndarray):  # this includes ma.masked_array
        X_names = None
        n_cols = X.shape[1]
    elif isinstance(X, _DataFrameType):
        X_names = list(map(str, X.columns))
        n_cols = len(X_names)
    elif isinstance(X, _SeriesType):
        X_names = None
        n_cols = 1
    elif isinstance(X, _spmatrix_or_sparray):
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

    n_ignored = (
        0 if feature_types is None else sum(1 for t in feature_types if t == "ignore")
    )

    if feature_names is None:
        if feature_types is None:
            feature_types = [None] * n_cols
            if X_names is None:
                feature_names = [None] * n_cols
            else:
                feature_names = X_names
        else:
            if len(feature_types) == n_cols:
                if X_names is None:
                    feature_names = [None] * n_cols
                else:
                    feature_names = X_names
            elif len(feature_types) == n_cols + n_ignored:
                if X_names is None:
                    feature_names = [None] * len(feature_types)
                else:
                    feature_names = []
                    i = 0
                    for t in feature_types:
                        if t == "ignore":
                            feature_names.append(None)
                        else:
                            feature_names.append(X_names[i])
                            i += 1
            else:
                msg = f"There are {len(feature_types)} feature_types, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
    else:
        feature_names = [None if name is None else str(name) for name in feature_names]
        if feature_types is None:
            if X_names is None:
                # ok, need to use position indexing
                if len(feature_names) != n_cols:
                    msg = f"There are {len(feature_names)} features, but X has {n_cols} columns."
                    _log.error(msg)
                    raise ValueError(msg)
            else:
                # we might be indexing by name
                X_names_unique = {
                    name for name, n_count in Counter(X_names).items() if n_count == 1
                }
                if any(name not in X_names_unique for name in feature_names):
                    warn(
                        "Using column positional indexing instead of feature_name indexing because of a naming mismatch."
                    )

                    if len(feature_names) != n_cols:
                        msg = f"There are {len(feature_names)} features, but X has {n_cols} columns."
                        _log.error(msg)
                        raise ValueError(msg)

            feature_types = [None] * len(feature_names)
        else:
            if len(feature_types) == len(feature_names):
                pass
            elif len(feature_types) == len(feature_names) + n_ignored:
                feature_names_clean = []
                i = 0
                for t in feature_types:
                    if t == "ignore":
                        feature_names_clean.append(None)
                    else:
                        feature_names_clean.append(feature_names[i])
                        i += 1
                feature_names = feature_names_clean
            else:
                msg = f"There are {len(feature_types)} feature_types and {len(feature_names)} feature_names which is a mismatch."
                _log.error(msg)
                raise ValueError(msg)

            if X_names is None:
                # ok, need to use position indexing
                if (
                    len(feature_types) != n_cols
                    and len(feature_types) != n_cols + n_ignored
                ):
                    msg = f"There are {len(feature_types)} features, but X has {n_cols} columns."
                    _log.error(msg)
                    raise ValueError(msg)
            else:
                # we might be indexing by name
                X_names_unique = {
                    name for name, n_count in Counter(X_names).items() if n_count == 1
                }
                if any(
                    name not in X_names_unique
                    for name, t in zip(feature_names, feature_types)
                    if t != "ignore"
                ):
                    warn(
                        "Using column positional indexing instead of feature_name indexing because of a naming mismatch."
                    )

                    if (
                        len(feature_types) != n_cols
                        and len(feature_types) != n_cols + n_ignored
                    ):
                        msg = f"There are {len(feature_types)} features, but X has {n_cols} columns."
                        _log.error(msg)
                        raise ValueError(msg)

    all_names = set()
    used_names = set()
    for name, t in zip(feature_names, feature_types):
        if name is not None:
            all_names.add(name)
            if t != "ignore":
                if name in used_names:
                    msg = f"feature name {name} is a duplicate."
                    _log.error(msg)
                    raise ValueError(msg)
                used_names.add(name)

    feature_idx = 0
    for i in range(len(feature_names)):
        name = feature_names[i]
        if name is None:
            while True:
                # give 4 digits to the number so that anything below 9999 gets sorted in the right order in string format
                name = f"feature_{feature_idx:04}"
                feature_idx += 1
                if name not in all_names:
                    break
            feature_names[i] = name

    return feature_names, feature_types


def _reshape_X(X, min_cols, n_samples, sample_source):
    if (
        X.ndim == 0 or X.shape[0] == 0
    ):  # zero dimensional arrays are possible, but really weird
        return empty((0, 0), X.dtype)
    if X.ndim != 1:
        # our caller will not call this function with 2 dimensions
        # we also accept 1 dimension as below, but do not encourage it
        msg = f"X must have 2 dimensions, but has {X.ndim}"
        _log.error(msg)
        raise ValueError(msg)

    if n_samples is not None:
        if n_samples == 1:
            return X.reshape((1, X.shape[0]))
        if n_samples == X.shape[0]:
            return X.reshape((n_samples, 1))
        msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
        _log.error(msg)
        raise ValueError(msg)
    if min_cols is None or min_cols == 1:
        return X.reshape((X.shape[0], 1))
    if min_cols <= X.shape[0]:
        return X.reshape((1, X.shape[0]))
    msg = "X is 1 dimensional"
    _log.error(msg)
    raise ValueError(msg)


def preclean_X(X, feature_names, feature_types, n_samples=None, sample_source="y"):
    # called under: fit or predict
    min_cols = _determine_min_cols(feature_names, feature_types)

    if isinstance(X, ndarray):  # this includes ma.masked_array
        if X.ndim != 2:
            X = _reshape_X(X, min_cols, n_samples, sample_source)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if isinstance(X, _DataFrameType):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if isinstance(X, _spmatrix_or_sparray):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if isinstance(X, _SeriesType):
        if min_cols is not None and min_cols != 1:
            msg = "X cannot be a pandas.Series unless there is only 1 feature"
            _log.error(msg)
            raise ValueError(msg)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if isinstance(X, dict):
        for val in X.values():
            if isinstance(val, ndarray) and val.ndim == 0:
                break
            # we don't support iterators for dict, so len should work
            if n_samples is not None and n_samples != len(val):
                msg = f"{sample_source} has {n_samples} samples, but X has {len(val)}"
                _log.error(msg)
                raise ValueError(msg)
            return X, len(val)
        if n_samples is not None and n_samples != 0:
            msg = f"{sample_source} has {n_samples} samples, but X has 0"
            _log.error(msg)
            raise ValueError(msg)
        return X, 0
    if isinstance(X, _list_tuple_types):
        is_copied = False
    elif callable(getattr(X, "__array__", None)):
        X = X.__array__()
        if X.ndim != 2:
            X = _reshape_X(X, min_cols, n_samples, sample_source)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif X is None:
        msg = "X cannot be None"
        _log.error(msg)
        raise TypeError(msg)
    elif isinstance(X, (str, bytes)):
        # str objects are iterable, so don't allow them to get to the list() conversion below
        # isinstance(, str) also works for np.str_
        msg = "X cannot be a str type"
        _log.error(msg)
        raise TypeError(msg)
    else:
        try:
            X = list(X)
            is_copied = True
        except TypeError:
            msg = "X must be an iterable"
            _log.error(msg)
            raise TypeError(msg)

    # for consistency with what the caller expects, we should mirror what np.array([[..], [..], .., [..]]) does
    # [1, 2, 3] is one sample with 3 features
    # [[1], [2], [3]] is three samples with 1 feature
    # [[1], [2], 3] is bug prone.  You could argue that it has to be a single sample since
    #   the 3 only makes sense in that context, but if the 2 value was removed it would change
    #   from being a single sample with 3 features to being two samples with a single feature,
    #   so force the user to have consistent inner lists/objects

    for idx in range(len(X)):
        sample = X[idx]
        if isinstance(sample, _list_tuple_types):
            pass
        elif isinstance(sample, masked_array):
            # do this before ndarray since ma.masked_array is a subclass of ndarray
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = _reshape_1D_if_possible(
                sample.astype(object_, copy=False).filled(nan)
            )
        elif isinstance(sample, ndarray):
            if sample.ndim == 1:
                pass
            else:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = _reshape_1D_if_possible(sample)
        elif isinstance(sample, _SeriesType):
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = sample.to_numpy(object_)
        elif isinstance(sample, _DataFrameType):
            if sample.shape[1] == 1 or sample.shape[0] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.to_numpy(object_).ravel()
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = empty(0, object_)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif isinstance(sample, _spmatrix_or_sparray):
            if sample.shape[1] == 1 or sample.shape[0] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.toarray().ravel()
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = empty(0, object_)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif callable(getattr(sample, "__array__", None)):
            sample = sample.__array__()
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = _reshape_1D_if_possible(sample)
        elif isinstance(sample, (str, bytes)):
            # isinstance(, str) also works for np.str_
            break  # this only legal if we have one sample
        else:
            try:
                sample = list(sample)
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample
            except TypeError:
                break  # this only legal if we have one sample

    # leave these as object_ for now and we'll try to densify per column where we're more likely to
    # succeed in densification since columns should generally be a single type
    X = np_array(X, object_)
    if X.ndim != 2:
        X = _reshape_X(X, min_cols, n_samples, sample_source)
    if n_samples is not None and n_samples != X.shape[0]:
        msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
        _log.error(msg)
        raise ValueError(msg)
    return X, X.shape[0]
