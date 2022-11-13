# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import math
from collections import Counter
from itertools import count, repeat, groupby
from warnings import warn
import numpy as np
import numpy.ma as ma
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted
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

from ._native import Native
from ._privacy import validate_eps_delta, calc_classic_noise_multi, calc_gdp_noise_multi, private_numeric_binning, private_categorical_binning

# BIG TODO LIST:
#- review this entire bin.py file
#- write a cython single instance prediction pathway
#- consider re-writing most of this bin.py functionality in cython for anything that gets used during prediction for speed
#- test: clean_dimensions with ma.masked_array... and other stuff in there
#- test: clean_X with pd.Series with missing values and maybe a categorical -> gets converted as N features and 1 sample
#- test: clean_X with list that CONTAINS a ma.masked_array sample entry with missing data and without missing data
#- add better processing for ignored columsn where we return the existing data if we can, and we return all None
#  values if not which our caller can detect.  Then unify_data2 can convert that to int(0) values which should work for
#  all feature types
#- disable 'ignore' columns temporarily.  We need to update C++ to make a distinction because you can have 3 real columns and 5 referencable columsn and our datastructures need to be updated to handle this in C++ first
#- handle the thorny questions of converting float to int for categorical strings
#  - in the object converter, convert all int64/uint64 and all floats objects to float64, then use the floor check
#    and compare with +-9007199254740991 to decide if they should be expressed as integers or floats
#  - after np.unique for categoricals, convert int64 and uint64 types to float64 and then re-run np.unique on those
#    values to figure out if there are collisions in the float64 space for integers.  We actually have more
#    work to do in this case since we'll also get bad reverse indexes with more categories than we have unique values
#    Perhaps we can just detect this scenario in the integer space by checking for 9007199254740991 < abs(x) with
#    integers and if it's true then convert to float64 before calling np.unique again?  It'll be infrequent to have
#    such large integers, and we only need to check with int64 and np.uint64 since they are the only ones that can make non-unique floats
#  - leave bools as "False"/"True", BUT we have a corner case in _densify_object_ndarray if we have mixed types
#    we convert to unicode, and bools become "False"/"True" and then subequently fail the test of being able to 
#    be converted to floats, so we need to record the bool types and convert them to 0/1 for the conversion to float
#    test.  First, we can detect if there are any bools via "types = set(map(type, X_col))", then we can 
#    find all the bools with np.logical_or(X_col == np.array(False), X_col == np.array(True)) or something like that
#  - strip leading and trailing spaces when attempting to convert to float BUT NOT FOR STRING CATEGORICALS!
#  - def convert_float_category_str(vals):
#        vals = vals.astype(np.float64, copy=False)
#        integerizable = np.logical_and(vals == np.floor(vals), vals.abs() <= THE_MAX_FLOAT)
#        integers = vals[integerizable]
#        floats = vals[~integerizable]
#        integers = integers.astype(np.int64).astype(np.unicode_)
#        floats = integers.astype(np.unicode_) # or perhaps shuttle it to C++
#        objs = np.empty(len(vals), dtype=np.object)
#        np.place(objs, integerizable, integers)
#        np.place(objs, ~integerizable, floats)
#        vals = objs.astype(np.unicode_)
#        return vals
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


# FUTURE TODOS in our callers and in JSON:
# - look into ISO 6093:1985 -> https://www.titanwolf.org/Network/q/4d680399-6711-4742-9900-74a42ad9f5d7/y
# - support "category compression" where we take a number like 10 and compress any categories together that
#   have less than that number of samples.  Internally, this works well for the prior_categories parameter since
#   we can have multiple strings map to identical numbers, so "low" and "medium" can be groups and separate from high
#   with {"low": 1, "medium": 1, "high":2} and in JSON we can record these as [["low", "medium"], "high"]
#   We support different category compressions for pairs or even individual features since we allow
#   separate category definitios per pair axis.  Our unify_columns generator can support these by extracting the
#   raw data once and then applying different category dictionaries to the raw data and then yielding those
#   the caller to the generator can quickly determine which categories we're responding to using the pointer id(..)
#   comparisons without examining all the internal dictionary definitions, and we can minimize
#   work done by having a single object with a single id(..) pointer that is shared between prior_categories objects
#   if they are identical at model load time.
# - if we recieve an unknown float64 value in a 'nominal' or 'ordinal', then check if all the categorical
#   value strings are convertible to float64.  If that's the case then find the mid-point between the categories
#   after they are converted to strings and create a pseudo-continuous value of the feature and figure out where
#   the previously unseen float64 should go.  WE do need to sort the category strings by float64, but we don't
#   to to compute the split points because we can just do a binary search against the categories after they are
#   converted to floats and then look at the distance between the upper and lower category and choose the one
#   that is closest, and choose the upper one if the distance is equal since then the cut would be on the value
#   and we use lower bound semantics (where the value gets into the upper bin if it's exactly the cut value)
# - eventually, we'll want to have an EBMData data frame that'll store just
#   floats and integers and convert strings to integers on the fly as data is added
#   AND more importantly, you could create this EBMData with a reference to a model
#   and then you could populate it with the correct integer mapping, so "low", "medium", "high"
#   get populated internally as 1, 2, 3 IDENTICALLY to the model from which the
#   EBMData frame was created from.  If we get a dataframe from anywhere else then
#   we can't be confident the mapping is identical, and we need to use a dictionary
#   of some kind, either from string to integer or integer to integer to do the mapping
#   so having our own dataframe makes it possible to have faster prediction scenarios
#   Unfortunately, taking a Pandas dataframe as input doesn't allow us to escape the hashtable
#   step, so whehter we get strings or integers is kind of similar in terms of processing speed
#   although hashing strings is slower.
# - the EBMData frame should be constructable by itself without a model reference if it's going to 
#   be used to train a model, so we sort of have 2 states:
#   - 1: no model reference, convert strings to integers using hashes on the fly
#   - 2: model reference.  Use the model's dictionary mapping initially, but allow new strings or integers
#     to be added as necessary, but anything below what the model knows about we map diretly to the right integers
# - we should create post-model modification routines so someone could construct an integer based
#   ordinal/categorical and build their model and evaluate it efficiently, BUT when they want
#   to view the model they can replace the "1", "2", "3" values with "low", "medium", "high" for graphing




# NOTES:
# - IMPORTANT INFO FOR BELOW: All newer hardware (including all Intel processors) use the IEEE-754 floating point
#   standard when encoding floating point numbers.  In IEEE-754, smaller whole integers have perfect representations 
#   in float64 representation.  Float64 looses the ability to distinquish between integers though above the number 
#   9007199254740991. 9007199254740992 and 9007199254740993 both become 9007199254740992 when converted to float64 
#   and back to ints.  All int32 and uint32 values have perfect float64 representation, but there are collisions
#   for int64 and uint64 values above these high numbers.
# - a desirable property for EBM models is that we can serialize them and evaluate them in different 
#   programming languages like C++, R, JavaScript, etc
# - ideally, we'd have just 1 serialization format, and JSON is a good choice as that format since we can then
#   load models into JavaScript easily, and it's also well supported accross other languages as well.
# - JSON also has the benefit that it's human readable, which is important for an intelligible model.
# - JSON and JavaScript have fairly limited support for data types.  Only strings and float64 numbers are recognized.
#   There are no integer datatypes in JavaScript or JSON.  This works for us though since we can use strings to 
#   encode nominals/ordinals, and float64 values to define 'continuous' cut points.
# - 'continuous' features should always be converted to float64 before discretization because:
#   - float64 is more universal accross programming languages.  Python's float type is a float64.  R only supports
#     float64.  JavaScript is only float64, etc.  GPUs are the excpetion where only float32 are sometimes supported
#     but we only do discretization at the injestion point before any GPUs get used, so that isn't a concern.
#   - our model definition in JSON is exclusively float64, and we don't to add complexity to indicate if a number
#     is a float64 or float32, and even then what would we do with a float32 in JavaScript?
#   - float64 continuous values gives us perfect separation and conversion of float32 values, which isn't true 
#     for the inverse
#   - The long double (float80) equivalent is pretty much dead and new hardware doesn't support it.  In the off
#     chance someone has data with this type then we loose some precision and some values which might have been
#     separable will be lumped together, but for continuous values the cut points are somewhat arbitary anyways, so
#     this is acceptable.
#   - Some big int64 or uint64 values collide when converting to float64 for numbers above 9007199254740991, 
#     so we loose the ability to distinquish them, but like for float80 values 
#     this loss in precision is acceptable since continuous features by nature group similar values together.  
#     The problem is worse for float32, so float64 is better in this regard.
# - 'nominal' and 'ordinal' features are pretty compatible between languages when presented to us as strings
#   but the caller can specify that integer/boolean/float values should be treated as 'nominal'/'ordinal' and 
#   then things become tricky for a number of reasons:
#   - it's pretty easy in python and in other languages to silently convert integers to floats.  Let's say 
#     we have a categorical where the possible values are 1, 2, 3, and 4.1, but 4.1 is very unlikely and might
#     occur zero times in any particular dataset.  If during training our unique values are np.array([1, 2, 3]), 
#     but during predict time let's say we observe np.array([1, 2, 3, 4.1]).  Python will silently convert these to 
#     floats resulting in np.array([1.0, 2.0, 3.0, 4.1]), and then when we convert to strings we get 
#     ["1.0", "2.0", "3.0", "4.1"] instead of our original categories of ["1", "2", "3"], so now none of our 
#     categories match.  This would be a very easy mistake to make and would result in a hard to diagnose bug.
#     A solution to this problem of silently converting integers to floats would be to change our text conversion
#     such that floats which are whole numbers are converted to integers in text.  So then we'd get
#     ["1", "2", "3", "4.1"] as our categories.  We can do this efficiently in python and in many other languages
#     by checking if floor(x) == x for float64 values.  I think it's also nicer visually in graphs of categoricals
#     that any numbers are shown as integers when possible
#   - another benefit of making whole number floats as integers is that integer to string conversions are relatively
#     easy to do cross-language, but floats are almost never converted to identical strings the same way across 
#     languages since there are many legal conversions. 
#     "33.3", "33.299999999999997", "3.3e1", "3.3e+01" are all legal text representations for the float value of 33.3
#   - we have an issue in that all numbers above 9007199254740991 (and in fact some numbers below that) will 
#     be equal to their floor, so will appear to be whole numbers.  We don't want 1.0e300 to be converted 
#     to an integer, so we need some kind of maximum value above which we change to floating point representation
#     Since integers don't exist in JavaScript, we can't really represent all numbers above 9007199254740991
#     with unique categoricals, so we can't have truely cross-platform integers above that value, so it makes
#     sense for us to make all whole numbers equal to or less than 9007199254740991 integers, and any number
#     above that point as a floating point.  This has the disadvantage that some integers above 9007199254740991
#     will have the same categorical strings and be non-separable, but having some collisions in extreme values
#     is probably better than the alternative of getting different categorical strings in different programming
#     languages where integers do not exist.  By making all numbers larger than 9007199254740991 as floating
#     point values, the caller will at least see that we're using exponential float representations instead of
#     integers, so although they may not understand why we switch to float representation above 9007199254740991
#     it will at least be apparent what is happening so they can correct the issues by converting to strings themselves.
#   - The only way we could guarantee that identical float64 values in different programming languages generate
#     the same text would be if we implemented a float to text converter in C++ (the standard library provides no
#     cross platform guarantees), and if we sent our floating point values into C++ for conversion.  This is possible
#     to do because we only care about performance during predict time for this converstion to strings, and at predict
#     time we already know if a feature is nominal/ordinal/continuous, and presumably there aren't too many
#     categories because otherwise the feature wouldn't be very useful, so we can pass the relatively few floating
#     point values into C++ and get back a single string separated by spaces of the text conversions.
#   - if we're presented with an array of np.object_, we can't give a guarantee that unique inputs will generate unique
#     categories since the caller could present us with int(0) and "0", or some object type who's __str__ function
#     generates a "0".  We can't obviously support generalized object types when we serialize to JSON, or any
#     other cross-language model serialization format.
#   - here's an interesting conundrum.  np.float64(np.float32("1.1")) != np.float64("1.1").  Also,
#     np.float64(np.float32(1.1)) gives "1.100000023841858".  The problem here is that the float32 converter finds 
#     the float32 value that is closest to 1.1.  That value is a float though so if you convert that to a float64
#     value all the lower mantissa bits are zeros in the float64 value.  If you take the string "1.1" and convert
#     it to float64 though the converter will find the closest float64 value where the text after the 1.1... isn't
#     required for roundtripping.  That float64 will have non-zero bits in the lower mantissa where the float32
#     value for "1.1" does not, so they are not equal.  This is a problem because if we build an EBM model in one
#     language with a float32 and in annother language with a float64 that is the same value we expect them to have
#     the same nominal or ordinal string, but they don't.  In the language with the float32 value we get "1.1"
#     and in the language with the float64 we get "1.100000023841858" and they don't match.  The solution is to 
#     convert all float32 values to float64 in all languages so that we get "1.100000023841858" in both.  This feels
#     odd since str(my_float32) might give "1.1" so it'll be confusing to the caller, but at least we'll get
#     consistent results.  I think we need to make the assumption that the caller has the same binary float
#     represetation in both langauges.  If that's true then any errors are caused by the caller really since
#     they are presenting slightly different data in both languages.  They should be able to resolve it by using
#     float64 everywhere which should be available in all mainstream languages, unlike float32.
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
#   - Python uses the gold standard for float/string conversion: http://www.netlib.org/fp/dtoa.c
#     https://github.com/python/cpython/blob/main/Python/dtoa.c
#     This code outputs the shortest possible string that uses IEEE 754 "exact rounding" using bankers' rounding 
#     which also guarantees rountrips precicely.  This is great for interpretability.  Unfortunatetly this means
#     that we'll need code in the other languages that generates the same strings and for converting back to floats.
#     Fortunately the python C++ code is available and we can use that to get the exact same conversions and make
#     that available in other languages to call into the C++ to harmonize floating point formats.
#     Python is our premier language and has poor performance if you try to do operations in loops, so we'll
#     force all the other platforms to conform to python specifications.
#   - when we recieve bool values in python we can probably keep the python string representations of "False" and "True".
#     Unlike float64 values, there are just 2 possible bool values and we express them as JavaScript bool items,
#     and with just 2 possible values there are no issues with different
#     hard to standardize string formats.  I like giving the user a little more context of the underlying value in
#     the graphs, and "True", "False" are a bit nicer than "false" and "true" or "FALSE" and "TRUE"
#   - If our caller gives us strings [" a ", "a"] we will consider those to be two separate categories since the caller 
#     could have some requirement to keep these as separate categories.  Eliminating the whitespace makes it impossible
#     for our caller to differentiate these.  If the caller wants these to be the same string then they can preprocess this
#     aspect themselves.
# - np.unique has some issues.  It doesn't like None values.  It considers int(4) and float(4.0) to be identical
#   it sucks in performance with np.object_ arrays since it uses python comparers.  It doesn't call
#   __str__ on objects, so we get collisions if the object later converts to a string that is already a category.
#   If there are many np.nan values, then the uniques array has many np.nan entries!  We've fixed all of these 
#   by filtering out None and np.nan values, and we've converted objects to a strong types
# - If we aren't given a feature type and we get data that is just [0, 1], should we treat this as
#   'nominal' or a 'continuous' value with a split at 0.5?  We'd rather our graphs be bar graphs showing 
#   a bar for 0 and annother bar for 1, which implies nominal, but this has a problem if the
#   feature can rarely be something like 1.1.  Maybe we just never saw a 1.1 in our data even though
#   it can occur.  If this happens then a string label of 1.1 doesn't match '1' and we fail.  If
#   we treated data this way then it wouldn't really be legal for production systems to not
#   specify one of the feature types since an unlikely occurence could produce a nominal type
#   from a continuous type and then fail at predict time.  Our solution is if we see new categories at predict time
#   to check if the new categories are convertible to float64 and if that's true and if all the other prior categories
#   that we saw during fit time are also convertible to float64, then we are allowed to switch to treating them as continuous
#   during predict time.  This way we get to have nice bar graphs of '0' and '1', but we won't generate an error
#   if we see 1.1 at predict time since it gets put into the [0.5 +inf) bin.  We treat
#   [0, 1, 2] and [0, 1, 9] and [1.1, 2.2] the same way and have a threshold of categories below which we treat these
#   as cateogoricals during training.
# - If we recieve pure floats from the caller we'll either generate a continuous feature_type and any differences
#   in the floating point cut points should be fairly minor.  Alternatively, we'll get a 'nominal' which is 
#   also ok since our floating point strings won't match the ones at fit time and then they'll be converted to 
#   continuous values and very likely end up in the same bin as the original floats as they'll be very close in value
#   since we soft-convert nominals with all float64 values into continuous values when necessary/possible
# - Let's say we get the strings ['0', '00', '0.0', '0.0e10'].  If the caller forced this as a nominal we'd have
#   4 values, but if we decided that this should be a 'continuous_auto' value then we'd be converting this to only 
#   one floating point value, which makes it useless.  What this is highlighting is that our unique cutoff point
#   where we choose whether a feature should be 'nominal_auto' or 'continuous_auto' should be decided by the number
#   of unique float64 values that the strings convert into.  Hopefully different platforms get the same floating point
#   values based on string inputs, which is annother reason why we should have a consistent C++ implementation.
# - we use the terms ordinal and nominal to indicate different types of categoricals 
#   (https://en.wikipedia.org/wiki/Ordinal_data).  A lot of ML pacakges use categorical instead of the more 
#   specific term nominal since they don't support ordinals (requiring ordinal data to be handled as 
#   continuous/numerical).  We however, being an interpretable package, want to have a built in oridinal 
#   feature type so that we can display "low", "medium", "high" instead of 1, 2, 3 on graphs, so
#   it makes sense for us to make the distinction of having nominal and ordinal features which are both categoricals
#   This also aligns nicely with the pandas.CategoricalDtype which is used to specify both ordinals and nominals.







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

    is_float_conversion = False
    for one_type in types:
        if one_type is str:
            pass # str objects have __iter__, so special case this to allow
        elif one_type is int:
            pass # int objects use the default __str__ function, so special case this to allow
        elif one_type is bool:
            pass # bool objects use the default __str__ function, so special case this to allow
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
            raise TypeError(msg)
        elif hasattr(one_type, '__iter__') or hasattr(one_type, '__getitem__'):
            # check for __iter__ and __getitem__ to filter out iterables
            # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
            msg = f"X contains the disallowed iterable type {one_type}"
            _log.error(msg)
            raise TypeError(msg)
        elif hasattr(one_type, '__contains__'):
            msg = f"X contains the disallowed set type {one_type}"
            _log.error(msg)
            raise TypeError(msg)
        elif one_type.__str__ is object.__str__:
            # if any object in our list uses the default object __str__ function then it'll
            # include the id(val) pointer in the string text, which isn't going to be useful as a categorical

            # use type(val) instead of val.__str__ to detect inherited __str__ functions per:
            # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object

            msg = f"X contains the type {one_type} which does not define a __str__ function"
            _log.error(msg)
            raise TypeError(msg)

    if is_float_conversion:
        # TODO: handle ints here too which need to be checked if they are larger than the safe int max value

        X_col = X_col.copy()
        places = np.fromiter((val_type is float or issubclass(val_type, np.floating) for val_type in map(type, X_col)), np.bool_, count=len(X_col))
        np.place(X_col, places, X_col[places].astype(np.float64))

    # TODO: converting object types first to pd.CatigoricalDType is somewhat faster than our code here which converts
    # to unicode.  We should consider either using a CatigoricalDTypes conversion first if pandas is installed, or
    # writing our own cython code that can be more efficient at walking through items in an array.  If we write
    # our own cython there is the added advantage that we can check types in the same loop and therefore eliminate
    # the costly "set(map(type, X_col))" calls above
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
    if processing == 'nominal_prevalence':
        if floats is None:
            categories = [(-item[0], item[1]) for item in zip(counts, uniques)]
        else:
            categories = [(-item[0], item[1], item[2]) for item in zip(counts, floats, uniques)]
        categories.sort()
        categories = [x[-1] for x in categories]
    elif processing != 'nominal_alphabetical' and floats is not None:
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        categories.sort()
        categories = [x[1] for x in categories]
    else:
        categories = uniques.tolist()
        categories.sort()

    categories = dict(zip(categories, count(1)))
    mapping = np.fromiter((categories[val] for val in uniques), np.int64, count=len(uniques))
    encoded = mapping[indexes]

    if nonmissings is not None:
        encoded_tmp = np.zeros(len(nonmissings), dtype=np.int64)
        np.place(encoded_tmp, nonmissings, encoded)
        encoded = encoded_tmp

    return encoded, categories

def _encode_categorical_existing(X_col, nonmissings, categories):
    # called under: predict

    # TODO: add special case handling if there is only 1 sample to make that faster
    # if we have just 1 sample, we can avoid making the mapping below

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

    mapping = np.fromiter((categories.get(val, -1) for val in uniques), np.int64, count=len(uniques))
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
            unknowns = encoded < 0
            np.place(bad, unknowns, uniques[indexes[unknowns]])
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
    elif processing == 'nominal_prevalence' or processing == 'nominal_alphabetical':
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
    # if we have just 1 sample, we can avoid making the mapping below

    mapping = np.fromiter((categories.get(val, -1) for val in pd_categories), np.int64, count=len(pd_categories))

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
    elif processing == 'nominal_prevalence' or processing == 'nominal_alphabetical':
        # called under: fit
        X_col, categories = _process_column_initial(X_col, nonmissings, processing, None)
        return 'nominal', X_col, categories, None
    elif processing == 'quantile' or processing == 'rounded_quantile' or processing == 'uniform' or processing == 'winsorized':
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
        mapping = np.empty(len(categories) + 1, np.object_)
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
            raise TypeError(msg)

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
            raise TypeError(msg)

def _reshape_1D_if_possible(col):
    if col.ndim != 1:
        if col.ndim == 0: 
            # 0 dimensional items exist, but are weird/unexpected. len fails, shape is length 0.
            return np.empty(0, col.dtype)

        # ignore dimensions that have just 1 item and assume the intent was to give us 1D
        is_found = False
        for n_items in col.shape:
            if 1 < n_items:
                if is_found:
                    msg = f"Cannot reshape to 1D. Original shape was {col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
                is_found = True
        col = col.reshape(-1)
    return col

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
    if isinstance(X_col.dtype, np.dtype):
        if issubclass(X_col.dtype.type, np.floating) or issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_:
            X_col = X_col.values
            return _process_ndarray(X_col, None, categories, feature_type, min_unique_continuous)
        elif X_col.dtype.type is np.object_:
            nonmissings = None
            if X_col.hasnans:
                # if hasnans is true then there is definetly a real missing value in there and not just a mask
                nonmissings = X_col.notna().values
                X_col = X_col.dropna()
            X_col = X_col.values
            return _process_ndarray(X_col, nonmissings, categories, feature_type, min_unique_continuous)
    elif isinstance(X_col.dtype, pd.CategoricalDtype):
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
    elif issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_:
        # this handles Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype, and BooleanDtype
        nonmissings = None
        if X_col.hasnans:
            # if hasnans is true then there is definetly a real missing value in there and not just a mask
            nonmissings = X_col.notna().values
            X_col = X_col.dropna()
        X_col = X_col.values
        X_col = X_col.astype(dtype=X_col.dtype.type, copy=False)
        return _process_ndarray(X_col, nonmissings, categories, feature_type, min_unique_continuous)

    # TODO: implement pd.SparseDtype
    # TODO: implement pd.StringDtype both the numpy and arrow versions
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.StringDtype.html#pandas.StringDtype
    msg = f"{type(X_col.dtype)} not supported"
    _log.error(msg)
    raise TypeError(msg)

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
        elif X_col.shape[0] == 1:
            X_col = X_col.astype(np.object_, copy=False).values.reshape(-1)
        elif X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif _scipy_installed and isinstance(X_col, sp.sparse.spmatrix):
        if X_col.shape[1] == 1 or X_col.shape[0] == 1:
            return _process_scipy_column(X_col, categories, feature_type, min_unique_continuous)
        elif X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(X_col, list) or isinstance(X_col, tuple):
        X_col = np.array(X_col, np.object_)
    elif isinstance(X_col, str):
        # don't allow strings to get to the np.array conversion below
        X_col_tmp = np.empty(1, np.object_)
        X_col_tmp.itemset(0, X_col)
        X_col = X_col_tmp
    else:
        try:
            # we don't support iterables that get exhausted on their first examination.  This condition
            # should be detected though in clean_X where we get the length or bin_native where we check the
            # number of samples on the 2nd run through the generator
            X_col = list(X_col)
            X_col = np.array(X_col, np.object_)
        except TypeError:
            # if our item isn't iterable, assume it has just 1 item and we'll check below if that's consistent
            X_col_tmp = np.empty(1, np.object_)
            X_col_tmp.itemset(0, X_col)
            X_col = X_col_tmp

    X_col = _reshape_1D_if_possible(X_col)
    return _process_numpy_column(X_col, categories, feature_type, min_unique_continuous)

def unify_columns(X, requests, feature_names_in, feature_types=None, min_unique_continuous=3, go_fast=False):
    # If the requests paramter contains a categories dictionary, then that same categories object is guaranteed to
    # be yielded back to the caller.  This guarantee can be used to rapidly identify which request is being 
    # yielded by using the id(categories) along with the feature_idx

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        if issubclass(X.dtype.type, np.complexfloating):
            msg = "Complex data not supported"
            _log.error(msg)
            raise TypeError(msg)
        elif issubclass(X.dtype.type, np.void):
            msg = "np.void data not supported"
            _log.error(msg)
            raise TypeError(msg)

        # TODO: in the future special case this to make single samples faster at predict time

        n_cols = X.shape[1]
        col_map = None
        if n_cols != len(feature_names_in):
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
            # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need 
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter((val != 'ignore' for val in feature_types), np.bool_, count=len(feature_types))
            if n_cols != keep_cols.sum():
                # called under: predict
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

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
        #if go_fast and X.flags.c_contiguous:
        #    # called under: predict
        #    # during predict we don't care as much about memory consumption, so speed it by transposing everything
        #    X = np.asfortranarray(X)

        for feature_idx, categories in requests:
            col_idx = feature_idx if col_map is None else col_map[feature_idx]
            X_col = X[:, col_idx]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_in, X_col, categories, bad = _process_numpy_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_type_in, X_col, categories, bad
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
            for feature_name_in in feature_names_in:
                if feature_name_in not in names_dict:
                    names_dict = None
                    break
        else:
            for feature_name_in, feature_type in zip(feature_names_in, feature_types):
                if feature_type != 'ignore' and feature_name_in not in names_dict:
                    names_dict = None
                    break

        if names_dict is None:
            if n_cols == len(feature_names_in):
                names_dict = dict(zip(feature_names_in, count()))
            else:
                # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
                # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need 
                # to check for legality on the dimensions of X
                names_dict = dict(zip((feature_name_in for feature_name_in, feature_type in zip(feature_names_in, feature_types) if feature_type != 'ignore'), count()))
                if n_cols != len(names_dict):
                    msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                    _log.error(msg)
                    raise ValueError(msg)

        # Pandas also sometimes uses a dense 2D ndarray instead of per column 1D ndarrays, which would benefit from 
        # transposing, but accessing the BlockManager is currently unsupported behavior. They are also planning to eliminate
        # the BlockManager in Pandas2, so not much benefit in special casing this while they move in that direction
        # https://uwekorn.com/2020/05/24/the-one-pandas-internal.html

        for feature_idx, categories in requests:
            col_idx = names_dict[feature_names_in[feature_idx]]
            X_col = X.iloc[:, col_idx]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_in, X_col, categories, bad = _process_pandas_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_type_in, X_col, categories, bad
    elif _scipy_installed and isinstance(X, sp.sparse.spmatrix):
        n_cols = X.shape[1]

        col_map = None
        if n_cols != len(feature_names_in):
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires 
            # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need 
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter((val != 'ignore' for val in feature_types), np.bool_, count=len(feature_types))
            if n_cols != keep_cols.sum():
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

        for feature_idx, categories in requests:
            col_idx = feature_idx if col_map is None else col_map[feature_idx]
            X_col = X.getcol(col_idx)
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_in, X_col, categories, bad = _process_scipy_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_type_in, X_col, categories, bad
    elif _pandas_installed and isinstance(X, pd.Series):
        # TODO: handle as a single feature model
        msg = "X as pandas.Series is unsupported"
        _log.error(msg)
        raise ValueError(msg)
    elif isinstance(X, dict):
        for feature_idx, categories in requests:
            X_col = X[feature_names_in[feature_idx]]
            feature_type = None if feature_types is None else feature_types[feature_idx]
            feature_type_in, X_col, categories, bad = _process_dict_column(X_col, categories, feature_type, min_unique_continuous)
            yield feature_type_in, X_col, categories, bad
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)

def determine_min_cols(feature_names=None, feature_types=None):
    if feature_types is None:
        return None if feature_names is None else len(feature_names)
    else:
        n_ignored = sum(1 for feature_type in feature_types if feature_type == 'ignore')
        if feature_names is None or len(feature_names) == len(feature_types) or len(feature_names) == len(feature_types) - n_ignored:
            return len(feature_types) - n_ignored
        else:
            msg = f"feature_names has length {len(feature_names)} which does not match the length of feature_types {len(feature_types)}"
            _log.error(msg)
            raise ValueError(msg)

def unify_feature_names(X, feature_names_given=None, feature_types_given=None):
    # called under: fit

    # if feature_names_given and feature_types_given were the outputs of a fit function, then this function
    # is re-callable because it will return the same feature names as previously generated

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        X_names = None
        n_cols = X.shape[1]
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        X_names = list(map(str, X.columns))
        n_cols = len(X_names)
    elif _pandas_installed and isinstance(X, pd.Series):
        X_names = None
        n_cols = 1
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

    n_ignored = 0
    if feature_types_given is not None:
       n_ignored = sum(1 for feature_type_given in feature_types_given if feature_type_given == 'ignore')

    if feature_names_given is None:
        if feature_types_given is not None:
            if len(feature_types_given) != n_cols and len(feature_types_given) != n_cols + n_ignored:
                msg = f"There are {len(feature_types_given)} feature_types, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            n_cols = len(feature_types_given)

        feature_names_in = X_names
        if X_names is None:
            feature_names_in = []
            # this isn't used other than to indicate new names need to be created
            feature_types_given = ['ignore'] * n_cols 
    else:
        n_final = len(feature_names_given)
        if feature_types_given is not None:
            n_final = len(feature_types_given)
            if n_final != len(feature_names_given) and n_final != len(feature_names_given) + n_ignored:
                msg = f"There are {n_final} feature_types and {len(feature_names_given)} feature_names which is a mismatch"
                _log.error(msg)
                raise ValueError(msg)

        feature_names_in = list(map(str, feature_names_given))

        if X_names is None:
            # ok, need to use position indexing
            if n_final != n_cols and n_final != n_cols + n_ignored:
                msg = f"There are {n_final} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
        else:
            # we might be indexing by name
            names_used = feature_names_in
            if feature_types_given is not None and len(feature_names_in) == len(feature_types_given):
                names_used = [feature_name_in for feature_name_in, feature_type_given in zip(feature_names_in, feature_types_given) if feature_type_given != 'ignore']

            X_names_unique = set(name for name, n_count in Counter(X_names).items() if n_count == 1)
            if any(name not in X_names_unique for name in names_used):
                # ok, need to use position indexing
                if n_final != n_cols and n_final != n_cols + n_ignored:
                    msg = f"There are {n_final} features, but X has {n_cols} columns"
                    _log.error(msg)
                    raise ValueError(msg)

    if feature_types_given is not None:
        if len(feature_types_given) == len(feature_names_in):
            if len(feature_names_in) - n_ignored != len(set(feature_name_in for feature_name_in, feature_type_given in zip(feature_names_in, feature_types_given) if feature_type_given != 'ignore')):
                msg = "cannot have duplicate feature names"
                _log.error(msg)
                raise ValueError(msg)

            return feature_names_in

        names_set = set(feature_names_in)

        names = []
        names_idx = 0
        feature_idx = 0
        for feature_type_given in feature_types_given:
            if feature_type_given == 'ignore':
                while True:
                    # non-devs looking at our models will like 1 indexing better than 0 indexing
                    # give 4 digits to the number so that anything below 9999 gets sorted in the right order in string format
                    feature_idx += 1
                    name = f"feature_{feature_idx:04}"
                    if name not in names_set:
                        break
            else:
                name = feature_names_in[names_idx]
                names_idx += 1
            names.append(name)

        feature_names_in = names

    if len(feature_names_in) != len(set(feature_names_in)):
        msg = "cannot have duplicate feature names"
        _log.error(msg)
        raise ValueError(msg)

    return feature_names_in

def clean_dimensions(data, param_name):
    # called under: fit

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
        elif callable(getattr(data, '__array__', None)):
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
                msg = f"{param_name} is not consistent in length for the second dimension"
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
                msg = f"{param_name} is not consistent in length for the second dimension"
                _log.error(msg)
                raise TypeError(msg)
            n_second_dim = 1
            idx = idx + 1
            continue

        if data is not list:
            data = list(data)

        n_items = len(item)
        if n_items == 1:
            data[idx] = item[0] # keep iterating down into them until they hit a non-1 length
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

            item[sub_idx] = subitem[0] # keep iterating down into them until they hit a non-1 length

        data[idx] = item # if it was an iterable or we dug into any of the items, we need to replace it
        idx = idx + 1

    if n_second_dim == 0:
        return np.empty(0, np.object_)

    data = np.array(data, np.object_, copy=False) # in case it was converted to list

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
        if all(one_type is int or issubclass(one_type, np.integer) for one_type in types):
            # the vec.astype call below can fail if we're passed an unsigned np.uint64
            # array with big values, but we don't want to surprise anyone by converting to
            # strings in that special case, so throw if we're presented this unusual type
            dtype = np.int64
        elif all(one_type is bool or issubclass(one_type, np.bool_) for one_type in types):
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
            if n_samples == 1: # then the sample dimension would have been eliminated
                if probs.ndim != 1:
                    msg = f"init_score.predict_proba(X) returned inconsistent number of dimensions"
                    _log.error(msg)
                    raise ValueError(msg)
                if probs.shape[0] <= 1: # 0 or 1 means 1 class
                    # only 1 class to predict means perfect prediction, and no scores for EBMs
                    # do not check if probs are all one in case there is floating point noise
                    return np.empty((1, 0), np.float64)
                probs = probs.reshape([1] + probs.shape)
            else:
                if probs.shape[0] == 0:
                    # having any dimension as zero length probably means 1 class, so treat it that way
                    return np.empty((n_samples, 0), np.float64)
                if probs.shape[0] != n_samples:
                    msg = f"init_score.predict_proba(X) returned inconsistent number of samples compared to y"
                    _log.error(msg)
                    raise ValueError(msg)
                if probs.ndim == 1:
                    # only 1 class to predict means perfect prediction, and no scores for EBMs
                    # do not check if probs are all one in case there is floating point noise
                    return np.empty((n_samples, 0), np.float64)
            probs = probs.astype(np.float64, copy=False)
            maxes = np.amax(probs, axis=1)
            with np.errstate(divide='ignore'):
                init_score = np.log(probs / maxes[:,np.newaxis])
            if init_score.shape[1] == 2: # binary classification
                init_score = init_score[:, 1] - init_score[:, 0]
            return init_score
        except AttributeError:
            init_score = clean_dimensions(init_score.decision_function(X_unclean), "init_score")
            if n_samples == 1: # then the sample dimension would have been eliminated
                if init_score.ndim != 1:
                    msg = f"init_score.decision_function(X) returned inconsistent number of dimensions"
                    _log.error(msg)
                    raise ValueError(msg)
                if init_score.shape[0] != 1:
                    init_score = init_score.reshape([1] + init_score.shape)
            else:
                if init_score.shape[0] == 0:
                    # must be a 1 class problem
                    return np.empty((n_samples, 0), np.float64)
                if init_score.shape[0] != n_samples:
                    msg = f"init_score.decision_function(X) returned inconsistent number of samples compared to y"
                    _log.error(msg)
                    raise ValueError(msg)
            init_score = init_score.astype(np.float64, copy=False)
            return init_score
    elif is_regressor(init_score):
        init_score = clean_dimensions(init_score.predict(X_unclean), "init_score")
        if init_score.ndim != 1:
            msg = f"init_score.predict(X) must have only 1 dimension"
            _log.error(msg)
            raise ValueError(msg)
        if init_score.shape[0] != n_samples:
            msg = f"init_score.predict(X) returned inconsistent number of samples compared to y"
            _log.error(msg)
            raise ValueError(msg)
        init_score = init_score.astype(np.float64, copy=False)

        # TODO Add link function to operate on predict's output when needed

        return init_score

    init_score = clean_dimensions(init_score, "init_score")
    if n_samples == 1: # then the sample dimension would have been eliminated
        if init_score.ndim != 1:
            msg = f"init_score has an inconsistent number of samples compared to y"
            _log.error(msg)
            raise ValueError(msg)
        if init_score.shape[0] != 1:
            init_score = init_score.reshape([1] + init_score.shape)
    else:
        if init_score.shape[0] == 0:
            # must be a 1 class problem
            return np.empty((n_samples, 0), np.float64)
        if init_score.shape[0] != n_samples:
            msg = f"init_score has an inconsistent number of samples compared to y"
            _log.error(msg)
            raise ValueError(msg)
    init_score = init_score.astype(np.float64, copy=False)
    return init_score

def reshape_X(X, min_cols, n_samples, sample_source):
    if X.ndim == 0 or X.shape[0] == 0: # zero dimensional arrays are possible, but really weird
        return np.empty((0, 0), X.dtype)
    elif X.ndim != 1:
        # our caller will not call this function with 2 dimensions
        # we also accept 1 dimension as below, but do not encourage it
        msg = f"X must have 2 dimensions, but has {X.ndim}"
        _log.error(msg)
        raise ValueError(msg)

    if n_samples is not None:
        if n_samples == 1:
            return X.reshape((1, X.shape[0]))
        elif n_samples == X.shape[0]:
            return X.reshape((n_samples, 1))
        else:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
    elif min_cols is None or min_cols == 1:
        return X.reshape((X.shape[0], 1))
    elif min_cols <= X.shape[0]:
        return X.reshape((1, X.shape[0]))
    else:
        msg = f"X is 1 dimensional"
        _log.error(msg)
        raise ValueError(msg)

def clean_X(X, min_cols=None, n_samples=None, sample_source="y"):
    # called under: fit or predict

    if isinstance(X, np.ndarray): # this includes ma.masked_array
        if X.ndim != 2:
            X = reshape_X(X, min_cols, n_samples, sample_source)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif _pandas_installed and isinstance(X, pd.Series):
        if min_cols is not None and min_cols != 1:
            msg = "X cannot be a pandas.Series unless there is only 1 feature"
            _log.error(msg)
            raise ValueError(msg)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif _scipy_installed and isinstance(X, sp.sparse.spmatrix):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif isinstance(X, dict):
        for val in X.values():
            if isinstance(val, np.ndarray) and val.ndim == 0:
                break
            # we don't support iterators for dict, so len should work
            if n_samples is not None and n_samples != len(val):
                msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
                _log.error(msg)
                raise ValueError(msg)
            return X, len(val)
        if n_samples is not None and n_samples != 0:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, 0
    elif isinstance(X, list) or isinstance(X, tuple):
        is_copied = False
    elif callable(getattr(X, '__array__', None)):
        X = X.__array__()
        if X.ndim != 2:
            X = reshape_X(X, min_cols, n_samples, sample_source)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    elif X is None:
        msg = "X cannot be None"
        _log.error(msg)
        raise TypeError(msg)
    elif isinstance(X, str):
        # str objects are iterable, so don't allow them to get to the list() conversion below
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
            if sample.shape[1] == 1 or sample.shape[0] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.astype(np.object_, copy=False).values.reshape(-1)
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = np.empty(0, np.object_)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif _scipy_installed and isinstance(sample, sp.sparse.spmatrix):
            if sample.shape[1] == 1 or sample.shape[0] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.toarray().reshape(-1)
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = np.empty(0, np.object_)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif callable(getattr(sample, '__array__', None)):
            sample = sample.__array__()
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = _reshape_1D_if_possible(sample)
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
    X = np.array(X, np.object_)
    if X.ndim != 2:
        X = reshape_X(X, min_cols, n_samples, sample_source)
    if n_samples is not None and n_samples != X.shape[0]:
        msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
        _log.error(msg)
        raise ValueError(msg)
    return X, X.shape[0]

def _cut_continuous(native, X_col, processing, binning, max_bins, min_samples_bin):
    # called under: fit
      
    if processing != 'quantile' and processing != 'rounded_quantile' and processing != 'uniform' and processing != 'winsorized' and not isinstance(processing, list) and not isinstance(processing, np.ndarray):
        if isinstance(binning, list) or isinstance(binning, np.ndarray):
            msg = f"illegal binning type {binning}"
            _log.error(msg)
            raise ValueError(msg)
        processing = binning

    if processing == 'quantile':
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 0, max_bins - 3)
    elif processing == 'rounded_quantile':
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_quantile(X_col, min_samples_bin, 1, max_bins - 3)
    elif processing == 'uniform':
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_uniform(X_col, max_bins - 3)
    elif processing == 'winsorized':
        # one bin for missing, one bin for unknown, and # of cuts is one less again
        cuts = native.cut_winsorized(X_col, max_bins - 3)
    elif isinstance(processing, np.ndarray):
        cuts = processing.astype(dtype=np.float64, copy=False)
    elif isinstance(processing, list):
        cuts = np.array(processing, np.float64)
    else:
        msg = f"illegal binning type {processing}"
        _log.error(msg)
        raise ValueError(msg)

    return cuts

def normalize_initial_seed(seed):
    # We want cross-language seeds, so we cannot use 64-bit integers since those are not supported
    # on all platforms. Unsigned integers are also not supported on many platforms. Most platforms will
    # either support 32-bit signed integers, or float64 values. Float64 values can represent all 32-bit integers
    # exactly, so limiting our seeds to 32-bit integers allows us to support the widest number of platforms.
    # 
    # If you want cross-platform results, use seeds in the range: -2147483648 <= seed and seed <= 21474836487
    #
    # But, if the platform supports 64-bit integers or for languages like python where integers have no limits, 
    # then we'd rather not throw an exception if something else is provided. We can convert non-portable seeds
    # into something that fits our API by taking modulos to force them into our cross-platform range
    #
    # We use a simple conversion because we use the same method in multiple languages, 
    # and we need to keep the results identical between them, so simplicity is key.
    # 
    # The result of the modulo operator is not standardized accross languages for negative numbers.
    # We can use rounding on divisions to get standardized modulos, and this will work with float64 inputs too.
    # https://torstencurdt.com/tech/posts/modulo-of-negative-numbers

    if seed is None:
        return None # non-deterministic random numbers
    if 2147483647 < seed:
        # we'd like to modulo by 2147483648 but that is not a legal 32-bit signed int
        # if the user provides 2147483647 exactly then we use it, but otherwise our range is restricted
        # add one to prevent generating 0 because we generate 0 for the negative multiples of -2147483648
        return int(round(seed - floor(seed / 2147483647) * 2147483647)) + 1
    if seed < -2147483648:
        # we'd like to modulo by -2147483649 but that is not a legal 32-bit signed int
        # if the user provides -2147483648 exactly then we use it, but otherwise our range is restricted
        return int(round(seed - floor(seed / -2147483648) * -2147483648))
    return seed

def increment_seed(seed):
    # in places where we need to pass seeds, and we want user repetability, like the EBMPreprocessor
    # we can increment the seed instead of generating a random and human-unpredictable seed
    if seed is None:
        return None
    if seed == 2147483647:
        return -2147483648
    return seed + 1

class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self, feature_names=None, feature_types=None, max_bins=256, binning="quantile", min_samples_bin=1, 
        min_unique_continuous=3, epsilon=None, delta=None, composition=None, privacy_schema=None, random_state=None,
    ):
        """ Initializes EBM preprocessor.

        Args:
            feature_names: Feature names as list.
            feature_types: Feature types as list, for example "continuous" or "nominal".
            max_bins: Max number of bins to process numeric features.
            binning: Strategy to compute bins: "quantile", "rounded_quantile", "uniform", or "private". 
            min_samples_bin: minimum number of samples to put into a quantile or rounded_quantile bin
            min_unique_continuous: number of unique numbers required before a feature is considered continuous
            epsilon: Privacy budget parameter. Only applicable when binning is "private".
            delta: Privacy budget parameter. Only applicable when binning is "private".
            privacy_schema: User specified min/max values for numeric features as dictionary. Only applicable when binning is "private".
            random_state: Random state.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.binning = binning
        self.min_samples_bin = min_samples_bin
        self.min_unique_continuous = min_unique_continuous
        self.epsilon = epsilon
        self.delta = delta
        self.composition = composition
        self.privacy_schema = privacy_schema
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """ Fits transformer to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Unused. Only included for scikit-learn compatibility
            sample_weight: Per-sample weights

        Returns:
            Itself.
        """

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        if sample_weight is not None:
            sample_weight = clean_dimensions(sample_weight, "sample_weight")
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1 dimensional")
            if n_samples is not None and n_samples != len(sample_weight):
                msg = f"y has {n_samples} samples and sample_weight has {len(sample_weight)} samples"
                _log.error(msg)
                raise ValueError(msg)
            n_samples = len(sample_weight)
            sample_weight = sample_weight.astype(np.float64, copy=False)

            min_weight = sample_weight.min() # NaN values are guaranteed to be the min if they exist
            # TODO: for now weights of zero are illegal, but in the future accept them
            if math.isnan(min_weight) or min_weight <= 0 or math.isinf(sample_weight.max()):
                msg = "illegal sample_weight value"
                _log.error(msg)
                raise ValueError(msg)

        min_cols = determine_min_cols(self.feature_names, self.feature_types)
        X, n_samples = clean_X(X, min_cols, n_samples, "sample_weight" if y is None else "y")

        # TODO: should preprocessors handle 0 samples?
        if n_samples == 0:
            msg = "X has 0 samples"
            _log.error(msg)
            raise ValueError(msg)

        feature_names_in = unify_feature_names(X, self.feature_names, self.feature_types)
        n_features = len(feature_names_in)

        noise_scale = None # only applicable for private binning
        if self.binning == 'private':
            validate_eps_delta(self.epsilon, self.delta)
            max_weight = 1 if sample_weight is None else np.max(sample_weight)
            if self.composition == 'classic':
                noise_scale = calc_classic_noise_multi(
                    total_queries = n_features, 
                    target_epsilon = self.epsilon, 
                    delta = self.delta, 
                    sensitivity = max_weight
                )
            elif self.composition == 'gdp':
                noise_scale = calc_gdp_noise_multi(
                    total_queries = n_features, 
                    target_epsilon = self.epsilon, 
                    delta = self.delta
                ) * max_weight # Alg Line 17"
            else:
                raise NotImplementedError(f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'.")

        feature_types_in = _none_list * n_features
        bins = _none_list * n_features
        bin_weights = _none_list * n_features
        feature_bounds = np.full((n_features, 2), np.nan, dtype=np.float64)
        histogram_counts = _none_list * n_features
        missing_val_counts = np.zeros(n_features, dtype=np.int64)
        unique_val_counts = np.zeros(n_features, dtype=np.int64)
        zero_val_counts = np.zeros(n_features, dtype=np.int64)

        native = Native.get_native_singleton()
        rng = native.create_rng(normalize_initial_seed(self.random_state))
        is_privacy_warning = False
        for feature_idx, (feature_type_in, X_col, categories, bad) in enumerate(unify_columns(X, zip(range(n_features), repeat(None)), feature_names_in, self.feature_types, self.min_unique_continuous, False)):
            if n_samples != len(X_col):
                msg = "The columns of X are mismatched in the number of of samples"
                _log.error(msg)
                raise ValueError(msg)

            max_bins = self.max_bins # TODO: in the future allow this to be per-feature
            if max_bins < 3:
                raise ValueError(f"max_bins was {max_bins}, but must be 3 or higher. One bin for missing, one bin for unknown, and one or more bins for the non-missing values.")

            if not X_col.flags.c_contiguous:
                # X_col could be a slice that has a stride.  We need contiguous for caling into C
                X_col = X_col.copy()

            feature_types_in[feature_idx] = feature_type_in
            if categories is None:
                # continuous feature
                if bad is not None:
                    msg = f"Feature {feature_names_in[feature_idx]} is indicated as continuous, but has non-numeric data"
                    _log.error(msg)
                    raise ValueError(msg)

                if self.binning == 'private':
                    if np.isnan(X_col).any():
                        msg = "missing values in X not supported for private binning"
                        _log.error(msg)
                        raise ValueError(msg)

                    bounds = None if self.privacy_schema is None else self.privacy_schema.get(feature_idx, None)
                    if bounds is None:
                        is_privacy_warning = True
                        min_feature_val = np.nanmin(X_col)
                        max_feature_val = np.nanmax(X_col)
                    else:
                        min_feature_val = bounds[0]
                        max_feature_val = bounds[1]
                    cuts, feature_bin_weights = private_numeric_binning(X_col, sample_weight, noise_scale, max_bins - 1, min_feature_val, max_feature_val, rng)
                    feature_bin_weights.append(0)
                    feature_bin_weights = np.array(feature_bin_weights, np.float64)
                else:
                    min_feature_val = np.nanmin(X_col)
                    max_feature_val = np.nanmax(X_col)
                    feature_type_given = None if self.feature_types is None else self.feature_types[feature_idx]
                    cuts = _cut_continuous(native, X_col, feature_type_given, self.binning, max_bins, self.min_samples_bin)
                    bin_indexes = native.discretize(X_col, cuts)
                    feature_bin_weights = np.bincount(bin_indexes, weights=sample_weight, minlength=len(cuts) + 3)
                    feature_bin_weights = feature_bin_weights.astype(np.float64, copy=False)

                    n_cuts = native.get_histogram_cut_count(X_col)
                    histogram_cuts = native.cut_uniform(X_col, n_cuts)
                    bin_indexes = native.discretize(X_col, histogram_cuts)
                    feature_histogram_counts = np.bincount(bin_indexes, minlength=len(histogram_cuts) + 3)
                    feature_histogram_counts = feature_histogram_counts.astype(np.int64, copy=False)

                    histogram_counts[feature_idx] = feature_histogram_counts

                    n_missing = len(X_col)
                    X_col = X_col[~np.isnan(X_col)]
                    n_missing = n_missing - len(X_col)
                    missing_val_counts.itemset(feature_idx, n_missing)
                    unique_val_counts.itemset(feature_idx, len(np.unique(X_col)))
                    zero_val_counts.itemset(feature_idx, len(X_col) - np.count_nonzero(X_col))

                bins[feature_idx] = cuts
                feature_bounds.itemset((feature_idx, 0), min_feature_val)
                feature_bounds.itemset((feature_idx, 1), max_feature_val)
            else:
                # categorical feature
                if bad is not None:
                    msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                    _log.error(msg)
                    raise ValueError(msg)

                if self.binning == 'private':
                    if np.count_nonzero(X_col) != len(X_col):
                        msg = "missing values in X not supported for private binning"
                        _log.error(msg)
                        raise ValueError(msg)

                    # TODO: clean up this hack that uses strings of the indexes
                    keep_bins, old_feature_bin_weights = private_categorical_binning(X_col, sample_weight, noise_scale, max_bins - 1, rng)
                    unknown_weight = 0
                    if keep_bins[-1] == 'DPOther':
                        unknown_weight = old_feature_bin_weights[-1]
                        keep_bins = keep_bins[:-1]
                        old_feature_bin_weights = old_feature_bin_weights[:-1]

                    keep_bins = keep_bins.astype(np.int64)
                    keep_bins = dict(zip(keep_bins, old_feature_bin_weights))

                    feature_bin_weights = np.empty(len(keep_bins) + 2, np.float64)
                    feature_bin_weights[0] = 0
                    feature_bin_weights[-1] = unknown_weight

                    categories = list(map(tuple, map(reversed, categories.items())))
                    categories.sort() # groupby requires sorted data

                    new_categories = {}
                    new_idx = 1
                    for idx, category_iter in groupby(categories, lambda x: x[0]):
                        bin_weight = keep_bins.get(idx, None)
                        if bin_weight is not None:
                            feature_bin_weights.itemset(new_idx, bin_weight)
                            for _, category in category_iter:
                                new_categories[category] = new_idx
                            new_idx += 1

                    categories = new_categories
                else:
                    n_unique_indexes = 0 if len(categories) == 0 else max(categories.values())
                    feature_bin_weights = np.bincount(X_col, weights=sample_weight, minlength=n_unique_indexes + 2)
                    feature_bin_weights = feature_bin_weights.astype(np.float64, copy=False)

                    feature_histogram_counts = np.bincount(X_col, minlength=n_unique_indexes + 2)
                    feature_histogram_counts = feature_histogram_counts.astype(np.int64, copy=False)
                    histogram_counts[feature_idx] = feature_histogram_counts

                    missing_val_counts.itemset(feature_idx, len(X_col) - np.count_nonzero(X_col))
                    unique_val_counts.itemset(feature_idx, len(categories))
                    zero_indexes = _none_list * n_unique_indexes
                    for category, idx in categories.items():
                        try:
                            val = float(category)
                            if val == 0.0:
                                zero_indexes[idx - 1] = True
                        except ValueError:
                            pass

                    n_zeros = 0
                    for idx, is_zero in enumerate(zero_indexes):
                        if is_zero:
                            n_zeros += np.count_nonzero(X_col == (idx + 1))
                    zero_val_counts.itemset(feature_idx, n_zeros)

                bins[feature_idx] = categories
            bin_weights[feature_idx] = feature_bin_weights

        if is_privacy_warning:
            warn("Possible privacy violation: assuming min/max values per feature are public info. "
                    "Pass a privacy schema with known public ranges per feature to avoid this warning.")

        self.feature_names_in_ = feature_names_in
        self.feature_types_in_ = feature_types_in
        self.bins_ = bins
        self.bin_weights_ = bin_weights
        self.noise_scale_ = noise_scale
        self.feature_bounds_ = feature_bounds
        self.histogram_counts_ = histogram_counts
        self.missing_val_counts_ = missing_val_counts
        self.unique_val_counts_ = unique_val_counts
        self.zero_val_counts_ = zero_val_counts
        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        min_cols = determine_min_cols(self.feature_names_in_, self.feature_types_in_)

        X, n_samples = clean_X(X, min_cols)

        X_binned = np.empty((n_samples, len(self.feature_names_in_)), np.int64, order='F')

        if 0 < n_samples:
            native = Native.get_native_singleton()
            category_iter = (category if isinstance(category, dict) else None for category in self.bins_)
            requests = zip(count(), category_iter)
            cols = unify_columns(X, requests, self.feature_names_in_, self.feature_types_in_, None, False)
            for feature_idx, bins, (_, X_col, _, _) in zip(count(), self.bins_, cols):
                if n_samples != len(X_col):
                    msg = "The columns of X are mismatched in the number of of samples"
                    _log.error(msg)
                    raise ValueError(msg)

                if not isinstance(bins, dict):
                    # continuous feature

                    if not X_col.flags.c_contiguous:
                        # X_col could be a slice that has a stride.  We need contiguous for caling into C
                        X_col = X_col.copy()

                    X_col = native.discretize(X_col, bins)

                X_binned[:, feature_idx] = X_col

        return X_binned

    def fit_transform(self, X, y=None, sample_weight=None):
        """ Fits and Transform on provided samples.

        Args:
            X: Numpy array for samples.
            y: Unused. Only included for scikit-learn compatibility
            sample_weight: Per-sample weights

        Returns:
            Transformed numpy array.
        """

        n_samples = None
        if y is not None:
            y = clean_dimensions(y, "y")
            if y.ndim != 1:
                raise ValueError("y must be 1 dimensional")
            n_samples = len(y)

        min_cols = determine_min_cols(self.feature_names, self.feature_types)
        X, _ = clean_X(X, min_cols, n_samples) # materialize any iterators first
        return self.fit(X, y, sample_weight).transform(X)

def _deduplicate_bins(bins):
    # calling this function before calling score_terms allows score_terms to operate more efficiently since it'll
    # be able to avoid re-binning data for pairs that have already been processed in mains or other pairs since we 
    # use the id of the bins to identify feature data that was previously binned

    uniques = dict()
    for feature_idx in range(len(bins)):
        bin_levels = bins[feature_idx]
        highest_key = None
        for level_idx, feature_bins in enumerate(bin_levels):
            if isinstance(feature_bins, dict):
                key = frozenset(feature_bins.items())
            else:
                key = tuple(feature_bins)
            existing = uniques.get(key, None)
            if existing is None:
                uniques[key] = feature_bins
            else:
                bin_levels[level_idx] = existing

            if highest_key != key:
                highest_key = key
                highest_idx = level_idx
        del bin_levels[highest_idx + 1:]

def construct_bins(
    X,
    y,
    sample_weight,
    feature_names_given, 
    feature_types_given, 
    max_bins_leveled, 
    binning='quantile', 
    min_samples_bin=1, 
    min_unique_continuous=3, 
    epsilon=None, 
    delta=None, 
    composition=None, 
    privacy_schema=None,
    random_state=None,
):
    is_mains = True
    native = Native.get_native_singleton()
    for max_bins in max_bins_leveled:
        preprocessor = EBMPreprocessor(
            feature_names_given, 
            feature_types_given, 
            max_bins, 
            binning, 
            min_samples_bin, 
            min_unique_continuous, 
            epsilon, 
            delta, 
            composition, 
            privacy_schema,
            random_state,
        )

        random_state = increment_seed(random_state)

        preprocessor.fit(X, y, sample_weight)
        if is_mains:
            is_mains = False
            bins = preprocessor.bins_
            for feature_idx in range(len(bins)):
                bins[feature_idx] = [bins[feature_idx]]

            feature_names_in = preprocessor.feature_names_in_
            feature_types_in = preprocessor.feature_types_in_
            bin_weights = preprocessor.bin_weights_
            feature_bounds = preprocessor.feature_bounds_
            histogram_counts = preprocessor.histogram_counts_
            missing_val_counts = preprocessor.missing_val_counts_
            unique_val_counts = preprocessor.unique_val_counts_
            zero_val_counts = preprocessor.zero_val_counts_
        else:
            if feature_names_in != preprocessor.feature_names_in_:
                raise RuntimeError("Mismatched feature_names")
            if feature_types_in != preprocessor.feature_types_in_:
                raise RuntimeError("Mismatched feature_types")
            if len(bins) != len(preprocessor.bins_):
                raise RuntimeError("Mismatched bin lengths")

            for bin_levels, feature_bins in zip(bins, preprocessor.bins_):
                bin_levels.append(feature_bins)

    _deduplicate_bins(bins)
    return feature_names_in, feature_types_in, bins, bin_weights, feature_bounds, histogram_counts, missing_val_counts, unique_val_counts, zero_val_counts


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

    n_weights = 0 if sample_weight is None else 1

    n_bytes = native.measure_dataset_header(len(requests), n_weights, 1)
    for (feature_idx, feature_bins), (_, X_col, _, bad) in zip(responses, unify_columns(X, requests, feature_names_in, feature_types_in, None, False)):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if not X_col.flags.c_contiguous:
            # X_col could be a slice that has a stride.  We need contiguous for caling into C
            X_col = X_col.copy()

        if isinstance(feature_bins, dict):
            # categorical feature
            n_bins = 1 if len(feature_bins) == 0 else (max(feature_bins.values()) + 1)
        else:
            # continuous feature
            X_col = native.discretize(X_col, feature_bins)
            n_bins = len(feature_bins) + 2

        if bad is not None:
            n_bins += 1
            X_col[bad != _none_ndarray] = n_bins - 1

        n_bytes += native.measure_feature(
            n_bins, 
            np.count_nonzero(X_col) != len(X_col), 
            bad is not None, 
            feature_types_in[feature_idx] == 'nominal', 
            X_col
        )

    if sample_weight is not None:
        n_bytes += native.measure_weight(sample_weight)
    
    if 0 <= n_classes:
        n_bytes += native.measure_classification_target(n_classes, y)
    else:
        n_bytes += native.measure_regression_target(y)

    dataset = np.empty(n_bytes, np.ubyte) # joblib loky doesn't support RawArray

    native.fill_dataset_header(len(requests), n_weights, 1, dataset)

    for (feature_idx, feature_bins), (_, X_col, _, bad) in zip(responses, unify_columns(X, requests, feature_names_in, feature_types_in, None, False)):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        if not X_col.flags.c_contiguous:
            # X_col could be a slice that has a stride.  We need contiguous for caling into C
            X_col = X_col.copy()

        if isinstance(feature_bins, dict):
            # categorical feature
            n_bins = 1 if len(feature_bins) == 0 else (max(feature_bins.values()) + 1)
        else:
            # continuous feature
            X_col = native.discretize(X_col, feature_bins)
            n_bins = len(feature_bins) + 2

        if bad is not None:
            n_bins += 1
            X_col[bad != _none_ndarray] = n_bins - 1

        native.fill_feature(
            n_bins, 
            np.count_nonzero(X_col) != len(X_col), 
            bad is not None, 
            feature_types_in[feature_idx] == 'nominal', 
            X_col, 
            dataset
        )

    if sample_weight is not None:
        native.fill_weight(sample_weight, dataset)

    if 0 <= n_classes:
        native.fill_classification_target(n_classes, y, dataset)
    else:
        native.fill_regression_target(y, dataset)

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

def unify_data2(X, n_samples, feature_names=None, feature_types=None, missing_data_allowed=False, min_unique_continuous=3):
    _log.info("Unifying data")

    if n_samples == 0:
        # for some callers this might be legal if they've been fitted before, but we'll let the caller
        # decide how they want to handle this condition
        msg = "X has 0 samples"
        _log.error(msg)
        raise ValueError(msg)

    # if feature_names_in and feature_types_in were generated in a call to fit(..) then unify_feature_names
    # and unify_columns will return the identical names and types
    feature_names_in = unify_feature_names(X, feature_names, feature_types)
    feature_types_in = _none_list * len(feature_names_in)

    # TODO: this could be made more efficient by storing continuous and categorical values in separate numpy arrays
    # and merging afterwards.  Categoricals are going to share the same objects, but we don't want object
    # fragmentation for continuous values which generates a lot of garbage to collect later
    X_unified = np.empty((n_samples, len(feature_names_in)), np.object_, order='F')

    for feature_idx, (feature_type_in, X_col, categories, bad) in enumerate(unify_columns(X, zip(range(len(feature_names_in)), repeat(None)), feature_names_in, feature_types, min_unique_continuous, False)):
        if n_samples != len(X_col):
            msg = "The columns of X are mismatched in the number of of samples"
            _log.error(msg)
            raise ValueError(msg)

        feature_types_in[feature_idx] = feature_type_in
        if categories is None:
            # continuous feature
            if bad is not None:
                msg = f"Feature {feature_names_in[feature_idx]} is indicated as continuous, but has non-numeric data"
                _log.error(msg)
                raise ValueError(msg)

            if not missing_data_allowed and np.isnan(X_col).any():
                msg = f"X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            X_unified[:, feature_idx] = X_col
        else:
            # categorical feature
            if bad is not None:
                msg = f"Feature {feature_names_in[feature_idx]} has unrecognized ordinal values"
                _log.error(msg)
                raise ValueError(msg)

            if not missing_data_allowed and np.count_nonzero(X_col) != len(X_col):
                msg = f"X cannot contain missing values"
                _log.error(msg)
                raise ValueError(msg)

            mapping = np.empty(len(categories) + 1, np.object_)
            mapping.itemset(0, np.nan)
            for category, idx in categories.items():
                mapping.itemset(idx, category)
            X_unified[:, feature_idx] = mapping[X_col]

    return X_unified, feature_names_in, feature_types_in
