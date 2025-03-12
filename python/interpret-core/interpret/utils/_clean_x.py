# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging
from warnings import warn
from collections import Counter
from itertools import count, repeat, compress
from operator import ne, truth, eq

import numpy as np
from numpy import ma

from ._misc import safe_isinstance

_log = logging.getLogger(__name__)

try:
    import pandas as pd

    _pandas_installed = True
except ImportError:
    _pandas_installed = False

# BIG TODO LIST:
# - review this entire bin.py file
# - write a cython single instance prediction pathway
# - consider re-writing most of this bin.py functionality in cython for anything that gets used during prediction for speed
# - test: clean_dimensions with ma.masked_array... and other stuff in there
# - test: preclean_X with pd.Series with missing values and maybe a categorical -> gets converted as N features and 1 sample
# - test: preclean_X with list that CONTAINS a ma.masked_array sample entry with missing data and without missing data
# - add better processing for ignored columsn where we return the existing data if we can, and we return all None
#  values if not which our caller can detect.  Then unify_data can convert that to int(0) values which should work for
#  all feature types
# - disable 'ignore' columns temporarily.  We need to update C++ to make a distinction because you can have 3 real columns and 5 referencable columsn and our datastructures need to be updated to handle this in C++ first
# - handle the thorny questions of converting float to int for categorical strings
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
# - if we recieve an unseen float64 value in a 'nominal' or 'ordinal', then check if all the categorical
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
#   - Python uses the gold standard for float/string conversion: https://www.netlib.org/fp/dtoa.c
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


_disallowed_types = frozenset(
    [
        complex,
        list,
        tuple,
        range,
        bytes,
        bytearray,
        memoryview,
        set,
        frozenset,
        dict,
        Ellipsis,
        np.csingle,
        np.complex128,
        np.clongdouble,
        np.void,
    ]
)
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
            return X_col.astype(np.str_)
        if bool in types:
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
            # we must have a big number that can only be represented by np.uint64
            # AND also signed integers mixed together if we do X_col.astype(np.uint64),
            # it will silently convert negative integers to unsigned!

            # TODO : should this be np.float64 with a check for big integers
            return X_col.astype(np.str_)

    if all(
        one_type is float or issubclass(one_type, np.floating) for one_type in types
    ):
        if all(one_type is np.float16 for one_type in types):
            return X_col.astype(np.float16)
        types.discard(np.float16)

        if all(one_type is np.float32 for one_type in types):
            return X_col.astype(np.float32)

        return X_col.astype(np.float64)

    is_float_conversion = False
    for one_type in types:
        if one_type is str:
            pass  # str objects have __iter__, so special case this to allow
        elif one_type is int:
            pass  # int objects use the default __str__ function, so special case this to allow
        elif one_type is bool:
            pass  # bool objects use the default __str__ function, so special case this to allow
        elif one_type is float:
            is_float_conversion = (
                True  # force to np.float64 to guarantee consistent string formatting
            )
        elif issubclass(one_type, np.generic):
            # numpy objects have __getitem__, so special case this to allow
            if one_type is np.float64:
                pass  # np.float64 is what we convert to for floats, so no need to convert this
            elif issubclass(one_type, np.floating):
                is_float_conversion = True  # force to np.float64 to ensure consistent string formatting of floats
        elif one_type in _disallowed_types:
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

            # use type(val) instead of val.__str__ to detect inherited __str__ functions per:
            # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object

            msg = f"X contains the type {one_type} which does not define a __str__ function"
            _log.error(msg)
            raise TypeError(msg)

    if is_float_conversion:
        # TODO: handle ints here too which need to be checked if they are larger than the safe int max value

        X_col = X_col.copy()
        places = np.fromiter(
            map(isinstance, X_col, repeat(float)), np.bool_, count=len(X_col)
        )
        places |= np.fromiter(
            map(issubclass, map(type, X_col), repeat(np.floating)),
            np.bool_,
            count=len(X_col),
        )
        np.place(X_col, places, X_col[places].astype(np.float64))

    # TODO: converting object types first to pd.CatigoricalDType is somewhat faster than our code here which converts
    # to unicode.  We should consider either using a CatigoricalDTypes conversion first if pandas is installed, or
    # writing our own cython code that can be more efficient at walking through items in an array.  If we write
    # our own cython there is the added advantage that we can check types in the same loop and therefore eliminate
    # the costly "set(map(type, X_col))" calls above
    return X_col.astype(np.str_)


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
        uniques = floats.astype(np.str_)
    else:
        uniques = uniques.astype(np.str_, copy=False)
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
            floats = floats[indexes]  # expand from the unique floats to expanded floats
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
    if processing == "nominal_prevalence":
        if floats is None:
            categories = [(-item[0], item[1]) for item in zip(counts, uniques)]
        else:
            categories = [
                (-item[0], item[1], item[2]) for item in zip(counts, floats, uniques)
            ]
        categories.sort()
        categories = [x[-1] for x in categories]
    elif processing != "nominal_alphabetical" and floats is not None:
        categories = [(item[0], item[1]) for item in zip(floats, uniques)]
        categories.sort()
        categories = [x[1] for x in categories]
    else:
        categories = uniques.tolist()
        categories.sort()

    categories = dict(zip(categories, count(1)))
    mapping = np.fromiter(
        map(categories.__getitem__, uniques), np.int64, count=len(uniques)
    )
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
    uniques = uniques.astype(np.str_, copy=False)

    mapping = np.fromiter(
        map(categories.get, uniques, repeat(-1)), np.int64, count=len(uniques)
    )
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
            unseens = encoded < 0
            np.place(bad, unseens, uniques[indexes[unseens]])
    else:
        bad = None
        if nonmissings is not None:
            encoded_tmp = np.zeros(len(nonmissings), dtype=np.int64)
            np.place(encoded_tmp, nonmissings, encoded)
            encoded = encoded_tmp

    return encoded, bad


def _encode_pandas_categorical_initial(X_col, pd_categories, is_ordered, processing):
    # called under: fit

    if processing == "nominal":
        if is_ordered:
            msg = "nominal type invalid for ordered pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
    elif processing == "ordinal":
        if not is_ordered:
            msg = "ordinal type invalid for unordered pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
    elif processing is None or processing == "auto":
        pass
    elif processing in ("nominal_prevalence", "nominal_alphabetical"):
        # TODO: we could instead handle this by re-ordering the pandas pd_categories.
        # Someone might want to construct it quickly but then override the pd_categories
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
                elif isinstance(item, (float, int, np.floating, np.integer)):
                    n_continuous += 1
        except TypeError:
            msg = f"{processing} type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)

        if n_continuous == n_items:
            msg = "continuous type invalid for pandas.CategoricalDtype"
            _log.error(msg)
            raise ValueError(msg)
        if n_ordinals == n_items:
            if not is_ordered:
                msg = "ordinal type invalid for unordered pandas.CategoricalDtype"
                _log.error(msg)
                raise ValueError(msg)

            # TODO: instead of throwing, we could match the ordinal values with the pandas pd_categories and
            # report the rest as bad items.  For now though, just assume it's bad to specify this
            msg = "cannot specify ordinal categories for a pandas.CategoricalDtype which already has categories"
            _log.error(msg)
            raise ValueError(msg)
        msg = f"{processing} type invalid for pandas.CategoricalDtype"
        _log.error(msg)
        raise ValueError(msg)

    categories = dict(zip(pd_categories, count(1)))
    # we'll need int64 for calling C++ anyways
    X_col = X_col.astype(dtype=np.int64, copy=False)
    X_col = X_col + 1
    return X_col, categories


def _encode_pandas_categorical_existing(X_col, pd_categories, categories):
    # called under: predict

    # TODO: add special case handling if there is only 1 sample to make that faster
    # if we have just 1 sample, we can avoid making the mapping below

    mapping = np.fromiter(
        map(categories.get, pd_categories, repeat(-1)),
        np.int64,
        count=len(pd_categories),
    )

    if len(mapping) <= len(categories):
        mapping_cmp = np.arange(1, len(mapping) + 1, dtype=np.int64)
        if np.array_equal(mapping, mapping_cmp):
            # avoid overflows for np.int8
            X_col = X_col.astype(dtype=np.int64, copy=False)
            X_col = X_col + 1
            return X_col, None
    else:
        mapping_cmp = np.arange(1, len(categories) + 1, dtype=np.int64)
        if np.array_equal(mapping[0 : len(mapping_cmp)], mapping_cmp):
            unseens = len(categories) <= X_col
            bad = np.full(len(X_col), None, dtype=np.object_)
            bad[unseens] = pd_categories[X_col[unseens]]
            # avoid overflows for np.int8
            X_col = X_col.astype(dtype=np.int64, copy=False)
            X_col = X_col + 1
            X_col[unseens] = -1
            return X_col, bad

    mapping = np.insert(mapping, 0, 0)
    encoded = mapping[X_col + 1]

    bad = None
    unseens = encoded < 0
    if unseens.any():
        bad = np.full(len(X_col), None, dtype=np.object_)
        bad[unseens] = pd_categories[X_col[unseens]]

    return encoded, bad


def _process_continuous(X_col, nonmissings):
    # called under: fit or predict

    if issubclass(X_col.dtype.type, np.floating):
        X_col = X_col.astype(dtype=np.float64, copy=False)
        return X_col, None
    if issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_:
        X_col = X_col.astype(dtype=np.float64)
        if nonmissings is not None:
            X_col_tmp = np.full(len(nonmissings), np.nan, dtype=np.float64)
            np.place(X_col_tmp, nonmissings, X_col)
            X_col = X_col_tmp

        return X_col, None
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
            # slice one item at a time keeping as an np.ndarray
            one_item_array = X_col[idx : idx + 1]
            try:
                # use .astype(..) instead of float(..) to ensure identical conversion results
                floats[idx] = one_item_array.astype(dtype=np.float64)[0]
            except TypeError:
                # use .astype instead of str(one_item_array) here to ensure identical string categories
                one_str_array = one_item_array.astype(dtype=np.str_)
                try:
                    # use .astype(..) instead of float(..) to ensure identical conversion results
                    floats[idx] = one_str_array.astype(dtype=np.float64)[0]
                except ValueError:
                    bad[idx] = one_str_array.item()
            except ValueError:
                bad[idx] = one_item_array.item()

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
    if processing == "continuous":
        # called under: fit or predict
        X_col, bad = _process_continuous(X_col, nonmissings)
        return "continuous", X_col, None, bad
    if processing == "nominal":
        if categories is None:
            # called under: fit
            X_col, categories = _process_column_initial(X_col, nonmissings, None, None)
            return "nominal", X_col, categories, None
        # called under: predict
        X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
        return "nominal", X_col, categories, bad
    if processing == "ordinal":
        if categories is None:
            # called under: fit
            # if the caller passes "ordinal" during fit, the only order that makes sense is either
            # alphabetical or based on float values. Frequency doesn't make sense
            # if the caller would prefer an error, they can check feature_types themselves
            X_col, categories = _process_column_initial(X_col, nonmissings, None, None)
            return "ordinal", X_col, categories, None
        # called under: predict
        X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
        return "ordinal", X_col, categories, bad
    if processing is None or processing == "auto":
        # called under: fit
        X_col, categories = _process_column_initial(
            X_col, nonmissings, None, min_unique_continuous
        )
        return (
            "continuous" if categories is None else "nominal",
            X_col,
            categories,
            None,
        )
    if processing in ("nominal_prevalence", "nominal_alphabetical"):
        # called under: fit
        X_col, categories = _process_column_initial(
            X_col, nonmissings, processing, None
        )
        return "nominal", X_col, categories, None
    if processing in ("quantile", "rounded_quantile", "uniform", "winsorized"):
        # called under: fit
        X_col, bad = _process_continuous(X_col, nonmissings)
        return "continuous", X_col, None, bad
    if isinstance(processing, int):
        # called under: fit
        X_col, categories = _process_column_initial(
            X_col, nonmissings, None, processing
        )
        return (
            "continuous" if categories is None else "nominal",
            X_col,
            categories,
            None,
        )
    if processing == "ignore":
        # called under: fit or predict
        X_col, categories = _process_column_initial(X_col, nonmissings, None, None)
        mapping = np.empty(len(categories) + 1, np.object_)
        mapping[0] = None
        for category, idx in categories.items():
            mapping[idx] = category
        bad = mapping[X_col]
        return "ignore", None, None, bad
    if isinstance(processing, str):
        # called under: fit

        # don't allow strings to get to the np.array conversion below
        msg = f"{processing} type invalid"
        _log.error(msg)
        raise ValueError(msg)
    # called under: fit

    n_items = 0
    n_ordinals = 0
    n_continuous = 0
    try:
        for item in processing:
            n_items += 1
            if isinstance(item, str):
                n_ordinals += 1
            elif isinstance(item, (float, int, np.floating, np.integer)):
                n_continuous += 1
    except TypeError:
        msg = f"{processing} type invalid"
        _log.error(msg)
        raise TypeError(msg)

    if n_continuous == n_items:
        # if n_items == 0 then it must be continuous since we
        # can have zero cut points, but not zero ordinal categories
        X_col, bad = _process_continuous(X_col, nonmissings)
        return "continuous", X_col, None, bad
    if n_ordinals == n_items:
        categories = dict(zip(processing, count(1)))
        X_col, bad = _encode_categorical_existing(X_col, nonmissings, categories)
        return "ordinal", X_col, categories, bad
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
            if n_items > 1:
                if is_found:
                    msg = f"Cannot reshape to 1D. Original shape was {col.shape}"
                    _log.error(msg)
                    raise ValueError(msg)
                is_found = True
        col = col.ravel()
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

    return _process_ndarray(
        X_col, nonmissings, categories, feature_type, min_unique_continuous
    )


def _process_pandas_column(X_col, categories, feature_type, min_unique_continuous):
    if isinstance(X_col.dtype, np.dtype):
        if (
            issubclass(X_col.dtype.type, np.floating)
            or issubclass(X_col.dtype.type, np.integer)
            or X_col.dtype.type is np.bool_
        ):
            X_col = X_col.values
            return _process_ndarray(
                X_col, None, categories, feature_type, min_unique_continuous
            )
        if X_col.dtype.type is np.object_:
            nonmissings = None
            if X_col.hasnans:
                # if hasnans is true then there is definetly a real missing value in there and not just a mask
                nonmissings = X_col.notna().values
                X_col = X_col.dropna()
            X_col = X_col.values
            return _process_ndarray(
                X_col, nonmissings, categories, feature_type, min_unique_continuous
            )
    elif isinstance(X_col.dtype, pd.CategoricalDtype):
        # unlike other missing value types, we get back -1's for missing here, so no need to drop them
        X_col = X_col.values
        is_ordered = X_col.ordered
        pd_categories = X_col.categories.values.astype(dtype=np.str_, copy=False)
        X_col = X_col.codes

        if feature_type == "ignore":
            pd_categories = pd_categories.astype(dtype=np.object_)
            pd_categories = np.insert(pd_categories, 0, None)
            bad = pd_categories[X_col + 1]
            return None, None, bad, "ignore"
        if categories is None:
            # called under: fit
            X_col, categories = _encode_pandas_categorical_initial(
                X_col, pd_categories, is_ordered, feature_type
            )
            bad = None
        else:
            # called under: predict
            X_col, bad = _encode_pandas_categorical_existing(
                X_col, pd_categories, categories
            )

        return "ordinal" if is_ordered else "nominal", X_col, categories, bad
    elif issubclass(X_col.dtype.type, np.integer) or X_col.dtype.type is np.bool_:
        # this handles Int8Dtype to Int64Dtype, UInt8Dtype to UInt64Dtype, and BooleanDtype
        nonmissings = None
        if X_col.hasnans:
            # if hasnans is true then there is definetly a real missing value in there and not just a mask
            nonmissings = X_col.notna().values
            X_col = X_col.dropna()
        X_col = X_col.values
        X_col = X_col.astype(dtype=X_col.dtype.type, copy=False)
        return _process_ndarray(
            X_col, nonmissings, categories, feature_type, min_unique_continuous
        )

    # TODO: implement pd.SparseDtype
    # TODO: implement pd.StringDtype both the numpy and arrow versions
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.StringDtype.html#pandas.StringDtype
    msg = f"{type(X_col.dtype)} not supported"
    _log.error(msg)
    raise TypeError(msg)


def _process_sparse_column(X_col, categories, feature_type, min_unique_continuous):
    X_col = X_col.toarray().ravel()

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

    return _process_ndarray(
        X_col, nonmissings, categories, feature_type, min_unique_continuous
    )


def _process_dict_column(X_col, categories, feature_type, min_unique_continuous):
    if isinstance(X_col, np.ndarray):  # this includes ma.masked_array
        pass
    elif _pandas_installed and isinstance(X_col, pd.Series):
        return _process_pandas_column(
            X_col, categories, feature_type, min_unique_continuous
        )
    elif _pandas_installed and isinstance(X_col, pd.DataFrame):
        if X_col.shape[1] == 1:
            X_col = X_col.iloc[:, 0]
            return _process_pandas_column(
                X_col, categories, feature_type, min_unique_continuous
            )
        if X_col.shape[0] == 1:
            X_col = X_col.astype(np.object_, copy=False).values.ravel()
        elif X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif safe_isinstance(X_col, "scipy.sparse.spmatrix") or safe_isinstance(
        X_col, "scipy.sparse.sparray"
    ):
        if X_col.shape[1] == 1 or X_col.shape[0] == 1:
            return _process_sparse_column(
                X_col, categories, feature_type, min_unique_continuous
            )
        if X_col.shape[1] == 0 or X_col.shape[0] == 0:
            X_col = np.empty(0, np.object_)
        else:
            msg = f"Cannot reshape to 1D. Original shape was {X_col.shape}"
            _log.error(msg)
            raise ValueError(msg)
    elif isinstance(X_col, (list, tuple)):
        X_col = np.array(X_col, np.object_)
    elif isinstance(X_col, str):
        # don't allow strings to get to the np.array conversion below
        X_col_tmp = np.empty(1, np.object_)
        X_col_tmp[0] = X_col
        X_col = X_col_tmp
    else:
        try:
            # we don't support iterables that get exhausted on their first examination.  This condition
            # should be detected though in preclean_X where we get the length or bin_native where we check the
            # number of samples on the 2nd run through the generator
            X_col = list(X_col)
            X_col = np.array(X_col, np.object_)
        except TypeError:
            # if our item isn't iterable, assume it has just 1 item and we'll check below if that's consistent
            X_col_tmp = np.empty(1, np.object_)
            X_col_tmp[0] = X_col
            X_col = X_col_tmp

    X_col = _reshape_1D_if_possible(X_col)
    return _process_numpy_column(X_col, categories, feature_type, min_unique_continuous)


def unify_columns(
    X,
    feature_idxs,
    categories,
    feature_names_in,
    feature_types,
    min_unique_continuous,
    go_fast,
):
    # preclean_X is always called on X prior to calling this function

    # unify_feature_names is always called on feature_names_in prior to calling this function

    # feature_names_in is guranteed not to contain duplicate names because unify_feature_names checks this.

    # feature_types can ONLY be None when called from unify_data OR when called from EBMPreprocessor.fit(...)
    # on all subsequent calls we pass a cleaned up feature_types from the results of the first call to EBMPreprocessor.fit(...)

    # If the categories paramter contains a dictionary, then that same categories object is guaranteed to
    # be yielded back to the caller.  This guarantee can be used to rapidly identify which request is being
    # yielded by using the id(categories) along with the feature_idx

    if isinstance(X, np.ndarray):  # this includes ma.masked_array
        if issubclass(X.dtype.type, np.complexfloating):
            msg = "Complex data not supported"
            _log.error(msg)
            raise ValueError(msg)
        elif issubclass(X.dtype.type, np.void):
            msg = "np.void data not supported"
            _log.error(msg)
            raise TypeError(msg)

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
        # if go_fast and X.flags.c_contiguous:
        #    # called under: predict
        #    # during predict we don't care as much about memory consumption, so speed it by transposing everything
        #    X = np.asfortranarray(X)

        n_cols = X.shape[1]
        if len(feature_names_in) == n_cols:
            if feature_types is None:
                for result in map(
                    _process_numpy_column,
                    map(
                        X.__getitem__,
                        zip(repeat(slice(None)), feature_idxs),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_numpy_column,
                    map(
                        X.__getitem__,
                        zip(repeat(slice(None)), feature_idxs),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
        else:
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires
            # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter(
                map(ne, repeat("ignore"), feature_types),
                np.bool_,
                count=len(feature_types),
            )
            if keep_cols.sum() != n_cols:
                # called under: predict
                msg = f"The model has {len(keep_cols)} features, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(keep_cols), np.int64)
            np.place(col_map, keep_cols, np.arange(len(keep_cols), dtype=np.int64))

            if feature_types is None:
                for result in map(
                    _process_numpy_column,
                    map(
                        X.__getitem__,
                        zip(
                            repeat(slice(None)),
                            map(
                                col_map.__getitem__,
                                feature_idxs,
                            ),
                        ),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_numpy_column,
                    map(
                        X.__getitem__,
                        zip(
                            repeat(slice(None)),
                            map(
                                col_map.__getitem__,
                                feature_idxs,
                            ),
                        ),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        cols = X.columns
        mapping = dict(zip(map(str, cols), cols))
        n_cols = len(cols)
        if len(mapping) != n_cols:
            warn(
                "Columns with duplicate names detected. This can happen for example if there are columns '0' and 0."
            )

            # We can handle duplicate names if they are not being used by the model.
            counts = Counter(map(str, cols))

            # sum is used to iterate outside the interpreter. The result is not used.
            sum(
                map(
                    truth,
                    map(
                        mapping.__delitem__,
                        compress(counts.keys(), map((1).__ne__, counts.values())),
                    ),
                )
            )

        if feature_types is None:
            if all(map(mapping.__contains__, feature_names_in)):
                # we can index by name, which is a lot faster in pandas

                if len(feature_names_in) != n_cols:
                    warn("Extra columns present in X that are not used by the model.")

                for result in map(
                    _process_pandas_column,
                    map(
                        X.__getitem__,
                        map(
                            mapping.__getitem__,
                            map(
                                feature_names_in.__getitem__,
                                feature_idxs,
                            ),
                        ),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                if len(feature_names_in) != n_cols:
                    msg = f"The model has {len(feature_names_in)} feature names, but X has {n_cols} columns."
                    _log.error(msg)
                    raise ValueError(msg)

                warn(
                    "Pandas dataframe X does not contain all feature names. Falling back to positional columns."
                )

                for result in map(
                    _process_pandas_column,
                    map(
                        X.iloc.__getitem__,
                        zip(repeat(slice(None)), feature_idxs),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
        else:
            if all(
                map(
                    mapping.__contains__,
                    compress(
                        feature_names_in,
                        map(ne, repeat("ignore"), feature_types),
                    ),
                )
            ):
                # we can index by name, which is a lot faster in pandas

                if len(feature_names_in) < n_cols:
                    warn("Extra columns present in X that are not used by the model.")

                for result in map(
                    _process_pandas_column,
                    map(
                        X.__getitem__,
                        map(
                            mapping.__getitem__,
                            map(
                                feature_names_in.__getitem__,
                                feature_idxs,
                            ),
                        ),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                if len(feature_names_in) == n_cols:
                    warn(
                        "Pandas dataframe X does not contain all feature names. Falling back to positional columns."
                    )

                    for result in map(
                        _process_pandas_column,
                        map(
                            X.iloc.__getitem__,
                            zip(
                                repeat(slice(None)),
                                feature_idxs,
                            ),
                        ),
                        categories,
                        map(
                            feature_types.__getitem__,
                            feature_idxs,
                        ),
                        repeat(min_unique_continuous),
                    ):
                        yield result
                else:
                    keep_cols = np.fromiter(
                        map(ne, repeat("ignore"), feature_types),
                        np.bool_,
                        count=len(feature_types),
                    )
                    if keep_cols.sum() != n_cols:
                        # called under: predict
                        msg = f"The model has {len(keep_cols)} features, but X has {n_cols} columns."
                        _log.error(msg)
                        raise ValueError(msg)
                    col_map = np.empty(len(keep_cols), np.int64)
                    np.place(
                        col_map, keep_cols, np.arange(len(keep_cols), dtype=np.int64)
                    )

                    warn(
                        "Pandas dataframe X does not contain all feature names. Falling back to positional columns."
                    )

                    for result in map(
                        _process_pandas_column,
                        map(
                            X.iloc.__getitem__,
                            zip(
                                repeat(slice(None)),
                                map(
                                    col_map.__getitem__,
                                    feature_idxs,
                                ),
                            ),
                        ),
                        categories,
                        map(
                            feature_types.__getitem__,
                            feature_idxs,
                        ),
                        repeat(min_unique_continuous),
                    ):
                        yield result
    elif safe_isinstance(X, "scipy.sparse.sparray"):
        if (
            safe_isinstance(X, "scipy.sparse.dia_array")
            or safe_isinstance(X, "scipy.sparse.bsr_array")
            or safe_isinstance(X, "scipy.sparse.coo_array")
        ):
            X = X.tocsc(copy=False)

        n_cols = X.shape[1]

        if len(feature_names_in) == n_cols:
            if feature_types is None:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.__getitem__,
                        zip(repeat(slice(None)), zip(feature_idxs)),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.__getitem__,
                        zip(repeat(slice(None)), zip(feature_idxs)),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
        else:
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires
            # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter(
                map(ne, repeat("ignore"), feature_types),
                np.bool_,
                count=len(feature_types),
            )
            if keep_cols.sum() != n_cols:
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

            if feature_types is None:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.__getitem__,
                        zip(
                            repeat(slice(None)),
                            zip(
                                map(
                                    col_map.__getitem__,
                                    feature_idxs,
                                )
                            ),
                        ),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.__getitem__,
                        zip(
                            repeat(slice(None)),
                            zip(
                                map(
                                    col_map.__getitem__,
                                    feature_idxs,
                                )
                            ),
                        ),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
    elif safe_isinstance(X, "scipy.sparse.spmatrix"):
        n_cols = X.shape[1]

        if len(feature_names_in) == n_cols:
            if feature_types is None:
                for result in map(
                    _process_sparse_column,
                    map(X.getcol, feature_idxs),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_sparse_column,
                    map(X.getcol, feature_idxs),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
        else:
            # during fit time unify_feature_names would only allow us to get here if this was legal, which requires
            # feature_types to not be None.  During predict time feature_types_in cannot be None, but we need
            # to check for legality on the dimensions of X
            keep_cols = np.fromiter(
                map(ne, repeat("ignore"), feature_types),
                np.bool_,
                count=len(feature_types),
            )
            if keep_cols.sum() != n_cols:
                msg = f"The model has {len(feature_types)} features, but X has {n_cols} columns."
                _log.error(msg)
                raise ValueError(msg)
            col_map = np.empty(len(feature_types), np.int64)
            np.place(col_map, keep_cols, np.arange(len(feature_types), dtype=np.int64))

            if feature_types is None:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.getcol,
                        map(
                            col_map.__getitem__,
                            feature_idxs,
                        ),
                    ),
                    categories,
                    repeat(None),
                    repeat(min_unique_continuous),
                ):
                    yield result
            else:
                for result in map(
                    _process_sparse_column,
                    map(
                        X.getcol,
                        map(
                            col_map.__getitem__,
                            feature_idxs,
                        ),
                    ),
                    categories,
                    map(
                        feature_types.__getitem__,
                        feature_idxs,
                    ),
                    repeat(min_unique_continuous),
                ):
                    yield result
    elif _pandas_installed and isinstance(X, pd.Series):
        # TODO: handle as a single feature model
        msg = "X as pandas.Series is unsupported"
        _log.error(msg)
        raise ValueError(msg)
    elif isinstance(X, dict):
        if feature_types is None:
            for result in map(
                _process_dict_column,
                map(
                    X.__getitem__,
                    map(
                        feature_names_in.__getitem__,
                        feature_idxs,
                    ),
                ),
                categories,
                repeat(None),
                repeat(min_unique_continuous),
            ):
                yield result
        else:
            for result in map(
                _process_dict_column,
                map(
                    X.__getitem__,
                    map(
                        feature_names_in.__getitem__,
                        feature_idxs,
                    ),
                ),
                categories,
                map(
                    feature_types.__getitem__,
                    feature_idxs,
                ),
                repeat(min_unique_continuous),
            ):
                yield result
    else:
        msg = "internal error"
        _log.error(msg)
        raise ValueError(msg)


def _determine_min_cols(feature_names=None, feature_types=None):
    if feature_types is None:
        return None if feature_names is None else len(feature_names)
    n_ignored = sum(map(eq, repeat("ignore"), feature_types))
    if (
        feature_names is None
        or len(feature_names) == len(feature_types)
        or len(feature_names) == len(feature_types) - n_ignored
    ):
        return len(feature_types) - n_ignored
    msg = f"feature_names has length {len(feature_names)} which does not match the length of feature_types {len(feature_types)}"
    _log.error(msg)
    raise ValueError(msg)


def unify_feature_names(X, feature_names_given=None, feature_types_given=None):
    # called under: fit

    # if feature_names_given and feature_types_given were the outputs of a fit function, then this function
    # is re-callable because it will return the same feature names as previously generated

    if isinstance(X, np.ndarray):  # this includes ma.masked_array
        X_names = None
        n_cols = X.shape[1]
    elif _pandas_installed and isinstance(X, pd.DataFrame):
        X_names = list(map(str, X.columns))
        n_cols = len(X_names)
    elif _pandas_installed and isinstance(X, pd.Series):
        X_names = None
        n_cols = 1
    elif safe_isinstance(X, "scipy.sparse.spmatrix") or safe_isinstance(
        X, "scipy.sparse.sparray"
    ):
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
        n_ignored = sum(
            1
            for feature_type_given in feature_types_given
            if feature_type_given == "ignore"
        )

    if feature_names_given is None:
        if feature_types_given is not None:
            if (
                len(feature_types_given) != n_cols
                and len(feature_types_given) != n_cols + n_ignored
            ):
                msg = f"There are {len(feature_types_given)} feature_types, but X has {n_cols} columns"
                _log.error(msg)
                raise ValueError(msg)
            n_cols = len(feature_types_given)

        feature_names_in = X_names
        if X_names is None:
            feature_names_in = []
            # this isn't used other than to indicate new names need to be created
            feature_types_given = ["ignore"] * n_cols
    else:
        n_final = len(feature_names_given)
        if feature_types_given is not None:
            n_final = len(feature_types_given)
            if (
                n_final != len(feature_names_given)
                and n_final != len(feature_names_given) + n_ignored
            ):
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
            if feature_types_given is not None and len(feature_names_in) == len(
                feature_types_given
            ):
                names_used = [
                    feature_name_in
                    for feature_name_in, feature_type_given in zip(
                        feature_names_in, feature_types_given
                    )
                    if feature_type_given != "ignore"
                ]

            X_names_unique = {
                name for name, n_count in Counter(X_names).items() if n_count == 1
            }
            if any(name not in X_names_unique for name in names_used):
                # ok, need to use position indexing
                if n_final != n_cols and n_final != n_cols + n_ignored:
                    msg = f"There are {n_final} features, but X has {n_cols} columns"
                    _log.error(msg)
                    raise ValueError(msg)

    if feature_types_given is not None:
        if len(feature_types_given) == len(feature_names_in):
            if len(feature_names_in) - n_ignored != len(
                {
                    feature_name_in
                    for feature_name_in, feature_type_given in zip(
                        feature_names_in, feature_types_given
                    )
                    if feature_type_given != "ignore"
                }
            ):
                msg = "cannot have duplicate feature names"
                _log.error(msg)
                raise ValueError(msg)

            return feature_names_in

        names_set = set(feature_names_in)

        names = []
        names_idx = 0
        feature_idx = 0
        for feature_type_given in feature_types_given:
            if feature_type_given == "ignore":
                while True:
                    # give 4 digits to the number so that anything below 9999 gets sorted in the right order in string format
                    name = f"feature_{feature_idx:04}"
                    feature_idx += 1
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


def _reshape_X(X, min_cols, n_samples, sample_source):
    if (
        X.ndim == 0 or X.shape[0] == 0
    ):  # zero dimensional arrays are possible, but really weird
        return np.empty((0, 0), X.dtype)
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

    if isinstance(X, np.ndarray):  # this includes ma.masked_array
        if X.ndim != 2:
            X = _reshape_X(X, min_cols, n_samples, sample_source)
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if _pandas_installed and isinstance(X, pd.DataFrame):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if safe_isinstance(X, "scipy.sparse.spmatrix") or safe_isinstance(
        X, "scipy.sparse.sparray"
    ):
        if n_samples is not None and n_samples != X.shape[0]:
            msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
            _log.error(msg)
            raise ValueError(msg)
        return X, X.shape[0]
    if _pandas_installed and isinstance(X, pd.Series):
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
    if isinstance(X, (list, tuple)):
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
        if isinstance(sample, (list, tuple)):
            pass
        elif isinstance(sample, ma.masked_array):
            # do this before np.ndarray since ma.masked_array is a subclass of np.ndarray
            if not is_copied:
                is_copied = True
                X = list(X)
            X[idx] = _reshape_1D_if_possible(
                sample.astype(np.object_, copy=False).filled(np.nan)
            )
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
                X[idx] = sample.astype(np.object_, copy=False).values.ravel()
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = np.empty(0, np.object_)
            else:
                msg = f"Cannot reshape to 1D. Original shape was {sample.shape}"
                _log.error(msg)
                raise ValueError(msg)
        elif safe_isinstance(sample, "scipy.sparse.spmatrix") or safe_isinstance(
            sample, "scipy.sparse.sparray"
        ):
            if sample.shape[1] == 1 or sample.shape[0] == 1:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = sample.toarray().ravel()
            elif sample.shape[1] == 0 or sample.shape[0] == 0:
                if not is_copied:
                    is_copied = True
                    X = list(X)
                X[idx] = np.empty(0, np.object_)
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
        elif isinstance(sample, str):
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

    # leave these as np.object_ for now and we'll try to densify per column where we're more likely to
    # succeed in densification since columns should generally be a single type
    X = np.array(X, np.object_)
    if X.ndim != 2:
        X = _reshape_X(X, min_cols, n_samples, sample_source)
    if n_samples is not None and n_samples != X.shape[0]:
        msg = f"{sample_source} has {n_samples} samples, but X has {X.shape[0]}"
        _log.error(msg)
        raise ValueError(msg)
    return X, X.shape[0]
