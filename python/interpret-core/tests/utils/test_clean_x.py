# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from itertools import repeat, count

import numpy as np
from numpy import ma
import pandas as pd
import scipy as sp

from interpret.utils import EBMPreprocessor
from interpret.utils._unify_data import unify_data
from interpret.utils._clean_x import preclean_X


def compare_bins(bins, expected_bins):
    assert len(bins) == len(expected_bins)
    for a, b in zip(bins, expected_bins):
        assert type(a) == type(b)
        if a is None:
            pass
        elif isinstance(a, dict):
            assert sorted(a.items()) == sorted(b.items())
        else:
            assert tuple(a) == tuple(b)


def unify_test(
    X,
    feature_names=None,
    feature_types=None,
    min_unique_continuous=0,
    missing_data_allowed=True,
    unseen_data_allowed=True,
    n_samples=None,
):
    X_check, feature_names_in, feature_types_in = unify_data(
        *preclean_X(X, feature_names, feature_types, n_samples=n_samples),
        feature_names,
        feature_types,
        missing_data_allowed=missing_data_allowed,
        unseen_data_allowed=unseen_data_allowed,
        min_unique_continuous=min_unique_continuous,
    )
    X_check[X_check != X_check] = None  # replace nan with None
    return X_check, feature_names_in, feature_types_in


class StringHolder:
    def __init__(self, internal_str):
        self.internal_str = internal_str

    def __str__(self):
        return self.internal_str

    def __lt__(self, other):
        return (
            True  # make all objects of this type identical to detect sorting failures
        )

    def __hash__(self):
        return 0  # make all objects of this type identical to detect hashing failures

    def __eq__(self, other):
        return (
            True  # make all objects of this type identical to detect hashing failures
        )


class DerivedStringHolder(StringHolder):
    def __init__(self, internal_str):
        StringHolder.__init__(self, internal_str)


class FloatHolder:
    def __init__(self, internal_float):
        self.internal_float = internal_float

    def __float__(self):
        return self.internal_float

    def __lt__(self, other):
        return (
            True  # make all objects of this type identical to detect sorting failures
        )

    def __hash__(self):
        return 0  # make all objects of this type identical to detect hashing failures

    def __eq__(self, other):
        return (
            True  # make all objects of this type identical to detect hashing failures
        )


class DerivedFloatHolder(FloatHolder):
    def __init__(self, internal_float):
        FloatHolder.__init__(self, internal_float)


class FloatAndStringHolder:
    def __init__(self, internal_float, internal_str):
        self.internal_float = internal_float
        self.internal_str = internal_str

    def __float__(self):
        return self.internal_float

    def __str__(self):
        return self.internal_str

    def __lt__(self, other):
        return (
            True  # make all objects of this type identical to detect sorting failures
        )

    def __hash__(self):
        return 0  # make all objects of this type identical to detect hashing failures

    def __eq__(self, other):
        return (
            True  # make all objects of this type identical to detect hashing failures
        )


class DerivedFloatAndStringHolder(FloatAndStringHolder):
    def __init__(self, internal_float, internal_str):
        FloatAndStringHolder.__init__(self, internal_float, internal_str)


class NothingHolder:
    # the result of calling str(..) includes the memory address, so they won't be dependable categories
    def __init__(self, internal_str):
        self.internal_str = internal_str


def check_pandas_normal(dtype, val1, val2):
    vals = [x[1] for x in sorted([(str(val1), val1), (str(val2), val2)])]

    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array(vals, dtype=np.object_), dtype=dtype)

    feature_names = None
    feature_types = ["nominal"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(X_check, np.array([[str(val1)], [str(val2)]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1"]
    assert pre.feature_types_in_ == ["nominal"]
    expected_bins = [dict(zip(map(str, vals), count(1)))]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[1], [2]], np.int64))

    # force reverse the categorical order
    pre.bins_ = [{str(vals[1]): 1, str(vals[0]): 2}]
    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[2], [1]], np.int64))


def check_pandas_missings(dtype, val1, val2):
    vals = [x[1] for x in sorted([(str(val1), val1), (str(val2), val2)])]
    val1, val2 = vals

    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        np.array([val2, val1, val1], dtype=np.object_), dtype=dtype
    )
    X["feature2"] = pd.Series(
        np.array([None, val2, val1], dtype=np.object_), dtype=dtype
    )
    X["feature3"] = pd.Series(
        np.array([val1, None, val2], dtype=np.object_), dtype=dtype
    )
    X["feature4"] = pd.Series(
        np.array([val2, val1, None], dtype=np.object_), dtype=dtype
    )

    feature_names = None
    feature_types = ["nominal", "nominal", "nominal", "nominal"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1", "feature2", "feature3", "feature4"]
    assert feature_types_in == ["nominal", "nominal", "nominal", "nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                [str(val2), None, str(val1), str(val2)],
                [str(val1), str(val2), None, str(val1)],
                [str(val1), str(val1), str(val2), None],
            ],
            np.object_,
        ),
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1", "feature2", "feature3", "feature4"]
    assert pre.feature_types_in_ == ["nominal", "nominal", "nominal", "nominal"]
    expected_bins = list(repeat(dict(zip(map(str, vals), count(1))), 4))
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[2, 0, 1, 2], [1, 2, 0, 1], [1, 1, 2, 0]], np.int64)
    )

    # force reverse the categorical order
    pre.bins_ = list(repeat({str(val2): 1, str(val1): 2}, 4))
    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[1, 0, 2, 1], [2, 1, 0, 2], [2, 2, 1, 0]], np.int64)
    )


def check_pandas_float(dtype, val1, val2):
    vals = sorted([dtype(val1), dtype(val2)])
    val1, val2 = vals

    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array([val2, val1, val1], dtype=dtype))
    X["feature2"] = pd.Series(np.array([np.nan, val2, val1], dtype=dtype))
    X["feature3"] = pd.Series(np.array([val1, np.nan, val2], dtype=dtype))
    X["feature4"] = pd.Series(np.array([val2, val1, np.nan], dtype=dtype))

    feature_names = None
    feature_types = ["continuous", "continuous", "continuous", "continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1", "feature2", "feature3", "feature4"]
    assert feature_types_in == ["continuous", "continuous", "continuous", "continuous"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                [float(val2), None, float(val1), float(val2)],
                [float(val1), float(val2), None, float(val1)],
                [float(val1), float(val1), float(val2), None],
            ],
            np.object_,
        ),
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1", "feature2", "feature3", "feature4"]
    assert pre.feature_types_in_ == [
        "continuous",
        "continuous",
        "continuous",
        "continuous",
    ]

    expected_bins = list(
        repeat(np.array([(float(val1) + float(val2)) / 2.0], np.float64), 4)
    )
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[2, 0, 1, 2], [1, 2, 0, 1], [1, 1, 2, 0]], np.int64)
    )


def check_numpy_throws(dtype_src, val1, val2):
    X = np.array([[val1, val2], [val1, val2]], dtype=dtype_src)
    pre = EBMPreprocessor()
    try:
        pre.fit(X)
    except:
        return
    raise AssertionError()


def test_process_continuous_obj_simple():
    X = np.array([[1], [2.5], ["3"], ["4.5"], [np.float32("5.5")]], dtype=np.object_)

    feature_names = None
    feature_types = ["continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check, np.array([[1.0], [2.5], [3.0], [4.5], [5.5]], np.object_)
    )


def test_process_continuous_obj_hard():
    X = np.array(
        [
            [1],
            [2.5],
            ["3"],
            ["4.5"],
            [np.float32("5.5")],
            [StringHolder("6.5")],
            [DerivedStringHolder("7.5")],
            [FloatHolder(8.5)],
            [DerivedFloatHolder(9.5)],
            [FloatAndStringHolder(10.5, "88")],
            [DerivedFloatAndStringHolder(11.5, "99")],
        ],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = ["continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check,
        np.array(
            [[1], [2.5], [3], [4.5], [5.5], [6.5], [7.5], [8.5], [9.5], [10.5], [11.5]],
            np.object_,
        ),
    )


def test_process_continuous_obj_hard_bad():
    X = np.array(
        [
            [1],
            [2.5],
            ["3"],
            ["4.5"],
            [np.float32("5.5")],
            [StringHolder("6.5")],
            ["bad"],
            [StringHolder("bad2")],
            [NothingHolder("bad3")],
        ],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = ["continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check,
        np.array(
            [[1.0], [2.5], [3.0], [4.5], [5.5], [6.5], [None], [None], [None]],
            np.object_,
        ),
    )


def test_process_column_initial_float32_float64():
    # np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 to float64 version has the lower mantisa bits
    # all set to zero, and there will be another float64 that will be closer to "0.1" in float64 representation, so
    # they aren't the same, but if to convert them to strings first then they are identical.  Strings are the
    # ultimate arbiter of categorical membership since strings are cross-platform and JSON encodable.  np.unique
    # will tend to separate the float32 and the float64 values since they aren't the same, but then serialize
    # them to the same string.  The our model has ["0.1", "0.1"] as the categories if we don't convert to float64!

    X = np.array([[np.float32(0.1)], [np.float64(0.1)]], dtype=np.object_)

    feature_names = None
    feature_types = ["continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(X_check, np.array([[0.10000000149011612], [0.1]], np.object_))


def test_process_column_initial_obj_obj():
    X = np.array([[StringHolder("abc")], [StringHolder("def")]], dtype=np.object_)

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(X_check, np.array([["abc"], ["def"]], np.object_))


def test_process_column_initial_alphabetical_nomissing():
    X = np.array([["xyz"], ["abc"], ["xyz"]], dtype=np.str_)

    feature_names = None
    feature_types = ["nominal_alphabetical"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"abc": 1, "xyz": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_alphabetical_missing():
    X = np.array([["xyz"], ["abc"], [None], ["xyz"]], dtype=np.object_)

    feature_names = None
    feature_types = ["nominal_alphabetical"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"abc": 1, "xyz": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_prevalence_nomissing():
    X = np.array([["xyz"], ["abc"], ["xyz"]], dtype=np.str_)

    feature_names = None
    feature_types = ["nominal_prevalence"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"xyz": 1, "abc": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_prevalence_missing():
    X = np.array([["xyz"], ["abc"], [None], ["xyz"]], dtype=np.object_)

    feature_names = None
    feature_types = ["nominal_prevalence"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"xyz": 1, "abc": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_alphabetical_nomissing_int8():
    X = np.array([[1], [-1], [1]], dtype=np.int8)

    feature_names = None
    feature_types = ["nominal_alphabetical"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"-1": 1, "1": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_prevalence_nomissing_int8():
    X = np.array([[1], [-1], [1]], dtype=np.int8)

    feature_names = None
    feature_types = ["nominal_prevalence"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"1": 1, "-1": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_alphabetical_nomissing_one_bool():
    X = np.array([[True], [True], [True]], dtype=np.bool_)

    feature_names = None
    feature_types = ["nominal_alphabetical"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"True": 1}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_alphabetical_nomissing_two_bool():
    X = np.array([[True], [True], [False], [True]], dtype=np.bool_)

    feature_names = None
    feature_types = ["nominal_alphabetical"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"False": 1, "True": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_prevalence_nomissing_one_bool():
    X = np.array([[True], [True], [True]], dtype=np.bool_)

    feature_names = None
    feature_types = ["nominal_prevalence"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"True": 1}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_prevalence_nomissing_two_bool():
    X = np.array([[True], [True], [False], [True]], dtype=np.bool_)

    feature_names = None
    feature_types = ["nominal_prevalence"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"True": 1, "False": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_int_small():
    X = np.array([[1], [np.int8(-2)], [np.uint64(3)]], dtype=np.object_)

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"-2": 1, "1": 2, "3": 3}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_int_big():
    X = np.array(
        [[1], [np.int8(-2)], [np.uint64("18446744073709551615")]], dtype=np.object_
    )

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"-2": 1, "1": 2, "18446744073709551615": 3}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_floats():
    # np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 to float64 version has the lower mantisa bits
    # all set to zero, and there will be another float64 that will be closer to "0.1" in float64 representation, so
    # they aren't the same, but if to convert them to strings first then they are identical.  Strings are the
    # ultimate arbiter of categorical membership since strings are cross-platform and JSON encodable.  np.unique
    # will tend to separate the float32 and the float64 values since they aren't the same, but then serialize
    # them to the same string.  The our model has ["0.1", "0.1"] as the categories if we don't convert to float64!

    X = np.array(
        [
            [1.1],
            [np.float16(2.2)],
            [np.float32(3.3)],
            [np.float64(4.4)],
            [np.longdouble(5.5)],
        ],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [
        {"1.1": 1, "2.19921875": 2, "3.299999952316284": 3, "4.4": 4, "5.5": 5}
    ]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_str_float64():
    X = np.array([["abc"], [np.float64(1.1)]], dtype=np.object_)

    feature_names = None
    feature_types = None

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"1.1": 1, "abc": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_str_float32():
    X = np.array([["abc"], [np.float32(1.1)]], dtype=np.object_)

    feature_names = None
    feature_types = None

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"1.100000023841858": 1, "abc": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_int_float():
    # this test is hard since np.unique seems to think int(4) == float(4) so naively it returns just "4"

    X = np.array([[4], [4.0]], dtype=np.object_)

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"4": 1, "4.0": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_int_float32():
    # if you take np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 version has the lower mantisa
    # bits all set to zero, and there will be another float64 that will be closer to "0.1" for float64s, so
    # they aren't the same, but if to convert them to strings first then they are identical.  I tend to think
    # of strings are the ultimate arbiter of categorical membership since strings are cross-platform
    # np.unique will tend to separate the float32 and the float64 values since they aren't the same, but then
    # serialize them to the same string.  The our model has ["0.1", "0.1"] as the categories!!

    X = np.array([[4], [np.float32(0.1)]], dtype=np.object_)

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"0.10000000149011612": 1, "4": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_encode_categorical_existing_obj_obj():
    X = np.array([[StringHolder("abc")], [StringHolder("def")]], dtype=np.object_)

    feature_names = None
    feature_types = None

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"abc": 1, "def": 2}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_choose_floatcategories():
    X = np.array(
        [[11.11], [2.2], [np.float32(2.2)], ["2.2"], [StringHolder("2.2")]],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"2.2": 1, "2.200000047683716": 2, "11.11": 3}]
    compare_bins(pre.bins_, expected_bins)


def test_process_column_initial_choose_floats():
    X = np.array(
        [
            [11.11],
            [2.2],
            [np.float32(2.2)],
            ["2.2"],
            [StringHolder("2.2")],
            [3.3],
            [3.3],
        ],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = ["nominal"]

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"2.2": 1, "2.200000047683716": 2, "3.3": 3, "11.11": 4}]
    compare_bins(pre.bins_, expected_bins)


def test_unify_columns_duplicates():
    X = pd.DataFrame()
    X["0"] = [1, 4]
    X[0] = [2, 5]
    X["feature3"] = [3, 6]

    feature_names = ["feature3"]
    feature_types = ["continuous"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature3"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(X_check, np.array([[3.0], [6.0]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature3"]
    assert pre.feature_types_in_ == ["continuous"]

    expected_bins = [np.array([(3.0 + 6.0) / 2.0], np.float64)]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[1], [2]], np.int64))


def test_unify_columns_numpy1():
    X = np.array([1, 2, 3])

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X, feature_names, feature_types, n_samples=1
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["continuous", "continuous", "continuous"]
    assert np.array_equal(X_check, np.array([[1.0, 2.0, 3.0]], np.object_))


def test_unify_columns_numpy2():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["continuous", "continuous", "continuous"]
    assert np.array_equal(
        X_check, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.object_)
    )


def test_unify_columns_numpy_ignore():
    X = np.array([["abc", None, "def"], ["ghi", "jkl", None]])

    feature_names = None
    feature_types = ["ignore", "ignore", "ignore"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["ignore", "ignore", "ignore"]
    assert np.array_equal(
        X_check, np.array([[None, None, None], [None, None, None]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000", "feature_0001", "feature_0002"]
    assert pre.feature_types_in_ == ["ignore", "ignore", "ignore"]
    expected_bins = list(repeat(None, 3))
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0, 0, 0], [0, 0, 0]], np.int64))


def test_unify_columns_scipy():
    X = sp.sparse.csc_matrix([[1, 2, 3], [4, 5, 6]])

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["continuous", "continuous", "continuous"]
    assert np.array_equal(
        X_check, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.object_)
    )


def test_unify_columns_dict1():
    X = {"feature1": [1], "feature2": "hi", "feature3": None}

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1", "feature2", "feature3"]
    assert feature_types_in == ["continuous", "nominal", "continuous"]
    assert np.array_equal(X_check, np.array([[1.0, "hi", None]], np.object_))


def test_unify_columns_dict2():
    X = {"feature1": [1, 4], "feature2": [2, 5], "feature3": [3, 6]}

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1", "feature2", "feature3"]
    assert feature_types_in == ["continuous", "continuous", "continuous"]
    assert np.array_equal(
        X_check, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.object_)
    )


def test_unify_columns_list1():
    X = [1, 2.0, "hi", None]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
        n_samples=1,
    )
    assert feature_names_in == [
        "feature_0000",
        "feature_0001",
        "feature_0002",
        "feature_0003",
    ]
    assert feature_types_in == ["continuous", "continuous", "nominal", "continuous"]
    assert np.array_equal(X_check, np.array([[1.0, 2.0, "hi", None]], np.object_))


def test_unify_columns_list2():
    P1 = pd.DataFrame()
    P1["feature1"] = pd.Series(np.array([1, None, np.nan], dtype=np.object_))
    P2 = pd.DataFrame()
    P2["feature1"] = pd.Series(np.array([1], dtype=np.float32))
    P2["feature2"] = pd.Series(np.array([None], dtype=np.object_))
    P2["feature3"] = pd.Series(np.array([np.nan], dtype=np.object_))
    S1 = sp.sparse.csc_matrix([[1, 2, 3]])
    S2 = sp.sparse.csc_matrix([[1], [2], [3]])
    X = [
        np.array([1, 2, 3], dtype=np.int8),
        pd.Series([4.0, None, np.nan]),
        [1, 2.0, "hi"],
        (np.double(4.0), "bye", None),
        {1, 2, 3},
        {"abc": 1, "def": 2, "ghi": 3}.keys(),
        {"abc": 1, "def": 2, "ghi": 3}.values(),
        range(1, 4),
        (x for x in [1, 2, 3]),
        np.array([1, 2, 3], dtype=np.object_),
        np.array([[1, 2, 3]], dtype=np.int8),
        np.array([[1], [2], [3]], dtype=np.int8),
        P1,
        P2,
        S1,
        S2,
    ]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["nominal", "nominal", "nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                ["1", "2", "3"],
                ["4.0", None, None],
                ["1", "2.0", "hi"],
                ["4.0", "bye", None],
                ["1", "2", "3"],
                ["abc", "def", "ghi"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", None, None],
                ["1.0", None, None],
                ["1", "2", "3"],
                ["1", "2", "3"],
            ],
            np.object_,
        ),
    )


def test_unify_columns_tuple1():
    X = (1, 2.0, "hi", None)

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
        n_samples=1,
    )
    assert feature_names_in == [
        "feature_0000",
        "feature_0001",
        "feature_0002",
        "feature_0003",
    ]
    assert feature_types_in == ["continuous", "continuous", "nominal", "continuous"]
    assert np.array_equal(X_check, np.array([[1.0, 2.0, "hi", None]], np.object_))


def test_unify_columns_tuple2():
    X = (
        np.array([1, 2, 3], dtype=np.int8),
        pd.Series([4, 5, 6]),
        [1, 2.0, "hi"],
        (np.double(4.0), "bye", None),
        {1, 2, 3},
        {"abc": 1, "def": 2, "ghi": 3}.keys(),
        {"abc": 1, "def": 2, "ghi": 3}.values(),
        range(1, 4),
        (x for x in [1, 2, 3]),
        np.array([1, 2, 3], dtype=np.object_),
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["nominal", "nominal", "nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                ["1", "2", "3"],
                ["4", "5", "6"],
                ["1", "2.0", "hi"],
                ["4.0", "bye", None],
                ["1", "2", "3"],
                ["abc", "def", "ghi"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
            ],
            np.object_,
        ),
    )


def test_unify_columns_generator1():
    X = (x for x in [1, 2.0, "hi", None])

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
        n_samples=1,
    )
    assert feature_names_in == [
        "feature_0000",
        "feature_0001",
        "feature_0002",
        "feature_0003",
    ]
    assert feature_types_in == ["continuous", "continuous", "nominal", "continuous"]
    assert np.array_equal(X_check, np.array([[1.0, 2.0, "hi", None]], np.object_))


def test_unify_columns_generator2():
    X = (
        x
        for x in [
            np.array([1, 2, 3], dtype=np.int8),
            pd.Series([4, 5, 6]),
            [1, 2.0, "hi"],
            (np.double(4.0), "bye", None),
            {1, 2, 3},
            {"abc": 1, "def": 2, "ghi": 3}.keys(),
            {"abc": 1, "def": 2, "ghi": 3}.values(),
            range(1, 4),
            (x for x in [1, 2, 3]),
            np.array([1, 2, 3], dtype=np.object_),
        ]
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]
    assert feature_types_in == ["nominal", "nominal", "nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                ["1", "2", "3"],
                ["4", "5", "6"],
                ["1", "2.0", "hi"],
                ["4.0", "bye", None],
                ["1", "2", "3"],
                ["abc", "def", "ghi"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
                ["1", "2", "3"],
            ],
            np.object_,
        ),
    )


def test_unify_columns_pandas_normal_int8():
    check_pandas_normal(np.int8, -128, 127)


def test_unify_columns_pandas_normal_uint8():
    check_pandas_normal(np.uint8, 0, 255)


def test_unify_columns_pandas_normal_int16():
    check_pandas_normal(np.int16, -32768, 32767)


def test_unify_columns_pandas_normal_uint16():
    check_pandas_normal(np.uint16, 0, 65535)


def test_unify_columns_pandas_normal_int32():
    check_pandas_normal(np.int32, -2147483648, 2147483647)


def test_unify_columns_pandas_normal_uint32():
    check_pandas_normal(np.uint32, 0, 4294967295)


def test_unify_columns_pandas_normal_int64():
    check_pandas_normal(np.int64, -9223372036854775808, 9223372036854775807)


def test_unify_columns_pandas_normal_uint64():
    check_pandas_normal(np.uint64, np.uint64("0"), np.uint64("18446744073709551615"))


def test_unify_columns_pandas_normal_bool():
    check_pandas_normal(np.bool_, False, True)


def test_unify_columns_pandas_missings_float64():
    check_pandas_float(np.float64, -1.1, 2.2)


def test_unify_columns_pandas_missings_longfloat():
    check_pandas_float(np.longdouble, -1.1, 2.2)


def test_unify_columns_pandas_missings_float32():
    check_pandas_float(np.float32, -1.1, 2.2)


def test_unify_columns_pandas_missings_float16():
    check_pandas_float(np.float16, -1.1, 2.2)


def test_unify_columns_pandas_missings_Int8Dtype():
    check_pandas_missings(pd.Int8Dtype(), -128, 127)


def test_unify_columns_pandas_missings_UInt8Dtype():
    check_pandas_missings(pd.UInt8Dtype(), 0, 255)


def test_unify_columns_pandas_missings_Int16Dtype():
    check_pandas_missings(pd.Int16Dtype(), -32768, 32767)


def test_unify_columns_pandas_missings_UInt16Dtype():
    check_pandas_missings(pd.UInt16Dtype(), 0, 65535)


def test_unify_columns_pandas_missings_Int32Dtype():
    check_pandas_missings(pd.Int32Dtype(), -2147483648, 2147483647)


def test_unify_columns_pandas_missings_UInt32Dtype():
    check_pandas_missings(pd.UInt32Dtype(), 0, 4294967295)


def test_unify_columns_pandas_missings_Int64Dtype():
    check_pandas_missings(pd.Int64Dtype(), -9223372036854775808, 9223372036854775807)


def test_unify_columns_pandas_missings_UInt64Dtype():
    check_pandas_missings(
        pd.UInt64Dtype(), np.uint64("0"), np.uint64("18446744073709551615")
    )


def test_unify_columns_pandas_missings_BooleanDtype():
    check_pandas_missings(pd.BooleanDtype(), False, True)


def test_unify_columns_pandas_missings_StringDtype():
    check_pandas_missings(pd.StringDtype(), "abc", "def")


def test_unify_columns_pandas_missings_str():
    check_pandas_missings(np.object_, "abc", "def")


def test_unify_columns_pandas_missings_nice_str():
    check_pandas_missings(np.object_, StringHolder("abc"), "def")


def test_unify_columns_pandas_missings_pure_ints():
    check_pandas_missings(np.object_, 1, 2)


def test_unify_columns_pandas_missings_pure_floats():
    check_pandas_missings(np.object_, 1.1, 2.2)


def test_unify_columns_pandas_missings_mixed_floats():
    check_pandas_missings(np.object_, 1.1, "2.2")


def test_unify_columns_pandas_missings_mixed_floats2():
    check_pandas_missings(np.object_, StringHolder("1.1"), "2.2")


def test_unify_columns_str_throw():
    X = "abc"
    try:
        unify_test(X)
    except:  # noqa: E722
        return
    raise AssertionError()


def test_unify_columns_int_throw():
    X = 1
    try:
        unify_test(X)
    except:  # noqa: E722
        return
    raise AssertionError()


def test_unify_columns_duplicate_colnames_throw():
    X = pd.DataFrame()
    X["0"] = [1, 2]
    X[0] = [3, 4]
    try:
        unify_test(X)
    except:  # noqa: E722
        return
    raise AssertionError()


def test_unify_columns_opaque_str_throw():
    # this should fail since the default string generator makes a useless as a category string like:
    # <interpret.glassbox.ebm.test.test_bin.NothingHolder object at 0x0000019525E9FE48>
    check_numpy_throws(np.object_, NothingHolder("abc"), "def")


def test_unify_columns_list_throw():
    check_numpy_throws(np.object_, ["abc", "bcd"], "def")


def test_unify_columns_tuple_throw():
    check_numpy_throws(np.object_, ("abc", "bcd"), "def")


def test_unify_columns_set_throw():
    check_numpy_throws(np.object_, {"abc", "bcd"}, "def")


def test_unify_columns_dict_throw():
    check_numpy_throws(np.object_, {"abc": 1, "bcd": 2}, "def")


def test_unify_columns_keys_throw():
    check_numpy_throws(np.object_, {"abc": 1, "bcd": 2}.keys(), "def")


def test_unify_columns_values_throw():
    check_numpy_throws(np.object_, {"abc": 1, "bcd": 2}.values(), "def")


def test_unify_columns_range_throw():
    check_numpy_throws(np.object_, range(1, 2), "def")


def test_unify_columns_generator_throw():
    check_numpy_throws(np.object_, (x for x in [1, 2]), "def")


def test_unify_columns_ndarray_throw():
    check_numpy_throws(np.object_, np.array([1, "abc"], dtype=np.object_), "def")


def test_unify_columns_pandas_obj_to_float():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        np.array(
            [
                None,
                np.nan,
                np.float16(np.nan),
                0,
                -1,
                2.2,
                "-3.3",
                np.float16("4.4"),
                StringHolder("-5.5"),
                np.float32("6.6").item(),
            ],
            dtype=np.object_,
        )
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                [None],
                [None],
                [None],
                [0.0],
                [-1.0],
                [2.2],
                [-3.3],
                [4.3984375],
                [-5.5],
                [6.5999999046325684],
            ],
            np.object_,
        ),
    )


def test_unify_columns_pandas_obj_to_str():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        np.array(
            [
                None,
                np.nan,
                np.float16(np.nan),
                0,
                -1,
                2.2,
                "-3.3",
                np.float16("4.4"),
                StringHolder("-5.5"),
                5.6843418860808014e-14,
                "None",
                "nan",
            ],
            dtype=np.object_,
        )
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [
                [None],
                [None],
                [None],
                ["0"],
                ["-1"],
                ["2.2"],
                ["-3.3"],
                ["4.3984375"],
                ["-5.5"],
                ["5.684341886080802e-14"],
                ["None"],
                ["nan"],
            ],
            np.object_,
        ),
    )


def test_unify_columns_pandas_categorical():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        [None, np.nan, "not_in_categories", "a", "bcd", "0"],
        dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=False),
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], [None], ["a"], ["bcd"], ["0"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"a": 1, "0": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    pre.bins_[0]["new"] = 4

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0], [0], [0], [1], [3], [2]], np.int64))


def test_unify_columns_pandas_ordinal():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        [None, np.nan, "not_in_categories", "a", "bcd", "0"],
        dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=True),
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["ordinal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], [None], ["a"], ["bcd"], ["0"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1"]
    assert pre.feature_types_in_ == ["ordinal"]

    expected_bins = [{"a": 1, "0": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    del pre.bins_[0]["0"]
    del pre.bins_[0]["bcd"]

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0], [0], [0], [1], [2], [2]], np.int64))


def test_unify_columns_pandas_categorical_unused():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        [None, np.nan, "not_in_categories", "a", "bcd"],
        dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=True),
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["ordinal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], [None], ["a"], ["bcd"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1"]
    assert pre.feature_types_in_ == ["ordinal"]

    expected_bins = [{"a": 1, "0": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    del pre.bins_[0]["bcd"]
    del pre.bins_[0]["0"]
    X.iloc[0, 0] = "0"

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[2], [0], [0], [1], [2]], np.int64))


def test_unify_columns_pandas_categorical_remap():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(
        [None, np.nan, "not_in_categories", "a", "bcd", "0"],
        dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=True),
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature1"]
    assert feature_types_in == ["ordinal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], [None], ["a"], ["bcd"], ["0"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature1"]
    assert pre.feature_types_in_ == ["ordinal"]

    expected_bins = [{"a": 1, "0": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    del pre.bins_[0]["bcd"]
    pre.bins_[0]["what"] = 3
    pre.bins_[0]["0"] = 1
    pre.bins_[0]["a"] = 2

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0], [0], [0], [2], [4], [1]], np.int64))


def test_unify_columns_extend_categories():
    X = [[None], [np.nan], ["a"], ["bcd"], ["0"]]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], ["a"], ["bcd"], ["0"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"0": 1, "a": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    pre.bins_[0]["x"] = 4
    X[0][0] = "new_thing"

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[5], [0], [2], [3], [1]], np.int64))


def test_unify_columns_reduce_categories():
    X = [[None], [np.nan], ["a"], ["bcd"], ["0"]]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(
        X_check, np.array([[None], [None], ["a"], ["bcd"], ["0"]], np.object_)
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"0": 1, "a": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    del pre.bins_[0]["bcd"]
    X[0][0] = "new_thing"

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[3], [0], [2], [3], [1]], np.int64))


def test_unify_columns_reduce_no_missing_categories():
    X = [["a"], ["bcd"], ["0"]]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(X_check, np.array([["a"], ["bcd"], ["0"]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"0": 1, "a": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    del pre.bins_[0]["bcd"]
    del pre.bins_[0]["a"]

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[2], [2], [1]], np.int64))


def test_unify_columns_remap_categories():
    X = [["a"], ["bcd"], ["0"]]

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(X_check, np.array([["a"], ["bcd"], ["0"]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"0": 1, "a": 2, "bcd": 3}]
    compare_bins(pre.bins_, expected_bins)

    pre.bins_[0]["bcd"] = 2
    pre.bins_[0]["a"] = 3
    X[0][0] = "new_thing"

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[4], [2], [1]], np.int64))


def test_unify_feature_names_data_frame1():
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

    _, feature_names_in, _ = unify_test(X)
    assert feature_names_in == ["0", "1", "2"]


def test_unify_feature_names_feature_types2():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    feature_types = ["continuous", "ignore", "continuous"]
    _, feature_names_in, _ = unify_test(X, feature_types=feature_types)
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]


def test_unify_feature_names_feature_types3():
    X = np.array([[1, 3], [4, 6]])

    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_types=feature_types)
    assert feature_names_in == ["feature_0000", "feature_0001", "feature_0002"]


def test_unify_pandas_ignored_existing():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]

    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_types=feature_types)
    assert feature_names_in == ["feature1", "feature2", "feature3"]


def test_unify_feature_names_pandas_feature_types3():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]

    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_types=feature_types)
    assert feature_names_in == ["feature1", "feature_0000", "feature3"]


def test_unify_feature_names_names1():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    feature_names = pd.Series([0, 1, 2])

    _, feature_names_in, _ = unify_test(X, feature_names)
    assert feature_names_in == ["0", "1", "2"]


def test_unify_feature_names_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    feature_names = [0, "SOMETHING", 2]

    _, feature_names_in, _ = unify_test(X, feature_names)
    assert feature_names_in == ["0", "SOMETHING", "2"]


def test_unify_feature_names_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]

    feature_names = pd.Series([0, 1, 2])

    _, feature_names_in, _ = unify_test(X, feature_names)
    assert feature_names_in == ["0", "1", "2"]


def test_unify_feature_names_types_ignored_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]

    feature_names = [0, "SOMETHING", 2]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["0", "SOMETHING", "2"]


def test_unify_feature_names_types_nondropped2_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    feature_names = [0, 2]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["0", "feature_0000", "2"]


def test_unify_feature_names_types_nondropped2_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]

    feature_names = [0, 2]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["0", "feature_0000", "2"]


def test_unify_feature_names_types_dropped2_names2():
    X = np.array([[1, 3], [4, 6]])

    feature_names = [0, 2]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["0", "feature_0000", "2"]


def test_unify_feature_names_types_dropped2_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]

    feature_names = [0, 2]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["0", "feature_0000", "2"]


def test_unify_feature_names_types_dropped3_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]

    feature_names = None
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["feature1", "feature2", "feature3"]


def test_unify_feature_names_types_rearrange1_drop1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]
    X, n_samples = preclean_X(X, None, None)

    feature_names = ["length", "SOMETHING", "width"]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "SOMETHING", "width"]


def test_unify_feature_names_types_rearrange1_drop2():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["length"] = [3, 6]

    feature_names = ["length", "SOMETHING", "width"]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "SOMETHING", "width"]


def test_unify_feature_names_types_rearrange2_drop1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]

    feature_names = ["length", "width"]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "feature_0000", "width"]


def test_unify_feature_names_types_rearrange_more1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED1"] = [2, 5]
    X["length"] = [3, 6]
    X["UNUSED2"] = [9, 9]

    feature_names = ["length", "SOMETHING", "width"]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "SOMETHING", "width"]


def test_unify_feature_names_types_rearrange_more2():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED1"] = [2, 5]
    X["length"] = [3, 6]
    X["UNUSED2"] = [9, 9]

    feature_names = ["length", "width"]
    feature_types = ["continuous", "ignore", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "feature_0000", "width"]


def test_unify_feature_names_types_rearrange_more3():
    X = pd.DataFrame()
    X["height"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]
    X["width"] = [8, 9]

    feature_names = ["length", "width", "height"]
    feature_types = ["continuous", "continuous", "continuous"]

    _, feature_names_in, _ = unify_test(X, feature_names, feature_types)
    assert feature_names_in == ["length", "width", "height"]


def test_unify_columns_ma_no_mask():
    X = ma.array([[np.nan], [1], [None]], dtype=np.object_)
    assert X.mask is ma.nomask

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(X_check, np.array([[None], [1.0], [None]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["continuous"]

    expected_bins = [np.array([], np.float64)]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0], [1], [0]], np.int64))


def test_unify_columns_ma_empty_mask():
    X = ma.array([[np.nan], [1], [None]], mask=[[0], [0], [0]], dtype=np.object_)
    assert X.mask is not ma.nomask

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(X_check, np.array([[None], [1.0], [None]], np.object_))

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["continuous"]

    expected_bins = [np.array([], np.float64)]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(binned, np.array([[0], [1], [0]], np.int64))


def test_unify_columns_ma_objects():
    X = ma.array(
        [[np.nan], [None], [1], [2], [None], [3], [np.nan]],
        mask=[[0], [1], [0], [1], [0], [0], [1]],
        dtype=np.object_,
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check,
        np.array([[None], [None], [1.0], [None], [None], [3.0], [None]], np.object_),
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["continuous"]

    expected_bins = [np.array([2.0], np.float64)]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[0], [0], [1], [0], [0], [2], [0]], np.int64)
    )


def test_unify_columns_ma_continuous():
    X = ma.array(
        [[np.nan], [None], [1], [2], [None], [3], [np.nan]],
        mask=[[0], [1], [0], [1], [0], [0], [1]],
        dtype=np.float64,
    )

    feature_names = None
    feature_types = None

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["continuous"]
    assert np.array_equal(
        X_check,
        np.array([[None], [None], [1.0], [None], [None], [3.0], [None]], np.object_),
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["continuous"]

    expected_bins = [np.array([2.0], np.float64)]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[0], [0], [1], [0], [0], [2], [0]], np.int64)
    )


def test_unify_columns_ma_objects_categorical():
    X = ma.array(
        [[np.nan], [None], [1], [2], [None], [3], [np.nan]],
        mask=[[0], [1], [0], [1], [0], [0], [1]],
        dtype=np.float64,
    )

    feature_names = None
    feature_types = ["nominal"]

    X_check, feature_names_in, feature_types_in = unify_test(
        X,
        feature_names,
        feature_types,
    )
    assert feature_names_in == ["feature_0000"]
    assert feature_types_in == ["nominal"]
    assert np.array_equal(
        X_check,
        np.array(
            [[None], [None], ["1.0"], [None], [None], ["3.0"], [None]], np.object_
        ),
    )

    pre = EBMPreprocessor(feature_names, feature_types)
    pre.fit(X)

    assert pre.feature_names_in_ == ["feature_0000"]
    assert pre.feature_types_in_ == ["nominal"]

    expected_bins = [{"1.0": 1, "3.0": 2}]
    compare_bins(pre.bins_, expected_bins)

    binned = pre.transform(X)
    assert np.array_equal(
        binned, np.array([[0], [0], [1], [0], [0], [2], [0]], np.int64)
    )
