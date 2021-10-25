# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy as sp

from collections import namedtuple
from itertools import repeat

from ..bin import *
from ..bin import _process_column_initial, _encode_categorical_existing, _process_continuous

class StringHolder:
    def __init__(self, internal_str):
        self.internal_str = internal_str
    def __str__(self):
        return self.internal_str
    def __lt__(self, other):
        return True # make all objects of this type identical to detect sorting failures
    def __hash__(self):
        return 0 # make all objects of this type identical to detect hashing failures
    def __eq__(self,other):
        return True # make all objects of this type identical to detect hashing failures

class DerivedStringHolder(StringHolder):
    def __init__(self, internal_str):
        StringHolder.__init__(self, internal_str)

class FloatHolder:
    def __init__(self, internal_float):
        self.internal_float = internal_float
    def __float__(self):
        return self.internal_float
    def __lt__(self, other):
        return True # make all objects of this type identical to detect sorting failures
    def __hash__(self):
        return 0 # make all objects of this type identical to detect hashing failures
    def __eq__(self,other):
        return True # make all objects of this type identical to detect hashing failures

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
        return True # make all objects of this type identical to detect sorting failures
    def __hash__(self):
        return 0 # make all objects of this type identical to detect hashing failures
    def __eq__(self,other):
        return True # make all objects of this type identical to detect hashing failures

class DerivedFloatAndStringHolder(FloatAndStringHolder):
    def __init__(self, internal_float, internal_str):
        FloatAndStringHolder.__init__(self, internal_float, internal_str)

class NothingHolder:
    # the result of calling str(..) includes the memory address, so they won't be dependable categories
    def __init__(self, internal_str):
        self.internal_str = internal_str

def check_pandas_normal(dtype, val1, val2):
    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array([val1, val2], dtype=np.object_), dtype=dtype)

    feature_types_in = ['nominal']

    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)

    X_cols = list(unify_columns(X, [(0, None)], feature_names_out, None))
    assert(len(X_cols) == 1)
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(len(X_cols[0][3]) == 2)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 2)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val2)])

    c1 = {str(val1) : 1, str(val2) : 2}
    X_cols = list(unify_columns(X, [(0, c1)], feature_names_out, feature_types_in))
    assert(len(X_cols) == 1)
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c1)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 2)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val2)])

    c2 = {str(val2) : 1, str(val1) : 2}
    X_cols = list(unify_columns(X, [(0, c2)], feature_names_out, feature_types_in))
    assert(len(X_cols) == 1)
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c2)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 2)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val2)])

def check_pandas_missings(dtype, val1, val2):
    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array([val2, val1, val1], dtype=np.object_), dtype=dtype)
    X["feature2"] = pd.Series(np.array([None, val2, val1], dtype=np.object_), dtype=dtype)
    X["feature3"] = pd.Series(np.array([val1, None, val2], dtype=np.object_), dtype=dtype)
    X["feature4"] = pd.Series(np.array([val2, val1, None], dtype=np.object_), dtype=dtype)

    c1 = {str(val1) : 1, str(val2) : 2}
    c2 = {str(val2) : 1, str(val1) : 2}
    feature_types_in = ['nominal', 'nominal', 'nominal', 'nominal']

    X, n_samples = clean_X(X)
    assert(n_samples == 3)
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)

    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None), (3, None)], feature_names_out, None))
    assert(4 == len(X_cols))

    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(len(X_cols[0][3]) == 2)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 3)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val2)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][2] == X_cols[0][3][str(val1)])

    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'nominal')
    assert(X_cols[1][4] is None)
    assert(len(X_cols[1][3]) == 2)
    assert(X_cols[1][2].dtype == np.int64)
    assert(len(X_cols[1][2]) == 3)
    assert(X_cols[1][2][0] == 0)
    assert(X_cols[1][2][1] == X_cols[1][3][str(val2)])
    assert(X_cols[1][2][2] == X_cols[1][3][str(val1)])

    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'nominal')
    assert(X_cols[2][4] is None)
    assert(len(X_cols[2][3]) == 2)
    assert(X_cols[2][2].dtype == np.int64)
    assert(len(X_cols[2][2]) == 3)
    assert(X_cols[2][2][0] == X_cols[2][3][str(val1)])
    assert(X_cols[2][2][1] == 0)
    assert(X_cols[2][2][2] == X_cols[2][3][str(val2)])
    
    assert(X_cols[3][0] == 3)
    assert(X_cols[3][1] == 'nominal')
    assert(X_cols[3][4] is None)
    assert(len(X_cols[3][3]) == 2)
    assert(X_cols[3][2].dtype == np.int64)
    assert(len(X_cols[3][2]) == 3)
    assert(X_cols[3][2][0] == X_cols[3][3][str(val2)])
    assert(X_cols[3][2][1] == X_cols[3][3][str(val1)])
    assert(X_cols[3][2][2] == 0)
    
    assert(np.array_equal(X_cols[1][2] == 0, X.iloc[:, 1].isna()))
    assert(np.array_equal(X_cols[2][2] == 0, X.iloc[:, 2].isna()))
    assert(np.array_equal(X_cols[3][2] == 0, X.iloc[:, 3].isna()))


    X_cols = list(unify_columns(X, [(0, c1), (1, c1), (2, c1), (3, c1)], feature_names_out, feature_types_in))
    assert(4 == len(X_cols))

    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c1)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 3)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val2)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][2] == X_cols[0][3][str(val1)])

    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'nominal')
    assert(X_cols[1][4] is None)
    assert(X_cols[1][3] is c1)
    assert(X_cols[1][2].dtype == np.int64)
    assert(len(X_cols[1][2]) == 3)
    assert(X_cols[1][2][0] == 0)
    assert(X_cols[1][2][1] == X_cols[1][3][str(val2)])
    assert(X_cols[1][2][2] == X_cols[1][3][str(val1)])

    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'nominal')
    assert(X_cols[2][4] is None)
    assert(X_cols[2][3] is c1)
    assert(X_cols[2][2].dtype == np.int64)
    assert(len(X_cols[2][2]) == 3)
    assert(X_cols[2][2][0] == X_cols[2][3][str(val1)])
    assert(X_cols[2][2][1] == 0)
    assert(X_cols[2][2][2] == X_cols[2][3][str(val2)])

    assert(X_cols[3][0] == 3)
    assert(X_cols[3][1] == 'nominal')
    assert(X_cols[3][4] is None)
    assert(X_cols[3][3] is c1)
    assert(X_cols[3][2].dtype == np.int64)
    assert(len(X_cols[3][2]) == 3)
    assert(X_cols[3][2][0] == X_cols[3][3][str(val2)])
    assert(X_cols[3][2][1] == X_cols[3][3][str(val1)])
    assert(X_cols[3][2][2] == 0)

    assert(np.array_equal(X_cols[1][2] == 0, X.iloc[:, 1].isna()))
    assert(np.array_equal(X_cols[2][2] == 0, X.iloc[:, 2].isna()))
    assert(np.array_equal(X_cols[3][2] == 0, X.iloc[:, 3].isna()))


    X_cols = list(unify_columns(X, [(0, c2), (1, c2), (2, c2), (3, c2)], feature_names_out, feature_types_in))
    assert(4 == len(X_cols))

    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c2)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 3)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val2)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][2] == X_cols[0][3][str(val1)])

    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'nominal')
    assert(X_cols[1][4] is None)
    assert(X_cols[1][3] is c2)
    assert(X_cols[1][2].dtype == np.int64)
    assert(len(X_cols[1][2]) == 3)
    assert(X_cols[1][2][0] == 0)
    assert(X_cols[1][2][1] == X_cols[1][3][str(val2)])
    assert(X_cols[1][2][2] == X_cols[1][3][str(val1)])

    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'nominal')
    assert(X_cols[2][4] is None)
    assert(X_cols[2][3] is c2)
    assert(X_cols[2][2].dtype == np.int64)
    assert(len(X_cols[2][2]) == 3)
    assert(X_cols[2][2][0] == X_cols[2][3][str(val1)])
    assert(X_cols[2][2][1] == 0)
    assert(X_cols[2][2][2] == X_cols[2][3][str(val2)])

    assert(X_cols[3][0] == 3)
    assert(X_cols[3][1] == 'nominal')
    assert(X_cols[3][4] is None)
    assert(X_cols[3][3] is c2)
    assert(X_cols[3][2].dtype == np.int64)
    assert(len(X_cols[3][2]) == 3)
    assert(X_cols[3][2][0] == X_cols[3][3][str(val2)])
    assert(X_cols[3][2][1] == X_cols[3][3][str(val1)])
    assert(X_cols[3][2][2] == 0)
    assert(np.array_equal(X_cols[1][2] == 0, X.iloc[:, 1].isna()))
    assert(np.array_equal(X_cols[2][2] == 0, X.iloc[:, 2].isna()))
    assert(np.array_equal(X_cols[3][2] == 0, X.iloc[:, 3].isna()))


    X_cols = list(unify_columns(X, [(0, c1), (1, c2), (2, c1), (3, c2)], feature_names_out, feature_types_in))
    assert(4 == len(X_cols))

    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c1)
    assert(X_cols[0][2].dtype == np.int64)
    assert(len(X_cols[0][2]) == 3)
    assert(X_cols[0][2][0] == X_cols[0][3][str(val2)])
    assert(X_cols[0][2][1] == X_cols[0][3][str(val1)])
    assert(X_cols[0][2][2] == X_cols[0][3][str(val1)])

    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'nominal')
    assert(X_cols[1][4] is None)
    assert(X_cols[1][3] is c2)
    assert(X_cols[1][2].dtype == np.int64)
    assert(len(X_cols[1][2]) == 3)
    assert(X_cols[1][2][0] == 0)
    assert(X_cols[1][2][1] == X_cols[1][3][str(val2)])
    assert(X_cols[1][2][2] == X_cols[1][3][str(val1)])

    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'nominal')
    assert(X_cols[2][4] is None)
    assert(X_cols[2][3] is c1)
    assert(X_cols[2][2].dtype == np.int64)
    assert(len(X_cols[2][2]) == 3)
    assert(X_cols[2][2][0] == X_cols[2][3][str(val1)])
    assert(X_cols[2][2][1] == 0)
    assert(X_cols[2][2][2] == X_cols[2][3][str(val2)])

    assert(X_cols[3][0] == 3)
    assert(X_cols[3][1] == 'nominal')
    assert(X_cols[3][4] is None)
    assert(X_cols[3][3] is c2)
    assert(X_cols[3][2].dtype == np.int64)
    assert(len(X_cols[3][2]) == 3)
    assert(X_cols[3][2][0] == X_cols[3][3][str(val2)])
    assert(X_cols[3][2][1] == X_cols[3][3][str(val1)])
    assert(X_cols[3][2][2] == 0)

    assert(np.array_equal(X_cols[1][2] == 0, X.iloc[:, 1].isna()))
    assert(np.array_equal(X_cols[2][2] == 0, X.iloc[:, 2].isna()))
    assert(np.array_equal(X_cols[3][2] == 0, X.iloc[:, 3].isna()))

def check_pandas_float(dtype, val1, val2):
    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array([val2, val1, val1], dtype=np.object_), dtype=dtype)
    X["feature2"] = pd.Series(np.array([None, val2, val1], dtype=np.object_), dtype=dtype)
    X["feature3"] = pd.Series(np.array([val1, None, val2], dtype=np.object_), dtype=dtype)
    X["feature4"] = pd.Series(np.array([val2, val1, None], dtype=np.object_), dtype=dtype)

    X, n_samples = clean_X(X)
    assert(n_samples == 3)
    feature_names_out = unify_feature_names(X)

    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out, min_unique_continuous=0))
    assert(4 == len(X_cols))

    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'continuous')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2].dtype == np.float64)
    assert(X_cols[0][2][0] == np.float64(dtype(val2)))
    assert(X_cols[0][2][1] == np.float64(dtype(val1)))
    assert(X_cols[0][2][2] == np.float64(dtype(val1)))

    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'continuous')
    assert(X_cols[1][4] is None)
    assert(X_cols[1][3] is None)
    assert(X_cols[1][2].dtype == np.float64)
    assert(np.isnan(X_cols[1][2][0]))
    assert(X_cols[1][2][1] == np.float64(dtype(val2)))
    assert(X_cols[1][2][2] == np.float64(dtype(val1)))

    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'continuous')
    assert(X_cols[2][4] is None)
    assert(X_cols[2][3] is None)
    assert(X_cols[2][2].dtype == np.float64)
    assert(X_cols[2][2][0] == np.float64(dtype(val1)))
    assert(np.isnan(X_cols[2][2][1]))
    assert(X_cols[2][2][2] == np.float64(dtype(val2)))

    assert(X_cols[3][0] == 3)
    assert(X_cols[3][1] == 'continuous')
    assert(X_cols[3][4] is None)
    assert(X_cols[3][3] is None)
    assert(X_cols[3][2].dtype == np.float64)
    assert(X_cols[3][2][0] == np.float64(dtype(val2)))
    assert(X_cols[3][2][1] == np.float64(dtype(val1)))
    assert(np.isnan(X_cols[3][2][2]))

def check_numpy_throws(dtype_src, val1, val2):
    X = np.array([[val1, val2], [val1, val2]], dtype=dtype_src)
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    try:
        X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
        assert(False)
    except:
        pass

def test_process_continuous_float64():
    vals, bad = _process_continuous(np.array([3.5, 4.5], dtype=np.float64), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([3.5, 4.5], dtype=np.float64)))

def test_process_continuous_float32():
    vals, bad = _process_continuous(np.array([3.1, np.nan], dtype=np.float32), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 2)
    assert(vals[0] == 3.0999999046325684)
    assert(np.isnan(vals[1]))

def test_process_continuous_int8():
    vals, bad = _process_continuous(np.array([7, -9], dtype=np.int8), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([7, -9], dtype=np.float64)))

def test_process_continuous_uint16_missing():
    vals, bad = _process_continuous(np.array([7], dtype=np.uint16), np.array([True, False], dtype=np.bool_))
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 2)
    assert(vals[0] == 7)
    assert(np.isnan(vals[1]))

def test_process_continuous_bool():
    vals, bad = _process_continuous(np.array([False, True], dtype=np.bool_), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([0, 1], dtype=np.float64)))

def test_process_continuous_bool_missing():
    vals, bad = _process_continuous(np.array([False, True], dtype=np.bool_), np.array([True, False, True], dtype=np.bool_))
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 3)
    assert(vals[0] == 0)
    assert(np.isnan(vals[1]))
    assert(vals[2] == 1)

def test_process_continuous_obj_simple():
    vals, bad = _process_continuous(np.array([1, 2.5, "3", "4.5", np.float32("5.5")], dtype=np.object_), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([1, 2.5, 3, 4.5, 5.5], dtype=np.float64)))

def test_process_continuous_obj_simple_missing():
    vals, bad = _process_continuous(np.array([1, 2.5, "3", "4.5", np.float32("5.5")], dtype=np.object_), np.array([True, True, True, True, True, False], dtype=np.bool_))
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 6)
    assert(vals[0] == 1)
    assert(vals[1] == 2.5)
    assert(vals[2] == 3)
    assert(vals[3] == 4.5)
    assert(vals[4] == 5.5)
    assert(np.isnan(vals[5]))

def test_process_continuous_obj_hard():
    vals, bad = _process_continuous(np.array([1, 2.5, "3", "4.5", np.float32("5.5"), StringHolder("6.5"), DerivedStringHolder("7.5"), FloatHolder(8.5), DerivedFloatHolder(9.5), FloatAndStringHolder(10.5, "88"), DerivedFloatAndStringHolder(11.5, "99")], dtype=np.object_), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([1, 2.5, 3, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], dtype=np.float64)))

def test_process_continuous_obj_hard_missing():
    vals, bad = _process_continuous(np.array([1, 2.5, "3", "4.5", np.float32("5.5"), StringHolder("6.5")], dtype=np.object_), np.array([True, True, True, True, True, True, False], dtype=np.bool_))
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 7)
    assert(vals[0] == 1)
    assert(vals[1] == 2.5)
    assert(vals[2] == 3)
    assert(vals[3] == 4.5)
    assert(vals[4] == 5.5)
    assert(vals[5] == 6.5)
    assert(np.isnan(vals[6]))

def test_process_continuous_obj_hard_bad():
    vals, bad = _process_continuous(np.array([1, 2.5, "3", "4.5", np.float32("5.5"), StringHolder("6.5"), "bad", StringHolder("bad2"), NothingHolder("bad3")], dtype=np.object_), np.array([True, True, True, True, True, True, True, False, True, True], dtype=np.bool_))
    assert(len(bad) == 10)
    assert(bad[0] is None)
    assert(bad[1] is None)
    assert(bad[2] is None)
    assert(bad[3] is None)
    assert(bad[4] is None)
    assert(bad[5] is None)
    assert(bad[6] == "bad")
    assert(bad[7] is None)
    assert(bad[8] == "bad2")
    assert(isinstance(bad[9], str))
    assert(vals.dtype == np.float64)
    assert(len(vals) == 10)
    assert(vals[0] == 1)
    assert(vals[1] == 2.5)
    assert(vals[2] == 3)
    assert(vals[3] == 4.5)
    assert(vals[4] == 5.5)
    assert(vals[5] == 6.5)
    assert(np.isnan(vals[7]))

def test_process_continuous_str_simple():
    vals, bad = _process_continuous(np.array(["1", "2.5"], dtype=np.unicode_), None)
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(np.array_equal(vals, np.array([1, 2.5], dtype=np.float64)))

def test_process_continuous_str_simple_missing():
    vals, bad = _process_continuous(np.array(["1", "2.5"], dtype=np.unicode_), np.array([True, True, False], dtype=np.bool_))
    assert(bad is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 3)
    assert(vals[0] == 1)
    assert(vals[1] == 2.5)
    assert(np.isnan(vals[2]))

def test_process_continuous_str_hard_bad():
    vals, bad = _process_continuous(np.array(["1", "2.5", "bad"], dtype=np.unicode_), np.array([True, True, True, False], dtype=np.bool_))
    assert(len(bad) == 4)
    assert(bad[0] is None)
    assert(bad[1] is None)
    assert(bad[2] == "bad")
    assert(bad[3] is None)
    assert(vals.dtype == np.float64)
    assert(len(vals) == 4)
    assert(vals[0] == 1)
    assert(vals[1] == 2.5)
    assert(np.isnan(vals[3]))

def test_process_column_initial_int_float():
    # this test is hard since np.unique seems to think int(4) == float(4.0) so naively it returns just "4"
    encoded, c = _process_column_initial(np.array([4, 4.0], dtype=np.object_), None, None, None)
    assert(len(c) == 2)
    assert(c["4"] == 1)
    assert(c["4.0"] == 2)
    assert(np.array_equal(encoded, np.array([c["4"], c["4.0"]], dtype=np.int64)))

def test_process_column_initial_float32_float64():
    # np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 to float64 version has the lower mantisa bits
    # all set to zero, and there will be another float64 that will be closer to "0.1" in float64 representation, so
    # they aren't the same, but if to convert them to strings first then they are identical.  Strings are the 
    # ultimate arbiter of categorical membership since strings are cross-platform and JSON encodable.  np.unique 
    # will tend to separate the float32 and the float64 values since they aren't the same, but then serialize 
    # them to the same string.  The our model has ["0.1", "0.1"] as the categories if we don't convert to float64!
    encoded, c = _process_column_initial(np.array([np.float32(0.1), np.float64(0.1)], dtype=np.object_), None, None, None)
    assert(len(c) == 2)
    assert(c["0.1"] == 1)
    assert(c["0.10000000149011612"] == 2)
    assert(np.array_equal(encoded, np.array([c["0.10000000149011612"], c["0.1"]], dtype=np.int64)))

def test_process_column_initial_obj_obj():
    encoded, c = _process_column_initial(np.array([StringHolder("abc"), StringHolder("def")], dtype=np.object_), None, None, None)
    assert(len(c) == 2)
    assert(c["abc"] == 1)
    assert(c["def"] == 2)
    assert(np.array_equal(encoded, np.array([c["abc"], c["def"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_nomissing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), None, 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["abc"] == 1)
    assert(c["xyz"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], c["xyz"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_missing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["abc"] == 1)
    assert(c["xyz"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], 0, c["xyz"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_nomissing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), None, 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["xyz"] == 1)
    assert(c["abc"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], c["xyz"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_missing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["xyz"] == 1)
    assert(c["abc"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], 0, c["xyz"]], dtype=np.int64)))

def test_process_column_initial_prevalence_nomissing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), None, 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["xyz"] == 1)
    assert(c["abc"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], c["xyz"]], dtype=np.int64)))

def test_process_column_initial_prevalence_missing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["xyz"] == 1)
    assert(c["abc"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], 0, c["xyz"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_nomissing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), None, 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["abc"] == 1)
    assert(c["xyz"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], c["xyz"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_missing():
    encoded, c = _process_column_initial(np.array(["xyz", "abc", "xyz"], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["abc"] == 1)
    assert(c["xyz"] == 2)
    assert(np.array_equal(encoded, np.array([c["xyz"], c["abc"], 0, c["xyz"]], dtype=np.int64)))

def test_process_column_initial_float64_nomissing():
    encoded, c = _process_column_initial(np.array(["11.1", "2.2", "11.1"], dtype=np.unicode_), None, 'ANYTHING_ELSE', None)
    assert(len(c) == 2)
    assert(c["2.2"] == 1)
    assert(c["11.1"] == 2)
    assert(np.array_equal(encoded, np.array([c["11.1"], c["2.2"], c["11.1"]], dtype=np.int64)))

def test_process_column_initial_float64_missing():
    encoded, c = _process_column_initial(np.array(["11.1", "2.2", "11.1"], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), 'ANYTHING_ELSE', None)
    assert(len(c) == 2)
    assert(c["2.2"] == 1)
    assert(c["11.1"] == 2)
    assert(np.array_equal(encoded, np.array([c["11.1"], c["2.2"], 0, c["11.1"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_nomissing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), None, 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["-1"] == 1)
    assert(c["1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], c["1"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_missing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["-1"] == 1)
    assert(c["1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], 0, c["1"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_nomissing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), None, 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["1"] == 1)
    assert(c["-1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], c["1"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_missing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["1"] == 1)
    assert(c["-1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], 0, c["1"]], dtype=np.int64)))

def test_process_column_initial_prevalence_nomissing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), None, 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["1"] == 1)
    assert(c["-1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], c["1"]], dtype=np.int64)))

def test_process_column_initial_prevalence_missing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["1"] == 1)
    assert(c["-1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], 0, c["1"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_nomissing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), None, 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["-1"] == 1)
    assert(c["1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], c["1"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_missing_int8():
    encoded, c = _process_column_initial(np.array([1, -1, 1], dtype=np.int8), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["-1"] == 1)
    assert(c["1"] == 2)
    assert(np.array_equal(encoded, np.array([c["1"], c["-1"], 0, c["1"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_nomissing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), None, 'nominal_alphabetical', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_nomissing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), None, 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["False"] == 1)
    assert(c["True"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_missing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_missing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), np.array([True, True, False, True, True], dtype=np.bool_), 'nominal_alphabetical', None)
    assert(len(c) == 2)
    assert(c["False"] == 1)
    assert(c["True"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_nomissing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), None, 'nominal_alphabetical_reversed', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_nomissing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), None, 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["True"] == 1)
    assert(c["False"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_missing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_alphabetical_reversed', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["True"]], dtype=np.int64)))

def test_process_column_initial_alphabetical_reversed_missing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), np.array([True, True, False, True, True], dtype=np.bool_), 'nominal_alphabetical_reversed', None)
    assert(len(c) == 2)
    assert(c["True"] == 1)
    assert(c["False"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_nomissing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), None, 'nominal_prevalence', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_nomissing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), None, 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["True"] == 1)
    assert(c["False"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_missing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_missing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), np.array([True, True, False, True, True], dtype=np.bool_), 'nominal_prevalence', None)
    assert(len(c) == 2)
    assert(c["True"] == 1)
    assert(c["False"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_nomissing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), None, 'nominal_prevalence_reversed', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_nomissing_two_bool():
    encoded, c = _process_column_initial(np.array([True, True, False, True], dtype=np.bool_), None, 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["False"] == 1)
    assert(c["True"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], c["False"], c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_missing_one_bool():
    encoded, c = _process_column_initial(np.array([True, True, True], dtype=np.bool_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence_reversed', None)
    assert(len(c) == 1)
    assert(c["True"] == 1)
    assert(np.array_equal(encoded, np.array([c["True"], c["True"], 0, c["True"]], dtype=np.int64)))

def test_process_column_initial_prevalence_reversed_missing_two_bool():
    encoded, c = _process_column_initial(np.array([True, False, True], dtype=np.bool_), np.array([True, True, False, True], dtype=np.bool_), 'nominal_prevalence_reversed', None)
    assert(len(c) == 2)
    assert(c["False"] == 1)
    assert(c["True"] == 2)
    assert(np.array_equal(encoded, np.array([c["True"], c["False"], 0, c["True"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_str():
    c = {"cd": 1, "ab": 2}
    encoded, bad = _encode_categorical_existing(np.array(["ab", "cd"], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["ab"], c["cd"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_bool():
    c = {"True": 1, "False": 2}
    encoded, bad = _encode_categorical_existing(np.array([True, False], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["True"], c["False"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_int_small():
    c = {"-2": 1, "3": 2, "1": 3}
    encoded, bad = _encode_categorical_existing(np.array([int(1), np.int8(-2), np.uint64(3)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["1"], c["-2"], c["3"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_int_big():
    c = {"-2": 1, "18446744073709551615": 2, "1": 3}
    encoded, bad = _encode_categorical_existing(np.array([int(1), np.int8(-2), np.uint64("18446744073709551615")], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["1"], c["-2"], c["18446744073709551615"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_floats():
    # np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 to float64 version has the lower mantisa bits
    # all set to zero, and there will be another float64 that will be closer to "0.1" in float64 representation, so
    # they aren't the same, but if to convert them to strings first then they are identical.  Strings are the 
    # ultimate arbiter of categorical membership since strings are cross-platform and JSON encodable.  np.unique 
    # will tend to separate the float32 and the float64 values since they aren't the same, but then serialize 
    # them to the same string.  The our model has ["0.1", "0.1"] as the categories if we don't convert to float64!

    c = {"1.1": 1, "2.19921875": 2, "3.299999952316284": 3, "4.4": 4, "5.5": 5}
    encoded, bad = _encode_categorical_existing(np.array([float(1.1), np.float16(2.2), np.float32(3.3), np.float64(4.4), np.longfloat(5.5)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["1.1"], c["2.19921875"], c["3.299999952316284"], c["4.4"], c["5.5"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_str_int():
    c = {"abc": 1, "1": 2}
    encoded, bad = _encode_categorical_existing(np.array(["abc", int(1)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["abc"], c["1"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_str_float():
    c = {"abc": 1, "1.1": 2}
    encoded, bad = _encode_categorical_existing(np.array(["abc", float(1.1)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["abc"], c["1.1"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_str_float64():
    c = {"abc": 1, "1.1": 2}
    encoded, bad = _encode_categorical_existing(np.array(["abc", np.float64(1.1)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["abc"], c["1.1"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_str_float32():
    c = {"abc": 1, "1.100000023841858": 2}
    encoded, bad = _encode_categorical_existing(np.array(["abc", np.float32(1.1)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["abc"], c["1.100000023841858"]], dtype=np.int64)))

def test_encode_categorical_existing_int_float():
    # this test is hard since np.unique seems to think int(4) == float(4) so naively it returns just "4"
    c = {"4": 1, "4.0": 2}
    encoded, bad = _encode_categorical_existing(np.array([int(4), 4.0], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["4"], c["4.0"]], dtype=np.int64)))

def test_encode_categorical_existing_int_float32():
    # if you take np.float64(np.float32(0.1)) != np.float64(0.1) since the float32 version has the lower mantisa
    # bits all set to zero, and there will be another float64 that will be closer to "0.1" for float64s, so
    # they aren't the same, but if to convert them to strings first then they are identical.  I tend to think
    # of strings are the ultimate arbiter of categorical membership since strings are cross-platform
    # np.unique will tend to separate the float32 and the float64 values since they aren't the same, but then
    # serialize them to the same string.  The our model has ["0.1", "0.1"] as the categories!!
    c = {"4": 1, "0.10000000149011612": 2}
    encoded, bad = _encode_categorical_existing(np.array([int(4), np.float32(0.1)], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["4"], c["0.10000000149011612"]], dtype=np.int64)))

def test_encode_categorical_existing_obj_obj():
    c = {"abc": 1, "def": 2}
    encoded, bad = _encode_categorical_existing(np.array([StringHolder("abc"), StringHolder("def")], dtype=np.object_), None, c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["abc"], c["def"]], dtype=np.int64)))

def test_encode_categorical_existing_str():
    c = {"abc": 1, "def": 2, "ghi": 3}
    encoded, bad = _encode_categorical_existing(np.array(["abc", "ghi", "def", "something"], dtype=np.unicode_), np.array([True, True, False, True, True], dtype=np.bool_), c)
    assert(np.array_equal(bad, np.array([None, None, None, None, "something"], dtype=np.object_)))
    assert(np.array_equal(encoded, np.array([c["abc"], c["ghi"], 0, c["def"], -1], dtype=np.int64)))

def test_encode_categorical_existing_int8():
    c = {"5": 1, "0": 2, "-9": 3}
    encoded, bad = _encode_categorical_existing(np.array([5, -9, 0, 0, -9, 5, 99], dtype=np.int8), np.array([True, True, True, False, True, True, True, True], dtype=np.bool_), c)
    assert(np.array_equal(bad, np.array([None, None, None, None, None, None, None, "99"], dtype=np.object_)))
    assert(np.array_equal(encoded, np.array([c["5"], c["-9"], c["0"], 0, c["0"], c["-9"], c["5"], -1], dtype=np.int64)))

def test_encode_categorical_existing_bool():
    c = {"False": 1, "True": 2}
    encoded, bad = _encode_categorical_existing(np.array([False, True, False], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), c)
    assert(bad is None)
    assert(np.array_equal(encoded, np.array([c["False"], c["True"], 0, c["False"]], dtype=np.int64)))

def test_encode_categorical_existing_bool_true():
    c = {"True": 1}
    encoded, bad = _encode_categorical_existing(np.array([False, True, False], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), c)
    assert(np.array_equal(bad, np.array(["False", None, None, "False"], dtype=np.object_)))
    assert(np.array_equal(encoded, np.array([-1, c["True"], 0, -1], dtype=np.int64)))

def test_encode_categorical_existing_bool_false():
    c = {"False": 1}
    encoded, bad = _encode_categorical_existing(np.array([False, True, False], dtype=np.unicode_), np.array([True, True, False, True], dtype=np.bool_), c)
    assert(np.array_equal(bad, np.array([None, "True", None, None], dtype=np.object_)))
    assert(np.array_equal(encoded, np.array([c["False"], -1, 0, c["False"]], dtype=np.int64)))

def test_process_column_initial_choose_floatcategories():
    encoded, c = _process_column_initial(np.array([11.11, 2.2, np.float32(2.2), "2.2", StringHolder("2.2")], dtype=np.object_), None, None, 4)
    assert(c["2.2"] == 1)
    assert(c["2.200000047683716"] == 2)
    assert(c["11.11"] == 3)
    assert(np.array_equal(encoded, np.array([c["11.11"], c["2.2"], c["2.200000047683716"], c["2.2"], c["2.2"]], dtype=np.int64)))

def test_process_column_initial_choose_floats():
    encoded, c = _process_column_initial(np.array([11.11, 2.2, np.float32(2.2), "2.2", StringHolder("2.2"), 3.3, 3.3], dtype=np.object_), None, None, 3)
    assert(c is None)
    assert(np.array_equal(encoded, np.array([11.11, 2.2, 2.200000047683716, 2.2, 2.2, 3.3, 3.3], dtype=np.float64)))

def test_unify_columns_numpy1():
    X = np.array([1, 2, 3])
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(np.array_equal(X_cols[0][2], np.array([X_cols[0][3]["1"]], dtype=np.int64)))
    assert(np.array_equal(X_cols[1][2], np.array([X_cols[1][3]["2"]], dtype=np.int64)))
    assert(np.array_equal(X_cols[2][2], np.array([X_cols[2][3]["3"]], dtype=np.int64)))

def test_unify_columns_numpy2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(np.array_equal(X_cols[0][2], np.array([X_cols[0][3]["1"], X_cols[0][3]["4"]], dtype=np.int64)))
    assert(np.array_equal(X_cols[1][2], np.array([X_cols[1][3]["2"], X_cols[1][3]["5"]], dtype=np.int64)))
    assert(np.array_equal(X_cols[2][2], np.array([X_cols[2][3]["3"], X_cols[2][3]["6"]], dtype=np.int64)))

def test_unify_columns_numpy_ignore():
    X = np.array([["abc", None, "def"], ["ghi", "jkl", None]])
    feature_types_in=['ignore', 'ignore', 'ignore']
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out, feature_types_in))
    assert(3 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'ignore')
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2] is None)
    assert(np.array_equal(X_cols[0][4], np.array(["abc", "ghi"], dtype=np.object_)))
    assert(X_cols[1][0] == 1)
    assert(X_cols[1][1] == 'ignore')
    assert(X_cols[1][3] is None)
    assert(X_cols[1][2] is None)
    assert(np.array_equal(X_cols[1][4], np.array([None, "jkl"], dtype=np.object_)))
    assert(X_cols[2][0] == 2)
    assert(X_cols[2][1] == 'ignore')
    assert(X_cols[2][3] is None)
    assert(X_cols[2][2] is None)
    assert(np.array_equal(X_cols[2][4], np.array(["def", None], dtype=np.object_)))

def test_unify_columns_scipy():
    X = sp.sparse.csc_matrix([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([X_cols[0][3]["1"], X_cols[0][3]["4"]], dtype=np.int64)))
    assert(X_cols[1][2].dtype == np.int64)
    assert(np.array_equal(X_cols[1][2], np.array([X_cols[1][3]["2"], X_cols[1][3]["5"]], dtype=np.int64)))
    assert(X_cols[2][2].dtype == np.int64)
    assert(np.array_equal(X_cols[2][2], np.array([X_cols[2][3]["3"], X_cols[2][3]["6"]], dtype=np.int64)))

def test_unify_columns_dict1():
    X = {"feature1" : [1], "feature2" : "hi", "feature3" : None}
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X, feature_names_in=["feature3", "feature2", "feature1"])
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(X_cols[0][2][0] == 0)
    assert(X_cols[1][2].dtype == np.int64)
    assert(X_cols[1][2][0] == X_cols[1][3]["hi"])
    assert(X_cols[2][2].dtype == np.int64)
    assert(X_cols[2][2][0] == X_cols[2][3]["1"])

def test_unify_columns_dict2():
    X = {"feature1" : [1, 4], "feature2" : [2, 5], "feature3" : [3, 6]}
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_names_in=["feature3", "feature2", "feature1"])
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([X_cols[0][3]["3"], X_cols[0][3]["6"]], dtype=np.int64)))
    assert(X_cols[1][2].dtype == np.int64)
    assert(np.array_equal(X_cols[1][2], np.array([X_cols[1][3]["2"], X_cols[1][3]["5"]], dtype=np.int64)))
    assert(X_cols[2][2].dtype == np.int64)
    assert(np.array_equal(X_cols[2][2], np.array([X_cols[2][3]["1"], X_cols[2][3]["4"]], dtype=np.int64)))

def test_unify_columns_list1():
    X = [1, 2.0, "hi", None]
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(4 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(X_cols[0][2][0] == X_cols[0][3]["1"])
    assert(X_cols[1][2].dtype == np.int64)
    assert(X_cols[1][2][0] == X_cols[1][3]["2.0"])
    assert(X_cols[2][2].dtype == np.int64)
    assert(X_cols[2][2][0] == X_cols[2][3]["hi"])
    assert(X_cols[3][2].dtype == np.int64)
    assert(X_cols[3][2][0] == 0)

def test_unify_columns_list2():
    P1 = pd.DataFrame()
    P1["feature1"] = pd.Series(np.array([1, None, np.nan], dtype=np.object_))
    P2 = pd.DataFrame()
    P2["feature1"] = pd.Series(np.array([1], dtype=np.float32))
    P2["feature2"] = pd.Series(np.array([None], dtype=np.object_))
    P2["feature3"] = pd.Series(np.array([np.nan], dtype=np.object_))
    S1 = sp.sparse.csc_matrix([[1, 2, 3]])
    S2 = sp.sparse.csc_matrix([[1], [2], [3]])
    X = [np.array([1, 2, 3], dtype=np.int8), pd.Series([4.0, None, np.nan]), [1, 2.0, "hi"], (np.double(4.0), "bye", None), {1, 2, 3}, {"abc": 1, "def": 2, "ghi":3}.keys(), {"abc": 1, "def": 2, "ghi":3}.values(), range(1, 4), (x for x in [1, 2, 3]), np.array([1, 2, 3], dtype=np.object_), np.array([[1, 2, 3]], dtype=np.int8), np.array([[1], [2], [3]], dtype=np.int8), P1, P2, S1, S2]
    X, n_samples = clean_X(X)
    assert(n_samples == 16)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([c["1"], c["4.0"], c["1"], c["4.0"], c["1"], c["abc"], c["1"], c["1"], c["1"], c["1"], c["1"], c["1"], c["1"], c["1.0"], c["1"], c["1"]], dtype=np.int64)))
    assert(X_cols[1][2].dtype == np.int64)
    c = X_cols[1][3]
    assert(np.array_equal(X_cols[1][2], np.array([c["2"], 0, c["2.0"], c["bye"], c["2"], c["def"], c["2"], c["2"], c["2"], c["2"], c["2"], c["2"], 0, 0, c["2"], c["2"]], dtype=np.int64)))
    assert(X_cols[2][2].dtype == np.int64)
    c = X_cols[2][3]
    assert(np.array_equal(X_cols[2][2], np.array([c["3"], 0, c["hi"], 0, c["3"], c["ghi"], c["3"], c["3"], c["3"], c["3"], c["3"], c["3"], 0, 0, c["3"], c["3"]], dtype=np.int64)))

def test_unify_columns_tuple1():
    X = (1, 2.0, "hi", None)
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(4 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(X_cols[0][2][0] == X_cols[0][3]["1"])
    assert(X_cols[1][2].dtype == np.int64)
    assert(X_cols[1][2][0] == X_cols[1][3]["2.0"])
    assert(X_cols[2][2].dtype == np.int64)
    assert(X_cols[2][2][0] == X_cols[2][3]["hi"])
    assert(X_cols[3][2].dtype == np.int64)
    assert(X_cols[3][2][0] == 0)

def test_unify_columns_tuple2():
    X = (np.array([1, 2, 3], dtype=np.int8), pd.Series([4, 5, 6]), [1, 2.0, "hi"], (np.double(4.0), "bye", None), {1, 2, 3}, {"abc": 1, "def": 2, "ghi":3}.keys(), {"abc": 1, "def": 2, "ghi":3}.values(), range(1, 4), (x for x in [1, 2, 3]), np.array([1, 2, 3], dtype=np.object_))
    X, n_samples = clean_X(X)
    assert(n_samples == 10)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([c["1"], c["4"], c["1"], c["4.0"], c["1"], c["abc"], c["1"], c["1"], c["1"], c["1"]], dtype=np.int64)))
    assert(X_cols[1][2].dtype == np.int64)
    c = X_cols[1][3]
    assert(np.array_equal(X_cols[1][2], np.array([c["2"], c["5"], c["2.0"], c["bye"], c["2"], c["def"], c["2"], c["2"], c["2"], c["2"]], dtype=np.int64)))
    assert(X_cols[2][2].dtype == np.int64)
    c = X_cols[2][3]
    assert(np.array_equal(X_cols[2][2], np.array([c["3"], c["6"], c["hi"], 0, c["3"], c["ghi"], c["3"], c["3"], c["3"], c["3"]], dtype=np.int64)))

def test_unify_columns_generator1():
    X = (x for x in [1, 2.0, "hi", None])
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(4 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    assert(X_cols[0][2][0] == X_cols[0][3]["1"])
    assert(X_cols[1][2].dtype == np.int64)
    assert(X_cols[1][2][0] == X_cols[1][3]["2.0"])
    assert(X_cols[2][2].dtype == np.int64)
    assert(X_cols[2][2][0] == X_cols[2][3]["hi"])
    assert(X_cols[3][2].dtype == np.int64)
    assert(X_cols[3][2][0] == 0)

def test_unify_columns_generator2():
    X = (x for x in [np.array([1, 2, 3], dtype=np.int8), pd.Series([4, 5, 6]), [1, 2.0, "hi"], (np.double(4.0), "bye", None), {1, 2, 3}, {"abc": 1, "def": 2, "ghi":3}.keys(), {"abc": 1, "def": 2, "ghi":3}.values(), range(1, 4), (x for x in [1, 2, 3]), np.array([1, 2, 3], dtype=np.object_)])
    X, n_samples = clean_X(X)
    assert(n_samples == 10)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(3 == len(X_cols))
    assert(X_cols[0][2].dtype == np.int64)
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([c["1"], c["4"], c["1"], c["4.0"], c["1"], c["abc"], c["1"], c["1"], c["1"], c["1"]], dtype=np.int64)))
    assert(X_cols[1][2].dtype == np.int64)
    c = X_cols[1][3]
    assert(np.array_equal(X_cols[1][2], np.array([c["2"], c["5"], c["2.0"], c["bye"], c["2"], c["def"], c["2"], c["2"], c["2"], c["2"]], dtype=np.int64)))
    assert(X_cols[2][2].dtype == np.int64)
    c = X_cols[2][3]
    assert(np.array_equal(X_cols[2][2], np.array([c["3"], c["6"], c["hi"], 0, c["3"], c["ghi"], c["3"], c["3"], c["3"], c["3"]], dtype=np.int64)))

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
    check_pandas_float(np.longfloat, -1.1, 2.2)

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
    check_pandas_missings(pd.UInt64Dtype(), np.uint64("0"), np.uint64("18446744073709551615"))

def test_unify_columns_pandas_missings_BooleanDtype():
    check_pandas_missings(pd.BooleanDtype(), False, True)

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
        X, n_samples = clean_X(X)
        assert(False)
    except:
        pass
    try:
        feature_names_out = unify_feature_names(X)
        assert(False)
    except:
        pass
    try:
        feature_names_out = ["ANYTHING"]
        X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
        assert(False)
    except:
        pass

def test_unify_columns_int_throw():
    X = 1
    try:
        X, n_samples = clean_X(X)
        assert(False)
    except:
        pass
    try:
        feature_names_out = unify_feature_names(X)
        assert(False)
    except:
        pass
    try:
        feature_names_out = ["ANYTHING"]
        X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
        assert(False)
    except:
        pass

def test_unify_columns_duplicate_colnames_throw():
    X = pd.DataFrame()
    X["0"] = [1, 2]
    X[0] = [3, 4]

    try:
        feature_names_out = unify_feature_names(X)
        assert(False)
    except:
        pass
    try:
        feature_names_out = ["ANYTHING"]
        X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
        assert(False)
    except:
        pass

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
    X["feature1"] = pd.Series(np.array([None, np.nan, np.float16(np.nan), 0, -1, 2.2, "-3.3", np.float16("4.4"), StringHolder("-5.5"), np.float32("6.6").item()], dtype=np.object_), dtype=np.object_)
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 10)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'continuous')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2].dtype == np.float64)
    assert(np.isnan(X_cols[0][2][0]))
    assert(np.isnan(X_cols[0][2][1]))
    assert(np.isnan(X_cols[0][2][2]))
    assert(X_cols[0][2][3] == 0)
    assert(X_cols[0][2][4] == -1)
    assert(X_cols[0][2][5] == 2.2)
    assert(X_cols[0][2][6] == -3.3)
    assert(X_cols[0][2][7] == 4.3984375)
    assert(X_cols[0][2][8] == -5.5)
    assert(X_cols[0][2][9] == 6.5999999046325684) # python internal objects are float64

def test_unify_columns_pandas_obj_to_str():
    X = pd.DataFrame()
    X["feature1"] = pd.Series(np.array([None, np.nan, np.float16(np.nan), 0, -1, 2.2, "-3.3", np.float16("4.4"), StringHolder("-5.5"), 5.6843418860808014e-14, "None", "nan"], dtype=np.object_), dtype=np.object_)
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 12)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))

    # For "5.684341886080802e-14", we need to round the 16th digit up for this to be the shortest string since 
    # "5.684341886080801e-14" doesn't work
    # https://www.exploringbinary.com/the-shortest-decimal-string-that-round-trips-may-not-be-the-nearest/
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["0"], c["-1"], c["2.2"], c["-3.3"], c["4.3984375"], c["-5.5"], c["5.684341886080802e-14"], c["None"], c["nan"]], dtype=np.int64)))
    assert(np.array_equal(na, X_cols[0][2] == 0))

def test_unify_columns_pandas_categorical():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(len(X_cols[0][3]) == 3)
    assert(X_cols[0][3]["a"] == 1)
    assert(X_cols[0][3]["0"] == 2)
    assert(X_cols[0][3]["bcd"] == 3)
    assert(X_cols[0][2].dtype == np.int64)
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_ordinal():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=True))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    X_cols = list(unify_columns(X, zip(range(len(feature_names_out)), repeat(None)), feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'ordinal')
    assert(X_cols[0][4] is None)
    assert(len(X_cols[0][3]) == 3)
    assert(X_cols[0][3]["a"] == 1)
    assert(X_cols[0][3]["0"] == 2)
    assert(X_cols[0][3]["bcd"] == 3)
    assert(X_cols[0][2].dtype == np.int64)
    c = X_cols[0][3]
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_shorter():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "0"], dtype=pd.CategoricalDtype(categories=["a", "0"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 5)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_equals():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "0", "bcd"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_longer():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "0", "bcd", "in_categories"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:2]))
    assert(all(~na[2:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(np.array_equal(X_cols[0][4], np.array([None, None, "in_categories", None, None, None], dtype=np.object_)))
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, -1, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_reordered_shorter():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "0"], dtype=pd.CategoricalDtype(categories=["0", "a"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 5)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_reordered_equals():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "bcd", "0"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_reordered_longer1():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "0", "in_categories", "bcd"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:2]))
    assert(all(~na[2:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(np.array_equal(X_cols[0][4], np.array([None, None, "in_categories", None, None, None], dtype=np.object_)))
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, -1, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_reordered_longer2():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["0", "a", "bcd", "in_categories"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:2]))
    assert(all(~na[2:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    c = {"a": 1, "0": 2, "bcd": 3}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(np.array_equal(X_cols[0][4], np.array([None, None, "in_categories", None, None, None], dtype=np.object_)))
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, -1, c["a"], c["bcd"], c["0"]], dtype=np.int64)))

def test_unify_columns_pandas_categorical_compressed_categories():
    X = pd.DataFrame()
    X["feature1"] = pd.Series([None, np.nan, "not_in_categories", "a", "bcd", "0"], dtype=pd.CategoricalDtype(categories=["a", "bcd", "0"], ordered=False))
    na = X["feature1"].isna()
    assert(all(na[0:3]))
    assert(all(~na[3:]))
    X, n_samples = clean_X(X)
    assert(n_samples == 6)
    feature_names_out = unify_feature_names(X)
    # here we're combining the "a" category and the "0" category into a single one that tracks both.
    # in JSON this can be expressed as the equivalent of [["a", "0"], "bcd"]
    c = {"a": 1, "0": 1, "bcd": 2}
    X_cols = list(unify_columns(X, [(0, c)], feature_names_out))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'nominal')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is c)
    assert(X_cols[0][2].dtype == np.int64)
    assert(np.array_equal(X_cols[0][2], np.array([0, 0, 0, c["a"], c["bcd"], c["0"]], dtype=np.int64)))


def test_unify_feature_names_numpy1():
    X = np.array([1, 2, 3])
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[2][2][0] == 3.0)

def test_unify_feature_names_numpy2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_data_frame1():
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_data_frame2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_scipy():
    X = sp.sparse.csc_matrix([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_dict1():
    X = {"feature1" : [1], "feature2" : [2], "feature3" : [3]}
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[2][2][0] == 3.0)

def test_unify_feature_names_dict2():
    X = {"feature2" : [1, 4], "feature1" : [2, 5], "feature3" : [3, 6]}
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 2.0)
    assert(X_cols[0][2][1] == 5.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_list1():
    X = [1, 2, 3]
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[2][2][0] == 3.0)

def test_unify_feature_names_list2():
    X = [pd.Series([1, 2, 3]), (4, 5, 6)]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_tuple1():
    X = (1, 2, 3)
    X, n_samples = clean_X(X)
    assert(n_samples == 1)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[2][2][0] == 3.0)

def test_unify_feature_names_tuple2():
    X = (np.array([1, 2, 3]), [4, 5, 6])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_feature_types1():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_feature_types2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_feature_types3():
    X = np.array([[1, 3], [4, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in = ['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature_0001", "feature_0002", "feature_0003"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_pandas_feature_types1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_pandas_ignored_existing():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    feature_types_in=['continuous', 'ignore', 'continuous']
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_pandas_feature_types3():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature_0001", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_names1():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]))
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2])
    assert(isinstance(feature_names_out, list))
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]))
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2])
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_names1():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_ignored_names1():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_ignored_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_ignored_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_ignored_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped_names1():
    X = np.array([[1, 3], [4, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped_names2():
    X = np.array([[1, 3], [4, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 1, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "1", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, "SOMETHING", 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "SOMETHING", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_nondropped2_names2():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "feature_0001", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_nondropped2_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "feature_0001", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped2_names2():
    X = np.array([[1, 3], [4, 6]])
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=[0, 2], feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "feature_0001", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped2_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=pd.Series([0, 2]), feature_types_in=feature_types_in)
    assert(feature_names_out == ["0", "feature_0001", "2"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_keep_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 2.0)
    assert(X_cols[1][2][1] == 5.0)
    assert(X_cols[2][2][0] == 3.0)
    assert(X_cols[2][2][1] == 6.0)

def test_unify_feature_names_types_dropped3_pandas_names1():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature2"] = [2, 5]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature2", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_dropped3_pandas_names2():
    X = pd.DataFrame()
    X["feature1"] = [1, 4]
    X["feature3"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_types_in=feature_types_in)
    assert(feature_names_out == ["feature1", "feature_0001", "feature3"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 1.0)
    assert(X_cols[0][2][1] == 4.0)
    assert(X_cols[1][2][0] == 3.0)
    assert(X_cols[1][2][1] == 6.0)

def test_unify_feature_names_types_rearrange1_drop1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "SOMETHING", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "SOMETHING", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange1_drop2():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["length"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "SOMETHING", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "SOMETHING", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange2_drop1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "feature_0001", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange2_drop2():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["length"] = [3, 6]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "feature_0001", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange_more1():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED1"] = [2, 5]
    X["length"] = [3, 6]
    X["UNUSED2"] = [9, 9]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "SOMETHING", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "SOMETHING", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange_more2():
    X = pd.DataFrame()
    X["width"] = [1, 4]
    X["UNUSED1"] = [2, 5]
    X["length"] = [3, 6]
    X["UNUSED2"] = [9, 9]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'ignore', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "width"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "feature_0001", "width"])
    X_cols = list(unify_columns(X, [(0, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(2 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 1.0)
    assert(X_cols[1][2][1] == 4.0)

def test_unify_feature_names_types_rearrange_more3():
    X = pd.DataFrame()
    X["height"] = [1, 4]
    X["UNUSED"] = [2, 5]
    X["length"] = [3, 6]
    X["width"] = [8, 9]
    X, n_samples = clean_X(X)
    assert(n_samples == 2)
    feature_types_in=['continuous', 'continuous', 'continuous']
    feature_names_out = unify_feature_names(X, feature_names_in=["length", "width", "height"], feature_types_in=feature_types_in)
    assert(feature_names_out == ["length", "width", "height"])
    X_cols = list(unify_columns(X, [(0, None), (1, None), (2, None)], feature_names_out, feature_types_in, min_unique_continuous=0))
    assert(3 == len(X_cols))
    assert(X_cols[0][2][0] == 3.0)
    assert(X_cols[0][2][1] == 6.0)
    assert(X_cols[1][2][0] == 8.0)
    assert(X_cols[1][2][1] == 9.0)
    assert(X_cols[2][2][0] == 1.0)
    assert(X_cols[2][2][1] == 4.0)

def test_unify_columns_ma_no_mask():
    X = ma.array([[np.nan], [1], [None]], dtype=np.object_)
    assert(X.mask is ma.nomask)
    X, n_samples = clean_X(X)
    assert(n_samples == 3)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001"])
    X_cols = list(unify_columns(X, [(0, None)], feature_names_out, min_unique_continuous=0))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'continuous')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2].dtype == np.float64)
    assert(np.isnan(X_cols[0][2][0]))
    assert(X_cols[0][2][1] == 1)
    assert(np.isnan(X_cols[0][2][2]))

def test_unify_columns_ma_empty_mask():
    X = ma.array([[np.nan], [1], [None]], mask=[[0], [0], [0]], dtype=np.object_)
    assert(X.mask is not ma.nomask)
    X, n_samples = clean_X(X)
    assert(n_samples == 3)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001"])
    X_cols = list(unify_columns(X, [(0, None)], feature_names_out, min_unique_continuous=0))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'continuous')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2].dtype == np.float64)
    assert(np.isnan(X_cols[0][2][0]))
    assert(X_cols[0][2][1] == 1)
    assert(np.isnan(X_cols[0][2][2]))

def test_unify_columns_ma_objects():
    X = ma.array([[np.nan], [None], [1], [2], [None], [3], [np.nan]], mask=[[0], [1], [0], [1], [0], [0], [1]], dtype=np.object_)
    X, n_samples = clean_X(X)
    assert(n_samples == 7)
    feature_names_out = unify_feature_names(X)
    assert(feature_names_out == ["feature_0001"])
    X_cols = list(unify_columns(X, [(0, None)], feature_names_out, min_unique_continuous=0))
    assert(1 == len(X_cols))
    assert(X_cols[0][0] == 0)
    assert(X_cols[0][1] == 'continuous')
    assert(X_cols[0][4] is None)
    assert(X_cols[0][3] is None)
    assert(X_cols[0][2].dtype == np.float64)
    assert(np.isnan(X_cols[0][2][0]))
    assert(np.isnan(X_cols[0][2][1]))
    assert(X_cols[0][2][2] == 1)
    assert(np.isnan(X_cols[0][2][3]))
    assert(np.isnan(X_cols[0][2][4]))
    assert(X_cols[0][2][5] == 3)
    assert(np.isnan(X_cols[0][2][6]))

def test_bin_native():
    X = np.array([["a", 1, np.nan], ["b", 2, 7], ["a", 2, 8], [None, 3, 9]], dtype=np.object_)
    feature_names_in = ["f1", 99, "f3"]
    feature_types_in = ['nominal', 'nominal', 'continuous']
    y = np.array(["a", 99, 99, "b"])
    w = np.array([0.5, 1.1, 0.1, 0.5])
    feature_idxs = range(len(feature_names_in)) if feature_types_in is None else [feature_idx for feature_idx, feature_type in zip(count(), feature_types_in) if feature_type != 'ignore']
    shared_dataset, feature_names_out, feature_types_out, bins_out, classes = bin_native(True, feature_idxs, repeat(256), X, y, w, feature_names_in, feature_types_in)
    assert(shared_dataset is not None)
    assert(feature_names_out is not None)
    assert(feature_types_out is not None)
    assert(bins_out is not None)

def test_score_terms():
    X = np.array([["a", 1, np.nan], ["b", 2, 8], ["a", 2, 9], [None, 3, "BAD_CONTINUOUS"]], dtype=np.object_)
    feature_names_out = ["f1", "99", "f3"]
    feature_types_out = ['nominal', 'nominal', 'continuous']

    shared_categores = {"a": 1} # "b" is unknown category
    shared_cuts = np.array([8.5], dtype=np.float64)

    TestPreprocessor = namedtuple('TestPreprocessor', 'bins_')

    preprocessor = TestPreprocessor([{"a": 1, "b": 2}, {"1": 1, "2": 2, "3": 3}, shared_cuts])
    # "b" is unknown category
    # we combine "2" and "3" into one category!
    pair_preprocessor = TestPreprocessor([shared_categores, {"1": 1, "2": 2, "3": 2}, shared_cuts])
    # collapse all our categories to keep the tensor small for testing
    higher_preprocessor = TestPreprocessor([shared_categores, {"1": 1, "2": 1, "3": 1}, np.array([], dtype=np.float64)])

    term0 = {}
    term0['features'] = [0]
    term0['scores'] = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    term1 = {}
    term1['features'] = [1]
    term1['scores'] = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)

    term2 = {}
    term2['features'] = [2]
    term2['scores'] = np.array([0.01, 0.02, 0.03], dtype=np.float64)

    term3 = {}
    term3['features'] = [0, 1]
    term3['scores'] = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]], dtype=np.float64)

    term4 = {}
    term4['features'] = [0, 2]
    term4['scores'] = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]], dtype=np.float64)

    term5 = {}
    term5['features'] = [0, 1, 2]
    term5['scores'] = np.array([[[0.001, 0.002], [0.003, 0.004]], [[0.005, 0.006], [0.007, 0.008]]], dtype=np.float64)

    terms = [term0, term1, term2, term3, term4, term5]

    result = list(score_terms(X, feature_names_out, feature_types_out, terms, [preprocessor, pair_preprocessor, higher_preprocessor]))

    assert(result[0][1][0] == 0.2)
    assert(result[0][1][1] == 0.3)
    assert(result[0][1][2] == 0.2)
    assert(result[0][1][3] == 0.1)

    assert(result[1][1][0] == 0.02)
    assert(result[1][1][1] == 0.03)
    assert(result[1][1][2] == 0.03)
    assert(result[1][1][3] == 0.04)

    assert(result[2][1][0] == 0.01)
    assert(result[2][1][1] == 0.02)
    assert(result[2][1][2] == 0.03)
    assert(result[2][1][3] == 0)

    # term4 finishes before term3 since shared_cuts allows the 3rd feature to be completed first
    assert(result[4][1][0] == 0.005)
    assert(result[4][1][1] == 0)
    assert(result[4][1][2] == 0.006)
    assert(result[4][1][3] == 0.003)

    # term4 finishes before term3 since shared_cuts allows the 3rd feature to be completed first
    assert(result[3][1][0] == 0.004)
    assert(result[3][1][1] == 0)
    assert(result[3][1][2] == 0.006)
    assert(result[3][1][3] == 0)

    assert(result[5][1][0] == 0.007)
    assert(result[5][1][1] == 0)
    assert(result[5][1][2] == 0.008)
    assert(result[5][1][3] == 0)

def test_deduplicate_bins():
    TestPreprocessor = namedtuple('TestPreprocessor', 'bins_')

    preprocessor =        TestPreprocessor([{"a": 1, "b": 2}, np.array([1, 2, 3], dtype=np.float64)])
    pair_preprocessor =   TestPreprocessor([{"a": 2, "b": 1}, np.array([1, 3, 2], dtype=np.float64)])
    higher_preprocessor = TestPreprocessor([{"b": 2, "a": 1}, np.array([1, 2, 3], dtype=np.float64)])

    deduplicate_bins([preprocessor, pair_preprocessor, higher_preprocessor])

    assert(id(preprocessor.bins_[0]) != id(pair_preprocessor.bins_[0]))
    assert(id(preprocessor.bins_[0]) == id(higher_preprocessor.bins_[0]))
    assert(id(pair_preprocessor.bins_[0]) != id(higher_preprocessor.bins_[0]))

    assert(id(preprocessor.bins_[1]) != id(pair_preprocessor.bins_[1]))
    assert(id(preprocessor.bins_[1]) == id(higher_preprocessor.bins_[1]))
    assert(id(pair_preprocessor.bins_[1]) != id(higher_preprocessor.bins_[1]))
