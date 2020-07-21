# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
import numpy as np
import pandas as pd
from .. import gen_feat_val_list, gen_name_from_class
from .. import reverse_map, unify_data, unify_vector


@pytest.fixture
def fixture_feat_val_list():
    return [("race", 3), ("age", -2), ("gender", 1)]


def test_unify_vector_on_ndim_array():
    y = np.array([[0], [1], [2], [3]])
    expected = np.array([0, 1, 2, 3])
    new_y = unify_vector(y)
    assert np.all(new_y == expected)


def test_unify_fails_on_missing():
    orig_data = np.array([[1, 2], [3, np.nan]])
    orig_labels = np.array([0, 1])

    with pytest.raises(ValueError):
        unify_data(orig_data, orig_labels)


def test_unify_dataframe_smoke():
    df = pd.DataFrame()
    df["f1"] = [1.5, "a"]
    df["f2"] = [3, "b"]
    df["label"] = [0, 1]

    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X = df[train_cols]
    y = df[label]

    unify_data(X, y)


def test_unify_list_data():
    orig_data = [[1, 2], [3, 4]]
    orig_labels = [0, 0]

    data, labels, feature_names, feature_types = unify_data(orig_data, orig_labels)
    assert feature_names is not None
    assert feature_types is not None
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 1


def test_that_names_generated():
    class SomeClass:
        pass

    some_class = SomeClass()

    name = gen_name_from_class(some_class)
    assert name == "SomeClass_0"


def test_that_feat_val_generated(fixture_feat_val_list):
    features = ["age", "race", "gender"]
    values = [-2, 3, 1]

    feat_val_list = gen_feat_val_list(features, values)
    assert feat_val_list == fixture_feat_val_list


def test_reverse_map():
    map = {"a": 1, "b": 2, "c": 3}
    actual_rev_map = reverse_map(map)
    expected_rev_map = {1: "a", 2: "b", 3: "c"}

    assert actual_rev_map == expected_rev_map
