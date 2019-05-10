# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
from .. import gen_feat_val_list, gen_name_from_class
from .. import reverse_map


@pytest.fixture
def fixture_feat_val_list():
    return [("race", 3), ("age", -2), ("gender", 1)]


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
