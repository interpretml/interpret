# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import pytest
from interpret.utils._explanation import gen_name_from_class


@pytest.fixture
def fixture_feat_val_list():
    return [("race", 3), ("age", -2), ("gender", 1)]


def test_that_names_generated():
    class SomeClass:
        pass

    some_class = SomeClass()

    name = gen_name_from_class(some_class)
    assert name == "SomeClass_0"
