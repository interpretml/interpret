# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..sensitivity import soft_min_max


def test_soft_min_max():
    big_values = [-10, 10]
    small_values = [-0.01, 0.01]

    expected_big = [-10, 10]
    actual_big = soft_min_max(big_values)
    assert actual_big == expected_big

    expected_small = [-0.01, 1.01]
    actual_small = soft_min_max(small_values)
    assert actual_small == expected_small
