# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.blackbox.sensitivity import _soft_min_max


def test_soft_min_max():
    big_values = [-10, 10]
    small_values = [-0.01, 0.01]

    expected_big = [-10, 10]
    actual_big = _soft_min_max(big_values)
    assert actual_big == expected_big

    expected_small = [-0.01, 1.01]
    actual_small = _soft_min_max(small_values)
    assert actual_small == expected_small
