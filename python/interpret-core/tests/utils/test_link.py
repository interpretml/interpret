# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pytest
from interpret.utils import inv_link, link_func


def test_link_func_monoclassification_22():
    predictions = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        link_func(predictions, "monoclassification")


def test_link_func_monoclassification_12():
    predictions = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError):
        link_func(predictions, "monoclassification")


def test_link_func_monoclassification_21_fail():
    predictions = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError):
        link_func(predictions, "monoclassification")


def test_link_func_monoclassification_21():
    predictions = np.array([[1.0], [1.0]])
    expected = np.array([-np.inf, -np.inf])
    result = link_func(predictions, "monoclassification")
    assert np.array_equal(result, expected)


def test_link_func_monoclassification_11_fail():
    predictions = np.array([[2.0]])
    with pytest.raises(ValueError):
        link_func(predictions, "monoclassification")


def test_link_func_monoclassification_11():
    predictions = np.array([[1.0]])
    expected = np.array([-np.inf])
    result = link_func(predictions, "monoclassification")
    assert np.array_equal(result, expected)


def test_link_func_monoclassification_10():
    predictions = np.array([[]])
    expected = np.array([-np.inf])
    result = link_func(predictions, "monoclassification")
    assert np.array_equal(result, expected)


def test_link_func_monoclassification_1():
    predictions = np.array([1.0])
    expected = -np.inf
    result = link_func(predictions, "monoclassification")
    assert result == expected


def test_link_func_monoclassification_0():
    predictions = np.array([])
    expected = -np.inf
    result = link_func(predictions, "monoclassification")
    assert result == expected


def test_link_func_monoclassification_empty():
    # this breaks our rule of needing the last dimesion, but the intent is unambiguous
    predictions = np.array(1.0)
    expected = -np.inf
    result = link_func(predictions, "monoclassification")
    assert result == expected


def test_link_func_monoclassification_value():
    # this breaks our rule of needing the last dimesion, but the intent is unambiguous
    predictions = 1.0
    expected = -np.inf
    result = link_func(predictions, "monoclassification")
    assert result == expected


def test_link_func_logit_31():
    predictions = np.array([[[0.75], [1.0], [0.0], [np.nan]]])
    expected = np.array([[1.0986123, np.inf, -np.inf, np.nan]])
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_32():
    predictions = np.array(
        [
            [
                [0.25, 0.75],
                [0.5, 1.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [np.nan, 0.0],
                [0.0, np.nan],
            ]
        ]
    )
    expected = np.array([[1.0986123, 1.0986123, np.inf, -np.inf, np.nan, np.nan]])
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_33():
    predictions = np.array([[[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]]])
    with pytest.raises(ValueError):
        link_func(predictions, "logit")


def test_link_func_logit_31():
    predictions = np.array([[[0.25, 0.5, 0.25]]])
    with pytest.raises(ValueError):
        link_func(predictions, "logit")


def test_link_func_logit_1():
    predictions = [0.75]
    expected = 1.0986123
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_12():
    predictions = np.array([0.25, 0.75])
    expected = 1.0986123
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_0():
    # this breaks our rule of needing the last dimesion, but the intent is unambiguous
    predictions = 0.75
    expected = 1.0986123
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_2():
    predictions = [[0.75]]
    expected = [1.0986123]
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_22():
    predictions = np.array([[0.25, 0.75]])
    expected = [1.0986123]
    result = link_func(predictions, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_logit_22():
    predictions = np.array([[0.25, 0.5, 0.25]])
    with pytest.raises(ValueError):
        link_func(predictions, "logit")


def test_link_func_vlogit():
    predictions = np.array([[[0.75, 1.0], [0.0, np.nan]]])
    expected = np.array([[[1.0986123, np.inf], [-np.inf, np.nan]]])
    result = link_func(predictions, "vlogit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_mlogit():
    predictions = np.array(
        [
            [0.25, 0.625, 0.125],
            [0.5, 0.25, 1.25],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, np.nan, 0.0],
        ]
    )
    expected = np.array(
        [
            [-0.9162907, 0.0, -1.6094379],
            [-0.9162907, -1.6094379, 0.0],
            [-np.inf, -np.inf, 0],
            [0, -np.inf, -np.inf],
            [np.nan, np.nan, np.nan],
        ]
    )
    result = link_func(predictions, "mlogit")
    np.testing.assert_almost_equal(result, expected)


def test_link_func_identity():
    predictions = np.array([[1.0, 2.25], [4.0, -6.0]])
    expected = predictions.copy()
    result = link_func(predictions, "identity")
    assert np.all(result == expected)


def test_link_func_log():
    predictions = np.array(
        [np.inf, 1 / np.exp(1), 1.0, np.exp(1), np.exp(2), 0.0, np.nan]
    )
    expected = np.array([np.inf, -1.0, 0.0, 1.0, 2.0, -np.inf, np.nan])
    result = link_func(predictions, "log")
    np.testing.assert_almost_equal(result, expected)


def test_inv_link_monoclassification_fail():
    scores = np.array([[-np.inf, -np.inf], [8.0, -np.inf]])
    with pytest.raises(ValueError):
        inv_link(scores, "monoclassification")


def test_inv_link_monoclassification1():
    scores = np.array([-np.inf])
    expected = np.array([[1.0]])
    result = inv_link(scores, "monoclassification")
    assert np.array_equal(result, expected)


def test_inv_link_monoclassification0():
    scores = np.array([])
    expected = np.empty((0, 1), np.float64)
    result = inv_link(scores, "monoclassification")
    assert np.array_equal(result, expected)


def test_inv_link_monoclassification_nan():
    scores = np.array([np.nan])
    expected = np.array([[np.nan]])
    result = inv_link(scores, "monoclassification")
    np.testing.assert_almost_equal(result, expected)


def test_inv_link_logit():
    scores = np.array([[np.inf, -np.inf, 999.0, -999.0, 0.0, 1.0986123, np.nan]])
    expected = np.array(
        [
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.25, 0.75],
                [np.nan, np.nan],
            ]
        ]
    )
    result = inv_link(scores, "logit")
    np.testing.assert_almost_equal(result, expected)


def test_inv_link_vlogit():
    scores = np.array([[np.inf, -np.inf, 999.0, -999.0, 0.0, 1.0986123, np.nan]])
    expected = np.array(
        [
            [
                1.0,
                0.0,
                1.0,
                0.0,
                0.5,
                0.75,
                np.nan,
            ]
        ]
    )
    result = inv_link(scores, "vlogit")
    np.testing.assert_almost_equal(result, expected)


def test_inv_link_mlogit():
    scores = np.array(
        [
            [
                [np.inf, np.nan, -np.inf],
                [-np.inf, -np.inf, -np.inf],
                [-np.inf, 0.0, 0.0],
                [999.0, -999.0, 0.0],
                [-0.9162907, 0.0, -1.6094379],
                [np.inf, -np.inf, 0.0],
                [np.inf, np.inf, 0.0],
                [np.inf, np.inf, np.inf],
                [np.inf, 0.0, 0.0],
            ]
        ]
    )
    expected = np.array(
        [
            [
                [np.nan, np.nan, np.nan],
                [1 / 3, 1 / 3, 1 / 3],
                [0.0, 0.5, 0.5],
                [1.0, 0.0, 0.0],
                [0.25, 0.625, 0.125],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1 / 3, 1 / 3, 1 / 3],
                [1.0, 0.0, 0.0],
            ]
        ]
    )
    result = inv_link(scores, "mlogit")
    np.testing.assert_almost_equal(result, expected)


def test_inv_link_identity():
    scores = np.array([[1.0, 2.25], [4.0, -6.0]])
    expected = scores.copy()
    result = inv_link(scores, "identity")
    assert np.all(result == expected)


def test_inv_link_log():
    scores = np.array([-np.inf, -999.0, -1.0, 0.0, 1.0, 2.0, 999.0, np.inf, np.nan])
    expected = np.array(
        [0.0, 0.0, 1 / np.exp(1), 1.0, np.exp(1), np.exp(2), np.inf, np.inf, np.nan]
    )
    result = inv_link(scores, "log")
    np.testing.assert_almost_equal(result, expected)
