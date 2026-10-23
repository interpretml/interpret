# Copyright (c) 2026 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
import pandas as pd
import pytest
from interpret.data import ClassHistogram


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["01", "1", "01", "1"], id="leading-zero-strings"),
        pytest.param(["1", "1.0", "1", "1.0"], id="decimal-strings"),
        pytest.param([2**53, 2**53 + 1, 2**53, 2**53 + 1], id="large-integers"),
        pytest.param(["cat", "dog", "cat", "dog"], id="text-control"),
        pytest.param([0, 1, 0, 1], id="integer-control"),
        pytest.param([False, True, False, True], id="boolean-control"),
    ],
)
@pytest.mark.parametrize("container", [np.array, list, pd.Series])
def test_class_histogram_preserves_labels(labels, container):
    X = np.arange(len(labels), dtype=float).reshape(-1, 1)
    y = container(labels)
    explanation = ClassHistogram().explain_data(X, y)

    assert explanation.data()["y"].tolist() == labels
    assert explanation.data(0)["y"].tolist() == labels
    assert np.unique(explanation.data()["y"]).size == len(set(labels))


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(["01", "1", "01", "1"], id="leading-zero-strings"),
        pytest.param([2**53, 2**53 + 1, 2**53, 2**53 + 1], id="large-integers"),
        pytest.param(["cat", "dog", "cat", "dog"], id="text-control"),
        pytest.param([0.0, 1.0, 0.0, 1.0], id="float-control"),
    ],
)
def test_class_histogram_keeps_class_counts(labels):
    X = np.array([["a"], ["b"], ["a"], ["b"]])
    explanation = ClassHistogram(feature_types=["nominal"]).explain_data(X, labels)

    overall = explanation.visualize()
    assert len(overall.data) == 2
    assert [trace.y[0] for trace in overall.data] == [2, 2]

    feature = explanation.visualize(0)
    assert len(feature.data) == 2
    assert [trace.name for trace in feature.data] == sorted({str(y) for y in labels})
    assert all(sum(trace.y) == 2 for trace in feature.data)


@pytest.mark.parametrize(
    "factory",
    [np.array, list, pd.Series, lambda y: np.array(y, dtype=object)],
)
def test_exact_unsigned_values_remain_positive(factory):
    labels = [2**63, 2**63 + 4096, 2**63, 2**63, 2**63 + 4096]
    X = np.array([["a"], ["b"], ["c"], ["b"], ["a"]])
    exp = ClassHistogram(feature_types=["nominal"]).explain_data(X, factory(labels))
    for key in (None, 0):
        assert [int(v) for v in exp.data(key)["y"]] == labels
    fig = exp.visualize()
    assert [int(t.x[0]) for t in fig.data] == [2**63, 2**63 + 4096]
    assert [int(t.y[0]) for t in fig.data] == [3, 2]


@pytest.mark.parametrize(
    "factory",
    [np.array, list, pd.Series, lambda y: np.array(y, dtype=object)],
)
def test_equal_float_zeros_remain_one_class(factory):
    labels = [0.0, -0.0, 0.0, -0.0, 0.0]
    X = np.array([["a"], ["b"], ["c"], ["b"], ["a"]])
    exp = ClassHistogram(feature_types=["nominal"]).explain_data(X, factory(labels))
    assert np.unique(exp.data()["y"]).size == 1
    assert len(exp.visualize().data) == 1
    assert exp.visualize().data[0].y == (5,)
    assert len(exp.visualize(0).data) == 1


def test_numeric_float_data_remains_numeric():
    labels = np.array([2.5, 10.5, 2.5, 2.5, 10.5])
    X = np.array([["a"], ["b"], ["c"], ["b"], ["a"]])
    exp = ClassHistogram(feature_types=["nominal"]).explain_data(X, labels)
    assert exp.data()["y"].dtype.kind == "f"
    np.testing.assert_array_equal(exp.data()["y"], labels)


def test_numeric_float_plot_order_remains_numeric():
    X = np.array([["a"], ["b"], ["c"], ["b"], ["a"]])
    exp = ClassHistogram(feature_types=["nominal"]).explain_data(
        X, np.array([2.5, 10.5, 2.5, 2.5, 10.5])
    )
    assert [float(t.x[0]) for t in exp.visualize().data] == [2.5, 10.5]
    assert [t.name for t in exp.visualize(0).data] == ["2.5", "10.5"]


@pytest.mark.parametrize(
    "container",
    [
        np.array,
        list,
        pd.Series,
        lambda y: np.array(y, dtype=object),
        lambda y: pd.Series(y, dtype="UInt64"),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [
        [2**63, 2**63 + 4096, 2**63, 2**63, 2**63 + 4096],
        [2**64 - 2, 2**64 - 1, 2**64 - 2, 2**64 - 2, 2**64 - 1],
    ],
    ids=["exact-unsigned-control", "uint64-maximum"],
)
def test_unsigned_class_counts_and_feature_labels(labels, container):
    X = np.array([["a", 0], ["b", 1], ["c", 2], ["b", 3], ["a", 4]], dtype=object)
    y = container(labels)
    before = repr(y)
    exp = ClassHistogram(feature_types=["nominal", "continuous"]).explain_data(X, y)
    for key in (None, 0, 1):
        assert [int(v) for v in exp.data(key)["y"]] == labels
    assert [int(t.x[0]) for t in exp.visualize().data] == sorted(set(labels))
    assert [int(t.y[0]) for t in exp.visualize().data] == [3, 2]
    for key in (0, 1):
        fig = exp.visualize(key)
        assert [float(t.name) for t in fig.data] == [
            float(v) for v in sorted(set(labels))
        ]
        assert [sum(t.y) for t in fig.data] == [3, 2]
        assert isinstance(fig.to_json(), str)
    assert repr(y) == before


@pytest.mark.parametrize(
    ("labels", "classes", "counts"),
    [
        ([0.0, -0.0, 0.0, -0.0, 0.0], [0.0], [5]),
        ([2.5, 10.5, 2.5, 2.5, 10.5], [2.5, 10.5], [3, 2]),
        ([1, 1.0, 2, 2.0, 1], [1.0, 2.0], [3, 2]),
    ],
    ids=["signed-zero", "numeric-order", "mixed-integer-float"],
)
@pytest.mark.parametrize(
    "container", [np.array, list, pd.Series, lambda y: np.array(y, dtype=object)]
)
def test_numeric_float_feature_semantics(labels, classes, counts, container):
    X = np.array([["a", 0], ["b", 1], ["c", 2], ["b", 3], ["a", 4]], dtype=object)
    exp = ClassHistogram(feature_types=["nominal", "continuous"]).explain_data(
        X, container(labels)
    )
    for key in (None, 0, 1):
        assert exp.data(key)["y"].dtype.kind == "f"
        np.testing.assert_array_equal(exp.data(key)["y"], labels)
        fig = exp.visualize(key)
        assert [sum(t.y) for t in fig.data] == counts
        if key is None:
            assert [t.x[0] for t in fig.data] == classes
        else:
            assert [t.name for t in fig.data] == [str(v) for v in classes]
        assert isinstance(fig.to_json(), str)
