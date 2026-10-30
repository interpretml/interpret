# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    LinearRegression,
)
from interpret.provider import PreserveProvider
from interpret.visual._udash import gen_plot
from interpret.visual.plot import plot_density, plot_line, plot_waterfall


def test_plot_line_bounds_smoke():
    data_dict = {
        "names": ["a", "b", "c"],
        "scores": [1, 2, 3],
        "upper_bounds": [1.5, 2.5, 3.5],
        "lower_bounds": [0.5, 1.5, 2.5],
    }
    figure = plot_line(data_dict)
    assert figure.data[0].name == "Lower Bound"


def test_plot_density_large_numbers():
    """Test that density plots handle large numbers correctly using the new number formatting."""
    data_dict = {
        "scores": [1.0, 1.0],
        "names": [9e13, 1e14, 1e15],  # 1e15 value will trigger new formatting
    }

    figure = plot_density(data_dict)

    # The x-axis tick text should show ranges using our new formatting
    assert "90T - 100T" in figure.layout.xaxis.ticktext[0]
    assert "100T - 1.00e+15" in figure.layout.xaxis.ticktext[1]


def test_plot_waterfall_basic():
    data_dict = {
        "names": ["Feature A", "Feature B", "Feature C"],
        "scores": [2.5, -1.2, 0.8],
        "values": [10, "Yes", 3.14],
        "extra": {"names": ["Intercept"], "scores": [5.0]},
        "perf": {
            "actual": 7.1,
            "predicted": 7.1,
            "actual_score": 7.1,
            "predicted_score": 7.1,
        },
    }

    fig = plot_waterfall(data_dict)
    assert fig is not None
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "waterfall"
    assert trace.orientation == "h"
    assert trace.measure[-1] == "total"
    assert trace.measure[0] == "relative"
    assert trace.y[0] == "Intercept"
    assert trace.y[-1] == "Prediction"
    assert trace.x[0] == 5.0
    assert "Actual: 7.1" in fig.layout.title.text
    assert "Predicted: 7.1" in fig.layout.title.text


def test_plot_waterfall_vertical():
    data_dict = {
        "names": ["F1", "F2"],
        "scores": [1.0, -0.5],
        "extra": {"names": ["Intercept"], "scores": [2.0]},
    }

    fig = plot_waterfall(data_dict, orientation="v", title="Custom Title")
    assert fig is not None
    trace = fig.data[0]
    assert trace.orientation == "v"
    assert trace.x[0] == "Intercept"
    assert trace.x[-1] == "Prediction"
    assert trace.y[0] == 2.0
    assert fig.layout.title.text == "Custom Title"


def test_plot_waterfall_top_n():
    data_dict = {
        "names": [f"F_{i}" for i in range(10)],
        "scores": [float(i) for i in range(10)],
        "extra": {"names": ["Intercept"], "scores": [1.0]},
    }

    fig = plot_waterfall(data_dict, top_n=3)
    assert fig is not None
    trace = fig.data[0]
    # Intercept + top 3 + Other features + Prediction = 6 steps
    assert len(trace.y) == 6
    assert "Other features" in trace.y
    assert trace.y[-1] == "Prediction"


def test_plot_waterfall_small_remainder():
    # Test that small non-zero values (e.g. 1e-10) are not discarded by an absolute threshold
    data_dict = {
        "names": [f"F_{i}" for i in range(20)],
        "scores": [1e-10 for _ in range(20)],
        "extra": {"names": ["Intercept"], "scores": [0.0]},
    }

    fig = plot_waterfall(data_dict, top_n=15)
    assert fig is not None
    trace = fig.data[0]
    # Intercept + top 15 + Other features + Prediction = 18 steps
    assert len(trace.y) == 18
    assert "Other features" in trace.y
    other_idx = list(trace.y).index("Other features")
    assert np.isclose(trace.x[other_idx], 5e-10)


def test_plot_waterfall_multiclass():
    data_dict = {
        "names": ["F1", "F2"],
        "scores": [np.array([0.5, -0.2, 0.1]), np.array([0.3, 0.8, -0.4])],
        "extra": {"names": ["Intercept"], "scores": [np.array([1.0, 2.0, 3.0])]},
        "perf": {
            "actual": 1,
            "predicted": 1,
            "actual_score": 0.8,
            "predicted_score": 0.8,
        },
        "meta": {"label_names": ["ClassA", "ClassB", "ClassC"]},
    }

    fig = plot_waterfall(data_dict, class_idx=1)
    assert fig is not None
    trace = fig.data[0]
    assert trace.type == "waterfall"
    assert trace.x[0] == 2.0  # ClassB intercept
    assert "ClassB" in fig.layout.title.text


def test_plot_waterfall_binary_title():
    data_dict = {
        "names": ["F1", "F2"],
        "scores": [0.5, -0.2],
        "extra": {"names": ["Intercept"], "scores": [0.1]},
        "perf": {
            "actual": 0,
            "predicted": 1,
            "actual_score": 0.3,
            "predicted_score": 0.7,
        },
        "meta": {"label_names": ["No", "Yes"]},
    }
    fig = plot_waterfall(data_dict)
    assert fig is not None
    assert "Actual Class: No" in fig.layout.title.text
    assert "Predicted Class: Yes" in fig.layout.title.text
    assert "Pr(y = Yes): 0.700" in fig.layout.title.text


def test_explanation_visualize_waterfall():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 2))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 3.0

    # Test EBMExplanation
    ebm = ExplainableBoostingRegressor(max_rounds=5, outer_bags=1, n_jobs=1)
    ebm.fit(X, y)
    exp = ebm.explain_local(X[:2], y[:2])
    # visualize with default title=None should generate summary in title
    fig = exp.visualize(0, plot_type="waterfall")
    assert fig is not None
    assert fig.data[0].type == "waterfall"
    assert "Local Explanation" in fig.layout.title.text
    assert "Predicted:" in fig.layout.title.text

    # Test LinearExplanation
    lr = LinearRegression()
    lr.fit(X, y)
    exp_lr = lr.explain_local(X[:2], y[:2])
    fig_lr = exp_lr.visualize(0, plot_type="waterfall")
    assert fig_lr is not None
    assert fig_lr.data[0].type == "waterfall"


def test_ebm_waterfall_with_init_score():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 2))
    y = 2.0 * X[:, 0] + 5.0
    init_score = np.full(30, 10.0)

    ebm = ExplainableBoostingRegressor(max_rounds=5, outer_bags=1, n_jobs=1)
    ebm.fit(X, y)
    exp = ebm.explain_local(X[:2], y[:2], init_score=init_score[:2])

    fig = exp.visualize(0, plot_type="waterfall")
    assert fig is not None
    trace = fig.data[0]
    assert "Init score" in trace.y
    init_idx = list(trace.y).index("Init score")
    assert np.isclose(trace.x[init_idx], 10.0)


def test_preserve_provider_plot_type():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    y = 2.0 * X[:, 0] + 1.0

    ebm = ExplainableBoostingRegressor(max_rounds=5, outer_bags=1, n_jobs=1)
    ebm.fit(X, y)
    exp = ebm.explain_local(X[:2], y[:2])

    provider = PreserveProvider()

    # Preserve with default bar should not raise TypeError
    provider.render(exp, key=0)

    # Preserve with waterfall
    provider.render(exp, key=0, plot_type="waterfall")


def test_udash_propagation_waterfall():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    y = 2.0 * X[:, 0] + 1.0

    ebm = ExplainableBoostingRegressor(max_rounds=5, outer_bags=1, n_jobs=1)
    ebm.fit(X, y)
    exp = ebm.explain_local(X[:2], y[:2])

    div = gen_plot(exp, 0, 0, 0, plot_type="waterfall")
    assert div is not None
