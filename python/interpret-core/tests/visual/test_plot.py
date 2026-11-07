# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.visual.plot import plot_line
from interpret.visual.plot import plot_density
from interpret.visual.plot import plot_pairwise_heatmap


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
    """
    Test that density plots handle large numbers correctly using the new number formatting
    """
    data_dict = {
        "scores": [1.0, 1.0],
        "names": [9e13, 1e14, 1e15],  # 1e15 value will trigger new formatting
    }

    figure = plot_density(data_dict)

    # The x-axis tick text should show ranges using our new formatting
    assert "90T - 100T" in figure.layout.xaxis.ticktext[0]
    assert "100T - 1.00e+15" in figure.layout.xaxis.ticktext[1]


def _heatmap_data_dict():
    return {
        "left_names": ["a", "b", "c"],
        "right_names": ["x", "y"],
        "scores": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "scores_range": (-2.0, 2.0),
        "global_scores_range": (-10.0, 10.0),
    }


def test_plot_pairwise_heatmap_interaction_scale():
    data_dict = _heatmap_data_dict()

    figure = plot_pairwise_heatmap(data_dict)

    assert figure.data[0].zmin == -2.0
    assert figure.data[0].zmax == 2.0


def test_plot_pairwise_heatmap_term_scale():
    data_dict = _heatmap_data_dict()

    figure = plot_pairwise_heatmap(data_dict, scale="term")

    assert figure.data[0].zmin == 1.0
    assert figure.data[0].zmax == 6.0


def test_plot_pairwise_heatmap_global_scale():
    data_dict = _heatmap_data_dict()

    figure = plot_pairwise_heatmap(data_dict, scale="global")

    assert figure.data[0].zmin == -10.0
    assert figure.data[0].zmax == 10.0


def test_plot_pairwise_heatmap_global_scale_falls_back_to_scores_range():
    data_dict = _heatmap_data_dict()
    del data_dict["global_scores_range"]

    figure = plot_pairwise_heatmap(data_dict, scale="global")

    assert figure.data[0].zmin == -2.0
    assert figure.data[0].zmax == 2.0
