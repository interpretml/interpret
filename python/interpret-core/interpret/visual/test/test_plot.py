# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..plot import plot_line


def test_plot_line_bounds_smoke():
    data_dict = {
        "names": ["a", "b", "c"],
        "scores": [1, 2, 3],
        "upper_bounds": [1.5, 2.5, 3.5],
        "lower_bounds": [0.5, 1.5, 2.5],
    }
    figure = plot_line(data_dict)
    assert figure.data[0].name == "Lower Bound"
