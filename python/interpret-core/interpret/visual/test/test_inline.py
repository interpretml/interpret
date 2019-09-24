# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# NOTE: Render call is tested with other providers elsewhere.

import plotly.graph_objects as go
from ..inline import _build_viz_figure
import pytest


def test_build_viz_figure():
    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    viz_fig = _build_viz_figure(fig)
    assert viz_fig["type"] == "plotly"
    assert viz_fig["figure"] != fig

    fig = "<h1>Some HTML</h1>"
    viz_fig = _build_viz_figure(fig)
    assert viz_fig["type"] == "html"
    assert isinstance(viz_fig["figure"], str)

    fig = None
    viz_fig = _build_viz_figure(fig)
    assert viz_fig["type"] == "none"
    assert viz_fig["figure"] == "null"

    # NOTE: Should produce HTML error message.
    fig = 1
    viz_fig = _build_viz_figure(fig)
    assert viz_fig["type"] == "html"
    assert isinstance(viz_fig["figure"], str)
