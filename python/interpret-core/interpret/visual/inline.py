# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# NOTE: This module is highly experimental. Expect changes every version.

import pkgutil
import os
from IPython.display import display, HTML
import uuid

from plotly.io import to_json
from plotly import graph_objs as go
import sys
import json

this = sys.modules[__name__]
this.jupyter_initialized = False


def _build_viz_figure(visualization):
    if visualization is None:
        _type = "none"
        figure = "null"
    elif isinstance(visualization, go.Figure):
        _type = "plotly"
        figure = json.loads(to_json(visualization))
    elif isinstance(visualization, str):
        _type = "html"
        figure = visualization
    else:
        raise RuntimeError(
            "Viz figure of type {} not supported".format(type(visualization))
        )

    return {"type": _type, "figure": figure}


def _build_viz_obj(explanation):
    overall = _build_viz_figure(explanation.visualize())
    if explanation.selector is None:
        specific = None
    else:
        specific = [
            _build_viz_figure(explanation.visualize(i))
            for i in range(len(explanation.selector))
        ]

    viz_obj = {
        "name": explanation.name,
        "overall": overall,
        "specific": specific,
        "selector": {
            "columns": list(explanation.selector.columns),
            "data": explanation.selector.to_dict("records"),
        },
    }
    return viz_obj


def _build_javascript(viz_obj, id_str=None, default_key=-1):
    script_path = os.path.dirname(os.path.abspath(__file__))
    js_path = os.path.join(script_path, "..", "lib", "interpret-inline.js")
    with open(js_path, 'r', encoding='utf-8') as f:
        show_js = f.read()

    init_js = """
    <script type="text/javascript">
    console.log("Initializing interpret-inline");
    {0}
    </script>
    """.format(
        show_js
    )

    if id_str is None:
        div_id = "_interpret-viz-{0}".format(uuid.uuid4())
    else:
        div_id = id_str

    body_js = """
    <div id="{0}"></div>
    <script type="text/javascript">

    (function universalLoad(root, callback) {{
      if(typeof exports === 'object' && typeof module === 'object') {{
        // CommonJS2
        console.log("CommonJS2");
        var interpretInline = require('interpret-inline');
        callback(interpretInline);
      }} else if(typeof define === 'function' && define.amd) {{
        // AMD
        console.log("AMD");
        require(['interpret-inline'], function(interpretInline) {{
          callback(interpretInline);
        }});
      }} else if(typeof exports === 'object') {{
        // CommonJS
        console.log("CommonJS");
        var interpretInline = require('interpret-inline');
        callback(interpretInline);
      }} else {{
        // Browser
        console.log("Browser");
        callback(root['interpret-inline']);
      }}
    }})(this, function(interpretInline) {{
        console.log(interpretInline);
        interpretInline.RenderApp("{0}", {1}, {2});
    }});

    </script>
    """.format(
        div_id, json.dumps(viz_obj), default_key
    )

    return init_js, body_js


def render(explanation, id_str=None, default_key=-1):
    _render_jupyter(explanation, id_str=id_str, default_key=default_key)


def _render_jupyter(explanation, id_str=None, default_key=-1):
    viz_obj = _build_viz_obj(explanation)
    init_js, body_js = _build_javascript(viz_obj, id_str, default_key=default_key)

    final_js = body_js
    if not this.jupyter_initialized:
        final_js = init_js + body_js
        this.jupyter_initialized = True

    display(HTML(final_js))
