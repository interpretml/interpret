# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# NOTE: This module is highly experimental. Expect changes every version.

import os
import time
import uuid

from plotly.io import to_json
from plotly import graph_objs as go
import sys
import json
import base64

import logging

log = logging.getLogger(__name__)
interpret_help_link = "https://interpret.ml/docs/ebm.html"
synapse_help_link = "https://aka.ms/synapse-ebm"

this = sys.modules[__name__]
this.jupyter_initialized = False


def _build_error_frame(msg):
    error_template = r"""
    <style>
    .center {{
        position: absolute;
        left: 50%;
        top: 50%;
        -webkit-transform: translate(-50%, -50%);
        transform: translate(-50%, -50%);
    }}
    </style>
    <div class='center'><h1>{}</h1></div>
    """
    html_str = error_template.format(msg)
    return _build_base64_frame_src(html_str)


def _build_base64_frame_src(html_str):
    html_hex64 = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
    return "data:text/html;base64,{}".format(html_hex64)

def _build_cytoscape_json(cytoscape):
    json_di = {
        "elements": cytoscape.elements,
        "layout": cytoscape.layout,
        "style": cytoscape.style,
        "stylesheet": cytoscape.stylesheet,
    }
    return json.dumps(json_di)

def _build_viz_figure(visualization, detected_envs=None):
    import dash_cytoscape as cyto

    help = {}

    if visualization is None:
        _type = "none"
        figure = "null"
    elif isinstance(visualization, go.Figure):
        _type = "plotly"
        figure = json.loads(to_json(visualization))
        if (hasattr(visualization, "_interpret_help_text")):
            if ("azuresynapse" in detected_envs):
                link = synapse_help_link 
            elif (hasattr(visualization, "_interpret_help_link")):
                link = visualization._interpret_help_link
            else:
                link = interpret_help_link
            help = {"text": visualization._interpret_help_text, "link": link}
    elif isinstance(visualization, str):
        _type = "html"
        figure = _build_base64_frame_src(visualization)
    elif isinstance(visualization, cyto.Cytoscape):
        _type = "cytoscape"
        figure = _build_cytoscape_json(visualization)
    else:
        # NOTE: This error is largely specific to Dash components,
        #       all Dash component visualizations are being replaced with D3 soon.
        _type = "html"
        msg = "This visualization is not yet supported in the cloud environment."
        log.debug("Visualization type cannot render: {}".format(type(visualization)))
        figure = _build_error_frame(msg)

    return {"type": _type, "figure": figure, "help": help}


def _build_viz_err_obj(err_msg):
    _type = "html"
    figure = _build_error_frame(err_msg)
    viz_figure = {"type": _type, "figure": figure}

    viz_obj = {
        "name": "Error",
        "overall": viz_figure,
        "specific": [],
        "selector": {"columns": [], "data": []},
    }
    return viz_obj


def _build_viz_obj(explanation, detected_envs):
    overall = _build_viz_figure(explanation.visualize(), detected_envs)
    if explanation.selector is None:
        # NOTE: Unsure if this should be a list or None in the long term.
        specific = []
        selector_obj = {"columns": [], "data": []}
    else:
        specific = [
            _build_viz_figure(explanation.visualize(i), detected_envs)
            for i in range(len(explanation.selector))
        ]
        selector_obj = {
            "columns": list(explanation.selector.columns),
            "data": explanation.selector.to_dict("records"),
        }

    viz_obj = {
        "name": explanation.name,
        "overall": overall,
        "specific": specific,
        "selector": selector_obj,
    }
    return viz_obj


def _build_javascript(viz_obj, id_str=None, default_key=-1, js_url=None):
    if js_url is None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        js_path = os.path.join(script_path, "..", "lib", "interpret-inline.js")
        js_last_modified = time.ctime(os.path.getmtime(js_path))

        with open(js_path, "r", encoding="utf-8") as f:
            show_js = f.read()
        init_js = """
        <script type="text/javascript">
        console.log("Initializing interpret-inline (last modified: {1})");
        {0}
        </script>
        """.format(
            show_js, js_last_modified
        )
    else:
        init_js = """
        <script type="text/javascript" src="{0}"></script>
        """.format(
            js_url
        )

    if id_str is None:
        div_id = "_interpret-viz-{0}".format(uuid.uuid4())
    else:
        div_id = id_str

    body_js = """
    <div id="{0}"></div>
    <script defer type="text/javascript">

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
        interpretInline.RenderApp("{0}", {1}, {2});
    }});

    </script>
    """.format(
        div_id, json.dumps(viz_obj), default_key
    )

    return init_js, body_js


# NOTE: Code mostly derived from Plotly's databricks render as linked below:
# https://github.com/plotly/plotly.py/blob/01a78d3fdac14848affcd33ddc4f9ec72d475232/packages/python/plotly/plotly/io/_base_renderers.py
def _render_databricks(js):  # pragma: no cover
    import inspect

    if _render_databricks.displayHTML is None:
        found = False
        for frame in inspect.getouterframes(inspect.currentframe()):
            global_names = set(frame.frame.f_globals)
            target_names = {"displayHTML", "display", "spark"}
            if target_names.issubset(global_names):
                _render_databricks.displayHTML = frame.frame.f_globals["displayHTML"]
                found = True
                break

        if not found:
            msg = "Could not find DataBrick's displayHTML function"
            log.error(msg)
            raise RuntimeError(msg)

    _render_databricks.displayHTML(js)


_render_databricks.displayHTML = None


def render(explanation, id_str=None, default_key=-1, detected_envs=None, js_url=None):
    from IPython.display import display, HTML

    if isinstance(explanation, list):
        msg = "Dashboard not yet supported in cloud environments."
        viz_obj = _build_viz_err_obj(msg)
    else:
        viz_obj = _build_viz_obj(explanation, detected_envs)

    init_js, body_js = _build_javascript(
        viz_obj, id_str, default_key=default_key, js_url=js_url
    )

    if "databricks" in detected_envs:
        _render_databricks(init_js + body_js)
    elif "colab" in detected_envs or "azureml" in detected_envs or "azuresynapse" in detected_envs:
        display(HTML(init_js + body_js))
    else:  # Fallthrough assumes we are in an IPython environment at a minimum.
        if not this.jupyter_initialized:
            this.jupyter_initialized = True
            display(HTML(init_js + body_js))
        else:
            display(HTML(body_js))
