# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the python side of the shared state of the explanation widget."""

import ipywidgets as widgets
from traitlets import Unicode, Dict


@widgets.register
class ExplanationWidget(widgets.DOMWidget):
    """The python widget definition for the explanation."""

    _view_name = Unicode('ExplanationView').tag(sync=True)
    _model_name = Unicode('ExplanationModel').tag(sync=True)
    _view_module = Unicode('interpret-ml-widget').tag(sync=True)
    _model_module = Unicode('interpret-ml-widget').tag(sync=True)
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    _model_module_version = Unicode('^0.1.0').tag(sync=True)
    value = Dict().tag(sync=True)
    request = Dict().tag(sync=True)
    response = Dict().tag(sync=True)
