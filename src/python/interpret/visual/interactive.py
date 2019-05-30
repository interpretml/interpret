# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from .dashboard import AppRunner
import sys
from plotly import graph_objs as go
from pandas.core.generic import NDFrame
import dash.development.base_component as dash_base

import logging

log = logging.getLogger(__name__)

this = sys.modules[__name__]
this.app_runner = None
this.app_addr = None


def set_show_addr(addr):
    """ Set a (ip, port) for inline visualizations and dashboard. Has side effects stated below.
    Side effect: restarts the app runner for 'show' method.

    Args:
        addr: (ip, port) tuple as address to assign show method to.
    Returns:
        None.
    """
    addr = (addr[0], int(addr[1]))
    init_show_server(addr)


def get_show_addr():
    """ Returns (ip, port) used for show method.

    Returns:
        Address tuple (ip, port).
    """
    return this.app_addr


def shutdown_show_server():
    """ This is a hard shutdown method for the show method's backing server.

    Returns:
        True if show server has stopped.
    """
    if this.app_runner is not None:
        return this.app_runner.stop()

    return True


def status_show_server():
    """ Returns status and associated information of show method's backing server.

    Returns:
        Status and associated information as a dictionary.
    """
    status_dict = {}

    if this.app_runner is not None:
        status_dict["app_runner_exists"] = True
        status_dict.update(this.app_runner.status())
    else:
        status_dict["app_runner_exists"] = False

    return status_dict


def init_show_server(addr=None, base_url=None, use_relative_links=False):
    """ Initializes show method's backing server.

    Args:
        addr: (ip, port) tuple as address to assign show method to.
        base_url: Base url path as string. Used mostly when server is running behind a proxy.
        use_relative_links: Use relative links for what's returned to client. Otherwise have absolute links.

    Returns:
        None.
    """
    if this.app_runner is not None:
        log.debug("Stopping previous app runner at {0}".format(this.app_addr))
        shutdown_show_server()
        this.app_runner = None

    log.debug("Create app runner at {0}".format(addr))
    this.app_runner = AppRunner(
        addr, base_url=base_url, use_relative_links=use_relative_links
    )
    this.app_runner.start()
    this.app_addr = this.app_runner.ip, this.app_runner.port

    return None


# TODO: Provide example in docstrings of share_tables usage.
def show(explanation, share_tables=None):
    """ Provides an interactive visualization for a given explanation(s).

    The visualization provided is not preserved when the notebook exits.

    Args:
        explanation: Either a scalar Explanation or a list of Explanations.
        share_tables: Boolean or dictionary that dictates if Explanations
                      should all use the same selector
                      (table used for selecting in the Dashboard).

    Returns:
        None.
    """
    try:
        # Initialize server if needed
        if this.app_runner is None:
            init_show_server(this.app_addr)

        log.debug("Running existing app runner.")

        # Register
        this.app_runner.register(explanation, share_tables=share_tables)

        # Display
        open_link = isinstance(explanation, list)
        this.app_runner.display(explanation, open_link=open_link)
    except Exception as e:
        log.error(e, exc_info=True)
        raise e

    return None


def preserve(explanation, selector_key=None, file_name=None, **kwargs):
    """ Preserves an explanation's visualization for Jupyter cell, or file.

    If file_name is not None the following occurs:
    - For Plotly figures, saves to HTML using `plot`.
    - For dataframes, saves to CSV using `to_csv`.
    - For strings (html), saves to HTML.
    - For Dash components, fails with exception. This is currently not supported.

    Args:
        explanation: An explanation.
        selector_key: Key into first column of the explanation's selector. If None, returns overall visual.
        file_name: If assigned, will save the visualization to this filename.
        **kwargs: Kwargs which are passed to the underlying render/export call.

    Returns:
        None.
    """

    from plotly.offline import iplot, plot, init_notebook_mode
    from IPython.display import display, HTML
    init_notebook_mode(connected=True)

    if selector_key is None:
        key = None
    else:
        series = explanation.selector[explanation.selector.columns[0]]
        key = series[series == selector_key].index[0]

    visual = explanation.visualize(key=key)
    if isinstance(visual, go.Figure):
        if file_name is None:
            iplot(visual, **kwargs)
        else:
            plot(visual, filename=file_name, **kwargs)
    elif isinstance(visual, NDFrame):
        if file_name is None:
            display(visual, **kwargs)
        else:
            visual.to_csv(file_name, **kwargs)
    elif isinstance(visual, str):
        if file_name is None:
            with(file_name, "w") as f:
                f.write(visual)
        else:
            HTML(visual, **kwargs)
    elif isinstance(visual, dash_base.Component):
        msg = "Preserving dash components is currently not supported."
        raise Exception(msg)
    else:
        msg = "Visualization cannot be preserved for type: {0}.".format(type(visual))
        raise Exception(msg)

    return None
