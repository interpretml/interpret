# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
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

    return True  # pragma: no cover


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
    from .dashboard import AppRunner

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
        explanation: As provided in 'show'.
        share_tables: As provided in 'show'.

    Returns:
        None.
    """

    try:
        # Initialize server if needed
        if this.app_runner is None:  # pragma: no cover
            init_show_server(this.app_addr)

        # Register
        this.app_runner.register(explanation, share_tables=share_tables)

        # Display
        open_link = isinstance(explanation, list)
        this.app_runner.display(explanation, open_link=open_link)
    except Exception as e:  # pragma: no cover
        log.error(e, exc_info=True)
        raise e

    return None


def show_link(explanation, share_tables=None):
    """ Provides the backing URL link behind the associated 'show' call for explanation.

    Args:
        explanation: Either a scalar Explanation or list of Explanations
                     that would be provided to 'show'.
        share_tables: Boolean or dictionary that dictates if Explanations
                      should all use the same selector as provided to 'show'.
                      (table used for selecting in the Dashboard).

    Returns:
        URL as a string.
    """
    # Initialize server if needed
    if this.app_runner is None:  # pragma: no cover
        init_show_server(this.app_addr)

    # Register
    this.app_runner.register(explanation, share_tables=share_tables)

    try:
        url = this.app_runner.display_link(explanation)
        return url
    except Exception as e:  # pragma: no cover
        log.error(e, exc_info=True)
        raise e


def preserve(explanation, selector_key=None, file_name=None, **kwargs):
    """ Preserves an explanation's visualization for Jupyter cell, or file.

    If file_name is not None the following occurs:
    - For Plotly figures, saves to HTML using `plot`.
    - For dataframes, saves to HTML using `to_html`.
    - For strings (html), saves to HTML.
    - For Dash components, fails with exception. This is currently not supported.

    Args:
        explanation: An explanation.
        selector_key: If integer, treat as index for explanation. Otherwise, looks up value in first column, gets index.
        file_name: If assigned, will save the visualization to this filename.
        **kwargs: Kwargs which are passed to the underlying render/export call.

    Returns:
        None.
    """

    try:
        # Get explanation key
        if selector_key is None:
            key = None
        elif isinstance(selector_key, int):
            key = selector_key
        else:
            series = explanation.selector[explanation.selector.columns[0]]
            key = series[series == selector_key].index[0]

        # Get visual object
        visual = explanation.visualize(key=key)

        # Output to front-end/file
        _preserve_output(
            explanation.name,
            visual,
            selector_key=selector_key,
            file_name=file_name,
            **kwargs
        )
        return None
    except Exception as e:  # pragma: no cover
        log.error(e, exc_info=True)
        raise e


def _preserve_output(
    explanation_name, visual, selector_key=None, file_name=None, **kwargs
):
    from plotly.offline import iplot, plot, init_notebook_mode
    from IPython.display import display, display_html
    from base64 import b64encode

    from plotly import graph_objs as go
    from pandas.core.generic import NDFrame
    import dash.development.base_component as dash_base

    init_notebook_mode(connected=True)

    def render_html(html_string):
        base64_html = b64encode(html_string.encode("utf-8")).decode("ascii")
        final_html = """<iframe src="data:text/html;base64,{data}" width="100%" height=400 frameBorder="0"></iframe>""".format(
            data=base64_html
        )
        display_html(final_html, raw=True)

    if visual is None:  # pragma: no cover
        msg = "No visualization for explanation [{0}] with selector_key [{1}]".format(
            explanation_name, selector_key
        )
        log.error(msg)
        if file_name is None:
            render_html(msg)
        else:
            pass
        return False

    if isinstance(visual, go.Figure):
        if file_name is None:
            iplot(visual, **kwargs)
        else:
            plot(visual, filename=file_name, **kwargs)
    elif isinstance(visual, NDFrame):
        if file_name is None:
            display(visual, **kwargs)
        else:
            visual.to_html(file_name, **kwargs)
    elif isinstance(visual, str):
        if file_name is None:
            render_html(visual)
        else:
            with open(file_name, "w") as f:
                f.write(visual)
    elif isinstance(visual, dash_base.Component):  # pragma: no cover
        msg = "Preserving dash components is currently not supported."
        if file_name is None:
            render_html(msg)
        log.error(msg)
        return False
    else:  # pragma: no cover
        msg = "Visualization cannot be preserved for type: {0}.".format(type(visual))
        if file_name is None:
            render_html(msg)
        log.error(msg)
        return False

    return True
