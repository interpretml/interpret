# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import sys
import logging
from ..provider import AutoVisualizeProvider, PreserveProvider, DashProvider

_log = logging.getLogger(__name__)

_current_module = sys.modules[__name__]

_current_module._preserve_provider = None
_current_module.visualize_provider = None


def get_visualize_provider():
    """Gets visualization provider for show() related calls.

    Returns:
        Visualization provider.
    """
    return _current_module.visualize_provider


def set_visualize_provider(provider):
    """Sets visualization provider for show() related calls.

    Args:
        provider: Visualization provider found in "interpret.provider.visualize".
    """
    has_render_method = hasattr(provider, "render")
    if provider is None or has_render_method:
        _current_module.visualize_provider = provider
    else:  # pragma: no cover
        raise ValueError(
            "Object of type {} is not a visualize provider.".format(type(provider))
        )


def set_show_addr(addr):
    """Set a (ip, port) for inline visualizations and dashboard. Has side effects stated below.
    Side effect: restarts the app runner for 'show' method.

    Args:
        addr: (ip, port) tuple as address to assign show method to.
    """
    addr = (addr[0], int(addr[1]))
    init_show_server(addr)


def get_show_addr():
    """Returns (ip, port) used for show method.

    Returns:
        Address tuple (ip, port).
    """
    if isinstance(_current_module.visualize_provider, DashProvider):
        addr = (
            _current_module.visualize_provider.app_runner.ip,
            _current_module.visualize_provider.app_runner.port,
        )
        return addr
    else:
        return None


def shutdown_show_server():
    """This is a hard shutdown method for the show method's backing server.

    Returns:
        True if show server has stopped.
    """
    if isinstance(_current_module.visualize_provider, DashProvider):
        return _current_module.visualize_provider.app_runner.stop()

    return True  # pragma: no cover


def status_show_server():
    """Returns status and associated information of show method's backing server.

    Returns:
        Status and associated information as a dictionary.
    """
    status_dict = {}

    if isinstance(_current_module.visualize_provider, DashProvider):
        status_dict["app_runner_exists"] = True
        status_dict.update(_current_module.visualize_provider.app_runner.status())
    else:
        status_dict["app_runner_exists"] = False

    return status_dict


def init_show_server(addr=None, base_url=None, use_relative_links=False):
    """Initializes show method's backing server.

    Args:
        addr: (ip, port) tuple as address to assign show method to.
        base_url: Base url path as string. Used mostly when server is running behind a proxy.
        use_relative_links: Use relative links for what's returned to client. Otherwise have absolute links.
    """

    # If the user uses old methods such as init_show_server, we do an immediate override to the visualization provider.
    if isinstance(_current_module.visualize_provider, DashProvider):
        _log.info("Stopping previous dash provider")
        shutdown_show_server()

    _log.info(
        "Replacing visualize provider: {} with {}".format(
            type(_current_module.visualize_provider), type(DashProvider)
        )
    )
    set_visualize_provider(
        DashProvider.from_address(
            addr=addr, base_url=base_url, use_relative_links=use_relative_links
        )
    )
    _current_module.visualize_provider.idempotent_start()

    addr = (
        _current_module.visualize_provider.app_runner.ip,
        _current_module.visualize_provider.app_runner.port,
    )
    _log.info("Running dash provider at {}".format(addr))


def _get_integer_key(key, explanation):
    if key is not None and not isinstance(key, int):
        series = explanation.selector[explanation.selector.columns[0]]
        if key not in series.values:  # pragma: no cover
            raise ValueError("Key {} not in explanation's selector".format(key))
        key = series[series == key].index[0]

    return key


def show(explanation, key=-1, **kwargs):
    """Provides an interactive visualization for a given explanation(s).

    By default, visualization provided is not preserved when the notebook exits.

    Args:
        explanation: Either a scalar Explanation or list of Explanations to render as visualization.
        key: Specific index of explanation to visualize.
        **kwargs: Kwargs passed down to provider's render() call.
    """

    try:
        # Get explanation key
        key = _get_integer_key(key, explanation)

        # Set default render if needed
        if _current_module.visualize_provider is None:
            _current_module.visualize_provider = AutoVisualizeProvider()

        # Render
        _current_module.visualize_provider.render(explanation, key=key, **kwargs)
    except Exception as e:  # pragma: no cover
        _log.error(e, exc_info=True)
        raise e


def show_link(explanation, share_tables=None):
    """Provides the backing URL link behind the associated 'show' call for explanation.

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
    if not isinstance(
        _current_module.visualize_provider, DashProvider
    ):  # pragma: no cover
        init_show_server()

    # Register
    _current_module.visualize_provider.app_runner.register(
        explanation, share_tables=share_tables
    )

    try:
        url = _current_module.visualize_provider.app_runner.display_link(explanation)
        return url
    except Exception as e:  # pragma: no cover
        _log.error(e, exc_info=True)
        raise e


def preserve(explanation, selector_key=None, file_name=None, **kwargs):
    """Preserves an explanation's visualization for Jupyter cell, or file.

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
    """
    if _current_module._preserve_provider is None:
        _current_module._preserve_provider = PreserveProvider()

    try:
        # Get explanation key
        key = _get_integer_key(selector_key, explanation)

        _current_module._preserve_provider.render(
            explanation, key=key, file_name=file_name, **kwargs
        )
        return None
    except Exception as e:  # pragma: no cover
        _log.error(e, exc_info=True)
        raise e
