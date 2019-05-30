# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from .dashboard import AppRunner
import sys

import logging

log = logging.getLogger(__name__)

this = sys.modules[__name__]
this.app_runner = None
this.app_addr = None


def set_show_addr(addr):
    """
    Set a (ip, port) for inline visualizations and dashboard.
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


# TODO: Remove this, we don't use this anymore nor expose it.
def old_show(explanation, selector=None, index_map=None):
    from plotly.offline import iplot, init_notebook_mode

    init_notebook_mode(connected=True)
    # if not show.imported:
    #     show.imported = True

    if isinstance(selector, str):
        if index_map is None:
            print(
                "If selector is a string, a list or dictionary index_map must be passed."
            )
        if isinstance(index_map, list):
            selector = index_map.index(selector)
        elif isinstance(index_map, dict):
            selector = index_map[selector]
        else:
            print("Not supported index_feature_map type. Use list or dictionary.")
            return None
    elif isinstance(selector, int):
        selector = selector
    elif selector is None:
        selector = None
    else:
        print("Argument 'selector' must be an int, string, or None.")
        return None

    fig = explanation.visualize(selector)
    if fig is not None:
        iplot(fig)
    else:
        print("No overall graph for this explanation. Pass in a selector.")


# show.imported = False
