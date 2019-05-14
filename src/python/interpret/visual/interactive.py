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
        addr: (ip, port) address to assign show method to.
    Returns:
        None.
    """
    addr = (addr[0], int(addr[1]))
    _init_app_runner(addr)


def get_show_addr():
    """ Returns (ip, port) used for show method.

    Returns:
        Address tuple (ip, port)
    """
    return this.app_addr


def _init_app_runner(addr=None):
    if this.app_runner is not None:
        log.debug("Stopping previous app runner at {0}".format(this.app_addr))
        this.app_runner.stop()
        this.app_runner = None

    log.debug("Create app runner at {0}".format(addr))
    this.app_runner = AppRunner(addr)
    this.app_runner.start()
    this.app_addr = this.app_runner.ip, this.app_runner.port


def show(explanation, share_tables=None):
    try:
        # Initialize server if needed
        if this.app_runner is None:
            _init_app_runner(this.app_addr)

        log.debug("Running existing app runner.")

        # Register
        this.app_runner.register(explanation, share_tables=share_tables)

        # Display
        open_link = isinstance(explanation, list)
        this.app_runner.display(explanation, open_link=open_link)
    except Exception as e:
        log.error(e, exc_info=True)
        raise e


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
