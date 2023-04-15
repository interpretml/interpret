# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ._version import __version__  # noqa: F401

from .visual.interactive import (  # noqa: F401
    show,
    show_link,
    set_show_addr,
    get_show_addr,
    preserve,
    shutdown_show_server,
    init_show_server,
    status_show_server,
    set_visualize_provider,
    get_visualize_provider,
)

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
