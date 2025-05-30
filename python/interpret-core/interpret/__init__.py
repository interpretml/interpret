# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import logging

from ._version import __version__  # noqa: F401
from .visual._interactive import (  # noqa: F401
    get_show_addr,
    get_visualize_provider,
    init_show_server,
    preserve,
    set_show_addr,
    set_visualize_provider,
    show,
    show_link,
    shutdown_show_server,
    status_show_server,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())
