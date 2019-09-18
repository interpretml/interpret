# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# Set default logging handler
import logging
from logging import NullHandler

# Export functions
from .version import __version__  # noqa: F401
from .visual.interactive import (  # noqa: F401
    show,
    show_link,
    set_show_addr,
    get_show_addr,
)
from .visual.interactive import preserve  # noqa: F401
from .visual.interactive import shutdown_show_server  # noqa: F401
from .visual.interactive import init_show_server  # noqa: F401
from .visual.interactive import status_show_server  # noqa: F401
from .visual.interactive import set_visualize_provider  # noqa: F401
from .visual.interactive import get_visualize_provider  # noqa: F401

logging.getLogger(__name__).addHandler(NullHandler())
