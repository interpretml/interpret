# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

# NOTE: Version is replaced by a regex script.
__version__ = "0.3.2"

# Set default logging handler
from logging import NullHandler, getLogger

# Export functions
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

getLogger(__name__).addHandler(NullHandler())
