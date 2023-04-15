# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from ._version import __version__  # noqa: F401

from .visual.interactive import show  # noqa: F401
from .visual.interactive import show_link  # noqa: F401
from .visual.interactive import set_show_addr  # noqa: F401
from .visual.interactive import get_show_addr  # noqa: F401
from .visual.interactive import preserve  # noqa: F401
from .visual.interactive import shutdown_show_server  # noqa: F401
from .visual.interactive import init_show_server  # noqa: F401
from .visual.interactive import status_show_server  # noqa: F401
from .visual.interactive import set_visualize_provider  # noqa: F401
from .visual.interactive import get_visualize_provider  # noqa: F401

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
