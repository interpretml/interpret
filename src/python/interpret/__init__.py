# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# Set default logging handler
import logging
from logging import NullHandler

# Export functions
from .visual.interactive import show, set_show_addr, get_show_addr  # noqa: F401

logging.getLogger(__name__).addHandler(NullHandler())

# Set name of package
name = "interpret"
