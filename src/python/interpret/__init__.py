# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# Set default logging handler
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

# Set name of package
name = "interpret"

# Export functions
from .visual.interactive import show
