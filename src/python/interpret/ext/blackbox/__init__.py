# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import sys

from interpret.ext.extension_utils import load_class_extensions

module_logger = logging.getLogger(__name__)

BLACKBOX_EXTENSION_KEY = "interpret_ext_blackbox"


def _is_valid_blackbox_explainer(proposed_blackbox_explainer):
    return True


# How to get the current module
# https://stackoverflow.com/questions/1676835
current_module = sys.modules[__name__]

load_class_extensions(current_module, BLACKBOX_EXTENSION_KEY, _is_valid_blackbox_explainer)
