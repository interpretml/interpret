# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import sys

from interpret.ext.extension_utils import load_class_extensions, _is_valid_explainer

module_logger = logging.getLogger(__name__)

BLACKBOX_EXTENSION_KEY = "interpret_ext_blackbox"


# TODO: More checks for blackbox validation, specifically on spec for explainer/explanation when instantiated.
def _is_valid_blackbox_explainer(proposed_blackbox_explainer):
    try:
        return _is_valid_explainer(proposed_blackbox_explainer, "blackbox")

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False


# How to get the current module
# https://stackoverflow.com/questions/1676835
current_module = sys.modules[__name__]

load_class_extensions(
    current_module, BLACKBOX_EXTENSION_KEY, _is_valid_blackbox_explainer
)
