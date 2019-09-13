# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import sys

from interpret.ext.extension_utils import load_class_extensions, _is_valid_explainer

module_logger = logging.getLogger(__name__)

GLASSBOX_EXTENSION_KEY = "interpret_ext_glassbox"


# TODO: More checks for glassbox validation, specifically on spec for explainer/explanation when instantiated.
def _is_valid_glassbox_explainer(proposed_glassbox_explainer):
    try:
        is_valid_explainer = _is_valid_explainer(proposed_glassbox_explainer, "glassbox")
        has_fit = hasattr(proposed_glassbox_explainer, "fit")
        has_predict = hasattr(proposed_glassbox_explainer, "predict")
        if not is_valid_explainer:
            module_logger.warning("Explainer not valid due to missing explain_local or global function.")
        if not has_fit:
            module_logger.warning("Explainer not valid due to missing fit function.")
        if not has_predict:
            module_logger.warning("Explainer not valid due to missing predict function.")
        return is_valid_explainer and has_fit and has_predict

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False


# How to get the current module
# https://stackoverflow.com/questions/1676835
current_module = sys.modules[__name__]

load_class_extensions(
    current_module, GLASSBOX_EXTENSION_KEY, _is_valid_glassbox_explainer
)
