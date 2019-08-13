# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import sys

from interpret.ext.extension_utils import load_class_extensions

module_logger = logging.getLogger(__name__)

BLACKBOX_EXTENSION_KEY = "interpret_ext_blackbox"


# TODO: More checks for blackbox validation, specifically on spec for explainer/explanation when instantiated.
def _is_valid_blackbox_explainer(proposed_blackbox_explainer):
    explainer_type = proposed_blackbox_explainer.explainer_type
    available_explanations = proposed_blackbox_explainer.available_explanations

    if explainer_type != "blackbox":
        module_logger.warning("Proposed explainer is not a blackbox.")
        return False

    for available_explanation in available_explanations:
        has_explain_method = hasattr(proposed_blackbox_explainer, "explain_" + available_explanation)
        if not has_explain_method:
            module_logger.warning(
                "Proposed explainer has available explanation {} but has no respective method.".format(available_explanation)
            )
            return False

    return True


# How to get the current module
# https://stackoverflow.com/questions/1676835
current_module = sys.modules[__name__]

load_class_extensions(
    current_module, BLACKBOX_EXTENSION_KEY, _is_valid_blackbox_explainer
)
