# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
module_logger = logging.getLogger(__name__)

PROVIDER_EXTENSION_KEY = "interpret_ext_provider"
BLACKBOX_EXTENSION_KEY = "interpret_ext_blackbox"


# TODO: More checks for blackbox validation, specifically on spec for explainer/explanation when instantiated.
def _is_valid_blackbox_explainer(proposed_blackbox_explainer):
    try:
        explainer_type = proposed_blackbox_explainer.explainer_type
        available_explanations = proposed_blackbox_explainer.available_explanations

        if explainer_type != "blackbox":
            module_logger.warning("Proposed explainer is not a blackbox.")
            return False

        for available_explanation in available_explanations:
            has_explain_method = hasattr(
                proposed_blackbox_explainer, "explain_" + available_explanation
            )
            if not has_explain_method:
                module_logger.warning(
                    "Proposed explainer has available explanation {} but has no respective method.".format(
                        available_explanation
                    )
                )
                return False

        return True

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False


def _is_valid_provider(proposed_provider):
    try:
        has_render_method = hasattr(proposed_provider, "render")
        has_parallel_method = hasattr(proposed_provider, "parallel")

        if has_parallel_method or has_render_method:
            return True
        return False

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False
