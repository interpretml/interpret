# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging

module_logger = logging.getLogger(__name__)

PROVIDER_EXTENSION_KEY = "interpret_ext_provider"
BLACKBOX_EXTENSION_KEY = "interpret_ext_blackbox"
GREYBOX_EXTENSION_KEY = "interpret_ext_greybox"
GLASSBOX_EXTENSION_KEY = "interpret_ext_glassbox"
DATA_EXTENSION_KEY = "interpret_ext_data"
PERF_EXTENSION_KEY = "interpret_ext_perf"


def _is_valid_explainer(proposed_explainer, expected_explainer_type):
    try:
        explainer_type = proposed_explainer.explainer_type
        available_explanations = proposed_explainer.available_explanations

        if explainer_type != expected_explainer_type:
            module_logger.warning(
                "Proposed explainer is not a {}.".format(expected_explainer_type)
            )
            return False

        for available_explanation in available_explanations:
            has_explain_method = hasattr(
                proposed_explainer, "explain_" + available_explanation
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


# TODO: More checks for blackbox validation, specifically on spec for explainer/explanation when instantiated.
def _is_valid_blackbox_explainer(proposed_explainer):
    return _is_valid_explainer(proposed_explainer, "blackbox")


def _is_valid_glassbox_explainer(proposed_explainer):
    try:
        is_valid_explainer = _is_valid_explainer(proposed_explainer, "model")
        has_fit = hasattr(proposed_explainer, "fit")
        has_predict = hasattr(proposed_explainer, "predict")
        if not is_valid_explainer:
            module_logger.warning(
                "Explainer not valid due to missing explain_local or global function."
            )
        if not has_fit:
            module_logger.warning("Explainer not valid due to missing fit function.")
        if not has_predict:
            module_logger.warning(
                "Explainer not valid due to missing predict function."
            )
        return is_valid_explainer and has_fit and has_predict

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False


def _is_valid_greybox_explainer(proposed_explainer):
    return _is_valid_explainer(proposed_explainer, "specific")


def _is_valid_data_explainer(proposed_explainer):
    return _is_valid_explainer(proposed_explainer, "data")


def _is_valid_perf_explainer(proposed_explainer):
    return _is_valid_explainer(proposed_explainer, "perf")


def _is_valid_provider(proposed_provider):
    try:
        has_render_method = hasattr(proposed_provider, "render")
        has_parallel_method = hasattr(proposed_provider, "parallel")

        if has_parallel_method or has_render_method:
            return True

        module_logger.warning("Proposed provider is not valid.")
        return False

    except Exception as e:
        module_logger.warning("Validate function threw exception {}".format(e))
        return False
