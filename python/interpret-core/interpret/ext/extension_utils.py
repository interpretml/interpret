# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import pkg_resources
import re
from warnings import warn
import traceback


module_logger = logging.getLogger(__name__)


# TODO: More checks for explainer validation, specifically on spec for explainer/explanation when instantiated.
def _validate_class_name(proposed_class_name):
    """ Used to validate class names before registration.

    Attributes:
        proposed_class_name: The string to name the class.
    """
    # regex for class name came from
    # https://stackoverflow.com/questions/10120295
    # TODO: Support other languages
    match = re.match(r"[a-zA-Z_][a-zA-Z_0-9]+", proposed_class_name)
    if match is None or match.group(0) != proposed_class_name:
        raise ValueError(
            "Invalid class name {}. Class names must start with a "
            "letter or an underscore and can continue with letters, "
            "numbers, and underscores.".format(proposed_class_name)
        )


def load_class_extensions(current_module, extension_key, extension_class_validator):
    """ Load all registered extensions under the `extension_key` namespace in entry_points.

    Attributes:
        current_module: The module itself where extension classes should be added.
        extension_key: The identifier as string for the entry_points to register within the current_module.
        extension_class_validator: A function(class) -> bool, that checks the class for correctness before it is registered.
    """
    for entrypoint in pkg_resources.iter_entry_points(extension_key):
        module_logger.debug("processing entrypoint {}".format(extension_key))
        try:
            extension_class_name = entrypoint.name
            extension_class = entrypoint.load()
            module_logger.debug(
                "loading entrypoint key {} with name {} with object {}".format(
                    extension_key, extension_class_name, extension_class
                )
            )
            _validate_class_name(extension_class_name)

            if not extension_class_validator(extension_class):
                raise ValueError("class {} failed validation.".format(extension_class))

            if getattr(current_module, extension_class_name, None) is not None:
                raise ValueError(
                    "class name {} already exists for module {}.".format(
                        extension_class_name, current_module.__name__
                    )
                )

            setattr(current_module, extension_class_name, extension_class)

        except Exception as e:  # pragma: no cover
            msg = "Failure while loading {}. Failed to load entrypoint {} with exception {}.".format(
                extension_key,
                entrypoint,
                "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            )
            module_logger.warning(msg)

            warn(msg)
