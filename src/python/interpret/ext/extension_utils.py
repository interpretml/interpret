# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
import pkg_resources
import re

module_logger = logging.getLogger(__name__)

blackbox_key = "interpret_ext_blackbox"


def _validate_class_name(proposed_class_name):
    """
    Used to validate class names before registration.

    :param proposed_class_name: The string to name the class
    :type proposed_class_name: str
    """
    # regex for class name came from
    # https://stackoverflow.com/questions/10120295
    # TODO: Support other languages
    match = re.match(r"[a-zA-Z_][a-zA-Z_0-9]+", proposed_class_name)
    if match is None or match.group(0) != proposed_class_name:
        raise ValueError("Invalid class name {}. Class names must start with a "
                         "letter or an underscore and can continue with letters, "
                         "numbers, and underscores.".format(proposed_class_name))


def load_class_extensions(current_module, extension_key, extension_class_validator):
    """
    Load all registered extensions under the `extension_key` namespace in entry_points.

    :param current_module: The module where extension classes should be added.
    :type current_module: module
    :param extension_key: The identifier for the entry_points to register within the current_module.
    :type extension_key: str
    :param extension_class_validator: A function that checks the class for correctness before it is registered.
    :type extension_class_validator: function(class) -> bool
    """
    for entrypoint in pkg_resources.iter_entry_points(extension_key):
        try:
            extension_class_name = entrypoint.name
            extension_class = entrypoint.load()
            module_logger.debug("loading entrypoint key {} with name {} with object {}".format(
                extension_key, extension_class_name, extension_class))

            _validate_class_name(extension_class_name)

            if not extension_class_validator(extension_class):
                raise ValueError("class {} failed validation.".format(extension_class))

            if getattr(current_module, extension_class_name, None) is not None:
                raise ValueError("class name {} already exists for module {}.".format(
                                 extension_class_name, current_module.__name__))

            setattr(current_module, extension_class_name, extension_class)

        except Exception as e:
            module_logger.warning("Failure while loading {}. Failed to load entrypoint {} with exception {}.".format(
                blackbox_key, entrypoint, e))
