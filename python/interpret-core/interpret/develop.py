# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import sys
import math

from .utils._native import Native

_current_module = sys.modules[__name__]
_current_module.is_debug_mode = False

# Global options
_develop_options = {
    "n_intercept_rounds_initial": 25,
    "n_intercept_rounds_final": 100,
    "intercept_learning_rate": 0.25,
    "cat_l2": 0.0,
    "min_samples_leaf_nominal": None,
    "learning_rate_scale": 1.0,
    "max_cat_threshold": 9223372036854775807,
    "cat_include": 1.0,
    "purify_boosting": False,
    "purify_result": False,
    "randomize_initial_feature_order": True,
    "randomize_greedy_feature_order": True,  # Randomize feature order if greedy.
    "randomize_feature_order": False,  # Randomize feature order. No cost to results.
    "acceleration": Native.AccelerationFlags_ALL,
}


def get_option(name):
    return _develop_options[name]


def set_option(name, value):
    if name not in _develop_options:
        msg = f"Unrecognized development option {name}."
        raise Exception(msg)

    _develop_options[name] = value


def print_debug_info(file=None):
    """This function varies version-by-version, prints debug info as a pretty string.

    Args:
        file: File to print to (default goes to sys.stdout).
    """
    import json

    debug_dict = debug_info()
    if file is None:
        file = sys.stdout

    print(json.dumps(debug_dict, indent=2), file=file)


def debug_info():
    """This function varies version-by-version, designed to help the authors of this package when there's an issue.

    Returns:
        A dictionary that contains debug info across the interpret package.
    """
    from ._version import __version__
    from .visual._interactive import status_show_server

    debug_dict = {}
    debug_dict["interpret.__version__"] = __version__
    debug_dict["interpret.status_show_server"] = status_show_server()
    debug_dict["interpret.static_system_info"] = static_system_info()
    debug_dict["interpret.dynamic_system_info"] = dynamic_system_info()

    return debug_dict


def dynamic_system_info():
    """Provides dynamic system information (available memory etc.) as a dictionary.

    Returns:
        A dictionary containing dynamic system information.
    """

    import numpy as np
    import psutil

    try:
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        virtual_memory = psutil.virtual_memory()._asdict()
        virtual_memory = {
            k: _sizeof_fmt(v) if k != "percent" else v
            for k, v in virtual_memory.items()
        }
        swap_memory = psutil.swap_memory()._asdict()
        swap_memory = {
            k: _sizeof_fmt(v) if k != "percent" else v for k, v in swap_memory.items()
        }

        system_info = {
            "psutil.virtual_memory": virtual_memory,
            "psutil.swap_memory": swap_memory,
            "psutil.avg_cpu_percent": (
                None if cpu_percent is None else np.mean(cpu_percent)
            ),
            "psutil.std_cpu_percent": (
                None if cpu_percent is None else np.std(cpu_percent)
            ),
            "psutil.cpu_freq": None if cpu_freq is None else cpu_freq._asdict(),
        }
    except Exception:  # pragma: no cover
        system_info = None

    return system_info


def static_system_info():
    """Provides static system information (machine architecture etc.) as a dictionary.

    Returns:
        A dictionary containing static system information.
    """
    import platform

    import psutil

    return {
        "platform": platform.platform(),
        "platform.architecture": platform.architecture(),
        "platform.machine": platform.machine(),
        "platform.processor": platform.processor(),
        "platform.python_version": platform.python_version(),
        "platform.release": platform.release(),
        "platform.system": platform.system(),
        "platform.version": platform.version(),
        "psutil.logical_cpu_count": psutil.cpu_count(logical=True),
        "psutil.physical_cpu_count": psutil.cpu_count(logical=False),
        "psutil.virtual_memory.total": _sizeof_fmt(psutil.virtual_memory().total),
        "psutil.swap_memory.total": _sizeof_fmt(psutil.swap_memory().total),
    }


def _sizeof_fmt(num, suffix="B"):
    """Returns bytes in human readable form. Taken from below link:
    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size

    Args:
        num: Number (bytes) to represent in human readable form.
        suffix: Suffix required for Python format string.

    Returns:
        Human readable form of bytes provided.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return "{:.1f}{}{}".format(num, "Yi", suffix)  # pragma: no cover


def debug_mode(log_filename="log.txt", log_level="INFO", native_debug=True):
    """Sets package into debug mode.

    Args:
        log_filename: A string that is the filepath to log to, or sys.stderr/sys.stdout.
        log_level: Logging level. For example, "DEBUG".
        native_debug: Load debug versions of native libraries if True.

    Returns:
        Logging handler.
    """
    import json
    import logging

    from .utils._native import Native

    # Exit fast on second call.
    if _current_module.is_debug_mode:
        msg = "Cannot call debug_mode more than once in the same session."
        raise Exception(msg)
    _current_module.is_debug_mode = True

    # Register log
    handler = register_log(log_filename, log_level)

    # Write basic system diagnostic
    debug_dict = debug_info()
    debug_str = json.dumps(debug_dict, indent=2)
    root = logging.getLogger("interpret")
    root.info(debug_str)

    # Load native libraries in debug mode if needed
    native = Native.get_native_singleton(is_debug=native_debug)
    native.set_logging(log_level)

    return handler


def register_log(filename, level="DEBUG"):
    """Registers file to have logs written to.

    Args:
        filename: A string that is the filepath to log to, or sys.stderr/sys.stdout.
        level: Logging level. For example, "DEBUG".

    Returns:
        Logging handler.
    """
    import logging
    import logging.handlers

    if filename is sys.stderr or filename is sys.stdout:
        handler = logging.StreamHandler(stream=filename)
    else:
        handler = logging.handlers.WatchedFileHandler(filename)

    formatter = logging.Formatter(
        "%(asctime)s | %(filename)-20s %(lineno)-4s %(funcName)25s() | %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("interpret")
    root.setLevel(level)
    root.addHandler(handler)

    return handler


if __name__ == "__main__":  # pragma: no cover
    import os

    import pytest

    register_log("test-log.txt")

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tests"
    )
    pytest.main([f"--rootdir={script_path}", script_path])
