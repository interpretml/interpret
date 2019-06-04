# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


def print_debug_info(file=None):
    """ This function varies version-by-version, prints debug info as a pretty string.

    Args:
        file: File to print to (default goes to sys.stdout).

    Returns:
        None
    """
    import json
    import sys

    debug_dict = debug_info()
    if file is None:
        file = sys.stdout

    print(json.dumps(debug_dict, indent=2), file=file)
    return None


def debug_info():
    """ This function varies version-by-version, designed to help the authors of this package when there's an issue.

    Returns:
        A dictionary that contains debug info across the interpret package.
    """
    from . import __version__, status_show_server

    debug_dict = {}
    debug_dict["interpret.__version__"] = __version__
    debug_dict["interpret.status_show_server"] = status_show_server()
    debug_dict["interpret.static_system_info"] = static_system_info()
    debug_dict["interpret.dynamic_system_info"] = dynamic_system_info()

    return debug_dict


# TODO: Fill this out once specs are provided.
def dynamic_system_info():
    """ Provides dynamic system information (available memory etc.) as a dictionary.

    Returns:
        A dictionary containing dynamic system information.
    """

    import psutil
    import numpy as np

    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_freq = psutil.cpu_freq()

    system_info = {
        "psutil.virtual_memory": psutil.virtual_memory()._asdict(),
        "psutil.swap_memory": psutil.swap_memory()._asdict(),
        "psutil.avg_cpu_percent": None if cpu_percent is None else np.mean(cpu_percent),
        "psutil.std_cpu_percent": None if cpu_percent is None else np.std(cpu_percent),
        "psutil.cpu_freq": None if cpu_freq is None else cpu_freq._asdict(),
    }

    return system_info


def static_system_info():
    """ Provides static system information (machine architecture etc.) as a dictionary.

    Returns:
        A dictionary containing static system information.
    """
    import platform
    import psutil

    system_info = {
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
        "psutil.virtual_memory.total": psutil.virtual_memory().total,
        "psutil.swap_memory.total": psutil.swap_memory().total,
    }

    return system_info


def register_log(filename, level="DEBUG"):
    """ Registers file to have logs written to.

    Args:
        filename: A string that is the filepath to log to, or sys.stderr/sys.stdout.
        level: Logging level. For example, "DEBUG".

    Returns:
        Logging handler.
    """
    import logging
    import logging.handlers
    import sys
    import multiprocessing

    if filename is sys.stderr or filename is sys.stdout:
        handler = logging.StreamHandler(stream=filename)
    else:
        handler = logging.handlers.WatchedFileHandler(filename)

    formatter = logging.Formatter(
        "%(asctime)s | %(filename)-20s %(lineno)-4s %(funcName)25s() | %(message)s"
    )
    handler.setFormatter(formatter)

    queue = multiprocessing.Queue(-1)  # no size limit
    queue_handler = logging.handlers.QueueHandler(queue)
    queue_handler.setFormatter(formatter)
    queue_listener = logging.handlers.QueueListener(queue, handler)
    queue_listener.start()

    root = logging.getLogger("interpret")
    root.setLevel(level)
    root.addHandler(queue_handler)
    return queue_handler


if __name__ == "__main__":  # pragma: no cover
    import pytest
    import os

    register_log("test-log.txt")

    script_path = os.path.dirname(os.path.abspath(__file__))
    pytest.main(["--rootdir={0}".format(script_path), script_path])
