# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


def debug_info():
    """ This function varies version-by-version, designed to help the authors of this package when there's an issue.

    Returns:
        A dictionary to be printed in helping resolve issues.
    """
    from . import __version__

    debug_dict = {}
    debug_dict["interpret.__version__"] = __version__
    debug_dict.update(static_system_info())

    return debug_dict


# TODO: Fill this out once specs are provided.
def dynamic_system_info():
    """ PLACEHOLDER: Provides dynamic system information (available memory etc.) as a dictionary.

    Returns:
        A dictionary containing dynamic system information.
    """

    # Currently we don't do anything yet.
    return {}


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
        "psutil.cpu_count": psutil.cpu_count(),
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
        None.
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
    return None


if __name__ == "__main__":
    import pytest
    import os

    register_log("test-log.txt")

    script_path = os.path.dirname(os.path.abspath(__file__))
    pytest.main(["--rootdir={0}".format(script_path), script_path])
