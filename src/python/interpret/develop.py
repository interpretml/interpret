# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


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
