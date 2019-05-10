# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging

log = logging.getLogger(__name__)


def register_log(filename, level="DEBUG"):
    import logging.handlers

    handler = logging.handlers.WatchedFileHandler(filename)
    formatter = logging.Formatter(
        "%(asctime)s | %(filename)-20s %(lineno)-4s %(funcName)25s() | %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("interpret")
    root.setLevel(level)
    root.addHandler(handler)
    return None


if __name__ == "__main__":
    import pytest
    import os

    register_log("test-log.txt")

    script_path = os.path.dirname(os.path.abspath(__file__))
    pytest.main(["--rootdir={0}".format(script_path), script_path])
