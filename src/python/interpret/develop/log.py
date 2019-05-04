# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging

def register_log(filename, log_level='INFO'):
    handler = logging.handlers.WatchedFileHandler(filename)
    formatter = logging.Formatter(
        "%(asctime)s | %(filename)-20s %(lineno)-4s %(funcName)20s() | %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger('interpret')
    root.setLevel(log_level)
    root.addHandler(handler)
    return None

