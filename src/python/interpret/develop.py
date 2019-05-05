# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import logging
log = logging.getLogger(__name__)

def register_log(filename, log_level='INFO'):
    import logging.handlers

    handler = logging.handlers.WatchedFileHandler(filename)
    formatter = logging.Formatter(
        "%(asctime)s | %(filename)-20s %(lineno)-4s %(funcName)20s() | %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger('interpret')
    root.setLevel(log_level)
    root.addHandler(handler)
    return None


if __name__ == '__main__':
    import pytest
    import os

    register_log('test-log.txt')

    script_path = os.path.dirname(os.path.abspath(__file__))
    package_path = os.path.abspath(os.path.join(script_path, '..'))
    pytest.main([package_path])
