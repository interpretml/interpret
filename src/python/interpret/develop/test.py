from .log import register_log

import logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    import pytest
    import os

    register_log('test-log.txt')

    script_path = os.path.dirname(os.path.abspath(__file__))
    package_path = os.path.abspath(os.path.join(script_path, '..', '..'))
    pytest.main([package_path])
