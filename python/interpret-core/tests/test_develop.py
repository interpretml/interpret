# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.develop import debug_info, debug_mode, print_debug_info, register_log


def test_debug_mode():
    import logging
    import sys

    import pytest

    handler = debug_mode(log_filename=sys.stderr, log_level="INFO", native_debug=False)
    root = logging.getLogger("interpret")
    root.removeHandler(handler)

    with pytest.raises(
        Exception, match="Cannot call debug_mode more than once in the same session."
    ):
        debug_mode()


def test_debug_info():
    debug_dict = debug_info()
    assert isinstance(debug_dict, dict)
    assert isinstance(debug_dict["interpret.__version__"], str)


def test_print_debug_info():
    # Light smoke test.
    print_debug_info()
    assert 1 == 1


def test_register_log():
    # Light smoke test.
    import logging
    import os
    import sys
    import tempfile

    # Output to stream
    handler = register_log(sys.stderr, "DEBUG")
    root = logging.getLogger("interpret")
    root.removeHandler(handler)

    # Output to file
    temp_log_path = os.path.join(tempfile.mkdtemp(), "test-log.txt")
    handler = register_log(temp_log_path, "DEBUG")
    handler.flush()
    handler.close()
    root = logging.getLogger("interpret")
    root.removeHandler(handler)
    # NOTE: Logging is not releasing the file despite close. This temporary file simply will not be deleted.
    # os.remove(temp_log_path)
