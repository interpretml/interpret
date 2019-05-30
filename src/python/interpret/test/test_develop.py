# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# TODO: Add test for log registration.

from ..develop import print_debug_info, debug_info


def test_debug_info():
    debug_dict = debug_info()
    assert isinstance(debug_dict, dict)
    assert isinstance(debug_dict["interpret.__version__"], str)


def test_print_debug_info():
    # Very light check, just testing if the function runs.
    print_debug_info()
    assert 1 == 1

