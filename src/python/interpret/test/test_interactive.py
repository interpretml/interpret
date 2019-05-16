# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..visual.interactive import set_show_addr, get_show_addr, shutdown_show_server


def test_shutdown():
    target_addr = ("127.0.0.1", 1337)
    set_show_addr(target_addr)

    actual_response = shutdown_show_server()
    expected_response = True
    assert actual_response == expected_response


def test_addr_assignment():
    target_addr = ("127.0.0.1", 1337)
    set_show_addr(target_addr)

    actual_addr = get_show_addr()

    assert target_addr == actual_addr
    shutdown_show_server()
