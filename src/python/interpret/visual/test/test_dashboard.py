# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..dashboard import AppRunner


def test_random_port():
    app_runner = AppRunner()

    app_runner.start()
    is_alive = app_runner.ping()
    assert is_alive
    app_runner.stop()
