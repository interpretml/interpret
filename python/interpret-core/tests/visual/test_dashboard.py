# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from time import sleep

import pytest
from interpret.visual.dashboard import AppRunner


@pytest.mark.slow
def test_random_port():
    for _ in range(10):
        app_runner = AppRunner()
        app_runner.start()
        sleep(20)
        is_alive = app_runner.ping()
        app_runner.stop()
        if is_alive:
            break
    assert is_alive
