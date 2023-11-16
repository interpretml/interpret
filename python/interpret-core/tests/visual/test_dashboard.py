# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret.visual.dashboard import AppRunner
from time import sleep

import pytest


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
