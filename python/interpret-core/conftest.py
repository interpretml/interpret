# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license
#
# Variant of below link:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
#

import pytest

collect_ignore_glob = ["js/*"]


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runselenium", action="store_true", default=False, help="run selenium tests"
    )


def pytest_collection_modifyitems(config, items):
    run_slow = False
    run_selenium = False
    if config.getoption("--runslow"):
        run_slow = True
    if config.getoption("--runselenium"):
        run_selenium = True

    skip_selenium = pytest.mark.skip(reason="need --runselenium option to run")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "selenium" in item.keywords and not run_selenium:
            item.add_marker(skip_selenium)
