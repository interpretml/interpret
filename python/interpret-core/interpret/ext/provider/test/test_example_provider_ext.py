# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ....glassbox import LinearRegression
from .. import _is_valid_provider


def test_invalid_provider():
    # NOTE: No parallel nor render method.
    class InvalidProvider:
        pass

    assert not _is_valid_provider(LinearRegression)
    assert not _is_valid_provider(InvalidProvider)
