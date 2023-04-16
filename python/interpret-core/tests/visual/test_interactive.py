# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from interpret import set_visualize_provider, get_visualize_provider
from interpret.provider import PreserveProvider


def test_provider_properties():
    provider = PreserveProvider()
    old_provider = get_visualize_provider()

    set_visualize_provider(provider)
    assert get_visualize_provider() == provider

    set_visualize_provider(old_provider)
    assert get_visualize_provider() == old_provider
