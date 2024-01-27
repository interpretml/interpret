#!/usr/bin/env python
# coding: utf-8

# Copyright (c) InterpretML.
# Distributed under the terms of the Modified BSD License.

import pytest

from ..stitch import StitchWidget


def test_stitch_creation_blank():
    w = StitchWidget()
    assert w.kernelmsg == ""
    assert w.clientmsg == ""
    assert w.srcdoc == "<p>srcdoc should be defined by the user</p>"
    assert w.initial_height == "1px"
    assert w.initial_width == "1px"
    assert w.initial_border == "0"
