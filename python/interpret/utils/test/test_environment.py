# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..environment import EnvironmentDetector


def test_environment_detector():
    detector = EnvironmentDetector()
    envs = detector.detect()
    assert len(envs) == 0
