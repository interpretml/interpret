# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import pytest
from interpret.core.base import BaseExplanation


def test_that_explanation_throws_exceptions_for_incomplete():
    class IncompleteExplanation(BaseExplanation):
        pass

    with pytest.raises(Exception):
        _ = IncompleteExplanation()


def test_that_explanation_works_for_complete():
    class CompleteExplanation(BaseExplanation):
        _internal_object = {"overall": None, "specific": [None]}
        explanation_type = "performance"
        selector = None
        name = ""

        def visualize(self, key=None):
            data_dict = self.data(key)
            # NOTE: Return a fig|df|text|dash-component
            return str(data_dict)

        def data(self, key=None):
            if key is None:
                return self._internal_object["overall"]
            return self._internal_object["specific"][key]

    try:
        explanation = CompleteExplanation()
        assert explanation.data() is None
        assert explanation.data(0) is None
        assert explanation.visualize(0) == str(None)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
