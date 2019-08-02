# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
from ..base import ExplainerMixin, ExplanationMixin


def test_that_explainer_throws_exceptions_for_incomplete():
    class IncompleteExplainer(ExplainerMixin):
        pass

    with pytest.raises(Exception):
        _ = IncompleteExplainer()  # noqa: F841


def test_that_explainer_works_for_complete():
    class CompleteExplainer(ExplainerMixin):
        available_explanations = ["performance"]
        explainer_type = "blackbox"

    try:
        explainer = CompleteExplainer()
        assert explainer.available_explanations == ["performance"]
        assert explainer.explainer_type == "blackbox"
    except Exception:
        pytest.fail("Unexpected exception raised.")


def test_that_explanation_throws_exceptions_for_incomplete():
    class IncompleteExplanation(ExplanationMixin):
        pass

    with pytest.raises(Exception):
        _ = IncompleteExplanation()  # noqa: F841


def test_that_explanation_works_for_complete():
    class CompleteExplanation(ExplanationMixin):
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
        pytest.fail("Unexpected exception raised: {0}".format(e))
