# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ....glassbox import LinearRegression
from ....api.base import ExplainerMixin
from .. import _is_valid_glassbox_explainer


def test_invalid_glassbox_explainer():
    # NOTE: Available method claims local exists, but there is not respective method.
    class InvalidGlassboxExplainer(ExplainerMixin):
        explainer_type = "glassbox"
        available_explanations = ["local"]

    class NotEvenAnExplainer:
        def nothing(self):
            pass

    assert not _is_valid_glassbox_explainer(LinearRegression)
    assert not _is_valid_glassbox_explainer(InvalidGlassboxExplainer)
    assert not _is_valid_glassbox_explainer(NotEvenAnExplainer)
