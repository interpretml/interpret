# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ....glassbox import LinearRegression
from ....api.base import ExplainerMixin
from .. import _is_valid_blackbox_explainer


def test_invalid_blackbox_explainer():
    # NOTE: Available method claims local exists, but there is not respective method.
    class InvalidBlackboxExplainer(ExplainerMixin):
        explainer_type = "blackbox"
        available_explanations = ["local"]

    class NotEvenAnExplainer:
        def nothing(self):
            pass

    assert not _is_valid_blackbox_explainer(LinearRegression)
    assert not _is_valid_blackbox_explainer(InvalidBlackboxExplainer)
    assert not _is_valid_blackbox_explainer(NotEvenAnExplainer)
