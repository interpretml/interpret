# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license


from ...glassbox import LinearRegression
from ...api.base import ExplainerMixin
from ..extension import (
    _is_valid_blackbox_explainer,
    _is_valid_provider,
    _is_valid_glassbox_explainer,
    _is_valid_greybox_explainer,
    _is_valid_data_explainer,
    _is_valid_perf_explainer,
)


class NotEvenAnExplainer:
    def nothing(self):
        pass


def test_invalid_provider():
    # NOTE: No parallel nor render method.
    class InvalidProvider:
        pass

    assert not _is_valid_provider(LinearRegression)
    assert not _is_valid_provider(InvalidProvider)


def test_invalid_data_explainer():
    class InvalidDataExplainer(ExplainerMixin):
        explainer_type = "data"
        available_explanations = ["local"]

    assert not _is_valid_data_explainer(LinearRegression)
    assert not _is_valid_data_explainer(InvalidDataExplainer)
    assert not _is_valid_data_explainer(NotEvenAnExplainer)


def test_invalid_perf_explainer():
    class InvalidPerfExplainer(ExplainerMixin):
        explainer_type = "perf"
        available_explanations = ["local"]

    assert not _is_valid_perf_explainer(LinearRegression)
    assert not _is_valid_perf_explainer(InvalidPerfExplainer)
    assert not _is_valid_perf_explainer(NotEvenAnExplainer)


def test_invalid_greybox_explainer():
    class InvalidGreyboxExplainer(ExplainerMixin):
        explainer_type = "specific"
        available_explanations = ["local"]

    assert not _is_valid_greybox_explainer(LinearRegression)
    assert not _is_valid_greybox_explainer(InvalidGreyboxExplainer)
    assert not _is_valid_greybox_explainer(NotEvenAnExplainer)


def test_invalid_glassbox_explainer():
    class InvalidGlassboxExplainer(ExplainerMixin):
        explainer_type = "model"
        available_explanations = ["local"]

    assert not _is_valid_glassbox_explainer(InvalidGlassboxExplainer)
    assert not _is_valid_glassbox_explainer(NotEvenAnExplainer)


def test_invalid_blackbox_explainer():
    # NOTE: Available method claims local exists, but there is not respective method.
    class InvalidBlackboxExplainer(ExplainerMixin):
        explainer_type = "blackbox"
        available_explanations = ["local"]

    assert not _is_valid_blackbox_explainer(LinearRegression)
    assert not _is_valid_blackbox_explainer(InvalidBlackboxExplainer)
    assert not _is_valid_blackbox_explainer(NotEvenAnExplainer)
