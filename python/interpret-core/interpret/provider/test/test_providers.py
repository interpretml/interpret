# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..compute import JobLibProvider, AzureMLProvider
from ..visualize import (
    AutoVisualizeProvider,
    DashProvider,
    PreserveProvider,
    InlineProvider,
)
import pytest

from ...test.utils import synthetic_classification
from ...glassbox import LogisticRegression


def task_fn(x, y):
    return x + y


task_args_iter = [[1, 1], [2, 2], [3, 3]]


@pytest.fixture("module")
def example_explanation():
    data = synthetic_classification()
    explainer = LogisticRegression()
    explainer.fit(data["train"]["X"], data["train"]["y"])
    explanation = explainer.explain_local(
        data["test"]["X"].head(), data["test"]["y"].head()
    )
    return explanation


def test_joblib_provider():
    provider = JobLibProvider()
    results = provider.parallel(task_fn, task_args_iter)
    assert results == [2, 4, 6]


def test_azureml_provider():
    with pytest.raises(NotImplementedError):
        provider = AzureMLProvider()
        provider.parallel(task_fn, task_args_iter)


@pytest.mark.slow
def test_auto_visualize_provider(example_explanation):
    # NOTE: We know this environment is going to use Dash.
    from ...visual.dashboard import AppRunner

    ip = "127.0.0.1"
    port = "7200"
    app_runner = AppRunner(addr=(ip, port))
    provider = AutoVisualizeProvider(app_runner=app_runner)
    provider.render(example_explanation)

    # Assert that the address is passed into Dash
    assert provider.provider.app_runner.ip == ip
    assert provider.provider.app_runner.port == port
    provider.provider.app_runner.stop()


def test_preserve_provider(example_explanation):
    provider = PreserveProvider()
    provider.render(example_explanation, key=0)


def test_inline_provider(example_explanation):
    provider = InlineProvider()
    provider.render(example_explanation, key=0)

    # NOTE: Should display error message, smoke test here.
    provider.render([example_explanation])


@pytest.mark.slow
def test_dash_provider(example_explanation):
    ip = "127.0.0.1"
    port = "7201"
    provider = DashProvider.from_address(addr=(ip, port))
    assert provider.app_runner.ip == ip
    assert provider.app_runner.port == port

    provider.render(example_explanation)
    link = provider.link(example_explanation)
    assert link is not None

    provider.app_runner.stop()
