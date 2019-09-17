# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..compute import JobLibProvider, AzureMLProvider
from ..visualize import AutoVisualizeProvider, DashProvider, PreserveProvider
import pytest

from ...test.utils import synthetic_classification
from ...glassbox import LogisticRegression


def task_fn(x, y):
    return x + y


task_args_iter = [
    [1, 1],
    [2, 2],
    [3, 3]
]


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


def test_auto_visualize_provider(example_explanation):
    # NOTE: We know this environment is going to use Dash.
    provider = AutoVisualizeProvider(addr=("127.0.0.1", "7200"))
    provider.render(example_explanation)
    provider.provider.app_runner.stop()


def test_preserve_provider(example_explanation):
    provider = PreserveProvider()
    provider.render(example_explanation, key=0)


def test_dash_provider(example_explanation):
    provider = DashProvider(addr=("127.0.0.1", "7201"))
    provider.render(example_explanation)
    link = provider.link(example_explanation)
    assert link is not None

    provider.app_runner.stop()

