# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from six import raise_from

from .utils import assert_valid_explanation
from .utils import synthetic_classification

from ..glassbox import LogisticRegression


def test_import_demo_explainer():
    try:
        from interpret.ext.blackbox import BlackboxExplainerExample  # noqa
    except ImportError as import_error:
        raise_from(
            Exception(
                "Failure in interpret.ext.blackbox while trying "
                "to load example explainers through extension_utils",
                import_error,
            ),
            None,
        )


def test_demo_explainer():
    from interpret.ext.blackbox import BlackboxExplainerExample

    data = synthetic_classification()
    blackbox = LogisticRegression()
    blackbox.fit(data["train"]["X"], data["train"]["y"])
    predict_fn = lambda x: blackbox.predict_proba(x)  # noqa: E731

    explainer = BlackboxExplainerExample(predict_fn, data["train"]["X"])
    explanation = explainer.explain_local(data["test"]["X"].head(), data["test"]["y"].head())
    assert_valid_explanation(explanation)
