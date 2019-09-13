# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..glassbox import LogisticRegression
from .utils import synthetic_classification


# TODO: Harden these tests later to check content from data method.
def test_mli_visualize_interop():
    data = synthetic_classification()
    lr = LogisticRegression()

    lr.fit(data["train"]["X"], data["train"]["y"])

    global_exp = lr.explain_global()
    assert "mli" in global_exp.data(-1)
    global_overall_viz = global_exp.visualize()
    assert global_overall_viz is not None
    global_specific_viz = global_exp.visualize(0)
    assert global_specific_viz is not None
    mli_global_specific_viz = global_exp.visualize(("mli", 0))
    assert mli_global_specific_viz is not None

    local_exp = lr.explain_local(data["test"]["X"].head(), data["test"]["y"].head())
    assert "mli" in local_exp.data(-1)
    local_specific_viz = local_exp.visualize(0)
    assert local_specific_viz is not None
    mli_local_specific_viz = local_exp.visualize(("mli", 0))
    assert mli_local_specific_viz is not None
