# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from six import raise_from


def test_import_demo_explainer():
    try:
        from interpret.ext.blackbox import BlackboxExplainerExample  # noqa
    except ImportError as import_error:
        raise_from(Exception("Failure in interpret.ext.blackbox while trying "
                             "to load example explainers through extension_utils", import_error))
