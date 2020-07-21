# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..visual.interactive import (
    set_show_addr,
    get_show_addr,
    shutdown_show_server,
    status_show_server,
)
from ..visual.interactive import show, init_show_server, preserve
from ..visual import interactive
from .utils import synthetic_classification
from ..glassbox import LogisticRegression, DecisionListClassifier
from .. import show_link

import requests
import os
import tempfile
import pytest


@pytest.fixture(scope="module")
def explanation():
    data = synthetic_classification()
    clf = LogisticRegression()
    clf.fit(data["train"]["X"], data["train"]["y"])

    global_exp = clf.explain_global()
    return global_exp


# TODO: Re-enable on skoperules working with latest scikit learn.
# @pytest.fixture(scope="module")
# def text_explanation(explanation):
#     data = synthetic_classification()
#     clf = DecisionListClassifier()
#     clf.fit(data["train"]["X"], data["train"]["y"])
#
#     global_exp = clf.explain_global()
#     return global_exp


def wait_for_reachable(timeout=5):
    from time import sleep

    max_tries = 3
    success = False
    for _ in range(max_tries):
        status = status_show_server()
        if status["http_reachable"]:
            success = True
            break
        sleep(timeout)

    return success


@pytest.mark.slow
def test_shutdown():
    target_addr = ("127.0.0.1", 7000)
    set_show_addr(target_addr)

    success = wait_for_reachable()
    assert success

    actual_response = shutdown_show_server()
    expected_response = True
    assert actual_response == expected_response

    # NOTE: Running it twice should still work.
    actual_response = shutdown_show_server()
    expected_response = True
    assert actual_response == expected_response


@pytest.mark.slow
def test_addr_assignment():
    target_addr = ("127.0.0.1", 7001)
    set_show_addr(target_addr)

    actual_addr = get_show_addr()

    assert target_addr == actual_addr

    shutdown_show_server()


@pytest.mark.slow
def test_status_show_server():
    target_addr = ("127.0.0.1", 7002)
    set_show_addr(target_addr)
    shutdown_show_server()

    pre_status = status_show_server()

    assert pre_status["app_runner_exists"]

    target_addr = ("127.0.0.1", 7003)
    set_show_addr(target_addr)

    post_status = status_show_server()
    assert post_status["app_runner_exists"]
    assert isinstance(post_status["addr"], tuple)

    shutdown_show_server()


@pytest.mark.slow
def test_init_show_server(explanation):
    port = 7004
    target_addr = ("127.0.0.1", port)
    base_url = "proxy/{0}".format(port)

    init_show_server(addr=target_addr, base_url=base_url, use_relative_links=False)
    show(explanation)

    # Assert that the address and url are passed to Dash
    provider = interactive.visualize_provider
    assert provider.app_runner.port == port
    assert provider.app_runner.base_url == base_url

    url = show_link(explanation)

    try:
        response = requests.get(url)
        assert response.status_code == 200
        success = True
    except requests.exceptions.RequestException:
        success = False

    assert success

    shutdown_show_server()


@pytest.mark.slow
def test_show_link(explanation):
    target_addr = ("127.0.0.1", 7005)
    set_show_addr(target_addr)

    actual_url = show_link(explanation)
    expected_id = str(id(explanation))
    status = status_show_server()
    expected_url = "http://127.0.0.1:{0}/{1}/".format(status["addr"][1], expected_id)

    assert actual_url == expected_url

    shutdown_show_server()


@pytest.mark.slow
@pytest.mark.visual
def test_show(explanation):
    explanation_li = [explanation, explanation]
    target_addr = ("127.0.0.1", 7006)
    set_show_addr(target_addr)

    show(explanation_li, share_tables=True)
    url = show_link(explanation_li)

    try:
        response = requests.get(url)
        assert response.status_code == 200
        success = True
    except requests.exceptions.RequestException:
        success = False

    assert success

    shutdown_show_server()


@pytest.mark.visual
@pytest.mark.slow
def test_preserve(explanation):
    # Overall
    result = preserve(explanation)
    assert result is None

    # Integer indexing
    selector_key = 0
    result = preserve(explanation, selector_key)
    assert result is None

    # Index by selector first column value
    selector_key = "A"
    result = preserve(explanation, selector_key)
    assert result is None

    # # Handle text explanations
    # result = preserve(text_explanation)
    # assert result is None

    # Output to file
    path = os.path.join(tempfile.mkdtemp(), "test_preserve_output.html")
    preserve(explanation, selector_key, file_name=path, auto_open=False)
    assert os.stat(path).st_size > 0

    os.remove(path)
