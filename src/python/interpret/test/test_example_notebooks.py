# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell
import pytest


def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name="python3")
    proc.allow_errors = True
    script_path = os.path.dirname(os.path.abspath(__file__))
    package_path = os.path.abspath(os.path.join(script_path, "..", ".."))

    # Add shutdown for show method.
    shutdown_cell = new_code_cell(
        "from interpret import shutdown_show_server\nshutdown_show_server()"
    )
    nb.cells.append(shutdown_cell)

    proc.preprocess(nb, {"metadata": {"path": package_path}})

    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)

    return nb, errors


@pytest.mark.skip
def test_example_notebooks():
    script_path = os.path.dirname(os.path.abspath(__file__))
    notebooks_path = os.path.abspath(
        os.path.join(script_path, "..", "..", "..", "..", "examples", "notebooks")
    )

    # NOTE: This test runs only when you have the source repo.
    if os.path.exists(notebooks_path):
        for entry in os.scandir(notebooks_path):
            if entry.is_file() and entry.path.endswith(".ipynb"):
                nb, errors = run_notebook(entry.path)

                assert errors == []
