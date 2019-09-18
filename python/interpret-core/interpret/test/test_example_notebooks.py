# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell
import pytest
import re


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
    # In the second cell, enable sampling
    nb.cells[1]["source"] = re.sub(
        "# df = df.sample", "df = df.sample", nb.cells[1]["source"]
    )

    proc.preprocess(nb, {"metadata": {"path": package_path}})

    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)

    return nb, errors


def extract_notebook_paths():
    script_path = os.path.dirname(os.path.abspath(__file__))
    notebooks_path = os.path.abspath(
        os.path.join(
            script_path, "..", "..", "..", "..", "examples", "python", "notebooks"
        )
    )

    # NOTE: This test runs only when you have the source repo.
    paths = []
    if os.path.exists(notebooks_path):
        for entry in os.scandir(notebooks_path):
            if entry.is_file() and entry.path.endswith(".ipynb"):
                paths.append(entry.path)
    return paths


notebook_paths = extract_notebook_paths()


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", notebook_paths)
def test_example_notebooks(notebook_path):
    def check_notebook(notebook_path):
        nb, errors = run_notebook(notebook_path)
        assert errors == []

    check_notebook(notebook_path)
