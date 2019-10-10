# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from setuptools import setup, find_packages

name = "interpret-core"
# NOTE: Version is replaced by a regex script.
version = "0.1.18"
long_description = """
Core system for **the** interpret package.

https://github.com/microsoft/interpret
"""

entry_points = {
    "interpret_ext_blackbox": [
        "ExampleBlackboxExplainer = interpret.ext.examples:ExampleBlackboxExplainer"
    ],
    "interpret_ext_data": [
        "ExampleDataExplainer = interpret.ext.examples:ExampleDataExplainer"
    ],
    "interpret_ext_perf": [
        "ExamplePerfExplainer = interpret.ext.examples:ExamplePerfExplainer"
    ],
    "interpret_ext_glassbox": [
        "ExampleGlassboxExplainer = interpret.ext.examples:ExampleGlassboxExplainer"
    ],
    "interpret_ext_greybox": [
        "ExampleGreyboxExplainer = interpret.ext.examples:ExampleGreyboxExplainer"
    ],
    "interpret_ext_provider": [
        "ExampleVisualizeProvider = interpret.ext.examples:ExampleVisualizeProvider"
    ],
}
package_data = {
    "interpret": [
        "lib/lib_ebmcore_win_x64.dll",
        "lib/lib_ebmcore_linux_x64.so",
        "lib/lib_ebmcore_mac_x64.dylib",
        "lib/lib_ebmcore_win_x64_debug.dll",
        "lib/lib_ebmcore_linux_x64_debug.so",
        "lib/lib_ebmcore_mac_x64_debug.dylib",
        "lib/lib_ebmcore_win_x64.pdb",
        "lib/lib_ebmcore_win_x64_debug.pdb",
        "lib/interpret-inline.js",
        "visual/assets/udash.css",
        "visual/assets/udash.js",
        "visual/assets/favicon.ico",
        "pytest.ini",
    ]
}
sklearn_dep = "scikit-learn>=0.18.1"
joblib_dep = "joblib>=0.11"
extras = {
    # Core
    "required": ["numpy>=1.11.1", "scipy>=0.18.1", "pandas>=0.19.2", sklearn_dep],
    "debug": ["psutil>=5.6.2"],
    "notebook": ["ipykernel>=5.1.0", "ipython>=7.4.0"],
    # Plotly (required if .visualize is ever called)
    "plotly": ["plotly>=3.8.1"],
    # Explainers
    "lime": ["lime>=0.1.1.33"],
    "sensitivity": ["SALib>=1.3.3"],
    "shap": ["shap>=0.28.5", "dill>=0.2.5"],
    "ebm": [joblib_dep],
    "linear": [],
    "decisiontree": [joblib_dep],
    "skoperules": ["skope-rules>=1.0.0"],
    "treeinterpreter": ["treeinterpreter>=0.2.2"],
    # Dash
    "dash": [
        "dash>=1.0.0",
        "dash-cytoscape>=0.1.1",
        "dash-table>=4.1.0",
        "gevent>=1.3.6",
        "requests>=2.19.0",
    ],
    # Testing
    "testing": [
        "pytest>=4.3.0",
        "pytest-runner>=4.4",
        "pytest-xdist>=1.29",
        "nbconvert>=5.4.1",
        "selenium>=3.141.0",
        "pytest-cov>=2.6.1",
        "flake8>=3.7.7",
        "jupyter>=1.0.0",
        "ipywidgets>=7.4.2",
    ],
}

setup(
    name=name,
    version=version,
    author="InterpretML Team",
    author_email="interpret@microsoft.com",
    description="Fit interpretable models. Explain blackbox machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/interpret",
    packages=find_packages(),
    package_data=package_data,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require=extras,
    entry_points=entry_points,
    install_requires=[],
)
