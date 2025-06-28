# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from setuptools import find_packages, setup

name = "interpret"
# NOTE: Version is replaced by a regex script.
version = "0.6.13"
long_description = """
In the beginning machines learned in darkness, and data scientists struggled in the void to explain them.

Let there be light.

https://github.com/interpretml/interpret
"""
interpret_core_extra = [
    "debug",
    "notebook",
    "plotly",
    # "lime",  # no longer maintained
    "sensitivity",
    "shap",
    # "skoperules",  # no longer maintained
    "linear",
    "dash",
    # "treeinterpreter",  # no longer maintained
    "aplr",
]

setup(
    name=name,
    version=version,
    author="The InterpretML Contributors",
    author_email="interpret@microsoft.com",
    description="Fit interpretable models. Explain blackbox machine learning.",
    long_description=long_description,
    url="https://github.com/interpretml/interpret",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "interpret-core[{}]=={}".format(",".join(interpret_core_extra), version)
    ],
)
