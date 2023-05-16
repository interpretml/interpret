# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from setuptools import setup, find_packages

name = "interpret"
# NOTE: Version is replaced by a regex script.
version = "0.4.1"
long_description = """
In the beginning machines learned in darkness, and data scientists struggled in the void to explain them.

Let there be light.

https://github.com/interpretml/interpret
"""
interpret_core_extra = [
    "required",
    "debug",
    "notebook",
    "plotly",
    "lime",
    "sensitivity",
    "shap",
    "ebm",
    "skoperules",
    "linear",
    "decisiontree",
    "dash",
    "treeinterpreter",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "interpret-core[{}]=={}".format(",".join(interpret_core_extra), version)
    ],
)
