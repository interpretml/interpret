# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from setuptools import setup, find_packages

name = "interpret-api"
# NOTE: Versioning for interpret-api does not update step-lock with other interpret packages.
version = "0.0.1"
long_description = """
Minimal dependency Interpret API for machine learning interpretability.

https://github.com/interpretml/interpret
"""

setup(
    name=name,
    version=version,
    author="The InterpretML Contributors",
    author_email="interpret@microsoft.com",
    description="Fit interpretable machine learning models. Explain blackbox machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/interpret",
    packages=find_packages(),
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "slicer>=0.0.5",
    ],
)
