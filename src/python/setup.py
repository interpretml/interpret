# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
from setuptools import setup, find_packages
import os
import re
import io

script_path = os.path.dirname(os.path.abspath(__file__))

needs_dev = {
    "pytest",
    "test",
    "ptr",
    "lint",
    "flake8",
    "doc",
    "build_sphinx",
}.intersection(sys.argv)
dev_tools = ["sphinx>=1.8.4", "flake8>=3.7.7", "pytest-cov>=2.6.1"] if needs_dev else []

long_description = """
In the beginning machines learned in darkness, and data scientists struggled in the void to explain them.

Let there be light.

https://github.com/microsoft/interpret
"""

name = "interpret"

# Version logic derived from below:
# https://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package
version = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open(os.path.join(script_path, name, "version.py"), encoding="utf_8_sig").read(),
).group(1)

###
# get install_requires data from requirements.txt
#
install_requires = []
with open("./requirements.txt") as f:
    install_requires = f.read().splitlines()
install_requires = [x for x in install_requires if not x.startswith('#')]

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
    package_data={
        "interpret": [
            "lib/lib_ebmcore_win_x64.dll",
            "lib/lib_ebmcore_linux_x64.so",
            "lib/lib_ebmcore_mac_x64.dylib",
            "lib/lib_ebmcore_win_x64_debug.dll",
            "lib/lib_ebmcore_linux_x64_debug.so",
            "lib/lib_ebmcore_mac_x64_debug.dylib",
            "lib/lib_ebmcore_win_x64.pdb",
            "lib/lib_ebmcore_win_x64_debug.pdb",
            "visual/assets/udash.css",
            "visual/assets/udash.js",
            "visual/assets/favicon.ico",
            "pytest.ini",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Extra
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", version),
            "source_dir": ("setup.py", "docs"),
        }
    },
    setup_requires=[] + dev_tools,
    tests_require=[] + dev_tools,
    install_requires=install_requires,
)
