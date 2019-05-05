# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
from setuptools import setup, find_packages
import os
script_path = os.path.dirname(os.path.abspath(__file__))

needs_dev = {
    'pytest', 'test', 'ptr', 'lint', 'flake8', 'doc', 'build_sphinx'
}.intersection(sys.argv)
dev_tools = [
    'sphinx>=1.8.4',
    'flake8>=3.7.6',
    'pytest-cov>=2.6.1',
] if needs_dev else []

long_description = """
In the beginning machines learned in darkness, and data scientists struggled in the void to explain them.

Let there be light.
"""

name = 'interpret'
version = '0.0.2'
setup(
    name=name,
    version=version,
    author='InterpretML Team',
    author_email='interpret@microsoft.com',
    description='Nobody knows how machine learning decisions are made. Now you do.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/interpret',
    packages=find_packages(),
    package_data={
        'interpret': [
            'lib/ebmcore_win_x64.dll',
            'lib/ebmcore_linux_x64.so',
            'lib/ebmcore_mac_x64.dylib',
            'lib/ebmcore_win_x64_debug.dll',
            'lib/ebmcore_linux_x64_debug.so',
            'lib/ebmcore_mac_x64_debug.dylib',

            'lib/ebmcore_win_x64.pdb',
            'lib/ebmcore_win_x64_debug.pdb',
            'visual/assets/udash.css',
            'visual/assets/udash.js',
            'visual/assets/favicon.ico',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Extra
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', version),
            'source_dir': ('setup.py', 'docs')
        }
    },
    # NOTE: Numpy here is a workaround to skope-rules' dependencies.
    setup_requires=[
        'numpy>=1.15.1',
        'scipy>=1.2.1',
    ] + dev_tools,
    tests_require=[] + dev_tools,
    install_requires=[
        # Algorithms
        'SALib>=1.3.3',
        'lime>=0.1.1.33',
        'shap>=0.28.5',
        'skope-rules>=1.0.0',

        # Service related
        # NOTE: Dash is pinned so to avoid dependency hell.
        'plotly>=3.8.1',
        'dash==0.39.0',
        'dash-core-components==0.44.0',
        'dash-cytoscape==0.1.1',
        'dash-html-components==0.14.0',
        'dash-renderer==0.20.0',
        'dash-table-experiments==0.6.0',
        'gevent>=1.4.0',

        # Core
        'joblib>=0.12.5',
        'pandas>=0.24.0',
        'scikit-learn>=0.20.0',
        'ipykernel>=5.1.0',
        'ipython>=7.4.0',
        'numpy>=1.15.1',
        'scipy>=1.2.1',

        # Testing
        'pytest>=4.3.0',
        'pytest-runner>=4.4',
        'hypothesis>=4.18.3',
        'nbconvert>=5.4.1',
    ]
)
