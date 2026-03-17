# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import os
import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build import build
from setuptools.command.sdist import sdist

# NOTE: Version is replaced by a regex script.
version = "0.7.7"


def _copy_native_code_to_setup():
    script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(script_path, "..", "..")
    sym_path = os.path.join(script_path, "interpret", "root")
    source_shared_path = os.path.join(root_path, "shared", "libebm")
    target_shared_path = os.path.join(sym_path, "shared", "libebm")

    # If native code exists two directories up, update setup.py's copy.
    if os.path.exists(source_shared_path):
        if os.path.exists(target_shared_path):
            shutil.rmtree(target_shared_path)
        shutil.copytree(source_shared_path, target_shared_path)

        file_names = ["build.bat", "build.sh", "LICENSE", "README.md"]
        for file_name in file_names:
            shutil.copy(
                os.path.join(root_path, file_name), os.path.join(sym_path, file_name)
            )
    elif not os.path.exists(target_shared_path):
        msg = "Shared directory in symbolic not found. This should be configured either by setup.py or alternative build processes."
        raise FileNotFoundError(msg)


def build_libebm():
    script_path = os.path.dirname(os.path.abspath(__file__))
    sym_path = os.path.join(script_path, "interpret", "root")

    # Native compile
    if os.name == "nt":
        build_script = os.path.join(sym_path, "build.bat")
        subprocess.check_call([build_script], cwd=sym_path)
    else:
        build_script = os.path.join(sym_path, "build.sh")
        subprocess.check_call(["/bin/sh", build_script], cwd=sym_path)


def build_vis_if_needed():
    script_path = os.path.dirname(os.path.abspath(__file__))
    js_bundle_dest = os.path.join(
        script_path, "interpret", "root", "bld", "lib", "interpret-inline.js"
    )
    if os.path.exists(js_bundle_dest):
        # already exists, so we are done
        return

    # JavaScript compile
    js_path = os.path.join(script_path, "..", "..", "shared", "vis")
    subprocess.run(
        "npm install && npm run build-prod", cwd=js_path, shell=True, check=False
    )

    js_bundle_src = os.path.join(js_path, "dist", "interpret-inline.js")
    os.makedirs(os.path.dirname(js_bundle_dest), exist_ok=True)
    shutil.copyfile(js_bundle_src, js_bundle_dest)

    js_bundle_src_lic = os.path.join(js_path, "dist", "interpret-inline.js.LICENSE.txt")
    js_bundle_dest_lic = os.path.join(
        script_path,
        "interpret",
        "root",
        "bld",
        "lib",
        "interpret-inline.js.LICENSE.txt",
    )
    shutil.copyfile(js_bundle_src_lic, js_bundle_dest_lic)


class BuildCommand(build):
    def run(self):
        # when making a wheel we depend on a cloud build platform to make the various OS and platform
        # shared library binaries, so we do not need to build them here. In conda-forge we use the
        # libebm dependency which is the shared library. When downloading from source we prefer to
        # build the library when used since the C++ code could change. The only place we want to
        # build the C++ code is when build is being called as part of an sdist installation, so
        # check if the symbolic path exists

        script_path = os.path.dirname(os.path.abspath(__file__))
        sym_path = os.path.join(script_path, "interpret", "root", "shared", "libebm")

        if os.path.exists(sym_path):
            # this should only be triggered in an sdist
            build_libebm()

        # IMPORTANT:
        #
        # When building our bdist, we rely on a build pipeline to make the
        # .dll, .so, and .dylib since this is a multi-OS process. The build pipeline
        # should put them in ./interpret/python/interpret-core/interpret/root/bld/lib
        # If you want to build a single platform bdist, you must manually build
        # and copy the shared library artifacts into the directory
        #
        # If this behavior were changed to build the shared library for this platform
        # then you must change how the conda-forge build works since we rely on the fact
        # that we're not making shared libraries here since otherwise the conda-forge
        # build process would include them into the package there!
        #

        build_vis_if_needed()

        build.run(self)


class SDistCommand(sdist):
    def run(self):
        # This needs to run pre-build to store native code in the sdist.
        _copy_native_code_to_setup()
        # the sdist is just for building on odd platforms, but js should work on any platform
        build_vis_if_needed()
        sdist.run(self)


setup(
    name="interpret-core",
    version=version,
    author="The InterpretML Contributors",
    author_email="interpret@microsoft.com",
    description="Fit interpretable models. Explain blackbox machine learning.",
    long_description="""
Minimal dependency core system for the interpret package.

https://github.com/interpretml/interpret
""",
    long_description_content_type="text/plain",
    url="https://github.com/interpretml/interpret",
    cmdclass={
        "sdist": SDistCommand,
        "build": BuildCommand,
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "interpret": [
            "root/bld/lib/libebm.dll",
            "root/bld/lib/libebm.pdb",
            "root/bld/lib/libebm_debug.dll",
            "root/bld/lib/libebm_debug.pdb",
            "root/bld/lib/libebm.so",
            "root/bld/lib/libebm_debug.so",
            "root/bld/lib/libebm.dylib",
            "root/bld/lib/libebm_debug.dylib",
            "root/bld/lib/libebm_win_x64.dll",
            "root/bld/lib/libebm_win_x64.pdb",
            "root/bld/lib/libebm_win_x64_debug.dll",
            "root/bld/lib/libebm_win_x64_debug.pdb",
            "root/bld/lib/libebm_linux_x64.so",
            "root/bld/lib/libebm_linux_x64_debug.so",
            "root/bld/lib/libebm_linux_arm.so",
            "root/bld/lib/libebm_linux_arm_debug.so",
            "root/bld/lib/libebm_mac_x64.dylib",
            "root/bld/lib/libebm_mac_x64_debug.dylib",
            "root/bld/lib/libebm_mac_arm.dylib",
            "root/bld/lib/libebm_mac_arm_debug.dylib",
            "root/bld/lib/interpret-inline.js",
            "root/bld/lib/interpret-inline.js.LICENSE.txt",
            "visual/assets/udash.css",
            "visual/assets/udash.js",
            "visual/assets/favicon.ico",
        ]
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
    ],
    entry_points={
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
    },
    install_requires=[
        "numpy>=1.25",
        "pandas>=0.24",
        "scikit-learn>=1.6.0",
        "joblib>=0.11",
    ],
    extras_require={
        "debug": ["psutil>=5.6.2"],
        "notebook": ["ipython>=5.5.0"],
        # Plotly (required if .visualize is ever called)
        "plotly": ["plotly>=3.8.1"],
        # Export
        "excel": [
            "Xlsxwriter>=3.0.1",
            "dotsi>=0.0.3",
            "seaborn>=0.13.2",
            "matplotlib>=3.9.1",
        ],
        # Explainers
        "lime": ["lime>=0.1.1.33"],
        "sensitivity": ["SALib>=1.3.3"],
        # installing ipywidgets removes crud in SHAP notebooks during fitting
        "shap": ["shap>=0.28.5", "ipywidgets>=7.4.2"],
        "linear": ["scikit-learn>=1.6.0"],
        "skoperules": ["skope-rules>=1.0.1"],
        "treeinterpreter": ["treeinterpreter>=0.2.2"],
        "aplr": ["aplr>=10.6.1"],
        # Dash
        "dash": [
            "dash>=2.0.0",
            "dash-cytoscape>=0.1.1",
            "flask>=1.0.4",
            "gevent>=1.3.6",
            "requests>=2.19.0",
        ],
        # Testing
        "testing": [
            "scipy>=0.18.1",
            "pytest>=4.3.0",
            "pytest-xdist>=1.29",
            "nbconvert>=5.4.1",
            "nbformat>=4.4.0",
            "selenium>=3.141.0",
            "pytest-cov>=2.6.1",
        ],
    },
)
