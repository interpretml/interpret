# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import subprocess
import os
import glob
import shutil
from setuptools.command.build import build
from setuptools.command.install import install
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist

# NOTE: Version is replaced by a regex script.
version = "0.6.1"


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
    else:  # Otherwise, ensure that native code exists for setup.py.
        if not os.path.exists(target_shared_path):
            raise Exception(
                "Shared directory in symbolic not found. This should be configured either by setup.py or alternative build processes."
            )


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


def build_copy_root_libebm():
    # this function is unused since we rely on the build pipeline to make
    # the multi-OS shared libraries .dll, .so, and .dylib

    script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(script_path, "..", "..")

    # Native compile
    if os.name == "nt":
        build_script = os.path.join(root_path, "build.bat")
        subprocess.check_call([build_script], cwd=root_path)
    else:
        build_script = os.path.join(root_path, "build.sh")
        subprocess.check_call(["/bin/sh", build_script], cwd=root_path)

    from_path = os.path.join(root_path, "bld", "lib")
    to_path = os.path.join(script_path, "interpret", "root", "bld", "lib")

    for filename in glob.glob(os.path.join(from_path, "*.dll")):
        shutil.copy(filename, to_path)
    for filename in glob.glob(os.path.join(from_path, "*.pdb")):
        shutil.copy(filename, to_path)
    for filename in glob.glob(os.path.join(from_path, "*.so")):
        shutil.copy(filename, to_path)
    for filename in glob.glob(os.path.join(from_path, "*.dylib")):
        shutil.copy(filename, to_path)


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
    subprocess.run("npm install && npm run build-prod", cwd=js_path, shell=True)

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


class InstallCommand(install):
    def run(self):
        install.run(self)


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
    url="https://github.com/interpretml/interpret",
    cmdclass={
        "install": InstallCommand,
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
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
        "numpy>=1.11.1",
        "scipy>=0.18.1",
        "pandas>=0.19.2",
        "scikit-learn>=0.18.1",
        "joblib>=0.11",
        "aplr>=10.4.5",
    ],
    extras_require={
        "debug": ["psutil>=5.6.2"],
        "notebook": ["ipykernel>=4.10.0", "ipython>=5.5.0"],
        # Plotly (required if .visualize is ever called)
        "plotly": ["plotly>=3.8.1"],
        # Explainers
        "lime": ["lime>=0.1.1.33"],
        "sensitivity": ["SALib>=1.3.3"],
        "shap": ["shap>=0.28.5", "dill>=0.2.5"],
        "linear": [],
        "skoperules": ["skope-rules>=1.0.1"],
        "treeinterpreter": ["treeinterpreter>=0.2.2"],
        # Dash
        "dash": [
            # dash 2.* removed the dependencies on: dash-html-components, dash-core-components, dash-table
            "dash>=1.0.0",
            "dash-core-components>=1.0.0",  # dash 2.* removes the need for this dependency
            "dash-html-components>=1.0.0",  # dash 2.* removes the need for this dependency
            "dash-table>=4.1.0",  # dash 2.* removes the need for this dependency
            "dash-cytoscape>=0.1.1",
            "gevent>=1.3.6",
            "requests>=2.19.0",
        ],
        # Testing
        "testing": [
            "scikit-learn>=1.0.0",
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
    },
)
