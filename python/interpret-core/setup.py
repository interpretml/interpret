# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from setuptools import setup, find_packages
from setuptools.command.sdist import sdist
from distutils.command.build import build
from distutils.command.install import install
from wheel.bdist_wheel import bdist_wheel


name = "interpret-core"
# NOTE: Version is replaced by a regex script.
version = "0.2.7"
long_description = """
Core system for **the** interpret package.

https://github.com/interpretml/interpret
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
        "lib/lib_ebm_native_win_x64.dll",
        "lib/lib_ebm_native_linux_x64.so",
        "lib/lib_ebm_native_mac_x64.dylib",
        "lib/lib_ebm_native_win_x64_debug.dll",
        "lib/lib_ebm_native_linux_x64_debug.so",
        "lib/lib_ebm_native_mac_x64_debug.dylib",
        "lib/lib_ebm_native_win_x64.pdb",
        "lib/lib_ebm_native_win_x64_debug.pdb",
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
    "required": [
        "numpy>=1.11.1",
        "scipy>=0.18.1",
        "pandas>=0.19.2",
        sklearn_dep,
        joblib_dep,
    ],
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
    "skoperules": ["skope-rules>=1.0.1"],
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
        "pytest>=4.3.0,<=6.0.2",
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

def _copy_native_code_to_setup():
    import os
    import shutil

    script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(script_path, '..', '..')
    sym_path = os.path.join(script_path, 'symbolic')
    source_shared_path = os.path.join(root_path, 'shared')
    target_shared_path = os.path.join(sym_path, 'shared')

    if os.path.exists(source_shared_path):  # If native code exists two directories up, update setup.py's copy.
        if os.path.exists(target_shared_path):
            shutil.rmtree(target_shared_path)
        shutil.copytree(source_shared_path, target_shared_path)

        file_names = ["build.bat", "build.sh", "LICENSE", "README.md"]
        for file_name in file_names:
            shutil.copy(
                os.path.join(root_path, file_name),
                os.path.join(sym_path, file_name)
            )
    else:  # Otherwise, ensure that native code exists for setup.py.
        if not os.path.exists(target_shared_path):
            raise Exception("Shared directory in symbolic not found. This should be configured either by setup.py or alternative build processes.")

class BuildCommand(build):
    def run(self):
        _copy_native_code_to_setup()

        # Run native compilation as well as JavaScript build.
        import subprocess
        import os
        import shutil

        script_path = os.path.dirname(os.path.abspath(__file__))
        sym_path = os.path.join(script_path, 'symbolic')

        # Native compile
        if os.name == 'nt':
            build_script = os.path.join(sym_path, "build.bat")
            subprocess.check_call([build_script], cwd=script_path)
        else:
            build_script = os.path.join(sym_path, "build.sh")
            subprocess.check_call(['bash', build_script], cwd=script_path)

        source_dir = os.path.join(sym_path, 'python', 'interpret-core', 'interpret', 'lib')
        target_dir = os.path.join(script_path, 'interpret', 'lib')
        os.makedirs(target_dir, exist_ok=True )
        file_names = os.listdir(source_dir)
        for file_name in file_names:
            shutil.move(
                os.path.join(source_dir, file_name),
                os.path.join(target_dir, file_name)
            )

        # JavaScript compile
        js_path = os.path.join(script_path, 'js')
        if os.getenv('AGENT_NAME') or os.name != 'nt':  # In DevOps / Linux
            subprocess.run(["npm install"], cwd=js_path, shell=True)
            subprocess.run(["npm run build-prod"], cwd=js_path, shell=True)
        else:
            subprocess.run(["npm", "install"], cwd=js_path, shell=True)
            subprocess.run(["npm", "run", "build-prod"], cwd=js_path, shell=True)
        js_bundle_src = os.path.join(js_path, "dist", "interpret-inline.js")
        js_bundle_dest = os.path.join(
            "interpret", "lib", "interpret-inline.js"
        )
        os.makedirs(os.path.dirname(js_bundle_dest), exist_ok=True)
        shutil.copyfile(js_bundle_src, js_bundle_dest)

        build.run(self)

class InstallCommand(install):
   def run(self):
       _copy_native_code_to_setup()  # This needs to run pre-build to store native code in the sdist.
       install.run(self)

class SDistCommand(sdist):
   def run(self):
       _copy_native_code_to_setup()  # This needs to run pre-build to store native code in the sdist.
       sdist.run(self)

class BDistWheelCommand(bdist_wheel):
   def run(self):
       bdist_wheel.run(self)

setup(
    name=name,
    version=version,
    author="InterpretML Team",
    author_email="interpret@microsoft.com",
    description="Fit interpretable machine learning models. Explain blackbox machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/interpret",
    cmdclass={
        'install': InstallCommand,
        'sdist': SDistCommand,
        'build': BuildCommand,
        'bdist_wheel': BDistWheelCommand,
    },
    packages=find_packages(),
    package_data=package_data,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require=extras,
    entry_points=entry_points,
    install_requires=[],
)
