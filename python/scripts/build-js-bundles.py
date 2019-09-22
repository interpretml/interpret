# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# Cross-platform build script for JS bundles required by Python layer.

import subprocess
import os
from shutil import copyfile


if __name__ == '__main__':
    script_path = os.path.dirname(os.path.abspath(__file__))
    js_dir = os.path.join(script_path, "..", "interpret-core", "js")

    # NOTE: Using shell=True can be a security hazard where there is user inputs.
    # In this case, there are no user inputs.
    subprocess.run(["npm", "install"], cwd=js_dir, shell=True)
    subprocess.run(["npm", "run", "build-prod"], cwd=js_dir, shell=True)
    js_bundle_src = os.path.join(js_dir, "dist", "bundle.js")
    js_bundle_dest = os.path.join(
        script_path, "..", "interpret-core",
        "interpret", "lib", "interpret-inline.js"
    )
    copyfile(js_bundle_src, js_bundle_dest)
