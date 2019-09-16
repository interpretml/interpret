# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license
# This replaces the version number with the new one
# Arguments are 'old-version new-version'

import re
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("{} old-version new-version".format(sys.argv[0]))
        sys.exit(1)

    old_version = sys.argv[1]
    new_version = sys.argv[2]

    # Replace in versions in setup.py files.
    script_path = os.path.dirname(os.path.abspath(__file__))
    interpret_core_setup = os.path.join(script_path, "..", "interpret-core", "setup.py")
    interpret_setup = os.path.join(script_path, "..", "interpret", "setup.py")
    interpret_core_version = os.path.join(script_path, "..", "interpret-core", "interpret", "version.py")
    targets = [
        (interpret_core_setup, 'version = "{}"'),
        (interpret_setup, 'version = "{}"'),
        (interpret_core_version, '__version__ = "{}"')
    ]
    for target_path, target_format in targets:
        new_lines = []
        with open(target_path, 'r') as f:
            matched = False
            for line in f:
                find = target_format.format(old_version)
                replace = target_format.format(new_version)
                if re.match(find, line) is not None:
                    matched = True
                new_line = re.sub(find, replace, line)
                new_lines.append(new_line)
            if not matched:
                print("Did not match: {}".format(find), file=sys.stderr)

        if matched:
            with open(target_path, 'w') as f:
                for new_line in new_lines:
                    f.write(new_line)
