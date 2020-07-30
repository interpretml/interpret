#!/bin/bash
# Initializes docker container on run.
# - Currently mirrors repo to first argument's filepath.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEST="$( readlink -f $1 )"
( cd $SCRIPT_DIR/.. && git ls-files -cmoz --exclude-standard | rsync -avh -0 --files-from=- . $DEST )