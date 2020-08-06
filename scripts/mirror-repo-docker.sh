#!/bin/bash
# Mirrors this repository into first argument's path.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEST="$( readlink -f $1 )"
mkdir -p $DEST
rsync -avh --exclude='node_modules' --exclude='tmp' --exclude='.git' $SCRIPT_DIR/.. $DEST