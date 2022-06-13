#!/bin/bash
set -x

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ../build/docker
cp Dockerfile ../build/docker/
cd .. && python -m build && cp dist/* build/docker/
cd build/docker && docker build -t interpretml/powerlift:0.0.1 .