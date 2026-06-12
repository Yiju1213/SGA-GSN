#!/usr/bin/env sh
set -eu
ROOT="$(pwd)"

cd "$ROOT/extensions/chamfer_dist"
python setup.py install --user

cd "$ROOT/extensions/emd"
python setup.py install --user
