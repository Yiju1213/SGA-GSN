#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"

cd "$ROOT/extensions/chamfer_dist"
python setup.py install --user

cd "$ROOT/extensions/emd"
python setup.py install --user
