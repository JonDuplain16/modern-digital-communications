#!/bin/bash
# Build (convert + execute) notebooks one process at a time: bash tests/build_all.sh [pattern]
cd "$(dirname "$0")/.."
for f in labs/${1:-lab}*.py; do
  b=$(basename "$f" .py)
  timeout 900 python3 tests/build_notebooks.py "$b" 2>&1 | grep -v WARNING | tail -3
done
echo ALLDONE
