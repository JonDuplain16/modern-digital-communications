#!/bin/bash
# Build (convert + execute) notebooks one process at a time: bash tests/build_all.sh [pattern]
# Works in Git Bash on Windows as well as Linux/macOS.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
for f in labs/${1:-lab}*.py; do
  b=$(basename "$f" .py)
  "$PY" tests/build_notebooks.py "$b" 2>&1 | tail -3
done
echo ALLDONE
