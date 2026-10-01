#!/bin/bash
# Build the book: figures (optional: ./build.sh figs) then three LaTeX passes + index.
# Works on Linux/macOS (TeX Live) and Windows Git Bash (MiKTeX, user install).
cd "$(dirname "$0")"
MK="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64"
[ -d "$MK" ] && export PATH="$MK:$PATH"
PY=python3; command -v python3 >/dev/null && python3 -c "" 2>/dev/null || PY=python
if [ "$1" == "figs" ]; then for s in figscripts/*_figs.py; do (cd figscripts && $PY $(basename $s)) || exit 1; done; fi
$PY tools/gen_credits.py
LATEX="pdflatex -interaction=nonstopmode -halt-on-error"
[ "$MK" ] && [ -d "$MK" ] && LATEX="pdflatex -interaction=nonstopmode --enable-installer"
$LATEX main.tex > build.log 2>&1
makeindex -q main.idx
$LATEX main.tex >> build.log 2>&1
$LATEX main.tex >> build.log 2>&1
grep -E "^!" build.log | sort | uniq -c | head -20
echo "pages: $($PY -c "import pymupdf;print(pymupdf.open('main.pdf').page_count)" 2>/dev/null)"
grep -c "undefined" main.log | xargs echo "undefined-ref warnings:"
