#!/bin/bash
# Build the book: figures (optional: ./build.sh figs) then three LaTeX passes + index.
cd "$(dirname "$0")"
if [ "$1" == "figs" ]; then for s in figscripts/ch*_figs.py; do (cd figscripts && python3 $(basename $s)) || exit 1; done; fi
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1
makeindex -q main.idx
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
grep -E "^!" build.log | sort | uniq -c | head -20
echo "pages: $(pdfinfo main.pdf 2>/dev/null | grep Pages)"
grep -c "undefined" main.log | xargs echo "undefined-ref warnings:"
