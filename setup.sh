#!/usr/bin/env bash
# One-time environment setup for the textbook + labs. Safe to re-run.
set -e
cd "$(dirname "$0")"

echo "== Python virtual environment (.venv) =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -q -r moderncomms-labs/requirements.txt pypdf pillow ipykernel
echo "   Python packages installed. Activate later with: source .venv/bin/activate"

echo "== Checking LaTeX toolchain =="
missing=0
for c in pdflatex makeindex pdftoppm pdfinfo; do
  if ! command -v $c >/dev/null 2>&1; then echo "   MISSING: $c"; missing=1; fi
done
for p in mathpazo tcolorbox pgfplots tocloft titlesec emptypage imakeidx siunitx bookmark; do
  if command -v kpsewhich >/dev/null && ! kpsewhich $p.sty >/dev/null; then echo "   MISSING LaTeX package: $p"; missing=1; fi
done
if [ $missing -eq 1 ]; then
  cat <<'MSG'
   Install the missing pieces, then re-run ./setup.sh:
     Ubuntu/Debian: sudo apt install texlive-latex-extra texlive-fonts-recommended \
                    texlive-science texlive-pictures latexmk poppler-utils fonts-texgyre
     macOS:         brew install --cask mactex-no-gui && brew install poppler
     Windows:       use WSL2 (Ubuntu) and follow the Ubuntu line above.
MSG
else
  echo "   LaTeX toolchain looks complete."
fi

echo "== Self-tests =="
python moderncomms-labs/tests/test_commlib.py
echo "== Done. Next: cd book && ./build.sh figs =="
