#!/bin/bash
# Build ONE chapter in an isolated temp copy (safe to run in parallel with other builds).
# Usage: ./build_chapter.sh ch08 [outdir]   -> writes <outdir>/ch08_only.pdf and prints errors/warnings.
# Cross-references to other chapters will show as ?? here; that's expected.
cd "$(dirname "$0")"
CH="$1"; OUT="${2:-$(pwd)/_chapbuild}"; mkdir -p "$OUT"
MK="$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64"; [ -d "$MK" ] && export PATH="$MK:$PATH"
W="$OUT/work_$CH"; rm -rf "$W"; mkdir -p "$W/chapters" "$W/frontback"
cp mdcstyle.sty "$W/"; cp chapters/$CH.tex "$W/chapters/"
ln -s "$(pwd)/figs" "$W/figs" 2>/dev/null || cp -r figs "$W/figs"
cat > "$W/only.tex" <<TEX
\documentclass[11pt,twoside,openright]{book}
\usepackage{mdcstyle}
\begin{document}
\hypertarget{toc}{}
\mainmatter\pagestyle{fancy}
$(case "$CH" in app*) printf "%s" "\appendix\setcounter{chapter}{$(( $(printf %d "'${CH#app}") - 65 ))}";; *) printf "%s" "\setcounter{chapter}{$((10#${CH#ch}-1))}";; esac)
\include{chapters/$CH}
\printindex
\end{document}
TEX
cd "$W"
for i in 1 2; do pdflatex -interaction=nonstopmode only.tex > build.log 2>&1; done
cp only.pdf "$OUT/${CH}_only.pdf" 2>/dev/null
echo "== errors"; grep -E "^!" -A2 build.log | head -40
echo "== overfull hboxes > 10pt"; grep -E "Overfull .hbox .([1-9][0-9]+)" build.log | head -20
echo "== missing figures"; grep -i "not found" build.log | head
echo "pages: $(python -c "import pymupdf;print(pymupdf.open('only.pdf').page_count)" 2>/dev/null)"
echo "pdf: $OUT/${CH}_only.pdf"
