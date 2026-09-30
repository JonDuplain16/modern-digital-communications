# Writing guide for chapter authors (human or agent)

Read `CLAUDE.md` (repo root) first — it has the mission, the chapter list with labels, and the
LaTeX conventions. This file adds the rules for working **in parallel** with other authors and
the quality bar.

## The quality bar
- Audience: advanced EE/CE students and practising engineers. Voice: the best comms engineer
  alive handing a lifetime of knowledge to an apprentice. Warm, confident, precise, concrete.
- **Length: 25–40 typeset pages per chapter** (roughly 1,600–2,600 lines of LaTeX including
  figures/boxes). Many subsections. Each chapter covers a big topic with many subtopics.
- Do more than state a formula: motivate it, derive it (or sketch the derivation honestly),
  give the intuition, show a number from a real system, say where it breaks.
- Every chapter mixes: derivations, **history** boxes (2–4), **inpractice** boxes (3–5) with real
  standards/products/numbers, **keyidea** (2–4), **pitfall** (2–3), **worked** examples (3–5 with
  real arithmetic), at least one **labbox** naming an exact lab file
  (`moderncomms-labs/labs/labNN_name.py`; if the right lab does not exist yet, name the one that
  *will* exist — e.g. `lab13_line_codes_eyes.py` — and list it in your final report).
- **Figures: 10–16 per chapter.** Data figures from `book/figscripts/chNN_figs.py`
  (`from figstyle import *`, `save(fig, "chNN_name")`, use `commlib` where it fits).
  TikZ block diagrams for architectures/transceivers/protocol stacks are encouraged (inline in
  the .tex; keep them in the navy/accent palette; wrap in `figure` with a label `fig:chNN_xxx`).
  Tables (booktabs) for standards parameters are great.
- End with `\problems` then an enumerate of 10–15 graded problems (some marked *(Simulation)*),
  then `\furtherreading` with 6–12 annotated references (real books/papers/standards only —
  **never invent citations, quotations, or numbers**; say "approximately" for estimates;
  cite standards by document number).
- Check what `\problems` / `\furtherreading` expand to in `mdcstyle.sty` and match earlier
  chapters' usage (look at the end of `chapters/ch05.tex`).

## Labels
- Chapter label exactly as in the CLAUDE.md table (other chapters already `\ref` them).
- Sections: `sec:chNN:short`, equations: `eq:chNN:short`, figures: `fig:chNN_name`
  (the `\mdcfig` macro makes that automatically), tables: `tab:chNN:short`.
- You may `\ref` other chapters by their `ch:` label freely (they resolve once written). Do not
  `\ref` figure/equation labels in chapters you have not read and verified exist.

## Parallel-work rules (important)
- Only create/edit **your** files: `book/chapters/chNN.tex`, `book/figscripts/chNN_figs.py`,
  `book/figs/chNN_*.pdf`. Do not edit other chapters, `main.tex`, `mdcstyle.sty`, the preface,
  or `CLAUDE.md`. If you need a style change, describe it in your final report instead.
- `commlib`: do **not** modify existing functions. If you need new reusable DSP code, add it in a
  **new** module named for your chapter topic (e.g. `commlib/linecodes.py`) and do not touch
  `commlib/__init__.py` (import your module explicitly: `from commlib import linecodes`).
  Otherwise put helpers inside your figure script.
- Build ONLY with `bash book/build_chapter.sh chNN` (isolated temp build, safe in parallel).
  Never run `book/build.sh` (the full build) — the coordinator does that.
- No git commands. The coordinator commits.
- Python is `python` (Windows, Python 3.12, numpy 2.x, scipy, matplotlib). Shell is Git Bash.
  Use the scratchpad or `book/_chapbuild/` for temporary files.

## Review loop (do it, don't skip)
1. `cd book/figscripts && python chNN_figs.py`
2. Write the chapter in several large chunks.
3. `bash book/build_chapter.sh chNN` → fix every `!` error, overfull boxes >10pt, missing figures.
4. Render pages and **look at them**:
   `python -c "import pymupdf;d=pymupdf.open('book/_chapbuild/chNN_only.pdf');[d[i].get_pixmap(dpi=50).save(f'book/_chapbuild/chNN_p{i+1:02d}.png') for i in range(d.page_count)]"`
   then view a sample of PNGs (Read tool) — check figures are legible, floats are sensible,
   boxes don't look broken, no pages of pure float.
5. Iterate until the chapter is something you would be proud to put your name on.

## Final report (keep it short)
Page count, figure list, lab files referenced (existing or to-be-created, with a one-line spec
for each new lab), any new commlib module, any style requests, known weaknesses.

## Reference chapter
`chapters/ch08.tex` is finished at the target quality/length (44 pp, 20 figures) — read it before
writing. Box titles: `\begin{worked}[Title]` (the `[title=Title]` form also works now).
Lab numbering already assigned: 13 line codes/eyes, 14 analog AM/FM, 15 superhet, 16 PCM,
17 multirate DSP, 18 satellite link, 19 information theory. Ask the coordinator (in your report)
for new numbers rather than inventing clashing ones; propose `labXX_topic.py` with a spec.
**Temp files:** always keep drafts/temp files in a folder unique to your chapter
(e.g. `<scratchpad>/chNN_work/`), never in the shared scratchpad root — parallel authors collided once.
