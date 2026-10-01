# CLAUDE.md — Modern Digital Communications textbook + Python labs

This file is the handoff from the claude.ai chat where this project started. Read it fully
before doing anything. Keep it updated (the **Status** and **Session log** sections) at the end
of every working session so the next session can continue without re-discovery.

## 1. The mission (from Jon, the owner)

Build a **college-textbook-length** communications-engineering book as a **PDF**, plus
**expanded, polished Python simulation labs**. Jon's words, paraphrased faithfully:

- "A real educational tool, not a small guide on select DSP topics."
- Full PDF with **images, figures, and real-world applications**. Do more than state a formula
  and explain it. **Make it an interesting read.** Long and very informational.
- Cover **old communication systems and all their aspects**, **history**, **DSP**,
  **satellite communications**, real-world applications, "and anything else you can think of".
- **Each chapter covers a large key topic with multiple subtopics.**
- **Easily navigable**: table of contents, page numbers, clickable quick links (hyperref +
  PDF bookmarks are already set up).
- Labs: **expanded, clean-looking, easy to use, relevant**. **Focus on Python simulation**
  for now (GNU Radio/hardware examples are secondary).
- Voice: "Act like you are the best comms engineer in the world handing a lifetime of
  knowledge to an apprentice. This is your life's masterpiece." Iterate many times; keep
  building and improving. Resources are not a constraint (Claude Max, sole project).

Audience: professional/advanced EE and CE students and engineers. Jon has an Ettus USRP B200
and GNU Radio, but the current priority is Python simulation.

## 2. Repository layout

```
comms-textbook/
  CLAUDE.md                 <- this file
  setup.sh                  <- environment setup (Python venv + checks for LaTeX)
  book/
    main.tex                <- master file; \chap{chNN} includes a chapter only if its .tex exists
    mdcstyle.sty            <- ALL styling: fonts, colours, boxes, headers, TOC, hyperref
    frontback/              <- titlepage.tex, preface.tex (preface already describes all 25 chapters)
    chapters/chNN.tex       <- one file per chapter
    figscripts/chNN_figs.py <- one Python script per chapter; writes vector PDFs into book/figs/
    figscripts/figstyle.py  <- shared matplotlib style (Palatino-like serif, palette, W1/W2 widths, save())
    figs/                   <- generated figures (chNN_name.pdf)
    build.sh                <- `./build.sh` = 3x pdflatex + makeindex; `./build.sh figs` regenerates all figures first
    main.pdf                <- last successful build (106 pages at handoff)
  moderncomms-labs/
    commlib/                <- shared DSP/comms library used by BOTH the labs and the book figures
    labs/labNN_*.py         <- lab sources (jupytext "percent" format) -> labNN_*.ipynb
    gnuradio/               <- 5 GNU Radio 3.10 flowgraphs (all have --sim); secondary priority
    tests/test_commlib.py   <- 12 library self-tests (all pass)
    tests/build_notebooks.py <- converts + executes labs in-process, embeds figures
    tests/dump_figs.py      <- stitches a notebook's figures into one PNG for visual review
```

`book/figscripts/figstyle.py` imports `commlib` via a relative path
(`../../moderncomms-labs`), so keep the two folders side by side.

## 3. Status (1 Oct 2026) — first complete edition

### Book — COMPLETE: 25 chapters + 2 appendices + bibliography + index (~1,300 pages)

| Ch | Label | Title | Status |
|----|-------|-------|--------|
| 1 | ch:history | The Story of Telecommunication | done (~49 pp), fact-checked |
| 2 | ch:signals | Signals, Spectra and Systems | done (~49 pp), fact-checked |
| 3 | ch:noise | Random Signals and Noise | done (~50 pp), fact-checked |
| 4 | ch:analog | Analog Modulation and the Classic Radio | done (~53 pp), fact-checked |
| 5 | ch:pcm | Sampling, Quantization and Digital Telephony | done (~49 pp), fact-checked |
| 6 | ch:dsp | Digital Filters and Multirate Processing | done (~52 pp), fact-checked |
| 7 | ch:sdr | The Radio Transceiver and the Software-Defined Radio | done (~46 pp), fact-checked |
| 8 | ch:baseband | Baseband Transmission and Pulse Shaping | done (44 pp), fact-checked |
| 9 | ch:modulation | Digital Modulation and Optimal Detection | done (40 pp), fact-checked |
| 10 | ch:sync | Synchronization | done (43 pp), fact-checked |
| 11 | ch:channels | The Wireless Channel | done (45 pp), fact-checked |
| 12 | ch:equalization | Equalization | done (41 pp), fact-checked |
| 13 | ch:infotheory | Information Theory | done (46 pp), fact-checked |
| 14 | ch:classiccodes | Classical Error-Control Codes | done (50 pp), fact-checked |
| 15 | ch:moderncodes | Turbo, LDPC and Polar Codes | done (47 pp), fact-checked |
| 16 | ch:sourcecoding | Source Coding: Voice, Audio, Images and Video | done (54 pp), fact-checked |
| 17 | ch:ofdm | OFDM and Multicarrier Transmission | done (45 pp), fact-checked |
| 18 | ch:spreadspectrum | Spread Spectrum, CDMA and Satellite Navigation | done (49 pp), fact-checked |
| 19 | ch:mimo | MIMO and Antenna Arrays | done (45 pp), fact-checked |
| 20 | ch:multipleaccess | Multiple Access and the Cellular Concept | done (47 pp), fact-checked |
| 21 | ch:cellular | Cellular Generations: From AMPS to 5G | done (57 pp), fact-checked |
| 22 | ch:wifi | Wi-Fi, Bluetooth and the Internet of Things | done (57 pp), fact-checked |
| 23 | ch:satellite | Satellite Communications | done (45 pp), fact-checked |
| 24 | ch:wireline | Wireline and Optical Communications | done (49 pp), fact-checked |
| 25 | ch:sixg | The Road to 6G | done (46 pp), fact-checked |
| A | appA | Mathematical Reference | done (~32 pp) |
| B | appB | Using the Companion Labs | done (~24 pp) |
| — | references | References (387 entries, generated by book/tools/bib/gen.py) | done |

Every chapter: chapterintro, history/inpractice/keyidea/pitfall/worked boxes, labbox naming exact
lab files, 15–18 problems, annotated Further Reading. Fact-check passes used web search where
available; residual "approximately"/"reportedly" hedges mark items not verified against primary sources.

**Possible future work** (none required): trim the longest chapters (21, 22, 16 are 54–57 pp) if a
shorter print edition is wanted; re-verify hedged standards numbers as 3GPP/IEEE documents evolve
(6G timeline in Ch 25 is stated as of Sept 2026); add GNU Radio flowgraphs for more chapters.

### Labs — COMPLETE: 37 notebooks (lab00 index + labs 01–36), all execute
`moderncomms-labs/LAB_QUEUE.md` lists every lab with chapter and spec (all done). Uniform template
(commlib/labkit.py): header, objectives, roadmap, numbered sections, widgets, "Try it yourself"
self-checks, key takeaways, hardware pointers. commlib modules (27+): filters, modulation, channel,
sync, equalize, eqadv, ofdm, ofdmadv, coding, turbo, gf, blockcodes, infotheory, cpm, mimo,
satellite, spread, linecodes, propagation, sourcecoding, cellular, iot, wireline, sixg, ltephy, rf,
labkit, plotting, iq. `python book/tools/check_labs.py` verifies every lab named in the book exists.

## 4. Conventions (follow them — consistency matters in a 700-page book)

### LaTeX (`mdcstyle.sty`)
- Chapter file starts: `\chapter{Title}` then `\label{ch:xxx}` then a `chapterintro` box.
- Boxes: `history`, `inpractice`, `keyidea`, `pitfall`, `worked`, `labbox` — each takes an
  optional `[title]`. Use them generously but purposefully (roughly 2–4 history, 3–5 in-practice,
  2–4 worked examples per chapter).
- Figures: `\mdcfig[width]{chNN_name}{caption}` → label `fig:chNN_name`. Refer with
  `Figure~\ref{fig:chNN_name}`. TikZ block diagrams are welcome for architectures.
- Index terms: `\term{...}` (bold + index entry). Macros: `\E \R \C \Q \ebno \esno \dB \sinc \re \im \vect{}`.
- Chapter ends with `\problems` (8–15 graded problems, some marked *(Simulation)*) and
  `\furtherreading` (annotated references).
- Numbers from standards: cite the document (e.g. 3GPP TS 38.211, ITU-R P.618, ETSI EN 302 307 for DVB-S2).
  Say "approximately" when a value is an engineering estimate. Do not invent quotations.

### Figures
- One script per chapter: `book/figscripts/chNN_figs.py`, `from figstyle import *`,
  functions named after figures, `save(fig, "chNN_name")`. Use `commlib` so figures and labs agree.
- Widths: `W1`/`W2` inches; keep fonts consistent (figstyle handles it).

### Build & review loop (do this every chapter)
1. `cd book/figscripts && python3 chNN_figs.py`
2. write/extend `book/chapters/chNN.tex`
3. `cd book && ./build.sh` → check the `!` error summary, page count, undefined refs
4. **Look at the output**: `pdftoppm -f A -l B -r 60 -png main.pdf /tmp/pg` and view the PNGs;
   fix overfull boxes, bad float placement, unreadable figures.
5. Update the Status table above.

### Labs
- Edit `labs/labNN_*.py` (jupytext percent format), then
  `python tests/build_notebooks.py labNN` to regenerate the executed notebook;
  `python tests/dump_figs.py labNN_name /tmp/x.png` to eyeball figures.
- Run `python tests/test_commlib.py` after any change to `commlib`.

## 4b. Windows workstation notes (Jon's PC, from 2026-09-30)
- LaTeX = MiKTeX 25.12, user install at `%LOCALAPPDATA%\Programs\MiKTeX` (auto-install of missing
  packages is on). `build.sh` adds it to PATH automatically. Python = `python` (3.12, numpy 2.x).
  PyMuPDF replaces poppler: `python book/render.py FIRST LAST [dpi] [outdir]` renders pages to PNG.
- `book/build_chapter.sh chNN` builds ONE chapter in an isolated temp dir (`book/_chapbuild/`),
  safe to run in parallel. `book/WRITING_GUIDE.md` = rules + quality bar for (parallel) chapter authors.
- Git: private repo https://github.com/JonDuplain16/modern-digital-communications (branch main).
  Commit + push after every completed chapter/lab batch.

## 5. Known issues / gotchas
- **Backslashes in this Windows Git-Bash tool get mangled** (heredocs collapse `\\`, `sed` turns
  `\r`/`\n` in replacements into control chars). Edit .tex with the Edit/Write tools, or write a
  Python script to a file with Write and run it.
- `\vect` is `\mathbf`, which breaks on lowercase Greek with mathpazo — use `\bm{\mu}` there.
- **GNU Radio + NumPy 2:** distro GNU Radio packages are built against NumPy 1.x; a pip NumPy 2
  in the same Python breaks `from gnuradio import gr`. Use radioconda, or run flowgraphs with
  `PYTHONPATH=/usr/lib/python3/dist-packages`. The labs themselves work with NumPy 1.24+ or 2.x.
- The LDPC PEG construction in `commlib/coding.py` was fixed (no 4-cycles; regression test).
- Slow figure scripts cache Monte Carlo results in `book/figscripts/cache/` (ch12, ch15) and
  `book/figscripts/_ch14_cache.npz`; delete a cache to force a full re-simulation (ch15 ≈ 50 min).
- Monte Carlo curves in lab 9 floor at the simulation resolution (1e-3 FER); increase frame counts
  when running on a fast machine.
- `figstyle.py` looks for TeX Gyre Pagella fonts under `/usr/share/texmf/...`; on macOS it falls
  back to other serif fonts — adjust the glob if you want exact matching.
- A separate, older short-form version of the text lives in a Claude Doc from the chat; this LaTeX
  book supersedes it.

## 6. Original first-session plan (completed)
1. Run `./setup.sh`, then `cd book && ./build.sh figs` to confirm the environment reproduces the PDF.
2. Write Chapter 8 from the outline above (figures exist). Build, render, review.
3. Continue Chapters 9–12 (Part III), then Satellite (23), then the rest. After each chapter,
   add or upgrade the matching lab.
4. Consider working in parallel: one pass/agent per chapter (figures + text), then a review
   pass per chapter for accuracy, length and readability.

## 7. Session log
- 2026-09-30 (claude.ai chat): built commlib, 12 labs, 5 GNU Radio flowgraphs, a short Claude Doc
  version, then the LaTeX book framework and Chapters 1–7 (+ Chapter 8 figures). Handed off to Claude Code.
- 2026-09-30 (Claude Code, Windows): installed MiKTeX + PyMuPDF, made build Windows-friendly, created
  GitHub repo, added build_chapter.sh + WRITING_GUIDE.md. Wave 1 in parallel agents: Ch 8, 9, 10, 11,
  12, 23 + labs engineer (labkit.py, polish labs 01-12, fix chapter mapping, new labs 13-18, lab00 index).
  Done+pushed: Ch 8-13, 23 (full book ~420 pp). Still in flight when usage limit hit: Ch 14, 15, 16, 17, 18
  and the labs engineer -- check their files exist/build (build_chapter.sh) before re-launching.
  Requested future labs are in moderncomms-labs/LAB_QUEUE.md. Remaining chapters: 19-22, 24, 25, App A/B, references;
  then deepen Ch 1-7; then an accuracy-review pass (agents flagged numbers to verify in their reports).
- Later same day: ALL 25 chapters + App A/B written (full build ~1140+ pp). Ch 1-4 deepened (49-53 pp each).
  Labs 00-22, 24, 25, 28 done. In flight at usage limit: deepen Ch 5, 6, 7; labs wave 3 (23,26,27,29-35);
  fact-check reviewers for Ch 8-16 and Ch 17-25+apps. Check their files build before relaunching.
  Still TODO: bibliography (book/chapters/references.tex, compile from all Further Reading), final full-build
  layout review, update Status table rows for ch14-25 (all written, 41-57 pp).
- 2026-10-01 (Claude Code): finished everything — Ch 5-7 deepened, labs 23/26/27/29-36, fact-check
  passes over all chapters (Ch 1-7, 8-16, 17-25+apps), 387-entry bibliography, top-level README,
  final full build + figure regeneration check. First complete edition pushed.
