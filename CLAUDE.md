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

## 3. Status at handoff (30 Sep 2026)

### Book — 25 chapters planned, 7 written (~106 pages)

| Ch | Label | Title | Status |
|----|-------|-------|--------|
| 1 | ch:history | The Story of Telecommunication | written (~16 pp) — expand |
| 2 | ch:signals | Signals, Spectra and Systems | written (~12 pp) — expand |
| 3 | ch:noise | Random Signals and Noise | written (~10 pp) — expand |
| 4 | ch:analog | Analog Modulation and the Classic Radio (AM, SSB, FM, stereo, superhet) | written (~16 pp) — expand |
| 5 | ch:pcm | Sampling, Quantization and Digital Telephony (PCM, companding, T1, sigma-delta) | written (~14 pp) |
| 6 | ch:dsp | Digital Filters and Multirate Processing (FIR/IIR, polyphase, CIC, NCO) | written (~12 pp) |
| 7 | ch:sdr | The Radio Transceiver and the Software-Defined Radio | written (~10 pp) |
| 8 | ch:baseband | Baseband Transmission and Pulse Shaping | written (44 pp) |
| 9 | ch:modulation | Digital Modulation and Optimal Detection (PSK/QAM/FSK/MSK/GMSK, noncoherent) | written (40 pp) |
| 10 | ch:sync | Synchronization (carrier, timing, frame; PLL theory) | written (43 pp) |
| 11 | ch:channels | The Wireless Channel (propagation, link budgets, fading, 3GPP models) | written (47 pp) |
| 12 | ch:equalization | Equalization (ZF/MMSE/LMS/RLS/CMA/DFE/MLSE, turbo equalization) | written (41 pp) |
| 13 | ch:infotheory | Information Theory | written (47 pp) |
| 14 | ch:classiccodes | Classical Codes (Hamming, cyclic/CRC, BCH, Reed-Solomon, convolutional, Viterbi) | todo |
| 15 | ch:moderncodes | Turbo, LDPC and Polar Codes | todo |
| 16 | ch:sourcecoding | Source Coding: voice, audio, image, video | todo |
| 17 | ch:ofdm | OFDM and Multicarrier | todo |
| 18 | ch:spreadspectrum | Spread Spectrum, CDMA and GNSS/GPS | todo |
| 19 | ch:mimo | MIMO and Antenna Arrays | todo |
| 20 | ch:multipleaccess | Multiple Access and the Cellular Concept | todo |
| 21 | ch:cellular | Cellular Generations: AMPS to 5G | todo |
| 22 | ch:wifi | Wi-Fi, Bluetooth and IoT (LoRa, Zigbee, NB-IoT) | todo |
| 23 | ch:satellite | Satellite Communications (orbits, link budgets, transponders, DVB-S2, Telstar to Starlink, NTN) | written (45 pp) |
| 24 | ch:wireline | Wireline and Optical (telephone plant, DSL, cable/DOCSIS, Ethernet SerDes, fibre, coherent optics) | todo |
| 25 | ch:sixg | The Road to 6G (ISAC, OTFS/AFDM, AI-native PHY, NTN) | todo |
| A,B | appA, appB | Math reference; Using the labs and GNU Radio (preface promises Appendix B) | todo |
| — | references | Bibliography | todo |

Existing chapters already `\ref` the planned labels above (55 forward references resolve
automatically once those chapters exist — keep the labels exactly).

**Length target:** existing chapters are 10–16 pages; Jon wants textbook length. Aim for
**25–40 pages per chapter**, then go back and deepen Chapters 1–7 to the same standard.
Target total: 700+ pages.

### Planned outline for Chapter 8 (figures already generated)
1. From bits to waveforms: the PAM model, symbol vs bit rate
2. Line codes: NRZ, RZ, Manchester, AMI/B8ZS/HDB3, 4B5B, 8b/10b, 64b/66b, 128b/130b, scramblers (fig ch08_linecodes)
3. Power spectral density of PAM signals (spectral lines from nonzero mean)
4. ISI and the Nyquist criterion; raised cosine and RRC (figs ch08_isi, ch08_raised_cosine)
5. The matched filter: Cauchy–Schwarz derivation, correlator equivalence (fig ch08_matched_filter)
6. Binary detection in AWGN: antipodal vs orthogonal vs on-off (fig ch08_ber_binary)
7. Eye diagrams: anatomy, jitter, bathtub curves, SerDes compliance masks (figs ch08_eyes, ch08_eye_anatomy, ch08_timing_sensitivity)
8. Multilevel PAM: PAM-4 in 400G Ethernet / PCIe 6.0 / GDDR (fig ch08_pam4)
9. Partial response: duobinary, precoding, PRML in disk drives (fig ch08_duobinary)
10. Faster-than-Nyquist signalling (brief); summary, problems, further reading

### Labs — 12 exist, need expansion and polish
Labs 01–12 in `moderncomms-labs/labs/` all execute cleanly (topics: baseband/IQ, modulation,
pulse shaping, synchronization, channels, equalization, OFDM, convolutional codes, LDPC/polar,
MIMO, air interfaces, frontiers). Jon wants them **expanded, cleaner and easier to use**.
Ideas: a consistent header/objectives/"what you will see" block; a small shared widget/plot
helper for a uniform look; more labs to match new chapters (analog AM/FM demodulation,
superhet receiver sim, PCM/companding, multirate DSP, line codes/eye diagrams, spread
spectrum/GPS acquisition, satellite link budget calculator, source coding, cellular system
simulator); a lab index notebook; each chapter's `labbox` should name the exact lab file.

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
- The LDPC PEG construction in `commlib/coding.py` is simplified and leaves a few 4-cycles
  (lab 9 prints the count) — a good improvement task.
- Monte Carlo curves in lab 9 floor at the simulation resolution (1e-3 FER); increase frame counts
  when running on a fast machine.
- `figstyle.py` looks for TeX Gyre Pagella fonts under `/usr/share/texmf/...`; on macOS it falls
  back to other serif fonts — adjust the glob if you want exact matching.
- A separate, older short-form version of the text lives in a Claude Doc from the chat; this LaTeX
  book supersedes it.

## 6. Suggested first session in Claude Code
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
