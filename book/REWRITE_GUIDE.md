# Second-edition rewrite guide (read this before touching a chapter)

Jon read the first edition and said: *too academic and hard to read.* He wants **an easy and
interesting read** for engineers — **keep all the same content**, but rewrite it so it is enjoyable:
**more real-world examples, stories, analogies, real images and figures. Every page should have an
image, figure or chart.** Not a math-and-theory information dump. Take your time; this is the
book's defining pass.

Also read `CLAUDE.md` (conventions, gotchas) and `WRITING_GUIDE.md` (parallel-work rules still apply).

## 1. The voice
- Write like the best engineer you ever worked with explaining it over coffee: warm, vivid,
  concrete, confident, occasionally funny, never condescending. Short paragraphs (3–6 lines).
  Plain words first, jargon second (define it the moment it appears).
- **Every section opens with a hook**, not a definition: a story, a puzzle, a real failure, a
  surprising number, a "have you ever wondered why your phone…". Then intuition, then a picture,
  then the math, then a real system, then "so what".
- **Intuition before equations.** Before any important equation, say in words what it will say
  and why it must be so. After it, read it back in words ("in other words: double the bandwidth and
  you double the capacity, but double the power and you gain only one bit").
- Prefer numbers people can feel: "a GPS signal arrives 1000× weaker than the noise — like hearing a
  whisper across a football stadium during a concert".
- Second person and "we" are fine. Rhetorical questions are fine. Lists of bullets are not a
  substitute for explanation.

## 2. Keep ALL the content
- Every topic, result, equation, worked example, table, problem and further-reading entry in the
  current chapter must survive (you may reorder, merge, re-explain, re-title).
- **Long derivations** move into `\begin{deeper}[Title] ... \end{deeper}` ("UNDER THE HOOD") boxes,
  so the main story flows and the full math is still there for those who want it. The main text
  keeps the result, its meaning and its consequences.
- **Keep every `\label` exactly** (other chapters reference them; grep `book/chapters/*.tex` for
  your chapter's labels before deleting anything). New labels follow `sec:chNN:x`, `eq:chNN:x`,
  `fig:chNN_x`, `tab:chNN:x`.
- Facts were fact-checked in the first edition — don't introduce new unverified numbers; when you
  add stories, use well-documented ones; hedge ("reportedly", "about") when unsure; never invent
  quotations, people or events.

## 3. Visuals: something on EVERY page
Target: `python book/tools/check_visuals.py book/_chapbuild/chNN_only.pdf` reports **≤ 5 %** of
pages without a visual (problems/further-reading pages are exempt automatically). Expect roughly
**one visual per page**, i.e. 40–70 per chapter. Mix them:
1. **Real photographs** (aim 6–12 per chapter where the subject allows: hardware, people,
   places, historic equipment, antennas, chips, satellites, test gear, spectrum analyzer screens).
   Use `python book/tools/fetch_image.py search "query"` then
   `python book/tools/fetch_image.py get "File:Name.jpg" chNN_short_name`. Only lines marked `OK`
   (PD/CC0/CC BY/CC BY-SA) can be downloaded; prefer images ≥ 800 px wide. Use with
   `\mdcphoto[width]{chNN_name}{Caption.}` (credit is appended automatically; label `fig:chNN_name`)
   or `\mdcsidephoto`. Never use non-free or unknown-license images. Check each downloaded photo
   with the Read tool — it must actually show what the caption says.
2. **Concept illustrations / analogy cartoons** (matplotlib or TikZ): e.g. a sieve for the matched
   filter, cars on a highway for multiplexing, a crowded party for SNR, a ball rolling into a valley
   for a PLL. Simple, clean, in the book palette (`figstyle.py`: NAVY, ACCENT, GREEN, ORANGE, PURPLE, GRAY).
3. **Data figures** from `book/figscripts/chNN_figs.py` (existing ones stay; add more), using
   `commlib` so they match the labs.
4. **Infographics**: "by the numbers" panels, timelines, comparison charts, block diagrams (TikZ),
   annotated screenshots of a lab (see §5).
5. Tables count as visuals when they are real tables (booktabs).
Layout helpers (all in `mdcstyle.sty`):
- `\mdcfig[w]{name}{cap}` normal float; `\mdcpair{a}{capA}{b}{capB}` two side by side;
- `\mdcside[w]{name}{cap}` / `\mdcsidephoto[w]{name}{cap}`: small figure wrapped by text on the
  right — use ONLY inside plain paragraphs with ≥ 12 lines of text after it (never next to lists,
  boxes, equations, section heads); make side figures single-panel and designed narrow
  (figsize ≈ 3.0 × 2.4 in in matplotlib) so text stays legible.
- Avoid float-only pages; place figures near the text that discusses them; every figure must be
  referenced in the text and have a caption that tells a reader what to look at.

## 4. Boxes (use generously, but every box must earn its place)
`history` (the story), `inpractice` (real systems, products, numbers), `analogy` (new: an everyday
analogy, with its limits stated), `keyidea`, `pitfall`, `worked` (real arithmetic), `tryit` (new:
a 2–4 line hands-on prompt, usually "open Lab NN, drag the X slider until Y — watch Z"),
`deeper` (new: UNDER THE HOOD derivations), `labbox` (end-of-chapter lab summary).
Aim per chapter: 4–6 history/story, 5–8 inpractice, 5–8 analogy, 4–6 tryit, plus the rest.

## 5. Labs are now live interactive apps
The Jupyter notebooks are being replaced by interactive desktop apps
(`moderncomms-labs/labs/labNN_name.py`, run with `python labNN_name.py`): sliders on the left, live
plots that update as you drag. In the text, `tryit` boxes should point to specific controls
("In Lab 3, set roll-off to 0 and drag the timing-offset slider…"). The labbox at the end lists the
lab file(s) and what each experiment shows. Lab numbers/files do not change (check
`moderncomms-labs/LAB_QUEUE.md`). If a lab experiment you want to reference does not exist yet,
describe it in your final report so the lab author can add it.

## 6. Mechanics
- Edit .tex with Write/Edit only (shell mangles backslashes). Write the new chapter in a few large
  chunks to a fresh file, then replace the old one; keep a copy of the original in your scratch folder.
- Build: `bash book/build_chapter.sh chNN`; zero errors, no overfull > 10 pt, no missing figures.
- Render and LOOK at pages (`python -c "import pymupdf; ..."` as in WRITING_GUIDE.md); fix ugly
  layouts, tiny text in side figures, orphaned headings, float pile-ups.
- Run `python book/tools/check_visuals.py book/_chapbuild/chNN_only.pdf` and iterate until ≤ 5 %.
- Run `python book/tools/gen_credits.py` after fetching photos (it must report no missing credits).
- Only edit your own files: `book/chapters/chNN.tex`, `book/figscripts/chNN_figs.py`,
  `book/figs/chNN_*`, `book/figs/photos/chNN_*` (each photo has its own `chNN_name.json` credit
  sidecar written by fetch_image.py — never hand-edit them).
- Final report: page count before/after, figure/photo counts, visual-coverage %, labels preserved,
  lab experiments you referenced that the lab author must provide, anything uncertain.

## 7. Lessons from the pilots (Ch 8, Ch 23)
1. fetch_image.py defaults to 1280 px (standard Wikimedia width); space searches ~10 s apart to avoid HTTP 429.
2. Backslashes are corrupted by Bash heredocs/sed in Python too ("\beta" -> backspace). Write scripts to files with the Write tool.
3. \mdcside/\mdcsidephoto are fragile: only inside long plain paragraphs (~16+ lines), never before a box/equation; if one misbehaves use \mdcfig[0.5] or \mdcpair.
4. Never use [b]-only floats. Small figures may use [H] to stop text-only pages.
5. Reach coverage by moving existing floats just before text-only stretches; re-run the checker after each move.
6. Locate floats by their \label when scripting moves, never by a shared prefix.
7. Narrow single-panel concept figures (~3.0 x 2.4 in) that carry a real simulated number give the best value per page.
8. Float parameters are now global in mdcstyle.sty (topfraction .8, bottomfraction .55, textfraction .12,
   floatpagefraction .75, 3 top / 2 bottom / 4 total). Do NOT set them per chapter.
9. Freeze content, then fix layout front to back: each fix only shifts later pages. Rebuild after each fix.
10. Prefer moving a figure a paragraph earlier over [H]; [H] leaves holes at page bottoms.
11. Long boxes cause text-only pages: put the figure BEFORE a full-page history/worked box, not after.
12. Wikimedia rate-limits parallel agents: fetch all photos in one early batch with long backoff, then view each.
13. A small page-map script (captions, headings, boxes and bottom gap per page) makes the coverage sweep fast.
14. Cheap, valuable visuals: simulation-driven analogy pictures; photos of front panels whose labelled knobs illustrate the text.
