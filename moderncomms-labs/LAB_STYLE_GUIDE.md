# Lab Style Guide — writing a studio lab

Every lab is now a **live, interactive desktop script**: sliders on the left, plots that
change instantly in the middle, a short friendly explanation on the right. No Jupyter, no
"run all cells". Lab 03 (`labs/lab03_pulse_shaping.py`) and Lab 14
(`labs/lab14_analog_am_fm.py`) are the reference implementations; read one before you start.
The API is documented in `studio/README.md`.

The audience is engineers, but the goal is **enjoyment**: the user should *discover* the idea
by moving a slider and seeing something happen, then read two short paragraphs that tell
them what they just saw. Not a math dump.

## 1. The workflow

1. Read the matching book chapter (and the lab's current version, if any). List the 4–8
   ideas worth *playing with*. Drop anything that is only a static table.
2. Write or edit `labs/labNN_name.py` as a studio lab (all 36 labs are studio apps; the
   Jupyter notebook versions are retired).
3. Put any reusable DSP in `commlib` (new module if needed, e.g. `commlib/analog.py`), with a
   test in `tests/test_commlib.py`. Lab-specific glue may live in the lab file. `studio`
   itself contains no DSP.
4. `python labs/labNN_name.py --selftest` until it passes, then **open every PNG** in
   `tests/screens/labNN/` (both `NN_slug.png` and `NN_slug_random.png`) and fix what looks
   wrong. Also run `--selftest --dark --out tests/screens/labNN_dark` once.
5. Launch it for real (`python labs/labNN_name.py --smoke 10`) and play with every control.
6. Calibrate every challenge numerically (a tiny script that sweeps the parameter) so each is
   achievable, not already true at the defaults, and not trivially true.
7. Add/update the entry in `studio/catalog.py` (title and one-line description).

## 2. File template

```python
"""Lab NN · Title   (Chapter C)

Run it:      python labs/labNN_name.py
Self-test:   python labs/labNN_name.py --selftest

Two or three sentences: why this matters and what the experiments are.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import numpy as np
import commlib as cl
import studio as st
from studio import (Experiment, Slider, LogSlider, Choice, Toggle, Button, Heading, Plot,
                    SpectrumPlot, Readout, Challenge, NAVY, RED, GREEN, GRAY)
from studio import v, keybox, good, bad


# ============================================================ shared helpers (lab-specific)
def helper(...):
    ...


# ============================================================ 1. first idea
class FirstIdea(Experiment):
    title = "Short noun phrase"                 # ≤ 32 characters (sidebar width)
    blurb = "One sentence hook, ≤ 90 characters."
    book = "sec:chCC:label"                     # the section that explains it
    controls = [
        Slider("x", "Plain-language name", lo, hi, default, step=..., unit="dB",
               help="Tooltip: what this knob physically is"),
    ]
    plots = [Plot("a", "Title that says what is plotted", x="quantity (unit)", y="quantity (unit)")]
    readouts = [Readout("k", "Name", "unit", ".1f", good=lambda x: x > 30)]
    challenges = [Challenge("Imperative goal with a number.", lambda s: ..., hint="…")]

    def update(self, p):
        ...compute with commlib...
        self.plot("a").line("curve", x, y, color=NAVY, name="…")
        self.readout(k=value)

    def story(self, p):
        return ("<h3>Three-to-six-word heading</h3>"
                f"<p>Sentence tied to {v(p.x, '.1f', 'dB')} …</p>"
                + keybox("The one idea to remember."))


LAB = st.Lab(NN, "Title", chapter=C, chapter_title="Book chapter title",
             experiments=[FirstIdea, ...])

if __name__ == "__main__":
    st.run(LAB)
```

## 3. Naming and structure

* File: `labs/labNN_short_name.py` (unchanged from the notebook era). Lab number = `NN`.
* Experiment classes: CamelCase nouns (`LiveEye`, `FMThreshold`). Order them as a story:
  first the simplest picture, then the physics, then the system, then the "wow".
* Control keys are short identifiers (`beta`, `esn0`, `rc`); labels are words a newcomer
  understands ("Roll-off β", "Detector time constant RC"). Units go in `unit=`, never in the label.
* Plot keys are short; plot titles describe what is shown ("Eye opening vs sampling phase").
  Axis labels always carry units in parentheses: `"time (ms)"`, `"power (dB)"`.
* Group controls with `Heading("Transmitter")`, `Heading("Channel")`, `Heading("Receiver")`.
* Do not name your own attributes after framework ones (`frame`, `playing`, `dragging`,
  `quick`, `rng`, `p`, `r`, `plot`, `readout`, … see studio/README.md). Class-level clashes are
  caught when the `Lab` is built, with a message naming the culprit. Use `make_frame`, not
  `frame`.
* Text input (a message, a bit pattern, a polynomial) uses `Text(...)`, never a local
  `QLineEdit` control: `Text("msg", "Message", "SOS", examples=[…], upper=True)` or, for bits,
  `Text("bits", "Bit pattern", "0110", allowed="01", live=False, mono=True)`.
* Choices may hold numbers: `Choice("M", "Order", [4, 16, 64], labels=["QPSK", "16-QAM",
  "64-QAM"])` gives `p.M` as an int; no parsing of label strings.
* Labels can be long: they wrap (captions, toggles) or elide with a tooltip (buttons,
  segments). Still prefer ≤ 30 characters; put detail in `help=`.

## 4. Writing the story ("What's going on")

* 2–4 short paragraphs, then one `keybox(...)` with the single idea to keep.
* **Tie it to the current state.** Use the parameters and readouts:
  "Your roll-off of **0.35** means the signal occupies **1.35×** the symbol rate."
  Branch on regimes ("Below threshold…" / "Above threshold…") and say what the user should
  notice in which plot ("the red stems", "top left").
* Plain language first, then the name of the idea in bold. Analogies welcome. One formula at
  most, in unicode (`3β²(β+1)·CNR`, `RC ≤ √(1 − μ²)/(2π f_m μ)`), never LaTeX.
* Helpers: `v(x, ".2f", "dB")` highlights a value, `good("…")`/`bad("…")` colour a verdict,
  `keybox("…")` makes the shaded key-idea box. Available CSS classes: `v`, `good`, `bad`,
  `muted`, `eq`.
* History or real-world hooks in one sentence ("This is why 400G Ethernet moved to PAM-4").
  Cite real numbers only when you are sure; otherwise say "about".

## 5. Challenges ("Try this")

* 2–4 per experiment. Imperative, specific, with a number: "Keep the eye at least 50 % open
  with Es/N0 of 19 dB or less."
* Each must be **false at the defaults**, achievable within the slider ranges, and require
  understanding (find a sweet spot, a threshold, a trade-off), not just "move slider to max".
* The check reads `s.p` (parameters) and `s.r` (readouts). Use tolerances for exact targets
  (`abs(s.p.beta - 0.25) < 0.006`). Exceptions count as "not yet".
* Add a `hint=` (shown as a tooltip) for anything non-obvious.
* Calibrate with a short sweep script (import the lab module, call the experiment's helpers
  with a `studio.Params(...)` and print the readout values over a grid of settings).

## 6. Readouts

* 2–4 per experiment, the numbers an engineer would put on a slide: BER, SNR, EVM,
  bandwidth, eye height, separation… Units in `unit=`.
* `good=lambda x: …` turns the number green/red. Use it when there is a clear target.
* Strings are fine for states ("overmodulated", "PAM-4", "—").
* A readout that does not apply in the current mode can simply be `float("nan")`: it shows
  "—" and is not a self-test failure. Use `floor=` for Monte Carlo results that cannot be
  resolved below some value ("< 1.0×10⁻⁵").

## 7. Plots and colours

* 1–4 plots per experiment. Use `layout` (rows of keys, repeat a key to span) and
  `row_stretch`/`col_stretch` so the main plot is the biggest.
* Colours: the book palette only — `NAVY` main signal/theory, `RED` measured/sampled/the thing
  to look at, `GREEN` ideal/target/good region, `ORANGE` secondary, `PURPLE` markers, `GRAY`
  references and copies. They map automatically to readable twins in dark mode.
* Fix axis ranges (`xlim`, `ylim`) whenever the data range is known, so plots do not jump while
  the user drags a slider. Autoscale only when the range genuinely varies by decades.
* Legends must not cover data: leave headroom in `ylim` and use `legend="tl", legend_cols=2/3`
  for a horizontal legend, or `legend=None` if the title says it all.
* Use `vline/hline/band` with labels for the things the story talks about (Nyquist frequency,
  threshold, best sampling phase).
* Eye diagrams: `EyePlot` + `plot.eye(...)` (persistence image), not hundreds of lines.
* Legends for bars, cell colours or images: give `bars(..., name=...)` (coloured swatch) or add
  `plot.legend_swatch("label", COLOR)` entries. Bars work on log axes.
* Shaded regions: `fill_between` (or `line(..., fill=baseline)`); both are safe for long
  stretches of zero area.
* Diagrams (trellises, shift registers, Venn diagrams, grids of bits): `Canvas(...)` or
  `Plot(..., axes=False)`; make them interactive with `on_click(self, key, x, y)`. Draggable
  points/bars: `plot.handles(...)` + `on_drag(self, key, item, i, x, y)` (see Lab 19,
  water-filling). Parametric curves (circles, trajectories) longer than 4000 points:
  `line(..., downsample=False)`.
* With `aspect=True`, or to change both ranges, use `set_range(xlim, ylim)`.

## 8. Performance rules

* `update` should take < 50 ms (the self-test fails at 0.5 s). Budget ~20 k–50 k samples per
  update. Use `scipy.signal.fftconvolve`, cache filter taps (`functools.lru_cache`), and
  vectorise loops (see `commlib.analog.envelope_detector`).
* Anything slower (Monte Carlo BER, sweeps) goes in `background(p)` as a generator that
  yields partial results; draw them in `progress(p, item)`. Keep the run short when
  `self.quick` is True (self-test). Never touch plots inside `background`.
* Streaming/"oscilloscope" experiments: `animate = True` (optionally `autoplay = True`),
  override `tick(p)` to accumulate persistence (`accumulate=True`), keep a frame under ~60 ms.
  A `tick` must redraw every item it wants visible (untouched items are hidden). To show a value
  the animation chose on a slider, use `self.set_control(key, v, refresh=False)` (no flicker).
  A "Run"/"Start" button can start the animation: `Button("go", "▶  Run", starts_play=True)`.
  Implement `on_reset(self, p)` if the experiment keeps state (trained weights, a running
  simulation clock) that the window's Reset should clear.
* Measurements on periodic test tones: use an integer number of periods or fit a DC term;
  FFT brick-wall filters on non-periodic records create artificial distortion floors.

## 9. Audio (optional)

`self.play_audio(x, fs)` plays mono (N) or stereo (N×2) audio, resampled to 48 kHz, through
whatever backend exists (sounddevice, winsound, afplay, aplay). Always synthesise the sound;
never download. Offer it with `Button("listen", "▶  Listen to …", primary=True)` and
`def on_listen(self, p)`. The self-test mutes audio.

## 10. Running and inspecting the self-test

```
python labs/labNN_name.py --selftest                 # this lab
python labs/labNN_name.py --selftest --dark --out tests/screens/labNN_dark
python tests/selftest_labs.py                        # every studio lab
python tests/test_commlib.py                         # after any commlib change
```

The self-test builds each experiment, runs it at the defaults (including the background job
in quick mode and a few animation frames), then at `--random` random control settings, and
fails on exceptions, infinite readouts/plot data, or slow updates (NaN readouts and all-NaN
traces are allowed). It saves a window
screenshot per experiment. Look at all of them with an image viewer: clipped labels,
legends over data, empty plots, absurd numbers (BER 10⁻¹³¹) and walls of text are bugs.

## 11. Checklist before you call a lab done

- [ ] 4–8 experiments, each with a clear "aha", ordered as a story
- [ ] Every control has a plain-language label, unit and sensible range/default
- [ ] Story tied to the live values, ≤ 4 short paragraphs + key box, book link set
- [ ] 2–4 calibrated challenges per experiment, none true at defaults
- [ ] Readouts with units; `good=` where there is a target
- [ ] Self-test passes (light and dark); every screenshot reviewed
- [ ] Real window smoke-tested; updates feel instant; Play mode smooth
- [ ] `studio/catalog.py` entry updated; reusable DSP in `commlib` with a test
