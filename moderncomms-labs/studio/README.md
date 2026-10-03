# studio — the interactive lab framework

`studio` turns a short, declarative Python file into a polished desktop lab: a sidebar of
experiments and controls, a grid of live plots, a strip of big live numbers, a "What's going
on" panel, self-ticking "Try this" challenges and a link into the book. It is built on
PySide6 + pyqtgraph and uses `commlib` for every piece of signal processing.

```
python labs/launcher.py                 # gallery of all labs (also: python -m studio)
python labs/lab03_pulse_shaping.py      # one lab
python labs/lab03_pulse_shaping.py --exp 3 --dark
python labs/lab03_pulse_shaping.py --selftest       # headless test + screenshots
python tests/selftest_labs.py                       # all studio labs
```

## Architecture

```
studio/
  __init__.py   public API (everything a lab imports) + story helpers v(), eq(), keybox(), good(), bad()
  core.py       Lab, Experiment, Readout, Challenge, Params, fmt_value/sci/slug, reserved-name check
  controls.py   Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading, Text (+ their Qt widgets)
  plots.py      Plot and presets: TimePlot, SpectrumPlot, ConstellationPlot, EyePlot, BERPlot,
                PolarPlot, ImagePlot, BarPlot, Canvas — keyed, update-in-place items
  app.py        MainWindow: layout, debounce, worker threads, animation, theme, export, CLI (run)
  selftest.py   --selftest: defaults + random settings, exception/inf/timing checks, screenshots
  theme.py      book palette, light/dark themes, Qt style sheet, story CSS
  book.py       resolves LaTeX labels ("sec:ch08:rc") to "§8.5 … (p. 376)" from book/*.aux and
                opens the PDF at that page
  audio.py      optional playback (sounddevice → winsound → afplay/aplay; never required)
  catalog.py    the list of all labs for the launcher, grouped by book part
  launcher.py   the gallery window
```

The update cycle: a control changes → 30 ms debounce → `Experiment.update(p)` runs on the GUI
thread → plot items it touched are updated in place, items it did not touch are hidden →
readouts, story and challenges refresh → if the experiment defines `background(p)`, that
generator starts in a worker thread and each `yield` is handed to `progress(p, item)` on the
GUI thread (a newer control change cancels it). While **Play** is on, a timer calls
`tick(p)` (default: `update(p)` again, i.e. fresh noise) at `fps`. **`tick` follows the same
rule as `update`: items it does not touch are hidden**, so a custom `tick` must redraw every
item it wants to keep (or call `self.update(p)`). Story and challenges are re-checked every
third frame and again when Play is paused.

## API cheat-sheet

```python
import _path                                   # labs/_path.py: puts commlib + studio on sys.path
import numpy as np
import commlib as cl
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading, Text,
                    Plot, Canvas, SpectrumPlot, EyePlot, ConstellationPlot, BERPlot, PolarPlot,
                    ImagePlot, BarPlot, Readout, Challenge, NAVY, RED, GREEN, ORANGE, GRAY)
from studio import v, keybox, good, bad

class Example(Experiment):
    title = "Pulses and their spectra"            # sidebar + page title
    blurb = "One line under the title."
    book = "sec:ch08:rc"                           # LaTeX label of the matching section
    controls = [Heading("Transmitter"),
                Choice("pulse", "Pulse", ["RC", "RRC"]),                 # p.pulse
                Choice("M", "Order", [4, 16, 64], labels=["QPSK", "16-QAM", "64-QAM"]),  # p.M is an int
                Slider("beta", "Roll-off β", 0, 1, 0.35, step=0.01),     # p.beta
                LogSlider("bw", "Bandwidth", 0.1, 10, 1, unit="× Rs",
                          enabled_if=lambda p: p.pulse == "RC"),
                IntSlider("span", "Length", 2, 32, 8, step=2, unit="symbols"),
                Toggle("noise", "Add noise", True),
                Text("msg", "Message", "SOS", examples=["SOS", "PARIS"], upper=True),   # p.msg
                Button("again", "New data"),                             # calls on_again(p)
                Button("go", "▶  Run", starts_play=True)]                # on_go(p), then Play
    plots = [Plot("t", "Pulse", x="time (T)", y="amplitude", xlim=(-6, 6), legend="tr"),
             SpectrumPlot("f", "Spectrum", x="f / Rs", ylim=(-80, 5))]
    layout = [["t", "f"]]                          # rows of plot keys; repeat a key to span
    row_stretch, col_stretch = None, [1, 1]
    readouts = [Readout("bw", "Bandwidth", "× Rs", ".2f", good=lambda x: x < 1.5),
                Readout("ber", "BER", "", "sci", floor=1e-6)]            # "< 1.0×10⁻⁶" below
    challenges = [Challenge("Use exactly 25 % excess bandwidth",
                            lambda s: abs(s.p.beta - 0.25) < 0.006, hint="…")]
    animate = False    # True: Play/Pause; tick(p) per frame.  autoplay, fps
    heavy = False      # True: update may exceed the 0.5 s self-test limit

    def setup(self): ...                          # once, before the first update
    def update(self, p):                          # p.beta, p["beta"]
        pt = self.plot("t")
        pt.line("h", t, h, color=NAVY, name="pulse")      # create-or-update by key
        self.readout(bw=1 + p.beta)
    def story(self, p):                            # HTML; may use self.r (readouts)
        return f"<p>Roll-off {v(p.beta)} …</p>" + keybox("Key idea")
    def on_again(self, p): ...                     # Button handler
    def on_click(self, key, x, y): ...             # left click in plot `key` (data units)
    def on_drag(self, key, item, i, x, y): ...     # a Plot.handles point is being dragged
    def on_reset(self, p): ...                     # the window's Reset (Ctrl+R) was pressed
    def background(self, p): yield ...             # optional worker-thread generator
    def progress(self, p, item): ...               # draws each yielded item (GUI thread)

LAB = st.Lab(3, "Pulse Shaping", chapter=8, chapter_title="Baseband Transmission",
             experiments=[Example])                # optional: controls_width=300 (pixels)
if __name__ == "__main__":
    st.run(LAB)
```

### Controls

* `Slider(key, label, lo, hi, default, step, unit, fmt, help, enabled_if)`; `LogSlider` (shown
  without scientific notation: 0.0033, 12500; below 10⁻⁴ or above 10⁷ as 2.5×10⁻⁶);
  `IntSlider`.
* `Choice(key, label, options, default, style="buttons"|"menu", labels=None)`. Options may be
  any values (numbers, tuples…): `p.key` is the option, the widget shows `str(option)` or
  `labels[i]`.
* `Toggle(key, label, default)`: long labels wrap; clicking the label flips it.
* `Button(key, label, action=None, primary=False, starts_play=False)`: calls `on_<key>(p)`
  (or `action(exp, p)`), refreshes, and with `starts_play=True` starts Play. Note that
  `Button("reset", …)` calls the same `on_reset(p)` as the window's Reset button.
* `Text(key, label, default, examples=[…], allowed=None, upper=False, max_len=600, live=True,
  clean=None, placeholder, mono=False, random=None)`: a one-line text entry with an optional
  "Examples…" menu. `allowed="01"` keeps only those characters (a bit field), `live=False`
  updates on Enter instead of every keystroke, `clean` post-processes (e.g.
  `lambda s: s or "0"`), `random(rng)` supplies self-test values.
* `Heading(text)`.

The controls column has a fixed width per window (`Lab(controls_width=300)`); long captions
wrap, long button/segment labels are elided (full text in the tooltip), menus never widen the
column, and the scroll area is sized to the *current* experiment's controls.

### Plots

`Plot(key, title, x, y, xlim, ylim, logx, logy, legend="tr"|"tl"|"br"|"bl"|None, legend_cols,
aspect=False, grid=True, axes=True, mouse=None)`. `axes=False` makes a drawing canvas (no axes,
no grid, no pan/zoom, no menu); `Canvas(key, title, xlim=(0, 1), ylim=(0, 1), aspect=False)` is
that preset. With `aspect=True` the view always shows at least `xlim × ylim` (one of them is
widened to keep circles round). Axes never use SI prefixes (0.5 stays 0.5).

**Plot item methods** (all keyed; first call creates, later calls update; reusing a key for a
different kind of item is fine):
`line(key, x, y, color, width, style "-"|"--"|":", name, fill=baseline, step, downsample=None)`
(curves over 4000 points with increasing x are peak-decimated; pass `downsample=False` for
parametric curves), `scatter(key, x, y, color, size, symbol, name, outline=colour)`, `stems`,
`bars(key, x, h, colors=[…], base)` (works on log axes; the legend swatch takes `color` or the
first of `colors`), `fill_between(key, x, y1, y2, color, alpha, name)` (a plain polygon; NaN
splits it; safe for long zero-area stretches), `legend_swatch(name, color, kind="box"|"line"|
"marker")` (legend-only entry), `vline/hline(key, pos, label=…)`, `band/hband(key, a, b)`,
`text(key, x, y, text, anchor)` (`y=None` pins it to the top of the view and follows later
range changes), `arrow(key, x, y, text, direction)`,
`image(key, data, x=(x0,x1), y=(y0,y1), cmap="heat"|"eye"|[colours]|[(pos, colour)]|name,
levels, colorbar)`, `eye(key, y, sps, n_sym, offset, yrange, accumulate, decay, jitter)`
(persistence eye), `eye_traces(key, traces, t, …)`, `psd(key, x, fs, nfft, scale)` (Welch via
commlib), `theory(key, x, y)` / `sim(key, x, y)` (every plot; BERPlot uses them most),
`handles(key, x, y, color, size, axis="y"|"x"|"both")` (draggable points → `on_drag`),
`ConstellationPlot.points/ideal`, `PolarPlot.pattern(key, θ, gain_dB)`.
Axes: `set_title`, `set_xlim`, `set_ylim`, `set_range(xlim, ylim)` (both at once),
`set_labels`, `set_xticks([(v, "label")])` (labels may contain `\n`; the axis grows),
`set_yticks`, `reset_view`, `reset_persistence`.

### Interaction

* **Clicks:** define `on_click(self, key, x, y)`; it receives left clicks inside any plot of
  the experiment in data units (log axes undone). The experiment refreshes afterwards unless it
  returns `False`.
* **Dragging:** draw `plot.handles("h", x, y, axis="y")`, define
  `on_drag(self, key, item, i, x, y)`; typically move a control with
  `self.set_control("g3", value, refresh=False)` and return True. `self.dragging` is True while
  the mouse button is down (e.g. freeze an axis range so the plot does not rescale under the
  cursor). Lab 19's water-filling vessel is the reference.
* **Reset:** `on_reset(self, p)` runs after the window's Reset has restored the defaults and
  before the refresh: leave special modes, clear accumulated state.

**Experiment helpers**: `self.plot(key)`, `self.readout(**values)`, `self.r` (current
readouts), `self.p` (current params), `self.rng` (seeded per experiment), `self.quick`
(True in the self-test: keep Monte Carlo short), `self.frame`, `self.playing`,
`self.dragging`, `self.status(text)`, `self.play_audio(x, fs)`,
`self.set_control(key, value, refresh=True)` (`refresh=False` moves the widget and `self.p`
without a redraw — use it from `tick`/`update` to avoid flicker).

**Readout formats**: `".2f"`, `"sci"` (2.3×10⁻⁴, and "< 10⁻¹²" below that), `"int"`, `"%"`,
a callable, or pass a string value (shown as-is, e.g. "PAM-4"). `NaN` shows as "—" (and is not
a self-test error; infinities are). `floor=1e-6` shows smaller magnitudes as "< 1.0×10⁻⁶".

**Reserved names** (checked when the `Lab` is built; reusing one raises a clear `TypeError`):
the state the framework sets — `frame`, `playing`, `dragging`, `quick`, `rng`, `p`, `r` — and the
helpers `plot`, `readout`, `status`, `set_control`, `play_audio`, `get_story`. Also avoid
redefining `update`, `tick`, `setup`, `background`, `progress`, `story`, `on_click`, `on_drag`,
`on_reset` with a different meaning, and the declarative attributes (`title`, `blurb`, `book`,
`controls`, `plots`, `layout`, `readouts`, `challenges`, `heavy`, `animate`, `autoplay`, `fps`).
Use `make_frame`, not `frame`.

## Command line (every lab)

| flag | effect |
|---|---|
| `--exp N` | open on experiment N |
| `--dark` | dark theme (the choice is remembered) |
| `--smoke S` | visit every experiment, quit after S seconds, print `SMOKE OK` |
| `--selftest [--random K] [--out DIR] [--dark]` | headless test; screenshots to `tests/screens/labNN/` |
| `--shot FILE` | screenshot the window (with `--exp`) and quit |

Keyboard: Ctrl+1…9 experiments, PgUp/PgDn, Ctrl+P play/pause, Ctrl+R reset, Ctrl+E export
PNG, Ctrl+D theme, F1 notes panel. Double-click a slider to reset it; type into its value box.

## Notes
* Headless runs set `QT_QPA_PLATFORM=offscreen`; studio points Qt at the system font folder
  automatically (Windows/macOS offscreen Qt has no fonts of its own).
* Completed challenges, the theme and the last experiment are remembered with QSettings
  (`ModernComms/LabStudio`); the self-test never writes them. The self-test re-evaluates every
  challenge at each random setting. Screenshot names are ASCII slugs of the experiment title
  (`μ-law` → `mu_law`).
* `book.py` reads `book/chapters/*.aux`; rebuild the book and section numbers/pages update.
  Section titles are shown as plain text (`$E_b/N_0$` → Eb/N0, `\texorpdfstring` resolved).
* **"Read more in the book"** opens `book/main.pdf` *at the section* when the default PDF viewer
  accepts a page: on Windows, browsers (Edge, Chrome, Firefox…) are started with
  `file:///…/main.pdf#page=N&nameddest=section.8.5`, SumatraPDF with `-page N`, Adobe Reader with
  `/A page=N`. The physical page is the printed page plus the front-matter offset read from the
  PDF's page labels (needs `pypdf`; without it browsers still jump via the named destination).
  Other viewers, macOS `open` and Linux `xdg-open` cannot be told a page portably, so the PDF
  opens at the start and the status bar names the page.
* Widget internals (`_cb`, `_combo`, `_edit`…) are not API.
