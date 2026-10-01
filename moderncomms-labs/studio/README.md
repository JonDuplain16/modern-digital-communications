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
  core.py       Lab, Experiment, Readout, Challenge, Params, fmt_value/sci
  controls.py   Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading (+ their Qt widgets)
  plots.py      Plot and presets: TimePlot, SpectrumPlot, ConstellationPlot, EyePlot, BERPlot,
                PolarPlot, ImagePlot, BarPlot — keyed, update-in-place items
  app.py        MainWindow: layout, debounce, worker threads, animation, theme, export, CLI (run)
  selftest.py   --selftest: defaults + random settings, NaN/exception/timing checks, screenshots
  theme.py      book palette, light/dark themes, Qt style sheet, story CSS
  book.py       resolves LaTeX labels ("sec:ch08:rc") to "§8.5 … (p. 376)" from book/*.aux
  audio.py      optional playback (sounddevice → winsound → afplay/aplay; never required)
  catalog.py    the list of all labs for the launcher, grouped by book part
  launcher.py   the gallery window
```

The update cycle: a control changes → 30 ms debounce → `Experiment.update(p)` runs on the GUI
thread → plot items it touched are updated in place, items it did not touch are hidden →
readouts, story and challenges refresh → if the experiment defines `background(p)`, that
generator starts in a worker thread and each `yield` is handed to `progress(p, item)` on the
GUI thread (a newer control change cancels it). While **Play** is on, a timer calls
`tick(p)` (default: `update(p)` again, i.e. fresh noise) at `fps`.

## API cheat-sheet

```python
import _path                                   # labs/_path.py: puts commlib + studio on sys.path
import numpy as np
import commlib as cl
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, EyePlot, ConstellationPlot, BERPlot, PolarPlot,
                    ImagePlot, BarPlot, Readout, Challenge, NAVY, RED, GREEN, ORANGE, GRAY)
from studio import v, keybox, good, bad

class Example(Experiment):
    title = "Pulses and their spectra"            # sidebar + page title
    blurb = "One line under the title."
    book = "sec:ch08:rc"                           # LaTeX label of the matching section
    controls = [Heading("Transmitter"),
                Choice("pulse", "Pulse", ["RC", "RRC"]),                 # p.pulse
                Slider("beta", "Roll-off β", 0, 1, 0.35, step=0.01),     # p.beta
                LogSlider("bw", "Bandwidth", 0.1, 10, 1, unit="× Rs",
                          enabled_if=lambda p: p.pulse == "RC"),
                IntSlider("span", "Length", 2, 32, 8, step=2, unit="symbols"),
                Toggle("noise", "Add noise", True),
                Button("again", "New data")]                             # calls on_again(p)
    plots = [Plot("t", "Pulse", x="time (T)", y="amplitude", xlim=(-6, 6), legend="tr"),
             SpectrumPlot("f", "Spectrum", x="f / Rs", ylim=(-80, 5))]
    layout = [["t", "f"]]                          # rows of plot keys; repeat a key to span
    row_stretch, col_stretch = None, [1, 1]
    readouts = [Readout("bw", "Bandwidth", "× Rs", ".2f", good=lambda x: x < 1.5)]
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
    def background(self, p): yield ...             # optional worker-thread generator
    def progress(self, p, item): ...               # draws each yielded item (GUI thread)

LAB = st.Lab(3, "Pulse Shaping", chapter=8, chapter_title="Baseband Transmission",
             experiments=[Example])
if __name__ == "__main__":
    st.run(LAB)
```

**Plot item methods** (all keyed; first call creates, later calls update):
`line(key, x, y, color, width, style "-"|"--"|":", name, fill=baseline, step)`,
`scatter(key, x, y, color, size, symbol, name, outline=colour)`, `stems`, `bars(key, x, h,
colors=[…], base)`, `fill_between`, `vline/hline(key, pos, label=…)`, `band/hband(key, a, b)`,
`text(key, x, y, text, anchor)`, `arrow(key, x, y, text, direction)`,
`image(key, data, x=(x0,x1), y=(y0,y1), cmap="heat"|"eye", levels, colorbar)`,
`eye(key, y, sps, n_sym, offset, yrange, accumulate, decay, jitter)` (persistence eye),
`eye_traces(key, traces, t, …)`, `psd(key, x, fs, nfft, scale)` (Welch via commlib),
`ConstellationPlot.points/ideal`, `BERPlot.theory/sim`, `PolarPlot.pattern(key, θ, gain_dB)`.
Axes: `set_title`, `set_xlim`, `set_ylim`, `set_labels`, `set_xticks([(v, "label")])`,
`reset_view`, `reset_persistence`.

**Experiment helpers**: `self.plot(key)`, `self.readout(**values)`, `self.r` (current
readouts), `self.p` (current params), `self.rng` (seeded per experiment), `self.quick`
(True in the self-test: keep Monte Carlo short), `self.frame`, `self.playing`,
`self.status(text)`, `self.play_audio(x, fs)`, `self.set_control(key, value)`.

**Readout formats**: `".2f"`, `"sci"` (2.3×10⁻⁴, and "< 10⁻¹²" below that), `"int"`, `"%"`,
a callable, or pass a string value (shown as-is, e.g. "—" or "PAM-4").

**Reserved attribute names** on an Experiment (do not reuse for your own methods/data):
`frame`, `playing`, `quick`, `rng`, `p`, `r`, `plot`, `readout`, `status`, `update`, `tick`,
`setup`, `background`, `progress`, `story`, `title`, `blurb`, `book`, `controls`, `plots`,
`layout`, `readouts`, `challenges`, `heavy`, `animate`, `autoplay`, `fps`.

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
  (`ModernComms/LabStudio`); the self-test never writes them.
* `book.py` reads `book/chapters/*.aux`; rebuild the book and section numbers/pages update.
