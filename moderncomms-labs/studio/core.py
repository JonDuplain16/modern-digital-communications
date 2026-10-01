"""The declarative building blocks of a lab: Lab, Experiment, Readout, Challenge.

A lab is a list of experiments. An experiment is a class with a few declarative
attributes and one method, ``update(self, p)``::

    class PulseSpectra(Experiment):
        title = "Pulses and their spectra"
        blurb = "Shape a symbol, see the bandwidth it occupies."
        book = "sec:ch08:rc"                       # a LaTeX label in the book
        controls = [Slider("beta", "Roll-off β", 0, 1, 0.35, step=0.01)]
        plots = [Plot("time", "Pulse", x="t / T"), SpectrumPlot("spec", "Spectrum")]
        readouts = [Readout("bw", "Bandwidth", unit="× Rs", fmt=".2f")]
        challenges = [Challenge("Use exactly 25 % excess bandwidth",
                                lambda s: abs(s.p.beta - 0.25) < 0.005)]

        def story(self, p):                        # or a plain HTML string
            return f"<p>A roll-off of <span class='v'>{p.beta:.2f}</span> ...</p>"

        def update(self, p):
            ...compute...
            self.plot("time").line("h", t, h)
            self.readout(bw=1 + p.beta)

Everything else (layout, debouncing, threads, animation, theming, export, self-test)
is the framework's job.
"""
from __future__ import annotations

import math
import time
import zlib

import numpy as np

__all__ = ["Params", "Readout", "Challenge", "Experiment", "Lab", "fmt_value", "sci"]


class Params(dict):
    """Parameter dictionary with attribute access: ``p.beta`` == ``p["beta"]``."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(v, digits=2):
    """Format a number as 2.3×10⁻⁴ (pretty scientific notation)."""
    if v is None or not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits - 1}f}×10{str(e).translate(_SUP)}"


def fmt_value(v, fmt):
    """Format a readout value. fmt: a format spec (".2f"), "sci", "int", "%" or a callable."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if callable(fmt):
        return fmt(v)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(fv):
        return "∞" if fv > 0 else ("−∞" if fv < 0 else "—")
    if fmt == "sci":
        if 0 < abs(fv) < 1e-12:
            return "< 10⁻¹²"
        return sci(fv)
    if fmt == "int":
        return f"{int(round(fv)):,}".replace(",", " ")
    if fmt == "%":
        return f"{100 * fv:.0f}"
    s = format(fv, fmt)
    return s.replace("-", "−")


class Readout:
    """A big live number above the plots.

    key     name used in ``self.readout(key=value)``
    label   caption, e.g. "Eye height"
    unit    shown after the value, e.g. "dB"
    fmt     ".2f" (default), "sci", "int", "%" (value is a fraction), or callable
    good    optional callable value -> True (green) / False (red) / None (neutral)
    help    tooltip text
    """

    def __init__(self, key, label, unit="", fmt=".2f", good=None, help=""):
        self.key, self.label, self.unit, self.fmt = key, label, unit, fmt
        self.good, self.help = good, help


class Challenge:
    """A "Try this" goal that ticks itself when ``check(s)`` returns True.

    ``s.p`` holds the current parameters and ``s.r`` the current readout values (both
    with attribute access); ``s.exp`` is the experiment, for anything else. Exceptions
    inside a check count as "not yet" (e.g. a readout not computed yet)."""

    def __init__(self, text, check, hint=""):
        self.text, self.check, self.hint = text, check, hint
        self.done = False


class Experiment:
    """Base class for one interactive page of a lab. Override the attributes and
    ``update``. Optional hooks: ``setup``, ``story``, ``tick``, ``background``,
    ``progress``, ``on_<button key>``."""

    title = "Untitled experiment"
    blurb = ""            # one line under the title
    book = None           # LaTeX label ("sec:ch08:rc") or (label, fallback text)
    story = ""            # HTML string, or a method story(self, p) -> HTML
    controls = []
    plots = []
    layout = None         # None = automatic, or rows of plot keys: [["a", "b"], ["c", "c"]]
    row_stretch = None    # e.g. [3, 2]
    col_stretch = None    # e.g. [2, 1]
    readouts = []
    challenges = []
    heavy = False         # True: update() may be slow (no 0.5 s limit in the self-test)
    animate = False       # True: show Play/Pause; tick() is called repeatedly while playing
    autoplay = False      # start playing as soon as the experiment is shown
    fps = 20              # animation frame rate

    def __init__(self):
        self.r = Params()          # current readout values
        self.p = Params()          # current parameters
        self.frame = 0             # animation frame counter
        self.playing = False
        self.quick = False         # True during --selftest: keep Monte Carlo short
        self.rng = np.random.default_rng(zlib.crc32(type(self).__name__.encode()))
        self._plots = {}
        self._window = None

    # ---------------------------------------------------------------- author hooks
    def setup(self):
        """Called once before the first update (precompute static data here)."""

    def update(self, p):
        """Recompute and redraw from the parameters ``p``. Must be fast (< ~50 ms ideal)."""
        raise NotImplementedError

    def tick(self, p):
        """One animation frame while playing. Default: run update() again (fresh noise)."""
        self.update(p)

    def background(self, p):
        """Optional generator run in a worker thread after update(). Each ``yield item``
        hands ``item`` to ``progress(p, item)`` on the GUI thread. Do not touch plots here;
        only compute. Restarted automatically whenever a control changes."""
        return None

    def progress(self, p, item):
        """Draw one partial result yielded by ``background`` (GUI thread)."""

    # ---------------------------------------------------------------- helpers for authors
    def plot(self, key):
        """The live Plot object declared with ``key``."""
        return self._plots[key]

    def readout(self, **values):
        """Set readout values: ``self.readout(ber=1e-3, snr=12.5)``."""
        self.r.update(values)
        if self._window is not None:
            self._window._show_readouts(self, values)

    def status(self, text):
        """Show a short message in the status bar."""
        if self._window is not None:
            self._window._status_msg(text)

    def play_audio(self, x, fs, label=""):
        """Play a signal through the speakers (if an audio backend is available)."""
        from . import audio
        ok, msg = audio.play(x, fs)
        self.status(msg if not ok else f"Playing {label}".strip())
        return ok

    def set_control(self, key, value):
        """Programmatically move a control (refreshes the experiment)."""
        if self._window is not None:
            self._window._set_control(self, key, value)

    def get_story(self, p):
        s = self.story
        if callable(s):
            return s(p)
        return s or ""


class Lab:
    """A complete lab: number, title, chapter and its experiments (classes)."""

    def __init__(self, number, title, chapter, experiments, subtitle="", chapter_title=""):
        self.number = int(number)
        self.title = title
        self.chapter = chapter
        self.chapter_title = chapter_title
        self.subtitle = subtitle
        self.experiments = list(experiments)
        self.t0 = time.time()

    @property
    def tag(self):
        return f"lab{self.number:02d}"
