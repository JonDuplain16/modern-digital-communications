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
import unicodedata
import zlib

import numpy as np

__all__ = ["Params", "Readout", "Challenge", "Experiment", "Lab", "fmt_value", "sci", "slug"]


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


def fmt_value(v, fmt, floor=None):
    """Format a readout value. fmt: a format spec (".2f"), "sci", "int", "%" or a callable.
    NaN shows as "—". ``floor``: magnitudes below it show as "< floor" (e.g. a BER below
    what the Monte Carlo can resolve)."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return fmt(v) if callable(fmt) else str(v)
    if np.isnan(fv):
        return "—"
    if floor is not None and np.isfinite(fv) and abs(fv) < floor:
        return "< " + fmt_value(floor, fmt)
    if callable(fmt):
        return fmt(v)
    if not np.isfinite(fv):
        return "∞" if fv > 0 else "−∞"
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


_SPECIAL = {"µ": "mu", "∞": "inf", "²": "2", "³": "3"}


def slug(text, maxlen=40):
    """File-name-safe ASCII slug: "μ-law and A-law (G.711)" -> "mu_law_and_a_law_g_711".
    Greek letters become their names, accents are dropped, symbols become "_"."""
    out = []
    for ch in str(text).lower():
        if ch.isascii():
            out.append(ch if ch.isalnum() else "_")
            continue
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
            continue
        name = unicodedata.name(ch, "")
        if "GREEK" in name and "LETTER" in name:
            out.append(name.split()[-1].lower())
            continue
        base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        out.append("".join(c if c.isalnum() else "_" for c in base.lower()) or "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:maxlen].strip("_") or "x"


class Readout:
    """A big live number above the plots.

    key     name used in ``self.readout(key=value)``
    label   caption, e.g. "Eye height"
    unit    shown after the value, e.g. "dB"
    fmt     ".2f" (default), "sci", "int", "%" (value is a fraction), or callable
    good    optional callable value -> True (green) / False (red) / None (neutral)
    help    tooltip text
    floor   optional: magnitudes below it show as "< floor" (e.g. floor=1e-5 for a BER
            that the simulation cannot resolve below 10⁻⁵)

    A NaN value shows as "—" (and is not an error in the self-test).
    """

    def __init__(self, key, label, unit="", fmt=".2f", good=None, help="", floor=None):
        self.key, self.label, self.unit, self.fmt = key, label, unit, fmt
        self.good, self.help, self.floor = good, help, floor


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
    ``progress``, ``on_<button key>``, ``on_click`` (plot clicks) and ``on_reset``
    (the window's Reset button)."""

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
        self.dragging = False      # True while the user drags a Plot.handles point
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

    def on_click(self, key, x, y):
        """Called when the user left-clicks inside plot ``key``; (x, y) in data units
        (log axes already undone). The experiment is refreshed afterwards unless this
        returns False. Override it to make diagrams interactive."""
        return False

    def on_drag(self, key, item, i, x, y):
        """Called while the user drags point ``i`` of the draggable ``handles`` item ``item``
        in plot ``key``; (x, y) is the new position in data units (already constrained to
        the handle's axis). Typically: move a control with ``self.set_control(...,
        refresh=False)`` or change your own state. The experiment is refreshed afterwards
        unless this returns False. ``self.dragging`` is True until the mouse is released."""
        return False

    def on_reset(self, p):
        """Called by the window's Reset button (Ctrl+R) after every control is back at its
        default and before the refresh: leave special modes, clear accumulated state.
        (A ``Button("reset", ...)`` calls this same method.)"""

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

    def set_control(self, key, value, refresh=True):
        """Programmatically move a control. ``refresh=True`` re-runs ``update`` (debounced);
        ``refresh=False`` only moves the widget and ``self.p`` (use it from ``tick`` or
        ``update`` to show a value the animation chose, without a redraw or flicker)."""
        if self._window is not None:
            self._window._set_control(self, key, value, refresh=refresh)
        elif key in self.p:
            self.p[key] = value

    def get_story(self, p):
        s = self.story
        if callable(s):
            return s(p)
        return s or ""


# Names the framework sets on every experiment instance (a class attribute or method with
# one of these names would be silently overwritten) and helper methods it relies on.
RESERVED_STATE = ("frame", "playing", "dragging", "quick", "rng", "p", "r")
RESERVED_METHODS = ("plot", "readout", "status", "set_control", "play_audio", "get_story")


def check_experiment(cls):
    """Raise TypeError if an Experiment subclass reuses a reserved framework name."""
    if not (isinstance(cls, type) and issubclass(cls, Experiment)):
        raise TypeError(f"{cls!r} is not an Experiment subclass")
    for klass in cls.__mro__:
        if klass is Experiment or klass is object:
            continue
        for name in RESERVED_STATE + RESERVED_METHODS:
            if name in vars(klass):
                what = ("is set by the framework on every experiment (yours would be "
                        "overwritten)" if name in RESERVED_STATE else
                        "is a framework helper method")
                alt = {"frame": "make_frame", "p": "params", "r": "results",
                       "rng": "my_rng", "plot": "draw_plot"}.get(name, name + "_")
                raise TypeError(
                    f"{klass.__name__}.{name}: '{name}' {what}. Rename it (e.g. '{alt}'). "
                    f"Reserved names: {', '.join(RESERVED_STATE + RESERVED_METHODS)}.")


class Lab:
    """A complete lab: number, title, chapter and its experiments (classes).

    ``controls_width`` (pixels, default 300) sets the fixed width of the controls column
    for this lab's window."""

    def __init__(self, number, title, chapter, experiments, subtitle="", chapter_title="",
                 controls_width=300):
        self.number = int(number)
        self.title = title
        self.chapter = chapter
        self.chapter_title = chapter_title
        self.subtitle = subtitle
        self.experiments = list(experiments)
        for cls in self.experiments:
            check_experiment(cls)
        self.controls_width = int(controls_width)
        self.t0 = time.time()

    @property
    def tag(self):
        return f"lab{self.number:02d}"
