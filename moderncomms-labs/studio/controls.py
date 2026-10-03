"""Control specifications (what the lab author declares) and their Qt widgets.

Declare controls as a class attribute of an Experiment::

    controls = [
        Heading("Transmitter"),
        Choice("pulse", "Pulse shape", ["Raised cosine", "Rectangular"]),
        Slider("beta", "Roll-off β", 0.0, 1.0, 0.35, step=0.01),
        LogSlider("bw", "Channel bandwidth", 0.1, 10, 1.0, unit="× Rs"),
        IntSlider("span", "Filter length", 2, 32, 8, step=2, unit="symbols"),
        Toggle("noise", "Add noise", True),
        Button("again", "New random data"),          # calls Experiment.on_again()
        Text("msg", "Message", "HELLO", examples=["SOS", "WHAT HATH GOD WROUGHT"]),
    ]

Every value-carrying control appears in the parameter object ``p`` passed to
``update(p)`` under its key (``p.beta``, ``p["beta"]``). ``enabled_if`` greys a
control out when it is irrelevant, e.g. ``enabled_if=lambda p: p.pulse != "Sinc"``.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

__all__ = ["Control", "Slider", "LogSlider", "IntSlider", "Choice", "Toggle", "Button", "Heading",
           "Text"]

_SHRINK = (QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)


def _name_label(text):
    """A control caption that wraps instead of widening the controls column."""
    name = QtWidgets.QLabel(text)
    name.setObjectName("ctlname")
    name.setWordWrap(True)
    name.setSizePolicy(*_SHRINK)
    name.setMinimumWidth(30)
    return name


class _ElideButton(QtWidgets.QPushButton):
    """A push button whose text is elided ("Long lab…") when the column is too narrow,
    instead of forcing the whole controls column wider. The full text is the tooltip."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._full = text
        self._tip = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(24)

    def setToolTip(self, tip):                      # noqa: N802 (Qt naming)
        self._tip = tip
        super().setToolTip(tip)

    def minimumSizeHint(self):                      # noqa: N802
        h = super().minimumSizeHint()
        return QtCore.QSize(24, h.height())

    def resizeEvent(self, ev):                      # noqa: N802
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        avail = max(8, self.width() - 14)
        fm = self.fontMetrics()
        txt = fm.elidedText(self._full, QtCore.Qt.ElideRight, avail)
        if txt != self.text():
            super().setText(txt)
        if not self._tip:                           # author tooltip wins; else show full text
            super().setToolTip(self._full if txt != self._full else "")


class _ClickLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, ev):                  # noqa: N802
        if ev.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(ev)


def _auto_decimals(step, lo, hi):
    if step is None:
        span = abs(hi - lo)
        return 0 if span >= 200 else 1 if span >= 20 else 2 if span >= 2 else 3
    if step >= 1 and float(step).is_integer():
        return 0
    return max(0, min(4, int(math.ceil(-math.log10(step) - 1e-9))))


def _plain3(v):
    """Three significant digits without scientific notation: 0.000123, 0.05, 2.5, 47.5,
    12500 (the default display of a LogSlider)."""
    v = float(v)
    if v == 0 or not math.isfinite(v):
        return f"{v:g}"
    e = int(math.floor(math.log10(abs(v))))
    if e < -4 or e >= 7:                    # too long for the value box: 2.5×10⁻⁶
        from .core import sci
        return sci(v, 3)
    if e >= 2:
        return f"{v:.0f}"
    s = f"{v:.{2 - e}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


class Control:
    """Base class. ``key`` names the parameter; ``label`` is what the user sees."""
    has_value = True

    def __init__(self, key, label="", help="", enabled_if=None):
        self.key = key
        self.label = label or key
        self.help = help
        self.enabled_if = enabled_if
        self.widget = None

    default = None

    def make_widget(self, on_change):  # pragma: no cover - overridden
        raise NotImplementedError

    def value(self):
        return self.default

    def set_value(self, v):
        pass

    def random_value(self, rng):
        return self.default

    def set_enabled(self, on):
        if self.widget is not None:
            self.widget.setEnabled(bool(on))


# ----------------------------------------------------------------------------- sliders
class Slider(Control):
    """A labelled slider showing its value and unit. Type into the value box to set it
    exactly; double-click the slider to return to the default."""

    def __init__(self, key, label, lo, hi, default, step=None, unit="", fmt=None, log=False,
                 help="", enabled_if=None, integer=False):
        super().__init__(key, label, help, enabled_if)
        if log and (lo <= 0 or hi <= 0):
            raise ValueError(f"LogSlider {key}: limits must be positive")
        self.lo, self.hi, self.default = float(lo), float(hi), default
        self.step, self.unit, self.log, self.integer = step, unit, log, integer
        if fmt is None:
            if integer:
                fmt = "d"
            elif log:
                fmt = "plain3"
            else:
                fmt = f".{_auto_decimals(step, lo, hi)}f"
        self.fmt = fmt
        if log:
            self.n = 400
        elif step:
            self.n = max(1, int(round((self.hi - self.lo) / step)))
        else:
            self.n = 500
        self._v = default

    # mapping between slider position and value
    def _to_pos(self, v):
        if self.log:
            f = (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        else:
            f = (v - self.lo) / (self.hi - self.lo) if self.hi != self.lo else 0
        return int(round(min(max(f, 0.0), 1.0) * self.n))

    def _from_pos(self, pos):
        f = pos / self.n
        if self.log:
            v = 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))
            return float(f"{v:.3g}")
        v = self.lo + f * (self.hi - self.lo)
        if self.step:
            v = self.lo + round((v - self.lo) / self.step) * self.step
        if self.integer:
            return int(round(v))
        return float(round(v, 10))

    def format(self, v):
        if self.fmt == "d":
            return f"{int(round(v))}"
        if self.fmt == "g3":
            return f"{v:.3g}"
        if self.fmt == "plain3":
            return _plain3(v)
        if callable(self.fmt):
            return self.fmt(v)
        return format(v, self.fmt)

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(1)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        name = _name_label(self.label)
        top.addWidget(name, 1)
        self._edit = QtWidgets.QLineEdit()
        self._edit.setObjectName("ctlvalue")
        self._edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._edit.setFixedWidth(64)
        self._edit.setToolTip("Type a value and press Enter")
        top.addWidget(self._edit)
        unit = QtWidgets.QLabel(self.unit)
        unit.setObjectName("rounit")
        unit.setMinimumWidth(10)
        unit.setMaximumWidth(90)
        if self.unit:
            fm = unit.fontMetrics()
            el = fm.elidedText(self.unit, QtCore.Qt.ElideRight, 90)
            if el != self.unit:
                unit.setText(el)
                unit.setToolTip(self.unit)
        top.addWidget(unit)
        lay.addLayout(top)
        self._sl = _Slider(QtCore.Qt.Horizontal)
        self._sl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self._sl.setRange(0, self.n)
        self._sl.setValue(self._to_pos(self.default))
        self._sl.setFixedHeight(20)
        lay.addWidget(self._sl)
        self._on_change = on_change
        self._sl.valueChanged.connect(self._moved)
        self._sl.reset_requested.connect(lambda: self.set_value(self.default, notify=True))
        self._edit.editingFinished.connect(self._typed)
        self._show(self.default)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _show(self, v):
        self._edit.setText(self.format(v))

    def _moved(self, pos):
        self._v = self._from_pos(pos)
        self._show(self._v)
        self._on_change(self.key, self._v)

    def _typed(self):
        txt = self._edit.text().strip().replace("−", "-").replace(",", ".")
        try:
            v = float(txt)
        except ValueError:
            self._show(self._v)
            return
        v = min(max(v, self.lo), self.hi)
        if self.integer:
            v = int(round(v))
        self.set_value(v, notify=True)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        self._v = int(v) if self.integer else v
        if self.widget is not None:
            self._sl.blockSignals(True)
            self._sl.setValue(self._to_pos(v))
            self._sl.blockSignals(False)
            self._show(v)
        if notify and self.widget is not None:
            self._on_change(self.key, self._v)

    def random_value(self, rng):
        return self._from_pos(int(rng.integers(0, self.n + 1)))


class _Slider(QtWidgets.QSlider):
    reset_requested = QtCore.Signal()

    def mouseDoubleClickEvent(self, ev):
        self.reset_requested.emit()

    def wheelEvent(self, ev):          # don't hijack scrolling of the control column
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


def LogSlider(key, label, lo, hi, default, unit="", fmt=None, help="", enabled_if=None):
    """A slider with logarithmic travel, for quantities spanning decades."""
    return Slider(key, label, lo, hi, default, unit=unit, fmt=fmt, log=True, help=help,
                  enabled_if=enabled_if)


def IntSlider(key, label, lo, hi, default, step=1, unit="", help="", enabled_if=None):
    """A slider that produces integers."""
    return Slider(key, label, lo, hi, default, step=step, unit=unit, help=help,
                  enabled_if=enabled_if, integer=True)


# ----------------------------------------------------------------------------- choices
class Choice(Control):
    """Pick one of several options. Short lists become a segmented button bar, longer ones
    a drop-down. Force either with ``style="buttons"`` or ``style="menu"``.

    Options are usually strings, but any values work (numbers, tuples, …): ``p.key`` is
    the option itself and the button/menu shows ``str(option)``, or ``labels[i]`` when
    ``labels=[...]`` is given, e.g. ``Choice("M", "Order", [4, 16, 64],
    labels=["QPSK", "16-QAM", "64-QAM"])``."""

    def __init__(self, key, label, options, default=None, help="", style=None, enabled_if=None,
                 labels=None):
        super().__init__(key, label, help, enabled_if)
        self.options = list(options)
        if not self.options:
            raise ValueError(f"Choice {key}: no options")
        self.labels = [str(x) for x in (labels if labels is not None else self.options)]
        if len(self.labels) != len(self.options):
            raise ValueError(f"Choice {key}: {len(self.labels)} labels for "
                             f"{len(self.options)} options")
        self.default = default if default is not None else self.options[0]
        if self._index(self.default) is None:
            raise ValueError(f"Choice {key}: default {self.default!r} not in options")
        if style is None:
            style = "buttons" if (len(self.options) <= 3 and
                                  sum(len(o) for o in self.labels) <= 26) else "menu"
        self.style = style
        self._v = self.default

    def _index(self, v):
        for i, o in enumerate(self.options):
            try:
                if o is v or bool(o == v):
                    return i
            except Exception:              # e.g. comparing arrays: treat as different
                continue
        return None

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(3)
        lay.addWidget(_name_label(self.label))
        self._on_change = on_change
        cur = self._index(self._v) or 0
        if self.style == "buttons":
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(0)
            self._group = QtWidgets.QButtonGroup(w)
            self._group.setExclusive(True)
            self._btns = []
            for i, o in enumerate(self.labels):
                b = _ElideButton(o)
                b.setObjectName("seg")
                b.setCheckable(True)
                b.setChecked(i == cur)
                b.setCursor(QtCore.Qt.PointingHandCursor)
                self._group.addButton(b, i)
                row.addWidget(b, 1)
                self._btns.append(b)
            self._group.idClicked.connect(lambda i: self._picked(self.options[i]))
            lay.addLayout(row)
        else:
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(self.labels)
            self._combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self._combo.setMinimumContentsLength(6)
            self._combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            self._combo.setCurrentIndex(cur)
            self._combo.currentIndexChanged.connect(
                lambda i: self._picked(self.options[i]) if 0 <= i < len(self.options) else None)
            lay.addWidget(self._combo)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _picked(self, o):
        if self._index(o) == self._index(self._v):
            return
        self._v = o
        self._on_change(self.key, o)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        i = self._index(v)
        if i is None:
            return
        self._v = self.options[i]
        if self.widget is not None:
            if self.style == "buttons":
                self._btns[i].setChecked(True)
            else:
                self._combo.blockSignals(True)
                self._combo.setCurrentIndex(i)
                self._combo.blockSignals(False)
        if notify and self.widget is not None:
            self._on_change(self.key, self._v)

    def random_value(self, rng):
        return self.options[int(rng.integers(0, len(self.options)))]


class Toggle(Control):
    """An on/off switch. Long labels wrap onto a second line; click the label to flip."""

    def __init__(self, key, label, default=False, help="", enabled_if=None):
        super().__init__(key, label, help, enabled_if)
        self.default = bool(default)
        self._v = self.default

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(2, 6, 2, 4)
        lay.setSpacing(8)
        self._cb = QtWidgets.QCheckBox()
        self._cb.setChecked(self.default)
        self._cb.setCursor(QtCore.Qt.PointingHandCursor)
        self._cb.toggled.connect(lambda on: self._flip(on, on_change))
        lay.addWidget(self._cb, 0, QtCore.Qt.AlignTop)
        lab = _ClickLabel(self.label)
        lab.setObjectName("togglename")
        lab.setWordWrap(True)
        lab.setSizePolicy(*_SHRINK)
        lab.setMinimumWidth(30)
        lab.setCursor(QtCore.Qt.PointingHandCursor)
        lab.clicked.connect(self._cb.toggle)
        lay.addWidget(lab, 1)
        self._lab = lab
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _flip(self, on, on_change):
        self._v = bool(on)
        on_change(self.key, self._v)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        self._v = bool(v)
        if self.widget is not None:
            self._cb.blockSignals(True)
            self._cb.setChecked(self._v)
            self._cb.blockSignals(False)
            if notify:
                self._cb.toggled.emit(self._v)

    def random_value(self, rng):
        return bool(rng.integers(0, 2))


class Button(Control):
    """A push button. Clicking calls ``Experiment.on_<key>(p)`` (or ``action(exp, p)``)
    and then refreshes the experiment. Buttons carry no parameter value.

    ``starts_play=True`` also starts Play (animation) after the handler, if the
    experiment animates and is not already playing, e.g. ``Button("go", "▶  Run the
    decoder", starts_play=True)``. Long labels are elided (full text in the tooltip)."""
    has_value = False

    def __init__(self, key, label, action=None, help="", enabled_if=None, primary=False,
                 starts_play=False):
        super().__init__(key, label, help, enabled_if)
        self.action = action
        self.primary = primary
        self.starts_play = starts_play

    def make_widget(self, on_click):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(2, 6, 2, 2)
        b = _ElideButton(self.label)
        if self.primary:
            b.setObjectName("action")
        b.setCursor(QtCore.Qt.PointingHandCursor)
        b.clicked.connect(lambda: on_click(self.key))
        lay.addWidget(b)
        if self.help:
            b.setToolTip(self.help)
        self.widget = w
        return w


class Heading(Control):
    """A small section heading that groups the controls below it."""
    has_value = False

    def __init__(self, text):
        super().__init__("_heading_" + text, text)

    def make_widget(self, _):
        lab = QtWidgets.QLabel(self.label.upper())
        lab.setObjectName("ctlheading")
        lab.setWordWrap(True)
        lab.setSizePolicy(*_SHRINK)
        self.widget = lab
        return lab


class Text(Control):
    """A one-line text entry (a message to encode, a bit pattern, a polynomial…).

    ``p.key`` is the (cleaned) text.

    default      initial text
    examples     optional list of strings offered in an "Examples…" menu under the field
    allowed      optional string of permitted characters (others are dropped as you type),
                 e.g. ``allowed="01"`` for a bit field; case-sensitive
    upper        True converts to upper case (Morse, telegraph codes)
    max_len      maximum number of characters
    live         True (default): every keystroke updates the experiment;
                 False: update when Enter is pressed or the field loses focus
    clean        optional callable str -> str applied after the rules above (it may
                 also substitute a fallback, e.g. ``lambda s: s or "0"``)
    placeholder  grey hint shown while the field is empty
    mono         True uses the monospace font (bit strings, hex)
    random       optional callable rng -> str for the self-test (default: an example,
                 the default text, or the empty string)
    """

    def __init__(self, key, label, default="", examples=(), help="", enabled_if=None,
                 allowed=None, upper=False, max_len=600, live=True, clean=None,
                 placeholder="type here…", mono=False, random=None):
        super().__init__(key, label, help, enabled_if)
        self.examples = [str(e) for e in examples]
        self.allowed, self.upper, self.max_len = allowed, upper, int(max_len)
        self.live, self.clean, self.placeholder, self.mono = live, clean, placeholder, mono
        self.random = random
        self.default = self._clean(default)
        self._v = self.default

    def _clean(self, s):
        s = "" if s is None else str(s)
        if self.upper:
            s = s.upper()
        if self.allowed is not None:
            s = "".join(c for c in s if c in self.allowed)
        s = s[:self.max_len]
        if self.clean is not None:
            s = str(self.clean(s))
        return s

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(3)
        lay.addWidget(_name_label(self.label))
        self._edit = QtWidgets.QLineEdit(self._v)
        self._edit.setObjectName("textctl")
        if self.mono:
            self._edit.setProperty("mono", "true")
        self._edit.setMaxLength(self.max_len + 16)
        self._edit.setPlaceholderText(self.placeholder)
        self._edit.setClearButtonEnabled(True)
        self._edit.setMinimumWidth(40)
        self._edit.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        if not self.live and not self.help:
            self._edit.setToolTip("Type, then press Enter")
        lay.addWidget(self._edit)
        self._on_change = on_change
        if self.live:
            self._edit.textEdited.connect(lambda t: self._typed(t, final=False))
        self._edit.editingFinished.connect(lambda: self._typed(self._edit.text(), final=True))
        self._combo = None
        if self.examples:
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(["Examples…"] + self.examples)
            self._combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self._combo.setMinimumContentsLength(6)
            self._combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            self._combo.activated.connect(self._pick)
            lay.addWidget(self._combo)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _typed(self, t, final):
        v = self._clean(t)
        if (final or v != t) and self._edit.text() != v:
            # show the cleaned text, keeping the cursor where it was while typing
            pos = self._edit.cursorPosition()
            self._edit.setText(v)
            self._edit.setCursorPosition(min(pos, len(v)))
        if v == self._v:
            return
        self._v = v
        self._on_change(self.key, v)

    def _pick(self, i):
        if i > 0:
            self.set_value(self.examples[i - 1], notify=True)
            self._combo.setCurrentIndex(0)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        self._v = self._clean(v)
        if self.widget is not None:
            self._edit.blockSignals(True)
            self._edit.setText(self._v)
            self._edit.blockSignals(False)
        if notify and self.widget is not None:
            self._on_change(self.key, self._v)

    def random_value(self, rng):
        if self.random is not None:
            return self._clean(self.random(rng))
        pool = self.examples + [self.default, ""]
        return self._clean(pool[int(rng.integers(0, len(pool)))])


def as_number(v):
    """Helper: True if ``v`` is a finite real number."""
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False
