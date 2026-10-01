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
    ]

Every value-carrying control appears in the parameter object ``p`` passed to
``update(p)`` under its key (``p.beta``, ``p["beta"]``). ``enabled_if`` greys a
control out when it is irrelevant, e.g. ``enabled_if=lambda p: p.pulse != "Sinc"``.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

__all__ = ["Control", "Slider", "LogSlider", "IntSlider", "Choice", "Toggle", "Button", "Heading"]


def _auto_decimals(step, lo, hi):
    if step is None:
        span = abs(hi - lo)
        return 0 if span >= 200 else 1 if span >= 20 else 2 if span >= 2 else 3
    if step >= 1 and float(step).is_integer():
        return 0
    return max(0, min(4, int(math.ceil(-math.log10(step) - 1e-9))))


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
                fmt = "g3"
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
        if callable(self.fmt):
            return self.fmt(v)
        return format(v, self.fmt)

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(1)
        top = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel(self.label)
        name.setObjectName("ctlname")
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
        top.addWidget(unit)
        lay.addLayout(top)
        self._sl = _Slider(QtCore.Qt.Horizontal)
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
    """Pick one of several options (strings). Short lists become a segmented button bar,
    longer ones a drop-down. Force either with ``style="buttons"`` or ``style="menu"``."""

    def __init__(self, key, label, options, default=None, help="", style=None, enabled_if=None):
        super().__init__(key, label, help, enabled_if)
        self.options = list(options)
        self.default = default if default is not None else self.options[0]
        if self.default not in self.options:
            raise ValueError(f"Choice {key}: default {self.default!r} not in options")
        if style is None:
            style = "buttons" if (len(self.options) <= 3 and
                                  sum(len(o) for o in self.options) <= 26) else "menu"
        self.style = style
        self._v = self.default

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(3)
        name = QtWidgets.QLabel(self.label)
        name.setObjectName("ctlname")
        lay.addWidget(name)
        self._on_change = on_change
        if self.style == "buttons":
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(0)
            self._group = QtWidgets.QButtonGroup(w)
            self._group.setExclusive(True)
            self._btns = []
            for i, o in enumerate(self.options):
                b = QtWidgets.QPushButton(o)
                b.setObjectName("seg")
                b.setCheckable(True)
                b.setChecked(o == self.default)
                b.setCursor(QtCore.Qt.PointingHandCursor)
                self._group.addButton(b, i)
                row.addWidget(b, 1)
                self._btns.append(b)
            self._group.idClicked.connect(lambda i: self._picked(self.options[i]))
            lay.addLayout(row)
        else:
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(self.options)
            self._combo.setCurrentText(self.default)
            self._combo.currentTextChanged.connect(self._picked)
            lay.addWidget(self._combo)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _picked(self, o):
        if o == self._v:
            return
        self._v = o
        self._on_change(self.key, o)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        if v not in self.options:
            return
        self._v = v
        if self.widget is not None:
            if self.style == "buttons":
                self._btns[self.options.index(v)].setChecked(True)
            else:
                self._combo.blockSignals(True)
                self._combo.setCurrentText(v)
                self._combo.blockSignals(False)
        if notify and self.widget is not None:
            self._on_change(self.key, v)

    def random_value(self, rng):
        return self.options[int(rng.integers(0, len(self.options)))]


class Toggle(Control):
    """An on/off switch."""

    def __init__(self, key, label, default=False, help="", enabled_if=None):
        super().__init__(key, label, help, enabled_if)
        self.default = bool(default)
        self._v = self.default

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(2, 6, 2, 4)
        self._cb = QtWidgets.QCheckBox(self.label)
        self._cb.setChecked(self.default)
        self._cb.setCursor(QtCore.Qt.PointingHandCursor)
        self._cb.toggled.connect(lambda on: self._flip(on, on_change))
        lay.addWidget(self._cb)
        lay.addStretch(1)
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
    and then refreshes the experiment. Buttons carry no parameter value."""
    has_value = False

    def __init__(self, key, label, action=None, help="", enabled_if=None, primary=False):
        super().__init__(key, label, help, enabled_if)
        self.action = action
        self.primary = primary

    def make_widget(self, on_click):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(2, 6, 2, 2)
        b = QtWidgets.QPushButton(self.label)
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
        self.widget = lab
        return lab


def as_number(v):
    """Helper: True if ``v`` is a finite real number."""
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False
