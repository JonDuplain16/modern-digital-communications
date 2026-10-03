"""The lab window: experiment list, controls, live plots, readouts, story and challenges.

Lab authors never touch this module; they call ``studio.run(LAB)``. Command-line flags
understood by every lab::

    python labNN_name.py                 open the lab
    python labNN_name.py --exp 3         open on experiment 3
    python labNN_name.py --dark          start in the dark theme
    python labNN_name.py --smoke 5       open, visit every experiment, quit after 5 s
    python labNN_name.py --selftest      run every experiment at defaults and at random
                                         settings, check for errors/NaNs/slow updates and
                                         save screenshots to tests/screens/labNN/
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")          # commlib imports matplotlib; keep it headless
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import book
from .controls import Button, Heading
from .core import Experiment, Params, fmt_value, slug
from .theme import DARK, LIGHT, PALETTE, THEMES, UI_FONT, story_css, stylesheet

__all__ = ["run", "MainWindow", "make_app"]

SETTINGS = ("ModernComms", "LabStudio")


# ============================================================================ helpers
class _Bridge(QtCore.QObject):
    """Carries results from worker threads to the GUI thread."""
    item = QtCore.Signal(int, object)
    done = QtCore.Signal(int, float)
    error = QtCore.Signal(int, str)


class _Page:
    """Runtime state of one experiment page (built lazily)."""

    def __init__(self, cls):
        self.cls = cls
        self.exp = None
        self.controls = []
        self.plots = {}
        self.challenges = []
        self.ro_widgets = {}
        self.ch_widgets = []
        self.ctl_widget = None
        self.center = None
        self.story_box = None
        self.story_cache = None
        self.dirty = True
        self.bg_running = False


class _Confetti(QtWidgets.QWidget):
    """A short burst of confetti, painted over the plots when a challenge is completed."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self._parts = []
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step)
        self.hide()

    def burst(self, n=90):
        w, h = self.parent().width(), self.parent().height()
        self.setGeometry(0, 0, w, h)
        rng = np.random.default_rng()
        cx = w / 2
        self._parts = [dict(x=cx + rng.normal(0, w * 0.08), y=h * 0.25,
                            vx=rng.normal(0, 6), vy=rng.uniform(-13, -4),
                            a=rng.uniform(0, 360), va=rng.normal(0, 14),
                            c=QtGui.QColor(PALETTE[int(rng.integers(0, 6))]),
                            s=rng.uniform(5, 10)) for _ in range(n)]
        self._age = 0
        self.show()
        self.raise_()
        self._timer.start(16)

    def _step(self):
        self._age += 1
        for p in self._parts:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.45
            p["vx"] *= 0.985
            p["a"] += p["va"]
        if self._age > 95:
            self._timer.stop()
            self.hide()
        self.update()

    def paintEvent(self, ev):
        if not self._parts:
            return
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        fade = max(0.0, 1 - max(0, self._age - 60) / 35)
        for p in self._parts:
            qp.save()
            qp.translate(p["x"], p["y"])
            qp.rotate(p["a"])
            c = QtGui.QColor(p["c"])
            c.setAlphaF(fade)
            qp.fillRect(QtCore.QRectF(-p["s"] / 2, -p["s"] / 4, p["s"], p["s"] / 2), c)
            qp.restore()


class _Toast(QtWidgets.QFrame):
    """A green banner that slides in with a message, then fades."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toast")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(18, 10, 18, 10)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("toasttext")
        lay.addWidget(self.label)
        self.eff = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.eff)
        self.anim = QtCore.QPropertyAnimation(self.eff, b"opacity", self)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._fade)
        self.hide()

    def pop(self, text, ms=3200):
        self.label.setText(text)
        self.adjustSize()
        pw = self.parent().width()
        self.move(max(10, (pw - self.width()) // 2), 70)
        self.anim.stop()
        self.eff.setOpacity(1.0)
        self.show()
        self.raise_()
        self.timer.start(ms)

    def _fade(self):
        self.anim.stop()
        self.anim.setDuration(600)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.finished.connect(self.hide)
        self.anim.start()


class _CurrentStack(QtWidgets.QStackedWidget):
    """A stacked widget sized by its *current* page, not the largest one: a short control
    page does not scroll through the empty space of a long one, and a wide page cannot
    widen the others."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def sizeHint(self):                                     # noqa: N802 (Qt naming)
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):                              # noqa: N802
        w = self.currentWidget()
        if w is None:
            return super().minimumSizeHint()
        return QtCore.QSize(0, w.minimumSizeHint().height())


def _section(text):
    lab = QtWidgets.QLabel(text.upper())
    lab.setObjectName("section")
    return lab


# ============================================================================ main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, lab, theme="light", persist=True, quick=False, celebrate=True,
                 raise_errors=False):
        super().__init__()
        self.lab = lab
        self.theme = THEMES.get(theme, LIGHT)
        self.persist = persist
        self.quick = quick
        self.celebrate = celebrate
        self.raise_errors = raise_errors
        self.settings = QtCore.QSettings(*SETTINGS) if persist else None
        self.pages = [_Page(c) for c in lab.experiments]
        self.cur = -1
        self._bg_gen = 0
        self._bg_page = None
        self._bg_t0 = 0.0
        self._bridge = _Bridge()
        self._bridge.item.connect(self._bg_item)
        self._bridge.done.connect(self._bg_done)
        self._bridge.error.connect(self._bg_error)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(lambda: self.refresh())
        self._anim = QtCore.QTimer(self)
        self._anim.timeout.connect(self._tick)
        self.last_update_ms = 0.0
        self.sync_background = False     # self-test: background jobs are run synchronously
        self.setWindowTitle(f"Lab {lab.number:02d} · {lab.title} — Modern Digital Communications")
        self._build_ui()
        self._shortcuts()
        self._apply_theme()

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # header -------------------------------------------------------------
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(16, 9, 14, 9)
        hl.setSpacing(10)
        badge = QtWidgets.QLabel(f"LAB {self.lab.number:02d}")
        badge.setObjectName("labbadge")
        hl.addWidget(badge)
        title = QtWidgets.QLabel(self.lab.title)
        title.setObjectName("labtitle")
        hl.addWidget(title)
        ch = f"Chapter {self.lab.chapter}"
        if self.lab.chapter_title:
            ch += f" · {self.lab.chapter_title}"
        chap = QtWidgets.QLabel(ch)
        chap.setObjectName("labchapter")
        hl.addWidget(chap)
        hl.addStretch(1)
        self.btn_play = QtWidgets.QPushButton("▶  Play")
        self.btn_play.setObjectName("play")
        self.btn_play.setToolTip("Stream new data continuously (Ctrl+P)")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("Return every control of this experiment to its default (Ctrl+R)")
        self.btn_reset.clicked.connect(self.reset_experiment)
        self.btn_export = QtWidgets.QPushButton("Export PNG")
        self.btn_export.setToolTip("Save the readouts and plots as an image (Ctrl+E)")
        self.btn_export.clicked.connect(self.export_png)
        self.btn_theme = QtWidgets.QPushButton("Dark")
        self.btn_theme.setToolTip("Switch between light and dark themes (Ctrl+D)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_panel = QtWidgets.QPushButton("Hide notes")
        self.btn_panel.setToolTip("Show or hide the explanation panel (F1)")
        self.btn_panel.clicked.connect(self.toggle_panel)
        for b in (self.btn_play, self.btn_reset, self.btn_export, self.btn_theme, self.btn_panel):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            if b is not self.btn_play:
                b.setObjectName("tool")
            hl.addWidget(b)
        root.addWidget(header)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        # sidebar: experiments + controls -----------------------------------
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(getattr(self.lab, "controls_width", 300))
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(12, 4, 12, 10)
        sl.setSpacing(0)
        sl.addWidget(_section("Experiments"))
        self.explist = QtWidgets.QListWidget()
        self.explist.setObjectName("explist")
        self.explist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.explist.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.explist.setTextElideMode(QtCore.Qt.ElideRight)
        self.explist.setCursor(QtCore.Qt.PointingHandCursor)
        for i, pg_ in enumerate(self.pages):
            it = QtWidgets.QListWidgetItem(self._exp_label(i))
            it.setToolTip(pg_.cls.title + (" — " + pg_.cls.blurb if pg_.cls.blurb else ""))
            self.explist.addItem(it)
        self._fit_explist()
        self.explist.currentRowChanged.connect(self.select)
        sl.addWidget(self.explist)
        sl.addWidget(_section("Controls"))
        self.ctl_stack = _CurrentStack()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.ctl_stack)
        sl.addWidget(scroll, 1)
        hint = QtWidgets.QLabel("Tip: double-click a slider to reset it; type into a value to set it.")
        hint.setObjectName("chcount")
        hint.setWordWrap(True)
        sl.addWidget(hint)
        body.addWidget(side)

        # centre: stacked pages ------------------------------------------------
        self.center_host = QtWidgets.QWidget()
        cl_ = QtWidgets.QVBoxLayout(self.center_host)
        cl_.setContentsMargins(14, 10, 14, 10)
        self.center_stack = QtWidgets.QStackedWidget()
        cl_.addWidget(self.center_stack)
        body.addWidget(self.center_host, 1)
        self.toast = _Toast(self.center_host)
        self.confetti = _Confetti(self.center_host)

        # right: story + challenges + book ------------------------------------
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("storypanel")
        self.panel.setFixedWidth(370)
        pl = QtWidgets.QVBoxLayout(self.panel)
        pl.setContentsMargins(14, 4, 12, 12)
        pl.setSpacing(4)
        pl.addWidget(_section("What's going on"))
        self.story = QtWidgets.QTextBrowser()
        self.story.setOpenExternalLinks(True)
        pl.addWidget(self.story, 3)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_section("Try this"))
        row.addStretch(1)
        self.ch_count = QtWidgets.QLabel("")
        self.ch_count.setObjectName("chcount")
        row.addWidget(self.ch_count)
        pl.addLayout(row)
        self.ch_stack = QtWidgets.QStackedWidget()
        self.ch_scroll = QtWidgets.QScrollArea()
        self.ch_scroll.setWidgetResizable(True)
        self.ch_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ch_scroll.setWidget(self.ch_stack)
        pl.addWidget(self.ch_scroll)
        self.booklink = QtWidgets.QLabel()
        self.booklink.setObjectName("booklink")
        self.booklink.setWordWrap(True)
        self.booklink.setTextFormat(QtCore.Qt.RichText)
        self.booklink.linkActivated.connect(self._open_book)
        self.booklink.setCursor(QtCore.Qt.PointingHandCursor)
        pl.addSpacing(6)
        pl.addWidget(self.booklink)
        body.addWidget(self.panel)

        # status bar ---------------------------------------------------------
        sb = self.statusBar()
        self.st_msg = QtWidgets.QLabel("")
        self.st_hover = QtWidgets.QLabel("")
        self.st_time = QtWidgets.QLabel("")
        sb.addWidget(self.st_msg, 1)
        sb.addPermanentWidget(self.st_hover)
        sb.addPermanentWidget(self.st_time)

    def _fit_explist(self):
        """Size the experiment list to show every row (row height depends on DPI and font)."""
        rh = max(self.explist.sizeHintForRow(0), 20) + 2
        self.explist.setFixedHeight(min(10, len(self.pages)) * rh + 6)

    def _exp_label(self, i):
        page = self.pages[i]
        done = ""
        chs = page.challenges if page.exp is not None else []
        if chs and all(c.done for c in chs):
            done = "   ✓"
        return f"{i + 1}.  {page.cls.title}{done}"

    def _shortcuts(self):
        def sc(keys, fn):
            s = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            s.activated.connect(fn)
        sc("Ctrl+P", self.toggle_play)
        sc("Ctrl+R", self.reset_experiment)
        sc("Ctrl+E", self.export_png)
        sc("Ctrl+D", self.toggle_theme)
        sc("F1", self.toggle_panel)
        sc("PgDown", lambda: self.explist.setCurrentRow(min(self.cur + 1, len(self.pages) - 1)))
        sc("PgUp", lambda: self.explist.setCurrentRow(max(self.cur - 1, 0)))
        for i in range(min(9, len(self.pages))):
            sc(f"Ctrl+{i + 1}", lambda i=i: self.explist.setCurrentRow(i))

    # ------------------------------------------------------------------ pages
    def _build_page(self, i):
        page = self.pages[i]
        exp = page.cls()
        exp._window = self
        exp.quick = self.quick
        page.exp = exp
        page.controls = [copy.copy(c) for c in exp.controls]
        page.challenges = [copy.copy(c) for c in exp.challenges]
        for c in page.challenges:
            c.done = False
            if self.settings is not None:
                c.done = self.settings.value(self._ch_key(page, c), False, type=bool)
        # controls column
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(4)
        for c in page.controls:
            if isinstance(c, Button):
                cw = c.make_widget(lambda key, page=page: self._on_button(page, key))
            elif isinstance(c, Heading):
                cw = c.make_widget(None)
            else:
                cw = c.make_widget(lambda key, v, page=page: self._on_control(page, key, v))
            lay.addWidget(cw)
        lay.addStretch(1)
        page.ctl_widget = w
        self.ctl_stack.addWidget(w)
        # centre: title, readouts, plots
        cw = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(cw)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)
        tl = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(exp.title)
        t.setObjectName("exptitle")
        tl.addWidget(t)
        if exp.blurb:
            b = QtWidgets.QLabel("—  " + exp.blurb)
            b.setObjectName("expblurb")
            tl.addWidget(b, 1)
        else:
            tl.addStretch(1)
        cv.addLayout(tl)
        if exp.readouts:
            rl = QtWidgets.QHBoxLayout()
            rl.setSpacing(8)
            for ro in exp.readouts:
                card = QtWidgets.QFrame()
                card.setObjectName("readout")
                card.setMinimumWidth(110)
                card.setMaximumWidth(260)
                v = QtWidgets.QVBoxLayout(card)
                v.setContentsMargins(12, 6, 12, 7)
                v.setSpacing(0)
                n = QtWidgets.QLabel(ro.label)
                n.setObjectName("roname")
                v.addWidget(n)
                hv = QtWidgets.QHBoxLayout()
                hv.setSpacing(5)
                val = QtWidgets.QLabel("—")
                val.setObjectName("rovalue")
                hv.addWidget(val)
                u = QtWidgets.QLabel(ro.unit)
                u.setObjectName("rounit")
                u.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
                hv.addWidget(u, 1)
                v.addLayout(hv)
                if ro.help:
                    card.setToolTip(ro.help)
                rl.addWidget(card, 1)
                page.ro_widgets[ro.key] = (ro, val)
            rl.addStretch(0)
            cv.addLayout(rl)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        specs = {p.key: p for p in exp.plots}
        for spec in exp.plots:
            p = copy.copy(spec)
            p._build(self.theme, hover_cb=self._hover)
            p.widget.setMinimumSize(220, 160)
            page.plots[p.key] = p
        exp._plots = page.plots
        for p in page.plots.values():
            p._drag_cb = (lambda pl, item, i, x, y, phase, page=page:
                          self._on_plot_drag(page, pl.key, item, i, x, y, phase))
        if type(exp).on_click is not Experiment.on_click:
            for p in page.plots.values():
                p._connect_click(lambda pl, x, y, page=page: self._on_plot_click(page, pl.key, x, y))
        layout = exp.layout or self._auto_layout(list(specs))
        cells = {}
        for r, rowkeys in enumerate(layout):
            for c, k in enumerate(rowkeys):
                if k is None:
                    continue
                r0, c0, r1, c1 = cells.get(k, (r, c, r, c))
                cells[k] = (min(r0, r), min(c0, c), max(r1, r), max(c1, c))
        for k, (r0, c0, r1, c1) in cells.items():
            grid.addWidget(page.plots[k].widget, r0, c0, r1 - r0 + 1, c1 - c0 + 1)
        for r, s in enumerate(exp.row_stretch or [1] * len(layout)):
            grid.setRowStretch(r, s)
        ncol = max(len(r) for r in layout)
        for c, s in enumerate(exp.col_stretch or [1] * ncol):
            grid.setColumnStretch(c, s)
        cv.addLayout(grid, 1)
        page.center = cw
        self.center_stack.addWidget(cw)
        # challenges
        chw = QtWidgets.QWidget()
        chl = QtWidgets.QVBoxLayout(chw)
        chl.setContentsMargins(0, 0, 0, 0)
        chl.setSpacing(5)
        for ch in page.challenges:
            fr = QtWidgets.QFrame()
            fr.setObjectName("challenge")
            hl = QtWidgets.QHBoxLayout(fr)
            hl.setContentsMargins(8, 6, 8, 6)
            tick = QtWidgets.QLabel("○")
            tick.setObjectName("chtick")
            tick.setFixedWidth(20)
            tick.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
            hl.addWidget(tick)
            tx = QtWidgets.QLabel(ch.text)
            tx.setObjectName("chtext")
            tx.setWordWrap(True)
            hl.addWidget(tx, 1)
            if ch.hint:
                fr.setToolTip("Hint: " + ch.hint)
            chl.addWidget(fr)
            page.ch_widgets.append((fr, tick))
            self._mark_challenge(fr, tick, ch.done)
        if not page.challenges:
            none = QtWidgets.QLabel("No challenges here: just explore.")
            none.setObjectName("chcount")
            chl.addWidget(none)
        self.ch_stack.addWidget(chw)
        page.ch_host = chw
        # parameters at their defaults, then the author's setup()
        exp.p = self._params(page)
        exp.setup()

    @staticmethod
    def _auto_layout(keys):
        n = len(keys)
        if n <= 2:
            return [keys]
        if n == 3:
            return [[keys[0], keys[1]], [keys[2], keys[2]]]
        if n == 4:
            return [keys[:2], keys[2:]]
        return [keys[:3], keys[3:6] + [None] * (6 - len(keys))]

    def _params(self, page):
        return Params({c.key: c.value() for c in page.controls if c.has_value})

    def _ch_key(self, page, ch):
        return f"{self.lab.tag}/{page.cls.__name__}/{ch.text[:40]}"

    def _mark_challenge(self, fr, tick, done):
        tick.setText("✓" if done else "○")
        for w in (fr, tick):
            w.setProperty("done", "true" if done else "false")
            w.style().unpolish(w)
            w.style().polish(w)

    # ------------------------------------------------------------------ selection
    def select(self, i):
        if i < 0 or i >= len(self.pages) or i == self.cur:
            return
        self._stop_play()
        self._bg_gen += 1                        # cancel any background job
        self.cur = i
        page = self.pages[i]
        if page.exp is None:
            self._build_page(i)
        if self.explist.currentRow() != i:
            self.explist.blockSignals(True)
            self.explist.setCurrentRow(i)
            self.explist.blockSignals(False)
        self.ctl_stack.setCurrentWidget(page.ctl_widget)
        self.center_stack.setCurrentWidget(page.center)
        self.ch_stack.setCurrentWidget(page.ch_host)
        self.ch_stack.setFixedHeight(page.ch_host.sizeHint().height())
        self._fit_challenges()
        text, pdf = book.resolve(page.exp.book)
        if text:
            self.booklink.setText(f'<span style="color:{self.theme.muted}">Read more in the book</span>'
                                  f'<br><a href="book" style="color:{self.theme.accent}; '
                                  f'text-decoration:none"><b>{text}</b></a>')
            self.booklink.show()
        else:
            self.booklink.hide()
        self._book_pdf = pdf
        self._book_ref = page.exp.book
        self.btn_play.setVisible(bool(page.exp.animate))
        self._update_enabled(page)
        page.story_cache = None
        self.refresh()
        self._update_ch_count(page)
        if self.settings is not None:
            self.settings.setValue(f"{self.lab.tag}/last", i)
        if page.exp.animate and page.exp.autoplay and not self.quick:
            self._start_play()

    @property
    def page(self):
        return self.pages[self.cur]

    # ------------------------------------------------------------------ control events
    def _on_control(self, page, key, value):
        page.exp.p = self._params(page)
        self._update_enabled(page)
        if page is self.page:
            self._debounce.start()

    def _on_button(self, page, key):
        exp = page.exp
        ctl = next(c for c in page.controls if c.key == key)
        try:
            if ctl.action is not None:
                ctl.action(exp, exp.p)
            else:
                fn = getattr(exp, f"on_{key}", None)
                if fn is not None:
                    fn(exp.p)
        except Exception:
            self._report_error(traceback.format_exc())
        self.refresh()
        if getattr(ctl, "starts_play", False) and exp.animate and not self._anim.isActive() \
                and page is self.page and not self.quick:
            self._start_play()

    def _set_control(self, exp, key, value, refresh=True):
        page = next(p for p in self.pages if p.exp is exp)
        for c in page.controls:
            if c.key == key:
                c.set_value(value)
        page.exp.p = self._params(page)
        self._update_enabled(page)
        if refresh:
            self._debounce.start()

    def _on_plot_drag(self, page, key, item, i, x, y, phase):
        exp = page.exp
        if exp is None or self.cur < 0 or self.page is not page:
            return
        exp.dragging = phase != "finish"
        try:
            res = exp.on_drag(key, item, i, x, y)
        except Exception:
            exp.dragging = False
            self._report_error(traceback.format_exc())
            return
        if res is not False or phase == "finish":
            if phase == "finish":
                self._debounce.stop()
                self.refresh(page)
            else:
                self._debounce.start()

    def _on_plot_click(self, page, key, x, y):
        if page.exp is None or self.cur < 0 or self.page is not page:
            return
        try:
            res = page.exp.on_click(key, x, y)
        except Exception:
            self._report_error(traceback.format_exc())
            return
        if res is not False:
            self.refresh(page)

    def _update_enabled(self, page):
        p = page.exp.p
        for c in page.controls:
            if c.enabled_if is not None:
                try:
                    c.set_enabled(c.enabled_if(p))
                except Exception:
                    c.set_enabled(True)

    # ------------------------------------------------------------------ the update cycle
    def refresh(self, page=None):
        """Recompute the current experiment from its controls."""
        page = page or (self.page if self.cur >= 0 else None)
        if page is None or page.exp is None:
            return
        exp = page.exp
        self._bg_gen += 1
        p = self._params(page)
        exp.p = p
        t0 = time.perf_counter()
        for pl in page.plots.values():
            pl._begin()
        try:
            exp.update(p)
        except Exception:
            if self.raise_errors:
                raise
            self._report_error(traceback.format_exc())
            return
        finally:
            for pl in page.plots.values():
                pl._end()
        self.last_update_ms = (time.perf_counter() - t0) * 1e3
        self.st_time.setText(f"update {self.last_update_ms:.0f} ms")
        page.dirty = False
        self._refresh_story(page)
        self._check_challenges(page)
        gen = exp.background(p)
        if gen is not None and not self.sync_background:
            self._start_bg(page, gen)

    def _refresh_story(self, page):
        try:
            html = page.exp.get_story(page.exp.p)
        except Exception:
            html = "<p class='bad'>(story error)</p><pre>" + traceback.format_exc() + "</pre>"
            if self.raise_errors:
                raise
        if html != page.story_cache:
            sb = self.story.verticalScrollBar()
            pos = sb.value()
            self.story.setHtml(html)
            sb.setValue(pos)
            page.story_cache = html

    def _show_readouts(self, exp, values):
        page = next((p for p in self.pages if p.exp is exp), None)
        if page is None:
            return
        for k, v in values.items():
            if k not in page.ro_widgets:
                continue
            ro, lab = page.ro_widgets[k]
            lab.setText(fmt_value(v, ro.fmt, getattr(ro, "floor", None)))
            col = self.theme.text
            if ro.good is not None and v is not None and not _is_nan(v):
                try:
                    g = ro.good(v)
                    col = self.theme.good if g is True else self.theme.bad if g is False else col
                except Exception:
                    pass
            lab.setStyleSheet(f"color: {col};")

    def _check_challenges(self, page):
        exp = page.exp
        s = SimpleNamespace(p=exp.p, r=exp.r, exp=exp)
        newly = []
        for ch, (fr, tick) in zip(page.challenges, page.ch_widgets):
            if ch.done:
                continue
            try:
                ok = bool(ch.check(s))
            except Exception:
                ok = False
            if ok:
                ch.done = True
                newly.append(ch)
                self._mark_challenge(fr, tick, True)
                if self.settings is not None:
                    self.settings.setValue(self._ch_key(page, ch), True)
        if newly:
            self._update_ch_count(page)
            self.explist.item(self.pages.index(page)).setText(self._exp_label(self.pages.index(page)))
            if self.celebrate:
                self.toast.pop("✓  Challenge complete: " + newly[0].text)
                self.confetti.burst()

    def _update_ch_count(self, page):
        n = len(page.challenges)
        d = sum(c.done for c in page.challenges)
        self.ch_count.setText(f"{d} / {n} done" if n else "")

    # ------------------------------------------------------------------ background jobs
    def _start_bg(self, page, gen):
        g = self._bg_gen
        self._bg_page = page
        self._bg_t0 = time.perf_counter()
        page.bg_running = True
        self._status_msg("Computing in the background…")

        def work():
            try:
                for item in gen:
                    if self._bg_gen != g:
                        return
                    self._bridge.item.emit(g, item)
                    time.sleep(0.001)            # let the GUI breathe
                if self._bg_gen == g:
                    self._bridge.done.emit(g, time.perf_counter() - self._bg_t0)
            except Exception:
                self._bridge.error.emit(g, traceback.format_exc())

        threading.Thread(target=work, daemon=True).start()

    def _bg_item(self, g, item):
        if g != self._bg_gen or self._bg_page is None:
            return
        page = self._bg_page
        try:
            page.exp.progress(page.exp.p, item)
        except Exception:
            self._report_error(traceback.format_exc())
            return
        self._refresh_story(page)
        self._check_challenges(page)

    def _bg_done(self, g, dt):
        if g == self._bg_gen and self._bg_page is not None:
            self._bg_page.bg_running = False
            self._status_msg(f"Background computation finished in {dt:.1f} s.")

    def _bg_error(self, g, tb):
        if g == self._bg_gen:
            self._report_error(tb)

    def run_background_sync(self, page, budget_s=60.0):
        """Run the background generator to completion on this thread (self-test)."""
        gen = page.exp.background(page.exp.p)
        if gen is None:
            return 0.0
        self._bg_gen += 1
        t0 = time.perf_counter()
        for item in gen:
            page.exp.progress(page.exp.p, item)
            if time.perf_counter() - t0 > budget_s:
                break
        self._refresh_story(page)
        self._check_challenges(page)
        dt = time.perf_counter() - t0
        self._status_msg(f"Background computation finished in {dt:.1f} s.")
        return dt

    # ------------------------------------------------------------------ animation
    def toggle_play(self):
        if self.cur < 0 or not self.page.exp.animate:
            return
        if self._anim.isActive():
            self._stop_play()
            # ticks only re-check every few frames: settle the story and challenges now
            self._refresh_story(self.page)
            self._check_challenges(self.page)
        else:
            self._start_play()

    def _start_play(self):
        exp = self.page.exp
        exp.playing = True
        self._anim.start(int(1000 / max(1, exp.fps)))
        self.btn_play.setText("❚❚  Pause")

    def _stop_play(self):
        self._anim.stop()
        if self.cur >= 0 and self.page.exp is not None:
            self.page.exp.playing = False
        self.btn_play.setText("▶  Play")

    def _tick(self):
        page = self.page
        exp = page.exp
        if self._debounce.isActive():
            return
        exp.frame += 1
        t0 = time.perf_counter()
        for pl in page.plots.values():
            pl._begin()
        try:
            exp.tick(exp.p)
        except Exception:
            self._stop_play()
            self._report_error(traceback.format_exc())
        finally:
            for pl in page.plots.values():
                pl._end()
        self.st_time.setText(f"frame {(time.perf_counter() - t0) * 1e3:.0f} ms")
        if exp.frame % 3 == 0:
            self._refresh_story(page)
            self._check_challenges(page)

    # ------------------------------------------------------------------ toolbar actions
    def reset_experiment(self):
        page = self.page
        for c in page.controls:
            if c.has_value:
                c.set_value(c.default)
        for pl in page.plots.values():
            pl.reset_view()
            pl.reset_persistence()
        page.exp.p = self._params(page)
        self._update_enabled(page)
        try:
            page.exp.on_reset(page.exp.p)
        except Exception:
            if self.raise_errors:
                raise
            self._report_error(traceback.format_exc())
        self.refresh()
        self._status_msg("Controls reset to their defaults.")

    def grab_center(self):
        return self.page.center.grab()

    def export_png(self, path=None):
        if path is None:
            default = os.path.join(QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.PicturesLocation),
                f"{self.lab.tag}_{slug(self.page.cls.title)}.png")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PNG", default,
                                                            "PNG image (*.png)")
            if not path:
                return
        self.grab_center().save(path)
        self._status_msg(f"Saved {path}")

    def toggle_theme(self):
        self.theme = LIGHT if self.theme is DARK else DARK
        self._apply_theme()
        for page in self.pages:
            if page.exp is not None:
                for pl in page.plots.values():
                    pl._restyle(self.theme)
                page.dirty = True
                page.story_cache = None
        if self.cur >= 0:
            self.select_refresh()
        if self.settings is not None:
            self.settings.setValue("theme", self.theme.name)

    def select_refresh(self):
        i = self.cur
        self.cur = -1
        self.select(i)

    def _apply_theme(self):
        app = QtWidgets.QApplication.instance()
        app.setStyleSheet(stylesheet(self.theme))
        self.story.document().setDefaultStyleSheet(story_css(self.theme))
        self._fit_explist()
        self.btn_theme.setText("Light" if self.theme is DARK else "Dark")
        for page in self.pages:
            for fr, tick in page.ch_widgets:
                for w in (fr, tick):
                    w.style().unpolish(w)
                    w.style().polish(w)

    def toggle_panel(self):
        vis = not self.panel.isVisible()
        self.panel.setVisible(vis)
        self.btn_panel.setText("Hide notes" if vis else "Show notes")

    def _open_book(self, _link):
        if not self._book_pdf:
            self._status_msg("The book PDF has not been built yet (book/main.pdf).")
            return
        how = book.open_pdf(self._book_pdf, getattr(self, "_book_ref", None))
        self._status_msg(how)

    # ------------------------------------------------------------------ status
    def _hover(self, plot, x, y):
        xl = plot.xlabel or "x"
        yl = plot.ylabel or "y"
        self.st_hover.setText(f"{xl} = {x:.4g}    {yl} = {y:.4g}    ")

    def _status_msg(self, text):
        self.st_msg.setText(text)

    def _report_error(self, tb):
        sys.stderr.write(tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "error"
        self._status_msg("⚠ " + last)
        self.st_msg.setStyleSheet(f"color: {self.theme.bad};")
        QtCore.QTimer.singleShot(6000, lambda: self.st_msg.setStyleSheet(""))

    def _fit_challenges(self):
        """Challenges take what they need, but at most ~40 % of the notes panel."""
        need = self.ch_stack.height() + 4
        self.ch_scroll.setFixedHeight(min(need, max(140, int(0.40 * self.panel.height()))))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.cur >= 0:
            self._fit_challenges()
        if self.confetti.isVisible():
            self.confetti.setGeometry(0, 0, self.center_host.width(), self.center_host.height())

    def closeEvent(self, ev):
        self._bg_gen += 1
        self._anim.stop()
        super().closeEvent(ev)


def _is_nan(v):
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False


# ============================================================================ entry point
def make_app():
    """Create (or return) the QApplication with the studio look."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Headless (offscreen) Qt has no font database of its own on Windows/macOS.
        if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen") and \
                not os.environ.get("QT_QPA_FONTDIR"):
            for d in (os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                      "/System/Library/Fonts", "/usr/share/fonts/truetype"):
                if os.path.isdir(d):
                    os.environ["QT_QPA_FONTDIR"] = d
                    break
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QtWidgets.QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major", useOpenGL=False)
    f = QtGui.QFont(UI_FONT)
    f.setPointSizeF(10)
    app.setFont(f)
    app.setApplicationName("Modern Digital Communications Lab Studio")
    return app


def run(lab, argv=None):
    """Run a lab: parse the command line, open the window (or self-test)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=f"Lab {lab.number:02d}: {lab.title}")
    ap.add_argument("--exp", type=int, default=None, help="open on experiment N (1-based)")
    ap.add_argument("--dark", action="store_true", help="start in the dark theme")
    ap.add_argument("--smoke", type=float, default=None, metavar="SECONDS",
                    help="visit every experiment and quit after SECONDS")
    ap.add_argument("--selftest", action="store_true", help="automatic test + screenshots")
    ap.add_argument("--out", default=None, help="screenshot folder for --selftest")
    ap.add_argument("--random", type=int, default=5, help="random settings per experiment")
    ap.add_argument("--shot", default=None, help="save a screenshot of --exp to this file and quit")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.selftest:
        from .selftest import run_selftest
        sys.exit(run_selftest(lab, out_dir=args.out, n_random=args.random, dark=args.dark))

    app = make_app()
    persist = args.smoke is None and args.shot is None
    win = MainWindow(lab, persist=persist)
    if persist and win.settings is not None:
        if win.settings.value("theme", "light") == "dark":
            win.theme = DARK
            win._apply_theme()
    if args.dark and win.theme is not DARK:
        win.toggle_theme()
    scr = app.primaryScreen().availableGeometry()
    win.resize(min(1640, int(scr.width() * 0.94)), min(1000, int(scr.height() * 0.92)))
    start = 0
    if args.exp:
        start = args.exp - 1
    elif persist and win.settings is not None:
        start = int(win.settings.value(f"{lab.tag}/last", 0))
    start = max(0, min(start, len(lab.experiments) - 1))
    win.show()
    win.explist.setCurrentRow(start)
    win.select(start)

    if args.shot:
        def shoot():
            win.grab().save(args.shot)
            app.quit()
        QtCore.QTimer.singleShot(800, shoot)
    if args.smoke is not None:
        n = len(lab.experiments)
        dt = max(0.2, args.smoke / (n + 1))
        for k in range(n):
            QtCore.QTimer.singleShot(int(1000 * dt * (k + 1)), lambda k=k: win.explist.setCurrentRow(k))

        def done():
            print(f"SMOKE OK {lab.tag}: {n} experiments shown, last update "
                  f"{win.last_update_ms:.0f} ms", flush=True)
            app.quit()
        QtCore.QTimer.singleShot(int(1000 * args.smoke), done)
    return app.exec()
