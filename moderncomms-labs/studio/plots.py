"""Live plots for comms experiments, built on pyqtgraph.

Declare plots as a class attribute of an Experiment::

    plots = [
        Plot("time", "Impulse response", x="time (symbols)", y="amplitude", xlim=(-6, 6)),
        SpectrumPlot("spec", "Spectrum", x="frequency (× Rs)", ylim=(-80, 5)),
        EyePlot("eye", "Eye diagram", yrange=(-1.6, 1.6)),
        ConstellationPlot("const", "At the sampler", lim=1.6),
        BERPlot("ber", "Bit error rate"),
        PolarPlot("beam", "Array pattern", floor_db=-40),
        ImagePlot("map", "Coverage", x="x (km)", y="y (km)"),
    ]

and inside ``update(p)`` push data by *item key*::

    pt = self.plot("time")
    pt.line("pulse", t, h, color=NAVY, name="raised cosine")
    pt.vline("cursor", 0.5, color=RED, label="sample here")

An item is created the first time its key is used and updated in place after that,
which is what makes the plots fast. Items that an ``update`` (or animation tick) does
**not** touch are hidden automatically, so conditional traces need no bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .theme import GRAY, GREEN, NAVY, PALETTE, UI_FONT

__all__ = ["Plot", "TimePlot", "SpectrumPlot", "ConstellationPlot", "EyePlot", "BERPlot",
           "PolarPlot", "ImagePlot", "BarPlot", "Canvas"]

_STYLES = {"-": QtCore.Qt.SolidLine, "--": QtCore.Qt.DashLine, ":": QtCore.Qt.DotLine,
           "-.": QtCore.Qt.DashDotLine}
_LEGEND_POS = {"tr": (-8, 8), "tl": (60, 8), "br": (-8, -40), "bl": (60, -40)}


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class _Axis(pg.AxisItem):
    """Linear axis with 'nice' tick spacing (1, 2, 5 × 10ⁿ) chosen for about one label
    per 80 px horizontally / 45 px vertically, and unlabelled minor ticks in between.
    Never rescales its labels with an SI prefix (0.5 stays 0.5, not 500 m)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.enableAutoSIPrefix(False)

    def tickSpacing(self, minVal, maxVal, size):
        if self.logMode or maxVal <= minVal or size <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        px = 70.0 if self.orientation in ("bottom", "top") else 40.0
        span = maxVal - minVal
        raw = span / max(1.0, size / px)
        e = 10 ** np.floor(np.log10(raw))
        nice = [m * e for m in (0.1, 0.2, 0.5, 1, 2, 5, 10)]
        i = next(k for k, v in enumerate(nice) if v >= raw)
        while i > 0 and span / nice[i] < 2.5:          # always at least ~3 labels
            i -= 1
        major = nice[i]
        minor = major / (4 if str(major / e).startswith("2") else 5)
        return [(major, 0), (minor, 0)]


class _LogAxis(pg.AxisItem):
    """Log axis whose decade labels read 10⁻³ rather than 0.001."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            vi = int(round(v))
            if abs(v - vi) < 1e-6:
                out.append("1" if vi == 0 else "10" if vi == 1 else "10" + str(vi).translate(_SUP))
            else:
                out.append("")
        return out


def _qcolor(c, alpha=1.0):
    q = pg.mkColor(c)
    q.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return q


class Plot:
    """A general x–y plot. Options:

    title, x, y      title and axis labels (put units in the label: "time (ms)")
    xlim, ylim       fixed ranges (None = autoscale). For log axes give linear values.
    logx, logy       logarithmic axes
    legend           "tr", "tl", "br", "bl" or None
    legend_cols      >1 lays the legend out horizontally (leave headroom with ylim)
    aspect           True locks 1:1 aspect (constellations, maps); the view always shows
                     at least xlim × ylim, widening one of them to keep circles round
    grid             show grid lines
    axes             False: a drawing canvas (no axes, no grid, no pan/zoom, no menu)
    mouse            pan/zoom with the mouse (default: True, False when axes=False)
    """

    def __init__(self, key, title="", x="", y="", xlim=None, ylim=None, logx=False, logy=False,
                 legend="tr", aspect=False, grid=True, legend_cols=1, axes=True, mouse=None):
        self.key, self.title_text, self.xlabel, self.ylabel = key, title, x, y
        self.legend_cols = legend_cols
        self.xlim, self.ylim, self.logx, self.logy = xlim, ylim, logx, logy
        self.legend_pos, self.aspect, self.grid = legend, aspect, grid and axes
        self.axes = axes
        self.mouse = axes if mouse is None else mouse
        self.widget = None

    # ------------------------------------------------------------------ construction
    def _build(self, theme, hover_cb=None):
        """Create the pyqtgraph widget (called by the framework)."""
        self.theme = theme
        self._items, self._kinds, self._names = {}, {}, {}
        self._touched = set()
        self._eye_acc = {}
        self._colorbars = {}
        axes = {"left": _LogAxis("left") if self.logy else _Axis("left"),
                "bottom": _LogAxis("bottom") if self.logx else _Axis("bottom")}
        self.widget = pg.PlotWidget(axisItems=axes)
        self.pi = self.widget.getPlotItem()
        self.vb = self.pi.getViewBox()
        self.pi.setMenuEnabled(True)
        self.pi.hideButtons()
        self.vb.setMouseMode(pg.ViewBox.PanMode)
        self.legend = None
        if self.legend_pos:
            self.legend = self.pi.addLegend(offset=_LEGEND_POS.get(self.legend_pos, (-8, 8)),
                                            labelTextSize="9pt")
            self.legend.setVisible(False)
            if self.legend_cols > 1:
                self.legend.setColumnCount(self.legend_cols)
        self.pi.setLogMode(x=self.logx, y=self.logy)
        if self.aspect:
            self.vb.setAspectLocked(True)
        self._style()
        self._apply_canvas()
        self.reset_view()
        self._extra_build()
        if hover_cb is not None:
            self.widget.scene().sigMouseMoved.connect(lambda pos: self._hover(pos, hover_cb))
        self.vb.sigRangeChanged.connect(lambda *_: self._repin_top())
        return self.widget

    def _apply_canvas(self):
        if not self.axes:
            for side in ("left", "bottom"):
                self.pi.hideAxis(side)
        if not self.mouse:
            self.vb.setMouseEnabled(False, False)
            self.pi.setMenuEnabled(False)

    def _connect_click(self, cb):
        """Call ``cb(self, x, y)`` (data units) on left clicks inside the plot area."""
        def handler(ev):
            try:
                if ev.button() != QtCore.Qt.LeftButton or self.widget is None:
                    return
                pos = ev.scenePos()
                if not self.vb.sceneBoundingRect().contains(pos):
                    return
                p = self.vb.mapSceneToView(pos)
                x, y = p.x(), p.y()
                if self.logx:
                    x = 10 ** x
                if self.logy:
                    y = 10 ** y
                cb(self, x, y)
            except Exception:                   # a click must never crash the lab
                import traceback
                traceback.print_exc()
        self.widget.scene().sigMouseClicked.connect(handler)

    def _extra_build(self):
        pass

    def _style(self):
        t = self.theme
        self.widget.setBackground(t.plot_bg)
        font = QtGui.QFont(UI_FONT)
        font.setPointSizeF(8.5)
        for side in ("left", "bottom"):
            ax = self.pi.getAxis(side)
            ax.setPen(pg.mkPen(t.border, width=1))
            ax.setTextPen(pg.mkPen(t.axis))
            ax.setTickFont(font)
            ax.setStyle(tickLength=-4, autoExpandTextSpace=True)
            ax.enableAutoSIPrefix(False)
            ax.setStyle(maxTextLevel=0)          # label major ticks only: clean axes
        lab = {"color": t.axis, "font-size": "9.5pt", "font-family": UI_FONT}
        self.pi.setLabel("bottom", self.xlabel, **lab)
        self.pi.setLabel("left", self.ylabel, **lab)
        self.set_title(self.title_text)
        if self.grid:
            self.pi.showGrid(x=True, y=True, alpha=t.grid / 255)
        if self.legend is not None:
            self.legend.setBrush(_qcolor(t.card, 0.86))
            self.legend.setPen(pg.mkPen(t.border))
            self.legend.setLabelTextColor(t.text)

    def _restyle(self, theme):
        """Switch theme: forget all items (they are recreated in the new colours)."""
        self.theme = theme
        self.clear()
        self._style()
        self._apply_canvas()
        self._extra_build()

    def reset_view(self):
        """Return to the declared axis ranges (or autoscale)."""
        if self.widget is None:
            return
        xr = yr = None
        if self.xlim is not None:
            lo, hi = self.xlim
            if self.logx:
                lo, hi = np.log10(lo), np.log10(hi)
            xr = (lo, hi)
        if self.ylim is not None:
            lo, hi = self.ylim
            if self.logy:
                lo, hi = np.log10(lo), np.log10(hi)
            yr = (lo, hi)
        # one joint call: with a locked aspect, separate x and y calls fight each other
        # (the second one shrinks the first); jointly, both ranges stay fully visible.
        if xr is not None or yr is not None:
            self.vb.setRange(xRange=xr, yRange=yr, padding=0, disableAutoRange=True)
        if xr is None:
            self.vb.enableAutoRange(axis="x")
        if yr is None:
            self.vb.enableAutoRange(axis="y")
        self._repin_top()

    def _hover(self, pos, cb):
        if not self.pi.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        x, y = p.x(), p.y()
        if self.logx:
            x = 10 ** x
        if self.logy:
            y = 10 ** y
        cb(self, x, y)

    # ------------------------------------------------------------------ bookkeeping
    def _begin(self):
        self._touched = set()

    def _end(self):
        for k, it in self._items.items():
            if k not in self._touched and it.isVisible():
                it.setVisible(False)
                self._legend_remove(k)

    def _get(self, key, kind, factory, ignore_bounds=False):
        """Return the item for ``key``, creating it (and re-showing it) as needed."""
        self._touched.add(key)
        it = self._items.get(key)
        if it is not None and self._kinds[key] != kind:
            self._remove(key)
            it = None
        if it is None:
            it = factory()
            self.pi.addItem(it, ignoreBounds=ignore_bounds)
            self._items[key], self._kinds[key] = it, kind
        elif not it.isVisible():
            it.setVisible(True)
        return it

    def _remove(self, key):
        self._legend_remove(key)               # before the item is forgotten
        it = self._items.pop(key, None)
        self._kinds.pop(key, None)
        if it is not None:
            self.pi.removeItem(it)

    def _legend_set(self, key, item, name):
        if self.legend is None:
            return
        old = self._names.get(key)
        if old == name:
            return
        if old is not None:
            self.legend.removeItem(item)
            self._names.pop(key, None)
        if name:
            self.legend.addItem(item, name)
            self._names[key] = name
        self.legend.setVisible(bool(self._names))

    def _legend_remove(self, key):
        if self.legend is not None and key in self._names:
            it = self._items.get(key)
            if it is not None:
                self.legend.removeItem(it)
            self._names.pop(key, None)
            self.legend.setVisible(bool(self._names))

    def _y(self, y):
        """Convert a y position to view coordinates (log axes)."""
        return np.log10(max(y, 1e-300)) if self.logy else y

    def _x(self, x):
        return np.log10(max(x, 1e-300)) if self.logx else x

    def _pen(self, color, width=2.0, style="-", alpha=1.0):
        return pg.mkPen(_qcolor(self.theme.c(color), alpha), width=width,
                        style=_STYLES.get(style, QtCore.Qt.SolidLine))

    def clear(self):
        """Remove every item from the plot."""
        for k in list(self._items):
            self._remove(k)
        self._eye_acc.clear()

    def item(self, key):
        """The underlying pyqtgraph item (for advanced customisation)."""
        return self._items.get(key)

    # ------------------------------------------------------------------ axes and titles
    def set_title(self, text):
        """Change the plot title (may contain simple HTML)."""
        self.title_text = text
        if self.widget is not None:
            if text:
                self.pi.setTitle(text, color=self.theme.text, size="10.5pt")
            else:
                self.pi.setTitle(None)

    def set_xlim(self, lo, hi):
        self.xlim = (lo, hi)
        self.reset_view()

    def set_ylim(self, lo, hi):
        self.ylim = (lo, hi)
        self.reset_view()

    def set_range(self, xlim=None, ylim=None):
        """Set both ranges at once (one view change; best for aspect-locked plots).
        None leaves that axis as it is."""
        if xlim is not None:
            self.xlim = tuple(xlim)
        if ylim is not None:
            self.ylim = tuple(ylim)
        self.reset_view()

    def set_xticks(self, ticks):
        """Label the x axis with explicit ticks: [(value, "label"), ...] (None = automatic).
        Labels may contain "\n" for two-line labels (the axis grows to fit)."""
        ax = self.pi.getAxis("bottom")
        ax.setTicks(None if ticks is None else [list(ticks)])
        lines = max((str(t[1]).count("\n") + 1 for t in (ticks or [])), default=1)
        if lines > 1 or getattr(self, "_xtick_lines", 1) > 1:
            fh = QtGui.QFontMetrics(ax.style.get("tickFont") or QtGui.QFont(UI_FONT)).height()
            lab_h = 22 if self.xlabel else 4
            ax.setHeight(int(lines * fh + lab_h + 10) if lines > 1 else None)
        self._xtick_lines = lines

    def set_yticks(self, ticks):
        """Label the y axis with explicit ticks: [(value, "label"), ...] (None = automatic)."""
        self.pi.getAxis("left").setTicks(None if ticks is None else [list(ticks)])

    def set_labels(self, x=None, y=None):
        if x is not None:
            self.xlabel = x
        if y is not None:
            self.ylabel = y
        self._style()

    # ------------------------------------------------------------------ data items
    def line(self, key, x, y=None, color=NAVY, width=2.0, style="-", name=None, alpha=1.0,
             fill=None, fill_alpha=0.15, step=False, z=0, downsample=None):
        """A curve. ``fill`` = baseline level to shade down to (e.g. 0 or -100 dB).
        ``step=True`` draws a staircase (x one longer than y). NaN breaks the line.

        ``downsample``: None (default) = automatic peak decimation for curves longer than
        4000 points whose x is increasing; False = never (parametric curves such as
        circles, trajectories, Lissajous figures); True = always."""
        if y is None:
            x, y = np.arange(len(x)), x
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.logy:
            y = np.where(y > 0, y, np.nan)
        it = self._get(key, "line", lambda: pg.PlotDataItem(connect="finite"))
        kw = dict(pen=self._pen(color, width, style, alpha))
        if fill is not None:
            kw.update(fillLevel=self._y(fill) if self.logy else fill,
                      brush=_qcolor(self.theme.c(color), fill_alpha))
        else:
            kw.update(fillLevel=None, brush=None)
        it.setData(x, y, stepMode="center" if step else None, **kw)
        if downsample is None:
            ds = len(x) > 4000 and bool(np.all(np.diff(x[np.isfinite(x)]) >= 0))
        else:
            ds = bool(downsample)
        if ds:
            it.setDownsampling(auto=True, method="peak")
            it.setClipToView(True)
        elif getattr(it, "_studio_ds", False):
            it.setDownsampling(ds=1, auto=False)
            it.setClipToView(False)
        it._studio_ds = ds
        it.setZValue(z)
        self._legend_set(key, it, name)
        return it

    def scatter(self, key, x, y, color=NAVY, size=6, symbol="o", name=None, alpha=0.85,
                outline=None, z=1):
        """Markers. symbol: o s t d + x star. ``outline`` gives hollow markers that colour."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.logy:
            y = np.where(y > 0, y, np.nan)
        it = self._get(key, "scatter", lambda: pg.PlotDataItem(pen=None))
        c = self.theme.c(color)
        if outline:
            brush = _qcolor(self.theme.plot_bg, 1.0)
            spen = pg.mkPen(_qcolor(self.theme.c(outline), alpha), width=1.6)
        else:
            brush = _qcolor(c, alpha)
            spen = None
        it.setData(x, y, pen=None, symbol=symbol, symbolSize=size, symbolBrush=brush,
                   symbolPen=spen)
        it.setZValue(z)
        self._legend_set(key, it, name)
        return it

    def stems(self, key, x, y, color=NAVY, base=0.0, width=1.6, size=7, name=None, z=1):
        """A stem plot (lines from ``base`` to each point, with a marker on top)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xs = np.repeat(x, 3)
        ys = np.empty(3 * len(x))
        ys[0::3], ys[1::3], ys[2::3] = base, y, np.nan
        self.line(key + "#stem", xs, ys, color=color, width=width, z=z)
        return self.scatter(key, x, y, color=color, size=size, name=name, z=z + 0.1)

    def bars(self, key, x, height, width=0.8, color=NAVY, name=None, base=0.0, alpha=0.9,
             colors=None):
        """Vertical bars from ``base`` up to ``height``. ``colors`` (list) colours each bar
        individually (the legend swatch uses ``color``, or the first of ``colors``).
        Works on log axes: on a log y axis a base ≤ 0 means "from the bottom of the view"."""
        x = np.asarray(x, float)
        h = np.asarray(height, float)
        b = np.broadcast_to(np.asarray(base, float), h.shape).astype(float)
        if colors is not None:
            brushes = [_qcolor(self.theme.c(c), alpha) for c in colors]
        else:
            brushes = [_qcolor(self.theme.c(color), alpha)] * len(x)
        w = np.broadcast_to(np.asarray(width, float), x.shape).astype(float)
        if self.logx:                           # bar edges in log10 units
            lo = np.log10(np.maximum(x - w / 2, 1e-300))
            hi = np.log10(np.maximum(x + w / 2, 1e-300))
            bad = x - w / 2 <= 0                 # e.g. width wider than x: keep it visible
            lo = np.where(bad, np.log10(np.maximum(x, 1e-300)) - 0.05, lo)
            x, w = (lo + hi) / 2, hi - lo
        if self.logy:
            pos = h[np.isfinite(h) & (h > 0)]
            if self.ylim is not None:
                floor = np.log10(self.ylim[0])
            else:
                floor = np.log10(pos.min()) - 1 if pos.size else 0.0
            yb = np.where(b > 0, np.log10(np.maximum(b, 1e-300)), floor)
            yt = np.where(h > 0, np.log10(np.maximum(h, 1e-300)), yb)
            b, h = yb, yt
        it = self._get(key, "bars", lambda: pg.BarGraphItem(x=x, height=h - b, width=w))
        legend_brush = brushes[0] if (colors is not None and len(brushes)) else \
            _qcolor(self.theme.c(color), alpha)
        it.setOpts(x=x, height=h - b, y0=b, width=w, brushes=brushes, brush=legend_brush,
                   pen=pg.mkPen(None))
        self._legend_set(key, it, name)
        return it

    def legend_swatch(self, name, color=NAVY, kind="box", style="-", width=2.0, symbol="o",
                      key=None):
        """A legend-only entry (nothing is drawn in the plot): a coloured box (``kind="box"``),
        a line (``"line"``, with ``style``/``width``) or a marker (``"marker"``, with
        ``symbol``). Use it to explain colours of bars, cells or images. Like every item it
        must be re-issued on each update (keyed by ``key`` or the name)."""
        k = key or ("#swatch:" + name)
        it = self._get(k, "swatch", lambda: pg.PlotDataItem([], []))
        c = self.theme.c(color)
        if kind == "line":
            it.setData([], [], pen=self._pen(color, width, style), symbol=None)
        else:
            sym = "s" if kind == "box" else symbol
            it.setData([], [], pen=None, symbol=sym, symbolSize=11 if kind == "box" else 9,
                       symbolBrush=_qcolor(c, 0.9), symbolPen=pg.mkPen(None))
        self._legend_set(k, it, name)
        return it

    def fill_between(self, key, x, y1, y2, color=NAVY, alpha=0.18, name=None):
        """Shade the area between two curves (scalars broadcast). NaN gaps split the shading.
        Drawn as one plain polygon per finite run, so long zero-area stretches are safe."""
        x = np.asarray(x, float)
        y1 = np.broadcast_to(np.asarray(y1, float), x.shape)
        y2 = np.broadcast_to(np.asarray(y2, float), x.shape)
        if self.logx:
            x = np.log10(np.where(x > 0, x, np.nan))
        if self.logy:
            y1 = np.log10(np.where(y1 > 0, y1, np.nan))
            y2 = np.log10(np.where(y2 > 0, y2, np.nan))
        ok = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
        path = QtGui.QPainterPath()
        idx = np.flatnonzero(ok)
        if idx.size:
            xs_, ys_, cn_ = [], [], []
            for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if r.size < 2:
                    continue
                xs_ += [x[r], x[r][::-1]]
                ys_ += [y2[r], y1[r][::-1]]
                c = np.ones(2 * r.size, bool)
                c[-1] = False                   # each run is its own (implicitly closed) subpath
                cn_.append(c)
            if xs_:
                path = pg.arrayToQPath(np.concatenate(xs_), np.concatenate(ys_),
                                       connect=np.concatenate(cn_))
        it = self._get(key, "fill", lambda: _PathItem())
        it.setPath(path)
        it.setBrush(_qcolor(self.theme.c(color), alpha))
        it.setPen(pg.mkPen(None))
        it.setZValue(-5)
        if name:
            self.legend_swatch(name, color, key=key + "#legend")
        return it

    def handles(self, key, x, y, color=None, size=13, symbol="o", axis="y", name=None):
        """Draggable markers. When the user drags point i, the experiment's
        ``on_drag(plot_key, key, i, x, y)`` is called with the new position and the
        experiment is refreshed (your update then redraws the handles where they belong).

        axis  "y": move vertically only (bar heights, levels) · "x": horizontally only
              · "both": anywhere
        The cursor changes over a handle; the plot does not pan while one is dragged."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        if self.logx:
            x = np.log10(np.where(x > 0, x, np.nan))
        if self.logy:
            y = np.log10(np.where(y > 0, y, np.nan))
        it = self._get(key, "handles", lambda: _Handles())
        c = self.theme.c(color) if color else self.theme.accent
        it.setData(x=x, y=y, symbol=symbol, size=size, brush=pg.mkBrush(_qcolor(c, 0.95)),
                   pen=pg.mkPen(self.theme.plot_bg, width=2))
        it._studio = (self, key, axis)
        it.setZValue(30)
        self._legend_set(key, it, name)
        return it

    def _on_drag(self, key, i, x, y, phase):
        cb = getattr(self, "_drag_cb", None)
        if self.logx:
            x = 10 ** x
        if self.logy:
            y = 10 ** y
        if cb is not None:
            cb(self, key, i, x, y, phase)

    # ------------------------------------------------------------------ annotations
    def vline(self, key, x, color=GRAY, style="--", width=1.2, label=None, label_pos=0.92):
        """A vertical marker line, optionally labelled near the top."""
        it = self._get(key, "vline", lambda: pg.InfiniteLine(angle=90, movable=False),
                       ignore_bounds=True)
        it.setPen(self._pen(color, width, style))
        it.setPos(self._x(x))
        self._line_label(it, label, color, label_pos)
        return it

    def hline(self, key, y, color=GRAY, style="--", width=1.2, label=None, label_pos=0.05):
        """A horizontal marker line, optionally labelled near the left."""
        it = self._get(key, "hline", lambda: pg.InfiniteLine(angle=0, movable=False),
                       ignore_bounds=True)
        it.setPen(self._pen(color, width, style))
        it.setPos(self._y(y))
        self._line_label(it, label, color, label_pos)
        return it

    def _line_label(self, it, label, color, pos):
        if label:
            if getattr(it, "_studio_label", None) is None:
                it._studio_label = pg.InfLineLabel(it, text=label, position=pos,
                                                   anchors=[(0, 0), (0, 0)])
            lab = it._studio_label
            lab.format = label.replace("{", "{{").replace("}", "}}")   # re-rendered on moves
            lab.setText(label)
            lab.setColor(self.theme.c(color))
            lab.setVisible(True)
            f = QtGui.QFont(UI_FONT)
            f.setPointSizeF(8.5)
            lab.setFont(f)
        elif getattr(it, "_studio_label", None) is not None:
            it._studio_label.setVisible(False)

    def band(self, key, x0, x1, color=GREEN, alpha=0.12, label=None):
        """A shaded vertical band from x0 to x1 (e.g. occupied bandwidth)."""
        it = self._get(key, "band", lambda: pg.LinearRegionItem(orientation="vertical",
                                                                movable=False),
                       ignore_bounds=True)
        it.setRegion((self._x(x0), self._x(x1)))
        it.setBrush(_qcolor(self.theme.c(color), alpha))
        for ln in it.lines:
            ln.setPen(pg.mkPen(_qcolor(self.theme.c(color), min(1, alpha * 4)), width=1))
        it.setZValue(-10)
        if label:
            self.text(key + "#lab", (x0 + x1) / 2, None, label, color=color, anchor=(0.5, 0))
        return it

    def hband(self, key, y0, y1, color=GREEN, alpha=0.12):
        """A shaded horizontal band from y0 to y1."""
        it = self._get(key, "hband", lambda: pg.LinearRegionItem(orientation="horizontal",
                                                                 movable=False),
                       ignore_bounds=True)
        it.setRegion((self._y(y0), self._y(y1)))
        it.setBrush(_qcolor(self.theme.c(color), alpha))
        for ln in it.lines:
            ln.setPen(pg.mkPen(None))
        it.setZValue(-10)
        return it

    def text(self, key, x, y, text, color=None, anchor=(0, 1), size=9.5, bold=False,
             html=False, fill=False):
        """A text label at data coordinates (x, y). ``y=None`` pins it to the top of the view.
        anchor (0,1) puts the text's bottom-left corner at the point; (0.5,0.5) centres it."""
        color = self.theme.c(color) if color else self.theme.text
        it = self._get(key, "text", lambda: pg.TextItem(anchor=anchor), ignore_bounds=True)
        it.setAnchor(pg.Point(*anchor))
        weight = "600" if bold else "normal"
        body = text if html else (str(text).replace("&", "&amp;").replace("<", "&lt;")
                                  .replace(">", "&gt;").replace("\n", "<br>"))
        bg = f"background-color:{_qcolor(self.theme.card, 1).name()};" if fill else ""
        it.setHtml(f'<div style="color:{color}; font-size:{size}pt; font-weight:{weight};'
                   f' font-family:{UI_FONT}; {bg}">{body}</div>')
        if y is None:
            (_, (ylo, yhi)) = self.vb.viewRange()
            yy = yhi - 0.04 * (yhi - ylo)
            it._studio_top = True
        else:
            yy = self._y(y)
            it._studio_top = False
        it._studio_x = self._x(x)
        it.setPos(it._studio_x, yy)
        it.setZValue(20)
        return it

    def _repin_top(self):
        """Keep text(y=None) labels at the top of the view when the y range changes
        (set_ylim, autoscale, pan/zoom)."""
        if self.widget is None or not getattr(self, "_items", None):
            return
        (_, (ylo, yhi)) = self.vb.viewRange()
        yy = yhi - 0.04 * (yhi - ylo)
        for it in self._items.values():
            if getattr(it, "_studio_top", False):
                it.setPos(it._studio_x, yy)

    def arrow(self, key, x, y, text="", color=None, direction="down", size=12):
        """An arrow whose tip is at (x, y), with an optional label at its tail.
        direction: where the arrow points ("down", "up", "left", "right")."""
        color = self.theme.c(color) if color else self.theme.text
        angle = {"down": -90, "up": 90, "left": 0, "right": 180}[direction]
        it = self._get(key, "arrow", lambda: pg.ArrowItem(), ignore_bounds=True)
        it.setStyle(angle=angle, headLen=size, tailLen=size * 1.6, tailWidth=2,
                    brush=pg.mkBrush(color), pen=pg.mkPen(color))
        it.setPos(self._x(x), self._y(y))
        it.setZValue(20)
        if text:
            anchor = {"down": (0.5, 1.0), "up": (0.5, 0.0), "left": (0.0, 0.5),
                      "right": (1.0, 0.5)}[direction]
            lab = self._get(key + "#lab", "text", lambda: pg.TextItem(anchor=anchor),
                            ignore_bounds=True)
            lab.setAnchor(pg.Point(*anchor))
            lab.setHtml(f'<div style="color:{color}; font-size:9pt; font-family:{UI_FONT}">'
                        f'{text}</div>')
            # place label at the tail end, in pixel units converted to view units
            px, py = self.vb.viewPixelSize()
            off = (size * 2.8)
            dx, dy = {"down": (0, off * py), "up": (0, -off * py), "left": (off * px, 0),
                      "right": (-off * px, 0)}[direction]
            lab.setPos(self._x(x) + dx, self._y(y) + dy)
            lab.setZValue(20)
        return it

    # ------------------------------------------------------------------ images
    def image(self, key, data, x=(0, 1), y=(0, 1), cmap="heat", levels=None, colorbar=False,
              cbar_label=""):
        """A heat map. ``data[row, col]`` with rows along y (bottom to top) and columns along
        x; x and y give the extent. cmap: "heat", "eye", or a list of colour stops."""
        data = np.asarray(data, float)
        it = self._get(key, "image", lambda: pg.ImageItem(axisOrder="row-major"))
        if levels is None:
            fin = data[np.isfinite(data)]
            levels = (float(fin.min()), float(fin.max())) if fin.size else (0, 1)
            if levels[1] <= levels[0]:
                levels = (levels[0], levels[0] + 1)
        it.setImage(data, autoLevels=False, levels=levels)
        it.setColorMap(self._cmap(cmap))
        it.setRect(QtCore.QRectF(x[0], y[0], x[1] - x[0], y[1] - y[0]))
        it.setZValue(-20)
        if colorbar:
            cb = self._colorbars.get(key)
            if cb is None:
                cb = pg.ColorBarItem(values=levels, colorMap=self._cmap(cmap), interactive=False,
                                     width=12, label=cbar_label)
                cb.setImageItem(it, insert_in=self.pi)
                cb.axis.enableAutoSIPrefix(False)
                self._colorbars[key] = cb
            elif getattr(cb, "_studio_img", None) is not it:     # image recreated (theme)
                cb.setImageItem(it)
                cb.setColorMap(self._cmap(cmap))
            cb._studio_img = it
            cb.setLevels(levels)
        return it

    def _cmap(self, cmap):
        if isinstance(cmap, pg.ColorMap):
            return cmap
        if isinstance(cmap, str):
            stops = {"heat": self.theme.heat_cmap, "eye": self.theme.eye_cmap}.get(cmap, cmap)
        else:
            stops = list(cmap)                  # a list/tuple of colours or (pos, colour) stops
        if isinstance(stops, str):
            try:
                return pg.colormap.get(stops)
            except Exception:
                return pg.colormap.get(stops, source="matplotlib")
        if isinstance(stops[0], (tuple, list)):
            return pg.ColorMap([float(q) for q, _ in stops], [pg.mkColor(c) for _, c in stops])
        return pg.ColorMap(np.linspace(0, 1, len(stops)), [pg.mkColor(s) for s in stops])

    def eye(self, key, y, sps, n_sym=2, offset=0, yrange=(-1.5, 1.5), cols=260, rows=200,
            accumulate=False, decay=0.85, jitter=None, rng=None):
        """Persistence eye diagram of the real waveform ``y`` (``sps`` samples per symbol).

        Traces of ``n_sym`` symbols start at ``offset`` + k·sps. ``jitter`` (rms, in symbol
        periods) shifts each trace horizontally at random, like trigger/sampling jitter.
        With ``accumulate=True`` successive calls build up persistence (older data fades by
        ``decay`` per call), which is what an oscilloscope in persistence mode shows.
        Returns the 2-D histogram (rows × cols)."""
        y = np.asarray(y).real
        L = n_sym * sps + 1
        starts = np.arange(offset, len(y) - L - sps, sps)
        if len(starts) == 0:
            return None
        t = np.arange(L) / sps
        if jitter:
            rng = np.random.default_rng(rng)
            d = rng.standard_normal(len(starts)) * jitter * sps     # in samples
            base = starts[:, None] + np.arange(L)[None, :] + d[:, None]
            base = np.clip(base, 0, len(y) - 2)
            i0 = np.floor(base).astype(int)
            w = base - i0
            traces = y[i0] * (1 - w) + y[i0 + 1] * w
        else:
            traces = y[starts[:, None] + np.arange(L)[None, :]]
        return self.eye_traces(key, traces, t, yrange, cols, rows, accumulate, decay)

    def eye_traces(self, key, traces, t, yrange=(-1.5, 1.5), cols=260, rows=200,
                   accumulate=False, decay=0.85):
        """Persistence image from explicit traces (N × len(t)); t in symbol periods."""
        traces = np.asarray(traces, float)
        t = np.asarray(t, float)
        sub = 6                                          # sub-columns per pixel column
        tc = np.linspace(t[0], t[-1], cols * sub)
        pos = np.interp(tc, t, np.arange(len(t)))
        i0 = np.minimum(np.floor(pos).astype(int), len(t) - 2)
        w = pos - i0
        v = traces[:, i0] * (1 - w) + traces[:, i0 + 1] * w           # (N, cols*sub)
        r = ((v - yrange[0]) / (yrange[1] - yrange[0]) * rows).astype(np.int64)
        c = np.broadcast_to(np.arange(cols * sub) // sub, v.shape)
        ok = (r >= 0) & (r < rows)
        H = np.bincount(r[ok] * cols + c[ok], minlength=rows * cols).reshape(rows, cols)
        H = H.astype(float)
        acc = self._eye_acc.get(key)
        if accumulate and acc is not None and acc.shape == H.shape:
            H = acc * decay + H
        self._eye_acc[key] = H
        # light vertical blur (hides row quantisation), then a phosphor-like log response
        Hb = H.copy()
        Hb[1:-1, :] += 0.5 * (H[:-2, :] + H[2:, :])
        img = np.log1p(Hb)
        ref = np.percentile(img[img > 0], 99.8) if np.any(img > 0) else 1.0
        img = np.clip(img / max(ref, 1e-12), 0, 1) ** 1.3
        self.image(key, img, x=(t[0], t[-1]), y=yrange, cmap="eye", levels=(0, 1))
        return H

    def reset_persistence(self, key=None):
        """Forget accumulated eye persistence (all eyes, or one)."""
        if key is None:
            self._eye_acc.clear()
        else:
            self._eye_acc.pop(key, None)

    # ------------------------------------------------------------------ spectra
    def psd(self, key, x, fs=1.0, nfft=1024, color=NAVY, name=None, normalize=True,
            scale=1.0, width=1.6, onesided=None, fill=None):
        """Welch power spectral density in dB (via commlib.welch_psd).

        normalize=True puts the peak at 0 dB. ``scale`` divides the frequency axis (e.g.
        1e3 to plot in kHz). Real signals are shown one-sided unless onesided=False.
        Returns (f, psd_db)."""
        import commlib as cl
        x = np.asarray(x)
        f, p = cl.welch_psd(x, fs, nfft)
        if onesided is None:
            onesided = not np.iscomplexobj(x)
        if onesided:
            k = f >= 0
            f, p = f[k], p[k]
        if normalize:
            p = p - p.max()
        self.line(key, f / scale, p, color=color, name=name, width=width, fill=fill)
        return f, p

    # ------------------------------------------------------------------ theory vs simulation
    def theory(self, key, x, y, color=NAVY, name=None, style="-", width=2.0):
        """A theory curve (a line). Available on every plot, not only BERPlot."""
        return self.line(key, x, y, color=color, name=name, style=style, width=width)

    def sim(self, key, x, y, color=PALETTE[1], name=None, size=9, connect=True):
        """Simulated points: hollow markers, optionally joined by a dotted line."""
        if connect:
            self.line(key + "#ln", x, y, color=color, width=1.0, style=":")
        return self.scatter(key, x, y, color=color, size=size, name=name, outline=color)


class _Handles(pg.ScatterPlotItem):
    """Scatter points that can be dragged with the left mouse button (Plot.handles)."""

    def __init__(self):
        super().__init__()
        self._studio = None
        self._drag = None
        self.setAcceptHoverEvents(True)

    def _cursor(self):
        axis = self._studio[2] if self._studio else "both"
        return {"y": QtCore.Qt.SizeVerCursor, "x": QtCore.Qt.SizeHorCursor}.get(
            axis, QtCore.Qt.SizeAllCursor)

    def hoverEvent(self, ev):                                   # noqa: N802 (pyqtgraph API)
        if ev.isExit():
            self.unsetCursor()
            return
        if len(self.pointsAt(ev.pos())):
            ev.acceptDrags(QtCore.Qt.LeftButton)
            self.setCursor(self._cursor())
        else:
            self.unsetCursor()

    def mouseDragEvent(self, ev):                               # noqa: N802
        if ev.button() != QtCore.Qt.LeftButton or self._studio is None:
            ev.ignore()
            return
        if ev.isStart():
            pts = self.pointsAt(ev.buttonDownPos())
            if not len(pts):
                ev.ignore()
                return
            p0 = pts[0].pos()
            self._drag = (pts[0].index(), p0.x(), p0.y())
        if self._drag is None:
            ev.ignore()
            return
        ev.accept()
        plot, key, axis = self._studio
        i, x0, y0 = self._drag
        pos = ev.pos()
        x = x0 if axis == "y" else pos.x()
        y = y0 if axis == "x" else pos.y()
        phase = "start" if ev.isStart() else "finish" if ev.isFinish() else "move"
        if ev.isFinish():
            self._drag = None
        try:
            plot._on_drag(key, i, x, y, phase)
        except Exception:                                       # never crash on a drag
            import traceback
            traceback.print_exc()


class _PathItem(QtWidgets.QGraphicsPathItem):
    """A filled path in data coordinates (fill_between). Bounds feed autoscaling."""

    def dataBounds(self, axis, frac=1.0, orthoRange=None):      # noqa: N802 (pyqtgraph API)
        r = self.path().boundingRect()
        if r.isNull() and r.width() == 0 and r.height() == 0:
            return None
        return (r.left(), r.right()) if axis == 0 else (r.top(), r.bottom())

    def pixelPadding(self):                                     # noqa: N802
        return 0


# ----------------------------------------------------------------------------- presets
def TimePlot(key, title="", x="time", y="amplitude", **kw):
    """An x–y plot with time-domain defaults."""
    return Plot(key, title, x=x, y=y, **kw)


def SpectrumPlot(key, title="", x="frequency", y="power (dB)", ylim=(-80, 5), **kw):
    """An x–y plot with spectrum defaults (dB scale, -80..5 dB)."""
    return Plot(key, title, x=x, y=y, ylim=ylim, **kw)


def BarPlot(key, title="", x="", y="", **kw):
    """An x–y plot meant for bars (use ``.bars``)."""
    kw.setdefault("legend", None)
    return Plot(key, title, x=x, y=y, **kw)


class ConstellationPlot(Plot):
    """Square I/Q scatter with ±lim axes. Use ``.points(key, z)`` and ``.ideal(key, pts)``."""

    def __init__(self, key, title="", lim=1.5, **kw):
        kw.setdefault("x", "in-phase I")
        kw.setdefault("y", "quadrature Q")
        kw.setdefault("legend", None)
        super().__init__(key, title, xlim=(-lim, lim), ylim=(-lim, lim), aspect=True, **kw)

    def points(self, key, z, color=NAVY, size=3, alpha=0.45, name=None):
        z = np.asarray(z)
        return self.scatter(key, z.real, z.imag, color=color, size=size, alpha=alpha, name=name)

    def ideal(self, key, pts, color=None, size=11):
        pts = np.asarray(pts)
        return self.scatter(key, pts.real, pts.imag, color=color or self.theme.text, size=size,
                            symbol="+", alpha=1.0, z=5)


class EyePlot(Plot):
    """An eye-diagram canvas: time in symbol periods, fixed vertical range."""

    def __init__(self, key, title="", yrange=(-1.5, 1.5), n_sym=2, **kw):
        kw.setdefault("x", "time (symbol periods)")
        kw.setdefault("y", "amplitude")
        kw.setdefault("legend", None)
        kw.setdefault("grid", False)
        super().__init__(key, title, xlim=(0, n_sym), ylim=yrange, **kw)
        self.yrange, self.n_sym = yrange, n_sym

    def _extra_build(self):
        self.set_xticks([(v, f"{v:g}") for v in np.arange(0, self.n_sym + 0.01, 0.5)])


class BERPlot(Plot):
    """Error-rate plot with a logarithmic y axis. ``.theory`` draws a curve, ``.sim``
    draws Monte Carlo points (zeros are simply not drawn)."""

    def __init__(self, key, title="", x="Eb/N0 (dB)", y="bit error rate", ylim=(1e-6, 0.5),
                 **kw):
        super().__init__(key, title, x=x, y=y, ylim=ylim, logy=True, **kw)


class PolarPlot(Plot):
    """Polar gain pattern in dB. Angles are measured from broadside (straight up) and
    increase clockwise, like an antenna pattern; ``floor_db`` is the centre of the plot."""

    def __init__(self, key, title="", floor_db=-40, **kw):
        kw.setdefault("legend", "tr")
        super().__init__(key, title, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect=True,
                         grid=False, **kw)
        self.floor_db = floor_db

    def _extra_build(self):
        for side in ("left", "bottom"):
            self.pi.hideAxis(side)
        pen = pg.mkPen(_qcolor(self.theme.axis, 0.25), width=1)
        th = np.linspace(0, 2 * np.pi, 361)
        n = 4
        for i in range(1, n + 1):
            r = i / n
            c = pg.PlotCurveItem(r * np.sin(th), r * np.cos(th), pen=pen)
            self.pi.addItem(c)
            db = self.floor_db * (1 - r)
            lab = pg.TextItem(f"{db:.0f} dB", color=self.theme.muted, anchor=(0, 1))
            lab.setPos(0.02, r)
            self.pi.addItem(lab)
        for a in range(0, 360, 30):
            ar = np.deg2rad(a)
            self.pi.addItem(pg.PlotCurveItem([0, np.sin(ar)], [0, np.cos(ar)], pen=pen))
            lab = pg.TextItem(f"{a if a <= 180 else a - 360}°", color=self.theme.muted,
                              anchor=(0.5, 0.5))
            lab.setPos(1.08 * np.sin(ar), 1.08 * np.cos(ar))
            self.pi.addItem(lab)

    def _restyle(self, theme):
        self.theme = theme
        self.pi.clear()
        self._items, self._kinds, self._names = {}, {}, {}
        if self.legend is not None:
            self.legend.clear()
        self._style()
        self._extra_build()

    def pattern(self, key, theta, gain_db, color=NAVY, name=None, width=2.0, fill=False):
        """Draw a pattern: theta in radians from broadside, gain in dB (0 dB = peak)."""
        g = np.asarray(gain_db, float)
        r = np.clip((g - self.floor_db) / (-self.floor_db), 0, None)
        th = np.asarray(theta, float)
        return self.line(key, r * np.sin(th), r * np.cos(th), color=color, name=name,
                         width=width, fill=0 if fill else None)


class Canvas(Plot):
    """A drawing surface for diagrams (shift registers, trellises, Venn diagrams, maps of
    cells): fixed ranges, no axes, no grid, no pan/zoom. Combine with ``on_click``."""

    def __init__(self, key, title="", xlim=(0, 1), ylim=(0, 1), aspect=False, **kw):
        kw.setdefault("legend", None)
        kw.setdefault("x", "")
        kw.setdefault("y", "")
        super().__init__(key, title, xlim=xlim, ylim=ylim, aspect=aspect, axes=False, **kw)


class ImagePlot(Plot):
    """A heat-map canvas; use ``.image(key, data, x=(x0, x1), y=(y0, y1))``."""

    def __init__(self, key, title="", x="", y="", **kw):
        kw.setdefault("legend", None)
        kw.setdefault("grid", False)
        super().__init__(key, title, x=x, y=y, **kw)
