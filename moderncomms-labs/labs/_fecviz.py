"""Shared drawing helpers for the coding labs (08, 09, 20, 23).

* ``Canvas``: a Plot with no axes, no grid and no mouse panning, for diagrams (shift
  registers, trellises, Tanner graphs, Venn diagrams, bit grids).
* ``on_click(exp, key, fn)``: call ``fn(x, y)`` (data coordinates) when the user clicks the
  plot ``key``; if it returns True the experiment is refreshed. (The studio framework has no
  click events yet; this is the local stand-in.)
* ``cells(...)``: coloured rectangular cells (bits, symbols) with optional labels.
* ``segments(...)``: many line segments in one plot item.
"""
import numpy as np
from PySide6 import QtCore

from studio import Plot


class Canvas(Plot):
    """A drawing surface: fixed ranges, no axes, no grid, no panning."""

    def __init__(self, key, title="", xlim=(0, 1), ylim=(0, 1), aspect=False, **kw):
        kw.setdefault("legend", None)
        super().__init__(key, title, x="", y="", xlim=xlim, ylim=ylim, aspect=aspect,
                         grid=False, **kw)

    def _extra_build(self):
        for side in ("left", "bottom"):
            self.pi.hideAxis(side)
        self.vb.setMouseEnabled(False, False)
        self.pi.setMenuEnabled(False)


def on_click(exp, key, fn):
    pl = exp.plot(key)

    def handler(ev):
        try:
            if ev.button() != QtCore.Qt.LeftButton:
                return
            pos = ev.scenePos()
            if not pl.vb.sceneBoundingRect().contains(pos):
                return
            pt = pl.vb.mapSceneToView(pos)
            if fn(pt.x(), pt.y()) and exp._window is not None:
                win = exp._window
                if win.cur >= 0 and win.page.exp is exp:
                    win.refresh()
        except Exception:          # a click must never crash the lab
            import traceback
            traceback.print_exc()

    pl.widget.scene().sigMouseClicked.connect(handler)


def cells(pl, key, x, y, colors, w=0.9, h=0.9, labels=None, text_color=None, size=9.5,
          alpha=0.9, bold=False):
    """Rectangles centred at (x[i], y[i]) of size w x h, coloured individually; optional
    text labels drawn in the centre (keys key#0, key#1, ...)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pl.bars(key, x, y + h / 2, width=w, base=y - h / 2, colors=list(colors), alpha=alpha)
    if labels is not None:
        tc = text_color if isinstance(text_color, (list, tuple)) else [text_color] * len(x)
        for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
            pl.text(f"{key}#{i}", xi, yi, str(lab), color=tc[i], anchor=(0.5, 0.5), size=size,
                    bold=bold)


def segments(pl, key, x0, y0, x1, y1, **kw):
    """Draw many separate segments (x0,y0)-(x1,y1) as one item."""
    x0, y0, x1, y1 = (np.atleast_1d(np.asarray(a, float)) for a in (x0, y0, x1, y1))
    n = len(x0)
    if n == 0:
        return None                       # untouched items are hidden by the framework
    xs = np.column_stack([x0, x1, np.full(n, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full(n, np.nan)]).ravel()
    return pl.line(key, xs, ys, **kw)


def pale(pl):
    """Neutral cell colour for the current theme (empty / zero cells)."""
    return pl.theme.border


def ink(pl):
    """Text colour that reads on a filled (palette-coloured) cell."""
    return pl.theme.plot_bg


def circle(pl, key, cx, cy, r, n=120, **kw):
    th = np.linspace(0, 2 * np.pi, n)
    return pl.line(key, cx + r * np.cos(th), cy + r * np.sin(th), **kw)


def bitstr(v):
    return "".join(str(int(b)) for b in v)
