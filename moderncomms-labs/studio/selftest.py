"""Automatic self-test for a lab (``python labNN_name.py --selftest``).

For every experiment it:

1. builds the page and runs ``update`` at the default settings (plus the background job,
   in quick mode, and a few animation frames when the experiment animates);
2. checks that nothing raised, that readouts and plotted data are finite, and times it;
3. saves a screenshot of the whole window to ``tests/screens/labNN/NN_slug.png``;
4. repeats at several random control settings (screenshot of the last one saved as
   ``NN_slug_random.png``) and flags updates slower than 0.5 s (``heavy`` experiments
   are exempt for ``update`` but their background job must finish within 60 s).

Exit status 0 = all good. Works headless with ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from PySide6 import QtWidgets

SLOW_S = 0.5          # interactive update limit
FIRST_S = 2.0         # first update may include one-off setup
BG_BUDGET_S = 60.0

_HERE = os.path.dirname(os.path.abspath(__file__))
SCREENS = os.path.abspath(os.path.join(_HERE, "..", "tests", "screens"))


def _slug(s):
    out = "".join(c if c.isalnum() else "_" for c in s.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:40]


def _finite_report(page):
    """Return a list of problems with non-finite data in readouts and plot items."""
    import pyqtgraph as pg
    probs = []
    for k, v in page.exp.r.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            if not np.isfinite(v):
                probs.append(f"readout {k} = {v}")
    for pk, pl in page.plots.items():
        for ik, it in pl._items.items():
            if not it.isVisible():
                continue
            if isinstance(it, pg.PlotDataItem):
                x, y = it.getData() if it.xData is not None else (None, None)
                if y is None or len(y) == 0:
                    continue
                if np.isinf(x).any() or np.isinf(y).any():
                    probs.append(f"plot {pk}/{ik}: infinite values")
                if not pl.logy and np.isnan(y).all():
                    probs.append(f"plot {pk}/{ik}: all NaN")
            elif isinstance(it, pg.ImageItem) and it.image is not None:
                if not np.isfinite(it.image).all():
                    probs.append(f"plot {pk}/{ik}: non-finite image")
    return probs


def _settle(app, n=3):
    for _ in range(n):
        app.processEvents()


def run_selftest(lab, out_dir=None, n_random=5, dark=False, seed=1):
    os.environ["STUDIO_MUTE"] = "1"
    from .app import MainWindow, make_app
    app = make_app()
    import commlib  # noqa: F401  (warm the heavy imports so timings measure the lab, not imports)
    import scipy.signal  # noqa: F401
    out_dir = out_dir or os.path.join(SCREENS, lab.tag)
    os.makedirs(out_dir, exist_ok=True)
    win = MainWindow(lab, persist=False, quick=True, celebrate=False, raise_errors=True)
    win.sync_background = True
    if dark:
        win.toggle_theme()
    win.resize(1600, 960)
    win.show()
    _settle(app)
    rng = np.random.default_rng(seed)
    rows, n_fail = [], 0
    t_lab = time.perf_counter()
    for i, cls in enumerate(lab.experiments):
        name = f"{i + 1:02d}_{_slug(cls.title)}"
        errs, warns, times = [], [], []
        bg_t = 0.0
        try:
            t0 = time.perf_counter()
            win.select(i)
            first = time.perf_counter() - t0
            page = win.page
            exp = page.exp
            if first > FIRST_S and not exp.heavy:
                errs.append(f"first update took {first:.2f} s")
            bg_t = win.run_background_sync(page, BG_BUDGET_S)
            if bg_t > BG_BUDGET_S:
                errs.append(f"background job exceeded {BG_BUDGET_S:.0f} s")
            if exp.animate:
                for _ in range(4):
                    t0 = time.perf_counter()
                    win._tick()
                    times.append(time.perf_counter() - t0)
            _settle(app)
            errs += _finite_report(page)
            win.grab().save(os.path.join(out_dir, name + ".png"))
            # random settings
            for k in range(n_random):
                for c in page.controls:
                    if c.has_value:
                        c.set_value(c.random_value(rng))
                win._update_enabled(page)
                t0 = time.perf_counter()
                win.refresh(page)
                dt = time.perf_counter() - t0
                times.append(dt)
                bt = win.run_background_sync(page, BG_BUDGET_S)
                bg_t = max(bg_t, bt)
                if exp.animate:
                    t0 = time.perf_counter()
                    win._tick()
                    times.append(time.perf_counter() - t0)
                pr = _finite_report(page)
                if pr:
                    errs.append(f"random setting {k + 1} {dict(exp.p)}: " + "; ".join(pr))
            # buttons
            for c in page.controls:
                if c.__class__.__name__ == "Button":
                    win._on_button(page, c.key)
                    win.run_background_sync(page, BG_BUDGET_S)
            _settle(app)
            win.grab().save(os.path.join(out_dir, name + "_random.png"))
            win.reset_experiment()
            _settle(app)
            slow = [t for t in times if t > SLOW_S]
            if slow and not exp.heavy:
                errs.append(f"{len(slow)} slow updates (max {max(slow):.2f} s)")
            elif slow:
                warns.append(f"heavy: max update {max(slow):.2f} s")
        except Exception:
            errs.append(traceback.format_exc())
        status = "FAIL" if errs else "ok"
        n_fail += bool(errs)
        rows.append((name, status, max(times) if times else 0.0, bg_t, errs, warns))
    win.close()
    total = time.perf_counter() - t_lab
    print(f"\nSelf-test {lab.tag} — {lab.title}  ({total:.1f} s)")
    print(f"{'experiment':44s} {'status':6s} {'max upd':>8s} {'bg':>7s}")
    for name, st, mx, bg, errs, warns in rows:
        print(f"{name:44s} {st:6s} {mx * 1e3:6.0f}ms {bg:6.1f}s")
        for e in errs:
            print("    ERROR:", e.strip().replace("\n", "\n    "))
        for w in warns:
            print("    note:", w)
    print(f"screenshots: {out_dir}")
    print(f"RESULT {lab.tag}: {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({len(rows) - n_fail}/{len(rows)} experiments ok)", flush=True)
    return 0 if n_fail == 0 else 1
