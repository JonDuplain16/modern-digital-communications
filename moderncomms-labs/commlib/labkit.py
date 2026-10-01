"""labkit: the shared look-and-feel and bookkeeping for every lab notebook.

A lab starts with

    import os, sys
    sys.path[:0] = [p for p in (os.path.abspath(".."), os.path.abspath("."))
                    if os.path.isdir(os.path.join(p, "commlib"))]
    import commlib as cl
    from commlib import labkit as lk
    rng = lk.setup(seed=13, lab="13")

and then uses a handful of helpers so all labs look and behave the same:

* figures      ``lk.fig(...)``, ``lk.show(fig)``, size presets that match the book
* panels       ``lk.constellation``, ``lk.eye``, ``lk.psd``, ``lk.ber_plot``
* numbers      ``lk.table(rows, headers)`` renders a tidy results table
* self-checks  ``lk.check(name, value, expected, rtol/atol)`` prints PASS / FAIL
* widgets      ``lk.interact``, ``lk.slider``, ``lk.choice`` (live in Jupyter,
               rendered once at their defaults when the notebook is pre-built)
* Monte Carlo  ``lk.ber_mc`` runs until enough errors are counted
* wrap-up      ``lk.summary()`` prints run time and self-check tally

Everything degrades gracefully: without Jupyter the rich outputs become plain
text, and without ipywidgets an interactive panel runs once at its defaults.
"""
from __future__ import annotations

import html as _html
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__all__ = ["setup", "PALETTE", "NAVY", "RED", "GREEN", "ORANGE", "PURPLE", "BLUE", "GRAY",
           "SIZES", "fig", "show", "constellation", "eye", "eye_density", "psd", "ber_plot",
           "ber_axes", "table", "check", "note", "header", "interact", "slider", "islider",
           "choice", "ber_mc", "db", "undb", "summary", "is_static"]

# ----------------------------------------------------------------------------- style
# Same palette as the book figures (book/figscripts/figstyle.py), so a figure in
# a lab and the matching figure in the text read as one family.
NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY = ("#1B3A5C", "#C0392B", "#1E7B4F", "#C0661A",
                                                "#6C3483", "#2E86C1", "#7F8C8D")
PALETTE = [NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, "#B7950B", "#117A65", "#884EA0"]

# Figure size presets (inches). "wide" suits one full-width panel, "row2"/"row3"
# two or three side-by-side panels, "square" a single constellation.
SIZES = {"wide": (9.0, 3.4), "tall": (9.0, 5.0), "square": (4.4, 4.2), "row2": (10.0, 3.6),
         "row3": (12.5, 3.6), "row4": (14.0, 3.5), "grid4": (10.0, 7.0), "ber": (7.0, 4.4)}

_STATE = {"t0": None, "checks": [], "lab": None, "seed": None}
_HOOK = None           # set by tests/build_notebooks.py to capture rich output in order


def _style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 110, "figure.facecolor": "white",
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titleweight": "normal",
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.framealpha": 0.85,
        "image.cmap": "viridis", "figure.constrained_layout.use": False,
    })


def is_static():
    """True when the notebook is being pre-built (widgets render once at defaults)."""
    return os.environ.get("MDC_STATIC", "") == "1"


def setup(seed=0, lab=None, style=True, quiet=False):
    """Seed a Generator, apply the course plot style and start the lab clock.

    Returns ``np.random.default_rng(seed)``. Every random draw in a lab should use
    this generator (or one derived from it) so results are reproducible.
    """
    _STATE.update(t0=time.time(), checks=[], lab=lab, seed=seed)
    if style:
        _style()
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    if not quiet:
        import scipy
        print(f"Lab {lab or ''} ready | numpy {np.__version__} | scipy {scipy.__version__} | "
              f"matplotlib {matplotlib.__version__} | seed {seed}")
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------- output
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"
    except Exception:  # pragma: no cover
        return False


def _emit(html_str, text):
    if _HOOK is not None:
        _HOOK(html_str, text)
    elif _in_notebook():
        from IPython.display import HTML, display
        display(HTML(html_str))
    else:
        print(text)


def note(text, kind="info"):
    """A small highlighted message ('info', 'good', 'warn')."""
    col = {"info": BLUE, "good": GREEN, "warn": ORANGE}.get(kind, BLUE)
    _emit(f'<div style="border-left:4px solid {col};padding:4px 10px;margin:4px 0;'
          f'background:#f7f9fb">{_html.escape(text)}</div>', f"[{kind}] {text}")


def header(number, title, chapter, minutes=None):
    """Compact banner, e.g. for the lab index. Labs use a Markdown title cell."""
    sub = f"Companion to {chapter}" + (f" · about {minutes} min" if minutes else "")
    _emit(f'<div style="background:{NAVY};color:white;padding:10px 14px;border-radius:6px">'
          f'<b style="font-size:1.25em">Lab {_html.escape(str(number))} — {_html.escape(title)}</b>'
          f'<br><span style="opacity:.85">{_html.escape(sub)}</span></div>',
          f"=== Lab {number} — {title} ({sub}) ===")


def table(rows, headers=None, title=None, fmt=None, align=None):
    """Render a results table. rows: list of sequences. fmt: dict col-> format spec
    or a single format applied to floats (default '.4g')."""
    rows = [list(r) for r in rows]
    ncol = max(len(r) for r in rows) if rows else len(headers or [])

    def cell(v, j):
        spec = fmt.get(j, fmt.get(headers[j] if headers else None)) if isinstance(fmt, dict) else fmt
        if isinstance(v, (float, np.floating)):
            return format(v, spec or ".4g")
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and spec and isinstance(fmt, dict):
            return format(v, spec)                  # only an explicit per-column format touches ints
        return str(v)

    srows = [[cell(v, j) for j, v in enumerate(r)] for r in rows]
    hdr = [str(h) for h in headers] if headers else None
    widths = [max([len(hdr[j]) if hdr else 0] + [len(r[j]) for r in srows if j < len(r)]) for j in range(ncol)]
    lines = []
    if title:
        lines.append(title)
    if hdr:
        lines.append("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
        lines.append("  ".join("-" * w for w in widths))
    for r in srows:
        lines.append("  ".join(v.rjust(w) for v, w in zip(r, widths)))
    th = "".join(f'<th style="padding:3px 10px;text-align:right;border-bottom:2px solid {NAVY}">'
                 f'{_html.escape(h)}</th>' for h in hdr) if hdr else ""
    trs = "".join("<tr>" + "".join(f'<td style="padding:2px 10px;text-align:right">{_html.escape(v)}</td>'
                                   for v in r) + "</tr>" for r in srows)
    cap = (f'<caption style="caption-side:top;text-align:left;font-weight:bold;color:{NAVY}">'
           f'{_html.escape(title)}</caption>') if title else ""
    _emit(f'<table style="border-collapse:collapse;font-family:monospace;font-size:0.92em">{cap}'
          f'<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>', "\n".join(lines))


def check(name, value, expected=None, rtol=0.05, atol=0.0, hint=None, cond=None):
    """Self-check for an exercise.

    * ``value is None`` -> "not attempted" (the exercise scaffold starts that way).
    * ``cond`` given    -> PASS iff bool(cond).
    * otherwise         -> PASS iff |value - expected| <= atol + rtol*|expected|
      (element-wise for arrays).
    """
    if value is None and cond is None:
        status, detail = "TODO", "not attempted yet: replace the None with your answer"
    else:
        if cond is not None:
            ok = bool(cond)
            detail = "" if value is None else f"value = {_fmt(value)}"
        else:
            v = np.asarray(value, dtype=complex if np.iscomplexobj(value) else float)
            e = np.asarray(expected, dtype=v.dtype)
            ok = bool(np.all(np.abs(v - e) <= atol + rtol * np.abs(e)))
            detail = f"your value {_fmt(value)}" + ("" if ok else f", expected about {_fmt(expected)}")
        status = "PASS" if ok else "FAIL"
    if status == "FAIL" and hint:
        detail += f"  | hint: {hint}"
    _STATE["checks"].append((name, status))
    col = {"PASS": GREEN, "FAIL": RED, "TODO": GRAY}[status]
    sym = {"PASS": "&#10004;", "FAIL": "&#10008;", "TODO": "&#9744;"}[status]
    _emit(f'<div style="font-family:monospace"><b style="color:{col}">{sym} {status}</b> '
          f'<b>{_html.escape(name)}</b> <span style="color:#555">{_html.escape(detail)}</span></div>',
          f"[{status}] {name}: {detail}")
    return status == "PASS"


def _fmt(v):
    a = np.asarray(v)
    if a.ndim == 0:
        x = a.item()
        return f"{x:.5g}" if isinstance(x, (float, complex)) and not isinstance(x, bool) else str(x)
    return np.array2string(a, precision=4, threshold=8)


def summary():
    """Print elapsed time and the self-check tally. Call at the end of a lab."""
    dt = time.time() - (_STATE["t0"] or time.time())
    c = [s for _, s in _STATE["checks"]]
    msg = (f"Lab {_STATE['lab'] or ''} finished in {dt:.1f} s. Self-checks: {c.count('PASS')} passed, "
           f"{c.count('FAIL')} failed, {c.count('TODO')} still to do.")
    note(msg, "good" if c.count("FAIL") == 0 else "warn")


# ----------------------------------------------------------------------------- figures
def fig(size="wide", nrows=1, ncols=1, **kw):
    """plt.subplots with a named size preset (see SIZES) or an explicit (w, h)."""
    figsize = SIZES.get(size, size) if isinstance(size, str) else size
    f, ax = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return f, ax


def show(f=None, title=None):
    """tight_layout + optional suptitle + plt.show()."""
    f = f or plt.gcf()
    if title:
        f.suptitle(title, fontsize=11.5, color=NAVY, fontweight="bold")
    f.tight_layout()
    plt.show()


def db(x, power=True):
    """10 log10 (power) or 20 log10 (amplitude) with a floor to avoid -inf."""
    x = np.maximum(np.abs(np.asarray(x, dtype=float if not np.iscomplexobj(x) else complex)), 1e-30)
    return (10 if power else 20) * np.log10(x)


def undb(x_db):
    return 10 ** (np.asarray(x_db) / 10)


def constellation(ax, y, ref=None, title=None, lim=None, n_max=5000, s=3, alpha=0.35, color=None):
    """Scatter of received samples with the ideal points as black crosses."""
    y = np.asarray(y).ravel()[:n_max]
    ax.scatter(y.real, y.imag, s=s, alpha=alpha, color=color or NAVY, edgecolors="none", rasterized=True)
    if ref is not None:
        ax.scatter(np.real(ref), np.imag(ref), marker="+", c=RED, s=45, linewidths=1.3, zorder=3)
    if lim is None:
        m = np.percentile(np.abs(np.concatenate([y.real, y.imag])), 99.5) if len(y) else 1
        lim = 1.25 * max(m, np.max(np.abs(ref)) if ref is not None else 0, 0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4); ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    if title:
        ax.set_title(title)
    return ax


def eye(ax, x, sps, n_sym=2, offset=0, n_traces=250, component="real", color=None, alpha=0.12):
    """Line-overlay eye diagram of a real/complex waveform (n_sym symbols wide)."""
    x = np.asarray(x)
    x = getattr(x, component) if np.iscomplexobj(x) else x
    L = n_sym * sps + 1
    starts = np.arange(offset, len(x) - L, sps)[:n_traces]
    tr = np.array([x[s:s + L] for s in starts])
    t = np.arange(L) / sps
    ax.plot(t, tr.T, color=color or NAVY, alpha=alpha, lw=0.8)
    ax.set_xlabel("Time (symbol periods)"); ax.set_ylabel("Amplitude")
    ax.set_xlim(0, n_sym)
    return tr


def eye_density(ax, x, sps, n_sym=2, offset=0, bins=(200, 160), ylim=None, cmap="magma", upsample=8):
    """Persistence-style (2-D histogram) eye, like a sampling scope. Traces are
    interpolated `upsample`x so the density is smooth even at low sps."""
    from scipy.signal import resample_poly
    x = np.real(np.asarray(x))
    xu = resample_poly(x, upsample, 1) if upsample > 1 else x
    spu = sps * upsample
    L = n_sym * spu + 1
    starts = np.arange(offset * upsample, len(xu) - L, spu)
    tr = np.array([xu[s:s + L] for s in starts])
    t = np.tile(np.arange(L) / spu, len(tr))
    lim = ylim or (1.15 * np.percentile(np.abs(tr), 99.8))
    # one time bin per (upsampled) sample, centred on the sample, so no empty columns
    half = 0.5 / spu
    H, xe, ye = np.histogram2d(t, tr.ravel(), bins=(L, bins[1]),
                               range=[[-half, n_sym + half], [-lim, lim]])
    ax.imshow(np.log1p(H.T), origin="lower", aspect="auto", cmap=cmap,
              extent=[0, n_sym, -lim, lim], interpolation="bilinear")
    ax.grid(False)
    ax.set_xlabel("Time (symbol periods)"); ax.set_ylabel("Amplitude")
    return H


def psd(ax, x, fs=1.0, nfft=2048, label=None, normalize=True, scale=1.0, unit="", color=None,
        lw=1.3, **kw):
    """Welch PSD (two-sided for complex, one-sided for real input) in dB.

    scale divides the frequency axis (e.g. 1e3 with unit="kHz")."""
    from scipy.signal import welch
    x = np.asarray(x)
    cplx = np.iscomplexobj(x)
    f, p = welch(x, fs=fs, nperseg=min(nfft, len(x)), return_onesided=not cplx, scaling="density",
                 window="blackmanharris", noverlap=min(nfft, len(x)) // 2, detrend=False)
    if cplx:
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    pdb = 10 * np.log10(p + 1e-30)
    if normalize:
        pdb -= pdb.max()
    ax.plot(f / scale, pdb, label=label, color=color, lw=lw, **kw)
    ax.set_xlabel(f"Frequency ({unit})" if unit else ("Frequency (cycles/sample)" if fs == 1.0 else "Frequency (Hz)"))
    ax.set_ylabel("PSD (dB, rel. peak)" if normalize else "PSD (dB/Hz)")
    return f, pdb


def ber_axes(ax, xlabel=r"$E_b/N_0$ (dB)", ylabel="Bit error rate", ylim=(1e-6, 0.5)):
    ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.12)
    if ylim:
        ax.set_ylim(*ylim)


def ber_plot(ax, x, sims=None, theory=None, xlabel=r"$E_b/N_0$ (dB)", ylabel="Bit error rate",
             ylim=(1e-6, 0.5), legend=True, x_theory=None):
    """Monte Carlo points (markers) and closed forms (lines) sharing one colour
    per curve. sims/theory: dict label -> y values. A theory entry whose label
    also appears in sims uses the same colour. Zero-error points are dropped
    (a BER of 0 has no place on a log axis)."""
    sims, theory = sims or {}, theory or {}
    labels = list(dict.fromkeys(list(sims) + list(theory)))
    markers = "osD^v<>ph*"
    xt = x if x_theory is None else x_theory
    for i, lab in enumerate(labels):
        col = PALETTE[i % len(PALETTE)]
        if lab in theory:
            ax.plot(xt, theory[lab], "-", color=col, lw=1.4,
                    label=f"{lab} (theory)" if lab in sims else lab)
        if lab in sims:
            y = np.asarray(sims[lab], dtype=float)
            y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, markers[i % len(markers)], color=col, ms=5, mfc="white" if lab in theory else col,
                    mew=1.3, ls="none" if lab in theory else "--", label=f"{lab} (sim)" if lab in theory else lab)
    ber_axes(ax, xlabel, ylabel, ylim)
    if legend:
        ax.legend(fontsize=8)
    return ax


# ----------------------------------------------------------------------------- Monte Carlo
def ber_mc(trial, min_errors=100, max_bits=1_000_000, batch=None):
    """Run ``trial(batch)`` -> (errors, bits) until min_errors or max_bits.

    Returns the error ratio (0.0 if no errors were seen; see ber_plot)."""
    errs = bits = 0
    while errs < min_errors and bits < max_bits:
        e, b = trial(batch)
        errs += int(e); bits += int(b)
    return errs / max(bits, 1)


# ----------------------------------------------------------------------------- widgets
class _W:
    """Stand-in widget used when ipywidgets is unavailable or the build is static."""

    def __init__(self, value, **kw):
        self.value = value
        self.__dict__.update(kw)


def slider(value, min, max, step=None, desc="", fmt=None):
    """A float slider that only updates on release (keeps heavy callbacks usable)."""
    try:
        import ipywidgets as w
        return w.FloatSlider(value=value, min=min, max=max, step=step or (max - min) / 100,
                             description=desc, continuous_update=False,
                             readout_format=fmt or ".3g", style={"description_width": "initial"},
                             layout=w.Layout(width="420px"))
    except ImportError:
        return _W(value)


def islider(value, min, max, step=1, desc=""):
    try:
        import ipywidgets as w
        return w.IntSlider(value=value, min=min, max=max, step=step, description=desc,
                           continuous_update=False, style={"description_width": "initial"},
                           layout=w.Layout(width="420px"))
    except ImportError:
        return _W(value)


def choice(options, value=None, desc=""):
    value = options[0] if value is None else value
    try:
        import ipywidgets as w
        return w.Dropdown(options=options, value=value, description=desc,
                          style={"description_width": "initial"})
    except ImportError:
        return _W(value)


def _default(v):
    if hasattr(v, "value"):
        return v.value
    if isinstance(v, (list, tuple)):
        return v[0]
    return v


def interact(f, **kw):
    """ipywidgets.interact, or a single call at the default values when the
    notebook is being pre-built / run outside Jupyter."""
    if is_static() or not _in_notebook():
        if is_static():
            note("Interactive panel, shown here at its default settings: run the "
                 "notebook in Jupyter to move the controls.")
        return f(**{k: _default(v) for k, v in kw.items()})
    import ipywidgets as w
    return w.interact(f, **kw)
