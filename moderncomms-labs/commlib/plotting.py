"""Small plotting helpers shared by the labs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_constellation", "plot_psd", "plot_eye", "ber_axes", "style"]


def style():
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})


def plot_constellation(ax, y, ref=None, title=None, lim=None, alpha=0.3, s=4):
    y = np.asarray(y)
    ax.scatter(y.real, y.imag, s=s, alpha=alpha)
    if ref is not None:
        ax.scatter(np.real(ref), np.imag(ref), marker="x", c="k", s=40, linewidths=1.5)
    lim = lim or 1.2 * max(np.max(np.abs(y.real)), np.max(np.abs(y.imag)), 1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    if title:
        ax.set_title(title)


def plot_psd(ax, x, fs=1.0, nfft=1024, label=None):
    from .filters import welch_psd
    f, p = welch_psd(x, fs, nfft)
    ax.plot(f, p, label=label)
    ax.set_xlabel("Frequency" + (" (Hz)" if fs != 1.0 else " (cycles/sample)"))
    ax.set_ylabel("PSD (dB)")


def plot_eye(ax, x, sps, n_sym=2, offset=0, component="real"):
    from .filters import eye_traces
    tr = eye_traces(getattr(np.asarray(x), component), sps, n_sym, offset)
    t = np.arange(tr.shape[1]) / sps
    ax.plot(t, tr.T, color="C0", alpha=0.15, lw=0.8)
    ax.set_xlabel("Time (symbols)")


def ber_axes(ax, xlabel="Eb/N0 (dB)"):
    ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Bit error rate")
    ax.grid(True, which="both", alpha=0.3)
