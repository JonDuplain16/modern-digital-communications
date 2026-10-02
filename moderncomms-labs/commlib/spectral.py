"""commlib.spectral -- practical spectral analysis and the complex envelope (Chapters 2 and 7).

Windows and their figures of merit, Fourier series, rms time/bandwidth, the five bandwidth
definitions, complex/real sampling aliases, blind I/Q-imbalance correction, a uniform
quantiser and the spectral-coherence estimator used for cyclostationary detection.
These are the same calculations as book/figscripts/ch02_figs.py, ch03_figs.py and
ch07_figs.py, collected here so the labs and the book agree.
Import explicitly:  ``from commlib import spectral as sp``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["WINDOWS", "window", "window_metrics", "scalloping_db", "fourier_coeffs",
           "fourier_sum", "rms_widths", "bandwidth_metrics", "alias_complex", "alias_real",
           "blind_iq_correct", "iq_fit", "quantize", "spectral_coherence"]


# ----------------------------------------------------------------------------- windows
def _bh(N):
    from scipy.signal import windows
    return windows.blackmanharris(N, sym=False)


def _flattop(N):
    from scipy.signal import windows
    return windows.flattop(N, sym=False)


WINDOWS = {
    "Rectangular": lambda N: np.ones(N),
    "Hann": lambda N: np.hanning(N + 1)[:-1],          # periodic (DFT-even) Hann
    "Blackman–Harris": _bh,
    "Flat-top": _flattop,
}


def window(name, N):
    """Window samples by name (see WINDOWS; "Blackman-Harris" with a hyphen also works)."""
    name = name.replace("-", "–") if name.startswith("Blackman") else name
    return np.asarray(WINDOWS[name](int(N)), float)


def scalloping_db(w, offsets):
    """Amplitude read by a DFT bin (dB re. on-bin) for tones ``offsets`` bins away from it."""
    w = np.asarray(w, float)
    n = np.arange(len(w))
    d = np.atleast_1d(np.asarray(offsets, float))
    resp = np.abs(np.exp(2j * np.pi * np.outer(d, n) / len(w)) @ w) / np.sum(w)
    return 20 * np.log10(np.maximum(resp, 1e-300))


def window_metrics(w, pad=64):
    """Figures of merit of a window: dict(enbw (bins), scallop (dB loss at 0.5 bin),
    sidelobe (highest sidelobe, dB), mainlobe (null-to-null half width, bins), cg (coherent gain))."""
    w = np.asarray(w, float)
    N = len(w)
    W = np.abs(np.fft.fft(w, pad * N))
    W = W / W[0]
    WdB = 20 * np.log10(np.maximum(W[: pad * N // 2], 1e-300))
    # edge of the main lobe = first local minimum deeper than -30 dB (the flat-top window
    # ripples inside its main lobe, so the very first minimum is not the null)
    d = np.diff(WdB)
    mins = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    mins = mins[WdB[mins] < -30]
    k = int(mins[0]) if len(mins) else len(WdB)
    side = float(WdB[k:].max()) if k < len(WdB) else -300.0
    return dict(enbw=float(N * np.sum(w ** 2) / np.sum(w) ** 2),
                scallop=float(-scalloping_db(w, [0.5])[0]),
                sidelobe=side, mainlobe=k / pad, cg=float(np.mean(w)))


# ----------------------------------------------------------------------------- Fourier series
def fourier_coeffs(kind, nmax, duty=0.5):
    """Harmonic amplitudes and phases (A_k, phi_k), k = 1..nmax, and the DC term c0, such that
    x(t) = c0 + sum A_k cos(2 pi k t + phi_k) for a unit-period waveform of peak 1:
    "Square" (+-1), "Triangle" (+-1), "Sawtooth" (-1..1 ramp), "Pulse train" (0/1, duty)."""
    k = np.arange(1, nmax + 1)
    A = np.zeros(nmax)
    ph = np.full(nmax, -np.pi / 2)                       # sine terms by default
    c0 = 0.0
    if kind == "Square":
        A = np.where(k % 2 == 1, 4 / (np.pi * k), 0.0)
    elif kind == "Triangle":
        A = np.where(k % 2 == 1, 8 / (np.pi ** 2 * k ** 2), 0.0) * ((-1.0) ** ((k - 1) // 2))
    elif kind == "Sawtooth":
        A = 2 / (np.pi * k) * (-1.0) ** (k + 1)
    elif kind == "Pulse train":
        c0 = duty
        A = 2 * np.sin(np.pi * k * duty) / (np.pi * k)
        ph = np.zeros(nmax)
    else:
        raise ValueError(kind)
    neg = A < 0                                          # keep amplitudes positive, fold the sign into phase
    A = np.abs(A)
    ph = np.where(neg, ph + np.pi, ph)
    return c0, A, ph


def fourier_sum(t, c0, A, ph, sigma=False):
    """Evaluate the partial sum at times t (periods); ``sigma`` applies Lanczos sigma factors."""
    k = np.arange(1, len(A) + 1)
    g = np.sinc(k / (len(A) + 1)) if sigma else np.ones(len(A))
    return c0 + np.cos(2 * np.pi * np.outer(np.asarray(t, float), k) + ph) @ (g * A)


# ----------------------------------------------------------------------------- uncertainty / bandwidth
def rms_widths(t, x, f, X):
    """RMS duration of |x|^2 and RMS bandwidth of |X|^2 (each about its own centroid)."""
    pt = np.abs(x) ** 2
    pt = pt / pt.sum()
    pf = np.abs(X) ** 2
    pf = pf / pf.sum()
    mt, mf = np.sum(t * pt), np.sum(f * pf)
    return float(np.sqrt(np.sum((t - mt) ** 2 * pt))), float(np.sqrt(np.sum((f - mf) ** 2 * pf)))


def bandwidth_metrics(f, S, ref=None, xdb=26.0):
    """Bandwidths of a two-sided PSD S(f) on a uniform grid f (linear power units).

    ref: in-band reference level (default: median of S for |f| < 0.3, as Chapter 2 uses).
    Returns dict(b3, bn, obw, bx, lo, hi, f3, fx): 3 dB width, noise-equivalent width,
    99 % occupied width (and its edges lo, hi), x-dB width, and the edge pairs."""
    f = np.asarray(f, float)
    S = np.asarray(S, float)
    df = f[1] - f[0]
    if ref is None:
        ref = float(np.median(S[np.abs(f) < 0.3]))
    above = np.flatnonzero(S >= ref / 2)
    ax = np.flatnonzero(S >= ref * 10 ** (-xdb / 10))
    cum = np.cumsum(S) / S.sum()
    lo, hi = f[np.searchsorted(cum, 0.005)], f[min(np.searchsorted(cum, 0.995), len(f) - 1)]
    return dict(b3=(above.max() - above.min() + 1) * df, bn=S.sum() * df / ref,
                obw=hi - lo, bx=(ax.max() - ax.min() + 1) * df, lo=lo, hi=hi,
                f3=(f[above.min()], f[above.max()]), fx=(f[ax.min()], f[ax.max()]))


# ----------------------------------------------------------------------------- sampling
def alias_complex(f, fs):
    """Apparent frequency of a tone at f after complex (I/Q) sampling at fs: wraps into [-fs/2, fs/2)."""
    return ((np.asarray(f, float) + fs / 2) % fs) - fs / 2


def alias_real(f, fs):
    """Apparent (non-negative) frequency of a tone at f after real sampling at fs: folds into [0, fs/2]."""
    return np.abs(alias_complex(f, fs))


# ----------------------------------------------------------------------------- I/Q imbalance
def blind_iq_correct(y):
    """Moment-based blind I/Q-imbalance correction (Chapter 7): remove the mean, then make Q
    orthogonal to I and of equal power. Valid for proper (circular) signals.
    Returns (corrected, gain_dB, phase_deg) where gain/phase are the estimated imbalance."""
    z = np.asarray(y) - np.mean(y)
    i, q = z.real, z.imag
    pi_ = np.mean(i * i)
    gs = np.mean(i * q) / pi_
    gc = np.sqrt(max(np.mean(q * q) / pi_ - gs ** 2, 1e-30))
    zc = i + 1j * (q - gs * i) / gc
    return zc, float(20 * np.log10(np.hypot(gs, gc))), float(np.rad2deg(np.arctan2(gs, gc)))


def iq_fit(y, x):
    """Least-squares fit y = mu x + nu conj(x) + c. Returns (mu, nu, c); IRR = |mu|^2/|nu|^2."""
    x = np.asarray(x)
    A = np.stack([x, np.conj(x), np.ones(len(x))], 1)
    c, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    return c[0], c[1], c[2]


# ----------------------------------------------------------------------------- quantiser
def quantize(x, bits, full_scale=1.0):
    """Mid-rise uniform quantiser with clipping, applied to I and Q separately (real input
    stays real). The output levels are odd multiples of half an LSB, 2*full_scale/2^bits."""
    q = 2 * full_scale / 2 ** bits

    def qr(v):
        v = np.clip(v, -full_scale, full_scale - q)
        return q * (np.floor(v / q) + 0.5)
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return qr(x.real) + 1j * qr(x.imag)
    return qr(x)


# ----------------------------------------------------------------------------- cyclostationarity
def spectral_coherence(x, nfft=256, win=True):
    """Spectral-coherence profiles of x (blocks of nfft samples, FFT-accumulation method).

    Returns (alpha, noncoh, conjcoh, P): cycle frequencies alpha (cycles/sample, -0.5..0.5),
    the maximum over frequency of the non-conjugate coherence |E X(k+a) X*(k)| / sqrt(P P) and of
    the conjugate coherence |E X(k) X(a-k)| / sqrt(P P), and the averaged periodogram P.
    The non-conjugate value at |alpha| <= 2 bins (1 without a window) is NaN: neighbouring
    bins are correlated by the window itself. Same estimator as the
    book's Figure ch03_cyclo, computed with two matrix products so it is fast."""
    x = np.asarray(x, complex)
    nblk = len(x) // nfft
    X = x[: nblk * nfft].reshape(nblk, nfft)
    if win:
        X = X * np.hanning(nfft)
    X = np.fft.fft(X, axis=1)
    C = (X.conj().T @ X) / nblk                      # C[k, l] = E X*(k) X(l)
    C2 = (X.T @ X) / nblk                            # C2[k, l] = E X(k) X(l)
    P = np.real(np.diag(C))
    k = np.arange(nfft)
    d = np.arange(-nfft // 2, nfft // 2)
    l1 = (k[None, :] + d[:, None]) % nfft
    nc = np.abs(C[k[None, :], l1]) / np.sqrt(P[l1] * P[k[None, :]])
    l2 = (d[:, None] - k[None, :]) % nfft
    cj = np.abs(C2[k[None, :], l2]) / np.sqrt(P[k[None, :]] * P[l2])
    noncoh = nc.max(axis=1)
    noncoh[np.abs(d) <= (2 if win else 1)] = np.nan       # window leakage links neighbours
    return d / nfft, noncoh, cj.max(axis=1), P
