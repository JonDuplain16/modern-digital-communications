"""Pulse shaping, matched filtering, interpolation and eye diagrams."""
from __future__ import annotations

import numpy as np

__all__ = ["rrc_taps", "rc_taps", "shape", "matched_filter", "fractional_delay_taps",
           "interp_cubic", "eye_traces", "welch_psd"]


def rc_taps(beta, sps, span=10):
    """Raised-cosine impulse response, peak = 1, length span*sps+1."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.sinc(t)
    if beta > 0:
        den = 1 - (2 * beta * t) ** 2
        sing = np.isclose(den, 0)
        h = np.where(sing, np.pi / 4 * np.sinc(1 / (2 * beta)),
                     h * np.cos(np.pi * beta * t) / np.where(sing, 1, den))
    return h


def rrc_taps(beta, sps, span=10):
    """Root-raised-cosine taps normalized to unit energy (sum h^2 = 1).

    RRC * RRC = RC, so a unit-energy RRC transmit filter followed by the same
    receive filter yields a Nyquist pulse whose peak equals 1.
    """
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def shape(symbols, taps, sps):
    """Upsample by sps (zero stuffing) and filter. Output length len(sym)*sps + len(taps) - 1."""
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    return np.convolve(up, taps)


def matched_filter(x, taps):
    return np.convolve(x, np.conj(taps[::-1]))


def fractional_delay_taps(d, ntaps=21):
    """Windowed-sinc FIR that delays by (ntaps-1)/2 + d samples, |d| < 1."""
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(n - d) * np.hamming(ntaps)
    return h / h.sum()


def interp_cubic(x, t):
    """Cubic (Farrow-structure Lagrange) interpolation of x at fractional index t."""
    t = np.atleast_1d(t)
    n = np.floor(t).astype(int)
    mu = t - n
    n = np.clip(n, 1, len(x) - 3)
    xm1, x0, x1, x2 = x[n - 1], x[n], x[n + 1], x[n + 2]
    # Lagrange cubic through points at -1, 0, 1, 2
    c0 = x0
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + c0


def eye_traces(x, sps, n_sym=2, offset=0, max_traces=300):
    """Cut a real or complex waveform into overlapping traces n_sym symbols long."""
    L = n_sym * sps + 1
    starts = np.arange(offset, len(x) - L, sps)[:max_traces]
    return np.array([x[s:s + L] for s in starts])


def welch_psd(x, fs=1.0, nfft=1024):
    """Two-sided Welch PSD in dB, frequency axis centered on 0."""
    from scipy.signal import welch
    f, p = welch(x, fs=fs, nperseg=min(nfft, len(x)), return_onesided=False,
                 scaling="density")
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-20)
