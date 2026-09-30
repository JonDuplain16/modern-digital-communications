"""Carrier, timing and frame synchronization algorithms.

Implementations follow Rice, "Digital Communications: A Discrete-Time Approach"
(Ch. 7-8) and Mengali & D'Andrea, "Synchronization Techniques for Digital
Receivers". They are written for clarity, one sample at a time, so you can
instrument every internal state.
"""
from __future__ import annotations

import numpy as np

from .filters import interp_cubic

__all__ = ["loop_gains", "pll_dd", "costas_qpsk", "gardner_sync", "mm_sync",
           "cfo_power_estimate", "zadoff_chu", "frame_sync", "schmidl_cox_metric",
           "cfo_from_repeated_preamble"]


def loop_gains(bn, zeta=0.7071, kd=1.0, k0=1.0):
    """Proportional/integral gains of a 2nd-order discrete PLL.

    bn: noise bandwidth normalized to the update rate (e.g. 0.01 = 1% of Rs).
    Returns (kp, ki) for: v = kp*e + integrator; integrator += ki*e.
    """
    theta = bn / (zeta + 1 / (4 * zeta))
    d = 1 + 2 * zeta * theta + theta ** 2
    kp = 4 * zeta * theta / d / (kd * k0)
    ki = 4 * theta ** 2 / d / (kd * k0)
    return kp, ki


def pll_dd(y, constellation, bn=0.01, zeta=0.7071):
    """Decision-directed carrier phase/frequency PLL on symbol-rate samples.

    Works for any constellation (PSK or QAM). Returns (corrected, phase_est).
    """
    kp, ki = loop_gains(bn, zeta)
    out = np.empty_like(y)
    phase = 0.0
    integ = 0.0
    ph_hist = np.empty(len(y))
    pts = constellation.points
    for n, s in enumerate(y):
        z = s * np.exp(-1j * phase)
        out[n] = z
        a = pts[np.argmin(np.abs(z - pts))]
        e = np.angle(z * np.conj(a))
        integ += ki * e
        phase += kp * e + integ
        ph_hist[n] = phase
    return out, ph_hist


def costas_qpsk(y, bn=0.01, zeta=0.7071):
    """Classic QPSK Costas loop with sign-based phase detector."""
    kp, ki = loop_gains(bn, zeta)
    out = np.empty_like(y)
    phase, integ = 0.0, 0.0
    freq = np.empty(len(y))
    for n, s in enumerate(y):
        z = s * np.exp(-1j * phase)
        out[n] = z
        e = np.sign(z.real) * z.imag - np.sign(z.imag) * z.real
        e /= np.sqrt(2)
        integ += ki * e
        phase += kp * e + integ
        freq[n] = integ
    return out, freq


def gardner_sync(x, sps, bn=0.005, zeta=0.7071, kd=None):
    """Gardner timing-error-detector symbol synchronizer with cubic interpolation.

    x: matched-filtered samples at sps samples/symbol (sps >= 2).
    Returns (symbols, timing_error_history, tau_history) where tau is the
    fractional-sample timing estimate.
    """
    if kd is None:
        kd = 2.0  # approximate detector gain for unit-energy symbols, RRC beta ~0.35
    kp, ki = loop_gains(bn, zeta, kd=kd)
    t = float(sps)          # current strobe position (samples)
    integ = 0.0
    prev = interp_cubic(x, t - sps)[0]
    syms, errs, taus = [], [], []
    while t + sps + 3 < len(x):
        cur = interp_cubic(x, t)[0]
        mid = interp_cubic(x, t - sps / 2)[0]
        e = np.real(np.conj(mid) * (cur - prev))
        integ += ki * e
        v = kp * e + integ
        syms.append(cur)
        errs.append(e)
        taus.append(t % sps)
        prev = cur
        t += sps - v * sps
    return np.array(syms), np.array(errs), np.array(taus)


def mm_sync(x, sps, constellation, bn=0.005, zeta=0.7071):
    """Mueller & Mueller decision-directed timing recovery (1 sample/symbol TED)."""
    kp, ki = loop_gains(bn, zeta, kd=1.0)
    t = float(sps)
    integ = 0.0
    y_prev, a_prev = 0j, 0j
    syms, errs = [], []
    pts = constellation.points
    while t + sps + 3 < len(x):
        y = interp_cubic(x, t)[0]
        a = pts[np.argmin(np.abs(y - pts))]
        e = np.real(np.conj(a) * y_prev - np.conj(a_prev) * y)
        integ += ki * e
        v = kp * e + integ
        syms.append(y)
        errs.append(e)
        y_prev, a_prev = y, a
        t += sps - v * sps
    return np.array(syms), np.array(errs)


def cfo_power_estimate(y, M, nfft=None):
    """Blind CFO estimate for M-PSK: raise to the M-th power and find the FFT peak.

    Returns CFO in cycles/sample (unambiguous range +-1/(2M)).
    """
    z = y ** M
    nfft = nfft or 1 << int(np.ceil(np.log2(len(z) * 8)))
    Z = np.fft.fft(z, nfft)
    k = np.argmax(np.abs(Z))
    f = np.fft.fftfreq(nfft)[k]
    return f / M


def zadoff_chu(u, N):
    """Zadoff-Chu sequence of odd length N and root u (gcd(u,N)=1)."""
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def frame_sync(y, preamble):
    """Cross-correlate with a known preamble.

    Returns (start_index, complex_gain, normalized_correlation_magnitude).
    """
    from scipy.signal import correlate
    c = correlate(y, preamble, mode="valid")
    energy = np.convolve(np.abs(y) ** 2, np.ones(len(preamble)), mode="valid")
    metric = np.abs(c) / np.sqrt(energy * np.sum(np.abs(preamble) ** 2) + 1e-12)
    k = int(np.argmax(metric))
    gain = c[k] / np.sum(np.abs(preamble) ** 2)
    return k, gain, metric


def schmidl_cox_metric(y, L):
    """Schmidl & Cox timing metric M(d) = |P(d)|^2 / R(d)^2 for a preamble whose
    two halves (length L each) are identical. Also returns P for CFO estimation."""
    p = y[L:] * np.conj(y[:-L])
    P = np.convolve(p, np.ones(L), mode="valid")
    R = np.convolve(np.abs(y[L:]) ** 2, np.ones(L), mode="valid")
    return np.abs(P) ** 2 / (R ** 2 + 1e-12), P


def cfo_from_repeated_preamble(y, L):
    """CFO (cycles/sample) from two identical halves of length L starting at y[0]."""
    P = np.sum(y[L:2 * L] * np.conj(y[:L]))
    return np.angle(P) / (2 * np.pi * L)
