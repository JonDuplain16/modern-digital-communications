"""Linear and adaptive equalizers (symbol-spaced, complex baseband)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

__all__ = ["conv_matrix", "zf_fir", "mmse_fir", "lms_equalizer", "cma_equalizer",
           "dfe_lms", "apply_fir"]


def conv_matrix(h, L):
    """(L+len(h)-1, L) convolution matrix H so that H @ w = h * w."""
    h = np.asarray(h, dtype=complex)
    col = np.concatenate([h, np.zeros(L - 1)])
    row = np.zeros(L, dtype=complex)
    row[0] = h[0]
    return toeplitz(col, row)


def zf_fir(h, L, delay=None):
    """Least-squares ZF FIR equalizer of length L forcing (h*w) ~ delta[n-delay]."""
    H = conv_matrix(h, L)
    delay = (H.shape[0] // 2) if delay is None else delay
    e = np.zeros(H.shape[0], dtype=complex)
    e[delay] = 1
    w, *_ = np.linalg.lstsq(H, e, rcond=None)
    return w, delay


def mmse_fir(h, L, n0, delay=None):
    """MMSE FIR equalizer for unit-energy i.i.d. symbols and noise variance n0."""
    H = conv_matrix(h, L)
    delay = (H.shape[0] // 2) if delay is None else delay
    R = H.conj().T @ H + n0 * np.eye(L)
    p = H.conj().T[:, delay]
    w = np.linalg.solve(R, p)
    return w, delay


def apply_fir(y, w, delay):
    """Filter and align so output[n] estimates the symbol a[n]."""
    z = np.convolve(y, w)
    return z[delay:delay + len(y)]


def lms_equalizer(y, train, L=11, mu=0.01, constellation=None, delay=None):
    """Training-then-decision-directed complex LMS.

    train: known symbols used for the first len(train) outputs; afterwards the
    reference is the nearest constellation point (requires constellation).
    Returns (outputs, errors, final_taps).
    """
    delay = L // 2 if delay is None else delay
    w = np.zeros(L, dtype=complex)
    w[delay] = 1
    ypad = np.concatenate([np.zeros(L - 1 - delay), y, np.zeros(delay)])
    out = np.empty(len(y), dtype=complex)
    err = np.empty(len(y))
    for n in range(len(y)):
        u = ypad[n:n + L][::-1]
        z = np.dot(w.conj(), u)
        if n < len(train):
            d = train[n]
        else:
            pts = constellation.points
            d = pts[np.argmin(np.abs(z - pts))]
        e = d - z
        w += mu * u * np.conj(e)
        out[n] = z
        err[n] = np.abs(e) ** 2
    return out, err, w


def cma_equalizer(y, L=11, mu=1e-3, R2=None, delay=None):
    """Godard / constant-modulus algorithm (blind). R2 = E|a|^4 / E|a|^2."""
    delay = L // 2 if delay is None else delay
    R2 = 1.0 if R2 is None else R2
    w = np.zeros(L, dtype=complex)
    w[delay] = 1
    ypad = np.concatenate([np.zeros(L - 1 - delay), y, np.zeros(delay)])
    out = np.empty(len(y), dtype=complex)
    cost = np.empty(len(y))
    for n in range(len(y)):
        u = ypad[n:n + L][::-1]
        z = np.dot(w.conj(), u)
        e = z * (np.abs(z) ** 2 - R2)
        w -= mu * u * np.conj(e)
        out[n] = z
        cost[n] = (np.abs(z) ** 2 - R2) ** 2
    return out, cost, w


def dfe_lms(y, train, constellation, Lf=11, Lb=5, mu=0.01):
    """LMS decision-feedback equalizer: feedforward (Lf) + feedback (Lb) taps."""
    delay = Lf // 2
    wf = np.zeros(Lf, dtype=complex)
    wf[delay] = 1
    wb = np.zeros(Lb, dtype=complex)
    ypad = np.concatenate([np.zeros(Lf - 1 - delay), y, np.zeros(delay)])
    past = np.zeros(Lb, dtype=complex)
    out = np.empty(len(y), dtype=complex)
    err = np.empty(len(y))
    pts = constellation.points
    for n in range(len(y)):
        u = ypad[n:n + Lf][::-1]
        z = np.dot(wf.conj(), u) - np.dot(wb.conj(), past)
        d = train[n] if n < len(train) else pts[np.argmin(np.abs(z - pts))]
        e = d - z
        wf += mu * u * np.conj(e)
        wb -= mu * past * np.conj(e)
        past = np.roll(past, 1)
        past[0] = d
        out[n] = z
        err[n] = np.abs(e) ** 2
    return out, err, (wf, wb)
