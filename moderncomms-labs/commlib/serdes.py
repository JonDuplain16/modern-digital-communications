"""commlib.serdes -- serial-link helpers to complement commlib.linecodes.

Companion module for Chapter 8 (Baseband Transmission) and Lab 13.
Import explicitly:  ``from commlib import serdes as sd``.

* ``b8zs`` / ``hdb3``       AMI with zero substitution (T1 / E1), ternary symbols with violations
* ``enc64b66b``             2-bit sync header + self-synchronously scrambled 64-bit payload
* ``dual_dirac_ber``, ``total_jitter``   the dual-Dirac jitter model and bathtub curve
* ``ctle_response``         a one-zero, two-pole continuous-time linear equaliser
* ``duobinary_*``           precoder, 1+D / 1-D^2 channels, decoders (with and without precoding)
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcinv

from .linecodes import scramble_ss

__all__ = ["ami_symbols", "b8zs", "hdb3", "enc64b66b", "qfunc", "qinv", "dual_dirac_ber",
           "total_jitter", "ctle_response", "duobinary_precode", "partial_response",
           "duobinary_decode_precoded", "duobinary_decode_feedback"]


def qfunc(x):
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


def qinv(p):
    return np.sqrt(2) * erfcinv(2 * np.asarray(p, float))


# ----------------------------------------------------------------------------- bipolar codes
def ami_symbols(bits):
    """Alternate mark inversion: 0 -> 0, ones alternate +1/-1 (first one = +1)."""
    b = np.asarray(bits, int)
    ones = np.cumsum(b)
    return np.where(b == 1, np.where(ones % 2 == 1, 1, -1), 0).astype(int)


def b8zs(bits):
    """AMI with bipolar 8-zero substitution (North American T1): every run of eight zeros is
    replaced by 000VB0VB, where V repeats the polarity of the previous pulse (a violation) and B
    follows the AMI rule. Returns (symbols in {-1,0,+1}, boolean mask of violation pulses)."""
    b = np.asarray(bits, int)
    out = np.zeros(len(b), int)
    viol = np.zeros(len(b), bool)
    last = -1                                   # polarity of the previous pulse
    i = 0
    while i < len(b):
        if b[i] == 0 and i + 8 <= len(b) and not b[i:i + 8].any():
            V, B = last, -last
            out[i + 3], out[i + 4], out[i + 6], out[i + 7] = V, B, -V, -B
            viol[i + 3] = viol[i + 6] = True
            last = -B
            i += 8
            continue
        if b[i]:
            last = -last
            out[i] = last
        i += 1
    return out, viol


def hdb3(bits):
    """High-density bipolar 3 (E1, ITU-T G.703): every run of four zeros becomes 000V or B00V,
    choosing B so that successive violations alternate in polarity (no DC).
    Returns (symbols, violation mask)."""
    b = np.asarray(bits, int)
    out = np.zeros(len(b), int)
    viol = np.zeros(len(b), bool)
    last = -1
    pulses_since_v = 0
    i = 0
    while i < len(b):
        if b[i] == 0 and i + 4 <= len(b) and not b[i:i + 4].any():
            if pulses_since_v % 2 == 1:         # odd: 000V
                V = last
                out[i + 3] = V
            else:                               # even: B00V
                B = -last
                out[i] = B
                V = B
                out[i + 3] = V
            viol[i + 3] = True
            last = V
            pulses_since_v = 0
            i += 4
            continue
        if b[i]:
            last = -last
            out[i] = last
            pulses_since_v += 1
        i += 1
    return out, viol


# ----------------------------------------------------------------------------- 64b/66b
def enc64b66b(bits, state=None):
    """Simplified 64b/66b data blocks: each 64-bit block is scrambled with the self-synchronous
    x^58 + x^39 + 1 scrambler and prefixed with the sync header 01. Returns the line bits."""
    b = np.asarray(bits, np.int8)
    n = len(b) // 64 * 64
    s = scramble_ss(b[:n], state=state).reshape(-1, 64)
    hdr = np.tile(np.array([0, 1], np.int8), (s.shape[0], 1))
    return np.concatenate([hdr, s], axis=1).ravel()


# ----------------------------------------------------------------------------- jitter
def dual_dirac_ber(x, rj, dj, rho=0.5):
    """Bathtub curve: BER at sampling phase x (UI from the left crossing) for random jitter rj
    (rms, UI) and dual-Dirac deterministic jitter dj (UI), transition density rho."""
    x = np.asarray(x, float)
    rj = max(float(rj), 1e-9)
    left = qfunc((x - dj / 2) / rj) + qfunc((x + dj / 2) / rj)
    right = qfunc((1 - x - dj / 2) / rj) + qfunc((1 - x + dj / 2) / rj)
    return rho / 2 * (left + right)


def total_jitter(ber, rj, dj, rho=0.5):
    """TJ(BER) = DJ + 2 Q^-1(BER / rho) RJ, in UI."""
    return dj + 2 * float(qinv(ber / rho)) * rj


# ----------------------------------------------------------------------------- CTLE
def ctle_response(f, peaking_db, f_peak):
    """Continuous-time linear equaliser H(s) = A (1 + s/wz) / ((1 + s/wp)^2) normalised to
    0 dB at DC, with poles at f_peak and the zero placed so that the boost at f_peak is
    ``peaking_db``. f and f_peak in the same units."""
    f = np.asarray(f, float)
    if peaking_db <= 0:
        return np.ones_like(f, dtype=complex)
    # |H(fp)| = sqrt(1 + (fp/fz)^2) / 2 = 10^(pk/20)  ->  fz
    g = 10 ** (peaking_db / 20)
    r = np.sqrt(max((2 * g) ** 2 - 1, 1e-9))
    fz = f_peak / r
    s = 1j * f
    return (1 + s / fz) / (1 + s / f_peak) ** 2


# ----------------------------------------------------------------------------- partial response
def duobinary_precode(d, order=1, b0=0):
    """Precoder b_k = d_k XOR b_{k-order} (order 1: duobinary, 2: modified duobinary)."""
    d = np.asarray(d, int)
    b = np.zeros(len(d) + order, int)
    b[:order] = b0
    for k in range(len(d)):
        b[k + order] = d[k] ^ b[k]
    return b[order:]


def partial_response(a, kind="1+D", a_prev=-1.0):
    """Sampled partial-response channel: 1+D gives a_k + a_{k-1}; 1-D^2 gives a_k - a_{k-2}."""
    a = np.asarray(a, float)
    if kind == "1+D":
        return a + np.r_[a_prev, a[:-1]]
    return a - np.r_[a_prev, a_prev, a[:-2]]


def duobinary_decode_precoded(y, kind="1+D"):
    """Memoryless decision for precoded partial response: 1 if |y| < 1 (1+D) / |y| > 1 (1-D^2)."""
    y = np.asarray(y, float)
    return (np.abs(y) < 1).astype(int) if kind == "1+D" else (np.abs(y) > 1).astype(int)


def duobinary_decode_feedback(y, a_prev=-1.0):
    """Unprecoded 1+D detection by decision feedback: a_k = sign(y_k - a_{k-1}); bits = (a+1)/2.
    One wrong decision propagates until the data breaks the chain."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    prev = a_prev
    for i, yi in enumerate(y):
        prev = 1.0 if yi - prev > 0 else -1.0
        out[i] = prev
    return ((out + 1) / 2).astype(int)
