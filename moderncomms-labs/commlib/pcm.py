"""Sampling, quantization and digital telephony: uniform quantizers, dither, smooth and
segmented (ITU-T G.711) mu-law / A-law companders, delta modulation and CVSD, sigma-delta
modulators of order 1-5, zero-order-hold reconstruction, and the T1/E1 frame layouts.

Used by Lab 16 and the Chapter 5 figures. Import as ``from commlib import pcm``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantize", "quantize_tread", "sqnr_db", "dither", "mulaw", "imulaw", "alaw", "ialaw",
           "g711_mu_encode", "g711_mu_decode", "g711_a_encode", "g711_a_decode", "g711_codec",
           "delta_mod", "cvsd", "ntf_coeffs", "sigma_delta", "sd_theory_sqnr",
           "alias_frequency", "nyquist_zone", "t1_superframe", "e1_frame"]


# ----------------------------------------------------------------------------- uniform quantizers
def quantize(x, bits, full_scale=1.0):
    """Mid-rise uniform quantizer with 2^bits levels over [-full_scale, full_scale)
    (clips beyond). Step = 2 full_scale / 2^bits."""
    d = 2.0 * full_scale / 2 ** bits
    return np.clip(d * (np.floor(np.asarray(x) / d) + 0.5), -full_scale + d / 2, full_scale - d / 2)


def quantize_tread(x, step=1.0):
    """Mid-tread quantizer with step ``step`` (no clipping): step * round(x / step)."""
    return step * np.round(np.asarray(x) / step)


def sqnr_db(x, xq):
    """Signal-to-quantization-noise ratio (dB) of xq against x."""
    x = np.asarray(x, float)
    e = np.asarray(xq, float) - x
    return float(10 * np.log10(np.mean(x ** 2) / max(np.mean(e ** 2), 1e-300)))


def dither(kind, n, step, rng=None):
    """Dither samples: 'RPDF' (uniform, +-step/2), 'TPDF' (triangular, +-step) or 'none'."""
    rng = np.random.default_rng(rng)
    if kind.upper().startswith("R") or kind.lower().startswith("sub"):
        return (rng.random(n) - 0.5) * step
    if kind.upper().startswith("T"):
        return (rng.random(n) - rng.random(n)) * step
    return np.zeros(n)


# ----------------------------------------------------------------------------- smooth companders
def mulaw(x, mu=255.0):
    """mu-law compressor, |x| <= 1."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def imulaw(y, mu=255.0):
    """mu-law expander."""
    y = np.asarray(y, float)
    return np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu


def alaw(x, A=87.6):
    """A-law compressor, |x| <= 1."""
    a = np.abs(np.asarray(x, float))
    y = np.where(a < 1 / A, A * a, 1 + np.log(np.maximum(A * a, 1e-300))) / (1 + np.log(A))
    return np.sign(x) * y


def ialaw(y, A=87.6):
    """A-law expander."""
    a = np.abs(np.asarray(y, float)) * (1 + np.log(A))
    return np.sign(y) * np.where(a < 1, a / A, np.exp(a - 1) / A)


# ----------------------------------------------------------------------------- G.711 segmented
def g711_mu_encode(x):
    """G.711 mu-law on the 14-bit scale (|x| <= 8159, integers). Returns the code fields
    (sign, chord 0-7, step 0-15)."""
    x = np.asarray(x)
    s = np.where(x < 0, -1, 1)
    m = np.minimum(np.abs(x), 8158).astype(np.int64) + 33
    e = np.floor(np.log2(m)).astype(np.int64) - 5
    q = (m >> (e + 1)) & 0xF
    return s, e, q


def g711_mu_decode(s, e, q):
    """Inverse of :func:`g711_mu_encode` (reconstruct at the middle of the step)."""
    return s * (((2 * np.asarray(q) + 33) << np.asarray(e)) - 33)


def g711_a_encode(x):
    """G.711 A-law on the 13-bit scale (|x| <= 4095, integers). Returns (sign, segment, step)."""
    x = np.asarray(x)
    s = np.where(x < 0, -1, 1)
    m = np.minimum(np.abs(x), 4095).astype(np.int64)
    e = np.where(m < 32, 0, np.floor(np.log2(np.maximum(m, 1))).astype(np.int64) - 4)
    q = np.where(e == 0, m >> 1, (m >> np.maximum(e, 1)) & 0xF)
    return s, e, q


def g711_a_decode(s, e, q):
    """Inverse of :func:`g711_a_encode`."""
    e = np.asarray(e)
    q = np.asarray(q)
    return s * np.where(e == 0, 2 * q + 1, (2 * q + 33) << np.maximum(e - 1, 0))


def g711_codec(x, law="mu"):
    """Encode and decode x (full scale +-1 = the codec's overload point) through the
    segmented G.711 law. Returns (y, sign, chord, step)."""
    x = np.asarray(x, float)
    if law == "mu":
        xi = np.round(np.clip(x, -1, 1) * 8159).astype(np.int64)
        s, e, q = g711_mu_encode(xi)
        return g711_mu_decode(s, e, q) / 8159.0, s, e, q
    xi = np.round(np.clip(x, -1, 1) * 4095).astype(np.int64)
    s, e, q = g711_a_encode(xi)
    return g711_a_decode(s, e, q) / 4095.0, s, e, q


# ----------------------------------------------------------------------------- delta modulation
def delta_mod(x, step):
    """Linear delta modulator (one bit per sample). Returns (bits +-1, staircase)."""
    x = np.asarray(x, float).tolist()
    bits = np.empty(len(x))
    est = np.empty(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        b = 1.0 if v >= acc else -1.0
        acc += b * step
        bits[i] = b
        est[i] = acc
    return bits, est


def cvsd(x, step_min, step_max, run=3, beta=0.98, gain=None):
    """Continuously variable slope delta modulation (Bluetooth / military voice style):
    after ``run`` identical bits in a row the step grows by ``gain`` (default step_min),
    otherwise it decays by ``beta`` toward step_min. Returns (bits, staircase, steps)."""
    x = np.asarray(x, float).tolist()
    g = step_min if gain is None else gain
    bits = np.empty(len(x))
    est = np.empty(len(x))
    steps = np.empty(len(x))
    acc, d = 0.0, step_min
    hist = [0] * run
    for i, v in enumerate(x):
        b = 1 if v >= acc else -1
        hist = hist[1:] + [b]
        if abs(sum(hist)) == run:
            d = min(d * beta + g, step_max)
        else:
            d = max(d * beta, step_min)
        acc += b * d
        bits[i], est[i], steps[i] = b, acc, d
    return bits, est, steps


# ----------------------------------------------------------------------------- sigma-delta
def ntf_coeffs(order, hinf=None):
    """Noise-transfer function NTF = b(z)/a(z) for a one-bit modulator of the given order.
    hinf=None: the pure differentiator (1 - z^-1)^L. Otherwise a high-pass Butterworth NTF
    with L zeros at z = 1 and maximum gain ``hinf`` (Lee's rule: 1.5 keeps it stable)."""
    if hinf is None:
        b = np.poly1d([1.0, -1.0]) ** order
        return np.asarray(b.coeffs, float), np.r_[1.0, np.zeros(order)]
    from scipy.signal import butter, freqz
    lo, hi = 1e-4, 0.99
    for _ in range(60):
        wc = np.sqrt(lo * hi)
        b, a = butter(order, wc, "high")
        b = b / b[0]
        _, h = freqz(b, a, 2048)
        if np.max(np.abs(h)) > hinf:
            hi = wc
        else:
            lo = wc
    b, a = butter(order, lo, "high")
    return b / b[0], a


def sigma_delta(x, b, a, limit=1e4):
    """One-bit sigma-delta modulator in error-feedback form: V = X + NTF*E with
    NTF = b/a (b[0] = a[0] = 1). Returns the +-1 output; if the internal state exceeds
    ``limit`` (the loop has gone unstable) the rest of the output is NaN."""
    L = len(a) - 1
    c = (np.asarray(b, float) - np.asarray(a, float))[1:].tolist()
    d = np.asarray(a, float)[1:].tolist()
    eh = [0.0] * L
    wh = [0.0] * L
    v = np.empty(len(x))
    xl = np.asarray(x, float).tolist()
    for n in range(len(xl)):
        w = 0.0
        for i in range(L):
            w += c[i] * eh[i] - d[i] * wh[i]
        if w > limit or w < -limit:
            v[n:] = np.nan
            return v
        u = xl[n] + w
        q = 1.0 if u >= 0 else -1.0
        eh = [q - u] + eh[:-1]
        wh = [w] + wh[:-1]
        v[n] = q
    return v


def sd_theory_sqnr(order, osr, amp=0.5):
    """Linear-model peak SQNR (dB) of a one-bit modulator with NTF (1 - z^-1)^L and a sine
    of amplitude ``amp`` (full scale 1): quantization noise power 1/3 (step 2), shaped."""
    L = order
    ps = amp ** 2 / 2
    pn = (1 / 3) * np.pi ** (2 * L) / ((2 * L + 1) * np.asarray(osr, float) ** (2 * L + 1))
    return 10 * np.log10(ps / pn)


# ----------------------------------------------------------------------------- sampling
def alias_frequency(f, fs):
    """Apparent frequency (0..fs/2) of a real tone at f after sampling at fs."""
    f = np.asarray(f, float)
    return np.abs(f - fs * np.round(f / fs))


def nyquist_zone(f, fs):
    """Nyquist zone (1 = 0..fs/2, 2 = fs/2..fs, ...) of frequency f."""
    return np.floor(np.asarray(f, float) / (fs / 2)).astype(int) + 1


# ----------------------------------------------------------------------------- telephony frames
def t1_superframe(robbed=True):
    """D4 superframe map, 12 frames x 193 bits: 0 = framing bit, 1 = voice bit,
    2 = robbed (signalling) bit in frames 6 and 12. Also returns the framing pattern."""
    fbits = np.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    sf = np.ones((12, 193), dtype=int)
    sf[:, 0] = 0
    if robbed:
        for fr in (5, 11):
            sf[fr, 8::8] = 2
    return sf, fbits


def e1_frame(multiframe=16):
    """E1 frame map, ``multiframe`` frames x 256 bits: 0 = TS0 (framing/alarms), 1 = voice,
    3 = TS16 (signalling)."""
    m = np.ones((multiframe, 256), dtype=int)
    m[:, 0:8] = 0
    m[:, 128:136] = 3
    return m
