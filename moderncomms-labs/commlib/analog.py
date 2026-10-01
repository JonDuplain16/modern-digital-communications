"""Analog modulation building blocks: test messages, AM and its envelope detector,
FM modulation and discrimination, and the FM stereo multiplex.

Used by Lab 14 (and the superheterodyne lab). Import as ``from commlib import analog``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

__all__ = ["voice_like", "envelope_detector", "am_modulate", "fm_complex", "fm_discriminator_hz",
           "brickwall", "stereo_mpx", "stereo_decode", "tone_level", "align_sinad"]


def voice_like(fs, dur, f0=(110.0, 150.0), formants=((730, 90), (1090, 110), (2440, 170)),
               band=(80.0, 4000.0)):
    """A synthetic vowel "ah": glottal pulse train (pitch gliding f0[0] -> f0[1]) through
    formant resonators, band-limited like a voice channel. Peak-normalised to 1."""
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    pitch = f0[0] + (f0[1] - f0[0]) * t / max(dur, 1e-9)
    phase = np.cumsum(pitch) / fs
    x = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    for fmt, bw in formants:
        r = np.exp(-np.pi * bw / fs)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fmt / fs), r * r], x)
    hi = min(band[1], 0.45 * fs)
    x = sosfiltfilt(butter(4, [band[0], hi], btype="band", fs=fs, output="sos"), x)
    return x / (np.max(np.abs(x)) + 1e-12)


def envelope_detector(x, fs, rc):
    """Ideal diode + RC envelope detector: v[n] = max(x[n], a v[n-1]), a = exp(-1/(RC fs)).

    Vectorised in blocks (the recursion becomes a running maximum of x[k] a^-k), so a
    million samples take milliseconds instead of a Python loop."""
    x = np.asarray(x, float)
    a = np.exp(-1.0 / (rc * fs))
    out = np.empty_like(x)
    B = int(max(1, min(len(x), 600 * rc * fs)))       # keep a^-B below e^600
    v = 0.0
    for s in range(0, len(x), B):
        seg = x[s:s + B]
        n = np.arange(len(seg))
        g = np.exp(n / (rc * fs))                     # a^-n
        w = np.maximum.accumulate(seg * g)
        w = np.maximum(w, a * v)
        y = w / g
        out[s:s + B] = y
        v = y[-1]
    return out


def am_modulate(m, mu, fc, fs, carrier=1.0):
    """Conventional AM: A[1 + mu m(t)] cos(2 pi fc t)."""
    t = np.arange(len(m)) / fs
    return carrier * (1 + mu * np.asarray(m)) * np.cos(2 * np.pi * fc * t)


def fm_complex(m, fs, kf, phase0=0.0):
    """Complex envelope of FM: exp(j 2 pi kf * integral of m). kf in Hz per unit of m."""
    return np.exp(1j * (2 * np.pi * kf * np.cumsum(m) / fs + phase0))


def fm_discriminator_hz(z, fs):
    """Instantaneous frequency (Hz) of a complex envelope by the phase-difference
    discriminator, same length as z. The first sample uses a circular difference, which is
    exact for periodic test signals (an integer number of message periods)."""
    return np.angle(z * np.conj(np.roll(z, 1))) * fs / (2 * np.pi)


def brickwall(x, fs, f_lo, f_hi=None):
    """Ideal (FFT) filter. One cutoff: low-pass |f| <= f_lo. Two: band-pass f_lo..f_hi
    (applied to |f| for real input, to f for complex input)."""
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1 / fs)
    if f_hi is None:
        keep = np.abs(f) <= f_lo
    else:
        ff = np.abs(f) if not np.iscomplexobj(x) else f
        keep = (ff >= f_lo) & (ff <= f_hi)
    X[~keep] = 0
    y = np.fft.ifft(X)
    return y if np.iscomplexobj(x) else y.real


def stereo_mpx(left, right, fs, pilot=0.1, rds=0.0, rng=None):
    """FM stereo multiplex: 0.45(L+R) + 0.45(L-R)cos(2pi 38k t) + pilot cos(2pi 19k t)
    [+ an RDS-like BPSK subcarrier at 57 kHz of amplitude ``rds``]."""
    t = np.arange(len(left)) / fs
    mpx = (0.45 * (left + right) + 0.45 * (left - right) * np.cos(2 * np.pi * 38e3 * t)
           + pilot * np.cos(2 * np.pi * 19e3 * t))
    if rds:
        rng = np.random.default_rng(rng)
        nb = int(len(t) / fs * 1187.5) + 2
        bits = 2.0 * rng.integers(0, 2, nb) - 1
        k = np.minimum((t * 1187.5).astype(int), nb - 1)
        ph = (t * 1187.5) % 1
        biphase = bits[k] * np.where(ph < 0.5, 1.0, -1.0) * np.sin(np.pi * 2 * ph) ** 2
        mpx = mpx + rds * biphase * np.cos(2 * np.pi * 57e3 * t)
    return mpx


def stereo_decode(mpx, fs, phase_error_deg=0.0, stereo=True):
    """Pilot-aided stereo decoder. Regenerates the 38 kHz carrier by doubling the phase of
    the band-pass-filtered 19 kHz pilot (plus an optional deliberate phase error), then
    L = (S + D)/2, R = (S - D)/2. Returns (L, R)."""
    from scipy.signal import hilbert
    lp15 = butter(8, 15e3, fs=fs, output="sos")
    S = sosfiltfilt(lp15, mpx) / 0.45
    if not stereo:
        return S / 2, S / 2
    bp19 = butter(4, [18.5e3, 19.5e3], btype="band", fs=fs, output="sos")
    p19 = hilbert(sosfiltfilt(bp19, mpx))
    c38 = np.cos(2 * np.angle(p19) + np.deg2rad(phase_error_deg))
    D = sosfiltfilt(lp15, 2 * mpx * c38) / 0.45
    return (S + D) / 2, (S - D) / 2


def tone_level(x, f0, fs, trim=0.1):
    """Amplitude of the f0 component of x (ignoring ``trim`` of each end)."""
    n = len(x)
    a, b = int(trim * n), int((1 - trim) * n)
    t = np.arange(a, b) / fs
    return 2 * np.abs(np.mean(x[a:b] * np.exp(-2j * np.pi * f0 * t)))


def align_sinad(y, ref, max_lag):
    """SINAD (dB) of y against ref after the best integer delay (0..max_lag) and gain.
    Returns (sinad_db, lag)."""
    y = np.asarray(y, float) - np.mean(y)
    ref = np.asarray(ref, float) - np.mean(ref)
    n = len(ref) - max_lag
    best, best_lag = -np.inf, 0
    for d in range(0, max_lag + 1, max(1, max_lag // 60)):
        yy, rr = y[d:d + n], ref[:n]
        g = np.dot(yy, rr) / (np.dot(rr, rr) + 1e-30)
        e = yy - g * rr
        s = 10 * np.log10(np.sum((g * rr) ** 2) / (np.sum(e ** 2) + 1e-30) + 1e-30)
        if s > best:
            best, best_lag = s, d
    return best, best_lag
