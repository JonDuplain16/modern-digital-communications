"""The superheterodyne receiver, simulated where it matters: around the IF.

A band of AM stations is described by a list of :class:`Station` objects. Instead of
simulating megahertz of RF, :func:`if_signal` computes exactly what lands in a window around
the intermediate frequency after the preselector and the mixer: each station contributes its
complex envelope (conjugated if it lies below the LO, i.e. the spectrum is flipped), shifted
to its offset from the IF and weighted by the preselector's response at its true RF
frequency. Records are periodic (all frequencies on the FFT grid), so every filter is an
exact FFT multiplication. Also: tuned-circuit and Butterworth responses, image rejection,
a feedback AGC loop and a two-tone intermodulation model.

Used by Lab 15. Import as ``from commlib import superhet``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = ["Station", "tuned_response", "irr_db", "butter_lp", "if_signal", "detect_am",
           "tone_sinad", "station_envelope", "agc_loop", "two_tone_cubic", "dbm_to_amp"]


@dataclass(frozen=True)
class Station:
    """An AM broadcast station: carrier frequency (Hz), carrier level (dBm at the antenna),
    programme ('tone' + tone frequency, 'voice' or 'music') and modulation depth."""
    f: float
    level: float
    kind: str = "music"
    tone: float = 1000.0
    depth: float = 0.5
    seed: int = 0
    name: str = ""


def dbm_to_amp(dbm):
    """Peak amplitude (V) of a sine of the given power in 50 ohms."""
    return np.sqrt(2 * 50 * 1e-3 * 10 ** (np.asarray(dbm, float) / 10))


def tuned_response(f, f0, Q, stages=1):
    """Complex response of ``stages`` isolated parallel tuned circuits at f0 with loaded Q:
    H = (1 + jQ(f/f0 - f0/f))^-stages."""
    f = np.asarray(f, float)
    rho = f / f0 - f0 / np.maximum(f, 1e-9)
    return (1.0 / (1.0 + 1j * Q * rho)) ** stages


def irr_db(f_rf, f_img, Q, stages=1):
    """Image rejection (dB) of the preselector: stages x 10 log10(1 + Q^2 rho^2)."""
    rho = f_img / f_rf - f_rf / f_img
    return stages * 10 * np.log10(1 + (Q * rho) ** 2)


@lru_cache(maxsize=32)
def _butter_poles(order):
    from scipy.signal import buttap
    z, p, k = buttap(order)
    return p


def butter_lp(f, f3, order):
    """Complex response of an analog Butterworth low-pass (cutoff f3 Hz) at frequencies f;
    as the low-pass equivalent of a band-pass IF filter it is evaluated at the offset from
    the IF centre."""
    p = _butter_poles(int(order))
    s = 1j * np.asarray(f, float) / f3
    H = np.ones_like(s)
    for pk in p:
        H = H * (-pk) / (s - pk)
    return H


@lru_cache(maxsize=64)
def _programme(kind, seed, fs, n, tone):
    """Unit-ish programme signal (periodic for 'tone'/'music'), cached."""
    t = np.arange(n) / fs
    if kind == "tone":
        return np.sin(2 * np.pi * tone * t)
    if kind == "voice":
        from .analog import voice_like
        f0 = (95.0 + 12 * (seed % 5), 140.0 + 9 * (seed % 7))
        v = voice_like(fs, n / fs, f0=f0)
        v = v - v.mean()
        return np.clip(v / (np.std(v) + 1e-12) * 0.35, -0.95, 0.95)
    # 'music': band-limited noise synthesised on the FFT grid (periodic), 4.5 kHz wide,
    # with a slow level contour so it sounds like programme rather than hiss
    rng = np.random.default_rng(1000 + seed)
    k = np.fft.rfftfreq(n, 1 / fs)
    X = (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k)))
    shape = np.where((k > 60) & (k < 4500), 1.0 / np.sqrt(1 + (k / 900.0) ** 2), 0.0)
    x = np.fft.irfft(X * shape, n)
    x = x / (np.std(x) + 1e-12) * 0.35
    return np.clip(x, -0.98, 0.98)


def station_envelope(st, fs, n):
    """Complex envelope A(1 + depth*m(t)) of a station (A = carrier amplitude)."""
    m = _programme(st.kind, st.seed, float(fs), int(n), float(st.tone))
    depth = st.depth if st.kind == "tone" else 1.0
    phase = 2 * np.pi * ((st.seed * 0.618 + st.f * 1e-6 * 0.37) % 1.0)
    return dbm_to_amp(st.level) * (1 + depth * m) * np.exp(1j * phase)


def if_signal(stations, f_lo, f_if, fs, n, pre_f0, pre_q, pre_stages, if_bw, if_order,
              noise_dbm_hz=-160.0, rng=None, return_parts=False):
    """Complex baseband of the IF strip (centred on f_if), before and after the IF filter.

    stations   list of Station
    f_lo       LO frequency (Hz); the mixer output |f - f_lo| near f_if is kept
    fs, n      sample rate and length of the IF-baseband record (its span is +-fs/2)
    pre_*      preselector: tuned at pre_f0 with Q and number of stages (pre_q=None: none)
    if_bw      IF filter bandwidth (Hz, -3 dB, two-sided); if_order Butterworth order
    noise_dbm_hz  antenna noise density
    Returns (z_before, z_after, info) where info lists each station's offset and gains."""
    rng = np.random.default_rng(rng)
    f = np.fft.fftfreq(n, 1 / fs)                       # offsets from the IF centre
    Zb = np.zeros(n, complex)
    info = []
    for st in stations:
        d = abs(st.f - f_lo) - f_if
        if abs(d) > 0.42 * fs:
            continue
        below = st.f < f_lo
        e = station_envelope(st, fs, n)
        E = np.fft.fft(e)
        if below:                                        # spectrum flipped: conj(e)
            E = np.conj(np.roll(E[::-1], 1))
        shift = int(round(d * n / fs))
        E = np.roll(E, shift)
        # RF frequency of each IF-baseband bin for this mixing product
        f_rf = (f_lo - (f_if + f)) if below else (f_lo + f_if + f)
        Hp = tuned_response(np.abs(f_rf), pre_f0, pre_q, pre_stages) if pre_q else 1.0
        if below and pre_q:
            Hp = np.conj(Hp)
        Zb += E * Hp
        g = abs(tuned_response(st.f, pre_f0, pre_q, pre_stages)) if pre_q else 1.0
        info.append(dict(station=st, offset=d, below=below, pre_gain=float(g)))
    # antenna noise from both mixing sides, shaped by the preselector
    N0 = 1e-3 * 10 ** (noise_dbm_hz / 10) * 50 * 2           # V^2/Hz (peak-amplitude scale)
    sig = np.sqrt(N0 * fs / 2)
    for below in (True, False):
        f_rf = (f_lo - (f_if + f)) if below else (f_lo + f_if + f)
        Hp = tuned_response(np.abs(f_rf), pre_f0, pre_q, pre_stages) if pre_q else 1.0
        w = sig * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        Zb += np.fft.fft(w) * np.abs(Hp)
    Hif = butter_lp(f, if_bw / 2, if_order)
    za = np.fft.ifft(Zb * Hif)
    zb = np.fft.ifft(Zb)
    if return_parts:
        return zb, za, info, f, Hif
    return zb, za, info


def detect_am(z, fs, audio_bw=5000.0):
    """Ideal envelope detector + DC block + brick-wall audio filter (periodic record)."""
    env = np.abs(z)
    a = env - env.mean()
    A = np.fft.rfft(a)
    A[np.fft.rfftfreq(len(a), 1 / fs) > audio_bw] = 0
    return np.fft.irfft(A, len(a)), env.mean()


def tone_sinad(y, f0, fs):
    """SINAD (dB) of the tone at f0 in a periodic record y: least-squares fit of sin/cos."""
    t = np.arange(len(y)) / fs
    B = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)], 1)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    fit = B @ c
    return float(10 * np.log10(np.sum(fit ** 2) / max(np.sum((y - fit) ** 2), 1e-300)))


def agc_loop(env_in, fs, det_tc, attack_tc, decay_tc, ref=1.0, hang=0.0, max_gain_db=60.0):
    """Feedback AGC on an envelope (as in Figure ch04_agc): an averaging detector on the
    output, and a gain integrator in dB that moves fast when the level is too high (attack)
    and slowly when it is too low (decay). Returns (out, gain_db, detector)."""
    kdet = 1.0 / max(det_tc * fs, 1.0)
    ka = 1.0 / max(attack_tc * fs, 1.0)
    kd = 1.0 / max(decay_tc * fs, 1.0)
    lref = 20 * np.log10(ref)
    g = 0.0
    det = ref
    x = np.asarray(env_in, float).tolist()
    out = np.empty(len(x))
    gains = np.empty(len(x))
    dets = np.empty(len(x))
    hang_n = int(hang * fs)
    hold = 0
    for i, e in enumerate(x):
        o = e * 10 ** (g / 20)
        out[i] = o
        det += kdet * (o - det)
        err = 20 * np.log10(det if det > 1e-9 else 1e-9) - lref
        if err > 0:
            g -= ka * err
            hold = hang_n
        elif hold > 0:
            hold -= 1
        else:
            g -= kd * err
        if g > max_gain_db:
            g = max_gain_db
        gains[i] = g
        dets[i] = det
    return out, gains, dets


def two_tone_cubic(x, a1, iip3_dbm):
    """Compressive third-order nonlinearity on a complex envelope x (volts, 50 ohms) with
    input intercept ``iip3_dbm``: the band-pass equivalent of y = a1 x + a3 x^3 is
    a1 x + (3/4) a3 |x|^2 x, and A_IIP3^2 = (4/3)|a1/a3|, so y = a1 (x - |x|^2 x / A_IIP3^2)."""
    A2 = dbm_to_amp(iip3_dbm) ** 2
    return a1 * (x - np.abs(x) ** 2 * x / A2)
