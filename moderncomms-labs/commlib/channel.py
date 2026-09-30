"""Channel impairments: AWGN, CFO, phase noise, IQ imbalance, multipath and fading."""
from __future__ import annotations

import numpy as np

from .filters import fractional_delay_taps

__all__ = ["awgn", "awgn_esn0", "apply_cfo", "phase_noise", "iq_imbalance",
           "fractional_delay", "multipath", "jakes_process", "tdl_channel",
           "TDL_PROFILES", "fspl_db", "log_distance_pl_db", "rapp_pa"]


def awgn(x, snr_db, rng=None):
    """Add complex AWGN so that mean|x|^2 / noise variance = snr_db."""
    rng = np.random.default_rng(rng)
    p = np.mean(np.abs(x) ** 2)
    n0 = p / 10 ** (snr_db / 10)
    n = np.sqrt(n0 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n


def awgn_esn0(x, esn0_db, sps=1, rng=None, es=None):
    """Add noise for a target Es/N0.

    Works on symbol-rate (sps=1) or oversampled waveforms built with unit-energy
    pulses. Es is measured from the signal (mean power x sps) unless given.
    Returns (y, n0) where n0 is the complex noise variance per sample.
    """
    rng = np.random.default_rng(rng)
    if es is None:
        es = np.mean(np.abs(x) ** 2) * sps
    n0 = es / 10 ** (esn0_db / 10)
    n = np.sqrt(n0 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + n, n0


def apply_cfo(x, f_norm, phase0=0.0):
    """Multiply by exp(j 2 pi f_norm n + phase0); f_norm is cycles/sample."""
    n = np.arange(len(x))
    return x * np.exp(1j * (2 * np.pi * f_norm * n + phase0))


def phase_noise(n, linewidth_norm, rng=None):
    """Wiener phase noise process; linewidth_norm = 3-dB linewidth / sample rate."""
    rng = np.random.default_rng(rng)
    steps = rng.standard_normal(n) * np.sqrt(2 * np.pi * linewidth_norm)
    return np.exp(1j * np.cumsum(steps))


def iq_imbalance(x, gain_db=0.5, phase_deg=3.0):
    """Receiver IQ imbalance: y = mu x + nu conj(x)."""
    g = 10 ** (gain_db / 20)
    phi = np.deg2rad(phase_deg)
    i = x.real
    q = g * (np.cos(phi) * x.imag + np.sin(phi) * x.real)
    return i + 1j * q


def fractional_delay(x, d, ntaps=21):
    """Delay x by d samples (any real d) using integer shift + windowed sinc."""
    di = int(np.floor(d))
    df = d - di
    h = fractional_delay_taps(df, ntaps)
    y = np.convolve(x, h)[(ntaps - 1) // 2:][:len(x)]
    return np.concatenate([np.zeros(di, dtype=y.dtype), y])[:len(x)] if di >= 0 else y


def multipath(x, delays, gains):
    """Static multipath with (possibly fractional) sample delays and complex gains."""
    y = np.zeros(len(x) + int(np.ceil(max(delays))) + 1, dtype=complex)
    for d, g in zip(delays, gains):
        xd = fractional_delay(np.concatenate([x, np.zeros(int(np.ceil(max(delays))) + 1)]), d)
        y += g * xd
    return y


def jakes_process(n, fd_norm, n_sin=16, rng=None):
    """Unit-power Rayleigh fading process with Clarke/Jakes Doppler spectrum.

    Sum-of-sinusoids model (Zheng & Xiao, 2003) with random angles and phases.
    fd_norm = maximum Doppler / sample rate.
    """
    rng = np.random.default_rng(rng)
    t = np.arange(n)
    theta = rng.uniform(-np.pi, np.pi)
    k = np.arange(1, n_sin + 1)
    alpha = (2 * np.pi * k - np.pi + theta) / (4 * n_sin)
    phi = rng.uniform(-np.pi, np.pi, n_sin)
    psi = rng.uniform(-np.pi, np.pi, n_sin)
    wd = 2 * np.pi * fd_norm
    hi = np.cos(wd * np.outer(t, np.cos(alpha)) + phi).sum(axis=1)
    hq = np.cos(wd * np.outer(t, np.sin(alpha)) + psi).sum(axis=1)
    return (hi + 1j * hq) / np.sqrt(n_sin)


# 3GPP TS 36.101 Annex B.2 extended models (delay in ns, relative power in dB)
TDL_PROFILES = {
    "EPA": ([0, 30, 70, 90, 110, 190, 410],
            [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]),
    "EVA": ([0, 30, 150, 310, 370, 710, 1090, 1730, 2510],
            [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]),
    "ETU": ([0, 50, 120, 200, 230, 500, 1600, 2300, 5000],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]),
}


def tdl_channel(x, fs, profile="EVA", fd_hz=5.0, rng=None, return_taps=False):
    """Time-varying tapped-delay-line channel.

    Each path is an independent Jakes Rayleigh process; delays are rounded to
    the nearest sample (adequate when fs >> 1/min delay spacing is not needed).
    """
    rng = np.random.default_rng(rng)
    delays_ns, pdb = TDL_PROFILES[profile] if isinstance(profile, str) else profile
    p = 10 ** (np.asarray(pdb) / 10)
    p = p / p.sum()
    d = np.round(np.asarray(delays_ns) * 1e-9 * fs).astype(int)
    L = d.max() + 1
    y = np.zeros(len(x) + L - 1, dtype=complex)
    taps = np.zeros((len(x), L), dtype=complex)
    for di, pi in zip(d, p):
        g = np.sqrt(pi) * jakes_process(len(x), fd_hz / fs, rng=rng)
        taps[:, di] += g
        y[di:di + len(x)] += g * x
    return (y, taps) if return_taps else y


def fspl_db(d_m, f_hz):
    """Free-space path loss (Friis) in dB."""
    return 20 * np.log10(4 * np.pi * np.asarray(d_m) * f_hz / 3e8)


def log_distance_pl_db(d_m, f_hz, n=3.0, d0=1.0, sigma_db=0.0, rng=None):
    """Log-distance path loss with optional log-normal shadowing."""
    rng = np.random.default_rng(rng)
    pl = fspl_db(d0, f_hz) + 10 * n * np.log10(np.asarray(d_m) / d0)
    return pl + sigma_db * rng.standard_normal(np.shape(pl))


def rapp_pa(x, sat=1.0, p=2.0):
    """Rapp solid-state PA model (AM/AM only)."""
    a = np.abs(x)
    g = 1 / (1 + (a / sat) ** (2 * p)) ** (1 / (2 * p))
    return x * g
