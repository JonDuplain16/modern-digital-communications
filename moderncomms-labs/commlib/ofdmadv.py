"""ofdmadv: OFDM system-level tools beyond commlib.ofdm.

Companion of Chapter 17 (*OFDM and Multicarrier*): clean copies of the helpers in
book/figscripts/ch17_figs.py plus a few additions (ACLR, a Rapp PA on OFDM, radar). Import
explicitly:

    from commlib import ofdmadv as oa
    H = oa.tdl_freq_response("EVA", k, 1024, 15.36e6, rng)       # one static channel draw
    Hlin, Hdft, Hmm = oa.chest_all(Hls_p, kp, k, n0, pdp, N)    # three estimators
    y = oa.clip_filter(x, cfg, 4.0, iters=4)                    # crest-factor reduction

Conventions follow commlib.ofdm: OFDMConfig(nfft, n_used, ncp) with signed subcarrier
indices cfg.k and FFT bins cfg.active; unit-power data subcarriers.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig

from .channel import TDL_PROFILES, jakes_process, phase_noise as _phase_noise, rapp_pa
from .ofdm import OFDMConfig, ofdm_modulate, ofdm_demodulate

__all__ = ["FS_LTE", "tdl_pdp", "tdl_freq_response", "tdl_static_taps", "chest_ls_interp", "chest_dft",
           "lmmse_weights", "chest_all", "sir_cfo_db", "sir_doppler_db", "cpe_correct", "clip",
           "clip_filter", "bussgang_evm_db", "aclr_db", "aclr_symbols", "wola_ofdm", "filtered_ofdm", "psd",
           "dsl_snr_db", "dmt_bit_loading", "waterfill_bisect", "radar_echo", "radar_map"]

FS_LTE = 15.36e6          # 10 MHz LTE/NR sampling rate (1024-point FFT at 15 kHz)
C0 = 299_792_458.0


# ============================================================================ channels
def tdl_pdp(profile, fs):
    """(sample delays, normalised powers) of a TDL profile ('EPA', 'EVA', 'ETU') at rate fs."""
    dl, pdb = TDL_PROFILES[profile]
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
    return np.round(np.asarray(dl) * 1e-9 * fs).astype(int), p


def tdl_freq_response(profile, k, N, fs=FS_LTE, rng=None, n_draws=None):
    """Frequency response H[k] of static Rayleigh draws of a TDL profile on signed subcarrier
    indices k of an N-point FFT at rate fs. Returns (n_draws, len(k)) (or (len(k),) if None)."""
    rng = np.random.default_rng() if rng is None else rng
    d, p = tdl_pdp(profile, fs)
    B = 1 if n_draws is None else n_draws
    g = np.sqrt(p / 2) * (rng.standard_normal((B, len(p))) + 1j * rng.standard_normal((B, len(p))))
    H = g @ np.exp(-2j * np.pi * np.outer(d, np.asarray(k)) / N)
    return H[0] if n_draws is None else H


def tdl_static_taps(profile, fs, rng=None):
    """One static realisation of a TDL profile as a sample-spaced impulse response."""
    rng = np.random.default_rng() if rng is None else rng
    d, p = tdl_pdp(profile, fs)
    h = np.zeros(d.max() + 1, dtype=complex)
    for di, pi in zip(d, p):
        h[di] += np.sqrt(pi / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
    return h


# ============================================================================ channel estimation
def chest_ls_interp(Hls_p, kp, k):
    """Linear interpolation of pilot LS estimates (real and imaginary parts) onto subcarriers k."""
    return np.interp(k, kp, Hls_p.real) + 1j * np.interp(k, kp, Hls_p.imag)


def chest_dft(Hls_p, kp, k, N, max_delay):
    """DFT (delay-domain) estimator: least-squares fit of taps on a delay grid of resolution
    N/(Np * Dp) samples up to max_delay samples, re-evaluated on all subcarriers k."""
    Np = len(kp)
    Dp = (kp[-1] - kp[0]) / max(Np - 1, 1)
    res = N / (Np * Dp)
    m = np.arange(int(np.ceil(max_delay / res))) * res
    Fp = np.exp(-2j * np.pi * np.outer(kp, m) / N)
    hk = np.linalg.lstsq(Fp, Hls_p, rcond=None)[0]
    return np.exp(-2j * np.pi * np.outer(k, m) / N) @ hk


def lmmse_weights(kp, k, n0, delays, powers, N):
    """LMMSE interpolation matrix W (len(k) x len(kp)) from the PDP prior:
    W = R_hp (R_pp + n0 I)^-1 with R(ka, kb) = sum_l p_l exp(-j 2 pi (ka - kb) d_l / N)."""
    def Rf(ka, kb):
        return (powers[None, None, :] * np.exp(-2j * np.pi * (ka[:, None, None] - kb[None, :, None])
                                               * delays[None, None, :] / N)).sum(-1)
    kp = np.asarray(kp, float); k = np.asarray(k, float)
    return Rf(k, kp) @ np.linalg.inv(Rf(kp, kp) + n0 * np.eye(len(kp)))


def chest_all(Hls_p, kp, k, n0, delays, powers, N, max_delay):
    """Return (linear interpolation, DFT-based, LMMSE) estimates on subcarriers k."""
    return (chest_ls_interp(Hls_p, kp, k), chest_dft(Hls_p, kp, k, N, max_delay),
            lmmse_weights(kp, k, n0, delays, powers, N) @ Hls_p)


# ============================================================================ ICI and phase noise
def sir_cfo_db(eps):
    """Signal-to-ICI ratio for a normalised CFO eps: sinc^2(eps) / (1 - sinc^2(eps))."""
    s2 = np.sinc(np.asarray(eps, float)) ** 2
    return 10 * np.log10(s2 / (1 - s2))


def sir_doppler_db(fdT):
    """Approximate signal-to-ICI ratio for Jakes Doppler fD*T: 6 / (pi fD T)^2."""
    return 10 * np.log10(6 / (np.pi * np.asarray(fdT, float)) ** 2)


def cpe_correct(Y, X_ref):
    """Remove the common phase error of each OFDM symbol (row) using reference symbols
    (pilots or decisions) X_ref of the same shape. Returns (corrected Y, CPE in radians)."""
    cpe = np.angle(np.sum(Y * np.conj(X_ref), axis=1, keepdims=True))
    return Y * np.exp(-1j * cpe), cpe[:, 0]


# ============================================================================ PAPR, PA and spectra
def clip(x, A):
    """Envelope clipping at amplitude A, preserving phase."""
    a = np.abs(x)
    return np.where(a > A, x * A / np.maximum(a, 1e-12), x)


def clip_filter(x, cfg, clip_db, iters=1):
    """Iterative clipping and filtering (Armstrong) of oversampled OFDM symbols without CP.

    x: (n_sym, cfg.nfft) time-domain symbols (cfg.nfft is the oversampled FFT size);
    clip_db: clipping level above the RMS. After each clip, out-of-band bins are zeroed."""
    x = np.atleast_2d(x)
    A = 10 ** (clip_db / 20) * np.sqrt(np.mean(np.abs(x) ** 2))
    mask = np.zeros(cfg.nfft, bool); mask[cfg.active] = True
    y = x.copy()
    for _ in range(iters):
        y = clip(y, A)
        Y = np.fft.fft(y, axis=1); Y[:, ~mask] = 0
        y = np.fft.ifft(Y, axis=1)
    return y


def bussgang_evm_db(y_syms, x_syms, cfg, grid):
    """In-band EVM (dB) after removing the Bussgang gain: demodulate oversampled symbols
    (n_sym, cfg.nfft) and compare with the transmitted grid (n_sym, n_used)."""
    Y = np.fft.fft(np.atleast_2d(y_syms), axis=1)[:, cfg.active]
    a = np.sum(Y * np.conj(grid)) / np.sum(np.abs(grid) ** 2)
    return 10 * np.log10(np.mean(np.abs(Y / a - grid) ** 2) / np.mean(np.abs(grid) ** 2) + 1e-15)


def psd(x, fs=1.0, nfft=4096):
    """Two-sided Welch PSD (Hann, 50% overlap), frequencies ascending."""
    f, p = _sig.welch(x, fs=fs, nperseg=nfft, return_onesided=False, window="hann", noverlap=nfft // 2)
    i = np.argsort(f)
    return f[i], p[i]


def aclr_db(x, fs, bw_main, offset, bw_adj=None, nfft=4096):
    """Adjacent-channel leakage ratio (dB): power in [-bw/2, bw/2] over the larger of the two
    adjacent channels centred at +-offset with width bw_adj (default bw_main)."""
    bw_adj = bw_main if bw_adj is None else bw_adj
    f, p = psd(x, fs, nfft)
    main = p[np.abs(f) <= bw_main / 2].sum()
    adj = max(p[np.abs(f - o) <= bw_adj / 2].sum() for o in (offset, -offset))
    return 10 * np.log10(main / adj)


def aclr_symbols(y_syms, cfg, guard=None):
    """ACLR (dB) of oversampled OFDM symbols without CP, from the average per-symbol (cyclic)
    spectrum, so only in-symbol distortion counts (not the symbol-boundary sidelobes).
    The main channel is the occupied band |k| <= n_used/2; the adjacent channels have the same
    width, starting `guard` subcarriers (default 5% of n_used) beyond the band edge."""
    y = np.atleast_2d(y_syms)
    n = y.shape[1]
    P = np.mean(np.abs(np.fft.fft(y, axis=1)) ** 2, axis=0)
    kk = np.fft.fftfreq(n, 1 / n)
    half = cfg.n_used / 2
    g = 0.05 * cfg.n_used if guard is None else guard
    main = P[np.abs(kk) <= half].sum()
    lo, hi = half + g, half + g + cfg.n_used
    adj = max(P[(kk > lo) & (kk <= hi)].sum(), P[(kk < -lo) & (kk >= -hi)].sum())
    return 10 * np.log10(main / adj)


def wola_ofdm(grid, nfft, ncp, w):
    """Weighted overlap-and-add CP-OFDM: each symbol is extended by w cyclic samples and
    tapered by raised-cosine ramps of w samples at both ends; adjacent symbols overlap by w."""
    grid = np.atleast_2d(grid)
    cfg = OFDMConfig(nfft, grid.shape[1], 0)
    xs = ofdm_modulate(grid, cfg).reshape(len(grid), nfft)
    Ls = nfft + ncp
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(w) + 0.5) / w))
    win = np.concatenate([ramp, np.ones(Ls - w), ramp[::-1]])
    ext = np.concatenate([xs[:, -ncp:], xs, xs[:, :w]], axis=1) * win
    out = np.zeros(len(grid) * Ls + w, complex)
    for s in range(len(grid)):
        out[s * Ls:s * Ls + Ls + w] += ext[s]
    return out


def filtered_ofdm(x, fs, passband_hz, ntap):
    """Filtered OFDM: a windowed-sinc low-pass (Hann^0.6 window) of total passband width
    passband_hz applied to the CP-OFDM waveform (delay compensated)."""
    n = np.arange(ntap) - ntap // 2
    bw = passband_hz / 2
    h = 2 * bw / fs * np.sinc(2 * bw / fs * n) * np.hanning(ntap) ** 0.6
    h /= h.sum()
    return np.convolve(x, h)[ntap // 2:ntap // 2 + len(x)]


# ============================================================================ DMT / DSL
def dsl_snr_db(f_hz, loop_km, tx_dbm_hz=-40.0, rfi_db=30.0, rfi_hz=1.35e6):
    """SNR per tone of the chapter's ADSL2+ model: attenuation 23 L sqrt(f/MHz) + 2 L dB,
    noise -140 dBm/Hz plus crosstalk rising as f^1.5, and an AM-band ingress bump."""
    f = np.asarray(f_hz, float)
    att = 23.0 * loop_km * np.sqrt(f / 1e6) + 2.0 * loop_km
    noise = 10 * np.log10(10 ** (-140 / 10) + 10 ** ((-128 + 15 * np.log10(f / 1e6)) / 10))
    rfi = rfi_db * np.exp(-0.5 * ((f - rfi_hz) / 8e3) ** 2)
    return tx_dbm_hz - att - (noise + rfi)


def dmt_bit_loading(snr_db, gap_db=11.8, bmax=15, min_bits=2):
    """Integer bits per tone floor(log2(1 + SNR/Gamma)), clipped to bmax; tones that would
    carry fewer than min_bits are not loaded (ADSL does not use 1-bit tones)."""
    b = np.clip(np.floor(np.log2(1 + 10 ** ((np.asarray(snr_db) - gap_db) / 10))), 0, bmax)
    b[b < min_bits] = 0
    return b.astype(int)


def waterfill_bisect(inv_gain, P_total, iters=200):
    """Water level and powers by bisection: P_k = max(mu - inv_gain_k, 0), sum = P_total."""
    lo, hi = 0.0, float(np.max(inv_gain) + P_total)
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        if np.sum(np.maximum(mu - inv_gain, 0)) > P_total:
            hi = mu
        else:
            lo = mu
    return np.maximum(mu - inv_gain, 0), mu


# ============================================================================ OFDM radar
def radar_echo(X, targets, df, Tsym, fc, n0=0.0, rng=None):
    """Monostatic OFDM radar echo on a grid X (subcarriers x symbols): each target
    (range_m, velocity_mps, amplitude) adds a X exp(-j2 pi m df 2R/c) exp(+j2 pi n Tsym 2 v fc/c)."""
    M, Nsym = X.shape
    m = np.arange(M)[:, None]; n = np.arange(Nsym)[None, :]
    Y = np.zeros_like(X, dtype=complex)
    for R, v, a in targets:
        Y += a * X * np.exp(-2j * np.pi * m * df * 2 * R / C0) * np.exp(2j * np.pi * n * Tsym * 2 * v * fc / C0)
    if n0 > 0:
        rng = np.random.default_rng() if rng is None else rng
        Y += np.sqrt(n0 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return Y


def radar_map(Y, X, window=True):
    """Range-Doppler periodogram: divide out the data, window, IFFT over subcarriers (range)
    and FFT over symbols (Doppler, fftshifted). Returns |map|^2 (range bins x Doppler bins)."""
    Z = Y / X
    M, Nsym = Z.shape
    if window:
        Z = Z * np.hanning(M)[:, None] * np.hanning(Nsym)[None, :]
    return np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifft(Z, axis=0), axis=1), axes=1)) ** 2
