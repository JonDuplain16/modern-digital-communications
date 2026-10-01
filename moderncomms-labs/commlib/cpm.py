"""cpm: continuous-phase modulation (MSK, GMSK, GFSK), waveform metrics and EVM.

Companion of Chapter 9 (*Digital Modulation and Optimal Detection*), sections on CPM and on
EVM. ``gmsk_baseband`` is the generator that drew the chapter's CPM figures
(book/figscripts/ch09_figs.py), so spectra and bandwidths match the book. Import explicitly:

    from commlib import cpm
    x, phase = cpm.gmsk_baseband(bits, sps=16, BT=0.3)       # GSM-like GMSK
    bw = cpm.occupied_bandwidth(x, fs=16)                      # 99% bandwidth in units of Rb
    rx = cpm.LaurentReceiver(BT=0.3, sps=16)                   # coherent linearised receiver

Conventions: one sample period = 1/sps of a bit; frequencies are in units of the bit rate
when fs = sps.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig
from scipy.special import erfc

from .filters import rrc_taps
from .modulation import get_constellation

__all__ = ["gmsk_baseband", "gmsk_freq_pulse", "laurent_c0", "msk_precode", "LaurentReceiver",
           "differential_detect", "discriminator_detect", "psd_normalized", "occupied_bandwidth",
           "shaped_waveform", "papr_ccdf", "evm", "evm_db", "mer_db", "snr_eff_db", "evm_budget_db",
           "phase_noise_awgn", "dc_offset", "rapp_pa_shaped"]


def _q(x):
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


# ============================================================================ generation
def gmsk_baseband(bits, sps, BT, h=0.5):
    """GMSK / MSK / GFSK complex envelope (unit amplitude) and its phase.

    bits: 0/1 array (mapped to a = 2b - 1); sps: samples per bit; BT: Gaussian filter
    bandwidth-time product (None gives plain CPFSK, i.e. MSK when h = 0.5); h: modulation index.
    The Gaussian filter spans +-2 bits and is centred on each bit (as in the book's figures)."""
    a = 2.0 * np.asarray(bits) - 1
    nrz = np.repeat(a, sps)
    if BT is None:
        freq = nrz
    else:
        t = np.arange(-2 * sps, 2 * sps + 1) / sps
        hg = np.exp(-2 * np.pi ** 2 * BT ** 2 * t ** 2 / np.log(2))
        hg /= hg.sum()
        freq = np.convolve(nrz, hg, mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps
    return np.exp(1j * phase), phase


def gmsk_freq_pulse(t, BT=None):
    """Frequency pulse g(t) * T_b (t in bit periods, centred on 0), eq. (9.gmsk). BT=None: MSK."""
    t = np.asarray(t, float)
    if BT is None:
        return np.where(np.abs(t) <= 0.5, 0.5, 0.0)
    k = 2 * np.pi * BT / np.sqrt(np.log(2))
    return 0.5 * (_q(k * (t - 0.5)) - _q(k * (t + 0.5)))


def laurent_c0(sps, BT=None, h=0.5, L=None):
    """Main Laurent pulse C0 (sampled at sps per bit) of binary CPM with a Gaussian (or
    rectangular) frequency pulse truncated to L bits. Its support is (L + 1) bits.
    For MSK (BT=None, L=1) C0 is the half-sine of duration 2T, and MSK is exactly OQPSK."""
    if L is None:
        L = 1 if BT is None else 4
    tt = (np.arange(L * sps * 8) + 0.5) / (8 * sps) * L - L / 2        # fine grid over the pulse
    g = gmsk_freq_pulse(tt, BT)
    q = np.concatenate([[0], np.cumsum(g) * (tt[1] - tt[0])])           # phase pulse, q(LT) ~ 1/2
    q = q / q[-1] * 0.5
    tq = np.concatenate([[0], tt + L / 2 + 0.5 * (tt[1] - tt[0])])      # time from pulse start
    def S(t):
        t = np.asarray(t, float)
        psi = np.where((t >= 0) & (t <= L), 2 * np.pi * h * np.interp(t, tq, q),
                       np.where((t > L) & (t <= 2 * L), np.pi * h - 2 * np.pi * h * np.interp(t - L, tq, q), 0.0))
        return np.sin(psi) / np.sin(np.pi * h)
    t = np.arange(int((L + 1) * sps) + 1) / sps
    c0 = np.ones_like(t)
    for i in range(L):
        c0 *= S(t + i)
    return c0


def msk_precode(c):
    """Differential precoding a_k = c_k c_{k-1} (in 0/1 form) so that a coherent linear (OQPSK-
    like) receiver reads the data bits c directly, without differential decoding."""
    s = 1 - 2 * np.asarray(c)                # 0 -> +1, 1 -> -1
    a = s * np.r_[1, s[:-1]]
    return ((1 + a) // 2).astype(int)           # bit 1 <-> a = +1 in gmsk_baseband


class LaurentReceiver:
    """Coherent linearised receiver for MSK/GMSK (h = 1/2): a filter matched to the main
    Laurent pulse C0, sampled once per bit and derotated by j^-k. With the transmit bits
    precoded by ``msk_precode`` the sign of the output is the data bit (0 -> +1).

    The sampling offset and the constant phase are found once from a noiseless training
    waveform with ``calibrate`` (data-aided synchronisation, Chapter 10's job)."""

    def __init__(self, sps, BT=None, L=None):
        self.sps, self.BT = sps, BT
        self.c0 = laurent_c0(sps, BT, 0.5, L)
        self.offset, self.rot = 0, 1.0

    def soft(self, y, n_bits):
        z = _sig.fftconvolve(y, self.c0[::-1])
        idx = self.offset + self.sps * np.arange(n_bits)
        idx = idx[idx < len(z)]
        k = np.arange(len(idx))
        return np.real(z[idx] * (1j) ** (-k) * self.rot) / np.sum(self.c0 ** 2)

    def calibrate(self, x_clean, data_bits):
        """Search the offset/phase that best match a known noiseless waveform."""
        s = 1 - 2 * np.asarray(data_bits)
        best = (-1, 0, 1.0)
        z = _sig.fftconvolve(x_clean, self.c0[::-1])
        k = np.arange(len(s))
        for off in range(0, 4 * self.sps):
            idx = off + self.sps * k
            if idx[-1] >= len(z):
                break
            v = z[idx] * (1j) ** (-k)
            c = np.vdot(s.astype(complex), v)
            if abs(c) > best[0]:
                best = (abs(c), off, np.exp(-1j * np.angle(c)))
        _, self.offset, self.rot = best
        return self.offset

    def detect(self, y, n_bits):
        return (self.soft(y, n_bits) < 0).astype(int)


def differential_detect(y, sps, delay_bits=1, prefilter_bits=1.0):
    """One-bit differential detector for MSK/GMSK/GFSK: sign of Im{y[n] y*[n - sps]} sampled at
    the end of each bit. An optional moving-average prefilter (length in bits) limits noise.
    Returns soft values whose sign gives a = 2b - 1 (positive -> bit 1)."""
    y = np.asarray(y)
    if prefilter_bits:
        m = max(1, int(round(prefilter_bits * sps)))
        y = _sig.lfilter(np.ones(m) / m, 1, y)
        shift = (m - 1) // 2
        y = np.r_[y[shift:], np.zeros(shift, complex)]
    d = sps * delay_bits
    y = np.r_[np.full(d, y[0]), y]              # the phase before the first bit is the start phase
    v = np.imag(y[d:] * np.conj(y[:-d]))        # v[j] compares y[j + d] with y[j]
    return v[np.arange(sps - 1, len(y) - d, sps)]


def discriminator_detect(y, sps, prefilter_bits=1.0):
    """Limiter-discriminator receiver: instantaneous frequency (phase difference per sample),
    integrated over each bit. Positive -> bit 1."""
    y = np.asarray(y)
    if prefilter_bits:
        m = max(1, int(round(prefilter_bits * sps)))
        y = _sig.lfilter(np.ones(m) / m, 1, y)
        shift = (m - 1) // 2
        y = np.r_[y[shift:], np.zeros(shift, complex)]
    f = np.angle(y[1:] * np.conj(y[:-1]))
    f = np.r_[0.0, f]
    n = len(y) // sps
    return f[:n * sps].reshape(n, sps).sum(axis=1)


# ============================================================================ spectra
def psd_normalized(x, fs=1.0, nper=4096):
    """Two-sided Welch PSD normalised to unit total power (fftshifted). Returns (f, p)."""
    f, p = _sig.welch(x, fs=fs, nperseg=min(nper, len(x)), return_onesided=False, window="blackmanharris")
    f = np.fft.fftshift(f); p = np.fft.fftshift(p)
    return f, p / np.trapezoid(p, f)


def occupied_bandwidth(x, fs=1.0, frac=0.99, nper=4096):
    """Bandwidth containing `frac` of the power (symmetric tails of (1 - frac)/2 each)."""
    f, p = psd_normalized(x, fs, nper)
    c = np.cumsum(p) * (f[1] - f[0])
    lo = np.interp((1 - frac) / 2, c, f)
    hi = np.interp(1 - (1 - frac) / 2, c, f)
    return hi - lo


def shaped_waveform(kind, nsym=4000, sps=16, beta=0.35, rng=None):
    """Baseband waveform of a modulation for envelope/PAPR studies: 'qpsk', 'oqpsk', 'pi4qpsk',
    '8psk', '16qam', '64qam', ... (RRC pulses, roll-off beta), or 'gmsk' (BT = 0.3, 2 bits per symbol
    period so the bit rate matches QPSK). Returns (x, sps)."""
    rng = np.random.default_rng(7) if rng is None else rng
    h = rrc_taps(beta, sps, span=12)
    up = lambda s: np.concatenate([np.asarray(s)[:, None], np.zeros((len(s), sps - 1))], axis=1).ravel()
    trim = lambda y: y[len(h):-len(h)]
    if kind == "gmsk":
        x, _ = gmsk_baseband(rng.integers(0, 2, 2 * nsym), sps // 2, 0.3)
        return x, sps
    if kind == "oqpsk":
        s = get_constellation("qpsk").points[rng.integers(0, 4, nsym)]
        xi = _sig.fftconvolve(up(s.real), h); xq = _sig.fftconvolve(up(s.imag), h)
        xq = np.r_[np.zeros(sps // 2), xq[:-(sps // 2)]]
        return trim(xi + 1j * xq), sps
    if kind == "pi4qpsk":
        d = rng.integers(0, 4, nsym)
        s = np.exp(1j * np.cumsum(np.array([1, 3, -3, -1])[d] * np.pi / 4))
        return trim(_sig.fftconvolve(up(s), h)), sps
    c = get_constellation(kind)
    s = c.points[rng.integers(0, c.M, nsym)]
    return trim(_sig.fftconvolve(up(s), h)), sps


def papr_ccdf(x, grid_db):
    """CCDF P(|x|^2 / mean|x|^2 > g) of the instantaneous power, for g in grid_db (dB)."""
    p = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
    pd = 10 * np.log10(np.maximum(p, 1e-12))
    pd.sort()
    return 1 - np.searchsorted(pd, np.asarray(grid_db), side="right") / len(pd)


# ============================================================================ EVM / impairments
def evm(y, s):
    """rms EVM (fraction), eq. (9.evm): sqrt(sum|y - s|^2 / sum|s|^2)."""
    y, s = np.asarray(y), np.asarray(s)
    return float(np.sqrt(np.sum(np.abs(y - s) ** 2) / np.sum(np.abs(s) ** 2)))


def evm_db(y, s):
    return 20 * np.log10(evm(y, s))


def mer_db(y, s):
    """Modulation error ratio (dB) = -EVM_dB when both are normalised to average power."""
    return -evm_db(y, s)


def snr_eff_db(snr_db, *evm_terms_db):
    """Effective SNR (dB) from a thermal SNR and any number of EVM terms (dB), eq. (9.evmsnr)."""
    tot = 10 ** (-np.asarray(snr_db, float) / 10)
    for e in evm_terms_db:
        tot = tot + 10 ** (np.asarray(e, float) / 10)
    return -10 * np.log10(tot)


def evm_budget_db(*terms_db):
    """Total EVM (dB) of independent contributions (error powers add)."""
    return 10 * np.log10(np.sum([10 ** (t / 10) for t in terms_db]))


def phase_noise_awgn(s, sigma_deg, rng=None):
    """Apply i.i.d. Gaussian phase jitter of sigma_deg rms (degrees) to symbols s."""
    rng = np.random.default_rng() if rng is None else rng
    return np.asarray(s) * np.exp(1j * np.deg2rad(sigma_deg) * rng.standard_normal(len(s)))


def dc_offset(s, level, phase=0.6):
    """Carrier leakage: add a constant level*exp(j phase) (relative to rms symbol amplitude)."""
    return np.asarray(s) + level * np.exp(1j * phase)


def rapp_pa_shaped(s, ibo_db, sps=8, beta=0.25, p=2.0):
    """Pass symbols through RRC shaping, a Rapp PA at the given input back-off (dB from the
    saturation level, relative to the rms of the shaped signal), and the matched filter.
    Returns the received symbols normalised to unit power (same length as s)."""
    from .channel import rapp_pa
    s = np.asarray(s)
    h = rrc_taps(beta, sps, span=12)
    u = np.zeros(len(s) * sps, complex); u[::sps] = s
    x = _sig.fftconvolve(u, h)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    sat = 10 ** (ibo_db / 20)
    xp = rapp_pa(x / rms, sat=sat, p=p) * rms
    z = _sig.fftconvolve(xp, h)[len(h) - 1::sps][:len(s)]
    g = np.vdot(s, z) / np.vdot(s, s)          # remove the average complex gain (as an analyser does)
    return z / g
