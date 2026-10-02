"""fading: small-scale fading generators, statistics and diversity theory (Chapter 11).

Complements commlib.channel (jakes_process, tdl_channel) and commlib.propagation with:

* SoSFader            -- continuous-time sum-of-sinusoids Rayleigh/Rician fader for several
                         independent paths; the same object always returns the same realisation,
                         so successive time windows join seamlessly (live animations)
* envelope_pdf        -- Rayleigh, Rician (K) and Nakagami-m envelope densities (unit power)
* power_cdf           -- P(|h|^2 < x): the probability of a fade deeper than x
* lcr_rayleigh, afd_rayleigh -- Rice's level-crossing rate and average fade duration
* fade_stats          -- measured crossings per second and fade durations of an envelope
* ber_bpsk_diversity  -- BPSK error rate over L i.i.d. Rician/Nakagami branches, MRC or selection
* freq_correlation, coherence_bandwidth -- from a power-delay profile
* shadowing_field     -- spatially correlated log-normal shadowing map (exponential correlation)

Import explicitly:  from commlib import fading as fd
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = ["SoSFader", "envelope_pdf", "power_cdf", "lcr_rayleigh", "afd_rayleigh", "fade_stats",
           "ber_bpsk_diversity", "freq_correlation", "coherence_bandwidth", "shadowing_field",
           "doppler_hz", "clarke_psd"]

C0 = 299_792_458.0


def doppler_hz(speed_kmh, f_hz):
    """Maximum Doppler shift v / lambda (Hz) for a speed in km/h."""
    return np.asarray(speed_kmh, float) / 3.6 * np.asarray(f_hz, float) / C0


def clarke_psd(f, fd):
    """Clarke/Jakes Doppler spectrum 1/(pi fd sqrt(1 - (f/fd)^2)) for |f| < fd (0 outside)."""
    f = np.asarray(f, float)
    x = f / fd
    out = np.zeros_like(x)
    m = np.abs(x) < 1
    out[m] = 1 / (np.pi * fd * np.sqrt(1 - x[m] ** 2))
    return out


# ------------------------------------------------------------------------ generator
class SoSFader:
    """Sum-of-sinusoids fader: n_paths independent unit-power processes, each the sum of
    n_sin plane waves arriving from angles stratified around the circle (isotropic
    scattering, so the Doppler spectrum approaches Clarke's bathtub).

    ``gains(t, fd_hz, K=0)`` returns an array (len(t), n_paths). K is the Rician factor
    (power in the line-of-sight wave / power in the scattered waves); K = 0 is Rayleigh.
    Because the angles and phases are fixed at construction, calling ``gains`` for
    consecutive time windows produces one continuous realisation.
    """

    def __init__(self, n_paths=1, n_sin=32, rng=None):
        rng = np.random.default_rng(rng)
        self.n_paths, self.n_sin = int(n_paths), int(n_sin)
        n = np.arange(self.n_sin)
        self.cos_a = np.cos(2 * np.pi * (n[None, :] + rng.uniform(0, 1, (self.n_paths, self.n_sin)))
                            / self.n_sin)
        self.phi = rng.uniform(-np.pi, np.pi, (self.n_paths, self.n_sin))
        self.los_cos = np.cos(rng.uniform(-np.pi, np.pi, self.n_paths))
        self.los_phi = rng.uniform(-np.pi, np.pi, self.n_paths)

    def gains(self, t, fd_hz, K=0.0):
        t = np.asarray(t, float)
        w = 2 * np.pi * fd_hz
        arg = w * t[:, None, None] * self.cos_a[None] + self.phi[None]
        h = np.exp(1j * arg).sum(axis=-1) / np.sqrt(self.n_sin)
        if K > 0:
            los = np.exp(1j * (w * t[:, None] * self.los_cos[None] + self.los_phi[None]))
            h = np.sqrt(1 / (K + 1)) * h + np.sqrt(K / (K + 1)) * los
        return h


# ------------------------------------------------------------------------ first-order statistics
def envelope_pdf(r, kind="rayleigh", K=0.0, m=1.0):
    """Density of the envelope r = |h| with E|h|^2 = 1. kind: 'rayleigh', 'rician', 'nakagami'."""
    r = np.asarray(r, float)
    if kind == "rayleigh" or (kind == "rician" and K <= 0):
        return 2 * r * np.exp(-r ** 2)
    if kind == "rician":
        x = 2 * r * np.sqrt(K * (K + 1))
        return 2 * (K + 1) * r * np.exp(-K - (K + 1) * r ** 2 + x) * special.i0e(x)
    if kind == "nakagami":
        with np.errstate(divide="ignore"):
            lg = (np.log(2) + m * np.log(m) - special.gammaln(m) + (2 * m - 1) * np.log(np.maximum(r, 1e-300)) - m * r ** 2)
        return np.where(r > 0, np.exp(lg), 0.0)
    raise ValueError(kind)


def power_cdf(x, kind="rayleigh", K=0.0, m=1.0):
    """P(|h|^2 < x) for unit-mean-power fading (x linear, relative to the mean)."""
    x = np.asarray(x, float)
    if kind == "rayleigh" or (kind == "rician" and np.all(np.asarray(K) <= 0)):
        return -np.expm1(-x)
    if kind == "rician":
        K = np.asarray(K, float)
        ric = stats.ncx2.cdf(2 * (K + 1) * x, df=2, nc=np.maximum(2 * K, 1e-9))
        return np.where(K > 0, ric, -np.expm1(-x))
    if kind == "nakagami":
        return stats.gamma.cdf(x, a=m, scale=1 / m)
    raise ValueError(kind)


# ------------------------------------------------------------------------ second-order statistics
def lcr_rayleigh(rho, fd):
    """Rice's level-crossing rate (crossings/s, one direction) of a Rayleigh envelope at
    rho = threshold / rms value, maximum Doppler fd (isotropic scattering)."""
    rho = np.asarray(rho, float)
    return np.sqrt(2 * np.pi) * fd * rho * np.exp(-rho ** 2)


def afd_rayleigh(rho, fd):
    """Average fade duration (s) below rho = threshold / rms of a Rayleigh envelope."""
    rho = np.asarray(rho, float)
    return np.expm1(rho ** 2) / (rho * fd * np.sqrt(2 * np.pi))


def fade_stats(env_db, thr_db, fs):
    """Measure fades of an envelope given in dB (any reference) against thr_db.

    Returns dict(rate=downward crossings per second, afd=mean fade duration (s) estimated as
    time-below / number of fades, frac=fraction of time below, durations=array of complete
    fade lengths (s), n=number of downward crossings)."""
    below = np.asarray(env_db) < thr_db
    T = len(below) / fs
    down = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    up = np.flatnonzero(~below[1:] & below[:-1]) + 1
    n = len(down)
    durs = []
    if n:
        ups = up[up > down[0]]
        k = min(len(ups), n)
        durs = (ups[:k] - down[:k]) / fs
    frac = float(np.mean(below)) if len(below) else 0.0
    rate = n / T if T > 0 else 0.0
    return dict(rate=rate, afd=(frac * T / n) if n else float("nan"), frac=frac,
                durations=np.asarray(durs, float), n=n)


# ------------------------------------------------------------------------ error rates
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def ber_bpsk_diversity(ebn0_db, L=1, K=0.0, m=None, combining="mrc"):
    """Average BPSK bit error rate over L independent, identically distributed branches,
    each with mean Eb/N0 = ebn0_db (dB). Fading: Rician with factor K (K = 0: Rayleigh), or
    Nakagami-m when m is given. combining: 'mrc' (maximal ratio) or 'sc' (selection).

    MRC uses Craig's formula with the moment-generating function; SC integrates
    P_b = integral of F(gamma)^L e^(-gamma) / (2 sqrt(pi gamma)) d gamma numerically."""
    g = 10 ** (np.atleast_1d(np.asarray(ebn0_db, float)) / 10)
    if combining == "mrc":
        th = (_GL_X + 1) * np.pi / 4                       # (0, pi/2)
        s = -1 / np.sin(th) ** 2                           # (n,)
        gs = g[:, None] * s[None, :]
        if m is not None:
            M = (1 - gs / m) ** (-m)
        else:
            M = (1 + K) / (1 + K - gs) * np.exp(K * gs / (1 + K - gs))
        out = (M ** L) @ _GL_W * (np.pi / 4) / np.pi
    else:
        u = np.linspace(0, 7, 1401)
        gam = u ** 2
        kind = "nakagami" if m is not None else "rician"
        F = power_cdf(gam[None, :] / g[:, None], kind, K=K, m=m if m is not None else 1.0)
        out = _trapz(F ** L * np.exp(-gam)[None, :], u, axis=1) / np.sqrt(np.pi)
    out = np.clip(out, 0, 0.5)
    return out if np.ndim(ebn0_db) else float(out[0])


# ------------------------------------------------------------------------ frequency selectivity
def freq_correlation(delays_s, p_db, df_hz):
    """|R_H(df)| = |sum p_i exp(-j 2 pi df tau_i)| / sum p_i for a power-delay profile."""
    p = 10 ** (np.asarray(p_db, float) / 10)
    p = p / p.sum()
    t = np.asarray(delays_s, float)
    df = np.atleast_1d(np.asarray(df_hz, float))
    return np.abs(np.exp(-2j * np.pi * df[:, None] * t[None, :]) @ p)


def coherence_bandwidth(delays_s, p_db, level=0.5, fmax=None):
    """Smallest frequency separation (Hz) at which |R_H| first falls below ``level``."""
    p = 10 ** (np.asarray(p_db, float) / 10)
    p = p / p.sum()
    t = np.asarray(delays_s, float)
    m = np.sum(p * t)
    trms = np.sqrt(max(np.sum(p * t ** 2) - m ** 2, 1e-30))
    fmax = fmax or 5.0 / trms
    df = np.linspace(0, fmax, 4001)
    R = freq_correlation(t, p_db, df)
    i = np.flatnonzero(R < level)
    if len(i) == 0:
        return float("inf")
    i = i[0]
    return float(np.interp(level, [R[i], R[i - 1]], [df[i], df[i - 1]]))


# ------------------------------------------------------------------------ shadowing
def shadowing_field(ny, nx, dx, dcorr, sigma_db, rng=None):
    """Spatially correlated Gaussian shadowing map (dB) on an ny x nx grid with spacing dx (m),
    approximately exponential autocorrelation exp(-d / dcorr) (Gudmundson), std sigma_db.
    Generated by spectral shaping of white noise (the map is periodic at its edges)."""
    rng = np.random.default_rng(rng)
    w = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny, dx)
    kx = np.fft.fftfreq(nx, dx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    S = (1 + (2 * np.pi * dcorr) ** 2 * k2) ** (-1.5)       # 2-D spectrum of exp(-r/dcorr)
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * np.sqrt(S)))
    f = f - f.mean()
    return sigma_db * f / max(f.std(), 1e-12)
