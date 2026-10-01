"""commlib.rf -- RF transceiver models: cascades, nonlinearity, phase noise, PAs and DPD.

Companion module for Chapter 7 (The Radio Transceiver and the SDR) and Lab 36. The PA,
memory-polynomial/ILA DPD, cascade, Leeson/PLL and efficiency models are the same ones used by
book/figscripts/ch07_figs.py, collected here so the lab and the book agree.
Import explicitly:  ``from commlib import rf``.

Units: powers in dBm (or mW where stated); amplitudes in sqrt(mW) so that a real tone of
amplitude A carries A^2/2 mW; frequencies in hertz.
"""
from __future__ import annotations

import numpy as np

__all__ = ["K_DBM", "db10", "cascade", "poly_stage", "two_tone", "leeson_dbc", "pll_components",
           "rms_phase_deg", "phase_noise_from_mask", "irr_exact", "pa_static", "pa_memory", "mp_basis",
           "ila_dpd", "eff_class_a", "eff_class_b", "eff_doherty", "eff_et", "avg_efficiency"]

K_DBM = -174.0          # kT0 in dBm/Hz


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


# ----------------------------------------------------------------------------- line-up
def cascade(stages):
    """stages: list of (name, gain_dB, NF_dB, IIP3_dBm or None).
    Returns a list of (name, cumulative gain dB, cumulative NF dB, cumulative IIP3 dBm) using the
    Friis formula and the coherent (worst-case) IIP3 cascade 1/IIP3 = sum G_before / IIP3_k."""
    G, F, inv = 1.0, 1.0, 0.0
    out = []
    for i, (nm, g, nf, iip3) in enumerate(stages):
        f = 10 ** (nf / 10)
        F = F + (f - 1) / G if i else f
        if iip3 is not None:
            inv += G / 10 ** (iip3 / 10)
        G *= 10 ** (g / 10)
        out.append((nm, db10(G), db10(F), -db10(inv) if inv > 0 else np.inf))
    return out


def poly_stage(gain_db, iip3_dbm=None):
    """Memoryless cubic stage y = a1 x + a3 x^3 with the given gain and input IP3
    (amplitudes in sqrt(mW): a tone of amplitude A has power A^2/2 mW)."""
    a1 = 10 ** (gain_db / 20)
    if iip3_dbm is None:
        return lambda x: a1 * x
    A2 = 2 * 10 ** (iip3_dbm / 10)                     # IIP3 amplitude squared
    a3 = -4 / 3 * a1 / A2
    return lambda x: a1 * x + a3 * x ** 3


def two_tone(system, pin_dbm, f1=0.1, df=0.01, n=1 << 14):
    """Drive ``system`` (callable on real samples) with two tones of pin_dbm each.
    Returns (P_tone_out dBm, P_IM3_out dBm, IIP3 estimate dBm = Pin + delta/2)."""
    t = np.arange(n)
    f1 = round(f1 * n) / n; f2 = f1 + round(df * n) / n
    A = np.sqrt(2 * 10 ** (pin_dbm / 10))
    x = A * (np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t))
    Y = np.fft.rfft(system(x)) / n * 2
    k1, k3 = int(round(f1 * n)), int(round((2 * f1 - f2) * n))
    pt, pi = db10(np.abs(Y[k1]) ** 2 / 2), db10(np.abs(Y[k3]) ** 2 / 2)
    return pt, pi, pin_dbm + (pt - pi) / 2


# ----------------------------------------------------------------------------- phase noise
def leeson_dbc(f, f0, Q, F_db, Ps_dbm, fc):
    """Leeson's phase-noise model L(f) in dBc/Hz."""
    return (K_DBM + F_db - 3 - Ps_dbm) + db10((1 + (f0 / (2 * Q * np.asarray(f, float))) ** 2) * (1 + fc / np.asarray(f, float)))


def pll_components(f, N=60, fbw=400e3):
    """Reference/PFD noise x N inside a 2nd-order loop, VCO noise outside (Chapter 7's 2.4 GHz PLL).
    Returns (ref, pfd, vco, in-band, out-of-band, total) in dBc/Hz."""
    f = np.asarray(f, float)
    ref = leeson_dbc(f, 40e6, 6e4, 5, 7, 2e3) + 20 * np.log10(N)
    pfd = np.full_like(f, -221 + 10 * np.log10(40e6) + 20 * np.log10(N))
    vco = leeson_dbc(f, 2.4e9, 8, 15, 0, 1e5)
    wn, z = 2 * np.pi * fbw / 2.06, 0.707
    s = 2j * np.pi * f
    H = (2 * z * wn * s + wn ** 2) / (s ** 2 + 2 * z * wn * s + wn ** 2)
    inb = db10(10 ** (ref / 10) + 10 ** (pfd / 10)) + db10(np.abs(H) ** 2)
    outb = vco + db10(np.abs(1 - H) ** 2)
    return ref, pfd, vco, inb, outb, db10(10 ** (inb / 10) + 10 ** (outb / 10))


def rms_phase_deg(f, L_dbc):
    """RMS phase error (degrees) from a single-sideband L(f) over the offsets f (sqrt(2 int L))."""
    return np.rad2deg(np.sqrt(2 * np.trapezoid(10 ** (np.asarray(L_dbc) / 10), f)))


def phase_noise_from_mask(n, fs, mask_f, mask_dbc, rng=None):
    """Synthesise n samples of phase phi(t) (rad) at rate fs whose two-sided PSD equals L(|f|)
    (so that the SSB phase noise of exp(j phi) matches the mask for small angles). The mask is
    interpolated log-linearly in frequency between the points (mask_f Hz, mask_dbc dBc/Hz) and held
    flat outside them."""
    rng = np.random.default_rng(rng)
    f = np.abs(np.fft.fftfreq(n, 1 / fs))
    f[0] = f[1]
    L = np.interp(np.log10(f), np.log10(mask_f), mask_dbc)
    W = np.fft.fft(rng.standard_normal(n))                # white, E|W|^2 = n
    return np.real(np.fft.ifft(W * np.sqrt(10 ** (L / 10) * fs)))


def irr_exact(g_db, phi_deg):
    """Image-rejection ratio (dB) for gain imbalance g_db and phase error phi_deg."""
    g = 10 ** (g_db / 20); ph = np.deg2rad(phi_deg)
    return db10((1 + 2 * g * np.cos(ph) + g ** 2) / (1 - 2 * g * np.cos(ph) + g ** 2))


# ----------------------------------------------------------------------------- PA and DPD
def pa_static(x, sat=1.0, p=2.0, ampm_deg=12.0):
    """Rapp AM/AM with Saleh-type AM/PM (phase in degrees reached at saturation drive)."""
    a = np.abs(x)
    g = 1 / (1 + (a / sat) ** (2 * p)) ** (1 / (2 * p))
    ph = np.deg2rad(ampm_deg) * 2 * (a / sat) ** 2 / (1 + (a / sat) ** 2)
    return x * g * np.exp(1j * ph)


_H_IN = np.array([1.0, 0.5 + 0.2j, 0.2]); _H_IN = _H_IN / _H_IN.sum()
_H_OUT = np.array([1.0, 0.25j]); _H_OUT = _H_OUT / _H_OUT.sum()


def pa_memory(x):
    """A PA with memory: filter, static nonlinearity, filter (Wiener-Hammerstein; Chapter 7)."""
    u = np.convolve(x, _H_IN)[:len(x)]
    return np.convolve(pa_static(u), _H_OUT)[:len(x)]


def mp_basis(u, K=7, M=3):
    """Memory-polynomial basis u[n-m] |u[n-m]|^(k-1), k = 1..K, m = 0..M."""
    cols = []
    for m in range(M + 1):
        um = np.concatenate([np.zeros(m, complex), u[:len(u) - m]])
        for k in range(1, K + 1):
            cols.append(um * np.abs(um) ** (k - 1))
    return np.stack(cols, axis=1)


def ila_dpd(x, pa, K=7, M=3, iters=5, G=1.0, lam=1e-6):
    """Indirect-learning DPD: fit a post-inverse y/G -> z (regularised least squares on a
    column-normalised memory-polynomial basis) and copy it in front of the PA.
    Returns (predistorted input z, PA output pa(z))."""
    z = x.copy()
    for _ in range(iters):
        y = pa(z)
        Phi = mp_basis(y / G, K, M)
        sc = np.sqrt(np.mean(np.abs(Phi) ** 2, axis=0))
        A = Phi[200:] / sc
        c = np.linalg.solve(A.conj().T @ A + lam * len(A) * np.eye(A.shape[1]), A.conj().T @ z[200:]) / sc
        z = mp_basis(x, K, M) @ c
    return z, pa(z)


def eff_class_a(v):
    """Ideal class-A drain efficiency at normalised output amplitude v (peak = 1)."""
    return 0.5 * np.asarray(v) ** 2


def eff_class_b(v):
    return np.pi / 4 * np.asarray(v)


def eff_doherty(v, alpha=0.5):
    """Ideal Doherty with class-B devices; peaking device turns on at v = alpha (0.5 symmetric)."""
    v = np.asarray(v, float)
    return np.where(v <= alpha, np.pi / 4 * v / alpha, np.pi / 4 * v ** 2 / (v * (1 + alpha) - alpha))


def eff_et(v, vmin=0.25, eta_pa=0.7, eta_sup=0.85):
    """Envelope tracking: a 70%-efficient PA on an 85%-efficient supply, tracking down to vmin."""
    v = np.asarray(v, float)
    return eta_pa * eta_sup * np.where(v >= vmin, 1.0, v / vmin)


def avg_efficiency(eff, v):
    """Average efficiency E[Pout]/E[Pdc] for normalised amplitude samples v (peak = 1)."""
    pout = np.asarray(v) ** 2
    return pout.mean() / (pout / np.maximum(eff(v), 1e-9)).mean()
