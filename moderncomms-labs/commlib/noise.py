"""commlib.noise -- noise, noise figure and detection (Chapter 3).

Q and inverse Q, noise-equivalent bandwidth, Friis cascades with per-stage contributions,
lossy lines ahead of an LNA, the Y-factor method, and the Neyman-Pearson detection
probabilities of the coherent, envelope and energy detectors. Same formulas as
book/figscripts/ch03_figs.py. Import explicitly:  ``from commlib import noise as nz``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcinv

__all__ = ["K_B", "T0", "Q", "Qinv", "neb_hz", "butterworth_neb_ratio", "friis",
           "lossy_line_tsys", "yfactor_f", "pd_coherent", "pd_envelope", "pd_energy",
           "thresholds", "rice_pdf", "rice_cdf"]

K_B = 1.380649e-23
T0 = 290.0


def Q(x):
    """Gaussian tail probability Q(x) = P(N(0,1) > x)."""
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


def Qinv(p):
    return np.sqrt(2) * erfcinv(2 * np.asarray(p, float))


def neb_hz(f, H2):
    """One-sided noise-equivalent bandwidth from a sampled power response |H(f)|^2 on f >= 0."""
    f = np.asarray(f, float)
    H2 = np.asarray(H2, float)
    return float(np.trapezoid(H2, f) / H2[0])


def butterworth_neb_ratio(n):
    """B_N / f_3dB of an n-pole Butterworth low-pass: (pi/2n) / sin(pi/2n)."""
    n = np.asarray(n, float)
    return (np.pi / (2 * n)) / np.sin(np.pi / (2 * n))


def friis(stages):
    """stages: list of (name, gain_dB, NF_dB). Returns dict with the cascade noise figure
    'nf' (dB), 'te' (K), and per stage 'add' (contribution to F - 1, i.e. (F_k - 1)/G_before),
    'te_add' (that contribution in kelvin), 'cum_nf' (dB) and 'cum_g' (dB)."""
    F, G = 1.0, 1.0
    add, cum_nf, cum_g = [], [], []
    for _, g_db, nf_db in stages:
        a = (10 ** (nf_db / 10) - 1) / G
        F += a
        G *= 10 ** (g_db / 10)
        add.append(a)
        cum_nf.append(10 * np.log10(F))
        cum_g.append(10 * np.log10(G))
    add = np.array(add)
    return dict(nf=10 * np.log10(F), te=T0 * (F - 1), add=add, te_add=T0 * add,
                cum_nf=np.array(cum_nf), cum_g=np.array(cum_g))


def lossy_line_tsys(loss_db, t_ant, t_lna, t_phys=T0):
    """System temperature referred to the LNA input for an antenna (t_ant) seen through a
    passive loss at physical temperature t_phys, followed by an LNA of noise temperature t_lna."""
    L = 10 ** (np.asarray(loss_db, float) / 10)
    return t_ant / L + t_phys * (1 - 1 / L) + t_lna


def yfactor_f(Y, enr_db, second_nf_db=None, gain_db=None):
    """Noise factor (linear) from a Y-factor (linear power ratio) and the source ENR, with the
    optional second-stage correction F1 = F_sys - (F2 - 1)/G1."""
    F = 10 ** (enr_db / 10) / (np.asarray(Y, float) - 1)
    if second_nf_db is not None:
        F = F - (10 ** (second_nf_db / 10) - 1) / 10 ** (gain_db / 10)
    return F


# ----------------------------------------------------------------------------- detection
def pd_coherent(pfa, snr):
    """Known signal, known phase: P_D = Q(Q^-1(P_FA) - sqrt(2 E/N0)); snr = E/N0 (linear)."""
    return Q(Qinv(pfa) - np.sqrt(2 * np.asarray(snr, float)))


def pd_envelope(pfa, snr):
    """Known signal, unknown phase (square-law envelope): noncentral chi-square, 2 dof."""
    from scipy.stats import chi2, ncx2
    return ncx2.sf(chi2.isf(pfa, 2), 2, 2 * np.asarray(snr, float))


def pd_energy(pfa, snr, N):
    """Unknown signal, energy of N complex samples: noncentral chi-square, 2N dof."""
    from scipy.stats import chi2, ncx2
    return ncx2.sf(chi2.isf(pfa, 2 * N), 2 * N, 2 * np.asarray(snr, float))


def thresholds(pfa, N=1):
    """Thresholds for the normalised statistics used above: coherent (N(0,1) under H0),
    envelope (chi-square 2 dof, i.e. 2|r|^2 with unit-variance complex noise), energy (2N dof)."""
    from scipy.stats import chi2
    return dict(coherent=float(Qinv(pfa)), envelope=float(chi2.isf(pfa, 2)),
                energy=float(chi2.isf(pfa, 2 * N)))


def rice_pdf(r, K):
    """Envelope pdf for unit mean power with Rician factor K (K = 0: Rayleigh)."""
    from scipy.stats import rice
    s = np.sqrt(K / (K + 1))
    sig = np.sqrt(1 / (2 * (K + 1)))
    return rice.pdf(r, s / sig, scale=sig)


def rice_cdf(r, K):
    from scipy.stats import rice
    s = np.sqrt(K / (K + 1))
    sig = np.sqrt(1 / (2 * (K + 1)))
    return rice.cdf(r, s / sig, scale=sig)
