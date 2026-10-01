"""propagation: path-loss models, diffraction, atmospheric losses and coverage statistics.

Companion of Chapter 11 (*The Wireless Channel*): clean copies of the calculators in
book/figscripts/ch11_figs.py, so the lab reproduces the chapter's tables and worked examples.
Import explicitly:

    from commlib import propagation as pr
    pr.cost231(1800, 30, 1.5, 0.7)                # COST-231 Hata path loss (dB)
    pr.pl_38901("UMa", d2d_m, 3.5, 25)            # 3GPP TR 38.901 (LOS, NLOS) path loss (dB)
    pr.knife_edge_loss(pr.fresnel_v(h, d1, d2, lam))
    pr.edge_margin_for_area(0.95, n=3.5, sigma=8)  # shadowing margin (dB) for 95% area coverage

Units: distances in metres unless the argument name says km; frequencies in Hz unless it
says MHz or GHz; losses are positive dB.
"""
from __future__ import annotations

import numpy as np
from scipy.special import fresnel, erf
from scipy.stats import norm
from scipy.optimize import brentq

__all__ = ["C0", "wavelength", "fspl_db", "dish_gain_dbi", "beamwidth_deg", "fresnel_radius",
           "earth_bulge", "fresnel_v", "knife_edge_loss", "knife_edge_loss_p526", "two_ray_gain_db",
           "hata", "cost231", "pl_38901", "SIGMA_SF_38901", "o2i_loss_db", "gas_atten", "rain_k_alpha",
           "rain_specific_atten", "thermal_noise_dbm", "sensitivity_dbm", "edge_coverage",
           "area_coverage", "edge_margin_for_area", "radius_for_mapl", "rms_delay_spread"]

C0 = 299_792_458.0


def wavelength(f_hz):
    return C0 / np.asarray(f_hz, float)


def fspl_db(d_m, f_hz):
    """Free-space path loss 20 log10(4 pi d / lambda)."""
    return 20 * np.log10(4 * np.pi * np.asarray(d_m, float) / wavelength(f_hz))


def dish_gain_dbi(diam_m, f_hz, eff=0.65):
    """Gain of a circular aperture: eff (pi D / lambda)^2 in dBi."""
    return 10 * np.log10(eff * (np.pi * np.asarray(diam_m, float) / wavelength(f_hz)) ** 2)


def beamwidth_deg(diam_m, f_hz, k=70.0):
    """Approximate half-power beamwidth k lambda / D (degrees)."""
    return k * wavelength(f_hz) / np.asarray(diam_m, float)


# ============================================================================ diffraction
def fresnel_radius(d1, d2, f_hz, n=1):
    """Radius of the n-th Fresnel zone at distances d1, d2 from the ends (m)."""
    lam = wavelength(f_hz)
    return np.sqrt(n * lam * d1 * d2 / (d1 + d2))


def earth_bulge(d1, d2, k=4 / 3, re=6.371e6):
    """Height of the Earth's bulge above the chord at d1, d2 for effective-radius factor k (m)."""
    return d1 * d2 / (2 * k * re)


def fresnel_v(h, d1, d2, lam):
    """Fresnel-Kirchhoff parameter v = h sqrt(2 (d1 + d2) / (lam d1 d2)); h > 0 = obstruction
    above the line of sight."""
    return h * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))


def knife_edge_loss(v):
    """Single knife-edge diffraction loss (dB, positive) from the Fresnel integrals (exact)."""
    S, C = fresnel(np.asarray(v, float))
    F = 0.5 * ((0.5 - C) ** 2 + (0.5 - S) ** 2)
    return -10 * np.log10(F)


def knife_edge_loss_p526(v):
    """ITU-R P.526 approximation, valid for v > -0.78 (0 dB below)."""
    v = np.asarray(v, float)
    return np.where(v > -0.78, 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1), 0.0)


def two_ray_gain_db(d, f_hz, ht, hr, eps_r=15.0, pol="h"):
    """Path gain (dB, negative) of the two-ray ground-reflection model with a Fresnel
    reflection coefficient for a dielectric ground (pol 'h' or 'v')."""
    d = np.asarray(d, float)
    lam = wavelength(f_hz)
    dlos = np.sqrt(d ** 2 + (ht - hr) ** 2)
    dref = np.sqrt(d ** 2 + (ht + hr) ** 2)
    sin_t = (ht + hr) / dref; cos_t = d / dref
    z = np.sqrt(eps_r - cos_t ** 2)
    G = (sin_t - z) / (sin_t + z) if pol == "h" else (eps_r * sin_t - z) / (eps_r * sin_t + z)
    k = 2 * np.pi / lam
    E = np.exp(-1j * k * dlos) / dlos + G * np.exp(-1j * k * dref) / dref
    return 10 * np.log10((lam / (4 * np.pi)) ** 2) + 20 * np.log10(np.abs(E))


# ============================================================================ empirical models
def hata(fmhz, hb, hm, dkm, env="urban"):
    """Okumura-Hata (150-1500 MHz, 1-20 km): 'urban' (small/medium city), 'suburban', 'open'."""
    lf = np.log10(fmhz)
    a = (1.1 * lf - 0.7) * hm - (1.56 * lf - 0.8)
    L = 69.55 + 26.16 * lf - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(dkm)
    if env == "suburban":
        L = L - 2 * np.log10(fmhz / 28) ** 2 - 5.4
    if env == "open":
        L = L - 4.78 * lf ** 2 + 18.33 * lf - 40.94
    return L


def cost231(fmhz, hb, hm, dkm, Cm=3):
    """COST-231 Hata extension (1500-2000 MHz). Cm = 0 dB for medium cities and suburbs (the
    chapter's 'Cell radius from Hata' worked example), 3 dB for metropolitan centres (the default,
    as in the chapter's path-loss figure)."""
    lf = np.log10(fmhz)
    a = (1.1 * lf - 0.7) * hm - (1.56 * lf - 0.8)
    return 46.3 + 33.9 * lf - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(dkm) + Cm


SIGMA_SF_38901 = {"UMa": (4.0, 6.0), "UMi": (4.0, 7.82), "InH": (3.0, 8.03)}   # (LOS, NLOS) dB


def pl_38901(scn, d2, fc_ghz, hbs, hut=1.5):
    """3GPP TR 38.901 Table 7.4.1-1 path loss. scn 'UMa', 'UMi' or 'InH'; d2 = 2-D distance (m);
    returns (PL_LOS, PL_NLOS) in dB (NLOS = max(LOS, NLOS formula)). The breakpoint uses
    effective heights reduced by 1 m."""
    d2 = np.asarray(d2, float)
    d3 = np.sqrt(d2 ** 2 + (hbs - hut) ** 2)
    dbp = 4 * (hbs - 1) * (hut - 1) * fc_ghz * 1e9 / C0
    lf = np.log10(fc_ghz)
    if scn == "UMa":
        l1 = 28.0 + 22 * np.log10(d3) + 20 * lf
        l2 = 28.0 + 40 * np.log10(d3) + 20 * lf - 9 * np.log10(dbp ** 2 + (hbs - hut) ** 2)
        los = np.where(d2 <= dbp, l1, l2)
        nl = 13.54 + 39.08 * np.log10(d3) + 20 * lf - 0.6 * (hut - 1.5)
    elif scn == "UMi":
        l1 = 32.4 + 21 * np.log10(d3) + 20 * lf
        l2 = 32.4 + 40 * np.log10(d3) + 20 * lf - 9.5 * np.log10(dbp ** 2 + (hbs - hut) ** 2)
        los = np.where(d2 <= dbp, l1, l2)
        nl = 22.4 + 35.3 * np.log10(d3) + 21.3 * lf - 0.3 * (hut - 1.5)
    elif scn == "InH":
        los = 32.4 + 17.3 * np.log10(d3) + 20 * lf
        nl = 17.30 + 38.3 * np.log10(d3) + 24.9 * lf
    else:
        raise ValueError(scn)
    return los, np.maximum(los, nl)


def o2i_loss_db(f_ghz, building="low", d_in=0.0):
    """Outdoor-to-indoor penetration loss of TR 38.901 (Tables 7.4.3-1/2), mean value:
    5 - 10 log10(sum p_i 10^(-L_i/10)) + 0.5 d_in. 'low' = 30% glass / 70% concrete,
    'high' = 70% IRR glass / 30% concrete."""
    f = np.asarray(f_ghz, float)
    glass, irr, concrete = 2 + 0.2 * f, 23 + 0.3 * f, 5 + 4 * f
    if building == "low":
        mix = 0.3 * 10 ** (-glass / 10) + 0.7 * 10 ** (-concrete / 10)
    else:
        mix = 0.7 * 10 ** (-irr / 10) + 0.3 * 10 ** (-concrete / 10)
    return 5 - 10 * np.log10(mix) + 0.5 * d_in


# ============================================================================ atmosphere
def gas_atten(f_ghz, rho=7.5):
    """Approximate sea-level specific attenuation (dB/km) of (oxygen, water vapour), from the
    simplified formulas of earlier editions of ITU-R P.676 (roughly 1-350 GHz); the 57-63 GHz
    oxygen complex is bridged by a smooth fit peaking near 15 dB/km."""
    f = np.atleast_1d(np.asarray(f_ghz, float))
    go = np.zeros_like(f)
    lo = f < 57; hi = f > 63; mid = ~lo & ~hi
    fl = f[lo]
    go[lo] = (7.19e-3 + 6.09 / (fl ** 2 + 0.227) + 4.81 / ((fl - 57) ** 2 + 1.50)) * fl ** 2 * 1e-3
    fh = f[hi]
    go[hi] = (3.79e-7 * fh + 0.265 / ((fh - 63) ** 2 + 1.59) + 0.028 / ((fh - 118) ** 2 + 1.47)) * (fh + 198) ** 2 * 1e-3
    x57 = (7.19e-3 + 6.09 / (57 ** 2 + 0.227) + 4.81 / 1.5) * 57 ** 2 * 1e-3
    x63 = (3.79e-7 * 63 + 0.265 / 1.59 + 0.028 / ((63 - 118) ** 2 + 1.47)) * 261 ** 2 * 1e-3
    A = np.array([[57 ** 2, 57, 1], [60 ** 2, 60, 1], [63 ** 2, 63, 1]])
    cc = np.linalg.solve(A, [x57, 15.0, x63])
    go[mid] = cc[0] * f[mid] ** 2 + cc[1] * f[mid] + cc[2]
    gw = (0.050 + 0.0021 * rho + 3.6 / ((f - 22.2) ** 2 + 8.5) + 10.6 / ((f - 183.3) ** 2 + 9.0)
          + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f ** 2 * rho * 1e-4
    return go, gw


def rain_k_alpha(f_ghz):
    """ITU-R P.838-3 coefficients (k, alpha), horizontal polarisation."""
    lf = np.log10(np.asarray(f_ghz, float))
    a = [-5.33980, -0.35351, -0.23789, -0.94158]; b = [-0.10008, 1.26970, 0.86036, 0.64552]
    c = [1.13098, 0.45400, 0.15354, 0.16817]
    k = 10 ** (sum(ai * np.exp(-((lf - bi) / ci) ** 2) for ai, bi, ci in zip(a, b, c)) - 0.18961 * lf + 0.71147)
    a = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721]; b = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980]
    c = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
    al = sum(ai * np.exp(-((lf - bi) / ci) ** 2) for ai, bi, ci in zip(a, b, c)) + 0.67849 * lf - 1.95537
    return k, al


def rain_specific_atten(f_ghz, rain_mmh):
    """gamma_R = k R^alpha (dB/km), horizontal polarisation."""
    k, al = rain_k_alpha(f_ghz)
    return k * np.asarray(rain_mmh, float) ** al


# ============================================================================ budgets & coverage
def thermal_noise_dbm(bw_hz, nf_db=0.0, t0=290.0):
    """kTB + NF in dBm (-174 dBm/Hz at 290 K)."""
    return 10 * np.log10(1.380649e-23 * t0 * 1e3) + 10 * np.log10(bw_hz) + nf_db


def sensitivity_dbm(bw_hz, nf_db, snr_req_db):
    return thermal_noise_dbm(bw_hz, nf_db) + snr_req_db


def edge_coverage(margin_db, sigma_db):
    """Probability that log-normal shadowing of std sigma stays within the margin."""
    return norm.cdf(np.asarray(margin_db, float) / sigma_db)


def area_coverage(p_edge, n, sigma_db):
    """Jakes-Reudink fraction of a circular cell's area covered, eq. (11.areacov), given the
    edge coverage probability, path-loss exponent n and shadowing std sigma (dB)."""
    a = -norm.ppf(np.asarray(p_edge, float)) / np.sqrt(2)
    b = 10 * n * np.log10(np.e) / (sigma_db * np.sqrt(2))
    return 0.5 * (1 - erf(a) + np.exp((1 - 2 * a * b) / b ** 2) * (1 - erf((1 - a * b) / b)))


def edge_margin_for_area(f_area, n, sigma_db):
    """Shadowing margin (dB) that gives area coverage f_area. Returns (margin, edge probability)."""
    pe = brentq(lambda p: area_coverage(p, n, sigma_db) - f_area, 0.01, 0.99999)
    return float(norm.ppf(pe) * sigma_db), pe


def radius_for_mapl(mapl_db, model, lo=0.01, hi=100.0):
    """Distance (same unit as the model's argument) at which model(d) equals the MAPL."""
    return brentq(lambda d: model(d) - mapl_db, lo, hi)


def rms_delay_spread(delays, p_db):
    """(mean delay, rms delay spread) of a power-delay profile (same units as delays)."""
    p = 10 ** (np.asarray(p_db, float) / 10); p /= p.sum()
    t = np.asarray(delays, float)
    m = np.sum(p * t)
    return m, np.sqrt(np.sum(p * t ** 2) - m ** 2)
