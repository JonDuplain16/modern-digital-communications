"""spreadkit: fast models for interactive spread-spectrum and GNSS experiments.

Companion of ``commlib.spread`` (Chapter 18), used by the interactive Lab 25. Where
``spread`` simulates signals sample by sample, these helpers use the standard analytic
models so that a slider can move them in real time:

    from commlib import spreadkit as sk
    sk.fh_ber(rho, ebj0_db, L)                 # FH/BFSK under partial-band jamming, L hops per bit
    sk.fh_worst_rho(ebj0_db, L)                # the jammer's best band fraction
    loop = sk.TrackingLoop(cn0=45)             # correlator-level DLL + Costas PLL (+ FLL) model
    out = loop.run(200)                        # 200 ms of 1 ms loop updates
    sk.sats_from_azel(user_ecef, az, el)       # GPS satellites at given azimuths / elevations
    sk.fix_cloud(sat, user, sigma, n)          # linearised least-squares position errors

Units: chips, Hz, metres, seconds; C/N0 in dB-Hz.
"""
from __future__ import annotations

from math import comb

import numpy as np

from .spread import CA_RATE, F_L1, GPS_A, C_LIGHT, ecef_to_lla, ecef_to_enu, loop_coeffs

__all__ = ["fh_hop_ber", "fh_ber", "fh_worst_rho", "TrackingLoop", "sats_from_azel",
           "fix_cloud", "ACCEL_HZ_PER_G"]

LAMBDA_L1 = C_LIGHT / F_L1
ACCEL_HZ_PER_G = 9.80665 / LAMBDA_L1          # Doppler rate (Hz/s) per g of line-of-sight acceleration


# ============================================================================ frequency hopping
def fh_hop_ber(rho, ebj0, ebn0=np.inf, L=1):
    """Error probability of ONE hop (chip) of non-coherent BFSK when each bit is split over L
    hops (energy Eb/L per hop), with a partial-band noise jammer of total density J0 spread
    over a fraction rho of the band (density J0/rho where it sits), plus thermal noise N0.
    ebj0, ebn0 linear. Returns (p_clear, p_jammed)."""
    rho = np.clip(np.asarray(rho, float), 1e-6, 1.0)
    inv_n0 = 0.0 if not np.isfinite(ebn0) else 1.0 / ebn0
    clear = 0.5 * np.exp(-1.0 / (2 * L * max(inv_n0, 1e-300))) if inv_n0 > 0 else 0.0
    jam = 0.5 * np.exp(-1.0 / (2 * L * (inv_n0 + 1.0 / (rho * ebj0))))
    return clear, jam


def fh_ber(rho, ebj0_db, L=1, ebn0_db=None):
    """Bit error rate of FH/BFSK with L hops per bit and hard-decision majority voting
    (ties broken at random) under partial-band jamming (Chapter 18). With L = 1 and no
    thermal noise this is (rho/2) exp(-rho Eb / 2J0)."""
    rho = np.atleast_1d(np.clip(np.asarray(rho, float), 1e-6, 1.0))
    ebj0 = 10 ** (ebj0_db / 10)
    ebn0 = np.inf if ebn0_db is None else 10 ** (ebn0_db / 10)
    pc, pj = fh_hop_ber(rho, ebj0, ebn0, L)
    p = rho * pj + (1 - rho) * pc                 # independent hops: each jammed w.p. rho
    out = np.zeros_like(rho)
    for k in range(L + 1):                        # k chip errors out of L
        w = comb(L, k) * p ** k * (1 - p) ** (L - k)
        if 2 * k > L:
            out += w
        elif 2 * k == L:
            out += 0.5 * w
    return out if out.size > 1 else float(out[0])


def fh_worst_rho(ebj0_db, L=1, ebn0_db=None, n=400):
    """The band fraction rho that maximises the BER (the smart jammer's choice), and that BER."""
    r = np.logspace(-4, 0, n)
    b = fh_ber(r, ebj0_db, L, ebn0_db)
    i = int(np.argmax(b))
    return float(r[i]), float(b[i])


# ============================================================================ tracking loops
class TrackingLoop:
    """Correlator-level model of a GPS C/A tracking channel: carrier-aided first-order DLL with
    a normalised early-minus-late envelope discriminator, and a Costas PLL (optionally FLL
    assisted) with the same loop-filter equations as ``commlib.spread.track``.

    Instead of synthesising 2 million samples per second, each 1 ms integrate-and-dump is
    computed from the triangle correlation function: P = A d R(e_tau) sinc(e_f T) e^{j e_phi} + n,
    with E and L at e_tau +- spacing/2 and correctly correlated noise. A^2 / (2 sigma^2) = C/N0 T.
    The state persists between calls to ``run``, so the loops keep running while you turn knobs.
    """

    def __init__(self, cn0=45.0, dll_bn=2.0, pll_bn=15.0, fll_bn=0.0, spacing=1.0,
                 code_err0=0.3, freq_err0=40.0, accel_g=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.cn0, self.dll_bn, self.pll_bn, self.fll_bn = cn0, dll_bn, pll_bn, fll_bn
        self.spacing, self.accel_g = spacing, accel_g
        self.T = 1e-3
        self.t = 0.0                      # seconds since start
        self.tau_err = code_err0          # true code delay minus estimate (chips)
        self.f_true = 1500.0              # true Doppler (Hz)
        self.ph_true = 0.0                # true carrier phase (rad)
        self.carr_f = self.f_true - freq_err0
        self.carr_int = self.carr_f
        self.ph_est = 0.0
        self.prev_p = None
        self.bit = 1.0
        self.ms = 0

    def run(self, n_ms):
        """Advance n_ms milliseconds. Returns a dict of per-ms arrays: t (s), code_err (true,
        chips), disc (DLL output, chips), phase_err (true, rad, wrapped to +-pi), f_err (Hz),
        IP, QP, bit (transmitted data bit), doppler (true, Hz), nco (Hz)."""
        T, d = self.T, self.spacing
        A = np.sqrt(2 * 10 ** (self.cn0 / 10) * T)
        wd = loop_coeffs(self.dll_bn)
        wp = loop_coeffs(self.pll_bn)
        wf = loop_coeffs(self.fll_bn) if self.fll_bn > 0 else 0.0
        K_dll = 4 * self.dll_bn * T                     # first-order loop gain for B_n
        C = np.array([[1, 1 - d / 2, max(1 - d, 0)], [1 - d / 2, 1, 1 - d / 2],
                      [max(1 - d, 0), 1 - d / 2, 1]])
        Lc = np.linalg.cholesky(C + 1e-9 * np.eye(3))
        acc = self.accel_g * 9.80665 / (C_LIGHT / F_L1)  # Hz/s
        keys = ("t", "code_err", "disc", "phase_err", "f_err", "IP", "QP", "bit", "doppler", "nco")
        out = {k: np.empty(n_ms) for k in keys}
        chip_per_hz = CA_RATE / F_L1
        for k in range(n_ms):
            if self.ms % 20 == 0:
                self.bit = 1.0 if self.rng.random() < 0.5 else -1.0
            f0 = self.f_true
            f1 = f0 + acc * T
            # true phase over the interval (quadratic), estimate linear
            ph_mid_true = self.ph_true + 2 * np.pi * (f0 * T / 2 + acc * T * T / 8)
            ph_mid_est = self.ph_est + 2 * np.pi * self.carr_f * T / 2
            e_ph = ph_mid_true - ph_mid_est
            e_f = (f0 + f1) / 2 - self.carr_f
            att = np.sinc(e_f * T)
            # code error drifts with the residual (unaided) code Doppler over the interval
            tau_mid = self.tau_err + e_f * chip_per_hz * T / 2
            R = np.maximum(0.0, 1 - np.abs(np.array([tau_mid + d / 2, tau_mid, tau_mid - d / 2])))
            noise = Lc @ (self.rng.standard_normal(3) + 1j * self.rng.standard_normal(3))
            E, P, L = A * self.bit * att * R * np.exp(1j * e_ph) + noise
            # DLL (carrier aided): unit-slope normalised early-minus-late envelope
            aE, aL = abs(E), abs(L)
            disc = (1 - d / 2) * (aL - aE) / (aE + aL + 1e-30)
            # Costas PLL (cycles) + optional FLL (Hz), as in commlib.spread.track
            pe = np.arctan(P.imag / P.real) / (2 * np.pi) if P.real != 0 else 0.0
            fe = 0.0
            if wf and self.prev_p is not None:
                cross = self.prev_p.real * P.imag - self.prev_p.imag * P.real
                dot = self.prev_p.real * P.real + self.prev_p.imag * P.imag
                fe = np.arctan(cross / dot) / (2 * np.pi * T) if dot != 0 else 0.0
            self.prev_p = P
            # advance truth and estimates to the end of the interval
            self.ph_true += 2 * np.pi * (f0 * T + acc * T * T / 2)
            self.ph_est += 2 * np.pi * self.carr_f * T
            self.tau_err += e_f * chip_per_hz * T - K_dll * disc
            self.carr_int += wp ** 2 * T * pe + (wf * T * fe if wf else 0.0)
            self.carr_f = self.carr_int + 2 * 0.7071 * wp * pe
            self.f_true = f1
            for key, val in (("t", self.t), ("code_err", self.tau_err), ("disc", disc),
                             ("phase_err", (e_ph + np.pi) % (2 * np.pi) - np.pi),
                             ("f_err", e_f), ("IP", P.real), ("QP", P.imag), ("bit", self.bit),
                             ("doppler", self.f_true), ("nco", self.carr_f)):
                out[key][k] = val
            self.t += T
            self.ms += 1
        return out


# ============================================================================ navigation
def sats_from_azel(user, az_deg, el_deg, radius=GPS_A):
    """ECEF positions (K, 3) of satellites on a sphere of ``radius`` seen from ECEF ``user``
    at the given azimuths and elevations (degrees)."""
    user = np.asarray(user, float)
    lat, lon, _ = ecef_to_lla(user)
    la, lo = np.radians(lat), np.radians(lon)
    Rm = np.array([[-np.sin(lo), np.cos(lo), 0],
                   [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                   [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    out = []
    for a_, e_ in zip(np.radians(np.atleast_1d(az_deg)), np.radians(np.atleast_1d(el_deg))):
        dv = Rm.T @ np.array([np.cos(e_) * np.sin(a_), np.cos(e_) * np.cos(a_), np.sin(e_)])
        b = 2 * user @ dv
        c = user @ user - radius ** 2
        s = (-b + np.sqrt(b * b - 4 * c)) / 2
        out.append(user + s * dv)
    return np.array(out)


def fix_cloud(sat, user, sigma_m, n=400, rng=None):
    """Position errors of n least-squares fixes with i.i.d. pseudorange errors of sigma_m, by the
    linearised solution dx = (G'G)^-1 G' e (exact to first order; one matrix product for all n).
    Returns (enu errors (n, 3) in metres, clock errors (n,) in metres)."""
    rng = np.random.default_rng(rng)
    sat = np.asarray(sat, float)
    user = np.asarray(user, float)
    lat, lon, _ = ecef_to_lla(user)
    d = sat - user[None, :]
    u = d / np.linalg.norm(d, axis=1)[:, None]
    G = np.hstack([-u, np.ones((len(sat), 1))])
    Hm = np.linalg.solve(G.T @ G, G.T)                     # (4, K)
    e = sigma_m * rng.standard_normal((len(sat), n))
    dx = Hm @ e                                            # (4, n)
    enu = ecef_to_enu(dx[:3], lat, lon).T
    return enu, dx[3]
