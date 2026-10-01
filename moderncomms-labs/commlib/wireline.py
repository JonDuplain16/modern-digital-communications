"""commlib.wireline -- copper-loop, DSL, fibre and coherent-optics models.

Used by Chapter 24 ("Wireline and Optical Communications") figures and by the
wireline/optical lab. Import explicitly:  ``from commlib import wireline as wl``.

Contents
--------
Copper
  * ``rlgc_twisted_pair``  -- the parametric RLGC model of twisted pair widely used in
    DSL work (the ANSI T1.417 / Starr-Cioffi-Silverman form), 24 and 26 AWG.
  * ``line_abcd``, ``series_abcd``, ``shunt_abcd``, ``cascade``, ``insertion_gain``
    -- two-port (ABCD) analysis of loops with bridged taps and loading coils.
  * ``next_coupling``, ``fext_coupling`` -- the 1 %-worst-case crosstalk models.
  * ``dmt_rate`` -- gap-approximation bit loading for DMT (ADSL/VDSL2/G.fast).
Fibre
  * ``fibre_attenuation_db_km`` -- Rayleigh + IR + UV + OH-peak loss model.
  * ``dispersion_ps_nm_km`` -- Sellmeier-like D(lambda) for G.652-type fibre.
  * ``cd_transfer`` / ``cd_compensate`` -- the chromatic-dispersion all-pass filter.
  * ``imdd_cd_response`` -- the power-fading response of intensity modulation + CD.
  * ``osnr_db``, ``gn_eta``, ``gn_snr`` -- amplified-link OSNR and the GN model of
    nonlinear interference.
  * ``cma_butterfly``, ``vv_carrier_recovery`` -- minimal coherent-receiver DSP.

SI units throughout unless a name says otherwise (``_db``, ``_km``, ``_nm``...).
These models are for teaching and first-cut design; real planning uses measured
cable data and the full standards.
"""
from __future__ import annotations

import numpy as np

C = 299_792_458.0          # speed of light in vacuum, m/s
H_PLANCK = 6.62607015e-34  # J s
Q_E = 1.602176634e-19      # C
K_B = 1.380649e-23         # J/K

# ---------------------------------------------------------------- copper: RLGC
# Parameters per km (ohm, H, F, S) for the parametric twisted-pair model.
_RLGC = {
    26: dict(roc=286.17578, ac=0.14769620, l0=675.36888e-6, linf=488.95186e-6,
             b=0.92930728, fm=806.33863e3, cinf=49e-9, g0=43e-9, ge=0.70),
    24: dict(roc=174.55888, ac=0.053073481, l0=617.29539e-6, linf=478.97099e-6,
             b=1.1529766, fm=553.760e3, cinf=50e-9, g0=234.87476e-15, ge=1.38),
}


def rlgc_twisted_pair(f, awg=26):
    """Return (R, L, G, C) per metre at frequency f (Hz) for 24 or 26 AWG PE-insulated pair."""
    p = _RLGC[awg]
    f = np.asarray(f, float)
    R = (p["roc"] ** 4 + p["ac"] * f ** 2) ** 0.25          # skin effect: ~sqrt(f) at HF
    x = (f / p["fm"]) ** p["b"]
    L = (p["l0"] + p["linf"] * x) / (1 + x)
    G = p["g0"] * f ** p["ge"]
    Cc = p["cinf"] * np.ones_like(f)
    return R / 1e3, L / 1e3, G / 1e3, Cc / 1e3


def propagation(f, awg=26):
    """Return (gamma, Z0): propagation constant (1/m) and characteristic impedance (ohm)."""
    R, L, G, Cc = rlgc_twisted_pair(f, awg)
    w = 2 * np.pi * np.asarray(f, float)
    Z = R + 1j * w * L
    Y = G + 1j * w * Cc
    return np.sqrt(Z * Y), np.sqrt(Z / Y)


def attenuation_db_per_km(f, awg=26):
    gamma, _ = propagation(f, awg)
    return 8.686 * gamma.real * 1e3


def line_abcd(f, length_m, awg=26):
    """ABCD matrix (shape (..., 2, 2)) of a uniform line section."""
    gamma, Z0 = propagation(f, awg)
    gl = gamma * length_m
    ch, sh = np.cosh(gl), np.sinh(gl)
    M = np.empty(np.shape(gl) + (2, 2), complex)
    M[..., 0, 0] = ch; M[..., 0, 1] = Z0 * sh
    M[..., 1, 0] = sh / Z0; M[..., 1, 1] = ch
    return M


def series_abcd(Z):
    Z = np.asarray(Z, complex)
    M = np.zeros(Z.shape + (2, 2), complex)
    M[..., 0, 0] = 1; M[..., 1, 1] = 1; M[..., 0, 1] = Z
    return M


def shunt_abcd(Y):
    Y = np.asarray(Y, complex)
    M = np.zeros(Y.shape + (2, 2), complex)
    M[..., 0, 0] = 1; M[..., 1, 1] = 1; M[..., 1, 0] = Y
    return M


def bridged_tap_abcd(f, length_m, awg=26):
    """An open-circuited bridged tap appears as a shunt admittance tanh(gamma l)/Z0."""
    gamma, Z0 = propagation(f, awg)
    return shunt_abcd(np.tanh(gamma * length_m) / Z0)


def cascade(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out


def insertion_gain(M, Zs=100.0, Zl=100.0):
    """Voltage transfer V_load / (V_source/2) for source and load impedances (1 = no loss
    when Zs = Zl and the two-port is a through connection)."""
    A, B, Cm, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    return (Zs + Zl) / (A * Zl + B + Cm * Zs * Zl + D * Zs)


def loop_gain(f, length_m, awg=26, Z=100.0):
    return insertion_gain(line_abcd(f, length_m, awg), Z, Z)


# ------------------------------------------------------------- copper: crosstalk
K_NEXT = 8.818e-14   # 1%-worst-case NEXT constant, 49 disturbers, f in Hz
K_FEXT = 7.74e-21    # 1%-worst-case FEXT constant, 49 disturbers, f in Hz, length in feet


def next_coupling(f, n_disturbers=49):
    """Power coupling |H_NEXT|^2 (1 % worst case)."""
    return K_NEXT * (n_disturbers / 49) ** 0.6 * np.asarray(f, float) ** 1.5


def fext_coupling(f, length_m, h_loop, n_disturbers=49):
    """Power coupling |H_FEXT|^2 = K f^2 l |H(f)|^2 (1 % worst case)."""
    f = np.asarray(f, float)
    return K_FEXT * (n_disturbers / 49) ** 0.6 * f ** 2 * (length_m / 0.3048) * np.abs(h_loop) ** 2


# ------------------------------------------------------------------ DMT loading
def dmt_rate(f_tones, psd_tx_dbm_hz, h2, noise_dbm_hz, gap_db=12.0, bmax=15,
             sym_rate=4000.0, df=4312.5):
    """Gap-approximation bit loading.

    f_tones: tone centre frequencies; psd_tx_dbm_hz: transmit PSD per tone (scalar or array);
    h2: |H(f)|^2 per tone; noise_dbm_hz: total noise PSD per tone (dBm/Hz).
    Returns (rate_bps, bits_per_tone, snr_db)."""
    snr_db = np.asarray(psd_tx_dbm_hz) + 10 * np.log10(np.maximum(h2, 1e-30)) - noise_dbm_hz
    b = np.floor(np.log2(1 + 10 ** ((snr_db - gap_db) / 10)))
    b = np.clip(b, 0, bmax)
    b[b < 1] = 0
    return b.sum() * sym_rate, b, snr_db


def db_sum(*dbm):
    """Power sum of quantities given in dB."""
    return 10 * np.log10(sum(10 ** (np.asarray(x) / 10) for x in dbm))


# ----------------------------------------------------------------------- fibre
def fibre_attenuation_db_km(lam_nm, water_peak=True):
    """Simple loss model of silica single-mode fibre (dB/km)."""
    lu = np.asarray(lam_nm, float) / 1e3
    rayleigh = 0.90 / lu ** 4                 # Rayleigh scattering, ~lambda^-4
    ir = 7.81e11 * np.exp(-48.48 / lu)        # Si-O infrared absorption tail
    uv = 1.1e-5 * np.exp(4.67 / lu)           # electronic (UV) absorption tail (small)
    floor = 0.01                              # waveguide imperfections
    oh = 0.0
    if water_peak:
        oh = 0.55 * np.exp(-0.5 * ((lam_nm - 1383) / 11) ** 2) + 0.06 * np.exp(-0.5 * ((lam_nm - 1240) / 12) ** 2)
    else:
        oh = 0.02 * np.exp(-0.5 * ((lam_nm - 1383) / 11) ** 2)
    return rayleigh + ir + uv + floor + oh


def dispersion_ps_nm_km(lam_nm, lam0_nm=1312.0, s0=0.092):
    """D(lambda) for G.652-type fibre from the zero-dispersion wavelength and slope (ITU-T G.652 form)."""
    lam = np.asarray(lam_nm, float)
    return s0 / 4 * (lam - lam0_nm ** 4 / lam ** 3)


def beta2_from_d(D_ps_nm_km=17.0, lam=1550e-9):
    """GVD parameter beta2 in s^2/m from D in ps/(nm km)."""
    D = D_ps_nm_km * 1e-6  # s/m^2
    return -D * lam ** 2 / (2 * np.pi * C)


def cd_transfer(f, length_m, D_ps_nm_km=17.0, lam=1550e-9):
    """All-pass CD transfer function exp(-j beta2 L w^2 / 2)."""
    b2 = beta2_from_d(D_ps_nm_km, lam)
    w = 2 * np.pi * np.asarray(f, float)
    return np.exp(-1j * b2 * length_m * w ** 2 / 2)


def apply_cd(x, fs, length_m, D_ps_nm_km=17.0, lam=1550e-9):
    f = np.fft.fftfreq(len(x), 1 / fs)
    return np.fft.ifft(np.fft.fft(x) * cd_transfer(f, length_m, D_ps_nm_km, lam))


def cd_compensate(x, fs, length_m, D_ps_nm_km=17.0, lam=1550e-9):
    """Frequency-domain CD compensation (whole-block FFT; a real receiver uses overlap-save)."""
    return apply_cd(x, fs, -length_m, D_ps_nm_km, lam)


def imdd_cd_response(f, length_m, D_ps_nm_km=17.0, lam=1550e-9):
    """Small-signal intensity response of a chirp-free IM/DD link with CD: cos(pi D L lam^2 f^2 / c)."""
    D = D_ps_nm_km * 1e-6
    return np.cos(np.pi * D * length_m * lam ** 2 * np.asarray(f, float) ** 2 / C)


def osnr_db(p_ch_dbm, span_loss_db, nf_db, n_spans, lam=1550e-9, b_ref=12.5e9):
    """OSNR in the reference bandwidth (0.1 nm = 12.5 GHz) after n identical amplified spans."""
    hnu_bref_dbm = 10 * np.log10(H_PLANCK * C / lam * b_ref * 1e3)
    return p_ch_dbm - span_loss_db - nf_db - hnu_bref_dbm - 10 * np.log10(n_spans)


def gn_eta(span_km=80, alpha_db_km=0.2, gamma=1.3e-3, D_ps_nm_km=17.0, rs=64e9,
           b_wdm=4.8e12, lam=1550e-9):
    """NLI coefficient eta (1/W^2) per span for a Nyquist-WDM comb (incoherent GN model):
    P_NLI = eta * P_ch^3 in the channel bandwidth rs."""
    a = alpha_db_km / 4.343 / 1e3          # power attenuation, 1/m
    L = span_km * 1e3
    leff = (1 - np.exp(-a * L)) / a
    leffa = 1 / a
    b2 = abs(beta2_from_d(D_ps_nm_km, lam))
    g_wdm_over_p = 1 / rs                  # PSD per unit power
    gnli = (8 / 27) * gamma ** 2 * leff ** 2 * np.arcsinh(np.pi ** 2 / 2 * b2 * leffa * b_wdm ** 2) \
        / (np.pi * b2 * leffa) * g_wdm_over_p ** 3
    return gnli * rs


def gn_snr(p_ch_w, n_spans, span_loss_db=16.0, nf_db=5.0, rs=64e9, eta=None, lam=1550e-9):
    """SNR (linear) after n spans: P / (N (P_ASE + eta P^3)), ASE in the signal bandwidth."""
    if eta is None:
        eta = gn_eta(rs=rs)
    G = 10 ** (span_loss_db / 10)
    p_ase = 10 ** (nf_db / 10) * H_PLANCK * C / lam * G * rs
    return p_ch_w / (n_spans * (p_ase + eta * p_ch_w ** 3))


# ------------------------------------------------------ coherent receiver DSP
def cma_butterfly(x, y, ntaps=15, mu=1e-3, sps=2, radius=1.0, n_pass=1):
    """Blind 2x2 CMA butterfly equaliser on T/sps-spaced inputs; outputs one sample per symbol.

    Initialised with a centre spike on the xx and yy filters (avoids the singularity)."""
    W = np.zeros((2, 2, ntaps), complex)
    W[0, 0, ntaps // 2] = 1; W[1, 1, ntaps // 2] = 1
    nsym = (len(x) - ntaps) // sps
    for _ in range(n_pass):
        zx = np.zeros(nsym, complex); zy = np.zeros(nsym, complex)
        for k in range(nsym):
            sx = x[k * sps:k * sps + ntaps][::-1]
            sy = y[k * sps:k * sps + ntaps][::-1]
            ox = np.dot(W[0, 0], sx) + np.dot(W[0, 1], sy)
            oy = np.dot(W[1, 0], sx) + np.dot(W[1, 1], sy)
            ex = ox * (radius - abs(ox) ** 2)
            ey = oy * (radius - abs(oy) ** 2)
            W[0, 0] += mu * ex * np.conj(sx); W[0, 1] += mu * ex * np.conj(sy)
            W[1, 0] += mu * ey * np.conj(sx); W[1, 1] += mu * ey * np.conj(sy)
            zx[k] = ox; zy[k] = oy
    return zx, zy, W


def fourth_power_fo(z, rs):
    """Frequency-offset estimate (Hz) of a QPSK signal from the peak of the 4th-power spectrum."""
    n = 1 << int(np.ceil(np.log2(len(z))))
    S = np.abs(np.fft.fft(z ** 4, n))
    f = np.fft.fftfreq(n, 1 / rs)
    return f[np.argmax(S)] / 4


def vv_carrier_recovery(z, block=32):
    """Viterbi-Viterbi 4th-power phase estimation with a sliding average and phase unwrapping."""
    z4 = z ** 4
    ker = np.ones(block) / block
    avg = np.convolve(z4, ker, mode="same")
    ph = np.unwrap(np.angle(-avg)) / 4      # -avg: QPSK at pi/4 offsets has z^4 = -|z|^4
    return z * np.exp(-1j * ph), ph
