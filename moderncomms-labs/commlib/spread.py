"""commlib.spread -- spread spectrum, CDMA and GNSS tools.

Used by Chapter 18 ("Spread Spectrum, CDMA and Satellite Navigation") figures and by the
spread-spectrum / GPS lab. Import explicitly:  ``from commlib import spread``.

Contents
--------
Sequences      lfsr, msequence, PRIMITIVE_POLYS, gold_codes, gold_bound, kasami_small,
               walsh, ovsf, BARKER11, bipolar, pcorr, acorr_aperiodic
GPS codes      GPS_G2_TAPS, gps_ca_code, sample_code
DSSS / CDMA    spread, despread, hop_pattern, rake_combine, mud_detect, cdma_capacity,
               jamming_margin_db
Other SS       lora_chirp, lora_demod, gaussian_monocycle
GNSS signals   gps_baseband (synthetic multi-satellite L1 C/A baseband), psd_bpsk, psd_boc
Receiver       acquire (parallel FFT code search), acq_pd (detection probability),
               dll_scurve, track (carrier-aided DLL + Costas PLL / FLL)
Navigation     gps_constellation_ecef, lla_to_ecef, ecef_to_lla, ecef_to_enu, azel,
               pseudoranges, solve_position, dop, iono_free, relativity_offsets

Conventions: chips are 0/1 integers from the generators; ``bipolar`` maps 0->+1, 1->-1.
SI units (m, s, Hz). Baseband signals are complex with unit noise power per sample
unless stated. Everything is written to be read, not to be fast.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0
F_L1, F_L2, F_L5 = 1575.42e6, 1227.60e6, 1176.45e6
CA_RATE = 1.023e6
CA_LEN = 1023
MU_E = 3.986004418e14            # WGS-84 GM, m^3/s^2
OMEGA_E = 7.2921151467e-5        # WGS-84 Earth rotation rate, rad/s
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
GPS_A = 26_559_700.0             # nominal GPS semi-major axis (m), ~20 200 km altitude

# ======================================================================= sequences
# Primitive polynomials, as lists of exponents (x^m + ... + 1).  One per degree.
PRIMITIVE_POLYS = {
    2: [2, 1, 0], 3: [3, 1, 0], 4: [4, 1, 0], 5: [5, 2, 0], 6: [6, 1, 0], 7: [7, 1, 0],
    8: [8, 4, 3, 2, 0], 9: [9, 4, 0], 10: [10, 3, 0], 11: [11, 2, 0],
    12: [12, 6, 4, 1, 0], 13: [13, 4, 3, 1, 0], 14: [14, 10, 6, 1, 0], 15: [15, 1, 0],
}

BARKER11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])   # IEEE 802.11 DSSS


def lfsr(poly, n, init=None):
    """Output ``n`` bits of the linear recurrence with characteristic polynomial ``poly``.

    ``poly`` is a list of exponents, e.g. [10, 3, 0] for x^10 + x^3 + 1. The recurrence is
    a[k+m] = XOR of a[k+e] over the exponents e < m. ``init`` is the first m bits
    (default: 0...01). With a primitive polynomial the period is 2^m - 1.
    """
    m = max(poly)
    taps = [e for e in poly if e < m]
    a = np.zeros(n + m, dtype=np.int8)
    a[:m] = np.array(init, dtype=np.int8) if init is not None else np.r_[np.zeros(m - 1), 1]
    for k in range(n):
        s = 0
        for e in taps:
            s ^= a[k + e]
        a[k + m] = s
    return a[:n].astype(int)


def msequence(m, poly=None, init=None):
    """One period (length 2^m - 1) of an m-sequence of degree m (0/1 chips)."""
    return lfsr(poly or PRIMITIVE_POLYS[m], 2 ** m - 1, init)


def bipolar(chips):
    """Map 0/1 chips to +1/-1 (0 -> +1, 1 -> -1), so XOR becomes multiplication."""
    return 1 - 2 * np.asarray(chips, dtype=float)


def pcorr(a, b=None):
    """Periodic (circular) cross-correlation r[k] = sum_n a[n] b*[n-k] via the FFT.

    Inputs are used as given (convert 0/1 chips with ``bipolar`` first). With b=None,
    the autocorrelation of a. Returns a real array if both inputs are real.
    """
    a = np.asarray(a); b = a if b is None else np.asarray(b)
    r = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))
    return r.real if np.isrealobj(a) and np.isrealobj(b) else r


def acorr_aperiodic(a):
    """Aperiodic autocorrelation for lags 0..N-1."""
    a = np.asarray(a, float)
    return np.correlate(a, a, "full")[len(a) - 1:]


def gold_bound(m):
    """Peak periodic cross-correlation t(m) of a Gold set: 2^((m+1)/2)+1 (m odd), 2^((m+2)/2)+1 (m even)."""
    return 2 ** ((m + 1) // 2) + 1 if m % 2 else 2 ** ((m + 2) // 2) + 1


def _preferred_decimation(m):
    if m % 2:
        return 3                       # k = 1: q = 2^k + 1
    if m % 4 == 2:
        return 2 ** ((m + 2) // 2) + 1
    raise ValueError("no preferred pairs exist when m is a multiple of 4")


def gold_codes(m, poly=None):
    """The Gold family of degree m: array (2^m + 1, 2^m - 1) of 0/1 chips.

    Row 0 is the m-sequence u, row 1 its preferred partner v (u decimated by q), and rows
    2.. are u XOR (v cyclically shifted by k), k = 0..N-1. Every pair of rows has periodic
    cross-correlation in {-1, -t(m), t(m) - 2}.
    """
    u = msequence(m, poly)
    N = len(u)
    q = _preferred_decimation(m)
    v = u[(q * np.arange(N)) % N]
    codes = [u, v] + [u ^ np.roll(v, -k) for k in range(N)]
    return np.array(codes)


def kasami_small(m, poly=None):
    """The small Kasami set (m even): array (2^(m/2), 2^m - 1) of 0/1 chips.

    Peak cross-correlation 2^(m/2) + 1, which meets the Welch bound asymptotically.
    """
    if m % 2:
        raise ValueError("small Kasami sets need even m")
    u = msequence(m, poly)
    N = len(u)
    s = 2 ** (m // 2) + 1
    w = u[(s * np.arange(N) + 1) % N]   # period 2^(m/2) - 1 (offset 1: phase 0 is all zeros)
    return np.array([u] + [u ^ np.roll(w, -k) for k in range(2 ** (m // 2) - 1)])


def walsh(N):
    """Sylvester-Hadamard (Walsh) matrix of order N (power of two), entries +/-1, natural order."""
    H = np.array([[1]])
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    return H


def ovsf(sf, k=None):
    """OVSF channelisation codes of spreading factor ``sf`` (3GPP TS 25.213 ordering).

    Returns the (sf, sf) matrix of codes C_{sf,0..sf-1}, or the single code C_{sf,k}.
    C_{2n,2i} = [C_{n,i}, C_{n,i}] and C_{2n,2i+1} = [C_{n,i}, -C_{n,i}].
    """
    C = np.array([[1]])
    while C.shape[0] < sf:
        C = np.vstack([np.hstack([c, s * c]) for c in C for s in (1, -1)])
    return C if k is None else C[k]


# ======================================================================= GPS C/A codes
# G2 phase-selector taps (IS-GPS-200, Table 3-Ia) for PRN 1..32.
GPS_G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10), 7: (1, 8), 8: (2, 9),
    9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9),
    23: (1, 3), 24: (4, 6), 25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6),
    30: (2, 7), 31: (3, 8), 32: (4, 9),
}


def gps_ca_code(prn):
    """The 1023-chip GPS L1 C/A code for satellite ``prn`` (1..32), as 0/1 chips.

    G1 = 1 + x^3 + x^10, G2 = 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10, both registers
    start at all ones; the chip is G1(10) XOR G2(t1) XOR G2(t2). The first ten chips of
    PRN 1 are 1100100000 (octal 1440), as listed in IS-GPS-200.
    """
    t1, t2 = GPS_G2_TAPS[prn]
    g1 = [1] * 10; g2 = [1] * 10
    out = np.empty(CA_LEN, dtype=int)
    for i in range(CA_LEN):
        out[i] = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [f1] + g1[:9]
        g2 = [f2] + g2[:9]
    return out


def sample_code(code, fs, n, chip_rate=CA_RATE, phase_chips=0.0):
    """Sample a (bipolar or 0/1) periodic code at rate ``fs``: n samples starting at code
    phase ``phase_chips``. Returns the code values (no filtering: rectangular chips)."""
    code = np.asarray(code)
    idx = np.floor(phase_chips + np.arange(n) * chip_rate / fs).astype(np.int64) % len(code)
    return code[idx]


# ======================================================================= DSSS and CDMA
def spread(symbols, code):
    """Direct-sequence spreading at one sample per chip: each symbol times the whole code."""
    return (np.asarray(symbols)[:, None] * np.asarray(code)[None, :]).ravel()


def despread(chips, code):
    """Correlate a chip-rate stream with ``code`` block by block (integrate-and-dump).
    Returns one statistic per symbol, normalised so a clean +1 symbol gives +1."""
    code = np.asarray(code); L = len(code)
    n = len(chips) // L
    return (np.asarray(chips)[:n * L].reshape(n, L) @ np.conj(code)) / np.sum(np.abs(code) ** 2)


def jamming_margin_db(gp_db, ebno_req_db, losses_db=0.0):
    """Jamming margin = processing gain - required Eb/N0 - implementation losses (dB)."""
    return gp_db - ebno_req_db - losses_db


def hop_pattern(n_hops, n_channels, rng=None, exclude=()):
    """A pseudo-random hop sequence over n_channels, avoiding channels in ``exclude``
    (adaptive frequency hopping) and never repeating a channel on consecutive hops."""
    rng = np.random.default_rng(rng)
    allowed = np.array([c for c in range(n_channels) if c not in set(exclude)])
    seq = [rng.choice(allowed)]
    for _ in range(n_hops - 1):
        c = rng.choice(allowed)
        while c == seq[-1]:
            c = rng.choice(allowed)
        seq.append(c)
    return np.array(seq)


def rake_combine(r, code, n_sym, delays, gains=None):
    """Rake receiver at one sample per chip.

    r: received chip-rate samples; code: spreading code of one symbol (length L);
    delays: finger delays in chips; gains: complex path gains (maximal-ratio combining),
    or None for equal-gain combining. Returns (combined statistics, per-finger outputs).
    """
    code = np.asarray(code); L = len(code)
    r = np.r_[np.asarray(r), np.zeros(max(delays) + L, complex)]
    fingers = np.array([despread(r[d:d + n_sym * L], code) for d in delays])
    w = np.ones(len(delays)) if gains is None else np.conj(np.asarray(gains))
    return w @ fingers, fingers


def mud_detect(y, S, A, method="conventional", sigma2=1.0):
    """Synchronous CDMA multiuser detection for y = S diag(A) b + n.

    S: (L, K) signature matrix with unit-energy columns; A: (K,) amplitudes; y: (L,) or
    (L, n) received vectors. method: 'conventional' (matched filter bank), 'decorrelator',
    'mmse', or 'sic' (successive cancellation, strongest first). Returns +/-1 decisions (K, n).
    """
    y = np.atleast_2d(np.asarray(y, float).T).T if np.ndim(y) == 1 else np.asarray(y, float)
    A = np.asarray(A, float)
    z = S.T @ y
    R = S.T @ S
    if method == "conventional":
        return np.sign(z)
    if method == "decorrelator":
        return np.sign(np.linalg.solve(R, z))
    if method == "mmse":
        return np.sign(np.linalg.solve(R + sigma2 * np.diag(1 / A ** 2), z))
    if method == "sic":
        b = np.zeros_like(z); resid = y.copy()
        for k in np.argsort(-A):
            b[k] = np.sign(S[:, k] @ resid)
            resid = resid - np.outer(S[:, k] * A[k], b[k])
        return b
    raise ValueError(method)


def cdma_capacity(W, R, ebi0_db, voice_activity=1.0, sector_gain=1.0, other_cell=0.0,
                  loading=1.0):
    """Users per cell of a power-controlled CDMA uplink (Gilhousen et al. 1991 style):

        N = 1 + (W/R) / (Eb/I0) * sector_gain * loading / (voice_activity * (1 + other_cell))

    W: chip rate (Hz), R: bit rate (b/s), ebi0_db: required Eb/(N0+I0), other_cell: ratio
    f of other-cell to own-cell interference, loading: fraction of the pole capacity used
    (1 = ignore thermal noise). An engineering estimate, not a law.
    """
    return 1 + (W / R) / 10 ** (np.asarray(ebi0_db) / 10) * sector_gain * loading / (
        voice_activity * (1 + other_cell))


# ======================================================================= other spread spectrum
def lora_chirp(sf, bw, fs, symbol=0, down=False):
    """One LoRa-style chirp symbol: 2^sf chips at chip rate bw, cyclically shifted by
    ``symbol``, sampled at fs (fs an integer multiple of bw). Complex baseband."""
    M = 2 ** sf; osr = int(round(fs / bw))
    n = np.arange(M * osr) / osr                          # time in chips
    k = (n + symbol) % M                                   # cyclic shift
    phase = 2 * np.pi * (k ** 2 / (2 * M) - k / 2)
    x = np.exp(1j * phase)
    return np.conj(x) if down else x


def lora_demod(x, sf, bw, fs):
    """Dechirp (multiply by the conjugate base up-chirp) and FFT; returns (symbol, spectrum)."""
    M = 2 ** sf; osr = int(round(fs / bw))
    d = x[:M * osr] * np.conj(lora_chirp(sf, bw, fs, 0))
    X = np.abs(np.fft.fft(d[::osr], M))
    return int(np.argmax(X)), X


def gaussian_monocycle(t, tau):
    """Gaussian monocycle (first derivative of a Gaussian), peak-normalised, width tau (s)."""
    x = t / tau
    return x * np.exp(0.5 - 0.5 * x ** 2)


# ======================================================================= GNSS signals
def psd_bpsk(f, chip_rate):
    """Unit-power PSD (1/Hz) of BPSK-R with rectangular chips: Tc sinc^2(f Tc)."""
    Tc = 1 / chip_rate
    return Tc * np.sinc(f * Tc) ** 2


def psd_boc(f, m, n, f0=1.023e6):
    """Unit-power PSD (1/Hz) of sine-phased BOC(m, n): subcarrier m*f0, chip rate n*f0."""
    fs, fc = m * f0, n * f0
    k = int(round(2 * fs / fc))
    f = np.where(np.abs(f) < 1e-3, 1e-3, f)
    if k % 2 == 0:
        g = fc * (np.sin(np.pi * f / fc) * np.tan(np.pi * f / (2 * fs)) / (np.pi * f)) ** 2
    else:
        g = fc * (np.cos(np.pi * f / fc) * np.tan(np.pi * f / (2 * fs)) / (np.pi * f)) ** 2
    return g


def gps_baseband(sats, fs, duration, rng=None, nav=True):
    """Synthetic complex-baseband GPS L1 C/A signal with unit-variance complex noise.

    sats: list of dicts with keys prn, cn0 (dB-Hz), doppler (Hz), code_phase (chips at t=0),
    optionally phase (rad) and bits (array of +/-1 navigation bits at 50 b/s; random if
    missing) and bit_offset_ms (0..19; bit edges fall on code epochs, as in GPS). Code
    Doppler is included. Returns (x, truth), truth being the list of dicts actually used.
    """
    rng = np.random.default_rng(rng)
    n = int(round(fs * duration)); t = np.arange(n) / fs
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    truth = []
    for s in sats:
        code = bipolar(gps_ca_code(s["prn"]))
        fd = s.get("doppler", 0.0)
        chip_rate = CA_RATE * (1 + fd / F_L1)
        amp = np.sqrt(10 ** (s["cn0"] / 10) / fs)          # C/N0 with N0 = 1/fs
        c = sample_code(code, fs, n, chip_rate, s.get("code_phase", 0.0))
        ph = s.get("phase", rng.uniform(0, 2 * np.pi))
        sig = amp * c * np.exp(1j * (2 * np.pi * fd * t + ph))
        if nav:
            off = s.get("bit_offset_ms", int(rng.integers(0, 20)))
            nb = int(np.ceil((duration * 1e3 + off) / 20)) + 1
            bits = np.asarray(s.get("bits", rng.choice([-1, 1], nb)))
            cp = s.get("code_phase", 0.0) + t * chip_rate          # bit edges on code epochs
            idx = np.floor((cp + off * CA_LEN) / (20 * CA_LEN)).astype(int)
            sig = sig * bits[np.minimum(idx, len(bits) - 1)]
        x = x + sig
        truth.append(dict(s, phase=ph))
    return x, truth


# ======================================================================= acquisition
def acquire(x, fs, prn, dopplers, n_coh_ms=1, n_noncoh=1, code=None):
    """Parallel code-phase search (FFT circular correlation) over a grid of Doppler bins.

    Uses n_noncoh consecutive blocks of n_coh_ms milliseconds each; each block is
    correlated coherently, the |.|^2 are summed non-coherently. Returns a dict with
    'grid' (len(dopplers), samples_per_ms; column k = code epoch starting at sample k),
    'doppler' (Hz), 'sample' (k of the peak), 'code_phase' (chips of the incoming code at
    sample 0, the form ``track`` expects) and 'metric' (peak / grid mean).
    """
    code = bipolar(gps_ca_code(prn)) if code is None else np.asarray(code)
    Ns = int(round(fs * 1e-3)); Nb = Ns * n_coh_ms
    local = sample_code(code, fs, Nb)
    Lf = np.conj(np.fft.fft(local))
    t = np.arange(Nb) / fs
    grid = np.zeros((len(dopplers), Ns))
    for i, fd in enumerate(dopplers):
        wipe = np.exp(-2j * np.pi * fd * t)
        for b in range(n_noncoh):
            blk = x[b * Nb:(b + 1) * Nb]
            r = np.fft.ifft(np.fft.fft(blk * wipe) * Lf)
            p = np.abs(r) ** 2
            grid[i] += p.reshape(n_coh_ms, Ns).max(axis=0) if n_coh_ms > 1 else p
    i, k = np.unravel_index(np.argmax(grid), grid.shape)
    return dict(grid=grid, doppler=dopplers[i], sample=k,
                code_phase=(-k * CA_RATE / fs) % len(code), metric=grid[i, k] / grid.mean())


def acq_pd(cn0_dbhz, t_coh, k_noncoh=1, pfa=1e-3, loss_db=0.0):
    """Probability of detection of one search cell for a square-law detector:
    K non-coherent sums of |z|^2 with coherent SNR = C/N0 * T (minus loss_db), at false-
    alarm probability pfa per cell. Uses non-central chi-square with 2K degrees of freedom."""
    from scipy import stats
    snr = 10 ** ((np.asarray(cn0_dbhz, float) - loss_db) / 10) * t_coh
    thr = stats.chi2.isf(pfa, 2 * k_noncoh)
    return stats.ncx2.sf(thr, 2 * k_noncoh, 2 * k_noncoh * snr)


# ======================================================================= tracking
def _tri(tau):
    return np.maximum(0.0, 1 - np.abs(tau))


def dll_scurve(err_chips, spacing=1.0, kind="nelp"):
    """DLL discriminator output versus true code error (chips) for an ideal triangle
    correlation function (infinite front-end bandwidth).

    spacing: early-late spacing d (chips; early at -d/2, late at +d/2).
    kind: 'emle' early-minus-late envelope, 'nelp' normalised early-minus-late power,
    'nemle' normalised early-minus-late envelope, 'elp' early-minus-late power. All scaled to unit slope at zero error, so the
    output reads directly in chips inside the linear region.
    """
    e = np.asarray(err_chips, float)
    E, L = _tri(e - spacing / 2), _tri(e + spacing / 2)
    if kind == "nelp":
        return (1 - spacing / 2) / 2 * (E ** 2 - L ** 2) / (E ** 2 + L ** 2 + 1e-12)
    if kind == "emle":
        return (E - L) / 2
    if kind == "nemle":
        return (1 - spacing / 2) * (E - L) / (E + L + 1e-12)
    if kind == "elp":
        return (E ** 2 - L ** 2) / (4 * (1 - spacing / 2))
    raise ValueError(kind)


def loop_coeffs(bn, zeta=0.7071):
    """Natural frequency (rad/s) of a 2nd-order loop with noise bandwidth bn (Hz)."""
    return 8 * zeta * bn / (4 * zeta ** 2 + 1)


def track(x, fs, prn, doppler0, code_phase0, n_ms=None, dll_bn=2.0, pll_bn=15.0,
          fll_bn=0.0, spacing=1.0, zeta=0.7071, code=None):
    """Carrier-aided DLL + Costas PLL (optionally FLL-assisted) tracking of one C/A signal.

    x: complex baseband samples at rate fs; doppler0 (Hz) and code_phase0 (chips) come
    from acquisition. One loop update per 1 ms code period (integrate and dump).
    spacing: early-late spacing in chips. Returns a dict of per-ms arrays:
    IP, QP, IE, QE, IL, QL (correlator outputs), doppler (Hz), code_rate (chips/s),
    code_err (DLL discriminator, chips), phase_err (Costas, cycles), code_phase (local code
    phase in chips at the first sample of each block), sample (index of that sample), cn0 (dB-Hz, moments estimate after
    the first quarter of the run).
    """
    code = bipolar(gps_ca_code(prn)) if code is None else np.asarray(code)
    N = len(code)
    ext = np.r_[code[-1], code, code[0]]                  # padding for early / late
    wd, wp, wf = loop_coeffs(dll_bn, zeta), loop_coeffs(pll_bn, zeta), loop_coeffs(fll_bn, zeta) if fll_bn else 0
    T = 1e-3
    carr_f = doppler0; carr_int = doppler0; carr_phase = 0.0
    code_int = 0.0
    rem_code = code_phase0 % N            # chips of code already elapsed at sample 0
    # start processing at the next code epoch so each block is one full code period
    code_rate = CA_RATE * (1 + carr_f / F_L1)
    start = int(np.ceil(((N - rem_code) % N) * fs / code_rate))
    pos = start; ph_code = 0.0            # code phase (chips) at 'pos'
    out = {k: [] for k in ("IP", "QP", "IE", "QE", "IL", "QL", "doppler", "code_rate",
                           "code_err", "phase_err", "code_phase", "sample")}
    prev_p = None
    k = 0; code_corr = 0.0
    while True:
        code_rate = CA_RATE * (1 + carr_f / F_L1) + code_corr
        step = code_rate / fs
        nblk = int(np.ceil((N - ph_code) / step))
        if pos + nblk > len(x) or (n_ms is not None and k >= n_ms):
            break
        tc = ph_code + np.arange(nblk) * step
        blk = x[pos:pos + nblk]
        tt = np.arange(nblk) / fs
        wipe = np.exp(-1j * (2 * np.pi * carr_f * tt + carr_phase))
        bb = blk * wipe
        def corr(off):
            i = np.floor(tc + off).astype(int)
            return np.sum(bb * ext[np.clip(i, -1, N) + 1])
        E, P, L = corr(spacing / 2), corr(0.0), corr(-spacing / 2)
        carr_phase = (carr_phase + 2 * np.pi * carr_f * nblk / fs) % (2 * np.pi)
        # Costas discriminator (cycles), insensitive to data bits
        pe = np.arctan(P.imag / P.real) / (2 * np.pi) if P.real != 0 else 0.0
        fe = 0.0
        if fll_bn and prev_p is not None:
            cross = prev_p.real * P.imag - prev_p.imag * P.real
            dot = prev_p.real * P.real + prev_p.imag * P.imag
            fe = np.arctan(cross / dot) / (2 * np.pi * T) if dot != 0 else 0.0
        prev_p = P
        # PLL (+FLL) loop filter -> carrier frequency (Hz)
        carr_int += wp ** 2 * T * pe + (wf * T * fe if fll_bn else 0.0)
        carr_f = carr_int + 2 * zeta * wp * pe
        # normalised early-minus-late envelope DLL -> code rate correction (chips/s)
        aE, aL = np.abs(E), np.abs(L)
        ce = (1 - spacing / 2) * (aE - aL) / (aE + aL + 1e-30)
        code_int += wd ** 2 * T * ce
        code_corr = code_int + 2 * zeta * wd * ce
        for key, v in (("IP", P.real), ("QP", P.imag), ("IE", E.real), ("QE", E.imag),
                       ("IL", L.real), ("QL", L.imag), ("doppler", carr_f),
                       ("code_rate", code_rate), ("code_err", ce), ("phase_err", pe),
                       ("sample", pos), ("code_phase", ph_code)):
            out[key].append(v)
        ph_code = (ph_code + nblk * step) - N
        pos += nblk
        k += 1
    res = {k: np.array(v) for k, v in out.items()}
    # C/N0 estimate by the moments method (insensitive to data-bit flips):
    # P2 = E|P|^2 = A^2 + 2s^2, P4 = E|P|^4 = A^4 + 8A^2 s^2 + 8s^4  ->  A^2 = sqrt(2 P2^2 - P4)
    p = res["IP"][len(res["IP"]) // 4:] ** 2 + res["QP"][len(res["QP"]) // 4:] ** 2
    if len(p) > 10:
        P2, P4 = p.mean(), (p ** 2).mean()
        a2 = np.sqrt(max(2 * P2 ** 2 - P4, 1e-30))
        res["cn0"] = 10 * np.log10(max(a2 / max(P2 - a2, 1e-30), 1e-6) / T)
    else:
        res["cn0"] = np.nan
    return res


# ======================================================================= navigation geometry
def lla_to_ecef(lat_deg, lon_deg, h=0.0):
    """WGS-84 geodetic latitude, longitude (deg), height (m) to ECEF (m)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    Nr = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(Nr + h) * np.cos(lat) * np.cos(lon), (Nr + h) * np.cos(lat) * np.sin(lon),
                     (Nr * (1 - e2) + h) * np.sin(lat)])


def ecef_to_lla(r):
    """ECEF (m) to WGS-84 latitude, longitude (deg), height (m) (iterative)."""
    x, y, z = r
    e2 = WGS84_F * (2 - WGS84_F)
    lon = np.arctan2(y, x); p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2)); h = 0.0
    for _ in range(10):
        Nr = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - Nr
        lat = np.arctan2(z, p * (1 - e2 * Nr / (Nr + h)))
    return np.degrees(lat), np.degrees(lon), h


def ecef_to_enu(d, lat_deg, lon_deg):
    """Rotate ECEF vector(s) d (3,) or (3, n) into local east-north-up at (lat, lon)."""
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    Rm = np.array([[-np.sin(lo), np.cos(lo), 0],
                   [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                   [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return Rm @ d


def azel(sat_ecef, user_ecef):
    """Azimuth and elevation (deg) of satellite(s) (3,) or (3, n) seen from user_ecef."""
    lat, lon, _ = ecef_to_lla(user_ecef)
    d = np.asarray(sat_ecef) - (np.asarray(user_ecef)[:, None] if np.ndim(sat_ecef) == 2 else user_ecef)
    e, n, u = ecef_to_enu(d, lat, lon)
    rng_ = np.sqrt(e ** 2 + n ** 2 + u ** 2)
    return np.degrees(np.arctan2(e, n)) % 360, np.degrees(np.arcsin(u / rng_))


def gps_constellation_ecef(t, n_planes=6, per_plane=4, inc_deg=55.0, a=GPS_A, phasing=2):
    """ECEF positions (n_sats, 3, len(t)) of an idealised Walker 24/6/2 GPS-like
    constellation (circular orbits, equal spacing). Real GPS slots are unevenly spaced
    and there are about 31 satellites; this is a teaching model."""
    from .satellite import orbit_eci
    t = np.atleast_1d(np.asarray(t, float))
    T = n_planes * per_plane
    pos = []
    for p in range(n_planes):
        for s in range(per_plane):
            M0 = 360.0 * s / per_plane + 360.0 * phasing * p / T
            r = orbit_eci(a, 0.0, inc_deg, 60.0 * p * 6 / n_planes, 0.0, M0, t)
            th = OMEGA_E * t
            x = np.cos(th) * r[0] + np.sin(th) * r[1]
            y = -np.sin(th) * r[0] + np.cos(th) * r[1]
            pos.append(np.vstack([x, y, r[2]]))
    return np.array(pos)


def pseudoranges(sat_pos, user_pos, clock_bias_m=0.0, sigma_m=0.0, rng=None):
    """Pseudoranges rho_k = |s_k - u| + b + noise for satellites sat_pos (K, 3) (m)."""
    rng = np.random.default_rng(rng)
    d = np.linalg.norm(np.asarray(sat_pos) - np.asarray(user_pos)[None, :], axis=1)
    return d + clock_bias_m + sigma_m * rng.standard_normal(len(d))


def solve_position(sat_pos, rho, x0=None, n_iter=10, weights=None, tol=1e-4):
    """Iterative least-squares (Gauss-Newton) solution of rho_k = |s_k - u| + b.

    sat_pos: (K, 3) ECEF (m); rho: (K,) pseudoranges (m); x0: initial [x, y, z, b]
    (default: Earth centre, zero bias). Returns (x (4,), history list of states, G matrix
    at the solution). Needs K >= 4.
    """
    S = np.asarray(sat_pos, float); rho = np.asarray(rho, float)
    x = np.zeros(4) if x0 is None else np.asarray(x0, float).copy()
    W = np.eye(len(rho)) if weights is None else np.diag(weights)
    hist = [x.copy()]
    for _ in range(n_iter):
        d = S - x[:3]
        r = np.linalg.norm(d, axis=1)
        G = np.hstack([-d / r[:, None], np.ones((len(r), 1))])
        dr = rho - (r + x[3])
        dx = np.linalg.solve(G.T @ W @ G, G.T @ W @ dr)
        x = x + dx
        hist.append(x.copy())
        if np.linalg.norm(dx) < tol:
            break
    d = S - x[:3]; r = np.linalg.norm(d, axis=1)
    G = np.hstack([-d / r[:, None], np.ones((len(r), 1))])
    return x, hist, G


def dop(sat_pos, user_pos):
    """Dilution of precision for satellites sat_pos (K, 3) seen from user_pos (ECEF, m).
    Returns dict GDOP, PDOP, HDOP, VDOP, TDOP (computed in the local ENU frame)."""
    lat, lon, _ = ecef_to_lla(user_pos)
    d = (np.asarray(sat_pos) - np.asarray(user_pos)[None, :]).T
    enu = ecef_to_enu(d, lat, lon)
    u = enu / np.linalg.norm(enu, axis=0)
    G = np.vstack([-u, np.ones(u.shape[1])]).T
    Q = np.linalg.inv(G.T @ G)
    return dict(GDOP=np.sqrt(np.trace(Q)), PDOP=np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2]),
                HDOP=np.sqrt(Q[0, 0] + Q[1, 1]), VDOP=np.sqrt(Q[2, 2]), TDOP=np.sqrt(Q[3, 3]))


def iono_free(p1, p2, f1=F_L1, f2=F_L2):
    """Ionosphere-free combination of pseudoranges (or carrier phases in metres)."""
    return (f1 ** 2 * np.asarray(p1) - f2 ** 2 * np.asarray(p2)) / (f1 ** 2 - f2 ** 2)


def iono_delay_m(tec_tecu, f):
    """First-order ionospheric group delay (m) for slant TEC in TEC units (1e16 e/m^2)."""
    return 40.3 * tec_tecu * 1e16 / f ** 2


def relativity_offsets(a=GPS_A, r_ground=WGS84_A):
    """Satellite clock rate offsets (microseconds per day) for a circular orbit of radius a:
    gravitational (general relativity, clock runs fast), velocity (special relativity, runs
    slow), and net. The Earth's rotation of the ground clock is neglected."""
    c2 = C_LIGHT ** 2
    gr = MU_E / c2 * (1 / r_ground - 1 / a)
    sr = -MU_E / a / (2 * c2)
    day = 86400e6
    return dict(gr_us_day=gr * day, sr_us_day=sr * day, net_us_day=(gr + sr) * day,
                net_fractional=gr + sr)
