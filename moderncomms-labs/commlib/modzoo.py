"""modzoo: the constellation zoo of Chapter 9 and the closed forms that go with it.

Companion of Chapter 9 (*Digital Modulation and Optimal Detection*) and Lab 02. The shapes
(APSK, cross QAM) and the exact error-rate formulas (Craig's PSK integral, coherent and
noncoherent orthogonal signalling) are the ones that drew the chapter's figures
(book/figscripts/ch09_figs.py), so numbers in the lab and the book agree. Import explicitly:

    from commlib import modzoo as mz
    c = mz.constellation("16apsk", ratio=2.85)       # a labelled cl.Constellation
    mz.ebn0_required("8psk", 1e-5)                  # 13.0 dB
    mz.shannon_ebn0_db(2.0)                         # 1.76 dB

All constellations have unit average energy (the course convention).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import comb, erfc
from scipy.stats import norm

from .modulation import Constellation, get_constellation, qfunc, ber_bpsk, ber_mqam_gray

__all__ = ["NAMES", "constellation", "apsk", "cross_qam", "natural_labels", "optimise_labels",
           "nn_bit_flips", "nearest_neighbours", "papr_db", "ser_psk_exact", "ser_orth_coherent",
           "ser_orth_noncoherent", "ber_orth", "union_bound_ser", "nn_approx_ser", "ser_theory",
           "ber_theory", "ebn0_required", "shannon_ebn0_db", "ber_dbpsk", "ber_dqpsk",
           "ber_rayleigh_mrc", "ber_fading_mrc", "ber_dbpsk_rayleigh", "ber_ncfsk_rayleigh", "ber_qam_rayleigh"]

_lin = lambda x: 10 ** (np.asarray(x, float) / 10)

# display name -> internal key
NAMES = {"BPSK": "bpsk", "QPSK": "qpsk", "8-PSK": "8psk", "16-PSK": "16psk", "4-PAM": "4pam",
         "8-PAM": "8pam", "16-QAM": "16qam", "32-cross QAM": "32cross", "64-QAM": "64qam",
         "256-QAM": "256qam", "16-APSK": "16apsk", "32-APSK": "32apsk"}


# ============================================================================ shapes
def apsk(rings, radii, offsets=None, name="APSK"):
    """Amplitude-phase shift keying: ``rings`` points per ring at relative ``radii``.
    Labels are a ring-by-ring Gray code (outer bits = ring, inner bits = Gray along the ring
    when the ring size is a power of two); use ``optimise_labels`` for a better one."""
    pts = []
    for i, (n, r) in enumerate(zip(rings, radii)):
        off = offsets[i] if offsets is not None else np.pi / n
        pts += list(r * np.exp(1j * (2 * np.pi * np.arange(n) / n + off)))
    pts = np.array(pts)
    return Constellation(pts, natural_labels(len(pts)), name)


def cross_qam(M):
    """32- or 128-point cross constellation (a square grid with its corners removed)."""
    L = 6 if M == 32 else 12
    lev = np.arange(-(L - 1), L, 2)
    I, Qg = np.meshgrid(lev, lev)
    p = (I + 1j * Qg).ravel()
    c = 1 if M == 32 else 2
    edge = lev[-c]
    keep = ~((np.abs(p.real) >= edge) & (np.abs(p.imag) >= edge))
    p = p[keep]
    assert len(p) == M, len(p)
    return Constellation(p, natural_labels(M), f"{M}-cross QAM")


def natural_labels(M):
    return np.arange(M)


def nearest_neighbours(points, tol=1e-6):
    """Index pairs (i, j), i < j, of points at the minimum distance."""
    p = np.asarray(points)
    d = np.abs(p[:, None] - p[None, :])
    dmin = d[d > tol].min()
    i, j = np.where(np.triu(np.abs(d - dmin) < tol * max(1.0, dmin) + 1e-9, 1))
    return i, j, dmin


def nn_bit_flips(c):
    """Average number of label bits that differ between nearest neighbours (1.0 = Gray)."""
    i, j, _ = nearest_neighbours(c.points)
    x = c.labels[i] ^ c.labels[j]
    return float(np.mean([bin(int(v)).count("1") for v in x]))


def _cost(points, labels, w):
    x = labels[:, None] ^ labels[None, :]
    hd = np.zeros_like(x, dtype=float)
    for b in range(int(np.log2(len(points)))):
        hd += (x >> b) & 1
    return float(np.sum(w * hd))


@lru_cache(maxsize=64)
def _optimised(key, k, passes):
    pts = np.array(key[0]) + 1j * np.array(key[1])
    M = len(pts)
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = d[d > 1e-9].min()
    w = np.exp(-((d / dmin) ** 2 - 1) * 3.0)          # weight close pairs most
    np.fill_diagonal(w, 0)
    lab = np.arange(M)
    best = _cost(pts, lab, w)
    for _ in range(passes):                            # binary switching (Zeger & Gersho)
        improved = False
        for a in range(M):
            for b in range(a + 1, M):
                lab[a], lab[b] = lab[b], lab[a]
                c = _cost(pts, lab, w)
                if c < best - 1e-12:
                    best, improved = c, True
                else:
                    lab[a], lab[b] = lab[b], lab[a]
        if not improved:
            break
    return tuple(int(v) for v in lab)


def optimise_labels(c, passes=3):
    """A good (often Gray) labelling for any constellation by binary switching: minimise the
    number of bits that differ between close neighbours. Cached, so cheap on repeat calls."""
    key = (tuple(np.round(c.points.real, 6)), tuple(np.round(c.points.imag, 6)))
    lab = np.array(_optimised(key, c.k, passes))
    return Constellation(c.points, lab, c.name)


def constellation(name, ratio=None, labels="gray"):
    """A named, labelled, unit-energy constellation.

    name: bpsk, qpsk, 8psk, 16psk, 4pam, 8pam, 16qam, 64qam, 256qam (Gray labels from
    commlib), 32cross, 16apsk (4+12, ring ratio ``ratio``, default 2.85 as in DVB-S2 rate 3/4),
    32apsk (4+12+16, radii 1 : ratio : 1.855·ratio, default 2.84).
    labels: "gray" (best available: true Gray, or binary-switching optimised), "natural",
    or "random"."""
    n = name.lower()
    if n == "32cross":
        c = cross_qam(32)
    elif n == "16apsk":
        r = 2.85 if ratio is None else ratio
        c = apsk([4, 12], [1, r], [np.pi / 4, np.pi / 12], "16-APSK")
    elif n == "32apsk":
        r = 2.84 if ratio is None else ratio
        c = apsk([4, 12, 16], [1, r, r * 5.27 / 2.84], [np.pi / 4, np.pi / 12, 0], "32-APSK")
    else:
        c = get_constellation(n)
        if labels == "gray":
            return c
    if labels == "gray":
        return optimise_labels(c)
    if labels == "natural":
        if n.endswith("qam") and n[:-3].isdigit():
            M = c.M
            L = int(round(np.sqrt(M)))
            kh = int(np.log2(L))
            lev = np.arange(-(L - 1), L, 2)
            I = np.searchsorted(lev, np.round(c.points.real / np.min(np.abs(c.points.real)), 6) - 1e-6)
            Qi = np.searchsorted(lev, np.round(c.points.imag / np.min(np.abs(c.points.imag)), 6) - 1e-6)
            return Constellation(c.points, (I << kh) | Qi, c.name)
        if n.endswith("psk") or n.endswith("pam") or n == "bpsk":
            if n == "bpsk":
                return c
            ang = np.angle(c.points) if not n.endswith("pam") else c.points.real
            order = np.argsort(np.argsort(np.mod(ang + 1e-9, 2 * np.pi) if not n.endswith("pam") else ang))
            return Constellation(c.points, order, c.name)
        return Constellation(c.points, natural_labels(c.M), c.name)
    rng = np.random.default_rng(7)
    return Constellation(c.points, rng.permutation(c.M), c.name)


def papr_db(c):
    """Peak-to-average power of the constellation points themselves (dB)."""
    return float(10 * np.log10(np.max(np.abs(c.points) ** 2) / np.mean(np.abs(c.points) ** 2)))


# ============================================================================ error rates
_TH = np.linspace(1e-6, 1.0, 400)


def ser_psk_exact(esn0_db, M):
    """Craig's formula for M-PSK symbol error probability (vectorised fixed quadrature)."""
    g = _lin(np.atleast_1d(esn0_db))[:, None]
    th = _TH[None, :] * (M - 1) * np.pi / M
    f = np.exp(-g * np.sin(np.pi / M) ** 2 / np.sin(th) ** 2)
    return np.trapezoid(f, th, axis=1) / np.pi


_U = np.linspace(-9, 9, 1801)


def ser_orth_coherent(esn0_db, M):
    """Exact SER of coherent M-ary orthogonal signalling (numerical integral)."""
    g = _lin(np.atleast_1d(esn0_db))[:, None]
    lc = norm.logcdf(_U[None, :] + np.sqrt(2 * g))
    return np.trapezoid(norm.pdf(_U)[None, :] * -np.expm1((M - 1) * lc), _U, axis=1)


def ser_orth_noncoherent(esn0_db, M):
    """Exact SER of noncoherent (energy-detected) M-ary orthogonal FSK.
    Small M: the classical alternating sum; large M (where that sum cancels catastrophically):
    Ps = 1 - integral of the Rice pdf times (1 - e^{-x^2})^(M-1)."""
    g = _lin(np.atleast_1d(esn0_db))
    if M <= 8:
        s = np.zeros_like(g)
        for n in range(1, M):
            s += (-1) ** (n + 1) * comb(M - 1, n) / (n + 1) * np.exp(-n * g / (n + 1))
        return s
    from scipy.special import i0e
    out = []
    for gg in g:
        a = np.sqrt(gg)
        x = np.linspace(1e-6, a + 9, 3000)
        f = 2 * x * np.exp(-(x - a) ** 2) * i0e(2 * x * a)
        cdf = np.exp((M - 1) * np.log1p(-np.exp(-x ** 2)))
        out.append(np.trapezoid(f * (1 - cdf), x))
    return np.array(out)


def ber_orth(ebn0_db, M, coherent=True):
    """Bit error rate of M-ary orthogonal FSK: Pb = M/(2(M-1)) Ps."""
    k = np.log2(M)
    es = np.asarray(ebn0_db, float) + 10 * np.log10(k)
    ps = ser_orth_coherent(es, M) if coherent else ser_orth_noncoherent(es, M)
    return M / (2 * (M - 1)) * ps


def union_bound_ser(c, esn0_db):
    """Full union bound (1/M) sum_i sum_{j != i} Q(d_ij / sqrt(2 N0)), Es = 1."""
    d = np.abs(c.points[:, None] - c.points[None, :])
    d = d[~np.eye(c.M, dtype=bool)]
    n0 = 1 / _lin(np.atleast_1d(esn0_db))
    return np.array([np.sum(qfunc(d / np.sqrt(2 * v))) / c.M for v in n0])


def nn_approx_ser(c, esn0_db):
    """Nearest-neighbour approximation N_min Q(d_min / sqrt(2 N0))."""
    i, j, dmin = nearest_neighbours(c.points)
    nbar = 2 * len(i) / c.M
    n0 = 1 / _lin(np.atleast_1d(esn0_db))
    return nbar * qfunc(dmin / np.sqrt(2 * n0))


def ser_theory(name, esn0_db):
    """Exact (or very tight) symbol error rate for the named constellation, Es/N0 in dB."""
    n = name.lower()
    es = np.atleast_1d(np.asarray(esn0_db, float))
    if n == "bpsk":
        return qfunc(np.sqrt(2 * _lin(es)))
    if n == "qpsk":
        p = qfunc(np.sqrt(_lin(es)))
        return 1 - (1 - p) ** 2
    if n.endswith("psk") and n[:-3].isdigit():
        return ser_psk_exact(es, int(n[:-3]))
    if n.endswith("pam"):
        M = int(n[:-3])
        return 2 * (1 - 1 / M) * qfunc(np.sqrt(6 / (M * M - 1) * _lin(es)))
    if n.endswith("qam") and n[:-3].isdigit():
        M = int(n[:-3])
        p = 2 * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * _lin(es) / (M - 1)))
        return 1 - (1 - p) ** 2
    return np.minimum(nn_approx_ser(constellation(n), es), 1.0)


def ber_theory(name, ebn0_db):
    """Gray-labelled bit error rate (Pb ~ Ps/k beyond 4 points; exact for BPSK/QPSK)."""
    n = name.lower()
    c_k = {"bpsk": 1, "qpsk": 2, "8psk": 3, "16psk": 4, "4pam": 2, "8pam": 3, "16qam": 4,
           "32cross": 5, "64qam": 6, "256qam": 8, "16apsk": 4, "32apsk": 5}[n]
    eb = np.atleast_1d(np.asarray(ebn0_db, float))
    if n in ("bpsk", "qpsk"):
        return ber_bpsk(eb)
    es = eb + 10 * np.log10(c_k)
    if n.endswith("qam") and n[:-3].isdigit():
        return ber_mqam_gray(es, int(n[:-3]))
    return ser_theory(n, es) / c_k


def ebn0_required(name, target=1e-5, fn=None):
    """Eb/N0 (dB) at which the Gray BER of ``name`` (or ``fn(ebn0)``) reaches ``target``.
    Coarse 1 dB scan, then a fine log-linear search inside the bracketing decibel."""
    f = fn or (lambda e: ber_theory(name, e))
    lt = np.log10(target)
    x = np.arange(-5.0, 61.0, 1.0)
    y = np.log10(np.maximum(f(x), 1e-300))
    i = np.where(y <= lt)[0]
    if len(i) == 0:
        return float("nan")
    i = i[0]
    if i == 0:
        return float(x[0])
    xf = np.linspace(x[i - 1], x[i], 41)
    yf = np.log10(np.maximum(f(xf), 1e-300))
    j = np.where(yf <= lt)[0][0]
    if j == 0:
        return float(xf[0])
    return float(np.interp(lt, [yf[j], yf[j - 1]], [xf[j], xf[j - 1]]))


def ber_fading_mrc(fn_awgn, ebn0_db, L=1):
    """Average a BER-vs-Eb/N0 function over flat Rayleigh fading with L-branch MRC: the
    combined SNR is Gamma(L)-distributed with mean L times the per-branch Eb/N0."""
    from scipy.special import gammaln
    x = np.geomspace(1e-7, 60, 1500)
    w = np.exp((L - 1) * np.log(x) - x - gammaln(L))
    eb = np.atleast_1d(np.asarray(ebn0_db, float))
    out = []
    for e in eb:
        g = x * 10 ** (e / 10)
        out.append(np.trapezoid(w * fn_awgn(10 * np.log10(g)), x))
    return np.array(out)


def shannon_ebn0_db(eta):
    """Minimum Eb/N0 (dB) for reliable communication at spectral efficiency eta (b/s/Hz)."""
    eta = np.asarray(eta, float)
    return 10 * np.log10((2 ** eta - 1) / eta)


# ---- noncoherent and fading closed forms (eqs. of Sections 9.8 and 9.11) ------------------
def ber_dbpsk(ebn0_db):
    return 0.5 * np.exp(-_lin(ebn0_db))


def ber_dqpsk(ebn0_db):
    """Gray DQPSK (Proakis): Q1(a, b) - 0.5 I0(ab) exp(-(a^2+b^2)/2)."""
    from scipy.stats import ncx2
    from scipy.special import i0e
    g = _lin(ebn0_db)
    a = np.sqrt(2 * g * (1 - 1 / np.sqrt(2)))
    b = np.sqrt(2 * g * (1 + 1 / np.sqrt(2)))
    return ncx2.sf(b ** 2, 2, a ** 2) - 0.5 * i0e(a * b) * np.exp(-(a - b) ** 2 / 2)


def ber_rayleigh_mrc(ebn0_db, L=1):
    """Coherent BPSK in flat Rayleigh fading with L-branch MRC (Eb/N0 per branch)."""
    g = _lin(ebn0_db)
    mu = np.sqrt(g / (1 + g))
    return ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))


def ber_dbpsk_rayleigh(ebn0_db):
    return 1 / (2 * (1 + _lin(ebn0_db)))


def ber_ncfsk_rayleigh(ebn0_db):
    return 1 / (2 + _lin(ebn0_db))


def ber_qam_rayleigh(ebn0_db, M):
    """Gray square M-QAM in Rayleigh fading: nearest-neighbour expression averaged exactly."""
    k = np.log2(M)
    a = 4 / k * (1 - 1 / np.sqrt(M))
    c = 3 * k / (M - 1)
    g = _lin(ebn0_db)
    return a * 0.5 * (1 - np.sqrt(c * g / 2 / (1 + c * g / 2)))
