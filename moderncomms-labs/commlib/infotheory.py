"""infotheory: entropy, source-code bounds, channel capacity and resource allocation.

The companion of Chapter 13 (*Information Theory*). These are clean copies of the helpers
that compute the chapter's figures (book/figscripts/ch13_figs.py), so a lab that calls them
reproduces the book's numbers.

    from commlib import infotheory as it
    it.entropy([0.5, 0.25, 0.125, 0.125])          # 1.75 bits
    it.blahut_arimoto(W)                            # capacity of any DMC
    it.qam_cm(16, snr)                              # constrained-input capacity of 16-QAM
    it.waterfill(noise_to_gain, P_total)            # power allocation over parallel channels

All logarithms are base 2 (bits) unless a name says otherwise. SNRs are linear unless
the argument name ends in ``_db``.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np
from scipy.special import exp1
from scipy.stats import norm
from scipy.optimize import brentq

__all__ = ["LOG2E", "hb", "entropy", "markov_stationary", "markov_entropy_rate", "ngram_entropy",
           "huffman_code", "huffman_lengths", "kraft_sum", "dmc_mutual_info", "blahut_arimoto",
           "bsc_capacity", "bec_capacity", "z_capacity", "awgn_capacity", "mi_pam", "bicm_pam",
           "mi_2d_mc", "qam_cm", "qam_bicm", "biawgn_capacity", "shannon_ebn0_db",
           "normal_approx_awgn", "fbl_snr_penalty_db", "waterfill", "gap_bit_loading",
           "rayleigh_ergodic_capacity", "rayleigh_outage_capacity", "mimo_ergodic_capacity"]

LOG2E = np.log2(np.e)


# ============================================================================ entropy
def hb(p):
    """Binary entropy function h(p) in bits (vectorised, h(0) = h(1) = 0)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1)
    q = np.clip(1 - p, 1e-300, 1)
    return -(p * np.log2(p) + q * np.log2(q))


def entropy(p, axis=-1):
    """Entropy H(p) = -sum p log2 p of a probability vector (zeros are ignored).
    Unnormalised counts are normalised first."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=axis)


def markov_stationary(P):
    """Stationary distribution pi of a row-stochastic transition matrix (pi P = pi)."""
    P = np.asarray(P, dtype=float)
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def markov_entropy_rate(P):
    """Entropy rate sum_i pi_i H(P_i,:) of a stationary Markov source (bits/symbol)."""
    P = np.asarray(P, dtype=float)
    return float(np.sum(markov_stationary(P) * entropy(P, axis=1)))


def ngram_entropy(seq, order):
    """Empirical conditional entropy H(X_n | previous `order` symbols) of a sequence
    (string or list), in bits per symbol. order=0 gives the single-symbol entropy.
    (Plug-in estimate: it is biased low when the number of contexts approaches the
    length of the data.)"""
    if order == 0:
        return float(entropy(np.array(list(Counter(seq).values()))))
    if isinstance(seq, str):
        grams = [seq[i - order:i + 1] for i in range(order, len(seq))]
    else:
        grams = [tuple(seq[i - order:i + 1]) for i in range(order, len(seq))]
    joint = Counter(grams)
    ctx = Counter(g[:-1] for g in grams)
    n = len(grams)
    return float(-sum(v / n * np.log2(v / ctx[k[:-1]]) for k, v in joint.items()))


# ============================================================================ source codes
def huffman_code(probs):
    """Huffman code for a dict symbol -> probability (or count).

    Returns dict symbol -> codeword string of '0'/'1'. Ties are broken by insertion order,
    and merged nodes are placed after equal-probability leaves (minimum-variance code)."""
    items = [(float(p), i, s) for i, (s, p) in enumerate(probs.items()) if p > 0]
    if len(items) == 1:
        return {items[0][2]: "0"}
    heap = [(p, i, (s,)) for p, i, s in items]
    heapq.heapify(heap)
    code = {s: "" for _, _, s in items}
    cnt = len(heap)
    while len(heap) > 1:
        p1, _, g1 = heapq.heappop(heap)
        p2, _, g2 = heapq.heappop(heap)
        for s in g1:
            code[s] = "0" + code[s]
        for s in g2:
            code[s] = "1" + code[s]
        cnt += 1
        heapq.heappush(heap, (p1 + p2, cnt, g1 + g2))
    return code


def huffman_lengths(freqs):
    """Codeword length per symbol of a Huffman code for a dict symbol -> count/probability."""
    return {s: len(c) for s, c in huffman_code(freqs).items()}


def kraft_sum(lengths, D=2):
    """Kraft sum  sum D^-l  (<= 1 for any uniquely decodable code)."""
    return float(np.sum(float(D) ** -np.asarray(list(lengths), dtype=float)))


# ============================================================================ DMCs
def dmc_mutual_info(px, W):
    """I(X;Y) in bits for input distribution px and channel matrix W[x, y] = P(y|x)."""
    px = np.asarray(px, float)
    W = np.asarray(W, float)
    return float(entropy(px @ W) - np.sum(px * entropy(W, axis=1)))


def blahut_arimoto(W, iters=200, tol=1e-12):
    """Capacity of the DMC W[x, y] = P(y|x) by the Blahut-Arimoto algorithm.

    Returns (C, p_opt, lower, upper): the capacity in bits/use, the optimising input
    distribution and the histories of the lower bound I(p_t) and upper bound max_x D(W_x || q)."""
    W = np.asarray(W, dtype=float)
    p = np.full(W.shape[0], 1 / W.shape[0])
    lo, up = [], []
    for _ in range(iters):
        q = p @ W
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.nansum(np.where(W > 0, W * np.log2(W / q[None, :]), 0), axis=1)
        lo.append(float(np.sum(p * D)))
        up.append(float(D.max()))
        if up[-1] - lo[-1] < tol:
            break
        p = p * 2.0 ** D
        p /= p.sum()
    return lo[-1], p, np.array(lo), np.array(up)


def bsc_capacity(p):
    return 1 - hb(p)


def bec_capacity(e):
    return 1 - np.asarray(e, float)


def z_capacity(p):
    """Capacity of the Z-channel (a one is received as a zero with probability p)."""
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(1 + (1 - p) * p ** (p / (1 - p)))


# ============================================================================ Gaussian channels
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(120)


def awgn_capacity(snr):
    """Shannon capacity log2(1 + SNR) in bits per complex channel use (linear SNR)."""
    return np.log2(1 + np.asarray(snr, float))


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def mi_pam(points, snr, probs=None):
    """I(X;Y) in bits for the real channel y = x + n, n ~ N(0, 1/snr), with the points
    normalised to unit energy (Gauss-Hermite quadrature, accurate to ~1e-6 bits).

    For a complex constellation built from two PAMs (square QAM), the complex-channel
    mutual information at Es/N0 = snr is 2 * mi_pam(levels, snr) (each real dimension has
    half the energy and half the noise)."""
    x = np.asarray(points, float)
    p = np.full(len(x), 1 / len(x)) if probs is None else np.asarray(probs, float)
    x = x / np.sqrt(np.sum(p * x ** 2))
    s2 = 1.0 / snr
    n = np.sqrt(2 * s2) * _GH_T
    w = _GH_W / np.sqrt(np.pi)
    d = x[:, None] - x[None, :]
    e = -((d[:, :, None] + n[None, None, :]) ** 2 - n[None, None, :] ** 2) / (2 * s2)
    e = e + np.log(p)[None, :, None]
    return float(-np.sum(p[:, None] * w[None, :] * _lse(e, 1)) * LOG2E)


def bicm_pam(con, snr):
    """BICM capacity (sum of bit-wise mutual informations) of a labelled real constellation
    (a commlib Constellation whose points are real) with uniform inputs, real AWGN at SNR snr."""
    x = con.points.real / np.sqrt(np.mean(con.points.real ** 2))
    bits = con.bit_matrix[con.labels]
    s2 = 1.0 / snr
    n = np.sqrt(2 * s2) * _GH_T
    w = _GH_W / np.sqrt(np.pi)
    d = x[:, None] - x[None, :]
    e = -((d[:, :, None] + n[None, None, :]) ** 2 - n[None, None, :] ** 2) / (2 * s2)
    tot = _lse(e, 1)
    I = 0.0
    for b in range(con.k):
        same = bits[:, b][:, None] == bits[:, b][None, :]
        es = np.where(same[:, :, None], e, -np.inf)
        I += 1 - np.mean(np.sum(w[None, :] * (tot - _lse(es, 1)), axis=1)) * LOG2E
    return float(I)


def mi_2d_mc(con, snr, n=60000, rng=None):
    """Constrained-input (coded-modulation) capacity of any complex constellation at
    Es/N0 = snr (linear), by Monte Carlo with n samples. Unit-energy points assumed."""
    rng = np.random.default_rng(1) if rng is None else rng
    idx = rng.integers(0, con.M, n)
    x = con.points[idx]
    N0 = 1 / snr
    z = np.sqrt(N0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    out = 0.0
    for i in range(0, n, 20000):                       # chunk to bound memory for large M
        xs, zs = x[i:i + 20000], z[i:i + 20000]
        e = -(np.abs(xs[:, None] + zs[:, None] - con.points[None, :]) ** 2 - np.abs(zs[:, None]) ** 2) / N0
        out += np.sum(_lse(e, 1))
    return float(np.log2(con.M) - out / n * LOG2E)


def qam_cm(M, snr):
    """Coded-modulation capacity (bits/symbol) of square M-QAM at Es/N0 = snr (linear)."""
    from .modulation import get_constellation
    L = int(round(np.sqrt(M)))
    return 2 * mi_pam(get_constellation(f"{L}pam").points.real, snr)


def qam_bicm(M, snr, natural=False):
    """BICM capacity of square M-QAM with Gray labels (or natural binary labels per axis)."""
    from .modulation import get_constellation, Constellation
    L = int(round(np.sqrt(M)))
    c = get_constellation(f"{L}pam")
    if natural:
        c = Constellation(np.sort(c.points.real), np.arange(L), "nat")
    return 2 * bicm_pam(c, snr)


def biawgn_capacity(esn0):
    """Capacity of binary-input (BPSK) AWGN at Es/N0 = esn0 (linear), bits per use."""
    return mi_pam([-1, 1], 2 * np.asarray(esn0, float)) if np.ndim(esn0) == 0 else \
        np.array([mi_pam([-1, 1], 2 * e) for e in np.ravel(esn0)])


def shannon_ebn0_db(eta):
    """Minimum Eb/N0 (dB) for spectral efficiency eta (b/s/Hz): (2^eta - 1)/eta."""
    eta = np.asarray(eta, float)
    return 10 * np.log10((2 ** eta - 1) / eta)


def normal_approx_awgn(n, snr, eps):
    """Polyanskiy-Poor-Verdu normal approximation of the largest rate (bits per REAL channel
    use) at block length n (real uses), linear SNR snr and block error probability eps."""
    C = 0.5 * np.log2(1 + snr)
    V = snr * (snr + 2) / (2 * (snr + 1) ** 2) * LOG2E ** 2
    return C - np.sqrt(V / n) * norm.isf(eps) + 0.5 * np.log2(n) / n


def fbl_snr_penalty_db(n, R, eps):
    """Extra SNR (dB) the normal approximation needs over Shannon at rate R (bits per real
    use), block length n and error probability eps."""
    Psh = 2 ** (2 * R) - 1
    P = brentq(lambda P: normal_approx_awgn(n, P, eps) - R, Psh, 1e6)
    return 10 * np.log10(P / Psh)


# ============================================================================ allocation
def waterfill(inv_gain, P_total):
    """Water-filling over parallel channels with noise-to-gain ratios inv_gain = N_k/|H_k|^2.

    Returns (powers, mu): P_k = max(mu - inv_gain_k, 0) with sum P_k = P_total."""
    inv_gain = np.asarray(inv_gain, float)
    s = np.sort(inv_gain)
    for k in range(len(s), 0, -1):
        mu = (P_total + s[:k].sum()) / k
        if mu > s[k - 1]:
            break
    return np.maximum(mu - inv_gain, 0), mu


def gap_bit_loading(snr, gap_db, max_bits=15):
    """Integer bits per subchannel with the SNR-gap approximation: floor(log2(1 + SNR/Gamma)),
    clipped to [0, max_bits]. snr is linear per subchannel."""
    G = 10 ** (gap_db / 10)
    return np.clip(np.floor(np.log2(1 + np.asarray(snr, float) / G)), 0, max_bits).astype(int)


# ============================================================================ fading / MIMO
def rayleigh_ergodic_capacity(snr):
    """Ergodic capacity E[log2(1 + snr |h|^2)] of Rayleigh fading with CSI at the receiver."""
    g = np.asarray(snr, float)
    return LOG2E * np.exp(1 / g) * exp1(1 / g)


def rayleigh_outage_capacity(snr, eps):
    """Largest rate whose outage probability over Rayleigh fading is eps."""
    return np.log2(1 - np.asarray(snr, float) * np.log(1 - eps))


def mimo_ergodic_capacity(nt, nr, snr, trials=2000, rng=None):
    """Ergodic capacity (b/s/Hz) of i.i.d. Rayleigh nt x nr MIMO, equal power, CSIR only."""
    rng = np.random.default_rng(11) if rng is None else rng
    H = (rng.standard_normal((trials, nr, nt)) + 1j * rng.standard_normal((trials, nr, nt))) / np.sqrt(2)
    ev = np.clip(np.linalg.eigvalsh(np.einsum("tij,tkj->tik", H, H.conj())), 0, None)
    snr = np.atleast_1d(np.asarray(snr, float))
    C = np.array([np.mean(np.sum(np.log2(1 + s / nt * ev), axis=1)) for s in snr])
    return C if C.size > 1 else float(C[0])
