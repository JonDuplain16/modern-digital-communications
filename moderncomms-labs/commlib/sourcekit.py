"""sourcekit: quantisers, rate-distortion, LZ77 and block motion search.

Companion helpers for Chapters 13 (rate-distortion) and 16 (source coding), used by the
interactive labs 19 and 24. They complement ``commlib.sourcecoding`` (Huffman, arithmetic
coding, LPC, toy JPEG, psychoacoustics) with the pieces that module does not have:

    from commlib import sourcekit as sk
    sk.lloyd_max(16)                      # optimal 4-bit quantiser of N(0, 1): levels, mse, entropy
    sk.ecsq(0.5)                          # entropy-coded uniform quantiser: (rate, mse)
    sk.reverse_waterfill(var, D)          # rate-distortion of independent Gaussian components
    sk.lz77_parse("abcabcabc", 64, 18)    # greedy LZ77 tokens
    sk.block_motion(cur, ref, 16, 7)      # full-search block matching (vectorised)

Everything is for unit-variance sources unless stated; rates in bits per sample.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn
from scipy.stats import norm

__all__ = ["source_pdf", "lloyd_max", "lloyd_iterations", "uniform_quantizer", "quantize",
           "ecsq", "gaussian_rd_snr_db", "reverse_waterfill", "lz77_parse", "lz77_bits",
           "block_motion", "dct8", "idct8", "zigzag_indices"]

_GRID = np.linspace(-10, 10, 20001)


def source_pdf(kind="gaussian", x=None):
    """Unit-variance pdf on a grid: 'gaussian', 'laplacian' or 'uniform'. Returns (x, f)."""
    x = _GRID if x is None else np.asarray(x, float)
    if kind == "gaussian":
        f = norm.pdf(x)
    elif kind == "laplacian":
        b = 1 / np.sqrt(2)
        f = np.exp(-np.abs(x) / b) / (2 * b)
    elif kind == "uniform":
        a = np.sqrt(3)
        f = np.where(np.abs(x) <= a, 1 / (2 * a), 0.0)
    else:
        raise ValueError(kind)
    return x, f


def _cells(levels):
    levels = np.sort(np.asarray(levels, float))
    return levels, (levels[1:] + levels[:-1]) / 2


def quantize(x, levels):
    """Nearest-level quantisation of x (levels need not be uniform). Returns (q, index)."""
    lev, th = _cells(levels)
    idx = np.searchsorted(th, x)
    return lev[idx], idx


def _stats(levels, kind):
    x, f = source_pdf(kind)
    dx = x[1] - x[0]
    lev, th = _cells(levels)
    idx = np.searchsorted(th, x)
    mse = float(np.sum((x - lev[idx]) ** 2 * f) * dx)
    P = np.bincount(idx, f, len(lev)) * dx
    P = P[P > 0] / P.sum()
    return mse, float(-np.sum(P * np.log2(P)))


def lloyd_iterations(L, n_iter, kind="gaussian", init=None):
    """Run n_iter Lloyd iterations (centroid / midpoint) for an L-level quantiser, starting
    from the best uniform quantiser (or ``init``). Returns the list of level arrays, one per
    iteration including the start (length n_iter + 1)."""
    x, f = source_pdf(kind)
    c = uniform_quantizer(L, kind)[0] if init is None else np.sort(np.asarray(init, float))
    hist = [c.copy()]
    for _ in range(n_iter):
        th = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(th, x)
        num = np.bincount(idx, x * f, L)
        den = np.bincount(idx, f, L)
        c = np.where(den > 1e-300, num / np.maximum(den, 1e-300), c)
        hist.append(c.copy())
    return hist


@lru_cache(maxsize=64)
def lloyd_max(L, kind="gaussian", n_iter=400):
    """Lloyd-Max (minimum-MSE fixed-rate) quantiser with L levels for a unit-variance source.
    Returns (levels, mse, output entropy in bits)."""
    c = lloyd_iterations(L, n_iter, kind)[-1]
    mse, H = _stats(c, kind)
    return c, mse, H


@lru_cache(maxsize=64)
def uniform_quantizer(L, kind="gaussian"):
    """Best uniform (midrise for even L) quantiser with L levels: the step minimising MSE.
    Returns (levels, mse, output entropy, step)."""
    from scipy.optimize import minimize_scalar

    def lev(step):
        return (np.arange(L) - (L - 1) / 2) * step

    hi = 2 * np.sqrt(3) / L * 2.5 + 2.0 / L
    res = minimize_scalar(lambda s: _stats(lev(s), kind)[0], bounds=(1e-3, max(hi, 0.05) * 2),
                          method="bounded", options=dict(xatol=1e-5))
    c = lev(res.x)
    mse, H = _stats(c, kind)
    return c, mse, H, float(res.x)


def ecsq(step, kind="gaussian"):
    """Entropy-coded uniform midtread quantiser (infinitely many levels, reconstruction at the
    cell centroid) of a unit-variance source. Returns (rate = output entropy, mse)."""
    x, f = source_pdf(kind)
    dx = x[1] - x[0]
    k = np.round(x / step).astype(int)
    k -= k.min()
    P = np.bincount(k, f) * dx
    m1 = np.bincount(k, x * f) * dx
    cen = np.where(P > 0, m1 / np.maximum(P, 1e-300), 0)
    mse = float(np.sum((x - cen[k]) ** 2 * f) * dx)
    Pn = P[P > 1e-15]
    Pn = Pn / Pn.sum()
    return float(-np.sum(Pn * np.log2(Pn))), mse


def gaussian_rd_snr_db(R):
    """Rate-distortion bound for a Gaussian source as an SNR: 10 log10(2^(2R)) = 6.02 R dB."""
    return 20 * np.log10(2) * np.asarray(R, float)


def reverse_waterfill(variances, D_total):
    """Reverse water-filling for independent Gaussian components (Chapter 13).

    Each component gets distortion D_i = min(theta, var_i) with sum D_i = D_total.
    Returns (D_i, R_i in bits, theta)."""
    v = np.asarray(variances, float)
    D_total = float(np.clip(D_total, 1e-12, v.sum()))
    lo, hi = 0.0, v.max()
    for _ in range(200):
        th = (lo + hi) / 2
        if np.minimum(th, v).sum() > D_total:
            hi = th
        else:
            lo = th
    th = (lo + hi) / 2
    D = np.minimum(th, v)
    R = 0.5 * np.log2(v / D)
    return D, R, th


# ============================================================================ LZ77
def lz77_parse(text, window=4096, max_len=18, min_len=3):
    """Greedy LZ77 parse (overlapping matches allowed, as in DEFLATE).

    Returns a list of tokens (pos, length, distance): distance 0 means a literal (length 1),
    otherwise ``length`` characters are copied from ``distance`` characters back."""
    toks = []
    i, n = 0, len(text)
    while i < n:
        lo = max(0, i - window)
        best_len, best_d = 0, 0
        L = min_len
        while i + L <= n and L <= max_len:
            j = text.rfind(text[i:i + L], lo, i + L - 1)
            if j < 0 or j >= i:
                break
            best_len, best_d = L, i - j
            L += 1
        if best_len >= min_len:
            toks.append((i, best_len, best_d))
            i += best_len
        else:
            toks.append((i, 1, 0))
            i += 1
    return toks


def lz77_bits(tokens, window=4096, max_len=18, min_len=3, literal_bits=8):
    """Bit cost of a token list with fixed-length fields: a 1-bit flag, then either a literal
    (literal_bits) or distance (ceil log2 window) + length (ceil log2 of the length range)."""
    dbits = int(np.ceil(np.log2(max(window, 2))))
    lbits = int(np.ceil(np.log2(max(max_len - min_len + 1, 2))))
    return int(sum(1 + (literal_bits if d == 0 else dbits + lbits) for _, _, d in tokens))


# ============================================================================ video
def block_motion(cur, ref, block=16, search=7):
    """Full-search block matching (sum of absolute differences), vectorised over blocks.

    cur, ref: 2-D frames with sides divisible by ``block``. Each block of ``cur`` is matched
    against ``ref`` displaced by (dy, dx) in [-search, search]^2; candidate blocks that fall
    outside the frame are not allowed. Returns (mv (nby, nbx, 2) as (dy, dx), prediction,
    SAD evaluations)."""
    cur = np.asarray(cur, float)
    ref = np.asarray(ref, float)
    H, W = cur.shape
    nby, nbx = H // block, W // block
    S = search
    pad = np.pad(ref, S, mode="constant", constant_values=np.nan)
    best = np.full((nby, nbx), np.inf)
    mv = np.zeros((nby, nbx, 2), int)
    cands = sorted(((dy, dx) for dy in range(-S, S + 1) for dx in range(-S, S + 1)),
                   key=lambda d: d[0] ** 2 + d[1] ** 2)      # ties go to the shortest vector
    for dy, dx in cands:
        sh = pad[S + dy:S + dy + H, S + dx:S + dx + W]
        sad = np.abs(cur - sh).reshape(nby, block, nbx, block).sum(axis=(1, 3))
        sad = np.where(np.isnan(sad), np.inf, sad)
        better = sad < best - 1e-9
        best = np.where(better, sad, best)
        mv[better] = (dy, dx)
    pred = np.empty_like(cur)
    for by in range(nby):
        for bx in range(nbx):
            dy, dx = mv[by, bx]
            y, x = by * block, bx * block
            pred[y:y + block, x:x + block] = ref[y + dy:y + dy + block, x + dx:x + dx + block]
    return mv, pred, nby * nbx * (2 * S + 1) ** 2


# ============================================================================ 8x8 DCT
def dct8(block):
    """Orthonormal 2-D DCT-II of a block (JPEG's transform, without the level shift)."""
    return dctn(np.asarray(block, float), norm="ortho")


def idct8(coef):
    return idctn(np.asarray(coef, float), norm="ortho")


def zigzag_indices(n=8):
    """(row, col) pairs of an n x n block in JPEG zig-zag order."""
    return sorted(((i, j) for i in range(n) for j in range(n)),
                  key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else t[1]))
