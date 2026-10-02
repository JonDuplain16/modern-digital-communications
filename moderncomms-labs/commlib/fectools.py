"""fectools: small, fast helpers for the interactive coding labs (Labs 08, 09, 20 and 23).

Companion of Chapters 13-15. It adds what the interactive experiments need on top of
``commlib.coding`` (ConvCode, LDPCCode, PolarCode), ``commlib.gf``, ``commlib.blockcodes``
and ``commlib.turbo``, without changing any of them:

* convolutional codes: a fully recorded hard-decision Viterbi (for step-by-step pictures),
  a batched Viterbi with a sliding decision (traceback) depth, the distance spectrum,
  soft and hard union bounds, the free distance of a punctured code, a soft-decision quantiser;
* LDPC: a batched flooding belief-propagation decoder (sum-product, min-sum, normalised and
  offset min-sum) that can record every iteration, random regular constructions, girth and
  4-cycle counts;
* polar: Bhattacharyya parameters on the BEC;
* finite-length limits: BI-AWGN capacity/dispersion and the normal approximation;
* turbo/EXIT: a fast J-function, EXIT transfer curves, S-random interleavers and the
  weight-2-input codeword enumeration behind the error floor;
* CRC: vectorised GF(2) remainders of many error patterns at once.

LLR convention as everywhere in commlib: L = log P(0)/P(1), BPSK 0 -> +1, 1 -> -1.
"""
from __future__ import annotations

import heapq
from math import comb

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc
from scipy.stats import norm

__all__ = ["qfunc", "bpsk_sigma", "bpsk_llr", "viterbi_trace", "viterbi_window",
           "distance_spectrum", "conv_union_bound", "punctured_dfree", "quantize_soft",
           "TannerBP", "random_regular_H", "count_4cycles", "tanner_girth",
           "polar_bec_z", "biawgn_cv", "biawgn_limit_db", "na_ebn0_db", "J_fast", "Jinv_fast",
           "exit_curve", "s_random_interleaver", "weight2_codeword_weights", "gf2_mod_many"]


# ============================================================================ channel
def qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def bpsk_sigma(ebn0_db, R):
    """Noise standard deviation per real dimension for BPSK at Eb/N0 (dB) and code rate R."""
    return np.sqrt(1 / (2 * R * 10 ** (np.asarray(ebn0_db, float) / 10)))


def bpsk_llr(bits, ebn0_db, R, rng):
    """Channel LLRs 2y/sigma^2 for BPSK-modulated ``bits`` over AWGN."""
    s = bpsk_sigma(ebn0_db, R)
    y = (1 - 2.0 * np.asarray(bits)) + s * rng.standard_normal(np.shape(bits))
    return 2 * y / s ** 2


# ============================================================================ convolutional codes
def _prev_tables(cc):
    K, S = cc.K, cc.S
    ns = np.arange(S)
    b_of_ns = ns >> (K - 2)
    low = (ns & ((1 << (K - 2)) - 1)) << 1
    prev = np.stack([low, low | 1], axis=1)
    sgn = 1 - 2 * cc.outputs.astype(float)
    bo = sgn[prev, b_of_ns[:, None]]                       # (S, 2, n_out)
    return prev, b_of_ns, bo


def viterbi_trace(cc, rx):
    """Hard-decision Viterbi on received bits ``rx`` (T, n_out), fully recorded.

    Returns dict with pm (T+1, S) path metrics (np.inf = unreachable), surv (T+1, S) the
    surviving predecessor of each state (-1 at t = 0), path (T+1,) the ML state sequence
    ending in state 0, and bits (T,) the decoded inputs (tail included)."""
    rx = np.asarray(rx, int).reshape(-1, cc.n_out)
    T, S = len(rx), cc.S
    pm = np.full((T + 1, S), np.inf)
    pm[0, 0] = 0
    surv = np.full((T + 1, S), -1)
    for t in range(T):
        for s in range(S):
            if not np.isfinite(pm[t, s]):
                continue
            for b in (0, 1):
                s2 = cc.next_state[s, b]
                m = pm[t, s] + int(np.sum(cc.outputs[s, b] != rx[t]))
                if m < pm[t + 1, s2]:
                    pm[t + 1, s2] = m
                    surv[t + 1, s2] = s
    path = [0]
    for t in range(T, 0, -1):
        path.append(surv[t, path[-1]])
    path = np.array(path[::-1])
    bits = np.array([path[t + 1] >> (cc.K - 2) for t in range(T)], dtype=int)
    return dict(pm=pm, surv=surv, path=path, bits=bits)


def viterbi_window(cc, llr, depth=None):
    """Batched Viterbi decoding of terminated frames with a sliding decision depth.

    llr: (F, T*n_out) metrics (LLRs, +/-1 hard values or quantised levels; 0 = erased).
    depth=None: full traceback from state 0 (as ConvCode.decode_batch). Otherwise bit t is
    decided by tracing back ``depth`` steps from the best state at time t + depth, as a
    hardware decoder with a finite survivor memory does. Returns (F, T) bits (tail included).
    """
    r = np.asarray(llr, float)
    F = r.shape[0]
    r = r.reshape(F, -1, cc.n_out)
    T = r.shape[1]
    S, K = cc.S, cc.K
    prev, _, bo = _prev_tables(cc)
    pm = np.full((F, S), 1e9)
    pm[:, 0] = 0.0
    dec = np.zeros((T, F, S), dtype=np.int8)
    best = np.zeros((T, F), dtype=np.int64)
    for t in range(T):
        bm = -np.einsum("sjn,fn->fsj", bo, r[:, t])
        cand = pm[:, prev] + bm
        ch = np.argmin(cand, axis=2)
        pm = np.take_along_axis(cand, ch[:, :, None], axis=2)[:, :, 0]
        pm -= pm.min(axis=1, keepdims=True)
        dec[t] = ch
        best[t] = np.argmin(pm, axis=1)
    fr = np.arange(F)
    bits = np.empty((F, T), dtype=np.int8)
    st = np.zeros(F, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        bits[:, t] = st >> (K - 2)
        st = prev[st, dec[t, fr, st]]
    if depth is None or depth >= T:
        return bits
    D = int(depth)
    n = T - D
    tau = np.arange(n) + D                                   # decision made at time t + D
    st = best[tau].T.copy()                                  # (F, n)
    fi = fr[:, None]
    for k in range(D):
        tt = tau - k
        st = prev[st, dec[tt[None, :], fi, st]]
    bits[:, :n] = st >> (K - 2)
    return bits


def distance_spectrum(cc, dmax=20, steps=300):
    """Error events of a feed-forward code: (d, A_d, B_d) with A_d the number of paths of
    output weight d leaving and first re-entering state 0, B_d their total input weight."""
    S = cc.S
    cnt = np.zeros((S, dmax + 1))
    iw = np.zeros((S, dmax + 1))
    A = np.zeros(dmax + 1)
    B = np.zeros(dmax + 1)
    w = int(cc.outputs[0, 1].sum())
    s1 = cc.next_state[0, 1]
    cnt[s1, w] = 1
    iw[s1, w] = 1
    for _ in range(steps):
        nc = np.zeros_like(cnt)
        ni = np.zeros_like(iw)
        for s in range(1, S):
            if not cnt[s].any():
                continue
            for b in (0, 1):
                w = int(cc.outputs[s, b].sum())
                s2 = cc.next_state[s, b]
                if w > dmax:
                    continue
                sc = np.zeros(dmax + 1)
                si = np.zeros(dmax + 1)
                sc[w:] = cnt[s, :dmax + 1 - w]
                si[w:] = iw[s, :dmax + 1 - w] + b * cnt[s, :dmax + 1 - w]
                if s2 == 0:
                    A += sc
                    B += si
                else:
                    nc[s2] += sc
                    ni[s2] += si
        cnt, iw = nc, ni
        if not cnt.any():
            break
    d = np.nonzero(A)[0]
    return d, A[d], B[d]


def _pd_hard(d, p):
    """Pairwise error probability of a weight-d event on a BSC(p) (ties broken at random)."""
    p = np.asarray(p, float)
    out = np.zeros_like(p)
    for e in range(d // 2 + 1, d + 1):
        out += comb(d, e) * p ** e * (1 - p) ** (d - e)
    if d % 2 == 0:
        out += 0.5 * comb(d, d // 2) * p ** (d // 2) * (1 - p) ** (d // 2)
    return out


def conv_union_bound(spectrum, ebn0_db, rate, hard=False):
    """Union bound on the bit error rate: sum_d B_d P_d, soft (Q(sqrt(2 d R Eb/N0))) or hard
    (BSC with p = Q(sqrt(2 R Eb/N0))). ``spectrum`` = distance_spectrum(...)."""
    d, _, B = spectrum
    e = 10 ** (np.asarray(ebn0_db, float) / 10)
    if hard:
        p = qfunc(np.sqrt(2 * rate * e))
        return sum(Bd * _pd_hard(int(dd), p) for dd, Bd in zip(d, B))
    return sum(Bd * qfunc(np.sqrt(2 * dd * rate * e)) for dd, Bd in zip(d, B))


def punctured_dfree(cc, keep):
    """Free distance of a convolutional code punctured with the periodic keep-mask ``keep``
    (over coded bits, length a multiple of n_out): shortest path search over (state, phase)."""
    keep = np.asarray(keep, bool).reshape(-1, cc.n_out)
    P = len(keep)
    best = np.inf
    for ph0 in range(P):
        s1 = cc.next_state[0, 1]
        w0 = int(np.sum(cc.outputs[0, 1] * keep[ph0]))
        dist = {(s1, (ph0 + 1) % P): w0}
        h = [(w0, s1, (ph0 + 1) % P)]
        while h:
            w, s, ph = heapq.heappop(h)
            if w >= best or dist.get((s, ph), np.inf) < w:
                continue
            if s == 0:
                best = min(best, w)
                continue
            for b in (0, 1):
                s2 = cc.next_state[s, b]
                w2 = w + int(np.sum(cc.outputs[s, b] * keep[ph]))
                k2 = (s2, (ph + 1) % P)
                if w2 < dist.get(k2, np.inf):
                    dist[k2] = w2
                    heapq.heappush(h, (w2, s2, k2[1]))
    return int(best)


def quantize_soft(y, sigma, bits=None):
    """Branch metrics for the Viterbi decoder from BPSK samples y (0 -> +1).

    bits=None: exact LLRs 2y/sigma^2. bits=1: hard decisions +/-1. bits=b>1: a uniform
    mid-rise quantiser with 2^b levels spanning about +/-1.5 (the levels are the metric)."""
    y = np.asarray(y, float)
    if bits is None:
        return 2 * y / sigma ** 2
    if bits <= 1:
        return np.where(y >= 0, 1.0, -1.0)
    lev = 2 ** int(bits)
    step = 3.0 / lev
    return np.clip(np.floor(y / step), -lev // 2, lev // 2 - 1) + 0.5


# ============================================================================ LDPC
class TannerBP:
    """Flooding belief propagation on the Tanner graph of a binary H, batched over frames.

    methods: "spa" (sum-product), "ms" (min-sum), "nms" (normalised min-sum, scale alpha),
    "oms" (offset min-sum, offset beta). decode(..., record=True) also returns the
    a-posteriori LLRs after every iteration and the check-to-variable messages (frame 0).
    """

    def __init__(self, H):
        self.H = (np.asarray(H) % 2).astype(np.int8)
        self.m, self.n = self.H.shape
        ce, ev = np.nonzero(self.H)
        self.edge_c, self.edge_v = ce, ev
        self.E = len(ce)
        dc = np.bincount(ce, minlength=self.m)
        self.dcmax = max(1, int(dc.max()))
        self.slot = np.full((self.m, self.dcmax), -1)
        pos = np.zeros(self.m, dtype=int)
        for e, c in enumerate(ce):
            self.slot[c, pos[c]] = e
            pos[c] += 1
        self.valid = self.slot >= 0
        self.Hi = self.H.astype(np.int64)

    def syndrome(self, c):
        return (np.atleast_2d(c).astype(np.int64) @ self.Hi.T) % 2

    def _check(self, q, method, alpha, beta):
        """q: (B, E) variable-to-check messages -> (B, E) check-to-variable messages."""
        B = q.shape[0]
        sl = np.maximum(self.slot, 0)
        Q = q[:, sl]                                          # (B, m, dcmax)
        V = self.valid[None]
        if method == "spa":
            t = np.where(V, np.tanh(np.clip(Q, -40, 40) / 2), 1.0)
            t = np.where(np.abs(t) < 1e-15, 1e-15, t)
            prod = np.prod(t, axis=2, keepdims=True)
            R = 2 * np.arctanh(np.clip(prod / t, -1 + 1e-15, 1 - 1e-15))
        else:
            mag = np.where(V, np.abs(Q), np.inf)
            sgn = np.where(V, np.where(Q < 0, -1.0, 1.0), 1.0)
            tot = np.prod(sgn, axis=2, keepdims=True)
            i1 = np.argmin(mag, axis=2)
            m1 = np.take_along_axis(mag, i1[:, :, None], 2)
            mag2 = mag.copy()
            np.put_along_axis(mag2, i1[:, :, None], np.inf, 2)
            m2 = mag2.min(axis=2, keepdims=True)
            mm = np.where(np.arange(self.dcmax)[None, None, :] == i1[:, :, None], m2, m1)
            mm = np.where(np.isfinite(mm), mm, 0.0)
            if method == "nms":
                mm = alpha * mm
            elif method == "oms":
                mm = np.maximum(mm - beta, 0.0)
            R = tot * sgn * mm
        r = np.zeros((B, self.E))
        r[:, self.slot[self.valid]] = R[:, self.valid]
        return r

    def decode(self, llr, iters=50, method="spa", alpha=0.8, beta=0.15, record=False,
               early_stop=True):
        """Returns (hard (B, n), iterations used (B,)) or, with record=True,
        (hard, iters, history) where history = dict(L=[(B, n) per iteration],
        r=[(E,) frame-0 check messages per iteration], q=[...])."""
        Lch = np.atleast_2d(np.asarray(llr, float))
        B = Lch.shape[0]
        r = np.zeros((B, self.E))
        L = Lch.copy()
        out = (L < 0).astype(np.int8)
        used = np.full(B, iters)
        done = np.zeros(B, bool)
        hist = dict(L=[Lch.copy()], r=[np.zeros(self.E)], q=[Lch[0, self.edge_v].copy()])
        for it in range(1, iters + 1):
            q = L[:, self.edge_v] - r
            r = self._check(q, method, alpha, beta)
            idx = (np.arange(B)[:, None] * self.n + self.edge_v[None, :]).ravel()
            L = Lch + np.bincount(idx, weights=r.ravel(), minlength=B * self.n).reshape(B, self.n)
            c = (L < 0).astype(np.int8)
            ok = ~self.syndrome(c).any(axis=1)
            new = ok & ~done
            out[new] = c[new]
            used[new] = it
            done |= ok
            out[~done] = c[~done]
            if record:
                hist["L"].append(L.copy())
                hist["r"].append(r[0].copy())
                hist["q"].append(q[0].copy())
            if early_stop and done.all():
                break
        if record:
            return out, used, hist
        return out, used


def random_regular_H(n, dv, dc, rng):
    """A (dv, dc)-regular parity-check matrix by random socket matching (Gallager/MacKay
    style, no cycle avoidance). Rare double edges are dropped."""
    m = n * dv // dc
    sockets = np.repeat(np.arange(n), dv)
    perm = rng.permutation(len(sockets))
    checks = np.repeat(np.arange(m), dc)[:len(sockets)]
    H = np.zeros((m, n), dtype=np.int8)
    H[checks, sockets[perm]] = 1
    return H


def count_4cycles(H):
    """Number of 4-cycles in the Tanner graph: pairs of checks sharing two variables."""
    Hm = np.asarray(H, dtype=np.int64)
    ov = Hm @ Hm.T
    np.fill_diagonal(ov, 0)
    return int(np.sum(ov * (ov - 1) // 2) // 2)


def tanner_girth(H, max_len=20):
    """Girth (length of the shortest cycle) of the Tanner graph of H, or 0 if acyclic up to
    ``max_len``. Breadth-first search from every variable node."""
    H = np.asarray(H) % 2
    m, n = H.shape
    vn = [np.flatnonzero(H[:, j]) for j in range(n)]
    cn = [np.flatnonzero(H[i]) for i in range(m)]
    girth = np.inf
    for root in range(n):
        dist = {("v", root): 0}
        parent = {("v", root): None}
        frontier = [("v", root)]
        d = 0
        while frontier and 2 * d < min(girth, max_len):
            nxt = []
            for node in frontier:
                kind, i = node
                nbrs = [("c", c) for c in vn[i]] if kind == "v" else [("v", v) for v in cn[i]]
                for nb in nbrs:
                    if nb == parent[node]:
                        continue
                    if nb in dist:
                        girth = min(girth, dist[node] + dist[nb] + 1)
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        nxt.append(nb)
            frontier = nxt
            d += 1
    return 0 if not np.isfinite(girth) else int(girth)


# ============================================================================ polar
def polar_bec_z(N, eps):
    """Erasure probabilities (Bhattacharyya parameters) of the N = 2^n synthetic channels of
    a BEC(eps), in the natural index order used by commlib.PolarCode (Z- = 2Z - Z^2, Z+ = Z^2)."""
    z = np.array([float(eps)])
    while len(z) < N:
        z = np.stack([2 * z - z * z, z * z], axis=1).ravel()
    return z


# ============================================================================ finite length
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(100)


def biawgn_cv(sigma):
    """Capacity (bits) and dispersion (bits^2) of the BPSK-input AWGN channel, noise std sigma."""
    y = 1 + np.sqrt(2) * sigma * _GH_T
    i = 1 - np.logaddexp(0, -2 * y / sigma ** 2) / np.log(2)
    w = _GH_W / np.sqrt(np.pi)
    C = np.sum(w * i)
    return C, np.sum(w * (i - C) ** 2)


def biawgn_limit_db(R):
    """Smallest Eb/N0 (dB) at which BPSK-input AWGN supports rate R (infinite length)."""
    s = brentq(lambda s: biawgn_cv(s)[0] - R, 0.05, 20)
    return float(10 * np.log10(1 / (2 * R * s ** 2)))


def na_ebn0_db(n, k, eps):
    """Eb/N0 (dB) at which the normal approximation (Polyanskiy-Poor-Verdu) allows k bits in
    n BPSK channel uses at frame error rate eps."""
    R = k / n
    f = lambda s: n * biawgn_cv(s)[0] - np.sqrt(n * biawgn_cv(s)[1]) * norm.isf(eps) \
        + 0.5 * np.log2(n) - k
    s = brentq(f, 0.05, 20)
    return float(10 * np.log10(1 / (2 * R * s ** 2)))


# ============================================================================ EXIT charts
_JS = np.linspace(0, 60, 3001)
_JT, _JW = np.polynomial.hermite.hermgauss(80)
_LL = _JS[:, None] ** 2 / 2 + np.sqrt(2) * _JS[:, None] * _JT[None, :]
_JV = 1 - np.sum(_JW * np.logaddexp(0, -_LL), axis=1) / np.sqrt(np.pi) / np.log(2)
_JV[0] = 0.0
del _LL


def J_fast(s):
    """I(X; L) of a consistent Gaussian LLR with standard deviation s (same as turbo.J)."""
    return np.interp(s, _JS, _JV)


def Jinv_fast(I):
    return np.interp(np.clip(I, 0, _JV[-1]), _JV, _JS)


def exit_curve(rsc, sigma, IA_grid, K=2000, reps=4, rng=None, maxlog=False):
    """EXIT transfer curve I_E(I_A) of one constituent RSC decoder of a rate-1/3 PCCC
    (as turbo.exit_curve_rsc, but with the fast J-function)."""
    from .turbo import bcjr, mi_llr
    rng = rng or np.random.default_rng(0)
    IA = np.repeat(np.asarray(IA_grid, float), reps)
    B = len(IA)
    u = rng.integers(0, 2, (B, K))
    s, p = rsc.encode(u)
    Ls = 2 * ((1 - 2.0 * s) + sigma * rng.standard_normal(s.shape)) / sigma ** 2
    Lp = 2 * ((1 - 2.0 * p) + sigma * rng.standard_normal(p.shape)) / sigma ** 2
    sA = Jinv_fast(IA)[:, None]
    x = 1.0 - 2.0 * u
    La = sA ** 2 / 2 * x + sA * rng.standard_normal(u.shape)
    La = np.concatenate([La, np.zeros((B, rsc.m))], axis=1)
    L = bcjr(rsc, Ls + La, Lp, maxlog=maxlog)
    Le = (L - Ls - La)[:, :K]
    IE = np.array([mi_llr(Le[b], u[b]) for b in range(B)])
    return np.clip(IE.reshape(-1, reps).mean(axis=1), 0, 1)


# ============================================================================ interleavers / floors
def s_random_interleaver(K, S, rng, tries=10):
    """S-random interleaver (Divsalar-Pollara): positions within S of each other in the input
    land at least S apart. Falls back to the best attempt if the spread is too ambitious."""
    best, best_bad = None, np.inf
    for _ in range(tries):
        pool = rng.permutation(K)
        out = np.empty(K, dtype=int)
        nbad = 0
        for n in range(K):
            recent = out[max(0, n - S):n]
            if len(recent):
                okm = np.all(np.abs(pool[:, None] - recent[None, :]) > S, axis=1)
                j = int(np.argmax(okm)) if okm.any() else 0
                nbad += not okm.any()
            else:
                j = 0
            out[n] = pool[j]
            pool = np.delete(pool, j)
        if nbad == 0:
            return out
        if nbad < best_bad:
            best, best_bad = out, nbad
    return best


def weight2_codeword_weights(tc, dmax_sep=56):
    """Total weights of the turbo codewords produced by weight-2 inputs whose ones are a
    multiple of 7 apart (up to ``dmax_sep``) in either encoder's input order (the inputs
    behind the error floor with the (13, 15) constituent code)."""
    K, pi = tc.K, np.asarray(tc.pi)
    pairs = set()
    for d in range(7, dmax_sep + 1, 7):
        for i in range(K - d):
            pairs.add((i, i + d))
            a, b = pi[i], pi[i + d]
            pairs.add((min(a, b), max(a, b)))
    pairs = np.array(sorted(pairs))
    out = []
    for s0 in range(0, len(pairs), 4000):
        pp = pairs[s0:s0 + 4000]
        u = np.zeros((len(pp), K), np.int8)
        u[np.arange(len(pp)), pp[:, 0]] = 1
        u[np.arange(len(pp)), pp[:, 1]] = 1
        out.append(tc.flatten(tc.encode(u)).sum(axis=1))
    return np.concatenate(out), pairs


# ============================================================================ CRC
def gf2_mod_many(e, g, nbits=64):
    """Remainders of many GF(2) polynomials ``e`` (uint64 array, bit i <-> x^i) modulo g
    (an int including its top term), vectorised. Undetected error patterns give 0."""
    e = np.asarray(e, dtype=np.uint64).copy()
    r = int(g).bit_length() - 1
    gg = np.uint64(g)
    for bit in range(nbits - 1, r - 1, -1):
        hit = ((e >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        e[hit] ^= gg << np.uint64(bit - r)
    return e
