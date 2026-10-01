"""Turbo codes, the BCJR algorithm, EXIT charts and density evolution.

Companion module for Chapter 15 (Turbo, LDPC and Polar Codes) and Lab 23. Import it
explicitly:  ``from commlib import turbo``.

LLR convention (same as commlib.coding): L = log P(bit=0)/P(bit=1); BPSK maps 0 -> +1,
1 -> -1, so for y = x + n with noise variance sigma^2 the channel LLR is 2 y / sigma^2.

Everything is vectorised across a *batch* of frames (first array axis), because the BCJR
recursions are sequential in time and numpy overhead would otherwise dominate.

Contents
--------
RSC                 recursive systematic convolutional code (trellis tables, encoder, termination)
qpp_interleaver     quadratic permutation polynomial interleaver pi(i) = (f1 i + f2 i^2) mod K
bcjr                batched log-MAP / max-log-MAP BCJR decoder for an RSC (APP LLRs of the inputs)
TurboCode           parallel-concatenated code (LTE-style: two 8-state RSCs, rate ~1/3, both
                    encoders terminated), with an iterative decoder that records every iteration
J, Jinv             mutual information of a consistent Gaussian LLR vs its standard deviation
gaussian_apriori    synthetic a-priori LLRs with a prescribed mutual information (EXIT charts)
mi_llr              mutual information estimate from LLRs and known bits (time average)
exit_curve_rsc      EXIT transfer curve of one constituent decoder
de_bec_regular      density evolution of a (dv, dc)-regular LDPC ensemble on the BEC
bec_threshold       BEC threshold of an ensemble given edge-perspective polynomials lambda, rho
de_bec_coupled      density evolution of a spatially coupled (dv, dc, L, w) ensemble on the BEC
"""
from __future__ import annotations

import numpy as np

__all__ = ["RSC", "qpp_interleaver", "bcjr", "TurboCode", "J", "Jinv", "gaussian_apriori",
           "mi_llr", "exit_curve_rsc", "de_bec_regular", "bec_threshold", "de_bec_coupled"]


# ----------------------------------------------------------------------------- RSC trellis
def _octal_taps(g, m):
    """Octal generator -> tap list [g_0 .. g_m] (g_0 multiplies the newest bit)."""
    b = [(g >> (m - i)) & 1 for i in range(m + 1)]
    return np.array(b, dtype=int)


class RSC:
    """Rate-1/2 recursive systematic convolutional code.

    Default (13, 15) octal: feedback 1 + D^2 + D^3, feed-forward 1 + D + D^3, the 8-state
    constituent code of UMTS/LTE turbo codes (3GPP TS 25.212, TS 36.212).
    State = (a_{k-1}, ..., a_{k-m}) packed with a_{k-1} as the most significant bit.
    """

    def __init__(self, fb=0o13, ff=0o15, m=3):
        self.m, self.S = m, 1 << m
        self.fb, self.ff = _octal_taps(fb, m), _octal_taps(ff, m)
        S = self.S
        self.next_state = np.zeros((S, 2), dtype=int)
        self.parity = np.zeros((S, 2), dtype=int)
        self.term_input = np.zeros(S, dtype=int)      # input that drives a_k to 0
        for s in range(S):
            reg = [(s >> (m - 1 - i)) & 1 for i in range(m)]     # a_{k-1} .. a_{k-m}
            fbsum = sum(self.fb[i + 1] * reg[i] for i in range(m)) & 1
            self.term_input[s] = fbsum
            for u in (0, 1):
                a = (u + fbsum) & 1
                p = (self.ff[0] * a + sum(self.ff[i + 1] * reg[i] for i in range(m))) & 1
                ns = (a << (m - 1)) | (s >> 1)
                self.next_state[s, u], self.parity[s, u] = ns, p
        # two predecessors of every state
        self.prev_s = np.zeros((S, 2), dtype=int)
        self.prev_u = np.zeros((S, 2), dtype=int)
        cnt = np.zeros(S, dtype=int)
        for s in range(S):
            for u in (0, 1):
                ns = self.next_state[s, u]
                self.prev_s[ns, cnt[ns]], self.prev_u[ns, cnt[ns]] = s, u
                cnt[ns] += 1
        assert np.all(cnt == 2)

    def encode(self, u, terminate=True):
        """u: (B, K) bits -> (sys, par) each (B, K + m) if terminated (tail included)."""
        u = np.atleast_2d(np.asarray(u, dtype=int))
        B, K = u.shape
        T = K + (self.m if terminate else 0)
        sys_ = np.zeros((B, T), dtype=np.int8)
        par = np.zeros((B, T), dtype=np.int8)
        st = np.zeros(B, dtype=int)
        for k in range(T):
            uk = u[:, k] if k < K else self.term_input[st]
            sys_[:, k] = uk
            par[:, k] = self.parity[st, uk]
            st = self.next_state[st, uk]
        if terminate:
            assert not st.any()
        return sys_, par


def qpp_interleaver(K, f1, f2):
    """Quadratic permutation polynomial interleaver pi(i) = (f1*i + f2*i^2) mod K.

    The LTE turbo interleaver (3GPP TS 36.212 Table 5.1.3-3) has this form; e.g. K=40 uses
    (f1, f2) = (3, 10) and K=6144 uses (263, 480).  Raises if the pair is not a permutation.
    """
    i = np.arange(K, dtype=np.int64)
    pi = (f1 * i + f2 * i * i) % K
    if len(np.unique(pi)) != K:
        raise ValueError("(f1, f2) does not define a permutation for this K")
    return pi.astype(int)


# ----------------------------------------------------------------------------- BCJR
def _maxstar(a, b, maxlog):
    if maxlog:
        return np.maximum(a, b)
    return np.maximum(a, b) + np.log1p(np.exp(-np.abs(a - b)))


def _maxstar_reduce(x, axis, maxlog):
    if maxlog:
        return x.max(axis=axis)
    m = x.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.exp(x - m).sum(axis=axis))


def bcjr(code: RSC, Lu, Lp, terminated=True, maxlog=False):
    """Batched BCJR (log-MAP, or max-log-MAP) for an RSC.

    Lu: (B, T) total input LLRs (channel systematic + a-priori), Lp: (B, T) parity LLRs.
    Returns APP LLRs of the T inputs, shape (B, T).  Starting state 0; ending state 0 if
    ``terminated``, otherwise all ending states equally likely.
    """
    Lu, Lp = np.atleast_2d(Lu), np.atleast_2d(Lp)
    B, T = Lu.shape
    S = code.S
    xs = np.array([1.0, -1.0])                                   # u = 0 -> +1
    xp = 1.0 - 2.0 * code.parity                                 # (S, 2)
    # branch metrics gamma[b, k, s, u]
    g = 0.5 * (Lu[:, :, None, None] * xs[None, None, None, :] +
               Lp[:, :, None, None] * xp[None, None, :, :])
    NEG = -1e30
    alpha = np.full((B, T + 1, S), NEG)
    alpha[:, 0, 0] = 0.0
    ps, pu = code.prev_s, code.prev_u
    for k in range(T):
        a = alpha[:, k]
        gk = g[:, k]
        c0 = a[:, ps[:, 0]] + gk[:, ps[:, 0], pu[:, 0]]
        c1 = a[:, ps[:, 1]] + gk[:, ps[:, 1], pu[:, 1]]
        an = _maxstar(c0, c1, maxlog)
        alpha[:, k + 1] = an - an.max(axis=1, keepdims=True)
    beta = np.full((B, S), NEG)
    if terminated:
        beta[:, 0] = 0.0
    else:
        beta[:] = 0.0
    ns = code.next_state
    L = np.empty((B, T))
    for k in range(T - 1, -1, -1):
        gk = g[:, k]
        m0 = alpha[:, k] + gk[:, :, 0] + beta[:, ns[:, 0]]      # (B, S) transitions with u=0
        m1 = alpha[:, k] + gk[:, :, 1] + beta[:, ns[:, 1]]
        L[:, k] = _maxstar_reduce(m0, 1, maxlog) - _maxstar_reduce(m1, 1, maxlog)
        bn = _maxstar(gk[:, :, 0] + beta[:, ns[:, 0]], gk[:, :, 1] + beta[:, ns[:, 1]], maxlog)
        beta = bn - bn.max(axis=1, keepdims=True)
    return L


# ----------------------------------------------------------------------------- turbo code
class TurboCode:
    """Parallel concatenation of two identical RSCs through interleaver ``pi``.

    Output streams (LTE style, TS 36.212 Sec. 5.1.3.2): systematic x, parity z (encoder 1),
    parity z' (encoder 2), each of length K, plus 4*m tail bits (both encoders terminated
    separately, each sending m systematic and m parity tail bits). Mother rate
    K / (3K + 4m); with ``puncture=True`` alternate parity bits of each encoder are
    deleted, giving rate ~1/2.
    """

    def __init__(self, K, pi=None, rsc=None, seed=0, puncture=False):
        self.K = K
        self.rsc = rsc or RSC()
        self.pi = np.asarray(pi) if pi is not None else np.random.default_rng(seed).permutation(K)
        self.puncture = puncture
        m = self.rsc.m
        self.keep1 = np.ones(K, bool)
        self.keep2 = np.ones(K, bool)
        if puncture:
            self.keep1[1::2] = False
            self.keep2[0::2] = False
        self.n = K + self.keep1.sum() + self.keep2.sum() + 4 * m

    @property
    def rate(self):
        return self.K / self.n

    def encode(self, u):
        """u: (B, K) -> dict of streams (0/1 int8): sys (B,K), p1 (B,K), p2 (B,K),
        t1s, t1p, t2s, t2p (B, m).  Punctured positions are still returned; use
        ``flatten`` to get the transmitted bits."""
        u = np.atleast_2d(u)
        K, m = self.K, self.rsc.m
        s1, p1 = self.rsc.encode(u)
        s2, p2 = self.rsc.encode(u[:, self.pi])
        return dict(sys=s1[:, :K], p1=p1[:, :K], p2=p2[:, :K], t1s=s1[:, K:], t1p=p1[:, K:],
                    t2s=s2[:, K:], t2p=p2[:, K:])

    def flatten(self, cw):
        return np.concatenate([cw["sys"], cw["p1"][:, self.keep1], cw["p2"][:, self.keep2],
                               cw["t1s"], cw["t1p"], cw["t2s"], cw["t2p"]], axis=1)

    def unflatten(self, L):
        """Inverse of flatten for LLRs; punctured positions get LLR 0."""
        K, m = self.K, self.rsc.m
        B = L.shape[0]
        i = 0
        out = {}
        out["sys"] = L[:, i:i + K]; i += K
        for name, keep in (("p1", self.keep1), ("p2", self.keep2)):
            a = np.zeros((B, K)); nk = keep.sum()
            a[:, keep] = L[:, i:i + nk]; i += nk
            out[name] = a
        for name in ("t1s", "t1p", "t2s", "t2p"):
            out[name] = L[:, i:i + m]; i += m
        return out

    def decode(self, Lch, iters=8, maxlog=False, record=False, scale=1.0, Linfo=None):
        """Iterative decoding.  Lch: (B, n) channel LLRs in ``flatten`` order.

        Returns hard decisions (B, K) after the last iteration; with ``record=True`` returns
        a list with the decisions after every full iteration.  ``scale`` multiplies the
        extrinsic information (about 0.7 is the usual correction for max-log-MAP).
        With ``Linfo`` (known info bits, (B,K)), also returns the mutual information of the
        a-priori input to each decoder at every half-iteration (EXIT trajectory).
        """
        c = self.unflatten(np.atleast_2d(Lch))
        K, pi = self.K, self.pi
        inv = np.argsort(pi)
        Ls = c["sys"]
        La = np.zeros_like(Ls)
        hist, traj = [], []
        for it in range(iters):
            Lu1 = np.concatenate([Ls + La, c["t1s"]], axis=1)
            Lp1 = np.concatenate([c["p1"], c["t1p"]], axis=1)
            L1 = bcjr(self.rsc, Lu1, Lp1, maxlog=maxlog)[:, :K]
            Le1 = scale * (L1 - Ls - La)
            if Linfo is not None:
                traj.append((mi_llr(La, Linfo), mi_llr(Le1, Linfo)))
            La2 = Le1[:, pi]
            Lu2 = np.concatenate([Ls[:, pi] + La2, c["t2s"]], axis=1)
            Lp2 = np.concatenate([c["p2"], c["t2p"]], axis=1)
            L2 = bcjr(self.rsc, Lu2, Lp2, maxlog=maxlog)[:, :K]
            Le2 = scale * (L2 - Ls[:, pi] - La2)
            if Linfo is not None:
                traj.append((mi_llr(La2, Linfo[:, pi]), mi_llr(Le2, Linfo[:, pi])))
            La = Le2[:, inv]
            if record:
                hist.append((L2[:, inv] < 0).astype(np.int8))
        out = hist if record else (L2[:, inv] < 0).astype(np.int8)
        return (out, traj) if Linfo is not None else out


# ----------------------------------------------------------------------------- EXIT tools
_JX = np.linspace(0, 60, 3001)
_JY = None


def _J_exact(s):
    """I(X;L) for L ~ N(s^2/2 * x, s^2) (consistent Gaussian LLR), by quadrature."""
    if s < 1e-6:
        return 0.0
    t, w = np.polynomial.hermite.hermgauss(80)
    L = s * s / 2 + np.sqrt(2) * s * t
    return float(1 - np.sum(w * np.logaddexp(0, -L)) / np.sqrt(np.pi) / np.log(2))


def J(s):
    """Mutual information (bits) between a bit and a consistent Gaussian LLR of std s."""
    global _JY
    if _JY is None:
        _JY = np.array([_J_exact(x) for x in _JX])
    return np.interp(s, _JX, _JY)


def Jinv(I):
    J(1.0)
    return np.interp(np.clip(I, 0, _JY[-1]), _JY, _JX)


def gaussian_apriori(bits, I, rng):
    """Consistent Gaussian a-priori LLRs for ``bits`` carrying mutual information I."""
    s = float(Jinv(I))
    x = 1.0 - 2.0 * np.asarray(bits)
    return s * s / 2 * x + s * rng.standard_normal(x.shape)


def mi_llr(L, bits):
    """Time-average estimate of I(X;L) = 1 - E[log2(1 + exp(-x L))] with x = 1 - 2 bit."""
    x = 1.0 - 2.0 * np.asarray(bits)
    return float(1 - np.mean(np.logaddexp(0, -x * np.asarray(L))) / np.log(2))


def exit_curve_rsc(rsc, sigma, IA_grid, K=10000, reps=8, rng=None, maxlog=False):
    """EXIT curve of one constituent decoder of a rate-1/3 PCCC (systematic LLRs included in
    its input, excluded from its extrinsic output).  Channel: BPSK, noise std ``sigma``.
    Extrinsic errors come in bursts (trellis error events), so the estimate is averaged over
    ``reps`` independent frames of K bits per a-priori value."""
    rng = rng or np.random.default_rng(0)
    IA = np.repeat(np.asarray(IA_grid, float), reps)
    B = len(IA)
    u = rng.integers(0, 2, (B, K))
    s, p = rsc.encode(u)
    Ls = 2 * ((1 - 2.0 * s) + sigma * rng.standard_normal(s.shape)) / sigma ** 2
    Lp = 2 * ((1 - 2.0 * p) + sigma * rng.standard_normal(p.shape)) / sigma ** 2
    La = np.stack([gaussian_apriori(u[b], IA[b], rng) for b in range(B)])
    La = np.concatenate([La, np.zeros((B, rsc.m))], axis=1)
    L = bcjr(rsc, Ls + La, Lp, maxlog=maxlog)
    Le = (L - Ls - La)[:, :K]
    IE = np.array([mi_llr(Le[b], u[b]) for b in range(B)])
    return IE.reshape(-1, reps).mean(axis=1)


# ----------------------------------------------------------------------------- DE on the BEC
def de_bec_regular(eps, dv, dc, iters=200):
    """Erasure probability of variable-to-check messages over iterations, (dv, dc) ensemble."""
    x = eps
    out = [x]
    for _ in range(iters):
        x = eps * (1 - (1 - x) ** (dc - 1)) ** (dv - 1)
        out.append(x)
    return np.array(out)


def bec_threshold(lam, rho, tol=1e-6):
    """BEC threshold for edge-perspective degree polynomials given as coefficient lists
    (lam[i] = fraction of edges on degree-(i+1) variable nodes, same for rho)."""
    lam_f = lambda x: sum(c * x ** i for i, c in enumerate(lam))
    rho_f = lambda x: sum(c * x ** i for i, c in enumerate(rho))
    xs = np.linspace(1e-6, 1, 20000)
    # eps* = min over x of x / lambda(1 - rho(1 - x))
    return float(np.min(xs / np.maximum(lam_f(1 - rho_f(1 - xs)), 1e-300)))


def de_bec_coupled(eps, dv, dc, L, w=None, iters=5000, tol=1e-12, record_every=None):
    """Density evolution for the (dv, dc, L, w) spatially coupled ensemble on the BEC
    (Kudekar, Richardson & Urbanke, IEEE Trans. IT 2011):

        x_i <- eps * (1 - 1/w sum_j (1 - 1/w sum_k x_{i+j-k})^(dc-1))^(dv-1),

    with x_i = 0 outside positions 0..L-1 (the known boundary that seeds the decoding wave).
    Returns the final erasure profile (and snapshots every ``record_every`` iterations)."""
    w = w or dv
    pad = 2 * (w - 1)
    x = np.zeros(L + 2 * pad)
    inside = np.zeros_like(x, bool)
    inside[pad:pad + L] = True
    x[inside] = eps
    ker = np.ones(w) / w
    snaps = []
    for t in range(iters):
        xm = np.convolve(x, ker, mode="full")[:len(x)]            # (1/w) sum_k x_{i-k}
        y = 1 - (1 - xm) ** (dc - 1)
        ym = np.convolve(y[::-1], ker, mode="full")[:len(x)][::-1]  # (1/w) sum_j y_{i+j}
        xn = np.where(inside, eps * ym ** (dv - 1), 0.0)
        if record_every and t % record_every == 0:
            snaps.append(xn[inside].copy())
        done = np.max(np.abs(xn - x)) < tol
        x = xn
        if done:
            break
    return (x[inside], snaps) if record_every else x[inside]
