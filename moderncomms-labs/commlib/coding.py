"""Channel codes used in the course: Hamming, convolutional/Viterbi, LDPC and polar.

LLR convention everywhere: L = log P(bit=0) / P(bit=1); BPSK maps 0 -> +1, 1 -> -1,
so for y = x + n with noise variance sigma^2 (real), L = 2 y / sigma^2.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hamming74_encode", "hamming74_decode", "ConvCode", "LDPCCode",
           "PolarCode", "crc_remainder", "CRC11_5G", "CRC24C_5G"]

# ---------------------------------------------------------------- Hamming(7,4)
_G74 = np.array([[1, 0, 0, 0, 1, 1, 0],
                 [0, 1, 0, 0, 1, 0, 1],
                 [0, 0, 1, 0, 0, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8)
_H74 = np.array([[1, 1, 0, 1, 1, 0, 0],
                 [1, 0, 1, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0, 0, 1]], dtype=np.int8)


def hamming74_encode(bits):
    u = np.asarray(bits).reshape(-1, 4)
    return (u @ _G74 % 2).reshape(-1).astype(np.int8)


def hamming74_decode(bits):
    """Syndrome decoding (corrects any single error per 7-bit block)."""
    c = np.asarray(bits).reshape(-1, 7).copy()
    s = c @ _H74.T % 2
    cols = {tuple(_H74[:, j]): j for j in range(7)}
    for i, si in enumerate(s):
        if si.any():
            c[i, cols[tuple(si)]] ^= 1
    return c[:, :4].reshape(-1).astype(np.int8)


# ---------------------------------------------------------------- CRC
CRC11_5G = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]          # D^11+D^10+D^9+D^5+1 (38.212)
CRC24C_5G = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]


def crc_remainder(bits, poly):
    """Remainder of bits(D) * D^r divided by poly(D) over GF(2); poly MSB first."""
    poly = np.asarray(poly, dtype=np.int8)
    r = len(poly) - 1
    reg = np.concatenate([np.asarray(bits, dtype=np.int8), np.zeros(r, dtype=np.int8)])
    for i in range(len(bits)):
        if reg[i]:
            reg[i:i + r + 1] ^= poly
    return reg[-r:]


# ---------------------------------------------------------------- Convolutional
class ConvCode:
    """Feed-forward convolutional code with Viterbi decoding.

    Default: the industry-standard K=7, rate-1/2 code with generators
    (133, 171) octal used in 802.11a/g/n, DVB-S and CCSDS.
    """

    def __init__(self, K=7, gens=(0o133, 0o171)):
        self.K, self.gens = K, gens
        self.n_out = len(gens)
        self.S = 1 << (K - 1)
        s = np.arange(self.S)
        self.next_state = np.zeros((self.S, 2), dtype=int)
        self.outputs = np.zeros((self.S, 2, self.n_out), dtype=np.int8)
        for b in (0, 1):
            reg = (b << (K - 1)) | s
            self.next_state[:, b] = reg >> 1
            for j, g in enumerate(gens):
                self.outputs[:, b, j] = np.array([bin(r & g).count("1") & 1 for r in reg])

    @property
    def rate(self):
        return 1 / self.n_out

    def encode(self, bits, terminate=True):
        bits = np.asarray(bits, dtype=int)
        if terminate:
            bits = np.concatenate([bits, np.zeros(self.K - 1, dtype=int)])
        st = 0
        out = np.empty((len(bits), self.n_out), dtype=np.int8)
        for i, b in enumerate(bits):
            out[i] = self.outputs[st, b]
            st = self.next_state[st, b]
        return out.reshape(-1)

    def decode(self, llr, terminated=True):
        """Viterbi decoding from LLRs (use +/-1 hard values for hard decisions)."""
        r = np.asarray(llr, dtype=float).reshape(-1, self.n_out)
        T = len(r)
        S, K = self.S, self.K
        ns = np.arange(S)
        b_of_ns = ns >> (K - 2)
        low = (ns & ((1 << (K - 2)) - 1)) << 1
        prev = np.stack([low, low | 1], axis=1)                 # (S, 2) predecessors
        sgn = 1 - 2 * self.outputs.astype(float)                # 0 -> +1
        # branch outputs for (prev state, input b_of_ns)
        bo = sgn[prev, b_of_ns[:, None]]                        # (S, 2, n_out)
        pm = np.full(S, np.inf)
        pm[0] = 0.0
        dec = np.zeros((T, S), dtype=np.int8)
        for t in range(T):
            bm = -(bo * r[t]).sum(axis=2)                       # (S, 2) lower is better
            cand = pm[prev] + bm
            choice = np.argmin(cand, axis=1)
            pm = cand[ns, choice]
            dec[t] = choice
        st = 0 if terminated else int(np.argmin(pm))
        bits = np.empty(T, dtype=np.int8)
        for t in range(T - 1, -1, -1):
            bits[t] = st >> (K - 2)
            st = prev[st, dec[t, st]]
        return bits[:T - (K - 1)] if terminated else bits


# ---------------------------------------------------------------- LDPC
def _gf2_rref(H):
    H = H.copy() % 2
    m, n = H.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        rows = np.where(H[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        H[[r, p]] = H[[p, r]]
        others = np.where(H[:, c])[0]
        others = others[others != r]
        H[others] ^= H[r]
        pivots.append(c)
        r += 1
    return H[:r], pivots


class LDPCCode:
    """Binary LDPC code built with a simplified progressive-edge-growth (PEG)
    construction (Hu, Eleftheriou & Arnold, 2005), which avoids short cycles.

    Encoding uses a systematic form derived from the RREF of H; decoding supports
    sum-product (belief propagation) and normalized min-sum.
    """

    def __init__(self, n=576, rate=0.5, dv=3, seed=1, H=None):
        rng = np.random.default_rng(seed)
        if H is None:
            m = int(round(n * (1 - rate)))
            H = self._peg(n, m, dv, rng)
        self.H = H.astype(np.int8)
        self.m, self.n = H.shape
        R, piv = _gf2_rref(self.H)
        self.pivots = np.array(piv)
        self.info = np.setdiff1d(np.arange(self.n), self.pivots)
        self.k = len(self.info)
        self.P = R[:, self.info]                      # parity = P @ u
        ce, ev = np.nonzero(self.H)
        self.edge_c, self.edge_v = ce, ev
        self.E = len(ce)
        dc = np.bincount(ce, minlength=self.m)
        self.dcmax = dc.max()
        self.slot = np.zeros((self.m, self.dcmax), dtype=int) - 1
        pos = np.zeros(self.m, dtype=int)
        for e, c in enumerate(ce):
            self.slot[c, pos[c]] = e
            pos[c] += 1

    @staticmethod
    def _peg(n, m, dv, rng):
        H = np.zeros((m, n), dtype=np.int8)
        cdeg = np.zeros(m, dtype=int)
        vnbrs = [[] for _ in range(n)]
        cnbrs = [[] for _ in range(m)]
        for v in range(n):
            for k in range(dv):
                if k == 0:
                    cand = np.where(cdeg == cdeg.min())[0]
                else:
                    reached = set(vnbrs[v])
                    frontier_v = {v}
                    last_unreached = None
                    while True:
                        new_c = set()
                        for vv in frontier_v:
                            new_c.update(vnbrs[vv])
                        new_c -= reached
                        unreached = np.array([c for c in range(m) if c not in reached and c not in new_c])
                        if len(unreached) == 0 or not new_c:
                            if len(unreached) == 0:
                                unreached = last_unreached
                            break
                        last_unreached = unreached
                        reached |= new_c
                        frontier_v = set()
                        for c in new_c:
                            frontier_v.update(cnbrs[c])
                    pool = unreached if unreached is not None and len(unreached) else np.setdiff1d(np.arange(m), vnbrs[v])
                    cand = pool[cdeg[pool] == cdeg[pool].min()]
                c = int(rng.choice(cand))
                H[c, v] = 1
                cdeg[c] += 1
                vnbrs[v].append(c)
                cnbrs[c].append(v)
        return H

    @property
    def rate(self):
        return self.k / self.n

    def encode(self, u):
        u = np.asarray(u, dtype=np.int8).reshape(-1, self.k)
        c = np.zeros((u.shape[0], self.n), dtype=np.int8)
        c[:, self.info] = u
        c[:, self.pivots] = (u.astype(int) @ self.P.T.astype(int)) % 2
        return c.reshape(-1)

    def syndrome_ok(self, c):
        return not ((self.H.astype(int) @ np.asarray(c, dtype=int)) % 2).any()

    def decode(self, llr, iters=50, method="minsum", alpha=0.8, return_iters=False):
        """Flooding BP. Returns hard decisions on the full codeword."""
        Lch = np.asarray(llr, dtype=float)
        r = np.zeros(self.E)
        L = Lch.copy()
        it = 0
        for it in range(1, iters + 1):
            q = L[self.edge_v] - r
            Q = np.where(self.slot >= 0, q[np.maximum(self.slot, 0)], np.nan)
            if method == "minsum":
                mag = np.where(np.isnan(Q), np.inf, np.abs(Q))
                sgn = np.where(np.isnan(Q), 1.0, np.sign(Q) + (Q == 0))
                tot = np.prod(sgn, axis=1, keepdims=True)
                i1 = np.argmin(mag, axis=1)
                m1 = mag[np.arange(self.m), i1]
                mag2 = mag.copy()
                mag2[np.arange(self.m), i1] = np.inf
                m2 = mag2.min(axis=1)
                R = alpha * tot * sgn * np.where(np.arange(self.dcmax)[None, :] == i1[:, None],
                                                 m2[:, None], m1[:, None])
            else:  # sum-product
                t = np.where(np.isnan(Q), 1.0, np.tanh(np.clip(Q, -30, 30) / 2))
                t = np.where(np.abs(t) < 1e-12, 1e-12, t)
                prod = np.prod(t, axis=1, keepdims=True)
                R = 2 * np.arctanh(np.clip(prod / t, -0.999999999, 0.999999999))
            valid = self.slot >= 0
            r = np.zeros(self.E)
            r[self.slot[valid]] = R[valid]
            L = Lch + np.bincount(self.edge_v, weights=r, minlength=self.n)
            c = (L < 0).astype(np.int8)
            if self.syndrome_ok(c):
                break
        c = (L < 0).astype(np.int8)
        return (c, it) if return_iters else c

    def info_bits(self, c):
        return np.asarray(c).reshape(-1, self.n)[:, self.info].reshape(-1)


# ---------------------------------------------------------------- Polar
def _f(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _g(a, b, u):
    return b + (1 - 2 * u) * a


class PolarCode:
    """Arikan polar code of length N=2^n with K information bits.

    Encoder x = u G_N with G_N = F^{(x)n}, F = [[1,0],[1,1]] (natural order).
    Construction: Bhattacharyya-parameter recursion at a design Es/N0.
    Decoders: successive cancellation (SC) and CRC-aided SC list (CA-SCL).
    """

    def __init__(self, N=256, K=128, design_snr_db=1.0, crc_poly=None):
        assert N & (N - 1) == 0
        self.N, self.K = N, K
        self.crc = None if crc_poly is None else np.asarray(crc_poly, dtype=np.int8)
        self.ncrc = 0 if crc_poly is None else len(crc_poly) - 1
        z0 = np.exp(-10 ** (design_snr_db / 10))
        z = self._bhat(N, z0)
        order = np.argsort(z)                          # most reliable first
        self.info_set = np.sort(order[:K + self.ncrc])
        self.frozen = np.ones(N, dtype=bool)
        self.frozen[self.info_set] = False
        self.z = z

    @staticmethod
    def _bhat(N, z0):
        if N == 1:
            return np.array([z0])
        return np.concatenate([PolarCode._bhat(N // 2, 2 * z0 - z0 ** 2),
                               PolarCode._bhat(N // 2, z0 ** 2)])

    @property
    def rate(self):
        return self.K / self.N

    @staticmethod
    def transform(u):
        u = np.asarray(u, dtype=np.int8)
        N = len(u)
        if N == 1:
            return u.copy()
        h = N // 2
        a = PolarCode.transform(u[:h])
        b = PolarCode.transform(u[h:])
        return np.concatenate([a ^ b, b])

    def encode(self, msg):
        msg = np.asarray(msg, dtype=np.int8)
        if self.crc is not None:
            msg = np.concatenate([msg, crc_remainder(msg, self.crc)])
        u = np.zeros(self.N, dtype=np.int8)
        u[self.info_set] = msg
        return self.transform(u)

    # --- list decoder over the code tree (L=1 gives SC) ---------------------
    def _node(self, alpha, frozen, pm, Lmax):
        n = alpha.shape[1]
        if n == 1:
            a = alpha[:, 0]
            if frozen[0]:
                pm = pm + np.where(a < 0, np.abs(a), 0.0)
                u = np.zeros((len(pm), 1), dtype=np.int8)
                return u, u, pm, np.arange(len(pm))
            p0 = pm + np.where(a < 0, np.abs(a), 0.0)
            p1 = pm + np.where(a >= 0, np.abs(a), 0.0)
            cand = np.concatenate([p0, p1])
            keep = np.argsort(cand, kind="stable")[:min(Lmax, len(cand))]
            parent = keep % len(pm)
            bit = (keep >= len(pm)).astype(np.int8)[:, None]
            return bit, bit, cand[keep], parent
        h = n // 2
        aL = _f(alpha[:, :h], alpha[:, h:])
        betaL, uL, pm, idx1 = self._node(aL, frozen[:h], pm, Lmax)
        alpha = alpha[idx1]
        aR = _g(alpha[:, :h], alpha[:, h:], betaL)
        betaR, uR, pm, idx2 = self._node(aR, frozen[h:], pm, Lmax)
        betaL, uL = betaL[idx2], uL[idx2]
        beta = np.concatenate([betaL ^ betaR, betaR], axis=1)
        u = np.concatenate([uL, uR], axis=1)
        return beta, u, pm, idx1[idx2]

    def decode(self, llr, L=1):
        """Returns K decoded message bits. L=1 is plain SC decoding."""
        alpha = np.asarray(llr, dtype=float)[None, :]
        _, u, pm, _ = self._node(alpha, self.frozen, np.zeros(1), L)
        order = np.argsort(pm)
        for i in order:
            m = u[i, self.info_set]
            if self.crc is None:
                return m[:self.K]
            msg, chk = m[:self.K], m[self.K:]
            if np.array_equal(crc_remainder(msg, self.crc), chk):
                return msg
        return u[order[0], self.info_set][:self.K]
