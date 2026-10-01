"""blockcodes: binary linear block codes, SECDED, interleavers and block-code performance.

Companion of Chapter 14 (*Classical Error-Control Codes*) together with ``commlib.gf``
(finite fields, Reed-Solomon, BCH and CRCs). Import explicitly:

    from commlib import blockcodes as bc
    code = bc.hamming_code(3)                  # the (7,4) code of eq. (14.h74)
    c = code.encode([1, 0, 1, 1]); u, s, flag = code.decode(c)
    sec = bc.hsiao_secded_72_64()              # the (72,64) ECC-memory code

Conventions: bit vectors are NumPy int arrays of 0/1; codewords are systematic with the
message first (c = [u | p]); H = [P^T | I].
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

__all__ = ["LinearBlockCode", "SECDED", "hamming_code", "extended_hamming_code", "hsiao_secded_72_64",
           "gf2_rank", "crc_divide", "crc_append", "block_interleave", "block_deinterleave",
           "gilbert_elliott", "block_hard_ber", "rs_ber", "rs_ser_out", "word_error_bounded"]


def gf2_rank(A):
    """Rank of a binary matrix over GF(2)."""
    A = np.array(A, dtype=np.uint8) % 2
    r = 0
    for c in range(A.shape[1]):
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        A[[r, p]] = A[[p, r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == A.shape[0]:
            break
    return r


class LinearBlockCode:
    """A systematic binary (n, k) code from its parity part P (k x (n-k)): G = [I | P],
    H = [P^T | I]. Decoding is by syndrome look-up of the coset leaders (all patterns of
    weight <= t_table), which is maximum-likelihood on a BSC for small codes."""

    def __init__(self, P, name="code", t_table=None):
        self.P = np.array(P, dtype=np.int64) % 2
        self.k, self.r = self.P.shape
        self.n = self.k + self.r
        self.name = name
        self.G = np.hstack([np.eye(self.k, dtype=np.int64), self.P])
        self.H = np.hstack([self.P.T, np.eye(self.r, dtype=np.int64)])
        self._table = None
        self._t_table = t_table

    # ---------------------------------------------------------------- basic operations
    def encode(self, u):
        """Encode one message (k bits) or a batch (..., k)."""
        u = np.asarray(u, dtype=np.int64)
        return np.concatenate([u, (u @ self.P) % 2], axis=-1)

    def syndrome(self, r):
        """s = r H^T (works on a batch)."""
        return (np.asarray(r, dtype=np.int64) @ self.H.T) % 2

    @staticmethod
    def _key(s):
        return int("".join(map(str, np.asarray(s, int))), 2) if len(s) else 0

    def coset_leaders(self):
        """Dict syndrome-integer -> minimum-weight error pattern (built lazily)."""
        if self._table is None:
            tab = {0: np.zeros(self.n, dtype=np.int64)}
            wmax = self.n if self._t_table is None else self._t_table
            for w in range(1, wmax + 1):
                for pos in combinations(range(self.n), w):
                    e = np.zeros(self.n, dtype=np.int64); e[list(pos)] = 1
                    key = self._key(self.syndrome(e))
                    if key not in tab:
                        tab[key] = e
                if len(tab) == 2 ** self.r:
                    break
            self._table = tab
        return self._table

    def decode(self, r):
        """Syndrome decoding. Returns (message estimate, syndrome, corrected_flag) where the
        flag is False when the syndrome has no leader in the (possibly truncated) table."""
        r = np.asarray(r, dtype=np.int64)
        s = self.syndrome(r)
        e = self.coset_leaders().get(self._key(s))
        if e is None:
            return r[:self.k].copy(), s, False
        c = (r + e) % 2
        return c[:self.k], s, True

    # ---------------------------------------------------------------- properties
    def codewords(self):
        """All 2^k codewords (only for small k)."""
        if self.k > 20:
            raise ValueError("too many codewords to list")
        u = (np.arange(2 ** self.k)[:, None] >> np.arange(self.k - 1, -1, -1)) & 1
        return self.encode(u)

    def weight_distribution(self):
        """A_w for w = 0..n (exhaustive; small k only)."""
        w = self.codewords().sum(axis=1)
        return np.bincount(w, minlength=self.n + 1)

    def dmin(self):
        A = self.weight_distribution()
        return int(np.nonzero(A[1:])[0][0] + 1)

    def __repr__(self):
        return f"LinearBlockCode({self.name}: n={self.n}, k={self.k})"


def hamming_code(m=3):
    """Systematic Hamming (2^m - 1, 2^m - 1 - m) code. For m = 3 the parity part is that of
    Chapter 14, eq. (14.h74): H = [1101100; 1011010; 0111001]."""
    if m == 3:
        P = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    else:
        cols = [c for c in range(1, 2 ** m) if bin(c).count("1") >= 2]
        P = np.array([[(c >> (m - 1 - i)) & 1 for i in range(m)] for c in cols])
    return LinearBlockCode(P, f"Hamming({2 ** m - 1},{2 ** m - 1 - m})", t_table=1)


def extended_hamming_code(m=3):
    """Extended Hamming (2^m, 2^m - 1 - m, 4) code: the Hamming code plus an overall parity bit."""
    h = hamming_code(m)
    par = (h.P.sum(axis=1) + 1) % 2             # makes every row of G even weight
    P = np.hstack([h.P, par[:, None]])
    return SECDED(P, f"ext. Hamming({2 ** m},{2 ** m - 1 - m})")


class SECDED(LinearBlockCode):
    """Single-error-correcting, double-error-detecting decoder for a code whose H columns are
    all distinct and of odd weight (extended Hamming or Hsiao). decode() returns
    (message, syndrome, status) with status 'ok', 'corrected' or 'detected'."""

    def __init__(self, P, name="SECDED"):
        super().__init__(P, name, t_table=1)
        self._cols = {self._key(self.H[:, j]): j for j in range(self.n)}

    def decode(self, r):
        r = np.asarray(r, dtype=np.int64).copy()
        s = self.syndrome(r)
        if not s.any():
            return r[:self.k], s, "ok"
        if s.sum() % 2 == 1:                      # odd-weight syndrome: assume a single error
            j = self._cols.get(self._key(s))
            if j is not None:
                r[j] ^= 1
                return r[:self.k], s, "corrected"
        return r[:self.k], s, "detected"


def hsiao_secded_72_64():
    """A (72,64) Hsiao SECDED code: 8 check bits, the 64 data columns of H are 56 distinct
    weight-3 columns plus 8 weight-5 columns (all odd weight), as in ECC memory modules."""
    cols3 = [c for c in combinations(range(8), 3)]
    cols5 = [c for c in combinations(range(8), 5)]
    # pick 8 weight-5 columns that keep the row weights balanced
    chosen5, rows = [], np.zeros(8, int)
    for c in cols3:
        rows[list(c)] += 1
    while len(chosen5) < 8:                      # greedy: keep the heaviest row as light as possible
        c = min((c for c in cols5 if c not in chosen5),
                key=lambda c: (np.max(rows + np.isin(np.arange(8), c)), rows[list(c)].sum(), c))
        chosen5.append(c); rows[list(c)] += 1
    P = np.zeros((64, 8), dtype=np.int64)
    for i, c in enumerate(cols3 + chosen5):
        P[i, list(c)] = 1
    return SECDED(P, "Hsiao(72,64)")


# ============================================================================ CRC (bit level)
def crc_divide(bits, poly_bits):
    """Remainder of bits(x) * x^0 divided by g(x) over GF(2); poly_bits is the generator
    MSB first including the x^r term (e.g. [1,0,0,1,1] for x^4 + x + 1). Returns r bits."""
    g = np.asarray(poly_bits, dtype=np.int64)
    r = len(g) - 1
    reg = np.array(bits, dtype=np.int64).copy()
    for i in range(len(reg) - r):
        if reg[i]:
            reg[i:i + r + 1] ^= g
    return reg[-r:]


def crc_append(bits, poly_bits):
    """Systematic CRC codeword: message followed by the remainder of x^r m(x) / g(x)."""
    r = len(poly_bits) - 1
    bits = np.asarray(bits, dtype=np.int64)
    return np.concatenate([bits, crc_divide(np.concatenate([bits, np.zeros(r, np.int64)]), poly_bits)])


# ============================================================================ interleaving / bursts
def block_interleave(x, rows, cols):
    """Write row-wise into a rows x cols array, read column-wise (len(x) == rows*cols)."""
    return np.asarray(x).reshape(rows, cols).T.reshape(-1)


def block_deinterleave(y, rows, cols):
    return np.asarray(y).reshape(cols, rows).T.reshape(-1)


def gilbert_elliott(n, p_gb, p_bg, e_good=0.0, e_bad=0.5, rng=None):
    """Two-state burst-error channel: returns an error indicator vector of length n.
    p_gb/p_bg are the good->bad and bad->good transition probabilities per symbol."""
    rng = np.random.default_rng() if rng is None else rng
    state = np.zeros(n, dtype=bool)
    u = rng.random(n)
    bad = rng.random() < p_gb / (p_gb + p_bg)
    for i in range(n):
        state[i] = bad
        bad = (u[i] >= p_bg) if bad else (u[i] < p_gb)
    return (rng.random(n) < np.where(state, e_bad, e_good)).astype(np.int64)


# ============================================================================ performance
def word_error_bounded(n, t, p):
    """Probability that more than t of n independent symbols are in error (bounded-distance
    decoding failure), for symbol error probability p."""
    p = np.asarray(p, dtype=float)
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(t + 1, n + 1))


def block_hard_ber(n, t, p):
    """Post-decoding bit error rate of a t-error-correcting binary code on a BSC(p), with the
    standard approximation that a failure with i channel errors leaves about i + t bit errors."""
    p = np.asarray(p, dtype=float)
    return sum(min(i + t, n) / n * comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(t + 1, n + 1))


def rs_ser_out(n, t, ps):
    """Reed-Solomon output symbol error rate from an independent input symbol error rate ps."""
    return block_hard_ber(n, t, ps)


def rs_ber(n, t, m, ps_sym):
    """Reed-Solomon post-decoding bit error rate (symbols of m bits; a wrong symbol has on
    average 2^(m-1)/(2^m - 1) of its bits wrong)."""
    return rs_ser_out(n, t, ps_sym) * 2 ** (m - 1) / (2 ** m - 1)
