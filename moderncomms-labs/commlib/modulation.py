"""Constellations, Gray mapping, soft demapping and closed-form error rates.

Convention used throughout the course: every constellation is normalized to
unit average symbol energy, E[|a|^2] = 1, so Es/N0 in dB equals the SNR per
symbol at the output of a unit-energy matched filter.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

__all__ = [
    "Constellation", "get_constellation", "qfunc", "random_bits",
    "ser_mpsk", "ser_mqam", "ber_mqam_gray", "ber_bpsk", "ber_bpsk_rayleigh",
    "ebn0_to_esn0", "esn0_to_ebn0",
]


def qfunc(x):
    """Gaussian tail probability Q(x) = P(N(0,1) > x)."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def random_bits(n, rng=None):
    rng = np.random.default_rng(rng)
    return rng.integers(0, 2, size=int(n), dtype=np.int8)


def _gray(n):
    n = np.asarray(n)
    return n ^ (n >> 1)


def _int_to_bits(v, k):
    """Integer array -> (len, k) bit matrix, MSB first."""
    v = np.asarray(v, dtype=np.int64)
    return ((v[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int8)


def _bits_to_int(b):
    b = np.asarray(b, dtype=np.int64)
    k = b.shape[-1]
    return (b << np.arange(k - 1, -1, -1)).sum(axis=-1)


class Constellation:
    """A labelled constellation: points[i] carries bit label labels[i]."""

    def __init__(self, points, labels, name="custom"):
        points = np.asarray(points, dtype=complex)
        self.points = points / np.sqrt(np.mean(np.abs(points) ** 2))
        self.M = len(points)
        self.k = int(np.log2(self.M))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.name = name
        # table: integer label -> constellation point
        self._lut = np.empty(self.M, dtype=complex)
        self._lut[self.labels] = self.points
        self.bit_matrix = _int_to_bits(np.arange(self.M), self.k)  # label bits

    # ---- mapping -------------------------------------------------------
    def modulate(self, bits):
        bits = np.asarray(bits).astype(np.int64)
        n = len(bits) // self.k * self.k
        idx = _bits_to_int(bits[:n].reshape(-1, self.k))
        return self._lut[idx]

    def nearest(self, y):
        """Index (label) of the nearest constellation point for each sample."""
        y = np.asarray(y).ravel()
        out = np.empty(len(y), dtype=np.int64)
        step = max(1, 2_000_000 // self.M)          # bound memory for large M
        for i in range(0, len(y), step):
            d = np.abs(y[i:i + step, None] - self._lut[None, :]) ** 2
            out[i:i + step] = np.argmin(d, axis=1)
        return out

    def decide(self, y):
        return self._lut[self.nearest(y)]

    def demodulate(self, y):
        """Hard-decision bits."""
        return self.bit_matrix[self.nearest(y)].reshape(-1)

    def llr(self, y, n0, exact=False, h=None):
        """Bit log-likelihood ratios, LLR = log P(b=0|y)/P(b=1|y).

        y: received symbols, n0: complex noise variance (scalar or per-symbol),
        h: optional per-symbol complex gain (y = h a + n).
        exact=False uses the max-log approximation.
        """
        y = np.asarray(y)[:, None]
        pts = self._lut[None, :]
        if h is not None:
            pts = np.asarray(h)[:, None] * pts
        n0 = np.broadcast_to(np.asarray(n0, dtype=float), (y.shape[0],))[:, None]
        metric = -np.abs(y - pts) ** 2 / n0  # (N, M)
        out = np.empty((y.shape[0], self.k))
        for b in range(self.k):
            zero = self.bit_matrix[:, b] == 0
            if exact:
                from scipy.special import logsumexp
                out[:, b] = logsumexp(metric[:, zero], axis=1) - logsumexp(metric[:, ~zero], axis=1)
            else:
                out[:, b] = metric[:, zero].max(axis=1) - metric[:, ~zero].max(axis=1)
        return out.reshape(-1)

    @property
    def dmin(self):
        d = np.abs(self.points[:, None] - self.points[None, :])
        return d[d > 0].min()

    def __repr__(self):
        return f"Constellation({self.name}, M={self.M})"


def _psk(M, offset=None):
    if offset is None:
        offset = np.pi / 4 if M == 4 else 0.0
    idx = np.arange(M)
    pts = np.exp(1j * (2 * np.pi * idx / M + offset))
    return pts, _gray(idx)


def _pam_levels(L):
    return np.arange(-(L - 1), L, 2).astype(float)


def _square_qam(M):
    L = int(round(np.sqrt(M)))
    assert L * L == M, "square QAM only"
    kh = int(np.log2(L))
    lev = _pam_levels(L)
    g = _gray(np.arange(L))
    I, Q = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    pts = lev[I] + 1j * lev[Q]
    labels = (g[I] << kh) | g[Q]
    return pts.ravel(), labels.ravel()


def get_constellation(name: str) -> Constellation:
    """Named constellations: bpsk, qpsk, 8psk, 16psk, 4pam, 8pam, 16qam ... 4096qam."""
    n = name.lower().replace("-", "")
    if n == "bpsk":
        return Constellation([1, -1], [0, 1], "BPSK")
    if n.endswith("psk"):
        M = 4 if n == "qpsk" else int(n[:-3])
        p, l = _psk(M)
        return Constellation(p, l, f"{M}-PSK")
    if n.endswith("pam"):
        L = int(n[:-3])
        return Constellation(_pam_levels(L), _gray(np.arange(L)), f"{L}-PAM")
    if n.endswith("qam"):
        M = int(n[:-3])
        p, l = _square_qam(M)
        return Constellation(p, l, f"{M}-QAM")
    raise ValueError(name)


# ---- closed-form performance ------------------------------------------------
def ebn0_to_esn0(ebn0_db, bits_per_symbol, code_rate=1.0):
    return ebn0_db + 10 * np.log10(bits_per_symbol * code_rate)


def esn0_to_ebn0(esn0_db, bits_per_symbol, code_rate=1.0):
    return esn0_db - 10 * np.log10(bits_per_symbol * code_rate)


def ber_bpsk(ebn0_db):
    return qfunc(np.sqrt(2 * 10 ** (np.asarray(ebn0_db) / 10)))


def ber_bpsk_rayleigh(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def ser_mpsk(esn0_db, M):
    """Tight approximation (exact for M=2)."""
    g = 10 ** (np.asarray(esn0_db) / 10)
    if M == 2:
        return qfunc(np.sqrt(2 * g))
    return 2 * qfunc(np.sqrt(2 * g) * np.sin(np.pi / M))


def ser_mqam(esn0_db, M):
    """Exact SER of square M-QAM in AWGN."""
    g = 10 ** (np.asarray(esn0_db) / 10)
    p = 2 * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * g / (M - 1)))
    return 1 - (1 - p) ** 2


def ber_mqam_gray(esn0_db, M):
    """Nearest-neighbour Gray-coded BER approximation for square M-QAM."""
    g = 10 ** (np.asarray(esn0_db) / 10)
    k = np.log2(M)
    return 4 / k * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * g / (M - 1)))
