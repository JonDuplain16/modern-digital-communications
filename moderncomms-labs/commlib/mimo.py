"""MIMO channels, detectors, space-time coding, capacity and array processing."""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["rayleigh_mimo", "zf_detect", "mmse_detect", "ml_detect", "mmse_sic_detect",
           "capacity_equal_power", "capacity_waterfilling", "waterfill",
           "alamouti_encode", "alamouti_decode", "mrc", "ula_steering", "array_factor_db",
           "correlated_mimo"]


def rayleigh_mimo(nr, nt, n=None, rng=None):
    """i.i.d. CN(0,1) channel matrix (or a stack of n matrices)."""
    rng = np.random.default_rng(rng)
    shape = (nr, nt) if n is None else (n, nr, nt)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def correlated_mimo(nr, nt, rho_r=0.0, rho_t=0.0, rng=None):
    """Kronecker model with exponential correlation matrices."""
    Rr = rho_r ** np.abs(np.subtract.outer(np.arange(nr), np.arange(nr)))
    Rt = rho_t ** np.abs(np.subtract.outer(np.arange(nt), np.arange(nt)))
    Hw = rayleigh_mimo(nr, nt, rng=rng)
    return np.linalg.cholesky(Rr + 1e-12 * np.eye(nr)) @ Hw @ np.linalg.cholesky(Rt + 1e-12 * np.eye(nt)).conj().T


def zf_detect(H, y):
    return np.linalg.pinv(H) @ y


def mmse_detect(H, y, n0, es=1.0):
    nt = H.shape[1]
    W = np.linalg.solve(H.conj().T @ H + (n0 / es) * np.eye(nt), H.conj().T)
    return W @ y


def ml_detect(H, y, points):
    """Exhaustive ML over all |points|^nt candidate vectors (small systems only)."""
    nt = H.shape[1]
    cands = np.array(list(itertools.product(points, repeat=nt))).T      # (nt, M^nt)
    d = np.sum(np.abs(y[:, :, None] - (H @ cands)[:, None, :]) ** 2, axis=0) \
        if y.ndim == 2 else np.sum(np.abs(y[:, None] - H @ cands) ** 2, axis=0)
    idx = np.argmin(d, axis=-1)
    return cands[:, idx]


def mmse_sic_detect(H, y, n0, points):
    """Ordered MMSE successive interference cancellation (V-BLAST style)."""
    H = H.copy()
    y = y.copy()
    nt = H.shape[1]
    remaining = list(range(nt))
    xhat = np.zeros(nt, dtype=complex)
    while remaining:
        Hr = H[:, remaining]
        W = np.linalg.solve(Hr.conj().T @ Hr + n0 * np.eye(len(remaining)), Hr.conj().T)
        # post-detection SINR ordering: smallest MSE first
        mse = np.real(np.diag(np.linalg.inv(Hr.conj().T @ Hr / n0 + np.eye(len(remaining)))))
        i = int(np.argmin(mse))
        z = W[i] @ y
        s = points[np.argmin(np.abs(z - points))]
        k = remaining.pop(i)
        xhat[k] = s
        y = y - H[:, k] * s
    return xhat


def capacity_equal_power(H, snr):
    """log2 det(I + snr/nt H H^H), bits/s/Hz."""
    nr, nt = H.shape[-2:]
    M = np.eye(nr) + snr / nt * H @ np.swapaxes(H.conj(), -1, -2)
    return np.real(np.log2(np.linalg.det(M)))


def waterfill(gains, total_power):
    """Water-filling power allocation over parallel channels with gains g_i (SNR per unit power)."""
    g = np.sort(np.asarray(gains))[::-1]
    for k in range(len(g), 0, -1):
        mu = (total_power + np.sum(1 / g[:k])) / k
        p = mu - 1 / g[:k]
        if p[-1] > 0:
            break
    out = np.maximum(mu - 1 / np.asarray(gains), 0)
    return out


def capacity_waterfilling(H, snr):
    s = np.linalg.svd(H, compute_uv=False)
    g = s ** 2
    p = waterfill(g, snr)
    return np.sum(np.log2(1 + p * g))


def alamouti_encode(sym):
    """2x1 Alamouti: returns (2, 2*len/2) matrix, rows = antennas, cols = time."""
    s = np.asarray(sym).reshape(-1, 2)
    tx = np.empty((2, 2 * len(s)), dtype=complex)
    tx[0, 0::2], tx[1, 0::2] = s[:, 0], s[:, 1]
    tx[0, 1::2], tx[1, 1::2] = -np.conj(s[:, 1]), np.conj(s[:, 0])
    return tx / np.sqrt(2)


def alamouti_decode(r, h):
    """r: received (time,), h: (2,) or (n_blocks, 2) channel constant over each block."""
    r = r.reshape(-1, 2)
    h = np.broadcast_to(h, (len(r), 2))
    h1, h2 = h[:, 0], h[:, 1]
    s1 = np.conj(h1) * r[:, 0] + h2 * np.conj(r[:, 1])
    s2 = np.conj(h2) * r[:, 0] - h1 * np.conj(r[:, 1])
    g = (np.abs(h1) ** 2 + np.abs(h2) ** 2) / np.sqrt(2)
    return np.stack([s1 / g, s2 / g], axis=1).reshape(-1)


def mrc(r, h):
    """Maximum-ratio combining. r, h: (n_rx, n_sym)."""
    return np.sum(np.conj(h) * r, axis=0) / np.sum(np.abs(h) ** 2, axis=0)


def ula_steering(n, theta_rad, d_over_lambda=0.5):
    """Uniform linear array steering vector(s); theta measured from broadside."""
    k = np.arange(n)[:, None]
    return np.exp(-2j * np.pi * d_over_lambda * k * np.sin(np.atleast_1d(theta_rad))[None, :])


def array_factor_db(w, thetas, d_over_lambda=0.5):
    a = ula_steering(len(w), thetas, d_over_lambda)
    af = np.abs(np.conj(w) @ a) ** 2
    return 10 * np.log10(af / af.max() + 1e-12)
