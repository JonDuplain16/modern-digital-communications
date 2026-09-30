"""Advanced equalizers for Chapter 12 (Equalization).

Complements commlib.equalize (ZF/MMSE FIR, LMS, CMA, LMS-DFE) with:

* mmse_dfe_fir   -- finite-length MMSE decision-feedback equalizer design
* dfe_run        -- run a fixed DFE with real decisions or genie feedback
* nlms_equalizer -- normalized LMS
* rls_equalizer  -- exponentially weighted recursive least squares
* dd_lms         -- decision-directed LMS starting from given taps (e.g. after CMA)
* viterbi_mlse   -- maximum-likelihood sequence estimation (Forney/Ungerboeck model)
* bcjr_isi_bpsk  -- log-MAP symbol detector on an ISI trellis (for turbo equalization)
* thp_precode    -- Tomlinson-Harashima precoder for M-PAM

Import explicitly:  from commlib import eqadv
"""
from __future__ import annotations

import numpy as np

from .equalize import conv_matrix

__all__ = ["mmse_dfe_fir", "dfe_run", "nlms_equalizer", "rls_equalizer", "dd_lms",
           "viterbi_mlse", "bcjr_isi_bpsk", "thp_precode", "mod_centered"]


# ---------------------------------------------------------------- DFE design
def mmse_dfe_fir(h, Lf, Lb, n0, delay=None):
    """Finite-length MMSE-DFE for unit-energy i.i.d. symbols and noise variance n0.

    The receiver forms z[n] = wf^H y_n - wb^H [a[n-delay-1], ..., a[n-delay-Lb]]
    where y_n = [y[n], y[n-1], ..., y[n-Lf+1]]. The feedback symbols are assumed
    correct in the design. Returns (wf, wb, delay, mse).
    """
    h = np.asarray(h, dtype=complex)
    Lh = len(h)
    delay = Lf - 1 if delay is None else delay
    Ns = max(Lf + Lh - 1, delay + Lb + 1)           # symbols a[n], ..., a[n-Ns+1]
    H = np.zeros((Lf, Ns), dtype=complex)
    Hc = conv_matrix(h, Lf).T                        # (Lf, Lf+Lh-1): row i -> y[n-i]
    H[:, :Lf + Lh - 1] = Hc
    S = np.zeros((Lb, Ns), dtype=complex)
    for k in range(Lb):
        S[k, delay + 1 + k] = 1
    A = np.vstack([H, S])
    R = A @ A.conj().T
    R[:Lf, :Lf] += n0 * np.eye(Lf)
    p = A[:, delay]
    w = np.linalg.solve(R, p)
    mse = float(np.real(1 - p.conj() @ w))
    wf, wb = w[:Lf], -w[Lf:]
    return wf, wb, delay, mse


def dfe_run(y, wf, wb, delay, constellation, genie=None):
    """Run a fixed DFE. If genie (the true symbols) is given, feed back the truth.

    Returns the soft outputs z (aligned so z[n] estimates a[n]) and decisions.
    """
    Lf, Lb = len(wf), len(wb)
    pts = constellation.points
    ypad = np.concatenate([np.zeros(Lf - 1, dtype=complex), np.asarray(y, dtype=complex)])
    N = len(y)
    z = np.zeros(N, dtype=complex)
    dec = np.zeros(N, dtype=complex)
    past = np.zeros(Lb, dtype=complex)
    wfc, wbc = wf.conj(), wb.conj()
    for n in range(delay, N):
        u = ypad[n:n + Lf][::-1]
        zn = wfc @ u - wbc @ past
        m = n - delay
        d = pts[np.argmin(np.abs(zn - pts))]
        z[m], dec[m] = zn, d
        fb = genie[m] if genie is not None else d
        if Lb:
            past = np.roll(past, 1)
            past[0] = fb
    return z[:N - delay], dec[:N - delay]


# ------------------------------------------------------------ adaptive filters
def _regressor(y, L, delay):
    return np.concatenate([np.zeros(L - 1 - delay, dtype=complex), np.asarray(y, dtype=complex),
                           np.zeros(delay, dtype=complex)])


def nlms_equalizer(y, train, L=11, mu=0.5, eps=1e-3, constellation=None, delay=None):
    """Normalized LMS: w += mu u e* / (eps + ||u||^2). Returns (out, err, w)."""
    delay = L // 2 if delay is None else delay
    w = np.zeros(L, dtype=complex)
    w[delay] = 1
    ypad = _regressor(y, L, delay)
    out = np.empty(len(y), dtype=complex)
    err = np.empty(len(y))
    for n in range(len(y)):
        u = ypad[n:n + L][::-1]
        z = np.dot(w.conj(), u)
        d = train[n] if n < len(train) else constellation.points[np.argmin(np.abs(z - constellation.points))]
        e = d - z
        w += mu * u * np.conj(e) / (eps + np.real(np.vdot(u, u)))
        out[n], err[n] = z, abs(e) ** 2
    return out, err, w


def rls_equalizer(y, train, L=11, lam=0.99, delta=0.01, constellation=None, delay=None):
    """Exponentially weighted RLS equalizer (a-priori error form). Returns (out, err, w)."""
    delay = L // 2 if delay is None else delay
    w = np.zeros(L, dtype=complex)
    P = np.eye(L, dtype=complex) / delta
    ypad = _regressor(y, L, delay)
    out = np.empty(len(y), dtype=complex)
    err = np.empty(len(y))
    for n in range(len(y)):
        u = ypad[n:n + L][::-1]
        Pu = P @ u
        k = Pu / (lam + np.vdot(u, Pu))
        z = np.dot(w.conj(), u)
        d = train[n] if n < len(train) else constellation.points[np.argmin(np.abs(z - constellation.points))]
        e = d - z
        w += k * np.conj(e)
        P = (P - np.outer(k, u.conj() @ P)) / lam
        out[n], err[n] = z, abs(e) ** 2
    return out, err, w


def dd_lms(y, w0, constellation, mu=1e-3, delay=None):
    """Decision-directed LMS continuing from taps w0 (e.g. CMA taps)."""
    L = len(w0)
    delay = L // 2 if delay is None else delay
    w = np.array(w0, dtype=complex)
    ypad = _regressor(y, L, delay)
    pts = constellation.points
    out = np.empty(len(y), dtype=complex)
    err = np.empty(len(y))
    for n in range(len(y)):
        u = ypad[n:n + L][::-1]
        z = np.dot(w.conj(), u)
        e = pts[np.argmin(np.abs(z - pts))] - z
        w += mu * u * np.conj(e)
        out[n], err[n] = z, abs(e) ** 2
    return out, err, w


# ------------------------------------------------------------------- MLSE
def _isi_tables(h, points):
    h = np.asarray(h, dtype=complex)
    P = np.asarray(points, dtype=complex)
    M, m = len(P), len(h) - 1
    S = M ** m
    s = np.arange(S)
    digits = np.array([(s // M ** k) % M for k in range(m)])       # digit k -> a[n-1-k]
    expected = np.empty((S, M), dtype=complex)
    for i in range(M):
        expected[:, i] = h[0] * P[i] + sum(h[k + 1] * P[digits[k]] for k in range(m))
    return M, m, S, expected


def viterbi_mlse(y, h, points):
    """MLSE of the symbol sequence from y[n] = sum_k h[k] a[n-k] + white noise.

    State = (a[n-1], ..., a[n-m]); branch metric |y - expected|^2. Returns the
    detected symbol values (same length as y).
    """
    P = np.asarray(points, dtype=complex)
    M, m, S, expected = _isi_tables(h, P)
    ns = np.arange(S)
    pred = np.array([(ns // M) + j * M ** (m - 1) for j in range(M)]).T   # (S, M)
    inp = ns % M
    metric = np.zeros(S)
    N = len(y)
    tb = np.empty((N, S), dtype=np.int16)
    for n in range(N):
        bm = np.abs(y[n] - expected) ** 2                 # (S, M)
        cand = metric[pred] + bm[pred, inp[:, None]]
        j = np.argmin(cand, axis=1)
        metric = cand[ns, j]
        tb[n] = j
    s = int(np.argmin(metric))
    out = np.empty(N, dtype=int)
    for n in range(N - 1, -1, -1):
        out[n] = s % M
        s = int(pred[s, tb[n, s]])
    return P[out]


def bcjr_isi_bpsk(y, h, sigma2, La=None):
    """Log-MAP BCJR detector for BPSK (bit 0 -> +1) over a real ISI channel h.

    y: (N,) or (B, N) real observations; sigma2: real noise variance;
    La: a-priori LLRs log P(+1)/P(-1), same shape as y. Returns a-posteriori LLRs.
    Batched over the first axis so many frames run in one pass.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    B, N = y.shape
    La = np.zeros_like(y) if La is None else np.atleast_2d(La)
    P = np.array([1.0, -1.0])
    M, m, S, expected = _isi_tables(h, P)
    expected = expected.real
    ns = np.arange(S)
    nxt = np.array([[i + M * (s % M ** (m - 1)) for i in range(M)] for s in range(S)])  # (S, M)
    # gamma[b, n, s, i]
    gam = np.empty((B, N, S, M))
    for i in range(M):
        gam[:, :, :, i] = (-(y[:, :, None] - expected[None, None, :, i]) ** 2 / (2 * sigma2)
                           + 0.5 * P[i] * La[:, :, None])
    alpha = np.empty((B, N + 1, S))
    alpha[:, 0] = -np.log(S)
    pred = np.array([(ns // M) + j * M ** (m - 1) for j in range(M)]).T
    inp = ns % M
    for n in range(N):
        a = alpha[:, n][:, pred] + gam[:, n][:, pred, inp[:, None]]
        a = np.logaddexp.reduce(a, axis=2)
        alpha[:, n + 1] = a - a.max(axis=1, keepdims=True)
    beta = np.zeros((B, S)) - np.log(S)
    L = np.empty((B, N))
    for n in range(N - 1, -1, -1):
        t = alpha[:, n][:, :, None] + gam[:, n] + beta[:, nxt]       # (B, S, M)
        L[:, n] = np.logaddexp.reduce(t[:, :, 0], axis=1) - np.logaddexp.reduce(t[:, :, 1], axis=1)
        b = np.logaddexp.reduce(gam[:, n] + beta[:, nxt], axis=2)
        beta = b - b.max(axis=1, keepdims=True)
    return L if L.shape[0] > 1 else L[0]


# -------------------------------------------------------------------- THP
def mod_centered(x, A):
    """Reduce x into [-A, A) (real and imaginary parts separately for complex x)."""
    f = lambda v: v - 2 * A * np.floor((v + A) / (2 * A))
    return f(x.real) + 1j * f(x.imag) if np.iscomplexobj(x) else f(x)


def thp_precode(a, h, M):
    """Tomlinson-Harashima precoding of M-PAM symbols (levels +-1, +-3, ...)
    for a causal channel h with h[0] != 0 (normalised internally to h[0] = 1).
    Returns the transmitted sequence x in [-M, M)."""
    h = np.asarray(h, dtype=float) / h[0]
    x = np.zeros(len(a))
    for n in range(len(a)):
        isi = sum(h[k] * x[n - k] for k in range(1, len(h)) if n - k >= 0)
        x[n] = mod_centered(a[n] - isi, M)
    return x
