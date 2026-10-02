"""commlib.mimokit -- diversity, MIMO and antenna-array tools for Chapter 19 and Labs 10/29.

Import explicitly:  ``from commlib import mimokit as mk``.  Everything here builds on
``commlib.mimo`` (which it never modifies) and is vectorised over stacks of channels so that an
interactive lab can evaluate thousands of channel draws per slider movement.

Contents
--------
* Diversity: exponential correlation, correlated Rayleigh branches, SC/EGC/MRC output SNR,
  BPSK error rates (MRC with arbitrary branch eigenvalues by Craig's formula, SC closed form).
* MIMO capacity: eigenvalue stacks, water-filling with the water level, DMT curves.
* Batched detectors for spatial multiplexing: ZF, MMSE, ordered MMSE-SIC and exhaustive ML.
* Multi-user precoding: MR, ZF, RZF and per-user SINR.
* Arrays: tapers, ULA/UPA array factors, beamwidth and sidelobe metrics, Bartlett/MVDR/MUSIC
  spectra with spatial smoothing, DFT (Type I) codebooks, sparse mm-wave channels, OMP hybrid
  precoding, LOS-MIMO channels.

Angles are in radians from broadside unless a name says ``_deg``; SNRs are linear unless a
name ends in ``_db``.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import comb, erfc

C0 = 299_792_458.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def cn(rng, shape):
    """Circularly-symmetric complex Gaussian CN(0, 1) samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def qfunc(x):
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


# ================================================================== diversity

def exp_corr(n, rho):
    """Exponential correlation matrix R[i, k] = rho^|i-k| (real, 0 <= rho < 1)."""
    k = np.arange(n)
    return float(rho) ** np.abs(np.subtract.outer(k, k))


def corr_sqrt(R):
    """A square root S with S S^H = R (eigen-decomposition: safe for singular R)."""
    w, V = np.linalg.eigh(R)
    return V * np.sqrt(np.clip(w, 0, None))


def correlated_branches(L, n, rho, rng):
    """(L, n) CN(0,1) branch gains with exponential correlation rho between neighbours."""
    return corr_sqrt(exp_corr(L, rho)) @ cn(rng, (L, n))


def combine_gain(h, kind):
    """Normalised output SNR of a combiner (multiply by the per-branch mean SNR).

    h: (L, n) branch gains. 'mrc': sum |h|^2; 'egc': (sum |h|)^2 / L; 'sc': max |h|^2."""
    a2 = np.abs(h) ** 2
    if kind == "mrc":
        return a2.sum(axis=0)
    if kind == "egc":
        return np.abs(h).sum(axis=0) ** 2 / h.shape[0]
    if kind == "sc":
        return a2.max(axis=0)
    raise ValueError(kind)


def ber_bpsk_mrc_iid(gbar, L):
    """BPSK, L-branch MRC over i.i.d. Rayleigh, mean SNR per branch gbar (linear)."""
    g = np.asarray(gbar, float)
    mu = np.sqrt(g / (1 + g))
    return ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))


def ber_bpsk_mrc_eig(gbar, lam):
    """BPSK with MRC over Rayleigh branches whose covariance has eigenvalues ``lam`` (mean SNR
    per branch gbar, linear). Craig's formula with the MGF, robust for repeated eigenvalues."""
    g = np.atleast_1d(np.asarray(gbar, float))
    lam = np.asarray(lam, float)
    th = (_GL_X + 1) * np.pi / 4
    s2 = np.sin(th) ** 2
    M = np.prod(1.0 / (1.0 + g[:, None, None] * lam[None, None, :] / s2[None, :, None]), axis=2)
    out = (M @ _GL_W) * (np.pi / 4) / np.pi
    return out if np.ndim(gbar) else float(out[0])


def ber_bpsk_mrc_corr(gbar, L, rho):
    """BPSK, L-branch MRC with exponential correlation rho (exact via the eigenvalues)."""
    return ber_bpsk_mrc_eig(gbar, np.linalg.eigvalsh(exp_corr(L, rho)))


def ber_bpsk_sc_iid(gbar, L):
    """BPSK, L-branch selection combining over i.i.d. Rayleigh.

    The textbook closed form sum_k (-1)^(k+1) C(L,k) (1 - sqrt(g/(g+k)))/2 cancels
    catastrophically at high SNR, so integrate P_b = (1/sqrt(pi)) int (1 - e^(-u^2/g))^L e^(-u^2) du
    instead (exact, stable)."""
    g = np.atleast_1d(np.asarray(gbar, float))
    F = (-np.expm1(-_U[None, :] ** 2 / g[:, None])) ** L
    integ = getattr(np, "trapezoid", None) or np.trapz
    out = integ(F * np.exp(-_U ** 2)[None, :], _U, axis=1) / np.sqrt(np.pi)
    return out if np.ndim(gbar) else float(out[0])


_U = np.linspace(0, 7, 701)


def gain_cdf_asymptote(kind, R):
    """Small-x behaviour F(x) ~ c x^L of a combiner's normalised output SNR for Rayleigh
    branches with covariance R (L x L, full rank). Returns c.

    The density of h at the origin is 1/(pi^L det R); multiply by the volume of the region
    {gain < x}: MRC pi^L x^L / L!, SC (pi x)^L, EGC (2 pi)^L (L x)^L / (2L)!."""
    from math import factorial
    L = R.shape[0]
    det = max(float(np.real(np.linalg.det(R))), 1e-300)
    if kind == "mrc":
        c = 1.0 / factorial(L)
    elif kind == "sc":
        c = 1.0
    else:
        c = (2.0 * L) ** L / factorial(2 * L)
    return c / det


def ber_bpsk_from_gains(G, gbar, c=None, L=1, q_tail=5e-4, f_asym=1e-12):
    """Average BPSK BER over a fading gain distribution known from samples G (normalised
    combiner output SNR, e.g. from ``combine_gain``), for mean SNRs ``gbar`` (linear array).

    P_b = (1/sqrt(pi)) * integral F(u^2/gbar) exp(-u^2) du, with F the empirical CDF of G above
    its ``q_tail`` quantile, the exact asymptote c x^L (``gain_cdf_asymptote``) where it is below
    ``f_asym``, and a log-log bridge in between. This reaches error rates far below 1/len(G),
    where plain Monte Carlo averaging is useless."""
    Gs = np.sort(np.asarray(G, float))
    n = len(Gs)
    i1 = max(int(q_tail * n), 10)
    x1, F1 = Gs[i1], (i1 + 1) / n
    g = np.atleast_1d(np.asarray(gbar, float))
    x = _U[None, :] ** 2 / g[:, None]
    F = np.interp(x, Gs, (np.arange(n) + 1) / n, left=0.0, right=1.0)
    if c is None:                                     # no asymptote: power law of order L
        c = F1 / x1 ** L
    xa = (f_asym / c) ** (1.0 / L)
    lx = np.log(np.maximum(x, 1e-300))
    if xa < x1:
        slope = (np.log(F1) - np.log(f_asym)) / (np.log(x1) - np.log(xa))
        bridge = np.exp(np.log(f_asym) + slope * (lx - np.log(xa)))
        F = np.where(x < x1, np.where(x < xa, c * np.exp(L * lx), bridge), F)
    else:
        F = np.where(x < x1, F1 * np.exp(L * (lx - np.log(x1))), F)
    integ = getattr(np, "trapezoid", None) or np.trapz
    out = integ(F * np.exp(-_U ** 2)[None, :], _U, axis=1) / np.sqrt(np.pi)
    return np.clip(out, 0, 0.5)


def snr_for_ber(snr_db, ber, target=1e-3):
    """Interpolate the SNR (dB) at which a decreasing BER curve crosses ``target`` (NaN if not)."""
    snr_db = np.asarray(snr_db, float)
    ly = np.log10(np.clip(np.asarray(ber, float), 1e-300, None))
    t = np.log10(target)
    idx = np.flatnonzero((ly[:-1] >= t) & (ly[1:] < t))
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    return float(np.interp(t, [ly[i + 1], ly[i]], [snr_db[i + 1], snr_db[i]]))


# ================================================================== capacity

def kron_channels(n, nr, nt, rho_r=0.0, rho_t=0.0, rng=None):
    """Stack (n, nr, nt) of Kronecker-correlated Rayleigh channels (exponential correlation)."""
    rng = np.random.default_rng(rng)
    Hw = cn(rng, (n, nr, nt))
    return corr_sqrt(exp_corr(nr, rho_r)) @ Hw @ corr_sqrt(exp_corr(nt, rho_t)).T


def keyhole_channels(n, nr, nt, rng=None):
    """Keyhole (pinhole) channel: rank one, H = a b^T with independent CN vectors."""
    rng = np.random.default_rng(rng)
    return cn(rng, (n, nr, 1)) @ cn(rng, (n, 1, nt))


def eig_stack(H):
    """Eigenvalues of H H^H (or H^H H, the smaller), descending, for a stack (n, nr, nt)."""
    nr, nt = H.shape[-2:]
    G = H @ np.swapaxes(H.conj(), -1, -2) if nr <= nt else np.swapaxes(H.conj(), -1, -2) @ H
    return np.clip(np.linalg.eigvalsh(G)[..., ::-1], 0, None)


def capacity_from_eigs(lam, snr, nt):
    """Equal-power capacity sum_i log2(1 + snr/nt * lam_i); lam (..., k), snr scalar or (m,)."""
    snr = np.asarray(snr, float)
    if snr.ndim == 0:
        return np.sum(np.log2(1 + snr / nt * lam), axis=-1)
    return np.sum(np.log2(1 + snr[:, None, None] / nt * lam[None]), axis=-1)


def waterfill_level(gains, total_power):
    """Water-filling over parallel channels: returns (powers, water level mu, active count)."""
    g = np.asarray(gains, float)
    order = np.argsort(g)[::-1]
    gs = np.maximum(g[order], 1e-300)
    mu, k = 0.0, 0
    for k in range(len(gs), 0, -1):
        mu = (total_power + np.sum(1 / gs[:k])) / k
        if mu - 1 / gs[k - 1] > 0:
            break
    p = np.maximum(mu - 1 / np.maximum(g, 1e-300), 0)
    return p, mu, int(np.sum(p > 0))


def dmt_optimal(nt, nr):
    """Corner points (k, (nt-k)(nr-k)) of the Zheng-Tse optimal trade-off d*(r)."""
    k = np.arange(0, min(nt, nr) + 1)
    return k, (nt - k) * (nr - k)


def dmt_value(r, nt, nr):
    """d*(r) by linear interpolation between the corner points."""
    k, d = dmt_optimal(nt, nr)
    return float(np.interp(r, k, d))


# ================================================================== batched detection

def batch_zf(H, Y):
    """Zero forcing for a stack: H (n, nr, nt), Y (n, nr) -> (n, nt)."""
    return np.einsum("nij,nj->ni", np.linalg.pinv(H), Y)


def batch_mmse(H, Y, n0):
    """Linear MMSE (unit-energy symbols) for a stack."""
    Hh = np.swapaxes(H.conj(), -1, -2)
    nt = H.shape[-1]
    A = Hh @ H + n0 * np.eye(nt)
    return np.linalg.solve(A, np.einsum("nij,nj->ni", Hh, Y)[..., None])[..., 0]


def slice_to(z, points):
    """Nearest constellation point for every entry of z."""
    idx = np.argmin(np.abs(z[..., None] - points), axis=-1)
    return points[idx]


def batch_mmse_sic(H, Y, n0, points):
    """Ordered MMSE-SIC (V-BLAST): at each stage detect the layer with the smallest MSE,
    slice it, cancel it. Vectorised over the stack."""
    n, nr, nt = H.shape
    H = H.copy()
    Y = Y.copy()
    alive = np.ones((n, nt), bool)
    xhat = np.zeros((n, nt), complex)
    rows = np.arange(n)
    for _ in range(nt):
        Hm = H * alive[:, None, :]                      # removed layers have zero columns
        Hh = np.swapaxes(Hm.conj(), -1, -2)
        A = Hh @ Hm + n0 * np.eye(nt)
        Ainv = np.linalg.inv(A)
        mse = np.real(np.diagonal(Ainv, axis1=1, axis2=2)) * n0
        mse = np.where(alive, mse, np.inf)
        k = np.argmin(mse, axis=1)
        w = Ainv[rows, k, :]                             # (n, nt) row of (H^H H + n0 I)^-1
        z = np.einsum("ni,nij,nj->n", w, Hh, Y)
        s = slice_to(z, points)
        xhat[rows, k] = s
        Y = Y - H[rows, :, k] * s[:, None]
        alive[rows, k] = False
    return xhat


def ml_candidates(points, nt):
    """All |points|^nt candidate vectors, shape (nt, M^nt)."""
    return np.array(list(itertools.product(points, repeat=nt))).T


def batch_ml(H, Y, cands, chunk=2000):
    """Exhaustive ML for a stack, given precomputed candidates (nt, K)."""
    out = np.empty((H.shape[0], cands.shape[0]), complex)
    for i in range(0, H.shape[0], chunk):
        Hc, Yc = H[i:i + chunk], Y[i:i + chunk]
        d = np.sum(np.abs(Yc[:, :, None] - Hc @ cands[None]) ** 2, axis=1)
        out[i:i + chunk] = cands[:, np.argmin(d, axis=1)].T
    return out


# ================================================================== multi-user precoding

def mu_precoder(H, kind, rho):
    """Downlink precoder for user channels H (K, M) (rows = users), unit-norm columns.

    kind: 'MR' (H^H), 'ZF' (H^H (H H^H)^-1) or 'RZF' (H^H (H H^H + K/rho I)^-1)."""
    K = H.shape[-2]
    Hh = np.swapaxes(H.conj(), -1, -2)
    if kind == "MR":
        W = Hh
    elif kind == "ZF":
        W = Hh @ np.linalg.inv(H @ Hh)
    else:
        W = Hh @ np.linalg.inv(H @ Hh + K / rho * np.eye(K))
    return W / np.linalg.norm(W, axis=-2, keepdims=True)


def mu_sinr(H, W, rho):
    """Per-user SINR with total power rho split equally over K users (noise variance 1).
    Returns (sinr, signal, interference) with shapes (..., K)."""
    K = H.shape[-2]
    G = np.abs(H @ W) ** 2 * rho / K
    sig = np.diagonal(G, axis1=-2, axis2=-1)
    itf = G.sum(axis=-1) - sig
    return sig / (itf + 1), sig, itf


def mu_sum_rate(M, K, snr_db, kind, rng, trials=150, H=None):
    """Average sum rate (b/s/Hz) of MR/ZF/RZF over i.i.d. Rayleigh channels (as in the book)."""
    rho = 10 ** (snr_db / 10)
    if H is None:
        H = cn(rng, (trials, K, M))
    W = mu_precoder(H, kind, rho)
    s, _, _ = mu_sinr(H, W, rho)
    return float(np.mean(np.sum(np.log2(1 + s), axis=-1)))


# ================================================================== arrays

def taper(n, kind="Uniform", sll_db=30.0):
    """Amplitude taper: 'Uniform', 'Chebyshev', 'Taylor' (nbar 4) or 'Hann'; peak 1."""
    from scipy.signal.windows import chebwin, taylor, hann
    if n < 2 or kind == "Uniform":
        return np.ones(n)
    if kind == "Chebyshev":
        w = chebwin(n, sll_db)
    elif kind == "Taylor":
        w = taylor(n, nbar=4, sll=sll_db)
    else:
        w = hann(n + 2)[1:-1]
    return w / w.max()


def ula_af(w, theta, d=0.5, theta0=0.0):
    """Linear array factor |sum w_n exp(j 2 pi d n (sin th - sin th0))|^2, normalised to the
    uniform peak (N^2 for unit weights): returns linear power gain over one element."""
    n = np.arange(len(w))
    u = np.sin(np.atleast_1d(theta)) - np.sin(theta0)
    af = np.abs(np.exp(2j * np.pi * d * np.outer(u, n)) @ w) ** 2
    return af / np.sum(np.abs(w) ** 2)


def af_u(w, u, d=0.5, u0=0.0):
    """Array factor (linear, gain over one element) versus u = sin(theta), not limited to |u| <= 1."""
    n = np.arange(len(w))
    af = np.abs(np.exp(2j * np.pi * d * np.outer(np.asarray(u) - u0, n)) @ w) ** 2
    return af / np.sum(np.abs(w) ** 2)


def beam_metrics(theta, g_db):
    """Main-lobe direction, half-power beamwidth and peak sidelobe (dB below the peak) of a
    pattern sampled on a fine angle grid (radians). Returns (peak_rad, hpbw_rad, psl_db)."""
    g = np.asarray(g_db, float)
    i = int(np.argmax(g))
    pk = g[i]
    lo = i
    while lo > 0 and g[lo] > pk - 3:
        lo -= 1
    hi = i
    while hi < len(g) - 1 and g[hi] > pk - 3:
        hi += 1
    hpbw = theta[hi] - theta[lo]
    # main lobe = between the first nulls (local minima) either side
    a = i
    while a > 0 and g[a - 1] < g[a]:
        a -= 1
    b = i
    while b < len(g) - 1 and g[b + 1] < g[b]:
        b += 1
    side = np.r_[g[:a], g[b + 1:]]
    psl = float(side.max() - pk) if side.size else -np.inf
    return float(theta[i]), float(hpbw), psl


def grating_lobes(d, theta0):
    """Directions (radians) of grating lobes in visible space for spacing d (wavelengths)."""
    u0 = np.sin(theta0)
    out = []
    for m in range(-6, 7):
        if m == 0:
            continue
        u = u0 + m / d
        if abs(u) <= 1:
            out.append(np.arcsin(u))
    return np.array(out)


def upa_af_uv(nx, ny, d, u0, v0, wx=None, wy=None, n=181):
    """Planar-array factor (dB, peak 0) on a (u, v) direction-cosine grid; NaN outside |u,v|>1."""
    u = np.linspace(-1, 1, n)
    wx = np.ones(nx) if wx is None else wx
    wy = np.ones(ny) if wy is None else wy
    ax = af_u(wx, u, d, u0)
    ay = af_u(wy, u, d, v0)
    af = np.outer(ay, ax)
    out = 10 * np.log10(af / af.max() + 1e-12)
    U, V = np.meshgrid(u, u)
    out[U ** 2 + V ** 2 > 1] = np.nan
    return u, out


def sample_cov(X):
    return X @ X.conj().T / X.shape[1]


def smooth_cov(R, sub):
    """Forward spatial smoothing: average the covariances of the overlapping sub-arrays of
    length ``sub`` (restores rank against coherent sources)."""
    N = R.shape[0]
    L = N - sub + 1
    return sum(R[i:i + sub, i:i + sub] for i in range(L)) / L


def doa_spectra(R, theta, n_src, d=0.5):
    """Bartlett, MVDR (Capon) and MUSIC pseudo-spectra (linear) on a grid of angles."""
    N = R.shape[0]
    n = np.arange(N)[:, None]
    a = np.exp(-2j * np.pi * d * n * np.sin(theta)[None, :])
    bart = np.real(np.sum(a.conj() * (R @ a), axis=0)) / N
    Ri = np.linalg.inv(R + 1e-9 * np.trace(R).real / N * np.eye(N))
    mvdr = 1.0 / np.real(np.sum(a.conj() * (Ri @ a), axis=0))
    w, V = np.linalg.eigh(R)
    En = V[:, :max(N - n_src, 1)]
    music = 1.0 / np.maximum(np.sum(np.abs(En.conj().T @ a) ** 2, axis=0), 1e-12)
    return bart, mvdr, music, w[::-1]


def find_peaks_db(theta, p_db, k, min_drop=3.0):
    """The k highest peaks of a spectrum with a topographic prominence of at least
    ``min_drop`` dB (scipy.signal.find_peaks). Returns angles (radians), sorted."""
    from scipy.signal import find_peaks
    p = np.asarray(p_db, float)
    idx, props = find_peaks(np.r_[-1e9, p, -1e9], prominence=min_drop)
    idx = idx - 1
    good = sorted(idx, key=lambda i: -p[i])[:k]
    return np.sort(np.asarray(theta)[good]) if good else np.array([])


def dft_beams(n, n_beams, oversample=1):
    """DFT codebook beams (columns, unit norm) for an n-element lambda/2 ULA; beam m points at
    u = 2m/(n*O) (wrapped into [-1, 1))."""
    m = np.arange(n_beams)
    u = ((2 * m / (n * oversample)) + 1) % 2 - 1
    return np.exp(-1j * np.pi * np.outer(np.arange(n), u)) / np.sqrt(n), u


def sparse_channel(nt, nr, n_cl, n_ray, rng, spread_deg=5.0):
    """Clustered mm-wave channel between ULAs (El Ayach et al. 2014 style).
    Returns (H, At, angles_t) with At the transmit steering vectors of the rays."""
    th_c = rng.uniform(-np.pi / 2, np.pi / 2, (2, n_cl))
    H = np.zeros((nr, nt), complex)
    At, ang = [], []
    for c in range(n_cl):
        for _ in range(n_ray):
            tt = th_c[0, c] + np.deg2rad(spread_deg) * rng.laplace() / np.sqrt(2)
            tr = th_c[1, c] + np.deg2rad(spread_deg) * rng.laplace() / np.sqrt(2)
            at = np.exp(-1j * np.pi * np.arange(nt) * np.sin(tt)) / np.sqrt(nt)
            ar = np.exp(-1j * np.pi * np.arange(nr) * np.sin(tr)) / np.sqrt(nr)
            H += cn(rng, 1)[0] * np.outer(ar, at.conj())
            At.append(at)
            ang.append(tt)
    return H * np.sqrt(nt * nr / (n_cl * n_ray)), np.array(At).T, np.array(ang)


def omp_precoder(Fopt, At, n_rf):
    """Orthogonal matching pursuit hybrid precoder: Frf (columns of At), Fbb (least squares),
    normalised so that ||Frf Fbb||_F^2 = Ns. Returns (Frf, Fbb, chosen indices)."""
    Fres, idx = Fopt.copy(), []
    Fbb = None
    Frf = At[:, :0]
    for _ in range(n_rf):
        k = int(np.argmax(np.sum(np.abs(At.conj().T @ Fres) ** 2, axis=1)))
        idx.append(k)
        Frf = At[:, idx]
        Fbb = np.linalg.lstsq(Frf, Fopt, rcond=None)[0]
        Fres = Fopt - Frf @ Fbb
        nrm = np.linalg.norm(Fres)
        if nrm < 1e-12:
            break
        Fres = Fres / nrm
    Fbb = Fbb * np.sqrt(Fopt.shape[1]) / np.linalg.norm(Frf @ Fbb)
    return Frf, Fbb, idx


def spectral_eff(H, F, snr, ns):
    """log2 det(I + snr/Ns H F F^H H^H) (optimal receiver)."""
    M = np.eye(H.shape[0]) + snr / ns * H @ F @ F.conj().T @ H.conj().T
    return float(np.real(np.log2(np.linalg.det(M))))


def los_channel(n, d, R, lam):
    """LOS channel between two broadside n-element arrays of spacing d at range R (spherical
    wavefronts, unit-modulus entries)."""
    y = (np.arange(n) - (n - 1) / 2) * d
    dist = np.sqrt(R ** 2 + np.subtract.outer(y, y) ** 2)
    return np.exp(-2j * np.pi * dist / lam)


def los_opt_spacing(n, R, lam):
    """Rayleigh spacing for equal transmit and receive arrays: d = sqrt(lam R / n)."""
    return np.sqrt(lam * R / n)
