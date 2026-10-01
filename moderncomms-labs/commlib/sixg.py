"""commlib.sixg -- toy models of candidate 6G physical-layer techniques.

Used by Chapter 25 ("The Road to 6G") figures and by the frontier labs.
Import explicitly:  ``from commlib import sixg``.

Everything here is deliberately small and written to be read:

* doubly-dispersive channels on a cyclic-prefixed block (integer delays, fractional Doppler);
* OTFS (ISFFT + OFDM "Heisenberg" transform with rectangular pulses) and AFDM (discrete affine
  Fourier transform) modulators, their effective channel matrices and an LMMSE detector;
* near-field (spherical-wave) and far-field (plane-wave) array responses, Rayleigh distance;
* RIS and relay link models in free space;
* OFDM radar (range-Doppler) processing.

Units: delays in samples, Doppler in cycles per sample (so a Doppler of one OFDM subcarrier
spacing on an M-point DFT is 1/M), distances in metres, frequencies in hertz.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


# ================================================================== doubly-dispersive channel
def random_dd_paths(n_paths, max_delay, max_doppler, rng, fractional=True, scale=1.0):
    """Random multipath with integer delays 0..max_delay (first path at 0) and Doppler in
    [-max_doppler, max_doppler] (cycles/sample).  With ``fractional=False`` the Doppler values are
    rounded to multiples of ``scale`` (e.g. the OTFS Doppler resolution 1/(MN)).
    Gains are i.i.d. complex Gaussian with equal power, total power one (Rayleigh)."""
    l = np.sort(rng.integers(0, max_delay + 1, n_paths)); l[0] = 0
    # Jakes-like Doppler: maximum Doppler times cos of a uniform angle
    nu = max_doppler * np.cos(rng.uniform(0, 2 * np.pi, n_paths))
    if not fractional:
        nu = np.round(nu / scale) * scale
    h = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2 * n_paths)
    return l, nu, h


def dd_channel_matrix(l, nu, h, L):
    """Time-domain channel matrix for an L-sample block protected by a cyclic prefix
    (so delays wrap circularly).  Path p contributes  h_p * diag(exp(j 2 pi nu_p n)) * Pi^{l_p}."""
    n = np.arange(L)
    H = np.zeros((L, L), dtype=complex)
    eye = np.eye(L)
    for lp, vp, hp in zip(l, nu, h):
        H += hp * np.exp(2j * np.pi * vp * n)[:, None] * np.roll(eye, int(lp), axis=0)
    return H


def apply_dd_channel(s, l, nu, h, cp=0):
    """Apply a doubly-dispersive channel to a (prefixed) time signal by direct summation.
    The first ``cp`` samples are the prefix; the time index of the Doppler phase runs over the
    whole signal.  Delays are applied as linear shifts (the prefix absorbs them)."""
    n = np.arange(len(s))
    r = np.zeros(len(s), dtype=complex)
    for lp, vp, hp in zip(l, nu, h):
        lp = int(lp)
        shifted = np.concatenate([np.zeros(lp, complex), s[:len(s) - lp]]) if lp else s
        r += hp * np.exp(2j * np.pi * vp * n) * shifted
    return r


# ================================================================== OTFS
def dft_matrix(N):
    """Unitary DFT matrix F_N (F @ x = fft(x)/sqrt(N))."""
    return np.fft.fft(np.eye(N)) / np.sqrt(N)


def isfft(X_dd):
    """Inverse symplectic finite Fourier transform: M x N delay-Doppler grid -> time-frequency grid.
    X_tf = F_M X_dd F_N^H (unitary)."""
    M, N = X_dd.shape
    return np.fft.ifft(np.fft.fft(X_dd, axis=0), axis=1) * np.sqrt(N) / np.sqrt(M)


def sfft(X_tf):
    """Symplectic finite Fourier transform (inverse of :func:`isfft`)."""
    M, N = X_tf.shape
    return np.fft.fft(np.fft.ifft(X_tf, axis=0), axis=1) * np.sqrt(M) / np.sqrt(N)


def otfs_modulate(X_dd):
    """OTFS with rectangular pulses and one prefix per frame: ISFFT, then an M-point IDFT per column
    (the OFDM 'Heisenberg' transform).  The two steps collapse to s = vec(X_dd F_N^H), i.e. an
    N-point IDFT along Doppler only.  Returns the length-MN time signal (no prefix)."""
    M, N = X_dd.shape
    X_tf = isfft(X_dd)
    S = np.fft.ifft(X_tf, axis=0) * np.sqrt(M)        # Heisenberg: one OFDM symbol per column
    return S.reshape(-1, order="F")


def otfs_demodulate(r, M, N):
    """Wigner transform (M-point DFT per symbol) followed by the SFFT; returns the M x N DD grid."""
    R = r.reshape(M, N, order="F")
    Y_tf = np.fft.fft(R, axis=0) / np.sqrt(M)
    return sfft(Y_tf)


def otfs_matrix(M, N):
    """MN x MN unitary modulation matrix A with s = A @ vec(X_dd) (column-major vec)."""
    return np.kron(dft_matrix(N).conj().T, np.eye(M))


# ================================================================== AFDM
def afdm_matrix(N, c1, c2=0.0):
    """Discrete affine Fourier transform matrix A = Lambda_c2 F Lambda_c1, with
    Lambda_c = diag(exp(-j 2 pi c n^2)).  AFDM transmits s = A^H x and the receiver computes A r."""
    n = np.arange(N)
    L1 = np.exp(-2j * np.pi * c1 * n ** 2)
    L2 = np.exp(-2j * np.pi * c2 * n ** 2)
    return L2[:, None] * dft_matrix(N) * L1[None, :]


def afdm_c1(N, max_doppler_bins, guard=1):
    """Chirp rate c1 = (2(alpha_max + xi) + 1) / (2N) of Bemani, Ksairi and Kountouris (2023),
    with alpha_max the maximum Doppler in DFT bins (1/N cycles/sample) and xi a guard for
    fractional Doppler."""
    return (2 * (max_doppler_bins + guard) + 1) / (2 * N)


# ================================================================== detection
def lmmse(H, y, n0):
    """Linear MMSE estimate of x (unit-power i.i.d. symbols) from y = H x + w, E|w|^2 = n0."""
    K = H.shape[1]
    return np.linalg.solve(H.conj().T @ H + n0 * np.eye(K), H.conj().T @ y)


# ================================================================== arrays: near and far field
def ula_positions(N, d):
    """Element x-positions (m) of an N-element ULA of spacing d centred on the origin."""
    return (np.arange(N) - (N - 1) / 2) * d


def farfield_response(N, d, fc, theta):
    """Plane-wave array response (unit-modulus entries) towards angle theta (rad from broadside),
    with the same sign convention as :func:`nearfield_response` (path length r - x sin(theta)),
    so the two agree when r is much larger than the Rayleigh distance."""
    lam = C0 / fc
    x = ula_positions(N, d)
    return np.exp(2j * np.pi * x * np.sin(theta) / lam)


def nearfield_response(N, d, fc, r, theta):
    """Spherical-wave array response for a point at range r (m) and angle theta (rad) measured
    from the array centre; phases are referenced to the centre (exact distances, no Fresnel
    approximation).  Amplitude variation across the aperture is ignored."""
    lam = C0 / fc
    x = ula_positions(N, d)
    px, py = r * np.sin(theta), r * np.cos(theta)
    dist = np.sqrt((px - x) ** 2 + py ** 2)
    return np.exp(-2j * np.pi * (dist - r) / lam)


def field_map(weights, d, fc, X, Y):
    """Normalised |array factor|^2 at points (X, Y) (array on the x axis, boresight +y) for
    element weights ``weights`` -- spherical waves, 1/r amplitude ignored (pure focusing gain)."""
    lam = C0 / fc
    x = ula_positions(len(weights), d)
    acc = np.zeros(X.shape, dtype=complex)
    for xi, wi in zip(x, weights):
        acc += wi * np.exp(-2j * np.pi * np.sqrt((X - xi) ** 2 + Y ** 2) / lam)
    return np.abs(acc) ** 2 / (np.sum(np.abs(weights)) ** 2)


def rayleigh_distance(D, fc):
    """Fraunhofer (Rayleigh) distance 2 D^2 / lambda for an aperture of size D (m)."""
    return 2 * np.asarray(D, float) ** 2 / (C0 / fc)


# ================================================================== RIS and relays (free space)
def fspl_gain(d, fc, gt=1.0, gr=1.0):
    """Friis power gain gt*gr*(lambda / 4 pi d)^2 (linear)."""
    lam = C0 / fc
    return gt * gr * (lam / (4 * np.pi * np.asarray(d, float))) ** 2


def ris_gain(n_elements, fc, d1, d2, gt=1.0, gr=1.0, elem_area=None, cos_in=1.0, cos_out=1.0):
    """Far-field power gain of the path Tx -> RIS -> Rx through an optimally phased RIS of
    ``n_elements`` elements each of area ``elem_area`` (default (lambda/2)^2):
        G = gt gr (N A)^2 cos_in cos_out / ((4 pi)^2 d1^2 d2^2).
    This is the 'product-distance' law: the RIS behaves like an aperture of area N A that
    first captures and then re-radiates the wave."""
    lam = C0 / fc
    A = (lam / 2) ** 2 if elem_area is None else elem_area
    return gt * gr * (n_elements * A) ** 2 * cos_in * cos_out / ((4 * np.pi) ** 2 * d1 ** 2 * d2 ** 2)


# ================================================================== OFDM radar
def ofdm_radar_map(Y, X, window=True):
    """Range-Doppler map from a received OFDM grid Y (subcarriers x symbols) and the known
    transmitted grid X: divide out the data, window, IDFT over subcarriers (range) and DFT over
    symbols (Doppler, fftshifted).  Returns the complex map (range bins x Doppler bins)."""
    Z = Y / X
    M, N = Z.shape
    if window:
        Z = Z * np.hanning(M)[:, None] * np.hanning(N)[None, :]
    return np.fft.fftshift(np.fft.fft(np.fft.ifft(Z, axis=0), axis=1), axes=1)


def ofdm_radar_echo(X, targets, df, Ts, fc, rng=None, n0=0.0):
    """Echo grid Y[m,n] = sum_t a_t X[m,n] exp(-j2 pi m df 2R/c) exp(j2 pi n Ts 2 v fc / c) + noise.
    ``targets`` is a list of (range_m, velocity_mps, amplitude)."""
    M, N = X.shape
    m = np.arange(M)[:, None]; n = np.arange(N)[None, :]
    Y = np.zeros_like(X, dtype=complex)
    for R, v, a in targets:
        Y += a * X * np.exp(-2j * np.pi * m * df * 2 * R / C0) * np.exp(2j * np.pi * n * Ts * 2 * v * fc / C0)
    if n0 > 0:
        rng = np.random.default_rng() if rng is None else rng
        Y += np.sqrt(n0 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return Y


def _selftest():
    rng = np.random.default_rng(1)
    M, N = 8, 6
    X = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    assert np.allclose(sfft(isfft(X)), X)
    s = otfs_modulate(X)
    assert np.allclose(s, otfs_matrix(M, N) @ X.reshape(-1, order="F"))
    assert np.allclose(otfs_demodulate(s, M, N), X)
    A = afdm_matrix(16, 0.1, 0.01)
    assert np.allclose(A @ A.conj().T, np.eye(16))
    assert abs(rayleigh_distance(1.0, 30e9) - 200.0) < 0.2
    print("sixg self-test passed")


if __name__ == "__main__":
    _selftest()
