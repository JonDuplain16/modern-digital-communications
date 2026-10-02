"""commlib.frontier -- small closed-form helpers for Chapter 25 ("The Road to 6G").

Companion to :mod:`commlib.sixg` (waveforms, arrays, RIS and radar processing). These are the
formulas that the chapter's figure script (``book/figscripts/ch25_figs.py``) and the frontier labs
share, so that a slider in a lab lands on the same number as the book:

* ``gas_atten_db_km``      approximate clear-air specific attenuation (oxygen + water vapour),
                           the simplified ITU-R P.676 formulas used for Figure 25 (absorption);
* ``pn_psd_dbc``, ``pn_ici_snr_db``   a toy PLL phase-noise profile and the SNR ceiling its
                           inter-carrier interference sets once the common phase error is removed;
* ``depth_of_focus``       the Bjornson-Demir-Sanguinetti estimate of the 3 dB focal region;
* ``ris_phase_loss_db``    coherent-gain loss of b-bit RIS phase quantisation;
* ``isac_floor_db``        the data-induced sidelobe floor (kappa - 1)/N of a correlation radar;
* ``dd_grid_numbers``      OTFS grid sizing for a given speed, carrier and numerology.

Import explicitly:  ``from commlib import frontier as fr``.
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ================================================================== molecular absorption
def gas_atten_db_km(f_ghz, rho=7.5):
    """Approximate sea-level specific attenuation (dB/km) of oxygen and water vapour at
    ``f_ghz`` (GHz) for a water-vapour density ``rho`` (g/m^3). Simplified formulas from earlier
    editions of ITU-R P.676 (valid roughly 1-350 GHz); identical to the Chapter 25 figure.
    Returns (gamma_oxygen, gamma_water)."""
    f = np.atleast_1d(np.asarray(f_ghz, float))
    go = np.zeros_like(f)
    lo = f < 57
    hi = f > 63
    mid = ~lo & ~hi
    fl = f[lo]
    go[lo] = (7.19e-3 + 6.09 / (fl ** 2 + 0.227) + 4.81 / ((fl - 57) ** 2 + 1.50)) * fl ** 2 * 1e-3
    fh = f[hi]
    go[hi] = (3.79e-7 * fh + 0.265 / ((fh - 63) ** 2 + 1.59) + 0.028 / ((fh - 118) ** 2 + 1.47)) \
        * (fh + 198) ** 2 * 1e-3
    fm = f[mid]
    x57 = (7.19e-3 + 6.09 / (57 ** 2 + 0.227) + 4.81 / 1.5) * 57 ** 2 * 1e-3
    x63 = (3.79e-7 * 63 + 0.265 / 1.59 + 0.028 / ((63 - 118) ** 2 + 1.47)) * 261 ** 2 * 1e-3
    A = np.array([[57 ** 2, 57, 1], [60 ** 2, 60, 1], [63 ** 2, 63, 1]])
    cc = np.linalg.solve(A, [x57, 15.0, x63])
    go[mid] = cc[0] * fm ** 2 + cc[1] * fm + cc[2]
    gw = (0.050 + 0.0021 * rho + 3.6 / ((f - 22.2) ** 2 + 8.5) + 10.6 / ((f - 183.3) ** 2 + 9.0)
          + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f ** 2 * rho * 1e-4
    if np.ndim(f_ghz) == 0:
        return float(go[0]), float(gw[0])
    return go, gw


# ================================================================== phase noise
def pn_psd_dbc(f, fc, L1M_28=-100.0, f_loop=500e3, floor=-150.0):
    """Toy PLL phase-noise profile L(f) (dBc/Hz): flat inside the loop bandwidth, -20 dB/decade
    outside, plus a white floor; -100 dBc/Hz at 1 MHz for a 28 GHz synthesiser, scaled by
    20 log10(fc / 28 GHz) (frequency multiplication)."""
    f = np.asarray(f, float)
    L = L1M_28 + 20 * np.log10(fc / 28e9)
    prof = L + np.where(f < f_loop, 0.0, -20 * np.log10(np.maximum(f, 1e-9) / f_loop))
    return 10 * np.log10(10 ** (prof / 10) + 10 ** ((floor + 20 * np.log10(fc / 28e9)) / 10))


def pn_ici_snr_db(fc, scs, L1M_28=-100.0, f_loop=500e3):
    """SNR ceiling (dB) set by phase-noise inter-carrier interference in OFDM once the common
    phase error has been removed: 1 / (2 int L(f) (1 - sinc^2(f T)) df), T = 1/scs."""
    f = np.logspace(2, 9, 4000)
    S = 10 ** (pn_psd_dbc(f, fc, L1M_28, f_loop) / 10)
    w = 1 - np.sinc(f / scs) ** 2
    return float(-10 * np.log10(2 * _trapz(S * w, f)))


# ================================================================== arrays, RIS, ISAC
def depth_of_focus(d_r, r_f):
    """Approximate 3 dB focal region (near, far) in metres of an array with Rayleigh distance
    ``d_r`` focused at ``r_f``. The far edge is infinite once r_f >= d_r / 10."""
    near = d_r * r_f / (d_r + 10 * r_f)
    far = d_r * r_f / (d_r - 10 * r_f) if d_r > 10 * r_f else np.inf
    return near, far


def ris_phase_loss_db(bits):
    """Coherent-gain loss (dB, positive) of uniform b-bit phase quantisation: -20 log10 sinc(2^-b)."""
    return float(-20 * np.log10(np.sinc(2.0 ** -np.asarray(bits, float))))


def isac_floor_db(points, n):
    """Data-induced sidelobe floor of a correlation (matched-filter) OFDM radar, in dB relative to
    the peak: (kappa - 1) / n with kappa = E|x|^4 / (E|x|^2)^2. Constant-modulus data gives -inf."""
    p = np.asarray(points)
    kappa = np.mean(np.abs(p) ** 4) / np.mean(np.abs(p) ** 2) ** 2
    return float(10 * np.log10(kappa - 1)) - 10 * np.log10(n) if kappa > 1 + 1e-12 else -np.inf


# ================================================================== OTFS grid sizing
def dd_grid_numbers(speed_kmh, fc, scs, M, N, tau_s):
    """Delay-Doppler numbers for a mobile at ``speed_kmh`` on carrier ``fc`` (Hz) with subcarrier
    spacing ``scs`` (Hz), an M x N grid and delay spread ``tau_s`` (s), CP ignored (Chapter 25).
    Returns a dict: nu_max (Hz), nu_T (= nu_max/scs), sir_db (OFDM signal-to-ICI, approx.
    -10log((pi nu T)^2/3)), delay_res (s), doppler_res (Hz), l_max and k_max (bins), frame (s)
    and guard (fraction of the grid taken by an embedded-pilot guard (2l+1)(4k+1))."""
    v = speed_kmh / 3.6
    nu = v * fc / C0
    T = 1 / scs
    nuT = nu * T
    sir = -10 * np.log10(max((np.pi * nuT) ** 2 / 3, 1e-30))
    l_max = tau_s * M * scs
    k_max = nu * N * T
    guard = (2 * round(l_max) + 1) * (4 * np.ceil(k_max) + 1) / (M * N)
    return dict(nu_max=nu, nu_T=nuT, sir_db=sir, delay_res=1 / (M * scs), doppler_res=1 / (N * T),
                l_max=l_max, k_max=k_max, frame=N * T, guard=min(guard, 1.0))


# ================================================================== OFDM on a DD channel
def ofdm_dd_frame(M, N, cp, l, nu, h, x, n0, rng, ici_aware=False):
    """N CP-OFDM symbols of M subcarriers (data ``x``, length M*N) through the doubly dispersive
    channel (l, nu, h) of :mod:`commlib.sixg` with complex noise power ``n0``; returns the
    equalised symbols. One-tap: divide by each symbol's average channel (the diagonal of
    F H F^H); ICI-aware: LMMSE with the full per-symbol matrix. Same receivers as Figure 25
    (otfs-ber)."""
    from . import sixg
    X = np.asarray(x).reshape(N, M)
    s = np.fft.ifft(X, axis=1) * np.sqrt(M)
    s = np.concatenate([s[:, -cp:], s], axis=1).ravel()
    r = sixg.apply_dd_channel(s, l, nu, h)
    r = r + np.sqrt(n0 / 2) * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))
    Y = np.fft.fft(r.reshape(N, M + cp)[:, cp:], axis=1) / np.sqrt(M)
    F = sixg.dft_matrix(M)
    eye = np.eye(M)
    out = np.empty((N, M), complex)
    for i in range(N):
        n0i = i * (M + cp) + cp
        Ht = np.zeros((M, M), complex)
        for lp, vp, hp in zip(l, nu, h):
            Ht += hp * np.exp(2j * np.pi * vp * (n0i + np.arange(M)))[:, None] * np.roll(eye, int(lp), axis=0)
        G = F @ Ht @ F.conj().T
        out[i] = sixg.lmmse(G, Y[i], n0) if ici_aware else Y[i] / np.diag(G)
    return out.ravel()


# ================================================================== a tiny neural network
class MLP:
    """A small multilayer perceptron (ReLU hidden layers, linear output) with softmax
    cross-entropy and Adam, in plain NumPy: the learned demapper and autoencoder of Chapter 25.

        net = MLP([4, 32, 16], rng)
        loss, g_in = net.train_step(X, labels, lr=3e-3)    # one Adam step, returns dL/dX too
        probs = net.predict_proba(X)
    """

    def __init__(self, sizes, rng=None):
        rng = np.random.default_rng(rng)
        self.sizes = list(sizes)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        self._a = [x]
        self._z = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self._a[-1] @ W + b
            self._z.append(z)
            self._a.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return self._a[-1]

    def predict_proba(self, x):
        z = self.forward(x)
        p = np.exp(z - z.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def train_step(self, x, labels, lr=3e-3):
        """One Adam step on the mean cross-entropy. Returns (loss, gradient w.r.t. the input)."""
        p = self.predict_proba(x)
        n = len(labels)
        loss = float(-np.mean(np.log(p[np.arange(n), labels] + 1e-12)))
        g = p
        g[np.arange(n), labels] -= 1
        g /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (self._z[i] > 0)
            gW[i] = self._a[i].T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
        self.t += 1
        for k, (prm, gr) in enumerate(zip(self.W + self.b, gW + gb)):
            self._m[k] = 0.9 * self._m[k] + 0.1 * gr
            self._v[k] = 0.999 * self._v[k] + 0.001 * gr * gr
            prm -= lr * (self._m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self._v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        return loss, g


class Adam:
    """Adam optimiser state for one free array (e.g. learned constellation points)."""

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, prm, grad, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        prm -= lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        return prm


def _selftest():
    rng = np.random.default_rng(0)
    net = MLP([2, 16, 4], rng)
    X = rng.standard_normal((512, 2))
    lab = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0)
    for _ in range(300):
        loss, _ = net.train_step(X, lab, 1e-2)
    assert np.mean(np.argmax(net.forward(X), 1) == lab) > 0.95, loss
    go, gw = gas_atten_db_km(60.0)
    assert 10 < go + gw < 20
    g300 = sum(gas_atten_db_km(300.0))
    assert 2 < g300 < 10, g300
    assert abs(ris_phase_loss_db(2) - 0.91) < 0.01 and abs(ris_phase_loss_db(1) - 3.92) < 0.01
    near, far = depth_of_focus(351.0, 6.0)
    assert abs(near - 5.1) < 0.1 and abs(far - 7.2) < 0.1
    d = dd_grid_numbers(500, 4e9, 30e3, 512, 64, 5e-6)
    assert abs(d["nu_max"] - 1852) < 2 and abs(d["frame"] - 2.133e-3) < 1e-5
    assert np.isinf(isac_floor_db(np.exp(2j * np.pi * np.arange(16) / 16), 1024))
    assert pn_ici_snr_db(28e9, 120e3) > pn_ici_snr_db(28e9, 15e3)
    print("frontier self-test passed")


if __name__ == "__main__":
    _selftest()
