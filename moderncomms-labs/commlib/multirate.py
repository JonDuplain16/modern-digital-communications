"""commlib.multirate -- multirate building blocks: polyphase filters, CIC, NCO, CORDIC, channelizer.

Companion module for Chapter 6 (Digital Filters and Multirate Processing) and Lab 17. The
algorithms are the ones used by book/figscripts/ch06_figs.py, collected here (vectorised and
bit-true where it matters) so that the lab and the book agree.
Import explicitly:  ``from commlib import multirate as mr``.

Frequencies are in cycles per sample unless stated otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sg

__all__ = ["polyphase_decimate", "polyphase_interpolate", "cic_decimate", "cic_response",
           "cic_bits", "cic_comp_taps", "quantize_coefs", "nco", "nco_sfdr", "cordic",
           "cordic_trajectory", "cordic_gain", "pfb_channelizer", "freq_db"]


# ----------------------------------------------------------------------------- polyphase
def polyphase_decimate(x, h, M):
    """y[n] = sum_k h[k] x[nM - k] (filter, keep every M-th output), computed with M
    sub-filters h[k::M] running at the low rate. Identical to np.convolve(x, h)[::M]."""
    x = np.asarray(x)
    h = np.asarray(h)
    L = int(np.ceil(len(h) / M)) * M
    hp = np.r_[h, np.zeros(L - len(h))].reshape(-1, M).T          # hp[k] = h[k::M]
    xp = np.r_[x, np.zeros((-len(x)) % M + M, dtype=x.dtype)]
    n_in = len(xp) // M
    y = np.zeros(n_in + hp.shape[1] - 1, dtype=np.result_type(x, h, float))
    for k in range(M):
        xk = xp[::M] if k == 0 else np.r_[np.zeros(1, dtype=xp.dtype), xp[M - k::M]][:n_in]
        y += np.convolve(xk, hp[k])
    return y[:-(-(len(x) + len(h) - 1) // M)]


def polyphase_interpolate(x, h, L):
    """Zero-stuff by L and filter with h, computed as L sub-filters h[k::L] at the input rate.
    Identical to np.convolve(upsampled, h) (truncated to len(x)*L + len(h) - 1 rounded up)."""
    x = np.asarray(x)
    phases = [np.convolve(x, h[k::L]) for k in range(L)]
    n = max(len(p) for p in phases)
    y = np.zeros(n * L, dtype=np.result_type(x, h, float))
    for k, p in enumerate(phases):
        y[k:k + len(p) * L:L] = p
    return y


# ----------------------------------------------------------------------------- CIC
def cic_bits(b_in, R, N, D=1):
    """Register width (bits) for a CIC decimator: B_in + ceil(N log2(R D))."""
    return int(b_in + np.ceil(N * np.log2(R * D) - 1e-12))


def cic_decimate(x_int, R, N, D=1, width=None):
    """Bit-true integer CIC decimator: N integrators at the input rate, downsample by R, N combs
    (differential delay D) at the output rate. With ``width`` the registers are two's-complement
    words of that many bits and wrap on overflow (Hogenauer); None = unbounded int64.
    Returns the integer output (gain (R D)^N)."""
    y = np.asarray(x_int, dtype=np.int64)
    if width is not None:
        mod = np.int64(1) << np.int64(width)
        half = mod >> np.int64(1)

        def wrap(v):
            return ((v + half) & (mod - 1)) - half
    else:
        def wrap(v):
            return v
    for _ in range(N):
        y = wrap(np.cumsum(y))
    y = y[R - 1::R]
    for _ in range(N):
        prev = np.r_[np.zeros(D, dtype=np.int64), y[:-D]] if len(y) > D else np.zeros_like(y)
        y = wrap(y - prev)
    return y


def cic_response(f, R, N, D=1):
    """Normalised CIC magnitude |sin(pi f R D) / (R D sin(pi f))|^N at f cycles per *input*
    sample (1 at DC)."""
    f = np.asarray(f, float)
    num = np.sin(np.pi * f * R * D)
    den = R * D * np.sin(np.pi * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(np.abs(den) < 1e-12, 1.0, num / np.where(np.abs(den) < 1e-12, 1, den))
    return np.abs(h) ** N


def cic_comp_taps(R, N, ntaps, fpb, fstop=None, D=1, weight_stop=3.0):
    """Least-squares droop compensator (FIR at the CIC *output* rate): inverse CIC response up
    to fpb, zero from fstop (cycles per output sample). Default fstop = 0.5 - fpb + ... (the
    band that aliases after a further decimation by two starts at 0.25 + (0.25 - fpb))."""
    if fstop is None:
        fstop = min(0.49, max(fpb + 0.04, 0.5 - fpb))
    grid = np.linspace(0, fpb, 40)
    Hg = cic_response(np.maximum(grid, 1e-9) / R, R, N, D)
    bands, des = [], []
    for i in range(len(grid) - 1):
        bands += [grid[i], grid[i + 1]]
        des += [1 / Hg[i], 1 / Hg[i + 1]]
    bands += [fstop, 0.5]
    des += [0, 0]
    wts = [1] * (len(grid) - 1) + [weight_stop]
    ntaps = int(ntaps) | 1
    return _sg.firls(ntaps, bands, des, weight=wts, fs=1)


# ----------------------------------------------------------------------------- quantization
def quantize_coefs(h, bits, full_scale=None):
    """Round coefficients to ``bits``-bit two's-complement words. The word's full scale is the
    smallest power of two >= max|h| (so a Q1.(bits-1) word for |h| < 1, Q2.(bits-2) for |h| < 2...)
    unless ``full_scale`` is given."""
    h = np.asarray(h, float)
    if full_scale is None:
        m = np.max(np.abs(h))
        full_scale = 2.0 ** np.ceil(np.log2(m * (1 + 1e-12))) if m > 0 else 1.0
    q = full_scale / 2 ** (bits - 1)
    return np.clip(np.round(h / q), -2 ** (bits - 1), 2 ** (bits - 1) - 1) * q


def freq_db(b, a=1.0, n=2048, fs=1.0, sos=None):
    """(f, |H| in dB) on [0, fs/2) for a transfer function b/a or second-order sections."""
    if sos is not None:
        w, H = _sg.sosfreqz(sos, worN=n, fs=fs)
    else:
        w, H = _sg.freqz(b, a, worN=n, fs=fs)
    return w, 20 * np.log10(np.abs(H) + 1e-15)


# ----------------------------------------------------------------------------- NCO
def nco(n, f, acc_bits=32, phase_bits=12, amp_bits=14, dither=False, rng=None, start=0):
    """Numerically controlled oscillator: an ``acc_bits`` accumulator incremented by the tuning
    word round(f 2^acc_bits), its top ``phase_bits`` address an ideal sin/cos table quantised to
    ``amp_bits`` (None = exact). With ``dither`` a uniform random value below one phase LSB is
    added before truncation. Returns (complex samples, tuning word, actual frequency)."""
    B = int(acc_bits)
    mod = 1 << B
    ftw = int(round(f * mod)) % mod
    k = np.arange(start, start + n, dtype=np.uint64)
    acc = (k * np.uint64(ftw)) & np.uint64(mod - 1) if B < 64 else k * np.uint64(ftw)
    acc = acc.astype(np.int64) if B < 63 else acc
    sh = B - int(phase_bits)
    if dither and sh > 0:
        rng = np.random.default_rng(rng)
        acc = (acc + rng.integers(0, 1 << sh, n)) % mod
    idx = (acc >> sh) if sh > 0 else acc
    ph = 2 * np.pi * idx / 2.0 ** phase_bits
    z = np.exp(1j * ph)
    if amp_bits is not None:
        s = 2 ** (amp_bits - 1) - 1
        z = (np.round(z.real * s) + 1j * np.round(z.imag * s)) / s
    return z, ftw, ftw / mod


def nco_sfdr(z, guard=12):
    """Spur-free dynamic range (dBc) of a complex NCO output: carrier peak against the
    largest other spectral line (Blackman-Harris window). Returns (sfdr, spur_bin_freq, f, P_dB)."""
    N = len(z)
    w = _sg.windows.blackmanharris(N)
    Z = np.abs(np.fft.fft(z * w)) ** 2
    Z = Z / Z.max()
    k = int(np.argmax(Z))
    mask = np.ones(N, bool)
    idx = (np.arange(-guard, guard + 1) + k) % N
    mask[idx] = False
    j = int(np.argmax(np.where(mask, Z, 0)))
    f = np.fft.fftfreq(N)
    return -10 * np.log10(Z[j] + 1e-30), float(f[j]), f, 10 * np.log10(Z + 1e-30)


# ----------------------------------------------------------------------------- CORDIC
def cordic_gain(n):
    """Total CORDIC stretch 1/K = prod sqrt(1 + 2^-2i), i < n."""
    return float(np.prod(np.sqrt(1 + 2.0 ** (-2 * np.arange(n)))))


def cordic(theta, n, bits=None, prerotate=True):
    """Rotation-mode CORDIC producing (cos theta, sin theta) with n iterations.

    bits=None: floating point. Otherwise a bit-true integer datapath of ``bits`` bits
    (x, y scaled by 2^(bits-2); angles by 2^(bits-1)/pi), arithmetic right shifts.
    ``prerotate`` adds the +-90 degree pre-rotation that extends convergence to the full circle;
    without it CORDIC converges only for |theta| < 99.9 degrees."""
    theta = np.atleast_1d(np.asarray(theta, float))
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    K = 1 / cordic_gain(n)
    sign = np.ones_like(theta)
    if prerotate:
        big = np.abs(theta) > np.pi / 2
        # rotate by pi: cos/sin both change sign
        sign = np.where(big, -1.0, 1.0)
        theta = np.where(big, theta - np.pi * np.sign(theta), theta)
    if bits is None:
        x = np.full_like(theta, K)
        y = np.zeros_like(theta)
        z = theta.copy()
        for i in range(n):
            d = np.where(z >= 0, 1.0, -1.0)
            x, y, z = x - d * y * 2.0 ** -i, y + d * x * 2.0 ** -i, z - d * np.arctan(2.0 ** -i)
        return sign * x, sign * y
    S = 2 ** (bits - 2)
    x = np.full(theta.shape, int(round(K * S)), dtype=np.int64)
    y = np.zeros_like(x)
    ang = np.round(np.arctan(2.0 ** -np.arange(n)) * 2 ** (bits - 1) / np.pi).astype(np.int64)
    z = np.round(theta * 2 ** (bits - 1) / np.pi).astype(np.int64)
    for i in range(n):
        d = np.where(z >= 0, 1, -1)
        x, y, z = x - d * (y >> i), y + d * (x >> i), z - d * ang[i]
    return sign * x / S, sign * y / S


def cordic_trajectory(theta, n, mode="rotate", x0=1.0, y0=0.0):
    """Floating-point CORDIC steps (no gain correction, no pre-rotation): returns arrays
    x[i], y[i], z[i] for i = 0..n. mode 'rotate' drives z to 0 starting from (x0, y0, theta);
    mode 'vector' drives y to 0 starting from (x0, y0, 0)."""
    x, y, z = float(x0), float(y0), float(theta if mode == "rotate" else 0.0)
    X, Y, Z = [x], [y], [z]
    for i in range(n):
        if mode == "rotate":
            d = 1.0 if z >= 0 else -1.0
        else:
            d = -1.0 if y >= 0 else 1.0
        x, y, z = x - d * y * 2.0 ** -i, y + d * x * 2.0 ** -i, z - d * np.arctan(2.0 ** -i)
        X.append(x)
        Y.append(y)
        Z.append(z)
    return np.array(X), np.array(Y), np.array(Z)


# ----------------------------------------------------------------------------- channelizer
def pfb_channelizer(x, h, M):
    """Critically sampled polyphase DFT analysis bank. Returns (n_out, M): column k is channel k
    (centred on k/M cycles/sample) at rate fs/M, equal to
    y_k[m] = sum_n x[n] h[mM - n] exp(-j 2 pi k (mM - n) / M)."""
    x = np.asarray(x, complex)
    L = len(h) // M
    h = np.asarray(h)[:L * M]
    E = h.reshape(L, M).T                                # E[p, l] = h[lM + p]
    nblk = len(x) // M
    X = x[:nblk * M].reshape(nblk, M)
    xp = np.zeros((M, nblk), dtype=complex)
    xp[0] = X[:, 0]
    for p in range(1, M):
        xp[p, 1:] = X[:-1, M - p]
    v = np.array([np.convolve(xp[p], E[p])[:nblk] for p in range(M)])
    return (np.fft.ifft(v, axis=0) * M).T
