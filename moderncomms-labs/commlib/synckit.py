"""synckit: streaming loops, estimators and sequences for Chapter 10 (*Synchronization*).

Companion of commlib.sync (whose whole-array loops drew the chapter's figures) and Lab 04.
Everything here is written to be *resumable*: a loop object keeps its state between calls,
so an interactive lab can feed it a few hundred symbols per animation frame and watch it
lock. Import explicitly:

    from commlib import synckit as sk
    pll = sk.CarrierLoop(bn=0.02, detector="dd", M=4)
    z, e = pll.run(y)                       # resumes where the last call stopped
    sk.est_kay(z), sk.crb_freq(N, esn0_db)

The small estimators (Kay, Fitz, Luise–Reggiannini, periodogram, Oerder–Meyr) and the
timing-detector S-curves are those of book/figscripts/ch10_figs.py.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from .sync import loop_gains

__all__ = ["wrap", "CarrierLoop", "PhaseLoop", "TimingLoop", "pll_step_response",
           "pll_closed_loop", "noise_bandwidth", "est_kay", "est_fitz", "est_lr",
           "est_periodogram", "est_mpower", "crb_freq", "phase_detector", "ted_scurve",
           "interp_taps", "interp_response", "barker13", "mseq", "nr_pss", "lte_pss_zc",
           "ofdm_time", "om_estimate"]


def wrap(x, period=2 * np.pi):
    """Wrap to (-period/2, period/2]."""
    return (np.asarray(x) + period / 2) % period - period / 2


# ============================================================================ carrier loops
class CarrierLoop:
    """Second-order (type-2) carrier loop on symbol-rate samples, resumable.

    detector: "dd"      decision-directed arg(z a*) for M-PSK (M = 2, 4, 8)
              "costas"  hard-limited Costas (QPSK: sgn(I)Q - sgn(Q)I; BPSK: sgn(I)Q)
              "da"      data-aided arg(z a*) with known symbols (pass ``ref``)
              "sin"     sinusoidal (analog-PLL-like) detector sin(phase error) on a tone
    The loop filter is proportional-plus-integral with gains from ``loop_gains(bn, zeta)``.
    ``run`` returns the derotated samples and the NCO phase after each sample."""

    def __init__(self, bn=0.02, zeta=0.7071, detector="dd", M=4, kd=1.0):
        self.kp, self.ki = loop_gains(bn, zeta, kd=kd)
        self.detector, self.M = detector, M
        self.phase = 0.0
        self.integ = 0.0

    def _err(self, z, a=None):
        d = self.detector
        if d == "sin":
            return z.imag / max(abs(z), 1e-12)
        if d == "da":
            return cmath.phase(z * a.conjugate())
        if d == "costas":
            if self.M == 2:
                return (1.0 if z.real >= 0 else -1.0) * z.imag
            return ((1.0 if z.real >= 0 else -1.0) * z.imag
                    - (1.0 if z.imag >= 0 else -1.0) * z.real) / math.sqrt(2)
        # decision-directed M-PSK: nearest point of exp(j(2 pi m / M + offset))
        M = self.M
        off = math.pi / 4 if M == 4 else 0.0
        ang = cmath.phase(z) - off
        step = 2 * math.pi / M
        dec = round(ang / step) * step
        return ang - dec

    def run(self, y, ref=None):
        y = np.asarray(y, complex)
        out = np.empty(len(y), complex)
        ph = np.empty(len(y))
        kp, ki = self.kp, self.ki
        phase, integ = self.phase, self.integ
        for n in range(len(y)):
            z = complex(y[n]) * cmath.exp(-1j * phase)
            out[n] = z
            e = self._err(z, None if ref is None else complex(ref[n]))
            integ += ki * e
            phase += kp * e + integ
            ph[n] = phase
        self.phase, self.integ = phase, integ
        return out, ph

    @property
    def freq(self):
        """Current NCO frequency estimate (cycles per sample): the integrator."""
        return self.integ / (2 * math.pi)


class PhaseLoop(CarrierLoop):
    """A PLL tracking an unmodulated tone (sinusoidal phase detector)."""

    def __init__(self, bn=0.02, zeta=0.7071):
        super().__init__(bn, zeta, detector="sin")


def noise_bandwidth(wn, zeta):
    """One-sided loop noise bandwidth B_L = (wn/2)(zeta + 1/(4 zeta)) (rad/s -> Hz if wn in rad/s)."""
    return wn / 2 * (zeta + 1 / (4 * zeta))


def pll_closed_loop(w, zeta, wn=1.0):
    """Closed-loop H(jw) and error transfer 1 - H(jw) of the ideal type-2 loop."""
    s = 1j * np.asarray(w, float)
    H = (2 * zeta * wn * s + wn ** 2) / (s ** 2 + 2 * zeta * wn * s + wn ** 2)
    return H, 1 - H


def pll_step_response(t, zeta, kind="phase", wn=1.0):
    """Phase error of the linear type-2 loop (closed form). t in seconds (or in 1/wn).

    kind="phase": unit phase step, E(s) = s/(s^2 + 2 zeta wn s + wn^2).
    kind="freq":  unit frequency step (1 rad/s), E(s) = 1/(s^2 + 2 zeta wn s + wn^2).
    kind="type1": type-1 loop (gain wn) after a unit frequency step: (1 - e^{-wn t})/wn."""
    t = np.asarray(t, float)
    if kind == "type1":
        return (1 - np.exp(-wn * t)) / wn
    a = zeta * wn
    if abs(zeta - 1) < 1e-6:
        h = t * np.exp(-a * t)                         # impulse response of 1/(s+wn)^2
        dh = np.exp(-a * t) * (1 - a * t)
    elif zeta < 1:
        wd = wn * math.sqrt(1 - zeta ** 2)
        h = np.exp(-a * t) * np.sin(wd * t) / wd
        dh = np.exp(-a * t) * (np.cos(wd * t) - a / wd * np.sin(wd * t))
    else:
        r = wn * math.sqrt(zeta ** 2 - 1)
        p1, p2 = -a + r, -a - r
        h = (np.exp(p1 * t) - np.exp(p2 * t)) / (p1 - p2)
        dh = (p1 * np.exp(p1 * t) - p2 * np.exp(p2 * t)) / (p1 - p2)
    return dh if kind == "phase" else h


# ============================================================================ timing loop
def _cubic(x, t):
    """Scalar cubic Lagrange interpolation (points -1, 0, 1, 2); x is a Python list/array."""
    n = int(math.floor(t))
    mu = t - n
    xm1, x0, x1, x2 = x[n - 1], x[n], x[n + 1], x[n + 2]
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + x0


class TimingLoop:
    """Asynchronous (Gardner-architecture) symbol timing loop, resumable.

    ted: "gardner" (2 samples/symbol, carrier-independent), "mm" (Mueller–Müller, 1 sample
    per symbol, decision-directed, QPSK decisions) or "el" (early–late, ±T/4 power detector).
    Feed matched-filter output in chunks with ``push``; it returns the strobes (symbols),
    the detector outputs and the fractional strobe positions (in samples, mod sps); the
    absolute strobe positions of the last call are in ``last_times``."""

    def __init__(self, sps, bn=0.01, zeta=0.7071, ted="gardner", kd=None):
        if kd is None:
            kd = {"gardner": 2.0, "mm": 1.0, "el": 2.0}[ted]
        self.kp, self.ki = loop_gains(bn, zeta, kd=kd)
        self.sps, self.ted = sps, ted
        self.buf = np.zeros(0, complex)
        self.base = 0                 # absolute index of buf[0]
        self.t = float(sps) + 2       # absolute strobe position
        self.integ = 0.0
        self.prev = 0j
        self.prev_dec = 0j

    def push(self, x):
        self.buf = np.concatenate([self.buf, np.asarray(x, complex)])
        sps, ted = self.sps, self.ted
        b = self.buf.tolist()
        syms, errs, taus, times = [], [], [], []
        kp, ki = self.kp, self.ki
        t, integ, prev, prev_dec = self.t, self.integ, self.prev, self.prev_dec
        base = self.base
        end = base + len(b) - 3
        r2 = 1 / math.sqrt(2)
        while t + sps / 2 + 1 < end:
            tl = t - base
            cur = _cubic(b, tl)
            if ted == "gardner":
                mid = _cubic(b, tl - sps / 2)
                e = (mid.conjugate() * (cur - prev)).real
            elif ted == "mm":
                dec = complex(r2 if cur.real >= 0 else -r2, r2 if cur.imag >= 0 else -r2)
                e = (dec.conjugate() * prev - prev_dec.conjugate() * cur).real
                prev_dec = dec
            else:
                ye = _cubic(b, tl + sps / 4)
                yl = _cubic(b, tl - sps / 4)
                e = abs(yl) ** 2 - abs(ye) ** 2
            integ += ki * e
            v = kp * e + integ
            syms.append(cur)
            errs.append(e)
            taus.append(t % sps)
            times.append(t)
            prev = cur
            t += sps - v * sps
        # keep a short tail of the buffer
        keep_from = max(0, int(t - base) - 2 * sps - 4)
        self.buf = self.buf[keep_from:]
        self.base = base + keep_from
        self.t, self.integ, self.prev, self.prev_dec = t, integ, prev, prev_dec
        self.last_times = np.array(times)          # absolute strobe positions (samples)
        return np.array(syms, complex), np.array(errs), np.array(taus)


def ted_scurve(x, d0, sps, taus, ted="gardner", nsym=None):
    """Mean TED output against static timing offset tau (in T) for matched-filter output x
    whose symbol 0 peaks at index d0. Vectorised (uses commlib.interp_cubic)."""
    from .filters import interp_cubic
    from .modulation import get_constellation
    q = get_constellation("qpsk")
    n = nsym or (len(x) - d0) // sps - 6
    k = np.arange(4, n)
    out = []
    for tau in taus:
        t = d0 + (k + tau) * sps
        y = interp_cubic(x, t)
        if ted == "gardner":
            ym = interp_cubic(x, t - sps / 2)
            yp = interp_cubic(x, t - sps)
            out.append(np.mean(np.real(np.conj(ym) * (y - yp))))
        elif ted == "mm":
            d = q.decide(y)
            out.append(np.mean(np.real(np.conj(d[1:]) * y[:-1] - np.conj(d[:-1]) * y[1:])))
        else:
            ye = interp_cubic(x, t + sps / 4)
            yl = interp_cubic(x, t - sps / 4)
            out.append(np.mean(np.abs(yl) ** 2 - np.abs(ye) ** 2))
    return np.array(out)


def om_estimate(x, sps):
    """Oerder–Meyr feedforward timing estimate (fraction of T) from |x|^2's symbol-rate line."""
    n = np.arange(len(x))
    X = np.sum(np.abs(x) ** 2 * np.exp(-2j * np.pi * n / sps))
    return -np.angle(X) / (2 * np.pi)


# ============================================================================ interpolators
def interp_taps(mu, kind="cubic"):
    """Taps (applied to samples at offsets -1, 0, 1, 2) that interpolate at 0 + mu.
    kind: "linear", "parabolic" (Erup–Gardner–Harris, alpha = 0.5), "cubic" (Lagrange)."""
    if kind == "linear":
        return np.array([0.0, 1 - mu, mu, 0.0])
    if kind == "parabolic":
        return _parabolic(mu)
    return np.array([-mu * (mu - 1) * (mu - 2) / 6, (mu + 1) * (mu - 1) * (mu - 2) / 2,
                     -(mu + 1) * mu * (mu - 2) / 2, (mu + 1) * mu * (mu - 1) / 6])


def _parabolic(mu, a=0.5):
    """Piecewise-parabolic Farrow interpolator of Erup, Gardner and Harris (1993)."""
    return np.array([a * mu ** 2 - a * mu,
                     -a * mu ** 2 + (a - 1) * mu + 1,
                     -a * mu ** 2 + (a + 1) * mu,
                     a * mu ** 2 - a * mu])


def interp_response(mu, f, kind="cubic"):
    """Frequency response of the interpolator at normalised frequencies f (cycles/sample):
    returns (|H| in dB, delay error in samples relative to the ideal mu)."""
    h = interp_taps(mu, kind)
    w = 2 * np.pi * np.asarray(f, float)
    n = np.arange(-1, 3)
    H = np.sum(h[None, :] * np.exp(-1j * w[:, None] * n[None, :]), axis=1)
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    ph = np.unwrap(np.angle(H))
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(w > 1e-9, -ph / np.where(w > 1e-9, w, 1), np.nan)
    if np.isnan(delay[0]) and len(delay) > 1:
        delay[0] = delay[1]
    # interpolating at +mu means a time ADVANCE of mu: ideal phase = +w mu, i.e. delay = -mu
    return mag, delay + mu


# ============================================================================ frequency estimators
def _R(z, m):
    return np.mean(z[m:] * np.conj(z[:-m]))


def est_kay(z):
    """Kay (1989): parabolically weighted average of adjacent phase differences."""
    N = len(z)
    k = np.arange(1, N)
    w = 1.5 * N / (N ** 2 - 1) * (1 - ((2 * k - N) / N) ** 2)
    return float(np.sum(w * np.angle(z[1:] * np.conj(z[:-1]))) / (2 * np.pi))


def est_fitz(z, L=None):
    """Fitz (1994): phases of the autocorrelation at lags 1..L. Range ±1/(2L)."""
    L = L or len(z) // 2
    return float(sum(np.angle(_R(z, m)) for m in range(1, L + 1)) / (np.pi * L * (L + 1)))


def est_lr(z, L=None):
    """Luise & Reggiannini (1995): phase of the summed autocorrelation. Range ±1/(L+1)."""
    L = L or len(z) // 2
    return float(np.angle(sum(_R(z, m) for m in range(1, L + 1))) / (np.pi * (L + 1)))


def est_periodogram(z, nfft=1 << 14):
    """Periodogram peak with parabolic refinement (the ML estimator, with a threshold)."""
    Z = np.abs(np.fft.fft(z, nfft))
    k = int(np.argmax(Z))
    a, b, c = Z[k - 1], Z[k], Z[(k + 1) % nfft]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return float(wrap((k + d) / nfft, 1.0))


def est_mpower(y, M=4, nfft=1 << 14):
    """Blind M-th power estimator (non-data-aided) for M-PSK; range ±1/(2M)."""
    return est_periodogram(np.asarray(y) ** M, nfft) / M


def crb_freq(N, esn0_db):
    """Cramér–Rao bound on the variance of a frequency estimate (cycles/symbol)^2 from N
    known symbols: 3 / (2 pi^2 N (N^2 - 1) Es/N0)."""
    return 3 / (2 * np.pi ** 2 * N * (N ** 2 - 1) * 10 ** (np.asarray(esn0_db, float) / 10))


# ============================================================================ phase detectors
def phase_detector(z, kind, a=None, M=4):
    """Vectorised phase-detector outputs for the S-curve experiments.
    kind: "da" Im{z a*}, "dd" arg(z â*), "costas2" 2·I·Q (BPSK Costas),
          "costas4" hard-limited QPSK Costas, "mpower" Im{z^M}/M (M-th power)."""
    z = np.asarray(z)
    if kind == "da":
        return np.imag(z * np.conj(a))
    if kind == "dd":
        off = np.pi / 4 if M == 4 else 0.0
        ang = np.angle(z) - off
        st = 2 * np.pi / M
        return ang - np.round(ang / st) * st
    if kind == "costas2":
        return 2 * z.real * z.imag
    if kind == "costas4":
        return (np.sign(z.real) * z.imag - np.sign(z.imag) * z.real) / np.sqrt(2)
    if kind == "mpower":
        return np.imag(-(z ** M) if M == 4 else z ** M) / M
    raise ValueError(kind)


# ============================================================================ sequences
def barker13():
    return np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], float)


def mseq(n=6):
    """±1 maximal-length sequence of length 2^n - 1."""
    from scipy.signal import max_len_seq
    return 1 - 2.0 * max_len_seq(n)[0]


def nr_pss(nid2=0):
    """5G NR PSS (3GPP TS 38.211 §7.4.2.2): length-127 BPSK m-sequence, cyclic shift 43·N_ID2."""
    x = np.zeros(127 + 7, int)
    x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(127):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    n = np.arange(127)
    return 1 - 2.0 * x[(n + 43 * nid2) % 127]


def lte_pss_zc(nid2=0):
    """LTE PSS (3GPP TS 36.211 §6.11.1): length-63 Zadoff–Chu, roots 25, 29, 34, DC punctured
    (62 elements)."""
    u = (25, 29, 34)[nid2]
    n = np.arange(63)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / 63)
    return np.delete(zc, 31)


def ofdm_time(seq, nfft=256, kind="nr"):
    """Map a PSS onto subcarriers around DC and return one unit-energy-per-sample OFDM symbol
    (no CP). kind "nr": 127 subcarriers -63..63; "lte": 62 subcarriers -31..31 without DC."""
    g = np.zeros(nfft, complex)
    if kind == "nr":
        k = np.arange(-63, 64)
    else:
        k = np.r_[np.arange(-31, 0), np.arange(1, 32)]
    g[k % nfft] = seq
    return np.fft.ifft(g) * np.sqrt(nfft)
