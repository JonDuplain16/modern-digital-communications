"""Line codes, block codes for DC balance, scramblers and serial-link channel models.

Used by Lab 13 (Chapter 8). Import explicitly:  ``from commlib import linecodes as lc``.

* ``encode_line``      NRZ-L, NRZI, unipolar NRZ/RZ, Manchester, AMI, MLT-3, 2B1Q, PAM-4 waveforms
* ``enc8b10b``         data-character 8b/10b encoder (Widmer–Franaszek) with running disparity
* ``prbs``             PRBS-7/9/15/23/31 from a Fibonacci LFSR (ITU-T O.150 polynomials)
* ``scramble_add`` / ``scramble_ss`` / ``descramble_ss``  additive and self-synchronous scramblers
* ``run_lengths``, ``running_digital_sum``
* ``backplane``        minimum-phase lossy-line impulse response (skin effect + dielectric loss)
* ``ffe_zf``           least-squares zero-forcing FFE taps from a sampled pulse response
"""
from __future__ import annotations

import numpy as np

__all__ = ["encode_line", "enc8b10b", "prbs", "PRBS_TAPS", "scramble_add", "scramble_ss",
           "descramble_ss", "run_lengths", "running_digital_sum", "minphase_from_mag", "backplane",
           "pulse_response", "ffe_zf"]


# ----------------------------------------------------------------------------- line codes
def encode_line(bits, code, sps=16):
    """Waveform with ``sps`` samples per bit for the named line code.

    codes: 'nrz' (polar), 'unrz' (unipolar NRZ), 'urz' (unipolar RZ, 50%), 'prz' (polar RZ),
    'nrzi', 'manchester' (IEEE 802.3: 1 = low->high), 'ami', 'mlt3', '2b1q' (sps per *bit*),
    'pam4' (Gray, levels +-1/3, +-1; sps per *bit*, i.e. 2*sps per symbol).
    """
    b = np.asarray(bits, dtype=int)
    h = sps // 2
    if code == "nrz":
        return np.repeat(2.0 * b - 1, sps)
    if code == "unrz":
        return np.repeat(b.astype(float), sps)
    if code == "urz":
        cell = np.r_[np.ones(h), np.zeros(sps - h)]
        return (b[:, None] * cell[None, :]).ravel().astype(float)
    if code == "prz":
        cell = np.r_[np.ones(h), np.zeros(sps - h)]
        return ((2.0 * b - 1)[:, None] * cell[None, :]).ravel()
    if code == "nrzi":
        lvl = np.cumsum(b) % 2
        return np.repeat(2.0 * lvl - 1, sps)
    if code == "manchester":
        cell = np.r_[-np.ones(h), np.ones(sps - h)]
        return ((2.0 * b - 1)[:, None] * cell[None, :]).ravel()
    if code == "ami":
        ones = np.cumsum(b)
        v = np.where(b == 1, np.where(ones % 2 == 1, 1.0, -1.0), 0.0)
        return np.repeat(v, sps)
    if code == "mlt3":
        cyc = np.array([0.0, 1.0, 0.0, -1.0])
        return np.repeat(cyc[np.cumsum(b) % 4], sps)
    if code in ("2b1q", "pam4"):
        pairs = b[:len(b) // 2 * 2].reshape(-1, 2)
        # Gray map (first bit = sign): 10 -> +3, 11 -> +1, 01 -> -1, 00 -> -3
        lut = {(1, 0): 3.0, (1, 1): 1.0, (0, 1): -1.0, (0, 0): -3.0}
        v = np.array([lut[tuple(p)] for p in pairs]) / 3.0
        return np.repeat(v, 2 * sps)
    raise ValueError(code)


# ----------------------------------------------------------------------------- 8b/10b
# 5b/6b and 3b/4b tables for RD- (IBM, Widmer & Franaszek 1983; IEEE 802.3 Clause 36).
_D5 = ["100111", "011101", "101101", "110001", "110101", "101001", "011001", "111000",
       "111001", "100101", "010101", "110100", "001101", "101100", "011100", "010111",
       "011011", "100011", "010011", "110010", "001011", "101010", "011010", "111010",
       "110011", "100110", "010110", "110110", "001110", "101110", "011110", "101011"]
_D3 = ["1011", "1001", "0101", "1100", "1101", "1010", "0110", "1110"]


def _disp(s):
    return s.count("1") * 2 - len(s)


def _comp(s):
    return "".join("1" if c == "0" else "0" for c in s)


def enc8b10b(data, rd=-1, return_rd=False):
    """Encode data bytes (0..255) as 8b/10b data characters D.x.y.

    Bits are output in transmission order (a b c d e i f g h j). Returns a 0/1 array and,
    optionally, the running disparity after each character (+1 / -1).
    """
    out, rds = [], []
    for B in np.asarray(data, dtype=int):
        x, y = B & 31, B >> 5
        a = _D5[x]
        if x == 7:
            a = "111000" if rd < 0 else "000111"
        elif _disp(a) != 0 and rd > 0:
            a = _comp(a)
        if _disp(a) != 0:
            rd = -rd
        if y == 7:
            alt = (rd < 0 and x in (17, 18, 20)) or (rd > 0 and x in (11, 13, 14))
            b = "0111" if alt else "1110"
            if rd > 0:
                b = _comp(b)
        else:
            b = _D3[y]
            if y == 3:
                b = "1100" if rd < 0 else "0011"
            elif _disp(b) != 0 and rd > 0:
                b = _comp(b)
        if _disp(b) != 0:
            rd = -rd
        out.append(a + b)
        rds.append(rd)
    bits = np.array([int(c) for c in "".join(out)], dtype=np.int8)
    return (bits, np.array(rds)) if return_rd else bits


# ----------------------------------------------------------------------------- PRBS / scramblers
PRBS_TAPS = {7: (7, 6), 9: (9, 5), 15: (15, 14), 23: (23, 18), 31: (31, 28)}   # x^n + x^m + 1


def prbs(order, n, seed=None):
    """n bits of PRBS-`order` from a Fibonacci LFSR with polynomial x^a + x^b + 1."""
    a, b_ = PRBS_TAPS[order]
    reg = np.ones(order, dtype=np.int8) if seed is None else np.asarray(seed, dtype=np.int8).copy()
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        bit = reg[a - 1] ^ reg[b_ - 1]
        out[i] = bit
        reg[1:] = reg[:-1]
        reg[0] = bit
    return out


def scramble_add(bits, order=15, seed=None):
    """Additive (frame-synchronous) scrambler: XOR with a PRBS. Its own inverse."""
    return (np.asarray(bits, dtype=np.int8) ^ prbs(order, len(bits), seed)).astype(np.int8)


def scramble_ss(bits, taps=(39, 58), state=None):
    """Self-synchronous scrambler s[n] = d[n] ^ s[n-t1] ^ s[n-t2] (default: IEEE 802.3 64b/66b)."""
    L = max(taps)
    reg = np.ones(L, dtype=np.int8) if state is None else np.asarray(state, dtype=np.int8).copy()
    buf = np.concatenate([reg, np.zeros(len(bits), dtype=np.int8)])
    d = np.asarray(bits, dtype=np.int8)
    for i in range(len(d)):
        buf[L + i] = d[i] ^ buf[L + i - taps[0]] ^ buf[L + i - taps[1]]
    return buf[L:].copy()


def descramble_ss(bits, taps=(39, 58), state=None):
    """Inverse of ``scramble_ss``: d[n] = s[n] ^ s[n-t1] ^ s[n-t2]. Synchronises by itself
    after max(taps) bits; one channel error causes 3 output errors (error multiplication)."""
    L = max(taps)
    reg = np.ones(L, dtype=np.int8) if state is None else np.asarray(state, dtype=np.int8)
    s = np.concatenate([reg, np.asarray(bits, dtype=np.int8)])
    return (s[L:] ^ s[L - taps[0]:len(s) - taps[0]] ^ s[L - taps[1]:len(s) - taps[1]]).astype(np.int8)


def run_lengths(bits):
    """Lengths of runs of identical consecutive symbols."""
    b = np.asarray(bits)
    idx = np.flatnonzero(np.diff(b)) + 1
    return np.diff(np.r_[0, idx, len(b)])


def running_digital_sum(bits):
    """Cumulative sum of +1 (for 1) / -1 (for 0): the DC wander a line code must bound."""
    return np.cumsum(2 * np.asarray(bits, dtype=int) - 1)


# ----------------------------------------------------------------------------- channels
def minphase_from_mag(mag, nfft):
    """Minimum-phase impulse response from |H| on an nfft-point full FFT grid (real cepstrum)."""
    c = np.fft.ifft(np.log(np.maximum(mag, 1e-12))).real
    w = np.zeros(nfft)
    w[0] = 1
    w[1:nfft // 2] = 2
    w[nfft // 2] = 1
    return np.fft.ifft(np.exp(np.fft.fft(c * w))).real


def backplane(fs, loss_db, f_ref, nfft=1 << 14, length=None, skin_frac=0.4):
    """Lossy transmission line: IL(f) = k (skin_frac sqrt(f/f_ref) + (1-skin_frac) f/f_ref) dB,
    scaled so that IL(f_ref) = loss_db. Returns a minimum-phase impulse response sampled at fs
    (unit DC gain), truncated to ``length`` samples (default nfft // 8)."""
    f = np.abs(np.fft.fftfreq(nfft, 1 / fs)) / f_ref
    il = loss_db * (skin_frac * np.sqrt(f) + (1 - skin_frac) * f)
    h = minphase_from_mag(10 ** (-il / 20), nfft)
    return h[:length or nfft // 8]


def pulse_response(h, sps):
    """Response of channel h (sampled at sps per UI) to a one-UI rectangular pulse."""
    return np.convolve(h, np.ones(sps))


def ffe_zf(pulse, sps, ntaps=5, pre=1, cursor=None):
    """Symbol-spaced FFE taps (``pre`` pre-cursor taps) that force the UI-sampled pulse
    response toward a unit main cursor with no ISI (least squares).

    Returns (taps, cursor_index, sampling_phase)."""
    cursor = int(np.argmax(pulse)) if cursor is None else cursor
    p = pulse[cursor % sps::sps]
    c = cursor // sps
    L = len(p) + ntaps - 1
    A = np.zeros((L, ntaps))
    for k in range(ntaps):
        A[k:k + len(p), k] = p
    target = np.zeros(L)
    target[c + pre] = 1.0
    w, *_ = np.linalg.lstsq(A, target, rcond=None)
    return w, cursor, cursor % sps
