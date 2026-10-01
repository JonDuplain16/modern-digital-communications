"""sourcecoding: lossless and lossy source coding for text, speech, images and audio.

Companion of Chapter 16 (*Source Coding: voice, audio, image, video*). Clean copies of the
helpers in book/figscripts/ch16_figs.py (Huffman lengths, adaptive context models, LPC with
Levinson-Durbin, the synthetic vowel, the toy baseline-JPEG coder, the psychoacoustic model),
plus a working integer arithmetic coder and canonical Huffman codes. Import explicitly:

    from commlib import sourcecoding as sc
    text = sc.load_text()                                   # 27-symbol English from Chapter 1
    bits = sc.ArithmeticCoder(order=2).encode(text)          # real, decodable bit stream
    rec, nbits = sc.toy_jpeg(img, quality=50)                # baseline-JPEG-like luma coder
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import signal as _sig
from scipy.fft import dctn, idctn

from .infotheory import huffman_lengths, huffman_code

__all__ = ["load_text", "load_image", "huffman_lengths", "huffman_code", "canonical_huffman",
           "huffman_encode", "huffman_decode", "adaptive_ctx_bits", "ArithmeticCoder", "arith_interval",
           "levinson", "lpc_frame", "lpc_to_lsf", "synth_vowel", "FORMANTS", "lpc_analysis_synthesis",
           "QLUM", "qtable", "ZIGZAG", "jpeg_category", "toy_jpeg", "psnr", "bark", "ath", "spreading",
           "masking_threshold", "shaped_noise"]

_HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================ data
def load_text(path=None, alphabet27=True):
    """English prose: Chapter 1 of the book (moderncomms-labs/data/english_text.txt). With
    alphabet27 the text is lower-cased and reduced to a-z and space, as in Chapters 13 and 16."""
    path = path or os.path.join(_HERE, "..", "data", "english_text.txt")
    raw = open(path, encoding="utf8").read().split("\n", 1)[1]
    if not alphabet27:
        return raw
    return re.sub(r" +", " ", re.sub(r"[^a-z ]", " ", raw.lower())).strip()


def load_image(size=256, color=False):
    """The public-domain portrait of Grace Hopper shipped with matplotlib, resized to 256 x 256
    exactly as in the book's figures (gray float array 0..255, or RGB if color)."""
    import matplotlib.cbook as cbook
    from PIL import Image
    im = Image.open(cbook.get_sample_data("grace_hopper.jpg"))
    im = im.resize((256, 300), Image.LANCZOS)        # halve: breaks the original 8x8 JPEG grid
    im = im.crop((0, 10, 256, 266))
    if color:
        return np.asarray(im).astype(float)
    return np.asarray(im.convert("L")).astype(float)


# ============================================================================ Huffman
def canonical_huffman(lengths):
    """Canonical Huffman code from a dict symbol -> code length (as DEFLATE and JPEG transmit it:
    only the lengths are sent). Symbols are ordered by (length, symbol)."""
    items = sorted(lengths.items(), key=lambda t: (t[1], t[0]))
    code, prev_len, out = 0, items[0][1], {}
    for i, (s, L) in enumerate(items):
        if i:
            code = (code + 1) << (L - prev_len)
        out[s] = format(code, f"0{L}b")
        prev_len = L
    return out


def huffman_encode(seq, code):
    return "".join(code[s] for s in seq)


def huffman_decode(bits, code):
    inv = {v: k for k, v in code.items()}
    out, cur = [], ""
    for b in bits:
        cur += b
        if cur in inv:
            out.append(inv[cur]); cur = ""
    return out


# ============================================================================ context models
def adaptive_ctx_bits(text, order, alpha=0.5, alphabet=None):
    """Ideal code length (bits) of an adaptive order-k context model with an add-alpha
    (Krichevsky-Trofimov for alpha = 1/2) estimator: what an arithmetic coder achieves."""
    A = sorted(set(text)) if alphabet is None else alphabet
    K = len(A)
    counts = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    bits = 0.0
    for i, c in enumerate(text):
        ctx = text[max(0, i - order):i]
        n = counts[ctx]
        bits -= np.log2((n[c] + alpha) / (tot[ctx] + alpha * K))
        n[c] += 1
        tot[ctx] += 1
    return bits


class ArithmeticCoder:
    """Integer (32-bit) arithmetic coder with an adaptive order-k context model (frequency
    counts start at 1 for every symbol, increment by `inc`). encode() returns a '0'/'1' string;
    decode() inverts it. The model and coder follow Witten, Neal and Cleary (1987)."""

    PREC = 32
    FULL = (1 << 32) - 1
    HALF = 1 << 31
    QTR = 1 << 30

    def __init__(self, order=0, alphabet=None, inc=16):
        self.order, self.alphabet, self.inc = order, alphabet, inc

    def _model(self):
        K = len(self.alphabet)
        return defaultdict(lambda: np.ones(K, dtype=np.int64))

    def encode(self, text):
        if self.alphabet is None:
            self.alphabet = sorted(set(text))
        idx = {s: i for i, s in enumerate(self.alphabet)}
        tables = self._model()
        lo, hi, pend, out = 0, self.FULL, 0, []
        for i, c in enumerate(text):
            f = tables[text[max(0, i - self.order):i]]
            k = idx[c]
            cum_lo, cum_hi, tot = int(f[:k].sum()), int(f[:k + 1].sum()), int(f.sum())
            rng_ = hi - lo + 1
            hi = lo + rng_ * cum_hi // tot - 1
            lo = lo + rng_ * cum_lo // tot
            while True:
                if hi < self.HALF:
                    out.append("0" + "1" * pend); pend = 0
                elif lo >= self.HALF:
                    out.append("1" + "0" * pend); pend = 0
                    lo -= self.HALF; hi -= self.HALF
                elif lo >= self.QTR and hi < 3 * self.QTR:
                    pend += 1; lo -= self.QTR; hi -= self.QTR
                else:
                    break
                lo, hi = 2 * lo, 2 * hi + 1
            f[k] += self.inc
            if f.sum() > (1 << 24):
                f[:] = (f + 1) // 2
        pend += 1
        out.append(("0" + "1" * pend) if lo < self.QTR else ("1" + "0" * pend))
        return "".join(out)

    def decode(self, bits, n):
        tables = self._model()
        bits = bits + "0" * self.PREC
        lo, hi = 0, self.FULL
        val, pos = int(bits[:self.PREC], 2), self.PREC
        out = []
        for i in range(n):
            ctx = "".join(out[max(0, i - self.order):i])
            f = tables[ctx]
            tot = int(f.sum())
            rng_ = hi - lo + 1
            target = ((val - lo + 1) * tot - 1) // rng_
            cum = np.cumsum(f)
            k = int(np.searchsorted(cum, target, side="right"))
            cum_lo, cum_hi = (int(cum[k - 1]) if k else 0), int(cum[k])
            hi = lo + rng_ * cum_hi // tot - 1
            lo = lo + rng_ * cum_lo // tot
            while True:
                if hi < self.HALF:
                    pass
                elif lo >= self.HALF:
                    lo -= self.HALF; hi -= self.HALF; val -= self.HALF
                elif lo >= self.QTR and hi < 3 * self.QTR:
                    lo -= self.QTR; hi -= self.QTR; val -= self.QTR
                else:
                    break
                lo, hi = 2 * lo, 2 * hi + 1
                val = 2 * val + int(bits[pos]); pos += 1
            out.append(self.alphabet[k])
            f[k] += self.inc
            if f.sum() > (1 << 24):
                f[:] = (f + 1) // 2
        return "".join(out)


def arith_interval(msg, probs):
    """Final [low, high) interval of ideal arithmetic coding of msg with fixed probabilities
    (dict, order of keys = order on the unit interval), and the shortest binary fraction inside it.
    Returns (low, high, codeword_string)."""
    order = list(probs)
    lo, hi = 0.0, 1.0
    for s in msg:
        w = hi - lo
        c = sum(probs[t] for t in order[:order.index(s)])
        lo, hi = lo + c * w, lo + (c + probs[s]) * w
    for nb in range(1, 64):
        m = int(np.ceil(lo * 2 ** nb))
        if m / 2 ** nb < hi:
            return lo, hi, format(m, f"0{nb}b")
    return lo, hi, None


# ============================================================================ speech: LPC
def levinson(r, p):
    """Levinson-Durbin recursion on autocorrelation r[0..p]. Returns (a, k, E): A(z) coefficients
    (a[0] = 1), reflection coefficients k_1..k_p and prediction-error energies E_0..E_p."""
    a = np.zeros(p + 1); a[0] = 1.0
    E = r[0]; ks, Es = [], [E]
    for i in range(1, p + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / E
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a_new[i] = k
        a = a_new
        E *= (1 - k * k)
        ks.append(k); Es.append(E)
    return a, np.array(ks), np.array(Es)


def lpc_frame(x, p=10):
    """Autocorrelation LPC of one frame (Hamming window, tiny white-noise correction).
    Returns ((a, k, E), windowed frame)."""
    w = np.hamming(len(x))
    xw = x * w
    r = np.correlate(xw, xw, "full")[len(xw) - 1:len(xw) + p]
    r = r.copy(); r[0] *= 1.0001
    return levinson(r, p), xw


def lpc_to_lsf(a):
    """Line spectral frequencies (radians, ascending in (0, pi)) of A(z)."""
    P = np.concatenate([a, [0]]) + np.concatenate([[0], a[::-1]])
    Q = np.concatenate([a, [0]]) - np.concatenate([[0], a[::-1]])
    ang = np.angle(np.concatenate([np.roots(P), np.roots(Q)]))
    return np.sort(ang[(ang > 1e-3) & (ang < np.pi - 1e-3)])


FORMANTS = [(730, 90), (1090, 110), (2440, 160), (3400, 250)]     # /a/: (frequency, bandwidth) Hz


def synth_vowel(dur=0.5, f0=120.0, fs=8000, seed=3, formants=FORMANTS):
    """Synthetic /a/: Rosenberg glottal pulses -> formant resonators -> lip radiation."""
    n = int(dur * fs)
    g = np.zeros(n)
    T0 = fs / f0
    t = 0.0
    r = np.random.default_rng(seed)
    while t < n:
        T = T0 * (1 + 0.01 * r.standard_normal())
        Tp, Tn = 0.40 * T, 0.16 * T
        i0 = int(t)
        for m in range(int(Tp + Tn)):
            if i0 + m >= n:
                break
            g[i0 + m] = 0.5 * (1 - np.cos(np.pi * m / Tp)) if m < Tp else np.cos(np.pi * (m - Tp) / (2 * Tn))
        t += T
    x = g - g.mean()
    for F, B in formants:
        rr = np.exp(-np.pi * B / fs); th = 2 * np.pi * F / fs
        x = _sig.lfilter([1 - rr], [1, -2 * rr * np.cos(th), rr * rr], x)
    x = _sig.lfilter([1, -0.98], [1], x)
    x = x / np.max(np.abs(x)) * 0.8
    return x + 1e-3 * r.standard_normal(n)


def lpc_analysis_synthesis(x, fs=8000, p=10, frame=0.02, f0_new=None, seed=0):
    """Frame-by-frame LPC vocoder. Analyse x in Hann-windowed frames (default 20 ms, 50% overlap),
    then resynthesise by filtering an excitation through each frame's all-pole filter 1/A(z)
    and overlap-adding. The excitation is a pulse train at f0_new Hz (voiced), or white noise
    if f0_new is None (whispered). Each frame's excitation power is matched to the frame's
    prediction-error power, so the loudness contour follows the original."""
    N = int(frame * fs); hop = N // 2
    r = np.random.default_rng(seed)
    if f0_new is None:
        exc = r.standard_normal(len(x) + N)
    else:
        exc = np.zeros(len(x) + N)
        exc[np.round(np.arange(0, len(exc), fs / f0_new)).astype(int)[:-1]] = 1.0
    out = np.zeros(len(x) + N)
    win = np.hanning(N)
    ham2 = np.sum(np.hamming(N) ** 2)
    for s in range(0, len(x) - N + 1, hop):
        (a, _, E), _ = lpc_frame(x[s:s + N], p)
        e = exc[max(0, s - N):s + N].copy()              # previous frame = run-in to settle the filter
        e *= np.sqrt(E[-1] / ham2 / max(np.mean(e ** 2), 1e-12))
        y = _sig.lfilter([1.0], a, e)[-N:]
        out[s:s + N] += win * y
    return out[:len(x)]


# ============================================================================ images: toy JPEG
QLUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])


def qtable(quality):
    """IJG quality scaling of the Annex K luminance table (quality 1..100)."""
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((QLUM * s + 50) / 100), 1, 255)


ZIGZAG = sorted(((i, j) for i in range(8) for j in range(8)),
                key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else t[1]))


def jpeg_category(v):
    """JPEG size category: number of bits of |v| (0 for v = 0)."""
    v = abs(int(v))
    return 0 if v == 0 else int(np.floor(np.log2(v))) + 1


def toy_jpeg(img, quality, return_coeffs=False):
    """Baseline-JPEG-like luma coder: 8x8 DCT, IJG-scaled quantisation, DPCM of DC, zigzag
    run-length (run, size) symbols with ZRL and EOB, and *optimised* Huffman tables (code
    lengths from the image's own statistics; headers and tables not counted).
    Returns (reconstruction, bits) (and the quantised blocks if return_coeffs)."""
    Q = qtable(quality)
    H, W = img.shape
    rec = np.zeros_like(img, dtype=float)
    dc_syms, ac_syms, extra = Counter(), Counter(), 0
    prev_dc = 0
    blocks = {}
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            F = dctn(img[r:r + 8, c:c + 8] - 128, norm="ortho")
            q = np.round(F / Q)
            blocks[(r, c)] = q
            rec[r:r + 8, c:c + 8] = idctn(q * Q, norm="ortho") + 128
            d = int(q[0, 0]) - prev_dc; prev_dc = int(q[0, 0])
            dc_syms[jpeg_category(d)] += 1; extra += jpeg_category(d)
            zz = [q[i, j] for (i, j) in ZIGZAG[1:]]
            last = max([k for k, v in enumerate(zz) if v != 0], default=-1)
            run = 0
            for k in range(last + 1):
                v = zz[k]
                if v == 0:
                    run += 1
                    if run == 16:
                        ac_syms[(15, 0)] += 1; run = 0
                    continue
                s = jpeg_category(v)
                ac_syms[(run, s)] += 1; extra += s; run = 0
            if last < 62:
                ac_syms[(0, 0)] += 1
    bits = extra
    for syms in (dc_syms, ac_syms):
        L = huffman_lengths(syms)
        bits += sum(syms[s] * L[s] for s in syms)
    rec = np.clip(rec, 0, 255)
    return (rec, bits, blocks) if return_coeffs else (rec, bits)


def psnr(a, b, peak=255.0):
    return 10 * np.log10(peak ** 2 / np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


# ============================================================================ audio: psychoacoustics
def bark(f):
    """Critical-band rate (Bark) of frequency f (Hz), Zwicker's approximation."""
    f = np.asarray(f, float)
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def ath(f):
    """Absolute threshold of hearing (dB SPL), Terhardt's approximation; f in Hz."""
    k = np.asarray(f, float) / 1000
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


def spreading(dz):
    """Schroeder spreading function (dB) versus Bark distance dz."""
    dz = np.asarray(dz, float)
    return 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)


def masking_threshold(x, fs=44100, full_scale_db=96.0):
    """Simplified MPEG-1 psychoacoustic model 1 for one frame (len(x) = N, Hann window).

    Tonal maskers are local maxima 7 dB above their neighbours (offset 14.5 + z dB); the rest of
    each critical band forms one noise masker (offset 5.5 dB); maskers spread with the Schroeder
    function and add in power with the threshold in quiet. Levels are dB SPL assuming a full-scale
    sine is `full_scale_db`. Returns (f, P_dB, T_dB)."""
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    P = full_scale_db + 10 * np.log10(np.abs(X) ** 2 / (np.sum(w) / 2) ** 2 + 1e-20)
    f = np.arange(N // 2 + 1) * fs / N
    z = bark(np.maximum(f, 1))
    tonal = [k for k in range(3, len(P) - 3) if P[k] > P[k - 1] and P[k] >= P[k + 1]
             and all(P[k] - P[k + j] >= 7 for j in (-3, -2, 2, 3))]
    lin = 10 ** (P / 10)
    used = np.zeros(len(P), bool)
    maskers = []
    for k in tonal:
        maskers.append((z[k], 10 * np.log10(lin[k - 1:k + 2].sum()), 14.5 + z[k])); used[k - 1:k + 2] = True
    for band in range(25):
        m = (np.floor(z) == band) & ~used & (f > 20)
        if np.any(m):
            maskers.append((np.mean(z[m]), 10 * np.log10(lin[m].sum()), 5.5))
    thr = 10 ** (ath(np.maximum(f, 20)) / 10)
    for zm, L, off in maskers:
        thr = thr + 10 ** ((L + spreading(z - zm) - off) / 10)
    return f, P, 10 * np.log10(thr)


def shaped_noise(x, fs=44100, offset_db=0.0, N=2048, seed=0, full_scale_db=96.0, f_max=16000.0, per_band=True):
    """Add noise whose spectrum follows the frame-by-frame masking threshold + offset_db (below
    f_max; the threshold in quiet explodes above ~18 kHz). White noise is shaped in the STFT domain
    (Hann windows, 50% overlap), then the whole noise is rescaled so that its measured spectrum
    sits on target in the median (self-calibration).

    The maskers of ``masking_threshold`` are band powers, so its threshold is a level per critical
    band. With per_band=True (the conservative, physically consistent choice) the allowed noise
    power is shared among the FFT bins of each critical band, i.e. the per-bin target is
    T - 10 log10(bins in that band); with per_band=False each bin is allowed the full T.
    Returns (y, noise)."""
    r = np.random.default_rng(seed)
    hop = N // 2
    w = np.hanning(N)
    xp = np.concatenate([x, np.zeros(N)])
    noise = np.zeros(len(x) + N)
    targets = []
    for s in range(0, len(x) - N + 1, hop):
        f, P, T = masking_threshold(xp[s:s + N], fs, full_scale_db)
        tgt = T + offset_db
        if per_band:
            zb = np.floor(bark(np.maximum(f, 1))).astype(int)
            tgt = tgt - 10 * np.log10(np.bincount(zb)[zb])
        tgt = np.where(f < f_max, tgt, -200.0)
        targets.append((s, tgt))
        amp = 10 ** ((tgt - full_scale_db) / 20)
        Z = amp * np.exp(2j * np.pi * r.random(len(f)))
        noise[s:s + N] += np.fft.irfft(Z, N) * w
    noise = noise[:len(x)]
    dev = []
    for s, tgt in targets[1:-1:3]:
        f, Pn, _ = masking_threshold(np.r_[noise, np.zeros(N)][s:s + N], fs, full_scale_db)
        band = (f > 100) & (f < f_max)
        dev.append(np.median(Pn[band] - tgt[band]))
    if dev:
        noise *= 10 ** (-np.median(dev) / 20)
    return x + noise, noise
