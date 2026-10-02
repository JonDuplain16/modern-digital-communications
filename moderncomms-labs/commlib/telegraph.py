"""The telegraph as a communication system: Morse timing and keying, the Morse/Huffman
comparison and Shannon's telegraph-channel capacity, Kelvin's RC cable, the semaphore
pipeline, analog repeaters versus regenerators, and loaded telephone lines.

Used by Lab 34 and the Chapter 1 figures. Import as ``from commlib import telegraph``.
All functions are vectorised NumPy; nothing here needs a GUI.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.special import erfc

__all__ = ["MORSE", "ENGLISH", "morse_units", "morse_string", "keyed_envelope", "shape_edges",
           "telegraph_capacity", "letter_stats", "cable_response", "alt_swing", "wpm_limit",
           "pipeline_times", "regenerator_ber", "line_gamma", "loaded_line", "loading_cutoff",
           "AWG", "qfunc"]

MORSE = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.',
         'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.',
         'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-',
         'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..',
         '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
         '6': '-....', '7': '--...', '8': '---..', '9': '----.'}

# Approximate relative frequencies of letters in English text (percent; standard tables).
ENGLISH = {'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33,
           'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41,
           'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29, 'V': 0.98,
           'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07}


def qfunc(x):
    """Gaussian tail probability Q(x)."""
    return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))


# ----------------------------------------------------------------------------- Morse
def morse_units(code, letter_gap=True):
    """Duration in dot units: dot 1, dash 3, 1-unit gaps inside the letter, plus the 3-unit
    letter gap (so E = 4, T = 6, Q = 16)."""
    d = sum(1 if c == "." else 3 for c in code) + (len(code) - 1)
    return d + (3 if letter_gap else 0)


def morse_string(text, max_chars=None):
    """The text in dots and dashes ('/' between words), skipping unknown characters."""
    words = []
    for w in text.upper().split():
        codes = [MORSE[c] for c in w if c in MORSE]
        if codes:
            words.append(" ".join(codes))
    s = " / ".join(words)
    if max_chars and len(s) > max_chars:
        s = s[:max_chars - 1] + "…"
    return s


def keyed_envelope(text, sps=20):
    """On/off keying envelope with the ITU timing (dot 1 unit, dash 3, element gap 1,
    letter gap 3, word gap 7), ``sps`` samples per unit. Returns (envelope, units)."""
    seq = []
    for word in text.upper().split():
        chars = [c for c in word if c in MORSE]
        for ci, ch in enumerate(chars):
            for si, sym in enumerate(MORSE[ch]):
                seq += [1] * (1 if sym == "." else 3)
                seq += [0]                      # 1-unit element gap
            seq += [0, 0]                       # letter gap = 3 units in total
        if chars:
            seq += [0] * 4                      # word gap = 7 units in total
    if not seq:
        seq = [0] * 7
    units = len(seq)
    return np.repeat(np.array(seq, float), sps), units


def shape_edges(env, sps, rise):
    """Raised-cosine keying edges: convolve with a Hann window ``rise`` units long (a
    rectangular envelope becomes one whose edges take about ``rise`` units)."""
    L = int(round(rise * sps))
    if L < 1:
        return env.copy()
    w = np.hanning(L + 2)[1:-1]
    w = w / w.sum()
    return np.convolve(env, w, mode="same")


def telegraph_capacity():
    """Shannon's capacity (bits per unit time) of the telegraph channel of his 1948 paper:
    dot 2 units, dash 4, letter space 3, word space 6, and no space after a space.
    Returns about 0.539."""
    from scipy.optimize import brentq
    f = lambda W: -(W ** -2 + W ** -4 - 1) - (W ** -3 + W ** -6) * (W ** -2 + W ** -4)
    return float(np.log2(brentq(f, 1.01, 3.0)))


def letter_stats(text=None):
    """Letter probabilities: the standard English table (text=None) or the letters of
    ``text``. Returns dict letter -> probability (only letters A-Z)."""
    if text is None:
        tot = sum(ENGLISH.values())
        return {k: v / tot for k, v in ENGLISH.items()}
    letters = [c for c in text.upper() if "A" <= c <= "Z"]
    if not letters:
        return {}
    cnt = Counter(letters)
    return {k: n / len(letters) for k, n in cnt.items()}


# ----------------------------------------------------------------------------- Kelvin's cable
def cable_response(levels, T_over_tau, t_over_tau):
    """Far-end voltage of a semi-infinite RC line (Kelvin's model) driven by a sequence of
    element ``levels`` (e.g. +-1, 0) each lasting T (in units of tau = RC l^2), sampled at
    times t (also in units of tau). Superposition of erfc step responses."""
    lv = np.r_[np.asarray(levels, float), 0.0]
    steps = np.diff(np.r_[0.0, lv])                       # level change at k*T
    k = np.nonzero(steps)[0]
    t = np.asarray(t_over_tau, float)
    tk = t[None, :] - k[:, None] * T_over_tau             # (n_steps, n_t)
    pos = tk > 0
    resp = np.where(pos, erfc(1.0 / (2.0 * np.sqrt(np.where(pos, tk, 1.0)))), 0.0)
    return steps[k] @ resp


def alt_swing(T_over_tau, nharm=399):
    """Peak-to-peak received swing (relative to the sent swing) of a +-1 alternating
    pattern with element duration T through the RC line H(w) = exp(-sqrt(j w tau))."""
    r = np.atleast_1d(np.asarray(T_over_tau, float))
    k = np.arange(1, nharm + 1, 2)
    out = np.empty(len(r))
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    for i, ri in enumerate(r):
        w0 = np.pi / ri
        H = np.exp(-np.sqrt(1j * k * w0))
        v = (4 / (np.pi * k) * H) @ np.exp(1j * np.outer(k, th))
        out[i] = (v.imag.max() - v.imag.min()) / 2
    return out


def wpm_limit(R_ohm_km, C_uF_km, length_km, swing_rule=0.24, elements_per_word=19):
    """Words per minute allowed by the 10 %-swing rule (T = 0.24 tau, about 19 cable-code
    elements per word, as in the Chapter 1 worked example)."""
    tau = R_ohm_km * C_uF_km * 1e-6 * np.asarray(length_km, float) ** 2
    return 60.0 / (elements_per_word * swing_rule * tau)


# ----------------------------------------------------------------------------- semaphore
def pipeline_times(n_signs, stations, t_station, t_slow=None, slow_index=None):
    """Arrival times of each sign at each station of a relay line (store-and-forward,
    pipelined). Station 0 originates a sign every t_station; each hop takes t_station
    except the slow station (index ``slow_index``) which takes ``t_slow``.
    Returns an array (n_signs, stations) of the times each sign is displayed."""
    hop = np.full(stations, float(t_station))
    if t_slow is not None and slow_index is not None and 0 <= slow_index < stations:
        hop[slow_index] = float(t_slow)
    T = np.zeros((n_signs, stations))
    for s in range(stations):
        for k in range(n_signs):
            ready_prev = T[k, s - 1] if s > 0 else 0.0           # sign visible upstream
            free = T[k - 1, s] if k > 0 else -np.inf             # this station free again
            T[k, s] = max(ready_prev, free) + hop[s]
    return T


# ----------------------------------------------------------------------------- repeaters
def regenerator_ber(snr_db, spans):
    """End-to-end bit error probability of a chain of binary regenerators, each deciding
    an antipodal signal at the per-span SNR (signal power / noise variance): an odd number
    of errors flips the bit, P = (1 - (1 - 2p)^N) / 2 (about N p when small)."""
    p = qfunc(np.sqrt(10 ** (np.asarray(snr_db, float) / 10)))
    N = np.asarray(spans, float)
    return 0.5 * (1 - np.exp(N * np.log1p(-2 * np.minimum(p, 0.5 - 1e-16))))


# ----------------------------------------------------------------------------- loaded lines
# Typical per-km primary constants of telephone cable pairs (approximate textbook values):
# gauge -> (R ohm/km loop, C uF/km)
AWG = {"19 AWG": (55.0, 0.0516), "22 AWG": (106.0, 0.0516), "24 AWG": (171.0, 0.0516),
       "26 AWG": (272.0, 0.0516)}


def line_gamma(f, R, L, G, C):
    """Propagation constant (per km) of a uniform line; R ohm/km, L H/km, G S/km, C F/km."""
    w = 2 * np.pi * np.asarray(f, float)
    return np.sqrt((R + 1j * w * L) * (G + 1j * w * C))


def loaded_line(f, R, L, G, C, spacing_km, Lcoil, Rcoil=4.0):
    """Attenuation (dB/km) and phase velocity (km/s) of a line with lumped loading coils
    of inductance ``Lcoil`` every ``spacing_km`` (image parameters of one periodic
    section: half section, coil, half section). Above cutoff the attenuation soars."""
    f = np.asarray(f, float)
    w = 2 * np.pi * f
    g = line_gamma(f, R, L, G, C)
    Z0 = np.sqrt((R + 1j * w * L) / (G + 1j * w * C))
    d = spacing_km
    ch, sh = np.cosh(g * d / 2), np.sinh(g * d / 2)
    A1, B1, C1 = ch, Z0 * sh, sh / Z0
    Zc = 1j * w * Lcoil + Rcoil
    A = A1 * (A1 + Zc * C1) + B1 * C1
    D = C1 * (B1 + A1 * Zc) + A1 * A1
    Gam = np.arccosh((A + D) / 2 + 0j)
    Gam = np.where(Gam.real < 0, -Gam, Gam)
    alpha = 20 / np.log(10) * Gam.real / d
    beta = np.abs(np.unwrap(Gam.imag)) / d
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.where(beta > 0, w / beta, np.nan)
    return alpha, vel


def loading_cutoff(Lcoil, C_per_km, spacing_km):
    """Cutoff frequency (Hz) of a coil-loaded line: f_c = 1 / (pi sqrt(L_coil C d))."""
    return 1.0 / (np.pi * np.sqrt(Lcoil * C_per_km * spacing_km))
