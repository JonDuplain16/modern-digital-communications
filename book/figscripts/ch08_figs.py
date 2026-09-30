"""Figures for Chapter 8: Baseband Transmission and Pulse Shaping."""
from figstyle import *
from scipy import signal as sps
from scipy.special import erfc
import itertools
import commlib as cl


# ----------------------------------------------------------------------------- helpers
def Qf(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def pam_mf(symbols, beta, sps_, esn0_db=None, r=None, span=12):
    """RRC-shaped real PAM through AWGN and an RRC matched filter.

    Returns (matched-filter output, index of the first symbol's peak, taps).
    Noise is added *before* the matched filter, so the eye shows filtered noise.
    """
    h = cl.rrc_taps(beta, sps_, span)
    x = cl.shape(symbols, h, sps_).real
    if esn0_db is not None:
        es = np.mean(symbols ** 2)
        n0 = es / 10 ** (esn0_db / 10)
        x = x + np.sqrt(n0 / 2) * r.standard_normal(len(x))
    y = np.convolve(x, h[::-1])
    return y, len(h) - 1, h


def eye_plot(ax, y, first_peak, sps_, color, alpha=0.08, n=300, skip=8, centre=0.5):
    """Plot two-symbol traces with a symbol peak at time `centre` (in symbols)."""
    off = first_peak + skip * sps_ - int(round(centre * sps_))
    tr = cl.eye_traces(y, sps_, 2, offset=off, max_traces=n)
    tt = np.arange(tr.shape[1]) / sps_
    ax.plot(tt, tr.T, color=color, alpha=alpha, lw=0.7)
    ax.set_xlim(0, 2)
    return tt, tr


def bitcells(ax, n):
    for k in range(n + 1):
        ax.axvline(k, color=GRAY, lw=0.4, alpha=0.5)


# ----------------------------------------------------------------------------- line codes
def encode_line(bits, code, sps_=64):
    """Return a waveform (sps_ samples per bit) for the named line code."""
    out = []
    if code == "nrzl":
        return np.repeat(2.0 * bits - 1, sps_)
    if code == "nrzi":
        lvl = -1.0
        for b in bits:
            if b: lvl = -lvl
            out.append(lvl)
        return np.repeat(out, sps_)
    if code == "rz":
        h = sps_ // 2
        return np.concatenate([np.r_[np.ones(h) * b, np.zeros(sps_ - h)] for b in bits])
    if code == "manch":  # IEEE 802.3: 1 = low->high, 0 = high->low
        h = sps_ // 2
        return np.concatenate([np.r_[-np.ones(h), np.ones(h)] if b else np.r_[np.ones(h), -np.ones(h)]
                               for b in bits])
    if code == "dmanch":  # transition always mid-bit; a 0 adds a transition at the start
        h = sps_ // 2; lvl = 1.0
        for b in bits:
            if b == 0: lvl = -lvl
            out.append(np.r_[np.ones(h) * lvl, -np.ones(h) * lvl]); lvl = -lvl
        return np.concatenate(out)
    if code == "ami":
        last = -1.0
        for b in bits:
            if b: last = -last; out.append(last)
            else: out.append(0.0)
        return np.repeat(out, sps_)
    if code == "mlt3":
        cyc = [0, 1, 0, -1]; i = 0
        for b in bits:
            if b: i = (i + 1) % 4
            out.append(float(cyc[i]))
        return np.repeat(out, sps_)
    if code == "2b1q":
        m = {(1, 0): 3, (1, 1): 1, (0, 1): -1, (0, 0): -3}
        v = [m[(bits[k], bits[k + 1])] / 3.0 for k in range(0, len(bits) - 1, 2)]
        return np.repeat(v, 2 * sps_)
    raise ValueError(code)


def line_codes():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    n = len(bits); s = 64
    t = np.arange(n * s) / s
    codes = [("NRZ-L (polar)", "nrzl"), ("NRZI (transition on 1)", "nrzi"),
             ("Unipolar RZ (50% duty)", "rz"), ("Manchester (IEEE 802.3: 1 = rising)", "manch"),
             ("Differential Manchester (Token Ring)", "dmanch"), ("AMI / bipolar (T1, E1)", "ami"),
             ("MLT-3 (100BASE-TX)", "mlt3"), ("2B1Q (ISDN BRI U-interface), levels $\\pm1,\\pm3$", "2b1q")]
    fig, ax = plt.subplots(len(codes) + 1, 1, figsize=(W2, 7.6), sharex=True)
    ax[0].step(np.arange(n + 1), np.r_[bits, bits[-1]], where="post", color=GRAY)
    for i, b in enumerate(bits):
        ax[0].text(i + 0.5, 1.25, str(b), ha="center", fontsize=8.5, fontweight="bold")
    ax[0].set_ylim(-0.2, 1.7); ax[0].set_yticks([]); ax[0].set_title("Data bits", fontsize=9, loc="left")
    bitcells(ax[0], n)
    for k, (name, c) in enumerate(codes, start=1):
        w = encode_line(bits, c, s)
        ax[k].plot(t, w, color=CYCLE[(k - 1) % 6], lw=1.3)
        ax[k].set_ylim(-1.45, 1.45); ax[k].set_yticks([-1, 0, 1]); ax[k].grid(False)
        ax[k].set_title(name, fontsize=8.5, loc="left", pad=2)
        bitcells(ax[k], n)
        ax[k].tick_params(labelsize=7)
    ax[-1].set_xlim(0, n); ax[-1].set_xlabel("time (bit periods)")
    fig.tight_layout(h_pad=0.25); save(fig, "ch08_linecodes")


def linecode_psd():
    """Analytic PSDs (T = 1, unit pulse amplitude) with Welch estimates overlaid."""
    f = np.linspace(1e-4, 2.5, 2000)
    S = {
        "polar NRZ": np.sinc(f) ** 2,
        "Manchester": np.sinc(f / 2) ** 2 * np.sin(np.pi * f / 2) ** 2,
        "AMI (full-width)": np.sinc(f) ** 2 * np.sin(np.pi * f) ** 2,
    }
    r = rng(11); nb = 200_000; s = 8
    rb = r.integers(0, 2, nb)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    for i, (nm, code) in enumerate([("polar NRZ", "nrzl"), ("Manchester", "manch"), ("AMI (full-width)", "ami")]):
        ax[0].plot(f, S[nm], color=CYCLE[i], label=nm)
        w = encode_line(rb, code, s)
        fw, pw = sps.welch(w, fs=s, nperseg=2048, return_onesided=False)
        m = (fw > 0.02) & (fw < 2.5)
        ax[0].plot(fw[m], pw[m], ".", color=CYCLE[i], ms=1.5, alpha=0.6)
    m_ = np.arange(-80, 81)
    Rm = 0.5 * np.real(((1 + 1j) / 2) ** np.abs(m_))       # E[a_k a_(k+m)] for MLT-3
    Smlt = np.sinc(f) ** 2 * np.real(np.exp(-2j * np.pi * np.outer(f, m_)) @ Rm)
    ax[0].plot(f, Smlt, color=ORANGE, lw=1.2, label="MLT-3")
    w = encode_line(rb, "mlt3", s)
    fw, pw = sps.welch(w, fs=s, nperseg=2048, return_onesided=False)
    mm = (fw > 0.02) & (fw < 2.5)
    ax[0].plot(fw[mm], pw[mm], ".", color=ORANGE, ms=1.5, alpha=0.6)
    ax[0].set_xlabel("frequency ($f T_b$)"); ax[0].set_ylabel("PSD $S(f)/T_b$")
    ax[0].set_title("Polar codes: continuous spectra", fontsize=9); ax[0].legend(fontsize=6.8)
    ax[0].set_xlim(0, 2.5); ax[0].set_ylim(0, 1.45)
    # unipolar: continuous part + spectral lines
    c_nrz = 0.25 * np.sinc(f) ** 2
    c_rz = (1 / 16) * np.sinc(f / 2) ** 2
    ax[1].plot(f, c_nrz, color=NAVY, label="unipolar NRZ, continuous part")
    ax[1].plot(f, c_rz, color=ACCENT, label="unipolar RZ, continuous part")
    ax[1].annotate("", xy=(0.0, 0.25), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    ax[1].text(0.08, 0.27, "DC line, power 1/4", color=NAVY, fontsize=7.5, va="top")
    for n_ in range(0, 3):
        pw_ = (1 / 16) * np.sinc(n_ / 2) ** 2
        if pw_ < 1e-6: continue
        ax[1].annotate("", xy=(n_ + 0.015, pw_), xytext=(n_ + 0.015, 0.0),
                       arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
    ax[1].text(1.05, 0.03, "clock line at $1/T_b$,\npower $1/(4\\pi^2)$", color=ACCENT, fontsize=7.5)
    ax[1].set_xlabel("frequency ($f T_b$)"); ax[1].set_xlim(-0.05, 2.5); ax[1].set_ylim(0, 0.3)
    ax[1].set_title("Unipolar codes: nonzero mean gives lines", fontsize=9); ax[1].legend(fontsize=6.8, loc="upper right")
    fig.tight_layout(); save(fig, "ch08_linecode_psd")


# ----------------------------------------------------------------------------- 8b/10b and scrambling
_D5 = ["100111", "011101", "101101", "110001", "110101", "101001", "011001", "111000",
       "111001", "100101", "010101", "110100", "001101", "101100", "011100", "010111",
       "011011", "100011", "010011", "110010", "001011", "101010", "011010", "111010",
       "110011", "100110", "010110", "110110", "001110", "101110", "011110", "101011"]
_D3 = ["1011", "1001", "0101", "1100", "1101", "1010", "0110", "1110"]


def _disp(s):
    return s.count("1") * 2 - len(s)


def _comp(s):
    return "".join("1" if c == "0" else "0" for c in s)


def enc8b10b(bytes_):
    """Data-character 8b/10b encoder (IBM/Widmer-Franaszek), returns bit array."""
    rd = -1; out = []
    for B in bytes_:
        x, y = B & 31, B >> 5
        a = _D5[x]
        if x == 7:
            a = "111000" if rd < 0 else "000111"
        elif _disp(a) != 0 and rd > 0:
            a = _comp(a)
        rd = rd + np.sign(_disp(a)) * 2 if _disp(a) != 0 else rd
        rd = 1 if rd > 0 else -1
        if y == 7:
            alt = (rd < 0 and x in (17, 18, 20)) or (rd > 0 and x in (11, 13, 14))
            b = "0111" if alt else "1110"
            if rd > 0: b = _comp(b)
        else:
            b = _D3[y]
            if y == 3:
                b = "1100" if rd < 0 else "0011"
            elif _disp(b) != 0 and rd > 0:
                b = _comp(b)
        if _disp(b) != 0:
            rd = -rd
        out.append(a + b)
    return np.array([int(c) for c in "".join(out)])


def scramble_ss(bits, taps=(39, 58), state=None):
    """Self-synchronous scrambler s[n] = d[n] ^ s[n-39] ^ s[n-58] (IEEE 802.3 64b/66b)."""
    L = max(taps)
    reg = list(state if state is not None else np.ones(L, int))
    out = np.empty(len(bits), int)
    for i, d in enumerate(bits):
        s = d ^ reg[-taps[0]] ^ reg[-taps[1]]
        out[i] = s; reg.append(s); reg.pop(0)
    return out


def run_lengths(b):
    idx = np.flatnonzero(np.diff(b)) + 1
    return np.diff(np.r_[0, idx, len(b)])


def rds_runs():
    r = rng(7); nbytes = 400
    data = r.integers(0, 256, nbytes)
    raw = np.unpackbits(data.astype(np.uint8)).astype(int)
    idle = np.zeros(nbytes * 8, int); idle[::97] = 1   # nearly-constant "idle" payload
    e10 = enc8b10b(data)
    e10_idle = enc8b10b(np.zeros(nbytes, int))
    scr_idle = scramble_ss(idle, state=r.integers(0, 2, 58))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for arr, nm, c in [(raw, "raw random data", GRAY), (idle, "raw idle (mostly zeros)", ORANGE),
                       (scr_idle, "idle, 64b/66b scrambled", GREEN),
                       (e10_idle, "idle, 8b/10b", ACCENT), (e10, "random, 8b/10b", NAVY)]:
        rds = np.cumsum(2 * arr - 1)
        ax[0].plot(np.arange(len(rds)), rds, color=c, lw=1.0, label=nm)
    ax[0].set_ylim(-230, 100); ax[0].set_xlim(0, 3200)
    ax[0].set_xlabel("bit index"); ax[0].set_ylabel("running digital sum")
    ax[0].legend(fontsize=6.3, loc="lower right", ncol=2, framealpha=0.95)
    ax[0].set_title("DC balance: the running digital sum", fontsize=9)
    rl_e = run_lengths(enc8b10b(r.integers(0, 256, 20000)))
    rl_s = run_lengths(scramble_ss(r.integers(0, 2, 160000)))
    k = np.arange(1, 16)
    ax[1].semilogy(k, [np.mean(rl_s == kk) for kk in k], "o-", color=GREEN, ms=3, label="scrambled")
    ax[1].semilogy(k, [max(np.mean(rl_e == kk), 1e-9) for kk in k], "s-", color=NAVY, ms=3, label="8b/10b")
    ax[1].set_ylim(1e-5, 1); ax[1].set_xlabel("run length (bits)"); ax[1].set_ylabel("fraction of runs")
    ax[1].axvline(5, color=NAVY, ls=":", lw=0.8); ax[1].text(5.3, 0.2, "max 5", color=NAVY, fontsize=7.5)
    ax[1].legend(fontsize=7); ax[1].set_title("Run-length distribution", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_rds")
    print("8b/10b max run", rl_e.max(), " RDS range", np.cumsum(2 * e10 - 1).min(), np.cumsum(2 * e10 - 1).max())


# ----------------------------------------------------------------------------- ISI and Nyquist
def isi_sum():
    t = np.linspace(-4, 8, 2000)
    a = np.array([1, -1, 1, 1, -1])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
    for axx, (name, p) in zip(ax, [("sinc pulses (ideal Nyquist)", lambda x: np.sinc(x)),
                                    ("rectangular pulses through an RC low-pass", None)]):
        tot = np.zeros_like(t)
        for k, ak in enumerate(a):
            if p is not None:
                y = ak * p(t - k)
            else:
                tau = 0.6; x = t - k
                y = ak * np.where(x < 0, 0, np.where(x < 1, 1 - np.exp(-x / tau),
                                                     (1 - np.exp(-1 / tau)) * np.exp(-(x - 1) / tau)))
            axx.plot(t, y, lw=0.8, color=GRAY, alpha=0.7); tot += y
        axx.plot(t, tot, color=NAVY, lw=1.6, label="sum")
        samp = np.arange(len(a)) + (0 if p is not None else 0.95)
        axx.plot(samp, np.interp(samp, t, tot), "o", color=ACCENT, ms=5, label="samples")
        if p is None:
            for k, ak in enumerate(a):
                axx.plot([k + 0.95] * 2, [ak * 0.81, np.interp(k + 0.95, t, tot)], color=ACCENT, lw=0.8, ls=":")
        axx.set_title(name, fontsize=9); axx.set_xlabel("time (symbols)")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "ch08_isi")


def rc_pulse(t, b):
    den = 1 - (2 * b * t) ** 2
    sing = np.isclose(den, 0)
    h = np.sinc(t) * np.cos(np.pi * b * t) / np.where(sing, 1, den)
    return np.where(sing, np.pi / 4 * np.sinc(1 / (2 * b)) if b > 0 else 1, h)


def raised_cosine():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    t = np.linspace(-4, 4, 2001)
    f = np.linspace(-1.2, 1.2, 1001)
    for i, b in enumerate([0.0, 0.25, 0.5, 1.0]):
        h = rc_pulse(t, b)
        ax[0].plot(t, h, label=f"$\\beta$ = {b}", color=CYCLE[i])
        H = np.where(np.abs(f) <= (1 - b) / 2, 1.0,
                     np.where(np.abs(f) <= (1 + b) / 2,
                              0.5 * (1 + np.cos(np.pi / max(b, 1e-9) * (np.abs(f) - (1 - b) / 2))), 0))
        ax[1].plot(f, H, color=CYCLE[i])
        tt = np.linspace(0.01, 12, 3000)
        ax[2].semilogy(tt, np.abs(rc_pulse(tt, b)) + 1e-7, color=CYCLE[i], lw=0.9)
    ax[0].set_xlabel("time ($t/T$)"); ax[0].set_title("Impulse response", fontsize=9); ax[0].legend(fontsize=6.5)
    ax[0].set_xticks(range(-4, 5))
    ax[1].set_xlabel("frequency ($fT$)"); ax[1].set_title("Spectrum", fontsize=9)
    for v in (-0.5, 0.5): ax[1].axvline(v, color=GRAY, ls=":", lw=0.8)
    ax[1].annotate("odd symmetry\nabout $1/2T$", xy=(0.5, 0.5), xytext=(0.62, 0.8), fontsize=6.5,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[2].set_ylim(1e-5, 1.5); ax[2].set_xlabel("time ($t/T$)"); ax[2].set_title("Tail decay $|p(t)|$", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_raised_cosine")


def eyes():
    r = rng(3); s = 16
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.0), sharex=True, sharey=True)
    for j, b in enumerate([0.1, 0.35, 1.0]):
        a = 2.0 * r.integers(0, 2, 1500) - 1
        for i, snr in enumerate([None, 15]):
            y, pk, _ = pam_mf(a, b, s, snr, r, span=16)
            eye_plot(ax[i, j], y, pk, s, NAVY if snr is None else PURPLE, alpha=0.07, n=300)
            ax[i, j].set_ylim(-1.9, 1.9)
            if i == 0: ax[i, j].set_title(f"$\\beta$ = {b}", fontsize=9)
        ax[1, j].set_xlabel("time (symbols)")
    ax[0, 0].set_ylabel("noise-free"); ax[1, 0].set_ylabel("$E_s/N_0$ = 15 dB")
    fig.tight_layout(); save(fig, "ch08_eyes")


def eye_anatomy():
    r = rng(5); s = 64
    a = 2.0 * r.integers(0, 2, 1200) - 1
    y, pk, _ = pam_mf(a, 0.5, s, 21, r)
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 3.3))
    eye_plot(ax, y, pk, s, NAVY, alpha=0.05, n=500, centre=0.5)
    # compliance-style hexagonal mask (keep-out region)
    mask = np.array([[0.22, 0], [0.36, 0.55], [0.64, 0.55], [0.78, 0], [0.64, -0.55], [0.36, -0.55]])
    ax.fill(mask[:, 0], mask[:, 1], color=ORANGE, alpha=0.25, lw=1.0, ec=ORANGE)
    ax.text(0.33, 0.1, "mask", ha="center", fontsize=7, color=ORANGE)
    ax.annotate("", xy=(0.5, 0.71), xytext=(0.5, -0.71), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.1))
    ax.text(0.515, -0.3, "eye\nheight", color=ACCENT, fontsize=7.5)
    ax.annotate("", xy=(0.12, 0.0), xytext=(0.88, 0.0), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.1))
    ax.text(0.2, -0.13, "eye width", color=GREEN, fontsize=7.5)
    ax.axvline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.text(0.62, -1.72, "optimum sampling instant", ha="center", fontsize=7.5, color=GRAY)
    ax.annotate("crossing jitter", xy=(1.0, 0.0), xytext=(1.12, 0.35), fontsize=7.5, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.annotate("slope = sensitivity\nto timing error", xy=(0.8, 0.62), xytext=(1.2, 1.5), fontsize=7.5,
                color=PURPLE, arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.annotate("ISI + noise spread\nof the '1' level", xy=(0.5, 1.02), xytext=(0.02, 1.5), fontsize=7.5,
                color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.annotate("overshoot", xy=(0.3, -1.25), xytext=(-0.02, -1.7), fontsize=7.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("time (symbols)"); ax.set_ylim(-1.85, 1.85); ax.set_xlim(0, 2)
    fig.tight_layout(); save(fig, "ch08_eye_anatomy")


def timing_sensitivity():
    r = rng(4); s = 64
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.8))
    for i, b in enumerate([0.1, 0.25, 0.5, 1.0]):
        a = 2.0 * r.integers(0, 2, 3000) - 1
        y, pk, _ = pam_mf(a, b, s, span=24)
        errs = np.linspace(-0.45, 0.45, 61)
        worst = []
        for e in errs:
            idx = pk + (np.arange(40, 2960) * s + int(round(e * s)))
            worst.append(np.min(a[40:2960] * y[idx]))
        ax.plot(errs, 20 * np.log10(np.maximum(worst, 1e-3)), label=f"$\\beta$ = {b}", color=CYCLE[i])
    ax.set_xlabel("timing error ($\\tau/T$)"); ax.set_ylabel("worst-case eye opening (dB)")
    ax.set_ylim(-30, 1); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch08_timing_sensitivity")


# ----------------------------------------------------------------------------- matched filter and BER
def matched_filter_demo():
    r = rng(1); s = 32
    a = np.array([1, -1, -1, 1, -1, 1, 1, 1, -1], float)
    x = np.repeat(a, s)
    y = x + 1.1 * r.standard_normal(len(x))
    # integrate-and-dump correlator
    iad = np.concatenate([np.cumsum(y[k * s:(k + 1) * s]) / s for k in range(len(a))])
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.4), sharex=True)
    t = np.arange(len(x)) / s
    ax[0].plot(t, y, color=GRAY, lw=0.6, label="received $r(t)$ (noisy)")
    ax[0].plot(t, x, color=NAVY, lw=1.6, label="transmitted $\\pm A$")
    ax[0].legend(fontsize=7, ncol=2, loc="upper right"); ax[0].set_ylim(-4, 4.8)
    ax[0].set_title("Per-sample SNR is about $-1$ dB: individual samples are unreliable", fontsize=8.5, loc="left")
    ax[1].plot(t + 1 / s, iad, color=GREEN, lw=1.3, label="integrate-and-dump output")
    k = np.arange(1, len(a) + 1)
    ax[1].plot(k, iad[k * s - 1], "o", color=ACCENT, ms=5, label="decision samples at $t = kT$")
    for kk, ak in zip(k, a):
        ax[1].text(kk - 0.5, 1.45, "+1" if ak > 0 else "$-$1", ha="center", fontsize=7.5, color=NAVY)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylim(-1.7, 1.8)
    ax[1].set_xlabel("time (symbols)")
    ax[1].set_title("After the matched filter: SNR gain of 32 samples = 15 dB", fontsize=8.5, loc="left")
    fig.tight_layout(); save(fig, "ch08_matched_filter")


def ber_binary():
    eb = np.linspace(0, 14, 200)
    g = 10 ** (eb / 10)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 3.2))
    ax.semilogy(eb, Qf(np.sqrt(2 * g)), label="antipodal (polar NRZ, BPSK)")
    ax.semilogy(eb, Qf(np.sqrt(g)), label="orthogonal, or on-off (average $E_b$)")
    ax.semilogy(eb, 0.75 * Qf(np.sqrt(0.8 * g)), label="4-PAM, Gray coded")
    ax.semilogy(eb, 0.5 * np.exp(-g / 2), ls="--", label="noncoherent orthogonal FSK")
    r = rng(2)
    for e in [2, 4, 6, 8, 9]:
        n = 2_000_000; b = r.integers(0, 2, n); x = 2.0 * b - 1
        y = x + r.standard_normal(n) / np.sqrt(2 * 10 ** (e / 10))
        ax.semilogy(e, np.mean((y > 0) != b), "o", color=NAVY, ms=4, label="simulation" if e == 2 else None)
    ax.annotate("", xy=(9.6, 1e-5), xytext=(12.6, 1e-5), arrowprops=dict(arrowstyle="<->", color=GRAY))
    ax.text(11.1, 1.6e-5, "3 dB", ha="center", fontsize=7.5, color=GRAY)
    ax.set_ylim(1e-7, 0.5); ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=7, loc="lower left"); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); save(fig, "ch08_ber_binary")


# ----------------------------------------------------------------------------- jitter and bathtub
def bathtub():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
    # jitter histogram: sinusoidal/ISI-like DJ (dual-Dirac-ish) convolved with RJ
    r = rng(12); n = 400_000
    dj = 0.12 * np.sign(r.standard_normal(n)) * 0.5 + 0.03 * np.sin(2 * np.pi * r.random(n))
    rj = 0.02 * r.standard_normal(n)
    ax[0].hist(dj + rj, bins=150, density=True, color=NAVY, alpha=0.8)
    ax[0].hist(rj, bins=150, density=True, color=ACCENT, alpha=0.35, label="RJ alone")
    ax[0].set_xlabel("edge position (UI)"); ax[0].set_ylabel("density")
    ax[0].set_title("Total jitter histogram", fontsize=9)
    ax[0].annotate("DJ peak-to-peak", xy=(0.075, 3), xytext=(0.0, 14), fontsize=7, ha="center",
                   arrowprops=dict(arrowstyle="-", color="w", lw=0))
    ax[0].annotate("", xy=(-0.075, 12), xytext=(0.075, 12), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax[0].legend(fontsize=7, loc="upper right")
    x = np.linspace(0, 1, 2001)
    rho = 0.5
    for i, (DJ, sig) in enumerate([(0.1, 0.01), (0.1, 0.02), (0.25, 0.02), (0.1, 0.035)]):
        def edge(u):
            return rho / 2 * (Qf((u - DJ / 2) / sig) + Qf((u + DJ / 2) / sig))
        ber = edge(x) + edge(1 - x)
        ax[1].semilogy(x, ber, color=CYCLE[i], label=f"DJ={DJ} UI, $\\sigma_{{RJ}}$={sig} UI")
    ax[1].axhline(1e-12, color=GRAY, ls=":", lw=0.9)
    ax[1].text(0.5, 3e-12, "BER $10^{-12}$", ha="center", fontsize=7, color=GRAY)
    # eye opening at 1e-12 for curve 2
    DJ, sig = 0.1, 0.02
    TJ = DJ + 2 * 7.03 * sig
    ax[1].annotate("", xy=(TJ / 2, 3e-13), xytext=(1 - TJ / 2, 3e-13), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
    ax[1].text(0.5, 6e-14, "eye opening at $10^{-12}$", ha="center", fontsize=7, color=ACCENT)
    ax[1].set_ylim(1e-15, 1); ax[1].set_xlim(0, 1); ax[1].set_xlabel("sampling phase (UI)")
    ax[1].set_ylabel("BER"); ax[1].set_title("Bathtub curves (dual-Dirac model)", fontsize=9)
    ax[1].legend(fontsize=6.3, loc="upper center")
    fig.tight_layout(); save(fig, "ch08_bathtub")


# ----------------------------------------------------------------------------- channel loss and equalisation
def minphase_from_mag(mag, nfft):
    """Minimum-phase impulse response from |H| on an nfft-point full FFT grid (real cepstrum)."""
    c = np.fft.ifft(np.log(np.maximum(mag, 1e-12))).real
    w = np.zeros(nfft); w[0] = 1; w[1:nfft // 2] = 2; w[nfft // 2] = 1
    return np.fft.ifft(np.exp(np.fft.fft(c * w))).real


def backplane(s, il_nyq_db, nfft=1 << 14):
    """Lossy-line model: IL(f) = a sqrt(f) + b f dB (f in units of the baud rate), min-phase."""
    f = np.fft.fftfreq(nfft, 1 / s)            # cycles per UI
    fa = np.abs(f)
    il = il_nyq_db / (0.4 * np.sqrt(0.5) + 0.6 * 0.5) * (0.4 * np.sqrt(fa) + 0.6 * fa)
    h = minphase_from_mag(10 ** (-il / 20), nfft)
    return np.r_[np.zeros(5 * s), h[:40 * s]], f, il


def channel_loss():
    s = 32
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    fs = np.linspace(0, 1.0, 400)
    for i, L in enumerate([10, 20, 30]):
        h, f, il = backplane(s, L)
        m = (f >= 0) & (f <= 1.0)
        ax[0].plot(f[m], -il[m], color=CYCLE[i], label=f"{L} dB at Nyquist")
        p = np.convolve(np.ones(s), h)
        pk = np.argmax(p)
        t = (np.arange(len(p)) - pk) / s
        ax[1].plot(t, p / p[pk], color=CYCLE[i])
        cur = pk + np.arange(-1, 8) * s
        ax[1].plot(t[cur], p[cur] / p[pk], "o", ms=3, color=CYCLE[i])
    ax[0].axvline(0.5, color=GRAY, ls=":", lw=0.8); ax[0].text(0.52, -5, "Nyquist\n$f = 1/2T$", fontsize=7)
    ax[0].set_xlabel("frequency ($fT$)"); ax[0].set_ylabel("insertion gain (dB)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Channel loss (skin + dielectric)", fontsize=9)
    ax[1].set_xlim(-2, 8); ax[1].set_xlabel("time (UI) relative to main cursor")
    ax[1].set_title("Single-bit (pulse) response", fontsize=9)
    ax[1].text(0.25, 0.95, "main cursor $h_0$", fontsize=7)
    ax[1].text(1.6, 0.45, "post-cursor ISI $h_1, h_2, \\ldots$", fontsize=7)
    ax[1].text(-1.9, 0.2, "pre-cursor\n$h_{-1}$", fontsize=7); ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_ylabel("normalized amplitude")
    fig.tight_layout(); save(fig, "ch08_channel_loss")


def equalized_eye():
    s = 32; r = rng(21)
    h, _, _ = backplane(s, 20)
    a = 2.0 * r.integers(0, 2, 3000) - 1
    up = np.repeat(a, s)
    y = np.convolve(up, h)[:len(up)] + 0.012 * r.standard_normal(len(up)) * 0
    p = np.convolve(np.ones(s), h)
    pk = np.argmax(p)
    # UI-spaced pulse response and zero-forcing FFE (2 pre, main, 6 post)
    hs = p[pk - 3 * s: pk + 12 * s: s]
    Lf = 9
    H = np.zeros((len(hs) + Lf - 1, Lf))
    for j in range(Lf): H[j:j + len(hs), j] = hs
    dly = 3 + 2
    e = np.zeros(H.shape[0]); e[dly] = 1
    w, *_ = np.linalg.lstsq(H, e, rcond=None)
    w = w / np.max(np.abs(w))
    wup = np.zeros((Lf - 1) * s + 1); wup[::s] = w
    z = np.convolve(y, wup)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6))
    for axx, sig, off, ttl, c in [(ax[0], y, pk - s + 8 * s, "Received (20 dB loss)", NAVY),
                                  (ax[1], z, pk - s + (2) * s + 8 * s, "After 9-tap FFE", GREEN)]:
        # centre eye on the main-cursor sample
        tr = cl.eye_traces(sig, s, 2, offset=off - s // 2 + s, max_traces=400)
        tt = np.arange(tr.shape[1]) / s
        sc = np.max(np.abs(tr))
        axx.plot(tt, tr.T / sc, color=c, alpha=0.06, lw=0.7)
        axx.set_title(ttl, fontsize=9); axx.set_xlabel("time (UI)"); axx.set_ylim(-1.1, 1.1); axx.set_xlim(0, 2)
    ax[2].stem(np.arange(Lf) - 2, w, basefmt=" ", linefmt=PURPLE, markerfmt="o")
    ax[2].set_xlabel("tap index (UI)"); ax[2].set_title("FFE taps (normalized)", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_equalized_eye")


# ----------------------------------------------------------------------------- PAM-4
def pam4_eye():
    r = rng(9); s = 32
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True)
    a2 = 2.0 * r.integers(0, 2, 1500) - 1
    y2, pk2, _ = pam_mf(a2, 0.5, s, 22, r)
    eye_plot(ax[0], y2, pk2, s, NAVY, alpha=0.05, n=400)
    ax[0].set_title("NRZ (PAM-2), 2 Gb/s at 2 GBd", fontsize=9)
    a4 = (2 * r.integers(0, 4, 1500) - 3) / 3.0
    y4, pk4, _ = pam_mf(a4, 0.5, s, 22 + 10 * np.log10(5 / 9), r)
    eye_plot(ax[1], y4, pk4, s, PURPLE, alpha=0.05, n=400)
    for lv in [-2 / 3, 0, 2 / 3]:
        ax[1].axhline(lv, color=ACCENT, ls=":", lw=0.8)
    for lv, lab in zip([-1, -1 / 3, 1 / 3, 1], ["00", "01", "11", "10"]):
        ax[1].text(2.03, lv, lab, fontsize=7.5, va="center", color=PURPLE, fontweight="bold")
    ax[1].annotate("", xy=(0.5, 1 / 3 + 0.02), xytext=(0.5, 1 - 0.02), arrowprops=dict(arrowstyle="<->", color=ACCENT))
    ax[1].text(0.56, 0.42, "1/3 of the\nNRZ opening", fontsize=7, color=ACCENT)
    ax[1].set_title("PAM-4, 2 Gb/s at 1 GBd (same peak swing)", fontsize=9)
    for a in ax: a.set_xlabel("time (symbols)"); a.set_ylim(-1.5, 1.5)
    fig.tight_layout(); save(fig, "ch08_pam4")


# ----------------------------------------------------------------------------- partial response
def duobinary():
    f = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5))
    ax[0].plot(f, np.where(f <= 0.5, 1, 0), label="ideal Nyquist")
    ax[0].plot(f, np.where(f <= 0.5, np.cos(np.pi * f), 0), label="duobinary $1+D$")
    ax[0].plot(f, np.where(f <= 0.5, np.sin(2 * np.pi * f), 0), label="mod. duobinary $1-D^2$")
    ax[0].set_xlabel("frequency ($fT$)"); ax[0].set_title("Spectra (normalized)", fontsize=9)
    ax[0].legend(fontsize=6.2, loc="upper right"); ax[0].set_ylim(0, 1.5)
    t = np.linspace(-3, 5, 1000)
    ax[1].plot(t, np.sinc(t), color=GRAY, lw=0.8, label="sinc$(t/T)$")
    ax[1].plot(t, np.sinc(t) + np.sinc(t - 1), color=ACCENT, label="sinc$(t/T)$+sinc$(t/T-1)$")
    ax[1].plot([0, 1], [1, 1], "o", color=ACCENT, ms=4)
    ax[1].set_xlabel("time ($t/T$)"); ax[1].set_title("Controlled ISI", fontsize=9)
    ax[1].legend(fontsize=6.2, loc="upper right"); ax[1].set_ylim(-0.4, 1.8)
    # duobinary eye: 1+D applied to RC(beta=0.3)-shaped symbols -> 3 levels
    r = rng(14); s = 32
    a = 2.0 * r.integers(0, 2, 1500) - 1
    d = a[1:] + a[:-1]
    y, pk, _ = pam_mf(d / 2, 0.3, s, 30, r)
    eye_plot(ax[2], y, pk, s, GREEN, alpha=0.05, n=350)
    ax[2].set_title("Duobinary eye: 3 levels", fontsize=9); ax[2].set_xlabel("time (symbols)")
    fig.tight_layout(); save(fig, "ch08_duobinary")


def prml_targets():
    f = np.linspace(0, 0.5, 500)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    for i, D in enumerate([1.0, 2.0, 3.0]):
        H = np.abs(2 * np.sin(np.pi * f)) * np.exp(-np.pi * D * f)
        ax[0].plot(f, H / H.max(), color=CYCLE[i], label=f"Lorentzian, $PW_{{50}}/T$ = {D:g}")
    ax[0].set_xlabel("frequency ($fT$)"); ax[0].set_ylabel("normalized magnitude")
    ax[0].set_title("Recording channel (dibit response)", fontsize=9); ax[0].legend(fontsize=6.5)
    for i, (nm, poly) in enumerate([("PR4  $1-D^2$", [1, 0, -1]), ("EPR4  $(1+D)(1-D^2)$", [1, 1, -1, -1]),
                                    ("E$^2$PR4  $(1+D)^2(1-D^2)$", [1, 2, 0, -2, -1])]):
        Hf = np.abs(np.polyval(poly[::-1], np.exp(-2j * np.pi * f)))
        ax[1].plot(f, Hf / Hf.max(), color=CYCLE[i + 3], label=nm)
    ax[1].set_xlabel("frequency ($fT$)"); ax[1].set_title("Partial-response targets", fontsize=9)
    ax[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch08_prml")


# ----------------------------------------------------------------------------- faster than Nyquist
def ftn_dmin(tau, maxlen=10):
    """Normalized min squared distance d^2/(4 Eb) for binary sinc FTN at spacing tau*T."""
    fgrid = np.linspace(-0.5, 0.5, 801)       # sinc bandwidth, in units of 1/T
    best = np.inf
    for L in range(1, maxlen + 1):
        E = np.exp(-2j * np.pi * np.outer(np.arange(L), fgrid) * tau)
        for tail in itertools.product((-1, 0, 1), repeat=L - 1):
            e = np.r_[1, tail]
            if L > 1 and e[-1] == 0: continue
            spec = np.abs(e @ E) ** 2
            d2 = np.trapezoid(spec, fgrid)   # error-pulse energy / (4 x pulse energy)
            best = min(best, d2)
    return best


def ftn():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    t = np.linspace(-3, 6, 1500)
    tau = 0.8
    a = np.array([1, 1, -1, 1, -1, -1])
    tot = 0
    for k, ak in enumerate(a):
        yk = ak * np.sinc(t - k * tau); tot = tot + yk
        ax[0].plot(t, yk, color=GRAY, lw=0.7)
    ax[0].plot(t, tot, color=NAVY, lw=1.5)
    kk = np.arange(len(a)) * tau
    ax[0].plot(kk, np.interp(kk, t, tot), "o", color=ACCENT, ms=4)
    ax[0].set_title(f"Sinc pulses sent every {tau}$T$", fontsize=9); ax[0].set_xlabel("time ($t/T$)")
    taus = np.r_[np.linspace(0.5, 1.0, 26)]
    d = [ftn_dmin(tt, 8) for tt in taus]
    ax[1].plot(taus, 10 * np.log10(d), "o-", color=ACCENT, ms=3)
    ax[1].axvline(0.802, color=GRAY, ls=":", lw=0.9)
    ax[1].text(0.81, -3.5, "Mazo limit\n$\\tau \\approx 0.802$", fontsize=7.5, color=GRAY)
    ax[1].set_xlabel("time compression $\\tau$"); ax[1].set_ylabel("$d^2_{\\min}/4E_b$ (dB)")
    ax[1].set_title("Minimum distance (error events $\\leq$ 8 bits)", fontsize=9)
    ax[1].set_ylim(-5, 0.5)
    fig.tight_layout(); save(fig, "ch08_ftn")
    print("FTN d2min:", list(zip(np.round(taus, 3), np.round(d, 3))))


if __name__ == "__main__":
    import sys as _s
    fns = [line_codes, linecode_psd, rds_runs, isi_sum, raised_cosine, eyes, eye_anatomy,
           timing_sensitivity, matched_filter_demo, ber_binary, bathtub, channel_loss,
           equalized_eye, pam4_eye, duobinary, prml_targets, ftn]
    sel = _s.argv[1:]
    for f in fns:
        if not sel or f.__name__ in sel:
            f()
