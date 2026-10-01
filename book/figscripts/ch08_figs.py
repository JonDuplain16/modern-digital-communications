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


# ============================================================================= second edition
# Concept illustrations and extra data figures (narrow ones are designed for \mdcside).
SIDE = (3.0, 2.4)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_visible(False)


def timeline():
    """History of baseband signalling, 1858-2026."""
    ev = [(1858, "First transatlantic\ncable: pulses smear", 1),
          (1928, "Nyquist: $2W$ pulses/s\nwithout interference", -1),
          (1962, "T1: 1.544 Mb/s\nAMI on copper", 1),
          (1963, "Lender:\nduobinary", -1),
          (1973, "Ethernet at PARC\n(Manchester)", 1),
          (1983, "Widmer & Franaszek:\n8b/10b", -1),
          (1990, "PRML disk\nread channels", 1),
          (1995, "Fast Ethernet:\n4B5B + MLT-3", -1),
          (2002, "10GbE: 64b/66b\n+ scrambling", 1),
          (2017, "400GbE:\nPAM-4 + FEC", -1),
          (2022, "PCIe 6.0:\nPAM-4 flits", 1),
          (2026, "200 Gb/s\nper lane", -1)]
    fig, ax = plt.subplots(figsize=(W2, 2.35))
    n = len(ev)
    ax.plot([-0.6, n - 0.4], [0, 0], color=NAVY, lw=2.2, solid_capstyle="round")
    ax.plot([0.5, 0.5], [-0.12, 0.12], color="white", lw=4)   # break: long gap 1858-1928
    ax.text(0.5, 0, "//", ha="center", va="center", fontsize=9, color=NAVY)
    for i, (y, txt, s) in enumerate(ev):
        x = i
        c = ACCENT if y < 1960 else (NAVY if y < 2000 else PURPLE)
        ax.plot([x, x], [0, 0.5 * s], color=c, lw=0.8)
        ax.plot(x, 0, "o", color=c, ms=5, zorder=3)
        ax.text(x, 0.56 * s, txt, ha="center", va="bottom" if s > 0 else "top", fontsize=6.4, color=c)
        ax.text(x, -0.12 * s, str(y), ha="center", va="top" if s > 0 else "bottom", fontsize=6.6,
                color=GRAY, fontweight="bold")
    ax.set_ylim(-1.35, 1.35); ax.set_xlim(-0.8, n - 0.2)
    _clean(ax)
    fig.tight_layout(); save(fig, "ch08_timeline")


def _diffusion_pulse(t, L, width=1.0):
    """Response of a distributed RC line (diffusion) of normalised length L to a pulse of given width."""
    def step(tt):
        tt = np.maximum(tt, 1e-9)
        return erfc(L / (2 * np.sqrt(tt)))
    return step(t) * (t > 0) - step(t - width) * (t > width)


def cable_smear():
    """Kelvin's law of squares: pulse smearing on a long RC cable."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    t = np.linspace(0, 40, 4000)
    t = np.linspace(0, 80, 8000)
    for i, L in enumerate([2.0, 4.0, 6.0]):
        y = _diffusion_pulse(t, L, 1.0)
        ax[0].plot(t, y / y.max(), color=CYCLE[i], label=f"length {L/2:g}$\\ell$")
        ax[0].plot(t[np.argmax(y)], 1, "o", color=CYCLE[i], ms=3)
    ax[0].plot([0, 0, 1, 1, 40], [0, 1, 1, 0, 0], color=GRAY, lw=0.8, ls="--", label="sent pulse")
    ax[0].set_xlim(0, 25); ax[0].set_ylim(0, 1.15)
    ax[0].set_xlabel("time (pulse widths)"); ax[0].set_ylabel("received (normalised)")
    ax[0].set_title("One pulse, three cable lengths", fontsize=9); ax[0].legend(fontsize=6.5)
    # a short message: dot dot dash at two lengths
    msg = [(0, 1), (2, 3), (4, 7)]   # (start, stop) in pulse widths
    tt = np.linspace(0, 60, 6000)
    for i, (L, c) in enumerate([(0.6, NAVY), (2.0, ACCENT)]):
        y = sum(_diffusion_pulse(tt - a, L, b - a) * (tt > a) for a, b in msg)
        ax[1].axhline(1.25 * (1 - i), color=c, lw=0.5, ls=":")
        ax[1].plot(tt, y / y.max() + 1.25 * (1 - i), color=c, lw=1.3,
                   label="short cable" if i == 0 else "cable 3.3$\\times$ longer")
    for a, b in msg:
        ax[1].fill_between([a, b], 2.45, 2.6, color=GRAY, alpha=0.6, lw=0)
    ax[1].text(8, 2.5, "sent: dot, dot, dash", fontsize=7, color=GRAY, va="center")
    ax[1].set_xlim(0, 45); ax[1].set_yticks([]); ax[1].set_ylim(-0.1, 2.75)
    ax[1].set_xlabel("time (pulse widths)"); ax[1].legend(fontsize=6.5, loc="center right")
    ax[1].set_title("The symbols run together", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_cable_smear")


def rc_p(t, b=0.35):
    return rc_pulse(t, b)


def bits_to_pam():
    """The PAM model: impulses a_k, scaled pulses a_k p(t-kT), and their sum."""
    a = np.array([1, -1, -1, 1, 1, -1, 1, -1])
    t = np.linspace(-2, 10, 3000)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.3), sharex=True)
    ax[0].stem(np.arange(len(a)), a, linefmt=NAVY, markerfmt="o", basefmt=" ")
    for k, ak in enumerate(a):
        ax[0].text(k + 0.15, ak * 0.55, f"{int(ak):+d}", fontsize=7, color=NAVY)
    ax[0].set_ylim(-1.6, 1.6); ax[0].set_title("1. symbols $a_k$, one every $T$ seconds", fontsize=8.5, loc="left")
    tot = 0
    for k, ak in enumerate(a):
        y = ak * rc_p(t - k, 0.5); tot = tot + y
        ax[1].plot(t, y, color=CYCLE[k % 6], lw=1.0)
    ax[1].set_ylim(-1.3, 1.3); ax[1].set_title("2. each symbol launches a scaled copy of the pulse $a_k\\,p(t-kT)$", fontsize=8.5, loc="left")
    ax[2].plot(t, tot, color=NAVY, lw=1.6)
    ax[2].plot(np.arange(len(a)), a, "o", color=ACCENT, ms=4)
    ax[2].set_title("3. the transmitted signal $s(t)=\\sum_k a_k\\,p(t-kT)$; samples at $kT$ return the symbols",
                    fontsize=8.5, loc="left")
    ax[2].set_ylim(-1.6, 1.6); ax[2].set_xlabel("time ($t/T$)"); ax[2].set_xlim(-1.5, 8.5)
    for x in ax: x.tick_params(labelsize=7)
    fig.tight_layout(h_pad=0.3); save(fig, "ch08_bits_to_pam")


def staircase():
    """Same symbol rate: binary vs PAM-4 (bits per symbol = steps of the staircase)."""
    r = rng(31)
    bits = r.integers(0, 2, 24)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    n = 12
    lv2 = 2 * bits[:n] - 1
    ax[0].step(np.arange(n + 1), np.r_[lv2, lv2[-1]], where="post", color=NAVY)
    for k in range(n):
        ax[0].text(k + 0.5, 1.25, str(bits[k]), ha="center", fontsize=7)
    ax[0].set_title("NRZ: 12 symbols carry 12 bits", fontsize=8.5)
    gray = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    lv4 = np.array([gray[(bits[2 * k], bits[2 * k + 1])] for k in range(n)]) / 3
    ax[1].step(np.arange(n + 1), np.r_[lv4, lv4[-1]], where="post", color=PURPLE)
    for k in range(n):
        ax[1].text(k + 0.5, 1.25, f"{bits[2*k]}{bits[2*k+1]}", ha="center", fontsize=6.5)
    for l_ in [-1, -1 / 3, 1 / 3, 1]:
        ax[1].axhline(l_, color=GRAY, lw=0.4, ls=":")
    ax[1].set_title("PAM-4: the same 12 symbols carry 24 bits", fontsize=8.5)
    for x in ax:
        x.set_ylim(-1.35, 1.5); x.set_xlabel("time (symbol periods)"); x.set_xlim(0, n); x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_staircase")


def line_code_jobs():
    """Why line codes exist: baseline wander (AC coupling) and clock starvation."""
    s = 32
    bits = np.r_[1, 0, 1, 1, 0, 1, np.ones(14, int), 0, 1, 0, 0, 1, 0]
    x = np.repeat(2.0 * bits - 1, s)
    bb, aa = sps.butter(1, 0.012 / (s / 2) * s / 4, btype="high")
    y = sps.lfilter(bb, aa, x)
    m = encode_line(bits, "manch", s)
    ym = sps.lfilter(bb, aa, m)
    t = np.arange(len(x)) / s
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax[0].plot(t, x, color=GRAY, lw=0.8, ls="--", label="sent (NRZ)")
    ax[0].plot(t, y, color=ACCENT, lw=1.3, label="after AC coupling")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].annotate("long run of ones:\nlevel sags toward\nthe threshold", xy=(17, y[17 * s]), xytext=(8.5, -1.75),
                   fontsize=6.8, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax[0].set_title("Baseline wander", fontsize=9); ax[0].set_ylim(-2.2, 1.6)
    ax[0].legend(fontsize=6.3, loc="upper right"); ax[0].set_xlabel("time (bits)")
    # clock drift: receiver clock 3% slow, only re-aligned at transitions
    ax[1].step(np.arange(len(bits) + 1), np.r_[2 * bits - 1, 2 * bits[-1] - 1], where="post", color=NAVY, lw=1.1)
    trans = np.r_[0, np.flatnonzero(np.diff(bits)) + 1]
    samp = []; last = 0
    for k in range(len(bits)):
        if k in trans: last = k
        samp.append(k + 0.5 + 0.045 * (k - last))
    samp = np.array(samp)
    off = samp - np.floor(samp)
    col = [GREEN if 0.2 < o < 0.8 else ACCENT for o in off]
    for k, (sp_, c) in enumerate(zip(samp, col)):
        ax[1].plot(sp_, 1.35, "v", color=c, ms=4)
    ax[1].annotate("no transitions: the\nsampling clock drifts\ntoward the bit edge", xy=(19.4, 1.35), xytext=(10.5, -0.6),
                   fontsize=6.8, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax[1].set_ylim(-1.4, 1.6); ax[1].set_title("Clock starvation", fontsize=9); ax[1].set_xlabel("time (bits)")
    ax[1].set_xlim(0, len(bits))
    fig.tight_layout(); save(fig, "ch08_line_code_jobs")


def b8zs_hdb3():
    """AMI with B8ZS and HDB3 substitutions, violations marked."""
    bits = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    def ami(b):
        out = []; last = -1
        for x in b:
            if x: last = -last; out.append(last)
            else: out.append(0)
        return np.array(out, float)
    def b8zs(b):
        out = []; last = -1; i = 0; viol = []
        while i < len(b):
            if b[i] == 0 and i + 8 <= len(b) and not b[i:i + 8].any():
                V = last; B = -last
                pat = [0, 0, 0, V, B, 0, B, V]
                viol += [i + 3, i + 6]
                out += pat; last = V; i += 8; continue
            if b[i]: last = -last; out.append(last)
            else: out.append(0)
            i += 1
        return np.array(out, float), viol
    def hdb3(b):
        out = []; last = -1; i = 0; viol = []; nb = 0; lastv = 1
        while i < len(b):
            if b[i] == 0 and i + 4 <= len(b) and not b[i:i + 4].any():
                if nb % 2 == 1:      # odd number of marks since last V: 000V
                    V = last; out += [0, 0, 0, V]
                else:                 # even: B00V
                    B = -last; V = B; out += [B, 0, 0, V]; last = B
                last = V; viol.append(i + 3); nb = 0; i += 4; continue
            if b[i]: last = -last; out.append(last); nb += 1
            else: out.append(0)
            i += 1
        return np.array(out, float), viol
    fig, ax = plt.subplots(4, 1, figsize=(W2, 3.6), sharex=True)
    n = len(bits)
    ax[0].step(np.arange(n + 1), np.r_[bits, bits[-1]], where="post", color=GRAY)
    for k, b in enumerate(bits): ax[0].text(k + 0.5, 1.2, str(b), ha="center", fontsize=7.5, fontweight="bold")
    ax[0].set_ylim(-0.2, 1.6); ax[0].set_yticks([]); ax[0].set_title("Data", fontsize=8.5, loc="left")
    rows = [("Plain AMI: a long silence, no timing information", ami(bits), [], NAVY),
            ("B8ZS (T1): eight zeros become 000VB0VB", *b8zs(bits), ACCENT),
            ("HDB3 (E1): four zeros become 000V or B00V", *hdb3(bits), GREEN)]
    for x, (nm, w, vi, c) in zip(ax[1:], rows):
        x.step(np.arange(n + 1), np.r_[w, w[-1]], where="post", color=c, lw=1.3)
        for v in vi:
            x.text(v + 0.5, w[v] * 0.45, "V", ha="center", va="center", fontsize=7.5, color="white",
                   bbox=dict(boxstyle="circle,pad=0.12", fc=c, ec="none"))
        x.set_ylim(-1.5, 1.5); x.set_yticks([-1, 0, 1]); x.set_title(nm, fontsize=8.5, loc="left", pad=2)
        x.tick_params(labelsize=7)
    for x in ax: bitcells(x, n)
    ax[-1].set_xlim(0, n); ax[-1].set_xlabel("time (bit periods)")
    fig.tight_layout(h_pad=0.3); save(fig, "ch08_b8zs")


def psd_factor():
    """PSD = pulse term x symbol term, shown for AMI."""
    f = np.linspace(0, 2.2, 800)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    P = np.sinc(f) ** 2; Sa = np.sin(np.pi * f) ** 2
    for x, y, ttl, c in [(ax[0], P, "pulse term $|P(f)|^2/T$", NAVY),
                         (ax[1], Sa, "symbol term $S_a(f)$", PURPLE),
                         (ax[2], P * Sa, "AMI spectrum = product", ACCENT)]:
        x.plot(f, y, color=c); x.fill_between(f, y, color=c, alpha=0.12)
        x.set_title(ttl, fontsize=8.5); x.set_xlabel("$fT_b$"); x.set_xlim(0, 2.2); x.set_ylim(0, 1.1)
        x.tick_params(labelsize=7)
    ax[0].text(2.45, 0.5, "$\\times$", fontsize=16, transform=ax[0].transData, va="center")
    ax[1].text(2.4, 0.5, "$=$", fontsize=16, va="center")
    fig.tight_layout(w_pad=2.2); save(fig, "ch08_psd_factor")


def nyquist_fold():
    """The Nyquist criterion as 'shifted copies must tile a flat floor'."""
    f = np.linspace(-1.6, 1.6, 3000)
    def rc(ff, b, W=0.5):
        a = np.abs(ff)
        if b == 0: return (a <= W).astype(float)
        return np.where(a <= W * (1 - b), 1, np.where(a <= W * (1 + b),
                        0.5 * (1 + np.cos(np.pi / (2 * W * b) * (a - W * (1 - b)))), 0))
    cases = [("too narrow ($W<1/2T$):\ngaps, so ISI is unavoidable", lambda ff: rc(ff, 0, 0.38)),
             ("brick wall at $1/2T$:\ntiles exactly (sinc pulse)", lambda ff: rc(ff, 0)),
             ("raised cosine, $\\beta=0.5$:\noverlaps sum to flat", lambda ff: rc(ff, 0.5))]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    for x, (ttl, X) in zip(ax, cases):
        tot = 0
        for k in range(-3, 4):
            y = X(f - k); tot = tot + y
            x.plot(f, y, color=CYCLE[(k + 3) % 6], lw=0.9, alpha=0.8)
        x.plot(f, tot, color="k", lw=1.6)
        x.set_title(ttl, fontsize=8); x.set_xlabel("$fT$"); x.set_xlim(-1.5, 1.5); x.set_ylim(0, 1.35)
        x.tick_params(labelsize=7)
    ax[0].set_ylabel("$\\sum_k X(f-k/T)$")
    fig.tight_layout(); save(fig, "ch08_nyquist_fold")


def rrc_vs_rc():
    t = np.linspace(-4, 4, 2001)
    h = cl.rrc_taps(0.35, 64, 8)
    th = (np.arange(len(h)) - (len(h) - 1) / 2) / 64
    fig, ax = plt.subplots(figsize=SIDE)
    ax.plot(th, h / h.max(), color=ORANGE, label="RRC (one half)")
    ax.plot(t, rc_pulse(t, 0.35), color=NAVY, label="RRC $*$ RRC = RC")
    k = np.arange(-4, 5)
    ax.plot(k, rc_pulse(k.astype(float), 0.35), "o", color=NAVY, ms=3.5)
    hk = np.interp(k, th, h / h.max())
    ax.plot(k[k != 0], hk[k != 0], "s", color=ORANGE, ms=3.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time ($t/T$)"); ax.legend(fontsize=6.5, loc="upper right"); ax.set_xticks(range(-4, 5))
    ax.set_title("Only the pair is Nyquist", fontsize=9); ax.set_ylim(-0.3, 1.15)
    fig.tight_layout(); save(fig, "ch08_rrc_vs_rc")


def papr():
    """CCDF of instantaneous power for RRC-shaped QPSK (as in the roll-off discussion)."""
    r = rng(0); s = 8
    fig, ax = plt.subplots(figsize=SIDE)
    for i, b in enumerate([0.35, 0.2, 0.05]):
        h = cl.rrc_taps(b, s, 40)
        sy = (2 * r.integers(0, 2, 100000) - 1 + 1j * (2 * r.integers(0, 2, 100000) - 1)) / np.sqrt(2)
        x = cl.shape(sy, h, s)[len(h):-len(h)]
        p = 10 * np.log10(np.abs(x) ** 2 / np.mean(np.abs(x) ** 2))
        ps = np.sort(p); cc = 1 - np.arange(len(ps)) / len(ps)
        ax.semilogy(ps, cc, color=CYCLE[i], label=f"$\\beta$ = {b}")
    ax.axhline(1e-4, color=GRAY, ls=":", lw=0.8)
    ax.set_xlim(0, 8); ax.set_ylim(1e-5, 1); ax.set_xlabel("power above average (dB)")
    ax.set_ylabel("probability exceeded"); ax.legend(fontsize=7); ax.set_title("Peaks grow as $\\beta$ shrinks", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_papr")


def rrc_truncation():
    from scipy.signal import freqz
    s = 8
    fig, ax = plt.subplots(figsize=SIDE)
    for i, span in enumerate([8, 20]):
        h = cl.rrc_taps(0.22, s, span)
        w, H = freqz(h, worN=8192, fs=s)
        ax.plot(w, 20 * np.log10(np.abs(H) / np.abs(H[0]) + 1e-9), color=CYCLE[i], lw=1.0, label=f"span {span} symbols")
    ax.axvline(0.61, color=GRAY, ls=":", lw=0.8)
    ax.text(0.65, -8, "band edge\n$(1+\\beta)/2T$", fontsize=6.5, color=GRAY)
    ax.set_xlim(0, 2); ax.set_ylim(-80, 5); ax.set_xlabel("frequency ($fT$)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.8, loc="lower left"); ax.set_title("Truncated RRC, $\\beta=0.22$", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_rrc_truncation")


def template_match():
    """Matched filter as template matching: find a known shape in noise."""
    r = rng(8); n = 1200
    tp = np.arange(80)
    pulse = np.sin(2 * np.pi * (tp / 80) * (1 + 2.5 * tp / 80)) * np.hanning(80)
    pulse /= np.sqrt(np.sum(pulse ** 2))
    x = np.zeros(n); pos = [250, 700]
    for p_ in pos: x[p_:p_ + 80] += 4.2 * pulse
    y = x + 0.9 * r.standard_normal(n)
    c = np.correlate(y, pulse, mode="full")[79:79 + n]
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.0), gridspec_kw={"height_ratios": [0.7, 1, 1]})
    ax[0].plot(tp, pulse, color=PURPLE); ax[0].set_title("the template: the pulse we are looking for", fontsize=8.5, loc="left")
    ax[0].set_xlim(0, n); _clean(ax[0])
    ax[1].plot(y, color=GRAY, lw=0.6); ax[1].plot(x, color=PURPLE, lw=0.9, alpha=0.8)
    ax[1].set_title("what arrives: two copies buried in noise (purple, invisible to the eye in grey)", fontsize=8.5, loc="left")
    ax[1].set_xlim(0, n); ax[1].tick_params(labelsize=7)
    ax[2].plot(c, color=NAVY, lw=1.0)
    for p_ in pos: ax[2].plot(p_, c[p_], "o", color=ACCENT, ms=5)
    ax[2].set_title("slide the template along and multiply-and-add: peaks mark the pulses", fontsize=8.5, loc="left")
    ax[2].set_xlim(0, n); ax[2].set_xlabel("time (samples)"); ax[2].tick_params(labelsize=7)
    fig.tight_layout(h_pad=0.3); save(fig, "ch08_template_match")


def mf_sieve():
    """Matched filter in frequency: pass where signal is strong (white) / signal-to-noise is high (coloured)."""
    f = np.linspace(0, 2, 600)
    P2 = np.sinc(f) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    ax[0].fill_between(f, P2, color=NAVY, alpha=0.15); ax[0].plot(f, P2, color=NAVY, label="signal $|P(f)|^2$")
    ax[0].axhline(0.25, color=GRAY, ls="--", lw=1, label="white noise")
    ax[0].plot(f, np.abs(np.sinc(f)), color=ACCENT, lw=1.5, label="matched filter $|H|=|P|$")
    ax[0].set_title("White noise: shape the sieve like the grain", fontsize=8.5)
    Sn = 0.08 + 0.5 * f ** 2
    H = np.abs(np.sinc(f)) / Sn; H /= H.max()
    ax[1].fill_between(f, P2, color=NAVY, alpha=0.15); ax[1].plot(f, P2, color=NAVY)
    ax[1].plot(f, Sn, color=GRAY, ls="--", lw=1, label="coloured noise $S_n(f)$")
    ax[1].plot(f, H, color=ACCENT, lw=1.5, label="$|P|/S_n$ (normalised)")
    ax[1].set_title("Coloured noise: avoid where noise is loud", fontsize=8.5)
    for x in ax:
        x.set_xlabel("frequency ($fT$)"); x.set_ylim(0, 1.25); x.set_xlim(0, 2); x.legend(fontsize=6.3, loc="upper right")
        x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_mf_sieve")


def decision():
    """Two Gaussians, threshold, error tails; MAP shift for unequal priors."""
    y = np.linspace(-3.5, 3.5, 1000)
    sg = 0.6
    g = lambda m: np.exp(-(y - m) ** 2 / (2 * sg ** 2)) / (sg * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    for x, (P0, P1, ttl) in zip(ax, [(0.5, 0.5, "equal priors: threshold at the midpoint"),
                                    (0.2, 0.8, "$P(+1)=0.8$: MAP threshold moves left")]):
        f0 = P0 * g(-1); f1 = P1 * g(1)
        gam = sg ** 2 / 2 * np.log(P0 / P1)
        x.plot(y, f0, color=NAVY, label="sent $-1$"); x.plot(y, f1, color=PURPLE, label="sent $+1$")
        x.fill_between(y, f0, where=y > gam, color=NAVY, alpha=0.35)
        x.fill_between(y, f1, where=y < gam, color=PURPLE, alpha=0.35)
        x.axvline(gam, color=ACCENT, lw=1.2); x.text(gam + 0.06, 0.62, "threshold", color=ACCENT, fontsize=7)
        x.set_title(ttl, fontsize=8.3); x.set_xlabel("matched-filter sample $y$"); x.tick_params(labelsize=7)
    ax[0].annotate("error tails\n(shaded)", xy=(0.25, 0.07), xytext=(1.6, 0.45), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_ylim(0, 0.75)
    fig.tight_layout(); save(fig, "ch08_decision")


def signal_geometry():
    """Antipodal, orthogonal and on-off signals as points: distance is everything."""
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1))
    pts = [("antipodal", [(-1, 0), (1, 0)], "$d=2\\sqrt{E_b}$"),
           ("orthogonal", [(1, 0), (0, 1)], "$d=\\sqrt{2E_b}$"),
           ("on-off (same average $E_b$)", [(0, 0), (np.sqrt(2), 0)], "$d=\\sqrt{2E_b}$")]
    for x, (nm, pp, dl) in zip(ax, pts):
        circ = plt.Circle((0, 0), 1, color=GRAY, fill=False, ls=":", lw=0.8)
        x.add_patch(circ)
        (x0, y0), (x1, y1) = pp
        x.plot([x0, x1], [y0, y1], color=ACCENT, lw=1.2, ls="--")
        x.plot([x0, x1], [y0, y1], "o", color=NAVY, ms=7)
        tx, ty = ((x0 + x1) / 2 + 0.42, (y0 + y1) / 2 + 0.22) if nm == "orthogonal" else ((x0 + x1) / 2, (y0 + y1) / 2 + 0.18)
        x.text(tx, ty, dl, ha="center", fontsize=7.5, color=ACCENT)
        x.set_title(nm, fontsize=8.5); x.set_aspect("equal"); x.set_xlim(-1.4, 1.6); x.set_ylim(-1.2, 1.3)
        x.axhline(0, color=GRAY, lw=0.4); x.axvline(0, color=GRAY, lw=0.4); _clean(x)
    ax[0].text(0, -1.15, "circle: radius $\\sqrt{E_b}$", ha="center", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch08_geometry")


def mpam_ber():
    eb = np.linspace(0, 26, 300); g = 10 ** (eb / 10)
    fig, ax = plt.subplots(figsize=SIDE)
    for i, M in enumerate([2, 4, 8, 16]):
        k = np.log2(M)
        Pb = 2 * (1 - 1 / M) * Qf(np.sqrt(6 * k / (M ** 2 - 1) * g)) / k
        ax.semilogy(eb, Pb, color=CYCLE[i], label=f"{M}-PAM")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=0.8)
    ax.set_ylim(1e-8, 0.5); ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("Each extra bit costs ~6 dB", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_mpam_ber")


def eye_build():
    """How an eye diagram is made: slice the waveform every 2 symbols and stack the slices."""
    r = rng(17); s = 32
    a = 2.0 * r.integers(0, 2, 400) - 1
    y, pk, _ = pam_mf(a, 0.5, s, 25, r)
    y = y[pk + 10 * s - s // 2:]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [2.2, 1]})
    t = np.arange(12 * s + 1) / s
    cols = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, "#2E86C1"]
    for k in range(6):
        seg = slice(2 * k * s, 2 * (k + 1) * s + 1)
        ax[0].plot(t[seg], y[seg], color=cols[k], lw=1.3)
        ax[1].plot(np.arange(2 * s + 1) / s, y[seg], color=cols[k], lw=1.3)
    for k in range(7): ax[0].axvline(2 * k, color=GRAY, lw=0.5, ls=":")
    tr = cl.eye_traces(y, s, 2, offset=0, max_traces=150)
    ax[1].plot(np.arange(tr.shape[1]) / s, tr.T, color=GRAY, alpha=0.08, lw=0.6)
    ax[0].set_title("the waveform, cut every two symbols", fontsize=8.5); ax[0].set_xlabel("time (symbols)")
    ax[1].set_title("...stacked: an eye", fontsize=8.5); ax[1].set_xlabel("time (symbols)"); ax[1].set_xlim(0, 2)
    for x in ax: x.set_ylim(-1.8, 1.8); x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_eye_build")


def jitter_types():
    r = rng(23); n = 120
    k = np.arange(n)
    rj = 0.03 * r.standard_normal(n)
    pj = 0.08 * np.sin(2 * np.pi * k / 40)
    bits = r.integers(0, 2, n)
    ddj = 0.07 * (bits - 0.5) * 2
    fig, ax = plt.subplots(3, 1, figsize=SIDE, sharex=True)
    for x, (y, nm, c) in zip(ax, [(rj, "random (Gaussian)", NAVY), (pj, "periodic (supply ripple)", ORANGE),
                                 (ddj, "data-dependent (ISI)", PURPLE)]):
        x.plot(k, y, ".", color=c, ms=2.5); x.axhline(0, color="k", lw=0.4)
        x.set_ylim(-0.13, 0.13); x.set_yticks([]); x.text(1, 0.085, nm, fontsize=6.8, color=c)
        x.tick_params(labelsize=6.5)
    ax[1].set_ylabel("edge error (UI)", fontsize=7); ax[-1].set_xlabel("edge number", fontsize=7.5)
    ax[0].set_title("Three kinds of jitter", fontsize=9)
    fig.tight_layout(h_pad=0.2); save(fig, "ch08_jitter_types")


def bangbang():
    """Alexander (bang-bang) phase detector: data and edge samples."""
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    s = 100; n = len(bits)
    lv = 2.0 * bits - 1
    w = np.repeat(np.r_[lv[0], lv, lv[-1]], s)
    w = np.convolve(w, np.ones(18) / 18, mode="same")[s:-s]
    t = np.arange(n * s) / s
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.plot(t, w, color=NAVY, lw=1.3)
    late = 0.12
    for k in range(n):
        tc = k + 0.5 + late
        ax.plot(tc, np.interp(tc, t, w), "o", color=GREEN, ms=5, label="data sample (bit centre)" if k == 0 else None)
        if k < n - 1:
            te = k + 1 + late
            ax.plot(te, np.interp(te, t, w), "s", color=ACCENT, ms=4.5, label="edge sample (bit boundary)" if k == 0 else None)
    for k in range(n + 1): ax.axvline(k, color=GRAY, lw=0.4, ls=":")
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22), frameon=False)
    ax.annotate("edge sample already equals the\nnew bit: the clock is late", xy=(1 + late, np.interp(1 + late, t, w)),
                xytext=(1.5, -0.55), fontsize=6.8, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax.set_ylim(-1.4, 1.4); ax.set_xlim(0, n); ax.set_yticks([-1, 1]); ax.set_xlabel("time (UI)")
    ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_bangbang")


def skin_effect():
    """Current crowding in a round conductor at three frequencies + loss components."""
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), gridspec_kw={"width_ratios": [1, 1, 1, 2.2]})
    R = 1.0
    xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 300), np.linspace(-1.1, 1.1, 300))
    rr = np.hypot(xx, yy)
    for x, (d, lab) in zip(ax[:3], [(5, "low frequency"), (0.3, "100 MHz"), (0.06, "10 GHz")]):
        J = np.exp(-(R - rr) / d); J[rr > R] = np.nan
        x.imshow(J, extent=(-1.1, 1.1, -1.1, 1.1), cmap="Oranges", vmin=0, vmax=1)
        x.add_patch(plt.Circle((0, 0), R, fill=False, color=NAVY, lw=1))
        x.set_title(lab, fontsize=7.8); x.set_aspect("equal"); _clean(x)
    ax[0].text(0, -1.45, "current density", ha="center", fontsize=6.8, color=GRAY)
    f = np.linspace(0.01, 30, 400)
    sk = 1.6 * np.sqrt(f); di = 0.55 * f
    ax[3].plot(f, sk, color=ORANGE, label="skin: $\\propto\\sqrt{f}$")
    ax[3].plot(f, di, color=PURPLE, label="dielectric: $\\propto f$")
    ax[3].plot(f, sk + di, color=NAVY, lw=1.6, label="total")
    ax[3].set_xlabel("frequency (GHz)", fontsize=7.5); ax[3].set_ylabel("loss (dB)", fontsize=7.5)
    ax[3].legend(fontsize=6.3); ax[3].tick_params(labelsize=6.5); ax[3].set_title("illustrative trace loss", fontsize=7.8)
    fig.tight_layout(w_pad=0.6); save(fig, "ch08_skin_effect")


def deemphasis():
    """Transmit de-emphasis: shrink the repeated bits so the transitions survive the channel."""
    s = 32
    bits = np.array([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0])
    c1 = -0.3
    pre = lambda a: (a + c1 * np.r_[a[0], a[:-1]]) / (1 - c1)
    a = 2.0 * bits - 1; ad = pre(a)
    h, _, _ = backplane(s, 18)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    t = np.arange(len(a) * s) / s
    ax[0].plot(t, np.repeat(a, s), color=GRAY, lw=0.8, ls="--", label="plain NRZ")
    ax[0].plot(t, np.repeat(ad, s), color=GREEN, lw=1.3, label="de-emphasised")
    ax[0].set_title("Transmitter output", fontsize=8.5); ax[0].legend(fontsize=6.2, loc="upper left", ncol=2)
    ax[0].set_ylim(-1.4, 1.75); ax[0].set_xlabel("time (UI)")
    r = rng(19); ar = 2.0 * r.integers(0, 2, 3000) - 1
    p = np.convolve(np.ones(s), h); pk = np.argmax(p)
    for x, sym, ttl, c in [(ax[1], ar, "eye, plain", GRAY), (ax[2], pre(ar), "eye, de-emphasised", GREEN)]:
        y = np.convolve(np.repeat(sym, s), h)
        tr = cl.eye_traces(y, s, 2, offset=pk + 20 * s - s // 2 - s // 2 + s, max_traces=300)
        tr = tr / np.percentile(np.abs(tr), 99.5)
        x.plot(np.arange(tr.shape[1]) / s, tr.T, color=c if c != GRAY else NAVY, alpha=0.08, lw=0.6)
        x.set_title(ttl, fontsize=8.5); x.set_xlim(0, 2); x.set_ylim(-1.2, 1.2); x.set_xlabel("time (UI)")
    for x in ax: x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_deemphasis")


def dfe_concept():
    s = 32
    h, _, _ = backplane(s, 20)
    p = np.convolve(np.ones(s), h); pk = np.argmax(p)
    cur = p[pk - 2 * s: pk + 8 * s: s] / p[pk]
    k = np.arange(-2, 8)
    fig, ax = plt.subplots(figsize=SIDE)
    ax.bar(k - 0.18, cur, width=0.36, color=NAVY, label="before DFE")
    after = cur.copy(); after[(k >= 1) & (k <= 4)] = 0
    ax.bar(k + 0.18, after, width=0.36, color=GREEN, label="after a 4-tap DFE")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("cursor index (UI)"); ax.set_ylabel("sampled pulse response"); ax.set_xticks(k)
    ax.legend(fontsize=6.8); ax.set_title("DFE: subtract the known tail", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_dfe_concept")


def nrz_pam4_loss():
    f = np.linspace(0, 65, 400)
    fig, ax = plt.subplots(figsize=SIDE)
    ax.plot(f, -0.9 * f, color=NAVY)
    for fn, lab, c in [(56, "NRZ Nyquist\n56 GHz: $-50$ dB", ACCENT), (28, "PAM-4 Nyquist\n28 GHz: $-25$ dB", PURPLE)]:
        ax.plot(fn, -0.9 * fn, "o", color=c, ms=6)
        ax.annotate(lab, xy=(fn, -0.9 * fn), xytext=(fn - 30 if fn > 40 else fn + 4, -0.9 * fn + 3 if fn > 40 else -0.9 * fn + 8),
                    fontsize=6.8, color=c, arrowprops=dict(arrowstyle="->", color=c, lw=0.7))
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("channel loss (dB)"); ax.set_ylim(-62, 2)
    ax.set_title("112 Gb/s over 0.9 dB/GHz", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_nrz_pam4_loss")


def lane_rates():
    """Ethernet electrical lane rates over time, with the unit interval."""
    data = [(1998, 1.25, "GbE\n8b/10b"), (2002, 10.3125, "10GbE\n64b/66b"), (2014, 25.78125, "100GbE\n4x25G"),
            (2018, 53.125, "50G PAM-4"), (2022, 106.25, "100G/lane\nPAM-4"), (2026, 212.5, "200G/lane\nPAM-4")]
    fig, ax = plt.subplots(figsize=SIDE)
    yrs = [d[0] for d in data]; rt = [d[1] for d in data]
    ax.semilogy(yrs, rt, "o-", color=NAVY, ms=4)
    for y, r_, lab in data:
        if y in (2014, 2018):
            ax.text(y + 0.8, r_ / 1.3, lab, fontsize=5.8, ha="left", va="top", color=NAVY)
        else:
            ax.text(y - 0.6, r_ * 1.25, lab, fontsize=5.8, ha="right", va="bottom", color=NAVY)
    ax.set_ylim(0.8, 600); ax.set_xlim(1992, 2029)
    ax.set_xlabel("year"); ax.set_ylabel("line rate (Gb/s)")
    ax2 = ax.secondary_yaxis("right", functions=(lambda r: 1000 / np.maximum(r, 1e-9), lambda u: 1000 / np.maximum(u, 1e-9)))
    ax2.set_ylabel("bit time (ps)", fontsize=7.5); ax2.tick_params(labelsize=6.5)
    ax.set_title("Per-lane rates, ~170x in 28 years", fontsize=8.5); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_lane_rates")


def duobinary_walk():
    """The precoded-duobinary worked example drawn out."""
    d = np.array([1, 0, 1, 1, 0, 0, 1])
    b = []; prev = 0
    for x in d: prev = x ^ prev; b.append(prev)
    b = np.array(b); a = 2 * b - 1
    y = a + np.r_[-1, a[:-1]]
    dec = (np.abs(y) < 1).astype(int)
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    rows = [("data $d_k$", d, NAVY), ("precoded $b_k$", b, PURPLE), ("level $a_k$", a, ORANGE),
            ("$y_k=a_k+a_{k-1}$", y, ACCENT), ("decided $\\hat d_k$", dec, GREEN)]
    for i, (nm, v, c) in enumerate(rows):
        yy = -i
        ax.text(-0.3, yy, nm, ha="right", va="center", fontsize=8, color=c)
        for k, x in enumerate(v):
            lbl = (("$+%d$" % x) if x > 0 else ("$%d$" % x)) if nm.startswith(("level", "$y")) else str(int(x))
            ax.text(k + 0.5, yy, lbl,
                    ha="center", va="center", fontsize=8.5, color=c,
                    bbox=dict(boxstyle="round,pad=0.25", fc=c, alpha=0.10, ec=c, lw=0.6))
    ax.text(7.3, -3, "three levels:\n$|y|<1 \\Rightarrow 1$", fontsize=7, color=ACCENT, va="center")
    ax.set_xlim(-2.6, 9.0); ax.set_ylim(-4.6, 0.6); _clean(ax)
    fig.tight_layout(); save(fig, "ch08_duobinary_walk")


def peak_detect():
    """Lorentzian read-back pulses: separable at low density, merged at high density."""
    t = np.linspace(-3, 12, 2000)
    lor = lambda x, pw: 1 / (1 + (2 * x / pw) ** 2)
    trans = [0, 1, 3, 4, 7]       # transition positions (bit cells)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for x, pw, ttl in [(ax[0], 0.6, "low density ($PW_{50}/T=0.6$): peaks are distinct"),
                       (ax[1], 2.0, "high density ($PW_{50}/T=2$): pulses merge")]:
        tot = 0
        for i, tr in enumerate(trans):
            y = (-1) ** i * lor(t - tr, pw); tot = tot + y
            x.plot(t, y, color=GRAY, lw=0.6, alpha=0.7)
        x.plot(t, tot, color=NAVY, lw=1.4)
        for tr in trans: x.axvline(tr, color=ACCENT, lw=0.5, ls=":")
        x.set_title(ttl, fontsize=8); x.set_xlabel("time (bit cells)"); x.set_xlim(-2, 10); x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_peak_detect")


def qfactor():
    r = rng(41)
    fig, ax = plt.subplots(figsize=SIDE)
    m0, m1, s0, s1 = 0.0, 1.0, 0.06, 0.09
    z = m0 + s0 * r.standard_normal(30000); o = m1 + s1 * r.standard_normal(30000)
    ax.hist(z, bins=120, density=True, orientation="horizontal", color=NAVY, alpha=0.6)
    ax.hist(o, bins=120, density=True, orientation="horizontal", color=PURPLE, alpha=0.6)
    th = (m0 * s1 + m1 * s0) / (s0 + s1)
    ax.axhline(th, color=ACCENT, lw=1)
    ax.text(3.2, th + 0.03, "best threshold", fontsize=6.8, color=ACCENT)
    ax.text(4.6, m1 + 0.2, "'1': $\\mu_1,\\ \\sigma_1$", fontsize=7, color=PURPLE, va="center")
    ax.text(4.6, m0 + 0.17, "'0': $\\mu_0,\\ \\sigma_0$", fontsize=7, color=NAVY, va="center")
    Q = (m1 - m0) / (s0 + s1)
    ax.set_title(f"$Q=(\\mu_1-\\mu_0)/(\\sigma_1+\\sigma_0)={Q:.1f}$", fontsize=8.5)
    ax.set_xlabel("histogram density"); ax.set_ylabel("sampled level"); ax.set_xlim(0, 7)
    fig.tight_layout(); save(fig, "ch08_qfactor")


def scrambler_demo():
    """Scrambler whitening an idle pattern, and a single line error tripled by the descrambler."""
    r = rng(5); n = 300
    idle = np.zeros(n, int)
    scr = scramble_ss(idle, state=r.integers(0, 2, 58))
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.3), gridspec_kw={"height_ratios": [1, 0.8]})
    ax[0].step(np.arange(120), 0.8 * idle[:120] + 1.3, where="post", color=ORANGE, lw=1.0)
    ax[0].step(np.arange(120), 0.8 * scr[:120], where="post", color=GREEN, lw=1.0)
    ax[0].text(121, 1.6, "idle (all zeros):\nno transitions", fontsize=6.8, color=ORANGE, va="center")
    ax[0].text(121, 0.4, "after the\nscrambler", fontsize=6.8, color=GREEN, va="center")
    ax[0].set_xlim(0, 145); _clean(ax[0])
    k = np.arange(0, 100)
    err = np.zeros(100); err[[20, 59, 78]] = 1
    ax[1].axhline(0, color=GRAY, lw=0.6)
    ax[1].stem([20, 59, 78], [1, 1, 1], linefmt=ACCENT, markerfmt="o", basefmt=" ")
    ax[1].text(20, 1.25, "line error at $n$", fontsize=6.8, ha="center", color=ACCENT)
    ax[1].text(59, 1.25, "$n+39$", fontsize=6.8, ha="center", color=ACCENT)
    ax[1].text(78, 1.25, "$n+58$", fontsize=6.8, ha="center", color=ACCENT)
    ax[1].set_ylim(0, 1.6); ax[1].set_yticks([]); ax[1].set_xlabel("bit index after descrambling"); ax[1].set_xlim(0, 145)
    ax[1].tick_params(labelsize=7); ax[1].grid(False)
    fig.tight_layout(); save(fig, "ch08_scrambler_demo")


def manchester_xor():
    """Manchester = NRZ data XOR the bit clock."""
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
    s = 64; n = len(bits)
    t = np.arange(n * s) / s
    d = np.repeat(bits, s)
    clk = (np.floor(t * 2) % 2 == 0).astype(int)          # high in first half of each bit
    man = d ^ clk                                          # 1 -> low-then-high (IEEE 802.3)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 2.3), sharex=True)
    for x, (y, nm, c) in zip(ax, [(d, "data (NRZ)", NAVY), (clk, "bit clock", GRAY),
                                 (man, "data XOR clock = Manchester", ACCENT)]):
        x.step(t, y, where="post", color=c, lw=1.3)
        x.set_ylim(-0.3, 1.45); x.set_yticks([]); x.text(-0.15, 0.5, nm, ha="right", va="center", fontsize=7.5, color=c)
        for k in range(n + 1): x.axvline(k, color=GRAY, lw=0.4, ls=":")
        x.grid(False)
    for k, b in enumerate(bits): ax[0].text(k + 0.5, 1.12, str(b), ha="center", fontsize=7.5, fontweight="bold", color=NAVY)
    for k in range(n): ax[2].plot(k + 0.5, 0.5, "o", ms=3.5, color=ACCENT, alpha=0.6)
    ax[2].text(n + 0.1, 0.5, "every bit has a\nmid-bit transition", fontsize=6.8, color=ACCENT, va="center")
    ax[-1].set_xlim(0, n); ax[-1].set_xlabel("time (bit periods)"); ax[-1].tick_params(labelsize=7)
    fig.tight_layout(h_pad=0.1); save(fig, "ch08_manchester_xor")


def cyclostationary():
    """Sample paths of a PAM signal and its variance, which repeats every T."""
    r = rng(29); s = 64
    t = np.arange(-3 * s, 6 * s) / s
    p = lambda x: np.where(np.abs(x) < 0.5, np.maximum(np.cos(np.pi * x), 0) ** 1.5, 0.0)   # smooth, narrower-than-T pulse
    fig, ax = plt.subplots(figsize=SIDE)
    var = np.zeros_like(t)
    for k in range(-6, 10):
        var += p(t - k) ** 2
    for i in range(5):
        a = 2.0 * r.integers(0, 2, 16) - 1
        y = sum(a[k + 6] * p(t - k) for k in range(-6, 10))
        ax.plot(t, y, color=CYCLE[i % 6], lw=0.8, alpha=0.6)
    ax.plot(t, var, color="k", lw=1.6, label=r"variance $\mathrm{E}[s^2(t)]$")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-1.3, 1.5); ax.set_xlabel("time ($t/T$)")
    ax.legend(fontsize=6.8, loc="upper right"); ax.set_title("Statistics that repeat every $T$", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_cyclostationary")


def mf_snr():
    """Output SNR of a first-order RC receive filter vs bandwidth, for a rectangular pulse, vs the matched filter."""
    BT = np.logspace(-1.3, 1.3, 300)          # 3-dB bandwidth x bit time
    # rectangular pulse of duration T through RC (time constant tau = 1/(2 pi B)); sample at t = T
    tau = 1 / (2 * np.pi * BT)
    # h(t) = (1/tau) exp(-t/tau): signal at t=T is 1-exp(-T/tau); noise var = (N0/2)/(2 tau).
    # SNR = 4 tau (1-exp(-1/tau))^2 / N0; matched filter gives 2E/N0 = 2/N0 (E = A^2 T = 1).
    rel = 10 * np.log10(2 * tau * (1 - np.exp(-1 / tau)) ** 2)
    fig, ax = plt.subplots(figsize=SIDE)
    ax.semilogx(BT, rel, color=NAVY, label="RC low-pass receive filter")
    ax.axhline(0, color=ACCENT, lw=1.2, label="matched filter (integrate-and-dump)")
    i = np.argmax(rel)
    ax.plot(BT[i], rel[i], "o", color=NAVY, ms=4)
    ax.annotate(rf"best: {rel[i]:.1f} dB at $BT\approx${BT[i]:.2f}", xy=(BT[i], rel[i]), xytext=(0.07, -6.5), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_ylim(-13, 1.5); ax.set_xlabel(r"filter bandwidth $\times$ bit time ($BT$)"); ax.set_ylabel("SNR relative to matched (dB)")
    ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("You cannot beat the matched filter", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_mf_snr")
    print("RC best", rel[i], BT[i])


def jitter_multiplier():
    ber = np.logspace(-3, -16, 300)
    fig, ax = plt.subplots(figsize=SIDE)
    from scipy.special import erfcinv
    qinv = lambda p: np.sqrt(2) * erfcinv(2 * p)
    ax.semilogx(ber, 2 * qinv(ber / 0.5), color=NAVY, label=r"$\rho_T=0.5$")
    ax.semilogx(ber, 2 * qinv(ber), color=ACCENT, ls="--", label=r"$\rho_T=1$")
    for b_ in [1e-6, 1e-12, 1e-15]:
        v = 2 * qinv(b_ / 0.5); ax.plot(b_, v, "o", color=NAVY, ms=4)
        ax.text(b_ * 1.6, v - 0.9, f"{v:.1f}", fontsize=7, color=NAVY)
    ax.invert_xaxis(); ax.set_xlabel("target BER"); ax.set_ylabel(r"RJ multiplier $2Q^{-1}(\mathrm{BER}/\rho_T)$")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("How far Gaussian jitter reaches", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_jitter_multiplier")


def eye_vs_loss():
    """Peak-distortion (worst-case) eye opening of unequalised NRZ against channel loss."""
    s = 32
    Ls = np.arange(2, 31, 1)
    eo = []
    for L in Ls:
        h, _, _ = backplane(s, float(L))
        p = np.convolve(np.ones(s), h); pk = np.argmax(p)
        cur = p[pk - 3 * s: pk + 30 * s: s]
        main = p[pk]; isi = np.sum(np.abs(cur)) - abs(main)
        eo.append((main - isi) / main)
    fig, ax = plt.subplots(figsize=SIDE)
    ax.plot(Ls, np.array(eo) * 100, color=NAVY)
    ax.axhline(0, color="k", lw=0.5)
    ax.fill_between(Ls, -100, 0, color=ACCENT, alpha=0.08)
    ax.text(21, -40, "eye closed\nfor some\npattern", fontsize=7, color=ACCENT, ha="center")
    ax.set_xlabel("loss at Nyquist (dB)"); ax.set_ylabel("worst-case eye opening (% of $h_0$)")
    ax.set_ylim(-100, 100); ax.set_title("Unequalised NRZ: peak distortion", fontsize=9)
    fig.tight_layout(); save(fig, "ch08_eye_vs_loss")


def nrzi_polarity():
    """Swap the wires: NRZ-L inverts every bit, NRZI does not care."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    n = len(bits)
    nrz = 2.0 * bits - 1
    lv = -1.0; nrzi = []
    for b in bits:
        if b: lv = -lv
        nrzi.append(lv)
    nrzi = np.array(nrzi)

    def dec_nrzi(w, start):
        prev = start; out = []
        for x in w:
            out.append(int(x != prev)); prev = x
        return np.array(out)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=True)
    cases = [(-nrz, "NRZ-L, wires swapped: every bit wrong", (-nrz > 0).astype(int), NAVY),
             (-nrzi, "NRZI, wires swapped: every bit right", dec_nrzi(-nrzi, 1.0), GREEN)]
    for x, (w, nm, dec, c) in zip(ax, cases):
        x.step(np.arange(n + 1), np.r_[w, w[-1]], where="post", color=c, lw=1.3)
        for k in range(n):
            ok = dec[k] == bits[k]
            x.text(k + 0.5, 1.35, str(dec[k]), ha="center", fontsize=7.5, color=c if ok else ACCENT,
                   fontweight="bold")
        x.set_title(nm, fontsize=8)
        x.set_ylim(-1.4, 1.75); x.set_xlim(0, n); x.set_yticks([-1, 1]); x.set_xlabel("time (bits)")
        x.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_nrzi_polarity")


def prbs_autocorr():
    """PRBS-7: two-valued periodic autocorrelation and coin-flip run lengths."""
    reg = [1] * 7; seq = []
    for _ in range(127):
        b = reg[5] ^ reg[6]            # x^7 + x^6 + 1
        seq.append(reg[6]); reg = [b] + reg[:-1]
    x = 2 * np.array(seq) - 1
    sh = np.arange(-140, 141)
    ac = np.array([np.sum(x * np.roll(x, k)) for k in sh])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].plot(sh, ac, color=NAVY, lw=1.0)
    ax[0].set_xlabel("shift (bits)"); ax[0].set_ylabel("autocorrelation")
    ax[0].set_title("PRBS-7: 127 at zero shift (and every 127 bits), $-1$ elsewhere", fontsize=8)
    ax[0].tick_params(labelsize=7)
    rl = run_lengths(np.array(seq))
    k = np.arange(1, 8)
    ax[1].bar(k, [np.sum(rl == kk) for kk in k], color=GREEN, width=0.6, label="PRBS-7 runs")
    ax[1].plot(k, len(rl) * 0.5 ** k, "o--", color=ACCENT, ms=3, lw=0.8, label="coin flips")
    ax[1].set_xlabel("run length (bits)"); ax[1].set_title("Runs, like coin flips", fontsize=8)
    ax[1].legend(fontsize=6.5); ax[1].tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_prbs")


def rlm():
    """PAM-4 level mismatch: compressed outer eye."""
    r = rng(33); s = 32
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    cases = [(np.array([-1, -1 / 3, 1 / 3, 1]), "equally spaced levels: three equal eyes"),
             (np.array([-1, -1 / 3, 1 / 3, 0.72]), "top level compressed: the upper eye shrinks")]
    for x, (lv, ttl) in zip(ax, cases):
        sym = lv[r.integers(0, 4, 1500)]
        y, pk, _ = pam_mf(sym, 0.5, s, 30, r)
        eye_plot(x, y, pk, s, PURPLE, alpha=0.05, n=350)
        x.set_title(ttl, fontsize=8); x.set_xlabel("time (symbols)"); x.tick_params(labelsize=7)
    ax[0].set_ylim(-1.4, 1.4)
    fig.tight_layout(); save(fig, "ch08_rlm")


def transponder():
    """DVB-S2 carriers filling a 36 MHz transponder at three roll-offs (worked example)."""
    f = np.linspace(-24, 24, 2000)
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    for i, b in enumerate([0.35, 0.20, 0.05]):
        Rs = 36 / (1 + b)
        a = np.abs(f) / Rs
        H = np.where(a <= (1 - b) / 2, 1.0, np.where(a <= (1 + b) / 2,
                     0.5 * (1 + np.cos(np.pi / b * (a - (1 - b) / 2))), 0))
        ax.plot(f, H * (1 - 0.12 * i), color=CYCLE[i], lw=1.4,
                label=rf"$\beta$ = {b}: {Rs:.1f} MBd")
    ax.axvspan(-18, 18, color=GRAY, alpha=0.08)
    ax.axvline(-18, color=GRAY, lw=0.8, ls="--"); ax.axvline(18, color=GRAY, lw=0.8, ls="--")
    ax.text(0, 0.12, "36 MHz transponder", ha="center", fontsize=7.5, color=GRAY)
    ax.set_xlabel("frequency offset from carrier (MHz)"); ax.set_yticks([]); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_transponder")


def mask_margin():
    """Mask margin: grow the keep-out hexagon until traces hit it."""
    r = rng(5); s = 64
    a = 2.0 * r.integers(0, 2, 1200) - 1
    y, pk, _ = pam_mf(a, 0.5, s, 18, r)
    base = np.array([[0.22, 0], [0.36, 0.45], [0.64, 0.45], [0.78, 0], [0.64, -0.45], [0.36, -0.45]])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    for x, sc, ttl in [(ax[0], 1.0, "standard mask: no hits, the eye passes"),
                       (ax[1], 1.4, "mask grown 40%: traces start to hit")]:
        tt, tr = eye_plot(x, y, pk, s, NAVY, alpha=0.05, n=400, centre=0.5)
        m = base.copy(); m[:, 0] = 0.5 + (m[:, 0] - 0.5) * sc; m[:, 1] *= sc
        x.fill(m[:, 0], m[:, 1], color=ORANGE, alpha=0.3, ec=ORANGE, lw=1.0)
        from matplotlib.path import Path
        path = Path(m)
        hits = 0
        for row in tr[::3]:
            pts = np.c_[tt, row]
            inside = path.contains_points(pts)
            if inside.any():
                hits += 1
                x.plot(tt[inside], row[inside], ".", color=ACCENT, ms=2)
        x.set_title(ttl, fontsize=8); x.set_xlabel("time (symbols)"); x.tick_params(labelsize=7)
    ax[0].set_ylim(-1.7, 1.7)
    fig.tight_layout(); save(fig, "ch08_mask_margin")


def blur_undo():
    """The duobinary analogy: a known 'shake' (1+D) blurs a word; knowing it, we undo it."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    f0 = plt.figure(figsize=(4, 1.0), dpi=100)
    f0.text(0.02, 0.25, "DUOBINARY", fontsize=40, family="DejaVu Sans", weight="bold")
    cv = FigureCanvasAgg(f0); cv.draw()
    img = np.asarray(cv.buffer_rgba())[:, :, 0].astype(float)
    plt.close(f0)
    img = 1 - img / 255.0
    sh = 14                                     # the 'shake': one copy shifted right by sh pixels
    blur = img.copy(); blur[:, sh:] += img[:, :-sh]
    rec = np.zeros_like(blur)
    for c in range(blur.shape[1]):              # undo recursively, column by column
        rec[:, c] = blur[:, c] - (rec[:, c - sh] if c >= sh else 0)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 2.4))
    for x, im, t in [(ax[0], img, "original"), (ax[1], blur, "after a known shake: each column + the column 14 pixels to its left ($1+D$)"),
                     (ax[2], rec, "undone by subtracting the known echo, left to right")]:
        x.imshow(im, cmap="Blues", aspect="auto", vmin=0, vmax=2)
        x.set_title(t, fontsize=8, loc="left"); _clean(x)
    fig.tight_layout(h_pad=0.4); save(fig, "ch08_blur_undo")


def t1_spectrum():
    """PSD of T1 AMI with 50% duty pulses (worked example), in MHz."""
    Rb = 1.544
    f = np.linspace(0.001, 4.0, 1000)
    S = np.sinc(f / (2 * Rb)) ** 2 * np.sin(np.pi * f / Rb) ** 2
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.plot(f, S / S.max(), color=NAVY, lw=1.5)
    ax.fill_between(f, S / S.max(), color=NAVY, alpha=0.12)
    for x0, lab in [(0.772, "peak 772 kHz"), (1.544, "null at $R_b$ = 1.544 MHz"), (3.088, "null at $2R_b$")]:
        ax.axvline(x0, color=ACCENT, lw=0.8, ls=":")
        ax.text(x0 + 0.04, 0.9 if x0 < 1 else 0.55, lab, fontsize=7, color=ACCENT)
    ax.annotate("no energy at DC: the pair can also\ncarry DC power to the repeaters", xy=(0.03, 0.01),
                xytext=(2.2, 0.8), fontsize=6.8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("PSD (normalised)"); ax.set_xlim(0, 4); ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch08_t1_spectrum")


def pam_ladders():
    """Same peak swing, more levels: the steps shrink."""
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    for i, (M, c) in enumerate([(2, NAVY), (4, PURPLE), (8, ORANGE), (16, GREEN)]):
        lv = np.linspace(-1, 1, M)
        for l_ in lv:
            ax.plot([i - 0.3, i + 0.3], [l_, l_], color=c, lw=1.6)
        d = 2 / (M - 1)
        ax.annotate("", xy=(i + 0.38, lv[-1]), xytext=(i + 0.38, lv[-2]),
                    arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.7))
        ax.text(i, -1.35, f"{M}-PAM: {int(np.log2(M))} bit{'s' if M > 2 else ''}/symbol\nstep = {d:.2f}"
                f" ({20*np.log10(d/2):.1f} dB)", ha="center", fontsize=7, color=c)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-1.75, 1.15); _clean(ax)
    fig.tight_layout(); save(fig, "ch08_pam_ladders")


if __name__ == "__main__":
    import sys as _s
    fns = [line_codes, linecode_psd, rds_runs, isi_sum, raised_cosine, eyes, eye_anatomy,
           timing_sensitivity, matched_filter_demo, ber_binary, bathtub, channel_loss,
           equalized_eye, pam4_eye, duobinary, prml_targets, ftn,
           timeline, cable_smear, bits_to_pam, staircase, line_code_jobs, b8zs_hdb3, psd_factor,
           nyquist_fold, rrc_vs_rc, papr, rrc_truncation, template_match, mf_sieve, decision,
           signal_geometry, mpam_ber, eye_build, jitter_types, bangbang, skin_effect, deemphasis,
           dfe_concept, nrz_pam4_loss, lane_rates, duobinary_walk, peak_detect, qfactor, scrambler_demo,
           manchester_xor, cyclostationary, mf_snr, jitter_multiplier, eye_vs_loss,
           nrzi_polarity, prbs_autocorr, rlm, transponder, mask_margin, blur_undo, t1_spectrum, pam_ladders]
    sel = _s.argv[1:]
    for f in fns:
        if not sel or f.__name__ in sel:
            f()
