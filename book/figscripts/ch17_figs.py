"""Figures for Chapter 17: OFDM and Multicarrier Transmission.

Run:  python ch17_figs.py            (all figures)
      python ch17_figs.py papr oob   (selected functions)
Uses commlib.ofdm / commlib.channel so the figures agree with Lab 7.
"""
from figstyle import *
from scipy import signal as sps
import commlib as cl
from commlib import ofdm as co

FS_LTE = 15.36e6                     # LTE/NR 10 MHz sampling rate (1024-point FFT, 15 kHz)
Q16 = cl.get_constellation("16qam")
QPSK = cl.get_constellation("qpsk")
Q64 = cl.get_constellation("64qam")


def _qam(n, c=Q16, r=None):
    return c.modulate(cl.random_bits(c.k * n, r))


def _psd(x, nfft=4096, fs=1.0):
    f, p = sps.welch(x, fs=fs, nperseg=nfft, return_onesided=False, window="hann", noverlap=nfft // 2)
    i = np.argsort(f)
    return f[i], p[i]


def _static_taps(profile, fs, r):
    """One static realisation of a TDL profile as a sample-spaced impulse response."""
    dl, pdb = cl.TDL_PROFILES[profile]
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
    d = np.round(np.asarray(dl) * 1e-9 * fs).astype(int)
    h = np.zeros(d.max() + 1, dtype=complex)
    for di, pi in zip(d, p):
        h[di] += np.sqrt(pi / 2) * (r.standard_normal() + 1j * r.standard_normal())
    return h


# ============================================================================ 1
def fdm_vs_ofdm():
    """Guard-banded FDM versus overlapping orthogonal subcarriers."""
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.15), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})
    # (a) time-domain subcarriers over one symbol
    t = np.linspace(0, 1, 800)
    for k, c in zip([1, 2, 3], [NAVY, ACCENT, GREEN]):
        ax[0].plot(t, np.cos(2 * np.pi * k * t) + 2.6 * (3 - k), color=c, lw=1.2)
        ax[0].text(1.03, 2.6 * (3 - k), f"$k={k}$", fontsize=7.5, va="center", color=c)
    ax[0].set_xlim(0, 1.18); ax[0].set_yticks([]); ax[0].set_xticks([0, 0.5, 1])
    ax[0].set_xlabel("time ($t/T$)")
    ax[0].set_title("(a) Whole cycles in $T$", fontsize=9)
    ax[0].grid(False)
    # (b) FDM with guard bands (RRC-like spectra)
    f = np.linspace(-1, 14, 3000)
    beta = 0.3
    def rc(f0, w):
        x = np.abs(f - f0) / w
        s = np.where(x <= (1 - beta) / 2, 1.0, 0.0)
        m = (x > (1 - beta) / 2) & (x <= (1 + beta) / 2)
        s = s + np.where(m, 0.5 * (1 + np.cos(np.pi / beta * (x - (1 - beta) / 2))), 0)
        return s
    for i in range(4):
        ax[1].fill_between(f, rc(1.5 + 3.4 * i, 2.0), color=CYCLE[i], alpha=0.35, lw=0)
        ax[1].plot(f, rc(1.5 + 3.4 * i, 2.0), color=CYCLE[i], lw=1)
    for i in range(3):
        g = 1.5 + 3.4 * i + 1.3
        ax[1].annotate("", xy=(g + 0.8, 0.5), xytext=(g, 0.5),
                       arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.7))
    ax[1].text(4.6, 0.28, "guard\nbands", fontsize=7, color=GRAY, ha="center")
    ax[1].set_xlim(-0.8, 13.6); ax[1].set_ylim(0, 1.25)
    ax[1].set_xlabel("frequency"); ax[1].set_yticks([]); ax[1].set_xticks([])
    ax[1].set_title("(b) Classical FDM", fontsize=9)
    # (c) OFDM: overlapping sincs
    f = np.linspace(-2, 9, 3000)
    tot = 0
    for k in range(7):
        s = np.sinc(f - k)
        ax[2].plot(f, s, color=CYCLE[k % 6], lw=1)
        tot = tot + s
    for k in range(7):
        ax[2].plot(k, 1, "o", color="k", ms=2.6)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlim(-2, 9); ax[2].set_ylim(-0.3, 1.25)
    ax[2].set_xlabel("frequency ($f\\,T$)"); ax[2].set_yticks([])
    ax[2].set_title("(c) OFDM: spacing $1/T$", fontsize=9)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_fdm_vs_ofdm")


# ============================================================================ 2
def ofdm_symbol():
    """Time-domain OFDM symbol with CP, and the PSD of an 802.11a-like signal."""
    r = rng(3)
    cfg = co.OFDMConfig(64, 52, 16)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    g = _qam(52 * 3, QPSK, r).reshape(3, 52)
    # 4x oversampled symbol for a smooth plot: zero-pad in frequency
    X = np.zeros((3, 256), dtype=complex)
    X[:, np.mod(cfg.k, 256)] = g
    x = np.fft.ifft(X, axis=1) * np.sqrt(256) * 2
    xcp = np.concatenate([x[:, -64:], x], axis=1).ravel()
    t = np.arange(len(xcp)) / 4 / 20.0            # microseconds (20 MHz sampling, x4)
    L = 320
    for s in range(3):
        a = s * L / 80.0
        ax[0].axvspan(a, a + 0.8, color=ACCENT, alpha=0.13, lw=0)
        ax[0].axvspan(a + 3.2, a + 4.0, color=GREEN, alpha=0.13, lw=0)
    ax[0].plot(t, np.abs(xcp), color=NAVY, lw=0.9)
    ax[0].annotate("", xy=(0.8, 2.75), xytext=(3.6, 2.75),
                   arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8, connectionstyle="arc3,rad=0.3"))
    ax[0].text(2.2, 3.35, "copy of the last 0.8 $\\mu$s", color=ACCENT, fontsize=7.5, ha="center")
    ax[0].set_xlim(0, 12); ax[0].set_ylim(0, 3.8)
    ax[0].set_xlabel("time ($\\mu$s)"); ax[0].set_ylabel("$|x(t)|$")
    ax[0].set_title("(a) Three 4 $\\mu$s symbols: CP (shaded) + 3.2 $\\mu$s", fontsize=9)
    # (b) PSD, long signal, 8x oversampled via zero-padded IFFT
    g = _qam(52 * 2000, Q16, r).reshape(2000, 52)
    X = np.zeros((2000, 512), dtype=complex)
    X[:, np.mod(cfg.k, 512)] = g
    x = np.fft.ifft(X, axis=1) * np.sqrt(512)
    x = np.concatenate([x[:, -128:], x], axis=1).ravel()
    f, p = _psd(x, 4096, 160.0)
    p = p / np.median(p[np.abs(f) < 7])
    ax[1].plot(f, 10 * np.log10(p + 1e-12), color=NAVY, lw=0.9)
    ax[1].axvspan(-8.125, 8.125, color=GREEN, alpha=0.08, lw=0)
    ax[1].set_xlim(-20, 20); ax[1].set_ylim(-45, 5)
    ax[1].set_xlabel("frequency (MHz)"); ax[1].set_ylabel("PSD (dB)")
    ax[1].set_title("(b) Spectrum: 52 subcarriers, 16.6 MHz", fontsize=9)
    ax[1].annotate("DC null", xy=(0.3, -8), xytext=(10, -17), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color=GRAY), color=GRAY)
    ax[1].text(-19, -40, "sinc sidelobes\nfall only as $1/f^2$", fontsize=7.2, color=GRAY)
    fig.tight_layout(w_pad=1.0); save(fig, "ch17_ofdm_symbol")


# ============================================================================ 3
def one_tap():
    r = rng(11)
    cfg = co.OFDMConfig(1024, 600, 72)
    g = _qam(600 * 4, Q16, r).reshape(4, 600)
    x = co.ofdm_modulate(g, cfg)
    h = _static_taps("EVA", FS_LTE, r)
    h = h / np.sqrt(np.sum(np.abs(h) ** 2))
    y = np.convolve(x, h)[:len(x)]
    y = cl.awgn(y, 30, r)
    Y = co.ofdm_demodulate(y, cfg)
    H = np.fft.fft(h, 1024)[cfg.active]
    Z = Y / H
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fk = cfg.k * 15e-3
    ax[0].plot(fk, 20 * np.log10(np.abs(H)), color=NAVY, lw=1)
    ax[0].set_xlabel("frequency (MHz)"); ax[0].set_ylabel("$|H_k|^2$ (dB)")
    ax[0].set_title("(a) EVA channel, 600 subcarriers", fontsize=9); ax[0].set_ylim(-30, 10)
    ax[1].scatter(Y.real, Y.imag, s=0.6, color=ACCENT, alpha=0.35, lw=0)
    ax[1].set_title("(b) $Y_k$ before equalizer", fontsize=9)
    ax[2].scatter(Z.real, Z.imag, s=0.6, color=NAVY, alpha=0.35, lw=0)
    ax[2].plot(Q16.points.real, Q16.points.imag, "+", color=ACCENT, ms=4, mew=0.8)
    ax[2].set_title("(c) $Y_k/H_k$", fontsize=9)
    for a, lim in [(ax[1], 3), (ax[2], 1.6)]:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(w_pad=0.6); save(fig, "ch17_one_tap")


# ============================================================================ 4
def cp_length():
    """EVM floor versus CP length for the LTE profiles at high SNR."""
    cps = np.arange(0, 104, 4)
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.5))
    for prof, c in zip(["EPA", "EVA", "ETU"], [GREEN, NAVY, ACCENT]):
        ev = []
        for ncp in cps:
            r = rng(5)
            cfg = co.OFDMConfig(1024, 600, int(ncp))
            e = []
            for trial in range(12):
                g = _qam(600 * 6, Q16, r).reshape(6, 600)
                x = co.ofdm_modulate(g, cfg)
                h = _static_taps(prof, FS_LTE, r); h /= np.sqrt(np.sum(np.abs(h) ** 2))
                y = np.convolve(x, h)[:len(x)]
                y = cl.awgn(y, 45, r)
                Y = co.ofdm_demodulate(y, cfg)
                H = np.fft.fft(h, 1024)[cfg.active]
                Z = Y[1:] / H
                num = np.abs(Z - g[1:]) ** 2 * np.abs(H) ** 2    # error power before division
                e.append(np.mean(num))
            ev.append(10 * np.log10(np.mean(e)))
        ax.plot(cps / FS_LTE * 1e6, ev, "o-", ms=2.5, color=c, label=prof)
        dmax = cl.TDL_PROFILES[prof][0][-1] / 1e3
        ax.axvline(dmax, color=c, ls=":", lw=0.8)
    ax.axvline(72 / FS_LTE * 1e6, color="k", lw=0.8, ls="--")
    ax.text(72 / FS_LTE * 1e6 - 0.1, -6, "LTE normal CP\n4.69 $\\mu$s", fontsize=7.5, ha="right")
    ax.set_xlabel("cyclic prefix length ($\\mu$s)")
    ax.set_ylabel("interference + noise (dB rel. signal)")
    ax.set_ylim(-48, 0); ax.legend(loc="center right")
    ax.set_title("SNR = 45 dB; dotted lines: maximum excess delay", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch17_cp_length")


# ============================================================================ 5
def timing_window():
    """EVM versus FFT-window position: the ISI-free region inside the CP."""
    r = rng(8)
    N, ncp = 256, 32
    cfg = co.OFDMConfig(N, 200, ncp)
    d = np.array([0, 3, 7, 12])          # a 13-sample channel
    pw = np.array([0, -3, -6, -10.0])
    offsets = np.arange(-12, 46)
    res = []
    for trial in range(30):
        h = np.zeros(13, complex)
        h[d] = 10 ** (pw / 20) * (r.standard_normal(4) + 1j * r.standard_normal(4)) / np.sqrt(2)
        g = _qam(200 * 6, Q16, r).reshape(6, 200)
        g[1] = np.exp(1j * np.pi / 4)                 # known training symbol (index 1)
        x = co.ofdm_modulate(g, cfg)
        y = np.convolve(x, h)[:len(x)]
        L = cfg.sym_len
        row = []
        for off in offsets:                           # off > 0: window starts early
            Ys = np.array([np.fft.fft(y[s * L + ncp - off:s * L + ncp - off + N]) / np.sqrt(N)
                           for s in range(2, 5)])[:, cfg.active]
            # ideal effective channel: FFT of h times the linear phase of the window shift
            He = np.fft.fft(h, N)[cfg.active] * np.exp(-2j * np.pi * cfg.k * off / N)
            row.append(np.mean(np.abs(Ys - He * g[2:5]) ** 2) / np.mean(np.abs(He * g[2:5]) ** 2))
        res.append(row)
    ev = 10 * np.log10(np.mean(res, axis=0) + 1e-6)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.4))
    ax.axvspan(0, ncp - 12, color=GREEN, alpha=0.12, lw=0)
    ax.plot(offsets, ev, "o-", ms=2.5, color=NAVY)
    ax.text((ncp - 12) / 2, -12, "ISI-free window\n$N_{\\mathrm{cp}}-\\nu+1$ positions", ha="center",
            fontsize=7.5, color=GREEN)
    ax.axvline(ncp, color=GRAY, ls=":", lw=0.8)
    ax.text(ncp + 0.5, -45, "start of CP", fontsize=7.5, color=GRAY)
    ax.text(-11.5, -45, "late", fontsize=7.5, color=GRAY)
    ax.set_xlabel("FFT window advance (samples before the nominal start)")
    ax.set_ylabel("interference / signal (dB)")
    ax.set_ylim(-62, 2)
    ax.set_title("$N=256$, $N_{\\mathrm{cp}}=32$, channel memory $\\nu=12$; no noise", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch17_timing_window")


# ============================================================================ 6
def ici():
    """(a) ICI leakage coefficients for CFO; (b) signal-to-ICI ratio: CFO and Doppler."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    N = 64
    l = np.arange(-6, 7)
    for eps, c, sh in [(0.05, NAVY, -0.2), (0.2, ACCENT, 0.0), (0.4, GREEN, 0.2)]:
        S = np.sin(np.pi * (l + eps)) / (N * np.sin(np.pi * (l + eps) / N))
        ax[0].plot(l, 20 * np.log10(np.abs(S)), "o-", color=c, ms=3.2, lw=0.9, label=f"$\\epsilon={eps}$")
    ax[0].set_ylim(-52, 3); ax[0].set_xlabel("offset from wanted subcarrier $l$")
    ax[0].set_ylabel("$|S_l|^2$ (dB)"); ax[0].legend(fontsize=6.5, loc="lower center", ncol=3)
    ax[0].set_title("(a) Leakage from a frequency offset", fontsize=9)
    # (b) simulation
    r = rng(2)
    cfg = co.OFDMConfig(256, 256 - 1, 0)
    cfg = co.OFDMConfig(256, 200, 0)
    eps_list = np.logspace(-2.3, -0.5, 12)
    sir_cfo, sir_dop = [], []
    g = _qam(200 * 40, Q16, r).reshape(40, 200)
    x = co.ofdm_modulate(g, cfg)
    for e in eps_list:
        Y = co.ofdm_demodulate(cl.apply_cfo(x, e / 256), cfg)
        cpe = np.exp(1j * np.angle(np.sum(Y * np.conj(g), axis=1, keepdims=True)))
        amp = np.abs(np.sum(Y * np.conj(g), axis=1, keepdims=True)) / np.sum(np.abs(g) ** 2, axis=1, keepdims=True)
        sir_cfo.append(10 * np.log10(np.mean(np.abs(g) ** 2) / np.mean(np.abs(Y - amp * cpe * g) ** 2)))
        # Doppler: flat Rayleigh fading with Jakes spectrum, fD = e * subcarrier spacing
        num = den = 0.0
        for trial in range(60):
            hfade = cl.jakes_process(len(x), e / 256, n_sin=24, rng=r)
            Yd = co.ofdm_demodulate(x * hfade, cfg)
            hbar = hfade.reshape(-1, 256).mean(axis=1, keepdims=True)
            num += np.mean(np.abs(Yd - hbar * g) ** 2)
            den += np.mean(np.abs(hfade) ** 2) * np.mean(np.abs(g) ** 2)
        sir_dop.append(10 * np.log10(den / num))
    ef = np.logspace(-2.3, -0.5, 100)
    ax[1].semilogx(ef, 10 * np.log10(np.sinc(ef) ** 2 / (1 - np.sinc(ef) ** 2)), color=NAVY, lw=1,
                   label="CFO: $\\mathrm{sinc}^2\\epsilon/(1-\\mathrm{sinc}^2\\epsilon)$")
    ax[1].semilogx(eps_list, sir_cfo, "o", color=NAVY, ms=3)
    ax[1].semilogx(ef, 10 * np.log10(6 / (np.pi * ef) ** 2), color=ACCENT, lw=1, ls="--",
                   label="Doppler: $6/(\\pi f_D T)^2$")
    ax[1].semilogx(eps_list, sir_dop, "s", color=ACCENT, ms=3)
    ax[1].set_xlabel("$\\epsilon$ or $f_DT$ (fraction of subcarrier spacing)")
    ax[1].set_ylabel("signal-to-ICI ratio (dB)")
    ax[1].legend(fontsize=6.8, loc="upper right"); ax[1].set_ylim(5, 50)
    ax[1].set_title("(b) Theory (lines) and simulation (markers)", fontsize=9)
    fig.tight_layout(w_pad=1.0); save(fig, "ch17_ici")


# ============================================================================ 7
def phase_noise():
    """Common phase error and ICI from Wiener phase noise, 64-QAM."""
    r = rng(4)
    N = 256
    cfg = co.OFDMConfig(N, 200, 16)
    g = _qam(200 * 60, Q64, r).reshape(60, 200)
    x = co.ofdm_modulate(g, cfg)
    beta = 0.004                 # linewidth / subcarrier spacing
    pn = cl.phase_noise(len(x), beta / N, rng=r)
    Y = co.ofdm_demodulate(x * pn, cfg)
    cpe = np.exp(1j * np.angle(np.sum(Y * np.conj(g), axis=1, keepdims=True)))
    Zc = Y / cpe
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.25), gridspec_kw={"width_ratios": [1, 1, 1.45]})
    for a, Z, t in [(ax[0], Y, "(a) Uncorrected"), (ax[1], Zc, "(b) CPE removed")]:
        a.scatter(Z.real, Z.imag, s=0.4, color=NAVY, alpha=0.3, lw=0)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=9)
    # (c) EVM vs linewidth
    bl = np.logspace(-4, -1.3, 10)
    ev_u, ev_c = [], []
    for b in bl:
        eu = ec = 0
        for trial in range(6):
            pn = cl.phase_noise(len(x), b / N, rng=r)
            Y = co.ofdm_demodulate(x * pn, cfg)
            cp = np.exp(1j * np.angle(np.sum(Y * np.conj(g), axis=1, keepdims=True)))
            eu += np.mean(np.abs(Y - g) ** 2); ec += np.mean(np.abs(Y / cp - g) ** 2)
        ev_u.append(10 * np.log10(eu / 6)); ev_c.append(10 * np.log10(ec / 6))
    bf = np.logspace(-4, -1.3, 100)
    ax[2].semilogx(bl, ev_u, "o-", ms=3, color=ACCENT, label="no CPE correction")
    ax[2].semilogx(bl, ev_c, "s-", ms=3, color=NAVY, label="CPE corrected")
    ax[2].semilogx(bf, 10 * np.log10(np.pi * bf / 3), color=GRAY, ls="--", lw=1,
                   label="ICI $\\approx\\pi\\beta T/3$")
    ax[2].set_xlabel("3-dB linewidth / subcarrier spacing, $\\beta T$")
    ax[2].set_ylabel("EVM (dB)"); ax[2].legend(fontsize=6.8, loc="upper left")
    ax[2].set_title("(c) Error vector magnitude", fontsize=9); ax[2].set_ylim(-45, 5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch17_phase_noise")


# ============================================================================ 8
def design_space():
    """Subcarrier spacing trade-off: CP overhead versus Doppler ICI."""
    df = np.logspace(np.log10(2e3), np.log10(1e6), 400)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.95))
    snr = 10 ** (25 / 10)
    scen = [("indoor Wi-Fi: CP 0.8 $\\mu$s, $f_D$ = 20 Hz", 0.8e-6, 20, GREEN),
            ("urban macro: CP 4.7 $\\mu$s, $f_D$ = 200 Hz", 4.7e-6, 200, NAVY),
            ("high-speed train: CP 4.7 $\\mu$s, $f_D$ = 1.5 kHz", 4.7e-6, 1500, ACCENT),
            ("SFN broadcast: CP 200 $\\mu$s, $f_D$ = 50 Hz", 200e-6, 50, ORANGE)]
    for lbl, tcp, fd, c in scen:
        T = 1 / df
        eff_cp = T / (T + tcp)
        pici = (np.pi * fd * T) ** 2 / 6
        sinr = 1 / (1 / snr + pici)
        se = eff_cp * np.log2(1 + sinr)
        ax[0].semilogx(df / 1e3, 100 * (1 - eff_cp), color=c, lw=1.1)
        ax[0].semilogx(df / 1e3, 100 * np.minimum(1, pici / (1 / snr + pici)), color=c, lw=1.1, ls="--")
        ax[1].semilogx(df / 1e3, se, color=c, lw=1.2, label=lbl)
        i = np.argmax(se)
        ax[1].plot(df[i] / 1e3, se[i], "o", color=c, ms=3.5)
    ax[0].set_ylim(0, 60); ax[0].set_xlim(2, 1000)
    ax[0].set_xlabel("subcarrier spacing (kHz)"); ax[0].set_ylabel("loss (%)")
    ax[0].set_title("(a) CP overhead (solid), ICI share (dashed)", fontsize=9)
    ax[1].set_xlim(2, 1000); ax[1].set_ylim(0, 9)
    ax[1].set_xlabel("subcarrier spacing (kHz)"); ax[1].set_ylabel("bits/s/Hz")
    ax[1].set_title("(b) Net efficiency at SNR 25 dB", fontsize=9)
    h_, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=7, frameon=False)
    for a in ax:
        for v in [15, 312.5]:
            a.axvline(v, color=GRAY, lw=0.5, ls=":")
    fig.tight_layout(w_pad=1.0, rect=(0, 0.12, 1, 1)); save(fig, "ch17_design_space")


# ============================================================================ 9
def numerology():
    """NR numerologies: slots and symbols within half a millisecond."""
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    for row, mu in enumerate(range(0, 5)):
        d = co.nr_numerology(mu)
        slot = d["slot_ms"] * 1000
        nsl = int(np.ceil(500 / slot))
        y = 4 - row
        for s in range(nsl):
            t0 = s * slot
            if t0 >= 500:
                break
            ax.add_patch(plt.Rectangle((t0, y - 0.32), min(slot, 500 - t0), 0.64,
                                       fc=NAVY if s % 2 == 0 else ACCENT, alpha=0.18 + 0.1 * (s % 2),
                                       ec=NAVY, lw=0.5))
            # symbol ticks
            if mu <= 2:
                for k in range(1, 14):
                    ax.plot([t0 + k * slot / 14] * 2, [y - 0.32, y + 0.32], color=NAVY, lw=0.25, alpha=0.6)
        ax.text(-8, y, f"$\\mu={mu}$: {d['scs_khz']:.0f} kHz", ha="right", va="center", fontsize=8)
        ax.text(508, y, f"slot {d['slot_ms'] * 1000:g} $\\mu$s, CP {d['cp_us']:.2f} $\\mu$s",
                ha="left", va="center", fontsize=7.5)
    ax.set_xlim(-120, 680); ax.set_ylim(-0.6, 4.6)
    ax.set_yticks([]); ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel("time ($\\mu$s)"); ax.grid(False)
    for sp in ["left"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("Half a subframe (0.5 ms) of each NR numerology; every slot holds 14 OFDM symbols",
                 fontsize=8.5)
    fig.tight_layout(); save(fig, "ch17_numerology")


# ============================================================================ 10
def pilots():
    """Pilot patterns on a (subcarrier x symbol) grid."""
    from matplotlib.colors import ListedColormap
    ns, nk = 14, 24
    pats = {}
    m = np.zeros((nk, ns)); m[:, 0] = 1; pats["(a) Block (preamble)"] = m
    m = np.zeros((nk, ns)); m[::4, :] = 1; pats["(b) Comb (802.11a: 4 of 52)"] = m
    m = np.zeros((nk, ns))
    for sym in [0, 4, 7, 11]:              # LTE CRS port 0, normal CP
        v = 0 if sym in (0, 7) else 3
        m[v::6, sym] = 1
    pats["(c) Scattered (LTE CRS, port 0)"] = m
    m = np.zeros((nk, ns))
    m[:, 0:2] = 2                          # CORESET / control region
    for sym in [2, 11]:                    # NR DMRS type 1, front-loaded + one additional
        m[0::2, sym] = 1
    pats["(d) NR DMRS type 1, 1 extra"] = m
    cmap = ListedColormap(["#FFFFFF", ACCENT, "#D6DEE8"])
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.55))
    for a, (t, m) in zip(ax, pats.items()):
        a.imshow(m, cmap=cmap, vmin=0, vmax=2, origin="lower", aspect="auto",
                 extent=(-0.5, ns - 0.5, -0.5, nk - 0.5))
        for k in range(nk + 1):
            a.axhline(k - 0.5, color=GRAY, lw=0.25)
        for s in range(ns + 1):
            a.axvline(s - 0.5, color=GRAY, lw=0.25)
        a.axhline(11.5, color=NAVY, lw=0.9)
        a.set_xticks([0, 6, 13]); a.set_yticks([0, 12, 23] if a is ax[0] else [])
        a.grid(False); a.set_title(t, fontsize=7.6)
        a.set_xlabel("OFDM symbol", fontsize=8)
    ax[0].set_ylabel("subcarrier", fontsize=8)
    fig.tight_layout(w_pad=0.5); save(fig, "ch17_pilots")


# ============================================================================ 11
def chest():
    """Channel estimation: LS + interpolation, DFT denoising, LMMSE."""
    N, Nu, Dp = 1024, 600, 6
    cfg = co.OFDMConfig(N, Nu, 72)
    prof = "EVA"
    dl, pdb = cl.TDL_PROFILES[prof]
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
    dsm = np.round(np.asarray(dl) * 1e-9 * FS_LTE).astype(int)
    k = cfg.k.astype(float)                    # signed subcarrier index (with DC gap)
    pidx = np.arange(0, Nu, Dp)                # pilot positions (index into used carriers)
    kp = k[pidx]
    # frequency correlation from the PDP (the LMMSE prior)
    def Rf(ka, kb):
        return (p[None, None, :] * np.exp(-2j * np.pi * (ka[:, None, None] - kb[None, :, None]) * dsm[None, None, :] / N)).sum(-1)
    Rpp = Rf(kp, kp); Rap = Rf(k, kp)
    Ltaps = 72

    def estimates(Hls_p, n0):
        # linear interpolation
        Hlin = np.interp(k, kp, Hls_p.real) + 1j * np.interp(k, kp, Hls_p.imag)
        # DFT-based (delay-domain) estimate: fit taps on a delay grid of resolution
        # N/(Np*Dp) samples, keep only those inside the CP, re-evaluate on all carriers
        Np = len(kp)
        res = N / (Np * Dp)
        m = np.arange(int(np.ceil(Ltaps / res))) * res
        Fp = np.exp(-2j * np.pi * np.outer(kp, m) / N)
        hk = np.linalg.lstsq(Fp, Hls_p, rcond=None)[0]
        Hdft = np.exp(-2j * np.pi * np.outer(k, m) / N) @ hk
        # LMMSE
        W = Rap @ np.linalg.inv(Rpp + n0 * np.eye(Np))
        Hmm = W @ Hls_p
        return Hlin, Hdft, Hmm

    r = rng(21)
    snrs = np.arange(0, 41, 5)
    mse = {n: [] for n in ["LS at pilots", "LS + linear interp.", "DFT-based", "LMMSE"]}
    for snr in snrs:
        n0 = 10 ** (-snr / 10)
        acc = np.zeros(4); cnt = 0
        for trial in range(40):
            gains = np.sqrt(p / 2) * (r.standard_normal(len(p)) + 1j * r.standard_normal(len(p)))
            H = (gains[None, :] * np.exp(-2j * np.pi * k[:, None] * dsm[None, :] / N)).sum(1)
            noise = np.sqrt(n0 / 2) * (r.standard_normal(Nu) + 1j * r.standard_normal(Nu))
            Hls = H + noise                       # unit-modulus pilots => LS = H + noise
            Hls_p = Hls[pidx]
            Hlin, Hdft, Hmm = estimates(Hls_p, n0)
            acc += [np.mean(np.abs(Hls_p - H[pidx]) ** 2), np.mean(np.abs(Hlin - H) ** 2),
                    np.mean(np.abs(Hdft - H) ** 2), np.mean(np.abs(Hmm - H) ** 2)]
            cnt += 1
        for i, n in enumerate(mse):
            mse[n].append(10 * np.log10(acc[i] / cnt))
        if snr == 15:
            ex = (H, Hls_p, Hlin, Hdft, Hmm)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    H, Hls_p, Hlin, Hdft, Hmm = ex
    sel = slice(0, 240)
    ax[0].plot(k[sel], 20 * np.log10(np.abs(H[sel])), color="k", lw=1.6, label="true $|H_k|$")
    msk = pidx < 240
    ax[0].plot(kp[msk], 20 * np.log10(np.abs(Hls_p[msk])), "o", ms=3, color=ORANGE, label="LS at pilots")
    ax[0].plot(k[sel], 20 * np.log10(np.abs(Hlin[sel])), color=ACCENT, lw=0.8, label="linear interp.")
    ax[0].plot(k[sel], 20 * np.log10(np.abs(Hmm[sel])), color=NAVY, lw=1.0, ls="--", label="LMMSE")
    ax[0].set_xlabel("subcarrier index"); ax[0].set_ylabel("dB")
    ax[0].set_title("(a) EVA, pilots every 6th subcarrier, SNR 15 dB", fontsize=8.5)
    ax[0].legend(fontsize=6.5, loc="lower left", ncol=2); ax[0].set_ylim(-25, 10)
    sty = {"LS at pilots": (ORANGE, "o"), "LS + linear interp.": (ACCENT, "^"),
           "DFT-based": (GREEN, "s"), "LMMSE": (NAVY, "d")}
    for n, v in mse.items():
        ax[1].plot(snrs, v, marker=sty[n][1], color=sty[n][0], ms=3, lw=1, label=n)
    ax[1].set_xlabel("pilot SNR (dB)"); ax[1].set_ylabel("MSE (dB)")
    ax[1].set_title("(b) Estimation error", fontsize=9)
    ax[1].legend(fontsize=6.5); ax[1].set_ylim(-50, 5)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_chest")
    print("chest MSE:", {n: np.round(v, 1).tolist() for n, v in mse.items()})


# ============================================================================ 12
def papr():
    """Envelope peaks and the PAPR CCDF of CP-OFDM, DFT-s-OFDM and single carrier."""
    r = rng(9)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) envelope of a few symbols, 4x oversampled
    cfg = co.OFDMConfig(4 * 256, 200, 0)
    g = _qam(200 * 3, QPSK, r).reshape(3, 200)
    x = co.ofdm_modulate(g, cfg)
    pw = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
    t = np.arange(len(x)) / (4 * 256)
    ax[0].plot(t, 10 * np.log10(pw + 1e-6), color=NAVY, lw=0.6)
    ax[0].axhline(0, color=GRAY, ls="--", lw=0.8)
    for s in range(3):
        seg = pw[s * 1024:(s + 1) * 1024]
        i = np.argmax(seg)
        ax[0].plot(t[s * 1024 + i], 10 * np.log10(seg[i]), "v", color=ACCENT, ms=4)
        ax[0].text(t[s * 1024 + i], 10 * np.log10(seg[i]) + 1.0, f"{10 * np.log10(seg[i]):.1f} dB",
                   fontsize=7, ha="center", color=ACCENT)
    ax[0].set_ylim(-20, 14); ax[0].set_xlabel("time (OFDM symbols)"); ax[0].set_ylabel("$|x|^2$ / mean (dB)")
    ax[0].set_title("(a) Instantaneous power, 200 QPSK subcarriers", fontsize=8.5)
    # (b) CCDF
    grid = np.linspace(0, 13, 131)
    for nu, nf, c in [(52, 64, GREEN), (300, 512, NAVY), (1200, 2048, PURPLE)]:
        cfg = co.OFDMConfig(4 * nf, nu, 0)
        nsym = min(int(4e5 // nu), 4000) if nu < 1000 else 400
        x = co.ofdm_modulate(_qam(nu * nsym, QPSK, r).reshape(nsym, nu), cfg)
        gg, cc = co.ccdf(co.papr_db(x, 4 * nf), grid)
        ax[1].semilogy(gg, np.where(cc > 0, cc, np.nan), color=c, lw=1.3, label=f"CP-OFDM, $N$ = {nu}")
        th = 1 - (1 - np.exp(-10 ** (grid / 10))) ** (2.8 * nu)
        ax[1].semilogy(grid, th, color=c, lw=0.8, ls=":")
    # DFT-s-OFDM with 300 subcarriers, QPSK and 16-QAM
    cfg = co.OFDMConfig(4 * 512, 300, 0)
    for const, lbl, c in [(QPSK, "DFT-s-OFDM, QPSK", ACCENT), (Q16, "DFT-s-OFDM, 16-QAM", ORANGE)]:
        s_ = _qam(300 * 1500, const, r)
        x = co.dft_s_ofdm_modulate(s_, cfg)
        gg, cc = co.ccdf(co.papr_db(x, 4 * 512), grid)
        ax[1].semilogy(gg, np.where(cc > 0, cc, np.nan), color=c, lw=1.3, label=lbl)
    s_ = _qam(300 * 800, QPSK, r)
    xs = cl.shape(s_, cl.rrc_taps(0.22, 4, 10), 4)[400:-400]
    gg, cc = co.ccdf(co.papr_db(xs, 1200), grid)
    ax[1].semilogy(gg, np.where(cc > 0, cc, np.nan), color=GRAY, lw=1.3, label="single carrier, RRC 0.22")
    ax[1].set_ylim(1e-3, 1); ax[1].set_xlim(0, 13)
    ax[1].set_xlabel("PAPR$_0$ (dB)"); ax[1].set_ylabel("Pr(PAPR > PAPR$_0$)")
    ax[1].legend(fontsize=6.3, loc="lower left")
    ax[1].set_title("(b) CCDF, 4$\\times$ oversampled (dotted: approx.)", fontsize=8.5)
    ax[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_papr")


def _clip(x, A):
    a = np.abs(x)
    return np.where(a > A, x * A / np.maximum(a, 1e-12), x)


# ============================================================================ 13
def clipping():
    """Iterative clipping and filtering (Armstrong)."""
    r = rng(12)
    nf, nu, L = 256, 200, 4
    cfg = co.OFDMConfig(L * nf, nu, 0)
    nsym = 1500
    g = _qam(nu * nsym, Q16, r).reshape(nsym, nu)
    x = co.ofdm_modulate(g, cfg).reshape(nsym, -1)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    CR = 10 ** (4.0 / 20)                 # clipping level 4 dB above RMS

    def clip_filter(x, iters):
        y = x.copy()
        mask = np.zeros(L * nf, bool); mask[cfg.active] = True
        for _ in range(iters):
            y = _clip(y, CR * rms)
            Y = np.fft.fft(y, axis=1)
            Y[:, ~mask] = 0
            y = np.fft.ifft(Y, axis=1)
        return y
    variants = [("original", x, NAVY), ("clipped at 4 dB, unfiltered", _clip(x, CR * rms), ACCENT),
                ("clip + filter, 1 iteration", clip_filter(x, 1), ORANGE),
                ("clip + filter, 4 iterations", clip_filter(x, 4), GREEN)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    grid = np.linspace(0, 12, 121)
    for lbl, y, c in variants:
        gg, cc = co.ccdf(co.papr_db(y.ravel(), L * nf), grid)
        Y = np.fft.fft(y, axis=1)[:, cfg.active] / np.sqrt(L * nf)
        a = np.sum(Y * np.conj(g)) / np.sum(np.abs(g) ** 2)     # remove Bussgang gain
        evm = 10 * np.log10(np.mean(np.abs(Y / a - g) ** 2) + 1e-12)
        lab = lbl if lbl == "original" else f"{lbl} (EVM {evm:.0f} dB)"
        ax[0].semilogy(gg, np.where(cc > 0, cc, np.nan), color=c, lw=1.2, label=lab)
        P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(y, axis=1)) ** 2, axis=0))
        fb = np.arange(-L * nf // 2, L * nf // 2)
        ax[1].plot(fb, 10 * np.log10(P / np.median(P[np.abs(fb) < 60]) + 1e-12), color=c, lw=0.9,
                   ls="--" if "4 it" in lbl else "-")
        print("clipping", lbl, "EVM", round(evm, 1))
    ax[0].set_ylim(1e-3, 1); ax[0].set_xlabel("PAPR$_0$ (dB)"); ax[0].set_ylabel("Pr(PAPR > PAPR$_0$)")
    ax[0].legend(fontsize=6.2, loc="lower left"); ax[0].grid(True, which="both", alpha=0.25)
    ax[0].set_title("(a) CCDF, 200 subcarriers, 16-QAM", fontsize=8.5)
    ax[1].set_xlim(-400, 400); ax[1].set_ylim(-70, 5)
    ax[1].set_xlabel("frequency (subcarrier spacings)"); ax[1].set_ylabel("PSD (dB)")
    ax[1].set_title("(b) Per-symbol spectrum: filtering removes splatter", fontsize=8.5)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_clipping")


# ============================================================================ 14
def oob():
    """Out-of-band emission: CP-OFDM, WOLA and filtered OFDM (LTE 10 MHz-like)."""
    r = rng(13)
    N, nu, ncp, L = 1024, 600, 72, 4
    fs = FS_LTE * L
    nsym = 600
    g = _qam(nu * nsym, Q16, r).reshape(nsym, nu)
    cfg = co.OFDMConfig(N * L, nu, 0)
    xs = co.ofdm_modulate(g, cfg).reshape(nsym, N * L)      # no CP yet, oversampled
    Ncp = ncp * L
    Ls = N * L + Ncp
    x_cp = np.concatenate([xs[:, -Ncp:], xs], axis=1).ravel()
    # WOLA: extend by W cyclic samples at the end; RC ramps of W samples at both ends
    W = 24 * L
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(W) + 0.5) / W))
    win = np.concatenate([ramp, np.ones(Ls - W), ramp[::-1]])
    ext = np.concatenate([xs[:, -Ncp:], xs, xs[:, :W]], axis=1) * win
    x_w = np.zeros(nsym * Ls + W, complex)
    for s in range(nsym):
        x_w[s * Ls:s * Ls + Ls + W] += ext[s]
    # filtered OFDM: windowed-sinc low-pass, passband = allocation + 2 subcarriers
    ntap = 512 * L + 1
    n = np.arange(ntap) - ntap // 2
    bw = (nu / 2 + 2) * 15e3
    h = 2 * bw / fs * np.sinc(2 * bw / fs * n) * np.hanning(ntap) ** 0.6
    h /= h.sum()
    x_f = np.convolve(x_cp, h)[ntap // 2:ntap // 2 + len(x_cp)]
    fig, ax = plt.subplots(figsize=(W1 * 0.9, 2.6))
    for lbl, y, c in [("CP-OFDM (rectangular)", x_cp, ACCENT), ("WOLA, 1.6 $\\mu$s RC taper", x_w, ORANGE),
                      ("filtered OFDM, 33 $\\mu$s filter", x_f, NAVY)]:
        f, p = _psd(y, 8192, fs)
        p = p / np.median(p[np.abs(f) < 3e6])
        ax.plot(f / 1e6, 10 * np.log10(p + 1e-14), color=c, lw=0.9, label=lbl)
    ax.axvspan(-4.5, 4.5, color=GREEN, alpha=0.07, lw=0)
    for v in (-5, 5):
        ax.axvline(v, color=GRAY, ls=":", lw=0.8)
    ax.text(5.1, -12, "channel\nedge", fontsize=7, color=GRAY)
    ax.set_xlim(-10, 10); ax.set_ylim(-100, 5)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("PSD (dB rel. in-band)")
    ax.legend(fontsize=7, loc="lower center")
    ax.set_title("600 subcarriers of 15 kHz (9 MHz) in a 10 MHz channel", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch17_oob")


# ============================================================================ 15
def coded():
    """Coded OFDM: frequency diversity through coding and bit interleaving.
    Simulated per subcarrier (Y_k = H_k X_k + W_k), i.e. assuming an adequate CP."""
    r = rng(15)
    code = cl.ConvCode()
    nu = 600
    ninfo = nu - (code.K - 1)                  # QPSK: 2 coded bits/subcarrier, rate 1/2
    N = 1024
    k = co.OFDMConfig(N, nu, 72).k
    perm = r.permutation(2 * nu)

    def chan(profile, B):
        if profile == "AWGN":
            return np.ones((B, nu), complex)
        dl, pdb = cl.TDL_PROFILES[profile]
        p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
        d = np.round(np.asarray(dl) * 1e-9 * FS_LTE).astype(int)
        gns = np.sqrt(p / 2) * (r.standard_normal((B, len(p))) + 1j * r.standard_normal((B, len(p))))
        return gns @ np.exp(-2j * np.pi * np.outer(d, k) / N)

    def run(profile, ebn0, coded_, inter, B=200, nblk=3):
        errs = tot = 0
        esn0 = ebn0 + 10 * np.log10(2 * (ninfo / (2 * nu) if coded_ else 1))
        n0 = 10 ** (-esn0 / 10)
        for _ in range(nblk):
            H = chan(profile, B)
            if coded_:
                u = r.integers(0, 2, (B, ninfo))
                c = code.encode_batch(u)
                ci = c[:, perm] if inter else c
            else:
                ci = r.integers(0, 2, (B, 2 * nu))
            s = QPSK.modulate(ci.ravel()).reshape(B, nu)
            y = H * s + np.sqrt(n0 / 2) * (r.standard_normal((B, nu)) + 1j * r.standard_normal((B, nu)))
            llr = QPSK.llr(y.ravel(), n0, h=H.ravel()).reshape(B, 2 * nu)
            if coded_:
                if inter:
                    l2 = np.empty_like(llr); l2[:, perm] = llr; llr = l2
                uh = code.decode_batch(llr)
                errs += np.sum(uh != u); tot += u.size
            else:
                errs += np.sum((llr < 0) != ci); tot += ci.size
        return errs / tot
    eb = np.arange(0, 21, 2)
    curves = [("uncoded, AWGN", "AWGN", False, False, GRAY, "-"),
              ("uncoded, ETU", "ETU", False, False, GRAY, "--"),
              ("coded, AWGN", "AWGN", True, False, "k", "-"),
              ("coded, ETU, no interleaver", "ETU", True, False, ORANGE, "-"),
              ("coded, ETU, interleaved", "ETU", True, True, NAVY, "-"),
              ("coded, EPA, interleaved", "EPA", True, True, ACCENT, "-")]
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.9))
    for lbl, prof, cd, it, c, ls in curves:
        ber = []
        for e in eb:
            b = run(prof, e, cd, it)
            ber.append(b)
            if b < 2e-6:
                break
        ber = np.array(ber, float)
        print("coded", lbl, np.round(ber, 6).tolist())
        ax.semilogy(eb[:len(ber)], np.where(ber > 0, ber, np.nan), marker="o", ms=2.5, color=c, ls=ls,
                    lw=1.1, label=lbl)
    ax.set_ylim(1e-5, 0.5); ax.set_xlim(0, 20)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.6, loc="lower left")
    ax.set_title("QPSK on 600 subcarriers, $K=7$ rate-1/2 code, one OFDM symbol per codeword", fontsize=8)
    fig.tight_layout(); save(fig, "ch17_coded")


# ============================================================================ 16
def waterfill():
    """DMT bit loading on a DSL loop and the water-filling picture."""
    df = 4312.5
    tones = np.arange(33, 512)                 # ADSL2+ downstream tones
    f = tones * df
    tx = -40.0                                 # dBm/Hz transmit PSD (flat mask)

    def snr_db(Lkm):
        att = 23.0 * Lkm * np.sqrt(f / 1e6) + 2.0 * Lkm
        noise = 10 * np.log10(10 ** (-140 / 10) + 10 ** ((-128 + 15 * np.log10(f / 1e6)) / 10))
        rfi = 30 * np.exp(-0.5 * ((f - 1.35e6) / 8e3) ** 2)       # AM broadcast ingress
        return tx - att - (noise + rfi)
    gap = 9.8 + 6.0 - 4.0                      # SNR gap + margin - coding gain (dB)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.55))
    s = 10 ** ((snr_db(3.0) - gap) / 10)
    inv = 1 / s
    P = len(tones) * 1.0
    lo, hi = 0, inv.max() + P
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if np.sum(np.maximum(mu - inv, 0)) > P:
            hi = mu
        else:
            lo = mu
    pa = np.maximum(mu - inv, 0)
    ax[0].fill_between(f / 1e6, inv, np.maximum(inv, mu), color="#5DADE2", alpha=0.5, lw=0, step="mid")
    ax[0].plot(f / 1e6, inv, color=NAVY, lw=1)
    ax[0].axhline(mu, color="#2E86C1", lw=1, ls="--")
    ax[0].text(0.15, mu * 1.5, "water level", fontsize=7.5, color="#2E86C1")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-6, 1e2)
    ax[0].set_xlabel("frequency (MHz)"); ax[0].set_ylabel("$\\Gamma/g_k$ (log scale)")
    ax[0].set_title("(a) Water-filling, 3 km loop", fontsize=9)
    rate_wf = 4000 * np.sum(np.log2(1 + pa * s))
    rate_flat = 4000 * np.sum(np.log2(1 + s))
    ax[0].text(0.9, 3e-6, f"water-filling: {rate_wf / 1e6:.2f} Mb/s\nflat PSD: {rate_flat / 1e6:.2f} Mb/s",
               fontsize=7, ha="left")
    for Lkm, c in [(1.0, GREEN), (3.0, NAVY), (5.0, ACCENT)]:
        b = np.clip(np.floor(np.log2(1 + 10 ** ((snr_db(Lkm) - gap) / 10))), 0, 15)
        b[b == 1] = 0                          # 1-bit tones are not loaded
        rate = 4000 * b.sum()
        ax[1].step(f / 1e6, b, where="mid", color=c, lw=1, label=f"{Lkm:.0f} km: {rate / 1e6:.1f} Mb/s")
        print("loop", Lkm, "km rate", rate / 1e6)
    ax[1].set_xlabel("frequency (MHz)"); ax[1].set_ylabel("bits per tone")
    ax[1].set_ylim(0, 16); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Bit loading, ADSL2+ downstream tones", fontsize=9)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_waterfill")
    print("waterfill rates (Mb/s):", rate_wf / 1e6, rate_flat / 1e6)


if __name__ == "__main__":
    import sys as _s
    fns = [fdm_vs_ofdm, ofdm_symbol, one_tap, cp_length, timing_window, ici, phase_noise,
           design_space, numerology, pilots, chest, papr, clipping, oob, coded, waterfill]
    sel = _s.argv[1:]
    for fn in fns:
        if not sel or fn.__name__ in sel:
            fn()
