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
        x = co.dft_s_ofdm_modulate(s_, cfg, contiguous=True)
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


# ============================================================================
# Second-edition concept illustrations and extra data figures
# ============================================================================
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon

NARROW = (3.0, 2.4)


def _clean_ax(a):
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)


def highway():
    """Analogy: one fast lane (single carrier) versus many slow lanes (OFDM)."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.45), gridspec_kw={"width_ratios": [1, 1.12]})
    # (a) single carrier: 24 short symbols; an echo delayed by 5 symbols overlaps them
    a = ax[0]
    for i in range(24):
        a.add_patch(Rectangle((i * 0.5, 1.6), 0.46, 0.7, color=NAVY, alpha=0.85, lw=0))
        a.add_patch(Rectangle((i * 0.5 + 2.6, 0.55), 0.46, 0.7, color=ACCENT, alpha=0.35, lw=0))
    a.text(0, 2.5, "direct path: 24 short symbols", fontsize=7.5, color=NAVY)
    a.text(2.6, 0.12, "echo, 5 symbols late", fontsize=7.5, color=ACCENT)
    a.annotate("", xy=(2.6, 1.42), xytext=(0, 1.42),
               arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    a.text(1.3, 1.36, "$\\tau$", fontsize=8, ha="center", color=GRAY, va="top")
    a.text(6.1, -0.55, "every symbol lands on 5 others:\nthe whole road is bumpy", fontsize=7.5,
           ha="center", color="k")
    a.set_xlim(-0.2, 12.2); a.set_ylim(-1.1, 2.9); _clean_ax(a)
    a.set_title("(a) One fast lane (single carrier)", fontsize=9)
    # (b) OFDM: 12 slow lanes, long symbols with a prefix; a notch hurts two lanes
    b = ax[1]
    nl = 12
    gains = np.array([1.0, 0.95, 0.85, 0.7, 0.45, 0.12, 0.08, 0.4, 0.75, 0.9, 1.0, 0.95])
    for k in range(nl):
        y = k * 0.24
        bad = gains[k] < 0.3
        for s in range(2):
            x0 = s * 5.4
            b.add_patch(Rectangle((x0, y), 0.9, 0.19, color=GREEN, alpha=0.5, lw=0))
            b.add_patch(Rectangle((x0 + 0.9, y), 4.4, 0.19, color=ACCENT if bad else NAVY,
                                  alpha=0.85 if bad else 0.7, lw=0))
    # channel |H| drawn at the right as a vertical curve
    yy = np.linspace(0, nl * 0.24, 200)
    gg = np.interp(yy, np.arange(nl) * 0.24 + 0.1, gains)
    b.plot(11.0 + 1.3 * gg, yy, color=ORANGE, lw=1.5)
    b.text(11.6, nl * 0.24 + 0.05, "$|H(f)|$", fontsize=7.5, color=ORANGE, ha="center")
    b.annotate("pothole (fade):\ntwo lanes slow", xy=(11.15, 5.5 * 0.24 + 0.1), xytext=(6.2, -0.75),
               fontsize=7.2, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    b.text(0.45, -0.35, "CP", fontsize=7, color=GREEN, ha="center")
    b.annotate("", xy=(13.2, nl * 0.24), xytext=(13.2, 0), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
    b.text(13.45, nl * 0.12, "frequency", rotation=90, fontsize=7, color=GRAY, va="center")
    b.set_xlim(-0.2, 13.8); b.set_ylim(-1.1, nl * 0.24 + 0.4); _clean_ax(b)
    b.set_title("(b) Many slow lanes (OFDM)", fontsize=9)
    fig.tight_layout(w_pad=0.6); save(fig, "ch17_highway")


def orth_products():
    """Orthogonality as whole cycles: the product of two subcarriers integrates to zero."""
    t = np.linspace(0, 1, 1000)
    s2, s3 = np.cos(2 * np.pi * 2 * t), np.cos(2 * np.pi * 3 * t)
    fig, ax = plt.subplots(2, 1, figsize=(3.0, 2.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    ax[0].plot(t, s2, color=NAVY, lw=1.2, label="subcarrier 2")
    ax[0].plot(t, s3, color=ACCENT, lw=1.2, label="subcarrier 3")
    ax[0].set_yticks([]); ax[0].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.32),
                                       frameon=False)
    p = s2 * s3
    ax[1].fill_between(t, p, 0, where=p > 0, color=GREEN, alpha=0.45, lw=0)
    ax[1].fill_between(t, p, 0, where=p < 0, color=ACCENT, alpha=0.35, lw=0)
    ax[1].plot(t, p, color="k", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].text(0.5, 1.12, f"product: average = {abs(np.trapezoid(p, t)):.3f}", ha="center", fontsize=7.5)
    ax[1].set_ylim(-1.15, 1.35); ax[1].set_yticks([])
    ax[1].set_xlabel("time ($t/T$)")
    fig.tight_layout(h_pad=0.3); save(fig, "ch17_orth_products")


def fft_bank():
    """The DFT as a bank of tuned forks: each bin's response is zero at every other bin."""
    N = 8
    f = np.linspace(-0.45, N - 0.55, 1500)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for k in range(N):
        d = f - k
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(np.abs(d) < 1e-9, 1.0, np.abs(np.sin(np.pi * d) / (N * np.sin(np.pi * d / N))))
        ax.plot(f, g, color=CYCLE[k % 6], lw=1.0, alpha=0.9)
    ax.axvline(3, color="k", lw=0.6, ls=":")
    ax.plot(np.arange(N), [0] * 3 + [1] + [0] * 4, "o", color="k", ms=3.2)
    ax.text(3.15, 1.04, "a tone at bin 3\nrings fork 3 only", fontsize=7, va="bottom")
    ax.set_xlabel("frequency (bins, $\\Delta f$)"); ax.set_ylabel("response of each bin")
    ax.set_ylim(0, 1.3); ax.set_xticks(range(N))
    fig.tight_layout(); save(fig, "ch17_fft_bank")


def symbol_build():
    """Four subcarriers with random QPSK phases add up to a noise-like OFDM waveform."""
    r = rng(4)
    t = np.linspace(0, 1, 800)
    ph = r.choice([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4], 16)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    tot = np.zeros_like(t)
    for k in range(1, 17):
        tot += np.cos(2 * np.pi * k * t + ph[k - 1])
    for i, k in enumerate([1, 2, 3, 4]):
        ax.plot(t, 0.42 * np.cos(2 * np.pi * k * t + ph[k - 1]) + 4.2 - 0.95 * i, color=CYCLE[i], lw=1.0)
        ax.text(1.02, 4.2 - 0.95 * i, f"$k={k}$", fontsize=7, va="center", color=CYCLE[i])
    ax.text(1.02, 0.55, "$\\vdots$", fontsize=8, va="center")
    ax.plot(t, 0.18 * tot - 0.7, color="k", lw=0.9)
    ax.text(1.02, -0.7, "sum of\n16", fontsize=7, va="center")
    ax.set_xlim(0, 1.17); _clean_ax(ax)
    ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("time ($t/T$)")
    ax.spines["bottom"].set_visible(True)
    fig.tight_layout(); save(fig, "ch17_symbol_build")


def fft_cost():
    """Complex multiplies per OFDM symbol: direct DFT versus FFT."""
    N = 2 ** np.arange(4, 16)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.loglog(N, N.astype(float) ** 2, "o-", ms=2.5, color=ACCENT, label="direct DFT, $N^2$")
    ax.loglog(N, N / 2 * np.log2(N), "s-", ms=2.5, color=NAVY, label="FFT, $\\frac{N}{2}\\log_2N$")
    for n, lab in [(64, "802.11a"), (2048, "LTE"), (32768, "DVB-T2")]:
        ax.annotate(lab, xy=(n, n / 2 * np.log2(n)), xytext=(n, n / 2 * np.log2(n) / 25), fontsize=7,
                    ha="center", arrowprops=dict(arrowstyle="->", lw=0.5, color=GRAY))
    r = 32768 ** 2 / (32768 / 2 * 15)
    ax.text(30000, 3, f"at $N=32768$: {r:.0f}$\\times$ fewer", fontsize=7.5, ha="right")
    ax.set_xlabel("FFT size $N$"); ax.set_ylabel("multiplies per symbol")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(1, 3e10)
    fig.tight_layout(); save(fig, "ch17_fft_cost")


def timeline():
    """Milestones of multicarrier transmission."""
    ev = [(1958, "Kineplex HF\nmodem (late 1950s)", 1), (1966, "Chang: orthogonal\nsubchannels", -1),
          (1971, "Weinstein & Ebert:\nthe DFT", 1), (1980, "Peled & Ruiz:\ncyclic prefix", -1),
          (1985, "Cimini: OFDM\nfor mobile radio", 1), (1990, "Bingham: 'an idea\nwhose time has come'", -1),
          (1993, "ADSL Olympics:\nDMT wins", 1), (1995, "DAB on air", -1), (1998, "DVB-T\nservices", 1),
          (1999, "802.11a", -1), (2005, "3GPP picks\nOFDMA / SC-FDMA", 1), (2009, "first LTE\nnetworks", -1),
          (2013, "DOCSIS 3.1", 1), (2019, "5G NR,\nWi-Fi 6", -1), (2024, "Wi-Fi 7", 1)]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.axhline(0, color=NAVY, lw=2)
    for yr, lab, s in ev:
        c = ACCENT if yr < 1993 else NAVY
        h = s * (0.55 if (ev.index((yr, lab, s)) // 2) % 2 == 0 else 1.15)
        ax.plot([yr, yr], [0, h * 0.82], color=c, lw=0.7)
        ax.plot(yr, 0, "o", color=c, ms=4)
        ax.text(yr, h, lab, ha="center", va="bottom" if s > 0 else "top", fontsize=6.6, color=c)
    for y in range(1960, 2030, 10):
        ax.text(y, -0.12 if y not in (1990,) else -0.12, str(y), ha="center", va="top", fontsize=7, color=GRAY)
    ax.set_xlim(1953, 2029); ax.set_ylim(-1.75, 1.75); _clean_ax(ax)
    fig.tight_layout(); save(fig, "ch17_timeline")


def circulant():
    """Circulant channel matrices are diagonalised by the DFT; Toeplitz ones are not."""
    N = 16
    h = np.array([1.0, 0.6 - 0.3j, 0.35j, -0.2])
    Hc = np.zeros((N, N), complex)
    for n in range(N):
        for m, hm in enumerate(h):
            Hc[n, (n - m) % N] += hm
    T = np.tril(Hc) * (np.abs(np.subtract.outer(np.arange(N), np.arange(N))) < len(h))  # no wrap-around
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    D1 = F @ Hc @ F.conj().T
    D2 = F @ T @ F.conj().T
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    ttl = ["(a) channel with CP: circulant", "(b) $\\mathbf{F}\\mathbf{H}_c\\mathbf{F}^H$: diagonal",
           "(c) no CP: leakage (ICI)"]
    for a, M, tl in zip(ax, [Hc, D1, D2], ttl):
        im = a.imshow(20 * np.log10(np.abs(M) + 1e-6), cmap="Blues", vmin=-40, vmax=6)
        a.set_title(tl, fontsize=8.5); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    off = np.sum(np.abs(D2 - np.diag(np.diag(D2))) ** 2) / np.sum(np.abs(np.diag(D2)) ** 2)
    ax[2].set_xlabel(f"off-diagonal energy {10 * np.log10(off):.0f} dB", fontsize=8)
    off1 = np.sum(np.abs(D1 - np.diag(np.diag(D1))) ** 2)
    ax[1].set_xlabel("off-diagonal energy: zero", fontsize=8)
    ax[0].set_xlabel("each row = previous, shifted", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("dB", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "ch17_circulant")


def papr_crowd():
    """Distribution of instantaneous power: mostly near the mean, rarely huge."""
    r = rng(21)
    N, L, ns = 256, 4, 4000
    X = np.zeros((ns, N * L), complex)
    idx = np.r_[1:101, N * L - 100:N * L]
    X[:, idx] = _qam(200 * ns, QPSK, r).reshape(ns, 200)
    x = np.fft.ifft(X, axis=1).ravel()
    p = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    g = np.linspace(0, 13, 200)
    cc = np.array([np.mean(p > 10 ** (gi / 10)) for gi in g])
    ax.semilogy(g, cc, color=NAVY, lw=1.8, label="simulated, 200 subcarriers")
    ax.semilogy(g, np.exp(-10 ** (g / 10)), "--", color=ACCENT, lw=1.0, label="$e^{-\\gamma}$ (Gaussian)")
    ax.axvline(10, color=GRAY, ls=":", lw=0.8)
    ax.text(10.2, 0.05, "10 dB:\n1 sample\nin $\\approx$ 22 000", fontsize=7, color=GRAY)
    ax.set_xlabel("instantaneous power above mean (dB)")
    ax.set_ylabel("fraction of samples above")
    ax.set_ylim(1e-6, 1.5); ax.set_xlim(0, 13); ax.legend(fontsize=6.6, loc="lower left")
    fig.tight_layout(); save(fig, "ch17_papr_crowd")


def hft_grid():
    """The channel as a landscape |H(f,t)|, sampled by a scattered pilot lattice."""
    r = rng(7)
    dl, pdb = cl.TDL_PROFILES["EVA"]
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
    nsc, nsym = 96, 28
    df, Ts, fD = 15e3, 1e-3 / 14, 300.0
    k = np.arange(nsc) - nsc / 2
    t = np.arange(nsym) * Ts
    H = np.zeros((nsc, nsym), complex)
    for tau, pi in zip(np.asarray(dl) * 1e-9, p):
        th = r.uniform(0, 2 * np.pi, 24); ph = r.uniform(0, 2 * np.pi, 24)
        a = np.sqrt(pi / 24) * np.exp(1j * (2 * np.pi * fD * np.outer(t, np.cos(th)) + ph)).sum(axis=1)
        H += np.exp(-2j * np.pi * np.outer(k * df, [tau]))[:, :1] * a[None, :]
    fig, ax = plt.subplots(figsize=(W1 * 0.82, 2.5))
    im = ax.imshow(20 * np.log10(np.abs(H)), origin="lower", aspect="auto", cmap="viridis", vmin=-20, vmax=8,
                   extent=(-0.5, nsym - 0.5, -0.5, nsc - 0.5))
    for s in range(nsym):
        if s % 7 in (0, 4):
            off = 0 if s % 7 == 0 else 3
            kk = np.arange(off, nsc, 6)
            ax.plot(np.full(len(kk), s), kk, "o", ms=2.6, mfc="white", mec="k", mew=0.4)
    ax.set_xlabel("OFDM symbol (time)"); ax.set_ylabel("subcarrier (frequency)")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("$|H(f,t)|^2$ (dB)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch17_hft_grid")


def waterfill_cartoon():
    """Water-filling as water poured over an uneven floor."""
    floor = np.array([0.35, 0.2, 0.3, 0.55, 0.9, 1.45, 0.75, 0.4, 0.25, 0.5, 1.1, 1.7])
    P = 4.0
    lo, hi = 0, 4
    for _ in range(60):
        mu = (lo + hi) / 2
        if np.sum(np.maximum(mu - floor, 0)) > P:
            hi = mu
        else:
            lo = mu
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i, fl in enumerate(floor):
        ax.add_patch(Rectangle((i, 0), 1, fl, color=GRAY, alpha=0.75, lw=0))
        if mu > fl:
            ax.add_patch(Rectangle((i, fl), 1, mu - fl, color="#2E86C1", alpha=0.45, lw=0))
    ax.axhline(mu, color="#2E86C1", lw=1.2, ls="--")
    ax.text(0.1, mu + 0.06, "water level $\\mu$", fontsize=7.5, color="#2E86C1")
    ax.annotate("too high:\nno power", xy=(11.5, 1.72), xytext=(8.0, 1.85), fontsize=7, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.6))
    ax.text(5.5, 0.12, "floor $\\Gamma/g_k$\n(noise / gain)", fontsize=6.8, color="white", ha="center")
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.25)
    ax.set_xlabel("subcarrier (tone)"); ax.set_ylabel("power")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); save(fig, "ch17_waterfill_cartoon")


def spacing_span():
    """Subcarrier spacings of real systems span more than three decades."""
    sysl = [("DVB-T2 32k", 279), ("DAB mode I", 1e3), ("DVB-T 8k", 1116), ("ADSL", 4312.5),
            ("LTE / NR $\\mu$=0", 15e3), ("HomePlug AV", 24414), ("DOCSIS 3.1", 50e3), ("NR $\\mu$=1", 30e3),
            ("Wi-Fi 6/7", 78125), ("NR $\\mu$=3 (FR2)", 120e3), ("Wi-Fi 4/5 (11a)", 312.5e3), ("NR $\\mu$=6", 960e3)]
    sysl.sort(key=lambda s: s[1])
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    for i, (n, d) in enumerate(sysl):
        c = ACCENT if "NR" in n or "LTE" in n else (GREEN if n.startswith(("DVB", "DAB")) else
                                                     (ORANGE if n.startswith(("ADSL", "DOCSIS", "HomePlug")) else NAVY))
        ax.plot(d, i, "o", color=c, ms=5)
        ax.plot([100, d], [i, i], color=c, lw=0.6, alpha=0.5)
        ax.text(d * 1.25, i, f"{n}: {d / 1e3:g} kHz;  $T$ = {1e6 / d:.1f} $\\mu$s",
                fontsize=6.8, va="center", color=c)
    ax.set_xscale("log"); ax.set_xlim(150, 3e7); ax.set_ylim(-0.8, len(sysl) - 0.2)
    ax.set_yticks([]); ax.set_xlabel("subcarrier spacing $\\Delta f$ (Hz)")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6]); ax.set_xticklabels(["1 kHz", "10 kHz", "100 kHz", "1 MHz"])
    ax.text(2e6, 1.0, "broadcast (green)\nwireline (orange)\ncellular (red)\nWi-Fi (blue)", fontsize=7,
            color=GRAY, va="bottom")
    fig.tight_layout(); save(fig, "ch17_spacing_span")


def ofdma_sched():
    """Multiuser diversity: a proportional-fair scheduler gives each RB to a user on a peak."""
    r = rng(12)
    U, nrb, ns = 3, 25, 40
    snr0 = np.array([12.0, 8.0, 4.0])
    # frequency/time correlated Rayleigh: smooth complex Gaussian fields
    g = np.zeros((U, nrb, ns))
    for u in range(U):
        w = r.standard_normal((nrb + 8, ns + 8)) + 1j * r.standard_normal((nrb + 8, ns + 8))
        ker = np.outer(np.hanning(7), np.hanning(7))
        from scipy.signal import fftconvolve
        f = fftconvolve(w, ker, mode="same")[4:-4, 4:-4]
        f /= np.sqrt(np.mean(np.abs(f) ** 2))
        g[u] = np.abs(f) ** 2
    snr = 10 ** (snr0[:, None, None] / 10) * g
    rate = np.log2(1 + snr)
    avg = np.ones(U) * 1.0
    alloc = np.zeros((nrb, ns), int)
    tp_pf = np.zeros(U); tp_rr = np.zeros(U)
    for s in range(ns):
        served = np.zeros(U)
        for b in range(nrb):
            u = np.argmax(rate[:, b, s] / avg)
            alloc[b, s] = u; served[u] += rate[u, b, s]
            tp_rr[(b + s) % U] += rate[(b + s) % U, b, s]
        tp_pf += served
        avg = 0.9 * avg + 0.1 * served / nrb + 1e-9
    gain = tp_pf.sum() / tp_rr.sum()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.1, 1]})
    cols = [NAVY, ACCENT, GREEN]
    for u in range(U):
        ax[0].plot(np.arange(nrb), 10 * np.log10(snr[u, :, 10]), "o-", ms=2.4, color=cols[u], lw=1.1,
                   label=f"user {u + 1}")
    ax[0].set_xlabel("resource block"); ax[0].set_ylabel("SNR (dB)")
    ax[0].legend(fontsize=6.8, ncol=3, loc="lower center"); ax[0].set_ylim(-25, 25)
    ax[0].set_title("(a) Three users' channels, one slot", fontsize=9)
    from matplotlib.colors import ListedColormap
    ax[1].imshow(alloc, origin="lower", aspect="auto", cmap=ListedColormap(cols), interpolation="nearest")
    ax[1].set_xlabel("slot"); ax[1].set_ylabel("resource block"); ax[1].grid(False)
    ax[1].set_title(f"(b) PF schedule: {100 * (gain - 1):.0f}% over round robin", fontsize=9)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_ofdma_sched")
    print("PF gain over RR:", gain)


def sfn_map():
    """Self-interference zones of a two-transmitter SFN with a short guard interval."""
    x = np.linspace(-60, 120, 600); y = np.linspace(-80, 80, 500)
    Xg, Yg = np.meshgrid(x, y)
    d1 = np.hypot(Xg, Yg) + 0.5; d2 = np.hypot(Xg - 60, Yg) + 0.5
    dd = np.abs(d1 - d2) / 0.3          # microseconds
    pr = 35 * np.abs(np.log10(d2 / d1))   # dB between the two signals (path-loss exponent 3.5)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bad = (dd > 112) & (pr < 15)
    ax.contourf(Xg, Yg, bad.astype(float), levels=[0.5, 1.5], colors=[ACCENT], alpha=0.35)
    cs = ax.contour(Xg, Yg, dd, levels=[50, 100, 150], colors=[GRAY], linewidths=0.6)
    ax.clabel(cs, fmt="%d $\\mu$s", fontsize=6.5)
    ax.plot([0, 60], [0, 0], "^", color=NAVY, ms=7)
    ax.text(0, 6, "Tx A", ha="center", fontsize=7.5, color=NAVY); ax.text(60, 6, "Tx B", ha="center", fontsize=7.5,
                                                                          color=NAVY)
    ax.set_xlabel("km"); ax.set_ylabel("km"); ax.set_aspect("equal")
    fig.tight_layout(); save(fig, "ch17_sfn_map")


def mimo_sv():
    """MIMO-OFDM: a 4x4 frequency-selective channel is a flat 4x4 matrix on every subcarrier."""
    r = rng(9)
    dl, pdb = cl.TDL_PROFILES["EVA"]
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum()
    d = np.round(np.asarray(dl) * 1e-9 * FS_LTE).astype(int)
    hmat = np.zeros((d.max() + 1, 4, 4), complex)
    for di, pi in zip(d, p):
        hmat[di] += np.sqrt(pi / 2) * (r.standard_normal((4, 4)) + 1j * r.standard_normal((4, 4)))
    Hf = np.fft.fft(hmat, 1024, axis=0)
    k = np.r_[1:301, 724:1024]
    sv = np.array([np.linalg.svd(Hf[i], compute_uv=False) for i in k])
    order = np.argsort(np.where(k > 512, k - 1024, k))
    fk = np.where(k > 512, k - 1024, k)[order] * 15e-3
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i in range(4):
        ax.plot(fk, 20 * np.log10(sv[order, i]), color=CYCLE[i], lw=1.0, label=f"$\\sigma_{i + 1}$")
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("singular value (dB)")
    ax.legend(fontsize=6.6, ncol=4, loc="lower center"); ax.set_ylim(-35, 15)
    fig.tight_layout(); save(fig, "ch17_mimo_sv")


def radar_map():
    """OFDM radar: range-Doppler map from dividing out the known data and a 2-D FFT."""
    r = rng(14)
    c0, fc, df = 3e8, 3.5e9, 30e3
    nsc, nsym = 1024, 128
    Ts = 1 / df * (1 + 144 / 2048)
    lam = c0 / fc
    tg = [(40.0, 12.0, 1.0), (95.0, -25.0, 0.5)]
    k = np.arange(nsc)[:, None]; l = np.arange(nsym)[None, :]
    X = _qam(nsc * nsym, QPSK, r).reshape(nsc, nsym)
    Y = np.zeros_like(X)
    for R, v, a in tg:
        Y += a * X * np.exp(-2j * np.pi * k * df * 2 * R / c0) * np.exp(2j * np.pi * (2 * v / lam) * l * Ts)
    Y += 10 ** (-10 / 20) * (r.standard_normal(X.shape) + 1j * r.standard_normal(X.shape)) / np.sqrt(2)
    D = Y / X
    D = D * np.hanning(nsc)[:, None] * np.hanning(nsym)[None, :]
    RD = np.fft.fftshift(np.fft.fft(np.fft.ifft(D, 4 * nsc, axis=0), 4 * nsym, axis=1), axes=1)
    P = 20 * np.log10(np.abs(RD) + 1e-12); P -= P.max()
    rng_ax = np.arange(4 * nsc) * c0 / (2 * 4 * nsc * df)
    vel = (np.arange(4 * nsym) - 2 * nsym) / (4 * nsym * Ts) * lam / 2
    sel = rng_ax < 160
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(P[sel].T, origin="lower", aspect="auto", cmap="magma", vmin=-50, vmax=0,
                   extent=(rng_ax[0], rng_ax[sel][-1], vel[0], vel[-1]))
    ax.set_ylim(-60, 60)
    ax.set_xlabel("range (m)"); ax.set_ylabel("radial velocity (m/s)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("dB", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch17_radar_map")


def wifi_rates():
    """Peak PHY rates of Wi-Fi generations, and the knobs that were turned."""
    gens = [("802.11a\n1999", 54e6, "20 MHz, 1 stream\n64-QAM"),
            ("802.11n\n2009", 600e6, "40 MHz, 4 streams"),
            ("802.11ac\n2013", 6.93e9, "160 MHz, 8 str.\n256-QAM"),
            ("802.11ax\n2021", 9.6e9, "1024-QAM,\n78 kHz spacing"),
            ("802.11be\n2024", 23e9, "320 MHz,\n4096-QAM")]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for i, (g, rt, kn) in enumerate(gens):
        ax.bar(i, rt, color=NAVY if i < 4 else ACCENT, width=0.65)
        lab = f"{rt / 1e9:.1f} Gb/s" if rt >= 1e9 else f"{rt / 1e6:.0f} Mb/s"
        ax.text(i, rt * 1.25, lab, ha="center", fontsize=6.8)
    ax.set_yscale("log"); ax.set_ylim(1.5e7, 1e11)
    ax.set_xticks(range(5)); ax.set_xticklabels([g[0] for g in gens], fontsize=6.6)
    ax.set_ylabel("peak PHY rate (b/s)")
    fig.tight_layout(); save(fig, "ch17_wifi_rates")


def dfts_env():
    """Envelope of CP-OFDM versus DFT-spread OFDM with the same 300 subcarriers (QPSK)."""
    r = rng(18)
    N, M, L, ns = 1024, 300, 4, 1500
    d = _qam(M * ns, QPSK, r).reshape(ns, M)
    X1 = np.zeros((ns, N * L), complex); X1[:, :M] = d
    X2 = np.zeros((ns, N * L), complex); X2[:, :M] = np.fft.fft(d, axis=1) / np.sqrt(M)
    x1 = np.fft.ifft(X1, axis=1); x2 = np.fft.ifft(X2, axis=1)
    x1 /= np.sqrt(np.mean(np.abs(x1) ** 2)); x2 /= np.sqrt(np.mean(np.abs(x2) ** 2))
    t = np.arange(3 * N * L) / (N * L)
    fig, ax = plt.subplots(2, 1, figsize=(3.0, 2.5), sharex=True)
    for a, x, c, n in [(ax[0], x1, ACCENT, "CP-OFDM"), (ax[1], x2, NAVY, "DFT-s-OFDM")]:
        pw = np.abs(x) ** 2
        papr = 10 * np.log10(pw.max(axis=1))
        q = np.quantile(papr, 1 - 1e-3 * 1.5)
        a.plot(t, pw[:3].ravel(), color=c, lw=0.6)
        a.axhline(1, color=GRAY, ls="--", lw=0.6)
        a.set_ylim(0, 11); a.set_yticks([0, 5, 10])
        a.set_title(f"{n}: 0.1% of symbols exceed {q:.1f} dB", fontsize=7.5, color=c, pad=2)
    ax[1].set_xlabel("time (three QPSK symbols, 300 subcarriers)")
    fig.text(0.01, 0.5, "power / mean", rotation=90, va="center", fontsize=8)
    fig.tight_layout(rect=(0.03, 0, 1, 1), h_pad=0.3); save(fig, "ch17_dfts_env")


def pa_curve():
    """A power amplifier's compression curve and where OFDM amplitudes fall on it."""
    r = rng(19)
    a_in = np.linspace(0, 2.0, 400)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for p, c, n in [(2, ORANGE, "Rapp $p=2$"), (10, NAVY, "with DPD ($p=10$)")]:
        ax.plot(a_in, np.abs(cl.rapp_pa(a_in.astype(complex), 1.0, p)), color=c, lw=1.5, label=n)
    ax.plot(a_in, a_in, ":", color=GRAY, lw=0.8)
    rms = 10 ** (-8 / 20)
    x = (r.standard_normal(200000) + 1j * r.standard_normal(200000)) / np.sqrt(2) * rms
    h, e = np.histogram(np.abs(x), bins=120, range=(0, 2), density=True)
    ax.fill_between(0.5 * (e[1:] + e[:-1]), 0, h / h.max() * 0.45, color=ACCENT, alpha=0.3, lw=0,
                    label="OFDM amplitudes")
    ax.axvline(rms, color=ACCENT, ls="--", lw=0.8)
    ax.text(rms + 0.03, 1.15, "RMS, 8 dB\nback-off", fontsize=7, color=ACCENT)
    ax.set_xlabel("input amplitude (saturation = 1)"); ax.set_ylabel("output amplitude")
    ax.set_ylim(0, 1.45); ax.set_xlim(0, 2); ax.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout(); save(fig, "ch17_pa_curve")


def optical_ofdm():
    """Real, non-negative OFDM for light: DC-biased and asymmetrically clipped."""
    r = rng(23)
    N, L = 64, 8
    X = np.zeros(N * L, complex)
    k = np.arange(1, 28)
    X[k] = _qam(len(k), QPSK, r); X[-k] = np.conj(X[k])
    x = np.fft.ifft(X).real; x /= x.std()
    Xa = np.zeros(N * L, complex)
    ko = np.arange(1, 28, 2)
    Xa[ko] = _qam(len(ko), QPSK, r); Xa[-ko] = np.conj(Xa[ko])
    xa = np.fft.ifft(Xa).real; xa /= xa.std()
    t = np.arange(N * L) / (N * L)
    fig, ax = plt.subplots(2, 1, figsize=(3.0, 2.5), sharex=True)
    dco = np.maximum(x + 2.0, 0)
    ax[0].plot(t, x + 2.0, color=GRAY, lw=0.6, ls=":")
    ax[0].plot(t, dco, color=ORANGE, lw=1.0)
    ax[0].axhline(2.0, color=GRAY, lw=0.5, ls="--")
    ax[0].set_title("DCO-OFDM: real signal + DC bias (dashed)", fontsize=8)
    ax[1].plot(t, xa, color=GRAY, lw=0.6, ls=":")
    ax[1].plot(t, np.maximum(xa, 0), color=PURPLE, lw=1.0)
    ax[1].set_title("ACO-OFDM: odd subcarriers, negatives clipped", fontsize=8)
    for a in ax:
        a.axhline(0, color="k", lw=0.5); a.set_yticks([0])
    ax[1].set_xlabel("time (one symbol)")
    fig.tight_layout(h_pad=0.2); save(fig, "ch17_optical_ofdm")


def fbmc_proto():
    """The FBMC (PHYDYAS, K=4) prototype pulse versus the rectangular OFDM window."""
    K = 4
    Hk = [1.0, 0.97195983, 1 / np.sqrt(2), 0.23514695]
    t = np.linspace(0, K, 4000)
    p = Hk[0] + 2 * sum((-1) ** k * Hk[k] * np.cos(2 * np.pi * k * t / K) for k in range(1, K))
    p /= np.sqrt(np.trapezoid(p ** 2, t))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 0.95, 2.1))
    ax[0].plot(t - K / 2, p, color=NAVY, lw=1.4, label="FBMC prototype")
    ax[0].plot([-0.5, -0.5, 0.5, 0.5], [0, 1, 1, 0], color=ACCENT, lw=1.2, label="OFDM rectangle")
    ax[0].set_xlabel("time (symbols)"); ax[0].legend(fontsize=6.6, loc="upper right")
    ax[0].set_title("(a) Pulses", fontsize=9)
    nf = 2 ** 16; dt = t[1] - t[0]
    Pf = np.abs(np.fft.fftshift(np.fft.fft(p, nf))) * dt
    f = np.fft.fftshift(np.fft.fftfreq(nf, dt))
    tr = np.linspace(-0.5, 0.5, 1000); Rf = np.abs(np.sinc(f))
    ax[1].plot(f, 20 * np.log10(Pf / Pf.max() + 1e-9), color=NAVY, lw=1.2)
    ax[1].plot(f, 20 * np.log10(Rf + 1e-9), color=ACCENT, lw=1.0)
    ax[1].set_xlim(0, 8); ax[1].set_ylim(-80, 3)
    ax[1].set_xlabel("frequency (subcarrier spacings)"); ax[1].set_ylabel("dB")
    ax[1].set_title("(b) Spectra of one subcarrier", fontsize=9)
    fig.tight_layout(w_pad=0.8); save(fig, "ch17_fbmc_proto")


def eq_cost():
    """Complex multiplies per data symbol: time-domain equaliser versus OFDM, 20 MHz."""
    fs = 20e6
    tau = np.logspace(-7, np.log10(2e-5), 80)
    nu = tau * fs
    tde = 4 * nu + 1
    cost_ofdm = []
    for n_ in nu:
        N = 64
        while N < 8 * max(n_, 2):
            N *= 2
        cost_ofdm.append((N / 2 * np.log2(N) + N) / N)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.loglog(tau * 1e6, tde, color=ACCENT, lw=1.6, label="time-domain FIR\n($\\approx4\\nu$ taps)")
    ax.loglog(tau * 1e6, cost_ofdm, color=NAVY, lw=1.6, label="OFDM: FFT + one tap")
    ax.axvline(5, color=GRAY, ls=":", lw=0.8); ax.text(5.4, 2.0, "urban\nmacro", fontsize=7, color=GRAY)
    ax.set_xlabel("channel delay spread ($\\mu$s)"); ax.set_ylabel("multiplies per symbol")
    ax.legend(fontsize=6.6, loc="upper left"); ax.set_ylim(1, 3000)
    fig.tight_layout(); save(fig, "ch17_eq_cost")


def plc_mask():
    """An illustrative power-line OFDM tone map: amateur bands notched, bits loaded per tone."""
    r = rng(31)
    df = 24.414e3
    f = np.arange(1.8e6, 30e6, df)
    ham = [(1.8, 2.0), (3.5, 4.0), (7.0, 7.3), (10.1, 10.15), (14.0, 14.35), (18.068, 18.168),
           (21.0, 21.45), (24.89, 24.99), (28.0, 29.7)]
    on = np.ones(len(f), bool)
    for a, b in ham:
        on &= ~((f >= a * 1e6) & (f <= b * 1e6))
    # a frequency-selective power-line channel: a few echoes plus loss rising with frequency
    tau = np.array([0, 0.18, 0.41, 0.77, 1.2]) * 1e-6
    g = np.array([1.0, -0.55, 0.38, -0.25, 0.15])
    H = np.abs(np.sum(g[:, None] * np.exp(-2j * np.pi * tau[:, None] * f[None, :]), axis=0))
    snr = 45 + 20 * np.log10(H + 1e-3) - 0.6 * f / 1e6 + 2 * r.standard_normal(len(f))
    bits = np.clip(np.floor(np.log2(1 + 10 ** ((snr - 9) / 10))), 0, 10) * on
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 2.2))
    ax.bar(f / 1e6, bits, width=df / 1e6, color=NAVY, lw=0)
    for a, b in ham:
        ax.axvspan(a, b, color=ACCENT, alpha=0.25, lw=0)
    ax.text(7.15, 10.3, "amateur bands\nnotched", fontsize=7, color=ACCENT, ha="center")
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("bits per subcarrier")
    ax.set_xlim(1.5, 30.2); ax.set_ylim(0, 12.5)
    fig.tight_layout(); save(fig, "ch17_plc_mask")


def rayleigh_lanes():
    """Uncoded OFDM on a fading channel: a few weak subcarriers make nearly all the errors."""
    r = rng(41)
    nsc, ntr = 600, 400
    snr_avg = 20.0
    allsnr = []
    for _ in range(ntr):
        h = _static_taps("ETU", FS_LTE, r); h /= np.sqrt(np.sum(np.abs(h) ** 2))
        H = np.fft.fft(h, 1024)[np.r_[1:301, 724:1024]]
        allsnr.append(snr_avg + 20 * np.log10(np.abs(H)))
    s = np.concatenate(allsnr)
    from scipy.special import erfc
    ser = erfc(np.sqrt(10 ** (s / 10) / 10 * 1.0)) * 1.5      # 16-QAM approximate SER
    ser = np.minimum(ser, 0.75)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    bins = np.linspace(-15, 30, 46)
    h, e = np.histogram(s, bins=bins)
    errs, _ = np.histogram(s, bins=bins, weights=ser)
    ax.bar(e[:-1], h / h.sum() * 100, width=1.0, align="edge", color=NAVY, alpha=0.7, label="share of subcarriers")
    ax.bar(e[:-1], errs / errs.sum() * 100, width=1.0, align="edge", color=ACCENT, alpha=0.6,
           label="share of symbol errors")
    weak = s < 10
    ax.text(-14, 23, f"{100 * weak.mean():.0f}% of subcarriers below 10 dB\nmake {100 * ser[weak].sum() / ser.sum():.0f}% of the errors",
            fontsize=6.8)
    ax.set_xlabel("subcarrier SNR (dB), average 20 dB"); ax.set_ylabel("percent")
    ax.legend(fontsize=6.4, loc="center right"); ax.set_ylim(0, 30)
    fig.tight_layout(); save(fig, "ch17_rayleigh_lanes")


def cfo_rotation():
    """A frequency offset rotates and blurs the constellation (CPE + ICI)."""
    r = rng(51)
    cfg = co.OFDMConfig(64, 52, 16)
    fig, ax = plt.subplots(1, 2, figsize=(3.2, 1.75))
    for a, eps in zip(ax, [0.04, 0.15]):
        Ys = []
        for rep_ in range(6):   # six independent first symbols: same rotation, fresh ICI
            g = _qam(52, Q16, r).reshape(1, 52)
            x = co.ofdm_modulate(g, cfg)
            n = np.arange(len(x))
            Ys.append(co.ofdm_demodulate(x * np.exp(2j * np.pi * eps * n / 64), cfg)[0])
        Y = np.concatenate(Ys)
        a.plot(Y.real, Y.imag, ".", color=NAVY, ms=2.5)
        a.plot(Q16.points.real, Q16.points.imag, "+", color=ACCENT, ms=4, mew=0.7)
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"$\\epsilon$ = {eps}", fontsize=8.5)
    fig.tight_layout(w_pad=0.3); save(fig, "ch17_cfo_rotation")


def doppler_speed():
    """Doppler signal-to-ICI ratio versus speed for three numerologies (Clarke spectrum)."""
    v = np.linspace(1, 500, 300) / 3.6
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for fc, scs, c, ls in [(3.5e9, 15e3, ACCENT, "-"), (3.5e9, 30e3, NAVY, "-"), (28e9, 120e3, GREEN, "--"),
                           (28e9, 30e3, GREEN, ":")]:
        fd = v * fc / 3e8
        sir = 6 / (np.pi * fd / scs) ** 2
        ax.plot(v * 3.6, 10 * np.log10(sir), color=c, ls=ls, lw=1.4,
                label=f"{fc / 1e9:g} GHz, {scs / 1e3:g} kHz")
    ax.axhline(25, color=GRAY, ls="--", lw=0.7); ax.text(10, 26, "64-QAM comfort zone", fontsize=6.8, color=GRAY)
    ax.set_xlabel("speed (km/h)"); ax.set_ylabel("Doppler SIR (dB)")
    ax.set_ylim(5, 60); ax.legend(fontsize=6.2, loc="upper right")
    fig.tight_layout(); save(fig, "ch17_doppler_speed")


def dmrs_types():
    """NR DMRS configurations on one slot of one resource block."""
    fig, ax = plt.subplots(1, 3, figsize=(W1 * 0.95, 2.0))
    cfgs = [("(a) Type 1, front-loaded", [2], "t1"), ("(b) Type 1 + 1 additional", [2, 11], "t1"),
            ("(c) Type 2 + 2 additional", [2, 7, 11], "t2")]
    for a, (tt, syms, typ) in zip(ax, cfgs):
        a.add_patch(Rectangle((2, 0), 12, 12, color=GREEN, alpha=0.12, lw=0))
        a.add_patch(Rectangle((0, 0), 2, 12, color=NAVY, alpha=0.2, lw=0))
        for s in syms:
            for k in range(12):
                on = (k % 2 == 0) if typ == "t1" else (k % 6 in (0, 1))
                if on:
                    a.add_patch(Rectangle((s, k), 1, 1, color=ACCENT, alpha=0.85, lw=0))
                elif typ == "t1":
                    a.add_patch(Rectangle((s, k), 1, 1, color=ORANGE, alpha=0.35, lw=0))
                elif k % 6 in (2, 3):
                    a.add_patch(Rectangle((s, k), 1, 1, color=ORANGE, alpha=0.35, lw=0))
                else:
                    a.add_patch(Rectangle((s, k), 1, 1, color=PURPLE, alpha=0.3, lw=0))
        for x in range(15):
            a.plot([x, x], [0, 12], color="white", lw=0.5)
        for y in range(13):
            a.plot([0, 14], [y, y], color="white", lw=0.5)
        a.add_patch(Rectangle((0, 0), 14, 12, fill=False, ec=GRAY, lw=0.6))
        a.set_xlim(-0.2, 14.2); a.set_ylim(-0.3, 12.3); a.set_aspect("equal")
        a.set_title(tt, fontsize=8); _clean_ax(a)
        a.set_xlabel("symbol", fontsize=7)
    ax[0].set_ylabel("subcarrier", fontsize=7)
    fig.text(0.5, -0.02, "red: DMRS of one CDM group; orange/purple: other CDM groups (more antenna ports); "
             "blue: control; green: data", ha="center", fontsize=6.8, color=GRAY)
    fig.tight_layout(w_pad=0.4); save(fig, "ch17_dmrs_types")


def sco_ramp():
    """Sampling-clock offset: a phase ramp across the subcarriers that grows symbol by symbol."""
    k = np.arange(-26, 27); k = k[k != 0]
    delta, N, L = 20e-6, 64, 80
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for nsym, c in [(50, GREEN), (150, NAVY), (300, ACCENT)]:
        ph = np.degrees(2 * np.pi * k * delta * L / N * nsym)
        ax.plot(k, ph, "o", ms=1.8, color=c, label=f"after {nsym} symbols")
    for kp in (-21, -7, 7, 21):
        ax.axvline(kp, color=GRAY, ls=":", lw=0.7)
    ax.text(7.6, -72, "pilots", fontsize=7, color=GRAY)
    ax.set_xlabel("subcarrier index $k$"); ax.set_ylabel("phase rotation (degrees)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, "ch17_sco_ramp")


def coverage_gain():
    """Cell-radius gain from saving back-off, for an uplink-limited cell."""
    db = np.linspace(0, 5, 100)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for n, c in [(3.0, GREEN), (3.5, NAVY), (4.0, ACCENT)]:
        ax.plot(db, 100 * (10 ** (db / (10 * n)) - 1), color=c, lw=1.5, label=f"path-loss exponent {n:g}")
    ax.axvspan(2, 3, color=GRAY, alpha=0.15, lw=0)
    ax.text(2.05, 2, "SC-FDMA\nvs OFDMA\n(QPSK)", fontsize=6.8, color=GRAY)
    ax.set_xlabel("back-off saved (dB)"); ax.set_ylabel("cell radius gain (%)")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_xlim(0, 5); ax.set_ylim(0, 50)
    fig.tight_layout(); save(fig, "ch17_coverage_gain")


def cp_overhead():
    """Prefix length and overhead of real systems."""
    sy = [("802.11a/g/n/ac", 0.8, 3.2), ("Wi-Fi 6/7 (0.8 GI)", 0.8, 12.8), ("LTE normal", 4.69, 66.67),
          ("LTE extended", 16.67, 66.67), ("NR $\\mu$=1", 2.34, 33.33), ("DVB-T2 32k, 1/16", 224, 3584)]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i, (n, tcp, T) in enumerate(sy):
        ov = 100 * tcp / (tcp + T)
        ax.barh(i, ov, color=ACCENT if ov > 15 else NAVY, height=0.6)
        ax.text(ov + 0.5, i, f"{ov:.1f}%  ($T_{{cp}}$={tcp:g} $\\mu$s)", va="center", fontsize=6.6)
    ax.set_yticks(range(len(sy))); ax.set_yticklabels([s[0] for s in sy], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 40); ax.set_xlabel("prefix overhead (% of air time)")
    fig.tight_layout(); save(fig, "ch17_cp_overhead")


def guard_layout():
    """802.11a: what each of the 64 FFT bins carries."""
    fig, ax = plt.subplots(figsize=(W1, 1.25))
    for k in range(-32, 32):
        if k == 0:
            c, h = GRAY, 0.25
        elif abs(k) > 26:
            c, h = "#DDDDDD", 0.25
        elif abs(k) in (7, 21):
            c, h = ACCENT, 0.8
        else:
            c, h = NAVY, 0.6
        ax.bar(k, h, width=0.8, color=c)
    ax.set_xlim(-33, 33); ax.set_ylim(0, 1.15); ax.set_yticks([])
    ax.set_xticks([-32, -26, -21, -7, 0, 7, 21, 26, 31])
    ax.set_xlabel("subcarrier index $k$ (312.5 kHz apart)")
    ax.text(-29.5, 0.4, "guard", ha="center", fontsize=7, color=GRAY)
    ax.text(29.5, 0.4, "guard", ha="center", fontsize=7, color=GRAY)
    ax.text(0, 0.3, "DC", ha="center", fontsize=7, color=GRAY)
    ax.text(-14, 0.88, "48 data subcarriers", ha="center", fontsize=7.5, color=NAVY)
    ax.text(14, 0.88, "4 pilots (red)", ha="center", fontsize=7.5, color=ACCENT)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch17_guard_layout")


def delay_vs_cp():
    """Typical maximum excess delays by environment against the prefixes of real systems."""
    env = [("indoor (Wi-Fi)", 0.05, 0.8), ("urban macro-cell", 1.0, 5.0), ("hilly terrain", 10, 20),
           ("broadcast SFN", 50, 500)]
    cps = [("802.11a 0.8", 0.8, NAVY), ("LTE 4.69", 4.69, ACCENT), ("LTE ext. 16.7", 16.7, ORANGE),
           ("DVB-T2 224", 224, GREEN)]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i, (n, a, b) in enumerate(env):
        ax.plot([a, b], [i, i], color=GRAY, lw=7, solid_capstyle="butt", alpha=0.6)
        ax.text(a, i + 0.32, n, fontsize=7)
    for n, v, c in cps:
        ax.axvline(v, color=c, ls="--", lw=0.9)
        ax.text(v * 1.08, -0.75, n, rotation=90, fontsize=6.3, color=c, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.03, 1500); ax.set_ylim(-0.8, 3.7)
    ax.set_yticks([]); ax.set_xlabel("maximum excess delay / prefix ($\\mu$s)")
    fig.tight_layout(); save(fig, "ch17_delay_vs_cp")


def dvbt2_guards():
    """DVB-T2 guard intervals: duration and overhead for the 8k and 32k modes (8 MHz)."""
    fr = [1 / 128, 1 / 32, 1 / 16, 19 / 256, 1 / 8, 19 / 128, 1 / 4]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for T, c, n, ok in [(896, ACCENT, "8k ($T$ = 896 $\\mu$s)", (1 / 128, 1 / 32, 1 / 16, 19 / 256, 1 / 8, 19 / 128, 1 / 4)),
                        (3584, NAVY, "32k ($T$ = 3584 $\\mu$s)", (1 / 128, 1 / 32, 1 / 16, 19 / 256, 1 / 8, 19 / 128))]:
        g =[f for f in fr if f in ok]
        ax.plot([f * T for f in g], [100 * f / (1 + f) for f in g], "o-", color=c, ms=3.5, label=n)
    ax.axvline(200, color=GRAY, ls=":", lw=0.9)
    ax.text(215, 1.0, "60 km SFN\nneeds 200 $\\mu$s", fontsize=7, color=GRAY)
    ax.set_xscale("log"); ax.set_xlabel("guard interval ($\\mu$s)"); ax.set_ylabel("overhead (%)")
    ax.legend(fontsize=6.6, loc="upper left")
    fig.tight_layout(); save(fig, "ch17_dvbt2_guards")


def pilot_limits():
    """Largest pilot spacing in time allowed by the Doppler, NR at 30 kHz, against speed."""
    v = np.linspace(5, 500, 300) / 3.6
    Ts = 1 / 30e3 * (1 + 144 / 2048)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for fc, c in [(2.0e9, GREEN), (3.5e9, NAVY), (28e9, ACCENT)]:
        fd = v * fc / 3e8
        ax.semilogy(v * 3.6, 1 / (2 * fd * Ts), color=c, lw=1.5, label=f"{fc / 1e9:g} GHz")
    for d, n in [(14, "1 DMRS / slot"), (7, "2 DMRS"), (4, "4 DMRS")]:
        ax.axhline(d, color=GRAY, ls=":", lw=0.8); ax.text(150, d * 1.08, n, fontsize=6.6, color=GRAY)
    ax.set_xlabel("speed (km/h)"); ax.set_ylabel("max. pilot spacing $D_t$ (symbols)")
    ax.set_ylim(1, 1000); ax.legend(fontsize=6.6, loc="upper right", title="30 kHz, carrier", title_fontsize=6.6)
    fig.tight_layout(); save(fig, "ch17_pilot_limits")


def adsl_bandplan():
    """ADSL2+ over POTS (Annex A): voice, upstream tones and downstream tones on one copper pair."""
    fig, ax = plt.subplots(figsize=(W1, 1.45))
    df = 4.3125
    ax.add_patch(Rectangle((0, 0), 4, 1.0, color=GREEN, alpha=0.6, lw=0))
    ax.add_patch(Rectangle((6 * df, 0), (32 - 6) * df, 0.7, color=ORANGE, alpha=0.7, lw=0))
    ax.add_patch(Rectangle((33 * df, 0), (511 - 33) * df, 0.55, color=NAVY, alpha=0.7, lw=0))
    ax.set_xscale("symlog", linthresh=20); ax.set_xlim(0, 2400); ax.set_ylim(0, 1.25)
    ax.text(2, 1.05, "voice\n0–4 kHz", fontsize=7, color=GREEN, ha="center", va="bottom")
    ax.text(60, 0.75, "upstream\ntones 6–32", fontsize=7, color=ORANGE, ha="center", va="bottom")
    ax.text(600, 0.6, "downstream: tones 33–511 (138 kHz–2.2 MHz)", fontsize=7, color=NAVY, ha="center",
            va="bottom")
    ax.set_xticks([0, 4, 25, 138, 1104, 2208]); ax.set_xticklabels(["0", "4", "25", "138", "1104", "2208"])
    ax.set_xlabel("frequency (kHz), 4.3125 kHz per tone"); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); save(fig, "ch17_adsl_bandplan")


def dft_est():
    """DFT-based estimation: in the delay domain the channel is a few taps and the rest is noise."""
    r = rng(61)
    npil = 100
    h = np.zeros(npil, complex)
    taps = [0, 1, 3, 5, 9, 14, 22]
    for i, t in enumerate(taps):
        h[t] = np.exp(-i / 2.5) * np.exp(2j * np.pi * r.random())
    noise = (r.standard_normal(npil) + 1j * r.standard_normal(npil)) * 0.07
    hh = h + noise
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    n = np.arange(npil)
    keep = n < 42
    ax.vlines(n[keep], 0, np.abs(hh[keep]), color=NAVY, lw=1.0)
    ax.vlines(n[~keep], 0, np.abs(hh[~keep]), color=ACCENT, lw=1.0, alpha=0.7)
    ax.axvspan(-0.5, 41.5, color=GREEN, alpha=0.08, lw=0)
    ax.text(20, 1.05, "keep: the channel lives here\n(within the prefix)", ha="center", fontsize=6.8, color=GREEN)
    ax.text(71, 0.35, "discard:\nnoise only", ha="center", fontsize=7, color=ACCENT)
    ax.set_xlabel("delay tap (IDFT of the pilot estimates)"); ax.set_ylabel("magnitude")
    ax.set_ylim(0, 1.3); ax.set_xlim(-1, 100)
    fig.tight_layout(); save(fig, "ch17_dft_est")


def lte_slot():
    """One 0.5 ms LTE slot (normal CP) at 30.72 MS/s: 7 symbols, prefixes of 160 and 144 samples."""
    fig, ax = plt.subplots(figsize=(W1 * 0.9, 0.95))
    x = 0
    for s in range(7):
        cp = 160 if s == 0 else 144
        ax.add_patch(Rectangle((x, 0), cp, 1, color=GREEN, alpha=0.7, lw=0))
        ax.add_patch(Rectangle((x + cp, 0), 2048, 1, color=NAVY, alpha=0.75 - 0.05 * (s % 2), lw=0))
        ax.text(x + cp + 1024, 0.5, f"symbol {s}", ha="center", va="center", fontsize=6.8, color="white")
        x += cp + 2048
    ax.text(80, 1.12, "CP 160", fontsize=6.5, color=GREEN, ha="center")
    ax.text(160 + 2048 + 72, 1.12, "144", fontsize=6.5, color=GREEN, ha="center")
    ax.set_xlim(0, x); ax.set_ylim(0, 1.35); ax.set_yticks([])
    ax.set_xticks([0, x / 2, x]); ax.set_xticklabels(["0", "0.25 ms", "0.5 ms = 15 360 samples"], fontsize=7)
    ax.grid(False)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); save(fig, "ch17_lte_slot")


def cpe_track():
    """Common phase error from Wiener phase noise, symbol by symbol, and its pilot estimate."""
    r = rng(71)
    N, ns = 256, 60
    beta = 0.004
    ph = np.cumsum(r.standard_normal(N * ns) * np.sqrt(2 * np.pi * beta / N))
    cpe = np.angle(np.exp(1j * ph).reshape(ns, N).mean(axis=1))
    est = cpe + r.standard_normal(ns) * 0.01
    t = np.arange(N * ns) / N
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(t, np.degrees(ph), color=GRAY, lw=0.6, label="oscillator phase")
    ax.step(np.arange(ns) + 0.5, np.degrees(cpe), where="mid", color=NAVY, lw=1.3, label="CPE per symbol")
    ax.plot(np.arange(ns) + 0.5, np.degrees(est), "o", ms=2.2, color=ACCENT, label="pilot estimate")
    ax.set_xlabel("OFDM symbol"); ax.set_ylabel("phase (degrees)")
    ax.legend(fontsize=6.3, loc="best")
    fig.tight_layout(); save(fig, "ch17_cpe_track")


def dfts_samples():
    """DFT-s-OFDM: every Q-th output sample is a data symbol; the rest is sinc interpolation."""
    r = rng(81)
    M, Q = 12, 8
    N = M * Q
    d = r.choice([-1.0, 1.0], M) + 1j * r.choice([-1.0, 1.0], M)
    X = np.zeros(N, complex); X[:M] = np.fft.fft(d) / np.sqrt(M)
    x = np.fft.ifft(X) * np.sqrt(N)
    n = np.arange(N)
    x = x * np.sqrt(Q)
    fig, ax = plt.subplots(figsize=(3.0, 1.7))
    ax.plot(n / Q, x.real, color=NAVY, lw=1.0, label="transmitted (real part)")
    ax.plot(np.arange(M), d.real, "o", color=ACCENT, ms=3.5, label="data symbols $d_m$")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("time (data-symbol periods)"); ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=6.3, loc="upper right", ncol=2); ax.set_ylim(-1.8, 2.8)
    fig.tight_layout(); save(fig, "ch17_dfts_samples")


def interleaver_map():
    """802.11a first interleaver permutation for 48 subcarriers (BPSK): code bit k -> position i."""
    Ncbps = 48
    k = np.arange(Ncbps)
    i = (Ncbps // 16) * (k % 16) + k // 16
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    ax.plot(k, i, "o", ms=2.6, color=NAVY)
    ax.plot(k[:6], i[:6], "-", color=ACCENT, lw=1.0)
    ax.set_xlabel("coded bit index $k$"); ax.set_ylabel("subcarrier slot $i$")
    ax.text(0.5, 38, "adjacent bits land\n3 slots apart", fontsize=6.8, color=ACCENT)
    fig.tight_layout(); save(fig, "ch17_interleaver_map")


NEW_FIGS = [lte_slot, cpe_track, dfts_samples, interleaver_map, dft_est, adsl_bandplan, guard_layout, delay_vs_cp, dvbt2_guards, pilot_limits, cp_overhead, dmrs_types, sco_ramp, coverage_gain, highway, orth_products, fft_bank, symbol_build, fft_cost, timeline, circulant, papr_crowd,
            hft_grid, waterfill_cartoon, spacing_span, ofdma_sched, sfn_map, mimo_sv, radar_map, wifi_rates,
            dfts_env, pa_curve, optical_ofdm, fbmc_proto, eq_cost, plc_mask, rayleigh_lanes, cfo_rotation,
            doppler_speed]


if __name__ == "__main__":
    import sys as _s
    fns = [fdm_vs_ofdm, ofdm_symbol, one_tap, cp_length, timing_window, ici, phase_noise,
           design_space, numerology, pilots, chest, papr, clipping, oob, coded, waterfill] + NEW_FIGS
    sel = _s.argv[1:]
    for fn in fns:
        if not sel or fn.__name__ in sel:
            fn()
