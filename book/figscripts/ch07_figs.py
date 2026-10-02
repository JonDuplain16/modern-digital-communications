"""Figures for Chapter 7: The Radio Transceiver and the Software-Defined Radio.

Run `python ch07_figs.py` to regenerate every figure; `python ch07_figs.py --numbers`
also prints the numbers quoted in the worked examples of the chapter.
"""
import sys
from figstyle import *
from scipy import signal as sps
import commlib as cl

K_DBM = -174.0                       # kT0 in dBm/Hz


# ----------------------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------------------
def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def ofdm_signal(n_sym=400, r=None, n_used=300, nfft=1024, ncp=72, lowpass=True, return_grid=False):
    cfg = cl.OFDMConfig(nfft=nfft, n_used=n_used, ncp=ncp)
    c = cl.get_constellation("64qam")
    g = c.modulate(cl.random_bits(6 * n_used * n_sym, r)).reshape(n_sym, n_used)
    x = cl.ofdm_modulate(g, cfg)
    if lowpass:
        x = np.convolve(x, sps.firwin(301, 1.15 * n_used / nfft), mode="same")   # tame sidelobes, zero delay
    s = np.sqrt(np.mean(np.abs(x) ** 2))
    if return_grid:
        return x / s, g, cfg
    return x / s


def psd_db(x, nper=2048, fs=1.0):
    f, P = sps.welch(x, fs=fs, nperseg=nper, return_onesided=False, window="blackmanharris",
                     detrend=False)
    return np.fft.fftshift(f), np.fft.fftshift(P)


def aclr_db(x, bw, offset, nper=4096):
    """Adjacent-channel leakage ratio (worst side), channel width bw, centre offset (cycles/sample)."""
    f, P = psd_db(x, nper)
    main = P[np.abs(f) < bw / 2].sum()
    up = P[np.abs(f - offset) < bw / 2].sum()
    lo = P[np.abs(f + offset) < bw / 2].sum()
    return db10(main / max(up, lo))


def pa_static(x, sat=1.0, p=2.0, ampm_deg=12.0):
    """Rapp AM/AM with Saleh-type AM/PM (phase in degrees reached at saturation drive)."""
    a = np.abs(x)
    g = 1 / (1 + (a / sat) ** (2 * p)) ** (1 / (2 * p))
    ph = np.deg2rad(ampm_deg) * 2 * (a / sat) ** 2 / (1 + (a / sat) ** 2)
    return x * g * np.exp(1j * ph)


H_IN = np.array([1.0, 0.5 + 0.2j, 0.2])          # Wiener-Hammerstein memory
H_IN = H_IN / H_IN.sum()
H_OUT = np.array([1.0, 0.25j])
H_OUT = H_OUT / H_OUT.sum()


def pa_memory(x):
    """A PA with memory: linear filter, static nonlinearity, linear filter (Wiener-Hammerstein)."""
    u = np.convolve(x, H_IN)[:len(x)]
    v = pa_static(u)
    return np.convolve(v, H_OUT)[:len(x)]


def mp_basis(u, K=7, M=3):
    """Memory-polynomial basis: u[n-m] |u[n-m]|^(k-1), k = 1..K (odd and even), m = 0..M."""
    cols = []
    for m in range(M + 1):
        um = np.concatenate([np.zeros(m, complex), u[:len(u) - m]])
        for k in range(1, K + 1):
            cols.append(um * np.abs(um) ** (k - 1))
    return np.stack(cols, axis=1)


def ila_dpd(x, pa, K=7, M=3, iters=5, G=1.0, lam=1e-6):
    """Indirect-learning DPD: fit a post-inverse y/G -> z (regularised least squares on
    column-normalised memory-polynomial basis), then copy it in front of the PA."""
    z = x.copy()
    for _ in range(iters):
        y = pa(z)
        Phi = mp_basis(y / G, K, M)
        sc = np.sqrt(np.mean(np.abs(Phi) ** 2, axis=0))
        A = Phi[200:] / sc
        c = np.linalg.solve(A.conj().T @ A + lam * len(A) * np.eye(A.shape[1]), A.conj().T @ z[200:]) / sc
        z = mp_basis(x, K, M) @ c
    return z, pa(z)


def cascade(stages):
    """stages: list of (name, gain_dB, NF_dB, IIP3_dBm). Returns cumulative lists."""
    G = 1.0
    F = 1.0
    inv_iip3 = 0.0
    out = []
    for i, (nm, g, nf, iip3) in enumerate(stages):
        f = 10 ** (nf / 10)
        F = F + (f - 1) / G if i else f
        if iip3 is not None:
            inv_iip3 += G / (10 ** (iip3 / 10))
        G *= 10 ** (g / 10)
        out.append((nm, db10(G), db10(F), -db10(inv_iip3) if inv_iip3 > 0 else np.inf))
    return out


# ----------------------------------------------------------------------------------------------
# existing figures (labels referenced elsewhere in the book)
# ----------------------------------------------------------------------------------------------
def fig_two_tone():
    N = 1 << 14
    n = np.arange(N)
    f1, f2 = 0.1, 0.11
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    a1, a3 = 1.0, -0.08
    x = 0.5 * (np.cos(2 * np.pi * f1 * n) + np.cos(2 * np.pi * f2 * n))
    y = a1 * x + a3 * x ** 3
    Y = np.abs(np.fft.rfft(y * np.blackman(N))) ** 2
    Y /= Y.max()
    f = np.fft.rfftfreq(N)
    ax[0].plot(f, 10 * np.log10(Y + 1e-16), color=NAVY, lw=0.8)
    ax[0].set_xlim(0.06, 0.15); ax[0].set_ylim(-90, 5)
    for fx, lab in [(2 * f1 - f2, "$2f_1-f_2$"), (2 * f2 - f1, "$2f_2-f_1$")]:
        ax[0].annotate(lab, xy=(fx, -40), xytext=(fx, -20), ha="center", fontsize=7,
                       arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].set_xlabel("cycles/sample"); ax[0].set_ylabel("dBc")
    ax[0].set_title("two-tone test: third-order products", fontsize=8.5)
    pin = np.linspace(-40, 10, 100)
    im3 = 3 * pin - 2 * 5                                  # IIP3 = 5 dBm by construction
    ax[1].plot(pin, pin, color=NAVY, label="fundamental (slope 1)")
    ax[1].plot(pin, im3, color=ACCENT, label="IM3 (slope 3)")
    comp = pin - 10 * np.log10(1 + 10 ** ((pin - 0) / 10))
    ax[1].plot(pin, comp, color=NAVY, ls=":", label="real output (compression)")
    ax[1].plot([5], [5], "ko", ms=4); ax[1].text(-6, 8, "IIP3 = OIP3\n(extrapolated)", fontsize=7)
    ax[1].set_xlim(-40, 12); ax[1].set_ylim(-90, 15)
    ax[1].set_xlabel("input power per tone (dBm)"); ax[1].set_ylabel("output (dBm)")
    ax[1].legend(fontsize=6.3, loc="lower right"); ax[1].set_title("third-order intercept", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_two_tone")


def fig_pa_regrowth():
    r = rng(7)
    x = ofdm_signal(300, r)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
    a = np.linspace(0, 2.5, 300)
    ax[0].plot(a, np.abs(cl.rapp_pa(a + 0j, sat=1.0, p=2.0)), color=NAVY, label="Rapp PA, p = 2")
    ax[0].plot(a, a, color=GRAY, ls=":", label="ideal linear")
    ax[0].set_xlabel("input amplitude"); ax[0].set_ylabel("output amplitude"); ax[0].legend(fontsize=6.5)
    ax[0].set_title("AM/AM characteristic", fontsize=8.5)
    for bo, c in [(12, GREEN), (6, ORANGE), (3, ACCENT)]:
        g = 10 ** (-bo / 20)
        y = cl.rapp_pa(x * g, sat=1.0, p=2.0) / g
        f, P = psd_db(y)
        ax[1].plot(f, 10 * np.log10(P / P.max()), color=c, lw=0.8, label=f"back-off {bo} dB")
    f, P = psd_db(x)
    ax[1].plot(f, 10 * np.log10(P / P.max()), color=NAVY, lw=0.8, label="input")
    ax[1].set_xlim(-0.5, 0.5); ax[1].set_ylim(-80, 5); ax[1].set_xlabel("cycles/sample")
    ax[1].set_ylabel("dB"); ax[1].legend(fontsize=6.3, loc="upper right")
    ax[1].set_title("OFDM spectral regrowth versus PA back-off", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_pa_regrowth")


DPD_IBO = 11.0      # input back-off used in the DPD figures (dB below saturation drive)


def _dpd_runs():
    r = rng(8)
    x = ofdm_signal(260, r, n_used=200)
    g = 10 ** (-DPD_IBO / 20)
    xin = x * g
    G = np.vdot(0.01 * xin, pa_memory(0.01 * xin)) / np.vdot(0.01 * xin, 0.01 * xin)   # small-signal gain
    y = pa_memory(xin)
    _, y_ml = ila_dpd(xin, pa_memory, K=7, M=0, G=G)       # memoryless polynomial DPD
    _, y_mp = ila_dpd(xin, pa_memory, K=7, M=3, G=G)       # memory polynomial DPD
    return xin, y, y_ml, y_mp, G


def fig_dpd():
    xin, y, y_ml, y_mp, G = _dpd_runs()
    bw, off = 200 / 1024 * 1.0, 1.2 * 200 / 1024
    fig, ax = plt.subplots(figsize=(W1 * 0.86, 2.6))
    for sig, lab, c, lw in [(G * xin, "ideal linear PA", NAVY, 0.9), (y, "PA alone", ACCENT, 0.8),
                            (y_ml, "memoryless polynomial DPD", ORANGE, 0.8),
                            (y_mp, "memory-polynomial DPD ($K=7$, $M=3$)", GREEN, 0.9)]:
        f, P = psd_db(sig, 4096)
        a = aclr_db(sig, bw, off)
        ax.plot(f, 10 * np.log10(P / P.max()), color=c, lw=lw, label=f"{lab}: ACLR {a:.0f} dB")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-85, 5); ax.set_xlabel("frequency (cycles/sample)"); ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=6.6, loc="upper right")
    ax.set_title(f"Digital predistortion of a PA with memory, {DPD_IBO:.0f} dB input back-off", fontsize=9)
    fig.tight_layout(); save(fig, "ch07_dpd")


def fig_dpd_amam():
    xin, y, y_ml, y_mp, G = _dpd_runs()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    sel = slice(5000, 5000 + 6000)
    a_in = np.abs(xin[sel])
    for yy, lab, c in [(y, "PA alone", ACCENT), (y_mp, "with memory-polynomial DPD", GREEN)]:
        ax[0].scatter(a_in, np.abs(yy[sel]) / abs(G), s=0.4, color=c, alpha=0.35, label=lab, rasterized=True)
        ax[1].scatter(a_in, np.rad2deg(np.angle(yy[sel] / xin[sel])), s=0.4, color=c, alpha=0.35,
                      rasterized=True)
    ax[0].plot([0, 1.2], [0, 1.2], color=NAVY, lw=0.8, ls="--", label="ideal")
    ax[0].set_xlim(0, 1.15); ax[0].set_ylim(0, 1.15)
    ax[0].set_xlabel("input amplitude $|x|$"); ax[0].set_ylabel("$|y|/G$")
    ax[0].set_title("AM/AM", fontsize=8.5)
    lg = ax[0].legend(fontsize=6.5, loc="upper left", markerscale=12)
    ax[1].axhline(np.rad2deg(np.angle(G)), color=NAVY, lw=0.8, ls="--")
    ax[1].set_xlim(0, 1.15); ax[1].set_ylim(-20, 25)
    ax[1].set_xlabel("input amplitude $|x|$"); ax[1].set_ylabel("phase of $y/x$ (degrees)")
    ax[1].set_title("AM/PM", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_dpd_amam")


def fig_phase_noise():
    r = rng(9)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    foff = np.logspace(2, 7, 300)
    L = 10 * np.log10((1e4 / foff) ** 2 * (1 + 1e5 / foff) + 10 ** (-16)) - 95
    ax[0].semilogx(foff, L, color=NAVY)
    ax[0].set_xlabel("offset from carrier (Hz)"); ax[0].set_ylabel("$\\mathcal{L}(f)$ (dBc/Hz)")
    ax[0].set_title("oscillator phase noise", fontsize=8.5)
    ax[0].text(3e2, -40, "flicker $1/f^3$", fontsize=7); ax[0].text(3e5, -118, "$1/f^2$", fontsize=7)
    ax[0].text(3e6, -150, "floor", fontsize=7)
    c = cl.get_constellation("64qam")
    s = c.modulate(cl.random_bits(6 * 4000, r))
    for a, sd, ttl in [(ax[1], 0.5, "0.5$^\\circ$ rms"), (ax[2], 3.0, "3$^\\circ$ rms")]:
        y = s * np.exp(1j * np.deg2rad(sd) * r.standard_normal(len(s)))
        y, _ = cl.awgn_esn0(y, 35, rng=r)
        a.scatter(y.real, y.imag, s=0.5, color=NAVY, alpha=0.4, rasterized=True)
        a.set_aspect("equal"); a.set_xlim(-1.4, 1.4); a.set_ylim(-1.4, 1.4)
        a.set_title(f"64-QAM, {ttl} phase jitter", fontsize=7.5); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, "ch07_phase_noise")


def fig_architectures():
    fig, ax = plt.subplots(4, 1, figsize=(W2, 4.1))

    def bump(a, c, w, h, col, lab=None, alpha=0.8):
        f = np.linspace(c - w, c + w, 50)
        a.fill_between(f, h * np.cos((f - c) / w * np.pi / 2) ** 2, color=col, alpha=alpha, label=lab)

    def lo(a, f, lab="LO"):
        a.vlines(f, 0, 0.9, color=ACCENT)
        a.text(f, 0.95, lab, ha="center", fontsize=7, color=ACCENT)
    # superhet
    a = ax[0]; bump(a, 2.0, 0.08, 1, NAVY, "wanted"); bump(a, 2.6, 0.08, 0.8, ORANGE, "image")
    lo(a, 2.3); bump(a, 0.3, 0.08, 1, NAVY); a.text(0.3, 1.1, "IF", ha="center", fontsize=7)
    a.plot([1.75, 1.8, 2.2, 2.25], [0, 1.2, 1.2, 0], color=GREEN, lw=0.9, label="RF image filter")
    a.set_title("superheterodyne: image removed by an RF filter, channel selected at a fixed IF", fontsize=8)
    # zero IF
    a = ax[1]; bump(a, 2.0, 0.08, 1, NAVY); lo(a, 2.0)
    bump(a, 0.0, 0.08, 1, NAVY); a.vlines(0, 0, 0.5, color=GRAY, lw=2)
    a.text(0.15, 0.55, "DC offset, 1/f noise, IM2;\nI/Q image = own mirror", fontsize=6.5)
    a.set_title("direct conversion (zero IF): LO at the carrier, I and Q straight to baseband", fontsize=8)
    # low IF
    a = ax[2]; bump(a, 2.0, 0.08, 1, NAVY); bump(a, 1.8, 0.08, 1.15, ORANGE, alpha=0.6); lo(a, 1.9)
    bump(a, 0.1, 0.08, 1, NAVY); bump(a, -0.1, 0.08, 0.25, ORANGE, alpha=0.6)
    a.text(0.25, 0.55, "adjacent channel's image lands\non the wanted: needs high IRR", fontsize=6.5)
    a.set_title("low IF: LO half a channel away; the image is a neighbour, rejected by I/Q balance", fontsize=8)
    # direct RF sampling
    a = ax[3]; bump(a, 2.0, 0.08, 1, NAVY)
    for k in range(1, 4):
        a.axvline(k * 0.75, color=GRAY, ls=":", lw=0.7)
    a.text(0.05, 1.0, "Nyquist zones of a 1.5 GS/s ADC", fontsize=6.5, color=GRAY)
    a.set_title("direct RF sampling: the ADC digitises the whole band; mixing and filtering are digital", fontsize=8)
    for a in ax:
        a.set_yticks([]); a.set_xlim(-0.3, 3.0); a.set_ylim(0, 1.35)
    ax[3].set_xlabel("frequency (GHz, schematic)")
    ax[0].legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.2, 1.02), ncol=3)
    fig.tight_layout(h_pad=0.4); save(fig, "ch07_architectures")


# ----------------------------------------------------------------------------------------------
# new figures
# ----------------------------------------------------------------------------------------------
def fig_zeroif_impairments():
    """Baseband spectrum of a zero-IF receiver before channel filtering, with all its fingerprints."""
    r = rng(21)
    fs, N = 20e6, 1 << 18
    t = np.arange(N) / fs
    sps_ = 10
    q16 = cl.get_constellation("16qam")
    h = cl.rrc_taps(0.25, sps_, 8)
    w = cl.shape(q16.modulate(cl.random_bits(4 * (N // sps_ + 20), r)), h, sps_)[:N]
    w = w / np.sqrt(np.mean(np.abs(w) ** 2))                           # wanted: 2 MS/s 16-QAM at 0 Hz
    # amplitude-varying blocker (OFDM-like noise band) at +6 MHz, 40 dB above the wanted
    b = r.standard_normal(N) + 1j * r.standard_normal(N)
    b = sps.lfilter(sps.firwin(255, 0.06), 1, b)
    b = b / np.sqrt(np.mean(np.abs(b) ** 2)) * 10 ** (40 / 20) * np.exp(2j * np.pi * 6e6 * t)
    x = w + b
    y = cl.iq_imbalance(x, 0.4, 2.0)                                    # IRR about 35 dB
    a2 = 10 ** (-86 / 20)
    y = y + a2 * np.abs(b) ** 2 * (1 + 0.3j)                            # IM2: blocker envelope at DC
    y = y + 10 ** (12 / 20) * (1 + 0.6j)                                # static DC offset (self-mixing)
    # flicker noise, corner about 300 kHz, plus white thermal noise 25 dB below the wanted PSD
    wn = (r.standard_normal(N) + 1j * r.standard_normal(N)) / np.sqrt(2)
    Wf = np.fft.fft(wn)
    f = np.fft.fftfreq(N, 1 / fs)
    shape = np.sqrt(1 + 3e5 / np.maximum(np.abs(f), fs / N))
    nz = np.fft.ifft(Wf * shape) * 10 ** (-12 / 20)
    y = y + nz
    nper = 8192
    ff, P = sps.welch(y, fs=fs, nperseg=nper, return_onesided=False, window="blackmanharris", detrend=False)
    ff, P = np.fft.fftshift(ff), np.fft.fftshift(P)
    _, Pw = sps.welch(w, fs=fs, nperseg=nper, return_onesided=False, window="blackmanharris", detrend=False)
    ref = np.max(Pw)
    PdB = 10 * np.log10(P / ref)
    fig, ax = plt.subplots(figsize=(W2, 2.7))
    ax.plot(ff / 1e6, PdB, color=NAVY, lw=0.7)
    ax.axvspan(-1.25, 1.25, color=GREEN, alpha=0.08)
    ax.text(-1.2, 49, "wanted channel\n(16-QAM, 2.5 MHz)", fontsize=6.8, color=GREEN)
    ax.annotate("blocker +40 dB\n(not yet filtered)", xy=(6, 40), xytext=(7.2, 46), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("I/Q image of blocker\n(IRR about 35 dB)", xy=(-6, 5), xytext=(-9.6, 22), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("DC offset\n(LO self-mixing)", xy=(0, 44), xytext=(2.0, 52), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("IM2: blocker envelope\nand 1/f noise at DC", xy=(0.15, 8), xytext=(2.4, 20), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlim(-10, 10); ax.set_ylim(-35, 60)
    ax.set_xlabel("baseband frequency (MHz)"); ax.set_ylabel("PSD relative to wanted (dB)")
    ax.set_title("What a zero-IF receiver delivers to its ADC (simulation)", fontsize=9)
    fig.tight_layout(); save(fig, "ch07_zeroif_impairments")


def irr_exact(g_db, phi_deg):
    g = 10 ** (np.asarray(g_db) / 20)
    ph = np.deg2rad(phi_deg)
    return db10((1 + 2 * g * np.cos(ph) + g ** 2) / (1 - 2 * g * np.cos(ph) + g ** 2))


def fig_irr():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    gd = np.linspace(0.0, 2.0, 300)
    pd = np.linspace(0.0, 10.0, 300)
    GG, PP = np.meshgrid(gd, pd)
    Z = irr_exact(GG + 1e-9, PP + 1e-9)
    cs = ax[0].contour(GG, PP, Z, levels=[20, 25, 30, 35, 40, 45, 50], colors=[NAVY], linewidths=0.8)
    ax[0].clabel(cs, fmt="%d dB", fontsize=6.5)
    ax[0].plot([0.1], [1.0], "o", color=ACCENT, ms=4)
    ax[0].annotate("0.1 dB, 1$^\\circ$: 39.6 dB", xy=(0.1, 1.0), xytext=(0.55, 0.4), fontsize=6.8,
                   color=ACCENT, arrowprops=dict(arrowstyle="->", lw=0.6, color=ACCENT))
    ax[0].set_xlabel("gain imbalance (dB)"); ax[0].set_ylabel("phase imbalance (degrees)")
    ax[0].set_title("image-rejection ratio", fontsize=8.5)
    ph = np.linspace(0.01, 10, 300)
    for gdb, c in [(0.0, NAVY), (0.1, GREEN), (0.5, ORANGE), (1.0, ACCENT)]:
        ax[1].plot(ph, irr_exact(gdb + 1e-9, ph), color=c, label=f"gain error {gdb} dB")
    for lvl, lab in [(25, "64-QAM, EVM 5%"), (35, "1024-QAM"), (60, "uncalibrated limit: about 25--40 dB")][:2]:
        ax[1].axhline(lvl, color=GRAY, ls=":", lw=0.7)
        ax[1].text(9.9, lvl + 1, lab, ha="right", fontsize=6.3, color=GRAY)
    ax[1].set_ylim(10, 70); ax[1].set_xlim(0, 10)
    ax[1].set_xlabel("phase imbalance (degrees)"); ax[1].set_ylabel("IRR (dB)")
    ax[1].legend(fontsize=6.3, loc="upper right"); ax[1].set_title("IRR versus phase error", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_irr")


def leeson_dbc(f, f0, Q, F_db, Ps_dbm, fc):
    return (K_DBM + F_db - 3 - Ps_dbm) + db10((1 + (f0 / (2 * Q * f)) ** 2) * (1 + fc / f))


def pll_components(f, N=60, fbw=400e3):
    """Reference/PFD noise multiplied by N inside the loop, VCO noise outside (2nd-order loop)."""
    ref = leeson_dbc(f, 40e6, 6e4, 5, 7, 2e3) + 20 * np.log10(N)
    pfd = np.full_like(f, -221 + 10 * np.log10(40e6) + 20 * np.log10(N))   # normalised PFD floor
    vco = leeson_dbc(f, 2.4e9, 8, 15, 0, 1e5)
    wn, z = 2 * np.pi * fbw / 2.06, 0.707
    s = 2j * np.pi * f
    H = (2 * z * wn * s + wn ** 2) / (s ** 2 + 2 * z * wn * s + wn ** 2)
    inb = db10(10 ** (ref / 10) + 10 ** (pfd / 10)) + db10(np.abs(H) ** 2)
    outb = vco + db10(np.abs(1 - H) ** 2)
    tot = db10(10 ** (inb / 10) + 10 ** (outb / 10))
    return ref, pfd, vco, inb, outb, tot


def rms_deg(f, L):
    return np.rad2deg(np.sqrt(2 * np.trapezoid(10 ** (L / 10), f)))


def fig_leeson():
    f = np.logspace(2, 8, 600)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    for Q, c in [(8, ACCENT), (30, ORANGE), (200, GREEN)]:
        ax[0].semilogx(f, leeson_dbc(f, 2.4e9, Q, 15, 0, 1e5), color=c, label=f"2.4 GHz VCO, $Q$ = {Q}")
    ax[0].semilogx(f, leeson_dbc(f, 40e6, 6e4, 5, 7, 2e3), color=NAVY, label="40 MHz crystal")
    ax[0].axvline(2.4e9 / 16, color=GRAY, ls=":", lw=0.6)
    ax[0].text(2.4e9 / 16 * 0.8, -60, "$f_0/2Q$\n($Q$=8)", fontsize=6.3, color=GRAY, ha="right")
    ax[0].set_ylim(-180, -30); ax[0].set_xlim(1e2, 1e8)
    ax[0].set_xlabel("offset frequency (Hz)"); ax[0].set_ylabel("$\\mathcal{L}(f)$ (dBc/Hz)")
    ax[0].legend(fontsize=6.2, loc="lower left"); ax[0].set_title("Leeson's model", fontsize=8.5)
    ref, pfd, vco, inb, outb, tot = pll_components(f)
    ax[1].semilogx(f, ref, color=NAVY, ls="--", lw=0.8, label="reference $\\times N$ ($+35.6$ dB)")
    ax[1].semilogx(f, pfd, color=PURPLE, ls="--", lw=0.8, label="PFD/charge-pump floor $\\times N$")
    ax[1].semilogx(f, vco, color=ACCENT, ls="--", lw=0.8, label="free-running VCO")
    ax[1].semilogx(f, tot, color=GREEN, lw=1.5, label="locked synthesiser output")
    ax[1].axvline(400e3, color=GRAY, ls=":", lw=0.6); ax[1].text(460e3, -55, "loop\nbandwidth", fontsize=6.3, color=GRAY)
    ax[1].set_ylim(-170, -40); ax[1].set_xlim(1e2, 1e8)
    ax[1].set_xlabel("offset frequency (Hz)"); ax[1].set_ylabel("dBc/Hz")
    ax[1].legend(fontsize=6.0, loc="lower left")
    ax[1].set_title("2.4 GHz PLL, $N$ = 60, $f_{\\mathrm{ref}}$ = 40 MHz", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_leeson")


def fig_reciprocal():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    f = np.linspace(-3e6, 3e6, 4001)
    off = 1e6
    Pb = -40.0
    _, _, _, _, _, tot = pll_components(np.maximum(np.abs(f - off), 50.0))
    L_poor = -100 - 20 * np.log10(np.maximum(np.abs(f - off), 1e3) / 1e6)        # -100 dBc/Hz at 1 MHz
    nf = 5.0
    floor = K_DBM + nf
    wanted = np.where(np.abs(f) < 100e3, -95 - 53, -300)
    ax[0].plot(f / 1e6, np.maximum(Pb + L_poor, floor), color=ACCENT, lw=0.9, label="blocker on noisy LO ($-100$ dBc/Hz at 1 MHz)")
    ax[0].plot(f / 1e6, np.maximum(Pb + tot, floor), color=GREEN, lw=0.9, label="blocker on a locked PLL ($-116$ dBc/Hz at 1 MHz)")
    ax[0].fill_between(f / 1e6, floor, np.maximum(wanted, floor), color=NAVY, alpha=0.6, label="wanted, $-95$ dBm in 200 kHz")
    ax[0].axhline(floor, color=GRAY, ls=":", lw=0.7)
    ax[0].text(-2.9, floor + 2, "thermal floor, NF 5 dB", fontsize=6.3, color=GRAY)
    ax[0].set_ylim(-175, -95); ax[0].set_xlim(-3, 3)
    ax[0].set_xlabel("frequency after down-conversion (MHz)"); ax[0].set_ylabel("PSD (dBm/Hz)")
    ax[0].set_title("$-40$ dBm blocker 1 MHz away", fontsize=8.5)
    ax[0].legend(fontsize=5.9, loc="upper left")
    # right: desensitisation vs blocker level
    B = 200e3
    pb = np.linspace(-80, -10, 200)
    nth = floor + 10 * np.log10(B)
    for Lv, c in [(-100, ACCENT), (-120, ORANGE), (-140, GREEN)]:
        nrm = pb + Lv + 10 * np.log10(B)
        ax[1].plot(pb, db10(10 ** (nrm / 10) + 10 ** (nth / 10)) - nth, color=c, label=f"$\\mathcal{{L}}$ = {Lv} dBc/Hz")
    ax[1].set_xlabel("blocker power (dBm)"); ax[1].set_ylabel("rise in noise floor (dB)")
    ax[1].set_title("desensitisation by reciprocal mixing", fontsize=8.5)
    ax[1].set_ylim(0, 40); ax[1].legend(fontsize=6.3, loc="upper left")
    fig.tight_layout(); save(fig, "ch07_reciprocal")


def fig_iq_correction():
    r = rng(31)
    N = 1 << 16
    sps_ = 8
    q16 = cl.get_constellation("16qam")
    h = cl.rrc_taps(0.25, sps_, 10)
    sym = q16.modulate(cl.random_bits(4 * (N // sps_ + 40), r))
    w = cl.shape(sym, h, sps_)[:N]
    w = w / np.sqrt(np.mean(np.abs(w) ** 2))
    n = np.arange(N)
    qp = cl.get_constellation("qpsk").modulate(cl.random_bits(2 * (N // sps_ + 40), r))
    b = cl.shape(qp, h, sps_)[:N]
    b = b / np.sqrt(np.mean(np.abs(b) ** 2)) * 10 ** (15 / 20) * np.exp(2j * np.pi * 0.27 * n)
    x = w + b
    y = cl.iq_imbalance(x, 1.5, 10.0) + (0.12 + 0.08j)
    y = y + 0.01 * (r.standard_normal(N) + 1j * r.standard_normal(N))
    # blind correction: remove the mean, then orthonormalise Q against I
    z = y - y.mean()
    I, Q = z.real, z.imag
    gs = np.mean(I * Q) / np.mean(I * I)
    gc = np.sqrt(np.mean(Q * Q) / np.mean(I * I) - gs ** 2)
    zc = I + 1j * (Q - gs * I) / gc
    print(f"  blind IQ estimate: g = {20*np.log10(np.hypot(gs, gc)):.2f} dB, phi = {np.rad2deg(np.arctan2(gs, gc)):.2f} deg")

    def demod(u):
        mf = np.convolve(u, h)[len(h) - 1::sps_][: N // sps_ - 20][10:]
        g_ = np.vdot(sym[10:10 + len(mf)], mf) / np.vdot(sym[10:10 + len(mf)], sym[10:10 + len(mf)])
        return mf / g_ / np.sqrt(sps_) * np.sqrt(sps_)
    fig = plt.figure(figsize=(W2, 2.7))
    gsp = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.9])
    for k, (u, ttl) in enumerate([(y, "before"), (zc, "after blind correction")]):
        a = fig.add_subplot(gsp[k])
        s = demod(u)
        a.scatter(s.real, s.imag, s=0.4, color=NAVY if k else ACCENT, alpha=0.35, rasterized=True)
        a.set_aspect("equal"); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"16-QAM at DC, {ttl}", fontsize=7.3)
    a = fig.add_subplot(gsp[2])
    for u, lab, c in [(y, "before: DC spike and image", ACCENT), (zc, "after", NAVY)]:
        f, P = psd_db(u, 2048)
        a.plot(f, 10 * np.log10(P / P.max()), color=c, lw=0.8, label=lab)
    a.annotate("image of the\n+0.27 carrier", xy=(-0.27, -23), xytext=(-0.48, -8), fontsize=6.5,
               arrowprops=dict(arrowstyle="->", lw=0.6))
    a.set_ylim(-75, 5); a.set_xlim(-0.5, 0.5); a.set_xlabel("cycles/sample"); a.set_ylabel("dB")
    a.legend(fontsize=6.3, loc="lower right"); a.set_title("1.5 dB, 10$^\\circ$ imbalance plus DC offset", fontsize=8)
    fig.tight_layout(); save(fig, "ch07_iq_correction")


def ofdm_envelope(r, n=400000, papr_clip_db=None):
    x = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    if papr_clip_db is not None:
        A = 10 ** (papr_clip_db / 20)
        a = np.abs(x)
        x = np.where(a > A, x / np.maximum(a, 1e-12) * A, x)
    return np.abs(x)


def eff_classA(v):
    return 0.5 * v ** 2


def eff_classB(v):
    return np.pi / 4 * v


def eff_doherty(v, alpha=0.5):
    return np.where(v <= alpha, np.pi / 4 * v / alpha, np.pi / 4 * v ** 2 / (v * (1 + alpha) - alpha))


def eff_et(v, vmin=0.25, eta_pa=0.7, eta_sup=0.85):
    return eta_pa * eta_sup * np.where(v >= vmin, 1.0, v / vmin)


def avg_eff(eff, v):
    """Average efficiency = E[Pout]/E[Pdc] for normalised amplitude samples v (peak = 1)."""
    pout = v ** 2
    return pout.mean() / (pout / np.maximum(eff(v), 1e-9)).mean()


def fig_pa_efficiency():
    r = rng(41)
    obo = np.linspace(0, 16, 300)
    v = 10 ** (-obo / 20)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw={"width_ratios": [1.45, 1]})
    curves = [("class A", eff_classA, GRAY), ("class B", eff_classB, NAVY),
              ("Doherty (6 dB)", eff_doherty, GREEN), ("Doherty 1:2 (9.5 dB)", lambda u: eff_doherty(u, 1 / 3), PURPLE),
              ("envelope tracking", eff_et, ORANGE)]
    for lab, e, c in curves:
        ax[0].plot(obo, 100 * e(v), color=c, label=lab)
    env = ofdm_envelope(r, papr_clip_db=8.0)
    vv = env / env.max()
    pb = -20 * np.log10(vv[vv > 1e-3])
    ax2 = ax[0].twinx()
    ax2.hist(pb, bins=120, range=(0, 30), density=True, color=ACCENT, alpha=0.18)
    ax2.set_yticks([]); ax2.set_ylim(0, 0.6); ax2.grid(False)
    ax2.spines["right"].set_visible(False)
    ax[0].text(9.5, 3, "OFDM envelope\nhistogram (PAPR 8 dB)", fontsize=6.3, color=ACCENT)
    ax[0].set_xlabel("output back-off from peak (dB)"); ax[0].set_ylabel("drain efficiency (%)")
    ax[0].set_xlim(0, 16); ax[0].set_ylim(0, 85); ax[0].legend(fontsize=6.0, loc="upper right")
    ax[0].set_title("ideal efficiency versus back-off", fontsize=8.5)
    names, vals = [], []
    for lab, e, c in curves:
        names.append(lab.replace(" (6 dB)", "").replace(" (9.5 dB)", "").replace("envelope tracking", "ET"))
        vals.append(100 * avg_eff(e, vv))
    cols = [c for _, _, c in curves]
    ax[1].barh(range(len(vals))[::-1], vals, color=cols, alpha=0.85)
    for i, val in enumerate(vals):
        ax[1].text(val + 1, len(vals) - 1 - i, f"{val:.0f}%", va="center", fontsize=7)
    ax[1].set_yticks(range(len(vals))[::-1]); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xlim(0, 75); ax[1].set_xlabel("average efficiency (%)")
    ax[1].set_title("with 8 dB PAPR OFDM", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_pa_efficiency")
    return dict(zip(names, vals))


def fig_doherty():
    v = np.linspace(0, 1, 400)
    a = 0.5
    Im = v
    Ip = np.where(v < a, 0, (v - a) / (1 - a))
    Vm = np.where(v < a, v / a, 1.0)
    Zm = np.where(v < a, 2.0, 1 / np.maximum(v, 1e-9) * 1.0)            # main load in units of Ropt
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].plot(v, Im, color=NAVY, label="main current")
    ax[0].plot(v, Ip, color=ACCENT, label="peaking current")
    ax[0].plot(v, Vm, color=GREEN, ls="--", label="main voltage swing")
    ax[0].plot(v, Zm / 2, color=ORANGE, ls=":", label="main load / $2R_{\\mathrm{opt}}$")
    ax[0].axvline(a, color=GRAY, ls=":", lw=0.6); ax[0].text(a - 0.02, 0.12, "peaking\nturns on", fontsize=6.5, color=GRAY, ha="right")
    ax[0].set_xlabel("normalised drive $v$"); ax[0].set_ylabel("normalised value")
    ax[0].legend(fontsize=6.2, loc="upper left"); ax[0].set_title("load modulation in a symmetric Doherty", fontsize=8.5)
    obo = np.linspace(0, 16, 300)
    vv = 10 ** (-obo / 20)
    ax[1].plot(obo, 100 * eff_classB(vv), color=NAVY, label="class B")
    ax[1].plot(obo, 100 * eff_doherty(vv, 0.5), color=GREEN, label="symmetric Doherty")
    ax[1].plot(obo, 100 * eff_doherty(vv, 1 / 3), color=PURPLE, label="asymmetric 1:2 Doherty")
    ax[1].axvline(6.02, color=GRAY, ls=":", lw=0.6); ax[1].axvline(9.54, color=GRAY, ls=":", lw=0.6)
    ax[1].set_xlabel("output back-off (dB)"); ax[1].set_ylabel("efficiency (%)")
    ax[1].set_ylim(0, 85); ax[1].legend(fontsize=6.3, loc="upper right"); ax[1].set_title("efficiency peaks at back-off", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_doherty")


def evm_after(x_t, grid, cfg, ncut=300):
    Y = cl.ofdm_demodulate(x_t, cfg)
    X = grid[: Y.shape[0]]
    Y, X = Y[2:-2], X[2:-2]
    g = np.vdot(X, Y) / np.vdot(X, X)
    return 100 * np.sqrt(np.sum(np.abs(Y - g * X) ** 2) / np.sum(np.abs(g * X) ** 2))


def fig_aclr_backoff():
    r = rng(51)
    x, grid, cfg = ofdm_signal(200, r, n_used=200, lowpass=True, return_grid=True)
    bw, off = 200 / 1024, 1.2 * 200 / 1024
    ibo = np.arange(2, 16.5, 1.0)
    acl, evm, obo, effb = [], [], [], []
    for b in ibo:
        g = 10 ** (-b / 20)
        y = pa_static(x * g)
        acl.append(aclr_db(y, bw, off))
        evm.append(evm_after(y, grid, cfg))
        pk = 1.0
        obo.append(-db10(np.mean(np.abs(y) ** 2) / pk))
        effb.append(100 * np.pi / 4 * np.mean(np.abs(y) ** 2) / np.mean(np.abs(y)))
    acl, evm, obo, effb = map(np.array, (acl, evm, obo, effb))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax[0].plot(obo, acl, "o-", color=NAVY, ms=2.5, label="ACLR")
    ax[0].axhline(30, color=ORANGE, ls="--", lw=0.8); ax[0].text(obo.max(), 31, "UE limit 30 dB", ha="right", fontsize=6.3, color=ORANGE)
    ax[0].axhline(45, color=ACCENT, ls="--", lw=0.8); ax[0].text(obo.max(), 46, "base station 45 dB", ha="right", fontsize=6.3, color=ACCENT)
    ax[0].set_xlabel("output back-off from saturation (dB)"); ax[0].set_ylabel("ACLR (dB)")
    ax[0].set_title("ACLR of 64-QAM OFDM through a PA", fontsize=8.5)
    a2 = ax[0].twinx(); a2.plot(obo, effb, color=GREEN, lw=0.9, ls=":"); a2.set_ylabel("class-B efficiency (%)", color=GREEN, fontsize=7.5)
    a2.tick_params(axis="y", colors=GREEN, labelsize=7); a2.grid(False); a2.spines["right"].set_visible(True)
    ax[1].semilogy(obo, evm, "o-", color=NAVY, ms=2.5)
    for lv, lab in [(8, "64-QAM limit 8%"), (3.5, "256-QAM limit 3.5%")]:
        ax[1].axhline(lv, color=ACCENT, ls="--", lw=0.8); ax[1].text(obo.max(), lv * 1.08, lab, ha="right", fontsize=6.3, color=ACCENT)
    ax[1].set_xlabel("output back-off from saturation (dB)"); ax[1].set_ylabel("EVM (%)")
    ax[1].set_title("in-band distortion (EVM)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_aclr_backoff")
    return obo, acl, evm, effb


def fig_level_diagram():
    """Receiver level plan for a narrowband (200 kHz) zero-IF receiver facing a -23 dBm blocker."""
    B = 200e3
    stages = [("antenna", 0, 0, None, 0), ("band filter", -2.5, 2.5, None, 0), ("LNA", 18, 1.5, -8, 0),
              ("mixer", 6, 9, 8, 0), ("BB filter\n+ TIA", 6, 18, 15, 22), ("VGA", 30, 15, 18, 20), ("ADC", 0, 0, None, 0)]
    nodes = [s[0] for s in stages]
    G, F = 0.0, 1.0
    sig, noi, blk = [], [], []
    Gl = 1.0
    blk_att = 0.0
    for i, (nm, g, nf, iip3, att) in enumerate(stages):
        if i > 0 and nm != "ADC":
            f = 10 ** (nf / 10)
            F = F + (f - 1) / Gl
            Gl *= 10 ** (g / 10)
            G += g
            blk_att += att
        sig.append(-99 + G)
        noi.append(K_DBM + 10 * np.log10(B) + 10 * np.log10(F) + G)
        blk.append(-23 + G - blk_att)
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    xs = np.arange(len(nodes))
    ax.plot(xs, blk, "s-", color=ACCENT, ms=3.5, label="blocker ($-23$ dBm at 3 MHz)")
    ax.plot(xs, sig, "o-", color=NAVY, ms=3.5, label="wanted ($-99$ dBm)")
    ax.plot(xs, noi, "^-", color=GRAY, ms=3.5, label="noise in 200 kHz (cascaded NF)")
    fs_adc = 4.0
    ax.hlines(fs_adc, len(nodes) - 1.4, len(nodes) - 0.6, color=GREEN, lw=2)
    ax.text(len(nodes) - 1.45, fs_adc + 3, "ADC full scale", fontsize=6.5, color=GREEN, ha="right")
    adc_nf = fs_adc - 74 - 10 * np.log10(61.44e6 / 2 / B)          # 12-bit ideal floor in 200 kHz (real-valued per branch)
    ax.hlines(adc_nf, len(nodes) - 1.4, len(nodes) - 0.6, color=GREEN, lw=2, ls="--")
    ax.text(len(nodes) - 1.45, adc_nf - 7, "12-bit quantisation\nnoise in 200 kHz", fontsize=6.5, color=GREEN, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(nodes, fontsize=7)
    ax.set_ylabel("level at stage output (dBm)"); ax.set_ylim(-130, 25)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Level diagram: the wanted signal, the noise and a blocker through the chain", fontsize=9)
    fig.tight_layout(); save(fig, "ch07_level_diagram")
    return nodes, sig, noi, blk, 10 * np.log10(F), adc_nf


def fig_cascade_tradeoff():
    Gl = np.linspace(0, 30, 121)
    B = 1e6
    nf, ii, sf = [], [], []
    for g in Gl:
        st = [("filter", -1.5, 1.5, None), ("LNA", g, 1.2, 0.0), ("mixer", 6, 10, 10), ("baseband", 30, 15, 10)]
        c = cascade(st)[-1]
        nf.append(c[2]); ii.append(c[3])
        sf.append(2 / 3 * (c[3] - (K_DBM + 10 * np.log10(B) + c[2])))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].plot(Gl, nf, color=NAVY, label="cascade NF (dB)")
    a2 = ax[0].twinx(); a2.plot(Gl, ii, color=ACCENT); a2.set_ylabel("cascade IIP3 (dBm)", color=ACCENT)
    a2.tick_params(axis="y", colors=ACCENT); a2.grid(False); a2.spines["right"].set_visible(True)
    ax[0].set_xlabel("LNA gain (dB)"); ax[0].set_ylabel("cascade NF (dB)", color=NAVY)
    ax[0].set_title("noise wants gain, linearity does not", fontsize=8.5)
    ax[1].plot(Gl, sf, color=GREEN)
    k = int(np.argmax(sf))
    ax[1].plot(Gl[k], sf[k], "o", color=GREEN, ms=4)
    ax[1].text(Gl[k], sf[k] - 4, f"best SFDR {sf[k]:.0f} dB\nat {Gl[k]:.0f} dB LNA gain", ha="center", fontsize=6.8)
    ax[1].set_xlabel("LNA gain (dB)"); ax[1].set_ylabel("SFDR in 1 MHz (dB)")
    ax[1].set_ylim(min(sf) - 2, max(sf) + 3)
    ax[1].set_title("spurious-free dynamic range", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_cascade_tradeoff")
    return Gl[k], sf[k]


def fig_dac_images():
    f = np.linspace(0, 3, 3000)
    zoh = np.abs(np.sinc(f))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for k, (L, ttl) in enumerate([(1, "DAC at the data rate"), (4, "after 4$\\times$ digital interpolation")]):
        a = ax[k]
        fsd = 1.0 * L
        ff = np.linspace(0, 3, 3000)
        env = 20 * np.log10(np.maximum(np.abs(np.sinc(ff / fsd)), 1e-6))
        a.plot(ff, env, color=GRAY, ls=":", lw=0.9, label="sinc (zero-order hold)")
        fsig = 0.2
        for m in range(0, 4):
            for sgn in (+1, -1):
                fc = m * fsd + sgn * fsig
                if 0 < fc < 3:
                    lvl = 20 * np.log10(abs(np.sinc(fc / fsd)))
                    c = NAVY if (m == 0) else ACCENT
                    a.vlines(fc, -80, lvl, color=c, lw=2)
        a.set_ylim(-60, 3); a.set_xlim(0, 3)
        a.set_xlabel("frequency / data rate"); a.set_ylabel("dB")
        a.set_title(ttl, fontsize=8.5)
        if L == 1:
            a.plot([0, 0.3, 0.8, 3], [0, 0, -60, -60], color=GREEN, lw=0.9, label="steep analog filter needed")
        else:
            a.plot([0, 0.3, 2.5, 3], [0, 0, -60, -60], color=GREEN, lw=0.9, label="gentle filter suffices")
        a.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(); save(fig, "ch07_dac_images")


def fig_cfr():
    r = rng(61)
    x, grid, cfg = ofdm_signal(300, r, n_used=300, lowpass=False, return_grid=True)
    bw = 300 / 1024
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    pp = np.linspace(0, 12, 200)

    def ccdf(u):
        p = db10(np.abs(u) ** 2 / np.mean(np.abs(u) ** 2))
        return np.array([np.mean(p > q) for q in pp])
    variants = [("original", x, NAVY)]
    for clip_db, iters, c in [(6.0, 1, ORANGE), (6.0, 4, GREEN)]:
        u = x.copy()
        for _ in range(iters):
            A = 10 ** (clip_db / 20) * np.sqrt(np.mean(np.abs(u) ** 2))
            a = np.abs(u)
            u = np.where(a > A, u / np.maximum(a, 1e-12) * A, u)
            U = np.fft.fft(u.reshape(-1, cfg.sym_len)[:, cfg.ncp:], axis=1)
            mask = np.zeros(cfg.nfft, bool); mask[cfg.active] = True
            U[:, ~mask] = 0
            sy = np.fft.ifft(U, axis=1)
            u = np.concatenate([sy[:, -cfg.ncp:], sy], axis=1).reshape(-1)
        e = evm_after(u, grid, cfg)
        variants.append((f"clip at 6 dB + filter, {iters} iter. (EVM {e:.1f}%)", u, c))
    for lab, u, c in variants:
        ax[0].semilogy(pp, np.maximum(ccdf(u), 1e-5), color=c, label=lab)
    ax[0].set_ylim(1e-4, 1); ax[0].set_xlabel("instantaneous-to-average power (dB)"); ax[0].set_ylabel("CCDF")
    ax[0].legend(fontsize=5.9, loc="lower left"); ax[0].set_title("crest-factor reduction", fontsize=8.5)
    def symspec(u):
        U = np.fft.fft(u.reshape(-1, cfg.sym_len)[:, cfg.ncp:], axis=1)
        P = np.fft.fftshift(np.mean(np.abs(U) ** 2, axis=0))
        return np.fft.fftshift(np.fft.fftfreq(cfg.nfft)), 10 * np.log10(P / P.max() + 1e-12)
    for lab, u, c in variants:
        f, P = symspec(u)
        ax[1].plot(f, P, color=c, lw=0.8)
    u = x.copy(); A = 10 ** (6 / 20); a = np.abs(u); u = np.where(a > A, u / a * A, u)
    f, P = symspec(u)
    ax[1].plot(f, P, color=ACCENT, lw=0.7, ls="--", label="clipping without filtering")
    ax[1].set_xlim(-0.5, 0.5); ax[1].set_ylim(-70, 5); ax[1].set_xlabel("cycles/sample"); ax[1].set_ylabel("dB")
    ax[1].legend(fontsize=6.0, loc="upper right"); ax[1].set_title("filtering keeps the spectrum clean", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_cfr")


def fig_freq_accuracy():
    fc = np.logspace(np.log10(70e6), np.log10(6e9), 200)
    fig, ax = plt.subplots(figsize=(W1 * 0.86, 2.6))
    for ppm, lab, c in [(20, "XO, $\\pm$20 ppm", ACCENT), (2, "TCXO, $\\pm$2 ppm (B200)", ORANGE),
                        (0.05, "OCXO / NR base station, $\\pm$0.05 ppm", GREEN), (1e-3, "GPSDO (locked), $\\approx$1 ppb", NAVY)]:
        ax.loglog(fc / 1e9, fc * ppm * 1e-6, color=c, label=lab)
    for scs, lab in [(15e3, "15 kHz subcarrier"), (312.5e3, "Wi-Fi 312.5 kHz subcarrier")]:
        ax.axhline(scs, color=GRAY, ls=":", lw=0.7)
        ax.text(0.075, scs * 1.15, lab, fontsize=6.3, color=GRAY)
    ax.plot([2.4], [4.8e3], "o", color=ORANGE, ms=4)
    ax.annotate("2 ppm at 2.4 GHz = 4.8 kHz", xy=(2.4, 4.8e3), xytext=(0.35, 3e4), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("carrier frequency (GHz)"); ax.set_ylabel("worst-case frequency error (Hz)")
    ax.set_xlim(0.07, 6); ax.set_ylim(0.05, 3e5); ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("Reference accuracy becomes carrier frequency offset", fontsize=9)
    fig.tight_layout(); save(fig, "ch07_freq_accuracy")


# ----------------------------------------------------------------------------------------------
# ==============================================================================================
# Second-edition concept illustrations and extra data figures
# ==============================================================================================
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

SKY = "#2E86C1"


def _arrow(ax, p, q, color=NAVY, lw=1.0, style="-|>", ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def fig_by_numbers():
    tiles = [("$-174$ dBm/Hz", "thermal noise at room\ntemperature: the floor"),
             ("$\\approx$100 dB", "between the weakest and strongest\nsignals a phone must handle"),
             ("$\\approx$100", "acoustic filters in a\nflagship 5G phone"),
             ("$\\pm$2 ppm", "a B200's TCXO: 4.8 kHz\nof error at 2.4 GHz"),
             ("61.44 MS/s", "the AD9364's top\nsample rate"),
             ("78.5%", "the best a class-B\namplifier can ever do"),
             ("\\$20", "an RTL-SDR dongle\n(2012): SDR for everyone"),
             ("1936", "Doherty's amplifier; inside\nmost base stations today")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, SKY, ACCENT, ORANGE, GREEN, PURPLE, NAVY, GRAY]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.64, big, ha="center", va="center", fontsize=12.5 if len(big) < 11 else 10.5,
                color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=6.2, color="#333333", linespacing=1.1)
    save(fig, "ch07_by_numbers")


def fig_spectrum_mic():
    """What the antenna 'hears': a crowded spectrum with a whisper among shouts."""
    r = rng(3)
    f = np.linspace(0, 3000, 6000)                     # MHz
    floor = -105 + 1.5 * r.standard_normal(f.size)
    P = 10 ** (floor / 10)
    sigs = [(98, 4, -35, "FM broadcast"), (600, 6, -55, "TV"), (850, 10, -48, "cellular\ndownlink"),
            (1575.42, 2, -128, "GPS\n(below noise)"), (1960, 12, -52, "cellular"), (2440, 20, -40, "Wi-Fi"),
            (2140, 5, -25, "nearby\nbase station"), (1200, 0.4, -100, "your faint\nsignal")]
    for fc, bw, lvl, _ in sigs:
        m = np.abs(f - fc) < bw / 2
        P[m] += 10 ** (lvl / 10)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(f, 10 * np.log10(P), color=NAVY, lw=0.6)
    for fc, bw, lvl, nm in sigs:
        col = ACCENT if "faint" in nm else (GREEN if "GPS" in nm else GRAY)
        y = max(lvl, -104) + 4
        ax.text(fc, y, nm, ha="center", va="bottom", fontsize=6.3, color=col, linespacing=1.0)
    ax.annotate("", xy=(2700, -25), xytext=(2700, -100), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
    ax.text(2730, -62, "75 dB\nbetween\nshout and\nwhisper", fontsize=6.5, color=ACCENT, va="center")
    ax.set_xlim(0, 3000); ax.set_ylim(-112, -10)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("power at antenna (dBm)")
    fig.tight_layout(); save(fig, "ch07_spectrum_mic")


def fig_whisper_chain():
    """Who adds the noise? Friis contributions of the worked line-up."""
    names = ["switch +\nbalun loss", "LNA", "mixer +\nTIA", "baseband\nfilter + VGA"]
    contrib = [0.778, 0.734, 0.160, 0.056]            # (F_k - 1)/G_before, from the worked example
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    cols = [GRAY, NAVY, ORANGE, GREEN]
    ax.bar(range(4), contrib, color=cols, width=0.62)
    for i, c in enumerate(contrib):
        ax.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("added noise factor, $(F_k-1)/G_{\\rm before}$", fontsize=7.5)
    ax.set_ylim(0, 0.95)
    ax.set_title("total $F=2.73$ (NF 4.36 dB)", fontsize=8)
    fig.tight_layout(); save(fig, "ch07_whisper_chain")


def fig_mixer_translator():
    """A mixer moves a signal: RF at f_RF times LO gives sum and difference."""
    fig, ax = plt.subplots(3, 1, figsize=(W1, 2.6), sharex=True)
    def blob(a, fc, h, col, lab, w=40):
        x = np.linspace(fc - w, fc + w, 100)
        a.fill_between(x, 0, h * np.cos(np.pi * (x - fc) / (2 * w)) ** 2, color=col, alpha=0.75, lw=0)
        a.text(fc, h + 0.08, lab, ha="center", fontsize=7, color=col)
    blob(ax[0], 900, 1, NAVY, "RF signal at 900 MHz")
    ax[1].vlines(830, 0, 1, color=ORANGE, lw=2); ax[1].text(830, 1.08, "LO at 830 MHz", ha="center", fontsize=7, color=ORANGE)
    blob(ax[2], 70, 0.5, GREEN, "")
    ax[2].text(130, 0.35, "difference: 70 MHz (the IF we keep)", fontsize=7, color=GREEN)
    blob(ax[2], 1730, 0.5, GRAY, "sum: 1730 MHz (filtered away)")
    for a, t in zip(ax, ["input", "local oscillator", "mixer output"]):
        a.set_ylim(0, 1.45); a.set_yticks([]); a.grid(False)
        a.spines["left"].set_visible(False)
        a.set_ylabel(t, rotation=0, ha="right", va="center", fontsize=7.5)
    ax[2].set_xlim(-10, 1900); ax[2].set_xlabel("frequency (MHz)")
    fig.tight_layout(h_pad=0.2); save(fig, "ch07_mixer_translator")


def fig_image_mirror():
    """The image: a station mirrored about the LO lands on the same IF."""
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    lo, fif = 1000, 100
    def blob(fc, h, col, w=18):
        x = np.linspace(fc - w, fc + w, 80)
        ax.fill_between(x, 0, h * np.cos(np.pi * (x - fc) / (2 * w)) ** 2, color=col, alpha=0.8, lw=0)
    blob(lo + fif, 0.6, NAVY); blob(lo - fif, 0.9, ACCENT); blob(fif, 0.6, NAVY); blob(fif, 0.9, ACCENT, w=10)
    ax.vlines(lo, 0, 1.2, color=ORANGE, lw=2)
    ax.text(lo, 1.27, "LO", ha="center", fontsize=7.5, color=ORANGE)
    ax.axvline(lo, color=ORANGE, ls=":", lw=0.8)
    ax.text(lo + fif, 0.68, "wanted\n$f_{\\rm LO}+f_{\\rm IF}$", ha="center", fontsize=7, color=NAVY)
    ax.text(lo - fif, 0.97, "image (mirror)\n$f_{\\rm LO}-f_{\\rm IF}$", ha="center", fontsize=7, color=ACCENT)
    ax.text(fif, 1.0, "both land\non the IF", ha="center", fontsize=7, color="#333333")
    _arrow(ax, (lo + fif - 10, 0.35), (fif + 25, 0.35), NAVY, 0.8)
    _arrow(ax, (lo - fif - 10, 0.2), (fif + 25, 0.2), ACCENT, 0.8)
    ax.annotate("", xy=(lo - fif, -0.12), xytext=(lo + fif, -0.12), arrowprops=dict(arrowstyle="<->", lw=0.8, color=GRAY))
    ax.text(lo, -0.27, "$2f_{\\rm IF}$ apart: the preselector must tell them apart", ha="center", fontsize=6.8, color=GRAY)
    ax.set_xlim(0, 1250); ax.set_ylim(-0.35, 1.45); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.set_xlabel("frequency (MHz)")
    fig.tight_layout(); save(fig, "ch07_image_mirror")


def fig_harmonic_mixing():
    """A switching mixer responds at odd LO harmonics."""
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    n = np.array([1, 3, 5, 7, 9])
    g = 20 * np.log10(1 / n)
    ax.bar(n, g - g.min() + 3, bottom=g.min() - 3, color=[NAVY] + [ACCENT] * 4, width=0.9)
    for k, v in zip(n, g):
        ax.text(k, v + 0.6, f"{v:.1f} dB", ha="center", fontsize=7)
    ax.set_xticks(n); ax.set_xticklabels([f"{k}$f_{{\\rm LO}}$" for k in n], fontsize=7.5)
    ax.set_ylabel("relative conversion gain (dB)"); ax.set_ylim(-23, 3)
    ax.set_title("tuned to 100 MHz, also hears\n300, 500, 700 MHz", fontsize=7.5)
    fig.tight_layout(); save(fig, "ch07_harmonic_mixing")


def fig_sfdr_window():
    """The dynamic-range window of the worked line-up (1 MHz, NF 4.4 dB, IIP3 -19 dBm)."""
    nf, iip3, B = 4.36, -19.0, 1e6
    pn = -174 + 10 * np.log10(B) + nf
    pmax = (2 * iip3 + pn) / 3
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.add_patch(Rectangle((0.3, pn), 0.9, pmax - pn, fc=GREEN, alpha=0.18, ec=GREEN))
    for y, t, c in [(iip3, f"IIP3 = {iip3:.0f} dBm (a fiction)", ACCENT), (pmax, f"largest clean interferer {pmax:.1f} dBm", ORANGE),
                    (pn, f"noise floor {pn:.1f} dBm", NAVY)]:
        ax.axhline(y, color=c, lw=1.1, ls="--" if c == ACCENT else "-")
        ax.text(1.3, y + 1.5, t, va="bottom", fontsize=6.6, color=c)
    ax.annotate("", xy=(0.75, pmax), xytext=(0.75, pn), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(0.75, (pn + pmax) / 2, f"SFDR\n{pmax - pn:.0f} dB", ha="center", va="center", fontsize=8, color=GREEN, weight="bold")
    ax.set_xlim(0, 3.2); ax.set_ylim(-120, -10); ax.set_xticks([])
    ax.set_ylabel("input-referred power (dBm)"); ax.grid(False)
    fig.tight_layout(); save(fig, "ch07_sfdr_window")


def fig_desense():
    """Small-signal gain in the presence of a large blocker, cubic model."""
    # y = a1 x + a3 x^3 ; incremental gain for weak signal with blocker amplitude B: a1 + 3/2 a3 B^2
    a1, a3 = 1.0, -1.0
    A1db2 = 0.145 * abs(a1 / a3)                  # P1dB amplitude^2
    pb = np.linspace(-25, 2, 300)                 # blocker power re P1dB, dB
    B2 = A1db2 * 10 ** (pb / 10)
    g_small = 20 * np.log10(np.abs(a1 + 1.5 * a3 * B2))
    g_large = 20 * np.log10(np.abs(a1 + 0.75 * a3 * B2))
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(pb, g_large, color=NAVY, label="gain for the blocker")
    ax.plot(pb, g_small, color=ACCENT, label="gain for a weak signal")
    ax.axvline(0, color=GRAY, ls=":", lw=0.8); ax.text(0.3, -5.5, "blocker at\nP1dB", fontsize=6.5, color=GRAY)
    ax.set_xlabel("blocker power re P1dB (dB)"); ax.set_ylabel("gain change (dB)")
    ax.set_ylim(-7, 0.5); ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch07_desense")
    return np.interp(0, pb, g_small)


def fig_duplexer():
    """Band-3 FDD duplexer: TX and RX filters and the leakage that slips through."""
    f = np.linspace(1650, 1950, 3000)
    def bpf(f0, bw, n=8, floor=-62):
        x = 2 * (f - f0) / bw
        h = -10 * np.log10(1 + x ** (2 * n))
        return np.maximum(h, floor) - 1.5
    tx = bpf(1747.5, 85); rx = bpf(1842.5, 85)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot(f, tx, color=ORANGE, label="transmit filter (antenna $\\leftarrow$ PA)")
    ax.plot(f, rx, color=NAVY, label="receive filter (antenna $\\rightarrow$ LNA)")
    ax.axvspan(1710, 1785, color=ORANGE, alpha=0.07); ax.axvspan(1805, 1880, color=NAVY, alpha=0.07)
    ax.text(1747.5, 3, "uplink 1710–1785", ha="center", fontsize=7, color=ORANGE)
    ax.text(1842.5, 3, "downlink 1805–1880", ha="center", fontsize=7, color=NAVY)
    ax.annotate("our own +23 dBm transmitter,\n55 dB down: $-32$ dBm at the LNA", xy=(1760, -55), xytext=(1660, -35),
                fontsize=6.6, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.set_ylim(-68, 9); ax.set_xlim(1650, 1950)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("transmission (dB)")
    ax.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout(); save(fig, "ch07_duplexer")


def fig_adc_ladder():
    """The 'how many ADC bits' worked example as a staircase."""
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    items = [("needed in channel", 98, NAVY), ("processing gain\n(61.44 MHz $\\to$ 200 kHz)", -24.9, GREEN),
             ("full-band SNR\nwithout filter", 73.1, ACCENT), ("3rd-order filter\nat 3 MHz", -29, GREEN),
             ("full-band SNR\nwith filter", 44.1, ORANGE)]
    x = 0; base = 0
    for i, (nm, v, c) in enumerate(items):
        if i in (0, 2, 4):
            ax.bar(i, v, color=c, width=0.6); ax.text(i, v + 2, f"{v:.0f} dB\n= {(v - 1.76) / 6.02:.1f} bits" if i else f"{v:.0f} dB",
                                                      ha="center", fontsize=7)
            base = v
        else:
            ax.bar(i, v, bottom=base, color=c, width=0.6, alpha=0.7)
            ax.text(i, base + 2, f"{v:+.0f} dB", ha="center", fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels([t for t, _, _ in items], fontsize=6.6)
    ax.set_ylabel("dynamic range (dB)"); ax.set_ylim(0, 118)
    fig.tight_layout(); save(fig, "ch07_adc_ladder")


def fig_crystal_tempco():
    """AT-cut crystal frequency-temperature curve (illustrative) and what TCXO/OCXO do with it."""
    T = np.linspace(-40, 95, 400)
    x = T - 25
    xo = 1.0e-4 * x ** 3 - 0.08 * x                     # ppm, illustrative cubic
    tcxo = 0.4 * np.sin(x / 9) * np.exp(-((x) / 120) ** 2)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(T, xo, color=GRAY, label="bare crystal (XO)")
    ax.plot(T, tcxo, color=NAVY, label="TCXO (compensated)")
    tp = 25 + np.sqrt(0.08 / 3e-4)
    ax.plot([tp], [1e-4 * (tp - 25) ** 3 - 0.08 * (tp - 25)], "o", color=ACCENT, ms=5)
    ax.annotate("OCXO: hold it at\nthe turning point", xy=(tp, -0.87), xytext=(-35, -9), fontsize=6.6, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("frequency error (ppm)")
    ax.legend(fontsize=6.4, loc="upper left"); ax.set_ylim(-14, 18)
    fig.tight_layout(); save(fig, "ch07_crystal_tempco")


def fig_fracn_dither():
    """Fractional-N: a MASH 1-1-1 dithers the divider so its average is N + alpha."""
    N, alpha, n = 60, 0.3, 120
    # MASH 1-1-1
    e1 = e2 = e3 = 0.0; c1p = c2p = 0; c2pp = 0; c3p = c3pp = 0
    seq = []
    y1 = y2 = 0
    acc1 = acc2 = acc3 = 0.0
    c2_hist = [0, 0]; c3_hist = [0, 0, 0]
    out = []
    for k in range(n):
        acc1 += alpha; c1 = int(acc1 >= 1); acc1 -= c1
        acc2 += acc1; c2 = int(acc2 >= 1); acc2 -= c2
        acc3 += acc2; c3 = int(acc3 >= 1); acc3 -= c3
        c2_hist = [c2] + c2_hist[:1]; c3_hist = [c3] + c3_hist[:2]
        d = c1 + (c2_hist[0] - c2_hist[1]) + (c3_hist[0] - 2 * c3_hist[1] + c3_hist[2])
        out.append(N + d)
    out = np.array(out)
    per = N + (np.floor((np.arange(n) + 1) * alpha) - np.floor(np.arange(n) * alpha))
    m = 60
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True)
    for ax, seqv, c, ttl in [(axs[0], per, ORANGE, "periodic $N$/$N{+}1$ switching: a regular pattern $\\to$ fractional spurs"),
                             (axs[1], out, NAVY, "sigma-delta (MASH 1-1-1): a busy, random-looking pattern $\\to$ noise pushed to high offsets")]:
        ax.step(np.arange(m), seqv[:m], where="mid", color=c, lw=1.0)
        ax.plot(np.arange(m), np.cumsum(seqv[:m]) / (np.arange(m) + 1), color=ACCENT, lw=1.3)
        ax.axhline(N + alpha, color=GREEN, ls="--", lw=0.9)
        ax.set_title(ttl, fontsize=7.5, loc="left"); ax.set_ylabel("ratio", fontsize=7.5)
        ax.set_yticks([57, 60, 63])
    axs[0].set_ylim(59.4, 61.6); axs[1].set_ylim(56.5, 64.5)
    axs[1].text(m + 0.5, N + alpha, "$N+\\alpha=60.3$", va="center", fontsize=7, color=GREEN)
    axs[0].text(m + 0.5, 61.2, "running\naverage", va="center", fontsize=6.5, color=ACCENT)
    axs[1].set_xlabel("reference cycle"); axs[1].set_xlim(0, m + 8)
    fig.tight_layout(); save(fig, "ch07_fracn_dither")
    return out.mean()


def fig_wobbly_lo():
    """Phase noise as a wobbly hand: phase wander in time and the skirt it makes in frequency."""
    r = rng(5)
    fs, n = 1.0, 1 << 16
    ph = np.cumsum(r.standard_normal(n)) * 0.012          # Wiener phase
    ph -= np.convolve(ph, np.ones(4001) / 4001, mode="same")  # loop removes the slow drift
    x = np.exp(1j * (2 * np.pi * 0.0 * np.arange(n) + ph))
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    a.plot(np.arange(4000), np.rad2deg(ph[20000:24000]), color=NAVY, lw=0.8)
    a.set_xlabel("time (samples)"); a.set_ylabel("phase error (degrees)")
    a.set_title("a wobbly hand on the tuning knob", fontsize=8)
    b = ax[1]
    f, P = sps.welch(x, fs=fs, nperseg=4096, return_onesided=False, window="blackmanharris", detrend=False)
    f = np.fft.fftshift(f); P = np.fft.fftshift(P)
    b.plot(f * 1e3, db10(P / P.max()), color=ACCENT, lw=1.0, label="real LO")
    b.vlines(0, -90, 0, color=NAVY, lw=1.8, label="ideal LO")
    b.set_xlim(-30, 30); b.set_ylim(-75, 3)
    b.set_xlabel("offset (milli-cycles/sample)"); b.set_ylabel("relative PSD (dB)")
    b.legend(fontsize=6.5, loc="upper right")
    b.set_title("a line becomes a line with skirts", fontsize=8)
    fig.tight_layout(); save(fig, "ch07_wobbly_lo")


def fig_image_tone():
    """A tone at +f and its I/Q-imbalance image at -f for several IRRs."""
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for irr, c in [(25, ACCENT), (40, ORANGE), (60, GREEN)]:
        ax.vlines(-1.0, -100, -irr, color=c, lw=3, alpha=0.85)
        ax.text(-1.15, -irr, f"IRR {irr} dB", ha="right", va="center", fontsize=6.8, color=c)
    ax.vlines(1.0, -100, 0, color=NAVY, lw=3)
    ax.text(1.0, 3, "tone at $+f$", ha="center", fontsize=7, color=NAVY)
    ax.text(-1.0, -13, "image at $-f$", ha="center", fontsize=7, color="#333333")
    ax.axvline(0, color=GRAY, ls=":", lw=0.8); ax.text(0.05, -95, "DC (LO)", fontsize=6.5, color=GRAY)
    ax.set_xlim(-2.4, 2); ax.set_ylim(-100, 12); ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["$-f$", "0", "$+f$"])
    ax.set_ylabel("power (dBc)"); ax.set_xlabel("baseband frequency")
    fig.tight_layout(); save(fig, "ch07_image_tone")


def fig_dc_notch():
    """Running-mean DC remover: notch width versus alpha."""
    f = np.logspace(-6, np.log10(0.5), 2000)
    z = np.exp(2j * np.pi * f)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for a, c in [(1e-2, ACCENT), (1e-3, NAVY), (1e-4, GREEN)]:
        H = (1 - 1 / z) * (1 - a) / (1 - (1 - a) / z)          # z = y - m, m[n]=m[n-1]+a(y[n]-m[n-1])
        ax.semilogx(f, 20 * np.log10(np.abs(H) + 1e-9), color=c, label=f"$\\alpha={a:g}$")
        ax.axvline(a / (2 * np.pi), color=c, ls=":", lw=0.7)
    ax.set_xlabel("frequency (cycles/sample)"); ax.set_ylabel("response (dB)")
    ax.set_ylim(-40, 3); ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("notch width $\\approx\\alpha f_s/2\\pi$ (dotted)", fontsize=7.5)
    fig.tight_layout(); save(fig, "ch07_dc_notch")


def fig_class_waveforms():
    """Conduction angle: drain current of class A, AB, B and C stages."""
    th = np.linspace(0, 4 * np.pi, 800)
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.7), sharey=True)
    for a, (nm, bias, c) in zip(ax, [("A: 360°", 1.0, NAVY), ("AB: ~240°", 0.5, SKY), ("B: 180°", 0.0, GREEN), ("C: ~120°", -0.5, ACCENT)]):
        i = np.maximum(bias + np.cos(th), 0)
        a.plot(th, i, color=c); a.fill_between(th, 0, i, color=c, alpha=0.15)
        a.set_title(nm, fontsize=8); a.set_xticks([]); a.grid(False)
        a.axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("drain current")
    ax[0].set_yticks([])
    fig.tight_layout(w_pad=0.4); save(fig, "ch07_class_waveforms")


def fig_backoff_shout():
    """OFDM envelope against the PA's saturation level: peaks that 'crack the voice'."""
    x = ofdm_signal(n_sym=6, r=rng(11))
    env = np.abs(x[2000:3200])
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    t = np.arange(env.size)
    for sat, c, lab in [(10 ** (5 / 20), ACCENT, "saturation 5 dB above average (too loud)"),
                        (10 ** (10 / 20), GREEN, "saturation 10 dB above average (backed off)")]:
        ax.hlines(sat, 0, env.size, color=c, linestyles="--", lw=1.0)
        ax.text(env.size + 10, sat, lab.replace(" (", "\n("), va="center", fontsize=6.5, color=c)
    ax.plot(t, env, color=NAVY, lw=0.7)
    over = env > 10 ** (5 / 20)
    ax.fill_between(t, 10 ** (5 / 20), env, where=over, color=ACCENT, alpha=0.5, lw=0)
    ax.axhline(1, color=GRAY, lw=0.6); ax.text(-8, 1, "rms", ha="right", va="center", fontsize=6.5, color=GRAY)
    ax.set_xlim(-30, env.size + 360); ax.set_ylim(0, 3.6)
    ax.set_xlabel("time (samples)"); ax.set_ylabel("envelope $|x|$"); ax.set_yticks([0, 1, 2, 3])
    fig.tight_layout(); save(fig, "ch07_backoff_shout")
    return 100 * over.mean()


def fig_aclr_channels():
    """What ACLR integrates: power in the assigned channel versus the neighbours."""
    x = ofdm_signal(n_sym=200, r=rng(2))
    y = pa_static(x * 10 ** (-6 / 20), p=2.0) * 10 ** (6 / 20)
    f, P = psd_db(y, 2048)
    Pd = db10(P / P.max())
    bw = 300 / 1024
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.plot(f, Pd, color=NAVY, lw=0.9)
    for c0, col, nm in [(0, GREEN, "assigned channel"), (bw * 1.1, ACCENT, "adjacent"), (-bw * 1.1, ACCENT, "adjacent")]:
        ax.axvspan(c0 - bw / 2, c0 + bw / 2, color=col, alpha=0.12)
        ax.text(c0, 4, nm, ha="center", fontsize=7, color=col)
    a = aclr_db(y, bw, bw * 1.1)
    ax.text(0.42, -30, f"ACLR = {a:.0f} dB", fontsize=8, color=ACCENT, ha="center")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-70, 10)
    ax.set_xlabel("frequency (cycles/sample)"); ax.set_ylabel("PSD (dB)")
    fig.tight_layout(); save(fig, "ch07_aclr_channels")
    return a


def fig_evm_budget():
    """The 256-QAM EVM budget of the worked example."""
    items = [("phase noise 1°", 1.75, ORANGE), ("I/Q image (40 dB)", 1.0, SKY), ("PA after DPD", 2.0, ACCENT),
             ("DAC, filters", 0.5, GREEN)]
    tot = np.sqrt(sum(v ** 2 for _, v, _ in items))
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    rows = items + [("root-sum-square total", tot, NAVY)]
    for i, (nm, v, c) in enumerate(rows):
        ax.barh(len(rows) - 1 - i, v, color=c, height=0.62)
        ax.text(v + 0.06, len(rows) - 1 - i, f"{v:.2f}%" if i == 4 else f"{v}%", va="center", fontsize=7, color=c)
    ax.axvline(3.5, color=ACCENT, ls="--", lw=1.0)
    ax.text(3.45, 4.45, "256-QAM limit 3.5%", ha="right", fontsize=6.5, color=ACCENT)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
    ax.set_xlabel("EVM (%)"); ax.set_xlim(0, 4.2); ax.set_ylim(-0.5, 4.9)
    fig.tight_layout(); save(fig, "ch07_evm_budget")


def fig_dpd_handwriting():
    """Predistorter (expansive) followed by PA (compressive) gives a straight line."""
    a = np.linspace(0, 1.25, 400)
    pa = a / (1 + a ** 4) ** 0.25                          # Rapp p=2, sat 1
    # inverse of the PA, clipped at saturation
    u = np.linspace(0, 0.97, 400)
    inv = u / (1 - u ** 4) ** 0.25
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.95))
    ax[0].plot(u, inv, color=GREEN); ax[0].set_title("predistorter: expands", fontsize=8)
    ax[1].plot(a, pa, color=ACCENT); ax[1].set_title("PA: compresses", fontsize=8)
    casc = np.interp(np.minimum(inv, 1.25), a, pa)
    ax[2].plot(u, casc, color=NAVY); ax[2].set_title("together: straight", fontsize=8)
    for x in ax:
        x.plot([0, 1.2], [0, 1.2], color=GRAY, ls=":", lw=0.7)
        x.set_xlim(0, 1.2); x.set_ylim(0, 1.6); x.set_xlabel("input amplitude", fontsize=7.5)
    ax[0].set_ylabel("output amplitude", fontsize=7.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch07_dpd_handwriting")


def fig_et_supply():
    """Envelope tracking: the supply follows the envelope and the wasted headroom vanishes."""
    x = ofdm_signal(n_sym=4, r=rng(4))
    env = np.abs(x[1500:2300]); env = env / env.max()
    t = np.arange(env.size)
    vmin = 0.25
    vet = np.maximum(env * 1.12 + 0.04, vmin)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.fill_between(t, env, 1.1, color=ACCENT, alpha=0.12, lw=0, label="fixed supply: wasted as heat")
    ax.fill_between(t, env, vet, color=GREEN, alpha=0.35, lw=0, label="tracked supply: small headroom")
    ax.axhline(1.1, color=ACCENT, lw=1.0)
    ax.plot(t, vet, color=GREEN, lw=1.0)
    ax.plot(t, env, color=NAVY, lw=0.8, label="RF envelope")
    ax.set_ylim(0, 1.25); ax.set_xlim(0, env.size)
    ax.set_xlabel("time (samples)"); ax.set_ylabel("voltage (normalised)")
    ax.legend(fontsize=6.5, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.2), frameon=False)
    fig.tight_layout(); save(fig, "ch07_et_supply")


def fig_fullduplex_stack():
    """Self-interference cancellation, stage by stage."""
    steps = [("transmitted", 20), ("after antenna\nisolation (40 dB)", -20), ("after analog\ncancellation (30 dB)", -50),
             ("after digital\ncancellation (40 dB)", -90), ("residual\nvs floor", -95)]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    lv = [s[1] for s in steps[:-1]]
    cols = [ACCENT, ORANGE, SKY, GREEN]
    for i, v in enumerate(lv):
        ax.bar(i, v + 110, bottom=-110, color=cols[i], width=0.6, alpha=0.85)
        ax.text(i, v + 2, f"{v:+d} dBm", ha="center", fontsize=7)
    ax.axhline(-90, color=NAVY, ls="--", lw=1.0)
    ax.text(1.0, -87, "receiver noise floor $-90$ dBm: 110 dB of suppression in total", fontsize=7, color=NAVY, ha="center",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps[:-1]], fontsize=6.8)
    ax.set_ylabel("self-interference (dBm)"); ax.set_ylim(-110, 30)
    fig.tight_layout(); save(fig, "ch07_fullduplex_stack")


def fig_sdr_timeline():
    ev = [(1921, "Cady: quartz\noscillator"), (1936, "Doherty\namplifier"), (1966, "Leeson's\nphase-noise model"),
          (1968, "Gilbert\nmultiplier"), (1992, "Mitola: 'software\nradio'; SPEAKeasy"), (1995, "Abidi's zero-IF paper;\nMitola's article"),
          (2001, "GNU Radio\n(Blossom)"), (2004, "first USRP\n(Ettus)"), (2012, "RTL-SDR\n\\$20 dongle"),
          (2013, "USRP B200/B210"), (2017, "RFSoC direct\nRF sampling")]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.axhline(0, color=NAVY, lw=1.5)
    lv = [0.35, -0.35, 1.25, -1.25]
    for k, (y, t) in enumerate(ev):
        h = lv[k % 4]
        ax.plot([y, y], [0, h], color=GRAY, lw=0.6)
        ax.plot(y, 0, "o", color=ACCENT if y >= 1990 else NAVY, ms=4)
        ax.text(y, h + (0.05 if h > 0 else -0.05), f"{y}\n{t}", ha="center", va="bottom" if h > 0 else "top",
                fontsize=6.0, linespacing=1.0)
    ax.set_xlim(1912, 2028); ax.set_ylim(-2.2, 2.2); ax.axis("off")
    fig.tight_layout(); save(fig, "ch07_sdr_timeline")


def fig_camera_app():
    """How much of the radio is software: analog vs digital share, illustrative."""
    gens = ["1980s\nsuperhet", "2000s\nzero-IF phone", "2013\nUSRP B200", "today\nRFSoC radio"]
    analog = [85, 45, 30, 15]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    x = np.arange(4)
    ax.bar(x, analog, color=GRAY, width=0.6, label="analog hardware")
    ax.bar(x, [100 - a for a in analog], bottom=analog, color=NAVY, width=0.6, label="digital / software")
    ax.set_xticks(x); ax.set_xticklabels(gens, fontsize=6.4)
    ax.set_ylabel("share of signal-chain functions (%)", fontsize=7.2); ax.set_ylim(0, 118)
    ax.legend(fontsize=6.2, loc="upper center", ncol=2, frameon=False)
    ax.text(1.5, -38, "illustrative", fontsize=6, color=GRAY, ha="center")
    fig.tight_layout(); save(fig, "ch07_camera_app")


def fig_usb_rates():
    rates = np.linspace(0, 62, 200)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for b, c, nm in [(4, NAVY, "sc16 (4 B/sample)"), (3, GREEN, "sc12 (3 B)"), (2, ORANGE, "sc8 (2 B)")]:
        ax.plot(rates, rates * b, color=c, label=nm)
    ax.axvline(61.44, color=GRAY, ls=":", lw=0.8)
    ax.text(60.5, 30, "61.44 MS/s", rotation=90, fontsize=6.5, color=GRAY, ha="right")
    ax.plot(61.44, 245.8, "o", color=ACCENT, ms=4); ax.text(40, 238, "246 MB/s", fontsize=7, color=ACCENT)
    ax.set_xlabel("complex sample rate (MS/s)"); ax.set_ylabel("USB payload (MB/s)")
    ax.legend(fontsize=6.4, loc="upper left"); ax.set_xlim(0, 64); ax.set_ylim(0, 270)
    fig.tight_layout(); save(fig, "ch07_usb_rates")


def fig_overflow():
    """A receive buffer: steady fill by the radio, bursty drain by the host; a stall overflows it."""
    r = rng(8)
    n = 600; cap = 100.0
    lvl = np.zeros(n); b = 20.0; ovf = []
    for k in range(n):
        b += 1.0                                         # radio produces 1 unit per tick
        stall = 250 <= k < 380
        if not stall and k % 10 == 0:
            b -= min(b, 14 + r.uniform(-1, 3))
        if b > cap:
            ovf.append(k); b = cap
        lvl[k] = b
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.plot(lvl, color=NAVY, lw=0.9)
    ax.axhline(cap, color=ACCENT, ls="--", lw=0.9); ax.text(5, cap + 3, "buffer full", fontsize=7, color=ACCENT)
    ax.axvspan(250, 380, color=ORANGE, alpha=0.12)
    ax.text(315, 40, "host busy\n(Python loop,\nCPU throttled)", ha="center", fontsize=6.8, color=ORANGE)
    if ovf:
        ax.plot(ovf, [cap] * len(ovf), "|", color=ACCENT, ms=8)
        ax.text(ovf[len(ovf) // 2], cap + 10, "'O' printed: samples lost", ha="center", fontsize=7, color=ACCENT)
    ax.set_ylim(0, 125); ax.set_xlim(0, n)
    ax.set_xlabel("time"); ax.set_ylabel("buffer fill (%)")
    fig.tight_layout(); save(fig, "ch07_overflow")


def fig_nf_measure():
    """Measuring noise figure with a terminated input and a known tone (the in-practice numbers)."""
    r = rng(9)
    f = np.linspace(-1, 1, 4096)
    floor_dbfs_hz = -111.0
    rbw = 1e3
    floor = floor_dbfs_hz + 10 * np.log10(rbw) + 0.8 * r.standard_normal(f.size)
    P = 10 ** (floor / 10)
    P[np.argmin(np.abs(f - 0.3))] += 10 ** (-22 / 10)
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.plot(f, 10 * np.log10(P), color=NAVY, lw=0.6)
    ax.text(0.33, -24, "$-80$ dBm tone reads $-22$ dBFS\n$\\Rightarrow$ 58 dB from dBm to dBFS", fontsize=6.8, color=ACCENT, va="top")
    ax.text(-0.95, -72, "terminated floor: $-111$ dBFS/Hz $\\Rightarrow$ $-169$ dBm/Hz $\\Rightarrow$ NF $=-169-(-174)=5$ dB",
            fontsize=6.8, color=GREEN)
    ax.set_ylim(-90, -10); ax.set_xlim(-1, 1)
    ax.set_xlabel("frequency offset (MHz)"); ax.set_ylabel("level (dBFS in 1 kHz)")
    fig.tight_layout(); save(fig, "ch07_nf_measure")


def fig_im2_envelope():
    """Second-order distortion demodulates a blocker's envelope onto DC."""
    t = np.linspace(0, 1, 4000)
    env = 1 + 0.6 * np.sin(2 * np.pi * 3 * t) * np.cos(2 * np.pi * 1.3 * t)
    rf = env * np.cos(2 * np.pi * 60 * t)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0))
    ax[0].plot(t, rf, color=GRAY, lw=0.5); ax[0].plot(t, env, color=NAVY, lw=1.2)
    ax[0].set_title("a strong AM/OFDM-like blocker far from DC", fontsize=8)
    ax[0].set_xlabel("time"); ax[0].set_yticks([])
    y2 = rf ** 2
    lp = np.convolve(y2, np.ones(80) / 80, mode="same")
    ax[1].plot(t, y2, color=GRAY, lw=0.4, alpha=0.6, label="$a_2x^2$ before filtering")
    ax[1].plot(t[60:-60], lp[60:-60], color=ACCENT, lw=1.4, label="what lands at baseband: $a_2 b^2(t)/2$")
    ax[1].set_title("after a square-law term: its envelope, at DC", fontsize=8)
    ax[1].set_xlabel("time"); ax[1].set_yticks([]); ax[1].legend(fontsize=6.3, loc="upper right")
    ax[1].set_ylim(0, 3.6)
    fig.tight_layout(); save(fig, "ch07_im2_envelope")


def fig_switching_mixer():
    """A switching mixer: RF times a +/-1 square wave at the LO frequency."""
    t = np.linspace(0, 1, 3000)
    rf = np.cos(2 * np.pi * 11 * t)
    lo = np.sign(np.cos(2 * np.pi * 10 * t))
    out = rf * lo
    bb, aa = sps.butter(4, 3 / 1500)
    lp = sps.filtfilt(bb, aa, out)
    fig, ax = plt.subplots(3, 1, figsize=(W1, 2.6), sharex=True)
    for a, y, c, nm in [(ax[0], rf, NAVY, "RF in (11 units)"), (ax[1], lo, ORANGE, "LO switch $\\pm1$ (10 units)"),
                        (ax[2], out, GRAY, "product")]:
        a.plot(t, y, color=c, lw=0.8); a.set_yticks([]); a.grid(False)
        a.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=7.5)
    ax[2].plot(t, np.pi / 2 * lp, color=ACCENT, lw=1.6)
    ax[2].text(0.5, 1.25, "after low-pass: the 1-unit difference frequency (red)", fontsize=7, color=ACCENT, ha="center")
    ax[2].set_ylim(-1.3, 1.6); ax[2].set_xlabel("time")
    fig.tight_layout(h_pad=0.2); save(fig, "ch07_switching_mixer")


def fig_quadrature_divider():
    """Divide-by-two from a 2x VCO: two flip-flops on opposite edges give I and Q."""
    t = np.linspace(0, 4, 4000)
    vco = (np.sin(2 * np.pi * 2 * t) > 0).astype(float)
    i = (np.sin(2 * np.pi * t) > 0).astype(float)
    q = (np.sin(2 * np.pi * t - np.pi / 2) > 0).astype(float)
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for k, (y, c, nm) in enumerate([(vco, ORANGE, "VCO at $2f_{\\rm LO}$"), (i, NAVY, "LO I (rising edges)"),
                                    (q, GREEN, "LO Q (falling edges)")]):
        ax.plot(t, y * 0.8 + 2 * (2 - k), color=c, lw=1.3)
        ax.text(-0.08, 2 * (2 - k) + 0.4, nm, ha="right", va="center", fontsize=7.5, color=c)
    for e in np.arange(0, 4, 0.25):
        ax.axvline(e, color=GRAY, lw=0.4, ls=":")
    ax.annotate("", xy=(0.5, -0.35), xytext=(0.25, -0.35), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.8))
    ax.text(0.55, -0.45, "a quarter LO period = 90°", fontsize=7, color=ACCENT, va="center")
    ax.set_xlim(-1.4, 4); ax.set_ylim(-0.8, 5.0); ax.axis("off")
    fig.tight_layout(); save(fig, "ch07_quadrature_divider")


def fig_lo_leakage():
    """Transmitter I/Q impairments: LO leakage and sideband image, before and after calibration."""
    r = rng(12)
    n = 1 << 15
    k = np.arange(n)
    x = np.exp(2j * np.pi * 0.11 * k)                       # single-sideband test tone
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    for (g, ph, dc, c, lab, lw) in [(10 ** (0.4 / 20), np.deg2rad(3), 0.03, ACCENT, "uncalibrated", 0.9),
                                    (10 ** (0.02 / 20), np.deg2rad(0.15), 0.001, NAVY, "after loopback calibration", 1.1)]:
        y = x.real + 1j * g * (np.cos(ph) * x.imag + np.sin(ph) * x.real) + dc * (1 + 1j)
        y = y + 1e-4 * (r.standard_normal(n) + 1j * r.standard_normal(n))
        f, P = sps.welch(y, nperseg=4096, return_onesided=False, window="blackmanharris", detrend=False)
        f = np.fft.fftshift(f); P = np.fft.fftshift(P)
        ax.plot(f, db10(P / P.max()), color=c, lw=lw, label=lab)
    ax.text(0.11, 3, "wanted tone", ha="center", fontsize=7, color="#333333")
    ax.text(0.0, -24, "LO leakage", ha="center", fontsize=7, color=ACCENT)
    ax.text(-0.11, -24, "image", ha="center", fontsize=7, color=ACCENT)
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(-95, 10)
    ax.set_xlabel("frequency (cycles/sample)"); ax.set_ylabel("PSD (dBc)")
    ax.legend(fontsize=6.6, loc="upper left")
    fig.tight_layout(); save(fig, "ch07_lo_leakage")


def fig_rapp_family():
    """Rapp AM/AM for several smoothness factors."""
    a = np.linspace(0, 2.2, 400)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for p, c in [(1, GRAY), (2, NAVY), (3, GREEN), (10, ACCENT)]:
        ax.plot(a, a / (1 + a ** (2 * p)) ** (1 / (2 * p)), color=c, label=f"$p={p}$")
    ax.plot([0, 1.2], [0, 1.2], color=GRAY, ls=":", lw=0.7)
    ax.axhline(1, color=GRAY, lw=0.5)
    ax.set_xlabel("input amplitude $G|x|/A_{\\rm sat}$"); ax.set_ylabel("output amplitude")
    ax.legend(fontsize=6.6, loc="lower right", title="Rapp smoothness", title_fontsize=6.6)
    ax.set_ylim(0, 1.25)
    fig.tight_layout(); save(fig, "ch07_rapp_family")


def fig_offset_tuning():
    """Offset tuning: move the DC spur and DC notch away from the signal."""
    r = rng(13)
    f = np.linspace(-1, 1, 2000)
    def spec(fc):
        P = 10 ** (-9) * (1 + 0.3 * r.standard_normal(f.size) ** 2)
        P += 1e-5 * (np.abs(f - fc) < 0.1)
        P[np.argmin(np.abs(f))] += 1e-3
        return 10 * np.log10(P)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0), sharey=True)
    ax[0].plot(f, spec(0.0), color=NAVY, lw=0.6); ax[0].set_title("tuned dead centre: spur on the signal", fontsize=8)
    ax[1].plot(f, spec(0.45), color=NAVY, lw=0.6); ax[1].set_title("LO offset 450 kHz: spur outside it", fontsize=8)
    for a in ax:
        a.set_xlabel("baseband frequency (MHz)"); a.annotate("DC spur", xy=(0, -32), xytext=(-0.85, -38), fontsize=7,
                                                             color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax[1].text(0.45, -45, "signal", ha="center", fontsize=7, color=GREEN)
    ax[0].set_ylabel("PSD (dB)"); ax[0].set_ylim(-95, -20)
    fig.tight_layout(); save(fig, "ch07_offset_tuning")


def fig_lineup_cumulative():
    """The worked B200-like line-up: cumulative NF and IIP3 stage by stage."""
    st = [("switch+balun", -2.5, 2.5, None), ("LNA", 20, 1.5, -10), ("mixer+TIA", 15, 10, 5), ("BB filter+VGA", 30, 20, 15)]
    rows = cascade(st)
    names = ["switch +\nbalun", "LNA", "mixer +\nTIA", "baseband\n+ VGA"]
    nf = [r[2] for r in rows]
    ip = [r[3] if np.isfinite(r[3]) else np.nan for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.1))
    ax[0].bar(range(4), nf, color=NAVY, width=0.6)
    for i, v in enumerate(nf):
        ax[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=7)
    ax[0].set_ylabel("cumulative NF (dB)"); ax[0].set_ylim(0, 5.2); ax[0].set_title("noise: settled by the LNA", fontsize=8)
    ax[1].plot(range(1, 4), ip[1:], "o-", color=ACCENT)
    for i in range(1, 4):
        ax[1].text(i, ip[i] + 1.0, f"{ip[i]:.1f}", ha="center", fontsize=7)
    ax[1].set_ylabel("cumulative IIP3 (dBm)"); ax[1].set_ylim(-23, -3); ax[1].set_xlim(-0.4, 3.4)
    ax[1].set_title("linearity: set by the last stage", fontsize=8)
    for a in ax:
        a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=6.8)
    fig.tight_layout(); save(fig, "ch07_lineup_cumulative")


def fig_irr_vs_freq():
    """Frequency-dependent imbalance: slightly mismatched I and Q baseband filters."""
    f = np.linspace(0.05e6, 12e6, 400)
    w = 2 * np.pi * f
    b1, a1 = sps.butter(5, 2 * np.pi * 10e6, analog=True)
    b2, a2 = sps.butter(5, 2 * np.pi * 10.1e6, analog=True)
    _, H1 = sps.freqs(b1, a1, w); _, H2 = sps.freqs(b2, a2, w)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for (gdb, ph, c, lab) in [(0.0, 0.0, NAVY, "filters only"), (0.1, 1.0, ACCENT, "+ 0.1 dB, 1° (LO)")]:
        g = 10 ** (gdb / 20) * H2 / H1 * np.exp(1j * np.deg2rad(ph))
        irr = 10 * np.log10(np.abs(1 + g) ** 2 / np.abs(1 - np.conj(g)) ** 2 + 1e-30)
        ax.plot(f / 1e6, np.minimum(irr, 80), color=c, label=lab)
    ax.axvline(10, color=GRAY, ls=":", lw=0.8); ax.text(9.8, 72, "filter\ncorner", fontsize=6.5, color=GRAY, ha="right")
    ax.set_xlabel("baseband frequency (MHz)"); ax.set_ylabel("IRR (dB)"); ax.set_ylim(20, 82)
    ax.legend(fontsize=6.4, loc="lower left")
    fig.tight_layout(); save(fig, "ch07_irr_vs_freq")


def fig_cable_loss():
    """Loss before the LNA: system NF against cable length, with and without a masthead LNA."""
    L = np.linspace(0, 10, 200)              # metres
    loss = 1.0 * L                           # approx. thin RG-174 at 2.4 GHz, dB
    nf_sdr = 5.0
    F_sdr = 10 ** (nf_sdr / 10)
    direct = loss + nf_sdr
    F1, G1 = 10 ** (0.8 / 10), 10 ** (20 / 10)
    Fc = 10 ** (loss / 10)
    pre = 10 * np.log10(F1 + (Fc - 1) / G1 + (F_sdr - 1) / (G1 / Fc))
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(L, direct, color=ACCENT, label="cable, then SDR (NF 5 dB)")
    ax.plot(L, pre, color=GREEN, label="LNA at antenna, then cable")
    ax.set_xlabel("thin coax length (m), ~1 dB/m"); ax.set_ylabel("system noise figure (dB)")
    ax.legend(fontsize=6.4, loc="upper left"); ax.set_ylim(0, 16)
    fig.tight_layout(); save(fig, "ch07_cable_loss")


def new_figs2():
    fig_lineup_cumulative(); fig_irr_vs_freq(); fig_cable_loss()
    fig_im2_envelope(); fig_switching_mixer(); fig_quadrature_divider(); fig_lo_leakage(); fig_rapp_family()
    fig_offset_tuning()


def new_figs():
    new_figs2()
    fig_by_numbers(); fig_spectrum_mic(); fig_whisper_chain(); fig_mixer_translator(); fig_image_mirror()
    fig_harmonic_mixing(); fig_sfdr_window(); print("desense at P1dB:", fig_desense()); fig_duplexer(); fig_adc_ladder()
    fig_crystal_tempco(); print("fracN mean:", fig_fracn_dither()); fig_wobbly_lo(); fig_image_tone(); fig_dc_notch()
    fig_class_waveforms(); print("% over 5 dB:", fig_backoff_shout()); print("aclr:", fig_aclr_channels())
    fig_evm_budget(); fig_dpd_handwriting(); fig_et_supply(); fig_fullduplex_stack(); fig_sdr_timeline()
    fig_camera_app(); fig_usb_rates(); fig_overflow(); fig_nf_measure()


def numbers():
    print("== IRR 0.1 dB / 1 deg:", irr_exact(0.1, 1.0))
    print("== IRR 0.5 dB / 3 deg:", irr_exact(0.5, 3.0), " 0.4/2:", irr_exact(0.4, 2.0))
    st = [("switch+balun", -2.5, 2.5, None), ("LNA", 20, 1.5, -10), ("mixer+TIA", 15, 10, 5), ("BB filter+VGA", 30, 20, 15)]
    for row in cascade(st):
        print("  cascade", row)
    f = np.logspace(3, 7, 2000)
    *_, tot = pll_components(f)
    print("== PLL rms phase error 1k-10M (deg):", rms_deg(f, tot))
    for fbw in (50e3, 200e3, 800e3):
        *_, t = pll_components(f, fbw=fbw)
        print("   loop bw", fbw, rms_deg(f, t))
    print("== Leeson VCO Q=8 at 1 MHz:", leeson_dbc(1e6, 2.4e9, 8, 15, 0, 1e5))
    *_, t = pll_components(np.array([1e3, 1e4, 1e5, 1e6, 1e7]))
    print("   PLL total at 1k..10M:", t)


if __name__ == "__main__":
    if "--numbers" in sys.argv:
        numbers()
        sys.exit()
    if "--new" in sys.argv:
        new_figs()
        sys.exit()
    new_figs()
    fig_two_tone(); fig_pa_regrowth(); fig_dpd(); fig_dpd_amam(); fig_phase_noise(); fig_architectures()
    fig_zeroif_impairments(); fig_irr(); fig_leeson(); fig_reciprocal(); fig_iq_correction()
    print("avg eff:", fig_pa_efficiency())
    fig_doherty()
    print("aclr/evm:", [np.round(a, 1) for a in fig_aclr_backoff()])
    print("level:", fig_level_diagram())
    print("cascade best:", fig_cascade_tradeoff())
    fig_dac_images(); fig_cfr(); fig_freq_accuracy()
