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
    fig_two_tone(); fig_pa_regrowth(); fig_dpd(); fig_dpd_amam(); fig_phase_noise(); fig_architectures()
    fig_zeroif_impairments(); fig_irr(); fig_leeson(); fig_reciprocal(); fig_iq_correction()
    print("avg eff:", fig_pa_efficiency())
    fig_doherty()
    print("aclr/evm:", [np.round(a, 1) for a in fig_aclr_backoff()])
    print("level:", fig_level_diagram())
    print("cascade best:", fig_cascade_tradeoff())
    fig_dac_images(); fig_cfr(); fig_freq_accuracy()
