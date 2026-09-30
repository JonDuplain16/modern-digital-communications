"""Figures for Chapter 6: Digital Filters and Multirate Processing."""
from figstyle import *
from scipy import signal as sps


def db(h):
    return 20 * np.log10(np.abs(h) + 1e-12)


def fig_fir_design():
    fs = 1.0
    N = 41
    bands = [0, 0.1, 0.15, 0.5]
    h_win = sps.firwin(N, 0.125, window="hamming")
    h_kai = sps.firwin(N, 0.125, window=("kaiser", 6))
    h_pm = sps.remez(N, bands, [1, 0], weight=[1, 10])
    h_ls = sps.firls(N, bands, [1, 1, 0, 0], weight=[1, 10])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for h, lab, c in [(h_win, "window (Hamming)", GRAY), (h_kai, "window (Kaiser $\\beta$=6)", GREEN),
                      (h_ls, "least squares", ORANGE), (h_pm, "Parks-McClellan (equiripple)", NAVY)]:
        w, H = sps.freqz(h, worN=4096, fs=fs)
        ax[0].plot(w, db(H), color=c, lw=1.0, label=lab)
    ax[0].axvspan(0.1, 0.15, color=GRAY, alpha=0.12)
    ax[0].set_ylim(-100, 5); ax[0].set_xlabel("frequency (cycles/sample)"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.5, loc="upper right"); ax[0].set_title("41-tap low-pass designs", fontsize=8.5)
    n = np.arange(N) - N // 2
    ax[1].stem(n, h_pm, basefmt=" ", linefmt=NAVY, markerfmt="o")
    plt.setp(ax[1].lines, markersize=2.5)
    ax[1].set_title("equiripple impulse response", fontsize=8.5); ax[1].set_xlabel("tap")
    fig.tight_layout(); save(fig, "ch06_fir_design")


def fig_iir_compare():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    designs = [("Butterworth", sps.butter(6, 0.2, output="sos"), NAVY),
               ("Chebyshev I (1 dB)", sps.cheby1(6, 1, 0.2, output="sos"), GREEN),
               ("elliptic (1 dB, 60 dB)", sps.ellip(6, 1, 60, 0.2, output="sos"), ACCENT)]
    for name, sos, c in designs:
        w, H = sps.sosfreqz(sos, worN=4096, fs=2)
        ax[0].plot(w / 2, db(H), color=c, label=name)
        w, gd = sps.group_delay(sps.sos2tf(sos), w=2048, fs=2)
        ax[1].plot(w / 2, gd, color=c)
    ax[0].set_ylim(-90, 3); ax[0].set_xlabel("cycles/sample"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.3); ax[0].set_title("6th-order IIR magnitude", fontsize=8.5)
    ax[1].set_xlim(0, 0.2); ax[1].set_ylim(0, 40); ax[1].set_xlabel("cycles/sample")
    ax[1].set_ylabel("samples"); ax[1].set_title("group delay in passband", fontsize=8.5)
    z, p, k = sps.ellip(6, 1, 60, 0.2, output="zpk")
    th = np.linspace(0, 2 * np.pi, 300)
    ax[2].plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    ax[2].plot(z.real, z.imag, "o", mfc="none", color=ACCENT, ms=5, label="zeros")
    ax[2].plot(p.real, p.imag, "x", color=NAVY, ms=5, label="poles")
    ax[2].set_aspect("equal"); ax[2].set_xlim(-1.3, 1.3); ax[2].set_ylim(-1.3, 1.3)
    ax[2].legend(fontsize=6.5, loc="lower left"); ax[2].set_title("elliptic pole-zero", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_iir_compare")


def fig_coef_quant():
    sos = sps.ellip(8, 0.5, 70, 0.15, output="sos")
    b, a = sps.sos2tf(sos)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.4))
    w, H = sps.freqz(b, a, worN=4096, fs=1)
    ax.plot(w, db(H), color=NAVY, label="double precision")
    for bits, c in [(16, GREEN), (12, ACCENT)]:
        q = 2.0 ** -(bits - 1 - 4)            # coefficients up to |16|
        bq, aq = np.round(b / q) * q, np.round(a / q) * q
        w, Hq = sps.freqz(bq, aq, worN=4096, fs=1)
        stable = np.all(np.abs(np.roots(aq)) < 1)
        ax.plot(w, db(Hq), color=c, lw=0.9, label=f"direct form, {bits}-bit coefs" + ("" if stable else " (UNSTABLE)"))
        sq = np.round(sos / 2 ** -(bits - 2)) * 2 ** -(bits - 2)
        w, Hs = sps.sosfreqz(sq, worN=4096, fs=1)
        ax.plot(w, db(Hs), color=c, ls=":", lw=1.1, label=f"cascaded biquads, {bits}-bit")
    ax.set_ylim(-100, 20); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.2, loc="lower left"); ax.set_title("8th-order elliptic: coefficient quantization")
    fig.tight_layout(); save(fig, "ch06_coef_quant")


def fig_decimation():
    r = rng(3)
    fs = 1.0
    N = 1 << 15
    x = sps.lfilter(sps.firwin(255, 0.06), 1, r.standard_normal(N))
    x += 1.0 * np.cos(2 * np.pi * 0.27 * np.arange(N))          # out-of-band interferer
    M = 4
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    def psd(a, sig, f_s, ttl):
        f, P = sps.welch(sig, fs=f_s, nperseg=2048, return_onesided=False)
        a.plot(np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) + 1e-12), color=NAVY, lw=0.8)
        a.set_title(ttl, fontsize=8); a.set_xlabel("cycles/input sample")
    psd(ax[0], x, 1, "input: band at $\\pm$0.03, tone at 0.27")
    psd(ax[1], x[::M], 1 / M, "naive downsample by 4:\ntone aliases in-band")
    h = sps.firwin(121, 0.1)
    psd(ax[2], sps.lfilter(h, 1, x)[::M], 1 / M, "filter first, then downsample")
    ax[0].set_ylabel("dB"); ax[0].set_ylim(-80, 45)
    fig.tight_layout(); save(fig, "ch06_decimation")


def fig_cic():
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    R = 16
    f = np.linspace(1e-4, 0.5, 4000)                     # output-rate units after decimation? use input rate
    fin = f / R * 1                                       # input-rate frequency for plotting up to fs_in/2? keep simple
    fi = np.linspace(1e-5, 0.5, 5000)
    for Nst, c in [(1, GRAY), (3, GREEN), (5, NAVY)]:
        H = np.abs(np.sin(np.pi * fi * R) / (R * np.sin(np.pi * fi))) ** Nst
        ax.plot(fi * R, 20 * np.log10(H + 1e-12), color=c, label=f"N = {Nst} stages")
    for k in range(1, 8):
        ax.axvline(k, color=ACCENT, lw=0.5, ls=":")
    ax.set_xlim(0, 8); ax.set_ylim(-120, 3)
    ax.set_xlabel("frequency / output sample rate ($R$ = 16)"); ax.set_ylabel("dB")
    ax.legend(fontsize=7); ax.set_title("CIC decimator response: nulls fall on the alias bands")
    fig.tight_layout(); save(fig, "ch06_cic")


def fig_nco_spurs():
    N = 1 << 14
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    acc_bits = 32
    fw = int(0.1234567 * 2 ** acc_bits)
    phase = (fw * np.arange(N)) % 2 ** acc_bits
    for a, pbits, ttl in [(ax[0], 8, "phase truncated to 8 bits"), (ax[1], 14, "phase truncated to 14 bits")]:
        p = (phase >> (acc_bits - pbits)) / 2 ** pbits
        s = np.exp(2j * np.pi * p)
        S = np.abs(np.fft.fftshift(np.fft.fft(s * np.blackman(N)))) ** 2
        S /= S.max()
        f = np.fft.fftshift(np.fft.fftfreq(N))
        a.plot(f, 10 * np.log10(S + 1e-20), color=NAVY, lw=0.5)
        a.set_title(ttl, fontsize=8.5); a.set_xlabel("cycles/sample"); a.set_ylim(-140, 5)
    ax[0].set_ylabel("dB")
    fig.suptitle("NCO spurs from phase truncation (rule: about 6 dB per phase bit)", fontsize=9)
    fig.tight_layout(); save(fig, "ch06_nco_spurs")


def fig_polyphase():
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    M = 8
    h = sps.firwin(M * 16, 1 / M * 0.9)
    w, H = sps.freqz(h, worN=4096, fs=1)
    ax.plot(w, db(H), color=GRAY, lw=0.8, label="prototype low-pass")
    for k in range(-M // 2, M // 2):
        hk = h * np.exp(2j * np.pi * k * np.arange(len(h)) / M)
        w, Hk = sps.freqz(hk, worN=4096, whole=True, fs=1)
        w = np.where(w >= 0.5, w - 1, w); o = np.argsort(w)
        ax.plot(w[o], db(Hk[o]), lw=0.8)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-80, 5); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.set_title("8-channel polyphase filter bank (channelizer)")
    fig.tight_layout(); save(fig, "ch06_polyphase")


if __name__ == "__main__":
    fig_fir_design(); fig_iir_compare(); fig_coef_quant(); fig_decimation(); fig_cic()
    fig_nco_spurs(); fig_polyphase()
