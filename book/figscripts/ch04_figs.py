"""Figures for Chapter 4: Analog Modulation and the Classic Radio."""
from figstyle import *
from scipy import signal as sps
from scipy.special import jv
import commlib as cl


def lowpass(x, fc, fs, order=6):
    sos = sps.butter(order, fc, fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def fig_am():
    fs = 200e3; t = np.arange(0, 4e-3, 1 / fs)
    fc, fm = 10e3, 500
    m = np.cos(2 * np.pi * fm * t)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    for a, mu in zip(ax, [0.5, 1.0, 1.4]):
        x = (1 + mu * m) * np.cos(2 * np.pi * fc * t)
        env = lowpass(np.abs(x), 3e3, fs) * np.pi / 2       # full-wave rectifier + low-pass
        a.plot(t * 1e3, x, color=NAVY, lw=0.5)
        a.plot(t * 1e3, 1 + mu * m, color=GREEN, lw=1.0, ls="--", label="$1+\\mu m(t)$")
        a.plot(t * 1e3, env, color=ACCENT, lw=1.2, label="envelope detector")
        a.set_title(f"$\\mu$ = {mu}" + (" (over-modulated)" if mu > 1 else ""), fontsize=8.5)
        a.set_xlabel("time (ms)")
    ax[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch04_am")


def voice_like(fs, dur, r):
    """Band-limited random 'message' with a speech-like spectrum 300-3400 Hz."""
    n = int(fs * dur)
    w = r.standard_normal(n)
    sos = sps.butter(4, [300, 3400], btype="band", fs=fs, output="sos")
    m = sps.sosfilt(sos, w)
    return m / np.max(np.abs(m))


def fig_am_family():
    r = rng(4); fs = 64e3; fc = 12e3
    m = voice_like(fs, 4.0, r)
    t = np.arange(len(m)) / fs
    mh = np.imag(sps.hilbert(m))
    c, s = np.cos(2 * np.pi * fc * t), np.sin(2 * np.pi * fc * t)
    sigs = {
        "AM (DSB with carrier), $\\mu$=0.8": (1 + 0.8 * m) * c,
        "DSB-SC": m * c,
        "SSB, upper sideband": m * c - mh * s,
        "SSB, lower sideband": m * c + mh * s,
    }
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.2), sharex=True, sharey=True)
    for a, (name, x) in zip(ax.ravel(), sigs.items()):
        f, p = sps.welch(x, fs, nperseg=8192)
        a.plot(f / 1e3, 10 * np.log10(p / p.max() + 1e-12), color=NAVY, lw=0.9)
        a.set_title(name, fontsize=8.5); a.set_ylim(-60, 3); a.set_xlim(6, 18)
        a.axvline(fc / 1e3, color=ACCENT, lw=0.6, ls=":")
    ax[1, 0].set_xlabel("frequency (kHz)"); ax[1, 1].set_xlabel("frequency (kHz)")
    ax[0, 0].set_ylabel("dB"); ax[1, 0].set_ylabel("dB")
    fig.tight_layout(); save(fig, "ch04_am_family")


def fig_ssb_phase_error():
    err = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.3))
    for gdb, c in [(0.0, NAVY), (0.1, GREEN), (0.5, ACCENT)]:
        g = 10 ** (gdb / 20); phi = np.deg2rad(err)
        # suppression = |1 + g e^{j phi}|^2 / |1 - g e^{j phi}|^2
        supp = 10 * np.log10(np.abs(1 + g * np.exp(1j * phi)) ** 2 / np.maximum(np.abs(1 - g * np.exp(1j * phi)) ** 2, 1e-12))
        ax.plot(err, supp, color=c, label=f"gain mismatch {gdb} dB")
    ax.set_ylim(15, 70); ax.set_xlabel("phase error in 90$^\\circ$ network (degrees)")
    ax.set_ylabel("sideband suppression (dB)"); ax.legend(fontsize=7)
    ax.set_title("Phasing-method SSB: unwanted sideband level")
    fig.tight_layout(); save(fig, "ch04_ssb_phase")


def fig_fm_bessel():
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
    b = np.linspace(0, 12, 500)
    axb = fig.add_subplot(gs[0, :])
    for n in range(0, 5):
        axb.plot(b, jv(n, b), label=f"$J_{n}(\\beta)$")
    axb.axhline(0, color="k", lw=0.5); axb.set_xlabel("modulation index $\\beta$"); axb.legend(ncol=5, fontsize=7, loc="upper right")
    axb.set_title("Bessel functions give the amplitudes of the FM carrier ($n=0$) and sidebands ($n\\geq1$)")
    for i, beta in enumerate([0.2, 1.0, 2.405, 5.0]):
        a = fig.add_subplot(gs[1, i])
        n = np.arange(-12, 13)
        a.stem(n, np.abs(jv(n, beta)), linefmt=NAVY, markerfmt=" ", basefmt=" ")
        a.set_title(f"$\\beta$ = {beta}" + (" (carrier null)" if beta == 2.405 else ""), fontsize=8)
        a.set_ylim(0, 1.05); a.set_xlabel("$(f-f_c)/f_m$", fontsize=8)
        if i: a.set_yticklabels([])
    fig.tight_layout(); save(fig, "ch04_fm_bessel")


def fig_carson():
    beta = np.linspace(0.1, 12, 200)
    n98 = []
    for bb in beta:
        n = np.arange(0, 60); p = jv(n, bb) ** 2
        cum = p[0] + 2 * np.cumsum(p[1:])
        n98.append(np.argmax(np.concatenate([[p[0]], p[0] + 2 * np.cumsum(p[1:])]) >= 0.98))
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.4))
    ax.plot(beta, 2 * np.array(n98), color=NAVY, drawstyle="steps-post", label="98% power bandwidth (exact)")
    ax.plot(beta, 2 * (beta + 1), color=ACCENT, ls="--", label="Carson: $2(\\beta+1)$")
    ax.set_xlabel("modulation index $\\beta$"); ax.set_ylabel("bandwidth / $f_m$"); ax.legend(fontsize=7.5)
    ax.set_title("Carson's rule versus the true FM bandwidth")
    fig.tight_layout(); save(fig, "ch04_carson")


def fm_output_snr(cnr_db, beta, fs, fm, r, n=200_000, deemph=False):
    t = np.arange(n) / fs
    m = np.cos(2 * np.pi * fm * t)
    fd = beta * fm
    phase = 2 * np.pi * fd * np.cumsum(m) / fs
    x = np.exp(1j * phase)
    B = 2 * (beta + 1) * fm                                  # IF (Carson) bandwidth of this signal
    Bref = 12 * fm                                           # CNR is referenced to the beta=5 Carson band
    n0 = 1 / 10 ** (cnr_db / 10) / Bref * fs
    y = x + np.sqrt(n0 / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))
    y = lowpass(y.real, B / 2, fs, 8) + 1j * lowpass(y.imag, B / 2, fs, 8)
    d = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi * fd)
    d = lowpass(d, 1.5 * fm, fs, 8)[2000:-2000]
    ref = m[1:][2000:-2000]
    a = np.dot(d, ref) / np.dot(ref, ref)
    err = d - a * ref
    return 10 * np.log10(np.mean((a * ref) ** 2) / np.mean(err ** 2))


def fig_fm_threshold():
    r = rng(5); fs = 400e3; fm = 1e3
    cnr = np.arange(0, 26, 2)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.9))
    for beta, c in [(2, GREEN), (5, NAVY)]:
        sim = [fm_output_snr(x, beta, fs, fm, r) for x in cnr]
        theory = cnr + 10 * np.log10(8) + 10 * np.log10(1.5 * beta ** 2 / 1.5 ** 2)   # 3 (fd/W)^2 P_m x baseband SNR, W = 1.5 fm
        ax.plot(cnr, sim, "o-", color=c, ms=3.5, label=f"FM $\\beta$={beta}: simulation")
        ax.plot(cnr, theory, ":", color=c, label=f"FM $\\beta$={beta}: above-threshold theory")
    ax.plot(cnr, cnr + 10 * np.log10(8) - 10 * np.log10(3), color=ACCENT, ls="--", label="AM ($\\mu$=1), same $C/N_0$")
    ax.plot(cnr, cnr + 10 * np.log10(8), color=GRAY, ls="-.", lw=1, label="baseband (no modulation)")
    ax.set_xlabel("carrier-to-noise ratio in the FM ($\\beta$=5) Carson bandwidth (dB)")
    ax.set_ylabel("output SNR (dB)"); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("The FM threshold effect (Monte Carlo, tone modulation)")
    fig.tight_layout(); save(fig, "ch04_fm_threshold")


def fig_stereo_mpx():
    f = np.linspace(0, 60, 3000)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    def blk(lo, hi, h, c, lab):
        ax.fill_between(f, np.where((f >= lo) & (f <= hi), h, 0), color=c, alpha=0.75, lw=0)
        ax.text((lo + hi) / 2, h + 0.04, lab, ha="center", fontsize=7.5)
    blk(0.03, 15, 0.9, NAVY, "L+R (mono)\n30 Hz-15 kHz")
    blk(23, 37.97, 0.45, GREEN, "L-R lower sideband")
    blk(38.03, 53, 0.45, GREEN, "L-R upper sideband")
    ax.vlines(19, 0, 0.3, color=ACCENT, lw=2); ax.text(19, 0.34, "19 kHz\npilot", ha="center", fontsize=7, color=ACCENT)
    ax.vlines(38, 0, 0.12, color=GRAY, lw=1, ls=":"); ax.text(38, 0.6, "38 kHz (suppressed)", ha="center", fontsize=6.5, color=GRAY)
    blk(55.6, 58.4, 0.15, ORANGE, "RDS\n57 kHz")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.25); ax.set_yticks([]); ax.set_xlabel("baseband frequency (kHz)")
    ax.set_title("The FM stereo multiplex (MPX) signal that frequency-modulates the carrier")
    fig.tight_layout(); save(fig, "ch04_stereo_mpx")


def fig_superhet():
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    f = np.linspace(80, 125, 2000)
    def peak(fc, h, c, w=0.3):
        ax.fill_between(f, h * np.exp(-((f - fc) / w) ** 2), color=c, alpha=0.8, lw=0)
    peak(98.1, 1.0, NAVY); ax.text(98.1, 1.05, "wanted\n98.1 MHz", ha="center", fontsize=7)
    ax.vlines(108.8, 0, 0.8, color=ACCENT, lw=2); ax.text(108.8, 0.84, "LO\n108.8", ha="center", fontsize=7, color=ACCENT)
    peak(119.5, 0.7, ORANGE); ax.text(119.5, 0.75, "image\n119.5 MHz", ha="center", fontsize=7, color=ORANGE)
    ax.annotate("", (108.8, 0.35), (98.1, 0.35), arrowprops=dict(arrowstyle="<->", color=GRAY))
    ax.annotate("", (119.5, 0.35), (108.8, 0.35), arrowprops=dict(arrowstyle="<->", color=GRAY))
    ax.text(103.4, 0.39, "10.7 MHz", ha="center", fontsize=7); ax.text(114.2, 0.39, "10.7 MHz", ha="center", fontsize=7)
    ff = np.linspace(80, 125, 400)
    ax.plot(ff, 1.15 / (1 + ((ff - 98.1) / 6) ** 4), color=GREEN, lw=1.1, ls="--")
    ax.text(88, 1.05, "RF preselector\n(tracks the tuning)", fontsize=7, color=GREEN)
    ax.set_ylim(0, 1.35); ax.set_yticks([]); ax.set_xlabel("frequency (MHz)")
    ax.set_title("High-side LO in an FM superhet: the image lies 2 IF above the wanted station")
    fig.tight_layout(); save(fig, "ch04_superhet")


if __name__ == "__main__":
    fig_am(); fig_am_family(); fig_ssb_phase_error(); fig_fm_bessel(); fig_carson(); fig_fm_threshold(); fig_stereo_mpx(); fig_superhet()
