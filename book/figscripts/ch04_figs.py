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




# ---------------------------------------------------------------------------------------------
# Figures added in the deepening pass
# ---------------------------------------------------------------------------------------------

def diode_detector(x, fs, rc):
    """Ideal-diode peak detector with an RC load: charges instantly, discharges with time constant rc."""
    a = np.exp(-1 / (fs * rc))
    v = np.empty_like(x); vp = 0.0
    for i, xi in enumerate(x):
        vp = max(xi, vp * a)
        v[i] = vp
    return v


def fig_envelope_clip():
    fs = 4e6; fc = 50e3; fm = 1e3; mu = 0.8
    t = np.arange(0, 2.2e-3, 1 / fs)
    env = 1 + mu * np.cos(2 * np.pi * fm * t)
    x = env * np.cos(2 * np.pi * fc * t)
    lim = np.sqrt(1 - mu ** 2) / (2 * np.pi * fm * mu)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    cases = [(8e-6, "RC = 8 $\\mu$s: too short (ripple)"),
             (60e-6, "RC = 60 $\\mu$s: about right"),
             (500e-6, "RC = 500 $\\mu$s: diagonal clipping")]
    for a, (rc, title) in zip(ax, cases):
        v = diode_detector(x, fs, rc)
        a.plot(t * 1e3, x, color=NAVY, lw=0.25, alpha=0.5)
        a.plot(t * 1e3, env, color=GREEN, lw=1.0, ls="--")
        a.plot(t * 1e3, v, color=ACCENT, lw=1.1)
        a.set_title(title, fontsize=8); a.set_xlabel("time (ms)")
        a.set_xlim(0.2, 2.2)
    ax[0].set_ylim(-0.2, 1.95)
    fig.suptitle(f"Diode detector, $f_c$ = 50 kHz, 1 kHz tone, $\\mu$ = 0.8: clipping limit RC $\\leq$ {lim*1e6:.0f} $\\mu$s",
                 fontsize=8.5, y=1.0)
    fig.tight_layout(); save(fig, "ch04_envelope_clip")


def two_sided_psd(x, fs, nper=4096):
    f, p = sps.welch(x, fs, nperseg=nper, return_onesided=False)
    f = np.fft.fftshift(f); p = np.fft.fftshift(p)
    return f, 10 * np.log10(p / p.max() + 1e-12)


def fig_weaver():
    r = rng(11); fs = 48e3
    m = voice_like(fs, 6.0, r)
    m = lowpass(m, 3000, fs, 8)
    t = np.arange(len(m)) / fs
    f0, fc = 1650.0, 12e3
    b1 = m * np.exp(-2j * np.pi * f0 * t)
    sos = sps.butter(10, 1400, fs=fs, output="sos")
    b2 = sps.sosfilt(sos, b1.real) + 1j * sps.sosfilt(sos, b1.imag)
    y = np.real(b2 * np.exp(2j * np.pi * (fc + f0) * t))
    stages = [(m, "(a) message $m(t)$ (real: both halves)"),
              (b1, "(b) after mixing with $e^{-j2\\pi f_0 t}$, $f_0$ = 1650 Hz"),
              (b2, "(c) after the two low-pass filters"),
              (y, "(d) output: $\\mathrm{Re}\\{\\cdot\\,e^{j2\\pi(f_c+f_0)t}\\}$, USB")]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.6))
    for a, (x, title) in zip(ax.ravel(), stages):
        f, p = two_sided_psd(x, fs)
        a.plot(f / 1e3, p, color=NAVY, lw=0.8)
        a.set_title(title, fontsize=8); a.set_ylim(-70, 3)
        a.axvline(0, color=GRAY, lw=0.6)
    for a in ax[0]: a.set_xlim(-6, 6)
    ax[1, 0].set_xlim(-6, 6)
    ax[1, 1].set_xlim(-18, 18)
    ax[0, 1].axvspan(-1.4, 1.4, color=GREEN, alpha=0.12)
    ax[1, 1].axvline(12, color=ACCENT, ls=":", lw=0.8); ax[1, 1].axvline(-12, color=ACCENT, ls=":", lw=0.8)
    for a in ax[1]: a.set_xlabel("frequency (kHz)")
    ax[0, 0].set_ylabel("dB"); ax[1, 0].set_ylabel("dB")
    fig.tight_layout(); save(fig, "ch04_weaver")


def fig_fm_spectra():
    r = rng(12); fs = 200e3; n = 2 ** 20
    t = np.arange(n) / fs
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    # (a) tone FM, beta = 5, simulated PSD versus Bessel lines
    fm, beta = 2e3, 5.0
    z = np.exp(1j * beta * np.sin(2 * np.pi * fm * t))
    Z = np.fft.fftshift(np.abs(np.fft.fft(z[:2 ** 16] * np.hanning(2 ** 16))))
    f = np.fft.fftshift(np.fft.fftfreq(2 ** 16, 1 / fs))
    Z = Z / Z.max() * np.max(np.abs(jv(np.arange(-12, 13), beta)))
    ax[0].plot(f / 1e3, 20 * np.log10(Z + 1e-9), color=NAVY, lw=0.7, label="simulated (FFT)")
    k = np.arange(-12, 13)
    ax[0].plot(k * fm / 1e3, 20 * np.log10(np.abs(jv(k, beta)) + 1e-9), "o", color=ACCENT, ms=3,
               mfc="none", label="$|J_n(5)|$")
    ax[0].axvspan(-(beta + 1) * fm / 1e3, (beta + 1) * fm / 1e3, color=GREEN, alpha=0.1, label="Carson band")
    ax[0].set_ylim(-60, 0); ax[0].set_xlim(-26, 26)
    ax[0].set_xlabel("$f-f_c$ (kHz)"); ax[0].set_ylabel("dB"); ax[0].legend(fontsize=6.5, loc="lower center")
    ax[0].set_title("Tone FM, $\\beta$ = 5, $f_m$ = 2 kHz", fontsize=8.5)
    # (b) Gaussian message: Woodward's theorem
    w = r.standard_normal(n)
    msg = lowpass(w, 1e3, fs, 8); msg /= msg.std()
    for sig_f, c, lab in [(10e3, NAVY, "rms deviation 10 kHz"), (1e3, ORANGE, "rms deviation 1 kHz")]:
        z = np.exp(2j * np.pi * sig_f * np.cumsum(msg) / fs)
        ff, p = sps.welch(z, fs, nperseg=8192, return_onesided=False)
        ff = np.fft.fftshift(ff); p = np.fft.fftshift(p)
        ax[1].plot(ff / 1e3, p / np.trapezoid(p, ff / 1e3), color=c, lw=0.9, label=lab)
    g = np.exp(-0.5 * (ff / 10e3) ** 2) / (np.sqrt(2 * np.pi) * 10)
    ax[1].plot(ff / 1e3, g, color=ACCENT, ls="--", lw=1, label="Gaussian pdf of $k_f m(t)$")
    ax[1].set_xlim(-40, 40); ax[1].set_xlabel("$f-f_c$ (kHz)"); ax[1].set_ylabel("PSD (normalised)")
    ax[1].legend(fontsize=6.5); ax[1].set_ylim(0, 0.12)
    ax[1].set_title("Gaussian message, 1 kHz bandwidth", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_fm_spectra")


def fig_discriminators():
    f0 = 10.7e6
    df = np.linspace(-250e3, 250e3, 1001)
    f = f0 + df
    def tuned(fr, Q):
        return 1 / np.sqrt(1 + (Q * (f / fr - fr / f)) ** 2)
    Q = 35
    slope = tuned(f0 + 220e3, Q); slope = slope - np.interp(0, df, slope)
    bal = tuned(f0 + 120e3, Q) - tuned(f0 - 120e3, Q)
    QL = 25
    quad = np.sin(np.arctan(QL * (f / f0 - f0 / f)))
    def norm(y):
        s = np.gradient(y, df)[500]
        return y / (s * 75e3)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.6))
    ax.axvspan(-75, 75, color=GREEN, alpha=0.08)
    ax.plot(df / 1e3, df / 75e3, color=GRAY, lw=1, ls=":", label="ideal (linear)")
    ax.plot(df / 1e3, norm(slope), color=ORANGE, label="slope detector (one tuned circuit)")
    ax.plot(df / 1e3, norm(bal), color=NAVY, label="balanced (Foster-Seeley / ratio)")
    ax.plot(df / 1e3, norm(quad), color=ACCENT, label="quadrature detector ($Q_L$ = 25)")
    ax.set_ylim(-2.2, 2.2); ax.set_xlim(-250, 250)
    ax.text(0, -1.95, "$\\pm$75 kHz broadcast deviation", ha="center", fontsize=7, color=GREEN)
    ax.set_xlabel("frequency offset from 10.7 MHz IF (kHz)"); ax.set_ylabel("output (normalised)")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_title("Discriminator S-curves", fontsize=9)
    fig.tight_layout(); save(fig, "ch04_discriminators")


def fig_preemphasis():
    f = np.logspace(np.log10(20), np.log10(20e3), 400)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for tau, c in [(75e-6, NAVY), (50e-6, ACCENT)]:
        g = 10 * np.log10(1 + (2 * np.pi * f * tau) ** 2)
        ax[0].semilogx(f, g, color=c, label=f"pre-emphasis {tau*1e6:.0f} $\\mu$s")
        ax[0].semilogx(f, -g, color=c, ls="--", lw=1, label=f"de-emphasis {tau*1e6:.0f} $\\mu$s")
    ax[0].set_xlabel("audio frequency (Hz)"); ax[0].set_ylabel("gain (dB)")
    ax[0].legend(fontsize=6.3, ncol=1, loc="upper left"); ax[0].set_ylim(-20, 20)
    ax[0].set_title("Broadcast emphasis curves", fontsize=8.5)
    fl = np.linspace(0, 15e3, 400); W = 15e3
    fl = np.linspace(50, 15e3, 400)
    ax[1].plot(fl / 1e3, 10 * np.log10((fl / W) ** 2), color=GRAY, label="FM output noise $\\propto f^2$")
    for tau, c in [(75e-6, NAVY), (50e-6, ACCENT)]:
        f1 = 1 / (2 * np.pi * tau)
        nd = (fl / W) ** 2 / (1 + (fl / f1) ** 2)
        x = W / f1
        imp = x ** 3 / (3 * (x - np.arctan(x)))
        ax[1].plot(fl / 1e3, 10 * np.log10(nd), color=c, label=f"after {tau*1e6:.0f} $\\mu$s de-emphasis: total {10*np.log10(imp):.1f} dB less")
    ax[1].set_xlabel("audio frequency (kHz)"); ax[1].set_ylabel("noise PSD (dB, relative)")
    ax[1].legend(fontsize=6.3, loc="lower right"); ax[1].set_ylim(-50, 3)
    ax[1].set_title("Effect on the parabolic noise, $W$ = 15 kHz", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_preemphasis")


def fig_fm_clicks():
    r = rng(13); fs = 400e3; fm = 1e3; beta = 5; fd = beta * fm
    T = 0.06; n = int(T * fs); t = np.arange(n) / fs
    m = np.cos(2 * np.pi * fm * t)
    x = np.exp(2j * np.pi * fd * np.cumsum(m) / fs)
    B = 2 * (beta + 1) * fm
    fig = plt.figure(figsize=(W2, 3.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    out = {}
    cut = slice(4000, 4000 + int(0.05 * fs))
    tt = (t[cut] - t[cut][0]) * 1e3
    for row, cnr in enumerate([16, 4]):
        n0 = 1 / 10 ** (cnr / 10) / B * fs
        y = x + np.sqrt(n0 / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))
        y = lowpass(y.real, B / 2, fs, 8) + 1j * lowpass(y.imag, B / 2, fs, 8)
        d = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi * fd)
        d = lowpass(np.r_[d[0], d], 4e3, fs, 6)
        out[cnr] = y
        a = fig.add_subplot(gs[row])
        a.plot(tt, d[cut], color=NAVY if cnr > 10 else ACCENT, lw=0.6)
        a.plot(tt, m[cut], color=GREEN, lw=0.7, ls="--")
        a.set_ylim(-3.2, 3.2); a.set_xlim(0, 50); a.set_xticklabels([])
        a.set_ylabel("audio", fontsize=8)
        a.text(49.6, 2.0, f"CNR = {cnr} dB", ha="right", fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85))
    y = out[4]
    a = fig.add_subplot(gs[2])
    pe = np.unwrap(np.angle(y * np.conj(x)))[cut] / (2 * np.pi)
    a.plot(tt, pe - np.round(pe[0]), color=ACCENT, lw=0.7)
    a.set_xlabel("time (ms)"); a.set_ylabel("phase err.\n(cycles)", fontsize=8)
    a.text(49.6, a.get_ylim()[0] + 0.1, "CNR = 4 dB: each click is a whole-cycle slip", ha="right", va="bottom", fontsize=7.5,
           bbox=dict(fc="white", ec="none", alpha=0.85))
    a.set_xlim(0, 50)
    fig.tight_layout(); save(fig, "ch04_fm_clicks")


def fig_capture():
    fs = 400e3; n = 2 ** 16; t = np.arange(n) / fs
    f1, f2, fd = 1000.0, 1700.0, 15e3
    m1 = np.sin(2 * np.pi * f1 * t); m2 = np.sin(2 * np.pi * f2 * t)
    sl = slice(4000, -4000)
    def sir(d):
        d = lowpass(d - d.mean(), 4e3, fs, 8)[sl]; ref = m1[1:][sl]
        a = np.dot(d, ref) / np.dot(ref, ref)
        return 10 * np.log10(np.mean((a * ref) ** 2) / np.mean((d - a * ref) ** 2))
    sirs = np.r_[np.linspace(-10, -0.25, 25), np.linspace(0.25, 20, 40)]
    fm_out, am_out = [], []
    for s in sirs:
        g = 10 ** (-s / 20)
        z = np.exp(2j * np.pi * fd * np.cumsum(m1) / fs) + g * np.exp(2j * np.pi * fd * np.cumsum(m2) / fs + 0.9j) * np.exp(2j * np.pi * 500 * t)
        d = np.angle(z[1:] * np.conj(z[:-1]))
        fm_out.append(sir(d))
        za = (1 + 0.5 * m1) + g * (1 + 0.5 * m2) * np.exp(1j * (0.9 + 2 * np.pi * 500 * t))
        da = np.abs(za)[1:]
        am_out.append(sir(da))
    fig, ax = plt.subplots(figsize=(W1 * 0.78, 2.5))
    ax.plot(sirs, fm_out, color=NAVY, label="FM ($\\Delta f$ = 15 kHz), discriminator")
    ax.plot(sirs, am_out, color=ACCENT, ls="--", label="AM ($\\mu$ = 0.5), envelope detector")
    ax.plot(sirs, sirs, color=GRAY, ls=":", lw=1, label="output = input")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("input carrier ratio, wanted/unwanted (dB)")
    ax.set_ylabel("output audio SIR (dB)")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("The capture effect (simulation, 500 Hz carrier offset)", fontsize=9)
    fig.tight_layout(); save(fig, "ch04_capture")


def irr_db(fs_, fi, Q, stages=1):
    rho = fi / fs_ - fs_ / fi
    return stages * 10 * np.log10(1 + (Q * rho) ** 2)


def fig_image_rejection():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    f = np.linspace(530, 1700, 300)
    for IF, Q, st, c, ls in [(455, 40, 1, NAVY, "-"), (455, 40, 2, NAVY, "--"), (262, 40, 1, ORANGE, "-")]:
        ax[0].plot(f, irr_db(f, f + 2 * IF, Q, st), color=c, ls=ls,
                   label=f"IF {IF} kHz, {st} tuned circuit{'s' if st > 1 else ''}, Q = {Q}")
    ax[0].set_xlabel("tuned frequency (kHz)"); ax[0].set_ylabel("image rejection (dB)")
    ax[0].set_title("Medium-wave AM", fontsize=8.5); ax[0].legend(fontsize=6.3); ax[0].set_ylim(0, 90)
    f = np.linspace(88, 108, 200)
    for IF, Q, st, c, ls in [(10.7, 30, 1, NAVY, "-"), (10.7, 30, 2, NAVY, "--"), (10.7, 15, 1, ORANGE, "-")]:
        ax[1].plot(f, irr_db(f, f + 2 * IF, Q, st), color=c, ls=ls,
                   label=f"IF {IF} MHz, {st} tuned circuit{'s' if st > 1 else ''}, Q = {Q}")
    ax[1].set_xlabel("tuned frequency (MHz)")
    ax[1].set_title("VHF FM broadcast", fontsize=8.5); ax[1].legend(fontsize=6.3); ax[1].set_ylim(0, 90)
    fig.tight_layout(); save(fig, "ch04_image_rejection")


def fig_agc():
    fs = 8000.0; T = 3.0; t = np.arange(0, T, 1 / fs)
    lvl = np.where(t < 0.6, 0, np.where(t < 1.5, 30, np.where(t < 2.2, -10, 5)))  # dB steps
    lvl = lvl + 4 * np.sin(2 * np.pi * 1.3 * t)                                  # slow fade
    mod = 1 + 0.6 * np.sin(2 * np.pi * 40 * t)                                   # audio (slowed to 40 Hz for visibility)
    env_in = 10 ** (lvl / 20) * mod
    def run(det_tc, att_tc, dec_tc):
        """Feedback AGC: averaging detector on the output, dB-domain integrator with attack/decay rates."""
        kdet = 1 / (det_tc * fs); ka, kd = 1 / (att_tc * fs), 1 / (dec_tc * fs)
        g = 0.0; out = np.empty_like(env_in); det = 1.0
        for i, e in enumerate(env_in):
            o = e * 10 ** (g / 20); out[i] = o
            det += kdet * (o - det)                              # averaging (AVC) detector
            err = 20 * np.log10(max(det, 1e-6))                  # level error in dB (reference 1)
            g -= (ka if err > 0 else kd) * err                   # fast attack, slow decay
        return out
    good = run(0.03, 0.06, 0.25); fast = run(0.001, 0.001, 0.002)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.6), sharex=True)
    ax[0].plot(t, env_in, color=NAVY, lw=0.4); ax[0].set_yscale("log"); ax[0].set_ylabel("input", fontsize=8)
    ax[0].set_title("AGC on an AM envelope: +30 dB, $-$40 dB and +15 dB steps with slow fading", fontsize=8.5)
    ax[1].plot(t, good, color=GREEN, lw=0.4); ax[1].set_ylabel("output", fontsize=8)
    ax[1].text(2.98, 2.75, "detector 30 ms, attack 60 ms, decay 250 ms: level held, modulation kept", ha="right", fontsize=7)
    ax[2].plot(t, fast, color=ACCENT, lw=0.4); ax[2].set_ylabel("output", fontsize=8)
    ax[2].text(2.98, 2.75, "detector 1 ms, attack 1 ms, decay 2 ms: AGC flattens the modulation", ha="right", fontsize=7)
    for a in ax[1:]:
        a.set_ylim(0, 3.3)
        for tx in a.texts: tx.set_bbox(dict(fc="white", ec="none", alpha=0.85))
    ax[2].set_xlabel("time (s)")
    fig.tight_layout(); save(fig, "ch04_agc")


def fig_ip3():
    fs = 1.0; n = 4096; t = np.arange(n)
    f1, f2 = 400 / n, 440 / n
    a1, a3 = 10.0, -2.0                      # gain 20 dB, compressive cubic
    pin = np.arange(-40, -2.5, 1.0)           # dB relative to 1 (per tone, amplitude A: P = A^2/2)
    pf, pi3 = [], []
    for p in pin:
        A = np.sqrt(2 * 10 ** (p / 10))
        x = A * (np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t))
        y = a1 * x + a3 * x ** 3
        Y = np.abs(np.fft.rfft(y)) / (n / 2)
        pf.append(10 * np.log10(Y[400] ** 2 / 2)); pi3.append(10 * np.log10(Y[360] ** 2 / 2 + 1e-30))
    pf, pi3 = np.array(pf), np.array(pi3)
    iip3 = 10 * np.log10(0.5 * 4 / 3 * abs(a1 / a3))
    G = 20 * np.log10(a1)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.8))
    ax.plot(pin, pf, "o", color=NAVY, ms=2.5, label="fundamental (simulated)")
    ax.plot(pin, pi3, "s", color=ACCENT, ms=2.5, label="IM3 at $2f_1-f_2$ (simulated)")
    xl = np.linspace(-40, iip3 + 3, 50)
    ax.plot(xl, xl + G, color=NAVY, lw=0.8, ls="--")
    ax.plot(xl, 3 * xl - 2 * iip3 + G, color=ACCENT, lw=0.8, ls="--")
    ax.plot(iip3, iip3 + G, "*", color="k", ms=8)
    ax.annotate(f"intercept point\nIIP3 = {iip3:.1f} dB, OIP3 = {iip3+G:.1f} dB", (iip3, iip3 + G), (-30, 28),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    floor = -60
    ax.axhline(floor, color=GRAY, lw=1, ls=":")
    ax.text(-39, floor + 2, "output noise floor", fontsize=7, color=GRAY)
    # SFDR: input level where IM3 = floor
    pin_sf = (floor - G + 2 * iip3) / 3
    ax.annotate("", (pin_sf, floor), (pin_sf, pin_sf + G), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1))
    ax.text(pin_sf + 0.8, (floor + pin_sf + G) / 2, f"SFDR = {pin_sf + G - floor:.0f} dB", fontsize=7, color=GREEN)
    ax.set_xlabel("input power per tone (dB)"); ax.set_ylabel("output power per tone (dB)")
    ax.set_ylim(-90, 40); ax.set_xlim(-40, 12); ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("Two-tone test of $y = a_1x + a_3x^3$", fontsize=9)
    fig.tight_layout(); save(fig, "ch04_ip3")


def ntsc_line(fs=100e6):
    """One NTSC composite line (75% colour bars), in IRE units."""
    fh = 4.5e6 / 286; fsc = 455 / 2 * fh; H = 1 / fh
    t = np.arange(0, H, 1 / fs)
    v = np.zeros_like(t)
    fp, sync, bw_ = 1.5e-6, 4.7e-6, 0.6e-6
    v[(t >= fp) & (t < fp + sync)] = -40
    b0 = fp + sync + bw_; nb = 9 / fsc
    burst = (t >= b0) & (t < b0 + nb)
    v[burst] = 20 * np.sin(2 * np.pi * fsc * t[burst] + np.pi)   # burst at 180 degrees on the B-Y axis
    a0 = 10.9e-6; a1 = H - 0.0e-6 - 1.5e-6 + 1.5e-6 - 0.0e-6
    a1 = a0 + 52.6e-6
    bars = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 0)]
    act = (t >= a0) & (t < a1)
    idx = np.minimum(((t[act] - a0) / (a1 - a0) * 8).astype(int), 7)
    rgb = 0.75 * np.array(bars)[idx]
    Y = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    U = 0.493 * (rgb[:, 2] - Y); V = 0.877 * (rgb[:, 0] - Y)
    ph = 2 * np.pi * fsc * t[act]
    c = U * np.sin(ph) + V * np.cos(ph)
    v[act] = 7.5 + 92.5 * Y + 92.5 * c
    return t, v, fsc, b0, nb, a0


def fig_ntsc_line():
    t, v, fsc, b0, nb, a0 = ntsc_line()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(t * 1e6, v, color=NAVY, lw=0.3)
    ax[0].set_xlabel("time ($\\mu$s)"); ax[0].set_ylabel("IRE units"); ax[0].set_ylim(-50, 135)
    ax[0].set_title("One line of NTSC colour bars (75%), 63.56 $\\mu$s", fontsize=8.5)
    names = ["white", "yellow", "cyan", "green", "magenta", "red", "blue", "black"]
    for i, nme in enumerate(names):
        ax[0].text(a0 * 1e6 + (i + 0.5) * 52.6 / 8, 125, nme, ha="center", fontsize=5.8, rotation=0)
    ax[0].annotate("sync", (3.8, -40), (3, -10), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].annotate("burst", ((b0 + nb / 2) * 1e6, 20), (9, 60), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    m = (t > 5e-6) & (t < 15e-6)
    ax[1].plot(t[m] * 1e6, v[m], color=NAVY, lw=0.6)
    ax[1].set_xlabel("time ($\\mu$s)"); ax[1].set_ylim(-50, 135)
    ax[1].set_title("Zoom: sync, 9-cycle burst,\nfirst bars", fontsize=8)
    fig.tight_layout(); save(fig, "ch04_ntsc_line")


def fig_ntsc_spectrum():
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.8), gridspec_kw={"height_ratios": [1.15, 1]})
    a = ax[0]
    f = np.linspace(0, 6, 3000)
    fv = 1.25
    vid = np.where(f < fv - 0.75, 0, np.where(f < fv - 0.5, (f - (fv - 0.75)) / 0.25, 1.0))
    vid = np.where(f > fv + 4.2, np.clip(1 - (f - fv - 4.2) / 0.3, 0, 1), vid)
    vid = vid * np.where(f < fv + 0.75, 1, np.exp(-(f - fv - 0.75) / 2.5))
    a.fill_between(f, 0, 0.85 * vid, color=NAVY, alpha=0.35, lw=0)
    a.vlines(fv, 0, 1.0, color=NAVY, lw=2); a.text(fv, 1.04, "picture carrier\n1.25 MHz", ha="center", fontsize=7)
    fsc = fv + 3.579545
    chroma = 0.3 * np.exp(-((f - fsc) / 0.45) ** 2) * (f < fsc + 0.6)
    a.fill_between(f, 0, chroma, color=ACCENT, alpha=0.5, lw=0)
    a.vlines(fsc, 0, 0.5, color=ACCENT, lw=1, ls=":"); a.text(fsc, 0.55, "colour\nsubcarrier\n+3.58 MHz", ha="center", fontsize=6.5, color=ACCENT)
    fa = fv + 4.5
    a.fill_between(f, 0, 0.45 * np.exp(-((f - fa) / 0.025) ** 2), color=GREEN, lw=0)
    a.text(fa + 0.08, 0.5, "FM sound\n+4.5 MHz", ha="left", fontsize=6.5, color=GREEN)
    a.annotate("", (fv - 0.75, 0.15), (fv, 0.15), arrowprops=dict(arrowstyle="<->", lw=0.6))
    a.text(fv - 0.4, 0.2, "vestige\n0.75", ha="center", fontsize=6)
    a.annotate("", (fv, -0.12), (fv + 4.2, -0.12), arrowprops=dict(arrowstyle="<->", lw=0.6), annotation_clip=False)
    a.text(fv + 2.1, -0.08, "luminance 4.2 MHz", ha="center", fontsize=6.5)
    a.set_xlim(0, 6); a.set_ylim(-0.2, 1.35); a.set_yticks([])
    a.set_xlabel("frequency above lower channel edge (MHz)")
    a.set_title("The 6 MHz NTSC channel (System M)", fontsize=8.5)
    # bottom: interleaving
    b = ax[1]
    fh = 4.5e6 / 286
    n0 = 227
    for k in range(n0 - 3, n0 + 5):
        for j in range(-3, 4):
            b.vlines((k + j * 0.06) , 0, (0.9 if j == 0 else 0.35 / (1 + abs(j))) * np.exp(-0.15 * abs(k - n0)), color=NAVY, lw=1.6 if j == 0 else 0.8)
        kc = k + 0.5
        for j in range(-3, 4):
            b.vlines(kc + j * 0.06, 0, (0.6 if j == 0 else 0.25 / (1 + abs(j))) * np.exp(-0.25 * abs(kc - 227.5)), color=ACCENT, lw=1.6 if j == 0 else 0.8)
    b.vlines(227.5, 0, 0.75, color=ACCENT, ls=":", lw=0.8)
    b.annotate("$f_{sc}=227.5f_H$ (suppressed)", (227.5, 0.62), (229.0, 1.05), fontsize=6.8, color=ACCENT,
               arrowprops=dict(arrowstyle="->", lw=0.6, color=ACCENT))
    b.set_xlim(223.6, 231.4); b.set_ylim(0, 1.3); b.set_yticks([])
    b.set_xlabel("frequency in units of the line rate $f_H$ = 15 734.27 Hz (around 3.58 MHz)")
    b.plot([], [], color=NAVY, label="luminance: clusters at $nf_H$")
    b.plot([], [], color=ACCENT, label="chrominance: clusters at $(n+\\frac{1}{2})f_H$")
    b.legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.tight_layout(); save(fig, "ch04_ntsc_spectrum")


def fig_fdm():
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.3))
    def chan(a, lo, hi, inverted, c, h=1.0):
        if inverted:
            a.fill([lo, hi, hi, lo], [0, 0, 0.25 * h, h], color=c, alpha=0.75, lw=0)
        else:
            a.fill([lo, hi, hi, lo], [0, 0, h, 0.25 * h], color=c, alpha=0.75, lw=0)
    a = ax[0]
    for i in range(12):
        carrier = 64 + 4 * i                 # 64, 68, ..., 108 kHz; lower sideband kept
        chan(a, carrier - 3.4, carrier - 0.3, True, NAVY if i % 2 == 0 else "#2E86C1")
        a.text(carrier - 1.85, 1.04, f"{i+1}", ha="center", fontsize=6)
    a.set_xlim(58, 110); a.set_ylim(0, 1.3); a.set_yticks([])
    a.set_xlabel("frequency (kHz)")
    a.set_title("Basic group: 12 voice channels, lower sidebands of carriers 64, 68, ..., 108 kHz (60-108 kHz)", fontsize=8)
    b = ax[1]
    cols = [NAVY, GREEN, ORANGE, PURPLE, ACCENT]
    for g in range(5):
        fcg = 420 + 48 * g                  # group carriers 420 ... 612 kHz, lower sideband of 60-108
        lo, hi = fcg - 108, fcg - 60
        # the inverted group: original group was inverted channels -> now upright
        for i in range(12):
            c0 = lo + 4 * (11 - i)
            chan(b, c0 + 0.6, c0 + 3.7, False, cols[g], 0.8)
        b.text((lo + hi) / 2, 0.95, f"group {g+1}\ncarrier {fcg} kHz", ha="center", fontsize=6.3)
    b.set_xlim(300, 565); b.set_ylim(0, 1.3); b.set_yticks([])
    b.set_xlabel("frequency (kHz)")
    b.set_title("Basic supergroup: five groups translated to 312-552 kHz (60 channels)", fontsize=8)
    fig.tight_layout(); save(fig, "ch04_fdm")


# ---------------------------------------------------------------------------------------------
# Second edition: concept illustrations, infographics and extra data figures
# ---------------------------------------------------------------------------------------------
SIDE = (3.0, 2.4)          # narrow single-panel figures for \mdcside


def fig_timeline():
    ev = [(1906, "Fessenden: AM voice\nPickard: crystal detector", 1),
          (1912, "Armstrong:\nregeneration", -1),
          (1918, "Armstrong:\nsuperheterodyne", 2.2),
          (1920, "KDKA: scheduled\nbroadcasting", -2.3),
          (1927, "Transatlantic SSB\nradiotelephone", 1),
          (1933, "Armstrong's\nFM patents", -1),
          (1941, "NTSC monochrome TV;\ncommercial FM", 2.2),
          (1953, "NTSC compatible\ncolour", -2.3),
          (1954, "Regency TR-1\ntransistor radio", 1),
          (1961, "FM stereo\n(pilot tone)", -1),
          (1983, "AMPS analog\ncellular", 2.2),
          (2009, "US analog TV\nswitched off", -2.3),
          (2017, "Norway switches\noff national FM", 1)]
    fig, ax = plt.subplots(figsize=(W2, 3.3))
    ax.axhline(0, color=NAVY, lw=2.2, zorder=1)
    for era, (a, b), c in [("spark & crystal", (1900, 1920), GRAY), ("golden age of analog radio & TV", (1920, 1962), ORANGE),
                            ("digital transition", (1962, 2030), GREEN)]:
        ax.fill_between([a, b], -0.18, 0.18, color=c, alpha=0.35, lw=0, zorder=0)
        ax.text((a + b) / 2, -0.38 if era != "golden age of analog radio & TV" else 0.3, era, ha="center",
                fontsize=6.8, color=c, style="italic", zorder=3)
    for yr, txt, h in ev:
        ax.plot([yr, yr], [0, h * 0.82], color=GRAY, lw=0.7, zorder=1)
        ax.plot(yr, 0, "o", color=ACCENT, ms=4.5, zorder=4)
        ax.text(yr, h * 0.85 + (0.05 if h > 0 else -0.05), f"{yr}\n{txt}", ha="center",
                va="bottom" if h > 0 else "top", fontsize=7.0, linespacing=1.05,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=NAVY, lw=0.5))
    ax.set_xlim(1895, 2029); ax.set_ylim(-3.6, 3.5)
    ax.axis("off")
    fig.tight_layout(); save(fig, "ch04_timeline")


def fig_spectrum_map():
    """Where the classic analog services live, on a log frequency axis."""
    bands = [  # (lo, hi) in Hz, row, label, colour
        (148.5e3, 283.5e3, 0, "LW AM", NAVY), (530e3, 1700e3, 0, "MW AM", NAVY),
        (2.3e6, 26.1e6, 0, "SW broadcast (AM)", NAVY),
        (1.8e6, 30e6, 1, "HF SSB: marine, aero, military, amateur", PURPLE),
        (54e6, 88e6, 2, "TV 2-6", ORANGE), (174e6, 216e6, 2, "TV 7-13", ORANGE),
        (470e6, 806e6, 2, "UHF TV 14-69", ORANGE),
        (88e6, 108e6, 0, "FM", ACCENT), (118e6, 137e6, 1, "aviation AM", GREEN),
        (156e6, 162e6, 3, "marine FM", GREEN), (824e6, 894e6, 3, "AMPS", PURPLE)]
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    for lo, hi, row, lab, c in bands:
        ax.barh(1.25 * row, hi - lo, left=lo, height=0.62, color=c, alpha=0.8)
        ax.text(np.sqrt(lo * hi), 1.25 * row + 0.36, lab, ha="center", va="bottom", fontsize=6.4, color=c,
                linespacing=1.0)
    ax.set_xscale("log"); ax.set_xlim(1e5, 1.5e9); ax.set_ylim(-1.0, 4.6); ax.set_yticks([])
    ax.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9]); ax.set_xticklabels(["100 kHz", "1 MHz", "10 MHz", "100 MHz", "1 GHz"])
    ax.grid(axis="y", visible=False)
    sec = ax.secondary_xaxis("top", functions=(lambda f: 3e8 / np.maximum(f, 1), lambda l: 3e8 / np.maximum(l, 1e-9)))
    sec.set_xticks([3000, 300, 30, 3, 0.3]); sec.set_xticklabels(["3 km", "300 m", "30 m", "3 m", "30 cm"])
    sec.set_xlabel("wavelength", fontsize=8)
    for name, lo, hi in [("LF", 3e4, 3e5), ("MF", 3e5, 3e6), ("HF", 3e6, 3e7), ("VHF", 3e7, 3e8), ("UHF", 3e8, 3e9)]:
        ax.axvline(lo, color=GRAY, lw=0.5, ls=":")
        ax.text(np.sqrt(max(lo, 1e5) * min(hi, 1.5e9)), -0.45, name, ha="center", va="top", fontsize=7, color=GRAY)
    fig.tight_layout(); save(fig, "ch04_spectrum_map")


def fig_wavelengths():
    items = [("1 kHz audio", 1e3, "75 km of wire"), ("1 MHz (MW AM)", 1e6, "75 m mast"),
             ("100 MHz (FM)", 1e8, "75 cm whip"), ("2.4 GHz (Wi-Fi)", 2.4e9, "3 cm, inside a phone")]
    fig, ax = plt.subplots(figsize=SIDE)
    for i, (lab, f, note) in enumerate(items):
        q = 3e8 / f / 4
        ax.barh(i, q, color=[NAVY, ORANGE, ACCENT, GREEN][i], alpha=0.85, height=0.6)
        ax.text(q * 1.6, i, note, va="center", fontsize=7.3)
    ax.set_xscale("log"); ax.set_xlim(0.01, 3e7)
    ax.set_yticks(range(4)); ax.set_yticklabels([x[0] for x in items], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("quarter-wave antenna length (m)")
    ax.set_xticks([0.01, 1, 100, 1e5]); ax.set_xticklabels(["1 cm", "1 m", "100 m", "100 km"])
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch04_wavelengths")


def fig_two_knobs():
    t = np.linspace(0, 1, 6000)
    m = 0.65 * np.sin(2 * np.pi * 2 * t) + 0.35 * np.sin(2 * np.pi * 5 * t + 0.6)
    m /= np.max(np.abs(m))
    fc = 40
    am = (1 + 0.8 * m) * np.cos(2 * np.pi * fc * t)
    fm = np.cos(2 * np.pi * fc * t + 2 * np.pi * 18 * np.cumsum(m) / len(t))
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.4), sharex=True)
    ax[0].plot(t, m, color=GREEN, lw=1.6); ax[0].set_title("the message: a voice or a melody, $m(t)$", fontsize=8.5, loc="left")
    ax[1].plot(t, am, color=NAVY, lw=0.6); ax[1].plot(t, 1 + 0.8 * m, color=GREEN, ls="--", lw=1.1)
    ax[1].set_title("AM: the message turns the volume knob (the envelope copies $m(t)$; the rhythm never changes)",
                    fontsize=8.5, loc="left")
    ax[2].plot(t, fm, color=ACCENT, lw=0.6)
    ax[2].set_title("FM: the message wiggles the pitch (crests crowd together where $m(t)$ is high; loudness never changes)",
                    fontsize=8.5, loc="left")
    for a in ax:
        a.set_yticks([]); a.grid(False)
    ax[2].set_xlabel("time")
    ax[2].set_xticks([])
    fig.tight_layout(h_pad=0.4); save(fig, "ch04_two_knobs")


def _arrow(ax, x0, y0, x1, y1, c, lw=1.8, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls,
                                                        shrinkA=0, shrinkB=0, mutation_scale=9))


def fig_phasors():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    th = np.deg2rad(50)
    for a, kind in zip(ax, ["AM", "NBFM"]):
        a.set_aspect("equal"); a.set_xlim(-0.1, 2.0); a.set_ylim(-0.8, 0.85); a.axis("off")
        _arrow(a, 0, 0, 1, 0, NAVY, 2.2)
        a.text(0.45, -0.1, "carrier", fontsize=7.5, color=NAVY, ha="center", va="top")
        s = 0.35
        u = (s * np.cos(th), s * np.sin(th))                      # upper sideband, rotating +
        if kind == "AM":
            l = (s * np.cos(-th), s * np.sin(-th))                # lower sideband, rotating -
        else:
            l = (-s * np.cos(-th), -s * np.sin(-th))              # inverted lower sideband
        _arrow(a, 1, 0, 1 + u[0], u[1], GREEN, 1.5)
        _arrow(a, 1 + u[0], u[1], 1 + u[0] + l[0], u[1] + l[1], ORANGE, 1.5)
        tip = (1 + u[0] + l[0], u[1] + l[1])
        _arrow(a, 0, 0, tip[0], tip[1], ACCENT, 1.3, "--")
        cc = plt.Circle((1, 0), s, fill=False, ls=":", color=GRAY, lw=0.8); a.add_patch(cc)
        a.text(1 + u[0] + 0.03, u[1] + 0.05, "USB ($+f_m$)", fontsize=7, color=GREEN)
        a.text(tip[0] + 0.05, tip[1] - 0.12 if kind == "AM" else tip[1] + 0.04,
               "LSB ($-f_m$)", fontsize=7, color=ORANGE)
        if kind == "AM":
            a.plot([1 - 2 * s, 1 + 2 * s], [0, 0], color=ACCENT, lw=4, alpha=0.25)
            a.set_title("AM: sidebands sum ALONG the carrier\n(length changes, angle does not)", fontsize=8.5)
        else:
            a.plot([1, 1], [-2 * s, 2 * s], color=ACCENT, lw=4, alpha=0.25)
            a.set_title("Narrowband FM: lower sideband flipped, sum is\nPERPENDICULAR (angle changes, length barely)",
                        fontsize=8.5)
        a.text(1.62, -0.3 if kind == "AM" else 0.72, "path of the\nresultant tip", fontsize=6.8, color=ACCENT, ha="center")
    fig.tight_layout(); save(fig, "ch04_phasors")


def fig_am_power():
    cases = [("AM, tone, $\\mu$=1", 2 / 3), ("AM, speech, $P_m$=0.15", 1 / 1.15), ("DSB-SC / SSB", 0.0)]
    fig, ax = plt.subplots(figsize=SIDE)
    for i, (lab, pc) in enumerate(cases):
        ax.barh(i, pc * 100, color=GRAY, alpha=0.7, height=0.55)
        ax.barh(i, (1 - pc) * 100, left=pc * 100, color=ACCENT, alpha=0.85, height=0.55)
        if pc > 0:
            ax.text(pc * 50, i, f"carrier {pc*100:.0f}%", ha="center", va="center", fontsize=7, color="white")
        ax.text(pc * 100 + (1 - pc) * 50 if pc < 0.8 else 101, i, f"{(1-pc)*100:.0f}%",
                ha="center" if pc < 0.8 else "left", va="center", fontsize=7, color="white" if pc < 0.8 else ACCENT)
    ax.set_yticks(range(3)); ax.set_yticklabels([c[0] for c in cases], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlim(0, 112); ax.set_xlabel("share of transmitted power (%)")
    ax.set_title("Red = sidebands (the message)", fontsize=8, color=ACCENT)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch04_am_power")


def fig_trapezoid():
    t = np.linspace(0, 1, 20000)
    m = np.sin(2 * np.pi * 3 * t)
    c = np.cos(2 * np.pi * 110 * t)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1), sharey=True)
    for a, mu, title in zip(ax, [0.5, 1.0, 1.3], ["$\\mu$ = 0.5: trapezoid", "$\\mu$ = 1: triangle",
                                                   "$\\mu$ = 1.3: over-modulated"]):
        env = np.maximum(1 + mu * m, 0)
        a.plot(m, env * c, color=NAVY, lw=0.3)
        a.plot(m, env, color=ACCENT, lw=1.0); a.plot(m, -env, color=ACCENT, lw=1.0)
        a.set_title(title, fontsize=8.5); a.set_xlabel("audio $m(t)$ (horizontal)")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    ax[0].set_ylabel("RF (vertical)")
    fig.tight_layout(); save(fig, "ch04_trapezoid")


def fig_switching_mod():
    fs = 400e3; t = np.arange(0, 2e-3, 1 / fs); fc = 10e3
    m = 0.8 * np.sin(2 * np.pi * 1e3 * t)
    p = np.sign(np.cos(2 * np.pi * fc * t))
    y = m * p
    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1); a3 = fig.add_subplot(gs[:, 1])
    a1.plot(t * 1e3, p * 0.3 + 1.6, color=GRAY, lw=0.6); a1.plot(t * 1e3, m, color=GREEN, lw=1.3)
    a1.text(2.02, 1.6, "switch\n$\\pm1$ at $f_c$", fontsize=6.8, color=GRAY, va="center")
    a1.text(2.02, 0, "message", fontsize=6.8, color=GREEN, va="center")
    a1.set_yticks([]); a1.set_title("a reversing switch ...", fontsize=8.5, loc="left"); a1.set_xlim(0, 2.45)
    a2.plot(t * 1e3, y, color=NAVY, lw=0.6); a2.plot(t * 1e3, m, color=GREEN, lw=0.8, ls="--")
    a2.plot(t * 1e3, -m, color=GREEN, lw=0.8, ls="--")
    a2.set_yticks([]); a2.set_xlabel("time (ms)"); a2.set_title("... multiplies the message by the carrier", fontsize=8.5, loc="left")
    f = np.fft.rfftfreq(len(y) * 4, 1 / fs); Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), len(y) * 4))
    a3.plot(f / 1e3, 20 * np.log10(Y / Y.max() + 1e-6), color=NAVY, lw=0.8)
    for k in (1, 3, 5):
        a3.text(k * fc / 1e3, 4, f"{k}$f_c$", ha="center", fontsize=7, color=ACCENT)
    a3.set_xlim(0, 58); a3.set_ylim(-60, 10); a3.set_xlabel("frequency (kHz)"); a3.set_ylabel("dB")
    a3.set_title("spectrum: DSB-SC at $f_c$, $3f_c$, $5f_c$;\nno carrier, no message", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_switching_mod")


def fig_ssb_mistune():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
    t = np.linspace(0, 2, 4000)
    m = np.sin(2 * np.pi * 7 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t))
    ax[0].plot(t, m * np.cos(2 * np.pi * 1.0 * t), color=NAVY, lw=0.7)
    ax[0].plot(t, np.abs(np.cos(2 * np.pi * 1.0 * t)), color=ACCENT, ls="--", lw=0.9)
    ax[0].plot(t, -np.abs(np.cos(2 * np.pi * 1.0 * t)), color=ACCENT, ls="--", lw=0.9)
    ax[0].set_title("DSB-SC, LO 1 Hz off: the voice\nfades in and out twice a second", fontsize=8.5)
    ax[0].set_xlabel("time (s)"); ax[0].set_yticks([])
    f0 = 120; k = np.arange(1, 26); amp = np.exp(-((k * f0 - 700) / 500) ** 2) + 0.6 * np.exp(-((k * f0 - 1200) / 300) ** 2) + 0.1
    ax[1].vlines(k * f0, 0, amp, color=NAVY, lw=1.4, label="correct tuning: $k\\times120$ Hz")
    ax[1].vlines(k * f0 + 70, 0, amp * 0.85, color=ACCENT, lw=1.4, label="SSB 70 Hz off: $k\\times120+70$ Hz")
    ax[1].set_xlim(0, 2000); ax[1].set_yticks([]); ax[1].set_xlabel("audio frequency (Hz)")
    ax[1].set_title("SSB, BFO 70 Hz off: every harmonic slides\nby the same 70 Hz (no longer harmonic)", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); save(fig, "ch04_ssb_mistune")


def fig_vsb():
    f = np.linspace(-1.5, 5.0, 2000)          # MHz relative to the picture carrier
    a = 0.75
    H = np.clip(0.5 + f / (2 * a), 0, 1) * (f < 4.2) + 0 * f
    H = np.where(f > 4.2, np.clip(1 - (f - 4.2) / 0.3, 0, 1), H)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=False)
    ax[0].fill_between(f, H, color=NAVY, alpha=0.25); ax[0].plot(f, H, color=NAVY)
    ax[0].axvline(0, color=ACCENT, ls=":"); ax[0].text(0.05, 1.05, "picture carrier: response = 1/2", color=ACCENT, fontsize=7)
    ax[0].text(-1.45, 0.75, "vestige of the\nlower sideband", fontsize=7, color=NAVY)
    ax[0].text(2.0, 0.6, "full upper sideband", fontsize=7, color=NAVY)
    ax[0].set_ylim(0, 1.25); ax[0].set_ylabel("$H(f)$"); ax[0].set_xlabel("frequency relative to picture carrier (MHz)")
    ax[0].set_title("Receiver IF with the 'Nyquist slope'", fontsize=8.5)
    fb = np.linspace(0, 4.2, 500)
    Hp = np.interp(fb, f, H); Hm = np.interp(-fb, f, H)
    ax[1].plot(fb, Hp, color=NAVY, label="$H(f_c+f)$")
    ax[1].plot(fb, Hm, color=ORANGE, label="$H(f_c-f)$ (folded vestige)")
    ax[1].plot(fb, Hp + Hm, color=GREEN, lw=2, label="sum: flat, so no distortion")
    ax[1].set_ylim(0, 1.3); ax[1].set_xlabel("video frequency after detection (MHz)"); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "ch04_vsb")


def fig_phasor_click():
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    tt = np.linspace(0, 1, 800)
    for i, (r, lab) in enumerate([(0.6, "weaker interferer, $r$ = 0.6"), (1.35, "noise/interferer larger, $r$ = 1.35")]):
        a = fig.add_subplot(gs[0, i]); a.set_aspect("equal"); a.axis("off")
        a.set_xlim(-0.5, 2.5); a.set_ylim(-2.3, 1.5)
        a.plot(0, 0, "ko", ms=3); a.text(-0.1, -0.25, "origin", fontsize=6.5, ha="right")
        _arrow(a, 0, 0, 1, 0, NAVY, 2)
        a.add_patch(plt.Circle((1, 0), r, fill=False, ls=":", color=ACCENT))
        ang = np.deg2rad(130)
        _arrow(a, 1, 0, 1 + r * np.cos(ang), r * np.sin(ang), ACCENT, 1.3)
        _arrow(a, 0, 0, 1 + r * np.cos(ang), r * np.sin(ang), GREEN, 1.2, "--")
        a.set_title(lab, fontsize=8)
        a.text(1.0, -1.65, "circle misses origin:\nphase only wobbles" if r < 1 else "circle encloses origin:\nphase slips $2\\pi$ = a click",
               ha="center", fontsize=6.8, color=GREEN if r < 1 else ACCENT)
    a = fig.add_subplot(gs[0, 2])
    th = 2 * np.pi * 3 * tt
    for r, c in [(0.6, GREEN), (1.35, ACCENT)]:
        ph = np.unwrap(np.angle(1 + r * np.exp(1j * th)))
        a.plot(tt, ph / (2 * np.pi), color=c, label=f"$r$ = {r}")
    a.set_xlabel("time"); a.set_ylabel("resultant phase (cycles)"); a.legend(fontsize=7)
    a.set_title("phase of the sum", fontsize=8.5); a.set_xticks([])
    fig.tight_layout(); save(fig, "ch04_phasor_click")


def fig_click_rate():
    from scipy.special import erfc
    rho_db = np.linspace(0, 14, 300); rho = 10 ** (rho_db / 10)
    nu = 180e3 / np.sqrt(12) * erfc(np.sqrt(rho))
    fig, ax = plt.subplots(figsize=SIDE)
    ax.semilogy(rho_db, nu, color=NAVY, lw=1.8)
    for d in (3, 5, 7, 10):
        v = 180e3 / np.sqrt(12) * erfc(np.sqrt(10 ** (d / 10)))
        ax.plot(d, v, "o", color=ACCENT, ms=4)
        ax.text(d + 0.3, v * 1.4, f"{v:.1f}/s" if v < 1 else f"{v:.0f}/s", fontsize=7, color=ACCENT)
    ax.set_xlabel("carrier-to-noise ratio $\\rho$ (dB)"); ax.set_ylabel("clicks per second")
    ax.set_ylim(1e-3, 3e4)
    ax.set_title("Rice's click rate, $B$ = 180 kHz", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_click_rate")


def fig_emphasis_concept():
    f = np.linspace(0.05, 15, 400)
    music = 1 / (1 + (f / 1.5) ** 2) ** 0.5
    tau = 75e-6; f1 = 1 / (2 * np.pi * tau) / 1e3
    pe = np.sqrt(1 + (f / f1) ** 2)
    noise = 0.010 * f ** 2
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True)
    ax[0].fill_between(f, music, color=GREEN, alpha=0.6); ax[0].set_title("1. music: little treble", fontsize=8)
    ax[1].fill_between(f, music * pe, color=GREEN, alpha=0.6); ax[1].set_title("2. pre-emphasis:\nshout the treble", fontsize=8)
    ax[2].fill_between(f, music * pe, color=GREEN, alpha=0.6); ax[2].fill_between(f, noise, color=ACCENT, alpha=0.5)
    ax[2].set_title("3. channel adds\nparabolic FM noise", fontsize=8)
    ax[3].fill_between(f, music, color=GREEN, alpha=0.6); ax[3].fill_between(f, noise / pe ** 2, color=ACCENT, alpha=0.8)
    ax[3].plot(f, noise, color=ACCENT, ls=":", lw=1); ax[3].text(9.0, 2.1, "noise\nwithout\nde-emphasis", fontsize=6.3, color=ACCENT)
    ax[3].set_title("4. de-emphasis: music\nrestored, hiss pushed down", fontsize=8)
    for a in ax:
        a.set_xlabel("kHz", fontsize=7.5); a.set_yticks([]); a.set_ylim(0, 2.8); a.grid(False)
    fig.tight_layout(); save(fig, "ch04_emphasis_concept")


def fig_tracking():
    from scipy.optimize import fsolve
    fif = 455e3; Cmin, Cmax = 10e-12, 365e-12
    Cs = (Cmax - 10.29 * Cmin) / 9.29
    Lr = 1 / ((2 * np.pi * 530e3) ** 2 * (Cmax + Cs))
    def f_rf(Cg): return 1 / (2 * np.pi * np.sqrt(Lr * (Cg + Cs)))
    def Cg_for(f): return 1 / ((2 * np.pi * f) ** 2 * Lr) - Cs
    def f_lo(Cg, Lo, Ct, Cp):
        Ctot = 1 / (1 / (Cg + Ct) + 1 / Cp) if Cp else Cg + Ct
        return 1 / (2 * np.pi * np.sqrt(Lo * Ctot))
    pts = [600e3, 1050e3, 1500e3]
    def eq(p):
        Lo, Ct, Cp = p[0] * 1e-6, p[1] * 1e-12, p[2] * 1e-12
        return [(f_lo(Cg_for(f), Lo, Ct, Cp) - f - fif) / 1e3 for f in pts]
    Lo, Ct, Cp = fsolve(eq, [120, 20, 450])
    Lo, Ct, Cp = Lo * 1e-6, Ct * 1e-12, Cp * 1e-12
    def eq2(p):  # no padder: trimmer + inductor only, exact at the two band edges
        L2, C2 = p[0] * 1e-6, p[1] * 1e-12
        return [(f_lo(Cg_for(f), L2, C2, 0) - f - fif) / 1e3 for f in (600e3, 1500e3)]
    L2, C2 = fsolve(eq2, [100, 30]); L2, C2 = L2 * 1e-6, C2 * 1e-12
    f = np.linspace(540e3, 1600e3, 400); Cg = Cg_for(f)
    e3 = f_lo(Cg, Lo, Ct, Cp) - f - fif
    e2 = f_lo(Cg, L2, C2, 0) - f - fif
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.4))
    ax.plot(f / 1e3, e2 / 1e3, color=GRAY, ls="--", label="trimmer only (exact at 2 points)")
    ax.plot(f / 1e3, e3 / 1e3, color=NAVY, lw=1.8, label="padder + trimmer (three-point tracking)")
    ax.plot(np.array(pts) / 1e3, [0, 0, 0], "o", color=ACCENT, ms=4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("dial frequency (kHz)"); ax.set_ylabel("tracking error (kHz)")
    ax.set_ylim(-40, 40); ax.legend(fontsize=7, loc="lower center")
    ax.set_title("Keeping the LO exactly 455 kHz above the preselector", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_tracking")
    print(f"   tracking: L_rf={Lr*1e6:.0f} uH, Lo={Lo*1e6:.1f} uH, Ct={Ct*1e12:.1f} pF, Cp={Cp*1e12:.0f} pF, max err={np.max(np.abs(e3))/1e3:.2f} kHz")


def fig_agc_static():
    pin = np.linspace(-120, -10, 400)
    noagc = np.minimum(pin + 100, 10)
    knee = -95
    agc = np.where(pin < knee, pin + 100, 5 + (pin - knee) * 0.06)
    fig, ax = plt.subplots(figsize=SIDE)
    ax.plot(pin, noagc, color=GRAY, ls="--", label="no AGC (overloads)")
    ax.plot(pin, agc, color=NAVY, lw=1.8, label="with AGC")
    ax.annotate("AGC starts\n(delayed AGC knee)", (knee, 5), (-112, 18), fontsize=7, arrowprops=dict(arrowstyle="->", color=ACCENT), color=ACCENT)
    ax.set_xlabel("antenna signal (dBm)"); ax.set_ylabel("level at detector (dB rel.)")
    ax.set_ylim(-25, 28); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("85 dB in, about 5 dB out", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_agc_static")


def fig_selectivity():
    df = np.linspace(-45, 45, 2000)            # kHz offset from the wanted station at 1000 kHz
    f0 = 1000.0
    def single(Q, f):
        rho = (f0 + f) / f0 - f0 / (f0 + f)
        return 1 / np.sqrt(1 + (Q * rho) ** 2)
    crystal = single(35, df)
    trf = single(100, df) ** 3
    x = 2 * 92 * df / 455.0
    dt = 2 / np.sqrt(4 + x ** 4)
    sup = dt ** 2 * single(40, df)
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 2.5))
    for k in range(-4, 5):
        ax.axvspan(10 * k - 4.5, 10 * k + 4.5, color=NAVY if k == 0 else GRAY, alpha=0.18 if k == 0 else 0.08, lw=0)
    for y, lab, c in [(crystal, "crystal set: one tuned circuit, $Q\\approx35$", ORANGE),
                      (trf, "TRF: three tuned stages, $Q$ = 100", GREEN),
                      (sup, "superhet: two 455 kHz IF transformers", NAVY)]:
        ax.plot(df, 20 * np.log10(y), color=c, lw=1.6, label=lab)
    ax.text(0, -57, "wanted", ha="center", fontsize=7, color=NAVY)
    ax.text(10, -57, "next\nchannel", ha="center", fontsize=6.5, color=GRAY)
    ax.set_xlim(-45, 45); ax.set_ylim(-60, 3); ax.set_xlabel("offset from wanted station at 1000 kHz (kHz)")
    ax.set_ylabel("response (dB)"); ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("Selectivity on a crowded band (10 kHz channels)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_selectivity")


def fig_superhet_concept():
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    st = [(600, NAVY, "A"), (1000, GREEN, "B"), (1450, ORANGE, "C")]
    ax.hlines(2, 500, 1700, color="k", lw=0.8); ax.text(470, 2, "antenna:\nRF dial", ha="right", va="center", fontsize=7.5)
    ax.hlines(0, 410, 1700, color="k", lw=0.8); ax.text(400, 0, "after the\nmixer: IF", ha="right", va="center", fontsize=7.5)
    for f, c, n in st:
        ax.add_patch(plt.Rectangle((f - 5, 2), 10, 0.45, color=c))
        ax.text(f, 2.55, f"station {n}\n{f} kHz", ha="center", fontsize=7, color=c)
        ax.annotate("", (455 + 4 * (st.index((f, c, n)) - 1), 0.55), (f, 1.95),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, connectionstyle="arc3,rad=0.15"))
        ax.text(f + 12, 1.65, f"tune LO to\n{f + 455} kHz", fontsize=6.8, color=c, va="top")
    ax.add_patch(plt.Rectangle((450, 0), 10, 0.45, color=ACCENT))
    ax.add_patch(plt.Rectangle((420, -0.05), 70, 0.6, fill=False, ec=ACCENT, lw=1.2, ls="--"))
    ax.text(560, 0.15, "fixed 455 kHz IF filter: every station lands here", fontsize=7.5, color=ACCENT)
    ax.set_xlim(250, 1720); ax.set_ylim(-0.3, 3.2); ax.axis("off")
    fig.tight_layout(); save(fig, "ch04_superhet_concept")


def fig_image_mirror():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.9))
    def pk(a, fc, h, c, w=8):
        f = np.linspace(fc - 30, fc + 30, 200)
        a.fill_between(f, h * np.exp(-((f - fc) / w) ** 2), color=c, alpha=0.85, lw=0)
    a = ax[0]
    pk(a, 1000, 1, NAVY); pk(a, 1910, 0.8, ORANGE)
    a.vlines(1455, 0, 1.15, color=ACCENT, lw=2)
    a.text(1455, 1.2, "LO 1455 kHz (the mirror)", ha="center", fontsize=7, color=ACCENT)
    a.text(1000, 1.08, "wanted 1000", ha="center", fontsize=7, color=NAVY)
    a.text(1910, 0.88, "image 1910", ha="center", fontsize=7, color=ORANGE)
    for x in (1000, 1910):
        a.annotate("", (1455, 0.45), (x, 0.45), arrowprops=dict(arrowstyle="<->", color=GRAY))
    a.text(1227, 0.5, "455", ha="center", fontsize=7); a.text(1682, 0.5, "455", ha="center", fontsize=7)
    a.set_xlim(800, 2100); a.set_ylim(0, 1.45); a.set_yticks([]); a.set_title("before the mixer", fontsize=8, loc="left")
    a = ax[1]
    pk(a, 455, 1, NAVY); pk(a, 455, 0.8, ORANGE)
    a.text(470, 0.95, "wanted and image now share 455 kHz:\nno filter after the mixer can separate them",
           fontsize=7.5, color=ACCENT, va="top", ha="left", transform=a.transData)
    a.set_xlim(300, 1200); a.set_ylim(0, 1.2); a.set_yticks([]); a.set_xlabel("frequency (kHz)")
    a.set_title("after the mixer", fontsize=8, loc="left")
    fig.tight_layout(); save(fig, "ch04_image_mirror")


def fig_raster():
    fig, ax = plt.subplots(figsize=SIDE)
    n = 9
    for k in range(n):
        y = 1 - (2 * k) / (2 * n)
        ax.annotate("", (1, y - 0.02), (0, y), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.1, mutation_scale=7))
        y2 = 1 - (2 * k + 1) / (2 * n)
        ax.annotate("", (1, y2 - 0.02), (0, y2), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1, ls="--", mutation_scale=7))
        if k < n - 1:
            ax.plot([1, 0], [y - 0.02, y - 2 / (2 * n)], color=GRAY, lw=0.4, ls=":")
    ax.text(0.5, 1.07, "field 1 (odd lines, solid)  +  field 2 (even lines, dashed)", ha="center", fontsize=6.8)
    ax.text(1.03, 0.5, "dotted:\nflyback\n(blanked)", fontsize=6.5, color=GRAY, va="center")
    ax.set_xlim(-0.05, 1.3); ax.set_ylim(-0.05, 1.12); ax.axis("off")
    fig.tight_layout(); save(fig, "ch04_raster")


def fig_reuse7():
    fig, ax = plt.subplots(figsize=SIDE)
    from matplotlib.patches import RegularPolygon
    cols = [NAVY, GREEN, ORANGE, PURPLE, ACCENT, "#2E86C1", GRAY]
    # axial coords; cluster label via (i + 3j) mod 7 gives the N=7 pattern
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q; y = np.sqrt(3) * (r + q / 2)
            if abs(x) > 4.6 or abs(y) > 4.2:
                continue
            lab = (q + 3 * r) % 7
            ax.add_patch(RegularPolygon((x, y), 6, radius=1.0, orientation=np.pi / 6 + np.pi / 6,
                                        fc=cols[lab], alpha=0.30 if lab else 0.75, ec="white", lw=1))
            ax.text(x, y, str(lab + 1), ha="center", va="center", fontsize=7, color="white" if lab == 0 else "k")
    ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-4.4, 4.4); ax.axis("off")
    ax.set_title("7-cell reuse: dark cells share channel set 1", fontsize=8)
    fig.tight_layout(); save(fig, "ch04_reuse7")


def fig_repeaters():
    N = np.arange(1, 201)
    snr1 = 70.0
    analog = snr1 - 10 * np.log10(N)
    p = 1e-9; ber = N * p
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    ax.semilogx(N, analog, color=ACCENT, lw=1.8, label="analog FDM: noise adds at every repeater")
    ax.semilogx(N, np.full_like(analog, 50.0, dtype=float), color=NAVY, lw=1.8,
                label="digital PCM: regenerated, quality set by the codec")
    ax.set_xlabel("number of repeater sections"); ax.set_ylabel("end-to-end SNR (dB)")
    ax.set_ylim(40, 75); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Why digital won: accumulation versus regeneration (illustrative)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_repeaters")


def fig_bandwidths():
    items = [("SSB voice (HF)", 2.7e3), ("AM broadcast, 5 kHz audio", 10e3), ("NBFM land mobile", 11e3),
             ("AMPS cellular voice", 30e3), ("mono FM broadcast", 180e3), ("stereo FM (Carson)", 256e3),
             ("NTSC TV channel", 6e6), ("analog satellite TV (FM)", 28e6)]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for i, (lab, b) in enumerate(items):
        ax.barh(i, b, color=CYCLE[i % 6], alpha=0.85, height=0.62)
        txt = f"{b/1e6:g} MHz" if b >= 1e6 else f"{b/1e3:g} kHz"
        ax.text(b * 1.25, i, txt, va="center", fontsize=7.3)
    ax.set_xscale("log"); ax.set_xlim(1e3, 2e8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("occupied bandwidth (Hz)"); ax.grid(axis="y", visible=False)
    ax.set_title("Analog services span four orders of magnitude in bandwidth", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_bandwidths")


def fig_mpx_time():
    fs = 1.92e6; t = np.arange(0, 2e-3, 1 / fs)
    L = np.sin(2 * np.pi * 1e3 * t); R = 0 * t; a = 0.45
    mpx = a * (L + R) + a * (L - R) * np.cos(2 * np.pi * 38e3 * t)
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.plot(t * 1e3, mpx, color=NAVY, lw=0.4, label="MPX (pilot omitted)")
    ax.plot(t * 1e3, 2 * a * L, color=GREEN, lw=1.4, ls="--", label="$2aL$: peaks of the MPX")
    ax.plot(t * 1e3, 2 * a * R, color=ORANGE, lw=1.4, ls="--", label="$2aR$ = 0: troughs of the MPX")
    ax.axhline(0.9, color=ACCENT, lw=0.8, ls=":"); ax.text(2.02, 0.9, "mono peak\nlevel", fontsize=6.5, color=ACCENT, va="center")
    ax.set_xlim(0, 2.25); ax.set_ylim(-1.45, 1.1); ax.set_xlabel("time (ms)"); ax.set_yticks([]); ax.legend(fontsize=6.5, loc="lower center", ncol=3)
    ax.set_title("Left channel only: the stereo MPX never exceeds the mono peak", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_mpx_time")


def fig_ssb_vs_am():
    t = np.linspace(0, 2, 4000); fc = 30
    am = (1 + np.cos(2 * np.pi * 1 * t)) * np.cos(2 * np.pi * fc * t)          # PEP envelope 2
    ssb = 2 * np.cos(2 * np.pi * (fc + 1) * t)                                  # same PEP, constant envelope
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for a, x, env, ttl in [(ax[0], am, 1 + np.cos(2 * np.pi * t), "AM, 100%: peak 2A, mostly carrier"),
                           (ax[1], ssb, 2 + 0 * t, "SSB tone, same peak: all sideband")]:
        a.plot(t, x, color=NAVY, lw=0.5); a.plot(t, env, color=ACCENT, lw=1.1, ls="--")
        a.axhline(2, color=GRAY, lw=0.6, ls=":"); a.set_ylim(-2.3, 2.5); a.set_yticks([]); a.set_xticks([])
        a.set_title(ttl, fontsize=8); a.grid(False)
    ax[2].bar([0, 1], [0, 9], color=[GRAY, GREEN], width=0.6)
    ax[2].text(0, 0.6, "AM\n0 dB", ha="center", fontsize=7.5); ax[2].text(1, 9.4, "SSB\n+9 dB", ha="center", fontsize=7.5)
    ax[2].set_xticks([]); ax[2].set_ylim(0, 13); ax[2].set_ylabel("output SNR gain (dB)", fontsize=7.5)
    ax[2].set_title("same PEP, same noise", fontsize=8)
    fig.tight_layout(); save(fig, "ch04_ssb_vs_am")


def fig_inst_freq():
    t = np.linspace(0, 1, 8000)
    m = np.sin(2 * np.pi * 2 * t)
    fc, kf = 30, 15
    ph = 2 * np.pi * fc * t + 2 * np.pi * kf * np.cumsum(m) / len(t)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))
    ax[0].plot(t, np.cos(ph), color=NAVY, lw=0.6); ax[0].set_yticks([]); ax[0].grid(False)
    ax[0].set_title("FM signal: the crests bunch up and spread out", fontsize=8.5, loc="left")
    ax[1].plot(t, fc + kf * m, color=ACCENT, lw=1.6, label="instantaneous frequency $f_i(t)=f_c+k_f m(t)$")
    ax[1].axhline(fc, color=GRAY, ls=":", lw=0.8); ax[1].text(1.0, fc + 1, "$f_c$", fontsize=8, color=GRAY, ha="right")
    ax[1].fill_between(t, fc - kf, fc + kf, color=ACCENT, alpha=0.07)
    ax[1].text(0.005, fc + kf - 3.5, "$\\pm\\Delta f$", fontsize=8, color=ACCENT)
    ax[1].set_ylabel("Hz (arb.)"); ax[1].set_xlabel("time"); ax[1].set_xticks([]); ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "ch04_inst_freq")


def fig_slope_detector():
    f = np.linspace(-6, 6, 600)
    H = 1 / np.sqrt(1 + (f / 1.6) ** 2)
    f0 = 1.6                         # carrier parked on the skirt
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.4))
    ax.plot(f, H, color=NAVY, lw=1.8, label="tuned-circuit response")
    dev = 0.7
    ax.axvspan(f0 - dev, f0 + dev, color=GREEN, alpha=0.15)
    ax.axvline(f0, color=GREEN, ls=":")
    hlo, hhi = 1 / np.sqrt(1 + ((f0 + dev) / 1.6) ** 2), 1 / np.sqrt(1 + ((f0 - dev) / 1.6) ** 2)
    ax.axhspan(hlo, hhi, xmin=0, xmax=1, color=ACCENT, alpha=0.12)
    ax.annotate("", (f0 + dev, -0.02), (f0 - dev, -0.02), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(f0 + dev + 0.15, 0.03, "frequency swing (the FM)", ha="left", fontsize=7, color=GREEN)
    ax.annotate("", (5.3, hlo), (5.3, hhi), arrowprops=dict(arrowstyle="<->", color=ACCENT))
    ax.text(5.15, (hlo + hhi) / 2, "amplitude swing\n(now AM)", ha="right", va="center", fontsize=7, color=ACCENT)
    ax.text(-5.8, 0.9, "carrier parked on\nthe skirt, not the peak", fontsize=7, color=NAVY)
    ax.set_xlabel("frequency offset from resonance (arb.)"); ax.set_ylabel("response"); ax.set_ylim(-0.08, 1.1)
    ax.set_yticks([]); ax.set_title("The slope detector: FM to AM by a sloping filter", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_slope_detector")


def fig_noise_phasor():
    r = rng(7)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for a, A, ttl in [(ax[0], 1.0, "strong carrier: noise barely tilts it"), (ax[1], 0.45, "weak carrier: same noise, big tilts")]:
        a.set_aspect("equal"); a.axis("off"); a.set_xlim(-0.15, 1.45); a.set_ylim(-0.55, 0.55)
        n = 0.11 * (r.standard_normal(400) + 1j * r.standard_normal(400))
        a.scatter(A + n.real, n.imag, s=2, color=ACCENT, alpha=0.45)
        _arrow(a, 0, 0, A, 0, NAVY, 2.2)
        k = np.argmax(np.abs(n.imag)); tip = A + n[k]
        _arrow(a, 0, 0, tip.real, tip.imag, GREEN, 1.2, "--")
        ang = np.degrees(np.arctan2(tip.imag, tip.real))
        a.text(0.05, -0.3, f"worst tilt about {abs(ang):.0f} degrees", fontsize=7, color=GREEN)
        a.set_title(ttl, fontsize=8.5)
        a.text(A + 0.05, -0.47, "noise along the carrier only\nchanges its length: the\nlimiter removes it", fontsize=6.3, color=GRAY)
    fig.tight_layout(); save(fig, "ch04_noise_phasor")


def fig_snr_ladder():
    items = [("AM, processed speech ($P_m$=0.15)", 10 * np.log10(0.15 / 1.15)), ("AM, tone, $\\mu$=1", 10 * np.log10(1 / 3)),
             ("baseband, DSB-SC, SSB", 0.0), ("FM, $D$=5, tone", 10 * np.log10(37.5)),
             ("FM, $D$=5 + 75 $\\mu$s de-emphasis", 10 * np.log10(37.5 * 20.9))]
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 2.2))
    for i, (lab, v) in enumerate(items):
        ax.barh(i, v, color=ACCENT if v < 0 else (GRAY if v == 0 else GREEN), alpha=0.85, height=0.6)
        ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f} dB", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=7.5)
    ax.set_xlim(-16, 36); ax.set_xlabel("output SNR relative to baseband, same received power (dB)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch04_snr_ladder")


def fig_rds_spectrum():
    r = rng(11)
    fs = 228e3; rb = 1187.5; sps_ = int(round(fs / rb))   # 192 samples/bit
    nb = 4000
    bits = r.integers(0, 2, nb)
    d = np.cumsum(bits) % 2                                 # differential encoding
    sym = 2 * d - 1
    half = sps_ // 2
    bb = np.concatenate([np.r_[np.ones(half), -np.ones(sps_ - half)] * s for s in sym])   # biphase
    # shape roughly as the standard's cosine filtering by a gentle low-pass
    bb = lowpass(bb, 2.4e3, fs, 4)
    t = np.arange(len(bb)) / fs
    x = bb * np.cos(2 * np.pi * 57e3 * t)
    f, p = sps.welch(x, fs, nperseg=16384)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.3))
    ax.plot(f / 1e3, 10 * np.log10(p / p.max() + 1e-9), color=ORANGE, lw=1.2)
    ax.axvline(57, color=GRAY, ls=":"); ax.text(59.55, -12, "57 kHz: a null,\nso no tone to\nupset the pilot\nloops", fontsize=6.5, color=GRAY)
    ax.set_xlim(52, 62); ax.set_ylim(-45, 3); ax.set_xlabel("MPX frequency (kHz)"); ax.set_ylabel("PSD (dB)")
    ax.set_title("The RDS subcarrier: biphase-coded BPSK at 1187.5 b/s", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_rds_spectrum")


def fig_selective_fading():
    fs = 200e3; t = np.arange(0, 6e-3, 1 / fs); fc = 20e3; fm = 600
    m = np.cos(2 * np.pi * fm * t)
    # carrier attenuated 20 dB by a selective fade, sidebands intact
    x = (0.1 + 0.8 * m) * np.cos(2 * np.pi * fc * t)
    env = lowpass(np.abs(x), 4e3, fs) * np.pi / 2
    syn = lowpass(2 * x * np.cos(2 * np.pi * fc * t), 4e3, fs)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for a, y, ttl, c in [(ax[0], env, "envelope detector: rectified, harsh", ACCENT),
                         (ax[1], syn, "synchronous detector: clean", GREEN)]:
        a.plot(t * 1e3, x, color=NAVY, lw=0.3, alpha=0.5)
        a.plot(t * 1e3, y, color=c, lw=1.5)
        a.set_title(ttl, fontsize=8.5); a.set_xlabel("time (ms)"); a.set_yticks([]); a.set_xlim(0.6, 5.6)
    fig.suptitle("Carrier faded by 20 dB, sidebands intact", fontsize=8.5, y=1.0)
    fig.tight_layout(); save(fig, "ch04_selective_fading")


def fig_fm_vs_pm():
    f = np.linspace(50, 15000, 400)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1))
    ax[0].loglog(f, 75e3 / f, color=NAVY, lw=1.6, label="FM, $\\Delta f$ = 75 kHz")
    ax[0].loglog(f, np.full_like(f, 5.0), color=ORANGE, lw=1.6, ls="--", label="PM, $k_p$ = 5 rad")
    ax[0].set_xlabel("audio tone frequency (Hz)"); ax[0].set_ylabel("peak phase swing $\\beta$ (rad)")
    ax[0].legend(fontsize=7); ax[0].set_title("phase swing", fontsize=8.5)
    ax[1].loglog(f, np.full_like(f, 75.0), color=NAVY, lw=1.6, label="FM")
    ax[1].loglog(f, 5 * f / 1e3, color=ORANGE, lw=1.6, ls="--", label="PM")
    ax[1].set_xlabel("audio tone frequency (Hz)"); ax[1].set_ylabel("peak deviation (kHz)")
    ax[1].legend(fontsize=7); ax[1].set_title("frequency swing", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch04_fm_vs_pm")


def fig_negclip():
    fs = 4e6; fc = 455e3; fm = 2e3; mu = 0.9
    t = np.arange(0, 1.6e-3, 1 / fs)
    env = 1 + mu * np.cos(2 * np.pi * fm * t)
    x = env * np.cos(2 * np.pi * fc * t)
    v = lowpass(diode_detector(x, fs, 10e-6), 15e3, fs, 4)   # after the audio filter
    vdc = lowpass(v, 300, fs, 2)                       # DC held by the coupling capacitor
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for a, ratio, ttl in [(ax[0], 0.96, "$R_{ac}/R_{dc}$ = 0.96: troughs followed"),
                          (ax[1], 0.82, "$R_{ac}/R_{dc}$ = 0.82: troughs clipped")]:
        # the diode cannot pull the output below the level set by the AC/DC load divider
        floor = vdc * (1 - ratio)
        y = np.maximum(v, vdc - ratio * (vdc - 0) + 0 * v)
        y = np.maximum(v, vdc * (1 - ratio) + 0.0)
        a.plot(t * 1e3, env, color=GREEN, ls="--", lw=1.0)
        a.plot(t * 1e3, y, color=ACCENT, lw=1.3)
        a.axhline(np.mean(vdc[len(vdc) // 3:]) * (1 - ratio), color=GRAY, ls=":", lw=0.8)
        a.set_title(ttl, fontsize=8.5); a.set_xlabel("time (ms)"); a.set_xlim(0.1, 1.6)
    ax[0].set_ylabel("detector output"); ax[0].set_ylim(0, 2.05)
    fig.tight_layout(); save(fig, "ch04_negclip")


def fig_analytic():
    f = np.linspace(-4.5, 4.5, 900)
    M = np.where((np.abs(f) > 0.3) & (np.abs(f) < 3), 1 - 0.2 * np.abs(f), 0)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    ax[0].fill_between(f, M, color=NAVY, alpha=0.7); ax[0].set_title("real message: two mirror halves", fontsize=8)
    ax[1].fill_between(f, 2 * M * (f > 0), color=GREEN, alpha=0.7); ax[1].set_title("$m+j\\hat m$: one side only", fontsize=8)
    g = np.linspace(-14, 14, 1400)
    Mg = lambda x: np.where((x > 0.3) & (x < 3), 1 - 0.2 * x, 0)
    ax[2].fill_between(g, Mg(g - 10) + Mg(-g - 10), color=ORANGE, alpha=0.8)
    ax[2].axvline(10, color=GRAY, ls=":", lw=0.8); ax[2].axvline(-10, color=GRAY, ls=":", lw=0.8)
    ax[2].set_title("SSB: shifted to $\\pm f_c$, upper side kept", fontsize=8)
    for a in ax:
        a.set_yticks([]); a.set_xlabel("frequency")
    ax[0].set_xticks([0]); ax[1].set_xticks([0]); ax[2].set_xticks([-10, 0, 10]); ax[2].set_xticklabels(["$-f_c$", "0", "$f_c$"])
    fig.tight_layout(); save(fig, "ch04_analytic")


NEWF2 = [fig_negclip, fig_analytic, fig_selective_fading, fig_fm_vs_pm, fig_ssb_vs_am, fig_inst_freq, fig_slope_detector, fig_noise_phasor, fig_snr_ladder, fig_rds_spectrum]


NEWF = [fig_timeline, fig_spectrum_map, fig_wavelengths, fig_two_knobs, fig_phasors, fig_am_power, fig_trapezoid,
        fig_switching_mod, fig_ssb_mistune, fig_vsb, fig_phasor_click, fig_click_rate, fig_emphasis_concept,
        fig_tracking, fig_agc_static, fig_selectivity, fig_superhet_concept, fig_image_mirror, fig_raster,
        fig_reuse7, fig_repeaters, fig_bandwidths, fig_mpx_time] + NEWF2


if __name__ == "__main__":
    import sys as _s
    allf = [fig_am, fig_am_family, fig_ssb_phase_error, fig_fm_bessel, fig_carson, fig_fm_threshold,
            fig_stereo_mpx, fig_superhet, fig_envelope_clip, fig_weaver, fig_fm_spectra, fig_discriminators,
            fig_preemphasis, fig_fm_clicks, fig_capture, fig_image_rejection, fig_agc, fig_ip3,
            fig_ntsc_line, fig_ntsc_spectrum, fig_fdm] + NEWF
    sel = _s.argv[1:]
    for fn in allf:
        if not sel or fn.__name__ in sel:
            fn()
