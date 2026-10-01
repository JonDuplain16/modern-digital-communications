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


if __name__ == "__main__":
    import sys as _s
    allf = [fig_am, fig_am_family, fig_ssb_phase_error, fig_fm_bessel, fig_carson, fig_fm_threshold,
            fig_stereo_mpx, fig_superhet, fig_envelope_clip, fig_weaver, fig_fm_spectra, fig_discriminators,
            fig_preemphasis, fig_fm_clicks, fig_capture, fig_image_rejection, fig_agc, fig_ip3,
            fig_ntsc_line, fig_ntsc_spectrum, fig_fdm]
    sel = _s.argv[1:]
    for fn in allf:
        if not sel or fn.__name__ in sel:
            fn()
