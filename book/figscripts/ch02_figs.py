"""Figures for Chapter 2: Signals, Spectra and Systems."""
from figstyle import *
from scipy import signal as sps
import commlib as cl


def fig_phasor():
    fig = plt.figure(figsize=(W2, 2.6))
    ax0 = fig.add_axes([0.02, 0.1, 0.28, 0.8]); ax1 = fig.add_axes([0.38, 0.18, 0.6, 0.72])
    th = np.linspace(0, 2 * np.pi, 200)
    ax0.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.8)
    a = 0.9
    for sgn, c, lab in [(1, NAVY, r"$\frac{A}{2}e^{+j\omega t}$"), (-1, ACCENT, r"$\frac{A}{2}e^{-j\omega t}$")]:
        z = np.exp(1j * sgn * a)
        ax0.annotate("", (z.real, z.imag), (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        ax0.text(z.real * 1.12, z.imag * 1.18, lab, color=c, ha="center", fontsize=9)
    ax0.annotate("", (2 * np.cos(a) * 0.5 * 2, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    ax0.text(0.95, -0.25, r"$A\cos\omega t$", color=GREEN, fontsize=9)
    ax0.set_aspect("equal"); ax0.set_xlim(-1.4, 1.5); ax0.set_ylim(-1.4, 1.4); ax0.axis("off")
    ax0.axhline(0, color="k", lw=0.5); ax0.axvline(0, color="k", lw=0.5)
    t = np.linspace(0, 2, 600)
    ax1.plot(t, np.cos(2 * np.pi * 1.5 * t), color=GREEN, label=r"$\cos(2\pi f_0 t)$ (real signal)")
    ax1.plot(t, np.real(np.exp(2j * np.pi * 1.5 * t) * np.exp(1j * 0.6)), color=NAVY, ls="--", label=r"$\mathrm{Re}\{e^{j(2\pi f_0 t+\phi)}\}$: phase shift $\phi$")
    ax1.set_xlabel("time (s), $f_0 = 1.5$ Hz"); ax1.legend(loc="lower right", fontsize=7)
    ax1.set_title("A real sinusoid is the sum of two counter-rotating phasors")
    save(fig, "ch02_phasor")


def fig_fourier_series():
    t = np.linspace(-1, 1, 4000)
    sq = np.sign(np.sin(2 * np.pi * t))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(t, sq, color=GRAY, lw=0.9, label="square wave")
    for N, c in [(1, ACCENT), (3, ORANGE), (9, GREEN), (49, NAVY)]:
        s = sum(4 / (np.pi * k) * np.sin(2 * np.pi * k * t) for k in range(1, N + 1, 2))
        ax[0].plot(t, s, color=c, lw=1.0, label=f"harmonics up to {N}")
    ax[0].set_xlim(-0.6, 0.6); ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_xlabel("time (periods)")
    ax[0].set_title("Partial Fourier sums and the Gibbs overshoot")
    k = np.arange(1, 30)
    ck = np.where(k % 2 == 1, 4 / (np.pi * k), 0)
    ax[1].stem(k, 20 * np.log10(np.maximum(ck, 1e-6)), basefmt=" ", linefmt=NAVY, markerfmt="o")
    ax[1].set_ylim(-35, 5); ax[1].set_xlabel("harmonic number $k$"); ax[1].set_ylabel("amplitude (dB)")
    ax[1].set_title("Odd harmonics fall as $1/k$ (-6 dB/octave)")
    fig.tight_layout(); save(fig, "ch02_fourier_series")


def fig_pairs():
    t = np.linspace(-3, 3, 3001); f = np.linspace(-4, 4, 3001)
    pairs = [
        ("rectangle", np.where(np.abs(t) <= 0.5, 1, 0), "sinc", np.sinc(f)),
        ("triangle", np.maximum(1 - np.abs(t), 0), "sinc$^2$", np.sinc(f) ** 2),
        ("Gaussian", np.exp(-np.pi * t ** 2), "Gaussian", np.exp(-np.pi * f ** 2)),
        ("one-sided exponential", np.where(t >= 0, np.exp(-2 * t), 0), "|Lorentzian|", 1 / np.sqrt(4 + (2 * np.pi * f) ** 2)),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(W2, 3.2))
    for i, (n1, x, n2, X) in enumerate(pairs):
        ax[0, i].plot(t, x, color=NAVY); ax[0, i].set_title(n1, fontsize=8.5); ax[0, i].set_xlim(-2, 2.5)
        ax[1, i].plot(f, X / np.max(np.abs(X)), color=ACCENT); ax[1, i].set_title(n2, fontsize=8.5)
        ax[0, i].set_xlabel("$t$", labelpad=0); ax[1, i].set_xlabel("$f$", labelpad=0)
        for a in (ax[0, i], ax[1, i]): a.set_yticks([0, 1])
    ax[0, 0].set_ylabel("time"); ax[1, 0].set_ylabel("frequency")
    fig.tight_layout(h_pad=0.6); save(fig, "ch02_pairs")


def fig_filters():
    """Magnitude, group delay and pulse response of three 5th-order low-pass filters (cutoff 1 kHz)."""
    fs = 48000; fc = 1000
    designs = [("Butterworth", sps.butter(5, fc, fs=fs, output="sos")),
               ("Chebyshev I (0.5 dB)", sps.cheby1(5, 0.5, fc, fs=fs, output="sos")),
               ("Bessel", sps.bessel(5, fc, fs=fs, output="sos", norm="mag")),
               ("Elliptic (0.5/50 dB)", sps.ellip(5, 0.5, 50, fc, fs=fs, output="sos"))]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5))
    x = np.zeros(int(0.012 * fs)); x[int(0.001 * fs):int(0.004 * fs)] = 1
    tt = np.arange(len(x)) / fs * 1e3
    for name, sos in designs:
        w, h = sps.sosfreqz(sos, 8192, fs=fs)
        ax[0].plot(w / 1e3, 20 * np.log10(np.abs(h) + 1e-12), label=name, lw=1.1)
        b, a = sps.sos2tf(sos)
        wg, gd = sps.group_delay((b, a), 4096, fs=fs)
        ax[1].plot(wg / 1e3, gd / fs * 1e3, lw=1.1)
        ax[2].plot(tt, sps.sosfilt(sos, x), lw=1.1)
    ax[0].set_xlim(0, 3); ax[0].set_ylim(-70, 5); ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("dB"); ax[0].set_title("magnitude")
    ax[0].legend(fontsize=6, loc="lower left")
    ax[1].set_xlim(0, 1.4); ax[1].set_ylim(0, 2.5); ax[1].set_xlabel("frequency (kHz)"); ax[1].set_ylabel("ms"); ax[1].set_title("group delay")
    ax[2].plot(tt, x, color=GRAY, lw=0.8, ls="--"); ax[2].set_xlabel("time (ms)"); ax[2].set_title("3 ms pulse response")
    fig.tight_layout(); save(fig, "ch02_filters")


def fig_windows():
    fs, N = 1000, 256
    n = np.arange(N)
    x = np.sin(2 * np.pi * 101.3 * n / fs) + 10 ** (-60 / 20) * np.sin(2 * np.pi * 131.1 * n / fs)
    wins = [("rectangular", np.ones(N)), ("Hann", np.hanning(N)), ("Blackman-Harris", sps.windows.blackmanharris(N)),
            ("flat-top", sps.windows.flattop(N))]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for name, w in wins:
        X = np.fft.rfft(x * w, 8192) / np.sum(w)
        f = np.fft.rfftfreq(8192, 1 / fs)
        ax[0].plot(f, 20 * np.log10(np.abs(X) + 1e-12) + 6.02, lw=0.9, label=name)
    ax[0].set_xlim(60, 170); ax[0].set_ylim(-130, 5); ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("dB")
    ax[0].annotate("weak tone, -60 dB", (131.1, -60), (140, -30), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[0].legend(fontsize=6.5, loc="upper right"); ax[0].set_title("Leakage: a -60 dB tone 30 Hz away")
    for name, w in wins:
        W = np.fft.fftshift(np.fft.fft(w, 16 * N)) / np.sum(w)
        fb = (np.arange(16 * N) - 8 * N) / 16
        ax[1].plot(fb, 20 * np.log10(np.abs(W) + 1e-12), lw=0.9)
    ax[1].set_xlim(-8, 8); ax[1].set_ylim(-120, 3); ax[1].set_xlabel("offset (DFT bins)"); ax[1].set_title("window spectra")
    fig.tight_layout(); save(fig, "ch02_windows")


def fig_bandwidth():
    f = np.linspace(-4, 4, 8001)
    S = np.sinc(f) ** 2
    Sdb = 10 * np.log10(S + 1e-12)
    fw = np.linspace(-20000, 20000, 4000001); Sw = np.sinc(fw) ** 2      # wide grid: the tails matter
    cum = np.cumsum(Sw) / np.sum(Sw)
    lo, hi = fw[np.searchsorted(cum, 0.005)], fw[np.searchsorted(cum, 0.995)]
    c90 = fw[np.searchsorted(cum, 0.95)]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.plot(f, Sdb, color=NAVY)
    ax.axvspan(-0.443, 0.443, color=GREEN, alpha=0.15, label="3 dB bandwidth: $0.89/T$")
    ax.axvline(-1, color=ACCENT, ls="--", lw=0.9); ax.axvline(1, color=ACCENT, ls="--", lw=0.9, label="null-to-null: $2/T$")
    ax.axvspan(-c90, c90, color=ORANGE, alpha=0.12, label=f"90% power: {2 * c90:.2f}/T")
    ax.annotate("", (4, -36), (2.6, -36), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("", (-4, -36), (-2.6, -36), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.text(0, -37.4, f"99% power needs {hi - lo:.1f}/T: far off this plot", ha="center", fontsize=7.5, color=ORANGE,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_ylim(-40, 3); ax.set_xlim(-4, 4); ax.set_xlabel("frequency $\\times T$"); ax.set_ylabel("PSD (dB)")
    ax.set_title("Bandwidth of a rectangular pulse of duration $T$ depends on the definition")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout(); save(fig, "ch02_bandwidth")


def fig_analytic():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1), sharey=True)
    f = np.linspace(-12, 12, 2000)
    shape = lambda x: np.where(np.abs(x) < 1.5, (1.5 - x) / 3 * 1.4 + 0.3, 0)
    Xp = shape(f - 8) + shape(-f - 8)
    ax[0].fill_between(f, Xp, color=NAVY, alpha=0.7); ax[0].set_title("real bandpass $X(f)$")
    ax[1].fill_between(f, 2 * shape(f - 8), color=GREEN, alpha=0.7); ax[1].set_title("analytic $X_+(f)$")
    ax[2].fill_between(f, 2 * shape(f), color=ACCENT, alpha=0.7); ax[2].set_title("complex envelope $\\tilde X(f)$")
    for a in ax:
        a.set_xticks([-8, 0, 8]); a.set_xticklabels(["$-f_c$", "0", "$f_c$"]); a.set_yticks([])
        a.axvline(0, color="k", lw=0.5)
    fig.tight_layout(); save(fig, "ch02_analytic")


def fig_dft_resolution():
    fs = 100
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for N, c in [(64, ACCENT), (256, NAVY)]:
        n = np.arange(N); x = np.cos(2 * np.pi * 20 * n / fs) + np.cos(2 * np.pi * 21.2 * n / fs)
        X = np.fft.rfft(x * np.hanning(N), 4096); f = np.fft.rfftfreq(4096, 1 / fs)
        ax[0].plot(f, 20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-9), color=c, label=f"N = {N} samples")
    ax[0].set_xlim(14, 28); ax[0].set_ylim(-50, 3); ax[0].legend(fontsize=7); ax[0].set_xlabel("Hz")
    ax[0].set_title("Resolution needs observation time")
    N = 32; n = np.arange(N); x = np.cos(2 * np.pi * 20.6 * n / fs)
    X = np.abs(np.fft.rfft(x)); Xz = np.abs(np.fft.rfft(x, 1024))
    ax[1].plot(np.fft.rfftfreq(1024, 1 / fs), Xz, color=GRAY, lw=0.9, label="zero-padded to 1024")
    ax[1].stem(np.fft.rfftfreq(N, 1 / fs), X, linefmt=NAVY, markerfmt="o", basefmt=" ", label="32-point DFT")
    ax[1].set_xlim(5, 35); ax[1].legend(fontsize=7); ax[1].set_xlabel("Hz"); ax[1].set_title("Zero padding interpolates, not resolves")
    fig.tight_layout(); save(fig, "ch02_dft_resolution")


# ----------------------------------------------------------------------------------------
# Figures added in the deepened edition of the chapter
# ----------------------------------------------------------------------------------------

def fig_projection():
    """Signals as vectors: orthogonal projection in the plane and on a function space."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    a = ax[0]
    x = np.array([2.0, 1.6]); y = np.array([2.6, 0.4])
    p = (x @ y) / (y @ y) * y
    for v, c, lab, off in [(x, NAVY, r"$\mathbf{x}$", (-0.25, 0.08)), (y, ACCENT, r"$\mathbf{y}$", (0.05, -0.25))]:
        a.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        a.text(v[0] + off[0], v[1] + off[1], lab, color=c, fontsize=11)
    a.annotate("", p, (0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    a.text(p[0] * 0.45, p[1] * 0.45 - 0.32, r"$\frac{\langle x,y\rangle}{\|y\|^2}\mathbf{y}$", color=GREEN, fontsize=10)
    a.plot([x[0], p[0]], [x[1], p[1]], color=GRAY, ls="--", lw=1)
    a.text((x[0] + p[0]) / 2 + 0.06, (x[1] + p[1]) / 2, "error\n$\\perp\\ \\mathbf{y}$", color=GRAY, fontsize=7.5)
    # right-angle marker
    u = y / np.linalg.norm(y); v = (x - p) / np.linalg.norm(x - p); s = 0.13
    a.plot([p[0] + s * v[0], p[0] + s * v[0] + s * u[0], p[0] + s * u[0]],
           [p[1] + s * v[1], p[1] + s * v[1] + s * u[1], p[1] + s * u[1]], color=GRAY, lw=0.8)
    a.set_aspect("equal"); a.set_xlim(-0.2, 3.0); a.set_ylim(-0.4, 2.0); a.axis("off")
    a.set_title("Projection in the plane", fontsize=9)
    b = ax[1]
    t = np.linspace(0, 1, 1000); dt = t[1] - t[0]
    xt = np.exp(-3 * t) + 0.25 * np.sin(6 * np.pi * t)
    basis = [np.ones_like(t), np.sqrt(3) * (2 * t - 1), np.sqrt(5) * (6 * t ** 2 - 6 * t + 1)]
    b.plot(t, xt, color=NAVY, lw=1.8, label="$x(t)$")
    approx = np.zeros_like(t)
    for k, (phi, c) in enumerate(zip(basis, [ORANGE, GREEN, ACCENT])):
        approx = approx + np.sum(xt * phi) * dt * phi
        e = np.sum((xt - approx) ** 2) * dt / (np.sum(xt ** 2) * dt)
        b.plot(t, approx, color=c, lw=1.1, ls="--" if k < 2 else "-",
               label=f"best fit with {k + 1} basis fn{'s' if k else ''} (error {100 * e:.1f}%)")
    b.set_xlabel("$t$"); b.legend(fontsize=6.5, loc="upper right")
    b.set_title("Projection onto Legendre polynomials", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_projection")


def fig_ft_limit():
    """Fourier series of a pulse train becomes the Fourier transform as the period grows."""
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    tau = 1.0
    f = np.linspace(-4, 4, 2001)
    for a, T0 in zip(ax, [2, 4, 8]):
        k = np.arange(-int(4 * T0), int(4 * T0) + 1)
        ck = tau / T0 * np.sinc(k * tau / T0)
        a.plot(f, tau * np.sinc(f * tau), color=GRAY, lw=0.9, ls="--")
        ml, sl, bl = a.stem(k / T0, T0 * ck, linefmt=NAVY, markerfmt="o", basefmt=" ")
        plt.setp(ml, markersize=2.5); plt.setp(sl, linewidth=0.9)
        a.set_title(f"period $T_0={T0}\\tau$: lines every $1/T_0$", fontsize=8.5)
        a.set_xlabel("frequency $\\times\\tau$"); a.set_xlim(-4, 4)
    ax[0].set_ylabel("$T_0\\,c_k$")
    fig.tight_layout(); save(fig, "ch02_ft_limit")


def fig_clock_emi():
    """Harmonic envelope of a trapezoidal clock and the effect of spread-spectrum clocking."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    f0 = 100e6; T0 = 1 / f0; d = 0.5
    n = np.arange(1, 400, 2)
    fn = n * f0
    a = ax[0]
    for tr, c in [(1e-9, ACCENT), (200e-12, NAVY)]:
        cn = 2 * d * np.abs(np.sinc(n * d)) * np.abs(np.sinc(fn * tr))
        a.plot(fn / 1e9, 20 * np.log10(cn + 1e-12), "o", ms=2.0, color=c, alpha=0.8, label=f"rise time {tr * 1e12:.0f} ps")
        ff = np.logspace(7.5, 10.7, 400)
        env = np.minimum(1.0, np.minimum(1 / (np.pi * d * T0 * ff), 1 / (np.pi * d * T0 * ff) / (np.pi * tr * ff)))
        a.plot(ff / 1e9, 20 * np.log10(2 * d * env), color=c, lw=0.9, ls="--")
        a.axvline(1 / (np.pi * tr) / 1e9, color=c, lw=0.6, ls=":")
    a.set_xscale("log"); a.set_xlim(0.05, 40); a.set_ylim(-93, 5)
    a.set_xlabel("frequency (GHz)"); a.set_ylabel("harmonic amplitude (dB)")
    a.text(0.13, -3, "$-20$ dB/dec", fontsize=7.5, rotation=-14); a.text(5.0, -50, "$-40$ dB/dec", fontsize=7.5, rotation=-28, color=NAVY)
    a.text(0.34, -90, r"$1/\pi t_r$", fontsize=7, color=ACCENT); a.text(1.7, -90, r"$1/\pi t_r$", fontsize=7, color=NAVY)
    a.legend(fontsize=7, loc="lower left"); a.set_title("100 MHz clock, 50% duty: harmonic envelope", fontsize=9)
    # spread-spectrum clocking on the 9th harmonic, measured with a 120 kHz RBW
    b = ax[1]; H = 9
    fs = 40e6; T = 0.002; t = np.arange(int(fs * T)) / fs
    fm, dev = 31.5e3, 0.005 * H * f0             # 0.5 % down-spread on the 9th harmonic = 4.5 MHz
    tri = 2 * np.abs((t * fm) % 1 - 0.5)          # 0..1 triangle
    zs = np.exp(2j * np.pi * np.cumsum(-dev * tri) / fs)
    z0 = np.ones_like(zs)
    Nw = int(1.5 * fs / 120e3)                    # Hann window with ENBW = 120 kHz
    w = np.hanning(Nw)
    def stft(z):
        segs = np.lib.stride_tricks.sliding_window_view(z, Nw)[::Nw // 4]
        return np.abs(np.fft.fftshift(np.fft.fft(segs * w, 8192, axis=1), axes=1)) ** 2 / np.sum(w) ** 2
    fr = np.fft.fftshift(np.fft.fftfreq(8192, 1 / fs)) + H * f0
    S0 = stft(z0).max(axis=0); S1 = stft(zs)
    Spk, Sav = S1.max(axis=0), S1.mean(axis=0)
    b.plot(fr / 1e6, 10 * np.log10(S0 + 1e-12), color=ACCENT, lw=1, label="fixed clock")
    b.plot(fr / 1e6, 10 * np.log10(Spk + 1e-12), color=NAVY, lw=1, label="SSC, peak (max-hold)")
    b.plot(fr / 1e6, 10 * np.log10(Sav + 1e-12), color=GREEN, lw=1, label="SSC, average")
    rpk, rav = 10 * np.log10(S0.max() / Spk.max()), 10 * np.log10(S0.max() / Sav.max())
    print(f"  SSC reduction (9th harmonic, 120 kHz RBW): peak {rpk:.1f} dB, average {rav:.1f} dB")
    b.set_xlim(893, 902); b.set_ylim(-40, 3); b.set_xlabel("frequency (MHz)"); b.set_ylabel("dB (120 kHz RBW)")
    b.legend(fontsize=6.5, loc="upper left"); b.set_title("9th harmonic with 0.5% down-spread SSC", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_clock_emi")


def _rms_widths(t, x, f, X):
    pt = np.abs(x) ** 2; pt = pt / pt.sum()
    pf = np.abs(X) ** 2; pf = pf / pf.sum()
    mt = np.sum(t * pt); mf = np.sum(f * pf)
    return np.sqrt(np.sum((t - mt) ** 2 * pt)), np.sqrt(np.sum((f - mf) ** 2 * pf))


def fig_uncertainty():
    """Time-bandwidth product: Gaussian minimises sigma_t sigma_f."""
    N = 2 ** 16; dt = 1 / 256; t = (np.arange(N) - N // 2) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))
    pulses = [("Gaussian", lambda tt: np.exp(-np.pi * tt ** 2 / 1.0), NAVY),
              ("raised-cosine (Hann) pulse", lambda tt: np.where(np.abs(tt) < 1.0, 0.5 * (1 + np.cos(np.pi * tt / 1.0)), 0), GREEN),
              ("half-sine pulse", lambda tt: np.where(np.abs(tt) < 0.75, np.cos(np.pi * tt / 1.5), 0), ORANGE),
              ("rectangle", lambda tt: np.where(np.abs(tt) < 0.5, 1.0, 0), ACCENT)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for name, fn, c in pulses:
        x = fn(t); x = x / np.sqrt(np.sum(x ** 2) * dt)
        X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
        st, sf = _rms_widths(t, x, f, X)
        lab = f"{name}: $\\sigma_t\\sigma_f$ = {st * sf:.3f}" if name != "rectangle" else f"{name}: $\\sigma_f\\to\\infty$"
        print(f"  {name}: sigma_t={st:.3f} sigma_f={sf:.3f} product={st * sf:.4f} (1/4pi={1 / (4 * np.pi):.4f})")
        ax[0].plot(t, x, color=c, lw=1.2, label=lab)
        ax[1].plot(f, 20 * np.log10(np.abs(X) / np.abs(X).max() + 1e-12), color=c, lw=1.1)
    ax[0].set_xlim(-1.5, 1.5); ax[0].set_xlabel("time $t$"); ax[0].legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(-0.02, 1.02))
    ax[0].set_title("Four unit-energy pulses", fontsize=9); ax[0].set_ylim(0, 2.1)
    ax[1].set_xlim(0, 6); ax[1].set_ylim(-80, 3); ax[1].set_xlabel("frequency $f$"); ax[1].set_ylabel("dB")
    ax[1].set_title("Their spectra (Gaussian lowest)", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_uncertainty")


def fig_convolution():
    """Convolution as flip, slide, multiply, integrate."""
    dt = 0.002; tau = np.arange(-1.5, 4.0, dt)
    x = np.where((tau >= 0) & (tau < 1), 1.0, 0.0)
    h = lambda tt: np.where(tt >= 0, 2 * np.exp(-tt / 0.5), 0.0)
    tt = np.arange(-0.5, 3.5, dt)
    y = np.array([np.sum(x * h(t0 - tau)) * dt for t0 in tt])
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.6))
    for a, t0 in zip(ax.flat[:3], [0.4, 1.0, 1.8]):
        hh = h(t0 - tau)
        a.plot(tau, x, color=NAVY, label=r"$x(\tau)$")
        a.plot(tau, hh, color=ACCENT, label=r"$h(t-\tau)$")
        a.fill_between(tau, x * hh, color=GREEN, alpha=0.35, label=r"product: area $=y(t)$")
        a.axvline(t0, color=GRAY, lw=0.7, ls=":")
        a.set_xlim(-1.2, 3.2); a.set_ylim(0, 2.2); a.set_xlabel(r"$\tau$")
        a.set_title(f"$t={t0}$: $y(t)$ = {np.sum(x * hh) * dt:.2f}", fontsize=9)
    ax[0, 0].legend(fontsize=6.8, loc="upper right")
    a = ax[1, 1]
    a.plot(tt, y, color=GREEN, lw=1.6, label="$y(t)=x*h$")
    for t0 in [0.4, 1.0, 1.8]:
        a.plot(t0, np.sum(x * h(t0 - tau)) * dt, "o", color=GREEN, ms=4)
    a.plot(tau, x, color=NAVY, lw=0.8, ls="--", label="$x(t)$")
    a.set_xlim(-1.2, 3.2); a.set_ylim(0, 1.2); a.set_xlabel("$t$"); a.legend(fontsize=7)
    a.set_title("output: an $RC$-smoothed pulse", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_convolution")


def fig_group_delay():
    """Phase delay vs group delay, and dispersion of a short pulse."""
    N = 2 ** 15; fs = 200.0; t = (np.arange(N) - N // 4) / fs
    f = np.fft.fftfreq(N, 1 / fs)
    fc, tg, tp = 10.0, 1.2, 0.3
    env = np.exp(-np.pi * (t / 0.6) ** 2)
    x = env * np.cos(2 * np.pi * fc * t)
    # H(f) = exp(-j(2 pi fc tp + 2 pi (f - fc) tg)) on f>0, Hermitian
    ph = 2 * np.pi * (fc * tp + (np.abs(f) - fc) * tg) * np.sign(f)
    y = np.real(np.fft.ifft(np.fft.fft(x) * np.exp(-1j * ph)))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    a.plot(t, x, color=GRAY, lw=0.8, label="input")
    a.plot(t, y, color=NAVY, lw=0.9, label="output")
    a.plot(t, np.exp(-np.pi * ((t - tg) / 0.6) ** 2), color=ACCENT, lw=1.2, ls="--", label=f"envelope delayed by $\\tau_g$={tg}")
    a.annotate("", (tg, 1.12), (0, 1.12), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
    a.text(tg / 2, 1.17, "$\\tau_g$", ha="center", color=ACCENT, fontsize=9)
    a.set_xlim(-0.8, 2.2); a.set_ylim(-1.15, 1.35); a.set_xlabel("time (s)")
    a.legend(fontsize=6.3, loc="lower right", ncol=1)
    a.set_title(f"Envelope moves by $\\tau_g$, carrier phase by $\\tau_p$ ({tp} s)", fontsize=8.5)
    # dispersion: quadratic phase acting on a short baseband pulse
    b = ax[1]
    N2 = 2 ** 14; dt = 0.01; t2 = (np.arange(N2) - N2 // 2) * dt; f2 = np.fft.fftfreq(N2, dt)
    p = np.exp(-np.pi * (t2 / 0.5) ** 2)
    for D, c in [(0, GRAY), (0.15, GREEN), (0.4, NAVY), (0.8, ACCENT)]:
        q = np.fft.ifft(np.fft.fft(p) * np.exp(-1j * np.pi * D * f2 ** 2 * 2 * np.pi))
        b.plot(t2, np.abs(q), color=c, lw=1.2, label=f"$\\beta={D}$" if D else "input")
    b.set_xlim(-4, 4); b.set_xlabel("time (s)"); b.set_ylabel("envelope")
    b.legend(fontsize=7, title="quadratic phase\n$e^{-j2\\pi^2\\beta f^2}$", title_fontsize=6.8)
    b.set_title("Dispersion: linear group delay spreads a pulse", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_group_delay")


def fig_iq_helix():
    """The complex exponential as a helix; the complex envelope of a real QPSK signal."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(W2, 2.7))
    a = fig.add_subplot(1, 3, 1, projection="3d")
    t = np.linspace(0, 2, 600); z = np.exp(2j * np.pi * t)
    a.plot(t, z.real, z.imag, color=NAVY, lw=1.6)
    a.plot(t, z.real, -1.4 * np.ones_like(t), color=ACCENT, lw=0.9)
    a.plot(t, 1.4 * np.ones_like(t), z.imag, color=GREEN, lw=0.9)
    a.set_xlabel("time", labelpad=-10, fontsize=7.5); a.set_ylabel("I", labelpad=-10, fontsize=7.5); a.text2D(0.78, 0.80, "Q", transform=a.transAxes, fontsize=7.5)
    a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
    a.set_ylim(-1.4, 1.4); a.set_zlim(-1.4, 1.4); a.view_init(18, -60)
    a.set_title("$e^{j2\\pi f t}$: a helix\nshadows: cos (red), sin (green)", fontsize=8)
    rg = rng(3)
    sps_ = 16; qp = cl.get_constellation("qpsk")
    s = cl.shape(qp.modulate(cl.random_bits(2 * 40, rg)), cl.rrc_taps(0.35, sps_, 8), sps_)
    s = s[8 * sps_:8 * sps_ + 16 * sps_]; s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    n = np.arange(len(s)); fc = 3.0 / sps_
    xp = np.real(s * np.exp(2j * np.pi * fc * n))
    b = fig.add_subplot(1, 3, (2, 3))
    tt = n / sps_
    b.plot(tt, xp, color=GRAY, lw=0.7, label=r"passband $x(t)=\mathrm{Re}\{\tilde x(t)e^{j2\pi f_c t}\}$")
    b.plot(tt, s.real, color=ACCENT, lw=1.2, label="$x_I(t)$")
    b.plot(tt, s.imag, color=GREEN, lw=1.2, label="$x_Q(t)$")
    b.plot(tt, np.abs(s), color=NAVY, lw=1.4, ls="--", label=r"envelope $|\tilde x(t)|$")
    b.set_xlabel("time (symbols)"); b.set_xlim(0, 16); b.legend(fontsize=6.5, loc="lower right", ncol=2)
    b.set_ylim(-3.6, 2.4)
    b.set_title("RRC-filtered QPSK: the slow complex envelope rides on a fast carrier", fontsize=8.5)
    ins = b.inset_axes([0.005, 0.01, 0.16, 0.3])
    ins.plot(s.real, s.imag, color=NAVY, lw=0.6); ins.plot(s.real[::sps_], s.imag[::sps_], ".", color=ACCENT, ms=3)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_aspect("equal"); ins.set_title("I/Q plane", fontsize=6, pad=1)
    fig.tight_layout(); save(fig, "ch02_iq_helix")


def fig_hilbert():
    """Analytic signal: envelope and instantaneous frequency via the Hilbert transform."""
    from scipy.signal import hilbert
    fs = 2000; t = np.arange(0, 1, 1 / fs)
    a_t = 0.6 + 0.4 * np.cos(2 * np.pi * 3 * t)
    x = a_t * np.cos(2 * np.pi * 40 * t)
    z = hilbert(x)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    a = ax[0]
    a.plot(t, x, color=NAVY, lw=0.8, label="$x(t)$")
    a.plot(t, z.imag, color=GREEN, lw=0.8, alpha=0.8, label=r"$\hat x(t)$ (Hilbert)")
    a.plot(t, np.abs(z), color=ACCENT, lw=1.5, label=r"$|x(t)+j\hat x(t)|$")
    a.set_xlim(0.1, 0.6); a.set_ylim(-1.15, 1.45); a.set_xlabel("time (s)"); a.legend(fontsize=6.5, loc="upper right", ncol=3)
    a.set_title("Envelope from the analytic signal", fontsize=9)
    f_i = 50 + 300 * t
    xc = np.cos(2 * np.pi * np.cumsum(f_i) / fs)
    zc = hilbert(xc)
    inst = np.diff(np.unwrap(np.angle(zc))) * fs / (2 * np.pi)
    b = ax[1]
    b.plot(t[1:], inst, color=NAVY, lw=1.1, label="from analytic-signal phase")
    b.plot(t, f_i, color=ACCENT, lw=0.9, ls="--", label="true $f_i(t)=50+300t$")
    b.set_xlim(0, 1); b.set_ylim(0, 400); b.set_xlabel("time (s)"); b.set_ylabel("Hz"); b.legend(fontsize=7)
    b.set_title("Instantaneous frequency of a chirp", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_hilbert")


def _bw_metrics(f, S):
    """Return dict of bandwidths for a two-sided PSD S(f) sampled on f (uniform grid)."""
    df = f[1] - f[0]
    pk = np.median(S[np.abs(f) < 0.3])     # in-band reference level (robust to estimation ripple)
    above = np.where(S >= pk / 2)[0]
    b3 = (above.max() - above.min() + 1) * df
    above26 = np.where(S >= pk * 10 ** (-2.6))[0]
    b26 = (above26.max() - above26.min() + 1) * df
    cum = np.cumsum(S) / S.sum()
    lo, hi = f[np.searchsorted(cum, 0.005)], f[np.searchsorted(cum, 0.995)]
    bn = S.sum() * df / pk
    return dict(b3=b3, b26=b26, obw=hi - lo, lo=lo, hi=hi, bn=bn,
                f3=(f[above.min()], f[above.max()]), f26=(f[above26.min()], f[above26.max()]))


def fig_bandwidth_defs():
    """Bandwidth definitions on a realistic PSD: RRC QPSK with PA spectral regrowth and a mask."""
    rg = rng(7)
    sps_ = 8; qp = cl.get_constellation("qpsk")
    s = cl.shape(qp.modulate(cl.random_bits(2 * 60000, rg)), cl.rrc_taps(0.22, sps_, 16), sps_)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    y = s - 0.035 * s * np.abs(s) ** 2           # mild third-order compression -> regrowth
    from scipy.signal import welch
    f, S = welch(y, fs=sps_, nperseg=4096, return_onesided=False, window="blackmanharris", detrend=False)
    f = np.fft.fftshift(f); S = np.fft.fftshift(S)
    m = _bw_metrics(f, S)
    Sdb = 10 * np.log10(S / np.median(S[np.abs(f) < 0.3]))
    fig, ax = plt.subplots(figsize=(W1, 3.4))
    ax.plot(f, Sdb, color=NAVY, lw=1.2, label="PSD (RRC $\\beta$=0.22 QPSK, mild PA regrowth)")
    ax.axvspan(*m["f3"], color=GREEN, alpha=0.18, label=f"3 dB: {m['b3']:.2f} $R_s$")
    ax.axvspan(-m["bn"] / 2, m["bn"] / 2, ymin=0.9, ymax=0.97, color=PURPLE, alpha=0.6, label=f"noise-equivalent: {m['bn']:.2f} $R_s$")
    ax.axvline(m["lo"], color=ORANGE, ls="--", lw=1); ax.axvline(m["hi"], color=ORANGE, ls="--", lw=1, label=f"99% occupied: {m['obw']:.2f} $R_s$")
    ax.plot(m["f26"], [-26, -26], color=ACCENT, lw=2.0, solid_capstyle="butt", label=f"$-26$ dB (x-dB): {m['b26']:.2f} $R_s$")
    mask_f = np.array([-3, -1.7, -1.0, -0.7, -0.62, 0.62, 0.7, 1.0, 1.7, 3])
    mask_d = np.array([-55, -55, -40, -30, 1, 1, -30, -40, -55, -55])
    ax.plot(mask_f, mask_d, color="k", lw=1.0, drawstyle="steps-mid", alpha=0.7, label="illustrative emission mask")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-75, 6); ax.set_xlabel("frequency offset / symbol rate $R_s$"); ax.set_ylabel("PSD (dB rel. peak)")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax.set_title("One spectrum, five bandwidths", fontsize=9.5)
    print("  bandwidth defs (x Rs):", {k: (round(v, 3) if np.isscalar(v) else v) for k, v in m.items()})
    fig.tight_layout(); save(fig, "ch02_bandwidth_defs")


def fig_scalloping():
    """Picket-fence effect and scalloping loss for common windows."""
    N = 64
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]; n = np.arange(N)
    for off, c, lab in [(0.0, NAVY, "tone on bin 10"), (0.5, ACCENT, "tone at bin 10.5")]:
        x = np.exp(2j * np.pi * (10 + off) * n / N)
        Xf = np.abs(np.fft.fft(x, 64 * N)) / N
        Xk = np.abs(np.fft.fft(x)) / N
        a.plot(np.arange(64 * N) / 64, 20 * np.log10(Xf + 1e-9), color=c, lw=0.8, alpha=0.6)
        ml, sl, bl = a.stem(n, 20 * np.log10(Xk + 1e-9), linefmt=c, markerfmt="o", basefmt=" ", bottom=-60, label=lab)
        plt.setp(ml, markersize=3)
    a.set_xlim(4, 17); a.set_ylim(-45, 3); a.set_xlabel("DFT bin"); a.set_ylabel("dB")
    a.annotate("$-3.9$ dB", (10, -3.92), (12.6, -8), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    a.legend(fontsize=6.8, loc="lower left"); a.set_title("Rectangular window: the picket fence", fontsize=9)
    b = ax[1]; d = np.linspace(-0.5, 0.5, 201); M = 256; m = np.arange(M)
    for name, w in [("rectangular", np.ones(M)), ("Hann", np.hanning(M)), ("Blackman-Harris", sps.windows.blackmanharris(M)),
                    ("flat-top", sps.windows.flattop(M))]:
        loss = [20 * np.log10(np.abs(np.sum(w * np.exp(2j * np.pi * dd * m / M))) / np.sum(w)) for dd in d]
        b.plot(d, loss, lw=1.2, label=f"{name} ({-min(loss):.2f} dB)")
    b.set_xlabel("tone offset from bin centre (bins)"); b.set_ylabel("dB"); b.set_ylim(-4.3, 0.4)
    b.legend(fontsize=6.5, loc="lower center"); b.set_title("Scalloping loss", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_scalloping")


def fig_spectrogram():
    """STFT time-frequency trade-off: a chirp with short and long windows, and an FSK burst."""
    fs = 8000; t = np.arange(0, 1.0, 1 / fs)
    chirp = np.cos(2 * np.pi * (200 * t + 0.5 * 2600 * t ** 2)) + 0.6 * np.cos(2 * np.pi * 1500 * t) * (t > 0.3) * (t < 0.36)
    rg = rng(5)
    syms = rg.integers(0, 4, 40); Ts = 0.02
    fsk_f = np.repeat(np.array([800, 1200, 1600, 2000])[syms], int(Ts * fs))
    tf = np.arange(len(fsk_f)) / fs
    fsk = np.cos(2 * np.pi * np.cumsum(fsk_f) / fs) * (tf > 0.1) * (tf < 0.7)
    fsk = fsk + 0.02 * rg.standard_normal(len(fsk))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.6), sharey=True)
    for a, sig, nper, ttl in [(ax[0], chirp, 64, "chirp, 8 ms window"), (ax[1], chirp, 1024, "chirp, 128 ms window"),
                              (ax[2], fsk, 128, "4-FSK burst, 16 ms window")]:
        ff, tt, Sx = sps.spectrogram(sig, fs, window="hann", nperseg=nper, noverlap=int(nper * 0.85), nfft=max(nper, 256))
        Sd = 10 * np.log10(Sx + 1e-12); Sd = Sd - Sd.max()
        a.pcolormesh(tt, ff / 1e3, Sd, vmin=-60, vmax=0, cmap="viridis", shading="auto", rasterized=True)
        a.set_title(ttl, fontsize=8.5); a.set_xlabel("time (s)"); a.grid(False)
    ax[0].set_ylabel("frequency (kHz)"); ax[0].set_ylim(0, 3.2)
    fig.tight_layout(); save(fig, "ch02_spectrogram")


def fig_intermod():
    """Two-tone test through a memoryless nonlinearity, and the third-order intercept."""
    fs = 1000.0; N = 2 ** 14; n = np.arange(N)
    f1, f2 = 100.0, 110.0
    a1, a2, a3 = 1.0, 0.05, -0.12
    A = 0.5
    x = A * (np.cos(2 * np.pi * f1 * n / fs) + np.cos(2 * np.pi * f2 * n / fs))
    y = a1 * x + a2 * x ** 2 + a3 * x ** 3
    w = sps.windows.blackmanharris(N)
    Y = np.abs(np.fft.rfft(y * w)) / np.sum(w) * 2
    fr = np.fft.rfftfreq(N, 1 / fs)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    a.plot(fr, 20 * np.log10(Y + 1e-9), color=NAVY, lw=0.9)
    for lab, fx, yy, ha in [("$f_1,f_2$", 105, 5, "center"), ("$2f_1-f_2$", 88, -28, "right"), ("$2f_2-f_1$", 122, -28, "left"),
                            ("$f_2-f_1$", 12, -52, "left"), (r"$2f_1,\,f_1{+}f_2,\,2f_2$", 210, -30, "center"),
                            (r"$3f_1,\,2f_1{+}f_2,\ldots$", 315, -40, "center")]:
        a.text(fx, yy, lab, fontsize=6.5, ha=ha, color=ACCENT)
    a.set_xlim(0, 360); a.set_ylim(-110, 12); a.set_xlabel("frequency (Hz)"); a.set_ylabel("dB")
    a.set_title("Two tones through $y=a_1x+a_2x^2+a_3x^3$", fontsize=9)
    b = ax[1]
    pin = np.linspace(-30, 15, 200); pc = np.linspace(-30, 2.5, 200)
    Ain = 10 ** (pin / 20)
    fund = 20 * np.log10(np.abs(a1 * 10 ** (pc / 20) + 9 / 4 * a3 * (10 ** (pc / 20)) ** 3) + 1e-12)
    im3 = 20 * np.log10(np.abs(3 / 4 * a3) * Ain ** 3 + 1e-12)
    b.plot(pc, fund, color=NAVY, label="fundamental (compresses)")
    b.plot(pin, im3, color=ACCENT, label="IM3 product")
    b.plot(pin, 20 * np.log10(a1 * Ain), color=NAVY, ls=":", lw=0.8)
    b.plot(pin, 20 * np.log10(np.abs(3 / 4 * a3) * Ain ** 3), color=ACCENT, ls=":", lw=0.8)
    iip3 = 20 * np.log10(np.sqrt(4 / 3 * abs(a1 / a3)))
    b.plot(iip3, iip3, "o", color=GREEN); b.annotate("IIP3", (iip3, iip3), (iip3 - 22, iip3 + 2), fontsize=7.5, color=GREEN)
    b.text(-28, -20, "slope 1", fontsize=7, color=NAVY); b.text(-12, -55, "slope 3", fontsize=7, color=ACCENT)
    b.set_xlim(-30, 18); b.set_ylim(-90, 25); b.set_xlabel("input level (dB)"); b.set_ylabel("output level (dB)")
    b.legend(fontsize=6.5, loc="lower right"); b.set_title("Third-order intercept", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_intermod")


def fig_autocorr():
    """Deterministic autocorrelation and ESD: Barker-13 vs a single chip."""
    b13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], float)
    rg = rng(11); rnd = rg.choice([-1.0, 1.0], 13)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    a = ax[0]
    a.step(np.arange(14), np.r_[b13, b13[-1]], where="post", color=NAVY, lw=1.2)
    a.set_ylim(-1.6, 1.6); a.set_xlabel("chip"); a.set_title("Barker-13 code", fontsize=9); a.set_yticks([-1, 0, 1])
    b = ax[1]; lags = np.arange(-12, 13)
    b.plot(lags, np.correlate(rnd, rnd, "full"), "s-", color=GRAY, ms=2.5, lw=0.8, label="random $\\pm1$ code")
    b.plot(lags, np.correlate(b13, b13, "full"), "o-", color=ACCENT, ms=3, lw=1.1, label="Barker-13")
    b.set_xlabel("lag (chips)"); b.set_title("Autocorrelation $R_x(\\tau)$", fontsize=9); b.legend(fontsize=6.5, loc="upper left")
    c = ax[2]; Nf = 4096; sp = 16
    up = np.repeat(b13, sp)
    E = np.abs(np.fft.fft(up, Nf)) ** 2; fr = np.fft.fftfreq(Nf, 1 / sp)
    E1 = np.abs(np.fft.fft(np.ones(sp), Nf)) ** 2 * 13
    o = np.argsort(fr)
    c.plot(fr[o], 10 * np.log10(E[o] / E1.max() + 1e-9), color=ACCENT, lw=0.9, label="Barker-13 ESD")
    c.plot(fr[o], 10 * np.log10(E1[o] / E1.max() + 1e-9), color=NAVY, lw=1.0, ls="--", label="13 $\\times$ one chip")
    c.set_xlim(-3, 3); c.set_ylim(-40, 15); c.set_xlabel("frequency $\\times T_c$"); c.set_ylabel("dB")
    c.legend(fontsize=6.3, loc="lower center"); c.set_title("Energy spectral density", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_autocorr")


ALL = [fig_phasor, fig_fourier_series, fig_pairs, fig_filters, fig_windows, fig_bandwidth, fig_analytic,
       fig_dft_resolution, fig_projection, fig_ft_limit, fig_clock_emi, fig_uncertainty, fig_convolution,
       fig_group_delay, fig_iq_helix, fig_hilbert, fig_bandwidth_defs, fig_scalloping, fig_spectrogram,
       fig_intermod, fig_autocorr]

if __name__ == "__main__":
    import sys as _sys
    want = _sys.argv[1:]
    for fn in ALL:
        if not want or fn.__name__[4:] in want:
            fn()
