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


# ----------------------------------------------------------------------------------------
# Second edition: concept illustrations and narrow single-idea figures
# ----------------------------------------------------------------------------------------
NW, NH = 3.0, 2.35     # narrow single-panel size (inches)


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def fig_two_domains():
    """The same chord seen on an oscilloscope (time) and a spectrum analyzer (frequency)."""
    fs = 20000; t = np.arange(0, 0.5, 1 / fs)
    notes = [261.63, 329.63, 392.00]
    x = np.zeros_like(t)
    for f0 in notes:
        for k, a in zip(range(1, 6), [1, 0.5, 0.3, 0.15, 0.08]):
            x += a * np.cos(2 * np.pi * k * f0 * t + k)
    x /= np.max(np.abs(x))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    a = ax[0]
    a.plot(t * 1e3, x, color=NAVY, lw=0.9)
    a.set_xlim(0, 25); a.set_xlabel("time (ms)"); a.set_ylabel("amplitude")
    a.set_title("Oscilloscope view: a tangle", fontsize=9)
    b = ax[1]
    w = sps.windows.blackmanharris(len(x))
    X = np.abs(np.fft.rfft(x * w)) / np.sum(w) * 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    b.plot(f, 20 * np.log10(X / X.max() + 1e-9), color=ACCENT, lw=0.9)
    for f0, nm, dx in zip(notes, ["C", "E", "G"], [-40, 0, 40]):
        b.text(f0 + dx, 4, nm, ha="center", fontsize=8, color=NAVY, fontweight="bold")
    b.set_xlim(0, 2100); b.set_ylim(-70, 12); b.set_xlabel("frequency (Hz)"); b.set_ylabel("dB")
    b.set_title("Spectrum view: three notes and their overtones", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_two_domains")


def fig_signal_zoo():
    """Six small panels: the axes along which engineers classify signals."""
    rg = rng(2)
    fig, ax = plt.subplots(2, 3, figsize=(W2, 3.0))
    t = np.linspace(0, 1, 500)
    s = np.sin(2 * np.pi * 2 * t) + 0.4 * np.sin(2 * np.pi * 5 * t + 1)
    a = ax[0, 0]; a.plot(t, s, color=NAVY); a.set_title("continuous time", fontsize=8.5)
    a = ax[0, 1]; n = np.linspace(0, 1, 26); sn = np.sin(2 * np.pi * 2 * n) + 0.4 * np.sin(2 * np.pi * 5 * n + 1)
    a.stem(n, sn, linefmt=NAVY, markerfmt="o", basefmt=" "); a.set_title("discrete time (samples)", fontsize=8.5)
    a = ax[0, 2]; q = np.round(sn * 2) / 2
    a.step(n, q, where="mid", color=ACCENT); a.plot(n, q, "o", color=ACCENT, ms=2.5)
    a.set_title("digital (quantized levels)", fontsize=8.5)
    a = ax[1, 0]; a.plot(t, np.cos(2 * np.pi * 3 * t), color=GREEN, label="deterministic")
    a.plot(t, 0.5 * rg.standard_normal(len(t)) - 2.2, color=GRAY, lw=0.6, label="random")
    a.set_title("deterministic vs random", fontsize=8.5)
    a = ax[1, 1]; a.plot(t, np.sign(np.sin(2 * np.pi * 3 * t)), color=ORANGE, lw=1)
    a.plot(t, 2 * np.exp(-((t - 0.5) / 0.08) ** 2) - 3.2, color=PURPLE)
    a.set_title("periodic vs aperiodic", fontsize=8.5)
    a = ax[1, 2]; z = np.exp(2j * np.pi * 2 * t) * (0.6 + 0.4 * np.cos(2 * np.pi * t))
    a.plot(z.real, z.imag, color=NAVY); a.set_aspect("equal")
    a.set_title("complex: a path in the I/Q plane", fontsize=8.5)
    for a in ax.flat:
        _off(a)
    fig.tight_layout(h_pad=1.2); save(fig, "ch02_signal_zoo")


def fig_three_insults():
    """What every medium does: attenuate, distort, add noise."""
    rg = rng(4)
    sps_ = 40; bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    x = np.repeat(2 * bits - 1.0, sps_)
    t = np.arange(len(x)) / sps_
    b, a_ = sps.butter(2, 0.035)
    att = 0.35 * x
    dis = sps.lfilter(b, a_, att)
    noi = dis + 0.09 * rg.standard_normal(len(x))
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.6), sharey=True)
    for a, y, ttl, c in zip(ax, [x, att, dis, noi], ["sent", "1. attenuated", "2. distorted", "3. noise added"],
                            [NAVY, GREEN, ORANGE, ACCENT]):
        a.plot(t, y, color=c, lw=1.0); a.set_title(ttl, fontsize=8.5); _off(a)
        a.axhline(0, color=GRAY, lw=0.4)
    ax[0].set_ylim(-1.2, 1.2)
    fig.tight_layout(w_pad=0.6); save(fig, "ch02_three_insults")


def fig_peak_avg():
    """Radar pulse train: peak power vs average power (worked example)."""
    fig, ax = plt.subplots(figsize=(NW, NH))
    t = np.linspace(0, 3.2, 6000)          # ms
    p = np.where((t % 1.0) < 0.03, 200.0, 0.0)  # width exaggerated 30x for visibility
    ax.fill_between(t, np.maximum(p, 1e-3), 1e-3, color=NAVY, alpha=0.8, label="instantaneous power")
    ax.axhline(0.2, color=ACCENT, lw=1.5, label="average 0.2 W")
    ax.set_yscale("log"); ax.set_ylim(0.01, 3000)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("power (W)")
    ax.text(1.07, 260, "peak 200 W", fontsize=8, color=NAVY)
    ax.annotate("", (1.55, 0.2), (1.55, 200), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(1.6, 5, "1000x = 30 dB", fontsize=8, color=GRAY)
    ax.set_title("1 $\\mu$s pulses every 1 ms (width not to scale)", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout(); save(fig, "ch02_peak_avg")


def fig_db_ruler():
    """The 'Richter scale for signals': everyday power levels on one dBm axis."""
    items = [(+77, "AM broadcast transmitter, 50 kW"), (+46, "base-station antenna port, 40 W"),
             (+20, "Wi-Fi router, 100 mW"), (0, "0 dBm = 1 mW"),
             (-30, "1 $\\mu$W"), (-60, "1 nW"), (-90, "a good cellular signal at the phone"),
             (-110, "phone at the cell edge"), (-128.5, "GPS L1 C/A at Earth's surface"),
             (-174, "thermal noise in 1 Hz at 290 K")]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.plot([0, 0], [-180, 85], color=NAVY, lw=3, solid_capstyle="butt")
    for v, lab in items:
        ax.plot([-0.08, 0.08], [v, v], color=NAVY, lw=1.2)
        c = ACCENT if v < -100 else NAVY
        ax.text(0.15, v, lab, va="center", fontsize=7.6, color=c)
        ax.text(-0.15, v, (f"{v:+g}" if v else "0") + " dBm", va="center", ha="right", fontsize=7.6)
        ax.text(-1.75, v, _watts(v), va="center", ha="left", fontsize=7, color=GRAY)
    ax.annotate("", (1.85, -174), (1.85, 77), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1))
    ax.text(1.9, -50, "251 dB\n= a factor of\n$1.3\\times10^{25}$", fontsize=7.5, color=ACCENT, va="center")
    ax.set_xlim(-1.8, 2.6); ax.set_ylim(-185, 88); _off(ax)
    ax.set_title("One ruler, 25 orders of magnitude", fontsize=9.5)
    fig.tight_layout(); save(fig, "ch02_db_ruler")


def _watts(dbm):
    w = 10 ** (dbm / 10) / 1000
    for unit, s in [(1e3, "kW"), (1, "W"), (1e-3, "mW"), (1e-6, "$\\mu$W"), (1e-9, "nW"), (1e-12, "pW"),
                    (1e-15, "fW"), (1e-18, "aW"), (1e-21, "zW")]:
        if w >= unit * 0.999:
            return f"{w / unit:.3g} {s}"
    return f"{w:.1e} W"


def fig_db_cascade():
    """Level diagram for the receiver worked example."""
    stages = [("antenna", 0), ("LNA", 18), ("cable", -2), ("mixer", -7), ("IF amp", 31)]
    lev = np.cumsum([-90] + [g for _, g in stages[1:]])
    fig, ax = plt.subplots(figsize=(NW + 0.4, NH))
    xs = np.arange(len(stages))
    ax.plot(xs, lev, "o-", color=NAVY, lw=1.6, ms=5)
    for i, (nm, g) in enumerate(stages):
        ax.text(i, lev[i] + 3, f"{lev[i]:.0f} dBm", ha="center", fontsize=7.5, color=NAVY)
    ax.set_xticks(xs); ax.set_xticklabels([s if not g else f"{s}\n{g:+d} dB" for s, g in stages], fontsize=7.5)
    ax.set_ylabel("signal level (dBm)"); ax.set_ylim(-100, -40)
    ax.set_title("A level diagram: add the gains", fontsize=9)
    fig.tight_layout(); save(fig, "ch02_db_cascade")


def fig_orthogonal_signals():
    """Four orthogonal pairs: the product integrates to zero."""
    t = np.linspace(0, 1, 2000)
    pairs = [("cosine and sine", np.cos(2 * np.pi * 3 * t), np.sin(2 * np.pi * 3 * t)),
             ("two time slots", np.where(t < 0.5, 1.0, 0), np.where(t >= 0.5, 1.0, 0)),
             ("subcarriers 3/T and 4/T", np.cos(2 * np.pi * 3 * t), np.cos(2 * np.pi * 4 * t)),
             ("Walsh codes", np.repeat([1, -1, 1, -1], 500), np.repeat([1, 1, -1, -1], 500))]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True)
    for a, (nm, x, y) in zip(ax, pairs):
        a.plot(t, x + 2.6, color=NAVY, lw=1); a.plot(t, y, color=ACCENT, lw=1)
        p = x * y
        a.fill_between(t, p - 2.8, -2.8, where=p > 0, color=GREEN, alpha=0.45, lw=0)
        a.fill_between(t, p - 2.8, -2.8, where=p < 0, color=ORANGE, alpha=0.45, lw=0)
        a.axhline(-2.8, color=GRAY, lw=0.5)
        a.set_title(nm, fontsize=8); _off(a)
        a.text(0.5, -4.45, f"area = {abs(np.trapezoid(p, t)):.2f}", ha="center", fontsize=7.5, color=GRAY)
    ax[0].text(-0.06, 2.6, "$x$", fontsize=8, color=NAVY, ha="right", va="center")
    ax[0].text(-0.06, 0, "$y$", fontsize=8, color=ACCENT, ha="right", va="center")
    ax[0].text(-0.06, -2.8, "$xy$", fontsize=8, color=GREEN, ha="right", va="center")
    ax[0].set_ylim(-4.7, 4.0)
    fig.tight_layout(w_pad=0.4); save(fig, "ch02_orthogonal_signals")


def fig_bicycle_wheel():
    """A phasor is a point on a spinning wheel; its shadow is a cosine."""
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=2.2)
    for k in range(12):
        ax.plot([0, np.cos(k * np.pi / 6)], [0, np.sin(k * np.pi / 6)], color=GRAY, lw=0.4)
    ang = 0.9
    px, py = np.cos(ang), np.sin(ang)
    ax.annotate("", (px, py), (0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8))
    ax.plot(px, py, "o", color=ACCENT, ms=7)
    ax.annotate("", (1.25, 0.95), (1.05, 0.45), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1,
                                                               connectionstyle="arc3,rad=0.4"))
    ax.text(0.95, 1.08, "spins at $f_0$ rev/s", fontsize=7.5, color=NAVY)
    ax.plot([px, 1.7], [py, py], color=ACCENT, ls=":", lw=0.9)
    tt = np.linspace(0, 2.2, 400)
    x0 = 1.7
    ax.plot(x0 + tt * 1.9, np.sin(ang + 2 * np.pi * 0.9 * tt), color=ACCENT, lw=1.4)
    ax.plot([x0, x0 + 4.3], [0, 0], color="k", lw=0.5)
    ax.text(x0 + 4.35, -0.05, "time", fontsize=7.5, va="center")
    ax.text(x0 + 0.1, -1.35, "height of the reflector, traced over time = a sinusoid", fontsize=7.5, color=ACCENT)
    ax.text(-0.95, -1.35, "$Ae^{j(2\\pi f_0 t+\\phi)}$", fontsize=9, color=ACCENT)
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 6.6); ax.set_ylim(-1.5, 1.3); _off(ax)
    fig.tight_layout(); save(fig, "ch02_bicycle_wheel")


def fig_neg_freq():
    """Complex samples tell +f from -f; a real signal cannot."""
    fs = 1000.0; N = 4096; n = np.arange(N)
    z = np.exp(2j * np.pi * 100 * n / fs) + 0.5 * np.exp(-2j * np.pi * 250 * n / fs)
    w = np.hanning(N)
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    Z = np.fft.fftshift(np.abs(np.fft.fft(z * w))) / w.sum()
    R = np.fft.fftshift(np.abs(np.fft.fft(z.real * w))) / w.sum()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    ax[0].plot(f, 20 * np.log10(Z + 1e-9), color=NAVY, lw=1)
    ax[0].set_title("complex (I and Q): $+100$ and $-250$ Hz are distinct", fontsize=8.5)
    ax[1].plot(f, 20 * np.log10(R + 1e-9), color=ACCENT, lw=1)
    ax[1].set_title("real part only: every tone gets a mirror", fontsize=8.5)
    for a in ax:
        a.set_xlim(-500, 500); a.set_ylim(-70, 5); a.set_xlabel("frequency (Hz)"); a.axvline(0, color="k", lw=0.5)
    ax[0].set_ylabel("dB")
    fig.tight_layout(); save(fig, "ch02_neg_freq")


def fig_phasor_sum():
    """Adding two sinusoids of the same frequency is adding two arrows."""
    fig, ax = plt.subplots(figsize=(NW, NH))
    a = 1.0 * np.exp(0.3j); b = 0.7 * np.exp(1.9j); s = a + b
    def arr(z0, z1, c, lw=1.6):
        ax.annotate("", (z1.real, z1.imag), (z0.real, z0.imag), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))
    arr(0, a, NAVY); arr(a, s, ACCENT); arr(0, s, GREEN, 2.2)
    ax.text(a.real * 0.55, a.imag * 0.55 - 0.15, "path 1", color=NAVY, fontsize=8)
    ax.text((a + s).real / 2 + 0.05, (a + s).imag / 2, "path 2", color=ACCENT, fontsize=8)
    ax.text(-0.55, 1.0, f"sum: length {abs(s):.2f}", color=GREEN, fontsize=8)
    b2 = 0.7 * np.exp(1j * (0.3 + np.pi)); s2 = a + b2
    ax.text(-0.5, -0.75, f"flip path 2 by 180 deg and the\nsum shrinks to {abs(s2):.2f}: a fade", color=GRAY, fontsize=7.5)
    ax.set_aspect("equal"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.95, 1.45)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_title("Same frequency: just add the phasors", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_phasor_sum")


def fig_prism():
    """Fourier analysis as a prism: a waveform split into its sinusoidal colours."""
    fig = plt.figure(figsize=(W2, 2.3))
    t = np.linspace(0, 1, 800)
    comps = [(1, 1.0, NAVY), (3, 0.5, GREEN), (5, 0.3, ORANGE), (7, 0.18, ACCENT)]
    x = sum(a * np.sin(2 * np.pi * k * t) for k, a, _ in comps)
    a0 = fig.add_axes([0.0, 0.25, 0.28, 0.55]); a0.plot(t, x, color="k", lw=1.2); _off(a0)
    a0.set_title("in: one waveform", fontsize=8.5)
    ap = fig.add_axes([0.29, 0.1, 0.17, 0.8]); _off(ap)
    ap.fill([0.1, 0.9, 0.5], [0.1, 0.1, 0.9], color=NAVY, alpha=0.12, ec=NAVY, lw=1.2)
    ap.text(0.5, 0.33, "Fourier\nanalysis", ha="center", fontsize=7.5, color=NAVY)
    ap.set_xlim(0, 1); ap.set_ylim(0, 1)
    for i, (k, a, c) in enumerate(comps):
        ax = fig.add_axes([0.5, 0.74 - 0.2 * i, 0.27, 0.16]); ax.plot(t, a * np.sin(2 * np.pi * k * t), color=c, lw=1.1)
        ax.set_ylim(-1.05, 1.05); _off(ax)
        ax.text(1.02, 0, f"{k} Hz", fontsize=7, color=c, va="center", transform=ax.get_yaxis_transform())
    ab = fig.add_axes([0.84, 0.18, 0.15, 0.62])
    ab.bar([k for k, _, _ in comps], [a for _, a, _ in comps], color=[c for _, _, c in comps], width=1.2)
    ab.set_xticks([1, 3, 5, 7]); ab.set_yticks([]); ab.set_title("spectrum", fontsize=8.5); ab.set_xlabel("Hz", fontsize=7.5)
    ab.spines["left"].set_visible(False); ab.grid(False)
    fig.text(0.635, 0.94, "out: its sinusoids", ha="center", fontsize=8.5)
    save(fig, "ch02_prism")


def fig_eq_bars():
    """A chord shown the way a graphic-equalizer display shows music: power in octave-ish bands."""
    fs = 22050; t = np.arange(0, 1.0, 1 / fs)
    x = np.zeros_like(t)
    for f0 in [130.81, 261.63, 329.63, 392.0]:
        for k in range(1, 12):
            x += (0.7 ** k) * np.cos(2 * np.pi * k * f0 * t + 0.3 * k)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    centres = 31.25 * 2 ** (np.arange(0, 20) / 2)     # half-octave bands, 31 Hz .. 16 kHz
    centres = centres[centres < 10000]
    P = np.array([X[(f >= c / 2 ** 0.25) & (f < c * 2 ** 0.25)].sum() for c in centres])
    Pd = 10 * np.log10(P / P.max() + 1e-12); Pd = np.maximum(Pd, -60)
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    ax.set_facecolor("#10202f")
    for i, v in enumerate(Pd):
        nseg = int((v + 60) / 4)
        for j in range(nseg):
            c = GREEN if j < 9 else (ORANGE if j < 13 else ACCENT)
            ax.add_patch(plt.Rectangle((i + 0.1, -60 + 4 * j + 0.4), 0.8, 3.2, color=c))
    ax.set_xlim(0, len(centres)); ax.set_ylim(-60, 2)
    lbl = [f"{c:.0f}" if c < 1000 else f"{c / 1000:.0f}k" for c in centres]
    ax.set_xticks(np.arange(len(centres))[::2] + 0.5); ax.set_xticklabels(lbl[::2], fontsize=6.5)
    ax.set_xlabel("band centre (Hz)"); ax.set_ylabel("dB"); ax.grid(False)
    ax.set_title("A C-major chord on an equalizer display", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_eq_bars")


def fig_gibbs_zoom():
    """The Gibbs overshoot narrows but never shrinks."""
    t = np.linspace(-0.04, 0.12, 6000)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for N, c in [(9, ORANGE), (49, GREEN), (199, NAVY)]:
        s = sum(4 / (np.pi * k) * np.sin(2 * np.pi * k * t) for k in range(1, N + 1, 2))
        ax.plot(t * 100, s, color=c, lw=1.0, label=f"up to harmonic {N}: peak {s.max():.3f}")
    ax.axhline(1.179, color=ACCENT, ls="--", lw=0.8); ax.text(8.5, 1.19, "1.179", color=ACCENT, fontsize=7.5)
    ax.axhline(1.0, color=GRAY, lw=0.6)
    ax.set_xlim(-1, 12); ax.set_ylim(0.6, 1.3); ax.set_xlabel("time after the jump (% of a period)")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("Gibbs: 9% overshoot, whatever $N$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_gibbs_zoom")


def fig_smoothness():
    """Smooth in time, compact in frequency: square, triangle and a smooth periodic pulse."""
    N = 4096; t = np.arange(N) / N
    sq = np.sign(np.sin(2 * np.pi * t))
    tri = 2 * np.abs(2 * ((t + 0.25) % 1) - 1) - 1
    sm = np.exp(-((t - 0.5) / 0.12) ** 2); sm = sm - sm.mean()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1.3]})
    for x, c, nm in [(sq, ACCENT, "square (jumps): $1/k$"), (tri, NAVY, "triangle (kinks): $1/k^2$"),
                     (sm, GREEN, "smooth bump: faster than any power")]:
        ax[0].plot(t, x / np.abs(x).max(), color=c, lw=1.1)
        C = np.abs(np.fft.rfft(x)) / N * 2
        k = np.arange(1, 40)
        ck = C[k]
        msk = (k % 2 == 1) if nm[0] in "st" and not nm.startswith("smooth") else ck > 1e-7 * C[1:].max()
        ax[1].plot(k[msk], 20 * np.log10(ck[msk] / C[1:].max()), "o-", color=c, ms=2.5, lw=0.8, label=nm)
    ax[0].set_xlabel("time (periods)"); ax[0].set_title("three periodic waveforms", fontsize=8.5); ax[0].set_yticks([-1, 0, 1])
    ax[1].set_xscale("log"); ax[1].set_ylim(-100, 5); ax[1].set_xlabel("harmonic $k$"); ax[1].set_ylabel("dB")
    ax[1].legend(fontsize=6.6, loc="lower left"); ax[1].set_title("harmonic decay", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_smoothness")


def fig_thd_bars():
    """Where a square wave's power lives (worked example)."""
    k = np.arange(1, 16, 2)
    P = 8 / (np.pi ** 2 * k ** 2)
    fig, ax = plt.subplots(figsize=(NW, NH))
    bars = ax.bar(k, 100 * P, color=[NAVY] + [ACCENT] * (len(k) - 1), width=1.3)
    for kk, p in zip(k[:4], P[:4]):
        ax.text(kk, 100 * p + 1.5, f"{100 * p:.1f}%", ha="center", fontsize=7.5)
    ax.text(8, 50, f"harmonics 3 and up:\n{100 * (1 - P[0]):.0f}% of the power", fontsize=8, color=ACCENT)
    ax.set_xticks(k); ax.set_xlabel("harmonic $k$"); ax.set_ylabel("share of power (%)"); ax.set_ylim(0, 92)
    ax.set_title("A $\\pm1$ square wave's power budget", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_thd_bars")


def fig_drum_violin():
    """Short drum hit vs long violin note: short in time = wide in frequency."""
    fs = 8000; t = np.arange(0, 1.0, 1 / fs)
    drum = np.exp(-t / 0.004) * np.cos(2 * np.pi * 440 * t)
    viol = np.where(t < 0.8, np.sin(np.pi * t / 0.8) ** 2, 0) * np.cos(2 * np.pi * 440 * t)
    fig, ax = plt.subplots(2, 2, figsize=(W2, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
    f = np.fft.rfftfreq(16 * len(t), 1 / fs)
    for i, (x, nm, c) in enumerate([(drum, "drum hit (a few ms)", ACCENT), (viol, "violin note (0.8 s)", NAVY)]):
        ax[i, 0].plot(t * 1e3, x, color=c, lw=0.6); ax[i, 0].set_xlim(0, 900 if i else 40)
        ax[i, 0].set_title(nm, fontsize=8.5); ax[i, 0].set_yticks([]); ax[i, 0].set_xlabel("time (ms)", labelpad=0)
        X = np.abs(np.fft.rfft(x, 16 * len(t))); X = 20 * np.log10(X / X.max() + 1e-9)
        ax[i, 1].plot(f, X, color=c, lw=1); ax[i, 1].set_xlim(0, 1200); ax[i, 1].set_ylim(-45, 3)
        ax[i, 1].set_ylabel("dB"); ax[i, 1].set_xlabel("frequency (Hz)", labelpad=0)
        bw = f[X > -3]
        ax[i, 1].set_title(f"spectrum: 3 dB width {bw.max() - bw.min():.0f} Hz", fontsize=8.5)
    fig.tight_layout(h_pad=0.8); save(fig, "ch02_drum_violin")


def fig_delay_phase():
    """Delay is a phase slope."""
    fig, ax = plt.subplots(figsize=(NW, NH))
    f = np.linspace(0, 1000, 200)
    for tau, c in [(0.5e-3, GREEN), (1e-3, NAVY), (2e-3, ACCENT)]:
        ax.plot(f, -360 * f * tau, color=c, label=f"delay {tau * 1e3:g} ms")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("phase (degrees)")
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("A pure delay: phase falls in a straight line", fontsize=8.5)
    ax.text(330, -120, "slope $=-360^\\circ\\times\\tau$", fontsize=7.5, color=NAVY)
    fig.tight_layout(); save(fig, "ch02_delay_phase")


def fig_mod_shift():
    """Modulation theorem: shift up, shift back, filter."""
    f = np.linspace(-12, 12, 3000)
    tri = lambda x: np.maximum(1 - np.abs(x), 0) * (0.7 + 0.5 * np.abs(x))
    fc = 6
    rows = [("baseband message $X(f)$", tri(f), [(0, "")]),
            ("after $\\times\\cos 2\\pi f_c t$: two half-copies", 0.5 * tri(f - fc) + 0.5 * tri(f + fc), []),
            ("after a second $\\times\\cos$: copies at 0 and $\\pm 2f_c$", 0.5 * tri(f) + 0.25 * tri(f - 2 * fc) + 0.25 * tri(f + 2 * fc), [])]
    fig, ax = plt.subplots(3, 1, figsize=(W1 * 0.62, 2.9), sharex=True)
    cols = [NAVY, GREEN, ORANGE]
    for a, (ttl, y, _), c in zip(ax, rows, cols):
        a.fill_between(f, y, color=c, alpha=0.6); a.set_ylim(0, 1.45); a.set_yticks([])
        a.set_title(ttl, fontsize=8, pad=2); a.axvline(0, color="k", lw=0.4)
    ax[2].plot([-1.6, -1.6, 1.6, 1.6], [0, 1.2, 1.2, 0], color=ACCENT, ls="--", lw=1)
    ax[2].text(1.8, 0.95, "low-pass", fontsize=7, color=ACCENT)
    ax[2].set_xlim(-13.3, 13.3); ax[2].set_xticks([-12, -6, 0, 6, 12]); ax[2].set_xticklabels(["$-2f_c$", "$-f_c$", "0", "$f_c$", "$2f_c$"])
    fig.tight_layout(h_pad=0.3); save(fig, "ch02_mod_shift")


def fig_sampling_comb():
    """Multiplying by an impulse train replicates the spectrum."""
    f = np.linspace(-3.5, 3.5, 3000)
    tri = lambda x: np.maximum(1 - np.abs(x) / 0.4, 0)
    fig, ax = plt.subplots(2, 1, figsize=(NW, NH), sharex=True)
    ax[0].fill_between(f, tri(f), color=NAVY, alpha=0.6); ax[0].set_title("spectrum of $x(t)$", fontsize=8)
    y = sum(tri(f - k) for k in range(-4, 5))
    ax[1].fill_between(f, y, color=GREEN, alpha=0.6); ax[1].set_title("after sampling at $f_s$: copies every $f_s$", fontsize=8)
    for a in ax:
        a.set_yticks([]); a.set_ylim(0, 1.3)
    ax[1].set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax[1].set_xticklabels(["$-3f_s$", "$-2f_s$", "$-f_s$", "0", "$f_s$", "$2f_s$", "$3f_s$"], fontsize=7)
    fig.tight_layout(h_pad=0.3); save(fig, "ch02_sampling_comb")


def fig_correlation_find():
    """Correlation finds a known Barker-13 preamble buried in noise."""
    rg = rng(21)
    b13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], float)
    sp_ = 8; pre = np.repeat(b13, sp_)
    N = 700; r = 1.25 * rg.standard_normal(N); start = 330
    r[start:start + len(pre)] += pre
    c = np.correlate(r, pre, "valid") / len(pre)
    fig, ax = plt.subplots(2, 1, figsize=(NW + 0.2, NH + 0.2), sharex=True)
    ax[0].plot(r, color=GRAY, lw=0.6); ax[0].axvspan(start, start + len(pre), color=ACCENT, alpha=0.15)
    ax[0].set_title("received: preamble hidden in noise (SNR $\\approx-2$ dB)", fontsize=8)
    ax[0].set_yticks([])
    ax[1].plot(c, color=NAVY, lw=0.9); ax[1].plot(start, c[start], "o", color=ACCENT, ms=4)
    ax[1].set_title("correlation with the known Barker-13", fontsize=8); ax[1].set_xlabel("sample")
    ax[1].set_yticks([])
    fig.tight_layout(h_pad=0.4); save(fig, "ch02_correlation_find")


def fig_pam_psd():
    """Random data has the spectrum shape of one pulse."""
    rg = rng(9)
    sps_ = 8; a = rg.choice([-1.0, 1.0], 40000)
    from scipy.signal import welch
    fig, ax = plt.subplots(figsize=(NW, NH))
    for p, c, nm in [(np.ones(sps_), ACCENT, "rectangular pulses"), (cl.rrc_taps(0.35, sps_, 8), NAVY, "RRC pulses ($\\beta=0.35$)")]:
        x = np.convolve(np.repeat(a, 1) * 0 + 0, [0]) if False else None
        up = np.zeros(len(a) * sps_); up[::sps_] = a
        s = np.convolve(up, p)
        f, S = welch(s, fs=sps_, nperseg=1024, return_onesided=False)
        o = np.argsort(f)
        P = np.abs(np.fft.fft(p, 4096) / sps_) ** 2; fp = np.fft.fftfreq(4096, 1 / sps_); op = np.argsort(fp)
        ref = 1.0
        ax.plot(f[o], 10 * np.log10(S[o] / ref + 1e-12), color=c, lw=3.0, alpha=0.3)
        ax.plot(fp[op], 10 * np.log10(P[op] / ref + 1e-12), color=c, lw=1.3, ls="--", label=nm)
    ax.set_xlim(0, 3); ax.set_ylim(-60, 8); ax.set_xlabel("frequency / symbol rate"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.8, loc="upper right"); ax.set_title("Simulated PSD (solid) vs $|P(f)|^2$ (dashed)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_pam_psd")


def fig_room_echo():
    """Convolution as smearing each note with the room's echo."""
    rg = rng(13)
    fs = 1000; t = np.arange(0, 3.0, 1 / fs)
    x = np.zeros_like(t)
    notes = [(0.2, 1.0, NAVY), (0.9, 0.6, GREEN), (1.3, 0.8, ORANGE), (2.0, 0.5, PURPLE)]
    for t0, a, _ in notes:
        x[int(t0 * fs)] = a
    th = np.arange(0, 0.9, 1 / fs)
    h = np.exp(-th / 0.18) * (0.4 + rg.standard_normal(len(th)) * 0.5); h[0] = 1.0
    h[int(0.03 * fs)] += 0.6; h[int(0.055 * fs)] += 0.45
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.2), sharex=False, gridspec_kw={"height_ratios": [1, 1, 1.4]})
    for t0, a, c in notes:
        ax[0].plot([t0, t0], [0, a], color=c, lw=2); ax[0].plot(t0, a, "o", color=c, ms=4)
    ax[0].set_xlim(0, 3); ax[0].set_ylim(0, 1.15); ax[0].set_title("input: four clean notes (impulses)", fontsize=8.5)
    ax[1].plot(th, h, color=GRAY, lw=0.7); ax[1].set_xlim(0, 3)
    ax[1].set_title("the room's impulse response $h(t)$: direct sound, early echoes, reverberant tail", fontsize=8.5)
    y = np.zeros(len(t) + len(th))
    for t0, a, c in notes:
        yi = np.zeros_like(y); yi[int(t0 * fs):int(t0 * fs) + len(th)] = a * h
        ax[2].plot(np.arange(len(y)) / fs, yi, color=c, lw=0.6, alpha=0.8)
        y += yi
    ax[2].plot(np.arange(len(y)) / fs, y, color="k", lw=0.5, alpha=0.5)
    ax[2].set_xlim(0, 3); ax[2].set_title("output $y=x*h$: each note smeared by its own copy of $h$, all added", fontsize=8.5)
    ax[2].set_xlabel("time (s)")
    for a in ax:
        a.set_yticks([])
    fig.tight_layout(h_pad=0.5); save(fig, "ch02_room_echo")


def fig_rc_bode():
    """RC low-pass from the worked example: magnitude, phase and step response."""
    RC = 1e-6; f3 = 1 / (2 * np.pi * RC)
    f = np.logspace(3, 8, 400); H = 1 / (1 + 1j * f / f3)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    a = ax[0]; a.semilogx(f, 20 * np.log10(np.abs(H)), color=NAVY, label="$|H|$ (dB)")
    a2 = a.twinx(); a2.semilogx(f, np.angle(H, deg=True), color=ACCENT, lw=1.1); a2.set_ylabel("phase (deg)", color=ACCENT)
    a2.set_ylim(-95, 5); a2.grid(False); a2.spines["right"].set_visible(True)
    a.axvline(f3, color=GRAY, ls=":", lw=0.8); a.text(f3 * 1.15, -35, "$f_3$ = 159 kHz", fontsize=7.5)
    a.set_ylim(-42, 3); a.set_xlabel("frequency (Hz)"); a.set_ylabel("magnitude (dB)", color=NAVY)
    a.set_title("$-3$ dB and $-45^\\circ$ at $f_3$; then $-20$ dB/decade", fontsize=8.5)
    b = ax[1]; t = np.linspace(0, 6e-6, 400); s = 1 - np.exp(-t / RC)
    b.plot(t * 1e6, s, color=GREEN)
    t10, t90 = RC * np.log(1 / 0.9), RC * np.log(10)
    b.axhline(0.1, color=GRAY, lw=0.5, ls=":"); b.axhline(0.9, color=GRAY, lw=0.5, ls=":")
    b.annotate("", (t90 * 1e6, 0.5), (t10 * 1e6, 0.5), arrowprops=dict(arrowstyle="<->", color=ACCENT))
    b.text((t10 + t90) / 2 * 1e6 + 0.9, 0.4, f"$t_r$ = {1e6 * (t90 - t10):.1f} $\\mu$s", ha="center", fontsize=7.5, color=ACCENT)
    b.set_xlabel("time ($\\mu$s)"); b.set_title("step response", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_rc_bode")


def fig_chord_arrival():
    """Group delay: the notes of a chord arrive at different times through a dispersive channel."""
    fs = 4000.0; N = 2 ** 14; t = (np.arange(N) - N // 8) / fs
    f = np.fft.fftfreq(N, 1 / fs)
    tones = [(100, NAVY), (250, GREEN), (400, ACCENT)]
    beta = 2e-6   # group delay = beta*2pi*|f| ... choose tau_g(f) = 0.1 + 1.0e-3*|f| s
    tg = lambda ff: 0.05 + 0.9e-3 * np.abs(ff)
    phase = -2 * np.pi * (0.05 * np.abs(f) + 0.45e-3 * f ** 2) * np.sign(f)
    fig, ax = plt.subplots(2, 1, figsize=(W1 * 0.72, 2.7), sharex=True)
    for f0, c in tones:
        x = np.exp(-np.pi * (t / 0.06) ** 2) * np.cos(2 * np.pi * f0 * t)
        y = np.real(np.fft.ifft(np.fft.fft(x) * np.exp(1j * phase)))
        ax[0].plot(t, x, color=c, lw=0.6); ax[1].plot(t, y, color=c, lw=0.6)
        ax[1].text(tg(f0), 1.15, f"{f0} Hz", color=c, fontsize=7.5, ha="center")
    ax[0].set_title("sent: three notes struck together", fontsize=8.5)
    ax[1].set_title("received: group delay grows with frequency, so the chord arrives as an arpeggio", fontsize=8.5)
    for a in ax:
        a.set_yticks([]); a.set_ylim(-1.2, 1.45)
    ax[1].set_xlim(-0.15, 0.55); ax[1].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.4); save(fig, "ch02_chord_arrival")


def fig_ideal_lpf():
    """The ideal low-pass impulse response is non-causal; delaying and truncating makes it real."""
    W = 1.0; t = np.linspace(-6, 10, 3000)
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    h = 2 * W * np.sinc(2 * W * t)
    ax.plot(t, h, color=GRAY, lw=1, label="ideal: starts at $t=-\\infty$")
    ax.axvspan(-6, 0, color=ACCENT, alpha=0.08); ax.text(-5.7, 1.7, "the future", fontsize=7.5, color=ACCENT)
    td = 3.0
    hd = np.where(t >= 0, 2 * W * np.sinc(2 * W * (t - td)), 0)
    ax.plot(t, hd, color=NAVY, lw=1.2, label=f"delayed by {td:g} s and truncated")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("time $\\times W$"); ax.set_ylim(-0.6, 3.2); ax.legend(fontsize=6.6, loc="upper right")
    ax.set_title("Brick-wall filters must wait", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_ideal_lpf")


def fig_minphase():
    """Same magnitude, different phase: minimum-phase energy arrives first."""
    hmin = np.array([1.0, 0.5]); hmax = np.array([0.5, 1.0])
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.1))
    ax[0].stem([0, 1], hmin, linefmt=NAVY, markerfmt="o", basefmt=" ", label="$1+0.5z^{-1}$ (minimum phase)")
    ax[0].stem([0.12, 1.12], hmax, linefmt=ACCENT, markerfmt="s", basefmt=" ", label="$0.5+z^{-1}$ (maximum phase)")
    ax[0].set_xticks([0, 1]); ax[0].set_xlabel("tap"); ax[0].set_ylim(0, 1.5); ax[0].legend(fontsize=6.3, loc="upper right")
    ax[0].set_title("impulse responses", fontsize=8.5)
    w = np.linspace(0, np.pi, 300)
    for h, c, ls in [(hmin, NAVY, "-"), (hmax, ACCENT, "--")]:
        H = h[0] + h[1] * np.exp(-1j * w)
        ax[1].plot(w / np.pi, 20 * np.log10(np.abs(H)), color=c, ls=ls, lw=1.4)
    ax[1].set_xlabel("frequency / Nyquist"); ax[1].set_ylabel("dB"); ax[1].set_title("identical magnitudes", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_minphase")


def fig_hilbert_spectra():
    """The analytic signal cancels negative frequencies."""
    f = np.linspace(-10, 10, 2000)
    bump = lambda x: np.exp(-((x) / 1.0) ** 2)
    X = bump(f - 6) + bump(f + 6)
    jXh = bump(f - 6) - bump(f + 6)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    for a, y, ttl, c in [(ax[0], X, "$X(f)$", NAVY), (ax[1], jXh, "$j\\hat X(f)=\\mathrm{sgn}(f)X(f)$", ORANGE),
                         (ax[2], X + jXh, "sum: $X_+(f)=2X(f)u(f)$", GREEN)]:
        a.fill_between(f, y, color=c, alpha=0.6); a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.set_title(ttl, fontsize=8.5); a.set_xticks([-6, 0, 6]); a.set_xticklabels(["$-f_0$", "0", "$f_0$"]); a.set_yticks([])
    ax[0].set_ylim(-1.2, 2.2)
    fig.tight_layout(); save(fig, "ch02_hilbert_spectra")


def fig_highway():
    """Complex baseband = describing motion in the highway's frame instead of the Earth's."""
    rg = rng(17)
    sps_ = 64; qp = cl.get_constellation("qpsk")
    s = cl.shape(qp.modulate(cl.random_bits(2 * 30, rg)), cl.rrc_taps(0.35, sps_, 6), sps_)
    s = s[6 * sps_:6 * sps_ + 14 * sps_]; s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    n = np.arange(len(s)); fc = 3.0 / sps_
    xa = s * np.exp(2j * np.pi * fc * n)
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.85, 2.6))
    ax[0].plot(xa.real, xa.imag, color=GRAY, lw=0.45)
    ax[0].set_title("Earth frame: analytic signal\nwhirls round at $f_c$", fontsize=8.5)
    ax[1].plot(s.real, s.imag, color=NAVY, lw=0.9)
    ax[1].plot(s.real[::sps_], s.imag[::sps_], "o", color=ACCENT, ms=4)
    ax[1].set_title("highway frame: complex envelope\nwanders slowly (dots: symbols)", fontsize=8.5)
    for a in ax:
        a.set_aspect("equal"); a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_xlabel("I"); a.set_ylabel("Q")
        a.axhline(0, color="k", lw=0.4); a.axvline(0, color="k", lw=0.4)
    fig.tight_layout(); save(fig, "ch02_highway")


def fig_two_path_fade():
    """Two-path interference at 5 GHz: a fade every 3 cm (worked example)."""
    lam = 3e8 / 5e9
    x = np.linspace(0, 0.15, 1500)
    s = np.abs(1 + 0.8 * np.exp(-2j * np.pi * x / lam))
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(x * 100, 20 * np.log10(s / 1.8), color=NAVY)
    ax.set_xlabel("receiver displacement (cm)"); ax.set_ylabel("power (dB)")
    ax.set_title("Two paths at 5 GHz: nulls every $\\lambda$ = 6 cm,\npeaks and nulls 3 cm apart", fontsize=8.5)
    ax.set_ylim(-25, 2)
    fig.tight_layout(); save(fig, "ch02_two_path_fade")


def fig_occupied_bw():
    """Cumulative power vs bandwidth: why 99% is brutal for rectangular pulses."""
    f = np.linspace(0, 30, 300001)
    Sr = np.sinc(f) ** 2
    from scipy.integrate import cumulative_trapezoid
    Cr = cumulative_trapezoid(Sr, f, initial=0); Cr /= 0.5
    b = 0.22; fn = np.abs(f)
    Hrc = np.where(fn <= (1 - b) / 2, 1, np.where(fn <= (1 + b) / 2, 0.5 * (1 + np.cos(np.pi / b * (fn - (1 - b) / 2))), 0))
    Cc = cumulative_trapezoid(Hrc, f, initial=0); Cc /= Cc[-1]
    fig, ax = plt.subplots(figsize=(NW + 0.2, NH))
    ax.semilogx(2 * f[1:], 100 * Cr[1:], color=ACCENT, label="rectangular pulse")
    ax.semilogx(2 * f[1:], 100 * Cc[1:], color=NAVY, label="RRC, $\\beta=0.22$")
    ax.axhline(99, color=GRAY, ls="--", lw=0.8); ax.text(0.12, 90, "99%", fontsize=7.5, color=GRAY)
    ax.axvline(20.5, color=ACCENT, lw=0.6, ls=":"); ax.text(11, 70, "20.5/T", color=ACCENT, fontsize=7.5)
    ax.axvline(1.08, color=NAVY, lw=0.6, ls=":"); ax.text(1.15, 60, "1.08/T", color=NAVY, fontsize=7.5)
    ax.set_xlim(0.1, 50); ax.set_ylim(0, 102); ax.set_xlabel("two-sided bandwidth $\\times T$"); ax.set_ylabel("power captured (%)")
    ax.legend(fontsize=6.8, loc="lower right"); ax.set_title("How much bandwidth holds 99%?", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_occupied_bw")


def fig_dft_wrap():
    """The DFT's implicit periodic repetition creates a jump unless the cycles fit."""
    N = 32; n = np.arange(3 * N)
    fig, ax = plt.subplots(2, 1, figsize=(W1 * 0.75, 2.4), sharex=True)
    for a, cyc, c, ttl in [(ax[0], 4.0, NAVY, "4 whole cycles per block: the repetition is seamless"),
                           (ax[1], 4.5, ACCENT, "4.5 cycles: a jump at every block edge = leakage")]:
        blk = np.cos(2 * np.pi * cyc * np.arange(N) / N)
        y = np.tile(blk, 3)
        a.plot(n, y, color=c, lw=1.0)
        for e in [N, 2 * N]:
            a.axvline(e - 0.5, color=GRAY, ls="--", lw=0.7)
        a.set_title(ttl, fontsize=8.5); a.set_yticks([])
        a.axvspan(N - 0.5, 2 * N - 0.5, color=c, alpha=0.07)
    ax[1].set_xlabel("sample (the shaded block is what was measured)")
    fig.tight_layout(h_pad=0.4); save(fig, "ch02_dft_wrap")


def fig_window_frame():
    """Windows in time: how each frame fades the edges of the view."""
    N = 256; n = np.arange(N)
    fig, ax = plt.subplots(figsize=(NW + 0.2, NH))
    for name, w, c in [("rectangular", np.ones(N), ACCENT), ("Hann", np.hanning(N), NAVY),
                       ("Blackman-Harris", sps.windows.blackmanharris(N), GREEN), ("flat-top", sps.windows.flattop(N), ORANGE)]:
        ax.plot(n / N, w, color=c, lw=1.3, label=name)
    ax.set_xlabel("position in the block"); ax.set_ylim(-0.1, 1.55); ax.legend(fontsize=6.6, loc="upper center", ncol=2)
    ax.set_title("Four windows: how fast the frame fades out", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_window_frame")


def fig_fft_ops():
    """Direct DFT vs FFT operation counts."""
    N = 2 ** np.arange(3, 17)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.loglog(N, N.astype(float) ** 2, "o-", color=ACCENT, ms=3, label="direct DFT: $N^2$")
    ax.loglog(N, N / 2 * np.log2(N), "s-", color=NAVY, ms=3, label="FFT: $\\frac{N}{2}\\log_2 N$")
    ax.annotate("N = 4096:\n680x fewer", (4096, 4096 / 2 * 12), (700, 40), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("transform length $N$"); ax.set_ylabel("complex multiplications")
    ax.legend(fontsize=6.8, loc="upper left"); ax.set_title("Why the FFT changed everything", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_fft_ops")


def fig_welch():
    """One periodogram vs a Welch average: same mean, far less scatter."""
    rg = rng(23)
    fs = 1e6; N = 1024; K = 200
    n = np.arange(N * K)
    x = (rg.standard_normal(len(n)) + 1j * rg.standard_normal(len(n))) * 1e-3 + 3e-4 * np.exp(2j * np.pi * 123e3 * n / fs)
    w = sps.windows.blackmanharris(N)
    segs = x.reshape(K, N) * w
    P = np.abs(np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)) ** 2 / np.sum(w ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs)) / 1e3
    fig, ax = plt.subplots(figsize=(NW + 0.2, NH))
    ax.plot(f, 10 * np.log10(P[0]), color=GRAY, lw=0.5, label="one FFT")
    ax.plot(f, 10 * np.log10(P.mean(axis=0)), color=NAVY, lw=1, label=f"average of {K}")
    ax.set_xlim(-300, 300); ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.8, loc="upper left"); ax.set_title("A weak spur emerges from averaged noise", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_welch")


def fig_rbw():
    """Same input, three resolution bandwidths: the noise floor drops 10 dB per decade of RBW."""
    rg = rng(29)
    fs = 2e6; T = 0.1; n = np.arange(int(fs * T))
    x = 1e-2 * (np.exp(2j * np.pi * 300e3 * n / fs) + 0.3 * np.exp(2j * np.pi * 330e3 * n / fs))
    x = x + 1e-4 * (rg.standard_normal(len(n)) + 1j * rg.standard_normal(len(n)))
    fig, ax = plt.subplots(figsize=(NW + 0.4, NH + 0.1))
    for rbw, c in [(100e3, ACCENT), (10e3, GREEN), (1e3, NAVY)]:
        Nw = int(1.5 * fs / rbw); Nw = min(Nw, len(n))
        w = np.hanning(Nw)
        segs = np.lib.stride_tricks.sliding_window_view(x, Nw)[::max(Nw // 2, 1)][:60]
        P = np.mean(np.abs(np.fft.fft(segs * w, 2 ** 16, axis=1)) ** 2, axis=0) / np.sum(w) ** 2
        f = np.fft.fftfreq(2 ** 16, 1 / fs)
        o = np.argsort(f)
        ax.plot(f[o] / 1e3, 10 * np.log10(P[o]), color=c, lw=0.9, label=f"RBW {rbw / 1e3:g} kHz")
    ax.set_xlim(200, 430); ax.set_ylim(-118, -30)
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("displayed level (dB)")
    ax.legend(fontsize=6.6, loc="upper right"); ax.set_title("Narrower RBW: lower floor, finer detail", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_rbw")


def fig_tf_tiles():
    """Gabor's logons: the STFT tiles the time-frequency plane with cells of fixed area."""
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.2))
    for a, (dt, df, ttl, c) in zip(ax, [(0.125, 2, "short window: sharp in time", ACCENT),
                                        (0.5, 0.5, "long window: sharp in frequency", NAVY)]):
        for i in range(int(round(1 / dt))):
            for j in range(int(4 / df)):
                a.add_patch(plt.Rectangle((i * dt, j * df), dt, df, fill=False, ec=c, lw=0.8))
        a.add_patch(plt.Rectangle((0, 0), dt, df, color=c, alpha=0.35))
        a.set_xlim(0, 1); a.set_ylim(0, 4); a.set_title(ttl, fontsize=8.5)
        a.set_xlabel("time"); a.set_ylabel("frequency"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.text(dt / 2, df / 2 if df < 1 else 0.25, "same\narea", fontsize=6.5, ha="center", va="center", color="k")
    fig.tight_layout(); save(fig, "ch02_tf_tiles")


def fig_ism_waterfall():
    """A synthetic 2.4 GHz-style waterfall: hopping narrowband bursts, wideband packets and a chirp."""
    rg = rng(31)
    fs = 20e6; T = 4e-3; N = int(fs * T); n = np.arange(N); t = n / fs
    x = 0.02 * (rg.standard_normal(N) + 1j * rg.standard_normal(N))
    hop = 625e-6 / 4
    for k in range(int(T / hop)):
        fk = rg.uniform(-9e6, 9e6); i0 = int(k * hop * fs); i1 = int(i0 + 0.7 * hop * fs)
        x[i0:i1] += 0.6 * np.exp(2j * np.pi * fk * t[i0:i1] + 1j * np.pi * 0.3 * np.cumsum(rg.choice([-1, 1], i1 - i0)) / 10)
    for s0, d in [(0.4e-3, 0.5e-3), (2.3e-3, 0.8e-3)]:
        i0, i1 = int(s0 * fs), int((s0 + d) * fs)
        nb = (rg.standard_normal(i1 - i0) + 1j * rg.standard_normal(i1 - i0))
        b = sps.firwin(129, 8e6, fs=fs)
        x[i0:i1] += 0.5 * np.convolve(nb, b, "same") * np.exp(2j * np.pi * -3e6 * t[i0:i1]) * 3
    i0, i1 = int(1.2e-3 * fs), int(3.6e-3 * fs); tc = t[i0:i1] - t[i0]
    x[i0:i1] += 0.4 * np.exp(2j * np.pi * (5e6 * tc + 0.5 * (2e6 / 0.4e-3) * ((tc % 0.4e-3) ** 2))) * 1
    ff, tt, S = sps.spectrogram(x, fs, window="hann", nperseg=256, noverlap=128, return_onesided=False)
    S = np.fft.fftshift(S, axes=0); ff = np.fft.fftshift(ff)
    Sd = 10 * np.log10(S + 1e-12); Sd -= Sd.max()
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.5))
    ax.pcolormesh(ff / 1e6, tt * 1e3, Sd.T, vmin=-55, vmax=0, cmap="viridis", shading="auto", rasterized=True)
    ax.set_xlabel("frequency offset (MHz)"); ax.set_ylabel("time (ms)"); ax.grid(False)
    ax.text(-9.5, 0.62, "packet", color="w", fontsize=7); ax.text(5.3, 1.3, "chirps", color="w", fontsize=7)
    ax.set_title("A synthetic waterfall: hops, packets and chirps", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_ism_waterfall")


def fig_compression():
    """A soft limiter: gain compression and the harmonics it creates."""
    A = np.linspace(0.01, 3, 400)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    t = np.linspace(0, 2, 2000)
    for a_in, c in [(0.3, NAVY), (1.0, GREEN), (2.5, ACCENT)]:
        ax[0].plot(t, np.tanh(a_in * np.sin(2 * np.pi * t)), color=c, lw=1.1, label=f"drive {a_in:g}")
    ax[0].set_title("$y=\\tanh(x)$: big inputs flatten into squares", fontsize=8.5); ax[0].set_xlabel("time (periods)")
    ax[0].legend(fontsize=6.5, loc="lower left")
    k = np.arange(1, 10)
    fs = 2000; tt = np.arange(fs) / fs
    for a_in, c in [(0.3, NAVY), (1.0, GREEN), (2.5, ACCENT)]:
        Y = np.abs(np.fft.rfft(np.tanh(a_in * np.sin(2 * np.pi * tt)))) / fs * 2
        ko = k[::2]
        ax[1].plot(ko, 20 * np.log10(Y[ko] / Y[1] + 1e-12), "o-", color=c, ms=3, lw=0.8)
    ax[1].set_ylim(-80, 5); ax[1].set_xlabel("harmonic $k$"); ax[1].set_ylabel("dBc")
    ax[1].set_title("odd harmonics climb with drive", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_compression")


def fig_im3_receiver():
    """Two strong interferers create an IM3 product on top of a weak wanted channel."""
    f = np.linspace(-1, 9, 4000)
    def line(f0, lvl, w=0.05):
        return lvl + 10 * np.log10(np.exp(-((f - f0) / w) ** 2) + 1e-12)
    floor = -121
    base = np.full_like(f, floor)
    sig = lambda *ls: 10 * np.log10(sum(10 ** (l / 10) for l in ls))
    wanted = line(2.0, -100, 0.15)
    before = sig(base, wanted, line(4.0, -25), line(6.0, -25))
    after = sig(base, wanted, line(4.0, -25), line(6.0, -25), line(2.0, -105, 0.12), line(8.0, -105, 0.12))
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.3))
    ax.plot(f, before, color=GRAY, lw=1.2, label="at the LNA input")
    ax.plot(f, after, color=ACCENT, lw=0.9, label="after the LNA (input-referred)")
    ax.text(2.0, -92, "wanted\n$-100$ dBm", ha="center", fontsize=7.5, color=NAVY)
    ax.text(5.0, -20, "interferers at $f_1, f_2$, $-25$ dBm each", ha="center", fontsize=7.5)
    ax.text(8.0, -98, "IM3 at\n$2f_2-f_1$", ha="center", fontsize=7, color=ACCENT)
    ax.text(2.6, -112, "IM3 at $2f_1-f_2$\nlands on the wanted signal", fontsize=7, color=ACCENT)
    ax.set_ylim(-125, -10); ax.set_xlim(-0.5, 9); ax.set_xticks([2, 4, 6, 8])
    ax.set_xticklabels(["$2f_1-f_2$", "$f_1$", "$f_2$", "$2f_2-f_1$"])
    ax.set_ylabel("dBm"); ax.legend(fontsize=6.6, loc="center left")
    ax.set_title("IIP3 = +15 dBm: two $-25$ dBm interferers make $-105$ dBm of IM3", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_im3_receiver")


def fig_db_numbers():
    """Decibel cheat sheet: power and voltage ratios for common dB values."""
    db = np.array([0.1, 1, 3, 6, 10, 20, 30])
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    y = np.arange(len(db))
    ax.barh(y + 0.18, 10 ** (db / 10), height=0.34, color=NAVY, label="power ratio")
    ax.barh(y - 0.18, 10 ** (db / 20), height=0.34, color=ACCENT, label="voltage ratio")
    for yy, d in zip(y, db):
        ax.text(10 ** (d / 10) * 1.15, yy + 0.18, f"{10 ** (d / 10):g}x" if d >= 10 else f"{10 ** (d / 10):.3g}x", va="center", fontsize=7, color=NAVY)
        ax.text(10 ** (d / 20) * 1.15, yy - 0.18, f"{10 ** (d / 20):.3g}x", va="center", fontsize=7, color=ACCENT)
    ax.set_xscale("log"); ax.set_xlim(0.9, 8000)
    ax.set_yticks(y); ax.set_yticklabels([f"{d:g} dB" for d in db])
    ax.set_xlabel("ratio"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Decibels to know by heart", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_db_numbers")


def fig_signal_distance():
    """Two signals, their difference, and its energy: the distance between signals."""
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * t); y = np.where(t < 0.5, 1.0, -1.0) * 0.85
    e = x - y
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    ax.plot(t, x, color=NAVY, label="$x(t)$"); ax.plot(t, y, color=ACCENT, label="$y(t)$")
    ax.fill_between(t, x, y, color=GREEN, alpha=0.25, label="difference $x-y$")
    Ex = np.trapezoid(x ** 2, t); Ey = np.trapezoid(y ** 2, t); d = np.sqrt(np.trapezoid(e ** 2, t))
    rho = np.trapezoid(x * y, t) / np.sqrt(Ex * Ey)
    ax.text(0.52, 0.72, f"$\\|x\\|$ = {np.sqrt(Ex):.2f}, $\\|y\\|$ = {np.sqrt(Ey):.2f}\n$d(x,y)$ = {d:.2f}\n$\\rho$ = {rho:.2f}"
            f" ($\\theta$ = {np.degrees(np.arccos(rho)):.0f}$^\\circ$)", fontsize=7.5)
    ax.set_xlabel("$t$"); ax.legend(fontsize=6.8, loc="lower left"); ax.set_ylim(-1.2, 1.25)
    ax.set_title("Signals have lengths, distances and angles", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_signal_distance")


def fig_radar_spectrum():
    """Spectrum of a 1 us rectangular burst at 3 GHz (worked example)."""
    f = np.linspace(2995, 3005, 4001)
    S = np.sinc((f - 3000) * 1.0) ** 2
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    ax.plot(f, 10 * np.log10(S + 1e-9), color=NAVY)
    ax.axhline(-13.3, color=ACCENT, ls="--", lw=0.8); ax.text(3001.6, -11.5, "first sidelobe $-13.3$ dB", fontsize=7, color=ACCENT)
    ax.annotate("", (3001, -32), (2999, -32), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(3000, -37.5, "main lobe 2 MHz", ha="center", fontsize=7.5, color=GREEN)
    ax.set_ylim(-40, 3); ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("dB")
    ax.set_title("A 1 $\\mu$s burst of 3 GHz carrier", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_radar_spectrum")


def fig_impulse_flat():
    """Narrower pulses of unit area have flatter spectra; the impulse is the limit."""
    f = np.linspace(0, 5, 1000)
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.0))
    for w, c in [(1.0, ORANGE), (0.4, GREEN), (0.1, NAVY)]:
        t = np.linspace(-1.2, 1.2, 2000)
        ax[0].plot(t, np.where(np.abs(t) < w / 2, 1 / w, 0), color=c, lw=1.2, label=f"width {w:g}")
        ax[1].plot(f, np.abs(np.sinc(f * w)), color=c, lw=1.2)
    ax[0].set_ylim(0, 11); ax[0].set_xlabel("time"); ax[0].set_title("pulses of unit area", fontsize=8.5)
    ax[0].legend(fontsize=6.6)
    ax[1].set_xlabel("frequency"); ax[1].set_title("spectra: flatter as the pulse narrows", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_impulse_flat")


def fig_lti_rules():
    """The two rules of an LTI system: superposition and time invariance."""
    t = np.linspace(0, 4, 2000); dt = t[1] - t[0]
    h = np.where(t < 2, np.exp(-t / 0.3), 0)
    sysf = lambda x: np.convolve(x, h)[:len(t)] * dt
    x1 = np.where((t > 0.2) & (t < 0.6), 1.0, 0); x2 = np.where((t > 1.2) & (t < 1.4), -1.5, 0)
    fig, ax = plt.subplots(2, 2, figsize=(W2 * 0.85, 2.6), sharex=True)
    ax[0, 0].plot(t, x1, color=NAVY); ax[0, 0].plot(t, x2, color=GREEN); ax[0, 0].plot(t, x1 + x2 + 0.0, color=ACCENT, ls=":", lw=1)
    ax[0, 0].set_title("inputs $x_1$, $x_2$ and their sum", fontsize=8)
    ax[0, 1].plot(t, sysf(x1), color=NAVY); ax[0, 1].plot(t, sysf(x2), color=GREEN)
    ax[0, 1].plot(t, sysf(x1 + x2), color=ACCENT, ls=":", lw=1.4)
    ax[0, 1].set_title("outputs: response to the sum = sum of responses", fontsize=8)
    ax[1, 0].plot(t, x1, color=NAVY); ax[1, 0].plot(t, np.roll(x1, 1000), color=PURPLE)
    ax[1, 0].set_title("an input and the same input 2 s later", fontsize=8)
    ax[1, 1].plot(t, sysf(x1), color=NAVY); ax[1, 1].plot(t, sysf(np.roll(x1, 1000)), color=PURPLE)
    ax[1, 1].set_title("outputs: the same response, 2 s later", fontsize=8)
    for a in ax.flat:
        a.set_yticks([])
    ax[1, 0].set_xlabel("time (s)"); ax[1, 1].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.5); save(fig, "ch02_lti_rules")


def fig_iono_delay():
    """Ionospheric group delay vs frequency for TEC = 10 TECU, with GPS L1/L2/L5 marked."""
    f = np.linspace(0.8e9, 2.0e9, 300); TEC = 1e17
    d = 40.3 * TEC / f ** 2
    fig, ax = plt.subplots(figsize=(NW + 0.2, NH))
    ax.plot(f / 1e9, d, color=NAVY, label="group delay (code arrives late)")
    ax.plot(f / 1e9, -d, color=ACCENT, label="phase advance (carrier early)")
    for nm, fx in [("L1", 1.57542e9), ("L2", 1.2276e9), ("L5", 1.17645e9)]:
        dd = 40.3 * TEC / fx ** 2
        ax.plot(fx / 1e9, dd, "o", color=NAVY, ms=4); ax.text(fx / 1e9 + (0.03 if nm != "L5" else -0.03), dd + (0.3 if nm != "L5" else -0.9), f"{nm}: {dd:.2f} m", fontsize=7, ha="left" if nm != "L5" else "right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("extra path (m)"); ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("Ionosphere, 10 TEC units: a $1/f^2$ law", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_iono_delay")


def fig_dtft_periodic():
    """The DTFT repeats every fs; all the information is in one period."""
    f = np.linspace(-2.5, 2.5, 3000)
    bump = lambda x: np.exp(-((x - 0.15) / 0.12) ** 2) + 0.6 * np.exp(-((x + 0.25) / 0.08) ** 2)
    y = sum(bump(f - k) for k in range(-4, 5))
    fig, ax = plt.subplots(figsize=(NW + 0.4, NH - 0.5))
    ax.fill_between(f, y, color=GRAY, alpha=0.4)
    m = np.abs(f) <= 0.5
    ax.fill_between(f[m], y[m], color=NAVY, alpha=0.7)
    ax.axvspan(-0.5, 0.5, color=NAVY, alpha=0.06)
    ax.set_xticks([-2, -1, -0.5, 0, 0.5, 1, 2]); ax.set_xticklabels(["$-2f_s$", "$-f_s$", "", "0", "", "$f_s$", "$2f_s$"])
    ax.text(0, 1.2, "$-f_s/2 \\leq f < f_s/2$", ha="center", fontsize=7.5, color=NAVY)
    ax.set_yticks([]); ax.set_ylim(0, 1.4); ax.set_xlabel("frequency")
    ax.set_title("A sampled signal's spectrum repeats every $f_s$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_dtft_periodic")


def fig_recap():
    """Six-panel visual recap of the chapter's big ideas."""
    fig, ax = plt.subplots(2, 3, figsize=(W2, 3.2))
    th = np.linspace(0, 2 * np.pi, 200); t = np.linspace(0, 1, 500)
    a = ax[0, 0]; a.plot(np.cos(th), np.sin(th), color=GRAY); a.annotate("", (np.cos(1), np.sin(1)), (0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT))
    a.plot(1.3 + t * 2, np.sin(1 + 2 * np.pi * 1.2 * t), color=ACCENT); a.set_aspect("equal"); a.set_title("1. a sinusoid is a spinning phasor", fontsize=7.5)
    a = ax[0, 1]; a.bar([1, 3, 5, 7], [1, 0.33, 0.2, 0.14], color=[NAVY, GREEN, ORANGE, ACCENT]); a.set_title("2. every signal is a sum of sinusoids", fontsize=7.5)
    a = ax[0, 2]; tt = np.linspace(-3, 3, 500)
    a.plot(tt, np.exp(-np.pi * (tt / 0.4) ** 2) + 1.2, color=ACCENT); a.plot(tt, np.exp(-np.pi * (tt / 2.0) ** 2) * 0.9 + 1.2, color=NAVY)
    a.plot(tt, np.exp(-np.pi * (tt * 0.4) ** 2), color=ACCENT); a.plot(tt, np.exp(-np.pi * (tt * 2) ** 2) * 0.9, color=NAVY)
    a.set_title("3. short in time = wide in frequency", fontsize=7.5)
    a = ax[1, 0]; s = np.exp(-t / 0.08) * (t > 0)
    for k, c in zip([0.1, 0.4, 0.6], [NAVY, GREEN, ORANGE]):
        a.plot(t, np.roll(s, int(k * 500)) * (t >= k), color=c)
    a.set_title("4. an LTI system smears with $h(t)$", fontsize=7.5)
    a = ax[1, 1]; z = (0.8 + 0.3 * np.cos(2 * np.pi * 2 * t)) * np.exp(1j * (1.5 * np.sin(2 * np.pi * t)))
    a.plot((z * np.exp(2j * np.pi * 15 * t)).real, (z * np.exp(2j * np.pi * 15 * t)).imag, color=GRAY, lw=0.4)
    a.plot(z.real, z.imag, color=NAVY, lw=1.5); a.set_aspect("equal"); a.set_title("5. strip the carrier: complex envelope", fontsize=7.5)
    a = ax[1, 2]; fb = np.linspace(-8, 8, 800)
    a.plot(fb, 20 * np.log10(np.abs(np.sinc(fb)) + 1e-4), color=ACCENT, lw=0.8)
    W = np.abs(np.sinc(fb) / (1 - fb ** 2 + 1e-9)); W[np.abs(np.abs(fb) - 1) < 1e-3] = 0.5
    a.plot(fb, 20 * np.log10(W + 1e-5), color=NAVY, lw=0.8); a.set_ylim(-80, 3)
    a.set_title("6. the FFT sees through a window", fontsize=7.5)
    for a in ax.flat:
        _off(a)
    fig.tight_layout(h_pad=1.0); save(fig, "ch02_recap")


def fig_wagon_wheel():
    """A fast forward rotation sampled by a camera looks like a slow backward one."""
    fr = 24.0; f_true = 22.0          # rev/s of a spoke pattern vs 24 frames/s
    t = np.linspace(0, 0.5, 2000); tk = np.arange(0, 0.5, 1 / fr)
    ang = (2 * np.pi * f_true * t)
    fig, ax = plt.subplots(figsize=(NW + 0.3, NH))
    ax.plot(t, np.cos(ang), color=GRAY, lw=0.7, label="true: +22 rev/s")
    ax.plot(tk, np.cos(2 * np.pi * f_true * tk), "o", color=ACCENT, ms=4, label="camera frames, 24/s")
    ax.plot(t, np.cos(2 * np.pi * (f_true - fr) * t), color=NAVY, lw=1.4, ls="--", label="what you see: $-2$ rev/s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("spoke height"); ax.set_ylim(-1.2, 1.9)
    ax.legend(fontsize=6.6, loc="upper center", ncol=1)
    ax.set_title("The wagon-wheel effect", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch02_wagon_wheel")


def fig_bb_equiv():
    """An off-centre bandpass filter and its complex baseband equivalent."""
    f = np.linspace(-12, 12, 3000); fc = 7.0; off = 0.6
    Hp = lambda x: 1 / np.sqrt(1 + ((x - off) / 1.0) ** 6)
    H = Hp(f - fc) + Hp(-f - fc)
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.0))
    ax[0].fill_between(f, H, color=NAVY, alpha=0.5); ax[0].axvline(fc, color=ACCENT, ls=":", lw=0.9); ax[0].axvline(-fc, color=ACCENT, ls=":", lw=0.9)
    ax[0].set_title("bandpass $|H(f)|$, tuned slightly high", fontsize=8.5)
    ax[0].set_xticks([-fc, 0, fc]); ax[0].set_xticklabels(["$-f_c$", "0", "$f_c$"])
    fb = np.linspace(-5, 5, 1000)
    ax[1].fill_between(fb, Hp(fb), color=GREEN, alpha=0.6); ax[1].axvline(0, color=ACCENT, ls=":", lw=0.9)
    ax[1].set_title("baseband $|\\tilde H(f)|$: not symmetric,\nso $\\tilde h(t)$ is complex", fontsize=8.5)
    for a in ax:
        a.set_yticks([]); a.set_ylim(0, 1.25)
    fig.tight_layout(); save(fig, "ch02_bb_equiv")


ALL = [fig_phasor, fig_fourier_series, fig_pairs, fig_filters, fig_windows, fig_bandwidth, fig_analytic,
       fig_dft_resolution, fig_projection, fig_ft_limit, fig_clock_emi, fig_uncertainty, fig_convolution,
       fig_group_delay, fig_iq_helix, fig_hilbert, fig_bandwidth_defs, fig_scalloping, fig_spectrogram,
       fig_intermod, fig_autocorr,
       fig_two_domains, fig_signal_zoo, fig_three_insults, fig_peak_avg, fig_db_ruler, fig_db_cascade,
       fig_orthogonal_signals, fig_bicycle_wheel, fig_neg_freq, fig_phasor_sum, fig_prism, fig_eq_bars,
       fig_gibbs_zoom, fig_smoothness, fig_thd_bars, fig_drum_violin, fig_delay_phase, fig_mod_shift,
       fig_sampling_comb, fig_correlation_find, fig_pam_psd, fig_room_echo, fig_rc_bode, fig_chord_arrival,
       fig_ideal_lpf, fig_minphase, fig_hilbert_spectra, fig_highway, fig_two_path_fade, fig_occupied_bw,
       fig_dft_wrap, fig_window_frame, fig_fft_ops, fig_welch, fig_rbw, fig_tf_tiles, fig_ism_waterfall,
       fig_compression, fig_im3_receiver, fig_db_numbers, fig_signal_distance, fig_radar_spectrum, fig_impulse_flat,
       fig_lti_rules, fig_iono_delay, fig_dtft_periodic, fig_recap, fig_wagon_wheel, fig_bb_equiv]

if __name__ == "__main__":
    import sys as _sys
    want = _sys.argv[1:]
    for fn in ALL:
        if not want or fn.__name__[4:] in want:
            fn()
