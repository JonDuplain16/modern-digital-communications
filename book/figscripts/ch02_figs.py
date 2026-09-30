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
    cum = np.cumsum(S) / np.sum(S)
    lo, hi = f[np.searchsorted(cum, 0.005)], f[np.searchsorted(cum, 0.995)]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.plot(f, Sdb, color=NAVY)
    ax.axvspan(-0.443, 0.443, color=GREEN, alpha=0.15, label="3 dB bandwidth: $0.89/T$")
    ax.axvline(-1, color=ACCENT, ls="--", lw=0.9); ax.axvline(1, color=ACCENT, ls="--", lw=0.9, label="null-to-null: $2/T$")
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.08, label=f"99% power: {hi - lo:.1f}/T")
    ax.set_ylim(-40, 3); ax.set_xlim(-4, 4); ax.set_xlabel("frequency $\\times T$"); ax.set_ylabel("PSD (dB)")
    ax.set_title("Bandwidth of a rectangular pulse of duration $T$ depends on the definition")
    ax.legend(fontsize=7, loc="lower center")
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


if __name__ == "__main__":
    fig_phasor(); fig_fourier_series(); fig_pairs(); fig_filters(); fig_windows(); fig_bandwidth(); fig_analytic(); fig_dft_resolution()
