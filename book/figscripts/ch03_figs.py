"""Figures for Chapter 3: Random Signals and Noise."""
import sys
from figstyle import *
from scipy import stats, signal as sps
from scipy.special import erfc, erfcinv
import commlib as cl

K_B = 1.380649e-23
H_P = 6.62607015e-34


def Q(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def Qinv(p):
    return np.sqrt(2) * erfcinv(2 * np.asarray(p))


# ---------------------------------------------------------------- probability
def fig_clt():
    r = rng(1)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True)
    x = np.linspace(-4, 4, 400)
    for a, n in zip(ax, [1, 2, 4, 12]):
        s = r.uniform(-0.5, 0.5, (200_000, n)).sum(1) / np.sqrt(n / 12)
        a.hist(s, bins=80, density=True, color=NAVY, alpha=0.7)
        a.plot(x, stats.norm.pdf(x), color=ACCENT, lw=1.2)
        a.set_title(f"sum of {n} uniform" + ("s" if n > 1 else ""), fontsize=8.5); a.set_xlim(-4, 4)
    ax[0].set_ylabel("density")
    fig.tight_layout(); save(fig, "ch03_clt")


def fig_clt_tails():
    """Tail convergence of the CLT: exact tails of normalised sums vs the Gaussian Q function."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    x = np.linspace(0, 7, 400)
    # sums of uniforms on [-1/2, 1/2]: exact density by repeated numerical convolution
    d = 1 / 400
    base = np.ones(400) / 400  # probability mass of one uniform on a grid of step d
    pm = base.copy()
    cols = {1: GRAY, 2: ORANGE, 4: GREEN, 12: PURPLE}
    for n in range(1, 13):
        if n > 1:
            pm = np.convolve(pm, base)
        if n in cols:
            grid = (np.arange(len(pm)) - (len(pm) - 1) / 2) * d
            sd = np.sqrt(n / 12)
            tail = np.cumsum(pm[::-1])[::-1]
            xs = grid / sd
            m = (xs >= 0) & (tail > 1e-15)
            ax[0].semilogy(xs[m], tail[m], color=cols[n], lw=1.3, label=f"$n={n}$")
    ax[0].semilogy(x, Q(x), color=NAVY, lw=1.8, ls="--", label="Gaussian $Q(x)$")
    ax[0].set_title("sums of $n$ uniforms", fontsize=9)
    ax[0].set_xlabel("normalised threshold $x$ (std devs)"); ax[0].set_ylabel(r"$P(S_n>x)$")
    ax[0].legend(fontsize=7, loc="lower left")
    # sums of +-1 coin flips: exact binomial tails
    for n, c in [(4, GRAY), (16, ORANGE), (64, GREEN), (1024, PURPLE)]:
        k = np.arange(n + 1)
        s = (2 * k - n) / np.sqrt(n)
        sf = stats.binom.sf(k - 1, n, 0.5)  # P(K >= k)
        m = s >= 0
        ax[1].step(s[m], sf[m], where="post", color=c, lw=1.2, label=f"$n={n}$")
    ax[1].semilogy(x, Q(x), color=NAVY, lw=1.8, ls="--", label="Gaussian $Q(x)$")
    ax[1].set_title(r"sums of $n$ random $\pm1$ (coin flips)", fontsize=9)
    ax[1].set_xlabel("normalised threshold $x$ (std devs)"); ax[1].legend(fontsize=7, loc="lower left")
    ax[0].set_ylim(1e-12, 1); ax[0].set_xlim(0, 7); ax[1].set_xlim(0, 7)
    fig.tight_layout(); save(fig, "ch03_clt_tails")


def fig_chisq():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    x = np.linspace(0.001, 14, 500)
    for k, c in [(1, NAVY), (2, ACCENT), (4, GREEN), (8, ORANGE)]:
        lab = {1: "$k=1$: $X^2$ (square-law)", 2: "$k=2$: $|n|^2$ (exponential)",
               4: "$k=4$: two complex samples", 8: "$k=8$: four complex samples"}[k]
        ax[0].plot(x, stats.chi2.pdf(x, k), color=c, label=lab)
    ax[0].set_ylim(0, 0.6); ax[0].set_xlim(0, 14); ax[0].set_xlabel("$y$")
    ax[0].set_title(r"chi-square densities, $k$ degrees of freedom", fontsize=9); ax[0].legend(fontsize=6.8)
    xdb = np.linspace(-40, 8, 400); xl = 10 ** (xdb / 10)
    for L, c in [(1, NAVY), (2, ACCENT), (4, GREEN), (8, ORANGE)]:
        cdf = stats.gamma.cdf(xl, a=L, scale=1 / L)  # sum of L unit-mean exponentials / L
        ax[1].semilogy(xdb, cdf, color=c, label=f"$L={L}$" + (" (Rayleigh)" if L == 1 else ""))
    ax[1].set_ylim(1e-5, 1); ax[1].set_xlim(-40, 8)
    ax[1].set_xlabel("normalised power (dB re mean)"); ax[1].set_ylabel("$P(\\mathrm{power}<x)$")
    ax[1].set_title("power of a sum of $L$ independent\ncomplex Gaussians (chi-square, $2L$ d.o.f.)", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "ch03_chisq")


def fig_q():
    x = np.linspace(0, 7, 400)
    q = Q(x)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.6))
    ax.semilogy(x, q, color=NAVY, lw=1.8, label=r"$Q(x)$")
    ax.semilogy(x, 0.5 * np.exp(-x ** 2 / 2), color=ACCENT, ls="--", label=r"Chernoff-type bound $\frac{1}{2}e^{-x^2/2}$")
    xx = x[x > 0.5]
    ax.semilogy(xx, np.exp(-xx ** 2 / 2) / (xx * np.sqrt(2 * np.pi)), color=GREEN, ls=":", lw=1.6, label=r"upper bound $\frac{e^{-x^2/2}}{x\sqrt{2\pi}}$")
    ax.set_ylim(1e-12, 1); ax.set_xlabel("$x$"); ax.legend(fontsize=7.5)
    for v in [1e-3, 1e-6, 1e-9]:
        xv = float(Qinv(v))
        ax.plot([xv], [v], "o", color=GRAY, ms=3)
        ax.annotate(f"$Q={v:.0e}$ at $x={xv:.2f}$", (xv, v), (6, 2), textcoords="offset points", fontsize=6.8)
    ax.set_title("The Gaussian tail function and two bounds")
    fig.tight_layout(); save(fig, "ch03_q")


def fig_qapprox():
    x = np.linspace(0.05, 8, 600)
    q = Q(x)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for y, lab, c, ls in [(0.5 * np.exp(-x ** 2 / 2), r"$\frac{1}{2}e^{-x^2/2}$ (Craig)", ACCENT, "--"),
                          (np.exp(-x ** 2 / 2), r"$e^{-x^2/2}$ (Chernoff)", ORANGE, "-."),
                          (phi / x, r"$\phi(x)/x$ (upper)", GREEN, ":"),
                          (phi * x / (1 + x ** 2), r"$\phi(x)\,x/(1+x^2)$ (lower)", PURPLE, "-")]:
        ax[0].plot(x, 10 * np.log10(y / q), color=c, ls=ls, label=lab)
    ax[0].axhline(0, color=NAVY, lw=0.8)
    ax[0].set_ylim(-3, 8); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("bound / $Q(x)$ (dB)")
    ax[0].set_title("bounds: how far from $Q(x)$?", fontsize=9); ax[0].legend(fontsize=6.5)
    a, b = 0.339, 5.510
    bs = phi / ((1 - a) * x + a * np.sqrt(x ** 2 + b))
    ch = np.exp(-x ** 2 / 2) / 12 + np.exp(-2 * x ** 2 / 3) / 4
    asy = phi / x * (1 - 1 / x ** 2 + 3 / x ** 4)
    for y, lab, c in [(bs, "Börjesson–Sundberg", NAVY), (asy, "asymptotic series (3 terms)", GREEN),
                      (ch, "Chiani et al. (two exponentials)", ACCENT)]:
        ax[1].plot(x, 100 * (y - q) / q, color=c, label=lab)
    ax[1].axhline(0, color=GRAY, lw=0.8)
    ax[1].set_ylim(-30, 30); ax[1].set_xlabel("$x$"); ax[1].set_ylabel("relative error (%)")
    ax[1].set_title("closed-form approximations", fontsize=9); ax[1].legend(fontsize=6.8, loc="lower left")
    fig.tight_layout(); save(fig, "ch03_qapprox")


def fig_envelopes():
    r = rng(3); n = 400_000
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    x = np.linspace(0, 4, 400)
    for K, c in [(0, NAVY), (3, GREEN), (10, ACCENT)]:
        s = np.sqrt(K / (K + 1)); sig = np.sqrt(1 / (2 * (K + 1)))
        z = s + sig * (r.standard_normal(n) + 1j * r.standard_normal(n))
        ax[0].hist(np.abs(z), bins=120, density=True, histtype="step", color=c, lw=1.0)
        ax[0].plot(x, stats.rice.pdf(x, s / sig, scale=sig), color=c, lw=1.4, label=f"K = {K}" + (" (Rayleigh)" if K == 0 else ""))
    ax[0].set_xlabel("envelope $|z|$ (unit mean power)"); ax[0].set_xlim(0, 2.5); ax[0].legend(fontsize=7); ax[0].set_title("Rayleigh and Rice envelopes")
    z = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    ax[1].hist(np.angle(z), bins=60, density=True, color=NAVY, alpha=0.6)
    ax[1].axhline(1 / (2 * np.pi), color=ACCENT); ax[1].set_xlabel("phase (rad)"); ax[1].set_title("phase of circular Gaussian: uniform")
    fig.tight_layout(); save(fig, "ch03_envelopes")


# ---------------------------------------------------------------- random processes
def fig_filtered_noise():
    r = rng(2); fs = 1000; N = 1 << 16
    w = r.standard_normal(N)
    b = sps.firwin(129, 50, fs=fs)
    y = np.convolve(w, b, mode="same") * np.sqrt(fs / (2 * 50))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    t = np.arange(300) / fs * 1e3
    ax[0].plot(t, w[:300], color=GRAY, lw=0.6, label="white"); ax[0].plot(t, y[:300], color=NAVY, lw=1.2, label="low-pass filtered")
    ax[0].set_xlabel("time (ms)"); ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_title("sample paths")
    for s, c in [(w, GRAY), (y, NAVY)]:
        f, p = sps.welch(s, fs, nperseg=2048)
        ax[1].plot(f, 10 * np.log10(p), color=c)
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("dB/Hz"); ax[1].set_title("PSD"); ax[1].set_ylim(-60, 0)
    lags = np.arange(-60, 61)
    for s, c in [(w, GRAY), (y, NAVY)]:
        R = np.array([np.mean(s[60:-60] * np.roll(s, -k)[60:-60]) for k in lags])
        ax[2].plot(lags / fs * 1e3, R / R[60], color=c)
    ax[2].set_xlabel("lag (ms)"); ax[2].set_title("autocorrelation")
    fig.tight_layout(); save(fig, "ch03_filtered_noise")


def fig_neb():
    f = np.linspace(0, 4, 4000)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    curves = []
    for n, c in [(1, NAVY), (2, GREEN), (4, ORANGE)]:
        H2 = 1 / (1 + f ** (2 * n))
        bn = (np.pi / (2 * n)) / np.sin(np.pi / (2 * n))
        curves.append((n, bn))
        ax[0].plot(f, H2, color=c, label=f"{n}-pole Butterworth, $B_N={bn:.3f}\\,f_{{3}}$")
        ax[0].plot([0, bn, bn], [1, 1, 0], color=c, lw=0.8, ls=":")
    ax[0].set_xlim(0, 3); ax[0].set_ylim(0, 1.08)
    ax[0].set_xlabel("$f/f_{3\\,\\mathrm{dB}}$"); ax[0].set_ylabel("$|H(f)|^2$")
    ax[0].set_title("power response and its equivalent rectangle", fontsize=9); ax[0].legend(fontsize=6.5)
    nn = np.arange(1, 11)
    ratio = (np.pi / (2 * nn)) / np.sin(np.pi / (2 * nn))
    ax[1].plot(nn, ratio, "o-", color=NAVY, ms=4, label="Butterworth (exact)")
    # Gaussian filter and single-tuned / RRC references
    ax[1].axhline(np.sqrt(np.pi / (4 * np.log(2))) , color=GREEN, ls="--", lw=1, label="Gaussian filter (1.064)")
    ax[1].axhline(1, color=GRAY, lw=0.8, ls=":")
    ax[1].set_xlabel("filter order $n$"); ax[1].set_ylabel("$B_N / f_{3\\,\\mathrm{dB}}$")
    ax[1].set_title("noise bandwidth exceeds 3 dB bandwidth", fontsize=9); ax[1].legend(fontsize=7)
    ax[1].set_ylim(0.95, 1.62)
    fig.tight_layout(); save(fig, "ch03_neb")


def fig_bandpass():
    r = rng(7); fs = 200.0; N = 1 << 15; fc = 20.0; B = 4.0
    w = r.standard_normal(N)
    bpf = sps.firwin(801, [fc - B / 2, fc + B / 2], pass_zero=False, fs=fs)
    n = np.convolve(w, bpf, mode="same")
    n /= np.std(n)
    t = np.arange(N) / fs
    an = sps.hilbert(n)
    env = an * np.exp(-2j * np.pi * fc * t)
    fig = plt.figure(figsize=(W2, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    a0 = fig.add_subplot(gs[0, :])
    sl = slice(4000, 4000 + int(4 * fs))
    tt = t[sl] - t[sl][0]
    a0.plot(tt, n[sl], color=GRAY, lw=0.6, label="bandpass noise $n(t)$")
    a0.plot(tt, np.abs(env[sl]), color=ACCENT, lw=1.3, label=r"envelope $|\tilde n(t)|$")
    a0.plot(tt, -np.abs(env[sl]), color=ACCENT, lw=1.3)
    a0.plot(tt, env[sl].real, color=NAVY, lw=1.0, ls="--", label="$n_I(t)$")
    a0.set_xlabel("time (s)"); a0.legend(fontsize=7, ncol=3, loc="upper right"); a0.set_ylim(-4.5, 5.2)
    a0.set_title(f"narrowband noise: carrier {fc:.0f} Hz, bandwidth {B:.0f} Hz", fontsize=9)
    a1 = fig.add_subplot(gs[1, 0])
    f, p = sps.welch(n, fs, nperseg=4096, return_onesided=False)
    fb, pb = sps.welch(env, fs, nperseg=4096, return_onesided=False)
    fi, pI = sps.welch(env.real, fs, nperseg=4096, return_onesided=False)
    o = np.argsort(f)
    a1.plot(f[o], 10 * np.log10(p[o] + 1e-12), color=GRAY, label="passband $n(t)$")
    a1.plot(f[o], 10 * np.log10(pI[o] + 1e-12), color=NAVY, label="$n_I$ (or $n_Q$)")
    a1.plot(f[o], 10 * np.log10(pb[o] + 1e-12), color=ACCENT, lw=1.0, ls="--", label=r"complex $\tilde n$")
    a1.set_xlim(-30, 30); a1.set_ylim(-35, 5); a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("dB/Hz")
    a1.legend(fontsize=6.5, loc="lower center"); a1.set_title("power spectral densities", fontsize=9)
    a2 = fig.add_subplot(gs[1, 1])
    sI = np.std(env.real)
    a2.plot(env.real[::25], env.imag[::25], ".", ms=1.6, color=NAVY, alpha=0.6)
    a2.set_aspect("equal"); a2.set_xlim(-4, 4); a2.set_ylim(-4, 4)
    a2.set_xlabel("$n_I$"); a2.set_ylabel("$n_Q$")
    rho = np.corrcoef(env.real, env.imag)[0, 1]
    a2.set_title(f"I/Q scatter: circular, corr = {rho:+.3f}", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_bandpass")


def fig_cyclo():
    """Spectral correlation: BPSK buried in noise shows features at cycle frequencies; noise does not."""
    r = rng(11)
    Nf, sps_ = 256, 8
    nblk = 3000
    nsym = Nf * nblk // sps_
    taps = cl.rrc_taps(0.35, sps_, 8)
    sym = r.choice([-1.0, 1.0], nsym)
    x = cl.shape(sym, taps, sps_)[: Nf * nblk]
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    k0 = 10  # carrier offset of 10 bins -> conjugate feature at 2*k0 = 20 bins
    nidx = np.arange(len(x))
    x = x * np.exp(2j * np.pi * k0 * nidx / Nf)
    snr_db = -10
    noise = (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2) * 10 ** (-snr_db / 20)
    win = np.hanning(Nf)

    def profiles(sig):
        X = np.fft.fft(sig.reshape(nblk, Nf) * win, axis=1)
        P = np.mean(np.abs(X) ** 2, 0)
        alphas = np.arange(-Nf // 2, Nf // 2)
        nc, cj = [], []
        k = np.arange(Nf)
        for d in alphas:
            # non-conjugate: E[X(k+d) X*(k)]
            S = np.mean(X[:, (k + d) % Nf] * np.conj(X), 0)
            coh = np.abs(S) / np.sqrt(P[(k + d) % Nf] * P)
            nc.append(np.max(coh) if d != 0 else np.nan)
            # conjugate: E[X(k) X(d-k)]
            S2 = np.mean(X * X[:, (d - k) % Nf], 0)
            coh2 = np.abs(S2) / np.sqrt(P * P[(d - k) % Nf])
            cj.append(np.max(coh2))
        return alphas / Nf, np.array(nc), np.array(cj), P

    a_s, nc_s, cj_s, P_s = profiles(x + noise)
    a_n, nc_n, cj_n, P_n = profiles(noise)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    fr = np.fft.fftshift(np.fft.fftfreq(Nf))
    ax[0].plot(fr, 10 * np.log10(np.fft.fftshift(P_s) / np.median(P_n)), color=NAVY, label="BPSK + noise")
    ax[0].plot(fr, 10 * np.log10(np.fft.fftshift(P_n) / np.median(P_n)), color=GRAY, lw=0.8, label="noise only")
    ax[0].set_xlabel("frequency ($f/f_s$)"); ax[0].set_ylabel("PSD (dB)"); ax[0].set_ylim(-1, 4)
    ax[0].set_title("PSD at SNR $-10$ dB", fontsize=9); ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].plot(a_n, nc_n, color=GRAY, lw=0.8); ax[1].plot(a_s, nc_s, color=NAVY)
    ax[1].set_xlabel(r"cycle frequency $\alpha/f_s$"); ax[1].set_ylabel("max spectral coherence")
    ax[1].set_title("non-conjugate", fontsize=9); ax[1].set_ylim(0, 0.6)
    ax[1].annotate(r"$\alpha=\pm R_s$", (1 / sps_, nc_s[np.argmin(abs(a_s - 1 / sps_))]), (8, 4), textcoords="offset points", fontsize=7, color=ACCENT)
    ax[2].plot(a_n, cj_n, color=GRAY, lw=0.8, label="noise only"); ax[2].plot(a_s, cj_s, color=NAVY, label="BPSK + noise")
    ax[2].set_xlabel(r"cycle frequency $\alpha/f_s$"); ax[2].set_title("conjugate", fontsize=9); ax[2].set_ylim(0, 1.0)
    ia = np.argmin(abs(a_s - 2 * k0 / Nf))
    ax[2].annotate(r"$\alpha=2f_0$", (a_s[ia], cj_s[ia]), (6, -4), textcoords="offset points", fontsize=7, color=ACCENT)
    ax[2].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, "ch03_cyclo")


# ---------------------------------------------------------------- physical noise
def fig_planck():
    f = np.logspace(8, 14.5, 600)
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for T, c in [(2.7, PURPLE), (4, NAVY), (20, GREEN), (77, ORANGE), (290, ACCENT)]:
        x = H_P * f / (K_B * T)
        ratio = x / np.expm1(x)
        ax.semilogx(f / 1e9, ratio, color=c, label=f"$T={T:g}$ K")
        f0 = K_B * T / H_P
    ax.axhline(1, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel(r"$\frac{hf/kT}{e^{hf/kT}-1}$")
    ax.set_title("Thermal noise per hertz relative to the classical $kT$", fontsize=9.5)
    ax.text(1.5e5, 0.55, "optical\n193 THz", fontsize=7, color=GRAY, ha="right")
    ax.axvline(193e3, color=GRAY, lw=0.6, ls="--")
    ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(0, 1.08); ax.set_xlim(0.1, 3e5)
    fig.tight_layout(); save(fig, "ch03_planck")


def fig_skynoise():
    """External noise temperature vs frequency: ITU-R P.372 median lines (HF/VHF) and an
    approximate clear-sky model at microwave frequencies."""
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    fM = np.logspace(0, np.log10(250), 200)  # MHz, P.372 range for these lines
    T0 = 290.0
    lines = [("man-made: city", 76.8, 27.7, ACCENT, "-"), ("man-made: residential", 72.5, 27.7, ORANGE, "-"),
             ("man-made: quiet rural", 53.6, 28.6, GREEN, "-"), ("galactic", 52.0, 23.0, NAVY, "-")]
    for lab, c_, d_, col, ls in lines:
        f_ok = fM if "galactic" not in lab else np.logspace(np.log10(10), np.log10(250), 100)
        Fa = c_ - d_ * np.log10(f_ok)
        ax.loglog(f_ok, T0 * 10 ** (Fa / 10), color=col, ls=ls, label=lab)
    # extend galactic beyond 250 MHz with the same spectral slope (synchrotron), dashed
    fg = np.logspace(np.log10(250), np.log10(3e3), 50)
    ax.loglog(fg, T0 * 10 ** ((52.0 - 23.0 * np.log10(fg)) / 10), color=NAVY, ls="--", lw=1.0)
    # microwave clear-sky brightness: simple oxygen + water-vapour opacity model (approximate)
    f = np.logspace(np.log10(500), np.log10(1e5), 600)  # MHz
    g = f / 1e3
    # zenith attenuation in dB (rough fit to ITU-R P.676 trends, mid-latitude, ~7.5 g/m^3 vapour)
    tau_db = (0.033 + 0.00004 * g ** 2 + 0.2 / (1 + ((g - 22.2) / 3.5) ** 2)
              + 140 / (1 + ((g - 60) / 1.8) ** 4) + 3 / (1 + ((g - 118.75) / 1.5) ** 2))
    for el, ls, lab in [(90, "-", "clear sky, zenith"), (10, "--", "clear sky, 10° elevation")]:
        a = 10 ** (-tau_db / np.sin(np.radians(el)) / 10)
        Tsky = 275 * (1 - a) + 2.7 * a
        ax.loglog(f, Tsky, color=PURPLE, ls=ls, lw=1.2, label=lab)
    ax.axhline(2.725, color=GRAY, ls=":", lw=0.9); ax.text(1.1, 3.2, "CMB 2.7 K", fontsize=6.8, color=GRAY)
    ax.axhline(290, color=GRAY, lw=0.6, ls="--"); ax.text(1.1, 340, "$T_0 = 290$ K", fontsize=6.8, color=GRAY)
    ax.fill_betweenx([1, 1e10], 1000, 10000, color=GREEN, alpha=0.07)
    ax.text(1150, 1e7, "microwave\nwindow", fontsize=7, color=GREEN)
    ax.set_ylim(1, 1e10); ax.set_xlim(1, 1e5)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("noise temperature (K)")
    ax.legend(fontsize=6.3, loc="upper right", ncol=1)
    ax.set_title("External noise seen by an antenna", fontsize=9.5)
    fig.tight_layout(); save(fig, "ch03_skynoise")


# ---------------------------------------------------------------- noise in systems
def fig_friis():
    """Cascade noise figure, two orderings: LNA first vs filter/cable first."""
    def cascade(stages):
        F, G = 1.0, 1.0; contrib = []
        for name, g_db, nf_db in stages:
            f = 10 ** (nf_db / 10); g = 10 ** (g_db / 10)
            add = (f - 1) / G
            contrib.append((name, add)); F += add; G *= g
        return 10 * np.log10(F), contrib
    A = [("cable 2 dB", -2, 2), ("filter 1.5 dB", -1.5, 1.5), ("LNA", 20, 1.0), ("mixer", -7, 9), ("IF amp", 25, 5)]
    B = [("LNA", 20, 1.0), ("cable 2 dB", -2, 2), ("filter 1.5 dB", -1.5, 1.5), ("mixer", -7, 9), ("IF amp", 25, 5)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for a, S, title in [(ax[0], A, "LNA after cable and filter"), (ax[1], B, "LNA at the antenna (mast-head)")]:
        nf, con = cascade(S)
        names = [c[0] for c in con]; vals = [c[1] for c in con]
        a.bar(range(len(vals)), vals, color=[ACCENT if "LNA" in n else NAVY for n in names])
        a.set_xticks(range(len(vals))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        a.set_title(f"{title}\ncascade NF = {nf:.2f} dB", fontsize=8.5)
    ax[0].set_ylabel("contribution to $F-1$")
    fig.tight_layout(); save(fig, "ch03_friis")


def fig_lossyline():
    """System noise temperature vs loss ahead of the LNA, for a cold-sky and a warm antenna."""
    Ldb = np.linspace(0, 3, 300); L = 10 ** (Ldb / 10); T0 = 290.0
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for Tant, Tl, c, ls in [(20, 35, NAVY, "-"), (20, 10, GREEN, "-"), (290, 35, ACCENT, "--"), (290, 75, ORANGE, "--")]:
        Tsys = Tant / L + T0 * (1 - 1 / L) + Tl  # referred to LNA input
        lab = f"$T_{{ant}}={Tant}$ K, $T_{{LNA}}={Tl}$ K"
        ax[0].plot(Ldb, Tsys, color=c, ls=ls, label=lab)
        Tsys0 = Tant + Tl
        ax[1].plot(Ldb, 10 * np.log10(Tsys / Tsys0) + Ldb, color=c, ls=ls, label=lab)
    ax[0].set_xlabel("loss ahead of LNA (dB)"); ax[0].set_ylabel("$T_{sys}$ at LNA input (K)")
    ax[0].set_title("system noise temperature", fontsize=9); ax[0].legend(fontsize=6.3, loc=(0.3, 0.42))
    ax[1].set_xlabel("loss ahead of LNA (dB)"); ax[1].set_ylabel("SNR (or $G/T$) degradation (dB)")
    ax[1].set_title("SNR penalty = loss + temperature rise", fontsize=9)
    ax[1].plot(Ldb, Ldb, color=GRAY, lw=0.7, ls=":"); ax[1].text(1.4, 0.4, "warm antenna: penalty = loss", fontsize=6.8, color=ACCENT)
    fig.tight_layout(); save(fig, "ch03_lossyline")


def fig_yfactor():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    Te = 60.0; Tc, Th = 77.0, 290.0
    Ts = np.linspace(-100, 330, 100)
    ax[0].plot(Ts, Ts + Te, color=NAVY)
    ax[0].plot([Tc, Th], [Tc + Te, Th + Te], "o", color=ACCENT, ms=5)
    ax[0].annotate("cold load\n(liquid N$_2$, 77 K)", (Tc, Tc + Te), (-8, 14), textcoords="offset points", fontsize=7, ha="right", bbox=dict(fc="white", ec="none", pad=0.5))
    ax[0].annotate("hot load\n(ambient, 290 K)", (Th, Th + Te), (-10, 10), textcoords="offset points", fontsize=7, ha="right")
    ax[0].plot([-Te], [0], "s", color=GREEN, ms=5)
    ax[0].annotate("intercept at $-T_e$", (-Te, 0), (8, 12), textcoords="offset points", fontsize=7, color=GREEN, bbox=dict(fc="white", ec="none", pad=0.5))
    ax[0].axhline(0, color=GRAY, lw=0.6); ax[0].axvline(0, color=GRAY, lw=0.6)
    ax[0].set_xlabel("source temperature $T_s$ (K)"); ax[0].set_ylabel("output noise / $kGB$ (K)")
    ax[0].set_title(f"Y-factor: two points fix a line ($Y={(Th + Te) / (Tc + Te):.2f}$)", fontsize=9)
    ax[0].set_ylim(-40, 420)
    # sensitivity of NF to a 0.05 dB error in Y
    nf = np.linspace(0.3, 14, 300); F = 10 ** (nf / 10)
    for enr_db, c in [(5, GREEN), (15, NAVY)]:
        enr = 10 ** (enr_db / 10)
        Y = enr / F + 1
        Yp = Y * 10 ** (0.05 / 10)
        Fp = enr / (Yp - 1)
        ax[1].plot(nf, np.abs(10 * np.log10(Fp / F)), color=c, label=f"ENR = {enr_db} dB")
    ax[1].set_xlabel("DUT noise figure (dB)"); ax[1].set_ylabel("NF error (dB)")
    ax[1].set_title("error from a 0.05 dB power-ratio error", fontsize=9); ax[1].legend(fontsize=7)
    ax[1].set_ylim(0, 0.6)
    fig.tight_layout(); save(fig, "ch03_yfactor")


def fig_sensitivity():
    B = np.logspace(2, 9, 200)
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    ax.semilogx(B, -174 + 10 * np.log10(B), color=NAVY, lw=1.6, label="thermal floor $kT_0B$")
    ax.semilogx(B, -174 + 10 * np.log10(B) + 5, color=NAVY, lw=0.9, ls="--", label="floor + 5 dB NF")
    ax.semilogx(B, -174 + 10 * np.log10(B) + 10, color=NAVY, lw=0.9, ls=":", label="floor + 10 dB NF")
    pts = [  # (bandwidth Hz, level dBm, label, colour, offset)
        (2.046e6, -130, "GPS L1 C/A signal\n(received level)", ACCENT, (6, -16)),
        (125e3, -137, "LoRa SF12\n(datasheet)", ACCENT, (6, -4)),
        (125e3, -123, "LoRa SF7", ACCENT, (6, -2)),
        (200e3, -102, "GSM ref. sens.\n(TS 45.005)", GREEN, (-60, 6)),
        (9e6, -97, "LTE 10 MHz REFSENS\n(TS 36.101, band 1)", GREEN, (-100, 5)),
        (20e6, -82, "802.11a 6 Mb/s", PURPLE, (7, -4)),
        (20e6, -65, "802.11a 54 Mb/s", PURPLE, (7, -4)),
    ]
    for b, lvl, lab, c, off in pts:
        ax.plot([b], [lvl], "o", color=c, ms=4.5)
        ax.annotate(lab, (b, lvl), off, textcoords="offset points", fontsize=6.5, color=c)
    ax.set_xlabel("noise bandwidth (Hz)"); ax.set_ylabel("power (dBm)")
    ax.set_ylim(-160, -55); ax.set_xlim(1e2, 1e9)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("Thermal floor versus sensitivity of real receivers", fontsize=9.5)
    fig.tight_layout(); save(fig, "ch03_sensitivity")


# ---------------------------------------------------------------- detection & estimation
def fig_roc():
    from scipy.stats import ncx2, chi2
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    y = np.linspace(-4, 7, 500); d = 2.5; th = 1.6
    ax[0].plot(y, stats.norm.pdf(y), color=NAVY, label="$H_0$: noise")
    ax[0].plot(y, stats.norm.pdf(y, d), color=ACCENT, label="$H_1$: signal + noise")
    ax[0].fill_between(y[y > th], stats.norm.pdf(y[y > th]), color=NAVY, alpha=0.35)
    ax[0].fill_between(y[y > th], stats.norm.pdf(y[y > th], d), color=ACCENT, alpha=0.12)
    ax[0].axvline(th, color=GRAY, ls="--", lw=0.9); ax[0].text(th + 0.1, 0.44, r"$\gamma$", fontsize=8)
    ax[0].text(0, 0.41, "$H_0$", fontsize=8, color=NAVY, ha="center"); ax[0].text(d, 0.41, "$H_1$", fontsize=8, color=ACCENT, ha="center")
    ax[0].text(2.2, 0.03, "$P_{FA}$", fontsize=7, color=NAVY); ax[0].text(3.6, 0.2, "$P_D$", fontsize=7, color=ACCENT)
    ax[0].set_xlabel("test statistic"); ax[0].set_ylim(0, 0.49)
    ax[0].set_title("two hypotheses", fontsize=9)
    pfa = np.logspace(-8, 0, 400)
    for dd, c in [(1, GRAY), (2, GREEN), (3, NAVY), (4, ACCENT)]:
        ax[1].semilogx(pfa, Q(Qinv(pfa) - dd), color=c, label=f"$d={dd}$")
    ax[1].set_xlabel("$P_{FA}$"); ax[1].set_ylabel("$P_D$"); ax[1].set_title("ROC (known signal)", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper left")
    snr_db = np.linspace(0, 22, 300); snr = 10 ** (snr_db / 10); Pfa = 1e-6
    ax[2].plot(snr_db, Q(Qinv(Pfa) - np.sqrt(2 * snr)), color=NAVY, label="coherent")
    thr = chi2.isf(Pfa, 2)
    ax[2].plot(snr_db, ncx2.sf(thr, 2, 2 * snr), color=ACCENT, label="envelope (random phase)")
    for Nn, c in [(16, GREEN)]:
        thrN = chi2.isf(Pfa, 2 * Nn)
        ax[2].plot(snr_db, ncx2.sf(thrN, 2 * Nn, 2 * snr), color=c, ls="--", label=f"energy det., $N={Nn}$")
    ax[2].axhline(0.9, color=GRAY, lw=0.6, ls=":")
    ax[2].set_xlabel("$E/N_0$ (dB)"); ax[2].set_title(r"$P_D$ at $P_{FA}=10^{-6}$", fontsize=9)
    ax[2].legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch03_roc")
    # print reference numbers for the text
    from scipy.optimize import brentq
    f = lambda s: ncx2.sf(thr, 2, 2 * 10 ** (s / 10)) - 0.9
    g = lambda s: Q(Qinv(Pfa) - np.sqrt(2 * 10 ** (s / 10))) - 0.9
    print("  E/N0 for Pd=0.9 Pfa=1e-6: envelope %.2f dB, coherent %.2f dB" % (brentq(f, 0, 30), brentq(g, 0, 30)))


def fig_crb():
    r = rng(5); N = 64; nfft = 1 << 16; trials = 400
    snrs = np.arange(-12, 22, 2.0)
    f0 = 0.1234
    n = np.arange(N)
    mse = []
    for s in snrs:
        A = 1.0; sig2 = A ** 2 / 10 ** (s / 10)
        x = A * np.exp(1j * (2 * np.pi * f0 * n + r.uniform(0, 2 * np.pi, (trials, 1))))
        x = x + np.sqrt(sig2 / 2) * (r.standard_normal((trials, N)) + 1j * r.standard_normal((trials, N)))
        X = np.abs(np.fft.fft(x, nfft, axis=1))
        fh = np.argmax(X, axis=1) / nfft
        e = (fh - f0 + 0.5) % 1 - 0.5
        mse.append(np.mean(e ** 2))
    crb = 6 / ((2 * np.pi) ** 2 * 10 ** (snrs / 10) * N * (N ** 2 - 1))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].semilogy(snrs, mse, "o", color=NAVY, ms=4, label="ML (periodogram peak)")
    ax[0].semilogy(snrs, crb, color=ACCENT, label="Cramér–Rao bound")
    ax[0].semilogy(snrs, np.full_like(snrs, 1 / 12), color=GRAY, ls=":", lw=0.9, label="random guess")
    ax[0].set_xlabel("SNR per sample (dB)"); ax[0].set_ylabel("MSE (cycles/sample)$^2$")
    ax[0].set_title(f"frequency estimation, $N={N}$", fontsize=9); ax[0].legend(fontsize=6.8, loc="lower left")
    ax[0].annotate("threshold", (-4, 3e-4), (-4, 3e-2), textcoords="data", fontsize=7, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT), ha="center")
    # amplitude (DC level) estimation: sample mean vs CRB, MMSE with prior
    Ns = np.unique(np.logspace(0, 3, 20).astype(int)); sig2 = 1.0; prior = 0.25
    mle, mmse = [], []
    for NN in Ns:
        A = r.normal(0, np.sqrt(prior), 4000)
        xbar = A + r.normal(0, np.sqrt(sig2 / NN), 4000)
        mle.append(np.mean((xbar - A) ** 2))
        w = prior / (prior + sig2 / NN)
        mmse.append(np.mean((w * xbar - A) ** 2))
    ax[1].loglog(Ns, mle, "o", color=NAVY, ms=3.5, label="ML: sample mean")
    ax[1].loglog(Ns, sig2 / Ns, color=ACCENT, label=r"CRB $\sigma^2/N$")
    ax[1].loglog(Ns, mmse, "s", color=GREEN, ms=3.5, label="MMSE (Gaussian prior)")
    ax[1].loglog(Ns, 1 / (1 / prior + Ns / sig2), color=GREEN, lw=0.9, ls="--")
    ax[1].axhline(prior, color=GRAY, lw=0.7, ls=":"); ax[1].text(1.2, prior * 1.15, "prior variance", fontsize=6.8, color=GRAY)
    ax[1].set_xlabel("number of samples $N$"); ax[1].set_ylabel("MSE")
    ax[1].set_title("estimating a level in noise", fontsize=9); ax[1].legend(fontsize=6.8, loc="lower left")
    fig.tight_layout(); save(fig, "ch03_crb")


# ---------------------------------------------------------------- link budget
def fig_linkbudget():
    fspl1 = 20 * np.log10(4 * np.pi * 1 * 5.5e9 / 3e8)
    pl = fspl1 + 30 * np.log10(50)
    vals = [20, 3, -pl, -8, 3]
    cum = np.cumsum([0] + vals)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    labels = ["Tx power\n(20 dBm)", "Tx antenna\n+3 dBi", f"path loss\n{-pl:.0f} dB", "wall\n-8 dB", "Rx antenna\n+3 dBi"]
    for i, v in enumerate(vals):
        lo, hi = sorted([cum[i], cum[i + 1]])
        ax.bar(i, hi - lo, bottom=lo, color=GREEN if v > 0 else ACCENT, width=0.6)
    ax.bar(len(vals), cum[-1] - (-110), bottom=-110, color=NAVY, width=0.6)
    noise = -174 + 10 * np.log10(20e6) + 7
    ax.axhline(noise, color=GRAY, ls="--", lw=1); ax.text(-0.4, noise + 2, f"noise floor (20 MHz, NF 7 dB) = {noise:.0f} dBm", fontsize=7)
    ax.annotate("", (len(vals) + 0.45, cum[-1]), (len(vals) + 0.45, noise), arrowprops=dict(arrowstyle="<->", color=NAVY))
    ax.text(len(vals) + 0.55, (cum[-1] + noise) / 2, f"SNR\n{cum[-1] - noise:.0f} dB", fontsize=7.5, color=NAVY, va="center")
    ax.set_xticks(range(len(vals) + 1)); ax.set_xticklabels(labels + [f"received\n{cum[-1]:.0f} dBm"], fontsize=7)
    ax.set_ylabel("power level (dBm)"); ax.set_ylim(-110, 30); ax.set_xlim(-0.6, len(vals) + 1.1)
    ax.set_title("Link budget of an indoor Wi-Fi link as a waterfall")
    fig.tight_layout(); save(fig, "ch03_linkbudget")


# ================================================================ second edition: concept figures
NARROW = (3.0, 2.4)


def fig_party():
    """SNR as a crowded party: one whisper, a crowd of voices, and what reaches the ear."""
    r = rng(21); fs = 4000; t = np.arange(0, 0.5, 1 / fs)
    env = np.exp(-((t - 0.25) / 0.09) ** 2)
    sig = env * np.sin(2 * np.pi * 180 * t) * (1 + 0.4 * np.sin(2 * np.pi * 7 * t))
    crowd = np.zeros_like(t); voices = []
    for k in range(40):
        f = r.uniform(90, 400); ph = r.uniform(0, 2 * np.pi); am = r.uniform(2, 9)
        v = (0.6 + 0.4 * np.sin(2 * np.pi * am * t + r.uniform(0, 6))) * np.sin(2 * np.pi * f * t + ph)
        voices.append(v); crowd += v
    for snr_db, name in [(0, "c")]:
        pass
    crowd *= np.sqrt(np.mean(sig[env > 0.3] ** 2) / np.mean(crowd ** 2)) * 10 ** (3 / 20)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    ax[0].plot(t * 1e3, sig, color=NAVY, lw=0.8); ax[0].set_title("the whisper (signal)", fontsize=9)
    ax[1].plot(t * 1e3, crowd, color=GRAY, lw=0.5); ax[1].set_title("the crowd: 40 voices added", fontsize=9)
    ax[2].plot(t * 1e3, sig + crowd, color=GRAY, lw=0.5, label="what the ear gets")
    ax[2].plot(t * 1e3, sig, color=NAVY, lw=0.9, label="the whisper inside")
    ax[2].set_title("what reaches the ear (SNR $\\approx -3$ dB)", fontsize=9); ax[2].legend(fontsize=6, loc="lower right")
    for a in ax:
        a.set_xlabel("time (ms)"); a.set_yticks([])
    ax[0].set_ylim(-3.4, 3.2)
    fig.tight_layout(); save(fig, "ch03_party")


def fig_floor_ladder():
    """How weak is weak: received powers of real systems against the thermal floor."""
    items = [(0.1, "Wi-Fi access point transmits (20 dBm)", GREEN),
             (1e-6, "strong Wi-Fi signal across a room ($-30$ dBm)", GREEN),
             (1e-13, "phone at the edge of a cell ($\\approx -100$ dBm)", NAVY),
             (1e-16, "GPS signal at the ground ($\\approx -130$ dBm)", ACCENT),
             (8e-15, "thermal noise in GPS's 2 MHz", GRAY),
             (1e-19, "Voyager 1 at a 70 m dish (order of magnitude)", PURPLE),
             (4e-21, "thermal noise in 1 Hz: $kT_0$", GRAY)]
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 2.5))
    for i, (p, lab, c) in enumerate(sorted(items, key=lambda z: -z[0])):
        y = np.log10(p)
        ax.plot([0, 1], [y, y], color=c, lw=2.2 if c != GRAY else 1.2, ls="-" if c != GRAY else "--")
        ax.text(1.05, y, lab, va="center", fontsize=7.2, color=c)
    ax.set_xlim(0, 4.2); ax.set_ylim(-21.5, 0); ax.set_xticks([])
    ax.set_ylabel("power (watts, $\\log_{10}$)")
    ax.set_yticks(range(-21, 1, 3)); ax.set_yticklabels([f"$10^{{{k}}}$" for k in range(-21, 1, 3)])
    sec = ax.secondary_yaxis("right", functions=(lambda y: 10 * y + 30, lambda d: (d - 30) / 10))
    sec.set_ylabel("dBm"); ax.spines["right"].set_visible(True)
    ax.grid(False)
    ax.set_title("A ladder of weakness: signals and the thermal floor (dashed)", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_floor_ladder")


def fig_moments():
    r = rng(4); t = np.linspace(0, 10, 2000)
    x = 0.8 + 0.35 * sps.lfilter([0.08], [1, -0.92], r.standard_normal(len(t))) / 0.2
    mu, sd = x.mean(), x.std()
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(t, x, color=NAVY, lw=0.6)
    ax.axhline(mu, color=ACCENT, lw=1.3); ax.text(10.1, mu, "mean\n(DC)", fontsize=6.8, color=ACCENT, va="center")
    ax.axhspan(mu - sd, mu + sd, color=GREEN, alpha=0.15)
    ax.text(10.1, mu + sd, "$+\\sigma$\n(RMS of\nAC part)", fontsize=6.3, color=GREEN, va="bottom")
    k = np.argmax(x); ax.plot(t[k], x[k], "v", color=ORANGE, ms=5)
    ax.annotate(f"peak: crest factor {(x[k] - mu) / sd:.1f}", (t[k], x[k]), (-40, 6), textcoords="offset points", fontsize=6.8, color=ORANGE)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_ylim(x.min() - 0.15, x.max() + 0.45)
    ax.set_xlim(0, 10); ax.set_xlabel("time (ms)"); ax.set_ylabel("voltage (V)")
    ax.set_title("Moments you can see on a scope", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_moments")


def fig_transforms():
    r = rng(8); n = 400_000
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    x = r.standard_normal(n)
    ax[0].hist(x ** 2, bins=np.linspace(0, 6, 120), density=True, color=NAVY, alpha=0.6)
    y = np.linspace(0.02, 6, 300); ax[0].plot(y, stats.chi2.pdf(y, 1), color=ACCENT)
    ax[0].set_ylim(0, 1.6); ax[0].set_title("square-law: $Y=X^2$", fontsize=9); ax[0].set_xlabel("$y$")
    th = r.uniform(0, 2 * np.pi, n)
    ax[1].hist(np.cos(th), bins=80, density=True, color=NAVY, alpha=0.6)
    yy = np.linspace(-0.995, 0.995, 400); ax[1].plot(yy, 1 / (np.pi * np.sqrt(1 - yy ** 2)), color=ACCENT)
    ax[1].set_ylim(0, 2.2); ax[1].set_title("sine at random instants", fontsize=9); ax[1].set_xlabel("$y=\\cos\\Theta$")
    p = r.exponential(1.0, n); pdb = 10 * np.log10(p)
    ax[2].hist(pdb, bins=np.linspace(-35, 10, 120), density=True, color=NAVY, alpha=0.6)
    ax[2].axvline(pdb.mean(), color=ACCENT, lw=1.2); ax[2].axvline(0, color=GREEN, lw=1.0, ls="--")
    ax[2].text(pdb.mean() - 1, 0.105, f"mean of dB\n= {pdb.mean():.1f} dB", fontsize=6.5, color=ACCENT, ha="right")
    ax[2].text(1, 0.1, "true mean\n= 0 dB", fontsize=6.5, color=GREEN)
    ax[2].set_ylim(0, 0.13)
    ax[2].set_title("noise power in decibels", fontsize=9); ax[2].set_xlabel("$10\\log_{10}P$ (dB)")
    fig.tight_layout(); save(fig, "ch03_transforms")


def fig_uncorrelated():
    r = rng(9); n = 3000
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.5))
    th = r.uniform(0, 2 * np.pi, n)
    ax[0].plot(np.cos(th), np.sin(th), ".", ms=1.5, color=NAVY)
    ax[0].set_title(f"$(\\cos\\Theta,\\sin\\Theta)$: corr = {abs(np.corrcoef(np.cos(th), np.sin(th))[0, 1]):.2f}\nuncorrelated, totally dependent", fontsize=8)
    g = r.standard_normal((2, n)) * 0.5
    ax[1].plot(g[0], g[1], ".", ms=1.5, color=GREEN)
    ax[1].set_title(f"Gaussian pair: corr = {abs(np.corrcoef(g)[0, 1]):.2f}\nuncorrelated, so independent", fontsize=8)
    for a in ax:
        a.set_aspect("equal"); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_xlabel("$I$")
    ax[0].set_ylabel("$Q$")
    fig.tight_layout(); save(fig, "ch03_uncorrelated")


def fig_bayes_alarm():
    r = rng(12)
    fig, ax = plt.subplots(figsize=(W1 * 0.9, 2.6))
    xx, yy = np.meshgrid(np.arange(200), np.arange(50))
    ax.plot(xx.ravel(), yy.ravel(), ",", color="#B8C2CC")
    fa = r.choice(10_000, 10, replace=False)
    ax.plot(fa % 200, fa // 200, "o", color=ACCENT, ms=5, label="false alarm (noise only): about 10")
    tp = 5123
    ax.plot([tp % 200], [tp // 200], "o", color=GREEN, ms=7, mec="black", label="the one real preamble")
    ax.set_xlim(-2, 201); ax.set_ylim(-2, 51); ax.axis("off")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    ax.set_title("10 000 tests of a detector with $P_{FA}=10^{-3}$, $P_D=0.99$: 10 alarms in 11 are false", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_bayes_alarm")


def fig_llr():
    A, s = 1.0, 0.6; r_ = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(r_, stats.norm.pdf(r_, -A, s), color=NAVY, label="$p(r\\mid -1)$")
    ax.plot(r_, stats.norm.pdf(r_, A, s), color=ACCENT, label="$p(r\\mid +1)$")
    ax.set_xlabel("received sample $r$"); ax.set_ylabel("likelihood"); ax.set_ylim(0, 0.75)
    a2 = ax.twinx(); a2.plot(r_, 2 * A * r_ / s ** 2, color=GREEN, lw=1.6, ls="--", label="LLR $=2Ar/\\sigma^2$")
    a2.axhline(0, color=GRAY, lw=0.6); a2.set_ylabel("LLR", color=GREEN); a2.set_ylim(-18, 18); a2.grid(False)
    a2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")
    ax.set_title("Soft information: sign = decision, size = confidence", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_llr")


def fig_galton_sim():
    r = rng(13); rows = 12; nb = 4000
    fig, ax = plt.subplots(figsize=(3.0, 3.6))
    for i in range(rows):
        for j in range(i + 1):
            ax.plot(j - i / 2, -i, "o", color=NAVY, ms=2.3)
    cols = [ACCENT, GREEN, ORANGE, PURPLE]
    for c in cols:
        steps = r.choice([-0.5, 0.5], rows); x = np.concatenate([[0], np.cumsum(steps)])
        ax.plot(x, -np.arange(rows + 1) + 0.3, color=c, lw=1.0, alpha=0.9)
    final = r.choice([-0.5, 0.5], (nb, rows)).sum(1)
    vals, cnt = np.unique(final, return_counts=True)
    scale = 7.0 / cnt.max()
    ax.bar(vals, cnt * scale, bottom=-rows - 8.5, width=0.85, color=NAVY, alpha=0.55)
    xs = np.linspace(-6.5, 6.5, 300)
    ax.plot(xs, nb * stats.norm.pdf(xs, 0, np.sqrt(rows) / 2) * scale - rows - 8.5, color=ACCENT, lw=1.5)
    ax.text(4.0, -rows - 2.2, "Gaussian\nprediction", fontsize=6.8, color=ACCENT)
    ax.text(-6.6, -1.5, f"{rows} rows of pegs:\neach bounce is\na coin flip", fontsize=6.8, color=NAVY)
    ax.text(-6.6, -rows - 3.2, f"{nb} balls", fontsize=6.8, color=NAVY)
    ax.set_xlim(-7, 7); ax.set_ylim(-rows - 9, 1); ax.axis("off")
    ax.set_title("A Galton board in software", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_galton_sim")


def fig_maxent():
    x = np.linspace(-4, 4, 800)
    dens = [("uniform", np.where(abs(x) < np.sqrt(3), 1 / (2 * np.sqrt(3)), 0), 0.5 * np.log2(12), GRAY),
            ("Laplacian", np.exp(-np.sqrt(2) * abs(x)) / np.sqrt(2), np.log2(2 * np.e / np.sqrt(2)), ORANGE),
            ("Gaussian", stats.norm.pdf(x), 0.5 * np.log2(2 * np.pi * np.e), ACCENT)]
    fig, ax = plt.subplots(figsize=NARROW)
    for name, d, h, c in dens:
        ax.plot(x, d, color=c, lw=1.5 if name == "Gaussian" else 1.1, label=f"{name}: $h$ = {h:.3f} bits")
    ax.set_xlabel("$x$ (unit variance)"); ax.set_ylim(0, 0.75); ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("Same power, different surprise", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_maxent")


def fig_q_tail():
    A, s = 1.0, 0.42; y = np.linspace(-2.6, 2.6, 600)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(y, stats.norm.pdf(y, -A, s), color=NAVY, label="sent $-A$")
    ax.plot(y, stats.norm.pdf(y, A, s), color=ACCENT, label="sent $+A$")
    m = y < 0
    ax.fill_between(y[m], stats.norm.pdf(y[m], A, s), color=ACCENT, alpha=0.35)
    ax.axvline(0, color=GRAY, ls="--", lw=0.9); ax.text(0.05, 0.98, "threshold", fontsize=6.8, color=GRAY)
    ax.annotate("", (0, 0.55), (A, 0.55), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(A / 2, 0.58, "$A=%.1f\\sigma$" % (A / s), fontsize=7.5, color=GREEN, ha="center")
    ax.annotate("error area $=Q(A/\\sigma)$", (-0.12, 0.05), (-2.55, 1.13),
                fontsize=6.6, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.set_xlabel("received sample"); ax.set_ylim(0, 1.25); ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("The tail that fools the receiver", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_q_tail")


def fig_cliff():
    eb = np.linspace(0, 14, 400); ber = Q(np.sqrt(2 * 10 ** (eb / 10)))
    fig, ax = plt.subplots(figsize=NARROW)
    for nbits, c, lab in [(100, GREEN, "100-bit message"), (12000, NAVY, "1500-byte packet"), (8e6, ACCENT, "1 MB file")]:
        ax.plot(eb, (1 - ber) ** nbits, color=c, label=lab)
    ax.set_xlabel("$E_b/N_0$ (dB), uncoded BPSK"); ax.set_ylabel("probability of no error")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_xlim(4, 14)
    ax.set_title("The digital cliff", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_cliff")


def fig_cn_cloud():
    r = rng(14); n = 4000
    z = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.plot(z.real, z.imag, ".", ms=1.2, color=NAVY, alpha=0.5, label="noise $n$")
    zr = z[:1500] * np.exp(1j * 1.0)
    ax.plot(zr.real, zr.imag, ".", ms=1.2, color=ACCENT, alpha=0.4, label="$e^{j57^\\circ}n$")
    for rad in [0.5, 1, 1.5, 2]:
        tt = np.linspace(0, 2 * np.pi, 200); ax.plot(rad * np.cos(tt), rad * np.sin(tt), color=GRAY, lw=0.5)
    ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$n_I$"); ax.set_ylabel("$n_Q$"); ax.legend(fontsize=6.3, loc="upper right", markerscale=5)
    ax.set_title("Circular: rotate it, nothing changes", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_cn_cloud")


def fig_fade_time():
    r = rng(15); fs = 5000; t = np.arange(0, 1.0, 1 / fs); fd = 40; M = 64
    h = np.zeros(len(t), complex)
    for k in range(M):
        a = r.uniform(0, 2 * np.pi); h += np.exp(1j * (2 * np.pi * fd * np.cos(a) * t + r.uniform(0, 2 * np.pi)))
    p = np.abs(h) ** 2 / np.mean(np.abs(h) ** 2); pdb = 10 * np.log10(p)
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.plot(t * 1e3, pdb, color=NAVY, lw=0.8)
    for lv, c in [(-10, ORANGE), (-20, ACCENT)]:
        ax.axhline(lv, color=c, lw=0.9, ls="--")
        ax.text(1002, lv, f"{lv} dB: below {100 * np.mean(pdb < lv):.1f}% of the time\n(theory {100 * (1 - np.exp(-10 ** (lv / 10))):.1f}%)", fontsize=6.5, color=c, va="center")
    ax.set_xlim(0, 1000); ax.set_ylim(-35, 8); ax.set_xlabel("time (ms)"); ax.set_ylabel("power (dB re mean)")
    ax.set_title("Rayleigh fading: the exponential power law in action (40 Hz Doppler)", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_fade_time")


def fig_ensemble():
    r = rng(16); t = np.linspace(0, 10, 600)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for k in range(5):
        x = sps.lfilter([0.15], [1, -0.85], r.standard_normal(len(t))) * 2.0
        ax[0].plot(t, x + 3 * k, color=CYCLE[k % 5], lw=0.7)
        ax[0].plot([4.0], [x[240] + 3 * k], "o", color="black", ms=3)
    ax[0].axvline(4.0, color=GRAY, ls="--", lw=0.8); ax[0].text(4.1, 13.6, "ensemble average:\ndown the column", fontsize=6.5)
    ax[0].annotate("", (9.8, -1.6), (0.2, -1.6), arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax[0].text(5, -2.8, "time average: along one path", fontsize=6.5, color=ACCENT, ha="center")
    ax[0].set_ylim(-3.5, 15); ax[0].set_yticks([]); ax[0].set_xlabel("time"); ax[0].set_title("ergodic: both agree", fontsize=9)
    for k in range(5):
        A = r.standard_normal()
        ax[1].plot(t, A + 0.08 * r.standard_normal(len(t)) + 3 * k, color=CYCLE[k % 5], lw=0.7)
    ax[1].set_ylim(-3.5, 15); ax[1].set_yticks([]); ax[1].set_xlabel("time")
    ax[1].set_title("not ergodic: random DC offset per unit", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_ensemble")


def fig_wk_pair():
    r = rng(17); N = 1 << 16
    fig, ax = plt.subplots(3, 3, figsize=(W2, 3.6))
    for i, (tc, c) in enumerate([(1.0, GRAY), (5.0, NAVY), (25.0, ACCENT)]):
        a = np.exp(-1 / tc)
        x = sps.lfilter([np.sqrt(1 - a ** 2)], [1, -a], r.standard_normal(N))
        ax[i, 0].plot(x[:300], color=c, lw=0.6); ax[i, 0].set_ylim(-4, 6); ax[i, 0].set_yticks([])
        lags = np.arange(0, 80); ax[i, 1].plot(lags, a ** lags, color=c); ax[i, 1].set_ylim(-0.05, 1.05)
        f, P = sps.welch(x, 1.0, nperseg=1024); ax[i, 2].semilogy(f, P, color=c); ax[i, 2].set_ylim(1e-2, 1e2)
        ax[i, 0].text(5, 4.3, f"$\\tau_c$ = {tc:g} samples", fontsize=7, color=c)
    ax[0, 0].set_title("sample path", fontsize=9); ax[0, 1].set_title("autocorrelation $R(\\tau)$", fontsize=9)
    ax[0, 2].set_title("PSD $S(f)$", fontsize=9)
    ax[2, 0].set_xlabel("time (samples)"); ax[2, 1].set_xlabel("lag $\\tau$"); ax[2, 2].set_xlabel("frequency ($f/f_s$)")
    fig.tight_layout(); save(fig, "ch03_wk_pair")


def fig_periodogram():
    r = rng(18); N = 8192; x = r.standard_normal(N)
    f1 = np.fft.rfftfreq(N); P1 = np.abs(np.fft.rfft(x)) ** 2 / N * 2
    f2, P2 = sps.welch(x, 1.0, nperseg=256)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(f1, 10 * np.log10(P1 + 1e-9), color=GRAY, lw=0.3, alpha=0.6, label="one periodogram")
    ax.plot(f2, 10 * np.log10(P2), color=NAVY, lw=2.0, label="Welch: 63 averaged")
    ax.axhline(10 * np.log10(2), color=ACCENT, ls="--", lw=1.0, label="true PSD")
    ax.set_ylim(-30, 15); ax.set_xlabel("frequency ($f/f_s$)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.3, loc="lower center"); ax.set_title("A single record never settles", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_periodogram")


def fig_ktc():
    C = np.logspace(-1, 2, 200) * 1e-12
    fig, ax = plt.subplots(figsize=NARROW)
    for T, c in [(300, ACCENT), (77, NAVY)]:
        ax.loglog(C * 1e12, np.sqrt(K_B * T / C) * 1e6, color=c, label=f"$\\sqrt{{kT/C}}$, {T} K")
    for bits, c in [(10, GRAY), (12, GREEN), (14, ORANGE), (16, PURPLE)]:
        q = 2.0 / 2 ** bits / np.sqrt(12) * 1e6
        ax.axhline(q, color=c, lw=0.8, ls=":"); ax.text(0.11, q * 1.08, f"{bits}-bit, 2 V: $\\Delta/\\sqrt{{12}}$", fontsize=6.2, color=c)
    ax.set_xlabel("sampling capacitance (pF)"); ax.set_ylabel("RMS noise ($\\mu$V)")
    ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("$kT/C$: noise frozen on a capacitor", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_ktc")


def fig_noise_colors():
    r = rng(19); N = 1 << 15
    w = r.standard_normal(N)
    W = np.fft.rfft(w); f = np.fft.rfftfreq(N); f[0] = f[1]
    pink = np.fft.irfft(W / np.sqrt(f), N); brown = np.fft.irfft(W / f, N)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for k, (x, c, lab) in enumerate([(w, GRAY, "white"), (pink, ACCENT, "pink ($1/f$, flicker)"), (brown, ORANGE, "brown ($1/f^2$, random walk)")]):
        x = x / x.std(); seg = x[:1500] - x[:1500].mean()
        ax[0].plot(seg / seg.std() * 0.8 - 4 * k, color=c, lw=0.5)
        ax[0].text(1520, -4 * k, lab.split(" ")[0], fontsize=7, color=c, va="center")
        ff, P = sps.welch(x, 1.0, nperseg=4096)
        ax[1].loglog(ff[1:], P[1:], color=c, label=lab)
    ax[0].set_yticks([]); ax[0].set_xlim(0, 1800); ax[0].set_xlabel("time (samples)"); ax[0].set_title("three colours of noise", fontsize=9)
    ax[1].set_xlabel("frequency ($f/f_s$)"); ax[1].set_ylabel("PSD"); ax[1].legend(fontsize=6.5, loc="lower left")
    ax[1].set_title("their spectra (log-log)", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_noise_colors")


def fig_shot():
    r = rng(20)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    for k, (lam, c) in enumerate([(5, ACCENT), (50, NAVY), (500, GREEN)]):
        tt = np.sort(r.uniform(0, 1, r.poisson(lam)))
        t = np.linspace(0, 1, 2000); i = np.zeros_like(t)
        for t0 in tt:
            i += np.exp(-((t - t0) / 0.01) ** 2)
        i /= i.max()
        ax[0].plot(t, i + 1.25 * k, color=c, lw=0.8)
        ax[0].text(1.02, 1.25 * k + 0.5, f"{lam}/s", fontsize=6.8, color=c, va="center")
    ax[0].set_yticks([]); ax[0].set_xlabel("time (s)"); ax[0].set_title("current as a hail of charges\n(each trace scaled to its peak)", fontsize=8.5)
    N = 20; k = np.arange(0, 45)
    ax[1].bar(k, stats.poisson.pmf(k, N), color=NAVY, alpha=0.7)
    ax[1].bar([0], [0.02], color=ACCENT)
    ax[1].annotate(f"$P(0)=e^{{-20}}\\approx{np.exp(-20):.0e}$:\na 'one' seen as 'zero'", (0, 0.02), (25, 0.07), fontsize=6.6, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax[1].set_xlabel("photoelectrons counted in a bit"); ax[1].set_title("Poisson counts, mean 20", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_shot")


def fig_impulsive():
    r = rng(22); n = 200_000; A = 0.1; G = 0.01
    m = r.poisson(A, n)
    var = (m / A + G) / (1 + G)
    z = np.sqrt(var / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))
    g = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax[0].plot(np.abs(g[:3000]), color=GRAY, lw=0.5, label="Gaussian")
    ax[0].plot(np.abs(z[:3000]) + 8, color=ACCENT, lw=0.5, label="class A ($A=0.1$)")
    ax[0].set_yticks([]); ax[0].set_xlabel("sample"); ax[0].set_title("same average power, very different life", fontsize=9)
    ax[0].legend(fontsize=6.5, loc="upper right")
    lv = np.linspace(-20, 25, 200)
    for x, c, lab in [(g, GRAY, "Gaussian (Rayleigh envelope)"), (z, ACCENT, "Middleton class A")]:
        e = 20 * np.log10(np.abs(x))
        ax[1].semilogy(lv, [np.mean(e > l) for l in lv], color=c, label=lab)
    ax[1].set_ylim(1e-5, 1.1); ax[1].set_xlabel("envelope level (dB re RMS)"); ax[1].set_ylabel("P(envelope > level)")
    ax[1].set_title("amplitude probability distribution", fontsize=9); ax[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch03_impulsive")


def fig_resistor_density():
    R = np.logspace(0, 6, 200)
    fig, ax = plt.subplots(figsize=NARROW)
    for T, c in [(290, ACCENT), (77, NAVY), (4, PURPLE)]:
        ax.loglog(R, np.sqrt(4 * K_B * T * R) * 1e9, color=c, label=f"{T} K")
    for Rm, lab in [(50, "50 $\\Omega$: 0.9 nV"), (1e3, "1 k$\\Omega$: 4 nV")]:
        v = np.sqrt(4 * K_B * 290 * Rm) * 1e9; ax.plot(Rm, v, "o", color=ACCENT, ms=4)
        ax.annotate(lab, (Rm, v), (5, -12), textcoords="offset points", fontsize=6.5)
    ax.axhline(1, color=GREEN, ls=":", lw=1.0); ax.text(1.3, 0.55, "good audio op-amp, 1 nV/$\\sqrt{\\mathrm{Hz}}$", fontsize=6.3, color=GREEN)
    ax.set_xlabel("resistance ($\\Omega$)"); ax.set_ylabel("noise density (nV/$\\sqrt{\\mathrm{Hz}}$)")
    ax.legend(fontsize=6.5, loc="lower right", title="temperature", title_fontsize=6.5)
    ax.set_title("Every resistor whispers $\\sqrt{4kTR}$", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_resistor_density")


def fig_whisper_chain():
    A = [("cable", -2, 2), ("filter", -1.5, 1.5), ("LNA", 20, 1.0), ("mixer", -7, 9), ("IF amp", 25, 5)]
    B = [("LNA", 20, 1.0), ("cable", -2, 2), ("filter", -1.5, 1.5), ("mixer", -7, 9), ("IF amp", 25, 5)]
    fig, ax = plt.subplots(figsize=NARROW)
    for S, c, lab in [(A, ACCENT, "LNA third"), (B, GREEN, "LNA first")]:
        F, G = 1.0, 1.0; nf = [0.0]
        for _, g, n in S:
            F += (10 ** (n / 10) - 1) / G; G *= 10 ** (g / 10); nf.append(10 * np.log10(F))
        ax.step(range(len(nf)), nf, where="post", color=c, lw=1.8, label=f"{lab}: {nf[-1]:.1f} dB")
        for i, (nm, _, _) in enumerate(S):
            ax.text(i + 0.5, nf[i + 1] + 0.18, nm, fontsize=5.8, color=c, ha="center")
    ax.set_xlabel("stages passed"); ax.set_ylabel("SNR lost so far (dB)"); ax.set_ylim(0, 6.2)
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_title("The whisper chain: SNR lost stage by stage", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_whisper_chain")


def fig_nf_te():
    nf = np.linspace(0, 10, 300); Te = (10 ** (nf / 10) - 1) * 290
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogy(nf, Te, color=NAVY, lw=1.6)
    for n, lab in [(0.5, "(satellite LNB)"), (1, "(GPS LNA)"), (3, "(as noisy as the room)"), (6, "(phone receiver)"), (10, "")]:
        T = (10 ** (n / 10) - 1) * 290; ax.plot(n, T, "o", color=ACCENT, ms=4)
        ax.annotate(f"{n:g} dB = {T:.0f} K {lab}", (n, T), (6, -3) if n < 10 else (-62, -3), textcoords="offset points", fontsize=6.0)
    ax.set_xlabel("noise figure (dB)"); ax.set_ylabel("noise temperature $T_e$ (K)"); ax.set_ylim(5, 5000)
    ax.set_title("Noise figure and noise temperature", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_nf_te")


def fig_sens_stack():
    steps = [("$kT_0$\n1 Hz", -174), ("$+10\\log_{10}B$\n20 MHz", 73), ("+ NF\n10 dB", 10), ("+ SNR\nreq. 4 dB", 4), ("+ margin\n5 dB", 5)]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    lvl = 0
    for i, (lab, v) in enumerate(steps):
        if i == 0:
            ax.bar(i, -174 + 180, bottom=-180, color=GRAY, width=0.6); lvl = -174
        else:
            ax.bar(i, v, bottom=lvl, color=[NAVY, ORANGE, GREEN, PURPLE][i - 1], width=0.6); lvl += v
        ax.text(i, lvl + 2, f"{lvl:.0f}", ha="center", fontsize=6.8)
    ax.axhline(-82, color=ACCENT, ls="--", lw=1.0); ax.text(-0.4, -79, "802.11a 6 Mb/s spec: $-82$ dBm", fontsize=6.6, color=ACCENT)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=6.2)
    ax.set_ylim(-180, -70); ax.set_ylabel("dBm"); ax.set_title("Building a sensitivity", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_sens_stack")


def fig_smoke_roc():
    y = np.linspace(-3, 7, 500); d = 2.2
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax[0].fill_between(y, stats.norm.pdf(y), color=NAVY, alpha=0.25)
    ax[0].fill_between(y, stats.norm.pdf(y, d), color=ACCENT, alpha=0.25)
    ax[0].text(-2.9, 0.3, "toast,\nsteam,\ndust", fontsize=6.5, color=NAVY)
    ax[0].text(4.1, 0.3, "real\nfire", fontsize=6.5, color=ACCENT)
    th = [(0.3, GREEN, "hair-trigger"), (1.6, ORANGE, "balanced"), (3.2, PURPLE, "dull")]
    for (t, c, lab), yy, ha in zip(th, [0.5, 0.455, 0.5], ["right", "center", "left"]):
        ax[0].plot([t, t], [0, yy - 0.01], color=c, lw=1.2, ls="--")
        ax[0].text(t, yy, lab, fontsize=6.3, color=c, ha=ha, va="bottom")
    ax[0].set_xlabel("smoke-sensor reading"); ax[0].set_ylim(0, 0.56); ax[0].set_xlim(-3, 6)
    ax[0].set_title("same sensor, three thresholds", fontsize=9)
    pfa = np.logspace(-4, 0, 300); ax[1].semilogx(pfa, Q(Qinv(pfa) - d), color=NAVY)
    offs = [(-16, -16, "right"), (10, -22, "left"), (8, -4, "left")]
    for (t, c, lab), (ox, oy, ha) in zip(th, offs):
        ax[1].plot(Q(t), Q(t - d), "o", color=c, ms=6)
        ax[1].annotate(lab + f"\nmiss {100 * (1 - Q(t - d)):.0f}%, false {100 * Q(t):.1f}%", (Q(t), Q(t - d)), (ox, oy), textcoords="offset points", fontsize=5.8, color=c, ha=ha, va="top")
    ax[1].set_xlabel("false-alarm probability"); ax[1].set_ylabel("detection probability")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("the ROC: one curve, every threshold", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_smoke_roc")


def fig_cfar():
    r = rng(23); n = 400
    lvl = np.where(np.arange(n) < 200, 1.0, 6.0)
    x = lvl * r.exponential(1.0, n)
    tg = [80, 300]; x[tg[0]] += 18; x[tg[1]] += 60
    fixed = 1.0 * np.log(1 / 1e-4)
    win, guard = 16, 2
    cf = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(i - win - guard, i + win + guard + 1) if 0 <= j < n and abs(j - i) > guard]
        cf[i] = np.mean(x[idx]) * (len(idx) * ((1e-4) ** (-1 / len(idx)) - 1))
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    ax.semilogy(x, color=GRAY, lw=0.7, label="received power per range cell")
    ax.semilogy(np.full(n, fixed), color=ORANGE, ls="--", lw=1.2, label="fixed threshold")
    ax.semilogy(cf, color=GREEN, lw=1.3, label="CFAR threshold")
    fa = np.where((x > fixed) & ~np.isin(np.arange(n), tg))[0]
    ax.plot(fa, x[fa], "x", color=ORANGE, ms=4, label=f"fixed-threshold false alarms ({len(fa)})")
    ax.plot(tg, x[tg], "o", color=ACCENT, ms=5, label="targets")
    ax.text(100, 2e2, "quiet sea", fontsize=7, color=NAVY, ha="center"); ax.text(300, 2e2, "rain clutter: noise level 6x higher", fontsize=7, color=NAVY, ha="center")
    ax.set_ylim(1e-5, 1e3); ax.set_xlabel("range cell"); ax.legend(fontsize=6, loc="lower left", ncol=3)
    ax.set_title("Constant false-alarm rate: let the threshold follow the noise", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_cfar")


def fig_crb_curvature():
    A = np.linspace(-1, 3, 400); r = rng(24)
    fig, ax = plt.subplots(figsize=NARROW)
    for N, c in [(4, ORANGE), (40, NAVY)]:
        x = 1.0 + r.standard_normal(N)
        ll = np.array([-np.sum((x - a) ** 2) / 2 for a in A])
        ax.plot(A, ll - ll.max(), color=c, label=f"$N={N}$: Fisher info $={N}$")
        ax.plot(A[np.argmax(ll)], 0, "v", color=c, ms=5)
    ax.axvline(1.0, color=GRAY, ls=":", lw=0.9); ax.text(1.03, -11.5, "true level", fontsize=6.5, color=GRAY)
    ax.set_ylim(-12, 0.8); ax.set_xlabel("candidate level $A$"); ax.set_ylabel("log-likelihood (rel. to peak)")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("Sharp peak = precise estimate", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_crb_curvature")


def fig_wiener():
    f = np.linspace(0, 5, 400); Ss = 10 / (1 + f ** 4); Sn = np.full_like(f, 1.0)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogy(f, Ss, color=NAVY, label="signal PSD $S_s$")
    ax.semilogy(f, Sn, color=GRAY, ls="--", label="noise PSD $S_n$")
    a2 = ax.twinx(); a2.plot(f, Ss / (Ss + Sn), color=ACCENT, lw=1.6, label="Wiener $H=S_s/(S_s+S_n)$")
    a2.set_ylim(0, 1.05); a2.set_ylabel("$H(f)$", color=ACCENT); a2.grid(False); a2.spines["right"].set_visible(True)
    ax.set_xlabel("frequency"); ax.set_ylim(1e-2, 30)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="lower left"); ax.set_title("Pass where signal wins", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_wiener")


def fig_fspl():
    f = np.logspace(0, 2, 100) * 1e9; lam = 3e8 / f; d = 1e3
    fixedG = 20 * np.log10(lam / (4 * np.pi * d))
    Ae = 0.6 * np.pi * 0.15 ** 2; G = 4 * np.pi * Ae / lam ** 2
    fixedA = fixedG + 2 * 10 * np.log10(G)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(f / 1e9, fixedG, color=ACCENT, label="0 dBi antennas (fixed gain)")
    ax.semilogx(f / 1e9, fixedA, color=GREEN, label="30 cm dishes (fixed area)")
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("$P_r/P_t$ over 1 km (dB)")
    ax.legend(fontsize=6.5, loc="center left"); ax.set_title("Is high frequency lossier? It depends", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_fspl")


def fig_timeline():
    ev = [(1918, "Schottky:\nshot noise"), (1928, "Johnson &\nNyquist: $4kTR$"), (1930, "Wiener:\nspectra of\nnoise"),
          (1933, "Jansky: noise\nfrom the galaxy;\nNeyman-Pearson"), (1942, "North: noise\nfactor"), (1944, "Friis cascade;\nRice: noise\nstatistics"),
          (1948, "Marcum: radar\ndetection;\nShannon"), (1965, "Penzias &\nWilson: CMB"), (1966, "Leeson:\nphase noise"), (1982, "Caves: quantum\nlimit of amplifiers")]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.plot([1914, 1988], [0, 0], color=NAVY, lw=1.5)
    tiers = [0.3, -0.3, 1.25, -1.25, 0.3, -0.3, 1.25, -0.3, 0.3, -0.3]
    for (y, txt), h in zip(ev, tiers):
        up = h > 0
        ax.plot([y, y], [0, h], color=GRAY, lw=0.7); ax.plot(y, 0, "o", color=ACCENT, ms=4)
        ax.text(y, h + (0.05 if up else -0.05), f"{y}\n{txt}" if up else f"{txt}\n{y}", ha="center", va="bottom" if up else "top", fontsize=5.9)
    ax.set_xlim(1912, 1990); ax.set_ylim(-2.25, 2.3); ax.axis("off")
    fig.tight_layout(); save(fig, "ch03_timeline")


def fig_cyclo_concept():
    r = rng(25); sps_ = 16; taps = cl.rrc_taps(0.5, sps_, 8)
    sym = r.choice([-1.0, 1.0], (3000,))
    x = cl.shape(sym, taps, sps_)
    x = x[len(taps):len(taps) + 2800 * sps_]
    v = np.var(x.reshape(-1, sps_), axis=0); v = np.tile(v / v.mean(), 2)
    fig, ax = plt.subplots(figsize=NARROW)
    ph = np.arange(2 * sps_) / sps_
    ax.plot(ph, v, color=NAVY, marker="o", ms=2.5, label="BPSK (RRC, $\\beta=0.5$)")
    ax.axhline(1.0, color=GRAY, ls="--", label="noise: flat")
    ax.set_xlabel("time within the symbol (symbols)"); ax.set_ylabel("variance (normalised)")
    ax.set_ylim(0, 1.6); ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("A signal's hidden heartbeat", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_cyclo_concept")


def fig_tsys_budget():
    parts = [("CMB", 2.7, PURPLE), ("atmosphere", 2.5, NAVY), ("spillover\n& ground", 5.0, GREEN),
             ("feed & waveguide", 3.0, ORANGE), ("cryogenic LNA", 5.0, ACCENT), ("follow-on", 1.0, GRAY)]
    fig, ax = plt.subplots(figsize=(W2 * 0.85, 1.25))
    left = 0
    for name, T, c in parts:
        ax.barh(0, T, left=left, color=c, height=0.5)
        ax.text(left + T / 2, 0.38, f"{name}\n{T:g} K", ha="center", va="bottom", fontsize=6.0, color=c)
        left += T
    ax.text(left + 0.3, 0, f"$T_{{sys}}\\approx{left:.0f}$ K", va="center", fontsize=8, color=NAVY)
    ax.set_xlim(0, left + 4); ax.set_ylim(-0.4, 1.3); ax.axis("off")
    fig.tight_layout(); save(fig, "ch03_tsys_budget")


def fig_adc_floor():
    G = np.linspace(0, 70, 300); Frx = 10 ** (3 / 10)
    fig, ax = plt.subplots(figsize=NARROW)
    for nfadc, c in [(27, NAVY), (11, GREEN)]:
        F = Frx + (10 ** (nfadc / 10) - 1) / 10 ** (G / 10)
        ax.plot(G, 10 * np.log10(F), color=c, label=f"ADC NF = {nfadc} dB")
    ax.axhline(3, color=GRAY, ls=":"); ax.text(25, 1.2, "analog chain alone: 3 dB", fontsize=6.5, color=GRAY)
    ax.axvspan(54, 70, color=ACCENT, alpha=0.08); ax.text(55, 20, "clipping\nrisk", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("analog gain ahead of the ADC (dB)"); ax.set_ylabel("total noise figure (dB)"); ax.set_ylim(0, 30)
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("Lift the noise above the ADC floor", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_adc_floor")


def fig_rice_phasor():
    r = rng(26); n = 1500
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.4))
    for a, (A, title) in zip(ax, [(3.0, "strong carrier: Rice, nearly Gaussian"), (0.6, "weak carrier: noise takes over")]):
        z = A + (r.standard_normal(n) + 1j * r.standard_normal(n)) * 0.5
        a.plot(z.real, z.imag, ".", ms=1.3, color=NAVY, alpha=0.5)
        a.annotate("", (A, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
        a.text(A / 2, 0.15, "$A$", color=ACCENT, fontsize=8, ha="center")
        tt = np.linspace(0, 2 * np.pi, 200)
        a.plot(A * np.cos(tt), A * np.sin(tt), color=GRAY, lw=0.6, ls=":")
        a.set_aspect("equal"); a.set_xlim(-2.2, 4.8); a.set_ylim(-2.4, 2.4); a.axhline(0, color=GRAY, lw=0.5); a.axvline(0, color=GRAY, lw=0.5)
        a.set_title(title, fontsize=8.5); a.set_xlabel("I")
    ax[0].set_ylabel("Q")
    fig.tight_layout(); save(fig, "ch03_rice_phasor")


def fig_leeson():
    df = np.logspace(1, 7, 400); f0 = 10e6 / (2 * 20); fc = 3e3
    L = -165 + 10 * np.log10(1 + (f0 / df) ** 2) + 10 * np.log10(1 + fc / df)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(df, L, color=NAVY, lw=1.6)
    ax.text(30, -60, "$1/\\Delta f^3$\n(flicker)", fontsize=6.8, color=ACCENT)
    ax.text(1.5e4, -118, "$1/\\Delta f^2$", fontsize=6.8, color=ACCENT)
    ax.text(1.2e6, -158, "white floor", fontsize=6.8, color=ACCENT)
    ax.set_xlabel("offset from carrier $\\Delta f$ (Hz)"); ax.set_ylabel("$\\mathcal{L}(\\Delta f)$ (dBc/Hz)")
    ax.set_title("Leeson's phase-noise shape (illustrative)", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_leeson")


def fig_fec_gain():
    ber = np.logspace(-13, -1, 300); snr = 20 * np.log10(Qinv(ber))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(ber, snr, color=NAVY, lw=1.6)
    for b, c, lab in [(1e-12, ACCENT, "no FEC: $10^{-12}$"), (1e-6, ORANGE, "$10^{-6}$"), (2e-4, GREEN, "KP4 FEC input: $2\\times10^{-4}$")]:
        s = 20 * np.log10(Qinv(b)); ax.plot(b, s, "o", color=c, ms=5)
        ax.annotate(f"{lab}\n{s:.1f} dB", (b, s), (8, -6) if b > 1e-5 else (8, 2), textcoords="offset points", fontsize=6.5, color=c)
    ax.set_ylim(0, 20); ax.invert_xaxis(); ax.set_xlabel("target bit-error rate"); ax.set_ylabel("required SNR $20\\log_{10}(A/\\sigma)$ (dB)")
    ax.set_title("Each decade of BER costs less and less", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_fec_gain")


def fig_fmin_circles():
    Fmin, Rn, Z0 = 10 ** (0.5 / 10), 10.0, 50.0
    Gopt = 0.45 * np.exp(1j * np.radians(60))
    g = np.linspace(-1, 1, 500); X, Y = np.meshgrid(g, g); G = X + 1j * Y
    F = Fmin + 4 * Rn / Z0 * np.abs(G - Gopt) ** 2 / ((1 - np.abs(G) ** 2) * np.abs(1 + Gopt) ** 2)
    NF = np.where(np.abs(G) < 0.999, 10 * np.log10(np.maximum(F, 1)), np.nan)
    fig, ax = plt.subplots(figsize=(2.9, 2.8))
    tt = np.linspace(0, 2 * np.pi, 300); ax.plot(np.cos(tt), np.sin(tt), color=GRAY, lw=1.0)
    for rr in [1 / 3, 1, 3]:
        c = rr / (1 + rr); rad = 1 / (1 + rr); ax.plot(c + rad * np.cos(tt), rad * np.sin(tt), color=GRAY, lw=0.4)
    ax.axhline(0, color=GRAY, lw=0.4)
    cs = ax.contour(X, Y, NF, levels=[0.75, 1.0, 1.5, 2.0, 3.0], colors=[NAVY, GREEN, ORANGE, PURPLE, ACCENT], linewidths=1.1)
    ax.clabel(cs, fmt="%.2g dB", fontsize=6)
    ax.plot(Gopt.real, Gopt.imag, "*", color=ACCENT, ms=8); ax.text(Gopt.real - 0.62, Gopt.imag + 0.12, "$\\Gamma_{opt}$: 0.5 dB", fontsize=6.5, color=ACCENT)
    ax.plot(0, 0, "o", color=NAVY, ms=4); ax.text(0.04, -0.12, "50 $\\Omega$", fontsize=6.5, color=NAVY)
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.axis("off")
    ax.set_title("Noise circles on the Smith chart (illustrative)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_fmin_circles")


def fig_sps_noise():
    r = rng(27); sps_ = 8; taps = cl.rrc_taps(0.35, sps_, 8)
    sym = r.choice([-1.0, 1.0], 60)
    taps = np.real(np.asarray(taps))
    x = np.real(np.asarray(cl.shape(sym, taps, sps_)))
    x = x / np.sqrt(np.mean(x ** 2))
    EsN0 = 10 ** (8 / 10); sig2 = sps_ / EsN0
    n = np.sqrt(sig2) * r.standard_normal(len(x))
    y = np.convolve(x + n, taps)
    best = max(range(0, 3 * len(taps)), key=lambda dd: abs(np.dot(y[dd:dd + 40 * sps_:sps_][:40], sym[:40])))
    ys = y[best::sps_][:len(sym)]
    ys = ys / np.mean(np.abs(ys))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1))
    k = np.arange(20 * sps_, 40 * sps_)
    ax[0].plot(k / sps_, (x + n)[k], color=GRAY, lw=0.6, label=r"samples: SNR per sample $\approx$ %.0f dB" % (10 * np.log10(1 / sig2)))
    ax[0].plot(k / sps_, x[k], color=NAVY, lw=1.2, label="clean signal")
    ax[0].set_xlabel("time (symbols)"); ax[0].legend(fontsize=6.3, loc="lower right"); ax[0].set_title("8 samples per symbol, before the matched filter", fontsize=8.5)
    ax[0].set_ylim(-5.5, 4.5)
    ax[1].plot(np.arange(len(ys)), ys, "o", color=NAVY, ms=3.5)
    ax[1].axhline(0, color=GRAY, lw=0.6); ax[1].set_ylim(-2.2, 2.2)
    ax[1].set_xlabel("symbol index"); ax[1].set_title("after the matched filter: $E_s/N_0=8$ dB", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch03_sps_noise")


def fig_sat_budget():
    items = [("EIRP", 52.0), ("path\nloss", -205.6), ("air", -0.5), ("$G/T$", 14.9), ("$-k$", 228.6)]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    lvl = 0
    for i, (lab, v) in enumerate(items):
        lo, hi = sorted([lvl, lvl + v])
        ax.bar(i, hi - lo, bottom=lo, color=GREEN if v > 0 else ACCENT, width=0.6)
        lvl += v
        ax.text(i, hi + 6, f"{v:+.1f}", ha="center", fontsize=6.5)
    ax.bar(len(items), lvl, bottom=0, color=NAVY, width=0.6); ax.text(len(items), lvl + 6, f"{lvl:.1f}", ha="center", fontsize=6.5, color=NAVY)
    ax.set_xticks(range(len(items) + 1)); ax.set_xticklabels([l for l, _ in items] + ["$C/N_0$\n(dB-Hz)"], fontsize=6.3)
    ax.axhline(0, color=GRAY, lw=0.6); ax.set_ylabel("dB"); ax.set_ylim(-175, 120)
    ax.set_title("A Ku-band TV downlink in five terms", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_sat_budget")


def fig_craig():
    th = np.linspace(1e-3, np.pi / 2, 400)
    fig, ax = plt.subplots(figsize=NARROW)
    for x, c in [(1, NAVY), (2, GREEN), (3, ACCENT)]:
        f = np.exp(-x ** 2 / (2 * np.sin(th) ** 2)) / np.pi
        ax.plot(th, f, color=c, label=f"$x={x}$: area $=Q({x})={float(Q(x)):.2g}$")
        ax.fill_between(th, f, color=c, alpha=0.15)
        ax.plot(np.pi / 2, f[-1], "o", color=c, ms=3.5)
    ax.text(0.08, 0.12, "peak value $\\frac{1}{\\pi}e^{-x^2/2}$ at $\\theta=\\pi/2$\n$\\Rightarrow Q(x)\\leq\\frac{1}{2}e^{-x^2/2}$", fontsize=6.5, color=GRAY, ha="left")
    ax.set_xlim(0, np.pi / 2 + 0.05); ax.set_ylim(0, 0.36)
    ax.set_xticks([0, np.pi / 4, np.pi / 2]); ax.set_xticklabels(["0", "$\\pi/4$", "$\\pi/2$"])
    ax.set_xlabel("$\\theta$"); ax.set_ylabel("integrand of Craig's formula")
    ax.legend(fontsize=6.3, loc="upper left"); ax.set_title("Q as a finite integral", fontsize=9)
    fig.tight_layout(); save(fig, "ch03_craig")


ALL = [fig_craig, fig_fec_gain, fig_fmin_circles, fig_sps_noise, fig_sat_budget, fig_rice_phasor, fig_leeson, fig_clt, fig_clt_tails, fig_chisq, fig_q, fig_qapprox, fig_envelopes, fig_filtered_noise, fig_neb,
       fig_bandpass, fig_cyclo, fig_planck, fig_skynoise, fig_friis, fig_lossyline, fig_yfactor,
       fig_sensitivity, fig_roc, fig_crb, fig_linkbudget,
       fig_party, fig_floor_ladder, fig_moments, fig_transforms, fig_uncorrelated, fig_bayes_alarm, fig_llr,
       fig_galton_sim, fig_maxent, fig_q_tail, fig_cliff, fig_cn_cloud, fig_fade_time, fig_ensemble, fig_wk_pair,
       fig_periodogram, fig_ktc, fig_noise_colors, fig_shot, fig_impulsive, fig_resistor_density,
       fig_whisper_chain, fig_nf_te, fig_sens_stack, fig_smoke_roc, fig_cfar, fig_crb_curvature, fig_wiener,
       fig_fspl, fig_timeline, fig_cyclo_concept, fig_tsys_budget, fig_adc_floor]

if __name__ == "__main__":
    sel = sys.argv[1:]
    for fn in ALL:
        if not sel or fn.__name__[4:] in sel:
            fn()
