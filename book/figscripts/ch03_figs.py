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


ALL = [fig_clt, fig_clt_tails, fig_chisq, fig_q, fig_qapprox, fig_envelopes, fig_filtered_noise, fig_neb,
       fig_bandpass, fig_cyclo, fig_planck, fig_skynoise, fig_friis, fig_lossyline, fig_yfactor,
       fig_sensitivity, fig_roc, fig_crb, fig_linkbudget]

if __name__ == "__main__":
    sel = sys.argv[1:]
    for fn in ALL:
        if not sel or fn.__name__[4:] in sel:
            fn()
