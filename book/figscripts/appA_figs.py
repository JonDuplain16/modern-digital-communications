"""Figures for Appendix A (Mathematical Reference)."""
from figstyle import *
from scipy import special, stats
from commlib import qfunc


def qfunc_bounds():
    x = np.linspace(0.05, 6, 600)
    q = qfunc(x)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    chern = 0.5 * np.exp(-x ** 2 / 2)
    up = phi / x
    lo = phi / x * (1 - 1 / x ** 2)
    chiani = np.exp(-x ** 2 / 2) / 12 + np.exp(-2 * x ** 2 / 3) / 4
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    a = ax[0]
    a.semilogy(x, q, color="k", lw=2.0, label=r"$\mathrm{Q}(x)$ exact")
    a.semilogy(x, chern, "--", color=ACCENT, label=r"Chernoff $\frac{1}{2}e^{-x^2/2}$")
    a.semilogy(x, up, color=NAVY, label=r"upper $\phi(x)/x$")
    a.semilogy(x[x > 1], lo[x > 1], color=GREEN, label=r"lower $\phi(x)(1-x^{-2})/x$")
    a.semilogy(x, chiani, ":", color=ORANGE, lw=1.8, label="Chiani et al. (2003)")
    a.set_ylim(1e-9, 1); a.set_xlim(0, 6)
    a.set_xlabel("$x$"); a.set_ylabel("value"); a.legend(loc="lower left", fontsize=6.8)
    a.set_title("The Gaussian tail and its bounds")
    b = ax[1]
    xx = x[x > 0.5]
    qq = qfunc(xx)
    for y, c, lab, ls in [(0.5 * np.exp(-xx ** 2 / 2), ACCENT, "Chernoff", "--"),
                          (np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi) / xx, NAVY, r"$\phi(x)/x$", "-"),
                          (np.exp(-xx ** 2 / 2) / 12 + np.exp(-2 * xx ** 2 / 3) / 4, ORANGE, "Chiani", ":")]:
        b.plot(xx, 10 * np.log10(y / qq), ls, color=c, label=lab)
    m = xx > 1.2
    b.plot(xx[m], 10 * np.log10(np.exp(-xx[m] ** 2 / 2) / np.sqrt(2 * np.pi) / xx[m] * (1 - xx[m] ** -2) / qq[m]),
           color=GREEN, label=r"$\phi(x)(1-x^{-2})/x$")
    b.axhline(0, color="k", lw=0.8)
    b.set_ylim(-3, 6); b.set_xlim(0.5, 6)
    b.set_xlabel("$x$"); b.set_ylabel("error relative to Q$(x)$ (dB)")
    b.set_title("How tight is each approximation?")
    b.legend(fontsize=6.8, loc="upper right")
    save(fig, "appA_qfunc")


def fading_pdfs():
    r = np.linspace(0, 3, 600)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    for K, c in [(0, NAVY), (1, GREEN), (3, ORANGE), (10, ACCENT)]:
        # unit mean power Omega = 1: nu^2 = K/(K+1), 2 sigma^2 = 1/(K+1)
        nu = np.sqrt(K / (K + 1)); s2 = 1 / (2 * (K + 1))
        pdf = r / s2 * np.exp(-(r - nu) ** 2 / (2 * s2)) * special.i0e(r * nu / s2)
        a.plot(r, pdf, color=c, label=("Rayleigh ($K=0$)" if K == 0 else f"Rice $K={K}$"))
    a.set_xlabel("envelope $r$ (unit mean power)"); a.set_ylabel("pdf $f_R(r)$")
    a.set_title("Rician envelopes"); a.legend(fontsize=7)
    b = ax[1]
    for mm, c in [(0.5, PURPLE), (1, NAVY), (2, GREEN), (4, ORANGE), (10, ACCENT)]:
        pdf = 2 * mm ** mm / special.gamma(mm) * r ** (2 * mm - 1) * np.exp(-mm * r ** 2)
        lab = f"$m={mm}$" + (" (one-sided Gaussian)" if mm == 0.5 else " (Rayleigh)" if mm == 1 else "")
        b.plot(r, pdf, color=c, label=lab)
    b.set_ylim(0, 4.0)
    b.set_xlabel("envelope $r$ (unit mean power)"); b.set_ylabel("pdf $f_R(r)$")
    b.set_title(r"Nakagami-$m$ envelopes ($\Omega=1$)"); b.legend(fontsize=7)
    save(fig, "appA_fading_pdfs")


def chi2_family():
    x = np.linspace(0.001, 16, 800)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    for k, c in [(1, PURPLE), (2, NAVY), (4, GREEN), (8, ORANGE), (16, ACCENT)]:
        a.plot(x, stats.chi2.pdf(x, k), color=c, label=f"$k={k}$")
    a.set_ylim(0, 0.55); a.set_xlim(0, 16)
    a.set_xlabel("$x$"); a.set_ylabel("pdf"); a.set_title(r"Central $\chi^2_k$ (unit-variance terms)")
    a.legend(fontsize=7)
    b = ax[1]
    for lam, c in [(0, NAVY), (2, GREEN), (5, ORANGE), (10, ACCENT)]:
        y = stats.chi2.pdf(x, 2) if lam == 0 else stats.ncx2.pdf(x, 2, lam)
        b.plot(x, y, color=c, label=fr"$\lambda={lam}$")
    b.set_xlim(0, 16); b.set_ylim(0, 0.52)
    b.set_xlabel("$x$"); b.set_ylabel("pdf")
    b.set_title(r"Noncentral $\chi'^2_2(\lambda)$: energy detector output")
    b.legend(fontsize=7)
    save(fig, "appA_chi2")


def bessel_marcum():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    x = np.linspace(0, 12, 600)
    a = ax[0]
    for n, c in zip(range(4), [NAVY, ACCENT, GREEN, ORANGE]):
        a.plot(x, special.jv(n, x), color=c, label=f"$J_{n}(x)$")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("$x$"); a.set_title("Bessel $J_n$ (FM sidebands)"); a.legend(fontsize=6.5, ncol=2, loc="upper right")
    a.set_ylim(-0.5, 1.45)
    b = ax[1]
    x2 = np.linspace(0.01, 6, 400)
    b.semilogy(x2, special.i0(x2), color=NAVY, label=r"$I_0(x)$")
    b.semilogy(x2, special.i1(x2), color=GREEN, label=r"$I_1(x)$")
    b.semilogy(x2[x2 > 0.5], np.exp(x2[x2 > 0.5]) / np.sqrt(2 * np.pi * x2[x2 > 0.5]), "--", color=ACCENT,
               label=r"$e^x/\sqrt{2\pi x}$")
    b.set_xlabel("$x$"); b.set_title("Modified Bessel $I_n$"); b.legend(fontsize=6.5)
    c_ = ax[2]
    bb = np.linspace(0, 8, 400)
    for av, c in [(0, NAVY), (1, PURPLE), (2, GREEN), (3, ORANGE), (4, ACCENT)]:
        c_.plot(bb, stats.ncx2.sf(bb ** 2, 2, av ** 2) if av > 0 else np.exp(-bb ** 2 / 2), color=c, label=f"$a={av}$")
    c_.set_xlabel("$b$"); c_.set_title(r"Marcum $Q_1(a,b)$"); c_.legend(fontsize=6.5)
    save(fig, "appA_bessel_marcum")


def bands():
    itu = [("ELF", 3, 30), ("SLF", 30, 300), ("ULF", 300, 3e3), ("VLF", 3e3, 30e3), ("LF", 30e3, 300e3),
           ("MF", 300e3, 3e6), ("HF", 3e6, 30e6), ("VHF", 30e6, 300e6), ("UHF", 300e6, 3e9),
           ("SHF", 3e9, 30e9), ("EHF", 30e9, 300e9), ("THF", 300e9, 3e12)]
    ieee = [("L", 1e9, 2e9), ("S", 2e9, 4e9), ("C", 4e9, 8e9), ("X", 8e9, 12e9), ("Ku", 12e9, 18e9),
            ("K", 18e9, 27e9), ("Ka", 27e9, 40e9), ("V", 40e9, 75e9), ("W", 75e9, 110e9), ("mm", 110e9, 300e9)]
    apps1 = [(20e3, "VLF navy\n(submarines)", 0), (198e3, "longwave\nbroadcast", 1), (1e6, "AM radio", 0),
             (14e6, "shortwave,\namateur", 1), (98e6, "FM radio", 0), (700e6, "TV, cellular\nlow band", 1),
             (40e9, "mm-wave", 0)]
    apps2 = [(1.575e9, "GPS L1", 0), (2.45e9, "Wi-Fi,\nBluetooth", 1), (3.5e9, "5G n78", 0),
             (6e9, "C-band\nsatellite", 1), (11.7e9, "DTH TV\n(Ku)", 0), (20e9, "Ka sat\ndown", 1),
             (28e9, "5G FR2", 0), (60e9, "60 GHz\nWiGig", 1), (77e9, "auto\nradar", 0),
             (140e9, "D band\n(6G research)", 1)]
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.6), gridspec_kw=dict(hspace=0.75))
    def draw(a, segs, cols, xlim, apps, fs):
        a.set_xscale("log")
        for i, (n, lo, hi) in enumerate(segs):
            a.add_patch(plt.Rectangle((lo, 1.0), hi - lo, 0.8, color=cols[i % 2], alpha=0.9, lw=0))
            a.text(np.sqrt(lo * hi), 1.4, n, ha="center", va="center", color="white", fontsize=fs, fontweight="bold")
        for f, t, lev in apps:
            if not t:
                continue
            y1 = 0.62 if lev == 0 else 0.05
            a.plot([f, f], [y1 + 0.05, 0.98], color=GRAY, lw=0.7)
            a.text(f, y1, t, ha="center", va="top", fontsize=6.2, color="#333333", linespacing=0.95)
        a.set_xlim(*xlim); a.set_ylim(-0.75, 1.85)
        a.set_yticks([]); a.spines["left"].set_visible(False); a.grid(False)
    draw(ax[0], itu, [NAVY, "#2E86C1"], (3, 3e12), apps1, 7)
    ax[0].set_xticks([3, 3e3, 3e6, 3e9, 3e12]); ax[0].set_xticklabels(["3 Hz", "3 kHz", "3 MHz", "3 GHz", "3 THz"])
    ax[0].set_title(r"Radio bands (ITU-R V.431): one decade each, band $N$ spans $0.3\times10^N$ to $3\times10^N$ Hz", fontsize=8.5)
    draw(ax[1], ieee, [ACCENT, ORANGE], (1e9, 300e9), apps2, 7)
    ax[1].set_xticks([1e9, 2e9, 4e9, 8e9, 12e9, 18e9, 27e9, 40e9, 75e9, 110e9, 300e9])
    ax[1].set_xticklabels(["1", "2", "4", "8", "12", "18", "27", "40", "75", "110", "300"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("frequency (GHz)")
    ax[1].set_title("IEEE Std 521 radar letter bands (microwave detail)", fontsize=8.5)
    save(fig, "appA_bands")


# ============================================================ second edition: visual reference

NARROW = (3.0, 2.4)


def _clean(a):
    a.spines["left"].set_position("zero") if False else None
    return a


def complex_baseband():
    """Bandpass spectrum and its complex envelope."""
    f = np.linspace(-12, 12, 2000)
    fc, B = 7.0, 5.0
    env = lambda x: np.clip(1 - np.abs(x) / (B / 2), 0, None) * (1 + 0.35 * x / (B / 2))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.9))
    a = ax[0]
    a.fill_between(f, 0.5 * env(f - fc), color=NAVY, alpha=0.8, lw=0)
    a.fill_between(f, 0.5 * env(-f - fc), color=NAVY, alpha=0.4, lw=0)
    a.set_title(r"real bandpass signal $x(t)$: $|X(f)|$")
    a.set_xticks([-fc, 0, fc]); a.set_xticklabels([r"$-f_c$", "0", r"$f_c$"])
    a.set_yticks([0, 0.5]); a.set_yticklabels(["0", r"$\frac{1}{2}$"]); a.set_ylim(0, 1.25)
    a.annotate("mirror image\n(Hermitian)", (-fc, 0.35), (0, 0.95), fontsize=7.5, ha="center",
               arrowprops=dict(arrowstyle="->", color=GRAY))
    b = ax[1]
    b.fill_between(f, env(f), color=ACCENT, alpha=0.85, lw=0)
    b.set_title(r"complex envelope $\tilde{x}(t)$: $|\tilde X(f)|$")
    b.set_xticks([-B / 2, 0, B / 2]); b.set_xticklabels([r"$-\frac{B}{2}$", "0", r"$\frac{B}{2}$"])
    b.set_yticks([0, 1]); b.set_ylim(0, 1.25); b.set_xlim(-12, 12)
    b.annotate("shift down by $f_c$,\nkeep one side, double", (1.5, 0.9), (7.0, 0.95), fontsize=7.5, ha="center",
               arrowprops=dict(arrowstyle="->", color=GRAY))
    for x in ax:
        x.set_xlabel("frequency")
        x.grid(False)
    save(fig, "appA_baseband")


def ft_pairs():
    """Sketches of the most-used Fourier pairs."""
    t = np.linspace(-3, 3, 1201)
    f = np.linspace(-3, 3, 1201)
    rows = [
        ("rect$(t)$", np.where(np.abs(t) < 0.5, 1.0, 0.0), r"sinc$(f)$", np.sinc(f)),
        (r"$\Lambda(t)$", np.clip(1 - np.abs(t), 0, None), r"sinc$^2(f)$", np.sinc(f) ** 2),
        (r"$e^{-\pi t^2}$", np.exp(-np.pi * t ** 2), r"$e^{-\pi f^2}$", np.exp(-np.pi * f ** 2)),
        (r"$e^{-|t|}$", np.exp(-np.abs(t)), r"$\frac{2}{1+(2\pi f)^2}$", 2 / (1 + (2 * np.pi * f) ** 2)),
        ("sinc$(t)$", np.sinc(t), "rect$(f)$", np.where(np.abs(f) < 0.5, 1.0, 0.0)),
        (r"$\cos 2\pi t$", None, r"$\frac{1}{2}[\delta(f\!-\!1)+\delta(f\!+\!1)]$", None),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(W2, 3.9), gridspec_kw=dict(hspace=0.95, wspace=0.28))
    for k, (lt, xt, lf, xf) in enumerate(rows):
        a, b = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        if xt is None:
            a.plot(t, np.cos(2 * np.pi * t), color=NAVY)
            for s in (-1, 1):
                b.annotate("", (s, 0.5), (s, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
            b.set_ylim(-0.05, 0.65)
        else:
            a.plot(t, xt, color=NAVY)
            b.plot(f, xf, color=ACCENT)
        a.set_title(lt, fontsize=8.5, color=NAVY)
        b.set_title(lf, fontsize=8.5, color=ACCENT)
        for x, lab in ((a, "$t$"), (b, "$f$")):
            x.axhline(0, color="k", lw=0.5)
            x.set_xlim(-3, 3); x.set_xticks([-2, 0, 2]); x.set_yticks([])
            x.tick_params(labelsize=7); x.set_xlabel(lab, fontsize=7.5, labelpad=0)
            x.spines["left"].set_visible(False)
        a.annotate("", (3.6, 0.5), (3.15, 0.5), xycoords=("data", "axes fraction"),
                   arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9), annotation_clip=False)
    save(fig, "appA_ftpairs")


def rc_pulses():
    t = np.linspace(-4, 4, 2001)
    f = np.linspace(-1.1, 1.1, 1001)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    for beta, c in [(0.0, GRAY), (0.35, NAVY), (1.0, ACCENT)]:
        with np.errstate(divide="ignore", invalid="ignore"):
            den = 1 - (2 * beta * t) ** 2
            x = np.sinc(t) * np.where(np.abs(den) < 1e-9, np.pi / 4, np.cos(np.pi * beta * t) / np.where(np.abs(den) < 1e-9, 1, den))
        if beta > 0:
            sing = np.abs(np.abs(t) - 1 / (2 * beta)) < 2e-3
            x[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
        ax[0].plot(t, x, color=c, label=fr"$\beta={beta}$")
        af = np.abs(f)
        X = np.where(af <= (1 - beta) / 2, 1.0,
                     np.where(af <= (1 + beta) / 2,
                              0.5 * (1 + np.cos(np.pi / max(beta, 1e-9) * (af - (1 - beta) / 2))), 0.0))
        ax[1].plot(f, X, color=c, label=fr"$\beta={beta}$")
    ax[0].plot(np.arange(-4, 5), np.r_[np.zeros(4), 1, np.zeros(4)], "o", color=ACCENT, ms=3.5, mfc="white", zorder=5)
    ax[0].set_title("RC pulse: zero ISI at $t=nT$")
    ax[0].set_xlabel("$t/T$"); ax[0].legend(fontsize=7, loc="upper right")
    ax[1].set_title("spectrum: odd symmetry about $1/2T$")
    ax[1].set_xlabel("$fT$"); ax[1].axvline(0.5, color=GRAY, ls=":", lw=0.9); ax[1].axvline(-0.5, color=GRAY, ls=":", lw=0.9)
    ax[1].set_ylabel("$X_{RC}(f)/T$")
    save(fig, "appA_rc")


def sampling_picture():
    f = np.linspace(-3.2, 3.2, 3000)
    tri = lambda x, w: np.clip(1 - np.abs(x) / w, 0, None)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=True)
    for a, w, title in [(ax[0], 0.4, r"$f_s>2W$: copies separate"),
                        (ax[1], 0.7, r"$f_s<2W$: overlap = aliasing")]:
        tot = np.zeros_like(f)
        for k in range(-3, 4):
            y = tri(f - k, w)
            tot += y
            a.fill_between(f, y, color=NAVY if k == 0 else GRAY, alpha=0.55 if k == 0 else 0.3, lw=0)
        if w > 0.5:
            a.plot(f, tot, color=ACCENT, lw=1.2, label="sum (what the samples see)")
            a.legend(fontsize=7, loc="upper right")
        a.set_xticks([-2, -1, 0, 1, 2]); a.set_xticklabels([r"$-2f_s$", r"$-f_s$", "0", r"$f_s$", r"$2f_s$"])
        a.set_title(title, fontsize=8.2); a.set_yticks([]); a.set_ylim(0, 1.6); a.grid(False)
        a.spines["left"].set_visible(False)
    save(fig, "appA_sampling")


def windows_fig():
    from scipy.signal import get_window
    N, NF = 64, 8192
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for name, lab, c in [("boxcar", "rectangular", GRAY), ("hann", "Hann", NAVY), ("hamming", "Hamming", GREEN),
                         ("blackman", "Blackman", ORANGE), ("blackmanharris", "Blackman-Harris", ACCENT)]:
        w = get_window(name, N, fftbins=False)
        ax[0].plot(np.arange(N), w, color=c, label=lab, lw=1.1)
        W = np.abs(np.fft.fftshift(np.fft.fft(w, NF)))
        W = 20 * np.log10(W / W.max() + 1e-12)
        b = (np.arange(NF) - NF / 2) * N / NF
        ax[1].plot(b, W, color=c, lw=1.0)
    ax[0].set_title("time shapes ($N=64$)"); ax[0].set_xlabel("sample"); ax[0].legend(fontsize=6.5, loc="lower center")
    ax[0].set_ylim(-0.02, 1.25)
    ax[1].set_xlim(0, 12); ax[1].set_ylim(-120, 3)
    ax[1].set_xlabel("frequency offset (bins of $f_s/N$)"); ax[1].set_ylabel("dB")
    ax[1].set_title("spectra: wider main lobe buys lower sidelobes")
    save(fig, "appA_windows")


def zplane():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    a = ax[0]
    th = np.linspace(0, 2 * np.pi, 400)
    a.fill(np.cos(th), np.sin(th), color=GREEN, alpha=0.08)
    a.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    pts = [(0.8, 0.5, NAVY, "stable"), (1.0, 1.1, ORANGE, "on circle"), (1.15, 2.2, ACCENT, "unstable")]
    for r, w, c, lab in pts:
        for s in (1, -1):
            a.plot(r * np.cos(w), s * r * np.sin(w), "x", color=c, ms=7, mew=2)
    a.plot([-1.0], [0], "o", color=PURPLE, mfc="white", ms=6)
    a.text(-0.95, 0.12, "zero", fontsize=7, color=PURPLE)
    a.text(0, 0.0, "stable\nregion", fontsize=7, color=GREEN, ha="center", va="center")
    a.set_aspect("equal"); a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
    a.set_title("$z$-plane: poles (x) and zero (o)"); a.set_xlabel("Re $z$"); a.set_ylabel("Im $z$")
    b = ax[1]
    n = np.arange(40)
    for r, w, c, lab in pts:
        b.plot(n, r ** n * np.cos(w * n), ".-", color=c, ms=3, lw=0.8, label=f"$r={r}$ ({lab})")
    b.set_ylim(-4, 4); b.set_xlabel("$n$"); b.set_title(r"impulse response $r^n\cos(\omega_0 n)$")
    b.legend(fontsize=7, loc="lower left")
    save(fig, "appA_zplane")


def sinc_dirichlet():
    x = np.linspace(-4.5, 4.5, 2000)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, np.sinc(x), color=NAVY, label=r"sinc $x$")
    ax.plot(x, np.sinc(x) ** 2, color=ACCENT, label=r"sinc$^2 x$ (NRZ spectrum)")
    N = 8
    om = 2 * np.pi * x / N
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.sin(N * om / 2) / (N * np.sin(om / 2))
    D[np.abs(np.sin(om / 2)) < 1e-9] = 1
    ax.plot(x, D, "--", color=GREEN, lw=1.1, label=r"Dirichlet $D_8$ (periodic)")
    ax.plot([1.43], [-0.217], "o", color=NAVY, ms=4)
    ax.annotate("$-13.3$ dB\nsidelobe", (1.43, -0.217), (2.5, -0.45), fontsize=7, arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.55, 1.15)
    ax.set_xlabel("$x$ (zeros at the non-zero integers)"); ax.legend(fontsize=6.5, loc="upper right")
    save(fig, "appA_sinc")


def q_tail():
    x = np.linspace(-4, 5, 800)
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, phi, color=NAVY)
    m = x >= 2
    ax.fill_between(x[m], phi[m], color=ACCENT, alpha=0.7, lw=0)
    ax.annotate(r"$\mathrm{Q}(2)=2.3\%$" "\nof the area", (2.5, 0.01), (2.6, 0.22), fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.axvline(2, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("$u$ (standard deviations)"); ax.set_ylabel(r"$\phi(u)$")
    ax.set_title("Q$(x)$: the area beyond $x$")
    ax.set_ylim(0, 0.45)
    save(fig, "appA_qtail")


def gamma_fig():
    from scipy import special
    x = np.linspace(0, 12, 600)
    fig, ax = plt.subplots(figsize=NARROW)
    for s, c in [(0.5, PURPLE), (1, NAVY), (2, GREEN), (4, ORANGE), (8, ACCENT)]:
        ax.plot(x, special.gammainc(s, x), color=c, label=f"$s={s}$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$P(s,x)$")
    ax.set_title("regularised incomplete gamma = gamma CDF")
    ax.legend(fontsize=6.8, loc="lower right")
    save(fig, "appA_gammainc")


def ergodic_fig():
    from scipy import special
    g = np.linspace(-10, 30, 300)
    gl = 10 ** (g / 10)
    awgn = np.log2(1 + gl)
    ray = np.log2(np.e) * np.exp(1 / gl) * special.exp1(1 / gl)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(g, awgn, color=NAVY, label="AWGN")
    ax.plot(g, ray, color=ACCENT, label="Rayleigh, receiver CSI")
    ax.plot([10, 10], [2.91, 3.46], "o", color=GRAY, ms=3.5)
    ax.annotate("3.46 vs 2.91\nat 10 dB", (10, 3.2), (14, 1.5), fontsize=7, arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel(r"mean SNR $\bar\gamma$ (dB)"); ax.set_ylabel("b/s/Hz")
    ax.set_title(r"ergodic capacity via $E_1(1/\bar\gamma)$")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "appA_ergodic")


def mgf_ber():
    from scipy.special import comb
    g = np.linspace(0, 40, 400)
    gl = 10 ** (g / 10)
    fig, ax = plt.subplots(figsize=NARROW)
    from commlib import qfunc as _q
    ax.semilogy(g, _q(np.sqrt(2 * gl)), color=NAVY, label="AWGN")
    mu = np.sqrt(gl / (1 + gl))
    for L, c in [(1, ACCENT), (2, ORANGE), (4, GREEN)]:
        p = ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))
        ax.semilogy(g, p, color=c, label=f"Rayleigh, {L}-branch MRC" if L > 1 else "Rayleigh")
    ax.plot([20], [2.48e-3], "o", color=ACCENT, ms=4)
    ax.set_ylim(1e-7, 0.5); ax.set_xlim(0, 40)
    ax.set_xlabel(r"mean $E_b/N_0$ (dB)"); ax.set_ylabel("BPSK bit error rate")
    ax.set_title("fading costs orders of magnitude")
    ax.legend(fontsize=6.5, loc="lower left")
    save(fig, "appA_mgfber")


def gauss_vectors():
    r = rng(3)
    C = np.array([[2.0, 1.2], [1.2, 1.0]])
    L = np.linalg.cholesky(C)
    w = r.standard_normal((2, 1500))
    x = L @ w
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    th = np.linspace(0, 2 * np.pi, 200)
    for a, d, Lm, t in [(ax[0], x, L, r"correlated: $\mathbf{x}=\mathbf{L}\mathbf{w}$, $\mathbf{C}=\mathbf{L}\mathbf{L}^T$"),
                        (ax[1], w, np.eye(2), r"whitened: $\mathbf{L}^{-1}\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$")]:
        a.plot(d[0], d[1], ".", color=NAVY, ms=1.6, alpha=0.5)
        for k in (1, 2, 3):
            e = Lm @ np.vstack([np.cos(th), np.sin(th)]) * k
            a.plot(e[0], e[1], color=ACCENT, lw=0.9)
        a.set_aspect("equal"); a.set_xlim(-5, 5); a.set_ylim(-4, 4)
        a.set_title(t, fontsize=8.5)
    ev, U = np.linalg.eigh(C)
    for k in range(2):
        v = U[:, k] * np.sqrt(ev[k]) * 2
        ax[0].annotate("", v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
    ax[0].text(2.6, 2.6, "eigenvectors", color=GREEN, fontsize=7)
    save(fig, "appA_gaussvec")


def processes_fig():
    tau = np.linspace(-6, 6, 1000)
    f = np.linspace(-0.5, 0.5, 1000)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1), gridspec_kw=dict(wspace=0.4))
    for a_, c in [(0.5, NAVY), (0.85, ACCENT)]:
        m = np.arange(-12, 13)
        R = a_ ** np.abs(m) / (1 - a_ ** 2)
        ax[0].stem(m, R, linefmt=c, markerfmt="o", basefmt=" ")
        om = 2 * np.pi * f
        S = 1 / np.abs(1 - a_ * np.exp(-1j * om)) ** 2
        ax[1].plot(f, S, color=c, label=f"$a={a_}$")
    for l in ax[0].get_lines():
        l.set_markersize(2.5)
    ax[0].set_title("AR(1): $R_x[m]$", fontsize=8.5); ax[0].set_xlabel("lag $m$")
    ax[1].set_title(r"$S_x$: slower $\Rightarrow$ narrower", fontsize=8.5); ax[1].set_xlabel(r"$\omega/2\pi$")
    ax[1].legend(fontsize=6.5)
    fn = np.linspace(0, 4, 800)
    H2 = 1 / (1 + fn ** 2)
    ax[2].plot(fn, H2, color=NAVY, label="RC: $|H|^2$")
    ax[2].fill_between([0, np.pi / 2], [1, 1], color=GREEN, alpha=0.25, lw=0, label=r"$B_N=\frac{\pi}{2}f_{3dB}$")
    ax[2].axvline(1, color=GRAY, ls=":", lw=0.9)
    ax[2].set_xlabel(r"$f/f_{3\,\mathrm{dB}}$"); ax[2].set_title("noise bandwidth", fontsize=8.5)
    ax[2].legend(fontsize=6.3, loc="upper right"); ax[2].set_ylim(0, 1.25)
    save(fig, "appA_processes")


def svd_fig():
    H = np.array([[1.4, 0.6], [0.2, 0.7]])
    U, s, Vh = np.linalg.svd(H)
    th = np.linspace(0, 2 * np.pi, 300)
    c = np.vstack([np.cos(th), np.sin(th)])
    e = H @ c
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(c[0], c[1], color=GRAY, lw=1, label=r"unit inputs $\|\mathbf{x}\|=1$")
    ax.plot(e[0], e[1], color=NAVY, lw=1.6, label=r"outputs $\mathbf{Hx}$")
    for k, col in enumerate([ACCENT, GREEN]):
        ax.annotate("", Vh[k], (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, ls="--"))
        ax.annotate("", U[:, k] * s[k], (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.text(*(U[:, k] * s[k] * 1.12), fr"$\sigma_{k + 1}={s[k]:.2f}$", color=col, fontsize=7.5, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.0); ax.set_ylim(-1.3, 1.3)
    ax.set_title("SVD: rotate, stretch, rotate")
    ax.legend(fontsize=6.3, loc="lower right")
    save(fig, "appA_svd")


def circulant_fig():
    N = 16
    c = np.zeros(N); c[:4] = [1.0, 0.6, -0.3, 0.15]
    T = np.zeros((N + 3, N))
    for i in range(N):
        T[i:i + 4, i] = c[:4]
    Cm = np.array([np.roll(c, k) for k in range(N)]).T
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    D = F @ Cm @ F.conj().T
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    for a, M, t in [(ax[0], T, "Toeplitz (linear conv.)"), (ax[1], Cm, "circulant (cyclic prefix)"),
                    (ax[2], np.abs(D), r"$|\mathbf{F}\mathbf{C}\mathbf{F}^H|$: diagonal")]:
        a.imshow(np.abs(M) if M is not D else M, cmap="Blues", interpolation="nearest", vmin=0)
        a.set_title(t, fontsize=8.3); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "appA_circulant")


def waterfill_fig():
    floor = np.array([0.1, 0.2, 1.0]); mu = 0.65
    fig, ax = plt.subplots(figsize=NARROW)
    x = np.arange(3)
    ax.bar(x, floor, width=1.0, color=GRAY, edgecolor="white", label="floor $1/g_k$")
    water = np.clip(mu - floor, 0, None)
    ax.bar(x, water, bottom=floor, width=1.0, color="#2E86C1", alpha=0.75, edgecolor="white", label="power $p_k$")
    ax.axhline(mu, color=NAVY, ls="--", lw=1)
    ax.text(2.45, mu + 0.03, r"water level $\mu=0.65$", ha="right", fontsize=7.5, color=NAVY)
    for k in range(2):
        ax.text(k, floor[k] + water[k] / 2, f"{water[k]:.2f}", ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.text(2, 1.03, "dry: $p_3=0$", ha="center", fontsize=7.5, color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(["$g=10$", "$g=5$", "$g=1$"]); ax.set_ylim(0, 1.25)
    ax.set_title("water-filling the worked example"); ax.legend(fontsize=6.5, loc="upper left"); ax.grid(False)
    save(fig, "appA_waterfill")


def mvdr_fig():
    N = 8
    th = np.linspace(-90, 90, 1801)
    a = lambda d: np.exp(1j * np.pi * np.arange(N)[:, None] * np.sin(np.radians(np.atleast_1d(d)))[None, :])
    a0 = a(0)[:, 0]; ai = a(-40)[:, 0]
    R = 100 * np.outer(ai, ai.conj()) + np.eye(N)
    w_c = a0 / N
    Ri = np.linalg.inv(R)
    w_m = Ri @ a0 / (a0.conj() @ Ri @ a0)
    A = a(th)
    fig, ax = plt.subplots(figsize=NARROW)
    for w, c, lab, ls in [(w_m, NAVY, "MVDR", "-"), (w_c, ORANGE, "conventional $\\mathbf{a}/N$", "--")]:
        g = 20 * np.log10(np.abs(w.conj() @ A) + 1e-6)
        ax.plot(th, g, color=c, label=lab, ls=ls, lw=1.4 if ls == "-" else 0.9)
    ax.axvline(-40, color=ACCENT, ls=":", lw=1); ax.text(-39, -55, "interferer", color=ACCENT, fontsize=7)
    ax.set_ylim(-60, 5); ax.set_xlim(-90, 90)
    ax.set_xlabel("angle (degrees)"); ax.set_ylabel("gain (dB)")
    ax.set_title("MVDR nulls the interferer"); ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "appA_mvdr")


def descent_fig():
    A = np.array([[3.0, 1.2], [1.2, 1.0]]); b = np.array([1.0, 2.0])
    xs = np.linalg.solve(A, b)
    X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-1, 5, 200))
    P = np.stack([X, Y], -1)
    J = 0.5 * np.einsum("...i,ij,...j", P, A, P) - P @ b
    fig, ax = plt.subplots(figsize=NARROW)
    ax.contour(X, Y, J, levels=14, colors=GRAY, linewidths=0.6)
    x = np.array([-2.0, -0.5]); path = [x.copy()]
    for _ in range(25):
        x = x - 0.3 * (A @ x - b); path.append(x.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], ".-", color=ACCENT, ms=3.5, lw=1, label="gradient descent")
    ax.plot([-2.0, xs[0]], [-0.5, xs[1]], "--", color=GREEN, lw=1.4, label="Newton: one step")
    ax.plot(*xs, "*", color=NAVY, ms=10)
    ax.set_title("descending a quadratic cost"); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "appA_descent")


def db_ruler():
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.6), gridspec_kw=dict(hspace=1.3))
    a = ax[0]
    db = np.arange(-10, 11)
    a.set_xlim(-10.5, 10.5); a.set_ylim(0, 1)
    for d in db:
        a.plot([d, d], [0.45, 0.75 if d % 10 == 0 else 0.62], color=NAVY, lw=0.9)
        a.text(d, 0.82, f"{d}", ha="center", fontsize=6.8, color=NAVY)
    for d, txt in [(-10, "0.1"), (-6, "1/4"), (-3, "1/2"), (0, "1"), (1, "1.26"), (3, "2"), (6, "4"), (7, "5"), (10, "10")]:
        a.text(d, 0.12, txt, ha="center", fontsize=7.2, color=ACCENT)
        a.plot([d, d], [0.32, 0.45], color=ACCENT, lw=0.9)
    a.axhline(0.45, color=NAVY, lw=1.2)
    a.text(-10.8, 0.82, "dB", ha="right", fontsize=7.5, color=NAVY, weight="bold")
    a.text(-10.8, 0.12, "ratio", ha="right", fontsize=7.5, color=ACCENT, weight="bold")
    a.axis("off"); a.set_title("the decibel ruler: power ratios", fontsize=9)
    b = ax[1]
    marks = [(60, "1 kW\nbroadcast"), (43, "20 W\nbase station"), (30, "1 W"), (20, "Wi-Fi\nTX"), (0, "1 mW"),
             (-50, "strong\nRX"), (-100, "LTE\nsensitivity"), (-130, "GPS at\nthe antenna"), (-174, "$kT_0$\nin 1 Hz")]
    b.set_xlim(-185, 70); b.set_ylim(-0.45, 1)
    b.axhline(0.55, color=NAVY, lw=1.2)
    for d in range(-180, 61, 10):
        b.plot([d, d], [0.55, 0.72 if d % 30 == 0 else 0.64], color=NAVY, lw=0.8)
        if d % 30 == 0:
            b.text(d, 0.8, f"{d}", ha="center", fontsize=6.8, color=NAVY)
    for d, txt in marks:
        b.plot([d], [0.55], "o", color=ACCENT, ms=3.5)
        yy = 0.38 if d not in (30, 60) else -0.05
        b.text(d, yy, txt, ha="center", va="top", fontsize=6.3, color="#333333", linespacing=0.95)
        if yy < 0.3:
            b.plot([d, d], [0.5, 0.0], color=GRAY, lw=0.5)
    b.text(-188, 0.8, "dBm", ha="right", fontsize=7.5, color=NAVY, weight="bold")
    b.axis("off"); b.set_title("absolute power in dBm: 23 orders of magnitude on one line", fontsize=9)
    save(fig, "appA_dbruler")


def link_staircase():
    steps = [("TX power", 30), ("TX antenna", 15), ("free space\n1 km, 2.4 GHz", -100.0), ("cables,\nwalls", -10),
             ("RX antenna", 0)]
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    lvl = 0; x = 0
    for name, d in steps:
        new = lvl + d
        col = GREEN if d > 0 else (ACCENT if d < 0 else GRAY)
        ax.bar(x, new - lvl, bottom=lvl, color=col, width=0.6, alpha=0.85)
        ax.text(x, max(lvl, new) + 3, f"{d:+.0f} dB" if x else f"{d:+.0f} dBm", ha="center", fontsize=7)
        ax.text(x, -128, name, ha="center", va="top", fontsize=7)
        if x:
            ax.plot([x - 1.3, x - 0.3], [lvl, lvl], color=GRAY, lw=0.6, ls=":")
        lvl = new; x += 1
    ax.axhline(lvl, color=NAVY, lw=1.1, ls="--")
    ax.text(-0.5, lvl - 9, "received $-65$ dBm", fontsize=7.5, color=NAVY)
    noise = -174 + 10 * np.log10(20e6) + 6
    ax.axhline(noise, color=PURPLE, lw=1.1)
    ax.text(-0.5, noise - 17, f"noise floor $-174+73+6\\approx{noise:.0f}$ dBm (20 MHz, NF 6 dB)", fontsize=7, color=PURPLE)
    ax.annotate("", (x + 0.2, noise), (x + 0.2, lvl), arrowprops=dict(arrowstyle="<->", color=NAVY))
    ax.text(x + 0.35, (lvl + noise) / 2, "SNR\n30 dB", fontsize=7.5, color=NAVY, va="center")
    ax.set_xlim(-0.6, x + 1.2); ax.set_ylim(-125, 60)
    ax.set_xticks([]); ax.set_ylabel("level (dBm)")
    ax.set_title("a link budget is a running sum in decibels")
    save(fig, "appA_linkstair")


def noisefloor_fig():
    B = np.logspace(0, 9, 200)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(B, -174 + 10 * np.log10(B), color=NAVY)
    for b, lab in [(25e3, "NBFM"), (200e3, "GSM"), (3.84e6, "WCDMA"), (20e6, "LTE/Wi-Fi"), (100e6, "NR FR1"), (400e6, "FR2")]:
        y = -174 + 10 * np.log10(b)
        ax.plot(b, y, "o", color=ACCENT, ms=3.5)
        ax.text(b / 1.6, y + 3, lab, fontsize=6.3, ha="right")
    ax.set_xlabel("bandwidth (Hz)"); ax.set_ylabel("$kT_0B$ (dBm)")
    ax.set_title("thermal floor: +10 dB per decade")
    ax.set_ylim(-178, -80)
    save(fig, "appA_noisefloor")


def vswr_fig():
    s = np.linspace(1, 6, 400)
    g = (s - 1) / (s + 1)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(s, -10 * np.log10(1 - g ** 2), color=NAVY, label="mismatch loss (dB)")
    ax2 = ax.twinx()
    ax2.plot(s, 100 * g ** 2, color=ACCENT, label="reflected power (%)")
    ax2.set_ylabel("reflected power (%)", color=ACCENT)
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax.axvline(2, color=GRAY, ls=":", lw=1)
    ax.text(2.08, 2.6, "VSWR 2:\n11%, 0.51 dB", fontsize=7)
    ax.set_xlabel("VSWR"); ax.set_ylabel("mismatch loss (dB)", color=NAVY)
    ax.set_title("antenna mismatch")
    save(fig, "appA_vswr")


def mixer_fig():
    f = np.linspace(0, 12, 1000)
    fig, ax = plt.subplots(figsize=NARROW)
    for x0, c, lab, h in [(5, NAVY, "$f_1$", 1.0), (8, NAVY, "$f_2$", 1.0), (3, ACCENT, "$f_2-f_1$", 0.5), (13, ACCENT, "$f_1+f_2$", 0.5)]:
        ax.annotate("", (x0, h), (x0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        ax.text(x0, h + 0.05, lab, ha="center", color=c, fontsize=8)
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 1.3); ax.set_yticks([])
    ax.set_xlabel("frequency"); ax.set_title(r"$\cos a\cos b=\frac{1}{2}[\cos(a-b)+\cos(a+b)]$", fontsize=9)
    ax.text(14.2, 1.2, "inputs (navy)\nmixer products (red)", fontsize=7, va="top", ha="right")
    ax.grid(False)
    save(fig, "appA_mixer")


def series_fig():
    x = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, np.log2(1 + x), color=NAVY, label=r"$\log_2(1+x)$")
    ax.plot(x, 1.443 * x, "--", color=ACCENT, label=r"$1.443x$")
    ax.plot(x, 10 * np.log10(1 + x) / 10, color=GREEN, label=r"$10\log_{10}(1+x)/10$")
    ax.plot(x, 0.434 * x, ":", color=GREEN, label=r"$0.434x$")
    ax.set_xlabel("$x$ (linear SNR, or fractional change)")
    ax.set_title("small-$x$ expansions")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "appA_series")


def fourier_series_fig():
    t = np.linspace(-1, 1, 2000)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    sq = np.sign(np.cos(2 * np.pi * t))
    ax[0].plot(t, sq, color=GRAY, lw=1)
    for K, c in [(1, ACCENT), (3, ORANGE), (15, NAVY)]:
        y = sum(4 / (np.pi * k) * (-1) ** ((k - 1) // 2) * np.cos(2 * np.pi * k * t) for k in range(1, K + 1, 2))
        ax[0].plot(t, y, color=c, lw=1.0, label=f"up to $k={K}$")
    ax[0].legend(fontsize=6.5, loc="lower left", ncol=3); ax[0].set_ylim(-1.6, 1.5)
    ax[0].set_xlabel("$t/T_0$"); ax[0].set_title("partial Fourier sums of a square wave")
    k = np.arange(1, 16, 2)
    p = 2 * (2 / (np.pi * k)) ** 2
    ax[1].bar(k, 100 * p, color=[ACCENT] + [NAVY] * (len(k) - 1), width=1.1)
    ax[1].text(1.8, 79, "81% in the\nfundamental", fontsize=7, color=ACCENT)
    ax[1].set_xlabel("harmonic $k$"); ax[1].set_ylabel("% of power"); ax[1].set_title("Parseval")
    save(fig, "appA_fseries")


def bandpass_fig():
    """Allowed bandpass sampling rates (white wedges) for f_H/B."""
    x = np.linspace(1, 6, 600)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.fill_between(x, 0, 8, color=ACCENT, alpha=0.18, lw=0)
    for n in range(1, 7):
        lo = 2 * x / n
        hi = np.where(n > 1, 2 * (x - 1) / max(n - 1, 1), 99)
        ok = (lo <= hi) & (n <= np.floor(x))
        ax.fill_between(x, lo, np.minimum(hi, 8), where=ok, color="white", lw=0)
    ax.plot(x, 2 * x, color=NAVY, lw=1, label=r"$f_s=2f_H$")
    ax.plot(x, 2 * np.ones_like(x), "--", color=GREEN, lw=1, label=r"$f_s=2B$")
    ax.set_xlim(1, 6); ax.set_ylim(0, 8)
    ax.set_xlabel(r"$f_H/B$"); ax.set_ylabel(r"$f_s/B$")
    ax.set_title("bandpass sampling: white = alias-free")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "appA_bandpass")


def marcum_rice():
    """Rician envelope histogram with Marcum tail shaded (for the Marcum section)."""
    r = rng(5)
    a = 2.0
    z = a + r.standard_normal(200000) + 1j * r.standard_normal(200000)
    env = np.abs(z)
    fig, ax = plt.subplots(figsize=NARROW)
    h, e = np.histogram(env, 120, range=(0, 6), density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax.bar(c, h, width=e[1] - e[0], color=NAVY, alpha=0.5, lw=0, label="simulated envelope")
    from scipy import special
    pdf = c * np.exp(-(c ** 2 + a ** 2) / 2) * special.i0(a * c)
    ax.plot(c, pdf, color=NAVY)
    b = 3.0
    m = c >= b
    ax.fill_between(c[m], pdf[m], color=ACCENT, alpha=0.6, lw=0, label=r"$\mathrm{Q}_1(2,3)$")
    from scipy import stats
    ax.text(3.9, 0.10, f"$=${stats.ncx2.sf(b ** 2, 2, a ** 2):.3f}", color=ACCENT, fontsize=8)
    ax.set_xlabel("envelope $r$"); ax.set_title("Marcum Q: a Rician tail")
    ax.legend(fontsize=6.5, loc="upper right")
    save(fig, "appA_marcumtail")


def gauss_integral_fig():
    x = np.linspace(-3, 5, 600)
    a, b = 1.0, 3.0
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, np.exp(-a * x ** 2), color=GRAY, label=r"$e^{-ax^2}$")
    ax.plot(x, np.exp(-a * x ** 2 + b * x), color=NAVY, label=r"$e^{-ax^2+bx}$")
    ax.axvline(b / (2 * a), color=ACCENT, ls=":", lw=1)
    ax.text(b / (2 * a) + 0.1, 8.6, r"peak at $b/2a$," "\n" r"height $e^{b^2/4a}$", fontsize=7, color=ACCENT)
    ax.set_xlabel("$x$"); ax.set_title("completing the square: same shape, moved")
    ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(0, 10.5)
    save(fig, "appA_gaussint")


def inequality_fig():
    x = np.linspace(0.05, 3, 400)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, np.log(x), color=NAVY, label=r"$\ln x$")
    ax.plot(x, x - 1, color=ACCENT, ls="--", label=r"$x-1$")
    ax.fill_between(x, np.log(x), x - 1, color=ACCENT, alpha=0.12, lw=0)
    ax.plot([1], [0], "o", color=NAVY, ms=4)
    ax.text(1.1, -0.45, "touch only at $x=1$", fontsize=7)
    ax.set_xlabel("$x$"); ax.set_ylim(-2.5, 2.1)
    ax.set_title(r"$\ln x\leq x-1$: why relative entropy $\geq 0$")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "appA_lnineq")


def wf_gain_fig():
    from commlib import infotheory as it
    g = np.array([10.0, 5.0, 1.0])
    Pdb = np.linspace(-15, 20, 200)
    wf, eq = [], []
    for P in 10 ** (Pdb / 10):
        p, _ = it.waterfill(1 / g, P)
        wf.append(np.sum(np.log2(1 + g * np.asarray(p))))
        eq.append(np.sum(np.log2(1 + g * P / 3)))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    gain = 100 * (np.array(wf) / np.array(eq) - 1)
    ax.plot(Pdb, gain, color=NAVY)
    g0 = 100 * (4.40 / 3.95 - 1)
    ax.plot([0], [g0], "o", color=ACCENT, ms=4)
    ax.annotate("worked example:\n4.40 vs 3.95 b/s/Hz", (0, g0), (4, 45), fontsize=7,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("total power $P$ (dB)"); ax.set_ylabel("rate gain over equal power (%)")
    ax.set_title("water-filling pays most at low power")
    ax.set_ylim(0, 95)
    save(fig, "appA_wfgain")


if __name__ == "__main__":
    wf_gain_fig()
    gauss_integral_fig()
    inequality_fig()
    qfunc_bounds()
    fading_pdfs()
    chi2_family()
    bessel_marcum()
    bands()
    complex_baseband()
    ft_pairs()
    rc_pulses()
    sampling_picture()
    windows_fig()
    zplane()
    sinc_dirichlet()
    q_tail()
    gamma_fig()
    ergodic_fig()
    mgf_ber()
    gauss_vectors()
    processes_fig()
    svd_fig()
    circulant_fig()
    waterfill_fig()
    mvdr_fig()
    descent_fig()
    db_ruler()
    link_staircase()
    noisefloor_fig()
    vswr_fig()
    mixer_fig()
    series_fig()
    fourier_series_fig()
    bandpass_fig()
    marcum_rice()
