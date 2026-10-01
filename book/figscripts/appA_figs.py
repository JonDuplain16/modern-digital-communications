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


if __name__ == "__main__":
    qfunc_bounds()
    fading_pdfs()
    chi2_family()
    bessel_marcum()
    bands()
