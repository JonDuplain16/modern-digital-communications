"""Figures for Chapter 3: Random Signals and Noise."""
from figstyle import *
from scipy import stats, signal as sps
from scipy.special import erfc
import commlib as cl


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


def fig_q():
    x = np.linspace(0, 7, 400)
    q = 0.5 * erfc(x / np.sqrt(2))
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.6))
    ax.semilogy(x, q, color=NAVY, lw=1.8, label=r"$Q(x)$")
    ax.semilogy(x, 0.5 * np.exp(-x ** 2 / 2), color=ACCENT, ls="--", label=r"Chernoff bound $\frac{1}{2}e^{-x^2/2}$")
    xx = x[x > 0.5]
    ax.semilogy(xx, np.exp(-xx ** 2 / 2) / (xx * np.sqrt(2 * np.pi)), color=GREEN, ls=":", lw=1.6, label=r"upper bound $\frac{e^{-x^2/2}}{x\sqrt{2\pi}}$")
    ax.set_ylim(1e-12, 1); ax.set_xlabel("$x$"); ax.legend(fontsize=7.5)
    for v in [1e-3, 1e-6, 1e-9]:
        xv = np.sqrt(2) * __import__("scipy.special", fromlist=["erfcinv"]).erfcinv(2 * v)
        ax.plot([xv], [v], "o", color=GRAY, ms=3)
        ax.annotate(f"$Q={v:.0e}$ at $x={xv:.2f}$", (xv, v), (6, 2), textcoords="offset points", fontsize=6.8)
    ax.set_title("The Gaussian tail function and two bounds")
    fig.tight_layout(); save(fig, "ch03_q")


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


def fig_skynoise():
    """Illustrative external noise temperature vs frequency (order-of-magnitude, after ITU-R P.372 trends)."""
    f = np.logspace(np.log10(3e6), np.log10(100e9), 400)
    galactic = 1e4 * (f / 30e6) ** -2.4
    manmade_city = 2e6 * (f / 10e6) ** -2.77
    atm = 3 + 280 * (1 - np.exp(-((f / 22.2e9) ** 2) * 0.05)) + 250 * np.exp(-((f - 60e9) / 4e9) ** 2)
    cmb = np.full_like(f, 2.7)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    ax.loglog(f / 1e6, galactic, label="galactic (cold sky)")
    ax.loglog(f / 1e6, manmade_city, label="man-made, city")
    ax.loglog(f / 1e6, atm, label="atmospheric (clear, low elevation)")
    ax.loglog(f / 1e6, cmb, ls=":", color=GRAY, label="cosmic background 2.7 K")
    ax.axhline(290, color=ACCENT, lw=0.8, ls="--"); ax.text(4, 330, "$T_0 = 290$ K", fontsize=7, color=ACCENT)
    ax.fill_betweenx([1, 1e7], 1000, 10000, color=GREEN, alpha=0.07); ax.text(1300, 2e6, "\"microwave\nwindow\"", fontsize=7, color=GREEN)
    ax.set_ylim(1, 1e7); ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("noise temperature (K)")
    ax.legend(fontsize=6.8, loc="upper right"); ax.set_title("External noise seen by an antenna (illustrative trends)")
    fig.tight_layout(); save(fig, "ch03_skynoise")


def fig_linkbudget():
    items = [("Tx power", 20), ("Tx antenna", 3), ("path loss\n(50 m, 5.5 GHz,\nn = 3)", None), ("wall", -8),
             ("Rx antenna", 3), ("received", None)]
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


if __name__ == "__main__":
    fig_clt(); fig_q(); fig_filtered_noise(); fig_envelopes(); fig_friis(); fig_skynoise(); fig_linkbudget()
