"""Figures for Chapter 21: Cellular Generations, from AMPS to 5G.

Run:  python ch21_figs.py               (all figures)
      python ch21_figs.py gmsk lte_grid (selected functions)

Market and traffic data are approximate values compiled from ITU-T/ITU-D statistics and the
Ericsson Mobility Reports; they are drawn to show trends, not to be quoted to three digits.
Physical-layer figures (GSM frame, GMSK, LTE grid, NR SSB) follow 3GPP TS 45.002, TS 45.004,
TS 36.211 and TS 38.211/38.213.
"""
from figstyle import *
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import ListedColormap
from scipy import signal as sps
from scipy.special import erfc
import commlib as cl

LIGHT = "#D6E4F0"
BLUE2 = "#2E86C1"
GEN_COL = {"1G": GRAY, "2G": NAVY, "3G": GREEN, "4G": ORANGE, "5G": ACCENT}


# ============================================================================ 1
def growth():
    """Subscriptions by generation and mobile data traffic (approximate)."""
    yrs = np.array([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
    total = np.array([0.011, 0.091, 0.74, 2.2, 5.3, 7.2, 8.0, 8.8])         # billions
    share = np.array([  # 1G, 2G, 3G, 4G, 5G
        [1.00, 0.00, 0.00, 0.00, 0.00],
        [0.62, 0.38, 0.00, 0.00, 0.00],
        [0.12, 0.88, 0.00, 0.00, 0.00],
        [0.01, 0.95, 0.04, 0.00, 0.00],
        [0.00, 0.78, 0.21, 0.01, 0.00],
        [0.00, 0.55, 0.31, 0.14, 0.00],
        [0.00, 0.20, 0.20, 0.57, 0.03],
        [0.00, 0.08, 0.07, 0.52, 0.33]])
    yy = np.arange(1990, 2025.01, 0.25)
    sh = np.array([np.interp(yy, yrs, share[:, k]) for k in range(5)])
    tot = np.exp(np.interp(yy, yrs, np.log(total)))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    ax[0].stackplot(yy, sh * tot, colors=[GEN_COL[g] for g in GEN_COL], alpha=0.85,
                    labels=list(GEN_COL), edgecolor="white", lw=0.3)
    ax[0].set_xlim(1990, 2025); ax[0].set_ylim(0, 9.5)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("subscriptions (billions)")
    ax[0].legend(loc="upper left", fontsize=7.5, ncol=1, frameon=False)
    ax[0].set_title("(a) Mobile subscriptions by generation", fontsize=9)
    for x, t in [(1983, ""), (2007.5, "iPhone"), (2009.9, "first LTE"), (2019.3, "first 5G")]:
        if t:
            ax[0].annotate(t, (x, np.interp(x, yy, tot)), xytext=(x - 6.5, np.interp(x, yy, tot) + 1.2),
                           fontsize=7, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    ty = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    eb = np.array([0.24, 0.9, 2.6, 7.5, 24, 50, 105, 160])               # EB per month
    ax[1].semilogy(ty, eb, "o-", color=NAVY, ms=4)
    p = np.polyfit(ty, np.log(eb), 1)
    ax[1].semilogy(ty, np.exp(np.polyval(p, ty)), "--", color=ACCENT, lw=1,
                   label=f"fit: x{np.exp(p[0]):.2f} per year")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("exabytes per month")
    ax[1].set_title("(b) Global mobile data traffic", fontsize=9)
    ax[1].legend(fontsize=7.5)
    for a in ax:
        a.text(0.99, 0.02, "approximate", transform=a.transAxes, ha="right", fontsize=6.5, color=GRAY)
    fig.tight_layout(w_pad=1.2); save(fig, "ch21_growth")
    print("traffic growth factor per year", np.exp(p[0]))


# ============================================================================ 2
def spectrum_bands():
    """Major cellular bands by generation of first use, on a log-frequency axis."""
    bands = [  # (label, lo MHz, hi MHz, generation, row)
        ("600\n(n71)", 617, 698, "5G", 4),
        ("700\n(B12/13/28)", 698, 806, "4G", 3),
        ("850\n(AMPS)", 824, 894, "1G", 0),
        ("900\n(GSM)", 880, 960, "2G", 1),
        ("1800 DCS / 1900 PCS", 1710, 1880, "2G", 1),
        ("", 1850, 1990, "2G", 1),
        ("2100\n(UMTS)", 1920, 2170, "3G", 2),
        ("AWS", 2110, 2200, "4G", 3),
        ("2.3 G", 2300, 2400, "4G", 3),
        ("2.6 G", 2500, 2690, "4G", 3),
        ("CBRS", 3550, 3700, "4G", 3),
        ("C-band\nn77/n78", 3300, 4200, "5G", 4),
        ("n79", 4400, 5000, "5G", 4),
        ("n258", 24250, 27500, "5G", 4),
        ("n257/n261", 26500, 29500, "5G", 4),
        ("n260", 37000, 40000, "5G", 4),
        ("n259", 39500, 43500, "5G", 4),
    ]
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ax.axvspan(410, 7125, color=LIGHT, alpha=0.5, lw=0)
    ax.axvspan(24250, 52600, color="#FBE3D6", alpha=0.6, lw=0)
    ax.text(1700, 5.75, "FR1: 410 MHz - 7.125 GHz", fontsize=7.5, color=NAVY, ha="center")
    ax.text(35000, 5.75, "FR2: 24.25 - 52.6 GHz", fontsize=7.5, color=ACCENT, ha="center")
    off = {}
    for lab, lo, hi, g, row in bands:
        k = off.get(row, 0)
        ax.add_patch(Rectangle((lo, row + 0.12), hi - lo, 0.55, color=GEN_COL[g], alpha=0.85, lw=0))
        yt = row + 0.12 - 0.05 - (0.0 if k % 2 == 0 else 0.0)
        ax.text(np.sqrt(lo * hi), row + 0.75 + 0.33 * (k % 2), lab.replace("\n", " "),
                fontsize=5.8, ha="center", va="bottom", color="black")
        off[row] = k + 1
    ax.set_xscale("log"); ax.set_xlim(400, 60000); ax.set_ylim(-0.1, 6.05)
    ax.set_yticks(np.arange(5) + 0.4); ax.set_yticklabels(["1G", "2G", "3G", "4G", "5G"])
    for t, c in zip(ax.get_yticklabels(), GEN_COL.values()):
        t.set_color(c); t.set_fontweight("bold")
    ax.set_xticks([500, 1000, 2000, 3500, 6000, 10000, 28000, 40000])
    ax.set_xticklabels(["0.5", "1", "2", "3.5", "6", "10", "28", "40"])
    ax.set_xlabel("frequency (GHz), log scale"); ax.grid(axis="y", alpha=0)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); save(fig, "ch21_spectrum_bands")


# ============================================================================ 3
def gsm_frames():
    """The GSM time hierarchy from hyperframe to burst (TS 45.002)."""
    fig, ax = plt.subplots(figsize=(W2, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    X0, X1 = 4, 96
    H = 6.5

    def row(y, n, labels=None, colors=None, title="", dur="", fs=6.2):
        w = (X1 - X0) / n
        for i in range(n):
            c = colors[i] if colors else LIGHT
            ax.add_patch(Rectangle((X0 + i * w, y), w, H, facecolor=c, edgecolor="white", lw=0.8))
            if labels and labels[i]:
                ax.text(X0 + (i + 0.5) * w, y + H / 2, labels[i], ha="center", va="center", fontsize=fs)
        ax.text(X0, y + H + 1.2, title, fontsize=7.6, color=NAVY, fontweight="bold", va="bottom")
        ax.text(X1, y + H + 1.2, dur, fontsize=7.4, color=ACCENT, va="bottom", ha="right")
        return w

    def zoom(y_from, xa, xb, y_to):
        for xs, xt in [(xa, X0), (xb, X1)]:
            ax.plot([xs, xt], [y_from, y_to + H], color=GRAY, lw=0.5, ls=":")

    ys = [86, 69, 52, 35, 18, 2]
    # hyperframe
    lab = ["0", "1", "2", "3", "", "", "...", "", "", "2045", "2046", "2047"]
    w = row(ys[0], 12, lab, [LIGHT] * 12, "Hyperframe = 2048 superframes", "3 h 28 min 53.76 s")
    ax.add_patch(Rectangle((X0 + 1 * w, ys[0]), w, H, facecolor=NAVY, alpha=0.35, lw=0))
    zoom(ys[0], X0 + w, X0 + 2 * w, ys[1])
    # superframe
    lab = ["0", "1", "2", "", "...", "", "48", "49", "50"]
    w = row(ys[1], 9, lab, None, "Superframe = 51 x 26-multiframes (or 26 x 51)", "6.12 s")
    ax.add_patch(Rectangle((X0 + 1 * w, ys[1]), w, H, facecolor=NAVY, alpha=0.35, lw=0))
    zoom(ys[1], X0 + w, X0 + 2 * w, ys[2])
    # 26-multiframe
    cols = [LIGHT] * 26
    cols[12] = "#F5CBA7"; cols[25] = "#D5D8DC"
    lab = [str(i) if i in (0, 11) else "" for i in range(26)]
    lab[12] = "S"; lab[25] = "I"
    w = row(ys[2], 26, lab, cols, "26-multiframe (traffic): 24 TCH + 1 SACCH (S) + 1 idle (I)", "120 ms")
    ax.add_patch(Rectangle((X0 + 3 * w, ys[2]), w, H, facecolor=NAVY, alpha=0.35, lw=0))
    zoom(ys[2], X0 + 3 * w, X0 + 4 * w, ys[3])
    # TDMA frame
    lab = [f"TS{i}" for i in range(8)]
    cols = [LIGHT] * 8; cols[2] = "#A9CCE3"
    w = row(ys[3], 8, lab, cols, "TDMA frame = 8 timeslots (one user per slot per carrier)", "4.615 ms = 60/13 ms")
    zoom(ys[3], X0 + 2 * w, X0 + 3 * w, ys[4])
    # timeslot -> bits
    ax.add_patch(Rectangle((X0, ys[4]), X1 - X0, H, facecolor="#A9CCE3", edgecolor="white"))
    ax.text(50, ys[4] + H / 2, r"timeslot: 156.25 bit periods of 48/13 $\mu$s = 3.69 $\mu$s",
            ha="center", va="center", fontsize=6.8)
    ax.text(X0, ys[4] + H + 1.2, "Timeslot", fontsize=7.6, color=NAVY, fontweight="bold", va="bottom")
    ax.text(X1, ys[4] + H + 1.2, r"576.9 $\mu$s", fontsize=7.4, color=ACCENT, va="bottom", ha="right")
    zoom(ys[4], X0, X1, ys[5])
    # normal burst
    fields = [("T", 3, "#D5D8DC"), ("57 data", 57, LIGHT), ("F", 1, "#F5CBA7"),
              ("26 training", 26, "#ABEBC6"), ("F", 1, "#F5CBA7"), ("57 data", 57, LIGHT),
              ("T", 3, "#D5D8DC"), ("G", 8.25, "white")]
    x = X0
    sc = (X1 - X0) / 156.25
    for name, n, c in fields:
        ax.add_patch(Rectangle((x, ys[5]), n * sc, H, facecolor=c, edgecolor=GRAY, lw=0.5))
        ax.text(x + n * sc / 2, ys[5] + H / 2, name, ha="center", va="center", fontsize=6.2)
        x += n * sc
    ax.text(X0, ys[5] + H + 1.2, "Normal burst (bits)", fontsize=7.6, color=NAVY,
            fontweight="bold", va="bottom")
    ax.text(X1, ys[5] + H + 1.2, "T = tail (3), F = stealing flag (1), G = guard (8.25)", fontsize=7, color=GRAY,
            va="bottom", ha="right")
    save(fig, "ch21_gsm_frames")


# ============================================================================ 4
def _gmsk(bits, bt, sps_):
    """Complex baseband GMSK (h = 1/2) with Gaussian frequency pulse of bandwidth-time product bt."""
    a = 2.0 * bits - 1
    t = (np.arange(-2 * sps_, 2 * sps_ + 1)) / sps_
    if bt is None:
        g = np.ones(sps_) / sps_
    else:
        sig = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        gg = np.exp(-t ** 2 / (2 * sig ** 2))
        g = np.convolve(gg / gg.sum(), np.ones(sps_)) / sps_
    up = np.zeros(len(a) * sps_); up[::sps_] = a
    freq = np.convolve(up, g)
    phase = np.pi / 2 * np.cumsum(freq)
    return np.exp(1j * phase), g, phase


def gmsk():
    r = rng(4)
    sps_ = 16
    Rb = 1625e3 / 6                      # 270.833 kb/s
    fs = Rb * sps_
    bits = r.integers(0, 2, 40000)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.65))
    for bt, c, lab in [(None, GRAY, "MSK"), (0.5, BLUE2, "GMSK BT = 0.5"), (0.3, NAVY, "GMSK BT = 0.3 (GSM)")]:
        x, g, _ = _gmsk(bits, bt, sps_)
        f, p = sps.welch(x, fs=fs, nperseg=4096, return_onesided=False, window="blackmanharris")
        i = np.argsort(f); f, p = f[i], p[i]
        p = p / p.max()
        ax[0].plot(f / 1e3, 10 * np.log10(p + 1e-14), color=c, lw=1.1, label=lab)
    mask_f = np.array([0, 100, 200, 250, 400, 600])
    mask_d = np.array([0.5, 0.5, -30, -33, -60, -60])
    ax[0].plot(np.r_[-mask_f[::-1], mask_f], np.r_[mask_d[::-1], mask_d], color=ACCENT,
               lw=0.9, ls="--", label="approx. GSM mask")
    ax[0].set_xlim(-600, 600); ax[0].set_ylim(-80, 5)
    ax[0].set_xlabel("offset from carrier (kHz)");     ax[0].set_ylabel("normalised PSD (dB)")
    ax[0].legend(fontsize=6.5, loc="lower center"); ax[0].set_title("(a) Spectrum at 270.833 kb/s", fontsize=9)
    for k in (-200, 200):
        ax[0].axvline(k, color=GRAY, lw=0.5, ls=":")
    # frequency pulses and phase paths
    t = np.arange(-2 * sps_, 3 * sps_) / sps_
    for bt, c, lab in [(None, GRAY, "MSK"), (0.3, NAVY, "GMSK 0.3")]:
        _, g, _ = _gmsk(np.array([1]), bt, sps_)
        gt = np.zeros_like(t); n0 = 2 * sps_ - (len(g) - sps_) // 2
        gt[max(n0, 0):max(n0, 0) + len(g)] = g[: len(gt) - max(n0, 0)]
        ax[1].plot(t, gt * sps_, color=c, lw=1.2, label=f"{lab} frequency pulse")
    ax[1].set_xlabel("time (bit periods)"); ax[1].set_ylabel("instantaneous frequency (norm.)")
    ax[1].set_title("(b) Gaussian filtering spreads each bit", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_xlim(-2, 3); ax[1].set_ylim(-0.05, 1.25)
    ax[1].axvspan(0, 1, color=LIGHT, alpha=0.5, lw=0)
    ax[1].text(0.5, 1.12, "one bit", ha="center", fontsize=7)
    fig.tight_layout(w_pad=1.0); save(fig, "ch21_gmsk")


# ============================================================================ 5
def power_control():
    """WCDMA 1500 Hz closed-loop power control in Rayleigh fading."""
    r = rng(7)
    slot = 1 / 1500
    fc = 2.0e9
    n = 30000
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    for v, c in [(3, NAVY), (50, GREEN), (120, ACCENT)]:
        fd = v / 3.6 * fc / 3e8
        h = cl.jakes_process(n, fd * slot, rng=r)
        g = np.abs(h) ** 2
        gdb = 10 * np.log10(g)
        p = np.zeros(n); sir = np.zeros(n)
        pc = 0.0
        tpc_err = r.random(n) < 0.04
        for k in range(n):
            p[k] = pc
            sir[k] = pc + gdb[k] + 0.0
            est = sir[k] + 0.7 * r.standard_normal()
            cmd = -1 if est > 0 else 1
            if tpc_err[k]:
                cmd = -cmd
            if k >= 1:
                pc = np.clip(pc + cmd * 1.0, -25, 25)
        res[v] = (gdb, p, sir)
        if v == 3:
            tt = np.arange(300) * slot * 1e3
            ax[0].plot(tt, gdb[1000:1300], color=GRAY, lw=1, label="channel gain")
            ax[0].plot(tt, p[1000:1300], color=NAVY, lw=1, label="UE transmit power")
            ax[0].plot(tt, sir[1000:1300], color=ACCENT, lw=1.1, label="received SIR")
        s = np.sort(sir[100:])
        ax[1].plot(s, np.arange(len(s)) / len(s), color=c, lw=1.2, label=f"closed loop, {v} km/h")
        print(f"{v} km/h: fd={fd:.1f} Hz  SIR std={np.std(sir[100:]):.2f} dB  mean tx power={10*np.log10(np.mean(10**(p/10))):.2f} dB")
    s = np.sort(res[3][0])
    ax[1].plot(s, np.arange(len(s)) / len(s), color=GRAY, lw=1.2, ls="--", label="no power control")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("dB (relative)")
    ax[0].set_title("(a) 3 km/h, 1 dB steps every 0.667 ms", fontsize=9)
    ax[0].legend(fontsize=6.8, loc="lower left", ncol=1); ax[0].set_ylim(-30, 15)
    ax[1].set_xlim(-25, 10); ax[1].set_xlabel("received SIR relative to target (dB)")
    ax[1].set_ylabel("CDF"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-3, 1)
    ax[1].set_title("(b) Distribution of received SIR", fontsize=9); ax[1].legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(w_pad=1.0); save(fig, "ch21_power_control")


# ============================================================================ 6
def soft_handover():
    """Macrodiversity: outage at the cell edge with hard vs soft handover under shadowing."""
    r = rng(9)
    n = 400000
    sig = 8.0
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    m = np.linspace(0, 20, 201)
    for rho, c in [(0.0, GREEN), (0.5, NAVY)]:
        a = r.standard_normal(n); b = r.standard_normal(n); cmn = r.standard_normal(n)
        s1 = sig * (np.sqrt(rho) * cmn + np.sqrt(1 - rho) * a)
        s2 = sig * (np.sqrt(rho) * cmn + np.sqrt(1 - rho) * b)
        best = np.maximum(s1, s2)
        out_soft = np.array([np.mean(best < -mm) for mm in m])
        ax[0].semilogy(m, out_soft, color=c, lw=1.3, label=f"best of two cells, $\\rho$ = {rho}")
    out_hard = 0.5 * erfc(m / sig / np.sqrt(2))
    ax[0].semilogy(m, out_hard, color=ACCENT, lw=1.3, ls="--", label="one cell (hard handover)")
    ax[0].axhline(0.1, color=GRAY, lw=0.6, ls=":")
    ax[0].set_xlabel("fade margin (dB)"); ax[0].set_ylabel("edge outage probability")
    ax[0].set_ylim(1e-3, 0.6); ax[0].legend(fontsize=6.8)
    ax[0].set_title("(a) Macrodiversity, shadowing $\\sigma$ = 8 dB", fontsize=9)
    # (b) pilot Ec/Io of two cells along a route from site A to site B (2 km apart)
    d = np.linspace(0.03, 0.97, 400)
    pa = -35 * np.log10(d * 2); pb = -35 * np.log10((1 - d) * 2)
    noise = -25.0                                     # thermal + other-cell, relative units (dB)
    io = 10 * np.log10(10 ** (pa / 10) + 10 ** (pb / 10) + 10 ** (noise / 10))
    eca = pa - 10 + 0 - io; ecb = pb - 10 - io        # pilot = 10% of cell power
    ax[1].plot(d, eca, color=NAVY, label="pilot A, $E_c/I_o$")
    ax[1].plot(d, ecb, color=GREEN, label="pilot B, $E_c/I_o$")
    both = np.abs(eca - ecb) < 4.0
    x0, x1 = d[both][0], d[both][-1]
    ax[1].axvspan(x0, x1, color=LIGHT, alpha=0.8, lw=0)
    ax[1].text(0.84, -18.0, "shaded: both\npilots within\n4 dB of the best\n= soft handover", ha="center", fontsize=6.8)
    ax[1].set_ylim(-22.5, -9)
    ax[1].set_xlabel("position from site A to site B")
    ax[1].set_ylabel("pilot $E_c/I_o$ (dB)")
    ax[1].set_title("(b) The active set along a route", fontsize=9)
    ax[1].legend(fontsize=6.8, loc="center left")
    fig.tight_layout(w_pad=1.0); save(fig, "ch21_soft_handover")
    i = np.argmin(np.abs(out_hard - 0.1))
    print("hard handover margin for 10% outage:", m[i])


# ============================================================================ 7
CQI = [(2, 78), (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 490), (4, 616),
       (6, 466), (6, 567), (6, 666), (6, 772), (6, 873), (6, 948)]


def cqi_harq():
    """LTE CQI table (TS 36.213 Table 7.2.3-1) vs Shannon; HARQ chase vs incremental redundancy."""
    se = np.array([q * r / 1024 for q, r in CQI])
    snr = np.linspace(-10, 25, 701)
    gap = 2.0
    req = 10 * np.log10(2 ** se - 1) + gap
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    ax[0].plot(snr, np.log2(1 + 10 ** (snr / 10)), "k--", lw=1, label="Shannon")
    idx = np.searchsorted(req, snr, side="right") - 1
    thr = np.where(idx >= 0, se[np.clip(idx, 0, None)], 0) * 0.9
    ax[0].plot(snr, thr, color=NAVY, lw=1.3, label="CQI 1-15, 10% BLER")
    for k, (q, rr) in enumerate(CQI):
        mk = {2: "o", 4: "s", 6: "^"}[q]
        ax[0].plot(req[k], se[k], mk, color={2: GREEN, 4: ORANGE, 6: ACCENT}[q], ms=3.5)
    ax[0].plot([], [], "o", color=GREEN, ms=3.5, label="QPSK")
    ax[0].plot([], [], "s", color=ORANGE, ms=3.5, label="16QAM")
    ax[0].plot([], [], "^", color=ACCENT, ms=3.5, label="64QAM")
    ax[0].set_xlabel("SINR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_xlim(-10, 25); ax[0].set_ylim(0, 7)
    ax[0].legend(fontsize=6.6, loc="upper left")
    ax[0].set_title(f"(a) LTE CQI table, {gap:.0f} dB gap model", fontsize=9)
    # (b) HARQ with a fixed high-rate first transmission
    R0 = 3.0                    # b/s/Hz on the first transmission (e.g. 64QAM, rate 1/2)
    snrl = 10 ** (snr / 10)
    gapl = 10 ** (gap / 10)

    def p_fail(eff_cap):        # smooth BLER: success when capacity exceeds required rate
        return 1 / (1 + np.exp((eff_cap - R0) / 0.12))
    for scheme, c in [("none", GRAY), ("chase", NAVY), ("ir", ACCENT)]:
        kmax = 1 if scheme == "none" else 4
        en = np.zeros_like(snr); succ = np.zeros_like(snr); pall = np.ones_like(snr)
        for k in range(1, kmax + 1):
            if scheme == "ir":
                cap = k * np.log2(1 + snrl / gapl)
            else:
                cap = np.log2(1 + k * snrl / gapl)
            pf = p_fail(cap)
            # probability transmission k happens = all previous failed
            en += pall
            ps_k = pall * (1 - pf)
            succ += ps_k
            pall = pall * pf
        tput = R0 * succ / en
        lab = {"none": "no HARQ", "chase": "chase combining (4 tx)", "ir": "incremental redundancy (4 tx)"}[scheme]
        ax[1].plot(snr, tput, color=c, lw=1.3, label=lab)
    ax[1].plot(snr, np.log2(1 + snrl / gapl), "k--", lw=0.8, label="capacity with gap")
    ax[1].set_xlim(-10, 20); ax[1].set_ylim(0, 4)
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("throughput (b/s/Hz)")
    ax[1].legend(fontsize=6.6, loc="upper left")
    ax[1].set_title("(b) HARQ, first transmission at 3 b/s/Hz", fontsize=9)
    fig.tight_layout(w_pad=1.0); save(fig, "ch21_cqi_harq")


# ============================================================================ 8
def lte_grid():
    """LTE FDD subframe 0, 1.4 MHz (6 RBs), normal CP, 2 CRS ports (TS 36.211)."""
    nsc, nsym = 72, 14
    G = np.full((nsc, nsym), 0)             # 0 PDSCH
    cfi = 3
    G[:, :cfi] = 1                           # control region (PDCCH/PCFICH/PHICH)
    pci = 0
    vsh = pci % 6
    # PBCH: slot 1 symbols 0..3 -> symbols 7..10, central 72 subcarriers (here all)
    G[:, 7:11] = 5
    # SSS symbol 5, PSS symbol 6 of slot 0; 62 subcarriers centred (5 reserved each side)
    G[5:67, 5] = 4; G[5:67, 6] = 3
    G[0:5, 5:7] = 7; G[67:72, 5:7] = 7
    # CRS ports 0/1: symbols 0 and 4 of each slot; v = 0 / 3 (port 0 symbol 0 v=0, symbol 4 v=3)
    for l in (0, 4, 7, 11):
        v = 0 if l in (0, 7) else 3
        for k in range(nsc):
            if (k - (v + vsh)) % 6 == 0:
                G[k, l] = 2                  # port 0
            if (k - ((v + 3) % 6 + vsh)) % 6 == 0:
                G[k, l] = 6                  # port 1
    # PBCH avoids CRS positions of 4 ports in symbols 7,8 (ports 2,3 in symbol 8 of the subframe = slot1 sym1)
    for k in range(nsc):
        if (k - vsh) % 3 == 0 and G[k, 8] == 5:
            G[k, 8] = 7
    cols = ["#E8F6EF", "#AED6F1", NAVY, "#E74C3C", "#F5B041", "#A569BD", "#2E86C1", "#FFFFFF"]
    names = ["PDSCH", "control region (PDCCH)", "CRS port 0", "PSS", "SSS", "PBCH", "CRS port 1", "reserved / unused"]
    fig = plt.figure(figsize=(W2, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1], hspace=0.32)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1])
    # frame strip
    a0.set_xlim(0, 10); a0.set_ylim(0, 1); a0.axis("off")
    for i in range(10):
        c = "#F5B041" if i in (0, 5) else LIGHT
        a0.add_patch(Rectangle((i, 0), 1, 1, facecolor=c, edgecolor="white", lw=1.5))
        a0.text(i + 0.5, 0.5, f"SF{i}", ha="center", va="center", fontsize=7)
    a0.text(0, 1.15, "10 ms radio frame = 10 subframes of 1 ms (= 2 slots of 0.5 ms); PSS/SSS in SF0 and SF5, PBCH in SF0",
            fontsize=7, color=NAVY)
    a1.imshow(G, aspect="auto", origin="lower", cmap=ListedColormap(cols), vmin=-0.5, vmax=7.5,
              interpolation="nearest", extent=(-0.5, 13.5, -0.5, 71.5))
    for x in np.arange(-0.5, 14, 1):
        a1.axvline(x, color="white", lw=0.4)
    for y in np.arange(-0.5, 72, 12):
        a1.axhline(y, color="black", lw=0.5)
    a1.axvline(6.5, color="black", lw=1.2)
    a1.set_xticks(range(14)); a1.set_xticklabels([str(i % 7) for i in range(14)], fontsize=7)
    a1.set_yticks(np.arange(6, 72, 12)); a1.set_yticklabels([f"RB {i}" for i in range(6)], fontsize=7)
    a1.set_xlabel("OFDM symbol (slot 0 | slot 1)"); a1.grid(False)
    a1.set_title("Subframe 0, 1.4 MHz carrier (6 RBs x 12 subcarriers), 2 antenna ports, CFI = 3", fontsize=8.5)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(facecolor=c, edgecolor=GRAY, lw=0.4, label=nm) for c, nm in zip(cols, names)],
              fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    save(fig, "ch21_lte_grid")


# ============================================================================ 9
def nr_frame():
    """NR: SS/PBCH block, SSB burst in a half-frame (case C) and a TDD slot pattern."""
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[0.8, 1.6], hspace=0.6, wspace=0.28)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, :])
    # (a) SSB
    S = np.zeros((240, 4))
    S[:, 1] = 3; S[:, 3] = 3
    S[0:48, 2] = 3; S[192:240, 2] = 3
    S[56:183, 0] = 1; S[56:183, 2] = 2
    cm = ListedColormap(["#FFFFFF", "#E74C3C", "#F5B041", "#A569BD", "#C39BD3"])
    a.imshow(S, aspect="auto", origin="lower", cmap=cm, vmin=-0.5, vmax=4.5, interpolation="nearest",
             extent=(-0.5, 3.5, -0.5, 239.5))
    a.set_xticks(range(4)); a.set_xlabel("OFDM symbol"); a.set_ylabel("subcarrier (20 RBs = 240)")
    a.set_yticks([0, 56, 182, 239]); a.grid(False)
    a.text(0, 120, "PSS", rotation=90, ha="center", va="center", fontsize=7, color="white")
    a.text(2, 120, "SSS", rotation=90, ha="center", va="center", fontsize=7)
    a.text(1, 120, "PBCH + DMRS", rotation=90, ha="center", va="center", fontsize=7, color="white")
    a.set_title("(a) One SS/PBCH block", fontsize=9)
    # (b) SSB burst case C, L = 8, 30 kHz, half frame of 5 ms = 10 slots
    b.set_xlim(0, 5); b.set_ylim(0, 1.3); b.set_yticks([])
    b.set_xlabel("time within the 5 ms half-frame (ms)")
    slot_ms = 0.5
    for sidx in range(10):
        b.add_patch(Rectangle((sidx * slot_ms, 0.15), slot_ms, 0.6, facecolor=LIGHT, edgecolor="white"))
    starts = [s + 14 * n for n in range(4) for s in (2, 8)]
    sym = slot_ms / 14
    beam_cols = plt.cm.viridis(np.linspace(0.1, 0.9, 8))
    for i, s0 in enumerate(starts):
        b.add_patch(Rectangle((s0 * sym, 0.15), 4 * sym, 0.6, facecolor=beam_cols[i], edgecolor="none"))
        b.text((s0 + 2) * sym, 0.85, str(i), ha="center", fontsize=6.5)
    b.text(2.9, 0.45, "remaining slots carry\nordinary traffic", fontsize=7, ha="center", va="center")
    b.text(0.0, 1.08, "SSB index (= beam) 0-7; burst repeats every 20 ms by default", fontsize=7)
    b.set_title("(b) SSB burst, case C (30 kHz), L = 8 beams", fontsize=9)
    b.grid(False)
    # (c) TDD pattern DDDSU at 30 kHz plus a mini-slot
    c.set_xlim(0, 5); c.set_ylim(0, 1.25); c.set_yticks([]); c.grid(False)
    c.set_xlabel("time (ms), 30 kHz numerology: 0.5 ms slots of 14 symbols")
    pat = "DDDSUDDDSU"
    for i, ch in enumerate(pat):
        x0 = i * slot_ms
        if ch == "D":
            c.add_patch(Rectangle((x0, 0.2), slot_ms, 0.6, facecolor=NAVY, alpha=0.75, edgecolor="white"))
            c.text(x0 + slot_ms / 2, 0.5, "D", color="white", ha="center", va="center", fontsize=8)
        elif ch == "U":
            c.add_patch(Rectangle((x0, 0.2), slot_ms, 0.6, facecolor=ORANGE, alpha=0.85, edgecolor="white"))
            c.text(x0 + slot_ms / 2, 0.5, "U", color="white", ha="center", va="center", fontsize=8)
        else:
            for k in range(14):
                col = NAVY if k < 10 else ("#FFFFFF" if k < 12 else ORANGE)
                c.add_patch(Rectangle((x0 + k * sym, 0.2), sym, 0.6, facecolor=col,
                                      alpha=0.75 if col != "#FFFFFF" else 1, edgecolor="white", lw=0.3))
            c.text(x0 + slot_ms / 2, 0.88, "S = 10D:2G:2U", ha="center", fontsize=6.5)
    # mini-slot example: 2-symbol downlink transmission inside slot 1
    x0 = 1 * slot_ms + 6 * sym
    c.add_patch(Rectangle((x0, 0.2), 2 * sym, 0.6, facecolor=ACCENT, edgecolor="black", lw=0.8))
    c.annotate("2-symbol mini-slot\n(URLLC pre-emption)", (x0 + sym, 0.8), xytext=(1.25, 1.02), fontsize=6.5,
               ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
    c.text(4.0, 1.02, "2.5 ms period: 74% DL, 23% UL, 3% guard symbols", fontsize=6.8, ha="center")
    c.set_title("(c) A common mid-band TDD pattern, DDDSU", fontsize=9)
    save(fig, "ch21_nr_frame")


# ============================================================================ 10
def peak_rates():
    """Peak downlink rate of each generation's standard/device category versus year."""
    pts = [  # (year, rate b/s, label, family)
        (1992, 9.6e3, "GSM CSD", "GSM"), (1999.5, 57.6e3, "HSCSD", "GSM"), (2000.5, 85.6e3, "GPRS (4 slots)", "GSM"),
        (2003, 236.8e3, "EDGE (4 slots)", "GSM"), (1996, 14.4e3, "IS-95A", "CDMA"),
        (2000.8, 153.6e3, "CDMA2000 1x", "CDMA"), (2002.5, 2.4e6, "EV-DO Rev 0", "CDMA"),
        (2006.5, 3.1e6, "EV-DO Rev A", "CDMA"),
        (2002, 384e3, "UMTS R99", "UMTS"), (2006, 14.4e6, "HSDPA", "UMTS"), (2009, 21.1e6, "HSPA+", "UMTS"),
        (2010.5, 42.2e6, "DC-HSPA+", "UMTS"),
        (2010, 100e6, "LTE Cat 3", "LTE"), (2012, 150e6, "LTE Cat 4", "LTE"), (2013.7, 300e6, "LTE-A Cat 6", "LTE"),
        (2015, 450e6, "Cat 9", "LTE"), (2017, 1.0e9, "Gigabit LTE", "LTE"),
        (2019.3, 2.0e9, "NR (early)", "NR"), (2021.5, 4.0e9, "NR FR1+FR2", "NR"), (2024, 7.5e9, "NR 2024 modems", "NR"),
    ]
    fam = {"GSM": NAVY, "CDMA": PURPLE, "UMTS": GREEN, "LTE": ORANGE, "NR": ACCENT}
    fig, ax = plt.subplots(figsize=(W2, 3.1))
    for f_, c in fam.items():
        xs = [p[0] for p in pts if p[3] == f_]; ys = [p[1] for p in pts if p[3] == f_]
        ax.semilogy(xs, ys, "o-", color=c, ms=4, lw=1, label=f_)
    offs = {"GPRS (4 slots)": (4, -10), "HSCSD": (-30, 6), "IS-95A": (4, -4), "CDMA2000 1x": (-50, 2),
            "UMTS R99": (-8, 8), "EDGE (4 slots)": (5, -4), "EV-DO Rev 0": (-50, 5), "GSM CSD": (5, -4),
            "LTE Cat 3": (-34, 6), "LTE Cat 4": (5, -6), "LTE-A Cat 6": (5, -7), "Cat 9": (-22, 5),
            "Gigabit LTE": (5, -6), "HSDPA": (-36, 3), "HSPA+": (3, -9), "DC-HSPA+": (4, -8),
            "NR (early)": (5, -7), "NR FR1+FR2": (5, -7), "NR 2024 modems": (-40, 6)}
    for x, y, lab, f_ in pts:
        dx, dy = offs.get(lab, (4, -9))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6, color=fam[f_])
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    pf = np.polyfit(xs, np.log10(ys), 1)
    xx = np.array([1991, 2025])
    ax.semilogy(xx, 10 ** np.polyval(pf, xx), ":", color=GRAY, lw=1)
    ax.text(2016.5, 2e5, f"dotted trend: x10 every {1 / pf[0]:.1f} years", fontsize=7, color=GRAY)
    ax.set_xlim(1990, 2026); ax.set_ylim(5e3, 3e10)
    ax.set_xlabel("year of first commercial availability (approximate)"); ax.set_ylabel("peak downlink rate (b/s)")
    ax.legend(fontsize=7, loc="upper left", ncol=5)
    fig.tight_layout(); save(fig, "ch21_peak_rates")
    print("decade time:", 1 / pf[0])


# ============================================================================ 11
def imt_capabilities():
    """IMT-2020 vs IMT-Advanced key capabilities (ITU-R M.2083), as improvement factors."""
    rows = [("Peak data rate", "1 Gb/s", "20 Gb/s", 20),
            ("User-experienced rate", "10 Mb/s", "100 Mb/s", 10),
            ("Spectrum efficiency", "1x", "3x", 3),
            ("Mobility", "350 km/h", "500 km/h", 500 / 350),
            ("Latency (air interface)", "10 ms", "1 ms", 10),
            ("Connection density", "10$^5$ /km$^2$", "10$^6$ /km$^2$", 10),
            ("Network energy efficiency", "1x", "100x", 100),
            ("Area traffic capacity", "0.1 Mb/s/m$^2$", "10 Mb/s/m$^2$", 100)]
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (nm, a, b, f) in zip(y, rows):
        ax.barh(yi, np.log10(f), color=ACCENT, alpha=0.85, height=0.55)
        ax.text(np.log10(f) + 0.04, yi, f"x{f:g}".replace("x1.42857", "x1.4"), va="center", fontsize=7.5)
        ax.text(-0.05, yi, f"{nm}", ha="right", va="center", fontsize=7.8)
        ax.text(2.55, yi, f"{a}  $\\rightarrow$  {b}", va="center", fontsize=7.2, color=NAVY)
    ax.set_xlim(0, 3.75); ax.set_yticks([])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["x1", "x10", "x100"])
    ax.text(2.55, len(rows) - 0.3, "IMT-Advanced $\\rightarrow$ IMT-2020", fontsize=7.5, fontweight="bold", color=NAVY)
    ax.spines["left"].set_visible(False); ax.grid(axis="y", alpha=0)
    ax.set_xlabel("improvement factor (log scale)")
    fig.tight_layout(); fig.subplots_adjust(left=0.28); save(fig, "ch21_imt_capabilities")


# ============================================================================ 12
def latency():
    """(a) Typical round-trip times by technology (approximate), (b) an air-interface budget."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    tech = [("GPRS", 500, 1000, NAVY), ("EDGE", 150, 500, NAVY), ("UMTS R99", 100, 200, GREEN),
            ("HSPA", 30, 100, GREEN), ("LTE", 20, 50, ORANGE), ("NR (eMBB)", 8, 25, ACCENT),
            ("NR URLLC\n(air, target)", 0.5, 2, ACCENT)]
    for i, (n, lo, hi, c) in enumerate(tech):
        ax[0].plot([lo, hi], [i, i], color=c, lw=6, solid_capstyle="butt", alpha=0.85)
    ax[0].set_xscale("log"); ax[0].set_yticks(range(len(tech)))
    ax[0].set_yticklabels([t[0] for t in tech], fontsize=7.5); ax[0].invert_yaxis()
    ax[0].set_xlabel("round-trip time (ms)"); ax[0].set_xlim(0.3, 1500)
    ax[0].set_title("(a) Typical RTT, approximate", fontsize=9)
    # (b) one-way DL user-plane budget (ms): frame alignment, TTI, tx processing, rx processing, HARQ (10% x RTT)
    comps = ["alignment", "TTI", "BS processing", "UE processing", "HARQ (10% x RTT)"]
    cases = {"LTE FDD\n1 ms TTI": [0.5, 1.0, 1.0, 1.5, 0.8],
             "NR 30 kHz\n0.5 ms slot": [0.25, 0.5, 0.5, 0.5, 0.2],
             "NR 120 kHz\n2-sym mini-slot": [0.009, 0.018, 0.1, 0.15, 0.05]}
    colsb = [GRAY, NAVY, GREEN, ORANGE, ACCENT]
    left = np.zeros(3)
    names = list(cases)
    vals = np.array([cases[k] for k in names])
    for j, cn in enumerate(comps):
        ax[1].barh(range(3), vals[:, j], left=left, color=colsb[j], height=0.55, label=cn)
        left += vals[:, j]
    for i in range(3):
        ax[1].text(left[i] + 0.06, i, f"{left[i]:.2f} ms", va="center", fontsize=7.5)
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels(names, fontsize=7.2); ax[1].invert_yaxis()
    ax[1].set_xlim(0, 6.2); ax[1].set_xlabel("one-way user-plane latency (ms)")
    ax[1].legend(fontsize=6.3, loc="lower right", frameon=True)
    ax[1].set_title("(b) Downlink air-interface budget", fontsize=9)
    fig.tight_layout(w_pad=0.8); save(fig, "ch21_latency")
    print("budgets", dict(zip(names, left)))


if __name__ == "__main__":
    import sys as _s
    fns = [growth, spectrum_bands, gsm_frames, gmsk, power_control, soft_handover, cqi_harq,
           lte_grid, nr_frame, peak_rates, imt_capabilities, latency]
    sel = _s.argv[1:]
    for fn in fns:
        if not sel or fn.__name__ in sel:
            fn()
