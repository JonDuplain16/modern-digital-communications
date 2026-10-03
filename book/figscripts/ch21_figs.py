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




# ############################################################################
# Second edition: concept illustrations, infographics and small simulations
# ############################################################################
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch, Wedge

SOFT = {"1G": "#E5E7E9", "2G": "#D6E4F0", "3G": "#D4EFDF", "4G": "#FAE5D3", "5G": "#F5D5D0"}


def _box(ax, x, y, w, h, txt, fc, ec=None, fs=7, color="#222222", weight="normal", r=0.04, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=0.8))
    tx = x + w / 2 if ha == "center" else x + 0.03
    ax.text(tx, y + h / 2, txt, ha=ha, va="center", fontsize=fs, color=color, weight=weight,
            linespacing=1.15)


def _arrow(ax, p, q, color=GRAY, lw=1.0, style="-|>", ms=8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def _hexagon(ax, cx, cy, R, fc, ec="white", lw=0.8, alpha=1.0, z=1):
    ang = np.deg2rad(np.arange(6) * 60 + 30)
    ax.add_patch(Polygon(np.c_[cx + R * np.cos(ang), cy + R * np.sin(ang)], closed=True, fc=fc,
                         ec=ec, lw=lw, alpha=alpha, zorder=z))


def _hexcenter(q, r_, R=1.0):
    return np.sqrt(3) * R * (q + r_ / 2), 1.5 * R * r_


def _erlang_b(A, C):
    b = 1.0
    for k in range(1, C + 1):
        b = A * b / (k + A * b)
    return b


def _erlang_capacity(C, B=0.02):
    lo, hi = 0.0, 2.0 * C + 10
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if _erlang_b(m, C) > B:
            hi = m
        else:
            lo = m
    return lo


# ---------------------------------------------------------------------------- by the numbers
def by_numbers():
    tiles = [("8.8 billion", "mobile subscriptions\nworldwide, 2025 (approx.)"),
             ("12", "channels for New York's\nmobile phones in 1976"),
             ("270.833 kb/s", "GSM bit rate:\n13 MHz / 48"),
             ("160", "characters in an SMS:\n140 octets of 7-bit text"),
             ("1500 Hz", "WCDMA power-control\ncommands per second"),
             ("1 ms", "LTE subframe: the\nscheduler's heartbeat"),
             ("2.34 Gb/s", "NR peak, 100 MHz, 4 layers,\n256QAM (TS 38.306)"),
             ("x10 / 5 yr", "growth of peak rates,\n1990s to 2020s")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, GRAY, NAVY, PURPLE, GREEN, ORANGE, ACCENT, BLUE2]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.63, big, ha="center", va="center", fontsize=12.5, color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=6.9, color="#333333", linespacing=1.1)
    save(fig, "ch21_by_numbers")


# ---------------------------------------------------------------------------- timeline
def timeline():
    ev = [(1946, "MTS, St. Louis", "0G", 1), (1965, "IMTS", "0G", -1), (1973, "Cooper's call", "1G", 2),
          (1979, "NTT Tokyo", "1G", -2), (1981, "NMT", "1G", 1), (1983, "AMPS Chicago", "1G", -3),
          (1987, "GSM MoU", "2G", 3), (1991, "first GSM call", "2G", -1), (1992, "first SMS", "2G", 2),
          (1995, "IS-95 CDMA", "2G", -2), (1999, "i-mode", "2G", 1), (2001, "FOMA (3G)", "3G", -3),
          (2005, "HSDPA", "3G", 2), (2007, "iPhone", "3G", -1), (2009, "first LTE", "4G", 3),
          (2012, "VoLTE", "4G", -2), (2017, "Gigabit LTE", "4G", 1), (2019, "first 5G", "5G", -3),
          (2024, "5G-Advanced", "5G", 2), (2030, "6G?", "6G", -1)]
    col = {"0G": GRAY, "1G": "#566573", "2G": NAVY, "3G": GREEN, "4G": ORANGE, "5G": ACCENT, "6G": PURPLE}
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    spans = [("0G", 1946, 1979), ("1G", 1979, 1991), ("2G", 1991, 2001), ("3G", 2001, 2009),
             ("4G", 2009, 2019), ("5G", 2019, 2030)]
    for g, a, b in spans:
        ax.add_patch(Rectangle((a, -0.12), b - a, 0.24, fc=col[g], alpha=0.85, lw=0, zorder=2))
        ax.text((a + b) / 2, 0, g, ha="center", va="center", fontsize=7.5, color="white", weight="bold", zorder=3)
    for yr, nm, c, lev in ev:
        up = 1 if lev > 0 else -1
        h = up * (0.42 + 0.48 * (abs(lev) - 1))
        ax.plot([yr, yr], [0.12 * up, h], color=col[c], lw=0.7)
        ax.text(yr, h + 0.05 * up, f"{nm}\n{yr}" if up > 0 else f"{yr}\n{nm}", ha="center",
                va="bottom" if up > 0 else "top", fontsize=6.0, color=col[c], linespacing=1.0)
    ax.set_xlim(1942, 2034); ax.set_ylim(-2.05, 1.85); ax.axis("off")
    save(fig, "ch21_timeline")


# ---------------------------------------------------------------------------- five road systems
def _car(ax, x, y, w, h, c, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=c, ec="none",
                                alpha=alpha, zorder=4))


def roads():
    fig, axs = plt.subplots(1, 5, figsize=(W1, 2.35))
    r = rng(11)
    titles = ["1G: one lane\nper car (FDMA)", "2G: traffic lights\n(TDMA)", "3G: shared motorway,\ncode lanes (CDMA)",
              "4G: all-IP highway\ngrid (OFDMA)", "5G: express lanes\nand slices"]
    for k, ax in enumerate(axs):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.25); ax.axis("off")
        ax.set_title(titles[k], fontsize=7.2, color=list(GEN_COL.values())[k] if k else "#566573", pad=2)
    # 1G: separate thin roads, one car each, many empty
    a = axs[0]
    for i in range(7):
        y = 0.08 + i * 0.155
        a.add_patch(Rectangle((0.05, y), 0.9, 0.1, fc="#D5D8DC", lw=0))
        if i in (1, 4, 5):
            _car(a, 0.15 + 0.5 * r.random(), y + 0.015, 0.2, 0.07, GRAY)
    # 2G: one wide road split in 8 time segments with lights
    a = axs[1]
    a.add_patch(Rectangle((0.05, 0.2), 0.9, 0.5, fc="#D5D8DC", lw=0))
    cols8 = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY, "#117A65"]
    for i in range(8):
        _car(a, 0.07 + i * 0.11, 0.38, 0.09, 0.14, cols8[i])
    for i in range(3):
        x = 0.2 + i * 0.3
        a.add_patch(Rectangle((x, 0.78), 0.07, 0.25, fc="#333333", lw=0))
        for j, c in enumerate([ACCENT, ORANGE, GREEN]):
            a.add_patch(Circle((x + 0.035, 0.82 + j * 0.075), 0.026, fc=c if j == (i % 3) else "#777777", lw=0))
    a.text(0.5, 0.1, "8 users take turns\non one 200 kHz road", ha="center", fontsize=6.2)
    # 3G: wide motorway, overlapping translucent cars (codes)
    a = axs[2]
    a.add_patch(Rectangle((0.05, 0.12), 0.9, 0.95, fc="#D5D8DC", lw=0))
    for i in range(14):
        _car(a, 0.08 + 0.7 * r.random(), 0.17 + 0.78 * r.random(), 0.2, 0.09, cols8[i % 8], alpha=0.45)
    a.text(0.5, 0.03, "everyone at once,\ntold apart by code", ha="center", fontsize=6.2, va="bottom")
    # 4G: grid of blocks
    a = axs[3]
    for i in range(6):
        for j in range(6):
            a.add_patch(Rectangle((0.06 + i * 0.148, 0.12 + j * 0.155), 0.13, 0.135, fc="#D5D8DC", lw=0))
            if r.random() < 0.55:
                _car(a, 0.075 + i * 0.148, 0.145 + j * 0.155, 0.10, 0.085, cols8[int(r.integers(0, 6))])
    a.text(0.5, 0.03, "scheduler fills a\ntime-frequency grid", ha="center", fontsize=6.2, va="bottom")
    # 5G: lanes of different widths (numerologies) + slice colours
    a = axs[4]
    lanes = [(0.12, 0.33, "#F5D5D0", "eMBB"), (0.47, 0.14, "#D4EFDF", "URLLC"), (0.63, 0.1, "#E8DAEF", "IoT"),
             (0.75, 0.3, "#D6E4F0", "private")]
    for y, h, c, nm in lanes:
        a.add_patch(Rectangle((0.05, y), 0.9, h, fc=c, lw=0))
        a.text(0.92, y + h / 2, nm, ha="right", va="center", fontsize=5.8, color="#333333")
    for i in range(4):
        _car(a, 0.08 + i * 0.16, 0.2, 0.12, 0.17, ACCENT)
    _car(a, 0.1, 0.5, 0.3, 0.08, GREEN)
    for i in range(6):
        _car(a, 0.08 + i * 0.1, 0.66, 0.05, 0.04, PURPLE)
    _car(a, 0.12, 0.85, 0.22, 0.12, NAVY)
    a.text(0.5, 0.03, "lanes sized and\nreserved per service", ha="center", fontsize=6.2, va="bottom")
    fig.tight_layout(w_pad=0.3); save(fig, "ch21_roads")


# ---------------------------------------------------------------------------- what each generation fixed
def what_fixed():
    gens = ["1G", "2G", "3G", "4G", "5G"]
    fixed = ["mobility itself:\ncells, reuse,\nhandoff",
             "capacity, privacy,\ncloning, roaming:\ndigital + SIM",
             "data, wideband\naccess: CDMA +\npacket core",
             "slow, deep\nnetworks: flat\nall-IP + OFDMA",
             "one-size LTE:\nnumerology, beams,\ncloud core"]
    left = ["capacity,\neavesdropping,\ncloning", "circuit-only;\ndata a retrofit",
            "RNC far from\nradio; sluggish\npacket data", "fixed numerology,\nalways-on CRS,\nmonolithic core",
            "cost per bit,\nenergy, sensing...\n(6G)"]
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    ax.set_xlim(0, 5); ax.set_ylim(0, 2.45); ax.axis("off")
    for i, g in enumerate(gens):
        c = list(GEN_COL.values())[i]
        _box(ax, i + 0.08, 1.95, 0.84, 0.38, g, c, fs=10, color="white", weight="bold")
        _box(ax, i + 0.08, 1.02, 0.84, 0.82, "fixed:\n" + fixed[i], SOFT[g], fs=6.6)
        _box(ax, i + 0.08, 0.06, 0.84, 0.8, "left behind:\n" + left[i], "#F2F3F4", fs=6.4, color="#555555")
        if i < 4:
            _arrow(ax, (i + 0.86, 0.5), (i + 1.12, 1.35), color=ACCENT, lw=1.2, rad=-0.25)
    save(fig, "ch21_what_fixed")


# ---------------------------------------------------------------------------- one tower versus cells
def imts_vs_cells():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
    for a in axs:
        a.set_aspect("equal"); a.axis("off"); a.set_xlim(-6.2, 6.2); a.set_ylim(-6.6, 5.9)
    a = axs[0]
    a.add_patch(Circle((0, 0), 5.3, fc="#D6E4F0", ec=NAVY, lw=1.0))
    a.plot([0, 0], [0, 1.6], color="#333333", lw=2); a.plot([-0.35, 0, 0.35], [0, 1.6, 0], color="#333333", lw=1)
    a.text(0, 2.0, "one high-power\ntransmitter", ha="center", fontsize=7)
    a.text(0, -2.6, "~12 channels for the\nwhole city: 12 calls", ha="center", fontsize=7.5, color=NAVY, weight="bold")
    a.set_title("(a) IMTS: one big cell", fontsize=8.5)
    a = axs[1]
    cols7 = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY]
    n = 0
    for q in range(-4, 5):
        for r_ in range(-4, 5):
            x, y = _hexcenter(q, r_, 1.0)
            if x * x + y * y < 5.3 ** 2:
                _hexagon(a, x, y, 1.0, cols7[(q + 3 * r_) % 7], alpha=0.35)
                n += 1
    a.add_patch(Circle((0, 0), 5.3, fc="none", ec=NAVY, lw=1.0, ls="--"))
    a.set_title(f"(b) cellular: {n} small cells, reuse N = 7", fontsize=8.5)
    a.text(0, -6.3, f"each channel used ~{n / 7:.0f} times: ~{n / 7:.0f}x the calls", ha="center", fontsize=7.2,
           color=NAVY, weight="bold")
    fig.tight_layout(w_pad=0.5); save(fig, "ch21_imts_vs_cells")
    print("cells in (b):", n)


# ---------------------------------------------------------------------------- hexagonal reuse
def hex_reuse():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = axs[0]; a.set_aspect("equal"); a.axis("off")
    cols7 = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY]
    for q in range(-5, 6):
        for r_ in range(-5, 6):
            x, y = _hexcenter(q, r_)
            if x * x + y * y < 4.9 ** 2:
                g = (q + 3 * r_) % 7
                _hexagon(a, x, y, 1.0, cols7[g], alpha=0.85 if g == 0 else 0.22)
                a.text(x, y, chr(65 + g), ha="center", va="center", fontsize=5.5,
                       color="white" if g == 0 else "#333333")
    x1, y1 = _hexcenter(1, 2)
    a.annotate("", (x1, y1), (0, 0), arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    a.text(x1 / 2 + 0.25, y1 / 2 - 0.35, r"$D=R\sqrt{3N}$", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.5))
    a.set_xlim(-5, 5); a.set_ylim(-5, 5)
    a.set_title("(a) N = 7: co-channel cells (A)", fontsize=8.5)
    a = axs[1]
    Ns = np.array([1, 3, 4, 7, 9, 12, 13, 19])
    for nexp, c in [(3.0, GRAY), (3.5, BLUE2), (4.0, NAVY)]:
        sir = 10 * np.log10((3 * Ns) ** (nexp / 2) / 6)
        a.plot(Ns, sir, "o-", color=c, ms=3.5, label=f"n = {nexp}")
    a.axhline(18, color=ACCENT, ls="--", lw=0.8); a.text(1.0, 19.2, "analog FM (AMPS) ~18 dB", fontsize=6.5,
                                                        color=ACCENT, ha="left")
    a.axhline(9, color=GREEN, ls="--", lw=0.8); a.text(19.3, 7.0, "GSM ~9 dB", fontsize=6.5, color=GREEN, ha="right")
    a.set_xlabel("cluster size N"); a.set_ylabel("cell-edge SIR (dB)")
    a.set_xticks(Ns); a.legend(fontsize=6.5, loc="lower right"); a.set_ylim(-10, 30)
    a.set_title(r"(b) SIR $\approx (3N)^{n/2}/6$", fontsize=8.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch21_hex_reuse")


# ---------------------------------------------------------------------------- AMPS baseband and RF
def amps_baseband():
    r = rng(5)
    fs = 480e3
    N = int(fs * 4)
    sos = sps.butter(6, [300, 3000], btype="band", fs=fs, output="sos")
    v = sps.sosfilt(sos, r.standard_normal(N))
    v = v / (3 * v.std()); v = np.clip(v, -1, 1)
    t = np.arange(N) / fs
    sat = np.cos(2 * np.pi * 6000 * t)
    m = 12e3 * 0.75 * v + 2e3 * sat           # peak deviation about 12 kHz, SAT +-2 kHz
    x = np.exp(1j * 2 * np.pi * np.cumsum(m) / fs)
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.45))
    a = axs[0]
    f = np.linspace(0, 12, 1200)
    voice = np.where((f > 0.3) & (f < 3.0), -10 - 4 * (f - 0.3), -70)
    voice = np.convolve(np.r_[np.full(20, -70.0), voice, np.full(20, -70.0)], np.ones(15) / 15, "same")[20:-20]
    a.fill_between(f, -70, voice, color=BLUE2, alpha=0.4, lw=0)
    a.plot(f, voice, color=NAVY, lw=0.9)
    for ft, nm, c, h in [(6.0, "SAT\n5970/6000/6030 Hz", GREEN, -12), (10.0, "ST\n10 kHz", ORANGE, -14)]:
        a.plot([ft, ft], [-70, h], color=c, lw=2); a.text(ft, h + 3, nm, ha="center", fontsize=6.5, color=c)
    a.text(1.6, -6, "voice\n0.3-3 kHz", ha="center", fontsize=6.8, color=NAVY)
    a.set_xlim(0, 12); a.set_ylim(-70, 12); a.set_xlabel("audio frequency (kHz)"); a.set_ylabel("level (dB)")
    a.set_title("(a) what the FM modulator sees", fontsize=8.5)
    a = axs[1]
    ff, p = sps.welch(x, fs=fs, nperseg=8192, return_onesided=False)
    i = np.argsort(ff); ff, p = ff[i] / 1e3, 10 * np.log10(p[i] / p.max())
    for off, c, lab in [(-30, GRAY, "neighbour"), (0, NAVY, "this channel"), (30, GRAY, None)]:
        a.plot(ff + off, p - (0 if off == 0 else 3), color=c, lw=1.0 if off == 0 else 0.7,
               ls="-" if off == 0 else ":", label=lab)
    for e in (-15, 15):
        a.axvline(e, color=ACCENT, lw=0.7, ls="--")
    a.text(0, 3, "30 kHz", ha="center", fontsize=7, color=ACCENT)
    a.set_xlim(-60, 60); a.set_ylim(-60, 8); a.set_xlabel("offset from carrier (kHz)")
    a.set_ylabel("PSD (dB)"); a.legend(fontsize=6.3, loc="lower right")
    a.set_title("(b) simulated FM spectrum, 12 kHz dev.", fontsize=8.5)
    fig.tight_layout(w_pad=0.8); save(fig, "ch21_amps_baseband")


# ---------------------------------------------------------------------------- trunking
def trunking():
    Cs = np.arange(1, 401)
    A = np.array([_erlang_capacity(int(c)) for c in Cs])
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    ax.semilogx(Cs, 100 * A / Cs, color=NAVY)
    a19, a395 = _erlang_capacity(19), _erlang_capacity(395)
    ax.plot(19, 100 * a19 / 19, "o", color=ACCENT, ms=5)
    ax.annotate(f"AMPS sector: 19 ch.\n{a19:.1f} E ({100 * a19 / 19:.0f}% busy)", (19, 100 * a19 / 19),
                xytext=(1.6, 80), fontsize=6.6, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    ax.plot(395, 100 * a395 / 395, "o", color=GREEN, ms=5)
    ax.annotate(f"one pool: 395 ch.\n{a395:.0f} E ({100 * a395 / 395:.0f}%)", (395, 100 * a395 / 395),
                xytext=(25, 40), fontsize=6.6, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))
    ax.set_xlabel("channels in the group"); ax.set_ylabel("utilisation at 2% blocking (%)")
    ax.set_ylim(0, 100); ax.set_xlim(1, 400)
    fig.tight_layout(); save(fig, "ch21_trunking")
    print("Erlang 19:", a19, " 21x19:", 21 * a19, " 395:", a395)


# ---------------------------------------------------------------------------- identity vs authentication
def _phone(ax, x, y, s=1.0, c=NAVY):
    ax.add_patch(FancyBboxPatch((x - 0.12 * s, y - 0.25 * s), 0.24 * s, 0.5 * s,
                                boxstyle="round,pad=0,rounding_size=0.04", fc=c, ec="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.09 * s, y - 0.08 * s), 0.18 * s, 0.26 * s, fc="white", alpha=0.85, zorder=4))


def cloning():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.35))
    for a in axs:
        a.set_xlim(0, 4); a.set_ylim(0, 2.6); a.axis("off")
    a = axs[0]
    a.set_title("(a) AMPS: the phone shouts its identity", fontsize=8.5, color=ACCENT)
    _phone(a, 0.5, 1.5); a.text(0.5, 0.95, "your phone", ha="center", fontsize=6.5)
    a.plot([3.4, 3.4], [1.0, 2.2], color="#333333", lw=2); a.text(3.4, 0.75, "base station", ha="center", fontsize=6.5)
    _box(a, 1.0, 1.7, 2.0, 0.5, "MIN 312-555-0100\nESN 8A3F...  (in the clear)", "#FADBD8", fs=6.4)
    _arrow(a, (0.75, 1.55), (3.3, 1.55), color=NAVY)
    _phone(a, 2.1, 0.62, 0.8, GRAY); a.text(2.1, 0.3, "eavesdropper\ncopies it", fontsize=6.3, color=GRAY, ha="center", va="top")
    _arrow(a, (2.1, 1.5), (2.1, 0.88), color=ACCENT, ls="--")
    _phone(a, 3.1, 0.62, 0.8, ACCENT); a.text(3.1, 0.3, "clone: same MIN/ESN,\nyour bill", fontsize=6.3, color=ACCENT, ha="center", va="top")
    _arrow(a, (2.3, 0.62), (2.9, 0.62), color=ACCENT)
    a = axs[1]
    a.set_title("(b) GSM: answer a fresh challenge", fontsize=8.5, color=GREEN)
    _phone(a, 0.5, 1.5); a.text(0.5, 0.95, "phone + SIM\n(holds secret $K_i$)", ha="center", fontsize=6.2, va="top")
    a.plot([3.4, 3.4], [1.0, 2.2], color="#333333", lw=2); a.text(3.4, 0.75, "network\n(also knows $K_i$)",
                                                                  ha="center", fontsize=6.2, va="top")
    _arrow(a, (3.3, 2.0), (0.75, 2.0), color=NAVY); a.text(2.0, 2.1, "1. RAND (random, 128 bits)", ha="center", fontsize=6.4)
    _arrow(a, (0.75, 1.45), (3.3, 1.45), color=GREEN)
    a.text(2.0, 1.55, "2. SRES = A3($K_i$, RAND)", ha="center", fontsize=6.4)
    _box(a, 1.0, 0.25, 2.1, 0.5, "a recorded SRES is useless:\nnext time RAND is different", "#D4EFDF", fs=6.4)
    fig.tight_layout(w_pad=0.6); save(fig, "ch21_cloning")


# ---------------------------------------------------------------------------- TDMA: 8 users take turns
def tdma_lights():
    fig, ax = plt.subplots(figsize=(W1, 1.85))
    cols8 = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY, "#117A65"]
    for fr in range(3):
        for s in range(8):
            x = fr * 8 + s
            ax.add_patch(Rectangle((x, 1.2), 0.94, 0.55, fc=cols8[s], alpha=0.9 if s == 2 else 0.3, lw=0))
            ax.add_patch(Rectangle((x + 3, 0.2), 0.94, 0.55, fc=cols8[s], alpha=0.9 if s == 2 else 0.3, lw=0))
            ax.text(x + 0.47, 1.47, str(s), ha="center", va="center", fontsize=6, color="white" if s == 2 else "#333")
            ax.text(x + 3.47, 0.47, str(s), ha="center", va="center", fontsize=6, color="white" if s == 2 else "#333")
    ax.text(-0.3, 1.47, "downlink\n(BTS sends)", ha="right", va="center", fontsize=7)
    ax.text(-0.3, 0.47, "uplink\n(phone sends)", ha="right", va="center", fontsize=7)
    for fr in range(3):
        ax.annotate("", (2.5 + fr * 8 + 3.0, 0.8), (2.5 + fr * 8, 1.18),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8))
    ax.text(5.6, 0.92, "3 slots later", fontsize=6.5, color=ACCENT)
    for fr in range(4):
        ax.axvline(fr * 8, color="#333", lw=0.6, ymin=0.55, ymax=0.95)
    ax.text(4, 1.95, "TDMA frame 4.615 ms", ha="center", fontsize=7, color=NAVY)
    ax.set_xlim(-4.2, 27.2); ax.set_ylim(0, 2.2); ax.axis("off")
    save(fig, "ch21_tdma_lights")


# ---------------------------------------------------------------------------- midamble
TSC0 = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1])


def midamble():
    s = 1 - 2 * TSC0.astype(float)
    core = s[5:21]
    lags = np.arange(-5, 6)
    ac = [np.dot(s[5 + l:21 + l], core) for l in lags]
    r = rng(8)
    h = np.array([0.0, 0.9, 0.0, -0.45 + 0.3j, 0.0, 0.25j])
    h = np.r_[h, np.zeros(0)]
    y = np.convolve(s, h)[:26] + 0.15 * (r.standard_normal(26) + 1j * r.standard_normal(26))
    est = np.array([np.dot(y[5 + l:21 + l], core) / 16 for l in range(0, 6)])
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.3))
    a = axs[0]
    a.stem(lags, ac, basefmt=" ", linefmt=NAVY, markerfmt="o")
    a.set_xlabel("lag (bits)"); a.set_ylabel("correlation")
    a.set_title("(a) TSC 0: ideal within $\\pm$5 bits", fontsize=8.5); a.set_ylim(-3, 18)
    a.text(0.5, 16.3, "16", fontsize=7)
    a = axs[1]
    k = np.arange(6)
    a.bar(k - 0.18, np.abs(h), width=0.34, color=GRAY, label="true channel")
    a.bar(k + 0.18, np.abs(est), width=0.34, color=NAVY, label="estimate from midamble")
    a.set_xlabel("delay (bits, 3.69 $\\mu$s each)"); a.set_ylabel("|tap|")
    a.set_title("(b) one correlation = channel estimate", fontsize=8.5); a.legend(fontsize=6.5)
    fig.tight_layout(w_pad=0.8); save(fig, "ch21_midamble")
    print("TSC0 autocorr", ac)


# ---------------------------------------------------------------------------- timing advance
def timing_advance():
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    Tb = 48 / 13
    slot = 156.25 * Tb
    rows = [("BTS receive window", 0, None), ("near phone (1 km), no TA", 2 * 1e3 / 3e8 * 1e6, None),
            ("far phone (30 km), no TA", 2 * 30e3 / 3e8 * 1e6, None),
            ("far phone (30 km), TA = 54 bits", 2 * 30e3 / 3e8 * 1e6 - 54 * Tb, None)]
    for i, (nm, d, _) in enumerate(rows):
        y = 3 - i
        for s in range(3):
            x0 = s * slot + (d if i else 0)
            c = NAVY if s == 1 else "#D5D8DC"
            if i == 0:
                c = "#D6E4F0" if s != 1 else BLUE2
            ax.add_patch(Rectangle((x0, y + 0.15), (slot - 8.25 * Tb) if i else slot, 0.6,
                                   fc=c if i == 0 or s == 1 else "none", ec=GRAY if s != 1 else "none", lw=0.6,
                                   alpha=0.9))
        ax.text(-15, y + 0.45, nm, ha="right", va="center", fontsize=6.8)
    ax.axvline(2 * slot, color=ACCENT, lw=0.8, ls="--"); ax.text(2 * slot + 8, 3.95, "next user's slot starts",
                                                                 fontsize=6.5, color=ACCENT)
    ax.text(3 * slot + 230, 1.45, "overlaps the next\nslot by ~170 $\\mu$s!", fontsize=6.5, color=ACCENT, va="center")
    ax.set_xlim(-560, 3 * slot + 560); ax.set_ylim(-0.1, 4.2); ax.axis("off")
    ax.text(slot * 1.5, -0.05, r"one slot = 577 $\mu$s; TA step = 1 bit = 3.69 $\mu$s round trip = 553 m",
            ha="center", fontsize=6.8, color=NAVY)
    save(fig, "ch21_timing_advance")


# ---------------------------------------------------------------------------- call set-up message chart
def call_flow():
    msgs = [("MS", "BSS", "Channel request (access burst)", "RACH"),
            ("BSS", "MS", "Immediate assignment: SDCCH + timing advance", "AGCH"),
            ("MS", "MSC", "CM service request (TMSI)", "SDCCH"),
            ("MSC", "MS", "Authentication request (RAND)", "SDCCH"),
            ("MS", "MSC", "Authentication response (SRES)", "SDCCH"),
            ("MSC", "MS", "Ciphering mode command: start A5", "SDCCH"),
            ("MS", "MSC", "Setup (called number)", "SDCCH"),
            ("MSC", "MS", "Call proceeding", "SDCCH"),
            ("BSS", "MS", "Assignment command: go to TCH", "SDCCH"),
            ("MS", "BSS", "Assignment complete", "FACCH"),
            ("MSC", "MS", "Alerting ... Connect", "FACCH"),
            ("MS", "MSC", "conversation (TCH) + measurements every 480 ms", "TCH+SACCH")]
    xs = {"MS": 0.6, "BSS": 2.6, "MSC": 4.6}
    fig, ax = plt.subplots(figsize=(W1, 3.5))
    n = len(msgs)
    for k, x in xs.items():
        _box(ax, x - 0.45, n + 0.25, 0.9, 0.45, {"MS": "phone (MS)", "BSS": "BTS + BSC", "MSC": "MSC/VLR"}[k],
             NAVY, fs=7, color="white", weight="bold")
        ax.plot([x, x], [0.2, n + 0.25], color=GRAY, lw=0.8, ls=":")
    for i, (a, b, txt, ch) in enumerate(msgs):
        y = n - 0.35 - i
        c = ACCENT if "uthent" in txt or "iphering" in txt else (GREEN if "conversation" in txt else NAVY)
        _arrow(ax, (xs[a], y), (xs[b], y), color=c, lw=0.9, ms=7)
        ax.text((xs[a] + xs[b]) / 2, y + 0.07, txt, ha="center", va="bottom", fontsize=6.0, color=c)
        ax.text(5.3, y, ch, ha="left", va="center", fontsize=6.0, color=GRAY)
    ax.text(5.3, n + 0.47, "logical\nchannel", fontsize=6.3, color=GRAY, va="center")
    ax.set_xlim(0, 6.0); ax.set_ylim(0, n + 0.8); ax.axis("off")
    save(fig, "ch21_call_flow")


# ---------------------------------------------------------------------------- speech bits and interleaving
def speech_bits():
    fig, axs = plt.subplots(2, 1, figsize=(W1, 2.9), gridspec_kw=dict(height_ratios=[1, 1.15]))
    a = axs[0]
    segs1 = [(50, ACCENT, "50 class 1a"), (132, ORANGE, "132 class 1b"), (78, GRAY, "78 class 2")]
    x = 0
    for n, c, nm in segs1:
        a.add_patch(Rectangle((x, 1.2), n, 0.6, fc=c, alpha=0.8, lw=0)); a.text(x + n / 2, 1.5, nm, ha="center",
                                                                               va="center", fontsize=6.5, color="white")
        x += n
    a.text(-6, 1.5, "260 speech bits\n(20 ms)", ha="right", va="center", fontsize=6.8)
    x = 0
    segs2 = [(2 * 189, NAVY, "378 = 2 x (182 + 3 CRC + 4 tail), rate-1/2 convolutional code"), (78, GRAY, "78")]
    for n, c, nm in segs2:
        a.add_patch(Rectangle((x, 0.2), n, 0.6, fc=c, alpha=0.8, lw=0)); a.text(x + n / 2, 0.5, nm, ha="center",
                                                                               va="center", fontsize=6.5, color="white")
        x += n
    a.text(-6, 0.5, "456 coded bits\n(22.8 kb/s)", ha="right", va="center", fontsize=6.8)
    for xa, xb in [(0, 0), (182, 378)]:
        a.plot([xa, xb], [1.2, 0.8], color=GRAY, lw=0.6, ls=":")
    a.set_xlim(-95, 460); a.set_ylim(0, 2); a.axis("off")
    a.set_title("(a) unequal error protection of a full-rate speech frame", fontsize=8.5, loc="left")
    a = axs[1]
    cA, cB, cC = NAVY, BLUE2, GREEN
    for bidx in range(12):
        x = bidx * 1.0
        # each burst: even bits from frame k, odd bits from frame k-1 (diagonal over 8 bursts)
        fa = bidx // 4
        top = [cA, cB, cC, ORANGE][fa % 4]
        bot = [GRAY, cA, cB, cC][fa % 4]
        lost = bidx == 6
        a.add_patch(Rectangle((x + 0.05, 0.75), 0.9, 0.5, fc=top, alpha=0.25 if lost else 0.85, lw=0))
        a.add_patch(Rectangle((x + 0.05, 0.2), 0.9, 0.5, fc=bot, alpha=0.25 if lost else 0.85, lw=0))
        if lost:
            a.plot([x + 0.05, x + 0.95], [0.2, 1.25], color=ACCENT, lw=1.5)
            a.plot([x + 0.05, x + 0.95], [1.25, 0.2], color=ACCENT, lw=1.5)
            a.text(x + 0.5, 1.33, "burst lost in a fade", ha="center", fontsize=6.3, color=ACCENT)
        a.text(x + 0.5, 0.05, f"{bidx}", ha="center", fontsize=6, color="#333")
    a.text(-0.1, 1.0, "57 bits\n57 bits", ha="right", va="center", fontsize=6.3)
    a.set_xlim(-1.2, 12.1); a.set_ylim(-0.1, 1.55); a.axis("off")
    a.set_title("(b) each speech frame is spread over 8 bursts (colours); a lost burst costs each frame only 1/8",
                fontsize=8.0, loc="left")
    fig.tight_layout(h_pad=0.4); save(fig, "ch21_speech_bits")


# ---------------------------------------------------------------------------- AMR modes
def amr_modes():
    modes = [12.2, 10.2, 7.95, 7.4, 6.7, 5.9, 5.15, 4.75]
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    y = np.arange(len(modes))
    ax.barh(y, modes, color=NAVY, label="speech")
    ax.barh(y, [22.8 - m for m in modes], left=modes, color=ORANGE, alpha=0.75, label="error protection")
    ax.set_yticks(y); ax.set_yticklabels([f"{m}" for m in modes], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("kb/s in a 22.8 kb/s full-rate channel")
    ax.set_ylabel("AMR mode (kb/s)"); ax.set_xlim(0, 22.8)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.text(0.3, -0.85, "near the site", fontsize=6.5, color=NAVY)
    ax.text(0.3, 7.8, "cell edge", fontsize=6.5, color=ORANGE, va="top")
    fig.tight_layout(); save(fig, "ch21_amr_modes")


# ---------------------------------------------------------------------------- frequency hopping
def _rayleigh_tf(r, ncar, nfr, fd, Tf, M=24):
    t = np.arange(nfr) * Tf
    out = np.zeros((ncar, nfr), complex)
    for c in range(ncar):
        th = r.uniform(0, 2 * np.pi, M); ph = r.uniform(0, 2 * np.pi, M)
        out[c] = np.exp(1j * (2 * np.pi * fd * np.cos(th)[:, None] * t[None, :] + ph[:, None])).sum(0) / np.sqrt(M)
    return out


def hopping():
    r = rng(21)
    Tf = 60 / 13 * 1e-3
    fd = 3 / 3.6 / 3e8 * 900e6              # 3 km/h at 900 MHz -> 2.5 Hz
    ncar = 8
    thr = -12.0
    # statistics over a long run
    nfr = int(120 / Tf)
    h = _rayleigh_tf(r, ncar, nfr, fd, Tf)
    p = 20 * np.log10(np.abs(h))
    bad = p < thr
    nfrm = (nfr - 8) // 4
    fixed_loss = hop_loss = 0
    for k in range(nfrm):
        idx = np.arange(4 * k, 4 * k + 8)
        fixed_loss += bad[0, idx].sum() > 2
        hop_loss += bad[idx % ncar, idx].sum() > 2
    fer_f, fer_h = fixed_loss / nfrm, hop_loss / nfrm
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    show = 110
    im = ax.imshow(np.clip(p[:, :show], -25, 8), aspect="auto", origin="lower", cmap="Blues_r",
                   extent=[0, show * Tf * 1e3, -0.5, ncar - 0.5], vmin=-25, vmax=8)
    tt = (np.arange(show) + 0.5) * Tf * 1e3
    ax.plot(tt, np.zeros(show), "s", color=ACCENT, ms=2.6, label=f"fixed carrier: {100 * fer_f:.1f}% speech frames lost")
    ax.plot(tt, np.arange(show) % ncar, "o", color=GREEN, ms=2.6, label=f"hopping: {100 * fer_h:.2f}% lost")
    ax.set_xlabel("time (ms), one dot per TDMA frame"); ax.set_ylabel("carrier")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("fade (dB)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax.set_title("A walker at 3 km/h, 900 MHz: deep fades last ~100 ms (statistics from a 2-minute run)",
                 fontsize=8)
    fig.tight_layout(); save(fig, "ch21_hopping")
    print("FER fixed/hop", fer_f, fer_h)


# ---------------------------------------------------------------------------- 217 Hz buzz
def buzz217():
    Tf = 60 / 13 * 1e-3
    d = 1 / 8
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.1))
    t = np.linspace(0, 20e-3, 4000)
    env = ((t % Tf) < d * Tf).astype(float)
    axs[0].fill_between(t * 1e3, 0, env * 2, color=NAVY, alpha=0.8, lw=0)
    axs[0].set_xlabel("time (ms)"); axs[0].set_ylabel("transmit power (W)")
    axs[0].set_title("(a) one burst every 4.615 ms", fontsize=8.5); axs[0].set_ylim(0, 2.4)
    f0 = 1 / Tf
    k = np.arange(1, 20)
    c = 2 * d * np.abs(np.sinc(k * d))
    axs[1].stem(k * f0, 20 * np.log10(c / c[0]), basefmt=" ", linefmt=ACCENT, markerfmt="o", bottom=-40)
    axs[1].set_xlabel("audio frequency (Hz)"); axs[1].set_ylabel("relative level (dB)")
    axs[1].set_title(f"(b) envelope harmonics of {f0:.1f} Hz", fontsize=8.5)
    axs[1].set_ylim(-40, 3); axs[1].set_xlim(0, 4200)
    fig.tight_layout(w_pad=0.8); save(fig, "ch21_buzz217")


# ---------------------------------------------------------------------------- SMS packing
def sms_packing():
    fig, ax = plt.subplots(figsize=(W1, 1.55))
    txt = "MERRY CH"
    cols = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY, "#117A65"]
    bits = []
    for i, ch in enumerate(txt):
        for b in range(7):
            bits.append(i)
    for j, ci in enumerate(bits):
        octet, pos = divmod(j, 8)
        x = octet * 8.6 + pos
        ax.add_patch(Rectangle((x, 0.3), 0.92, 0.7, fc=cols[ci], alpha=0.85, lw=0))
    for o in range(7):
        ax.text(o * 8.6 + 4, 1.15, f"octet {o + 1}", ha="center", fontsize=6.5)
    for i, ch in enumerate(txt):
        j = i * 7 + 3
        octet, pos = divmod(j, 8)
        ax.text(octet * 8.6 + pos + 0.46, 0.65, ch if ch != " " else "_", ha="center", va="center", fontsize=7,
                color="white", weight="bold")
    ax.text(30, -0.15, "8 characters x 7 bits = 56 bits = 7 octets;  160 characters x 7 bits = 1120 bits = 140 octets",
            ha="center", fontsize=7, color=NAVY)
    ax.set_xlim(-0.5, 60.5); ax.set_ylim(-0.35, 1.35); ax.axis("off")
    save(fig, "ch21_sms_packing")


# ---------------------------------------------------------------------------- near-far
def near_far():
    d = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    pl = 128.1 + 37.6 * np.log10(d)          # a common macro-cell model at 2 GHz (3GPP TR 25.942 style)
    rx = 21 - pl                              # every phone at +21 dBm
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    x = np.arange(len(d))
    ax.bar(x - 0.2, rx + 140, bottom=-140, width=0.38, color=ACCENT, label="all at full power")
    ax.bar(x + 0.2, rx.min() + 140, bottom=-140, width=0.38, color=GREEN, label="with power control")
    for i in range(len(d)):
        ax.text(i - 0.2, rx[i] + 1.5, f"+{rx[i] - rx.min():.0f}", ha="center", fontsize=6.5, color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g} km" for v in d])
    ax.set_ylabel("received power at the base (dBm)"); ax.set_ylim(-140, -40)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); save(fig, "ch21_near_far")


# ---------------------------------------------------------------------------- EDGE 8-PSK
def edge_8psk():
    r = rng(2)
    n = 400; sp = 16
    k = r.integers(0, 8, n)
    tt = np.arange(-2 * sp, 2 * sp + 1) / sp
    g = np.exp(-tt ** 2 / (2 * 0.45 ** 2))        # approximate linearised-GMSK pulse
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
    for a, rot, title in [(axs[0], 0.0, "(a) plain 8-PSK"), (axs[1], 3 * np.pi / 8, r"(b) EDGE: rotate by $3\pi/8$ per symbol")]:
        s = np.exp(1j * (2 * np.pi * k / 8 + rot * np.arange(n)))
        up = np.zeros(n * sp, complex); up[::sp] = s
        x = np.convolve(up, g)[2 * sp:-2 * sp]
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        a.plot(x.real, x.imag, color=NAVY, lw=0.3, alpha=0.6)
        a.add_patch(Circle((0, 0), np.min(np.abs(x[4 * sp:-4 * sp])), fc=ACCENT, alpha=0.25, lw=0))
        mn = 20 * np.log10(np.min(np.abs(x[4 * sp:-4 * sp])) / np.sqrt(np.mean(np.abs(x) ** 2)))
        a.set_aspect("equal"); a.set_xlim(-1.9, 1.9); a.set_ylim(-1.9, 1.9)
        a.set_title(title, fontsize=8.5)
        a.text(0, -1.75, "passes through zero" if mn < -40 else f"never below {mn:.0f} dB re rms", ha="center", fontsize=6.8, color=ACCENT)
        a.axis("off")
    fig.tight_layout(w_pad=0.5); save(fig, "ch21_edge_8psk")


# ---------------------------------------------------------------------------- 3GPP releases
def releases():
    rel = [("R99", 2000.2, "3G", "first UMTS"), ("Rel-4", 2001.2, "3G", ""), ("Rel-5", 2002.5, "3G", "HSDPA"),
           ("Rel-6", 2005.2, "3G", "HSUPA"), ("Rel-7", 2007.9, "3G", "HSPA+"), ("Rel-8", 2008.95, "4G", "first LTE"),
           ("Rel-9", 2009.95, "4G", ""), ("Rel-10", 2011.3, "4G", "LTE-Advanced"), ("Rel-11", 2013.2, "4G", "CoMP"),
           ("Rel-12", 2015.2, "4G", "dual conn."), ("Rel-13", 2016.2, "4G", "NB-IoT, LTE-A Pro"),
           ("Rel-14", 2017.4, "4G", "V2X"), ("Rel-15", 2018.5, "5G", "first NR"), ("Rel-16", 2020.5, "5G", "URLLC, IIoT"),
           ("Rel-17", 2022.4, "5G", "RedCap, NTN"), ("Rel-18", 2024.3, "5G", "5G-Advanced"),
           ("Rel-19", 2025.9, "5G", ""), ("Rel-20", 2027.5, "5G", "6G studies")]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, (nm, yr, g, note) in enumerate(rel):
        c = GEN_COL[g]
        y = (i % 3)
        ax.plot(yr, 0, "o", color=c, ms=5, zorder=3)
        ax.plot([yr, yr], [0, 0.35 + 0.45 * y], color=c, lw=0.6)
        ax.text(yr, 0.38 + 0.45 * y, nm + ("\n" + note if note else ""), ha="center", va="bottom", fontsize=5.8,
                color=c, linespacing=1.0)
    ax.axhline(0, color=GRAY, lw=1)
    for yr in range(2000, 2029, 4):
        ax.text(yr, -0.18, str(yr), ha="center", va="top", fontsize=6.5, color=GRAY)
    ax.set_xlim(1998.8, 2028.8); ax.set_ylim(-0.45, 2.3); ax.axis("off")
    save(fig, "ch21_releases")


# ---------------------------------------------------------------------------- OVSF tree
def ovsf_tree():
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    ax.axis("off")

    def node(code, depth, y0, y1, used=None):
        y = (y0 + y1) / 2
        x = depth * 1.0
        return x, y

    pos = {}
    codes = {(0, 0): [1]}
    for d in range(1, 4):
        for i in range(2 ** d):
            parent = codes[(d - 1, i // 2)]
            codes[(d, i)] = parent + parent if i % 2 == 0 else parent + [-v for v in parent]
    for d in range(4):
        n = 2 ** d
        for i in range(n):
            pos[(d, i)] = (d * 1.25, (i + 0.5) / n * 8)
    used = (2, 1)
    blocked = {(1, 0), (0, 0), (3, 2), (3, 3)}
    for (d, i), (x, y) in pos.items():
        if d < 3:
            for j in (2 * i, 2 * i + 1):
                x2, y2 = pos[(d + 1, j)]
                ax.plot([x, x2], [y, y2], color=GRAY, lw=0.6)
    for (d, i), (x, y) in pos.items():
        c = ACCENT if (d, i) == used else (GRAY if (d, i) in blocked else NAVY)
        ax.plot(x, y, "o", color=c, ms=5)
        if d == 3 or (d, i) == used:
            s = "".join("+" if v > 0 else "-" for v in codes[(d, i)])
            ax.text(x + 0.12, y, s, va="center", fontsize=6.5, color=c, family="monospace")
    for d in range(4):
        ax.text(d * 1.25, 8.5, f"SF {2 ** d}", ha="center", fontsize=7, color=NAVY)
    ax.text(0, -0.6, "red: in use (SF 4); grey: blocked\n(its parents and children)", fontsize=6.6, color="#333")
    ax.set_xlim(-0.3, 5.3); ax.set_ylim(-1.2, 9)
    fig.tight_layout(); save(fig, "ch21_ovsf_tree")


# ---------------------------------------------------------------------------- hard handover with hysteresis
def handover_hyst():
    r = rng(31)
    dx = 5.0
    x = np.arange(250, 3750 + dx, dx)
    def shad(sig=6.0, dcorr=50.0):
        a = np.exp(-dx / dcorr)
        s = np.zeros_like(x); s[0] = r.normal(0, sig)
        for i in range(1, len(x)):
            s[i] = a * s[i - 1] + np.sqrt(1 - a * a) * r.normal(0, sig)
        return s
    pA = 46 - (128.1 + 37.6 * np.log10(np.maximum(x, 30) / 1e3)) + shad()
    pB = 46 - (128.1 + 37.6 * np.log10(np.maximum(4000 - x, 30) / 1e3)) + shad()
    def run(hyst, ttt):
        serv = 0; cnt = 0; s = np.zeros(len(x), int); timer = 0
        for i in range(len(x)):
            other = pB[i] if serv == 0 else pA[i]; me = pA[i] if serv == 0 else pB[i]
            if other > me + hyst:
                timer += 1
                if timer * dx >= ttt:
                    serv = 1 - serv; cnt += 1; timer = 0
            else:
                timer = 0
            s[i] = serv
        return s, cnt
    s0, c0 = run(0.0, 0.0)
    s3, c3 = run(3.0, 40.0)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.plot(x / 1e3, pA, color=NAVY, lw=0.9, label="cell A")
    ax.plot(x / 1e3, pB, color=ORANGE, lw=0.9, label="cell B")
    ax.fill_between(x / 1e3, -125, -125 + 4 * s0, color=ACCENT, alpha=0.6, lw=0, step="mid",
                    label=f"no hysteresis: {c0} handovers")
    ax.fill_between(x / 1e3, -132, -132 + 4 * s3, color=GREEN, alpha=0.7, lw=0, step="mid",
                    label=f"3 dB hysteresis + 40 m trigger: {c3}")
    ax.set_xlabel("position along the road (km)"); ax.set_ylabel("received power (dBm)")
    ax.set_ylim(-134, -40); ax.legend(fontsize=6.4, loc="upper center", ncol=2)
    ax.text(4.0, -126, "bars: serving cell B", ha="right", fontsize=6.3, color=GRAY)
    fig.tight_layout(); save(fig, "ch21_handover_hyst")
    print("handovers", c0, c3)


# ---------------------------------------------------------------------------- cell breathing
def cell_breathing():
    eta = np.linspace(0, 0.9, 200)
    nr = -10 * np.log10(1 - eta)
    rr = 10 ** (-nr / (10 * 3.5))
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    ax.plot(100 * eta, 100 * rr, color=NAVY, label="cell radius")
    ax.plot(100 * eta, 100 * rr ** 2, color=ACCENT, label="cell area")
    ax2 = ax.twinx(); ax2.plot(100 * eta, nr, color=GRAY, ls="--", lw=1); ax2.set_ylabel("noise rise (dB)", color=GRAY)
    ax2.spines["right"].set_visible(True); ax2.grid(False); ax2.tick_params(labelsize=7.5)
    for e in (0.5, 0.75):
        ax.axvline(100 * e, color=GRAY, lw=0.5, ls=":")
    ax.text(51, 30, "3 dB", fontsize=6.5, color=GRAY); ax.text(76, 30, "6 dB", fontsize=6.5, color=GRAY)
    ax.set_xlabel("uplink load (% of pole capacity)"); ax.set_ylabel("% of empty-cell value")
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_ylim(0, 105)
    fig.tight_layout(); save(fig, "ch21_cell_breathing")


# ---------------------------------------------------------------------------- RRC states
def rrc_states():
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.35, 0.8, 1]))
    sets = [("UMTS (5 states)", ["CELL_DCH", "CELL_FACH", "CELL_PCH", "URA_PCH", "IDLE"], GREEN),
            ("LTE (2 states)", ["CONNECTED", "IDLE"], ORANGE),
            ("NR (3 states)", ["CONNECTED", "INACTIVE", "IDLE"], ACCENT)]
    for a, (t, st, c) in zip(axs, sets):
        a.set_xlim(0, 1); a.set_ylim(-0.22, 1.1); a.axis("off"); a.set_title(t, fontsize=8.5, color=c)
        n = len(st)
        ys = np.linspace(0.85, 0.08, n)
        for i, (s, y) in enumerate(zip(st, ys)):
            _box(a, 0.12, y - 0.065, 0.76, 0.13, s, c if i == 0 else SOFT["3G" if c == GREEN else "4G" if c == ORANGE else "5G"],
                 fs=6.6, color="white" if i == 0 else "#222")
            if i:
                _arrow(a, (0.42, y + 0.065), (0.42, ys[i - 1] - 0.065), color=GRAY, lw=0.7, ms=6)
                _arrow(a, (0.58, ys[i - 1] - 0.065), (0.58, y + 0.065), color=GRAY, lw=0.7, ms=6)
    axs[0].text(0.5, -0.06, "moving up can take\nhundreds of ms to seconds", ha="center", fontsize=6.2, color=GRAY)
    axs[1].text(0.5, -0.06, "idle to connected\n< 100 ms target", ha="center", fontsize=6.2, color=GRAY)
    axs[2].text(0.5, -0.06, "INACTIVE keeps context:\nresume quickly", ha="center", fontsize=6.2, color=GRAY)
    fig.tight_layout(w_pad=0.3); save(fig, "ch21_rrc_states")


# ---------------------------------------------------------------------------- proportional fair
def pf_sched():
    r = rng(41)
    Tslot = 1.67e-3; fd = 10.0
    nsl = 3000
    def fade(n):
        return _rayleigh_tf(r, n, nsl, fd, Tslot)
    means = np.array([10, 6, 3, 0.0])
    h = fade(4)
    snr = 10 ** (means[:, None] / 10) * np.abs(h) ** 2
    rate = np.log2(1 + snr)
    avg = np.ones(4) * 0.5; serv = np.zeros(nsl, int); tc = 50.0
    for t in range(nsl):
        k = np.argmax(rate[:, t] / avg); serv[t] = k
        got = np.zeros(4); got[k] = rate[k, t]
        avg = (1 - 1 / tc) * avg + got / tc
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.45), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = axs[0]
    cols = [NAVY, GREEN, ORANGE, PURPLE]
    sh = slice(0, 240)
    tt = np.arange(nsl)[sh] * Tslot * 1e3
    for k in range(4):
        a.plot(tt, 10 * np.log10(snr[k, sh]), color=cols[k], lw=0.7, label=f"user {k + 1} (mean {means[k]:.0f} dB)")
        m = serv[sh] == k
        a.plot(tt[m], 10 * np.log10(snr[k, sh][m]), "o", color=cols[k], ms=2.2)
    a.set_xlabel("time (ms)"); a.set_ylabel("SNR (dB)"); a.set_ylim(-20, 22)
    a.legend(fontsize=5.8, loc="lower left", ncol=2); a.set_title("(a) dots: who the PF scheduler serves", fontsize=8.5)
    a = axs[1]
    Us = [1, 2, 4, 8, 16, 32]
    rr_, pf_ = [], []
    for U in Us:
        hh = _rayleigh_tf(r, U, 1500, fd, Tslot)
        rt = np.log2(1 + 10 ** 0.5 * np.abs(hh) ** 2)
        rr_.append(rt.mean())
        av = np.ones(U) * 0.5; tot = 0
        for t in range(1500):
            k = np.argmax(rt[:, t] / av); tot += rt[k, t]
            g = np.zeros(U); g[k] = rt[k, t]; av = (1 - 1 / tc) * av + g / tc
        pf_.append(tot / 1500)
    a.plot(Us, rr_, "s-", color=GRAY, ms=3.5, label="round robin")
    a.plot(Us, pf_, "o-", color=NAVY, ms=3.5, label="proportional fair")
    a.set_xscale("log", base=2); a.set_xticks(Us); a.set_xticklabels(Us)
    a.set_xlabel("users in the cell (all 5 dB mean)"); a.set_ylabel("cell throughput (b/s/Hz)")
    a.legend(fontsize=6.5); a.set_title("(b) multi-user diversity gain", fontsize=8.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch21_pf_sched")
    print("PF gain at 32 users", pf_[-1] / rr_[-1])


# ---------------------------------------------------------------------------- equaliser complexity
def eq_complexity():
    B = np.logspace(0, 2, 60) * 1e6
    tau = 5e-6
    td = B * (B * tau)                       # taps x sample rate (complex MACs/s)
    Nfft = 2 ** np.ceil(np.log2(B / 15e3 * 1.3))
    ofdm = B * (np.log2(Nfft) / 2 + 1) * 1.15
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    ax.loglog(B / 1e6, td, color=ACCENT, label="time-domain equaliser")
    ax.loglog(B / 1e6, ofdm, color=NAVY, label="OFDM: FFT + 1 tap/subcarrier")
    for b in (5, 20, 100):
        ax.axvline(b, color=GRAY, lw=0.5, ls=":")
    ax.text(5, 3e10, "WCDMA", fontsize=6.3, rotation=90, color=GRAY); ax.text(20, 3e10, "LTE", fontsize=6.3, rotation=90, color=GRAY)
    ax.text(100, 3e10, "NR", fontsize=6.3, rotation=90, color=GRAY, ha="right")
    ax.set_xlabel("bandwidth (MHz)"); ax.set_ylabel("complex MACs per second")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_title(r"5 $\mu$s delay spread, one antenna", fontsize=8)
    fig.tight_layout(); save(fig, "ch21_eq_complexity")


# ---------------------------------------------------------------------------- PAPR
def papr():
    r = rng(51)
    M, Nfft, os_ = 300, 512, 4
    nsym = 3000
    def ccdf(x):
        p = np.abs(x) ** 2
        pp = 10 * np.log10(p.max(1) / p.mean(1))
        g = np.linspace(2, 12, 101)
        return g, [(pp > v).mean() for v in g]
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    for mod, ls in [(4, "-"), (16, "--")]:
        if mod == 4:
            d = (r.choice([-1, 1], (nsym, M)) + 1j * r.choice([-1, 1], (nsym, M))) / np.sqrt(2)
        else:
            lv = np.array([-3, -1, 1, 3])
            d = (r.choice(lv, (nsym, M)) + 1j * r.choice(lv, (nsym, M))) / np.sqrt(10)
        X = np.zeros((nsym, Nfft * os_), complex)
        X[:, 1:M + 1] = d
        g, c = ccdf(np.fft.ifft(X, axis=1))
        ax.semilogy(g, np.maximum(c, 1e-4), color=ACCENT, ls=ls, label=f"OFDMA, {'QPSK' if mod == 4 else '16QAM'}")
        Xs = np.zeros((nsym, Nfft * os_), complex)
        Xs[:, 1:M + 1] = np.fft.fft(d, axis=1) / np.sqrt(M)
        g, c = ccdf(np.fft.ifft(Xs, axis=1))
        ax.semilogy(g, np.maximum(c, 1e-4), color=NAVY, ls=ls, label=f"SC-FDMA, {'QPSK' if mod == 4 else '16QAM'}")
    ax.set_xlabel("PAPR threshold (dB)"); ax.set_ylabel("P(PAPR > threshold)")
    ax.set_ylim(1e-3, 1.1); ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(); save(fig, "ch21_papr")


# ---------------------------------------------------------------------------- HARQ processes
def harq_processes():
    cols = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY, "#117A65"]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    n = 20
    for sf in range(n):
        p = sf % 8
        retx = (sf == 8)
        ax.add_patch(Rectangle((sf, 1.2), 0.94, 0.6, fc=cols[p], alpha=1.0 if p == 0 else 0.35, lw=0))
        ax.text(sf + 0.47, 1.5, ("P0\nretx" if retx else f"P{p}") if p == 0 else f"P{p}", ha="center",
                va="center", fontsize=5.6, color="white" if p == 0 else "#333", linespacing=0.9)
        ax.text(sf + 0.47, 2.0, str(sf), ha="center", fontsize=6, color=GRAY)
    ax.add_patch(Rectangle((4, 0.25), 0.94, 0.5, fc=ACCENT, lw=0)); ax.text(4.47, 0.5, "NACK", ha="center",
                                                                            va="center", fontsize=5.8, color="white")
    ax.add_patch(Rectangle((12, 0.25), 0.94, 0.5, fc=GREEN, lw=0)); ax.text(12.47, 0.5, "ACK", ha="center",
                                                                            va="center", fontsize=5.8, color="white")
    _arrow(ax, (0.5, 1.18), (4.3, 0.78), color=ACCENT, lw=0.8, ms=6)
    _arrow(ax, (4.7, 0.78), (8.4, 1.18), color=ACCENT, lw=0.8, ms=6)
    _arrow(ax, (8.5, 1.18), (12.3, 0.78), color=GREEN, lw=0.8, ms=6)
    ax.text(-0.3, 1.5, "downlink\ndata", ha="right", va="center", fontsize=6.8)
    ax.text(-0.3, 0.5, "uplink\nfeedback", ha="right", va="center", fontsize=6.8)
    ax.text(10, 2.35, "subframe (1 ms): data in n, feedback in n+4, retransmission from n+8", ha="center",
            fontsize=6.8, color=NAVY)
    ax.set_xlim(-2.8, 20.2); ax.set_ylim(0.1, 2.6); ax.axis("off")
    save(fig, "ch21_harq_processes")


# ---------------------------------------------------------------------------- LTE peak-rate waterfall
def lte_waterfall():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = axs[0]
    steps = [("all REs", 16800, NAVY), ("control\n(1 symbol)", -1200, ACCENT), ("CRS\n(4 ports)", -2000, ACCENT),
             ("PDSCH", 13600, GREEN)]
    base = 0
    for i, (nm, v, c) in enumerate(steps):
        if v > 0:
            a.bar(i, v, color=c, width=0.6); base = v
        else:
            a.bar(i, -v, bottom=base + v, color=c, width=0.6); base += v
        a.text(i, (base if v < 0 else v) + 300, f"{abs(v):,}", ha="center", fontsize=6.5)
    a.set_xticks(range(4)); a.set_xticklabels([s[0] for s in steps], fontsize=6.5)
    a.set_ylabel("resource elements per ms"); a.set_ylim(0, 19000); a.set_title("(a) 20 MHz: 100 RBs x 14 symbols", fontsize=8.5)
    a = axs[1]
    vals = [13.6, 13.6 * 6, 13.6 * 6 * 4, 13.6 * 6 * 4 * 0.92]
    nms = ["x1 bit\n(per RE)", "x6 bits\n64QAM", "x4 layers", "x0.92\ncode rate"]
    a.bar(range(4), vals, color=[GRAY, BLUE2, NAVY, GREEN], width=0.6)
    for i, v in enumerate(vals):
        a.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=6.8)
    a.set_xticks(range(4)); a.set_xticklabels(nms, fontsize=6.5); a.set_ylabel("Mb/s"); a.set_ylim(0, 360)
    a.set_title("(b) from REs to ~300 Mb/s", fontsize=8.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch21_lte_waterfall")


# ---------------------------------------------------------------------------- carrier aggregation
def carrier_agg():
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    hold = [(800, 10, "800\n10 MHz"), (1800, 20, "1800\n20 MHz"), (2100, 15, "2100\n15 MHz"), (2600, 20, "2600\n20 MHz")]
    cols = [NAVY, GREEN, ORANGE, ACCENT]
    for i, (f, bw, nm) in enumerate(hold):
        x = np.log10(f)
        ax.add_patch(Rectangle((x - 0.012 * bw / 10, 1.3), 0.024 * bw / 10, 0.5, fc=cols[i], lw=0))
        ax.text(x + (-0.025 if i == 1 else 0.025 if i == 2 else 0), 1.9, nm, ha="center", fontsize=6.5, color=cols[i])
        xe = 3.05 + sum(h[1] for h in hold[:i]) * 0.0085
        ax.add_patch(Rectangle((xe, 0.2), bw * 0.0085, 0.45, fc=cols[i], lw=0))
        _arrow(ax, (x, 1.28), (xe + bw * 0.0085 / 2, 0.68), color=cols[i], lw=0.6, ms=6, ls=":")
    ax.text(3.05 + 65 * 0.0085 + 0.02, 0.42, "one 65 MHz pipe\nto the phone (PCell + 3 SCells)", fontsize=6.7,
            va="center", color="#333")
    ax.text(2.88, 1.55, "operator's\nspectrum (MHz)", fontsize=6.6, ha="right", va="center")
    ax.set_xlim(2.7, 3.75); ax.set_ylim(0, 2.3); ax.axis("off")
    save(fig, "ch21_carrier_agg")


# ---------------------------------------------------------------------------- VoLTE capacity
def volte_capacity():
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    nm = ["GSM\nfull rate", "GSM\nhalf rate", "VoLTE\ndynamic", "VoLTE\nSPS"]
    lo = [30, 60, 200, 500]; hi = [30, 60, 400, 900]
    c = [NAVY, NAVY, ORANGE, ORANGE]
    for i in range(4):
        ax.bar(i, lo[i], color=c[i], alpha=0.9, width=0.6)
        if hi[i] > lo[i]:
            ax.bar(i, hi[i] - lo[i], bottom=lo[i], color=c[i], alpha=0.35, width=0.6)
        ax.text(i, hi[i] + 15, f"{lo[i]}" + (f"-{hi[i]}" if hi[i] > lo[i] else ""), ha="center", fontsize=6.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(nm, fontsize=6.8)
    ax.set_ylabel("voice calls per sector in 10 MHz"); ax.set_ylim(0, 1000)
    ax.text(-0.4, 960, "rough estimates\n(see worked example)", fontsize=6, color=GRAY, ha="left", va="top")
    fig.tight_layout(); save(fig, "ch21_volte_capacity")


# ---------------------------------------------------------------------------- IMT-2020 usage triangle
def usage_triangle():
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    P = np.array([[0.5, 0.95], [0.03, 0.08], [0.97, 0.08]])
    ax.add_patch(Polygon(P, closed=True, fc="#FBEEE6", ec=ACCENT, lw=1.0))
    ax.text(0.5, 1.0, "eMBB\nGb/s, capacity", ha="center", va="bottom", fontsize=7.5, color=ACCENT, weight="bold")
    ax.text(0.0, 0.0, "mMTC\n10$^6$ devices/km$^2$", ha="left", va="top", fontsize=7, color=PURPLE, weight="bold")
    ax.text(1.0, 0.0, "URLLC\n1 ms, $1-10^{-5}$", ha="right", va="top", fontsize=7, color=GREEN, weight="bold")
    apps = [(0.5, 0.68, "4K/8K video,\nfixed wireless"), (0.5, 0.52, "AR/VR"), (0.25, 0.22, "smart meters,\nsensors"),
            (0.75, 0.22, "factory control,\nremote surgery"), (0.62, 0.38, "vehicles"), (0.38, 0.38, "smart city")]
    for x, y, t in apps:
        ax.text(x, y, t, ha="center", va="center", fontsize=6.2, color="#333")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.18, 1.18); ax.axis("off")
    save(fig, "ch21_usage_triangle")


# ---------------------------------------------------------------------------- NR numerology
def numerology():
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    for mu in range(4):
        scs = 15 * 2 ** mu
        ns = 2 ** mu
        y = 3 - mu
        for s in range(ns):
            w = 1.0 / ns
            ax.add_patch(Rectangle((s * w, y + 0.15), w * 0.97, 0.6, fc=list(GEN_COL.values())[mu + 1], alpha=0.25 + 0.15 * (s % 2),
                                   lw=0))
            if ns <= 2:
                for k in range(14):
                    ax.plot([s * w + k * w / 14] * 2, [y + 0.15, y + 0.75], color="white", lw=0.4)
        ax.text(-0.02, y + 0.45, f"$\\mu$={mu}: {scs} kHz", ha="right", va="center", fontsize=7)
        ax.text(1.02, y + 0.45, f"slot {1000 / ns:g} $\\mu$s, symbol {66.7 / ns:.1f} $\\mu$s", ha="left",
                va="center", fontsize=6.6, color="#333")
    ax.text(0.5, 4.0, "one 1 ms subframe (each block = one 14-symbol slot)", ha="center", fontsize=7, color=NAVY)
    ax.set_xlim(-0.28, 1.45); ax.set_ylim(-0.05, 4.25); ax.axis("off")
    save(fig, "ch21_numerology")


# ---------------------------------------------------------------------------- bandwidth part
def bwp():
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.add_patch(Rectangle((0, 0), 10, 100, fc="#F2F3F4", lw=0))
    segs = [(0, 3, 40, 20), (3, 6, 0, 100), (6, 10, 40, 20)]
    for t0, t1, f0, bw in segs:
        ax.add_patch(Rectangle((t0, f0), t1 - t0, bw, fc=NAVY if bw > 50 else BLUE2, alpha=0.75, lw=0))
    for t in (3, 6):
        ax.plot(t, 30, "v", color=ACCENT, ms=6); ax.text(t, 26, "DCI\nswitch", ha="center", va="top", fontsize=6, color=ACCENT, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(1.5, 50, "browsing:\n20 MHz BWP", ha="center", va="center", fontsize=6.3, color="white")
    ax.text(4.5, 85, "download:\nfull 100 MHz", ha="center", fontsize=6.3, color="white", va="center")
    ax.set_xlabel("time"); ax.set_ylabel("frequency in the carrier (MHz)")
    ax.set_xticks([]); ax.set_xlim(0, 10); ax.set_ylim(0, 100)
    fig.tight_layout(); save(fig, "ch21_bwp")


# ---------------------------------------------------------------------------- SSB beam sweep
def beam_sweep():
    Nel = 16
    th = np.linspace(-90, 90, 721)
    steer = np.linspace(-52.5, 52.5, 8)
    fig = plt.figure(figsize=(3.2, 2.5))
    ax = fig.add_subplot(111, projection="polar")
    ue = 22.0
    best, bg = None, -99
    cols = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY, "#117A65"]
    for i, s in enumerate(steer):
        n = np.arange(Nel)
        w = np.exp(1j * np.pi * n * np.sin(np.deg2rad(s)))
        af = np.abs(np.exp(1j * np.pi * np.outer(np.sin(np.deg2rad(th)), n)) @ w.conj()) / Nel
        g = 20 * np.log10(af + 1e-6)
        ax.plot(np.deg2rad(th), np.clip(g + 20, 0, 20), color=cols[i], lw=0.9)
        gu = np.interp(ue, th, g)
        if gu > bg:
            bg, best = gu, i
    ax.plot(np.deg2rad(ue), 21, "*", color="k", ms=8)
    ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([10, 20]); ax.set_yticklabels(["-10", "0 dB"], fontsize=6)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title(f"8 SSB beams; phone (star) at {ue:.0f}$^\\circ$ picks SSB {best}", fontsize=7.5)
    fig.tight_layout(); save(fig, "ch21_beam_sweep")


# ---------------------------------------------------------------------------- network slicing
def slicing():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    lanes = [("eMBB slice: video, browsing (best effort, wide)", ACCENT, 0.9),
             ("URLLC slice: factory robots (reserved, short delay)", GREEN, 0.45),
             ("mMTC slice: meters, sensors (many tiny cars)", PURPLE, 0.45),
             ("private slice: emergency services (guaranteed)", NAVY, 0.55)]
    y = 0.2
    ax.add_patch(Rectangle((0, 0.1), 10, 2.55, fc="#D5D8DC", lw=0))
    r = rng(3)
    for nm, c, h in lanes:
        ax.add_patch(Rectangle((0.0, y), 10, h, fc=c, alpha=0.15, lw=0))
        ax.plot([0, 10], [y + h, y + h], color="white", lw=1.5, ls=(0, (6, 4)))
        if c == PURPLE:
            for k in range(22):
                _car(ax, 0.2 + k * 0.45, y + 0.12, 0.25, 0.2, c, 0.8)
        elif c == GREEN:
            for k in range(3):
                _car(ax, 0.6 + 3.1 * k, y + 0.1, 0.7, 0.25, c)
        elif c == ACCENT:
            for k in range(5):
                _car(ax, 0.3 + 2.0 * k + 0.4 * r.random(), y + 0.15 + 0.3 * r.random(), 1.1, 0.35, c, 0.85)
        else:
            _car(ax, 4.0, y + 0.12, 1.4, 0.3, c)
        ax.text(10.1, y + h / 2, nm, fontsize=6.5, va="center", color=c)
        y += h + 0.05
    ax.text(5, 2.75, "one physical network (spectrum, sites, core) -- four logically separate networks", ha="center",
            fontsize=7, color="#333")
    ax.set_xlim(-0.1, 17.5); ax.set_ylim(0, 2.95); ax.axis("off")
    save(fig, "ch21_slicing")


# ---------------------------------------------------------------------------- edge latency
def mec_latency():
    cases = [("edge server\n(10 km, local UPF)", 10), ("regional DC\n(300 km)", 300), ("distant DC\n(1500 km)", 1500)]
    radio, core, server = 4.0, 0.5, 1.0
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for i, (nm, km) in enumerate(cases):
        fib = 2 * km * 5e-3 * 1.0 + 0.2 * (km / 100) ** 0.5
        parts = [(radio, NAVY, "NR radio (both ways)"), (fib, ORANGE, "fibre + switching"), (core, GRAY, "core (UPF)"),
                 (server, GREEN, "server")]
        x = 0
        for v, c, lab in parts:
            ax.barh(i, v, left=x, color=c, height=0.55, label=lab if i == 0 else None); x += v
        ax.text(x + 0.2, i, f"{x:.1f} ms", va="center", fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels([c[0] for c in cases], fontsize=6.8); ax.invert_yaxis()
    ax.set_xlabel("illustrative round-trip time (ms)"); ax.set_xlim(0, 26)
    ax.legend(fontsize=6.3, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(); save(fig, "ch21_mec_latency")


# ---------------------------------------------------------------------------- growth decomposition
def growth_decomp():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    f = [("bandwidth: 200 kHz to ~400 MHz aggregated", 2000, NAVY),
         ("MIMO layers: 1 to 8", 8, GREEN), ("modulation: 1 to 8 bits/symbol", 8, ORANGE),
         ("coding, overhead, slots per user", 9.4, PURPLE)]
    x = 0
    for nm, v, c in f:
        ax.barh(0, np.log10(v), left=x, color=c, height=0.5)
        ax.text(x + np.log10(v) / 2, 0, f"x{v:g}", ha="center", va="center", fontsize=7.5, color="white", weight="bold")
        k_ = [i_ for i_, ff in enumerate(f) if ff[0] == nm][0]
        yl = 0.42 if k_ % 2 == 0 else -0.42
        ax.text(x + np.log10(v) / 2 + (0.3 if k_ == 3 else 0), yl, nm, ha="center", va="bottom" if yl > 0 else "top", fontsize=6.2, color=c)
        x += np.log10(v)
    ax.set_xlim(0, 6.6); ax.set_ylim(-0.75, 0.75)
    ax.set_xticks(range(7)); ax.set_xticklabels(["1", "10", "100", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax.set_yticks([]); ax.set_xlabel("growth factor of peak rate, GSM data slot to 5G (log scale; approximate)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch21_growth_decomp")


# ---------------------------------------------------------------------------- economics
def economics():
    yrs = np.arange(2010, 2025)
    traffic = 1.6 ** (yrs - 2010)
    revenue = np.ones_like(traffic, dtype=float)
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    ax.semilogy(yrs, traffic, color=NAVY, label="traffic (x1.6 per year)")
    ax.semilogy(yrs, revenue, color=GREEN, label="revenue (roughly flat)")
    ax.semilogy(yrs, revenue / traffic, color=ACCENT, ls="--", label="allowed cost per bit")
    ax.set_ylabel("index (2010 = 1)"); ax.set_xlabel("year")
    ax.legend(fontsize=6.4, loc="lower left"); ax.set_title("schematic", fontsize=7.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch21_economics")


# ---------------------------------------------------------------------------- flattening
def flattening():
    rows = [("GPRS (2.5G)", ["BTS", "BSC/PCU", "SGSN", "GGSN"], NAVY),
            ("UMTS R99 (3G)", ["Node B", "RNC", "SGSN", "GGSN"], GREEN),
            ("LTE (4G)", ["eNB", "S-GW", "P-GW"], ORANGE),
            ("NR SA (5G)", ["gNB\n(CU+DU)", "UPF"], ACCENT)]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, (nm, nodes, c) in enumerate(rows):
        y = 3 - i
        ax.text(-0.15, y, nm, ha="right", va="center", fontsize=7, color=c, weight="bold")
        _phone(ax, 0.15, y, 0.55, GRAY)
        for k, n in enumerate(nodes):
            x = 0.6 + k * 1.25
            _box(ax, x, y - 0.22, 1.0, 0.44, n, SOFT[["2G", "3G", "4G", "5G"][i]], ec=c, fs=6.5)
            _arrow(ax, (x - 0.25 if k else 0.3, y), (x, y), color=c, lw=0.8, ms=6)
        ax.text(0.6 + len(nodes) * 1.25, y, "internet", va="center", fontsize=6.5, color=GRAY)
        _arrow(ax, (0.6 + (len(nodes) - 1) * 1.25 + 1.0, y), (0.6 + len(nodes) * 1.25 - 0.05, y), color=c, lw=0.8, ms=6)
        hq = {0: "ARQ in BSC", 1: "ARQ in RNC", 2: "HARQ in eNB", 3: "HARQ in DU"}[i]
        ax.text(6.95, y, hq, va="center", fontsize=6.3, color="#555")
    ax.text(6.95, 3.6, "retransmission", fontsize=6.5, color=GRAY, weight="bold")
    ax.set_xlim(-2.0, 8.2); ax.set_ylim(-0.4, 3.8); ax.axis("off")
    save(fig, "ch21_flattening")


# ---------------------------------------------------------------------------- cell splitting
def cell_splitting():
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.2))
    for k, a in enumerate(axs):
        a.set_aspect("equal"); a.axis("off"); a.set_xlim(-5.2, 5.2); a.set_ylim(-5.0, 5.4)
    cols7 = [NAVY, BLUE2, GREEN, ORANGE, ACCENT, PURPLE, GRAY]
    titles = ["(a) R = 2 km: 7 cells", "(b) hot spot split: R/2", "(c) whole area at R/2: 4x the cells"]
    for k, a in enumerate(axs):
        a.set_title(titles[k], fontsize=8)
        R = 2.0
        for q in range(-2, 3):
            for r_ in range(-2, 3):
                x, y = _hexcenter(q, r_, R)
                if x * x + y * y > 3.6 ** 2:
                    continue
                if k == 0 or (k == 1 and (q, r_) != (0, 0)):
                    _hexagon(a, x, y, R, cols7[(q + 3 * r_) % 7], alpha=0.35)
        if k >= 1:
            n = 0
            for q in range(-6, 7):
                for r_ in range(-6, 7):
                    x, y = _hexcenter(q, r_, 1.0)
                    inside = (x * x + y * y < 2.0 ** 2) if k == 1 else (x * x + y * y < 4.6 ** 2)
                    if inside:
                        _hexagon(a, x, y, 1.0, cols7[(q + 3 * r_) % 7], alpha=0.5)
                        n += 1
    axs[0].text(0, -4.9, "capacity: 1x", ha="center", fontsize=7.5, color=NAVY, weight="bold")
    axs[1].text(0, -4.9, "extra capacity only where needed", ha="center", fontsize=7.0, color=NAVY, weight="bold")
    axs[2].text(0, -4.9, "capacity: ~4x, sites: ~4x", ha="center", fontsize=7.5, color=NAVY, weight="bold")
    fig.tight_layout(w_pad=0.2); save(fig, "ch21_cell_splitting")


# ---------------------------------------------------------------------------- GSM 900 band plan
def gsm_band():
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    for lo, hi, nm, c in [(890, 915, "uplink (phone transmits)\n890-915 MHz", NAVY), (935, 960, "downlink (base transmits)\n935-960 MHz", ACCENT)]:
        for k in range(125):
            f = lo + 0.2 * k
            ax.add_patch(Rectangle((f, 0.3), 0.18, 0.5 + 0.25 * (k % 2), fc=c, alpha=0.75, lw=0))
        ax.text((lo + hi) / 2, 1.25, nm, ha="center", va="bottom", fontsize=7, color=c)
    ax.annotate("", (935, 0.12), (890, 0.12), arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8))
    ax.text(912.5, -0.05, "45 MHz duplex spacing: carrier n up pairs with carrier n down", ha="center", va="top", fontsize=6.8)
    ax.add_patch(Rectangle((915, 0.3), 20, 0.75, fc="#F2F3F4", lw=0))
    ax.text(925, 0.67, "guard\n20 MHz", ha="center", va="center", fontsize=6.5, color=GRAY)
    ax.text(962, 0.55, "124 carriers x 200 kHz\neach direction,\n8 timeslots each", fontsize=6.6, va="center", color="#333")
    ax.set_xlim(886, 985); ax.set_ylim(-0.55, 1.75); ax.axis("off")
    save(fig, "ch21_gsm_band")


# ---------------------------------------------------------------------------- CDMA capacity
def cdma_capacity():
    eta = np.linspace(0, 0.95, 200)
    nr = -10 * np.log10(1 - eta)
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    for nu, ls, lab in [(1.0, "--", "no voice activity gain"), (0.4, "-", "voice activity 0.4")]:
        N = 1 + 128 / 10 ** 0.7 * eta / (nu * 1.6)
        ax.plot(nr, N, color=NAVY, ls=ls, label=lab)
    ax.axvline(6, color=ACCENT, lw=0.7, ls=":"); ax.text(6.2, 3, "6 dB noise rise", fontsize=6.5, color=ACCENT)
    ax.axhline(2, color=GRAY, lw=0.8); ax.text(0.2, 3.2, "AMPS: ~2 channels per sector\nin 1.25 MHz", fontsize=6.3, color=GRAY)
    ax.set_xlabel("noise rise (dB)"); ax.set_ylabel("users per carrier per sector")
    ax.set_xlim(0, 13); ax.set_ylim(0, 45); ax.legend(fontsize=6.4, loc="upper left")
    fig.tight_layout(); save(fig, "ch21_cdma_capacity")


# ---------------------------------------------------------------------------- NR peak rates (TS 38.306)
def nr_peak_bars():
    def rate(nl, qm, nprb, mu, oh, f=1.0):
        Ts = 1e-3 / (14 * 2 ** mu)
        return nl * qm * f * 948 / 1024 * 12 * nprb / Ts * (1 - oh) / 1e9
    cases = [("FR1 100 MHz, 30 kHz\n2 layers, 256QAM", rate(2, 8, 273, 1, 0.14)),
             ("FR1 100 MHz, 30 kHz\n4 layers, 256QAM", rate(4, 8, 273, 1, 0.14)),
             ("same, DDDSU TDD\n(~74% downlink)", 0.74 * rate(4, 8, 273, 1, 0.14)),
             ("FR2 400 MHz, 120 kHz\n2 layers, 64QAM", rate(2, 6, 264, 3, 0.18)),
             ("2 x FR2 + 1 x FR1\n(aggregated)", 2 * rate(2, 6, 264, 3, 0.18) + 0.74 * rate(4, 8, 273, 1, 0.14))]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    y = np.arange(len(cases))
    v = [c[1] for c in cases]
    ax.barh(y, v, color=[BLUE2, NAVY, NAVY, ACCENT, PURPLE], height=0.6)
    for i, x in enumerate(v):
        ax.text(x + 0.1, i, f"{x:.2f} Gb/s", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cases], fontsize=6.6); ax.invert_yaxis()
    ax.set_xlabel("approximate peak downlink rate from TS 38.306 (Gb/s)"); ax.set_xlim(0, 10)
    fig.tight_layout(); save(fig, "ch21_nr_peak_bars")
    print([round(x, 3) for x in v])


# ---------------------------------------------------------------------------- VoLTE header compression
def rohc():
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    rows = [("IPv4/UDP/RTP", 40), ("IPv6/UDP/RTP", 60), ("after ROHC", 3)]
    pay = 253 / 8 + 1
    for i, (nm, h) in enumerate(rows):
        ax.barh(i, h, color=ACCENT, height=0.55, label="headers" if i == 0 else None)
        ax.barh(i, pay, left=h, color=NAVY, height=0.55, label="AMR-WB speech (12.65 kb/s)" if i == 0 else None)
        ax.text(h + pay + 1.5, i, f"{100 * h / (h + pay):.0f}% overhead", va="center", fontsize=6.6)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("bytes per 20 ms voice packet"); ax.set_xlim(0, 130)
    ax.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch21_rohc")


# ---------------------------------------------------------------------------- LTE uplink resources
def lte_uplink():
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    nrb, nsym = 25, 14
    for t in range(nsym):
        for f in range(nrb):
            ax.add_patch(Rectangle((t, f), 0.95, 0.9, fc="#F2F3F4", lw=0))
    # PUCCH: edges, hopping at slot boundary
    for t in range(nsym):
        top = t < 7
        for f in ([0, 1] if top else [23, 24]):
            ax.add_patch(Rectangle((t, f), 0.95, 0.9, fc=ORANGE, lw=0))
        for f in ([23, 24] if top else [0, 1]):
            ax.add_patch(Rectangle((t, f), 0.95, 0.9, fc="#F5CBA7", lw=0))
    # PUSCH for two users
    for t in range(nsym):
        for f in range(3, 12):
            ax.add_patch(Rectangle((t, f), 0.95, 0.9, fc=NAVY if t not in (3, 10) else GREEN, alpha=0.85, lw=0))
        for f in range(13, 21):
            ax.add_patch(Rectangle((t, f), 0.95, 0.9, fc=BLUE2 if t not in (3, 10) else GREEN, alpha=0.85, lw=0))
    for f in range(2, 23):
        ax.add_patch(Rectangle((13, f), 0.95, 0.9, fc=PURPLE, lw=0))
    ax.axvline(7 - 0.025, color="k", lw=0.8)
    leg = [(ORANGE, "PUCCH (control), hopping between band edges"), (NAVY, "PUSCH user 1"), (BLUE2, "PUSCH user 2"),
           (GREEN, "DMRS: the midamble of each slot"), (PURPLE, "SRS: sounding in the last symbol")]
    for k, (c, t) in enumerate(leg):
        ax.add_patch(Rectangle((15.2, 21 - 4.5 * k), 0.8, 2.5, fc=c, lw=0))
        ax.text(16.3, 22.2 - 4.5 * k, t, fontsize=6.8, va="center")
    ax.set_xlim(0, 30); ax.set_ylim(0, 25); ax.set_xticks([3.5, 10.5]); ax.set_xticklabels(["slot 0", "slot 1"])
    ax.set_yticks([0.5, 24.5]); ax.set_yticklabels(["RB 0", "RB 24"]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, labelsize=7)
    ax.set_title("LTE uplink subframe, 5 MHz (illustrative allocation)", fontsize=8, loc="left")
    save(fig, "ch21_lte_uplink")


# ---------------------------------------------------------------------------- lessons scorecard
def scorecard():
    cols = ["radio", "architecture", "security", "ecosystem\n& scale", "timing &\nbusiness"]
    rows = ["1G", "2G", "3G", "4G", "5G"]
    # 2 = got it right, 1 = mixed, 0 = got it wrong (summary of the lessons section)
    v = [[2, 1, 0, 0, 2],
         [2, 0, 1, 2, 2],
         [2, 0, 2, 1, 0],
         [2, 2, 2, 2, 2],
         [2, 2, 2, 2, 1]]
    notes = [["FM + reuse", "analog switch", "cloning", "national\nsystems", "people paid"],
             ["GMSK, hopping", "circuit only", "secret A5,\none-way", "MoU, SIM,\nroaming", "SMS, prepaid"],
             ["WCDMA,\nturbo", "RNC far\nfrom radio", "mutual\nAKA", "WCDMA vs\nCDMA2000", "auction debt,\ndata late"],
             ["OFDMA, MIMO\n(CRS regret)", "flat all-IP", "AKA kept", "one global\nstandard", "met the\nsmartphone"],
             ["numerology,\nbeams", "cloud core,\nslicing", "SUCI", "global", "weak business\ncase"]]
    cmap = {2: "#D4EFDF", 1: "#FCF3CF", 0: "#FADBD8"}
    edge = {2: GREEN, 1: ORANGE, 0: ACCENT}
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for i, g in enumerate(rows):
        ax.text(-0.15, 4.5 - i, g, ha="right", va="center", fontsize=10, weight="bold", color=list(GEN_COL.values())[i])
        for j in range(5):
            ax.add_patch(FancyBboxPatch((j + 0.04, 4.06 - i), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=cmap[v[i][j]], ec=edge[v[i][j]], lw=0.8))
            ax.text(j + 0.5, 4.5 - i, notes[i][j], ha="center", va="center", fontsize=6.2, linespacing=1.05)
    for j, c in enumerate(cols):
        ax.text(j + 0.5, 5.1, c, ha="center", va="bottom", fontsize=7.2, color=NAVY, weight="bold")
    for k, (lab, key) in enumerate([("got it right", 2), ("mixed", 1), ("got it wrong", 0)]):
        ax.add_patch(Rectangle((5.25, 3.9 - 0.6 * k), 0.25, 0.3, fc=cmap[key], ec=edge[key], lw=0.8))
        ax.text(5.6, 4.05 - 0.6 * k, lab, fontsize=6.8, va="center")
    ax.set_xlim(-0.6, 6.5); ax.set_ylim(-0.05, 5.7); ax.axis("off")
    save(fig, "ch21_scorecard")


# ---------------------------------------------------------------------------- WCDMA power-control loops
def tpc_loop():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(-0.3, 3.2); ax.axis("off")
    _phone(ax, 0.6, 1.4, 1.4, NAVY)
    ax.text(0.6, 0.6, "phone: step power\nup or down 1 dB", ha="center", va="top", fontsize=6.6)
    _box(ax, 2.6, 1.85, 1.9, 0.6, "Node B: measure\nreceived SIR", "#D6E4F0", ec=NAVY, fs=6.8)
    _box(ax, 5.2, 1.85, 1.9, 0.6, "compare with\nSIR target", "#D6E4F0", ec=NAVY, fs=6.8)
    _box(ax, 5.2, 0.45, 1.9, 0.6, "send 1 TPC bit\nevery slot (1500 Hz)", "#FADBD8", ec=ACCENT, fs=6.8)
    _box(ax, 7.9, 1.85, 1.9, 0.6, "RNC outer loop:\nBLER vs 1% target", "#D4EFDF", ec=GREEN, fs=6.8)
    _arrow(ax, (0.95, 1.9), (2.55, 2.15), color=NAVY, lw=1.0)
    ax.text(1.75, 2.25, "uplink signal\n(through fading)", ha="center", fontsize=6.2, color=NAVY)
    _arrow(ax, (4.5, 2.15), (5.15, 2.15), color=NAVY)
    _arrow(ax, (6.15, 1.83), (6.15, 1.08), color=ACCENT)
    _arrow(ax, (5.15, 0.75), (0.95, 1.1), color=ACCENT)
    ax.text(2.4, 0.42, "downlink TPC command\n(inner loop, every 0.67 ms)", fontsize=6.4, color=ACCENT,
            ha="center", va="top")
    _arrow(ax, (8.85, 2.47), (6.6, 2.47), color=GREEN, rad=0.35)
    ax.text(7.7, 3.0, "adjusts the target slowly (tens of ms to s)", ha="center", fontsize=6.4, color=GREEN)
    save(fig, "ch21_tpc_loop")


# ---------------------------------------------------------------------------- IMT-2000 family
def imt2000_family():
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _box(ax, 0.6, 8.4, 9.1, 1.2, "IMT-2000 (ITU-R M.1457)\none family, five radio interfaces", NAVY,
         fs=6.8, color="white", weight="bold")
    items = [("IMT-DS: WCDMA\n(3GPP, UTRA FDD)", GREEN, True), ("IMT-MC: CDMA2000\n(3GPP2)", ORANGE, True),
             ("IMT-TC: UTRA TDD,\nTD-SCDMA (3GPP)", PURPLE, True), ("IMT-SC: UWC-136\n(EDGE-based)", GRAY, False),
             ("IMT-FT: DECT", GRAY, False)]
    for k, (t, c, big) in enumerate(items):
        y = 6.6 - 1.55 * k
        _box(ax, 3.3, y, 6.4, 1.25, t, SOFT["3G"] if c == GREEN else ("#FAE5D3" if c == ORANGE else
             ("#E8DAEF" if c == PURPLE else "#F2F3F4")), ec=c, fs=6.4, color="#222" if big else "#666")
        ax.plot([2.0, 3.3], [y + 0.62, y + 0.62], color=GRAY, lw=0.8)
    ax.plot([2.0, 2.0], [8.4, 6.6 - 1.55 * 4 + 0.62], color=GRAY, lw=0.8)
    ax.text(0.1, 3.5, "commercially\nimportant:\nthe first three", fontsize=6.0, color=NAVY, va="center")
    save(fig, "ch21_imt2000_family")


if __name__ == "__main__":
    import sys as _s
    fns = [growth, spectrum_bands, gsm_frames, gmsk, power_control, soft_handover, cqi_harq,
           lte_grid, nr_frame, peak_rates, imt_capabilities, latency,
           by_numbers, timeline, roads, what_fixed, imts_vs_cells, hex_reuse, amps_baseband, trunking, cloning, tdma_lights, midamble, timing_advance, call_flow, speech_bits, amr_modes, hopping, buzz217, sms_packing, near_far, edge_8psk, releases, ovsf_tree, handover_hyst, cell_breathing, rrc_states, pf_sched, eq_complexity, papr, harq_processes, lte_waterfall, carrier_agg, volte_capacity, usage_triangle, numerology, bwp, beam_sweep, slicing, mec_latency, growth_decomp, economics, flattening,
           cell_splitting, gsm_band, cdma_capacity, nr_peak_bars, rohc, lte_uplink, scorecard, tpc_loop, imt2000_family]
    sel = _s.argv[1:]
    for fn in fns:
        if not sel or fn.__name__ in sel:
            fn()
