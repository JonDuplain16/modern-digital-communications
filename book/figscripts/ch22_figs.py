"""Figures for Chapter 22: Wi-Fi, Bluetooth and the Internet of Things."""
from figstyle import *
from matplotlib.patches import Rectangle, Polygon
from commlib import iot

LBLUE = "#2E86C1"


# ----------------------------------------------------------------- spectrum
def _hump(ax, fc, bw, y0, h, color, alpha=0.35, lw=0.8, label=None):
    """Draw a flat-topped channel mask (trapezoid) centred at fc of width bw."""
    e = 0.06 * bw
    xs = [fc - bw / 2, fc - bw / 2 + e, fc + bw / 2 - e, fc + bw / 2]
    ys = [y0, y0 + h, y0 + h, y0]
    ax.fill(xs, ys, color=color, alpha=alpha, lw=0, label=label)
    ax.plot(xs, ys, color=color, lw=lw)


def fig_bands():
    fig = plt.figure(figsize=(W2, 5.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1, 1], hspace=0.62)
    # --- 2.4 GHz: Wi-Fi, BLE and 802.15.4
    ax = fig.add_subplot(gs[0])
    for ch in range(1, 14):
        fc = 2407 + 5 * ch
        main = ch in (1, 6, 11)
        _hump(ax, fc, 20, 2.0 if main else 2.0, 1.0, NAVY if main else GRAY,
              alpha=0.30 if main else 0.0, lw=1.0 if main else 0.35)
        if main:
            ax.text(fc, 3.12, f"Wi-Fi {ch}", ha="center", fontsize=7, color=NAVY)
    for k in range(40):
        fc = 2402 + 2 * k
        adv = fc in (2402, 2426, 2480)
        ax.add_patch(Rectangle((fc - 0.9, 1.0), 1.8, 0.65, color=ACCENT if adv else ORANGE,
                               alpha=0.9 if adv else 0.45, lw=0))
    ax.text(2402, 0.62, "37", ha="center", fontsize=6.5, color=ACCENT)
    ax.text(2426, 0.62, "38", ha="center", fontsize=6.5, color=ACCENT)
    ax.text(2480, 0.62, "39", ha="center", fontsize=6.5, color=ACCENT)
    for ch in range(11, 27):
        fc = 2405 + 5 * (ch - 11)
        clear = ch in (15, 20, 25, 26)
        ax.add_patch(Rectangle((fc - 1, 0.0), 2, 0.45, color=GREEN, alpha=0.9 if clear else 0.35, lw=0))
    for ch in (15, 20, 25, 26):
        ax.text(2405 + 5 * (ch - 11), -0.42, str(ch), ha="center", fontsize=6.5, color=GREEN)
    ax.text(2486.5, 2.4, "Wi-Fi 20 MHz\n(1, 6, 11 shaded)", fontsize=6.5, color=NAVY, va="center")
    ax.text(2486.5, 1.3, "BLE 2 MHz\n(advertising red)", fontsize=6.5, color=ACCENT, va="center")
    ax.text(2486.5, 0.2, "802.15.4\n5 MHz grid", fontsize=6.5, color=GREEN, va="center")
    ax.set_xlim(2395, 2510); ax.set_ylim(-0.6, 3.5); ax.set_yticks([])
    ax.axvspan(2400, 2483.5, color=GRAY, alpha=0.06)
    ax.set_title("2.4 GHz ISM band (2400--2483.5 MHz): three technologies, one band", fontsize=9)
    ax.set_xlabel("frequency (MHz)"); ax.grid(False)
    ax.spines["left"].set_visible(False)
    # --- 5 GHz
    ax = fig.add_subplot(gs[1])
    unii = [("U-NII-1", 5150, 5250, NAVY), ("U-NII-2A", 5250, 5350, ORANGE), ("U-NII-2C", 5470, 5725, ORANGE),
            ("U-NII-3", 5725, 5850, NAVY), ("U-NII-4", 5850, 5925, PURPLE)]
    for nm, a, b, c in unii:
        ax.add_patch(Rectangle((a, 3.05), b - a, 0.55, color=c, alpha=0.25, lw=0))
        ax.text((a + b) / 2, 3.32, nm, ha="center", va="center", fontsize=6.3, color=c)
    ch20 = list(range(36, 65, 4)) + list(range(100, 145, 4)) + list(range(149, 178, 4))
    for c in ch20:
        _hump(ax, 5000 + 5 * c, 20, 2.1, 0.6, NAVY, alpha=0.18, lw=0.4)
    for c in (38, 46, 54, 62, 102, 110, 118, 126, 134, 142, 151, 159, 167, 175):
        _hump(ax, 5000 + 5 * c, 40, 1.4, 0.55, LBLUE, alpha=0.18, lw=0.4)
    for c in (42, 58, 106, 122, 138, 155, 171):
        _hump(ax, 5000 + 5 * c, 80, 0.7, 0.55, GREEN, alpha=0.2, lw=0.5)
    for c in (50, 114, 163):
        _hump(ax, 5000 + 5 * c, 160, 0.0, 0.55, ACCENT, alpha=0.2, lw=0.5)
    for y, t in [(2.4, "20"), (1.67, "40"), (0.97, "80"), (0.27, "160")]:
        ax.text(5935, y, t + " MHz", fontsize=6.5, va="center")
    ax.text(5410, 1.7, "DFS required\n5250--5725 MHz", ha="center", fontsize=6.5, color=ORANGE)
    ax.set_xlim(5140, 5990); ax.set_ylim(-0.15, 3.7); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("5 GHz (United States): U-NII sub-bands and Wi-Fi channel plan", fontsize=9)
    ax.set_xlabel("frequency (MHz)")
    # --- 6 GHz
    ax = fig.add_subplot(gs[2])
    for nm, a, b in [("U-NII-5", 5925, 6425), ("U-NII-6", 6425, 6525), ("U-NII-7", 6525, 6875), ("U-NII-8", 6875, 7125)]:
        ax.add_patch(Rectangle((a, 3.05), b - a, 0.55, color=NAVY if nm in ("U-NII-5", "U-NII-7") else ORANGE,
                               alpha=0.25, lw=0))
        ax.text((a + b) / 2, 3.32, nm, ha="center", va="center", fontsize=6.3)
    for n in range(1, 234, 4):
        _hump(ax, 5950 + 5 * n, 20, 2.1, 0.6, NAVY, alpha=0.18, lw=0.3)
    for n in range(15, 234, 32):
        _hump(ax, 5950 + 5 * n, 160, 1.1, 0.7, GREEN, alpha=0.2, lw=0.5)
    for i, n in enumerate((31, 95, 159, 63, 127, 191)):
        y = 0.35 if i < 3 else 0.0
        _hump(ax, 5950 + 5 * n, 320, y, 0.3, ACCENT, alpha=0.25, lw=0.5)
    for y, t in [(2.4, "20 MHz (59)"), (1.45, "160 MHz (7)"), (0.33, "320 MHz (6, overlapping)")]:
        ax.text(7140, y, t, fontsize=6.5, va="center")
    ax.set_xlim(5900, 7520); ax.set_ylim(-0.15, 3.7); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("6 GHz (United States, 5925--7125 MHz): standard power with AFC in U-NII-5/7", fontsize=9)
    ax.set_xlabel("frequency (MHz)")
    save(fig, "ch22_bands")


# ----------------------------------------------------------------- preamble
def fig_preamble_sync():
    r = rng(3)
    stf, ltf = iot.wifi_legacy_preamble()
    pre = np.concatenate([stf, ltf])
    data = (r.standard_normal(320) + 1j * r.standard_normal(320)) / np.sqrt(2)  # stand-in for SIG/data
    pkt = np.concatenate([pre, data])
    h = np.array([1, 0, 0.45 * np.exp(1j * 1.1), 0, 0, 0.2j])
    lead = 200
    x = np.concatenate([np.zeros(lead), np.convolve(pkt, h)[:len(pkt)], np.zeros(100)])
    cfo = 80e3                                    # Hz, i.e. 16 ppm at 5 GHz
    n = np.arange(len(x))
    snr_db = 10
    x = x * np.exp(2j * np.pi * cfo / 20e6 * n)
    x = x + 10 ** (-snr_db / 20) * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x))) / np.sqrt(2)
    m, P = iot.delay_correlate(x, 16, 48)
    t = np.arange(len(x)) / 20.0               # microseconds
    fig, ax = plt.subplots(3, 1, figsize=(W1, 4.6), sharex=True)
    ax[0].plot(t, np.abs(x), color=NAVY, lw=0.6)
    for a, b, lab, c in [(lead, lead + 160, "L-STF", GREEN), (lead + 160, lead + 320, "L-LTF", ORANGE),
                         (lead + 320, lead + 400, "L-SIG / data", GRAY)]:
        ax[0].axvspan(a / 20, b / 20, color=c, alpha=0.12)
        ax[0].text((a + b) / 40, 2.35, lab, ha="center", fontsize=7, color="k")
    ax[0].set_ylabel("$|r(t)|$"); ax[0].set_ylim(0, 2.7)
    ax[0].set_title("Received 802.11 legacy preamble: multipath, 80 kHz CFO, SNR 10 dB", fontsize=9)
    tm = np.arange(len(m)) / 20.0
    ax[1].plot(tm, m, color=NAVY)
    ax[1].axhline(0.5, color=ACCENT, ls="--", lw=0.8)
    ax[1].text(1, 0.56, "detection threshold", fontsize=7, color=ACCENT)
    ax[1].set_ylabel("$|P|^2/R^2$"); ax[1].set_ylim(0, 1.1)
    ax[1].set_title("Delay-16 autocorrelation (packet detection, AGC window)", fontsize=9)
    # coarse CFO from STF and fine from LTF
    k0 = lead + 40
    cfo_hat = -np.angle(P[k0 + 40]) / (2 * np.pi * 16) * 20e6
    # LTF cross-correlation with the known 64-sample symbol (after coarse correction)
    xc = x * np.exp(-2j * np.pi * cfo_hat / 20e6 * n)
    L = np.zeros(64, complex)
    for k, v in zip(range(-26, 27), iot.wifi_ltf_freq()):
        L[k % 64] = v
    l = np.fft.ifft(L) * 64 / np.sqrt(52)
    xcorr = np.abs(np.convolve(xc, np.conj(l[::-1]), "valid")) / 64
    ax[2].plot(np.arange(len(xcorr)) / 20.0, xcorr, color=GREEN)
    ax[2].set_ylabel("|xcorr|"); ax[2].set_xlabel(r"time ($\mu$s)")
    ax[2].set_title(f"Cross-correlation with the known L-LTF symbol (fine timing); coarse CFO estimate {cfo_hat/1e3:.1f} kHz",
                    fontsize=9)
    ax[2].set_xlim(0, t[-1])
    fig.tight_layout(); save(fig, "ch22_preamble_sync")
    return cfo_hat


# ----------------------------------------------------------------- rate vs range
# approximate SNR (dB) needed for ~10% PER with 1500-byte packets, typical published receiver
# sensitivity spacing; MCS 0..13 (BPSK 1/2 ... 4096-QAM 5/6)
SNR_REQ = np.array([2, 5, 9, 11, 15, 18, 20, 25, 29, 31, 34, 37, 40, 43])
BITS = np.array([0.5, 1, 1.5, 2, 3, 4, 4.5, 5, 6, 6.67, 7.5, 8.33, 9, 10])   # info bits per subcarrier
NSD = {20: 234, 40: 468, 80: 980, 160: 1960, 320: 3920}


def rate_mbps(mcs, bw, nss=1, gi=0.8):
    return NSD[bw] * BITS[mcs] * nss / (12.8 + gi)


def fig_rate_range():
    d = np.logspace(0, np.log10(120), 400)
    pt, gtr = 20.0, 3.0                       # dBm, combined antenna gains dB
    f = 5.5e9
    pl = 20 * np.log10(4 * np.pi * f / 3e8) + 35 * np.log10(d)      # 1 m free space + n = 3.5
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.65))
    for bw, c in [(20, NAVY), (80, GREEN), (160, ACCENT)]:
        noise = -174 + 10 * np.log10(bw * 1e6) + 7          # 7 dB noise figure
        snr = pt + gtr - pl - noise
        idx = np.searchsorted(SNR_REQ, snr, side="right") - 1
        rate = np.where(idx >= 0, rate_mbps(np.clip(idx, 0, 13), bw, nss=2), 0)
        ax[0].plot(d, rate, color=c, label=f"{bw} MHz", drawstyle="steps-post")
        if bw == 20:
            ax[1].plot(d, snr, color=c, label="SNR, 20 MHz")
        ax[1].plot(d, snr, color=c, lw=0.8 if bw != 20 else 1.4, ls="-" if bw == 20 else "--")
    ax[0].set_xscale("log"); ax[0].set_xlabel("distance (m)"); ax[0].set_ylabel("PHY rate, 2 streams (Mb/s)")
    ax[0].set_title("Rate versus range (indoor, $n=3.5$)", fontsize=9); ax[0].legend(fontsize=7)
    ax[0].set_ylim(0, 3200)
    for m in (0, 4, 7, 9, 11, 13):
        ax[1].axhline(SNR_REQ[m], color=GRAY, lw=0.5, ls=":")
        ax[1].text(105, SNR_REQ[m] + 0.6, f"MCS {m}", fontsize=6, color=GRAY, ha="right")
    ax[1].set_xscale("log"); ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("SNR (dB)")
    ax[1].set_title("SNR (solid 20 MHz; dashed 80, 160 MHz)", fontsize=9)
    ax[1].set_ylim(-5, 70)
    fig.tight_layout(); save(fig, "ch22_rate_range")


# ----------------------------------------------------------------- Minstrel
def _per(snr, mcs):
    return 1 / (1 + np.exp(1.6 * (snr - SNR_REQ[mcs])))      # logistic waterfall, ~10% at threshold - 1.4 dB


def fig_minstrel():
    r = rng(7)
    T = 6000                                    # 1 ms per packet-slot
    t = np.arange(T) / 1000
    # slow walk (user moving) + fast Rician-like fading
    slow = 24 + 9 * np.sin(2 * np.pi * t / 6.0 + 0.6) + np.cumsum(r.standard_normal(T)) * 0.03
    fast_c = np.convolve(r.standard_normal(T) + 1j * r.standard_normal(T), np.ones(60) / np.sqrt(60), "same") / np.sqrt(2)
    K = 4.0
    fast = np.abs(np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * fast_c) ** 2
    snr = slow + 10 * np.log10(fast)
    mcs_set = np.arange(12)
    thr_rate = rate_mbps(mcs_set, 80)          # 80 MHz, 1 stream
    # oracle
    exp_tp = (1 - _per(snr[:, None], mcs_set[None, :])) * thr_rate[None, :]
    oracle = exp_tp.max(axis=1)
    # Minstrel-like: EWMA success prob per rate, updated every 100 ms; 10% lookaround
    prob = np.full(12, 0.5); succ = np.zeros(12); att = np.zeros(12)
    cur = 0; chosen = np.zeros(T, int); got = np.zeros(T)
    for k in range(T):
        if r.random() < 0.1:
            m = r.integers(0, 12)
        else:
            m = cur
        ok = r.random() > _per(snr[k], m)
        att[m] += 1; succ[m] += ok
        chosen[k] = m; got[k] = thr_rate[m] * ok
        if k % 100 == 99:
            upd = att > 0
            prob[upd] = 0.75 * prob[upd] + 0.25 * succ[upd] / att[upd]
            succ[:] = 0; att[:] = 0
            cur = int(np.argmax(prob * thr_rate * (prob > 0.1)))
    fixed = (1 - _per(snr, 6)) * thr_rate[6]
    w = 100
    sm = lambda v: np.convolve(v, np.ones(w) / w, "same")
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4]))
    ax[0].plot(t, snr, color=GRAY, lw=0.4)
    ax[0].set_ylabel("SNR (dB)"); ax[0].set_title("Channel seen by a moving station (80 MHz, one stream)", fontsize=9)
    ax[1].plot(t, sm(oracle), color=GREEN, label=f"oracle (mean {oracle.mean():.0f} Mb/s)")
    ax[1].plot(t, sm(got), color=NAVY, label=f"Minstrel-like (mean {got.mean():.0f} Mb/s)")
    ax[1].plot(t, sm(fixed), color=ACCENT, lw=1.0, label=f"fixed MCS 6 (mean {fixed.mean():.0f} Mb/s)")
    ax[1].set_ylabel("goodput (Mb/s, 0.1 s avg)"); ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=7, loc="lower left", ncol=1)
    ax[1].set_xlim(0, 6)
    fig.tight_layout(); save(fig, "ch22_minstrel")
    return oracle.mean(), got.mean(), fixed.mean()


# ----------------------------------------------------------------- MAC efficiency
def fig_mac_efficiency():
    rates = np.logspace(np.log10(6), np.log10(5000), 120)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    res = {}
    for n, c, lab in [(1, ACCENT, "single MPDU + ACK"), (8, ORANGE, "A-MPDU, 8 MPDUs"),
                      (64, GREEN, "A-MPDU, 64"), (256, NAVY, "A-MPDU, 256")]:
        out = np.array([iot.mac_exchange(R, n_agg=n) for R in rates])
        ax[0].loglog(rates, out[:, 0], color=c, label=lab)
        ax[1].semilogx(rates, 100 * out[:, 1], color=c, label=lab)
        res[n] = out
    ax[0].loglog(rates, rates, color=GRAY, ls=":", lw=0.8, label="PHY rate")
    cap = iot.mac_exchange(1e7, n_agg=1)[0]
    ax[0].axhline(cap, color=ACCENT, lw=0.6, ls="--")
    ax[0].text(7, cap * 1.25, f"ceiling {cap:.0f} Mb/s", fontsize=7, color=ACCENT)
    ax[0].set_xlabel("PHY rate (Mb/s)"); ax[0].set_ylabel("MAC throughput (Mb/s)")
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_title("Throughput, 1500-byte packets", fontsize=9)
    ax[1].set_xlabel("PHY rate (Mb/s)"); ax[1].set_ylabel("MAC efficiency (%)"); ax[1].set_ylim(0, 100)
    ax[1].set_title("Efficiency = throughput / PHY rate", fontsize=9)
    for r0 in (54, 600, 2400):
        ax[1].axvline(r0, color=GRAY, lw=0.5, ls=":")
    ax[1].text(54 * 1.08, 4, "54", fontsize=6.5, color=GRAY)
    ax[1].text(600 * 1.08, 4, "600", fontsize=6.5, color=GRAY)
    ax[1].text(2400 * 1.08, 4, "2400", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch22_mac_efficiency")
    return cap


# ----------------------------------------------------------------- RU tiling and puncturing
def fig_ru_tiling():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.5), gridspec_kw=dict(height_ratios=[1.35, 1]))
    a = ax[0]
    # 20 MHz HE: tone indices -128..127 (256 tones). Approximate RU layouts (with guards/DC)
    rows = [("26-tone RUs (9)", 26, 9, NAVY), ("52-tone (4) + 26", 52, 4, GREEN),
            ("106-tone (2) + 26", 106, 2, ORANGE), ("242-tone (1)", 242, 1, ACCENT)]
    for i, (lab, size, n, c) in enumerate(rows):
        y = 3 - i
        # layout: left half n/2 RUs, centre 26 (except 26x9 which has the 5th RU in centre, and 242)
        if size == 242:
            blocks = [(-121, 242)]
        elif size == 26:
            starts = [-121, -95, -68, -42, -13, 16, 42, 69, 95]
            blocks = [(s, 26) for s in starts]
        else:
            half = n // 2
            left = [-122 + k * size for k in range(half)]
            right = [122 - (k + 1) * size for k in range(half)][::-1]
            blocks = [(s, size) for s in left] + [(-13, 26)] + [(s, size) for s in right]
        for s, w in blocks:
            col = c if w == size else GRAY
            a.add_patch(Rectangle((s, y - 0.38), w - 1.2, 0.76, color=col, alpha=0.55, lw=0))
        a.text(-140, y, lab, ha="right", va="center", fontsize=7)
    a.axvline(0, color="k", lw=0.5, ls=":")
    a.set_xlim(-135, 135); a.set_ylim(-0.6, 3.6); a.set_yticks([]); a.grid(False)
    a.set_xlabel("subcarrier index (78.125 kHz spacing)")
    a.set_title("802.11ax resource units in one 20 MHz channel (256 tones)", fontsize=9)
    a.spines["left"].set_visible(False)
    # puncturing
    a = ax[1]
    occ = [1, 1, 0, 1]   # 80 MHz with one 20 MHz subchannel punctured
    for k, o in enumerate(occ):
        x0 = 5490 + 20 * k
        if o:
            a.add_patch(Rectangle((x0 + 0.5, 0.0), 19, 1.0, color=NAVY, alpha=0.5, lw=0))
            a.text(x0 + 10, 0.5, "EHT PPDU", ha="center", va="center", fontsize=7, color="white")
        else:
            a.add_patch(Rectangle((x0 + 2, 0.0), 16, 0.7, color=ACCENT, alpha=0.5, lw=0))
            a.text(x0 + 10, 0.35, "incumbent\n(radar / other BSS)", ha="center", va="center", fontsize=6.3)
            a.text(x0 + 10, 1.12, "punctured", ha="center", fontsize=7, color=ACCENT)
    a.annotate("", (5490, 1.3), (5570, 1.3), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    a.text(5500, 1.36, "80 MHz channel", fontsize=7, color=GRAY)
    a.set_xlim(5470, 5590); a.set_ylim(-0.1, 1.6); a.set_yticks([]); a.grid(False)
    a.spines["left"].set_visible(False)
    a.set_xlabel("frequency (MHz)")
    a.set_title("Preamble puncturing: an 80 MHz transmission with a 20 MHz hole", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_ru_tiling")


# ----------------------------------------------------------------- BLE GFSK
def fig_ble_gfsk():
    r = rng(5)
    bits = r.integers(0, 2, 2000)
    sps = 16
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    x1, f1 = iot.gfsk_mod(bits, sps=sps, bt=0.5, h=0.5)
    nshow = 16
    tt = np.arange(nshow * sps) / sps
    ax[0].step(np.arange(nshow + 1), np.r_[2 * bits[:nshow] - 1, 2 * bits[nshow - 1] - 1] * 0.25, where="post",
               color=GRAY, lw=0.8, label="NRZ $\\times h/2$")
    ax[0].plot(tt, f1[:nshow * sps], color=NAVY, label="Gaussian, $BT=0.5$")
    ax[0].set_xlabel("time (bits)"); ax[0].set_ylabel("frequency ($\\times R_s$)")
    ax[0].set_title("BLE 1M: frequency trajectory", fontsize=9); ax[0].legend(fontsize=6.0, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.6)
    ax[0].set_ylim(-0.4, 0.62)
    # spectra: 1M and 2M on same absolute axis (MHz), sampled at 16 MS/s
    fs = 16.0
    for rs, c, lab in [(1, NAVY, "LE 1M"), (2, ACCENT, "LE 2M")]:
        sp = int(fs / rs)
        x, _ = iot.gfsk_mod(r.integers(0, 2, 8000 // rs), sps=sp, bt=0.5, h=0.5)
        from scipy.signal import welch
        fr, p = welch(x, fs=fs, nperseg=1024, return_onesided=False)
        idx = np.argsort(fr)
        ax[1].plot(fr[idx], 10 * np.log10(p[idx] / p.max()), color=c, label=lab)
    ax[1].set_xlim(-4, 4); ax[1].set_ylim(-60, 3)
    ax[1].axvspan(-1, 1, color=GRAY, alpha=0.1)
    ax[1].set_xlabel("frequency (MHz)"); ax[1].set_ylabel("PSD (dB)")
    ax[1].set_title("Spectra (2 MHz channel shaded)", fontsize=9); ax[1].legend(fontsize=6.5)
    # discriminator eye diagram with noise
    xn = x1 + 10 ** (-20 / 20) * (r.standard_normal(len(x1)) + 1j * r.standard_normal(len(x1))) / np.sqrt(2)
    fd = iot.fm_discriminator(xn) * sps
    fd = np.convolve(fd, np.ones(4) / 4, "same")
    for k in range(40, 240):
        seg = fd[k * sps + sps // 2: k * sps + sps // 2 + 2 * sps]
        if len(seg) == 2 * sps:
            ax[2].plot(np.arange(2 * sps) / sps, seg, color=NAVY, alpha=0.12, lw=0.6)
    ax[2].set_xlabel("time (bits)"); ax[2].set_ylabel("discriminator out")
    ax[2].set_title("Discriminator eye, SNR 20 dB", fontsize=9); ax[2].set_ylim(-0.5, 0.5)
    fig.tight_layout(); save(fig, "ch22_ble_gfsk")


# ----------------------------------------------------------------- BLE battery
def ble_adv_charge(payload=31, i_tx=6.0e-3, i_cpu=3.0e-3, t_cpu=0.4e-3, ramp=0.15e-3, rate=1e6, n_ch=3):
    """Charge (C) per advertising event: n_ch packets (preamble 1 + AA 4 + header 2 + AdvA 6 + data + CRC 3)."""
    nbytes = 1 + 4 + 2 + 6 + payload + 3
    t_pkt = nbytes * 8 / rate
    return n_ch * (t_pkt + ramp) * i_tx + t_cpu * i_cpu, t_pkt


def fig_ble_battery():
    cap_mah = 225.0
    i_sleep = 2e-6
    intervals = np.logspace(np.log10(0.02), np.log10(10.24), 200)
    q_adv, t_pkt = ble_adv_charge()
    # connection event: one empty packet exchange each way (80 us each on 1M) + 150 us IFS + ramps, RX window widening
    q_conn = (2 * 80e-6 + 150e-6 + 2 * 0.15e-3) * 6e-3 + 0.4e-3 * 3e-3
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    for q, c, lab in [(q_adv, NAVY, "advertising, 31 B, 3 channels"), (q_conn, GREEN, "connection, empty packets")]:
        iavg = q / intervals + i_sleep
        life_y = cap_mah * 1e-3 / iavg / 24 / 365
        ax[0].loglog(intervals, iavg * 1e6, color=c, label=lab)
        ax[1].loglog(intervals, life_y, color=c, label=lab)
    ax[0].axhline(i_sleep * 1e6, color=GRAY, ls=":", lw=0.8)
    ax[0].text(0.025, i_sleep * 1e6 * 1.2, "sleep floor 2 $\\mu$A", fontsize=6.5, color=GRAY)
    ax[0].set_xlabel("event interval (s)"); ax[0].set_ylabel(r"average current ($\mu$A)")
    ax[0].set_title("Average current", fontsize=9); ax[0].legend(fontsize=6.5)
    ax[1].axhline(10, color=ACCENT, ls="--", lw=0.7)
    ax[1].text(0.025, 11.5, "10 years (self-discharge dominates)", fontsize=6.5, color=ACCENT)
    ax[1].set_xlabel("event interval (s)"); ax[1].set_ylabel("CR2032 life (years)")
    ax[1].set_title("Battery life, 225 mAh", fontsize=9)
    ax[1].set_ylim(0.01, 30)
    fig.tight_layout(); save(fig, "ch22_ble_battery")
    return q_adv, t_pkt, q_conn


# ----------------------------------------------------------------- 802.15.4 O-QPSK
def fig_oqpsk154():
    tab = iot.ieee802154_chips()
    sps = 16
    x = iot.oqpsk_halfsine(tab[[3, 9]].ravel(), sps=sps)
    t = np.arange(len(x)) / sps
    fig = plt.figure(figsize=(W2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 1.1], wspace=0.45)
    a = fig.add_subplot(gs[0])
    a.plot(t, x.real + 1.4, color=NAVY, lw=0.9)
    a.plot(t, x.imag - 1.4, color=ACCENT, lw=0.9)
    a.plot(t, np.abs(x) * 0 + 0, color="w", lw=0)
    a.text(-4.5, 1.4, "I", fontsize=8, color=NAVY, va="center"); a.text(-4.5, -1.4, "Q", fontsize=8, color=ACCENT, va="center")
    a.axvline(32, color=GRAY, ls=":", lw=0.7)
    a.text(16, 2.65, "symbol 3", ha="center", fontsize=7); a.text(48, 2.65, "symbol 9", ha="center", fontsize=7)
    a.set_xlim(-6, 66); a.set_ylim(-2.7, 3.0); a.set_yticks([])
    a.set_xlabel("time (chips, $T_c=0.5\\,\\mu$s)")
    a.set_title("Half-sine O-QPSK: Q offset by one chip", fontsize=9)
    a2 = fig.add_subplot(gs[1])
    mid = x[2 * sps:-2 * sps]
    a2.plot(mid.real, mid.imag, color=NAVY, lw=0.6)
    a2.set_aspect("equal"); a2.set_xlim(-1.3, 1.3); a2.set_ylim(-1.3, 1.3)
    a2.set_title("constant envelope\n(it is MSK)", fontsize=9)
    a2.set_xticks([-1, 0, 1]); a2.set_yticks([-1, 0, 1])
    a3 = fig.add_subplot(gs[2])
    b = 2 * tab - 1
    C = b @ b.T / 32
    im = a3.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    a3.set_title("chip correlation", fontsize=9); a3.set_xlabel("symbol"); a3.set_ylabel("symbol")
    a3.set_xticks([0, 5, 10, 15]); a3.set_yticks([0, 5, 10, 15]); a3.grid(False)
    cb = fig.colorbar(im, ax=a3, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6.5)
    save(fig, "ch22_oqpsk154")
    off = C[~np.eye(16, dtype=bool)]
    return off.max(), off.min()


# ----------------------------------------------------------------- LoRa
def fig_lora_chirps():
    sf, osr = 7, 4
    M = 2 ** sf
    pre = iot.lora_symbols(sf, [0] * 8, osr)
    sync = iot.lora_symbols(sf, [24, 32], osr)
    sfd = np.concatenate([iot.lora_symbols(sf, [0, 0], osr, down=True), iot.lora_symbols(sf, [0], osr, down=True)[:M * osr // 4]])
    data = iot.lora_symbols(sf, [5, 100, 64, 23, 127, 90], osr)
    x = np.concatenate([pre, sync, sfd, data])
    from scipy.signal import spectrogram
    f, t, S = spectrogram(x, fs=osr, nperseg=64, noverlap=56, return_onesided=False, detrend=False)
    idx = np.argsort(f)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax[0].pcolormesh(t / M, f[idx], 10 * np.log10(S[idx] + 1e-6), cmap="Blues", shading="auto", vmin=-25)
    ax[0].set_ylim(-0.6, 0.6); ax[0].set_xlabel("time (symbols, $2^{SF}/B$ each)")
    ax[0].set_ylabel("frequency ($\\times B$)")
    for xx, lab in [(4, "preamble up-chirps"), (9, "sync"), (11.1, "SFD"), (15.2, "data symbols")]:
        ax[0].text(xx, 0.54, lab, ha="center", fontsize=6.5)
    ax[0].set_title("A LoRa frame (SF7)", fontsize=9); ax[0].grid(False)
    # dechirp of one data symbol with noise
    r = rng(2)
    s = iot.lora_symbols(sf, [100])
    y = s + 10 ** (7.5 / 20) * (r.standard_normal(M) + 1j * r.standard_normal(M)) / np.sqrt(2)   # SNR -7.5 dB
    d, X = iot.lora_demod(y, sf)
    ax[1].plot(np.arange(M), X[0] ** 2 / M, color=NAVY, lw=0.8)
    ax[1].annotate(f"symbol {d[0]}", (d[0], X[0, d[0]] ** 2 / M), (-55, -6), textcoords="offset points",
                   fontsize=7, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax[1].set_xlabel("FFT bin"); ax[1].set_ylabel("$|Y[k]|^2/M$")
    ax[1].set_title("Dechirp + FFT at SNR $-7.5$ dB", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_lora_chirps")


def fig_lora_ser():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    cols = [NAVY, LBLUE, GREEN, ORANGE, ACCENT, PURPLE]
    req = {}
    for sf, c in zip(range(7, 13), cols):
        snr = np.linspace(-26, -2, 60)
        th = iot.lora_ser_theory(sf, snr)
        ax[0].semilogy(snr, th, color=c, label=f"SF{sf}")
        req[sf] = np.interp(np.log10(1e-3), np.log10(th[::-1] + 1e-30), snr[::-1])
        pts = np.arange(round(req[sf]) - 3, round(req[sf]) + 2, 1.0)
        n = 4000 if sf <= 9 else 1500
        sim = [iot.lora_ser_sim(sf, p, n, rng=sf * 10 + int(p)) for p in pts]
        sim = np.array(sim); ok = sim > 0
        ax[0].semilogy(pts[ok], sim[ok], "o", ms=2.8, color=c)
    ax[0].set_ylim(1e-4, 1); ax[0].set_xlim(-26, -2)
    ax[0].set_xlabel("SNR in bandwidth $B$ (dB)"); ax[0].set_ylabel("symbol error rate")
    ax[0].set_title("Dechirp--FFT detector, AWGN (dots: simulation)", fontsize=9)
    ax[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    sfs = np.arange(7, 13)
    ax[1].plot(sfs, [req[s] for s in sfs], "o-", color=NAVY, label="SER $10^{-3}$, this model")
    ax[1].plot(sfs, [iot.LORA_SNR_REQ[s] for s in sfs], "s--", color=ACCENT, label="datasheet demod. SNR")
    for s_ in sfs:
        ax[1].annotate(f"{iot.lora_bitrate(s_):.0f} b/s", (s_, req[s_]), (-40, -10), textcoords="offset points",
                       fontsize=6.3, color=GREEN)
    ax[1].set_xlabel("spreading factor"); ax[1].set_ylabel("required SNR (dB)")
    ax[1].set_title("Each SF step: about $-2.5$ dB SNR, half the rate", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper right"); ax[1].set_xlim(6.7, 12.8); ax[1].set_ylim(-24, -6)
    fig.tight_layout(); save(fig, "ch22_lora_ser")
    return req


def fig_lora_budget():
    sfs = np.arange(7, 13)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    for pl, c in [(10, GREEN), (20, NAVY), (51, ACCENT)]:
        toa = [iot.lora_time_on_air(pl, s) * 1e3 for s in sfs]
        ax[0].semilogy(sfs, toa, "o-", color=c, label=f"{pl}-byte payload")
    ax[0].axhline(400, color=GRAY, ls=":", lw=0.8)
    ax[0].text(7.1, 470, "US915 400 ms dwell limit", fontsize=6.5, color=GRAY)
    ax[0].set_xlabel("spreading factor"); ax[0].set_ylabel("time on air (ms)")
    ax[0].set_title("Time on air, 125 kHz, CR 4/5", fontsize=9); ax[0].legend(fontsize=6.5)
    # range: 14 dBm EIRP, 3 dBi gateway antenna, 10 dB margin, two path-loss models at 868 MHz
    f = 868e6
    pl0 = 20 * np.log10(4 * np.pi * f / 3e8)
    for n, c, lab in [(2.8, NAVY, "suburban, $n=2.8$"), (3.5, ACCENT, "urban, $n=3.5$")]:
        mapl = 14 + 3 - np.array([iot.lora_sensitivity_dbm(s) for s in sfs]) - 10
        dkm = 10 ** ((mapl - pl0) / (10 * n)) / 1e3
        ax[1].semilogy(sfs, dkm, "o-", color=c, label=lab)
    ax[1].set_xlabel("spreading factor"); ax[1].set_ylabel("range (km)")
    ax[1].set_title("Range: 14 dBm, 10 dB fade margin", fontsize=9); ax[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch22_lora_budget")


# ----------------------------------------------------------------- RFID
def fig_rfid_range():
    d = np.linspace(0.3, 30, 400)
    f = 915e6
    lam = 3e8 / f
    eirp = 36.0
    pt_tag = eirp + 2.15 - 3 - 20 * np.log10(4 * np.pi * d / lam)   # 3 dB polarisation loss
    pr = iot.rfid_backscatter_dbm(d, eirp, f)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    ax.plot(d, pt_tag, color=NAVY, label="power at the tag (forward link, 3 dB pol. loss)")
    ax.plot(d, pr, color=ACCENT, label="backscatter at reader (reverse link)")
    for th, ls, lab in [(-15, "--", "older tag IC, $-15$ dBm"), (-22, "-.", "modern tag IC, $-22$ dBm")]:
        ax.axhline(th, color=NAVY, ls=ls, lw=0.7)
        dmax = lam / (4 * np.pi) * 10 ** ((eirp + 2.15 - 3 - th) / 20)
        ax.plot([dmax], [th], "o", color=NAVY, ms=3.5)
        ax.text(1.6, th + 1.3, lab + f" $\\to$ {dmax:.1f} m", fontsize=6.8, color=NAVY)
    ax.axhline(-85, color=ACCENT, ls="--", lw=0.7)
    ax.text(0.45, -82.5, "reader sensitivity $\\approx -85$ dBm", fontsize=6.8, color=ACCENT)
    ax.set_xscale("log"); ax.set_xlabel("distance (m)"); ax.set_ylabel("power (dBm)")
    ax.set_xlim(0.3, 30); ax.set_ylim(-100, 25)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.set_title("UHF RFID at 915 MHz, 4 W EIRP: the forward link limits range", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_rfid_range")


# =====================================================================================
# Second-edition concept illustrations and extra data figures
# =====================================================================================
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, FancyArrowPatch, Ellipse

PW, PH = 3.1, 2.4          # size of one panel of a side-by-side pair


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_unlicensed_timeline():
    ev = [(1947, "ISM bands set aside\n(Atlantic City)", NAVY, 1),
          (1985, "FCC: licence-free\nspread spectrum", ACCENT, -1),
          (1990, "NCR\nWaveLAN", GRAY, 2),
          (1997, "802.11 (2 Mb/s);\nU-NII 5 GHz", NAVY, 1),
          (1999, "11b, 11a; 'Wi-Fi';\nBluetooth 1.0", GREEN, -2),
          (2003, "11g;\n802.15.4", NAVY, -1),
          (2009, "11n\n(MIMO)", NAVY, 1),
          (2010, "Bluetooth\nLE", GREEN, -2),
          (2013, "11ac", NAVY, 2),
          (2015, "LoRa\nAlliance", ORANGE, -1),
          (2016, "NB-IoT,\nLTE-M", PURPLE, 1),
          (2020, "6 GHz opened\n(1200 MHz)", ACCENT, -2),
          (2022, "Matter\n1.0", ORANGE, 2),
          (2024, "Wi-Fi 7; AFC;\nBT Channel\nSounding", NAVY, -1)]
    X = lambda y: y if y >= 1985 else 1985 - (1985 - y) * 0.22
    fig, ax = plt.subplots(figsize=(W2, 2.7))
    ax.axhline(0, color=NAVY, lw=2)
    for y, tx, c, s in ev:
        h = {1: 0.45, 2: 1.25, -1: -0.45, -2: -1.25}[s]
        x = X(y)
        ax.plot([x, x], [0, h], color=c, lw=0.8)
        ax.plot(x, 0, "o", color=c, ms=4)
        ax.text(x, h + (0.06 if s > 0 else -0.06), f"{y}\n{tx}" if s > 0 else f"{tx}\n{y}",
                ha="center", va="bottom" if s > 0 else "top", fontsize=6.4, color=c)
    ax.axvspan(X(1947), 1985, color=GRAY, alpha=0.07)
    ax.text((X(1947) + 1985) / 2 - 1, -1.0, "the 'garbage\nband' years\n(compressed)", ha="center", va="top",
            fontsize=6.5, color=GRAY, style="italic")
    ax.set_xlim(X(1947) - 3, 2028); ax.set_ylim(-2.35, 2.35); _clean(ax)
    fig.tight_layout(); save(fig, "ch22_unlicensed_timeline")


def fig_psd_limit():
    bw = np.array([20, 40, 80, 160, 320])
    lpi = np.minimum(5 + 10 * np.log10(bw), 30)
    u3 = np.full(bw.shape, 36.0)
    noise = -174 + 60 + 10 * np.log10(bw) + 7
    pl = 80
    fig, ax = plt.subplots(1, 2, figsize=(2 * PW, PH))
    ax[0].plot(bw, lpi, "o-", color=NAVY, label="6 GHz LPI (5 dBm/MHz)")
    ax[0].plot(bw[:4], u3[:4], "s--", color=ACCENT, label="5 GHz U-NII-3 (36 dBm cap)")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(bw); ax[0].set_xticklabels(bw)
    ax[0].set_xlabel("channel width (MHz)"); ax[0].set_ylabel("max EIRP (dBm)")
    ax[0].set_title("Total power allowed", fontsize=9); ax[0].legend(fontsize=6.3, loc="center right"); ax[0].set_ylim(10, 40)
    ax[1].plot(bw, lpi - pl - noise, "o-", color=NAVY)
    ax[1].plot(bw[:4], u3[:4] - pl - noise[:4], "s--", color=ACCENT)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(bw); ax[1].set_xticklabels(bw)
    ax[1].set_xlabel("channel width (MHz)"); ax[1].set_ylabel("SNR at 80 dB path loss (dB)")
    ax[1].set_title("SNR per subcarrier", fontsize=9)
    ax[1].text(40, 33.5, "PSD limit: flat", color=NAVY, fontsize=7)
    ax[1].set_ylim(25, 56); ax[1].text(30, 52.5, "total limit: $-3$ dB per doubling", color=ACCENT, fontsize=7)
    fig.tight_layout(); save(fig, "ch22_psd_limit")


def fig_dfs():
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    segs = [(0, 60, GRAY, "channel availability\ncheck: listen 60 s"),
            (60, 200, GREEN, "operating, monitoring for radar"),
            (200, 210, ACCENT, "leave\n<10 s"),
            (210, 400, ORANGE, "non-occupancy: stay off 30 min (on another channel)")]
    for a, b, c, t in segs:
        ax.add_patch(Rectangle((a, 0), b - a, 1, color=c, alpha=0.35, lw=0))
        ax.text((a + b) / 2, 0.5, t, ha="center", va="center", fontsize=6.8, color="k")
    ax.annotate("radar pulses\ndetected", xy=(200, 1.0), xytext=(170, 1.6), fontsize=7, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT), ha="center")
    for x in np.arange(196, 204, 1.6):
        ax.plot([x, x], [1.0, 1.35], color=ACCENT, lw=1)
    ax.set_xlim(-5, 405); ax.set_ylim(-0.2, 2.1); _clean(ax)
    ax.text(400, -0.15, "time (not to scale)", ha="right", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch22_dfs")


def fig_afc():
    fig, ax = plt.subplots(figsize=(PW, PH))
    a, b = np.array([0.5, 0.8]), np.array([5.5, 3.6])
    ax.plot(*zip(a, b), color=PURPLE, lw=2)
    for p in (a, b):
        ax.plot(*p, "^", color=PURPLE, ms=9)
    ax.text(0.4, 1.15, "microwave\nlink tower", fontsize=6.5, color=PURPLE)
    # exclusion corridor
    d = (b - a) / np.linalg.norm(b - a); nrm = np.array([-d[1], d[0]])
    poly = [a + 0.45 * nrm, b + 0.45 * nrm, b - 0.45 * nrm, a - 0.45 * nrm]
    ax.add_patch(Polygon(poly, color=PURPLE, alpha=0.12, lw=0))
    r = rng(3)
    for k in range(14):
        p = r.uniform([0, 0], [6, 4.4])
        dist = abs(np.dot(p - a, nrm))
        ok = dist > 0.45
        ax.plot(*p, "o", color=GREEN if ok else ACCENT, ms=6)
    ax.plot([], [], "o", color=GREEN, label="AP: channel granted")
    ax.plot([], [], "o", color=ACCENT, label="AP: channel denied/lower power")
    ax.legend(fontsize=6.3, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 6); ax.set_ylim(0, 4.6); _clean(ax)
    ax.set_title("Automated frequency coordination", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_afc")


def fig_rate_history():
    gens = [("802.11", 1997, 2), ("11b", 1999, 11), ("11a", 1999, 54), ("11g", 2003, 54),
            ("11n\nWi-Fi 4", 2009, 600), ("11ac\nWi-Fi 5", 2013, 6933), ("11ax\nWi-Fi 6", 2021, 9608),
            ("11be\nWi-Fi 7", 2024, 23059)]
    fig, ax = plt.subplots(figsize=(PW, PH))
    ys = [g[1] for g in gens]; rs = [g[2] for g in gens]
    ax.semilogy(ys, rs, "o-", color=NAVY)
    for nm, y, r_ in gens:
        dy = {"11a": 0.45, "11ax\nWi-Fi 6": 0.3}.get(nm, 1.7)
        ax.text(y + (0.5 if nm != "11b" else -0.5), r_ * dy, nm, fontsize=6.2,
                ha="left" if nm != "11b" else "right", va="center")
    yy = np.array([1997, 2024]); ax.semilogy(yy, 2 * 10 ** ((yy - 1997) / 27 * np.log10(23059 / 2)), ":", color=GRAY, lw=0.8)
    ax.text(2005, 4, "about $\\times$10 every\n7 years", fontsize=6.5, color=GRAY)
    ax.set_xlabel("year"); ax.set_ylabel("peak PHY rate (Mb/s)")
    ax.set_xlim(1995, 2030); ax.set_ylim(1, 1e5)
    ax.set_title("Twenty-seven years of Wi-Fi", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_rate_history")


def fig_barker():
    b = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
    ac = np.correlate(b, b, "full")
    fig, ax = plt.subplots(1, 2, figsize=(2 * PW, 2.1))
    ax[0].step(np.arange(12), np.r_[b, b[-1]], where="post", color=NAVY)
    ax[0].set_ylim(-1.6, 1.6); ax[0].set_xlabel("chip (91 ns each)"); ax[0].set_yticks([-1, 1])
    ax[0].set_title("The 11-chip Barker sequence", fontsize=9)
    lags = np.arange(-10, 11)
    ax[1].stem(lags, ac, linefmt=NAVY, markerfmt="o", basefmt=" ")
    ax[1].set_xlabel("lag (chips)"); ax[1].set_title("Autocorrelation: peak 11, sidelobes $\\leq 1$", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_barker")


def fig_rate_factors():
    # 54 Mb/s (11a) -> 23 Gb/s (11be): multiply by each factor in turn
    steps = [("802.11a\n54 Mb/s", 1.0), ("x8\nstreams", 8.0), ("x81.7\nsubcarriers\n48$\\to$3920", 3920 / 48),
             ("x2\nbits/sym\n6$\\to$12", 2.0), ("x1.11\ncode rate\n3/4$\\to$5/6", (5 / 6) / 0.75),
             ("$\\div$3.4\nlonger\nsymbol", 4.0 / 13.6)]
    vals = [54.0]
    for _, f in steps[1:]:
        vals.append(vals[-1] * f)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    prev = None
    for k, ((nm, f), v) in enumerate(zip(steps, vals)):
        lo = 54 if k == 0 else min(prev, v); hi = 54 if k == 0 else max(prev, v)
        if k == 0:
            ax.bar(k, 54, bottom=1, color=NAVY, width=0.6)
        else:
            ax.bar(k, hi - lo, bottom=lo, color=GREEN if v > prev else ACCENT, width=0.6)
        ax.text(k, (max(v, prev) if k else 54) * 1.3, f"{v/1000:.2f} Gb/s" if v > 1000 else f"{v:.0f} Mb/s",
                ha="center", fontsize=6.8)
        prev = v
    ax.bar(len(steps), vals[-1], bottom=1, color=NAVY, width=0.6)
    ax.text(len(steps), vals[-1] * 1.25, "23 Gb/s", ha="center", fontsize=7, weight="bold")
    ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels([s[0] for s in steps] + ["802.11be\npeak"], fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylim(10, 3e5); ax.set_ylabel("PHY rate (Mb/s)")
    ax.set_title("Where Wi-Fi 7's 23 Gb/s comes from: one formula, five factors", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_rate_factors")


def fig_preamble_overhead():
    L = np.logspace(np.log10(40), np.log10(65000), 200)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for mcs, c in [(0, ACCENT), (7, ORANGE), (11, NAVY)]:
        bps = 2 * 980 * BITS[mcs]               # bits per 13.6 us symbol, 2 streams, 80 MHz
        nsym = np.ceil((8 * L + 22) / bps)
        t = 52 + nsym * 13.6
        ax.semilogx(L, 100 * 52 / t, color=c, label=f"MCS {mcs}")
    ax.axvline(1500, color=GRAY, ls=":", lw=0.7); ax.text(1600, 85, "1500 B", fontsize=6.5, color=GRAY)
    ax.axvline(100, color=GRAY, ls=":", lw=0.7); ax.text(105, 10, "TCP ACK", fontsize=6.5, color=GRAY)
    ax.set_xlabel("PSDU size (bytes)"); ax.set_ylabel("airtime spent on preamble (%)")
    ax.set_title("Preamble tax (11ax, 2 streams, 80 MHz)", fontsize=8.5); ax.legend(fontsize=6.5)
    ax.set_ylim(0, 100)
    fig.tight_layout(); save(fig, "ch22_preamble_overhead")


def fig_mumimo_beams():
    N = 8; d = 0.5
    th = np.linspace(-np.pi / 2, np.pi / 2, 721)
    a = lambda t: np.exp(2j * np.pi * d * np.arange(N)[:, None] * np.sin(np.atleast_1d(t))[None, :])
    users = np.deg2rad([-40, 5, 35])
    H = a(users).T.conj()                      # rows: users
    W = np.linalg.pinv(H)                      # zero forcing
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    fig = plt.figure(figsize=(PW, PH + 0.2))
    ax = fig.add_subplot(111, projection="polar")
    for k, c in enumerate([NAVY, ACCENT, GREEN]):
        g = np.abs(W[:, k].conj() @ a(th)) ** 2
        ax.plot(th, 10 * np.log10(g / N + 1e-6) + 30, color=c, lw=1.2)
        ax.plot([users[k]] * 2, [0, 32], color=c, ls=":", lw=0.8)
    ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(0, 32); ax.set_yticks([10, 20, 30]); ax.set_yticklabels(["-20", "-10", "0 dB"], fontsize=6)
    ax.set_title("Zero-forcing beams, 8 antennas, 3 users", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_mumimo_beams")


def fig_ofdma():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True)
    t = 0
    for k in range(9):
        ax[0].add_patch(Rectangle((t, 0), 52, 1, color=ORANGE, alpha=0.7, lw=0))
        ax[0].add_patch(Rectangle((t + 52, 0), 14, 1, color=CYCLE[k % 6], alpha=0.8, lw=0))
        t += 66 + 16 + 44 + 67.5
    ax[0].text(t + 20, 0.5, f"{t:.0f} $\\mu$s", va="center", fontsize=7)
    ax[0].set_title("Nine short packets, one at a time (each: contention + preamble + data + ACK)", fontsize=8)
    ax[1].add_patch(Rectangle((0, 0), 52, 9, color=ORANGE, alpha=0.7, lw=0))
    for k in range(9):
        ax[1].add_patch(Rectangle((52, k), 4 * 13.6, 0.9, color=CYCLE[k % 6], alpha=0.8, lw=0))
    t2 = 52 + 4 * 13.6 + 16 + 100 + 44 + 67.5
    ax[1].text(t2 + 20, 4.5, f"{t2:.0f} $\\mu$s", va="center", fontsize=7)
    ax[1].text(26, 4.5, "one\npreamble", ha="center", va="center", fontsize=6.5, rotation=90)
    ax[1].set_title("The same nine packets in one OFDMA PPDU (nine 26-tone RUs)", fontsize=8)
    for a_ in ax:
        a_.set_yticks([]); a_.grid(False); a_.spines["left"].set_visible(False)
    ax[0].set_ylim(-0.2, 1.4); ax[1].set_ylim(-0.3, 9.3)
    ax[1].set_xlabel("time ($\\mu$s; orange = preamble, grey gaps = contention and acknowledgement)")
    ax[1].set_xlim(0, t + 220)
    fig.tight_layout(); save(fig, "ch22_ofdma")


def fig_mlo():
    r = rng(11)
    n = 200000
    # each link busy (by neighbours) with prob p; waiting time until idle ~ geometric in 0.5 ms slots,
    # with a correlated long blocking (microwave/DFS) on link 1 occasionally
    def wait(p, n):
        return r.geometric(1 - p, n) - 1
    w1 = wait(0.6, n) * 0.5 + (r.random(n) < 0.03) * r.uniform(5, 40, n)
    w2 = wait(0.5, n) * 0.5
    w3 = wait(0.55, n) * 0.5
    single = w1; mlo = np.minimum(np.minimum(w1, w2), w3)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for v, c, lab in [(single, ACCENT, "one link (5 GHz)"), (mlo, NAVY, "MLO, three links")]:
        s = np.sort(v); ccdf = 1 - np.arange(n) / n
        ax.semilogy(s, ccdf, color=c, label=lab)
    ax.set_xlabel("access delay (ms)"); ax.set_ylabel("P(delay > x)")
    ax.set_xlim(0, 40); ax.set_ylim(1e-4, 1); ax.legend(fontsize=6.8)
    ax.set_title("Multi-link operation cuts the tail", fontsize=9)
    p99s = np.percentile(single, 99); p99m = np.percentile(mlo, 99)
    ax.text(14, 0.2, f"99th percentile:\n{p99s:.1f} ms $\\to$ {p99m:.1f} ms", fontsize=6.8)
    fig.tight_layout(); save(fig, "ch22_mlo")
    return p99s, p99m


def fig_sector_sweep():
    fig = plt.figure(figsize=(PW, PH + 0.2))
    ax = fig.add_subplot(111, projection="polar")
    cent = np.deg2rad(np.arange(-56, 57, 16))
    th = np.linspace(-np.pi / 2, np.pi / 2, 600)
    target = np.deg2rad(22)
    best = np.argmin(np.abs(cent - target))
    for k, c0 in enumerate(cent):
        g = np.exp(-((th - c0) / np.deg2rad(9)) ** 2)
        ax.plot(th, g, color=ACCENT if k == best else NAVY, lw=1.8 if k == best else 0.7,
                alpha=1 if k == best else 0.6)
    ax.plot([target, target], [0, 1.15], color=GREEN, lw=1.2, ls="--")
    ax.text(target, 1.22, "peer", color=GREEN, fontsize=7, ha="center")
    ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_yticks([]); ax.set_ylim(0, 1.3)
    ax.set_title("60 GHz sector sweep: try every beam, keep the best", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_sector_sweep")


def fig_sensing():
    r = rng(5)
    T = 600; K = 52
    t = np.arange(T) * 0.05                       # 20 packets/s for 30 s
    f = np.arange(K)
    paths = [(1.0, 20e-9), (0.5, 55e-9), (0.3, 90e-9)]
    H = np.zeros((T, K), complex)
    for g, tau in paths:
        H += g * np.exp(-2j * np.pi * (f[None, :] * 312.5e3) * tau)
    # person: walking 5-15 s (moving path), breathing 20-30 s (small periodic)
    walk = (t > 5) & (t < 15)
    tau_p = 40e-9 + 30e-9 * np.sin(2 * np.pi * (t - 5) / 10)
    H += (0.45 * walk)[:, None] * np.exp(-2j * np.pi * f[None, :] * 312.5e3 * tau_p[:, None])
    br = (t > 18)
    H += (0.25 * br)[:, None] * np.exp(-2j * np.pi * f[None, :] * 312.5e3 * (60e-9 + 0.8e-9 * np.sin(2 * np.pi * 0.25 * t))[:, None]) * np.exp(1j * 0.6 * np.sin(2 * np.pi * 0.25 * t))[:, None]
    H += 0.02 * (r.standard_normal((T, K)) + 1j * r.standard_normal((T, K)))
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    im = ax.imshow(20 * np.log10(np.abs(H).T), aspect="auto", origin="lower", extent=[0, t[-1], -26, 26], cmap="viridis")
    ax.set_xlabel("time (s)"); ax.set_ylabel("subcarrier")
    ax.text(10, 22, "someone walks across the room", ha="center", color="w", fontsize=7)
    ax.text(24, 22, "someone sits and breathes", ha="center", color="w", fontsize=7)
    ax.text(2.5, 22, "empty room", ha="center", color="w", fontsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("|H| (dB)", fontsize=7)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch22_sensing")


def fig_contention_party():
    """Slot-level DCF picture: four stations, random backoff, freeze, a collision."""
    r = rng(4)
    names = ["laptop", "phone", "TV", "tablet"]
    cols = [NAVY, GREEN, ORANGE, PURPLE]
    cw = [15] * 4
    cnt = [int(r.integers(0, 16)) for _ in range(4)]
    cnt = [7, 3, 11, 3]                        # chosen to show a collision first
    t = 0.0; slot = 9; tx_len = 120; difs = 34
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    t += difs
    events = 0
    while events < 4:
        m = min(cnt)
        for k in range(4):
            if cnt[k] > 0:
                for s in range(m):
                    ax.add_patch(Rectangle((t + s * slot, k - 0.18), slot * 0.9, 0.36, color=cols[k], alpha=0.25, lw=0))
                ax.text(t + 1, k + 0.27, str(cnt[k]), fontsize=5.5, color=cols[k])
        t += m * slot
        winners = [k for k in range(4) if cnt[k] == m]
        for k in range(4):
            cnt[k] -= m
        coll = len(winners) > 1
        for k in winners:
            ax.add_patch(Rectangle((t, k - 0.35), tx_len, 0.7, color=ACCENT if coll else cols[k], alpha=0.85, lw=0))
            ax.text(t + tx_len / 2, k, "collision!" if coll else "talks", ha="center", va="center", fontsize=6.5, color="w")
        t += tx_len + 16 + (0 if coll else 30) + difs
        for k in winners:
            if coll:
                cw[k] = 2 * cw[k] + 1
            else:
                cw[k] = 15
            cnt[k] = int(r.integers(1, cw[k] + 1)) if not coll else [9, 21][winners.index(k)]
        events += 1
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("time ($\\mu$s); pale blocks are idle slots counted down (number = counter)")
    ax.set_ylim(-0.6, 3.6); ax.set_xlim(0, t); ax.grid(False)
    ax.set_title("Polite conversation: wait for a pause, roll a die, speak when your count reaches zero", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_contention_party")


def fig_dcf_scaling():
    r = rng(2)
    Ns = np.arange(1, 41)
    out = {}
    for cwmin, c in [(15, NAVY), (63, GREEN)]:
        S = []; P = []
        for N in Ns:
            cw = np.full(N, cwmin); bo = r.integers(0, cwmin + 1, N)
            busy = 0.0; idle = 0; succ = 0; coll = 0; t = 0.0
            for it in range(3000):
                m = bo.min(); t += m * 9
                bo -= m
                w = np.where(bo == 0)[0]
                t += 1500 * 8 / 54 + 34 + 16 + 44
                if len(w) == 1:
                    succ += 1; cw[w] = cwmin
                else:
                    coll += 1; cw[w] = np.minimum(2 * cw[w] + 1, 1023)
                bo[w] = r.integers(0, cw[w] + 1)
            S.append(succ * 12000 / t); P.append(coll / (succ + coll))
        out[cwmin] = (np.array(S), np.array(P))
    fig, ax = plt.subplots(figsize=(PW, PH))
    for cwmin, c in [(15, NAVY), (63, GREEN)]:
        ax.plot(Ns, out[cwmin][0], color=c, label=f"throughput, CWmin {cwmin}")
    ax.set_xlabel("saturated stations"); ax.set_ylabel("total throughput (Mb/s)")
    ax.set_ylim(0, 35)
    ax2 = ax.twinx(); ax2.grid(False)
    for cwmin, c in [(15, NAVY), (63, GREEN)]:
        ax2.plot(Ns, 100 * out[cwmin][1], color=c, ls="--", lw=0.9)
    ax2.set_ylabel("collision fraction (%, dashed)"); ax2.set_ylim(0, 60)
    ax2.spines["right"].set_visible(True)
    ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("A crowded room at 54 Mb/s", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_dcf_scaling")


def fig_edca_share():
    r = rng(8)
    acs = [("voice", 2, 3, 7), ("video", 2, 7, 15), ("best effort", 3, 15, 1023), ("background", 7, 15, 1023)]
    wins = np.zeros(4)
    cw = np.array([a[2] for a in acs]); cwmin = cw.copy(); cwmax = np.array([a[3] for a in acs])
    aifsn = np.array([a[1] for a in acs])
    bo = np.array([r.integers(0, c + 1) for c in cw]) + aifsn
    for it in range(40000):
        m = bo.min(); bo -= m
        w = np.where(bo == 0)[0]
        if len(w) == 1:
            wins[w[0]] += 1; cw[w] = cwmin[w]
        else:
            cw[w] = np.minimum(2 * cw[w] + 1, cwmax[w])
        for k in w:
            bo[k] = r.integers(0, cw[k] + 1)
        bo += 0
        bo[w] += aifsn[w]
    fig, ax = plt.subplots(figsize=(PW, PH))
    sh = 100 * wins / wins.sum()
    ax.bar(range(4), sh, color=[ACCENT, ORANGE, NAVY, GRAY])
    for k, v in enumerate(sh):
        ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([a[0] for a in acs], fontsize=7.5)
    ax.set_ylabel("share of successful accesses (%)"); ax.set_ylim(0, 80)
    ax.set_title("Four saturated queues, EDCA defaults", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_edca_share")
    return sh


def fig_hidden_node():
    fig, ax = plt.subplots(figsize=(PW, PH))
    pos = {"A": (-1.5, 0), "AP": (0, 0), "C": (1.5, 0)}
    for k, c in [("A", NAVY), ("C", GREEN)]:
        ax.add_patch(Circle(pos[k], 1.75, color=c, alpha=0.08))
        ax.add_patch(Circle(pos[k], 1.75, fill=False, color=c, ls="--", lw=0.8))
    for k, (x, y) in pos.items():
        ax.plot(x, y, "o", ms=11, color=ACCENT if k == "AP" else (NAVY if k == "A" else GREEN))
        ax.text(x, y - 0.35, k, ha="center", fontsize=8)
    ax.annotate("", xy=(-0.15, 0.08), xytext=(-1.35, 0.08), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.annotate("", xy=(0.15, 0.08), xytext=(1.35, 0.08), arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.text(0, 0.55, "collision\nat the AP", ha="center", color=ACCENT, fontsize=7)
    ax.text(-1.5, 1.95, "A's range", ha="center", color=NAVY, fontsize=7)
    ax.text(1.5, 1.95, "C's range", ha="center", color=GREEN, fontsize=7)
    ax.text(0, -1.2, "A and C cannot hear each other:\ncarrier sense fails", ha="center", fontsize=7)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.0, 2.2); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(); save(fig, "ch22_hidden_node")


def fig_parcels():
    """Aggregation as one parcel versus 64 envelopes: airtime at 600 Mb/s, to scale."""
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    T0 = 194.5; Td = 20.5
    t = 0
    for k in range(64):
        ax.add_patch(Rectangle((t, 1.1), T0, 0.6, color=GRAY, alpha=0.5, lw=0))
        ax.add_patch(Rectangle((t + T0 - 60, 1.1), Td, 0.6, color=NAVY, lw=0))
        t += T0 + Td
    ax.text(t + 150, 1.4, f"64 envelopes: {t/1000:.1f} ms", va="center", fontsize=7.5)
    ax.add_patch(Rectangle((0, 0.1), 198.5, 0.6, color=GRAY, alpha=0.5, lw=0))
    ax.add_patch(Rectangle((140, 0.1), 1318, 0.6, color=NAVY, lw=0))
    ax.text(1516 + 150, 0.4, "one parcel (A-MPDU): 1.5 ms", va="center", fontsize=7.5)
    ax.set_xlim(0, t + 3500); ax.set_ylim(-0.1, 1.9); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time ($\\mu$s) at 600 Mb/s; navy = data, grey = waiting, preamble and acknowledgement")
    fig.tight_layout(); save(fig, "ch22_parcels")


def fig_blockack():
    r = rng(9)
    n = 32
    fail = np.zeros(n, bool); fail[[5, 6, 17, 28]] = True
    fig, ax = plt.subplots(figsize=(W1, 1.6))
    for k in range(n):
        ax.add_patch(Rectangle((k, 1), 0.9, 0.7, color=ACCENT if fail[k] else NAVY, alpha=0.85, lw=0))
        ax.text(k + 0.45, 0.65, "0" if fail[k] else "1", ha="center", fontsize=6.5, color=ACCENT if fail[k] else NAVY)
    ax.text(-0.5, 1.35, "A-MPDU", ha="right", va="center", fontsize=7)
    ax.text(-0.5, 0.7, "BA bitmap", ha="right", va="center", fontsize=7)
    for j, k in enumerate(np.where(fail)[0]):
        ax.add_patch(Rectangle((n + 2 + j, 1), 0.9, 0.7, color=ORANGE, alpha=0.85, lw=0))
    ax.text(n + 4, 1.95, "resent in\nnext aggregate", ha="center", fontsize=6.5, color=ORANGE)
    ax.set_xlim(-6, n + 8); ax.set_ylim(0.4, 2.4); _clean(ax)
    fig.tight_layout(); save(fig, "ch22_blockack")


def fig_rate_anomaly():
    fig, ax = plt.subplots(figsize=(PW, PH))
    lab = ["A alone", "A (DCF)", "B (DCF)", "A (ATF)", "B (ATF)"]
    v = [55.8, 4.9, 4.9, 27.9, 2.7]
    c = [GRAY, NAVY, ACCENT, NAVY, ACCENT]
    ax.bar(range(5), v, color=c, alpha=0.85)
    for k, x in enumerate(v):
        ax.text(k, x + 1.2, f"{x:.1f}", ha="center", fontsize=7)
    ax.axvline(0.5, color=GRAY, lw=0.5); ax.axvline(2.5, color=GRAY, lw=0.5)
    ax.text(1.5, 52, "equal turns:\ntotal 9.8", ha="center", fontsize=7)
    ax.text(3.5, 52, "equal time:\ntotal 30.6", ha="center", fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=6.8)
    ax.set_ylabel("throughput (Mb/s)"); ax.set_ylim(0, 62)
    ax.set_title("A at 600 Mb/s, B at 6 Mb/s", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_rate_anomaly")


def fig_powersave():
    t = np.arange(0, 1.0, 1e-4)
    i = np.full_like(t, 0.05)                    # mA floor (illustrative)
    for k in range(10):
        t0 = k * 0.1024
        on = (t >= t0) & (t < t0 + 0.002)
        i[on] = 60 if k % 3 == 0 else 45
    on = (t >= 0.4096) & (t < 0.42); i[on] = 200
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.plot(t * 1000, i, color=NAVY, lw=0.9)
    ax.set_yscale("log"); ax.set_ylim(0.02, 400)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("current (mA, log)")
    ax.annotate("wake for every beacon\n(102.4 ms)", xy=(102.4, 45), xytext=(150, 150), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("TIM bit set:\nfetch buffered frames", xy=(412, 200), xytext=(520, 150), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.text(700, 0.08, "asleep between beacons", fontsize=6.8, color=GRAY)
    ax.set_title("Legacy power save, schematically (currents illustrative)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_powersave")


def fig_roaming():
    x = np.linspace(0, 40, 400)
    r = rng(12)
    def rssi(x0):
        d = np.abs(x - x0) + 1
        return 15 - 40 - 35 * np.log10(d) + np.convolve(r.standard_normal(len(x)) * 2.5, np.ones(15) / np.sqrt(15) / 2, "same")
    a, b = rssi(2), rssi(38)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(x, a, color=NAVY, lw=1, label="AP 1")
    ax.plot(x, b, color=GREEN, lw=1, label="AP 2")
    sticky = np.argmax(a < -78); good = np.argmax(b > a + 5)
    ax.axvline(x[good], color=GREEN, ls="--", lw=0.8)
    ax.axvline(x[sticky], color=ACCENT, ls="--", lw=0.8)
    ax.text(x[good] - 0.5, -45, "11k/v\nroam", fontsize=6.5, color=GREEN, ha="right")
    ax.text(x[sticky] + 0.5, -45, "sticky\nclient", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("position along corridor (m)"); ax.set_ylabel("RSSI (dBm)")
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_ylim(-95, -35)
    ax.set_title("When should the phone let go?", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_roaming")


def fig_mesh_hops():
    h = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(h, 100 / h, "o-", color=ACCENT, label="one shared channel ($1/h$)")
    ax.plot(h, 100 / h * np.r_[1, 0.8 ** np.arange(1, 5)], "s:", color=ORANGE, label="... with hop interference")
    ax.plot(h, np.r_[100, np.full(4, 90)] , "^-", color=NAVY, label="dedicated backhaul radio")
    ax.set_xlabel("wireless hops"); ax.set_ylabel("end-to-end throughput (% of one hop)")
    ax.set_xticks(h); ax.legend(fontsize=6.3); ax.set_ylim(0, 110)
    ax.set_title("Every relay costs airtime (idealised)", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_mesh_hops")


def fig_wep_iv():
    n = np.logspace(1, 7.5, 300)
    N = 2.0 ** 24
    p = 1 - np.exp(-n * (n - 1) / (2 * N))
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.semilogx(n, p, color=ACCENT)
    n50 = np.sqrt(2 * N * np.log(2))
    ax.axvline(n50, color=GRAY, ls=":", lw=0.8)
    ax.text(n50 * 1.15, 0.3, f"50% after\n{n50:.0f} frames", fontsize=7)
    ax.set_xlabel("frames sent with random 24-bit IVs"); ax.set_ylabel("P(some IV repeats)")
    ax.set_title("WEP's birthday problem", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_wep_iv")
    return n50


def fig_walls():
    items = [("glass (plain)", 2, 3), ("plasterboard", 3, 5), ("wooden door", 3, 6), ("brick", 10, 15),
             ("concrete wall", 12, 20), ("floor slab", 15, 25), ("low-E glass", 20, 35)]
    fig, ax = plt.subplots(figsize=(PW, PH))
    for k, (nm, lo, hi) in enumerate(items):
        ax.barh(k, hi - lo, left=lo, color=NAVY if hi < 10 else (ORANGE if hi < 26 else ACCENT), alpha=0.8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=7)
    ax.set_xlabel("loss at 5 GHz (dB, rough ranges)"); ax.invert_yaxis()
    ax.set_title("What walls cost", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_walls")


def fig_neighbours():
    N = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(N, 100 / (N + 1), "o-", color=NAVY, label="all networks busy")
    ax.plot(N, 100 / (1 + 0.3 * N), "s--", color=GREEN, label="neighbours 30% busy")
    ax.set_xlabel("audible co-channel networks"); ax.set_ylabel("your share of airtime (%)")
    ax.legend(fontsize=6.8); ax.set_ylim(0, 105)
    ax.set_title("Neighbours are co-tenants", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_neighbours")


def fig_bt_hopping():
    r = rng(14)
    n = 160
    hops = r.integers(0, 79, n)
    wifi = (hops >= 22) & (hops <= 41)                 # Wi-Fi ch 6: 2427-2447 MHz -> BT ch 25..45 approx
    wifi = (hops >= 25 - 2) & (hops <= 45 - 2)
    good = np.r_[np.arange(0, 23), np.arange(44, 79)]
    afh = good[hops % len(good)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for a_, h, ttl in [(ax[0], hops, "Classic hopping: about one hop in five\nlands on a busy Wi-Fi channel"),
                       (ax[1], afh, "Adaptive frequency hopping:\nbad channels remapped")]:
        a_.axhspan(2402 + 23, 2402 + 43, color=ORANGE, alpha=0.2)
        bad = (h >= 23) & (h <= 43)
        a_.scatter(np.arange(n)[~bad] * 0.625, 2402 + h[~bad], s=4, color=NAVY)
        a_.scatter(np.arange(n)[bad] * 0.625, 2402 + h[bad], s=7, color=ACCENT, marker="x")
        a_.set_xlabel("time (ms)"); a_.set_title(ttl, fontsize=8)
        a_.text(98, 2436, "Wi-Fi ch 6", fontsize=6.5, color=ORANGE, ha="right")
    ax[0].set_ylabel("frequency (MHz)")
    fig.tight_layout(); save(fig, "ch22_bt_hopping")
    return np.mean((hops >= 23) & (hops <= 43))


def fig_piconets():
    N = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(N, 100 * (1 - 1 / 79) ** (N - 1), color=NAVY, label="79 channels")
    ax.plot(N, 100 * (1 - 1 / 20) ** (N - 1), color=ACCENT, ls="--", label="20 channels (AFH minimum)")
    ax.set_xlabel("piconets in range"); ax.set_ylabel("clean slots (%)")
    ax.legend(fontsize=6.8); ax.set_ylim(0, 102)
    ax.set_title("Graceful degradation, no coordination", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_piconets")


def fig_lighthouse():
    """Discovery latency: advertiser every T_adv (+0..10 ms), scanner window 30 ms every 100 ms rotating channels."""
    r = rng(15)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for tadv, c in [(0.1, NAVY), (0.5, GREEN), (1.0, ORANGE)]:
        lat = []
        for trial in range(1500):
            t = r.uniform(0, tadv)
            while True:
                # scanner: window 30 ms at start of each 100 ms interval, channel cycles 37,38,39
                k = int(t // 0.1); ph = t - k * 0.1; ch = k % 3
                # advertiser sends on 37,38,39 at t, t+0.5ms, t+1ms
                ta = t + 0.0005 * ch
                k2 = int(ta // 0.1)
                if ta - k2 * 0.1 < 0.030 and (k2 % 3) == ch:
                    lat.append(t); break
                t += tadv + r.uniform(0, 0.01)
                if t > 30:
                    lat.append(t); break
        s = np.sort(lat)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=c, label=f"advertise every {tadv*1000:.0f} ms")
    ax.set_xscale("log"); ax.set_xlabel("time to discovery (s)"); ax.set_ylabel("fraction found")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_xlim(0.01, 30)
    ax.set_title("How quickly a phone spots the lighthouse", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_lighthouse")


def fig_ble_throughput():
    pl = np.arange(20, 252)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for rate, c, nm in [(1e6, NAVY, "LE 1M"), (2e6, GREEN, "LE 2M"), (125e3, ACCENT, "LE Coded S=8")]:
        if rate == 125e3:
            tpk = 80e-6 + 256e-6 + 16e-6 + 24e-6 + (2 + pl + 4 + 3) * 8 * 8e-6
            tack = 80e-6 + 256e-6 + 16e-6 + 24e-6 + (2 + 4 + 3) * 8 * 8e-6
        else:
            pre = 1 if rate == 1e6 else 2
            tpk = (pre + 4 + 2 + pl + 4 + 3) * 8 / rate
            tack = (pre + 4 + 2 + 3) * 8 / rate
        T = tpk + tack + 2 * 150e-6
        ax.plot(pl, pl * 8 / T / 1e3, color=c, label=nm)
    ax.set_xlabel("payload per packet (bytes)"); ax.set_ylabel("application throughput (kb/s)")
    ax.legend(fontsize=6.8); ax.set_title("BLE throughput, ideal back-to-back", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_ble_throughput")


def fig_aoa():
    th = np.linspace(-90, 90, 361)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(th, np.rad2deg(np.pi * np.sin(np.deg2rad(th))), color=NAVY, label="$d=\\lambda/2$")
    ax.plot(th, np.rad2deg(np.angle(np.exp(1j * 2 * np.pi * np.sin(np.deg2rad(th))))), color=ACCENT, ls="--", lw=0.9,
            label="$d=\\lambda$ (ambiguous)")
    ax.set_xlabel("angle of arrival $\\theta$ (deg)"); ax.set_ylabel("phase difference (deg)")
    ax.legend(fontsize=6.8); ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_title("Two antennas, one phase difference", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_aoa")


def fig_channel_sounding():
    r = rng(16)
    f = 2402e6 + np.arange(0, 79) * 1e6
    fig, ax = plt.subplots(figsize=(PW, PH))
    for d, c in [(2.0, NAVY), (7.5, GREEN), (20.0, ORANGE)]:
        ph = -4 * np.pi * f * d / 3e8 + 0.15 * r.standard_normal(len(f))
        un = np.unwrap(np.angle(np.exp(1j * ph)))
        slope = np.polyfit(f, un, 1)[0]
        dh = -3e8 / (4 * np.pi) * slope
        ax.plot((f - f[0]) / 1e6, un - un[0], ".", ms=2.5, color=c, label=f"d = {d} m (est. {dh:.2f} m)")
    ax.set_xlabel("tone frequency offset (MHz)"); ax.set_ylabel("round-trip phase (rad, unwrapped)")
    ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("Distance is the slope", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_channel_sounding")


def fig_coex_levels():
    fig, ax = plt.subplots(figsize=(PW, PH))
    # left column: what Wi-Fi listens for; right column: what the low-power radio experiences
    left = [(-62, "energy-detect threshold\n(anything, 20 MHz)", ACCENT), (-82, "preamble-detect\nthreshold (Wi-Fi only)", NAVY)]
    right = [(-70, "Zigbee/BLE packet\n(2 MHz wide)", GREEN), (-80, "Wi-Fi power landing\nin that 2 MHz", ORANGE),
             (-92, "Zigbee sensitivity,\nroughly", GRAY)]
    for y, t, c in left:
        ax.plot([0, 1], [y, y], color=c, lw=3)
        ax.text(0.5, y + 0.8, t, fontsize=6, ha="center", va="bottom", color=c)
    for y, t, c in right:
        ax.plot([1.6, 2.6], [y, y], color=c, lw=3)
        ax.text(2.1, y + 0.8, t, fontsize=6, ha="center", va="bottom", color=c)
    ax.annotate("", xy=(1.35, -62), xytext=(1.35, -70), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text(1.3, -66.5, "8 dB\nbelow", fontsize=5.8, ha="right", va="center", color=GRAY)
    ax.text(0.5, -52, "Wi-Fi listens for", ha="center", fontsize=7, weight="bold", color=NAVY)
    ax.text(2.1, -52, "Zigbee lives with", ha="center", fontsize=7, weight="bold", color=GREEN)
    ax.set_xlim(-0.1, 2.7); ax.set_ylim(-96, -50); ax.set_xticks([])
    ax.set_ylabel("level (dBm)"); ax.set_title("Wi-Fi cannot hear what it hurts", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_coex_levels")


def fig_microwave():
    t = np.linspace(0, 50e-3, 5000)
    oven = (np.sin(2 * np.pi * 60 * t) > 0).astype(float)
    r = rng(17)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax[0].fill_between(t * 1e3, 0, oven, color=ACCENT, alpha=0.25, lw=0, label="oven radiating")
    for k in range(18):
        t0 = r.uniform(0, 49); L = 0.4
        hit = oven[int(t0 / 50 * 4999)] > 0 or oven[min(4999, int((t0 + L) / 50 * 4999))] > 0
        ax[0].add_patch(Rectangle((t0, 1.15), L, 0.3, color=ACCENT if hit else GREEN))
    for k in range(6):
        t0 = r.uniform(0, 46); L = 3.0
        ax[0].add_patch(Rectangle((t0, 1.6), L, 0.3, color=ACCENT))
    ax[0].text(51, 1.3, "short\npackets", fontsize=6.5, va="center"); ax[0].text(51, 1.75, "long", fontsize=6.5, va="center")
    ax[0].set_xlim(0, 58); ax[0].set_ylim(0, 2.1); ax[0].set_yticks([]); ax[0].set_xlabel("time (ms), 60 Hz mains")
    ax[0].set_title("Packets in the oven's off half-cycles survive", fontsize=8)
    L = np.linspace(0.05, 8.3, 200)
    ps = np.clip(1 - (L + 0) / 8.33, 0, 1) * 0.5 + 0.0
    ps = np.clip((8.33 - L) / 16.67, 0, 1)
    ax[1].plot(L, 100 * ps, color=NAVY)
    ax[1].set_xlabel("packet duration (ms)"); ax[1].set_ylabel("success if sent at random (%)")
    ax[1].set_title("Shorter is safer", fontsize=8)
    fig.tight_layout(); save(fig, "ch22_microwave_oven")


def fig_slide_whistle():
    sf = 7; M = 2 ** sf
    syms = [0, 32, 96, 64]
    fig, ax = plt.subplots(figsize=(PW, PH))
    for k, s in enumerate(syms):
        n = np.arange(M)
        f = ((n + s) % M) / M - 0.5
        ax.plot(k + n / M, f * 125, color=NAVY, lw=1.4)
        ax.plot(k, (s / M - 0.5) * 125, "o", color=ACCENT, ms=5)
        ax.text(k + 0.5, 70, f"s={s}", ha="center", fontsize=7)
    ax.set_xlabel("symbol periods"); ax.set_ylabel("frequency (kHz)")
    ax.set_ylim(-70, 80); ax.set_title("Slide whistle: the starting pitch is the data", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_slide_whistle")


def fig_lora_offsets():
    sf = 8; M = 2 ** sf
    cfo_bins = 12.0
    n = np.arange(M)
    up = iot.lora_symbols(sf, [0]); dn = iot.lora_symbols(sf, [0], down=True)
    rot = np.exp(2j * np.pi * cfo_bins * n / M)
    Yu = np.abs(np.fft.fft(up * rot * np.conj(up)))
    Yd = np.abs(np.fft.fft(dn * rot * np.conj(dn)))
    # timing offset of 5 chips: cyclic shift
    tau = 5
    Yu2 = np.abs(np.fft.fft(np.roll(up, -tau) * rot * np.conj(up)))
    Yd2 = np.abs(np.fft.fft(np.roll(dn, -tau) * rot * np.conj(dn)))
    fig, ax = plt.subplots(figsize=(PW, PH))
    k = (np.arange(M) + M // 2) % M - M // 2
    o = np.argsort(k)
    ax.plot(k[o], Yu2[o] / M, color=NAVY, label="up-chirp: CFO + timing")
    ax.plot(k[o], Yd2[o] / M, color=ACCENT, ls="--", label="down-chirp: CFO $-$ timing")
    ax.set_xlim(-30, 30); ax.set_xlabel("FFT bin after dechirp"); ax.set_ylabel("normalised magnitude")
    ax.legend(fontsize=6.3, loc="upper left"); ax.set_ylim(0, 1.25)
    ax.set_title("12-bin CFO, 5-chip timing error", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_lora_offsets")


def fig_aloha():
    G = np.linspace(0, 3, 300)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(G, G * np.exp(-2 * G), color=ACCENT, label="pure ALOHA (LoRaWAN uplink)")
    ax.plot(G, G * np.exp(-G), color=NAVY, label="slotted ALOHA (RFID Gen2)")
    ax.plot(0.5, 0.5 * np.exp(-1), "o", color=ACCENT); ax.plot(1, np.exp(-1), "o", color=NAVY)
    ax.text(0.55, 0.19, "18%", fontsize=7, color=ACCENT); ax.text(1.05, 0.375, "37%", fontsize=7, color=NAVY)
    ax.set_xlabel("offered load $G$ (packets per packet time)"); ax.set_ylabel("throughput $S$")
    ax.legend(fontsize=6.3); ax.set_ylim(0, 0.45)
    ax.set_title("Talking whenever you like has a ceiling", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_aloha")


def fig_sigfox():
    r = rng(18)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for k in range(22):
        f0 = r.uniform(5, 187); t0 = r.uniform(0, 18); c = CYCLE[k % 6]
        for rep in range(3):
            f = f0 if rep == 0 else r.uniform(5, 187)
            ax.add_patch(Rectangle((t0 + rep * 2.2, f - 0.6), 2.0, 1.2, color=c, alpha=0.85, lw=0))
    ax.set_xlim(0, 25); ax.set_ylim(0, 192)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency in the 192 kHz band (kHz)")
    ax.set_title("Ultra-narrowband: random frequencies,\nthree copies each", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_sigfox")


def fig_nbiot_rep():
    R = 2 ** np.arange(0, 12)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.semilogx(R, 10 * np.log10(R), "o-", color=NAVY, label="ideal combining gain")
    ax.semilogx(R, 10 * np.log10(R) - 0.6 * np.log2(R) * 0.5, "s--", color=ORANGE, label="with imperfect channel est. (illustr.)")
    ax.axhline(13, color=GRAY, ls=":", lw=0.8); ax.text(1.2, 14, "worked example: 13 dB", fontsize=6.5, color=GRAY)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 256, 2048]); ax.set_xticklabels([1, 4, 16, 64, 256, 2048])
    ax.set_xlabel("repetitions $R$"); ax.set_ylabel("SNR gain (dB)")
    ax.legend(fontsize=6.3, loc="upper left"); ax.set_title("Coverage bought with time", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_nbiot_rep")


def fig_psm():
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    t = np.linspace(0, 100, 10000)
    i = np.full_like(t, 0.003)
    i[(t > 2) & (t < 4)] = 150
    for k in range(5):
        a = 5 + k * 5
        i[(t > a) & (t < a + 0.3)] = 30
    i[(t > 90) & (t < 92)] = 150
    ax.plot(t, i, color=NAVY, lw=0.9); ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3)
    ax.axvspan(4, 30, color=GREEN, alpha=0.08); ax.text(17, 300, "eDRX: brief paging checks", ha="center", fontsize=6.8)
    ax.axvspan(30, 90, color=NAVY, alpha=0.06); ax.text(60, 300, "PSM: radio off, still registered", ha="center", fontsize=6.8)
    ax.text(3, 300, "send", ha="center", fontsize=6.8); ax.text(91, 300, "send", ha="center", fontsize=6.8)
    ax.set_xlabel("time (arbitrary; hours to days in reality)"); ax.set_ylabel("current (mA)")
    ax.set_xticks([])
    fig.tight_layout(); save(fig, "ch22_psm")


def fig_lpwan_map():
    tech = [("LoRaWAN", 0.3, 22, 150, 157, ORANGE), ("Sigfox", 0.1, 0.6, 154, 156, PURPLE),
            ("NB-IoT", 20, 60, 163, 165, NAVY), ("LTE-M", 300, 1000, 155, 157, "#2E86C1"),
            ("Wi-Fi HaLow", 150, 40000, 130, 140, GREEN), ("BLE Coded", 100, 130, 110, 115, ACCENT),
            ("BLE 1M", 900, 1100, 95, 100, ACCENT), ("Wi-Fi 6 (2.4 GHz)", 6000, 300000, 90, 100, GRAY)]
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for nm, r0, r1, b0, b1, c in tech:
        ax.add_patch(Rectangle((r0, b0), r1 - r0, b1 - b0, color=c, alpha=0.35, lw=0))
        ax.text(np.sqrt(r0 * r1), b1 + 0.8, nm, ha="center", fontsize=6.8, color=c)
    ax.set_xscale("log"); ax.set_xlim(0.05, 1e6); ax.set_ylim(85, 172)
    ax.set_xlabel("data rate (kb/s)"); ax.set_ylabel("link budget / MCL (dB)")
    ax.set_title("The low-power wide-area trade: every decade of rate costs about 10 dB of reach", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_lpwan_map")


def fig_backscatter_mirror():
    r = rng(19)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    a = ax[0]
    a.add_patch(Rectangle((0, 1.2), 0.6, 0.8, color=NAVY)); a.text(0.3, 0.95, "reader", ha="center", fontsize=7)
    a.add_patch(Rectangle((4.3, 1.3), 0.2, 0.6, color=GREEN)); a.text(4.4, 0.95, "tag", ha="center", fontsize=7)
    for k in range(5):
        a.annotate("", xy=(4.2, 1.55 + 0.04 * k), xytext=(0.7, 1.55 + 0.04 * k),
                   arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6, alpha=0.6))
    a.annotate("", xy=(4.2, 1.85), xytext=(0.7, 1.85), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
    a.text(2.45, 2.0, "carrier (the flashlight)", ha="center", fontsize=6.8, color=ORANGE)
    a.annotate("", xy=(0.7, 1.3), xytext=(4.2, 1.3), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0, ls="--"))
    a.text(2.45, 1.05, "weak modulated reflection\n(the mirror flicking)", ha="center", fontsize=6.8, color=GREEN)
    a.set_xlim(-0.2, 4.8); a.set_ylim(0.5, 2.4); _clean(a)
    a.set_title("Signalling with someone else's light", fontsize=8.5)
    b = ax[1]
    bits = r.integers(0, 2, 400)
    G = np.where(bits, 0.2 + 0.1j, -0.15 - 0.05j)
    clutter = 1.0 + 0.4j
    y = clutter + G + 0.03 * (r.standard_normal(400) + 1j * r.standard_normal(400))
    b.plot(y.real, y.imag, ".", ms=2, color=NAVY)
    b.plot(clutter.real, clutter.imag, "x", color=ACCENT, ms=8)
    b.text(clutter.real, clutter.imag + 0.08, "static carrier leakage\n+ clutter", fontsize=6.3, ha="center", color=ACCENT)
    b.text(1.22, 0.36, "$\\Gamma_1$", fontsize=8); b.text(0.73, 0.27, "$\\Gamma_2$", fontsize=8)
    b.set_xlabel("I"); b.set_ylabel("Q"); b.set_aspect("equal")
    b.set_title("What the reader sees (baseband)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_backscatter_mirror")


def fig_gen2_q():
    r = rng(20)
    def inventory(ntag, adaptive):
        left = ntag; slots = 0; Qf = 4.0
        while left > 0 and slots < 20000:
            Q = int(round(Qf)) if adaptive else 4
            F = 2 ** Q
            picks = r.integers(0, F, left)
            cnt = np.bincount(picks, minlength=F)
            singles = np.sum(cnt == 1)
            left -= singles; slots += F
            if adaptive:
                e = np.sum(cnt == 0); c = np.sum(cnt > 1)
                Qf = np.clip(Qf + 0.6 * (c - e) / max(1, F) * 3, 0, 15)
                Qf = np.clip(np.log2(max(1, left) * 1.0 + 1e-9), 0, 15) if left > 0 else Qf
        return slots
    n = np.array([10, 30, 100, 300, 1000])
    fig, ax = plt.subplots(figsize=(PW, PH))
    for ad, c, lab in [(False, ACCENT, "fixed frame, Q = 4"), (True, NAVY, "adaptive Q")]:
        eff = [ni / np.mean([inventory(ni, ad) for _ in range(5)]) for ni in n]
        ax.semilogx(n, 100 * np.array(eff), "o-", color=c, label=lab)
    ax.axhline(100 / np.e, color=GRAY, ls=":", lw=0.8); ax.text(12, 39, "1/e = 37%", fontsize=6.5, color=GRAY)
    ax.set_xlabel("tags in the field"); ax.set_ylabel("slots carrying one tag (%)")
    ax.legend(fontsize=6.5); ax.set_ylim(0, 50)
    ax.set_title("Reading a crowd of tags", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_gen2_q")


def fig_nfc_falloff():
    rr = np.logspace(-2, 0, 200)     # 1 cm to 1 m
    a = 0.03
    H = a ** 2 / (2 * (a ** 2 + rr ** 2) ** 1.5)
    P = (H / H[0]) ** 2
    ff = (rr[0] / rr) ** 2
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.loglog(rr * 100, P, color=NAVY, label="near-field coupling ($H^2$)")
    ax.loglog(rr * 100, ff, color=GRAY, ls="--", label="far-field $1/r^2$, for comparison")
    ax.axvline(4, color=ACCENT, ls=":", lw=0.8); ax.text(4.3, 1e-1, "4 cm", fontsize=7, color=ACCENT)
    ax.set_xlabel("distance (cm), 3 cm loop radius"); ax.set_ylabel("relative power at the card")
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_ylim(1e-9, 2)
    ax.set_title("Why you have to tap", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_nfc_falloff")


def fig_energy_ladder():
    items = [("Wi-Fi 7 AP, transmitting", 5), ("phone Wi-Fi, active", 1), ("Wi-Fi station, power save", 0.02),
             ("BLE sensor (1 s events)", 4e-5), ("LoRa/NB-IoT meter, daily", 5e-6),
             ("passive RFID chip, while read", 1e-5),
             ("10 cm$^2$ solar cell indoors", 1e-4), ("RF harvest, 3 m from Wi-Fi AP", 3e-7)]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for k, (nm, p) in enumerate(items):
        c = GREEN if ("solar" in nm or "harvest" in nm) else NAVY
        ax.barh(k, np.log10(p) + 8, left=-8, color=c, alpha=0.8)
        ax.text(np.log10(p) + 0.15, k, nm, va="center", fontsize=6.8)
    ax.set_yticks([]); ax.invert_yaxis()
    ax.set_xticks(range(-7, 2)); ax.set_xticklabels(["0.1 µW", "1 µW", "10 µW", "100 µW", "1 mW", "10 mW", "100 mW", "1 W", "10 W"], fontsize=6.5)
    ax.set_xlim(-7, 4.5); ax.set_xlabel("average power (log scale; green = what can be harvested)")
    ax.set_title("Six orders of magnitude of radio power (approximate)", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_energy_ladder")


def fig_by_numbers():
    fig, ax = plt.subplots(figsize=(W2, 1.6))
    cards = [("1985", "the FCC opens\nunlicensed spread spectrum"), ("1200 MHz", "of new spectrum at\n6 GHz (2020, US)"),
             ("23 Gb/s", "Wi-Fi 7 peak\n(8 streams, 320 MHz)"), ("10 years", "BLE beacon on a\ncoin cell (on paper)"),
             ("$-137$ dBm", "LoRa SF12\nsensitivity"), ("0 batteries", "in a passive\nRFID tag")]
    for k, (big, small) in enumerate(cards):
        x = k * 1.0
        ax.add_patch(FancyBboxPatch((x + 0.04, 0.05), 0.92, 0.9, boxstyle="round,pad=0.02", color=NAVY, alpha=0.08 + 0.0, lw=0))
        ax.text(x + 0.5, 0.66, big, ha="center", va="center", fontsize=11, color=ACCENT, weight="bold")
        ax.text(x + 0.5, 0.3, small, ha="center", va="center", fontsize=6.6, color=NAVY)
    ax.set_xlim(0, 6); ax.set_ylim(0, 1); _clean(ax)
    fig.tight_layout(); save(fig, "ch22_by_numbers")


def fig_ofdm_subcarriers():
    k = np.arange(-32, 32)
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    for kk in k:
        if kk == 0 or abs(kk) > 26:
            c, h = GRAY, 0.25
        elif abs(kk) in (7, 21):
            c, h = ACCENT, 1.0
        else:
            c, h = NAVY, 0.8
        ax.bar(kk, h, width=0.7, color=c)
    ax.text(0, 0.35, "DC\nnull", ha="center", fontsize=6.5, color=GRAY)
    ax.text(-29.5, 0.35, "guard", ha="center", fontsize=6.5, color=GRAY)
    ax.text(29.5, 0.35, "guard", ha="center", fontsize=6.5, color=GRAY)
    for p in (-21, -7, 7, 21):
        ax.text(p, 1.06, "pilot", ha="center", fontsize=6, color=ACCENT)
    ax.set_xlim(-33, 33); ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("subcarrier index (312.5 kHz apart; 48 data in navy, 4 pilots in red)")
    fig.tight_layout(); save(fig, "ch22_ofdm_subcarriers")


def fig_format_detect():
    r = rng(21)
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.0))
    cases = [("L-SIG: BPSK", [0, 0]), ("HT-SIG (11n): QBPSK", [1, 1]), ("VHT-SIG-A: BPSK, then QBPSK", [0, 1])]
    for a, (t, rot) in zip(ax, cases):
        for j, (rr, c) in enumerate(zip(rot, [NAVY, ACCENT])):
            b = 2 * r.integers(0, 2, 48) - 1.0
            z = b * (1j if rr else 1) + 0.15 * (r.standard_normal(48) + 1j * r.standard_normal(48))
            a.plot(z.real, z.imag, "o" if j == 0 else "s", ms=2.8, color=c, alpha=0.8,
                   label="1st symbol" if j == 0 else "2nd symbol")
        a.axhline(0, color=GRAY, lw=0.5); a.axvline(0, color=GRAY, lw=0.5)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=7.5)
    ax[0].legend(fontsize=6, loc="lower left", handletextpad=0.2)
    fig.tight_layout(); save(fig, "ch22_format_detect")


def fig_obss_pd():
    pd = np.linspace(-82, -62, 100)
    tx = np.minimum(21 - (pd + 82), 21)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.plot(pd, tx, color=NAVY, lw=2)
    ax.fill_between(pd, tx, 25, color=ACCENT, alpha=0.08)
    ax.text(-70, 16, "not allowed", color=ACCENT, fontsize=7.5)
    ax.text(-80, 3, "allowed", color=NAVY, fontsize=7.5)
    ax.set_xlabel("OBSS PD threshold (dBm)"); ax.set_ylabel("max transmit power (dBm)")
    ax.set_title("Ignore more, shout less", fontsize=9); ax.set_ylim(0, 23)
    fig.tight_layout(); save(fig, "ch22_obss_pd")


def fig_twt():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    rows = [("sensor (every 30 s)", 30.0, 0.0, GREEN), ("game pad (every 10 ms)", 0.010, 0.0, ACCENT),
            ("thermostat (every 5 s)", 5.0, 2.0, ORANGE)]
    T = 60.0
    for k, (nm, per, off, c) in enumerate(rows):
        ax.add_patch(Rectangle((0, k - 0.3), T, 0.6, color=GRAY, alpha=0.12, lw=0))
        if per < 0.1:
            ax.add_patch(Rectangle((0, k - 0.3), T, 0.6, color=c, alpha=0.25, lw=0))
            ax.text(T / 2, k, "100 short wake-ups a second", ha="center", va="center", fontsize=6.8)
        else:
            t = off
            while t < T:
                ax.add_patch(Rectangle((t, k - 0.3), 0.6, 0.6, color=c, lw=0))
                t += per
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, T); ax.set_xlabel("time (s); coloured = awake at the agreed service period, grey = radio off")
    ax.grid(False); ax.set_ylim(-0.6, 2.6)
    fig.tight_layout(); save(fig, "ch22_twt")


def fig_mlo_links():
    r = rng(23)
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    names = ["2.4 GHz", "5 GHz", "6 GHz"]
    first_free = []
    for k in range(3):
        t = 0
        busy_until = r.uniform(0.5, 4) + k * 0.7
        ax.add_patch(Rectangle((0, k - 0.3), busy_until, 0.6, color=GRAY, alpha=0.45, lw=0))
        first_free.append(busy_until)
        t = busy_until + 3.5
        while t < 12:
            L = r.uniform(1, 2.5)
            ax.add_patch(Rectangle((t, k - 0.3), L, 0.6, color=GRAY, alpha=0.45, lw=0))
            t += L + r.uniform(1.5, 4)
    kbest = int(np.argmin(first_free))
    ax.add_patch(Rectangle((first_free[kbest] + 0.2, kbest - 0.3), 3.0, 0.6, color=NAVY, lw=0))
    ax.text(first_free[kbest] + 1.7, kbest, "your frame", ha="center", va="center", color="w", fontsize=7)
    ax.axvline(0, color=ACCENT, lw=1)
    ax.text(0.1, 2.5, "frame arrives", fontsize=6.8, color=ACCENT)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(-0.2, 12); ax.set_ylim(-0.6, 2.8); ax.grid(False)
    ax.set_xlabel("time (ms); grey = link busy with other traffic")
    fig.tight_layout(); save(fig, "ch22_mlo_links")


def fig_channel_overlap():
    f = np.linspace(2395, 2460, 1000)
    def mask(fc):
        d = np.abs(f - fc)
        return np.where(d < 9, 0, np.where(d < 11, -20 * (d - 9) / 2, np.where(d < 20, -20 - 8 * (d - 11) / 9, np.where(d < 30, -28 - 12 * (d - 20) / 10, -45))))
    fig, ax = plt.subplots(figsize=(PW, PH))
    for ch, c, ls in [(1, NAVY, "-"), (3, ACCENT, "--"), (6, GREEN, "-")]:
        fc = 2407 + 5 * ch
        ax.plot(f, mask(fc), color=c, ls=ls, label=f"channel {ch}")
    ax.set_ylim(-50, 5); ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("relative PSD (dB, mask)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Channel 3 overlaps both neighbours", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_channel_overlap")


def fig_summary_map():
    items = [("NFC", 0.04, 0.4, 106, 424, PURPLE), ("UHF RFID", 2, 15, 40, 640, PURPLE),
             ("BLE", 5, 50, 125, 2000, ACCENT), ("Classic BT", 5, 30, 700, 3000, ACCENT),
             ("Zigbee/Thread", 10, 60, 250, 250, GREEN), ("Wi-Fi", 10, 80, 6e3, 2e6, NAVY),
             ("Wi-Fi HaLow", 100, 1000, 150, 4e4, NAVY), ("LoRa", 700, 1.5e4, 0.29, 5.5, ORANGE),
             ("Sigfox", 1000, 3e4, 0.1, 0.6, ORANGE), ("NB-IoT / LTE-M", 1000, 2e4, 20, 1000, GRAY)]
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    for nm, d0, d1, r0, r1, c in items:
        if r1 <= r0 * 1.05:
            r0, r1 = r0 / 1.15, r1 * 1.15
        ax.add_patch(Rectangle((d0, r0), d1 - d0, r1 - r0, color=c, alpha=0.3, lw=0))
        lx, ly = {"Classic BT": (2.2, 2500), "BLE": (70, 1200), "Zigbee/Thread": (120, 170),
                  "UHF RFID": (5.5, 25)}.get(nm, (np.sqrt(d0 * d1), np.sqrt(r0 * r1)))
        ax.text(lx, ly, nm, ha="center", va="center", fontsize=7, color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.02, 5e4); ax.set_ylim(0.03, 5e6)
    ax.set_xlabel("typical range (m)"); ax.set_ylabel("data rate (kb/s)")
    ax.set_title("The family portrait: every radio in this chapter (rough, typical values)", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_summary_map")


def fig_ltf_estimate():
    r = rng(24)
    L = iot.wifi_ltf_freq()
    k = np.arange(-26, 27)
    h = np.array([0.8, 0, 0.45 * np.exp(1j * 1.0), 0, 0, 0.3 * np.exp(-1j * 2.0)])
    H = np.array([np.sum(h * np.exp(-2j * np.pi * kk * np.arange(len(h)) / 64)) for kk in k])
    snr = 10 ** (10 / 10)
    fig, ax = plt.subplots(figsize=(PW, PH))
    for navg, c, lab in [(1, ORANGE, "one LTF symbol"), (2, NAVY, "two symbols averaged")]:
        Y = np.mean([H * L + (r.standard_normal(53) + 1j * r.standard_normal(53)) / np.sqrt(2 * snr)
                     for _ in range(navg)], axis=0)
        Hh = np.where(L != 0, Y / np.where(L == 0, 1, L), np.nan)
        ax.plot(k, 20 * np.log10(np.abs(Hh)), "o", ms=2.5, color=c, label=lab)
    ax.plot(k, 20 * np.log10(np.abs(H)), color=GRAY, lw=1.2, label="true channel")
    ax.set_xlabel("subcarrier"); ax.set_ylabel("|H| (dB)")
    ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("Channel estimate from the L-LTF, 10 dB SNR", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch22_ltf_estimate")


def fig_minstrel_choice():
    mcs = np.arange(10)
    rate = rate_mbps(mcs, 80)
    snr = 24.0
    p = 1 - _per(snr, mcs)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ax.bar(mcs - 0.2, rate, width=0.4, color=GRAY, alpha=0.6, label="PHY rate $R_m$")
    tp = p * rate
    best = int(np.argmax(tp))
    ax.bar(mcs + 0.2, tp, width=0.4, color=[ACCENT if m == best else NAVY for m in mcs],
           label="expected $p_m R_m$")
    for m in mcs:
        ax.text(m, rate[m] + 12, f"{100*p[m]:.0f}%", ha="center", fontsize=5.8, color=GRAY)
    ax.set_xticks(mcs); ax.set_xlabel("MCS (80 MHz, one stream)"); ax.set_ylabel("Mb/s")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title(f"Minstrel's choice at {snr:.0f} dB: MCS {best}", fontsize=9)
    fig.tight_layout(); save(fig, "ch22_minstrel_choice")


def fig_dechirp_steps():
    M = 128; s = 40
    n = np.arange(M)
    f_rx = ((n + s) % M) / M - 0.5
    f_ref = -(n / M - 0.5)
    f_out = np.mod(f_rx + f_ref + 0.5, 1) - 0.5
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.9), sharey=True)
    for a, f, c, t in [(ax[0], f_rx, NAVY, "received chirp (s = 40)"), (ax[1], f_ref, ACCENT, "times a down-chirp"),
                       (ax[2], f_out, GREEN, "equals a steady tone")]:
        a.plot(n, f * 125, ".", ms=1.5, color=c)
        a.set_title(t, fontsize=8); a.set_xlabel("chip")
    ax[2].axhline(s / M * 125, color=GRAY, lw=0.6, ls=":")
    ax[2].text(5, s / M * 125 + 6, "frequency $\\propto s$", fontsize=6.8, color=GREEN)
    ax[0].set_ylabel("frequency (kHz)")
    ax[0].set_ylim(-70, 70)
    fig.tight_layout(); save(fig, "ch22_dechirp_steps")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    def run(f):
        if not only or any(o in f.__name__ for o in only):
            res = f()
            if res is not None:
                print(f.__name__, res)
    for f in [fig_bands, fig_preamble_sync, fig_rate_range, fig_minstrel, fig_mac_efficiency, fig_ru_tiling,
              fig_ble_gfsk, fig_ble_battery, fig_oqpsk154, fig_lora_chirps, fig_lora_ser, fig_lora_budget,
              fig_rfid_range,
              fig_unlicensed_timeline, fig_psd_limit, fig_dfs, fig_afc, fig_rate_history, fig_barker,
              fig_rate_factors, fig_preamble_overhead, fig_mumimo_beams, fig_ofdma, fig_mlo, fig_sector_sweep,
              fig_sensing, fig_contention_party, fig_dcf_scaling, fig_edca_share, fig_hidden_node, fig_parcels,
              fig_blockack, fig_rate_anomaly, fig_powersave, fig_roaming, fig_mesh_hops, fig_wep_iv, fig_walls,
              fig_neighbours, fig_bt_hopping, fig_piconets, fig_lighthouse, fig_ble_throughput, fig_aoa,
              fig_channel_sounding, fig_coex_levels, fig_microwave, fig_slide_whistle, fig_lora_offsets,
              fig_aloha, fig_sigfox, fig_nbiot_rep, fig_psm, fig_lpwan_map, fig_backscatter_mirror, fig_gen2_q,
              fig_nfc_falloff, fig_energy_ladder, fig_by_numbers,
              fig_ofdm_subcarriers, fig_format_detect, fig_obss_pd, fig_twt, fig_mlo_links,
              fig_channel_overlap, fig_summary_map, fig_ltf_estimate, fig_minstrel_choice, fig_dechirp_steps]:
        run(f)
