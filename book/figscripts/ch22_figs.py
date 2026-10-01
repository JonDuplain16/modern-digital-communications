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


if __name__ == "__main__":
    fig_bands()
    print("cfo est", fig_preamble_sync())
    fig_rate_range()
    print("minstrel oracle/got/fixed", fig_minstrel())
    print("MAC cap", fig_mac_efficiency())
    fig_ru_tiling()
    fig_ble_gfsk()
    print("ble adv q, tpkt, conn q", fig_ble_battery())
    print("154 xcorr max/min", fig_oqpsk154())
    fig_lora_chirps()
    print("lora req", fig_lora_ser())
    fig_lora_budget()
    fig_rfid_range()
