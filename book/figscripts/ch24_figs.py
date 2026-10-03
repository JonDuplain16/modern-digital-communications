"""Figures for Chapter 24: Wireline and Optical Communications."""
from figstyle import *
import commlib as cl
from commlib import wireline as wl

MHZ = 1e6


# ======================================================================= copper
def cat_il_db_100m(f_mhz, cat="6"):
    """TIA-568-style cable insertion loss per 100 m (approximate formulas)."""
    k = {"5e": (1.967, 0.023, 0.050), "6": (1.808, 0.017, 0.200), "6A": (1.82, 0.0091, 0.25)}[cat]
    return k[0] * np.sqrt(f_mhz) + k[1] * f_mhz + k[2] / np.sqrt(f_mhz)


def coax_db_100m(f_mhz):
    """Typical RG-6 drop cable (approximate)."""
    return 0.62 * np.sqrt(f_mhz) + 0.0015 * f_mhz


def fig_line_loss():
    f = np.logspace(3, 8.3, 400)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    a.loglog(f / MHZ, wl.attenuation_db_per_km(f, 26), color=NAVY, label="26 AWG (0.4 mm) pair")
    a.loglog(f / MHZ, wl.attenuation_db_per_km(f, 24), color=ACCENT, label="24 AWG (0.5 mm) pair")
    fc = np.logspace(0, 8.3 - 6, 100)
    a.loglog(fc, 10 * cat_il_db_100m(fc, "6"), color=GREEN, label="Category 6 pair")
    fx = np.logspace(0, 3, 100)
    a.loglog(fx, 10 * coax_db_100m(fx), color=ORANGE, label="RG-6 coax (typical)")
    # slope guides
    g = np.logspace(0.3, 1.7, 10)
    a.loglog(g, 5.0 * np.sqrt(g), color=GRAY, ls=":", lw=1.0)
    a.text(6, 4.2, r"slope $\propto\sqrt{f}$", fontsize=7, color=GRAY)
    g = np.logspace(1.7, 2.3, 10)
    a.loglog(g, 0.35 * g, color=GRAY, ls="--", lw=0.8)
    a.text(25, 40, r"$\propto f$", fontsize=7, color=GRAY)
    a.set_xlabel("frequency (MHz)"); a.set_ylabel("attenuation (dB/km)")
    a.set_xlim(1e-3, 200); a.set_ylim(0.8, 1500)
    a.legend(fontsize=6.6, loc="upper left")
    a.set_title("loss grows with frequency")
    b = ax[1]
    b2 = b.twinx()
    for awg, col in [(26, NAVY), (24, ACCENT)]:
        _, Z0 = wl.propagation(f, awg)
        b.loglog(f / MHZ, np.abs(Z0), color=col, label=f"{awg} AWG")
        b2.semilogx(f / MHZ, np.degrees(np.angle(Z0)), color=col, ls="--", lw=0.8)
    b.axhline(100, color=GRAY, lw=0.7, ls=":")
    b.text(2e-3, 84, "100 $\\Omega$ (DSL, Ethernet)", fontsize=6.6, color=GRAY)
    b.axhline(600, color=GRAY, lw=0.7, ls=":")
    b.text(0.03, 650, "600 $\\Omega$ (voiceband)", fontsize=6.6, color=GRAY)
    b2.text(0.03, -125, "phase (dashed, right axis)", fontsize=6.6, color=GRAY)
    b.set_ylim(70, 1500); b2.set_ylim(-200, 0)
    b2.set_ylabel("phase of $Z_0$ (deg)", fontsize=8); b2.tick_params(labelsize=7.5)
    b2.spines["right"].set_visible(True); b2.grid(False)
    b.set_xlim(1e-3, 200)
    b.set_xlabel("frequency (MHz)"); b.set_ylabel(r"$|Z_0|$ ($\Omega$)")
    b.legend(fontsize=6.6, loc="upper right")
    b.set_title("characteristic impedance")
    fig.tight_layout(); save(fig, "ch24_line_loss")


def fig_loop_impairments():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    # --- loaded vs unloaded 5.5 km (18 kft) 26 AWG, H88 loading, 600 ohm terminations
    f = np.linspace(100, 8000, 600)
    L_tot = 18000 * 0.3048
    unl = wl.cascade(wl.line_abcd(f, L_tot, 26))
    coil = wl.series_abcd(8.0 + 1j * 2 * np.pi * f * 88e-3)
    d = 6000 * 0.3048
    secs = [wl.line_abcd(f, d / 2, 26)]
    for _ in range(2):
        secs += [coil, wl.line_abcd(f, d, 26)]
    secs += [coil, wl.line_abcd(f, L_tot - d / 2 - 2 * d, 26)]
    lod = wl.cascade(*secs)
    a = ax[0]
    a.plot(f / 1e3, 20 * np.log10(np.abs(wl.insertion_gain(unl, 600, 600))), color=NAVY, label="unloaded")
    a.plot(f / 1e3, 20 * np.log10(np.abs(wl.insertion_gain(lod, 600, 600))), color=ACCENT, label="H88 loaded")
    a.axvspan(0.3, 3.4, color=GREEN, alpha=0.08)
    a.text(0.5, -27.5, "voice band", fontsize=7, color=GREEN)
    a.set_ylim(-30, 0); a.set_xlim(0, 8)
    a.set_xlabel("frequency (kHz)"); a.set_ylabel("insertion gain (dB)")
    a.set_title("5.5 km of 26 AWG, 600 $\\Omega$ ends")
    a.legend(fontsize=7, loc="upper right")
    # --- bridged taps on a 1 km 26 AWG loop
    f = np.linspace(20e3, 30e6, 3000)
    b = ax[1]
    base = wl.line_abcd(f, 300, 24)
    b.plot(f / MHZ, 20 * np.log10(np.abs(wl.insertion_gain(base))), color=NAVY, label="300 m, no tap")
    for lt, col in [(25, ACCENT), (80, GREEN)]:
        M = wl.cascade(wl.line_abcd(f, 200, 24), wl.bridged_tap_abcd(f, lt, 24), wl.line_abcd(f, 100, 24))
        b.plot(f / MHZ, 20 * np.log10(np.abs(wl.insertion_gain(M))), color=col, label=f"+ {lt} m bridged tap", lw=1.0)
    b.set_xlabel("frequency (MHz)"); b.set_ylabel("insertion gain (dB)")
    b.set_ylim(-45, 0); b.set_xlim(0, 30)
    b.set_title("bridged taps carve notches (24 AWG)")
    b.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "ch24_loop_impairments")


# --------------------------------------------------------------------- DSL
NOISE = -140.0      # dBm/Hz background noise
GAP = 12.0          # dB: 9.75 dB uncoded gap + 6 dB margin - ~3.75 dB coding gain


def adsl_rate(L, kind="adsl2+", n_fext=10, return_all=False):
    if kind == "adsl":
        tones = np.arange(33, 256)
    else:
        tones = np.arange(33, 512)
    f = tones * 4312.5
    h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
    psd = -40.0
    fext = psd + 10 * np.log10(wl.fext_coupling(f, L, np.sqrt(h2), n_fext) + 1e-30)
    noise = wl.db_sum(np.full_like(f, NOISE), fext)
    r, b, snr = wl.dmt_rate(f, psd, h2, noise, GAP, 15)
    if kind == "adsl":
        r = min(r, 8.1e6)        # G.992.1 framing limits the net rate to about 8 Mb/s
    if return_all:
        return r, f, b, snr
    return r


VDSL_DS = [(138e3, 3.75e6), (5.2e6, 8.5e6), (12e6, 17.664e6)]   # band plan 998ADE17 (downstream)


def vdsl_rate(L, n_fext=20, cancel_db=None, return_all=False, p_tot_dbm=14.5):
    f = np.arange(1, 4096) * 4312.5
    m = np.zeros_like(f, bool)
    for lo, hi in VDSL_DS:
        m |= (f >= lo) & (f < hi)
    f = f[m]
    h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
    psd = -55.0
    for _ in range(4):   # spread the 14.5 dBm aggregate over the usable tones, PSD capped at -40 dBm/Hz
        fx = 10 * np.log10(wl.fext_coupling(f, L, np.sqrt(h2), n_fext) + 1e-40) + psd
        if cancel_db is not None:
            fx = fx - cancel_db
        noise = NOISE if cancel_db == np.inf else wl.db_sum(np.full_like(f, NOISE), fx)
        r, b, snr = wl.dmt_rate(f, psd, h2, noise, GAP, 15)
        nused = max((b > 0).sum(), 1)
        psd = min(-40.0, p_tot_dbm - 10 * np.log10(nused * 4312.5))
    if return_all:
        return r, f, b, snr
    return r


def gfast_rate(L, n_fext=20, cancel_db=30.0):
    df = 51.75e3
    f = np.arange(40, 2048) * df           # ~2.07 MHz to 106 MHz
    h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
    psd = np.where(f < 30e6, -65.0, -76.0)
    fx = 10 * np.log10(wl.fext_coupling(f, L, np.sqrt(h2), n_fext) + 1e-40) + psd - cancel_db
    noise = wl.db_sum(np.full_like(f, NOISE), fx)
    r, b, snr = wl.dmt_rate(f, psd, h2, noise, GAP, 12, sym_rate=48000)
    return r * 0.94   # remove the sync-symbol/TDD guard overhead approximately (aggregate rate)


def fig_crosstalk_vectoring():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    f = np.logspace(4, np.log10(30e6), 400)
    L = 1000
    h = wl.loop_gain(f, L, 26)
    a = ax[0]
    a.semilogx(f / MHZ, 20 * np.log10(np.abs(h)), color=NAVY, label="through signal $|H|^2$")
    a.semilogx(f / MHZ, 10 * np.log10(wl.next_coupling(f, 10)), color=ACCENT, label="NEXT, 10 disturbers")
    a.semilogx(f / MHZ, 10 * np.log10(wl.fext_coupling(f, L, h, 10)), color=GREEN, label="FEXT, 10 disturbers")
    a.set_ylim(-110, 0); a.set_xlim(0.01, 30)
    a.set_xlabel("frequency (MHz)"); a.set_ylabel("power gain (dB)")
    a.set_title("1 km of 26 AWG")
    a.legend(fontsize=6.8, loc="lower left")
    b = ax[1]
    L = 600
    for lab, kw, col in [("FEXT-limited (no vectoring)", dict(cancel_db=None), ACCENT),
                         ("no crosstalk", dict(cancel_db=np.inf), NAVY),
                         ("vectored, 25 dB cancellation", dict(cancel_db=25.0), GREEN)]:
        r, ff, bb, snr = vdsl_rate(L, return_all=True, **kw)
        bb = bb.astype(float)
        # plot per band to avoid lines across gaps
        for lo, hi in VDSL_DS:
            m = (ff >= lo) & (ff < hi)
            b.plot(ff[m] / MHZ, bb[m], color=col, lw=1.0 if col != NAVY else 2.4, alpha=1 if col != NAVY else 0.35,
                   label=f"{lab}: {r/1e6:.0f} Mb/s" if lo == VDSL_DS[0][0] else None)
    b.set_xlabel("frequency (MHz)"); b.set_ylabel("bits per tone")
    b.set_title("VDSL2 17a downstream, 600 m, 20 disturbers")
    b.set_ylim(0, 17); b.set_xlim(0, 18)
    b.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(); save(fig, "ch24_crosstalk_vectoring")


def fig_dsl_rate_reach():
    L = np.linspace(100, 5000, 60)
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.semilogy(L / 1e3, [adsl_rate(x, "adsl") / 1e6 for x in L], color=GRAY, label="ADSL (G.992.1), 1.1 MHz")
    ax.semilogy(L / 1e3, [adsl_rate(x) / 1e6 for x in L], color=NAVY, label="ADSL2+ (G.992.5), 2.2 MHz")
    ax.semilogy(L / 1e3, [vdsl_rate(x) / 1e6 for x in L], color=ORANGE, label="VDSL2 17a (G.993.2), no vectoring")
    ax.semilogy(L / 1e3, [vdsl_rate(x, cancel_db=25.0) / 1e6 for x in L], color=GREEN,
                label="VDSL2 17a, vectored (G.993.5)")
    Lg = np.linspace(20, 600, 40)
    ax.semilogy(Lg / 1e3, [gfast_rate(x) / 1e6 for x in Lg], color=ACCENT,
                label="G.fast 106 MHz (G.9701), aggregate")
    ax.set_xlabel("loop length (km of 26 AWG / 0.4 mm)")
    ax.set_ylabel("line rate (Mb/s)")
    ax.set_xlim(0, 5); ax.set_ylim(0.5, 2000)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("rate versus reach (downstream; G.fast down+up)")
    fig.tight_layout(); save(fig, "ch24_dsl_rate_reach")


# ---------------------------------------------------------------------- cable
def fig_docsis_spectrum():
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    rows = [
        ("DOCSIS 3.0", [(5, 42, "up", "SC-QAM"), (54, 1002, "down", "6 MHz SC-QAM channels (bond up to 32)")]),
        ("DOCSIS 3.1", [(5, 204, "up", "OFDMA"), (258, 1218, "down", "OFDM blocks of 24--192 MHz")]),
        ("DOCSIS 4.0 FDX", [(5, 85, "up", ""), (108, 684, "fdx", "full duplex"), (684, 1218, "down", "")]),
        ("DOCSIS 4.0 FDD", [(5, 684, "up", "upstream to 684"), (834, 1794, "down", "downstream to 1794 MHz")]),
    ]
    cols = {"up": ORANGE, "down": NAVY, "fdx": PURPLE}
    for i, (name, bands) in enumerate(rows):
        y = len(rows) - 1 - i
        for lo, hi, kind, txt in bands:
            ax.add_patch(plt.Rectangle((lo, y - 0.32), hi - lo, 0.64, color=cols[kind], alpha=0.75 if kind != "fdx" else 0.55,
                                       lw=0, hatch="//" if kind == "fdx" else None))
            if txt and hi - lo > 100:
                ax.text((lo + hi) / 2, y, txt, ha="center", va="center", fontsize=6.6, color="white")
        ax.text(-30, y, name, ha="right", va="center", fontsize=7.5)
    if True:
        # 6 MHz channel ticks on DOCSIS 3.0 row
        y = 3
        for c in np.arange(54, 1002, 6 * 8):
            ax.plot([c, c], [y - 0.32, y + 0.32], color="white", lw=0.3)
    ax.set_xlim(0, 1850); ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("frequency (MHz)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="upstream"), Patch(color=NAVY, label="downstream"),
                       Patch(color=PURPLE, alpha=0.6, label="shared (echo-cancelled)")],
              fontsize=7, loc="upper right", ncol=3, bbox_to_anchor=(1.0, -0.33), frameon=False)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.27, top=0.97)
    save(fig, "ch24_docsis_spectrum")


# -------------------------------------------------------------------- Ethernet
def _psd_rect(levels, rate, os=16, nfft=4096):
    from scipy.signal import welch
    x = np.repeat(levels, os)
    f, p = welch(x, fs=rate * os, nperseg=nfft)
    return f, p


def fig_ethernet():
    r = rng(3)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    n = 1 << 17
    bits = r.integers(0, 2, n)
    # 10BASE-T Manchester: 20 MBd half-bit symbols
    man = np.column_stack([2 * bits - 1, 1 - 2 * bits]).ravel()
    f, p = _psd_rect(man, 20e6)
    a.semilogx(f[1:] / MHZ, 10 * np.log10(p[1:] / p[1:].max()), color=GRAY, label="10BASE-T Manchester")
    # 100BASE-TX MLT-3 at 125 MBd (scrambled)
    st = np.cumsum(bits) % 4
    mlt = np.array([0, 1, 0, -1])[st]
    f, p = _psd_rect(mlt, 125e6)
    a.semilogx(f[1:] / MHZ, 10 * np.log10(p[1:] / p[1:].max()), color=NAVY, label="100BASE-TX MLT-3")
    # 1000BASE-T PAM-5 125 MBd with 0.75+0.25D shaping
    s = r.integers(-2, 3, n).astype(float)
    s = 0.75 * s + 0.25 * np.concatenate([[0], s[:-1]])
    f, p = _psd_rect(s, 125e6)
    a.semilogx(f[1:] / MHZ, 10 * np.log10(p[1:] / p[1:].max()), color=GREEN, label="1000BASE-T PAM-5")
    s = r.integers(0, 16, n) * 2 - 15.0
    f, p = _psd_rect(s, 800e6, os=8)
    a.semilogx(f[1:] / MHZ, 10 * np.log10(p[1:] / p[1:].max()), color=ACCENT, label="10GBASE-T PAM-16")
    a.set_xlim(0.3, 1000); a.set_ylim(-40, 3)
    a.set_xlabel("frequency (MHz)"); a.set_ylabel("PSD (dB, normalised)")
    a.set_title("twisted-pair Ethernet spectra")
    a.legend(fontsize=6.4, loc="lower left")
    b = ax[1]
    fm = np.logspace(0, 2, 200)
    il = cat_il_db_100m(fm, "5e") * 0.94 + 4 * 0.04 * np.sqrt(fm)      # 90 m cable + cords + connectors (approx)
    rl = np.where(fm < 20, 17.0, 17 - 10 * np.log10(fm / 20))
    psnext = 27.1 + 15 * np.log10(100 / fm)
    psnext = np.minimum(psnext, 57)
    elfext = 14.4 + 20 * np.log10(100 / fm)
    b.semilogx(fm, -il, color=NAVY, label="far-end signal")
    b.semilogx(fm, -rl, color=ACCENT, label="echo (return loss)")
    b.semilogx(fm, -psnext, color=ORANGE, label="NEXT, power sum of 3 pairs")
    b.semilogx(fm, -(elfext + il), color=GREEN, label="FEXT, power sum")
    b.set_xlabel("frequency (MHz)"); b.set_ylabel("level at receiver (dB)")
    b.set_title("1000BASE-T: 100 m Cat 5e channel")
    b.set_ylim(-70, 0)
    b.legend(fontsize=6.4, loc="lower left")
    fig.tight_layout(); save(fig, "ch24_ethernet")


def fig_access_rates():
    data = {
        "voiceband modems": (GRAY, "o", [(1962, 300e-6, "Bell 103"), (1984, 2.4e-3, "V.22bis"), (1984.4, 9.6e-3, "V.32"),
                                          (1991, 14.4e-3, "V.32bis"), (1994, 28.8e-3, "V.34"), (1998, 56e-3, "V.90")]),
        "DSL": (NAVY, "s", [(1988, 0.144, "ISDN BRI"), (1993, 1.544, "HDSL"), (1999, 8, "ADSL"), (2003, 24, "ADSL2+"),
                            (2006, 100, "VDSL2"), (2014, 1000, "G.fast")]),
        "cable (DOCSIS)": (ORANGE, "^", [(1997, 40, "1.0"), (2006, 1000, "3.0"), (2013, 10000, "3.1"), (2019.5, 10000, "4.0")]),
        "PON": (GREEN, "D", [(1998, 622, "BPON"), (2003, 2488, "GPON"), (2010, 10000, "XG-PON"), (2021, 50000, "50G-PON")]),
    }
    OFFS = {"Bell 103": (0, 0.55, "center"), "V.22bis": (1.0, 0.55, "left"), "V.32": (-1.0, 1.6, "right"),
            "V.32bis": (0, 0.55, "center"), "V.34": (0, 0.55, "center"), "V.90": (0.6, 0.55, "left"),
            "3.1": (0, 0.42, "center"), "4.0": (0, 0.42, "center"), "1.0": (-0.8, 1.0, "right"),
            "3.0": (0.8, 0.6, "left"), "ISDN BRI": (0.8, 0.55, "left"), "HDSL": (0.8, 0.55, "left"),
            "ADSL": (0.8, 0.55, "left"), "ADSL2+": (0.8, 0.55, "left"), "VDSL2": (0.8, 0.55, "left"),
            "G.fast": (0.8, 0.7, "left")}
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    for name, (col, mk, pts) in data.items():
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        ax.semilogy(x, y, mk + "-", color=col, ms=4.5, lw=1.0, label=name)
        for xx, yy, t in pts:
            dx, dy, ha = OFFS.get(t, (0, 1.6, "center"))
            ax.text(xx + dx, yy * dy, t, fontsize=6.2, color=col, ha=ha, va="bottom" if dy > 1 else "top")
    yrs = np.array([1983, 2025])
    ax.semilogy(yrs, 0.3e-3 * 1.5 ** (yrs - 1983) * 10, color=GRAY, ls=":", lw=0.8)
    ax.text(2008, 2.0e2*0.06, "+50 % per year", fontsize=6.5, color=GRAY, rotation=27)
    ax.set_xlabel("year (approximate standard publication)"); ax.set_ylabel("maximum downstream rate (Mb/s)")
    ax.set_xlim(1958, 2026); ax.set_ylim(1e-4, 3e5)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "ch24_access_rates")


# ======================================================================= fibre
def fig_fibre_loss():
    lam = np.linspace(800, 1700, 900)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    bands = [("O", 1260, 1360), ("E", 1360, 1460), ("S", 1460, 1530), ("C", 1530, 1565), ("L", 1565, 1625), ("U", 1625, 1675)]
    for i, (n, lo, hi) in enumerate(bands):
        ax.axvspan(lo, hi, color=[NAVY, GRAY, GREEN, ACCENT, ORANGE, PURPLE][i], alpha=0.07, lw=0)
        ax.text((lo + hi) / 2, 2.6, n, ha="center", fontsize=8, color=[NAVY, GRAY, GREEN, ACCENT, ORANGE, PURPLE][i])
    lu = lam / 1e3
    ax.semilogy(lam, wl.fibre_attenuation_db_km(lam, True), color=NAVY, label="standard SMF (G.652.A/B, with water peak)")
    ax.semilogy(lam, wl.fibre_attenuation_db_km(lam, False), color=GREEN, ls="--", label="low-water-peak SMF (G.652.D)")
    ax.semilogy(lam, 0.90 / lu ** 4, color=GRAY, ls=":", lw=0.9, label=r"Rayleigh scattering $\propto\lambda^{-4}$")
    ax.semilogy(lam, 7.81e11 * np.exp(-48.48 / lu), color=ORANGE, ls=":", lw=0.9, label="infrared absorption")
    ax.annotate("OH$^-$ peak 1383 nm", (1375, 0.75), (1080, 1.0), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("about 0.17--0.2 dB/km\nat 1550 nm", (1550, 0.187), (1580, 0.45), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.axvline(850, color=GRAY, lw=0.6); ax.text(858, 2.6, "850 nm\n(multimode)", fontsize=6.6, color=GRAY, va="top")
    ax.set_xlim(800, 1700); ax.set_ylim(0.1, 3.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3]); ax.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "1", "2", "3"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("attenuation (dB/km)")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch24_fibre_loss")


def fig_dispersion():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.95))
    lam = np.linspace(1250, 1650, 300)
    a = ax[0]
    a.plot(lam, wl.dispersion_ps_nm_km(lam), color=NAVY, label="standard SMF (G.652)")
    a.plot(lam, wl.dispersion_ps_nm_km(lam, 1450, 0.07), color=GREEN, label="non-zero DSF (G.655), approx.")
    a.plot(lam, wl.dispersion_ps_nm_km(lam, 1550, 0.07), color=ORANGE, label="dispersion-shifted (G.653)")
    a.axhline(0, color="k", lw=0.5)
    a.axvspan(1530, 1565, color=ACCENT, alpha=0.08); a.text(1535, -14, "C", color=ACCENT, fontsize=8)
    a.set_xlabel("wavelength (nm)"); a.set_ylabel(r"$D$ (ps/(nm$\cdot$km))")
    a.set_ylim(-16, 24)
    a.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    a.set_title("chromatic dispersion")
    b = ax[1]
    f = np.linspace(0.01, 30, 800) * 1e9
    for Lk, col in [(2, GREEN), (10, NAVY), (80, ACCENT)]:
        h = wl.imdd_cd_response(f, Lk * 1e3)
        b.plot(f / 1e9, 20 * np.log10(np.abs(h) + 1e-6), color=col, label=f"{Lk} km at 1550 nm")
    for fn, t in [(5, "10G NRZ"), (13.3, "26.6 GBd"), (26.6, "53 GBd")]:
        b.axvline(fn, color=GRAY, lw=0.6, ls=":")
        b.text(fn + 0.3, -27, t, fontsize=6.3, color=GRAY, rotation=90, va="bottom")
    b.set_ylim(-30, 3); b.set_xlim(0, 30)
    b.set_xlabel("modulation frequency (GHz)"); b.set_ylabel("IM/DD response (dB)")
    b.set_title("power fading in direct detection")
    b.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False)
    fig.tight_layout(); save(fig, "ch24_dispersion")


def fig_rx_sensitivity():
    from scipy.special import erfc
    q = wl.Q_E
    Rb = 10e9; B = 0.7 * Rb
    Rresp = 0.9                       # A/W
    i_th = 15e-12                     # input-referred TIA noise, A/sqrt(Hz)
    sT = i_th * np.sqrt(B)
    kA = 0.5

    def Qfac(P_avg, M=1.0, kA=kA):
        P1 = 2 * P_avg
        F = kA * M + (1 - kA) * (2 - 1 / M) if M > 1 else 1.0
        I1 = Rresp * M * P1
        s1 = np.sqrt(2 * q * Rresp * P1 * M ** 2 * F * B + sT ** 2)
        return I1 / (s1 + sT)

    P_dbm = np.linspace(-40, -10, 300)
    P = 1e-3 * 10 ** (P_dbm / 10)
    ber = lambda Q: 0.5 * erfc(Q / np.sqrt(2))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    a.semilogy(P_dbm, ber(Qfac(P)), color=NAVY, label="PIN (thermal-limited)")
    Ms = np.linspace(1.5, 40, 200)
    Qapd = np.array([max(Qfac(p, M) for M in Ms) for p in P])
    a.semilogy(P_dbm, ber(Qapd), color=ACCENT, label="APD, best $M$")
    # quantum limit: P_e = 0.5 exp(-N1), average photons/bit Nbar = N1/2
    hnu = wl.H_PLANCK * wl.C / 1550e-9
    Nbar = P / (hnu * Rb)
    a.semilogy(P_dbm, 0.5 * np.exp(-2 * Nbar), color=GRAY, ls="--", label="quantum limit")
    a.set_ylim(1e-13, 1e-1); a.set_xlim(-55, -10)
    P_dbm2 = np.linspace(-55, -10, 400); P2 = 1e-3 * 10 ** (P_dbm2 / 10)
    a.semilogy(P_dbm2, 0.5 * np.exp(-2 * P2 / (hnu * Rb)), color=GRAY, ls="--")
    a.axhline(1e-12, color=GRAY, lw=0.5)
    a.set_xlabel("average received power (dBm)"); a.set_ylabel("bit error rate")
    a.set_title("10 Gb/s OOK at 1550 nm")
    a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.42, 1.0))
    b = ax[1]
    for pd, col in [(-30, NAVY), (-26, GREEN), (-22, ORANGE)]:
        p = 1e-3 * 10 ** (pd / 10)
        Mg = np.linspace(1.01, 40, 300)
        b.plot(Mg, [20 * np.log10(Qfac(p, M)) for M in Mg], color=col, label=f"{pd} dBm")
    b.set_xlabel("APD gain $M$"); b.set_ylabel("$20\\log_{10}Q$ (dB)")
    b.set_title("an optimum avalanche gain")
    b.axhline(20 * np.log10(7.03), color=GRAY, lw=0.6, ls=":")
    b.text(25, 20 * np.log10(7.03) + 0.4, "BER $10^{-12}$", fontsize=6.6, color=GRAY)
    b.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout(); save(fig, "ch24_rx_sensitivity")
    # print sensitivities for the text
    for name, Qc in [("PIN", Qfac(P)), ("APD", Qapd)]:
        print(name, "sens @1e-12:", np.interp(7.03, Qc, P_dbm), "dBm")


def req_snr_db(M, ber=2e-2):
    """SNR (Es/N0 per polarisation) for a given pre-FEC BER, Gray-coded square QAM."""
    from scipy.optimize import brentq
    f = lambda s: cl.ber_mqam_gray(s, M) - ber
    return brentq(f, -5, 40)


def fig_osnr_nli():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4))
    span_db = 80 * 0.22  # 17.6 dB incl. splices
    N = np.arange(1, 41)
    a = ax[0]
    for p, col in [(-2, GREEN), (0, NAVY), (2, ACCENT)]:
        a.plot(N * 80, wl.osnr_db(p, span_db, 5.0, N), color=col, label=f"{p:+d} dBm/ch")
    r1 = req_snr_db(4) + 10 * np.log10(32 / 12.5) + 2
    r2 = req_snr_db(16) + 10 * np.log10(64 / 12.5) + 2
    for r, t in [(r1, "100G DP-QPSK"), (r2, "400G DP-16QAM")]:
        a.axhline(r, color=GRAY, ls=":", lw=0.8)
        a.text(60, r + 0.5, t, fontsize=6.2, color=GRAY)
    a.set_xlabel("distance (km), 80 km spans"); a.set_ylabel("OSNR in 0.1 nm (dB)")
    a.set_title("linear OSNR budget"); a.legend(fontsize=6.2, loc="upper right")
    a.set_ylim(8, 38); a.set_xlim(0, 3200)
    b = ax[1]
    Pdbm = np.linspace(-6, 8, 200); Pw = 1e-3 * 10 ** (Pdbm / 10)
    eta = wl.gn_eta()
    for n, col in [(5, GREEN), (13, NAVY), (40, ACCENT)]:
        snr = 10 * np.log10(wl.gn_snr(Pw, n, span_db, 5.0, eta=eta))
        b.plot(Pdbm, snr, color=col, label=f"{n*80} km")
        k = np.argmax(snr); b.plot(Pdbm[k], snr[k], "o", color=col, ms=3)
        lin = 10 * np.log10(wl.gn_snr(Pw, n, span_db, 5.0, eta=0.0))
        b.plot(Pdbm, lin, color=col, ls=":", lw=0.7)
    b.set_xlabel("launch power (dBm/ch)"); b.set_ylabel("SNR (dB)")
    b.set_title("ASE + nonlinearity"); b.legend(fontsize=6.2, loc="upper left")
    b.set_ylim(4, 28)
    c = ax[2]
    dist = np.arange(1, 81) * 80
    se = []
    for n in range(1, 81):
        snr = wl.gn_snr(Pw, n, span_db, 5.0, eta=eta).max()
        se.append(2 * np.log2(1 + snr) * 64 / 75)
    c.semilogx(dist, se, color=NAVY)
    c.set_xlabel("distance (km)"); c.set_ylabel("spectral efficiency (b/s/Hz)")
    c.set_title("nonlinear Shannon limit")
    c.set_xlim(80, 6400)
    fig.tight_layout(); save(fig, "ch24_osnr_nli")
    snr13 = 10 * np.log10(wl.gn_snr(Pw, 13, span_db, 5.0, eta=eta))
    print("req OSNR qpsk/16qam", r1, r2, "| 13 spans opt", Pdbm[np.argmax(snr13)], snr13.max(),
          "| OSNR 13 spans 1 dBm", wl.osnr_db(1, span_db, 5, 13), "| eta", eta)
    print("SE at 1000, 6400 km", se[12], se[-1])


def fig_coherent_dsp():
    r = rng(7)
    rs = 32e9; sps = 2; fs = rs * sps
    nsym = 12000
    qpsk = (np.array([1, -1])[r.integers(0, 2, (2, nsym))] + 1j * np.array([1, -1])[r.integers(0, 2, (2, nsym))]) / np.sqrt(2)
    taps = cl.rrc_taps(0.1, sps, span=32)
    tx = np.array([cl.shape(qpsk[p], taps, sps) for p in range(2)])
    tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2, axis=1, keepdims=True))
    Lm = 80e3
    # polarization rotation + small DGD
    th, ph = 0.6, 1.1
    U = np.array([[np.cos(th), -np.sin(th) * np.exp(-1j * ph)], [np.sin(th) * np.exp(1j * ph), np.cos(th)]])
    f = np.fft.fftfreq(tx.shape[1], 1 / fs)
    dgd = 8e-12
    Xf = np.fft.fft(tx, axis=1)
    Xf[0] *= np.exp(-1j * np.pi * f * dgd); Xf[1] *= np.exp(1j * np.pi * f * dgd)
    rx = U @ np.fft.ifft(Xf, axis=1)
    rx = np.array([wl.apply_cd(rx[p], fs, Lm) for p in range(2)])
    # laser phase noise (combined linewidth 200 kHz) and 150 MHz frequency offset
    n = rx.shape[1]
    dphi = r.normal(0, np.sqrt(2 * np.pi * 200e3 / fs), n)
    phase = np.cumsum(dphi) + 2 * np.pi * 150e6 * np.arange(n) / fs
    rx = rx * np.exp(1j * phase)
    snr_db = 16.0
    sig = np.mean(np.abs(rx) ** 2)
    rx = rx + np.sqrt(sig / 10 ** (snr_db / 10) / 2 * sps) * (r.normal(size=rx.shape) + 1j * r.normal(size=rx.shape))
    # matched filter
    mf = np.array([np.convolve(rx[p], taps, mode="same") for p in range(2)])
    stage0 = mf[0, ::sps][200:-200]
    cdc = np.array([wl.cd_compensate(mf[p], fs, Lm) for p in range(2)])
    stage1 = cdc[0, ::sps][200:-200]
    zx, zy, W = wl.cma_butterfly(cdc[0] / np.sqrt(np.mean(np.abs(cdc[0]) ** 2)),
                                 cdc[1] / np.sqrt(np.mean(np.abs(cdc[1]) ** 2)), ntaps=15, mu=2e-3, sps=2)
    stage2 = zx[3000:]
    fo = wl.fourth_power_fo(zx[3000:], rs)
    zc = zx[3000:] * np.exp(-1j * 2 * np.pi * fo * np.arange(len(zx) - 3000) / rs)
    stage3, _ = wl.vv_carrier_recovery(zc, block=40)
    stage3 = stage3[50:-50]
    print("FO estimate (MHz):", fo / 1e6)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85))
    titles = ["received (80 km CD)", "after CD compensation", "after CMA butterfly", "after carrier recovery"]
    for a, z, t in zip(ax, [stage0, stage1, stage2, stage3], titles):
        z = z[-3000:]
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        a.plot(z.real, z.imag, ".", ms=0.8, alpha=0.4, color=NAVY)
        a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_aspect("equal")
        a.set_title(t, fontsize=7.5); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
        a.tick_params(labelsize=6.5)
    fig.tight_layout(); save(fig, "ch24_coherent_dsp")


# ======================================================================= 2nd edition: concept figures
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch, Ellipse
SKY = "#2E86C1"


def _clean(ax):
    ax.set_aspect("equal"); ax.axis("off")


def _arrow(ax, p0, p1, col=NAVY, lw=1.0, style="-|>", ms=8, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=col, lw=lw, **kw))


def fig_by_numbers():
    tiles = [("> 90 %", "of intercontinental traffic\nrides submarine cables"),
             ("0.17 dB/km", "best fibre at 1550 nm: half\nthe light survives 17 km"),
             ("~50 THz", "low-loss window of\none strand of glass"),
             ("250 Tb/s", "design capacity of one\ntransatlantic cable (Dunant)"),
             ("$-$48 V", "on a telephone pair,\nas in the 1890s"),
             ("> $10^7\\times$", "growth of home access rates,\n300 b/s to 10 Gb/s"),
             ("~1 Gb/s", "G.fast over 50 m of\nold telephone wire"),
             ("~10 photons", "per bit: the quantum\nlimit at BER $10^{-9}$")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, SKY, ACCENT, ORANGE, GREEN, PURPLE, NAVY, GRAY]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.64, big, ha="center", va="center", fontsize=13, color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=7.0, color="#333333", linespacing=1.1)
    save(fig, "ch24_by_numbers")


def fig_rope():
    """A flick on a rope: a pulse on a line and its reflection from open, shorted and matched ends."""
    x = np.linspace(0, 1, 600)
    g = lambda u: np.exp(-((u) / 0.05) ** 2)
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.3), sharey=True)
    cases = [("open end ($\\Gamma=+1$)", 1.0, ACCENT), ("shorted end ($\\Gamma=-1$)", -1.0, NAVY),
             ("matched load ($\\Gamma=0$)", 0.0, GREEN)]
    times = [0.3, 0.6, 0.9, 1.2, 1.5]
    for a, (t, G, col) in zip(ax, cases):
        for k, tt in enumerate(times):
            v = g(x - tt) + G * g(2 - x - tt)
            off = -1.6 * k
            a.plot(x, v + off, color=col, lw=1.2)
            a.axhline(off, color=GRAY, lw=0.4)
            a.text(-0.04, off + 0.15, f"t{k+1}", fontsize=6.5, color=GRAY, ha="right")
        a.plot([1, 1], [-7.3, 1.3], color="k", lw=2)
        a.set_title(t, fontsize=8)
        a.set_xlim(-0.1, 1.08); a.axis("off")
    ax[0].text(0.02, 1.25, "pulse launched $\\rightarrow$", fontsize=6.6, color=GRAY)
    fig.tight_layout(); save(fig, "ch24_rope")


def fig_skin():
    """Current crowding in a 0.4 mm copper wire at four frequencies (exact Bessel solution)."""
    from scipy.special import jv
    a0 = 0.2e-3
    mu0 = 4e-7 * np.pi; sig = 5.8e7
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.75))
    n = 241
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    rr = np.hypot(xx, yy)
    for a, f in zip(ax, [1e3, 100e3, 1e6, 10e6]):
        d = 1 / np.sqrt(np.pi * f * mu0 * sig)
        k = (1 - 1j) / d
        J = np.abs(jv(0, k * rr * a0)) / np.abs(jv(0, k * a0))
        J = np.where(rr <= 1, J, np.nan)
        a.imshow(J, extent=[-1, 1, -1, 1], cmap="YlOrRd", vmin=0, vmax=1, origin="lower")
        a.add_patch(Circle((0, 0), 1, fc="none", ec="k", lw=0.6))
        lab = f"{f/1e3:.0f} kHz" if f < 1e6 else f"{f/1e6:.0f} MHz"
        ds = f"{d*1e3:.2f} mm" if d > 1e-4 else f"{d*1e6:.0f} $\\mu$m"
        a.set_title(lab, fontsize=8.5)
        a.text(0, -1.32, f"$\\delta$ = {ds}", ha="center", fontsize=7.5, color=NAVY)
        a.set_xlim(-1.1, 1.1); a.set_ylim(-1.5, 1.1); _clean(a)
    fig.suptitle("current density in a 0.4 mm copper wire (darker = more current)", fontsize=8, y=1.0)
    fig.tight_layout(); save(fig, "ch24_skin")


def fig_tdr():
    """Simulated TDR of a loop: 24 AWG, a bridged tap, a gauge change and an open end."""
    N = 2 ** 14; fmax = 60e6
    f = np.linspace(fmax / N, fmax, N)
    Zl = 1e9
    sec1 = wl.line_abcd(f, 250, 24)
    tap = wl.bridged_tap_abcd(f, 60, 24)
    sec2 = wl.line_abcd(f, 200, 24)
    sec3 = wl.line_abcd(f, 200, 26)
    M = wl.cascade(sec1, tap, sec2, sec3)
    A, B, Cm, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    Zin = (A * Zl + B) / (Cm * Zl + D)
    Zs = 100.0
    Gam = (Zin - Zs) / (Zin + Zs)
    tau = 60e-9
    P = np.exp(-(np.pi * f * tau) ** 2)
    spec = np.concatenate([[0], Gam * P])
    h = np.fft.irfft(spec)
    dt = 1 / (2 * fmax)
    t = np.arange(len(h)) * dt
    v = 0.66 * 3e8
    dist = v * t / 2
    m = (dist < 900) & (dist > 100)
    h = h / np.abs(h[m]).max()
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(dist[m], h[m], color=NAVY)
    ax.axhline(0, color=GRAY, lw=0.5)
    for d0, txt, y in [(250, "bridged-tap junction:\nimpedance dips", -0.8), (310, "end of the\n60 m tap", 0.75),
                       (450, "24$\\rightarrow$26 AWG:\nimpedance rises", 0.75), (650, "open end\nof the loop", 0.75)]:
        ax.axvline(d0, color=ACCENT, lw=0.6, ls=":")
        ax.text(d0 + 8, y, txt, fontsize=6.6, color=ACCENT, va="center")
    ax.set_xlabel("distance from the tester (m), assuming $v=0.66c$"); ax.set_ylabel("echo (normalised)")
    ax.set_xlim(150, 850); ax.set_ylim(-1.1, 1.15)
    ax.set_title("a time-domain reflectometer listening to a loop", fontsize=9)
    fig.tight_layout(); save(fig, "ch24_tdr")


def fig_twist():
    """Why twisting works: induced voltage accumulated along a pair, untwisted vs twisted."""
    z = np.linspace(0, 2.0, 2000)
    fig, ax = plt.subplots(2, 1, figsize=(3.1, 2.7), gridspec_kw=dict(height_ratios=[0.8, 1.4]))
    a = ax[0]
    p = 0.25
    a.plot(z, 0.5 * np.sin(2 * np.pi * z / p), color=NAVY, lw=1.6)
    a.plot(z, -0.5 * np.sin(2 * np.pi * z / p), color=ACCENT, lw=1.6)
    for k in range(8):
        zc = (k + 0.5) * p / 2
        a.text(zc, 0, "+" if k % 2 == 0 else "$-$", ha="center", va="center", fontsize=7, color=GREEN, weight="bold")
    a.set_xlim(0, 2); a.set_ylim(-0.75, 0.75); a.axis("off")
    a.set_title("each half-twist picks up an opposite-sign voltage", fontsize=7.5)
    b = ax[1]
    hfield = 1 + 0.3 * np.sin(2 * np.pi * z / 1.7)      # slowly varying interfering field
    dz = z[1] - z[0]
    b.plot(z, np.cumsum(hfield) * dz, color=GRAY, ls="--", label="untwisted (parallel) pair")
    sgn = np.sign(np.sin(2 * np.pi * z / p + 1e-9))
    b.plot(z, np.cumsum(hfield * sgn) * dz, color=GREEN, label="twisted pair")
    b.set_xlabel("distance along the cable (m)", fontsize=8); b.set_ylabel("induced voltage", fontsize=8)
    b.legend(fontsize=6.5, loc="upper left"); b.tick_params(labelsize=7); b.set_yticks([])
    fig.tight_layout(); save(fig, "ch24_twist")


def fig_xtalk_geometry():
    """NEXT and FEXT drawn as two pairs in one cable."""
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.4, 0.75), 7.2, 1.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc=GRAY, alpha=0.08, ec=GRAY))
    zz = np.linspace(1.6, 8.4, 400)
    ax.plot(zz, 1.9 + 0.08 * np.sin(zz * 14), color=NAVY, lw=1.4)
    ax.plot(zz, 1.9 - 0.08 * np.sin(zz * 14), color=NAVY, lw=1.4)
    ax.plot(zz, 1.2 + 0.08 * np.sin(zz * 11), color=ACCENT, lw=1.4)
    ax.plot(zz, 1.2 - 0.08 * np.sin(zz * 11), color=ACCENT, lw=1.4)
    ax.text(0.9, 1.9, "Tx", ha="center", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc=NAVY, ec="none"))
    ax.text(9.1, 1.9, "Rx", ha="center", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc=NAVY, ec="none"))
    ax.text(0.9, 1.2, "Rx", ha="center", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc=ACCENT, ec="none"))
    ax.text(9.1, 1.2, "Rx", ha="center", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc=ACCENT, ec="none"))
    ax.text(5, 2.55, "disturbing pair: signal travels left to right", ha="center", fontsize=7.5, color=NAVY)
    ax.text(5, 0.45, "victim pair", ha="center", fontsize=7.5, color=ACCENT)
    for x0 in [2.2, 3.4, 4.6]:
        _arrow(ax, (x0, 1.78), (x0 - 0.9, 1.32), col=ORANGE, lw=0.9)
    for x0 in [5.6, 6.8, 8.0]:
        _arrow(ax, (x0 - 0.9, 1.78), (x0, 1.32), col=PURPLE, lw=0.9)
    ax.text(0.5, 0.55, "NEXT: heard at the near end,\nstrong, unattenuated", fontsize=7, color=ORANGE, ha="left")
    ax.text(9.6, 0.55, "FEXT: heard at the far end,\nattenuated with the signal", fontsize=7, color=PURPLE, ha="right")
    save(fig, "ch24_xtalk_geometry")


def fig_thl():
    """Transhybrid loss of a compromise balance network across real loops."""
    f = np.linspace(200, 3400, 300)
    w = 2 * np.pi * f
    ZB = 900 + 1 / (1j * w * 2.16e-6)
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    loops = [("26 AWG, 1 km", 1000, 26, NAVY), ("26 AWG, 3 km", 3000, 26, ACCENT),
             ("24 AWG, 5 km", 5000, 24, GREEN), ("26 AWG, 0.3 km", 300, 26, ORANGE)]
    for name, L, awg, col in loops:
        M = wl.line_abcd(f, L, awg)
        A, B, Cm, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
        ZL = 600.0
        Zin = (A * ZL + B) / (Cm * ZL + D)
        thl = -20 * np.log10(np.abs((Zin - ZB) / (Zin + ZB)))
        ax.plot(f / 1e3, thl, color=col, label=name)
    ax.axhspan(6, 20, color=GRAY, alpha=0.08)
    ax.set_xlabel("frequency (kHz)", fontsize=8); ax.set_ylabel("transhybrid loss (dB)", fontsize=8)
    ax.set_ylim(0, 40); ax.legend(fontsize=6.3, loc="upper right"); ax.tick_params(labelsize=7)
    ax.set_title("one balance network, many loops", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_thl")


def fig_modem_rates():
    """Voiceband modem rates against the Shannon capacity of the analog voice channel."""
    pts = [(1962, 0.3, "Bell 103"), (1976, 9.6, "V.29"), (1980, 1.2, "212A"), (1984, 2.4, "V.22bis"),
           (1985, 9.6, "V.32"), (1991, 14.4, "V.32bis"), (1994, 28.8, "V.34"), (1996, 33.6, "V.34+"),
           (1998, 56, "V.90")]
    fig, ax = plt.subplots(figsize=(3.1, 2.55))
    for x, y, t in pts:
        col = ACCENT if t == "V.90" else NAVY
        ax.semilogy(x, y, "o", color=col, ms=4)
        if t == "V.90":
            ax.text(x - 0.8, y * 0.95, t, fontsize=6.2, color=col, va="top", ha="right")
        else:
            ax.text(x + 0.6, y * 0.9, t, fontsize=6.2, color=col, va="top")
    B = 3100
    for snr, ls, fac, va in [(30, ":", 0.92, "top"), (38, "--", 1.06, "bottom")]:
        C = B * np.log2(1 + 10 ** (snr / 10)) / 1e3
        ax.axhline(C, color=GREEN, ls=ls, lw=0.9)
        ax.text(1961, C * fac, f"Shannon, 3.1 kHz at {snr} dB: {C:.0f} kb/s", fontsize=6, color=GREEN, va=va)
    ax.axhline(64, color=GRAY, lw=0.8)
    ax.text(1961, 70, "64 kb/s PCM channel", fontsize=6, color=GRAY)
    ax.set_xlim(1960, 2003); ax.set_ylim(0.2, 120)
    ax.set_xlabel("year", fontsize=8); ax.set_ylabel("rate (kb/s)", fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_modem_rates")


def mulaw_levels():
    vals = []
    for s in range(8):
        for q in range(16):
            vals.append(((2 * q + 33) << s) - 33)
    return np.array(sorted(set(vals)))


def fig_v90_levels():
    lv = mulaw_levels()                     # 128 non-negative magnitudes, 0..8031 (14-bit units)
    allv = np.concatenate([-lv[::-1], lv[1:]])
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.1), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    a.plot(np.arange(len(allv)) - len(allv) // 2, allv / 8031, ".", ms=2.2, color=NAVY)
    a.set_xlabel("PCM codeword", fontsize=8); a.set_ylabel("line voltage (normalised)", fontsize=8)
    a.set_title("the 255 $\\mu$-law levels a line card can produce", fontsize=8)
    a.tick_params(labelsize=7)
    b = ax[1]
    pos = lv / 8031
    use = np.zeros(len(pos), bool); last = -1.0
    for i, v in enumerate(pos):
        if v - last >= 0.006:
            use[i] = True; last = v
    b.vlines(pos[~use], 0, 1, color=GRAY, lw=0.6)
    b.vlines(pos[use], 0, 1, color=ACCENT, lw=0.8)
    b.set_xlim(0, 0.2); b.set_yticks([])
    b.set_xlabel("level (normalised), zoom near zero", fontsize=8); b.tick_params(labelsize=7)
    b.set_title("a usable subset (red), spaced well apart", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_v90_levels")


def fig_stuffing():
    """Positive justification: a tributary buffer and the stuffing decisions of a DS2 multiplexer."""
    nfr = 60
    r_in = 1.544e6; fr = 6.312e6 / 1176
    per = r_in / fr                # ~287.67 bits per frame
    fill = 4.0; F = []; ev = []
    for k in range(nfr):
        fill += per - 287
        if fill > 4.5:
            fill -= 1; ev.append(1)
        else:
            ev.append(0)
        F.append(fill)
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.step(np.arange(nfr), F, where="post", color=NAVY, label="tributary buffer fill (bits)")
    for k, e in enumerate(ev):
        ax.plot(k, 3.6, "|", color=GREEN if e else ACCENT, ms=7)
    ax.text(nfr + 0.5, 3.6, "stuff opportunity:\ngreen = data, red = dummy", fontsize=6.3, va="center")
    ax.set_xlabel("DS2 frame number (1176 bits each)", fontsize=8); ax.set_ylabel("bits", fontsize=8)
    ax.set_xlim(-1, nfr + 13); ax.set_ylim(3.3, 4.8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.6, loc="upper right")
    fig.tight_layout(); save(fig, "ch24_stuffing")


def fig_rate_ladder():
    items = [("DS0", 64e3, GRAY), ("DS1 (T1)", 1.544e6, NAVY), ("E1", 2.048e6, NAVY), ("DS3 (T3)", 44.736e6, NAVY),
             ("E4", 139.264e6, NAVY), ("STS-1", 51.84e6, GREEN), ("OC-3 / STM-1", 155.52e6, GREEN),
             ("OC-12 / STM-4", 622.08e6, GREEN), ("OC-48 / STM-16", 2.48832e9, GREEN),
             ("OC-192 / STM-64", 9.95328e9, GREEN), ("OC-768 / STM-256", 39.81312e9, GREEN),
             ("OTU2", 10.709e9, ACCENT), ("OTU4", 111.81e9, ACCENT), ("OTUC4 ($\\approx$)", 4 * 105.26e9, ACCENT)]
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    y = np.arange(len(items))[::-1]
    for yy, (n, r, c) in zip(y, items):
        ax.barh(yy, np.log10(r) - 4, left=4, color=c, alpha=0.8, height=0.65)
        rs = f"{r/1e9:.4g} Gb/s" if r >= 1e9 else (f"{r/1e6:.4g} Mb/s" if r >= 1e6 else f"{r/1e3:.0f} kb/s")
        ax.text(np.log10(r) + 0.08, yy, rs, va="center", fontsize=6.5, color=c)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=7)
    ax.set_xticks(range(4, 13)); ax.set_xticklabels(["10 k", "100 k", "1 M", "10 M", "100 M", "1 G", "10 G", "100 G", "1 T"], fontsize=7)
    ax.set_xlim(4, 13.2); ax.set_xlabel("line rate (b/s)", fontsize=8)
    ax.text(12.0, 12.5, "PDH", color=NAVY, fontsize=8, weight="bold")
    ax.text(12.0, 11.7, "SONET/SDH", color=GREEN, fontsize=8, weight="bold")
    ax.text(12.0, 10.9, "OTN", color=ACCENT, fontsize=8, weight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch24_rate_ladder")


def fig_ring():
    """A SONET ring before and after a fibre cut (line switching loops traffic back)."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.4))
    ang = np.radians([90, 18, -54, -126, 162])
    names = ["A", "B", "C", "D", "E"]
    for k, a in enumerate(ax):
        th = np.linspace(0, 2 * np.pi, 300)
        a.plot(np.cos(th), np.sin(th), color=NAVY, lw=2.2, alpha=0.35)
        a.plot(0.82 * np.cos(th), 0.82 * np.sin(th), color=ACCENT, lw=1.2, alpha=0.35, ls="--")
        for n, t in zip(names, ang):
            a.add_patch(Circle((np.cos(t), np.sin(t)), 0.13, fc="white", ec=NAVY, lw=1.2, zorder=5))
            a.text(np.cos(t), np.sin(t), n, ha="center", va="center", fontsize=8, zorder=6, color=NAVY)
        if k == 0:
            tt = np.linspace(ang[0], ang[2], 100)
            a.plot(np.cos(tt), np.sin(tt), color=GREEN, lw=2.6)
            a.set_title("normal: A$\\rightarrow$C via B (working fibre)", fontsize=8)
        else:
            cut = np.radians(-18)
            a.plot([0.75 * np.cos(cut), 1.2 * np.cos(cut)], [0.75 * np.sin(cut), 1.2 * np.sin(cut)], color="k", lw=2)
            a.text(1.22 * np.cos(cut), 1.22 * np.sin(cut) - 0.12, "cut!", fontsize=8, color="k")
            tt = np.linspace(ang[0], ang[1], 60)
            a.plot(np.cos(tt), np.sin(tt), color=GREEN, lw=2.6)
            tt2 = np.linspace(ang[1], ang[1] + 2 * np.pi - (ang[1] - ang[2]) - 2 * np.pi + 2 * np.pi, 10)
            tb = np.linspace(ang[1], -2 * np.pi + ang[2] + 2 * np.pi + 2 * np.pi, 200)
            tb = np.linspace(ang[1], ang[1] + (2 * np.pi - (ang[1] - ang[2])), 200)
            a.plot(0.82 * np.cos(tb), 0.82 * np.sin(tb), color=ORANGE, lw=2.2)
            a.set_title("after the cut: B loops traffic back\nround the protection fibre (< 50 ms)", fontsize=8)
        a.set_xlim(-1.4, 1.5); a.set_ylim(-1.3, 1.3); _clean(a)
    fig.tight_layout(); save(fig, "ch24_ring")


def fig_dsl_bandplan():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    rows = [("POTS", [(0.3e3, 3.4e3, GRAY, "voice")]),
            ("ADSL", [(25.875e3, 138e3, ORANGE, "up"), (138e3, 1.104e6, NAVY, "down")]),
            ("ADSL2+", [(25.875e3, 138e3, ORANGE, "up"), (138e3, 2.208e6, NAVY, "down")]),
            ("VDSL2 17a\n(998ADE17)", [(25e3, 138e3, ORANGE, ""), (138e3, 3.75e6, NAVY, "down"), (3.75e6, 5.2e6, ORANGE, ""),
                                       (5.2e6, 8.5e6, NAVY, ""), (8.5e6, 12e6, ORANGE, ""), (12e6, 17.664e6, NAVY, "")]),
            ("G.fast", [(2.2e6, 106e6, PURPLE, "TDD: both ways, taking turns")])]
    for k, (name, bands) in enumerate(rows):
        y = len(rows) - 1 - k
        for lo, hi, c, t in bands:
            ax.add_patch(Rectangle((np.log10(lo), y - 0.32), np.log10(hi) - np.log10(lo), 0.64, fc=c, alpha=0.75, ec="white"))
            if t:
                ax.text((np.log10(lo) + np.log10(hi)) / 2, y, t, ha="center", va="center", fontsize=6.3, color="white")
        ax.text(2.35, y, name, ha="right", va="center", fontsize=7.5, color=NAVY)
    ax.set_xlim(2.4, 8.2); ax.set_ylim(-0.5, len(rows) - 0.3)
    ticks = [3, 4, 5, 6, 7, 8]
    ax.set_xticks(ticks); ax.set_xticklabels(["1 kHz", "10 kHz", "100 kHz", "1 MHz", "10 MHz", "100 MHz"], fontsize=7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.text(2.45, len(rows) - 0.55, "orange = upstream, navy = downstream", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch24_dsl_bandplan")


def fig_bit_loading():
    r, f, b, snr = adsl_rate(2000, return_all=True)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.bar(f / 1e6, b, width=4312.5 / 1e6, color=NAVY, alpha=0.8, label="bits loaded on each tone")
    ax2 = ax.twinx()
    ax2.plot(f / 1e6, snr, color=ACCENT, lw=1.0, label="SNR (right axis)")
    ax2.set_ylabel("SNR (dB)", fontsize=8, color=ACCENT); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.tick_params(labelsize=7)
    ax.set_xlabel("frequency (MHz)", fontsize=8); ax.set_ylabel("bits per tone", fontsize=8)
    ax.set_ylim(0, 16); ax2.set_ylim(0, 70); ax.tick_params(labelsize=7)
    ax.text(0.2, 15.2, "strong bridges: full lorries", fontsize=6.8, color=NAVY)
    ax.text(1.62, 11.0, "weaker bridges:\nlighter loads", fontsize=6.8, color=NAVY)
    ax.set_title(f"ADSL2+ on 2 km of 0.4 mm cable: {b.sum():.0f} bits per DMT symbol, about {r/1e6:.0f} Mb/s",
                 fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6, loc="upper right")
    fig.tight_layout(); save(fig, "ch24_bit_loading")


def fig_vector_matrix():
    """|H| of an 8-line binder on one tone, before and after precoding."""
    r = rng(3)
    Nl = 8; f = np.array([8e6]); L = 600
    h = np.abs(wl.loop_gain(f, L, 26))[0]
    fx = np.sqrt(wl.fext_coupling(f, L, h, 1))[0] / h       # FEXT relative to direct
    mag = fx * 10 ** (r.normal(0, 6, (Nl, Nl)) / 20)
    Cm = mag * np.exp(2j * np.pi * r.random((Nl, Nl)))
    np.fill_diagonal(Cm, 0)
    H = np.eye(Nl) + Cm
    Cest = Cm * (1 + 0.02 * (r.normal(size=Cm.shape) + 1j * r.normal(size=Cm.shape)))
    P = np.linalg.inv(np.eye(Nl) + Cest)
    E = H @ P
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.45))
    for a, M, t in [(ax[0], H, "without vectoring"), (ax[1], E, "with precoding (2 % estimation error)")]:
        im = a.imshow(20 * np.log10(np.abs(M) + 1e-9), cmap="viridis", vmin=-80, vmax=0)
        a.set_title(t, fontsize=8)
        a.set_xticks(range(Nl)); a.set_yticks(range(Nl)); a.tick_params(labelsize=6)
        a.set_xlabel("transmitting line", fontsize=7); a.set_ylabel("receiving line", fontsize=7)
        a.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("coupling (dB re direct)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "ch24_vector_matrix")
    print("FEXT mean (dB):", 20 * np.log10(np.abs(Cm[Cm != 0])).mean(), "after:", 20 * np.log10(np.abs(E - np.diag(np.diag(E)))[~np.eye(Nl, dtype=bool)]).mean())


def fig_eth_waveforms():
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    fig, ax = plt.subplots(3, 1, figsize=(W1, 2.6), sharex=True)
    n = len(bits)
    # Manchester (IEEE 802.3: 1 = low-to-high)
    t = []; v = []
    for k, b in enumerate(bits):
        t += [k, k + 0.5, k + 0.5, k + 1]; v += ([-1, -1, 1, 1] if b else [1, 1, -1, -1])
    ax[0].plot(t, v, color=NAVY); ax[0].set_ylabel("Manch.", fontsize=7)
    # MLT-3
    lv = [0, 1, 0, -1]; s = 0; vv = []
    for b in bits:
        if b: s = (s + 1) % 4
        vv.append(lv[s])
    ax[1].step(np.arange(n + 1), vv + [vv[-1]], where="post", color=GREEN); ax[1].set_ylabel("MLT-3", fontsize=7)
    # PAM-5 (two bits per symbol)
    mp = {(0, 0): -2, (0, 1): -1, (1, 1): 1, (1, 0): 2}
    sy = [mp[(bits[2 * k], bits[2 * k + 1])] for k in range(n // 2)]
    ax[2].step(np.arange(0, n + 1, 2), sy + [sy[-1]], where="post", color=ACCENT); ax[2].set_ylabel("PAM-5", fontsize=7)
    ax[2].set_yticks([-2, -1, 0, 1, 2])
    for k, b in enumerate(bits):
        ax[0].text(k + 0.5, 1.45, str(b), ha="center", fontsize=7, color=GRAY)
    ax[0].set_ylim(-1.5, 1.9)
    ax[0].set_title("the same twelve bits: 10BASE-T, 100BASE-TX and (one pair of) 1000BASE-T", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=6.5); a.set_xticks(range(0, n + 1))
    ax[2].set_xlabel("bit periods", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_eth_waveforms")


def fig_noise_funnel():
    r = rng(5)
    homes = np.arange(1, 801)
    ing = 10 ** (r.normal(-60, 6, 800) / 10)          # per-home ingress power (dBmV-like units)
    tot = 10 * np.log10(np.cumsum(ing))
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.semilogx(homes, tot, color=ACCENT, label="ingress summed at the node")
    ax.semilogx(homes, -60 + 10 * np.log10(homes) + 6 ** 2 * np.log(10) / 20, color=GRAY, ls=":", label="+10 log$_{10}N$")
    ax.set_xlabel("homes feeding one upstream", fontsize=8); ax.set_ylabel("noise at the node (dB, rel.)", fontsize=8)
    ax.legend(fontsize=6.4); ax.tick_params(labelsize=7)
    ax.set_title("noise funnelling", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_noise_funnel")


def fig_tir():
    """Total internal reflection: a guided ray skips along; a steep ray escapes."""
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.add_patch(Rectangle((0, -1.6), 10, 3.2, fc=SKY, alpha=0.10, ec="none"))
    ax.add_patch(Rectangle((0, -0.8), 10, 1.6, fc=SKY, alpha=0.25, ec="none"))
    ax.text(9.9, 1.25, "cladding $n_2$", ha="right", fontsize=7, color=NAVY)
    ax.text(9.9, 0.55, "core $n_1 > n_2$", ha="right", fontsize=7, color=NAVY)
    # guided ray: shallow angle
    xs = [0.2]; ys = [0.0]; x = 0.2; y = 0.0; s = 0.28; d = 1
    while x < 9.8:
        yt = 0.8 * d; dx = (yt - y) / s * d if False else abs(yt - y) / s
        x2 = min(x + dx, 9.8); y2 = y + d * s * (x2 - x)
        xs.append(x2); ys.append(y2); x, y = x2, y2; d = -d
    ax.plot(xs, ys, color=GREEN, lw=1.5)
    ax.text(3.0, -1.35, "shallow ray: totally reflected every time (guided)", fontsize=7, color=GREEN)
    # escaping ray
    ax.plot([0.2, 1.4], [0.0, 0.8], color=ACCENT, lw=1.3)
    ax.plot([1.4, 2.1], [0.8, 1.6], color=ACCENT, lw=1.3, ls="--")
    ax.text(2.2, 1.35, "steep ray: refracts out and is lost", fontsize=7, color=ACCENT)
    ax.set_xlim(0, 10); ax.set_ylim(-1.6, 1.6); ax.axis("off")
    fig.tight_layout(); save(fig, "ch24_tir")


def fig_modes():
    fig, ax = plt.subplots(3, 2, figsize=(W1, 2.9), gridspec_kw=dict(width_ratios=[3.2, 1]))
    x = np.linspace(0, 10, 1200)
    tri = lambda x, p: 2 * np.abs(2 * (x / p - np.floor(x / p + 0.5))) - 1
    rows = [("step-index multimode (core 50--62.5 $\\mu$m)", 1.0, [(None, 0), (3.0, 1), (1.4, 1)]),
            ("graded-index multimode", 1.0, [(None, 0), (2.4, 2), (2.4, 2.2)]),
            ("single-mode (core $\\approx$ 9 $\\mu$m)", 0.25, [(None, 0)])]
    cols = [NAVY, GREEN, ACCENT]
    widths = [0.9, 0.35, 0.12]
    for k, (name, h, rays) in enumerate(rows):
        a = ax[k, 0]
        a.add_patch(Rectangle((0, -h), 10, 2 * h, fc=SKY, alpha=0.18 if k != 1 else 0.0, ec="none"))
        if k == 1:
            for j in range(20):
                hh = h * (1 - j / 20)
                a.add_patch(Rectangle((0, -hh), 10, 2 * hh, fc=SKY, alpha=0.03, ec="none"))
        for (p, typ), c in zip(rays, cols):
            if p is None:
                a.plot(x, 0 * x, color=c, lw=1.1)
            elif typ == 1:
                a.plot(x, 0.95 * h * tri(x, p), color=c, lw=1.0)
            else:
                amp = 0.6 if typ == 2 else 0.9
                a.plot(x, amp * h * np.sin(2 * np.pi * x / p), color=c, lw=1.0)
        a.set_xlim(0, 10); a.set_ylim(-1.1, 1.1); a.axis("off")
        a.set_title(name, fontsize=7.5, loc="left")
        b = ax[k, 1]
        t = np.linspace(-2, 2, 300)
        b.plot(t, np.exp(-(t / 0.15) ** 2), color=GRAY, lw=0.8, ls=":")
        b.plot(t, np.exp(-(t / (0.15 + widths[k])) ** 2) * 0.15 / (0.15 + widths[k]) ** 0.5 / 0.15 ** 0.5, color=NAVY)
        b.set_xlim(-2, 2); b.set_ylim(0, 1.1); b.axis("off")
        if k == 0:
            b.set_title("pulse in / out", fontsize=7)
    fig.tight_layout(); save(fig, "ch24_modes")


def fig_runners():
    """Chromatic dispersion as runners: a short pulse spreads, with its colours sorted in time."""
    fs = 2e12; n = 2 ** 13
    t = (np.arange(n) - n / 2) / fs
    T0 = 4e-12
    e0 = np.exp(-(t / T0) ** 2 / 2).astype(complex)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    import matplotlib.cm as cm
    for k, Lkm in enumerate([0, 0.5, 1, 2]):
        e = wl.apply_cd(e0, fs, Lkm * 1e3)
        I = np.abs(e) ** 2
        ph = np.unwrap(np.angle(e))
        finst = -np.gradient(ph, t) / (2 * np.pi)       # optical frequency offset (sign: + = bluer)
        m = I > 0.01 * I.max()
        off = 1.25 * (3 - k)
        tt = t[m] * 1e12; II = I[m] / I.max() * (1 if k == 0 else (I.max() / np.abs(e0).max() ** 2))
        cc = np.clip(0.5 - finst[m] / 3e11, 0, 1)
        for j in range(len(tt) - 1):
            ax.fill_between(tt[j:j + 2], off, off + II[j:j + 2], color=cm.jet(cc[j]), lw=0)
        ax.text(-58, off + 0.3, f"{Lkm:g} km", fontsize=7.5, color=NAVY)
    ax.set_xlim(-60, 60); ax.set_yticks([])
    ax.set_xlabel("time (ps)", fontsize=8); ax.tick_params(labelsize=7)
    ax.text(26, 4.4, "colour = instantaneous frequency\n(blue = shorter wavelength)", fontsize=6.6, color=GRAY)
    ax.set_title("a 4 ps pulse at 1550 nm in standard fibre: the blue runners pull ahead", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_runners")


def fig_pmd():
    mean = 3.0
    x = np.linspace(0, 12, 400)
    a = 2 * mean / np.sqrt(np.pi) / 2 * np.sqrt(np.pi) / 2 * 0 + mean * np.sqrt(np.pi / 8)
    p = np.sqrt(2 / np.pi) * x ** 2 / a ** 3 * np.exp(-x ** 2 / (2 * a ** 2))
    from scipy.stats import maxwell
    tail = maxwell.sf(3 * mean, scale=a)
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.plot(x, p, color=NAVY)
    m = x >= 3 * mean
    ax.fill_between(x[m], 0, p[m], color=ACCENT, alpha=0.5)
    ax.axvline(mean, color=GRAY, ls=":", lw=0.8); ax.text(mean + 0.15, p.max() * 0.95, "mean 3 ps", fontsize=6.5, color=GRAY)
    ax.annotate(f"P(DGD > 9 ps) $\\approx$ {tail:.0e}", (9.6, 0.004), (6.3, 0.12), fontsize=6.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.6))
    ax.set_xlabel("differential group delay (ps)", fontsize=8); ax.set_ylabel("probability density", fontsize=8)
    ax.set_title("PMD: Maxwellian DGD, 1000 km of modern fibre", fontsize=7.8); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_pmd")
    print("PMD tail", tail)


def fig_edfa_levels():
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    lv = [(0, "$^4I_{15/2}$ ground"), (0.80, "$^4I_{13/2}$ (lives $\\approx$10 ms)"), (1.27, "$^4I_{11/2}$")]
    for y, t in lv:
        ax.plot([0.5, 3.2], [y, y], color=NAVY, lw=2)
        ax.text(3.3, y, t, fontsize=7, va="center", color=NAVY)
    _arrow(ax, (0.8, 0.0), (0.8, 1.27), col=ORANGE, lw=1.6, ms=10)
    ax.text(0.55, 0.62, "pump\n980 nm", fontsize=6.6, color=ORANGE, ha="right")
    _arrow(ax, (1.3, 0.0), (1.3, 0.80), col=GOLD if False else ORANGE, lw=1.0, ms=8, ls="--")
    ax.text(1.38, 0.32, "or\n1480", fontsize=6.3, color=ORANGE)
    _arrow(ax, (1.0, 1.27), (1.9, 0.82), col=GRAY, lw=0.9, ms=7)
    ax.text(1.55, 1.13, "fast decay", fontsize=6.3, color=GRAY)
    _arrow(ax, (2.5, 0.80), (2.5, 0.0), col=ACCENT, lw=1.8, ms=10)
    ax.text(2.6, 0.4, "signal\n1530--1565 nm\n(stimulated)", fontsize=6.6, color=ACCENT)
    ax.set_xlim(0, 5.3); ax.set_ylim(-0.15, 1.45); ax.axis("off")
    ax.set_title("erbium ions as a light amplifier", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_edfa_levels")


def fig_power_staircase():
    nsp = 5; Ls = 80; a = 0.22; P0 = 1.0; NF = 5.0
    G = Ls * a
    z = []; P = []; ase = []; zs = []
    ase_lin = 0.0
    for k in range(nsp):
        zz = np.linspace(0, Ls, 50)
        z += list(k * Ls + zz); P += list(P0 - a * zz)
        ase_lin += 10 ** ((-58 + NF + G) / 10)
        zs.append((k + 1) * Ls)
        ase.append(10 * np.log10(ase_lin))
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot(z, P, color=NAVY, label="signal power per channel")
    ax.step([0] + zs, [ase[0] - 40] + ase, where="post", color=ACCENT, label="accumulated ASE in 0.1 nm")
    for k, zz in enumerate(zs):
        ax.plot([zz, zz], [P0 - G, P0], color=GREEN, lw=1.5)
        ax.plot(zz, P0, "^", color=GREEN, ms=6)
        ax.annotate("", (zz + 3, P0 - 1), (zz + 3, ase[k] + 1), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.6))
        ax.text(zz + 5, (P0 + ase[k]) / 2, f"OSNR\n{P0 - ase[k]:.0f} dB", fontsize=6.2, color=GRAY, va="center")
    ax.text(5, P0 - G + 1, "each 80 km span loses 17.6 dB;\neach EDFA (green) gives it back\nand adds a little noise",
            fontsize=6.6, color=NAVY)
    ax.set_xlabel("distance (km)", fontsize=8); ax.set_ylabel("power (dBm)", fontsize=8)
    ax.set_ylim(-40, 5); ax.set_xlim(0, 460); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout(); save(fig, "ch24_power_staircase")


def fig_wdm_comb():
    import matplotlib.cm as cm
    f0 = 191.35; n = 96; df = 0.05
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ff = np.linspace(191.2, 196.2, 6000)
    tot = np.full_like(ff, -40.0)
    for k in range(n):
        fc = f0 + k * df
        sp = np.where(np.abs(ff - fc) < 0.016, 0.0, -45.0)
        tilt = 0.8 * np.sin((fc - 191.3) / 4.8 * np.pi)
        mk = np.abs(ff - fc) < 0.02
        ax.fill_between(ff[mk], -36, (sp + tilt)[mk], color=cm.turbo(k / n), lw=0)
    ax.axhline(-30, color=GRAY, lw=0.6, ls=":")
    ax.text(191.3, -29, "ASE noise floor", fontsize=6.5, color=GRAY)
    ax.set_xlim(191.2, 196.3); ax.set_ylim(-36, 4)
    ax.set_xlabel("optical frequency (THz)", fontsize=8); ax.set_ylabel("power (dB, rel.)", fontsize=8)
    ax.tick_params(labelsize=7)
    sec = ax.secondary_xaxis("top", functions=(lambda f: 299792.458 / f, lambda l: 299792.458 / l))
    sec.set_xlabel("wavelength (nm)", fontsize=7); sec.tick_params(labelsize=6.5)
    ax.set_title("96 channels on the 50 GHz grid fill the C band", fontsize=8.5, pad=22)
    fig.tight_layout(); save(fig, "ch24_wdm_comb")


def fig_mzm():
    V = np.linspace(-2.2, 2.2, 400)    # in units of V_pi
    E = np.cos(np.pi * (V + 1) / 2)
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot(V, E ** 2, color=NAVY, label="output power")
    ax.plot(V, E, color=ACCENT, ls="--", label="output field")
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.plot(-0.5, 0.5, "o", color=GREEN); ax.text(-0.45, 0.58, "quadrature bias:\nintensity modulator", fontsize=6.2, color=GREEN)
    ax.plot(0, 0, "o", color=PURPLE); ax.text(0.1, -0.35, "null bias: field\nflips sign (BPSK, IQ)", fontsize=6.2, color=PURPLE)
    ax.set_xlabel("drive voltage ($V_\\pi$ units)", fontsize=8); ax.set_ylabel("transfer", fontsize=8)
    ax.legend(fontsize=6.2, loc="upper right"); ax.tick_params(labelsize=7); ax.set_ylim(-1.1, 1.35)
    ax.set_title("the Mach--Zehnder modulator", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_mzm")


def fig_iq_vs_intensity():
    r = rng(2)
    pts = np.array([a + 1j * b for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
    s = pts[r.integers(0, 16, 3000)] + 0.18 * (r.normal(size=3000) + 1j * r.normal(size=3000))
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
    a = ax[0]
    a.plot(s.real, s.imag, ".", ms=1.2, color=NAVY, alpha=0.5)
    a.set_aspect("equal"); a.set_title("coherent receiver: field (16 points)", fontsize=8)
    a.set_xlabel("I", fontsize=8); a.set_ylabel("Q", fontsize=8); a.tick_params(labelsize=7)
    b = ax[1]
    b.hist(np.abs(s) ** 2, bins=120, color=ACCENT, alpha=0.8)
    for v in (2, 10, 18):
        b.text(v, b.get_ylim()[1] * 0.9 if False else 0, "", fontsize=6)
    b.set_title("photodiode: power only (3 levels)", fontsize=8)
    b.set_xlabel("$|E|^2$", fontsize=8); b.set_yticks([]); b.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_iq_vs_intensity")


def fig_pcs():
    lv = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    lam = 0.035
    p1 = np.exp(-lam * lv ** 2); p1 /= p1.sum()
    P = np.outer(p1, p1)
    H = -(P * np.log2(P)).sum()
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = ax[0]
    X, Y = np.meshgrid(lv, lv)
    a.scatter(X, Y, s=P * 2400, color=NAVY, alpha=0.8)
    a.set_aspect("equal"); a.set_xticks(lv); a.set_yticks(lv); a.tick_params(labelsize=6)
    a.set_title(f"PCS-64QAM: dot area = probability\n(entropy {H:.2f} of 6 bits/symbol)", fontsize=7.5)
    b = ax[1]
    b.bar(lv, p1, color=NAVY, alpha=0.8, label="shaped (Maxwell--Boltzmann)")
    b.plot(lv, np.full(8, 1 / 8), "--", color=ACCENT, label="uniform")
    b.set_xticks(lv); b.tick_params(labelsize=7); b.set_ylabel("probability of each amplitude", fontsize=7.5)
    b.legend(fontsize=6.3, loc="upper right"); b.set_ylim(0, 0.3)
    fig.tight_layout(); save(fig, "ch24_pcs")


def fig_fibre_timeline():
    ev = [(1966, "Kao & Hockham:\nglass can do < 20 dB/km", 1), (1970, "Corning: first\nlow-loss fibre", -1),
          (1977, "first field trials\n(Chicago, 45 Mb/s)", 1), (1979, "NTT: 0.2 dB/km\nat 1550 nm", -1),
          (1987, "EDFA demonstrated\n(Southampton, Bell Labs)", 1), (1988, "TAT-8: first\ntransatlantic fibre", -1),
          (1996, "TAT-12/13: first\namplified transatlantic", 1), (2001, "telecom crash:\ndark fibre glut", -1),
          (2008, "Nortel: 40G\ndigital coherent", 1), (2010, "100G DP-QPSK\nbecomes standard", -1),
          (2018, "MAREA in service\n(~160--200 Tb/s)", 1), (2020, "OIF 400ZR:\ncoherent pluggable", -1),
          (2021, "Dunant: 250 Tb/s,\n12 fibre pairs", 1)]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.axhline(0, color=NAVY, lw=2)
    for k, (y, t, s) in enumerate(ev):
        h = s * (0.9 + 0.75 * ((k // 2) % 2))
        ax.plot([y, y], [0, h * 0.85], color=GRAY, lw=0.6)
        ax.plot(y, 0, "o", color=ACCENT if y < 1987 else (GREEN if y < 2008 else PURPLE), ms=5)
        ax.text(y, h, t, ha="center", va="bottom" if s > 0 else "top", fontsize=5.9, color=NAVY)
        ax.text(y, -0.12 if s > 0 else 0.12, str(y), ha="center", va="top" if s > 0 else "bottom", fontsize=5.8, color=GRAY)
    ax.set_xlim(1961, 2026); ax.set_ylim(-2.4, 2.4); ax.axis("off")
    save(fig, "ch24_fibre_timeline")


def fig_sdm_power():
    n = np.arange(1, 33)
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    for snr_db, col in [(30, NAVY), (20, GREEN), (10, ORANGE)]:
        s = 10 ** (snr_db / 10)
        C = n * np.log2(1 + s / n)
        ax.plot(n, C / np.log2(1 + s), color=col, label=f"all power in one pair: {snr_db} dB")
    ax.set_xlabel("fibre pairs sharing the same power", fontsize=8); ax.set_ylabel("capacity (rel. to one pair)", fontsize=8)
    ax.legend(fontsize=6.2, loc="upper left"); ax.tick_params(labelsize=7)
    ax.set_title("spreading a fixed power over more fibres", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_sdm_power")


def fig_pon_budget():
    steps = [("OLT launch", 1.5), ("20 km fibre", -6.0), ("1:32 splitter", -17.0), ("4 connectors", -1.2), ("10 splices", -1.0)]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    lvl = 0
    for k, (n, d) in enumerate(steps):
        if k == 0:
            ax.bar(k, d, color=GREEN, width=0.6); lvl = d
        else:
            ax.bar(k, d, bottom=lvl, color=ACCENT if "split" in n else ORANGE, width=0.6)
            lvl += d
        ax.text(k, lvl + (0.6 if k == 0 else -1.2), f"{lvl:+.1f} dBm", ha="center", fontsize=6.6, color=NAVY)
    ax.bar(len(steps), lvl, color=NAVY, width=0.6)
    ax.text(len(steps), lvl - 1.2, f"{lvl:.1f} dBm\nreceived", ha="center", va="top", fontsize=6.6, color=NAVY)
    ax.axhline(-27, color=PURPLE, ls="--", lw=0.9)
    ax.text(-0.4, -28.3, "ONU sensitivity $-$27 dBm", fontsize=6.6, color=PURPLE)
    ax.annotate("", (len(steps) + 0.42, lvl), (len(steps) + 0.42, -27), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(len(steps) + 0.5, (lvl - 27) / 2, f"margin\n{lvl+27:.1f} dB", fontsize=6.6, color=GREEN, va="center")
    ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels([s[0] for s in steps] + ["ONU"], fontsize=6.8)
    ax.set_ylabel("power (dBm)", fontsize=8); ax.set_ylim(-31, 4); ax.set_xlim(-0.6, len(steps) + 1.2)
    ax.axhline(0, color=GRAY, lw=0.5); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_pon_budget")


def fig_pon_ranging():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.3), sharex=True)
    onus = [("ONU 1 (2 km)", 2, NAVY), ("ONU 2 (11 km)", 11, GREEN), ("ONU 3 (19 km)", 19, ORANGE)]
    slots = [0, 12, 24]
    for k, a in enumerate(ax):
        for (n, d, c), s in zip(onus, slots):
            delay = 2 * d * 5 / 10       # round trip in microseconds, scaled (5 us/km one way -> x 0.1 units)
            start = s + (delay if k == 0 else 20)
            a.add_patch(Rectangle((start, 0.2), 10, 0.6, fc=c, alpha=0.8))
            a.text(start + 5, 0.5, n, ha="center", va="center", fontsize=6.3, color="white")
        a.set_ylim(0, 1); a.set_yticks([]); a.set_xlim(0, 60)
    ax[0].set_title("without ranging: bursts arrive at the OLT overlapping", fontsize=8)
    ax[1].set_title("with ranging: each ONU adds an equalisation delay, bursts interleave", fontsize=8)
    ax[1].set_xlabel("arrival time at the OLT (arbitrary units)", fontsize=8); ax[1].tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_pon_ranging")


def fig_pam4_bands():
    r = rng(4)
    rs = 53.125e9; sps = 16; fs = rs * sps; nsym = 3000
    lev = np.array([0, 1, 2, 3]) / 3.0
    sym = lev[r.integers(0, 4, nsym)]
    p = 0.15 + 0.85 * sym                                   # power levels with finite extinction
    taps = cl.rc_taps(0.3, sps, span=12) if hasattr(cl, "rc_taps") else None
    x = np.repeat(p, sps)
    h = np.ones(sps // 2) / (sps // 2)
    x = np.convolve(x, np.convolve(h, h), mode="same") / np.sum(np.convolve(h, h)) * (sps // 2)
    x = np.clip(x, 0.01, None)
    E = np.sqrt(x).astype(complex)
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.0), sharey=True)
    cases = [("back to back", 0, 17), ("2 km, O band ($D\\approx$0)", 2e3, 0.3), ("2 km, C band ($D$=17)", 2e3, 17)]
    for a, (t, L, D) in zip(ax, cases):
        e = wl.apply_cd(E, fs, L, D) if L > 0 else E
        I = np.abs(e) ** 2
        seg = I[sps * 100: sps * 100 + sps * 2 * 400].reshape(-1, 2 * sps)
        tt = np.arange(2 * sps) / sps
        for row in seg:
            a.plot(tt, row, color=NAVY, lw=0.3, alpha=0.15)
        a.set_title(t, fontsize=7.5); a.set_xticks([0, 1, 2]); a.tick_params(labelsize=6.5)
        a.set_xlabel("symbol periods", fontsize=7)
    ax[0].set_ylabel("detected power", fontsize=7.5)
    fig.suptitle("53 GBd PAM-4 with direct detection", fontsize=8.5, y=1.02)
    fig.tight_layout(); save(fig, "ch24_pam4_bands")


def fig_fso_fog():
    V = np.logspace(-1.3, 1.7, 300)    # km
    lam = 1550.0
    q = np.where(V > 50, 1.6, np.where(V > 6, 1.3, np.where(V > 1, 0.16 * V + 0.34, np.where(V > 0.5, V - 0.5, 0.0))))
    att = 10 * np.log10(np.e) * 3.91 / V * (lam / 550) ** (-q)
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.loglog(V, att, color=NAVY)
    for v, t in [(0.05, "dense fog"), (0.5, "moderate fog"), (2, "haze"), (20, "clear")]:
        ax.axvline(v, color=GRAY, lw=0.5, ls=":")
        ax.text(v * 1.08, 0.12, t, rotation=90, fontsize=6.3, color=GRAY, va="bottom")
    ax.set_xlabel("visibility (km)", fontsize=8); ax.set_ylabel("attenuation at 1550 nm (dB/km)", fontsize=8)
    ax.set_title("free-space optics vs the weather (Kim model)", fontsize=7.8); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_fso_fog")


def fig_poe():
    std = ["802.3af\n(2003)", "802.3at\n(2009)", "802.3bt\nType 3", "802.3bt\nType 4"]
    pse = [15.4, 30, 60, 90]; pd = [12.95, 25.5, 51, 71.3]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    x = np.arange(4)
    ax.bar(x - 0.18, pse, width=0.36, color=NAVY, label="at the switch port")
    ax.bar(x + 0.18, pd, width=0.36, color=GREEN, label="at the device")
    for xx, a, b in zip(x, pse, pd):
        ax.text(xx - 0.18, a + 1.5, f"{a:g}", ha="center", fontsize=6.2, color=NAVY)
        ax.text(xx + 0.18, b + 1.5, f"{b:g}", ha="center", fontsize=6.2, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(std, fontsize=6.5); ax.set_ylabel("power (W)", fontsize=8)
    ax.legend(fontsize=6.4, loc="upper left"); ax.tick_params(labelsize=7); ax.set_ylim(0, 105)
    ax.set_title("Power over Ethernet classes", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch24_poe_classes")


def fig_signalling():
    """In-band signalling: DTMF keypad frequencies and the 2600 Hz supervisory tone in the voice band."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a = ax[0]
    rows = [697, 770, 852, 941]; cols = [1209, 1336, 1477, 1633]
    keys = [["1", "2", "3", "A"], ["4", "5", "6", "B"], ["7", "8", "9", "C"], ["*", "0", "#", "D"]]
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            a.add_patch(FancyBboxPatch((j + 0.12, 3 - i + 0.12), 0.76, 0.76, boxstyle="round,pad=0,rounding_size=0.12",
                                       fc=NAVY if j < 3 else GRAY, ec="none", alpha=0.85))
            a.text(j + 0.5, 3 - i + 0.5, keys[i][j], ha="center", va="center", color="white", fontsize=9, weight="bold")
        a.text(-0.1, 3 - i + 0.5, f"{r} Hz", ha="right", va="center", fontsize=6.6, color=ACCENT)
    for j, c in enumerate(cols):
        a.text(j + 0.5, 4.15, f"{c}", ha="center", fontsize=6.6, color=GREEN)
    a.text(2.0, 4.55, "high tone (Hz)", ha="center", fontsize=6.6, color=GREEN)
    a.set_xlim(-1.3, 4.1); a.set_ylim(-0.1, 4.8); _clean(a)
    a.set_title("DTMF: each key is two tones", fontsize=8)
    b = ax[1]
    f = np.linspace(0, 4000, 800)
    sp = np.where((f > 300) & (f < 3400), -20 - 12 * np.log10(np.maximum(f, 300) / 500) ** 2 * 3, -60)
    b.fill_between(f, -60, sp, color=SKY, alpha=0.35, lw=0)
    b.plot(f, sp, color=SKY, lw=0.8)
    b.annotate("", (2600, -5), (2600, -60), arrowprops=dict(arrowstyle="-", color=ACCENT, lw=2))
    b.text(2650, -9, "2600 Hz:\n'trunk idle'", fontsize=6.6, color=ACCENT)
    b.text(700, -32, "speech", fontsize=7, color=NAVY)
    b.axvspan(300, 3400, color=GRAY, alpha=0.07)
    b.set_xlabel("frequency (Hz)", fontsize=8); b.set_ylabel("level (dB, illustrative)", fontsize=8)
    b.set_ylim(-60, 0); b.set_xlim(0, 4000); b.tick_params(labelsize=7)
    b.set_title("in-band signalling shares the voice channel", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_signalling")


def fig_2b1q():
    r = rng(8)
    bits = r.integers(0, 2, 24)
    mp = {(1, 0): 3, (1, 1): 1, (0, 1): -1, (0, 0): -3}
    sy = [mp[(bits[2 * k], bits[2 * k + 1])] for k in range(12)]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.1), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    a.step(np.arange(13) * 12.5, sy + [sy[-1]], where="post", color=NAVY)
    for k in range(12):
        a.text(k * 12.5 + 6.25, 3.6, f"{bits[2*k]}{bits[2*k+1]}", ha="center", fontsize=6, color=GRAY)
    a.set_yticks([-3, -1, 1, 3]); a.set_ylim(-3.8, 4.2)
    a.set_xlabel("time ($\\mu$s)", fontsize=8); a.set_title("2B1Q: two bits per quaternary symbol, 80 kBd", fontsize=8)
    a.tick_params(labelsize=7)
    b = ax[1]
    f = np.linspace(1, 400, 400)
    psd = np.sinc(f / 80) ** 2
    b.plot(f, 10 * np.log10(psd + 1e-6), color=NAVY)
    b.axvspan(25.875, 138, color=ORANGE, alpha=0.15); b.text(145, -37, "ADSL upstream band", fontsize=6.3, color=ORANGE)
    b.set_xlabel("frequency (kHz)", fontsize=8); b.set_ylabel("PSD (dB)", fontsize=8)
    b.set_ylim(-40, 3); b.tick_params(labelsize=7)
    b.set_title("ISDN's spectrum: first null at 80 kHz", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_2b1q")


def fig_gfast_tdd():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    nl = 4; T = 36            # symbols per TDD frame (illustrative split)
    for l in range(nl):
        y = nl - 1 - l
        for fr in range(3):
            x0 = fr * T
            ax.add_patch(Rectangle((x0, y + 0.15), 26, 0.7, fc=NAVY, alpha=0.8, ec="white"))
            ax.add_patch(Rectangle((x0 + 27, y + 0.15), 8, 0.7, fc=ORANGE, alpha=0.85, ec="white"))
        ax.text(-1.5, y + 0.5, f"line {l+1}", ha="right", va="center", fontsize=7, color=NAVY)
    ax.text(13, nl - 0.5, "downstream", ha="center", va="center", fontsize=7, color="white")
    ax.text(31, nl - 0.5, "up", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(-9, 3 * T + 1); ax.set_ylim(-0.2, nl + 0.7); ax.axis("off")
    ax.text(3 * T / 2, -0.15, "time: TDD frames; every line switches direction at the same instant, so no receiver listens while a neighbour sends",
            ha="center", va="top", fontsize=6.6, color=GRAY)
    save(fig, "ch24_gfast_tdd")


def fig_fdx():
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    def blk(x0, x1, y, c, t):
        ax.add_patch(Rectangle((x0, y), x1 - x0, 0.7, fc=c, alpha=0.8, ec="white"))
        ax.text((x0 + x1) / 2, y + 0.35, t, ha="center", va="center", fontsize=6.6, color="white")
    blk(5, 85, 1.2, ORANGE, "up"); blk(108, 1218, 1.2, NAVY, "downstream")
    blk(5, 85, 0.2, ORANGE, "up"); blk(108, 684, 0.2, PURPLE, "FDX: both directions at once")
    blk(684, 1218, 0.2, NAVY, "downstream")
    ax.text(-20, 1.55, "DOCSIS 3.1\n(low split)", ha="right", va="center", fontsize=7, color=NAVY)
    ax.text(-20, 0.55, "DOCSIS 4.0\nfull duplex", ha="right", va="center", fontsize=7, color=NAVY)
    ax.set_xlim(-250, 1250); ax.set_ylim(-0.7, 2.1)
    ax.set_yticks([]); ax.set_xticks([0, 108, 684, 1218]); ax.tick_params(labelsize=7)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("frequency (MHz)", fontsize=8); ax.grid(False)
    ax.text(396, -0.12, "the node cancels its own downstream echo here", ha="center", fontsize=6.4, color=PURPLE)
    save(fig, "ch24_fdx")


def fig_fwm():
    """Four-wave mixing products f_i + f_j - f_k of four equally spaced channels."""
    ch = np.array([0, 1, 2, 3]) * 100.0          # GHz offsets on a 100 GHz grid
    prods = {}
    for i in ch:
        for j in ch:
            for k in ch:
                if k == i or k == j:
                    continue
                f = i + j - k
                prods[f] = prods.get(f, 0) + (1 if i == j else 2)
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.vlines(ch, 0, 30, color=NAVY, lw=7, alpha=0.25)
    for f, n in sorted(prods.items()):
        on = f in ch
        ax.vlines(f, 0, n, color=ACCENT if on else ORANGE, lw=2.2)
    ax.text(150, 31, "four WDM channels (shaded)", ha="center", fontsize=7, color=NAVY)
    ax.text(-260, 10, "mixing products land\nbeside the channels (orange)\nand on top of them (red)", fontsize=6.6, color=ORANGE)
    ax.set_xlabel("frequency offset (GHz), 100 GHz grid", fontsize=8); ax.set_ylabel("number of products", fontsize=8)
    ax.set_xlim(-320, 620); ax.set_ylim(0, 34); ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch24_fwm")


def fig_poisson():
    """Photon counting: Poisson statistics of a 'one' and the quantum-limit error."""
    from scipy.stats import poisson
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    n = np.arange(0, 45)
    for N, col in [(5, ORANGE), (20, NAVY)]:
        ax.bar(n + (0.2 if N == 20 else -0.2), poisson.pmf(n, N), width=0.4, color=col, alpha=0.85,
               label=f"mean {N} photons: P(0) = {np.exp(-N):.0e}")
    ax.set_xlabel("photons detected in a 'one' bit", fontsize=8); ax.set_ylabel("probability", fontsize=8)
    ax.legend(fontsize=6.2, loc="upper right"); ax.tick_params(labelsize=7)
    ax.set_title("the quantum limit: errors only when\na 'one' brings no photon at all", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_poisson")


def fig_loop_states():
    """Voltage on a telephone line: idle battery, ringing, off-hook, pulse dialling (illustrative)."""
    t = np.linspace(0, 6, 6000)
    v = np.full_like(t, -48.0)
    ring = (t > 0.5) & (t < 2.5)
    v[ring] = -48 + 90 * np.sqrt(2) * np.sin(2 * np.pi * 20 * t[ring])
    off = t >= 3.0
    v[off] = -8.0
    pulses = (t > 4.0) & (t < 4.5) & (((t - 4.0) * 10) % 1 < 0.6)
    v[pulses] = -48.0
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot(t, v, color=NAVY, lw=0.7)
    ax.text(0.02, -95, "idle:\n$-48$ V", fontsize=6.4, color=NAVY)
    ax.text(0.9, 90, "ringing: 90 V rms, 20 Hz", fontsize=6.4, color=ACCENT)
    ax.text(3.05, 10, "off-hook:\nloop current\nflows", fontsize=6.4, color=GREEN)
    ax.text(4.0, -62, "dialling \"5\"", fontsize=6.4, color=ORANGE)
    ax.set_xlabel("time (s)", fontsize=8); ax.set_ylabel("line voltage (V)", fontsize=8)
    ax.set_ylim(-190, 120); ax.tick_params(labelsize=7)
    ax.set_title("a telephone line's working day (illustrative)", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_loop_states")


def fig_v21_split():
    f = np.linspace(0, 4000, 1000)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    def bump(fc, w=180):
        return np.exp(-((f - fc) / w) ** 2 * 3)
    ax.fill_between(f, 0, bump(1080), color=NAVY, alpha=0.7, label="caller sends (1080 Hz)")
    ax.fill_between(f, 0, bump(1750), color=ORANGE, alpha=0.7, label="answerer sends (1750 Hz)")
    ax.axvspan(300, 3400, color=GRAY, alpha=0.08)
    ax.text(2500, 0.6, "rest of the\nvoice band", fontsize=6.5, color=GRAY)
    ax.set_xlabel("frequency (Hz)", fontsize=8); ax.set_yticks([]); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, loc="upper right"); ax.set_ylim(0, 1.35)
    ax.set_title("V.21: one band each way (FSK $\\pm$100 Hz)", fontsize=8)
    fig.tight_layout(); save(fig, "ch24_v21_split")


def fig_otn_frame():
    fig, ax = plt.subplots(figsize=(W1, 1.5))
    parts = [(0, 16, ACCENT, "OH\n16"), (16, 3824, NAVY, "payload (OPU): 3808 columns"), (3824, 4080, GREEN, "FEC\n256")]
    for r in range(4):
        for x0, x1, c, t in parts:
            ax.add_patch(Rectangle((x0, 3 - r), x1 - x0, 0.9, fc=c, alpha=0.8, ec="white"))
    for x0, x1, c, t in parts:
        ax.text((x0 + x1) / 2 if x1 - x0 > 100 else x1 + 20, 4.25, t.replace("\n", " "), ha="center" if x1 - x0 > 100 else "left",
                fontsize=7, color=c)
    ax.text(-30, 2, "4 rows", rotation=90, ha="right", va="center", fontsize=7, color=NAVY)
    ax.set_xlim(-120, 4300); ax.set_ylim(-0.9, 4.7); ax.axis("off")
    ax.text(2040, -0.6, "4080 byte columns, sent row by row; the same frame at every OTU rate", ha="center", fontsize=6.8, color=GRAY)
    save(fig, "ch24_otn_frame")


NEW_FIGS = [fig_loop_states, fig_v21_split, fig_otn_frame, fig_fwm, fig_poisson, fig_signalling, fig_2b1q, fig_gfast_tdd, fig_fdx, fig_by_numbers, fig_rope, fig_skin, fig_tdr, fig_twist, fig_xtalk_geometry, fig_thl, fig_modem_rates,
            fig_v90_levels, fig_stuffing, fig_rate_ladder, fig_ring, fig_dsl_bandplan, fig_bit_loading,
            fig_vector_matrix, fig_eth_waveforms, fig_noise_funnel, fig_tir, fig_modes, fig_runners, fig_pmd,
            fig_edfa_levels, fig_power_staircase, fig_wdm_comb, fig_mzm, fig_iq_vs_intensity, fig_pcs,
            fig_fibre_timeline, fig_sdm_power, fig_pon_budget, fig_pon_ranging, fig_pam4_bands, fig_fso_fog, fig_poe]


def worked_numbers():
    f = np.array([1e6])
    print("3 km 26 AWG @1MHz IL", 20 * np.log10(np.abs(wl.loop_gain(f, 3000, 26))), "att/km", wl.attenuation_db_per_km(f, 26))
    _, Z0 = wl.propagation(f, 26); print("Z0 @1MHz", Z0)
    for L in (1000, 2000, 3000, 4000):
        r, ff, b, snr = adsl_rate(L, return_all=True)
        last = ff[b > 0].max() if (b > 0).any() else 0
        print(f"ADSL2+ {L} m: {r/1e6:.1f} Mb/s, bits/sym {b.sum():.0f}, last tone {last/1e6:.2f} MHz, "
              f"SNR@138k {snr[0]:.1f}, b@tone33 {b[0]}")
    r, ff, b, snr = adsl_rate(2000, return_all=True)
    for ft in (0.2e6, 0.5e6, 1.0e6, 1.5e6):
        k = np.argmin(abs(ff - ft))
        print(f"  2 km: f={ff[k]/1e6:.2f} MHz loss={20*np.log10(abs(wl.loop_gain(ff[k:k+1],2000,26)))[0]:.1f} SNR={snr[k]:.1f} b={b[k]}")
    for L in (300, 600, 1000):
        print("VDSL2", L, vdsl_rate(L) / 1e6, vdsl_rate(L, cancel_db=25.0) / 1e6, vdsl_rate(L, cancel_db=np.inf) / 1e6)
    for L in (50, 100, 200, 300, 500):
        print("Gfast", L, gfast_rate(L) / 1e6)
    for L in (100, 300, 1000, 3000):
        print("ADSL", L, adsl_rate(L, 'adsl') / 1e6)


if __name__ == "__main__":
    import sys
    worked_numbers()
    figs = [fig_line_loss, fig_loop_impairments, fig_crosstalk_vectoring, fig_dsl_rate_reach,
            fig_docsis_spectrum, fig_ethernet, fig_access_rates, fig_fibre_loss, fig_dispersion,
            fig_rx_sensitivity, fig_osnr_nli, fig_coherent_dsp] + NEW_FIGS
    sel = sys.argv[1:]
    for fn in figs:
        if not sel or any(s in fn.__name__ for s in sel):
            fn()
