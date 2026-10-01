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
            fig_rx_sensitivity, fig_osnr_nli, fig_coherent_dsp]
    sel = sys.argv[1:]
    for fn in figs:
        if not sel or any(s in fn.__name__ for s in sel):
            fn()
