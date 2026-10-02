"""Figures for Chapter 11: The Wireless Channel.

Uses commlib.channel (fspl_db, log_distance_pl_db, jakes_process, tdl_channel,
TDL_PROFILES) and commlib.modulation.ber_bpsk_rayleigh so that the book and
Lab 5 (moderncomms-labs/labs/lab05_channels.py) agree.
"""
from figstyle import *
from scipy.special import j0, i0, gamma as G, fresnel, erf, erfc, comb
import commlib as cl

C0 = 299_792_458.0


# ----------------------------------------------------------------------------- antennas
def fig_antennas():
    fig = plt.figure(figsize=(W2, 2.55))
    th = np.linspace(1e-4, 2 * np.pi - 1e-4, 2000)
    # (a) short dipole and half-wave dipole, elevation cut (theta from the z axis)
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    short = np.abs(np.sin(th))
    half = np.abs(np.cos(np.pi / 2 * np.cos(th)) / np.sin(th))
    ax.plot(th, short, color=GRAY, ls="--", label="short dipole")
    ax.plot(th, half / half.max(), color=NAVY, label=r"$\lambda/2$ dipole")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rticks([0.5, 1]); ax.set_yticklabels([]); ax.tick_params(labelsize=6.5)
    ax.set_title("(a) dipole, E-plane", fontsize=8.5, pad=10)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)

    # (b) uniform linear array, 8 elements lambda/2, broadside and steered 30 deg
    ax = fig.add_subplot(1, 3, 2)
    phi = np.linspace(-90, 90, 2001)
    N = 8
    for st, c, lab in [(0, NAVY, "broadside"), (30, ACCENT, r"steered to $30^\circ$")]:
        psi = np.pi * (np.sin(np.deg2rad(phi)) - np.sin(np.deg2rad(st)))
        af = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2) + 1e-12))
        af[np.abs(np.sin(psi / 2)) < 1e-9] = 1
        ax.plot(phi, 20 * np.log10(af + 1e-6), color=c, label=lab)
    ax.set_ylim(-40, 2); ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("angle from broadside (deg)"); ax.set_ylabel("array factor (dB)")
    ax.axhline(-13.3, color=GRAY, lw=0.7, ls=":")
    ax.text(-88, -12, "first sidelobe $-13.3$ dB", fontsize=6.3, color=GRAY)
    ax.set_title("(b) 8-element array", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="lower center")

    # (c) gain of a dish of fixed diameter vs frequency
    ax = fig.add_subplot(1, 3, 3)
    f = np.logspace(9, 11, 200)
    for D, c in [(0.6, GREEN), (3.7, NAVY), (70, ACCENT)]:
        lam = C0 / f
        g = 10 * np.log10(0.6 * (np.pi * D / lam) ** 2)
        ax.semilogx(f / 1e9, g, color=c, label=f"D = {D:g} m")
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("gain (dBi)")
    ax.set_title(r"(c) dish, $\eta=0.6$", fontsize=8.5); ax.legend(fontsize=6.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch11_antennas")


def fig_aperture():
    f = np.logspace(9, 11, 300)
    d = 10e3
    lam = C0 / f
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    # fixed-gain antennas: Friis falls 20 dB/decade
    pr_iso = 30 - cl.fspl_db(d, f)
    # fixed area, one end: gain rises as f^2 -> flat
    A = 0.5
    G_ap = 10 * np.log10(4 * np.pi * A / lam ** 2)
    pr_one = pr_iso + G_ap
    pr_both = pr_iso + 2 * G_ap
    ax[0].semilogx(f / 1e9, pr_iso, color=NAVY, label="isotropic both ends")
    ax[0].semilogx(f / 1e9, pr_one, color=GREEN, label=r"0.5 m$^2$ aperture at one end")
    ax[0].semilogx(f / 1e9, pr_both, color=ACCENT, label=r"0.5 m$^2$ apertures at both ends")
    ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("received power (dBm)")
    ax[0].set_title("1 W over 10 km, free space", fontsize=9)
    ax[0].legend(fontsize=6.4, loc="upper left")
    # effective aperture of isotropic antenna
    ax[1].loglog(f / 1e9, lam ** 2 / (4 * np.pi) * 1e4, color=NAVY)
    ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel(r"$A_e=\lambda^2/4\pi$ (cm$^2$)")
    ax[1].set_title("effective area of an isotropic antenna", fontsize=9)
    for fx, lab in [(0.9e9, "900 MHz\n88 cm$^2$"), (28e9, "28 GHz\n0.09 cm$^2$")]:
        a = (C0 / fx) ** 2 / (4 * np.pi) * 1e4
        ax[1].plot(fx / 1e9, a, "o", color=ACCENT, ms=4)
        ax[1].annotate(lab, (fx / 1e9, a), (6, 4), textcoords="offset points", fontsize=7)
    fig.tight_layout(); save(fig, "ch11_aperture")


# ----------------------------------------------------------------------------- diffraction
def knife_edge_loss(v):
    S, C = fresnel(v)   # scipy returns (S, C)
    F = 0.5 * ((0.5 - C) ** 2 + (0.5 - S) ** 2)  # |E/E0|^2, integral from v to infinity
    return -10 * np.log10(F)


def fig_knife_edge():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.05, 1]})
    # (a) geometry with first Fresnel ellipsoid
    a = ax[0]
    d = 10.0
    x = np.linspace(0, d, 300)
    lam = 0.35
    r1 = np.sqrt(lam * x * (d - x) / d)
    a.fill_between(x, 1.5 - r1, 1.5 + r1, color=NAVY, alpha=0.12, lw=0)
    a.plot(x, 1.5 + r1, color=NAVY, lw=0.8); a.plot(x, 1.5 - r1, color=NAVY, lw=0.8)
    a.plot([0, d], [1.5, 1.5], color=NAVY, lw=1.2, ls="--")
    a.plot([0, 0], [0, 1.5], color="k", lw=2); a.plot([d, d], [0, 1.5], color="k", lw=2)
    a.fill([3.5, 4.5, 4.0], [0, 0, 1.95], color=GRAY, alpha=0.8)
    a.annotate("", (4.0, 1.5), (4.0, 1.95), arrowprops=dict(arrowstyle="<->", lw=0.8, color=ACCENT))
    a.text(4.2, 1.72, "$h$", fontsize=8, color=ACCENT)
    a.text(1.5, 2.25, "first Fresnel zone", fontsize=7, color=NAVY)
    a.annotate("", (0, -0.25), (4.0, -0.25), arrowprops=dict(arrowstyle="<->", lw=0.7))
    a.annotate("", (4.0, -0.25), (d, -0.25), arrowprops=dict(arrowstyle="<->", lw=0.7))
    a.text(1.8, -0.5, "$d_1$", fontsize=8); a.text(6.8, -0.5, "$d_2$", fontsize=8)
    a.text(-0.3, 1.65, "Tx", fontsize=7.5); a.text(d - 0.3, 1.65, "Rx", fontsize=7.5)
    a.set_xlim(-0.6, d + 0.6); a.set_ylim(-0.7, 2.6); a.axis("off")
    a.set_title("(a) knife-edge geometry", fontsize=8.5)
    # (b) loss vs v
    b = ax[1]
    v = np.linspace(-3, 5, 600)
    L = knife_edge_loss(v)
    b.plot(v, -L, color=NAVY, label="exact (Fresnel integrals)")
    va = v[v > -0.78]
    Lap = 6.9 + 20 * np.log10(np.sqrt((va - 0.1) ** 2 + 1) + va - 0.1)
    b.plot(va, -Lap, color=ACCENT, ls="--", label="ITU-R P.526 approx.")
    b.axvline(0, color=GRAY, lw=0.7, ls=":"); b.axhline(-6, color=GRAY, lw=0.7, ls=":")
    b.text(0.1, -4.5, "grazing: $-6$ dB", fontsize=6.8, color=GRAY)
    b.set_xlabel(r"Fresnel--Kirchhoff parameter $\nu$"); b.set_ylabel("gain relative to free space (dB)")
    b.set_ylim(-30, 4); b.legend(fontsize=6.5, loc="lower left")
    b.set_title(r"(b) knife-edge diffraction", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_knife_edge")


# ----------------------------------------------------------------------------- two-ray
def fig_two_ray():
    f = 900e6; lam = C0 / f; ht, hr = 30.0, 1.5
    d = np.logspace(0, 4.7, 6000)
    dlos = np.sqrt(d ** 2 + (ht - hr) ** 2)
    dref = np.sqrt(d ** 2 + (ht + hr) ** 2)
    # ground reflection coefficient, vertical polarization, eps_r = 15
    er = 15
    sin_t = (ht + hr) / dref; cos_t = d / dref
    z = np.sqrt(er - cos_t ** 2)
    Gam = (er * sin_t - z) / (er * sin_t + z)
    Gh = (sin_t - z) / (sin_t + z)
    k = 2 * np.pi / lam
    E = np.exp(-1j * k * dlos) / dlos + Gam * np.exp(-1j * k * dref) / dref
    Eh = np.exp(-1j * k * dlos) / dlos + Gh * np.exp(-1j * k * dref) / dref
    pr0 = 10 * np.log10((lam / (4 * np.pi)) ** 2)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    ax.semilogx(d, pr0 + 20 * np.log10(np.abs(Eh)), color=NAVY, lw=0.9, label="two-ray, horizontal pol.")
    ax.semilogx(d, pr0 + 20 * np.log10(np.abs(E)), color=GREEN, lw=0.9, alpha=0.8, label=r"two-ray, vertical pol. ($\epsilon_r=15$)")
    ax.semilogx(d, -cl.fspl_db(d, f), color=GRAY, ls="--", label=r"free space ($d^{-2}$)")
    ax.semilogx(d, 20 * np.log10(ht * hr / d ** 2), color=ACCENT, ls="--", label=r"$h_th_r/d^2$ asymptote ($d^{-4}$)")
    dc = 4 * ht * hr / lam
    ax.axvline(dc, color=ORANGE, lw=0.9, ls=":")
    ax.text(dc * 0.9, -45, f"last maximum\n$4h_th_r/\\lambda\\approx${dc/1e3:.2f} km", fontsize=7, color=ORANGE, ha="right")
    dx_ = 4 * np.pi * ht * hr / lam
    ax.axvline(dx_, color=PURPLE, lw=0.9, ls=":")
    ax.text(dx_ * 1.1, -45, f"crossover\n$4\\pi h_th_r/\\lambda\\approx${dx_/1e3:.1f} km", fontsize=7, color=PURPLE)
    ax.set_ylim(-160, -30); ax.set_xlim(1, 5e4)
    ax.set_xlabel("distance (m)"); ax.set_ylabel("path gain (dB)")
    ax.set_title(r"Two-ray ground reflection, 900 MHz, $h_t=30$ m, $h_r=1.5$ m", fontsize=9)
    ax.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout(); save(fig, "ch11_two_ray")


# ----------------------------------------------------------------------------- ionosphere
def fig_ionosphere():
    def chapman(h, Nm, hm, H):
        z = (h - hm) / H
        return Nm * np.exp(0.5 * (1 - z - np.exp(-z)))
    h = np.linspace(50, 600, 800)
    day = chapman(h, 1e12, 300, 55) + chapman(h, 2.5e11, 190, 25) + chapman(h, 1.5e11, 110, 12) + chapman(h, 1e9, 80, 8)
    night = chapman(h, 3e11, 330, 60) + chapman(h, 6e9, 110, 10)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax[0].semilogx(day, h, color=ACCENT, label="daytime")
    ax[0].semilogx(night, h, color=NAVY, label="night")
    ax[0].set_xlim(1e8, 3e12); ax[0].set_xlabel(r"electron density (m$^{-3}$)"); ax[0].set_ylabel("height (km)")
    for y, t in [(75, "D"), (112, "E"), (190, "F1"), (300, "F2")]:
        ax[0].text(1.3e8, y - 8, t, fontsize=7, color=GRAY, fontweight="bold")
    ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_title("(a) ionospheric layers (illustrative)", fontsize=8.5)
    # MUF vs ground range for single hop, flat-earth-corrected spherical geometry
    Re = 6371.0
    D = np.linspace(100, 4000, 400)
    ax2 = ax[1]
    for foF2, c, lab in [(10, ACCENT, "day, $f_oF2=10$ MHz"), (4, NAVY, "night, $f_oF2=4$ MHz")]:
        hr = 300.0
        ang = D / (2 * Re)
        # elevation-dependent incidence angle at the reflection height (spherical earth)
        horiz = Re * np.sin(ang); vert = Re + hr - Re * np.cos(ang)
        inc = np.arctan2(horiz, vert)
        muf = foF2 / np.cos(inc)
        ax2.plot(D, muf, color=c, label=lab)
    ax2.set_xlabel("single-hop ground range (km)"); ax2.set_ylabel("MUF (MHz)")
    ax2.set_title(r"(b) MUF $= f_o\sec\theta_i$, layer at 300 km", fontsize=8.5)
    ax2.legend(fontsize=7, loc="upper left"); ax2.set_ylim(0, 40)
    fig.tight_layout(); save(fig, "ch11_ionosphere")


# ----------------------------------------------------------------------------- gases & rain
def gas_atten(f, rho=7.5):
    """Approximate sea-level specific attenuation (dB/km) of oxygen and water vapour.

    Simplified formulas from earlier editions of ITU-R P.676 (valid roughly 1-350 GHz);
    the 57-63 GHz oxygen complex is bridged by a smooth fit peaking near 15 dB/km.
    """
    f = np.asarray(f, float)
    go = np.zeros_like(f)
    lo = f < 57; hi = f > 63; mid = ~lo & ~hi
    fl = f[lo]
    go[lo] = (7.19e-3 + 6.09 / (fl ** 2 + 0.227) + 4.81 / ((fl - 57) ** 2 + 1.50)) * fl ** 2 * 1e-3
    fh = f[hi]
    go[hi] = (3.79e-7 * fh + 0.265 / ((fh - 63) ** 2 + 1.59) + 0.028 / ((fh - 118) ** 2 + 1.47)) * (fh + 198) ** 2 * 1e-3
    fm = f[mid]
    x57 = (7.19e-3 + 6.09 / (57 ** 2 + 0.227) + 4.81 / 1.5) * 57 ** 2 * 1e-3
    x63 = (3.79e-7 * 63 + 0.265 / 1.59 + 0.028 / ((63 - 118) ** 2 + 1.47)) * (261) ** 2 * 1e-3
    # quadratic in f through (57, x57), (60, 15), (63, x63)
    A = np.array([[57 ** 2, 57, 1], [60 ** 2, 60, 1], [63 ** 2, 63, 1]])
    cc = np.linalg.solve(A, [x57, 15.0, x63])
    go[mid] = cc[0] * fm ** 2 + cc[1] * fm + cc[2]
    gw = (0.050 + 0.0021 * rho + 3.6 / ((f - 22.2) ** 2 + 8.5) + 10.6 / ((f - 183.3) ** 2 + 9.0)
          + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f ** 2 * rho * 1e-4
    return go, gw


def rain_k_alpha(f):
    """ITU-R P.838-3 coefficients, horizontal polarization."""
    lf = np.log10(f)
    a = [-5.33980, -0.35351, -0.23789, -0.94158]; b = [-0.10008, 1.26970, 0.86036, 0.64552]
    c = [1.13098, 0.45400, 0.15354, 0.16817]
    k = 10 ** (sum(ai * np.exp(-((lf - bi) / ci) ** 2) for ai, bi, ci in zip(a, b, c)) - 0.18961 * lf + 0.71147)
    a = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721]; b = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980]
    c = [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990]
    al = sum(ai * np.exp(-((lf - bi) / ci) ** 2) for ai, bi, ci in zip(a, b, c)) + 0.67849 * lf - 1.95537
    return k, al


def fig_atmosphere():
    f = np.logspace(0, np.log10(350), 1500)
    go, gw = gas_atten(f)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax[0].loglog(f, go, color=NAVY, lw=1.0, label="oxygen")
    ax[0].loglog(f, gw, color=GREEN, lw=1.0, label=r"water vapour (7.5 g/m$^3$)")
    ax[0].loglog(f, go + gw, color=ACCENT, lw=1.5, label="total")
    ax[0].set_ylim(1e-3, 50); ax[0].set_xlim(1, 350)
    for fx, t in [(22.2, "22 GHz\nH$_2$O"), (60, "60 GHz\nO$_2$"), (183, "183 GHz\nH$_2$O")]:
        ax[0].annotate(t, (fx, 30 if fx != 22.2 else 0.3), fontsize=6.3, ha="center", color=GRAY)
    ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("specific attenuation (dB/km)")
    ax[0].set_title("(a) clear air, sea level (approx.)", fontsize=8.5)
    ax[0].legend(fontsize=6.3, loc="lower right")
    fr = np.logspace(0, 2, 400)
    k, al = rain_k_alpha(fr)
    for R, c in [(5, "#2E86C1"), (25, GREEN), (50, ORANGE), (100, ACCENT)]:
        ax[1].loglog(fr, k * R ** al, color=c, label=f"{R} mm/h")
    ax[1].set_ylim(1e-3, 60); ax[1].set_xlim(1, 100)
    ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel(r"$\gamma_R=kR^\alpha$ (dB/km)")
    ax[1].set_title("(b) rain, ITU-R P.838-3 (horiz. pol.)", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="lower right", title="rain rate", title_fontsize=6.5)
    fig.tight_layout(); save(fig, "ch11_atmosphere")


# ----------------------------------------------------------------------------- empirical models
def hata(fmhz, hb, hm, dkm, env="urban"):
    lf = np.log10(fmhz)
    a = (1.1 * lf - 0.7) * hm - (1.56 * lf - 0.8)
    L = 69.55 + 26.16 * lf - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(dkm)
    if env == "suburban":
        L = L - 2 * np.log10(fmhz / 28) ** 2 - 5.4
    if env == "open":
        L = L - 4.78 * lf ** 2 + 18.33 * lf - 40.94
    return L


def cost231(fmhz, hb, hm, dkm, Cm=3):
    lf = np.log10(fmhz)
    a = (1.1 * lf - 0.7) * hm - (1.56 * lf - 0.8)
    return 46.3 + 33.9 * lf - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(dkm) + Cm


def pl_38901(scn, d2, fc, hbs, hut=1.5):
    """3GPP TR 38.901 Table 7.4.1-1 (fc in GHz, d in m). Returns (PL_LOS, PL_NLOS)."""
    d3 = np.sqrt(d2 ** 2 + (hbs - hut) ** 2)
    dbp = 4 * (hbs - 1) * (hut - 1) * fc * 1e9 / C0
    lf = np.log10(fc)
    if scn == "UMa":
        l1 = 28.0 + 22 * np.log10(d3) + 20 * lf
        l2 = 28.0 + 40 * np.log10(d3) + 20 * lf - 9 * np.log10(dbp ** 2 + (hbs - hut) ** 2)
        los = np.where(d2 <= dbp, l1, l2)
        nl = 13.54 + 39.08 * np.log10(d3) + 20 * lf - 0.6 * (hut - 1.5)
    elif scn == "UMi":
        l1 = 32.4 + 21 * np.log10(d3) + 20 * lf
        l2 = 32.4 + 40 * np.log10(d3) + 20 * lf - 9.5 * np.log10(dbp ** 2 + (hbs - hut) ** 2)
        los = np.where(d2 <= dbp, l1, l2)
        nl = 22.4 + 35.3 * np.log10(d3) + 21.3 * lf - 0.3 * (hut - 1.5)
    else:  # InH
        los = 32.4 + 17.3 * np.log10(d3) + 20 * lf
        nl = 17.30 + 38.3 * np.log10(d3) + 24.9 * lf
    return los, np.maximum(los, nl)


def fig_pathloss_models():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    dkm = np.logspace(0, np.log10(20), 200)
    a = ax[0]
    a.semilogx(dkm, cl.fspl_db(dkm * 1e3, 900e6), color=GRAY, ls="--", label="free space")
    for env, c in [("urban", ACCENT), ("suburban", ORANGE), ("open", GREEN)]:
        a.semilogx(dkm, hata(900, 30, 1.5, dkm, env), color=c, label=f"Hata {env}")
    a.semilogx(dkm, cost231(1800, 30, 1.5, dkm, 3), color=PURPLE, label="COST-231, 1800 MHz, metro")
    a.set_xlabel("distance (km)"); a.set_ylabel("path loss (dB)"); a.invert_yaxis()
    a.set_title(r"(a) 900 MHz, $h_b=30$ m, $h_m=1.5$ m", fontsize=8.5)
    a.legend(fontsize=6.3, loc="upper right")
    b = ax[1]
    d = np.logspace(1, np.log10(5000), 300)
    fc = 3.5
    b.semilogx(d, cl.fspl_db(d, fc * 1e9), color=GRAY, ls="--", label="free space")
    los, nl = pl_38901("UMa", d, fc, 25)
    b.semilogx(d, los, color=NAVY, label="UMa LOS"); b.semilogx(d, nl, color=NAVY, ls="-.", label="UMa NLOS")
    los, nl = pl_38901("UMi", d, fc, 10)
    b.semilogx(d, los, color=GREEN, label="UMi LOS"); b.semilogx(d, nl, color=GREEN, ls="-.", label="UMi NLOS")
    dd = d[d <= 150]
    los, nl = pl_38901("InH", dd, fc, 3, 1)
    b.semilogx(dd, nl, color=ORANGE, ls="-.", label="InH NLOS")
    b.set_xlabel("2D distance (m)"); b.set_ylabel("path loss (dB)"); b.invert_yaxis()
    b.set_title("(b) 3GPP TR 38.901, 3.5 GHz", fontsize=8.5)
    b.legend(fontsize=6.0, loc="upper right", ncol=1)
    fig.tight_layout(); save(fig, "ch11_pathloss_models")


# ----------------------------------------------------------------------------- three scales
def fig_scales():
    r = rng(11)
    f = 2e9; lam = C0 / f
    from scipy.signal import lfilter
    d = np.linspace(50, 1000, 250_000)      # ~ 3.8 mm spacing, ~40 points per wavelength
    pl = cl.log_distance_pl_db(d, f, n=3.5)
    # correlated shadowing (Gudmundson exponential model, decorrelation distance 50 m)
    dx = d[1] - d[0]
    a = np.exp(-dx / 50)
    w = r.standard_normal(len(d)); w[0] /= np.sqrt(1 - a * a)
    sh = 7 * lfilter([np.sqrt(1 - a * a)], [1, -a], w)
    # small-scale fading: fD normalised as cycles per sample in space = dx/lam
    h = cl.jakes_process(len(d), dx / lam, n_sin=32, rng=r)
    ff = 20 * np.log10(np.abs(h))
    pt = 43
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(d, pt - pl + sh + ff, color=GRAY, lw=0.3, label="received (all three)")
    ax[0].plot(d, pt - pl + sh, color=NAVY, lw=1.2, label="path loss + shadowing")
    ax[0].plot(d, pt - pl, color=ACCENT, lw=1.6, label="path loss only ($n=3.5$)")
    ax[0].set_xlabel("distance from base station (m)"); ax[0].set_ylabel("received power (dBm)")
    ax[0].set_ylim(-130, -40); ax[0].legend(fontsize=6.5, loc="upper right")
    ax[0].set_title("(a) three scales of propagation, 2 GHz", fontsize=8.5)
    sel = (d > 400) & (d < 402)
    ax[1].plot((d[sel] - 400) / lam, (pt - pl + sh + ff)[sel], color=GRAY, lw=0.9)
    ax[1].plot((d[sel] - 400) / lam, (pt - pl + sh)[sel], color=NAVY, lw=1.2)
    ax[1].set_xlabel(r"displacement (wavelengths)"); ax[1].set_title("(b) zoom: 2 m around 400 m", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_scales")


# ----------------------------------------------------------------------------- shadowing & coverage
def fig_coverage():
    from scipy.stats import norm
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    M = np.linspace(0, 20, 200)
    for s, c in [(4, GREEN), (6, NAVY), (8, ORANGE), (10, ACCENT)]:
        ax[0].plot(M, 100 * norm.cdf(M / s), color=c, label=rf"$\sigma={s}$ dB")
    ax[0].set_xlabel("fade margin at cell edge (dB)"); ax[0].set_ylabel("edge coverage (%)")
    ax[0].set_title("(a) probability of coverage at the edge", fontsize=8.5)
    ax[0].legend(fontsize=6.8); ax[0].set_ylim(50, 100.5)
    # Jakes / Reudink area coverage
    Pedge = np.linspace(0.5, 0.99, 200)
    for n, s, c, ls in [(3.5, 8, NAVY, "-"), (3.5, 4, GREEN, "-"), (2, 8, ORANGE, "--"), (4, 10, ACCENT, "-.")]:
        a_ = -norm.ppf(Pedge) / np.sqrt(2)   # (x0 - Pr(R))/(sigma sqrt 2)
        b_ = 10 * n * np.log10(np.e) / (s * np.sqrt(2))
        Fu = 0.5 * (1 - erf(a_) + np.exp((1 - 2 * a_ * b_) / b_ ** 2) * (1 - erf((1 - a_ * b_) / b_)))
        ax[1].plot(100 * Pedge, 100 * Fu, color=c, ls=ls, label=rf"$n={n:g},\ \sigma={s}$ dB")
    ax[1].plot([50, 100], [50, 100], color=GRAY, lw=0.6, ls=":")
    ax[1].set_xlabel("edge coverage probability (%)"); ax[1].set_ylabel("area coverage (%)")
    ax[1].set_title("(b) Jakes--Reudink area coverage", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="lower right"); ax[1].set_xlim(50, 99); ax[1].set_ylim(50, 100)
    fig.tight_layout(); save(fig, "ch11_coverage")


# ----------------------------------------------------------------------------- PDP and freq response
def rms_delay(delays_ns, pdb):
    p = 10 ** (np.asarray(pdb) / 10); p /= p.sum(); t = np.asarray(delays_ns) * 1e-9
    m = np.sum(p * t)
    return m, np.sqrt(np.sum(p * t ** 2) - m ** 2)


def fig_pdp():
    r = rng(4)
    fig = plt.figure(figsize=(W2, 2.7))
    gs = fig.add_gridspec(3, 2, hspace=0.15, wspace=0.3)
    cols = {"EPA": GREEN, "EVA": NAVY, "ETU": ACCENT}
    axl = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for a, name in zip(axl, ["EPA", "EVA", "ETU"]):
        dl, p = cl.TDL_PROFILES[name]
        m, s = rms_delay(dl, p)
        a.vlines(np.array(dl) / 1e3, -25, p, color=cols[name], lw=1.4)
        a.plot(np.array(dl) / 1e3, p, "o", color=cols[name], ms=2.5)
        a.set_ylim(-25, 3); a.set_xlim(-0.1, 5.2); a.set_yticks([-20, -10, 0])
        a.text(3.9, -14, f"{name}\n$\\tau_{{rms}}$ = {s*1e9:.0f} ns", fontsize=6.6, ha="center", color=cols[name])
        a.tick_params(labelsize=7)
        if name != "ETU":
            a.set_xticklabels([])
    axl[0].set_title("(a) LTE power delay profiles (dB)", fontsize=8.5)
    axl[1].set_ylabel("relative power (dB)", fontsize=8)
    axl[2].set_xlabel(r"excess delay ($\mu$s)")
    ax = [None, fig.add_subplot(gs[:, 1])]
    # frequency responses of one random realisation over 20 MHz
    fq = np.linspace(-10e6, 10e6, 2001)
    for name in ["EPA", "ETU"]:
        dl, p = cl.TDL_PROFILES[name]
        pw = 10 ** (np.array(p) / 10); pw /= pw.sum()
        g = np.sqrt(pw / 2) * (r.standard_normal(len(pw)) + 1j * r.standard_normal(len(pw)))
        H = (g[None, :] * np.exp(-2j * np.pi * fq[:, None] * np.array(dl)[None, :] * 1e-9)).sum(1)
        m, s = rms_delay(dl, p)
        ax[1].plot(fq / 1e6, 20 * np.log10(np.abs(H)), color=cols[name],
                   label=f"{name}, $B_c\\approx 1/(5\\tau_{{rms}})$ = {1/(5*s)/1e6:.1f} MHz")
    ax[1].set_xlabel("frequency offset (MHz)"); ax[1].set_ylabel(r"$|H(f)|$ (dB)"); ax[1].set_ylim(-35, 12)
    ax[1].set_title("(b) one realisation over 20 MHz", fontsize=8.5); ax[1].legend(fontsize=6.3, loc="lower left")
    save(fig, "ch11_pdp")


def fig_tf_response():
    r = rng(7)
    fs = 15.36e6; nfft = 1024; nsym = 140; step = 1024 * 4
    n = nsym * step
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True)
    for a, (prof, fd, ttl) in zip(ax, [("EPA", 5.0, "(a) EPA, $f_D=5$ Hz (pedestrian)"),
                                       ("ETU", 300.0, "(b) ETU, $f_D=300$ Hz (vehicular)")]):
        _, taps = cl.tdl_channel(np.ones(n), fs, prof, fd, rng=r, return_taps=True)
        Ht = np.fft.fftshift(np.fft.fft(taps[::step], nfft, axis=1), axes=1)
        im = a.imshow(20 * np.log10(np.abs(Ht) + 1e-6), aspect="auto", vmin=-25, vmax=8,
                      extent=[-fs / 2e6, fs / 2e6, nsym * step / fs * 1e3, 0], cmap="viridis")
        a.set_xlabel("frequency (MHz)"); a.set_title(ttl, fontsize=8.5); a.grid(False)
    ax[0].set_ylabel("time (ms)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label(r"$|H(f,t)|$ (dB)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    save(fig, "ch11_tf_response")


# ----------------------------------------------------------------------------- Doppler
def fig_doppler():
    r = rng(8)
    fs = 2000.0; fd = 100.0
    g = cl.jakes_process(1 << 19, fd / fs, n_sin=64, rng=r)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    # (a) geometry-derived spectrum vs simulated (averaged over 200 independent realisations)
    from scipy.signal import welch
    P = 0
    for _ in range(200):
        gi = cl.jakes_process(1 << 13, fd / fs, n_sin=32, rng=r)
        fr, Pi = welch(gi, fs, nperseg=1024, return_onesided=False, detrend=False)
        P = P + Pi
    idx = np.argsort(fr); fr, P = fr[idx], P[idx]
    ax[0].plot(fr, 10 * np.log10(P / P[np.abs(fr) < 50].mean()), color=GRAY, lw=0.8, label="simulated")
    ff = np.linspace(-0.999, 0.999, 800) * fd
    S = 1 / (np.pi * fd * np.sqrt(1 - (ff / fd) ** 2))
    ax[0].plot(ff, 10 * np.log10(S / S[np.abs(ff) < 50].mean()), color=ACCENT, label="Clarke")
    ax[0].set_xlim(-160, 160); ax[0].set_ylim(-20, 15)
    ax[0].set_xlabel("Doppler (Hz)"); ax[0].set_ylabel("PSD (dB)"); ax[0].legend(fontsize=6.5, loc="lower center")
    ax[0].set_title(r"(a) spectrum, $f_D=100$ Hz", fontsize=8.5)
    lags = np.arange(0, 120)
    ac = np.array([np.mean(g[:len(g) - 200] * np.conj(g[l:len(g) - 200 + l])) for l in lags]).real
    ax[1].plot(lags / fs * 1e3, ac / ac[0], color=GRAY, lw=1.6, label="simulated")
    ax[1].plot(lags / fs * 1e3, j0(2 * np.pi * fd * lags / fs), color=NAVY, ls="--", label=r"$J_0(2\pi f_D\tau)$")
    ax[1].axvline(0.423 / fd * 1e3, color=ORANGE, lw=0.8, ls=":")
    ax[1].text(6.5, 0.5, r"$T_c\approx0.423/f_D$", fontsize=6.8, color=ORANGE)
    ax[1].set_xlabel("lag (ms)"); ax[1].set_title("(b) autocorrelation", fontsize=8.5); ax[1].legend(fontsize=6.5, loc="upper right")
    # (c) Doppler vs speed for several carriers
    v = np.linspace(0, 350, 100)
    for fc, c in [(0.9, GREEN), (3.5, NAVY), (28, ACCENT)]:
        ax[2].plot(v, v / 3.6 * fc * 1e9 / C0 / 1e3, color=c, label=f"{fc:g} GHz")
    ax[2].set_xlabel("speed (km/h)"); ax[2].set_ylabel(r"$f_D$ (kHz)"); ax[2].set_yscale("log")
    ax[2].set_ylim(1e-3, 20); ax[2].legend(fontsize=6.5); ax[2].set_title("(c) max. Doppler", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_doppler")


# ----------------------------------------------------------------------------- envelope, distributions
def fig_envelope():
    r = rng(9)
    fs = 10e3; fd = 50.0
    n = 5000
    g = cl.jakes_process(n, fd / fs, n_sin=32, rng=r)
    K = 10.0
    t = np.arange(n) / fs
    rice = np.sqrt(K / (K + 1)) * np.exp(1j * 2 * np.pi * 0.7 * fd * t) + np.sqrt(1 / (K + 1)) * cl.jakes_process(n, fd / fs, n_sin=32, rng=r)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.plot(t * 1e3, 20 * np.log10(np.abs(rice)), color=GREEN, lw=1.0, label="Rician, $K=10$")
    ax.plot(t * 1e3, 20 * np.log10(np.abs(g)), color=NAVY, lw=1.0, label="Rayleigh ($K=0$)")
    rho_db = -10
    ax.axhline(rho_db, color=ACCENT, lw=0.8, ls="--")
    below = 20 * np.log10(np.abs(g)) < rho_db
    ax.fill_between(t * 1e3, -40, 10, where=below, color=ACCENT, alpha=0.15, lw=0)
    ax.text(t[-1] * 1e3 * 0.995, -38, r"threshold $\rho=-10$ dB (dashed)", fontsize=7, color=ACCENT, ha="right")
    ax.set_ylim(-40, 10); ax.set_xlim(0, t[-1] * 1e3)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("envelope re. rms (dB)")
    ax.set_title(r"Fading envelopes, $f_D=50$ Hz (shaded: Rayleigh fades below threshold)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout(); save(fig, "ch11_envelope")


def fig_distributions():
    from scipy.stats import rice as rice_d, nakagami
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    x = np.linspace(0, 3, 500)
    for K, c in [(0, NAVY), (3, GREEN), (10, ACCENT)]:
        s = np.sqrt(K / (K + 1)); sig = np.sqrt(1 / (2 * (K + 1)))
        ax[0].plot(x, rice_d.pdf(x, s / sig, scale=sig), color=c, label=f"Rice $K={K}$" if K else "Rayleigh")
    for m, c in [(0.5, GRAY), (2, PURPLE)]:
        ax[0].plot(x, nakagami.pdf(x, m, scale=1), color=c, ls="--", label=f"Nakagami $m={m:g}$")
    ax[0].set_xlabel(r"envelope $r$ (unit mean power)"); ax[0].set_ylabel("pdf")
    ax[0].set_title("(a) envelope distributions", fontsize=8.5); ax[0].legend(fontsize=6.4)
    # CDF of power in dB: probability of fade deeper than x dB
    xd = np.linspace(-40, 10, 400); xl = 10 ** (xd / 10)
    from scipy.stats import ncx2, gamma as gam
    for K, c in [(0, NAVY), (3, GREEN), (10, ACCENT)]:
        if K == 0:
            cdf = 1 - np.exp(-xl)
        else:
            cdf = ncx2.cdf(2 * (K + 1) * xl, 2, 2 * K)
        ax[1].semilogy(xd, cdf, color=c, label=f"Rice $K={K}$" if K else "Rayleigh")
    for m, c in [(2, PURPLE), (4, GRAY)]:
        ax[1].semilogy(xd, gam.cdf(xl, m, scale=1 / m), color=c, ls="--", label=f"Nakagami $m={m}$")
    ax[1].set_ylim(1e-4, 1); ax[1].set_xlim(-40, 8)
    ax[1].set_xlabel(r"power relative to mean (dB)"); ax[1].set_ylabel(r"$P(|h|^2<x)$")
    ax[1].set_title("(b) probability of a fade deeper than $x$", fontsize=8.5); ax[1].legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch11_distributions")


def fig_lcr_afd():
    r = rng(10)
    fs = 5e3; fd = 50.0; n = 2_000_000
    g = cl.jakes_process(n, fd / fs, n_sin=32, rng=r)
    env = np.abs(g) / np.sqrt(np.mean(np.abs(g) ** 2))
    rho_db = np.arange(-30, 7, 3)
    rho = 10 ** (rho_db / 20)
    lcr_s, afd_s = [], []
    for rr in rho:
        b = env < rr
        ups = np.count_nonzero(~b[:-1] & b[1:])
        T = n / fs
        lcr_s.append(ups / T)
        afd_s.append(np.mean(b) / max(ups / T, 1e-9))
    rc = np.logspace(-2, np.log10(3), 300)
    rcd = 20 * np.log10(rc)
    lcr = np.sqrt(2 * np.pi) * fd * rc * np.exp(-rc ** 2)
    afd = (np.exp(rc ** 2) - 1) / (rc * fd * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax[0].semilogy(rcd, lcr / fd, color=NAVY, label="theory")
    ax[0].semilogy(rho_db, np.array(lcr_s) / fd, "o", color=ACCENT, ms=3.5, label="simulated")
    ax[0].set_xlabel(r"threshold $\rho$ (dB re. rms)"); ax[0].set_ylabel(r"$N_R/f_D$")
    ax[0].set_title("(a) level-crossing rate", fontsize=8.5); ax[0].legend(fontsize=7); ax[0].set_ylim(1e-2, 2); ax[0].set_xlim(-32, 8)
    ax[1].semilogy(rcd, afd * fd, color=NAVY, label="theory")
    ax[1].semilogy(rho_db, np.array(afd_s) * fd, "o", color=ACCENT, ms=3.5, label="simulated")
    ax[1].set_xlabel(r"threshold $\rho$ (dB re. rms)"); ax[1].set_ylabel(r"$\bar\tau\, f_D$")
    ax[1].set_title("(b) average fade duration", fontsize=8.5); ax[1].legend(fontsize=7); ax[1].set_xlim(-32, 8)
    fig.tight_layout(); save(fig, "ch11_lcr_afd")


# ----------------------------------------------------------------------------- regimes
def fig_regimes():
    fig, ax = plt.subplots(figsize=(W1 * 0.92, 3.3))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e3); ax.set_ylim(1e-7, 10)
    ax.axvline(1, color="k", lw=0.8); ax.axhline(1, color="k", lw=0.8)
    ax.fill_between([1e-3, 1], 1e-7, 1, color=GREEN, alpha=0.07)
    ax.fill_between([1, 1e3], 1e-7, 1, color=NAVY, alpha=0.07)
    ax.fill_between([1e-3, 1], 1, 10, color=ORANGE, alpha=0.10)
    ax.fill_between([1, 1e3], 1, 10, color=ACCENT, alpha=0.08)
    kw = dict(fontsize=7.5, fontweight="bold", ha="center")
    ax.text(0.03, 4e-7, "flat, slow", color=GREEN, **kw)
    ax.text(30, 4e-7, "frequency-selective, slow", color=NAVY, **kw)
    ax.text(0.03, 3.0, "flat, fast", color=ORANGE, **kw)
    ax.text(30, 3.0, "selective, fast", color=ACCENT, **kw)
    # (W*tau_rms*5 ~ W/Bc, Ts/Tc) approximate operating points
    pts = [
        ("LoRa SF12 (walking)", 125e3 * 5 * 0.1e-6, 32.8e-3 / (0.423 / 5)),
        ("GSM, urban car", 200e3 * 5 * 1e-6, 3.7e-6 / (0.423 / 100)),
        ("LTE 20 MHz, EVA 70 Hz", 20e6 * 5 * 0.357e-6, 71.4e-6 / (0.423 / 70)),
        ("Wi-Fi 80 MHz indoor", 80e6 * 5 * 50e-9, 4e-6 / (0.423 / 10)),
        ("NR 100 MHz, 3.5 GHz", 100e6 * 5 * 0.3e-6, 35.7e-6 / (0.423 / 400)),
        ("HF 3 kHz skywave", 3e3 * 5 * 1e-3, 0.4e-3 / (0.423 / 1)),
        ("Bluetooth LE indoor", 1e6 * 5 * 50e-9, 1e-6 / (0.423 / 10)),
        ("NR train, 350 km/h", 100e6 * 5 * 0.1e-6, 35.7e-6 / (0.423 / 1134)),
    ]
    for name, x, y in pts:
        ax.plot(x, y, "o", color=NAVY, ms=4)
        ax.annotate(name, (x, y), (5, 3), textcoords="offset points", fontsize=6.3)
    ax.set_xlabel(r"signal bandwidth / coherence bandwidth $\;W/B_c$")
    ax.set_ylabel(r"symbol duration / coherence time $\;T_s/T_c$")
    ax.set_title("The four fading regimes (approximate operating points)", fontsize=9)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch11_regimes")


# ----------------------------------------------------------------------------- BER in fading
def fig_ber_fading():
    eb = np.linspace(0, 50, 250)
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.semilogy(eb, cl.ber_bpsk(eb), color="k", lw=1.8, label="AWGN")
    ax.semilogy(eb, cl.ber_bpsk_rayleigh(eb), color=NAVY, lw=1.8, label="Rayleigh, no diversity ($L=1$)")
    g = 10 ** (eb / 10)
    for L, c in [(2, GREEN), (4, ORANGE)]:
        gb = g / L  # total Eb/N0 split over L branches (fair comparison)
        mu = np.sqrt(gb / (1 + gb))
        pb = ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))
        ax.semilogy(eb, pb, color=c, ls="--", label=f"Rayleigh, {L}-branch MRC (total energy fixed)")
    ax.semilogy(eb, 1 / (4 * g), color=GRAY, ls=":", label=r"$1/(4\bar\gamma_b)$ asymptote")
    # simulated Rician K=10 (flat, perfect CSI)
    r = rng(12)
    nb = 400_000
    x = 1 - 2 * r.integers(0, 2, nb)
    K = 10
    h = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * (r.standard_normal(nb) + 1j * r.standard_normal(nb)) / np.sqrt(2)
    es = np.arange(0, 21, 2)
    ber = []
    for e in es:
        n0 = 10 ** (-e / 10)
        y = h * x + np.sqrt(n0 / 2) * (r.standard_normal(nb) + 1j * r.standard_normal(nb))
        ber.append(np.mean(np.sign(np.real(np.conj(h) * y)) != x))
    ber = np.array(ber); ok = ber > 0
    ax.semilogy(es[ok], ber[ok], "s", color=ACCENT, ms=3.5, label="Rician $K=10$ (simulated)")
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(0, 50)
    ax.set_xlabel(r"average $E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.set_title("Coherent BPSK: the price of fading and the value of diversity", fontsize=9)
    ax.annotate("", (9.6, 1e-5), (44.0, 1e-5), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
    ax.text(20, 1.4e-5, "about 34 dB at $10^{-5}$", fontsize=7, color=ACCENT)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); save(fig, "ch11_ber_fading")


# =============================================================================
# Second-edition additions: concept illustrations, infographics and extra data figures
# =============================================================================
SW, SH = 3.0, 2.4    # narrow "side" figure size (inches)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_power_ladder():
    """By the numbers: power levels along a cellular downlink, on one dBm axis."""
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    pts = [(64, "EIRP of a macro sector\n(40 W + 18 dBi)", NAVY, 1),
           (46, "power amplifier\noutput, 40 W", NAVY, -1),
           (20, "a phone's own\ntransmitter (0.1 W)", GREEN, 1),
           (-50, "phone right under\nthe tower", GREEN, -1),
           (-85, "typical indoor\nsignal", ORANGE, 1),
           (-110, "cell edge", ACCENT, -1),
           (-125, "thermal noise in\n200 kHz (-121 dBm)", GRAY, 1)]
    ax.axhline(0, color=NAVY, lw=6, alpha=0.15)
    xs = np.linspace(-130, 70, 300)
    ax.scatter(xs, np.zeros_like(xs), c=xs, cmap="RdYlBu_r", s=30, marker="s", lw=0, zorder=2)
    for x, t, c, sgn in pts:
        y = 0.55 * sgn
        ax.plot([x, x], [0, y * 0.75], color=c, lw=0.9)
        ax.plot(x, 0, "o", color=c, ms=6, zorder=3, mec="white", mew=0.8)
        lab = (f"${x:+d}$ dBm\n" + t) if x != -125 else "$-121$ dBm\nthermal noise\nin 200 kHz"
        ax.text(x, y, lab, ha="center",
                va="bottom" if sgn > 0 else "top", fontsize=6.8, color=c)
    ax.annotate("", (-112, -1.25), (64, -1.25), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0))
    ax.text(-24, -1.18, r"about 175 dB: the edge user receives roughly $3\times10^{-18}$ of what the antenna radiates",
            ha="center", va="bottom", fontsize=7.2, color=ACCENT)
    ax.set_xlim(-140, 78); ax.set_ylim(-1.4, 1.35)
    ax.set_yticks([]); ax.set_xlabel("power (dBm)  [0 dBm = 1 mW;  every 10 dB is a factor of ten]")
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.set_title("The journey of a downlink signal, on one logarithmic ruler", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_power_ladder")


def fig_spray():
    """Path loss as paint from a spray can: the same paint spread over ever larger spheres."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    r = rng(3)
    th0 = np.deg2rad(np.linspace(-28, 28, 200))
    ax.add_patch(plt.Polygon([[-0.2, -0.12], [-0.2, 0.12], [0.08, 0.06], [0.08, -0.06]], color=NAVY))
    ax.text(-0.06, -0.3, "transmitter", ha="center", fontsize=6.8, color=NAVY)
    for k, (R, c) in enumerate([(1.0, NAVY), (2.0, GREEN), (3.0, ORANGE)]):
        ax.plot(R * np.cos(th0), R * np.sin(th0), color=c, lw=1.2)
        # same number of dots on each arc -> density falls as 1/R^2 (area grows as R^2)
        n = 60
        t = r.uniform(-np.deg2rad(28), np.deg2rad(28), n)
        rr = R + r.uniform(-0.04, 0.04, n)
        ax.scatter(rr * np.cos(t), rr * np.sin(t), s=6 / (R ** 1.2), color=c, alpha=0.8, lw=0)
        ax.text(R * np.cos(th0[-1]) + 0.05, R * np.sin(th0[-1]) + 0.08,
                ["$d$", "$2d$", "$3d$"][k], fontsize=8, color=c)
        ax.text(R + 0.02, -0.03 - 0.0, ["1", "1/4", "1/9"][k], fontsize=8, color=c, ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.text(1.6, -1.75, "same paint, area $\\propto d^2$:\ndensity $\\propto 1/d^2$", ha="center", fontsize=7.5, color=GRAY)
    ax.set_xlim(-0.35, 3.4); ax.set_ylim(-1.95, 1.75); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(); save(fig, "ch11_spray")


def fig_wave_eh():
    """A plane wave: E and H perpendicular to each other and to the direction of travel."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(SW, SH))
    ax = fig.add_subplot(111, projection="3d")
    z = np.linspace(0, 4 * np.pi, 400)
    E = np.sin(z); H = np.sin(z)
    ax.plot(z, np.zeros_like(z), E, color=ACCENT, lw=1.5)
    ax.plot(z, H, np.zeros_like(z), color=NAVY, lw=1.5)
    for zi in z[::16]:
        ax.plot([zi, zi], [0, 0], [0, np.sin(zi)], color=ACCENT, lw=0.6, alpha=0.7)
        ax.plot([zi, zi], [0, np.sin(zi)], [0, 0], color=NAVY, lw=0.6, alpha=0.7)
    ax.plot([0, 4 * np.pi + 1.5], [0, 0], [0, 0], color="k", lw=0.8)
    ax.text(4 * np.pi + 1.6, 0, 0, "travel", fontsize=7)
    ax.text(1.6, 0, 1.25, r"$\mathbf{E}$", color=ACCENT, fontsize=9)
    ax.text(1.6, 1.3, 0, r"$\mathbf{H}$", color=NAVY, fontsize=9)
    ax.set_axis_off(); ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((3, 1, 1))
    fig.subplots_adjust(0, 0, 1, 1); save(fig, "ch11_wave_eh")


def fig_pattern3d():
    """3-D radiation patterns: the dipole's doughnut versus a directive panel's flashlight beam."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(W2, 2.7))
    th = np.linspace(1e-3, np.pi - 1e-3, 90); ph = np.linspace(0, 2 * np.pi, 120)
    T, P = np.meshgrid(th, ph)
    # half-wave dipole along z
    U = (np.cos(np.pi / 2 * np.cos(T)) / np.sin(T)) ** 2
    Rr = U / U.max()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    X = Rr * np.sin(T) * np.cos(P); Y = Rr * np.sin(T) * np.sin(P); Z = Rr * np.cos(T)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.Blues(0.35 + 0.6 * Rr), rstride=2, cstride=2, lw=0, antialiased=True, shade=False)
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color="k", lw=2)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(18, 30)
    ax.set_title("(a) half-wave dipole: 2.15 dBi\n(a doughnut, like a bare bulb)", fontsize=8)
    # directive pattern: 8x4 array of patches facing +x
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    th = np.linspace(0, np.pi, 160); ph = np.linspace(-np.pi, np.pi, 200)
    T, P = np.meshgrid(th, ph)
    ux = np.sin(T) * np.cos(P); uy = np.sin(T) * np.sin(P); uz = np.cos(T)
    def af(N, u):
        psi = np.pi * u
        a = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2) + 1e-12)); a[np.abs(np.sin(psi / 2)) < 1e-9] = 1
        return a
    el = np.clip(ux, 0, None) ** 1.2
    G = (el * af(8, uz) * af(4, uy)) ** 2
    Rr = np.clip(10 * np.log10(G / G.max() + 1e-6) + 30, 0, None) / 30
    X = Rr * ux; Y = Rr * uy; Z = Rr * uz
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.Oranges(0.25 + 0.7 * Rr), rstride=2, cstride=2, lw=0, antialiased=True, shade=False)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(20, -55)
    ax.set_title("(b) $8\\times4$ patch panel: about 18 dBi\n(a flashlight; dB scale, 30 dB range)", fontsize=8)
    fig.subplots_adjust(0, 0, 1, 0.88, wspace=0.0); save(fig, "ch11_pattern3d")


def fig_polarization():
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(W2, 2.2))
    z = np.linspace(0, 4 * np.pi, 500)
    cases = [("(a) vertical linear", np.zeros_like(z), np.sin(z), ACCENT),
             ("(b) slant $45^\\circ$ linear", np.sin(z) / np.sqrt(2), np.sin(z) / np.sqrt(2), ORANGE),
             ("(c) circular (RHCP)", np.cos(z), np.sin(z), NAVY)]
    for i, (t, ex, ey, c) in enumerate(cases):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot(z, ex, ey, color=c, lw=1.3)
        for zi in range(0, len(z), 14):
            ax.plot([z[zi], z[zi]], [0, ex[zi]], [0, ey[zi]], color=c, lw=0.5, alpha=0.6)
        ax.plot([0, 4 * np.pi + 1], [0, 0], [0, 0], color="k", lw=0.7)
        ax.set_axis_off(); ax.set_box_aspect((2.6, 1, 1)); ax.view_init(22, -50)
        ax.set_title(t, fontsize=8, pad=-4)
    fig.subplots_adjust(0, 0, 1, 0.92, wspace=0.0); save(fig, "ch11_polarization")


def fig_lte_waterfall():
    items = [("UE power", 23.0), ("antenna &\nbody loss", -3.0), ("path loss\n(MAPL)", -130.7),
             ("BS antenna", 18.0), ("feeder", -2.0), ("interference\nmargin", -2.0),
             ("shadowing\nmargin", -8.7), ("building\npenetration", -15.0)]
    fig, ax = plt.subplots(figsize=(W2, 2.7))
    lvl = 0.0; xs = []
    for i, (n, v) in enumerate(items):
        if i == 0:
            lo, hi = min(0, v), max(0, v); lvl = v
            c = NAVY
        else:
            new = lvl + v; lo, hi = min(lvl, new), max(lvl, new)
            c = GREEN if v > 0 else (ACCENT if "path" in n else ORANGE)
            ax.plot([i - 1 + 0.35, i - 0.35], [lvl, lvl], color=GRAY, lw=0.6)
            lvl = new
        ax.bar(i, hi - lo, bottom=lo, color=c, width=0.7, alpha=0.85)
        ax.text(i, hi + 2.5, f"{v:+.1f}", ha="center", fontsize=6.8)
    ax.bar(len(items), 0.0, bottom=0)
    sens = -120.4
    ax.axhline(sens, color=PURPLE, ls="--", lw=1)
    ax.text(len(items) - 0.5, sens - 8, f"receiver sensitivity {sens} dBm\n(the budget just closes)", fontsize=7,
            color=PURPLE, ha="right", va="top")
    ax.set_xticks(range(len(items))); ax.set_xticklabels([n for n, _ in items], fontsize=6.6)
    ax.set_ylabel("signal level (dBm)"); ax.set_ylim(-150, 35); ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_title("The LTE uplink budget of Table 11.1 as a waterfall", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_lte_waterfall")


def fig_wifi_rings():
    fig, ax = plt.subplots(figsize=(W2, 2.7))
    # office floor: grid of rooms 8 m x 6 m along two corridors
    for x in range(0, 120, 8):
        for y0 in (0, 9, 24, 33):
            ax.add_patch(plt.Rectangle((x, y0), 8, 6, fill=False, lw=0.4, ec=GRAY))
    ax.add_patch(plt.Rectangle((0, 0), 120, 39, fill=False, lw=1.2, ec="k"))
    ap = (56, 19.5)
    ax.plot(*ap, "^", color="k", ms=8); ax.text(ap[0] + 1.5, ap[1] + 0.8, "AP", fontsize=7.5)
    for R, lab, c, pos in [(57, "MCS 0|BPSK 1/2", NAVY, (106, 19.5)), (26, "MCS 4|16-QAM", GREEN, (82, 19.5)),
                           (16, "MCS 7|64-QAM", ORANGE, (40, 19.5)), (11, "MCS 9|256-QAM", ACCENT, (56, 30.5))]:
        ax.add_patch(plt.Circle(ap, R, color=c, alpha=0.10, lw=0))
        ax.add_patch(plt.Circle(ap, R, fill=False, color=c, lw=1.1))
        ax.text(*pos, lab.replace("|", "\n") + f"\n{R} m", fontsize=6.2, color=c, ha="center", va="center",
                bbox=dict(fc="white", ec=c, lw=0.5, alpha=0.9, pad=0.6))
    ax.set_xlim(-2, 122); ax.set_ylim(-2, 41); ax.set_aspect("equal"); _clean(ax)
    ax.set_title("Approximate 802.11ac range rings through interior walls (3GPP InH NLOS model, 5.5 GHz)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_wifi_rings")


def fig_fresnel_refl():
    psi = np.linspace(0.01, 90, 600); s = np.sin(np.deg2rad(psi)); c2 = np.cos(np.deg2rad(psi)) ** 2
    fig, ax = plt.subplots(figsize=(SW, SH))
    for er, ls in [(15, "-"), (4, "--")]:
        z = np.sqrt(er - c2)
        gv = (er * s - z) / (er * s + z); gh = (s - z) / (s + z)
        ax.plot(psi, np.abs(gh), color=NAVY, ls=ls, label=rf"horizontal, $\epsilon_r={er}$")
        ax.plot(psi, np.abs(gv), color=ACCENT, ls=ls, label=rf"vertical, $\epsilon_r={er}$")
    ax.set_xlabel("grazing angle $\\psi$ (deg)"); ax.set_ylabel(r"$|\Gamma|$")
    b = np.rad2deg(np.arctan(1 / np.sqrt(15)))
    ax.annotate("Brewster dip", (b, 0.02), (30, 0.35), fontsize=6.8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax.legend(fontsize=5.8, loc="upper right"); ax.set_xlim(0, 90); ax.set_ylim(0, 1.02)
    fig.tight_layout(); save(fig, "ch11_fresnel_refl")


def fig_fresnel_football():
    """The worked microwave example: Fresnel 'football', Earth bulge, a tree line and the masts."""
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    D = 30.0; x = np.linspace(0, D, 400); lam = 0.05
    k = 4 / 3; Re = 6371.0
    bulge = (x * (D - x)) / (2 * k * Re) * 1e3      # metres
    bulge23 = (x * (D - x)) / (2 * (2 / 3) * Re) * 1e3
    r1 = np.sqrt(lam * x * 1e3 * (D - x) * 1e3 / (D * 1e3))
    h = 45.0
    ax.fill_between(x, -5, bulge, color="#8B6F47", alpha=0.35, lw=0)
    ax.plot(x, bulge, color="#8B6F47", lw=1)
    ax.plot(x, bulge23, color="#8B6F47", lw=0.8, ls=":")
    ax.text(0.8, 17.5, "Earth bulge, $k=2/3$", fontsize=6.5, color="#6B4F27")
    ax.text(3.5, 2, "Earth bulge, $k=4/3$", fontsize=6.5, color="#6B4F27")
    ax.fill_between(x, h - r1, h + r1, color=NAVY, alpha=0.13, lw=0)
    ax.plot(x, h + r1, color=NAVY, lw=0.8); ax.plot(x, h - r1, color=NAVY, lw=0.8)
    ax.plot(x, h - 0.6 * r1, color=GREEN, lw=0.8, ls="--")
    ax.plot([0, D], [h, h], color=ACCENT, lw=1.2)
    for xx in (0, D):
        ax.plot([xx, xx], [0, h], color="k", lw=2.5); ax.plot(xx, h, "s", color="k", ms=4)
    # tree line at mid-path
    for xt in np.linspace(13.6, 16.4, 9):
        b = (xt * (D - xt)) / (2 * k * Re) * 1e3
        ax.add_patch(plt.Polygon([[xt - 0.25, b], [xt + 0.25, b], [xt, b + 20]], color=GREEN, alpha=0.75))
    ax.text(15, 36.5, "20 m trees", fontsize=6.8, ha="center", color=GREEN)
    ax.text(19.0, h + 8, "first Fresnel zone\n($r_1=19.4$ m at mid-path)", fontsize=6.8, color=NAVY)
    ax.text(25.0, h - 0.6 * r1[330] - 6, "60% clearance", fontsize=6.5, color=GREEN)
    ax.text(0.5, h + 2, "45 m mast", fontsize=6.8); ax.text(D - 0.5, h + 2, "45 m mast", fontsize=6.8, ha="right")
    ax.set_xlim(-1, D + 1); ax.set_ylim(-3, 70)
    ax.set_xlabel("distance along path (km)"); ax.set_ylabel("height (m)")
    ax.set_title("A 30 km, 6 GHz hop: the 'football' must clear trees and the curved Earth (vertical scale exaggerated)", fontsize=8)
    fig.tight_layout(); save(fig, "ch11_fresnel_football")


def fig_interference_map():
    """Two-ray lobing: field strength over distance and receiver height."""
    f = 900e6; lam = C0 / f; ht = 30.0
    d = np.linspace(20, 2500, 900); hr = np.linspace(0.2, 60, 500)
    Dg, Hg = np.meshgrid(d, hr)
    k = 2 * np.pi / lam
    dl = np.sqrt(Dg ** 2 + (ht - Hg) ** 2); dr = np.sqrt(Dg ** 2 + (ht + Hg) ** 2)
    E = np.exp(-1j * k * dl) / dl - np.exp(-1j * k * dr) / dr
    P = 20 * np.log10(np.abs(E) * Dg)       # relative to free space
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    im = ax.imshow(P, origin="lower", aspect="auto", extent=[d[0], d[-1], hr[0], hr[-1]], cmap="magma", vmin=-25, vmax=6)
    ax.plot([0, 0], [0, ht], color="w", lw=3)
    ax.axhline(1.5, color="cyan", lw=0.8, ls="--"); ax.text(1700, 3, "phone height 1.5 m", color="cyan", fontsize=7)
    ax.set_xlabel("distance from a 30 m mast (m)"); ax.set_ylabel("receiver height (m)")
    ax.set_title("Direct ray + ground reflection at 900 MHz: interference lobes (power relative to free space)", fontsize=8.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("dB re. free space", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); save(fig, "ch11_interference_map")


def fig_horizon():
    h = np.logspace(0, 4.1, 200)
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.loglog(h, 4.12 * np.sqrt(h), color=NAVY)
    for hh, t in [(1.5, "phone"), (30, "mast"), (300, "tall tower"), (10000, "airliner")]:
        dd = 4.12 * np.sqrt(hh)
        ax.plot(hh, dd, "o", color=ACCENT, ms=4)
        ax.annotate(f"{t}\n{dd:.0f} km", (hh, dd), (-6, 6), textcoords="offset points", fontsize=6.5, ha="right")
    ax.set_xlabel("antenna height (m)"); ax.set_ylabel("radio horizon (km)")
    ax.set_title(r"$d_{hor}\approx4.12\sqrt{h}$ ($k=4/3$)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_horizon")


def fig_rain_fade_bar():
    fr = np.array([4, 8, 12, 20, 30, 40, 60.0])
    k, al = rain_k_alpha(fr)
    A = k * 50 ** al * 3.0
    fig, ax = plt.subplots(figsize=(SW, SH))
    cols = [GREEN if a < 5 else (ORANGE if a < 15 else ACCENT) for a in A]
    ax.bar(range(len(fr)), A, color=cols, alpha=0.85)
    for i, a in enumerate(A):
        ax.text(i, a + 1, f"{a:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(fr))); ax.set_xticklabels([f"{f:g}" for f in fr])
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("rain fade (dB)")
    ax.set_title("50 mm/h over 3 km effective path", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_rain_fade_bar")


def fig_exponent_ladder():
    envs = [("corridor / factory LOS", 1.6, 1.8, GREEN), ("free space", 2.0, 2.0, NAVY),
            ("urban cellular", 2.7, 3.5, ORANGE), ("shadowed urban", 3.0, 5.0, ACCENT),
            ("in building, through walls/floors", 4.0, 6.0, PURPLE)]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for i, (n, a, b, c) in enumerate(envs):
        if a == b:
            ax.plot(a, i, "D", color=c, ms=7)
        else:
            ax.barh(i, b - a, left=a, height=0.5, color=c, alpha=0.8)
        ax.text(b + 0.08, i, n, va="center", fontsize=7.5, color=c)
    ax.set_yticks([]); ax.set_xlim(1.4, 8.2); ax.set_xlabel("path-loss exponent $n$  (power falls as $d^{-n}$)")
    ax.invert_yaxis(); ax.spines["left"].set_visible(False)
    ax.set_title("Measured path-loss exponents by environment (typical ranges)", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_exponent_ladder")


def fig_los_prob():
    d = np.linspace(1, 500, 500)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for dd2, c, lab in [(36, GREEN, "UMi (street canyon)"), (63, NAVY, "UMa ($h_{UT}\\leq13$ m)")]:
        p = np.where(d <= 18, 1.0, 18 / d + np.exp(-d / dd2) * (1 - 18 / d))
        ax.plot(d, p, color=c, label=lab)
    ax.plot(100, 18 / 100 + np.exp(-100 / 36) * 0.82, "o", color=ACCENT, ms=4)
    ax.annotate("100 m: about 1 in 4", (100, 0.23), (150, 0.55), fontsize=6.8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax.set_xlabel("2D distance (m)"); ax.set_ylabel("probability of line of sight")
    ax.legend(fontsize=6.5); ax.set_ylim(0, 1.05)
    ax.set_title("TR 38.901 LOS probability", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_los_prob")


def fig_o2i_bars():
    f = np.array([0.9, 3.5, 28])
    mats = [("standard glass", 2 + 0.2 * f, "#2E86C1"), ("IRR glass", 23 + 0.3 * f, PURPLE),
            ("concrete", 5 + 4 * f, GRAY), ("wood", 4.85 + 0.12 * f, ORANGE)]
    fig, ax = plt.subplots(figsize=(SW, SH))
    w = 0.2
    for j, (n, L, c) in enumerate(mats):
        Lc = np.minimum(L, 60)
        ax.bar(np.arange(3) + (j - 1.5) * w, Lc, w, color=c, label=n)
        for i, (l, lc) in enumerate(zip(L, Lc)):
            if l > 60:
                ax.text(i + (j - 1.5) * w, 61, f"{l:.0f}", ha="center", fontsize=6.3, color=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.9 GHz", "3.5 GHz", "28 GHz"])
    ax.set_ylabel("penetration loss (dB)"); ax.set_ylim(0, 68)
    ax.legend(fontsize=6.2, loc="upper left"); ax.set_title("TR 38.901 material losses", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_o2i_bars")


def fig_hata_cells():
    """Indoor (0.70 km) versus outdoor (1.9 km) cell radius: hexagons needed to cover the same city."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    L = 8.0
    for a, R, c, t in [(ax[0], 0.70, ACCENT, "indoor service: $R=0.70$ km"), (ax[1], 1.9, GREEN, "outdoor service: $R=1.9$ km")]:
        n = 0
        dx = 1.5 * R; dy = np.sqrt(3) * R
        for i in range(-2, int(L / dx) + 3):
            for j in range(-2, int(L / dy) + 3):
                cx = i * dx; cy = j * dy + (dy / 2 if i % 2 else 0)
                ang = np.linspace(0, 2 * np.pi, 7)
                a.fill(cx + R * np.cos(ang), cy + R * np.sin(ang), fc=c, alpha=0.10, lw=0)
                a.plot(cx + R * np.cos(ang), cy + R * np.sin(ang), color=c, lw=0.6)
                a.plot(cx, cy, "^", color=c, ms=2.5)
                if 0 <= cx <= L and 0 <= cy <= L:
                    n += 1
        a.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=1.5, ec="k"))
        a.set_xlim(-0.3, L + 0.3); a.set_ylim(-0.3, L + 0.3); a.set_aspect("equal"); _clean(a)
        a.set_title(f"{t}\nabout {n} sites in an 8 km $\\times$ 8 km city", fontsize=8)
    fig.tight_layout(); save(fig, "ch11_hata_cells")


def fig_shadow_map():
    """Coverage of one cell with correlated log-normal shadowing: a ragged, island-strewn edge."""
    from scipy.ndimage import gaussian_filter
    r = rng(21)
    N = 400; L = 3000.0
    x = np.linspace(-L / 2, L / 2, N); X, Y = np.meshgrid(x, x)
    d = np.maximum(np.hypot(X, Y), 20)
    pl = 128.1 + 37.6 * np.log10(d / 1000)       # 3GPP-style macro model at ~2 GHz
    w = gaussian_filter(r.standard_normal((N, N)), sigma=50 / (L / N) / 1.4, mode="wrap")
    w = 8 * w / w.std()
    pr = 46 + 15 - 18 - pl - w     # 18 dB building entry loss: an indoor service
    thr = -85.0
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    im = ax[0].imshow(pr, origin="lower", extent=[-1.5, 1.5, -1.5, 1.5], cmap="viridis", vmin=-115, vmax=-55)
    ax[0].set_title("(a) received power (dBm), $\\sigma=8$ dB", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6.5)
    Rm = 10 ** ((46 + 15 - 18 - thr - 128.1) / 37.6)
    ax[1].imshow(pr >= thr, origin="lower", extent=[-1.5, 1.5, -1.5, 1.5], cmap=matplotlib.colors.ListedColormap(["#EBA9A0", "#93D3AC"]))
    th = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(Rm * np.cos(th), Rm * np.sin(th), color="k", ls="--", lw=1)
    ax[1].text(0, -Rm - 0.17, "median-model edge", ha="center", fontsize=6.8)
    ax[1].set_title("(b) covered (green) vs not (pink)", fontsize=8.5)
    for a in ax:
        a.plot(0, 0, "^", color=ACCENT, ms=7); a.set_xlabel("km"); a.grid(False)
    ax[0].set_ylabel("km")
    fig.tight_layout(); save(fig, "ch11_shadow_map")


def fig_shadow_hist():
    r = rng(5)
    from scipy.ndimage import gaussian_filter1d
    fig, ax = plt.subplots(figsize=(SW, SH))
    # many random "obstacle" losses multiply: their dB values add -> Gaussian
    n_obst = r.poisson(6, 50_000)
    tot = np.array([r.exponential(2.0, k).sum() for k in n_obst])
    tot = tot - tot.mean()
    ax.hist(tot, 70, density=True, color=NAVY, alpha=0.55, label="sum of random\nobstacle losses")
    xx = np.linspace(-20, 25, 300); s = tot.std()
    ax.plot(xx, np.exp(-xx ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)), color=ACCENT, label=f"Gaussian, $\\sigma$={s:.1f} dB")
    ax.set_xlabel("loss about the median (dB)"); ax.set_yticks([]); ax.legend(fontsize=6.3)
    ax.set_title("why shadowing is log-normal", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_shadow_hist")


def fig_multipath_cartoon():
    """Top view of a street: rays bounce off buildings and arrive with different delays."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.65, 1]})
    a = ax[0]
    blds = [(0, 4, 3, 3), (4, 4.6, 4, 2.4), (9, 4.3, 3, 2.7), (13, 4.5, 3, 2.5),
            (1, -3, 3.5, 2.3), (5.5, -3.2, 3, 2.5), (9.5, -3, 3.5, 2.2), (14, -3.1, 2.5, 2.4), (6.6, 1.1, 1.0, 1.0)]
    for (x, y, w, h) in blds:
        a.add_patch(plt.Rectangle((x, y), w, h, color=GRAY, alpha=0.35))
    BS = (1.0, 1.6); UE = (15.0, 0.4)
    a.plot(*BS, "^", color=NAVY, ms=9); a.text(BS[0] - 1.2, BS[1] - 1.0, "base\nstation", fontsize=6.8, color=NAVY)
    a.plot(*UE, "s", color=ACCENT, ms=7); a.text(UE[0] - 0.9, UE[1] - 0.85, "phone", fontsize=6.8, color=ACCENT)
    paths = [[BS, UE], [BS, (6.0, 4.6), UE], [BS, (7.5, -0.7), UE], [BS, (11.5, 4.3), UE], [BS, (3.0, -0.7), (11.8, -0.8), UE]]
    cols = [NAVY, GREEN, ORANGE, PURPLE, "#2E86C1"]
    lens = []
    for p, c in zip(paths, cols):
        p = np.array(p); a.plot(p[:, 0], p[:, 1], color=c, lw=1.1, alpha=0.9)
        lens.append(np.sum(np.hypot(*np.diff(p, axis=0).T)))
    a.set_xlim(-0.5, 17); a.set_ylim(-3.4, 7.2); a.set_aspect("equal"); _clean(a)
    a.set_title("(a) five ways to reach the phone (top view)", fontsize=8.5)
    b = ax[1]
    scale = 100.0   # 1 drawing unit = 100 m -> delay in us
    L0 = lens[0]
    for Lp, c, k in zip(lens, cols, range(5)):
        tau = (Lp - L0) * scale / C0 * 1e6
        amp = (L0 / Lp) ** 2 * (1 if k == 0 else 0.6 ** (1 + (k == 4)))
        b.vlines(tau, 0, amp, color=c, lw=2.2); b.plot(tau, amp, "o", color=c, ms=4)
    b.set_xlabel(r"excess delay ($\mu$s)"); b.set_ylabel("amplitude")
    b.set_title("(b) what the phone receives:\nan impulse response", fontsize=8.5); b.set_ylim(0, 1.15)
    fig.tight_layout(); save(fig, "ch11_multipath_cartoon")


def fig_speckle():
    """Interference of plane waves: 2 waves give stripes, 30 random waves give a speckle of fades."""
    r = rng(17)
    n = 400; L = 4.0   # wavelengths
    x = np.linspace(0, L, n); X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    for a, N, t in [(ax[0], 1, "(a) one wave: uniform"), (ax[1], 2, "(b) two waves: stripes"), (ax[2], 30, "(c) 30 waves: speckle")]:
        E = np.zeros_like(X, complex)
        angs = [0.3] if N == 1 else ([0.2, 0.2 + np.pi * 0.85] if N == 2 else r.uniform(0, 2 * np.pi, N))
        for th in angs:
            E += np.exp(2j * np.pi * (X * np.cos(th) + Y * np.sin(th)) + 1j * r.uniform(0, 2 * np.pi))
        P = 20 * np.log10(np.abs(E) / np.sqrt(np.mean(np.abs(E) ** 2)) + 1e-3)
        a.imshow(P, origin="lower", extent=[0, L, 0, L], cmap="magma", vmin=-25, vmax=7)
        a.set_title(t, fontsize=8); a.set_xticks([0, 1, 2, 3, 4]); a.set_yticks([0, 1, 2, 3, 4]); a.grid(False)
        a.set_xlabel(r"$x/\lambda$", fontsize=8); a.tick_params(labelsize=6.5)
    ax[0].set_ylabel(r"$y/\lambda$", fontsize=8)
    fig.tight_layout(); save(fig, "ch11_speckle")


def fig_doppler_geometry():
    """Clarke's ring of scatterers around a moving receiver; each angle maps to f_D cos(theta)."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    cmap = plt.cm.coolwarm
    for t in th:
        c = cmap(0.5 + 0.5 * np.cos(t))
        a.plot(np.cos(t), np.sin(t), "o", color=c, ms=6)
        a.annotate("", (0.18 * np.cos(t), 0.18 * np.sin(t)), (0.88 * np.cos(t), 0.88 * np.sin(t)),
                   arrowprops=dict(arrowstyle="->", color=c, lw=0.7))
    a.add_patch(plt.Rectangle((-0.13, -0.07), 0.26, 0.14, color="k"))
    a.annotate("", (0.55, 0), (0.15, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    a.text(0.25, 0.08, "$v$", fontsize=9)
    a.text(1.12, 0, "$+f_D$\n(ahead)", fontsize=6.8, color=cmap(1.0), va="center")
    a.text(-1.12, 0, "$-f_D$\n(behind)", fontsize=6.8, color=cmap(0.0), va="center", ha="right")
    a.text(0, 1.15, "$0$ (side)", fontsize=6.8, ha="center")
    a.set_xlim(-1.6, 1.6); a.set_ylim(-1.3, 1.35); a.set_aspect("equal"); _clean(a)
    a.set_title("(a) scatterers all around", fontsize=8.5)
    b = ax[1]
    r = rng(2)
    t = r.uniform(0, 2 * np.pi, 400_000)
    fr = np.cos(t)
    hst, edges = np.histogram(fr, 40, range=(-1, 1), density=True)
    cen = 0.5 * (edges[1:] + edges[:-1])
    b.bar(cen, hst, width=0.05, color=cmap(0.5 + 0.5 * cen), alpha=0.9)
    ff = np.linspace(-0.995, 0.995, 400)
    b.plot(ff, 1 / (np.pi * np.sqrt(1 - ff ** 2)), color="k", lw=1)
    b.set_ylim(0, 2.2); b.set_xlabel(r"Doppler shift $f/f_D=\cos\theta$"); b.set_ylabel("power density")
    b.set_title("(b) uniform angles pile up at $\\pm f_D$:\nthe 'bathtub'", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_doppler_geometry")


def fig_phasor_sum():
    r = rng(23)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    N = 12
    # (a) one realisation: head-to-tail random walk of equal phasors
    a = ax[0]
    ph = r.uniform(0, 2 * np.pi, N); z = np.concatenate([[0], np.cumsum(np.exp(1j * ph) / np.sqrt(N))])
    for i in range(N):
        a.annotate("", (z[i + 1].real, z[i + 1].imag), (z[i].real, z[i].imag), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    a.annotate("", (z[-1].real, z[-1].imag), (0, 0), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.6))
    a.text(z[-1].real, z[-1].imag + 0.08, "sum $h$", color=ACCENT, fontsize=7)
    a.set_aspect("equal"); a.set_xlim(-0.8, 0.8); a.set_ylim(-0.8, 0.8); a.set_title("(a) 12 echoes, random phases", fontsize=8)
    a.tick_params(labelsize=6.5)
    for b, K, t, c in [(ax[1], 0, "(b) Rayleigh: no dominant path", NAVY), (ax[2], 10, "(c) Rician $K=10$: strong LOS", GREEN)]:
        M = 3000
        los = np.sqrt(K / (K + 1))
        sc = np.sqrt(1 / (K + 1)) * (np.exp(1j * r.uniform(0, 2 * np.pi, (M, N))).sum(1) / np.sqrt(N))
        h = los + sc
        b.plot(h.real, h.imag, ".", color=c, ms=1.2, alpha=0.4)
        b.add_patch(plt.Circle((0, 0), 0.316, fill=False, color=ACCENT, lw=0.9, ls="--"))
        if K:
            b.annotate("", (los, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
        b.set_aspect("equal"); b.set_xlim(-2, 2); b.set_ylim(-2, 2); b.set_title(t, fontsize=8)
        b.tick_params(labelsize=6.5)
        b.text(-1.9, -1.85, "dashed: $-10$ dB", fontsize=6.3, color=ACCENT)
    fig.tight_layout(); save(fig, "ch11_phasor_sum")


def fig_scattering_fn():
    tau = np.linspace(0, 5, 300); nu = np.linspace(-1.2, 1.2, 300)
    T, Nu = np.meshgrid(tau, nu)
    S = np.zeros_like(T)
    for t0, n0, p, st, sn in [(0.1, 0.0, 1.0, 0.08, 0.95), (0.8, 0.6, 0.5, 0.15, 0.25), (1.6, -0.5, 0.3, 0.2, 0.3), (3.4, 0.2, 0.12, 0.3, 0.35)]:
        S += p * np.exp(-((T - t0) ** 2) / (2 * st ** 2)) * np.exp(-((Nu - n0) ** 2) / (2 * sn ** 2))
    S[:, :] *= 1
    fig = plt.figure(figsize=(W1, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 3], hspace=0.06, wspace=0.05)
    a = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=a); side = fig.add_subplot(gs[1, 1], sharey=a)
    a.imshow(10 * np.log10(S + 1e-4), origin="lower", aspect="auto", extent=[0, 5, -1.2, 1.2], cmap="viridis", vmin=-25, vmax=0)
    a.set_xlabel(r"delay $\tau$ ($\mu$s)"); a.set_ylabel(r"Doppler $\nu/f_D$"); a.grid(False)
    for (x, y, t) in [(0.1, 1.0, "local scatterers\n(all directions)"), (0.85, 0.85, "building ahead"), (1.6, -0.95, "building behind"), (3.4, 0.6, "distant hill")]:
        a.text(x + 0.08, y, t, color="w", fontsize=6.5, va="center")
    top.fill_between(tau, S.sum(0) / S.sum(0).max(), color=NAVY, alpha=0.5); top.set_ylabel("PDP", fontsize=7)
    top.set_yticks([]); plt.setp(top.get_xticklabels(), visible=False)
    side.fill_betweenx(nu, S.sum(1) / S.sum(1).max(), color=ACCENT, alpha=0.5); side.set_xlabel("Doppler\nspectrum", fontsize=7)
    side.set_xticks([]); plt.setp(side.get_yticklabels(), visible=False)
    top.set_title(r"A scattering function $\mathcal{S}(\tau,\nu)$ (illustrative) and its two marginals", fontsize=8.5)
    save(fig, "ch11_scattering_fn")


def fig_model_timeline():
    ev = [(1963, "Bello: WSSUS", NAVY, 1, 0), (1968, "Okumura's Tokyo|drive tests; Clarke", NAVY, -1, 0),
          (1974, "Jakes: Microwave|Mobile Communications", NAVY, 1, 0), (1980, "Hata's formula", ORANGE, -1, 0),
          (1989, "COST 207|(GSM: TU, BU,|RA, HT)", GREEN, 1, -1.5), (1991, "Gudmundson:|correlated shadowing", ORANGE, -1, 0),
          (1997, "ITU-R M.1225|(IMT-2000)", GREEN, 1, 2.5), (2008, "3GPP EPA/EVA/ETU|(LTE)", GREEN, -1, -1.5),
          (2013, "mmWave campaigns|28/38/73 GHz", ORANGE, 1, 0), (2017, "3GPP TR 38.901|TDL/CDL, 0.5-100 GHz", ACCENT, -1, 2.0)]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.axhline(0, color=NAVY, lw=2)
    for y, t, c, s, dx in ev:
        t = t.replace("|", chr(10))
        ax.plot(y, 0, "o", color=c, ms=6, zorder=3)
        ax.plot([y, y + dx], [0, 0.42 * s], color=c, lw=0.8)
        ax.text(y + dx, 0.47 * s, f"{y}\n{t}" if s > 0 else f"{t}\n{y}", ha="center", va="bottom" if s > 0 else "top", fontsize=6.4, color=c)
    ax.set_xlim(1958, 2022); ax.set_ylim(-1.25, 1.25); _clean(ax)
    ax.text(1958.5, -1.2, "navy: theory   orange: measurements   green/red: standard models", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch11_model_timeline")


def fig_diversity_branches():
    r = rng(31)
    fs = 2000.0; fd = 10.0; n = 4000
    t = np.arange(n) / fs
    g1 = cl.jakes_process(n, fd / fs, n_sin=32, rng=r); g2 = cl.jakes_process(n, fd / fs, n_sin=32, rng=r)
    e1 = 20 * np.log10(np.abs(g1)); e2 = 20 * np.log10(np.abs(g2))
    sel = np.maximum(e1, e2)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.plot(t, e1, color=NAVY, lw=0.8, alpha=0.75, label="antenna 1")
    ax.plot(t, e2, color=GREEN, lw=0.8, alpha=0.75, label="antenna 2")
    ax.plot(t, sel, color=ACCENT, lw=1.5, label="pick the stronger (selection)")
    ax.axhline(-10, color=GRAY, ls="--", lw=0.7)
    ax.fill_between(t, -35, 10, where=(e1 < -10), color=NAVY, alpha=0.08, lw=0)
    ax.fill_between(t, -35, 10, where=(e2 < -10), color=GREEN, alpha=0.08, lw=0)
    p1 = np.mean(e1 < -10); p2 = np.mean(sel < -10)
    ax.text(t[-1], -33, f"time below $-10$ dB: one antenna {100*p1:.0f}%, selection {100*p2:.1f}%", ha="right", fontsize=7, color=ACCENT)
    ax.set_ylim(-35, 10); ax.set_xlim(0, t[-1]); ax.set_xlabel("time (s)"); ax.set_ylabel("envelope re. rms (dB)")
    ax.legend(fontsize=6.6, loc="upper right", ncol=3)
    ax.set_title(r"Two independently fading antennas rarely fade together ($f_D=10$ Hz)", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_diversity_branches")


def fig_numerology():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.set_xscale("log"); ax.set_yscale("log")
    bands = [(0.01, 0.2, "indoor", GREEN), (0.1, 0.3, "urban micro", "#2E86C1"), (0.5, 3, "urban macro", ORANGE), (10, 20, "hilly terrain", ACCENT), (50, 300, "SFN broadcast", PURPLE)]
    for lo, hi, n, c in bands:
        ax.axhspan(lo, hi, color=c, alpha=0.10)
        ax.text(560, np.sqrt(lo * hi), n, fontsize=6.6, color=c, va="center", ha="left")
    pts = [(15e3, 4.7, "LTE normal CP"), (15e3, 16.7, "LTE extended CP"), (30e3, 2.3, "NR 30 kHz"),
           (60e3, 1.2, "NR 60 kHz"), (120e3, 0.59, "NR 120 kHz"), (312.5e3, 0.8, "Wi-Fi 802.11a/n/ac"),
           (78.125e3, 3.2, "802.11ax (3.2 µs GI)"), (78.125e3, 0.8, "802.11ax (0.8 µs GI)"), (1116.0, 224.0, "DVB-T 8k, GI 1/4")]
    offs = {"NR 60 kHz": (-6, -8, "right"), "LTE normal CP": (-6, 2, "right"), "LTE extended CP": (-6, 2, "right"), "802.11ax (0.8 µs GI)": (-6, -9, "right"), "NR 120 kHz": (-4, -11, "center"),
            "Wi-Fi 802.11a/n/ac": (6, 2, "left"), "NR 30 kHz": (5, 3, "left")}
    for f, cp, n in pts:
        ax.plot(f, cp, "o", color=NAVY, ms=4.5)
        dx, dy, ha = offs.get(n, (5, 2, "left"))
        ax.annotate(n, (f, cp), (dx, dy), textcoords="offset points", fontsize=6.5, ha=ha)
    ax.set_xlim(500, 2.5e6); ax.set_ylim(0.008, 500)
    ax.set_xlabel("subcarrier spacing (Hz)  $\\rightarrow$ more Doppler tolerance")
    ax.set_ylabel(r"cyclic prefix / guard ($\mu$s)")
    ax.set_title("OFDM numerologies versus the delay spreads they must cover", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_numerology")


def fig_pilot_grid():
    r = rng(41)
    nsc, nsym = 72, 28
    fs = 15.36e6
    _, taps = cl.tdl_channel(np.ones(nsym * 1096), fs, "ETU", 300.0, rng=r, return_taps=True)
    Ht = np.fft.fft(taps[::1096][:nsym], 1024, axis=1)[:, :nsc].T
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.imshow(20 * np.log10(np.abs(Ht)), origin="lower", aspect="auto", cmap="Blues_r", vmin=-15, vmax=8,
              extent=[-0.5, nsym - 0.5, -0.5, nsc - 0.5], alpha=0.85)
    for s in range(nsym):
        sl = s % 7
        if sl in (0, 4):
            off = 0 if sl == 0 else 3
            for k in range(off, nsc, 6):
                ax.plot(s, k, "s", color=ACCENT, ms=2.2)
    ax.set_xlabel("OFDM symbol (time)  -- two 1 ms subframes"); ax.set_ylabel("subcarrier (15 kHz each)")
    ax.set_title("Pilots (red) sample the channel $|H(f,t)|$ (blue, ETU 300 Hz) like survey points on a map", fontsize=8.5)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch11_pilot_grid")


def fig_sounder():
    from scipy.signal import max_len_seq
    seq = 2.0 * max_len_seq(7)[0] - 1   # 127 chips
    r = rng(51)
    taps = {0: 1.0, 9: 0.6, 23: 0.35, 47: 0.2}
    rx = np.zeros(3 * 127)
    tx = np.tile(seq, 3)
    for d, a in taps.items():
        rx += a * np.roll(tx, d)
    rx += 0.3 * r.standard_normal(len(rx))
    corr = np.array([np.dot(rx[127:254], np.roll(tx, k)[127:254]) for k in range(127)]) / 127
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1))
    ax[0].step(np.arange(40), seq[:40], where="post", color=NAVY, lw=0.9); ax[0].set_ylim(-1.6, 1.6)
    ax[0].set_title("(a) transmitted PN chips", fontsize=8); ax[0].set_xlabel("chip")
    ac = np.array([np.dot(seq, np.roll(seq, k)) for k in range(127)]) / 127
    ax[1].plot(np.arange(-20, 21), np.roll(ac, 20)[:41], color=NAVY, marker=".", ms=3, lw=0.8)
    ax[1].set_title("(b) its autocorrelation: a spike", fontsize=8); ax[1].set_xlabel("lag (chips)")
    ml, sl, bl = ax[2].stem(np.arange(64), corr[:64], basefmt=" ", linefmt=ACCENT, markerfmt="o"); ml.set_markersize(2.5); sl.set_linewidth(0.8)
    for d, a in taps.items():
        ax[2].plot(d, a, "x", color="k", ms=5)
    ax[2].set_title("(c) correlate the echo: $h(\\tau)$ appears", fontsize=8); ax[2].set_xlabel("delay (chips)")
    for a in ax:
        a.tick_params(labelsize=6.5)
    fig.tight_layout(); save(fig, "ch11_sounder")


def fig_thorp():
    f = np.logspace(-1, 2.5, 300)
    a = 0.11 * f ** 2 / (1 + f ** 2) + 44 * f ** 2 / (4100 + f ** 2) + 2.75e-4 * f ** 2 + 0.003
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.loglog(f, a, color=NAVY)
    for fx in (10, 50):
        ax.plot(fx, np.interp(fx, f, a), "o", color=ACCENT, ms=4)
        ax.annotate(f"{fx} kHz: {np.interp(fx, f, a):.0f} dB/km", (fx, np.interp(fx, f, a)), (-60, 8), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("absorption (dB/km)")
    ax.set_title("sea-water absorption (Thorp)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_thorp")


def fig_v2x_train():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    # (a) train passing a trackside mast: Doppler swing
    v = 350 / 3.6; fc = 2.6e9; fD = v * fc / C0
    t = np.linspace(-6, 6, 600)
    for dmin, c in [(50, NAVY), (200, GREEN)]:
        x = v * t
        fd = fD * (-x) / np.sqrt(x ** 2 + dmin ** 2)
        ax[0].plot(t, fd, color=c, label=f"mast {dmin} m from track")
    ax[0].set_xlabel("time relative to passing the mast (s)"); ax[0].set_ylabel("Doppler shift (Hz)")
    ax[0].set_title("(a) high-speed train, 350 km/h, 2.6 GHz", fontsize=8); ax[0].legend(fontsize=6.4)
    # (b) V2V: convolution of two Clarke spectra
    r = rng(61)
    th1 = r.uniform(0, 2 * np.pi, 600_000); th2 = r.uniform(0, 2 * np.pi, 600_000)
    hst, e = np.histogram(np.cos(th1) + np.cos(th2), 120, range=(-2, 2), density=True)
    c_ = 0.5 * (e[1:] + e[:-1])
    hs1, _ = np.histogram(np.cos(th1), 120, range=(-2, 2), density=True)
    ax[1].plot(c_, hs1, color=GRAY, lw=1, label="one end moving")
    ax[1].plot(c_, hst, color=ACCENT, lw=1.3, label="both ends moving (V2V)")
    ax[1].set_xlabel("Doppler / $f_D$"); ax[1].set_ylim(0, 1.6); ax[1].set_yticks([])
    ax[1].set_title("(b) vehicle-to-vehicle Doppler spectrum", fontsize=8); ax[1].legend(fontsize=6.4, loc="upper right")
    fig.tight_layout(); save(fig, "ch11_v2x_train")


def fig_coherence_cartoon():
    """Coherence time and bandwidth read directly off a fading surface."""
    r = rng(71)
    fs = 15.36e6
    _, taps = cl.tdl_channel(np.ones(600 * 2048), fs, "EVA", 70.0, rng=r, return_taps=True)
    Ht = np.fft.fftshift(np.fft.fft(taps[::2048], 1024, axis=1), axes=1)
    P = 20 * np.log10(np.abs(Ht))
    t = np.arange(600) * 2048 / fs * 1e3; f = (np.arange(1024) - 512) * fs / 1024 / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax[0].plot(f, P[0], color=NAVY, lw=0.9); ax[0].set_xlabel("frequency (MHz)"); ax[0].set_ylabel("$|H|$ (dB)")
    ax[0].set_xlim(-4, 4)
    ax[0].annotate("", (0, 9), (0.56, 9), arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color=ACCENT, lw=1.5))
    ax[0].text(0.7, 8.5, "$B_c\\approx560$ kHz", fontsize=7, color=ACCENT)
    ax[0].set_title("(a) one instant, across frequency", fontsize=8); ax[0].set_ylim(-30, 13)
    ax[1].plot(t, P[:, 512], color=GREEN, lw=0.9); ax[1].set_xlabel("time (ms)")
    Tc = 0.423 / 70 * 1e3
    ax[1].annotate("", (2, 9), (2 + Tc, 9), arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color=ACCENT, lw=1.5))
    ax[1].text(2 + Tc + 0.5, 8.5, f"$T_c\\approx${Tc:.0f} ms", fontsize=7, color=ACCENT)
    ax[1].set_title("(b) one frequency, over time (70 Hz)", fontsize=8); ax[1].set_ylim(-30, 13)
    fig.tight_layout(); save(fig, "ch11_coherence_cartoon")


def fig_dish_beam():
    """The worked-example 60 cm dish at 12 GHz versus 1.2 GHz: uniform circular aperture pattern."""
    from scipy.special import j1
    th = np.linspace(-40, 40, 2000)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for f, c in [(12e9, NAVY), (1.2e9, ACCENT)]:
        lam = C0 / f; D = 0.6
        u = np.pi * D / lam * np.sin(np.deg2rad(th))
        p = (2 * j1(u) / np.where(u == 0, 1e-12, u)) ** 2
        p[np.abs(u) < 1e-9] = 1
        g = 10 * np.log10(0.65 * (np.pi * D / lam) ** 2)
        ax.plot(th, g + 10 * np.log10(p + 1e-9), color=c, label=f"{f/1e9:g} GHz: {g:.1f} dBi")
    ax.axvline(2, color=GREEN, lw=1, ls="--")
    ax.text(3, -16, "next satellite\n$2^\\circ$ away", fontsize=6.5, color=GREEN)
    ax.set_ylim(-20, 40); ax.set_xlim(-40, 40)
    ax.set_xlabel("angle off axis (deg)"); ax.set_ylabel("gain (dBi)")
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("one 60 cm dish, two frequencies", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_dish_beam")


def fig_fspl_lines():
    d = np.logspace(-2, 2, 200)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for f, c in [(0.9, GREEN), (2.4, NAVY), (5.8, ORANGE), (28, ACCENT)]:
        ax.semilogx(d, cl.fspl_db(d * 1e3, f * 1e9), color=c, label=f"{f:g} GHz")
    ax.invert_yaxis(); ax.set_xlabel("distance (km)"); ax.set_ylabel("free-space loss (dB)")
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("20 dB per decade, isotropic antennas", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_fspl_lines")


def fig_sens_stack():
    steps = [("$kT$ per Hz", -174.0, GRAY), ("+ bandwidth\n360 kHz", 55.6, NAVY), ("+ noise\nfigure", 2.0, ORANGE),
             ("+ required\nSINR", -4.0, ACCENT)]
    fig, ax = plt.subplots(figsize=(SW, SH))
    lvl = 0
    for i, (n, v, c) in enumerate(steps):
        if i == 0:
            ax.bar(i, 4, bottom=v - 2, color=c); lvl = v
        else:
            ax.bar(i, v, bottom=lvl, color=c, alpha=0.85); lvl += v
        ax.text(i, lvl + (3 if v >= 0 else -3), f"{lvl:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=6.6)
    ax.axhline(lvl, color=PURPLE, ls="--", lw=0.9); ax.text(3.4, lvl + 6, "sensitivity", color=PURPLE, fontsize=7, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps], fontsize=6.3)
    ax.set_ylim(-185, -105); ax.set_ylabel("dBm")
    ax.set_title("building the receiver sensitivity", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_sens_stack")


def fig_tworay_geom():
    fig, ax = plt.subplots(figsize=(SW, SH))
    ht, hr, d = 3.0, 1.2, 8.0
    ax.fill_between([-0.5, d + 0.5], -0.25, 0, color="#C8B48A")
    ax.plot([0, 0], [0, ht], color="k", lw=2); ax.plot([d, d], [0, hr], color="k", lw=2)
    ax.plot([0, d], [ht, hr], color=NAVY, lw=1.4)
    xr = d * ht / (ht + hr)
    ax.plot([0, xr, d], [ht, 0, hr], color=ACCENT, lw=1.4)
    ax.plot([0, 0], [0, -ht], color="k", lw=1, ls=":"); ax.plot([0, xr], [-ht, 0], color=ACCENT, lw=0.8, ls=":")
    ax.text(0.15, -ht + 0.2, "image\nantenna", fontsize=6.5, color=GRAY)
    ax.text(3.2, 2.35, "direct", fontsize=7, color=NAVY); ax.text(1.8, 0.6, "reflected", fontsize=7, color=ACCENT)
    ax.text(-0.45, ht / 2, "$h_t$", fontsize=8); ax.text(d + 0.12, hr / 2, "$h_r$", fontsize=8)
    ax.text(xr - 0.5, -0.65, r"$\Gamma\approx-1$", fontsize=7, color=ACCENT)
    ax.set_xlim(-0.8, d + 0.6); ax.set_ylim(-ht - 0.2, ht + 0.4); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(); save(fig, "ch11_tworay_geom")


def fig_pond():
    """Ripples from several stones dropped at once: interference of circular waves."""
    r = rng(81)
    n = 500; x = np.linspace(0, 10, n); X, Y = np.meshgrid(x, x)
    srcs = r.uniform(1, 9, (5, 2))
    E = np.zeros_like(X, complex)
    for (sx, sy) in srcs:
        R = np.hypot(X - sx, Y - sy) + 0.3
        E += np.exp(2j * np.pi * R / 0.8) / np.sqrt(R)
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.imshow(np.real(E), origin="lower", extent=[0, 10, 0, 10], cmap="Blues", vmin=-2.2, vmax=2.2)
    ax.plot(srcs[:, 0], srcs[:, 1], "o", color=ACCENT, ms=4)
    _clean(ax); ax.set_title("five stones, one pond", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_pond")


def fig_rms_bar():
    dl, p = cl.TDL_PROFILES["EVA"]
    m, s = rms_delay(dl, p)
    pw = 10 ** (np.array(p) / 10); pw /= pw.max()
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.vlines(np.array(dl) / 1e3, 0, pw, color=NAVY, lw=2.2); ax.plot(np.array(dl) / 1e3, pw, "o", color=NAVY, ms=3.5)
    ax.axvline(m * 1e6, color=ACCENT, lw=1)
    ax.annotate("", (m * 1e6 - s * 1e6, 1.08), (m * 1e6 + s * 1e6, 1.08), arrowprops=dict(arrowstyle="<->", color=ORANGE))
    ax.text(m * 1e6 + 0.05, 0.75, f"mean {m*1e9:.0f} ns", color=ACCENT, fontsize=7)
    ax.text(m * 1e6 + s * 1e6 + 0.05, 1.05, f"$\\pm\\tau_{{rms}}$ = {s*1e9:.0f} ns", color=ORANGE, fontsize=7)
    ax.set_ylim(0, 1.2); ax.set_xlabel(r"excess delay ($\mu$s)"); ax.set_ylabel("relative power (linear)")
    ax.set_title("EVA: centre of gravity and width", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_rms_bar")


def fig_lcr_cartoon():
    r = rng(91)
    fs = 2000.0; fd = 20.0; n = 1600
    g = cl.jakes_process(n, fd / fs, n_sin=32, rng=r)
    e = 20 * np.log10(np.abs(g)); t = np.arange(n) / fs * 1e3
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.plot(t, e, color=NAVY, lw=0.9); ax.axhline(-10, color=ACCENT, ls="--", lw=0.8)
    b = e < -10
    ax.fill_between(t, -35, 8, where=b, color=ACCENT, alpha=0.15, lw=0)
    idx = np.where(~b[:-1] & b[1:])[0]
    ax.plot(t[idx], e[idx], "v", color=ACCENT, ms=5)
    ax.text(t[-1], -33, f"{len(idx)} downward crossings", ha="right", fontsize=6.8, color=ACCENT)
    ax.set_ylim(-35, 8); ax.set_xlabel("time (ms)"); ax.set_ylabel("dB re. rms")
    ax.set_title("crossings (triangles), fades (shaded)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_lcr_cartoon")


def fig_sos_ring():
    r = rng(5)
    N = 12
    a = (2 * np.pi * np.arange(N) + r.uniform(-np.pi, np.pi, N)) / N
    fig, ax = plt.subplots(figsize=(SW, SH))
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GRAY, lw=0.6, ls=":")
    for ai in a:
        c = plt.cm.coolwarm(0.5 + 0.5 * np.cos(ai))
        ax.annotate("", (0.12 * np.cos(ai), 0.12 * np.sin(ai)), (np.cos(ai), np.sin(ai)), arrowprops=dict(arrowstyle="->", color=c, lw=1.1))
        ax.plot(np.cos(ai), np.sin(ai), "o", color=c, ms=5)
    ax.text(0, -1.35, f"{N} sinusoids, angles $\\alpha_n$ randomised\nwithin equal sectors (Zheng--Xiao)", ha="center", fontsize=6.8)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.65, 1.25); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(); save(fig, "ch11_sos_ring")


def fig_snr_pdf():
    g = np.linspace(-20, 60, 600)
    mean = 44.0
    lin = 10 ** (g / 10); m = 10 ** (mean / 10)
    pdf_db = lin / m * np.exp(-lin / m) * np.log(10) / 10
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.semilogy(g, pdf_db, color=NAVY)
    sel = g <= 0
    ax.fill_between(g[sel], 1e-9, pdf_db[sel], color=ACCENT, alpha=0.5)
    ax.text(-19, 3e-4, "below 0 dB:" + chr(10) + "where the" + chr(10) + "errors are", fontsize=6.5, color=ACCENT)
    ax.axvline(mean, color=GRAY, ls=":"); ax.text(mean - 1, 0.3, "mean 44 dB", fontsize=6.5, ha="right", color=GRAY)
    ax.set_xlabel("instantaneous SNR (dB)"); ax.set_ylabel("density (per dB)"); ax.set_ylim(1e-7, 1)
    ax.set_title("Rayleigh: SNR density, log scale", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_snr_pdf")


def fig_mrc_outage():
    x = np.linspace(-30, 10, 300); xl = 10 ** (x / 10)
    from scipy.stats import gamma as gam
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for L, c in [(1, NAVY), (2, GREEN), (4, ORANGE), (8, PURPLE)]:
        ax.semilogy(x, gam.cdf(xl, L, scale=1 / L), color=c, label=f"$L={L}$ branches (MRC)")
    ax.set_ylim(1e-6, 1); ax.set_xlim(-30, 8)
    ax.set_xlabel("combined SNR relative to its mean (dB)"); ax.set_ylabel("probability SNR is below")
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("Diversity hardens the channel: outage probability of $L$-branch MRC (equal total energy)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_mrc_outage")


def fig_ici():
    v = np.linspace(1, 500, 300)
    fig, ax = plt.subplots(figsize=(SW, SH))
    fc = 3.5e9
    for df, c in [(15e3, ACCENT), (30e3, NAVY), (120e3, GREEN)]:
        fd = v / 3.6 * fc / C0
        ici = (np.pi * fd / df) ** 2 / 3
        ax.semilogy(v, ici, color=c, label=f"{df/1e3:g} kHz")
    ax.set_xlabel("speed (km/h) at 3.5 GHz"); ax.set_ylabel("ICI power (rel. signal)")
    ax.legend(fontsize=6.5); ax.set_title(r"ICI $\approx(\pi f_DT_u)^2/3$ (approx.)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch11_ici")


def fig_summary_scales():
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    ax.set_xscale("log"); ax.set_xlim(5e-3, 2e5); ax.set_ylim(0, 3.4)
    bands = [(0.03, 0.5, 2.5, ACCENT, "multipath fading", "20--40 dB swings every $\\lambda/2$\nequalizers, OFDM, diversity, codes"),
             (10, 300, 1.5, ORANGE, "shadowing", "4--12 dB, log-normal\nmargins, overlap, handover"),
             (300, 5e4, 0.5, NAVY, "path loss", "$d^{-2}$ to $d^{-4}$\nlink budget, antennas, cell size")]
    for a, b, y, c, n, t in bands:
        ax.add_patch(plt.Rectangle((a, y - 0.2), b - a, 0.95, color=c, alpha=0.25, lw=0))
        ax.text(np.sqrt(a * b), y + 0.52, n, ha="center", fontsize=8, color=c, fontweight="bold")
        ax.text(np.sqrt(a * b), y + 0.18, t, ha="center", fontsize=6.3, color=c, va="center")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_xlabel("distance over which the effect changes (m), at about 2 GHz")
    ax.set_title("The chapter in one picture: three effects, three scales, three toolboxes", fontsize=9)
    fig.tight_layout(); save(fig, "ch11_summary_scales")


def fig_cdl_clusters():
    """A CDL-style channel: clusters with delay, power and angle of arrival (illustrative values)."""
    r = rng(101)
    fig = plt.figure(figsize=(W2, 2.5))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    cl_ = [(0.0, 0, 0), (35, -3, 0.3), (-60, -5, 0.7), (110, -8, 1.2), (-150, -12, 2.1), (170, -15, 3.0)]
    cols = [NAVY, GREEN, ORANGE, PURPLE, ACCENT, GRAY]
    b = fig.add_subplot(1, 2, 2)
    for (az, pdb, tau), c in zip(cl_, cols):
        rays = az + r.normal(0, 6, 20)
        pw = 10 ** (pdb / 10) * r.uniform(0.5, 1, 20) / 20
        ax.scatter(np.deg2rad(rays), 10 * np.log10(pw) + 35, s=10, color=c)
        b.vlines(tau + r.normal(0, 0.03, 20), 0, 10 * np.log10(pw) + 35, color=c, lw=0.8)
    ax.set_ylim(0, 25); ax.set_yticklabels([]); ax.set_title("(a) clusters by angle of arrival", fontsize=8.5, pad=12)
    ax.tick_params(labelsize=6.5)
    b.set_xlabel(r"delay (normalised to $\tau_{rms}$)"); b.set_ylabel("ray power (dB, rel.)")
    b.set_title("(b) the same rays by delay", fontsize=8.5); b.set_ylim(0, 25)
    fig.tight_layout(); save(fig, "ch11_cdl_clusters")


NEW = ["fig_cdl_clusters", "fig_summary_scales", "fig_dish_beam", "fig_fspl_lines", "fig_sens_stack", "fig_tworay_geom", "fig_pond", "fig_rms_bar",
       "fig_lcr_cartoon", "fig_sos_ring", "fig_snr_pdf", "fig_mrc_outage", "fig_ici",
       "fig_power_ladder", "fig_spray", "fig_wave_eh", "fig_pattern3d", "fig_polarization", "fig_lte_waterfall",
       "fig_wifi_rings", "fig_fresnel_refl", "fig_fresnel_football", "fig_interference_map", "fig_horizon",
       "fig_rain_fade_bar", "fig_exponent_ladder", "fig_los_prob", "fig_o2i_bars", "fig_hata_cells", "fig_shadow_map",
       "fig_shadow_hist", "fig_multipath_cartoon", "fig_speckle", "fig_doppler_geometry", "fig_phasor_sum",
       "fig_scattering_fn", "fig_model_timeline", "fig_diversity_branches", "fig_numerology", "fig_pilot_grid",
       "fig_sounder", "fig_thorp", "fig_v2x_train", "fig_coherence_cartoon"]

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        for nm in _sys.argv[1:]:
            globals()[nm if nm.startswith("fig_") else "fig_" + nm]()
    else:
        fig_antennas(); fig_aperture(); fig_knife_edge(); fig_two_ray(); fig_ionosphere(); fig_atmosphere()
        fig_pathloss_models(); fig_scales(); fig_coverage(); fig_pdp(); fig_tf_response(); fig_doppler()
        fig_envelope(); fig_distributions(); fig_lcr_afd(); fig_regimes(); fig_ber_fading()
        for nm in NEW:
            globals()[nm]()
