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


if __name__ == "__main__":
    fig_antennas(); fig_aperture(); fig_knife_edge(); fig_two_ray(); fig_ionosphere(); fig_atmosphere()
    fig_pathloss_models(); fig_scales(); fig_coverage(); fig_pdp(); fig_tf_response(); fig_doppler()
    fig_envelope(); fig_distributions(); fig_lcr_afd(); fig_regimes(); fig_ber_fading()
