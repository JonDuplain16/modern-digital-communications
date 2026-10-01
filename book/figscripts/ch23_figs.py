"""Figures for Chapter 23: Satellite Communications."""
from figstyle import *
from matplotlib.patches import RegularPolygon, Circle
import commlib as cl
from commlib import satellite as sat

KM = 1e3

# a few reference systems (altitude km) -- nominal values
SYSTEMS = [("ISS", 420), ("Starlink", 550), ("Iridium", 780), ("OneWeb", 1200),
           ("O3b", 8062), ("GPS", 20200), ("GEO", 35786)]


# ----------------------------------------------------------------- orbits
def fig_orbits_scale():
    h = np.logspace(np.log10(250), np.log10(40000), 400) * KM
    T = sat.orbital_period(sat.R_E + h) / 60
    v = sat.circular_speed(h) / KM
    rtt = sat.bent_pipe_rtt(h, 90) * 1e3
    rtt30 = sat.bent_pipe_rtt(h, 30) * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35))
    ax[0].loglog(h / KM, T, color=NAVY)
    ax[0].axhline(sat.SIDEREAL_DAY / 60, color=GRAY, ls=":", lw=0.9)
    ax[0].text(300, sat.SIDEREAL_DAY / 60 * 1.12, "sidereal day", fontsize=6.5, color=GRAY)
    ax[0].set_ylabel("period (min)"); ax[0].set_title("orbital period")
    ax[1].semilogx(h / KM, v, color=GREEN); ax[1].set_ylabel("speed (km/s)"); ax[1].set_title("orbital speed")
    ax[2].loglog(h / KM, rtt, color=ACCENT, label="both ends at zenith")
    ax[2].loglog(h / KM, rtt30, color=ACCENT, ls="--", label="both at 30$^\\circ$ elev.")
    ax[2].set_ylabel("round trip (ms)"); ax[2].set_title("bent-pipe round-trip time")
    ax[2].legend(fontsize=6.3, loc="upper left")
    for a, y in zip(ax, [T, v, rtt]):
        a.set_xlabel("altitude (km)"); a.set_xlim(250, 40000)
        for name, hs in SYSTEMS:
            yi = np.interp(np.log10(hs), np.log10(h / KM), y)
            a.plot(hs, yi, "o", ms=2.8, color="k")
    for name, hs in SYSTEMS[4:]:
        yi = np.interp(np.log10(hs), np.log10(h / KM), T)
        off = {"O3b": (-18, 5), "GPS": (-16, 5), "GEO": (-20, -9)}[name]
        ax[0].annotate(name, (hs, yi), off, textcoords="offset points", fontsize=6)
    ax[0].annotate("ISS, Starlink,\nIridium, OneWeb", (600, 97), (-8, 24), textcoords="offset points", fontsize=6,
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY))
    fig.tight_layout(); save(fig, "ch23_orbits_scale")


def fig_ground_tracks():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    # Starlink-like shell: 53 deg, 550 km, three orbits
    a = sat.R_E + 550 * KM; T = sat.orbital_period(a)
    t = np.linspace(0, 3 * T, 3000)
    lat, lon = sat.ground_track(a, 0.0, 53, 0, 0, 0, t)
    brk = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for i, (la, lo) in enumerate(zip(np.split(lat, brk), np.split(lon, brk))):
        ax.plot(lo, la, color=NAVY, lw=1.1, label="LEO 550 km, $i=53^\\circ$ (3 orbits)" if i == 0 else None)
    # Molniya: 12 h, e=0.74, i=63.4, perigee in the south
    a = sat.semi_major_axis(sat.SIDEREAL_DAY / 2)
    t = np.linspace(0, sat.SIDEREAL_DAY, 4000)
    lat, lon = sat.ground_track(a, 0.74, 63.4, 0, 270, 0, t, lon0_deg=0)
    brk = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for i, (la, lo) in enumerate(zip(np.split(lat, brk), np.split(lon, brk))):
        ax.plot(lo, la, color=ACCENT, lw=1.3, label="Molniya 12 h, $e=0.74$, $i=63.4^\\circ$" if i == 0 else None)
    # dots every hour on the Molniya track to show dwell near apogee
    th = np.arange(0, 24) * 3600.0
    la, lo = sat.ground_track(a, 0.74, 63.4, 0, 270, 0, th)
    ax.plot(lo, la, "o", ms=2.6, color=ACCENT)
    # inclined geosynchronous: figure of eight
    a = sat.geo_radius(); t = np.linspace(0, sat.SIDEREAL_DAY, 1000)
    lat, lon = sat.ground_track(a, 0.0, 8, 0, 0, 0, t, lon0_deg=110)
    ax.plot(lon, lat, color=GREEN, lw=1.6, label="geosynchronous, $i=8^\\circ$")
    ax.plot([-60], [0], "*", ms=9, color=ORANGE, label="geostationary ($i=0$): a fixed point")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
    ax.legend(fontsize=6.6, loc="lower left", ncol=2, framealpha=0.9)
    ax.set_title("Ground tracks (dots on the Molniya track are one hour apart)")
    fig.tight_layout(); save(fig, "ch23_ground_tracks")


def fig_geometry_curves():
    el = np.linspace(0, 90, 181)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    for (name, hk), c in zip([("LEO 550 km", 550), ("LEO 1200 km", 1200), ("MEO 8062 km", 8062), ("GEO", 35786)],
                             [NAVY, GREEN, ORANGE, ACCENT]):
        d = sat.slant_range(hk * KM, el)
        ax[0].semilogy(el, d / KM, color=c, label=name)
    ax[0].set_xlabel("elevation angle (deg)"); ax[0].set_ylabel("slant range (km)")
    sec = ax[0].secondary_yaxis("right", functions=(lambda x: x / 299.792458, lambda y: y * 299.792458))
    sec.set_ylabel("one-way delay (ms)")
    ax[0].set_ylim(150, 8e4)
    ax[0].legend(fontsize=6.3, loc="lower right", ncol=2); ax[0].set_title("slant range and delay")
    h = np.logspace(np.log10(300), np.log10(40000), 300) * KM
    for e_min, ls in [(10, "-"), (25, "--"), (40, ":")]:
        ax[1].semilogx(h / KM, 100 * sat.coverage_fraction(h, e_min), color=NAVY, ls=ls,
                       label=f"min. elevation {e_min}$^\\circ$")
    ax[1].set_xlabel("altitude (km)"); ax[1].set_ylabel("Earth surface covered (%)")
    ax[1].set_title("one satellite's coverage"); ax[1].legend(fontsize=6.6)
    fig.tight_layout(); save(fig, "ch23_geometry_curves")


def fig_geo_visibility():
    lon = np.linspace(-100, 100, 401); lat = np.linspace(-85, 85, 341)
    LO, LA = np.meshgrid(lon, lat)
    az, el, d = sat.geo_look_angles(LA, LO, 0.0)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    cs = ax[0].contourf(LO, LA, np.where(el > 0, el, np.nan), levels=np.arange(0, 91, 10), cmap="Blues")
    ax[0].contour(LO, LA, el, levels=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80], colors="k", linewidths=0.4)
    fig.colorbar(cs, ax=ax[0], label="elevation (deg)", shrink=0.9)
    ax[0].plot(0, 0, "*", color=ACCENT, ms=8)
    ax[0].set_xlabel("longitude relative to satellite (deg)"); ax[0].set_ylabel("latitude (deg)")
    ax[0].set_title("elevation to a GEO satellite")
    # right: elevation vs latitude on the satellite meridian and 40 deg off, plus azimuth
    la = np.linspace(0, 81, 300)
    for dl, c in [(0, NAVY), (30, GREEN), (60, ORANGE)]:
        a_, e_, d_ = sat.geo_look_angles(la, -dl, 0.0)
        ax[1].plot(la, e_, color=c, label=f"$\\Delta\\lambda={dl}^\\circ$")
    ax[1].axhline(0, color=GRAY, lw=0.8)
    ax[1].set_ylim(-5, 92); ax[1].set_xlabel("station latitude (deg)"); ax[1].set_ylabel("elevation (deg)")
    ax[1].set_title("elevation vs latitude"); ax[1].legend(fontsize=6.6)
    fig.tight_layout(); save(fig, "ch23_geo_visibility")


def fig_leo_doppler():
    f = 11.7e9
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.1), sharex=True)
    for me, c in [(90, NAVY), (50, GREEN), (25, ORANGE)]:
        p = sat.leo_pass(550 * KM, f, me, 10.0, 0.5)
        tm = p["t"] / 60
        ax[0, 0].plot(tm, p["elev"], color=c, label=f"max elevation {me}$^\\circ$")
        ax[0, 1].plot(tm, p["delay"] * 1e3, color=c)
        ax[1, 0].plot(tm, p["doppler"] / 1e3, color=c)
        ax[1, 1].plot(tm, p["doppler_rate"] / 1e3, color=c)
    ax[0, 0].set_ylabel("elevation (deg)"); ax[0, 0].legend(fontsize=6.6)
    ax[0, 1].set_ylabel("one-way delay (ms)")
    ax[1, 0].set_ylabel("Doppler shift (kHz)"); ax[1, 1].set_ylabel("Doppler rate (kHz/s)")
    for a in ax[1]:
        a.set_xlabel("time from closest approach (min)")
    fig.suptitle("A 550 km LEO pass at 11.7 GHz (elevation > 10$^\\circ$)", fontsize=10)
    fig.tight_layout(); save(fig, "ch23_leo_doppler")


# ----------------------------------------------------------------- propagation
def fig_rain():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    f = np.logspace(np.log10(2), np.log10(50), 300)
    for R, c in zip([5, 12.5, 25, 50, 100], [GRAY, GREEN, NAVY, ORANGE, ACCENT]):
        ax[0].loglog(f, sat.rain_specific_atten(f, R), color=c, label=f"{R:g} mm/h")
    for fb, nm in [(4, "C"), (12, "Ku"), (20, "Ka"), (40, "Q")]:
        ax[0].axvline(fb, color=GRAY, lw=0.5, ls=":")
        ax[0].text(fb * 1.03, 2e-4, nm, fontsize=7, color=GRAY)
    ax[0].set_xticks([2, 5, 10, 20, 50]); ax[0].set_xticklabels(["2", "5", "10", "20", "50"])
    ax[0].minorticks_off()
    ax[0].set_ylim(1e-4, 40); ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("specific attenuation (dB/km)")
    ax[0].set_title("rain: $\\gamma_R = kR^\\alpha$"); ax[0].legend(fontsize=6.4, loc="upper left", title="rain rate", title_fontsize=6.4)
    p = np.logspace(-3, 0.3, 200)
    for fg, c in [(12, NAVY), (20, GREEN), (30, ORANGE), (40, ACCENT)]:
        A = sat.rain_atten_p618(fg, 35, p, 42, lat_deg=45)
        ax[1].loglog(p, A, color=c, label=f"{fg} GHz")
    ax[1].invert_xaxis(); ax[1].set_xlabel("time percentage exceeded (%)")
    ax[1].set_ylabel("rain attenuation (dB)"); ax[1].set_ylim(0.05, 100)
    top = ax[1].secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top.set_xticks([1, 0.1, 0.01, 0.001]); top.set_xticklabels(["99", "99.9", "99.99", "99.999"], fontsize=7)
    top.set_xlabel("availability (%)", fontsize=8)
    ax[1].legend(fontsize=6.6, loc="lower right"); ax[1].set_title("slant path, 35$^\\circ$, $R_{0.01}=42$ mm/h", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_rain")


# ----------------------------------------------------------------- link budgets
def dth_budget():
    f = 11.7e9
    az, el, d = sat.geo_look_angles(48.0, 2.0, 19.2)        # Paris-ish to 19.2E
    G = sat.dish_gain_dbi(0.6, f, 0.65)
    tsys = sat.system_noise_temp(t_ant=35.0, t_rx=sat.nf_to_temp(0.7))
    L = sat.fspl_db(d, f)
    lb = sat.LinkBudget("GEO Ku DTH downlink")
    lb.add("Satellite EIRP (edge of coverage)", 51.0, "dBW")
    lb.add("Free-space loss", -L)
    lb.add("Gases, pointing, polarisation", -0.8)
    lb.add("Receive antenna gain", G, "dBi")
    lb.add("System noise temperature", -sat.db(tsys), "dBK")
    lb.add("Boltzmann's constant", -sat.K_BOLTZ_DB, "")
    return lb, dict(el=el, d=d, G=G, tsys=tsys, L=L)


def fig_dth_budget():
    lb, info = dth_budget()
    vals = [v for _, v, _ in lb.items]
    cum = np.cumsum([0] + vals)
    labels = ["EIRP\n51.0\ndBW", f"free\nspace\n{vals[1]:.1f}", "misc.\nlosses\n$-$0.8",
              f"60 cm\ndish\n+{vals[3]:.1f}", f"$T_s$ = \n{info['tsys']:.0f} K\n{vals[4]:.1f}", "$-k$\n\n+228.6"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[2.1, 1]))
    a = ax[0]
    for i, v in enumerate(vals):
        lo, hi = sorted([cum[i], cum[i + 1]])
        a.bar(i, hi - lo, bottom=lo, color=GREEN if v > 0 else ACCENT, width=0.62)
        a.plot([i + 0.31, i + 0.69], [cum[i + 1]] * 2, color=GRAY, lw=0.6)
    a.bar(len(vals), cum[-1], color=NAVY, width=0.62)
    a.text(len(vals), cum[-1] + 5, f"{cum[-1]:.1f}\ndB-Hz", ha="center", fontsize=7, color=NAVY)
    a.set_xticks(range(len(vals) + 1)); a.set_xticklabels(labels + ["$C/N_0$\n\n"], fontsize=6.6)
    a.set_ylabel("running total (dB)"); a.set_ylim(-170, 110)
    a.set_title("downlink $C/N_0$ build-up", fontsize=9)
    # right: combining contributions
    cn_down = cum[-1] - 10 * np.log10(27.5e6)
    parts = [("uplink", 25.0), ("downlink", cn_down), ("adjacent\nsatellites", 21.0), ("cross-pol.", 22.0),
             ("intermod.", 20.0)]
    tot = sat.combine_cn_db(*[v for _, v in parts])
    b = ax[1]
    names = [p[0] for p in parts] + ["total"]
    vv = [p[1] for p in parts] + [tot]
    b.barh(range(len(vv)), vv, color=[GRAY] * len(parts) + [NAVY], height=0.6)
    for i, v in enumerate(vv):
        b.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7)
    b.set_yticks(range(len(vv))); b.set_yticklabels(names, fontsize=6.8); b.invert_yaxis()
    b.set_xlim(0, 30); b.set_xlabel("$C/N$ or $C/I$ in 27.5 MHz (dB)")
    b.set_title("combining impairments", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_dth_budget")
    return lb, info, cn_down, tot


def fig_deep_space():
    """Supportable data rate vs distance for a fixed X-band link (Voyager-like parameters)."""
    au = np.logspace(-1, 2.5, 200)
    d = au * sat.AU
    f = 8.42e9
    eirp = 10 * np.log10(20) + sat.dish_gain_dbi(3.7, f, 0.55)
    gt70 = sat.dish_gain_dbi(70, f, 0.70) - sat.db(20.0)
    gt34 = sat.dish_gain_dbi(34, f, 0.70) - sat.db(25.0)
    ebn0_req = 2.5  # dB, concatenated coding plus a little implementation margin (assumed)
    margin = 3.0 + 4.0  # 3 dB link margin + ~4 dB modulation, circuit and pointing losses (assumed)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for gt, lab, c in [(gt70, "70 m antenna ($T_s$ = 20 K)", NAVY), (gt34, "34 m antenna ($T_s$ = 25 K)", GREEN)]:
        cn0 = sat.cn0_dbhz(eirp, sat.fspl_db(d, f), gt)
        R = 10 ** ((cn0 - ebn0_req - margin) / 10)
        ax.loglog(au, R, color=c, label=lab)
    for nm, x in [("Mars (max)", 2.67), ("Jupiter", 5.2), ("Saturn", 9.5), ("Neptune", 30), ("Voyager 2 (2026)", 141),
                  ("Voyager 1 (2026)", 173)]:
        ax.axvline(x, color=GRAY, lw=0.5, ls=":")
        ax.text(x * 1.04, 3e5 if "Voy" not in nm else (1.2e5 if "1" in nm else 3e4), nm, fontsize=6, rotation=90, color=GRAY, va="bottom")
    ax.set_xlabel("distance (AU)"); ax.set_ylabel("supportable bit rate (b/s)")
    ax.set_ylim(10, 1e8); ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("A 20 W, 3.7 m, X-band spacecraft link: rate falls as $1/d^2$", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_deep_space")


# ----------------------------------------------------------------- antennas
def fig_dish_pattern():
    th = np.linspace(0.01, 10, 2000)
    f = 14.25e9
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    for D, c in [(0.6, NAVY), (1.2, GREEN), (2.4, ORANGE)]:
        G0 = sat.dish_gain_dbi(D, f, 0.65)
        ax[0].plot(th, G0 + sat.dish_pattern_db(th, D, f, taper=1), color=c, lw=1.0,
                   label=f"{D} m ({G0:.1f} dBi, $\\theta_{{3dB}}\\approx${sat.dish_beamwidth_deg(D, f):.1f}$^\\circ$)")
    tm = np.linspace(1, 10, 100)
    ax[0].plot(tm, sat.itu_s580_mask_dbi(tm), color=ACCENT, lw=1.8, ls="--", label="$29-25\\log_{10}\\theta$ dBi")
    ax[0].axvline(2, color=GRAY, lw=0.7, ls=":"); ax[0].text(2.1, 44, "next GEO\nslot (2$^\\circ$)", fontsize=6.5, color=GRAY)
    ax[0].set_ylim(-15, 52); ax[0].set_xlim(0, 10)
    ax[0].set_xlabel("off-axis angle (deg)"); ax[0].set_ylabel("gain (dBi)")
    ax[0].legend(fontsize=5.9, loc="upper right"); ax[0].set_title("reflector patterns at 14.25 GHz", fontsize=9)
    # phased array: scan loss and patterns
    N = 32; d = 0.5
    u = np.linspace(-1, 1, 4001)
    for sc, c in [(0, NAVY), (30, GREEN), (55, ORANGE)]:
        s0 = np.sin(np.radians(sc))
        psi = 2 * np.pi * d * (u - s0)
        af = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2) + 1e-12))
        af[np.isclose(np.sin(psi / 2), 0)] = 1
        elem = np.sqrt(np.clip(1 - u ** 2, 0, None)) ** 1.3
        g = 20 * np.log10(af * elem + 1e-9)
        ax[1].plot(np.degrees(np.arcsin(u)), g, color=c, lw=0.9, label=f"scan {sc}$^\\circ$")
    ax[1].set_ylim(-40, 2); ax[1].set_xlim(-90, 90); ax[1].set_xticks(range(-90, 91, 30))
    ax[1].set_xlabel("angle from broadside (deg)"); ax[1].set_ylabel("relative gain (dB)")
    ax[1].legend(fontsize=6.4, loc="lower left"); ax[1].set_title("32-element array, cos$^{1.3}$ element", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_antennas")


def fig_beam_reuse():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    cols = {(0, 0): NAVY, (1, 0): GREEN, (0, 1): ORANGE, (1, 1): ACCENT}
    labs = {(0, 0): "f1 RHCP", (1, 0): "f2 RHCP", (0, 1): "f1 LHCP", (1, 1): "f2 LHCP"}
    R = 1.0
    a = ax[0]
    for q in range(-4, 5):
        for r in range(-3, 4):
            x = np.sqrt(3) * R * (q + r / 2); y = 1.5 * R * r
            if abs(x) > 6.5 or abs(y) > 4.6:
                continue
            key = (q % 2, r % 2)
            a.add_patch(RegularPolygon((x, y), 6, radius=R * 0.98, orientation=0, facecolor=cols[key], alpha=0.55,
                                       edgecolor="white", lw=1))
            a.add_patch(Circle((x, y), R * 1.08, fill=False, lw=0.4, color="k", alpha=0.35))
    a.set_xlim(-6.6, 6.6); a.set_ylim(-4.9, 4.9); a.set_aspect("equal"); a.axis("off")
    for i, k in enumerate(cols):
        a.add_patch(plt.Rectangle((-6.4 + 3.3 * i, -5.9), 0.5, 0.5, color=cols[k], alpha=0.6, clip_on=False))
        a.text(-5.8 + 3.3 * i, -5.65, labs[k], fontsize=6.6, va="center")
    a.set_title("4-colour reuse: 2 bands $\\times$ 2 polarisations", fontsize=9)
    # right: beam cuts along a row, same colour 2 spacings apart
    from scipy.special import jv
    x = np.linspace(-3.2, 3.2, 1200)
    s = np.sqrt(3)  # beam spacing along a row (units of R)
    def beam(x0):
        u = 3.2 * (x - x0) / s * 1.2
        u = np.where(np.abs(u) < 1e-9, 1e-9, u)
        return 20 * np.log10(np.abs(8 * jv(2, u) / u ** 2) + 1e-9)
    for k, c in zip(range(-2, 3), [NAVY, GREEN, NAVY, GREEN, NAVY]):
        ax[1].plot(x / s, beam(k * s), color=c, lw=1.2 if k == 0 else 0.9, ls="-" if k == 0 else "--")
    ax[1].axvline(0.5, color=GRAY, lw=0.6, ls=":"); ax[1].text(0.55, -39, "beam edge", fontsize=6.5, color=GRAY, rotation=90, va="bottom")
    ax[1].set_ylim(-40, 2); ax[1].set_xlim(-1.8, 1.8)
    ax[1].set_xlabel("position (beam spacings)"); ax[1].set_ylabel("relative gain (dB)")
    ax[1].set_title("co-channel beams are two spacings apart", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_beam_reuse")


# ----------------------------------------------------------------- payload nonlinearity
def fig_twta():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    r = np.logspace(-2, 0.6, 400)
    rs = 1 / np.sqrt(sat.SALEH_TWTA["ba"])
    A, P = sat.saleh_am_am_pm(r)
    As = sat.saleh_am_am_pm(rs)[0]
    pin = 20 * np.log10(r / rs); pout = 20 * np.log10(A / As)
    ax[0].plot(pin, pout, color=NAVY, label="TWTA (Saleh model)")
    ax[0].plot(pin, pin + 20 * np.log10(sat.saleh_am_am_pm(1e-4)[0] / 1e-4 * rs / As), color=GRAY, ls=":", lw=0.9, label="linear extrapolation")
    xr = 2 * r / rs          # same small-signal gain (6 dB) as the TWTA
    ssp = 20 * np.log10(np.abs(cl.rapp_pa(xr.astype(complex), sat=1.0, p=3.0)))
    ax[0].plot(pin, ssp, color=GREEN, ls="--", label="SSPA (Rapp, $p=3$)")
    ax[0].set_xlim(-20, 8); ax[0].set_ylim(-20, 3)
    ax[0].set_xlabel("input power rel. saturation (dB)"); ax[0].set_ylabel("output power rel. saturation (dB)")
    ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_title("AM/AM", fontsize=9)
    ax[1].plot(pin, np.degrees(P), color=ACCENT)
    ax[1].set_xlim(-20, 8); ax[1].set_xlabel("input power rel. saturation (dB)"); ax[1].set_ylabel("phase shift (deg)")
    ax[1].set_title("AM/PM (TWTA)", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_twta")


def _td_curve(pts, thr_db, ibos, rng_, linear_rx=True, N=12000, sps=8, beta=0.2):
    taps = cl.rrc_taps(beta, sps, 12)
    idx = rng_.integers(0, len(pts), N); s = pts[idx]
    out = []
    for ibo in ibos:
        y = sat.saleh_twta(cl.shape(s, taps, sps), ibo)
        obo = -10 * np.log10(np.mean(np.abs(y) ** 2))
        z = cl.matched_filter(y, taps)[len(taps) - 1::sps][:N]
        sl = slice(50, -50)
        if linear_rx:
            g = np.vdot(s[sl], z[sl]) / np.vdot(s[sl], s[sl])
            e = z[sl] - g * s[sl]; sig = np.abs(g) ** 2 * np.mean(np.abs(s) ** 2)
        else:
            cent = np.array([z[sl][idx[sl] == k].mean() for k in range(len(pts))])
            e = z[sl] - cent[idx[sl]]; sig = np.mean(np.abs(cent[idx[sl]]) ** 2)
        mer = sig / np.mean(np.abs(e) ** 2); thr = 10 ** (thr_db / 10)
        req = 1 / (1 / thr - 1 / mer) if mer > thr else np.inf
        out.append((obo, obo + 10 * np.log10(req / thr)))
    return np.array(out)


def fig_apsk_vs_qam():
    r = rng(5)
    q = cl.get_constellation("16qam").points; q = q / np.sqrt(np.mean(np.abs(q) ** 2))
    a = sat.dvbs2_16apsk("3/4")
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    for axx, pts, nm in [(ax[0], q, "16-QAM"), (ax[1], a, "16-APSK (4+12)")]:
        y = sat.saleh_twta(pts, 0.0)          # symbol-level, IBO 0 dB (mean power)
        y = y / np.sqrt(np.mean(np.abs(y) ** 2))
        axx.plot(pts.real, pts.imag, "o", ms=4, mfc="none", color=GRAY, label="input")
        axx.plot(y.real, y.imag, "o", ms=3.2, color=NAVY if "QAM" in nm else GREEN, label="TWTA output")
        for p0, p1 in zip(pts, y):
            axx.plot([p0.real, p1.real], [p0.imag, p1.imag], color=GRAY, lw=0.4)
        axx.set_aspect("equal"); axx.set_xlim(-1.6, 1.6); axx.set_ylim(-1.6, 1.6)
        axx.set_title(nm, fontsize=9); axx.tick_params(labelsize=7)
        papr = 10 * np.log10(np.max(np.abs(pts) ** 2) / np.mean(np.abs(pts) ** 2))
        axx.text(0, -1.5, f"symbol PAPR {papr:.2f} dB", ha="center", fontsize=6.5)
    ax[0].legend(fontsize=5.8, loc="upper left", framealpha=0.9)
    ibos = np.arange(-2, 10.1, 0.5)
    for pts, nm, c in [(q, "16-QAM", NAVY), (a, "16-APSK", GREEN)]:
        for lin, ls in [(True, "-"), (False, "--")]:
            td = _td_curve(pts, 10.2, ibos, r, linear_rx=lin)
            ax[2].plot(td[:, 0], td[:, 1], color=c, ls=ls,
                       label=f"{nm}, {'gain/phase rx' if lin else 'centroid rx'}")
    ax[2].set_xlabel("output back-off (dB)"); ax[2].set_ylabel("total degradation (dB)")
    ax[2].set_ylim(1, 6); ax[2].set_xlim(0.6, 5.5)
    ax[2].legend(fontsize=5.6, loc="upper right"); ax[2].set_title("RRC 0.2, rate-3/4 threshold", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_apsk_vs_qam")


def fig_intermod():
    r = rng(7)
    fs = 64.0; N = 1 << 16
    t = np.arange(N) / fs
    # 8 carriers of RRC QPSK, spaced 3 units, symbol rate 2
    sps = 32; taps = cl.rrc_taps(0.25, sps, 10)
    x = np.zeros(N, complex)
    offs = (np.arange(8) - 3.5) * 3.0
    offs = offs[offs != offs[5]]  # leave one slot empty to reveal IM products
    for fo in offs:
        s = (r.choice([-1, 1], N // sps + 20) + 1j * r.choice([-1, 1], N // sps + 20)) / np.sqrt(2)
        b = cl.shape(s, taps, sps)[:N]
        x += b * np.exp(2j * np.pi * fo * t / 1.0 * (1 / 1))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    from scipy.signal import welch
    for ibo, c in [(10, GREEN), (3, ORANGE), (0, ACCENT)]:
        y = sat.saleh_twta(x, ibo)
        f, p = welch(y, fs=fs, nperseg=2048, return_onesided=False)
        f = np.fft.fftshift(f); p = np.fft.fftshift(p)
        ax[0].plot(f, 10 * np.log10(p / p.max()), color=c, lw=0.8, label=f"IBO {ibo} dB")
    ax[0].set_xlim(-22, 22); ax[0].set_ylim(-60, 3)
    ax[0].set_xlabel("frequency (arb. units)"); ax[0].set_ylabel("PSD (dB rel. peak)")
    ax[0].legend(fontsize=6.5, loc="lower center"); ax[0].set_title("7 carriers, one slot left empty", fontsize=9)
    ax[0].annotate("IM products fill\nthe empty slot", (offs[4] + 3.0, -30), (8, -12), fontsize=6.5,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    # C/IM and output power vs IBO (C/IM measured in the empty slot)
    ibos = np.arange(-2, 16.1, 1.0)
    cim, obo = [], []
    fe = offs[4] + 3.0
    for ibo in ibos:
        y = sat.saleh_twta(x, ibo)
        f, p = welch(y, fs=fs, nperseg=2048, return_onesided=False)
        inslot = np.abs(f - fe) < 1.0
        oncar = np.abs(f - offs[4]) < 1.0
        cim.append(10 * np.log10(p[oncar].mean() / p[inslot].mean()))
        obo.append(-10 * np.log10(np.mean(np.abs(y) ** 2)))
    ax[1].plot(ibos, cim, color=NAVY, label="$C/I_{IM}$ (dB)")
    ax[1].plot(ibos, obo, color=ACCENT, label="output back-off (dB)")
    ax[1].set_xlabel("input back-off (dB)"); ax[1].set_ylabel("dB")
    ax[1].legend(fontsize=6.6); ax[1].set_title("the back-off trade", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_intermod")


# ----------------------------------------------------------------- DVB-S2
def fig_dvbs2_modcods():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    s = np.linspace(-4, 20, 300)
    ax.plot(s, np.log2(1 + 10 ** (s / 10)), color=GRAY, lw=1.2, label="Shannon: $\\log_2(1+E_s/N_0)$")
    mk = {"QPSK": ("o", NAVY), "8PSK": ("s", GREEN), "16APSK": ("^", ORANGE), "32APSK": ("D", ACCENT)}
    for m, (sym, c) in mk.items():
        pts = [(thr, eta, nm) for nm, eta, thr in sat.DVBS2_MODCODS if nm.split()[0] == m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], sym, color=c, ms=4, label=m)
    for nm, eta, thr in sat.DVBS2_MODCODS:
        if nm in ("QPSK 1/4", "QPSK 1/2", "QPSK 9/10", "8PSK 3/5", "8PSK 9/10", "16APSK 2/3", "32APSK 9/10"):
            ax.annotate(nm.split()[1], (thr, eta), (4, -6), textcoords="offset points", fontsize=6)
    # ACM staircase
    es = np.linspace(-3, 18, 800)
    stair = [acm[1] if (acm := sat.acm_select(e, 0.0)) else 0 for e in es]
    ax.plot(es, stair, color=NAVY, lw=0.8, alpha=0.6, drawstyle="steps-post", label="ACM: best MODCOD that closes")
    ax.set_xlabel("required $E_s/N_0$ (dB), AWGN, PER $10^{-7}$"); ax.set_ylabel("spectral efficiency (b/s/Hz)")
    ax.set_xlim(-4, 18); ax.set_ylim(0, 5.2); ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("DVB-S2 MODCODs (after ETSI EN 302 307-1)", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_dvbs2_modcods")


# ----------------------------------------------------------------- protocols
def fig_tcp_delay():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    rtt = np.logspace(np.log10(5e-3), np.log10(1.0), 200)
    for W, c, lab in [(64 * 1024, ACCENT, "64 KiB (no window scaling)"), (1 << 20, ORANGE, "1 MiB"), (16 << 20, NAVY, "16 MiB")]:
        ax[0].loglog(rtt * 1e3, sat.tcp_window_limited_rate(W, rtt) / 1e6, color=c, label=lab)
    marks = [("fibre, 1000 km", 0.012), ("LEO", 0.035), ("MEO", 0.15), ("GEO", 0.6)]
    for nm, x in marks:
        for a in ax:
            a.axvline(x * 1e3, color=GRAY, lw=0.5, ls=":")
        ax[0].text(x * 1e3 * 1.06, 0.25, nm, fontsize=6.3, rotation=90, color=GRAY)
    ax[0].set_xlabel("round-trip time (ms)"); ax[0].set_ylabel("max. throughput (Mb/s)")
    ax[0].legend(fontsize=6.3, loc="upper right", title="TCP window", title_fontsize=6.3); ax[0].set_title("window-limited throughput", fontsize=9)
    # transfer time for objects with slow start (IW=10 MSS), bandwidth unconstrained
    mss = 1460; iw = 10 * mss
    for size, c in [(100e3, GREEN), (1e6, NAVY), (10e6, ACCENT)]:
        nrt = 1 + 1 + np.ceil(np.log2(1 + size / iw))   # TCP handshake + request + slow-start rounds
        ax[1].loglog(rtt * 1e3, nrt * rtt, color=c, label=f"{size/1e6:g} MB object")
    ax[1].set_xlabel("round-trip time (ms)"); ax[1].set_ylabel("transfer time (s)")
    ax[1].legend(fontsize=6.3, loc="upper left"); ax[1].set_title("slow start: time is RTTs, not bits", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_tcp_delay")


# ================================================================= second-edition illustrations
from matplotlib.patches import Wedge, Polygon, FancyArrowPatch, Rectangle, Ellipse, FancyBboxPatch
from matplotlib.patches import Arc
SKY = "#2E86C1"
EARTH = "#5D8AA8"


def _clean(ax):
    ax.set_aspect("equal"); ax.axis("off")


def fig_by_numbers():
    """'Satellite communication by the numbers' tiles for the chapter opening."""
    tiles = [("35 786 km", "altitude of the\ngeostationary ring"),
             ("0.5 s", "round trip through\none GEO satellite"),
             ("205 dB", "spreading loss on a\nKu-band GEO downlink"),
             ("23 ppm", "LEO Doppler: 273 kHz\nat 11.7 GHz"),
             ("42 %", "of the Earth seen by\none GEO satellite"),
             ("~8 min", "longest LEO pass\nabove 10$^\\circ$ at 550 km"),
             ("173 AU", "Voyager 1 in late 2026:\none light-day away"),
             ("10$^{-19}$ W", "power received\nfrom Voyager")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, SKY, ACCENT, ORANGE, GREEN, PURPLE, NAVY, GRAY]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.62, big, ha="center", va="center", fontsize=13.5, color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=7.2, color="#333333", linespacing=1.1)
    save(fig, "ch23_by_numbers")


def fig_altitude_scale():
    """Orbits drawn to scale around the Earth, with a zoom on low Earth orbit."""
    Re = 6371.0
    fig = plt.figure(figsize=(W1, 3.3))
    ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
    th = np.linspace(0, np.pi / 2, 300)
    ax.add_patch(Wedge((0, 0), 1.0, 0, 90, fc=EARTH, ec=NAVY, lw=0.8))
    rings = [("LEO (Starlink 550 km)", 550, NAVY), ("O3b MEO 8062 km", 8062, GREEN),
             ("GPS 20 200 km", 20200, ORANGE), ("GEO 35 786 km", 35786, ACCENT)]
    for nm, h, c in rings:
        rr = (Re + h) / Re
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=c, lw=1.3 if h > 1000 else 0.9)
    lab_ang = {"O3b MEO 8062 km": 62, "GPS 20 200 km": 50, "GEO 35 786 km": 38}
    for nm, h, c in rings[1:]:
        rr = (Re + h) / Re; a = np.radians(lab_ang[nm])
        ax.plot(rr * np.cos(a), rr * np.sin(a), "o", color=c, ms=5)
        ax.text(rr * np.cos(a) + 0.15, rr * np.sin(a) + 0.12, nm, fontsize=7.5, color=c)
    ax.text(0.12, 0.3, "Earth", color="white", fontsize=7, weight="bold")
    ax.annotate("LEO hugs the\nEarth: 550 km is\n9% of its radius", (np.cos(1.35) * 1.09, np.sin(1.35) * 1.09), (0.05, 3.0),
                fontsize=7, color=NAVY, arrowprops=dict(arrowstyle="->", lw=0.6, color=NAVY))
    ax.set_xlim(-0.1, 7.2); ax.set_ylim(-0.1, 7.2); _clean(ax)
    ax.text(0.05, 7.05, "drawn to scale", fontsize=7.5, color=GRAY, style="italic")
    ax.text(4.35, 0.15, "The Moon would be\n60 Earth radii away:\nnine times farther\nthan GEO.", fontsize=6.8, color=GRAY, va="bottom")
    # zoom: low orbit
    b = fig.add_axes([0.66, 0.1, 0.33, 0.85])
    b.add_patch(Rectangle((0, -300), 1, 300, fc=EARTH, ec="none"))
    items = [("ISS", 420), ("Starlink", 550), ("Iridium", 780), ("OneWeb", 1200)]
    for k, (nm, h) in enumerate(items):
        b.axhline(h, color=NAVY, lw=0.7, ls="--", alpha=0.6)
        b.plot(0.18 + 0.2 * k, h, "s", color=NAVY, ms=4.5)
        b.text(0.25 + 0.2 * k, h + 30, f"{nm}\n{h} km", fontsize=6.4, color=NAVY)
    b.axhspan(0, 100, color=SKY, alpha=0.18); b.text(0.03, 30, "atmosphere (to ~100 km)", fontsize=6.2, color=NAVY)
    b.axhspan(100, 400, color=GRAY, alpha=0.10); b.text(0.03, 230, "drag pulls\nsatellites down", fontsize=6.2, color=GRAY)
    b.set_xlim(0, 1); b.set_ylim(-300, 1500); b.set_xticks([])
    b.set_ylabel("altitude (km)", fontsize=8); b.tick_params(labelsize=7)
    b.set_title("zoom: low Earth orbit", fontsize=8.5); b.grid(False)
    save(fig, "ch23_altitude_scale")


def fig_clarke_three():
    """Left: Clarke's three geostationary relays. Right: today's crowded arc and a small dish's beam."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.9), gridspec_kw=dict(width_ratios=[1, 1.05]))
    a = ax[0]
    rg = 42164 / 6378
    a.add_patch(Circle((0, 0), 1, fc=EARTH, ec=NAVY))
    a.add_patch(Circle((0, 0), rg, fc="none", ec=GRAY, ls=":", lw=0.8))
    for k, c in zip(range(3), [NAVY, GREEN, ORANGE]):
        ang = np.radians(90 + 120 * k)
        S = rg * np.array([np.cos(ang), np.sin(ang)])
        half = np.arcsin(1 / rg)
        for s in (-1, 1):
            # tangent points
            phi = ang + s * (np.pi / 2 - half)
            T = np.array([np.cos(phi), np.sin(phi)])
            a.plot([S[0], T[0]], [S[1], T[1]], color=c, lw=0.8)
        p1 = ang - (np.pi / 2 - half); p2 = ang + (np.pi / 2 - half)
        a.add_patch(Polygon([S, [np.cos(p1), np.sin(p1)], [np.cos(p2), np.sin(p2)]], fc=c, alpha=0.12, ec="none"))
        arc = np.linspace(p1, p2, 60)
        a.plot(1.12 * np.cos(arc), 1.12 * np.sin(arc), color=c, lw=2.2, alpha=0.9, solid_capstyle="butt")
        a.plot(*S, "o", color=c, ms=6)
    a.text(0, -0.15, "north\npole", ha="center", fontsize=6.5, color="white")
    a.set_xlim(-rg - 0.6, rg + 0.6); a.set_ylim(-rg - 0.6, rg + 0.6); _clean(a)
    a.set_title("Clarke, 1945: three relays\ncover all but the poles", fontsize=8.5)
    b = ax[1]
    # arc of GEO seen from an earth station: satellites 2 deg apart and a 3 deg beam
    b.set_xlim(-7, 7); b.set_ylim(-0.6, 9.2); b.axis("off")
    xs = np.arange(-6, 6.1, 2.0)
    for x in xs:
        b.add_patch(Rectangle((x - 0.25, 7.9), 0.5, 0.35, fc=NAVY if x == 0 else GRAY, ec="none"))
        b.plot([x - 0.6, x + 0.6], [8.07, 8.07], color=NAVY if x == 0 else GRAY, lw=2.2)
    b.text(0, 8.75, "GEO arc, satellites 2$^\\circ$ apart", ha="center", fontsize=7.5)
    b.add_patch(Polygon([[0, 0.3], [-2.7, 7.9], [2.7, 7.9]], fc=ACCENT, alpha=0.10, ec="none"))
    b.add_patch(Polygon([[0, 0.3], [-1.5, 7.9], [1.5, 7.9]], fc=ACCENT, alpha=0.25, ec="none"))
    b.add_patch(Ellipse((0, 0.3), 1.2, 0.45, angle=0, fc=GRAY, ec=NAVY))
    b.text(0.8, 0.05, "60 cm dish", fontsize=7)
    b.annotate("the beam's skirts\nreach the\nneighbours", (1.75, 5.6), (2.8, 3.4), fontsize=7, color=ACCENT,
               arrowprops=dict(arrowstyle="->", lw=0.6, color=ACCENT))
    b.set_title("today: one crowded ring", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_clarke_three")


def fig_kepler():
    """Kepler's second law on a Molniya-like ellipse: dots at equal time steps."""
    e = 0.74; a = 1.0; b_ = a * np.sqrt(1 - e ** 2)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    E = np.linspace(0, 2 * np.pi, 400)
    ax.plot(a * np.cos(E), b_ * np.sin(E), color=NAVY, lw=1)
    ax.add_patch(Circle((a * e, 0), 0.12, fc=EARTH, ec=NAVY, zorder=3))
    M = np.linspace(0, 2 * np.pi, 13)[:-1]
    from commlib.satellite import kepler_E
    Ek = np.array([kepler_E(m, e) for m in M])
    x = a * np.cos(Ek); y = b_ * np.sin(Ek)
    for k in range(len(M)):
        k2 = (k + 1) % len(M)
        if k in (0, 6):
            Es = np.linspace(Ek[k], Ek[k2] if k2 else 2 * np.pi, 30)
            poly = [[a * e, 0]] + [[a * np.cos(t), b_ * np.sin(t)] for t in Es]
            ax.add_patch(Polygon(poly, fc=ORANGE if k == 0 else GREEN, alpha=0.35, ec="none"))
    ax.plot(x, y, "o", color=ACCENT, ms=3.5, zorder=4)
    ax.text(a * e + 0.05, -0.2, "Earth", fontsize=6.5, ha="center")
    ax.text(-1.0, 0.0, "apogee:\nslow", fontsize=6.5, ha="right", va="center", color=GREEN)
    ax.text(1.03, 0.1, "perigee:\nfast", fontsize=6.5, ha="left", color=ORANGE)
    ax.text(-0.1, -0.88, "dots: equal time steps; shaded\nsectors: equal areas", fontsize=6.3, ha="center", color=GRAY)
    ax.set_xlim(-1.6, 1.5); ax.set_ylim(-1.0, 0.8); _clean(ax)
    save(fig, "ch23_kepler")


def fig_constellation_shell():
    """A Walker-delta shell (72 planes x 22, 53 deg, 550 km) and its latitude distribution."""
    P, S, inc, F = 72, 22, np.radians(53), 1
    rr = (6378 + 550) / 6378
    raan = np.repeat(np.arange(P) * 2 * np.pi / P, S)
    u = np.tile(np.arange(S) * 2 * np.pi / S, P) + np.repeat(np.arange(P), S) * F * 2 * np.pi / (P * S)
    x = np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * np.cos(inc)
    y = np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * np.cos(inc)
    z = np.sin(u) * np.sin(inc)
    pos = rr * np.vstack([x, y, z])
    # view rotation: look from lat 25, lon 0
    vl = np.radians(25)
    Rx = np.array([[1, 0, 0], [0, np.cos(vl), -np.sin(vl)], [0, np.sin(vl), np.cos(vl)]])
    q = Rx @ np.vstack([pos[1], pos[2], pos[0]])          # screen x, screen y, depth
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.8), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = ax[0]
    a.add_patch(Circle((0, 0), 1, fc=EARTH, ec=NAVY, zorder=2))
    # equator and graticule
    t = np.linspace(0, 2 * np.pi, 200)
    for latg in [0, 53, -53]:
        la = np.radians(latg)
        g = Rx @ np.vstack([np.cos(la) * np.sin(t), np.sin(la) * np.ones_like(t), np.cos(la) * np.cos(t)])
        vis = g[2] > 0
        gx = np.where(vis, g[0], np.nan); gy = np.where(vis, g[1], np.nan)
        a.plot(gx, gy, color="white", lw=0.5 if latg else 0.8, alpha=0.7, zorder=3, ls="-" if latg == 0 else ":")
    front = q[2] > -np.sqrt(np.maximum(0, 1 - q[0] ** 2 - q[1] ** 2)) * 0 if False else None
    behind = (q[0] ** 2 + q[1] ** 2 < 1) & (q[2] < 0)
    a.scatter(q[0][~behind & (q[2] < 0)], q[1][~behind & (q[2] < 0)], s=0.5, color=GRAY, zorder=1)
    a.scatter(q[0][q[2] >= 0], q[1][q[2] >= 0], s=0.7, color=NAVY, zorder=4)
    # one plane highlighted
    up = np.linspace(0, 2 * np.pi, 300); ra = 0.6
    xp = np.cos(ra) * np.cos(up) - np.sin(ra) * np.sin(up) * np.cos(inc)
    yp = np.sin(ra) * np.cos(up) + np.cos(ra) * np.sin(up) * np.cos(inc)
    zp = np.sin(up) * np.sin(inc)
    g = Rx @ (rr * np.vstack([yp, zp, xp]))
    occl = (g[0] ** 2 + g[1] ** 2 < 1) & (g[2] < 0)
    a.plot(np.where(occl, np.nan, g[0]), np.where(occl, np.nan, g[1]), color=ACCENT, lw=1.0, zorder=5)
    a.text(0, -1.32, "1584 satellites: 72 planes $\\times$ 22,\n550 km, 53$^\\circ$ (one plane in red)",
           ha="center", fontsize=6.6)
    a.set_xlim(-1.25, 1.25); a.set_ylim(-1.55, 1.2); _clean(a)
    b = ax[1]
    # latitude histogram over time (many snapshots)
    lats = []
    for tt in np.linspace(0, 2 * np.pi, 40):
        zz = np.sin(u + tt) * np.sin(inc)
        lats.append(np.degrees(np.arcsin(zz)))
    lats = np.concatenate(lats)
    b.hist(lats, bins=np.arange(-60, 61, 4), color=NAVY, alpha=0.8, orientation="horizontal",
           weights=np.ones_like(lats) * 100 / len(lats))
    b.set_ylabel("latitude (deg)"); b.set_xlabel("share of satellites (%)")
    b.set_yticks(range(-60, 61, 20)); b.set_ylim(-62, 62)
    for la, nm in [(48.9, "Paris"), (40.7, "New York"), (-33.9, "Sydney"), (1.3, "Singapore")]:
        b.axhline(la, color=GRAY, lw=0.5, ls=":"); b.text(7.2, la + 1, nm, fontsize=6.2, color=GRAY, ha="right")
    b.set_title("satellites pile up near $\\pm53^\\circ$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_constellation_shell")


def fig_doppler_whistle():
    """A passing satellite as a train whistle: geometry cartoon and the Doppler S-curve at 2 GHz."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.55), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    a.add_patch(Wedge((0, -30), 30.4, 78, 102, fc=EARTH, ec="none"))
    hgt = 3.0
    xs = np.linspace(-5, 5, 9)
    obs = np.array([0, 0.45])
    a.plot([-5.6, 5.6], [hgt, hgt], color=GRAY, ls=":", lw=0.8)
    for k, x in enumerate(xs):
        vr = x / np.hypot(x, hgt - obs[1])          # +1 = receding at full speed
        c = plt.cm.coolwarm(0.5 + 0.5 * vr)
        a.plot(x, hgt, "s", color=c, ms=6, mec="k", mew=0.3)
        if k in (0, 4, 8):
            a.plot([x, obs[0]], [hgt, obs[1]], color=c, lw=0.7, alpha=0.9)
    a.annotate("", (2.5, hgt + 0.6), (-2.5, hgt + 0.6), arrowprops=dict(arrowstyle="->", lw=1, color=NAVY))
    a.text(0, hgt + 0.75, "7.6 km/s", ha="center", fontsize=7, color=NAVY)
    a.plot(*obs, "^", color="k", ms=6)
    a.text(0.35, 0.3, "you", fontsize=7)
    a.text(-5.6, 1.3, "approaching:\npitch up", fontsize=6.8, color=plt.cm.coolwarm(0.02))
    a.text(3.4, 1.3, "receding:\npitch down", fontsize=6.8, color=plt.cm.coolwarm(0.98))
    a.set_xlim(-6.0, 6.0); a.set_ylim(-0.1, 4.2); _clean(a)
    b = ax[1]
    for me, c in [(90, NAVY), (40, GREEN), (20, ORANGE)]:
        p = sat.leo_pass(550 * KM, 2e9, me, 10.0, 1.0)
        b.plot(p["t"] / 60, p["doppler"] / 1e3, color=c, label=f"max elev. {me}$^\\circ$")
    b.axhline(0, color=GRAY, lw=0.6)
    b.set_xlabel("time from closest approach (min)"); b.set_ylabel("Doppler at 2 GHz (kHz)")
    b.legend(fontsize=6.4, loc="upper right"); b.set_title("the whistle's pitch: Doppler vs time", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_doppler_whistle")


def fig_handover():
    """A relay race: elevations of successive LEO satellites and which one serves the terminal."""
    r = rng(3)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    T = 20 * 60.0
    tt = np.arange(0, T, 1.0)
    best = np.full_like(tt, -90.0); who = np.full(tt.shape, -1)
    curves = []
    k = 0
    for t0 in np.arange(-240, T + 240, 75.0):
        me = r.uniform(28, 88)
        p = sat.leo_pass(550 * KM, 11.7e9, me, 25.0, 1.0)
        tk = p["t"] + t0 + r.uniform(-15, 15)
        el = np.interp(tt, tk, p["elev"], left=np.nan, right=np.nan)
        curves.append(el)
        upd = np.nan_to_num(el, nan=-90) > best
        best[upd] = el[upd]; who[upd] = k
        k += 1
    for i, el in enumerate(curves):
        ax.plot(tt / 60, el, color=GRAY, lw=0.7, alpha=0.6)
    # serving segments, alternating colours
    seg_start = 0; cols = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, SKY]
    hos = 0
    for i in range(1, len(tt) + 1):
        if i == len(tt) or who[i] != who[seg_start]:
            ax.plot(tt[seg_start:i] / 60, best[seg_start:i], color=cols[who[seg_start] % len(cols)], lw=2.4)
            if i < len(tt):
                ax.plot(tt[i] / 60, best[i], "o", ms=3, color="k"); hos += 1
            seg_start = i
    ax.axhline(25, color=GRAY, ls="--", lw=0.7); ax.text(0.2, 26.5, "25$^\\circ$ minimum elevation", fontsize=6.5, color=GRAY)
    ax.set_xlim(0, T / 60); ax.set_ylim(20, 92)
    ax.set_xlabel("time (min)"); ax.set_ylabel("elevation (deg)")
    ax.set_title(f"Who serves you? {hos} handovers in 20 minutes (highest satellite wins; black dots)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_handover")


def fig_bands():
    """Satellite bands on a log-frequency axis, with rain attenuation overlaid."""
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    bands = [("L", 1, 2, SKY), ("S", 2, 4, GREEN), ("C", 4, 8, NAVY), ("X", 8, 12, PURPLE),
             ("Ku", 12, 18, ORANGE), ("K", 18, 27, GRAY), ("Ka", 27, 40, ACCENT), ("Q/V", 40, 75, "#8E44AD"),
             ("E", 75, 90, "#5D6D7E")]
    for nm, f1, f2, c in bands:
        ax.axvspan(f1, f2, ymin=0, ymax=1, color=c, alpha=0.13, lw=0)
        ax.text(np.sqrt(f1 * f2), 60, nm, ha="center", fontsize=8.5, color=c, weight="bold")
    f = np.logspace(0, np.log10(90), 300)
    for R, ls in [(5, ":"), (25, "--"), (100, "-")]:
        ax.loglog(f, sat.rain_specific_atten(f, R), color=NAVY, ls=ls, lw=1.1, label=f"rain {R} mm/h")
    uses = [(1.55, "phones,\nships,\nGNSS"), (6, "TV\ndistribution"), (11.7, "DTH TV,\nVSAT,\nStarlink"),
            (8.4, "deep\nspace"), (25, "HTS\nbroad-\nband"), (50, "feeder\nlinks")]
    for x, t in uses:
        ax.text(x, 4e-6, t, ha="center", va="bottom", fontsize=6.2, color="#333333")
    ax.set_xlim(1, 90); ax.set_ylim(3e-6, 150)
    ax.set_xticks([1, 2, 4, 8, 12, 18, 27, 40, 75]); ax.set_xticklabels(["1", "2", "4", "8", "12", "18", "27", "40", "75"])
    ax.minorticks_off()
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("rain attenuation (dB/km)")
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.55))
    ax.set_title("The satellite bands: more bandwidth upstairs, more rain too", fontsize=9)
    fig.tight_layout(); save(fig, "ch23_bands")


def fig_voyager_bank():
    """The Voyager link as a decibel bank account, with a zoom on the last few dB."""
    items = [("deposit: Voyager EIRP", 60.7), ("withdrawal: 173 AU of\nfree space", -319.2),
             ("deposit: 70 m dish G/T", 61.3), ("deposit: $-k$", 228.6)]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]; run = 0.0
    for i, (nm, v) in enumerate(items):
        lo, hi = sorted([run, run + v])
        a.bar(i, hi - lo, bottom=lo, color=GREEN if v > 0 else ACCENT, width=0.6)
        a.text(i, hi + 8, f"{v:+.1f}", ha="center", fontsize=7)
        if i < len(items) - 1:
            a.plot([i + 0.3, i + 0.7], [run + v] * 2, color=GRAY, lw=0.6)
        run += v
    a.bar(len(items), run, color=NAVY, width=0.6); a.text(len(items), run + 8, f"{run:.1f}", ha="center", fontsize=7, color=NAVY)
    a.set_xticks(range(len(items) + 1))
    a.set_xticklabels(["EIRP", "free\nspace", "DSN\n$G/T$", "$-k$", "balance\n$C/N_0$"], fontsize=7)
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("account balance (dB)"); a.set_ylim(-290, 110)
    a.set_title("the big transactions", fontsize=8.5)
    b = ax[1]
    steps = [("$C/N_0$", 31.4), ("160 b/s", -22.0), ("losses", -4.0), ("needed", -2.5)]
    run = 0.0
    for i, (nm, v) in enumerate(steps):
        lo, hi = sorted([run, run + v])
        b.bar(i, hi - lo, bottom=lo, color=(NAVY if i == 0 else ACCENT), width=0.6)
        b.text(i, hi + 0.8, f"{v:+.1f}", ha="center", fontsize=7)
        run += v
    b.bar(len(steps), run, color=GREEN, width=0.6); b.text(len(steps), run + 0.8, f"{run:.1f}", ha="center", fontsize=7, color=GREEN)
    b.set_xticks(range(len(steps) + 1)); b.set_xticklabels([s[0] for s in steps] + ["margin"], fontsize=6.8, rotation=0)
    b.set_ylim(0, 36); b.set_ylabel("dB")
    b.set_title("what is left to spend", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_voyager_bank")


def fig_rain_path():
    """Cartoon of the P.618 geometry: only the path below the rain height is wet."""
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    th = np.radians(35); hR = 4.0
    L = 11.0
    ax.add_patch(Rectangle((-1, -0.6), 13, 0.6, fc=EARTH, ec="none"))
    ax.axhline(hR, color=SKY, ls="--", lw=0.9); ax.text(10.6, hR + 0.15, "rain height $h_R$ (0$^\\circ$C isotherm)", fontsize=7, color=SKY, ha="right")
    # rain cell
    ax.add_patch(Ellipse((3.6, 3.9), 4.2, 1.2, fc=GRAY, alpha=0.45, ec="none"))
    rr = rng(2)
    for _ in range(70):
        x0 = rr.uniform(1.8, 5.4); y0 = rr.uniform(0.1, 3.4)
        ax.plot([x0, x0 - 0.06], [y0, y0 - 0.25], color=SKY, lw=0.7)
    xe = hR / np.tan(th)
    ax.plot([0, L * np.cos(th) * 1.3], [0, L * np.sin(th) * 1.3], color=NAVY, lw=1.0)
    ax.plot([0, xe], [0, hR], color=ACCENT, lw=2.6, label="wet part of the path")
    ax.add_patch(Ellipse((0, 0.15), 0.6, 0.3, fc=NAVY))
    ax.text(-0.75, 0.45, "earth\nstation", fontsize=7, color=NAVY)
    ax.annotate("", (xe, -0.3), (0, -0.3), arrowprops=dict(arrowstyle="<->", lw=0.7, color="white"))
    ax.text(xe / 2, -0.55, "$L_G$", fontsize=8, ha="center", va="bottom", color="white")
    ax.text(xe / 2 - 0.6, hR / 2 + 0.3, "$L_s$", fontsize=9, color=ACCENT)
    ax.add_patch(Arc((0, 0), 2.0, 2.0, theta1=0, theta2=35, color="k", lw=0.6)); ax.text(1.1, 0.25, "$\\theta$", fontsize=8)
    ax.text(9.2, 6.4, "to the satellite", fontsize=7, color=NAVY, rotation=35)
    ax.text(5.9, 1.6, "rain cells are only a few km\nacross: a reduction factor\nshortens the effective path", fontsize=6.8, color="#333333")
    ax.set_xlim(-0.8, 11); ax.set_ylim(-0.6, 7.2); _clean(ax)
    save(fig, "ch23_rain_path")


def fig_mirror_translator():
    """Bent pipe (noise accumulates) versus regenerative (errors add): QPSK, uplink Eb/N0 fixed."""
    from scipy.special import erfc
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    ber = lambda ebn0_db: Q(np.sqrt(2 * 10 ** (ebn0_db / 10)))
    d = np.linspace(4, 16, 300)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    for up, c in [(11.0, NAVY), (13.0, GREEN)]:
        tot = sat.combine_cn_db(np.full_like(d, up), d)
        ax.semilogy(d, ber(tot), color=c, label=f"bent pipe, up {up:.0f} dB")
        ax.semilogy(d, ber(up) + ber(d), color=c, ls="--", label=f"regenerative, up {up:.0f} dB")
    ax.set_ylim(1e-8, 1e-1); ax.set_xlabel("downlink $E_b/N_0$ (dB)"); ax.set_ylabel("end-to-end BER")
    ax.legend(fontsize=5.6, loc="lower left"); ax.set_title("mirror vs translator (QPSK)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_mirror_translator")


def fig_aloha():
    """Pure/slotted ALOHA theory and a Monte Carlo of CRDSA (two replicas, iterative SIC)."""
    r = rng(11)
    G = np.linspace(0.02, 1.6, 200)
    fig, ax = plt.subplots(figsize=(3.0, 2.45))
    ax.plot(G, G * np.exp(-2 * G), color=GRAY, label="pure ALOHA")
    ax.plot(G, G * np.exp(-G), color=NAVY, label="slotted ALOHA")
    Ns = 200; Gs = np.arange(0.05, 1.01, 0.05); S = []
    for g in Gs:
        ok = 0; tot = 0
        for _ in range(30):
            n = r.poisson(g * Ns)
            slots = np.array([r.choice(Ns, 2, replace=False) for _ in range(n)]) if n else np.zeros((0, 2), int)
            alive = np.ones(n, bool); done = np.zeros(n, bool)
            changed = True
            while changed:
                changed = False
                occ = np.zeros(Ns, int)
                for i in np.where(alive)[0]:
                    occ[slots[i]] += 1
                for i in np.where(alive)[0]:
                    if occ[slots[i, 0]] == 1 or occ[slots[i, 1]] == 1:
                        alive[i] = False; done[i] = True; changed = True
            ok += done.sum(); tot += Ns
        S.append(ok / tot)
    ax.plot(Gs, S, "o-", color=ACCENT, ms=2.5, label="CRDSA (2 replicas, sim.)")
    ax.set_xlabel("offered load $G$ (packets/slot)"); ax.set_ylabel("throughput $S$")
    ax.set_ylim(0, 0.62); ax.set_xlim(0, 1.6); ax.legend(fontsize=6.0, loc="upper right")
    fig.tight_layout(); save(fig, "ch23_aloha")


def fig_hts_capacity():
    """Spot beams multiply capacity: 5 GHz of spectrum, 4-colour reuse, 2.5 b/s/Hz average."""
    beams = [1, 20, 50, 100, 200]
    cap = [5 * 2.5] + [b * 1.25 * 2.5 for b in beams[1:]]
    fig, ax = plt.subplots(figsize=(3.0, 2.45))
    ax.bar(range(len(beams)), cap, color=[GRAY] + [NAVY] * 4, width=0.6)
    for i, c in enumerate(cap):
        ax.text(i, c + 12, f"{c:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(beams))); ax.set_xticklabels(["1 wide\nbeam"] + [f"{b}\nbeams" for b in beams[1:]], fontsize=6.8)
    ax.set_ylabel("satellite capacity (Gb/s)"); ax.set_ylim(0, 700)
    ax.axhline(140, color=ACCENT, ls=":", lw=0.8); ax.text(-0.35, 118, "ViaSat-1 (2011)", fontsize=6.3, color=ACCENT)
    ax.set_title("same spectrum, more beams", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_hts_capacity")


def fig_d2d():
    """Direct-to-device: uplink C/N0 from a phone versus the satellite's array area at 2 GHz."""
    A = np.logspace(-0.3, 2.2, 200)
    lam = 3e8 / 2e9
    G = 10 * np.log10(4 * np.pi * A * 0.69 / lam ** 2)
    gt = G - 10 * np.log10(500)
    cn0 = -10 - 159.1 - 4 + gt + 228.6
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    ax.semilogx(A, cn0 - 10 * np.log10(180e3), color=NAVY, label="SNR in 180 kHz (one RB)")
    ax.semilogx(A, cn0 - 10 * np.log10(5e6), color=ACCENT, label="SNR in 5 MHz")
    ax.axhline(0, color=GRAY, lw=0.6)
    for a_, nm in [(1, "1 m$^2$"), (10, "10 m$^2$"), (64, "64 m$^2$")]:
        ax.axvline(a_, color=GRAY, ls=":", lw=0.6); ax.text(a_ * 1.07, -14.5, nm, fontsize=6.3, color=GRAY)
    ax.set_xlabel("satellite array area (m$^2$)"); ax.set_ylabel("uplink SNR (dB)")
    ax.set_ylim(-16, 24); ax.legend(fontsize=6.0, loc="upper left")
    ax.set_title("a phone's whisper to orbit", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_d2d")


def fig_dvbs2_frame():
    """DVB-S2 physical-layer frame: PLHEADER, 90-symbol slots, pilot blocks every 16 slots."""
    fig, ax = plt.subplots(figsize=(W1, 1.45))
    x = 0.0
    def blk(w, c, t, tc="white", fs=6.5):
        nonlocal x
        ax.add_patch(Rectangle((x, 0), w, 1, fc=c, ec="white", lw=0.8))
        if t:
            ax.text(x + w / 2, 0.5, t, ha="center", va="center", fontsize=fs, color=tc)
        x += w
    blk(1.0, ACCENT, "SOF\n26")
    blk(2.2, ORANGE, "PLS code\n64 ($\\pi$/2-BPSK)")
    for k in range(16):
        blk(0.55, NAVY if k % 2 == 0 else SKY, "" if k not in (0, 15) else "slot", fs=5.5)
    blk(0.45, GREEN, "P", fs=6)
    for k in range(6):
        blk(0.55, NAVY if k % 2 == 0 else SKY, "")
    ax.text(x + 0.35, 0.5, "$\\cdots$", fontsize=10, va="center")
    ax.annotate("", (3.2, 1.25), (3.2 + 16 * 0.55, 1.25), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(3.2 + 8 * 0.55, 1.33, "16 slots of 90 symbols (data, scrambled)", ha="center", fontsize=6.8)
    ax.text(0.0 + 1.6, -0.35, "PLHEADER: 90 symbols, 7 bits of MODCOD/frame-type signalling", ha="center", fontsize=6.6, color=ORANGE)
    ax.text(3.2 + 16 * 0.55 + 0.22, -0.35, "pilot block: 36 known symbols", ha="center", fontsize=6.6, color=GREEN)
    ax.set_xlim(-0.1, x + 1.0); ax.set_ylim(-0.6, 1.6); ax.axis("off")
    save(fig, "ch23_dvbs2_frame")


def fig_tsys_budget():
    """Where the noise comes from in a Ku-band DTH terminal, clear sky, with a lossy feed, and in rain."""
    lnb = sat.nf_to_temp(0.7)
    cases = []
    cases.append(("clear sky", [("sky", 15.0), ("ground spillover", 20.0), ("feed loss", 0.0), ("rain emission", 0.0), ("LNB", lnb)]))
    Lf = 10 ** (0.3 / 10)
    cases.append(("0.3 dB cable\nbefore LNB", [("sky", 15.0 / Lf), ("ground spillover", 20.0 / Lf),
                                               ("feed loss", 290 * (1 - 1 / Lf)), ("rain emission", 0.0), ("LNB", lnb)]))
    A = 3.0
    dT = sat.rain_noise_temp(A)
    cases.append(("3 dB rain fade", [("sky", 15.0 * 10 ** (-A / 10)), ("ground spillover", 20.0), ("feed loss", 0.0),
                                     ("rain emission", dT), ("LNB", lnb)]))
    cols = {"sky": SKY, "ground spillover": GREEN, "feed loss": ORANGE, "rain emission": GRAY, "LNB": NAVY}
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for i, (nm, parts) in enumerate(cases):
        left = 0
        for pn, v in parts:
            if v > 0:
                ax.barh(i, v, left=left, color=cols[pn], height=0.6, label=pn if i == (0 if pn != "feed loss" and pn != "rain emission" else (1 if pn == "feed loss" else 2)) else None)
                if v > 14:
                    ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8, color="white")
                left += v
        ax.text(left + 4, i, f"$T_{{sys}}$ = {left:.0f} K", va="center", fontsize=7.5)
    ax.set_yticks(range(len(cases))); ax.set_yticklabels([c[0] for c in cases], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("system noise temperature contributions (K)"); ax.set_xlim(0, 265)
    ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
    fig.tight_layout(); save(fig, "ch23_tsys_budget")


def fig_acm_year():
    """Spectral efficiency through an average year: ACM versus a CCM design for 99.9% availability."""
    f = 19.7; el = 34.3
    G = sat.dish_gain_dbi(0.75, f * 1e9, 0.65)
    p = np.logspace(-2.3, 1.7, 400)       # % of time exceeded
    A = sat.rain_atten_p618(f, el, p, 42, lat_deg=40)
    tclear = sat.system_noise_temp(t_ant=60.0, t_rx=sat.nf_to_temp(1.5), loss_db=0.3)
    eff = []
    for a in A:
        ts = sat.system_noise_temp(t_ant=60.0 * 10 ** (-a / 10) + sat.rain_noise_temp(a), t_rx=sat.nf_to_temp(1.5), loss_db=0.3)
        cn = 11.8 - a - 10 * np.log10(ts / tclear)
        tot = sat.combine_cn_db(cn, 20.0)
        m = sat.acm_select(tot, 0.5)
        eff.append(m[1] if m else 0.0)
    eff = np.array(eff)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    avail = 100 - p
    ax.semilogx(p, eff, color=NAVY, drawstyle="steps-post", label="ACM: best MODCOD that closes")
    ax.fill_between(p, eff, step="post", color=NAVY, alpha=0.12)
    ax.axhline(0.79, color=ACCENT, ls="--", lw=1.1, label="CCM sized for 99.9% (QPSK 2/5)")
    ax.axvline(0.1, color=GRAY, ls=":", lw=0.7); ax.text(0.105, 3.15, "99.9%", fontsize=6.8, color=GRAY)
    ax.invert_xaxis()
    ax.set_xlabel("percentage of the year the rain fade is worse than this (%)")
    ax.set_ylabel("spectral efficiency (b/s/Hz)"); ax.set_ylim(0, 3.4)
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("Ka-band spot beam through an average year (worked example, $R_{0.01}$ = 42 mm/h)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_acm_year")


def fig_gso_arc():
    """Sky view from 35N: the geostationary arc, a LEO shell's satellites, and the arc-avoidance band."""
    lat0 = 35.0
    slon = np.linspace(-80, 80, 400)
    az, el, _ = sat.geo_look_angles(lat0, 0.0, slon)
    ok = el > 0
    # LEO satellites from a 53-deg, 550 km Walker shell, snapshot
    P, S, inc = 72, 22, np.radians(53)
    rr = (6378 + 550) / 6378
    raan = np.repeat(np.arange(P) * 2 * np.pi / P, S)
    u = np.tile(np.arange(S) * 2 * np.pi / S, P) + np.repeat(np.arange(P), S) * 2 * np.pi / (P * S)
    pts = []
    for tt in np.linspace(0, 2 * np.pi / 22, 14):
        uu = u + tt
        x = np.cos(raan) * np.cos(uu) - np.sin(raan) * np.sin(uu) * np.cos(inc)
        y = np.sin(raan) * np.cos(uu) + np.cos(raan) * np.sin(uu) * np.cos(inc)
        z = np.sin(uu) * np.sin(inc)
        pts.append(rr * np.vstack([x, y, z]))
    pts = np.hstack(pts)
    la = np.radians(lat0)
    obs = np.array([np.cos(la), 0, np.sin(la)])
    up = obs; east = np.array([0, 1, 0]); north = np.cross(up, east)
    rel = pts - obs[:, None]
    rng_ = np.linalg.norm(rel, axis=0)
    e_ = np.degrees(np.arcsin(rel.T @ up / rng_))
    a_ = np.degrees(np.arctan2(rel.T @ east, rel.T @ north)) % 360
    vis = e_ > 25
    a_, e_, rel, rng_ = a_[vis], e_[vis], rel[:, vis], rng_[vis]
    # direction vectors to GEO arc
    rg = 42164 / 6378
    garc = rg * np.vstack([np.cos(np.radians(slon)), np.sin(np.radians(slon)), np.zeros_like(slon)])
    gv = garc - obs[:, None]; gv /= np.linalg.norm(gv, axis=0)
    lv = rel / rng_
    sep = np.degrees(np.arccos(np.clip((lv.T @ gv).max(axis=1), -1, 1)))
    near = np.abs(e_ - np.interp(a_, np.sort(az[ok]), el[ok][np.argsort(az[ok])], left=-90, right=-90)) < 13
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.fill_between(np.sort(az[ok]), 0, 0, color=ACCENT)
    order = np.argsort(az[ok])
    azs = az[ok][order]; els = el[ok][order]
    ax.fill_between(azs, els - 13, els + 13, color=ACCENT, alpha=0.12, lw=0, label="avoidance band around the GEO arc (illustrative)")
    ax.plot(azs, els, color=ACCENT, lw=2, label="geostationary arc")
    ax.scatter(a_[~near], e_[~near], s=5, color=NAVY, label="LEO satellites usable (elevation > 25$^\\circ$)")
    ax.scatter(a_[near], e_[near], s=5, facecolors="none", edgecolors=GRAY, label="LEO satellites too close to the arc")
    ax.set_xlim(60, 300); ax.set_ylim(0, 92)
    ax.set_xticks([90, 135, 180, 225, 270]); ax.set_xticklabels(["E 90", "SE 135", "S 180", "SW 225", "W 270"])
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)")
    ax.legend(fontsize=6.3, loc="upper left", ncol=2, framealpha=0.95)
    ax.set_title("The southern sky seen from 35$^\\circ$N: where a LEO terminal may and may not look", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_gso_arc")


def fig_timeline():
    """Eighty years of satellite communication on one line, coloured by orbit."""
    # (year, label, category, level): level >0 above the axis, <0 below; |level| sets the stem height
    ev = [(1945, "Clarke's article", "idea", 1), (1957, "Sputnik 1", "LEO", -1), (1960, "Echo 1", "LEO", 2),
          (1962, "Telstar 1", "LEO", -2), (1964, "Syncom 3", "GEO", 1), (1965, "Early Bird", "GEO", -3),
          (1965.6, "Molniya 1", "HEO", 3), (1982, "Inmarsat", "GEO", 1), (1989, "Astra / Sky DTH", "GEO", -1),
          (1998, "Iridium", "LEO", 2), (2005, "DVB-S2", "std", -2), (2011, "ViaSat-1 HTS", "GEO", 1),
          (2013, "O3b", "MEO", -1), (2019, "Starlink", "LEO", 3), (2022, "3GPP NTN", "std", -3),
          (2023, "DSOC laser", "deep", 2), (2024, "direct-to-cell", "LEO", -2)]
    col = {"idea": GRAY, "LEO": NAVY, "GEO": ACCENT, "HEO": PURPLE, "MEO": GREEN, "std": ORANGE, "deep": SKY}
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.axhline(0, color=GRAY, lw=1.2, zorder=1)
    for yr, nm, c, lev in ev:
        up = 1 if lev > 0 else -1
        h = up * (0.35 + 0.5 * (abs(lev) - 1))
        ax.plot([yr, yr], [0, h], color=col[c], lw=0.7)
        ax.plot(yr, 0, "o", color=col[c], ms=5, zorder=3)
        ax.text(yr, h + 0.05 * up, f"{nm} ({int(yr)})", ha="center", va="bottom" if up > 0 else "top",
                fontsize=6.2, color=col[c])
    for i, (k, c) in enumerate([("LEO", NAVY), ("GEO", ACCENT), ("MEO", GREEN), ("HEO", PURPLE),
                                ("standard", ORANGE), ("deep space", SKY)]):
        ax.plot(1946 + 9.5 * i, -1.95, "s", color=c, ms=5)
        ax.text(1947.2 + 9.5 * i, -1.95, k, fontsize=6.5, va="center")
    ax.set_xlim(1938, 2032); ax.set_ylim(-2.1, 1.75); ax.axis("off")
    save(fig, "ch23_timeline")


def fig_elements():
    """The orbital elements in a simple perspective view, with a key on the right."""
    az, el = np.radians(35), np.radians(22)

    def proj(p):
        x, y, z = p
        return np.array([y * np.cos(az) - x * np.sin(az),
                         z * np.cos(el) - (x * np.cos(az) + y * np.sin(az)) * np.sin(el)])
    fig, (ax, key) = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    t = np.linspace(0, 2 * np.pi, 400)
    eq = np.array([proj((1.6 * np.cos(a), 1.6 * np.sin(a), 0)) for a in t])
    ax.fill(eq[:, 0], eq[:, 1], color=SKY, alpha=0.12); ax.plot(eq[:, 0], eq[:, 1], color=SKY, lw=0.7)
    inc, Om, w, e, a = np.radians(50), np.radians(40), np.radians(75), 0.3, 1.3

    def rot(p):
        x, y, z = p
        x1, y1 = x * np.cos(w) - y * np.sin(w), x * np.sin(w) + y * np.cos(w)
        y2, z2 = y1 * np.cos(inc), y1 * np.sin(inc)
        return (x1 * np.cos(Om) - y2 * np.sin(Om), x1 * np.sin(Om) + y2 * np.cos(Om), z2)
    orb = np.array([rot((a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E), 0)) for E in t])
    P = np.array([proj(p) for p in orb])
    above = orb[:, 2] >= 0
    ax.plot(np.where(above, P[:, 0], np.nan), np.where(above, P[:, 1], np.nan), color=NAVY, lw=1.5, zorder=5)
    ax.plot(np.where(~above, P[:, 0], np.nan), np.where(~above, P[:, 1], np.nan), color=NAVY, lw=1.0, ls="--", zorder=1)
    ax.add_patch(Circle((0, 0), 0.22, fc=EARTH, ec=NAVY, zorder=3))
    n1 = proj((1.75 * np.cos(Om), 1.75 * np.sin(Om), 0)); n2 = proj((-1.75 * np.cos(Om), -1.75 * np.sin(Om), 0))
    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color=GRAY, lw=0.8, ls=":", zorder=2)
    ax.text(n1[0] + 0.06, n1[1] + 0.02, "line of nodes", fontsize=6.5, color=GRAY)
    r0 = proj((1.9, 0, 0))
    ax.annotate("", r0, proj((0.25, 0, 0)), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    ax.text(r0[0] - 0.05, r0[1] - 0.16, "reference direction", fontsize=6.3, color=GRAY, ha="center")
    per = proj(rot((a * (1 - e), 0, 0)))
    ax.plot([0, per[0]], [0, per[1]], color=ORANGE, lw=0.9, zorder=4)
    ax.plot(*per, "o", color=ORANGE, ms=4, zorder=6)
    ax.text(per[0] + 0.05, per[1], "perigee", fontsize=6.5, color=ORANGE, va="center")
    s_ = proj(rot((a * (np.cos(2.2) - e), a * np.sqrt(1 - e ** 2) * np.sin(2.2), 0)))
    ax.plot(*s_, "s", color=ACCENT, ms=5, zorder=6)
    ax.text(s_[0] - 0.06, s_[1] + 0.06, "satellite", fontsize=6.5, color=ACCENT, ha="right")
    ax.text(eq[300, 0] - 0.2, eq[300, 1] - 0.22, "equatorial plane", fontsize=6.5, color=SKY)
    ax.text(0.0, 1.35, "orbit plane, tilted by $i$", fontsize=6.5, color=NAVY, ha="center")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.1, 1.5); _clean(ax)
    key.axis("off")
    rows = [("$a$, $e$", "size and shape of the ellipse"), ("$i$", "tilt of the orbit plane"),
            ("$\\Omega$", "where it crosses the equator\ngoing north (the node)"),
            ("$\\omega$", "where in the plane the\nperigee lies"), ("$M$", "where the satellite is,\nat a given time")]
    for k, (sym, txt) in enumerate(rows):
        key.text(0.0, 0.95 - k * 0.2, sym, fontsize=8.5, color=NAVY, va="top")
        key.text(0.22, 0.95 - k * 0.2, txt, fontsize=7, va="top", color="#333333")
    key.set_title("six numbers fix an orbit", fontsize=8.5)
    save(fig, "ch23_elements")


def fig_eclipse():
    """GEO eclipse seasons through the year, and LEO eclipse duration versus beta angle."""
    day = np.arange(0, 366)
    dec = 23.44 * np.sin(2 * np.pi * (day - 80) / 365.25)
    Tg = sat.SIDEREAL_DAY / 60
    geo = np.array([sat.eclipse_fraction(sat.GEO_ALT, b) for b in dec]) * Tg
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax[0].fill_between(day, geo, color=NAVY, alpha=0.6)
    ax[0].set_xlim(0, 365); ax[0].set_ylim(0, 95)
    ax[0].set_xticks([0, 59, 120, 181, 243, 304]); ax[0].set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax[0].set_ylabel("eclipse per day (min)"); ax[0].set_title("GEO: two eclipse seasons a year", fontsize=8.5)
    for d0, nm in [(80, "March\nequinox"), (266, "September\nequinox")]:
        ax[0].text(d0, 93, nm, ha="center", fontsize=6.3, color=GRAY, va="top")
    b = np.linspace(0, 80, 200)
    T = sat.orbital_period(sat.R_E + 550e3) / 60
    ax[1].plot(b, np.array([sat.eclipse_fraction(550e3, x) for x in b]) * T, color=ACCENT)
    ax[1].set_xlabel("Sun angle to orbit plane, $\\beta$ (deg)"); ax[1].set_ylabel("eclipse per orbit (min)")
    ax[1].set_title("LEO 550 km: most orbits", fontsize=8.5); ax[1].set_ylim(0, 40)
    fig.tight_layout(); save(fig, "ch23_eclipse")


def fig_access_schemes():
    """FDMA, TDMA, MF-TDMA and CDMA as tilings of the time-frequency plane."""
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.75))
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    a = ax[0]
    for k in range(4):
        a.add_patch(Rectangle((0.03, k * 0.25 + 0.02), 0.94, 0.21, color=cols[k], alpha=0.75))
    a.set_title("FDMA / SCPC", fontsize=8)
    a = ax[1]
    for k in range(8):
        a.add_patch(Rectangle((k * 0.125 + 0.01, 0.03), 0.105, 0.94, color=cols[k % 4], alpha=0.75))
    a.set_title("TDMA", fontsize=8)
    a = ax[2]
    r = rng(4)
    for f in range(4):
        for t in range(6):
            a.add_patch(Rectangle((t / 6 + 0.01, f * 0.25 + 0.02), 1 / 6 - 0.02, 0.21, color=cols[r.integers(4)], alpha=0.75))
    a.set_title("MF-TDMA", fontsize=8)
    a = ax[3]
    for k in range(4):
        a.add_patch(Rectangle((0.03, 0.03), 0.94, 0.94, color=cols[k], alpha=0.22))
    a.text(0.5, 0.5, "all users,\nall the time,\nall the band:\nseparated\nby codes", ha="center", va="center", fontsize=6.3)
    a.set_title("CDMA", fontsize=8)
    for a in ax:
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_xlabel("time", fontsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(True)
    ax[0].set_ylabel("frequency", fontsize=7)
    fig.tight_layout(); save(fig, "ch23_access_schemes")


def fig_ntn_ta():
    """Round-trip delay and Doppler across an NTN beam (600 km, 2 GHz)."""
    el = np.linspace(10, 90, 300)
    d = sat.slant_range(600e3, el)
    rtt = 2 * d / sat.C * 1e3
    r = sat.R_E + 600e3
    v = sat.circular_speed(600e3)
    # worst-case Doppler at each elevation if the satellite moves in the plane containing the user
    eta = np.radians(sat.nadir_angle(600e3, el))
    fd = 2e9 * v * np.sin(eta) / sat.C / 1e3
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
    ax[0].plot(el, rtt, color=NAVY)
    ax[0].axvspan(30, 90, color=SKY, alpha=0.12); ax[0].text(60, 7.0, "the beam of the\nworked example", ha="center", fontsize=6.5, color=NAVY)
    ax[0].set_xlabel("elevation (deg)"); ax[0].set_ylabel("service-link round trip (ms)"); ax[0].set_title("timing advance needed", fontsize=8.5)
    ax[1].plot(el, fd, color=ACCENT)
    ax[1].axvspan(30, 90, color=SKY, alpha=0.12)
    ax[1].axhline(15, color=GRAY, ls=":", lw=0.8); ax[1].text(12, 16, "one 15 kHz subcarrier", fontsize=6.3, color=GRAY)
    ax[1].set_xlabel("elevation (deg)"); ax[1].set_ylabel("max. Doppler at 2 GHz (kHz)"); ax[1].set_title("Doppler to pre-compensate", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_ntn_ta")


def fig_combine_curves():
    """Combining uplink and downlink C/N: like resistors in parallel."""
    d = np.linspace(0, 30, 300)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for up, c in [(10, ACCENT), (15, ORANGE), (20, GREEN), (25, NAVY)]:
        tot = sat.combine_cn_db(np.full_like(d, up), d)
        ax.plot(d, tot, color=c, label=f"uplink $C/N$ = {up} dB")
        ax.plot(up, up - 3.01, "o", color=c, ms=4)
    ax.plot(d, d, color=GRAY, ls=":", lw=0.9, label="downlink alone")
    ax.annotate("equal links cost 3 dB", (15, 12), (19, 6.5), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("downlink $C/N$ (dB)"); ax.set_ylabel("end-to-end $C/N$ (dB)")
    ax.set_xlim(0, 30); ax.set_ylim(0, 26); ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("A bent pipe is never better than its weaker half", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_combine_curves")


def fig_dish_gain():
    """Reflector gain versus diameter in three bands, and the Ruze surface-error penalty."""
    D = np.linspace(0.3, 10, 300)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
    for f, nm, c in [(4e9, "C (4 GHz)", GREEN), (12e9, "Ku (12 GHz)", NAVY), (20e9, "Ka (20 GHz)", ACCENT)]:
        ax[0].semilogx(D, sat.dish_gain_dbi(D, f, 0.65), color=c, label=nm)
    ax[0].plot(0.6, sat.dish_gain_dbi(0.6, 11.7e9, 0.65), "o", color=NAVY, ms=4)
    ax[0].annotate("60 cm TV dish", (0.6, 35.5), (0.9, 26), fontsize=6.6, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax[0].set_xticks([0.3, 1, 3, 10]); ax[0].set_xticklabels(["0.3", "1", "3", "10"])
    ax[0].set_xlabel("dish diameter (m)"); ax[0].set_ylabel("gain (dBi), $\\eta$ = 0.65")
    ax[0].legend(fontsize=6.5); ax[0].set_title("double the diameter: +6 dB", fontsize=8.5)
    f = np.linspace(1, 50, 300)
    for eps, ls in [(0.25, ":"), (0.5, "-"), (1.0, "--")]:
        lam = 3e8 / (f * 1e9)
        loss = -10 * np.log10(np.exp(-(4 * np.pi * eps * 1e-3 / lam) ** 2))
        ax[1].plot(f, loss, color=PURPLE, ls=ls, label=f"rms error {eps} mm")
    ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel("surface-error loss (dB)")
    ax[1].set_ylim(0, 6); ax[1].legend(fontsize=6.5); ax[1].set_title("Ruze: a rough dish fails at high frequency", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_dish_gain")


def fig_rain_noise():
    """Rain hurts twice: C/N degradation versus rain attenuation for a quiet and a noisy receiver."""
    A = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.plot(A, A, color=GRAY, ls=":", label="attenuation alone")
    for ts, c in [(86, NAVY), (300, GREEN), (800, ORANGE)]:
        deg = A + 10 * np.log10((ts + sat.rain_noise_temp(A)) / ts)
        ax.plot(A, deg, color=c, label=f"clear-sky $T_{{sys}}$ = {ts} K")
    ax.set_xlabel("rain attenuation $A$ (dB)"); ax.set_ylabel("loss of $C/N$ (dB)")
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.legend(fontsize=6.6, loc="upper left")
    ax.set_title("The quieter the receiver, the more the rain's own noise costs", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_rain_noise")


def fig_dvbs2_constellations():
    """The DVB-S2 constellations: QPSK, 8PSK, 16APSK (4+12) and 32APSK (4+12+16)."""
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.75))
    sets = [("QPSK", np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))),
            ("8PSK", np.exp(1j * np.arange(8) * np.pi / 4)),
            ("16APSK 4+12", sat.dvbs2_16apsk("3/4")), ("32APSK 4+12+16", sat.dvbs2_32apsk("3/4"))]
    for a, (nm, p) in zip(ax, sets):
        p = p / np.sqrt(np.mean(np.abs(p) ** 2))
        r = np.unique(np.round(np.abs(p), 3))
        for rr in r:
            a.add_patch(Circle((0, 0), rr, fill=False, color=GRAY, lw=0.5, ls=":"))
        a.plot(p.real, p.imag, "o", ms=3.2, color=NAVY)
        a.set_aspect("equal"); a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45)
        a.set_xticks([]); a.set_yticks([]); a.set_title(nm, fontsize=8)
    fig.tight_layout(); save(fig, "ch23_dvbs2_constellations")


def fig_coding_ladder():
    """Why deep space loved codes: Eb/N0 needed for BER 1e-5 with successive generations (approximate)."""
    items = [("uncoded BPSK", 9.6, GRAY), ("convolutional $K$=7, $r$=1/2\n(Viterbi)", 4.4, NAVY),
             ("Reed-Solomon + convolutional\n(Voyager, DVB-S)", 2.5, GREEN),
             ("Shannon limit, rate 1/2", 0.2, ACCENT)]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for k, (nm, v, c) in enumerate(items):
        ax.barh(k, v, color=c, height=0.55)
        ax.text(v + 0.15, k, f"{v:.1f} dB", va="center", fontsize=7)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("$E_b/N_0$ needed for a bit error rate of $10^{-5}$ (dB, approximate)"); ax.set_xlim(0, 11.5)
    ax.set_title("Every decibel of coding gain is a decibel of data rate across the solar system", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch23_coding_ladder")


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1:
        for nm in _s.argv[1:]:
            globals()["fig_" + nm]()
        raise SystemExit
    fig_orbits_scale(); fig_ground_tracks(); fig_geometry_curves(); fig_geo_visibility(); fig_leo_doppler()
    fig_rain(); lb, info, cnd, tot = fig_dth_budget(); fig_deep_space(); fig_dish_pattern(); fig_beam_reuse()
    fig_twta(); fig_apsk_vs_qam(); fig_intermod(); fig_dvbs2_modcods(); fig_tcp_delay()
    fig_by_numbers(); fig_altitude_scale(); fig_clarke_three(); fig_kepler(); fig_constellation_shell()
    fig_doppler_whistle(); fig_handover(); fig_bands(); fig_voyager_bank(); fig_rain_path()
    fig_mirror_translator(); fig_aloha(); fig_hts_capacity(); fig_d2d(); fig_dvbs2_frame()
    print(lb.table()); print(info); print("C/N down", cnd, "total", tot)
