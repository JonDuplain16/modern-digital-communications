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


if __name__ == "__main__":
    fig_orbits_scale(); fig_ground_tracks(); fig_geometry_curves(); fig_geo_visibility(); fig_leo_doppler()
    fig_rain(); lb, info, cnd, tot = fig_dth_budget(); fig_deep_space(); fig_dish_pattern(); fig_beam_reuse()
    fig_twta(); fig_apsk_vs_qam(); fig_intermod(); fig_dvbs2_modcods(); fig_tcp_delay()
    print(lb.table()); print(info); print("C/N down", cnd, "total", tot)
