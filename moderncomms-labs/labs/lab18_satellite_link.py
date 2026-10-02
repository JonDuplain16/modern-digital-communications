"""Lab 18 · Satellite Links: Budgets, Orbits, Rain, ACM, the TWTA and Voyager   (Chapter 23)

Run it:      python labs/lab18_satellite_link.py
Self-test:   python labs/lab18_satellite_link.py --selftest

A satellite link is the purest link budget in engineering: a transmitter 36 000 km (or 550 km,
or 173 astronomical units) away, free space in between, then a few kilometres of weather that
decide whether you get your television tonight. Seven experiments built on commlib.satellite,
the module that drew every data figure in Chapter 23: the decibel bank account, GEO versus LEO
geometry, a live LEO pass, rain fades, adaptive coding through a year of weather, the
travelling-wave tube, and the most extreme link humans close every day.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

import commlib as cl
from commlib import satellite as sat
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ConstellationPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

KM = 1e3
DB = sat.db


# =============================================================================== shared helpers
def waterfall(plot, key, items, total_name, total_color=NAVY, fmt="{:+.1f}", total_fmt="{:.1f}",
              label_size=8.5):
    """Draw a decibel 'bank statement': each item a floating bar from the running balance,
    deposits green, withdrawals red, and the final balance as a solid bar. Returns the total."""
    run = 0.0
    names = []
    for i, (nm, val) in enumerate(items):
        lo, hi = sorted([run, run + val])
        plot.bars(f"{key}{i}", [i], [hi], base=lo, width=0.62, color=GREEN if val >= 0 else RED)
        plot.text(f"{key}t{i}", i, hi, fmt.format(val), anchor=(0.5, 1.05), size=label_size,
                  bold=True)
        if i < len(items) - 1:
            plot.line(f"{key}c{i}", [i + 0.31, i + 0.69], [run + val] * 2, color=GRAY, width=1.0)
        run += val
        names.append(nm)
    n = len(items)
    lo, hi = sorted([0.0, run])
    plot.bars(f"{key}T", [n], [hi], base=lo, width=0.62, color=total_color)
    plot.text(f"{key}tT", n, hi, total_fmt.format(run), anchor=(0.5, 1.05), size=label_size + 0.5,
              bold=True, color=total_color)
    plot.set_xticks([(i, s) for i, s in enumerate(names + [total_name])])
    plot.set_xlim(-0.6, n + 0.6)
    return run


def best_modcod(cn_db, margin=0.0):
    m = sat.acm_select(float(cn_db), margin)
    return m


# ---------------------------------------------------------------- the three worked examples
def _scenarios():
    _, el_dth, d_dth = sat.geo_look_angles(48.0, 2.0, 19.2)
    _, el_ka, d_ka = sat.geo_look_angles(40.0, 0.0, 30.0)
    d_leo = sat.slant_range(550 * KM, 40.0)
    return {
        "GEO Ku DTH (Paris)": dict(
            f=11.7e9, d=float(d_dth), el=float(el_dth), eff=0.65, tant=35.0, feed=0.0, misc=0.8,
            rs=27.5e6, interf=[("uplink", 25.0), ("adjacent sats", 21.0), ("cross-pol.", 22.0),
                               ("intermod.", 20.0)],
            ref=("8PSK 2/3", 6.62), eirp=51.0, dish=0.6, nf=0.7,
            what="a 60 cm TV dish near Paris, looking at 19.2°E"),
        "Ka HTS spot beam": dict(
            f=19.7e9, d=float(d_ka), el=float(el_ka), eff=0.65, tant=60.0, feed=0.3, misc=0.8,
            rs=200e6, interf=[("co-channel beams", 20.0)], ref=("16APSK 3/4", 10.21),
            eirp=58.0, dish=0.75, nf=1.5,
            what="a 75 cm broadband terminal at 40°N in a high-throughput spot beam"),
        "LEO user terminal": dict(
            f=11.7e9, d=float(d_leo), el=40.0, eff=0.70, tant=40.0, feed=0.0, misc=1.3,
            rs=240e6, interf=[], ref=("32APSK 9/10", 16.05), eirp=36.0,
            dish=2 * np.sqrt(0.23 / np.pi), nf=1.39,
            what="a 0.23 m² flat phased array talking to a 550 km satellite at 40° elevation"),
    }


SCEN = _scenarios()


# =============================================================================== 1. link budget
class LinkBudgetBank(Experiment):
    title = "Link budget: a decibel bank account"
    blurb = "Deposits (EIRP, G/T), withdrawals (path, rain), and what is left for the demodulator."
    book = "sec:ch23:linkbudget"
    controls = [
        Choice("scen", "Worked example", list(SCEN), "GEO Ku DTH (Paris)", style="menu",
               help="Choosing one loads its EIRP, antenna and LNB values"),
        Heading("Satellite"),
        Slider("eirp", "Satellite EIRP", 20, 65, 51.0, step=0.5, unit="dBW"),
        Heading("Ground terminal"),
        Slider("dish", "Dish diameter", 0.2, 2.4, 0.6, step=0.01, unit="m",
               help="For the LEO flat panel: the diameter of a dish with the same area"),
        Slider("nf", "LNB noise figure", 0.3, 3.0, 0.7, step=0.05, unit="dB"),
        Heading("Weather and neighbours"),
        Slider("rain", "Rain fade", 0, 10, 0, step=0.1, unit="dB"),
        Toggle("sky", "Rain also adds sky noise", True),
        Toggle("interf", "Include uplink and interference", True),
    ]
    plots = [
        BarPlot("bank", "The downlink C/N₀, deposit by withdrawal", y="running balance (dB)"),
        BarPlot("spend", "What is left: C/N and the interference bill", y="dB"),
        BarPlot("tsys", "Where the noise comes from", y="noise temperature (K)"),
    ]
    layout = [["bank", "bank"], ["spend", "tsys"]]
    row_stretch = [1, 1]
    col_stretch = [3, 2]
    readouts = [
        Readout("cn0", "C/N₀", "dB-Hz", ".1f"),
        Readout("cn", "Downlink C/N", "dB", ".1f"),
        Readout("tot", "Total C/(N+I)", "dB", ".1f"),
        Readout("mc", "Best DVB-S2 MODCOD", "", None),
    ]
    challenges = [
        Challenge("DTH: shrink the dish until the total C/(N+I) only just stays above the 6.6 dB "
                  "that 8PSK 2/3 needs (no rain).",
                  lambda s: (s.p.scen.startswith("GEO") and s.p.rain == 0 and s.p.interf
                             and abs(s.p.eirp - 51) < 0.01 and abs(s.p.nf - 0.7) < 0.01
                             and 6.62 <= s.r.tot and s.exp.tot_smaller < 6.62),
                  hint="One more centimetre smaller and the picture freezes."),
        Challenge("DTH, 60 cm dish: raise the LNB noise figure until the total C/(N+I) has lost "
                  "a full 1 dB.",
                  lambda s: (s.p.scen.startswith("GEO") and abs(s.p.dish - 0.6) < 0.005
                             and s.p.interf and s.p.rain == 0 and s.r.tot <= s.exp.tot_ref - 1)),
        Challenge("Ka spot beam in a 5 dB rain fade (sky noise on): keep 8PSK 2/3 (6.6 dB) "
                  "running with the smallest dish you can (≤ 1.2 m).",
                  lambda s: (s.p.scen.startswith("Ka") and s.p.rain >= 5 and s.p.sky
                             and s.p.interf and s.r.tot >= 6.62 and s.p.dish <= 1.2)),
    ]

    def setup(self):
        self._scen = None

    def budget(self, S, eirp, dish, nf, rain, sky, interf):
        L = float(sat.fspl_db(S["d"], S["f"]))
        G = float(sat.dish_gain_dbi(dish, S["f"], S["eff"]))
        t_sky = S["tant"] * 10 ** (-rain / 10)
        t_rain = float(sat.rain_noise_temp(rain)) if sky else 0.0
        t_lnb = float(sat.nf_to_temp(nf))
        tsys = float(sat.system_noise_temp(t_sky + t_rain, t_lnb, loss_db=S["feed"]))
        items = [("EIRP", eirp), ("free space", -L), ("misc.", -S["misc"])]
        if rain > 0:
            items.append(("rain", -rain))
        items += [("dish gain", G), (f"Tsys {tsys:.0f} K", -float(DB(tsys))), ("−k", 228.6)]
        cn0 = sum(v_ for _, v_ in items)
        cn = cn0 - float(DB(S["rs"]))
        terms = [x for _, x in S["interf"]] if interf else []
        tot = float(sat.combine_cn_db(cn, *terms)) if terms else cn
        return dict(items=items, cn0=cn0, cn=cn, tot=tot, tsys=tsys, G=G, L=L,
                    parts=dict(sky=t_sky, rain=t_rain, lnb=t_lnb,
                               feed=tsys - t_lnb - (t_sky + t_rain) / 10 ** (S["feed"] / 10)))

    def update(self, p):
        if p.scen != self._scen:
            first = self._scen is None
            self._scen = p.scen
            S = SCEN[p.scen]
            if not first:
                for k in ("eirp", "dish", "nf"):
                    self.set_control(k, round(float(S[k]), 3))
                p = st.Params(p, eirp=S["eirp"], dish=round(float(S["dish"]), 3), nf=S["nf"])
        S = SCEN[p.scen]
        b = self.budget(S, p.eirp, p.dish, p.nf, p.rain, p.sky, p.interf)
        self.tot_smaller = self.budget(S, p.eirp, p.dish - 0.01, p.nf, p.rain, p.sky, p.interf)["tot"]
        self.tot_ref = self.budget(S, p.eirp, p.dish, S["nf"], p.rain, p.sky, p.interf)["tot"]
        pb = self.plot("bank")
        cn0 = waterfall(pb, "w", b["items"], "C/N₀")
        pb.set_ylim(min(-170, cn0 - 20), 125)
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        # what is left: the downlink C/N, then each interference term's bill, then the total
        ps = self.plot("spend")
        steps = [("downlink C/N", b["cn"])]
        run = b["cn"]
        if p.interf:
            for nm, x in S["interf"]:
                nxt = float(sat.combine_cn_db(run, x))
                steps.append((nm, nxt - run))
                run = nxt
        for i, (nm, val) in enumerate(steps):
            lo, hi = sorted([0 if i == 0 else sum(v_ for _, v_ in steps[:i]),
                             sum(v_ for _, v_ in steps[:i + 1])])
            ps.bars(f"s{i}", [i], [hi], base=lo, width=0.6, color=NAVY if i == 0 else RED)
            ps.text(f"st{i}", i, hi, f"{val:+.1f}" if i else f"{val:.1f}", anchor=(0.5, 1.05),
                    size=8.5, bold=True)
        n = len(steps)
        ps.bars("sT", [n], [max(run, 0)], base=min(run, 0), width=0.6,
                color=GREEN if run >= S["ref"][1] else RED)
        ps.text("stT", n, max(run, 0), f"{run:.1f} dB", anchor=(0.5, 1.05), size=9.5, bold=True)
        ps.set_xticks([(i, s_) for i, (s_, _) in enumerate(steps)] + [(n, "total C/(N+I)")])
        ps.set_xlim(-0.6, n + 0.6)
        ps.set_ylim(min(-2, run - 2), max(b["cn"], S["ref"][1]) * 1.2 + 2)
        ps.hline("thr", S["ref"][1], color=ORANGE, style="--",
                 label=f"{S['ref'][0]} needs {S['ref'][1]:.1f} dB", label_pos=0.42)
        if S["ref"][1] != 6.62:
            ps.hline("thr66", 6.62, color=PURPLE, style=":", label="8PSK 2/3: 6.6 dB",
                     label_pos=0.02)
        # noise temperature stacked bars: clear sky vs now
        pt = self.plot("tsys")
        clear = self.budget(S, p.eirp, p.dish, p.nf, 0.0, p.sky, p.interf)
        comps = [("sky + ground", "sky", BLUE), ("rain glow", "rain", GRAY),
                 ("feed loss", "feed", ORANGE), ("LNB", "lnb", NAVY)]
        for j, bb in enumerate((clear, b)):
            base = 0.0
            for k, (nm, key, col) in enumerate(comps):
                val = max(bb["parts"][key], 0.0)
                pt.bars(f"t{j}{k}", [j], [base + val], base=base, width=0.55, color=col)
                if j == 1 and val > 0.06 * max(bb["tsys"], 1):
                    pt.text(f"tl{k}", 1.32, base + val / 2, f"{nm} {val:.0f} K", color=col,
                            size=8.5, anchor=(0, 0.5))
                base += val
            pt.text(f"tt{j}", j, base, f"{bb['tsys']:.0f} K", anchor=(0.5, 1.05), bold=True, size=9)
        pt.set_xticks([(0, "clear sky"), (1, "now")])
        pt.set_xlim(-0.5, 2.3)
        pt.set_ylim(0, max(clear["tsys"], b["tsys"]) * 1.25 + 10)
        mc = best_modcod(b["tot"])
        self.readout(cn0=b["cn0"], cn=b["cn"], tot=b["tot"],
                     mc=(mc[0] if mc else "outage"))
        self.b = b

    def story(self, p):
        S = SCEN[p.scen]
        b = getattr(self, "b", None)
        if b is None:
            return ""
        s = (f"<p>This is {S['what']}: {v(S['d'] / KM, ',.0f', 'km')} away at "
             f"{v(S['f'] / 1e9, '.1f', 'GHz')}. Read the top chart as a bank statement. The "
             f"satellite deposits {v(p.eirp, '.1f', 'dBW')} of EIRP; free space withdraws "
             f"{v(b['L'], '.1f', 'dB')} (a factor of 10<sup>{b['L'] / 10:.0f}</sup>); the dish "
             f"deposits {v(b['G'], '.1f', 'dBi')}; the receiver's noise temperature "
             f"({v(b['tsys'], '.0f', 'K')}) and Boltzmann's constant convert what is left into "
             f"C/N₀ = {v(b['cn0'], '.1f', 'dB-Hz')}.</p>")
        s += (f"<p>Spread over {v(S['rs'] / 1e6, 'g', 'Mbaud')} that is a downlink C/N of "
              f"{v(b['cn'], '.1f', 'dB')}. ")
        if p.interf and S["interf"]:
            s += (f"Then the neighbours send their bill: uplink noise and interference add as "
                  f"reciprocals, like resistors in parallel, and the total drops to "
                  f"{v(b['tot'], '.1f', 'dB')}. ")
        ok = b["tot"] >= S["ref"][1]
        s += (f"{S['ref'][0]} needs {v(S['ref'][1], '.1f', 'dB')}: "
              + (good(f"{b['tot'] - S['ref'][1]:.1f} dB of margin.") if ok else bad("not enough."))
              + "</p>")
        if p.rain > 0:
            s += (f"<p>The {v(p.rain, '.1f', 'dB')} rain fade withdraws twice: once as attenuation "
                  f"and once as noise" + (f" (the wet sky glows: Tsys rose to {v(b['tsys'], '.0f', 'K')})"
                                          if p.sky else " (sky noise switched off)") + ".</p>")
        return "<h3>Deposits and withdrawals</h3>" + s + keybox(
            "C/N₀ = EIRP − L<sub>fs</sub> − L<sub>a</sub> + G/T + 228.6 dB-Hz. "
            "Then 1/(C/N)<sub>tot</sub> = Σ 1/(C/X)<sub>i</sub>.")


# =============================================================================== 2. GEO vs LEO geometry
class GeoVsLeo(Experiment):
    title = "GEO versus LEO in one slider"
    blurb = "Altitude sets the range, the delay, the coverage, the pass time and the Doppler."
    book = "sec:ch23:geometry"
    controls = [
        LogSlider("alt", "Altitude", 300, 35786, 550, unit="km", fmt=",.0f",
                  help="Drag to the top for the geostationary orbit (35 786 km)"),
        LogSlider("f", "Frequency", 0.1, 40, 11.7, unit="GHz", fmt=".2f"),
        Slider("emax", "Pass maximum elevation", 10, 90, 60, step=1, unit="°"),
        Slider("emin", "Minimum elevation", 0, 40, 25, step=1, unit="°"),
    ]
    plots = [
        Plot("pic", "To scale: Earth, orbit and the line of sight", x="Earth radii",
             y="Earth radii", aspect=True,
             grid=False, legend="tl"),
        Plot("rng", "Slant range vs elevation", x="elevation (°)", y="slant range (km)", logy=True,
             xlim=(0, 90), ylim=(250, 5e4), legend="tr"),
        Plot("dop", "One pass: Doppler shift", x="time from closest approach (min)",
             y="Doppler (kHz)", legend="tr"),
    ]
    layout = [["pic", "rng"], ["pic", "dop"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("rng", "Slant range at max elevation", "km", ",.0f"),
        Readout("delay", "One-way delay", "ms", ".2f"),
        Readout("dur", "Pass above min elevation", "min", None),
        Readout("dop", "Peak Doppler", "kHz", ".1f"),
    ]
    challenges = [
        Challenge("Find the altitude whose orbit takes 12 hours: the GPS orbit.",
                  lambda s: abs(s.exp.period_h - 12) < 0.15,
                  hint="Kepler's third law: T² ∝ (R⊕ + h)³."),
        Challenge("Direct-to-phone: 2 GHz from 600 km. Set it up and read the peak Doppler "
                  "(under 50 kHz).",
                  lambda s: abs(s.p.f - 2) < 0.1 and abs(s.p.alt - 600) < 30 and s.r.dop < 50),
        Challenge("Keep a satellite above 25° elevation for more than 10 minutes per pass "
                  "with the lowest orbit you can (below 2000 km).",
                  lambda s: (s.p.emin >= 25 and s.exp.dur_min > 10 and s.p.alt < 2000
                             and s.p.emax >= 89)),
    ]

    def update(self, p):
        geo = p.alt >= 35000
        h = sat.GEO_ALT if geo else p.alt * KM
        r = 1 + h / sat.R_E
        emax = max(p.emax, p.emin + 1)
        rng_max = float(sat.slant_range(h, emax))
        self.period_h = float(sat.orbital_period(sat.R_E + h)) / 3600
        # picture, in Earth radii
        pp = self.plot("pic")
        th = np.linspace(-np.pi, np.pi, 721)
        x = np.linspace(-1, 1, 401)
        pp.fill_between("earth", x, -np.sqrt(1 - x ** 2), np.sqrt(1 - x ** 2), color=BLUE, alpha=0.18)
        pp.line("eo", np.sin(th), np.cos(th), color=BLUE, width=1.4)
        pp.line("orb", r * np.sin(th), r * np.cos(th), color=GRAY, width=1.0, style="--",
                name="orbit")
        g = float(sat.central_angle(h, emax))
        sx, sy = r * np.sin(g), r * np.cos(g)
        gc = float(sat.central_angle(h, p.emin))
        cap = np.linspace(g - gc, g + gc, 200)
        pp.line("cap", 1.003 * np.sin(cap), 1.003 * np.cos(cap), color=GREEN, width=5,
                name=f"coverage above {p.emin:.0f}°")
        pp.line("los", [0, sx], [1, sy], color=RED, width=1.8, name="line of sight")
        pp.scatter("sat", [sx], [sy], color=NAVY, size=12, symbol="s", name="satellite")
        pp.scatter("you", [0], [1], color=RED, size=10, symbol="t", name="you")
        if r > 2.2:
            lim = r * 1.1
            pp.set_xlim(-lim, lim)
            pp.set_ylim(-lim * 0.6, lim)
        else:                               # zoom on the station for low orbits
            size = 1.25 * max(r - 1 + 0.12, abs(sx) + 0.08, float(np.sin(gc)) + 0.08)
            pp.set_xlim(-size, size)
            pp.set_ylim(1 - 0.75 * size, 1 + 0.95 * size)
        # slant range vs elevation
        el = np.linspace(0, 90, 181)
        pr = self.plot("rng")
        pr.line("geo", el, sat.slant_range(sat.GEO_ALT, el) / KM, color=GRAY, width=1.6, style="--",
                name="GEO")
        if not geo:
            pr.line("cur", el, sat.slant_range(h, el) / KM, color=NAVY, width=2.2,
                    name=f"{p.alt:,.0f} km")
        pr.band("below", 0, p.emin, color=RED, alpha=0.06)
        pr.scatter("now", [emax], [rng_max / KM], color=RED, size=12, symbol="d", name="max elevation")
        # pass Doppler
        pd = self.plot("dop")
        f = p.f * 1e9
        if geo:
            pd.line("flat", [-60, 60], [0, 0], color=NAVY, width=2.2, name="GEO: no Doppler")
            pd.set_xlim(-60, 60)
            pd.set_ylim(-1, 1)
            dur, peak = "∞ (stationary)", 0.0
            self.dur_min = 1e9
        else:
            T = float(sat.orbital_period(sat.R_E + h))
            ps = sat.leo_pass(h, f, emax, p.emin, max(1.0, T / 3000))
            tm = ps["t"] / 60
            pd.line("d", tm, ps["doppler"] / KM, color=NAVY, width=2.2, name=f"{p.f:.2f} GHz")
            pd.hline("z", 0, color=GRAY, style="-", width=0.7)
            pk = float(np.max(np.abs(ps["doppler"]))) / KM
            pd.set_xlim(tm[0] * 1.05 if len(tm) > 1 else -1, tm[-1] * 1.05 if len(tm) > 1 else 1)
            pd.set_ylim(-1.2 * pk - 1e-3, 1.2 * pk + 1e-3)
            self.dur_min = (ps["t"][-1] - ps["t"][0]) / 60
            dur = f"{self.dur_min:.1f}"
            peak = pk
        self.readout(rng=rng_max / KM, delay=rng_max / sat.C * 1e3, dur=dur, dop=peak)

    def story(self, p):
        geo = p.alt >= 35000
        r = self.r
        s = (f"<p>The left picture is to scale. At {v('GEO' if geo else f'{p.alt:,.0f} km')} the "
             f"orbit takes {v(self.period_h, '.2f', 'h')}. Seen at {v(max(p.emax, p.emin + 1), '.0f', '°')} "
             f"elevation the satellite is {v(r.get('rng', 0), ',.0f', 'km')} away: "
             f"{v(r.get('delay', 0), '.1f', 'ms')} one way, so a request and its answer through a "
             f"bent pipe take about four times that.</p>")
        if geo:
            s += ("<p>In the geostationary orbit the satellite turns with the Earth and hangs still "
                  "in the sky: no tracking, no Doppler, no handovers, and a third of the planet in "
                  "view. The price is 36 000 km of free space: about 33 dB more path loss than "
                  "from 550 km, and a quarter of a second of round-trip delay.</p>")
        else:
            s += (f"<p>From this orbit a pass above {v(p.emin, '.0f', '°')} lasts only "
                  f"{v(r.get('dur', '—'))} minutes, and the satellite rushes past at about "
                  f"{v(float(sat.circular_speed(p.alt * KM)) / KM, '.2f', 'km/s')}: the carrier is "
                  f"shifted by up to ±{v(r.get('dop', 0), '.1f', 'kHz')} at {v(p.f, '.2f', 'GHz')}, "
                  f"sweeping through zero at closest approach (bottom right). Drag the altitude up: "
                  f"the passes stretch, the Doppler collapses, and the delay climbs.</p>")
        return "<h3>Altitude decides almost everything</h3>" + s + keybox(
            "Geometry fixes path loss, delay, Doppler and pass time before any electronics is "
            "designed: GEO trades distance for stillness, LEO the reverse.")


# =============================================================================== 3. LEO pass, live
class LEOPass(Experiment):
    title = "A LEO pass, live"
    blurb = "Watch a satellite cross your sky and its carrier slide down like a train whistle."
    book = "sec:ch23:doppler"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Slider("alt", "Altitude", 300, 2000, 550, step=10, unit="km"),
        LogSlider("f", "Carrier frequency", 0.1, 30, 11.7, unit="GHz", fmt=".3f",
                  help="NOAA weather satellites transmit at 0.137 GHz; S-band NTN near 2 GHz"),
        Slider("emax", "Pass maximum elevation", 10, 90, 70, step=1, unit="°"),
        Slider("warp", "Playback speed", 5, 120, 30, step=5, unit="× real time"),
        Toggle("comp", "Terminal pre-compensates (NTN)", False,
               help="The terminal knows its GNSS position and the satellite's orbit, predicts the "
                    "Doppler and removes it; what is left comes from a 100 m position error"),
        Button("listen", "▶  Listen to the whistle", primary=True),
    ]
    plots = [
        Plot("sky", "Your sky (zenith at the centre)", x="", y="", aspect=True, grid=False,
             xlim=(-100, 100), ylim=(-100, 100), legend=None),
        Plot("dop", "Doppler shift", x="time (min)", y="Doppler (kHz)", legend="tr"),
        Plot("rate", "Doppler rate", x="time (min)", y="rate (kHz/s)", legend="tr"),
    ]
    layout = [["sky", "dop"], ["sky", "rate"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("el", "Elevation", "°", ".1f"),
        Readout("delay", "One-way delay", "ms", ".2f"),
        Readout("dop", "Doppler now", "kHz", ".2f"),
        Readout("rate", "Doppler rate now", "Hz/s", ".0f"),
    ]
    challenges = [
        Challenge("Catch the satellite at closest approach: Doppler within 1 % of its peak "
                  "from zero (press Pause).",
                  lambda s: abs(s.r.dop) < 0.01 * s.exp.peak / 1e3 and s.exp.t_now is not None,
                  hint="Pause near the middle of the pass; drag the speed down to fine-tune."),
        Challenge("A Ku-band (11.7 GHz) overhead pass whose Doppler rate exceeds 5 kHz/s.",
                  lambda s: abs(s.p.f - 11.7) < 0.3 and s.exp.peak_rate > 5000,
                  hint="Rate ∝ v²/h: lower the orbit."),
        Challenge("Tune in a NOAA weather satellite: 0.137 GHz from 850 km, and keep the peak "
                  "Doppler under 3.5 kHz.",
                  lambda s: abs(s.p.f - 0.137) < 0.005 and abs(s.p.alt - 850) < 20
                  and s.exp.peak / 1e3 < 3.5),
    ]

    def setup(self):
        self.t_now = None

    def track(self, p):
        h = p.alt * KM
        f = p.f * 1e9
        ps = sat.leo_pass(h, f, p.emax, 0.0, 1.0)
        # sky coordinates: re-create the pass geometry to get the azimuth
        re, r = sat.R_E, sat.R_E + h
        w = np.sqrt(sat.MU_E / r ** 3)
        beta = float(sat.central_angle(h, p.emax)) if p.emax < 90 else 0.0
        obs = re * np.array([np.cos(beta), 0.0, np.sin(beta)])
        ph = w * ps["t"]
        s = r * np.vstack([np.cos(ph), np.sin(ph), np.zeros_like(ph)])
        rel = s - obs[:, None]
        up = obs / re
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(up, e1)
        az = np.arctan2(rel.T @ e1, rel.T @ e2)
        zd = 90 - ps["elev"]
        ps["sx"], ps["sy"] = zd * np.sin(az), zd * np.cos(az)
        return ps

    def update(self, p):
        self.ps = self.track(p)
        t = self.ps["t"]
        if self.t_now is None or not (t[0] <= self.t_now <= t[-1]):
            self.t_now = float(t[0])
        self.peak = float(np.max(np.abs(self.ps["doppler"])))
        self.peak_rate = float(np.max(np.abs(self.ps["doppler_rate"])))
        self.draw(p)

    def tick(self, p):
        t = self.ps["t"]
        self.t_now += p.warp / self.fps
        if self.t_now > t[-1]:
            self.t_now = float(t[0])
        self.draw(p)

    def residual(self, p):
        """Doppler left after NTN pre-compensation with a 100 m along-track position error."""
        v_ = float(sat.circular_speed(p.alt * KM))
        dt = 100.0 / v_
        return np.gradient(self.ps["doppler"], self.ps["t"]) * dt

    def draw(self, p):
        ps = self.ps
        t, tm = ps["t"], ps["t"] / 60
        i = int(np.clip(np.searchsorted(t, self.t_now), 0, len(t) - 1))
        sk = self.plot("sky")
        a = np.linspace(0, 2 * np.pi, 181)
        for k, rr in enumerate((90, 60, 30)):
            sk.line(f"r{k}", rr * np.cos(a), rr * np.sin(a), color=GRAY, width=1.0 if k else 1.6,
                    style="-" if k == 0 else ":")
            sk.text(f"rl{k}", 2, rr, f"{90 - rr}°", color=GRAY, size=8, anchor=(0, 1))
        sk.text("N", 0, 97, "N", color=GRAY, size=9, anchor=(0.5, 1))
        sk.set_xticks([])
        sk.set_yticks([])
        sk.line("trk", ps["sx"], ps["sy"], color=NAVY, width=2.0, style="--")
        sk.scatter("sat", [ps["sx"][i]], [ps["sy"][i]], color=RED, size=16, symbol="s")
        sk.scatter("me", [0], [0], color=GREEN, size=8, symbol="+")
        dop = ps["doppler"] / KM
        rate = ps["doppler_rate"] / KM
        if p.comp:
            res = self.residual(p) / KM
            dop_show = res
        else:
            dop_show = dop
        pd = self.plot("dop")
        pd.line("d", tm, dop, color=GRAY if p.comp else NAVY, width=1.4 if p.comp else 2.2,
                name="raw Doppler")
        if p.comp:
            pd.line("res", tm, res, color=GREEN, width=2.2, name="after pre-compensation")
        pd.hline("z", 0, color=GRAY, style="-", width=0.7)
        pd.vline("now", tm[i], color=RED, style=":")
        pd.scatter("pt", [tm[i]], [dop_show[i]], color=RED, size=11)
        pk = max(np.max(np.abs(dop)), 1e-3)
        pd.set_xlim(tm[0], tm[-1])
        pd.set_ylim(-1.2 * pk, 1.25 * pk)
        pr = self.plot("rate")
        pr.line("r", tm, rate, color=PURPLE, width=2.0, name="d(Doppler)/dt")
        pr.vline("now", tm[i], color=RED, style=":")
        pr.scatter("pt", [tm[i]], [rate[i]], color=RED, size=11)
        pr.set_xlim(tm[0], tm[-1])
        mn = min(np.min(rate), 0)
        pr.set_ylim(1.2 * mn - 1e-3, 0.15 * abs(mn) + 1e-3)
        self.readout(el=float(ps["elev"][i]), delay=float(ps["delay"][i] * 1e3),
                     dop=float(dop_show[i]), rate=float(ps["doppler_rate"][i]))

    def on_listen(self, p):
        """The whistle: a tone that follows the Doppler curve, the whole pass in 6 seconds."""
        ps = self.ps
        fs = 22050
        n = int(6 * fs)
        tt = np.linspace(ps["t"][0], ps["t"][-1], n)
        d = np.interp(tt, ps["t"], ps["doppler"]) / max(self.peak, 1e-9)
        f_inst = 900 + 500 * d
        x = 0.4 * np.sin(2 * np.pi * np.cumsum(f_inst) / fs)
        x *= np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.05 * fs))
        self.play_audio(x, fs, "the Doppler whistle (pitch ∝ Doppler)")

    def story(self, p):
        r = self.r
        s = (f"<p>The red square crosses your sky (left) in {v((self.ps['t'][-1] - self.ps['t'][0]) / 60, '.1f', 'min')} "
             f"from horizon to horizon. Its distance changes fast, and so does its frequency: "
             f"approaching, the carrier arrives high; overhead, the shift passes through zero; "
             f"receding, it arrives low. At {v(p.f, '.3f', 'GHz')} the peak shift is "
             f"±{v(self.peak / 1e3, '.1f', 'kHz')}, about 23 parts per million for a 550 km orbit, "
             f"and it changes fastest overhead: up to {v(self.peak_rate / 1e3, '.2f', 'kHz/s')}.</p>")
        if p.comp:
            s += ("<p>With <b>pre-compensation</b> (3GPP NTN, Release 17) the terminal predicts the "
                  "Doppler from its GNSS position and the broadcast ephemeris and shifts its own "
                  "carrier: the green residual is what a 100 m position error leaves, a few hertz, "
                  "well inside what an ordinary LTE/NR receiver tolerates.</p>")
        else:
            s += ("<p>A receiver must acquire over that whole range and then track the rate with a "
                  "frequency-locked loop. Press <b>Listen</b> to hear the pass as a falling whistle "
                  "(the same sound radio amateurs hear on a 137 MHz weather satellite).</p>")
        return "<h3>The train whistle in the sky</h3>" + s + keybox(
            "Doppler f<sub>D</sub> = −f·ḋ/c: biggest near the horizon, zero and changing fastest "
            "overhead.")


# =============================================================================== 4. rain fade
class RainFade(Experiment):
    title = "Rain fade and availability"
    blurb = "ITU-R P.618 turns one rain statistic into the fade you must survive, and how often."
    book = "sec:ch23:linkbudget"
    controls = [
        LogSlider("f", "Frequency", 4, 50, 12, unit="GHz", fmt=".1f"),
        Slider("el", "Elevation", 5, 90, 35, step=1, unit="°"),
        Slider("r001", "Rain rate exceeded 0.01 %", 10, 120, 42, step=1, unit="mm/h",
               help="R0.01 from the ITU-R P.837 maps: 20–40 mm/h temperate, 60–150 tropical"),
        Slider("lat", "Latitude", 0, 60, 45, step=1, unit="°"),
        Heading("Receiver"),
        Toggle("sky", "Rain also adds sky noise", False),
        Slider("tsys", "Clear-sky system temperature", 50, 600, 86, step=1, unit="K"),
        Slider("margin", "Link margin", 0, 25, 3, step=0.1, unit="dB"),
    ]
    plots = [
        Plot("ccdf", "Fade exceeded for a percentage of the year", x="time exceeded (%)",
             y="C/N loss (dB)", logx=True, logy=True, xlim=(0.001, 5), ylim=(0.05, 80),
             legend="tr"),
        Plot("path", "The wet part of the path (P.618 geometry)", x="horizontal distance (km)",
             y="height (km)", xlim=(-0.5, 12), ylim=(-0.3, 6.5), legend=None),
        Plot("deg", "Rain hurts a downlink twice", x="rain attenuation A (dB)",
             y="C/N loss (dB)", xlim=(0, 12), ylim=(0, 16), legend="tl"),
    ]
    layout = [["ccdf", "ccdf"], ["path", "deg"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("m999", "Margin for 99.9 %", "dB", ".1f"),
        Readout("m9999", "Margin for 99.99 %", "dB", ".1f"),
        Readout("avail", "Availability with your margin", "%", ".3f"),
        Readout("hours", "Outage per year", "h", ".1f"),
    ]
    challenges = [
        Challenge("Ka band (19.7 GHz or more): set the margin that just buys 99.9 % availability.",
                  lambda s: s.p.f >= 19.6 and 99.9 <= s.r.avail < 99.92),
        Challenge("Quiet receiver, loud rain: at 12 GHz with Tsys ≤ 100 K, see the sky noise more "
                  "than double the 99.9 % margin compared with attenuation alone.",
                  lambda s: (abs(s.p.f - 12) < 0.3 and s.p.tsys <= 100 and s.p.sky
                             and s.r.m999 >= 2 * s.exp.a999)),
        Challenge("Find the frequency where 99.99 % availability needs more than 20 dB of margin "
                  "(35°, 42 mm/h).",
                  lambda s: (abs(s.p.el - 35) < 0.5 and abs(s.p.r001 - 42) < 0.5
                             and s.r.m9999 > 20 and s.p.f < 30)),
    ]
    pgrid = np.logspace(-3, np.log10(5), 300)

    def loss(self, A, p):
        A = np.asarray(A, float)
        if not p.sky:
            return A
        return A + DB((p.tsys + sat.rain_noise_temp(A)) / p.tsys)

    def update(self, p):
        A = sat.rain_atten_p618(p.f, p.el, self.pgrid, p.r001, lat_deg=p.lat)
        Lc = self.loss(A, p)
        A999 = float(sat.rain_atten_p618(p.f, p.el, 0.1, p.r001, lat_deg=p.lat))
        A9999 = float(sat.rain_atten_p618(p.f, p.el, 0.01, p.r001, lat_deg=p.lat))
        self.a999 = A999
        m999, m9999 = float(self.loss(A999, p)), float(self.loss(A9999, p))
        # availability: where the loss curve crosses the margin (loss falls with p)
        if p.margin <= Lc[-1]:
            pout = 5.0
        elif p.margin >= Lc[0]:
            pout = 0.001
        else:
            pout = float(10 ** np.interp(p.margin, Lc[::-1], np.log10(self.pgrid[::-1])))
        pc = self.plot("ccdf")
        pc.line("A", self.pgrid, A, color=GRAY, width=1.6, style="--", name="attenuation alone")
        pc.line("L", self.pgrid, Lc, color=NAVY, width=2.4,
                name="attenuation + sky noise" if p.sky else "C/N loss")
        for f_, col in ((12, BLUE), (20, GREEN), (30, ORANGE)):
            if abs(p.f - f_) > 0.5:
                ref = self.loss(sat.rain_atten_p618(f_, p.el, self.pgrid, p.r001, lat_deg=p.lat), p)
                pc.line(f"ref{f_}", self.pgrid, ref, color=col, width=1.0, alpha=0.5,
                        name=f"{f_} GHz")
        pc.hline("m", max(p.margin, 0.051), color=RED, style="--", label=f"your margin {p.margin:.1f} dB",
                 label_pos=0.02)
        pc.vline("p", pout, color=RED, style=":", label=f"{100 - pout:.3f} % available",
                 label_pos=0.15)
        pc.vline("p1", 0.1, color=GRAY, style=":", width=0.8)
        pc.vline("p2", 0.01, color=GRAY, style=":", width=0.8)
        # geometry cartoon
        pg = self.plot("path")
        hR = 4.36
        th = np.radians(p.el)
        xr = hR / np.tan(th)
        xmax = 12.0
        pg.line("ground", [-0.5, 12], [0, 0], color=GREEN, width=3)
        pg.fill_between("layer", np.array([-0.5, 12]), np.zeros(2), np.full(2, hR), color=BLUE,
                        alpha=0.08)
        pg.hline("hR", hR, color=BLUE, style="--", label="rain height ≈ 4.4 km", label_pos=0.6)
        xe = min(xmax, 6.4 / np.tan(th)) if th > 0 else xmax
        pg.line("dry", [min(xr, xmax), xe], [min(hR, xmax * np.tan(th)), xe * np.tan(th)],
                color=NAVY, width=2.0)
        pg.line("wet", [0, min(xr, xmax)], [0, min(hR, xmax * np.tan(th))], color=RED, width=3.5)
        pg.text("ls", min(xr, xmax) / 2 + 0.4, min(hR, xmax * np.tan(th)) / 2,
                f"wet path {hR / np.sin(th):.1f} km", color=RED, size=9, anchor=(0, 0))
        pg.scatter("dish", [0], [0], color=NAVY, size=10, symbol="t")
        # degradation vs attenuation
        Ag = np.linspace(0, 12, 121)
        pd = self.plot("deg")
        pd.line("a", Ag, Ag, color=GRAY, width=1.4, style="--", name="attenuation alone")
        for t0, col in ((86.0, NAVY), (195.0, GREEN), (500.0, ORANGE)):
            pd.line(f"t{t0}", Ag, Ag + DB((t0 + sat.rain_noise_temp(Ag)) / t0), color=col,
                    width=1.3, alpha=0.6, name=f"Tsys {t0:.0f} K")
        pd.scatter("now", [A999], [m999], color=RED, size=12, symbol="d", name="your 99.9 % fade")
        self.readout(m999=m999, m9999=m9999, avail=100 - pout, hours=pout / 100 * 8766)

    def story(self, p):
        r = self.r
        s = (f"<p>Raindrops a few millimetres across are a fair fraction of a wavelength above "
             f"10 GHz: they absorb and scatter. ITU-R P.618 turns the local rain climate (R0.01 = "
             f"{v(p.r001, '.0f', 'mm/h')}) and the wet length of your path (bottom left: only the "
             f"part below the rain height is wet, and low elevations make it long) into the fade "
             f"exceeded for any fraction of the year.</p>")
        s += (f"<p>At {v(p.f, '.1f', 'GHz')}: 99.9 % availability needs "
              f"{v(r.get('m999', 0), '.1f', 'dB')} of margin, 99.99 % needs "
              f"{v(r.get('m9999', 0), '.1f', 'dB')}. Your {v(p.margin, '.1f', 'dB')} buys "
              f"{v(r.get('avail', 0), '.3f', '%')}: about {v(r.get('hours', 0), '.1f', 'hours')} of "
              f"outage a year.</p>")
        if p.sky:
            s += (f"<p>A wet sky also <b>glows</b>: a fade of A dB adds 275·(1 − 10<sup>−A/10</sup>) K "
                  f"of noise. On a {v(p.tsys, '.0f', 'K')} receiver the 99.9 % fade of "
                  f"{v(self.a999, '.1f', 'dB')} costs {v(r.get('m999', 0), '.1f', 'dB')} of C/N in "
                  f"total: the quieter the receiver, the more of the 'rain loss' is rain noise.</p>")
        return "<h3>A wet window</h3>" + s + keybox(
            "No fixed margin covers the tail at Ka band and above: those systems adapt (ACM, "
            "uplink power control, site diversity) instead.")


# =============================================================================== 5. ACM vs CCM
MODS = sat.DVBS2_MODCODS
_THR = np.array([m[2] for m in MODS])
_ETA = np.array([m[1] for m in MODS])
_ORD = np.argsort(_THR)
THR_S = _THR[_ORD]
ETA_ENV = np.maximum.accumulate(_ETA[_ORD])          # best efficiency closing at each threshold


def acm_eff(cn, margin):
    """Best DVB-S2 efficiency closing at C/(N+I) = cn with the margin (vectorised)."""
    i = np.searchsorted(THR_S + margin, cn, side="right") - 1
    return np.where(i >= 0, ETA_ENV[np.maximum(i, 0)], 0.0), i


class ACMYear(Experiment):
    title = "ACM versus CCM: a year of rain"
    blurb = "Lock one MODCOD for the worst hour, or adapt frame by frame? Run both through a year."
    book = "sec:ch23:dvbs2"
    controls = [
        Heading("The Ka-band spot beam"),
        Slider("r001", "Rain climate R0.01", 10, 100, 42, step=1, unit="mm/h"),
        Slider("dish", "Terminal dish", 0.45, 1.8, 0.75, step=0.01, unit="m"),
        Heading("Constant coding and modulation"),
        Choice("design", "CCM designed for", ["99 %", "99.5 %", "99.9 %", "99.95 %"], "99.9 %"),
        Heading("Adaptive coding and modulation"),
        Slider("margin", "ACM margin", 0, 3, 0.5, step=0.1, unit="dB"),
        Slider("delay", "Feedback delay", 0, 10, 0, step=1, unit="min",
               help="How old the terminal's Es/N0 report is when the hub picks the MODCOD"),
        Button("again", "Another year of weather"),
    ]
    plots = [
        Plot("rain", "The rainiest week of the year", x="day", y="rain fade (dB)", legend="tr"),
        Plot("eff", "Spectral efficiency through that week", x="day", y="b/s/Hz",
             ylim=(-0.1, 4.6), legend="tr", legend_cols=2),
        Plot("year", "The whole year, sorted", x="percentage of the year (%)", y="b/s/Hz",
             logx=True, xlim=(0.003, 100), ylim=(0, 4.6), legend="tl"),
    ]
    layout = [["rain", "year"], ["eff", "year"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("acm", "ACM data per year", "TB", ",.0f"),
        Readout("ccm", "CCM data per year", "TB", ",.0f"),
        Readout("ratio", "ACM ÷ CCM", "×", ".2f"),
        Readout("avail", "ACM availability", "%", ".3f", good=lambda x: x >= 99.9),
    ]
    challenges = [
        Challenge("Make ACM deliver at least 4 times the CCM data volume.",
                  lambda s: s.exp.ratio_num >= 4,
                  hint="CCM must survive the design hour all year: make that hour worse."),
        Challenge("With a 5-minute feedback delay, win ACM's availability back to 99.5 % or better.",
                  lambda s: s.p.delay >= 5 and s.r.avail >= 99.5,
                  hint="An old report misses the start of a fade: spend some margin."),
        Challenge("Find a climate where CCM delivers at least half of ACM's volume.",
                  lambda s: s.exp.ratio_num <= 2.0),
    ]
    n_min = 365 * 24 * 60
    rs = 200e6

    def setup(self):
        self.seed = 23
        self._key = None
        f = 19.7e9
        _, self.el, self.d = sat.geo_look_angles(40.0, 0.0, 30.0)
        self.el, self.d = float(self.el), float(self.d)
        self.tclear = float(sat.system_noise_temp(60.0, sat.nf_to_temp(1.5), loss_db=0.3))
        self.f = f

    def on_again(self, p):
        self.seed += 1

    def year(self, r001):
        key = (self.seed, r001)
        if key != self._key:
            rng = np.random.default_rng(self.seed)
            a = np.exp(-1 / 30.0)
            x = lfilter([np.sqrt(1 - a ** 2)], [1, -a], rng.standard_normal(self.n_min))
            pe = 100 * norm.sf(x)
            A = np.zeros(self.n_min)
            m = pe < 5
            A[m] = sat.rain_atten_p618(19.7, self.el, np.clip(pe[m], 1e-3, 5), r001, lat_deg=40)
            self._A = A
            self._key = key
        return self._A

    def cnt(self, A, dish):
        G = float(sat.dish_gain_dbi(dish, self.f, 0.65))
        ts = sat.system_noise_temp(60.0 * 10 ** (-A / 10) + sat.rain_noise_temp(A),
                                   sat.nf_to_temp(1.5), loss_db=0.3)
        cn0 = 58.0 - float(sat.fspl_db(self.d, self.f)) - 0.8 - A + G - DB(ts) - sat.K_BOLTZ_DB
        return sat.combine_cn_db(np.asarray(cn0 - DB(self.rs), float), 20.0)

    def update(self, p):
        A = self.year(p.r001)
        cn = self.cnt(A, p.dish)
        d = int(p.delay)
        cn_rep = np.r_[np.full(d, cn[0]), cn[:len(cn) - d]] if d else cn
        eff_sel, idx = acm_eff(cn_rep, p.margin)
        thr_sel = np.where(idx >= 0, THR_S[np.maximum(idx, 0)], np.inf)
        # the frame fails if the true C/(N+I) is below the chosen MODCOD's threshold
        ok = cn >= thr_sel
        eff_acm = np.where(ok, eff_sel, 0.0)
        pdz = {"99 %": 1.0, "99.5 %": 0.5, "99.9 %": 0.1, "99.95 %": 0.05}[p.design]
        Ad = float(sat.rain_atten_p618(19.7, self.el, pdz, p.r001, lat_deg=40))
        ccm = sat.acm_select(float(self.cnt(np.array(Ad), p.dish)), 0.5)
        if ccm is None:
            eff_ccm = np.zeros_like(cn)
            ccm_name = "none closes"
        else:
            eff_ccm = np.where(cn >= ccm[2], ccm[1], 0.0)
            ccm_name = ccm[0]
        sec = 60.0
        tb = lambda e: float(np.sum(e) * self.rs * sec / 8 / 1e12)
        acm_tb, ccm_tb = tb(eff_acm), tb(eff_ccm)
        # rainiest week
        c = int(np.argmax(A))
        lo = max(0, c - 3 * 1440)
        hi = min(self.n_min, lo + 7 * 1440)
        sl = slice(lo, hi, 5)
        days = np.arange(self.n_min)[sl] / 1440
        pr = self.plot("rain")
        pr.line("A", days, A[sl], color=BLUE, width=1.4, fill=0, fill_alpha=0.2, name="rain fade")
        pr.set_xlim(days[0], days[-1])
        pr.set_ylim(0, max(2.0, 1.15 * A.max()))
        pe = self.plot("eff")
        pe.line("acm", days, eff_acm[sl], color=NAVY, width=1.8, step=False, name="ACM")
        pe.line("ccm", days, eff_ccm[sl], color=RED, width=1.8, style="--", name=f"CCM ({ccm_name})")
        pe.set_xlim(days[0], days[-1])
        # sorted year
        py = self.plot("year")
        srt = np.sort(eff_acm)
        pct = 100 * (np.arange(len(srt)) + 1) / len(srt)
        k = np.unique(np.r_[np.geomspace(1, len(srt), 1500).astype(int) - 1, len(srt) - 1])
        py.line("acm", pct[k], srt[k], color=NAVY, width=2.0, step=False, fill=0, fill_alpha=0.15,
                name="ACM (worst minutes on the left)")
        py.hline("ccm", ccm[1] if ccm else 0.0, color=RED, style="--",
                  label=f"CCM {ccm_name}", label_pos=0.55)
        py.vline("d", pdz, color=GRAY, style=":", label=f"design {p.design}", label_pos=0.9)
        self.ratio_num = acm_tb / ccm_tb if ccm_tb > 0 else np.inf
        self.readout(acm=acm_tb, ccm=ccm_tb,
                     ratio=self.ratio_num if ccm_tb > 0 else "∞ (CCM never closes)",
                     avail=100 * float(np.mean(eff_acm > 0)))
        self.ccm_name, self.cn_clear = ccm_name, float(self.cnt(np.array(0.0), p.dish))

    def story(self, p):
        r = self.r
        mc = sat.acm_select(self.cn_clear, p.margin)
        s = (f"<p>A synthetic year of one-minute rain fades at 19.7 GHz, with exactly the P.618 "
             f"statistics for R0.01 = {v(p.r001, '.0f', 'mm/h')} and storms that last as long as "
             f"real ones. In clear sky the terminal sees {v(self.cn_clear, '.1f', 'dB')} "
             f"C/(N+I): enough for {v(mc[0] if mc else 'nothing')}.</p>"
             f"<p><b>CCM</b> must pick one MODCOD that still closes in the {p.design} design hour: "
             f"{v(self.ccm_name)}, all year long, wasting the clear sky. <b>ACM</b> (the terminal "
             f"reports its Es/N0, the hub picks the best MODCOD frame by frame) steps down the "
             f"staircase only while a storm passes and climbs straight back. Over the year: "
             f"{v(r.get('acm', 0), ',.0f', 'TB')} versus {v(r.get('ccm', 0), ',.0f', 'TB')}, "
             f"{v(r.get('ratio', 0), '.2f', '×')} the traffic.</p>")
        if p.delay > 0:
            s += (f"<p>With a {v(int(p.delay), 'd')}-minute old report, the hub sometimes picks a "
                  f"MODCOD the fading link can no longer carry: availability "
                  f"{v(r.get('avail', 0), '.3f', '%')}. A little extra margin buys it back.</p>")
        return "<h3>Adapt to the weather</h3>" + s + keybox(
            "ACM recovers the clear-sky margin CCM wastes: about 3.7× the throughput in the book's "
            "Ka-band example, at the same availability.")


# =============================================================================== 6. TWTA
def predistort(x, ibo_db):
    """Ideal Saleh-inverse predistorter (amplitudes beyond saturation clip at saturation)."""
    q = sat.SALEH_TWTA
    r_sat = 1 / np.sqrt(q["ba"])
    a_sat = q["aa"] * r_sat / (1 + q["ba"] * r_sat ** 2)
    xs = x / np.sqrt(np.mean(np.abs(x) ** 2)) * r_sat * 10 ** (-ibo_db / 20)
    a_des = np.minimum(np.abs(xs) * q["aa"], a_sat * 0.999)
    r = (q["aa"] - np.sqrt(np.maximum(q["aa"] ** 2 - 4 * q["ba"] * a_des ** 2, 0))) / \
        (2 * q["ba"] * np.maximum(a_des, 1e-12))
    _, P = sat.saleh_am_am_pm(r)
    return r * np.exp(1j * (np.angle(xs) - P))


def through_twta(x, ibo, pd, ampm=True):
    if pd:
        xin = predistort(x, ibo)
        eff_ibo = -float(DB(np.mean(np.abs(xin) ** 2) / (1 / sat.SALEH_TWTA["ba"])))
        return sat.saleh_twta(xin, ibo_db=eff_ibo, am_pm=ampm)
    return sat.saleh_twta(x, ibo, am_pm=ampm)


TWTA_CONS = {"16QAM": (None, 10.21), "16APSK": (None, 10.21), "QPSK": (None, 4.03)}


def twta_points(name):
    if name == "16QAM":
        q = cl.get_constellation("16qam").points
        return q / np.sqrt(np.mean(np.abs(q) ** 2))
    if name == "16APSK":
        return sat.dvbs2_16apsk("3/4")
    return cl.get_constellation("qpsk").points


class TWTA(Experiment):
    title = "The TWTA: 16QAM vs 16APSK"
    blurb = "Drive the tube harder for more power, until its compression costs more than it gives."
    book = "sec:ch23:dvbs2"
    controls = [
        Choice("con", "Constellation", ["16QAM", "16APSK", "QPSK"], "16QAM"),
        Slider("ibo", "Input back-off", 0, 12, 8, step=0.25, unit="dB",
               help="Mean input power below the tube's single-carrier saturation"),
        Toggle("pd", "Predistorter", False,
               help="Applies the inverse of the tube's AM/AM and AM/PM before it"),
        Toggle("ampm", "AM/PM conversion", True,
               help="The tube's phase shift grows with drive (Saleh model)"),
    ]
    plots = [
        ConstellationPlot("const", "Received symbols (no noise)", lim=1.6),
        Plot("amam", "The tube, and where your signal drives it",
             x="input power rel. saturation (dB)", y="output (dB) / phase (×0.1°)",
             xlim=(-20, 8), ylim=(-22, 6), legend="tl"),
        Plot("td", "Total degradation = OBO + ΔEs/N0", x="input back-off (dB)",
             y="total degradation (dB)", xlim=(0, 12), ylim=(0, 10), legend="tr"),
    ]
    layout = [["const", "td"], ["amam", "td"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("obo", "Output back-off", "dB", ".2f"),
        Readout("mer", "MER at the receiver", "dB", ".1f"),
        Readout("td", "Total degradation", "dB", ".2f"),
        Readout("best", "Best input back-off", "dB", None),
    ]
    challenges = [
        Challenge("16QAM: find its sweet spot, within 0.2 dB of the minimum total degradation.",
                  lambda s: (s.p.con == "16QAM" and not s.p.pd and s.p.ampm
                             and s.r.td <= s.exp.td_min + 0.2 and s.exp.td_min < 50)),
        Challenge("Show why DVB-S2 chose APSK: 16APSK at a total degradation at least 0.5 dB "
                  "below 16QAM's best.",
                  lambda s: (s.p.con == "16APSK" and not s.p.pd and s.p.ampm
                             and s.r.td <= s.exp.best_td.get(("16QAM", False), -99) - 0.5)),
        Challenge("Constant envelope: drive QPSK to within 1 dB of saturation and keep the "
                  "total degradation under 1.5 dB.",
                  lambda s: s.p.con == "QPSK" and s.p.ibo <= 1 and s.r.td < 1.5),
    ]
    N, sps = 3000, 4
    ibos = np.arange(0, 12.01, 0.5)

    def setup(self):
        self.taps = cl.rrc_taps(0.2, self.sps, 10)
        rng = np.random.default_rng(18)
        self.idx = rng.integers(0, 16, self.N)
        self.curves = {}
        self.best_td = {}
        self.td_min = 99.0

    def run_ibo(self, name, ibo, pd, ampm, N=None):
        pts = twta_points(name)
        N = N or self.N
        s = pts[self.idx[:N] % len(pts)]
        x = cl.shape(s, self.taps, self.sps)
        y = through_twta(x, ibo, pd, ampm)
        obo = -float(DB(np.mean(np.abs(y) ** 2)))
        z = cl.matched_filter(y, self.taps)[len(self.taps) - 1::self.sps][:N][40:-40]
        ss = s[40:-40]
        g = np.vdot(ss, z) / np.vdot(ss, ss)
        mer = float(np.abs(g) ** 2 * np.mean(np.abs(ss) ** 2) / np.mean(np.abs(z - g * ss) ** 2))
        thr = 10 ** (TWTA_CONS[name][1] / 10)
        td = obo + float(DB(1 / (1 / thr - 1 / mer) / thr)) if mer > thr * 1.0001 else 99.0
        return dict(obo=obo, mer=float(DB(mer)), td=td, z=z / g, x=x)

    def update(self, p):
        r = self.run_ibo(p.con, p.ibo, p.pd, p.ampm)
        pts = twta_points(p.con)
        pc = self.plot("const")
        pc.points("z", r["z"][:1500], color=NAVY, size=3, alpha=0.4)
        pc.ideal("i", pts, color=RED, size=10)
        # AM/AM and AM/PM
        pa = self.plot("amam")
        rr = np.logspace(-2, 0.6, 300)
        rs = 1 / np.sqrt(sat.SALEH_TWTA["ba"])
        A, P = sat.saleh_am_am_pm(rr)
        As = sat.saleh_am_am_pm(rs)[0]
        pin = 20 * np.log10(rr / rs)
        pa.line("am", pin, 20 * np.log10(A / As), color=NAVY, width=2.2, name="AM/AM: output (dB)")
        pa.line("pm", pin, np.degrees(P) / 10 - 20,
                color=ORANGE, width=1.6, style="--", name="AM/PM: phase/10 − 20")
        inst = 20 * np.log10(np.abs(r["x"]) / np.sqrt(np.mean(np.abs(r["x"]) ** 2)) + 1e-9) - p.ibo
        h, e = np.histogram(inst, bins=60, range=(-20, 8))
        h = h / max(h.max(), 1) * 6
        pa.line("hist", np.r_[e[0], np.repeat(e[1:-1], 2), e[-1]], np.repeat(h, 2) - 22,
                color=GREEN, width=1.2, fill=-22, fill_alpha=0.25, name="your signal's power spread")
        pa.vline("mean", -p.ibo, color=GREEN, style=":", label="mean drive", label_pos=0.88)
        pa.vline("sat", 0, color=GRAY, style=":", label="saturation", label_pos=0.95)
        # TD curves (background)
        self.key = (p.con, p.pd, p.ampm)
        self.draw_td(p, r)
        self.readout(obo=r["obo"], mer=r["mer"], td=min(r["td"], 99.0))
        self.cur = r

    def draw_td(self, p, r=None):
        pt = self.plot("td")
        r = r or getattr(self, "cur", None)
        for k, (nm, col) in enumerate((("16QAM", RED), ("16APSK", NAVY), ("QPSK", GREEN))):
            for pd in (False, True):
                c = self.curves.get((nm, pd, p.ampm))
                if c is None:
                    continue
                sel = nm == p.con and pd == p.pd
                pt.line(f"{nm}{pd}", c["x"], np.minimum(c["td"], 20), color=col,
                        width=2.4 if sel else 1.1, style="--" if pd else "-",
                        alpha=1.0 if sel else 0.45,
                        name=f"{nm}{' + predistorter' if pd else ''}" if (sel or not pd) else None)
        c = self.curves.get(self.key)
        if c is not None and len(c["x"]) == len(self.ibos):
            i = int(np.argmin(c["td"]))
            self.td_min = float(c["td"][i])
            pt.scatter("min", [c["x"][i]], [c["td"][i]], color=GOLD, size=12, symbol="star",
                       name="minimum")
            self.readout(best=f"{c['x'][i]:.1f} dB")
        else:
            self.readout(best="…")
        if r is not None:
            pt.scatter("now", [p.ibo], [min(r["td"], 9.8)], color=ORANGE, size=14, symbol="d",
                       name="you")

    def background(self, p):
        order = [(p.con, p.pd, p.ampm)] + [(n, d, p.ampm) for n in ("16QAM", "16APSK", "QPSK")
                                           for d in (False, True)]
        for key in order:
            if key in self.curves and len(self.curves[key]["x"]) == len(self.ibos):
                continue
            nm, pd, ampm = key
            out = dict(x=[], td=[])
            for ibo in self.ibos:
                rr = self.run_ibo(nm, ibo, pd, ampm, N=1500 if self.quick else self.N)
                out["x"].append(ibo)
                out["td"].append(rr["td"])
            yield key, dict(x=np.array(out["x"]), td=np.array(out["td"]))

    def progress(self, p, item):
        key, c = item
        self.curves[key] = c
        self.best_td[(key[0], key[1])] = float(np.min(c["td"]))
        self.draw_td(p)

    def story(self, p):
        r = self.r
        s = ("<p>A travelling-wave tube is most efficient at saturation, where it squashes "
             "amplitude (AM/AM) and twists phase (AM/PM). The green histogram (bottom left) shows "
             "how much of your RRC-filtered signal is driven into that region.</p>")
        s += (f"<p>At {v(p.ibo, '.2f', 'dB')} input back-off the output is "
              f"{v(r.get('obo', 0), '.2f', 'dB')} below saturation and the distortion leaves an MER "
              f"of {v(r.get('mer', 0), '.1f', 'dB')}. The figure of merit is the <b>total "
              f"degradation</b>: output power given up plus the extra Es/N0 the receiver needs to "
              f"make up for the distortion, here {v(r.get('td', 0), '.2f', 'dB')}. Too much back-off "
              f"wastes the tube; too little drowns the link in distortion: the bowl has a "
              f"minimum.</p>")
        if p.con == "16QAM":
            s += ("<p>16QAM has three amplitude levels; the outer corners compress and rotate "
                  "relative to the inner points.</p>")
        elif p.con == "16APSK":
            s += ("<p>16APSK has only two rings (4 + 12 points), so compression just shrinks a "
                  "ring, which the receiver can learn: its bowl is lower and sits closer to "
                  "saturation. That is why DVB-S2 chose APSK.</p>")
        else:
            s += ("<p>QPSK has one ring: a constant-envelope constellation the tube can hardly "
                  "hurt, apart from the filtering's small envelope ripple.</p>")
        if p.pd:
            s += ("<p>The predistorter applies the inverse curves before the tube. It straightens the "
                  "response below saturation, but nothing can create power beyond it.</p>")
        return "<h3>Finding the sweet spot</h3>" + s + keybox(
            "TD = OBO + ΔEs/N0. Fewer amplitude rings (APSK) and predistortion move the optimum "
            "closer to saturation.")


# =============================================================================== 7. Voyager
GROUND = {"70 m dish": (70.0, 1.0), "34 m dish": (34.0, 1.0), "4 × 34 m array": (34.0, 4.0)}
PLANETS = [("Mars (max)", 2.67), ("Jupiter", 5.2), ("Saturn", 9.5), ("Neptune", 30.0),
           ("Voyager 2", 141.0), ("Voyager 1", 173.0)]


class Voyager(Experiment):
    title = "Voyager: one light-day away"
    blurb = "20 watts, 173 astronomical units, a 319 dB withdrawal, and still a few dB to spare."
    book = "sec:ch23:deepspace"
    controls = [
        LogSlider("au", "Distance", 0.5, 200, 173, unit="AU", fmt=".1f"),
        Heading("Spacecraft"),
        Slider("pw", "Transmitter power", 1, 100, 20, step=1, unit="W"),
        Slider("dsc", "High-gain dish", 0.5, 5.0, 3.7, step=0.1, unit="m"),
        Choice("band", "Band", ["X (8.42 GHz)", "Ka (32 GHz)"]),
        Heading("Ground and data"),
        Choice("gnd", "Deep Space Network antenna", list(GROUND), "70 m dish"),
        LogSlider("rate", "Data rate", 10, 1e6, 160, unit="b/s", fmt=",.0f"),
    ]
    plots = [
        BarPlot("bank", "The big transactions", y="account balance (dB)"),
        BarPlot("spend", "What is left to spend", y="dB"),
        Plot("dist", "Supportable data rate vs distance (margin 0)", x="distance (AU)",
             y="bit rate (b/s)", logx=True, logy=True, xlim=(0.5, 450), ylim=(1, 1e8), legend="bl"),
    ]
    layout = [["bank", "spend"], ["dist", "dist"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("cn0", "C/N₀", "dB-Hz", ".1f"),
        Readout("ebn0", "Eb/N₀ after losses", "dB", ".1f"),
        Readout("margin", "Margin", "dB", ".1f", good=lambda x: x >= 0),
        Readout("light", "Light time, one way", "h", ".1f"),
    ]
    challenges = [
        Challenge("At one light-day with a single 34 m dish, find the highest data rate that "
                  "still closes (margin 0 to 1 dB).",
                  lambda s: (s.p.au >= 170 and s.p.gnd == "34 m dish" and s.p.band.startswith("X")
                             and abs(s.p.pw - 20) < 0.5 and abs(s.p.dsc - 3.7) < 0.05
                             and 0 <= s.r.margin <= 1)),
        Challenge("At Jupiter (5.2 AU), get the Voyager hardware above 300 kb/s with a positive "
                  "margin.",
                  lambda s: (abs(s.p.au - 5.2) < 0.3 and s.p.rate >= 3e5 and s.r.margin >= 0
                             and abs(s.p.pw - 20) < 0.5 and abs(s.p.dsc - 3.7) < 0.05)),
        Challenge("Move to Ka band and support at least 1 kb/s from 173 AU with the 70 m dish.",
                  lambda s: (s.p.au >= 170 and s.p.band.startswith("Ka") and s.p.gnd == "70 m dish"
                             and s.p.rate >= 1000 and s.r.margin >= 0
                             and abs(s.p.pw - 20) < 0.5 and abs(s.p.dsc - 3.7) < 0.05)),
    ]
    LOSS, REQ = 4.0, 2.5

    def link(self, p, au=None):
        au = p.au if au is None else au
        ka = p.band.startswith("Ka")
        f = 32e9 if ka else 8.42e9
        d = np.asarray(au, float) * sat.AU
        eirp = 10 * np.log10(p.pw) + float(sat.dish_gain_dbi(p.dsc, f, 0.55))
        D, n = GROUND[p.gnd]
        tsys = 30.0 if ka else 20.0
        g = float(sat.dish_gain_dbi(D, f, 0.70)) + 10 * np.log10(n)
        gt = g - float(DB(tsys))
        atm = 1.0 if ka else 0.0
        L = sat.fspl_db(d, f)
        cn0 = eirp - L - atm + gt + 228.6
        return dict(eirp=eirp, L=L, gt=gt, atm=atm, cn0=cn0, f=f)

    def update(self, p):
        k = self.link(p)
        L, cn0 = float(k["L"]), float(k["cn0"])
        pb = self.plot("bank")
        items = [("EIRP", k["eirp"]), ("free space", -L)]
        if k["atm"]:
            items.append(("weather", -k["atm"]))
        items += [("DSN G/T", k["gt"]), ("−k", 228.6)]
        waterfall(pb, "b", items, "C/N₀")
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        pb.set_ylim(-300, 120)
        ps = self.plot("spend")
        rdb = 10 * np.log10(p.rate)
        margin = cn0 - rdb - self.LOSS - self.REQ
        steps = [("C/N₀", cn0), (f"{p.rate:,.0f} b/s", -rdb), ("losses", -self.LOSS),
                 ("code needs", -self.REQ)]
        run = 0.0
        for i, (nm, val) in enumerate(steps):
            lo, hi = sorted([run, run + val])
            ps.bars(f"s{i}", [i], [hi], base=lo, width=0.6, color=NAVY if i == 0 else RED)
            ps.text(f"st{i}", i, hi, f"{val:+.1f}", anchor=(0.5, 1.05), size=8.5, bold=True)
            run += val
        lo, hi = sorted([0, margin])
        ps.bars("sT", [4], [hi], base=lo, width=0.6, color=GREEN if margin >= 0 else RED)
        ps.text("stT", 4, hi, f"{margin:+.1f}", anchor=(0.5, 1.05), size=9.5, bold=True)
        ps.set_xticks([(i, s) for i, (s, _) in enumerate(steps)] + [(4, "margin")])
        ps.set_xlim(-0.6, 4.6)
        ps.set_ylim(min(-5, margin - 5), max(cn0, 10) * 1.15 + 3)
        ps.hline("z", 0, color=GRAY, style="-", width=0.8)
        # rate vs distance
        pdist = self.plot("dist")
        au = np.logspace(np.log10(0.5), np.log10(250), 200)
        for gname, col in (("70 m dish", NAVY), ("34 m dish", GREEN), ("4 × 34 m array", ORANGE)):
            pp = st.Params(p, gnd=gname)
            rr = 10 ** ((self.link(pp, au)["cn0"] - self.LOSS - self.REQ) / 10)
            pdist.line(gname, au, rr, color=col, width=2.4 if gname == p.gnd else 1.2,
                       alpha=1.0 if gname == p.gnd else 0.5, name=gname)
        for j, (nm, x) in enumerate(PLANETS):
            pdist.vline(f"pl{j}", x, color=GRAY, style=":", width=0.8, label=nm,
                        label_pos=0.93 - 0.07 * (j % 3))
        pdist.scatter("now", [p.au], [p.rate], color=GREEN if margin >= 0 else RED, size=14,
                      symbol="d", name="your link")
        self.readout(cn0=cn0, ebn0=cn0 - rdb - self.LOSS, margin=margin,
                     light=p.au * sat.AU / sat.C / 3600)
        self.k = k

    def story(self, p):
        k, r = self.k, self.r
        s = (f"<p>Voyager 1 was launched in 1977 with a 20 W X-band transmitter and a 3.7 m dish. "
             f"At {v(p.au, '.0f', 'AU')} its signal takes {v(r.get('light', 0), '.1f', 'hours')} to "
             f"reach us, and free space takes a withdrawal of {v(float(k['L']), '.1f', 'dB')}: a "
             f"factor of 10<sup>{float(k['L']) / 10:.0f}</sup>, the largest in this book.</p>"
             f"<p>Only the giant ears of the Deep Space Network (G/T {v(k['gt'], '.1f', 'dB/K')}, "
             f"cryogenic amplifiers at about 20 K) bring the balance back to "
             f"{v(r.get('cn0', 0), '.1f', 'dB-Hz')}. Spend {v(10 * np.log10(p.rate), '.1f', 'dB')} "
             f"on {v(p.rate, ',.0f', 'b/s')}, {v(self.LOSS, '.0f', 'dB')} on carrier tracking, "
             f"pointing and weather, {v(self.REQ, '.1f', 'dB')} on what the concatenated code needs, "
             f"and the margin is {v(r.get('margin', 0), '+.1f', 'dB')}"
             + (": " + good("the link closes.") if r.get("margin", 0) >= 0 else ": " + bad("silence."))
             + "</p>")
        s += ("<p>The rate falls as 1/d²: 20 dB per decade of distance (bottom). The escapes are "
              "bigger or arrayed antennas, higher frequencies (a fixed dish has more gain at shorter "
              "wavelengths), more power, better codes, or light itself: NASA's DSOC laser sent "
              "hundreds of Mb/s from tens of millions of kilometres in 2023.</p>")
        return "<h3>The most extreme bank account</h3>" + s + keybox(
            "Received power here is a few times 10⁻¹⁹ W, and the link still closes every day.")


# =============================================================================== the lab
LAB = st.Lab(18, "Satellite Links", chapter=23, chapter_title="Satellite Communications",
             experiments=[LinkBudgetBank, GeoVsLeo, LEOPass, RainFade, ACMYear, TWTA, Voyager])

if __name__ == "__main__":
    st.run(LAB)
