"""Lab 22 · Propagation and the Link Budget: a Coverage Planner   (Chapter 11)

Run it:      python labs/lab22_propagation_link_budget.py
Self-test:   python labs/lab22_propagation_link_budget.py --selftest

Every radio project starts with a spreadsheet: transmit power, antenna gains, losses, the
noise floor, the SNR the modem needs, and the margin statistics demand. What you can afford
to lose is the maximum allowable path loss; a propagation model turns it into a cell radius
and a site count. Eight experiments make every line of that spreadsheet live: Friis and the
antenna, the "decibel bank account", Fresnel zones and knife edges, the two-ray breakpoint,
Hata/COST-231/38.901, rain and oxygen, edge versus area coverage, and a coverage map.
Library code: commlib.propagation (the chapter's calculators) and commlib.fading.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.special import j1
from scipy.stats import norm

import commlib as cl
from commlib import fading as fdg
from commlib import propagation as pr
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, PolarPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

C0 = pr.C0


def dish_pattern_db(theta, diam, f_hz):
    """Uniform circular aperture: (2 J1(u)/u)^2 in dB, u = pi D sin(theta) / lambda."""
    u = np.pi * diam / pr.wavelength(f_hz) * np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.abs(u) < 1e-9, 1.0, (2 * j1(u) / np.where(np.abs(u) < 1e-9, 1, u)) ** 2)
    return 10 * np.log10(g + 1e-12)


# =============================================================================== 1. Friis
class Friis(Experiment):
    title = "Friis: free space and antennas"
    blurb = "Does path loss really grow with frequency? It depends on what you hold fixed."
    book = "sec:ch11:friis"
    controls = [
        Slider("ptx", "Transmit power", 0, 50, 30, step=1, unit="dBm"),
        LogSlider("d", "Distance", 0.01, 100, 1.0, unit="km"),
        LogSlider("f", "Frequency", 0.1, 100, 0.9, unit="GHz"),
        Heading("Antennas (both ends)"),
        Choice("ant", "Antenna type", ["Fixed gain", "Fixed aperture"]),
        Slider("g", "Gain", 0, 40, 0, step=0.5, unit="dBi", enabled_if=lambda p: p.ant == "Fixed gain"),
        LogSlider("diam", "Dish diameter", 0.05, 3.0, 0.3, unit="m",
                  enabled_if=lambda p: p.ant == "Fixed aperture"),
    ]
    plots = [
        Plot("dist", "Received power vs distance", x="distance (km)", y="received power (dBm)",
             logx=True, xlim=(0.01, 100), ylim=(-160, 20), legend=None),
        Plot("freq", "Received power vs frequency, at your distance", x="frequency (GHz)",
             y="received power (dBm)", logx=True, xlim=(0.1, 100), legend="bl"),
        PolarPlot("pat", "Antenna pattern (gain re peak)", floor_db=-40, legend=None),
    ]
    layout = [["dist", "pat"], ["freq", "pat"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("fspl", "Free-space path loss", "dB", ".1f"),
        Readout("prx", "Received power", "dBm", ".1f"),
        Readout("gain", "Antenna gain (each)", "dBi", ".1f"),
        Readout("delta", "Change vs 900 MHz", "dB", "+.1f",
                help="Received power at this frequency minus at 900 MHz, same antennas and distance"),
    ]
    challenges = [
        Challenge("Fixed-gain antennas: lose about 30 dB by moving from 900 MHz to 28 GHz.",
                  lambda s: s.p.ant == "Fixed gain" and abs(s.p.f - 28) < 1 and s.r.delta < -29),
        Challenge("Fixed apertures: make the higher frequency win by at least 20 dB.",
                  lambda s: s.p.ant == "Fixed aperture" and s.r.delta >= 20,
                  hint="Gain of a dish grows as f², twice (both ends); FSPL grows as f² once."),
        Challenge("Point-to-point backhaul: −70 dBm or better over 50 km with 30 dBm and dishes no "
                  "larger than 0.3 m.",
                  lambda s: s.p.ant == "Fixed aperture" and s.p.d >= 49.9 and s.p.diam <= 0.301
                  and s.p.ptx <= 30 and s.r.prx >= -70),
    ]
    eff = 0.65

    def gain_db(self, p, f_hz):
        if p.ant == "Fixed gain":
            return np.full(np.shape(f_hz), float(p.g))
        return pr.dish_gain_dbi(p.diam, f_hz, self.eff)

    def update(self, p):
        f = p.f * 1e9
        G = float(self.gain_db(p, f))
        d = np.logspace(-2, 2, 300)
        prx_d = p.ptx + 2 * G - pr.fspl_db(d * 1e3, f)
        pd = self.plot("dist")
        pd.line("c", d, prx_d, color=NAVY, width=2.2)
        fspl = float(pr.fspl_db(p.d * 1e3, f))
        prx = p.ptx + 2 * G - fspl
        pd.scatter("now", [p.d], [prx], color=RED, size=12, symbol="d")
        for k, dd in enumerate((0.1, 1, 10)):
            pd.vline(f"dec{k}", dd, color=GRAY, style=":", width=0.8)
        pd.text("lab", 0.012, prx_d[0] - 50, "−20 dB per decade\nof distance", color=GRAY, size=8.5)
        fr = np.logspace(-1, 2, 300)
        pf = self.plot("freq")
        pf.line("gain", fr, p.ptx + 2 * p.g - pr.fspl_db(p.d * 1e3, fr * 1e9), color=GRAY,
                width=1.8 if p.ant == "Fixed gain" else 1.2, style="-" if p.ant == "Fixed gain" else "--",
                name=f"fixed gain ({p.g:g} dBi)")
        pf.line("ap", fr, p.ptx + 2 * pr.dish_gain_dbi(p.diam, fr * 1e9, self.eff)
                - pr.fspl_db(p.d * 1e3, fr * 1e9), color=GREEN,
                width=1.8 if p.ant == "Fixed aperture" else 1.2,
                style="-" if p.ant == "Fixed aperture" else "--", name=f"fixed aperture ({p.diam:.2g} m)")
        pf.scatter("now", [p.f], [prx], color=RED, size=12, symbol="d")
        pf.vline("ref", 0.9, color=GRAY, style=":", label="900 MHz", label_pos=0.92)
        lo = min(p.ptx + 2 * p.g - pr.fspl_db(p.d * 1e3, 100e9), prx) - 10
        hi = max(p.ptx + 2 * pr.dish_gain_dbi(p.diam, 100e9, self.eff) - pr.fspl_db(p.d * 1e3, 100e9),
                 p.ptx + 2 * p.g - pr.fspl_db(p.d * 1e3, 0.1e9), prx) + 10
        pf.set_ylim(lo, hi)
        # pattern of the equivalent aperture
        lam = pr.wavelength(f)
        D = p.diam if p.ant == "Fixed aperture" else lam / np.pi * np.sqrt(10 ** (G / 10) / self.eff)
        th = np.linspace(-np.pi / 2, np.pi / 2, 1441)
        pp = self.plot("pat")
        if G < 3:
            pp.pattern("p", np.linspace(-np.pi, np.pi, 361), np.zeros(361), color=NAVY, fill=True)
        else:
            pp.pattern("p", th, dish_pattern_db(th, D, f), color=NAVY, fill=True)
        bw = float(pr.beamwidth_deg(D, f)) if G >= 3 else 360.0
        pp.set_title("Antenna pattern: isotropic" if G < 3 else
                     f"Antenna pattern: beamwidth ≈ {bw:.3g}°")
        G9 = float(self.gain_db(p, 0.9e9))
        prx9 = p.ptx + 2 * G9 - float(pr.fspl_db(p.d * 1e3, 0.9e9))
        self.readout(fspl=fspl, prx=prx, gain=G, delta=prx - prx9)

    def story(self, p):
        lam = C0 / (p.f * 1e9)
        s = (f"<p>Free-space path loss is 20·log₁₀(4πd/λ) = {v(self.r.get('fspl', 0), '.1f', 'dB')} "
             f"at {v(p.d, '.3g', 'km')} and {v(p.f, '.3g', 'GHz')}: every tenfold increase of "
             f"distance costs exactly 20 dB, because the power spreads over a sphere of area 4πd².</p>")
        if p.ant == "Fixed gain":
            s += ("<p>The λ² in the formula is not the air's fault: it is the shrinking <b>effective "
                  "area</b> of an antenna with fixed gain, A = Gλ²/4π. A 0 dBi antenna at 28 GHz is a "
                  "tiny thing that catches little of the passing wave, hence the 30 dB penalty.</p>")
        else:
            s += (f"<p>A {v(p.diam, '.2g', 'm')} dish has a fixed area, so its gain η(πD/λ)² grows as f²: "
                  f"{v(self.r.get('gain', 0), '.1f', 'dBi')} here, at both ends. Between two fixed "
                  f"apertures, higher frequency <b>wins</b>, which is why backhaul climbs to E-band. "
                  f"The price is a pencil beam (right) that must be pointed precisely.</p>")
        return "<h3>What λ² really means</h3>" + s + keybox(
            "P_r = P_t + G_t + G_r − FSPL (dB). Path loss rises with f only if you hold the antenna "
            "gain fixed; hold the antenna size fixed and it falls.")


# =============================================================================== 2. link budget
class LinkBudget(Experiment):
    title = "The decibel bank account"
    blurb = "Income, expenses, savings: balance the books and read off the cell radius."
    book = "sec:ch11:linkbudget"
    controls = [
        Heading("Phone (uplink transmitter)"),
        Slider("ptx", "Transmit power", 0, 30, 23, step=0.5, unit="dBm"),
        Slider("gtx", "Antenna gain − body loss", -10, 6, -3, step=0.5, unit="dB"),
        Heading("Base station receiver"),
        Slider("grx", "Antenna / array gain", 0, 30, 18, step=0.5, unit="dBi"),
        Slider("feeder", "Feeder loss", 0, 6, 2, step=0.5, unit="dB"),
        Slider("nf", "Noise figure", 0.5, 10, 2, step=0.1, unit="dB"),
        LogSlider("bw", "Bandwidth", 180, 20000, 360, unit="kHz"),
        Slider("sinr", "Required SINR", -10, 25, -4, step=0.5, unit="dB"),
        Heading("Margins"),
        Slider("interf", "Interference margin", 0, 8, 2, step=0.5, unit="dB"),
        Slider("sigma", "Shadowing σ", 2, 12, 8, step=0.5, unit="dB"),
        Slider("area", "Area coverage target", 50, 99, 95, step=1, unit="%"),
        Slider("pen", "Building penetration", 0, 30, 15, step=0.5, unit="dB"),
        Heading("Propagation"),
        Slider("f", "Frequency", 700, 3800, 1800, step=50, unit="MHz"),
        Choice("model", "Model", ["COST-231 Hata", "TR 38.901 UMa NLOS"], style="menu"),
        Slider("hbs", "Mast height", 10, 60, 30, step=1, unit="m"),
    ]
    plots = [
        BarPlot("wf", "The budget as a waterfall (defaults = the book's LTE uplink table)",
                y="level (dBm)", ylim=(-140, 52)),
        Plot("pl", "Where the budget runs out", x="distance (km)", y="path loss (dB)", logx=True,
             xlim=(0.05, 20), ylim=(90, 170), legend="tl"),
    ]
    layout = [["wf"], ["pl"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("sens", "Receiver sensitivity", "dBm", ".1f"),
        Readout("mapl", "Max allowable path loss", "dB", ".1f"),
        Readout("R", "Cell radius", "km", ".2f"),
        Readout("sites", "Sites per 100 km²", "", ".0f"),
    ]
    challenges = [
        Challenge("Outdoor users only: drop the building loss and reach a radius of at least 1.8 km.",
                  lambda s: s.p.pen == 0 and s.r.R >= 1.8),
        Challenge("Keep 15 dB building loss, 23 dBm and 1800 MHz, but reach a 1 km radius by "
                  "improving the base station.",
                  lambda s: s.p.pen >= 15 and s.p.ptx <= 23 and s.p.f == 1800 and s.r.R >= 1.0,
                  hint="Each dB of noise figure, antenna gain or feeder loss is a dB of MAPL."),
        Challenge("Move to 3500 MHz (TR 38.901, 25 m mast, 15 dB building loss) and still cover 0.9 km: "
                  "buy it back with a massive-MIMO array gain (2 dB feeder, NF 2 dB kept).",
                  lambda s: s.p.f == 3500 and s.p.model.startswith("TR") and s.p.pen >= 15
                  and s.p.hbs == 25 and s.p.feeder >= 2 and s.p.nf >= 2 and s.r.R >= 0.9),
    ]

    def path_loss(self, p, dkm):
        dkm = np.asarray(dkm, float)
        if p.model == "COST-231 Hata":
            if p.f <= 1500:
                return pr.hata(p.f, p.hbs, 1.5, dkm)
            return pr.cost231(p.f, p.hbs, 1.5, dkm, 0)
        return pr.pl_38901("UMa", dkm * 1e3, p.f / 1e3, p.hbs)[1]

    def slope_n(self, p):
        if p.model == "COST-231 Hata":
            return (44.9 - 6.55 * np.log10(p.hbs)) / 10
        return 3.908

    def update(self, p):
        noise = float(pr.thermal_noise_dbm(p.bw * 1e3, p.nf))
        sens = noise + p.sinr
        n = self.slope_n(p)
        try:
            margin, pedge = pr.edge_margin_for_area(p.area / 100, n, p.sigma)
        except ValueError:
            margin, pedge = 0.0, 0.5
        eirp = p.ptx + p.gtx
        steps = [("Phone\npower", p.ptx, GREEN), ("antenna\n−body", p.gtx, RED if p.gtx < 0 else GREEN),
                 ("BS\nantenna", p.grx, GREEN), ("feeder", -p.feeder, RED),
                 ("interf.\nmargin", -p.interf, ORANGE), ("shadow\nmargin", -margin, ORANGE),
                 ("building", -p.pen, ORANGE)]
        level = 0.0
        lows, highs, cols, labels, vals = [], [], [], [], []
        for name, val, col in steps:
            a, b = (0.0, val) if name.startswith("Phone") else (level, level + val)
            lows.append(min(a, b)); highs.append(max(a, b)); cols.append(col)
            labels.append(name); vals.append(val)
            level = b if not name.startswith("Phone") else val
        mapl = level - sens
        lows.append(sens); highs.append(level); cols.append(NAVY); labels.append("path loss\n= MAPL")
        vals.append(-mapl)
        pw = self.plot("wf")
        x = np.arange(len(lows))
        pw.bars("b", x, np.array(highs), base=np.array(lows), width=0.62, colors=cols)
        for i, (lo, hi, val) in enumerate(zip(lows, highs, vals)):
            pw.text(f"t{i}", i, hi + 2, f"{val:+.1f}", color=cols[i], size=8.5, anchor=(0.5, 1), bold=True)
        pw.hline("noise", noise, color=GRAY, style=":", label=f"noise floor {noise:.1f} dBm",
                 label_pos=0.01)
        pw.hline("sens", sens, color=RED, style="--", label=f"sensitivity {sens:.1f} dBm",
                 label_pos=0.72)
        pw.set_xticks([(i, s.replace("\n", " ")) for i, s in enumerate(labels)])
        pw.set_xlim(-0.6, len(lows) - 0.4)
        d = np.logspace(np.log10(0.05), np.log10(20), 300)
        L = self.path_loss(p, d)
        ok = np.flatnonzero(L <= mapl)
        R = float(np.interp(mapl, L, d)) if len(ok) and L[-1] > mapl and L[0] < mapl else (
            d[-1] if len(ok) == len(d) else d[0])
        pp = self.plot("pl")
        pp.line("L", d, L, color=NAVY, width=2.2, name=p.model)
        pp.hline("mapl", mapl, color=RED, style="--", label=f"MAPL {mapl:.1f} dB", label_pos=0.02)
        pp.vline("R", R, color=GREEN, style="--", label=f"R = {R:.2f} km", label_pos=0.9)
        self.margin, self.pedge = margin, pedge
        self.readout(sens=sens, mapl=mapl, R=R, sites=100 / (2.598 * R ** 2))

    def story(self, p):
        R = self.r.get("R", 0.7)
        s = (f"<p>Read the waterfall left to right like a bank statement. The phone starts with "
             f"{v(p.ptx, '.0f', 'dBm')}; antennas add, cables and margins subtract. What is left "
             f"above the receiver's <b>sensitivity</b> (noise floor + NF + required SINR = "
             f"{v(self.r.get('sens', 0), '.1f', 'dBm')}) is what the path may take: the "
             f"<b>maximum allowable path loss</b>, {v(self.r.get('mapl', 0), '.1f', 'dB')}. The "
             f"navy bar dwarfs everything else.</p>")
        s += (f"<p>The orange bars are the price of an uncertain world: {v(self.margin, '.1f', 'dB')} "
              f"of shadowing margin buys {v(p.area, '.0f', '%')} area coverage "
              f"({v(100 * self.pedge, '.0f', '%')} at the cell edge), and walls take "
              f"{v(p.pen, '.0f', 'dB')}.</p>")
        s += (f"<p>The model turns the MAPL into a radius of {v(R, '.2f', 'km')}, i.e. "
              f"{v(self.r.get('sites', 0), '.0f')} sites per 100 km². With a slope of about 35 dB per "
              f"decade, every decibel is worth about 7 % of range and 14 % of the site count.</p>")
        return "<h3>Balance the books</h3>" + s + keybox(
            "MAPL = EIRP + G_r − L_r − sensitivity − margins. The uplink (a 200 mW phone) almost "
            "always limits coverage, so it is the budget engineers draw first.")


# =============================================================================== 3. knife edge
class KnifeEdge(Experiment):
    title = "Fresnel zones and knife edges"
    blurb = "Radio needs more than a line of sight: it needs a fat, football-shaped clearance."
    book = "sec:ch11:mechanisms"
    controls = [
        Slider("D", "Path length", 2, 60, 30, step=1, unit="km"),
        LogSlider("f", "Frequency", 0.5, 40, 6.0, unit="GHz"),
        Slider("hant", "Antenna heights (both ends)", 10, 100, 45, step=1, unit="m"),
        Heading("Obstacle"),
        Slider("hobs", "Tree line / hill height", 0, 80, 20, step=1, unit="m"),
        Slider("pos", "Obstacle position", 0.05, 0.95, 0.5, step=0.01, unit="of path"),
        Slider("k", "Effective Earth factor k", 0.5, 3.0, 4 / 3, step=0.01,
               help="Refraction bends the rays: k = 4/3 is the standard atmosphere, k < 1 "
                    "(sub-refraction) makes the Earth bulge more"),
    ]
    plots = [
        Plot("prof", "Path profile and the first Fresnel zone", x="distance (km)", y="height (m)",
             legend="tl", legend_cols=3),
        Plot("J", "Knife-edge diffraction loss", x="Fresnel parameter ν", y="gain re free space (dB)",
             xlim=(-3, 5), ylim=(-30, 4), legend="bl"),
    ]
    layout = [["prof", "prof"], ["J", "J"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("clr", "Worst clearance", "× r₁", ".2f", good=lambda x: x >= 0.6),
        Readout("nu", "Fresnel parameter ν", "", ".2f"),
        Readout("loss", "Diffraction loss", "dB", ".1f", good=lambda x: x < 1),
        Readout("r1", "r₁ at mid-path", "m", ".1f"),
    ]
    challenges = [
        Challenge("Graze the ray: make the diffraction loss 6 dB ± 0.5 dB.",
                  lambda s: abs(s.r.loss - 6) <= 0.5),
        Challenge("Sub-refraction (k ≤ 0.67) with the book's 20 m trees at mid-path: keep the loss "
                  "under 1 dB with towers of 60 m or less.",
                  lambda s: s.p.k <= 0.67 and s.p.hobs >= 20 and abs(s.p.pos - 0.5) < 0.06
                  and s.p.hant <= 60 and s.r.loss < 1, hint="Shorter hops have a smaller bulge."),
        Challenge("Make the loss exceed 20 dB while the obstacle pokes no more than 10 m above the "
                  "line of sight.",
                  lambda s: s.r.loss > 20 and 0 < s.exp.h_above <= 10,
                  hint="ν grows with √f: the same hill costs much more at millimetre waves."),
    ]
    width_km = 1.5

    def update(self, p):
        D = p.D * 1e3
        x = np.linspace(0, D, 801)
        bulge = pr.earth_bulge(x, D - x, p.k)
        f = p.f * 1e9
        lam = pr.wavelength(f)
        r1 = pr.fresnel_radius(np.maximum(x, 1), np.maximum(D - x, 1), f)
        obst = bulge + np.where(np.abs(x - p.pos * D) < self.width_km * 500, p.hobs, 0.0)
        los = np.full_like(x, p.hant)
        h = obst - los                                    # > 0: obstruction above the ray
        inner = (x > 0.02 * D) & (x < 0.98 * D)
        ratio = np.where(inner, h / np.maximum(r1, 1e-9), -np.inf)
        i = int(np.argmax(ratio))
        nu = float(pr.fresnel_v(h[i], max(x[i], 1), max(D - x[i], 1), lam))
        loss = float(pr.knife_edge_loss(nu))
        self.h_above = float(h[i])
        xk = x / 1e3
        pp = self.plot("prof")
        pp.fill_between("fz", xk, los - r1, los + r1, color=NAVY, alpha=0.10)
        pp.fill_between("fz6", xk, los - 0.6 * r1, los + 0.6 * r1, color=NAVY, alpha=0.14)
        pp.line("los", xk, los, color=NAVY, width=1.6, style="--", name="line of sight")
        pp.line("r1u", xk, los + r1, color=NAVY, width=0.8, alpha=0.5, name="1st Fresnel zone")
        pp.line("r1d", xk, los - r1, color=NAVY, width=0.8, alpha=0.5)
        pp.line("ground", xk, obst, color=GREEN, width=1.8, fill=-5, fill_alpha=0.35,
                name=f"Earth bulge (k = {p.k:.2f}) + obstacle")
        pp.scatter("worst", [xk[i]], [obst[i]], color=RED if h[i] > -0.6 * r1[i] else GREEN, size=11,
                   symbol="d")
        top = max(p.hant + r1.max(), obst.max()) * 1.18 + 5
        pp.set_ylim(-5, top)
        pp.set_xlim(0, p.D)
        pj = self.plot("J")
        vv = np.linspace(-3, 5, 400)
        pj.line("ex", vv, -pr.knife_edge_loss(vv), color=NAVY, width=2.2, name="exact (Fresnel integrals)")
        pj.line("ap", vv, -pr.knife_edge_loss_p526(vv), color=ORANGE, width=1.5, style="--",
                name="ITU-R P.526 approximation")
        pj.hline("g6", -6, color=GRAY, style=":", label="grazing: −6 dB", label_pos=0.75)
        pj.vline("v0", 0, color=GRAY, style=":")
        pj.scatter("now", [np.clip(nu, -3, 5)], [-min(loss, 30)], color=RED, size=13, symbol="d")
        self.readout(clr=float(-h[i] / r1[i]), nu=nu, loss=loss,
                     r1=float(pr.fresnel_radius(D / 2, D / 2, f)))

    def story(self, p):
        clr = self.r.get("clr", 0)
        loss = self.r.get("loss", 0)
        s = (f"<p>Radio energy does not travel along a thin ray: it fills the <b>first Fresnel "
             f"zone</b>, a football-shaped region {v(2 * self.r.get('r1', 0), '.0f', 'm')} thick at "
             f"mid-path for {v(p.D, '.0f', 'km')} at {v(p.f, '.3g', 'GHz')}. The Earth's bulge "
             f"(drawn with k = {v(p.k, '.2f')}) and the obstacle push into it from below.</p>")
        if clr >= 0.6:
            s += (f"<p>The worst point clears by {v(clr, '.2f')} r₁: more than the planners' 60 % rule, "
                  f"so the path behaves like free space ({v(loss, '.1f', 'dB')}).</p>")
        elif clr > 0:
            s += (f"<p>The worst point clears the ray but only by {v(clr, '.2f')} r₁: part of the "
                  f"football is blocked and you pay {v(loss, '.1f', 'dB')}.</p>")
        else:
            s += (f"<p>{bad('Obstructed.')} The obstacle rises above the line of sight (ν = "
                  f"{v(self.r.get('nu', 0), '.2f')}): energy reaches the far end only by "
                  f"diffraction over the edge, at a cost of {v(loss, '.1f', 'dB')}.</p>")
        s += ("<p>Drop k toward 2/3 (a sub-refractive morning) and the bulge grows: microwave links "
              "are planned to survive both k = 4/3 and the worst-case k.</p>")
        return "<h3>Mind the football</h3>" + s + keybox(
            "Keep 60 % of r₁ = √(λ·d₁·d₂/(d₁+d₂)) clear. A knife edge exactly on the ray costs 6 dB, "
            "and ν ∝ √f makes the same hill far worse at higher frequency.")


# =============================================================================== 4. two-ray
class TwoRay(Experiment):
    title = "Two rays and the breakpoint"
    blurb = "A direct ray and a ground bounce: wild ripples close in, 40 dB per decade far out."
    book = "sec:ch11:tworay"
    controls = [
        LogSlider("f", "Frequency", 100, 6000, 900, unit="MHz"),
        Slider("ht", "Base-station height", 2, 100, 30, step=1, unit="m"),
        Slider("hr", "Mobile height", 0.5, 20, 1.5, step=0.1, unit="m"),
        Choice("pol", "Polarisation", ["Horizontal", "Vertical"]),
        Slider("eps", "Ground permittivity ε_r", 2, 80, 15, step=1,
               help="Dry ground about 4–15, wet ground 25–30, sea water about 80"),
    ]
    plots = [
        Plot("g", "Path gain of the two-ray model", x="distance (m)", y="path gain (dB)", logx=True,
             xlim=(1, 50000), ylim=(-170, -20), legend="bl"),
        Plot("gam", "Ground reflection coefficient |Γ|", x="distance (m)", y="|Γ|", logx=True,
             xlim=(1, 50000), ylim=(0, 1.05), legend="br"),
    ]
    layout = [["g"], ["gam"]]
    row_stretch = [3, 1]
    readouts = [
        Readout("lastmax", "Last maximum 4h_t h_r/λ", "km", ".2f"),
        Readout("cross", "Crossover 4πh_t h_r/λ", "km", ".2f"),
        Readout("slope", "Slope at 2–5× crossover", "dB/decade", ".1f"),
        Readout("vs_fs", "At 10 km, vs free space", "dB", "+.1f"),
    ]
    challenges = [
        Challenge("Push the crossover beyond 20 km.",
                  lambda s: s.r.cross > 20, hint="It is 4π·h_t·h_r/λ: height and frequency both help."),
        Challenge("Keep the phone at 1.5 m and put the last maximum beyond 1 km at 900 MHz.",
                  lambda s: abs(s.p.hr - 1.5) < 0.05 and abs(s.p.f - 900) < 10 and s.r.lastmax > 1.0),
        Challenge("Vertical polarisation: move the Brewster dip (where the ground reflection "
                  "vanishes) beyond 300 m.",
                  lambda s: s.p.pol == "Vertical" and s.exp.dbrew > 300,
                  hint="The dip sits at a fixed grazing angle, set by ε_r: taller antennas push it out."),
    ]

    def gamma(self, d, p):
        dref = np.sqrt(d ** 2 + (p.ht + p.hr) ** 2)
        sin_t = (p.ht + p.hr) / dref
        cos_t = d / dref
        z = np.sqrt(p.eps - cos_t ** 2)
        if p.pol == "Horizontal":
            return (sin_t - z) / (sin_t + z)
        return (p.eps * sin_t - z) / (p.eps * sin_t + z)

    def update(self, p):
        f = p.f * 1e6
        lam = pr.wavelength(f)
        d = np.logspace(0, np.log10(50000), 5000)
        g = pr.two_ray_gain_db(d, f, p.ht, p.hr, p.eps, "h" if p.pol == "Horizontal" else "v")
        pg = self.plot("g")
        pg.line("fs", d, -pr.fspl_db(d, f), color=GRAY, style="--", width=1.6, name="free space (1/d²)")
        pg.line("d4", d, 20 * np.log10(p.ht * p.hr / d ** 2), color=RED, style="--", width=1.6,
                name="h_t·h_r/d²  (1/d⁴)")
        pg.line("g", d, g, color=NAVY, width=1.3, name=f"two-ray, {p.pol.lower()} polarisation")
        lm = 4 * p.ht * p.hr / lam
        cr = np.pi * lm
        pg.vline("lm", lm, color=ORANGE, style=":", label="last maximum", label_pos=0.95)
        pg.vline("cr", cr, color=PURPLE, style=":", label="crossover", label_pos=0.86)
        G = np.abs(self.gamma(d, p))
        pgm = self.plot("gam")
        pgm.line("G", d, G, color=NAVY, width=2, name=f"{p.pol.lower()}")
        other = Params_other(p)
        pgm.line("Go", d, np.abs(self.gamma(d, other)), color=GRAY, width=1.2, style="--",
                 name=f"{other.pol.lower()}")
        Gv = np.abs(self.gamma(d, p if p.pol == "Vertical" else other))
        self.dbrew = float(d[np.argmin(Gv)])
        pgm.vline("brew", self.dbrew, color=GREEN, style=":", label="Brewster dip (vertical)",
                  label_pos=0.85)
        d1, d2 = 2 * cr, 5 * cr
        if d2 < 50000:
            g1, g2 = np.interp([np.log10(d1), np.log10(d2)], np.log10(d), g)
            slope = -(g2 - g1) / np.log10(d2 / d1)
        else:
            slope = float("nan")
        g10 = float(np.interp(4, np.log10(d), g))
        self.readout(lastmax=lm / 1e3, cross=cr / 1e3, slope=slope if np.isfinite(slope) else "—",
                     vs_fs=g10 + float(pr.fspl_db(1e4, f)))

    def story(self, p):
        cr = self.r.get("cross", 1)
        s = (f"<p>Over flat ground the receiver hears the direct ray plus a reflection that "
             f"travelled a little further. Close in, the path difference is many wavelengths and "
             f"the two beat: ripples and deep nulls. Beyond the <b>last maximum</b> "
             f"({v(self.r.get('lastmax', 0), '.2f', 'km')}) the difference shrinks below λ/2, the "
             f"reflection (Γ ≈ −1 at grazing) cancels the direct ray more and more, and past the "
             f"<b>crossover</b> ({v(cr, '.2f', 'km')}) the power falls as 1/d⁴: 40 dB per decade.</p>")
        s += ("<p>Out there the received power ≈ (h_t·h_r/d²)², independent of frequency: raising "
              "the mast is the cheapest way to buy range, which is why broadcast towers are tall.</p>")
        if p.pol == "Vertical":
            s += ("<p>Vertical polarisation has a <b>Brewster angle</b> where the ground reflection "
                  "nearly vanishes (bottom plot): around that distance the ripples die out.</p>")
        return "<h3>The ground is a mirror</h3>" + s + keybox(
            "Breakpoint ≈ 4h_t·h_r/λ: inside it, roughly free space with ripples; outside, 1/d⁴. "
            "Real cells sit in between, hence exponents of 3–4.")


def Params_other(p):
    q = st.Params(dict(p))
    q.pol = "Vertical" if p.pol == "Horizontal" else "Horizontal"
    return q


# =============================================================================== 5. empirical models
class PathLossModels(Experiment):
    title = "Hata, COST-231 and 3GPP 38.901"
    blurb = "Real cities are fitted, not solved: the models every planning tool is built on."
    book = "sec:ch11:models"
    controls = [
        LogSlider("f", "Frequency", 0.15, 40, 1.8, unit="GHz"),
        Slider("hbs", "Base-station height", 10, 200, 30, step=1, unit="m"),
        Slider("hm", "Mobile height", 1.0, 10, 1.5, step=0.1, unit="m"),
        Choice("scn", "38.901 scenario", ["UMa", "UMi", "InH"]),
        Choice("o2i", "Users indoors (38.901 O2I)", ["Outdoor", "Low-loss building",
                                                     "High-loss building"], style="menu"),
    ]
    plots = [
        Plot("pl", "Path loss vs distance", x="distance (m)", y="path loss (dB)", logx=True,
             xlim=(10, 20000), ylim=(40, 200), legend="br", legend_cols=2),
        Plot("o2i", "Outdoor-to-indoor penetration (TR 38.901)", x="frequency (GHz)",
             y="O2I loss (dB)", logx=True, xlim=(0.5, 100), ylim=(0, 60), legend="tl"),
    ]
    layout = [["pl", "o2i"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("hslope", "Hata slope", "dB/decade", ".1f"),
        Readout("pl1", "38.901 NLOS at 1 km", "dB", ".1f"),
        Readout("dbp", "38.901 LOS breakpoint", "m", ".0f"),
        Readout("o2i", "O2I loss", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Flatten the Hata slope below 31 dB per decade.",
                  lambda s: s.r.hslope < 31, hint="The slope is 44.9 − 6.55·log₁₀(h_b)."),
        Challenge("Push the 38.901 UMa line-of-sight breakpoint beyond 1 km.",
                  lambda s: s.p.scn == "UMa" and s.r.dbp > 1000,
                  hint="d_BP = 4(h_BS − 1)(h_UT − 1)·f/c: three knobs."),
        Challenge("Let a window eat the budget: high-loss building at a frequency where O2I exceeds 35 dB.",
                  lambda s: s.p.o2i == "High-loss building" and s.r.o2i > 35),
    ]

    def update(self, p):
        d = np.logspace(1, np.log10(20000), 400)
        f = p.f * 1e9
        pp = self.plot("pl")
        pp.line("fs", d, pr.fspl_db(d, f), color=GRAY, style="--", width=1.5, name="free space")
        dk = d[d >= 1000] / 1e3
        fm = p.f * 1e3
        if 150 <= fm <= 2000:
            sty = "-" if fm <= 1500 else ":"
            for env, col in (("urban", GREEN), ("suburban", TEAL), ("open", GOLD)):
                pp.line("h_" + env, d[d >= 1000], pr.hata(fm, p.hbs, p.hm, dk, env), color=col,
                        width=2.0, style=sty, name=f"Hata {env}")
            if fm >= 1500:
                pp.line("cost", d[d >= 1000], pr.cost231(fm, p.hbs, p.hm, dk, 3), color=PURPLE,
                        width=2.2, name="COST-231 (metropolitan)")
        else:
            pp.text("hv", 1200, 60, "Hata / COST-231: valid only 0.15–2 GHz", color=GREEN, size=8.5)
        dd = d if p.scn != "InH" else d[d <= 150]
        los, nlos = pr.pl_38901(p.scn, dd, p.f, p.hbs, p.hm)
        o2i = 0.0 if p.o2i == "Outdoor" else float(pr.o2i_loss_db(p.f, "low" if p.o2i.startswith("Low") else "high"))
        pp.line("los", dd, los, color=NAVY, width=2.2, name=f"38.901 {p.scn} LOS")
        pp.line("nlos", dd, nlos, color=RED, width=2.2, name=f"38.901 {p.scn} NLOS")
        if o2i > 0:
            pp.line("in", dd, nlos + o2i, color=ORANGE, width=2.2, style="--", name="NLOS + O2I (indoor)")
        dbp = 4 * (p.hbs - 1) * (p.hm - 1) * f / C0
        if p.scn != "InH":
            pp.vline("bp", dbp, color=NAVY, style=":", label="LOS breakpoint", label_pos=0.55)
        po = self.plot("o2i")
        fg = np.logspace(np.log10(0.5), 2, 200)
        po.line("lo", fg, pr.o2i_loss_db(fg, "low"), color=NAVY, width=2, name="low-loss (glass + concrete)")
        po.line("hi", fg, pr.o2i_loss_db(fg, "high"), color=RED, width=2, name="high-loss (IRR glass)")
        po.vline("f", p.f, color=GRAY, style=":")
        if o2i > 0:
            po.scatter("now", [p.f], [o2i], color=ORANGE, size=12, symbol="d")
        pl1 = float(pr.pl_38901(p.scn, 1000.0, p.f, p.hbs, p.hm)[1])
        self.readout(hslope=44.9 - 6.55 * np.log10(p.hbs), pl1=pl1, dbp=dbp, o2i=o2i)

    def story(self, p):
        s = (f"<p>Nobody solves Maxwell's equations for a city. Okumura drove a measurement van around "
             f"Tokyo in the 1960s; Hata fitted formulas to his curves, and COST-231 stretched them to "
             f"2 GHz (dotted Hata curves: used beyond their 1.5 GHz range). Their slope, "
             f"{v(self.r.get('hslope', 0), '.1f', 'dB')} per decade at a "
             f"{v(p.hbs, '.0f', 'm')} mast, is the city's path-loss exponent ×10.</p>"
             f"<p>3GPP TR 38.901 is the modern reference, from 0.5 to 100 GHz, with separate LOS and "
             f"NLOS laws and a LOS breakpoint (here {v(self.r.get('dbp', 0), '.0f', 'm')}).</p>")
        if p.o2i != "Outdoor":
            s += (f"<p>Indoors, the building adds {v(self.r.get('o2i', 0), '.1f', 'dB')} at "
                  f"{v(p.f, '.3g', 'GHz')}. Metallised (IRR) glass is nearly a Faraday cage at "
                  "millimetre waves: often the single biggest term in a 5G budget.</p>")
        return "<h3>Fitted, not solved</h3>" + s + keybox(
            "Use each model inside its validity range. Above 2 GHz, Hata is extrapolation; below "
            "0.5 GHz, 38.901 is.")


# =============================================================================== 6. atmosphere
class Atmosphere(Experiment):
    title = "Rain and oxygen"
    blurb = "Above 10 GHz the sky joins the budget: rain fades, oxygen and water-vapour lines."
    book = "sec:ch11:atmos"
    controls = [
        LogSlider("f", "Frequency", 1, 350, 12, unit="GHz"),
        Slider("R", "Rain rate", 0, 150, 50, step=1, unit="mm/h",
               help="5 drizzle, 25 heavy rain, 50 a downpour, 100+ a tropical storm"),
        LogSlider("L", "Path length in the rain", 0.1, 20, 3.0, unit="km"),
        Slider("rho", "Water-vapour density", 0, 20, 7.5, step=0.5, unit="g/m³"),
    ]
    plots = [
        Plot("spec", "Specific attenuation (sea level)", x="frequency (GHz)", y="attenuation (dB/km)",
             logx=True, logy=True, xlim=(1, 350), ylim=(1e-3, 100), legend="tl", legend_cols=2),
        BarPlot("bar", "Loss over your path", y="loss (dB)"),
    ]
    layout = [["spec", "bar"]]
    col_stretch = [3, 1]
    readouts = [
        Readout("gr", "Rain γ_R = kR^α", "dB/km", ".2f"),
        Readout("gg", "Clear air (gases)", "dB/km", ".3g"),
        Readout("tot", "Total over the path", "dB", ".1f"),
        Readout("ka", "k, α (P.838-3)", "", None),
    ]
    challenges = [
        Challenge("Reproduce the book's storm: 20 GHz and 50 mm/h give about 5.7 dB/km.",
                  lambda s: abs(s.p.f - 20) < 0.3 and s.p.R == 50 and 5.4 <= s.r.gr <= 6.0),
        Challenge("Find the oxygen wall: more than 10 dB/km of clear air.",
                  lambda s: s.r.gg > 10, hint="Look near 60 GHz, the band WiGig chose on purpose."),
        Challenge("Keep 3 km of a 50 mm/h downpour under 6 dB at a frequency above 8 GHz.",
                  lambda s: s.p.f > 8 and s.p.R >= 50 and s.p.L >= 2.95 and s.r.tot < 6),
    ]

    def update(self, p):
        f = np.logspace(0, np.log10(350), 700)
        go, gw = pr.gas_atten(f, p.rho)
        ps = self.plot("spec")
        ps.line("o", f, go, color=NAVY, width=1.6, name="oxygen")
        ps.line("w", f, gw, color=TEAL, width=1.6, name=f"water vapour ({p.rho:g} g/m³)")
        ps.line("g", f, go + gw, color=GRAY, width=2.2, style="--", name="all gases")
        if p.R > 0:
            fr = f[f <= 100]
            ps.line("r", fr, pr.rain_specific_atten(fr, p.R), color=RED, width=2.4,
                    name=f"rain {p.R:g} mm/h")
        ps.vline("f", p.f, color=ORANGE, style=":", label=f"{p.f:.3g} GHz", label_pos=0.06)
        go1, gw1 = pr.gas_atten(p.f, p.rho)
        gg = float(go1[0] + gw1[0])
        gr = float(pr.rain_specific_atten(min(p.f, 100), p.R)) if p.R > 0 else 0.0
        k, a = pr.rain_k_alpha(min(p.f, 100))
        tot = (gr + gg) * p.L
        pb = self.plot("bar")
        vals = [gg * p.L, gr * p.L, tot]
        pb.bars("b", [0, 1, 2], vals, width=0.6, colors=[TEAL, RED, NAVY])
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.1f} dB", anchor=(0.5, 1), size=9, bold=True)
        pb.set_xticks([(0, "gases"), (1, "rain"), (2, "total")])
        pb.set_xlim(-0.6, 2.6)
        pb.set_ylim(0, max(5, 1.2 * tot))
        self.readout(gr=gr, gg=gg, tot=tot, ka=f"{float(k):.3g}, {float(a):.3f}")

    def story(self, p):
        s = (f"<p>Raindrops are a few millimetres across: at {v(p.f, '.3g', 'GHz')} (λ = "
             f"{v(30 / p.f, '.2g', 'cm')}) they absorb and scatter strongly once λ approaches their "
             f"size. ITU-R P.838 fits this as γ_R = k·R^α: {v(self.r.get('gr', 0), '.2f', 'dB/km')} "
             f"in {v(p.R, '.0f', 'mm/h')} of rain. Over {v(p.L, '.3g', 'km')} the path loses "
             f"{v(self.r.get('tot', 0), '.1f', 'dB')} in total.</p>"
             "<p>Oxygen molecules resonate near 60 GHz (about 15 dB/km at sea level) and 118 GHz; water "
             "vapour at 22 and 183 GHz. These lines make some bands useless for long links, and "
             "useful for exactly that reason: a 60 GHz Wi-Fi link does not reach the neighbours.</p>")
        return "<h3>Weather in the budget</h3>" + s + keybox(
            "Below about 10 GHz the sky is transparent. Above it, satellite and backhaul links carry "
            "a rain margin sized from ITU-R P.618 statistics for an availability target.")


# =============================================================================== 7. coverage statistics
class Coverage(Experiment):
    title = "Edge coverage vs area coverage"
    blurb = "Shadowing makes coverage a probability. The cell edge is the worst place to stand."
    book = "sec:ch11:shadowing"
    controls = [
        Slider("sigma", "Shadowing σ", 2, 12, 8, step=0.5, unit="dB"),
        Slider("n", "Path-loss exponent n", 2, 5, 3.5, step=0.05),
        Slider("M", "Fade margin at the edge", 0, 20, 4, step=0.1, unit="dB"),
        Slider("target", "Target area coverage", 50, 99, 95, step=1, unit="%"),
        Button("again", "New users"),
    ]
    plots = [
        Plot("cell", "Users in one cell (green = covered)", x="x / R", y="y / R",
             xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect=True, legend=None),
        Plot("rad", "Probability of coverage vs distance", x="distance from the site / R",
             y="P(covered) (%)", xlim=(0, 1), ylim=(40, 101), legend=None),
        Plot("cov", "Coverage vs margin", x="fade margin at the edge (dB)", y="coverage (%)",
             xlim=(0, 20), ylim=(45, 101), legend="br"),
    ]
    layout = [["cell", "cov"], ["rad", "cov"]]
    row_stretch = [3, 2]
    col_stretch = [2, 3]
    readouts = [
        Readout("edge", "Edge coverage", "%", ".1f"),
        Readout("area", "Area coverage (Jakes–Reudink)", "%", ".1f", good=lambda x: x >= 95),
        Readout("mc", "Area coverage (simulated users)", "%", ".1f"),
        Readout("need", "Margin for the target", "dB", ".2f",
                help="Edge margin that gives the target area coverage (Jakes–Reudink)"),
    ]
    challenges = [
        Challenge("The book's numbers: σ = 8 dB, n = 3.5, reach 95 % area coverage with no more "
                  "than 8.8 dB of margin.",
                  lambda s: s.p.sigma == 8 and abs(s.p.n - 3.5) < 0.01 and s.r.area >= 95
                  and s.p.M <= 8.8),
        Challenge("Only 75 % at the edge (±1 %) but at least 90 % of the area covered.",
                  lambda s: abs(s.r.edge - 75) <= 1 and s.r.area >= 90),
        Challenge("With σ = 8 dB and a 95 % target, find a path-loss exponent that needs less than "
                  "8 dB of edge margin.",
                  lambda s: s.p.sigma == 8 and s.p.target == 95 and s.r.need < 8,
                  hint="A steep exponent means users inside the cell gain signal fast."),
    ]
    N = 2500

    def setup(self):
        self.on_again(None)

    def on_again(self, p):
        r = np.sqrt(self.rng.random(self.N))
        a = 2 * np.pi * self.rng.random(self.N)
        self.ux, self.uy, self.ur = r * np.cos(a), r * np.sin(a), r
        self.z = self.rng.standard_normal(self.N)

    def update(self, p):
        excess = p.M - 10 * p.n * np.log10(np.maximum(self.ur, 1e-6)) + p.sigma * self.z
        ok = excess >= 0
        pc = self.plot("cell")
        t = np.linspace(0, 2 * np.pi, 300)
        pc.line("edge", np.cos(t), np.sin(t), color=GRAY, width=1.4)
        pc.scatter("ok", self.ux[ok], self.uy[ok], color=GREEN, size=4, alpha=0.6)
        pc.scatter("bad", self.ux[~ok], self.uy[~ok], color=RED, size=5, alpha=0.9)
        pc.scatter("site", [0], [0], color=NAVY, size=14, symbol="t")
        rr = np.linspace(0.02, 1, 200)
        prad = self.plot("rad")
        prad.line("p", rr, 100 * norm.cdf((p.M - 10 * p.n * np.log10(rr)) / p.sigma), color=NAVY,
                  width=2.2, fill=40, fill_alpha=0.12)
        M = np.linspace(0, 20, 201)
        pe = norm.cdf(M / p.sigma)
        area = pr.area_coverage(pe, p.n, p.sigma)
        pv = self.plot("cov")
        pv.line("e", M, 100 * pe, color=NAVY, width=2, name="at the edge")
        pv.line("a", M, 100 * area, color=GREEN, width=2.4, name="over the area")
        pv.hline("95", p.target, color=GRAY, style=":", label=f"target {p.target:.0f} %", label_pos=0.02)
        e_now = float(norm.cdf(p.M / p.sigma))
        a_now = float(pr.area_coverage(e_now, p.n, p.sigma))
        pv.scatter("en", [p.M], [100 * e_now], color=NAVY, size=11, symbol="d")
        pv.scatter("an", [p.M], [100 * a_now], color=GREEN, size=11, symbol="d")
        pv.scatter("mc", [p.M], [100 * np.mean(ok)], color=RED, size=9, outline=RED,
                   name="simulated users")
        try:
            need = pr.edge_margin_for_area(p.target / 100, p.n, p.sigma)[0]
        except ValueError:
            need = float("nan")
        if np.isfinite(need):
            pv.vline("need", need, color=GREEN, style="--", label=f"{need:.1f} dB", label_pos=0.1)
        self.readout(edge=100 * e_now, area=100 * a_now, mc=100 * np.mean(ok),
                     need=need if np.isfinite(need) else "—")

    def story(self, p):
        e, a = self.r.get("edge", 0), self.r.get("area", 0)
        s = (f"<p>Every dot is a user with its own shadowing draw (σ = {v(p.sigma, '.1f', 'dB')}). "
             f"At the edge, the median signal sits exactly {v(p.M, '.1f', 'dB')} above the threshold, "
             f"so a user there is covered with probability Φ(M/σ) = {v(e, '.1f', '%')}.</p>"
             f"<p>Users inside the cell are closer, and with an exponent of {v(p.n, '.2f')} their "
             f"median rises quickly: averaged over the disc, {v(a, '.1f', '%')} of the area is covered "
             f"(Jakes–Reudink formula; the red marker is the simulated users). The red dots cluster at "
             f"the rim.</p>")
        return "<h3>The edge is the worst place to stand</h3>" + s + keybox(
            "Operators promise area coverage (95 %) but must design the edge for less (about 86 % "
            "with σ = 8 dB, n = 3.5): an 8.7 dB margin, as in the book's LTE budget.")


# =============================================================================== 8. coverage map
@lru_cache(maxsize=8)
def _fields(seed, dcorr, n, dx, nsite):
    rng = np.random.default_rng(seed)
    return np.stack([fdg.shadowing_field(n, n, dx, dcorr, 1.0, rng) for _ in range(nsite)])


class CoverageMap(Experiment):
    title = "A coverage map"
    blurb = "A few sites, real shadowing, and the question every operator asks: where are the holes?"
    book = "sec:ch11:shadowing"
    controls = [
        IntSlider("sites", "Number of sites", 1, 7, 3),
        Slider("isd", "Site spacing", 0.5, 5.0, 2.0, step=0.1, unit="km"),
        Slider("eirp", "EIRP per site", 30, 75, 60, step=1, unit="dBm"),
        Slider("f", "Frequency", 0.7, 3.8, 1.8, step=0.05, unit="GHz"),
        Heading("Environment"),
        Slider("sigma", "Shadowing σ", 0, 12, 8, step=0.5, unit="dB"),
        LogSlider("dcorr", "Shadowing correlation distance", 30, 1000, 200, unit="m"),
        Choice("metric", "Quality measure", ["SNR (noise only)", "SINR (one shared channel)"],
               style="menu"),
        Slider("thr", "Service threshold", -10, 20, 0, step=0.5, unit="dB"),
        Button("again", "New shadowing"),
    ]
    plots = [
        ImagePlot("map", "Best-server quality across 8 km × 8 km (dB)", x="x (km)", y="y (km)",
                  xlim=(-4, 4), ylim=(-4, 4), aspect=True),
        Plot("cdf", "Share of the area at or below a quality", x="quality (dB)", y="area (%)",
             xlim=(-15, 45), ylim=(0, 100), legend=None),
    ]
    layout = [["map", "cdf"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("cov", "Area above threshold", "%", ".1f", good=lambda x: x >= 95),
        Readout("med", "Median quality", "dB", ".1f"),
        Readout("p5", "Worst 5 % (cell edge)", "dB", ".1f"),
        Readout("dens", "Sites per km²", "", ".3f"),
    ]
    challenges = [
        Challenge("Cover 95 % of the map at 0 dB SNR with σ = 8 dB, using at most 3 sites.",
                  lambda s: s.p.metric.startswith("SNR") and s.p.thr >= 0 and s.p.sigma >= 8
                  and s.p.sites <= 3 and s.r.cov >= 95),
        Challenge("Interference changes the game: with all 7 sites sharing a channel, raise the "
                  "EIRP and watch the edge SINR barely move. Then find a layout that lifts the "
                  "worst 5 % above −2.5 dB.",
                  lambda s: s.p.metric.startswith("SINR") and s.p.sites == 7 and s.r.p5 > -2.5,
                  hint="Once interference dominates, power cancels out; geometry (spacing) and "
                       "shadowing decide."),
        Challenge("Spread 7 sites 4 km apart and still cover 90 % at 5 dB SNR (σ ≥ 6 dB) with no "
                  "more than 58 dBm EIRP.",
                  lambda s: s.p.metric.startswith("SNR") and s.p.sites == 7 and s.p.isd >= 4
                  and s.p.thr >= 5 and s.p.sigma >= 6 and s.p.eirp <= 58 and s.r.cov >= 90),
    ]
    n, span = 160, 8000.0
    hbs, bw_hz, nf = 25.0, 10e6, 7.0

    def setup(self):
        self.seed = 22

    def on_again(self, p):
        self.seed += 1

    def site_xy(self, p):
        pts = [(0.0, 0.0)] + [(p.isd * np.cos(a), p.isd * np.sin(a))
                              for a in np.deg2rad(np.arange(30, 390, 60))]
        return np.array(pts[:p.sites]) * 1e3

    def update(self, p):
        dx = self.span / self.n
        xs = (np.arange(self.n) + 0.5) * dx - self.span / 2
        X, Y = np.meshgrid(xs, xs)
        sites = self.site_xy(p)
        F = _fields(self.seed, round(p.dcorr, 1), self.n, dx, 7)[:p.sites]
        rx = []
        for (sx, sy), fld in zip(sites, F):
            d = np.maximum(np.hypot(X - sx, Y - sy), 35.0)
            pl = pr.pl_38901("UMa", d, p.f, self.hbs)[1]
            rx.append(p.eirp - pl + p.sigma * fld)
        rx = np.array(rx)
        noise = float(pr.thermal_noise_dbm(self.bw_hz, self.nf))
        best = rx.max(axis=0)
        if p.metric.startswith("SNR"):
            q = best - noise
        else:
            lin = 10 ** (rx / 10)
            tot = lin.sum(axis=0)
            q = best - 10 * np.log10(tot - 10 ** (best / 10) + 10 ** (noise / 10))
        pm = self.plot("map")
        half = self.span / 2e3
        pm.image("img", np.clip(q, -10, 40), x=(-half, half), y=(-half, half), cmap="heat",
                 levels=(-10, 40), colorbar=True, cbar_label="dB")
        holes = (q < p.thr).astype(float)
        pm.image("holes", holes, x=(-half, half), y=(-half, half),
                 cmap=((0, "#C0392B00"), (1, "#C0392BA0")), levels=(0, 1))
        pm.item("holes").setZValue(-15)
        pm.scatter("sites", sites[:, 0] / 1e3, sites[:, 1] / 1e3, color=GREEN, size=14, symbol="t")
        qs = np.sort(q.ravel())
        pc = self.plot("cdf")
        pc.line("c", qs, 100 * np.arange(1, len(qs) + 1) / len(qs), color=NAVY, width=2.2)
        pc.vline("thr", p.thr, color=RED, style="--", label=f"threshold {p.thr:g} dB", label_pos=0.9)
        cov = 100 * np.mean(q >= p.thr)
        pc.hline("cov", 100 - cov, color=GRAY, style=":", label=f"{100 - cov:.1f} % in holes",
                 label_pos=0.6)
        self.readout(cov=cov, med=float(np.median(q)), p5=float(np.percentile(q, 5)),
                     dens=p.sites / (self.span / 1e3) ** 2)

    def story(self, p):
        cov = self.r.get("cov", 0)
        s = (f"<p>{v(p.sites, 'd')} site(s), {v(p.isd, '.1f', 'km')} apart, each radiating "
             f"{v(p.eirp, '.0f', 'dBm')} EIRP at {v(p.f, '.2f', 'GHz')} (TR 38.901 UMa NLOS). Shadowing "
             f"is a correlated random map (σ = {v(p.sigma, '.1f', 'dB')}, correlation distance "
             f"{v(p.dcorr, '.0f', 'm')}), drawn independently for each site: every point is served "
             f"by whichever site is strongest there. Red patches are the holes below "
             f"{v(p.thr, '.1f', 'dB')}: {v(100 - cov, '.1f', '%')} of the map.</p>")
        if p.metric.startswith("SINR"):
            s += ("<p>With every site on the same channel, the others are not silent: they are "
                  "<b>interference</b>. Raising all powers raises signal and interference together, "
                  "so the SINR at a cell edge is set by geometry, not by power. This is why cellular "
                  "systems are interference-limited (Chapter 20).</p>")
        else:
            s += ("<p>Against noise alone, more power or more sites always helps. Count the cost: "
                  f"{v(self.r.get('dens', 0), '.3f')} sites per km².</p>")
        return "<h3>Where are the holes?</h3>" + s + keybox(
            "Planning tools do exactly this on real terrain and building maps, then add sites where "
            "the holes are: coverage is bought one site at a time.")


# =============================================================================== the lab
LAB = st.Lab(22, "Propagation and Link Budgets", chapter=11, chapter_title="The Wireless Channel",
             experiments=[Friis, LinkBudget, KnifeEdge, TwoRay, PathLossModels, Atmosphere,
                          Coverage, CoverageMap])

if __name__ == "__main__":
    st.run(LAB)
