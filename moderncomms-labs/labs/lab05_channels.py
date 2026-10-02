"""Lab 05 · The Wireless Channel: Path Loss, Multipath, Fading and Doppler   (Chapter 11)

Run it:      python labs/lab05_channels.py
Self-test:   python labs/lab05_channels.py --selftest

The channel is the adversary every other chapter fights. It attenuates (path loss), it
shadows (buildings, hills, bodies), it echoes (multipath and delay spread) and it moves
(Doppler). Eight experiments take those effects apart one scale at a time: a measurement
drive, the standing-wave speckle inside a room, a live fading envelope at highway speed,
the Rayleigh/Rice/Nakagami family, echoes as notches in frequency, the 3GPP channel models
as a moving time-frequency picture, what fading does to the error rate (and how diversity
buys it back), and finally how a channel sounder measures all of it.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import j0
from scipy.stats import norm

import commlib as cl
from commlib import fading as fdg
from commlib import propagation as pr
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, BERPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

C0 = 299_792_458.0


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def rician_or_nakagami(kind, K, m, n, rng):
    """n unit-power complex fading gains: Rayleigh, Rician(K) or Nakagami-m (random phase)."""
    if kind == "Nakagami-m":
        amp = np.sqrt(rng.gamma(m, 1 / m, n))
        return amp * np.exp(2j * np.pi * rng.random(n))
    Kk = K if kind == "Rician" else 0.0
    sc = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return np.sqrt(Kk / (Kk + 1)) + np.sqrt(1 / (Kk + 1)) * sc


# =============================================================================== 1. path loss
class PathLossShadowing(Experiment):
    title = "Path loss and shadowing"
    blurb = "Drive away from a mast: the median falls on a straight line, the readings scatter."
    book = "sec:ch11:models"
    controls = [
        Heading("Environment"),
        Choice("model", "Median path loss", ["Free space", "Log-distance", "Two-ray ground"],
               "Log-distance", style="menu"),
        Slider("n", "Path-loss exponent n", 1.6, 5.0, 3.5, step=0.05,
               help="2 in free space, 2.7–3.5 in towns, 4 and more through walls and clutter",
               enabled_if=lambda p: p.model == "Log-distance"),
        Slider("sigma", "Shadowing σ", 0.0, 12.0, 6.0, step=0.5, unit="dB",
               help="Standard deviation of the log-normal shadowing around the median"),
        LogSlider("f", "Carrier frequency", 0.4, 6.0, 1.8, unit="GHz"),
        Heading("Link"),
        Slider("eirp", "Transmit EIRP", 20, 70, 50, step=1, unit="dBm"),
        Slider("sens", "Receiver sensitivity", -130, -80, -105, step=1, unit="dBm"),
        Slider("margin", "Planned fade margin", 0, 25, 0, step=0.5, unit="dB",
               help="Plan the cell edge where the median is this far above the sensitivity"),
        Button("drive", "New measurement drive"),
    ]
    plots = [
        Plot("pl", "Received power along a drive (each dot = one measurement)",
             x="distance from the mast (m)", y="received power (dBm)", logx=True,
             xlim=(10, 10000), ylim=(-160, 5), legend="tr"),
        Plot("hist", "Shadowing: readings minus the median", x="deviation (dB)",
             y="probability density", xlim=(-36, 36), ylim=(0, 0.2), legend="tr"),
        Plot("edge", "Coverage at the planned edge", x="fade margin (dB)",
             y="edge coverage (%)", xlim=(0, 25), ylim=(40, 102), legend=None),
    ]
    layout = [["pl", "pl"], ["hist", "edge"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("slope", "Fitted slope", "dB/decade", ".1f",
                help="Least-squares fit of the dots against log10(distance)"),
        Readout("sig", "Measured σ", "dB", ".1f"),
        Readout("radius", "Planned cell radius", "km", ".2f"),
        Readout("cov", "Edge coverage", "%", ".0f", good=lambda x: x >= 90,
                help="Probability that a user at the planned edge is above the sensitivity: Φ(margin/σ)"),
    ]
    challenges = [
        Challenge("Plan the cell edge for at least 90 % coverage with σ of 8 dB or more.",
                  lambda s: s.p.sigma >= 8 and s.r.cov >= 90,
                  hint="Coverage at the edge is Φ(M/σ): you need a margin of about 1.28σ."),
        Challenge("Show that free space loses 20 dB per decade: fitted slope within 1 dB of 20 "
                  "(you may need to calm the shadowing).",
                  lambda s: s.p.model == "Free space" and abs(s.r.slope - 20) < 1),
        Challenge("Two-ray ground (30 m mast, 1.5 m phone): put the breakpoint at 1 km ± 5 %.",
                  lambda s: s.p.model == "Two-ray ground" and abs(s.exp.dbp - 1000) <= 50,
                  hint="Breakpoint d = 4·h_t·h_r/λ: solve for λ, then f = c/λ."),
        Challenge("With σ = 12 dB, get at least 97 % of the drive inside the planned cell above "
                  "the sensitivity.",
                  lambda s: s.p.sigma >= 12 and s.exp.inside >= 97),
    ]
    ht, hr = 30.0, 1.5

    def setup(self):
        self.on_drive(None)

    def on_drive(self, p):
        self.d = np.sort(10 ** self.rng.uniform(1, 4, 700))
        self.z = self.rng.standard_normal(700)

    def median_pl(self, d, p):
        f = p.f * 1e9
        if p.model == "Free space":
            return pr.fspl_db(d, f)
        if p.model == "Log-distance":
            return cl.log_distance_pl_db(d, f, n=p.n)
        return -pr.two_ray_gain_db(d, f, self.ht, self.hr)

    def update(self, p):
        dd = np.logspace(1, 4, 600)
        med = p.eirp - self.median_pl(dd, p)
        meas = p.eirp - self.median_pl(self.d, p) + p.sigma * self.z
        thr = p.sens + p.margin
        ok = np.flatnonzero(med >= thr)
        R = dd[ok[-1]] if len(ok) else 10.0
        pl = self.plot("pl")
        pl.fill_between("band", dd, med - p.sigma, med + p.sigma, color=NAVY, alpha=0.10)
        good_ = meas >= p.sens
        pl.scatter("ok", self.d[good_], meas[good_], color=NAVY, size=4, alpha=0.55,
                   name="reading above sensitivity")
        pl.scatter("bad", self.d[~good_], meas[~good_], color=RED, size=5, alpha=0.8,
                   name="reading below sensitivity")
        pl.line("med", dd, med, color=NAVY, width=2.4, name="median (model)")
        pl.hline("sens", p.sens, color=RED, style="--", label=f"sensitivity {p.sens:.0f} dBm",
                 label_pos=0.02)
        pl.vline("R", R, color=GREEN, style="--", label=f"planned edge {R / 1e3:.2f} km",
                 label_pos=0.08)
        dbp = self.dbp = 4 * self.ht * self.hr / (C0 / (p.f * 1e9))
        if p.model == "Two-ray ground":
            pl.vline("bp", dbp, color=ORANGE, style=":", label="breakpoint", label_pos=0.95)
        # fit
        A = np.vstack([np.log10(self.d), np.ones_like(self.d)]).T
        coef, *_ = np.linalg.lstsq(A, meas, rcond=None)
        slope = -coef[0]
        resid = meas - (A @ coef)
        dev = meas - (p.eirp - self.median_pl(self.d, p))
        ph = self.plot("hist")
        if p.sigma > 0.2:
            h, e = np.histogram(dev, bins=48, range=(-36, 36), density=True)
            ph.line("h", e, h, step=True, color=NAVY, width=1.4, fill=0, fill_alpha=0.25,
                    name="drive")
            x = np.linspace(-36, 36, 400)
            ph.line("g", x, norm.pdf(x, 0, p.sigma), color=RED, width=2, name=f"Gaussian, σ = {p.sigma:g} dB")
            ph.set_ylim(0, max(0.12, 1.25 * norm.pdf(0, 0, p.sigma)) if p.sigma >= 2 else 0.25)
        else:
            ph.text("none", -30, 0.15, "no shadowing: every reading\nsits on the median", color=GRAY)
            ph.set_ylim(0, 0.2)
        cov = 100 * (norm.cdf(p.margin / p.sigma) if p.sigma > 0 else 1.0)
        pe = self.plot("edge")
        M = np.linspace(0, 25, 300)
        for i, s_ in enumerate((4, 8, 12)):
            pe.line(f"c{s_}", M, 100 * norm.cdf(M / s_), color=GRAY, width=1.0, alpha=0.6)
            pe.text(f"t{s_}", 1.2 * s_, 100 * norm.cdf(1.2), f"σ = {s_} dB", color=GRAY,
                    size=8, anchor=(0, 1))
        if p.sigma > 0:
            pe.line("cur", M, 100 * norm.cdf(M / p.sigma), color=NAVY, width=2.2)
        pe.scatter("now", [p.margin], [cov], color=RED, size=12, symbol="d")
        pe.hline("90", 90, color=GREEN, style=":", width=1)
        inside = self.d <= R
        self.inside = 100 * np.mean(meas[inside] >= p.sens) if inside.any() else 0.0
        self.readout(slope=slope, sig=float(np.std(dev)), radius=R / 1e3, cov=cov)

    def story(self, p):
        sl = self.r.get("slope", 0)
        s = (f"<p>Every dot is one reading of a phone driving away from a mast radiating "
             f"{v(p.eirp, '.0f', 'dBm')}. The thick line is the <b>median path loss</b>; the "
             f"scatter around it is <b>shadowing</b>, the slow ±{v(p.sigma, '.0f', 'dB')} wander "
             f"caused by buildings, trees and hills between you and the mast. A straight-line fit "
             f"gives {v(sl, '.1f', 'dB')} per decade of distance.</p>")
        if p.model == "Free space":
            s += "<p>Free space spreads the power over a sphere: 1/d², exactly 20 dB per decade.</p>"
        elif p.model == "Log-distance":
            s += (f"<p>The log-distance model just replaces the 2 of free space by n = {v(p.n)}: "
                  f"{v(10 * p.n, '.0f', 'dB')} per decade. Ground reflections, clutter and walls "
                  "all steepen the slope.</p>")
        else:
            s += ("<p>Over flat ground the direct and the ground-reflected ray beat against each "
                  "other near the mast, then cancel more and more beyond the breakpoint (orange), "
                  "where the loss steepens to 40 dB per decade.</p>")
        s += (f"<p>Design for the median and half the users at the edge fail. A margin of "
              f"{v(p.margin, '.1f', 'dB')} buys {v(self.r.get('cov', 0), '.0f', '%')} edge "
              f"coverage, at the price of a smaller cell ({v(self.r.get('radius', 0), '.2f', 'km')}). "
              f"Lab 22 turns this into a full coverage plan.</p>")
        return "<h3>Three scales, first two</h3>" + s + keybox(
            "Path loss is the trend, shadowing the slow scatter around it. Fast fading, the third "
            "scale, lives inside every one of these dots: see experiments 2–4.")


# =============================================================================== 2. room
class RoomSpeckle(Experiment):
    title = "Standing waves in a room"
    blurb = "Walls reflect, rays interfere: a speckle of hot spots and dead spots half a wavelength apart."
    book = "sec:ch11:multipath"
    controls = [
        LogSlider("f", "Frequency", 0.3, 1.8, 0.9, unit="GHz"),
        Slider("gamma", "Wall reflection coefficient |Γ|", 0.0, 0.95, 0.6, step=0.01,
               help="0 = perfectly absorbing walls (an anechoic chamber), near 1 = metal walls"),
        IntSlider("order", "Reflections per ray (max)", 0, 4, 2,
                  help="Image method: how many wall bounces each ray may take"),
        Heading("Positions"),
        Slider("tx", "Transmitter x", 0.3, 5.7, 1.2, step=0.05, unit="m"),
        Slider("ty", "Transmitter y", 0.3, 3.7, 1.0, step=0.05, unit="m"),
        Slider("wy", "Walk along y =", 0.2, 3.8, 2.6, step=0.05, unit="m"),
    ]
    plots = [
        ImagePlot("room", "Received power in a 6 m × 4 m room (dB re local mean)",
                  x="x (m)", y="y (m)", xlim=(0, 6), ylim=(0, 4), aspect=True),
        Plot("walk", "Power along the walk (white line)", x="position x (m)",
             y="power re local mean (dB)", xlim=(0, 6), ylim=(-35, 12), legend=None),
        Plot("hist", "How the room's power is distributed", x="power re local mean (dB)",
             y="probability density", xlim=(-30, 10), ylim=(0, 0.16), legend="tl"),
    ]
    layout = [["room", "room"], ["walk", "hist"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("lam", "Wavelength", "cm", ".1f"),
        Readout("spacing", "Hot-spot spacing on the walk", "cm", ".1f"),
        Readout("deep", "Room > 10 dB below its mean", "%", ".1f",
                help="Rayleigh fading predicts 9.5 %"),
        Readout("spread", "Spread (1 % to 99 %)", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Make the room nearly Rayleigh: at least 6.5 % of it more than 10 dB down.",
                  lambda s: s.r.deep >= 6.5,
                  hint="Many strong reflections of similar strength: shiny walls and more bounces."),
        Challenge("Keep reflections on (≥ 2 bounces) but limit 10 dB fades to under 1 % of the room.",
                  lambda s: s.p.order >= 2 and s.r.deep < 1,
                  hint="Absorbing walls: the direct ray dominates every echo."),
        Challenge("Tune the frequency so the hot spots along the walk are about 12 cm apart (11–13 cm).",
                  lambda s: 11 <= s.r.spacing <= 13 and s.r.deep >= 3,
                  hint="The speckle grain scales with the wavelength: shorter λ, finer pattern."),
    ]
    Lx, Ly, nx, ny = 6.0, 4.0, 300, 200

    @staticmethod
    def images(tx, ty, Lx, Ly, order):
        """Image sources of a rectangular room: (x, y, number of reflections)."""
        out = []
        for n in range(-3, 4):
            for sx, base in ((+1, 0), (-1, 1)):
                rx = abs(2 * n) if sx > 0 else abs(2 * n - 1)
                X = 2 * n * Lx + sx * tx
                for m in range(-3, 4):
                    for sy in (+1, -1):
                        ry = abs(2 * m) if sy > 0 else abs(2 * m - 1)
                        if rx + ry <= order:
                            out.append((X, 2 * m * Ly + sy * ty, rx + ry))
        return out

    def field(self, p, x, y):
        k = 2 * np.pi * p.f * 1e9 / C0
        E = np.zeros(np.broadcast(x, y).shape, complex)
        for X, Y, nref in self.images(p.tx, p.ty, self.Lx, self.Ly, p.order):
            r = np.sqrt((x - X) ** 2 + (y - Y) ** 2) + 1e-3
            E += (-p.gamma) ** nref * np.exp(-1j * k * r) / r
        return E

    def update(self, p):
        dx = self.Lx / self.nx
        xs = (np.arange(self.nx) + 0.5) * dx
        ys = (np.arange(self.ny) + 0.5) * dx
        X, Y = np.meshgrid(xs, ys)
        P = np.abs(self.field(p, X, Y)) ** 2
        lam = C0 / (p.f * 1e9)
        win = int(max(5, round(1.5 * lam / dx))) | 1
        Pm = uniform_filter(P, size=win, mode="nearest")
        Pdb = db10(P / Pm)
        Pdb[np.hypot(X - p.tx, Y - p.ty) < 0.3] = 0       # near field of the antenna: not shown
        pr_ = self.plot("room")
        pr_.image("img", np.clip(Pdb, -30, 10), x=(0, self.Lx), y=(0, self.Ly), cmap="heat",
                  levels=(-30, 10), colorbar=True, cbar_label="dB")
        pr_.line("walk", [0, self.Lx], [p.wy, p.wy], color="#FFFFFF", width=2.6, style="--")
        pr_.scatter("tx", [p.tx], [p.ty], color=GREEN, size=14, symbol="star", outline=None)
        # walk
        xw = np.linspace(0, self.Lx, 3000)
        Pw = np.abs(self.field(p, xw, np.full_like(xw, p.wy))) ** 2
        kw = max(5, int(round(1.5 * lam / (xw[1] - xw[0])))) | 1
        Pwm = uniform_filter(Pw, size=kw, mode="nearest")
        w_db = db10(Pw / Pwm)
        pw = self.plot("walk")
        pw.hband("deep", -35, -10, color=RED, alpha=0.08)
        pw.line("w", xw, np.clip(w_db, -35, 12), color=NAVY, width=1.4)
        pw.hline("m10", -10, color=RED, style=":", width=1, label="−10 dB", label_pos=0.01)
        # deep fades along the walk: local minima below -10 dB
        mins = np.flatnonzero((w_db[1:-1] < w_db[:-2]) & (w_db[1:-1] <= w_db[2:]) & (w_db[1:-1] < -10)) + 1
        peaks = np.flatnonzero((w_db[1:-1] > w_db[:-2]) & (w_db[1:-1] >= w_db[2:]) & (w_db[1:-1] > -3)) + 1
        spacing = 100 * np.mean(np.diff(xw[peaks])) if len(peaks) > 1 else float("nan")
        if len(mins):
            pw.scatter("mins", xw[mins], np.maximum(w_db[mins], -34), color=RED, size=6)
        # histogram vs Rayleigh
        inside = np.hypot(X - p.tx, Y - p.ty) > 0.3
        vals = Pdb[inside]
        h, e = np.histogram(vals, bins=60, range=(-30, 10), density=True)
        ph = self.plot("hist")
        ph.line("h", e, h, step=True, color=NAVY, width=1.4, fill=0, fill_alpha=0.25,
                name="this room")
        yy = np.linspace(-30, 10, 400)
        xl = 10 ** (yy / 10)
        ph.line("ray", yy, np.log(10) / 10 * xl * np.exp(-xl), color=RED, width=2,
                name="Rayleigh (many equal rays)")
        ph.set_ylim(0, max(0.16, 1.1 * h.max()))
        self.readout(lam=100 * lam, spacing=spacing if np.isfinite(spacing) else "—",
                     deep=100 * np.mean(vals < -10),
                     spread=float(np.percentile(vals, 99) - np.percentile(vals, 1)))

    def story(self, p):
        lam = C0 / (p.f * 1e9)
        deep = self.r.get("deep", 0)
        s = (f"<p>The transmitter (green star) radiates at {v(p.f, '.2f', 'GHz')}, a wavelength of "
             f"{v(100 * lam, '.0f', 'cm')}. Every wall returns a copy weakened by "
             f"|Γ| = {v(p.gamma)}; the <b>image method</b> replaces each bounce by a mirror-image "
             f"transmitter behind the wall. Where the copies arrive in phase you get a hot spot, "
             f"where they cancel a dead spot: a <b>standing-wave speckle</b> with a grain of about "
             f"λ/2 = {v(50 * lam, '.0f', 'cm')}.</p>")
        if p.order == 0 or p.gamma < 0.05:
            s += ("<p>With no reflections there is only the direct ray: power falls smoothly with "
                  "distance and there is nothing to fade.</p>")
        elif deep > 5:
            s += (f"<p>{v(deep, '.1f', '%')} of the room is more than 10 dB below its local mean, "
                  f"approaching the {v(9.5, '.1f', '%')} that <b>Rayleigh</b> statistics predict when "
                  "many rays of similar strength add with random phases (red curve).</p>")
        else:
            s += (f"<p>Only {v(deep, '.1f', '%')} of the room is 10 dB down: the direct ray is "
                  "still the strongest, so the field is closer to <b>Rician</b> than Rayleigh.</p>")
        s += ("<p>Walk along the white line and the power dives every few centimetres (bottom left). "
              "Move a phone by a hand's width and the signal can change by 20 dB: this is why "
              "phones and routers carry two antennas a few centimetres apart.</p>")
        return "<h3>Ripples in a pond</h3>" + s + keybox(
            "Small-scale fading is a pattern in space. Moving through it at speed v turns it into a "
            "fading signal in time, with a Doppler spread of v/λ: experiment 3.")


# =============================================================================== 3. Doppler
class DopplerFading(Experiment):
    title = "Doppler: driving through the speckle"
    blurb = "A live Rayleigh envelope. Speed up and the fades come faster and last shorter."
    book = "sec:ch11:clarke"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Slider("speed", "Speed", 1, 300, 50, step=1, unit="km/h",
               enabled_if=lambda p: not p.direct),
        LogSlider("f", "Carrier frequency", 0.4, 6.0, 2.0, unit="GHz",
                  enabled_if=lambda p: not p.direct),
        Toggle("direct", "Set the Doppler directly", False),
        Slider("fdd", "Max Doppler f_D", 1, 500, 50, step=1, unit="Hz",
               enabled_if=lambda p: p.direct),
        Slider("thr", "Fade threshold", -30, 5, -10, step=0.5, unit="dB re rms",
               help="A fade = the envelope below this level"),
    ]
    plots = [
        Plot("env", "Received envelope |h(t)|² (scrolling, real time)", x="time (s)",
             y="power re mean (dB)", xlim=(0, 1), ylim=(-40, 12), legend=None),
        SpectrumPlot("spec", "Doppler spectrum vs Clarke", x="frequency / f_D",
                     y="PSD (dB)", xlim=(-1.6, 1.6), ylim=(-30, 12), legend=None),
        Plot("acf", "Correlation vs J₀", x="distance moved (wavelengths)",
             y="correlation", xlim=(0, 2.5), ylim=(-0.5, 1.05), legend=None),
        Plot("lcr", "Fade rate vs Rice", x="threshold (dB re rms)",
             y="crossings per second", xlim=(-30, 6), legend=None),
    ]
    layout = [["env", "env", "env"], ["spec", "acf", "lcr"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("fd", "Max Doppler f_D", "Hz", ".1f"),
        Readout("tc", "Coherence time", "ms", ".2f", help="≈ 0.423 / f_D"),
        Readout("rate", "Fades per second (counted)", "/s", ".1f"),
        Readout("afd", "Average fade duration", "ms", ".2f"),
    ]
    challenges = [
        Challenge("Reproduce the book's car: 900 MHz, 60 km/h, threshold −10 dB. Count about 36 "
                  "fades per second (watch for 3 s or more).",
                  lambda s: not s.p.direct and abs(s.p.f - 0.9) < 0.01 and abs(s.p.speed - 60) < 0.6
                  and abs(s.p.thr + 10) < 0.3 and s.exp.T_acc >= 3 and 28 <= s.r.rate <= 44),
        Challenge("Make the coherence time one LTE subframe: 1 ms ± 5 %.",
                  lambda s: abs(s.r.tc - 1.0) <= 0.05),
        Challenge("Walking pace at 900 MHz: make the average 10 dB fade last longer than 40 ms "
                  "(the book's theory says about 50 ms at 2.5 Hz).",
                  lambda s: not s.p.direct and abs(s.p.thr + 10) < 0.3 and s.exp.afd_th > 0.040
                  and abs(s.p.f - 0.9) < 0.02,
                  hint="Fade duration scales as 1/f_D = λ/v."),
    ]
    window = 1.0

    def setup(self):
        self.fader = fdg.SoSFader(1, 48, rng=self.rng)
        self.t_now = 0.0

    def fs_for(self, fd):
        return float(min(max(1000.0, 50 * fd), 40000.0))

    def gen(self, t0, t1, fd, fs):
        t = np.arange(t0, t1, 1 / fs)
        return t, self.fader.gains(t, fd)[:, 0]

    def update(self, p):
        fd = float(p.fdd) if p.direct else float(fdg.doppler_hz(p.speed, p.f * 1e9))
        self.fd, self.fs = fd, self.fs_for(fd)
        self.t_now = max(self.t_now, self.window)
        self.tb, self.hb = self.gen(self.t_now - self.window, self.t_now, fd, self.fs)
        env = db10(np.abs(self.hb) ** 2)
        st_ = fdg.fade_stats(env, p.thr, self.fs)
        self.T_acc, self.n_acc, self.below_acc = self.window, st_["n"], st_["frac"] * self.window
        self.last_below = bool(env[-1] < p.thr)
        self.draw(p)

    def tick(self, p):
        dt = 1.0 / self.fps
        t, h = self.gen(self.t_now, self.t_now + dt, self.fd, self.fs)
        self.t_now = t[-1] + 1 / self.fs if len(t) else self.t_now + dt
        env = db10(np.abs(h) ** 2)
        below = env < p.thr
        prev = np.r_[self.last_below, below[:-1]]
        self.n_acc += int(np.sum(below & ~prev))
        self.below_acc += np.sum(below) / self.fs
        self.T_acc += len(h) / self.fs
        self.last_below = bool(below[-1]) if len(below) else self.last_below
        n = len(h)
        self.tb = np.r_[self.tb[n:], t]
        self.hb = np.r_[self.hb[n:], h]
        self.draw(p)

    def draw(self, p):
        fd, fs = self.fd, self.fs
        env = db10(np.abs(self.hb) ** 2)
        tt = self.tb - self.tb[0]
        pe = self.plot("env")
        k = max(1, len(env) // 2500)                 # display: block minimum keeps every fade
        n = len(env) // k * k
        ed = env[:n].reshape(-1, k).min(axis=1) if k > 1 else env
        td = tt[:n:k]
        below = ed < p.thr
        pe.hband("fz", -40, p.thr, color=RED, alpha=0.07)
        pe.line("e", td, np.clip(ed, -40, 12), color=NAVY, width=1.3)
        if below.any():
            pe.line("eb", td, np.where(below, np.clip(ed, -40, 12), np.nan), color=RED, width=1.8)
        pe.hline("thr", p.thr, color=RED, style="--", label=f"threshold {p.thr:g} dB",
                 label_pos=0.01)
        pe.hline("z", 0, color=GRAY, style=":", width=0.8)
        # Doppler spectrum (normalised frequency)
        ps = self.plot("spec")
        if fd > 0.5:
            nfft = int(2 ** np.round(np.log2(max(64, 12 * fs / fd))))
            f, P = cl.welch_psd(self.hb, fs, min(nfft, len(self.hb)))
            k = np.abs(f) < 1.6 * fd
            mid = np.abs(f) < 0.5 * fd
            off = np.mean(P[mid]) if mid.any() else np.max(P)
            ps.line("P", f[k] / fd, P[k] - off, color=NAVY, width=1.5, name="measured")
            x = np.linspace(-0.995, 0.995, 400)
            th = db10(1 / np.sqrt(1 - x ** 2))
            ps.line("cl", x, th - np.mean(th[np.abs(x) < 0.5]), color=RED, width=2, style="--",
                    name="Clarke's bathtub")
        # autocorrelation vs distance in wavelengths
        h = self.hb
        nl = int(min(len(h) // 3, 2.5 / fd * fs)) if fd > 0 else 10
        lags = np.arange(nl)
        H = np.fft.fft(h, 2 * len(h))
        ac = np.fft.ifft(np.abs(H) ** 2)[:nl].real
        ac = ac / ac[0]
        pa = self.plot("acf")
        dist = lags / fs * fd
        pa.line("sim", dist, ac, color=NAVY, width=1.6, name="measured")
        xx = np.linspace(0, 2.5, 300)
        pa.line("th", xx, j0(2 * np.pi * xx), color=RED, width=2, style="--", name="J₀(2π d/λ)")
        pa.vline("z", 0.383, color=GRAY, style=":", label="first zero 0.38λ", label_pos=0.08)
        pa.hline("zz", 0, color=GRAY, style="-", width=0.6)
        # crossing rate vs threshold
        pl = self.plot("lcr")
        thr_grid = np.arange(-30, 6, 2.5)
        meas = [fdg.fade_stats(env, t_, fs)["rate"] for t_ in thr_grid]
        rr = np.linspace(-30, 6, 300)
        pl.line("th", rr, fdg.lcr_rayleigh(10 ** (rr / 20), fd), color=NAVY, width=2,
                name="Rice's formula")
        pl.scatter("m", thr_grid, meas, color=RED, size=7, name="this 1-s window")
        pl.vline("cur", p.thr, color=RED, style=":")
        pl.set_ylim(0, max(2.0, 1.25 * fd))
        rate = self.n_acc / self.T_acc if self.T_acc > 0 else 0
        afd = self.below_acc / self.n_acc * 1e3 if self.n_acc else float("nan")
        rho = 10 ** (p.thr / 20)
        self.rate_th = float(fdg.lcr_rayleigh(rho, fd))
        self.afd_th = float(fdg.afd_rayleigh(rho, fd)) if fd > 0 else float("inf")
        self.readout(fd=fd, tc=0.423 / fd * 1e3 if fd > 0 else float("nan"), rate=rate,
                     afd=afd if np.isfinite(afd) else "—")

    def story(self, p):
        fd = getattr(self, "fd", 1.0)
        lam = C0 / (p.f * 1e9)
        where = ("You have set" if p.direct else
                 f"At {v(p.speed, '.0f', 'km/h')} and {v(p.f, '.2f', 'GHz')} (λ = "
                 f"{v(100 * lam, '.0f', 'cm')}) you cross")
        s = (f"<p>{where} {v(fd, '.0f')} wavelengths of the speckle "
             f"pattern every second: the <b>maximum Doppler shift</b> f_D = v/λ = "
             f"{v(fd, '.1f', 'Hz')}. Each echo arrives from a different direction and is shifted "
             f"by f_D·cos θ, so the spectrum is spread over ±f_D with horns at the edges "
             f"(bottom left). Red curves are theory: Clarke's bathtub, J₀ and Rice's formula.</p>")
        s += (f"<p>Rice's formula predicts {v(self.__dict__.get('rate_th', 0), '.1f')} fades per "
              f"second below {v(p.thr, '.0f', 'dB')}, each lasting about "
              f"{v(1e3 * self.__dict__.get('afd_th', 0), '.2f', 'ms')}; the counters (top) "
              f"converge to those numbers as you watch. Deep fades are rare <i>and</i> short.</p>")
        s += ("<p>The correlation falls to zero after only 0.38 wavelengths (bottom centre): two "
              "antennas half a wavelength apart fade almost independently. That is the basis of "
              "space diversity.</p>")
        return "<h3>Doppler turns space into time</h3>" + s + keybox(
            "Coherence time ≈ 0.423/f_D. Codes and interleavers must span several fades; pilots must "
            "come faster than the channel changes.")


# =============================================================================== 4. statistics
class FadingStatistics(Experiment):
    title = "Rayleigh, Rice and Nakagami"
    blurb = "How deep and how often: the envelope distribution and the probability of a fade."
    book = "sec:ch11:distributions"
    controls = [
        Choice("kind", "Fading model", ["Rayleigh", "Rician", "Nakagami-m"], "Rician"),
        Slider("K", "Rician K-factor", 0.0, 20.0, 0.0, step=0.1,
               help="Power in the line-of-sight ray divided by the power in the scattered rays",
               enabled_if=lambda p: p.kind == "Rician"),
        Slider("m", "Nakagami m", 0.5, 10.0, 2.0, step=0.05,
               help="m = 1 is Rayleigh; m < 1 is worse; large m approaches no fading",
               enabled_if=lambda p: p.kind == "Nakagami-m"),
        Button("again", "New samples"),
    ]
    plots = [
        Plot("hist", "Envelope |h| (unit mean power)", x="envelope |h|", y="probability density",
             xlim=(0, 3), ylim=(0, 1.2), legend="tr"),
        Plot("cdf", "Probability of a fade deeper than x", x="x: fade depth re mean power (dB)",
             y="P(|h|² < x)", logy=True, xlim=(-40, 6), ylim=(1e-10, 2), legend="br"),
        Plot("p20", "Probability of a 20 dB fade as the model changes", x="K-factor",
             y="P(20 dB fade)", logy=True, ylim=(1e-12, 1), legend=None),
    ]
    layout = [["hist", "cdf"], ["p20", "cdf"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("p10", "10 dB fade", "% of the time", ".2f"),
        Readout("p20", "20 dB fade", "of the time", "sci"),
        Readout("af", "Amount of fading", "", ".3f", help="var(|h|²)/E[|h|²]²: 1 for Rayleigh"),
        Readout("meq", "Equivalent Nakagami m", "", ".2f", help="1 / amount of fading"),
    ]
    challenges = [
        Challenge("Line of sight to the rescue: push a 20 dB fade below 10⁻⁵ with a Rician channel.",
                  lambda s: s.p.kind == "Rician" and s.r.p20 < 1e-5),
        Challenge("Find a model that fades worse than Rayleigh: a 20 dB fade more than 3 % of the time.",
                  lambda s: s.r.p20 > 0.03),
        Challenge("Find the Rician K at which a 10 dB fade happens 1 % of the time (0.9–1.1 %).",
                  lambda s: s.p.kind == "Rician" and 0.9 <= s.r.p10 <= 1.1),
        Challenge("Match a Rician K = 6 channel: pick the Nakagami m with the same amount of fading "
                  "(within 2 %).",
                  lambda s: s.p.kind == "Nakagami-m" and abs(s.p.m - 49 / 13) / (49 / 13) < 0.02,
                  hint="Amount of fading of Rice: (2K+1)/(K+1)²; of Nakagami: 1/m."),
    ]
    N = 120_000

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def params(self, p):
        if p.kind == "Rayleigh":
            return "rayleigh", 0.0, 1.0
        if p.kind == "Rician":
            return "rician", p.K, 1.0
        return "nakagami", 0.0, p.m

    def update(self, p):
        kind, K, m = self.params(p)
        rng = np.random.default_rng(self.seed)
        h = rician_or_nakagami(p.kind, p.K, p.m, self.N, rng)
        r = np.abs(h)
        hh, e = np.histogram(r, bins=90, range=(0, 3), density=True)
        ph = self.plot("hist")
        ph.line("h", e, hh, step=True, color=NAVY, width=1.3, fill=0, fill_alpha=0.22,
                name="simulated")
        x = np.linspace(0, 3, 600)
        pdf = fdg.envelope_pdf(x, kind, K=K, m=m)
        ph.line("pdf", x, pdf, color=RED, width=2.2, name=f"{p.kind} pdf")
        if p.kind != "Rayleigh":
            ph.line("ray", x, fdg.envelope_pdf(x), color=GRAY, width=1.4, style="--", name="Rayleigh")
        ph.set_ylim(0, max(1.0, 1.15 * max(pdf.max(), hh.max())))
        pc = self.plot("cdf")
        xd = np.linspace(-40, 6, 300)
        cdf = fdg.power_cdf(10 ** (xd / 10), kind, K=K, m=m)
        pc.line("ray", xd, fdg.power_cdf(10 ** (xd / 10)), color=GRAY, width=1.4, style="--",
                name="Rayleigh: 1 decade per 10 dB")
        pc.line("th", xd, np.maximum(cdf, 1e-30), color=NAVY, width=2.4, name=f"{p.kind} theory")
        xs = np.arange(-35, 6, 2.5)
        meas = np.array([np.mean(r ** 2 < 10 ** (q / 10)) for q in xs])
        ok = meas > 0
        pc.scatter("sim", xs[ok], meas[ok], color=RED, size=7, name="simulated")
        pc.vline("v10", -10, color=GRAY, style=":", label="10 dB", label_pos=0.95)
        pc.vline("v20", -20, color=GRAY, style=":", label="20 dB", label_pos=0.95)
        p10 = float(fdg.power_cdf(0.1, kind, K=K, m=m))
        p20 = float(fdg.power_cdf(0.01, kind, K=K, m=m))
        pp = self.plot("p20")
        if p.kind == "Nakagami-m":
            mm = np.linspace(0.5, 10, 200)
            pp.line("c", mm, fdg.power_cdf(0.01, "nakagami", m=mm), color=NAVY, width=2)
            pp.scatter("now", [p.m], [p20], color=RED, size=12, symbol="d")
            pp.set_labels(x="Nakagami m")
            pp.set_xlim(0.5, 10)
        else:
            kk = np.linspace(0, 20, 200)
            pp.line("c", kk, fdg.power_cdf(0.01, "rician", K=kk), color=NAVY, width=2)
            pp.scatter("now", [K], [p20], color=RED, size=12, symbol="d")
            pp.set_labels(x="Rician K-factor")
            pp.set_xlim(0, 20)
        if kind == "rician" and K > 0:
            af = (2 * K + 1) / (K + 1) ** 2
        elif kind == "nakagami":
            af = 1 / m
        else:
            af = 1.0
        self.readout(p10=100 * p10, p20=p20, af=af, meq=1 / af)

    def story(self, p):
        p20 = self.r.get("p20", 0.01)
        s = ("<p>When many echoes of similar strength add with random phases, the complex gain is "
             "Gaussian and its magnitude is <b>Rayleigh</b>: a 10 dB fade 10 % of the time, a 20 dB "
             "fade 1 % of the time, one decade per 10 dB (the dashed line, right).</p>")
        if p.kind == "Rician":
            s += (f"<p>Add a steady line-of-sight ray carrying K = {v(p.K, '.1f')} times the "
                  f"scattered power and the envelope clusters around 1: the histogram narrows, and "
                  f"a 20 dB fade now happens only {v(st.sci(p20))} of the time. Satellite, "
                  f"rural and indoor-hotspot links often have K of 5–15.</p>")
        elif p.kind == "Nakagami-m":
            s += (f"<p>Nakagami-m is the engineer's flexible fit: m = {v(p.m, '.2f')}. Its fade "
                  f"probability falls as x^m, so m acts like a diversity order: m = 1 is Rayleigh, "
                  f"m = 2 behaves like two-branch diversity, m < 1 is worse than Rayleigh.</p>")
        else:
            s += ("<p>Try the Rician model: a line of sight is itself a form of diversity. Then try "
                  "Nakagami with m above and below 1.</p>")
        return "<h3>How deep, how often</h3>" + s + keybox(
            "The slope of the fade-probability curve is the diversity order. Error rates in fading "
            "follow it: experiment 7.")


# =============================================================================== 5. echoes in frequency
class EchoesAndNotches(Experiment):
    title = "Echoes become notches"
    blurb = "A delay in time is a ripple in frequency. Delay spread sets the coherence bandwidth."
    book = "sec:ch11:multipath"
    controls = [
        Choice("mode", "Multipath", ["Two paths", "Many paths"]),
        LogSlider("tau", "Echo delay", 0.05, 5.0, 0.5, unit="µs",
                  enabled_if=lambda p: p.mode == "Two paths"),
        Slider("a", "Echo strength", 0.0, 1.0, 0.7, step=0.01, help="Amplitude relative to the direct path",
               enabled_if=lambda p: p.mode == "Two paths"),
        LogSlider("trms", "RMS delay spread", 10, 3000, 300, unit="ns",
                  help="Exponential power-delay profile with this spread",
                  enabled_if=lambda p: p.mode == "Many paths"),
        Heading("Signal"),
        LogSlider("bw", "Signal bandwidth", 0.2, 100.0, 2.0, unit="MHz"),
        Button("again", "New multipath realisation"),
    ]
    plots = [
        Plot("cir", "Impulse response: what one impulse turns into", x="delay (µs)", y="|h| (linear)",
             ylim=(0, 1.15), legend=None),
        Plot("H", "Frequency response |H(f)|²", x="frequency offset (MHz)", y="gain (dB)",
             ylim=(-40, 12), legend=None),
        Plot("corr", "Frequency correlation |R(Δf)|", x="frequency separation Δf (MHz)",
             y="correlation", ylim=(0, 1.05), legend=None),
    ]
    layout = [["H", "H"], ["cir", "corr"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("trms", "RMS delay spread", "ns", ".0f"),
        Readout("bc", "Coherence bandwidth", "MHz", ".3g", help="Where |R(Δf)| first falls to 0.5"),
        Readout("ratio", "Bandwidth / Bc", "", ".2g"),
        Readout("ripple", "In-band ripple", "dB", ".1f", good=lambda x: x < 3),
    ]
    challenges = [
        Challenge("Two paths: place the notches exactly 1 MHz apart.",
                  lambda s: s.p.mode == "Two paths" and abs(s.p.tau - 1.0) < 0.02),
        Challenge("Carve a notch deeper than 25 dB inside your signal band.",
                  lambda s: s.r.ripple > 25),
        Challenge("Many paths: make a 20 MHz Wi-Fi channel flat (ripple under 3 dB).",
                  lambda s: s.p.mode == "Many paths" and 18 <= s.p.bw <= 22 and s.r.ripple < 3),
        Challenge("Make the signal at least 10× wider than the coherence bandwidth (a job for OFDM).",
                  lambda s: s.r.ratio >= 10),
    ]

    def setup(self):
        self.seed = 5

    def on_again(self, p):
        self.seed += 1

    def taps(self, p):
        if p.mode == "Two paths":
            d = np.array([0.0, p.tau * 1e-6])
            g = np.array([1.0, p.a * np.exp(1j * 2.0)]) / np.sqrt(1 + p.a ** 2)
            return d, g
        rng = np.random.default_rng(self.seed)
        tr = p.trms * 1e-9
        n = 40
        d = np.sort(np.r_[0.0, rng.exponential(tr, n - 1) * 1.0])
        pw = np.exp(-d / tr)
        g = np.sqrt(pw / pw.sum()) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        g = g / np.sqrt(np.sum(np.abs(g) ** 2))
        return d, g

    def update(self, p):
        d, g = self.taps(p)
        pw = np.abs(g) ** 2
        m1 = np.sum(pw * d) / pw.sum()
        trms = np.sqrt(max(np.sum(pw * d ** 2) / pw.sum() - m1 ** 2, 0))
        bw = p.bw * 1e6
        span = 0.75 * bw
        f = np.linspace(-span, span, 4001)
        H = np.exp(-2j * np.pi * f[:, None] * d[None, :]) @ g
        Hdb = db10(np.abs(H) ** 2)
        inb = np.abs(f) <= bw / 2
        ripple = float(min(Hdb[inb].max() - Hdb[inb].min(), 60))
        pH = self.plot("H")
        pH.band("sig", -bw / 2e6, bw / 2e6, color=GREEN, alpha=0.10)
        pH.line("H", f / 1e6, np.clip(Hdb, -40, 12), color=NAVY, width=1.8)
        pH.set_xlim(-span / 1e6, span / 1e6)
        pH.text("lab", -bw / 2e6, 10, "your signal band", color=GREEN, size=8.5, anchor=(0, 0))
        pc = self.plot("cir")
        pc.stems("h", d * 1e6, np.abs(g), color=NAVY, size=6)
        pc.set_xlim(-0.05 * max(d.max(), 0.2e-6) * 1e6, 1.08 * max(d.max(), 0.2e-6) * 1e6)
        if trms > 0:
            pc.vline("tr", (m1 + trms) * 1e6, color=RED, style=":", label="mean + τ_rms", label_pos=0.9)
        # frequency correlation from the power delay profile
        df = np.linspace(0, max(span, 5 / max(trms, 1e-9) * 0.3), 1500)
        R = np.abs(np.exp(-2j * np.pi * df[:, None] * d[None, :]) @ pw) / pw.sum()
        below = np.flatnonzero(R < 0.5)
        bc = df[below[0]] if len(below) else float("inf")
        pr_ = self.plot("corr")
        pr_.line("R", df / 1e6, R, color=NAVY, width=2, name="|R(Δf)|")
        pr_.hline("half", 0.5, color=GRAY, style=":")
        if np.isfinite(bc):
            pr_.vline("bc", bc / 1e6, color=RED, style="--", label=f"B_c = {bc / 1e6:.3g} MHz",
                      label_pos=0.9)
        if trms > 0:
            pr_.vline("rule", 1 / (5 * trms) / 1e6, color=ORANGE, style=":",
                      label="1/(5τ_rms)", label_pos=0.55)
        pr_.set_xlim(0, df[-1] / 1e6)
        self.readout(trms=trms * 1e9, bc=bc / 1e6 if np.isfinite(bc) else "∞",
                     ratio=bw / bc if np.isfinite(bc) else 0.0, ripple=ripple)

    def story(self, p):
        rip = self.r.get("ripple", 0)
        if p.mode == "Two paths":
            s = (f"<p>An echo {v(p.tau, '.2f', 'µs')} late adds a delayed copy. In frequency, the two "
                 f"copies alternately add and cancel: notches every 1/τ = {v(1 / p.tau, '.3g', 'MHz')}, "
                 f"as deep as the echo is strong ({v(p.a)} of the direct path). The same comb filter "
                 f"gave analog TV its ghosts and makes a flanger sound like a jet.</p>")
        else:
            s = (f"<p>Forty echoes with an exponential power-delay profile of "
                 f"{v(self.r.get('trms', 0), '.0f', 'ns')} rms. Their sum is a random, bumpy frequency "
                 f"response. Two frequencies closer than the <b>coherence bandwidth</b> "
                 f"(≈ {v(self.r.get('bc', 0) if not isinstance(self.r.get('bc'), str) else 0, '.3g', 'MHz')}) "
                 f"fade together; farther apart they fade independently.</p>")
        if rip < 3:
            s += (f"<p>Your {v(p.bw, '.3g', 'MHz')} signal sees an almost flat channel "
                  f"(ripple {v(rip, '.1f', 'dB')}): <b>flat fading</b>. One complex gain describes it, "
                  "and no equalizer is needed.</p>")
        else:
            s += (f"<p>Your {v(p.bw, '.3g', 'MHz')} signal sees {v(rip, '.0f', 'dB')} of ripple across "
                  f"its band: <b>frequency-selective fading</b>. In time, that is inter-symbol "
                  f"interference; you need an equalizer (Chapter 12) or OFDM (Chapter 17).</p>")
        return "<h3>Delay spread ↔ coherence bandwidth</h3>" + s + keybox(
            "B_c ≈ 1/(5·τ_rms) is only a rule of thumb (compare the orange and red lines), but the "
            "inverse relation is exact: longer echoes, narrower fades.")


# =============================================================================== 6. TDL time-frequency
class TimeFrequency(Experiment):
    title = "3GPP channel models in motion"
    blurb = "EPA, EVA and ETU as a live picture of |H(f, t)|: stripes from delay, motion from Doppler."
    book = "sec:ch11:standardmodels"
    animate = True
    fps = 12
    controls = [
        Choice("prof", "Profile (TS 36.101)", ["EPA", "EVA", "ETU"], "EVA"),
        LogSlider("scale", "Delay-spread scale", 0.25, 4.0, 1.0, unit="×",
                  help="Stretch all delays, as TR 38.901 scales its normalised TDL profiles"),
        Slider("fd", "Max Doppler f_D", 1, 500, 70, step=1, unit="Hz"),
        Choice("bw", "Bandwidth shown", ["5 MHz", "10 MHz", "20 MHz"], "10 MHz"),
    ]
    plots = [
        ImagePlot("tf", "|H(f, t)|² in dB  (Play: the channel evolves)", x="frequency (MHz)",
                  y="time (ms)"),
        Plot("pdp", "Power-delay profile", x="delay (µs)", y="relative power (dB)",
             ylim=(-25, 3), legend=None),
        Plot("corr", "Frequency correlation", x="Δf (MHz)", y="|R(Δf)|", ylim=(0, 1.05), legend=None),
    ]
    layout = [["tf", "pdp"], ["tf", "corr"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("trms", "RMS delay spread", "ns", ".0f"),
        Readout("bc", "B_c ≈ 1/(5τ_rms)", "kHz", ".0f"),
        Readout("tc", "Coherence time", "ms", ".2f"),
        Readout("sc", "15 kHz subcarriers per B_c", "", ".0f"),
    ]
    challenges = [
        Challenge("Scale EVA so its coherence bandwidth 1/(5τ_rms) is 1 MHz ± 5 %.",
                  lambda s: s.p.prof == "EVA" and abs(s.r.bc - 1000) <= 50),
        Challenge("Make the coherence time shorter than one LTE subframe (1 ms).",
                  lambda s: s.r.tc < 1.0),
        Challenge("Find a channel that is flat over the whole band shown and nearly frozen: "
                  "B_c wider than the band and T_c longer than 50 ms.",
                  lambda s: s.r.bc * 1e3 > float(s.p.bw.split()[0]) * 1e6 and s.r.tc > 50),
    ]
    Tspan, nt, nf = 50e-3, 220, 256

    def setup(self):
        self.fader = fdg.SoSFader(9, 24, rng=self.rng)
        self.t0 = 0.0

    def update(self, p):
        self.draw(p)

    def tick(self, p):
        self.t0 += 1e-3
        self.draw(p)

    def draw(self, p):
        dly, pdb = cl.TDL_PROFILES[p.prof]
        d = np.array(dly) * 1e-9 * p.scale
        pw = 10 ** (np.array(pdb) / 10)
        pw = pw / pw.sum()
        B = float(p.bw.split()[0]) * 1e6
        f = np.linspace(-B / 2, B / 2, self.nf)
        t = self.t0 + np.linspace(0, self.Tspan, self.nt)
        G = self.fader.gains(t, p.fd)[:, :len(d)] * np.sqrt(pw)[None, :]
        H = G @ np.exp(-2j * np.pi * d[:, None] * f[None, :])
        Hdb = db10(np.abs(H) ** 2)
        ptf = self.plot("tf")
        ptf.image("img", np.clip(Hdb, -30, 10), x=(-B / 2e6, B / 2e6), y=(0, self.Tspan * 1e3),
                  cmap="heat", levels=(-30, 10), colorbar=True, cbar_label="dB")
        ptf.set_xlim(-B / 2e6, B / 2e6)
        ptf.set_ylim(0, self.Tspan * 1e3)
        m1 = np.sum(pw * d)
        trms = np.sqrt(np.sum(pw * d ** 2) - m1 ** 2)
        pp = self.plot("pdp")
        pp.stems("s", d * 1e6, 10 * np.log10(pw / pw.max()), color=NAVY, base=-25)
        pp.set_xlim(-0.1 * d.max() * 1e6 - 0.05, 1.1 * d.max() * 1e6 + 0.05)
        df = np.linspace(0, B, 800)
        R = fdg.freq_correlation(d, 10 * np.log10(pw), df)
        pc = self.plot("corr")
        pc.line("R", df / 1e6, R, color=NAVY, width=2)
        pc.hline("h", 0.5, color=GRAY, style=":")
        bc = 1 / (5 * trms)
        if bc < B:
            pc.vline("bc", bc / 1e6, color=RED, style="--", label="1/(5τ_rms)", label_pos=0.9)
        pc.set_xlim(0, B / 1e6)
        self.readout(trms=trms * 1e9, bc=bc / 1e3, tc=0.423 / p.fd * 1e3, sc=bc / 15e3)

    def story(self, p):
        trms = self.r.get("trms", 300)
        tc = self.r.get("tc", 1)
        s = (f"<p>Each row of the picture is the channel's frequency response at one instant; time "
             f"runs upward over 50 ms. {p.prof} scaled by {v(p.scale, '.2f', '×')} has an rms delay "
             f"spread of {v(trms, '.0f', 'ns')}: the fades in frequency (vertical stripes) are about "
             f"{v(1e3 / max(trms, 1) / 5, '.2f', 'MHz')} wide.</p>")
        if tc > 50:
            s += (f"<p>With f_D = {v(p.fd, '.0f', 'Hz')} the coherence time is "
                  f"{v(tc, '.0f', 'ms')}: the stripes are frozen over the whole picture. A pedestrian "
                  "user's channel barely changes during a frame.</p>")
        else:
            s += (f"<p>With f_D = {v(p.fd, '.0f', 'Hz')} the coherence time is {v(tc, '.2f', 'ms')}: "
                  "the stripes break up into a moving speckle. Press Play to watch it evolve "
                  "(slow motion: 1 ms per frame).</p>")
        s += ("<p>An OFDM system sees each 15 kHz subcarrier as flat; the pilot pattern must sample "
              "this picture finely enough in both directions to interpolate between pilots.</p>")
        return "<h3>Stripes and speckle</h3>" + s + keybox(
            "Delay spread sets the grain in frequency (B_c), Doppler the grain in time (T_c). "
            "Their product decides whether a channel can be tracked at all.")


# =============================================================================== 7. BER and diversity
class FadingBER(Experiment):
    title = "Error rate in fading, and diversity"
    blurb = "Fading turns waterfalls into slopes. Independent branches turn them back."
    book = "sec:ch11:ber"
    heavy = True
    controls = [
        IntSlider("L", "Diversity branches L", 1, 8, 1),
        Choice("comb", "Combining", ["Maximal ratio", "Selection"]),
        Slider("K", "Line-of-sight K-factor", 0.0, 20.0, 0.0, step=0.5),
        Toggle("fixed", "Hold the total received energy fixed", False,
               help="On: each branch gets 1/L of the energy, so only the diversity effect remains"),
    ]
    plots = [BERPlot("ber", "BPSK bit error rate vs average Eb/N0", x="average Eb/N0 per bit (dB)",
                     ylim=(1e-6, 0.5), xlim=(0, 40), legend="tr")]
    readouts = [
        Readout("need", "Eb/N0 for BER 10⁻⁴", "dB", ".1f"),
        Readout("gain", "Gain over one branch", "dB", ".1f"),
        Readout("gap", "Distance from AWGN", "dB", ".1f", good=lambda x: x < 5),
        Readout("slope", "High-SNR slope", "decades / 10 dB", ".1f"),
    ]
    challenges = [
        Challenge("Get within 5 dB of the AWGN curve at BER 10⁻⁴.",
                  lambda s: s.r.gap < 5),
        Challenge("Buy back more than 17 dB at 10⁻⁴ with just two branches (no line of sight).",
                  lambda s: s.p.L == 2 and s.p.K == 0 and s.r.gain > 17),
        Challenge("One branch only: let a line of sight win back at least 15 dB at 10⁻⁴.",
                  lambda s: s.p.L == 1 and s.r.gain > 15),
        Challenge("Make the slope at least 4 decades per 10 dB, holding the total energy fixed.",
                  lambda s: s.p.fixed and s.r.slope >= 3.95),
    ]

    def theory(self, p, e):
        eb = e - (10 * np.log10(p.L) if p.fixed else 0.0)
        return fdg.ber_bpsk_diversity(eb, L=p.L, K=p.K,
                                      combining="mrc" if p.comb == "Maximal ratio" else "sc")

    @staticmethod
    def needed(x, y, target=1e-4):
        ly = np.log10(np.maximum(y, 1e-300))
        i = np.flatnonzero(ly <= np.log10(target))
        if len(i) == 0 or i[0] == 0:
            return float("nan")
        i = i[0]
        return float(np.interp(np.log10(target), [ly[i], ly[i - 1]], [x[i], x[i - 1]]))

    def update(self, p):
        x = np.linspace(0, 40, 201)
        pb = self.plot("ber")
        aw = cl.ber_bpsk(x)
        ray = cl.ber_bpsk_rayleigh(x)
        th = self.theory(p, x)
        pb.theory("awgn", x, aw, color=GRAY, style="--", width=2.0, name="AWGN (no fading)")
        pb.theory("ray", x, ray, color=ORANGE, style=":", width=2.0, name="Rayleigh, one branch")
        lab = (f"L = {p.L}, {'MRC' if p.comb == 'Maximal ratio' else 'selection'}"
               + (f", K = {p.K:g}" if p.K else ""))
        pb.theory("th", x, np.maximum(th, 1e-12), color=NAVY, width=2.4, name=lab + " (theory)")
        pb.hline("e4", 1e-4, color=GRAY, style=":", width=0.8)
        need = self.needed(x, th)
        ref = self.needed(x, ray)
        awn = self.needed(x, aw)
        # high-SNR slope (decades per 10 dB) around BER 1e-6, from the theory on a wider grid
        xw = np.linspace(0, 90, 451)
        e6 = self.needed(xw, self.theory(p, xw), 1e-6)
        slope = float("nan")
        if np.isfinite(e6):
            pa, pb_ = self.theory(p, np.array([e6, e6 + 3.0]))
            slope = (np.log10(pa) - np.log10(max(pb_, 1e-300))) / 0.3
        self.sim = dict(x=[], y=[])
        self.readout(need=need if np.isfinite(need) else "> 40", gain=ref - need if np.isfinite(need) else 0.0,
                     gap=need - awn if np.isfinite(need) else 99.0,
                     slope=slope if np.isfinite(slope) else 0.0)

    def background(self, p):
        rng = np.random.default_rng(7)
        quick = self.quick
        target, cap = (40, 1e5) if quick else (100, 2e6)
        Kk = p.K
        for e in np.arange(0, 41, 2.5):
            eb = e - (10 * np.log10(p.L) if p.fixed else 0.0)
            n0 = 10 ** (-eb / 10)
            errs = bits = 0
            while errs < target and bits < cap:
                n = 50_000
                b = rng.integers(0, 2, n)
                s = 1 - 2.0 * b
                h = (np.sqrt(Kk / (Kk + 1)) + np.sqrt(1 / (Kk + 1)) *
                     (rng.standard_normal((p.L, n)) + 1j * rng.standard_normal((p.L, n))) / np.sqrt(2))
                w = np.sqrt(n0 / 2) * (rng.standard_normal((p.L, n)) + 1j * rng.standard_normal((p.L, n)))
                y = h * s + w
                if p.comb == "Maximal ratio":
                    z = np.sum(np.conj(h) * y, axis=0).real
                else:
                    k = np.argmax(np.abs(h), axis=0)
                    z = (np.conj(h[k, np.arange(n)]) * y[k, np.arange(n)]).real
                errs += int(np.sum((z < 0) != b.astype(bool)))
                bits += n
            ber = errs / bits
            yield dict(x=e, y=ber)
            if ber < (1e-5 if quick else 2e-6) or errs == 0:
                break

    def progress(self, p, it):
        if it["y"] > 0:
            self.sim["x"].append(it["x"])
            self.sim["y"].append(it["y"])
        if self.sim["x"]:
            self.plot("ber").sim("sim", self.sim["x"], self.sim["y"], color=RED, name="Monte Carlo")

    def story(self, p):
        need = self.r.get("need", 0)
        gain = self.r.get("gain", 0)
        s = ("<p>In AWGN the error rate falls off a cliff. In Rayleigh fading it falls only one decade "
             "per 10 dB, because almost all errors happen during the rare deep fades, and the "
             "probability of a fade falls only as 1/SNR. At 10⁻⁴ the price is about 26 dB.</p>")
        if p.L > 1:
            s += (f"<p>With {v(p.L, 'd')} independently fading branches, all must fade at once: the "
                  f"curve falls about {v(p.L, 'd')} decades per 10 dB and needs "
                  f"{v(need if isinstance(need, float) else 40, '.1f', 'dB')}, a gain of "
                  f"{v(gain, '.1f', 'dB')} over one branch.")
            if p.fixed:
                s += " Even with the total energy held fixed, the gain is huge: it is diversity, not power."
            if p.comb == "Selection":
                s += (" Selection combining just picks the strongest branch; maximal-ratio "
                      "combining weights and adds them all, and is about 1–2 dB better.")
            s += "</p>"
        if p.K > 0:
            s += (f"<p>A line of sight with K = {v(p.K, '.1f')} keeps the envelope away from zero, "
                  "steepening the curve toward AWGN: a strong direct ray is itself a form of "
                  "diversity.</p>")
        return "<h3>Exponential to polynomial and back</h3>" + s + keybox(
            "Diversity order = slope. Time (coding + interleaving), frequency (OFDM, Rake), space "
            "(antennas): every modern system is built to collect it.")


# =============================================================================== 8. sounding
class ChannelSounding(Experiment):
    title = "Sounding a channel"
    blurb = "Transmit a sequence with a perfect autocorrelation, correlate, and the echoes appear."
    book = "sec:ch11:sounding"
    controls = [
        Choice("seq", "Sounding sequence", ["Zadoff–Chu", "Random QPSK"]),
        Choice("N", "Sequence length", ["63", "127", "255", "511", "1023"], "255", style="menu"),
        IntSlider("P", "Periods averaged", 1, 64, 4),
        Slider("snr", "SNR at the receiver", -20, 30, 0, step=1, unit="dB"),
        Button("again", "New channel"),
    ]
    plots = [
        Plot("cir", "Estimated impulse response", x="delay (samples)", y="|h| (dB)",
             xlim=(-1, 60), ylim=(-50, 3), legend="tr"),
        Plot("acf", "Periodic autocorrelation of the sequence", x="lag (samples)",
             y="|correlation| (dB)", ylim=(-70, 3), legend=None),
        Plot("gain", "Estimation error vs processing gain", x="processing gain N·P (dB)",
             y="NMSE (dB)", xlim=(15, 50), ylim=(-50, 15), legend="tr"),
    ]
    layout = [["cir", "cir"], ["acf", "gain"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("pg", "Processing gain N·P", "dB", ".1f"),
        Readout("nmse", "Estimation error (NMSE)", "dB", ".1f", good=lambda x: x < -20),
        Readout("side", "Sequence sidelobes", "dB", ".1f"),
        Readout("taps", "Echoes found", "", None),
    ]
    challenges = [
        Challenge("Below the noise: at an SNR of −10 dB or less, estimate the channel with NMSE "
                  "under −17 dB.",
                  lambda s: s.p.snr <= -10 and s.r.nmse < -17,
                  hint="Every doubling of N·P buys 3 dB."),
        Challenge("See why sounders use CAZAC sequences: a 1023-symbol random QPSK sequence at "
                  "SNR ≥ 25 dB is still stuck above −15 dB NMSE.",
                  lambda s: s.p.seq == "Random QPSK" and s.p.snr >= 25 and s.p.N == "1023"
                  and s.r.nmse > -15),
        Challenge("Be frugal: NMSE under −14 dB at 0 dB SNR sending at most 2048 symbols (N·P).",
                  lambda s: s.p.snr <= 0 and int(s.p.N) * s.p.P <= 2048 and s.r.nmse < -14),
    ]
    W = 60

    def setup(self):
        self.on_again(None)

    def on_again(self, p):
        rng = self.rng
        d = np.sort(rng.choice(np.arange(1, 45), 5, replace=False))
        d = np.r_[0, d]
        pw = np.exp(-d / 12.0)
        g = np.sqrt(pw) * (rng.standard_normal(6) + 1j * rng.standard_normal(6)) / np.sqrt(2)
        g[0] = np.abs(g[0]) + 0.5
        self.h = np.zeros(self.W, complex)
        self.h[d] = g
        self.h /= np.linalg.norm(self.h)
        self.noise_seed = int(rng.integers(1 << 30))

    @lru_cache(maxsize=16)
    def sequence(self, kind, N):
        if kind == "Zadoff–Chu":
            return cl.zadoff_chu(1 if N % 2 else 1, N)
        r = np.random.default_rng(N)
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * r.integers(0, 4, N)))

    def estimate(self, kind, N, P, snr, seed):
        s = self.sequence(kind, N)
        tx = np.tile(s, P + 1)
        rx = np.convolve(tx, self.h)[:len(tx)]
        rng = np.random.default_rng(seed)
        n0 = 10 ** (-snr / 10)
        rx = rx + np.sqrt(n0 / 2) * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
        seg = rx[N:N * (P + 1)].reshape(P, N).mean(axis=0)
        cir = np.fft.ifft(np.fft.fft(seg) * np.conj(np.fft.fft(s))) / N
        return cir[:self.W]

    def update(self, p):
        N = int(p.N)
        cir = self.estimate(p.seq, N, p.P, p.snr, self.noise_seed)
        nmse = db10(np.sum(np.abs(cir - self.h) ** 2) / np.sum(np.abs(self.h) ** 2))
        pc = self.plot("cir")
        k = np.arange(self.W)
        est_db = np.maximum(20 * np.log10(np.abs(cir) + 1e-12), -50)
        floor = 10 * np.log10(10 ** (-p.snr / 10) / (N * p.P))
        pc.hband("noise", -50, floor + 6, color=GRAY, alpha=0.12)
        pc.stems("est", k, est_db, color=NAVY, base=-50, size=5, name="estimate")
        nz = np.flatnonzero(self.h)
        pc.scatter("true", nz, 20 * np.log10(np.abs(self.h[nz])), color=RED, size=12, symbol="x",
                   name="true echo")
        pc.hline("fl", floor, color=GRAY, style="--", label="noise floor after averaging",
                 label_pos=0.6)
        found = int(np.sum(est_db[nz] > floor + 6))
        s = self.sequence(p.seq, N)
        ac = np.fft.ifft(np.abs(np.fft.fft(s)) ** 2) / N
        ac_db = np.maximum(20 * np.log10(np.abs(ac) + 1e-12), -70)
        pa = self.plot("acf")
        lags = np.arange(-N // 2, N // 2)
        pa.line("ac", lags, np.fft.fftshift(ac_db), color=NAVY, width=1.2)
        pa.set_xlim(-N / 2, N / 2)
        side = float(np.max(ac_db[1:]))
        pg = self.plot("gain")
        x = np.linspace(15, 50, 100)
        theory = 10 * np.log10(self.W) - p.snr - x
        pg.line("th", x, theory, color=NAVY, width=2, name="noise only: W/(SNR·N·P)")
        pg.scatter("now", [10 * np.log10(N * p.P)], [nmse], color=RED, size=12, symbol="d",
                   name="this measurement")
        self.readout(pg=10 * np.log10(N * p.P), nmse=nmse, side=side,
                     taps=f"{found} of {len(nz)}")

    def story(self, p):
        N = int(p.N)
        s = (f"<p>The transmitter repeats a known sequence of {v(N, 'd')} symbols; the receiver "
             f"averages {v(p.P, 'd')} periods and cross-correlates with the original. Because the "
             f"sequence's correlation with itself is a single spike, the output is the channel's "
             f"impulse response: each echo appears at its delay (red crosses = truth).</p>"
             f"<p>Correlation collects the energy of N·P symbols: a processing gain of "
             f"{v(self.r.get('pg', 0), '.1f', 'dB')}, which is why the echoes stand out even at "
             f"{v(p.snr, '.0f', 'dB')} SNR.</p>")
        if p.seq == "Random QPSK":
            s += (f"<p>A random sequence's sidelobes sit near {v(self.r.get('side', 0), '.0f', 'dB')}: "
                  "they leak every echo into every other delay, an error floor no averaging removes. "
                  "Zadoff–Chu sequences have a perfectly flat periodic autocorrelation (CAZAC); that is "
                  "why LTE and NR use them for random access and sounding.</p>")
        return "<h3>Measuring the channel</h3>" + s + keybox(
            "Real sounders (and gnuradio/gr03_channel_sounder.py with a USRP) do exactly this, and "
            "measure τ_rms, B_c and the Doppler spectrum from the result.")


# =============================================================================== the lab
LAB = st.Lab(5, "The Wireless Channel", chapter=11, chapter_title="The Wireless Channel",
             experiments=[PathLossShadowing, RoomSpeckle, DopplerFading, FadingStatistics,
                          EchoesAndNotches, TimeFrequency, FadingBER, ChannelSounding])

if __name__ == "__main__":
    st.run(LAB)
