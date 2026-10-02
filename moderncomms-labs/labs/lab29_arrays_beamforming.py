"""Lab 29 · Antenna Arrays and Beamforming   (Chapter 19)

Run it:      python labs/lab29_arrays_beamforming.py
Self-test:   python labs/lab29_arrays_beamforming.py --selftest

A 28 GHz base-station panel is a matchbox-sized grid of 64 patches, each with its own amplifier
and phase shifter; together they buy the 20-30 dB that makes millimetre waves usable. Seven
experiments build that panel and use it: steer a line of antennas and meet grating lobes, size
a planar panel and its link budget, watch phase shifters squint across a wide band, find
directions with Bartlett, MVDR and MUSIC, sweep NR's SSB beams and pick a codebook precoder,
factor a precoder into analog and digital parts, and space the antennas of a line-of-sight
MIMO hop. Helpers are in commlib.mimokit, the code behind Chapter 19's array figures.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

import commlib as cl
from commlib import mimokit as mk
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, PolarPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

C0 = mk.C0
TAPERS = ["Uniform", "Taylor", "Chebyshev", "Hann"]


def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def undb(x):
    return 10 ** (np.asarray(x, float) / 10)


def waterfall(plot, key, items, total_name, fmt="{:+.1f}"):
    """A decibel 'bank statement' (as in Lab 18): floating bars from the running balance."""
    run = 0.0
    names = []
    for i, (nm, val) in enumerate(items):
        lo, hi = sorted([run, run + val])
        plot.bars(f"{key}{i}", [i], [hi], base=lo, width=0.62, color=GREEN if val >= 0 else RED)
        plot.text(f"{key}t{i}", i, hi, fmt.format(val), anchor=(0.5, 1.05), size=8.5, bold=True)
        if i < len(items) - 1:
            plot.line(f"{key}c{i}", [i + 0.31, i + 0.69], [run + val] * 2, color=GRAY, width=1.0)
        run += val
        names.append(nm)
    n = len(items)
    lo, hi = sorted([0.0, run])
    plot.bars(f"{key}T", [n], [hi], base=lo, width=0.62, color=NAVY)
    plot.text(f"{key}tT", n, hi, f"{run:.1f}", anchor=(0.5, 1.05), size=9, bold=True, color=NAVY)
    plot.set_xticks([(i, s) for i, s in enumerate(names + [total_name])])
    plot.set_xlim(-0.6, n + 0.6)
    return run


# =============================================================================== 1. ULA
class SteerArray(Experiment):
    title = "Steer a line of antennas"
    blurb = "Phase the elements to point the beam; space them too far and ghosts appear."
    book = "sec:ch19:arrays"
    controls = [
        IntSlider("N", "Elements", 2, 64, 16),
        Slider("steer", "Steering angle", -80.0, 80.0, 20.0, step=0.5, unit="°"),
        Slider("d", "Element spacing", 0.25, 1.5, 0.5, step=0.01, unit="λ"),
        Heading("Taper"),
        Choice("taper", "Amplitude taper", TAPERS, "Uniform", style="menu"),
        Slider("sll", "Design sidelobe level", 20.0, 50.0, 30.0, step=1.0, unit="dB",
               enabled_if=lambda p: p.taper in ("Taylor", "Chebyshev")),
    ]
    plots = [
        PolarPlot("pol", "Array pattern (front half-space)", floor_db=-40, legend=None),
        Plot("cart", "Array factor", x="angle from broadside (°)", y="gain re one element (dB)",
             xlim=(-90, 90), legend="tl", legend_cols=2),
        Plot("u", "The pattern repeats every 1/d in sin θ: only |sin θ| ≤ 1 is real",
             x="u = sin θ", y="dB re peak", xlim=(-3, 3), ylim=(-45, 5), legend=None),
    ]
    layout = [["pol", "cart"], ["pol", "u"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("gain", "Array gain", "dB", ".1f"),
        Readout("hpbw", "Half-power beamwidth", "°", ".1f"),
        Readout("psl", "Peak sidelobe", "dB", ".1f", good=lambda x: x < -25),
        Readout("gl", "Grating lobes", "", None),
    ]
    challenges = [
        Challenge("Steer to 60° with the largest spacing that still has no grating lobe "
                  "(d within 0.02 λ of the limit).",
                  lambda s: (abs(s.p.steer) >= 59.9 and s.r.gl == "none"
                             and s.p.d >= 1 / (1 + np.sin(np.radians(abs(s.p.steer)))) - 0.02)),
        Challenge("Sidelobes 30 dB down or better with a beamwidth no wider than 8°.",
                  lambda s: s.r.psl <= -30 and s.r.hpbw <= 8,
                  hint="Tapers lower the sidelobes but widen the beam: buy it back with elements."),
        Challenge("Make a broadside array with a full-height grating lobe at endfire.",
                  lambda s: abs(s.p.steer) < 0.3 and s.p.d >= 0.995 and s.r.gl != "none"),
    ]

    def weights(self, p):
        return mk.taper(p.N, p.taper, p.sll)

    def update(self, p):
        th = np.radians(np.linspace(-90, 90, 1441))
        th0 = np.radians(p.steer)
        w = self.weights(p)
        af = mk.ula_af(w, th, p.d, th0)
        gdb = db(af)
        g0 = float(db(np.abs(w.sum()) ** 2 / np.sum(w ** 2)))
        rel = gdb - gdb.max()
        pp = self.plot("pol")
        pp.pattern("af", th, rel, color=NAVY, width=2.2)
        pp.line("dir", [0, 1.05 * np.sin(th0)], [0, 1.05 * np.cos(th0)], color=RED, style="--",
                width=1.2)
        gls = mk.grating_lobes(p.d, th0)
        pc = self.plot("cart")
        if p.taper != "Uniform":
            afu = db(mk.ula_af(np.ones(p.N), th, p.d, th0))
            pc.line("uni", np.degrees(th), afu, color=GRAY, width=1.2, name="uniform")
        pc.line("af", np.degrees(th), gdb, color=NAVY, width=2.2, name=p.taper)
        pc.vline("st", p.steer, color=RED, style="--", label="steered", label_pos=0.9)
        for i, g in enumerate(gls):
            pc.vline(f"gl{i}", np.degrees(g), color=ORANGE, style=":", width=1.6,
                     label="grating lobe", label_pos=0.7)
        top = db(p.N) + 3
        pc.set_ylim(top - 50, top + 4)
        # u-space
        u = np.linspace(-3, 3, 1801)
        afu = db(mk.af_u(w, u, p.d, np.sin(th0)))
        pu = self.plot("u")
        pu.band("vis", -1, 1, color=GREEN, alpha=0.12)
        pu.text("vl", 0, 3.5, "visible: real angles", color=GREEN, anchor=(0.5, 0), size=8.5)
        pu.line("af", u, afu - afu.max(), color=NAVY, width=1.6)
        pu.vline("u0", np.sin(th0), color=RED, style="--")
        for m in range(-6, 7):
            if m:
                pu.vline(f"r{m}", np.sin(th0) + m / p.d, color=ORANGE, style=":", width=1.0)
        _, hp, psl = mk.beam_metrics(th, gdb)
        mainlobe_db = gdb.max()
        if len(gls):
            psl = max(psl, float(np.max(np.interp(gls, th, gdb)) - mainlobe_db))
        self.readout(gain=g0, hpbw=float(np.degrees(hp)), psl=psl,
                     gl="none" if len(gls) == 0 else f"{len(gls)} at " +
                     ", ".join(f"{np.degrees(g):.0f}°" for g in gls))

    def story(self, p):
        r = self.r
        lim = 1 / (1 + np.sin(np.radians(abs(p.steer))))
        s = (f"<p>{v(p.N, 'd')} antennas a {v(p.d, '.2f', 'λ')} apart, each delayed so their "
             f"signals add in phase towards {v(p.steer, '.0f', '°')}. Coherent addition gives "
             f"{v(r.get('gain', 0), '.1f', 'dB')} of array gain and a beam "
             f"{v(r.get('hpbw', 0), '.1f', '°')} wide (about 101.5°/(N·cos θ₀) at λ/2).</p>"
             f"<p>The bottom plot shows why spacing matters: the array factor is periodic in "
             f"u = sin θ with period 1/d = {v(1 / p.d, '.2f')}, and only the green window is real "
             f"space. Steering slides the whole comb; when a copy of the main lobe slides into the "
             f"window it is a <b>grating lobe</b>, a full-strength beam in the wrong direction. "
             f"At {v(abs(p.steer), '.0f', '°')} the spacing must stay below "
             f"{v(lim, '.2f', 'λ')}.</p>")
        if p.taper != "Uniform":
            s += (f"<p>The {v(p.taper)} taper turns down the edge elements: sidelobes fall from "
                  f"−13 dB to {v(r.get('psl', 0), '.1f', 'dB')}, the main lobe widens and the gain "
                  f"drops a little (grey: uniform).</p>")
        return "<h3>Pointing without moving</h3>" + s + keybox(
            "No grating lobes for any steering angle: d ≤ λ/2. Uniform weights: −13.3 dB first "
            "sidelobes; tapers trade beamwidth for lower sidelobes.")


# =============================================================================== 2. planar panel
class PlanarPanel(Experiment):
    title = "A planar panel and its EIRP"
    blurb = "An 8 × 8 grid of patches at 28 GHz: beam, EIRP and the millimetre-wave link budget."
    book = "sec:ch19:arrays"
    controls = [
        Heading("Panel"),
        IntSlider("nx", "Columns", 1, 32, 8),
        IntSlider("ny", "Rows", 1, 32, 8),
        Slider("az", "Steer: azimuth", -60.0, 60.0, 0.0, step=1.0, unit="°"),
        Slider("el", "Steer: elevation", -45.0, 45.0, 0.0, step=1.0, unit="°"),
        Choice("fc", "Carrier", ["3.5 GHz", "28 GHz", "39 GHz", "140 GHz"], "28 GHz"),
        Slider("pel", "Power per element", -5.0, 30.0, 10.0, step=0.5, unit="dBm"),
        Slider("gel", "Element gain", 0.0, 8.0, 5.0, step=0.5, unit="dBi"),
        Heading("Link"),
        LogSlider("dist", "Distance", 10, 2000, 200, unit="m", fmt=".0f"),
        Slider("bw", "Channel bandwidth", 20.0, 2000.0, 400.0, step=10.0, unit="MHz"),
        Slider("nf", "Receiver noise figure", 3.0, 15.0, 10.0, step=0.5, unit="dB"),
        Slider("gue", "Handset array gain", 0.0, 20.0, 9.0, step=0.5, unit="dBi"),
    ]
    plots = [
        ImagePlot("uv", "Beam in direction cosines (dB)", x="u = sin θ cos φ", y="v = sin θ sin φ",
                  aspect=True),
        BarPlot("bud", "Link budget", y="dBm"),
    ]
    layout = [["uv", "bud"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("gain", "Panel gain", "dBi", ".1f"),
        Readout("eirp", "EIRP", "dBm", ".1f"),
        Readout("hpbw", "Beamwidth (azimuth)", "°", ".1f"),
        Readout("snr", "Line-of-sight SNR", "dB", ".1f", good=lambda x: x > 10),
    ]
    challenges = [
        Challenge("Reach an EIRP of 60 dBm with no more than 10 dBm per element.",
                  lambda s: s.r.eirp >= 60 and s.p.pel <= 10,
                  hint="EIRP grows as N²: N times the power and N times the gain."),
        Challenge("140 GHz, 400 MHz, 500 m: keep a line-of-sight SNR of 20 dB with no more "
                  "than 1024 elements and 10 dBm each.",
                  lambda s: (s.p.fc == "140 GHz" and s.p.dist >= 495 and s.p.bw >= 400
                             and s.p.nx * s.p.ny <= 1024 and s.p.pel <= 10 and s.r.snr >= 20)),
        Challenge("Steer the 8 × 8 panel 45° in azimuth and see the beam widen to about 18° "
                  "(within 1°).",
                  lambda s: (s.p.nx == 8 and abs(abs(s.p.az) - 45) < 0.5 and abs(s.p.el) < 0.5
                             and abs(s.r.hpbw - 18) <= 1)),
    ]

    def update(self, p):
        f = float(p.fc.split()[0]) * 1e9
        lam = C0 / f
        az, el = np.radians(p.az), np.radians(p.el)
        u0, v0 = np.sin(az) * np.cos(el), np.sin(el)
        u, img = mk.upa_af_uv(p.nx, p.ny, 0.5, u0, v0, n=161)
        pi = self.plot("uv")
        pi.image("img", np.nan_to_num(np.clip(img, -40, 0), nan=-40.0), x=(-1, 1), y=(-1, 1),
                 cmap="heat", levels=(-40, 0), colorbar=True, cbar_label="dB")
        t = np.linspace(0, 2 * np.pi, 181)
        pi.line("circ", np.cos(t), np.sin(t), color=GRAY, width=1.2)
        pi.scatter("aim", [u0], [v0], color=RED, size=10, symbol="+")
        pi.set_xlim(-1.05, 1.05)
        pi.set_ylim(-1.05, 1.05)
        N = p.nx * p.ny
        # gain: directive gain of the aperture, with cos(theta) scan loss (foreshortening)
        cos_t = max(np.cos(az) * np.cos(el), 1e-3)
        scan = float(db(cos_t))
        gain = float(db(N) + p.gel + scan)
        ptot = float(p.pel + db(N))
        eirp = ptot + gain
        fspl = float(20 * np.log10(4 * np.pi * p.dist / lam))
        prx = eirp - fspl + p.gue
        noise = -174 + float(db(p.bw * 1e6)) + p.nf
        snr = prx - noise
        items = [("conducted", ptot), ("panel gain", float(db(N) + p.gel))]
        if scan < -0.05:
            items.append(("scan loss", scan))
        items += [("free space", -fspl), ("handset", p.gue), ("− noise", -noise)]
        pb = self.plot("bud")
        tot = waterfall(pb, "w", items, "SNR (dB)")
        pb.set_ylim(min(-80, prx - 10, noise - 12), max(eirp, ptot) + 25)
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        pb.hline("noise", noise, color=PURPLE, style=":", label=f"noise floor {noise:.1f} dBm",
                 label_pos=0.02)
        hp = 101.5 / (p.nx * max(np.cos(az), 1e-3))
        self.readout(gain=gain, eirp=eirp, hpbw=hp, snr=snr)
        self.lam, self.fspl, self.prx = lam, fspl, prx

    def story(self, p):
        r = self.r
        N = p.nx * p.ny
        s = (f"<p>A {v(p.nx, 'd')} × {v(p.ny, 'd')} panel of λ/2-spaced patches at {v(p.fc)}: "
             f"λ = {v(self.lam * 1e3, '.1f', 'mm')}, so the panel is about "
             f"{v(p.nx * self.lam / 2 * 1e3, '.0f', 'mm')} wide. Its {v(N, 'd')} amplifiers deliver "
             f"{v(p.pel + 10 * np.log10(N), '.1f', 'dBm')} together, and the array focuses it with "
             f"{v(r.get('gain', 0), '.1f', 'dBi')} of gain: EIRP "
             f"{v(r.get('eirp', 0), '.1f', 'dBm')}. That is the <b>N² law</b>: N times the power "
             f"and N times the gain.</p>"
             f"<p>Over {v(p.dist, '.0f', 'm')} free space takes {v(self.fspl, '.1f', 'dB')}; the "
             f"handset's own array adds {v(p.gue, '.1f', 'dBi')}; in "
             f"{v(p.bw, '.0f', 'MHz')} the noise floor is {v(-174 + 10 * np.log10(p.bw * 1e6) + p.nf, '.1f', 'dBm')}. "
             f"Line-of-sight SNR: {v(r.get('snr', 0), '.1f', 'dB')}, the margin that has to survive "
             f"bodies, foliage and walls.</p>")
        if abs(p.az) > 20 or abs(p.el) > 20:
            s += ("<p>Steered away from broadside the aperture looks smaller (scan loss) and the "
                  "beam widens as 1/cos θ; left, the beam moves inside the unit circle of real "
                  "directions.</p>")
        return "<h3>A matchbox of antennas</h3>" + s + keybox(
            "The book's worked example: 8 × 8 at 28 GHz, 10 dBm and 5 dBi per element → EIRP "
            "51 dBm, about 31 dB SNR at 200 m in 400 MHz.")


# =============================================================================== 3. beam squint
class BeamSquint(Experiment):
    title = "Beam squint: phase vs true delay"
    blurb = "A phase shifter is right at one frequency only: across a wide band the beam wanders."
    book = "sec:ch19:arrays"
    controls = [
        LogSlider("N", "Elements", 8, 1024, 64, fmt=".0f"),
        Slider("steer", "Steering angle", 0.0, 70.0, 45.0, step=0.5, unit="°"),
        Choice("fc", "Carrier", ["28 GHz", "60 GHz", "140 GHz"], "28 GHz"),
        LogSlider("bw", "Signal bandwidth", 0.05, 10, 2.0, unit="GHz", fmt=".2f"),
        Choice("bf", "Beamformer", ["Phase shifters", "True time delay"]),
    ]
    plots = [
        Plot("beams", "Beams at the band edges and centre", x="angle (°)", y="array gain (dBi)",
             legend="tl"),
        Plot("loss", "Gain towards the target across the band", x="frequency offset (GHz)",
             y="gain change (dB)", ylim=(-25, 2), legend="bl"),
        ImagePlot("rain", "Beam direction vs frequency (dB)",
                  x="frequency offset (GHz)", y="angle (°)"),
    ]
    layout = [["beams", "rain"], ["loss", "rain"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("ta", "Aperture fill time", "ns", ".2f"),
        Readout("b3", "3 dB bandwidth ≈ 0.9/Ta", "GHz", ".2f"),
        Readout("edge", "Loss at the band edge", "dB", ".1f", good=lambda x: x > -3),
        Readout("sq", "Squint at the band edge", "°", ".2f"),
    ]
    challenges = [
        Challenge("28 GHz, 256 elements, 45°: find the widest bandwidth that loses no more than "
                  "3 dB at the edges (within 10 %).",
                  lambda s: (s.p.fc == "28 GHz" and abs(s.p.N - 256) < 6 and abs(s.p.steer - 45) < 0.3
                             and s.p.bf == "Phase shifters" and -3.0 <= s.r.edge <= -2.2)),
        Challenge("Sub-terahertz: 1024 elements at 140 GHz with 8 GHz of bandwidth steered 30° or "
                  "more, and no squint loss at all.",
                  lambda s: (s.p.fc == "140 GHz" and s.p.N >= 1000 and s.p.bw >= 7.9
                             and s.p.steer >= 30 and s.r.edge > -0.1),
                  hint="Delay each element by the right time, not the right phase."),
        Challenge("Keep phase shifters but lose under 1 dB over 2 GHz with 64 elements at 28 GHz, "
                  "steering as far as you can (at least 12°).",
                  lambda s: (s.p.bf == "Phase shifters" and abs(s.p.N - 64) < 2 and s.p.bw >= 1.99
                             and s.p.fc == "28 GHz" and s.p.steer >= 12 and s.r.edge > -1)),
    ]

    @staticmethod
    def gain(N, d_lam0, f_rel, th, th0, ttd):
        """Closed-form array gain (re one element): element phase progression
        psi = 2 pi d/lambda0 (f_rel sin th - (f_rel if ttd else 1) sin th0)."""
        psi = 2 * np.pi * d_lam0 * (f_rel * np.sin(th) - (f_rel if ttd else 1.0) * np.sin(th0))
        x = psi / 2
        num = np.sin(N * x)
        den = np.sin(x)
        small = np.abs(den) < 1e-9
        af = np.where(small, float(N), num / np.where(small, 1, den))
        return np.abs(af) ** 2 / N

    def update(self, p):
        N = int(round(p.N))
        f0 = float(p.fc.split()[0]) * 1e9
        th0 = np.radians(p.steer)
        ttd = p.bf == "True time delay"
        B = p.bw * 1e9
        span = max(4 * 101.5 / (N * max(np.cos(th0), 0.2)), 2.0)
        th = np.radians(np.linspace(max(-90, p.steer - span), min(90, p.steer + span), 801))
        pb = self.plot("beams")
        for key, df, col, nm in (("lo", -B / 2, RED, "lower edge"), ("mid", 0.0, NAVY, "centre"),
                                 ("hi", B / 2, GREEN, "upper edge")):
            g = self.gain(N, 0.5, 1 + df / f0, th, th0, ttd)
            pb.line(key, np.degrees(th), db(g), color=col, width=2.2 if key == "mid" else 1.6,
                    name=f"{(f0 + df) / 1e9:.2f} GHz")
        pb.vline("t", p.steer, color=GRAY, style=":", label="target", label_pos=0.95)
        pb.set_xlim(np.degrees(th[0]), np.degrees(th[-1]))
        pb.set_ylim(db(N) - 30, db(N) + 12)
        dfs = np.linspace(-B / 2, B / 2, 201)
        gl = db(self.gain(N, 0.5, 1 + dfs / f0, th0, th0, ttd) / N)
        pl = self.plot("loss")
        pl.line("ttd", dfs / 1e9, np.zeros_like(dfs), color=GREEN, style="--", width=1.6,
                name="true time delay")
        if not ttd:
            pl.line("ps", dfs / 1e9, gl, color=NAVY, width=2.4, name="phase shifters")
        pl.hline("m3", -3, color=RED, style=":", label="−3 dB", label_pos=0.02)
        pl.set_xlim(-B / 2e9, B / 2e9)
        # rainbow image: gain vs frequency and angle
        fr = np.linspace(-B / 2, B / 2, 61)
        ang = np.radians(np.linspace(max(-90, p.steer - span), min(90, p.steer + span), 161))
        G = self.gain(N, 0.5, 1 + fr[None, :] / f0, ang[:, None], th0, ttd)
        img = np.clip(db(G / N), -30, 0)
        pr = self.plot("rain")
        pr.image("img", img, x=(fr[0] / 1e9, fr[-1] / 1e9), y=(np.degrees(ang[0]), np.degrees(ang[-1])),
                 cmap="heat", levels=(-30, 0), colorbar=True, cbar_label="dB")
        pr.line("t", [fr[0] / 1e9, fr[-1] / 1e9], [p.steer, p.steer], color=GRAY, style="--",
                width=1.2)
        pr.set_xlim(fr[0] / 1e9, fr[-1] / 1e9)
        pr.set_ylim(np.degrees(ang[0]), np.degrees(ang[-1]))
        d = C0 / f0 / 2
        Ta = (N - 1) * d * np.sin(th0) / C0
        sq = 0.0 if ttd else float(np.degrees(np.arcsin(np.clip(np.sin(th0) / (1 + B / 2 / f0), -1, 1))) - p.steer)
        self.readout(ta=Ta * 1e9, b3=(0.9 / Ta / 1e9) if Ta > 0 else float("inf"),
                     edge=0.0 if ttd else float(min(gl[0], gl[-1])), sq=abs(sq))

    def story(self, p):
        r = self.r
        N = int(round(p.N))
        s = (f"<p>To steer {v(p.steer, '.0f', '°')} the wave must reach the far end of the "
             f"{v(N, 'd')}-element array {v(r.get('ta', 0), '.2f', 'ns')} after the near end: the "
             f"<b>aperture fill time</b>. ")
        if p.bf == "Phase shifters":
            s += (f"A phase shifter imitates that delay with a phase that is only correct at the "
                  f"carrier. At other frequencies the beam points at sin θ = (f₀/f)·sin θ₀: it "
                  f"<b>squints</b> by {v(r.get('sq', 0), '.2f', '°')} at the band edges, and the "
                  f"target sees {v(r.get('edge', 0), '.1f', 'dB')}. The right image shows the beam "
                  f"tilting across the band like a rainbow. Trouble starts when the bandwidth "
                  f"approaches 1/Ta (here about {v(r.get('b3', 0), '.2f', 'GHz')}).</p>")
        else:
            s += ("True time delay applies that delay itself, which is right at every frequency: "
                  "the beam stays put across the whole band (flat rainbow, zero loss). Digital "
                  "beamforming with a phase per OFDM subcarrier achieves the same.</p>")
        return "<h3>Phase is not delay</h3>" + s + keybox(
            "Squint matters when B ≳ 1/Ta = c/((N−1)d sin θ₀): big arrays, wide bands, large "
            "steering angles. Sub-THz systems will need true time delay.")


# =============================================================================== 4. DOA
class DirectionFinding(Experiment):
    title = "Direction finding: MUSIC & co"
    blurb = "Three signals arrive; can Bartlett, MVDR and MUSIC tell the close pair apart?"
    book = "sec:ch19:arrays"
    controls = [
        IntSlider("N", "Elements", 4, 32, 10),
        Slider("sep", "Separation of sources 2 and 3", 0.5, 20.0, 6.0, step=0.5, unit="°"),
        Slider("snr", "SNR per element", -10.0, 30.0, 5.0, step=1.0, unit="dB"),
        LogSlider("K", "Snapshots", 5, 2000, 200, fmt=".0f"),
        Toggle("coh", "Sources 2 and 3 coherent (multipath)", False),
        Toggle("smooth", "Spatial smoothing", False,
               help="Average the covariances of overlapping sub-arrays (costs aperture)"),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("spec", "Spatial spectra (normalised)", x="angle (°)", y="dB", xlim=(-60, 40),
             ylim=(-45, 3), legend="bl", legend_cols=3),
        Plot("eig", "Eigenvalues of the sample covariance", x="index", y="dB", legend=None),
    ]
    layout = [["spec"], ["eig"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bart", "Bartlett: close pair", "", None),
        Readout("mvdr", "MVDR: close pair", "", None),
        Readout("music", "MUSIC: close pair", "", None),
        Readout("bw", "Array beamwidth", "°", ".1f"),
    ]
    challenges = [
        Challenge("Resolve a pair 3° apart or closer with MUSIC using no more than 50 snapshots.",
                  lambda s: s.p.sep <= 3 and s.p.K <= 51 and s.r.music == "resolved" and not s.p.coh),
        Challenge("Find a setting where MVDR resolves all three but Bartlett does not.",
                  lambda s: s.r.mvdr == "resolved" and s.r.bart != "resolved"),
        Challenge("10 elements, pair 10° apart or closer and coherent: rescue MUSIC with spatial "
                  "smoothing.",
                  lambda s: (s.p.coh and s.p.smooth and s.p.N <= 10 and s.p.sep <= 10
                             and s.r.music == "resolved"),
                  hint="Without smoothing the coherent pair is one lump; smoothing may need a little SNR."),
    ]

    def setup(self):
        self.seed = 9

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        K = int(round(p.K))
        srcs = np.radians([-25.0, 8.0, 8.0 + p.sep])
        A = cl.ula_steering(p.N, srcs)
        S = mk.cn(rng, (3, K)) * np.sqrt(undb(p.snr))
        if p.coh:
            S[2] = S[1] * np.exp(1j * 0.7)
        X = A @ S + mk.cn(rng, (p.N, K))
        R = mk.sample_cov(X)
        if p.smooth:
            R = mk.smooth_cov(R, max(4, p.N - 2))
        th = np.radians(np.linspace(-90, 90, 3601))
        bart, mvdr, music, ev = mk.doa_spectra(R, th, 3)
        ps = self.plot("spec")
        out = {}
        for key, sp, col, nm in (("bart", bart, GRAY, "Bartlett"), ("mvdr", mvdr, GREEN, "MVDR (Capon)"),
                                 ("music", music, NAVY, "MUSIC")):
            sdb = db(sp / sp.max())
            ps.line(key, np.degrees(th), sdb, color=col, width=2.4 if key == "music" else 1.8, name=nm)
            pk = np.degrees(mk.find_peaks_db(th, sdb, 8, min_drop=3.0))
            n_pair = int(np.sum((pk > 8 - 0.5 * p.sep - 1) & (pk < 8 + 1.5 * p.sep + 1)))
            out[key] = "resolved" if n_pair >= 2 else "one lump"
        for i, a in enumerate(np.degrees(srcs)):
            ps.vline(f"s{i}", a, color=RED, style=":", width=1.2,
                     label="true" if i == 0 else None, label_pos=0.95)
        pe = self.plot("eig")
        k = np.arange(1, len(ev) + 1)
        pe.stems("ev", k, db(ev), color=NAVY, base=float(db(ev).min()) - 3)
        pe.hline("nf", 0, color=GRAY, style="--", label="noise level", label_pos=0.6)
        pe.set_xlim(0.3, len(ev) + 0.7)
        pe.set_ylim(float(db(ev).min()) - 4, float(db(ev).max()) + 4)
        pe.set_xticks([(i, str(i)) for i in k])
        self.readout(bart=out["bart"], mvdr=out["mvdr"], music=out["music"],
                     bw=101.5 / (p.N * (1 if not p.smooth else (p.N - 2) / p.N) * np.cos(np.radians(8))))
        self.nsig = int(np.sum(db(ev) > 3))

    def story(self, p):
        r = self.r
        s = (f"<p>Three sources at −25°, 8° and {v(8 + p.sep, '.1f', '°')}; the array's beamwidth "
             f"is about {v(r.get('bw', 0), '.0f', '°')}. The <b>Bartlett</b> scan (grey) is just a "
             f"beam swept across the sky: it cannot split sources closer than a beamwidth "
             f"(close pair: {v(r.get('bart', '?'))}). <b>MVDR</b> (green) keeps unit gain in each look "
             f"direction and minimises everything else, so it nulls the neighbour and sharpens "
             f"({v(r.get('mvdr', '?'))}). <b>MUSIC</b> (navy) splits the covariance eigenvectors "
             f"into a 3-dimensional signal subspace and a noise subspace, and looks for steering "
             f"vectors orthogonal to the noise: super-resolution ({v(r.get('music', '?'))}).</p>"
             f"<p>Bottom: three eigenvalues stand above the noise floor, one per source. ")
        if p.coh:
            s += ("Coherent sources (a reflection of the same signal) merge into one eigenvalue: "
                  "the covariance loses rank and MUSIC and MVDR fail. ")
            s += ("Spatial smoothing averages shifted sub-arrays, whose phases differ, and "
                  "restores the rank.</p>" if p.smooth else "Try spatial smoothing.</p>")
        else:
            s += "Fewer snapshots or less SNR blur the gap between them.</p>"
        return "<h3>Super-resolution from eigenvectors</h3>" + s + keybox(
            "Bartlett: resolution = beamwidth. MVDR and MUSIC: limited by SNR and snapshots, "
            "not by aperture alone.")


# =============================================================================== 5. SSB sweep and codebook
N1 = 8


class SSBSweep(Experiment):
    title = "SSB beam sweep and codebook"
    blurb = "The cell sweeps its beams, the phone reports the best one, then picks a precoder."
    book = "sec:ch19:hybrid"
    animate = True
    autoplay = True
    fps = 6
    controls = [
        Slider("ue", "Phone direction", -60.0, 60.0, 17.0, step=0.5, unit="°"),
        Choice("nssb", "SSB beams in the burst", ["4", "8", "16"], "8"),
        Choice("o1", "Codebook oversampling O1", ["1", "2", "4", "8"], "4"),
        Slider("scat", "Scattered power re line of sight", -30.0, 5.0, -10.0, step=1.0, unit="dB"),
        Button("again", "New scatterers"),
    ]
    plots = [
        PolarPlot("pol", "SSB beams (the one on air in red)", floor_db=-30, legend=None),
        BarPlot("rsrp", "RSRP reported per SSB", x="SSB index", y="dB re best"),
        Plot("cb", "Type I codebook: gain of every codeword", x="codeword (beam m, co-phase φ)",
             y="dB re eigen-beamformer", ylim=(-30, 2), legend="bl"),
    ]
    layout = [["pol", "rsrp"], ["pol", "cb"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("ssb", "Best SSB beam", "", None),
        Readout("pmi", "Best PMI (m, φ)", "", None),
        Readout("loss", "Codebook loss", "dB", ".2f", good=lambda x: x > -1),
        Readout("bits", "PMI feedback", "bits", "int"),
    ]
    challenges = [
        Challenge("Make the Type I codebook lose more than 3 dB against the eigen-beamformer.",
                  lambda s: s.r.loss < -3,
                  hint="A DFT beam fits one plane wave; give the channel many."),
        Challenge("With the scattering at −10 dB or weaker and the phone at 17°, choose the coarsest "
                  "oversampling that comes within 0.1 dB of the finest (O1 = 8).",
                  lambda s: (s.p.scat <= -10 and abs(s.p.ue - 17) < 0.3 and s.exp.coarsest_ok),
                  hint="More oversampling = more codewords = more feedback bits."),
        Challenge("Put the phone exactly between two of 8 SSB beams (best two RSRPs within 1 dB).",
                  lambda s: s.p.nssb == "8" and s.exp.top2 < 1.0),
    ]

    def setup(self):
        self.seed = 12
        self.k = 0
        self.rsrp_seen = None

    def on_again(self, p):
        self.seed += 1

    def channel(self, p):
        r = np.random.default_rng(self.seed)
        angs = np.r_[np.radians(p.ue), r.uniform(-np.pi / 3, np.pi / 3, 4)]
        gains = np.r_[1.0, np.full(4, np.sqrt(undb(p.scat) / 4))]
        pol = mk.cn(r, (5, 2)) * np.array([[1.0, 0.3]])
        A = cl.ula_steering(N1, angs)
        h = sum(g * np.r_[pp[0] * a, pp[1] * a] for g, pp, a in zip(gains, pol, A.T))
        return h

    def codebook_loss(self, h, O1):
        n = np.arange(N1)
        best, gains = None, []
        for m in range(N1 * O1):
            vv = np.exp(1j * 2 * np.pi * n * m / (N1 * O1))
            for phi in (1, 1j, -1, -1j):
                w = np.r_[vv, phi * vv] / np.sqrt(2 * N1)
                g = np.abs(np.vdot(w, h)) ** 2
                gains.append(g)
                if best is None or g > best[0]:
                    best = (g, m, phi)
        return np.array(gains), best

    def compute(self, p):
        nssb = int(p.nssb)
        self.ssb_ang = np.degrees(np.arcsin(np.linspace(-1, 1, nssb + 1)[:-1] + 1 / nssb))
        self.W = cl.ula_steering(N1, np.radians(self.ssb_ang)) / np.sqrt(N1)
        h = self.channel(p)
        self.rsrp = db(np.abs(self.W.conj().T @ h[:N1]) ** 2 + np.abs(self.W.conj().T @ h[N1:]) ** 2)
        O1 = int(p.o1)
        gains, best = self.codebook_loss(h, O1)
        opt = np.linalg.norm(h) ** 2
        self.cbg = db(gains / opt)
        self.best = best
        self.loss = float(db(best[0] / opt))
        ref8 = float(db(self.codebook_loss(h, 8)[1][0] / opt))
        ok = {o: float(db(self.codebook_loss(h, o)[1][0] / opt)) >= ref8 - 0.1 for o in (1, 2, 4, 8)}
        self.coarsest_ok = ok[O1] and not any(ok[o] for o in (1, 2, 4, 8) if o < O1)
        srt = np.sort(self.rsrp)[::-1]
        self.top2 = float(srt[0] - srt[1])

    def draw(self, p):
        nssb = len(self.ssb_ang)
        k = self.k % nssb
        th = np.radians(np.linspace(-90, 90, 721))
        pp = self.plot("pol")
        for i in range(16):
            if i < nssb:
                g = db(np.abs(self.W[:, i].conj() @ cl.ula_steering(N1, th)) ** 2)
                g = g - db(N1)
                pp.pattern(f"b{i}", th, g, color=RED if i == k else GRAY,
                           width=2.8 if i == k else 0.9)
        ue = np.radians(p.ue)
        pp.scatter("ue", [1.06 * np.sin(ue)], [1.06 * np.cos(ue)], color=NAVY, size=14, symbol="t")
        pp.text("uel", 1.06 * np.sin(ue), 1.06 * np.cos(ue), "  phone", color=NAVY, size=9,
                anchor=(0, 0.5))
        pr = self.plot("rsrp")
        rel = self.rsrp - self.rsrp.max()
        seen = np.arange(nssb) <= k if self.playing else np.ones(nssb, bool)
        best = int(np.argmax(self.rsrp))
        cols = [RED if (i == best and seen.all()) else (ORANGE if i == k else NAVY) for i in range(nssb)]
        hgt = np.where(seen, rel + 25, 0) - 25
        pr.bars("b", np.arange(nssb), hgt, base=-25, width=0.7, colors=cols)
        pr.set_ylim(-25, 4)
        pr.set_xlim(-0.6, nssb - 0.4)
        pr.set_xticks([(i, str(i)) for i in range(nssb)])
        if seen.all():
            pr.text("best", best, 0, "best", color=RED, anchor=(0.5, 1.05), bold=True, size=9)
        pc = self.plot("cb")
        x = np.arange(len(self.cbg))
        pc.scatter("all", x, self.cbg, color=NAVY, size=4, alpha=0.6, name="codewords")
        i_best = int(np.argmax(self.cbg))
        pc.scatter("best", [i_best], [self.cbg[i_best]], color=RED, size=12, symbol="d",
                   name="reported PMI")
        pc.hline("opt", 0, color=GREEN, style="--", label="eigen-beamformer (perfect CSI)",
                 label_pos=0.45)
        pc.set_xlim(-1, len(x))
        _, m, phi = self.best
        phis = {1: "1", 1j: "j", -1: "−1", -1j: "−j"}[phi]
        self.readout(ssb=f"{best} ({self.ssb_ang[best]:.0f}°)", pmi=f"m = {m}, φ = {phis}",
                     loss=self.loss, bits=int(np.log2(4 * N1 * int(p.o1))))

    def update(self, p):
        self.compute(p)
        self.draw(p)

    def tick(self, p):
        self.k += 1
        self.draw(p)

    def story(self, p):
        r = self.r
        s = (f"<p>A millimetre-wave cell cannot shout in all directions, so it transmits its "
             f"synchronisation blocks (SSBs) in a burst of {v(int(p.nssb), 'd')} beams, one after "
             f"another (left, red = the beam on air now). The phone measures the reference "
             f"signal received power of each (right) and reports the best: beam "
             f"{v(r.get('ssb', '—'))}. That is the coarse pointing.</p>"
             f"<p>For data the phone picks a precoder from NR's <b>Type I codebook</b>: oversampled "
             f"DFT beams for each polarisation plus a co-phasing term. Best codeword "
             f"{v(r.get('pmi', '—'))}, {v(r.get('bits', 0), 'd')} bits of feedback, within "
             f"{v(-r.get('loss', 0), '.2f', 'dB')} of a perfect eigen-beamformer. ")
        if p.scat > -5:
            s += ("With strong scattering no single beam fits; that is why NR added Type II "
                  "codebooks that combine several beams.</p>")
        else:
            s += ("One dominant path is a plane wave, and a DFT beam <i>is</i> a plane wave: the "
                  "codebook is nearly optimal.</p>")
        return "<h3>Sweep, report, refine</h3>" + s + keybox(
            "Beam management in 5G NR: SSB sweep (P1), CSI-RS refinement (P2/P3), then PMI from "
            "a codebook.")


# =============================================================================== 6. hybrid precoding
class HybridOMP(Experiment):
    title = "Hybrid precoding by OMP"
    blurb = "64 antennas, a handful of RF chains: how close does analog + digital get to digital?"
    book = "sec:ch19:hybrid"
    heavy = True
    controls = [
        IntSlider("nrf", "RF chains", 1, 8, 4),
        IntSlider("ns", "Data streams Ns", 1, 3, 2),
        IntSlider("ncl", "Scattering clusters", 1, 10, 5),
        IntSlider("nray", "Rays per cluster", 1, 10, 8),
        Choice("effort", "Channel draws", ["30", "100"]),
    ]
    plots = [
        Plot("se", "Spectral efficiency, 64 × 16 arrays", x="SNR before array gain (dB)",
             y="b/s/Hz", xlim=(-30, 0), legend="tl"),
        PolarPlot("beams", "Rays (dots) and OMP's analog beams", floor_db=-25,
                  legend=None),
    ]
    layout = [["se", "beams"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("ratio", "Hybrid ÷ fully digital at 0 dB", "%", ".0f", good=lambda x: x >= 90),
        Readout("hyb", "Hybrid at 0 dB", "b/s/Hz", ".1f"),
        Readout("ana", "Analog only at 0 dB", "b/s/Hz", ".1f"),
        Readout("done", "Channels averaged", "", None),
    ]
    challenges = [
        Challenge("Two streams at 90 % or more of fully digital with only 2 RF chains.",
                  lambda s: s.p.ns == 2 and s.p.nrf == 2 and s.r.ratio >= 90 and s.exp.finished,
                  hint="OMP needs the channel to be sparse: few clusters, few rays."),
        Challenge("Starve the digital stage: two streams, two RF chains, a rich channel, and "
                  "hybrid below 88 % of fully digital.",
                  lambda s: s.p.ns == 2 and s.p.nrf == 2 and s.r.ratio < 88 and s.exp.finished),
        Challenge("Find a channel where analog beam steering alone is as good as fully digital "
                  "(one stream, within 0.05 b/s/Hz).",
                  lambda s: s.p.ns == 1 and s.exp.finished and s.exp.gap_ana < 0.05,
                  hint="When is one steering vector exactly the best precoder?"),
    ]
    snrs = np.arange(-30, 0.1, 5.0)
    Nt, Nr = 64, 16

    def update(self, p):
        self.acc = dict(n=0, opt=np.zeros(len(self.snrs)), hyb=np.zeros(len(self.snrs)),
                        ana=np.zeros(len(self.snrs)))
        self.finished = False
        self.gap_ana = 99.0
        rng = np.random.default_rng(3)
        H, At, ang = mk.sparse_channel(self.Nt, self.Nr, p.ncl, p.nray, rng)
        V = np.linalg.svd(H)[2].conj().T[:, :p.ns]
        _, _, idx = mk.omp_precoder(V, At, max(p.nrf, p.ns))
        th = np.linspace(-np.pi / 2, np.pi / 2, 721)
        a = np.exp(-1j * np.pi * np.arange(self.Nt)[:, None] * np.sin(th)[None, :]) / np.sqrt(self.Nt)
        pb = self.plot("beams")
        cols = [RED, ORANGE, GREEN, PURPLE, BLUE, GOLD, TEAL, NAVY]
        for i in range(8):
            if i < len(idx):
                g = np.abs(At[:, idx[i]].conj() @ a) ** 2
                pb.pattern(f"b{i}", th, db(g / g.max()), color=cols[i], width=1.8)
        r_ = 1.04
        pb.scatter("rays", r_ * np.sin(ang), r_ * np.cos(ang), color=NAVY, size=6)
        self.readout(ratio=None, hyb=None, ana=None, done="0")
        pse = self.plot("se")
        pse.hline("z", 0, color=GRAY, style="-", width=0.5)

    def background(self, p):
        T = 6 if self.quick else int(p.effort)
        rng = np.random.default_rng(31)
        g = undb(self.snrs)
        for t in range(T):
            H, At, _ = mk.sparse_channel(self.Nt, self.Nr, p.ncl, p.nray, rng)
            V = np.linalg.svd(H)[2].conj().T[:, :p.ns]
            Frf, Fbb, _ = mk.omp_precoder(V, At, max(p.nrf, p.ns))
            Fh = Frf @ Fbb
            k = np.argsort(np.sum(np.abs(H @ At) ** 2, axis=0))[::-1][:p.ns]
            Fa = At[:, k] / np.linalg.norm(At[:, k]) * np.sqrt(p.ns)
            res = np.array([[mk.spectral_eff(H, F, s_, p.ns) for s_ in g] for F in (V, Fh, Fa)])
            yield dict(res=res, last=(t == T - 1))

    def progress(self, p, it):
        a = self.acc
        a["n"] += 1
        a["opt"] += it["res"][0]
        a["hyb"] += it["res"][1]
        a["ana"] += it["res"][2]
        n = a["n"]
        pse = self.plot("se")
        pse.line("opt", self.snrs, a["opt"] / n, color=NAVY, width=2.4, name="fully digital (SVD, 64 chains)")
        pse.line("hyb", self.snrs, a["hyb"] / n, color=RED, width=1.6,
                 name=f"hybrid OMP ({max(p.nrf, p.ns)} chains)")
        pse.scatter("hybp", self.snrs, a["hyb"] / n, color=RED, size=8)
        pse.line("ana", self.snrs, a["ana"] / n, color=ORANGE, style="--", width=1.8,
                 name="analog beam steering only")
        pse.set_ylim(0, 1.15 * max(a["opt"][-1] / n, 1))
        self.finished = it["last"]
        self.gap_ana = float((a["opt"][-1] - a["ana"][-1]) / n)
        self.readout(ratio=100 * a["hyb"][-1] / a["opt"][-1], hyb=a["hyb"][-1] / n,
                     ana=a["ana"][-1] / n, done=f"{n}" + (" ✓" if self.finished else " …"))

    def story(self, p):
        r = self.r
        nrf = max(p.nrf, p.ns)
        s = (f"<p>A fully digital 64-element array needs 64 RF chains (converters, mixers, "
             f"power). A <b>hybrid</b> array uses {v(nrf, 'd')}: each chain feeds all 64 phase "
             f"shifters (the analog beams F<sub>RF</sub>, right), and a small digital precoder "
             f"F<sub>BB</sub> mixes the chains.</p>"
             f"<p>Millimetre-wave channels have a few clusters of rays (dots), so the ideal "
             f"precoder is close to a combination of a few steering vectors. <b>Orthogonal "
             f"matching pursuit</b> (El Ayach et al., 2014) picks them greedily, then fits "
             f"F<sub>BB</sub> by least squares. Here hybrid reaches "
             f"{v(r.get('ratio') or 0, '.0f', '%')} of fully digital at 0 dB.</p>")
        if p.ncl * p.nray > 40:
            s += ("<p>Many clusters and rays make the channel rich, not sparse: a few beams can no "
                  "longer capture it.</p>")
        return "<h3>Few chains, many antennas</h3>" + s + keybox(
            "F = F<sub>RF</sub>F<sub>BB</sub>: analog for gain, digital for streams. NR FR2 base "
            "stations and phones are built this way.")


# =============================================================================== 7. LOS MIMO
class LOSMIMO(Experiment):
    title = "Line-of-sight MIMO spacing"
    blurb = "Without scattering, two streams need antennas far enough apart to see different paths."
    book = "sec:ch19:beyond"
    controls = [
        LogSlider("f", "Carrier", 6, 150, 80, unit="GHz", fmt=".1f"),
        LogSlider("R0", "Design range", 100, 5000, 1000, unit="m", fmt=".0f"),
        Choice("N", "Antennas per end", ["2", "4"]),
        Slider("k", "Spacing re optimal", 0.25, 2.0, 1.0, step=0.01, unit="×"),
        Slider("snr", "SNR", 5.0, 40.0, 25.0, step=0.5, unit="dB"),
    ]
    plots = [
        Plot("eig", "Eigen-channel gains versus range", x="range (m)", y="gain re orthogonal (dB)",
             logx=True, ylim=(-30, 6), legend="bl"),
        Plot("cap", "Capacity versus range", x="range (m)", y="b/s/Hz", logx=True, legend="bl"),
        Plot("geo", "The link (vertical scale exaggerated)", x="range (m)", y="height (m)",
             legend=None),
    ]
    layout = [["eig", "cap"], ["geo", "geo"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("d", "Antenna spacing", "m", ".2f"),
        Readout("len", "Array length", "m", ".2f"),
        Readout("cap", "Capacity at design range", "b/s/Hz", ".1f"),
        Readout("cond", "Eigen-channel spread", "dB", ".1f", good=lambda x: x > -3),
    ]
    challenges = [
        Challenge("Design a 2 × 2 hop at 38 GHz over 2 km and read the spacing (about 2.8 m).",
                  lambda s: (abs(s.p.f - 38) < 0.6 and abs(s.p.R0 - 2000) < 40 and s.p.N == "2"
                             and abs(s.p.k - 1) < 0.01)),
        Challenge("Squeeze the antennas to 80 % of the optimal spacing or less and keep the "
                  "weakest eigen-channel within 6 dB of the strongest.",
                  lambda s: s.p.k <= 0.8 and s.r.cond >= -6,
                  hint="Tower space is expensive: how much can you save?"),
        Challenge("4 × 4 at 80 GHz over 1 km: reach 30 b/s/Hz at the design range with the lowest "
                  "SNR you can (≤ 25 dB).",
                  lambda s: (s.p.N == "4" and abs(s.p.f - 80) < 1.5 and abs(s.p.R0 - 1000) < 20
                             and s.r.cap >= 30 and s.p.snr <= 25)),
    ]

    def update(self, p):
        N = int(p.N)
        lam = C0 / (p.f * 1e9)
        d = p.k * mk.los_opt_spacing(N, p.R0, lam)
        Rs = np.logspace(np.log10(p.R0) - 1.3, np.log10(p.R0) + 1.0, 220)
        sv = np.array([np.linalg.svd(mk.los_channel(N, d, R, lam), compute_uv=False) ** 2 / N
                       for R in Rs])
        pe = self.plot("eig")
        cols = [NAVY, RED, GREEN, ORANGE]
        for i in range(4):
            if i < N:
                pe.line(f"s{i}", Rs, db(sv[:, i] + 1e-6), color=cols[i], width=2.0,
                        name=f"σ{i + 1}²/N")
        pe.vline("R0", p.R0, color=GRAY, style=":", label="design", label_pos=0.95)
        pe.set_xlim(Rs[0], Rs[-1])
        snr = undb(p.snr)
        C = np.sum(np.log2(1 + snr / N * sv * N), axis=1)
        pc = self.plot("cap")
        pc.line("c", Rs, C, color=NAVY, width=2.4, name=f"LOS {N} × {N}")
        pc.hline("ideal", N * np.log2(1 + snr), color=NAVY, style="--", width=1.0,
                 label=f"{N} × SISO", label_pos=0.55)
        pc.hline("siso", np.log2(1 + snr), color=GRAY, style=":", label="SISO", label_pos=0.02)
        pc.vline("R0", p.R0, color=GRAY, style=":")
        pc.set_xlim(Rs[0], Rs[-1])
        pc.set_ylim(0, N * np.log2(1 + snr) * 1.15)
        sv0 = np.linalg.svd(mk.los_channel(N, d, p.R0, lam), compute_uv=False) ** 2 / N
        c0 = float(np.sum(np.log2(1 + snr / N * sv0 * N)))
        # geometry sketch
        pg = self.plot("geo")
        y = (np.arange(N) - (N - 1) / 2) * d + 30
        for i in range(N):
            for j in range(N):
                pg.line(f"l{i}{j}", [0, p.R0], [y[i], y[j]], color=GRAY if i != j else BLUE,
                        width=0.8 if i != j else 1.4)
        pg.line("t1", [0, 0], [0, y.max() + 2], color=NAVY, width=4)
        pg.line("t2", [p.R0, p.R0], [0, y.max() + 2], color=NAVY, width=4)
        pg.scatter("a1", np.zeros(N), y, color=RED, size=10, symbol="s")
        pg.scatter("a2", np.full(N, p.R0), y, color=RED, size=10, symbol="s")
        pg.text("dl", 0.02 * p.R0, y.mean(), f" d = {d:.2f} m", color=RED, anchor=(0, 0.5), bold=True)
        pg.set_xlim(-0.05 * p.R0, 1.05 * p.R0)
        pg.set_ylim(30 - 0.9 * N * d - 0.5, 30 + 0.9 * N * d + 0.5)
        self.readout(d=d, len=(N - 1) * d, cap=c0, cond=float(db(sv0.min() / sv0.max())) + 0.0)

    def story(self, p):
        r = self.r
        N = int(p.N)
        lam = C0 / (p.f * 1e9)
        s = (f"<p>Two dishes facing each other with nothing in between: a plane-wave channel is "
             f"rank one, one stream. But over {v(p.R0, '.0f', 'm')} the wavefronts are spherical, "
             f"and if the antennas are far enough apart the crossed paths (grey) are longer than the "
             f"straight ones (blue) by half a wavelength ({v(lam * 1e3, '.1f', 'mm')}/2). Then the "
             f"channel matrix is a scaled unitary matrix: {v(N, 'd')} equally strong eigen-channels "
             f"and {v(r.get('cap', 0), '.1f', 'b/s/Hz')}.</p>"
             f"<p>The condition is d<sub>t</sub>d<sub>r</sub> = λR/N, here d = "
             f"{v(r.get('d', 0), '.2f', 'm')}. Beyond the design range the second eigenvalue sinks "
             f"(left): the arrays become small compared with the Fresnel zone.</p>")
        return "<h3>Spherical waves make room for two</h3>" + s + keybox(
            "Commercial E-band LOS-MIMO radios use this with two polarisations: four streams in one "
            "channel, if the towers don't sway too much.")


# =============================================================================== the lab
LAB = st.Lab(29, "Antenna Arrays and Beamforming", chapter=19,
             chapter_title="MIMO and Antenna Arrays",
             experiments=[SteerArray, PlanarPanel, BeamSquint, DirectionFinding, SSBSweep,
                          HybridOMP, LOSMIMO])

if __name__ == "__main__":
    st.run(LAB)
