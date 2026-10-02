"""Lab 30 · Wireline and Optical: Loops, DSL, Vectoring, Dispersion, Coherent DSP and PON   (Chapter 24)

Run it:      python labs/lab30_wireline_optical.py
Self-test:   python labs/lab30_wireline_optical.py --selftest

Wires and fibres are the channels wireless engineers envy: static, private and enormous. Follow a
bit from the telephone exchange to the transoceanic cable: the loss and impedance of a twisted
pair, loading coils and bridged taps, DMT bit loading and rate against reach, a 16-line binder
cancelled by vectoring, dispersion that kills intensity modulation but not a coherent receiver, a
live coherent DSP chain that untangles two polarisations, the optimum launch power of an amplified
link, and the power budget of a passive optical network. Built on commlib.wireline, the module
behind Chapter 24's figures.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

import commlib as cl
from commlib import wireline as wl
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, EyePlot, ConstellationPlot, BarPlot, ImagePlot, Readout,
                    Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

FT = 0.3048


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def waterfall(plot, key, items, total_name, total_color=NAVY, label_size=8.5):
    """A decibel 'bank statement' (as in Lab 18): floating bars from the running balance."""
    run = 0.0
    names = []
    for i, (nm, val) in enumerate(items):
        lo, hi = sorted([run, run + val])
        plot.bars(f"{key}{i}", [i], [hi], base=lo, width=0.62, color=GREEN if val >= 0 else RED)
        plot.text(f"{key}t{i}", i, hi, f"{val:+.1f}", anchor=(0.5, 1.05), size=label_size, bold=True)
        if i < len(items) - 1:
            plot.line(f"{key}c{i}", [i + 0.31, i + 0.69], [run + val] * 2, color=GRAY, width=1.0)
        run += val
        names.append(nm)
    n = len(items)
    lo, hi = sorted([0.0, run])
    plot.bars(f"{key}T", [n], [hi], base=lo, width=0.62, color=total_color)
    plot.text(f"{key}tT", n, hi, f"{run:.1f}", anchor=(0.5, 1.05), size=label_size + 0.5, bold=True,
              color=total_color)
    plot.set_xticks([(i, s) for i, s in enumerate(names + [total_name])])
    plot.set_xlim(-0.6, n + 0.6)
    return run


# =============================================================================== 1. line constants
class LineConstants(Experiment):
    title = "The twisted pair as a line"
    blurb = "Skin effect, characteristic impedance and loss: why copper is a low-pass filter."
    book = "sec:ch24:tline"
    controls = [
        Choice("awg", "Wire gauge", ["26 AWG (0.4 mm)", "24 AWG (0.5 mm)"], "26 AWG (0.4 mm)",
               style="menu"),
        Slider("L", "Loop length", 0.1, 6.0, 3.0, step=0.05, unit="km"),
        LogSlider("fk", "Look at frequency", 1, 30000, 1000, unit="kHz", fmt=",.0f"),
    ]
    plots = [
        Plot("att", "Attenuation per km", x="frequency (Hz)", y="dB/km", logx=True, logy=True,
             xlim=(1e3, 3e7), ylim=(0.5, 200), legend="tl"),
        Plot("z0", "Characteristic impedance |Z₀|", x="frequency (Hz)", y="|Z₀| (Ω)", logx=True,
             logy=True, xlim=(1e3, 3e7), ylim=(80, 2000), legend="tr"),
        Plot("il", "Insertion loss of your loop (100 Ω ends)", x="frequency (Hz)", y="loss (dB)",
             logx=True, xlim=(1e3, 3e7), ylim=(0, 160), legend=None),
    ]
    layout = [["att", "z0"], ["il", "il"]]
    readouts = [
        Readout("R", "Resistance at f", "Ω/km", ",.0f"),
        Readout("Z0", "|Z₀| at f", "Ω", ".0f"),
        Readout("IL", "Loop loss at f", "dB", ".1f"),
        Readout("dly", "Loop delay", "µs", ".1f"),
    ]
    challenges = [
        Challenge("Reproduce the book: 3 km of 26 AWG loses about 76 dB at 1 MHz (±1 dB) — then "
                  "find the length of 24 AWG that loses 60 dB at 2 MHz (±1 dB).",
                  lambda s: (s.p.awg.startswith("24") and abs(s.p.fk - 2000) < 100
                             and abs(s.r.IL - 60) <= 1)),
        Challenge("Find the lowest frequency (within 25 %) at which 26 AWG's |Z₀| is within 5 % "
                  "of 100 Ω.",
                  lambda s: (s.p.awg.startswith("26") and s.r.Z0 <= 105
                             and s.p.fk * 1e3 <= 1.25 * s.exp.f105),
                  hint="Below that, R dominates ωL and Z₀ ≈ √(R/jωC) is large and capacitive."),
        Challenge("Make a voice-band-only loop: lose more than 60 dB at 1 MHz but less than 10 dB "
                  "at 3 kHz.",
                  lambda s: s.exp.il_1m > 60 and s.exp.il_3k < 10),
    ]

    def setup(self):
        f = np.logspace(3, 7.5, 2000)
        z = np.abs(wl.propagation(f, 26)[1])
        self.f105 = float(f[np.argmax(z <= 105)])

    def update(self, p):
        p = st.Params(p, f=p.fk * 1e3)
        awg = 26 if p.awg.startswith("26") else 24
        other = 24 if awg == 26 else 26
        f = np.logspace(3, np.log10(3e7), 300)
        pa = self.plot("att")
        pa.line("o", f, wl.attenuation_db_per_km(f, other), color=GRAY, width=1.4,
                name=f"{other} AWG")
        pa.line("a", f, wl.attenuation_db_per_km(f, awg), color=NAVY, width=2.4, name=f"{awg} AWG")
        pa.line("sq", f, wl.attenuation_db_per_km(1e6, awg)[()] * np.sqrt(f / 1e6), color=ORANGE,
                width=1.0, style=":", name="∝ √f")
        pa.vline("f", p.f, color=RED, style="--")
        _, Z0 = wl.propagation(f, awg)
        pz = self.plot("z0")
        pz.line("z", f, np.abs(Z0), color=NAVY, width=2.4, name=f"{awg} AWG")
        _, Z0o = wl.propagation(f, other)
        pz.line("zo", f, np.abs(Z0o), color=GRAY, width=1.4, name=f"{other} AWG")
        pz.hline("100", 100, color=GREEN, style=":", label="100 Ω", label_pos=0.02)
        pz.vline("f", p.f, color=RED, style="--")
        L = p.L * 1e3
        il = -20 * np.log10(np.abs(wl.loop_gain(f, L, awg)))
        pi_ = self.plot("il")
        pi_.line("il", f, il, color=NAVY, width=2.4, fill=0, fill_alpha=0.08)
        for fb, lab in ((3.4e3, "voice"), (1.1e6, "ADSL"), (2.2e6, "ADSL2+"), (17.7e6, "VDSL2 17a")):
            pi_.vline(f"b{lab}", fb, color=GRAY, style=":", label=lab, label_pos=0.92)
        ilf = float(-20 * np.log10(np.abs(wl.loop_gain(np.array([p.f]), L, awg)))[0])
        pi_.scatter("now", [p.f], [ilf], color=RED, size=13, symbol="d")
        pi_.set_ylim(0, max(40.0, min(250.0, float(il.max()) * 1.1)))
        R, Lh, G, Cc = wl.rlgc_twisted_pair(np.array([p.f]), awg)
        gam, Z0f = wl.propagation(np.array([p.f]), awg)
        vph = 2 * np.pi * p.f / gam.imag[0]
        self.il_1m = float(-20 * np.log10(np.abs(wl.loop_gain(np.array([1e6]), L, awg)))[0])
        self.il_3k = float(-20 * np.log10(np.abs(wl.loop_gain(np.array([3e3]), L, awg)))[0])
        self.vph = vph
        self.readout(R=R[0] * 1e3, Z0=abs(Z0f[0]), IL=ilf, dly=L / vph * 1e6)

    def story(self, p):
        r = self.r
        s = (f"<p>A twisted pair is a transmission line with resistance R, inductance L, "
             f"conductance G and capacitance C per metre. At {v(p.fk, ',.0f', 'kHz')} the resistance "
             f"is {v(r.get('R', 0), ',.0f', 'Ω/km')}: the current crowds into the skin of the wire, "
             f"so R and the loss in dB grow as √f (dotted line, top left).</p>"
             f"<p>The characteristic impedance falls from kilohms in the voice band towards about "
             f"100 Ω above a few hundred kilohertz, where ωL dominates R: that is why DSL modems "
             f"terminate in 100 Ω. Your {v(p.L, '.2f', 'km')} loop loses "
             f"{v(r.get('IL', 0), '.1f', 'dB')} here and delays the signal by "
             f"{v(r.get('dly', 0), '.1f', 'µs')} (velocity {v(self.vph / 3e8, '.2f')}c).</p>")
        return "<h3>A low-pass filter made of copper</h3>" + s + keybox(
            "α ≈ R/(2Z₀) at high frequency: loss in dB grows as √f and in proportion to length. "
            "Every DSL generation is a fight with this curve.")


# =============================================================================== 2. impairments
COILS = {"H88 (88 mH every 1.83 km)": (88e-3, 6000 * FT), "D66 (66 mH every 1.37 km)": (66e-3, 4500 * FT)}


class LoopImpairments(Experiment):
    title = "Loading coils and bridged taps"
    blurb = "Two old tricks of the telephone plant that DSL had to undo."
    book = "sec:ch24:tline"
    controls = [
        Heading("Voice band: a 5.5 km 26 AWG loop"),
        Toggle("coils", "Loading coils", True),
        Choice("ctype", "Loading", list(COILS), "H88 (88 mH every 1.83 km)", style="menu"),
        Heading("DSL band: a 300 m 24 AWG drop"),
        Slider("tap", "Bridged tap length", 0, 300, 50, step=1, unit="m",
               help="An open-circuited branch left over from an old connection (0 = none)"),
        Slider("pos", "Tap position along the drop", 10, 290, 200, step=5, unit="m"),
    ]
    plots = [
        Plot("voice", "Voice band: insertion gain (600 Ω ends)", x="frequency (kHz)", y="gain (dB)",
             xlim=(0, 8), ylim=(-30, 2), legend="bl"),
        Plot("dsl", "DSL band: insertion gain of the drop (100 Ω ends)", x="frequency (MHz)",
             y="gain (dB)", xlim=(0, 30), ylim=(-45, 2), legend="bl"),
    ]
    layout = [["voice"], ["dsl"]]
    readouts = [
        Readout("v1k", "Loss at 1 kHz", "dB", ".1f"),
        Readout("cut", "Loaded cut-off (−3 dB re 1 kHz)", "kHz", None),
        Readout("notch", "First tap notch", "MHz", None),
        Readout("depth", "Worst notch depth", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Push the loaded loop's cut-off above 3.8 kHz.",
                  lambda s: s.p.coils and s.exp.cut_num > 3.8,
                  hint="Lighter inductance, closer spacing: the cut-off is about 1/(π√(L·C·d))."),
        Challenge("Put the first bridged-tap notch inside VDSL2's 5.2–8.5 MHz downstream band.",
                  lambda s: 5.2 <= s.exp.notch_num <= 8.5),
        Challenge("Tame the tap: keep its worst notch shallower than 9.5 dB without removing it.",
                  lambda s: s.p.tap >= 5 and s.r.depth < 9.5,
                  hint="An echo that travels far down a lossy tap comes back weak."),
    ]

    def update(self, p):
        fv = np.linspace(100, 8000, 400)
        Ltot = 18000 * FT
        unl = wl.line_abcd(fv, Ltot, 26)
        g_unl = 20 * np.log10(np.abs(wl.insertion_gain(unl, 600, 600)))
        pv = self.plot("voice")
        pv.band("vb", 0.3, 3.4, color=GREEN, alpha=0.08)
        pv.line("u", fv / 1e3, g_unl, color=GRAY, width=1.8, name="unloaded")
        Lc, dsp = COILS[p.ctype]
        if p.coils:
            coil = wl.series_abcd(8.0 + 1j * 2 * np.pi * fv * Lc)
            secs = [wl.line_abcd(fv, dsp / 2, 26)]
            run = dsp / 2
            while run + dsp < Ltot:
                secs += [coil, wl.line_abcd(fv, dsp, 26)]
                run += dsp
            secs += [coil, wl.line_abcd(fv, Ltot - run, 26)]
            g_l = 20 * np.log10(np.abs(wl.insertion_gain(wl.cascade(*secs), 600, 600)))
            pv.line("l", fv / 1e3, g_l, color=NAVY, width=2.4, name="loaded")
            ref = np.interp(1000, fv, g_l)
            below = np.where((g_l < ref - 3) & (fv > 1000))[0]
            self.cut_num = fv[below[0]] / 1e3 if len(below) else 8.0
            pv.vline("cut", self.cut_num, color=RED, style="--", label="cut-off", label_pos=0.9)
            v1k = -ref
            cut = f"{self.cut_num:.2f}"
        else:
            self.cut_num = 0.0
            v1k = -float(np.interp(1000, fv, g_unl))
            cut = "—"
        # bridged tap
        f = np.linspace(20e3, 30e6, 1500)
        base = wl.line_abcd(f, 300, 24)
        g0 = 20 * np.log10(np.abs(wl.insertion_gain(base)))
        pd = self.plot("dsl")
        for a, b in ((0.138, 3.75), (5.2, 8.5), (12, 17.664)):
            pd.band(f"ds{a}", a, b, color=GREEN, alpha=0.06)
        pd.line("base", f / 1e6, g0, color=GRAY, width=1.8, name="300 m, no tap")
        if p.tap > 0:
            pos = min(p.pos, 299.0)
            M = wl.cascade(wl.line_abcd(f, pos, 24), wl.bridged_tap_abcd(f, p.tap, 24),
                           wl.line_abcd(f, 300 - pos, 24))
            gt = 20 * np.log10(np.abs(wl.insertion_gain(M)))
            pd.line("tap", f / 1e6, gt, color=NAVY, width=1.8, name=f"+ {p.tap:.0f} m bridged tap")
            gam, _ = wl.propagation(np.array([5e6]), 24)
            vph = 2 * np.pi * 5e6 / gam.imag[0]
            self.notch_num = vph / (4 * p.tap) / 1e6
            if self.notch_num <= 30:
                pd.vline("n", self.notch_num, color=RED, style="--", label="λ/4 notch",
                         label_pos=0.08)
            depth = float(np.max(g0 - gt))
            notch = f"{self.notch_num:.2f}"
        else:
            self.notch_num = np.inf
            depth = 0.0
            notch = "—"
        self.readout(v1k=v1k, cut=cut, notch=notch, depth=depth)

    def story(self, p):
        s = ("<p><b>Loading coils</b> (top): a long loop's capacitance makes the voice band droop. "
             "Adding series inductance every mile or so makes the line look like a low-loss "
             "lumped low-pass filter: flat to the cut-off, then a cliff. Perfect for 1950s voice, "
             "fatal for DSL, so coils must be removed before a loop can carry broadband.</p>")
        if p.tap > 0:
            s += (f"<p>A <b>bridged tap</b> (bottom) is an open branch of {v(p.tap, '.0f', 'm')}. "
                  f"A wave runs down it, reflects off the open end and returns in antiphase "
                  f"whenever the tap is an odd number of quarter wavelengths long: notches at "
                  f"(2k+1)v/(4ℓ), the first at {v(self.notch_num, '.2f', 'MHz')}. DMT simply loads "
                  f"fewer bits on the notched tones.</p>")
        else:
            s += "<p>No bridged tap: the drop's response is smooth.</p>"
        return "<h3>Fixes of one era, faults of the next</h3>" + s + keybox(
            "A loaded loop cannot carry DSL; a bridged tap carves λ/4 notches. Copper qualification "
            "tests look for both.")


# =============================================================================== 3. DSL rate vs reach
NOISE, GAP = -140.0, 12.0                              # Chapter 24's constants
VDSL_DS = [(138e3, 3.75e6), (5.2e6, 8.5e6), (12e6, 17.664e6)]
AM_RFI = [(0.54e6, 1.7e6)]                              # medium-wave broadcast band


def rfi_noise(f, on):
    """Ingress from AM broadcast stations: narrow peaks 30-40 dB above the floor."""
    if not on:
        return np.full_like(f, -300.0)
    out = np.full_like(f, -300.0)
    for fc, lvl in ((0.65e6, -100), (0.81e6, -105), (0.99e6, -98), (1.21e6, -104), (1.45e6, -102),
                    (6.1e6, -108), (7.3e6, -104), (9.6e6, -106), (11.8e6, -110)):
        out = wl.db_sum(out, lvl - 0.5 * ((f - fc) / 5e3) ** 2)
    return out


def dsl_rate(kind, L, nfext, noise, gap, rfi, return_all=False):
    """Downstream line rate (b/s) of ADSL, ADSL2+, VDSL2 17a or G.fast on L m of 26 AWG
    (the models of Chapter 24's figure script)."""
    if kind in ("ADSL", "ADSL2+"):
        f = np.arange(33, 256 if kind == "ADSL" else 512) * 4312.5
        h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
        psd = -40.0
        fx = psd + db(wl.fext_coupling(f, L, np.sqrt(h2), nfext) + 1e-30) if nfext else np.full_like(f, -300.0)
        nz = wl.db_sum(np.full_like(f, noise), fx, rfi_noise(f, rfi))
        r, b, snr = wl.dmt_rate(f, psd, h2, nz, gap, 15)
        if kind == "ADSL":
            r = min(r, 8.1e6)
    elif kind == "VDSL2 17a":
        f = np.arange(1, 4096) * 4312.5
        m = np.zeros_like(f, bool)
        for lo, hi in VDSL_DS:
            m |= (f >= lo) & (f < hi)
        f = f[m]
        h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
        psd = -55.0
        for _ in range(4):
            fx = db(wl.fext_coupling(f, L, np.sqrt(h2), nfext) + 1e-40) + psd if nfext else np.full_like(f, -300.0)
            nz = wl.db_sum(np.full_like(f, noise), fx, rfi_noise(f, rfi))
            r, b, snr = wl.dmt_rate(f, psd, h2, nz, gap, 15)
            psd = min(-40.0, 14.5 - 10 * np.log10(max((b > 0).sum(), 1) * 4312.5))
    else:                                   # G.fast 106 MHz, vectored (30 dB FEXT cancellation)
        df = 51.75e3
        f = np.arange(40, 2048) * df
        h2 = np.abs(wl.loop_gain(f, L, 26)) ** 2
        psd = np.where(f < 30e6, -65.0, -76.0)
        fx = db(wl.fext_coupling(f, L, np.sqrt(h2), nfext) + 1e-40) + psd - 30.0 if nfext else np.full_like(f, -300.0)
        nz = wl.db_sum(np.full_like(f, noise), fx, rfi_noise(f, rfi))
        r, b, snr = wl.dmt_rate(f, psd, h2, nz, gap, 12, sym_rate=48000)
        r = r * 0.94
    if return_all:
        return r, f, b, snr
    return r


STANDARDS = {"ADSL": GRAY, "ADSL2+": NAVY, "VDSL2 17a": ORANGE, "G.fast": GREEN}


class DSLRateReach(Experiment):
    title = "DSL: bits on every tone"
    blurb = "DMT loads each 4 kHz tone to its SNR: watch rate fall with reach, noise and crosstalk."
    book = "sec:ch24:dsl"
    controls = [
        Choice("std", "Standard", list(STANDARDS), "ADSL2+"),
        Slider("L", "Loop length", 0.05, 5.5, 2.0, step=0.05, unit="km"),
        Heading("Noise"),
        IntSlider("nfext", "Crosstalking neighbours (FEXT)", 0, 49, 10),
        Slider("noise", "Background noise", -150, -110, -140, step=1, unit="dBm/Hz"),
        Toggle("rfi", "AM radio ingress", False),
        Slider("gap", "SNR gap Γ", 6, 18, 12, step=0.5, unit="dB",
               help="Gap to capacity: uncoded gap + margin − coding gain"),
    ]
    plots = [
        Plot("snr", "SNR per tone", x="frequency (MHz)", y="SNR (dB)", ylim=(-10, 75), legend=None),
        Plot("bits", "Bits loaded per tone", x="frequency (MHz)", y="bits", ylim=(0, 16.5),
             legend=None),
        Plot("reach", "Rate against reach (background sweep)", x="loop length (km)",
             y="line rate (Mb/s)", logy=True, xlim=(0, 5.5), ylim=(0.5, 2000), legend="tr"),
    ]
    layout = [["snr", "reach"], ["bits", "reach"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("rate", "Line rate", "Mb/s", ".1f"),
        Readout("bps", "Bits per DMT symbol", "", ",.0f"),
        Readout("top", "Highest loaded tone", "MHz", ".2f"),
        Readout("loss", "Loop loss at 1 MHz", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Reproduce the book's ADSL2+ table: about 17.9 Mb/s at 2 km (±0.3) with 10 "
                  "neighbours, −140 dBm/Hz and a 12 dB gap — then push the same loop to 3 km.",
                  lambda s: (s.p.std == "ADSL2+" and abs(s.p.L - 3.0) < 0.03 and s.p.nfext == 10
                             and abs(s.p.gap - 12) < 0.01 and abs(s.p.noise + 140) < 0.5
                             and 6 <= s.r.rate <= 10)),
        Challenge("Find the longest loop on which VDSL2 17a still beats 50 Mb/s (within 50 m).",
                  lambda s: (s.p.std == "VDSL2 17a" and s.r.rate >= 50
                             and s.exp.rate_at(s.p, s.p.L + 0.05) < 50e6)),
        Challenge("Show who limits short loops: on a 300 m VDSL2 loop, quiet the background by "
                  "20 dB and gain less than 10 % — then remove the neighbours instead.",
                  lambda s: (s.p.std == "VDSL2 17a" and abs(s.p.L - 0.3) < 0.03 and s.p.nfext == 0
                             and s.r.rate > 1.3 * s.exp.rate_at(s.p, s.p.L, nfext=20))),
    ]

    def setup(self):
        self.curves = {}
        self.ckey = None
        self.done = False

    def rate_at(self, p, L, nfext=None):
        return dsl_rate(p.std, L * 1e3, p.nfext if nfext is None else nfext, p.noise, p.gap, p.rfi)

    def update(self, p):
        r, f, b, snr = dsl_rate(p.std, p.L * 1e3, p.nfext, p.noise, p.gap, p.rfi, return_all=True)
        fm = f / 1e6
        xmax = {"ADSL": 1.2, "ADSL2+": 2.3, "VDSL2 17a": 18.0, "G.fast": 106.0}[p.std]
        ps = self.plot("snr")
        ps.line("snr", fm, snr, color=NAVY, width=1.6, name="SNR")
        ps.hline("gap", p.gap, color=RED, style="--", label=f"gap Γ = {p.gap:.1f} dB", label_pos=0.6)
        ps.set_xlim(0, xmax)
        pb = self.plot("bits")
        pb.line("b", fm, b, color=NAVY, width=1.4, fill=0, fill_alpha=0.25)
        pb.set_xlim(0, xmax)
        key = (p.nfext, p.noise, p.gap, p.rfi)
        if key != self.ckey:
            self.ckey, self.curves, self.done = key, {}, False
        self.draw_reach(p, r)
        loss = float(-20 * np.log10(np.abs(wl.loop_gain(np.array([1e6]), p.L * 1e3, 26)))[0])
        self.readout(rate=r / 1e6, bps=float(b.sum()), top=float(fm[b > 0].max()) if (b > 0).any() else 0.0,
                     loss=loss)

    def draw_reach(self, p, r_now=None):
        pr = self.plot("reach")
        for k, col in STANDARDS.items():
            if k in self.curves:
                Ls, rs = self.curves[k]
                pr.line(k, Ls, np.maximum(rs, 0.3), color=col, width=2.4 if k == p.std else 1.4,
                        name=k + (" (vectored)" if k == "G.fast" else ""))
        if r_now is None:
            r_now = self.r.get("rate", 1.0) * 1e6
        pr.scatter("now", [p.L], [max(r_now / 1e6, 0.3)], color=RED, size=13, symbol="d")
        pr.hline("50", 50, color=GRAY, style=":", label="50 Mb/s", label_pos=0.02)

    def background(self, p):
        if self.done:
            return
        key = self.ckey
        n = 12 if self.quick else 40
        for k in STANDARDS:
            Lmax = 0.8 if k == "G.fast" else 5.5
            Ls = np.linspace(0.05, Lmax, n)
            rs = np.array([dsl_rate(k, x * 1e3, p.nfext, p.noise, p.gap, p.rfi) for x in Ls]) / 1e6
            yield dict(key=key, std=k, curve=(Ls, rs))
        yield dict(key=key, done=True)

    def progress(self, p, item):
        if item["key"] != self.ckey:
            return
        if item.get("done"):
            self.done = True
        else:
            self.curves[item["std"]] = item["curve"]
        self.draw_reach(p)

    def story(self, p):
        r = self.r
        s = (f"<p>DMT splits the loop into thousands of narrow tones and gives each the number of "
             f"bits its SNR can carry: b = ⌊log₂(1 + SNR/Γ)⌋, at most 15. On "
             f"{v(p.L, '.2f', 'km')} {v(p.std)} loads {v(r.get('bps', 0), ',.0f')} bits per symbol, "
             f"{v(r.get('rate', 0), '.1f', 'Mb/s')}. The SNR (top left) falls with frequency "
             f"because the loop's loss rises as √f; the bits follow it down.</p>")
        if p.nfext > 0:
            s += (f"<p>{v(p.nfext, 'd')} neighbours in the binder leak in by <b>far-end "
                  f"crosstalk</b>, which grows as f²: on short loops it, not the background noise, "
                  f"caps the high tones (vectoring removes it: next experiment).</p>")
        if p.rfi:
            s += ("<p>AM radio stations ride in on the unbalanced drop: DMT just loads zero bits on "
                  "the few tones they hit, a graceful degradation single-carrier modems never had.</p>")
        return "<h3>Rate against reach</h3>" + s + keybox(
            "Rate falls steeply with reach: G.fast for the last 100 m, VDSL2 to a few hundred metres, "
            "ADSL2+ to a few kilometres. Fibre moves closer; copper covers the rest.")


# =============================================================================== 4. vectoring
class Vectoring(Experiment):
    title = "Crosstalk and vectoring"
    blurb = "A 16-line binder is a 16 × 16 MIMO channel: precode in the DSLAM and the noise vanishes."
    book = "sec:ch24:vectoring"
    controls = [
        Slider("L", "Loop length", 100, 1500, 600, step=50, unit="m"),
        Choice("pre", "Precoding", ["None", "Partial", "Full ZF"], "Full ZF"),
        IntSlider("k", "Crosstalkers cancelled per line", 1, 15, 3,
                  enabled_if=lambda p: p.pre == "Partial",
                  help="Partial vectoring cancels only the strongest few disturbers of each line"),
        IntSlider("alien", "Alien (non-vectored) lines", 0, 4, 0),
        Button("again", "New binder"),
    ]
    plots = [
        ImagePlot("H", "The binder at 8 MHz: |Hᵢⱼ| (dB)",
                  x="from transmitter j", y="to receiver i"),
        Plot("bits", "Bits per tone on line 1", x="frequency (MHz)", y="bits", xlim=(0, 18),
             ylim=(0, 17), legend="tr"),
        BarPlot("rates", "Rate of each line", x="line", y="Mb/s"),
    ]
    layout = [["H", "bits"], ["H", "rates"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("mean", "Mean rate", "Mb/s", ".1f"),
        Readout("gain", "Gain over no vectoring", "%", ".0f"),
        Readout("bound", "Crosstalk-free bound", "Mb/s", ".1f"),
        Readout("pen", "Precoder power penalty", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Partial vectoring: cancel as few crosstalkers per line as you can while keeping "
                  "90 % of full vectoring's mean rate.",
                  lambda s: (s.p.pre == "Partial" and s.r.mean >= 0.9 * s.exp.full_mean
                             and s.p.k <= s.exp.k90 + 1e-9),
                  hint="Crosstalk is dominated by a few close neighbours in the cable."),
        Challenge("Let alien lines in until full vectoring loses at least a quarter of its gain.",
                  lambda s: s.p.pre == "Full ZF" and s.p.alien >= 1 and s.exp.lost >= 0.25),
        Challenge("Find the loop lengths on which vectoring adds at least 60 % to the rate.",
                  lambda s: s.p.pre == "Full ZF" and s.p.alien == 0 and s.r.gain >= 60,
                  hint="FEXT grows with frequency, and short loops use the high frequencies."),
    ]
    K = 16
    DECIM = 8

    def setup(self):
        self.seed = 4
        self.full_mean = 1.0
        self.k90 = 15
        self.lost = 0.0

    def on_again(self, p):
        self.seed += 1

    def binder(self, p):
        r = np.random.default_rng(self.seed)
        f = np.arange(1, 4096, self.DECIM) * 4312.5
        m = np.zeros_like(f, bool)
        for lo, hi in VDSL_DS:
            m |= (f >= lo) & (f < hi)
        f = f[m]
        h = wl.loop_gain(f, p.L, 26)
        Kt = self.K + p.alien
        # neighbours in the cable differ: a fixed coupling-strength profile (a few strong ones)
        pos = np.r_[np.arange(self.K), r.uniform(0, self.K, p.alien)]       # aliens sit anywhere
        strength = np.exp(-np.abs(np.subtract.outer(pos, pos)) / 2.5)
        strength *= r.lognormal(0, 0.6, (Kt, Kt))
        np.fill_diagonal(strength, 0)
        strength = strength / strength.sum(1, keepdims=True)           # rows sum to the 1 %-model
        pf = wl.fext_coupling(f, p.L, h, Kt - 1)                        # total FEXT power per line
        ph = r.uniform(0, 2 * np.pi, (len(f), Kt, Kt))
        H = np.sqrt(pf[:, None, None] * strength[None]) * np.exp(1j * ph)
        idx = np.arange(Kt)
        H[:, idx, idx] = h[:, None]
        return f, h, H

    def rates(self, p, f, h, H, mode, k=3):
        K = self.K
        P = 10 ** (-50 / 10)
        N = 10 ** (NOISE / 10)
        Hv = H[:, :K, :K]
        sig = P * np.abs(h)[:, None] ** 2 * np.ones((1, K))
        alien = P * np.sum(np.abs(H[:, :K, K:]) ** 2, axis=2)
        pen = 1.0
        if mode == "None":
            xt = P * (np.sum(np.abs(Hv) ** 2, axis=2) - np.abs(h)[:, None] ** 2)
            snr = sig / (xt + alien + N)
        else:
            D = h[:, None, None] * np.eye(K)[None]
            if mode == "Full ZF":
                Pm = np.linalg.inv(Hv) @ D
            else:
                C = Hv / h[:, None, None]
                C[:, np.arange(K), np.arange(K)] = 0
                mask = np.zeros_like(C, bool)
                order = np.argsort(-np.abs(C), axis=2)[:, :, :k]
                ti = np.arange(len(f))[:, None, None]
                ri = np.arange(K)[None, :, None]
                mask[ti, ri, order] = True
                Pm = np.eye(K)[None] - np.where(mask, C, 0)
            scale = np.max(np.sum(np.abs(Pm) ** 2, axis=2), axis=1)
            pen = float(scale.max())
            E = Hv @ Pm
            dg = np.abs(np.einsum("tii->ti", E)) ** 2
            xt = np.sum(np.abs(E) ** 2, axis=2) - dg
            snr = (P * dg / scale[:, None]) / (P * xt / scale[:, None] + alien + N)
        bits = np.clip(np.floor(np.log2(1 + snr / 10 ** (GAP / 10))), 0, 15)
        rate = bits.sum(axis=0) * 4000 * self.DECIM / 1e6
        return rate, bits, pen

    def update(self, p):
        f, h, H = self.binder(p)
        r_none, b_none, _ = self.rates(p, f, h, H, "None")
        r_full, b_full, pen_full = self.rates(p, f, h, H, "Full ZF")
        r_cur, b_cur, pen = self.rates(p, f, h, H, p.pre, p.k)
        N = 10 ** (NOISE / 10)
        P = 10 ** (-5.0)
        bound_bits = np.clip(np.floor(np.log2(1 + P * np.abs(h) ** 2 / N / 10 ** (GAP / 10))), 0, 15)
        bound = bound_bits.sum() * 4000 * self.DECIM / 1e6
        # alien-free reference for the 'lost' challenge
        if p.alien > 0:
            p0 = st.Params(p, alien=0)
            f0, h0, H0 = self.binder(p0)
            rn0, _, _ = self.rates(p0, f0, h0, H0, "None")
            rf0, _, _ = self.rates(p0, f0, h0, H0, "Full ZF")
            g0 = rf0.mean() - rn0.mean()
            self.lost = 1 - (r_full.mean() - r_none.mean()) / max(g0, 1e-9)
        else:
            self.lost = 0.0
        self.full_mean = r_full.mean()
        # smallest k reaching 90 % of full (cached per binder)
        kkey = (self.seed, p.L, p.alien)
        if getattr(self, "_kkey", None) != kkey:
            self._kkey = kkey
            self.k90 = 15
            for kk in range(1, 16):
                if self.rates(p, f, h, H, "Partial", kk)[0].mean() >= 0.9 * self.full_mean:
                    self.k90 = kk
                    break
        # binder image at 8 MHz
        it = int(np.argmin(np.abs(f - 8e6)))
        Ht = H[it]
        Kt = Ht.shape[0]
        img = 20 * np.log10(np.abs(Ht) / np.abs(h[it]) + 1e-12)
        ph = self.plot("H")
        ph.image("img", np.clip(img, -70, 0), x=(0.5, Kt + 0.5), y=(0.5, Kt + 0.5), cmap="heat",
                 levels=(-70, 0), colorbar=True, cbar_label="dB")
        ph.set_xlim(0.5, Kt + 0.5)
        ph.set_ylim(0.5, Kt + 0.5)
        if p.alien:
            ph.vline("al", self.K + 0.5, color=RED, style="--", width=1.6)
            ph.hline("alh", self.K + 0.5, color=RED, style="--", width=1.6)
        fm = f / 1e6
        pb = self.plot("bits")

        def bands(y):
            out = np.array(y, float)
            gaps = np.where(np.diff(fm) > 0.5)[0]
            return np.insert(fm, gaps + 1, np.nan), np.insert(out, gaps + 1, np.nan)

        x, y = bands(bound_bits)
        pb.line("bnd", x, y, color=GREEN, width=2.6, name="no crosstalk")
        x, y = bands(b_none[:, 0])
        pb.line("none", x, y, color=RED, width=1.2, name="no vectoring")
        if p.pre != "None":
            x, y = bands(b_cur[:, 0])
            pb.line("cur", x, y, color=NAVY, width=1.4, name=f"{p.pre.lower()} vectoring")
        pr = self.plot("rates")
        lines = np.arange(1, self.K + 1)
        pr.bars("none", lines - 0.2, r_none, width=0.38, color=GRAY)
        pr.bars("cur", lines + 0.2, r_cur, width=0.38, color=NAVY if p.pre != "None" else RED)
        pr.hline("bnd", bound * 1.0, color=GREEN, style="--")
        pr.text("bl", self.K + 0.6, bound, "crosstalk-free", color=GREEN, anchor=(1, 1.1), size=8.5)
        pr.text("leg", 0.5, bound * 1.32, "grey: no vectoring    navy: your precoder", color=GRAY,
                anchor=(0, 0.5), size=8.5)
        pr.set_xlim(0.3, self.K + 0.7)
        pr.set_ylim(0, bound * 1.42 + 1)
        pr.set_xticks([(i, str(i)) for i in lines])
        self.r_none = r_none.mean()
        self.readout(mean=r_cur.mean(), gain=100 * (r_cur.mean() / max(r_none.mean(), 1e-9) - 1),
                     bound=bound, pen=10 * np.log10(pen))

    def story(self, p):
        r = self.r
        s = (f"<p>Sixteen VDSL2 lines share a {v(p.L, '.0f', 'm')} binder. On each tone the binder "
             f"is a 16 × 16 matrix (left): the bright diagonal is each line's own loop, the "
             f"off-diagonal speckle is <b>far-end crosstalk</b>, 30–60 dB weaker but adding up, and "
             f"strongest between neighbouring pairs. Without vectoring it limits the high tones "
             f"(red, top right): {v(self.r_none, '.0f', 'Mb/s')} per line on average.</p>")
        if p.pre == "None":
            s += "<p>Choose a precoder to cancel it.</p>"
        else:
            s += (f"<p>All 16 transmitters sit in one DSLAM, so it can <b>precode</b>: send "
                  f"x = P·s with P chosen so that each receiver sees only its own loop (G.993.5). "
                  f"Because the matrix is strongly diagonal, P ≈ I and the power penalty is just "
                  f"{v(r.get('pen', 0), '.2f', 'dB')}. Result: {v(r.get('mean', 0), '.0f', 'Mb/s')}, "
                  f"against a crosstalk-free bound of {v(r.get('bound', 0), '.0f', 'Mb/s')}.</p>")
        if p.alien:
            s += (f"<p>{v(p.alien, 'd')} <b>alien</b> line(s) (beyond the dashed lines) belong to "
                  f"another operator's DSLAM: their crosstalk cannot be cancelled, which is why "
                  f"operators fought for exclusive access to street cabinets.</p>")
        return "<h3>Crosstalk is not noise</h3>" + s + keybox(
            "Crosstalk is known signal, not noise: precoding in the DSLAM turns a binder back into "
            "a bundle of isolated pairs, if every line is in the vectored group.")


# =============================================================================== 5. dispersion
WAVES = {"1550 nm (C band)": (1550e-9, float(wl.dispersion_ps_nm_km(1550.0))),
         "1310 nm (O band)": (1310e-9, float(wl.dispersion_ps_nm_km(1310.0)))}


def eye_opening(y, sym, sps, d, levels):
    """Worst inner eye opening (fraction of the level spacing) at the best sampling phase."""
    k = np.arange(10, len(sym) - 10)
    best = -np.inf
    for ph in range(sps):
        s = y[d + k * sps + ph - sps // 2]
        op = np.inf
        for j in range(len(levels) - 1):
            lo, hi = s[sym[k] == j], s[sym[k] == j + 1]
            if len(lo) and len(hi):
                op = min(op, hi.min() - lo.max())
        best = max(best, op)
    return best


class FibreDispersion(Experiment):
    title = "Dispersion: IM/DD against coherent"
    blurb = "The same fibre closes a PAM-4 eye and leaves a coherent constellation untouched."
    book = "sec:ch24:fibre"
    controls = [
        Slider("L", "Fibre length", 0, 100, 10, step=0.5, unit="km"),
        Choice("baud", "Symbol rate", ["10 GBd", "28 GBd", "56 GBd"], "28 GBd"),
        Choice("wave", "Wavelength", list(WAVES), "1550 nm (C band)", style="menu"),
        Toggle("cdc", "Coherent receiver compensates CD", True),
        Button("again", "New symbols"),
    ]
    plots = [
        SpectrumPlot("resp", "IM/DD power-fading response", x="modulation frequency (GHz)",
                     y="response (dB)", xlim=(0, 40), ylim=(-30, 3), legend=None),
        EyePlot("eye", "PAM-4, direct detection (photodiode)", yrange=(-1.6, 1.6)),
        ConstellationPlot("con", "QPSK, coherent receiver", lim=1.8),
    ]
    layout = [["resp", "resp"], ["eye", "con"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("DL", "Accumulated dispersion", "ps/nm", ",.0f"),
        Readout("spread", "Pulse spread", "symbols", ".1f"),
        Readout("null", "First IM/DD null", "GHz", None),
        Readout("eye", "PAM-4 eye opening", "%", ".0f", good=lambda x: x > 20),
    ]
    challenges = [
        Challenge("At 28 GBd in the C band, find the longest fibre that keeps the PAM-4 eye at "
                  "least 20 % open.",
                  lambda s: (s.p.baud == "28 GBd" and s.p.wave.startswith("1550") and s.r.eye >= 20
                             and s.exp.eye_at(s.p, s.p.L + 1.0) < 20)),
        Challenge("Run 56 GBd PAM-4 over 20 km or more with an open eye (≥ 20 %).",
                  lambda s: s.p.baud == "56 GBd" and s.p.L >= 20 and s.r.eye >= 20,
                  hint="Data-centre optics moved to the O band for a reason."),
        Challenge("Switch the coherent receiver's CD compensation off at 80 km and see the "
                  "constellation dissolve.",
                  lambda s: not s.p.cdc and s.p.L >= 80),
    ]
    sps = 8
    nsym = 1600

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def links(self, p, L_km, eye_only=False):
        r = np.random.default_rng(self.seed)
        baud = float(p.baud.split()[0]) * 1e9
        lam, D = WAVES[p.wave]
        fs = baud * self.sps
        taps = self._taps()
        lev = r.integers(0, 4, self.nsym)
        rc = self._rc()                                   # Nyquist intensity pulses: no ISI at 0 km
        pam = cl.shape((lev / 3.0) * 0.8 + 0.1, rc, self.sps).real
        field = np.sqrt(np.maximum(pam, 1e-4))
        det = np.abs(wl.apply_cd(field, fs, L_km * 1e3, D, lam)) ** 2
        det = det - det.mean() + 0.01 * r.standard_normal(len(det))   # AC-coupled, a little noise
        det = det / 0.4                                       # ideal levels at ±0.6, ±0.2... ≈ ±1
        d = (len(rc) - 1) // 2
        op = eye_opening(det, lev, self.sps, d, np.arange(4))
        op_frac = op / (0.8 / 3 / 0.4)
        if eye_only:
            return op_frac
        qp = cl.get_constellation("qpsk").points[r.integers(0, 4, self.nsym)]
        coh = cl.shape(qp, taps, self.sps)
        rx = wl.apply_cd(coh, fs, L_km * 1e3, D, lam)
        rx = rx + 0.08 * np.sqrt(np.mean(np.abs(rx) ** 2)) * (r.standard_normal(len(rx))
                                                            + 1j * r.standard_normal(len(rx)))
        if p.cdc:
            rx = wl.cd_compensate(rx, fs, L_km * 1e3, D, lam)
        y = np.convolve(rx, taps, "full")[2 * d::self.sps][20:self.nsym - 20]
        y = y / np.sqrt(np.mean(np.abs(y) ** 2))
        return det, d, op_frac, y, qp

    @lru_cache(maxsize=4)
    def _taps(self):
        return cl.rrc_taps(0.3, self.sps, span=16)

    @lru_cache(maxsize=4)
    def _rc(self):
        h = cl.rc_taps(0.3, self.sps, 16)
        return h / h.max()

    def eye_at(self, p, L_km):
        return 100 * self.links(p, L_km, eye_only=True)

    def update(self, p):
        baud = float(p.baud.split()[0])
        lam, D = WAVES[p.wave]
        det, d, op, y, qp = self.links(p, p.L)
        fr_ = np.linspace(0.01, 40, 600) * 1e9
        h = wl.imdd_cd_response(fr_, max(p.L, 1e-3) * 1e3, D, lam)
        pr = self.plot("resp")
        pr.line("h", fr_ / 1e9, 20 * np.log10(np.abs(h) + 1e-6), color=NAVY, width=2.2)
        pr.band("sig", 0, baud / 2, color=GREEN, alpha=0.08)
        pr.vline("nyq", baud / 2, color=GREEN, style="--", label="signal's Nyquist frequency",
                 label_pos=0.08)
        if abs(D) * p.L > 1e-6:
            f1 = np.sqrt(wl.C / (2 * abs(D) * 1e-6 * p.L * 1e3 * lam ** 2)) / 1e9
        else:
            f1 = np.inf
        if f1 < 40:
            pr.vline("null", f1, color=RED, style=":", label="first null", label_pos=0.3)
        pe = self.plot("eye")
        pe.eye("eye", det, self.sps, n_sym=2, offset=d - self.sps + 20 * self.sps,
               yrange=(-1.6, 1.6))
        pc = self.plot("con")
        pc.points("y", np.clip(y.real, -1.75, 1.75) + 1j * np.clip(y.imag, -1.75, 1.75), color=NAVY,
                  size=3, alpha=0.5)
        pc.ideal("i", cl.get_constellation("qpsk").points / np.sqrt(np.mean(np.abs(cl.get_constellation("qpsk").points) ** 2)))
        DL = abs(D) * p.L
        dl_nm = lam ** 2 * baud * 1e9 / wl.C * 1e9
        spread = DL * dl_nm * baud / 1e3
        self.f1 = f1
        self.readout(DL=DL, spread=spread, null=("—" if not np.isfinite(f1) or f1 > 999 else f"{f1:.1f}"),
                     eye=max(100 * op, -100.0))

    def story(self, p):
        r = self.r
        baud = float(p.baud.split()[0])
        s = (f"<p>Different wavelengths travel at different speeds: {v(r.get('DL', 0), ',.0f', 'ps/nm')} "
             f"of dispersion after {v(p.L, '.1f', 'km')}, enough to smear each symbol over "
             f"{v(r.get('spread', 0), '.1f')} neighbours at {v(p.baud)}.</p>")
        s += ("<p><b>Direct detection</b> measures only power. The two sidebands of the intensity "
              "modulation are rotated in opposite directions by dispersion and cancel at the nulls "
              "of the response (top): ")
        if np.isfinite(self.f1) and self.f1 < baud / 2:
            s += (f"{bad('the first null')} at {v(self.f1, '.1f', 'GHz')} sits inside the signal "
                  f"band, and the PAM-4 eye closes.</p>")
        else:
            s += f"{good('no null inside the signal band')}, the eye stays open.</p>"
        s += ("<p>A <b>coherent</b> receiver measures the optical field itself, so dispersion is just "
              "a known all-pass filter that DSP undoes" + (": the QPSK points are clean at any length."
                                                          if p.cdc else
                                                          ". Switched off, the field is a smeared cloud.")
              + "</p>")
        return "<h3>Power fading versus field recovery</h3>" + s + keybox(
            "IM/DD at 50–100 GBd lives within a few km or moves to the O band; coherent links undo "
            "thousands of ps/nm in DSP and cross oceans.")


# =============================================================================== 6. coherent DSP
def cma_block(W, x, y, start, nsym, ntaps, mu, sps=2):
    """Run the 2 × 2 CMA butterfly over ``nsym`` symbols starting at symbol ``start`` (in place on
    W). Returns the X and Y outputs and the CMA error |1 − |z|²|² for each symbol."""
    zx = np.empty(nsym, complex)
    zy = np.empty(nsym, complex)
    err = np.empty(nsym)
    for i in range(nsym):
        k = start + i
        sx = x[k * sps:k * sps + ntaps][::-1]
        sy = y[k * sps:k * sps + ntaps][::-1]
        ox = W[0, 0] @ sx + W[0, 1] @ sy
        oy = W[1, 0] @ sx + W[1, 1] @ sy
        ex = ox * (1 - abs(ox) ** 2)
        ey = oy * (1 - abs(oy) ** 2)
        W[0, 0] += mu * ex * sx.conj()
        W[0, 1] += mu * ex * sy.conj()
        W[1, 0] += mu * ey * sx.conj()
        W[1, 1] += mu * ey * sy.conj()
        zx[i], zy[i] = ox, oy
        err[i] = (1 - abs(ox) ** 2) ** 2
    return zx, zy, err


class CoherentDSP(Experiment):
    title = "Inside a coherent receiver"
    blurb = "Watch CMA untangle two polarisations and Viterbi–Viterbi stop the spinning, live."
    book = "sec:ch24:coherent"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("The link (32 GBd DP-QPSK)"),
        Slider("L", "Fibre length", 0, 2000, 80, step=10, unit="km"),
        Slider("theta", "Polarisation rotation", 0, 1.57, 0.6, step=0.01, unit="rad"),
        Slider("lw", "Combined laser linewidth", 10, 20000, 200, step=10, unit="kHz"),
        Slider("fo", "Laser frequency offset", -1000, 1000, 150, step=10, unit="MHz"),
        Slider("snr", "SNR", 6, 30, 16, step=0.5, unit="dB"),
        Heading("The DSP"),
        LogSlider("mu", "CMA step size", 1e-4, 1e-2, 2e-3, unit="", fmt=".4f"),
        IntSlider("blk", "Phase-averaging block", 4, 128, 40, unit="symbols"),
        Button("restart", "Restart the equaliser"),
    ]
    plots = [
        ConstellationPlot("raw", "After CD compensation", lim=2.0),
        ConstellationPlot("cma", "After the CMA butterfly", lim=2.0),
        ConstellationPlot("cpr", "After frequency and phase recovery", lim=2.0),
        Plot("conv", "CMA convergence", x="symbols processed", y="CMA error (dB)",
             ylim=(-25, 5), legend=None),
    ]
    layout = [["raw", "cma", "cpr"], ["conv", "conv", "conv"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("n", "Symbols processed", "", "int"),
        Readout("err", "CMA error", "dB", ".1f", good=lambda x: x < -10),
        Readout("fo", "Offset estimate", "MHz", ".0f"),
        Readout("evm", "EVM after recovery", "dB", ".1f", good=lambda x: x < -12),
    ]
    challenges = [
        Challenge("Converge the butterfly within 700 symbols (CMA error below −10 dB).",
                  lambda s: s.exp.conv_at is not None and s.exp.conv_at <= 700,
                  hint="A larger step size learns faster but jitters more."),
        Challenge("Cross the Pacific on a thin budget: 2000 km at an SNR of 14 dB or less, with an "
                  "EVM better than −12.5 dB.",
                  lambda s: (s.p.L >= 2000 and s.p.snr <= 14 and s.r.n > 3000
                             and isinstance(s.r.evm, float) and s.r.evm < -12.5)),
        Challenge("Cheap lasers: with a 20 MHz linewidth, reach an EVM better than −18 dB.",
                  lambda s: (s.p.lw >= 19990 and s.r.n > 3000 and isinstance(s.r.evm, float)
                             and s.r.evm < -18),
                  hint="A long averaging block cannot follow a fast-wandering phase; give it more "
                       "SNR and a short block."),
    ]
    RS, SPS, NSYM, NTAPS = 32e9, 2, 9000, 15

    def setup(self):
        self.stream_key = None

    def on_restart(self, p):
        self.stream_key = None

    def make_stream(self, p):
        r = np.random.default_rng(7)
        rs, sps = self.RS, self.SPS
        fs = rs * sps
        n = self.NSYM
        qp = (np.array([1, -1])[r.integers(0, 2, (2, n))]
              + 1j * np.array([1, -1])[r.integers(0, 2, (2, n))]) / np.sqrt(2)
        taps = cl.rrc_taps(0.1, sps, span=32)
        tx = np.array([cl.shape(qp[k], taps, sps) for k in range(2)])
        tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2, axis=1, keepdims=True))
        th = p.theta
        U = np.array([[np.cos(th), -np.sin(th) * np.exp(-1.1j)], [np.sin(th) * np.exp(1.1j), np.cos(th)]])
        f = np.fft.fftfreq(tx.shape[1], 1 / fs)
        Xf = np.fft.fft(tx, axis=1)
        Xf[0] *= np.exp(-1j * np.pi * f * 8e-12)
        Xf[1] *= np.exp(1j * np.pi * f * 8e-12)                      # 8 ps of DGD
        rx = U @ np.fft.ifft(Xf, axis=1)
        rx = np.array([wl.apply_cd(rx[k], fs, p.L * 1e3) for k in range(2)])
        m = rx.shape[1]
        phase = (np.cumsum(r.normal(0, np.sqrt(2 * np.pi * p.lw * 1e3 / fs), m))
                 + 2 * np.pi * p.fo * 1e6 * np.arange(m) / fs)
        rx = rx * np.exp(1j * phase)
        rx = rx + np.sqrt(np.mean(np.abs(rx) ** 2) / 10 ** (p.snr / 10) / 2 * sps) * (
            r.normal(size=rx.shape) + 1j * r.normal(size=rx.shape))
        mf = np.array([np.convolve(rx[k], taps, mode="same") for k in range(2)])
        cdc = np.array([wl.cd_compensate(mf[k], fs, p.L * 1e3) for k in range(2)])
        cdc = cdc / np.sqrt(np.mean(np.abs(cdc) ** 2, axis=1, keepdims=True))
        self.x, self.y = cdc[0], cdc[1]
        self.W = np.zeros((2, 2, self.NTAPS), complex)
        self.W[0, 0, self.NTAPS // 2] = 1
        self.W[1, 1, self.NTAPS // 2] = 1
        self.k = 0
        self.total = 0
        self.zx = np.zeros(0, complex)
        self.err_hist = []
        self.conv_at = None
        self.last_cpr = None

    def step(self, p, nsym=350):
        nmax = (len(self.x) - self.NTAPS) // self.SPS - 1
        if self.k + nsym > nmax:
            self.k = 0
        zx, zy, err = cma_block(self.W, self.x, self.y, self.k, nsym, self.NTAPS, p.mu, self.SPS)
        self.k += nsym
        self.total += nsym
        self.zx = np.r_[self.zx, zx][-2400:]
        e_db = 10 * np.log10(np.mean(err) + 1e-12)
        self.err_hist.append((self.total, e_db))
        if self.conv_at is None and e_db < -10:
            self.conv_at = self.total

    def recover(self, p):
        z = self.zx
        if len(z) < 400:
            return None, None, None
        fo = wl.fourth_power_fo(z, self.RS)
        zc = z * np.exp(-2j * np.pi * fo * np.arange(len(z)) / self.RS)
        z3, _ = wl.vv_carrier_recovery(zc, block=int(p.blk))
        z3 = z3[p.blk:-p.blk] if len(z3) > 2 * p.blk + 50 else z3
        z3 = z3 / np.sqrt(np.mean(np.abs(z3) ** 2))
        ref = (np.sign(z3.real) + 1j * np.sign(z3.imag)) / np.sqrt(2)
        evm = 10 * np.log10(np.mean(np.abs(z3 - ref) ** 2))
        return z3, fo, evm

    def draw(self, p):
        k0 = max(self.k - 1200, 0)
        raw = self.x[k0 * self.SPS:(k0 + 1200) * self.SPS:self.SPS]
        clip = lambda z: np.clip(z.real, -1.95, 1.95) + 1j * np.clip(z.imag, -1.95, 1.95)
        self.plot("raw").points("p", clip(raw), color=GRAY, size=2, alpha=0.5)
        self.plot("cma").points("p", clip(self.zx[-1200:]) if len(self.zx) else np.zeros(1), color=NAVY,
                                size=2, alpha=0.5)
        z3, fo, evm = self.recover(p)
        pc = self.plot("cpr")
        if z3 is not None:
            pc.points("p", clip(z3[-1200:]), color=GREEN, size=2, alpha=0.6)
        pc.ideal("i", np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2))
        pv = self.plot("conv")
        if self.err_hist:
            h = np.array(self.err_hist)
            pv.line("e", h[:, 0], h[:, 1], color=NAVY, width=2.0)
            pv.set_xlim(0, max(3000, h[-1, 0] * 1.05))
        pv.hline("th", -10, color=GREEN, style="--", label="converged", label_pos=0.02)
        if self.conv_at is not None:
            pv.vline("c", self.conv_at, color=GREEN, style=":")
        e = self.err_hist[-1][1] if self.err_hist else 0.0
        self.readout(n=self.total, err=e, fo=fo / 1e6 if fo is not None else "—",
                     evm=evm if evm is not None else "—")

    def update(self, p):
        key = (p.L, p.theta, p.lw, p.fo, p.snr)
        if key != self.stream_key:
            self.make_stream(p)
            self.stream_key = key
        self.step(p, 350)
        self.draw(p)

    def tick(self, p):
        self.step(p, 350)
        self.draw(p)

    def story(self, p):
        r = self.r
        n = r.get("n", 0)
        s = (f"<p>A dual-polarisation coherent receiver mixes the light with a local laser and "
             f"samples four streams. Left: after the fixed <b>chromatic-dispersion</b> filter for "
             f"{v(p.L, '.0f', 'km')} the symbols are a ring: the two polarisations are still mixed "
             f"by the fibre ({v(p.theta, '.2f', 'rad')} of rotation and 8 ps of DGD) and the phase is "
             f"spinning at {v(p.fo, '.0f', 'MHz')}.</p>"
             f"<p>Middle: the <b>2 × 2 CMA butterfly</b> adapts four FIR filters blindly, pushing "
             f"every output towards constant modulus; it has seen {v(n, ',d')} symbols and its "
             f"error is {v(r.get('err', 0), '.1f', 'dB')} (bottom). Right: the frequency offset "
             f"from the fourth-power spectrum, then <b>Viterbi–Viterbi</b> phase estimation over "
             f"{v(int(p.blk), 'd')}-symbol blocks: four clean clusters.</p>")
        if p.lw > 3000:
            s += ("<p>These lasers are noisy (cheap DFBs reach megahertz linewidths): a long "
                  "averaging block cannot follow the phase; a short one averages too little noise.</p>")
        return "<h3>Four problems, four algorithms</h3>" + s + keybox(
            "CD compensation → polarisation demultiplexing (CMA) → frequency offset → carrier phase. "
            "Viterbi–Viterbi leaves a four-fold ambiguity, resolved with pilots or differential coding.")


# =============================================================================== 7. OSNR / GN
FORMATS = {"100G DP-QPSK, 32 GBd": (4, 32e9), "400G DP-16QAM, 64 GBd": (16, 64e9),
           "600G DP-64QAM, 64 GBd": (64, 64e9)}


@lru_cache(maxsize=8)
def req_snr_db(M, ber=2e-2):
    return brentq(lambda s: cl.ber_mqam_gray(s, M) - ber, -5, 40)


class LinkPlanning(Experiment):
    title = "An amplified link: OSNR and the GN model"
    blurb = "More launch power helps until the glass itself starts to talk back."
    book = "sec:ch24:wdm"
    controls = [
        IntSlider("n", "Number of spans", 1, 80, 13),
        Slider("span", "Span length", 40, 120, 80, step=1, unit="km",
               help="Loss 0.22 dB/km including splices"),
        Slider("nf", "Amplifier noise figure", 3, 8, 5, step=0.1, unit="dB"),
        Slider("p", "Launch power per channel", -6, 8, 1, step=0.1, unit="dBm"),
        Choice("fmt", "Format", list(FORMATS), "400G DP-16QAM, 64 GBd", style="menu"),
    ]
    plots = [
        Plot("snr", "SNR against launch power", x="launch power (dBm per channel)", y="SNR (dB)",
             xlim=(-6, 8), legend="bl"),
        Plot("osnr", "OSNR against distance", x="distance (km)", y="OSNR in 0.1 nm (dB)", logx=True,
             xlim=(40, 10000), ylim=(5, 45), legend="tr"),
        BarPlot("reach", "Reach at the optimum launch power", y="km"),
    ]
    layout = [["snr", "snr"], ["osnr", "reach"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("osnr", "OSNR", "dB", ".1f"),
        Readout("snr", "SNR with nonlinearity", "dB", ".1f"),
        Readout("popt", "Optimum launch power", "dBm", ".1f"),
        Readout("margin", "Margin", "dB", ".1f", good=lambda x: x >= 0),
    ]
    challenges = [
        Challenge("Set the launch power to the optimum (±0.1 dB).",
                  lambda s: abs(s.p.p - s.r.popt) <= 0.1 + 1e-9),
        Challenge("Carry 400G DP-16QAM over 2000 km or more with a positive margin.",
                  lambda s: (s.p.fmt.startswith("400G") and s.p.n * s.p.span >= 2000
                             and s.r.margin >= 0),
                  hint="Shorter spans: less loss per span. Better amplifiers: lower NF."),
        Challenge("Overdrive: launch at least 3 dB above the optimum and lose 2 dB of SNR or more.",
                  lambda s: s.p.p >= s.r.popt + 3 and s.exp.drop >= 2),
    ]

    def update(self, p):
        M, rs = FORMATS[p.fmt]
        span_db = 0.22 * p.span
        eta = wl.gn_eta(span_km=p.span, rs=rs)
        Pd = np.linspace(-6, 8, 281)
        Pw = 1e-3 * 10 ** (Pd / 10)
        snr_gn = 10 * np.log10(wl.gn_snr(Pw, p.n, span_db, p.nf, rs=rs, eta=eta))
        snr_ase = 10 * np.log10(wl.gn_snr(Pw, p.n, span_db, p.nf, rs=rs, eta=0.0))
        need = req_snr_db(M) + 2.0
        k = int(np.argmax(snr_gn))
        snr_now = float(10 * np.log10(wl.gn_snr(1e-3 * 10 ** (p.p / 10), p.n, span_db, p.nf, rs=rs,
                                                 eta=eta)))
        ps = self.plot("snr")
        ps.line("ase", Pd, snr_ase, color=GRAY, width=1.6, style="--", name="ASE noise only")
        ps.line("gn", Pd, snr_gn, color=NAVY, width=2.6, name="ASE + nonlinear interference")
        ps.hline("need", need, color=RED, style=":", label=f"needed ({need:.1f} dB incl. 2 dB)",
                 label_pos=0.02)
        ps.scatter("opt", [Pd[k]], [snr_gn[k]], color=GREEN, size=12, name="optimum")
        ps.vline("now", p.p, color=ORANGE, style="--")
        ps.scatter("now", [p.p], [snr_now], color=ORANGE, size=13, symbol="d")
        ps.set_ylim(min(need, snr_gn.min()) - 4, max(snr_gn.max(), need) + 6)
        Ns = np.arange(1, 126)
        po = self.plot("osnr")
        osnr_n = wl.osnr_db(p.p, span_db, p.nf, Ns)
        po.line("o", Ns * p.span, osnr_n, color=NAVY, width=2.2, name=f"at {p.p:.1f} dBm")
        po.hline("need", need + 10 * np.log10(rs / 12.5e9), color=RED, style=":",
                 label="required OSNR", label_pos=0.02)
        po.vline("d", p.n * p.span, color=ORANGE, style="--")
        # reach per format at the optimum launch power
        pr = self.plot("reach")
        reach = []
        for i, (nm, (Mi, rsi)) in enumerate(FORMATS.items()):
            etai = wl.gn_eta(span_km=p.span, rs=rsi)
            needi = req_snr_db(Mi) + 2.0
            best = [10 * np.log10(wl.gn_snr(Pw, n_, span_db, p.nf, rs=rsi, eta=etai).max())
                    for n_ in range(1, 201)]
            ok = [n_ + 1 for n_, s_ in enumerate(best) if s_ >= needi]
            reach.append((max(ok) if ok else 0) * p.span)
        names = ["100G QPSK", "400G 16QAM", "600G 64QAM"]
        cur = list(FORMATS).index(p.fmt)
        pr.bars("b", [0, 1, 2], reach, width=0.6, colors=[NAVY if i == cur else GRAY for i in range(3)])
        for i, val in enumerate(reach):
            pr.text(f"t{i}", i, val, f"{val:,.0f} km", anchor=(0.5, 1.05), size=8.5, bold=True)
        pr.set_xticks([(i, n_) for i, n_ in enumerate(names)])
        pr.set_xlim(-0.6, 2.6)
        pr.set_ylim(0, max(reach) * 1.25 + 100)
        self.drop = float(snr_gn[k] - snr_now)
        osnr = float(wl.osnr_db(p.p, span_db, p.nf, p.n))
        self.need = need
        self.readout(osnr=osnr, snr=snr_now, popt=float(Pd[k]), margin=snr_now - need)

    def story(self, p):
        r = self.r
        s = (f"<p>{v(p.n, 'd')} spans of {v(p.span, '.0f', 'km')} ({v(p.n * p.span, ',.0f', 'km')}), "
             f"each losing {v(0.22 * p.span, '.1f', 'dB')} and each followed by an amplifier that adds "
             f"noise: OSNR = P − L<sub>s</sub> − NF − 10·log₁₀N + 58 = {v(r.get('osnr', 0), '.1f', 'dB')}. "
             f"Every doubling of the spans costs 3 dB.</p>"
             f"<p>Raising the launch power raises the OSNR, but the Kerr effect makes the glass "
             f"respond to the light's intensity, and every channel mixes with every other: "
             f"nonlinear interference grows as P³ (the GN model). The SNR peaks at "
             f"{v(r.get('popt', 0), '.1f', 'dBm')} per channel and then <b>falls</b>.</p>"
             f"<p>At {v(p.p, '.1f', 'dBm')} {p.fmt.split(',')[0]} has "
             + (good(f"{r.get('margin', 0):.1f} dB of margin.") if r.get("margin", -1) >= 0 else
                bad(f"{-r.get('margin', 0):.1f} dB too little SNR.")) + "</p>")
        return "<h3>The nonlinear Shannon limit</h3>" + s + keybox(
            "An amplified link has an optimum launch power: ASE noise below it, Kerr nonlinearity "
            "above. Reach falls steeply with the constellation.")


# =============================================================================== 8. PON
OPTICS = {"GPON class B+": dict(down=(1.5, -27.0), up=(0.5, -28.0)),
          "GPON class C+": dict(down=(3.0, -30.0), up=(0.5, -32.0))}
SPLITS = {"1:16": 16, "1:32": 32, "1:64": 64, "1:128": 128}


class PONBudget(Experiment):
    title = "A passive optical network budget"
    blurb = "One fibre, one splitter, 32 homes: is there light enough for all of them?"
    book = "sec:ch24:pon"
    controls = [
        Choice("split", "Splitter", list(SPLITS), "1:32"),
        Slider("d", "Distance to the farthest home", 0, 40, 20, step=0.5, unit="km"),
        Choice("dir", "Direction", ["Downstream (1490 nm)", "Upstream (1310 nm)"],
               "Downstream (1490 nm)", style="menu"),
        Choice("opt", "Optics class (approximate)", list(OPTICS), "GPON class B+", style="menu"),
        IntSlider("con", "Connectors (0.3 dB each)", 0, 10, 4),
        IntSlider("spl", "Splices (0.1 dB each)", 0, 30, 10),
    ]
    plots = [
        BarPlot("bud", "From the transmitter to the receiver (dBm / dB)", y="dBm"),
        Plot("dist", "Received power against distance", x="distance (km)", y="received power (dBm)",
             xlim=(0, 40), ylim=(-40, 0), legend=None),
    ]
    layout = [["bud", "dist"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("loss", "Total loss", "dB", ".1f"),
        Readout("rx", "Received power", "dBm", ".1f"),
        Readout("margin", "Margin", "dB", ".1f", good=lambda x: x >= 0),
        Readout("reach", "Maximum reach", "km", ".1f"),
    ]
    challenges = [
        Challenge("Serve 64 homes 20 km away with at least 1 dB of downstream margin.",
                  lambda s: (s.p.split == "1:64" and s.p.d >= 20 and s.p.dir.startswith("Down")
                             and s.r.margin >= 1)),
        Challenge("B+ optics, 1:32 split: find the maximum downstream reach (within 1 km).",
                  lambda s: (s.p.opt.endswith("B+") and s.p.split == "1:32" and s.p.dir.startswith("Down")
                             and s.r.margin >= 0 and s.p.d >= s.r.reach - 1)),
        Challenge("Make the fibre, not the splitter, the biggest loss in the budget.",
                  lambda s: s.exp.fibre_db > s.exp.split_db,
                  hint="Small split, long fibre, the lossier wavelength."),
    ]

    def update(self, p):
        down = p.dir.startswith("Down")
        alpha = 0.30 if down else 0.35
        launch, sens = OPTICS[p.opt]["down" if down else "up"]
        N = SPLITS[p.split]
        split_db = 10 * np.log10(N) + 0.39 * np.log2(N)
        fibre_db = alpha * p.d
        items = [("launch", launch), ("splitter", -split_db), ("fibre", -fibre_db),
                 ("connectors", -0.3 * p.con), ("splices", -0.1 * p.spl)]
        pb = self.plot("bud")
        rx = waterfall(pb, "w", items, "received")
        pb.hline("sens", sens, color=ORANGE, style="--", label=f"sensitivity {sens:.0f} dBm",
                 label_pos=0.02)
        pb.set_ylim(min(rx, sens) - 6, launch + 6)
        loss = split_db + fibre_db + 0.3 * p.con + 0.1 * p.spl
        fixed = split_db + 0.3 * p.con + 0.1 * p.spl
        reach = max(0.0, (launch - sens - fixed) / alpha)
        dd = np.linspace(0, 40, 81)
        pd = self.plot("dist")
        for nm, Nn in SPLITS.items():
            sd = 10 * np.log10(Nn) + 0.39 * np.log2(Nn)
            yy = launch - sd - 0.3 * p.con - 0.1 * p.spl - alpha * dd
            pd.line(nm, dd, yy, color=NAVY if nm == p.split else GRAY,
                    width=2.4 if nm == p.split else 1.0)
            pd.text("t" + nm, 1, yy[2], nm, color=NAVY if nm == p.split else GRAY, anchor=(0, 1.1),
                    size=8.5, bold=nm == p.split)
        pd.hline("sens", sens, color=ORANGE, style="--", label="sensitivity", label_pos=0.02)
        pd.scatter("now", [p.d], [rx], color=RED, size=13, symbol="d")
        self.fibre_db, self.split_db = fibre_db, split_db
        self.readout(loss=loss, rx=rx, margin=rx - sens, reach=min(reach, 99.0))

    def story(self, p):
        r = self.r
        s = (f"<p>A <b>passive optical network</b> shares one fibre and one transceiver at the "
             f"exchange among {v(SPLITS[p.split], 'd')} homes through unpowered splitters. Each "
             f"halving of the light costs 3 dB plus a little excess: the {v(p.split)} splitter alone "
             f"takes {v(self.split_db, '.1f', 'dB')}, the {v(p.d, '.1f', 'km')} of fibre only "
             f"{v(self.fibre_db, '.1f', 'dB')}.</p>"
             f"<p>Total loss {v(r.get('loss', 0), '.1f', 'dB')}, received "
             f"{v(r.get('rx', 0), '.1f', 'dBm')}: "
             + (good(f"{r.get('margin', 0):.1f} dB of margin.") if r.get('margin', -1) >= 0 else
                bad(f"{-r.get('margin', 0):.1f} dB short.")) +
             f" The book's worked example (B+, 1:32, 20 km, 4 connectors, 10 splices) has 3.3 dB.</p>")
        return "<h3>The splitter pays the bill</h3>" + s + keybox(
            "PON design is a power budget: the splitter, not the fibre, dominates, and higher splits "
            "or longer reach need stronger optics classes.")


# =============================================================================== the lab
LAB = st.Lab(30, "Wireline and Optical: Loops, DSL, Fibre and Coherent DSP", chapter=24,
             chapter_title="Wireline and Optical Communications",
             experiments=[LineConstants, LoopImpairments, DSLRateReach, Vectoring, FibreDispersion,
                          CoherentDSP, LinkPlanning, PONBudget])

if __name__ == "__main__":
    st.run(LAB)
