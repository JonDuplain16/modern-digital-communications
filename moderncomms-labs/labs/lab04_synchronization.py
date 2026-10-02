"""Lab 04 · Synchronization: Carrier, Timing and Frame   (Chapter 10)

Run it:      python labs/lab04_synchronization.py
Self-test:   python labs/lab04_synchronization.py --selftest

Before a coherent receiver can decide a single bit it must find four unknowns: the carrier
frequency, the carrier phase, the symbol timing and the start of the frame. Nine experiments
build each piece and watch it work: diagnose impairments from the constellation, step a
phase-locked loop and make it slip, compare phase detectors, watch a Costas loop and a
timing loop lock live, race frequency estimators against the Cramér–Rao bound, tune a Farrow
interpolator, set a preamble threshold, and search for an LTE/NR cell under frequency offset.
Library code: commlib/sync.py and commlib/synckit.py.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as ss
from scipy import stats

import commlib as cl
from commlib import synckit as sk
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

QPSK = cl.get_constellation("qpsk")


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


@lru_cache(maxsize=16)
def mf_waveform(beta, sps, nsym, seed, span=16):
    """QPSK symbols and their matched-filter output (RRC pair); symbol k peaks at d0 + k·sps."""
    rng = np.random.default_rng(seed)
    a = QPSK.points[rng.integers(0, 4, nsym)]
    h = cl.rrc_taps(beta, sps, span)
    x = ss.fftconvolve(cl.shape(a, h, sps), h)
    return a, x, len(h) - 1


def wn_from_bn(bn, zeta):
    """Natural frequency (rad per update) of the digital loop with noise bandwidth BnT."""
    return 2 * bn / (zeta + 1 / (4 * zeta))


# =============================================================================== 1. diagnosis
KINDS = ["Phase offset", "Frequency offset", "Timing offset", "Clock offset"]


class Diagnosis(Experiment):
    title = "What each error looks like"
    blurb = "Four synchronization errors, four signatures. Learn them, then diagnose a mystery."
    book = "sec:ch10:problem"
    controls = [
        Heading("Impairments"),
        Slider("phase", "Carrier phase offset", -180.0, 180.0, 0.0, step=1, unit="°"),
        Slider("cfo", "Carrier frequency offset", 0.0, 20.0, 0.0, step=0.1, unit="× 10⁻⁴ Rs"),
        Slider("tau", "Timing offset", -0.5, 0.5, 0.0, step=0.01, unit="T"),
        Slider("ppm", "Sample-clock offset", 0.0, 500.0, 0.0, step=5, unit="ppm"),
        Slider("esn0", "Es/N0", 5.0, 40.0, 25.0, step=0.5, unit="dB"),
        Heading("Play doctor"),
        Button("mystery", "New mystery receiver", primary=True),
        Choice("guess", "Your diagnosis", ["—"] + KINDS, "—", style="menu"),
        Button("mine", "Back to my own settings"),
    ]
    plots = [
        ConstellationPlot("const", "Received symbols (light: first half, dark: second half)",
                          lim=1.6),
        Plot("ph", "Phase of each symbol relative to what was sent", x="symbol index",
             y="phase error (degrees)", xlim=(0, 2000), ylim=(-200, 200), legend=None),
        Plot("amp", "Amplitude of each symbol", x="symbol index", y="|received| / |sent|",
             xlim=(0, 2000), ylim=(0, 1.8), legend=None),
    ]
    layout = [["const", "ph"], ["const", "amp"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("evm", "EVM", "dB", ".1f", good=lambda x: x < -20),
        Readout("slope", "Phase drift", "°/1000 sym", ".1f"),
        Readout("mode", "Showing", "", None),
        Readout("score", "Correct diagnoses", "", "int", good=lambda x: x >= 3),
    ]
    challenges = [
        Challenge("Turn the four points into a ring using one impairment only.",
                  lambda s: s.p.cfo >= 5 and s.p.phase == 0 and s.p.tau == 0 and s.p.ppm == 0
                  and s.exp.mystery is None),
        Challenge("Make four clouds of ISI with a timing error alone, EVM worse than −10 dB.",
                  lambda s: s.exp.mystery is None and s.p.cfo == 0 and s.p.ppm == 0
                  and s.r.evm > -10 and abs(s.p.tau) > 0),
        Challenge("Diagnose three mystery receivers correctly.",
                  lambda s: s.exp.score >= 3,
                  hint="Press 'New mystery receiver', study the plots, then pick a diagnosis."),
    ]
    sps = 8
    nsym = 2000

    def setup(self):
        self.mystery = None
        self.scored = False
        self.score = 0

    def on_mystery(self, p):
        rng = self.rng
        kind = KINDS[int(rng.integers(0, 4))]
        val = {"Phase offset": rng.choice([-1, 1]) * rng.uniform(25, 70),
               "Frequency offset": rng.uniform(3, 12),
               "Timing offset": rng.choice([-1, 1]) * rng.uniform(0.25, 0.45),
               "Clock offset": rng.uniform(150, 400)}[kind]
        self.mystery = (kind, float(val))
        self.scored = False
        self.set_control("guess", "—")

    def update(self, p):
        if self.mystery is not None:
            kind, val = self.mystery
            ph = val if kind == "Phase offset" else 0.0
            cfo = val if kind == "Frequency offset" else 0.0
            tau = val if kind == "Timing offset" else 0.0
            ppm = val if kind == "Clock offset" else 0.0
            esn0 = 25.0
        else:
            ph, cfo, tau, ppm, esn0 = p.phase, p.cfo, p.tau, p.ppm, p.esn0
        a, x, d0 = mf_waveform(0.35, self.sps, self.nsym + 60, 1)
        k = np.arange(self.nsym)
        pos = d0 + tau * self.sps + k * self.sps * (1 + ppm * 1e-6)
        z = cl.interp_cubic(x, pos)
        z = z * np.exp(1j * (np.deg2rad(ph) + 2 * np.pi * cfo * 1e-4 * k))
        rng = np.random.default_rng(5)
        z, _ = cl.awgn_esn0(z, esn0, rng=rng, es=1.0)
        s = a[:self.nsym]
        pc = self.plot("const")
        h = self.nsym // 2
        pc.points("a", z[:h], color=GRAY, size=3, alpha=0.4)
        pc.points("b", z[h:], color=NAVY, size=3, alpha=0.5)
        pc.ideal("i", QPSK.points, color=RED)
        err = np.rad2deg(np.angle(z * np.conj(s)))
        pp = self.plot("ph")
        pp.scatter("e", k, err, color=NAVY, size=3, alpha=0.4)
        pp.hline("z", 0, color=GRAY, style="-", width=0.7)
        pa = self.plot("amp")
        pa.scatter("m", k, np.abs(z) / np.abs(s), color=PURPLE, size=3, alpha=0.4)
        pa.hline("one", 1, color=GRAY, style="-", width=0.7)
        unw = np.rad2deg(np.unwrap(np.angle(z * np.conj(s))))
        slope = np.polyfit(k, unw, 1)[0] * 1000
        evm = float(db10(np.mean(np.abs(z - s) ** 2)))
        if self.mystery is not None and p.guess != "—" and not self.scored:
            self.scored = True
            if p.guess == self.mystery[0]:
                self.score += 1
                self.status("Correct diagnosis!")
            else:
                self.status(f"Not quite: it was a {self.mystery[0].lower()}.")
        mode = "your settings" if self.mystery is None else (
            "mystery" if not self.scored else f"mystery: {self.mystery[0].lower()}")
        self.readout(evm=evm, slope=slope, mode=mode, score=self.score)

    def story(self, p):
        s = ("<p>Four pictures every receiver engineer knows by heart:</p>"
             "<p><b>Phase offset</b>: the constellation is rotated but sharp, and the phase plot "
             "(top right) is a flat line away from zero. <b>Frequency offset</b>: the rotation "
             "grows with time, so the points smear into a ring and the phase plot is a ramp. "
             "<b>Timing offset</b>: sampling off the eye's centre lets neighbours leak in: four "
             "clouds of ISI, phase and amplitude both noisy. <b>Clock offset</b>: the timing "
             "error grows with time, so the first half (light) is sharp and the second (dark) "
             "blurred, and the amplitude plot breathes.</p>")
        if self.mystery is not None:
            s += ("<p>A mystery receiver is on screen: your sliders are ignored until you press "
                  "'Back to my own settings'. Pick your diagnosis in the menu.</p>")
        return "<h3>Reading a constellation</h3>" + s + keybox(
            "Rotation = phase, ring = frequency, clouds = timing, growing clouds = clock.")

    def on_mine(self, p):
        self.mystery = None


# =============================================================================== 2. PLL
class PLLLoop(Experiment):
    title = "Inside the PLL"
    blurb = "Damping, bandwidth, steps and slips: the second-order loop in the time and frequency domains."
    book = "sec:ch10:pll"
    controls = [
        Slider("zeta", "Damping ζ", 0.2, 3.0, 0.707, step=0.01),
        LogSlider("bn", "Loop noise bandwidth BnT", 0.003, 0.1, 0.02,
                  help="Noise bandwidth relative to the update (symbol) rate"),
        Choice("step", "Disturbance", ["Phase step", "Frequency step"], "Phase step"),
        Slider("dph", "Phase step", 0.0, 180.0, 60.0, step=1, unit="°",
               enabled_if=lambda p: p.step == "Phase step"),
        Slider("df", "Frequency step", 0.0, 40.0, 5.0, step=0.1, unit="× 10⁻³ cyc/sym",
               enabled_if=lambda p: p.step == "Frequency step"),
        Slider("esn0", "Es/N0", 0.0, 40.0, 40.0, step=0.5, unit="dB"),
    ]
    plots = [
        Plot("err", "Phase error after the step (red: the real loop, dashed: linear theory)",
             x="symbols since the step", y="phase error (degrees)", xlim=(0, 600), legend="tr"),
        Plot("bode", "Closed loop |H| (solid) and error |1 − H| (dotted)", x="ω / ωn",
             y="gain (dB)", logx=True, xlim=(0.01, 30), ylim=(-40, 10), legend="bl"),
    ]
    layout = [["err"], ["bode"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("over", "Overshoot", "%", ".0f"),
        Readout("settle", "Settling (to 10 %)", "symbols", ".0f"),
        Readout("lock", "Lock-in range 2ζωn", "× 10⁻³ cyc/sym", ".1f"),
        Readout("slips", "Cycle slips", "", "int", good=lambda x: x == 0),
    ]
    challenges = [
        Challenge("Make the loop ring: more than 40 % overshoot on a phase step.",
                  lambda s: s.p.step == "Phase step" and s.r.over > 40),
        Challenge("Apply a frequency step bigger than the lock-in range and watch the loop slip "
                  "cycles before it locks.",
                  lambda s: s.p.step == "Frequency step" and s.r.slips >= 1 and s.exp.locked),
        Challenge("Settle a phase step (to 10 %) within 30 symbols with BnT no larger than 0.03.",
                  lambda s: s.p.step == "Phase step" and s.r.settle <= 30 and s.p.bn <= 0.03,
                  hint="Bandwidth is capped, so try the other knob."),
    ]
    N = 1500

    def setup(self):
        self.locked = False

    def update(self, p):
        wn = wn_from_bn(p.bn, p.zeta)
        n = np.arange(self.N)
        if p.step == "Phase step":
            theta = np.where(n >= 0, np.deg2rad(p.dph), 0.0)
        else:
            theta = 2 * np.pi * p.df * 1e-3 * n
        rng = np.random.default_rng(3)
        y = np.exp(1j * theta)
        n0 = 10 ** (-p.esn0 / 10)
        y = y + np.sqrt(n0 / 2) * (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N))
        loop = sk.CarrierLoop(p.bn, p.zeta, detector="sin")
        _, est = loop.run(y)
        est = np.r_[0.0, est[:-1]]                       # estimate used for sample n
        err = theta - est
        errd = np.rad2deg(err)
        if p.step == "Phase step":
            lin = np.rad2deg(np.deg2rad(p.dph) * sk.pll_step_response(n, p.zeta, "phase", wn))
            big = abs(p.dph) if p.dph else 1.0
        else:
            lin = np.rad2deg(2 * np.pi * p.df * 1e-3 * sk.pll_step_response(n, p.zeta, "freq", wn))
            big = max(np.max(np.abs(lin)), 1e-6)
        pe = self.plot("err")
        top = max(np.max(np.abs(errd[:600])), np.max(np.abs(lin[:600])), 5)
        pe.set_ylim(-1.15 * top, 1.25 * top)
        pe.hline("z", 0, color=GRAY, style="-", width=0.7)
        for kslip in (-2, -1, 1, 2):
            if abs(360 * kslip) < 1.2 * top:
                pe.hline(f"s{kslip}", 360 * kslip, color=ORANGE, style=":", width=1.0)
        pe.line("lin", n, lin, color=NAVY, width=1.6, style="--", name="linear model")
        pe.line("err", n, errd, color=RED, width=2.0, name="simulated loop")
        # metrics (on the simulated loop)
        final = errd[-200:].mean()
        slips = int(round(final / 360.0))
        resid = errd - 360 * slips
        if p.step == "Phase step":
            over = max(0.0, -np.min(resid[:600]) / big * 100) if p.dph else 0.0
        else:
            peak = np.max(np.abs(resid[:600]))
            over = 0.0 if peak == 0 else max(0.0, -np.min(resid[:600]) / max(peak, 1e-9) * 100)
        tol = 0.1 * big
        outside = np.where(np.abs(resid) > tol)[0]
        settle = float(outside[-1] + 1) if len(outside) else 0.0
        self.locked = bool(np.std(resid[-200:]) < 20 and abs(np.mean(resid[-200:])) < 20)
        pb = self.plot("bode")
        w = np.logspace(-2, np.log10(30), 300)
        H, E = sk.pll_closed_loop(w, p.zeta)
        pb.line("H", w, 20 * np.log10(np.abs(H)), color=NAVY, width=2.2, name="|H|: tracks")
        pb.line("E", w, 20 * np.log10(np.abs(E) + 1e-12), color=RED, width=2.0, style=":",
                name="|1 − H|: error")
        pb.hline("z", 0, color=GRAY, style="-", width=0.7)
        self.readout(over=over, settle=settle,
                     lock=2 * p.zeta * wn / (2 * np.pi) * 1e3, slips=abs(slips))

    def story(self, p):
        wn = wn_from_bn(p.bn, p.zeta)
        over, sl = self.r.get("over", 0), self.r.get("slips", 0)
        s = (f"<p>A second-order loop has two knobs. The noise bandwidth BnT = {v(p.bn, '.3f')} sets "
             f"how fast it reacts (ωn = {v(wn, '.3f')} rad per symbol); the damping ζ = "
             f"{v(p.zeta)} sets how it settles. ζ ≈ 0.707 is the classic compromise: a small "
             f"overshoot ({v(over, '.0f', '%')} here) and little noise peaking in |H| (bottom).</p>")
        if p.step == "Phase step":
            s += ("<p>After a phase step the integrator is not needed: the error decays to zero, "
                  "ringing if the loop is underdamped. The dashed curve is the linear model; the "
                  "red one is a real loop with a sinusoidal phase detector, close to it until the "
                  "step approaches 180°.</p>")
        else:
            s += ("<p>After a frequency step the error rises, then the integrator charges up to "
                  "the new frequency and pulls the error back to zero: a type-2 loop tracks a "
                  "frequency offset with no standing phase error. ")
            if sl:
                s += (bad(f"Here the step exceeds the lock-in range, so the loop slipped {sl} "
                          f"cycle{'s' if sl > 1 else ''}") + " (orange lines are ±360°) before pulling "
                      "in. In a coherent receiver every slip rotates all later symbols.</p>")
            else:
                s += "Inside the lock-in range it locks without slipping.</p>"
        return "<h3>The phase-locked loop</h3>" + s + keybox(
            "Lock-in ≈ 2ζωn. Beyond it a type-2 loop still pulls in, slowly (T_p ∝ Δω²/(ζωn³)), "
            "slipping cycles on the way.")


# =============================================================================== 3. detectors
DETS = {"Data-aided  Im{z·a*}": "da", "Decision-directed  arg(z·â*)": "dd",
        "BPSK Costas  2·I·Q": "costas2", "QPSK Costas (hard-limited)": "costas4",
        "M-th power  Im{z^M}/M": "mpower"}


class PhaseDetectors(Experiment):
    title = "Phase detector S-curves"
    blurb = "What each detector reports for a given phase error, and how many places it can lock."
    book = "sec:ch10:carrier"
    controls = [
        Choice("det", "Phase detector", list(DETS), "Decision-directed  arg(z·â*)", style="menu"),
        Choice("mod", "Modulation", ["BPSK", "QPSK", "8-PSK"], "QPSK"),
        Slider("esn0", "Es/N0", -5.0, 30.0, 10.0, step=0.5, unit="dB"),
        Slider("phi", "Phase error to inspect", -180.0, 180.0, 30.0, step=1, unit="°"),
    ]
    plots = [
        Plot("s", "S-curve: mean detector output against phase error", x="phase error (degrees)",
             y="mean output", xlim=(-180, 180), legend="tl"),
        ConstellationPlot("const", "Symbols at that phase error", lim=1.7),
    ]
    layout = [["s", "const"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("kd", "Slope at zero (Kd)", "/ rad", ".2f"),
        Readout("rel", "Slope vs high SNR", "%", ".0f"),
        Readout("locks", "Stable lock points", "", "int"),
        Readout("out", "Output at your phase", "", ".3f"),
    ]
    challenges = [
        Challenge("Find a detector with no phase ambiguity at all (one stable lock point).",
                  lambda s: s.r.locks == 1),
        Challenge("Watch noise flatten a decision-directed S-curve: slope below half its "
                  "high-SNR value.",
                  lambda s: s.p.det.startswith("Decision") and s.r.rel < 50),
        Challenge("Show the 8-fold ambiguity of a blind 8-PSK loop.",
                  lambda s: s.p.mod == "8-PSK" and s.r.locks == 8),
    ]
    nsym = 3000

    @staticmethod
    @lru_cache(maxsize=64)
    def curve(det, mod, esn0):
        M = {"BPSK": 2, "QPSK": 4, "8-PSK": 8}[mod]
        c = cl.get_constellation({"BPSK": "bpsk", "QPSK": "qpsk", "8-PSK": "8psk"}[mod])
        rng = np.random.default_rng(4)
        a = c.points[rng.integers(0, c.M, 2000)]
        w = (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)) / np.sqrt(2)
        nz = np.sqrt(10 ** (-esn0 / 10)) * w
        phis = np.deg2rad(np.linspace(-180, 180, 181))
        z = a[None, :] * np.exp(1j * phis[:, None]) + nz[None, :]
        out = sk.phase_detector(z.ravel(), det, np.tile(a, len(phis)), M).reshape(z.shape)
        return np.rad2deg(phis), out.mean(axis=1)

    def update(self, p):
        det = DETS[p.det]
        phis, S = self.curve(det, p.mod, round(p.esn0, 1))
        _, Shi = self.curve(det, p.mod, 40.0)
        ps = self.plot("s")
        top = max(np.max(np.abs(S)), np.max(np.abs(Shi)), 0.05)
        ps.set_ylim(-1.3 * top, 1.5 * top)
        ps.hline("z", 0, color=GRAY, style="-", width=0.7)
        ps.line("hi", phis, Shi, color=GRAY, width=1.2, style="--", name="at 40 dB")
        ps.line("S", phis, S, color=NAVY, width=2.4, name=f"at {p.esn0:.1f} dB")
        # stable lock points: zero crossings with positive slope (restoring force)
        sgn = np.sign(S)
        idx = np.where((sgn[:-1] <= 0) & (sgn[1:] > 0) & (np.diff(S) > 0))[0]
        locks = []
        for i in idx:
            x0 = np.interp(0, [S[i], S[i + 1]], [phis[i], phis[i + 1]])
            if not locks or abs(x0 - locks[-1]) > 5:
                locks.append(x0)
        if len(locks) > 1 and abs(locks[0] + 360 - locks[-1]) < 5:
            locks = locks[:-1]
        ps.scatter("locks", locks, np.zeros(len(locks)), color=GREEN, size=13, name="stable lock points")
        out = float(np.interp(p.phi, phis, S))
        ps.vline("phi", p.phi, color=RED, style="--", width=1.2)
        ps.scatter("you", [p.phi], [out], color=RED, size=11)
        i0 = np.argmin(np.abs(phis))
        kd = (S[i0 + 1] - S[i0 - 1]) / np.deg2rad(phis[i0 + 1] - phis[i0 - 1])
        kdh = (Shi[i0 + 1] - Shi[i0 - 1]) / np.deg2rad(phis[i0 + 1] - phis[i0 - 1])
        c = cl.get_constellation({"BPSK": "bpsk", "QPSK": "qpsk", "8-PSK": "8psk"}[p.mod])
        rng = np.random.default_rng(9)
        a = c.points[rng.integers(0, c.M, 600)]
        z, _ = cl.awgn_esn0(a * np.exp(1j * np.deg2rad(p.phi)), p.esn0, rng=rng, es=1.0)
        pc = self.plot("const")
        pc.points("z", z, color=NAVY, size=3, alpha=0.45)
        pc.ideal("i", c.points, color=RED)
        self.readout(kd=kd, rel=100 * kd / kdh if abs(kdh) > 1e-9 else 0.0, locks=len(locks), out=out)

    def story(self, p):
        L = self.r.get("locks", 1)
        rel = self.r.get("rel", 100)
        s = ("<p>A phase detector turns a phase error into a number; the loop pushes the estimate "
             "the way that number says. Averaged over data and noise, the output against the "
             "error is the <b>S-curve</b>. Where it crosses zero going upwards (green), the loop "
             "is pulled back from both sides: a stable lock point.</p>")
        if L == 1:
            s += ("<p>One lock point: known data remove the modulation completely, so there is "
                  "no ambiguity. That is the power of a preamble or pilots.</p>")
        else:
            s += (f"<p>{v(L, 'd')} lock points, {v(360 / L, '.0f', '°')} apart: a blind detector "
                  f"cannot tell the constellation from its rotated copy, so the loop may lock "
                  f"{360 / L:.0f}° off. A known preamble or differential encoding must resolve it.</p>")
        if rel < 80:
            s += (f"<p>Noise also flattens the curve: the slope at zero is only {v(rel, '.0f', '%')} "
                  "of its high-SNR value, which lowers the loop gain and bandwidth unless the "
                  "designer compensates.</p>")
        return "<h3>What the loop sees</h3>" + s + keybox(
            "Decision-directed and Costas detectors have period 2π/M: an M-fold ambiguity.")


# =============================================================================== 4. Costas
class CostasLoop(Experiment):
    title = "A Costas loop locks on"
    blurb = "Watch a carrier loop pull a spinning QPSK ring into four tight clusters."
    book = "sec:ch10:carrier"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Choice("loop", "Phase detector", ["Costas (hard-limited)", "Decision-directed"]),
        Slider("cfo", "Carrier frequency offset", 0.0, 30.0, 8.0, step=0.1, unit="× 10⁻³ cyc/sym"),
        LogSlider("bn", "Loop bandwidth BnT", 0.002, 0.1, 0.02),
        Slider("esn0", "Es/N0", 0.0, 30.0, 15.0, step=0.5, unit="dB"),
        Button("restart", "Restart the loop", primary=True),
    ]
    plots = [
        ConstellationPlot("const", "After the loop (grey: older, navy: latest)", lim=1.6),
        Plot("perr", "Phase error (modulo 90°)", x="symbol", y="phase error (degrees)",
             ylim=(-50, 50), legend=None),
        Plot("freq", "The loop's frequency estimate (NCO)", x="symbol",
             y="frequency (× 10⁻³ cyc/sym)", legend="br"),
    ]
    layout = [["const", "perr"], ["const", "freq"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("n", "Symbols processed", "", "int"),
        Readout("f", "NCO frequency", "× 10⁻³ cyc/sym", ".2f"),
        Readout("jit", "Residual jitter (rms)", "°", ".1f", good=lambda x: x < 5),
        Readout("state", "State", "", None),
    ]
    challenges = [
        Challenge("Lock onto an offset of at least 0.02 cycles per symbol.",
                  lambda s: s.p.cfo >= 20 and s.r.state == "locked"),
        Challenge("Hold the residual jitter below 3° at Es/N0 = 12 dB or less.",
                  lambda s: s.p.esn0 <= 12 and s.r.state == "locked" and s.r.jit < 3,
                  hint="Jitter variance ≈ BnT / (Es/N0): narrow the loop, but not so far it cannot pull in."),
        Challenge("Make the loop fail: still not locked after 3000 symbols.",
                  lambda s: s.r.n >= 3000 and s.r.state != "locked"),
    ]
    chunk = 120
    hist = 1500

    def setup(self):
        self.reset_loop(None)

    def reset_loop(self, p):
        self.loop = None
        self.n = 0
        self.z = np.zeros(0, complex)
        self.err = np.zeros(0)
        self.fr = np.zeros(0)
        self.k = np.zeros(0)

    def on_restart(self, p):
        self.reset_loop(p)

    def step(self, p):
        if self.loop is None:
            self.loop = sk.CarrierLoop(p.bn, detector="costas" if p.loop.startswith("Costas") else "dd",
                                       M=4, kd=1.0)
            self.rngd = np.random.default_rng(17)
        n0, N = self.n, self.chunk
        a = QPSK.points[self.rngd.integers(0, 4, N)]
        k = np.arange(n0, n0 + N)
        true = 2 * np.pi * p.cfo * 1e-3 * k + 0.7
        y, _ = cl.awgn_esn0(a * np.exp(1j * true), p.esn0, rng=self.rngd, es=1.0)
        z, ph = self.loop.run(y)
        est = np.r_[ph[0] if n0 == 0 else self.last_ph, ph[:-1]]
        self.last_ph = ph[-1]
        e = np.rad2deg(sk.wrap(true - est, np.pi / 2))
        self.n += N
        keep = lambda arr, new: np.r_[arr, new][-self.hist:]
        self.z = keep(self.z, z)
        self.err = keep(self.err, e)
        self.fr = keep(self.fr, np.r_[np.diff(np.r_[est, ph[-1]])] / (2 * np.pi) * 1e3)
        self.k = keep(self.k, k)

    def draw(self, p):
        pc = self.plot("const")
        z = self.z
        pc.points("old", z[-600:-150], color=GRAY, size=3, alpha=0.35)
        pc.points("new", z[-150:], color=NAVY, size=4, alpha=0.7)
        pc.ideal("i", QPSK.points, color=RED)
        pe = self.plot("perr")
        pe.scatter("e", self.k, self.err, color=NAVY, size=2, alpha=0.5)
        pe.hline("z", 0, color=GRAY, style="-", width=0.7)
        pe.set_xlim(self.k[0], self.k[0] + self.hist)
        pf = self.plot("freq")
        sm = np.convolve(self.fr, np.ones(25) / 25, mode="valid")
        pf.line("f", self.k[12:12 + len(sm)], sm, color=NAVY, width=2.0, name="NCO (25-symbol average)")
        pf.hline("true", p.cfo, color=RED, style="--", label="true offset", label_pos=0.85)
        pf.set_xlim(self.k[0], self.k[0] + self.hist)
        pf.set_ylim(min(-2, p.cfo - 15), p.cfo + 15)
        recent = self.err[-300:]
        jit = float(np.std(recent))
        fnow = float(np.mean(self.fr[-200:]))
        locked = abs(fnow - p.cfo) < max(0.5, 0.05 * p.cfo) and jit < 15 and self.n >= 300
        self.readout(n=self.n, f=fnow, jit=jit, state="locked" if locked else "acquiring")

    def update(self, p):
        self.reset_loop(p)
        for _ in range(3):
            self.step(p)
        self.draw(p)

    def tick(self, p):
        if self.n < 30000:
            self.step(p)
        self.draw(p)

    def story(self, p):
        state, jit = self.r.get("state", "acquiring"), self.r.get("jit", 0)
        bnt = p.bn
        th = np.rad2deg(np.sqrt(bnt / 10 ** (p.esn0 / 10)))
        s = (f"<p>The receiver's oscillator is off by {v(p.cfo / 1000, '.3f')} cycles per symbol, "
             "so without correction the QPSK points spin round in a ring. The loop measures the "
             "phase error of every symbol, filters it, and steers a numerically controlled "
             "oscillator (bottom right) until the spin stops.</p>")
        if state == "locked":
            s += (f"<p>{good('Locked.')} The NCO has found the offset and the clusters are tight: "
                  f"{v(jit, '.1f', '°')} rms of residual jitter, against roughly "
                  f"{v(th, '.1f', '°')} predicted by σ² ≈ BnT/(Es/N0).</p>")
        else:
            s += ("<p>Still acquiring: the phase error (top right) is sweeping through its ±45° "
                  "range. A wider loop pulls in faster but lets in more noise; a narrow one may "
                  "never get there from a large offset.</p>")
        return "<h3>Carrier recovery, live</h3>" + s + keybox(
            "Restart the loop after changing a setting to watch acquisition again. Any lock point "
            "is 90° ambiguous: the frame sync must fix it.")


# =============================================================================== 5. freq estimation
EST = ["Periodogram (ML)", "Kay", "Fitz", "Luise–Reggiannini", "4th power (blind)"]
EST_COL = [NAVY, RED, GREEN, ORANGE, PURPLE]


def run_estimator(name, y, a):
    z = y * np.conj(a)
    N = len(z)
    if name.startswith("Periodogram"):
        return sk.est_periodogram(z)
    if name == "Kay":
        return sk.est_kay(z)
    if name == "Fitz":
        return sk.est_fitz(z, N // 2)
    if name.startswith("Luise"):
        return sk.est_lr(z, N // 2)
    return sk.est_mpower(y, 4)


class FrequencyEstimation(Experiment):
    title = "Frequency estimators vs the bound"
    blurb = "Five estimators race the Cramér–Rao bound, and each one hits a wall somewhere."
    book = "sec:ch10:freq"
    heavy = True
    controls = [
        Choice("est", "Highlight", EST, "Kay", style="menu"),
        IntSlider("N", "Preamble length N", 16, 256, 64, step=8, unit="symbols"),
        Slider("esn0", "Es/N0 (single trial)", -5.0, 30.0, 10.0, step=0.5, unit="dB"),
        Slider("nu", "True offset", -0.25, 0.25, 0.01, step=0.001, unit="cyc/sym"),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run the MSE sweep again", primary=True),
    ]
    plots = [
        BERPlot("mse", "Mean-square error against Es/N0 (dashed: Cramér–Rao bound)",
                x="Es/N0 (dB)", y="MSE ((cyc/sym)²)", ylim=(1e-11, 1e-1), xlim=(-6, 31), legend="tr"),
        Plot("range", "Estimate against the true offset (single trials)", x="true offset (cyc/sym)",
             y="estimate (cyc/sym)", xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), legend="tl"),
    ]
    layout = [["mse", "range"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("err", "Error of this trial", "× 10⁻³ cyc/sym", ".3f"),
        Readout("crb", "CRB (rms) at this Es/N0", "× 10⁻³ cyc/sym", ".3f"),
        Readout("rng", "Unambiguous range", "± cyc/sym", ".3f"),
        Readout("pts", "Sweep", "", None),
    ]
    challenges = [
        Challenge("Choose a preamble long enough that the bound is below 10⁻⁴ cycles/symbol rms "
                  "at Es/N0 = 10 dB.",
                  lambda s: abs(s.p.esn0 - 10) < 0.3 and s.r.crb < 0.1),
        Challenge("Break the blind estimator: put the true offset outside its ±1/8 range.",
                  lambda s: s.p.est.startswith("4th") and abs(s.p.nu) > 0.125 and abs(s.r.err) > 50),
        Challenge("Break Fitz: an offset beyond ±1/N wraps its answer.",
                  lambda s: s.p.est == "Fitz" and abs(s.p.nu) > 1 / s.p.N and abs(s.r.err) > 5),
    ]

    def setup(self):
        self.run_id = 0
        self.sim = {}

    def on_rerun(self, p):
        self.run_id += 1

    def ranges(self, name, N):
        return {"Periodogram (ML)": 0.5, "Kay": 0.5, "Fitz": 1 / N,
                "Luise–Reggiannini": 1 / (N // 2 + 1), "4th power (blind)": 0.125}[name]

    def update(self, p):
        N = p.N
        rng = np.random.default_rng(7)
        a = QPSK.points[rng.integers(0, 4, N)]
        nn = np.arange(N)
        noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        y = a * np.exp(1j * (2 * np.pi * p.nu * nn + 0.4)) + np.sqrt(10 ** (-p.esn0 / 10)) * noise
        est = run_estimator(p.est, y, a)
        # estimate against true offset
        pr = self.plot("range")
        tv = np.linspace(-0.5, 0.5, 81)
        pr.line("ideal", tv, tv, color=GRAY, style=":", width=1.0)
        r = self.ranges(p.est, N)
        pr.band("ok", -r, r, color=GREEN, alpha=0.08)
        idx = EST.index(p.est)
        outs = []
        for t in tv:
            yy = a * np.exp(1j * (2 * np.pi * t * nn + 0.4)) + np.sqrt(10 ** (-p.esn0 / 10)) * noise
            outs.append(run_estimator(p.est, yy, a))
        pr.scatter("est", tv, outs, color=EST_COL[idx], size=6, name=p.est)
        pr.scatter("you", [p.nu], [est], color=RED, size=13)
        pm = self.plot("mse")
        es = np.linspace(-6, 31, 75)
        pm.theory("crb", es, sk.crb_freq(N, es), color=GRAY, style="--", name=f"CRB, N = {N}")
        pm.vline("cur", p.esn0, color=RED, style=":", width=1.0)
        self.sim = {k: ([], []) for k in EST}
        crb = float(np.sqrt(sk.crb_freq(N, p.esn0)))
        self.readout(err=(est - p.nu) * 1e3, crb=crb * 1e3, rng=r, pts="0")

    def background(self, p):
        N = p.N
        rng = np.random.default_rng(100 + self.run_id)
        trials = 30 if self.quick else (400 if p.effort == "Thorough" else 120)
        snrs = np.arange(-6, 31, 3.0 if not self.quick else 6.0)
        for e in snrs:
            errs = {k: [] for k in EST}
            for _ in range(trials):
                nu = rng.uniform(-0.01, 0.01)
                a = QPSK.points[rng.integers(0, 4, N)]
                nn = np.arange(N)
                y = a * np.exp(1j * (2 * np.pi * nu * nn + rng.uniform(0, 6.28)))
                y = y + np.sqrt(10 ** (-e / 10) / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
                for k in EST:
                    errs[k].append(sk.wrap(run_estimator(k, y, a) - nu, 1.0))
            yield dict(e=e, mse={k: float(np.mean(np.square(errs[k]))) for k in EST})
        yield dict(done=True)

    def progress(self, p, it):
        if it.get("done"):
            self.readout(pts=f"{len(self.sim[EST[0]][0])} ✓")
            return
        pm = self.plot("mse")
        for i, k in enumerate(EST):
            xs, ys = self.sim[k]
            xs.append(it["e"])
            ys.append(max(it["mse"][k], 1e-12))
            pm.sim(k, xs, ys, color=EST_COL[i], name=k, size=11 if k == p.est else 7)
        self.readout(pts=f"{len(self.sim[EST[0]][0])} …")

    def story(self, p):
        crb, r = self.r.get("crb", 0), self.r.get("rng", 0.5)
        s = (f"<p>A preamble of {v(p.N, 'd')} known symbols, rotating at an unknown rate. No "
             f"unbiased estimator can beat the Cramér–Rao bound: here {v(crb, '.3f')} × 10⁻³ cycles "
             "per symbol rms. It falls as N³: double the preamble and the variance drops "
             "eightfold, because a longer window both averages more noise and gives a longer "
             "lever arm to measure a phase slope.</p>"
             "<p>All five estimators touch the bound at high SNR (left). Each has a weakness: "
             "<b>Kay</b> breaks down first as SNR falls; the <b>periodogram</b> holds out longest; "
             "<b>Fitz</b> and <b>Luise–Reggiannini</b> trade range for robustness; the <b>blind "
             "4th-power</b> estimator needs no preamble but multiplies noise and wraps beyond ±1/8.</p>"
             f"<p>{p.est}'s unambiguous range is ±{v(r, '.3f')} cycles per symbol (green band, right).</p>")
        return "<h3>How well can a frequency be measured?</h3>" + s + keybox(
            "CRB(ν) = 3 / (2π² N (N² − 1) Es/N0). Nonlinear estimators have a threshold SNR "
            "below which outliers dominate.")


# =============================================================================== 6. timing
TEDS = {"Gardner": "gardner", "Mueller–Müller": "mm", "Early–late": "el"}


class TimingRecovery(Experiment):
    title = "Symbol timing recovery"
    blurb = "A timing loop slides its strobes to the eye's centre and keeps up with a drifting clock."
    book = "sec:ch10:timing"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Choice("ted", "Timing error detector", list(TEDS), "Gardner"),
        Slider("delay", "Initial timing offset", 0.0, 4.0, 1.7, step=0.05, unit="samples"),
        Slider("ppm", "Clock offset", 0.0, 2000.0, 200.0, step=10, unit="ppm"),
        LogSlider("bn", "Loop bandwidth BnT", 0.001, 0.05, 0.01),
        Slider("esn0", "Es/N0", 5.0, 40.0, 20.0, step=0.5, unit="dB"),
        Slider("beta", "RRC roll-off β", 0.1, 1.0, 0.35, step=0.05),
        Button("restart", "Restart", primary=True),
    ]
    plots = [
        Plot("scurve", "Detector S-curve", x="timing offset (T)", y="mean output (normalised)",
             xlim=(-0.5, 0.5), ylim=(-1.3, 1.3), legend=None),
        Plot("tau", "Timing error of each strobe", x="symbol", y="timing error (T)",
             ylim=(-0.55, 0.55), legend=None),
        ConstellationPlot("const", "Strobes (grey: older, navy: latest 200)", lim=1.6),
    ]
    layout = [["tau", "tau"], ["scurve", "const"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("n", "Symbols processed", "", "int"),
        Readout("rms", "Timing jitter (rms)", "% of T", ".1f", good=lambda x: x < 3),
        Readout("mer", "MER of the strobes", "dB", ".1f", good=lambda x: x > 18),
        Readout("state", "State", "", None),
    ]
    challenges = [
        Challenge("Lock the decision-directed Mueller–Müller detector at Es/N0 ≤ 10 dB with "
                  "timing jitter under 5 % of a symbol.",
                  lambda s: s.p.ted == "Mueller–Müller" and s.p.esn0 <= 10 and s.r.state == "locked"
                  and s.r.rms < 5),
        Challenge("Starve the Gardner detector: a roll-off of 0.1 and watch the jitter exceed 4 %.",
                  lambda s: s.p.ted == "Gardner" and s.p.beta <= 0.1 + 1e-6 and s.r.rms > 4),
        Challenge("Track a 2000 ppm clock offset with a narrow loop (BnT ≤ 0.004) and stay locked.",
                  lambda s: s.p.ppm >= 1999 and s.p.bn <= 0.004 and s.r.state == "locked"
                  and s.r.n > 3000,
                  hint="Too narrow and the loop cannot keep up with the drift."),
    ]
    sps = 4
    total = 8000
    chunk = 160

    def setup(self):
        self.reset_loop()

    def reset_loop(self):
        self.loop = None
        self.pos = 0
        self.zs = np.zeros(0, complex)
        self.terr = np.zeros(0)
        self.ks = np.zeros(0)
        self.count = 0

    def on_restart(self, p):
        self.reset_loop()

    @staticmethod
    @lru_cache(maxsize=8)
    def received(beta, delay, ppm, esn0, sps=4, nsym=8000):
        a, x, d0 = mf_waveform(beta, sps, nsym + 60, 3)
        eps = ppm * 1e-6
        n = np.arange(int((len(x) - 40) / (1 + eps)) - 8)
        c0 = d0 - (sps + 2)                          # the loop's first strobe (sample sps + 2) is on a peak
        t = n * (1 + eps) - delay + c0               # received sample n = x(t)
        ok = (t > 2) & (t < len(x) - 3)
        r = np.zeros(len(n), complex)
        r[ok] = cl.interp_cubic(x, t[ok])
        rng = np.random.default_rng(12)
        n0 = 10 ** (-esn0 / 10)
        r = r + np.sqrt(n0 / 2) * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))
        return r, d0, eps

    @staticmethod
    @lru_cache(maxsize=32)
    def scurve(ted, beta):
        a, x, d0 = mf_waveform(beta, 8, 1500, 4)
        taus = np.linspace(-0.5, 0.5, 51)
        S = sk.ted_scurve(x, d0, 8, taus, ted)
        return taus, S / max(np.max(np.abs(S)), 1e-12)

    def true_err(self, times, d0, eps, delay):
        # sample n holds x(n(1+eps) - delay + c0); symbols peak at x(d0 + k sps)
        u = (times * (1 + eps) - delay - (self.sps + 2)) / self.sps
        return sk.wrap(u, 1.0)

    def step(self, p):
        r, d0, eps = self.received(round(p.beta, 2), round(p.delay, 2), round(p.ppm), round(p.esn0, 1))
        if self.loop is None:
            self.loop = sk.TimingLoop(self.sps, p.bn, ted=TEDS[p.ted])
            self.pos = 0
        if self.pos + self.chunk * self.sps >= len(r):
            return
        seg = r[self.pos:self.pos + self.chunk * self.sps]
        self.pos += len(seg)
        z, e, taus = self.loop.push(seg)
        if len(z) == 0:
            return
        te = self.true_err(self.loop.last_times, d0, eps, p.delay)
        keep = lambda arr, new, n=2000: np.r_[arr, new][-n:]
        self.zs = keep(self.zs, z)
        self.terr = keep(self.terr, te)
        self.ks = keep(self.ks, np.arange(self.count, self.count + len(z)))
        self.count += len(z)

    def draw(self, p):
        taus, S = self.scurve(TEDS[p.ted], round(p.beta, 2))
        ps = self.plot("scurve")
        ps.hline("z", 0, color=GRAY, style="-", width=0.7)
        ps.vline("v", 0, color=GRAY, style=":", width=0.7)
        ps.line("S", taus, S, color=NAVY, width=2.4)
        pt = self.plot("tau")
        if len(self.ks):
            pt.scatter("e", self.ks, self.terr, color=NAVY, size=3, alpha=0.6)
            pt.set_xlim(self.ks[0], max(self.ks[0] + 2000, self.ks[-1]))
        pt.hline("z", 0, color=GREEN, style="--", width=1.2, label="eye centre", label_pos=0.02)
        z = self.zs
        pc = self.plot("const")
        if len(z) > 200:
            pc.points("old", z[-800:-200], color=GRAY, size=3, alpha=0.35)
        pc.points("new", z[-200:], color=NAVY, size=4, alpha=0.7)
        pc.ideal("i", QPSK.points, color=RED)
        rec = self.terr[-300:] if len(self.terr) else np.zeros(1)
        rms = 100 * float(np.sqrt(np.mean(rec ** 2)))
        zz = z[-300:] if len(z) else np.zeros(1, complex)
        mer = float(-db10(np.mean(np.abs(zz - QPSK.decide(zz)) ** 2))) if len(z) else 0.0
        locked = self.count > 300 and rms < 10 and abs(np.mean(rec)) < 0.08
        self.readout(n=self.count, rms=rms, mer=mer, state="locked" if locked else "acquiring")

    def update(self, p):
        self.reset_loop()
        for _ in range(2):
            self.step(p)
        self.draw(p)

    def tick(self, p):
        self.step(p)
        self.draw(p)

    def story(self, p):
        st_, rms = self.r.get("state", "acquiring"), self.r.get("rms", 0)
        s = ("<p>The ADC runs on its own clock; the symbols arrive on the transmitter's. The loop "
             "decides where each strobe goes, interpolating between samples (Farrow), and nudges "
             "the next strobe earlier or later according to its timing-error detector. The top "
             "plot is the error of each strobe from the eye's true centre.</p>")
        if p.ted == "Gardner":
            s += ("<p><b>Gardner</b> looks at the sample halfway between symbols: when a transition "
                  "happens it should be zero, and its sign says early or late. It needs two "
                  "samples per symbol but no carrier phase. Its S-curve shrinks with the roll-off: "
                  "small β leaves it little excess bandwidth to work with.</p>")
        elif p.ted == "Mueller–Müller":
            s += ("<p><b>Mueller–Müller</b> uses one sample per symbol and the decisions: "
                  "Re{â*ₖ₋₁yₖ − â*ₖyₖ₋₁}. Cheap, the favourite of SerDes and cable modems, but it "
                  "needs decent decisions, so it struggles at low SNR or before carrier lock.</p>")
        else:
            s += ("<p><b>Early–late</b> compares the power a quarter-symbol early and late: at the "
                  "peak they balance. Simple and robust, the classic analog approach.</p>")
        if p.ppm > 0:
            s += (f"<p>A {v(p.ppm, '.0f', 'ppm')} clock offset makes the true timing drift; the "
                  "loop's integrator learns the drift rate so the error stays near zero.</p>")
        s += f"<p>Status: {good('locked') if st_ == 'locked' else bad('acquiring')}, jitter {v(rms, '.1f', '%')} of a symbol.</p>"
        return "<h3>Finding the eye's centre</h3>" + s + keybox(
            "Press Restart after changing a setting to watch acquisition from scratch.")


# =============================================================================== 7. interpolators
INTERP = {"Linear": "linear", "Piecewise parabolic (Farrow)": "parabolic",
          "Cubic Lagrange (Farrow)": "cubic"}


@lru_cache(maxsize=32)
def interp_reference(beta, sps, nsym=600):
    """Matched-filter output at sps and at 32× sps (the exact waveform)."""
    up = 32
    rng = np.random.default_rng(8)
    a = QPSK.points[rng.integers(0, 4, nsym)]
    h = cl.rrc_taps(beta, sps * up, 16)
    xf = ss.fftconvolve(cl.shape(a, h, sps * up), h)
    xf = xf / np.max(np.abs(xf))
    return xf, up


def interp_mer(beta, sps, mu, kind):
    xf, up = interp_reference(beta, sps)
    xs = xf[::up]                                   # samples at sps per symbol
    n = np.arange(200, len(xs) - 200)
    mu = round(mu * up) / up                        # evaluate exactly on the fine grid
    h = sk.interp_taps(mu, kind)
    est = h[0] * xs[n - 1] + h[1] * xs[n] + h[2] * xs[n + 1] + h[3] * xs[n + 2]
    exact = xf[(n * up + int(round(mu * up)))]
    return float(-db10(np.mean(np.abs(est - exact) ** 2) / np.mean(np.abs(exact) ** 2)))


class Interpolators(Experiment):
    title = "Interpolators and Farrow"
    blurb = "Strobes fall between samples: how good must the interpolator be?"
    book = "sec:ch10:timing"
    controls = [
        Choice("kind", "Interpolator", list(INTERP), "Cubic Lagrange (Farrow)", style="menu"),
        Slider("mu", "Fractional position μ", 0.0, 1.0, 0.5, step=0.01),
        IntSlider("sps", "Samples per symbol", 2, 8, 2),
        Slider("beta", "RRC roll-off β", 0.1, 1.0, 0.35, step=0.05),
    ]
    plots = [
        Plot("wave", "Samples (dots), the true waveform and the interpolants (diamonds)",
             x="sample index", y="in-phase amplitude", xlim=(0, 24), ylim=(-1.4, 1.6),
             legend="tl", legend_cols=3),
        Plot("mag", "Magnitude response at this μ (green: the signal band)",
             x="frequency (cycles per sample)", y="gain (dB)", xlim=(0, 0.5), ylim=(-12, 1.5),
             legend=None),
        Plot("mer", "Interpolation error (MER) against μ", x="μ", y="MER (dB)", xlim=(0, 1),
             ylim=(0, 95), legend="tl", legend_cols=3),
    ]
    layout = [["wave", "wave"], ["mag", "mer"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("mer", "MER at this μ", "dB", ".1f", good=lambda x: x > 30),
        Readout("worst", "Worst MER over μ", "dB", ".1f", good=lambda x: x > 30),
        Readout("droop", "Gain at the band edge", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Find the μ where linear interpolation is worst.",
                  lambda s: s.p.kind == "Linear" and abs(s.p.mu - 0.5) < 0.03),
        Challenge("Make linear interpolation good enough for 64-QAM: worst MER above 30 dB.",
                  lambda s: s.p.kind == "Linear" and s.r.worst > 30,
                  hint="Oversample: each doubling of samples per symbol buys about 12 dB."),
        Challenge("Reach a worst-case MER of 40 dB with only 3 samples per symbol.",
                  lambda s: s.p.sps == 3 and s.r.worst > 40,
                  hint="Pick the interpolator, then check the roll-off too."),
    ]

    def update(self, p):
        kind = INTERP[p.kind]
        sps = p.sps
        xf, up = interp_reference(round(p.beta, 2), sps)
        xs = xf[::up]
        start = 300
        L = 24
        pw = self.plot("wave")
        tf = np.arange(start * up, (start + L) * up) / up - start
        pw.line("true", tf, xf[start * up:(start + L) * up].real, color=GRAY, width=1.4,
                name="true waveform")
        n = np.arange(L)
        pw.scatter("s", n, xs[start:start + L].real, color=NAVY, size=8, name="samples")
        h = sk.interp_taps(p.mu, kind)
        m = np.arange(1, L - 2)
        est = (h[0] * xs[start + m - 1] + h[1] * xs[start + m] + h[2] * xs[start + m + 1]
               + h[3] * xs[start + m + 2])
        pw.scatter("i", m + p.mu, est.real, color=RED, size=10, symbol="d", name="interpolants")
        f = np.linspace(0.001, 0.5, 250)
        pm = self.plot("mag")
        edge = (1 + p.beta) / (2 * sps)
        pm.band("sig", 0, min(edge, 0.5), color=GREEN, alpha=0.08)
        for k, (nm, kd) in enumerate(INTERP.items()):
            mag, _ = sk.interp_response(p.mu, f, kd)
            pm.line(kd, f, mag, color=[GRAY, ORANGE, NAVY][k], width=2.4 if kd == kind else 1.0,
                    name=nm.split(" (")[0])
        mag, _ = sk.interp_response(p.mu, f, kind)
        droop = float(np.interp(min(edge, 0.5), f, mag))
        pr = self.plot("mer")
        mus = np.linspace(0, 1, 21)
        for k, (nm, kd) in enumerate(INTERP.items()):
            curve = [min(interp_mer(round(p.beta, 2), sps, float(mu_), kd), 80) for mu_ in mus]
            pr.line(kd, mus, curve, color=[GRAY, ORANGE, NAVY][k], width=2.4 if kd == kind else 1.0,
                    name=nm.split(" (")[0])
            if kd == kind:
                worst = float(np.min(curve[1:-1]))
        mer = min(interp_mer(round(p.beta, 2), sps, p.mu, kind), 80)
        pr.scatter("you", [p.mu], [mer], color=RED, size=12)
        pr.hline("q64", 30, color=GREEN, style=":", label="64-QAM needs ≈ 30 dB", label_pos=0.55)
        self.readout(mer=mer, worst=worst, droop=droop)

    def story(self, p):
        w = self.r.get("worst", 0)
        s = ("<p>The timing loop wants a sample at m + μ, but the ADC only produced samples at "
             "the integers. An <b>interpolator</b> is a filter whose coefficients depend on μ; when "
             "they are polynomials in μ it can be built as a <b>Farrow structure</b>, a few fixed "
             "filters combined by Horner's rule, so μ can change every symbol at no cost.</p>"
             f"<p>{p.kind} at {v(p.sps, 'd')} samples per symbol: worst-case error {v(w, '.1f', 'dB')} "
             "below the signal. Its magnitude response (bottom left) droops across the signal "
             "band (green) most at μ = ½, where the interpolant is furthest from any sample.</p>")
        if p.sps == 2:
            s += ("<p>At two samples per symbol the signal fills most of the band, so a short "
                  "interpolator struggles; a cubic is the usual minimum for QAM.</p>")
        return "<h3>Between the samples</h3>" + s + keybox(
            "commlib.interp_cubic is the cubic Lagrange Farrow interpolator; GNU Radio's Symbol "
            "Sync uses an 8-tap MMSE polyphase design.")


# =============================================================================== 8. frame sync
SEQS = ["Barker-13", "m-sequence (63)", "Zadoff–Chu (63)", "Random QPSK (32)", "Random QPSK (64)"]


@lru_cache(maxsize=8)
def sync_seq(name):
    if name == "Barker-13":
        return sk.barker13().astype(complex)
    if name.startswith("m-seq"):
        return sk.mseq(6).astype(complex)
    if name.startswith("Zadoff"):
        return cl.zadoff_chu(25, 63)
    N = 32 if "32" in name else 64
    return QPSK.points[np.random.default_rng(N).integers(0, 4, N)]


def cfo_loss(N, nu):
    """|Σ e^{j2πνn}| / N for an offset of nu cycles per symbol."""
    n = np.arange(N)
    return float(np.abs(np.mean(np.exp(2j * np.pi * nu * n))))


class FrameSync(Experiment):
    title = "Finding the frame"
    blurb = "A preamble, a correlator and a threshold: misses against false alarms."
    book = "sec:ch10:burst"
    animate = True
    fps = 8
    controls = [
        Choice("seq", "Preamble", SEQS, "Random QPSK (32)", style="menu"),
        Slider("esn0", "Es/N0", -10.0, 10.0, 0.0, step=0.5, unit="dB"),
        Slider("thr", "Threshold Λt", 2.0, 25.0, 8.0, step=0.1),
        Slider("cfo", "Frequency offset", 0.0, 1.5, 0.0, step=0.01, unit="× 1/N",
               help="Carrier offset in units of 1/N cycles per symbol"),
        Button("clear", "Clear the counts"),
    ]
    plots = [
        Plot("corr", "Normalised correlation Λ over a 600-lag search", x="lag (symbols)",
             y="Λ = |c|² / (N σ²)", xlim=(0, 600), legend="tr"),
        Plot("acf", "The preamble's autocorrelation (aperiodic)", x="lag", y="|R| / N",
             xlim=(-64, 64), ylim=(0, 1.1), legend=None),
        BERPlot("roc", "False alarm and miss against threshold", x="threshold Λt",
                y="probability", ylim=(1e-7, 1.5), xlim=(0, 25), legend="br"),
    ]
    layout = [["corr", "corr"], ["acf", "roc"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("pfa", "False alarm per lag", "", "sci"),
        Readout("pmiss", "Miss probability", "", "sci", good=lambda x: x < 1e-2),
        Readout("count", "Bursts: found / missed / false", "", None),
        Readout("psl", "Peak-to-sidelobe", "dB", ".1f"),
    ]
    challenges = [
        Challenge("The book's example: a 32-symbol preamble at 0 dB with the threshold for a 10⁻³ "
                  "false-alarm rate per 600-lag search (Λt ≈ 13.3). How often does it miss?",
                  lambda s: s.p.seq == "Random QPSK (32)" and abs(s.p.esn0) < 0.3
                  and abs(s.p.thr - 13.3) < 0.15 and s.p.cfo == 0),
        Challenge("Same threshold and SNR: choose a preamble that misses less than one burst in "
                  "a thousand.",
                  lambda s: s.p.thr >= 13.2 and s.p.esn0 <= 0.25 and s.r.pmiss < 1e-3),
        Challenge("Kill a long preamble with frequency offset alone: miss probability above 50 % at "
                  "Es/N0 ≥ 5 dB.",
                  lambda s: s.p.esn0 >= 5 and s.p.cfo > 0 and s.r.pmiss > 0.5),
    ]
    nlag = 600

    def setup(self):
        self.found = self.missed = self.false = 0

    def on_clear(self, p):
        self.found = self.missed = self.false = 0

    def burst(self, p):
        pre = sync_seq(p.seq)
        N = len(pre)
        rng = self.rng
        pos = int(rng.integers(100, self.nlag - 100))
        data = QPSK.points[rng.integers(0, 4, self.nlag + N)]
        x = data.copy()
        x[pos:pos + N] = pre
        n = np.arange(len(x))
        x = x * np.exp(1j * (2 * np.pi * p.cfo / N * n + rng.uniform(0, 6.28)))
        n0 = 10 ** (-p.esn0 / 10)
        y = x + np.sqrt(n0 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
        c = np.correlate(y, pre, "valid")[:self.nlag]
        lam = np.abs(c) ** 2 / (N * (1 + n0))
        return lam, pos, N

    def theory(self, p, N, thr):
        n0 = 10 ** (-p.esn0 / 10)
        g = cfo_loss(N, p.cfo / N)
        pfa = np.exp(-thr)
        pmiss = stats.ncx2.cdf(2 * thr * (1 + n0) / n0, 2, 2 * N * g ** 2 / n0)
        return pfa, pmiss

    def draw(self, p, count):
        lam, pos, N = self.burst(p)
        pc = self.plot("corr")
        pc.line("lam", np.arange(len(lam)), lam, color=NAVY, width=1.2)
        pc.hline("thr", p.thr, color=RED, style="--", label=f"threshold {p.thr:.1f}", label_pos=0.02)
        pc.vline("true", pos, color=GREEN, style=":", width=1.2, label="preamble", label_pos=0.85)
        over = np.where(lam > p.thr)[0]
        hit = np.any(np.abs(over - pos) <= 1)
        fa = np.any(np.abs(over - pos) > 1)
        if len(over):
            pc.scatter("over", over, lam[over], color=RED, size=7, name="above threshold")
        pc.set_ylim(0, max(p.thr * 1.6, float(np.max(lam)) * 1.15))
        if count:
            self.found += int(hit)
            self.missed += int(not hit)
            self.false += int(fa)
        pre = sync_seq(p.seq)
        R = np.abs(np.correlate(pre, pre, "full")) / N
        lags = np.arange(-(N - 1), N)
        pa = self.plot("acf")
        pa.stems("r", lags, R, color=NAVY, size=4)
        side = np.max(np.delete(R, N - 1)) if N > 1 else 1e-9
        pr = self.plot("roc")
        t = np.linspace(0.01, 25, 200)
        pfa_c, pm_c = self.theory(p, N, t)
        pr.theory("pfa", t, pfa_c, color=GRAY, name="false alarm per lag")
        pr.theory("pm", t, np.maximum(pm_c, 1e-12), color=NAVY, name=f"miss, N = {N}")
        pfa, pm = self.theory(p, N, p.thr)
        pr.vline("thr", p.thr, color=RED, style="--", width=1.2)
        pr.scatter("cur", [p.thr, p.thr], [max(pfa, 1e-12), max(pm, 1e-12)], color=RED, size=10)
        self.readout(pfa=pfa, pmiss=float(pm), count=f"{self.found} / {self.missed} / {self.false}",
                     psl=float(-20 * np.log10(max(side, 1e-9))))

    def update(self, p):
        self.found = self.missed = self.false = 0
        self.draw(p, True)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        pm, pfa = self.r.get("pmiss", 0), self.r.get("pfa", 0)
        N = len(sync_seq(p.seq))
        s = (f"<p>A burst receiver correlates everything it hears with the {v(N, 'd')}-symbol "
             "preamble. Off the peak the correlation is a sum of random terms, so Λ is "
             f"exponentially distributed and each lag raises a false alarm with probability "
             f"e^(−Λt) = {v(st.sci(pfa))}, regardless of N or SNR. On the peak Λ grows with the "
             f"total preamble energy N·Es/N0, and the miss probability here is {v(st.sci(pm))}.</p>"
             "<p>Press Play to send bursts and count: green dotted is where the preamble really "
             "is, red dots are lags above your threshold.</p>")
        if p.cfo > 0:
            s += (f"<p>A frequency offset of {v(p.cfo, '.2f')}/N cycles per symbol rotates the "
                  "preamble during the correlation, so its terms stop adding coherently: at 1/N the "
                  "peak vanishes entirely. Long preambles need a frequency search, or a "
                  "differential correlator.</p>")
        return "<h3>Misses against false alarms</h3>" + s + keybox(
            "Size the preamble at the lowest SNR you must work at: 3 dB more preamble energy can "
            "cut misses a thousandfold.")


# =============================================================================== 9. OFDM / cell search
MODES = ["Schmidl–Cox (OFDM)", "LTE PSS (Zadoff–Chu)", "NR PSS (m-sequence)"]


@lru_cache(maxsize=4)
def sc_frame(nfft=64, cp=16, seed=5):
    rng = np.random.default_rng(seed)
    X = np.zeros(nfft, complex)
    even = np.r_[np.arange(2, 27, 2), np.arange(nfft - 26, nfft, 2)]
    X[even] = QPSK.points[rng.integers(0, 4, len(even))] * np.sqrt(2)
    pre = np.fft.ifft(X) * np.sqrt(nfft)
    syms = []
    for _ in range(4):
        D = np.zeros(nfft, complex)
        used = np.r_[1:27, 38:64]
        D[used] = QPSK.points[rng.integers(0, 4, len(used))]
        t = np.fft.ifft(D) * np.sqrt(nfft)
        syms.append(np.r_[t[-cp:], t])
    frame = np.concatenate([np.zeros(100), pre[-cp:], pre] + syms + [np.zeros(60)])
    return frame, 100 + cp


@lru_cache(maxsize=8)
def pss_time(kind, nid2):
    seq = sk.lte_pss_zc(nid2) if kind == "lte" else sk.nr_pss(nid2)
    return sk.ofdm_time(seq, 256, kind)


class CellSearch(Experiment):
    title = "OFDM timing and cell search"
    blurb = "Schmidl–Cox finds an OFDM symbol; a phone hunts for the LTE or 5G PSS under frequency offset."
    book = "sec:ch10:frame"
    controls = [
        Choice("mode", "Synchronizer", MODES, "LTE PSS (Zadoff–Chu)", style="menu"),
        Slider("cfo", "Frequency offset", -1.5, 1.5, 0.0, step=0.01, unit="subcarriers"),
        Slider("snr", "SNR", -10.0, 30.0, 10.0, step=0.5, unit="dB"),
        Choice("nid", "Cell's N_ID2 (sent)", ["0", "1", "2"], "0",
               enabled_if=lambda p: "PSS" in p.mode),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("metric", "Timing metric", x="sample", y="metric", legend="tr"),
        Plot("aux", "What a frequency offset does", x="frequency offset (subcarriers)",
             y="", xlim=(-1.5, 1.5), legend="bl"),
    ]
    layout = [["metric"], ["aux"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("terr", "Timing error", "samples", "int", good=lambda x: abs(x) <= 2),
        Readout("peak", "Peak height", "× ideal", ".2f"),
        Readout("est", "Detected", "", None),
        Readout("cfoe", "CFO estimate", "subcarriers", ".3f"),
    ]
    challenges = [
        Challenge("Fool LTE's PSS: find a frequency offset that puts a near-full-height peak "
                  "(> 0.7) at the wrong time.",
                  lambda s: s.p.mode.startswith("LTE") and abs(s.r.terr) > 20 and s.r.peak > 0.7),
        Challenge("Show that NR's PSS cannot be fooled the same way: a one-subcarrier offset "
                  "kills the peak (< 0.35) instead of moving it.",
                  lambda s: s.p.mode.startswith("NR") and abs(abs(s.p.cfo) - 1) < 0.05
                  and s.r.peak < 0.35),
        Challenge("Schmidl–Cox: estimate a 0.4-subcarrier offset to within 0.02 at an SNR of "
                  "5 dB or less.",
                  lambda s: s.p.mode.startswith("Schmidl") and abs(s.p.cfo - 0.4) < 0.011
                  and s.p.snr <= 5 and abs(s.r.cfoe - 0.4) < 0.02),
    ]

    def setup(self):
        self.seed = 2

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    @lru_cache(maxsize=4)
    def pss_vs_cfo(kind):
        sig = pss_time(kind, 0)
        E = np.sum(np.abs(sig) ** 2)
        eps = np.linspace(-1.5, 1.5, 121)
        n = np.arange(256)
        hts, lags = [], []
        for e in eps:
            y = sig * np.exp(2j * np.pi * e * n / 256)
            c = np.abs(np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(sig)))) / E
            k = int(np.argmax(c))
            hts.append(c.max())
            lags.append(k if k < 128 else k - 256)
        return eps, np.array(hts), np.array(lags)

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        pm, pa = self.plot("metric"), self.plot("aux")
        if p.mode.startswith("Schmidl"):
            frame, start = sc_frame()
            nfft = 64
            n = np.arange(len(frame))
            y = frame * np.exp(2j * np.pi * p.cfo * n / nfft)
            n0 = 10 ** (-p.snr / 10)
            y = y + np.sqrt(n0 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
            M, P = cl.schmidl_cox_metric(y, nfft // 2)
            pm.set_title("Schmidl–Cox metric M(d) = |P(d)|² / R(d)²")
            pm.set_labels(x="sample d", y="M(d)")
            pm.set_xlim(0, 420)
            pm.set_ylim(0, 1.3)
            pm.band("cp", 100, 100 + 16, color=GREEN, alpha=0.15)
            pm.text("cpl", 108, 1.25, "CP plateau", color=GREEN, anchor=(0.5, 1), size=8.5)
            pm.line("M", np.arange(len(M)), M, color=NAVY, width=1.6, name="M(d)")
            # timing: midpoint of the 90 % points around the maximum
            k = int(np.argmax(M))
            lvl = 0.9 * M[k]
            lo = k
            while lo > 0 and M[lo - 1] > lvl:
                lo -= 1
            hi = k
            while hi < len(M) - 1 and M[hi + 1] > lvl:
                hi += 1
            d = (lo + hi) // 2
            pm.vline("d", d, color=RED, style="--", label="estimate", label_pos=0.9)
            pm.vline("true", start, color=GREEN, style=":", label="true start", label_pos=0.78)
            cfoe = float(np.angle(P[d]) / np.pi)
            eps = np.linspace(-1.5, 1.5, 121)
            pa.set_title("Schmidl–Cox CFO estimate: angle(P)/π, unambiguous for ±1 subcarrier")
            pa.set_labels(y="estimate (subcarriers)")
            pa.set_ylim(-1.6, 1.6)
            pa.line("ideal", eps, eps, color=GRAY, style=":", width=1.0, name="truth")
            pa.line("est", eps, sk.wrap(eps, 2.0), color=NAVY, width=2.0, name="estimate (noise-free)")
            pa.scatter("you", [p.cfo], [cfoe], color=RED, size=12, name="this frame")
            self.readout(terr=int(d - start), peak=float(M[k]), est="OFDM symbol", cfoe=cfoe)
            return
        kind = "lte" if p.mode.startswith("LTE") else "nr"
        nid = int(p.nid)
        sig = pss_time(kind, nid)
        E = np.sum(np.abs(sig) ** 2)
        start = 300
        L = 800
        rx = np.zeros(L, complex)
        # surround the PSS with OFDM-like random data of the same power
        rx[:] = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2) * 0.5
        rx[start:start + 256] = sig
        n = np.arange(L)
        rx = rx * np.exp(2j * np.pi * p.cfo * n / 256)
        n0 = 10 ** (-p.snr / 10)
        rx = rx + np.sqrt(n0 / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        pm.set_title(("LTE" if kind == "lte" else "NR") + " PSS: correlation with the three candidates")
        pm.set_labels(x="lag (samples, N_FFT = 256)", y="|correlation| / ideal")
        pm.set_xlim(0, L - 256)
        pm.set_ylim(0, 1.35)
        best = (0, 0, 0)
        cols = [NAVY, ORANGE, PURPLE]
        for h_ in range(3):
            ref = pss_time(kind, h_)
            c = np.abs(np.correlate(rx, ref, "valid")) / E
            pm.line(f"c{h_}", np.arange(len(c)), c, color=cols[h_], width=2.0 if h_ == nid else 1.0,
                    name=f"N_ID2 = {h_}")
            k = int(np.argmax(c))
            if c[k] > best[0]:
                best = (c[k], k, h_)
        pm.vline("true", start, color=GREEN, style=":", label="true timing", label_pos=0.9)
        pm.scatter("pk", [best[1]], [best[0]], color=RED, size=12, name="detected peak")
        eps, hts, lags = self.pss_vs_cfo(kind)
        pa.set_title("Peak height (navy) and how far the peak moves, in hundreds of samples (red)")
        pa.set_labels(y="height / shift")
        pa.set_ylim(-1.6, 1.4)
        pa.line("h", eps, hts, color=NAVY, width=2.0, name="peak height")
        pa.line("lag", eps, lags / 100, color=RED, width=2.0, style="--", name="peak shift / 100 samples")
        pa.vline("cur", p.cfo, color=GRAY, style=":", width=1.2)
        self.readout(terr=int(best[1] - start), peak=float(best[0]),
                     est=f"N_ID2 = {best[2]}" + (" ✓" if best[2] == nid else " ✗"), cfoe="—")

    def story(self, p):
        if p.mode.startswith("Schmidl"):
            s = ("<p>Schmidl and Cox send a training symbol whose two halves are identical. "
                 "Correlating the signal with itself half a symbol later gives a metric that rises "
                 "to a plateau as long as the cyclic prefix (green): the start of the symbol lies "
                 "on it. The phase of that correlation is the frequency offset, which travels "
                 "half a symbol: it is unambiguous for ±1 subcarrier spacing (bottom); the integer "
                 "part needs a second training symbol.</p>")
            return "<h3>Delay-and-correlate</h3>" + s + keybox(
                "802.11's short training field is the same idea with a 16-sample period: ten "
                "repetitions for detection, AGC and coarse frequency.")
        s = ("<p>A phone switched on in an unknown place correlates against the three possible "
             "primary synchronization signals to find the symbol timing and N_ID2. Its crystal "
             "may be several ppm off, i.e. a subcarrier or more.</p>")
        if p.mode.startswith("LTE"):
            s += ("<p>LTE's PSS is a Zadoff–Chu sequence in frequency. A one-subcarrier offset "
                  "shifts it by one position, which for a Zadoff–Chu sequence is a delay in time: "
                  "the correlation peak <b>moves</b> (red dashed, bottom) and stays almost full "
                  "height. A false timing.</p>")
        else:
            s += ("<p>NR's PSS is a BPSK m-sequence. A frequency offset does not move its peak; "
                  "it simply <b>destroys</b> it beyond about half a subcarrier. The receiver must "
                  "try a few frequency hypotheses, but it can never be fooled into the wrong timing.</p>")
        return "<h3>Cell search under frequency offset</h3>" + s + keybox(
            "The coupling of timing and frequency in Zadoff–Chu PSS is one reason 5G NR switched to "
            "m-sequences (3GPP TS 38.211).")


# =============================================================================== the lab
LAB = st.Lab(4, "Synchronization: Carrier, Timing and Frame", chapter=10,
             chapter_title="Synchronization",
             experiments=[Diagnosis, PLLLoop, PhaseDetectors, CostasLoop, FrequencyEstimation,
                          TimingRecovery, Interpolators, FrameSync, CellSearch])

if __name__ == "__main__":
    st.run(LAB)
