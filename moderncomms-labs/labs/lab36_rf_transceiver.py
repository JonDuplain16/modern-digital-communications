"""Lab 36 · The RF Transceiver: Linearity, Noise, Phase Noise, PAs and DPD   (Chapter 7)

Run it:      python labs/lab36_rf_transceiver.py
Self-test:   python labs/lab36_rf_transceiver.py --selftest

A radio's data sheet is a list of compromises: noise figure against linearity, phase noise
against cost, efficiency against spectral purity. This is the RF engineer's bench: drive an
amplifier with two tones, build a receiver line-up stage by stage, smear a blocker with a noisy
local oscillator, skew a constellation with I/Q imbalance, push an OFDM signal into a power
amplifier, rescue its efficiency with Doherty and crest-factor reduction, and finally teach a
digital predistorter to cancel the PA's distortion, one click at a time. All the models are in
commlib/rf.py, shared with Chapter 7's figures.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as sps

import commlib as cl
from commlib import rf
import studio as st
from studio import (Experiment, Slider, IntSlider, Choice, Toggle, Button, Heading, Plot,
                    SpectrumPlot, ConstellationPlot, BarPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
@lru_cache(maxsize=8)
def ofdm(n_sym=40, seed=8, n_used=200, nfft=1024, ncp=72, lowpass=True):
    """Unit-power 64-QAM OFDM (Chapter 7's test signal). Returns (x, grid, cfg)."""
    r = np.random.default_rng(seed)
    cfg = cl.OFDMConfig(nfft=nfft, n_used=n_used, ncp=ncp)
    g = cl.get_constellation("64qam").modulate(cl.random_bits(6 * n_used * n_sym, r)).reshape(n_sym, n_used)
    x = cl.ofdm_modulate(g, cfg)
    if lowpass:
        x = np.convolve(x, sps.firwin(301, 1.15 * n_used / nfft), mode="same")
    s = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / s, g, cfg


def psd(x, nper=2048):
    f, P = sps.welch(x, nperseg=nper, return_onesided=False, window="blackmanharris", detrend=False)
    return np.fft.fftshift(f), np.fft.fftshift(P)


def aclr_db(x, bw, off, nper=2048):
    f, P = psd(x, nper)
    main = P[np.abs(f) < bw / 2].sum()
    adj = max(P[np.abs(f - off) < bw / 2].sum(), P[np.abs(f + off) < bw / 2].sum())
    return float(rf.db10(main / max(adj, 1e-300))), f, P


def evm_pct(y, grid, cfg):
    Y = cl.ofdm_demodulate(y, cfg)
    X = grid[:Y.shape[0]]
    Y, X = Y[2:-2], X[2:-2]
    g = np.vdot(X, Y) / np.vdot(X, X)
    return 100 * float(np.sqrt(np.sum(np.abs(Y - g * X) ** 2) / np.sum(np.abs(g * X) ** 2))), Y / g


def dbm(a2):
    """Power in dBm of a real tone of amplitude a (sqrt(mW)), given a² = |amplitude|²."""
    return rf.db10(a2 / 2)


# =============================================================================== 1. two-tone
class TwoTone(Experiment):
    title = "Two-tone test and IIP3"
    blurb = "Two equal tones in, intermodulation out: measure the third-order intercept."
    book = "sec:ch07:metrics"
    controls = [
        Heading("Device under test"),
        Choice("model", "Amplifier model", ["Cubic (textbook)", "Soft limiter (real)"],
               "Soft limiter (real)", style="menu",
               help="Cubic: y = a₁x + a₃x³ exactly. Soft limiter: a tanh that also saturates"),
        Slider("gain", "Gain", 0, 30, 15, step=0.5, unit="dB"),
        Slider("iip3", "Input IP3", -20, 30, 0, step=0.5, unit="dBm"),
        Heading("Test signal"),
        Slider("pin", "Input power per tone", -70, 10, -40, step=0.5, unit="dBm"),
    ]
    plots = [
        SpectrumPlot("spec", "Output spectrum (tones at 4.0 and 4.4 MHz)", x="frequency (MHz)",
                     y="output power (dBm)", xlim=(2.6, 5.8), ylim=(-140, 45), legend=None),
        Plot("sweep", "Tones and IM3 against drive", x="input power per tone (dBm)",
             y="output power (dBm)", xlim=(-70, 30), ylim=(-150, 60), legend="br"),
        Plot("comp", "Single-tone gain compression", x="input power (dBm)", y="gain change (dB)",
             xlim=(-70, 30), ylim=(-6, 1), legend=None),
    ]
    layout = [["spec", "spec"], ["sweep", "comp"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("delta", "IM3 below the tones", "dBc", ".1f"),
        Readout("est", "IIP3 = P_in + Δ/2", "dBm", ".1f"),
        Readout("im3in", "IM3 referred to input", "dBm", ".1f"),
        Readout("p1db", "Input P1dB", "dBm", ".1f"),
    ]
    challenges = [
        Challenge("Set the drive so the IM3 products sit exactly 60 dB below the tones (±1 dB).",
                  lambda s: abs(s.r.delta - 60) < 1,
                  hint="Every 1 dB more drive costs 2 dB of Δ: Δ = 2(IIP3 − P_in)."),
        Challenge("A blocking test: with two −30 dBm interferers, keep the input-referred IM3 below "
                  "−100 dBm by choosing the amplifier's IIP3.",
                  lambda s: abs(s.p.pin + 30) < 0.3 and s.r.im3in <= -100),
        Challenge("Fool the single-point method: drive the soft limiter so hard that P_in + Δ/2 is off "
                  "by more than 3 dB from the true IIP3.",
                  lambda s: s.p.model.startswith("Soft") and abs(s.r.est - s.p.iip3) > 3),
        Challenge("Find the 1 dB compression point: set the drive within 0.3 dB of P1dB and check "
                  "it sits about 9.6 dB below IIP3.",
                  lambda s: s.p.model.startswith("Soft") and abs(s.p.pin - s.r.p1db) < 0.3),
    ]
    n = 4096
    k1, k2 = 400, 440                                   # FFT bins of the two tones
    fs_mhz = 40.96

    @staticmethod
    def system(model, gain, iip3):
        a1 = 10 ** (gain / 20)
        A_ip = np.sqrt(2 * 10 ** (iip3 / 10))           # input amplitude at IIP3
        if model.startswith("Cubic"):
            a3 = -4 / 3 * a1 / A_ip ** 2
            return lambda x: a1 * x + a3 * x ** 3
        Asat = a1 * A_ip / 2                            # tanh(u) = u − u³/3 → this IIP3
        return lambda x: Asat * np.tanh(a1 * x / Asat)

    def tones(self, sysf, pin):
        t = np.arange(self.n)
        A = np.sqrt(2 * 10 ** (pin / 10))
        x = A * (np.cos(2 * np.pi * self.k1 * t / self.n) + np.cos(2 * np.pi * self.k2 * t / self.n))
        Y = np.fft.rfft(sysf(x)) / self.n * 2
        return Y

    def measure(self, sysf, pin):
        Y = self.tones(sysf, pin)
        kd = self.k2 - self.k1
        pt = dbm(np.abs(Y[self.k1]) ** 2)
        pi_ = dbm(max(np.abs(Y[self.k1 - kd]) ** 2, np.abs(Y[self.k2 + kd]) ** 2))
        return pt, pi_, Y

    @lru_cache(maxsize=64)
    def sweeps(self, model, gain, iip3):
        sysf = self.system(model, gain, iip3)
        pins = np.arange(-70, 30.1, 2.0)
        res = np.array([self.measure(sysf, p)[:2] for p in pins])
        # single tone
        t = np.arange(self.n)
        g = []
        for p in pins:
            A = np.sqrt(2 * 10 ** (p / 10))
            Y = np.fft.rfft(sysf(A * np.cos(2 * np.pi * self.k1 * t / self.n))) / self.n * 2
            g.append(dbm(np.abs(Y[self.k1]) ** 2) - p - gain)
        g = np.array(g)
        pf = np.linspace(-70, 30, 2001)
        gf = np.interp(pf, pins, g)
        p1 = pf[np.argmax(gf <= -1)] if np.any(gf <= -1) else np.nan
        return pins, res, g, p1

    def update(self, p):
        sysf = self.system(p.model, p.gain, p.iip3)
        pt, pim, Y = self.measure(sysf, p.pin)
        f = np.arange(len(Y)) / self.n * self.fs_mhz
        P = dbm(np.abs(Y) ** 2 + 1e-17)
        ps = self.plot("spec")
        ps.line("P", f, P, color=NAVY, width=1.6, fill=-140, fill_alpha=0.08)
        kd = (self.k2 - self.k1) / self.n * self.fs_mhz
        f1, f2 = self.k1 / self.n * self.fs_mhz, self.k2 / self.n * self.fs_mhz
        ps.scatter("im3", [f1 - kd, f2 + kd], [pim, pim], color=RED, size=11, symbol="d")
        ps.text("im3l", f2 + kd + 0.05, pim, "IM3 (2f₂ − f₁)", color=RED, anchor=(0, 0.5), size=9)
        ps.text("im3r", f1 - kd - 0.05, pim, "2f₁ − f₂", color=RED, anchor=(1, 0.5), size=9)
        ps.hline("pt", pt, color=GREEN, style=":", width=1.0)
        ps.hline("pi", pim, color=RED, style=":", width=1.0)
        delta = pt - pim
        ps.arrow("da", f2 + 0.2, pim, f"Δ = {delta:.1f} dB", color=ORANGE, direction="up", size=10)
        pins, res, g, p1 = self.sweeps(p.model, round(p.gain, 2), round(p.iip3, 2))
        pw = self.plot("sweep")
        pp = np.linspace(-70, 30, 50)
        pw.line("l1", pp, pp + p.gain, color=NAVY, width=1.0, style="--")
        pw.line("l3", pp, 3 * pp - 2 * p.iip3 + p.gain, color=RED, width=1.0, style="--")
        pw.scatter("t", pins, res[:, 0], color=NAVY, size=6, name="tone out (slope 1)")
        pw.scatter("i", pins, np.maximum(res[:, 1], -160), color=RED, size=6, name="IM3 out (slope 3)")
        pw.scatter("ip", [p.iip3], [p.iip3 + p.gain], color=GREEN, size=16, symbol="star",
                   name="intercept")
        pw.vline("now", p.pin, color=ORANGE, style="-", width=1.4, label="your drive", label_pos=0.06)
        pc = self.plot("comp")
        pc.hline("m1", -1, color=RED, style=":", width=1.2, label="−1 dB", label_pos=0.05)
        pc.line("g", pins, np.maximum(g, -6), color=NAVY, width=2.0)
        if np.isfinite(p1):
            pc.vline("p1", p1, color=RED, style="--", width=1.2, label=f"P1dB {p1:.1f} dBm", label_pos=0.9)
        pc.vline("now", p.pin, color=ORANGE, style="-", width=1.4)
        self.readout(delta=float(delta), est=float(p.pin + delta / 2),
                     im3in=float(3 * p.pin - 2 * p.iip3), p1db=float(p1) if np.isfinite(p1) else None)

    def story(self, p):
        r = self.r
        d = r.get("delta", 0)
        s = (f"<p>Two equal tones at {v(p.pin, '.1f', 'dBm')} each go in. Any curvature of the "
             f"amplifier mixes them: the products at 2f₁ − f₂ and 2f₂ − f₁ (red) land right next to "
             f"the tones, in band, where no filter can reach them. Here they are "
             f"{v(d, '.1f', 'dB')} below the tones.</p>"
             "<p>Raise the drive 1 dB and the tones rise 1 dB but IM3 rises 3 dB (bottom left). "
             "The extrapolated lines meet at the <b>intercept</b>, IIP3 = P_in + Δ/2: a fiction "
             "no device survives, but it predicts the IM3 at any lower level.</p>")
        if p.model.startswith("Soft"):
            err = r.get("est", 0) - p.iip3
            if abs(err) > 1:
                s += (f"<p>{bad('Too hot.')} Near compression the tones stop growing and the "
                      f"single-point estimate is off by {v(err, '+.1f', 'dB')}: always measure IIP3 "
                      f"well below P1dB.</p>")
            else:
                s += (f"<p>A real amplifier also <b>compresses</b> (bottom right): the 1 dB point is "
                      f"{v(r.get('p1db') or 0, '.1f', 'dBm')}, about 9.6 dB below IIP3, the engineer's "
                      f"rule of thumb.</p>")
        return "<h3>The intercept that never happens</h3>" + s + keybox(
            "IIP3 = P_in + Δ/2.  Input-referred IM3 = 3·P_in − 2·IIP3.  P1dB ≈ IIP3 − 9.6 dB.")


# =============================================================================== 2. line-up
class LineUp(Experiment):
    title = "Receiver line-up builder"
    blurb = "Gain, noise figure and IIP3 stage by stage: noise wants gain early, linearity late."
    book = "sec:ch07:metrics"
    controls = [
        Heading("Before the LNA"),
        Slider("loss", "Switch + balun loss", 0, 5, 2.5, step=0.1, unit="dB"),
        Heading("LNA"),
        Slider("lg", "LNA gain", 0, 30, 20, step=0.5, unit="dB"),
        Slider("lnf", "LNA noise figure", 0.5, 4, 1.5, step=0.1, unit="dB"),
        Slider("lip", "LNA IIP3", -25, 10, -10, step=0.5, unit="dBm"),
        Heading("Mixer + TIA (15 dB gain)"),
        Slider("mnf", "Mixer noise figure", 5, 16, 10, step=0.5, unit="dB"),
        Slider("mip", "Mixer IIP3", -10, 20, 5, step=0.5, unit="dBm"),
        Heading("Baseband filter + VGA (OIP3 +45 dBm)"),
        Slider("vg", "VGA gain", 0, 40, 30, step=0.5, unit="dB"),
        Choice("bw", "Channel bandwidth", ["200 kHz", "1 MHz", "20 MHz"], "1 MHz"),
    ]
    plots = [
        Plot("nf", "Cumulative noise figure", x="", y="NF (dB)", ylim=(0, 10), legend=None),
        Plot("ip", "Cumulative IIP3 (input-referred)", x="", y="IIP3 (dBm)", ylim=(-40, 20),
             legend=None),
        Plot("trade", "The LNA-gain trade-off", x="LNA gain (dB)", y="dB / dBm", xlim=(0, 30),
             ylim=(-40, 80), legend="tr"),
        Plot("win", "Dynamic-range window", x="", y="input level (dBm)", ylim=(-130, 10),
             legend=None),
    ]
    layout = [["nf", "ip", "win"], ["trade", "trade", "win"]]
    col_stretch = [3, 3, 2]
    readouts = [
        Readout("nf", "Noise figure", "dB", ".2f"),
        Readout("ip", "IIP3", "dBm", ".1f"),
        Readout("sfdr", "SFDR", "dB", ".1f", good=lambda x: x >= 65),
        Readout("sens", "Sensitivity (10 dB SNR)", "dBm", ".1f"),
    ]
    challenges = [
        Challenge("Push the SFDR in 1 MHz above 65 dB.",
                  lambda s: s.p.bw == "1 MHz" and s.r.sfdr >= 65,
                  hint="Which stage dominates the IIP3? Turn down the gain in front of it."),
        Challenge("The gain-table trick: lower only the VGA gain to raise IIP3 to −15 dBm or better "
                  "while the noise figure stays below 4.7 dB.",
                  lambda s: s.r.ip >= -15 and s.r.nf < 4.7 and abs(s.p.lg - 20) < 0.3
                  and abs(s.p.loss - 2.5) < 0.05 and s.p.vg < 30),
        Challenge("Loss before the LNA counts double: with 5 dB of loss, get the noise figure below "
                  "6 dB using the LNA alone. (Compare: the same LNA after 2.5 dB gives 4.4 dB.)",
                  lambda s: s.p.loss >= 4.95 and s.r.nf < 6 and abs(s.p.mnf - 10) < 0.3),
    ]

    @staticmethod
    def stages(p, lg=None):
        lg = p.lg if lg is None else lg
        return [("switch", -p.loss, p.loss, None), ("LNA", lg, p.lnf, p.lip),
                ("mixer", 15.0, p.mnf, p.mip), ("VGA", p.vg, 20.0, 45.0 - p.vg)]

    def update(self, p):
        B = {"200 kHz": 200e3, "1 MHz": 1e6, "20 MHz": 20e6}[p.bw]
        rows = rf.cascade(self.stages(p))
        nf, ip = rows[-1][2], rows[-1][3]
        floor = rf.K_DBM + 10 * np.log10(B) + nf
        sfdr = 2 / 3 * (ip - floor)
        x = np.arange(4)
        ticks = [(i, n) for i, n in enumerate(["switch", "LNA", "mixer", "VGA"])]
        pn = self.plot("nf")
        cnf = [r_[2] for r_ in rows]
        pn.bars("b", x, cnf, width=0.6, colors=[GRAY, NAVY, NAVY, NAVY])
        for i, val in enumerate(cnf):
            pn.text(f"t{i}", i, val, f"{val:.2f}", anchor=(0.5, 1.05), size=9, bold=True)
        pn.set_xticks(ticks)
        pn.set_xlim(-0.6, 3.6)
        pn.set_ylim(0, max(10, max(cnf) * 1.25))
        pi_ = self.plot("ip")
        cip = [r_[3] if np.isfinite(r_[3]) else np.nan for r_ in rows]
        pi_.bars("b", x[1:], np.array(cip[1:]), width=0.6, colors=[RED, RED, RED], base=-40)
        for i in range(1, 4):
            pi_.text(f"t{i}", i, cip[i], f"{cip[i]:.1f}", anchor=(0.5, 1.05), size=9, bold=True)
        pi_.set_xticks(ticks)
        pi_.set_xlim(-0.6, 3.6)
        # trade-off over LNA gain
        Gl = np.linspace(0, 30, 121)
        NF, IP, SF = [], [], []
        for g in Gl:
            c = rf.cascade(self.stages(p, g))[-1]
            NF.append(c[2])
            IP.append(c[3])
            SF.append(2 / 3 * (c[3] - (rf.K_DBM + 10 * np.log10(B) + c[2])))
        k = int(np.argmax(SF))
        pt = self.plot("trade")
        pt.line("nf", Gl, NF, color=NAVY, width=2.0, name="NF (dB)")
        pt.line("ip", Gl, IP, color=RED, width=2.0, name="IIP3 (dBm)")
        pt.line("sf", Gl, SF, color=GREEN, width=2.4, name="SFDR (dB)")
        pt.scatter("best", [Gl[k]], [SF[k]], color=GREEN, size=12, symbol="star")
        pt.vline("now", p.lg, color=ORANGE, style="-", width=1.4, label="your LNA", label_pos=0.06)
        # dynamic-range window
        pmax = (2 * ip + floor) / 3
        sens = floor + 10
        pw = self.plot("win")
        pw.hband("sf", floor, pmax, color=GREEN, alpha=0.18)
        pw.hline("fl", floor, color=GRAY, style="--", width=1.4, label=f"noise floor {floor:.0f}",
                 label_pos=0.02)
        pw.hline("se", sens, color=NAVY, style=":", width=1.2, label=f"sensitivity {sens:.0f}",
                 label_pos=0.02)
        pw.hline("pm", pmax, color=ORANGE, style="--", width=1.4, label=f"IM3 = floor {pmax:.0f}",
                 label_pos=0.02)
        pw.hline("ip", ip, color=RED, style="-", width=1.6, label=f"IIP3 {ip:.0f}", label_pos=0.02)
        pw.set_xlim(0, 1)
        pw.set_xticks([])
        pw.text("sfl", 0.95, (floor + pmax) / 2, f"SFDR\n{sfdr:.0f} dB", color=GREEN, anchor=(1, 0.5),
                size=10, bold=True)
        self.readout(nf=nf, ip=ip, sfdr=sfdr, sens=sens, best=float(Gl[k]))

    def story(self, p):
        r = self.r
        s = (f"<p>Friis: each stage's noise is divided by the gain in front of it, so the first "
             f"stages set the noise figure ({v(r.get('nf', 0), '.2f', 'dB')}). The "
             f"{v(p.loss, '.1f', 'dB')} of loss before the LNA adds its full value: it is the most "
             f"expensive decibel in the radio.</p>"
             f"<p>Linearity is the mirror image: each stage's intercept is <i>divided</i> by the gain "
             f"in front of it, so the last, high-gain stage usually dominates the IIP3 "
             f"({v(r.get('ip', 0), '.1f', 'dBm')}).</p>"
             f"<p>The <b>SFDR</b>, ⅔(IIP3 − floor), is the window between them: "
             f"{v(r.get('sfdr', 0), '.1f', 'dB')} in {p.bw}. Its best LNA gain is about "
             f"{v(r.get('best', 0), '.0f', 'dB')}: more gain buys a vanishing NF improvement and "
             f"costs linearity.</p>")
        return "<h3>Noise wants gain early, linearity wants it late</h3>" + s + keybox(
            "F = F₁ + (F₂−1)/G₁ + …   1/IIP3 = 1/IIP3₁ + G₁/IIP3₂ + …   SFDR = ⅔(IIP3 − N).")


# =============================================================================== 3. phase noise
class PhaseNoise(Experiment):
    title = "Phase noise and reciprocal mixing"
    blurb = "A noisy LO smears a strong neighbour onto your channel and jitters your constellation."
    book = "sec:ch07:recip"
    controls = [
        Heading("LO phase-noise mask L(f)"),
        Slider("L1", "L at 1 MHz offset", -150, -80, -100, step=1, unit="dBc/Hz"),
        Choice("slope", "Slope", ["20 dB/dec", "30 dB/dec"]),
        Slider("plat", "In-loop plateau", -110, -70, -85, step=1, unit="dBc/Hz",
               help="Inside the PLL bandwidth the noise is flat"),
        Slider("floor", "Far-out floor", -170, -130, -150, step=1, unit="dBc/Hz"),
        Heading("Blocker"),
        Slider("pb", "Blocker power", -80, -10, -40, step=1, unit="dBm"),
        Slider("off", "Blocker offset", 0.4, 3.0, 1.0, step=0.05, unit="MHz"),
    ]
    plots = [
        Plot("mask", "LO phase noise: mask (line), synthesised (dots)", x="offset from carrier (Hz)",
             y="L(f) (dBc/Hz)", logx=True, xlim=(1e3, 8e6), ylim=(-175, -60), legend=None),
        SpectrumPlot("blk", "The blocker after mixing with the noisy LO", x="frequency (MHz)",
                     y="power density (dBm/Hz)", xlim=(-3.2, 3.2), ylim=(-190, -60), legend=None),
        ConstellationPlot("con", "64-QAM through this LO", lim=1.35),
        Plot("des", "Desensitisation", x="blocker power (dBm)", y="noise-floor rise (dB)",
             xlim=(-80, -10), ylim=(0, 40), legend="tl"),
    ]
    layout = [["mask", "blk"], ["con", "des"]]
    readouts = [
        Readout("rms", "RMS phase error", "°", ".2f"),
        Readout("rm", "Reciprocal-mixing noise", "dBm", ".1f"),
        Readout("des", "Desensitisation", "dB", ".1f", good=lambda x: x < 1),
        Readout("evm", "64-QAM EVM", "%", ".2f", good=lambda x: x < 1),
    ]
    challenges = [
        Challenge("The book's requirement: a −40 dBm blocker 1 MHz away may cost at most 0.5 dB of "
                  "sensitivity. Find the L(1 MHz) that achieves it.",
                  lambda s: abs(s.p.pb + 40) < 0.5 and abs(s.p.off - 1) < 0.03 and s.r.des <= 0.5),
        Challenge("With L(1 MHz) = −120 dBc/Hz, find the strongest blocker at 1 MHz that costs no "
                  "more than 3 dB.",
                  lambda s: abs(s.p.L1 + 120) < 0.5 and abs(s.p.off - 1) < 0.03 and s.r.des <= 3
                  and s.p.pb >= -50),
        Challenge("Get 64-QAM EVM below 1 % from phase noise alone.",
                  lambda s: s.r.evm < 1, hint="EVM ≈ σ_φ in radians. Which part of the mask holds most of the power?"),
    ]
    fs = 16e6
    n = 1 << 17
    B, NF = 200e3, 5.0

    @lru_cache(maxsize=16)
    def phi(self, L1, slope, plat, floor):
        mf = np.logspace(2, np.log10(self.fs / 2), 80)
        md = np.clip(L1 - slope * np.log10(mf / 1e6), floor, plat)
        ph = rf.phase_noise_from_mask(self.n, self.fs, mf, md, np.random.default_rng(3))
        return mf, md, ph

    def update(self, p):
        slope = 20.0 if p.slope.startswith("20") else 30.0
        mf, md, ph = self.phi(p.L1, slope, p.plat, p.floor)
        t = np.arange(self.n) / self.fs
        blk = np.sqrt(10 ** (p.pb / 10)) * np.exp(2j * np.pi * p.off * 1e6 * t + 1j * ph)
        f, P = sps.welch(blk, self.fs, nperseg=4096, return_onesided=False, window="blackmanharris")
        f, P = np.fft.fftshift(f), np.fft.fftshift(P)
        df = f[1] - f[0]
        thermal = rf.K_DBM + 10 * np.log10(self.B) + self.NF
        Lb = float(np.interp(np.log10(p.off * 1e6), np.log10(mf), md))
        rm_pred = p.pb + Lb + 10 * np.log10(self.B)
        inband = np.sum(P[np.abs(f) < self.B / 2]) * df
        rm = float(rf.db10(inband))
        des = float(rf.db10(10 ** (thermal / 10) + 10 ** (rm / 10)) - thermal)
        # plots
        pm = self.plot("mask")
        pm.line("m", mf, md, color=NAVY, width=2.2)
        sel = (f > p.off * 1e6 + 1.5e3)
        fo = f[sel] - p.off * 1e6
        Lm = rf.db10(P[sel] / 10 ** (p.pb / 10))
        idx = np.unique(np.geomspace(1, len(fo) - 1, 140).astype(int))
        pm.scatter("meas", fo[idx], Lm[idx], color=RED, size=4)
        pm.vline("b", p.off * 1e6, color=ORANGE, style="--", width=1.2, label="blocker offset",
                 label_pos=0.92)
        pb_ = self.plot("blk")
        pb_.line("P", f / 1e6, rf.db10(P), color=RED, width=1.2, name="blocker + LO noise")
        pb_.band("ch", -self.B / 2e6, self.B / 2e6, color=NAVY, alpha=0.22)
        pb_.hline("th", thermal - 10 * np.log10(self.B), color=GRAY, style="--", width=1.4,
                  label="thermal floor (NF 5 dB)", label_pos=0.03)
        pb_.text("chl", -0.15, -66, "your channel", color=NAVY, anchor=(1, 0), size=9)
        # 64-QAM constellation: symbols at 1 MBd see the LO phase sampled every 16 samples
        qam = cl.get_constellation("64qam")
        sym = qam.points[np.random.default_rng(1).integers(0, 64, 2000)]
        phs = ph[::16][:2000]
        z = sym * np.exp(1j * phs)
        evm = 100 * float(np.sqrt(np.mean(np.abs(z - sym) ** 2) / np.mean(np.abs(sym) ** 2)))
        pc = self.plot("con")
        pc.points("z", z, color=NAVY, size=3, alpha=0.5)
        pc.ideal("i", qam.points, color=RED, size=8)
        pd = self.plot("des")
        pbv = np.linspace(-80, -10, 141)
        for Lv, col in [(-100, RED), (-120, ORANGE), (-140, GREEN)]:
            pd.line(f"L{-Lv}", pbv, rf.db10(10 ** ((pbv + Lv + 10 * np.log10(self.B)) / 10)
                                          + 10 ** (thermal / 10)) - thermal, color=col, width=1.4,
                    style="--", name=f"L = {Lv} dBc/Hz")
        pd.line("now", pbv, rf.db10(10 ** ((pbv + Lb + 10 * np.log10(self.B)) / 10) + 10 ** (thermal / 10))
                - thermal, color=NAVY, width=2.4, name="your LO")
        pd.scatter("pt", [p.pb], [min(des, 39.5)], color=NAVY, size=13, symbol="d")
        ff = np.logspace(3, np.log10(self.fs / 2), 400)
        rms = rf.rms_phase_deg(ff, np.interp(np.log10(ff), np.log10(mf), md))
        self.readout(rms=float(rms), rm=rm, des=des, evm=evm, pred=float(rm_pred), thermal=float(thermal),
                     Lb=Lb)

    def story(self, p):
        r = self.r
        s = (f"<p>A real oscillator's phase wanders: L(f) (top left) is the noise power per hertz at "
             f"offset f from the carrier. Integrated, this LO has {v(r.get('rms', 0), '.2f', '°')} of "
             f"RMS phase error, which smears every constellation point along an arc (bottom left).</p>"
             f"<p>Worse, the receiver multiplies <i>every</i> signal by the LO, including a "
             f"{v(p.pb, '.0f', 'dBm')} blocker {v(p.off, '.2f', 'MHz')} away. Its copy of the LO's "
             f"skirt lands on your channel: P_b + L(Δf) + 10·log B = "
             f"{v(r.get('pred', 0), '.1f', 'dBm')} (measured {v(r.get('rm', 0), '.1f', 'dBm')}), against "
             f"a thermal floor of {v(r.get('thermal', 0), '.1f', 'dBm')}. This is <b>reciprocal "
             f"mixing</b>.</p>")
        d = r.get("des", 0)
        s += ("<p>" + (bad(f"Desensitised by {d:.1f} dB.") if d > 3 else
                       good(f"Only {d:.1f} dB of desensitisation.")) +
              " That is why cellular blocking tests, not the plain sensitivity test, set the "
              "synthesiser's phase-noise specification.</p>")
        return "<h3>The LO's skirt</h3>" + s + keybox(
            "Reciprocal-mixing noise = P_blocker + L(Δf) + 10·log₁₀B.  EVM ≈ σ_φ (radians).")


# =============================================================================== 4. IQ imbalance
class IQImbalance(Experiment):
    title = "I/Q imbalance and the image"
    blurb = "A tenth of a decibel and a degree: how a mirror image appears, and how DSP removes it."
    book = "sec:ch07:iq"
    controls = [
        Heading("Receiver imbalance"),
        Slider("g", "Gain imbalance", 0, 2, 0.5, step=0.01, unit="dB"),
        Slider("ph", "Phase error", 0, 10, 3, step=0.05, unit="°"),
        Toggle("corr", "Blind digital correction", False,
               help="Estimate g and φ from the received statistics and undo them"),
        Heading("Scenario"),
        Choice("scen", "Signals", ["Zero-IF: one 16-QAM signal", "Capture: strong neighbour at the mirror"],
               style="menu"),
        Slider("nlev", "Neighbour above wanted", 0, 50, 30, step=1, unit="dB",
               enabled_if=lambda p: p.scen.startswith("Capture")),
    ]
    plots = [
        ImagePlot("irr", "Image rejection (dB) vs gain and phase error", x="gain imbalance (dB)",
                  y="phase error (°)"),
        SpectrumPlot("spec", "Received spectrum", x="frequency (MHz)", y="level (dB)",
                     xlim=(-4, 4), ylim=(-90, 5), legend="tr"),
        ConstellationPlot("con", "Wanted 16-QAM after the receiver", lim=1.5),
    ]
    layout = [["irr", "con"], ["spec", "spec"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("irr", "Image rejection", "dB", ".1f", good=lambda x: x >= 45),
        Readout("sir", "Signal / image on the wanted", "dB", ".1f", good=lambda x: x >= 15),
        Readout("mer", "MER of the wanted", "dB", ".1f"),
        Readout("est", "Estimated g, φ", "", None),
    ]
    challenges = [
        Challenge("Reproduce the book: 0.1 dB and 1° of imbalance give 39.6 dB of image rejection.",
                  lambda s: abs(s.p.g - 0.1) < 0.006 and abs(s.p.ph - 1) < 0.03 and not s.p.corr),
        Challenge("A neighbour 30 dB stronger sits on your mirror and you need 15 dB: reach it "
                  "without correction (find how good the analog must be).",
                  lambda s: s.p.scen.startswith("Capture") and s.p.nlev >= 30 and not s.p.corr
                  and s.r.sir >= 15),
        Challenge("Leave a sloppy 1 dB and 5° and let DSP fix it: image rejection above 50 dB.",
                  lambda s: s.p.g >= 0.99 and s.p.ph >= 4.95 and s.p.corr and s.r.irr >= 50),
    ]
    fs = 16e6
    sps_ = 16

    def setup(self):
        r = np.random.default_rng(4)
        self.qam = cl.get_constellation("16qam")
        h = cl.rrc_taps(0.25, self.sps_, 12)
        self.h = h
        self.sw = self.qam.points[r.integers(0, 16, 1500)]
        self.sn = self.qam.points[r.integers(0, 16, 1500)]
        self.bw = cl.shape(self.sw, h, self.sps_)
        self.bn = cl.shape(self.sn, h, self.sps_)[:len(self.bw)]
        n = np.arange(len(self.bw))
        self.cw = np.exp(-2j * np.pi * 2e6 / self.fs * n)       # wanted at −2 MHz
        self.cn = np.exp(2j * np.pi * 2e6 / self.fs * n)        # neighbour at +2 MHz
        G, P = np.meshgrid(np.linspace(0, 2, 101), np.linspace(0, 10, 101))
        with np.errstate(divide="ignore"):
            self.irr_map = rf.irr_exact(G, P)

    @staticmethod
    def correct(y):
        yi, yq = y.real - y.real.mean(), y.imag - y.imag.mean()
        gs = np.mean(yi * yq) / np.mean(yi ** 2)
        g2 = np.mean(yq ** 2) / np.mean(yi ** 2)
        gc = np.sqrt(max(g2 - gs ** 2, 1e-12))
        g = np.sqrt(g2)
        return yi + 1j * (yq - gs * yi) / gc, g, np.arcsin(np.clip(gs / g, -1, 1))

    def update(self, p):
        capture = p.scen.startswith("Capture")
        if capture:
            x = self.bw * self.cw + 10 ** (p.nlev / 20) * self.bn * self.cn
        else:
            x = self.bw.copy()
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        x = x + 1e-4 * (self.rng.standard_normal(len(x)) + 1j * self.rng.standard_normal(len(x)))
        y = cl.iq_imbalance(x, p.g, p.ph)
        est = None
        if p.corr:
            y, ge, pe = self.correct(y)
            est = f"{20 * np.log10(ge):.2f} dB, {np.degrees(pe):.2f}°"
        # effective image rejection measured with a test tone through the same chain
        n = np.arange(4096)
        tone = np.exp(2j * np.pi * 0.1 * n)
        yt = cl.iq_imbalance(tone, p.g, p.ph)
        if p.corr:
            # apply the correction estimated from the signal
            yi, yq = yt.real, yt.imag
            gs = np.sin(pe) * ge
            gc = np.sqrt(max(ge ** 2 - gs ** 2, 1e-12))
            yt = yi + 1j * (yq - gs * yi) / gc
        Wt = np.abs(np.vdot(tone, yt)) ** 2
        It = np.abs(np.vdot(np.conj(tone), yt)) ** 2
        irr = float(rf.db10(Wt / max(It, 1e-30)))
        # wanted signal: shift to 0, filter, sample
        if capture:
            z = np.convolve(y * np.conj(self.cw), self.h)
        else:
            z = np.convolve(y, self.h)
        d = len(self.h) - 1
        zs = z[d::self.sps_][:len(self.sw)][20:-20]
        ref = self.sw[20:-20]
        gfit = np.vdot(ref, zs) / np.vdot(ref, ref)
        zs = zs / gfit
        mer = float(-rf.db10(np.mean(np.abs(zs - ref) ** 2) / np.mean(np.abs(ref) ** 2)))
        sir = irr - (p.nlev if capture else 0.0)
        pi_ = self.plot("irr")
        pi_.image("m", np.clip(self.irr_map, 10, 70), x=(0, 2), y=(0, 10), cmap="heat", levels=(10, 70),
                  colorbar=True, cbar_label="IRR (dB)")
        pi_.scatter("now", [p.g], [p.ph], color=GREEN, size=15, symbol="d")
        pi_.scatter("book", [0.1], [1.0], color=NAVY, size=10, symbol="+")
        pi_.set_xlim(0, 2)
        pi_.set_ylim(0, 10)
        ps = self.plot("spec")
        f, P = cl.welch_psd(y, self.fs, 2048)
        ps.line("P", f / 1e6, P - P.max(), color=NAVY, width=1.4, fill=-90, fill_alpha=0.08)
        if capture:
            ps.band("w", -2.65, -1.35, color=GREEN, alpha=0.12)
            ps.text("wl", -2, 4, "wanted", color=GREEN, anchor=(0.5, 0), size=9)
            ps.text("nl", 2, 4, "neighbour", color=GRAY, anchor=(0.5, 0), size=9)
            ps.arrow("im", -1.3, max(-85, -irr), "the neighbour's image", color=RED,
                     direction="left", size=10)
        pc = self.plot("con")
        pc.points("z", zs[np.abs(zs) < 2], color=NAVY, size=4, alpha=0.6)
        pc.ideal("i", self.qam.points, color=RED, size=9)
        self.readout(irr=irr, sir=float(sir), mer=mer, est=est if est else "—")

    def story(self, p):
        r = self.r
        s = (f"<p>The Q branch has {v(p.g, '.2f', 'dB')} more gain than I and is "
             f"{v(p.ph, '.2f', '°')} off quadrature. Mathematically the receiver then delivers "
             f"μ·x + ν·x*: the wanted signal plus a weak <b>mirror image</b> of it, "
             f"{v(r.get('irr', 0), '.1f', 'dB')} down. IRR ≈ 4/(ε² + φ²): phase counts more than gain "
             f"(1° is worth about 0.15 dB).</p>")
        if p.scen.startswith("Capture"):
            s += (f"<p>In a wide capture the image of a strong signal at +2 MHz falls on whatever sits "
                  f"at −2 MHz. With the neighbour {v(p.nlev, '.0f', 'dB')} stronger, your signal sees "
                  f"an interferer only {v(r.get('sir', 0), '.1f', 'dB')} below it.</p>")
        else:
            s += ("<p>For a single signal at DC the image is its own mirror: subcarrier +k lands on "
                  "−k, and a single-carrier constellation is skewed into a parallelogram.</p>")
        if p.corr:
            s += (f"<p>{good('Corrected.')} Communication signals are <i>proper</i>: I and Q have equal "
                  f"power and are uncorrelated. Measuring how the receiver broke that symmetry gives "
                  f"g and φ ({r.get('est', '')}) and a 2×2 matrix undoes them.</p>")
        return "<h3>A mirror in the receiver</h3>" + s + keybox(
            "IRR = (1 + 2g cos φ + g²)/(1 − 2g cos φ + g²) ≈ 4/(ε² + φ²). Build it good enough, "
            "then calibrate the rest in DSP.")


# =============================================================================== 5. PA
class PANonlinearity(Experiment):
    title = "Power amplifier: ACLR and EVM"
    blurb = "Drive OFDM into a PA: spectral regrowth outside, error vectors inside."
    book = "sec:ch07:tx"
    controls = [
        Slider("ibo", "Input back-off", 0, 16, 5, step=0.25, unit="dB",
               help="Average input power below the input that would saturate the PA"),
        Slider("pp", "PA smoothness (Rapp p)", 1, 6, 2, step=0.1,
               help="Large p: sharp clipping, like a well-biased solid-state PA"),
        Slider("ampm", "AM/PM at saturation", 0, 30, 12, step=0.5, unit="°"),
        Toggle("limits", "Show 3GPP limits", True),
    ]
    plots = [
        SpectrumPlot("spec", "Spectrum at the PA output", x="frequency (× f_s)", y="PSD (dB)",
                     xlim=(-0.5, 0.5), ylim=(-90, 5), legend="tr"),
        ConstellationPlot("con", "Demodulated subcarriers", lim=1.5),
        Plot("am", "AM/AM", x="input amplitude", y="output amplitude", xlim=(0, 2),
             ylim=(0, 1.2), legend=None),
        Plot("pm", "AM/PM", x="input amplitude", y="phase shift (°)", xlim=(0, 2),
             ylim=(-2, 32), legend=None),
        Plot("curve", "ACLR against output back-off", x="output back-off (dB)", y="ACLR (dB)",
             xlim=(0, 16), ylim=(15, 70), legend="tl"),
    ]
    layout = [["spec", "spec", "con"], ["am", "pm", "curve"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("aclr", "ACLR", "dB", ".1f", good=lambda x: x >= 45),
        Readout("evm", "EVM", "%", ".2f", good=lambda x: x <= 3.5),
        Readout("obo", "Output back-off", "dB", ".1f"),
        Readout("eff", "Class-B efficiency", "%", ".0f"),
    ]
    challenges = [
        Challenge("A phone: meet 3GPP's 30 dB ACLR with as little back-off as you can (output "
                  "back-off ≤ 6 dB).",
                  lambda s: s.r.aclr >= 30 and s.r.obo <= 6),
        Challenge("A base station needs 45 dB. Find the back-off, and read the efficiency left.",
                  lambda s: s.r.aclr >= 45),
        Challenge("256-QAM needs EVM ≤ 3.5 %: meet it with at least 25 % class-B efficiency.",
                  lambda s: s.r.evm <= 3.5 and s.r.eff >= 25),
    ]
    bw, off = 200 / 1024, 1.2 * 200 / 1024

    @lru_cache(maxsize=32)
    def curve(self, pp, ampm):
        x, grid, cfg = ofdm(20, 51)
        out = []
        for b in np.arange(0, 16.1, 1.0):
            y = rf.pa_static(x * 10 ** (-b / 20), 1.0, pp, ampm)
            out.append((-rf.db10(np.mean(np.abs(y) ** 2)), aclr_db(y, self.bw, self.off, 1024)[0]))
        return np.array(out)

    def update(self, p):
        x, grid, cfg = ofdm(40, 51)
        u = x * 10 ** (-p.ibo / 20)
        y = rf.pa_static(u, 1.0, p.pp, p.ampm)
        aclr, f, P = aclr_db(y, self.bw, self.off)
        _, f0, P0 = aclr_db(u, self.bw, self.off)
        evm, Y = evm_pct(y, grid, cfg)
        obo = float(-rf.db10(np.mean(np.abs(y) ** 2)))
        eff = 100 * np.pi / 4 * np.mean(np.abs(y) ** 2) / np.mean(np.abs(y))
        ps = self.plot("spec")
        ps.band("ch", -self.bw / 2, self.bw / 2, color=GREEN, alpha=0.08)
        ps.band("a1", self.off - self.bw / 2, self.off + self.bw / 2, color=RED, alpha=0.06)
        ps.band("a2", -self.off - self.bw / 2, -self.off + self.bw / 2, color=RED, alpha=0.06)
        ps.line("lin", f0, rf.db10(P0 / P0.max()), color=GRAY, width=1.2, name="linear PA")
        ps.line("P", f, rf.db10(P / P.max()), color=NAVY, width=1.6, name="this PA")
        ps.text("al", self.off, -3, "adjacent", color=RED, anchor=(0.5, 0), size=8.5)
        ps.text("al2", -self.off, -3, "adjacent", color=RED, anchor=(0.5, 0), size=8.5)
        pc = self.plot("con")
        pc.points("z", Y.ravel()[::4][:3000], color=NAVY, size=3, alpha=0.45)
        pc.ideal("i", cl.get_constellation("64qam").points, color=RED, size=6)
        a = np.linspace(0, 2, 200)
        sel = slice(2000, 6000)
        pa_ = self.plot("am")
        pa_.scatter("s", np.abs(u[sel]), np.abs(y[sel]), color=NAVY, size=2, alpha=0.3)
        pa_.line("c", a, np.abs(rf.pa_static(a + 0j, 1.0, p.pp, p.ampm)), color=RED, width=2.0)
        pa_.line("lin", [0, 1.2], [0, 1.2], color=GRAY, width=1.0, style="--")
        pa_.vline("rms", 10 ** (-p.ibo / 20), color=GREEN, style=":", width=1.2, label="rms drive",
                  label_pos=0.9)
        pp_ = self.plot("pm")
        pp_.line("c", a, np.degrees(np.angle(rf.pa_static(a + 0j, 1.0, p.pp, p.ampm))), color=RED, width=2.0)
        pp_.vline("rms", 10 ** (-p.ibo / 20), color=GREEN, style=":", width=1.2)
        cv = self.curve(round(p.pp, 2), round(p.ampm, 2))
        pk = self.plot("curve")
        pk.line("c", cv[:, 0], cv[:, 1], color=NAVY, width=2.0, name="this PA")
        pk.scatter("now", [obo], [aclr], color=ORANGE, size=14, symbol="d", name="now")
        if p.limits:
            pk.hline("ue", 30, color=ORANGE, style="--", width=1.2, label="UE 30 dB", label_pos=0.75)
            pk.hline("bs", 45, color=RED, style="--", width=1.2, label="base station 45 dB", label_pos=0.6)
        self.readout(aclr=aclr, evm=evm, obo=obo, eff=float(eff))

    def story(self, p):
        r = self.r
        s = (f"<p>OFDM's envelope has peaks about 10 dB above its average. At "
             f"{v(p.ibo, '.1f', 'dB')} input back-off the PA flattens the biggest ones (AM/AM, bottom "
             f"left) and twists their phase (AM/PM). The distortion products are the signal mixed "
             f"with itself: third order spreads over three times the bandwidth, fifth order five "
             f"times. That is <b>spectral regrowth</b>, measured as an ACLR of "
             f"{v(r.get('aclr', 0), '.1f', 'dB')}.</p>"
             f"<p>Inside the channel the same distortion is an error vector: EVM "
             f"{v(r.get('evm', 0), '.2f', '%')} (NR allows a phone 8 % for 64-QAM, 3.5 % for 256-QAM).</p>"
             f"<p>Back off and both improve, but the efficiency, {v(r.get('eff', 0), '.0f', '%')} for "
             f"an ideal class-B stage, falls with it. Every transmitter lives on that curve.</p>")
        return "<h3>Linearity costs watts</h3>" + s + keybox(
            "ACLR improves about 2 dB per dB of back-off; efficiency falls with it. DPD and Doherty "
            "exist to escape this trade.")


# =============================================================================== 6. efficiency
def envelope(kind, papr_db, n=200_000, seed=41):
    r = np.random.default_rng(seed)
    if kind.startswith("OFDM"):
        x = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
    elif kind.startswith("Single"):
        s = cl.get_constellation("16qam").points[r.integers(0, 16, n // 8)]
        x = cl.shape(s, cl.rrc_taps(0.22, 8, 12), 8)[:n]
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    else:
        return np.ones(n)
    A = 10 ** (papr_db / 20)
    a = np.abs(x)
    x = np.where(a > A, x / np.maximum(a, 1e-12) * A, x)
    return np.abs(x) / np.abs(x).max()


class PAEfficiency(Experiment):
    title = "PA classes, Doherty and ET"
    blurb = "Why a high-PAPR signal wastes power, and the architectures that win it back."
    book = "sec:ch07:tx"
    controls = [
        Choice("sig", "Signal", ["OFDM", "Single-carrier 16-QAM", "Constant envelope"], style="menu"),
        Slider("papr", "PAPR after CFR", 4, 12, 8, step=0.25, unit="dB",
               enabled_if=lambda p: p.sig != "Constant envelope"),
        Slider("alpha", "Doherty transition α", 0.2, 0.8, 0.5, step=0.01,
               help="The peaking device turns on at this fraction of full drive. 0.5 = symmetric"),
        Slider("pout", "Average output power", 1, 80, 40, step=1, unit="W"),
    ]
    plots = [
        Plot("eff", "Efficiency against back-off (shaded: where the signal spends its time)",
             x="output back-off from peak (dB)", y="drain efficiency (%)", xlim=(0, 20), ylim=(0, 90),
             legend="tr"),
        BarPlot("avg", "Average efficiency on this signal", y="%", ylim=(0, 90)),
        Plot("doh", "Doherty load modulation", x="normalised drive", y="normalised current / voltage",
             xlim=(0, 1), ylim=(0, 1.15), legend="tl"),
    ]
    layout = [["eff", "eff"], ["avg", "doh"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("doh", "Doherty average efficiency", "%", ".1f"),
        Readout("cb", "Class-B average efficiency", "%", ".1f"),
        Readout("heat", "Doherty heat at this power", "W", ".0f"),
        Readout("saved", "Saved vs class B", "W", ".0f"),
    ]
    challenges = [
        Challenge("Get the Doherty to 65 % average efficiency on OFDM.",
                  lambda s: s.p.sig == "OFDM" and s.r.doh >= 65,
                  hint="Match α to where the envelope lives; crest-factor reduction helps too."),
        Challenge("At 10 dB PAPR, tune α to beat the symmetric Doherty by at least 3 points.",
                  lambda s: s.p.sig == "OFDM" and abs(s.p.papr - 10) < 0.13 and s.r.gain_vs_sym >= 3),
        Challenge("Show why GSM and Bluetooth use constant-envelope modulation: class B at 78 % or more.",
                  lambda s: s.p.sig == "Constant envelope" and s.r.cb >= 78),
    ]

    @lru_cache(maxsize=64)
    def env(self, sig, papr):
        return envelope(sig, papr)

    def update(self, p):
        vv = self.env(p.sig, round(p.papr, 3))
        curves = [("class A", rf.eff_class_a, GRAY), ("class B", rf.eff_class_b, NAVY),
                  (f"Doherty α = {p.alpha:.2f}", lambda u: rf.eff_doherty(u, p.alpha), GREEN),
                  ("envelope tracking", rf.eff_et, ORANGE)]
        obo = np.linspace(0, 20, 400)
        u = 10 ** (-obo / 20)
        pe = self.plot("eff")
        if p.sig != "Constant envelope":
            h, edges = np.histogram(-20 * np.log10(np.maximum(vv, 1e-4)), bins=80, range=(0, 20),
                                    weights=vv ** 2)
            h = 80 * h / h.max()
            pe.line("hist", edges, h, color=RED, width=0.5, step=True, fill=0, fill_alpha=0.12)
        else:
            pe.vline("ce", 0.05, color=RED, style="-", width=4.0, label="constant envelope: always at peak",
                     label_pos=0.5)
        for name, e, c in curves:
            pe.line(name, obo, 100 * e(u), color=c, width=2.0, name=name)
        avgs = [100 * rf.avg_efficiency(e, vv) for _, e, _ in curves]
        sym = 100 * rf.avg_efficiency(lambda q: rf.eff_doherty(q, 0.5), vv)
        pb = self.plot("avg")
        pb.bars("b", np.arange(4), avgs, width=0.6, colors=[c for *_, c in curves])
        for i, a in enumerate(avgs):
            pb.text(f"t{i}", i, a, f"{a:.0f} %", anchor=(0.5, 1.05), bold=True, size=9)
        pb.set_xticks([(0, "A"), (1, "B"), (2, "Doherty"), (3, "ET")])
        pb.set_xlim(-0.6, 3.6)
        pd = self.plot("doh")
        a = np.linspace(0, 1, 400)
        pd.line("im", a, a, color=NAVY, width=2.0, name="main current")
        pd.line("ip", a, np.where(a < p.alpha, 0, (a - p.alpha) / (1 - p.alpha)), color=RED, width=2.0,
                name="peaking current")
        pd.line("vm", a, np.where(a < p.alpha, a / p.alpha, 1.0), color=GREEN, width=2.0, style="--",
                name="main voltage swing")
        pd.vline("al", p.alpha, color=GRAY, style=":", width=1.2)
        pdc_d = p.pout / max(avgs[2] / 100, 1e-3)
        pdc_b = p.pout / max(avgs[1] / 100, 1e-3)
        self.readout(doh=avgs[2], cb=avgs[1], heat=pdc_d - p.pout, saved=pdc_b - pdc_d,
                     gain_vs_sym=avgs[2] - sym, et=avgs[3], peak2=-20 * np.log10(p.alpha))

    def story(self, p):
        r = self.r
        if p.sig == "Constant envelope":
            s = ("<p>A constant-envelope signal (FM, GMSK, FSK) always sits at the peak, so even a "
                 f"class-B stage reaches its 78.5 % ceiling ({v(r.get('cb', 0), '.0f', '%')}), and a "
                 "switching class-E PA approaches 100 %. That is why GSM, Bluetooth and most IoT "
                 "radios use these modulations: battery life.</p>")
        else:
            s = (f"<p>The shaded histogram is where the signal's power lives: mostly 4–12 dB below its "
                 f"peak. Class B's efficiency falls in proportion to the amplitude, so on this signal "
                 f"it averages only {v(r.get('cb', 0), '.1f', '%')}.</p>"
                 f"<p>The <b>Doherty</b> has a second, 'peaking' amplifier that turns on at "
                 f"α = {v(p.alpha, '.2f')} of full drive and modulates the main device's load, so the "
                 f"main device saturates (max efficiency) already {v(r.get('peak2', 0), '.1f', 'dB')} "
                 f"below peak. Average: {v(r.get('doh', 0), '.1f', '%')}.</p>")
        s += (f"<p>At {v(p.pout, '.0f', 'W')} out, the Doherty dissipates {v(r.get('heat', 0), '.0f', 'W')} "
              f"and saves {v(r.get('saved', 0), '.0f', 'W')} against class B: per carrier, per sector, "
              f"at hundreds of thousands of sites.</p>")
        return "<h3>Efficiency lives in the back-off</h3>" + s + keybox(
            "Average efficiency = E[P_out] / E[P_dc]. Match the architecture's high-efficiency region "
            "to the signal's amplitude distribution.")


# =============================================================================== 7. CFR
class CFR(Experiment):
    title = "Crest-factor reduction"
    blurb = "Clip the peaks, filter the splatter, repeat: trade a little EVM for a lot of efficiency."
    book = "sec:ch07:tx"
    controls = [
        Slider("clip", "Clip level above rms", 3, 12, 9, step=0.25, unit="dB"),
        IntSlider("iters", "Clip-and-filter iterations", 0, 8, 1),
        Toggle("filt", "Filter after clipping", True,
               help="Zero the unused subcarriers after each clip"),
    ]
    plots = [
        Plot("ccdf", "How often the power exceeds a level (CCDF)", x="instantaneous / average power (dB)",
             y="probability", logy=True, xlim=(0, 12), ylim=(1e-5, 1), legend="bl"),
        SpectrumPlot("spec", "Spectrum per subcarrier", x="frequency (× f_s)", y="power (dB)",
                     xlim=(-0.5, 0.5), ylim=(-80, 5), legend="tr"),
        ConstellationPlot("con", "Subcarriers after CFR", lim=1.5),
    ]
    layout = [["ccdf", "con"], ["spec", "spec"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("papr", "PAPR at 10⁻⁴", "dB", ".1f"),
        Readout("evm", "EVM", "%", ".1f", good=lambda x: x <= 8),
        Readout("oob", "Guard-band level", "dB", ".0f", good=lambda x: x < -50),
        Readout("doh", "Doherty efficiency", "%", ".1f"),
    ]
    challenges = [
        Challenge("Get the PAPR below 6.5 dB while keeping EVM within the 8 % allowed for 64-QAM.",
                  lambda s: s.r.papr < 6.5 and s.r.evm <= 8 and s.r.oob < -50,
                  hint="One deep clip costs more EVM than several gentle ones."),
        Challenge("Switch the filter off and see clipping splatter into the guard band (above −40 dB).",
                  lambda s: not s.p.filt and s.r.oob > -40),
        Challenge("For 256-QAM (EVM ≤ 3.5 %), take the PAPR below 7.5 dB.",
                  lambda s: s.r.evm <= 3.5 and s.r.papr < 7.5 and s.r.oob < -50),
    ]

    def update(self, p):
        x, grid, cfg = ofdm(60, 61, n_used=300, lowpass=False)
        u = x.copy()
        mask = np.zeros(cfg.nfft, bool)
        mask[cfg.active] = True
        for _ in range(int(p.iters)):
            A = 10 ** (p.clip / 20) * np.sqrt(np.mean(np.abs(u) ** 2))
            a = np.abs(u)
            u = np.where(a > A, u / np.maximum(a, 1e-12) * A, u)
            if p.filt:
                U = np.fft.fft(u.reshape(-1, cfg.sym_len)[:, cfg.ncp:], axis=1)
                U[:, ~mask] = 0
                sy = np.fft.ifft(U, axis=1)
                u = np.concatenate([sy[:, -cfg.ncp:], sy], axis=1).reshape(-1)
        pp = np.linspace(0, 12, 121)

        def ccdf(w):
            pw = rf.db10(np.abs(w) ** 2 / np.mean(np.abs(w) ** 2))
            s_ = np.sort(pw)
            return 1 - np.searchsorted(s_, pp) / len(s_), s_

        c0, _ = ccdf(x)
        c1, s1 = ccdf(u)
        papr = float(s1[int((1 - 1e-4) * len(s1))])
        evm, Y = evm_pct(u, grid, cfg)
        pc = self.plot("ccdf")
        pc.line("o", pp, np.maximum(c0, 1e-6), color=GRAY, width=1.6, name="original OFDM")
        pc.line("c", pp, np.maximum(c1, 1e-6), color=NAVY, width=2.2, name="after CFR")
        pc.hline("e4", 1e-4, color=GRAY, style=":", width=1.0)
        pc.vline("pp", papr, color=RED, style="--", width=1.2, label=f"PAPR {papr:.1f} dB", label_pos=0.85)

        def symspec(w):
            U = np.fft.fft(w.reshape(-1, cfg.sym_len)[:, cfg.ncp:], axis=1)
            Pq = np.fft.fftshift(np.mean(np.abs(U) ** 2, axis=0))
            return np.fft.fftshift(np.fft.fftfreq(cfg.nfft)), rf.db10(Pq / Pq.max())

        f, P = symspec(u)
        ps = self.plot("spec")
        f0, P0 = symspec(x)
        ps.line("o", f0, np.maximum(P0, -100), color=GRAY, width=1.0, name="original")
        ps.line("c", f, np.maximum(P, -100), color=NAVY, width=1.4, name="after CFR")
        guard = (np.abs(f) > 0.17) & (np.abs(f) < 0.45)
        oob = float(np.mean(P[guard])) if np.any(P[guard] > -100) else -100.0
        ps.hline("g", oob, color=RED, style=":", width=1.2, label="guard band", label_pos=0.02)
        pk = self.plot("con")
        pk.points("z", Y.ravel()[::5][:3000], color=NAVY, size=3, alpha=0.45)
        pk.ideal("i", cl.get_constellation("64qam").points, color=RED, size=6)
        vv = np.abs(u) / np.abs(u).max()
        doh = 100 * rf.avg_efficiency(lambda q: rf.eff_doherty(q, 0.5), vv[::4])
        self.readout(papr=papr, evm=evm, oob=max(oob, -100), doh=float(doh))

    def story(self, p):
        r = self.r
        s = ("<p>OFDM's rare, tall peaks force the PA to back off for all the time it is <i>not</i> "
             "peaking. <b>Crest-factor reduction</b> clips the envelope at a threshold, then removes "
             "the out-of-band products the clipping made by zeroing the unused subcarriers, and "
             "repeats, because filtering regrows some peaks.</p>"
             f"<p>Here the PAPR (at 10⁻⁴ probability) is {v(r.get('papr', 0), '.1f', 'dB')} at an EVM "
             f"cost of {v(r.get('evm', 0), '.1f', '%')}: the clipping noise stays inside the channel, "
             f"on the constellation. A symmetric Doherty on this signal reaches "
             f"{v(r.get('doh', 0), '.1f', '%')}.</p>")
        if not p.filt:
            s += ("<p>" + bad("No filtering:") + " the clipping splatters across the guard band, "
                  "exactly the emission the spectrum mask forbids.</p>")
        return "<h3>Shaving the peaks</h3>" + s + keybox(
            "CFR spends EVM (inside the channel) to buy PAPR, and PAPR buys efficiency. Production "
            "systems use peak cancellation, which does it in one pass.")


# =============================================================================== 8. DPD
class DPD(Experiment):
    title = "Digital predistortion"
    blurb = "Learn the PA's inverse from its own output (indirect learning), one iteration per click."
    book = "sec:ch07:dpd"
    animate = True
    fps = 2
    controls = [
        Slider("ibo", "Input back-off", 4, 20, 11, step=0.5, unit="dB"),
        Toggle("mem", "PA has memory", True, help="Off: a memoryless Rapp + AM/PM PA"),
        Heading("Predistorter (memory polynomial)"),
        IntSlider("K", "Nonlinearity order K", 1, 9, 7),
        IntSlider("M", "Memory depth M", 0, 6, 3),
        Button("step", "▶  Run one ILA iteration", primary=True),
        Button("reset", "Reset the predistorter"),
    ]
    plots = [
        SpectrumPlot("spec", "Output spectrum", x="frequency (× f_s)", y="PSD (dB)", xlim=(-0.5, 0.5),
                     ylim=(-90, 5), legend="tr"),
        Plot("am", "AM/AM: output vs input", x="|x| (input)", y="|y| / gain", xlim=(0, 1.0),
             ylim=(0, 1.0), legend="tl"),
        Plot("ph", "AM/PM: the cloud is memory", x="|x| (input)", y="phase of y/x (°)",
             xlim=(0, 1.0), ylim=(-20, 25), legend=None),
        Plot("hist", "ACLR after each iteration", x="iteration", y="ACLR (dB)", xlim=(-0.5, 10.5),
             ylim=(30, 70), legend=None),
    ]
    layout = [["spec", "spec", "hist"], ["am", "ph", "hist"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("a0", "ACLR, PA alone", "dB", ".1f"),
        Readout("a1", "ACLR with DPD", "dB", ".1f", good=lambda x: x >= 55),
        Readout("it", "Iterations", "", "int"),
        Readout("nc", "Coefficients", "", "int"),
    ]
    challenges = [
        Challenge("Run the predistorter until ACLR reaches 55 dB or more.",
                  lambda s: s.r.a1 >= 55, hint="Click the button (or press Play) a few times."),
        Challenge("Show why memory matters: with M = 0 on the PA with memory, the DPD stalls below "
                  "50 dB after 4 or more iterations; then reach 56 dB with M ≥ 2.",
                  lambda s: s.exp.saw_memless and s.r.a1 >= 56 and s.p.M >= 2 and s.p.mem),
        Challenge("Drive the PA into saturation (back-off 6 dB or less) and see that no predistorter "
                  "can reach 50 dB.",
                  lambda s: s.p.ibo <= 6 and s.r.it >= 4 and s.r.a1 < 50),
    ]
    bw, off = 200 / 1024, 1.2 * 200 / 1024

    def setup(self):
        self.key = None
        self.saw_memless = False

    def pa(self, mem):
        return rf.pa_memory if mem else (lambda u: rf.pa_static(u))

    def reset_state(self, p):
        x, grid, cfg = ofdm(30, 8)
        self.x = x * 10 ** (-p.ibo / 20)
        pa = self.pa(p.mem)
        s = 0.01 * self.x
        self.G = np.vdot(s, pa(s)) / np.vdot(s, s)
        self.z = self.x.copy()
        self.c = None
        self.y0 = pa(self.x)
        self.a0 = aclr_db(self.y0, self.bw, self.off)[0]
        self.hist_a = [self.a0]
        self.key = (p.ibo, p.mem, p.K, p.M)

    def ila_step(self, p):
        pa = self.pa(p.mem)
        K, M = int(p.K), int(p.M)
        y = pa(self.z)
        sl = slice(400, 400 + 16000)
        Phi = rf.mp_basis(y[sl] / self.G, K, M)
        sc = np.sqrt(np.mean(np.abs(Phi) ** 2, axis=0)) + 1e-12
        A = Phi / sc
        c = np.linalg.solve(A.conj().T @ A + 1e-6 * len(A) * np.eye(A.shape[1]),
                            A.conj().T @ self.z[sl]) / sc
        self.c = c
        self.z = rf.mp_basis(self.x, K, M) @ c
        self.hist_a.append(aclr_db(pa(self.z), self.bw, self.off)[0])
        if M == 0 and p.mem and len(self.hist_a) >= 5 and self.hist_a[-1] < 50:
            self.saw_memless = True

    def on_step(self, p):
        if self.key != (p.ibo, p.mem, p.K, p.M):
            self.reset_state(p)
        if len(self.hist_a) <= 10:
            self.ila_step(p)

    def on_reset(self, p):
        self.reset_state(p)

    def tick(self, p):
        self.on_step(p)
        self.update(p)

    def update(self, p):
        if self.key != (p.ibo, p.mem, p.K, p.M):
            self.reset_state(p)
        pa = self.pa(p.mem)
        y = pa(self.z)
        a1, f, P = aclr_db(y, self.bw, self.off)
        _, f0, P0 = aclr_db(self.y0, self.bw, self.off)
        _, fl, Pl = aclr_db(self.G * self.x, self.bw, self.off)
        ref = Pl.max()
        ps = self.plot("spec")
        ps.band("ch", -self.bw / 2, self.bw / 2, color=GREEN, alpha=0.08)
        ps.band("a1", self.off - self.bw / 2, self.off + self.bw / 2, color=RED, alpha=0.06)
        ps.band("a2", -self.off - self.bw / 2, -self.off + self.bw / 2, color=RED, alpha=0.06)
        ps.line("lin", fl, rf.db10(Pl / ref), color=GRAY, width=1.0, name="ideal linear PA")
        ps.line("pa", f0, rf.db10(P0 / ref), color=RED, width=1.2, name=f"PA alone: {self.a0:.1f} dB")
        if self.c is not None:
            ps.line("dpd", f, rf.db10(P / ref), color=GREEN, width=1.6, name=f"with DPD: {a1:.1f} dB")
        sel = slice(3000, 6000)
        g = abs(self.G)
        rot = self.G / g
        pa_ = self.plot("am")
        pa_.line("d", [0, 1], [0, 1], color=GRAY, width=1.0, style="--")
        pa_.scatter("pa", np.abs(self.x[sel]), np.abs(self.y0[sel]) / g, color=RED, size=2, alpha=0.3,
                    name="PA alone")
        pp_ = self.plot("ph")
        pp_.scatter("pa", np.abs(self.x[sel]), np.degrees(np.angle(self.y0[sel] / self.x[sel] / rot)),
                    color=RED, size=2, alpha=0.3)
        if self.c is not None:
            pa_.scatter("dp", np.abs(self.x[sel]), np.abs(y[sel]) / g, color=GREEN, size=2, alpha=0.35,
                        name="with DPD")
            pp_.scatter("dp", np.abs(self.x[sel]), np.degrees(np.angle(y[sel] / self.x[sel] / rot)),
                        color=GREEN, size=2, alpha=0.35)
        ph = self.plot("hist")
        k = np.arange(len(self.hist_a))
        ph.bars("b", k, self.hist_a, width=0.6, colors=[RED] + [GREEN] * (len(k) - 1), base=30)
        ph.hline("45", 45, color=GRAY, style=":", width=1.0)
        ph.set_xticks([(i, "PA" if i == 0 else str(i)) for i in range(11)])
        papr_in = rf.db10(np.max(np.abs(self.x)) ** 2 / np.mean(np.abs(self.x) ** 2))
        papr_z = rf.db10(np.max(np.abs(self.z)) ** 2 / np.mean(np.abs(self.z) ** 2))
        self.readout(a0=self.a0, a1=a1 if self.c is not None else self.a0, it=len(self.hist_a) - 1,
                     nc=int(p.K) * (int(p.M) + 1), dpapr=float(papr_z - papr_in))

    def story(self, p):
        r = self.r
        it = r.get("it", 0)
        s = ("<p>A predistorter puts an approximate <i>inverse</i> of the PA in front of it, so the "
             "pair is linear. Real PAs have <b>memory</b> (bias networks, matching, heat): the AM/AM "
             "and AM/PM 'curves' are clouds. So the inverse is a <b>memory polynomial</b>: "
             f"z[n] = Σ c_km x[n−m]|x[n−m]|^(k−1), {v(r.get('nc', 0), 'd')} coefficients.</p>"
             "<p><b>Indirect learning</b> fits a post-inverse from the PA output (divided by the target "
             "gain) back to its input by least squares, copies it in front, and repeats.</p>")
        if it == 0:
            s += "<p>Press <b>Run one ILA iteration</b> (or Play) and watch the green spectrum appear.</p>"
        else:
            s += (f"<p>After {v(it, 'd')} iterations the ACLR went from {v(r.get('a0', 0), '.1f', 'dB')} "
                  f"to {v(r.get('a1', 0), '.1f', 'dB')}; the drive signal's PAPR grew by "
                  f"{v(r.get('dpapr', 0), '.1f', 'dB')} (DPD pre-expands the peaks, so it is always "
                  f"paired with crest-factor reduction).</p>")
        if p.ibo <= 6:
            s += ("<p>" + bad("Saturated.") + " No predistorter can make the PA deliver power it "
                  "does not have: the pre-expanded peaks drive it harder still, and the learning "
                  "loop can even diverge.</p>")
        return "<h3>Teach the PA's inverse</h3>" + s + keybox(
            "DPD + Doherty: run the PA near saturation for efficiency, and linearise it digitally. "
            "Every macro base station since the late 2000s does both.")


# =============================================================================== the lab
LAB = st.Lab(36, "The RF Transceiver", chapter=7,
             chapter_title="The Radio Transceiver and the Software-Defined Radio",
             experiments=[TwoTone, LineUp, PhaseNoise, IQImbalance, PANonlinearity, PAEfficiency,
                          CFR, DPD])

if __name__ == "__main__":
    st.run(LAB)
