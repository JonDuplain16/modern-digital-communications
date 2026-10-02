"""Lab 21 · Constant Envelope and Modulation Quality: CPM, PAPR and EVM   (Chapter 9)

Run it:      python labs/lab21_cpm_evm.py
Self-test:   python labs/lab21_cpm_evm.py --selftest

Two questions decide what a radio may transmit. Can the power amplifier run flat out? Only
if the envelope is constant, which is why GSM, Bluetooth and DECT use MSK, GMSK and GFSK.
How clean is the transmitted constellation? That is the error vector magnitude, and it caps
how many bits per symbol a link can carry. Eight experiments: phase trajectories, spectra,
MSK as offset QPSK, three receivers, PAPR, an amplifier's spectral regrowth, impairment
signatures and an EVM budget. Library code: commlib/cpm.py.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as ss

import commlib as cl
from commlib import cpm
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, EyePlot, BERPlot, BarPlot, Readout,
                    Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
BOOK_BITS = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1])


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


@lru_cache(maxsize=64)
def min_phase_step(BT, h, sps=8, n=3000):
    """Smallest and median |phase change| over one bit (degrees) for random data."""
    rng = np.random.default_rng(5)
    _, ph = cpm.gmsk_baseband(rng.integers(0, 2, n), sps, BT, h)
    step = np.rad2deg(np.abs(np.diff(ph[sps - 1::sps])))[10:-10]
    return float(step.min()), float(np.median(step))


@lru_cache(maxsize=128)
def obw(BT, h, frac=0.99, sps=8, n=12000):
    """Occupied bandwidth (× Rb) of GMSK/GFSK (BT=None: plain CPFSK/MSK)."""
    rng = np.random.default_rng(11)
    x, _ = cpm.gmsk_baseband(rng.integers(0, 2, n), sps, BT, h)
    return float(cpm.occupied_bandwidth(x, sps, frac, 4096))


# =============================================================================== 1. phase
class PhaseTrajectories(Experiment):
    title = "Phase that remembers"
    blurb = "MSK, GMSK and GFSK: the phase integrates the data, the envelope never moves."
    book = "sec:ch09:constenv"
    controls = [
        Toggle("gauss", "Gaussian pre-filter (GMSK / GFSK)", True),
        Slider("BT", "Bandwidth–time product BT", 0.1, 1.0, 0.5, step=0.01,
               help="Gaussian filter bandwidth × bit period (GSM 0.3, Bluetooth 0.5)",
               enabled_if=lambda p: p.gauss),
        Slider("h", "Modulation index h", 0.2, 1.0, 0.5, step=0.01,
               help="Each bit moves the phase by ±h·180° in total (MSK: h = 0.5)"),
        Choice("pattern", "Bits", ["Book pattern", "Alternating", "Random"]),
        Button("again", "New random bits", enabled_if=lambda p: p.pattern == "Random"),
    ]
    plots = [
        Plot("tree", "Phase trajectory over the MSK phase tree", x="time (bit periods)",
             y="phase (degrees)", xlim=(0, 12), ylim=(-290, 330), legend="bl", legend_cols=2),
        Plot("pulse", "Frequency pulse g(t)·T", x="time (bit periods)", y="g(t)·T",
             xlim=(-2.5, 2.5), ylim=(-0.03, 0.62), legend="tr"),
        Plot("iq", "IQ trajectory", x="I", y="Q", xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect=True,
             legend=None),
    ]
    layout = [["tree", "tree"], ["pulse", "iq"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("full", "Phase change per bit (total)", "°", ".0f"),
        Readout("minstep", "Smallest step within a bit", "°", ".0f",
                help="Worst case over random data: a bit's own phase move after ISI"),
        Readout("bw", "99 % bandwidth", "× Rb", ".2f"),
        Readout("papr", "Peak / average", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Set up classic Bluetooth (BR): BT = 0.5 and h = 0.32.",
                  lambda s: s.p.gauss and abs(s.p.BT - 0.5) < 0.011 and abs(s.p.h - 0.32) < 0.006),
        Challenge("Squeeze the spectrum below 0.85 Rb (99 %) while keeping h = 0.5.",
                  lambda s: abs(s.p.h - 0.5) < 0.006 and s.r.bw < 0.85,
                  hint="Lower BT spreads each bit's phase change over more bits."),
        Challenge("Find the BT where an isolated bit moves the phase by less than 30° within its "
                  "own bit period (h = 0.5).",
                  lambda s: s.p.gauss and abs(s.p.h - 0.5) < 0.006 and s.r.minstep < 30),
    ]
    sps = 64

    def setup(self):
        self.rbits = np.random.default_rng(3).integers(0, 2, 12)

    def on_again(self, p):
        self.rbits = self.rng.integers(0, 2, 12)

    def bits(self, p):
        if p.pattern == "Book pattern":
            return BOOK_BITS
        if p.pattern == "Alternating":
            return np.array([1, 0] * 6)
        return self.rbits

    def update(self, p):
        BT = p.BT if p.gauss else None
        b = self.bits(p)
        n = self.sps
        t = np.arange(len(b) * n) / n
        _, ph_msk = cpm.gmsk_baseband(b, n, None)
        _, ph = cpm.gmsk_baseband(np.r_[b, [0, 0]], n, BT, p.h)
        ph = ph[:len(t)]
        pt = self.plot("tree")
        xs, ys = [], []
        for k in range(len(b)):
            for m in range(-k, k + 1, 2):
                for d in (1, -1):
                    xs += [k, k + 1, np.nan]
                    ys += [m * 90, (m + d) * 90, np.nan]
        pt.line("tree", xs, ys, color=GRAY, width=0.6, alpha=0.5)
        pt.line("msk", t, np.rad2deg(ph_msk), color=NAVY, width=1.8, name="MSK")
        lab = (f"GFSK, BT = {p.BT:.2f}, h = {p.h:.2f}" if p.gauss else f"CPFSK, h = {p.h:.2f}")
        pt.line("g", t, np.rad2deg(ph), color=RED, width=2.4, name=lab)
        for i, bb in enumerate(b):
            pt.text(f"b{i}", i + 0.5, 300, str(int(bb)), anchor=(0.5, 0.5), size=10, bold=True,
                    color=NAVY)
            pt.vline(f"v{i}", i, color=GRAY, style=":", width=0.6)
        tt = np.linspace(-2.5, 2.5, 501)
        pp = self.plot("pulse")
        pp.line("msk", tt, cpm.gmsk_freq_pulse(tt, None), color=NAVY, width=1.8, name="MSK (rectangle)")
        if p.gauss:
            pp.line("g", tt, cpm.gmsk_freq_pulse(tt, p.BT), color=RED, width=2.4,
                    name=f"Gaussian, BT = {p.BT:.2f}")
        x, _ = cpm.gmsk_baseband(np.random.default_rng(1).integers(0, 2, 240), 12, BT, p.h)
        pi_ = self.plot("iq")
        pi_.line("iq", x.real, x.imag, color=NAVY, width=1.0, alpha=0.8)
        k = np.arange(11, len(x), 12)
        pi_.scatter("pts", x[k].real, x[k].imag, color=RED, size=5)
        mn, _ = min_phase_step(BT, round(p.h, 2))
        self.readout(full=180 * p.h, minstep=mn, bw=obw(BT, round(p.h, 2)),
                     papr=float(db10(np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2))))

    def story(self, p):
        mn = self.r.get("minstep", 90)
        s = ("<p>A continuous-phase signal never changes amplitude (bottom right: the IQ trajectory "
             "is a circle) and never jumps in phase. Each bit pushes the phase up for a 1 or down "
             f"for a 0, by ±{v(180 * p.h, '.0f', '°')} in total. With h = 0.5 and a rectangular "
             "frequency pulse this is <b>MSK</b>: the navy path runs exactly along the edges of the "
             "phase tree.</p>")
        if p.gauss:
            s += (f"<p>The Gaussian filter (BT = {v(p.BT)}) rounds every corner: each bit's phase "
                  f"change is smeared over two or three bits (bottom left), which narrows the "
                  f"spectrum. The price is controlled ISI: in an alternating pattern a bit now moves "
                  f"the phase by as little as {v(mn, '.0f', '°')} within its own period.</p>")
        if abs(p.h - 0.5) > 0.01:
            s += (f"<p>With h = {v(p.h)} the red path leaves the MSK tree: "
                  + ("Bluetooth BR uses h ≈ 0.32, a deliberately small deviation."
                     if p.h < 0.5 else "a wider deviation and a wider spectrum.") + "</p>")
        return "<h3>The phase remembers every bit</h3>" + s + keybox(
            "CPM: s(t) = exp(j·2πh·Σ aₙ q(t − nT)). Constant envelope means a saturated, efficient "
            "power amplifier: GSM, Bluetooth, DECT, many IoT radios.")


# =============================================================================== 2. spectra
class Spectra(Experiment):
    title = "Spectra and 99 % bandwidth"
    blurb = "Continuous phase kills the sidelobes; the Gaussian filter narrows the main lobe."
    book = "sec:ch09:constenv"
    controls = [
        Slider("BT", "GMSK bandwidth–time BT", 0.15, 1.0, 0.5, step=0.01),
        Slider("h", "Modulation index h", 0.25, 1.0, 0.5, step=0.01),
        Toggle("refs", "Show QPSK references", True),
    ]
    plots = [
        SpectrumPlot("psd", "Power spectral density at the same bit rate",
                     x="frequency offset (× bit rate Rb)", y="PSD (dB)", xlim=(0, 3),
                     ylim=(-90, 9), legend="tr"),
        Plot("bw", "99 % bandwidth against BT", x="BT", y="99 % bandwidth (× Rb)",
             xlim=(0.1, 1.05), ylim=(0.4, 1.6), legend="br"),
    ]
    layout = [["psd", "bw"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("bw", "99 % bandwidth", "× Rb", ".2f"),
        Readout("bw999", "99.9 % bandwidth", "× Rb", ".2f"),
        Readout("side", "PSD at 1.5 Rb", "dB", ".0f", good=lambda x: x < -50),
        Readout("gsm", "Power outside ±100 kHz (GSM)", "%", ".2f",
                help="GSM: 270.8 kb/s in a 200 kHz channel, i.e. ±0.369 Rb"),
    ]
    challenges = [
        Challenge("Reproduce GSM (BT = 0.3, h = 0.5) and confirm the book's 0.91 Rb.",
                  lambda s: abs(s.p.BT - 0.3) < 0.006 and abs(s.p.h - 0.5) < 0.006),
        Challenge("Keep less than 0.5 % of the power outside GSM's 200 kHz channel.",
                  lambda s: s.r.gsm < 0.5,
                  hint="Both BT and h shape the main lobe."),
        Challenge("Push the PSD at 1.5 Rb below −70 dB with h = 0.5.",
                  lambda s: abs(s.p.h - 0.5) < 0.006 and s.r.side < -70),
    ]
    sps = 8

    @staticmethod
    @lru_cache(maxsize=1)
    def refs():
        sps = 8
        rng = np.random.default_rng(2)
        s = cl.get_constellation("qpsk").points[rng.integers(0, 4, 8000)]
        xr = np.repeat(s, 2 * sps)
        u = np.zeros(len(s) * 2 * sps, complex)
        u[::2 * sps] = s
        xc = ss.fftconvolve(u, cl.rrc_taps(0.35, 2 * sps, span=40))
        _, msk = cpm.gmsk_baseband(rng.integers(0, 2, 16000), sps, None)
        out = []
        for x in (xr, xc, np.exp(1j * msk)):
            f, pp = cpm.psd_normalized(x, sps, 2048)
            out.append((f, db10(pp)))
        return out

    @staticmethod
    @lru_cache(maxsize=32)
    def bw_curve(h):
        BTs = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0])
        return BTs, np.array([obw(float(b), h, n=6000) for b in BTs])

    def update(self, p):
        h = round(p.h, 2)
        rng = np.random.default_rng(4)
        x, _ = cpm.gmsk_baseband(rng.integers(0, 2, 16000), self.sps, p.BT, h)
        f, pp = cpm.psd_normalized(x, self.sps, 2048)
        ps = self.plot("psd")
        if p.refs:
            (f1, p1), (f2, p2), _ = self.refs()
            ps.line("rect", f1, p1, color=GRAY, width=1.2, style=":", name="QPSK, rectangular")
            ps.line("rrc", f2, p2, color=ORANGE, width=1.4, style="--", name="QPSK, RRC β = 0.35")
        _, _, (f3, p3) = self.refs()
        ps.line("msk", f3, p3, color=NAVY, width=1.4, name="MSK")
        ps.line("g", f, db10(pp), color=RED, width=2.4, name=f"GMSK BT = {p.BT:.2f}, h = {h:.2f}")
        bw = cpm.occupied_bandwidth(x, self.sps, 0.99, 4096)
        bw9 = cpm.occupied_bandwidth(x, self.sps, 0.999, 4096)
        ps.band("occ", 0, bw / 2, color=GREEN, alpha=0.10)
        ps.vline("gsm", 0.369, color=PURPLE, style="--", label="GSM channel edge", label_pos=0.25)
        side = float(np.interp(1.5, f, db10(pp)))
        df = f[1] - f[0]
        outside = float(np.sum(pp[np.abs(f) > 0.369]) * df * 100)
        pb = self.plot("bw")
        BTs, bws = self.bw_curve(h)
        pb.line("c", BTs, bws, color=NAVY, width=2.0, name=f"GFSK, h = {h:.2f}")
        pb.scatter("cs", BTs, bws, color=NAVY, size=6)
        pb.hline("msk", obw(None, h), color=GRAY, style=":", label="no Gaussian filter",
                 label_pos=0.6)
        pb.scatter("cur", [p.BT], [bw], color=RED, size=14, name="yours")
        self.readout(bw=bw, bw999=bw9, side=side, gsm=outside)

    def story(self, p):
        bw = self.r.get("bw", 1)
        s = ("<p>All curves carry the same bit rate. Rectangular QPSK (dotted) has phase jumps, "
             "so its sidelobes fall only as f⁻²: 99 % of its power needs many times the bit rate. "
             "MSK's continuous phase makes them fall as f⁻⁴, and the Gaussian filter of GMSK makes "
             "them plunge (red).</p>"
             f"<p>Your signal fits 99 % of its power in {v(bw, '.2f')} Rb (green band, one side "
             f"shown). GSM squeezes 270.8 kb/s into 200 kHz channels (purple line), only about "
             f"0.74 Rb, so even GMSK 0.3 spills a little into the neighbours, which the frequency "
             f"plan absorbs.</p>")
        if p.refs:
            s += ("<p>RRC-shaped QPSK (orange) is the most compact of all, but it is not "
                  "constant-envelope (experiment 5).</p>")
        return "<h3>Smooth phase, tight spectrum</h3>" + s + keybox(
            "Book values (99 %): MSK 1.19 Rb, GMSK 0.5 about 1.03 Rb, GMSK 0.3 about 0.91 Rb.")


# =============================================================================== 3. MSK is OQPSK
@lru_cache(maxsize=64)
def laurent_fit(BT, sps=16, n=400):
    """Fit the transmitted CPM waveform with the main Laurent pulse alone.
    Returns (t, x, s_lin, residual_dB, c0)."""
    rng = np.random.default_rng(9)
    c = rng.integers(0, 2, n)
    a = cpm.msk_precode(c)
    x, _ = cpm.gmsk_baseband(a, sps, BT)
    c0 = cpm.laurent_c0(sps, BT)
    am = 2.0 * a - 1
    b = np.empty(n, complex)
    prev = 1.0
    for k in range(n):
        prev = 1j * am[k] * prev
        b[k] = prev
    u = np.zeros(n * sps, complex)
    u[::sps] = b
    sl = ss.fftconvolve(u, c0)
    lo, hi = 8 * sps, len(x) - 8 * sps
    best = (np.inf, 0, 1.0)
    for d in range(-6 * sps, 2 * sps):
        if lo - d < 0 or hi - d > len(sl):
            continue
        seg = sl[lo - d:hi - d]
        g = np.vdot(seg, x[lo:hi]) / np.vdot(seg, seg)
        r = np.mean(np.abs(x[lo:hi] - g * seg) ** 2) / np.mean(np.abs(x[lo:hi]) ** 2)
        if r < best[0]:
            best = (r, d, g)
    r, d, g = best
    idx = np.arange(len(x)) - d
    ok = (idx >= 0) & (idx < len(sl))
    slin = np.zeros(len(x), complex)
    slin[ok] = g * sl[idx[ok]]
    t = np.arange(len(x)) / sps
    return t, x, slin, float(db10(r)), c0


class MSKisOQPSK(Experiment):
    title = "MSK is offset QPSK"
    blurb = "Laurent's discovery: CPM is (almost) a sum of shaped pulses on I and Q, offset."
    book = "sec:ch09:constenv"
    controls = [
        Toggle("gauss", "Gaussian pre-filter (GMSK)", True),
        Slider("BT", "Bandwidth–time product BT", 0.15, 1.0, 0.3, step=0.01,
               enabled_if=lambda p: p.gauss),
        Slider("start", "Window start", 0, 360, 40, step=1, unit="bits"),
    ]
    plots = [
        Plot("rails", "I and Q rails (solid) and the linear OQPSK model (dashed)",
             x="time (bits)", y="amplitude", ylim=(-1.35, 1.75), legend="tl", legend_cols=3),
        Plot("c0", "The main Laurent pulse C₀(t)", x="time (bits)", y="C₀(t)", xlim=(-3, 3),
             ylim=(-0.05, 1.15), legend="tr"),
        Plot("err", "What the model misses: |x − model|", x="time (bits)", y="error (dB)",
             ylim=(-80, 0), legend=None),
    ]
    layout = [["rails", "rails"], ["c0", "err"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("res", "Linear-model error", "dB", ".1f", good=lambda x: x < -20),
        Readout("dur", "C₀ length (>1 % of peak)", "bits", ".1f"),
    ]
    challenges = [
        Challenge("Confirm that MSK is exactly OQPSK: linear-model error below −60 dB.",
                  lambda s: s.r.res < -60),
        Challenge("Find the smallest BT at which one Laurent pulse still captures the GMSK signal "
                  "to within −20 dB.",
                  lambda s: s.p.gauss and s.r.res < -20 and s.p.BT <= 0.27,
                  hint="The error grows as BT falls; somewhere near a quarter."),
        Challenge("Show the model failing: linear-model error worse than −15 dB.",
                  lambda s: s.r.res > -15),
    ]

    def update(self, p):
        BT = round(p.BT, 2) if p.gauss else None
        t, x, slin, res, c0 = laurent_fit(BT)
        a, b = p.start, p.start + 24
        m = (t >= a) & (t <= b)
        pr = self.plot("rails")
        pr.set_xlim(a, b)
        pr.line("I", t[m], x.real[m], color=NAVY, width=2.4, name="I (actual)")
        pr.line("Q", t[m], x.imag[m], color=RED, width=2.4, name="Q (actual)")
        pr.line("Il", t[m], slin.real[m], color=GOLD, width=1.3, style="--", name="linear model")
        pr.line("Ql", t[m], slin.imag[m], color=GOLD, width=1.3, style="--")
        pr.hline("z", 0, color=GRAY, style="-", width=0.6)
        pc = self.plot("c0")
        c_msk = cpm.laurent_c0(16, None)
        tm = np.arange(len(c_msk)) / 16 - (len(c_msk) - 1) / 32
        pc.line("msk", tm, c_msk, color=NAVY, width=1.8, name="MSK: half-sine, 2 bits")
        if p.gauss:
            tg = np.arange(len(c0)) / 16 - (len(c0) - 1) / 32
            pc.line("g", tg, c0, color=RED, width=2.4, name=f"GMSK, BT = {p.BT:.2f}")
            dur = np.sum(c0 > 0.01 * c0.max()) / 16
        else:
            dur = np.sum(c_msk > 0.01) / 16
        pe = self.plot("err")
        pe.set_xlim(a, b)
        e = db10(np.abs(x[m] - slin[m]) ** 2)
        pe.line("e", t[m], np.maximum(e, -79), color=PURPLE, width=1.4, fill=-80, fill_alpha=0.15)
        self.readout(res=max(res, -120), dur=dur)

    def story(self, p):
        res = self.r.get("res", 0)
        s = ("<p>Laurent (1986) showed that binary CPM with h = ½ is, to an excellent approximation, "
             "a <i>linear</i> modulation: Σ bₖ C₀(t − kT), where bₖ = j·aₖ·bₖ₋₁ alternates between "
             "the real and the imaginary axis. So the I rail carries every other bit, the Q rail "
             "the rest, half a symbol later: <b>offset QPSK</b>.</p>")
        if not p.gauss:
            s += (f"<p>For MSK the model is exact (error {v(res, '.0f', 'dB')}): C₀ is a half-sine two "
                  "bits long, and the dashed model sits on the solid rails. That is why a coherent "
                  "MSK receiver gets BPSK's error rate.</p>")
        else:
            s += (f"<p>For GMSK with BT = {v(p.BT)}, C₀ is smoother and about "
                  f"{v(self.r.get('dur', 3), '.1f')} bits long, and it captures the signal to within "
                  f"{v(res, '.1f', 'dB')}. What it misses (bottom right) is small ISI from the other "
                  f"Laurent pulses. GSM's Viterbi equaliser is built on exactly this model.</p>")
        return "<h3>A nonlinear modulation that is secretly linear</h3>" + s + keybox(
            "With differential precoding, the sign of Re{j⁻ᵏ zₖ} after a filter matched to C₀ "
            "is the data bit. IEEE 802.15.4 (Zigbee) sends OQPSK with half-sine pulses: MSK.")


# =============================================================================== 4. receivers
SIGS = {"MSK": None, "GMSK BT = 0.5": 0.5, "GMSK BT = 0.3": 0.3, "GMSK BT = 0.2": 0.2}


@lru_cache(maxsize=8)
def laurent_rx(BT, sps):
    rng = np.random.default_rng(21)
    c = rng.integers(0, 2, 1000)
    c[0] = 0              # the reference phase depends on the first (precoded) bit: fix it
    x, _ = cpm.gmsk_baseband(cpm.msk_precode(c), sps, BT)
    rx = cpm.LaurentReceiver(sps, BT)
    rx.calibrate(x, c)
    return rx


class Receivers(Experiment):
    title = "Three receivers"
    blurb = "Coherent, differential and discriminator detection: price against performance."
    book = "sec:ch09:constenv"
    heavy = True
    controls = [
        Choice("sig", "Signal", list(SIGS), "GMSK BT = 0.3", style="menu"),
        Choice("rx", "Receiver to inspect", ["Coherent", "Differential", "Discriminator"],
               "Discriminator"),
        Slider("ebn0", "Eb/N0 for the eye", 0.0, 25.0, 12.0, step=0.5, unit="dB"),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run BER again", primary=True),
    ]
    plots = [
        EyePlot("eye", "Receiver output before the decision", yrange=(-1.8, 1.8)),
        Plot("hist", "Decision statistic: bit 1 (navy) and bit 0 (orange)", x="normalised value",
             y="count", xlim=(-1.8, 1.8), legend=None),
        BERPlot("ber", "Bit error rate (Monte Carlo)", x="Eb/N0 (dB)", ylim=(1e-5, 0.5),
                xlim=(0, 14), legend="bl"),
    ]
    layout = [["eye", "ber"], ["hist", "ber"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("min", "Smallest phase step per bit", "°", ".0f"),
        Readout("ber", "BER at the eye's Eb/N0", "", "sci"),
        Readout("pen", "Coherent loss vs BPSK at 10⁻³", "dB", ".2f"),
        Readout("pts", "Points done", "", None),
    ]
    challenges = [
        Challenge("Show that coherent MSK costs nothing: within 0.3 dB of BPSK at 10⁻³.",
                  lambda s: s.p.sig == "MSK" and s.r.pen is not None and abs(s.r.pen) < 0.3),
        Challenge("Find a signal whose discriminator receiver errs even with almost no noise "
                  "(BER above 10⁻³ at 25 dB).",
                  lambda s: s.p.rx == "Discriminator" and s.p.ebn0 >= 24.5 and s.r.ber > 1e-3),
        Challenge("Get the differential receiver below 10⁻³ at Eb/N0 ≤ 10 dB.",
                  lambda s: s.p.rx == "Differential" and s.p.ebn0 <= 10 and 0 <= s.r.ber < 1e-3
                  and s.exp.nb >= 20000),
    ]
    sps = 8

    def setup(self):
        self.run_id = 0
        self.nb = 0

    def on_rerun(self, p):
        self.run_id += 1

    def detect(self, BT, y, a, c, rx):
        if rx == "Coherent":
            r = laurent_rx(BT, self.sps)
            soft = -r.soft(y, len(c))
            return soft, c
        if rx == "Differential":
            return cpm.differential_detect(y, self.sps), a
        return cpm.discriminator_detect(y, self.sps), a

    def update(self, p):
        BT = SIGS[p.sig]
        rng = np.random.default_rng(31)
        n = 6000
        c = rng.integers(0, 2, n)
        c[0] = 0
        a = cpm.msk_precode(c)
        x, _ = cpm.gmsk_baseband(a, self.sps, BT)
        y, _ = cl.awgn_esn0(x, p.ebn0, sps=self.sps, rng=rng)
        soft, ref = self.detect(BT, y, a, c, p.rx)
        soft = soft[:n]
        dec = (soft > 0).astype(int)
        nb = min(len(dec), len(ref)) - 10
        ber = float(np.mean(dec[5:nb] != ref[5:nb]))
        self.nb = nb
        scale = np.median(np.abs(soft)) or 1.0
        z = soft / scale
        pe = self.plot("eye")
        # waveform view of the same statistic: per-sample instantaneous frequency or correlation
        m = self.sps
        yf = ss.lfilter(np.ones(m) / m, 1, y)
        if p.rx == "Discriminator":
            # integrated frequency over the last bit = phase change over one bit
            w = np.angle(yf[m:] * np.conj(yf[:-m]))
            off = m // 2
        elif p.rx == "Differential":
            w = np.imag(yf[m:] * np.conj(yf[:-m]))
            off = m // 2
        else:
            r = laurent_rx(BT, self.sps)
            zz = ss.fftconvolve(y, r.c0[::-1])
            k = np.arange(len(zz)) / self.sps
            w = -np.real(zz * (1j) ** (-(k - r.offset / self.sps)) * r.rot) / np.sum(r.c0 ** 2)
            off = (r.offset - self.sps) % self.sps
        w = w / (np.percentile(np.abs(w), 90) or 1)
        w = w[:1500 * self.sps]
        pe.eye("eye", np.clip(w, -1.79, 1.79), self.sps, n_sym=2, offset=off + 20 * self.sps,
               yrange=(-1.8, 1.8))
        ph = self.plot("hist")
        bins = np.linspace(-1.8, 1.8, 121)
        zc = np.clip(z / (np.percentile(np.abs(z), 90) or 1), -1.79, 1.79)
        r1 = ref[:len(zc)] == 1
        h1, _ = np.histogram(zc[r1], bins)
        h0, _ = np.histogram(zc[~r1], bins)
        ph.line("h1", bins, h1, step=True, color=NAVY, width=1.4, fill=0, fill_alpha=0.25)
        ph.line("h0", bins, h0, step=True, color=ORANGE, width=1.4, fill=0, fill_alpha=0.25)
        ph.vline("thr", 0, color=RED, style="--", label="threshold", label_pos=0.9)
        pb = self.plot("ber")
        eb = np.linspace(0, 14, 113)
        pb.theory("bpsk", eb, cl.ber_bpsk(eb), color=GRAY, style="--", name="BPSK")
        pb.vline("cur", p.ebn0, color=RED, style=":", width=1.0)
        self.sim = {k: ([], []) for k in ("Coherent", "Differential", "Discriminator")}
        mn, _ = min_phase_step(BT, 0.5)
        self.readout(min=mn, ber=ber, pen=None, pts="0")

    def background(self, p):
        BT = SIGS[p.sig]
        rng = np.random.default_rng(500 + self.run_id)
        thorough = p.effort == "Thorough" and not self.quick
        nbits = 8000 if self.quick else (60000 if thorough else 20000)
        reps = 1 if self.quick else (4 if thorough else 2)
        for eb in np.arange(0, 15, 1.0 if not self.quick else 3.0):
            errs = {k: 0 for k in self.sim}
            tot = 0
            for _ in range(reps):
                c = rng.integers(0, 2, nbits)
                c[0] = 0
                a = cpm.msk_precode(c)
                x, _ = cpm.gmsk_baseband(a, self.sps, BT)
                y, _ = cl.awgn_esn0(x, eb, sps=self.sps, rng=rng)
                for k in errs:
                    soft, ref = self.detect(BT, y, a, c, k)
                    n = min(len(soft), len(ref)) - 10
                    errs[k] += int(np.sum((soft[5:n] > 0).astype(int) != ref[5:n]))
                tot += nbits - 15
            yield dict(eb=eb, ber={k: errs[k] / tot for k in errs})
            if all(v_ / tot < 2e-5 for v_ in errs.values()):
                break
        yield dict(done=True)

    def progress(self, p, it):
        pb = self.plot("ber")
        if it.get("done"):
            self.readout(pts=f"{len(self.sim['Coherent'][0])} ✓")
            return
        for k, col in (("Coherent", NAVY), ("Differential", GREEN), ("Discriminator", RED)):
            xs, ys = self.sim[k]
            if it["ber"][k] > 0:
                xs.append(it["eb"])
                ys.append(it["ber"][k])
            if xs:
                pb.sim(k, xs, ys, color=col, name=k.lower())
        xs, ys = self.sim["Coherent"]
        pen = None
        if len(xs) >= 2 and min(ys) < 1e-3 < max(ys):
            ly = np.log10(ys)
            i = int(np.where(ly <= -3)[0][0])
            x3 = np.interp(-3, [ly[i], ly[i - 1]], [xs[i], xs[i - 1]])
            pen = float(x3 - 6.79)
        self.readout(pen=pen, pts=f"{len(self.sim['Coherent'][0])} …")

    def story(self, p):
        mn = self.r.get("min", 90)
        s = ("<p>Three ways to read a CPM signal. The <b>coherent</b> receiver filters with the "
             "Laurent pulse C₀ and needs the carrier phase. The <b>differential</b> detector "
             "compares the phase now with the phase one bit ago. The <b>limiter–discriminator</b> "
             "just measures instantaneous frequency: the cheapest receiver there is (Bluetooth, "
             "DECT, pagers).</p>")
        s += (f"<p>{p.sig}: in the worst data pattern a bit moves the phase only "
              f"{v(mn, '.0f', '°')} within its own period. ")
        if mn < 40:
            s += (bad("That is too little for the noncoherent receivers") + ": their eyes are "
                  "nearly closed by ISI alone, and their BER curves flatten out. Practical GFSK "
                  "radios use BT = 0.5; GSM went coherent with a Viterbi equaliser.</p>")
        else:
            s += ("Enough for the simple receivers: they cost a few dB against coherent detection "
                  "but need no carrier-recovery loop at all.</p>")
        return "<h3>Price against performance</h3>" + s + keybox(
            "Coherent MSK = BPSK. Differential and discriminator detection trade 3–4 dB (more for "
            "small BT) for receivers with no carrier recovery.")


# =============================================================================== 5. PAPR
PAPR_SIGS = {"GMSK": "gmsk", "OQPSK": "oqpsk", "π/4-QPSK": "pi4qpsk", "QPSK": "qpsk",
             "8-PSK": "8psk", "16-QAM": "16qam", "64-QAM": "64qam"}
PAPR_COLS = [RED, BLUE, PURPLE, NAVY, TEAL, ORANGE, GOLD]


@lru_cache(maxsize=32)
def papr_set(beta):
    out = {}
    for name, k in PAPR_SIGS.items():
        x, _ = cpm.shaped_waveform(k, nsym=8000, sps=8, beta=beta, rng=np.random.default_rng(7))
        out[name] = x / np.sqrt(np.mean(np.abs(x) ** 2))
    return out


@lru_cache(maxsize=4)
def ofdm_ref():
    rng = np.random.default_rng(9)
    X = (rng.choice([-1, 1], (300, 256)) + 1j * rng.choice([-1, 1], (300, 256)))
    X[:, 100:156] = 0
    x = np.fft.ifft(np.concatenate([X[:, :128], np.zeros((300, 768)), X[:, 128:]], axis=1), axis=1).ravel()
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


class PAPR(Experiment):
    title = "Envelope and PAPR"
    blurb = "Amplifiers care about peaks, not averages: how far must each signal back off?"
    book = "sec:ch09:constenv"
    controls = [
        Choice("sig", "Highlight", list(PAPR_SIGS), "QPSK", style="menu"),
        Slider("beta", "RRC roll-off β", 0.1, 1.0, 0.25, step=0.05),
        Toggle("ofdm", "Show OFDM for reference", True),
    ]
    plots = [
        Plot("ccdf", "How often the power exceeds its average by x dB (CCDF)",
             x="instantaneous power above average (dB)", y="probability", xlim=(-0.5, 11),
             ylim=(1e-5, 1.5), logy=True, legend="tr"),
        Plot("traj", "IQ trajectory (dashed circle: 0.3 × rms)", x="I", y="Q", xlim=(-2.2, 2.2),
             ylim=(-2.2, 2.2), aspect=True, legend=None),
        Plot("env", "Envelope over 40 symbols", x="time (symbols)", y="|s| / rms", xlim=(0, 40),
             ylim=(0, 2.4), legend=None),
    ]
    layout = [["ccdf", "traj"], ["env", "env"]]
    row_stretch = [3, 2]
    col_stretch = [3, 2]
    readouts = [
        Readout("p3", "PAPR at 10⁻³", "dB", ".2f"),
        Readout("p4", "PAPR at 10⁻⁴", "dB", ".2f"),
        Readout("min", "Smallest envelope", "× rms", ".2f"),
        Readout("eff", "Ideal class-B efficiency", "%", ".0f", good=lambda x: x > 50,
                help="78.5 % at saturation, falling with the back-off needed for the 10⁻⁴ peaks"),
    ]
    challenges = [
        Challenge("Bring QPSK's PAPR at 10⁻⁴ below 4 dB by choosing the roll-off.",
                  lambda s: s.p.sig == "QPSK" and s.r.p4 < 4.0,
                  hint="Gentler filters ring less."),
        Challenge("Find a non-constant-envelope signal whose envelope never falls below half its "
                  "rms value.",
                  lambda s: s.p.sig != "GMSK" and s.r.min >= 0.5),
        Challenge("Keep 16-QAM's ideal class-B efficiency above 40 %.",
                  lambda s: s.p.sig == "16-QAM" and s.r.eff > 40),
    ]

    def update(self, p):
        beta = round(p.beta, 2)
        sigs = papr_set(beta)
        g = np.linspace(-0.5, 11, 231)
        pc = self.plot("ccdf")
        for i, (name, x) in enumerate(sigs.items()):
            cc = np.maximum(cpm.papr_ccdf(x, g), 1e-9)
            pc.line(name, g, cc, color=PAPR_COLS[i], width=3.0 if name == p.sig else 1.2,
                    alpha=1.0 if name == p.sig else 0.7, name=name)
        if p.ofdm:
            pc.line("ofdm", g, np.maximum(cpm.papr_ccdf(ofdm_ref(), g), 1e-9), color=GRAY,
                    style="--", width=1.4, name="OFDM")
        x = sigs[p.sig]
        cc = cpm.papr_ccdf(x, g)
        lc = -np.log10(np.maximum(cc, 1e-12))
        p3 = float(np.interp(3, lc, g))
        p4 = float(np.interp(4, lc, g))
        pc.hline("e4", 1e-4, color=GRAY, style=":", width=0.8)
        pc.vline("p4", p4, color=PAPR_COLS[list(PAPR_SIGS).index(p.sig)], style="--", width=1.0)
        seg = x[800:800 + 300 * 8]
        pt = self.plot("traj")
        pt.line("t", seg.real, seg.imag, color=PAPR_COLS[list(PAPR_SIGS).index(p.sig)], width=0.8,
                alpha=0.8)
        th = np.linspace(0, 2 * np.pi, 120)
        pt.line("c", 0.3 * np.cos(th), 0.3 * np.sin(th), color=GREEN, style="--", width=1.2)
        pe = self.plot("env")
        e = np.abs(x[800:800 + 40 * 8])
        pe.line("e", np.arange(len(e)) / 8, e, color=PAPR_COLS[list(PAPR_SIGS).index(p.sig)],
                width=2.0, fill=0, fill_alpha=0.12)
        pe.hline("rms", 1.0, color=GRAY, style=":", label="rms", label_pos=0.02)
        pe.hline("pk", 10 ** (p4 / 20), color=RED, style="--", label="10⁻⁴ peak", label_pos=0.9)
        mn = float(np.abs(x[200:-200]).min())
        self.readout(p3=p3, p4=p4, min=mn, eff=78.5 * 10 ** (-max(p4, 0) / 20))

    def story(self, p):
        p4, mn, eff = self.r.get("p4", 0), self.r.get("min", 0), self.r.get("eff", 78.5)
        s = (f"<p>A linear amplifier must leave headroom for the peaks. {p.sig}'s power exceeds "
             f"its average by {v(p4, '.1f', 'dB')} one time in 10⁴, so an ideal class-B stage, "
             f"78.5 % efficient at saturation, delivers only about {v(eff, '.0f', '%')} here. In a "
             "handset that is battery life; in a satellite, kilograms of solar panel.</p>")
        if p.sig == "GMSK":
            s += "<p>GMSK is a vertical line at 0 dB: the envelope never moves.</p>"
        elif p.sig in ("OQPSK", "π/4-QPSK"):
            s += (f"<p>{p.sig} never lets I and Q flip together, so the trajectory avoids the "
                  f"origin: the envelope dips only to {v(mn, '.2f')} × rms. The peaks barely "
                  f"change, but a nonlinear amplifier hates zero crossings most.</p>")
        else:
            s += (f"<p>Watch the trajectory pass through the dashed circle: when consecutive "
                  f"symbols are opposite, the filtered signal swings through zero (minimum "
                  f"{v(mn, '.2f')} × rms) and overshoots on the way. Larger β rings less.</p>")
        return "<h3>Peaks cost power</h3>" + s + keybox(
            "Approximate PAPR at 10⁻⁴ with β = 0.25: GMSK 0 dB, OQPSK ~4 dB, QPSK ~4.6 dB, "
            "16-QAM ~6.3 dB, OFDM ~9.6 dB.")


# =============================================================================== 6. PA regrowth
PA_SIGS = ["GMSK BT = 0.3", "OQPSK", "QPSK", "16-QAM", "64-QAM"]


@lru_cache(maxsize=8)
def pa_input(sig):
    sps = 8
    rng = np.random.default_rng(12)
    if sig.startswith("GMSK"):
        x, _ = cpm.gmsk_baseband(rng.integers(0, 2, 2 * 6000), sps // 2, 0.3)
        return x, None, None, sps
    kind = {"OQPSK": "oqpsk", "QPSK": "qpsk", "16-QAM": "16qam", "64-QAM": "64qam"}[sig]
    if kind == "oqpsk":
        x, _ = cpm.shaped_waveform("oqpsk", nsym=6000, sps=sps, beta=0.25, rng=rng)
        return x / np.sqrt(np.mean(np.abs(x) ** 2)), None, None, sps
    c = cl.get_constellation(kind)
    s = c.points[rng.integers(0, c.M, 6000)]
    h = cl.rrc_taps(0.25, sps, 32)
    u = np.zeros(len(s) * sps, complex)
    u[::sps] = s
    x = ss.fftconvolve(u, h)
    r = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / r, s, h, sps


def aclr(x, fs, bw):
    f, p = cpm.psd_normalized(x, fs, 2048)
    df = f[1] - f[0]
    main = np.sum(p[np.abs(f) <= bw / 2]) * df
    adj = np.sum(p[(f > bw / 2) & (f <= 1.5 * bw)]) * df
    return float(db10(adj / main)), f, p


class PARegrowth(Experiment):
    title = "Through a saturating amplifier"
    blurb = "Drive the PA hard: linear signals splatter into the neighbours, GMSK does not care."
    book = "sec:ch09:constenv"
    controls = [
        Choice("sig", "Signal", PA_SIGS, "16-QAM", style="menu"),
        Slider("ibo", "Input back-off from saturation", 0.0, 12.0, 3.0, step=0.25, unit="dB"),
        Slider("p", "Amplifier knee sharpness (Rapp p)", 1.0, 8.0, 2.0, step=0.1,
               help="Large p: a hard limiter with a sharp knee (like a well-linearised PA)"),
    ]
    plots = [
        SpectrumPlot("psd", "Spectrum before (grey) and after (red) the amplifier",
                     x="frequency (× symbol rate)", y="PSD (dB)", xlim=(-3, 3), ylim=(-80, 5),
                     legend=None),
        Plot("amam", "AM/AM (navy) and where the envelope lives (orange)",
             x="input amplitude (× rms)", y="output amplitude", xlim=(0, 3), ylim=(0, 2.2),
             legend=None),
        ConstellationPlot("const", "After the amplifier and matched filter", lim=1.6),
    ]
    layout = [["psd", "psd"], ["amam", "const"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("aclr", "Adjacent-channel leakage", "dB", ".1f", good=lambda x: x < -40),
        Readout("regrow", "Regrowth (vs linear)", "dB", ".1f", good=lambda x: x < 1),
        Readout("evm", "EVM after the PA", "%", ".2f"),
        Readout("obo", "Output back-off", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Saturate the amplifier (back-off ≤ 0.5 dB) with a signal whose spectrum does "
                  "not grow at all (regrowth < 0.5 dB).",
                  lambda s: s.p.ibo <= 0.5 and s.r.regrow < 0.5),
        Challenge("Keep 16-QAM's leakage below −40 dB with no more than 6 dB of back-off.",
                  lambda s: s.p.sig == "16-QAM" and s.p.ibo <= 6 and s.r.aclr < -40,
                  hint="A sharper knee (a linearised PA) behaves better below saturation."),
        Challenge("Find where 64-QAM's EVM from compression alone reaches 3 % (±0.3 %).",
                  lambda s: s.p.sig == "64-QAM" and abs(s.r.evm - 3.0) < 0.3),
    ]

    def update(self, p):
        x, s, h, sps = pa_input(p.sig)
        sat = 10 ** (p.ibo / 20)
        y = cl.rapp_pa(x, sat=sat, p=p.p)
        bw_rs = 1.25 if not p.sig.startswith("GMSK") else 1.82
        a_in, f, p_in = aclr(x, sps, bw_rs)
        a_out, f2, p_out = aclr(y, sps, bw_rs)
        ps = self.plot("psd")
        ps.band("ch", -bw_rs / 2, bw_rs / 2, color=GREEN, alpha=0.08)
        ps.band("adj", bw_rs / 2, 1.5 * bw_rs, color=RED, alpha=0.05)
        ps.text("adjl", bw_rs / 2 + 0.05, -8, "adjacent channel", color=RED, size=8.5)
        ps.line("in", f, db10(p_in / p_in.max()), color=GRAY, width=1.4)
        ps.line("out", f2, db10(p_out / p_in.max()), color=RED, width=2.0)
        pa = self.plot("amam")
        ai = np.linspace(0, 3, 300)
        pa.line("lin", ai, ai, color=GRAY, style="--", width=1.4, name="linear")
        pa.line("pa", ai, ai / (1 + (ai / sat) ** (2 * p.p)) ** (1 / (2 * p.p)), color=NAVY,
                width=2.4, name="Rapp PA")
        hist, edges = np.histogram(np.abs(x), bins=60, range=(0, 3))
        pa.line("hist", edges, 1.2 * hist / hist.max(), step=True, color=ORANGE, width=1.0,
                fill=0, fill_alpha=0.2, name="input envelope (histogram)")
        pa.hline("sat", sat, color=RED, style="--", label="saturation", label_pos=0.85)
        obo = float(db10(sat ** 2 / np.mean(np.abs(y) ** 2)))
        pc = self.plot("const")
        evm = "—"
        if s is not None:
            z = ss.fftconvolve(y, h)[len(h) - 1::sps][:len(s)]
            z = z[20:-20]
            ref = s[20:-20]
            g = np.vdot(ref, z) / np.vdot(ref, ref)
            z = z / g
            evm = 100 * cpm.evm(z, ref)
            pc.points("z", z, color=NAVY, size=3, alpha=0.35)
            pc.ideal("i", np.unique(np.round(s, 6)), color=RED)
            pc.set_title("After the amplifier and matched filter")
        else:
            seg = y[1000:1000 + 2000]
            pc.points("z", seg / np.sqrt(np.mean(np.abs(seg) ** 2)), color=NAVY, size=2, alpha=0.4)
            pc.set_title("Trajectory after the amplifier (constant envelope)")
        self.readout(aclr=a_out, regrow=a_out - a_in, evm=evm, obo=obo)

    def story(self, p):
        rg, a = self.r.get("regrow", 0), self.r.get("aclr", 0)
        s = ("<p>The amplifier is linear for small signals and flattens at saturation (bottom "
             "left). The orange histogram shows where your signal's envelope spends its time; "
             f"everything beyond the red line is squashed. With {v(p.ibo, '.1f', 'dB')} of back-off "
             f"the adjacent channel sees {v(a, '.1f', 'dB')}, ")
        s += (good("no worse than the clean signal.") if rg < 0.5 else
              bad(f"{rg:.1f} dB worse than the clean signal: spectral regrowth.")) + "</p>"
        if p.sig.startswith("GMSK"):
            s += ("<p>GMSK's envelope is constant, so compression changes nothing but the power: "
                  "run the amplifier into saturation, efficiently, and the spectrum stays put.</p>")
        else:
            s += ("<p>Squashing the peaks of a varying envelope is a nonlinearity, and a "
                  "nonlinearity widens the spectrum (third-order products spill into the "
                  "neighbours) while the constellation corners shrink. Regulators limit the first, "
                  "EVM specifications the second, and digital predistortion fights both.</p>")
        return "<h3>Why constant envelope matters</h3>" + s + keybox(
            "Every dB of back-off is efficiency lost; every dB less is regrowth gained. "
            "Constant-envelope signals escape the trade.")


# =============================================================================== 7. EVM signatures
QAMS = {"16-QAM": ("16qam", -19.0), "64-QAM": ("64qam", -27.0), "256-QAM": ("256qam", -32.0),
        "1024-QAM": ("1024qam", -35.0)}
MCS = [(-5, "BPSK 1/2"), (-10, "QPSK 1/2"), (-13, "QPSK 3/4"), (-16, "16-QAM 1/2"),
       (-19, "16-QAM 3/4"), (-22, "64-QAM 2/3"), (-25, "64-QAM 3/4"), (-27, "64-QAM 5/6"),
       (-30, "256-QAM 3/4"), (-32, "256-QAM 5/6"), (-35, "1024-QAM")]


def apply_impairments(s, p, rng, which=None):
    """Apply the selected impairment(s) to symbols s. which=None: all of them."""
    y = s.copy()
    on = lambda k: which is None or which == k
    if on("pa") and p.ibo < 19.9:
        y = cpm.rapp_pa_shaped(y, p.ibo)
    if on("pn") and p.pn > 0:
        y = cpm.phase_noise_awgn(y, p.pn, rng)
    if on("iq") and (p.iqg > 0 or p.iqp > 0):
        y = cl.iq_imbalance(y, p.iqg, p.iqp)
        # an analyser removes the average gain; keep the parallelogram
        y = y / (np.vdot(s, y) / np.vdot(s, s))
    if on("cfo") and p.cfo > 0:
        y = y * np.exp(1j * 2 * np.pi * p.cfo * 1e-6 * np.arange(len(y)))
    if on("dc") and p.dc > 0:
        y = cpm.dc_offset(y, p.dc)
    if on("awgn"):
        y, _ = cl.awgn_esn0(y, p.snr, rng=rng, es=1.0)
    return y


class EVMSignatures(Experiment):
    title = "EVM: impairment signatures"
    blurb = "Every transmitter flaw leaves a fingerprint on the constellation and a line in the budget."
    book = "sec:ch09:evm"
    controls = [
        Choice("con", "Constellation", list(QAMS), "64-QAM"),
        Heading("Impairments"),
        Slider("snr", "Thermal SNR", 10.0, 50.0, 40.0, step=0.5, unit="dB"),
        Slider("pn", "Phase noise (rms)", 0.0, 5.0, 1.0, step=0.05, unit="°"),
        Slider("iqg", "IQ gain imbalance", 0.0, 2.0, 0.0, step=0.05, unit="dB"),
        Slider("iqp", "IQ phase error", 0.0, 10.0, 0.0, step=0.1, unit="°"),
        Slider("ibo", "PA input back-off (20 = linear)", 0.0, 20.0, 20.0, step=0.25, unit="dB"),
        Slider("cfo", "Residual frequency offset", 0.0, 30.0, 0.0, step=0.5, unit="ppm of Rs"),
        Slider("dc", "Carrier leakage", 0.0, 0.2, 0.0, step=0.005, unit="× rms"),
        Button("again", "New symbols"),
    ]
    plots = [
        ConstellationPlot("const", "Measured constellation", lim=1.45),
        Plot("diag", "Error against symbol amplitude (diagnostic)", x="|ideal symbol|",
             y="|error vector|", xlim=(0, 1.5), legend=None),
        BarPlot("bud", "EVM budget: each impairment alone (dB)", x="", y="EVM (dB)"),
    ]
    layout = [["const", "diag"], ["const", "bud"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("evm", "EVM", "%", ".2f"),
        Readout("evmdb", "EVM", "dB", ".1f"),
        Readout("snr", "Effective SNR", "dB", ".1f"),
        Readout("mcs", "Best 802.11 rate passed", "", None),
    ]
    challenges = [
        Challenge("Pass the 1024-QAM limit (−35 dB) with at least 0.5° of phase noise.",
                  lambda s: s.p.con == "1024-QAM" and s.p.pn >= 0.5 and s.r.evmdb < -35),
        Challenge("Check EVM ≈ σφ: phase noise alone of 2° should give about 3.5 % (SNR ≥ 45 dB, "
                  "nothing else on).",
                  lambda s: abs(s.p.pn - 2) < 0.06 and s.p.snr >= 45 and s.p.iqg == 0
                  and s.p.iqp == 0 and s.p.ibo >= 19.9 and s.p.cfo == 0 and s.p.dc == 0
                  and 3.2 < s.r.evm < 3.8),
        Challenge("Find the PA back-off at which compression alone costs 64-QAM −30 dB (±1 dB).",
                  lambda s: s.p.con == "64-QAM" and s.exp.terms.get("PA", 0) is not None
                  and abs(s.exp.terms.get("PA", 0) + 30) < 1),
    ]
    N = 4000

    def setup(self):
        self.seed = 1
        self.terms = {}

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        key, limit = QAMS[p.con]
        c = cl.get_constellation(key)
        rng = np.random.default_rng(self.seed)
        s = c.points[rng.integers(0, c.M, self.N)]
        y = apply_impairments(s, p, rng)
        e = cpm.evm(y, s)
        terms = {}
        for k, lab in (("pn", "phase noise"), ("iq", "IQ"), ("pa", "PA"), ("cfo", "freq. offset"),
                       ("dc", "leakage")):
            active = {"pn": p.pn > 0, "iq": p.iqg > 0 or p.iqp > 0, "pa": p.ibo < 19.9,
                      "cfo": p.cfo > 0, "dc": p.dc > 0}[k]
            if active:
                yy = apply_impairments(s, p, np.random.default_rng(7), which=k)
                terms[lab] = float(cpm.evm_db(yy, s))
        terms["thermal"] = -p.snr
        self.terms = {("PA" if k == "PA" else k): v_ for k, v_ in terms.items()}
        pc = self.plot("const")
        pc.points("y", y, color=NAVY, size=2 if c.M >= 256 else 3, alpha=0.35)
        pc.ideal("i", c.points, color=RED)
        pd = self.plot("diag")
        amp = np.abs(s) + 0.01 * rng.standard_normal(len(s))
        err = np.abs(y - s)
        pd.scatter("e", amp, err, color=NAVY, size=3, alpha=0.3)
        pd.set_ylim(0, max(0.05, float(np.percentile(err, 99.5)) * 1.2))
        pb = self.plot("bud")
        names = list(terms) + ["total"]
        vals = list(terms.values()) + [float(cpm.evm_db(y, s))]
        x = np.arange(len(names))
        cols = [GRAY if n_ == "thermal" else ORANGE for n_ in names[:-1]] + \
               [GREEN if vals[-1] < limit else RED]
        base = -60
        pb.bars("b", x, vals, base=base, width=0.6, colors=cols)
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.1f}", anchor=(0.5, 1.1), size=8.5, bold=(i == len(vals) - 1))
        pb.hline("lim", limit, color=RED, style="--", label=f"{p.con} limit {limit:.0f} dB",
                 label_pos=0.02)
        pb.set_xticks([(i, n_) for i, n_ in enumerate(names)])
        pb.set_xlim(-0.6, len(names) - 0.4)
        pb.set_ylim(base, max(-5, max(vals) + 6))
        edb = 20 * np.log10(e)
        passed = [n_ for lim_, n_ in MCS if edb < lim_]
        self.readout(evm=100 * e, evmdb=edb, snr=-edb, mcs=passed[-1] if passed else "none")

    def story(self, p):
        e, edb = self.r.get("evm", 0), self.r.get("evmdb", 0)
        worst = max(((k, v_) for k, v_ in self.terms.items()), key=lambda kv: kv[1], default=("thermal", 0))
        s = (f"<p>EVM is the rms distance between where symbols land and where they should be, "
             f"relative to the signal: here {v(e, '.2f', '%')} ({v(edb, '.1f', 'dB')}). When the "
             f"errors are noise-like, SNR ≈ 1/EVM², so EVM is an SNR in disguise.</p>"
             f"<p>The biggest contributor now is {v(worst[0])}. Each flaw has a fingerprint: phase "
             "noise draws arcs whose error grows with amplitude (right, a rising wedge); IQ "
             "imbalance squashes the grid into a parallelogram; PA compression pulls in only the "
             "outer corners; a residual frequency offset rotates the whole grid; carrier leakage "
             "shifts it.</p>")
        return "<h3>Reading a constellation like a doctor</h3>" + s + keybox(
            "Independent impairments add as error powers: EVM_total² = Σ EVMᵢ². 1° rms of phase "
            "noise alone is about 1.7 % (−35 dB).")


# =============================================================================== 8. EVM budget
class EVMBudget(Experiment):
    title = "An EVM budget for 1024-QAM"
    blurb = "Close the transmitter budget, then see how its EVM caps every link it serves."
    book = "sec:ch09:evm"
    controls = [
        Choice("target", "Constellation", list(QAMS), "1024-QAM"),
        Heading("Transmitter contributions"),
        Slider("pn", "Phase noise (rms)", 0.1, 2.0, 0.5, step=0.01, unit="°"),
        Slider("irr", "Image rejection", 25.0, 60.0, 45.0, step=0.5, unit="dB"),
        Slider("pa", "PA (after DPD)", -50.0, -25.0, -38.0, step=0.5, unit="dB"),
        Slider("dac", "DAC, clock jitter, analog noise", -50.0, -30.0, -40.0, step=0.5,
               unit="dB"),
        Heading("The link"),
        Slider("rx", "Receiver thermal SNR", 15.0, 50.0, 38.0, step=0.5, unit="dB"),
    ]
    plots = [
        BarPlot("bars", "Contributions (error powers add)", x="", y="EVM (dB)"),
        Plot("cap", "Effective SNR against the receiver's thermal SNR", x="thermal SNR (dB)",
             y="effective SNR (dB)", xlim=(15, 50), ylim=(15, 50), legend="tl"),
    ]
    layout = [["bars", "cap"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("tot", "Transmitter EVM", "dB", ".1f"),
        Readout("margin", "Margin to the limit", "dB", ".1f", good=lambda x: x >= 0),
        Readout("eff", "Effective SNR at the receiver", "dB", ".1f"),
        Readout("cap", "SNR ceiling (perfect receiver)", "dB", ".1f"),
    ]
    challenges = [
        Challenge("The book's fix: back the PA off to −40 dB (others as in the book) and pass "
                  "with a margin of at least 0 dB.",
                  lambda s: s.p.target == "1024-QAM" and abs(s.p.pa + 40) < 0.3
                  and abs(s.p.pn - 0.5) < 0.02 and abs(s.p.irr - 45) < 0.3 and abs(s.p.dac + 40) < 0.3
                  and s.r.margin >= 0),
        Challenge("Leave the PA at −38 dB and close the budget with a better oscillator only "
                  "(IRR 45 dB, DAC −40 dB).",
                  lambda s: s.p.target == "1024-QAM" and abs(s.p.pa + 38) < 0.3
                  and abs(s.p.irr - 45) < 0.3 and abs(s.p.dac + 40) < 0.3 and s.r.margin >= 0),
        Challenge("With the transmitter just passing 1024-QAM, find how strong the receiver's SNR "
                  "must be for an effective SNR of 33 dB.",
                  lambda s: s.p.target == "1024-QAM" and s.r.margin >= 0 and s.r.margin < 1
                  and abs(s.r.eff - 33) < 0.3),
    ]

    def update(self, p):
        _, limit = QAMS[p.target]
        terms = {"phase noise": float(20 * np.log10(np.deg2rad(p.pn))),
                 "IQ image": -p.irr, "PA": p.pa, "DAC / analog": p.dac}
        tot = float(cpm.evm_budget_db(*terms.values()))
        names = list(terms) + ["total"]
        vals = list(terms.values()) + [tot]
        pb = self.plot("bars")
        base = -60
        x = np.arange(len(names))
        pb.bars("b", x, vals, base=base, width=0.6,
                colors=[ORANGE] * len(terms) + [GREEN if tot <= limit else RED])
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.1f} dB\n{100 * 10 ** (val / 20):.2f} %",
                    anchor=(0.5, 1.05), size=8.5, bold=(i == len(vals) - 1))
        pb.hline("lim", limit, color=RED, style="--", label=f"limit {limit:.0f} dB", label_pos=0.02)
        pb.set_xticks([(i, n_) for i, n_ in enumerate(names)])
        pb.set_xlim(-0.6, len(names) - 0.4)
        pb.set_ylim(base, -15)
        pc = self.plot("cap")
        snr = np.linspace(15, 50, 141)
        pc.line("ideal", snr, snr, color=GRAY, style=":", name="perfect transmitter")
        for e_tx, col in ((-30, ORANGE), (-40, GREEN)):
            pc.line(f"r{e_tx}", snr, cpm.snr_eff_db(snr, e_tx), color=col, width=1.0,
                    name=f"TX EVM {e_tx} dB")
        pc.line("you", snr, cpm.snr_eff_db(snr, tot), color=NAVY, width=2.6,
                name=f"your TX ({tot:.1f} dB)")
        eff = float(cpm.snr_eff_db(p.rx, tot))
        pc.scatter("pt", [p.rx], [eff], color=RED, size=13)
        pc.hline("capl", -tot, color=NAVY, style="--", width=1.0)
        self.readout(tot=tot, margin=limit - tot, eff=eff, cap=-tot)

    def story(self, p):
        tot, m, eff = self.r.get("tot", 0), self.r.get("margin", 0), self.r.get("eff", 0)
        s = ("<p>A transmitter EVM budget lists every independent source of error and adds their "
             "<i>powers</i>. Phase noise of σ° rms contributes 20·log₁₀(σ in radians), an image "
             "rejection of R dB contributes −R dB, and so on.</p>"
             f"<p>Your total is {v(tot, '.1f', 'dB')}: "
             + (good(f"it passes with {m:.1f} dB to spare.") if m >= 0 else
                bad(f"it fails by {-m:.1f} dB.")) +
             " The book's budget (0.5°, 45 dB, PA −38 dB, DAC −40 dB) totals −34.4 dB and fails; "
             "backing the PA off to −40 dB passes with no margin at all.</p>"
             f"<p>Right: the transmitter's EVM is a ceiling. However close the client sits, the "
             f"effective SNR cannot exceed {v(-tot, '.1f', 'dB')}; at your receiver SNR of "
             f"{v(p.rx, '.1f', 'dB')} it is {v(eff, '.1f', 'dB')}.</p>")
        return "<h3>Closing the budget</h3>" + s + keybox(
            "1/SNR_eff = 1/SNR_rx + EVM_tx² (+ EVM_rx²). Wi-Fi 6's 1024-QAM needs about −35 dB at "
            "the transmitter.")


# =============================================================================== the lab
LAB = st.Lab(21, "Constant Envelope, PAPR and EVM", chapter=9,
             chapter_title="Digital Modulation and Optimal Detection",
             experiments=[PhaseTrajectories, Spectra, MSKisOQPSK, Receivers, PAPR, PARegrowth,
                          EVMSignatures, EVMBudget])

if __name__ == "__main__":
    st.run(LAB)
