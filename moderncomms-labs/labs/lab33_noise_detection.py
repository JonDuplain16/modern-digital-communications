"""Lab 33 · Noise and Detection   (Chapter 3)

Run it:      python labs/lab33_noise_detection.py
Self-test:   python labs/lab33_noise_detection.py --selftest

Every number in a link budget, every bit-error-rate curve and every detector threshold rests
on a few facts about noise: it is Gaussian, filtering shapes its spectrum, bandpass noise is two
independent baseband halves, every stage of a receiver adds its own, and deciding "signal or no
signal" trades detections against false alarms. Nine experiments build each fact from random
numbers and then use it: a Friis cascade you can rearrange, a simulated Y-factor measurement,
a Neyman-Pearson detector with a threshold you set, and a cyclostationary detector that finds a
BPSK signal 15 dB below the noise.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as sps
from scipy import stats

import commlib as cl
from commlib import noise as nz
from commlib import spectral as sp
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


def db(x, floor=1e-30):
    return 10 * np.log10(np.maximum(x, floor))


# =============================================================================== 1. CLT
SOURCES = {
    # name: (sampler of zero-mean unit-variance terms, skewness, excess kurtosis)
    "Uniform": (lambda r, s: r.uniform(-np.sqrt(3), np.sqrt(3), s), 0.0, -1.2),
    "Coin flip (±1)": (lambda r, s: r.choice([-1.0, 1.0], s), 0.0, -2.0),
    "Exponential": (lambda r, s: r.exponential(1.0, s) - 1.0, 2.0, 6.0),
    "Sine-wave sample": (lambda r, s: np.sqrt(2) * np.sin(2 * np.pi * r.random(s)), 0.0, -1.5),
}


class CentralLimit(Experiment):
    title = "Why noise is Gaussian"
    blurb = "Add up random things of almost any shape and a bell curve appears."
    book = "sec:ch03:gauss"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Choice("src", "Each term is", list(SOURCES), style="menu",
               help="The distribution of one microscopic contribution"),
        IntSlider("n", "Terms added", 1, 60, 1,
                  help="Thermal noise adds up ~10²³ electrons; here you add n"),
        Button("clear", "Start the histogram again"),
    ]
    plots = [
        Plot("hist", "Histogram of the normalised sum", x="value (standard deviations)",
             y="probability density", xlim=(-4.5, 4.5), ylim=(0, 0.75), legend="tr"),
        Plot("tail", "The tail P(X > x), where errors live", x="x (standard deviations)",
             y="probability", logy=True, xlim=(0, 5), ylim=(1e-6, 1), legend="bl"),
    ]
    layout = [["hist", "tail"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("count", "Samples so far", "", "int"),
        Readout("skew", "Skewness", "", ".3f", good=lambda x: abs(x) < 0.3,
                help="Theory for the sum: (one term's skewness)/√n. Gaussian: 0"),
        Readout("kurt", "Excess kurtosis", "", ".3f", good=lambda x: abs(x) < 0.1,
                help="Theory for the sum: (one term's excess kurtosis)/n. Gaussian: 0"),
        Readout("t3", "P(X > 3) / Q(3)", "", ".2f",
                help="How well the tail at three sigma matches the Gaussian"),
    ]
    challenges = [
        Challenge("The classic trick: the fewest uniform terms whose sum has |excess kurtosis| ≤ 0.1.",
                  lambda s: s.p.src == "Uniform" and s.p.n == 12,
                  hint="Excess kurtosis of the sum is −1.2/n."),
        Challenge("Make a sum of exponentials Gaussian enough: skewness below 0.3.",
                  lambda s: s.p.src == "Exponential" and 2 / np.sqrt(s.p.n) < 0.3),
        Challenge("A sine-wave interferer is not noise: with one term its kurtosis is −1.5. "
                  "How many must you add to reach |excess kurtosis| < 0.05?",
                  lambda s: s.p.src == "Sine-wave sample" and 1.5 / s.p.n < 0.05),
    ]
    edges = np.linspace(-4.5, 4.5, 181)
    per_tick = 20000

    def setup(self):
        self.reset_acc()

    def reset_acc(self):
        self.counts = np.zeros(len(self.edges) - 1)
        self.tail_counts = np.zeros(51)
        self.m = np.zeros(4)                         # sums of x, x², x³, x⁴
        self.total = 0

    def on_clear(self, p):
        self.reset_acc()

    def add(self, p):
        draw = SOURCES[p.src][0]
        x = draw(self.rng, (self.per_tick, p.n)).sum(axis=1) / np.sqrt(p.n)
        self.counts += np.histogram(x, self.edges)[0]
        self.tail_counts += (x[:, None] > np.linspace(0, 5, 51)[None, :]).sum(axis=0)
        self.m += [x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum()]
        self.total += len(x)

    def update(self, p):
        if (p.src, p.n) != getattr(self, "_key", None):
            self.reset_acc()
            self._key = (p.src, p.n)
        self.add(p)
        self.draw(p)

    def tick(self, p):
        if self.total < 5_000_000:
            self.add(p)
        self.draw(p)

    def draw(self, p):
        w = self.edges[1] - self.edges[0]
        dens = self.counts / max(self.total, 1) / w
        ph = self.plot("hist")
        ph.line("h", self.edges, dens, color=NAVY, width=1.2, step=True, fill=0, fill_alpha=0.30,
                name=f"sum of {p.n} term{'s' if p.n > 1 else ''}")
        xs = np.linspace(-4.5, 4.5, 400)
        ph.line("g", xs, stats.norm.pdf(xs), color=RED, width=2.2, name="Gaussian N(0, 1)")
        ph.set_ylim(0, max(0.75, float(dens.max()) * 1.1))
        xt = np.linspace(0, 5, 51)
        emp = self.tail_counts / max(self.total, 1)
        pt = self.plot("tail")
        pt.line("q", xt, nz.Q(xt), color=RED, width=2.2, name="Gaussian Q(x)")
        pt.scatter("e", xt[emp > 0], emp[emp > 0], color=NAVY, size=6, name="your sum")
        N = max(self.total, 1)
        mu = self.m[0] / N
        var = self.m[1] / N - mu ** 2
        m3 = self.m[2] / N - 3 * mu * self.m[1] / N + 2 * mu ** 3
        m4 = (self.m[3] / N - 4 * mu * self.m[2] / N + 6 * mu ** 2 * self.m[1] / N - 3 * mu ** 4)
        skew = m3 / max(var, 1e-12) ** 1.5
        kurt = m4 / max(var, 1e-12) ** 2 - 3
        t3 = emp[30] / nz.Q(3.0)
        self.readout(count=self.total, skew=skew, kurt=kurt, t3=t3)

    def story(self, p):
        _, s1, k1 = SOURCES[p.src]
        s = (f"<p>Each sample in the histogram is the sum of {v(p.n, 'd')} independent "
             f"\"{p.src.lower()}\" terms, scaled to unit variance. Theory says the sum's skewness is "
             f"{v(s1 / np.sqrt(p.n), '.3f')} and its excess kurtosis {v(k1 / p.n, '.3f')}: both shrink "
             f"towards the Gaussian's zero as n grows. That is the <b>central limit theorem</b>.</p>")
        if p.n == 1:
            s += ("<p>With one term you see the raw shape. Now drag n up and watch it melt into the "
                  "red bell curve.</p>")
        s += ("<p>The right plot is the one engineers care about: the <b>tail</b>, where bit errors "
              "live. A sum of a few terms can look Gaussian in the middle and still have the wrong "
              "tail: a sum of n coin flips can never exceed √n.</p>")
        return "<h3>Many small causes, one bell</h3>" + s + keybox(
            "Thermal noise is Gaussian because it is a sum of very many small independent effects.")


# =============================================================================== 2. Q function
class QTails(Experiment):
    title = "The Q-function and the tails"
    blurb = "Every bit error rate is a Gaussian tail. How far out can a simulation see?"
    book = "sec:ch03:qfunc"
    heavy = True
    controls = [
        Slider("x", "Threshold x", 0.0, 8.0, 3.0, step=0.01,
               unit="σ", help="Distance from the signal to the decision boundary, in noise standard deviations"),
        Choice("nmc", "Monte Carlo samples", ["10⁴", "10⁶", "10⁷"], "10⁶"),
        Button("again", "Run the Monte Carlo again"),
    ]
    plots = [
        Plot("pdf", "N(0, 1) on a log scale, and the tail beyond x", x="value (σ)", y="density",
             logy=True, xlim=(-4, 8), ylim=(1e-16, 1), legend="tr"),
        Plot("q", "Q(x): theory, approximations and Monte Carlo", x="x (σ)", y="probability",
             logy=True, xlim=(0, 8), ylim=(1e-16, 1), legend="bl"),
    ]
    layout = [["pdf", "q"]]
    readouts = [
        Readout("q", "Q(x)", "", "sci"),
        Readout("snr", "SNR = 20·log₁₀(x)", "dB", ".1f"),
        Readout("mc", "Monte Carlo estimate", "", "sci"),
        Readout("need", "Samples for 100 errors", "", "sci"),
    ]
    challenges = [
        Challenge("Optical links promise a BER of 10⁻¹²: find x where Q(x) = 10⁻¹² (within 5 %).",
                  lambda s: abs(np.log10(nz.Q(s.p.x)) + 12) < 0.021),
        Challenge("A KP4 FEC accepts a BER of 2×10⁻⁴ before correction: what SNR is that? "
                  "(Q within 5 %)",
                  lambda s: abs(np.log10(nz.Q(s.p.x) / 2e-4)) < 0.021),
        Challenge("Run out of Monte Carlo: with 10⁶ samples, find an x where the simulation sees "
                  "zero errors although Q(x) is still above 10⁻⁷.",
                  lambda s: s.p.nmc == "10⁶" and s.r.mc == 0 and nz.Q(s.p.x) > 1e-7),
    ]

    def setup(self):
        self.run = 0

    def on_again(self, p):
        self.run += 1

    def update(self, p):
        x = p.x
        y = np.linspace(-4, 8, 600)
        pp = self.plot("pdf")
        pp.line("pdf", y, stats.norm.pdf(y), color=NAVY, width=2.2, name="noise pdf")
        yt = y[y >= x]
        if len(yt) > 1:
            pp.line("tail", yt, stats.norm.pdf(yt), color=RED, width=1.4, fill=1e-16, fill_alpha=0.45,
                    name=f"area = Q({x:.2f})")
        pp.vline("x", x, color=RED, style="--", label=f"x = {x:.2f}", label_pos=0.9)
        xs = np.linspace(0.01, 8, 400)
        pq = self.plot("q")
        pq.line("Q", xs, nz.Q(xs), color=NAVY, width=2.4, name="Q(x)")
        pq.line("ub", xs[xs > 0.5], stats.norm.pdf(xs[xs > 0.5]) / xs[xs > 0.5], color=ORANGE,
                width=1.4, style="--", name="φ(x)/x (upper bound)")
        pq.line("ch", xs, 0.5 * np.exp(-xs ** 2 / 2), color=GRAY, width=1.4, style=":",
                name="½·e^(−x²/2) (Chernoff)")
        pq.scatter("now", [x], [nz.Q(x)], color=RED, size=14, symbol="d")
        pq.hline("res", 1 / self.n(p), color=PURPLE, style=":", label="1 / samples", label_pos=0.75)
        self.mc_errs, self.mc_n = 0, 0
        self.readout(q=float(nz.Q(x)), snr=20 * np.log10(max(x, 1e-6)), mc=None,
                     need=100 / max(float(nz.Q(x)), 1e-300))

    def n(self, p):
        return {"10⁴": 10_000, "10⁶": 1_000_000, "10⁷": 10_000_000}[p.nmc]

    def background(self, p):
        rng = np.random.default_rng(1000 + self.run)
        total = self.n(p) if not self.quick else min(self.n(p), 1_000_000)
        done = 0
        while done < total:
            m = min(1_000_000, total - done)
            e = int(np.count_nonzero(rng.standard_normal(m) > p.x))
            done += m
            yield (e, m)

    def progress(self, p, item):
        e, m = item
        self.mc_errs += e
        self.mc_n += m
        est = self.mc_errs / self.mc_n
        pq = self.plot("q")
        if est > 0:
            pq.scatter("mc", [p.x], [est], color=GREEN, size=11, outline=GREEN,
                       name=f"Monte Carlo ({self.mc_errs} errors)")
        self.readout(mc=est)

    def story(self, p):
        q = float(nz.Q(p.x))
        s = (f"<p>A detector decides wrongly when the noise pushes the sample past the decision "
             f"boundary, here {v(p.x, '.2f')} standard deviations away. The chance is the red area: "
             f"<b>Q(x)</b> = {v(fmt_sci(q))}. That is the SNR {v(20 * np.log10(max(p.x, 1e-6)), '.1f', 'dB')} "
             f"measured as (distance/σ)².</p>")
        need = 100 / max(q, 1e-300)
        s += (f"<p>To <i>measure</i> that by simulation you need about 100 errors, so about "
              f"{v(fmt_sci(need))} samples. Past about 10⁻⁷ no laptop can do it: below the purple line "
              f"the Monte Carlo sees nothing at all. That is why BER curves at 10⁻¹² are always "
              f"theory or extrapolation, and why the Q-function's approximations matter.</p>")
        s += ("<p>Chapter 3's numbers: BER 10⁻¹² needs Q⁻¹ = 7.03 (16.9 dB); 10⁻⁶ needs 4.75 "
              "(13.5 dB); a KP4 FEC that accepts 2×10⁻⁴ needs only 3.54 (11 dB). Every dB matters "
              "because the tail falls like e^(−x²/2).</p>")
        return "<h3>Life in the tails</h3>" + s + keybox(
            "Error probability = Q(distance / σ). Each extra dB of SNR buys a factor of several "
            "in BER, more the further out you are.")


def fmt_sci(x):
    return st.sci(x)


# =============================================================================== 3. Rayleigh / Rice
class RayleighRice(Experiment):
    title = "Rayleigh and Rice envelopes"
    blurb = "Complex Gaussian noise, with and without a steady component: the statistics of fading."
    book = "sec:ch03:complex"
    controls = [
        Slider("K", "Rician K-factor", 0, 20, 0, step=0.1,
               help="K = 0: no line of sight (Rayleigh). Large K: a strong steady component"),
        Slider("depth", "Fade depth", -30, 0, -10, step=0.5, unit="dB",
               help="Relative to the mean power"),
        Slider("doppler", "Doppler spread", 5, 200, 50, step=5, unit="Hz"),
        Button("again", "New fading record"),
    ]
    plots = [
        ConstellationPlot("iq", "Complex samples (I/Q)", lim=2.6),
        Plot("env", "Envelope distribution", x="envelope |z| (unit mean power)", y="density",
             xlim=(0, 3), ylim=(0, 3.2), legend="tr"),
        Plot("time", "A fading signal in time", x="time (ms)", y="power (dB re. mean)",
             xlim=(0, 500), ylim=(-40, 10), legend="bl"),
    ]
    layout = [["iq", "env"], ["time", "time"]]
    col_stretch = [2, 3]
    row_stretch = [3, 2]
    readouts = [
        Readout("pout", "Time in deep fades", "%", ".2f", help="Measured from 200 000 samples"),
        Readout("theory", "Theory P(fade)", "%", ".2f"),
        Readout("kdb", "K-factor", "dB", ".1f"),
        Readout("rate", "Fades per second", "", ".0f"),
    ]
    challenges = [
        Challenge("Rayleigh (K = 0): find the fade depth that the signal is below just 1 % of the time.",
                  lambda s: s.p.K == 0 and abs(s.r.theory - 1.0) < 0.1,
                  hint="P(power < mean·d) = 1 − e^(−d) ≈ d for small d."),
        Challenge("A strong line of sight: find the smallest K where 10 dB fades happen less than 0.1 % of the time.",
                  lambda s: abs(s.p.depth + 10) < 0.01 and s.r.theory < 0.1
                  and s.exp.theory_p(s.p.K - 0.1, -10) >= 0.1,
                  hint="Raise K in small steps while watching the theory readout."),
        Challenge("Count fades: at 200 Hz Doppler, a Rayleigh signal dips 10 dB below its mean more "
                  "than 100 times a second. Show it.",
                  lambda s: s.p.K == 0 and abs(s.p.depth + 10) < 0.01 and s.p.doppler >= 199
                  and s.r.rate > 100),
    ]
    fs = 10_000.0

    def setup(self):
        self.seed = 3

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def theory_p(K, depth_db):
        K = max(K, 0.0)
        return 100 * float(nz.rice_cdf(np.sqrt(10 ** (depth_db / 10)), K))

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        K = p.K
        s = np.sqrt(K / (K + 1))
        sig = np.sqrt(1 / (2 * (K + 1)))
        z = s + sig * (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000))
        a = np.abs(z)
        thr = np.sqrt(10 ** (p.depth / 10))
        pi_ = self.plot("iq")
        th = np.linspace(0, 2 * np.pi, 200)
        pi_.points("z", z[:3000], color=NAVY, size=3, alpha=0.35)
        pi_.line("thr", thr * np.cos(th), thr * np.sin(th), color=RED, width=1.8)
        pi_.scatter("los", [s], [0], color=GREEN, size=12, symbol="+")
        h, e = np.histogram(a, 90, (0, 3), density=True)
        pe = self.plot("env")
        pe.line("h", e, h, color=NAVY, width=1.0, step=True, fill=0, fill_alpha=0.25, name="simulated")
        r = np.linspace(0, 3, 400)
        pe.line("pdf", r, nz.rice_pdf(r, K), color=RED, width=2.2,
                name="Rayleigh pdf" if K == 0 else "Rice pdf")
        pe.vline("thr", thr, color=RED, style="--", label=f"{p.depth:.0f} dB", label_pos=0.9)
        pe.set_ylim(0, max(1.0, float(nz.rice_pdf(r, K).max()) * 1.15))
        # a Jakes-faded record with the same K
        n = 5000
        g = cl.jakes_process(n, p.doppler / self.fs, rng=rng)
        zz = s + np.sqrt(1 / (K + 1)) * g
        pw = 10 * np.log10(np.abs(zz) ** 2 + 1e-12)
        t = np.arange(n) / self.fs * 1e3
        pt = self.plot("time")
        pt.line("p", t, pw, color=NAVY, width=1.4, name="received power")
        below = np.where(pw < p.depth, pw, np.nan)
        if np.isfinite(below).any():
            pt.line("b", t, below, color=RED, width=2.4, name="in a fade")
        pt.hline("d", p.depth, color=RED, style="--")
        pt.hline("m", 0, color=GRAY, style=":", label="mean", label_pos=0.01)
        crossings = np.count_nonzero(np.diff((pw < p.depth).astype(int)) == 1)
        self.readout(pout=100 * np.mean(a < thr), theory=self.theory_p(K, p.depth),
                     kdb=10 * np.log10(K) if K > 0 else "−∞", rate=crossings / (n / self.fs))

    def story(self, p):
        r = self.r
        if p.K == 0:
            s = ("<p>Complex Gaussian noise (or a signal arriving only by scattered paths) has I and Q "
                 "independent and equal: a round cloud. Its envelope is <b>Rayleigh</b> and its phase "
                 "uniform. The cloud is densest near the middle, so deep fades are common.</p>")
        else:
            s = (f"<p>A steady component (green cross) moves the cloud off-centre: K = "
                 f"{v(p.K, '.1f')}, the line-of-sight power over the scattered power. The envelope "
                 f"becomes <b>Rician</b>, narrower and centred near √(K/(K+1)). The origin, where the "
                 f"deep fades live, is now far from most points.</p>")
        s += (f"<p>The signal spends {v(r.get('pout', 0), '.2f', '%')} of the time more than "
              f"{v(-p.depth, '.0f', 'dB')} below its mean (theory "
              f"{v(r.get('theory', 0), '.2f', '%')}). For Rayleigh, 1 − e^(−d): about 10 % for 10 dB, "
              f"1 % for 20 dB. That is the <b>fade margin</b> a link budget must carry (Chapter 11).</p>")
        s += (f"<p>Bottom: the same statistics in time, with a {v(p.doppler, '.0f', 'Hz')} Doppler "
              f"spread. Faster movement, more frequent but shorter fades.</p>")
        return "<h3>Fading is complex noise</h3>" + s + keybox(
            "Rayleigh: |complex Gaussian|. Each 10 dB of extra margin cuts the outage tenfold; a line "
            "of sight (Rice) cuts it far faster.")


# =============================================================================== 4. filtered noise
FILTERS = ["RC (one pole)", "Butterworth", "Gaussian", "Brick wall"]


class FilteredNoise(Experiment):
    title = "Filtered noise and its bandwidth"
    blurb = "White noise through a filter: the PSD becomes |H|², the power N₀·B_N."
    book = "sec:ch03:psd"
    controls = [
        Choice("filt", "Filter", FILTERS, "Butterworth", style="menu"),
        IntSlider("order", "Butterworth order", 1, 8, 2, enabled_if=lambda p: p.filt == "Butterworth"),
        Slider("fc", "3 dB cut-off", 10, 200, 50, step=1, unit="Hz"),
        Heading("Noise on a capacitor (kT/C)"),
        LogSlider("cap", "Sampling capacitor", 0.1, 100, 1.0, unit="pF"),
        Slider("temp", "Temperature", 4, 400, 300, step=1, unit="K"),
    ]
    plots = [
        Plot("path", "Sample paths: white and filtered", x="time (ms)", y="amplitude",
             xlim=(0, 300), ylim=(-4.5, 5.5), legend="tl", legend_cols=2),
        SpectrumPlot("psd", "PSD: measured vs |H(f)|²·N₀", x="frequency (Hz)", y="PSD (dB/Hz)",
                     xlim=(0, 500), ylim=(-50, 6), legend="tr"),
        Plot("neb", "|H(f)|² and its equivalent rectangle", x="frequency (Hz)", y="|H(f)|²",
             xlim=(0, 250), ylim=(0, 1.2), legend="tr"),
        Plot("acf", "Autocorrelation", x="lag (ms)", y="R(τ) / R(0)", xlim=(-60, 60),
             ylim=(-0.4, 1.1), legend="tr"),
    ]
    layout = [["path", "psd"], ["neb", "acf"]]
    readouts = [
        Readout("bn", "Noise bandwidth B_N", "Hz", ".1f"),
        Readout("ratio", "B_N / f_3dB", "", ".3f"),
        Readout("pw", "Measured power / (N₀·B_N)", "", ".3f", good=lambda x: abs(x - 1) < 0.05),
        Readout("ktc", "kT/C noise", "µV rms", ".1f"),
    ]
    challenges = [
        Challenge("Bring B_N within 2 % of the 3 dB bandwidth with the lowest Butterworth order that can.",
                  lambda s: s.p.filt == "Butterworth" and s.p.order == 5),
        Challenge("kT/C: keep a sampling capacitor's noise at 300 K below 35 µV (a 14-bit ADC's "
                  "quantisation noise) with the smallest capacitor that does it (within 10 %).",
                  lambda s: abs(s.p.temp - 300) < 0.5 and s.r.ktc < 35 and s.p.cap < 3.75),
        Challenge("Cool it: get a 1 pF capacitor below 10 µV of noise.",
                  lambda s: abs(s.p.cap - 1.0) < 0.05 and s.r.ktc < 10),
    ]
    fs = 1000.0

    @staticmethod
    @lru_cache(maxsize=1)
    def white():
        return np.random.default_rng(2).standard_normal(1 << 16) * np.sqrt(1000.0 / 2)

    def H2(self, p, f):
        f = np.abs(np.asarray(f, float))
        if p.filt.startswith("RC"):
            return 1 / (1 + (f / p.fc) ** 2)
        if p.filt == "Butterworth":
            return 1 / (1 + (f / p.fc) ** (2 * p.order))
        if p.filt == "Gaussian":
            return np.exp(-np.log(2) * (f / p.fc) ** 2)
        return (f <= p.fc).astype(float)

    def update(self, p):
        w = self.white()                                   # two-sided PSD N0/2 = 1/2 (N0 = 1)
        N = len(w)
        f = np.fft.fftfreq(N, 1 / self.fs)
        y = np.real(np.fft.ifft(np.fft.fft(w) * np.sqrt(self.H2(p, f))))
        t = np.arange(300) * 1e3 / self.fs
        pp = self.plot("path")
        pp.line("w", t, w[:300] / np.sqrt(self.fs / 2), color=GRAY, width=0.8, alpha=0.8,
                name="white (scaled)")
        pp.line("y", t, y[:300] / max(np.std(y), 1e-12), color=NAVY, width=2.0, name="filtered (scaled)")
        fw, pw = sps.welch(y, self.fs, nperseg=2048)
        fr = np.linspace(0, 500, 1001)
        H2 = self.H2(p, fr)
        ps = self.plot("psd")
        ps.line("m", fw, db(pw), color=NAVY, width=1.6, name="Welch estimate (one-sided)")
        ps.line("t", fr, db(H2), color=RED, width=2.0, style="--", name="|H(f)|²·N₀")
        bn = nz.neb_hz(np.linspace(0, 500, 20001), self.H2(p, np.linspace(0, 500, 20001)))
        pn = self.plot("neb")
        pn.line("H2", fr, H2, color=NAVY, width=2.2, fill=0, fill_alpha=0.18, name="|H(f)|²")
        pn.line("rect", [0, bn, bn], [1, 1, 0], color=RED, width=2.0, style="--",
                name=f"rectangle of equal area: B_N = {bn:.1f} Hz")
        pn.vline("f3", p.fc, color=GRAY, style=":", label="f_3dB", label_pos=0.08)
        pn.set_xlim(0, min(500, 5 * p.fc))
        lags = np.arange(-60, 61)
        Y = np.fft.fft(y)
        R = np.real(np.fft.ifft(np.abs(Y) ** 2))
        Rl = R[lags % N] / R[0]
        pa = self.plot("acf")
        pa.line("R", lags * 1e3 / self.fs, Rl, color=NAVY, width=2.0, name="filtered")
        pa.scatter("Rw", [0], [1], color=GRAY, size=9, name="white: a single spike")
        pa.hline("z", 0, color=GRAY, style="-", width=0.6)
        ktc = np.sqrt(nz.K_B * p.temp / (p.cap * 1e-12)) * 1e6
        self.readout(bn=bn, ratio=bn / p.fc, pw=np.var(y) / bn, ktc=ktc)

    def story(self, p):
        r = self.r
        s = (f"<p>White noise has the same power in every hertz, so it wanders with no memory (grey). "
             f"Through the filter it slows down (blue), its PSD takes the shape |H(f)|² (top right), "
             f"and its autocorrelation widens (bottom right): narrower spectrum, longer memory.</p>"
             f"<p>The total power is N₀ times the <b>noise-equivalent bandwidth</b> B_N, the width of "
             f"the rectangle with the same area as |H(f)|² (bottom left): here "
             f"{v(r.get('bn', 0), '.1f', 'Hz')}, {v(r.get('ratio', 0), '.3f')}× the 3 dB bandwidth. ")
        if p.filt.startswith("RC"):
            s += "One pole lets in π/2 times more noise than its 3 dB width suggests.</p>"
        elif p.filt == "Butterworth":
            s += (f"For an n-pole Butterworth the ratio is (π/2n)/sin(π/2n): steeper skirts, less "
                  f"excess noise.</p>")
        else:
            s += "</p>"
        s += (f"<p>The same maths gives the <b>kT/C</b> noise on a sampling capacitor: the RC's B_N is "
              f"1/(4RC), so the noise power kT·4R·B_N = kT/C, whatever R is. "
              f"{v(p.cap, '.3g', 'pF')} at {v(p.temp, '.0f', 'K')} gives "
              f"{v(r.get('ktc', 0), '.1f', 'µV')} rms, which is why high-resolution ADCs use big "
              f"sampling capacitors.</p>")
        return "<h3>Noise takes the filter's shape</h3>" + s + keybox(
            "Output PSD = |H(f)|²·N₀/2; output power = N₀·B_N, and B_N ≥ f_3dB.")


# =============================================================================== 5. bandpass noise
class BandpassNoise(Experiment):
    title = "Bandpass noise: two baseband halves"
    blurb = "Narrowband noise is a carrier with a wandering amplitude and phase."
    book = "sec:ch03:bandpass"
    controls = [
        Slider("fc", "Band centre", 5, 60, 20, step=0.5, unit="Hz"),
        Slider("B", "Bandwidth", 1, 20, 4, step=0.5, unit="Hz"),
        Slider("lo", "LO offset", -6, 6, 0, step=0.1, unit="Hz",
               help="Mix with a carrier this far from the band centre"),
    ]
    plots = [
        Plot("time", "Bandpass noise: a carrier with a wandering envelope", x="time (s)",
             y="amplitude", xlim=(0, 4), ylim=(-4.8, 6.0), legend="tl", legend_cols=3),
        SpectrumPlot("psd", "Power spectral densities", x="frequency (Hz)", y="PSD (dB/Hz)",
                     xlim=(-70, 70), ylim=(-40, 5), legend="tr"),
        ConstellationPlot("iq", "I/Q scatter of the complex envelope", lim=4.2),
    ]
    layout = [["time", "time"], ["psd", "iq"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("pn", "Power of n(t)", "", ".3f"),
        Readout("pi", "Power of n_I", "", ".3f"),
        Readout("pq", "Power of n_Q", "", ".3f"),
        Readout("tc", "Envelope correlation time", "s", ".2f",
                help="Lag at which the envelope's autocorrelation falls to half"),
    ]
    challenges = [
        Challenge("Slow the envelope down: make its correlation time longer than 0.25 s.",
                  lambda s: s.r.tc > 0.25, hint="Memory ~ 1/bandwidth."),
        Challenge("Mistune the down-converter by half the bandwidth or more. I and Q keep equal "
                  "power, but look at the complex spectrum.",
                  lambda s: abs(s.p.lo) >= s.p.B / 2 and abs(s.r.pi / s.r.pq - 1) < 0.1),
        Challenge("Find the bandwidth that gives an envelope correlation time of 0.20 s (±0.01 s).",
                  lambda s: abs(s.r.tc - 0.20) <= 0.01,
                  hint="The correlation time scales as 1/B."),
    ]
    fs, N = 200.0, 1 << 15

    @staticmethod
    @lru_cache(maxsize=1)
    def white():
        return np.random.default_rng(7).standard_normal(1 << 15)

    @lru_cache(maxsize=64)
    def bandpass(self, fc, B):
        w = self.white()
        f = np.fft.fftfreq(self.N, 1 / self.fs)
        H = ((np.abs(f) >= fc - B / 2) & (np.abs(f) <= fc + B / 2)).astype(float)
        H = np.convolve(H, np.hanning(9) / np.hanning(9).sum(), mode="same")   # soft edges
        n = np.real(np.fft.ifft(np.fft.fft(w) * H))
        return n / np.std(n)

    def update(self, p):
        n = self.bandpass(round(p.fc, 2), round(p.B, 2))
        t = np.arange(self.N) / self.fs
        an = sps.hilbert(n)                                 # analytic signal: positive half only
        env = an * np.exp(-2j * np.pi * (p.fc + p.lo) * t)  # complex envelope (I + jQ)
        a0 = 4000
        k = slice(a0, a0 + int(4.05 * self.fs))
        tt = t[k] - t[a0]
        pt = self.plot("time")
        pt.line("n", tt, n[k], color=GRAY, width=0.8, name="n(t)")
        pt.line("e1", tt, np.abs(env[k]), color=RED, width=2.0, name="envelope |ñ(t)|")
        pt.line("e2", tt, -np.abs(env[k]), color=RED, width=2.0)
        pt.line("ni", tt, env[k].real, color=NAVY, width=1.2, style="--", name="n_I(t)")
        f1, P1 = sps.welch(n, self.fs, nperseg=2048, return_onesided=False)
        f2, P2 = sps.welch(env, self.fs, nperseg=2048, return_onesided=False)
        f3, P3 = sps.welch(env.real, self.fs, nperseg=2048, return_onesided=False)
        o = np.argsort(f1)
        ps = self.plot("psd")
        ps.line("n", f1[o], db(P1[o]), color=GRAY, width=1.6, name="passband n(t)")
        ps.line("c", f2[o], db(P2[o]), color=RED, width=1.4, style="--", name="complex ñ")
        ps.line("i", f3[o], db(P3[o]), color=NAVY, width=2.0, name="n_I alone")
        ps.set_ylim(-40, max(5, float(db(P2.max())) + 4))
        pq = self.plot("iq")
        pq.points("z", env[::12], color=NAVY, size=3, alpha=0.4)
        ea = np.abs(env) - np.mean(np.abs(env))
        R = np.real(np.fft.ifft(np.abs(np.fft.fft(ea, 2 * self.N)) ** 2))[:4000]
        R = R / R[0]
        half = np.flatnonzero(R < 0.5)
        tc = half[0] / self.fs if len(half) else 4000 / self.fs
        self.readout(pn=np.var(n), pi=np.var(env.real), pq=np.var(env.imag), tc=tc)

    def story(self, p):
        r = self.r
        s = (f"<p>Noise filtered to a {v(p.B, '.1f', 'Hz')} band around {v(p.fc, '.1f', 'Hz')} looks "
             f"like a carrier whose amplitude (red) and phase wander about once every 1/B seconds. "
             f"Write it as n(t) = n_I(t)·cos 2πf_c t − n_Q(t)·sin 2πf_c t.</p>"
             f"<p>The bookkeeping that confuses everyone once: n_I and n_Q each carry the <b>same</b> "
             f"power as n(t) ({v(r.get('pi', 0), '.2f')}, {v(r.get('pq', 0), '.2f')} vs "
             f"{v(r.get('pn', 0), '.2f')}), squeezed into half the width with twice the PSD (blue "
             f"curve, 3 dB above the grey). The complex envelope carries twice the power. The I/Q "
             f"cloud is round: I and Q are independent.</p>")
        if abs(p.lo) >= p.B / 2:
            s += ("<p>With the LO off-centre the complex spectrum is lopsided (red dashed), yet I and "
                  "Q still have equal power. A complex envelope need not be symmetric; I alone, a "
                  "real signal, always is.</p>")
        return "<h3>Two independent halves</h3>" + s + keybox(
            "Bandpass Gaussian noise = two independent baseband Gaussian processes on cos and sin. "
            "Every complex-baseband simulation in this book relies on it.")


# =============================================================================== 6. Friis
BASE_STAGES = [("switch", -0.5, 0.5), ("SAW filter", -1.5, 1.5), ("LNA", 18.0, 1.0),
               ("mixer", 8.0, 10.0), ("baseband", 30.0, 12.0), ("ADC", 0.0, 27.0)]


class FriisCascade(Experiment):
    title = "Friis: who adds the noise?"
    blurb = "Rearrange a receiver line-up and watch the first stages dominate."
    book = "sec:ch03:nf"
    controls = [
        Heading("Line-up"),
        IntSlider("pos", "LNA position", 1, 6, 3,
                  help="1 = right at the antenna. The other stages keep their order"),
        Slider("feed", "Feed-line loss", 0.0, 4.0, 2.0, step=0.1, unit="dB"),
        Heading("LNA"),
        Slider("lnanf", "LNA noise figure", 0.3, 4.0, 1.0, step=0.05, unit="dB"),
        Slider("lnag", "LNA gain", 6, 30, 18, step=0.5, unit="dB"),
        Heading("Antenna"),
        Slider("tant", "Antenna temperature", 10, 300, 290, step=5, unit="K",
               help="A dish pointed at the cold sky can be 20 K; a terrestrial antenna sees ~290 K"),
    ]
    plots = [
        BarPlot("contrib", "Noise each stage adds (referred to the antenna)", y="added noise temperature (K)"),
        Plot("cum", "Cumulative noise figure along the chain", x="", y="noise figure (dB)",
             legend=None),
        Plot("cold", "SNR penalty vs loss ahead of the LNA", x="loss ahead of the LNA (dB)",
             y="SNR penalty (dB)", xlim=(0, 6), ylim=(0, 12), legend="tl"),
    ]
    layout = [["contrib", "contrib"], ["cum", "cold"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("nf", "Cascade noise figure", "dB", ".2f", good=lambda x: x < 2),
        Readout("te", "Receiver noise temperature", "K", ".0f"),
        Readout("tsys", "System temperature", "K", ".0f"),
        Readout("floor", "Noise floor in 1 MHz", "dBm", ".1f"),
    ]
    challenges = [
        Challenge("Chapter 3's handset: remove the feed line and confirm the 3.58 dB cascade noise figure.",
                  lambda s: s.p.feed == 0 and s.p.pos == 3 and abs(s.p.lnanf - 1) < 0.01
                  and abs(s.p.lnag - 18) < 0.01 and abs(s.r.nf - 3.58) < 0.01),
        Challenge("Keep the cascade below 2 dB with 2 dB of feed line, without an LNA better than 1 dB.",
                  lambda s: s.p.feed >= 2 and s.p.lnanf >= 1 and s.r.nf < 2,
                  hint="A mast-head amplifier, or a satellite dish's LNB, sits right at the antenna."),
        Challenge("Satellite ground station: with a 20 K sky, get the system temperature below 80 K "
                  "with 1 dB or more of feed loss.",
                  lambda s: s.p.tant <= 20 and s.p.feed >= 1 and s.r.tsys < 80),
        Challenge("With the LNA first, find the lowest LNA gain that keeps the cascade within 0.5 dB "
                  "of the LNA's own noise figure (no feed line).",
                  lambda s: s.p.pos == 1 and s.p.feed == 0 and s.r.nf - s.p.lnanf <= 0.5
                  and s.exp.nf_for(s.p, s.p.lnag - 0.5) - s.p.lnanf > 0.5),
    ]

    def stages(self, p, lnag=None):
        st_ = [list(x) for x in BASE_STAGES]
        lna = [s for s in st_ if s[0] == "LNA"][0]
        lna[1], lna[2] = (p.lnag if lnag is None else lnag), p.lnanf
        others = [s for s in st_ if s[0] != "LNA"]
        if p.feed > 0:                                      # the feed line comes first, so an LNA at
            others = [["feed line", -p.feed, p.feed]] + others    # position 1 is a mast-head amplifier
        chain = others[:p.pos - 1] + [lna] + others[p.pos - 1:]
        return [tuple(s) for s in chain]

    def nf_for(self, p, lnag):
        return nz.friis(self.stages(p, lnag))["nf"]

    def update(self, p):
        stg = self.stages(p)
        fr = nz.friis(stg)
        names = [s[0] for s in stg]
        pb = self.plot("contrib")
        x = np.arange(len(stg))
        cols = [RED if n == "LNA" else ORANGE if n == "feed line" else NAVY for n in names]
        te_add = fr["te_add"]
        pb.bars("b", x, te_add, width=0.62, colors=cols)
        for i, tv in enumerate(te_add):
            pb.text(f"t{i}", i, tv, f"{tv:.0f} K" if tv >= 1 else f"{tv:.2f} K", anchor=(0.5, 1.05),
                    bold=True, size=9)
        pb.set_xticks([(i, n) for i, n in enumerate(names)])
        pb.set_xlim(-0.6, len(stg) - 0.4)
        pb.set_ylim(0, max(50, float(te_add.max()) * 1.25))
        pc = self.plot("cum")
        pc.line("c", np.arange(len(stg) + 1) - 0.5, fr["cum_nf"], color=NAVY, width=2.4, step=True,
                fill=0, fill_alpha=0.12)
        pc.set_xticks([(i, n) for i, n in enumerate(names)])
        pc.set_xlim(-0.6, len(stg) - 0.4)
        pc.set_ylim(0, max(4, float(fr["nf"]) * 1.25))
        pc.hline("fin", fr["nf"], color=RED, style="--", label=f"{fr['nf']:.2f} dB", label_pos=0.02)
        # SNR penalty of loss ahead of the LNA: cold sky vs warm antenna
        L = np.linspace(0, 6, 200)
        t_lna = nz.T0 * (10 ** (p.lnanf / 10) - 1)
        pk = self.plot("cold")
        for ta, col, sty, nm in [(p.tant, NAVY, "-", f"your antenna ({p.tant:.0f} K)"),
                                 (290, GRAY, "--", "warm antenna (290 K)")]:
            pen = 10 * np.log10(nz.lossy_line_tsys(L, ta, t_lna) / (ta + t_lna)) + L
            pk.line("p" + nm[:4], L, pen, color=col, width=2.2, style=sty, name=nm)
        ahead = sum(-s[1] for s in stg[:names.index("LNA")] if s[1] < 0)
        pen_now = 10 * np.log10(nz.lossy_line_tsys(ahead, p.tant, t_lna) / (p.tant + t_lna)) + ahead
        pk.scatter("now", [ahead], [pen_now], color=RED, size=14, symbol="d", name="your line-up")
        tsys = p.tant + fr["te"]
        self.readout(nf=fr["nf"], te=fr["te"], tsys=tsys,
                     floor=10 * np.log10(nz.K_B * tsys * 1e6 * 1e3))

    def story(self, p):
        r = self.r
        s = (f"<p>Each stage's noise is divided by all the gain in front of it: the <b>Friis "
             f"formula</b>, F = F₁ + (F₂ − 1)/G₁ + (F₃ − 1)/(G₁G₂) + … The bars show each stage's "
             f"share in kelvin at the antenna. Cascade: {v(r.get('nf', 0), '.2f', 'dB')}.</p>")
        if p.pos > 1 or p.feed > 0:
            s += ("<p>Everything ahead of the LNA counts in full: a passive loss L at room "
                  "temperature has a noise figure of exactly L dB. Move the LNA to position 1 and "
                  "the stages behind it almost vanish from the budget.</p>")
        if p.tant < 100:
            s += (f"<p>With a cold {v(p.tant, '.0f', 'K')} antenna, loss ahead of the LNA hurts twice: "
                  f"it attenuates the signal and it radiates 290 K of noise of its own into a "
                  f"receiver that was expecting the cold sky (bottom right). This is why satellite "
                  f"dishes put the LNB at the focus.</p>")
        s += (f"<p>System temperature {v(r.get('tsys', 0), '.0f', 'K')}: a noise floor of "
              f"{v(r.get('floor', 0), '.1f', 'dBm')} in every megahertz.</p>")
        return "<h3>First stages first</h3>" + s + keybox(
            "Put gain with low noise as early as possible. Loss before the LNA adds its full value "
            "to the noise figure.")


# =============================================================================== 7. Y-factor
class YFactor(Experiment):
    title = "The Y-factor measurement"
    blurb = "Measure a noise figure with a noise source switched on and off, and see its uncertainty."
    book = "sec:ch03:yfactor"
    controls = [
        Heading("Noise source"),
        Choice("enr", "Excess noise ratio (ENR)", ["5 dB", "15 dB"], "15 dB"),
        Heading("Device under test"),
        Slider("dutnf", "True noise figure", 0.3, 10, 1.5, step=0.05, unit="dB"),
        Slider("dutg", "Gain", 0, 40, 20, step=0.5, unit="dB"),
        Heading("Analyser"),
        Slider("ananf", "Analyser noise figure", 3, 25, 8, step=0.5, unit="dB"),
        Choice("navg", "Samples per power reading", ["10³", "10⁴", "10⁵", "10⁶"], "10⁴", style="menu"),
        Toggle("corr", "Subtract the analyser's noise", True),
        Button("again", "Measure again"),
    ]
    plots = [
        Plot("hist", "400 simulated measurements", x="measured DUT noise figure (dB)", y="count",
             ylim=(0, 80), legend="tr"),
        Plot("line", "Output noise vs source temperature", x="source temperature (K)",
             y="output noise / (kGB) (K)", legend="tl"),
        Plot("sens", "NF error from a 0.05 dB error in Y", x="DUT noise figure (dB)",
             y="NF error (dB)", xlim=(0, 14), ylim=(0, 0.6), legend="tl"),
    ]
    layout = [["hist", "hist"], ["line", "sens"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("Y", "Measured Y", "dB", ".2f"),
        Readout("mean", "Mean measured NF", "dB", ".2f"),
        Readout("std", "Spread (std dev)", "dB", ".3f", good=lambda x: x < 0.05),
        Readout("bias", "Bias", "dB", ".2f", good=lambda x: abs(x) < 0.05),
    ]
    challenges = [
        Challenge("Measure the 1.5 dB LNA to ±0.02 dB (standard deviation) with the fewest samples.",
                  lambda s: abs(s.p.dutnf - 1.5) < 0.01 and s.r.std < 0.02 and s.p.navg == "10⁵"),
        Challenge("Forget the second-stage correction on a 10 dB-gain DUT: a bias of more than 1 dB.",
                  lambda s: not s.p.corr and s.p.dutg <= 10 and abs(s.r.bias) > 1),
        Challenge("A noisy DUT (NF ≥ 8 dB) with the 5 dB source: Y is so close to 1 that the spread "
                  "is more than twice the 15 dB source's.",
                  lambda s: s.p.enr == "5 dB" and s.p.dutnf >= 8 and s.r.std > 2 * s.exp.std_other),
    ]
    trials = 400

    def setup(self):
        self.run = 0

    def on_again(self, p):
        self.run += 1

    def measure(self, p, enr_db, rng):
        n = {"10³": 1e3, "10⁴": 1e4, "10⁵": 1e5, "10⁶": 1e6}[p.navg]
        T0 = nz.T0
        te_dut = T0 * (10 ** (p.dutnf / 10) - 1)
        te_ana = T0 * (10 ** (p.ananf / 10) - 1)
        G = 10 ** (p.dutg / 10)
        th = T0 * (10 ** (enr_db / 10) + 1)
        out = lambda ts: G * (ts + te_dut) + te_ana
        # a power estimate from n samples of complex Gaussian noise: Gamma(n, 1/n) x true power
        Y = out(th) * rng.gamma(n, 1 / n, self.trials) / (out(T0) * rng.gamma(n, 1 / n, self.trials))
        F = nz.yfactor_f(Y, enr_db, p.ananf if p.corr else None, p.dutg if p.corr else None)
        return Y, 10 * np.log10(np.maximum(F, 1e-3))

    def update(self, p):
        enr = float(p.enr.split()[0])
        rng = np.random.default_rng(100 + self.run)
        Y, nf = self.measure(p, enr, rng)
        other = 15.0 if enr == 5 else 5.0
        self.std_other = float(np.std(self.measure(p, other, np.random.default_rng(200 + self.run))[1]))
        ph = self.plot("hist")
        lo, hi = p.dutnf - 1.5, p.dutnf + 1.5
        if not p.corr:
            hi = max(hi, float(np.percentile(nf, 99.5)) + 0.2)
        h, e = np.histogram(nf, 60, (lo, hi))
        ph.line("h", e, h, color=NAVY, width=1.0, step=True, fill=0, fill_alpha=0.35, name="measurements")
        ph.vline("true", p.dutnf, color=GREEN, style="-", width=2.0, label="true NF", label_pos=0.93)
        ph.vline("mean", float(np.mean(nf)), color=RED, style="--", label="mean measured", label_pos=0.8)
        ph.set_xlim(lo, hi)
        ph.set_ylim(0, max(30, float(h.max()) * 1.2))
        # the Y-factor straight line
        te = nz.T0 * (10 ** (p.dutnf / 10) - 1)
        ts = np.linspace(-te - 50, nz.T0 * (10 ** (enr / 10) + 1) * 1.1, 50)
        pl = self.plot("line")
        pl.line("l", ts, ts + te, color=NAVY, width=2.0, name="T_s + T_e")
        th = nz.T0 * (10 ** (enr / 10) + 1)
        pl.scatter("pts", [nz.T0, th], [nz.T0 + te, th + te], color=RED, size=11,
                   name="source off (290 K) and on")
        pl.scatter("ic", [-te], [0], color=GREEN, size=11, symbol="s", name=f"intercept −T_e = −{te:.0f} K")
        pl.hline("z", 0, color=GRAY, style="-", width=0.6)
        pl.vline("z2", 0, color=GRAY, style="-", width=0.6)
        # sensitivity
        nfg = np.linspace(0.1, 14, 300)
        F = 10 ** (nfg / 10)
        pk = self.plot("sens")
        for e_db, col in [(5.0, GREEN), (15.0, NAVY)]:
            e_ = 10 ** (e_db / 10)
            Yt = e_ / F + 1
            Fp = e_ / (Yt * 10 ** (0.005) - 1)
            pk.line(f"s{e_db}", nfg, np.abs(10 * np.log10(Fp / F)), color=col, width=2.0,
                    name=f"ENR {e_db:.0f} dB")
        pk.vline("now", p.dutnf, color=RED, style="--", label="your DUT", label_pos=0.08)
        self.readout(Y=10 * np.log10(np.median(Y)), mean=float(np.mean(nf)), std=float(np.std(nf)),
                     bias=float(np.mean(nf) - p.dutnf))

    def story(self, p):
        r = self.r
        s = ("<p>A noise source has two temperatures: 290 K when off, much hotter when on (its "
             "<b>ENR</b>). The output noise power is a straight line in the source temperature "
             "(bottom left) whose intercept is −T_e. Two powers fix the line: their ratio Y gives "
             "F = ENR/(Y − 1).</p>")
        s += (f"<p>But a power meter averages only a finite number of noise samples, so each reading "
              f"wobbles and so does the answer: 400 repeats spread by {v(r.get('std', 0), '.3f', 'dB')}. "
              f"Ten times more samples, √10 less spread.</p>")
        if not p.corr:
            s += (f"<p>{bad('No second-stage correction')}: the analyser's own noise is blamed on the "
                  f"DUT, a bias of {v(r.get('bias', 0), '.2f', 'dB')}. With "
                  f"{v(p.dutg, '.0f', 'dB')} of DUT gain the analyser is "
                  + ("well hidden." if p.dutg > 25 else "far from hidden.") + "</p>")
        s += ("<p>Chapter 3's example: ENR 15 dB, a 20 dB LNA and an 8 dB analyser give Y = 13.5 dB, "
              "1.70 dB for the system and 1.54 dB for the LNA. The right plot is the warning: a "
              "low-ENR source suits low-NF devices.</p>")
        return "<h3>Hot, cold, and a straight line</h3>" + s + keybox(
            "F = ENR/(Y − 1), minus (F₂ − 1)/G₁ for the analyser. Average long, and match the ENR to the DUT.")


# =============================================================================== 8. ROC
DETS = ["Coherent (known phase)", "Envelope (unknown phase)", "Energy (nothing known)"]
D_COL = {DETS[0]: NAVY, DETS[1]: RED, DETS[2]: GREEN}


class ROCDetector(Experiment):
    title = "Detection: thresholds and ROC curves"
    blurb = "Set the threshold, trade detections against false alarms, and meet a radar spec."
    book = "sec:ch03:detection"
    controls = [
        Choice("det", "Detector", DETS, style="menu"),
        Slider("snr", "Signal energy E/N₀", 0, 22, 10, step=0.1, unit="dB"),
        LogSlider("pfa", "False-alarm rate P_FA", 1e-8, 0.3, 1e-2,
                  help="Neyman–Pearson: fix P_FA, then the threshold follows"),
        IntSlider("N", "Energy-detector samples", 1, 64, 16,
                  enabled_if=lambda p: p.det.startswith("Energy")),
        Heading("Base rate"),
        LogSlider("prior", "Prior P(signal)", 1e-5, 0.5, 1e-2),
    ]
    plots = [
        Plot("pdf", "Test statistic: noise only (H₀) and signal + noise (H₁)", x="test statistic",
             y="density", legend="tr"),
        Plot("roc", "ROC: detection vs false alarm", x="false-alarm probability P_FA",
             y="detection probability P_D", logx=True, xlim=(1e-8, 1), ylim=(0, 1.03), legend="tl"),
        Plot("pd", "P_D vs E/N₀ at this P_FA", x="E/N₀ (dB)", y="P_D", xlim=(0, 22),
             ylim=(0, 1.03), legend="tl"),
    ]
    layout = [["pdf", "pdf"], ["roc", "pd"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("pd", "Detection probability", "", ".4f", good=lambda x: x >= 0.9),
        Readout("need", "E/N₀ for P_D = 0.9", "dB", ".1f"),
        Readout("ppv", "Alarms that are real", "%", ".1f", good=lambda x: x >= 50),
        Readout("mc", "Monte Carlo P_D", "", ".3f"),
    ]
    challenges = [
        Challenge("Radar spec: P_FA ≤ 10⁻⁶ and P_D ≥ 0.9 with a coherent detector, at the lowest "
                  "E/N₀ that does it (within 0.3 dB).",
                  lambda s: s.p.det.startswith("Coh") and s.p.pfa <= 1.01e-6 and s.r.pd >= 0.9
                  and s.p.snr <= s.r.need + 0.3),
        Challenge("Meet the same spec (P_FA ≤ 10⁻⁶, P_D ≥ 0.9) with an energy detector over 64 samples.",
                  lambda s: s.p.det.startswith("Energy") and s.p.N == 64 and s.p.pfa <= 1.01e-6
                  and s.r.pd >= 0.9,
                  hint="Knowing nothing about the signal costs several dB."),
        Challenge("Base rates: with a prior of 10⁻⁴, make at least half the alarms real while keeping P_D ≥ 0.9.",
                  lambda s: abs(np.log10(s.p.prior) + 4) < 0.05 and s.r.ppv >= 50 and s.r.pd >= 0.9),
    ]

    def theory(self, det, pfa, snr, N):
        if det.startswith("Coh"):
            return nz.pd_coherent(pfa, snr)
        if det.startswith("Env"):
            return nz.pd_envelope(pfa, snr)
        return nz.pd_energy(pfa, snr, N)

    def update(self, p):
        E = 10 ** (p.snr / 10)
        N = p.N
        rng = np.random.default_rng(6)
        M = 20000
        if p.det.startswith("Coh"):
            h0 = rng.standard_normal(M)
            h1 = np.sqrt(2 * E) + rng.standard_normal(M)
            thr = nz.thresholds(p.pfa)["coherent"]
            y = np.linspace(-5, max(8, np.sqrt(2 * E) + 5), 600)
            f0, f1 = stats.norm.pdf(y), stats.norm.pdf(y, np.sqrt(2 * E))
        else:
            dof = 2 if p.det.startswith("Env") else 2 * N
            nn = 1 if p.det.startswith("Env") else N
            w0 = rng.standard_normal((M, 2 * nn))
            ph = rng.uniform(0, 2 * np.pi, (M, nn))
            amp = np.sqrt(2 * E / nn)
            w1 = rng.standard_normal((M, 2 * nn))
            w1[:, :nn] += amp * np.cos(ph)
            w1[:, nn:] += amp * np.sin(ph)
            h0, h1 = (w0 ** 2).sum(1), (w1 ** 2).sum(1)
            thr = nz.thresholds(p.pfa, N)["envelope" if dof == 2 else "energy"]
            y = np.linspace(0, max(stats.chi2.isf(1e-6, dof), 2 * E + dof + 8 * np.sqrt(4 * E + dof)), 600)
            f0, f1 = stats.chi2.pdf(y, dof), stats.ncx2.pdf(y, dof, 2 * E)
        pd = float(self.theory(p.det, p.pfa, E, N))
        pp = self.plot("pdf")
        pp.line("f0", y, f0, color=NAVY, width=2.2, name="H₀: noise only")
        pp.line("f1", y, f1, color=ORANGE, width=2.2, name="H₁: signal + noise")
        k = y >= thr
        if k.sum() > 1:
            pp.line("a1", y[k], f1[k], color=ORANGE, width=0.5, fill=0, fill_alpha=0.25)
            pp.line("a0", y[k], f0[k], color=NAVY, width=0.5, fill=0, fill_alpha=0.55)
        pp.vline("thr", thr, color=RED, style="--", width=2.0, label="threshold", label_pos=0.9)
        pp.set_xlim(float(y[0]), float(y[-1]))
        pp.set_ylim(0, float(max(f0.max(), f1.max())) * 1.25)
        pf = np.logspace(-8, 0, 300)
        pr = self.plot("roc")
        for d in DETS:
            pr.line("r" + d[:3], pf, self.theory(d, pf, E, N), color=D_COL[d],
                    width=2.8 if d == p.det else 1.2, alpha=1 if d == p.det else 0.6, name=d.split()[0])
        q = np.logspace(-4, -0.05, 12)
        thq = np.quantile(h0, 1 - q)
        mc = np.array([np.mean(h1 > t_) for t_ in thq])
        pr.scatter("mc", np.mean(h0[None, :] > thq[:, None], axis=1), mc, color=PURPLE, size=7,
                   outline=PURPLE, name="Monte Carlo")
        pr.scatter("now", [p.pfa], [pd], color=RED, size=15, symbol="d", name="operating point")
        sd = np.linspace(0, 22, 221)
        pq = self.plot("pd")
        need = None
        for d in DETS:
            c = self.theory(d, p.pfa, 10 ** (sd / 10), N)
            pq.line("d" + d[:3], sd, c, color=D_COL[d], width=2.8 if d == p.det else 1.2,
                    alpha=1 if d == p.det else 0.6, name=d.split()[0])
            if d == p.det:
                i = np.flatnonzero(c >= 0.9)
                need = float(sd[i[0]]) if len(i) else None
        pq.hline("p9", 0.9, color=GRAY, style=":", label="P_D = 0.9", label_pos=0.75)
        pq.scatter("now", [p.snr], [pd], color=RED, size=14, symbol="d")
        ppv = 100 * pd * p.prior / (pd * p.prior + p.pfa * (1 - p.prior))
        mc_now = float(np.mean(h1 > np.quantile(h0, 1 - p.pfa))) if p.pfa >= 1e-3 else None
        self.readout(pd=pd, need=need if need is not None else "> 22", ppv=ppv,
                     mc=mc_now if mc_now is not None else "—")

    def story(self, p):
        r = self.r
        s = (f"<p>The receiver computes one number and compares it with a threshold. Noise alone "
             f"(blue) occasionally crosses it: a <b>false alarm</b>, rate {v(st.sci(p.pfa))}. A signal "
             f"(orange) crosses it with probability P_D = {v(r.get('pd', 0), '.3f')}. Moving the "
             f"threshold trades one for the other; the <b>ROC curve</b> is every possible trade. "
             f"Neyman–Pearson: fix the false-alarm rate you can live with, then maximise P_D.</p>")
        s += ("<p>Knowing more about the signal buys decibels: the envelope detector (unknown phase) "
              "pays about half a decibel, the energy detector (unknown everything) several.</p>")
        ppv = r.get("ppv", 0)
        s += (f"<p>And the base rate: if a signal is present only {v(st.sci(p.prior))} of the time, "
              f"just {v(ppv, '.1f', '%')} of alarms are real. "
              + (bad("Most alarms are false.") if ppv < 50 else good("Most alarms are real.")) +
              " Chapter 3: P_D = 0.99, P_FA = 10⁻³, prior 10⁻⁴: right only 9 % of the time.</p>")
        return "<h3>Signal or no signal?</h3>" + s + keybox(
            "Detection = a threshold on a statistic. P_FA sets the threshold; E/N₀ and what you know "
            "about the signal set P_D.")


# =============================================================================== 9. cyclostationary
@lru_cache(maxsize=12)
def cyclo_signal(kind, sps_, k0, nblk, seed):
    nfft = 256
    rng = np.random.default_rng(seed)
    L = nfft * nblk
    nsym = L // sps_ + 20
    if kind == "BPSK":
        sym = rng.choice([-1.0, 1.0], nsym) + 0j
    elif kind == "QPSK":
        sym = (rng.choice([-1.0, 1.0], nsym) + 1j * rng.choice([-1.0, 1.0], nsym)) / np.sqrt(2)
    else:
        sym = np.zeros(nsym, complex)
    up = np.zeros(nsym * sps_, complex)
    up[::sps_] = sym
    x = sps.fftconvolve(up, cl.rrc_taps(0.35, sps_, 8))[:L]
    if kind != "Noise only":
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    x = x * np.exp(2j * np.pi * k0 * np.arange(L) / nfft)
    w = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    return x, w


class Cyclostationary(Experiment):
    title = "Finding a signal below the noise"
    blurb = "Invisible in the spectrum, obvious in the spectral correlation: cyclostationarity."
    book = "sec:ch03:cyclo"
    heavy = True
    controls = [
        Choice("kind", "Hidden signal", ["BPSK", "QPSK", "Noise only"]),
        Slider("snr", "SNR", -25, 5, -10, step=0.5, unit="dB"),
        IntSlider("sps", "Samples per symbol", 4, 16, 8, help="Sets the symbol rate R_s = f_s / sps"),
        IntSlider("k0", "Carrier offset", 0, 40, 10, unit="bins of fs/256"),
        Choice("nblk", "Observation length", ["100 blocks", "300 blocks", "1000 blocks", "2000 blocks"],
               "300 blocks", style="menu", help="Blocks of 256 samples averaged by the estimator"),
        Button("again", "New noise"),
    ]
    plots = [
        SpectrumPlot("psd", "Power spectrum: can you see the signal?", x="frequency (f / fs)",
                     y="PSD (dB re. noise)", xlim=(-0.5, 0.5), ylim=(-1, 4), legend="tr"),
        Plot("nc", "Spectral coherence vs cycle frequency (non-conjugate)",
             x="cycle frequency α (f / fs)", y="max coherence", xlim=(-0.5, 0.5), ylim=(0, 0.7),
             legend="tr"),
        Plot("cj", "Conjugate spectral coherence", x="cycle frequency α (f / fs)",
             y="max coherence", xlim=(-0.5, 0.5), ylim=(0, 1.0), legend="tr"),
    ]
    layout = [["psd", "psd"], ["nc", "cj"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("bump", "PSD rise above the noise", "dB", ".2f"),
        Readout("feat", "Feature at α = R_s", "× estimator floor", ".1f", good=lambda x: x > 1.8),
        Readout("conj", "Feature at α = 2f₀", "× estimator floor", ".1f"),
        Readout("verdict", "Detector says", "", None),
    ]
    challenges = [
        Challenge("Detect BPSK at an SNR of −15 dB or lower (R_s feature above 1.8× the floor).",
                  lambda s: s.p.kind == "BPSK" and s.p.snr <= -15 and s.r.feat > 1.8,
                  hint="A longer observation lowers the estimator's floor."),
        Challenge("Energy detection is hopeless here: a PSD rise below 1 dB while the cyclic feature "
                  "is detected.",
                  lambda s: s.p.kind != "Noise only" and s.r.bump < 1 and s.r.feat > 1.8),
        Challenge("QPSK hides one feature: keep the R_s line but make the conjugate 2f₀ line vanish "
                  "(below 1.5×).",
                  lambda s: s.p.kind == "QPSK" and s.r.feat > 1.8 and s.r.conj < 1.5),
    ]

    def setup(self):
        self.seed = 11

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def ratio(alpha, prof, target):
        """Feature height at cycle frequency `target` over the median of the profile elsewhere."""
        ok = np.isfinite(prof)
        d = np.abs(((alpha - target + 0.5) % 1) - 0.5)
        near = d <= 1.5 / 256
        if not np.any(near & ok):
            return 0.0
        floor = np.median(prof[ok & (d > 4 / 256)])
        return float(np.nanmax(prof[near & ok]) / max(floor, 1e-9))

    def update(self, p):
        nblk = int(p.nblk.split()[0])
        if self.quick:
            nblk = min(nblk, 300)
        x, w = cyclo_signal(p.kind, p.sps, p.k0, nblk, self.seed)
        y = x * 10 ** (p.snr / 20) + w
        alpha, nc, cj, P = sp.spectral_coherence(y, 256)
        _, nc0, cj0, Pn = sp.spectral_coherence(w, 256)
        fr = np.fft.fftshift(np.fft.fftfreq(256))
        ref = np.median(Pn)
        pp = self.plot("psd")
        pp.line("n", fr, db(np.fft.fftshift(Pn) / ref), color=ORANGE, width=1.2, name="noise only")
        pp.line("s", fr, db(np.fft.fftshift(P) / ref), color=NAVY, width=2.0, name="what you receive")
        Rs = 1 / p.sps
        pn = self.plot("nc")
        pn.line("n0", alpha, nc0, color=ORANGE, width=1.0, name="noise only")
        pn.line("s", alpha, nc, color=NAVY, width=1.8, name="what you receive")
        pn.vline("r1", Rs, color=RED, style=":", label="α = R_s", label_pos=0.06)
        pn.vline("r2", -Rs, color=RED, style=":")
        pc = self.plot("cj")
        f0 = p.k0 / 256
        pc.line("n0", alpha, cj0, color=ORANGE, width=1.0, name="noise only")
        pc.line("s", alpha, cj, color=NAVY, width=1.8, name="what you receive")
        a2 = ((2 * f0 + 0.5) % 1) - 0.5
        pc.vline("f0", a2, color=RED, style=":", label="α = 2f₀", label_pos=0.06)
        feat = max(self.ratio(alpha, nc, Rs), self.ratio(alpha, nc, -Rs))
        conj = self.ratio(alpha, cj, a2)
        inband = np.abs(((fr - f0 + 0.5) % 1) - 0.5) < 0.5 / p.sps
        bump = float(db(np.mean(np.fft.fftshift(P)[inband]) / np.mean(np.fft.fftshift(Pn)[inband])))
        self.readout(bump=bump, feat=feat, conj=conj,
                     verdict="signal present" if feat > 1.8 else "only noise")

    def story(self, p):
        r = self.r
        s = (f"<p>A {p.kind if p.kind != 'Noise only' else 'nothing'} hidden at "
             f"{v(p.snr, '.1f', 'dB')} SNR. The power spectrum (top) rises by only "
             f"{v(r.get('bump', 0), '.2f', 'dB')}: an energy detector would need to know the noise "
             f"level to a fraction of a decibel, which nobody does.</p>"
             "<p>But noise is <b>stationary</b> and modulated signals are <b>cyclostationary</b>: "
             "their spectral components a symbol rate apart are correlated, because the same "
             "symbols produced them. The spectral coherence measures exactly that. Noise gives "
             "a flat floor (orange); the signal gives lines at α = ±R_s and, for BPSK, at twice the "
             "carrier offset in the conjugate version.</p>")
        if p.kind == "QPSK":
            s += ("<p>QPSK's I and Q are independent and equal, so E[x²] = 0: the conjugate line "
                  "disappears while the symbol-rate line stays. The features tell you the "
                  "modulation, not just that something is there.</p>")
        s += (f"<p>The floor of the estimate falls as 1/√(blocks): a longer look reaches lower SNR. "
              f"Feature at R_s: {v(r.get('feat', 0), '.1f', '×')} the floor.</p>")
        return "<h3>Hidden, but periodic</h3>" + s + keybox(
            "Cyclostationary detectors find signals by their rhythm, not their power. Spectrum "
            "sensing and signal intelligence rely on it.")


# =============================================================================== the lab
LAB = st.Lab(33, "Noise and Detection", chapter=3, chapter_title="Random Signals and Noise",
             experiments=[CentralLimit, QTails, RayleighRice, FilteredNoise, BandpassNoise,
                          FriisCascade, YFactor, ROCDetector, Cyclostationary])

if __name__ == "__main__":
    st.run(LAB)
