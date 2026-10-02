"""Lab 10 · MIMO: Diversity, Capacity, Detection and Massive MIMO   (Chapter 19)

Run it:      python labs/lab10_mimo.py
Self-test:   python labs/lab10_mimo.py --selftest

Multiple antennas give three different things, and good engineers never confuse them:
diversity (independent fades averaged, so the error curve gets steeper), array gain (coherent
combining, so the SNR goes up) and spatial multiplexing (several streams through the same
bandwidth). Nine experiments, from watching fades line up (or not) on a live trace, through
water-filling over eigen-channels and the detectors of a MIMO receiver, to the 64-antenna
base stations of 5G. Numbers come from commlib.mimo and commlib.mimokit, the same code as the
book's Chapter 19 figures.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")     # small matrices: avoid thread thrashing

import numpy as np
from scipy.stats import gamma as gamma_dist

import commlib as cl
from commlib import mimokit as mk
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ConstellationPlot, BERPlot, PolarPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

QPSK = cl.get_constellation("qpsk")
QAM16 = cl.get_constellation("16qam")
BRANCH_COLS = [BLUE, ORANGE, GREEN, PURPLE, GOLD, TEAL, RED, GRAY]
COMB = {"Selection": "sc", "Equal gain": "egc", "Maximal ratio": "mrc"}
COMB_COL = {"sc": ORANGE, "egc": GREEN, "mrc": NAVY}


def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def undb(x):
    return 10 ** (np.asarray(x, float) / 10)


_SAMPLES = {}


def branch_samples(L, rho, n=60000):
    """Cached correlated Rayleigh branch draws (the same draws every time: smooth curves)."""
    key = (L, round(rho, 3), n)
    if key not in _SAMPLES:
        if len(_SAMPLES) > 24:
            _SAMPLES.clear()
        _SAMPLES[key] = mk.correlated_branches(L, n, rho, np.random.default_rng(1000 + L))
    return _SAMPLES[key]


def ber_curve(kind, L, rho, g):
    """BPSK BER of a combiner (exact for MRC, sample-based with exact tail for SC/EGC)."""
    if kind == "mrc":
        return mk.ber_bpsk_mrc_corr(g, L, rho)
    if kind == "sc" and rho == 0:
        return mk.ber_bpsk_sc_iid(g, L)
    R = mk.exp_corr(L, rho)
    G = mk.combine_gain(branch_samples(L, rho), kind)
    return mk.ber_bpsk_from_gains(G, g, mk.gain_cdf_asymptote(kind, R), L)


# =============================================================================== 1. fades live
class FadingLive(Experiment):
    title = "Fading branches, live"
    blurb = "Independent fades rarely line up: watch a combiner ride over them."
    book = "sec:ch19:combining"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Heading("Antennas"),
        IntSlider("L", "Receive antennas", 1, 8, 2),
        Choice("comb", "Combiner", list(COMB), "Maximal ratio"),
        Slider("rho", "Branch correlation |ρ|", 0.0, 0.99, 0.0, step=0.01,
               help="Correlation between neighbouring antennas (exponential model)"),
        Heading("Channel"),
        LogSlider("fd", "Doppler spread", 5, 200, 40, unit="Hz", fmt=".0f",
                  help="40 Hz is a pedestrian at 6 GHz or a car at about 20 km/h at 2 GHz"),
        Slider("thr", "Outage threshold", -25.0, 0.0, -10.0, step=0.5, unit="dB",
               help="Count the time the combiner output spends below this level"),
    ]
    plots = [
        Plot("env", "Branch SNRs (thin) and the combiner output (thick)", x="time (ms)",
             y="SNR re mean branch (dB)", xlim=(0, 250), ylim=(-35, 16), legend="tl",
             legend_cols=3),
        Plot("hist", "Output SNR distribution", x="output SNR re mean branch (dB)",
             y="probability density (1/dB)", xlim=(-30, 14), legend="tl"),
        Plot("cdf", "Probability of a fade deeper than x", x="x (dB re mean branch)",
             y="probability", logy=True, xlim=(-30, 10), ylim=(1e-4, 1.5), legend="tl"),
    ]
    layout = [["env", "env"], ["hist", "cdf"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("meas", "Time below threshold (measured)", "%", ".2f"),
        Readout("exp", "Expected", "%", ".2f"),
        Readout("gain", "Mean output gain", "dB", ".1f"),
        Readout("p1", "1 % fade level", "dB", ".1f", good=lambda x: x > -1),
    ]
    challenges = [
        Challenge("Selection combining only: keep the expected time below −10 dB under 1 %.",
                  lambda s: s.p.comb == "Selection" and s.p.thr <= -9.99 and s.r.exp < 1.0,
                  hint="Selection picks the best antenna; all of them must fade together."),
        Challenge("Correlation steals diversity: two MRC branches with |ρ| ≥ 0.9 that spend at "
                  "least 3× as long below −10 dB as independent branches would.",
                  lambda s: (s.p.L == 2 and s.p.comb == "Maximal ratio" and s.p.rho >= 0.9
                             and s.p.thr <= -9.99 and s.r.exp >= 3 * s.exp.exp_iid)),
        Challenge("Lift the 1 % fade level above −1 dB (the output is within 1 dB of an average "
                  "single antenna 99 % of the time) with no more than 4 antennas.",
                  lambda s: s.p.L <= 4 and s.r.p1 > -1,
                  hint="Array gain and diversity both help: which combiner collects both?"),
    ]
    n_win = 500
    win_s = 0.25

    def setup(self):
        r = np.random.default_rng(19)
        self.alpha = r.uniform(0, 2 * np.pi, (8, 24))
        self.phi = r.uniform(0, 2 * np.pi, (8, 24))
        self.t0 = 0.0
        self._stat_key = None

    def branches(self, p, t):
        """Sum-of-sinusoids Rayleigh processes, then correlated across antennas."""
        L = p.L
        ph = (2 * np.pi * p.fd * np.cos(self.alpha[:L, :, None]) * t[None, None, :]
              + self.phi[:L, :, None])
        h = np.exp(1j * ph).sum(axis=1) / np.sqrt(self.alpha.shape[1])
        return mk.corr_sqrt(mk.exp_corr(L, p.rho)) @ h

    def stats(self, p):
        key = (p.L, round(p.rho, 3), p.comb)
        if key != self._stat_key:
            kind = COMB[p.comb]
            G = np.sort(mk.combine_gain(branch_samples(p.L, p.rho), kind))
            self.Gs = G
            hb = np.histogram(db(G), bins=88, range=(-30, 14), density=True)
            self.th_hist = hb
            G1 = np.sort(np.abs(branch_samples(1, 0.0)[0]) ** 2)
            self.G1 = G1
            Gi = np.sort(mk.combine_gain(branch_samples(p.L, 0.0), kind))
            self.Gi = Gi
            self._stat_key = key
        return self.Gs

    def frac_below(self, G, thr_db):
        return float(np.searchsorted(G, undb(thr_db)) / len(G))

    def draw(self, p, accumulate):
        G = self.stats(p)
        t = self.t0 + np.arange(self.n_win) * self.win_s / self.n_win
        h = self.branches(p, t)
        kind = COMB[p.comb]
        out = db(mk.combine_gain(h, kind))
        tm = (t - self.t0) * 1e3
        pe = self.plot("env")
        for l in range(8):
            if l < p.L:
                pe.line(f"b{l}", tm, db(np.abs(h[l]) ** 2), color=BRANCH_COLS[l], width=1.0,
                        alpha=0.75, name="antennas" if l == 0 else None)
        pe.line("out", tm, out, color=NAVY, width=2.6, name=f"{p.comb} output")
        if np.any(out < p.thr):
            pe.line("below", tm, np.where(out < p.thr, out, np.nan), color=RED, width=3.2,
                    name="below threshold")
        pe.hline("thr", p.thr, color=RED, style="--", label=f"threshold {p.thr:.1f} dB",
                 label_pos=0.82)
        pe.hline("z", 0, color=GRAY, style=":", width=0.8)
        # accumulate statistics
        if not accumulate or getattr(self, "acc", None) is None:
            self.acc = dict(n=0, below=0, hist=np.zeros(88))
        a = self.acc
        a["n"] += len(out)
        a["below"] += int(np.sum(out < p.thr))
        a["hist"] += np.histogram(out, bins=88, range=(-30, 14))[0]
        edges = np.linspace(-30, 14, 89)
        ph = self.plot("hist")
        dens = a["hist"] / max(a["hist"].sum(), 1) / (edges[1] - edges[0])
        ph.line("th", edges, self.th_hist[0], step=True, color=GRAY, width=1.4,
                name="expected")
        ph.line("live", edges, dens, step=True, color=NAVY, width=1.8, fill=0, fill_alpha=0.2,
                name="measured (live)")
        ph.vline("thr", p.thr, color=RED, style="--")
        ph.set_ylim(0, max(0.12, 1.25 * float(np.max(self.th_hist[0]))))
        # CDF: one antenna versus the combiner
        x = np.linspace(-30, 10, 161)
        pc = self.plot("cdf")
        F1 = np.searchsorted(self.G1, undb(x)) / len(self.G1)
        F = np.searchsorted(G, undb(x)) / len(G)
        pc.line("one", x, np.maximum(F1, 1e-9), color=GRAY, width=1.6, style="--",
                name="one antenna")
        pc.line("cur", x, np.maximum(F, 1e-9), color=NAVY, width=2.4,
                name=f"{p.L} × {p.comb.lower()}")
        pc.vline("thr", p.thr, color=RED, style="--")
        pc.hline("p1", 0.01, color=GRAY, style=":", width=0.8)
        exp_ = 100 * self.frac_below(G, p.thr)
        self.exp_iid = 100 * self.frac_below(self.Gi, p.thr)
        p1 = float(db(G[int(0.01 * len(G))]))
        self.readout(meas=100 * a["below"] / a["n"], exp=exp_, gain=float(db(np.mean(G))), p1=p1)

    def update(self, p):
        self.acc = None
        self.draw(p, False)

    def tick(self, p):
        self.t0 += 1.0 / self.fps
        self.draw(p, True)

    def story(self, p):
        r = self.r
        if p.L == 1:
            s = ("<p>One antenna in a multipath field: the signal is the sum of many reflections, "
                 "and every few centimetres they cancel. The trace dives 20–30 dB at random "
                 f"moments; at a {v(p.fd, '.0f', 'Hz')} Doppler spread that happens tens of times a "
                 "second. A fade 10 dB down happens about 10 % of the time, 20 dB down 1 %: one "
                 "decade of probability per 10 dB, the signature of <b>Rayleigh fading</b> "
                 "(bottom right, dashed).</p>")
        else:
            s = (f"<p>{v(p.L, 'd')} antennas, each fading on its own (thin lines). The "
                 f"{v(p.comb.lower())} combiner (thick) only fades deeply when <i>all</i> branches "
                 f"fade at once, which is rare: the red stretches below "
                 f"{v(p.thr, '.0f', 'dB')} add up to {v(r.get('meas', 0), '.2f', '%')} of the "
                 f"time (expected {v(r.get('exp', 0), '.2f', '%')}).</p>"
                 f"<p>Bottom right is the whole story: the combiner's fade probability falls "
                 f"{v(p.L, 'd')} decades per 10 dB instead of one. That slope is the "
                 f"<b>diversity order</b>. ")
            if p.comb == "Maximal ratio":
                s += (f"MRC also adds the branch powers, a mean gain of "
                      f"{v(r.get('gain', 0), '.1f', 'dB')}: <b>array gain</b>.</p>")
            elif p.comb == "Selection":
                s += ("Selection just switches to the strongest antenna: the same slope, less "
                      "average gain, and only one receiver chain is needed.</p>")
            else:
                s += ("Equal-gain combining co-phases the branches without weighting them: "
                      "within about a decibel of MRC.</p>")
            if p.rho > 0.5:
                s += (f"<p>With |ρ| = {v(p.rho)} the antennas fade together more often: watch the "
                      "thin traces move in step. Correlation costs gain long before it costs "
                      "the slope.</p>")
        return "<h3>Fades rarely line up</h3>" + s + keybox(
            "Diversity changes the <i>slope</i> of the fade statistics; array gain shifts them. "
            "Half a wavelength of spacing is enough at a handset.")


# =============================================================================== 2. BER and the slope
class DiversityBER(Experiment):
    title = "SC, EGC and MRC error rates"
    blurb = "Each extra independent branch adds a decade per 10 dB: the slope is the diversity."
    book = "sec:ch19:combining"
    controls = [
        IntSlider("L", "Receive antennas", 1, 8, 1),
        Slider("rho", "Branch correlation |ρ|", 0.0, 0.99, 0.0, step=0.01),
        Choice("target", "Target bit error rate", ["10⁻³", "10⁻⁵"]),
        Slider("snr", "Your operating SNR", 0.0, 35.0, 15.0, step=0.5, unit="dB",
               help="Average SNR per antenna (per branch)"),
    ]
    plots = [
        BERPlot("ber", "BPSK in Rayleigh fading", x="average SNR per branch (dB)",
                xlim=(0, 35), ylim=(1e-7, 1), legend="tr"),
        BarPlot("need", "SNR needed at the target", y="SNR per branch (dB)"),
        Plot("slope", "Local slope of the BER curve", x="average SNR per branch (dB)",
             y="decades per 10 dB", xlim=(0, 40), ylim=(0, 9), legend="tl", legend_cols=3),
    ]
    layout = [["ber", "need"], ["ber", "slope"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("mrc", "MRC needs", "dB", ".1f"),
        Readout("gain", "Gain over one antenna", "dB", ".1f"),
        Readout("scpen", "Selection penalty vs MRC", "dB", ".1f"),
        Readout("ber", "MRC BER at your SNR", "", "sci"),
    ]
    challenges = [
        Challenge("At 10⁻⁵, make a second antenna worth more than 20 dB.",
                  lambda s: s.p.L == 2 and s.p.target == "10⁻⁵" and s.r.gain > 20),
        Challenge("Find the correlation at which two-branch MRC loses 2 dB (± 0.2) at 10⁻³ "
                  "compared with independent branches.",
                  lambda s: (s.p.L == 2 and s.p.target == "10⁻³"
                             and abs(s.exp.corr_loss - 2.0) <= 0.2),
                  hint="Move |ρ| slowly; the loss stays small until about 0.7."),
        Challenge("Show why switched diversity was a bargain: with 4 antennas, selection within "
                  "3.5 dB of MRC at 10⁻³.",
                  lambda s: s.p.L == 4 and s.p.target == "10⁻³" and s.r.scpen <= 3.5),
    ]
    sgrid = np.arange(0, 50.01, 0.25)

    def update(self, p):
        g = undb(self.sgrid)
        tgt = 1e-3 if p.target == "10⁻³" else 1e-5
        curves = {k: ber_curve(k, p.L, p.rho, g) for k in ("sc", "egc", "mrc")}
        one = mk.ber_bpsk_mrc_iid(g, 1)
        awgn = mk.qfunc(np.sqrt(2 * g))
        pb = self.plot("ber")
        pb.theory("awgn", self.sgrid, awgn, color=GRAY, style=":", width=1.4, name="no fading")
        pb.theory("one", self.sgrid, one, color=GRAY, style="--", width=1.6, name="one antenna")
        if p.L > 1:
            pb.theory("sc", self.sgrid, curves["sc"], color=ORANGE, width=1.8,
                      name=f"selection, L = {p.L}")
            pb.theory("egc", self.sgrid, curves["egc"], color=GREEN, width=1.8,
                      name=f"equal gain, L = {p.L}")
        pb.theory("mrc", self.sgrid, curves["mrc"], color=NAVY, width=2.6,
                  name=f"maximal ratio, L = {p.L}")
        if p.rho > 0 and p.L > 1:
            pb.theory("iid", self.sgrid, mk.ber_bpsk_mrc_iid(g, p.L), color=NAVY, style="--",
                      width=1.0, name="MRC, independent")
        pb.hline("tgt", tgt, color=RED, style=":", width=1.0)
        pb.vline("snr", p.snr, color=PURPLE, style="--", label=f"{p.snr:.1f} dB", label_pos=0.95)
        need = {k: mk.snr_for_ber(self.sgrid, c, tgt) for k, c in curves.items()}
        need1 = mk.snr_for_ber(self.sgrid, one, tgt)
        needa = mk.snr_for_ber(self.sgrid, awgn, tgt)
        self.corr_loss = (need["mrc"] - mk.snr_for_ber(self.sgrid, mk.ber_bpsk_mrc_iid(g, p.L), tgt))
        # bars
        pn = self.plot("need")
        items = [("one ant.", need1, GRAY), ("SC", need["sc"], ORANGE), ("EGC", need["egc"], GREEN),
                 ("MRC", need["mrc"], NAVY), ("no fading", needa, BLUE)]
        top = 1.0
        for i, (nm, val, col) in enumerate(items):
            val = float(val) if np.isfinite(val) else 45.0
            pn.bars(f"b{i}", [i], [val], width=0.62, color=col)
            pn.text(f"t{i}", i, val, f"{val:.1f}" if val < 45 else "> 40", anchor=(0.5, 1.05),
                    size=8.5, bold=True)
            top = max(top, val)
        pn.set_xticks([(i, it[0]) for i, it in enumerate(items)])
        pn.set_xlim(-0.6, len(items) - 0.4)
        pn.set_ylim(0, top * 1.22 + 2)
        # local slope
        ps = self.plot("slope")
        for k, col, nm in (("sc", ORANGE, "SC"), ("egc", GREEN, "EGC"), ("mrc", NAVY, "MRC")):
            if p.L == 1 and k != "mrc":
                continue
            y = -np.gradient(np.log10(np.maximum(curves[k], 1e-300)), self.sgrid / 10)
            ps.line(k, self.sgrid, y, color=col, width=2.0 if k == "mrc" else 1.4, name=nm)
        ps.hline("L", p.L, color=RED, style="--", label=f"diversity order {p.L}", label_pos=0.6)
        ber_here = float(np.interp(p.snr, self.sgrid, curves["mrc"]))
        self.readout(mrc=need["mrc"], gain=need1 - need["mrc"], scpen=need["sc"] - need["mrc"],
                     ber=ber_here)

    def story(self, p):
        r = self.r
        tgt = p.target
        s = (f"<p>One antenna in Rayleigh fading needs about {v(r.get('mrc', 0) + r.get('gain', 0), '.1f', 'dB')} "
             f"for a BER of {tgt}, against 6.8 dB (10⁻³) without fading: the average error rate is "
             f"set by the rare deep fades, and it falls only one decade per 10 dB.</p>")
        if p.L > 1:
            s += (f"<p>With {v(p.L, 'd')} antennas and maximal-ratio combining the target needs "
                  f"{v(r.get('mrc', 0), '.1f', 'dB')}: a gain of {v(r.get('gain', 0), '.1f', 'dB')}. "
                  f"Only {v(10 * np.log10(p.L), '.1f', 'dB')} of that is array gain; the rest is "
                  f"diversity, and it grows the lower the target. Bottom right: every slope climbs "
                  f"towards {v(p.L, 'd')} decades per 10 dB. Selection is only "
                  f"{v(r.get('scpen', 0), '.1f', 'dB')} behind MRC, with one receiver chain.</p>")
        if p.rho > 0.2 and p.L > 1:
            s += (f"<p>Correlation |ρ| = {v(p.rho)} costs {v(self.corr_loss, '.1f', 'dB')} "
                  f"(navy solid vs dashed): the eigenvalues of the branch covariance spread out, so "
                  f"one 'virtual branch' gets weaker. The slope survives until |ρ| → 1.</p>")
        return "<h3>Steeper, not just shifted</h3>" + s + keybox(
            "P<sub>b</sub> ∝ SNR<sup>−L</sup> at high SNR: diversity order L.")


# =============================================================================== 3. Alamouti
SCHEMES = ["1×1", "2×1 same symbol", "2×1 Alamouti", "1×2 MRC", "2×1 TX beamforming",
           "2×2 Alamouti"]
SCHEME_COL = {"1×1": GRAY, "2×1 same symbol": ORANGE, "2×1 Alamouti": RED,
              "1×2 MRC": NAVY, "2×1 TX beamforming": GREEN, "2×2 Alamouti": PURPLE}


def scheme_ber(name, snr_lin):
    """Theory: QPSK per-bit BER with Gray mapping (each bit sees BPSK at Es/N0/2)."""
    gb = snr_lin / 2
    if name in ("1×1", "2×1 same symbol"):
        return mk.ber_bpsk_mrc_iid(gb, 1)
    if name == "2×1 Alamouti":
        return mk.ber_bpsk_mrc_iid(gb / 2, 2)
    if name in ("1×2 MRC", "2×1 TX beamforming"):
        return mk.ber_bpsk_mrc_iid(gb, 2)
    return mk.ber_bpsk_mrc_iid(gb / 2, 4)


class Alamouti(Experiment):
    title = "Alamouti versus MRC"
    blurb = "Two antennas at the transmitter, none of its channel knowledge: how much is left?"
    book = "sec:ch19:stc"
    controls = [
        Choice("scheme", "Scheme", SCHEMES, "2×1 Alamouti", style="menu"),
        Slider("snr", "SNR per receive antenna", 0.0, 35.0, 12.0, step=0.5, unit="dB",
               help="Total transmit power over the noise at one receive antenna"),
        Slider("eps", "Channel change between the two slots", 0.0, 0.5, 0.0, step=0.01,
               help="Fraction of the fade renewed between the two Alamouti time slots",
               enabled_if=lambda p: "Alamouti" in p.scheme),
        Button("again", "New data"),
    ]
    plots = [
        BERPlot("ber", "QPSK bit error rate", x="SNR per receive antenna (dB)", xlim=(0, 35),
                ylim=(1e-6, 0.5), legend="bl", legend_cols=2),
        ConstellationPlot("con", "Decoded symbols", lim=2.0),
        Plot("eff", "Effective channel gain, block by block", x="block", y="gain (dB)",
             xlim=(0, 150), ylim=(-30, 12), legend="bl", legend_cols=2),
    ]
    layout = [["ber", "con"], ["eff", "eff"]]
    row_stretch = [3, 2]
    col_stretch = [3, 2]
    readouts = [
        Readout("sim", "Simulated BER", "", "sci"),
        Readout("th", "Theory", "", "sci"),
        Readout("need", "Needs at 10⁻³", "dB", ".1f"),
        Readout("div", "Diversity order", "", None),
    ]
    challenges = [
        Challenge("Find the 3 dB: set 2×1 Alamouti to the SNR at which it matches 1×2 MRC at 12 dB "
                  "(± 0.25 dB).",
                  lambda s: s.p.scheme == "2×1 Alamouti" and abs(s.p.snr - 15.0) <= 0.25,
                  hint="Alamouti splits the power between two antennas; MRC collects it twice."),
        Challenge("Break Alamouti's orthogonality: 2×1 Alamouti at 25 dB or more with a "
                  "simulated BER above 10⁻³.",
                  lambda s: (s.p.scheme == "2×1 Alamouti" and s.p.snr >= 25 and s.r.sim > 1e-3),
                  hint="The decoder assumes the channel is the same in both slots."),
        Challenge("Recover the 3 dB without a second receive antenna (the transmitter needs to "
                  "know the channel).",
                  lambda s: s.p.scheme == "2×1 TX beamforming"),
    ]
    nblk = 3000

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def simulate(self, p):
        rng = np.random.default_rng(self.seed)
        nb = self.nblk
        bits = cl.random_bits(4 * nb, rng)
        s = QPSK.modulate(bits).reshape(nb, 2)
        n0 = 1 / undb(p.snr)
        noise = lambda shape: np.sqrt(n0 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        sc = p.scheme
        if sc in ("1×1", "2×1 same symbol", "1×2 MRC", "2×1 TX beamforming"):
            h = mk.cn(rng, (2, nb * 2))
            x = s.reshape(-1)
            if sc == "1×1":
                heff = h[0]
                z = (heff * x + noise(x.shape)) / heff
                gain = np.abs(h[0, ::2]) ** 2
            elif sc == "2×1 same symbol":
                heff = (h[0] + h[1]) / np.sqrt(2)
                z = (heff * x + noise(x.shape)) / heff
                gain = np.abs(heff[::2]) ** 2
            elif sc == "1×2 MRC":
                r = h * x + noise(h.shape)
                z = cl.mrc(r, h)
                gain = np.sum(np.abs(h[:, ::2]) ** 2, axis=0)
            else:
                nrm = np.sqrt(np.sum(np.abs(h) ** 2, axis=0))
                z = (nrm * x + noise(x.shape)) / nrm
                gain = nrm[::2] ** 2
        else:
            nr = 1 if sc == "2×1 Alamouti" else 2
            e = p.eps
            ha = mk.cn(rng, (nr, nb, 2))
            hb = np.sqrt(1 - e ** 2) * ha + e * mk.cn(rng, (nr, nb, 2))
            s1, s2 = s[:, 0], s[:, 1]
            r1 = (ha[..., 0] * s1 + ha[..., 1] * s2) / np.sqrt(2) + noise((nr, nb))
            r2 = (-hb[..., 0] * np.conj(s2) + hb[..., 1] * np.conj(s1)) / np.sqrt(2) + noise((nr, nb))
            hh = (ha + hb) / 2                                # what a block estimate sees
            h1, h2 = hh[..., 0], hh[..., 1]
            g = np.sum(np.abs(h1) ** 2 + np.abs(h2) ** 2, axis=0) / np.sqrt(2)
            z1 = np.sum(np.conj(h1) * r1 + h2 * np.conj(r2), axis=0) / g
            z2 = np.sum(np.conj(h2) * r1 - h1 * np.conj(r2), axis=0) / g
            z = np.stack([z1, z2], 1).reshape(-1)
            gain = np.sum(np.abs(ha[..., 0]) ** 2 + np.abs(ha[..., 1]) ** 2, axis=0) / 2
        ber = float(np.mean(QPSK.demodulate(z) != bits))
        return z, ber, gain

    def update(self, p):
        x = np.arange(0, 35.01, 0.25)
        g = undb(x)
        pb = self.plot("ber")
        for nm in SCHEMES:
            cur = nm == p.scheme
            if nm == "2×1 same symbol" and not cur:
                continue
            pb.theory(nm, x, scheme_ber(nm, g), color=SCHEME_COL[nm], width=2.8 if cur else 1.3,
                      style="-" if cur or nm in ("1×1",) else "--", name=nm)
        pb.hline("e3", 1e-3, color=GRAY, style=":", width=0.8)
        z, ber, gain = self.simulate(p)
        th = float(scheme_ber(p.scheme, undb(p.snr)))
        pb.scatter("pt", [p.snr], [max(ber, 1e-7)], color=RED, size=12, symbol="d",
                   name="simulated here")
        pc = self.plot("con")
        pc.points("z", z[:2500], color=SCHEME_COL[p.scheme], size=3, alpha=0.4)
        pc.ideal("i", QPSK.points)
        pc.set_title(f"Decoded symbols, {p.scheme}")
        pe = self.plot("eff")
        k = np.arange(150)
        rng = np.random.default_rng(self.seed + 99)
        ref = np.abs(mk.cn(rng, 150)) ** 2
        pe.line("ref", k, db(ref), color=GRAY, width=1.0, name="one antenna (for comparison)")
        pe.line("cur", k, db(gain[:150]), color=SCHEME_COL[p.scheme], width=2.2, name=p.scheme)
        pe.hline("z", 0, color=GRAY, style=":", width=0.8)
        need = mk.snr_for_ber(x, scheme_ber(p.scheme, g), 1e-3)
        div = {"1×1": "1", "2×1 same symbol": "1", "2×1 Alamouti": "2", "1×2 MRC": "2",
               "2×1 TX beamforming": "2", "2×2 Alamouti": "4"}[p.scheme]
        self.readout(sim=ber, th=th, need=need, div=div)

    def story(self, p):
        r = self.r
        sc = p.scheme
        if sc == "1×1":
            s = ("<p>The reference: one antenna at each end. About 27 dB for 10⁻³, because the "
                 "rare deep fades (bottom, grey) dominate the average error rate.</p>")
        elif sc == "2×1 same symbol":
            s = ("<p>The obvious idea fails: send the same symbol from both antennas and the two "
                 "copies add with random phases. The sum h₁ + h₂ is just another Rayleigh "
                 "coefficient (orange trace, as deep as the grey one): no diversity at all. "
                 "Delaying one copy (cyclic delay diversity) or coding across time is needed.</p>")
        elif "Alamouti" in sc:
            s = (f"<p>Alamouti (1998) sends s₁, s₂ in one slot and −s₂*, s₁* in the next. The "
                 f"receiver's linear combination separates them perfectly, and each symbol sees "
                 f"|h₁|² + |h₂|²: full diversity with no channel knowledge at the transmitter. "
                 f"The price is power: each antenna sends half, so 2×1 Alamouti is exactly 3 dB "
                 f"behind 1×2 MRC (it needs {v(r.get('need', 0), '.1f', 'dB')} at 10⁻³).</p>")
            if p.eps > 0:
                s += (f"<p>With {v(100 * p.eps, '.0f', '%')} of the fade renewed between slots the "
                      f"code is no longer orthogonal: each symbol leaks into the other, the "
                      f"constellation smears, and an error floor appears at high SNR "
                      f"({v(r.get('sim', 0), '.1e')} here).</p>")
        elif sc == "1×2 MRC":
            s = ("<p>Two receive antennas and MRC: the same diversity as Alamouti, plus 3 dB of "
                 "array gain because both antennas collect the full transmitted power.</p>")
        else:
            s = ("<p>If the transmitter knows h (TDD reciprocity or feedback), it can pre-rotate "
                 "and weight the two signals so that they add in phase at the receiver: "
                 "<b>maximum-ratio transmission</b>, identical to 1×2 MRC. Channel knowledge buys "
                 "back the 3 dB.</p>")
        return "<h3>Diversity from the transmitter</h3>" + s + keybox(
            "Alamouti: rate one, diversity two, no CSI at the transmitter, linear decoding. It "
            "is in GSM EDGE, Wi-Fi (STBC) and LTE transmit diversity.")


# =============================================================================== 4. water-filling
class EigenWaterfill(Experiment):
    title = "Eigen-channels and water-filling"
    blurb = "The SVD turns a matrix channel into parallel pipes; pour power where it pays."
    book = "sec:ch19:capacity"
    controls = [
        IntSlider("nt", "Transmit antennas", 1, 8, 4),
        IntSlider("nr", "Receive antennas", 1, 8, 4),
        Slider("snr", "SNR", -10.0, 30.0, 10.0, step=0.5, unit="dB"),
        Slider("rho", "Antenna correlation ρ", 0.0, 0.99, 0.0, step=0.01,
               help="Exponential correlation at both ends (Kronecker model)"),
        Button("again", "New channel"),
    ]
    plots = [
        BarPlot("wf", "Water-filling over the eigen-channels", y="power and 1/λ"),
        Plot("cap", "This channel's capacity", x="SNR (dB)", y="capacity (b/s/Hz)",
             xlim=(-10, 30), legend="tl"),
        Plot("share", "Share of the power per eigen-channel", x="SNR (dB)",
             y="fraction of the power", xlim=(-10, 30), ylim=(0, 1.05), legend="tr"),
    ]
    layout = [["wf", "cap"], ["wf", "share"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("cwf", "Water-filling", "b/s/Hz", ".2f"),
        Readout("cep", "Equal power", "b/s/Hz", ".2f"),
        Readout("gain", "Water-filling gain", "b/s/Hz", ".2f"),
        Readout("act", "Eigen-channels used", "", None),
    ]
    challenges = [
        Challenge("Make channel knowledge at the transmitter worth at least 1 b/s/Hz "
                  "(water-filling over equal power).",
                  lambda s: s.r.gain >= 1.0,
                  hint="Water-filling helps most when some eigen-channels are hopeless: "
                       "low SNR, correlation, or more transmit than receive antennas."),
        Challenge("Push a 4×4 transmitter into pure beamforming: one eigen-channel in use at an "
                  "SNR of 5 dB or more.",
                  lambda s: s.p.nt == 4 and s.p.nr == 4 and s.p.snr >= 5 and s.exp.n_act == 1),
        Challenge("4×4, no correlation: find the lowest SNR (within 1 dB) at which all four "
                  "eigen-channels get power.",
                  lambda s: (s.p.nt == 4 and s.p.nr == 4 and s.p.rho == 0 and s.exp.n_act == 4
                             and s.p.snr <= s.exp.snr_all + 1.0)),
    ]

    def setup(self):
        self.seed = 7

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        H = mk.kron_channels(1, p.nr, p.nt, p.rho, p.rho, rng=self.seed * 101 + p.nt * 9 + p.nr)[0]
        lam = np.sort(np.linalg.svd(H, compute_uv=False) ** 2)[::-1]
        k = len(lam)
        P = float(undb(p.snr))
        pw, mu, nact = mk.waterfill_level(lam, P)
        cwf = float(np.sum(np.log2(1 + pw * lam)))
        cep = float(np.sum(np.log2(1 + P / p.nt * lam)))
        self.n_act = nact
        # lowest total power that activates the weakest eigen-channel
        self.snr_all = float(db(np.sum(1 / lam[-1] - 1 / lam))) if k > 1 else -99.0
        pb = self.plot("wf")
        floor = 1 / np.maximum(lam, 1e-12)
        top = 1.7 * mu
        for i in range(k):
            fl = min(floor[i], top * 1.5)
            pb.bars(f"f{i}", [i], [fl], width=0.7, color=GRAY, alpha=0.55)
            if pw[i] > 0:
                pb.bars(f"p{i}", [i], [floor[i] + pw[i]], base=floor[i], width=0.7, color=BLUE)
                pb.text(f"t{i}", i, floor[i] + pw[i], f"p = {pw[i]:.2g}", anchor=(0.5, 1.05),
                        size=8.5, bold=True, color=BLUE)
            else:
                pb.bars(f"p{i}", [i], [0], width=0.01, color=BLUE)
                pb.text(f"t{i}", i, min(fl, top * 0.92), "dry", anchor=(0.5, 1.05), size=8.5,
                        color=RED, bold=True)
        pb.hline("mu", mu, color=ORANGE, style="--", width=2.0)
        pb.set_title(f"Water-filling: water level μ = {mu:.3g} (orange)")
        pb.set_xticks([(i, f"λ{i + 1} {db(lam[i]):+.0f} dB") for i in range(k)])
        pb.set_xlim(-0.6, k - 0.4)
        pb.set_ylim(0, top)
        # capacity vs SNR
        sg = np.linspace(-10, 30, 81)
        Pg = undb(sg)
        cw = np.array([np.sum(np.log2(1 + mk.waterfill_level(lam, q)[0] * lam)) for q in Pg])
        ce = np.sum(np.log2(1 + Pg[:, None] / p.nt * lam[None, :]), axis=1)
        pc = self.plot("cap")
        pc.line("siso", sg, np.log2(1 + Pg), color=GRAY, style=":", width=1.4,
                name="one antenna pair, unit gain")
        pc.line("ep", sg, ce, color=RED, style="--", width=1.8, name="equal power (no CSI)")
        pc.line("wf", sg, cw, color=NAVY, width=2.4, name="water-filling (CSI at TX)")
        pc.vline("now", p.snr, color=PURPLE, style=":")
        pc.scatter("pt", [p.snr], [cwf], color=PURPLE, size=10)
        pc.set_ylim(0, max(3.0, 1.12 * cw[-1]))
        # power share
        sh = np.array([mk.waterfill_level(lam, q)[0] / q for q in Pg])
        psh = self.plot("share")
        cols = [NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GOLD, TEAL]
        for i in range(8):
            if i < k:
                psh.line(f"s{i}", sg, sh[:, i], color=cols[i], width=1.8, name=f"λ{i + 1}")
        psh.vline("now", p.snr, color=PURPLE, style=":")
        self.readout(cwf=cwf, cep=cep, gain=cwf - cep, act=f"{nact} of {k}")
        self.lam = lam

    def story(self, p):
        r = self.r
        lam = self.lam
        s = (f"<p>The singular-value decomposition H = UΣVᴴ turns this {v(p.nr, 'd')}×{v(p.nt, 'd')} "
             f"channel into {v(len(lam), 'd')} independent pipes with power gains λᵢ from "
             f"{v(db(lam[0]), '+.1f', 'dB')} down to {v(db(lam[-1]), '+.1f', 'dB')}. A transmitter "
             f"that knows H sends along V and pours its power like water into a vessel whose floor "
             f"is 1/λᵢ (grey): deep where the pipe is good, dry where it is hopeless.</p>")
        if r.get("gain", 0) > 0.3:
            s += (f"<p>Here that is worth {v(r.get('gain', 0), '.2f', 'b/s/Hz')} over spreading the "
                  f"power equally: some eigen-channels are too weak to deserve it.</p>")
        else:
            s += ("<p>At this SNR the water is deep and the allocation is nearly equal: knowing the "
                  "channel at the transmitter is worth little. Lower the SNR or raise the "
                  "correlation and watch the weak pipes run dry (bottom right).</p>")
        if p.rho > 0.6:
            s += (f"<p>Correlation ρ = {v(p.rho)} squeezes the eigenvalues apart: the channel "
                  f"behaves like fewer antennas.</p>")
        return "<h3>Parallel pipes</h3>" + s + keybox(
            "pᵢ = (μ − 1/λᵢ)⁺. Low SNR: beamform on the best eigen-channel. High SNR: spread "
            "almost equally, and capacity grows as min(Nt, Nr)·log₂ SNR.")


# =============================================================================== 5. ergodic and outage
CHANS = ["i.i.d. Rayleigh", "Correlated", "Keyhole"]


class ErgodicOutage(Experiment):
    title = "Ergodic and outage capacity"
    blurb = "More antennas move the capacity distribution right and pull in its tail."
    book = "sec:ch19:capacity"
    controls = [
        Choice("chan", "Channel", CHANS, "i.i.d. Rayleigh", style="menu"),
        IntSlider("nt", "Transmit antennas", 1, 8, 2),
        IntSlider("nr", "Receive antennas", 1, 8, 2),
        Slider("rho", "Correlation ρ", 0.0, 0.99, 0.7, step=0.01,
               enabled_if=lambda p: p.chan.startswith("Corr")),
        Slider("snr", "SNR per receive antenna", -5.0, 35.0, 10.0, step=0.5, unit="dB"),
        Slider("eps", "Outage probability", 1.0, 50.0, 10.0, step=1.0, unit="%"),
    ]
    plots = [
        Plot("cdf", "Capacity distribution at your SNR", x="capacity (b/s/Hz)", y="CDF",
             ylim=(0, 1.02), legend="br"),
        Plot("erg", "Ergodic capacity versus SNR", x="SNR per receive antenna (dB)",
             y="b/s/Hz", xlim=(-5, 35), legend="tl"),
    ]
    layout = [["cdf", "erg"]]
    readouts = [
        Readout("erg", "Ergodic capacity", "b/s/Hz", ".2f"),
        Readout("out", "Outage capacity C_ε", "b/s/Hz", ".2f"),
        Readout("ratio", "Ergodic ÷ 1×1", "×", ".2f"),
        Readout("slope", "High-SNR slope", "b/s/Hz per 3 dB", ".2f"),
    ]
    challenges = [
        Challenge("Reproduce the book's 4×4 worked example: i.i.d. at 20 dB, ergodic capacity "
                  "about 22 b/s/Hz and 10 % outage capacity near 19.6.",
                  lambda s: (s.p.chan == CHANS[0] and s.p.nt == 4 and s.p.nr == 4
                             and abs(s.p.snr - 20) < 0.3 and abs(s.p.eps - 10) < 0.5)),
        Challenge("Make a 4×4 link behave like a single pair of antennas: high-SNR slope below "
                  "1.2 b/s/Hz per 3 dB.",
                  lambda s: s.p.nt == 4 and s.p.nr == 4 and s.r.slope < 1.2,
                  hint="Which channel has only one path from transmitter to receiver?"),
        Challenge("At 10 dB, get a 1 % outage capacity of at least 5 b/s/Hz with no more than "
                  "6 antennas in total.",
                  lambda s: (s.p.nt + s.p.nr <= 6 and abs(s.p.eps - 1) < 0.5
                             and abs(s.p.snr - 10) < 0.3 and s.r.out >= 5.0),
                  hint="Diversity pulls in the tail; multiplexing moves the median."),
    ]
    ndraw = 4000

    def setup(self):
        self._cache = {}

    def eigs(self, chan, nt, nr, rho):
        key = (chan, nt, nr, round(rho, 3) if chan.startswith("Corr") else 0)
        if key not in self._cache:
            if len(self._cache) > 30:
                self._cache.clear()
            rng = np.random.default_rng(55 + nt * 10 + nr)
            if chan == CHANS[0]:
                H = mk.kron_channels(self.ndraw, nr, nt, 0, 0, rng)
            elif chan == CHANS[1]:
                H = mk.kron_channels(self.ndraw, nr, nt, rho, rho, rng)
            else:
                H = mk.keyhole_channels(self.ndraw, nr, nt, rng)
            self._cache[key] = mk.eig_stack(H)
        return self._cache[key]

    def update(self, p):
        lam = self.eigs(p.chan, p.nt, p.nr, p.rho)
        lam1 = self.eigs(CHANS[0], 1, 1, 0)
        snr = float(undb(p.snr))
        C = np.sort(mk.capacity_from_eigs(lam, snr, p.nt))
        C1 = np.sort(mk.capacity_from_eigs(lam1, snr, 1))
        F = (np.arange(len(C)) + 1) / len(C)
        pc = self.plot("cdf")
        pc.line("one", C1, F, color=GRAY, width=1.6, style="--", name="1×1")
        if p.chan != CHANS[0]:
            Ci = np.sort(mk.capacity_from_eigs(self.eigs(CHANS[0], p.nt, p.nr, 0), snr, p.nt))
            pc.line("iid", Ci, F, color=GREEN, width=1.4, style="--",
                    name=f"{p.nt}×{p.nr} i.i.d.")
        pc.line("cur", C, F, color=NAVY, width=2.6, name=f"{p.nt}×{p.nr} {p.chan.split(' ')[0]}")
        eps = p.eps / 100
        cout = float(np.quantile(C, eps))
        pc.hline("eps", eps, color=RED, style=":", label=f"ε = {p.eps:.0f} %", label_pos=0.85)
        pc.vline("cout", cout, color=RED, style="--", label=f"C_ε = {cout:.2f}", label_pos=0.6)
        pc.set_xlim(0, max(4.0, 1.08 * C[-1]))
        sg = np.linspace(-5, 35, 41)
        erg = mk.capacity_from_eigs(lam, undb(sg), p.nt).mean(axis=1)
        erg1 = mk.capacity_from_eigs(lam1, undb(sg), 1).mean(axis=1)
        pe = self.plot("erg")
        pe.line("one", sg, erg1, color=GRAY, width=1.6, style="--", name="1×1")
        if p.chan != CHANS[0]:
            ei = mk.capacity_from_eigs(self.eigs(CHANS[0], p.nt, p.nr, 0), undb(sg), p.nt).mean(axis=1)
            pe.line("iid", sg, ei, color=GREEN, width=1.4, style="--", name=f"{p.nt}×{p.nr} i.i.d.")
        pe.line("cur", sg, erg, color=NAVY, width=2.6, name=f"{p.nt}×{p.nr} {p.chan.split(' ')[0]}")
        m = min(p.nt, p.nr)
        asym = erg[-1] + m * np.log2(10) / 10 * (sg - sg[-1])
        pe.line("asym", sg, np.where(asym > 0, asym, np.nan), color=RED, width=1.0, style=":",
                name=f"slope min(Nt,Nr) = {m}")
        pe.vline("now", p.snr, color=PURPLE, style=":")
        pe.set_ylim(0, max(10.0, 1.08 * erg[-1]))
        erg_now = float(C.mean())
        slope = float((erg[-1] - erg[-11]) / 10 * 3)
        self.readout(erg=erg_now, out=cout, ratio=erg_now / max(float(C1.mean()), 1e-9), slope=slope)

    def story(self, p):
        r = self.r
        m = min(p.nt, p.nr)
        s = (f"<p>Each curve on the left is the capacity of {self.ndraw:,} random channels, sorted. "
             f"The {v(p.nt, 'd')}×{v(p.nr, 'd')} link averages {v(r.get('erg', 0), '.2f', 'b/s/Hz')}: "
             f"{v(r.get('ratio', 0), '.1f', '×')} a single antenna pair. Its "
             f"{v(p.eps, '.0f', '%')}-outage capacity, the rate a slowly fading link can promise "
             f"{100 - p.eps:.0f} % of the time, is {v(r.get('out', 0), '.2f', 'b/s/Hz')}.</p>"
             f"<p>Two effects: the curve moves right (more streams: <b>multiplexing</b>) and gets "
             f"steeper (more fading coefficients averaged: <b>diversity</b>). At high SNR capacity "
             f"grows by min(Nt, Nr) = {v(m, 'd')} bits per 3 dB; here the slope is "
             f"{v(r.get('slope', 0), '.2f')}.</p>")
        if p.chan.startswith("Key"):
            s += ("<p>The <b>keyhole</b> (Chizhik, Foschini and Valenzuela, 2000): rich scattering at "
                  "both ends but a single narrow opening in between, like a corridor. Every "
                  "antenna fades, yet the matrix has rank one: one stream only.</p>")
        elif p.chan.startswith("Corr"):
            s += (f"<p>Correlation ρ = {v(p.rho)} (closely spaced antennas, little angular spread) "
                  f"makes the eigenvalues unequal: the green i.i.d. curve shows what is lost.</p>")
        return "<h3>Faster and more reliable</h3>" + s + keybox(
            "C = log₂ det(I + (ρ/Nt) HHᴴ). Ergodic capacity for fast fading, outage capacity "
            "for slow fading; both grow as min(Nt, Nr)·log₂ ρ.")


# =============================================================================== 6. detectors
CONFIGS = {"2×2, QPSK": (2, 2, QPSK), "4×4, QPSK": (4, 4, QPSK),
           "2×2, 16-QAM": (2, 2, QAM16), "2×4, QPSK": (2, 4, QPSK)}
DET_COL = {"ZF": ORANGE, "MMSE": GREEN, "MMSE-SIC": PURPLE, "ML": NAVY}


class Detectors(Experiment):
    title = "ZF, MMSE, SIC and ML"
    blurb = "Several streams through one channel: four ways to pull them apart, four slopes."
    book = "sec:ch19:detection"
    heavy = True
    controls = [
        Choice("cfg", "Configuration", list(CONFIGS), "2×2, QPSK", style="menu"),
        Slider("snr", "SNR for the constellations", 0.0, 30.0, 12.0, step=0.5, unit="dB"),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [
        BERPlot("ser", "Symbol error rate", x="SNR per receive antenna (dB)", y="symbol error rate",
                xlim=(-0.5, 30.5), ylim=(1e-5, 1), legend="bl"),
        ConstellationPlot("zf", "After zero forcing", lim=2.0),
        ConstellationPlot("mm", "After MMSE", lim=2.0),
    ]
    layout = [["ser", "zf"], ["ser", "mm"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("gap", "ML gain over ZF at 10⁻²", "dB", ".1f"),
        Readout("enh", "ZF noise enhancement (median)", "dB", ".1f"),
        Readout("cands", "ML candidates per vector", "", "int"),
        Readout("points", "SNR points done", "", None),
    ]
    challenges = [
        Challenge("A thorough run: measure an ML advantage of at least 6 dB over ZF at a symbol "
                  "error rate of 10⁻² (2×2 QPSK).",
                  lambda s: (s.p.cfg == "2×2, QPSK" and s.p.effort == "Thorough"
                             and s.r.gap is not None and s.r.gap >= 6.0),
                  hint="Wait for the run to reach 10⁻² for both detectors."),
        Challenge("Give zero forcing its diversity back: a configuration where the median ZF noise "
                  "enhancement is below 2 dB.",
                  lambda s: s.r.enh < 2.0,
                  hint="ZF's diversity is Nr − Nt + 1: spare receive antennas."),
        Challenge("See exponential complexity: a configuration whose ML search tests 256 "
                  "candidates per received vector.",
                  lambda s: s.r.cands == 256),
    ]

    def setup(self):
        self.run_id = 0
        self.sim = None

    def on_rerun(self, p):
        self.run_id += 1

    def channel_vectors(self, nt, nr, con, n, n0, rng):
        H = mk.cn(rng, (n, nr, nt))
        X = con.points[rng.integers(0, con.M, (n, nt))]
        Y = np.einsum("nij,nj->ni", H, X) + np.sqrt(n0 / 2) * (
            rng.standard_normal((n, nr)) + 1j * rng.standard_normal((n, nr)))
        return H, X, Y

    def update(self, p):
        nt, nr, con = CONFIGS[p.cfg]
        rng = np.random.default_rng(3)
        n0 = nt * 10 ** (-p.snr / 10)
        H, X, Y = self.channel_vectors(nt, nr, con, 700, n0, rng)
        zf = mk.batch_zf(H, Y)
        mm = mk.batch_mmse(H, Y, n0)
        lim = 2.0
        for key, z, col in (("zf", zf, ORANGE), ("mm", mm, GREEN)):
            pc = self.plot(key)
            pc.points("z", z.reshape(-1), color=col, size=3, alpha=0.45)
            pc.ideal("i", con.points)
            pc.set_xlim(-lim, lim)
            pc.set_ylim(-lim, lim)
        self.plot("zf").set_title(f"After zero forcing, {p.snr:.1f} dB")
        self.plot("mm").set_title(f"After MMSE, {p.snr:.1f} dB")
        # ZF noise enhancement: per stream, ||h_k||^2 [(H^H H)^-1]_kk (>= 1)
        Hh = np.swapaxes(H.conj(), -1, -2)
        inv = np.linalg.inv(Hh @ H)
        enh = np.real(np.diagonal(inv, axis1=1, axis2=2)) * np.sum(np.abs(H) ** 2, axis=1)
        self.sim = dict(x=[], y={k: [] for k in DET_COL}, done=False)
        pb = self.plot("ser")
        pb.hline("e2", 1e-2, color=GRAY, style=":", width=0.8)
        self.readout(enh=float(db(np.median(enh))), cands=con.M ** nt, gap=None, points="0")

    def background(self, p):
        nt, nr, con = CONFIGS[p.cfg]
        thorough = p.effort == "Thorough" and not self.quick
        target = 20 if self.quick else (300 if thorough else 80)
        maxv = 4000 if self.quick else (300000 if thorough else 60000)
        rng = np.random.default_rng(self.run_id * 7919 + 10)
        cands = mk.ml_candidates(con.points, nt)
        alive = set(DET_COL)
        snrs = np.arange(0, 30.1, 3.0) if not self.quick else np.arange(0, 30.1, 6.0)
        for snr in snrs:
            n0 = nt * 10 ** (-snr / 10)
            err = {k: 0 for k in DET_COL}
            tot = 0
            while tot < maxv and min(err[k] for k in alive) < target:
                H, X, Y = self.channel_vectors(nt, nr, con, 2000, n0, rng)
                outs = {"ZF": mk.slice_to(mk.batch_zf(H, Y), con.points),
                        "MMSE": mk.slice_to(mk.batch_mmse(H, Y, n0), con.points),
                        "MMSE-SIC": mk.batch_mmse_sic(H, Y, n0, con.points),
                        "ML": mk.batch_ml(H, Y, cands)}
                for k in DET_COL:
                    err[k] += int(np.sum(np.abs(outs[k] - X) > 1e-9))
                tot += X.size
            yield dict(snr=snr, ser={k: err[k] / tot for k in DET_COL}, n=tot)
            for k in list(alive):
                if err[k] / tot < 2e-5 or err[k] == 0:
                    alive.discard(k)
            if not alive:
                break
        yield dict(done=True)

    def progress(self, p, it):
        sim = self.sim
        if it.get("done"):
            sim["done"] = True
        else:
            sim["x"].append(it["snr"])
            for k in DET_COL:
                sim["y"][k].append(it["ser"][k])
        pb = self.plot("ser")
        x = np.array(sim["x"])
        for k, col in DET_COL.items():
            y = np.array(sim["y"][k], float)
            if len(y):
                y = np.where(y > 0, y, np.nan)
                pb.sim(k, x, y, color=col, name=k)
        self.readout(gap=self.gap(), points=f"{len(sim['x'])}" + (" ✓" if sim["done"] else " …"))

    def gap(self):
        sim = self.sim
        x = np.array(sim["x"])
        if len(x) < 2:
            return None
        out = {}
        for k in ("ZF", "ML"):
            y = np.array(sim["y"][k], float)
            ok = y > 0
            if ok.sum() < 2 or y[ok].min() > 1e-2 or y[ok].max() < 1e-2:
                return None
            out[k] = mk.snr_for_ber(x[ok], y[ok], 1e-2)
            if not np.isfinite(out[k]):
                return None
        return float(out["ZF"] - out["ML"])

    def story(self, p):
        nt, nr, con = CONFIGS[p.cfg]
        r = self.r
        s = (f"<p>{v(nt, 'd')} streams leave {v(nt, 'd')} antennas and arrive mixed at "
             f"{v(nr, 'd')}. <b>Zero forcing</b> inverts the channel: the streams separate, but "
             f"when H is nearly singular the inversion amplifies the noise (median "
             f"{v(r.get('enh', 0), '.1f', 'dB')} here; see the smeared cloud at the top right). "
             f"<b>MMSE</b> regularises the inverse and smears less. <b>SIC</b> detects the most "
             f"reliable stream first and subtracts it. <b>ML</b> tries all "
             f"{v(con.M ** nt, ',d')} candidate vectors.</p>"
             f"<p>The slopes tell the story: linear detectors get diversity Nr − Nt + 1 = "
             f"{v(nr - nt + 1, 'd')}, ML gets Nr = {v(nr, 'd')}. ")
        if r.get("gap") is not None:
            s += f"At 10⁻², ML is {v(r.get('gap'), '.1f', 'dB')} ahead of ZF.</p>"
        else:
            s += "The Monte Carlo run fills in the curves point by point.</p>"
        return "<h3>Untangling streams</h3>" + s + keybox(
            "Real receivers sit between MMSE and ML: sphere decoders and K-best searches reach "
            "near-ML performance at a fraction of the M<sup>Nt</sup> cost.")


# =============================================================================== 7. DMT
class DMT(Experiment):
    title = "The diversity–multiplexing trade-off"
    blurb = "Spend the channel's randomness on rate or on reliability: Zheng and Tse's curve."
    book = "sec:ch19:dmt"
    heavy = True                 # a new antenna count draws 10⁵ channels (once, then cached)
    controls = [
        IntSlider("nt", "Transmit antennas", 1, 4, 2),
        IntSlider("nr", "Receive antennas", 1, 4, 2),
        Slider("r", "Multiplexing gain r", 0.3, 3.9, 1.0, step=0.05,
               help="The rate grows as r·log₂(SNR): r = 1 is one 'stream' worth of growth"),
    ]
    plots = [
        Plot("dmt", "Optimal trade-off d*(r)", x="multiplexing gain r", y="diversity gain d",
             xlim=(0, 4.2), ylim=(0, 17), legend="tr"),
        BERPlot("out", "Outage probability for rate r·log₂(SNR)", x="SNR (dB)",
                y="outage probability", xlim=(0, 60), ylim=(1e-5, 1), legend="bl"),
    ]
    layout = [["dmt", "out"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("d", "d*(r)", "", ".2f"),
        Readout("meas", "Measured outage slope", "decades / 10 dB", ".2f"),
        Readout("rate", "Rate at 30 dB", "b/s/Hz", ".1f"),
        Readout("dmax", "Maximum diversity Nt·Nr", "", "int"),
    ]
    challenges = [
        Challenge("With a 2×2, measure an outage slope of at least 2 decades per 10 dB.",
                  lambda s: s.p.nt == 2 and s.p.nr == 2 and s.r.meas is not None and s.r.meas >= 2.0,
                  hint="Ask for little rate growth: small r."),
        Challenge("Use a 2×2 at nearly full multiplexing (r ≥ 1.8) and watch the diversity "
                  "collapse: measured slope below 0.3.",
                  lambda s: (s.p.nt == 2 and s.p.nr == 2 and s.p.r >= 1.8 and s.r.meas is not None
                             and s.r.meas < 0.3)),
        Challenge("Find r for which a 4×4 still offers diversity 4 (d*(r) = 4 ± 0.1).",
                  lambda s: s.p.nt == 4 and s.p.nr == 4 and abs(s.r.d - 4) <= 0.1),
    ]
    sg = np.arange(0, 60.1, 2.5)

    def setup(self):
        self._cache = {}

    def eigs(self, nt, nr):
        key = (nt, nr)
        if key not in self._cache:
            n = 40000 if self.quick else (200000 if nt * nr <= 6 else 100000)
            H = mk.kron_channels(n, nr, nt, 0, 0, np.random.default_rng(77 + nt * 5 + nr))
            self._cache[key] = mk.eig_stack(H).astype(np.float32)
        return self._cache[key]

    def outage(self, lam, nt, r):
        snr = undb(self.sg)
        R = r * np.log2(snr)
        out = np.empty(len(snr))
        for i, (s_, R_) in enumerate(zip(snr, R)):
            C = np.sum(np.log2(1 + np.float32(s_ / nt) * lam), axis=1)
            out[i] = np.mean(C < R_)
        pk = int(np.argmax(out))
        out[:pk] = np.nan                     # below the peak the rate is still tiny: not the regime
        return out

    def update(self, p):
        m = min(p.nt, p.nr)
        r = min(p.r, m - 0.05)
        k, d = mk.dmt_optimal(p.nt, p.nr)
        pd = self.plot("dmt")
        pd.line("opt", k, d, color=NAVY, width=2.6, name=f"optimal {p.nt}×{p.nr}")
        pd.scatter("corner", k, d, color=NAVY, size=7)
        if p.nt == 2 and p.nr == 2:
            pd.line("rep", [0, 0.5], [4, 0], color=GRAY, style="--", width=1.4, name="repetition")
            pd.line("ala", [0, 1], [4, 0], color=RED, width=1.6, name="Alamouti")
            pd.line("vml", [0, 2], [2, 0], color=GREEN, width=1.6, name="V-BLAST, ML")
            pd.line("vzf", [0, 2], [1, 0], color=ORANGE, width=1.6, name="V-BLAST, ZF")
        dstar = mk.dmt_value(r, p.nt, p.nr)
        pd.scatter("now", [r], [dstar], color=RED, size=13, symbol="d", name="your r")
        pd.set_xlim(0, max(2.2, m + 0.2))
        pd.set_ylim(0, p.nt * p.nr * 1.15 + 0.5)
        lam = self.eigs(p.nt, p.nr)
        pout = self.outage(lam, p.nt, r)
        po = self.plot("out")
        po.sim("cur", self.sg, np.where(pout > 0, pout, np.nan), color=NAVY, name=f"r = {r:.2f}")
        floor = 10.0 / len(lam)
        ok = np.flatnonzero(np.isfinite(pout) & (pout > floor) & (pout < 0.3))
        meas = None
        if len(ok) >= 2:
            j = ok[-3:] if len(ok) >= 3 else ok
            meas = float(-np.polyfit(self.sg[j] / 10, np.log10(pout[j]), 1)[0])
            i = ok[-1]
            snr = undb(self.sg)
            ref = pout[i] * (snr / snr[i]) ** (-dstar)
            po.theory("ref", self.sg, np.where((ref < 0.5) & (ref > 1e-6), ref, np.nan), color=RED,
                      style=":", width=1.8, name=f"slope d*(r) = {dstar:.2f}")
        po.hline("floor", floor, color=GRAY, style=":", width=0.8, label="simulation floor",
                 label_pos=0.75)
        self.readout(d=dstar, meas=meas, rate=r * np.log2(1000), dmax=p.nt * p.nr)

    def story(self, p):
        r = self.r
        m = min(p.nt, p.nr)
        s = (f"<p>A code whose rate grows as r·log₂(SNR) uses r of the channel's {v(m, 'd')} spatial "
             f"degrees of freedom. Its error probability can fall no faster than SNR<sup>−d*(r)</sup>, "
             f"with d*(r) joining the points (k, (Nt−k)(Nr−k)) (left). At r = {v(min(p.r, m - 0.05))} "
             f"that is {v(r.get('d', 0), '.2f')}; the full diversity Nt·Nr = {v(p.nt * p.nr, 'd')} is "
             f"only available at a fixed rate (r → 0).</p>"
             f"<p>The right plot is the experiment: the probability that a random "
             f"{v(p.nt, 'd')}×{v(p.nr, 'd')} channel cannot carry the rate. Once the rate is "
             f"no longer tiny it falls with a slope that approaches the red dotted line (measured "
             f"{v(r.get('meas') or 0, '.2f')} decades per 10 dB). Small r: steep, hard to measure. "
             f"r near {v(m, 'd')}: almost flat.</p>")
        if p.nt == 2 and p.nr == 2:
            s += ("<p>Alamouti sits on the curve at r = 0 but cannot exceed one stream; V-BLAST "
                  "takes two streams but, with ZF, diversity one.</p>")
        return "<h3>Two uses of one resource</h3>" + s + keybox(
            "Rank adaptation in LTE and NR is the DMT in practice: many layers when the channel is "
            "strong, one beamformed layer at the cell edge.")


# =============================================================================== 8. massive MIMO
class MassiveMIMO(Experiment):
    title = "Massive MIMO: channel hardening"
    blurb = "With hundreds of antennas the fading averages away and users become orthogonal."
    book = "sec:ch19:massive"
    heavy = True
    controls = [
        LogSlider("M", "Base-station antennas M", 1, 256, 4, fmt=".0f"),
        Choice("rel", "Reliability", ["90 %", "99 %", "99.9 %"], "99 %"),
        Heading("Pilot contamination"),
        IntSlider("Lc", "Other cells reusing your pilot", 0, 10, 0),
        Slider("beta", "Their gain relative to yours", -20.0, 0.0, -10.0, step=0.5, unit="dB",
               enabled_if=lambda p: p.Lc > 0),
    ]
    plots = [
        Plot("hard", "Channel hardening: ‖h‖²/M", x="‖h‖²/M (dB)", y="probability density (1/dB)",
             xlim=(-25, 6), legend="tl"),
        Plot("fav", "Favourable propagation: |h₁ᴴh₂|/M", x="|h₁ᴴh₂| / M",
             y="probability density", xlim=(0, 1.6), legend="tr"),
        Plot("net", "Array gain minus fade margin, vs one antenna", x="antennas M",
             y="gain (dB)", logx=True, xlim=(1, 256), ylim=(0, 45), legend="tl"),
        Plot("pc", "Uplink rate with MR, 0 dB SNR", x="antennas M", y="b/s/Hz", logx=True,
             xlim=(1, 1024), legend="tl"),
    ]
    layout = [["hard", "fav"], ["net", "pc"]]
    readouts = [
        Readout("margin", "Fade margin", "dB", ".1f"),
        Readout("net", "Net gain over one antenna", "dB", ".1f"),
        Readout("ip", "Mean |h₁ᴴh₂|/M", "", ".3f"),
        Readout("gain", "Rate gain 64 → 1024 antennas", "b/s/Hz", ".2f"),
    ]
    challenges = [
        Challenge("Reproduce the book's worked example: at 99 % reliability, a net gain over one "
                  "antenna of at least 36 dB.",
                  lambda s: s.p.rel == "99 %" and s.r.net >= 36),
        Challenge("Harden the channel: a 99 % fade margin below 2 dB with no more than 48 antennas.",
                  lambda s: s.p.rel == "99 %" and s.r.margin < 2 and s.p.M <= 48),
        Challenge("Pilot contamination: with at least 6 interfering cells at −10 dB or louder, see "
                  "the rate gain from 64 to 1024 antennas fall below 1 b/s/Hz.",
                  lambda s: (s.p.Lc >= 6 and s.p.beta >= -10 and s.r.gain is not None
                             and s.r.gain < 1.0)),
    ]
    Ms = np.unique(np.round(np.logspace(0, np.log10(1024), 14)).astype(int))

    def setup(self):
        self.pcurve = None

    def update(self, p):
        rng = np.random.default_rng(13)
        q = {"90 %": 0.10, "99 %": 0.01, "99.9 %": 0.001}[p.rel]
        M = int(round(p.M))
        g = rng.gamma(M, 1.0, 40000) / M
        x = db(g)
        edges = np.linspace(-25, 6, 125)
        g1 = rng.exponential(size=40000)
        ph = self.plot("hard")
        h1 = np.histogram(db(g1), bins=edges, density=True)[0]
        hM = np.histogram(x, bins=edges, density=True)[0]
        ph.line("one", edges, h1, step=True, color=GRAY, width=1.4, name="M = 1")
        ph.line("M", edges, hM, step=True, color=NAVY, width=2.0, fill=0, fill_alpha=0.18,
                name=f"M = {M}")
        marg = -float(db(gamma_dist.ppf(q, M) / M))
        ph.vline("q", -marg, color=RED, style="--", label=f"{100 * (1 - q):g} % above", label_pos=0.85)
        ph.set_ylim(0, max(0.2, 1.15 * hM.max()))
        # favourable propagation: given h1, h1^H h2 ~ CN(0, ||h1||^2)
        n1 = rng.gamma(M, 1.0, 40000)
        ip = np.sqrt(n1) * np.abs(mk.cn(rng, 40000)) / M
        pf = self.plot("fav")
        e2 = np.linspace(0, 1.6, 81)
        ip1 = np.sqrt(rng.gamma(1, 1.0, 40000)) * np.abs(mk.cn(rng, 40000))
        pf.line("one", e2, np.histogram(ip1, bins=e2, density=True)[0], step=True, color=GRAY,
                width=1.4, name="M = 1")
        hip = np.histogram(ip, bins=e2, density=True)[0]
        pf.line("M", e2, hip, step=True, color=GREEN, width=2.0, fill=0, fill_alpha=0.18,
                name=f"M = {M}")
        pf.set_ylim(0, max(2.0, 1.1 * hip.max()))
        # net gain vs M
        Mg = np.unique(np.round(np.logspace(0, np.log10(256), 60)).astype(int))
        m1 = -float(db(gamma_dist.ppf(q, 1)))
        margins = -db(gamma_dist.ppf(q, Mg) / Mg)
        net = db(Mg) - margins + m1
        pn = self.plot("net")
        pn.line("arr", Mg, db(Mg), color=GRAY, style="--", width=1.4, name="array gain alone")
        pn.line("net", Mg, net, color=NAVY, width=2.4, name="array gain + saved margin")
        netM = float(db(M) - marg + m1)
        pn.scatter("now", [M], [netM], color=RED, size=12, symbol="d")
        pn.text("lab", M, netM, f" {netM:.1f} dB", color=RED, anchor=(0, 1.1), bold=True)
        self.readout(margin=marg, net=netM, ip=float(np.mean(ip)))
        self.pcurve = dict(M=[], o=[], c=[])
        self.readout(gain=None)

    def background(self, p):
        rng = np.random.default_rng(29)
        T = 40 if self.quick else 160
        beta = undb(p.beta)
        L = p.Lc
        snr, tau = 1.0, 10.0
        for M in self.Ms:
            out = []
            for contaminate in (False, True):
                if contaminate and L == 0:
                    out.append(np.nan)
                    continue
                h0 = mk.cn(rng, (T, M))
                hi = np.sqrt(beta) * mk.cn(rng, (T, max(L, 1), M)) if L > 0 else np.zeros((T, 1, M))
                est = h0 + (hi.sum(axis=1) if contaminate else 0) + mk.cn(rng, (T, M)) / np.sqrt(tau)
                sig = snr * np.abs(np.sum(est.conj() * h0, axis=1)) ** 2
                itf = snr * np.sum(np.abs(np.einsum("tlm,tm->tl", hi.conj(), est)) ** 2, axis=1) if L > 0 else 0
                nrm = np.sum(np.abs(est) ** 2, axis=1)
                out.append(float(np.mean(np.log2(1 + sig / (itf + nrm)))))
            yield dict(M=int(M), o=out[0], c=out[1])

    def progress(self, p, it):
        c = self.pcurve
        c["M"].append(it["M"])
        c["o"].append(it["o"])
        c["c"].append(it["c"])
        pp = self.plot("pc")
        pp.line("o", c["M"], c["o"], color=NAVY, width=2.2, name="orthogonal pilots")
        pp.scatter("os", c["M"], c["o"], color=NAVY, size=6)
        top = max(c["o"])
        if p.Lc > 0:
            pp.line("c", c["M"], c["c"], color=RED, width=2.2, name=f"pilot reused in {p.Lc} cells")
            pp.scatter("cs", c["M"], c["c"], color=RED, size=6)
            lim = np.log2(1 + 1 / (p.Lc * undb(p.beta) ** 2))
            pp.hline("lim", lim, color=RED, style=":", label=f"limit {lim:.1f} b/s/Hz", label_pos=0.05)
            top = max(top, min(lim, 25))
        pp.set_ylim(0, 1.15 * top + 0.5)
        if 1024 in c["M"]:
            arr = np.array(c["M"])
            i64 = int(np.argmin(np.abs(arr - 64)))
            key = "c" if p.Lc > 0 else "o"
            self.readout(gain=float(c[key][-1] - c[key][i64]))

    def story(self, p):
        r = self.r
        s = (f"<p>With MRC over {v(int(round(p.M)), 'd')} antennas the received SNR is the mean SNR times "
             f"‖h‖²/M times M. Top left: as M grows that random factor collapses onto 1 (0 dB): the "
             f"channel <b>hardens</b> and stops fading. To be reliable {p.rel} of the time you need "
             f"only {v(r.get('margin', 0), '.1f', 'dB')} of fade margin, against 20 dB for one "
             f"antenna at 99 %. Adding the {v(10 * np.log10(p.M), '.1f', 'dB')} of array gain: "
             f"{v(r.get('net', 0), '.1f', 'dB')} better than a single antenna.</p>"
             f"<p>Top right: two users' channels become nearly <b>orthogonal</b> (|h₁ᴴh₂|/M → 0), so "
             f"plain matched filtering separates them.</p>")
        if p.Lc > 0:
            s += (f"<p>But the base station learns h from uplink pilots, and {v(p.Lc, 'd')} users in "
                  f"other cells reuse your pilot. Their channels contaminate your estimate, and the "
                  f"beam aimed at you also aims at them: the rate saturates (bottom right, red) no "
                  f"matter how many antennas you add.</p>")
        else:
            s += ("<p>Bottom right (computed in the background): the uplink rate keeps growing with "
                  "M while pilots are orthogonal. Add other cells reusing the pilot to see the "
                  "ceiling that Marzetta called pilot contamination.</p>")
        return "<h3>The law of large numbers, on a mast</h3>" + s + keybox(
            "Hardening and favourable propagation make simple MR/ZF processing near-optimal; "
            "pilot contamination is the limit that remains.")


# =============================================================================== 9. MU-MIMO
PRECODERS = ["MR", "ZF", "RZF"]
PRE_COL = {"MR": ORANGE, "ZF": NAVY, "RZF": GREEN}


def fast_sum_rates(G, rho, kinds=PRECODERS):
    """Sum rates of MR/ZF/RZF from the Gram matrices G = H H^H (trials, K, K) only:
    W = H^H A with column norms sqrt(diag(A^H G A)), and H W = G A."""
    K = G.shape[-1]
    out = {}
    eye = np.eye(K)
    for kind in kinds:
        if kind == "MR":
            A = np.broadcast_to(eye, G.shape)
        elif kind == "ZF":
            A = np.linalg.inv(G + 1e-12 * eye)
        else:
            A = np.linalg.inv(G + K / rho * eye)
        HW = G @ A
        nrm = np.sqrt(np.real(np.diagonal(np.swapaxes(A.conj(), -1, -2) @ G @ A, axis1=1, axis2=2)))
        P = np.abs(HW / nrm[:, None, :]) ** 2 * rho / K
        sig = np.diagonal(P, axis1=1, axis2=2)
        itf = P.sum(axis=2) - sig
        out[kind] = float(np.mean(np.sum(np.log2(1 + sig / (itf + 1)), axis=1)))
    return out


class MUMIMO(Experiment):
    title = "MU-MIMO precoding: MR, ZF, RZF"
    blurb = "Serve many users at once: aim at each, null the others, or balance the two."
    book = "sec:ch19:mumimo"
    controls = [
        Choice("chan", "Channel", ["i.i.d. Rayleigh", "Line of sight"]),
        IntSlider("M", "Base-station antennas M", 4, 256, 64),
        IntSlider("K", "Users K", 1, 16, 8),
        Slider("snr", "SNR", -10.0, 30.0, 10.0, step=0.5, unit="dB"),
        Choice("show", "Beams shown", PRECODERS, "ZF"),
        Button("again", "New users"),
    ]
    plots = [
        Plot("rate", "Sum rate versus SNR", x="SNR (dB)", y="sum rate (b/s/Hz)", xlim=(-10, 30),
             legend="tl"),
        PolarPlot("beam", "User 1's beam (line of sight)", floor_db=-40, legend=None),
        BarPlot("users", "Each user: signal and interference", y="power (dB)"),
    ]
    layout = [["rate", "beam"], ["users", "beam"]]
    col_stretch = [3, 2]
    row_stretch = [3, 2]
    readouts = [
        Readout("mr", "MR sum rate", "b/s/Hz", ".1f"),
        Readout("zf", "ZF sum rate", "b/s/Hz", ".1f"),
        Readout("rzf", "RZF sum rate", "b/s/Hz", ".1f"),
        Readout("best", "Best here", "", None),
    ]
    challenges = [
        Challenge("Find a setting where plain MR beats ZF.",
                  lambda s: s.r.mr > s.r.zf + 0.05,
                  hint="ZF spends power on nulling: costly when M ≈ K or the SNR is low."),
        Challenge("Line of sight at 20 dB: make ZF deliver at least 1.8 times MR's sum rate.",
                  lambda s: (s.p.chan.startswith("Line") and s.p.snr >= 20
                             and s.r.zf >= 1.8 * s.r.mr),
                  hint="MR's sidelobes hurt most when the users crowd the array: fewer antennas "
                       "per user."),
        Challenge("i.i.d.: serve 16 users at 10 dB with a ZF sum rate of at least 50 b/s/Hz using "
                  "no more than 32 antennas.",
                  lambda s: (s.p.chan.startswith("i.i.d") and s.p.K == 16 and s.p.snr <= 10
                             and s.p.M <= 32 and s.r.zf >= 50)),
    ]
    trials = 60

    def setup(self):
        self.seed = 21

    def on_again(self, p):
        self.seed += 1

    def channels(self, p, n, rng):
        if p.chan.startswith("i.i.d"):
            return mk.cn(rng, (n, p.K, p.M)), None
        th = rng.uniform(-np.pi / 3, np.pi / 3, (n, p.K))
        ph = rng.uniform(0, 2 * np.pi, (n, p.K))
        m = np.arange(p.M)
        A = np.exp(-1j * np.pi * m[None, None, :] * np.sin(th)[..., None])
        return np.conj(A) * np.exp(1j * ph)[..., None], th

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        H, th = self.channels(p, self.trials, rng)
        G = H @ np.swapaxes(H.conj(), -1, -2)
        sg = np.arange(-10, 30.1, 2.5)
        curves = {k: [] for k in PRECODERS}
        for s_ in sg:
            res = fast_sum_rates(G, undb(s_))
            for k in PRECODERS:
                curves[k].append(res[k])
        here = fast_sum_rates(G, undb(p.snr))
        pr = self.plot("rate")
        for k in PRECODERS:
            pr.line(k, sg, curves[k], color=PRE_COL[k], width=2.6 if k == p.show else 1.6, name=k)
        pr.vline("now", p.snr, color=PURPLE, style=":")
        pr.set_ylim(0, max(5.0, 1.1 * max(max(c) for c in curves.values())))
        # one realisation: beams and per-user powers
        h1 = H[0]
        rho = undb(p.snr)
        W = mk.mu_precoder(h1, p.show, rho)
        sinr, sig, itf = mk.mu_sinr(h1, W, rho)
        pb = self.plot("users")
        k = np.arange(p.K)
        pb.bars("s", k - 0.2, db(sig), width=0.38, color=NAVY, base=-40)
        pb.bars("i", k + 0.2, db(np.maximum(itf, 1e-4)), width=0.38, color=RED, base=-40)
        pb.hline("n", 0, color=GRAY, style="--", label="noise", label_pos=0.01)
        pb.set_xticks([(i, str(i + 1)) for i in k] if p.K <= 16 else None)
        pb.set_xlim(-0.7, p.K - 0.3)
        top = float(max(db(sig).max(), 5))
        pb.set_ylim(-40, top + 12)
        pb.text("leg", -0.6, top + 10, "navy: wanted signal   red: interference", size=8.5,
                anchor=(0, 0))
        pp = self.plot("beam")
        if th is not None:
            ang = np.linspace(-np.pi / 2, np.pi / 2, 721)
            a = np.exp(-1j * np.pi * np.arange(p.M)[:, None] * np.sin(ang)[None, :])
            resp = np.abs(a.conj().T @ W[:, 0]) ** 2
            gdb = db(resp / resp.max())
            pp.pattern("b", ang, gdb, color=PRE_COL[p.show], width=2.2, fill=True)
            th0 = th[0]
            others = th0[1:]
            pp.scatter("u1", [1.05 * np.sin(th0[0])], [1.05 * np.cos(th0[0])], color=RED, size=13,
                       symbol="t")
            pp.scatter("uo", 1.05 * np.sin(others), 1.05 * np.cos(others), color=NAVY, size=10,
                       symbol="t")
            pp.text("note", -1.1, -0.95, "red ▲ user 1, navy ▲ the others", size=8.5,
                    anchor=(0, 0))
            pp.set_title(f"User 1's {p.show} beam (line of sight)")
        else:
            pp.text("note", -0.95, 0.05, "Rich scattering: there is no single\ndirection to draw."
                    "\nChoose 'Line of sight' to see beams.", size=9.5, anchor=(0, 0.5))
            pp.set_title("User 1's beam (line of sight only)")
        best = max(here, key=here.get)
        self.readout(mr=here["MR"], zf=here["ZF"], rzf=here["RZF"], best=best)

    def story(self, p):
        r = self.r
        s = (f"<p>{v(p.M, 'd')} antennas, {v(p.K, 'd')} single-antenna users, all served in the same "
             f"time and frequency. <b>MR</b> points each beam straight at its user and ignores the "
             f"rest; <b>ZF</b> steers each beam into the null space of everyone else; <b>RZF</b> "
             f"blends the two according to the SNR. At {v(p.snr, '.0f', 'dB')}: MR "
             f"{v(r.get('mr', 0), '.1f')}, ZF {v(r.get('zf', 0), '.1f')}, RZF "
             f"{v(r.get('rzf', 0), '.1f')} b/s/Hz.</p>")
        if p.M >= 4 * p.K:
            s += ("<p>With many more antennas than users, nulling is cheap: ZF and RZF grow by about "
                  f"{v(p.K, 'd')} b/s/Hz per 3 dB ({p.K} interference-free streams) while MR "
                  "saturates on interference.</p>")
        else:
            s += ("<p>With M close to K, inverting HHᴴ is ill-conditioned and ZF wastes power; RZF's "
                  "regularisation rescues it.</p>")
        if p.chan.startswith("Line"):
            s += ("<p>Right: user 1's beam. With ZF, look for the deep nulls sitting exactly on "
                  "the other users (navy triangles); MR's beam has sidelobes wherever they "
                  "happen to fall.</p>")
        return "<h3>Many users, one array</h3>" + s + keybox(
            "Bottom left: navy bars are each user's wanted power, red bars the interference it "
            "receives. ZF trades a little signal for (almost) no red.")


# =============================================================================== the lab
LAB = st.Lab(10, "MIMO: Diversity, Capacity, Detection and Massive MIMO", chapter=19,
             chapter_title="MIMO and Antenna Arrays",
             experiments=[FadingLive, DiversityBER, Alamouti, EigenWaterfill, ErgodicOutage,
                          Detectors, DMT, MassiveMIMO, MUMIMO])

if __name__ == "__main__":
    st.run(LAB)
