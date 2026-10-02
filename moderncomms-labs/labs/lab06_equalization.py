"""Lab 06 · Equalization: ZF, MMSE, Adaptive, Blind, DFE, MLSE and SerDes   (Chapter 12)

Run it:      python labs/lab06_equalization.py
Self-test:   python labs/lab06_equalization.py --selftest

When the channel's memory approaches the symbol period, every received sample is a blend of
several symbols. An equalizer undoes the blend, and the whole chapter is one trade-off seen
from many angles: inverting the channel removes ISI but amplifies noise wherever the channel
is weak. Nine experiments: the ISI channel, zero forcing versus MMSE, length and delay, LMS
learning live, LMS versus RLS, blind CMA, decision feedback and its error bursts, the Viterbi
equalizer's trellis, and the CTLE/FFE/DFE chain of a 100 Gb/s SerDes.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.linalg import toeplitz

import commlib as cl
from commlib import eqadv
from commlib import linecodes as lc
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, EyePlot, ConstellationPlot, BERPlot, BarPlot, Readout,
                    Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

QPSK = cl.get_constellation("qpsk")
QAM16 = cl.get_constellation("16qam")
BPSK = cl.get_constellation("bpsk")
PROAKIS = {"B": np.array([0.407, 0.815, 0.407]),
           "C": np.array([0.227, 0.460, 0.688, 0.460, 0.227])}
CHANNELS = ["3-tap [1, a·e^jθ, 0.2]", "Two-path [1, a]", "Proakis B", "Proakis C"]


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(np.abs(x), 1e-30))


def make_channel(kind, a=0.5, theta_deg=20.0):
    """Unit-energy symbol-spaced channels used throughout the chapter."""
    if kind.startswith("3-tap"):
        h = np.array([1, a * np.exp(1j * np.deg2rad(theta_deg)), 0.2])
    elif kind.startswith("Two-path"):
        h = np.array([1.0, a], complex)
    elif kind == "Proakis B":
        h = PROAKIS["B"].astype(complex)
    else:
        h = PROAKIS["C"].astype(complex)
    return h / np.linalg.norm(h)


def freqresp_db(h, n=512):
    f = np.linspace(-0.5, 0.5, n, endpoint=False)
    return f, db(np.abs(np.fft.fftshift(np.fft.fft(h, n))) ** 2)


def snr_limits(h, snr_db, n=8192):
    """Unbiased output SNRs (linear) of infinite-length ZF, MMSE, ZF-DFE, MMSE-DFE and the MFB
    for unit-energy symbols (the formulas of Chapter 12, as in book/figscripts/ch12_figs.py)."""
    H2 = np.abs(np.fft.fft(h, n)) ** 2
    s = 10 ** (snr_db / 10)
    n0 = 1 / s
    zf = 1 / np.mean(n0 / np.maximum(H2, 1e-30))
    mmse = 1 / np.mean(n0 / (H2 + n0)) - 1
    zfdfe = s * np.exp(np.mean(np.log(np.maximum(H2, 1e-30))))
    dfe = np.exp(np.mean(np.log((H2 + n0) / n0))) - 1
    mfb = s * np.sum(np.abs(h) ** 2)
    return zf, mmse, zfdfe, dfe, mfb


def unbiased_snr(z, s):
    """Measured unbiased SNR (dB) of equalizer output z against the symbols s."""
    g = np.vdot(s, z) / np.vdot(s, s)
    e = z - g * s
    return float(db(np.abs(g) ** 2 / max(np.mean(np.abs(e) ** 2), 1e-30)))


def csign(x):
    return np.sign(x.real) + 1j * np.sign(x.imag)


def transmit(con, h, n, esn0, rng, real_noise=False):
    s = con.modulate(cl.random_bits(con.k * n, rng))
    x = np.convolve(s, h)[:n]
    n0 = 10 ** (-esn0 / 10)
    if real_noise:
        w = np.sqrt(n0 / 2) * rng.standard_normal(n)
    else:
        w = np.sqrt(n0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return s, x + w, n0


# =============================================================================== 1. the ISI channel
class ISIChannel(Experiment):
    title = "Symbols that collide"
    blurb = "Echoes one symbol late: the received constellation smears into clouds."
    book = "sec:ch12:intro"
    controls = [
        Choice("ch", "Channel", CHANNELS, style="menu"),
        Slider("a", "Echo amplitude a", 0.0, 1.0, 0.5, step=0.01,
               enabled_if=lambda p: p.ch.startswith(("3-tap", "Two"))),
        Slider("th", "Echo phase θ", 0, 180, 20, step=1, unit="°",
               enabled_if=lambda p: p.ch.startswith("3-tap")),
        Slider("esn0", "Es/N0", 0, 40, 25, step=0.5, unit="dB"),
        Button("again", "New symbols"),
    ]
    plots = [
        Plot("taps", "Channel taps |h_k|", x="tap k (symbol periods)", y="|h_k|",
             xlim=(-0.6, 4.6), ylim=(0, 1.05), legend=None),
        Plot("H", "Channel frequency response |H(f)|²", x="frequency (multiples of Rs)",
             y="gain (dB)", xlim=(-0.5, 0.5), ylim=(-40, 10), legend=None),
        ConstellationPlot("con", "Received QPSK samples (no equalizer)", lim=2.0),
    ]
    layout = [["con", "taps"], ["con", "H"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("pd", "Peak distortion", "", ".2f", good=lambda x: x < 1,
                help="Σ|h_k| / |h_main| over the other taps: below 1 the eye is open"),
        Readout("null", "Deepest spectral notch", "dB", ".1f"),
        Readout("ser", "Symbol errors, no equalizer", "", "sci"),
        Readout("mfb", "Matched-filter bound", "dB", ".1f", help="SNR of one isolated symbol: Es/N0·‖h‖²"),
    ]
    challenges = [
        Challenge("Close the eye: peak distortion above 1 (the slicer alone cannot cope).",
                  lambda s: s.r.pd > 1),
        Challenge("Carve a spectral notch deeper than 30 dB.",
                  lambda s: s.r.null < -30),
        Challenge("Find a channel whose eye is still open (peak distortion < 1) but whose notch is "
                  "deeper than 20 dB.",
                  lambda s: s.r.pd < 1 and s.r.null < -20,
                  hint="Two paths: an echo slightly weaker than the direct path."),
    ]

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        h = make_channel(p.ch, p.a, p.th)
        rng = np.random.default_rng(self.seed)
        s, y, n0 = transmit(QPSK, h, 3000, p.esn0, rng)
        main = int(np.argmax(np.abs(h)))
        z = y[main:] / h[main]
        ref = s[:len(z)]
        ser = float(np.mean(QPSK.decide(z[10:]) != ref[10:]))
        pd = float((np.sum(np.abs(h)) - np.abs(h[main])) / np.abs(h[main]))
        f, H = freqresp_db(h, 1024)
        pt = self.plot("taps")
        k = np.arange(len(h))
        cols = [NAVY if i == main else RED for i in k]
        pt.bars("b", k, np.abs(h), width=0.35, colors=cols)
        pt.set_xticks([(i, "main" if i == main else f"{i - main:+d}") for i in k])
        ph = self.plot("H")
        ph.line("H", f, np.clip(H, -40, 10), color=NAVY, width=2.2, fill=-40, fill_alpha=0.12)
        ph.hline("z", 0, color=GRAY, style=":")
        pc = self.plot("con")
        pc.points("y", z[:2000], color=NAVY, size=3, alpha=0.4)
        pc.ideal("pts", QPSK.points)
        self.readout(pd=pd, null=float(max(H.min(), -60)), ser=ser, mfb=p.esn0)

    def story(self, p):
        pd = self.r.get("pd", 0)
        s = ("<p>The receiver samples once per symbol, but each sample also carries the tails of "
             "its neighbours: the red taps leak earlier and later symbols into the main one (navy). "
             "Each QPSK point therefore splits into a cloud of sub-points, one per combination of "
             "neighbours: <b>inter-symbol interference</b>.</p>")
        if pd < 1:
            s += (f"<p>Peak distortion {v(pd)} &lt; 1: even the worst combination of neighbours cannot "
                  "push a sample across the decision line, so with little noise a plain slicer works. "
                  "The eye is open.</p>")
        else:
            s += (f"<p>Peak distortion {v(pd)} ≥ 1: some neighbour patterns push samples into the "
                  "wrong quadrant. " + bad("No amount of SNR fixes this") + "; you need an equalizer.</p>")
        s += ("<p>In frequency the same echo is a ripple: the deeper the notch (bottom right), the "
              "harder any equalizer must work to undo it, and the more noise it lifts with it.</p>")
        return "<h3>When symbols collide</h3>" + s + keybox(
            "Time view: ISI. Frequency view: a non-flat channel. Equalization fights one by "
            "flattening the other.")


# =============================================================================== 2. ZF vs MMSE
class ZFvsMMSE(Experiment):
    title = "Zero forcing vs MMSE"
    blurb = "Invert the channel and the noise explodes at the notch. MMSE knows when to stop."
    book = "sec:ch12:mmse"
    controls = [
        Choice("ch", "Channel", CHANNELS[:3], "Two-path [1, a]", style="menu"),
        Slider("a", "Echo amplitude a", 0.0, 1.0, 0.9, step=0.01,
               enabled_if=lambda p: p.ch != "Proakis B"),
        Slider("esn0", "Es/N0", 5, 40, 20, step=0.5, unit="dB"),
        IntSlider("L", "Equalizer taps L", 3, 61, 31, step=2),
        Choice("mod", "Modulation", ["QPSK", "16-QAM"]),
    ]
    plots = [
        Plot("f", "Channel, equalizers and what they do to the noise", x="frequency (multiples of Rs)",
             y="gain (dB)", xlim=(-0.5, 0.5), ylim=(-30, 30), legend="tl", legend_cols=3),
        ConstellationPlot("zf", "After zero forcing", lim=1.8),
        ConstellationPlot("mm", "After MMSE", lim=1.8),
        BarPlot("snr", "Output SNR, L → ∞ (book formulas)", y="SNR (dB)"),
    ]
    layout = [["f", "f", "snr"], ["zf", "mm", "snr"]]
    row_stretch = [2, 3]
    col_stretch = [2, 2, 2]
    readouts = [
        Readout("zf", "ZF output SNR", "dB", ".1f"),
        Readout("mmse", "MMSE output SNR", "dB", ".1f"),
        Readout("enh", "ZF noise enhancement", "dB", ".1f", help="10·log₁₀ Σ|w_k|²"),
        Readout("ser", "SER after MMSE", "", "sci"),
    ]
    challenges = [
        Challenge("Make zero forcing lose more than 9.5 dB to the matched-filter bound.",
                  lambda s: s.p.esn0 - s.r.zf > 9.5),
        Challenge("High SNR makes them agree: ZF and MMSE within 0.5 dB on a two-path channel "
                  "with a ≥ 0.5.",
                  lambda s: s.p.ch.startswith("Two") and s.p.a >= 0.5 and abs(s.r.mmse - s.r.zf) < 0.5),
        Challenge("Find where MMSE matters most: on the a = 0.9 two-path channel, make MMSE beat ZF "
                  "by 4 dB or more.",
                  lambda s: s.p.ch.startswith("Two") and abs(s.p.a - 0.9) < 0.006
                  and s.r.mmse - s.r.zf >= 4,
                  hint="The N₀I brake matters most when N₀ is large."),
    ]

    def update(self, p):
        h = make_channel(p.ch, p.a, 20.0)
        con = QPSK if p.mod == "QPSK" else QAM16
        rng = np.random.default_rng(3)
        s, y, n0 = transmit(con, h, 6000, p.esn0, rng)
        wz, dz = cl.zf_fir(h, p.L)
        wm, dm = cl.mmse_fir(h, p.L, n0)
        zz, zm = cl.apply_fir(y, wz, dz), cl.apply_fir(y, wm, dm)
        sl = slice(100, -100)
        snr_zf, snr_mm = unbiased_snr(zz[sl], s[sl]), unbiased_snr(zm[sl], s[sl])
        g = np.vdot(s[sl], zm[sl]) / np.vdot(s[sl], s[sl])
        ser = float(np.mean(con.decide(zm[sl] / g) != s[sl]))
        pf = self.plot("f")
        for key, w_, col, name, wd in [("h", h, GRAY, "channel", 2.4), ("wz", wz, RED, "ZF", 1.8),
                                       ("wm", wm, GREEN, "MMSE", 1.8)]:
            f, H = freqresp_db(w_)
            pf.line(key, f, np.clip(H, -30, 30), color=col, width=wd, name=name)
        pf.hline("z", 0, color=GRAY, style=":")
        self.plot("zf").points("z", zz[sl][:2500], color=RED, size=3, alpha=0.35)
        self.plot("zf").ideal("p", con.points)
        self.plot("mm").points("z", zm[sl][:2500] / g, color=GREEN, size=3, alpha=0.35)
        self.plot("mm").ideal("p", con.points)
        lim = 10 * np.log10(snr_limits(h, p.esn0))
        pb = self.plot("snr")
        vals = [lim[4], lim[3], lim[2], lim[1], lim[0]]
        names = ["MFB", "MMSE-DFE", "ZF-DFE", "MMSE", "ZF"]
        cols = [GRAY, TEAL, BLUE, GREEN, RED]
        vals = [max(vv, -5) for vv in vals]
        pb.bars("b", np.arange(5), vals, width=0.6, colors=cols, base=-5)
        for i, vv in enumerate(vals):
            pb.text(f"t{i}", i, vv, f"{vv:.1f}", anchor=(0.5, 1), size=8.5, bold=True)
        pb.set_xticks([(i, n) for i, n in enumerate(names)])
        pb.set_xlim(-0.7, 4.7)
        pb.set_ylim(-5, p.esn0 + 6)
        self.readout(zf=snr_zf, mmse=snr_mm, enh=float(db(np.sum(np.abs(wz) ** 2))),
                     ser=ser if ser > 0 else "< 2×10⁻⁴")

    def story(self, p):
        zf, mm = self.r.get("zf", 0), self.r.get("mmse", 0)
        s = ("<p>The <b>zero-forcing</b> equalizer (red) is the channel's mirror image: wherever the "
             "channel dips, it boosts. That removes the ISI completely, but the noise at the notch "
             f"is boosted just as faithfully: {v(self.r.get('enh', 0), '.1f', 'dB')} of noise "
             "enhancement. The <b>MMSE</b> equalizer (green) minimises signal error plus noise, so it "
             "caps its gain where boosting would cost more noise than it removes ISI.</p>")
        s += (f"<p>Measured: ZF {v(zf, '.1f', 'dB')}, MMSE {v(mm, '.1f', 'dB')}, against a "
              f"matched-filter bound of {v(p.esn0, '.1f', 'dB')}. The bars on the right show the "
              "chapter's infinite-length formulas, including the decision-feedback equalizers of "
              "experiment 7, which do much better on deep notches.</p>")
        if abs(mm - zf) < 0.5:
            s += "<p>At high SNR, noise matters little and MMSE converges to ZF.</p>"
        return "<h3>Invert, but not too hard</h3>" + s + keybox(
            "w_MMSE = (HᴴH + N₀I)⁻¹Hᴴe_Δ: the N₀I is the brake. Never zero-force a channel you have "
            "not looked at.")


# =============================================================================== 3. length and delay
def _reflect(h, inside):
    """Same |H(f)|, all zeros moved inside (minimum phase) or outside (maximum phase)."""
    z = np.roots(h)
    z = np.where((np.abs(z) > 1) == inside, 1 / np.conj(z), z)
    g = np.real(np.poly(z))
    return g / np.linalg.norm(g)


class LengthDelay(Experiment):
    title = "Equalizer length and delay"
    blurb = "An FIR approximation of an inverse needs enough taps, and the right sampling delay."
    book = "sec:ch12:fir"
    controls = [
        Choice("ch", "Channel (same |H(f)|)", ["Mixed phase", "Minimum phase", "Maximum phase"]),
        IntSlider("L", "Equalizer length L", 1, 41, 11),
        IntSlider("d", "Decision delay Δ", 0, 44, 2, unit="symbols"),
        Slider("snr", "SNR", 10, 40, 25, step=0.5, unit="dB"),
    ]
    plots = [
        Plot("dly", "MMSE vs decision delay (this L)", x="decision delay Δ (symbols)", y="MSE (dB)",
             xlim=(-0.5, 45), ylim=(-35, 3), legend="tl"),
        Plot("q", "Channel ⊛ equalizer: should be a single spike", x="combined tap", y="amplitude",
             ylim=(-0.4, 1.15), legend=None),
        Plot("len", "Best MSE vs length", x="equalizer length L", y="MSE (dB)", xlim=(0, 42),
             ylim=(-35, 3), legend="tr"),
    ]
    layout = [["dly", "dly"], ["q", "len"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("mse", "MSE at your Δ", "dB", ".1f"),
        Readout("best", "Best delay for this L", "", "int"),
        Readout("loss", "Loss vs best delay", "dB", ".1f", good=lambda x: x < 0.2),
        Readout("inf", "Loss vs infinite length", "dB", ".1f", good=lambda x: x < 1),
    ]
    challenges = [
        Challenge("Find the best delay for L = 11 (within 0.1 dB).",
                  lambda s: s.p.L == 11 and s.r.loss < 0.1),
        Challenge("Use the shortest equalizer within 1 dB of infinite length (L ≤ 11, SNR 25 dB, "
                  "best delay).",
                  lambda s: s.p.L <= 11 and abs(s.p.snr - 25) < 0.3 and s.r.inf < 1 and s.r.loss < 0.1
                  and s.p.ch == "Mixed phase"),
        Challenge("Maximum-phase channel, L = 21: show that Δ = 0 is a disaster (10 dB or more "
                  "worse than the best delay).",
                  lambda s: s.p.ch == "Maximum phase" and s.p.L == 21 and s.p.d == 0 and s.r.loss >= 10,
                  hint="Its energy arrives late: the inverse is anti-causal and needs a delay."),
    ]
    MIXED = np.array([0.3, -0.5, 1.0, 0.65, 0.35])

    def channel(self, kind):
        h = self.MIXED / np.linalg.norm(self.MIXED)
        if kind == "Minimum phase":
            return _reflect(h, True)
        if kind == "Maximum phase":
            return _reflect(h, False)
        return h

    @staticmethod
    def mse_all(h, L, n0):
        H = cl.conv_matrix(h, L)
        R = H.conj().T @ H + n0 * np.eye(L)
        Ri = np.linalg.inv(R)
        P = H.conj().T                                  # columns: cross-correlation per delay
        m = 1 - np.real(np.einsum("id,ij,jd->d", P.conj(), Ri, P))
        return np.maximum(m, 1e-6), Ri, P

    def update(self, p):
        h = self.channel(p.ch)
        n0 = 10 ** (-p.snr / 10)
        mse, Ri, P = self.mse_all(h, p.L, n0)
        nd = len(mse)
        d = min(p.d, nd - 1)
        best = int(np.argmin(mse))
        w = Ri @ P[:, d]
        q = np.convolve(h, w)
        pdl = self.plot("dly")
        pdl.line("m", np.arange(nd), db(mse), color=NAVY, width=2, name=f"L = {p.L}")
        pdl.scatter("pts", np.arange(nd), db(mse), color=NAVY, size=5)
        pdl.scatter("now", [d], [db(mse[d])], color=RED, size=13, symbol="d", name="your Δ")
        inf_mse = 1 / (snr_limits(h, p.snr)[1] + 1)
        pdl.hline("inf", db(inf_mse), color=GREEN, style="--", label="infinite-length MMSE",
                  label_pos=0.75)
        if p.d >= nd:
            pdl.text("warn", 1, -32, f"Δ beyond L + 4 = {nd - 1}: clipped", color=RED, size=8.5)
        pq = self.plot("q")
        pq.stems("q", np.arange(len(q)), q.real, color=NAVY, size=5)
        pq.scatter("tgt", [d], [1.0], color=GREEN, size=11, symbol="+")
        pq.set_xlim(-1, len(q))
        Ls = np.arange(1, 42, 2)
        bestm = [db(self.mse_all(h, Lx, n0)[0].min()) for Lx in Ls]
        pl = self.plot("len")
        pl.line("b", Ls, bestm, color=NAVY, width=2, name="best delay")
        pl.hline("inf", db(inf_mse), color=GREEN, style="--")
        pl.scatter("now", [p.L], [db(mse.min())], color=RED, size=12, symbol="d")
        self.readout(mse=float(db(mse[d])), best=best, loss=float(db(mse[d]) - db(mse[best])),
                     inf=float(db(mse.min()) - db(inf_mse)))

    def story(self, p):
        s = ("<p>An FIR equalizer approximates the channel's inverse, which is usually infinitely "
             "long. With L taps it can only make the combined response (bottom left) <i>nearly</i> a "
             "single spike, at the position Δ you ask for (green cross).</p>")
        if p.ch == "Minimum phase":
            s += ("<p>A <b>minimum-phase</b> channel has its energy up front; its inverse is causal, so "
                  "small delays work.</p>")
        elif p.ch == "Maximum phase":
            s += ("<p>A <b>maximum-phase</b> channel has the same |H(f)| but its energy arrives late. "
                  "Its stable inverse is anti-causal: the equalizer must wait (a large Δ) to collect "
                  "the energy of the symbol it estimates.</p>")
        else:
            s += ("<p>The book's <b>mixed-phase</b> channel needs a delay in the middle: too early "
                  "and the equalizer has not yet seen the symbol's energy; too late and the energy has "
                  "slid out of its window (the wall on the right of the top plot).</p>")
        s += (f"<p>Your Δ = {v(min(p.d, p.L + 3), 'd')} costs {v(self.r.get('loss', 0), '.1f', 'dB')} "
              f"against the best delay ({v(self.r.get('best', 0), 'd')}); the best L = {v(p.L, 'd')} "
              f"equalizer is {v(self.r.get('inf', 0), '.1f', 'dB')} from infinite length.</p>")
        return "<h3>Taps and timing</h3>" + s + keybox(
            "Practical rule: L of a few times the channel memory, Δ near the middle, then check the "
            "MSE-vs-Δ curve: adaptive equalizers inherit whatever delay you initialise them with.")


# =============================================================================== 4. LMS live
class _Stream:
    """A long symbol stream through a channel, with a regressor view for adaptive filters."""

    def __init__(self, con, h, n, esn0, L, rng, phase=0.0):
        self.s, y, self.n0 = transmit(con, h, n, esn0, rng)
        self.y = y * np.exp(1j * phase)
        self.L = L
        self.delay = L // 2
        self.ypad = np.concatenate([np.zeros(L - 1 - self.delay), self.y, np.zeros(self.delay)])

    def u(self, n):
        return self.ypad[n:n + self.L][::-1]


class LMSLive(Experiment):
    title = "LMS learning, live"
    blurb = "Watch the equalizer learn: the error falls, the clouds shrink, the taps take shape."
    book = "sec:ch12:lms"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Choice("algo", "Algorithm", ["LMS", "Normalized LMS", "Sign-sign LMS"]),
        LogSlider("mu", "Step size μ", 1e-4, 0.3, 0.01),
        IntSlider("L", "Taps L", 3, 31, 15, step=2),
        Heading("Signal and channel"),
        Choice("mod", "Modulation", ["QPSK", "16-QAM"], "16-QAM"),
        Slider("a", "Echo amplitude a", 0.0, 0.95, 0.8, step=0.01),
        Slider("esn0", "Es/N0", 10, 40, 25, step=0.5, unit="dB"),
        IntSlider("train", "Training symbols", 0, 3000, 500, step=50,
                  help="Known symbols at the start; afterwards the slicer's decisions are used"),
        Button("restart", "Restart learning"),
    ]
    plots = [
        Plot("mse", "Learning curve (50-symbol average)", x="symbol", y="squared error (dB)",
             xlim=(0, 6000), ylim=(-35, 10), legend="tr"),
        ConstellationPlot("con", "Equalizer output (last 400 symbols)", lim=1.6),
        Plot("taps", "Tap magnitudes |w_k|", x="tap k", y="|w_k|", legend="tr"),
    ]
    layout = [["mse", "mse"], ["con", "taps"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("n", "Symbols processed", "", "int"),
        Readout("mse", "Current MSE (last 500)", "dB", ".1f"),
        Readout("excess", "Excess over MMSE", "dB", ".2f", good=lambda x: x < 1,
                help="Misadjustment: how far the jittering taps sit above the Wiener optimum"),
        Readout("conv", "Converged at symbol", "", "int", help="First time within 1 dB of the floor"),
    ]
    challenges = [
        Challenge("Converge to within 1 dB of the MMSE floor in fewer than 400 symbols.",
                  lambda s: isinstance(s.r.conv, (int, float)) and s.r.conv < 400),
        Challenge("Make LMS diverge: MSE above +5 dB after 1000 symbols.",
                  lambda s: s.r.n >= 1000 and s.r.mse > 5,
                  hint="Stability needs μ < 2/(L·P_y) roughly; much less in practice."),
        Challenge("Precision: finish 6000 16-QAM symbols less than 0.8 dB above the MMSE floor.",
                  lambda s: s.p.mod == "16-QAM" and s.r.n >= 6000 and s.r.excess < 0.8,
                  hint="Misadjustment grows with μ and with the number of taps; wrong decisions add more."),
        Challenge("No training at all (decision-directed from the start) and still converge on "
                  "QPSK with a ≥ 0.6.",
                  lambda s: s.p.train == 0 and s.p.mod == "QPSK" and s.p.a >= 0.6
                  and s.r.n >= 3000 and s.r.excess < 2),
    ]
    N = 6000
    chunk = 120

    def setup(self):
        self.seed = 11

    def on_restart(self, p):
        self.seed += 1

    def update(self, p):
        con = QPSK if p.mod == "QPSK" else QAM16
        h = make_channel("Two-path [1, a]", p.a)
        rng = np.random.default_rng(self.seed)
        self.st = _Stream(con, h, self.N, p.esn0, p.L, rng)
        self.con = con
        wm, dm = cl.mmse_fir(h, p.L, self.st.n0, delay=self.st.delay)
        zm = cl.apply_fir(self.st.y, wm, dm)
        self.floor = float(db(np.mean(np.abs(zm[200:-200] - self.st.s[200:-200]) ** 2)))
        self.wopt = wm
        self.w = np.zeros(p.L, complex)
        self.w[self.st.delay] = 1
        self.n = 0
        self.z = np.zeros(self.N, complex)
        self.e2 = np.zeros(self.N)
        self.conv = None
        self.run(p, 200)

    def tick(self, p):
        if self.n < self.N:
            self.run(p, self.chunk)
        else:
            self.draw(p)

    def run(self, p, k):
        st_, w, pts = self.st, self.w, self.con.points
        mu = p.mu
        for n in range(self.n, min(self.n + k, self.N)):
            u = st_.u(n)
            z = np.dot(w.conj(), u)
            if not np.isfinite(z) or abs(z) > 1e6:
                z = 1e6
                self.z[n], self.e2[n] = z, 1e12
                continue
            d = st_.s[n] if n < p.train else pts[np.argmin(np.abs(z - pts))]
            e = d - z
            if p.algo == "LMS":
                w += mu * u * np.conj(e)
            elif p.algo == "Normalized LMS":
                w += mu * u * np.conj(e) / (1e-3 + np.real(np.vdot(u, u)))
            else:
                w += mu * 0.05 * csign(u) * np.conj(csign(e))
            self.z[n], self.e2[n] = z, abs(e) ** 2
        self.n = min(self.n + k, self.N)
        self.draw(p)

    def draw(self, p):
        n = self.n
        e = self.e2[:n]
        k = 50
        sm = db(np.convolve(np.minimum(e, 1e6), np.ones(k) / k, mode="valid")) if n > k else db(e)
        x = np.arange(len(sm)) + (k if n > k else 0)
        pm = self.plot("mse")
        pm.line("c", x, np.clip(sm, -35, 10), color=NAVY, width=1.6, name=p.algo)
        pm.hline("fl", self.floor, color=GREEN, style="--", label=f"MMSE floor {self.floor:.1f} dB",
                 label_pos=0.6)
        if p.train > 0:
            pm.vline("tr", p.train, color=ORANGE, style=":", label="training ends", label_pos=0.92)
        if self.conv is None and len(sm) and np.any(sm <= self.floor + 1):
            self.conv = int(x[np.argmax(sm <= self.floor + 1)])
        lo = max(0, n - 400)
        zz = self.z[lo:n]
        pc = self.plot("con")
        pc.points("z", zz[np.abs(zz) < 10], color=NAVY if n >= p.train else ORANGE, size=3, alpha=0.5)
        pc.ideal("p", self.con.points)
        pt = self.plot("taps")
        kk = np.arange(len(self.w)) - self.st.delay
        pt.stems("w", kk, np.minimum(np.abs(self.w), 5), color=NAVY, size=6, name="LMS taps now")
        pt.scatter("o", kk, np.abs(self.wopt), color=GREEN, size=11, symbol="x",
                   name="Wiener optimum")
        pt.set_ylim(0, 1.15 * max(1.0, np.abs(self.wopt).max(), min(np.abs(self.w).max(), 5)))
        cur = float(db(np.mean(np.minimum(e[-500:], 1e6)))) if n >= 500 else (float(sm[-1]) if len(sm) else 0.0)
        self.readout(n=n, mse=cur, excess=cur - self.floor,
                     conv=self.conv if self.conv is not None else "—")

    def story(self, p):
        n = self.r.get("n", 0)
        s = ("<p>Each symbol, LMS nudges every tap a little way down the slope of the squared error: "
             "w ← w + μ·u·e*. No matrix, no channel knowledge, just L multiplies per update. "
             f"With μ = {v(p.mu, '.3g')}, ")
        exc = self.r.get("excess", 0)
        if isinstance(self.r.get("mse"), float) and self.r.get("mse", 0) > 5:
            s += bad("the steps are too big: the taps overshoot, oscillate and diverge.") + "</p>"
        elif self.conv is not None:
            s += (f"it reached the Wiener floor (green) after about {v(self.conv, 'd')} symbols and now "
                  f"jitters {v(exc, '.2f', 'dB')} above it: the <b>misadjustment</b>, proportional to "
                  "μ.</p>")
        else:
            s += "it is still descending toward the Wiener floor (green).</p>"
        s += ("<p>Big μ learns fast but jitters; small μ is precise but slow. Real receivers "
              "<b>gear-shift</b>: a large μ to acquire, a small one to track. After the training "
              "symbols (orange) the slicer's own decisions become the reference: "
              "<b>decision-directed</b> mode, which works once most decisions are right.</p>")
        return "<h3>Learning by small steps</h3>" + s + keybox(
            "Speed and accuracy trade through μ: convergence time ≈ 1/(μλ_min), misadjustment ≈ μ·L·P_y/2.")


# =============================================================================== 5. LMS vs RLS
def haykin_curves(W, mu, lam, runs, N=400, L=11, delay=7, var_v=0.001, seed=0):
    """Ensemble learning curves of LMS and RLS on Haykin's raised-cosine channel (batched)."""
    r = np.random.default_rng(seed)
    h = 0.5 * (1 + np.cos(2 * np.pi / W * (np.arange(1, 4) - 2)))
    a = r.choice([-1.0, 1.0], size=(runs, N + L + 5))
    u = np.empty_like(a)
    for i in range(runs):
        u[i] = np.convolve(a[i], np.r_[0, h])[:a.shape[1]]
    u += np.sqrt(var_v) * r.standard_normal(a.shape)
    out = {}
    for algo in ("lms", "rls"):
        w = np.zeros((runs, L))
        P = np.repeat(np.eye(L)[None] / 0.004, runs, axis=0)
        mse = np.empty(N)
        for n in range(N):
            t = n + L + 2
            U = u[:, t - L + 1:t + 1][:, ::-1]
            d = a[:, t - delay]
            e = d - np.sum(w * U, axis=1)
            if algo == "lms":
                w += mu * e[:, None] * U
            else:
                PU = np.einsum("rij,rj->ri", P, U)
                k = PU / (lam + np.sum(U * PU, axis=1))[:, None]
                w += k * e[:, None]
                P = (P - np.einsum("ri,rj->rij", k, PU)) / lam
            mse[n] = np.mean(np.minimum(e ** 2, 1e6))
        out[algo] = mse
    r0 = np.sum(h ** 2) + var_v
    rr = [r0, h[0] * h[1] + h[1] * h[2], h[0] * h[2]] + [0] * (L - 3)
    lamv = np.linalg.eigvalsh(toeplitz(rr))
    return out["lms"], out["rls"], lamv


def settle_time(m, frac=2.0):
    """First iteration after which the smoothed curve stays below frac × its final value."""
    k = 10
    sm = np.convolve(m, np.ones(k) / k, mode="valid")
    final = np.mean(m[-60:])
    below = np.flatnonzero(sm <= frac * final)
    return int(below[0] + k) if len(below) else len(m)


class LMSvsRLS(Experiment):
    title = "LMS vs RLS: eigenvalue spread"
    blurb = "Haykin's classic: a coloured input slows LMS to a crawl. RLS does not care."
    book = "sec:ch12:rls"
    controls = [
        Slider("W", "Channel parameter W", 2.9, 3.6, 3.1, step=0.01,
               help="Larger W = more ISI = a more coloured equalizer input = bigger eigenvalue spread"),
        LogSlider("mu", "LMS step size μ", 0.005, 0.15, 0.05),
        Slider("lam", "RLS forgetting factor λ", 0.95, 1.0, 1.0, step=0.001),
        Choice("runs", "Runs averaged", ["50", "200"]),
    ]
    plots = [
        Plot("lc", "Ensemble-averaged learning curves (11 taps)", x="iteration n",
             y="mean squared error", logy=True, xlim=(0, 400), ylim=(1e-3, 3), legend="tr"),
        BarPlot("eig", "Eigenvalues of the input correlation matrix", x="eigenvalue index",
                y="λ_i"),
    ]
    layout = [["lc", "eig"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("chi", "Eigenvalue spread χ", "", ".1f"),
        Readout("tl", "LMS settles in", "iterations", "int"),
        Readout("tr", "RLS settles in", "iterations", "int"),
        Readout("ratio", "LMS / RLS time", "×", ".1f"),
    ]
    challenges = [
        Challenge("Make LMS at least 8× slower than RLS.", lambda s: s.r.ratio >= 8),
        Challenge("With W = 2.9, choose μ so LMS settles in under 70 iterations without blowing up.",
                  lambda s: abs(s.p.W - 2.9) < 0.006 and 0 < s.r.tl < 70 and s.exp.final_lms < 0.05),
        Challenge("Push μ until LMS goes unstable (final MSE above 1).",
                  lambda s: s.exp.final_lms > 1),
    ]

    def update(self, p):
        lms, rls, lamv = haykin_curves(p.W, p.mu, p.lam, int(p.runs))
        pl = self.plot("lc")
        n = np.arange(len(lms))
        pl.line("l", n, np.clip(lms, 1e-4, 1e3), color=RED, width=1.6, name=f"LMS, μ = {p.mu:.3g}")
        pl.line("r", n, np.clip(rls, 1e-4, 1e3), color=NAVY, width=1.6, name=f"RLS, λ = {p.lam:.3f}")
        pl.hline("j", 0.001, color=GRAY, style=":", label="noise floor", label_pos=0.02)
        pe = self.plot("eig")
        lv = np.sort(lamv)[::-1]
        pe.bars("b", np.arange(len(lv)), lv, width=0.6,
                colors=[RED if i in (0, len(lv) - 1) else NAVY for i in range(len(lv))])
        pe.set_ylim(0, 1.15 * lv.max())
        pe.text("ratio", len(lv) - 1, lv.max() * 1.1, f"χ = λ_max/λ_min = {lv.max() / lv.min():.1f}",
                anchor=(1, 0), size=9, bold=True)
        pe.set_xlim(-0.6, len(lv) - 0.4)
        self.final_lms = float(np.mean(lms[-60:]))
        tl = settle_time(lms) if self.final_lms < 1 else 999
        tr = settle_time(rls)
        self.readout(chi=float(lv.max() / lv.min()), tl=tl, tr=tr, ratio=tl / max(tr, 1))

    def story(self, p):
        chi = self.r.get("chi", 1)
        s = (f"<p>LMS descends the error bowl along its gradient. When the equalizer's input is "
             f"coloured (here by a channel with W = {v(p.W, '.2f')}), the bowl is a long narrow valley: "
             f"the ratio of its steepest to its flattest direction is the eigenvalue spread χ = "
             f"{v(chi, '.1f')} (bars, right). μ must be small enough for the steep direction, so the "
             f"flat one crawls: convergence time grows roughly in proportion to χ.</p>"
             "<p><b>RLS</b> keeps a running inverse of the correlation matrix, which reshapes the valley "
             "into a round bowl. It converges in about 2L iterations whatever χ is, at O(L²) cost per "
             "symbol instead of O(L).</p>")
        if self.__dict__.get("final_lms", 0) > 1:
            s += "<p>" + bad("LMS has diverged:") + " μ exceeds the stability limit set by the largest eigenvalue.</p>"
        return "<h3>Speed costs computation</h3>" + s + keybox(
            "LMS: cheap, robust, slow on coloured inputs. RLS: fast, expensive, numerically delicate. "
            "Burst modems and GSM-class receivers choose RLS-like methods; SerDes choose sign-sign LMS.")


# =============================================================================== 6. CMA
class BlindCMA(Experiment):
    title = "Blind equalization: CMA"
    blurb = "No training, no carrier lock: the constant-modulus algorithm opens the eye anyway."
    book = "sec:ch12:blind"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Choice("mod", "Modulation", ["QPSK", "16-QAM"]),
        Slider("a", "Channel echo a", 0.0, 0.9, 0.5, step=0.01),
        Slider("phase", "Unknown carrier phase", 0, 90, 30, step=1, unit="°"),
        LogSlider("mu", "Step size μ", 1e-4, 2e-2, 2e-3),
        Toggle("dd", "Then decision-directed LMS", True,
               help="After 12000 symbols, switch from CMA to decision-directed LMS"),
        Button("restart", "Restart"),
    ]
    plots = [
        ConstellationPlot("rx", "Received (before the equalizer)", lim=2.0),
        ConstellationPlot("out", "Equalizer output (last 500)", lim=1.8),
        Plot("cost", "Constant-modulus dispersion (100-symbol average)", x="symbol",
             y="dispersion (dB)", xlim=(0, 20000), ylim=(-30, 10), legend=None),
    ]
    layout = [["rx", "out"], ["cost", "cost"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("n", "Symbols", "", "int"),
        Readout("disp", "Dispersion", "dB", ".1f", good=lambda x: x < -10),
        Readout("rot", "Residual rotation", "°", ".1f"),
        Readout("mse", "Error after DD", "dB", ".1f", good=lambda x: x < -15),
    ]
    challenges = [
        Challenge("Open a QPSK eye blind and fast: dispersion below −18 dB within 3000 symbols.",
                  lambda s: s.p.mod == "QPSK" and s.r.n <= 3000 and s.r.disp < -18,
                  hint="A bigger μ learns faster, until it gets noisy."),
        Challenge("16-QAM: CMA then decision-directed LMS, error below −20 dB.",
                  lambda s: s.p.mod == "16-QAM" and s.p.dd and s.r.mse < -20,
                  hint="DD only works if the slicer is mostly right: with 16-QAM, a large carrier "
                       "rotation puts the outer points in the wrong boxes. Keep the phase small."),
        Challenge("See the phase blindness: CMA converged (dispersion < −8 dB) yet rotated by more "
                  "than 20°, with DD off.",
                  lambda s: not s.p.dd and s.r.disp < -8 and abs(s.r.rot) > 20),
    ]
    N = 20000
    chunk = 400
    switch = 12000

    def setup(self):
        self.seed = 4

    def on_restart(self, p):
        self.seed += 1

    def update(self, p):
        con = QPSK if p.mod == "QPSK" else QAM16
        self.con = con
        h = np.array([1, p.a * np.exp(0.7j), -0.25j * p.a, 0.1 * p.a])
        h = h / np.linalg.norm(h)
        rng = np.random.default_rng(self.seed)
        self.st = _Stream(con, h, self.N, 28.0, 21, rng, phase=np.deg2rad(p.phase))
        pts = con.points
        self.R2 = float(np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2))
        self.w = np.zeros(21, complex)
        self.w[self.st.delay] = 1
        self.n = 0
        self.z = np.zeros(self.N, complex)
        self.cost = np.zeros(self.N)
        self.err = np.full(self.N, np.nan)
        self.run(p, 300)

    def tick(self, p):
        if self.n < self.N:
            self.run(p, self.chunk)
        else:
            self.draw(p)

    def run(self, p, k):
        st_, w, pts, R2 = self.st, self.w, self.con.points, self.R2
        for n in range(self.n, min(self.n + k, self.N)):
            u = st_.u(n)
            z = np.dot(w.conj(), u)
            if not np.isfinite(z) or abs(z) > 1e4:
                w[:] = 0
                w[st_.delay] = 1
                z = 0j
            if p.dd and n >= self.switch:
                e = pts[np.argmin(np.abs(z - pts))] - z
                w += 1e-3 * u * np.conj(e)
                self.err[n] = abs(e) ** 2
            else:
                w -= p.mu * u * np.conj(z * (abs(z) ** 2 - R2))
            self.z[n] = z
            self.cost[n] = (abs(z) ** 2 - R2) ** 2 / R2 ** 2
        self.n = min(self.n + k, self.N)
        self.draw(p)

    def draw(self, p):
        n = self.n
        lo = max(0, n - 500)
        zz = self.z[lo:n]
        pr_ = self.plot("rx")
        pr_.points("y", self.st.y[lo:n], color=GRAY, size=3, alpha=0.4)
        pr_.ideal("p", self.con.points)
        po = self.plot("out")
        col = GREEN if (p.dd and n > self.switch) else NAVY
        po.points("z", zz, color=col, size=3, alpha=0.5)
        po.ideal("p", self.con.points)
        t = np.linspace(0, 2 * np.pi, 200)
        r = np.sqrt(self.R2)
        po.line("ring", r * np.cos(t), r * np.sin(t), color=ORANGE, width=1.0, style="--")
        k = 100
        c = self.cost[:n]
        sm = db(np.convolve(c, np.ones(k) / k, mode="valid")) if n > k else db(c)
        pc = self.plot("cost")
        pc.line("c", np.arange(len(sm)) + min(k, n), np.clip(sm, -30, 10), color=NAVY, width=1.5)
        if p.dd:
            pc.vline("sw", self.switch, color=GREEN, style=":", label="→ decision-directed", label_pos=0.9)
        recent = zz[-300:] if len(zz) else np.zeros(1)
        rot = float(np.rad2deg(np.angle(-np.mean(recent ** 4)) / 4)) if len(recent) > 10 else 0.0
        er = self.err[max(0, n - 300):n]
        er = er[np.isfinite(er)]
        self.readout(n=n, disp=float(sm[-1]) if len(sm) else 0.0, rot=rot,
                     mse=float(db(np.mean(er))) if len(er) > 50 else "—")

    def story(self, p):
        s = ("<p>CMA knows nothing about the symbols except that their magnitude should be "
             "constant: it minimises E[(|z|² − R₂)²], pulling every output toward the orange ring. ISI "
             "spreads the magnitudes; removing ISI is the only way to make them constant. And because "
             "|z| ignores phase, CMA works <b>before</b> the carrier is locked.</p>")
        s += (f"<p>The same blindness means CMA cannot see the {v(p.phase, '.0f', '°')} carrier "
              "rotation: the cleaned-up constellation comes out rotated (readout). ")
        if p.dd:
            s += ("After 12 000 symbols the receiver switches to <b>decision-directed LMS</b> (green), "
                  "which uses the slicer's decisions, fine-tunes the taps and rotates the constellation "
                  "onto the nearest valid orientation.</p>")
        else:
            s += "A carrier-recovery loop must remove it afterwards (Chapter 10).</p>"
        if p.mod == "16-QAM":
            s += ("<p>16-QAM's three rings make the CM error noisy even at the optimum, so CMA alone "
                  "leaves a fuzzy constellation: the DD stage is essential.</p>")
        return "<h3>Seeing without looking</h3>" + s + keybox(
            "Godard (1980): blind equalization made multipoint modems and cable-TV QAM possible. GNU "
            "Radio's PSK receiver still starts with a CMA equalizer.")


# =============================================================================== 7. DFE
def dfe_bpsk(y, wf, wb, delay, fb_truth=None):
    """Fast BPSK decision-feedback equalizer (real decisions). Returns decisions (aligned to a)."""
    N = len(y)
    zff = np.convolve(y, np.conj(wf))[:N].real
    wbr = list(np.real(np.conj(wb)))
    Lb = len(wbr)
    M = N - delay
    dec = [0.0] * M
    fb = fb_truth if fb_truth is not None else dec
    zl = zff[delay:].tolist()
    for m in range(M):
        z = zl[m]
        for k in range(min(Lb, m)):
            z -= wbr[k] * fb[m - 1 - k]
        dec[m] = 1.0 if z >= 0 else -1.0
    return np.array(dec)


def bursts(err, gap):
    """Lengths (number of errors) of error bursts: errors closer than `gap` belong together."""
    idx = np.flatnonzero(err)
    if len(idx) == 0:
        return np.array([], int)
    br = np.flatnonzero(np.diff(idx) > gap)
    starts = np.r_[0, br + 1]
    ends = np.r_[br + 1, len(idx)]
    return ends - starts


class DFE(Experiment):
    title = "Decision feedback and error bursts"
    blurb = "Subtract the echoes of symbols you already decided: no noise boost, but errors breed."
    book = "sec:ch12:dfe"
    heavy = True
    controls = [
        Choice("ch", "Channel", ["Proakis B", "Two-path [1, a]", "Proakis C"], style="menu"),
        Slider("a", "Echo amplitude a", 0.0, 1.0, 0.95, step=0.01,
               enabled_if=lambda p: p.ch.startswith("Two")),
        Slider("ebn0", "Eb/N0", 0, 20, 9, step=0.5, unit="dB"),
        IntSlider("Lb", "Feedback taps", 1, 6, 2),
    ]
    plots = [
        Plot("err", "Where the errors fall (first 4000 bits)", x="bit index", y="",
             xlim=(0, 4000), ylim=(-0.6, 2.6), legend=None),
        BarPlot("hist", "Error burst lengths", x="errors per burst", y="bursts"),
        BERPlot("ber", "BPSK bit error rate", x="Eb/N0 (dB)", ylim=(1e-5, 0.5), xlim=(0, 16),
                legend="bl"),
    ]
    layout = [["err", "err"], ["hist", "ber"]]
    row_stretch = [2, 3]
    col_stretch = [2, 3]
    readouts = [
        Readout("lin", "BER, linear MMSE", "", "sci"),
        Readout("dfe", "BER, DFE", "", "sci"),
        Readout("gen", "BER, DFE with correct feedback", "", "sci"),
        Readout("burst", "Mean burst length (DFE)", "errors", ".2f"),
    ]
    challenges = [
        Challenge("Watch errors breed: make the mean DFE burst 4 errors or longer.",
                  lambda s: s.r.burst >= 4,
                  hint="A channel with long, strong post-cursors, or too few feedback taps."),
        Challenge("Proakis B: get the DFE below 10⁻³ with an Eb/N0 of 12 dB or less.",
                  lambda s: s.p.ch == "Proakis B" and s.p.ebn0 <= 12 and 0 < s.r.dfe < 1e-3
                  or (s.p.ch == "Proakis B" and s.p.ebn0 <= 12 and s.r.dfe == 0 and s.r.gen == 0)),
        Challenge("Beat the linear MMSE equalizer by 10× in BER on Proakis B.",
                  lambda s: s.p.ch == "Proakis B" and s.r.dfe > 0 and s.r.lin >= 10 * s.r.dfe),
    ]
    N = 20000

    def channel(self, p):
        return make_channel(p.ch, p.a).real

    def simulate(self, h, ebn0, Lb, N, rng, mlse=False):
        n0 = 10 ** (-ebn0 / 10)
        a = rng.choice([-1.0, 1.0], N)
        y = np.convolve(a, h)[:N] + np.sqrt(n0 / 2) * rng.standard_normal(N)
        w, d = cl.mmse_fir(h, 31, n0 / 2)
        zl = cl.apply_fir(y, w, d).real
        e_lin = (np.sign(zl) != a)
        wf, wb, dl, _ = eqadv.mmse_dfe_fir(h, 11, Lb, n0 / 2)
        dec = dfe_bpsk(y, wf, wb, dl)
        e_dfe = dec != a[:len(dec)]
        dg = dfe_bpsk(y, wf, wb, dl, fb_truth=a.tolist())
        e_gen = dg != a[:len(dg)]
        out = dict(lin=e_lin, dfe=e_dfe, gen=e_gen)
        if mlse:
            ah = eqadv.viterbi_mlse(y, h, [1.0, -1.0]).real
            out["mlse"] = ah != a
        return out

    def update(self, p):
        h = self.channel(p)
        rng = np.random.default_rng(6)
        r = self.simulate(h, p.ebn0, p.Lb, self.N, rng)
        sl = slice(60, -60)
        ber = {k: float(np.mean(e[sl])) for k, e in r.items()}
        pe = self.plot("err")
        for row, (k, col, name) in enumerate([("lin", ORANGE, "linear MMSE"), ("dfe", RED, "DFE"),
                                              ("gen", GREEN, "DFE, genie")]):
            idx = np.flatnonzero(r[k][:4000])
            pe.scatter(k, idx, np.full(len(idx), 2 - row), color=col, size=6, symbol="s")
        pe.set_yticks([(2, "linear"), (1, "DFE"), (0, "genie DFE")])
        bl = bursts(r["dfe"][sl], p.Lb)
        bg = bursts(r["gen"][sl], p.Lb)
        mx = 8
        hd = np.bincount(np.minimum(bl, mx), minlength=mx + 1)[1:]
        hg = np.bincount(np.minimum(bg, mx), minlength=mx + 1)[1:]
        x = np.arange(1, mx + 1)
        ph = self.plot("hist")
        ph.bars("d", x - 0.18, hd, width=0.34, color=RED, name="DFE")
        ph.bars("g", x + 0.18, hg, width=0.34, color=GREEN, name="genie")
        ph.set_xticks([(i, str(i) if i < mx else f"{mx}+") for i in x])
        ph.set_xlim(0.4, mx + 0.6)
        ph.set_ylim(0, max(5, 1.15 * max(hd.max(initial=0), hg.max(initial=0))))
        pb = self.plot("ber")
        xx = np.linspace(0, 16, 161)
        pb.theory("mfb", xx, cl.ber_bpsk(xx), color=GRAY, style="--", name="no ISI (MFB)")
        pb.vline("now", p.ebn0, color=GRAY, style=":")
        self.curves = {k: ([], []) for k in ("lin", "dfe", "gen", "mlse")}
        self.readout(lin=ber["lin"], dfe=ber["dfe"], gen=ber["gen"],
                     burst=float(np.mean(bl)) if len(bl) else 0.0)

    def background(self, p):
        h = self.channel(p)
        rng = np.random.default_rng(9)
        grid = np.arange(0, 17, 2.0 if self.quick else 1.0)
        N = 4000 if self.quick else 30000
        for e in grid:
            r = self.simulate(h, e, p.Lb, N, rng, mlse=True)
            yield {k: (e, float(np.mean(v_[60:-60]))) for k, v_ in r.items()}

    def progress(self, p, it):
        pb = self.plot("ber")
        sty = {"lin": (ORANGE, "linear MMSE (31 taps)"), "dfe": (RED, "MMSE-DFE"),
               "gen": (GREEN, "DFE, correct feedback"), "mlse": (NAVY, "MLSE (Viterbi)")}
        for k, (e, b) in it.items():
            if b > 0:
                self.curves[k][0].append(e)
                self.curves[k][1].append(b)
            xs, ys = self.curves[k]
            if xs:
                pb.sim(k, xs, ys, color=sty[k][0], name=sty[k][1], size=7)

    def story(self, p):
        s = ("<p>A <b>decision-feedback equalizer</b> uses a short feedforward filter for the "
             "precursors, then <i>subtracts</i> the post-cursor ISI computed from symbols it has "
             "already decided. Decisions carry no noise, so the DFE needs no noise-boosting inverse "
             "of the notch.</p>")
        s += (f"<p>The catch is <b>error propagation</b>: a wrong decision subtracts the wrong echo, "
              f"doubling the ISI on the next {v(p.Lb, 'd')} symbol(s) and often causing another error. "
              f"Errors arrive in bursts (top: red squares cluster, green ones do not); the mean DFE "
              f"burst here is {v(self.r.get('burst', 0), '.2f')} errors.</p>"
              "<p>The background run (right) compares all four receivers; MLSE (experiment 8) is "
              "the best of all.</p>")
        return "<h3>Feed back, carefully</h3>" + s + keybox(
            "DFE: no noise enhancement for post-cursor ISI, at the price of error bursts. Precoding "
            "(THP, 1/(1⊕D)) moves the feedback to the transmitter to avoid them.")


# =============================================================================== 8. MLSE trellis
class MLSETrellis(Experiment):
    title = "MLSE: the Viterbi equalizer"
    blurb = "Stop fighting the ISI and use it: the trellis keeps every hypothesis alive."
    book = "sec:ch12:mlse"
    animate = True
    autoplay = True
    fps = 6
    controls = [
        Choice("ch", "Channel", ["Proakis B", "Two-path a=0.9", "Proakis C"],
               help="Proakis B: 4 states; two-path: 2 states; Proakis C: 16 states"),
        Slider("ebn0", "Eb/N0", 0, 16, 6, step=0.5, unit="dB"),
        Button("reset", "Restart"),
    ]
    plots = [
        Plot("tr", "Trellis: survivor paths (gray), best path (green), truth (dashed)",
             x="time (symbols)", y="state (previous symbols)", legend=None),
        Plot("rx", "Received samples and the noiseless levels the trellis expects",
             x="time (symbols)", y="amplitude", legend=None),
    ]
    layout = [["tr"], ["rx"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("d2", "d²_min / MFB distance", "", ".3f"),
        Readout("loss", "Asymptotic MLSE loss", "dB", ".2f"),
        Readout("merge", "Survivors merge after", "symbols", "int"),
        Readout("ber", "BER so far", "", "sci"),
    ]
    challenges = [
        Challenge("Proakis C: see the 5 dB MLSE loss of the book's worked example.",
                  lambda s: s.p.ch.startswith("Proakis C") and abs(s.r.loss - 5.0) < 0.3),
        Challenge("Catch the survivors disagreeing for 8 symbols or more before they merge.",
                  lambda s: s.r.merge >= 8,
                  hint="More states and more noise keep more hypotheses alive for longer."),
        Challenge("Run 500 symbols on Proakis B at Eb/N0 ≤ 9 dB with fewer than 0.5 % errors.",
                  lambda s: s.p.ch.startswith("Proakis B") and s.p.ebn0 <= 9 and s.exp.ndec >= 500
                  and s.r.ber < 0.005),
    ]
    W = 24
    D = 20

    def setup(self):
        self.seed = 3

    def on_reset(self, p):
        self.seed += 1

    def channel(self, p):
        if p.ch.startswith("Proakis B"):
            return make_channel("Proakis B").real
        if p.ch.startswith("Two"):
            return make_channel("Two-path [1, a]", 0.9).real
        return make_channel("Proakis C").real

    @staticmethod
    def dmin2(h, maxlen=8):
        """Minimum squared output distance over BPSK error events (ε_k ∈ {0, ±2})."""
        best = np.inf
        for L in range(1, maxlen + 1):
            for code in range(3 ** (L - 1)):
                e = [2.0]
                c = code
                for _ in range(L - 1):
                    e.append([0.0, 2.0, -2.0][c % 3])
                    c //= 3
                if e[-1] == 0:
                    continue
                best = min(best, float(np.sum(np.convolve(e, h) ** 2)))
        return best

    def update(self, p):
        h = self.channel(p)
        self.h = h
        m = len(h) - 1
        self.m, self.S = m, 2 ** m
        ns = np.arange(self.S)
        P = np.array([1.0, -1.0])
        digits = np.array([(ns >> k) & 1 for k in range(m)])         # digit k -> a[n-1-k]
        self.exp_out = np.empty((self.S, 2))
        for i in range(2):
            self.exp_out[:, i] = h[0] * P[i] + sum(h[k + 1] * P[digits[k]] for k in range(m))
        self.pred = np.array([(ns >> 1) + j * 2 ** (m - 1) for j in range(2)]).T
        self.inp = ns & 1
        self.metric = np.zeros(self.S)
        self.tb = []
        rng = np.random.default_rng(self.seed)
        self.rng_run = rng
        self.a = []
        self.y = []
        self.st_true = []
        self.state = 0
        self.ndec = 0
        self.nerr = 0
        d2 = self.dmin2(h, 6 if m > 2 else 8)
        self.d2ratio = d2 / 4.0
        for _ in range(self.W + 4):
            self.step(p)
        self.draw(p)

    def step(self, p):
        rng = self.rng_run
        bit = int(rng.integers(0, 2))
        new_state = bit + 2 * (self.state % 2 ** (self.m - 1)) if self.m > 1 else bit
        n0 = 10 ** (-p.ebn0 / 10)
        yv = self.exp_out[self.state, bit] + np.sqrt(n0 / 2) * rng.standard_normal()
        self.state = new_state
        self.a.append(1 - 2 * bit)
        self.y.append(yv)
        self.st_true.append(new_state)
        bm = (yv - self.exp_out) ** 2
        cand = self.metric[self.pred] + bm[self.pred, self.inp[:, None]]
        j = np.argmin(cand, axis=1)
        self.metric = cand[np.arange(self.S), j]
        self.metric -= self.metric.min()
        self.tb.append(j)
        # decide symbol n-D by tracing back from the best state
        n = len(self.y) - 1
        if n >= self.D:
            s = int(np.argmin(self.metric))
            for t in range(n, n - self.D, -1):
                s = int(self.pred[s, self.tb[t][s]])
            dec = 1 - 2 * (s & 1)
            self.ndec += 1
            self.nerr += int(dec != self.a[n - self.D])

    def tick(self, p):
        self.step(p)
        self.draw(p)

    def draw(self, p):
        n = len(self.y)
        t0 = max(0, n - self.W)
        S = self.S
        xs, ys = [], []
        for t in range(t0 + 1, n):
            for s in range(S):
                xs += [t - 1, t, np.nan]
                ys += [int(self.pred[s, self.tb[t][s]]), s, np.nan]
        pt = self.plot("tr")
        pt.line("surv", xs, ys, color=GRAY, width=1.0, alpha=0.6)
        # best path from the current best state
        s = int(np.argmin(self.metric))
        path = [s]
        for t in range(n - 1, t0, -1):
            s = int(self.pred[s, self.tb[t][s]])
            path.append(s)
        path = path[::-1]
        tt = np.arange(n - len(path), n)
        pt.line("best", tt, path, color=GREEN, width=3.0)
        pt.line("true", np.arange(t0, n), self.st_true[t0:n], color=NAVY, width=1.6, style="--")
        gx, gy = np.meshgrid(np.arange(t0, n), np.arange(S))
        pt.scatter("nodes", gx.ravel(), gy.ravel(), color=NAVY, size=4 if S > 4 else 6)
        # merge depth: how far back all survivors agree
        heads = list(range(S))
        merge = 0
        for t in range(n - 1, t0, -1):
            heads = [int(self.pred[h_, self.tb[t][h_]]) for h_ in heads]
            merge += 1
            if len(set(heads)) == 1:
                break
        else:
            merge = self.W
        self.merge = merge
        pt.vline("dec", n - 1 - self.D, color=RED, style=":", label="decided", label_pos=0.95)
        pt.set_xlim(t0 - 0.5, t0 + self.W + 0.5)
        pt.set_ylim(-0.6, S - 0.4)
        lab = lambda s_: " ".join("+" if not (s_ >> k) & 1 else "−" for k in range(self.m))
        if S <= 4:
            pt.set_yticks([(s_, lab(s_)) for s_ in range(S)])
        else:
            pt.set_yticks([(s_, str(s_)) for s_ in range(0, S, 3)])
        pr_ = self.plot("rx")
        levels = np.unique(np.round(self.exp_out.ravel(), 6))
        for i, L in enumerate(levels):
            pr_.hline(f"l{i}", L, color=ORANGE, style=":", width=1.3)
        pr_.scatter("y", np.arange(t0, n), self.y[t0:n], color=NAVY, size=8)
        pr_.set_xlim(t0 - 0.5, t0 + self.W + 0.5)
        lim = max(np.abs(levels).max() + 0.8, 2.0)
        pr_.set_ylim(-lim, lim)
        self.readout(d2=self.d2ratio, loss=float(-10 * np.log10(self.d2ratio)), merge=merge,
                     ber=self.nerr / self.ndec if self.ndec else 0.0)

    def story(self, p):
        s = (f"<p>The channel has memory {v(self.m, 'd')}, so the noiseless received sample depends on "
             f"the current bit and the {v(self.m, 'd')} before it: {v(self.S, 'd')} <b>states</b> (rows) "
             f"and {v(2 * self.S, 'd')} possible levels (dotted lines, bottom). Instead of slicing each "
             "sample, the Viterbi algorithm keeps, for every state, the single best path into it: the "
             "<b>survivors</b> (gray).</p>")
        s += (f"<p>Looking back, the survivors merge into one path after about "
              f"{v(self.r.get('merge', 0), 'd')} symbols: everything before the merge is decided, "
              f"whichever state wins later. The receiver decides with a fixed delay of {self.D} "
              f"symbols (red line).</p>"
              f"<p>MLSE uses all the energy the echoes carry. It loses only "
              f"{v(self.r.get('loss', 0), '.2f', 'dB')} to the matched-filter bound here, set by the "
              f"closest pair of sequences (d²_min).</p>")
        return "<h3>Every hypothesis, kept alive</h3>" + s + keybox(
            "Forney (1972): MLSE is optimal but costs M^memory states. GSM receivers ran a 16-state "
            "Viterbi equalizer on every burst; 224G SerDes are bringing it back.")


# =============================================================================== 9. SerDes
@lru_cache(maxsize=32)
def _backplane(loss_db, sps, nfft):
    return lc.backplane(sps, loss_db, 0.5, nfft=nfft, length=nfft // 2)


class SerDes(Experiment):
    title = "SerDes: CTLE, FFE and DFE"
    blurb = "A 100 Gb/s lane over a lossy backplane: the whole equalizer family in one receiver."
    book = "sec:ch12:serdes"
    animate = True
    fps = 8
    controls = [
        Choice("mod", "Signalling", ["NRZ", "PAM-4"], "PAM-4"),
        Slider("loss", "Channel loss at Nyquist", 5, 40, 28, step=0.5, unit="dB"),
        Slider("snr", "Input SNR (cursor / noise)", 15, 50, 32, step=0.5, unit="dB"),
        Heading("Receiver equalizers"),
        Slider("pk", "CTLE peaking", 0, 20, 0, step=0.5, unit="dB"),
        IntSlider("ffe", "FFE taps", 1, 15, 1, help="1 = no FFE (a single gain)"),
        IntSlider("dfe", "DFE taps", 0, 4, 0),
    ]
    plots = [
        EyePlot("eye", "Eye at the slicer", yrange=(-1.8, 1.8)),
        Plot("pulse", "Pulse response (one UI) and its cursors", x="time (UI, 0 = main cursor)",
             y="amplitude (re main cursor)", xlim=(-4, 12), ylim=(-0.5, 1.2), legend="tr"),
        Plot("freq", "Frequency responses", x="frequency (× baud rate)", y="gain (dB)",
             xlim=(0, 1.0), ylim=(-50, 25), legend="bl"),
    ]
    layout = [["eye", "pulse"], ["eye", "freq"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("height", "Eye height", "%", ".0f", good=lambda x: x > 30),
        Readout("isi", "Residual ISI (peak)", "%", ".0f", good=lambda x: x < 20),
        Readout("enh", "Noise enhancement", "dB", ".1f"),
        Readout("snr", "SNR at the slicer", "dB", ".1f"),
    ]
    challenges = [
        Challenge("NRZ over a 25 dB channel: open the eye at least 40 % with the CTLE alone.",
                  lambda s: s.p.mod == "NRZ" and s.p.loss >= 25 and s.p.ffe == 1 and s.p.dfe == 0
                  and s.r.height >= 40),
        Challenge("PAM-4 over 28 dB: eye at least 35 % open using no more than 5 FFE taps.",
                  lambda s: s.p.mod == "PAM-4" and s.p.loss >= 28 and s.p.ffe <= 5 and s.r.height >= 35),
        Challenge("Show what DFE taps are worth: PAM-4 over 32 dB, at most 3 FFE taps, and a 35 % "
                  "eye height.",
                  lambda s: s.p.mod == "PAM-4" and s.p.loss >= 32 and s.p.ffe <= 3 and s.r.height >= 35,
                  hint="Too much CTLE peaking boosts noise; let the DFE take the post-cursors."),
    ]
    sps, nfft = 16, 1 << 13
    nsym = 900

    def design(self, p):
        sps = self.sps
        hch = _backplane(round(p.loss, 2), sps, self.nfft)
        Nf = self.nfft
        f = np.fft.rfftfreq(Nf, 1 / sps)
        fp1, fp2 = 0.5, 1.0
        fz = fp1 / 10 ** (p.pk / 20)
        Hc = (1 + 1j * f / fz) / ((1 + 1j * f / fp1) * (1 + 1j * f / fp2))
        ctle = np.fft.irfft(Hc, Nf)[:Nf // 2]
        rect = np.ones(sps)
        p_raw = np.convolve(hch, rect)[:Nf // 2]
        p_eq = np.convolve(np.convolve(hch, ctle)[:Nf // 2], rect)[:Nf // 2]
        c = int(np.argmax(p_eq))
        pre = min(3, (p.ffe - 1) // 3) if p.ffe > 1 else 0
        npre, npost = 4, 30
        idx = c + sps * np.arange(-npre, npost + 1)
        idx = idx[(idx >= 0) & (idx < len(p_eq))]
        hs = p_eq[idx]
        main_i = npre if c - npre * sps >= 0 else int(np.argmax(hs))
        L = p.ffe
        H = cl.conv_matrix(hs, L).real
        tgt = main_i + pre
        keep = [i for i in range(H.shape[0]) if not (tgt < i <= tgt + p.dfe)]
        e = np.zeros(H.shape[0])
        e[tgt] = 1
        w = np.linalg.lstsq(H[keep], e[keep], rcond=None)[0] if L > 1 else np.array([1 / hs[main_i]])
        q = H @ w
        b = q[tgt + 1:tgt + 1 + p.dfe]
        return dict(hch=hch, ctle=ctle, p_raw=p_raw, p_eq=p_eq, c=c, pre=pre, w=w, q=q, tgt=tgt, b=b,
                    f=f, Hc=Hc)

    def update(self, p):
        self.dsn = self.design(p)
        self.plot("eye").reset_persistence()
        self.draw(p, accumulate=False)

    def tick(self, p):
        self.draw(p, accumulate=True)

    def draw(self, p, accumulate):
        D, sps = self.dsn, self.sps
        rng = self.rng
        lev = np.array([-1.0, 1.0]) if p.mod == "NRZ" else np.array([-1, -1 / 3, 1 / 3, 1.0])
        a = lev[rng.integers(0, len(lev), self.nsym)]
        up = np.zeros(self.nsym * sps)
        up[::sps] = a
        x = np.convolve(up, np.ones(sps))[:len(up)]
        y = np.convolve(x, D["hch"])[:len(up)]
        sigma = D["p_raw"].max() * 10 ** (-p.snr / 20)
        y = y + sigma * rng.standard_normal(len(y))
        y = np.convolve(y, D["ctle"])[:len(up)]
        w = D["w"]
        z = np.zeros_like(y)
        for k, wk in enumerate(w):
            z[k * sps:] += wk * y[:len(y) - k * sps]
        base = D["c"] + D["pre"] * sps
        for k, bk in enumerate(D["b"], start=1):
            for n in range(k, self.nsym):
                c0 = base + n * sps
                if c0 + sps // 2 > len(z):
                    break
                z[c0 - sps // 2:c0 + sps // 2] -= bk * a[n - k]
        z = z / D["q"][D["tgt"]]
        first = 40
        start = base + first * sps - sps
        pe = self.plot("eye")
        pe.eye("eye", z[:base + (self.nsym - 20) * sps], sps, n_sym=2, offset=start,
               yrange=(-1.8, 1.8), accumulate=accumulate, decay=0.85)
        # eye opening at the best phase
        k = np.arange(first, self.nsym - 25)
        phases = np.arange(-sps // 2 + 1, sps // 2)
        S = z[base + k[:, None] * sps + phases[None, :]]
        sym = a[k]
        sp = lev[1] - lev[0]
        opening = np.full(len(phases), np.inf)
        for j in range(len(lev) - 1):
            lo, hi = S[sym == lev[j]], S[sym == lev[j + 1]]
            if len(lo) and len(hi):
                opening = np.minimum(opening, hi.min(axis=0) - lo.max(axis=0))
        opening /= sp
        bi = int(np.argmax(opening))
        err = S[:, bi] - sym
        snr = float(db(np.mean(lev ** 2) / max(np.mean(err ** 2), 1e-12)))
        pe.vline("best", 1 + phases[bi] / sps, color=RED, style="--", width=1.0)
        # pulse responses
        pp = self.plot("pulse")
        t = (np.arange(len(D["p_raw"])) - D["c"]) / sps
        m = (t > -4.5) & (t < 12.5)
        pp.line("raw", t[m], D["p_raw"][m] / D["p_eq"].max() * (D["p_eq"].max() / D["p_raw"].max()),
                color=GRAY, width=1.4, name="channel only")
        pp.line("eq", t[m], D["p_eq"][m] / D["p_eq"].max(), color=ORANGE, width=1.6, name="+ CTLE")
        q = D["q"] / D["q"][D["tgt"]]
        kq = np.arange(len(q)) - D["tgt"]
        mk = (kq >= -4) & (kq <= 12)
        cols = [GREEN if (0 < kk <= p.dfe) else NAVY for kk in kq[mk]]
        pp.bars("q", kq[mk], q[mk], width=0.18, colors=cols)
        pp.scatter("qd", kq[mk], q[mk], color=NAVY, size=6, name="after FFE (cursors)")
        pp.hline("z", 0, color=GRAY, style="-", width=0.6)
        # frequency
        pf = self.plot("freq")
        f = np.fft.rfftfreq(self.nfft, 1 / sps)
        Hch = np.fft.rfft(D["hch"], self.nfft)
        Wf = np.fft.rfft(np.concatenate([np.r_[wk, np.zeros(sps - 1)] for wk in w]), self.nfft)
        mf = f <= 1.0
        band = f <= 0.5
        g0 = np.sqrt(np.mean(np.abs(Hch[band] * D["Hc"][band] * Wf[band]) ** 2))
        pf.line("ch", f[mf], db(np.abs(Hch[mf]) ** 2), color=GRAY, width=2, name="channel")
        pf.line("ct", f[mf], db(np.abs(D["Hc"][mf]) ** 2), color=ORANGE, width=1.6, name="CTLE")
        pf.line("ff", f[mf], db(np.abs(Wf[mf] / g0) ** 2), color=PURPLE, width=1.4, name="FFE")
        pf.line("tot", f[mf], db(np.abs(Hch[mf] * D["Hc"][mf] * Wf[mf] / g0) ** 2), color=NAVY,
                width=2.2, name="total")
        pf.vline("nyq", 0.5, color=GRAY, style=":", label="Nyquist", label_pos=0.95)
        # noise enhancement: output noise per unit cursor vs input noise per unit cursor
        g = np.convolve(D["ctle"], np.concatenate([np.r_[wk, np.zeros(sps - 1)] for wk in w]))
        c0, q0 = self.reference()
        enh = float(db((np.sum(g ** 2) / D["q"][D["tgt"]] ** 2) / (np.sum(c0 ** 2) / q0 ** 2)))
        resid = np.sum(np.abs(np.delete(q, [D["tgt"]] + list(range(D["tgt"] + 1, D["tgt"] + 1 + p.dfe)))))
        self.readout(height=max(100 * float(opening[bi]), -100), isi=100 * float(resid), enh=enh, snr=snr)

    def reference(self):
        """CTLE impulse response and main cursor with no peaking and no FFE (the noise reference)."""
        key = round(self.p.loss, 2)
        if getattr(self, "_ref_key", None) != key:
            q = st.Params(dict(self.p))
            q.pk, q.ffe, q.dfe = 0.0, 1, 0
            D0 = self.design(q)
            self._ref = (D0["ctle"], D0["p_eq"].max())
            self._ref_key = key
        return self._ref

    def story(self, p):
        h = self.r.get("height", 0)
        s = (f"<p>A {v(p.loss, '.0f', 'dB')} channel at Nyquist turns each one-UI pulse into a low, long "
             f"smear (gray, top right): only a fraction of its energy is in the main cursor and the rest "
             f"spills over dozens of UIs. The receiver fights back in stages:</p>"
             f"<p><b>CTLE</b> ({v(p.pk, '.0f', 'dB')} of analog peaking) boosts the highs, cheap but "
             f"noisy. <b>FFE</b> ({v(p.ffe, 'd')} taps) zero-forces the precursors and the tail. "
             f"<b>DFE</b> ({v(p.dfe, 'd')} taps, green cursors) cancels the first post-cursors "
             f"using decisions, with no noise boost. Noise enhancement so far: "
             f"{v(self.r.get('enh', 0), '.1f', 'dB')}.</p>")
        s += ("<p>" + (good(f"The eye is {h:.0f} % open.") if h > 30 else
                       bad(f"The eye is only {max(h, 0):.0f} % open.") if h > 0 else bad("The eye is closed."))
              + (" PAM-4 has three eyes, each a third of the swing: it needs a cleaner channel than NRZ "
                 "at the same baud rate." if p.mod == "PAM-4" else "")
              + (" The DFE subtracts its correction a whole UI at a time, hence the vertical seams at "
                 "the UI boundaries; only the centre, where the slicer samples, matters." if p.dfe else "")
              + "</p>")
        return "<h3>The equalizer family at work</h3>" + s + keybox(
            "Real 112G/224G SerDes: TX FFE + CTLE + ADC + tens of FFE taps + DFE or MLSE, adapted by "
            "sign-sign LMS, all for a raw BER better than the FEC threshold (about 2.4×10⁻⁴ for KP4).")


# =============================================================================== the lab
LAB = st.Lab(6, "Equalization", chapter=12, chapter_title="Equalization",
             experiments=[ISIChannel, ZFvsMMSE, LengthDelay, LMSLive, LMSvsRLS, BlindCMA, DFE,
                          MLSETrellis, SerDes])

if __name__ == "__main__":
    st.run(LAB)
