"""Lab 23 · Turbo Codes and EXIT Charts   (Chapter 15)

Run it:      python labs/lab23_turbo_exit.py
Self-test:   python labs/lab23_turbo_exit.py --selftest

In 1993 Berrou, Glavieux and Thitimajshima showed a code within about half a decibel of the
Shannon limit, and the turbo principle (exchange soft extrinsic information between simple
decoders) went on to power 3G, 4G, deep-space links and, through EXIT charts and density
evolution, the design of every modern code. Seven experiments: the recursive constituent
encoder and the QPP interleaver, BCJR against brute force, iterative decoding iteration by
iteration, the EXIT chart and its tunnel, density evolution on the erasure channel, the
decoding wave of a spatially coupled code, and the error floor set by the interleaver.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import itertools
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from commlib import fectools as ft
from commlib import infotheory as it
from commlib import turbo as tb
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, GRAY, TEAL, GOLD)
from studio import v, keybox, good, bad
from _fecviz import Canvas, cells, segments, pale, ink


# =============================================================================== shared helpers
RSC = tb.RSC()
QPP = {40: (3, 10), 256: (15, 32), 1024: (31, 64)}          # TS 36.212 Table 5.1.3-3


@lru_cache(maxsize=16)
def interleaver(K, kind):
    if kind == "QPP (LTE)":
        return tb.qpp_interleaver(K, *QPP[K])
    rng = np.random.default_rng(5)
    if kind == "Random":
        return rng.permutation(K)
    S = {40: 3, 256: 8, 1024: 12}[K]
    return ft.s_random_interleaver(K, S, rng)


@lru_cache(maxsize=16)
def turbo(K, kind="QPP (LTE)"):
    return tb.TurboCode(K, interleaver(K, kind))


@lru_cache(maxsize=8)
def bpsk_limit_db(R):
    es = brentq(lambda e: it.biawgn_capacity(e) - R, 1e-4, 50)
    return float(10 * np.log10(es / R))


def interp_cross(x, y, target):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = y > 0
    x, y = x[ok], y[ok]
    if len(x) < 2 or y.min() > target or y.max() < target:
        return None
    ly = np.log10(y)
    i = np.where(ly <= np.log10(target))[0][0]
    if i == 0:
        return None
    return float(np.interp(np.log10(target), [ly[i], ly[i - 1]], [x[i], x[i - 1]]))


def ff_parity(u, taps=(1, 1, 0, 1)):
    """Feed-forward parity with 1 + D + D^3 (the 15 octal tap set, without feedback)."""
    return np.convolve(u, taps)[:len(u)] % 2


# =============================================================================== 1. RSC + interleaver
class Constituents(Experiment):
    title = "Recursive encoders and the interleaver"
    blurb = "Why the constituent codes must be recursive, and why the interleaver must scatter."
    book = "sec:ch15:turbo"
    controls = [
        Choice("enc", "Constituent encoder", ["Recursive (13, 15), LTE", "Feed-forward (15)"],
               "Recursive (13, 15), LTE", style="menu"),
        Choice("pat", "Input pattern", ["A single 1", "Two 1s"], "Two 1s"),
        IntSlider("sep", "Distance between the two 1s", 1, 30, 5),
        Heading("Interleaver"),
        Choice("K", "Block length K", ["40", "256", "1024"], "40"),
        Choice("kind", "Interleaver", ["QPP (LTE)", "Random", "S-random"], "QPP (LTE)"),
        IntSlider("pos", "Position of the first 1", 0, 39, 3),
    ]
    plots = [
        Plot("enc1", "Encoder 1: input (orange) and parity (navy)", x="time k",
             y="", xlim=(-0.5, 39.5), ylim=(-0.3, 2.9), legend=None, grid=False),
        Plot("pi", "The interleaver: where the inputs go", x="input position i",
             y="π(i)", legend=None),
        Plot("enc2", "Encoder 2 sees the interleaved input", x="time k (interleaved order)",
             y="", xlim=(-0.5, 39.5), ylim=(-0.3, 2.9), legend=None, grid=False),
    ]
    layout = [["enc1", "pi"], ["enc2", "pi"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("w1", "Parity weight, encoder 1", "", "int"),
        Readout("sep2", "Distance after interleaving", "", "int"),
        Readout("w2", "Parity weight, encoder 2", "", "int"),
        Readout("wt", "Whole codeword weight", "", "int", good=lambda x: x >= 20),
    ]
    challenges = [
        Challenge("Find a two-ones input that the recursive encoder 1 finishes quickly: its parity "
                  "stops (separation a multiple of 7).",
                  lambda s: s.p.enc.startswith("Recursive") and s.p.pat == "Two 1s" and s.p.sep % 7 == 0),
        Challenge("Show why recursion matters: with the feed-forward encoder, a single 1 gives a "
                  "parity weight of only 3.", lambda s: s.p.enc.startswith("Feed") and s.p.pat == "A single 1"
                  and s.r.w1 == 3),
        Challenge("Find a weight-2 input that is short for both encoders: whole codeword weight "
                  "below 26 with K = 40.", lambda s: s.p.K == "40" and s.p.pat == "Two 1s"
                  and s.p.enc.startswith("Recursive") and s.r.wt < 26,
                  hint="Look for a separation that is a multiple of 7 before and after π."),
    ]

    def update(self, p):
        K = int(p.K)
        pi = interleaver(K, p.kind)
        pos = min(p.pos, K - 1)
        u = np.zeros(K, int)
        u[pos] = 1
        if p.pat == "Two 1s":
            u[min(pos + p.sep, K - 1)] = 1
        rec = p.enc.startswith("Recursive")
        if rec:
            _, par1 = RSC.encode(u[None], terminate=False)
            par1 = par1[0]
            _, par2 = RSC.encode(u[pi][None], terminate=False)
            par2 = par2[0]
        else:
            par1, par2 = ff_parity(u), ff_parity(u[pi])
        ui = u[pi]
        ones2 = np.flatnonzero(ui)
        sep2 = int(ones2[-1] - ones2[0]) if len(ones2) == 2 else 0
        # windows of 40 steps around the action
        w0 = max(0, min(pos - 3, K - 40))
        w2 = max(0, min(int(ones2[0]) - 3, K - 40))
        for key, uu, pp, w in (("enc1", u, par1, w0), ("enc2", ui, par2, w2)):
            pl = self.plot(key)
            pl.set_xlim(w - 0.5, w + 39.5)
            t = np.arange(w, w + 40)
            on = t[uu[w:w + 40] == 1]
            pl.stems("u", on, np.full(len(on), 2.4), color=ORANGE, base=1.6, size=9)
            pp_on = t[pp[w:w + 40] == 1]
            pl.stems("p", pp_on, np.full(len(pp_on), 0.9), color=NAVY, base=0.0, size=7)
            pl.set_yticks([(0.45, "parity"), (2.0, "input")])
        pp_ = self.plot("pi")
        pp_.set_xlim(-0.5, K - 0.5)
        pp_.set_ylim(-0.5, K - 0.5)
        pp_.scatter("all", np.arange(K), pi, color=GRAY, size=4 if K <= 256 else 2, alpha=0.6)
        src = np.flatnonzero(u)
        dst = np.array([int(np.flatnonzero(pi == s_)[0]) for s_ in src])   # interleaved index of s_
        pp_.scatter("hl", dst, src, color=RED, size=12, z=5)
        segments(pp_, "hlv", dst, np.full(len(dst), -0.5), dst, src, color=RED, width=1.0, style=":")
        pp_.set_labels(x="interleaved position k", y="original input position π(k)")
        cw = turbo(K, p.kind).flatten(turbo(K, p.kind).encode(u[None]))[0] if rec else None
        self.readout(w1=int(par1.sum()), sep2=sep2, w2=int(par2.sum()),
                     wt=int(cw.sum()) if rec else int(u.sum() + par1.sum() + par2.sum()))

    def story(self, p):
        r = self.r
        rec = p.enc.startswith("Recursive")
        s = []
        if rec:
            s.append("<p>The LTE constituent code feeds its output back into the register "
                     "(feedback 1 + D² + D³, octal 13). A single 1 therefore never dies out: the "
                     "register cycles through its states for ever and the parity keeps coming. Only "
                     "inputs divisible by the feedback polynomial return the register to zero; the "
                     "lightest are <b>two 1s a multiple of 7 apart</b> (7 is the period of 1 + D² + "
                     "D³).</p>")
        else:
            s.append("<p>A feed-forward encoder forgets a lone 1 after K − 1 steps: a single input "
                     "1 produces a short, light codeword, whatever the interleaver does. In a turbo "
                     "code that would leave codewords of weight about 1 + 3 + 3 = 7: a high error "
                     "floor.</p>")
        s.append(f"<p>The interleaver scrambles the input before encoder 2. Your two 1s, "
                 f"{v(p.sep, 'd')} apart, land {v(r.get('sep2', 0), 'd')} apart for encoder 2 "
                 f"(right, red). The whole codeword weighs {v(r.get('wt', 0), 'd')}: low weight needs "
                 "a pattern that is short for <i>both</i> encoders, and a good interleaver makes that "
                 "rare.</p>")
        return "<h3>Recursion plus scrambling</h3>" + "".join(s) + keybox(
            "Recursive constituents make every low-weight input a weight-2 (or heavier) pattern; "
            "the interleaver makes sure no such pattern is short for both encoders.")


# =============================================================================== 2. BCJR
KB = 6


class BcjrCheck(Experiment):
    title = "BCJR: log-MAP against max-log"
    blurb = "The forward–backward algorithm checked against brute force over every codeword."
    book = "sec:ch15:soft"
    controls = [
        Slider("ebn0", "Eb/N0", -3.0, 8.0, 1.0, step=0.1, unit="dB"),
        Button("again", "New message and noise"),
    ]
    plots = [
        Plot("llr", f"A-posteriori LLRs of the {KB} message bits", x="bit k",
             y="LLR  log P(0)/P(1)", xlim=(-0.6, KB - 0.4), legend="tl", legend_cols=3),
        Plot("mx", "The term max-log drops: ln(1 + e^−|a−b|)",
             x="|a − b|", y="correction term", xlim=(0, 6), ylim=(0, 0.75), legend=None),
    ]
    layout = [["llr", "mx"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("err", "|log-MAP − brute force|", "", "sci", good=lambda x: x < 1e-8),
        Readout("over", "Max-log overconfidence", "%", ".0f"),
        Readout("agree", "Hard decisions agree", "", None),
        Readout("words", "Codewords summed", "", "int"),
    ]
    challenges = [
        Challenge("Make max-log-MAP overconfident by more than 15 % on average.",
                  lambda s: s.r.over > 15, hint="Low SNR, and try a few noise draws."),
        Challenge("Find a noise draw where max-log and log-MAP disagree on a bit's sign.",
                  lambda s: s.r.agree == "no", hint="Very low SNR; press New a few times."),
        Challenge("At high SNR, get max-log within 5 % of the exact LLRs.",
                  lambda s: abs(s.r.over) < 5 and s.p.ebn0 >= 3),
    ]

    def setup(self):
        self.seed = 11

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        u = rng.integers(0, 2, (1, KB))
        s_, p_ = RSC.encode(u)
        sig = ft.bpsk_sigma(p.ebn0, 0.5)
        noise = rng.standard_normal((2,) + s_.shape)
        Ls = 2 * ((1 - 2.0 * s_) + sig * noise[0]) / sig ** 2
        Lp = 2 * ((1 - 2.0 * p_) + sig * noise[1]) / sig ** 2
        Lm = tb.bcjr(RSC, Ls, Lp)[0, :KB]
        Lx = tb.bcjr(RSC, Ls, Lp, maxlog=True)[0, :KB]
        allu = np.array(list(itertools.product([0, 1], repeat=KB)))
        cs, cp = RSC.encode(allu)
        metric = 0.5 * ((1 - 2.0 * cs) @ Ls[0] + (1 - 2.0 * cp) @ Lp[0])
        Lb = np.array([np.logaddexp.reduce(metric[allu[:, k] == 0]) -
                       np.logaddexp.reduce(metric[allu[:, k] == 1]) for k in range(KB)])
        pl = self.plot("llr")
        k = np.arange(KB)
        pl.bars("b", k - 0.27, Lb, width=0.25, color=GRAY)
        pl.bars("m", k, Lm, width=0.25, color=NAVY)
        pl.bars("x", k + 0.27, Lx, width=0.25, color=RED)
        for key, col, lab in (("lb", GRAY, "brute force"), ("lm", NAVY, "log-MAP"),
                              ("lx", RED, "max-log-MAP")):
            pl.scatter(key, [-9], [0], color=col, size=10, symbol="s", name=lab)
        top = max(3.0, float(np.max(np.abs(np.r_[Lb, Lx]))) * 1.35)
        pl.set_ylim(-top, top)
        pl.hline("z", 0, color=pl.theme.text, style="-", width=0.8)
        for i in range(KB):
            pl.text(f"u{i}", i, -top * 0.92, f"sent {u[0, i]}", anchor=(0.5, 0.5), size=8.5,
                    color=GREEN if (Lm[i] < 0) == bool(u[0, i]) else RED)
        pm = self.plot("mx")
        pm.vb.setRange(xRange=(0, 6), yRange=(0, 0.75), padding=0)
        d = np.linspace(0, 6, 200)
        pm.line("c", d, np.log1p(np.exp(-d)), color=NAVY, width=2.4, fill=0)
        pm.text("t", 1.2, 0.6, "largest (ln 2 = 0.69) when the two\npaths are equally likely",
                anchor=(0, 0.5), size=8.5, color=GRAY)
        over = float(100 * (np.mean(np.abs(Lx)) / max(np.mean(np.abs(Lb)), 1e-12) - 1))
        self.readout(err=float(np.max(np.abs(Lm - Lb))), over=over,
                     agree="yes" if np.array_equal(Lm < 0, Lx < 0) else "no", words=2 ** KB)

    def story(self, p):
        r = self.r
        err_s = f"{r.get('err', 0):.0e}"
        s = (f"<p>The a-posteriori LLR of a bit is a sum over every codeword in which it is 0 "
             f"against every codeword in which it is 1. For {KB} message bits that is "
             f"{v(2 ** KB, 'd')} codewords (grey bars, brute force); for an LTE block of 6144 bits "
             "it would be 2⁶¹⁴⁴. The <b>BCJR</b> algorithm gets exactly the same numbers with one "
             "forward and one backward pass over the trellis: the error here is "
             f"{v(err_s)}.</p>"
             "<p>Working with logarithms turns the sums into the operation max*(a, b) = "
             "max(a, b) + ln(1 + e^−|a−b|). <b>Max-log-MAP</b> drops the correction term (right): "
             "cheaper, same decisions as a Viterbi decoder, but its reliabilities are off, "
             f"here by {v(r.get('over', 0), '+.0f', '%')} on average, mostly too large at low SNR. Fed back as a-priori "
             "information inside a turbo decoder, that overconfidence costs a few tenths of a "
             "dB, which a scaling factor of about 0.7 mostly recovers.</p>")
        return "<h3>Exact, and linear in length</h3>" + s + keybox(
            "BCJR (Bahl, Cocke, Jelinek, Raviv, 1974) is the soft-in soft-out engine of turbo "
            "decoders, turbo equalisers and every iterative receiver.")


# =============================================================================== 3. iterations
DECODERS = {"log-MAP": (False, 1.0), "max-log-MAP": (True, 1.0), "max-log-MAP, scaled 0.7": (True, 0.7)}
SHOW_IT = [1, 2, 4, 8]


class TurboIterations(Experiment):
    title = "Turbo decoding, iteration by iteration"
    blurb = "Each pass of extrinsic information moves the waterfall left, with diminishing returns."
    book = "sec:ch15:turbo"
    heavy = True
    controls = [
        Choice("K", "Block length K", ["40", "256", "1024"], "1024"),
        Choice("dec", "Constituent decoder", list(DECODERS), "log-MAP", style="menu"),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [BERPlot("ber", "Bit error rate after 1, 2, 4 and 8 iterations (LTE turbo code, rate ⅓)",
                     x="Eb/N0 (dB)", xlim=(-1, 3.5), ylim=(1e-5, 0.3), legend="tr")]
    readouts = [
        Readout("x1", "10⁻³ after 1 iteration", "dB", ".2f"),
        Readout("x8", "10⁻³ after 8 iterations", "dB", ".2f"),
        Readout("lim", "BPSK limit for this rate", "dB", ".2f"),
        Readout("bits", "Bits simulated", "", "int"),
    ]
    challenges = [
        Challenge("Measure an iteration gain of at least 1 dB at 10⁻³ (1 against 8 iterations).",
                  lambda s: s.r.x1 is not None and s.r.x8 is not None and s.r.x1 - s.r.x8 >= 1.0),
        Challenge("Show that block length matters: K = 40 needs at least 1 dB more than K = 1024 "
                  "after 8 iterations.", lambda s: "40" in s.exp.memo and "1024" in s.exp.memo
                  and s.exp.memo["40"] - s.exp.memo["1024"] >= 1.0),
        Challenge("Measure what unscaled max-log costs at K = 1024: at least 0.15 dB against log-MAP.",
                  lambda s: ("log-MAP" in s.exp.dmemo and "max-log-MAP" in s.exp.dmemo
                             and s.exp.dmemo["max-log-MAP"] - s.exp.dmemo["log-MAP"] >= 0.15)),
    ]

    def setup(self):
        self.run_id = 0
        self.memo, self.dmemo = {}, {}

    def on_rerun(self, p):
        self.run_id += 1

    def ebs(self, K):
        if self.quick:
            return [0.5, 1.5]
        return [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0] if K >= 1024 else \
            ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5] if K >= 256 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    def update(self, p):
        K = int(p.K)
        tc = turbo(K)
        self.curves = {i: ([], []) for i in SHOW_IT}
        self.bits = 0
        pb = self.plot("ber")
        eb = np.linspace(-1, 3.5, 200)
        pb.theory("unc", eb, ft.qfunc(np.sqrt(2 * 10 ** (eb / 10))), color=GRAY, width=1.4,
                  name="uncoded BPSK")
        lim = bpsk_limit_db(round(tc.rate, 4))
        pb.vline("lim", lim, color=RED, style=":", label=f"limit {lim:.2f} dB", label_pos=0.08)
        self.readout(x1=None, x8=None, lim=lim, bits=0)

    def background(self, p):
        K = int(p.K)
        tc = turbo(K)
        ml, sc = DECODERS[p.dec]
        rng = np.random.default_rng(K + self.run_id)
        B = 10 if self.quick else (16 if K >= 1024 else 60)
        nb = 1 if self.quick else 2
        for eb in self.ebs(K):
            errs = np.zeros(8)
            n = 0
            for _ in range(nb):
                u = rng.integers(0, 2, (B, K))
                L = ft.bpsk_llr(tc.flatten(tc.encode(u)), eb, tc.rate, rng)
                hist = tc.decode(L, iters=8, maxlog=ml, scale=sc, record=True)
                errs += [np.sum(h != u) for h in hist]
                n += u.size
            yield dict(eb=eb, ber=errs / n, n=n)
            if errs[0] == 0:
                break

    def progress(self, p, it_):
        self.bits += it_["n"]
        cols = {1: RED, 2: ORANGE, 4: GREEN, 8: NAVY}
        pb = self.plot("ber")
        for i in SHOW_IT:
            b = it_["ber"][i - 1]
            if b > 0:
                self.curves[i][0].append(it_["eb"])
                self.curves[i][1].append(b)
            if self.curves[i][0]:
                pb.sim(f"i{i}", *self.curves[i], color=cols[i],
                       name=f"{i} iteration{'s' if i > 1 else ''}")
        x1 = interp_cross(*self.curves[1], 1e-3)
        x8 = interp_cross(*self.curves[8], 1e-3)
        if x8 is not None:
            if p.dec == "log-MAP":
                self.memo[p.K] = x8
            if p.K == "1024":
                self.dmemo[p.dec] = x8
        self.readout(x1=x1, x8=x8, bits=self.bits)

    def story(self, p):
        r = self.r
        s = ("<p>The decoder runs BCJR on encoder 1, subtracts what it was told to obtain the "
             "<b>extrinsic</b> information (what the parity of encoder 1 alone adds), interleaves "
             "it and hands it to the decoder of encoder 2 as a-priori knowledge, and back again. "
             "Each half-iteration, each decoder starts from a better guess.</p>")
        if r.get("x1") is not None and r.get("x8") is not None:
            s += (f"<p>Measured: 10⁻³ needs {v(r['x1'], '.2f', 'dB')} after one iteration and "
                  f"{v(r['x8'], '.2f', 'dB')} after eight, {v(r['x8'] - r.get('lim', 0), '.1f', 'dB')} "
                  "from the limit for this rate.</p>")
        else:
            s += "<p>The curves fill in as the simulation runs (about 15 s for K = 1024).</p>"
        s += ("<p>Turbo gain grows with the block length: a long interleaver decorrelates the two "
              "decoders' errors. At K = 40, the smallest LTE block, the interleaver has no room to "
              "work.</p>")
        return "<h3>The turbo principle</h3>" + s + keybox(
            "Never feed a decoder back its own information: exchange only extrinsic LLRs. That one "
            "rule makes iterative decoding converge.")


# =============================================================================== 4. EXIT chart
IA_GRID = np.linspace(0, 1, 21)


@lru_cache(maxsize=64)
def exit_T(eb):
    """Measured transfer curve, smoothed by a low-order polynomial fit (Monte Carlo noise of
    a few hundredths of a bit would otherwise make the tunnel flicker)."""
    raw = ft.exit_curve(RSC, float(ft.bpsk_sigma(eb, 1 / 3)), IA_GRID, K=3000, reps=6,
                        rng=np.random.default_rng(int(round(eb * 10)) + 50))
    c = np.polyfit(IA_GRID, raw, 4)
    T = np.clip(np.polyval(c, IA_GRID), 0, 1)
    T[-1] = 1.0
    return np.maximum.accumulate(T)


def staircase(T, n=40):
    """Predicted trajectory between T (decoder 1) and its mirror (decoder 2)."""
    f = lambda x: float(np.interp(x, IA_GRID, np.maximum.accumulate(T)))
    X, Y = [0.0], [0.0]
    x = 0.0
    steps = 0
    for _ in range(n):
        y = f(x)
        X += [x]; Y += [y]
        x2 = f(y)
        X += [x2]; Y += [y]
        steps += 1
        if x2 > 0.99 or abs(x2 - x) < 1e-4:
            x = x2
            break
        x = x2
    return np.array(X), np.array(Y), steps, x


class ExitChart(Experiment):
    title = "EXIT chart: watch the tunnel open"
    blurb = "Predict iterative decoding from two curves, then compare with a real decoder."
    book = "sec:ch15:turbo"
    heavy = True
    controls = [
        Slider("ebn0", "Eb/N0", -1.0, 1.5, -0.5, step=0.1, unit="dB"),
        Toggle("real", "Also run a real decoder (K = 2000)", True),
    ]
    plots = [
        Plot("exit", "EXIT chart: decoder 1 (navy) and decoder 2 mirrored (red)",
             x="I_A1 = I_E2 (mutual information)", y="I_E1 = I_A2", xlim=(0, 1), ylim=(0, 1.02),
             legend="br"),
        Plot("tc", "One decoder at several Eb/N0", x="a-priori information I_A",
             y="extrinsic information I_E", xlim=(0, 1), ylim=(0, 1.02), legend="tl"),
    ]
    layout = [["exit", "tc"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("ie0", "I_E from the channel alone", "bits", ".3f"),
        Readout("gap", "Tunnel width (min)", "bits", ".3f", good=lambda x: x > 0),
        Readout("pred", "Predicted iterations to 0.99", "", None),
        Readout("meas", "Real decoder reaches", "bits", ".3f"),
    ]
    challenges = [
        Challenge("Find the lowest Eb/N0 at which the tunnel is open (the code's threshold).",
                  lambda s: s.r.gap is not None and s.r.gap > 0 and s.p.ebn0 <= s.exp.thr + 0.15
                  and s.exp.thr < 9),
        Challenge("Watch the real decoder get stuck below 0.6 bits where the tunnel is closed.",
                  lambda s: s.r.gap is not None and s.r.gap < 0 and s.r.meas is not None
                  and s.r.meas < 0.6 and s.p.real),
        Challenge("Reach 0.99 bits in at most 4 predicted iterations.",
                  lambda s: s.exp.steps is not None and s.exp.steps <= 4 and s.exp.final > 0.99),
    ]

    def setup(self):
        self.thr = 99.0
        self.steps, self.final = None, 0.0
        self.open_at = {}

    def update(self, p):
        pe = self.plot("exit")
        pe.line("diag", [0, 1], [0, 1], color=GRAY, style=":", width=1.0)
        self.steps, self.final = None, 0.0
        self.readout(ie0=None, gap=None, pred="…", meas=None)
        pt = self.plot("tc")
        pt.line("diag", [0, 1], [0, 1], color=GRAY, style=":", width=1.0)

    def background(self, p):
        eb = round(p.ebn0, 1)
        yield dict(kind="curve", eb=eb, T=exit_T(eb))
        for e2 in (-1.0, 0.0, 1.0):
            yield dict(kind="ref", eb=e2, T=exit_T(e2))
        if p.real and not self.quick:
            K = 2000
            tc = tb.TurboCode(K, np.random.default_rng(9).permutation(K))
            rng = np.random.default_rng(4)
            u = rng.integers(0, 2, (2, K))
            L = ft.bpsk_llr(tc.flatten(tc.encode(u)), eb, tc.rate, rng)
            _, traj = tc.decode(L, iters=10, Linfo=u)
            yield dict(kind="traj", traj=traj)

    def progress(self, p, it_):
        if it_["kind"] == "curve":
            T = it_["T"]
            pe = self.plot("exit")
            pe.line("d1", IA_GRID, T, color=NAVY, width=2.6, name="decoder 1")
            pe.line("d2", T, IA_GRID, color=RED, width=2.6, name="decoder 2 (mirrored)")
            X, Y, steps, fin = staircase(T)
            pe.line("st", X, Y, color=ORANGE, width=1.6, name="predicted staircase")
            gap = float(np.min(np.maximum.accumulate(T)[:-1] - IA_GRID[:-1]))
            self.open_at[it_["eb"]] = gap > 0
            opens = [e for e, o in self.open_at.items() if o]
            closed = [e for e, o in self.open_at.items() if not o]
            # the threshold estimate: lowest open point with a closed point at most 0.1 dB below
            cand = [e for e in opens if any(abs(e - 0.1 - c) < 1e-6 for c in closed)]
            self.thr = min(cand) if cand else 99.0
            self.steps, self.final = steps, fin
            self.readout(ie0=float(T[0]), gap=gap,
                         pred=(f"{steps}" if fin > 0.99 else f"stuck at {fin:.2f}"))
        elif it_["kind"] == "ref":
            cols = {-1.0: GRAY, 0.0: PURPLE, 1.0: GREEN}
            self.plot("tc").line(f"r{it_['eb']}", IA_GRID, it_["T"], color=cols[it_["eb"]],
                                 width=1.6, name=f"{it_['eb']:+.1f} dB")
        else:
            traj = it_["traj"]
            X, Y = [0.0], [0.0]
            for j, (ia, ie) in enumerate(traj):
                if j % 2 == 0:
                    X.append(X[-1]); Y.append(ie)
                else:
                    X.append(ie); Y.append(Y[-1])
            pe = self.plot("exit")
            pe.line("tr", X, Y, color=pe.theme.text, width=1.0, name="real decoder, K = 2000")
            pe.scatter("trp", X[1:], Y[1:], color=pe.theme.text, size=4)
            self.readout(meas=float(max(max(X), max(Y))))

    def story(self, p):
        r = self.r
        s = ("<p>Feed one constituent decoder a-priori LLRs that carry I_A bits of information "
             "about the message, and measure how much its extrinsic output carries: I_E. That "
             "<b>transfer curve</b> (navy) characterises the decoder at this Eb/N0. The second "
             "decoder is the same, drawn with the axes swapped (red).</p>"
             "<p>Iterative decoding bounces between the two curves: up to decoder 1, across to "
             "decoder 2, up again (orange staircase). It reaches the top corner, perfect "
             "knowledge, only if the <b>tunnel</b> between the curves is open.</p>")
        gap = r.get("gap")
        if gap is not None:
            if gap > 0:
                s += ("<p>" + good(f"Open here (narrowest point {gap:.3f} bits).") +
                      f" Predicted: {r.get('pred')} iterations. Near the threshold the steps get "
                      "tiny, which is why decoding close to the limit needs many iterations.</p>")
            else:
                s += ("<p>" + bad("Closed: the curves cross,") + " and the staircase stops at "
                      "the crossing, however many iterations you allow. The waterfall's left edge, "
                      "read off a picture without a single BER simulation.</p>")
        return "<h3>Two curves predict everything</h3>" + s + keybox(
            "EXIT charts (ten Brink, 2001) turned code design into curve fitting: match the shapes "
            "of the two transfer curves and the tunnel opens close to capacity.")


# =============================================================================== 5. DE on the BEC
ENSEMBLES = {"(3,6) regular": ([0, 0, 1], [0] * 5 + [1]),
             "(4,8) regular": ([0, 0, 0, 1], [0] * 7 + [1]),
             "Irregular: λ = 0.3x + 0.3x² + 0.4x⁷, ρ = x⁶": ([0, 0.3, 0.3, 0, 0, 0, 0, 0.4], [0] * 6 + [1])}


def poly(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))


def design_rate(lam, rho):
    return 1 - sum(c / (i + 1) for i, c in enumerate(rho)) / sum(c / (i + 1) for i, c in enumerate(lam))


class DensityEvolution(Experiment):
    title = "Density evolution on the erasure channel"
    blurb = "One number per iteration predicts an infinitely long LDPC code exactly."
    book = "sec:ch15:de"
    controls = [
        Choice("ens", "Ensemble", list(ENSEMBLES), "(3,6) regular", style="menu"),
        Slider("eps", "Channel erasure probability ε", 0.30, 0.50, 0.45, step=0.001),
    ]
    plots = [
        Plot("de", "One iteration of BP on the erasure channel",
             x="erasure probability now, x_ℓ", y="after one more iteration, x_ℓ₊₁", xlim=(0, 0.5),
             ylim=(0, 0.5), legend="tl"),
        Plot("traj", "Erasure probability over iterations", x="iteration",
             y="erased messages", logy=True, xlim=(0, 300), ylim=(1e-10, 1), legend=None),
    ]
    layout = [["de", "traj"]]
    readouts = [
        Readout("thr", "BP threshold ε*", "", ".4f"),
        Readout("rate", "Design rate", "", ".3f"),
        Readout("frac", "Threshold / capacity limit", "%", "%"),
        Readout("its", "Iterations to 10⁻⁹", "", None),
    ]
    challenges = [
        Challenge("Find the (3,6) threshold from below: decode with ε no more than 0.002 under it.",
                  lambda s: s.p.ens.startswith("(3,6)") and s.exp.ok and s.r.thr - 0.002 <= s.p.eps < s.r.thr),
        Challenge("Find an ensemble that decodes at ε = 0.45.",
                  lambda s: abs(s.p.eps - 0.45) < 0.0006 and s.exp.ok),
        Challenge("See a long plateau: converge, but only after more than 100 iterations.",
                  lambda s: s.exp.ok and s.exp.n_it > 100),
    ]

    def setup(self):
        self.ok, self.n_it = False, 0

    def update(self, p):
        lam, rho = ENSEMBLES[p.ens]
        thr = tb.bec_threshold(lam, rho)
        R = design_rate(lam, rho)
        xs = np.linspace(0, 0.5, 400)
        f = p.eps * poly(lam, 1 - poly(rho, 1 - xs))
        pd = self.plot("de")
        pd.line("f", xs, f, color=NAVY, width=2.4, name="one iteration of BP")
        pd.line("d", [0, 0.5], [0, 0.5], color=GRAY, width=1.0, name="no progress (x = x)")
        x = p.eps
        X, Y, traj = [x], [x], [x]
        for _ in range(2000):
            fx = p.eps * poly(lam, 1 - poly(rho, 1 - x))
            X += [x, fx]
            Y += [fx, fx]
            x = fx
            traj.append(x)
            if x < 1e-12:
                break
        X, Y = np.array(X[1:]), np.array(Y[:-1])
        n = min(len(X), 400)
        pd.line("st", X[:n], Y[:n], color=RED, width=1.2, name="decoder trajectory")
        tr = np.maximum(np.array(traj), 1e-12)
        pt = self.plot("traj")
        pt.line("t", np.arange(len(tr)), tr, color=NAVY, width=2.2)
        pt.set_xlim(0, max(60, min(len(tr) + 10, 300 if tr[-1] > 1e-9 else 2000)))
        pt.hline("t9", 1e-9, color=GREEN, style=":")
        self.ok = tr[-1] < 1e-9
        self.n_it = int(np.argmax(tr < 1e-9)) if self.ok else 0
        self.readout(thr=thr, rate=R, frac=thr / (1 - R),
                     its=str(self.n_it) if self.ok else f"stuck at {tr[-1]:.2f}")

    def story(self, p):
        r = self.r
        lam, rho = ENSEMBLES[p.ens]
        s = ("<p>On the erasure channel a BP message is either known or erased, so the whole "
             "decoder is described by one number: the probability x that a message is still "
             "erased. One iteration maps it to ε·λ(1 − ρ(1 − x)) (navy curve), where λ and ρ "
             "describe how many edges sit on nodes of each degree.</p>")
        if r.get("its", "").startswith("stuck"):
            s += ("<p>" + bad(f"At ε = {p.eps:.3f} the curve touches the diagonal:") + " the "
                  f"staircase is trapped at a fixed point (x = {r.get('its', '')[9:]}) and a fraction of "
                  f"the bits is never recovered. The threshold of this ensemble is ε* = {r.get('thr', 0):.4f}.</p>")
        else:
            s += ("<p>" + good(f"Below the threshold ε* = {r.get('thr', 0):.4f}:") + " the curve "
                  "stays under the diagonal and the staircase reaches zero. Close to ε* it must "
                  "squeeze through a narrow gap, which shows up as a long plateau on the right.</p>")
        s += (f"<p>The capacity limit for rate {v(r.get('rate', 0.5), '.3f')} is ε = "
              f"{v(1 - r.get('rate', 0.5), '.3f')}: this ensemble reaches "
              f"{v(100 * r.get('frac', 0), '.1f', '%')} of it. Irregular degrees (some bits in many "
              "checks) close most of the remaining gap.</p>")
        return "<h3>Thresholds without simulation</h3>" + s + keybox(
            "Density evolution (Richardson and Urbanke, 2001) gives the exact threshold of an "
            "infinitely long code: the design tool behind every capacity-approaching LDPC code.")


# =============================================================================== 6. spatial coupling
@lru_cache(maxsize=32)
def coupled(eps, L, w):
    prof, snaps = tb.de_bec_coupled(eps, 3, 6, L, w=w, iters=3000, record_every=1)
    return prof, np.array(snaps)


def coupled_rate(L, w=3):
    """Design rate of the terminated (3, 6, L, w) chain (old lab formula, w = 3)."""
    return 0.5 - 0.5 * (3 + 1 - 2 * sum((i / 3) ** 6 for i in range(4))) / L


class Coupling(Experiment):
    title = "Spatial coupling: the decoding wave"
    blurb = "Chain codes together with a known boundary and decoding sweeps in like a wave."
    book = "sec:ch15:sc"
    animate = True
    fps = 12
    controls = [
        Slider("eps", "Channel erasure probability ε", 0.40, 0.50, 0.46, step=0.001),
        IntSlider("L", "Chain length L", 8, 96, 48, step=4, unit="positions"),
        IntSlider("t", "Show iteration", 0, 600, 0, step=1),
    ]
    plots = [
        Plot("wave", "Erasure probability along the chain", x="position in the chain",
             y="erased messages", ylim=(-0.01, 0.5), legend="tr"),
        Plot("prog", "Worst position against iteration", x="iteration", y="max erasure probability",
             logy=True, ylim=(1e-9, 1), legend=None),
    ]
    layout = [["wave"], ["prog"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("unc", "Uncoupled (3,6) threshold", "", ".4f"),
        Readout("map", "MAP threshold (the target)", "", ".4f"),
        Readout("done", "Wave finishes after", "", None),
        Readout("loss", "Rate loss of the chain", "%", ".1f", good=lambda x: x < 5),
    ]
    challenges = [
        Challenge("Decode at ε = 0.47 or more, where the uncoupled (3,6) code is hopeless.",
                  lambda s: s.p.eps >= 0.47 and s.exp.ok),
        Challenge("Find an ε where even the coupled chain stalls.",
                  lambda s: not s.exp.ok),
        Challenge("Keep the rate loss below 5 % and still decode at ε ≥ 0.47.",
                  lambda s: s.r.loss < 5 and s.p.eps >= 0.47 and s.exp.ok),
    ]

    def setup(self):
        self.ok = True

    def tick(self, p):
        prof, snaps = coupled(round(p.eps, 3), p.L, 3)
        q = st.Params(p)
        q.t = (p.t + max(1, len(snaps) // 60)) % (len(snaps) + 1)
        self.set_control("t", q.t)
        self.update(q)

    def update(self, p):
        prof, snaps = coupled(round(p.eps, 3), p.L, 3)
        n = len(snaps)
        self.ok = prof.max() < 1e-6
        t = min(p.t, n - 1)
        pw = self.plot("wave")
        pw.set_xlim(-0.5, p.L - 0.5)
        x = np.arange(p.L)
        for k, frac in enumerate((0.15, 0.35, 0.6)):
            j = int(frac * (n - 1))
            pw.line(f"g{k}", x, snaps[j], color=GRAY, width=1.3,
                    name="earlier snapshots" if k == 0 else None)
        pw.line("now", x, snaps[t], color=NAVY, width=2.8, name=f"iteration {t}")
        unc = tb.de_bec_regular(p.eps, 3, 6, iters=max(t, 1))[-1]
        pw.hline("unc", unc, color=RED, style="--", label="uncoupled (3,6) code at this iteration",
                 label_pos=0.3)
        pp = self.plot("prog")
        mx = np.maximum(snaps.max(axis=1), 1e-12)
        pp.set_xlim(0, n)
        pp.line("mx", np.arange(n), mx, color=NAVY, width=2.0)
        pp.vline("now", t, color=ORANGE, style=":")
        rl = 100 * (0.5 - coupled_rate(p.L)) / 0.5
        self.readout(unc=0.4294, map=0.4881, loss=rl,
                     done=(f"{int(np.argmax(mx < 1e-6))} iterations" if self.ok else "never"))

    def story(self, p):
        r = self.r
        s = ("<p>Take L copies of a (3,6) LDPC code in a row and let the edges of each spread "
             "over its neighbours. At the two ends the positions see fewer unknowns (the chain is "
             "terminated with known bits), so they decode first; their neighbours then see a "
             "little more help, and a <b>decoding wave</b> sweeps in from both ends (press Play).</p>")
        if self.ok:
            s += (f"<p>{good('The wave reaches the middle:')} the chain decodes at ε = "
                  f"{v(p.eps, '.3f')}" + (", where the uncoupled code (red line) is stuck. "
                                          if p.eps > 0.4294 else ". ") +
                  "Coupling lifts the BP threshold from 0.4294 to the MAP threshold 0.4881 of "
                  "the underlying code: <b>threshold saturation</b> (Kudekar, Richardson and "
                  "Urbanke, 2011).</p>")
        else:
            s += f"<p>{bad('Above the MAP threshold the wave stalls')} and the middle stays erased.</p>"
        s += (f"<p>The price is a rate loss of {v(r.get('loss', 0), '.1f', '%')} for the terminated "
              "chain, which shrinks as 1/L, and a decoding delay that grows with L.</p>")
        return "<h3>Threshold saturation</h3>" + s + keybox(
            "Spatially coupled LDPC codes reach capacity with plain BP decoding; they appear in "
            "high-throughput optical transport.")


# =============================================================================== 7. error floor
class ErrorFloor(Experiment):
    title = "The error floor and the interleaver"
    blurb = "A handful of light codewords sets the floor; the interleaver decides how light."
    book = "sec:ch15:turbo"
    heavy = True
    controls = [
        Choice("K", "Block length K", ["256", "1024"], "256"),
        Choice("kind", "Interleaver", ["QPP (LTE)", "Random", "S-random"], "QPP (LTE)"),
    ]
    plots = [
        Plot("hist", "Codeword weights from weight-2 inputs",
             x="codeword weight", y="number of codewords", xlim=(9.5, 60.5), legend=None),
        BERPlot("floor", "Error-floor estimate from the lightest codewords",
                x="Eb/N0 (dB)", xlim=(0, 4), ylim=(1e-11, 1e-2), legend="bl"),
    ]
    layout = [["hist", "floor"]]
    readouts = [
        Readout("dmin", "Lightest codeword found", "", "int"),
        Readout("mult", "How many at that weight", "", "int"),
        Readout("fl2", "Floor at 2 dB", "", "sci", good=lambda x: x < 1e-7),
        Readout("n", "Weight-2 inputs checked", "", "int"),
    ]
    challenges = [
        Challenge("Find the interleaver with the lowest floor at K = 1024 (below 10⁻⁸ at 2 dB).",
                  lambda s: s.p.K == "1024" and s.r.fl2 is not None and s.r.fl2 < 1e-8),
        Challenge("Show that a random interleaver lets through a codeword lighter than 20.",
                  lambda s: s.p.kind == "Random" and s.r.dmin is not None and s.r.dmin < 20),
        Challenge("Compare at K = 1024: the QPP interleaver's lightest word is at least twice as "
                  "heavy as the random interleaver's.", lambda s: s.exp.cmp.get("QPP (LTE)", 0)
                  >= 2 * s.exp.cmp.get("Random", 99)),
    ]

    def setup(self):
        self.cmp = {}

    def update(self, p):
        eb = np.linspace(0, 4, 160)
        pf = self.plot("floor")
        pf.theory("unc", eb, ft.qfunc(np.sqrt(2 * 10 ** (eb / 10))), color=GRAY, width=1.2,
                  name="uncoded BPSK (for scale)")
        pf.vline("v2", 2.0, color=GRAY, style=":")
        self.readout(dmin=None, mult=None, fl2=None, n=None)

    def background(self, p):
        K = int(p.K)
        tc = turbo(K, p.kind)
        yield dict(w=tuple(ft.weight2_codeword_weights(tc, 56 if not self.quick else 14)[0]))

    def progress(self, p, it_):
        K = int(p.K)
        tc = turbo(K, p.kind)
        w = np.array(it_["w"])
        dmin = int(w.min())
        ph = self.plot("hist")
        h = np.bincount(w, minlength=61)[:61]
        x = np.arange(len(h))
        ph.bars("b", x, h, width=0.8, colors=[RED if xi < dmin + 4 else NAVY for xi in x])
        ph.ylim = (0, max(5, int(h[:61].max() * 1.1) + 1))
        ph.vb.setRange(xRange=(9.5, 60.5), yRange=ph.ylim, padding=0)
        eb = np.linspace(0, 4, 160)
        low = w[w <= dmin + 6]
        e = 10 ** (eb / 10)
        floor = sum((2 / K) * ft.qfunc(np.sqrt(2 * tc.rate * d * e)) for d in low)
        cols = {"QPP (LTE)": NAVY, "Random": RED, "S-random": GREEN}
        self.plot("floor").theory("fl", eb, np.maximum(floor, 1e-300), color=cols[p.kind], width=2.4,
                                  name=f"{p.kind}, K = {K}")
        fl2 = float(sum((2 / K) * ft.qfunc(np.sqrt(2 * tc.rate * d * 10 ** 0.2)) for d in low))
        if K == 1024:
            self.cmp[p.kind] = dmin
        self.readout(dmin=dmin, mult=int(np.sum(w == dmin)), fl2=fl2, n=len(w))

    def story(self, p):
        r = self.r
        s = ("<p>Below the waterfall a turbo code's BER flattens into an <b>error floor</b>, set by "
             "its few lightest codewords. With recursive constituents those come from weight-2 "
             "inputs whose two 1s are a multiple of 7 apart for encoder 1 <i>and</i> after "
             "interleaving for encoder 2. Here every such input up to a separation of 56 is "
             "encoded and weighed (left).</p>")
        if r.get("dmin") is not None:
            fl_s = f"{r.get('fl2', 0):.1e}"
            s += (f"<p>The lightest codeword weighs {v(r['dmin'], 'd')} ({v(r.get('mult', 0), 'd')} "
                  f"of them). Each contributes about (2/K)·Q(√(2R·d·Eb/N0)) to the BER: at 2 dB the "
                  f"floor is {v(fl_s)}. It falls only "
                  "slowly with SNR, while the waterfall falls by orders of magnitude per dB.</p>")
        s += ("<p>Random interleavers occasionally map a short pattern onto another short pattern; "
              "S-random interleavers forbid it, and LTE's QPP is designed for spread too.</p>")
        return "<h3>Light codewords, low floors</h3>" + s + keybox(
            "Floors are predicted from weight spectra, not simulated: a 10⁻⁹ floor would take "
            "days of Monte Carlo.")


# =============================================================================== the lab
LAB = st.Lab(23, "Turbo Codes and EXIT Charts", chapter=15,
             chapter_title="Turbo, LDPC and Polar Codes",
             experiments=[Constituents, BcjrCheck, TurboIterations, ExitChart, DensityEvolution,
                          Coupling, ErrorFloor])

if __name__ == "__main__":
    st.run(LAB)
