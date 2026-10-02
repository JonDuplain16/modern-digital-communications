"""Lab 09 · LDPC and Polar Codes   (Chapter 15)

Run it:      python labs/lab09_ldpc_polar.py
Self-test:   python labs/lab09_ldpc_polar.py --selftest

5G NR carries user data with LDPC codes and control information with polar codes; Wi-Fi,
DVB-S2 and 10GBASE-T use LDPC too. Both come within about a decibel of the Shannon limit, and
both are decoded by passing soft messages along a graph. Seven experiments: the Tanner graph
and its cycles, belief propagation iteration by iteration, the check-node rule (sum-product
against min-sum and its corrections), iterations and early stopping, channel polarization,
successive cancellation against CRC-aided list decoding, and how close short codes get to
the finite-length limit.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import blockcodes as bc
from commlib import fectools as ft
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, GRAY, TEAL, GOLD)
from studio import v, keybox, good, bad
from _fecviz import Canvas, cells, segments, pale, ink


# =============================================================================== shared helpers
@lru_cache(maxsize=32)
def make_H(n, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "PEG":
        return cl.LDPCCode._peg(n, n // 2, 3, rng).astype(np.int8)
    return ft.random_regular_H(n, 3, 6, rng)


@lru_cache(maxsize=4)
def ldpc(n, seed=3):
    return cl.LDPCCode(n=n, rate=0.5, dv=3, seed=seed)


@lru_cache(maxsize=4)
def bp_for(n, seed=3):
    return ft.TannerBP(ldpc(n, seed).H)


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


def tanner_xy(m, n):
    """Positions: variables on top (y = 1), checks at the bottom (y = 0), both spread on [0, 1]."""
    xv = (np.arange(n) + 0.5) / n
    xc = (np.arange(m) + 0.5) / m
    return xv, xc


METHODS = {"Sum-product": "spa", "Min-sum": "ms", "Normalised min-sum (×0.8)": "nms",
           "Offset min-sum (−0.15)": "oms"}


# =============================================================================== 1. Tanner graph
class TannerGraph(Experiment):
    title = "Tanner graph and short cycles"
    blurb = "A sparse H drawn as a graph: bits above, checks below, and the 4-cycles that hurt BP."
    book = "sec:ch15:ldpc"
    controls = [
        IntSlider("n", "Code length n", 12, 192, 24, step=6, unit="bits"),
        Choice("kind", "Construction", ["Random sockets", "PEG"], "Random sockets"),
        Toggle("cyc", "Highlight 4-cycles", True),
        Button("again", "New random draw"),
    ]
    plots = [
        Canvas("graph", "Tanner graph: variable nodes (bits, top), check nodes (parities, bottom)",
               xlim=(-0.03, 1.03), ylim=(-0.12, 1.14)),
        ImagePlot("H", "Parity-check matrix H (dark = 1)", x="variable (bit)", y="check"),
    ]
    layout = [["graph"], ["H"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("four", "4-cycles", "", "int", good=lambda x: x == 0),
        Readout("girth", "Girth", "", None),
        Readout("rate", "Rate k/n", "", ".3f"),
        Readout("edges", "Edges", "", "int"),
    ]
    challenges = [
        Challenge("Find the shortest PEG code (n ≤ 42) with no 4-cycles at all.",
                  lambda s: s.p.kind == "PEG" and s.p.n <= 42 and s.r.four == 0),
        Challenge("Build a code whose shortest cycle has length 8 (girth 8).",
                  lambda s: s.r.girth == "8"),
        Challenge("Show that random construction keeps 4-cycles even at n ≥ 120 (more than 15 of "
                  "them).", lambda s: s.p.kind.startswith("Random") and s.p.n >= 120 and s.r.four > 15),
    ]

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        H = make_H(p.n, "PEG" if p.kind == "PEG" else "rand", self.seed)
        m, n = H.shape
        xv, xc = tanner_xy(m, n)
        ce, ev = np.nonzero(H)
        pg_ = self.plot("graph")
        # 4-cycle edges: pairs of checks sharing >= 2 variables
        Hm = H.astype(int)
        ov = Hm @ Hm.T
        np.fill_diagonal(ov, 0)
        bad_edge = np.zeros(len(ce), bool)
        if p.cyc:
            for a, b in zip(*np.nonzero(np.triu(ov) >= 2)):
                shared = np.flatnonzero(H[a] & H[b])
                bad_edge |= (np.isin(ce, [a, b]) & np.isin(ev, shared))
        w = 0.9 if n <= 60 else 0.5
        segments(pg_, "e", xv[ev[~bad_edge]], np.ones((~bad_edge).sum()), xc[ce[~bad_edge]],
                 np.zeros((~bad_edge).sum()), color=GRAY, width=w, alpha=0.7)
        if bad_edge.any():
            segments(pg_, "eb", xv[ev[bad_edge]], np.ones(bad_edge.sum()), xc[ce[bad_edge]],
                     np.zeros(bad_edge.sum()), color=RED, width=1.8, z=2)
        sz = 11 if n <= 48 else (7 if n <= 96 else 4)
        pg_.scatter("v", xv, np.ones(n), color=NAVY, size=sz, z=4)
        pg_.scatter("c", xc, np.zeros(m), color=TEAL, size=sz + 2, symbol="s", z=4)
        pg_.text("vl", 0.0, 1.09, f"{n} variable nodes, degree 3", anchor=(0, 0.5), size=9, color=NAVY)
        pg_.text("cl", 0.0, -0.08, f"{m} check nodes, degree about 6", anchor=(0, 0.5), size=9, color=TEAL)
        ph = self.plot("H")
        ph.xlim, ph.ylim = (0, n), (0, m)
        ph.vb.setRange(xRange=(0, n), yRange=(0, m), padding=0)
        ph.image("img", H[::-1].astype(float), x=(0, n), y=(0, m),
                 cmap=((0, pale(ph)), (1, NAVY)), levels=(0, 1))
        self.H = H
        k = n - bc.gf2_rank(H)
        self.readout(four=ft.count_4cycles(H), girth="…", rate=k / n, edges=len(ce))

    def background(self, p):
        yield ft.tanner_girth(self.H, max_len=12)

    def progress(self, p, g):
        self.readout(girth=str(g) if g else "> 12")

    def story(self, p):
        r = self.r
        n = p.n
        s = ("<p>An LDPC code is a sparse parity-check matrix H (bottom). Drawn as a graph, each "
             "bit is a node on top, each check a square below, and every 1 in H an edge: the "
             "<b>Tanner graph</b>. Every bit is in 3 checks, every check has about 6 bits: a "
             f"(3,6)-regular code of rate ½ with {v(r.get('edges', 0), 'd')} edges.</p>"
             "<p>Belief propagation passes messages along these edges and is exact as long as "
             "the graph looks like a tree. A short cycle lets a message come back to its sender "
             "and reinforce itself. The shortest possible, a <b>4-cycle</b> (two checks sharing "
             "two bits), is drawn in red.</p>")
        if p.kind.startswith("Random"):
            s += (f"<p>Random socket matching leaves {v(r.get('four', 0), 'd')} of them. "
                  "Progressive edge growth (PEG) adds edges one at a time, always to the check "
                  "that is farthest away in the current graph: switch to it.</p>")
        else:
            s += (f"<p>PEG construction: {v(r.get('four', 0), 'd')} four-cycles, girth "
                  f"{v(r.get('girth', '…'))}. Longer codes leave PEG more room, and the girth "
                  "grows.</p>")
        return "<h3>Bits, checks and cycles</h3>" + s + keybox(
            "The 5G NR base graphs, Wi-Fi and DVB-S2 codes are all designed to avoid short "
            "cycles: the graph's girth is a code designer's first concern.")


# =============================================================================== 2. BP live
BP_N = 48
BP_SEED = 5


@lru_cache(maxsize=1)
def bp_demo_code():
    H = cl.LDPCCode._peg(BP_N, BP_N // 2, 3, np.random.default_rng(BP_SEED)).astype(np.int8)
    return cl.LDPCCode(H=H), ft.TannerBP(H)


class BPLive(Experiment):
    title = "Belief propagation, live"
    blurb = "Watch beliefs flow along the graph and wrong bits flip back, one iteration at a time."
    book = "sec:ch15:ldpc"
    animate = True
    fps = 2
    controls = [
        Slider("ebn0", "Eb/N0", 0.0, 6.0, 3.0, step=0.1, unit="dB"),
        Choice("method", "Check-node rule", list(METHODS), "Sum-product", style="menu"),
        IntSlider("it", "Show iteration", 0, 20, 0),
        Button("again", "New noise"),
    ]
    plots = [
        Canvas("graph", "The (48,24) code's Tanner graph: red bits are wrong, red checks unsatisfied",
               xlim=(-0.03, 1.03), ylim=(-0.14, 1.16)),
        Plot("llr", "Belief in the right value per bit (above 0 = correct)", x="bit",
             y="LLR × sign of the sent bit", xlim=(-0.8, 47.8), ylim=(-12, 30), legend="tl",
             legend_cols=2),
        Plot("hist", "Wrong bits and failing checks per iteration", x="iteration", y="count",
             xlim=(-0.3, 20.3), ylim=(0, 14), legend="tr"),
    ]
    layout = [["graph", "graph"], ["llr", "hist"]]
    row_stretch = [3, 2]
    col_stretch = [3, 2]
    readouts = [
        Readout("ch", "Channel bit errors", "", "int"),
        Readout("now", "Wrong bits now", "", "int", good=lambda x: x == 0),
        Readout("uns", "Failing checks", "", "int", good=lambda x: x == 0),
        Readout("conv", "Converged at iteration", "", None),
    ]
    challenges = [
        Challenge("Watch BP correct at least 5 channel errors.",
                  lambda s: s.r.ch >= 5 and s.exp.final_ok),
        Challenge("Find a noise pattern that BP cannot fix in 20 iterations.",
                  lambda s: s.r.ch > 0 and not s.exp.final_ok),
        Challenge("Find a frame that min-sum fixes but where plain min-sum needs more iterations "
                  "than sum-product.", lambda s: s.p.method == "Min-sum" and s.exp.final_ok
                  and s.exp.ms_slower,
                  hint="Compare the 'converged at' readout for the same noise under both rules."),
    ]

    def setup(self):
        self.seed = 7
        self.final_ok = True
        self.ms_slower = False

    def on_again(self, p):
        self.seed += 1

    def tick(self, p):
        q = st.Params(p)
        q.it = (p.it + 1) % 21
        self.set_control("it", q.it)
        self.update(q)

    def run_bp(self, p, method):
        code, bp = bp_demo_code()
        rng = np.random.default_rng(self.seed)
        u = rng.integers(0, 2, code.k)
        c = code.encode(u)
        llr = ft.bpsk_llr(c, p.ebn0, code.rate, rng)
        out, used, hist = bp.decode(llr[None], 20, method, record=True, early_stop=False)
        return code, bp, c, llr, out, used, hist

    def update(self, p):
        code, bp, c, llr, out, used, hist = self.run_bp(p, METHODS[p.method])
        if p.method in ("Min-sum", "Sum-product"):
            o_s = self.run_bp(p, "spa")
            o_m = self.run_bp(p, "ms")
            self.ms_slower = (o_m[5][0] > o_s[5][0])
        sgn = 1 - 2.0 * c
        T = len(hist["L"]) - 1
        it = min(p.it, T)
        L = hist["L"][it][0]
        r_msg = hist["r"][it]
        wrong = (L < 0) != (c == 1)
        hard = (L < 0).astype(int)
        uns = (bp.Hi @ hard) % 2
        errs = [int(np.sum((h[0] < 0) != (c == 1))) for h in hist["L"]]
        fails = [int(((bp.Hi @ (h[0] < 0).astype(int)) % 2).sum()) for h in hist["L"]]
        # graph
        pg_ = self.plot("graph")
        m, n = bp.m, bp.n
        xv, xc = tanner_xy(m, n)
        ev, ce = bp.edge_v, bp.edge_c
        if it == 0:
            segments(pg_, "e", xv[ev], np.ones(len(ev)), xc[ce], np.zeros(len(ev)), color=GRAY,
                     width=0.7, alpha=0.6)
        else:
            agree = (r_msg * sgn[ev]) > 0
            strong = np.abs(r_msg) > 1.0
            for key, sel, col, wdt in (("eg", agree & ~strong, NAVY, 0.6), ("eG", agree & strong, NAVY, 1.4),
                                       ("er", ~agree & ~strong, RED, 0.6), ("eR", ~agree & strong, RED, 1.6)):
                segments(pg_, key, xv[ev[sel]], np.ones(sel.sum()), xc[ce[sel]], np.zeros(sel.sum()),
                         color=col, width=wdt, alpha=0.75 if col == RED else 0.45)
        sz = np.clip(6 + 1.2 * np.abs(L), 7, 22)
        for key, sel, col in (("vok", ~wrong, NAVY), ("vbad", wrong, RED)):
            if sel.any():
                pg_.scatter(key, xv[sel], np.ones(sel.sum()), color=col, size=float(np.median(sz[sel])), z=4)
        cs = uns.astype(bool)
        pg_.scatter("cok", xc[~cs], np.zeros((~cs).sum()), color=GREEN, size=12, symbol="s", z=4)
        if cs.any():
            pg_.scatter("cbad", xc[cs], np.zeros(cs.sum()), color=RED, size=14, symbol="s", z=4)
        pg_.text("vl", 0.0, 1.10, "bits (variable nodes)", anchor=(0, 0.5), size=9, color=NAVY)
        pg_.text("cl", 0.0, -0.09, "parity checks", anchor=(0, 0.5), size=9, color=TEAL)
        pg_.text("itl", 1.0, 1.10, f"iteration {it}" + (" (channel only)" if it == 0 else ""),
                 anchor=(1, 0.5), size=10, bold=True, color=ORANGE)
        # LLR bars
        pl = self.plot("llr")
        ch = llr * sgn
        now = L * sgn
        x = np.arange(n)
        pl.bars("ch", x - 0.2, np.clip(ch, -12, 30), width=0.38, color=GRAY)
        pl.bars("now", x + 0.2, np.clip(now, -12, 30), width=0.38,
                colors=[RED if q < 0 else NAVY for q in now])
        pl.scatter("lg1", [-50], [0], color=GRAY, size=10, symbol="s", name="channel only")
        pl.scatter("lg2", [-50], [0], color=NAVY, size=10, symbol="s", name=f"after {it} iterations")
        pl.hline("z", 0, color=pl.theme.text, style="-", width=0.8)
        # history
        ph = self.plot("hist")
        its = np.arange(len(errs))
        ph.line("err", its, errs, color=RED, width=2.2, name="wrong bits")
        ph.line("fail", its, fails, color=TEAL, width=2.0, style="--", name="failing checks")
        ph.vline("now", it, color=ORANGE, style=":", width=1.2)
        ph.set_ylim(0, max(14, max(errs + fails) + 3))
        self.final_ok = errs[-1] == 0
        conv = next((i for i, (e_, f_) in enumerate(zip(errs, fails)) if f_ == 0 and i > 0), None)
        if errs[0] == 0 and fails[0] == 0:
            conv = 0
        self.readout(ch=errs[0], now=errs[it], uns=int(uns.sum()),
                     conv="never (20)" if conv is None else str(conv))

    def story(self, p):
        r = self.r
        s = ("<p>Iteration 0: each bit knows only what the channel said (grey bars). Bits whose "
             "noise pushed them across zero are wrong (red circles), and every check that "
             "contains an odd number of wrong bits fails (red squares).</p>"
             "<p>Each iteration, every check tells each of its bits: \"given what the <i>other</i> "
             "bits in me believe, this is what you should be\". Navy edges carry advice towards "
             "the truth, red ones towards the error; thick edges are confident. Each bit adds the "
             "advice of its three checks to its channel belief.</p>")
        ch, now = r.get("ch", 0), r.get("now", 0)
        conv = r.get("conv", "")
        if ch == 0:
            s += "<p>No channel errors in this frame: lower Eb/N0 or press New noise.</p>"
        elif conv.startswith("never"):
            s += ("<p>" + bad(f"{ch} channel errors and BP is stuck:") + " the beliefs settle into "
                  "a wrong configuration (often a small trapping set) and the checks keep failing. "
                  "The decoder reports failure, and the CRC or HARQ takes over.</p>")
        else:
            s += ("<p>" + good(f"{ch} channel errors, all fixed by iteration {conv}.") +
                  " The decoder stops as soon as every check is satisfied (early stopping).</p>")
        return "<h3>Messages on a graph</h3>" + s + keybox(
            "Belief propagation is local, parallel and soft: thousands of tiny processors "
            "exchanging LLRs, which is why LDPC decoders reach tens of gigabits per second.")


# =============================================================================== 3. check-node rules
class CheckRules(Experiment):
    title = "Sum-product against min-sum"
    blurb = "The exact check-node rule, its cheap approximation, and the two fixes hardware uses."
    book = "sec:ch15:ldpc"
    heavy = True
    controls = [
        Heading("One check node"),
        IntSlider("dc", "Check degree", 3, 12, 6),
        Slider("q", "Confidence of the other inputs |q|", 0.2, 6.0, 4.0, step=0.05),
        Slider("alpha", "Normalisation α", 0.5, 1.0, 0.8, step=0.01),
        Slider("beta", "Offset β", 0.0, 1.5, 0.15, step=0.01),
        Heading("Decoders on the (576,288) code"),
        Button("rerun", "Run the simulation again", primary=True),
    ]
    plots = [
        Plot("rule", "Message from a check node against one input's |LLR|",
             x="|LLR| of the varying input", y="|outgoing message|", xlim=(0, 8), ylim=(0, 6.5),
             legend="tl"),
        BERPlot("fer", "Frame error rate, (576,288) PEG code, up to 30 iterations",
                x="Eb/N0 (dB)", y="frame error rate", xlim=(0.5, 3.5), ylim=(3e-3, 1.0),
                legend="bl"),
    ]
    layout = [["rule", "fer"]]
    readouts = [
        Readout("over", "Min-sum overconfidence", "×", ".2f"),
        Readout("nerr", "Normalised min-sum error", "%", ".1f", good=lambda x: abs(x) < 5),
        Readout("loss", "Min-sum loss at FER 10⁻¹", "dB", ".2f"),
        Readout("frames", "Frames simulated", "", "int"),
    ]
    challenges = [
        Challenge("Tune α so that normalised min-sum matches the exact rule within 3 % at |q| = 4 "
                  "and degree 6.", lambda s: s.p.dc == 6 and abs(s.p.q - 4) < 0.03 and abs(s.r.nerr) < 3),
        Challenge("Find a setting where plain min-sum overstates the message by more than a factor 2.",
                  lambda s: s.r.over > 2.0),
        Challenge("Measure what plain min-sum costs against sum-product: at least 0.2 dB at "
                  "FER 10⁻¹.", lambda s: s.r.loss is not None and s.r.loss >= 0.2),
    ]

    def setup(self):
        self.run_id = 0

    def on_rerun(self, p):
        self.run_id += 1

    def update(self, p):
        x = np.linspace(0.001, 8, 400)
        k = p.dc - 2
        tq = np.tanh(p.q / 2) ** k
        spa = 2 * np.arctanh(np.clip(np.tanh(x / 2) * tq, 0, 1 - 1e-15))
        ms = np.minimum(x, p.q)
        pr = self.plot("rule")
        pr.line("spa", x, spa, color=NAVY, width=3.0, name="sum-product (exact)")
        pr.line("ms", x, ms, color=RED, width=2.0, name="min-sum")
        pr.line("nms", x, p.alpha * ms, color=GREEN, width=2.0, style="--", name=f"normalised ×{p.alpha:.2f}")
        pr.line("oms", x, np.maximum(ms - p.beta, 0), color=ORANGE, width=2.0, style=":",
                name=f"offset −{p.beta:.2f}")
        pr.vline("q", p.q, color=GRAY, style=":", label="|q| of the others", label_pos=0.45)
        x0 = p.q
        e_spa = 2 * np.arctanh(min(np.tanh(x0 / 2) * tq, 1 - 1e-15))
        self.readout(over=float(p.q / max(e_spa, 1e-9)),
                     nerr=float(100 * (p.alpha * p.q / max(e_spa, 1e-9) - 1)))
        self.sim = {m: ([], []) for m in METHODS.values()}
        self.frames = 0
        pf = self.plot("fer")
        self.readout(loss=None, frames=0)

    def background(self, p):
        code = ldpc(576)
        bp = bp_for(576)
        rng = np.random.default_rng(90 + self.run_id)
        B = 8 if self.quick else 60
        ebs = [1.5, 2.5] if self.quick else [1.0, 1.5, 2.0, 2.5, 3.0]
        for eb in ebs:
            u = rng.integers(0, 2, (B, code.k))
            C = code.encode(u).reshape(B, -1)
            llr = ft.bpsk_llr(C, eb, 0.5, rng)
            for mname, mk in METHODS.items():
                out, _ = bp.decode(llr, 30 if not self.quick else 10, mk, alpha=p.alpha, beta=p.beta)
                fer = float(np.mean(np.any(out != C, axis=1)))
                yield dict(m=mk, eb=eb, fer=fer, B=B)

    def progress(self, p, it_):
        xs, ys = self.sim[it_["m"]]
        if it_["fer"] > 0:
            xs.append(it_["eb"])
            ys.append(it_["fer"])
        if it_["m"] == "spa":
            self.frames += it_["B"]
        pf = self.plot("fer")
        cols = {"spa": NAVY, "ms": RED, "nms": GREEN, "oms": ORANGE}
        names = {v_: k_ for k_, v_ in METHODS.items()}
        for mk, (x_, y_) in self.sim.items():
            if x_:
                pf.sim(mk, x_, y_, color=cols[mk], name=names[mk].replace(" (×0.8)", f" (×{p.alpha:.2f})")
                       .replace(" (−0.15)", f" (−{p.beta:.2f})"))
        a = interp_cross(*self.sim["spa"], 0.1)
        b = interp_cross(*self.sim["ms"], 0.1)
        self.readout(loss=None if (a is None or b is None) else b - a, frames=self.frames)

    def story(self, p):
        r = self.r
        s = ("<p>A check node with d_c edges sends each neighbour a message built from the other "
             "d_c − 1 inputs. The exact (<b>sum-product</b>) rule multiplies tanh(L/2) of the "
             "inputs: its output is weaker than the weakest input, because every other "
             "uncertain bit adds doubt. <b>Min-sum</b> keeps only the sign product and the "
             "smallest magnitude: no tanh, no multiplications, ideal for silicon, but "
             f"overconfident. Here it overstates the message {v(r.get('over', 1), '.2f')}×.</p>"
             "<p>Two cheap corrections: scale the min-sum output by α ≈ 0.75–0.85 "
             f"(<b>normalised</b>; at α = {v(p.alpha, '.2f')} the error is "
             f"{v(r.get('nerr', 0), '+.1f', '%')}) or subtract an offset β (<b>offset</b> min-sum).</p>")
        if r.get("loss") is not None:
            s += (f"<p>On the (576,288) code plain min-sum costs {v(r['loss'], '.2f', 'dB')} at "
                  "FER 10⁻¹; the corrected versions land almost on sum-product.</p>")
        else:
            s += "<p>Right: the frame error rate of all four decoders, simulated live.</p>"
        return "<h3>The check-node rule</h3>" + s + keybox(
            "Every LDPC chip in phones and modems runs layered normalised or offset min-sum: "
            "within about 0.1 dB of sum-product at a fraction of the cost.")


# =============================================================================== 4. iterations
class Iterations(Experiment):
    title = "Iterations and early stopping"
    blurb = "Most frames finish in a few iterations; the slow ones are usually the failures."
    book = "sec:ch15:ldpc"
    heavy = True
    controls = [
        Slider("ebn0", "Eb/N0", 0.5, 4.0, 2.0, step=0.1, unit="dB"),
        IntSlider("imax", "Maximum iterations", 5, 60, 30),
        Choice("method", "Check-node rule", list(METHODS), "Normalised min-sum (×0.8)", style="menu"),
    ]
    plots = [Plot("h", "Iterations needed per frame (200 frames of the (576,288) code)",
                  x="iterations to satisfy every check", y="frames",
                  xlim=(0, 61), ylim=(0, 60), legend="tr")]
    readouts = [
        Readout("mean", "Mean iterations", "", ".1f"),
        Readout("fer", "Frame error rate", "", "sci"),
        Readout("eff", "Average work / worst case", "%", "%", good=lambda x: x < 0.25),
        Readout("n", "Frames", "", "int"),
    ]
    challenges = [
        Challenge("Bring the average below 5 iterations with the frame error rate under 1 %.",
                  lambda s: s.r.n >= 150 and s.r.mean < 5 and s.r.fer < 0.01),
        Challenge("Find an Eb/N0 where more than 30 % of frames fail at 30 iterations or more.",
                  lambda s: s.r.n >= 150 and s.p.imax >= 30 and s.r.fer > 0.3),
        Challenge("Show that more iterations help: cut the FER by at least half by raising the "
                  "limit from 10 to 50 at the same Eb/N0.", lambda s: s.exp.halved),
    ]

    def setup(self):
        self.memo = {}
        self.halved = False

    def update(self, p):
        self.its = []
        self.fails = 0
        ph = self.plot("h")
        ph.set_xlim(0, p.imax + 1)
        ph.vline("max", p.imax, color=RED, style="--", label="limit (failures pile up here)",
                 label_pos=0.6)
        self.readout(mean=None, fer=None, eff=None, n=0)

    def background(self, p):
        code = ldpc(576)
        bp = bp_for(576)
        rng = np.random.default_rng(31)
        N = 20 if self.quick else 200
        B = 20 if self.quick else 50
        for _ in range(N // B):
            u = rng.integers(0, 2, (B, code.k))
            C = code.encode(u).reshape(B, -1)
            llr = ft.bpsk_llr(C, p.ebn0, 0.5, rng)
            out, used = bp.decode(llr, p.imax, METHODS[p.method])
            yield dict(used=used, fail=np.any(out != C, axis=1))

    def progress(self, p, it_):
        self.its.extend(it_["used"].tolist())
        self.fails += int(it_["fail"].sum())
        its = np.array(self.its)
        h = np.bincount(its, minlength=p.imax + 1)
        ph = self.plot("h")
        x = np.arange(len(h))
        ph.bars("b", x, h, width=0.8, colors=[RED if xi == p.imax else NAVY for xi in x])
        ph.set_ylim(0, max(60, int(h.max() * 1.15) + 1))
        fer = self.fails / len(its)
        key = (round(p.ebn0, 2), p.method)
        if len(its) >= 150:
            self.memo.setdefault(key, {})[p.imax] = fer
            d = self.memo[key]
            lo = [f for k_, f in d.items() if k_ <= 10]
            hi = [f for k_, f in d.items() if k_ >= 50]
            self.halved = bool(lo and hi and max(lo) > 0 and min(hi) <= 0.5 * max(lo))
        self.readout(mean=float(its.mean()), fer=fer, eff=float(its.mean() / p.imax), n=len(its))

    def story(self, p):
        r = self.r
        s = ("<p>BP stops as soon as the hard decisions satisfy every parity check. At good SNR "
             "that happens after a handful of iterations; noisy frames take longer, and frames "
             f"that never converge run all the way to the limit of {v(p.imax, 'd')} (red bar).</p>")
        if r.get("mean") is not None:
            s += (f"<p>Here the decoder needs {v(r['mean'], '.1f')} iterations on average, "
                  f"{v(100 * r.get('eff', 0), '.0f', '%')} of the worst case, and "
                  f"{v(100 * r.get('fer', 0), '.1f', '%')} of frames fail. A chip must be sized for "
                  "the worst case (latency) but its power follows the average.</p>")
        return "<h3>Work follows the SNR</h3>" + s + keybox(
            "Early stopping makes an LDPC decoder's energy per bit fall as the channel improves: "
            "a free gift of the syndrome check.")


# =============================================================================== 5. polarization
class Polarization(Experiment):
    title = "Channel polarization"
    blurb = "Arıkan's transform splits N copies of a channel into nearly perfect and useless ones."
    book = "sec:ch15:polar"
    controls = [
        IntSlider("lg", "Levels n (N = 2ⁿ channels)", 1, 12, 3),
        Slider("eps", "Erasure probability ε of the channel", 0.01, 0.99, 0.5, step=0.01),
        Slider("thr", "Call a channel good below", 0.001, 0.2, 0.01, step=0.001),
    ]
    plots = [
        Canvas("tree", "One step at a time: Z⁻ = 2Z − Z² (worse), Z⁺ = Z² (better)",
               xlim=(-0.6, 6.6), ylim=(-0.05, 1.12)),
        Plot("idx", "Erasure probability of every synthetic channel", x="channel index i",
             y="Z(Wᵢ)", ylim=(-0.03, 1.03), legend=None),
        Plot("sorted", "Sorted capacities: towards a step at 1 − ε", x="fraction of channels",
             y="capacity 1 − Z", xlim=(0, 1), ylim=(-0.03, 1.03), legend="tr"),
    ]
    layout = [["tree", "tree"], ["idx", "sorted"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("good", "Good channels", "%", "%"),
        Readout("bad", "Useless channels", "%", "%"),
        Readout("mid", "Still undecided", "%", "%", good=lambda x: x < 0.1),
        Readout("cap", "Average capacity (= 1 − ε)", "", ".3f"),
    ]
    challenges = [
        Challenge("At ε = 0.5, get at least 90 % of the channels decided (good or useless).",
                  lambda s: abs(s.p.eps - 0.5) < 0.005 and s.r.mid <= 0.10,
                  hint="Polarization is slow: it takes the largest N and a lenient threshold."),
        Challenge("Make a quarter of the channels good (25 % ± 1 %) with N ≥ 256.",
                  lambda s: s.p.lg >= 8 and abs(s.r.good - 0.25) <= 0.01),
        Challenge("With N = 16, find the erasure probability at which even the best channel "
                  "loses more than 1 % (Z > 0.01).", lambda s: s.p.lg == 4 and s.exp.zbest > 0.01),
    ]

    def setup(self):
        self.zbest = 0.0

    def update(self, p):
        N = 2 ** p.lg
        z = ft.polar_bec_z(N, p.eps)
        self.zbest = float(z.min())
        # the tree (first up to 6 levels)
        pt = self.plot("tree")
        L = min(p.lg, 6)
        pt.set_xlim(-0.6, L + 0.6)
        zl = [np.array([p.eps])]
        for _ in range(L):
            q = zl[-1]
            zl.append(np.stack([2 * q - q * q, q * q], axis=1).ravel())
        X0, Y0, X1, Y1 = [], [], [], []
        for lev in range(L):
            for j, zz in enumerate(zl[lev]):
                for b in (0, 1):
                    X0.append(lev); Y0.append(zz); X1.append(lev + 1); Y1.append(zl[lev + 1][2 * j + b])
        segments(pt, "br", X0, Y0, X1, Y1, color=GRAY, width=1.0 if L <= 4 else 0.5, alpha=0.8)
        for lev in range(L + 1):
            zz = zl[lev]
            cols = np.where(zz < p.thr, 1, np.where(zz > 1 - p.thr, -1, 0))
            for key, sel, col in ((f"g{lev}", cols == 1, GREEN), (f"b{lev}", cols == -1, RED),
                                  (f"m{lev}", cols == 0, NAVY)):
                if sel.any():
                    pt.scatter(key, np.full(sel.sum(), lev), zz[sel], color=col,
                               size=10 if lev <= 3 else 6, z=3)
            pt.text(f"lv{lev}", lev, 1.08, f"N = {2 ** lev}", anchor=(0.5, 0.5), size=8.5, color=GRAY)
        pt.hline("eps", p.eps, color=ORANGE, style=":", width=1.0)
        pt.text("ztop", -0.55, 1.0, "Z = 1 (useless)", anchor=(0, 0.5), size=8.5, color=RED)
        pt.text("zbot", -0.55, 0.0, "Z = 0 (perfect)", anchor=(0, 0.5), size=8.5, color=GREEN)
        # per index
        pi = self.plot("idx")
        pi.set_xlim(-0.5, N - 0.5)
        idx = np.arange(N)
        good = z < p.thr
        badc = z > 1 - p.thr
        mid = ~good & ~badc
        sz = 8 if N <= 64 else (4 if N <= 512 else 2.5)
        for key, sel, col in (("g", good, GREEN), ("b", badc, RED), ("m", mid, NAVY)):
            if sel.any():
                pi.scatter(key, idx[sel], z[sel], color=col, size=sz)
        # sorted
        ps = self.plot("sorted")
        frac = (np.arange(N) + 0.5) / N
        ps.line("cur", frac, np.sort(1 - z)[::-1], color=NAVY, width=2.6, name=f"N = {N}", step=False)
        for k_, col in ((1, GRAY), (4, GRAY), (8, GRAY)):
            if k_ != p.lg:
                zz = np.sort(1 - ft.polar_bec_z(2 ** k_, p.eps))[::-1]
                ps.line(f"ref{k_}", (np.arange(2 ** k_) + 0.5) / 2 ** k_, zz, color=col, width=1.0,
                        style=":", name="N = 2, 16, 256" if k_ == 1 else None)
        ps.line("step", [0, 1 - p.eps, 1 - p.eps, 1], [1, 1, 0, 0], color=GREEN, width=1.4,
                style="--", name="the limit N → ∞")
        self.readout(good=float(good.mean()), bad=float(badc.mean()), mid=float(mid.mean()),
                     cap=float(np.mean(1 - z)))

    def story(self, p):
        r = self.r
        N = 2 ** p.lg
        s = ("<p>Take two uses of an erasure channel and send u₁ ⊕ u₂ on the first and u₂ on "
             "the second. Decoding u₁ needs <i>both</i> outputs (worse: erased with probability "
             "2Z − Z²); decoding u₂ afterwards, knowing u₁, can use either (better: Z²). One step "
             "turns two equal channels into a worse and a better one, with the same total "
             "capacity. Repeat n times (top, left to right) and the channels <b>polarize</b>.</p>"
             f"<p>With N = {v(N, 'd')}: {v(100 * r.get('good', 0), '.0f', '%')} of the channels "
             f"are nearly perfect (green), {v(100 * r.get('bad', 0), '.0f', '%')} useless (red), "
             f"{v(100 * r.get('mid', 0), '.0f', '%')} still in between. The average capacity stays "
             f"exactly 1 − ε = {v(1 - p.eps, '.2f')}.</p>"
             "<p>A polar code puts information on the good channels and <b>freezes</b> the bad "
             "ones to known zeros. Polarization is slow, though: at practical lengths many "
             "channels are mediocre, which is why polar codes need list decoding and a CRC.</p>")
        return "<h3>Good, bad and undecided</h3>" + s + keybox(
            "Polar codes (Arıkan 2009) were the first codes proven to reach capacity with a "
            "practical decoder; 5G NR uses them for its control channels.")


# =============================================================================== 6. SC vs SCL
@lru_cache(maxsize=16)
def polar_code(N, crc):
    return cl.PolarCode(N, N // 2, design_snr_db=2.0, crc_poly=cl.CRC11_5G if crc else None)


class PolarDecoders(Experiment):
    title = "SC against CRC-aided list decoding"
    blurb = "Successive cancellation commits bit by bit; a list keeps alternatives and a CRC picks."
    book = "sec:ch15:polar"
    heavy = True
    controls = [
        Choice("N", "Block length N (rate ½)", ["128", "256"], "128"),
        Choice("L", "List size L", ["1", "2", "4", "8", "16"], "1"),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [
        Canvas("set", "Which bit-channels carry information (navy), CRC (orange), frozen (grey)",
               xlim=(-1, 129), ylim=(-0.8, 1.6)),
        BERPlot("fer", "Frame error rate", x="Eb/N0 (dB)", y="frame error rate",
                xlim=(0.5, 4.5), ylim=(3e-3, 1.0), legend="bl"),
    ]
    layout = [["set"], ["fer"]]
    row_stretch = [1, 5]
    readouts = [
        Readout("sc", "SC at FER 10⁻¹", "dB", ".2f"),
        Readout("scl", "List, no CRC", "dB", ".2f"),
        Readout("ca", "List + CRC-11", "dB", ".2f"),
        Readout("gain", "CRC-aided gain over SC (10⁻¹)", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Gain at least 0.5 dB over plain SC at FER 10⁻¹ with a list decoder.",
                  lambda s: s.r.sc is not None and min(x for x in (s.r.scl, s.r.ca, 99) if x is not None)
                  <= s.r.sc - 0.5),
        Challenge("Find where the CRC-aided list overtakes the plain list at high SNR (the plain "
                  "list's floor).", lambda s: s.exp.overtake,
                  hint="Short block, a list of 8 or more, and wait for the high-SNR points."),
    ]

    def setup(self):
        self.run_id = 0

    def on_rerun(self, p):
        self.run_id += 1

    def update(self, p):
        N = int(p.N)
        pc = polar_code(N, True)
        info = pc.info_set
        ps = self.plot("set")
        ps.set_xlim(-1, N + 1)
        kinds = np.zeros(N, int)
        kinds[info] = 1
        # the CRC bits occupy the last 11 positions of the information set (encoder order)
        kinds[info[pc.K:]] = 2
        cols = [GRAY if k == 0 else (NAVY if k == 1 else ORANGE) for k in kinds]
        cells(ps, "c", np.arange(N) + 0.5, np.full(N, 0.5), cols, w=0.85, h=0.9,
              alpha=0.9)
        ps.text("lab", 0, -0.45, f"{pc.K} message bits + 11 CRC bits on the most reliable of {N} "
                                 "channels; the rest frozen to 0", anchor=(0, 0.5), size=9, color=GRAY)
        self.sim = {k: ([], []) for k in ("sc", "scl", "ca")}
        self.fers = {}
        self.overtake = False
        self.readout(sc=None, scl=None, ca=None, gain=None)

    def background(self, p):
        N = int(p.N)
        L = int(p.L)
        rng = np.random.default_rng(300 + self.run_id)
        F = 12 if self.quick else 200
        ebs = [2.0, 3.0] if self.quick else [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        pc_n, pc_c = polar_code(N, False), polar_code(N, True)
        done = {"sc": False, "scl": False, "ca": False}
        for eb in ebs:
            for key, pc, LL in (("sc", pc_n, 1), ("scl", pc_n, L), ("ca", pc_c, L)):
                if done[key] or (key == "scl" and L == 1):
                    continue
                fe = 0
                for _ in range(F):
                    u = rng.integers(0, 2, pc.K)
                    llr = ft.bpsk_llr(pc.encode(u), eb, 0.5, rng)
                    fe += not np.array_equal(pc.decode(llr, L=LL), u)
                yield dict(k=key, eb=eb, fer=fe / F)
                if fe == 0:
                    done[key] = True

    def progress(self, p, it_):
        xs, ys = self.sim[it_["k"]]
        if it_["fer"] > 0:
            xs.append(it_["eb"])
            ys.append(it_["fer"])
        pf = self.plot("fer")
        names = {"sc": "SC (L = 1, no CRC)", "scl": f"list L = {p.L}, no CRC",
                 "ca": f"list L = {p.L} + CRC-11"}
        cols = {"sc": PURPLE, "scl": ORANGE, "ca": NAVY}
        for k, (x_, y_) in self.sim.items():
            if x_:
                pf.sim(k, x_, y_, color=cols[k], name=names[k])
        self.fers.setdefault(it_["eb"], {})[it_["k"]] = it_["fer"]
        self.overtake = any(d.get("scl", 0) > 0 and "ca" in d and d["ca"] < 0.6 * d["scl"]
                            and eb >= 2.5 for eb, d in self.fers.items())
        a, b, c = (interp_cross(*self.sim[k], 0.1) for k in ("sc", "scl", "ca"))
        self.readout(sc=a, scl=b, ca=c, gain=None if (a is None or c is None) else a - c)

    def story(self, p):
        r = self.r
        s = ("<p>The successive-cancellation (SC) decoder decides u₁, u₂, … in order, each "
             "decision using all the earlier ones. A frozen bit costs nothing (it is known), but "
             "an early wrong decision on a mediocre channel is never revisited and dooms the "
             "frame.</p>"
             f"<p>A <b>list</b> decoder keeps the {v(p.L)} most likely partial paths at every "
             "information bit and finally picks the best one, close to maximum likelihood. With "
             "a <b>CRC</b> appended to the message it picks the most "
             "likely survivor that passes the check: the list supplies candidates, the CRC "
             "the judgment (Tal and Vardy, 2015).</p>"
             "<p>The CRC is not free: its 11 bits sit on channels that would otherwise be frozen, "
             "so at high error rates the CRC-aided list trails the plain list. Lower down the plain "
             "list hits a floor set by the polar code's small minimum distance, and the CRC-aided "
             "curve dives past it; the CRC also tells the receiver when decoding failed.</p>")
        if r.get("gain") is not None:
            s += (f"<p>Measured: the CRC-aided list gains {v(r['gain'], '.2f', 'dB')} over SC at "
                  "FER 10⁻¹.</p>")
        if p.L == "1":
            s += "<p>L = 1 is plain SC; raise the list size.</p>"
        return "<h3>Keep the alternatives</h3>" + s + keybox(
            "5G NR control channels use CRC-aided list decoding with lists of about 8 and "
            "CRC-24 or CRC-11 (3GPP TS 38.212).")


# =============================================================================== 7. short blocks
SHORT = {"LDPC (96,48), BP": ("ldpc", 96), "LDPC (576,288), BP": ("ldpc", 576),
         "Polar (128,64), CA-SCL 8": ("polar", 128), "Polar (256,128), CA-SCL 8": ("polar", 256)}


@lru_cache(maxsize=64)
def na_curve(n, k):
    fers = np.logspace(-4, -0.5, 30)
    return fers, np.array([ft.na_ebn0_db(n, k, e) for e in fers])


class ShortBlocks(Experiment):
    title = "Short blocks and the finite-length limit"
    blurb = "How close real codes get to the best any code of that length could possibly do."
    book = "sec:ch15:compare"
    heavy = True
    controls = [
        Choice("code", "Code", list(SHORT), "Polar (128,64), CA-SCL 8", style="menu"),
        LogSlider("target", "Frame error rate of interest", 1e-3, 0.3, 1e-1),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [BERPlot("fer", "Frame error rate against the normal approximation for its length",
                     x="Eb/N0 (dB)", y="frame error rate", xlim=(0, 5), ylim=(1e-3, 1.0),
                     legend="bl")]
    readouts = [
        Readout("lim", "Limit for any length (rate ½)", "dB", ".2f"),
        Readout("na", "Best possible at this length", "dB", ".2f"),
        Readout("meas", "This code (measured)", "dB", ".2f"),
        Readout("gap", "Gap to the length limit", "dB", ".2f", good=lambda x: x < 1.0),
    ]
    challenges = [
        Challenge("Find how much the length alone costs: the length-128 limit is more than 1.5 dB "
                  "above Shannon at FER 10⁻³.", lambda s: "128" in s.p.code and s.p.target <= 1.05e-3
                  and s.r.na - s.r.lim > 1.5),
        Challenge("Find a code within 1 dB of the limit for its own length at FER 10⁻¹.",
                  lambda s: s.r.gap is not None and s.r.gap < 1.0 and s.p.target >= 0.095),
        Challenge("Compare the families: show the length-128 polar code beats the length-96 LDPC "
                  "code's measured point.", lambda s: s.exp.best.get("Polar (128,64), CA-SCL 8", 99)
                  < s.exp.best.get("LDPC (96,48), BP", -99)),
    ]

    def setup(self):
        self.run_id = 0
        self.best = {}

    def on_rerun(self, p):
        self.run_id += 1

    def update(self, p):
        fam, n = SHORT[p.code]
        k = n // 2
        pf = self.plot("fer")
        lim = ft.biawgn_limit_db(0.5)
        pf.vline("lim", lim, color=GRAY, style=":", label="Shannon, any length", label_pos=0.95)
        cols = {96: PURPLE, 128: GREEN, 256: TEAL, 576: ORANGE}
        for nn in (96, 128, 256, 576):
            fe, eb = na_curve(nn, nn // 2)
            mine = nn == n
            pf.theory(f"na{nn}", eb, fe, color=cols[nn], width=2.4 if mine else 1.0,
                      style="-" if mine else "--", name=f"limit for n = {nn}")
        na = ft.na_ebn0_db(n, k, p.target)
        pf.hline("tg", p.target, color=ORANGE, style=":", width=1.0)
        self.sim = ([], [])
        self.readout(lim=lim, na=na, meas=None, gap=None)

    def background(self, p):
        fam, n = SHORT[p.code]
        rng = np.random.default_rng(500 + self.run_id)
        F = 10 if self.quick else 120
        ebs = [2.0, 3.0] if self.quick else [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        for eb in ebs:
            if fam == "ldpc":
                code, bp = ldpc(n), bp_for(n)
                u = rng.integers(0, 2, (F, code.k))
                C = code.encode(u).reshape(F, -1)
                out, _ = bp.decode(ft.bpsk_llr(C, eb, 0.5, rng), 40, "spa")
                fer = float(np.mean(np.any(out != C, axis=1)))
            else:
                pc = polar_code(n, True)
                fe = 0
                for _ in range(F):
                    u = rng.integers(0, 2, pc.K)
                    fe += not np.array_equal(pc.decode(ft.bpsk_llr(pc.encode(u), eb, 0.5, rng), L=8), u)
                fer = fe / F
            yield dict(eb=eb, fer=fer)
            if fer == 0:
                break

    def progress(self, p, it_):
        if it_["fer"] > 0:
            self.sim[0].append(it_["eb"])
            self.sim[1].append(it_["fer"])
        if self.sim[0]:
            self.plot("fer").sim("s", self.sim[0], self.sim[1], color=RED, name=p.code + " (simulated)")
        x = interp_cross(*self.sim, p.target)
        if x is not None:
            self.best[p.code] = x
        self.readout(meas=x, gap=None if x is None else x - self.r.get("na", 0))

    def story(self, p):
        r = self.r
        fam, n = SHORT[p.code]
        s = ("<p>Shannon's limit (0.19 dB for rate ½ with BPSK) assumes infinitely long blocks. "
             "At a finite length n and a frame error rate ε, the best possible code needs more, "
             "by roughly √(V/n)·Q⁻¹(ε) in rate (the <b>normal approximation</b> of Polyanskiy, "
             "Poor and Verdú, 2010). The coloured curves are that limit for each length: the "
             "shorter the block, the further right.</p>"
             f"<p>At FER {v(f'{p.target:.0e}')} and n = {v(n, 'd')} no code can do better than "
             f"{v(r.get('na', 0), '.2f', 'dB')}, already {v(r.get('na', 0) - r.get('lim', 0), '.1f', 'dB')} "
             "above Shannon.")
        if r.get("gap") is not None:
            s += f" The simulated {p.code.split(',')[0]} code is {v(r['gap'], '.2f', 'dB')} from it."
        s += ("</p><p>At these short lengths CRC-aided polar codes beat LDPC codes, which is "
              "why 5G uses polar codes for short control messages and LDPC for long data blocks.</p>")
        return "<h3>Length costs decibels</h3>" + s + keybox(
            "The gap to capacity has two parts: the code's own imperfection and the length "
            "penalty that even a perfect code pays.")


# =============================================================================== the lab
LAB = st.Lab(9, "LDPC and Polar Codes", chapter=15, chapter_title="Turbo, LDPC and Polar Codes",
             experiments=[TannerGraph, BPLive, CheckRules, Iterations, Polarization,
                          PolarDecoders, ShortBlocks])

if __name__ == "__main__":
    st.run(LAB)
