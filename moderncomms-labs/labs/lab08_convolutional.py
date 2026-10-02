"""Lab 08 · Convolutional Codes and the Viterbi Algorithm   (Chapters 13 and 14)

Run it:      python labs/lab08_convolutional.py
Self-test:   python labs/lab08_convolutional.py --selftest

Shannon told us in 1948 how far from perfect every uncoded link is; the convolutional code
decoded by Viterbi's algorithm was the first practical tool that closed a large part of that
gap (Voyager, GSM, 802.11a/g, DVB-S). Eight experiments: the gap itself, a shift-register
encoder you can clock bit by bit, the trellis and the free distance, the Viterbi decoder step
by step (click the received bits to inject errors), hard against soft decisions, puncturing
(design your own pattern), traceback depth and quantisation, and burst errors with a CRC and
an interleaver.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import heapq
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

import commlib as cl
from commlib import blockcodes as bc
from commlib import fectools as ft
from commlib import infotheory as it
from commlib import modzoo as mz
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, BarPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, GRAY, TEAL)
from studio import v, keybox, good, bad
from _fecviz import Canvas, on_click, cells, segments, circle, pale, ink, bitstr


# =============================================================================== shared helpers
CODES = {"K = 3  (7, 5)": (3, (0o7, 0o5)), "K = 5  (23, 35)": (5, (0o23, 0o35)),
         "K = 7  (133, 171)": (7, (0o133, 0o171)), "K = 9  (561, 753)": (9, (0o561, 0o753))}
CODE_NAMES = list(CODES)
K7 = "K = 7  (133, 171)"


@lru_cache(maxsize=8)
def conv(name):
    K, g = CODES[name]
    return cl.ConvCode(K, g)


@lru_cache(maxsize=8)
def spectrum(name):
    cc = conv(name)
    return ft.distance_spectrum(cc, 26)


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def ebn0_at(fn, target, lo=-2.0, hi=20.0):
    """Eb/N0 (dB) where a decreasing error-rate function reaches ``target`` (None if never)."""
    g = lambda e: np.log10(max(float(fn(e)), 1e-300)) - np.log10(target)
    try:
        if g(lo) < 0 or g(hi) > 0:
            return None
        return float(brentq(g, lo, hi))
    except ValueError:
        return None


def interp_cross(x, y, target):
    """Eb/N0 where measured points (x ascending, y decreasing) cross ``target`` (log-linear)."""
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


def free_path(cc):
    """The minimum-weight error event: list of states (starting and ending in 0), inputs and
    branch output weights."""
    S = cc.S
    s1 = cc.next_state[0, 1]
    start = int(cc.outputs[0, 1].sum())
    h = [(start, 1, s1, (0, s1), (1,), (start,))]
    best = {}
    while h:
        w, n, s, path, ins, ws = heapq.heappop(h)
        if s == 0:
            return list(path), list(ins), list(ws)
        if best.get(s, 1e9) <= w or n > 40:
            continue
        best[s] = w
        for b in (0, 1):
            s2 = cc.next_state[s, b]
            wb = int(cc.outputs[s, b].sum())
            heapq.heappush(h, (w + wb, n + 1, s2, path + (s2,), ins + (b,), ws + (wb,)))
    return [0], [], []


def bpsk_rx(c, ebn0_db, rate, rng):
    s = ft.bpsk_sigma(ebn0_db, rate)
    return 1 - 2.0 * c + s * rng.standard_normal(np.shape(c)), s


# =============================================================================== 1. the gap
SCHEMES = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "16-QAM": 4, "64-QAM": 6, "256-QAM": 8}


@lru_cache(maxsize=4)
def capacity_curves():
    es = np.linspace(-5, 30, 71)
    lin = 10 ** (es / 10)
    out = {"Gaussian input (Shannon)": np.log2(1 + lin)}
    out["BPSK"] = np.array([it.mi_pam([-1, 1], 2 * x) for x in lin])
    out["QPSK"] = 2 * np.array([it.mi_pam([-1, 1], x) for x in lin])
    for M in (16, 64, 256):
        out[f"{M}-QAM"] = np.array([it.qam_cm(M, x) for x in lin])
    return es, out


@lru_cache(maxsize=64)
def k7_coded_point(target):
    sp = spectrum(K7)
    return ebn0_at(lambda e: ft.conv_union_bound(sp, e, 0.5), target, 0, 15)


class ShannonGap(Experiment):
    title = "How far is perfect?"
    blurb = "Shannon's limit against the uncoded constellations: the gap every code tries to close."
    book = "sec:ch13:plane"
    controls = [
        Slider("eta", "Spectral efficiency η", 0.1, 8.0, 2.0, step=0.02, unit="b/s/Hz",
               help="Bits per second per hertz of bandwidth (bits per complex symbol)"),
        Choice("scheme", "Uncoded scheme", list(SCHEMES), "QPSK", style="menu"),
        LogSlider("ber", "Target bit error rate", 1e-8, 1e-2, 1e-5,
                  help="The error rate at which the uncoded scheme is measured"),
    ]
    plots = [
        Plot("cap", "Capacity: Gaussian input against real constellations",
             x="Es/N0 (dB)", y="bits per symbol", xlim=(-5, 30), ylim=(0, 9.2), legend="tl"),
        Plot("plane", "The bandwidth–power plane", x="Eb/N0 (dB)", y="spectral efficiency η (b/s/Hz)",
             xlim=(-3, 28), ylim=(0.1, 14), logy=True, legend="br"),
    ]
    layout = [["cap", "plane"]]
    readouts = [
        Readout("lim", "Shannon limit at η", "dB", ".2f"),
        Readout("need", "Uncoded scheme needs", "dB", ".2f"),
        Readout("gap", "Gap to its limit", "dB", ".2f", good=lambda x: x < 3),
        Readout("k7", "K = 7 code + QPSK gap", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Find the spectral efficiency at which the Shannon limit is exactly 0 dB "
                  "(±0.03 dB).", lambda s: abs(s.r.lim) < 0.03,
                  hint="(2^η − 1)/η = 1. Try a round number."),
        Challenge("Find an uncoded scheme more than 10 dB away from its own Shannon limit.",
                  lambda s: s.r.gap > 10.0,
                  hint="The simplest scheme, measured at a demanding error rate."),
        Challenge("Bring 256-QAM within 6 dB of its limit by relaxing the target error rate.",
                  lambda s: s.p.scheme == "256-QAM" and s.r.gap < 6.0),
    ]

    def update(self, p):
        es, curves = capacity_curves()
        pc = self.plot("cap")
        cols = {"Gaussian input (Shannon)": NAVY, "BPSK": GRAY, "QPSK": GREEN, "16-QAM": ORANGE,
                "64-QAM": PURPLE, "256-QAM": TEAL}
        for i, (nm, y) in enumerate(curves.items()):
            pc.line(f"c{i}", es, y, color=cols[nm], width=2.6 if i == 0 else 1.6,
                    name=nm if i == 0 else nm)
        snr_lim = db10(2 ** p.eta - 1)
        pc.hline("eta", p.eta, color=RED, style=":", width=1.0)
        if snr_lim > -5:
            pc.scatter("pt", [snr_lim], [p.eta], color=RED, size=11, z=5)
        # the plane
        eta = np.logspace(-1.3, np.log10(14), 200)
        pp = self.plot("plane")
        pp.line("bound", it.shannon_ebn0_db(eta), eta, color=NAVY, width=2.6, name="Shannon limit")
        pp.vline("ult", -1.59, color=GRAY, style=":", label="−1.59 dB", label_pos=0.95)
        pp.text("imp", 1.0, 11.0, "impossible: left of the limit", color=GRAY, size=9)
        lim = float(it.shannon_ebn0_db(p.eta))
        pp.scatter("eta", [lim], [p.eta], color=GREEN, size=11, z=5, name="your η on the limit")
        k = SCHEMES[p.scheme]
        xs, ys = [], []
        for n, (nm, kk) in enumerate(SCHEMES.items()):
            x = mz.ebn0_required(mz.NAMES[nm], p.ber)
            xs.append(x)
            ys.append(kk)
            pp.text(f"t{n}", x + 0.35, kk, nm, color=RED if nm == p.scheme else GRAY, size=8.5,
                    anchor=(0, 0.5))
        pp.scatter("unc", xs, ys, color=GRAY, size=8, name="uncoded schemes")
        need = mz.ebn0_required(mz.NAMES[p.scheme], p.ber)
        lim_k = float(it.shannon_ebn0_db(k))
        pp.scatter("sel", [need], [k], color=RED, size=13, z=6)
        pp.line("gap", [lim_k, need], [k, k], color=ORANGE, width=2.4)
        pp.text("gapt", (lim_k + need) / 2, k * 1.12, f"gap {need - lim_k:.1f} dB", color=ORANGE,
                anchor=(0.5, 1), size=9)
        x7 = k7_coded_point(float(f"{p.ber:.3g}"))
        if x7 is not None:
            pp.scatter("k7", [x7], [1.0], color=PURPLE, size=11, symbol="s", z=6,
                       name="K = 7 code, rate ½, QPSK")
        self.readout(lim=lim, need=need, gap=need - lim_k,
                     k7=None if x7 is None else x7 - float(it.shannon_ebn0_db(1.0)))

    def story(self, p):
        r = self.r
        k = SCHEMES[p.scheme]
        s = (f"<p>Shannon's theorem says that at {v(p.eta, '.2f', 'b/s/Hz')} reliable communication is "
             f"possible down to Eb/N0 = {v(r.get('lim', 0), '.2f', 'dB')} and impossible below it, "
             "however clever the engineering. The limit is the navy curve on the right; as η → 0 "
             "it approaches −1.59 dB, the ultimate limit.</p>"
             f"<p>{p.scheme} without coding carries {v(k, 'd')} bit{'s' if k > 1 else ''} per symbol "
             f"and needs {v(r.get('need', 0), '.1f', 'dB')} to reach a bit error rate of "
             f"{v(f'{p.ber:.0e}')}. Its own limit is {v(it.shannon_ebn0_db(k), '.2f', 'dB')}, so "
             f"the orange gap is {v(r.get('gap', 0), '.1f', 'dB')}: a factor of "
             f"{v(10 ** (r.get('gap', 0) / 10), '.0f')} in transmit power, or antenna area, or "
             "range squared.</p>")
        if r.get("k7") is not None:
            s += (f"<p>The purple square is the K = 7 convolutional code of the rest of this lab "
                  f"(rate ½ on QPSK, η = 1): it closes the gap to {v(r['k7'], '.1f', 'dB')}. "
                  "Turbo, LDPC and polar codes (Labs 9 and 23) take it below 1 dB.</p>")
        return "<h3>The gap</h3>" + s + keybox(
            "Uncoded systems sit 7–10 dB from Shannon at practical error rates. Every decibel "
            "a code wins back is worth the same as a decibel of transmitter power.")


# =============================================================================== 2. the encoder
ENC_CODES = CODE_NAMES[:3]


class ShiftRegister(Experiment):
    title = "The shift-register encoder"
    blurb = "Clock a message through the register and watch each pair of output bits appear."
    book = "sec:ch14:conv"
    animate = True
    fps = 2
    controls = [
        Choice("code", "Code", ENC_CODES, ENC_CODES[0], style="menu"),
        IntSlider("msg", "Message (8 bits, as a number)", 0, 255, 178,
                  help="The 8 message bits, most significant bit sent first"),
        IntSlider("step", "Clock step", 0, 14, 3, help="How many bits have been shifted in"),
        Toggle("tail", "Flush with K − 1 zero tail bits", True),
    ]
    plots = [
        Canvas("reg", "The encoder at this clock step", xlim=(-3.6, 6.5), ylim=(-3.0, 3.0),
               aspect=True),
        Canvas("tl", "Input and output streams (the column being computed is highlighted)",
               xlim=(-3.0, 15.0), ylim=(-0.9, 3.7)),
    ]
    layout = [["reg"], ["tl"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("state", "Register state", "", None),
        Readout("out", "Output now", "", None),
        Readout("w", "Codeword weight", "", "int"),
        Readout("dfree", "Free distance", "", "int"),
    ]
    challenges = [
        Challenge("With K = 3, find a message whose codeword weight is at least 14.",
                  lambda s: s.p.code == ENC_CODES[0] and s.p.tail and s.r.w >= 14,
                  hint="Weight comes from ones in the register; where should they be?"),
        Challenge("With K = 7, find a non-zero message whose codeword weight is exactly the free "
                  "distance (10).", lambda s: s.p.code == ENC_CODES[2] and s.p.tail and s.p.msg > 0
                  and s.r.w == 10),
        Challenge("Clock all the way to the end of the tail and see the register return to zero.",
                  lambda s: s.p.tail and s.exp.at_end and s.p.msg > 0),
    ]

    def setup(self):
        self.at_end = False

    def bits(self, p):
        cc = conv(p.code)
        u = np.array([(p.msg >> (7 - i)) & 1 for i in range(8)])
        full = np.concatenate([u, np.zeros(cc.K - 1, int)]) if p.tail else u
        return cc, u, full

    def tick(self, p):
        cc, u, full = self.bits(p)
        nxt = (p.step + 1) % (len(full) + 1)
        q = st.Params(p)
        q.step = nxt
        self.set_control("step", nxt)
        self.update(q)

    def update(self, p):
        cc, u, full = self.bits(p)
        K = cc.K
        T = len(full)
        stp = min(p.step, T)
        self.at_end = stp == T
        reg = [full[stp - 1 - j] if stp - 1 - j >= 0 else 0 for j in range(K)]    # newest first
        outs = cc.encode(full, terminate=False).reshape(-1, 2)
        pr = self.plot("reg")
        dx = 1.25
        xb = np.arange(K) * dx
        pr.set_xlim(-3.6, xb[-1] + 3.6)
        cells(pr, "box", xb, np.zeros(K), [NAVY if b else pale(pr) for b in reg], w=0.9, h=0.9,
              labels=[str(b) for b in reg], text_color=[ink(pr) if b else pr.theme.text for b in reg],
              size=13, bold=True)
        for j in range(K):
            pr.text(f"lab{j}", xb[j], -0.62, "uₜ" if j == 0 else f"uₜ₋{j}".translate(
                str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")), color=GRAY, anchor=(0.5, 0), size=8.5)
        xa = xb.mean()
        for gi, (g, ya, col) in enumerate(zip(cc.gens, (2.2, -2.2), (RED, GREEN))):
            taps = [(g >> (K - 1 - j)) & 1 for j in range(K)]
            tj = [j for j in range(K) if taps[j]]
            y0 = 0.45 if ya > 0 else -0.45
            segments(pr, f"tap{gi}", xb[tj], np.full(len(tj), y0), np.full(len(tj), xa),
                     np.full(len(tj), ya - np.sign(ya) * 0.32), color=col, width=1.6)
            circle(pr, f"add{gi}", xa, ya, 0.32, color=col, width=2.0)
            pr.text(f"plus{gi}", xa, ya, "+", color=col, anchor=(0.5, 0.5), size=14, bold=True)
            ob = int(outs[stp - 1, gi]) if stp > 0 else None
            pr.line(f"ol{gi}", [xa + 0.32, xb[-1] + 1.3], [ya, ya], color=col, width=1.6)
            pr.text(f"o{gi}", xb[-1] + 1.45, ya, f"c{gi + 1} = {'–' if ob is None else ob}", color=col,
                    anchor=(0, 0.5), size=12, bold=True)
            pr.text(f"g{gi}", xa - 0.5, ya, f"g{gi + 1} = {oct(g)[2:]}₈", color=col, anchor=(1, 0.5),
                    size=9)
        nxt = full[stp:stp + 6]
        pr.text("in", -0.75, 0.0, ("next in: " + " ".join(str(b) for b in nxt)) if len(nxt) else
                "all bits sent", color=pr.theme.text, anchor=(1, 0.5), size=9.5)
        segments(pr, "shift", xb[:-1] + 0.45, np.zeros(K - 1), xb[1:] - 0.45, np.zeros(K - 1),
                 color=GRAY, width=1.2)
        # timeline
        ptl = self.plot("tl")
        ptl.set_xlim(-3.0, max(15.0, T + 0.5))
        t = np.arange(T)
        done = t < stp
        cur = t == stp - 1
        cin = [ORANGE if cu else (NAVY if (b and d) else (pale(ptl) if d else ptl.theme.plot_bg))
               for b, d, cu in zip(full, done, cur)]
        cells(ptl, "in", t, np.full(T, 3.0), cin, labels=[str(b) for b in full],
              text_color=[ink(ptl) if (d and (b or cu)) else ptl.theme.text for b, d, cu
                          in zip(full, done, cur)], size=10)
        for row, (y, col) in enumerate(((1.6, RED), (0.6, GREEN))):
            ob = outs[:, row]
            cc_ = [col if (d and b) else (pale(ptl) if d else ptl.theme.plot_bg) for b, d in zip(ob, done)]
            cells(ptl, f"o{row}", t, np.full(T, y), cc_,
                  labels=[str(b) if d else "" for b, d in zip(ob, done)],
                  text_color=[ink(ptl) if (d and b) else ptl.theme.text for b, d in zip(ob, done)],
                  size=10)
        ptl.text("r0", -0.7, 3.0, "input u", anchor=(1, 0.5), size=9)
        ptl.text("r1", -0.7, 1.6, "output c₁", anchor=(1, 0.5), size=9.5, color=RED)
        ptl.text("r2", -0.7, 0.6, "output c₂", anchor=(1, 0.5), size=9.5, color=GREEN)
        if p.tail:
            ptl.band("tailb", 7.5, T - 0.5, color=GRAY, alpha=0.10)
            ptl.text("tailt", (7.5 + T - 0.5) / 2, -0.35, "tail", color=GRAY, anchor=(0.5, 0.5), size=8.5)
        state = "".join(str(b) for b in reg[:K - 1])
        w_all = int(outs.sum()) if p.tail else int(outs.sum())
        self.readout(state=state, out=("–" if stp == 0 else f"{outs[stp - 1, 0]}{outs[stp - 1, 1]}"),
                     w=w_all, dfree=int(spectrum(p.code)[0][0]))

    def story(self, p):
        cc = conv(p.code)
        K = cc.K
        s = (f"<p>A convolutional encoder is a shift register of {v(K, 'd')} cells (the constraint "
             f"length K) and two XOR gates. Each clock tick shifts in one message bit and produces "
             f"<b>two</b> output bits, each the parity of the cells its generator taps (red: "
             f"g₁ = {oct(cc.gens[0])[2:]}, green: g₂ = {oct(cc.gens[1])[2:]}, in octal): rate ½.</p>"
             f"<p>Every input bit stays in the register for K = {K} ticks, so it influences "
             f"{v(2 * K, 'd')} output bits. That memory is the whole trick: a single bit error in "
             "the channel cannot hide, because many outputs vouch for each input. The codeword "
             f"for this message has weight {v(self.r.get('w', 0), 'd')}; the lightest possible "
             f"non-zero codeword has weight {v(self.r.get('dfree', 0), 'd')}, the <b>free "
             "distance</b>.</p>")
        if p.tail:
            s += ("<p>The grey tail is K − 1 zeros that flush the register back to the all-zero "
                  "state, so the decoder knows where the path ends.</p>")
        else:
            s += ("<p>Without the tail the last bits are protected by fewer outputs: the decoder "
                  "has to guess the final state.</p>")
        return "<h3>A machine with memory</h3>" + s + keybox(
            "Press Play to clock the register. The K = 7 code with generators 133 and 171 (octal) "
            "is in Voyager, 802.11a/g/n, DVB-S and countless modems.")


# =============================================================================== 3. trellis
def bend(x0, y0, x1, y1, amount, n=24):
    """Quadratic Bezier from (x0,y0) to (x1,y1) bowed sideways by ``amount``."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy) or 1.0
    cx, cy = mx - dy / L * amount, my + dx / L * amount
    t = np.linspace(0, 1, n)
    return ((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)


class Trellis(Experiment):
    title = "Trellis and free distance"
    blurb = "The state diagram unrolled in time, and the lightest detour that sets the error rate."
    book = "sec:ch14:conv"
    controls = [
        Choice("code", "Code", CODE_NAMES, CODE_NAMES[0], style="menu"),
        Slider("ebn0", "Eb/N0 for the error-rate terms", 0.0, 10.0, 7.0, step=0.1, unit="dB"),
        Toggle("labels", "Label the state diagram (K = 3)", True),
    ]
    plots = [
        Canvas("trellis", "Trellis (solid: input 0, dashed: input 1); red: the free-distance detour",
               xlim=(-1.0, 10.4), ylim=(-0.8, 4.0)),
        Canvas("state", "State diagram", xlim=(-1.6, 1.6), ylim=(-1.45, 1.45), aspect=True),
        Plot("terms", "Union bound, term by term: B_d·Q(√(2dR·Eb/N0))", x="output weight d (red: d_free)",
             y="bit error rate contribution", xlim=(3.5, 26.5), ylim=(1e-14, 1), logy=True,
             legend=None),
    ]
    layout = [["trellis", "trellis"], ["state", "terms"]]
    row_stretch = [5, 4]
    col_stretch = [2, 3]
    readouts = [
        Readout("dfree", "Free distance", "", "int"),
        Readout("A", "Detours at d_free", "", "int"),
        Readout("gain", "Asymptotic coding gain", "dB", ".2f"),
        Readout("share", "d_free term's share", "%", "%"),
    ]
    challenges = [
        Challenge("Find a code whose asymptotic coding gain exceeds 7.5 dB.",
                  lambda s: s.r.gain > 7.5),
        Challenge("For the K = 7 code, find the Eb/N0 above which the d_free detours alone carry "
                  "90 % of the union bound.",
                  lambda s: s.p.code == K7 and s.r.share >= 0.9,
                  hint="At high SNR the lightest detours dominate everything."),
        Challenge("Find a setting where heavier detours still carry more than half of the bound.",
                  lambda s: s.r.share < 0.5),
    ]

    def setup(self):
        for n in CODE_NAMES:                      # distance spectra: ~0.4 s once
            spectrum(n)

    def update(self, p):
        cc = conv(p.code)
        S, K = cc.S, cc.K
        T = 10
        pt = self.plot("trellis")
        ystep = 3.2 / max(S - 1, 1)
        ypos = lambda s: 3.2 - s * ystep
        for b, sty in ((0, "-"), (1, "--")):
            x0, y0, x1, y1 = [], [], [], []
            for t in range(T):
                for s in range(S):
                    x0.append(t)
                    y0.append(ypos(s))
                    x1.append(t + 1)
                    y1.append(ypos(cc.next_state[s, b]))
            segments(pt, f"br{b}", x0, y0, x1, y1, color=GRAY, width=1.2 if S <= 16 else 0.4,
                     style=sty, alpha=1.0 if S <= 16 else 0.35)
        path, ins, ws = free_path(cc)
        tp = np.arange(len(path)) + 1
        pt.line("fp", tp, [ypos(s) for s in path], color=RED, width=3.2, z=3)
        if S <= 16:
            xs, ys = np.repeat(np.arange(T + 1), S), np.tile([ypos(s) for s in range(S)], T + 1)
            pt.scatter("nodes", xs, ys, color=NAVY, size=7 if S <= 4 else 5, z=4)
            for s in range(S):
                pt.text(f"sl{s}", -0.25, ypos(s), format(s, f"0{K - 1}b"), anchor=(1, 0.5),
                        size=8.5 if S <= 8 else 7, color=GRAY)
        else:
            pt.text("many", -0.25, 1.6, f"{S}\nstates", anchor=(1, 0.5), size=9, color=GRAY)
        for i, (t, w) in enumerate(zip(tp[:-1], ws)):
            pt.text(f"w{i}", t + 0.5, 3.62, f"+{w}", color=RED, anchor=(0.5, 0.5), size=8.5)
        pt.text("wt", -0.25, 3.62, "weight", color=RED, anchor=(1, 0.5), size=8.5)
        pt.set_xlim(-1.0 if S <= 16 else -1.1, T + 0.4)
        # state diagram
        ps = self.plot("state")
        if S == 4:
            pos = {0: (-1.0, 0.0), 2: (0.0, 0.85), 1: (0.0, -0.85), 3: (1.0, 0.0)}
        else:
            ang = np.pi / 2 - 2 * np.pi * np.arange(S) / S
            pos = {s: (np.cos(a), np.sin(a)) for s, a in enumerate(ang)}
        fp_edges = set(zip(path[:-1], path[1:]))
        for b in (0, 1):
            XX, YY, XR, YR = [], [], [], []
            for s in range(S):
                ns = cc.next_state[s, b]
                (x0, y0), (x1, y1) = pos[s], pos[ns]
                if s == ns:
                    r = 0.22 if S == 4 else 0.06
                    cx, cy = x0 * (1 + r / max(np.hypot(x0, y0), 1e-9)), y0 * (1 + r / max(np.hypot(x0, y0), 1e-9))
                    th = np.linspace(0, 2 * np.pi, 30)
                    xx, yy = cx + r * np.cos(th), cy + r * np.sin(th)
                else:
                    xx, yy = bend(x0, y0, x1, y1, 0.3 if S == 4 else 0.05, n=24 if S <= 16 else 2)
                tgt = (XR, YR) if (s, ns) in fp_edges else (XX, YY)
                tgt[0].extend(list(xx) + [np.nan])
                tgt[1].extend(list(yy) + [np.nan])
                if S == 4 and p.labels:
                    o = "".join(str(x) for x in cc.outputs[s, b])
                    k = len(xx) // 2
                    lx, ly = xx[k], yy[k]
                    if s == ns:
                        lx, ly = x0 * 1.22, 0.42
                    ps.text(f"el{s}{b}", lx, ly, f"{b}/{o}", color=NAVY if (s, ns) not in fp_edges else RED,
                            anchor=(0.5, 0.5), size=8, fill=True)
            ps.line(f"e{b}", XX, YY, color=GRAY, width=1.0 if S <= 16 else 0.5,
                    style="-" if b == 0 else "--", alpha=0.85 if S <= 16 else 0.5)
            if XR:
                ps.line(f"er{b}", XR, YR, color=RED, width=2.6, style="-" if b == 0 else "--", z=2)
        pts = np.array([pos[s] for s in range(S)])
        ps.scatter("n", pts[:, 0], pts[:, 1], color=NAVY, size=26 if S == 4 else (7 if S <= 64 else 4),
                   z=4)
        if S == 4:
            for s in range(S):
                ps.text(f"nl{s}", pos[s][0], pos[s][1], format(s, "02b"), color=ink(ps),
                        anchor=(0.5, 0.5), size=9, bold=True)
        ps.set_title("State diagram" if S == 4 else f"State diagram: {S} states, {2 * S} branches")
        # union bound terms
        d, A, B = spectrum(p.code)
        e = 10 ** (p.ebn0 / 10)
        terms = B * ft.qfunc(np.sqrt(2 * d * 0.5 * e))
        pb = self.plot("terms")
        lo = 10.0 ** np.clip(np.floor(np.log10(max(terms[:6].min(), 1e-300))) - 1, -40, -6)
        pb.set_ylim(lo, 1)
        segments(pb, "tm", d, np.full(len(d), lo), d, np.maximum(terms, lo), color=NAVY, width=6)
        pb.scatter("tmd", d[:1], np.maximum(terms[:1], lo), color=RED, size=11, z=3)
        ub = terms.sum()
        pb.hline("ub", max(ub, lo), color=ORANGE, style="--", label=f"sum = {ub:.1e}",
                 label_pos=0.62)
        pb.hline("unc", max(float(ft.qfunc(np.sqrt(2 * e))), lo), color=GRAY, style=":",
                 label="uncoded BPSK", label_pos=0.83)
        self.readout(dfree=int(d[0]), A=int(A[0]), gain=db10(0.5 * d[0]),
                     share=float(terms[0] / max(ub, 1e-300)))

    def story(self, p):
        cc = conv(p.code)
        r = self.r
        s = (f"<p>The encoder's memory holds K − 1 = {cc.K - 1} bits, so it has {v(cc.S, 'd')} "
             "states. The <b>trellis</b> shows every possible path through them over time: from "
             "every state two branches leave (input 0 solid, input 1 dashed), and every codeword "
             "is one path. The <b>state diagram</b> is the same machine without the time axis.</p>"
             "<p>Because the code is linear, the error rate is decided by how the decoder can "
             "mistake the all-zero path for a <b>detour</b>: a path that leaves state 0 and "
             f"rejoins it. The red detour is the lightest, weight {v(r.get('dfree', 0), 'd')}; there "
             f"are {v(r.get('A', 0), 'd')} such detours. Soft-decision decoding gains about "
             f"10·log₁₀(R·d_free) = {v(r.get('gain', 0), '.1f', 'dB')} at high SNR.</p>")
        share = r.get("share", 0)
        s += (f"<p>Bottom right: each detour weight's contribution to the union bound at "
              f"{v(p.ebn0, '.1f', 'dB')}. The d_free detours carry {v(100 * share, '.0f', '%')} "
              + ("of it: high SNR, where only the nearest neighbours matter.</p>" if share > 0.6 else
                 "of it; heavier detours are so numerous that they still count. The bound is "
                 "loose here and real BER is lower.</p>"))
        return "<h3>Paths, detours and d_free</h3>" + s + keybox(
            "Each extra unit of K doubles the states (decoder work) and adds roughly one or two "
            "to d_free: the price/performance curve of convolutional codes.")


# =============================================================================== 4. Viterbi step by step
VC = cl.ConvCode(3, (0o7, 0o5))
YPOS = {0: 3.0, 2: 2.0, 1: 1.0, 3: 0.0}


class ViterbiSteps(Experiment):
    title = "Viterbi, step by step"
    blurb = "Click received bits to corrupt them; watch survivors, metrics and the traceback."
    book = "sec:ch14:viterbi"
    animate = True
    fps = 1.5
    controls = [
        IntSlider("msg", "Message (6 bits, as a number)", 0, 63, 45),
        IntSlider("step", "Decoder step", 0, 8, 8),
        Toggle("discard", "Show the discarded branches", True),
        Heading("Channel errors (or click a received bit)"),
        Button("two", "2 random errors"),
        Button("three", "3 random errors"),
        Button("clear", "Clear all errors"),
    ]
    plots = [
        Canvas("tr", "Trellis of the K = 3 (7, 5) code: survivors, path metrics and the decision",
               xlim=(-1.25, 8.55), ylim=(-1.15, 4.35)),
        Canvas("bits", "Sent message and decoded message (tail in grey)", xlim=(-1.25, 8.55),
               ylim=(-0.7, 1.7)),
    ]
    layout = [["tr"], ["bits"]]
    row_stretch = [4, 1]
    readouts = [
        Readout("nerr", "Channel errors", "", "int"),
        Readout("pm", "Best path metric", "", None),
        Readout("derr", "Decoded bit errors", "", "int", good=lambda x: x == 0),
        Readout("verdict", "Verdict", "", None),
    ]
    challenges = [
        Challenge("Make the decoder fail with only 3 channel errors.",
                  lambda s: s.r.nerr == 3 and s.r.derr > 0 and s.p.step == 8,
                  hint="Bunch them up: errors close together look like a detour."),
        Challenge("Get 4 channel errors corrected.",
                  lambda s: s.r.nerr == 4 and s.r.derr == 0 and s.p.step == 8,
                  hint="Spread them so no detour is cheaper than the true path."),
        Challenge("Rare: get 5 channel errors corrected (fewer than 1 % of patterns work).",
                  lambda s: s.r.nerr == 5 and s.r.derr == 0 and s.p.step == 8),
    ]

    def setup(self):
        self.err = np.zeros(16, dtype=int)
        self.err[[3, 8]] = 1
        on_click(self, "tr", self.clicked)

    def clicked(self, x, y):
        if abs(y - 3.82) > 0.3:
            return False
        t = int(np.floor(x))
        if not 0 <= t < 8:
            return False
        j = 0 if (x - t) < 0.5 else 1
        self.err[2 * t + j] ^= 1
        return True

    def _random(self, k):
        self.err[:] = 0
        self.err[self.rng.choice(16, k, replace=False)] = 1

    def on_two(self, p):
        self._random(2)

    def on_three(self, p):
        self._random(3)

    def on_clear(self, p):
        self.err[:] = 0

    def tick(self, p):
        nxt = (p.step + 1) % 9
        q = st.Params(p)
        q.step = nxt
        self.set_control("step", nxt)
        self.update(q)

    def update(self, p):
        u = np.array([(p.msg >> (5 - i)) & 1 for i in range(6)])
        c = VC.encode(u)                                 # 16 bits, terminated
        rx = (c ^ self.err).reshape(-1, 2)
        tr = ft.viterbi_trace(VC, rx)
        pm, surv = tr["pm"], tr["surv"]
        stp = p.step
        T = 8
        pl = self.plot("tr")
        # received pairs (clickable)
        xs, cols, labs, tcol = [], [], [], []
        for t in range(T):
            for j in range(2):
                bit = rx[t, j]
                e = self.err[2 * t + j]
                xs.append(t + 0.27 + 0.46 * j)
                cols.append(RED if e else pale(pl))
                labs.append(str(bit))
                tcol.append(ink(pl) if e else pl.theme.text)
        cells(pl, "rx", xs, np.full(len(xs), 3.82), cols, w=0.42, h=0.46, labels=labs,
              text_color=tcol, size=10, bold=True)
        pl.text("rxl", -0.25, 3.82, "received", anchor=(1, 0.5), size=9)
        # branches
        sx0, sy0, sx1, sy1 = [], [], [], []
        dx0, dy0, dx1, dy1 = [], [], [], []
        for t in range(min(stp, T)):
            for s in range(4):
                if not np.isfinite(pm[t, s]):
                    continue
                for b in (0, 1):
                    s2 = VC.next_state[s, b]
                    if surv[t + 1, s2] == s:
                        sx0.append(t); sy0.append(YPOS[s]); sx1.append(t + 1); sy1.append(YPOS[s2])
                    else:
                        dx0.append(t); dy0.append(YPOS[s]); dx1.append(t + 1); dy1.append(YPOS[s2])
        segments(pl, "surv", sx0, sy0, sx1, sy1, color=NAVY, width=2.0)
        if p.discard:
            segments(pl, "disc", dx0, dy0, dx1, dy1, color=GRAY, width=1.0, style=":")
        # traceback
        if stp == T:
            path = tr["path"]
            pl.line("tb", np.arange(T + 1), [YPOS[s] for s in path], color=ORANGE, width=7,
                    alpha=0.55, z=-1)
        # nodes and metrics
        nx, ny, lab = [], [], []
        for t in range(min(stp, T) + 1):
            for s in range(4):
                if np.isfinite(pm[t, s]):
                    nx.append(t); ny.append(YPOS[s]); lab.append(str(int(pm[t, s])))
        pl.scatter("nodes", nx, ny, color=pl.theme.plot_bg, size=22, outline=NAVY, z=5)
        for i, (a, b_, l) in enumerate(zip(nx, ny, lab)):
            pl.text(f"m{i}", a, b_, l, color=NAVY, anchor=(0.5, 0.5), size=9, bold=True)
        for s in range(4):
            pl.text(f"st{s}", -0.3, YPOS[s], format(s, "02b"), anchor=(1, 0.5), size=9.5, color=GRAY)
        pl.vline("now", stp, color=ORANGE, style="--", width=1.0)
        # decoded vs sent
        pb = self.plot("bits")
        full = np.concatenate([u, [0, 0]])
        dec = tr["bits"]
        sent_c = [NAVY if b else pale(pb) for b in full]
        cells(pb, "sent", np.arange(T) + 0.5, np.full(T, 1.0), sent_c, labels=list(full),
              text_color=[ink(pb) if b else pb.theme.text for b in full], size=10)
        if stp == T:
            dc = [RED if a != b else (NAVY if a else pale(pb)) for a, b in zip(dec, full)]
            cells(pb, "dec", np.arange(T) + 0.5, np.zeros(T), dc, labels=list(dec),
                  text_color=[ink(pb) if (a != b or a) else pb.theme.text for a, b in zip(dec, full)],
                  size=10)
        else:
            pb.text("wait", 4.0, 0.0, "decision after the last step (traceback)", color=GRAY,
                    anchor=(0.5, 0.5), size=9)
        pb.band("tail", 6.0, 8.0, color=GRAY, alpha=0.12)
        pb.text("l1", -0.25, 1.0, "sent", anchor=(1, 0.5), size=9)
        pb.text("l2", -0.25, 0.0, "decoded", anchor=(1, 0.5), size=9)
        nerr = int(self.err.sum())
        derr = int(np.sum(dec[:6] != u)) if stp == T else 0
        verdict = ("—" if stp < T else ("no errors" if nerr == 0 and derr == 0 else
                                        ("corrected" if derr == 0 else "decoding error")))
        self.readout(nerr=nerr, pm=str(int(pm[T, 0])) if stp == T else "…", derr=derr if stp == T else None,
                     verdict=verdict)
        self.final_metric = int(pm[T, 0])

    def story(self, p):
        r = self.r
        nerr = r.get("nerr", 0)
        s = ("<p>At every step the Viterbi decoder extends each of the four survivors by its two "
             "branches, adds the <b>branch metric</b> (how many received bits disagree with the "
             "branch's output), and keeps, for each state, only the cheaper of the two paths that "
             "arrive there (navy); the loser (dotted) is discarded for ever. The numbers in the "
             "circles are the accumulated disagreements.</p>")
        if p.step < 8:
            s += (f"<p>Step {v(p.step, 'd')} of 8. Press Play or move the step slider: watch the "
                  "survivors of different states merge into a common history.</p>")
        else:
            pmv = getattr(self, "final_metric", 0)
            s += (f"<p>After the last step the tail forces the path back to state 00 and the "
                  f"orange traceback reads the decisions. Its metric {v(pmv, 'd')} is the number "
                  f"of received bits it had to disbelieve; you injected {v(nerr, 'd')}. ")
            if r.get("derr", 0):
                s += bad("Here a wrong path was closer to what arrived than the right one: the "
                         "decoder made an error event, a short detour of wrong bits.") + "</p>"
            elif nerr:
                s += good("Every error was corrected.") + "</p>"
            else:
                s += "Click any received bit to flip it.</p>"
        s += ("<p>The free distance of this code is 5, so any 2 errors are always corrected; 3 "
              "errors only sometimes, and well-spread larger patterns can survive too.</p>")
        return "<h3>Survivors and the traceback</h3>" + s + keybox(
            "Viterbi = shortest path through the trellis. Work grows with the number of states, "
            "not with the number of possible messages: 4 states here, 64 in the K = 7 code.")


# =============================================================================== 5. hard vs soft
class HardSoft(Experiment):
    title = "Hard against soft decisions"
    blurb = "Live Monte Carlo: unquantised, 3-bit and hard-decision Viterbi against the union bounds."
    book = "sec:ch14:softhard"
    heavy = True
    controls = [
        Choice("code", "Code", CODE_NAMES, K7, style="menu"),
        Toggle("bounds", "Show the union bounds", True),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [BERPlot("ber", "Bit error rate after Viterbi decoding", x="Eb/N0 (dB)",
                     xlim=(0, 10), ylim=(1e-6, 0.5), legend="tr")]
    readouts = [
        Readout("sh", "Soft gain over hard at 10⁻⁴", "dB", ".2f"),
        Readout("q3", "3-bit loss at 10⁻⁴", "dB", ".2f"),
        Readout("cg", "Coding gain at 10⁻⁵ (bound)", "dB", ".2f"),
        Readout("bits", "Bits simulated", "", "int"),
    ]
    challenges = [
        Challenge("Measure a soft-decision advantage of at least 1.5 dB at 10⁻⁴.",
                  lambda s: s.r.sh is not None and s.r.sh >= 1.5),
        Challenge("Show that 3-bit soft decisions lose less than 0.4 dB against perfect ones.",
                  lambda s: s.r.q3 is not None and s.r.q3 < 0.4),
        Challenge("Pick a code whose bound promises at least 6 dB of coding gain at 10⁻⁵.",
                  lambda s: s.r.cg >= 6.0),
    ]
    MODES = [("soft", "soft (unquantised)", NAVY), ("q3", "soft, 3-bit quantised", GREEN),
             ("hard", "hard decisions", RED)]

    def setup(self):
        self.run_id = 0
        self.sim = {}

    def on_rerun(self, p):
        self.run_id += 1

    def update(self, p):
        self.sim = {m: ([], []) for m, _, _ in self.MODES}
        self.tot = 0
        eb = np.linspace(0, 10, 201)
        pb = self.plot("ber")
        pb.theory("unc", eb, ft.qfunc(np.sqrt(2 * 10 ** (eb / 10))), color=GRAY, width=1.4,
                  name="uncoded BPSK")
        sp = spectrum(p.code)
        if p.bounds:
            pb.theory("ubs", eb, np.minimum(ft.conv_union_bound(sp, eb, 0.5), 0.5), color=NAVY,
                      style="--", width=1.5, name="union bound, soft")
            pb.theory("ubh", eb, np.minimum(ft.conv_union_bound(sp, eb, 0.5, hard=True), 0.5),
                      color=RED, style="--", width=1.5, name="union bound, hard")
        x5 = ebn0_at(lambda e: ft.conv_union_bound(sp, e, 0.5), 1e-5, 0, 15)
        self.readout(sh=None, q3=None, cg=None if x5 is None else 9.588 - x5, bits=0)

    def background(self, p):
        cc = conv(p.code)
        rng = np.random.default_rng(800 + self.run_id)
        thorough = p.effort == "Thorough" and not self.quick
        scale = {3: 2.0, 5: 1.5, 7: 1.0, 9: 0.4}[cc.K]
        max_bits = 2e4 if self.quick else (scale * (1.5e6 if thorough else 3e5))
        target = 20 if self.quick else (200 if thorough else 80)
        floor = 2e-3 if self.quick else (5e-6 if thorough else 2e-5)
        k, B = 400, 50
        alive = {m: True for m, _, _ in self.MODES}
        for eb in np.arange(0.0, 10.01, 1.0 if not thorough else 0.5):
            if not any(alive.values()):
                break
            for m, _, _ in self.MODES:
                if not alive[m]:
                    continue
                errs = bits = 0
                while errs < target and bits < max_bits:
                    u = rng.integers(0, 2, (B, k))
                    c = cc.encode_batch(u)
                    y, s = bpsk_rx(c, eb, 0.5, rng)
                    met = ft.quantize_soft(y, s, None if m == "soft" else (3 if m == "q3" else 1))
                    errs += int(np.sum(cc.decode_batch(met) != u))
                    bits += u.size
                ber = errs / bits
                yield dict(mode=m, eb=eb, ber=ber, bits=bits)
                if ber < floor or errs == 0:
                    alive[m] = False
        yield dict(done=True)

    def progress(self, p, item):
        if item.get("done"):
            self.status("Simulation finished.")
            return
        xs, ys = self.sim[item["mode"]]
        if item["ber"] > 0:
            xs.append(item["eb"])
            ys.append(item["ber"])
        self.tot += item["bits"]
        pb = self.plot("ber")
        for m, lab, col in self.MODES:
            if self.sim[m][0]:
                pb.sim(f"s{m}", self.sim[m][0], self.sim[m][1], color=col, name=lab)
        xsoft = interp_cross(*self.sim["soft"], 1e-4)
        xhard = interp_cross(*self.sim["hard"], 1e-4)
        xq = interp_cross(*self.sim["q3"], 1e-4)
        self.readout(sh=None if (xsoft is None or xhard is None) else xhard - xsoft,
                     q3=None if (xsoft is None or xq is None) else xq - xsoft, bits=self.tot)

    def story(self, p):
        r = self.r
        cc = conv(p.code)
        s = ("<p>Three decoders see the same noisy samples. The <b>hard</b> one is told only the "
             "sign of each sample (a 0 or a 1), the <b>soft</b> one the sample itself, so it knows "
             "which bits are shaky. The 3-bit decoder sees the sample rounded to 8 levels, as a "
             "real modem does.</p>")
        if r.get("sh") is not None:
            s += (f"<p>Measured: soft decisions win {v(r['sh'], '.1f', 'dB')} at 10⁻⁴. Throwing "
                  "away the reliability costs about 2 dB, the same lesson that drives every "
                  "modern decoder to want LLRs.</p>")
        else:
            s += "<p>The circles appear as the simulation runs (about 80 errors per point).</p>"
        if r.get("q3") is not None:
            s += (f"<p>Three bits per sample are almost as good as infinitely many: the loss is "
                  f"{v(r['q3'], '.2f', 'dB')}.</p>")
        if r.get("cg") is not None:
            s += (f"<p>The K = {cc.K} code's soft union bound reaches 10⁻⁵ "
                  f"{v(r['cg'], '.1f', 'dB')} before uncoded BPSK. The bounds (dashed) are tight "
                  "below about 10⁻⁴ and too pessimistic at low SNR.</p>")
        return "<h3>Soft information is worth 2 dB</h3>" + s + keybox(
            "Hard decisions waste about 2 dB; 3-bit quantisation keeps all but a few tenths.")


# =============================================================================== 6. puncturing
PATTERNS = {"1/2 (mother code)": [1, 1], "2/3 (802.11)": [1, 1, 1, 0],
            "3/4 (802.11)": [1, 1, 1, 0, 0, 1], "5/6 (802.11)": [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            "Custom, period 3": None}
CUSTOM_KEYS = ["a1", "b1", "a2", "b2", "a3", "b3"]


class Puncturing(Experiment):
    title = "Puncturing: one decoder, many rates"
    blurb = "Delete coded bits in a pattern to raise the rate; design your own pattern."
    book = "sec:ch14:conv"
    heavy = True
    controls = [Choice("pat", "Puncturing pattern", list(PATTERNS), "3/4 (802.11)", style="menu"),
                Heading("Custom pattern: keep these bits")] + [
        Toggle(k, f"{'c₁' if k[0] == 'a' else 'c₂'} at step {k[1]}", True,
               enabled_if=lambda p: p.pat.startswith("Custom")) for k in CUSTOM_KEYS] + [
        Button("rerun", "Run again", primary=True)]
    plots = [
        Canvas("mask", "Which coded bits are sent (navy) and which are deleted (red ✕)",
               xlim=(-1.6, 12.2), ylim=(-0.75, 1.75)),
        BERPlot("ber", "Bit error rate, K = 7 code, soft Viterbi", x="Eb/N0 (dB)",
                xlim=(0, 10), ylim=(1e-6, 0.5), legend="tr"),
    ]
    layout = [["mask"], ["ber"]]
    row_stretch = [1, 4]
    readouts = [
        Readout("rate", "Code rate", "", None),
        Readout("dfree", "Free distance", "", "int"),
        Readout("gain", "Asymptotic gain", "dB", ".2f"),
        Readout("x4", "Eb/N0 at 10⁻⁴ (measured)", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Find a standard rate above ½ that loses less than 1 dB of asymptotic gain.",
                  lambda s: s.p.pat.startswith("2/3")),
        Challenge("Design your own rate-¾ pattern (keep 4 of 6) that matches 802.11's free "
                  "distance of 5.", lambda s: s.p.pat.startswith("Custom") and s.exp.kept == 4
                  and s.r.dfree == 5),
        Challenge("Find a rate-¾ pattern that is worse than the standard one (free distance 4).",
                  lambda s: s.p.pat.startswith("Custom") and s.exp.kept == 4 and s.r.dfree == 4),
    ]

    def setup(self):
        self.run_id = 0
        self.kept = 6

    def on_rerun(self, p):
        self.run_id += 1

    def pattern(self, p):
        pat = PATTERNS[p.pat]
        if pat is None:
            pat = [int(p[k]) for k in CUSTOM_KEYS]
        return np.array(pat, int)

    def update(self, p):
        cc = conv(K7)
        pat = self.pattern(p)
        steps = len(pat) // 2
        self.kept = int(pat.sum())
        ok = self.kept >= steps
        R = steps / self.kept if self.kept else np.inf
        self.R = R
        dfree = ft.punctured_dfree(cc, pat) if ok and self.kept else 0
        self.ok = ok and dfree > 0
        pm = self.plot("mask")
        T = 12
        m = np.tile(pat.reshape(-1, 2), (T // steps + 1, 1))[:T]
        for row, y in ((0, 1.0), (1, 0.0)):
            keep = m[:, row]
            cells(pm, f"r{row}", np.arange(T), np.full(T, y), [NAVY if k else pale(pm) for k in keep],
                  labels=[("" if k else "✕") for k in keep], text_color=RED, size=12, bold=True)
            pm.text(f"l{row}", -0.6, y, "c₁" if row == 0 else "c₂", anchor=(1, 0.5), size=10)
        for j in range(0, T + 1, steps):
            pm.vline(f"per{j}", j - 0.5, color=ORANGE, style="--", width=1.0)
        pb = self.plot("ber")
        eb = np.linspace(0, 10, 201)
        pb.theory("unc", eb, ft.qfunc(np.sqrt(2 * 10 ** (eb / 10))), color=GRAY, width=1.4,
                  name="uncoded BPSK")
        pb.theory("ub", eb, np.minimum(ft.conv_union_bound(spectrum(K7), eb, 0.5), 0.5), color=NAVY,
                  style="--", width=1.5, name="rate ½ (union bound)")
        self.sim = ([], [])
        rate_s = "—" if not ok else (f"{steps}/{self.kept}" if self.kept != 2 * steps else "1/2")
        if ok and self.kept and self.kept % steps == 0 and self.kept != 2 * steps:
            rate_s = f"{steps // np.gcd(steps, self.kept)}/{self.kept // np.gcd(steps, self.kept)}"
        self.readout(rate=rate_s, dfree=dfree, gain=db10(R * dfree) if self.ok else None, x4=None)

    def background(self, p):
        if not self.ok:
            yield dict(done=True)
            return
        cc = conv(K7)
        pat = self.pattern(p).astype(bool)
        rng = np.random.default_rng(900 + self.run_id)
        k, B = 396, 50
        n = 2 * (k + 6)
        mask = np.tile(pat, n // len(pat) + 1)[:n]
        max_bits = 1.5e4 if self.quick else 2.5e5
        target = 20 if self.quick else 80
        for eb in np.arange(1.0, 10.01, 1.0 if self.quick else 0.5):
            errs = bits = 0
            while errs < target and bits < max_bits:
                u = rng.integers(0, 2, (B, k))
                c = cc.encode_batch(u)[:, mask]
                y, s = bpsk_rx(c, eb, self.R, rng)
                L = np.zeros((B, n))
                L[:, mask] = 2 * y / s ** 2
                errs += int(np.sum(cc.decode_batch(L) != u))
                bits += u.size
            yield dict(eb=eb, ber=errs / bits)
            if errs == 0 or errs / bits < (2e-3 if self.quick else 3e-6):
                break
        yield dict(done=True)

    def progress(self, p, item):
        if item.get("done"):
            return
        if item["ber"] > 0:
            self.sim[0].append(item["eb"])
            self.sim[1].append(item["ber"])
        if self.sim[0]:
            self.plot("ber").sim("sim", self.sim[0], self.sim[1], color=RED,
                                 name=f"rate {self.r.get('rate', '')} (simulated)")
        self.readout(x4=interp_cross(self.sim[0], self.sim[1], 1e-4))

    def story(self, p):
        r = self.r
        if not self.ok:
            return ("<h3>Too much deleted</h3><p>" + bad("This pattern keeps fewer coded bits than "
                    "message bits (rate above 1) or leaves a path of weight zero: nothing can be "
                    "decoded.") + "</p>" + keybox("Keep at least one bit per message bit."))
        s = (f"<p>The mother code produces two bits per message bit. Deleting some of them in a "
             f"repeating pattern (red crosses) sends {v(self.kept, 'd')} bits for every "
             f"{v(len(self.pattern(p)) // 2, 'd')} message bits: rate {v(r.get('rate', ''))}. The "
             "receiver puts an LLR of zero (\"no idea\") where the deleted bits should be, and "
             "the very same Viterbi decoder runs unchanged.</p>"
             f"<p>The price: the free distance drops from 10 to {v(r.get('dfree', 0), 'd')}, and "
             f"the asymptotic gain 10·log₁₀(R·d_free) to {v(r.get('gain', 0), '.1f', 'dB')} "
             "(7.0 dB at rate ½).")
        if r.get("x4") is not None:
            s += f" Measured: 10⁻⁴ at {v(r['x4'], '.1f', 'dB')}."
        s += "</p>"
        if p.pat.startswith("Custom"):
            s += ("<p>Which bits you delete matters as much as how many: some rate-¾ patterns keep "
                  "d_free = 5, others only 4. Standards publish the best patterns found by search.</p>")
        return "<h3>Rate on demand</h3>" + s + keybox(
            "Puncturing gives a whole family of rates from one encoder and one decoder: 802.11a/g "
            "switches between ½, ⅔ and ¾ frame by frame, DVB-S offers up to ⅞.")


# =============================================================================== 7. traceback & quantisation
DEPTHS = [3, 5, 7, 10, 14, 20, 28, 40, 56]
QUANTS = [("hard", 1), ("2 bits", 2), ("3 bits", 3), ("4 bits", 4), ("unquantised", None)]


class Traceback(Experiment):
    title = "Traceback depth and quantisation"
    blurb = "A real decoder has finite memory and few bits per sample: how little is enough?"
    book = "sec:ch14:viterbi"
    heavy = True
    controls = [
        Choice("code", "Code", CODE_NAMES[:3], K7, style="menu"),
        IntSlider("depth", "Your decision depth", 2, 60, 10, unit="steps",
                  help="How far back the decoder traces before it commits to a bit"),
        Choice("quant", "Soft-decision resolution", [q for q, _ in QUANTS], "3 bits", style="menu"),
        Slider("ebn0", "Eb/N0", 1.0, 5.0, 3.0, step=0.25, unit="dB"),
    ]
    plots = [
        BERPlot("dep", "BER against decision depth (same noise for every depth)",
                x="decision depth (trellis steps)", xlim=(0, 60), ylim=(1e-5, 0.5), legend="tr"),
        Plot("q", "BER against bits per sample, at your depth", x="", y="bit error rate",
             logy=True, ylim=(1e-5, 0.5), xlim=(-0.6, 4.6), legend=None, grid=False),
    ]
    layout = [["dep", "q"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("mine", "BER at your depth", "", "sci"),
        Readout("full", "BER, full traceback", "", "sci"),
        Readout("pen", "Penalty factor", "×", ".2f", good=lambda x: x < 1.1),
        Readout("rule", "Depth / K", "", ".1f"),
    ]
    challenges = [
        Challenge("With the K = 7 code at 3 dB or less, find a depth of at most 40 whose BER is "
                  "within 25 % of full traceback.", lambda s: s.p.code == K7 and s.p.depth <= 40
                  and s.p.ebn0 <= 3.0 and s.r.pen is not None and s.r.pen < 1.25 and s.exp.ready),
        Challenge("Show that a depth of only K steps multiplies the BER by more than 10.",
                  lambda s: s.r.pen is not None and s.exp.ready and s.p.depth <= conv(s.p.code).K
                  and s.r.pen > 10),
        Challenge("Show that hard decisions cost the K = 7 code more than a factor 10 in BER at "
                  "4 dB.", lambda s: s.p.code == K7 and s.p.ebn0 >= 4 and s.exp.hard_ratio > 10),
    ]

    def setup(self):
        self.ready = False
        self.hard_ratio = 0.0

    def update(self, p):
        self.ready = False
        self.dep = {}
        self.qres = {}
        pd = self.plot("dep")
        pd.vline("mine", p.depth, color=ORANGE, style="--", label="your depth", label_pos=0.95)
        K = conv(p.code).K
        pd.vline("5k", 5 * K, color=GRAY, style=":", label="5K rule of thumb", label_pos=0.86)
        pq = self.plot("q")
        pq.set_xticks([(i, q) for i, (q, _) in enumerate(QUANTS)])
        self.readout(mine=None, full=None, pen=None, rule=p.depth / K)

    def background(self, p):
        cc = conv(p.code)
        rng = np.random.default_rng(77)
        F = 60 if self.quick else 400
        k = 120 if self.quick else 250
        u = rng.integers(0, 2, (F, k))
        c = cc.encode_batch(u)
        y, s = bpsk_rx(c, p.ebn0, 0.5, rng)
        qb = dict(QUANTS)[p.quant]
        met = ft.quantize_soft(y, s, qb)
        full = float(np.mean(ft.viterbi_window(cc, met)[:, :k] != u))
        yield dict(kind="full", ber=full)
        mine = float(np.mean(ft.viterbi_window(cc, met, p.depth)[:, :k] != u))
        yield dict(kind="mine", ber=mine)
        for D in DEPTHS:
            yield dict(kind="dep", D=D, ber=float(np.mean(ft.viterbi_window(cc, met, D)[:, :k] != u)))
        for i, (q, b) in enumerate(QUANTS):
            m2 = ft.quantize_soft(y, s, b)
            yield dict(kind="q", i=i, ber=float(np.mean(ft.viterbi_window(cc, m2, p.depth)[:, :k] != u)))
        yield dict(kind="done")

    def progress(self, p, it_):
        floor = 1.0 / (60 * 120 if self.quick else 400 * 250)
        if it_["kind"] == "full":
            self.full = it_["ber"]
            self.readout(full=it_["ber"])
            self.plot("dep").hline("full", max(it_["ber"], floor), color=GREEN, style="--",
                                   label="full traceback", label_pos=0.05)
        elif it_["kind"] == "mine":
            self.mine = it_["ber"]
            pen = it_["ber"] / max(self.full, floor) if it_["ber"] > 0 else 1.0
            self.readout(mine=it_["ber"], pen=pen)
            self.plot("dep").scatter("me", [p.depth], [max(it_["ber"], floor)], color=ORANGE, size=13,
                                     symbol="d", z=5)
        elif it_["kind"] == "dep":
            self.dep[it_["D"]] = it_["ber"]
            xs = sorted(self.dep)
            self.plot("dep").sim("curve", xs, [max(self.dep[x], floor) for x in xs], color=NAVY,
                                 name="measured")
        elif it_["kind"] == "q":
            self.qres[it_["i"]] = it_["ber"]
            idx = sorted(self.qres)
            cols = [ORANGE if QUANTS[i][0] == p.quant else NAVY for i in idx]
            pq = self.plot("q")
            for i, col in zip(idx, cols):
                pq.line(f"b{i}", [i, i], [1e-5, max(self.qres[i], floor)], color=col, width=34)
            if 0 in self.qres and 4 in self.qres:
                self.hard_ratio = self.qres[0] / max(self.qres[4], floor)
        else:
            self.ready = True

    def story(self, p):
        r = self.r
        K = conv(p.code).K
        s = ("<p>The textbook Viterbi decoder waits for the end of the frame before tracing back. "
             "A hardware decoder cannot: it keeps a fixed window of survivor history and, at "
             "every step, commits to the bit that left the window, following the path of the "
             "currently best state.</p>"
             f"<p>Your depth is {v(p.depth, 'd')} steps, {v(p.depth / K, '.1f')}× the constraint "
             "length. ")
        if r.get("pen") is not None:
            pen = r["pen"]
            s += (good(f"That is enough: the BER is within {100 * (pen - 1):.0f} % of full traceback.")
                  if pen < 1.1 else bad(f"Too short: the BER is {pen:.1f}× that of full traceback; "
                                        "the survivors have not merged yet when you commit."))
        s += (" The classical rule of thumb is 5K (grey line); punctured codes need more.</p>"
              "<p>On the right, the same depth with fewer bits per sample: 3 bits are nearly free, "
              "1 bit (hard decisions) is not.</p>")
        return "<h3>Finite memory, few bits</h3>" + s + keybox(
            "Survivor paths merge a few constraint lengths back: a decision depth of about 5K "
            "and 3-bit samples give essentially ideal Viterbi performance.")


# =============================================================================== 8. bursts
DEPTH_CHOICES = ["1 (no interleaver)", "2", "4", "8", "16", "32", "64", "128"]
CRC16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]          # x^16 + x^12 + x^5 + 1
K_INFO = 1002
N_CODED = 2 * (K_INFO + 16 + 6)                                       # 2048


@lru_cache(maxsize=2)
def crc_matrix(n, poly=tuple(CRC16)):
    """(n, r) GF(2) matrix M with crc(m) = m·M mod 2 (rows: x^(n-1-i+r) mod g)."""
    r = len(poly) - 1
    g = int("".join(map(str, poly)), 2)
    rows = []
    v = 1 << r                                       # x^r
    for _ in range(n):
        v2 = v
        while v2.bit_length() > r:
            v2 ^= g << (v2.bit_length() - 1 - r)
        rows.append([(v2 >> (r - 1 - j)) & 1 for j in range(r)])
        v = v2 << 1
    return np.array(rows[::-1], dtype=np.int64)


def crc16_batch(bits):
    return (np.asarray(bits, np.int64) @ crc_matrix(bits.shape[1])) % 2


class Bursts(Experiment):
    title = "Bursts, CRC and interleaving"
    blurb = "A fading channel delivers errors in bursts; an interleaver scatters them for Viterbi."
    book = "sec:ch14:concat"
    controls = [
        Heading("Channel (Gilbert–Elliott)"),
        Slider("blen", "Mean burst length", 1, 80, 30, step=1, unit="bits"),
        Slider("frac", "Time spent in bursts", 0.2, 10, 3.0, step=0.1, unit="%"),
        Heading("Receiver"),
        Choice("depth", "Interleaver depth", DEPTH_CHOICES, DEPTH_CHOICES[0], style="menu",
               help="Columns of the block interleaver: adjacent channel bits come from coded bits "
                    "this far apart"),
        Button("again", "New channel realisation"),
    ]
    plots = [
        ImagePlot("chan", "Channel errors, frame 1, in transmission order", x="bit (column)",
                  y="row", xlim=(0, 64), ylim=(0, 32)),
        ImagePlot("deint", "The same errors after de-interleaving (decoder order)",
                  x="bit (column)", y="row", xlim=(0, 64), ylim=(0, 32)),
        BarPlot("frames", "Decoded bit errors per frame (green: CRC passes, red: CRC catches it)",
                x="frame", y="bit errors", xlim=(0.3, 8.7), ylim=(0, 60), legend=None),
    ]
    layout = [["chan", "deint"], ["frames", "frames"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("raw", "Raw channel BER", "", "sci"),
        Readout("dec", "Decoded BER", "", "sci", good=lambda x: x < 1e-4),
        Readout("crc", "Frames flagged by CRC", "", None),
        Readout("run", "Worst 35-bit window at decoder", "errors", "int"),
    ]
    challenges = [
        Challenge("With mean bursts of at least 30 bits and 3 % of the time in bursts, deliver all "
                  "8 frames error-free.", lambda s: s.p.blen >= 30 and s.p.frac >= 3.0
                  and s.r.dec == 0),
        Challenge("Without an interleaver, find a channel with raw BER below 1 % that still "
                  "breaks at least 4 frames.", lambda s: s.p.depth.startswith("1 ") and s.r.raw < 0.01
                  and s.exp.nbad >= 4),
        Challenge("Find the smallest interleaver depth that cleans up bursts of 50+ bits at 2 % "
                  "or more (no frame errors).", lambda s: s.p.blen >= 50 and s.p.frac >= 2.0
                  and s.r.dec == 0 and int(s.p.depth.split()[0]) <= 32),
    ]

    def setup(self):
        self.seed = 8
        self.nbad = 0

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def simulate(p, seed):
        """Encode 8 CRC-protected frames, interleave, Gilbert-Elliott channel, decode."""
        rng = np.random.default_rng(seed)
        cc = conv(K7)
        D = int(p.depth.split()[0])
        F = 8
        info = rng.integers(0, 2, (F, K_INFO))
        crc = crc16_batch(info)
        msg = np.concatenate([info, crc], axis=1)
        c = cc.encode_batch(msg)
        pbg = 1.0 / p.blen
        frac = p.frac / 100
        pgb = frac * pbg / (1 - frac)
        e = bc.gilbert_elliott(F * N_CODED, pgb, pbg, e_good=0.0, e_bad=0.5, rng=rng).reshape(F, N_CODED)
        tx = np.array([bc.block_interleave(x, N_CODED // D, D) for x in c])
        rx = tx ^ e
        de = np.array([bc.block_deinterleave(x, N_CODED // D, D) for x in rx])
        err_dec_order = de ^ c
        dec = cc.decode_batch(1 - 2.0 * de)
        nerr = (dec[:, :K_INFO] != info).sum(axis=1)
        ok_crc = np.all(crc16_batch(dec[:, :K_INFO]) == dec[:, K_INFO:], axis=1)
        # the most errors the decoder sees in any 35-bit window (5 constraint lengths)
        best = int(max(np.convolve(err_dec_order[f], np.ones(35), "valid").max() for f in range(F)))
        return e, err_dec_order, nerr, ok_crc, best

    def update(self, p):
        F = 8
        e, err_dec_order, nerr, ok_crc, best = self.simulate(p, self.seed)
        cmap = ((0, pale(self.plot("chan"))), (1, RED))
        self.plot("chan").image("img", e[0].reshape(32, 64)[::-1], x=(0, 64), y=(0, 32),
                                cmap=cmap, levels=(0, 1))
        self.plot("deint").image("img", err_dec_order[0].reshape(32, 64)[::-1], x=(0, 64),
                                 y=(0, 32), cmap=cmap, levels=(0, 1))
        pf = self.plot("frames")
        cols = [GREEN if (o and n == 0) else (RED if not o else PURPLE) for o, n in zip(ok_crc, nerr)]
        pf.bars("b", np.arange(1, F + 1), np.maximum(nerr, 0.6), colors=cols, width=0.6)
        pf.set_ylim(0, max(60, int(nerr.max() * 1.2) + 1))
        self.nbad = int(np.sum(nerr > 0))
        und = int(np.sum(ok_crc & (nerr > 0)))
        self.readout(raw=float(e.mean()), dec=float(nerr.sum() / (F * K_INFO)),
                     crc=f"{int(np.sum(~ok_crc))} / {F}" + (f"  ({und} missed!)" if und else ""),
                     run=best)

    def story(self, p):
        r = self.r
        D = int(p.depth.split()[0])
        s = (f"<p>The channel is quiet most of the time, then enters a bad state for "
             f"{v(p.blen, '.0f')} bits on average, during which half the bits are wrong (top "
             "left, red). The average error rate is only "
             f"{v(100 * r.get('raw', 0), '.2f', '%')}, but the Viterbi decoder needs its errors "
             "<i>isolated</i>: a burst longer than a few constraint lengths overwhelms any "
             "detour comparison.</p>")
        if D == 1:
            s += ("<p>Without interleaving the decoder sees the bursts exactly as they arrived "
                  f"(up to {v(r.get('run', 0), 'd')} errors inside 35 consecutive bits, five "
                  "constraint lengths) and the frames that contain them fail.</p>")
        else:
            s += (f"<p>The block interleaver writes the coded bits into rows and reads them out "
                  f"by columns, so neighbours on the channel are {v(D, 'd')} bits apart for the "
                  f"decoder (top right): the worst 35-bit stretch it sees holds "
                  f"{v(r.get('run', 0), 'd')} errors. The price is latency: a whole block must arrive before decoding starts.</p>")
            rows = N_CODED // D
            if p.blen > 0.5 * rows:
                s += (f"<p>{bad('Too deep for this block:')} with {v(D, 'd')} columns the 2048-bit "
                      f"block has only {v(rows, 'd')} rows, so a burst longer than that wraps "
                      "around into the next column and lands next to its own beginning again. "
                      "Depth and burst length must both fit in the block.</p>")
        s += ("<p>Every frame also carries a 16-bit CRC: green bars passed it, red ones were "
              "caught. Higher layers retransmit the red ones (ARQ).</p>")
        return "<h3>Scatter the bursts</h3>" + s + keybox(
            "Convolutional codes want random errors. Interleavers (GSM, DVB, 802.11) turn bursts "
            "into scattered errors, and a CRC tells you when even that was not enough.")


# =============================================================================== the lab
LAB = st.Lab(8, "Convolutional Codes and the Viterbi Algorithm", chapter=14,
             chapter_title="Classical Codes",
             experiments=[ShannonGap, ShiftRegister, Trellis, ViterbiSteps, HardSoft, Puncturing,
                          Traceback, Bursts])

if __name__ == "__main__":
    st.run(LAB)
