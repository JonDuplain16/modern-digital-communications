"""Lab 19 · Information Theory: Entropy, Codes and Capacity   (Chapter 13)

Run it:      python labs/lab19_information_theory.py
Self-test:   python labs/lab19_information_theory.py --selftest

Shannon's two great numbers, live. Entropy says how few bits a source really needs;
capacity says how many bits a channel can carry without error. Nine experiments: the
surprise of a coin and a sensor with memory, the entropy of your own text and of a
photograph, a Huffman tree that rebuilds as you drag probabilities, Blahut–Arimoto on a
channel you design, the AWGN limit and its −1.59 dB wall, what real constellations and
short packets cost, water-filling over sub-channels, and the rate–distortion bound that
every quantiser chases.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import bz2
import lzma
import zlib
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import infotheory as it
from commlib import sourcecoding as sc
from commlib import sourcekit as sk
import studio as st
from studio import (Text, Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ConstellationPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
GRAYS = ("#000000", "#FFFFFF")


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def undb(x):
    return 10 ** (np.asarray(x, float) / 10)


def show_char(c):
    return "␣" if c == " " else c


@lru_cache(maxsize=1)
def book_text():
    return sc.load_text()


@lru_cache(maxsize=1)
def photo():
    try:
        return sc.load_image()
    except Exception:                                  # no matplotlib sample data: synthesise
        return synthetic_image()


@lru_cache(maxsize=1)
def synthetic_image(n=256):
    """A deterministic test card: smooth gradients, discs, stripes and a little noise."""
    y, x = np.mgrid[:n, :n] / n
    img = 90 + 80 * x + 40 * np.sin(2 * np.pi * y * 1.5)
    for cx, cy, r, lev in [(0.3, 0.35, 0.16, 225), (0.72, 0.62, 0.2, 40), (0.6, 0.22, 0.09, 180)]:
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = lev
    band = (y > 0.78) & (y < 0.92)
    img[band] = 128 + 100 * np.sign(np.sin(2 * np.pi * x[band] * 12))
    img += np.random.default_rng(5).normal(0, 3, img.shape)
    return np.clip(np.round(img), 0, 255)


def blended_seq(text, order=3, beta=1.0):
    """Per-character code length (bits) of an interpolated context model (a simple cousin of
    PPM): p_k(c) = (n_k(c) + d·p_(k−1)(c)) / (n_k + d), from order 0 up to ``order``, where d
    is β times the number of distinct symbols seen in that context (Witten–Bell escape), on top
    of a uniform base over the text's alphabet. It learns much faster than a single high-order
    model on short texts."""
    counts = [defaultdict(Counter) for _ in range(order + 1)]
    K = max(len(set(text)), 2)
    out = np.empty(len(text))
    for i, c in enumerate(text):
        pr = 1.0 / K
        for k in range(order + 1):
            if i < k:
                break
            cnt = counts[k][text[i - k:i]]
            n = sum(cnt.values())
            d = beta * max(len(cnt), 1)              # Witten–Bell: escape weight = distinct symbols
            pr = (cnt[c] + d * pr) / (n + d)
        out[i] = -np.log2(pr)
        for k in range(order + 1):
            if i >= k:
                counts[k][text[i - k:i]][c] += 1
    return out


def compressors(data):
    """bits per symbol of three general-purpose compressors on a byte string."""
    n = max(len(data), 1)
    return {"zlib (LZ77 + Huffman)": 8 * len(zlib.compress(data, 9)) / n,
            "bzip2 (BWT)": 8 * len(bz2.compress(data, 9)) / n,
            "xz (LZMA)": 8 * len(lzma.compress(data, preset=9)) / n}


@lru_cache(maxsize=32)
def text_stats(text):
    plug = [it.ngram_entropy(text, k) for k in range(4)]
    adap = [sc.adaptive_ctx_bits(text, k) / len(text) for k in range(4)]
    bl = blended_seq(text[:6000], 3)
    return plug, adap, compressors(text.encode("utf8", "replace")), bl


@lru_cache(maxsize=4)
def image_stats(which):
    img = (photo() if which == "Photograph" else synthetic_image()).astype(int)
    h_pix = float(it.entropy(np.bincount(img.ravel(), minlength=256)))
    left = np.diff(img, axis=1, prepend=128)
    h_left = float(it.entropy(np.bincount((left + 255).ravel())))
    pred = np.zeros_like(img)
    pred[1:, 1:] = img[1:, :-1] + img[:-1, 1:] - img[:-1, :-1]       # planar predictor
    a, b, c = img[1:, :-1], img[:-1, 1:], img[:-1, :-1]
    med = np.where(c >= np.maximum(a, b), np.minimum(a, b),
                   np.where(c <= np.minimum(a, b), np.maximum(a, b), a + b - c))   # LOCO-I MED
    planar = np.zeros_like(img)
    planar[1:, 1:] = img[1:, 1:] - med
    planar[0, :] = left[0, :]
    planar[:, 0] = left[:, 0]
    h_pl = float(it.entropy(np.bincount((planar + 255).ravel())))
    joint = np.bincount((img[:, :-1] * 256 + img[:, 1:]).ravel(), minlength=65536).reshape(256, 256)
    h_cond = float(it.entropy(joint.ravel()) - it.entropy(joint.sum(axis=1)))
    png = 8 * len(zlib.compress(img.astype(np.uint8).tobytes(), 9)) / img.size
    return dict(img=img, left=left, planar=planar, h_pix=h_pix, h_left=h_left, h_pl=h_pl,
                h_cond=h_cond, png=png)


# =============================================================================== 1. surprise
class Surprise(Experiment):
    title = "Surprise and entropy"
    blurb = "Rare events carry more information. Average the surprise and you get entropy."
    book = "sec:ch13:entropy"
    controls = [
        Choice("src", "Source", ["Biased coin", "Markov sensor"],
               help="A memoryless coin, or a sensor whose next report depends on the last one"),
        Slider("p", "P(busy) for the coin", 0.01, 0.99, 0.2, step=0.01,
               enabled_if=lambda p: p.src == "Biased coin"),
        Heading("Markov sensor (quiet ↔ busy)"),
        Slider("s0", "P(stay quiet)", 0.01, 0.99, 0.9, step=0.01,
               help="Probability that a quiet report is followed by another quiet one",
               enabled_if=lambda p: p.src == "Markov sensor"),
        Slider("s1", "P(stay busy)", 0.01, 0.99, 0.7, step=0.01,
               enabled_if=lambda p: p.src == "Markov sensor"),
        Heading("Sample"),
        IntSlider("n", "Reports drawn", 50, 3000, 400, step=50),
        Button("again", "New sample"),
    ]
    plots = [
        Plot("h", "Binary entropy h(p) and the surprise of each outcome", x="probability of 'busy'",
             y="bits", xlim=(0, 1), ylim=(0, 4.2), legend="tr"),
        ImagePlot("map", "Entropy rate of the two-state sensor", x="P(stay quiet)",
                  y="P(stay busy)", xlim=(0, 1), ylim=(0, 1)),
        Plot("path", "Surprise of each report (red = busy, blue = quiet) and the running average (navy)",
             x="report number", y="bits", ylim=(0, 5.0), legend=None),
    ]
    layout = [["h", "map"], ["path", "path"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("H", "Entropy of one report", "bits", ".3f"),
        Readout("rate", "Entropy rate (with memory)", "bits", ".3f"),
        Readout("avg", "Measured average surprise", "bits", ".3f"),
        Readout("save", "Saving from memory", "%", ".0f"),
    ]
    challenges = [
        Challenge("Tune the coin so that each toss carries half a bit (0.50 ± 0.01).",
                  lambda s: s.p.src == "Biased coin" and abs(s.r.H - 0.5) < 0.01,
                  hint="Two answers: one rare 'busy', or one rare 'quiet'."),
        Challenge("Sensor: busy about half the time (45–55 %) yet an entropy rate below 0.3 bits.",
                  lambda s: s.p.src == "Markov sensor" and 0.45 <= s.exp.pi1 <= 0.55
                  and s.r.rate < 0.3,
                  hint="Make both states sticky: long runs of quiet, long runs of busy."),
        Challenge("Make memory pay: the entropy rate less than half the single-report entropy.",
                  lambda s: s.p.src == "Markov sensor" and s.r.save > 50),
        Challenge("Watch the law of large numbers: 2500 reports or more, average surprise within "
                  "0.02 bits of the entropy rate.",
                  lambda s: s.p.n >= 2500 and abs(s.r.avg - s.r.rate) < 0.02),
    ]

    def setup(self):
        self.seed = 1
        g = np.linspace(0.005, 0.995, 100)
        S0, S1 = np.meshgrid(g, g)
        self.map = self.rate_of(S0, S1)
        self.pi1 = 0.5

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def rate_of(s0, s1):
        pi1 = (1 - s0) / ((1 - s0) + (1 - s1))
        return (1 - pi1) * it.hb(1 - s0) + pi1 * it.hb(1 - s1)

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        if p.src == "Biased coin":
            s0, s1 = 1 - p.p, p.p                       # a memoryless coin is this Markov chain
        else:
            s0, s1 = p.s0, p.s1
        pi1 = (1 - s0) / ((1 - s0) + (1 - s1))
        self.pi1 = pi1
        Hm = float(it.hb(pi1))
        rate = float(self.rate_of(s0, s1))
        # sample path
        u = rng.random(p.n)
        x = np.zeros(p.n, int)
        x[0] = u[0] < pi1
        for k in range(1, p.n):
            x[k] = (u[k] < s1) if x[k - 1] else (u[k] >= s0)
        prev = np.r_[int(round(pi1 > 0.5)), x[:-1]]
        pstay = np.where(prev == 1, s1, s0)
        pcond = np.where(x == prev, pstay, 1 - pstay)
        pcond[0] = pi1 if x[0] else 1 - pi1
        surp = -np.log2(pcond)
        run = np.cumsum(surp) / np.arange(1, p.n + 1)
        # h(p) and surprise curves
        q = np.linspace(0.001, 0.999, 400)
        ph = self.plot("h")
        ph.line("s1", q, -np.log2(q), color=RED, style="--", width=1.4, name="surprise of 'busy' −log₂p")
        ph.line("s0", q, -np.log2(1 - q), color=BLUE, style="--", width=1.4,
                name="surprise of 'quiet' −log₂(1−p)")
        ph.line("hb", q, it.hb(q), color=NAVY, width=2.6, name="entropy h(p) = average surprise")
        ph.scatter("pt", [pi1], [Hm], color=RED, size=13, symbol="d", name="your source (one report)")
        if p.src == "Markov sensor":
            ph.hline("rate", rate, color=GREEN, style="-", width=2.0,
                     label=f"entropy rate {rate:.3f} bits", label_pos=0.62)
        # map
        pm = self.plot("map")
        pm.image("m", self.map, x=(0, 1), y=(0, 1), cmap="heat", levels=(0, 1), colorbar=True,
                 cbar_label="bits per report")
        pm.line("diag", [0, 1], [1, 0], color=GRAY, style=":", width=1.4)
        pm.text("dlab", 0.04, 0.9, "dotted: memoryless coins", color=GRAY, size=8.5, anchor=(0, 0.5),
                fill=True)
        pm.scatter("pt", [s0], [s1], color=RED, size=14, symbol="d", outline=RED)
        # path
        pp = self.plot("path")
        m = min(p.n, 400)
        k = np.arange(m)
        pp.line("x", k, 4.3 + 0.45 * x[:m], color=GRAY, width=1.2, step=False)
        pp.text("xl", 0, 4.25, "the reports (up = busy)", color=GRAY, size=8.5, anchor=(0, 1),
                fill=True)
        busy = x[:m] == 1
        pp.stems("sb", k[busy], np.minimum(surp[:m][busy], 4.1), color=RED, width=1.0, size=4, name="'busy' report")
        pp.stems("sq", k[~busy], np.minimum(surp[:m][~busy], 4.1), color=BLUE, width=1.0, size=4, name="'quiet' report")
        pp.line("run", np.arange(p.n), run, color=NAVY, width=2.4, name="running average")
        pp.hline("H", rate, color=GREEN, style="--", width=1.6, label="entropy rate",
                 label_pos=0.82)
        pp.set_xlim(-2, p.n)
        pp.set_ylim(0, 5.0)
        save = 100 * (1 - rate / Hm) if Hm > 1e-9 else 0.0
        self.readout(H=Hm, rate=rate, avg=float(run[-1]), save=save)

    def story(self, p):
        H, rate = self.r.get("H", 0), self.r.get("rate", 0)
        if p.src == "Biased coin":
            pb = p.p
            s = (f"<p>A 'busy' report with probability {v(pb, '.2f')} surprises you by "
                 f"{v(-np.log2(pb), '.2f', 'bits')}; a 'quiet' one by "
                 f"{v(-np.log2(1 - pb), '.2f', 'bits')}. Rare events are news. Averaging the "
                 f"surprise over many tosses gives the <b>entropy</b>, here {v(H, '.3f', 'bits')} per "
                 f"toss: the navy curve, highest (1 bit) for a fair coin.</p>"
                 "<p>Watch the bottom plot: each stem is one toss's surprise, and the running "
                 "average settles onto the entropy. That settling is the <b>asymptotic "
                 "equipartition property</b>, the reason compression works: long sequences "
                 "all cost about n·H bits.</p>")
        else:
            s = (f"<p>The sensor is busy {v(100 * self.pi1, '.0f', '%')} of the time, so a coder that "
                 f"ignores memory pays h = {v(H, '.3f', 'bits')} per report. But a quiet report "
                 f"is followed by another with probability {v(p.s0, '.2f')}: once you know the last "
                 f"report, the next is less surprising. The <b>entropy rate</b> is "
                 f"{v(rate, '.3f', 'bits')}, a saving of {v(self.r.get('save', 0), '.0f', '%')}.</p>"
                 "<p>On the map every point is a sensor; the dotted diagonal holds the memoryless "
                 "coins. Leave it toward the corners (sticky states) and the rate collapses even "
                 "though the busy fraction may stay at one half.</p>")
        return "<h3>Information is surprise</h3>" + s + keybox(
            "H = average of −log₂ p. Memory lowers the entropy rate below the single-symbol entropy.")


# =============================================================================== 2. text and images
TEXTS = [
    "the quick brown fox jumps over the lazy dog while the five boxing wizards jump quickly",
    "abababababababababababababababababababababababababababababab",
    "to be or not to be that is the question whether tis nobler in the mind to suffer the "
    "slings and arrows of outrageous fortune",
    "for i in range(10): total = total + i * i  # sum of squares; for j in range(10): total += j",
    "le petit prince demanda au renard ce que signifie apprivoiser et le renard repondit",
    "qzv xkwj pfyb gmtu hdrc nloe sazi vqxw kjpf ybgm tuhd rcnl oesa zivq",
]
DEFAULT_TEXT = ("information is the resolution of uncertainty. the more predictable a message is, "
                "the fewer bits it needs. shannon measured english and found that once you know "
                "the context, each letter carries only about one bit.")


class TextImage(Experiment):
    title = "Entropy of text and images"
    blurb = "Type anything, or pick the book or a photograph. How many bits does it really need?"
    book = "sec:ch13:entropy"
    controls = [
        Choice("src", "Source", ["Your text", "Book (Chapter 1)", "Photograph", "Synthetic image"],
               style="menu"),
        Text("text", "Your text", DEFAULT_TEXT, examples=TEXTS,
                help="Up to 600 characters; every keystroke updates the plots",
                enabled_if=lambda p: p.src == "Your text"),
        Choice("pred", "Image predictor", ["None (raw pixels)", "Left neighbour",
                                           "Planar (LOCO-I)"], style="menu",
               help="Code the difference between each pixel and a prediction from its neighbours",
               enabled_if=lambda p: "image" in p.src.lower() or p.src == "Photograph"),
    ]
    plots = [
        BarPlot("hist", "Symbol histogram", x="symbol", y="probability"),
        BarPlot("stair", "The entropy staircase", y="bits per symbol", legend=None),
        Plot("view", "Surprise of each character", legend=None),
    ]
    layout = [["hist", "stair"], ["view", "view"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("H0", "Single-symbol entropy", "bits", ".2f"),
        Readout("best", "Best model / predictor", "bits", ".2f"),
        Readout("zip", "Best compressor", "bits", ".2f"),
        Readout("red", "Redundancy", "%", ".0f",
                help="1 − (best model) / (fixed-length code): how much of a plain encoding is predictable"),
    ]
    challenges = [
        Challenge("Type at least 20 characters whose single-letter entropy is below 2 bits.",
                  lambda s: s.p.src == "Your text" and len(s.p.text) >= 20 and s.r.H0 < 2,
                  hint="Use only a few different letters."),
        Challenge("Type 40+ characters so predictable that the adaptive coder needs under 1.2 bits "
                  "per character, while the single-letter entropy stays above 3 bits.",
                  lambda s: s.p.src == "Your text" and len(s.p.text) >= 40 and s.r.best < 1.2
                  and s.r.H0 > 3,
                  hint="Many letters, but repeat a phrase: context makes it predictable."),
        Challenge("Get the photograph below 5.2 bits per pixel with a predictor.",
                  lambda s: s.p.src == "Photograph" and s.r.best < 5.2),
    ]

    def setup(self):
        self._aspect = None

    def _set_aspect(self, on):
        if self._aspect != on:
            self.plot("view").vb.setAspectLocked(on)
            self._aspect = on

    def update(self, p):
        if p.src in ("Your text", "Book (Chapter 1)"):
            self.text_mode(p)
        else:
            self.image_mode(p)

    # ------------------------------------------------------------------ text
    def text_mode(self, p):
        text = p.text if p.src == "Your text" else book_text()
        if len(text) < 2:
            text = (text + "ab")[:2] if text else "ab"
        plug, adap, comp, bl = text_stats(text)
        blend = float(np.mean(bl))
        cnt = Counter(text)
        syms = sorted(cnt, key=lambda c: (-cnt[c], c))
        K = len(syms)
        prob = np.array([cnt[c] for c in syms], float) / len(text)
        top = syms[:32]
        ph = self.plot("hist")
        ph.set_title(f"Symbol histogram ({K} different symbols, {len(text):,} characters)")
        ph.set_labels(x="symbol (most frequent first)", y="share of characters (%)")
        ph.bars("b", np.arange(len(top)), 100 * prob[:len(top)], width=0.75, color=NAVY)
        ph.set_xticks([(i, show_char(c)) for i, c in enumerate(top)])
        ph.set_xlim(-0.7, max(len(top), 8) - 0.3)
        ph.set_ylim(0, max(20, 100 * prob.max() * 1.15))
        # staircase
        ps = self.plot("stair")
        ps.set_title("<span style='color:#7F8C8D'>fixed</span> · "
                     "<span style='color:#2E6DA4'>n-gram</span> · "
                     "<span style='color:#1E8449'>adaptive</span> · "
                     "<span style='color:#C0392B'>blend</span> (bits/char)")
        ps.set_labels(x="", y="bits per character")
        uni = np.log2(max(K, 2))
        x = np.arange(4)
        ps.bars("uni", [-1], [uni], width=0.6, color=GRAY)
        ps.bars("plug", x - 0.18, plug, width=0.34, color=NAVY)
        ps.bars("adap", x + 0.18, adap, width=0.34, color=GREEN)
        ps.bars("blend", [4], [blend], width=0.6, color=RED)
        for i, (xx, val) in enumerate(zip([-1, 0.18, 1.18, 2.18, 3.18, 4],
                                          [uni] + list(adap) + [blend])):
            ps.text(f"v{i}", xx, val, f"{val:.2f}", anchor=(0.5, 1.05), size=8)
        cols = [ORANGE, PURPLE, TEAL]
        ymax = max(6.0, uni * 1.25)
        for i, (name, val) in enumerate(comp.items()):
            if len(text) >= 200 and val < ymax:
                ps.hline(f"c{i}", val, color=cols[i], style="--", width=1.3,
                         label=f"{name.split(' ')[0]} {val:.2f}", label_pos=0.3 + 0.22 * i)
        if p.src == "Book (Chapter 1)":
            ps.hband("human", 0.6, 1.3, color=RED, alpha=0.08)
            ps.text("hl", -1.5, 1.32, "Shannon's human readers", color=RED, size=8, anchor=(0, 0))
        ps.set_xticks([(-1, "log₂K"), (0, "order 0"), (1, "order 1"), (2, "order 2"),
                       (3, "order 3"), (4, "blend")])
        ps.set_xlim(-1.6, 4.6)
        ps.set_ylim(0, ymax * 1.12)
        # surprise view
        self._set_aspect(False)
        pv = self.plot("view")
        pv.set_title("Surprise of each character under the blended model "
                     "(green: predictable, orange: so-so, red: news)")
        pv.set_labels(x="character", y="bits")
        pv.set_yticks(None)
        start = 0 if p.src == "Your text" else 2000
        sv = bl[start:]
        n = min(len(sv), 90)
        pv.bars("sv", np.arange(n), sv[:n], width=0.8,
                colors=[RED if q > 4 else (ORANGE if q > 2 else GREEN) for q in sv[:n]])
        pv.set_xticks([(i, show_char(c)) for i, c in enumerate(text[start:start + n])])
        pv.hline("avg", blend, color=NAVY, style="--", width=1.4,
                 label=f"average {blend:.2f} bits/char", label_pos=0.86)
        pv.set_xlim(-0.7, max(n, 10) - 0.3)
        pv.set_ylim(0, 11)
        best = min(min(adap), blend)
        zipb = min(comp.values())
        self._mode = "text"
        self._info = dict(K=K, uni=uni, plug=plug, adap=adap, comp=comp, n=len(text),
                          best_order=int(np.argmin(adap)), blend=blend)
        self.readout(H0=plug[0], best=best, zip=zipb, red=100 * max(0.0, 1 - best / uni))

    # ------------------------------------------------------------------ images
    def image_mode(self, p):
        s = image_stats(p.src)
        img = s["img"]
        key = {"None (raw pixels)": None, "Left neighbour": "left", "Planar (LOCO-I)": "planar"}[p.pred]
        res = None if key is None else s[key]
        ph = self.plot("hist")
        ph.set_title("Histogram: pixels (gray) and prediction residuals (navy)")
        ph.set_labels(x="value", y="share of pixels (%)")
        hp = 100 * np.bincount(img.ravel(), minlength=256) / img.size
        ph.line("hp", np.arange(-0.5, 256), hp, step=True, color=GRAY, width=1.2, fill=0,
                fill_alpha=0.3)
        if res is not None:
            hr = 100 * np.bincount((res + 255).ravel(), minlength=511) / img.size
            ph.line("hr", np.arange(-255.5, 256), hr, step=True, color=NAVY, width=1.4, fill=0,
                    fill_alpha=0.25)
            ph.set_ylim(0, max(hr.max(), hp.max()) * 1.1)
        else:
            ph.set_ylim(0, hp.max() * 1.15)
        ph.set_xlim(-80, 260)
        ph.set_xticks([(k, str(k)) for k in (-64, 0, 64, 128, 192, 255)])
        ps = self.plot("stair")
        ps.set_title("Bits per pixel (red: your predictor)")
        ps.set_labels(x="", y="bits per pixel")
        vals = [8.0, s["h_pix"], s["h_cond"], s["h_left"], s["h_pl"]]
        sel = {None: 1, "left": 3, "planar": 4}[key]
        ps.bars("img", np.arange(5), vals, width=0.6,
                colors=[GRAY, NAVY, PURPLE, NAVY, NAVY], name=None)
        ps.bars("sel", [sel], [vals[sel]], width=0.6, color=RED)
        for i, val in enumerate(vals):
            ps.text(f"t{i}", i, val, f"{val:.2f}", anchor=(0.5, 1.05), size=8.5, bold=True)
        ps.hline("png", s["png"], color=ORANGE, style="--", width=1.3,
                 label=f"zlib on raw bytes (PNG-like) {s['png']:.2f}", label_pos=0.55)
        ps.set_xticks([(0, "raw"), (1, "H(pixel)"), (2, "H(x | left)"), (3, "left residual"),
                       (4, "planar residual")])
        ps.set_xlim(-0.6, 4.6)
        ps.set_ylim(0, 9.6)
        self._set_aspect(True)
        pv = self.plot("view")
        pv.set_labels(x="", y="")
        pv.set_xticks([])
        pv.set_yticks([])
        if res is None:
            pv.set_title(f"{p.src} (8 bits per pixel)")
            pv.image("im", img[::-1].astype(float), x=(0, img.shape[1]), y=(0, img.shape[0]),
                     cmap=GRAYS, levels=(0, 255))
        else:
            pv.set_title("Prediction residual (mid-gray = 0): what the coder actually sends")
            pv.image("im", np.clip(res[::-1], -48, 48).astype(float), x=(0, img.shape[1]),
                     y=(0, img.shape[0]), cmap=GRAYS, levels=(-48, 48))
        pv.set_xlim(0, img.shape[1])
        pv.set_ylim(0, img.shape[0])
        best = vals[sel]
        self._mode = "image"
        self._info = dict(vals=vals, png=s["png"])
        self.readout(H0=s["h_pix"], best=best, zip=s["png"], red=100 * (1 - best / 8))

    def story(self, p):
        if getattr(self, "_mode", "text") == "text":
            i = self._info
            src = "your text" if p.src == "Your text" else "the text of Chapter 1"
            s = (f"<p>{src.capitalize()} uses {v(i['K'], 'd')} different symbols, so a fixed-length "
                 f"code needs {v(i['uni'], '.2f', 'bits')} per character. Counting letters alone "
                 f"(order 0) gives {v(i['plug'][0], '.2f', 'bits')}: frequent letters are cheap.</p>"
                 f"<p>Context helps much more. The green bars are a real adaptive coder that "
                 f"predicts each character from the previous ones; its best is "
                 f"{v(min(i['adap']), '.2f', 'bits')} at order {v(i['best_order'], 'd')}. The red "
                 f"bar blends orders 0 to 3, as PPM compressors do, and learns faster: "
                 f"{v(i['blend'], '.2f', 'bits')}. The bottom plot shows where it pays: green "
                 f"characters were predictable, red ones were news.</p>")
            if i["n"] < 200:
                s += ("<p class='muted'>With fewer than 200 characters the plug-in n-gram "
                      "estimates (navy) are wildly optimistic: there are more contexts than "
                      "data. Trust the green bars.</p>")
            else:
                s += ("<p>The dashed lines are the compressors on your computer. They beat "
                      "letter counting but not a human reader, whom Shannon measured at about one "
                      "bit per letter.</p>")
            key = "Entropy depends on the model: the better you predict, the fewer bits you need."
        else:
            i = self._info
            s = (f"<p>Each pixel takes 8 bits, but its histogram has an entropy of only "
                 f"{v(i['vals'][1], '.2f', 'bits')}. That is not much of a saving: pixels are "
                 f"spread over the whole gray scale.</p>"
                 f"<p>Neighbouring pixels, though, are almost equal. Predict each pixel from its "
                 f"neighbours and send only the error: the residual histogram (navy) is a narrow "
                 f"spike around zero, with entropy "
                 f"{v(i['vals'][3], '.2f', 'bits')} (left neighbour) or "
                 f"{v(i['vals'][4], '.2f', 'bits')} (planar, as in lossless JPEG-LS). The "
                 f"residual image is mostly flat gray: the edges are all that is left to say.</p>")
            key = "Images are redundant in space: predict, then entropy-code the surprise."
        return "<h3>How many bits, really?</h3>" + s + keybox(key)


# =============================================================================== 3. Huffman
SYMS = "ABCDEF"
W_DEFAULT = [40, 20, 20, 10, 10, 5]


class Huffman(Experiment):
    title = "Huffman coding"
    blurb = "Drag the probabilities and watch the optimal prefix code rebuild itself."
    book = "sec:ch13:source"
    controls = ([IntSlider("n", "Number of symbols", 2, 6, 5),
                 Toggle("pairs", "Code pairs of symbols", False,
                        help="Build the Huffman code on pairs (AB, AC, …) instead of single symbols")] +
                [Heading("Probability weights")] +
                [Slider(f"w{i}", f"Symbol {SYMS[i]}", 0.0, 100.0, W_DEFAULT[i], step=0.5,
                        help="Relative weight; the weights are normalised to probabilities",
                        enabled_if=(lambda i: lambda p: p.n > i)(i)) for i in range(6)])
    plots = [
        Plot("tree", "The Huffman tree (left branch = 0, right = 1)", x="",
             y="depth = codeword length (bits)", legend=None),
        BarPlot("len", "Codeword length (bars) vs the ideal −log₂ p (red ◆)", x="symbol", y="bits", legend=None),
        Plot("block", "Coding blocks of symbols closes the gap", x="block length (symbols per codeword)",
             y="bits per source symbol", legend="tr"),
    ]
    layout = [["tree", "tree"], ["len", "block"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("H", "Entropy H", "bits", ".3f"),
        Readout("L", "Average length", "bits/symbol", ".3f"),
        Readout("eff", "Efficiency H / L", "%", ".1f", good=lambda x: x > 97),
        Readout("blk", "Best with blocks", "%", ".1f",
                help="Efficiency of a Huffman code on blocks of symbols (longest block shown)"),
    ]
    challenges = [
        Challenge("Make the code perfect: average length equal to the entropy (within 0.005 bits), "
                  "with at least 4 symbols, coding single symbols.",
                  lambda s: s.p.n >= 4 and not s.p.pairs and s.r.L - s.r.H < 0.005,
                  hint="Huffman is perfect when every probability is a power of ½."),
        Challenge("Make one symbol dominate (probability 0.9 or more) so that single-symbol coding "
                  "wastes more than 0.4 bit per symbol.",
                  lambda s: not s.p.pairs and s.exp.pmax >= 0.9 and s.r.L - s.r.H > 0.4),
        Challenge("Then switch on pairs and close at least half of that gap.",
                  lambda s: s.p.pairs and s.exp.pmax >= 0.9
                  and (s.r.L - s.r.H) < 0.5 * (s.exp.L1 - s.r.H),
                  hint="Longer blocks close it further: see the curve bottom right."),
    ]

    def setup(self):
        self.L1, self.pmax = 1.0, 0.5

    def probs(self, p):
        w = np.array([max(p[f"w{i}"], 0.0) for i in range(p.n)], float)
        if w.sum() <= 0:
            w = np.ones(p.n)
        w = np.maximum(w, 1e-4 * w.sum())
        return w / w.sum()

    def update(self, p):
        pr = self.probs(p)
        syms = SYMS[:p.n]
        P = dict(zip(syms, pr))
        code = it.huffman_code(P)
        H = float(it.entropy(pr))
        L = float(sum(P[s] * len(code[s]) for s in syms))
        self.L1, self.pmax = L, float(pr.max())
        self.draw_tree(code, P)
        # lengths vs ideal
        pl = self.plot("len")
        x = np.arange(p.n)
        lens = [len(code[s]) for s in syms]
        ideal = -np.log2(pr)
        pl.bars("L", x, lens, width=0.6, color=NAVY)
        pl.scatter("id", x, ideal, color=RED, size=12, symbol="d")
        pl.set_xticks([(i, s) for i, s in enumerate(syms)])
        pl.set_xlim(-0.7, p.n - 0.3)
        pl.set_ylim(0, max(max(lens), ideal.max()) * 1.3 + 0.5)
        # block coding
        nmax = int(max(1, min(6, np.floor(np.log(1024) / np.log(p.n)))))
        Lb = []
        for nb in range(1, nmax + 1):
            grids = np.meshgrid(*([pr] * nb), indexing="ij")
            pb = np.prod(np.stack([g.ravel() for g in grids]), axis=0)
            lens_b = it.huffman_lengths(dict(enumerate(pb)))
            Lb.append(sum(pb[k] * lens_b[k] for k in range(len(pb))) / nb)
        ns = np.arange(1, nmax + 1)
        pk = self.plot("block")
        nn = np.linspace(1, nmax, 100)
        pk.line("bound", nn, H + 1 / nn, color=GRAY, style=":", width=1.5, name="bound H + 1/n")
        pk.hline("H", H, color=GREEN, style="--", width=1.8, label=f"entropy {H:.3f}", label_pos=0.7)
        pk.line("Lb", ns, Lb, color=NAVY, width=2.0, name="Huffman on blocks")
        pk.scatter("Lbp", ns, Lb, color=NAVY, size=9)
        k_you = 2 if (p.pairs and nmax >= 2) else 1
        pk.scatter("you", [k_you], [Lb[k_you - 1]], color=RED, size=15, symbol="d",
                   name="your code")
        pk.set_xlim(0.8, nmax + 0.2)
        pk.set_ylim(max(0, H - 0.2), H + 1.15)
        pk.set_xticks([(k, str(k)) for k in ns])
        kraft = it.kraft_sum(lens)
        self._st = dict(code=code, P=P, kraft=kraft, nmax=nmax, Lb=Lb)
        Lyou = Lb[k_you - 1]
        self.readout(H=H, L=Lyou, eff=100 * H / Lyou, blk=100 * H / Lb[-1])

    def draw_tree(self, code, P):
        pt = self.plot("tree")
        leaves = sorted(code, key=lambda s: code[s])
        xpos = {code[s]: i for i, s in enumerate(leaves)}
        nodes = set()
        for c in code.values():
            for k in range(len(c) + 1):
                nodes.add(c[:k])
        prob = {}

        def place(node):
            if node in xpos:
                prob[node] = P[[s for s in code if code[s] == node][0]]
                return xpos[node]
            a, b = place(node + "0"), place(node + "1")
            xpos[node] = (a + b) / 2
            prob[node] = prob[node + "0"] + prob[node + "1"]
            return xpos[node]

        place("")
        maxd = max(len(c) for c in code.values())
        ex, ey = [], []
        for nd in nodes:
            for b in "01":
                ch = nd + b
                if ch in nodes:
                    ex += [xpos[nd], xpos[ch], np.nan]
                    ey += [-len(nd), -len(ch), np.nan]
                    pt.text(f"e{ch}", (xpos[nd] + xpos[ch]) / 2 + (-0.12 if b == "0" else 0.12),
                            -(len(nd) + 0.5), b, color=ORANGE, size=9, bold=True, anchor=(0.5, 0.5))
        pt.line("edges", ex, ey, color=GRAY, width=2.0)
        inner = [nd for nd in nodes if nd not in code.values()]
        pt.scatter("inner", [xpos[n] for n in inner], [-len(n) for n in inner], color=GRAY, size=9)
        for nd in inner:
            pt.text(f"pi{nd}", xpos[nd], -len(nd) + 0.12, f"{prob[nd]:.2f}", color=GRAY, size=8,
                    anchor=(0.5, 1))
        lx = [xpos[code[s]] for s in leaves]
        ly = [-len(code[s]) for s in leaves]
        pt.scatter("leaves", lx, ly, color=NAVY, size=15)
        for s in leaves:
            pt.text(f"l{s}", xpos[code[s]], -len(code[s]) - 0.15,
                    f"{s}  ({P[s]:.2f})\n{code[s]}", color=NAVY, size=9, bold=True, anchor=(0.5, 0))
        pt.set_xlim(-0.8, len(leaves) - 0.2)
        pt.set_ylim(-(maxd + 1.0), 0.6)
        pt.set_yticks([(-d, str(d)) for d in range(maxd + 1)])
        pt.set_xticks([])

    def story(self, p):
        stt = getattr(self, "_st", None)
        if stt is None:
            return ""
        H, L = self.r.get("H", 0), self.r.get("L", 0)
        code, P = stt["code"], stt["P"]
        top = max(P, key=P.get)
        s = (f"<p>Huffman's recipe: repeatedly merge the two least likely symbols into one. "
             f"The tree above is the result; each leaf's depth is its codeword length, and no "
             f"codeword is the start of another, so the bit stream decodes without commas. "
             f"The likeliest symbol, {v(top, 's')} ({v(P[top], '.2f')}), gets "
             f"{v(code[top], 's')}.</p>"
             f"<p>The average length is {v(L, '.3f', 'bits')} against an entropy of "
             f"{v(H, '.3f', 'bits')}. No uniquely decodable code can beat the entropy, and Huffman "
             f"is always within one bit of it. ")
        if L - H < 0.005:
            s += good("Perfect: all probabilities are powers of ½.") + "</p>"
        else:
            s += ("The red diamonds show why it is not exact: ideal lengths −log₂ p are not "
                  "integers.</p>")
        if p.pairs:
            s += (f"<p>You are coding <b>pairs</b>: {v(p.n ** 2, 'd')} pair-symbols, each with its "
                  f"own codeword, for {v(L, '.3f', 'bits')} per original symbol.</p>")
        s += (f"<p>Coding blocks of symbols spreads that rounding loss over the block: with "
              f"{v(stt['nmax'], 'd')} symbols per codeword the code reaches "
              f"{v(self.r.get('blk', 0), '.1f', '%')} efficiency. Arithmetic coding takes this to "
              f"its limit.</p>")
        return "<h3>The optimal prefix code</h3>" + s + keybox(
            "H ≤ L̄ &lt; H + 1. Code n symbols at a time and the gap shrinks to 1/n.")


# =============================================================================== 4. DMC capacity
class DMCCapacity(Experiment):
    title = "DMC capacity"
    blurb = "Design a noisy channel; Blahut–Arimoto finds how many bits it can carry."
    book = "sec:ch13:dmc"
    controls = [
        Choice("ch", "Channel", ["BSC", "BEC", "Z-channel", "Custom"],
               help="BSC flips bits, BEC erases them, the Z-channel only turns ones into zeros"),
        Slider("pp", "Error / erasure probability", 0.0, 1.0, 0.1, step=0.005,
               enabled_if=lambda p: p.ch != "Custom"),
        Heading("Custom channel: 0, 1 in → 0, ?, 1 out"),
        Slider("a0", "P(0 received as 1)", 0.0, 0.5, 0.05, step=0.005,
               enabled_if=lambda p: p.ch == "Custom"),
        Slider("e0", "P(0 erased)", 0.0, 0.9, 0.1, step=0.005, enabled_if=lambda p: p.ch == "Custom"),
        Slider("a1", "P(1 received as 0)", 0.0, 0.5, 0.05, step=0.005,
               enabled_if=lambda p: p.ch == "Custom"),
        Slider("e1", "P(1 erased)", 0.0, 0.9, 0.1, step=0.005, enabled_if=lambda p: p.ch == "Custom"),
        Toggle("useless", "Add a useless input X = u", False,
               help="A third input whose output distribution is the average of the other two"),
    ]
    plots = [
        Plot("diag", "The channel", legend=None, grid=False),
        Plot("mi", "Mutual information vs the input distribution", x="P(X = 1)", y="I(X;Y) (bits)",
             xlim=(0, 1), ylim=(0, 1.08), legend="tr"),
        Plot("ba", "Blahut–Arimoto: two bounds squeeze C",
             x="iteration", y="bits per use", xlim=(0, 30), legend="tr"),
        Plot("cap", "Closed forms: capacity vs error probability", x="error / erasure probability",
             y="capacity (bits per use)", xlim=(0, 1), ylim=(0, 1.08), legend="tr", legend_cols=1),
    ]
    layout = [["diag", "mi"], ["ba", "cap"]]
    readouts = [
        Readout("C", "Capacity", "bits/use", ".4f"),
        Readout("pstar", "Best P(X = 1)", "", ".3f"),
        Readout("unif", "I with equal inputs", "bits", ".4f"),
        Readout("iters", "BA iterations to 10⁻⁶", "", "int",
                help="Started from P(X = 1) = 0.15, to make the convergence visible"),
    ]
    challenges = [
        Challenge("Find a BSC that flips most bits (p ≥ 0.85) yet carries more than 0.35 bits per use.",
                  lambda s: s.p.ch == "BSC" and s.p.pp >= 0.85 and s.r.C > 0.35,
                  hint="A channel that almost always lies is as good as one that almost never does."),
        Challenge("Make the best input clearly unequal: P*(X = 1) below 0.42.",
                  lambda s: s.r.pstar < 0.42),
        Challenge("Custom: a channel that flips and erases (each at least 5 % for both inputs) with "
                  "capacity between 0.60 and 0.65 bits.",
                  lambda s: s.p.ch == "Custom" and min(s.p.a0, s.p.a1, s.p.e0, s.p.e1) >= 0.05
                  and 0.60 <= s.r.C <= 0.65),
    ]

    def matrix(self, p):
        q = p.pp
        if p.ch == "BSC":
            return np.array([[1 - q, 0, q], [q, 0, 1 - q]])
        if p.ch == "BEC":
            return np.array([[1 - q, q, 0], [0, q, 1 - q]])
        if p.ch == "Z-channel":
            return np.array([[1, 0, 0], [q, 0, 1 - q]])
        W = np.array([[1 - p.a0 - p.e0, p.e0, p.a0], [p.a1, p.e1, 1 - p.a1 - p.e1]], float)
        for r in range(2):
            if W[r, 0 if r == 0 else 2] < 0:
                W[r] = np.maximum(W[r], 0)
                W[r] /= W[r].sum()
        return W

    def full_matrix(self, p):
        W = self.matrix(p)
        if p.useless:
            W = np.vstack([W, W.mean(axis=0)])
        return W

    def update(self, p):
        W = self.full_matrix(p)
        C, popt, _, _ = it.blahut_arimoto(W, iters=3000, tol=1e-12)
        self.pu = float(popt[2]) if len(popt) > 2 else None
        lo, up = self.ba_history(W, 31)
        gap = up - lo
        n6 = int(np.argmax(gap < 1e-6)) + 1 if np.any(gap < 1e-6) else 31
        self.draw_diagram(W)
        pm = self.plot("mi")
        if len(popt) == 2:
            pm.set_title("Mutual information vs the input distribution")
            pm.set_labels(x="P(X = 1)", y="I(X;Y) (bits)")
            q = np.linspace(0, 1, 201)
            I = np.array([it.dmc_mutual_info([1 - a, a], W) for a in q])
            pm.line("I", q, I, color=NAVY, width=2.4, name="I(X;Y)")
            pm.vline("half", 0.5, color=GRAY, style=":", width=1.0)
            pm.scatter("u", [0.5], [it.dmc_mutual_info([0.5, 0.5], W)], color=ORANGE, size=10,
                       name="equal inputs")
            pm.scatter("c", [popt[1]], [C], color=RED, size=14, symbol="d", name="capacity (BA)")
            pm.set_xticks(None)
            pm.set_xlim(0, 1)
            pm.set_ylim(0, 1.08)
        else:
            pm.set_title("Capacity-achieving input distribution (Blahut–Arimoto)")
            pm.set_labels(x="input", y="probability (%)")
            pm.bars("pb", [0, 1, 2], 100 * popt, width=0.55, colors=[NAVY, NAVY, RED])
            for i, val in enumerate(popt):
                pm.text(f"pt{i}", i, 100 * val, f"{val:.3f}", anchor=(0.5, 1.05), size=9, bold=True)
            pm.set_xticks([(0, "X = 0"), (1, "X = 1"), (2, "X = u (useless)")])
            pm.set_xlim(-0.6, 2.6)
            pm.set_ylim(0, 75)
        pb = self.plot("ba")
        k = np.arange(len(lo))
        pb.line("lo", k, lo, color=NAVY, width=2.0, name="lower bound")
        pb.line("up", k, up, color=RED, width=2.0, name="upper bound")
        pb.scatter("lop", k, lo, color=NAVY, size=5)
        pb.scatter("upp", k, up, color=RED, size=5)
        pb.hline("C", C, color=GREEN, style="--", width=1.4, label=f"C = {C:.4f}", label_pos=0.75)
        span = max(up[0] - C, C - lo[0], 0.02)
        pb.set_ylim(max(0, C - 1.2 * span), C + 1.25 * span)
        pc = self.plot("cap")
        x = np.linspace(1e-4, 1 - 1e-4, 300)
        pc.line("bsc", x, it.bsc_capacity(x), color=NAVY, width=2.0, name="BSC 1 − h(p)")
        pc.line("bec", x, it.bec_capacity(x), color=GREEN, width=2.0, name="BEC 1 − ε")
        pc.line("z", x, it.z_capacity(x), color=ORANGE, width=2.0, name="Z-channel")
        if p.ch != "Custom":
            pc.scatter("now", [p.pp], [C], color=RED, size=13, symbol="d", outline=RED)
        unif = it.dmc_mutual_info(np.full(len(popt), 1 / len(popt)), W)
        self.readout(C=C, pstar=float(popt[1]), unif=unif, iters=n6)

    @staticmethod
    def ba_history(W, n):
        """Blahut–Arimoto bounds per iteration, started from a lopsided input so that the
        convergence is visible even for symmetric channels."""
        p = np.r_[0.85, np.full(W.shape[0] - 1, 0.15 / (W.shape[0] - 1))]
        lo, up = [], []
        for _ in range(n):
            q = p @ W
            with np.errstate(divide="ignore", invalid="ignore"):
                D = np.nansum(np.where(W > 0, W * np.log2(W / q[None, :]), 0), axis=1)
            lo.append(float(np.sum(p * D)))
            up.append(float(D.max()))
            p = p * 2.0 ** D
            p /= p.sum()
        return np.array(lo), np.array(up)

    def draw_diagram(self, W):
        pd = self.plot("diag")
        pd.pi.hideAxis("left")
        pd.pi.hideAxis("bottom")
        yin = {0: 1.0, 1: 0.0, 2: 0.5}
        yout = [1.0, 0.5, 0.0]
        names = ["0", "?", "1"]
        nin = W.shape[0]
        for i in range(nin):
            for j in range(3):
                w = W[i, j]
                key = f"w{i}{j}"
                if w > 1e-6:
                    col = GREEN if (i == 0 and j == 0) or (i == 1 and j == 2) else (
                        ORANGE if j == 1 else RED)
                    if i == 2:
                        col = GRAY
                    pd.line(key, [0, 1], [yin[i], yout[j]], color=col, width=1 + 9 * w, alpha=0.85)
                    t = 0.68
                    pd.text("t" + key, t, yin[i] + t * (yout[j] - yin[i]) + 0.04, f"{w:.3f}",
                            color=col, size=9, bold=True, anchor=(0.5, 0.5), fill=True)
        pd.scatter("in", [0] * nin, [yin[i] for i in range(nin)], color=NAVY, size=26)
        pd.scatter("out", [1, 1, 1], yout, color=NAVY, size=26)
        for i in range(nin):
            pd.text(f"ni{i}", -0.08, yin[i], f"X = {'01u'[i]}", color=NAVY if i < 2 else GRAY,
                    size=10, bold=True, anchor=(1, 0.5))
        for j in range(3):
            pd.text(f"no{j}", 1.08, yout[j], f"Y = {names[j]}", color=NAVY, size=10, bold=True,
                    anchor=(0, 0.5))
        pd.set_xlim(-0.45, 1.45)
        pd.set_ylim(-0.2, 1.2)

    def story(self, p):
        C, ps = self.r.get("C", 0), self.r.get("pstar", 0.5)
        s = (f"<p>A <b>discrete memoryless channel</b> is just the table of arrows on the left. Its "
             f"capacity is the most mutual information you can get through it by choosing how "
             f"often to send 0 and 1: here {v(C, '.4f', 'bits per use')}, reached at "
             f"P(X = 1) = {v(ps, '.3f')} (red diamond, top right).</p>")
        if p.ch == "BSC":
            s += (f"<p>The BSC is symmetric, so equal inputs are optimal and C = 1 − h(p). "
                  f"At p = 0.5 the output is independent of the input: nothing gets through.</p>")
        elif p.ch == "BEC":
            s += ("<p>The erasure channel is kinder: the receiver knows which symbols it lost, "
                  "and C = 1 − ε exactly, the fraction that arrives. Every bit counts.</p>")
        elif p.ch == "Z-channel":
            s += ("<p>The Z-channel only corrupts ones (like an optical receiver that misses "
                  "photons but never invents them). Sending fewer ones is wiser, so the optimal "
                  "input is <i>not</i> equal: a rare case where uniform signalling is not "
                  "capacity-achieving.</p>")
        else:
            s += ("<p>Your custom channel mixes flips and erasures. There is no closed form in "
                  "general, but the algorithm does not care.</p>")
        if p.useless and getattr(self, "pu", None) is not None:
            s += (f"<p>The extra input u produces the average of the other two output "
                  f"distributions: it tells the receiver nothing it could not get by flipping a coin "
                  f"between 0 and 1. Blahut–Arimoto starves it: P*(u) = {v(self.pu, '.3f')}, and the "
                  f"capacity is unchanged.</p>")
        s += (f"<p><b>Blahut–Arimoto</b> (bottom left) alternates between the best output "
              f"distribution and the best input, with a lower and an upper bound that squeeze "
              f"the capacity: here to 10⁻⁶ in {v(self.r.get('iters', 0), 'd')} iterations.</p>")
        return "<h3>What a noisy channel can carry</h3>" + s + keybox(
            "C = max over p(x) of I(X;Y). Below C, coding makes errors vanish; above it, nothing can.")


# =============================================================================== 5. AWGN limit
class ShannonLimit(Experiment):
    title = "Shannon's AWGN limit"
    blurb = "Power, bandwidth and rate: what is possible, and the −1.59 dB wall."
    book = "sec:ch13:awgn"
    controls = [
        Slider("pn0", "Received P/N0", 40, 100, 70, step=0.5, unit="dB-Hz",
               help="P/N0 sets the capacity with unlimited bandwidth: 1.44 × P/N0 bits per second"),
        LogSlider("bw", "Bandwidth", 0.001, 100.0, 1.0, unit="MHz", fmt=".3g"),
        LogSlider("rate", "Your data rate", 0.001, 1000.0, 2.0, unit="Mb/s", fmt=".3g"),
    ]
    plots = [
        Plot("cb", "Capacity vs bandwidth at your P/N0", x="bandwidth (MHz)", y="capacity (Mb/s)",
             logx=True, xlim=(0.001, 100), legend="tl"),
        Plot("plane", "The bandwidth–power plane", x="Eb/N0 (dB)",
             y="spectral efficiency (b/s/Hz)", logy=True, xlim=(-3, 30), ylim=(0.01, 30),
             legend="br"),
    ]
    layout = [["cb", "plane"]]
    readouts = [
        Readout("snr", "SNR in your bandwidth", "dB", ".1f"),
        Readout("C", "Capacity", "Mb/s", ".3g"),
        Readout("ebn0", "Your Eb/N0", "dB", ".2f"),
        Readout("margin", "Margin over Shannon", "dB", ".2f", good=lambda x: x >= 0,
                help="Your Eb/N0 minus the minimum Eb/N0 at your spectral efficiency"),
    ]
    challenges = [
        Challenge("Spectral efficiency 10: carry 10 Mb/s in 1 MHz with a margin between 0 and 1 dB.",
                  lambda s: abs(s.p.rate / 10 - 1) < 0.03 and abs(s.p.bw - 1) < 0.03
                  and 0 <= s.r.margin <= 1,
                  hint="Set the rate and bandwidth first, then find the P/N0 that just works."),
        Challenge("Approach the wall: a working link (margin ≥ 0) with Eb/N0 below −0.6 dB.",
                  lambda s: s.r.margin >= 0 and s.r.ebn0 < -0.6,
                  hint="Very low spectral efficiency: lots of bandwidth for the rate."),
        Challenge("Find a bandwidth so generous that doubling it adds less than 5 % capacity.",
                  lambda s: s.exp.dbl < 0.05),
    ]

    def setup(self):
        self.dbl = 1.0

    def update(self, p):
        P = undb(p.pn0)                          # W/Hz ratio, i.e. P/N0 in Hz
        B = p.bw * 1e6
        R = p.rate * 1e6
        snr = P / B
        C = B * np.log2(1 + snr)
        self.dbl = (2 * B * np.log2(1 + snr / 2)) / C - 1
        cinf = P * np.log2(np.e)
        b = np.logspace(-3, 2, 300)
        pc = self.plot("cb")
        cb = b * 1e6 * np.log2(1 + P / (b * 1e6)) / 1e6
        pc.line("c", b, cb, color=NAVY, width=2.4, name="C = B log₂(1 + P/N0B)")
        pc.hline("inf", cinf / 1e6, color=GREEN, style="--", width=1.4,
                 label=f"limit 1.44·P/N0 = {cinf / 1e6:.3g} Mb/s", label_pos=0.05)
        pc.hline("R", p.rate, color=RED, style=":", width=1.4, label="your rate", label_pos=0.82)
        pc.vline("B", p.bw, color=GRAY, style=":", width=1.2)
        pc.scatter("now", [p.bw], [C / 1e6], color=RED if C < R else GREEN, size=13, symbol="d",
                   outline=None, name="your bandwidth")
        pc.set_ylim(0, max(cinf / 1e6, p.rate) * 1.25)
        # plane
        eta = np.logspace(-2, np.log10(30), 300)
        pp = self.plot("plane")
        lim = it.shannon_ebn0_db(eta)
        pp.fill_between("ok", lim, np.full_like(eta, 0.01), eta, color=GREEN, alpha=0.10)
        pp.line("lim", lim, eta, color=NAVY, width=2.4, name="Shannon limit")
        pp.vline("wall", -1.59, color=RED, style="--", width=1.3, label="−1.59 dB", label_pos=0.9)
        pts = [(9.6, 1, "BPSK"), (9.6, 2, "QPSK"), (13.4, 4, "16-QAM"), (17.8, 6, "64-QAM")]
        pp.scatter("unc", [a for a, _, _ in pts], [b_ for _, b_, _ in pts], color=GRAY, size=9,
                   symbol="s", name="uncoded, BER 10⁻⁵ (approx.)")
        for a, b_, n in pts:
            pp.text("u" + n, a + 0.4, b_, n, color=GRAY, size=8, anchor=(0, 0.5))
        eta_u = R / B
        ebn0 = p.pn0 - db(R)
        lim_u = (it.shannon_ebn0_db(eta_u) if eta_u < 500 else
                 eta_u * 10 * np.log10(2) - 10 * np.log10(eta_u))     # avoid 2^η overflow
        margin = float(ebn0 - lim_u)
        pp.scatter("you", [ebn0], [eta_u], color=GREEN if margin >= 0 else RED, size=15,
                   symbol="d", name="you")
        self.readout(snr=float(db(snr)), C=C / 1e6, ebn0=float(ebn0), margin=margin)

    def story(self, p):
        m = self.r.get("margin", 0)
        eta = p.rate / p.bw
        s = (f"<p>Your receiver collects P/N0 = {v(p.pn0, '.1f', 'dB-Hz')}. In "
             f"{v(p.bw, '.3g', 'MHz')} that is an SNR of {v(self.r.get('snr', 0), '.1f', 'dB')} and "
             f"a capacity of {v(self.r.get('C', 0), '.3g', 'Mb/s')}. You ask for "
             f"{v(p.rate, '.3g', 'Mb/s')}: a spectral efficiency of {v(eta, '.3g', 'b/s/Hz')}. ")
        s += ((good("Possible") + f", with {v(m, '.2f', 'dB')} to spare.") if m >= 0 else
              (bad("Impossible") + f": you are {v(-m, '.2f', 'dB')} short. No code, however clever, "
               "will do it."))
        s += "</p>"
        s += ("<p>Left: more bandwidth always helps, but less and less. With unlimited bandwidth "
              "the capacity tends to 1.44·P/N0, because the SNR per hertz falls as fast as the "
              "hertz arrive. Right: every working system lives in the green region. The gap "
              "between the uncoded squares and the curve is what channel coding is for: about "
              "10 dB at BPSK, which turbo and LDPC codes recover to within a fraction of a dB.</p>")
        return "<h3>Possible and impossible</h3>" + s + keybox(
            "C = B log₂(1 + P/N0B). Reliable communication needs Eb/N0 &gt; (2^η − 1)/η ≥ −1.59 dB.")


# =============================================================================== 6. constrained
CONS = {"BPSK": "bpsk", "QPSK": "qpsk", "8-PSK": "8psk", "16-QAM": "16qam", "64-QAM": "64qam",
        "256-QAM": "256qam"}
SD = np.linspace(-10, 35, 91)


@lru_cache(maxsize=16)
def cm_curve(name):
    """Coded-modulation (constrained-input) capacity on the SD grid (bits per symbol)."""
    con = cl.get_constellation(CONS[name])
    g = undb(SD)
    if name == "BPSK":
        return np.array([it.mi_pam([-1.0, 1.0], 2 * x) for x in g])
    if name == "8-PSK":
        rng = np.random.default_rng(1)
        cm = np.array([it.mi_2d_mc(con, x, n=3000, rng=rng) for x in g])
        return np.maximum.accumulate(np.clip(cm, 0, con.k))
    return np.array([it.qam_cm(con.M, x) for x in g])


@lru_cache(maxsize=16)
def bicm_curve(name, natural=False):
    """BICM capacity (square QAM only; BPSK = CM; None for PSK)."""
    if name == "BPSK":
        return cm_curve(name)
    if name == "8-PSK":
        return None
    con = cl.get_constellation(CONS[name])
    return np.array([it.qam_bicm(con.M, x, natural=natural) for x in undb(SD)])


def cap_curves(name, natural=False):
    """(CM, BICM or None, hard-decision or None) capacity curves on SD for a constellation."""
    g = undb(SD)
    hard = None
    if name == "BPSK":
        hard = 1 - it.hb(cl.qfunc(np.sqrt(2 * g)))
    elif name == "QPSK":
        hard = 2 * (1 - it.hb(cl.qfunc(np.sqrt(g))))
    return cm_curve(name), bicm_curve(name, natural), hard


def snr_for(curve, rate):
    if curve is None or rate >= curve.max() * 0.999 or rate <= curve.min():
        return None
    return float(np.interp(rate, curve, SD))


class Constrained(Experiment):
    title = "Constrained capacity"
    blurb = "Real transmitters send PSK and QAM, not Gaussian noise. What does that cost?"
    book = "sec:ch13:constrained"
    controls = [
        Choice("con", "Constellation", list(CONS), "QPSK", style="menu"),
        Choice("lab", "Bit labels (for BICM)", ["Gray", "Natural"],
               help="BICM decodes each bit separately; its capacity depends on the labelling"),
        Toggle("others", "Show the other constellations", False),
        Slider("snr", "Es/N0", -10.0, 35.0, 8.0, step=0.25, unit="dB"),
        Slider("rate", "Target rate", 0.25, 7.5, 1.0, step=0.05, unit="bits/symbol"),
    ]
    plots = [
        Plot("cap", "Capacity vs SNR: Gaussian input, coded modulation (CM) and BICM",
             x="Es/N0 (dB)", y="bits per symbol", xlim=(-10, 35), ylim=(0, 9.2), legend="tl"),
        Plot("gap", "Extra SNR over Shannon to reach a rate", x="rate (bits per symbol)",
             y="extra Es/N0 (dB)", xlim=(0, 8), ylim=(0, 4), legend="tr"),
        ConstellationPlot("pts", "Constellation and bit labels", lim=1.55),
    ]
    layout = [["cap", "cap"], ["gap", "pts"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("cm", "CM capacity at your SNR", "bits", ".3f"),
        Readout("bicm", "BICM capacity", "bits", ".3f"),
        Readout("need", "Es/N0 for the target", "dB", ".2f"),
        Readout("gap", "Gap to Shannon", "dB", ".2f", good=lambda x: x < 0.5),
    ]
    challenges = [
        Challenge("See why QPSK is not used at high rates: ask it for 1.9 bits/symbol or more and "
                  "find a gap to Shannon above 2.5 dB.",
                  lambda s: s.p.con == "QPSK" and s.p.rate >= 1.89 and s.r.gap > 2.5),
        Challenge("16-QAM: find a target rate where natural labels cost BICM a full decibel "
                  "or more over Gray labels.",
                  lambda s: s.p.con == "16-QAM" and s.exp.lab_loss is not None
                  and s.exp.lab_loss >= 1.0,
                  hint="Switch the labels and compare the Es/N0 needed; try rates around 2."),
        Challenge("Soft beats hard: with BPSK, find the Es/N0 where soft decisions gain the most "
                  "over hard decisions (at least 0.115 bit).",
                  lambda s: s.p.con == "BPSK" and s.exp.soft_gain >= 0.115,
                  hint="Watch the gap between the navy and the dotted orange curve."),
    ]

    def setup(self):
        self.lab_loss = None
        self.soft_gain = 0.0

    def update(self, p):
        con = cl.get_constellation(CONS[p.con])
        cm, bi, hard = cap_curves(p.con, p.lab == "Natural")
        pc = self.plot("cap")
        pc.line("sh", SD, np.log2(1 + undb(SD)), color=RED, width=2.0, name="Shannon (Gaussian input)")
        if p.others:
            for i, nm in enumerate(CONS):
                if nm != p.con:
                    pc.line("o" + nm, SD, cm_curve(nm), color=GRAY, width=1.0, alpha=0.7,
                            name="others (CM)" if i == 0 or (i == 1 and p.con == "BPSK") else None)
        pc.line("cm", SD, cm, color=NAVY, width=2.6, name=f"{p.con}, CM")
        if bi is not None and p.con not in ("BPSK",):
            pc.line("bi", SD, bi, color=GREEN, width=2.0, style="--", name=f"BICM, {p.lab} labels")
        if hard is not None:
            pc.line("hd", SD, hard, color=ORANGE, width=1.8, style=":", name="hard decisions")
        pc.vline("snr", p.snr, color=GRAY, style=":", width=1.2)
        pc.hline("rate", p.rate, color=PURPLE, style=":", width=1.2, label="target", label_pos=0.95)
        g = undb(p.snr)
        cm_now = float(np.interp(p.snr, SD, cm))
        bi_now = float(np.interp(p.snr, SD, bi)) if bi is not None else None
        pc.scatter("now", [p.snr], [cm_now], color=RED, size=12, symbol="d")
        # gap vs rate
        pg = self.plot("gap")
        ok = (cm > 0.03) & (cm < 0.985 * con.k)
        pg.line("cm", cm[ok], SD[ok] - db(2 ** cm[ok] - 1), color=NAVY, width=2.2, name="CM")
        if bi is not None and p.con != "BPSK":
            okb = (bi > 0.03) & (bi < 0.985 * con.k)
            pg.line("bi", bi[okb], SD[okb] - db(2 ** bi[okb] - 1), color=GREEN, width=2.0,
                    style="--", name="BICM")
        pg.hline("ug", db(np.pi * np.e / 6), color=RED, style=":", width=1.2,
                 label="shaping gap πe/6 = 1.53 dB", label_pos=0.03)
        need = snr_for(cm, p.rate)
        gap = None if need is None else need - float(db(2 ** p.rate - 1))
        if gap is not None:
            pg.scatter("you", [p.rate], [gap], color=RED, size=13, symbol="d")
        pg.set_xlim(0, con.k + 0.3)
        # label loss and soft gain
        self.lab_loss = None
        if p.con == "16-QAM":
            ng = snr_for(bicm_curve(p.con, False), p.rate)
            nn = snr_for(bicm_curve(p.con, True), p.rate)
            if ng is not None and nn is not None:
                self.lab_loss = nn - ng
        self.soft_gain = (cm_now - float(np.interp(p.snr, SD, hard))) if (
            hard is not None and p.con == "BPSK") else 0.0
        # constellation
        pp = self.plot("pts")
        pts = con.points
        pp.ideal("pts", pts, color=NAVY, size=10 if con.M <= 64 else 6)
        if con.M <= 16:
            labels = self.labels(con, p.lab == "Natural")
            for i, z in enumerate(pts):
                pp.text(f"l{i}", z.real, z.imag + 0.04, labels[i], color=ORANGE if con.M > 16 else RED,
                        size=7 if con.M > 16 else 8.5, anchor=(0.5, 1))
        lim = 1.55 if con.M > 2 else 1.4
        pp.set_xlim(-lim, lim)
        pp.set_ylim(-lim, lim)
        self.readout(cm=cm_now, bicm=bi_now if bi_now is not None else "n/a",
                     need=need if need is not None else "—", gap=gap if gap is not None else "—")

    @staticmethod
    def labels(con, natural):
        k = con.k
        if not natural or "psk" in con.name.lower() or con.M <= 2:
            return [format(int(l), f"0{k}b") for l in con.labels]
        L = int(round(np.sqrt(con.M)))
        lev = np.sort(np.unique(np.round(con.points.real, 6)))
        out = []
        for z in con.points:
            i = int(np.argmin(np.abs(lev - z.real)))
            q = int(np.argmin(np.abs(lev - z.imag)))
            out.append(format(i, f"0{k // 2}b") + format(q, f"0{k // 2}b"))
        return out

    def story(self, p):
        cm, gap = self.r.get("cm", 0), self.r.get("gap")
        sh = np.log2(1 + undb(p.snr))
        s = (f"<p>At {v(p.snr, '.1f', 'dB')} a Gaussian-input code could carry "
             f"{v(sh, '.2f', 'bits')} per symbol; {p.con} with ideal coding carries "
             f"{v(cm, '.2f', 'bits')}. A constellation can never carry more than its "
             f"{v(int(np.log2(cl.get_constellation(CONS[p.con]).M)), 'd')} bits, so each curve "
             f"bends over and saturates.</p>")
        if p.con == "16-QAM" and self.lab_loss is not None:
            s += (f"<p>At this target rate, BICM with natural labels needs "
                  f"{v(self.lab_loss, '.2f', 'dB')} more than with Gray labels.</p>")
        if isinstance(gap, float):
            s += (f"<p>To reach {v(p.rate, '.2f', 'bits/symbol')} you need "
                  f"{v(self.r.get('need'), '.2f', 'dB')}, {v(gap, '.2f', 'dB')} more than Shannon. "
                  "Run a constellation at about half to three quarters of its bits and the gap "
                  "is small; push it near its maximum and the gap explodes. That is why adaptive "
                  "modulation switches to a denser constellation long before the old one "
                  "saturates.</p>")
        else:
            s += "<p>Your target rate is beyond what this constellation can carry.</p>"
        s += ("<p>At high rates even a perfect square QAM stays 1.53 dB from Shannon: its points "
              "are uniform, not Gaussian. <b>Probabilistic shaping</b> (5G NR, 800G optics) "
              "recovers most of that. BICM with Gray labels, what every modern standard does, "
              "costs almost nothing; natural labels cost a lot.</p>")
        return "<h3>The price of a constellation</h3>" + s + keybox(
            "Choose a constellation with 1–2 bits more than your code rate needs; label it Gray.")


# =============================================================================== 7. finite blocklength
class FiniteBlocklength(Experiment):
    title = "Finite blocklength"
    blurb = "Shannon's capacity needs infinitely long codes. Short packets pay a price."
    book = "sec:ch13:fbl"
    controls = [
        LogSlider("n", "Block length n", 50, 100000, 1000, unit="uses", fmt=".0f",
                  help="Real channel uses per codeword (a complex symbol is two)"),
        IntSlider("lg", "Block error rate ε = 10^x, x =", -9, -1, -5),
        Slider("snr", "SNR", -5.0, 20.0, 0.0, step=0.25, unit="dB"),
    ]
    plots = [
        Plot("rate", "Highest rate at block length n (normal approximation)",
             x="block length n (real channel uses)", y="rate (bits per use)", logx=True,
             xlim=(30, 1e5), legend="br"),
        Plot("need", "Block length needed to reach a fraction of capacity",
             x="fraction of capacity (%)", y="block length n", logy=True, xlim=(50, 99),
             ylim=(10, 1e7), legend="tl"),
        Plot("pen", "Extra SNR to send k bits at rate ½ per use", x="information bits k",
             y="extra SNR over Shannon (dB)", logx=True, xlim=(30, 3e4), ylim=(0, 6), legend="tr"),
    ]
    layout = [["rate", "rate"], ["need", "pen"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("C", "Capacity", "bits/use", ".3f"),
        Readout("R", "Best rate at n", "bits/use", ".3f"),
        Readout("frac", "Fraction of capacity", "%", ".1f", good=lambda x: x > 90),
        Readout("pen", "SNR penalty", "dB", ".2f"),
    ]
    challenges = [
        Challenge("URLLC: n ≤ 500, ε ≤ 10⁻⁵, and still at least 75 % of capacity.",
                  lambda s: s.p.n <= 500 and s.p.lg <= -5 and s.r.frac >= 75,
                  hint="The fraction improves with SNR: at high SNR the dispersion stops growing "
                       "while capacity keeps climbing."),
        Challenge("Find where ε = 10⁻⁹ costs about 1 dB of SNR (0.9–1.1 dB).",
                  lambda s: s.p.lg == -9 and 0.9 <= s.r.pen <= 1.1),
        Challenge("Get within 2 % of capacity with ε ≤ 10⁻³ at an SNR of 0 dB or below.",
                  lambda s: s.p.lg <= -3 and s.p.snr <= 0 and s.r.frac >= 98),
    ]

    def update(self, p):
        g = undb(p.snr)
        eps = 10.0 ** p.lg
        C = 0.5 * np.log2(1 + g)
        nn = np.logspace(np.log10(30), 5, 200)
        pr = self.plot("rate")
        for e, col in [(1e-1, GREEN), (1e-3, BLUE), (1e-5, ORANGE), (1e-9, PURPLE)]:
            if abs(np.log10(e) - p.lg) > 0.1:
                pr.line(f"e{e}", nn, np.maximum(it.normal_approx_awgn(nn, g, e), 0), color=col,
                        width=1.0, alpha=0.6, name=f"ε = {e:g}")
        pr.line("cur", nn, np.maximum(it.normal_approx_awgn(nn, g, eps), 0), color=NAVY, width=2.6,
                name=f"ε = 10^{p.lg}")
        pr.hline("C", C, color=RED, style="--", width=1.5, label=f"capacity {C:.3f}", label_pos=0.05)
        R = float(it.normal_approx_awgn(p.n, g, eps))
        pr.scatter("now", [p.n], [max(R, 0)], color=RED, size=13, symbol="d")
        pr.set_ylim(0, C * 1.15)
        # n needed for a fraction of capacity
        V = g * (g + 2) / (2 * (g + 1) ** 2) * it.LOG2E ** 2
        from scipy.stats import norm
        f = np.linspace(0.5, 0.99, 100)
        pn = self.plot("need")
        for e, col in [(1e-1, GREEN), (1e-3, BLUE), (1e-5, ORANGE), (1e-9, PURPLE)]:
            if abs(np.log10(e) - p.lg) > 0.1:
                pn.line(f"e{e}", 100 * f, V * norm.isf(e) ** 2 / (C * (1 - f)) ** 2, color=col,
                        width=1.0, alpha=0.6)
        pn.line("cur", 100 * f, V * norm.isf(eps) ** 2 / (C * (1 - f)) ** 2, color=NAVY, width=2.4,
                name="n ≈ V·Q⁻¹(ε)² / (C(1−f))²")
        frac = 100 * max(R, 0) / C
        if 50 <= frac <= 99:
            pn.scatter("now", [frac], [p.n], color=RED, size=12, symbol="d")
        # penalty vs k at rate 1/2
        ks = np.logspace(np.log10(30), np.log10(3e4), 30)
        pp = self.plot("pen")
        pp.line("p", ks, [it.fbl_snr_penalty_db(2 * k, 0.5, eps) for k in ks], color=NAVY,
                width=2.2, name=f"ε = 10^{p.lg}")
        pp.line("p1", ks, [it.fbl_snr_penalty_db(2 * k, 0.5, 1e-1) for k in ks], color=GREEN,
                width=1.0, alpha=0.7, name="ε = 0.1")
        k_now = p.n / 2
        if 30 <= k_now <= 3e4:
            pp.vline("k", k_now, color=GRAY, style=":", width=1.2, label="k = n/2", label_pos=0.9)
        pen = float(p.snr - db(2 ** (2 * R) - 1)) if R > 0 else float("nan")
        self.readout(C=C, R=R, frac=frac, pen=pen if np.isfinite(pen) else "—")

    def story(self, p):
        frac = self.r.get("frac", 0)
        s = (f"<p>A codeword of {v(p.n, '.0f')} channel uses at {v(p.snr, '.1f', 'dB')} SNR, "
             f"decoded wrongly no more than once in 10^{v(-p.lg, 'd')} tries, can carry at most "
             f"{v(self.r.get('R', 0), '.3f')} bits per use: {v(frac, '.1f', '%')} of capacity. "
             f"The price is about √(V/n)·Q⁻¹(ε): it shrinks only with the square root of the "
             f"block length, and grows as you demand fewer errors.</p>"
             "<p>Bottom left: to get within a few percent of capacity you need blocks of tens "
             "of thousands of symbols. Bottom right: a 100-bit control message pays 2 to 3 dB, "
             "which is why 5G uses polar codes and why ultra-reliable low-latency links "
             "(URLLC) are hard.</p>")
        return "<h3>Short packets, high price</h3>" + s + keybox(
            "R*(n, ε) ≈ C − √(V/n)·Q⁻¹(ε): the normal approximation of Polyanskiy, Poor and Verdú (2010).")


# =============================================================================== 8. water-filling
G_DEFAULT = [22, 12, 25, 4, 15, -4]


class WaterFilling(Experiment):
    title = "Water-filling"
    blurb = "Pour power into sub-channels like water into a vessel with an uneven floor."
    book = "sec:ch13:waterfill"
    controls = ([Choice("mode", "Channel", ["6 sub-channels", "DSL line (256 tones)"]),
                 Slider("ptot", "Total power", -15.0, 30.0, 5.0, step=0.5, unit="dB",
                        enabled_if=lambda p: p.mode == "6 sub-channels"),
                 Slider("gap", "SNR gap Γ for bit loading", 0.0, 12.0, 0.0, step=0.2, unit="dB",
                        help="Real codes need Γ more SNR than capacity; 9.8 dB is uncoded QAM at 10⁻⁷"),
                 Heading("Sub-channel gain-to-noise (6-channel mode)")] +
                [Slider(f"g{i}", f"Sub-channel {i + 1}", -10.0, 30.0, G_DEFAULT[i], step=0.5,
                        unit="dB", enabled_if=lambda p: p.mode == "6 sub-channels")
                 for i in range(6)] +
                [Heading("DSL line"),
                 Slider("loop", "Loop loss", 20.0, 120.0, 78.0, step=1.0, unit="dB/√MHz",
                        enabled_if=lambda p: p.mode != "6 sub-channels"),
                 Slider("ingress", "AM radio ingress", 0.0, 50.0, 25.0, step=1.0, unit="dB",
                        enabled_if=lambda p: p.mode != "6 sub-channels"),
                 Slider("psd", "Transmit PSD", -60.0, -30.0, -40.0, step=1.0, unit="dBm/Hz",
                        enabled_if=lambda p: p.mode != "6 sub-channels")])
    plots = [
        Plot("vessel", "The vessel: drag a gray floor (●) · blue water = power poured",
             x="sub-channel", y="power (linear units)", legend=None),
        BarPlot("bits", "Bits: gray equal · navy water-filling · ■ integer",
                x="sub-channel", y="bits per symbol", legend=None),
        Plot("curve", "Total capacity vs total power", x="total power (dB)", y="bits per symbol",
             legend="tl"),
    ]
    layout = [["vessel", "vessel"], ["bits", "curve"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("cwf", "Water-filling capacity", "", ".2f"),
        Readout("ceq", "Equal power", "", ".2f"),
        Readout("gain", "Water-filling gain", "%", ".1f"),
        Readout("used", "Sub-channels in use", "", None),
    ]
    challenges = [
        Challenge("Starve the weak: find a total power at which water-filling uses only 3 of the "
                  "6 sub-channels.",
                  lambda s: s.p.mode == "6 sub-channels" and s.exp.n_used == 3),
        Challenge("Make water-filling worth it: beat equal power by more than 40 %.",
                  lambda s: s.p.mode == "6 sub-channels" and s.r.gain > 40,
                  hint="Low total power, and a few sub-channels much better than the rest."),
        Challenge("High-SNR lesson: find a setting where water-filling gains less than 1 %, with "
                  "all 6 sub-channels in use.",
                  lambda s: s.p.mode == "6 sub-channels" and s.r.gain < 1 and s.exp.n_used == 6),
        Challenge("DSL: at a PSD of −40 dBm/Hz or less, find a loop with loss of at least "
                  "56 dB/√MHz that still delivers 15 Mb/s.",
                  lambda s: s.p.mode != "6 sub-channels" and s.p.psd <= -40 and s.p.loop >= 56
                  and s.r.cwf > 15),
    ]

    def setup(self):
        self.n_used = 0
        self._top = None

    def on_drag(self, key, item, i, x, y):
        """Drag a sub-channel's floor N/|H|²: a lower floor is a better channel."""
        if key != "vessel" or item != "grab":
            return False
        g_db = -10 * np.log10(max(y, 1e-3))
        g_db = float(np.clip(np.round(g_db * 2) / 2, -10.0, 30.0))
        self.set_control(f"g{i}", g_db, refresh=False)
        return True

    def update(self, p):
        if p.mode == "6 sub-channels":
            self.eight(p)
        else:
            self.dsl(p)

    def eight(self, p):
        g = undb([p[f"g{i}"] for i in range(6)])
        P = float(undb(p.ptot))
        inv = 1 / g
        pw, mu = it.waterfill(inv, P)
        self.n_used = int(np.sum(pw > 1e-12))
        c_wf = np.log2(1 + pw * g)
        c_eq = np.log2(1 + P / 6 * g)
        G = float(undb(p.gap))
        bits = np.floor(np.log2(1 + pw * g / G))
        x = np.arange(1, 7)
        pv = self.plot("vessel")
        pv.set_labels(x="sub-channel", y="power (linear units)")
        top = 1.6 * mu
        if self.dragging and self._top:          # hold the scale still while a floor is dragged
            top = self._top
        self._top = top
        floor = np.minimum(inv, top * 1.2)
        pv.bars("floor", x, floor, width=0.92, color=GRAY)
        pv.bars("water", x, floor + pw, base=floor, width=0.92, color=BLUE, alpha=0.55)
        pv.hline("mu", mu, color=NAVY, style="--", width=1.8, label=f"water level μ = {mu:.3g}",
                 label_pos=0.02)
        pv.handles("grab", x, np.minimum(inv, 0.97 * top), color=NAVY, size=13, axis="y")
        for i in range(6):
            if pw[i] > 0.04 * top:
                pv.text(f"t{i}", x[i], floor[i] + pw[i] / 2, f"{pw[i]:.2g}", color=NAVY, size=8.5,
                        anchor=(0.5, 0.5))
        pv.set_xlim(0.4, 6.6)
        pv.set_ylim(0, top)
        pv.set_xticks([(k, str(k)) for k in x])
        pb = self.plot("bits")
        pb.set_labels(x="sub-channel", y="bits per symbol")
        pb.bars("eq", x - 0.2, c_eq, width=0.38, color=GRAY)
        pb.bars("wf", x + 0.2, c_wf, width=0.38, color=NAVY)
        pb.scatter("int", x + 0.2, bits, color=ORANGE, size=11, symbol="s",
                   name=f"integer bits, Γ = {p.gap:.1f} dB")
        pb.set_xticks([(k, str(k)) for k in x])
        pb.set_xlim(0.4, 6.6)
        pb.set_ylim(0, max(3, c_wf.max(), c_eq.max()) * 1.3)
        pc = self.plot("curve")
        pc.set_labels(x="total power (dB)", y="bits per symbol (all 6)")
        pp = np.linspace(-15, 30, 91)
        cw = [np.sum(np.log2(1 + it.waterfill(inv, undb(q))[0] * g)) for q in pp]
        ce = [np.sum(np.log2(1 + undb(q) / 6 * g)) for q in pp]
        pc.line("wf", pp, cw, color=NAVY, width=2.2, name="water-filling")
        pc.line("eq", pp, ce, color=GRAY, width=2.0, style="--", name="equal power")
        pc.scatter("now", [p.ptot], [c_wf.sum()], color=RED, size=12, symbol="d")
        pc.set_xlim(-15, 30)
        pc.set_ylim(0, max(cw) * 1.08)
        C1, C0 = float(c_wf.sum()), float(c_eq.sum())
        self._st = dict(mu=mu, n=self.n_used, bits=int(bits.sum()))
        self.readout(cwf=C1, ceq=C0, gain=100 * (C1 / C0 - 1) if C0 > 1e-9 else 0.0,
                     used=f"{self.n_used} of 6")

    def dsl(self, p):
        K, df = 256, 4312.5
        fr = (np.arange(K) + 0.5) * df
        att = 10 + p.loop * np.sqrt(fr / 1e6) + 4 * fr / 1e6
        Hg = undb(-att)
        N = undb(-140 - 30) * np.ones(K)
        N[(fr > 650e3) & (fr < 700e3)] *= undb(p.ingress)
        N = N + undb(-150 - 30) * (fr / 1e6) ** 1.5 * 3
        Psd = undb(p.psd - 30) * df
        inv = N * df / Hg
        pw, mu = it.waterfill(inv, Psd * K)
        snr_flat, snr_wf = Psd / inv, pw / inv
        G = float(undb(max(p.gap, 0)))
        b = it.gap_bit_loading(snr_wf, p.gap)
        sym_rate = 4000
        f_m = fr / 1e6
        pv = self.plot("vessel")
        pv.set_labels(x="frequency (MHz)", y="power per tone (relative)")
        top = 1.6 * mu
        pv.line("d_floor", f_m, np.minimum(inv, 2 * top), color=GRAY, width=1.6, fill=0,
                fill_alpha=0.35, name="floor N/|H|²")
        pv.fill_between("d_water", f_m, np.minimum(inv, top), np.maximum(np.minimum(inv, top), mu),
                        color=BLUE, alpha=0.4)
        pv.hline("d_mu", mu, color=NAVY, style="--", width=1.8, label="water level μ", label_pos=0.02)
        pv.band("d_am", 0.65, 0.70, color=RED, alpha=0.12, label="AM ingress")
        pv.set_xlim(0, 1.11)
        pv.set_ylim(0, top)
        pv.set_xticks([(q, f"{q:.1f}") for q in np.arange(0, 1.11, 0.2)])
        pb = self.plot("bits")
        pb.set_labels(x="frequency (MHz)", y="bits per tone")
        pb.line("d_cap", f_m, np.log2(1 + snr_wf), color=NAVY, width=1.4, name="capacity per tone")
        pb.line("d_int", f_m, b, color=ORANGE, width=1.6, step=False,
                name=f"loaded bits, Γ = {p.gap:.1f} dB")
        pb.set_xlim(0, 1.11)
        pb.set_ylim(0, 22)
        pb.set_xticks([(q, f"{q:.1f}") for q in np.arange(0, 1.11, 0.2)])
        pc = self.plot("curve")
        pc.set_labels(x="transmit PSD (dBm/Hz)", y="data rate (Mb/s)")
        pp = np.linspace(-60, -30, 31)
        cw, ce = [], []
        for q in pp:
            ps_ = undb(q - 30) * df
            w_, _ = it.waterfill(inv, ps_ * K)
            cw.append(np.sum(np.log2(1 + w_ / inv)) * sym_rate / 1e6)
            ce.append(np.sum(np.log2(1 + ps_ / inv)) * sym_rate / 1e6)
        pc.line("d_wf", pp, cw, color=NAVY, width=2.2, name="water-filling")
        pc.line("d_eq", pp, ce, color=GRAY, width=2.0, style="--", name="flat PSD")
        C1 = float(np.sum(np.log2(1 + snr_wf)) * sym_rate / 1e6)
        C0 = float(np.sum(np.log2(1 + snr_flat)) * sym_rate / 1e6)
        pc.scatter("d_now", [p.psd], [C1], color=RED, size=12, symbol="d")
        pc.set_xlim(-60, -30)
        pc.set_ylim(0, max(cw) * 1.1 + 0.1)
        self.n_used = int(np.sum(pw > 0))
        self._st = dict(mu=mu, n=self.n_used, bits=int(b.sum()), rate=b.sum() * sym_rate / 1e6)
        self.readout(cwf=C1, ceq=C0, gain=100 * (C1 / C0 - 1) if C0 > 1e-9 else 0.0,
                     used=f"{self.n_used} of 256")

    def story(self, p):
        stt = getattr(self, "_st", {"mu": 0, "n": 0, "bits": 0})
        gain = self.r.get("gain", 0)
        if p.mode == "6 sub-channels":
            s = (f"<p>Each sub-channel's floor is its noise divided by its gain: good channels "
                 f"sit low, bad ones high. Pour the total power in like water: it settles at "
                 f"one level μ = {v(stt['mu'], '.3g')}, so good channels get more power, and "
                 f"channels whose floor sticks out above the water get <b>none</b>. Right now "
                 f"{v(stt['n'], 'd')} of 6 are in use. Drag a floor's dot up or down to make "
                 f"that sub-channel worse or better.</p>"
                 f"<p>Water-filling beats spreading the power equally by "
                 f"{v(gain, '.1f', '%')}. The gain is large when power is scarce (do not waste "
                 f"it on hopeless channels) and vanishes at high SNR, where every channel is "
                 f"worth using and log(1 + SNR) is flat.</p>")
        else:
            s = (f"<p>A real DSL line: 256 tones, 4.3125 kHz apart. The copper loses more at high "
                 f"frequency, so the floor rises to the right, and an AM radio station leaks in "
                 f"near 680 kHz. With a {v(p.gap, '.1f', 'dB')} gap the modem loads "
                 f"{v(stt['bits'], 'd')} bits per DMT symbol: "
                 f"{v(stt.get('rate', 0), '.1f', 'Mb/s')} at 4000 symbols per second.</p>"
                 "<p>Real ADSL transmits a flat PSD (regulators cap it) and adapts the bits per "
                 "tone instead; that loses very little against true water-filling, as the "
                 "dashed curve shows. The ingress tones simply carry fewer bits.</p>")
        return "<h3>Water finds its level</h3>" + s + keybox(
            "P_k = max(μ − N_k/|H_k|², 0). Every OFDM, DSL and MIMO system allocates power this way.")


# =============================================================================== 9. rate-distortion
VARS = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.125])


class RateDistortion(Experiment):
    title = "Rate–distortion"
    blurb = "How few bits for a given fidelity? Every quantiser chases the same curve."
    book = "sec:ch13:rd"
    controls = [
        Heading("Quantising a Gaussian source"),
        Choice("q", "Quantiser", ["Uniform", "Lloyd–Max", "Uniform + entropy coding"],
               "Uniform", style="menu"),
        IntSlider("bits", "Bits per sample", 1, 6, 2,
                  enabled_if=lambda p: p.q != "Uniform + entropy coding"),
        LogSlider("step", "Step size Δ", 0.05, 3.0, 0.5, unit="σ",
                  enabled_if=lambda p: p.q == "Uniform + entropy coding"),
        Heading("Six components (reverse water-filling)"),
        Slider("rtot", "Total rate", 0.0, 16.0, 6.0, step=0.1, unit="bits/vector"),
    ]
    plots = [
        Plot("rd", "SNR vs rate: the bound and three quantiser families", x="rate (bits per sample)",
             y="SNR = σ²/D (dB)", xlim=(0, 6.5), ylim=(0, 40), legend="tl"),
        Plot("stair", "Your quantiser on the Gaussian pdf", x="input x (σ)", y="output q(x) (σ)",
             xlim=(-4, 4), ylim=(-4, 4), legend="tl"),
        BarPlot("rwf", "Reverse water-filling (navy = described)",
                x="component", y="variance (σ²)", legend=None),
    ]
    layout = [["rd", "stair"], ["rd", "rwf"]]
    col_stretch = [5, 4]
    readouts = [
        Readout("R", "Rate", "bits/sample", ".2f"),
        Readout("snr", "SNR", "dB", ".2f"),
        Readout("gap", "Gap to R(D)", "dB", ".2f", good=lambda x: x < 1.6),
        Readout("theta", "Water level θ", "σ²", ".3f"),
    ]
    challenges = [
        Challenge("Beat the uniform quantiser by at least 1 dB at the same number of bits.",
                  lambda s: s.p.q == "Lloyd–Max" and s.exp.gain_u >= 1.0,
                  hint="Non-uniform levels matter most with many levels."),
        Challenge("Entropy coding: get within 1.6 dB of the bound at more than 3 bits per sample.",
                  lambda s: s.p.q == "Uniform + entropy coding" and s.r.R > 3 and s.r.gap < 1.6),
        Challenge("Choose a total rate at which the two weakest components get no bits at all, "
                  "but the other four all do.",
                  lambda s: s.exp.n_zero == 2),
    ]

    def setup(self):
        self.gain_u = 0.0
        self.n_zero = 0
        self.uq = [sk.uniform_quantizer(2 ** b) for b in range(1, 7)]
        self.lm = [sk.lloyd_max(2 ** b) for b in range(1, 7)]
        steps = np.logspace(np.log10(0.04), np.log10(3.2), 50)
        self.ec = np.array([sk.ecsq(s_) for s_ in steps])

    def update(self, p):
        x, f = sk.source_pdf("gaussian", np.linspace(-4, 4, 801))
        if p.q == "Uniform":
            lev, mse, H, _ = self.uq[p.bits - 1]
            R = p.bits
        elif p.q == "Lloyd–Max":
            lev, mse, H = self.lm[p.bits - 1]
            R = p.bits
        else:
            R, mse = sk.ecsq(p.step)
            lev = np.arange(-int(5 / p.step) - 1, int(5 / p.step) + 2) * p.step
        snr = float(db(1 / mse))
        gap = float(sk.gaussian_rd_snr_db(R) - snr)
        self.gain_u = snr - float(db(1 / self.uq[p.bits - 1][1])) if p.q == "Lloyd–Max" else 0.0
        pr = self.plot("rd")
        r = np.linspace(0, 6.5, 50)
        pr.line("bound", r, sk.gaussian_rd_snr_db(r), color=RED, width=2.4,
                name="R(D) bound: 6.02 dB per bit")
        pr.line("ecb", r, sk.gaussian_rd_snr_db(r) - 1.53, color=GREEN, style=":", width=1.0)
        bits = np.arange(1, 7)
        pr.scatter("uq", bits, [db(1 / u[1]) for u in self.uq], color=GRAY, size=10, symbol="s",
                   name="uniform (fixed rate)")
        pr.scatter("lm", bits, [db(1 / l_[1]) for l_ in self.lm], color=NAVY, size=10,
                   name="Lloyd–Max (fixed rate)")
        ok = self.ec[:, 0] < 6.5
        pr.line("ec", self.ec[ok, 0], db(1 / self.ec[ok, 1]), color=GREEN, width=2.0,
                name="uniform + entropy coding")
        pr.scatter("you", [R], [snr], color=RED, size=16, symbol="d", name="you")
        pr.text("g153", 5.0, sk.gaussian_rd_snr_db(5.0) - 1.53 - 1.2, "1.53 dB", color=GREEN,
                size=8.5, anchor=(0, 1))
        # staircase over pdf
        ps = self.plot("stair")
        q, _ = sk.quantize(x, lev)
        ps.line("pdf", x, -4 + 8 * f / f.max() * 0.35, color=GRAY, width=1.0, fill=-4,
                fill_alpha=0.25, name="pdf (scaled)")
        ps.line("id", [-4, 4], [-4, 4], color=GRAY, style=":", width=1.0)
        ps.line("q", x, q, color=NAVY, width=2.2, name="q(x)")
        levs = lev[np.abs(lev) < 4]
        ps.scatter("lev", np.full(len(levs), -3.85), levs, color=RED, size=6, symbol="t")
        # reverse water-filling
        D, Ri, th = self.rwf(p.rtot)
        self.n_zero = int(np.sum(Ri < 1e-9))
        pw = self.plot("rwf")
        k = np.arange(1, 7)
        pw.bars("var", k, VARS, width=0.7, color=GRAY, alpha=0.5)
        pw.bars("desc", k, VARS, base=np.minimum(VARS, th), width=0.7, color=NAVY)
        pw.hline("th", th, color=RED, style="--", width=1.6, label=f"θ = {th:.3f}", label_pos=0.6)
        for i in range(6):
            pw.text(f"r{i}", k[i], VARS[i], f"{Ri[i]:.2f} b", color=NAVY, size=8.5,
                    anchor=(0.5, 1.05), bold=True)
        pw.set_xticks([(i, str(i)) for i in k])
        pw.set_xlim(0.4, 6.6)
        pw.set_ylim(0, 5.0)
        self.readout(R=R, snr=snr, gap=gap, theta=th)

    @staticmethod
    def rwf(R):
        if R <= 0:
            return VARS.copy(), np.zeros(6), VARS.max()
        lo, hi = 1e-9, VARS.max()
        for _ in range(100):
            th = np.sqrt(lo * hi)
            r = np.sum(np.maximum(0, 0.5 * np.log2(VARS / th)))
            if r > R:
                lo = th
            else:
                hi = th
        th = np.sqrt(lo * hi)
        D = np.minimum(th, VARS)
        return D, 0.5 * np.log2(VARS / D), th

    def story(self, p):
        snr, gap, R = self.r.get("snr", 0), self.r.get("gap", 0), self.r.get("R", 0)
        s = (f"<p>Shannon's bound for a Gaussian source is brutally simple: each bit of rate buys "
             f"at most 6.02 dB of SNR. Your {p.q.lower()} quantiser spends "
             f"{v(R, '.2f', 'bits')} per sample for {v(snr, '.2f', 'dB')}: "
             f"{v(gap, '.2f', 'dB')} short of the bound.</p>")
        if p.q == "Uniform":
            s += ("<p>A uniform quantiser wastes levels in the tails, where samples are rare. "
                  "The staircase (top right) has equal steps whatever the pdf says.</p>")
        elif p.q == "Lloyd–Max":
            s += (f"<p>Lloyd–Max moves the levels toward the bulk of the pdf: each level sits at "
                  f"the centroid of its cell, each threshold halfway between levels. That gains "
                  f"{v(self.gain_u, '.2f', 'dB')} over uniform at {v(p.bits, 'd')} bits.</p>")
        else:
            s += ("<p>Here the levels are uniform but their indices are entropy-coded: frequent "
                  "levels get short codewords. At high rate this is only 1.53 dB (a quarter of a "
                  "bit) from the bound, the same πe/6 that separates QAM from Gaussian "
                  "signalling. Closing that last gap needs vector quantisation.</p>")
        s += (f"<p>Bottom right: six independent components of decreasing variance share "
              f"{v(p.rtot, '.1f', 'bits')}. The optimal allocation is <b>reverse "
              f"water-filling</b>: every component is described down to the same distortion θ, "
              f"and components weaker than θ get no bits at all. That is why codecs throw away "
              f"high-frequency coefficients.</p>")
        return "<h3>Fidelity has a price</h3>" + s + keybox(
            "D(R) = σ²·2^(−2R): 6.02 dB per bit. Spend bits only where the variance exceeds the water level.")


# =============================================================================== the lab
LAB = st.Lab(19, "Information Theory", chapter=13, chapter_title="Information Theory",
             experiments=[Surprise, TextImage, Huffman, DMCCapacity, ShannonLimit, Constrained,
                          FiniteBlocklength, WaterFilling, RateDistortion])

if __name__ == "__main__":
    st.run(LAB)
