"""Lab 24 · Source Coding: Text, Speech, Images, Audio and Video   (Chapter 16)

Run it:      python labs/lab24_source_coding.py
Self-test:   python labs/lab24_source_coding.py --selftest

Before a single bit reaches the channel, a source coder decides how few bits are needed.
Nine experiments walk through the tools inside every codec: Huffman versus arithmetic
coding on text you type, the LZ77 sliding window behind ZIP and PNG, quantisers and the
Lloyd–Max algorithm, the DCT's energy compaction, a working JPEG coder with a quality
slider, an LPC vocoder that turns a buzz into a vowel, the psychoacoustic masking that MP3
and AAC exploit (you can hear it), and the motion search at the heart of video coding.
Everything is synthesised or bundled; nothing is downloaded.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import bz2
import io
import lzma
import zlib
from collections import Counter
from functools import lru_cache

import numpy as np
from PySide6 import QtWidgets
from scipy import signal

import commlib as cl
from commlib import infotheory as it
from commlib import sourcecoding as sc
from commlib import sourcekit as sk
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad
from studio.controls import Control


# =============================================================================== a text-entry control
class TextBox(Control):
    """A one-line text field (studio has no text control yet), with a menu of examples
    that fills it. The value is the text; every keystroke updates the experiment."""

    def __init__(self, key, label, default="", examples=(), help="", max_len=600,
                 enabled_if=None):
        super().__init__(key, label, help, enabled_if)
        self.default = default
        self.examples = list(examples)
        self.max_len = max_len
        self._v = default

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(3)
        name = QtWidgets.QLabel(self.label)
        name.setObjectName("ctlname")
        lay.addWidget(name)
        self._edit = QtWidgets.QLineEdit(self.default)
        self._edit.setObjectName("search")
        self._edit.setMaxLength(self.max_len)
        self._edit.setPlaceholderText("type anything…")
        self._edit.setClearButtonEnabled(True)
        self._edit.setMinimumWidth(40)
        self._edit.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self._edit)
        self._on_change = on_change
        self._edit.textChanged.connect(self._typed)
        if self.examples:
            self._combo = QtWidgets.QComboBox()
            self._combo.addItems(["Examples…"] + [e[:40] + ("…" if len(e) > 40 else "")
                                                  for e in self.examples])
            self._combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self._combo.setMinimumContentsLength(8)
            self._combo.activated.connect(self._pick)
            lay.addWidget(self._combo)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _typed(self, t):
        if t == self._v:
            return
        self._v = t
        self._on_change(self.key, t)

    def _pick(self, i):
        if i > 0:
            self._edit.setText(self.examples[i - 1])
            self._combo.setCurrentIndex(0)

    def value(self):
        return self._v

    def set_value(self, v, notify=False):
        self._v = str(v)
        if self.widget is not None:
            self._edit.blockSignals(True)
            self._edit.setText(self._v)
            self._edit.blockSignals(False)
        if notify and self.widget is not None:
            self._on_change(self.key, self._v)

    def set_enabled(self, on):
        if self.widget is not None:
            self.widget.setEnabled(on)

    def random_value(self, rng):
        pool = self.examples + [self.default, "", "a", "zzzzzzzzzzzzzzzz"]
        return pool[int(rng.integers(0, len(pool)))]


# =============================================================================== shared helpers
GRAYS = ("#000000", "#FFFFFF")


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def show_char(c):
    return "␣" if c == " " else c


@lru_cache(maxsize=1)
def book_text():
    return sc.load_text()[:3000]


@lru_cache(maxsize=1)
def photo():
    try:
        return sc.load_image()
    except Exception:
        return synthetic_image()


@lru_cache(maxsize=1)
def synthetic_image(n=256):
    """A deterministic test card: gradients, discs, stripes, a little noise."""
    y, x = np.mgrid[:n, :n] / n
    img = 90 + 80 * x + 40 * np.sin(2 * np.pi * y * 1.5)
    for cx, cy, r, lev in [(0.3, 0.35, 0.16, 225), (0.72, 0.62, 0.2, 40), (0.6, 0.22, 0.09, 180)]:
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = lev
    band = (y > 0.78) & (y < 0.92)
    img[band] = 128 + 100 * np.sign(np.sin(2 * np.pi * x[band] * 12))
    img += np.random.default_rng(5).normal(0, 3, img.shape)
    return np.clip(np.round(img), 0, 255)


def image_of(name):
    return photo() if name == "Photograph" else synthetic_image()


def show_image(plot, key, img, levels=(0, 255), cmap=GRAYS):
    """Draw a 2-D array as an image with row 0 at the top and square pixels."""
    h, w = img.shape
    plot.image(key, np.asarray(img, float)[::-1], x=(0, w), y=(0, h), cmap=cmap, levels=levels)
    plot.set_xlim(0, w)
    plot.set_ylim(0, h)
    plot.set_xticks([])
    plot.set_yticks([])
    if getattr(plot, "_lab_aspect", False):        # show the whole image despite the aspect lock
        plot.vb.setRange(xRange=(0, w), yRange=(0, h), padding=0)


def lock_aspect(plot):
    if not getattr(plot, "_lab_aspect", False):
        plot.vb.setAspectLocked(True)
        plot._lab_aspect = True


# =============================================================================== 1. Huffman vs arithmetic
def arith_costs(text, order, inc=16):
    """Per-character ideal cost (bits) under ArithmeticCoder's adaptive model: counts start at
    1 for every symbol of the alphabet and grow by ``inc`` (same model as commlib's coder)."""
    alpha = sorted(set(text))
    idx = {c: i for i, c in enumerate(alpha)}
    tables = {}
    out = np.empty(len(text))
    for i, c in enumerate(text):
        ctx = text[max(0, i - order):i]
        f = tables.get(ctx)
        if f is None:
            f = tables[ctx] = np.ones(len(alpha))
        out[i] = -np.log2(f[idx[c]] / f.sum())
        f[idx[c]] += inc
    return out


@lru_cache(maxsize=64)
def text_codes(text, order):
    cnt = Counter(text)
    lengths = it.huffman_lengths(cnt)
    canon = sc.canonical_huffman(lengths)
    huff_bits = sc.huffman_encode(text, canon)
    huff_ok = "".join(sc.huffman_decode(huff_bits, canon)) == text
    ac = sc.ArithmeticCoder(order=order)
    bits = ac.encode(text)
    dec = sc.ArithmeticCoder(order=order, alphabet=ac.alphabet).decode(bits, len(text))
    data = text.encode("utf8", "replace")
    comp = {"zlib": 8 * len(zlib.compress(data, 9)), "bzip2": 8 * len(bz2.compress(data, 9)),
            "xz": 8 * len(lzma.compress(data, preset=9))}
    return dict(cnt=cnt, lengths=lengths, canon=canon, huff=len(huff_bits), huff_ok=huff_ok,
                arith=len(bits), arith_ok=dec == text, costs=arith_costs(text, order), comp=comp)


TEXTS = [
    "she sells sea shells by the sea shore and the shells she sells are sea shells for sure",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
    "to be or not to be that is the question whether tis nobler in the mind to suffer",
    "the quick brown fox jumps over the lazy dog " * 5,
    "abracadabra abracadabra abracadabra abracadabra",
]
DEFAULT_TEXT = ("source coding removes redundancy: frequent symbols get short codewords, rare "
                "ones get long codewords, and a good model predicts what comes next. huffman "
                "codes each symbol on its own; arithmetic coding codes the whole message at once.")


class HuffmanArithmetic(Experiment):
    title = "Huffman vs arithmetic coding"
    blurb = "Two ways to spend fractional bits. Type a message and watch both encoders work."
    book = "sec:ch16:lossless"
    controls = [
        Choice("src", "Text", ["Your text", "Book (Chapter 1, 3000 chars)"]),
        TextBox("text", "Your text", DEFAULT_TEXT, examples=TEXTS,
                enabled_if=lambda p: p.src == "Your text"),
        IntSlider("order", "Arithmetic coder context order", 0, 3, 1,
                  help="How many previous characters the adaptive model looks at"),
    ]
    plots = [
        BarPlot("code", "Huffman codeword length (bars) vs the ideal −log₂ p (◆)",
                x="symbol (most frequent first)", y="bits"),
        BarPlot("cmp", "Real encoded size (green dashed: order-0 entropy)", y="bits per character"),
        Plot("cum", "Bits spent so far, character by character", x="characters coded",
             y="cumulative bits", legend="tl"),
    ]
    layout = [["code", "cmp"], ["cum", "cum"]]
    readouts = [
        Readout("H", "Entropy (order 0)", "bits/char", ".3f"),
        Readout("huff", "Huffman", "bits/char", ".3f"),
        Readout("arith", "Arithmetic", "bits/char", ".3f"),
        Readout("ok", "Both decode exactly", "", None),
    ]
    challenges = [
        Challenge("Beat Huffman by more than 0.5 bits per character with the arithmetic coder.",
                  lambda s: s.r.huff - s.r.arith > 0.5,
                  hint="Context is the key: try the book's text and a higher order."),
        Challenge("Huffman's blind spot: 20+ characters with entropy below 0.5 bits, where Huffman "
                  "still pays a whole bit per character.",
                  lambda s: s.p.src == "Your text" and len(s.p.text) >= 20 and s.r.H < 0.5
                  and s.r.huff >= 0.999,
                  hint="One symbol almost all the time."),
        Challenge("See the learning cost: on a text shorter than 100 characters, find an order "
                  "where arithmetic coding does worse than Huffman.",
                  lambda s: s.p.src == "Your text" and 10 <= len(s.p.text) < 100
                  and s.r.arith > s.r.huff),
    ]

    def update(self, p):
        text = p.text if p.src == "Your text" else book_text()
        if len(text) < 2:
            text = "ab"
        d = text_codes(text, p.order)
        n = len(text)
        cnt = d["cnt"]
        syms = sorted(cnt, key=lambda c: (-cnt[c], c))[:30]
        pr = np.array([cnt[c] for c in syms]) / n
        pc = self.plot("code")
        x = np.arange(len(syms))
        pc.bars("L", x, [d["lengths"][c] for c in syms], width=0.7, color=NAVY)
        pc.scatter("id", x, -np.log2(pr), color=RED, size=9, symbol="d")
        pc.set_xticks([(i, show_char(c)) for i, c in enumerate(syms)])
        pc.set_xlim(-0.7, max(len(syms), 6) - 0.3)
        pc.set_ylim(0, max(max(d["lengths"].values()), 2) + 1.5)
        top = syms[0]
        pc.text("cw", -0.5, None, f"canonical codeword of '{show_char(top)}': {d['canon'][top]}",
                color=NAVY, size=8.5, anchor=(0, 0), bold=True)
        # comparison bars
        H = float(it.entropy(np.array(list(cnt.values()))))
        K = len(cnt)
        vals = [np.ceil(np.log2(max(K, 2))), d["huff"] / n, d["arith"] / n,
                d["comp"]["zlib"] / n, d["comp"]["bzip2"] / n, d["comp"]["xz"] / n]
        names = ["fixed", "Huffman", f"arith. k={p.order}", "zlib", "bzip2", "xz"]
        cols = [GRAY, NAVY, RED, ORANGE, ORANGE, ORANGE]
        pm = self.plot("cmp")
        pm.bars("b", np.arange(6), vals, width=0.65, colors=cols)
        ymax = max(8.5, min(max(vals), 14) * 1.12)
        for i, val in enumerate(vals):
            pm.text(f"t{i}", i, min(val, ymax * 0.93), f"{val:.2f}", anchor=(0.5, 1.05), size=8.5,
                    bold=True)
        pm.hline("H", H, color=GREEN, style="--", width=1.5)
        pm.set_xticks([(i, nm) for i, nm in enumerate(names)])
        pm.set_xlim(-0.6, 5.6)
        pm.set_ylim(0, ymax)
        # cumulative bits
        hc = np.cumsum([d["lengths"][c] for c in text])
        ac = np.cumsum(d["costs"])
        k = np.arange(1, n + 1)
        pu = self.plot("cum")
        pu.line("fix", [0, n], [0, n * vals[0]], color=GRAY, style=":", width=1.4,
                name="fixed-length code")
        pu.line("h", k, hc, color=NAVY, width=2.2, name="Huffman (static, order 0)")
        pu.line("a", k, ac, color=RED, width=2.2, name=f"arithmetic, adaptive order {p.order}")
        pu.line("H", [0, n], [0, n * H], color=GREEN, style="--", width=1.2,
                name="entropy × characters")
        pu.set_xlim(0, n)
        pu.set_ylim(0, max(n * vals[0], hc[-1], ac[-1]) * 1.05)
        ok = d["huff_ok"] and d["arith_ok"]
        self._d = dict(n=n, K=K, H=H, top=top, canon=d["canon"][top])
        self.readout(H=H, huff=d["huff"] / n, arith=d["arith"] / n,
                     ok="✓ yes" if ok else "✗ no")

    def story(self, p):
        d = getattr(self, "_d", None)
        if d is None:
            return ""
        hu, ar = self.r.get("huff", 0), self.r.get("arith", 0)
        s = (f"<p>Your {v(d['n'], 'd')} characters use {v(d['K'], 'd')} different symbols. "
             f"<b>Huffman</b> gives each symbol a whole number of bits: the most common, "
             f"'{show_char(d['top'])}', gets the canonical codeword {v(d['canon'], 's')}. Only "
             f"the code lengths need sending; that is how DEFLATE and JPEG transmit their tables. "
             f"Result: {v(hu, '.3f', 'bits/char')} against an order-0 entropy of "
             f"{v(d['H'], '.3f')}.</p>"
             f"<p><b>Arithmetic coding</b> narrows an interval by each character's probability, "
             f"so a likely character costs a fraction of a bit. Its adaptive model learns as it "
             f"goes and here looks back {v(p.order, 'd')} character(s): "
             f"{v(ar, '.3f', 'bits/char')}. ")
        s += (good("It wins") + " because it uses context, which Huffman on single symbols cannot."
              if ar < hu else bad("It loses") + ": a short text does not give the adaptive model "
              "time to learn (the red curve starts steep).") + "</p>"
        s += ("<p>The orange bars are real compressors. On a few hundred characters their headers "
              "dominate; on the book they land between Huffman and a good context model.</p>")
        return "<h3>Whole bits versus fractions</h3>" + s + keybox(
            "Huffman: within 1 bit of the entropy per symbol. Arithmetic: within 2 bits per "
            "message, and any model you like.")


# =============================================================================== 2. LZ77
LZ_TEXTS = [
    "she sells sea shells by the sea shore, the shells she sells are sea shells for sure",
    "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
    "to be or not to be, that is the question; to be or not to be, that was the question",
    "the rain in spain stays mainly in the plain, the rain in spain stays mainly in the plain",
    "an pharmacy analysis panama banana bandana cabana",
]


class LZ77(Experiment):
    title = "LZ77: the sliding window"
    blurb = "Say it once; next time, point back to it. Press Play to watch the parser work."
    book = "sec:ch16:lz"
    animate = True
    fps = 2
    controls = [
        TextBox("text", "Text to compress", LZ_TEXTS[0], examples=LZ_TEXTS),
        IntSlider("lw", "Window size (log₂)", 3, 12, 5, unit="",
                  help="The parser looks this many characters back: 2^x (DEFLATE uses 32 768)"),
        IntSlider("maxlen", "Longest match", 3, 64, 18, unit="chars",
                  help="DEFLATE allows 258"),
        Button("step", "Next token  ▸"),
    ]
    plots = [
        Plot("parse", "The parse: red = literal, navy = copied (arc = where it was copied from)",
             x="position in the text", y="", legend=None, grid=False),
        Plot("toks", "Tokens around the cursor", legend=None, grid=False),
        Plot("size", "Compressed size vs window size", x="window (characters)",
             y="bits per character", logx=True, xlim=(6, 6000), ylim=(0, 10), legend="tr"),
    ]
    layout = [["parse", "parse"], ["toks", "size"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("ntok", "Tokens", "", "int"),
        Readout("lit", "Literals", "%", ".0f"),
        Readout("bpc", "Size", "bits/char", ".2f", good=lambda x: x < 4),
        Readout("longest", "Longest match", "chars", "int"),
    ]
    challenges = [
        Challenge("Get your text below 3 bits per character.",
                  lambda s: s.r.bpc < 3,
                  hint="Repetition is what LZ77 eats: the abc example, or repeat a phrase."),
        Challenge("Copy a single match longer than 30 characters.",
                  lambda s: s.r.longest > 30,
                  hint="Raise the longest-match limit; a match may overlap the text it creates."),
        Challenge("Window too small: with the default sea-shells text, find a window where a "
                  "bigger one would save more than 20 %.",
                  lambda s: s.p.text == LZ_TEXTS[0] and s.exp.loss > 0.2),
    ]

    def setup(self):
        self.cur = 0
        self.loss = 0.0

    def on_step(self, p):
        self.cur += 1

    def parse(self, p):
        text = p.text if p.text else "a"
        W = 2 ** p.lw
        toks = sk.lz77_parse(text, W, p.maxlen)
        return text, W, toks

    def update(self, p):
        text, W, toks = self.parse(p)
        if not self.playing:
            mlen = [t[1] for t in toks]
            self.cur = int(np.argmax(mlen)) if max(mlen) > 1 else 0
        self.draw(p, text, W, toks)

    def tick(self, p):
        text, W, toks = self.parse(p)
        self.cur = (self.cur + 1) % len(toks)
        self.draw(p, text, W, toks)

    def draw(self, p, text, W, toks):
        self.cur = min(self.cur, len(toks) - 1)
        pos, L, dist = toks[self.cur]
        n = len(text)
        span = 64
        v0 = int(np.clip(pos + L // 2 - span // 2, 0, max(0, n - span)))
        v1 = min(n, v0 + span)
        lit = np.zeros(n, bool)
        for q, l_, d in toks:
            if d == 0:
                lit[q] = True
        xs = np.arange(v0, v1)
        pp = self.plot("parse")
        pp.pi.hideAxis("left")
        pp.band("win", max(-0.5, pos - W - 0.5), pos - 0.5, color=GREEN, alpha=0.10)
        pp.bars("cells", xs, np.full(len(xs), 1.0), width=0.9,
                colors=[RED if lit[i] else NAVY for i in xs], alpha=0.25)
        if dist:
            pp.bars("src", np.arange(pos - dist, pos - dist + L), np.full(L, 1.0), width=0.9,
                    color=ORANGE, alpha=0.55)
        pp.bars("now", np.arange(pos, pos + L), np.full(L, 1.0), width=0.9,
                color=RED if dist == 0 else BLUE, alpha=0.6)
        for j in range(span):
            i = v0 + j
            if i < v1:
                pp.text(f"c{j}", i, 0.5, show_char(text[i]), color=None, size=10, bold=True,
                        anchor=(0.5, 0.5))
        # arcs for every copy in view, the current one highlighted
        ax, ay = [], []
        for q, l_, d in toks:
            if d and q + l_ > v0 and q < v1:
                a, b = q - d + l_ / 2 - 0.5, q + l_ / 2 - 0.5
                t = np.linspace(0, 1, 24)
                hgt = 0.5 + min(d, 60) / 60 * 2.2
                ax += list(a + (b - a) * t) + [np.nan]
                ay += list(1.05 + hgt * 4 * t * (1 - t)) + [np.nan]
        if ax:
            pp.line("arcs", ax, ay, color=GRAY, width=1.0)
        if dist:
            a, b = pos - dist + L / 2 - 0.5, pos + L / 2 - 0.5
            t = np.linspace(0, 1, 40)
            hgt = 0.5 + min(dist, 60) / 60 * 2.2
            pp.line("arc", a + (b - a) * t, 1.05 + hgt * 4 * t * (1 - t), color=RED, width=2.6)
        pp.text("win_l", max(v0, pos - W) - 0.4, -0.15, f"search window ({W} chars)", color=GREEN,
                size=8.5, anchor=(0, 0))
        pp.set_xlim(v0 - 0.8, v0 + span - 0.2)
        pp.set_ylim(-0.6, 3.9)
        # token list
        pt = self.plot("toks")
        pt.pi.hideAxis("left")
        pt.pi.hideAxis("bottom")
        lo = max(0, self.cur - 4)
        for r, k in enumerate(range(lo, lo + 9)):
            if k >= len(toks):
                break
            q, l_, d = toks[k]
            if d == 0:
                s = f"#{k:>3}  literal  '{show_char(text[q])}'"
            else:
                s = f"#{k:>3}  copy {l_} chars from {d} back: '{text[q:q + l_][:18]}'"
            pt.text(f"r{r}", 0, -r, s, color=RED if k == self.cur else None, size=9.5,
                    bold=k == self.cur, anchor=(0, 0.5))
        for r in range(9):
            if lo + r >= len(toks):
                pt.text(f"r{r}", 0, -r, "", anchor=(0, 0.5))
        pt.set_xlim(-0.05, 1)
        pt.set_ylim(-8.6, 0.6)
        # size vs window
        ws = 2 ** np.arange(3, 13)
        bpc = [sk.lz77_bits(sk.lz77_parse(text, w, p.maxlen), w, p.maxlen) / n for w in ws]
        ps = self.plot("size")
        ps.line("b", ws, bpc, color=NAVY, width=2.2, name="LZ77, fixed-length fields")
        ps.scatter("bp", ws, bpc, color=NAVY, size=7)
        ps.hline("raw", 8, color=GRAY, style=":", width=1.2, label="8 bits/char raw", label_pos=0.05)
        bits = sk.lz77_bits(toks, W, p.maxlen)
        ps.scatter("now", [W], [bits / n], color=RED, size=14, symbol="d", name="your window")
        ps.set_ylim(0, max(10, max(bpc) * 1.1))
        self.loss = 1 - min(bpc) / (bits / n) if bits else 0.0
        nlit = sum(1 for t in toks if t[2] == 0)
        self._st = dict(tok=(pos, L, dist), W=W, best=min(bpc), bestw=int(ws[int(np.argmin(bpc))]))
        self.readout(ntok=len(toks), lit=100 * nlit / len(toks), bpc=bits / n,
                     longest=max(t[1] for t in toks))

    def story(self, p):
        stt = getattr(self, "_st", None)
        if stt is None:
            return ""
        pos, L, dist = stt["tok"]
        s = ("<p>LZ77 (Lempel and Ziv, 1977) keeps the last characters it has sent as a "
             "<b>dictionary</b>. At each position it looks back through the green window for the "
             "longest string that matches what comes next, and sends a pointer (distance, length) "
             "instead of the characters. No match? It sends a literal.</p>")
        if dist:
            s += (f"<p>The highlighted token at position {v(pos, 'd')} copies "
                  f"{v(L, 'd')} characters from {v(dist, 'd')} back (orange to blue, red arc). "
                  f"Press <b>Play</b> or <b>Next token</b> to step through the parse.</p>")
        else:
            s += (f"<p>The highlighted token at position {v(pos, 'd')} is a literal. Press "
                  f"<b>Play</b> to step through the parse.</p>")
        s += (f"<p>A bigger window finds more matches but every pointer costs more bits: the "
              f"best window for this text is {v(stt['bestw'], 'd')} characters "
              f"({v(stt['best'], '.2f', 'bits/char')}). ZIP, gzip and PNG use LZ77 with a 32 kB "
              f"window and then Huffman-code the tokens (DEFLATE).</p>")
        return "<h3>Point back instead of repeating</h3>" + s + keybox(
            "LZ77 needs no statistics: it learns the source as it goes, and is asymptotically optimal.")


# =============================================================================== 3. quantisers
class Quantisers(Experiment):
    title = "Quantisers and Lloyd–Max"
    blurb = "Where should the levels go? Run Lloyd's algorithm one iteration at a time."
    book = "sec:ch16:lossy"
    controls = [
        Choice("pdf", "Source distribution", ["Gaussian", "Laplacian", "Uniform"],
               help="Laplacian: like speech samples and DCT coefficients, peaky with long tails"),
        IntSlider("bits", "Bits per sample", 1, 5, 2),
        IntSlider("iters", "Lloyd iterations", 0, 40, 0,
                  help="0 = the best uniform quantiser; each iteration moves levels to cell centroids"),
    ]
    plots = [
        Plot("pdf", "Levels (▲) and cell boundaries on the pdf",
             x="input x (σ)", y="pdf", xlim=(-4, 4), legend=None),
        Plot("stair", "Input → output map", x="input x (σ)", y="output q(x) (σ)",
             xlim=(-4, 4), ylim=(-4, 4), legend=None),
        Plot("conv", "Gain over the best uniform quantiser as Lloyd's algorithm iterates",
             x="iteration", y="SNR gain (dB)", xlim=(-0.5, 40.5), legend="br"),
    ]
    layout = [["pdf", "stair"], ["conv", "conv"]]
    readouts = [
        Readout("snr", "SNR", "dB", ".2f"),
        Readout("gain", "Gain over uniform", "dB", ".2f"),
        Readout("H", "Output entropy", "bits", ".2f",
                help="What an entropy coder would spend on the indices"),
        Readout("bound", "Rule of thumb 6.02·R", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Laplacian source, 4 bits: let Lloyd beat the uniform quantiser by more than 2 dB.",
                  lambda s: s.p.pdf == "Laplacian" and s.p.bits == 4 and s.r.gain > 2.0,
                  hint="Give the algorithm enough iterations."),
        Challenge("Show where Lloyd–Max gains nothing: at least 2 bits and 20 iterations, gain "
                  "below 0.01 dB.",
                  lambda s: s.p.bits >= 2 and s.p.iters >= 20 and abs(s.r.gain) < 0.01,
                  hint="Which pdf is already perfectly served by equal steps?"),
        Challenge("Leave room for an entropy coder: output entropy at least 0.4 bit below the "
                  "fixed rate.",
                  lambda s: s.p.bits - s.r.H >= 0.4),
    ]

    def update(self, p):
        kind = p.pdf.lower()
        L = 2 ** p.bits
        hist = sk.lloyd_iterations(L, 40, kind)
        lev = hist[p.iters]
        x, f = sk.source_pdf(kind)
        dx = x[1] - x[0]

        def stats(c):
            q, idx = sk.quantize(x, c)
            mse = np.sum((x - q) ** 2 * f) * dx
            P = np.bincount(idx, f, len(c)) * dx
            P = P[P > 1e-12] / P.sum()
            return float(db(1 / mse)), float(-np.sum(P * np.log2(P)))

        snrs = [stats(c)[0] for c in hist]
        snr, H = stats(lev)
        xs, fs_ = sk.source_pdf(kind, np.linspace(-4, 4, 801))
        pp = self.plot("pdf")
        pp.line("f", xs, fs_, color=NAVY, width=2.0, fill=0, fill_alpha=0.15)
        th = (lev[1:] + lev[:-1]) / 2
        th = th[np.abs(th) < 4]
        tx = np.repeat(th, 3)
        ty = np.tile([0, fs_.max() * 1.15, np.nan], len(th))
        pp.line("th", tx, ty, color=GRAY, style=":", width=1.2)
        lv = lev[np.abs(lev) < 4]
        pp.scatter("lev", lv, np.zeros(len(lv)), color=RED, size=11, symbol="t")
        pp.set_ylim(-0.03, fs_.max() * 1.2)
        ps = self.plot("stair")
        q, _ = sk.quantize(xs, lev)
        ps.line("id", [-4, 4], [-4, 4], color=GRAY, style=":", width=1.0)
        ps.line("q", xs, q, color=NAVY, width=2.2)
        pc = self.plot("conv")
        k = np.arange(41)
        gains = np.array(snrs) - snrs[0]
        pc.line("c", k, gains, color=NAVY, width=2.2, name="Lloyd iterations")
        pc.scatter("cp", k, gains, color=NAVY, size=5)
        pc.hline("u", 0, color=GRAY, style="--", width=1.2, label="best uniform quantiser",
                 label_pos=0.6)
        pc.scatter("now", [p.iters], [snr - snrs[0]], color=RED, size=14, symbol="d", name="you")
        pc.set_ylim(-0.1, max(0.5, gains.max() * 1.25))
        self.readout(snr=snr, gain=snr - snrs[0], H=H, bound=6.02 * p.bits)

    def story(self, p):
        g = self.r.get("gain", 0)
        s = (f"<p>A {v(2 ** p.bits, 'd')}-level quantiser replaces every input with the nearest "
             f"red level. The best <b>uniform</b> quantiser (iteration 0) spaces them equally; for a "
             f"{p.pdf.lower()} source that wastes levels in the rarely visited tails.</p>"
             "<p><b>Lloyd's algorithm</b> alternates two rules: each boundary goes halfway between "
             "its levels, each level moves to the centroid (mean) of its cell. Drag the iteration "
             f"slider: the levels huddle toward the peak and the SNR climbs, here by "
             f"{v(g, '.2f', 'dB')}.</p>")
        if p.pdf == "Uniform":
            s += "<p>For a uniform source the uniform quantiser is already optimal: nothing moves.</p>"
        s += (f"<p>The output is not equiprobable either: its entropy is "
              f"{v(self.r.get('H', 0), '.2f', 'bits')}, not {v(p.bits, 'd')}. An entropy coder "
              f"after the quantiser collects that difference, which is why JPEG, MP3 and video "
              f"codecs all pair a simple quantiser with Huffman or arithmetic coding.</p>")
        return "<h3>Placing the levels</h3>" + s + keybox(
            "Lloyd–Max: thresholds midway, levels at centroids. Each extra bit buys about 6 dB.")


# =============================================================================== 4. DCT
class DCTCompaction(Experiment):
    title = "DCT energy compaction"
    blurb = "Pick any 8×8 block: a handful of DCT coefficients rebuilds it."
    book = "sec:ch16:image"
    controls = [
        Choice("img", "Image", ["Photograph", "Synthetic image", "White noise"]),
        IntSlider("bx", "Block column", 0, 31, 16),
        IntSlider("by", "Block row", 0, 31, 13),
        IntSlider("keep", "Coefficients kept (zig-zag order)", 1, 64, 6),
    ]
    plots = [
        ImagePlot("img", "The image (red square = your block)"),
        ImagePlot("blk", "Your 8×8 block"),
        ImagePlot("rec", "Rebuilt from the kept coefficients"),
        ImagePlot("coef", "|DCT coefficients|, kept ones joined"),
        Plot("curve", "Quality of the rebuild vs coefficients kept", x="coefficients kept",
             y="PSNR (dB)", xlim=(0, 65), ylim=(10, 62), legend="br"),
    ]
    layout = [["img", "blk", "rec"], ["coef", "curve", "curve"]]
    readouts = [
        Readout("k35", "Coefficients for 35 dB (DCT)", "", "int"),
        Readout("kpix", "Pixels for 35 dB (no transform)", "", "int"),
        Readout("psnr", "PSNR of your rebuild", "dB", ".1f", good=lambda x: x > 35),
        Readout("gain", "Coding gain (whole image)", "dB", ".1f",
                help="Arithmetic over geometric mean of the coefficient variances"),
    ]
    challenges = [
        Challenge("Find a photograph block that 3 coefficients rebuild at 35 dB or better.",
                  lambda s: s.p.img == "Photograph" and s.r.k35 <= 3,
                  hint="Smooth areas: the background, the jacket."),
        Challenge("Find a photograph block that needs more than 40 coefficients for 35 dB.",
                  lambda s: s.p.img == "Photograph" and s.r.k35 > 40,
                  hint="Edges and fine texture: the stars, the cap badge, the glasses."),
        Challenge("Find a block with real detail (pixel spread above 10) where the DCT needs at "
                  "least 30 fewer coefficients than the pixels for 35 dB.",
                  lambda s: s.p.img != "White noise" and s.exp.blk_std > 10
                  and s.r.kpix - s.r.k35 >= 30,
                  hint="Smooth shading: a gradient is spread over every pixel but sits in two "
                       "or three DCT coefficients."),
    ]

    def setup(self):
        self.zz = sk.zigzag_indices(8)
        self.gains = {}
        self.blk_std = 0.0

    def image(self, name):
        if name == "White noise":
            return np.clip(128 + 40 * np.random.default_rng(3).standard_normal((256, 256)), 0, 255)
        return image_of(name)

    def coding_gain(self, name):
        if name not in self.gains:
            img = self.image(name) - 128
            B = img.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
            C = np.array([sk.dct8(b) for b in B])
            var = C.reshape(-1, 64).var(axis=0) + 1e-9
            self.gains[name] = float(db(var.mean() / np.exp(np.mean(np.log(var)))))
        return self.gains[name]

    def update(self, p):
        img = self.image(p.img)
        r0, c0 = p.by * 8, p.bx * 8
        blk = img[r0:r0 + 8, c0:c0 + 8]
        self.blk_std = float(blk.std())
        C = sk.dct8(blk - 128)
        mask = np.zeros((8, 8), bool)
        for (i, j) in self.zz[:p.keep]:
            mask[i, j] = True
        rec = np.clip(sk.idct8(np.where(mask, C, 0)) + 128, 0, 255)
        pi_ = self.plot("img")
        lock_aspect(pi_)
        show_image(pi_, "im", img)
        y0 = 256 - r0
        pi_.line("box", [c0 - 1, c0 + 9, c0 + 9, c0 - 1, c0 - 1],
                 [y0 + 1, y0 + 1, y0 - 9, y0 - 9, y0 + 1], color=RED, width=2.4)
        for key, data in (("blk", blk), ("rec", rec)):
            pl = self.plot(key)
            lock_aspect(pl)
            show_image(pl, "im", data)
        pc = self.plot("coef")
        lock_aspect(pc)
        show_image(pc, "im", db(C ** 2 + 1e-3), levels=(-10, 60), cmap="heat")
        zx = [j + 0.5 for (i, j) in self.zz[:p.keep]]
        zy = [8 - i - 0.5 for (i, j) in self.zz[:p.keep]]
        pc.line("zz", zx, zy, color=RED, width=1.6)
        pc.scatter("zzp", zx, zy, color=RED, size=6)
        # PSNR vs number kept: DCT in zig-zag order, best DCT coefficients, best pixels
        e = np.array([C[i, j] ** 2 for (i, j) in self.zz])
        k = np.arange(1, 65)

        def ps(err):
            return np.minimum(db(255 ** 2 / np.maximum(err, 1e-9)), 60)

        ps_zz = ps((e.sum() - np.cumsum(e)) / 64)
        es = np.sort(e)[::-1]
        ps_best = ps((es.sum() - np.cumsum(es)) / 64)
        d = np.sort(((blk - blk.mean()) ** 2).ravel())[::-1]     # keep the worst pixels, mean elsewhere
        ps_pix = ps((d.sum() - np.cumsum(d)) / 64)
        pu = self.plot("curve")
        pu.line("pix", k, ps_pix, color=GRAY, width=1.8, style="--",
                name="largest pixels (no transform)")
        pu.line("best", k, ps_best, color=GREEN, width=1.6, name="largest DCT coefficients")
        pu.line("zz", k, ps_zz, color=NAVY, width=2.4, name="DCT, zig-zag order")
        pu.vline("keep", p.keep, color=RED, style=":", width=1.4, label="kept", label_pos=0.95)
        pu.hline("35", 35, color=GRAY, style=":", width=1.0, label="35 dB", label_pos=0.02)

        def first(curve):
            ok = np.nonzero(curve >= 35)[0]
            return int(ok[0] + 1) if len(ok) else 64

        mse = np.mean((rec - blk) ** 2)
        psnr = float(min(db(255 ** 2 / max(mse, 1e-9)), 60))
        self.readout(k35=first(ps_zz), kpix=first(ps_pix), psnr=psnr, gain=self.coding_gain(p.img))

    def story(self, p):
        k35, kp = self.r.get("k35", 0), self.r.get("kpix", 0)
        s = (f"<p>The 8×8 DCT rewrites the 64 pixels of a block as 64 weights of cosine patterns, "
             f"from flat (top left, DC) to fine checkerboards (bottom right). Neighbouring pixels "
             f"are similar, so most of the energy falls on the few low-frequency patterns. Your "
             f"block needs {v(k35, 'd')} coefficients in zig-zag order for a 35 dB rebuild; "
             f"keeping its most deviant <i>pixels</i> instead would need {v(kp, 'd')}.</p>"
             f"<p>Keeping {v(p.keep, 'd')} of 64 rebuilds it at "
             f"{v(self.r.get('psnr', 0), '.1f', 'dB')}. JPEG does exactly this, except that it "
             f"quantises the coefficients instead of dropping them.</p>")
        if p.img == "White noise":
            s += ("<p>" + bad("White noise has no correlation") + ": its energy is spread evenly "
                  "over all coefficients and the transform gains nothing (coding gain near 0 dB).</p>")
        else:
            s += (f"<p>Over the whole image the DCT's <b>coding gain</b> is "
                  f"{v(self.r.get('gain', 0), '.1f', 'dB')}: that much SNR for free at the same bit "
                  f"rate, compared with quantising pixels directly.</p>")
        return "<h3>Energy in a few coefficients</h3>" + s + keybox(
            "Transforms do not compress by themselves; they concentrate energy so that quantisation "
            "can throw most coefficients away.")


# =============================================================================== 5. toy JPEG
JPEG_QS = [3, 6, 10, 15, 22, 30, 40, 50, 60, 70, 80, 88, 95]
_RD = {}


@lru_cache(maxsize=96)
def jpeg(name, q):
    rec, bits = sc.toy_jpeg(image_of(name), q)
    return rec, bits


@lru_cache(maxsize=4)
def pil_curve(name):
    from PIL import Image
    img = image_of(name)
    out = []
    for q in [5, 10, 20, 35, 50, 70, 85, 93]:
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, "JPEG", quality=q, optimize=True)
        n = buf.tell()
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("L")).astype(float)
        out.append((8 * n / img.size, sc.psnr(img, dec)))
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, "PNG", optimize=True)
    return np.array(out), 8 * buf.tell() / img.size


class ToyJPEG(Experiment):
    title = "A toy JPEG"
    blurb = "A complete baseline JPEG coder on the bundled photograph. Turn quality down."
    book = "sec:ch16:image"
    controls = [
        Choice("img", "Image", ["Photograph", "Synthetic image"]),
        IntSlider("q", "Quality", 1, 100, 50,
                  help="IJG quality scaling of the standard luminance quantisation table"),
        Choice("view", "Right-hand view", ["Decoded image", "Error ×4"]),
    ]
    plots = [
        ImagePlot("orig", "Original (8 bits per pixel)"),
        ImagePlot("rec", "Decoded"),
        ImagePlot("zoom", "Zoom: 48 × 48 pixels"),
        Plot("rd", "Rate–distortion: PSNR vs bits per pixel", x="bits per pixel",
             y="PSNR (dB)", xlim=(0, 3), ylim=(20, 52), legend="br"),
    ]
    layout = [["orig", "rec", "zoom"], ["rd", "rd", "rd"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bpp", "Bit rate", "bits/pixel", ".3f"),
        Readout("ratio", "Compression", ": 1", ".1f"),
        Readout("psnr", "PSNR", "dB", ".1f", good=lambda x: x > 35),
        Readout("kb", "File size (no headers)", "kB", ".1f"),
    ]
    challenges = [
        Challenge("Photograph sweet spot: 1 bit per pixel or less with a PSNR of at least 32 dB.",
                  lambda s: s.p.img == "Photograph" and s.r.bpp <= 1 and s.r.psnr >= 32,
                  hint="A narrow window of qualities in the mid fifties."),
        Challenge("Compress the photograph 40:1 or more and zoom in on the blocks.",
                  lambda s: s.p.img == "Photograph" and s.r.ratio >= 40),
        Challenge("Find the quality where the toy coder spends 2.0 ± 0.1 bits per pixel on the "
                  "photograph.",
                  lambda s: s.p.img == "Photograph" and abs(s.r.bpp - 2) <= 0.1),
    ]

    def setup(self):
        for name in ("Photograph", "Synthetic image"):
            pil_curve(name)                      # warm the reference curves (8 encodes each)

    def update(self, p):
        img = image_of(p.img)
        rec, bits = jpeg(p.img, p.q)
        bpp = bits / img.size
        psnr = sc.psnr(img, rec)
        po = self.plot("orig")
        lock_aspect(po)
        show_image(po, "im", img)
        y0, x0 = (96, 104) if p.img == "Photograph" else (60, 60)
        po.line("box", [x0, x0 + 48, x0 + 48, x0, x0],
                [256 - y0, 256 - y0, 208 - y0, 208 - y0, 256 - y0], color=RED, width=2.0)
        pr = self.plot("rec")
        lock_aspect(pr)
        if p.view == "Decoded image":
            pr.set_title(f"Decoded at quality {p.q}")
            show_image(pr, "im", rec)
            z = rec[y0:y0 + 48, x0:x0 + 48]
            pz_levels, cm = (0, 255), GRAYS
        else:
            pr.set_title("Error ×4 (mid-gray = 0)")
            show_image(pr, "im", np.clip(128 + 4 * (rec - img), 0, 255))
            z = 128 + 4 * (rec - img)[y0:y0 + 48, x0:x0 + 48]
            pz_levels, cm = (0, 255), GRAYS
        pz = self.plot("zoom")
        lock_aspect(pz)
        show_image(pz, "im", np.clip(z, 0, 255), levels=pz_levels, cmap=cm)
        grid_x, grid_y = [], []
        for g in range(0, 49, 8):
            off = (8 - x0 % 8) % 8
            gx = g + off
            if gx <= 48:
                grid_x += [gx, gx, np.nan]
                grid_y += [0, 48, np.nan]
            gy = 48 - (g + (8 - y0 % 8) % 8)
            if gy >= 0:
                grid_x += [0, 48, np.nan]
                grid_y += [gy, gy, np.nan]
        pz.line("grid", grid_x, grid_y, color=ORANGE, width=0.8, alpha=0.6)
        prd = self.plot("rd")
        pts, png = pil_curve(p.img)
        prd.line("pil", pts[:, 0], pts[:, 1], color=GRAY, width=1.6, style="--",
                 name="libjpeg (Pillow, with headers)")
        prd.scatter("pilp", pts[:, 0], pts[:, 1], color=GRAY, size=7, symbol="s")
        prd.text("png", 2.95, 51, f"PNG (lossless) needs {png:.2f} bpp →", color=PURPLE, size=9,
                 anchor=(1, 0))
        toy = _RD.get(p.img, {})
        if toy:
            qq = sorted(toy)
            prd.line("toy", [toy[q][0] for q in qq], [toy[q][1] for q in qq], color=NAVY, width=2.2,
                     name="this toy coder (no headers)")
        prd.scatter("now", [bpp], [psnr], color=RED, size=15, symbol="d", name="your quality")
        prd.set_xlim(0, max(3.0, bpp * 1.05))
        self.readout(bpp=bpp, ratio=8 / bpp, psnr=psnr, kb=bits / 8 / 1024)

    def background(self, p):
        name = p.img
        done = _RD.setdefault(name, {})
        todo = [q for q in JPEG_QS if q not in done]
        if self.quick:
            todo = todo[:2]
        for q in todo:
            rec, bits = jpeg(name, q)
            img = image_of(name)
            yield (name, q, bits / img.size, sc.psnr(img, rec))

    def progress(self, p, item):
        name, q, bpp, ps = item
        _RD.setdefault(name, {})[q] = (bpp, ps)
        toy = _RD[name]
        qq = sorted(toy)
        self.plot("rd").line("toy", [toy[k][0] for k in qq], [toy[k][1] for k in qq], color=NAVY,
                             width=2.2, name="this toy coder (no headers)")

    def story(self, p):
        bpp, psnr = self.r.get("bpp", 1), self.r.get("psnr", 30)
        s = (f"<p>Baseline JPEG in five steps: cut the image into 8×8 blocks, DCT each one, divide "
             f"every coefficient by a step from the quantisation table (scaled by your quality "
             f"{v(p.q, 'd')}), round, and entropy-code the zig-zag runs of zeros with Huffman "
             f"codes. Result: {v(bpp, '.3f', 'bits per pixel')} ({v(8 / bpp, '.1f')}:1) at "
             f"{v(psnr, '.1f', 'dB')}.</p>")
        if psnr < 28:
            s += ("<p>" + bad("Artefacts.") + " At low quality whole blocks collapse to their DC "
                  "value (<b>blocking</b>) and edges ring with the few surviving cosines "
                  "(<b>ringing</b>). The zoom shows the 8×8 grid (orange).</p>")
        elif psnr > 38:
            s += ("<p>At this quality the error is invisible; look at the error view to see it "
                  "is mostly fine noise near edges.</p>")
        else:
            s += ("<p>This is the everyday JPEG regime: around 1 bit per pixel, errors hidden in "
                  "textured areas where the eye is least sensitive. (This portrait is grainy, so "
                  "its PSNR runs a few dB lower than on smooth images.)</p>")
        s += ("<p>The curve: this coder, without headers, sits close to the real libjpeg; PNG "
              "(lossless) needs several times more bits. JPEG 2000, HEIC and AVIF do better at low "
              "rates, mainly by avoiding blocks.</p>")
        return "<h3>Quantise what the eye forgives</h3>" + s + keybox(
            "JPEG = DCT + perceptual quantisation table + run-length + Huffman. Quality scales the table.")


# =============================================================================== 6. LPC
FS_V = 8000
VOWELS = {"/a/ (father)": sc.FORMANTS,
          "/i/ (see)": [(270, 60), (2290, 100), (3010, 150), (3500, 250)],
          "/u/ (boot)": [(300, 60), (870, 80), (2240, 120), (3400, 250)]}


@lru_cache(maxsize=48)
def vowel(name, f0):
    return sc.synth_vowel(dur=0.6, f0=float(f0), fs=FS_V, formants=VOWELS[name])


class LPCVocoder(Experiment):
    title = "LPC: a buzz and a filter"
    blurb = "Model the vocal tract as an all-pole filter; drive it with a buzz or a hiss."
    book = "sec:ch16:lpc"
    controls = [
        Heading("The speaker"),
        Choice("vow", "Vowel", list(VOWELS), style="menu"),
        Slider("f0", "Original pitch", 80, 320, 120, step=5, unit="Hz"),
        Heading("The vocoder"),
        IntSlider("order", "LPC order p", 2, 24, 10),
        Choice("exc", "Resynthesis excitation", ["Pulse train", "Noise (whisper)"]),
        Slider("f0new", "New pitch", 60, 400, 180, step=5, unit="Hz",
               enabled_if=lambda p: p.exc == "Pulse train"),
        Button("listen", "▶  Listen: original", primary=True),
        Button("listen2", "▶  Listen: LPC resynthesis", primary=True),
    ]
    plots = [
        Plot("spec", "Spectrum and the LPC envelope (green: true formants)", x="frequency (Hz)",
             y="level (dB)", xlim=(0, 4000), legend="tr"),
        Plot("z", "Poles (×) and line spectral frequencies (○)", x="real", y="imaginary",
             xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), legend=None, aspect=True),
        Plot("wave", "Speech and the prediction residual", x="time (ms)", y="", xlim=(200, 240),
             ylim=(-2.4, 1.3), legend="tr", legend_cols=2),
        Plot("out", "Original vs LPC resynthesis", x="time (ms)", y="", xlim=(300, 325),
             ylim=(-1.4, 1.4), legend="tr", legend_cols=2),
    ]
    layout = [["spec", "z"], ["wave", "out"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("pg", "Prediction gain", "dB", ".1f"),
        Readout("nf", "Formants found", "of 4", "int", good=lambda x: x == 4,
                help="True formants with an LPC pole (radius > 0.8) within 10 % (at least 60 Hz)"),
        Readout("err", "F1 error", "Hz", ".0f", good=lambda x: abs(x) < 60),
        Readout("rate", "Vocoder bit rate", "kb/s", ".1f",
                help="3 bits per LSF + 5 (gain) + 7 (pitch) every 20 ms"),
    ]
    challenges = [
        Challenge("Two poles per formant: find an order of 8 or less that finds all four "
                  "formants of /i/.",
                  lambda s: s.p.vow.startswith("/i/") and s.p.order <= 8 and s.r.nf == 4),
        Challenge("High voices fool LPC: /a/ at 280 Hz or more, order 10 or more, and the first "
                  "formant missed by more than 80 Hz.",
                  lambda s: s.p.vow.startswith("/a/") and s.p.f0 >= 280 and s.p.order >= 10
                  and abs(s.r.err) > 80,
                  hint="With widely spaced harmonics the all-pole fit locks onto a harmonic."),
        Challenge("A 2.4 kb/s vocoder for a 250 Hz voice or higher that still finds all four "
                  "formants.",
                  lambda s: s.p.f0 >= 250 and s.r.rate <= 2.4 and s.r.nf == 4),
    ]

    def analyse(self, p):
        x = vowel(p.vow, int(p.f0))
        fr = x[1600:1856]
        (a, ks, Es), xw = sc.lpc_frame(fr, p.order)
        return x, fr, a, Es, xw

    def update(self, p):
        x, fr, a, Es, xw = self.analyse(p)
        nfft = 1024
        fq = np.arange(nfft // 2 + 1) * FS_V / nfft
        Pxx = np.abs(np.fft.rfft(xw, nfft)) ** 2
        _, Hh = signal.freqz([1], a, worN=fq, fs=FS_V)
        env = Es[-1] * np.abs(Hh) ** 2
        ps = self.plot("spec")
        top = db(env.max())
        ps.line("p", fq, db(Pxx + 1e-12), color=GRAY, width=1.0, name="periodogram (harmonics)")
        ps.line("e", fq, db(env), color=RED, width=2.4, name=f"LPC envelope, p = {p.order}")
        for i, (F, _) in enumerate(VOWELS[p.vow]):
            ps.vline(f"F{i}", F, color=GREEN, style=":", width=1.3, label=f"F{i + 1}",
                     label_pos=0.08)
        ps.set_ylim(top - 70, top + 10)
        poles = np.roots(a)
        lsf = sc.lpc_to_lsf(a)
        pz = self.plot("z")
        th = np.linspace(0, 2 * np.pi, 200)
        pz.line("uc", np.cos(th), np.sin(th), color=GRAY, width=1.0)
        pz.scatter("lsf", np.cos(lsf), np.sin(lsf), color=NAVY, size=8, outline=NAVY,
                   name="LSFs (on the circle)")
        pz.scatter("poles", poles.real, poles.imag, color=RED, size=11, symbol="x", name="poles")
        # formant estimates
        up = poles[(poles.imag > 1e-6) & (np.abs(poles) > 0.8)]
        ff = np.sort(np.angle(up) * FS_V / (2 * np.pi))
        F = VOWELS[p.vow]
        errs = [float(ff[np.argmin(np.abs(ff - f))] - f) if len(ff) else 9999.0 for f, _ in F]
        nf = int(sum(abs(e) <= max(0.1 * f, 60) for e, (f, _) in zip(errs, F)))
        # residual and resynthesis
        e = signal.lfilter(a, [1], x)
        t = np.arange(len(x)) / FS_V * 1e3
        pw = self.plot("wave")
        k = (t >= 195) & (t <= 245)
        pw.line("x", t[k], x[k] / np.max(np.abs(x)), color=NAVY, width=1.8, name="speech")
        pw.line("e", t[k], 3 * e[k] / np.max(np.abs(x)) - 1.45, color=RED, width=1.4,
                name="residual × 3")
        y = self.resynth(p, x)
        po = self.plot("out")
        k2 = (t >= 295) & (t <= 345)
        sx = np.max(np.abs(x[k2])) + 1e-9
        po.line("x", t[k2], x[k2] / sx, color=GRAY, width=2.4, alpha=0.6,
                name=f"original ({p.f0:.0f} Hz)")
        po.line("y", t[k2], y[k2] / (np.max(np.abs(y[k2])) + 1e-9), color=NAVY, width=1.6,
                name="whisper" if p.exc != "Pulse train" else f"resynthesised ({p.f0new:.0f} Hz)")
        pg = float(db(Es[0] / Es[-1]))
        self._st = dict(ff=ff[:5])
        self.readout(pg=pg, nf=nf, err=errs[0] if abs(errs[0]) < 9000 else "—",
                     rate=(3 * p.order + 5 + 7) / 0.02 / 1e3)

    def resynth(self, p, x):
        return sc.lpc_analysis_synthesis(x, FS_V, p.order,
                                         f0_new=None if p.exc != "Pulse train" else p.f0new)

    def on_listen(self, p):
        x = vowel(p.vow, int(p.f0))
        self.play_audio(np.tile(x, 2) * 0.8, FS_V, "the original vowel")

    def on_listen2(self, p):
        x = vowel(p.vow, int(p.f0))
        y = self.resynth(p, x)
        self.play_audio(np.tile(y / (np.max(np.abs(y)) + 1e-9), 2) * 0.8, FS_V,
                        "the LPC resynthesis")

    def story(self, p):
        stt = getattr(self, "_st", {"ff": []})
        ff = ", ".join(f"{f:.0f}" for f in stt["ff"]) or "none"
        s = ("<p>Your voice is a buzz from the vocal folds (pitch) shaped by the throat and mouth "
             "(formants). <b>Linear prediction</b> fits an all-pole filter 1/A(z) that predicts "
             "each sample from the previous p: its response is the smooth red envelope, and its "
             f"poles near the unit circle sit on the formants (poles at {v(ff, 's')} Hz; "
             f"{v(self.r.get('nf', 0), 'd')} of 4 formants found).</p>"
             f"<p>What the predictor cannot predict, the <b>residual</b>, is almost a pulse train "
             f"at the pitch: the filter removed {v(self.r.get('pg', 0), '.1f', 'dB')} of the "
             f"signal's power. A vocoder sends only the filter (as line spectral frequencies, "
             f"the circles on the unit circle), a gain and the pitch: "
             f"{v(self.r.get('rate', 0), '.1f', 'kb/s')} instead of 64 kb/s PCM.</p>"
             "<p>Swap the excitation and the same mouth speaks at a new pitch, or whispers: "
             "press Listen. Phone codecs (AMR, EVS) refine this with CELP, which searches a "
             "codebook for the best excitation.</p>")
        return "<h3>Source and filter</h3>" + s + keybox(
            "Speech ≈ excitation (pitch) through an all-pole vocal-tract filter. Send the filter, "
            "not the waveform.")


# =============================================================================== 7. masking
FS_A = 44100
NFR = 2048


def tone(f, level_db, n=NFR, fs=FS_A, phase=0.0):
    return 10 ** ((level_db - 96) / 20) * np.sin(2 * np.pi * f * np.arange(n) / fs + phase)


class MaskingTone(Experiment):
    title = "Masking: hide a tone"
    blurb = "A loud tone makes nearby quiet tones inaudible. Find out how near and how quiet."
    book = "sec:ch16:audio"
    controls = [
        Heading("Masker"),
        LogSlider("fm", "Masker frequency", 100, 10000, 1000, unit="Hz", fmt=".0f"),
        Slider("lm", "Masker level", 20, 90, 70, step=1, unit="dB SPL"),
        Heading("Probe"),
        LogSlider("fp", "Probe frequency", 50, 16000, 1400, unit="Hz", fmt=".0f"),
        Slider("lp", "Probe level", -10, 90, 45, step=1, unit="dB SPL"),
        Button("listen", "▶  Listen: masker + beeping probe", primary=True),
    ]
    plots = [
        Plot("thr", "The masking threshold: anything below the red curve is inaudible",
             x="frequency (Hz)", y="level (dB SPL)", logx=True, xlim=(20, 20000),
             ylim=(-15, 105), legend="tr"),
        Plot("bark", "The same on the Bark (critical-band) scale", x="critical-band rate (Bark)",
             y="level (dB SPL)", xlim=(0, 25), ylim=(-15, 105), legend=None),
    ]
    layout = [["thr"], ["bark"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("thr", "Threshold at the probe", "dB SPL", ".1f"),
        Readout("margin", "Probe above threshold", "dB", ".1f"),
        Readout("aud", "Probe is", "", None),
        Readout("dz", "Distance from masker", "Bark", ".2f"),
    ]
    challenges = [
        Challenge("Hide a probe of at least 55 dB SPL completely.",
                  lambda s: s.p.lp >= 55 and s.r.margin < 0),
        Challenge("Upward spread: hide a 40 dB probe a full octave or more above the masker.",
                  lambda s: s.p.fp >= 2 * s.p.fm and s.p.lp >= 40 and s.r.margin < 0,
                  hint="Masking spreads further upward than downward, more so for loud maskers."),
        Challenge("Make a probe 40 dB weaker than the masker still clearly audible (at least 10 dB "
                  "above threshold).",
                  lambda s: s.p.lm - s.p.lp >= 40 and s.r.margin >= 10),
    ]

    def update(self, p):
        x = tone(p.fm, p.lm)
        f, P, T = sc.masking_threshold(x, FS_A)
        k = (f >= 20) & (f <= 20000)
        pt = self.plot("thr")
        ath = sc.ath(np.maximum(f[k], 20))
        pt.line("P", f[k], np.maximum(P[k], -14), color=GRAY, width=1.0, fill=-15, fill_alpha=0.15,
                name="masker spectrum")
        pt.line("ath", f[k], np.minimum(ath, 104), color=GREEN, width=1.6, style="--",
                name="threshold in quiet")
        pt.line("T", f[k], np.minimum(T[k], 104), color=RED, width=2.6, name="masking threshold")
        thr = float(np.interp(np.log(p.fp), np.log(np.maximum(f[1:], 1)), T[1:]))
        margin = p.lp - thr
        pt.scatter("probe", [p.fp], [p.lp], color=GREEN if margin >= 0 else RED, size=16,
                   symbol="d", name="probe")
        pt.line("pl", [p.fp, p.fp], [-15, p.lp], color=GREEN if margin >= 0 else RED, width=1.2,
                style=":")
        z = sc.bark(np.maximum(f[k], 1))
        pb = self.plot("bark")
        pb.line("P", z, np.maximum(P[k], -14), color=GRAY, width=1.0, fill=-15, fill_alpha=0.15)
        pb.line("ath", z, np.minimum(ath, 104), color=GREEN, width=1.4, style="--")
        pb.line("T", z, np.minimum(T[k], 104), color=RED, width=2.4)
        zm, zp = float(sc.bark(p.fm)), float(sc.bark(p.fp))
        pb.scatter("probe", [zp], [p.lp], color=GREEN if margin >= 0 else RED, size=14, symbol="d")
        pb.vline("zm", zm, color=NAVY, style=":", width=1.2, label="masker", label_pos=0.95)
        self.readout(thr=thr, margin=margin, aud="audible" if margin >= 0 else "masked",
                     dz=zp - zm)

    def on_listen(self, p):
        n = int(1.8 * FS_A)
        t = np.arange(n) / FS_A
        m = tone(p.fm, p.lm, n)
        gate = ((t % 0.5) < 0.25).astype(float)
        gate = np.convolve(gate, np.hanning(441) / np.hanning(441).sum(), "same")
        y = m + tone(p.fp, p.lp, n) * gate
        y = y / max(1.0, np.max(np.abs(y)) / 0.9)
        self.play_audio(y, FS_A, "masker and probe")

    def story(self, p):
        m = self.r.get("margin", 0)
        s = (f"<p>A {v(p.fm, '.0f', 'Hz')} tone at {v(p.lm, '.0f', 'dB')} excites a whole region "
             f"of the inner ear's basilar membrane. Any other sound that falls in that region and "
             f"is weaker than the red <b>masking threshold</b> is simply not heard. Your probe at "
             f"{v(p.fp, '.0f', 'Hz')} is {v(abs(m), '.1f', 'dB')} "
             + (good("above the threshold: audible.") if m >= 0 else bad("below the threshold: masked.")) +
             "</p>"
             "<p>On the Bark scale (bottom), which follows the ear's critical bands, the threshold "
             "is a tent: steep on the low side, shallow on the high side. Loud sounds mask upward "
             "far more than downward. Press Listen: the probe beeps on and off, if you can hear it.</p>"
             "<p>MP3 and AAC compute this threshold every few milliseconds and spend bits only "
             "where the quantisation noise would poke above it.</p>")
        return "<h3>The ear's blind spots</h3>" + s + keybox(
            "Perceptual coding: noise below the masking threshold costs nothing, because nobody hears it.")


# =============================================================================== 8. shaped noise
@lru_cache(maxsize=4)
def music(kind, dur=1.2):
    n = np.arange(int(dur * FS_A))
    r = np.random.default_rng(2)
    if kind == "Harmonic tone + hiss":
        m = sum(0.5 / h ** 1.1 * np.cos(2 * np.pi * 220 * h * (1 + 0.0005 * h * h) / FS_A * n
                                        + r.uniform(0, 2 * np.pi)) for h in range(1, 15))
        bb, aa = signal.butter(4, [6000, 9000], btype="band", fs=FS_A)
        m = m + signal.lfilter(bb, aa, 0.08 * r.standard_normal(len(n)))
    else:
        m = 0
        for f0 in (261.6, 329.6, 392.0):                     # C major chord
            m = m + sum(0.4 / h ** 1.4 * np.cos(2 * np.pi * f0 * h / FS_A * n + r.uniform(0, 6.3))
                        for h in range(1, 10))
        m = m * np.exp(-n / FS_A / 1.5)
    return 0.5 * m / np.max(np.abs(m))


@lru_cache(maxsize=4)
def shaped(kind):
    x = music(kind)
    y, nz = sc.shaped_noise(x, FS_A, 0.0, seed=1)
    return nz


class NoiseUnderMask(Experiment):
    title = "Noise under the mask"
    blurb = "Add a lot of noise, shaped so the ear cannot hear it. Then listen."
    book = "sec:ch16:audio"
    controls = [
        Choice("kind", "Music", ["Harmonic tone + hiss", "Chord (three notes)"]),
        Slider("off", "Noise relative to the threshold", -20, 20, -6, step=1, unit="dB"),
        Choice("noise", "Noise", ["Shaped under the mask", "White, same power"]),
        Slider("tpos", "Frame shown", 0.1, 1.0, 0.6, step=0.05, unit="s"),
        Button("listen", "▶  Listen: music alone", primary=True),
        Button("listen2", "▶  Listen: music + noise", primary=True),
    ]
    plots = [
        Plot("frame", "One frame: music, allowed noise and the noise you added", x="frequency (kHz)",
             y="level (dB SPL)", xlim=(0, 20), ylim=(-30, 100), legend="tr"),
        BarPlot("nmr", "Noise-to-mask ratio per critical band (above 0 = audible)",
                x="critical band (Bark)", y="NMR (dB)", ylim=(-30, 40)),
    ]
    layout = [["frame"], ["nmr"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("snr", "SNR", "dB", ".1f"),
        Readout("aud", "Bands with audible noise", "of 24", "int", good=lambda x: x == 0),
        Readout("worst", "Worst band NMR", "dB", ".1f", good=lambda x: x < 0),
        Readout("bits", "Bits a PCM coder would need", "bits/sample", ".1f",
                help="SNR / 6.02: a plain quantiser with this much noise, versus 16 for CD audio"),
    ]
    challenges = [
        Challenge("Hide noise only 10 dB below the music (SNR under 10 dB) with no audible band.",
                  lambda s: s.p.noise.startswith("Shaped") and s.r.snr < 10 and s.r.aud == 0,
                  hint="Some music masks much more than other music."),
        Challenge("Same noise power, wrong shape: switch to white noise at the same SNR and make "
                  "at least 10 bands audible.",
                  lambda s: s.p.noise.startswith("White") and s.r.aud >= 10),
        Challenge("Go too far: shaped noise 10 dB or more above the threshold, and listen.",
                  lambda s: s.p.noise.startswith("Shaped") and s.p.off >= 10),
    ]

    def setup(self):
        for kind in ("Harmonic tone + hiss", "Chord (three notes)"):
            shaped(kind)                          # shape the noise once (about 0.25 s each)

    def noise(self, p):
        x = music(p.kind)
        nz = shaped(p.kind) * 10 ** (p.off / 20)
        if p.noise.startswith("White"):
            w = np.random.default_rng(4).standard_normal(len(x))
            nz = w * np.sqrt(np.mean(nz ** 2))
        return x, nz

    def update(self, p):
        x, nz = self.noise(p)
        s0 = int(np.clip(p.tpos * FS_A, 0, len(x) - NFR))
        f, P, T = sc.masking_threshold(x[s0:s0 + NFR], FS_A)
        _, Pn, _ = sc.masking_threshold(nz[s0:s0 + NFR], FS_A)
        zb = np.floor(sc.bark(np.maximum(f, 1))).astype(int)
        allow = T - 10 * np.log10(np.bincount(zb)[zb])
        fk = f / 1e3
        pf = self.plot("frame")
        pf.line("P", fk, P, color=NAVY, width=1.0, name="music frame")
        pf.line("T", fk, T, color=RED, width=1.2, style="--", name="masking threshold (per band)")
        pf.line("A", fk, allow, color=RED, width=2.0, name="allowed noise per FFT bin")
        pf.line("N", fk, Pn, color=GREEN if p.noise.startswith("Shaped") else ORANGE, width=0.9,
                alpha=0.85, name="your noise")
        # NMR per band (power sums), bands below 16 kHz
        band = (f > 20) & (f < 16000)
        nmr = []
        for b in range(25):
            m = band & (zb == b)
            if np.any(m):
                pn = 10 * np.log10(np.sum(10 ** (Pn[m] / 10)))
                pa = 10 * np.log10(np.sum(10 ** (allow[m] / 10)))
                nmr.append((b, pn - pa))
        bb = np.array([b for b, _ in nmr])
        vv = np.array([q for _, q in nmr])
        pn_ = self.plot("nmr")
        pn_.bars("b", bb, np.clip(vv, -30, 40), width=0.8,
                 colors=[RED if q > 0 else GREEN for q in vv])
        pn_.hline("z", 0, color=GRAY, style="-", width=1.0)
        pn_.set_xlim(-0.6, 24.6)
        snr = float(db(np.mean(x ** 2) / np.mean(nz ** 2)))
        self.readout(snr=snr, aud=int(np.sum(vv > 0)), worst=float(vv.max()),
                     bits=max(0.0, snr / 6.02))

    def on_listen(self, p):
        self.play_audio(music(p.kind), FS_A, "the music")

    def on_listen2(self, p):
        x, nz = self.noise(p)
        y = x + nz
        self.play_audio(y / max(1.0, np.max(np.abs(y)) / 0.95), FS_A, "music + noise")

    def story(self, p):
        snr, aud = self.r.get("snr", 0), self.r.get("aud", 0)
        s = (f"<p>The noise you added is only {v(snr, '.1f', 'dB')} below the music: a coder that "
             f"made this much error would need about {v(snr / 6.02, '.1f', 'bits')} per sample "
             f"instead of 16. ")
        if p.noise.startswith("Shaped"):
            s += ("It is shaped to follow the red threshold, hiding under every loud harmonic and "
                  "pulling back where the music is quiet. ")
            s += (good("No critical band exceeds its mask: it should be inaudible.") if aud == 0
                  else bad(f"{aud} bands poke above their mask: you may hear it."))
        else:
            s += ("This time it is <b>white</b>: the same power spread evenly. It sits far above "
                  "the threshold in the quiet gaps between harmonics and at high frequency: "
                  + bad(f"{aud} bands are audible."))
        s += ("</p><p>Press both Listen buttons and compare. That is the whole trick of MP3 and "
              "AAC: quantise each band just coarsely enough that its noise stays under the mask. "
              "A 128 kb/s MP3 is an 11:1 compression of CD audio that most listeners cannot tell "
              "apart.</p>")
        return "<h3>Inaudible noise is free</h3>" + s + keybox(
            "Perceptual coders minimise audible noise (NMR), not SNR.")


# =============================================================================== 9. motion
class MotionSearch(Experiment):
    title = "Motion estimation"
    blurb = "Predict the next video frame by moving blocks of the last one."
    book = "sec:ch16:video"
    controls = [
        Heading("The scene"),
        IntSlider("panx", "Camera pan →", -6, 6, 3, unit="px"),
        IntSlider("pany", "Camera pan ↓", -6, 6, 2, unit="px"),
        IntSlider("vx", "Ball moves →", -12, 12, 9, unit="px"),
        IntSlider("vy", "Ball moves ↓", -12, 12, -6, unit="px"),
        Heading("The encoder"),
        Choice("blk", "Block size", ["8 × 8", "16 × 16"], "16 × 16"),
        IntSlider("search", "Search range ±", 0, 12, 7, unit="px"),
    ]
    plots = [
        ImagePlot("cur", "Current frame and motion vectors"),
        ImagePlot("diff", "|Frame difference|"),
        ImagePlot("res", "|Motion-compensated residual|"),
        ImagePlot("sad", "Search surface for the ball's block (SAD)", x="dx (px)",
                  y="dy (px, + = down)"),
    ]
    layout = [["cur", "diff"], ["sad", "res"]]
    readouts = [
        Readout("e0", "Frame-difference energy", "MSE", ".0f"),
        Readout("e1", "Residual energy", "MSE", ".0f"),
        Readout("gain", "Prediction gain", "dB", ".1f", good=lambda x: x > 10),
        Readout("ops", "SAD evaluations", "×1000", ".1f"),
    ]
    challenges = [
        Challenge("Make motion compensation worth more than 15 dB over plain frame differencing.",
                  lambda s: s.r.gain > 15),
        Challenge("Starve the search: pan the camera 5 px or more and shrink the search range "
                  "until the gain falls below 4 dB.",
                  lambda s: max(abs(s.p.panx), abs(s.p.pany)) >= 5 and s.r.gain < 4,
                  hint="Watch the green box in the search surface lose the pale spot."),
        Challenge("Small blocks follow the ball better: with 8 × 8 blocks get the residual below "
                  "40 % of what 16 × 16 blocks leave.",
                  lambda s: s.p.blk.startswith("8") and s.exp.ratio816 < 0.4),
    ]

    def setup(self):
        self.ratio816 = 1.0
        self.big = photo().astype(float)

    def make_frame(self, dx, dy, ox, oy, seed):
        f = self.big[60 + dy:60 + dy + 128, 64 + dx:64 + dx + 128].copy()
        yy, xx = np.mgrid[:128, :128]
        m = (xx - ox) ** 2 + (yy - oy) ** 2 < 13 ** 2
        tex = 200 + 40 * np.sin((xx - ox) / 2.5) * np.cos((yy - oy) / 3.0)
        f[m] = tex[m]
        return f + 1.5 * np.random.default_rng(seed).standard_normal(f.shape)

    def update(self, p):
        f0 = self.make_frame(0, 0, 50, 70, 1)
        f1 = self.make_frame(p.panx, p.pany, 50 + p.vx, 70 + p.vy, 2)
        B = 8 if p.blk.startswith("8") else 16
        mv, pred, ops = sk.block_motion(f1, f0, B, p.search)
        e0 = float(np.mean((f1 - f0) ** 2))
        e1 = float(np.mean((f1 - pred) ** 2))
        if B == 8:
            _, pred16, _ = sk.block_motion(f1, f0, 16, p.search)
            self.ratio816 = e1 / max(float(np.mean((f1 - pred16) ** 2)), 1e-9)
        else:
            self.ratio816 = 1.0
        pc = self.plot("cur")
        lock_aspect(pc)
        show_image(pc, "im", f1)
        ax, ay = [], []
        hx, hy = [], []
        for by in range(mv.shape[0]):
            for bx in range(mv.shape[1]):
                dy, dx = mv[by, bx]
                cx, cy = bx * B + B / 2, 128 - (by * B + B / 2)
                ax += [cx, cx + 1.4 * dx, np.nan]
                ay += [cy, cy - 1.4 * dy, np.nan]
                hx.append(cx + 1.4 * dx)
                hy.append(cy - 1.4 * dy)
        pc.line("mv", ax, ay, color=RED, width=2.0)
        pc.scatter("mvh", hx, hy, color=RED, size=4)
        for key, data in (("diff", f1 - f0), ("res", f1 - pred)):
            pl = self.plot(key)
            lock_aspect(pl)
            show_image(pl, "im", np.abs(data), levels=(0, 60), cmap="heat")
        # SAD surface of the block that contains the ball's centre
        by = int(np.clip((70 + p.vy) // B, 0, 128 // B - 1))
        bx = int(np.clip((50 + p.vx) // B, 0, 128 // B - 1))
        cur = f1[by * B:(by + 1) * B, bx * B:(bx + 1) * B]
        S = 12
        sad = np.full((2 * S + 1, 2 * S + 1), np.nan)
        for dy in range(-S, S + 1):
            for dx in range(-S, S + 1):
                y, x = by * B + dy, bx * B + dx
                if 0 <= y and y + B <= 128 and 0 <= x and x + B <= 128:
                    sad[dy + S, dx + S] = np.abs(cur - f0[y:y + B, x:x + B]).mean()
        fin = np.where(np.isfinite(sad), sad, np.nanmax(sad))
        ps = self.plot("sad")
        lock_aspect(ps)
        ps.image("im", fin, x=(-S - 0.5, S + 0.5), y=(-S - 0.5, S + 0.5), cmap="heat",
                 levels=(0, float(np.nanpercentile(sad, 95))), colorbar=True,
                 cbar_label="mean |error|")
        ps.vb.setRange(xRange=(-S - 0.5, S + 0.5), yRange=(-S - 0.5, S + 0.5), padding=0)
        r = p.search
        ps.line("win", [-r - 0.5, r + 0.5, r + 0.5, -r - 0.5, -r - 0.5],
                [-r - 0.5, -r - 0.5, r + 0.5, r + 0.5, -r - 0.5], color=GREEN, width=2.0)
        i, j = np.unravel_index(np.nanargmin(sad), sad.shape)
        ps.scatter("best", [j - S], [i - S], color=GREEN, size=12, symbol="+")
        dyc, dxc = mv[by, bx]
        ps.scatter("chosen", [dxc], [dyc], color=RED, size=12, symbol="x")
        ps.text("lab", -S, S, "green box: search range · + true best · × chosen", color=None,
                size=8, anchor=(0, 0), fill=True)
        self.readout(e0=e0, e1=e1, gain=float(db(e0 / max(e1, 1e-9))), ops=ops / 1000)

    def story(self, p):
        g = self.r.get("gain", 0)
        s = ("<p>Consecutive video frames are nearly identical, but not pixel for pixel: the "
             "camera pans and things move. Subtracting the previous frame (middle) leaves bright "
             "edges everywhere. Instead the encoder cuts the new frame into blocks and, for each, "
             "searches the previous frame for the best-matching block: the red arrows are those "
             "<b>motion vectors</b>.</p>"
             f"<p>Predicting from the moved blocks leaves the residual on the right, "
             f"{v(g, '.1f', 'dB')} weaker than the plain difference. Only the vectors and that "
             f"residual (DCT-coded, as in JPEG) are sent. The cost is search: "
             f"{v(self.r.get('ops', 0), '.0f')} thousand block comparisons for this small frame, "
             f"which is why real encoders use clever, non-exhaustive searches.</p>"
             "<p>Bottom left is the search seen from one block (the ball's): the error for every "
             "candidate shift, palest at the best match. If the green search box does not reach "
             "that pale spot, the encoder settles for a worse one. Blocks that straddle two motions "
             "cannot follow both: H.264 and HEVC split blocks down to 4×4 where needed.</p>")
        return "<h3>Send the motion, not the picture</h3>" + s + keybox(
            "Video coding = motion-compensated prediction + transform coding of the residual.")


# =============================================================================== the lab
LAB = st.Lab(24, "Source Coding", chapter=16,
             chapter_title="Source Coding: Voice, Audio, Images and Video",
             experiments=[HuffmanArithmetic, LZ77, Quantisers, DCTCompaction, ToyJPEG, LPCVocoder,
                          MaskingTone, NoiseUnderMask, MotionSearch])

if __name__ == "__main__":
    st.run(LAB)
