"""Lab 13 · Line Codes, Scramblers, Jitter and SerDes Eyes   (Chapter 8)

Run it:      python labs/lab13_line_codes_eyes.py
Self-test:   python labs/lab13_line_codes_eyes.py --selftest

Every wire that carries bits, from a T1 span to a 112 Gb/s SerDes lane, has the same problems:
no DC through transformers and coupling capacitors, enough transitions to recover a clock, a
channel that eats high frequencies by tens of dB, and a 10⁻¹² error rate despite jitter. Type
bit patterns into line codes, watch a baseline wander, break a scrambler, pin 8b/10b's running
disparity, read total jitter off a bathtub, equalise NRZ and PAM-4 over a lossy backplane, and
see duobinary's error propagation vanish with one XOR. (Lab 3 covers Nyquist pulses.)
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from PySide6 import QtCore, QtWidgets
from scipy.signal import fftconvolve, lfilter, welch

import commlib as cl
from commlib import linecodes as lc
from commlib import serdes as sd
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading, Plot,
                    SpectrumPlot, EyePlot, BERPlot, BarPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL, PALETTE)
from studio import v, keybox, good, bad
from studio.controls import Control


# =============================================================================== a text control
class BitField(Control):
    """A one-line text box for a bit pattern (only 0 and 1 are kept). Implemented here because
    studio has no text-entry control yet."""

    def __init__(self, key, label, default="", help="", maxlen=48):
        super().__init__(key, label, help)
        self.default = default
        self.maxlen = maxlen
        self._v = default

    def _clean(self, s):
        s = "".join(c for c in str(s) if c in "01")[:self.maxlen]
        return s or "0"

    def make_widget(self, on_change):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(2, 4, 2, 2)
        lay.setSpacing(3)
        name = QtWidgets.QLabel(self.label)
        name.setObjectName("ctlname")
        lay.addWidget(name)
        self._edit = QtWidgets.QLineEdit(self._v)
        self._edit.setMinimumHeight(26)
        self._edit.setMaxLength(self.maxlen + 8)
        self._edit.setToolTip(self.help or "Type 0s and 1s, then press Enter")
        self._edit.editingFinished.connect(lambda: self._typed(on_change))
        lay.addWidget(self._edit)
        if self.help:
            w.setToolTip(self.help)
        self.widget = w
        return w

    def _typed(self, on_change):
        v_ = self._clean(self._edit.text())
        self._edit.setText(v_)
        if v_ != self._v:
            self._v = v_
            on_change(self.key, v_)

    def value(self):
        return self._v

    def set_value(self, v_, notify=False):
        self._v = self._clean(v_)
        if self.widget is not None:
            self._edit.setText(self._v)

    def random_value(self, rng):
        return "".join(rng.choice(["0", "1", "0000"], 12))


# =============================================================================== shared helpers
CODES = {"NRZ (polar)": "nrz", "NRZI": "nrzi", "Unipolar NRZ": "unrz", "Unipolar RZ": "urz",
         "Manchester": "manchester", "AMI": "ami", "B8ZS (T1)": "b8zs", "HDB3 (E1)": "hdb3",
         "MLT-3": "mlt3", "2B1Q / PAM-4": "pam4"}


def encode(bits, code, sps):
    """Waveform for any code in CODES (B8ZS/HDB3 via commlib.serdes)."""
    if code in ("b8zs", "hdb3"):
        s, _ = (sd.b8zs if code == "b8zs" else sd.hdb3)(bits)
        return np.repeat(s.astype(float), sps)
    return lc.encode_line(bits, code, sps)


def violations(bits, code):
    if code in ("b8zs", "hdb3"):
        return (sd.b8zs if code == "b8zs" else sd.hdb3)(bits)[1]
    return np.zeros(len(bits), bool)


def longest_flat(x, sps):
    """Longest stretch (in bit periods) without a level change in a waveform."""
    lev = x[sps // 2::sps]
    if len(lev) < 2:
        return 0
    return int(lc.run_lengths(np.round(lev * 3)).max())


def eye_metrics(y, sym, d, sps, levels, nphase=48):
    """Worst inner eye opening (fraction of the level spacing) vs sampling phase (Lab 3)."""
    M = len(levels)
    spacing = levels[1] - levels[0]
    K = len(sym)
    phases = np.linspace(-0.5, 0.5, nphase, endpoint=False)
    k = np.arange(8, K - 8)
    pos = np.clip(d + k[:, None] * sps + phases[None, :] * sps, 0, len(y) - 2)
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    S = y[i0] * (1 - w) + y[i0 + 1] * w
    idx = np.searchsorted(levels, sym[k] - 1e-9)
    opening = np.full(nphase, np.inf)
    for j in range(M - 1):
        lo, hi = S[idx == j], S[idx == j + 1]
        if len(lo) and len(hi):
            opening = np.minimum(opening, hi.min(axis=0) - lo.max(axis=0))
    opening = opening / spacing
    return float(opening.max()), phases, opening


# =============================================================================== 1. line codes
class LineCodes(Experiment):
    title = "Line codes & spectra"
    blurb = "Type a bit pattern; see it as NRZ, Manchester, AMI, MLT-3… and what each does to the spectrum."
    book = "sec:ch08:linecodes"
    controls = [
        BitField("bits", "Bit pattern (type, then Enter)", "0110100000000000111010",
                 help="Up to 48 bits of 0 and 1"),
        Choice("code", "Line code (for the spectrum)", list(CODES), "AMI", style="menu"),
        Toggle("side", "Compare NRZ, Manchester, AMI, MLT-3", True),
    ]
    plots = [
        Plot("wave", "Your pattern on the wire", x="time (bit periods)", y="", legend=None),
        SpectrumPlot("psd", "Power spectrum for random data (solid) and its spectral lines",
                     x="frequency (multiples of the bit rate R_b)", y="PSD (dB)", xlim=(0, 3),
                     ylim=(-45, 15), legend="tr"),
    ]
    layout = [["wave"], ["psd"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("dc", "Power in the DC line", "%", ".1f"),
        Readout("clk", "Clock line at R_b", "% of power", ".2f"),
        Readout("flat", "Longest flat stretch", "bits", "int"),
        Readout("bw90", "90 % power bandwidth", "× R_b", ".2f"),
    ]
    challenges = [
        Challenge("Type a pattern with at least 12 zeros in a row and keep the longest transition-free "
                  "stretch to 3 bits or less, with a three-level code.",
                  lambda s: "0" * 12 in s.p.bits and s.r.flat <= 3
                  and s.p.code in ("AMI", "B8ZS (T1)", "HDB3 (E1)", "MLT-3"),
                  hint="AMI goes silent on zeros. Which substitution code fills a run every 4 zeros?"),
        Challenge("Find a code with a discrete spectral line at the bit rate: a free clock.",
                  lambda s: s.r.clk > 1,
                  hint="Lines come from a non-zero mean; nulls from correlation."),
        Challenge("Find a code with a spectral null at DC (no power near 0 Hz at all) without "
                  "Manchester's price: 90 % bandwidth under 2 R_b.",
                  lambda s: s.r.null < -20 and s.r.bw90 < 2,
                  hint="A null at DC needs correlation between symbols: alternate something."),
    ]
    sps = 16

    @lru_cache(maxsize=32)
    def spectrum(self, code):
        bits = cl.random_bits(60_000, np.random.default_rng(13))
        x = encode(bits, code, self.sps)
        f, P = welch(x, fs=self.sps, nperseg=4096, return_onesided=True, detrend=False)
        P = P / 2                                     # back to a two-sided density
        n = len(x) // self.sps
        m = x[:n * self.sps].reshape(n, self.sps).mean(axis=0)   # cyclostationary mean
        c = np.fft.fft(m) / self.sps
        ptot = np.mean(x ** 2)
        dc = abs(c[0]) ** 2 / ptot
        clk = 2 * abs(c[1]) ** 2 / ptot
        # 90 % bandwidth from the continuous part plus lines
        cum = np.cumsum(P) * (f[1] - f[0]) * 2
        bw90 = float(f[np.searchsorted(cum / cum[-1], 0.9)])
        Pd = 10 * np.log10(P + 1e-12)
        null = float(Pd[1] - Pd.max())                   # first non-zero bin vs the peak
        return f, Pd, 100 * dc, 100 * clk, bw90, abs(c[0]) ** 2, 2 * abs(c[1]) ** 2, null

    def update(self, p):
        bits = np.array([int(c) for c in p.bits], int)
        nb = len(bits)
        s = 32
        t = np.arange(nb * s) / s
        pw = self.plot("wave")
        rows = [p.code]
        if p.side:
            rows = ["NRZ (polar)", "Manchester", "AMI", "MLT-3"]
            if p.code not in rows:
                rows.append(p.code)
        n = len(rows)
        for i, name in enumerate(rows):
            off = 3.0 * (n - 1 - i)
            x = encode(bits, CODES[name], s)
            tt = t[:len(x)]
            pw.hline(f"z{i}", off, color=GRAY, style=":", width=0.8)
            pw.line(f"w{i}", tt, x + off, color=PALETTE[i % len(PALETTE)], width=2.2)
            pw.text(f"n{i}", -0.2, off + 1.25, name, color=PALETTE[i % len(PALETTE)], anchor=(1, 0.5),
                    size=9, bold=True)
            vi = np.flatnonzero(violations(bits, CODES[name]))
            if len(vi):
                pw.scatter(f"v{i}", vi + 0.5, x[vi * s + s // 2] + off + 0.35 * np.sign(x[vi * s + s // 2]),
                           color=RED, size=9, symbol="t")
                for j, k in enumerate(vi[:12]):
                    pw.text(f"vl{i}_{j}", k + 0.5, x[k * s + s // 2] + off + 0.75 * np.sign(x[k * s + s // 2]),
                            "V", color=RED, anchor=(0.5, 0.5), size=8, bold=True)
        top = 3.0 * (n - 1) + 1.9
        for k in range(nb + 1):
            pw.vline(f"g{k}", k, color=GRAY, style="-", width=0.4)
        for k, b in enumerate(bits[:48]):
            pw.text(f"b{k}", k + 0.5, top, str(b), color=NAVY, anchor=(0.5, 0.5), size=9, bold=True)
        pw.set_xlim(-3.2, max(nb, 8) + 0.2)
        pw.set_ylim(-1.6, top + 0.8)
        pw.set_yticks([])
        f, P, dc, clk, bw90, l0, l1, null = self.spectrum(CODES[p.code])
        ps = self.plot("psd")
        ps.line("P", f, P, color=NAVY, width=1.6, name=p.code)
        lines_f, lines_h = [], []
        for k, val in ((0, l0), (1, l1)):
            if val > 1e-4:
                lines_f.append(k)
                lines_h.append(10 * np.log10(val) + 10)
        if lines_f:
            ps.stems("lines", lines_f, lines_h, color=RED, base=-45, size=10, name="spectral lines (their power, dB)")
        ps.vline("rb", 1.0, color=GRAY, style=":", width=1.0, label="R_b", label_pos=0.06)
        ps.vline("b90", bw90, color=GREEN, style="--", width=1.2, label="90 % of the power", label_pos=0.9)
        x = encode(bits, CODES[p.code], s)
        self.readout(dc=dc, clk=clk, flat=longest_flat(x, s), bw90=bw90, null=null)

    def story(self, p):
        r = self.r
        s = ("<p>The same bits, several ways. Look at a run of zeros: <b>NRZ</b> and <b>AMI</b> go "
             "flat, and a flat line carries no clock; <b>Manchester</b> keeps ticking every bit, at the "
             "price of twice the bandwidth; <b>MLT-3</b> cycles 0, +, 0, − on each one, slowing the "
             "signal down (100BASE-TX).</p>")
        if p.code in ("B8ZS (T1)", "HDB3 (E1)"):
            s += ("<p>Your code substitutes long zero runs with a pattern containing deliberate "
                  "<b>bipolar violations</b> (red V): two pulses of the same polarity in a row, which no "
                  "real AMI data could produce, so the receiver knows to remove them.</p>")
        s += (f"<p>{p.code} for random data puts {v(r.get('dc', 0), '.1f', '%')} of its power in a DC "
              f"line and {v(r.get('clk', 0), '.2f', '%')} in a line at the bit rate. Lines come from a "
              f"non-zero <i>mean</i>; nulls (AMI and Manchester at DC) come from <i>correlation</i> "
              f"between symbols.</p>")
        return "<h3>Bits need a shape</h3>" + s + keybox(
            "S(f) = σ²|P(f)|²/T + (μ²/T²) Σ |P(m/T)|² δ(f − m/T): the first term is the hill, the "
            "second the spectral lines.")


# =============================================================================== 2. wander
class BaselineWander(Experiment):
    title = "Spectral lines & DC wander"
    blurb = "AC coupling blocks DC; unbalanced data drags the baseline across the decision threshold."
    book = "sec:ch08:psd"
    controls = [
        Choice("code", "Line code", ["Polar NRZ", "Manchester", "AMI", "8b/10b + NRZ"]),
        Slider("p1", "Ones in the data", 0.05, 0.95, 0.5, step=0.01, help="Fraction of the data bits that are 1"),
        LogSlider("fc", "AC-coupling corner", 1e-4, 0.1, 0.003, unit="× R_b",
                  help="−3 dB corner of the coupling capacitor's high-pass, relative to the bit rate"),
        Slider("noise", "Receiver noise rms", 0.0, 0.3, 0.08, step=0.01, help="Relative to half the signal swing"),
        Button("again", "New data"),
    ]
    plots = [
        Plot("wave", "The received waveform after AC coupling", x="time (bit periods)", y="amplitude",
             xlim=(0, 300), ylim=(-2.2, 2.6), legend="tl", legend_cols=3),
        SpectrumPlot("psd", "Spectrum and the coupling high-pass", x="frequency (× R_b)", y="dB",
                     xlim=(0, 1.5), ylim=(-45, 12), legend="tr"),
        Plot("rds", "Running digital sum (the DC the capacitor must hold)", x="bit", y="ones − zeros",
             legend=None),
    ]
    layout = [["wave", "wave"], ["psd", "rds"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("dcl", "DC line", "% of power", ".1f"),
        Readout("shift", "Worst baseline shift", "% of half-swing", ".0f", good=lambda x: x < 30),
        Readout("ber", "Bit errors", "per 1000", ".1f", good=lambda x: x == 0),
        Readout("rate", "Line bits per data bit", "", ".2f"),
    ]
    challenges = [
        Challenge("Put exactly 25 % of polar NRZ's power into the DC line.",
                  lambda s: s.p.code == "Polar NRZ" and abs(s.r.dcl - 25) < 1.5,
                  hint="The DC line holds μ² of the power, and μ = 2p − 1 for polar NRZ."),
        Challenge("With 90 % ones and a corner of 0.01 R_b, let polar NRZ's wander cause more than "
                  "10 errors per 1000.",
                  lambda s: s.p.code == "Polar NRZ" and s.p.p1 >= 0.895 and s.p.fc >= 0.0095
                  and s.r.ber > 10),
        Challenge("Same punishing data (≥ 90 % ones, corner ≥ 0.01 R_b): find a code with no errors.",
                  lambda s: s.p.code != "Polar NRZ" and s.p.p1 >= 0.895 and s.p.fc >= 0.0095
                  and s.r.ber == 0 and s.p.noise >= 0.05),
    ]
    sps = 8
    nbits = 3000

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def make(self, p):
        rng = np.random.default_rng(self.seed)
        data = (rng.random(self.nbits) < p.p1).astype(int)
        if p.code == "8b/10b + NRZ":
            byts = np.packbits(data[: self.nbits // 8 * 8])
            line = lc.enc8b10b(byts)
            x = lc.encode_line(line, "nrz", self.sps)
            rate = 1.25
            sym = 2.0 * line - 1
        elif p.code == "Manchester":
            x = lc.encode_line(data, "manchester", self.sps)
            line, rate, sym = data, 1.0, None
        elif p.code == "AMI":
            x = lc.encode_line(data, "ami", self.sps)
            line, rate, sym = data, 1.0, None
        else:
            x = lc.encode_line(data, "nrz", self.sps)
            line, rate = data, 1.0
            sym = 2.0 * data - 1
        a = np.exp(-2 * np.pi * p.fc / self.sps)
        y = lfilter([(1 + a) / 2, -(1 + a) / 2], [1, -a], x)        # first-order high-pass
        y = y + p.noise * rng.standard_normal(len(y))
        return data, line, x, y, rate

    def decide(self, p, line, y):
        s = self.sps
        if p.code == "Manchester":
            first = y[s // 4::s][:len(line)]
            second = y[3 * s // 4::s][:len(line)]
            return (second - first > 0).astype(int)
        if p.code == "AMI":
            return (np.abs(y[s // 2::s][:len(line)]) > 0.5).astype(int)
        return (y[s // 2::s][:len(line)] > 0).astype(int)

    def update(self, p):
        data, line, x, y, rate = self.make(p)
        dec = self.decide(p, line, y)
        errs = int(np.sum(dec[50:] != np.asarray(line)[50:len(dec)]))
        ber = 1000 * errs / max(len(dec) - 50, 1)
        # baseline: low-pass of the waveform's departure (difference between x and AC-coupled)
        s = self.sps
        a = np.exp(-2 * np.pi * p.fc / s)
        xc = lfilter([(1 + a) / 2, -(1 + a) / 2], [1, -a], x)
        wander = x - xc
        k = slice(200 * s, 500 * s)
        t = np.arange(len(x)) / s
        pw = self.plot("wave")
        pw.line("y", t[k] - 200, y[k], color=GRAY, width=1.0, name="received")
        pw.line("bl", t[k] - 200, -wander[k], color=ORANGE, width=2.4, name="baseline")
        pw.hline("th", 0, color=RED, style="--", width=1.2)
        bad_idx = np.flatnonzero(dec != np.asarray(line)[:len(dec)])
        bad_idx = bad_idx[(bad_idx >= 200) & (bad_idx < 500)]
        if p.code != "8b/10b + NRZ":
            pw.scatter("err", bad_idx + 0.5 - 200, y[bad_idx * s + s // 2], color=RED, size=9, symbol="x",
                       name="bit errors")
        mu = np.mean(x)
        ptot = np.mean(x ** 2)
        dcl = 100 * mu ** 2 / ptot
        f, P = welch(x, fs=s, nperseg=2048, detrend=False)
        ps = self.plot("psd")
        ps.line("P", f, 10 * np.log10(P / 2 + 1e-12), color=NAVY, width=1.6, name="line signal")
        if dcl > 0.05:
            ps.stems("dc", [0.004], [10 * np.log10(mu ** 2) + 10], color=RED, base=-45, size=10,
                     name="DC line")
        ff = np.linspace(1e-4, 1.5, 600)
        H = 20 * np.log10(ff / np.sqrt(ff ** 2 + p.fc ** 2))
        ps.line("hp", ff, H, color=ORANGE, width=1.6, style="--", name="coupling high-pass")
        rds = np.cumsum(2.0 * np.asarray(line) - 1)
        pr = self.plot("rds")
        pr.line("r", np.arange(len(rds)), rds, color=NAVY, width=1.6)
        pr.hline("z", 0, color=GRAY, style="-", width=0.8)
        shift = 100 * float(np.max(np.abs(wander[50 * s:])))
        self.readout(dcl=dcl, shift=shift, ber=ber, rate=rate)

    def story(self, p):
        r = self.r
        s = (f"<p>A coupling capacitor (or a transformer) passes everything except DC. Your data is "
             f"{v(100 * p.p1, '.0f', '%')} ones. "
             + ("Even balanced data has long stretches that are locally unbalanced, " if abs(p.p1 - 0.5) < 0.05
                else "Its average is not zero, ")
             + f"so the capacitor charges up and the signal drifts (orange) relative to the decision "
             f"threshold (red dashes). That is <b>baseline wander</b>, here up to "
             f"{v(r.get('shift', 0), '.0f', '%')} of the half-swing.</p>")
        if p.code == "Polar NRZ":
            s += (f"<p>In the spectrum the imbalance is a <b>DC line</b> holding "
                  f"{v(r.get('dcl', 0), '.1f', '%')} of the power, exactly where the high-pass (dashed) "
                  f"has nothing to give. The running digital sum (bottom right) is a random walk "
                  f"with no bound.</p>")
        else:
            s += (f"<p>{p.code} is <b>DC-balanced by construction</b>: its running digital sum stays "
                  "bounded whatever the data, the spectrum has a null at DC, and the coupling "
                  "capacitor has nothing to charge.</p>")
        return "<h3>Why line codes exist</h3>" + s + keybox(
            "A line code's first job is to keep the running digital sum bounded, so the spectrum "
            "has nothing at DC for the coupling to remove.")


# =============================================================================== 3. scramblers
SCR = ["None", "Additive PRBS-7", "Additive PRBS-15", "Self-synchronous 64b/66b"]
DATA = ["Idle: all zeros", "Repeating 1100", "Random", "The scrambler's own PRBS-7"]


class Scramblers(Experiment):
    title = "Scramblers"
    blurb = "Turn idle patterns into noise-like bits; flip one line bit and count the damage."
    book = "sec:ch08:scramblers"
    controls = [
        Choice("data", "Data", DATA, style="menu"),
        Choice("scr", "Scrambler", SCR, "None", style="menu"),
        Toggle("flip", "Flip one bit on the line (at bit 1000)", False),
    ]
    plots = [
        ImagePlot("raster", "The line bits, 64 per row (white = 1)", x="bit within row", y="row"),
        Plot("runs", "Run lengths on the line", x="run length (bits)", y="number of runs (log)",
             xlim=(0.4, 16.6), legend="tr"),
        SpectrumPlot("psd", "Line spectrum (NRZ)", x="frequency (× R_b)", y="dB", xlim=(0, 0.5),
                     ylim=(-60, 25), legend=None),
        Plot("err", "Errors after the descrambler", x="bit index", y="error",
             xlim=(990, 1070), ylim=(0, 1.35), legend=None),
    ]
    layout = [["raster", "runs"], ["psd", "err"]]
    readouts = [
        Readout("run", "Longest run", "bits", "int", good=lambda x: x < 30),
        Readout("errs", "Errors from one flipped bit", "", "int"),
        Readout("flat", "Spectral peak above average", "dB", ".1f", good=lambda x: x < 6),
        Readout("period", "Scrambler sequence period", "", None),
    ]
    challenges = [
        Challenge("Flip one line bit with the self-synchronous scrambler and count the output errors.",
                  lambda s: s.p.flip and s.p.scr.startswith("Self") and s.r.errs == 3,
                  hint="Each received bit is used three times: once as data, and as each of the two taps."),
        Challenge("A scrambler is not a line code: make the additive scrambler's output a run longer "
                  "than 100 bits.",
                  lambda s: s.p.scr.startswith("Additive") and s.r.run > 100),
        Challenge("Make the idle (all-zeros) line look like noise: longest run under 30 and a spectrum "
                  "within 6 dB of flat.",
                  lambda s: s.p.data.startswith("Idle") and s.r.run < 30 and s.r.flat < 6),
    ]
    n = 4096

    @lru_cache(maxsize=32)
    def line(self, data, scr):
        rng = np.random.default_rng(7)
        if data.startswith("Idle"):
            d = np.zeros(self.n, np.int8)
        elif data.startswith("Repeating"):
            d = np.tile(np.array([1, 1, 0, 0], np.int8), self.n // 4)
        elif data.startswith("Random"):
            d = cl.random_bits(self.n, rng).astype(np.int8)
        else:
            d = lc.prbs(7, self.n)
        if scr == "None":
            s = d.copy()
        elif scr == "Additive PRBS-7":
            s = lc.scramble_add(d, 7)
        elif scr == "Additive PRBS-15":
            s = lc.scramble_add(d, 15)
        else:
            s = lc.scramble_ss(d)
        return d, s

    def descramble(self, scr, rx):
        if scr == "None":
            return rx
        if scr.startswith("Additive"):
            return lc.scramble_add(rx, 7 if "7" in scr else 15)
        return lc.descramble_ss(rx)

    def update(self, p):
        d, s = self.line(p.data, p.scr)
        rx = s.copy()
        if p.flip:
            rx[1000] ^= 1
        out = self.descramble(p.scr, rx)
        err = np.flatnonzero(out != d)
        rl = lc.run_lengths(s)
        img = s[:64 * 48].reshape(48, 64).astype(float)
        pr = self.plot("raster")
        pr.image("r", img[::-1], x=(0, 64), y=(0, 48), cmap=("#1b2a4a", "#f4f1e8"), levels=(0, 1))
        pr.set_xlim(0, 64)
        pr.set_ylim(0, 48)
        h = np.bincount(np.minimum(rl, 16), minlength=17)[1:17]
        pu = self.plot("runs")
        k = np.arange(1, 17)
        geo = len(rl) * 0.5 ** k
        pu.bars("h", k, np.log10(np.maximum(h, 10 ** -0.5)), width=0.6, color=NAVY, base=-0.5)
        pu.line("g", k, np.log10(geo), color=GREEN, width=2.0, style="--", name="random bits: halves per bit")
        pu.set_ylim(-0.5, 4)
        pu.set_yticks([(q, f"{10 ** q:g}") for q in range(0, 4)])
        if rl.max() > 16:
            pu.text("long", 16.2, 2.6, f"longest run: {rl.max()} →", color=RED, anchor=(1, 0.5), size=9,
                    bold=True)
        f, P = welch(2.0 * s - 1, fs=1.0, nperseg=512, detrend=False)
        Pd = 10 * np.log10(P + 1e-9)
        ps = self.plot("psd")
        ps.line("P", f, Pd, color=NAVY, width=1.4, fill=-60, fill_alpha=0.08)
        flat = float(Pd.max() - 10 * np.log10(np.mean(P) + 1e-12))
        pe = self.plot("err")
        if p.flip:
            pe.vline("flip", 1000, color=ORANGE, style="--", width=1.2, label="line error", label_pos=0.92)
            if len(err):
                pe.stems("e", err, np.ones(len(err)), color=RED, size=9)
                for i, e in enumerate(err[:6]):
                    pe.text(f"t{i}", e, 1.08, str(int(e)), color=RED, anchor=(0.5, 1), size=9)
        else:
            pe.text("hint", 1030, 0.65, "switch on ‘Flip one bit’", color=GRAY, anchor=(0.5, 0.5), size=10)
        period = {"None": "—", "Additive PRBS-7": "127 bits", "Additive PRBS-15": "32 767 bits",
                  SCR[3]: "no fixed period"}[p.scr]
        self.readout(run=int(rl.max()), errs=len(err) if p.flip else 0, flat=flat, period=period)

    def story(self, p):
        r = self.r
        s = ("<p>A <b>linear-feedback shift register</b> with a primitive polynomial cycles through every "
             "non-zero state: a pseudo-random binary sequence (PRBS) whose autocorrelation is nearly a "
             "spike. XOR it onto the data and long runs and repeating patterns become coin flips: "
             "transitions for the clock, a flat spectrum for EMI.</p>")
        if p.scr.startswith("Self"):
            s += ("<p>The <b>self-synchronous</b> scrambler of 64b/66b Ethernet, s = d ⊕ s₋₃₉ ⊕ s₋₅₈, "
                  "feeds back its own output, so the descrambler needs no frame alignment. The price: "
                  "each received bit is used three times, so one line error becomes three output errors, "
                  "39 and 58 bits apart.</p>")
        elif p.scr.startswith("Additive"):
            s += ("<p>The <b>additive</b> scrambler XORs a free-running PRBS: errors do not multiply, "
                  "but transmitter and receiver must start the PRBS in step.</p>")
            if p.data.startswith("The scrambler"):
                s += ("<p>" + bad("Defeated.") + " Data equal to the scrambler's own sequence XORs to all "
                      "zeros: a malicious (or unlucky) payload can still starve the clock. A scrambler "
                      "makes long runs unlikely, not impossible.</p>")
        else:
            s += f"<p>No scrambler: the line is the data, with runs up to {v(r.get('run', 0), 'd', 'bits')}.</p>"
        return "<h3>Making bits look random</h3>" + s + keybox(
            "Scramblers randomise statistically; only a line code (8b/10b, sync headers) "
            "<i>guarantees</i> transitions.")


# =============================================================================== 4. 8b/10b
DATA8 = ["All-zeros idle (0x00)", "0x0F repeated", "0xFF repeated", "Random bytes",
         "Zeros, then 0x0F, then random"]


class Code8b10b(Experiment):
    title = "8b/10b running disparity"
    blurb = "A byte becomes ten bits chosen to keep ones and zeros in balance, whatever the data."
    book = "sec:ch08:linecodes"
    controls = [
        Choice("data", "Data", DATA8, DATA8[4], style="menu"),
        Choice("coding", "Coding", ["Raw bytes", "8b/10b", "64b/66b (scrambled)"], "8b/10b"),
        IntSlider("byte", "Inspect byte", 0, 255, 0),
    ]
    plots = [
        Plot("rds", "Running digital sum: ones minus zeros so far", x="data byte", y="RDS",
             legend=None),
        Plot("runs", "Run lengths", x="run length (bits)", y="number of runs (log)", xlim=(0.4, 12.6),
             legend=None),
        Plot("code", "The code groups of one byte", x="", y="", xlim=(0, 10), ylim=(-0.6, 3.2),
             legend=None, grid=False),
    ]
    layout = [["rds", "rds"], ["runs", "code"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("maxrds", "Largest |RDS|", "", "int", good=lambda x: x <= 10),
        Readout("run", "Longest run", "bits", "int"),
        Readout("ovh", "Overhead", "%", ".1f"),
        Readout("rate", "Line rate for 10 Gb/s of data", "Gb/s", ".3f"),
    ]
    challenges = [
        Challenge("Feed the all-zeros idle pattern through 8b/10b and watch the RDS stay within ±3.",
                  lambda s: s.p.data.startswith("All-zeros") and s.p.coding == "8b/10b" and s.r.maxrds <= 3),
        Challenge("Send raw bytes and make the running digital sum wander beyond 1000.",
                  lambda s: s.p.coding == "Raw bytes" and s.r.maxrds > 1000),
        Challenge("Get the overhead below 5 % while keeping every run shorter than 70 bits.",
                  lambda s: s.r.ovh < 5 and s.r.run < 70),
        Challenge("Find a byte whose 8b/10b code group is balanced (five ones), so both disparities "
                  "send the same ten bits.",
                  lambda s: s.r.balanced is True and s.p.byte != 0),
    ]
    nbytes = 600

    @lru_cache(maxsize=16)
    def stream(self, data, coding):
        rng = np.random.default_rng(10)
        n = self.nbytes
        if data.startswith("All-zeros"):
            B = np.zeros(n, int)
        elif data.startswith("0x0F"):
            B = np.full(n, 0x0F)
        elif data.startswith("0xFF"):
            B = np.full(n, 0xFF)
        elif data.startswith("Random"):
            B = rng.integers(0, 256, n)
        else:
            B = np.r_[np.zeros(n // 3, int), np.full(n // 3, 0x0F), rng.integers(0, 256, n - 2 * (n // 3))]
        raw = ((B[:, None] >> np.arange(7, -1, -1)) & 1).ravel()
        if coding == "Raw bytes":
            line, per = raw, 8
        elif coding == "8b/10b":
            line, per = lc.enc8b10b(B), 10
        else:
            line, per = sd.enc64b66b(raw), 66 / 8
        return B, raw, np.asarray(line, int), per

    def update(self, p):
        B, raw, line, per = self.stream(p.data, p.coding)
        rds = np.cumsum(2 * line - 1)
        xb = np.arange(len(line)) / per
        pr = self.plot("rds")
        pr.hline("z", 0, color=GRAY, style="-", width=0.8)
        pr.line("r", xb, rds, color=NAVY if p.coding != "Raw bytes" else RED, width=1.8)
        third = self.nbytes // 3
        if p.data.startswith("Zeros"):
            pr.vline("a", third, color=GRAY, style=":", width=1.0, label="0x0F from here", label_pos=0.92)
            pr.vline("b", 2 * third, color=GRAY, style=":", width=1.0, label="random from here", label_pos=0.82)
        rl = lc.run_lengths(line)
        h = np.bincount(np.minimum(rl, 12), minlength=13)[1:13]
        pu = self.plot("runs")
        pu.bars("h", np.arange(1, 13), np.log10(np.maximum(h, 10 ** -0.5)), width=0.6,
                colors=[NAVY] * 11 + [RED], base=-0.5)
        pu.set_ylim(-0.5, 4)
        pu.set_yticks([(q, f"{10 ** q:g}") for q in range(0, 4)])
        pu.set_xticks([(k, str(k) if k < 12 else "12+") for k in range(1, 13)])
        # code-group inspector
        neg = lc.enc8b10b([p.byte], rd=-1)
        pos = lc.enc8b10b([p.byte], rd=+1)
        pc = self.plot("code")
        x, y = p.byte & 31, p.byte >> 5
        pc.text("ttl", 5, 3.0, f"byte 0x{p.byte:02X} = D.{x}.{y}", color=NAVY, anchor=(0.5, 0.5), size=11,
                bold=True)
        for row, (bits, lab) in enumerate([(neg, "RD−"), (pos, "RD+")]):
            yy = 1.8 - 1.3 * row
            pc.text(f"l{row}", -0.1, yy, lab, color=GRAY, anchor=(1, 0.5), size=10, bold=True)
            for i, b in enumerate(bits):
                pc.text(f"b{row}_{i}", i + 0.5, yy, str(int(b)), color=GREEN if b else ORANGE,
                        anchor=(0.5, 0.5), size=15, bold=True)
            d = int(2 * np.sum(bits) - 10)
            pc.text(f"d{row}", 10.1, yy, f"{d:+d}", color=GRAY, anchor=(0, 0.5), size=10)
        pc.set_xlim(-1.2, 11.2)
        pc.set_xticks([])
        pc.set_yticks([])
        bal = bool(np.sum(neg) == 5 and np.array_equal(neg, pos))
        ovh = 100 * (per / 8 - 1)
        self.readout(maxrds=int(np.max(np.abs(rds))), run=int(rl.max()), ovh=ovh, rate=10 * per / 8,
                     balanced=bal)

    def story(self, p):
        r = self.r
        if p.coding == "8b/10b":
            s = ("<p><b>8b/10b</b> (Widmer and Franaszek, IBM 1983) maps each byte to a 10-bit code group "
                 "with at most five identical bits in a row. Unbalanced groups come in two versions "
                 "(bottom right); the encoder picks whichever pulls the <b>running disparity</b> back "
                 f"towards zero. Result: the RDS never strays beyond {v(r.get('maxrds', 0), 'd')}, "
                 "whatever the data.</p>")
        elif p.coding == "Raw bytes":
            s = (f"<p>Raw bytes put the data straight on the wire: an idle stream of zeros drives the RDS "
                 f"to {v(r.get('maxrds', 0), 'd')}, a capacitor's worth of DC, with runs of "
                 f"{v(r.get('run', 0), 'd', 'bits')} and no transitions to clock on.</p>")
        else:
            s = ("<p><b>64b/66b</b> scrambles each 64-bit block and adds a two-bit sync header (01 or "
                 "10), which guarantees a transition every 66 bits and frames the blocks. Balance is "
                 "statistical, not guaranteed, but the overhead falls from 25 % to 3.1 %.</p>")
        if p.coding != "Raw bytes":
            s += (f"<p>Overhead {v(r.get('ovh', 0), '.1f', '%')}: 10 Gb/s of data needs a "
                  f"{v(r.get('rate', 0), '.3f', 'Gb/s')} line"
                  + (", which is why 10G Ethernet signals at 10.3125 Gb/s." if p.coding.startswith("64")
                     else ", which is why Gigabit Ethernet's 1000BASE-X runs at 1.25 GBd.") + "</p>")
        return "<h3>Balance by construction</h3>" + s + keybox(
            "8b/10b guarantees run length ≤ 5 and a bounded RDS at 25 % overhead; 64b/66b and 128b/130b "
            "trade the guarantee for 3 % and 1.5 %.")


# =============================================================================== 5. jitter
RATES = {"10.3125 Gb/s": 10.3125, "25.78125 Gb/s": 25.78125, "53.125 Gb/s": 53.125}
BERS = {"10⁻¹²": 1e-12, "10⁻¹⁵": 1e-15}


class EyesBathtub(Experiment):
    title = "Eyes & bathtub"
    blurb = "Random and deterministic jitter close the eye differently: read total jitter off the bathtub."
    book = "sec:ch08:jitter"
    animate = True
    autoplay = True
    fps = 10
    controls = [
        Heading("Jitter"),
        Slider("rj", "Random jitter (rms)", 0.002, 0.06, 0.03, step=0.001, unit="UI"),
        Slider("dj", "Deterministic (pk-pk)", 0.0, 0.4, 0.10, step=0.01, unit="UI"),
        Choice("djtype", "Kind of DJ", ["Dual-Dirac", "Periodic (sinusoid)", "Data-dependent (ISI)"],
               style="menu"),
        Heading("Link"),
        Choice("rate", "Bit rate", list(RATES), "25.78125 Gb/s", style="menu"),
        Choice("target", "Target BER", list(BERS), "10⁻¹²"),
        Toggle("persist", "Persistence", True),
    ]
    plots = [
        EyePlot("eye", "NRZ eye (persistence)", yrange=(-1.5, 1.5)),
        Plot("hist", "Where the edges land (time-interval error)", x="edge position (UI)", y="density",
             xlim=(-0.4, 0.4), legend="tr"),
        Plot("tub", "Bathtub: BER against sampling phase", x="sampling phase (UI)", y="BER",
             logy=True, xlim=(0, 1), ylim=(1e-16, 1), legend="tl"),
    ]
    layout = [["eye", "tub"], ["hist", "tub"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("tj", "Total jitter at target BER", "UI", ".3f"),
        Readout("width", "Eye width at target BER", "UI", ".3f", good=lambda x: x >= 0.5),
        Readout("ps", "Eye width", "ps", ".1f"),
        Readout("rjshare", "Share of TJ due to RJ", "%", ".0f"),
    ]
    challenges = [
        Challenge("Reproduce the book's 25G budget: DJ 0.15 UI and RJ 0.018 UI. Read the eye width at "
                  "10⁻¹² (about 0.60 UI).",
                  lambda s: abs(s.p.dj - 0.15) < 0.006 and abs(s.p.rj - 0.018) < 0.0006
                  and s.p.target == "10⁻¹²"),
        Challenge("From the book's budget, halve RJ instead of DJ and open the 10⁻¹² eye to 0.72 UI.",
                  lambda s: abs(s.p.dj - 0.15) < 0.006 and s.p.rj <= 0.0095 and s.r.width >= 0.72
                  and s.p.target == "10⁻¹²"),
        Challenge("Keep the 10⁻¹⁵ eye at least 0.5 UI wide with DJ of 0.25 UI or more.",
                  lambda s: s.p.target == "10⁻¹⁵" and s.p.dj >= 0.245 and s.r.width >= 0.5),
    ]
    spu = 32
    nb = 500

    def setup(self):
        self.mc = []

    def edges(self, p, b, rng):
        n = len(b)
        e = p.rj * rng.standard_normal(n)
        if p.djtype.startswith("Dual"):
            e = e + p.dj / 2 * rng.choice([-1, 1], n)
        elif p.djtype.startswith("Periodic"):
            e = e + p.dj / 2 * np.sin(2 * np.pi * 0.0137 * np.arange(n) + rng.uniform(0, 6.28))
        else:   # data-dependent: an edge after a run of identical bits arrives late
            prev_same = np.r_[False, False, (b[1:-1] == b[:-2])][:n]
            e = e + np.where(prev_same, p.dj / 2, -p.dj / 2)
        return e

    def make_frame(self, p):
        rng = self.rng
        b = rng.integers(0, 2, self.nb)
        e = self.edges(p, b, rng)
        trans = np.r_[False, b[1:] != b[:-1]]
        L = 2.0 * b - 1
        T = np.arange(self.nb) + e                      # edge k sits between bit k-1 and bit k
        tt = np.arange(1, (self.nb - 1) * self.spu) / self.spu
        j = np.clip(np.floor(tt).astype(int), 1, self.nb - 2)
        S = lambda u: 0.5 * (1 + np.tanh(u / 0.035))    # smooth edge, rise time about 0.07 UI
        wave = (L[j - 1] + (L[j] - L[j - 1]) * S(tt - T[j])
                + (L[j + 1] - L[j]) * S(tt - T[j + 1]))
        wave = np.r_[np.full(self.spu, L[0]), wave]
        return wave, e[trans]

    def draw(self, p, accumulate):
        wave, tie = self.make_frame(p)
        pe = self.plot("eye")
        # traces centred so the crossings sit at 0.5 and 1.5 UI
        pe.eye("eye", wave, self.spu, n_sym=2, offset=20 * self.spu - self.spu // 2,
               yrange=(-1.5, 1.5), accumulate=accumulate and p.persist, decay=0.9)
        ber = BERS[p.target]
        tj = sd.total_jitter(ber, p.rj, p.dj)
        width = max(0.0, 1 - tj)
        ph = self.plot("hist")
        hcount, edges = np.histogram(tie, bins=80, range=(-0.4, 0.4), density=True)
        ph.line("h", edges, hcount, color=NAVY, width=1.2, step=True, fill=0, fill_alpha=0.2,
                name="this frame")
        xx = np.linspace(-0.4, 0.4, 400)
        g = lambda m: np.exp(-0.5 * ((xx - m) / p.rj) ** 2) / (p.rj * np.sqrt(2 * np.pi))
        ph.line("m", xx, 0.5 * (g(-p.dj / 2) + g(p.dj / 2)), color=RED, width=1.8, name="dual-Dirac model")
        ph.set_ylim(0, max(hcount.max(), 1) * 1.2)
        pt = self.plot("tub")
        x = np.linspace(0.0, 1.0, 801)
        bt = sd.dual_dirac_ber(x, p.rj, p.dj)
        pt.line("th", x, np.maximum(bt, 1e-18), color=RED, width=2.2, name="dual-Dirac model")
        pt.hline("t", ber, color=GRAY, style=":", width=1.2, label=f"target {p.target}", label_pos=0.02)
        if width > 0:
            pt.band("w", tj / 2, 1 - tj / 2, color=GREEN, alpha=0.10)
            pt.text("wl", 0.5, ber * 30, f"eye {width:.2f} UI", color=GREEN, anchor=(0.5, 1), size=10,
                    bold=True)
        if self.mc:
            xs, ys = zip(*self.mc)
            pt.scatter("mc", xs, ys, color=NAVY, size=7, outline=NAVY, name="simulated")
        ui_ps = 1e3 / RATES[p.rate]
        rjpart = 2 * float(sd.qinv(ber / 0.5)) * p.rj
        self.readout(tj=tj, width=width, ps=width * ui_ps, rjshare=100 * rjpart / max(tj, 1e-9))

    def update(self, p):
        self.mc = []
        self.plot("eye").reset_persistence()
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def background(self, p):
        rng = np.random.default_rng(11)
        n = 40_000 if self.quick else 1_000_000
        b = rng.integers(0, 2, n)
        e = self.edges(p, b, rng)
        trans = np.r_[False, b[1:] != b[:-1]]
        nxt_t = np.r_[trans[1:], False]
        nxt_e = np.r_[e[1:], 0.0]
        for x in np.linspace(0.04, 0.96, 24 if not self.quick else 8):
            errs = (trans & (e > x)) | (nxt_t & (nxt_e + 1 < x))
            r_ = errs.mean()
            if r_ > 0:
                yield (float(x), float(r_))

    def progress(self, p, item):
        self.mc.append(item)
        xs, ys = zip(*self.mc)
        self.plot("tub").scatter("mc", xs, ys, color=NAVY, size=7, outline=NAVY, name="simulated")

    def story(self, p):
        r = self.r
        s = (f"<p>Every edge lands a little off the grid. <b>Random jitter</b> (thermal and phase noise) "
             f"is Gaussian and unbounded; <b>deterministic jitter</b> ({p.djtype.lower()}) is bounded. "
             f"The histogram (bottom left) is the two convolved: two Gaussian humps "
             f"{v(p.dj, '.2f', 'UI')} apart.</p>"
             f"<p>At {p.target} you need the eye open where only one edge in "
             f"{'a trillion' if p.target == '10⁻¹²' else 'a quadrillion'} strays. A bounded component "
             f"costs its peak-to-peak; a Gaussian one about 14σ (16σ at 10⁻¹⁵). So TJ = DJ + "
             f"2·Q⁻¹(BER/ρ)·RJ = {v(r.get('tj', 0), '.3f', 'UI')}, leaving "
             f"{v(r.get('width', 0), '.3f', 'UI')} = {v(r.get('ps', 0), '.1f', 'ps')} of eye; RJ is "
             f"{v(r.get('rjshare', 0), '.0f', '%')} of the total.</p>"
             "<p>The open circles are a real Monte Carlo count: a bit-error-rate tester sees the walls "
             "down to 10⁻⁶ or so and extrapolates the rest with exactly this model.</p>")
        return "<h3>Noise measured in seconds</h3>" + s + keybox(
            "TJ(BER) = DJ + 2·Q⁻¹(BER/ρ)·RJ. At low BER, random jitter is the expensive kind.")


# =============================================================================== 6. PAM-4 channel
class PAM4Channel(Experiment):
    title = "PAM-4 over a lossy channel"
    blurb = "A backplane eats the high frequencies: equalise NRZ and PAM-4 with CTLE, FFE and DFE."
    book = "sec:ch08:channel"
    animate = True
    fps = 8
    controls = [
        Heading("Channel"),
        Slider("loss", "Loss at R_b/2", 5, 40, 25, step=0.5, unit="dB",
               help="Insertion loss at the NRZ Nyquist frequency"),
        Slider("snr", "Peak signal / noise", 20, 50, 34, step=0.5, unit="dB"),
        Heading("Equalisers"),
        Slider("ctle", "CTLE peaking", 0, 20, 0, step=0.5, unit="dB",
               help="Analog boost at each scheme's Nyquist frequency. Boosts the noise too"),
        IntSlider("ffe", "FFE taps (0 = off)", 0, 15, 0),
        IntSlider("dfe", "DFE taps", 0, 4, 0, help="Cancel post-cursor ISI using past decisions"),
        Toggle("persist", "Persistence", True),
    ]
    plots = [
        EyePlot("nrz", "NRZ, 1 bit per symbol", yrange=(-1.6, 1.6)),
        EyePlot("pam", "PAM-4, 2 bits per symbol", yrange=(-1.6, 1.6)),
        SpectrumPlot("resp", "Channel, CTLE and their product", x="frequency (× bit rate)", y="gain (dB)",
                     xlim=(0, 0.75), ylim=(-50, 25), legend="tr"),
        Plot("pulse", "PAM-4 pulse response at the symbol instants", x="symbols from the main cursor",
             y="amplitude", xlim=(-2.5, 8.5), ylim=(-0.4, 1.15), legend="tr"),
    ]
    layout = [["nrz", "pam"], ["resp", "pulse"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("nrz", "NRZ eye height", "% of swing", ".0f", good=lambda x: x > 0),
        Readout("pam", "PAM-4 eye height", "% of swing", ".0f", good=lambda x: x > 0),
        Readout("lq", "Loss at R_b/4 (PAM-4 Nyquist)", "dB", ".1f"),
        Readout("win", "Wider eye", "", None),
    ]
    challenges = [
        Challenge("At 35 dB of loss, find an equaliser setting where the PAM-4 eye opens and the NRZ "
                  "eye stays shut.",
                  lambda s: s.p.loss >= 35 and s.r.pam > 0 and s.r.nrz <= 0),
        Challenge("Open the NRZ eye at 20 dB of loss with the CTLE alone (no FFE, no DFE).",
                  lambda s: s.p.loss >= 20 and s.p.ffe == 0 and s.p.dfe == 0 and s.r.nrz > 0
                  and s.p.ctle > 0),
        Challenge("At 30 dB, open both eyes by at least 10 % of the swing with no more than 3 FFE "
                  "taps.",
                  lambda s: s.p.loss >= 30 and s.p.ffe <= 3 and s.r.nrz >= 10 and s.r.pam >= 10,
                  hint="The FFE alone runs out of taps; something else can cancel post-cursors without "
                       "boosting the noise."),
    ]
    sb = 16                 # samples per NRZ bit
    nbits = 1000

    def chan(self, p, sps):
        """Channel × CTLE impulse response at sb samples per bit, the CTLE tuned to the scheme's
        Nyquist frequency (sps = samples per symbol)."""
        key = (round(p.loss, 2), round(p.ctle, 2), sps)
        cache = getattr(self, "_cc", {})
        if key in cache:
            return cache[key]
        h = lc.backplane(self.sb, p.loss, 0.5, nfft=1 << 13, length=48 * self.sb, skin_frac=0.15)
        nf = 1 << 13
        f = np.fft.rfftfreq(nf, 1 / self.sb)
        fny = self.sb / sps / 2
        C = sd.ctle_response(f, p.ctle, fny)
        hc = np.fft.irfft(np.fft.rfft(h, nf) * C, nf)[:64 * self.sb]
        cache[key] = (h, hc, C)
        self._cc = cache
        return h, hc, C

    def link(self, p, M):
        sps = self.sb * (1 if M == 2 else 2)
        nsym = self.nbits // (1 if M == 2 else 2)
        lev = np.linspace(-1, 1, M)
        a = lev[self.rng.integers(0, M, nsym)]
        h, hc, C = self.chan(p, sps)
        # noise is added at the receiver input, before the CTLE (which amplifies it)
        x = fftconvolve(np.repeat(a, sps), h)[:nsym * sps]
        x = x + 10 ** (-p.snr / 20) * self.rng.standard_normal(len(x))
        nf = 1 << int(np.ceil(np.log2(len(x) + 64)))
        Cfull = sd.ctle_response(np.fft.rfftfreq(nf, 1 / self.sb), p.ctle, self.sb / sps / 2)
        y = np.fft.irfft(np.fft.rfft(x, nf) * Cfull, nf)[:len(x)]
        pr = np.convolve(hc, np.ones(sps))
        if p.ffe > 0:
            w, _, _ = lc.ffe_zf(pr, sps, ntaps=int(p.ffe), pre=min(1, int(p.ffe) - 1))
            pre = min(1, int(p.ffe) - 1)
            y, pr = self.fir(y, w, sps, pre), self.fir(pr, w, sps, pre)
        d = int(np.argmax(pr))
        g = pr[d]
        y, pr = y / g, pr / g
        if p.dfe > 0:
            taps = np.array([pr[d + k * sps] if d + k * sps < len(pr) else 0.0 for k in range(1, int(p.dfe) + 1)])
            n_idx = np.floor((np.arange(len(y)) - d + sps / 2) / sps).astype(int)
            corr = np.zeros(len(y))
            for k, bk in enumerate(taps, start=1):
                j = n_idx - k
                ok = (j >= 0) & (j < nsym)
                corr[ok] += bk * a[j[ok]]
            y = y - corr
            pr = pr.copy()
            for k in range(1, int(p.dfe) + 1):
                i = d + k * sps
                if i < len(pr):
                    pr[i] = 0.0
        # scale so the largest possible excursion (all cursors adding) is ±1: an ISI-free NRZ eye
        # is then 100 % open, and a closed eye shows its true extent
        idx = np.arange(d % sps, len(pr), sps)
        A = float(np.sum(np.abs(pr[idx])))
        return y / A, a, d, sps, lev / A, pr

    @staticmethod
    def fir(x, w, s, pre):
        out = np.zeros(len(x))
        for i, wi in enumerate(w):
            sh = (i - pre) * s
            if sh >= 0:
                out[sh:] += wi * x[:len(x) - sh]
            else:
                out[:sh] += wi * x[-sh:]
        return out

    def draw(self, p, accumulate):
        out = {}
        for key, M in (("nrz", 2), ("pam", 4)):
            y, a, d, s, lev, pr = self.link(p, M)
            self.plot(key).eye(key, y, s, n_sym=2, offset=d - s + 40 * s, yrange=(-1.6, 1.6),
                               accumulate=accumulate and p.persist, decay=0.85)
            hgt, _, _ = eye_metrics(y, a * lev[-1], d, s, lev)
            out[key] = hgt * (lev[1] - lev[0]) / 2
            if M == 4:
                k = np.arange(-2, 9)
                idx = d + k * s
                ok = (idx >= 0) & (idx < len(pr))
                pp = self.plot("pulse")
                pp.hline("z", 0, color=GRAY, style="-", width=0.8)
                pp.line("pr", (np.arange(len(pr)) - d) / s, pr, color=GRAY, width=1.2)
                pp.stems("c", k[ok], pr[idx[ok]], color=RED, size=8, name="ISI cursors")
                pp.stems("m", [0], [1.0], color=GREEN, size=10, name="main cursor")
        h, hc, C = self.chan(p, self.sb)
        nf = 1 << 13
        f = np.fft.rfftfreq(nf, 1 / self.sb)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, nf)) + 1e-9)
        Cd = 20 * np.log10(np.abs(C) + 1e-9)
        pr_ = self.plot("resp")
        pr_.line("h", f, H, color=NAVY, width=2.0, name="channel")
        if p.ctle > 0:
            pr_.line("c", f, Cd, color=ORANGE, width=1.6, style="--", name="CTLE (NRZ)")
            pr_.line("t", f, H + Cd, color=GREEN, width=2.0, name="channel × CTLE")
        lq = float(np.interp(0.25, f, H))
        lh = float(np.interp(0.5, f, H))
        pr_.vline("q", 0.25, color=PURPLE, style=":", width=1.2, label=f"R_b/4: {lq:.0f} dB", label_pos=0.9)
        pr_.vline("hh", 0.5, color=RED, style=":", width=1.2, label=f"R_b/2: {lh:.0f} dB", label_pos=0.78)
        win = "PAM-4" if out["pam"] > out["nrz"] else "NRZ"
        if out["pam"] <= 0 and out["nrz"] <= 0:
            win = "neither"
        self.readout(nrz=100 * out["nrz"], pam=100 * out["pam"], lq=-lq, win=win)

    def update(self, p):
        for k in ("nrz", "pam"):
            self.plot(k).reset_persistence()
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        r = self.r
        s = (f"<p>Both lanes carry the same bit rate. NRZ must get through the loss at R_b/2 "
             f"({v(p.loss, '.1f', 'dB')}); PAM-4, at half the symbol rate, only sees the loss at R_b/4 "
             f"({v(r.get('lq', 0), '.1f', 'dB')}), but stacks three eyes in the same swing (−9.5 dB).</p>")
        eq = []
        if p.ctle > 0:
            eq.append(f"the <b>CTLE</b> boosts the Nyquist frequency by {v(p.ctle, '.1f', 'dB')}, and the "
                      f"noise with it")
        if p.ffe > 0:
            eq.append(f"a {v(p.ffe, 'd')}-tap <b>FFE</b> subtracts scaled neighbours, flattening the "
                      f"pulse response")
        if p.dfe > 0:
            eq.append(f"a {v(p.dfe, 'd')}-tap <b>DFE</b> cancels the first post-cursors using past "
                      f"decisions, without amplifying noise (but it cannot touch pre-cursors)")
        if eq:
            s += "<p>Now " + "; ".join(eq) + ".</p>"
        else:
            s += ("<p>No equaliser yet: the pulse response (bottom right) smears over many symbols. "
                  "Every real SerDes uses a CTLE, an FFE and a DFE together.</p>")
        s += (f"<p>Wider eye: {v(r.get('win', '—'))}. Past about 30 dB, NRZ cannot be rescued while "
              f"PAM-4 still can: the arithmetic behind 400G Ethernet and PCIe 6.0.</p>")
        return "<h3>Equalising a backplane</h3>" + s + keybox(
            "CTLE: cheap analog boost. FFE: linear, shapes pre- and post-cursors. DFE: removes "
            "post-cursors without noise gain. PAM-4 wins when the loss slope is steep.")


# =============================================================================== 7. duobinary
class Duobinary(Experiment):
    title = "Duobinary & precoding"
    blurb = "Controlled ISI fits the Nyquist band; one XOR at the transmitter stops errors spreading."
    book = "sec:ch08:pr"
    controls = [
        Choice("kind", "Partial response", ["Duobinary 1+D", "Modified 1−D²"]),
        Toggle("pre", "Precoding at the transmitter", False),
        Slider("ebn0", "Eb/N0", 2, 16, 16, step=0.5, unit="dB"),
        Toggle("inject", "Force one channel error (at bit 6)", False),
        Button("again", "New bits"),
    ]
    plots = [
        Plot("walk", "Bit by bit", x="bit k", y="", legend=None, grid=False),
        EyePlot("eye", "Three-level eye", yrange=(-2.8, 2.8)),
        BERPlot("ber", "Bit error rate (symbol-by-symbol detection)", x="Eb/N0 (dB)",
                ylim=(1e-5, 0.5), xlim=(2, 14), legend="bl"),
    ]
    layout = [["walk", "walk"], ["eye", "ber"]]
    row_stretch = [3, 3]
    readouts = [
        Readout("cerr", "Channel errors (these 20 bits)", "", "int"),
        Readout("berr", "Decoded bit errors", "", "int"),
        Readout("burst", "Longest error burst", "bits", "int"),
        Readout("simber", "Simulated BER here", "", "sci"),
    ]
    challenges = [
        Challenge("Without precoding, inject one channel error and watch it spread to two or more "
                  "decoded bits.",
                  lambda s: not s.p.pre and s.p.inject and s.r.cerr == 1 and s.r.berr >= 2,
                  hint="Try New bits until the error lands before a run of ones."),
        Challenge("Switch precoding on: the same single channel error now costs exactly one bit.",
                  lambda s: s.p.pre and s.p.inject and s.r.cerr == 1 and s.r.berr == 1),
        Challenge("Precoded duobinary: reach a BER below 10⁻³ at an Eb/N0 of 11 dB or less "
                  "(binary needs 6.8 dB: this is the price of three levels).",
                  lambda s: s.p.pre and s.p.kind.startswith("Duo") and s.r.simber_v is not None
                  and 0 < s.r.simber_v < 1e-3 and s.p.ebn0 <= 11),
    ]
    nwin = 20

    def setup(self):
        self.seed = 3
        self.sim = []

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def chain(d, kind, pre, sigma, rng, inject=None):
        order = 1 if kind.startswith("Duo") else 2
        b = sd.duobinary_precode(d, order) if pre else d
        a = 2.0 * b - 1
        k_ = "1+D" if order == 1 else "1-D2"
        y0 = sd.partial_response(a, k_)
        y = y0 + sigma * rng.standard_normal(len(a))
        if inject is not None:
            y[inject] = (2.0 if abs(y0[inject]) < 1 else 0.0)
        if pre:
            dh = sd.duobinary_decode_precoded(y, k_)
        else:
            if order == 1:
                dh = sd.duobinary_decode_feedback(y)
            else:
                out = np.empty(len(y))
                p1, p2 = -1.0, -1.0
                for i, yi in enumerate(y):
                    ai = 1.0 if yi + p2 > 0 else -1.0
                    out[i] = ai
                    p2, p1 = p1, ai
                dh = ((out + 1) / 2).astype(int)
        lev = np.array([-2.0, 0.0, 2.0])
        ych = lev[np.argmin(np.abs(y[:, None] - lev[None, :]), axis=1)]
        cerr = ych != y0
        return b, a, y0, y, dh, cerr

    @staticmethod
    def sigma(ebn0):
        return np.sqrt(2 / (2 * 10 ** (ebn0 / 10)))

    def update(self, p):
        self.r["simber_v"] = None
        rng = np.random.default_rng(self.seed)
        d = rng.integers(0, 2, self.nwin)
        b, a, y0, y, dh, cerr = self.chain(d, p.kind, p.pre, self.sigma(p.ebn0), rng,
                                           inject=6 if p.inject else None)
        wrong = dh != d
        pw = self.plot("walk")
        rows = [("data d", d, None), ("precoded b" if p.pre else "sent b = d", b, None),
                ("level a", a, None), ("received y", y, cerr), ("decided d̂", dh, wrong)]
        for r_, (lab, vals, flag) in enumerate(rows):
            yy = 4 - r_
            pw.text(f"l{r_}", -0.7, yy, lab, color=GRAY, anchor=(1, 0.5), size=9.5, bold=True)
            for k in range(self.nwin):
                val = vals[k]
                txt = f"{val:+.1f}" if lab.startswith("received") else (f"{int(val):+d}" if lab.startswith("level")
                                                                      else str(int(val)))
                col = RED if (flag is not None and flag[k]) else (NAVY if r_ != 1 or p.pre else GRAY)
                pw.text(f"c{r_}_{k}", k, yy, txt, color=col, anchor=(0.5, 0.5),
                        size=10 if r_ != 3 else 8.5, bold=flag is not None and flag[k])
        if p.inject:
            pw.band("inj", 5.5, 6.5, color=ORANGE, alpha=0.15)
        pw.set_xlim(-4.2, self.nwin - 0.4)
        pw.set_ylim(-0.6, 4.6)
        pw.set_yticks([])
        pw.set_xticks([(k, str(k)) for k in range(self.nwin)])
        # burst length
        burst, cur = 0, 0
        for w in wrong:
            cur = cur + 1 if w else 0
            burst = max(burst, cur)
        # eye: sinc-based duobinary waveform
        sp = 16
        tt = np.arange(-12 * sp, 12 * sp + 1) / sp
        pulse = np.sinc(tt) + (np.sinc(tt - 1) if p.kind.startswith("Duo") else -np.sinc(tt - 2))
        ae = 2.0 * rng.integers(0, 2, 700) - 1
        up = np.zeros(len(ae) * sp)
        up[::sp] = ae
        yw = fftconvolve(up, pulse) + self.sigma(p.ebn0) * 0.5 * rng.standard_normal(len(up) + len(pulse) - 1)
        self.plot("eye").eye("eye", yw[12 * sp - sp // 2 + 30 * sp:-30 * sp], sp, n_sym=2,
                             yrange=(-2.8, 2.8))
        pb = self.plot("ber")
        xg = np.linspace(2, 14, 121)
        pb.theory("bin", xg, cl.ber_bpsk(xg), name="binary antipodal")
        if self.sim:
            for key, col, name in (("pre", GREEN, "precoded"), ("fb", RED, "no precoding")):
                pts = [(x_, r_) for x_, k_, r_ in self.sim if k_ == key and r_ > 0]
                if pts:
                    xs, ys = zip(*pts)
                    pb.sim(key, xs, ys, color=col, name=name)
        pb.vline("now", p.ebn0, color=ORANGE, style="-", width=1.4)
        self.readout(cerr=int(cerr.sum()), berr=int(wrong.sum()), burst=burst)

    def background(self, p):
        self.sim = []
        rng = np.random.default_rng(5)
        n = 20_000 if self.quick else 150_000
        # the current point first (for the readout), then the curves
        d = rng.integers(0, 2, n)
        *_, dh, _ = self.chain(d, p.kind, p.pre, self.sigma(p.ebn0), rng)
        yield ("now", float(np.mean(dh != d)), n)
        for e in np.arange(2, 14.1, 2.0 if self.quick else 1.0):
            for pre, key in ((True, "pre"), (False, "fb")):
                d = rng.integers(0, 2, n)
                *_, dh, _ = self.chain(d, p.kind, pre, self.sigma(e), rng)
                yield (float(e), key, float(np.mean(dh != d)))

    def progress(self, p, item):
        if item[0] == "now":
            ber, n = item[1], item[2]
            self.readout(simber=ber if ber > 0 else f"< {st.sci(1 / n, 1)}", simber_v=ber)
            return
        self.sim.append(item)
        pb = self.plot("ber")
        for key, col, name in (("pre", GREEN, "precoded"), ("fb", RED, "no precoding")):
            pts = [(x_, r_) for x_, k_, r_ in self.sim if k_ == key and r_ > 0]
            if pts:
                xs, ys = zip(*pts)
                pb.sim(key, xs, ys, color=col, name=name)

    def story(self, p):
        r = self.r
        duo = p.kind.startswith("Duo")
        s = ((f"<p><b>Duobinary</b> adds each symbol to the previous one on purpose: y = a_k + a_(k−1). "
              f"The pulse fits in exactly the Nyquist bandwidth with no excess, yet rolls off gently, and "
              f"the eye has three levels (−2, 0, +2).</p>") if duo else
             ("<p><b>Modified duobinary</b> (1 − D²) also has a spectral null at DC: the class-4 partial "
              "response of magnetic recording (PRML).</p>"))
        if p.pre:
            s += ("<p>With <b>precoding</b> (b_k = d_k ⊕ b_(k−1)) the receiver decides each bit from one "
                  "sample: |y| &lt; 1 means a one. No memory, so an error stays where it happened: "
                  f"{v(r.get('berr', 0), 'd')} decoded error(s).</p>")
        else:
            s += ("<p>Without precoding the receiver must subtract its previous decision, â_k = y_k − "
                  "â_(k−1). One wrong decision feeds the next: the red run in the bottom row "
                  f"(longest burst {v(r.get('burst', 0), 'd', 'bits')}).</p>")
        s += ("<p>The price of symbol-by-symbol detection is a few dB against binary (the circles "
              "against the line): each of the two eyes is half the swing. A Viterbi detector on the "
              "two-state trellis wins almost all of it back.</p>")
        return "<h3>A blur you can undo</h3>" + s + keybox(
            "Precoding moves the inverse of the channel's memory to the transmitter, where there is "
            "no noise. Tomlinson–Harashima and MIMO precoders use the same idea.")


# =============================================================================== the lab
LAB = st.Lab(13, "Line Codes, Jitter and SerDes Eyes", chapter=8,
             chapter_title="Baseband Transmission and Pulse Shaping",
             experiments=[LineCodes, BaselineWander, Scramblers, Code8b10b, EyesBathtub, PAM4Channel,
                          Duobinary])

if __name__ == "__main__":
    st.run(LAB)
