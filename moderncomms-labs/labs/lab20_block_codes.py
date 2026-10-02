"""Lab 20 · Block Codes: Hamming to Reed–Solomon   (Chapter 14)

Run it:      python labs/lab20_block_codes.py
Self-test:   python labs/lab20_block_codes.py --selftest

The algebraic codes of the 1950s and 1960s are still everywhere: a Hamming-derived SECDED code
guards every word of server memory, a CRC ends every Ethernet frame and every 5G transport
block, and Reed–Solomon codes protect QR codes, CDs, deep-space links and RAID-6 arrays.
Eight experiments: Hamming (7,4) on a Venn diagram (click bits to flip them), SECDED in
memory, a CRC calculator with all the conventions, what a CRC can and cannot detect, a
GF(2^m) playground, Reed–Solomon with errors and erasures, interleaving against bursts, and
the coding gain of the whole family at your target error rate.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import brentq

from commlib import blockcodes as bc
from commlib import fectools as ft
from commlib import gf
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, GRAY, TEAL, GOLD)
from studio import v, keybox, good, bad
from _fecviz import Canvas, on_click, cells, segments, circle, pale, ink, bitstr


# =============================================================================== 1. Hamming (7,4)
HAM = bc.hamming_code(3)
#               d1            d2             d3            d4          p1            p2           p3
BIT_POS = [(0.0, 0.92), (-0.66, -0.28), (0.66, -0.28), (0.0, 0.08), (-1.12, 0.78), (1.12, 0.78),
           (0.0, -1.22)]
CIRCLES = [(-0.5, 0.33), (0.5, 0.33), (0.0, -0.5)]
BIT_NAMES = ["d₁", "d₂", "d₃", "d₄", "p₁", "p₂", "p₃"]


class HammingVenn(Experiment):
    title = "Hamming (7,4) on a Venn diagram"
    blurb = "Click any bit to flip it; failing parity circles turn red and point at the culprit."
    book = "sec:ch14:hamming"
    controls = [
        IntSlider("msg", "Message (4 bits, as a number)", 0, 15, 11),
        Heading("Channel errors (or click a bit)"),
        Button("one", "1 random error"),
        Button("two", "2 random errors"),
        Button("clear", "Clear all errors"),
    ]
    plots = [
        Canvas("venn", "Three parity circles: each must contain an even number of ones",
               xlim=(-2.1, 2.1), ylim=(-1.85, 1.75), aspect=True),
        Canvas("mat", "The syndrome s = r·Hᵀ picks a column of H: flip that bit",
               xlim=(-2.8, 9.6), ylim=(-3.9, 3.1)),
    ]
    layout = [["venn", "mat"]]
    col_stretch = [5, 4]
    readouts = [
        Readout("syn", "Syndrome", "", None),
        Readout("ptr", "Points at", "", None),
        Readout("nerr", "Errors injected", "", "int"),
        Readout("verdict", "Decoder result", "", None),
    ]
    challenges = [
        Challenge("Make the decoder miscorrect: it outputs a wrong codeword without any warning.",
                  lambda s: s.r.verdict == "miscorrected"),
        Challenge("Use exactly two errors to make the decoder flip the centre bit d₄.",
                  lambda s: s.r.nerr == 2 and s.r.syn == "111",
                  hint="Which two columns of H add up to d₄'s column?"),
        Challenge("Find an error pattern the code cannot even see: errors present, syndrome 000.",
                  lambda s: s.r.nerr > 0 and s.r.syn == "000",
                  hint="The pattern must itself be a codeword."),
    ]

    def setup(self):
        self.err = np.zeros(7, int)
        self.err[5] = 1                                    # the chapter's example: bit 6
        on_click(self, "venn", self.clicked_venn)
        on_click(self, "mat", self.clicked_rows)

    def clicked_venn(self, x, y):
        d = [np.hypot(x - a, y - b) for a, b in BIT_POS]
        j = int(np.argmin(d))
        if d[j] < 0.32:
            self.err[j] ^= 1
            return True
        return False

    def clicked_rows(self, x, y):
        if abs(y + 1.75) < 0.4:
            j = int(round(x))
            if 0 <= j < 7:
                self.err[j] ^= 1
                return True
        return False

    def _rand(self, k):
        self.err[:] = 0
        self.err[self.rng.choice(7, k, replace=False)] = 1

    def on_one(self, p):
        self._rand(1)

    def on_two(self, p):
        self._rand(2)

    def on_clear(self, p):
        self.err[:] = 0

    def update(self, p):
        u = np.array([(p.msg >> (3 - i)) & 1 for i in range(4)])
        c = HAM.encode(u)
        r = (c + self.err) % 2
        s = HAM.syndrome(r)
        uh, _, _ = HAM.decode(r)
        chat = HAM.encode(uh)
        col = next((j for j in range(7) if np.array_equal(HAM.H[:, j], s)), None)
        pv = self.plot("venn")
        fails = [(HAM.H[i] @ r) % 2 for i in range(3)]
        for i, ((cx, cy), f) in enumerate(zip(CIRCLES, fails)):
            circle(pv, f"c{i}", cx, cy, 1.0, color=RED if f else GREEN, width=4 if f else 2.2)
            lx, ly = [(-1.55, 1.42), (1.55, 1.42), (0.0, -1.68)][i]
            pv.text(f"cl{i}", lx, ly, f"check {i + 1}: {'odd ✗' if f else 'even ✓'}",
                    color=RED if f else GREEN, anchor=(0.5, 0.5), size=9.5, bold=True)
        xs, ys = zip(*BIT_POS)
        colors = []
        for j in range(7):
            if self.err[j]:
                colors.append(RED)
            elif col == j:
                colors.append(ORANGE)
            else:
                colors.append(NAVY if j < 4 else TEAL)
        cells(pv, "bits", xs, ys, colors, w=0.42, h=0.42, labels=[str(b) for b in r],
              text_color=ink(pv), size=13, bold=True)
        for j in range(7):
            pv.text(f"n{j}", xs[j] + 0.25, ys[j] + 0.22, BIT_NAMES[j], color=GRAY, size=8.5,
                    anchor=(0, 1))
        # the matrix view
        pm = self.plot("mat")
        X, Y = np.meshgrid(np.arange(7), [2.4, 1.6, 0.8])
        Hc = [NAVY if b else pale(pm) for b in HAM.H.ravel()]
        cells(pm, "H", X.ravel(), Y.ravel(), Hc, w=0.8, h=0.7,
              labels=[str(b) for b in HAM.H.ravel()],
              text_color=[ink(pm) if b else pm.theme.text for b in HAM.H.ravel()], size=10)
        pm.text("Hl", -0.7, 1.6, "H", anchor=(1, 0.5), size=13, bold=True)
        for j in range(7):
            pm.text(f"hn{j}", j, 3.0, BIT_NAMES[j], color=GRAY, anchor=(0.5, 0.5), size=9)
        if col is not None:
            pm.band("colband", col - 0.48, col + 0.48, color=ORANGE, alpha=0.25)
        cells(pm, "s", np.full(3, 8.2), [2.4, 1.6, 0.8], [ORANGE if b else pale(pm) for b in s],
              w=0.8, h=0.7, labels=[str(b) for b in s],
              text_color=[ink(pm) if b else pm.theme.text for b in s], size=10)
        pm.text("sl", 8.2, 3.0, "s", anchor=(0.5, 0.5), size=11, bold=True)
        pm.text("eq", 7.5, 1.6, "=", anchor=(0.5, 0.5), size=13)
        for k, (row, y, lab) in enumerate(((c, -0.55, "sent"), (r, -1.75, "received"),
                                             (chat, -2.95, "decoded"))):
            cl = [RED if row[j] != c[j] else (NAVY if row[j] else pale(pm)) for j in range(7)]
            cells(pm, f"row{k}", np.arange(7), np.full(7, y), cl, w=0.8, h=0.75,
                  labels=[str(b) for b in row],
                  text_color=[ink(pm) if (row[j] != c[j] or row[j]) else pm.theme.text
                              for j in range(7)], size=11)
            pm.text(f"rl{k}", -0.7, y, lab, anchor=(1, 0.5), size=9.5)
        nerr = int(self.err.sum())
        ok = np.array_equal(chat, c)
        verdict = ("no errors" if nerr == 0 else ("corrected" if ok else
                   ("undetected" if not s.any() else "miscorrected")))
        self.readout(syn=bitstr(s), ptr="nothing" if col is None else f"bit {col + 1} ({BIT_NAMES[col]})",
                     nerr=nerr, verdict=verdict)

    def story(self, p):
        r = self.r
        verdict = r.get("verdict", "")
        s = ("<p>Four data bits (navy) sit in the overlaps of three circles; each circle adds a "
             "parity bit (teal) that makes the number of ones inside it even. Seven bits, three "
             "checks: the (7,4) Hamming code of 1950.</p>"
             "<p>Flip one bit and exactly the circles that contain it go odd (red). The pattern "
             "of failing circles is the <b>syndrome</b>, and it is different for every one of the "
             "seven positions: it equals that bit's column of H (right). So the decoder knows "
             "which bit to flip back (orange).</p>")
        if verdict == "corrected":
            s += "<p>" + good(f"One error, syndrome {r['syn']}: found and corrected.") + "</p>"
        elif verdict == "miscorrected":
            s += ("<p>" + bad("Two errors: their columns add up to a third column, and the "
                              "decoder confidently flips a bit that was right. Now three bits "
                              "are wrong.") + " With d_min = 3 the code can correct one error "
                  "or detect two, never both.</p>")
        elif verdict == "undetected":
            s += ("<p>" + bad("The errors form a codeword themselves: every circle is even, and "
                              "nothing looks wrong.") + "</p>")
        else:
            s += "<p>Click a bit (on the circles or in the rows) to flip it.</p>"
        return "<h3>Parity circles</h3>" + s + keybox(
            "The syndrome depends only on the error, not on the message: decoding is a table "
            "look-up. 2³ − 1 = 7 syndromes for 7 single errors: a perfect code.")


# =============================================================================== 2. SECDED
SEC_CODES = {"Hamming (7,4): correct only": "h74", "Extended Hamming (8,4): SECDED": "e84",
             "Hsiao (72,64): ECC memory": "hsiao"}


@lru_cache(maxsize=4)
def sec_code(key):
    return {"h74": HAM, "e84": bc.extended_hamming_code(3), "hsiao": bc.hsiao_secded_72_64()}[key]


@lru_cache(maxsize=4)
def sec_outcomes(key):
    """Fractions (corrected, detected, silent) for 1..4 bit flips (exhaustive or 30 000 samples)."""
    code = sec_code(key)
    n = code.n
    H = code.H
    cols = {int("".join(map(str, H[:, j])), 2): j for j in range(n)}
    secded = key != "h74"
    rng = np.random.default_rng(20)
    out = []
    for k in range(1, 5):
        if comb(n, k) <= 70000:
            pats = np.array(list(combinations(range(n), k)))
        else:
            pats = np.array([rng.choice(n, k, replace=False) for _ in range(30000)])
        E = np.zeros((len(pats), n), dtype=np.int64)
        np.put_along_axis(E, pats, 1, axis=1)
        S = (E @ H.T) % 2
        sint = S @ (1 << np.arange(H.shape[0] - 1, -1, -1))
        wt = S.sum(axis=1)
        corr = det = sil = 0
        for i in range(len(pats)):
            si = int(sint[i])
            if si == 0:
                sil += 1
                continue
            if secded and wt[i] % 2 == 0:
                det += 1
                continue
            j = cols.get(si)
            if j is None:
                det += 1
            elif k == 1 and j == pats[i][0]:
                corr += 1
            else:
                sil += 1
        N = len(pats)
        out.append((corr / N, det / N, sil / N))
    return np.array(out)


class Secded(Experiment):
    title = "SECDED: guarding memory"
    blurb = "Correct one flip, detect two: what server memory does with 1, 2, 3 and 4 bit flips."
    book = "sec:ch14:hamming"
    controls = [
        Choice("code", "Code", list(SEC_CODES), "Hsiao (72,64): ECC memory", style="menu"),
        LogSlider("p", "Probability that a stored bit flips", 1e-9, 1e-2, 1e-4,
                  help="Per bit, between two reads of the word (scrub interval)"),
    ]
    plots = [
        Plot("bars", "What the decoder does with k flipped bits", x="bits flipped in the word",
             y="fraction of patterns (%)", xlim=(0.4, 4.6), ylim=(0, 118), legend="tl",
             legend_cols=3),
        Plot("rates", "Per word: silent corruption and detected errors",
             x="bit flip probability p", y="probability per word", logx=True, logy=True,
             xlim=(1e-9, 1e-2), ylim=(1e-30, 1), legend="tl"),
    ]
    layout = [["bars", "rates"]]
    readouts = [
        Readout("sil", "Silent corruption / word", "", "sci", good=lambda x: x < 1e-15),
        Readout("det", "Detected, uncorrectable / word", "", "sci"),
        Readout("ovh", "Check-bit overhead", "%", ".1f"),
        Readout("two", "2 flips → silent", "%", ".0f"),
    ]
    challenges = [
        Challenge("Find the code for which two flipped bits silently corrupt the data.",
                  lambda s: s.r.two > 50),
        Challenge("With p of at least 10⁻⁶, keep silent corruption below 10⁻¹⁵ per word.",
                  lambda s: s.p.p >= 1e-6 * 0.999 and s.r.sil < 1e-15),
        Challenge("With at most 15 % overhead and p ≥ 10⁻⁶, keep silent corruption below 10⁻¹² "
                  "per word.", lambda s: s.r.ovh <= 15 and s.p.p >= 1e-6 * 0.999 and s.r.sil < 1e-12),
    ]

    def word_rates(self, key, p):
        code = sec_code(key)
        n = code.n
        fr = sec_outcomes(key)
        p = np.asarray(p, float)
        Pk = [comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(1, 5)]
        sil = sum(Pk[i] * fr[i, 2] for i in range(4))
        det = sum(Pk[i] * fr[i, 1] for i in range(4))
        return sil, det

    def update(self, p):
        key = SEC_CODES[p.code]
        fr = sec_outcomes(key)
        pb = self.plot("bars")
        k = np.arange(1, 5)
        pb.bars("c", k - 0.26, 100 * fr[:, 0], width=0.25, color=GREEN)
        pb.bars("d", k, 100 * fr[:, 1], width=0.25, color=ORANGE)
        pb.bars("s", k + 0.26, 100 * fr[:, 2], width=0.25, color=RED)
        for lk_, col, lab in (("lc", GREEN, "corrected"), ("ld", ORANGE, "detected"),
                              ("ls", RED, "silent corruption")):
            pb.scatter(lk_, [-5], [-5], color=col, size=11, symbol="s", name=lab)
        pb.set_xticks([(i, str(i)) for i in k])
        pp = self.plot("rates")
        ps = np.logspace(-9, -2, 120)
        named = False
        for name, kk in SEC_CODES.items():
            sil, det = self.word_rates(kk, ps)
            mine = kk == key
            nm = "silent (this code)" if mine else (None if named else "silent (other codes)")
            named = named or not mine
            pp.line(f"s{kk}", ps, np.maximum(sil, 1e-300), color=RED if mine else GRAY,
                    width=2.6 if mine else 1.0, name=nm)
            if mine:
                pp.line("det", ps, np.maximum(det, 1e-300), color=ORANGE, width=2.2, style="--",
                        name="detected (this code)")
        pp.line("o1", ps, ps * 0 + 1e-15, color=GREEN, style=":", width=1.0)
        pp.text("o1t", 1.4e-9, 1e-15, "10⁻¹⁵", color=GREEN, anchor=(0, 1), size=8.5)
        sil, det = self.word_rates(key, p.p)
        pp.vline("p", p.p, color=NAVY, style="--")
        pp.scatter("me", [p.p], [max(float(sil), 1e-300)], color=RED, size=12, z=5)
        code = sec_code(key)
        self.readout(sil=float(sil), det=float(det), ovh=100 * code.r / code.k, two=100 * fr[1, 2])

    def story(self, p):
        key = SEC_CODES[p.code]
        code = sec_code(key)
        fr = sec_outcomes(key)
        s = (f"<p>The {v(p.code.split(':')[0])} code stores {v(code.k, 'd')} data bits with "
             f"{v(code.r, 'd')} check bits. Left: what its decoder does with every pattern of k "
             "flipped bits.</p>")
        if key == "h74":
            s += ("<p>" + bad("A plain Hamming code has no way to tell two errors from one: "
                              "every non-zero syndrome points at some bit, so two flips are "
                              "always 'corrected' into a third wrong bit.") + "</p>")
        else:
            s += ("<p>Adding one overall parity bit (or, in Hsiao's construction, giving every "
                  "column of H odd weight) splits the syndromes: odd weight means an odd number "
                  "of errors (assume one and correct it), even weight means an even number "
                  "(stop and raise an alarm). One flip: always corrected. Two: always detected. "
                  f"Three: {v(100 * fr[2, 2], '.0f', '%')} are silently 'corrected' into wrong "
                  "data, the classic SECDED pitfall.</p>")
        sil_s = f"{self.r.get('sil', 0):.1e}"
        s += (f"<p>At p = {v(f'{p.p:.0e}')} per bit, a word is silently corrupted with "
              f"probability {v(sil_s)}. Memory "
              "systems scrub (read and rewrite) every word regularly so that single flips never "
              "get the chance to pair up.</p>")
        return "<h3>Correct one, detect two</h3>" + s + keybox(
            "Silent corruption scales as p³ for SECDED and p² for a plain Hamming code: the "
            "extra check bit buys orders of magnitude.")


# =============================================================================== 3. CRC calculator
MESSAGES = ["123456789", "Hello, world", "MDC", "The quick brown fox jumps over the lazy dog",
            "four zero bytes"]
CRC_NAMES = list(gf.CRC_CATALOG)


def msg_bytes(name):
    return b"\x00\x00\x00\x00" if name == "four zero bytes" else name.encode()


def crc_trace(data, width, poly, init, refin):
    """Register contents after every message bit (bit-serial, Rocksoft model)."""
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    reg = init
    states = []
    for byte in data:
        if refin:
            byte = int(format(byte, "08b")[::-1], 2)
        for i in range(7, -1, -1):
            bit = (byte >> i) & 1
            fb = ((reg & top) != 0) ^ bit
            reg = (reg << 1) & mask
            if fb:
                reg ^= poly
            states.append(reg)
    return states


class CrcCalculator(Experiment):
    title = "A CRC calculator"
    blurb = "Polynomial division in a shift register, plus the conventions every standard adds."
    book = "sec:ch14:crc"
    controls = [
        Choice("msg", "Message", MESSAGES, "123456789", style="menu"),
        Choice("crc", "CRC", CRC_NAMES, "CRC-16/CCITT-FALSE", style="menu"),
        Choice("conv", "Conventions", ["As in the catalogue", "Custom"], "As in the catalogue"),
        Toggle("c_init", "Initial register all ones", False, enabled_if=lambda p: p.conv == "Custom"),
        Toggle("c_refin", "Reflect each input byte", False, enabled_if=lambda p: p.conv == "Custom"),
        Toggle("c_refout", "Reflect the result", False, enabled_if=lambda p: p.conv == "Custom"),
        Toggle("c_xor", "Final XOR with all ones", False, enabled_if=lambda p: p.conv == "Custom"),
        Heading("Avalanche"),
        IntSlider("flip", "Flip message bit number", 0, 72, 0, help="0 = no flip"),
    ]
    plots = [
        ImagePlot("reg", "The CRC register after every message bit (dark = 1)",
                  x="message bit", y="register bit"),
        Canvas("out", "The CRC: original message (top) and with the flipped bit (bottom)",
               xlim=(-3.5, 33), ylim=(-0.9, 2.0)),
    ]
    layout = [["reg"], ["out"]]
    row_stretch = [3, 1]
    readouts = [
        Readout("crc", "CRC", "", None),
        Readout("check", "Catalogue check (\"123456789\")", "", None),
        Readout("width", "Width", "bits", "int"),
        Readout("changed", "CRC bits changed by the flip", "", "int"),
    ]
    challenges = [
        Challenge("Turn CRC-16/CCITT-FALSE into CRC-16/KERMIT by changing only the conventions "
                  "(CRC of \"123456789\" = 0x2189).",
                  lambda s: s.p.crc == "CRC-16/CCITT-FALSE" and s.p.msg == "123456789"
                  and s.exp.value == 0x2189),
        Challenge("Get a CRC of exactly zero for four zero bytes: see why standards start from "
                  "all ones.", lambda s: s.p.msg == "four zero bytes" and s.exp.value == 0),
        Challenge("Flip one message bit so that at least 20 of the CRC-32's bits change.",
                  lambda s: s.p.crc.startswith("CRC-32 (") and s.p.flip > 0 and s.r.changed >= 20),
    ]

    def setup(self):
        self.value = None

    def params(self, p):
        w, poly, i0, ri, ro, xo, chk = gf.CRC_CATALOG[p.crc]
        if p.conv == "Custom":
            m = (1 << w) - 1
            i0, ri, ro, xo = (m if p.c_init else 0), p.c_refin, p.c_refout, (m if p.c_xor else 0)
        return w, poly, i0, ri, ro, xo, chk

    def update(self, p):
        w, poly, i0, ri, ro, xo, chk = self.params(p)
        data = msg_bytes(p.msg)
        val = gf.crc_bits(data, w, poly, i0, ri, ro, xo)
        self.value = val
        states = crc_trace(data, w, poly, i0, ri)
        img = np.array([[(s >> (w - 1 - b)) & 1 for s in states] for b in range(w)], float)
        pr = self.plot("reg")
        nb = len(states)
        pr.set_xlim(0, nb)
        pr.set_ylim(0, w)
        pr.image("img", img[::-1], x=(0, nb), y=(0, w),
                 cmap=((0, pale(pr)), (1, NAVY)), levels=(0, 1))
        for k in range(8, nb, 8):
            pr.vline(f"byte{k}", k, color=GRAY, style=":", width=0.6)
        nbits = 8 * len(data)
        fl = min(p.flip, nbits)
        if fl > 0:
            d2 = bytearray(data)
            d2[(fl - 1) // 8] ^= 1 << (7 - (fl - 1) % 8)
            val2 = gf.crc_bits(bytes(d2), w, poly, i0, ri, ro, xo)
            pr.vline("flip", fl - 0.5, color=RED, style="--", label="flipped bit", label_pos=0.9)
        else:
            val2 = val
        po = self.plot("out")
        po.set_xlim(-3.8, max(w, 16) + 0.5)
        b1 = [(val >> (w - 1 - j)) & 1 for j in range(w)]
        b2 = [(val2 >> (w - 1 - j)) & 1 for j in range(w)]
        cells(po, "c1", np.arange(w) + 0.5, np.full(w, 1.15), [NAVY if b else pale(po) for b in b1],
              labels=b1 if w <= 24 else None, text_color=[ink(po) if b else po.theme.text for b in b1],
              size=9)
        cells(po, "c2", np.arange(w) + 0.5, np.full(w, 0.0),
              [RED if b != a else (NAVY if b else pale(po)) for a, b in zip(b1, b2)],
              labels=b2 if w <= 24 else None,
              text_color=[ink(po) if (b or b != a) else po.theme.text for a, b in zip(b1, b2)], size=9)
        po.text("l1", -0.3, 1.15, f"0x{val:0{w // 4}X}", anchor=(1, 0.5), size=10, bold=True)
        po.text("l2", -0.3, 0.0, f"0x{val2:0{w // 4}X}" if fl else "no flip", anchor=(1, 0.5),
                size=10, bold=True, color=RED if fl else GRAY)
        chk_now = gf.crc_bits(b"123456789", w, poly, i0, ri, ro, xo)
        self.readout(crc=f"0x{val:0{w // 4}X}",
                     check=(f"✓ 0x{chk:0{w // 4}X}" if chk_now == chk else f"✗ ≠ 0x{chk:0{w // 4}X}"),
                     width=w, changed=bin(val ^ val2).count("1"))

    def story(self, p):
        w, poly, i0, ri, ro, xo, chk = self.params(p)
        s = (f"<p>A CRC is the remainder of a polynomial division: the message bits are the "
             f"coefficients of a long polynomial, divided (XOR instead of subtraction) by the "
             f"{v(w, 'd')}-bit generator 0x{poly:0{w // 4}X}. The shift register on top does it "
             "bit by bit: each column is the register after one more message bit. Whatever is "
             "left at the end is the CRC, appended to the frame.</p>"
             "<p>Real standards add four conventions (Williams' \"Rocksoft model\"): the initial "
             f"register value (here 0x{i0:0{w // 4}X}), whether each byte is fed LSB first "
             f"({'yes' if ri else 'no'}), whether the result is reflected "
             f"({'yes' if ro else 'no'}), and a final XOR (0x{xo:0{w // 4}X}). Change any of them "
             "and you get a different, incompatible CRC: every catalogue therefore lists the CRC "
             "of the ASCII string \"123456789\" as a check.</p>")
        if i0 == 0 and p.msg == "four zero bytes":
            s += ("<p>" + bad("With an all-zero start, leading zero bytes leave the register at "
                              "zero: the CRC cannot tell how many there were.") + " That is why "
                  "Ethernet starts from all ones.</p>")
        if p.flip:
            s += (f"<p>Flipping one message bit changed {v(self.r.get('changed', 0), 'd')} of the "
                  f"{w} CRC bits (bottom row, red): the CRC of a one-bit error is the generator's "
                  "remainder pattern, never zero, so a single error is always detected.</p>")
        return "<h3>Division in a shift register</h3>" + s + keybox(
            "Ethernet (CRC-32), USB, Bluetooth, 5G (CRC-24/16/11/6): the same few lines of logic, "
            "different polynomials and conventions.")


# =============================================================================== 4. what a CRC detects
CRC_POLYS = {"CRC-4: x⁴ + x + 1": 0b10011,
             "CRC-8/ATM: x⁸ + x² + x + 1": 0x107,
             "x⁸ + 1 (a poor choice)": 0x101,
             "CRC-8/AUTOSAR (0x2F)": 0x12F,
             "CRC-16/CCITT: x¹⁶ + x¹² + x⁵ + 1": 0x11021}
FRAME = 64


class CrcDetection(Experiment):
    title = "What a CRC can and cannot see"
    blurb = "Throw bursts and random errors at 64-bit frames and count what slips through."
    book = "sec:ch14:crc"
    heavy = True
    controls = [
        Choice("poly", "Generator polynomial", list(CRC_POLYS), "CRC-4: x⁴ + x + 1", style="menu"),
        IntSlider("blen", "Burst length to inspect", 1, 32, 6, unit="bits"),
        Button("rerun", "Throw more errors", primary=True),
    ]
    plots = [
        Plot("burst", "Undetected bursts (first and last bit wrong, random in between)",
             x="burst length (bits)", y="fraction undetected", xlim=(0.5, 32.5), ylim=(1e-5, 1),
             logy=True, legend="tr"),
        Plot("weight", "Undetected random errors of a given weight", x="number of bit errors",
             y="fraction undetected", xlim=(0.5, 8.5), ylim=(1e-5, 1), logy=True, legend=None),
    ]
    layout = [["burst", "weight"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("r", "CRC degree r", "", "int"),
        Readout("bl", "Undetected at your burst length", "", "sci"),
        Readout("w2", "Undetected double errors", "", "sci"),
        Readout("odd", "Undetected odd-weight errors", "", "sci"),
    ]
    challenges = [
        Challenge("Find a polynomial that catches every double-bit error in a 64-bit frame.",
                  lambda s: s.r.w2 is not None and s.r.w2 == 0 and s.exp.done,
                  hint="x⁴ + x + 1 repeats with period 15: two errors 15 apart cancel."),
        Challenge("Find a polynomial that catches every odd number of errors (weights 1, 3, 5, 7).",
                  lambda s: s.r.odd is not None and s.r.odd == 0 and s.exp.done),
        Challenge("For an 8-bit CRC, find the burst length that slips through about twice as "
                  "often as long bursts do.", lambda s: s.r.r == 8 and s.p.blen == 9),
    ]

    def setup(self):
        self.run_id = 0
        self.done = False

    def on_rerun(self, p):
        self.run_id += 1

    def update(self, p):
        g = CRC_POLYS[p.poly]
        r = g.bit_length() - 1
        self.res_b, self.res_w = {}, {}
        self.done = False
        pb = self.plot("burst")
        pb.line("t1", [0.5, 32.5], [2.0 ** -r] * 2, color=GREEN, style="--", width=1.4,
                name=f"2⁻ʳ = {2.0 ** -r:.1e}")
        pb.line("t2", [0.5, 32.5], [2.0 ** -(r - 1)] * 2, color=ORANGE, style=":", width=1.4,
                name="2⁻⁽ʳ⁻¹⁾ (bursts of r + 1)")
        pb.band("safe", 0.5, r + 0.5, color=GREEN, alpha=0.10)
        pb.text("safet", 0.7, 2e-5, f"length ≤ {r}: always caught", color=GREEN, anchor=(0, 0),
                size=8.5)
        pb.vline("me", p.blen, color=NAVY, style="--", width=1.0)
        self.plot("weight").hline("t1", 2.0 ** -r, color=GREEN, style="--")
        self.readout(r=r, bl=None, w2=None, odd=None)

    def background(self, p):
        g = CRC_POLYS[p.poly]
        rng = np.random.default_rng(2000 + self.run_id)
        N = 3000 if self.quick else 40000
        for L in range(1, 33):
            st_ = rng.integers(0, FRAME - L + 1, N).astype(np.uint64)
            if L == 1:
                pat = np.ones(N, dtype=np.uint64)
            else:
                inner = rng.integers(0, 2 ** max(L - 2, 0), N, dtype=np.uint64) if L > 2 else \
                    np.zeros(N, dtype=np.uint64)
                pat = (np.uint64(1) << np.uint64(L - 1)) | (inner << np.uint64(1)) | np.uint64(1)
            e = pat << st_
            yield dict(kind="b", L=L, u=float(np.mean(ft.gf2_mod_many(e, g) == 0)), N=N)
        for wgt in range(1, 9):
            pos = np.argsort(rng.random((N, FRAME)), axis=1)[:, :wgt].astype(np.uint64)
            e = np.bitwise_or.reduce(np.uint64(1) << pos, axis=1)
            yield dict(kind="w", w=wgt, u=float(np.mean(ft.gf2_mod_many(e, g) == 0)), N=N)
        yield dict(kind="done")

    def progress(self, p, it_):
        floor = 1.2e-5
        if it_["kind"] == "b":
            self.res_b[it_["L"]] = it_["u"]
            xs = sorted(self.res_b)
            ys = [max(self.res_b[x], floor) for x in xs]
            pb = self.plot("burst")
            nz = [x for x in xs if self.res_b[x] > 0]
            zz = [x for x in xs if self.res_b[x] == 0]
            segments(pb, "bars", nz, np.full(len(nz), 1e-5), nz, [self.res_b[x] for x in nz],
                     color=NAVY, width=9)
            segments(pb, "zero", zz, np.full(len(zz), 1e-5), zz, np.full(len(zz), 2.5e-5),
                     color=GREEN, width=9)
            if p.blen in self.res_b:
                self.readout(bl=self.res_b[p.blen])
                pb.scatter("meb", [p.blen], [max(self.res_b[p.blen], floor)], color=ORANGE, size=12,
                           symbol="d", z=5)
        elif it_["kind"] == "w":
            self.res_w[it_["w"]] = it_["u"]
            xs = sorted(self.res_w)
            pw = self.plot("weight")
            segments(pw, "bars", xs, np.full(len(xs), 1e-5), xs,
                     [max(self.res_w[x], floor) for x in xs], color=PURPLE, width=14)
            if 2 in self.res_w:
                self.readout(w2=self.res_w[2])
            odd = [self.res_w[k] for k in (1, 3, 5, 7) if k in self.res_w]
            if len(odd) == 4:
                self.readout(odd=float(np.mean(odd)))
        else:
            self.done = True

    def story(self, p):
        g = CRC_POLYS[p.poly]
        r = g.bit_length() - 1
        s = ("<p>Because a CRC is linear, an error pattern slips through exactly when the generator "
             "polynomial divides it. Some patterns can never be divisible, and the CRC "
             "<b>guarantees</b> to catch them:</p>"
             f"<p>• every burst no longer than r = {v(r, 'd')} bits (green band: a burst of "
             "length ≤ r is x^i times a polynomial of degree &lt; r, never a multiple of g);<br>"
             "• every odd number of errors, if g has the factor x + 1 (an even number of terms);<br>"
             "• every double error, if g has a factor of high enough order.</p>"
             f"<p>Everything else is caught <b>statistically</b>: about 2⁻ʳ = "
             f"{v(f'{2.0 ** -r:.1e}')} of long bursts and random garbage gets through, and bursts "
             "of exactly r + 1 twice as often.</p>")
        if g == 0x101:
            s += ("<p>" + bad("x⁸ + 1 = (x + 1)⁸ is a terrible choice: two errors exactly 8, 16, "
                              "24 … bits apart are divisible by it.") + " Good generators are "
                  "chosen by exhaustive search (Koopman's tables).</p>")
        return "<h3>Guarantees and odds</h3>" + s + keybox(
            "A CRC-32 lets one corrupted frame in about four billion through, and none with a "
            "burst of 32 bits or fewer.")


# =============================================================================== 5. GF(2^m)
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup_poly(p):
    return _sup(gf.poly_str(p))


def _sup(txt):
    out, up = [], False
    for ch in txt:
        if ch == "^":
            up = True
            continue
        if up and ch.isdigit():
            out.append(ch.translate(SUP))
            continue
        up = False
        out.append(ch)
    return "".join(out)


def apow(i):
    return "α" + str(int(i)).translate(SUP)


class GaloisPlayground(Experiment):
    title = "The GF(2ᵐ) playground"
    blurb = "Add with XOR, multiply by adding angles: the arithmetic inside Reed–Solomon and AES."
    book = "sec:ch14:gf"
    controls = [
        IntSlider("m", "Field GF(2ᵐ): m", 2, 8, 3),
        IntSlider("a", "a = α^i, i", 0, 254, 2),
        IntSlider("b", "b = α^j, j", 0, 254, 5),
        Choice("op", "Operation", ["a + b", "a × b", "a ÷ b"], "a + b"),
        Toggle("bylog", "Order the table by logarithm", False),
    ]
    plots = [
        Canvas("ring", "The non-zero elements on a circle: α^i at angle i·360°/(2ᵐ − 1)",
               xlim=(-1.45, 1.45), ylim=(-1.45, 1.45), aspect=True),
        Canvas("tab", "Log / antilog table", xlim=(0, 10), ylim=(0, 17.5)),
        ImagePlot("mul", "Multiplication table (colour = product)", x="b", y="a", aspect=True),
    ]
    layout = [["ring", "tab"], ["ring", "mul"]]
    col_stretch = [3, 2]
    row_stretch = [1, 1]
    readouts = [
        Readout("A", "a", "", None),
        Readout("B", "b", "", None),
        Readout("R", "Result", "", None),
        Readout("prim", "Primitive polynomial", "", None),
    ]
    challenges = [
        Challenge("In GF(8), find two different elements whose sum is 1 (α⁰).",
                  lambda s: s.p.m == 3 and s.p.op == "a + b" and s.exp.res == 1
                  and s.exp.ai != s.exp.bi),
        Challenge("Find the inverse of α⁵ in GF(16): a × b = 1 with a = α⁵.",
                  lambda s: s.p.m == 4 and s.p.op == "a × b" and s.exp.ai == 5 and s.exp.res == 1),
        Challenge("In GF(256), find the power i with α^i = 1 + α (Zech's logarithm of 1).",
                  lambda s: s.p.m == 8 and s.p.op == "a + b" and {s.exp.ai, s.exp.bi} == {0, 1}),
    ]

    def setup(self):
        self.res, self.ai, self.bi = 0, 0, 0

    def update(self, p):
        F = gf.GF(p.m)
        N = F.n
        ai, bi = p.a % N, p.b % N
        self.ai, self.bi = ai, bi
        A, B = F.alpha(ai), F.alpha(bi)
        res = {"a + b": F.add(A, B), "a × b": F.mul(A, B), "a ÷ b": F.div(A, B)}[p.op]
        self.res = res
        ang = lambda i: np.pi / 2 - 2 * np.pi * i / N
        pr = self.plot("ring")
        th = np.linspace(0, 2 * np.pi, 200)
        pr.line("circ", np.cos(th), np.sin(th), color=GRAY, width=1.0)
        idx = np.arange(N)
        pr.scatter("all", np.cos(ang(idx)), np.sin(ang(idx)), color=GRAY, size=8 if N < 64 else 4)
        if N <= 31:
            for i in range(N):
                x, y = 1.2 * np.cos(ang(i)), 1.2 * np.sin(ang(i))
                pr.text(f"l{i}", x, y, apow(i) + ("\n" + format(F.alpha(i), f"0{p.m}b") if N <= 15 else ""),
                        color=pr.theme.text, anchor=(0.5, 0.5), size=8.5 if N <= 15 else 7.5)
        ri = None if res == 0 else int(F.log[res])
        for key, i, col, lab in (("pa", ai, NAVY, "a"), ("pb", bi, GREEN, "b"), ("pr", ri, RED, "result")):
            if i is None:
                continue
            x, y = np.cos(ang(i)), np.sin(ang(i))
            pr.line(key + "ray", [0, x], [0, y], color=col, width=2.4)
            pr.scatter(key, [x], [y], color=col, size=15, z=5)
        if p.op in ("a × b", "a ÷ b") and ri is not None:
            sgn = 1 if p.op == "a × b" else -1
            t = np.linspace(0, 1, 60)
            arc = ang(ai) - sgn * 2 * np.pi * bi / N * t
            pr.line("arc", 0.45 * np.cos(arc), 0.45 * np.sin(arc), color=RED, width=2.0, style="--")
        if res == 0:
            pr.scatter("zero", [0], [0], color=RED, size=16, z=5)
            pr.text("zt", 0.06, -0.06, "0", color=RED, anchor=(0, 0), size=11, bold=True)
        # log / antilog table
        pt = self.plot("tab")
        rows = list(range(min(N, 16)))
        for k, i in enumerate(rows):
            y = 16.2 - k
            hl = RED if i == ri else (NAVY if i == ai else (GREEN if i == bi else None))
            pt.text(f"p{k}", 0.3, y, apow(i), color=hl or pt.theme.text, anchor=(0, 0.5), size=8.5,
                    bold=hl is not None)
            pt.text(f"v{k}", 2.6, y, format(F.alpha(i), f"0{p.m}b"), color=hl or pt.theme.text,
                    anchor=(0, 0.5), size=8.5, bold=hl is not None)
            pt.text(f"d{k}", 6.6, y, str(F.alpha(i)), color=hl or GRAY, anchor=(0, 0.5), size=8.5)
        pt.text("h1", 0.3, 17.2, "power", color=GRAY, anchor=(0, 0.5), size=8.5, bold=True)
        pt.text("h2", 2.6, 17.2, "bits", color=GRAY, anchor=(0, 0.5), size=8.5, bold=True)
        pt.text("h3", 6.6, 17.2, "integer", color=GRAY, anchor=(0, 0.5), size=8.5, bold=True)
        pt.set_title("Log / antilog table" + (" (first 16 rows)" if N > 16 else ""))
        # multiplication table
        q = F.q
        order = np.r_[0, F.exp[:N]] if p.bylog else np.arange(q)
        M = F.vmul(order[:, None], order[None, :])
        lg = np.where(M > 0, F.log[np.maximum(M, 1)], -1).astype(float)
        pm = self.plot("mul")
        pm.xlim = pm.ylim = (0, q)
        pm.vb.setRange(xRange=(0, q), yRange=(0, q), padding=0)
        pm.image("img", (lg if p.bylog else M.astype(float))[::-1], x=(0, q), y=(0, q), cmap="heat")
        pm.set_title("a × b, ordered by " + ("logarithm" if p.bylog else "value"))
        self.readout(A=f"{apow(ai)} = {format(A, f'0{p.m}b')}", B=f"{apow(bi)} = {format(B, f'0{p.m}b')}",
                     R=("0" if res == 0 else f"{apow(ri)} = {format(res, f'0{p.m}b')}"),
                     prim=sup_poly(F.prim))

    def story(self, p):
        F = gf.GF(p.m)
        N = F.n
        s = (f"<p>GF({F.q}) has {F.q} elements: zero and the {N} powers of a primitive element α, "
             f"a root of the primitive polynomial {v(sup_poly(F.prim))}. Each element is also an "
             f"{p.m}-bit vector (its coefficients as a polynomial in α).</p>")
        if p.op == "a + b":
            s += ("<p><b>Addition</b> is bitwise XOR of the vectors: no carries, and every element "
                  "is its own negative (a + a = 0). On the circle it looks random: addition "
                  "scrambles the powers, which is exactly what makes these fields useful.</p>")
        else:
            s += ("<p><b>Multiplication</b> adds the exponents modulo "
                  f"{N}: on the circle it is a rotation (red arc). Division rotates backwards. So a "
                  "hardware multiplier is two table look-ups (log), one addition and one look-up "
                  "(antilog).</p>")
        if p.bylog:
            s += ("<p>Ordered by logarithm, the multiplication table turns into neat diagonal "
                  "stripes: the field's multiplicative group is cyclic.</p>")
        else:
            s += "<p>Tick <i>Order the table by logarithm</i> to see the hidden structure.</p>"
        return "<h3>Arithmetic on bytes</h3>" + s + keybox(
            "GF(256) with 0x11D is the arithmetic of DVB, QR codes, CDs and RAID-6; AES uses the "
            "same field with a different polynomial (0x11B).")


# =============================================================================== 6. Reed–Solomon
RS_CODES = {"RS(255,223), t = 16 (CCSDS)": (255, 223, 8), "RS(255,239), t = 8": (255, 239, 8),
            "RS(204,188), t = 8 (DVB)": (204, 188, 8), "RS(255,251), t = 2": (255, 251, 8),
            "RS(15,11), t = 2 over GF(16)": (15, 11, 4)}


@lru_cache(maxsize=8)
def rs_code(name):
    n, k, m = RS_CODES[name]
    return gf.ReedSolomon(n, k, gf.GF(m))


def rs_trial(rs, nerr, neras, rng, burst=False):
    """One random word with nerr errors and neras erasures: (status, positions...)."""
    n, F = rs.n, rs.F
    msg = rng.integers(0, F.q, rs.k)
    c = rs.encode(msg)
    tot = min(nerr + neras, n)
    if burst:
        s0 = int(rng.integers(0, n - tot + 1))
        pos = np.arange(s0, s0 + tot)
        rng.shuffle(pos)
    else:
        pos = rng.choice(n, tot, replace=False)
    r = c.copy()
    r[pos] ^= rng.integers(1, F.q, tot)
    eras = sorted(int(x) for x in pos[nerr:])
    mh, nc = rs.decode(r, erasures=eras)
    status = "ok" if (nc >= 0 and np.array_equal(mh, msg)) else ("fail" if nc < 0 else "mis")
    return status, pos[:nerr], np.array(eras, int), c, r


class ReedSolomon(Experiment):
    title = "Reed–Solomon: errors, erasures, the cliff"
    blurb = "Corrupt symbols, mark some as erased, and find the exact edge of the decoder's power."
    book = "sec:ch14:rs"
    heavy = True
    controls = [
        Choice("code", "Code", list(RS_CODES), "RS(255,223), t = 16 (CCSDS)", style="menu"),
        IntSlider("nerr", "Symbol errors (unknown positions)", 0, 40, 10),
        IntSlider("neras", "Erasures (positions known)", 0, 40, 0),
        Toggle("burst", "All in one burst", False),
        Button("again", "New random word"),
    ]
    plots = [
        Canvas("word", "One codeword, symbol by symbol", xlim=(-0.8, 17.3), ylim=(-1.6, 15.6)),
        Plot("cliff", "The cliff: symbol error rate after decoding",
             x="channel symbol error rate", y="decoded symbol error rate", logx=True, logy=True,
             xlim=(1e-4, 0.3), ylim=(1e-15, 1), legend=None),
    ]
    layout = [["word", "cliff"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("budget", "2·errors + erasures vs n − k", "", None),
        Readout("this", "This word", "", None),
        Readout("succ", "Success over 60 words", "%", "%"),
        Readout("mis", "Silent miscorrections", "", "int"),
    ]
    challenges = [
        Challenge("Spend the whole budget: decode with 2·errors + erasures exactly n − k, using "
                  "both errors and erasures.",
                  lambda s: s.exp.spent == s.exp.nk and s.p.nerr > 0 and s.p.neras > 0
                  and s.r.this == "decoded"),
        Challenge("Repair more corrupted symbols than t by telling the decoder where they are.",
                  lambda s: s.p.nerr + s.p.neras > s.exp.t and s.r.this == "decoded"),
        Challenge("Catch the decoder miscorrecting: a wrong codeword, delivered without warning.",
                  lambda s: s.r.mis is not None and s.r.mis > 0,
                  hint="Small t, a few more errors than it can handle."),
    ]

    def setup(self):
        self.seed = 20
        self.spent, self.nk, self.t = 0, 32, 16

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rs = rs_code(p.code)
        n, k = rs.n, rs.k
        nerr, neras = min(p.nerr, n), min(p.neras, n - min(p.nerr, n))
        self.spent, self.nk, self.t = 2 * nerr + neras, n - k, rs.t
        rng = np.random.default_rng(self.seed)
        status, epos, eras, c, r = rs_trial(rs, nerr, neras, rng, p.burst)
        cols = 17 if n > 15 else 5
        rows = int(np.ceil(n / cols))
        pw = self.plot("word")
        pw.set_xlim(-0.8, cols + 0.3)
        pw.set_ylim(-1.6, rows + 0.6)
        idx = np.arange(n)
        x = idx % cols + 0.5
        y = rows - 1 - idx // cols + 0.5
        colors = [NAVY if i < k else TEAL for i in idx]
        colors = np.array(colors, dtype=object)
        okcol = ORANGE if status == "ok" else RED
        colors[epos] = okcol
        colors[eras] = PURPLE
        labels = None
        if n <= 15:
            labels = [format(int(vv), "x") for vv in r]
        cells(pw, "sym", x, y, list(colors), w=0.86, h=0.86, alpha=0.85,
              labels=labels, text_color=ink(pw), size=10)
        legend = [(NAVY, "data"), (TEAL, "parity"),
                  (okcol, "error (fixed)" if status == "ok" else "error (not fixed)"),
                  (PURPLE, "erasure")]
        for i, (cc, lab) in enumerate(legend):
            cells(pw, f"lg{i}", [0.5 + 4.2 * i], [-1.0], [cc], w=0.8, h=0.8)
            pw.text(f"lgt{i}", 1.1 + 4.2 * i, -1.0, lab, anchor=(0, 0.5), size=8.5)
        pw.set_title(p.code.split("), ")[0] + "): " + {"ok": "decoded correctly",
                                                          "fail": "failure detected (sent upstairs as an erasure)",
                                                          "mis": "MISCORRECTED"}[status])
        # the cliff
        pc = self.plot("cliff")
        ps = np.logspace(-4, np.log10(0.3), 150)
        for tt, col in ((1, GRAY), (2, PURPLE), (4, GREEN), (8, NAVY), (16, ORANGE)):
            if 2 * tt < n:
                yv = bc.rs_ser_out(n, tt, ps)
                pc.line(f"t{tt}", ps, np.maximum(yv, 1e-300),
                        color=RED if tt == rs.t else col, width=2.8 if tt == rs.t else 1.2)
                xl = float(np.interp(-13, np.log10(np.maximum(yv, 1e-300)), np.log10(ps)))
                pc.text(f"tl{tt}", 10 ** xl * 1.15, 1e-13, f"t = {tt}", anchor=(0, 0.5), size=8.5,
                        color=RED if tt == rs.t else col, bold=tt == rs.t)
        pc.line("unc", ps, ps, color=GRAY, style=":", width=1.0)
        pc.text("unct", 1.3e-4, 2.5e-4, "no code", color=GRAY, anchor=(0, 1), size=8.5)
        pin = max((nerr + neras) / n, 1e-4)
        pc.vline("me", pin, color=RED, style="--", label="this word", label_pos=0.5)
        self.mis_count = 0
        self.readout(budget=f"{2 * nerr + neras} / {n - k}",
                     this={"ok": "decoded", "fail": "failure detected", "mis": "miscorrected"}[status],
                     succ=None, mis=None)

    def background(self, p):
        rs = rs_code(p.code)
        n = rs.n
        nerr, neras = min(p.nerr, n), min(p.neras, n - min(p.nerr, n))
        rng = np.random.default_rng(1000 + self.seed)
        W = 12 if self.quick else 60
        ok = mis = 0
        for i in range(W):
            stt = rs_trial(rs, nerr, neras, rng, p.burst)[0]
            ok += stt == "ok"
            mis += stt == "mis"
            if i % 6 == 5 or i == W - 1:
                yield dict(ok=ok, mis=mis, n=i + 1)

    def progress(self, p, it_):
        self.readout(succ=it_["ok"] / it_["n"], mis=it_["mis"])

    def story(self, p):
        rs = rs_code(p.code)
        r = self.r
        n, k = rs.n, rs.k
        s = (f"<p>Reed–Solomon codes work on <b>symbols</b> of {rs.F.m} bits, not on bits: "
             f"{v(k, 'd')} data symbols plus {v(n - k, 'd')} parity symbols. A symbol is either "
             "right or wrong, however many of its bits were hit, which makes RS codes superb "
             "against bursts.</p>"
             f"<p>The decoder can fix ν unknown errors and e erasures (symbols flagged as "
             f"unreliable) whenever 2ν + e ≤ n − k = {v(n - k, 'd')}: an erasure costs half as "
             f"much as an error, because its position is already known. Budget used: "
             f"{v(r.get('budget', ''))}.</p>")
        if self.spent > self.nk:
            s += ("<p>Beyond the budget the decoder almost always <b>notices</b> that it cannot "
                  "decode (the error-locator polynomial has the wrong number of roots) and says "
                  "so. ")
            if rs.t <= 2:
                s += bad("With t this small, though, a random pattern often lands within t of "
                         "another codeword, and the decoder 'succeeds' with the wrong data.")
            s += "</p>"
        s += ("<p>Right: the cliff. Below the threshold the output error rate falls like the "
              "(t+1)-th power of the input; above it the code is useless. RS(255,223) turns 1 % "
              "symbol errors into about 10⁻¹⁰.</p>")
        return "<h3>The MDS code</h3>" + s + keybox(
            "Reed–Solomon codes meet the Singleton bound: n − k parity symbols repair n − k "
            "erasures or (n − k)/2 errors, the best any code can do.")


# =============================================================================== 7. interleaving
@lru_cache(maxsize=1)
def text_image(h=96, w=223):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yy, xx = np.mgrid[0:h, 0:w]
    img = 200 + 40 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    f = plt.figure(figsize=(w / 50, h / 50), dpi=50)
    a = f.add_axes([0, 0, 1, 1])
    a.set_axis_off()
    a.text(0.5, 0.66, "Reed-Solomon", ha="center", va="center", fontsize=24, family="serif")
    a.text(0.5, 0.24, "1960", ha="center", va="center", fontsize=26, family="serif",
           fontweight="bold")
    f.canvas.draw()
    txt = np.asarray(f.canvas.buffer_rgba())[:h, :w, :3].mean(axis=2)
    plt.close(f)
    return np.clip(np.where(txt < 128, txt * 0.3, img), 0, 255).astype(np.int64)


@lru_cache(maxsize=16)
def tx_order(D, R=96):
    """(row, column) of every transmitted byte: groups of D codewords sent column by column."""
    order = []
    for i in range(0, R, D):
        g = np.arange(i, min(i + D, R))
        rows = np.tile(g, 255)
        cols = np.repeat(np.arange(255), len(g))
        order.append(np.column_stack([rows, cols]))
    return np.concatenate(order)


IL_DEPTHS = ["1 (none)", "2", "4", "8", "16", "32", "48", "96"]
RS_IL = gf.ReedSolomon(255, 223, gf.GF(8))


@lru_cache(maxsize=1)
def image_codewords():
    img = text_image()
    return img, np.stack([RS_IL.encode(row) for row in img])


class Interleaving(Experiment):
    title = "Interleaving against bursts"
    blurb = "A picture protected row by row with RS(255,223), hit by long bursts on the way."
    book = "sec:ch14:concat"
    controls = [
        Slider("blen", "Burst length", 20, 2000, 300, step=10, unit="bytes"),
        IntSlider("nb", "Number of bursts", 1, 5, 3),
        Choice("depth", "Interleaver depth (codewords)", IL_DEPTHS, "1 (none)", style="menu"),
        Button("again", "Move the bursts"),
    ]
    plots = [
        ImagePlot("rx", "Received picture (data part of each codeword)", x="byte", y="codeword"),
        ImagePlot("dec", "After Reed–Solomon decoding", x="byte", y="codeword"),
        Plot("cnt", "Symbol errors in each codeword (t = 16 can be corrected)", x="codeword",
             y="symbol errors", xlim=(-1, 97), ylim=(0, 60), legend=None),
    ]
    layout = [["rx", "dec"], ["cnt", "cnt"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("mx", "Worst codeword", "errors", "int", good=lambda x: x <= 16),
        Readout("fail", "Codewords lost", "", "int", good=lambda x: x == 0),
        Readout("guar", "Guaranteed burst (depth × t)", "bytes", "int"),
        Readout("lat", "Interleaver span", "bytes", "int"),
    ]
    challenges = [
        Challenge("Recover the whole picture through three bursts of at least 300 bytes.",
                  lambda s: s.p.nb >= 3 and s.p.blen >= 300 and s.r.fail == 0),
        Challenge("Survive a single burst of 1000 bytes or more with the shallowest interleaver "
                  "that works.", lambda s: s.p.blen >= 1000 and s.r.fail == 0
                  and int(s.p.depth.split()[0]) <= 96 and s.exp.min_ok),
        Challenge("Break the picture even at full depth (96).",
                  lambda s: s.p.depth.startswith("96") and s.r.fail > 0),
    ]

    def setup(self):
        self.seed = 3
        self.min_ok = False
        image_codewords()

    def on_again(self, p):
        self.seed += 1

    def errors_for(self, p, D, seed):
        img, code = image_codewords()
        order = tx_order(D, code.shape[0])
        L = len(order)
        rng = np.random.default_rng(seed)
        hit = np.zeros(L, bool)
        blen = int(p.blen)
        starts = np.sort(rng.integers(0, L - blen, p.nb))
        for s0 in starts:
            hit[s0:s0 + blen] = True
        E = np.zeros(code.shape, bool)
        E[order[hit, 0], order[hit, 1]] = True
        return E

    def update(self, p):
        img, code = image_codewords()
        D = int(p.depth.split()[0])
        E = self.errors_for(p, D, self.seed)
        rng = np.random.default_rng(self.seed + 99)
        noise = rng.integers(0, 256, code.shape)
        rx = np.where(E, code ^ np.where(noise == 0, 1, noise), code)
        cnt = E.sum(axis=1)
        dec = rx[:, :223].copy()
        lost = np.zeros(len(cnt), bool)
        for i in np.flatnonzero(cnt > 0):
            if cnt[i] <= RS_IL.t:
                dec[i] = img[i]                       # guaranteed by the bounded-distance decoder
            else:                                     # beyond t: the decoder declares failure
                lost[i] = True                        # (miscorrection odds for t = 16: ~10⁻¹³)
        # is a shallower depth also enough? (for the challenge)
        self.min_ok = False
        di = IL_DEPTHS.index(p.depth)
        if not lost.any():
            if di == 0:
                self.min_ok = True
            else:
                Dp = int(IL_DEPTHS[di - 1].split()[0])
                self.min_ok = self.errors_for(p, Dp, self.seed).sum(axis=1).max() > RS_IL.t
        cm = ((0, "#000000"), (1, "#FFFFFF"))
        pr = self.plot("rx")
        pr.set_xlim(0, 223)
        pr.set_ylim(0, 96)
        pr.image("img", rx[::-1, :223] / 255.0, x=(0, 223), y=(0, 96), cmap=cm, levels=(0, 1))
        pd = self.plot("dec")
        pd.set_xlim(0, 223)
        pd.set_ylim(0, 96)
        pd.image("img", dec[::-1] / 255.0, x=(0, 223), y=(0, 96), cmap=cm, levels=(0, 1))
        pc = self.plot("cnt")
        cols = [RED if l_ else (ORANGE if c_ > 0 else GREEN) for l_, c_ in zip(lost, cnt)]
        pc.bars("b", np.arange(96), np.maximum(cnt, 0.3), width=0.8, colors=cols)
        pc.hline("t", RS_IL.t, color=RED, style="--", label="t = 16", label_pos=0.02)
        pc.set_ylim(0, max(60, int(cnt.max() * 1.15) + 1))
        self.readout(mx=int(cnt.max()), fail=int(lost.sum()), guar=D * RS_IL.t, lat=D * 255)

    def story(self, p):
        D = int(p.depth.split()[0])
        r = self.r
        s = (f"<p>Each of the 96 rows of the picture is one RS(255,223) codeword (223 bytes of "
             f"picture + 32 parity bytes), able to correct 16 wrong bytes. The channel destroys "
             f"{v(p.nb, 'd')} burst{'s' if p.nb > 1 else ''} of {v(p.blen, '.0f')} consecutive "
             "bytes.</p>")
        if D == 1:
            s += ("<p>Sent row after row, a burst lands inside one or two codewords, hundreds of "
                  "errors where 16 can be fixed: those rows are lost (red bars), however few "
                  "errors the channel makes on average.</p>")
        else:
            s += (f"<p>With depth {v(D, 'd')} the transmitter writes {D} codewords as rows of an "
                  "array and sends it column by column, so consecutive channel bytes belong to "
                  f"different codewords: a burst of B bytes puts only about B/{D} errors in each. "
                  f"The worst codeword now has {v(r.get('mx', 0), 'd')} errors. Any single burst "
                  f"up to depth × t = {v(D * 16, 'd')} bytes is guaranteed harmless.</p>")
        s += (f"<p>The price is delay and memory: {v(D * 255, 'd')} bytes must be stored before "
              "the first codeword is complete. (Rows with more than 16 errors are reported as "
              "decoder failures and left as received.)</p>")
        return "<h3>Spread the damage</h3>" + s + keybox(
            "Interleaving turns a code's random-error power into burst protection: the CD's "
            "cross-interleaved RS code rides over 4000-bit scratches this way.")


# =============================================================================== 8. coding gain
def _bin(n, k, t):
    return lambda e: bc.block_hard_ber(n, t, ft.qfunc(np.sqrt(2 * k / n * 10 ** (np.asarray(e) / 10))))


def _rs(e):
    p = ft.qfunc(np.sqrt(2 * 223 / 255 * 10 ** (np.asarray(e) / 10)))
    return bc.rs_ber(255, 16, 8, 1 - (1 - p) ** 8)


GAIN_CODES = {
    "Hamming (7,4), t = 1": (4 / 7, _bin(7, 4, 1), GRAY),
    "Hamming (15,11), t = 1": (11 / 15, _bin(15, 11, 1), TEAL),
    "Hamming (31,26), t = 1": (26 / 31, _bin(31, 26, 1), GOLD),
    "Golay (23,12), t = 3": (12 / 23, _bin(23, 12, 3), PURPLE),
    "BCH (63,45), t = 3": (45 / 63, _bin(63, 45, 3), GREEN),
    "BCH (127,64), t = 10": (64 / 127, _bin(127, 64, 10), NAVY),
    "BCH (255,239), t = 2": (239 / 255, _bin(255, 239, 2), ORANGE),
    "RS (255,223), t = 16 bytes": (223 / 255, _rs, RED),
}


def required(fn, target):
    g = lambda e: np.log10(max(float(fn(e)), 1e-300)) - np.log10(target)
    try:
        return float(brentq(g, -5, 25))
    except ValueError:
        return None


class CodingGain(Experiment):
    title = "Coding gain at your target"
    blurb = "The whole family of hard-decision block codes against uncoded BPSK."
    book = "sec:ch14:why"
    controls = [
        LogSlider("target", "Target bit error rate", 1e-10, 1e-2, 1e-5),
        Choice("code", "Highlight", list(GAIN_CODES), "BCH (127,64), t = 10", style="menu"),
    ]
    plots = [
        BERPlot("ber", "Bit error rate (hard decisions, bounded-distance decoding)",
                x="Eb/N0 (dB)", xlim=(2, 14), ylim=(1e-11, 0.5), legend="bl"),
        Plot("gain", "Coding gain at the target", x="coding gain (dB)", y="",
             xlim=(-2, 6.5), ylim=(-0.7, 7.7), legend=None, grid=True),
    ]
    layout = [["ber", "gain"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("rate", "Code rate", "", ".3f"),
        Readout("unc", "Uncoded needs", "dB", ".2f"),
        Readout("need", "Coded needs", "dB", ".2f"),
        Readout("gain", "Coding gain", "dB", ".2f", good=lambda x: x > 0),
    ]
    challenges = [
        Challenge("Find a target error rate at which Hamming (7,4) costs more than it gains.",
                  lambda s: s.p.code.startswith("Hamming (7,4)") and s.r.gain < 0),
        Challenge("Find a code and target with more than 5 dB of coding gain.",
                  lambda s: s.r.gain > 5.0),
        Challenge("Find where Reed–Solomon (255,223) overtakes BCH (127,64).",
                  lambda s: s.p.code.startswith("RS") and s.exp.gains.get("BCH (127,64), t = 10", 9)
                  < s.r.gain),
    ]

    def setup(self):
        self.gains = {}

    def update(self, p):
        eb = np.linspace(2, 14, 241)
        pb = self.plot("ber")
        unc = lambda e: ft.qfunc(np.sqrt(2 * 10 ** (np.asarray(e) / 10)))
        pb.theory("unc", eb, unc(eb), color=GRAY, width=2.6, style="--", name="uncoded BPSK")
        for i, (name, (R, fn, col)) in enumerate(GAIN_CODES.items()):
            mine = name == p.code
            pb.theory(f"c{i}", eb, np.maximum(fn(eb), 1e-30), color=col if mine else GRAY,
                      width=3.0 if mine else 1.0, name=name if mine else None)
        pb.hline("tg", p.target, color=ORANGE, style="--")
        u = required(unc, p.target)
        gains = {}
        for name, (R, fn, col) in GAIN_CODES.items():
            x = required(fn, p.target)
            gains[name] = None if x is None or u is None else u - x
        self.gains = gains
        names = list(GAIN_CODES)
        pg = self.plot("gain")
        y = np.arange(len(names))[::-1]
        vals = np.array([gains[n] if gains[n] is not None else 0 for n in names])
        for i, (n, yy) in enumerate(zip(names, y)):
            col = GAIN_CODES[n][2] if n == p.code else GRAY
            pg.line(f"g{i}", [0, vals[i]], [yy, yy], color=col, width=14 if n == p.code else 10)
            pg.text(f"t{i}", -1.9, yy + 0.38, n, anchor=(0, 0.5), size=8.5,
                    color=pg.theme.text if n == p.code else GRAY, bold=n == p.code, fill=True)
        pg.vline("z", 0, color=pg.theme.text, style="-", width=1.0)
        pg.set_yticks([])
        x = required(GAIN_CODES[p.code][1], p.target)
        if x is not None:
            pb.vline("need", x, color=GAIN_CODES[p.code][2], style=":", width=1.2)
        if u is not None:
            pb.vline("uncv", u, color=pb.theme.text, style=":", width=1.2)
        self.readout(rate=GAIN_CODES[p.code][0], unc=u, need=x, gain=gains[p.code])

    def story(self, p):
        r = self.r
        g = r.get("gain")
        s = (f"<p>Coding gain is the reduction in Eb/N0 a code buys at a given error rate, "
             f"counting the energy spent on parity: at rate {v(r.get('rate', 1), '.2f')} each "
             f"coded bit carries only {v(r.get('rate', 1), '.2f')} of the energy of an uncoded "
             "one, so every channel bit is noisier and the code must first earn that back.</p>")
        if g is not None:
            s += (f"<p>{p.code.rsplit(', t', 1)[0]} needs {v(r.get('need', 0), '.1f', 'dB')} for a bit "
                  f"error rate of {v(f'{p.target:.0e}')}, uncoded BPSK {v(r.get('unc', 0), '.1f', 'dB')}: ")
            s += (good(f"a gain of {g:.2f} dB.") if g > 0 else
                  bad(f"a loss of {-g:.2f} dB: at this error rate the parity costs more than the "
                      "correction gives back."))
            s += "</p>"
        s += ("<p>Gains grow as the target falls: the error rate of a t-error-correcting code "
              "falls as the (t+1)-th power of the channel's. Long, powerful codes (BCH 127, "
              "RS 255) win at low targets; all of them are hard-decision and leave about 2 dB on "
              "the table compared with soft decoding.</p>")
        return "<h3>What a code is worth</h3>" + s + keybox(
            "Quote a coding gain only with its target error rate: Hamming (7,4) gains 0.4 dB at "
            "10⁻⁵ and loses at 10⁻².")


# =============================================================================== the lab
LAB = st.Lab(20, "Block Codes: Hamming to Reed–Solomon", chapter=14,
             chapter_title="Classical Codes",
             experiments=[HammingVenn, Secded, CrcCalculator, CrcDetection, GaloisPlayground,
                          ReedSolomon, Interleaving, CodingGain])

if __name__ == "__main__":
    st.run(LAB)
