"""Lab 34 · The Telegraph and the Birth of Signalling   (Chapter 1)

Run it:      python labs/lab34_telegraph_history.py
Self-test:   python labs/lab34_telegraph_history.py --selftest

The electric telegraph of the 1840s already posed every question this book answers: how to
represent symbols and how long each should take (Morse is a variable-length source code a
century before Huffman), how fast a line can be keyed before pulses smear together (Kelvin's
law of squares is the first theory of intersymbol interference), and what happens when a
weak signal is relayed many times (regenerate, don't amplify: why everything went digital).
Seven experiments replay those questions with the models behind Chapter 1's figures, and let
you change the history: type your own message, lengthen the cable, add repeaters.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import numpy as np
from scipy.signal import fftconvolve, welch

from commlib import infotheory as it
from commlib import telegraph as tg
import studio as st
from studio import (Text, Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD)
from studio import v, keybox, good, bad

C_TEL = tg.telegraph_capacity()
MESSAGES = ["SOS", "WHAT HATH GOD WROUGHT", "PARIS PARIS", "CQ CQ DE W1AW",
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]


# =============================================================================== 1. keying
class KeyClicks(Experiment):
    title = "Morse keying and key clicks"
    blurb = "Key a carrier on and off. Sharp edges splatter; shaped edges stay in their lane."
    book = "sec:ch01:morse"
    controls = [
        Text("text", "Message", "SOS", examples=MESSAGES,
                help="Letters and digits are keyed; anything else is skipped"),
        Slider("wpm", "Speed", 5, 40, 20, step=1, unit="wpm",
               help="Words per minute, PARIS standard: one dot unit lasts 1.2/wpm seconds"),
        Slider("rise", "Edge shaping", 0.0, 0.5, 0.25, step=0.01, unit="units",
               help="Raised-cosine rise and fall time, in dot units (0 = a hard on/off key)"),
        Toggle("hard", "Overlay hard keying", True),
        Button("listen", "▶  Listen (700 Hz sidetone)", primary=True),
    ]
    plots = [
        Plot("env", "The keyed carrier envelope", x="time (s)", y="carrier on/off",
             ylim=(-0.15, 1.95), legend="tr", legend_cols=2),
        SpectrumPlot("spec", "Spectrum of the keyed carrier (offset from the carrier)",
                     x="frequency offset (Hz)", y="power (dB)", xlim=(0, 300), ylim=(-100, 5),
                     legend="tr"),
    ]
    layout = [["env"], ["spec"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("unit", "Dot length", "ms", ".0f"),
        Readout("dur", "Message time", "s", ".2f"),
        Readout("bw", "Occupied bandwidth (−40 dB)", "Hz", ".0f", good=lambda x: x < 120,
                help="Two-sided width outside which the spectrum stays 40 dB below its peak"),
        Readout("click", "Worst click ≥ 150 Hz away", "dB", ".0f", good=lambda x: x < -60),
    ]
    challenges = [
        Challenge("Ragchew speed: send at 25 wpm or faster with an occupied bandwidth under 150 Hz.",
                  lambda s: s.p.wpm >= 25 and s.r.bw < 150,
                  hint="Faster keying widens the main lobe; only the edge shaping can claw it back."),
        Challenge("Bury the clicks: everything 150 Hz away or more below −75 dB, at 20 wpm or faster.",
                  lambda s: s.p.wpm >= 20 and s.r.click < -75),
        Challenge("Be a 1900s spark-gap nuisance: hard keying (shaping below 0.03) at 30 wpm, with "
                  "clicks above −45 dB at 150 Hz.",
                  lambda s: s.p.rise < 0.03 and s.p.wpm >= 30 and s.r.click > -45),
    ]

    def make(self, p, reps=1):
        tu = 1.2 / p.wpm
        sps = int(np.clip(np.ceil(900 * tu), 16, 260))     # Nyquist >= 450 Hz
        text = p.text if any(c.upper() in tg.MORSE for c in p.text) else "E"
        env, units = tg.keyed_envelope(" ".join([text] * reps), sps)
        L = int(round(p.rise * sps))
        if L >= 1:
            w = np.hanning(L + 2)[1:-1]
            soft = fftconvolve(env, w / w.sum(), mode="same")
        else:
            soft = env.copy()
        return env, soft, sps, tu, units

    def update(self, p):
        env, soft, sps, tu, units = self.make(p)
        fs = sps / tu
        t = np.arange(len(env)) / fs
        pe = self.plot("env")
        if p.hard:
            pe.line("hard", t, env, color=NAVY, width=0.8, fill=0, fill_alpha=0.22,
                    name="hard keying")
        pe.line("soft", t, soft, color=RED, width=2.0, name=f"edges shaped over {p.rise:.2f} unit")
        pe.set_xlim(0, max(t[-1], 0.5))
        # letter labels above their first mark
        letters, pos, k = [], [], 0
        for wd in p.text.upper().split():
            for ch in wd:
                if ch in tg.MORSE:
                    letters.append(ch)
                    pos.append(k)
                    k += tg.morse_units(tg.MORSE[ch])
            k += 4 if any(c in tg.MORSE for c in wd) else 0
        for i in range(60):
            if i < min(len(letters), 60):
                pe.text(f"L{i}", pos[i] * tu, 1.12, letters[i], color=PURPLE, size=9, bold=True,
                        anchor=(0, 1))
        # spectrum of a long repetition (steady-state keying)
        env4, soft4, *_ = self.make(p, reps=max(4, int(np.ceil(1500 / units))))
        nper = int(min(len(env4), 2 ** int(np.round(np.log2(fs / 2.5)))))   # ~2.5 Hz resolution
        f, Ps = welch(soft4 - soft4.mean(), fs=fs, nperseg=nper, window="hann")
        Ps = 10 * np.log10(Ps + 1e-30)
        ref = Ps.max()
        ps = self.plot("spec")
        if p.hard:
            f2, Ph = welch(env4 - env4.mean(), fs=fs, nperseg=nper, window="hann")
            Ph = 10 * np.log10(Ph + 1e-30)
            ps.line("hard", f2, Ph - Ph.max(), color=NAVY, width=1.0, alpha=0.75,
                    name="hard keying")
        ps.line("soft", f, Ps - ref, color=RED, width=1.8, name="shaped keying")
        ps.vline("unit", 1 / tu, color=GREEN, style=":", label="dot rate 1/T", label_pos=0.9)
        ps.hline("m40", -40, color=GRAY, style=":", width=1.0)
        above = np.nonzero(Ps - ref > -40)[0]
        bw = 2 * f[above[-1]] if len(above) else 0.0
        ps.band("occ", 0, bw / 2, color=GREEN, alpha=0.08)
        click = float(np.max((Ps - ref)[f >= 150]))      # worst click 150 Hz away or more
        self.readout(unit=1000 * tu, dur=len(env) / fs, bw=bw, click=max(click, -150))

    def on_listen(self, p):
        env, soft, sps, tu, units = self.make(p)
        fs_a = 8000
        tt = np.arange(int(len(soft) / sps * tu * fs_a)) / fs_a
        e = np.interp(tt, np.arange(len(soft)) * tu / sps, soft)
        self.play_audio(0.6 * e * np.sin(2 * np.pi * 700 * tt), fs_a, "the Morse sidetone")

    def story(self, p):
        tu = 1.2 / p.wpm
        s = (f"<p>A telegraph key switches a carrier on and off with the International timing: a dot "
             f"is one unit, a dash three, gaps of one, three and seven. At {v(p.wpm, '.0f', 'wpm')} the "
             f"unit is {v(1000 * tu, '.0f', 'ms')} (the word PARIS is exactly 50 units).</p>")
        if p.rise < 0.03:
            s += ("<p>" + bad("Hard keying.") + " Each square edge has a spectrum falling only "
                  "6 dB per octave: the clicks heard on every receiver for kilohertz around. Early "
                  "operators added \"click filters\" to their keys for exactly this reason.</p>")
        else:
            s += (f"<p>Rounding the edges over {v(p.rise)} unit makes the red spectrum fall far faster "
                  f"than the blue hard-keyed one: clicks 150 Hz away are "
                  f"{v(self.r.get('click', 0), '.0f', 'dB')}. Nothing is lost in readability: the "
                  f"ear only needs the dots and dashes.</p>")
        s += ("<p>Speed up and the whole spectrum widens in proportion (the green dot-rate line "
              "moves right): a faster operator needs more bandwidth whatever the edges look like.</p>")
        return "<h3>Every edge has a price</h3>" + s + keybox(
            "Your first pulse-shaping lesson: bandwidth ∝ signalling rate, and the skirts are set by "
            "how smooth the pulse is. Raised-cosine edges here, raised-cosine pulses in Chapter 8.")


# =============================================================================== 2. Morse vs Huffman
class MorseHuffman(Experiment):
    title = "Morse versus Huffman"
    blurb = "Vail counted the type in a printer's case. How close did he get to Shannon?"
    book = "sec:ch01:morse"
    controls = [
        Text("text", "Your text", "WHAT HATH GOD WROUGHT, THE FIRST MESSAGE ON THE "
                "WASHINGTON TO BALTIMORE LINE IN 1844",
                examples=["THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
                          "ZYZZYVA QUIZ JAZZ BUZZ FIZZ",
                          "TEN TEES AND TEN EES",
                          "LE TELEGRAPHE OPTIQUE DE CHAPPE RELIAIT PARIS A LILLE",
                          "DIE NACHRICHT WURDE UEBER DAS KABEL GESENDET"]),
        Choice("stats", "Letter statistics", ["English table", "Your text"],
               help="Design the Huffman code (and measure everything) with the standard English "
                    "frequencies, or with the letters of your own text"),
    ]
    plots = [
        BarPlot("len", "Code length per letter (most frequent first)", x="letter",
                y="length (units or bits)", ylim=(0, 21), legend="tr", legend_cols=4),
        BarPlot("avg", "Average cost per letter", y="units or bits per letter", ylim=(0, 15)),
        Plot("cum", "Cost of your message, letter by letter", x="letters sent",
             y="cumulative bits", legend="tl"),
    ]
    layout = [["len", "len"], ["avg", "cum"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("H", "Entropy H", "bits/letter", ".2f"),
        Readout("huff", "Huffman average", "bits/letter", ".2f"),
        Readout("morse", "Morse average", "units/letter", ".2f"),
        Readout("eff", "Morse efficiency", "%", ".0f", good=lambda x: x > 80,
                help="H / (Morse units × 0.539 bit per unit, Shannon's telegraph capacity)"),
    ]
    challenges = [
        Challenge("Make Morse look bad: a text (with its own statistics) on which Morse is under "
                  "60 % efficient.",
                  lambda s: s.p.stats == "Your text" and s.r.eff < 60,
                  hint="Morse was tuned for English. Fill the box with rare letters (Q, Z, J, X, Y)."),
        Challenge("Beat Baudot: a text whose Morse average is below 5 units per letter.",
                  lambda s: s.r.morse < 5,
                  hint="Only E (4 units) is cheaper than Baudot's five. Short letters: E, T, I."),
        Challenge("Predictable prose: a text of at least 20 letters with entropy under 2 bits per letter "
                  "(own statistics).",
                  lambda s: s.p.stats == "Your text" and s.r.H < 2 and s.r.nlet >= 20),
    ]

    def update(self, p):
        letters = [c for c in p.text.upper() if "A" <= c <= "Z"]
        if p.stats == "Your text" and letters:
            prob = tg.letter_stats(p.text)
        else:
            prob = tg.letter_stats(None)
        code = it.huffman_code(prob)
        H = -sum(q * np.log2(q) for q in prob.values() if q > 0)
        avg_h = sum(prob[k] * len(code[k]) for k in prob)
        avg_m = sum(prob[k] * tg.morse_units(tg.MORSE[k]) for k in prob)
        order = sorted(prob, key=lambda k: -prob[k])
        x = np.arange(len(order))
        pl = self.plot("len")
        pl.bars("m", x - 0.2, [tg.morse_units(tg.MORSE[k]) for k in order], width=0.4, color=NAVY)
        pl.bars("h", x + 0.2, [len(code[k]) for k in order], width=0.4, color=RED)
        pl.line("lm", [-5, -4], [-5, -5], color=NAVY, width=9, name="Morse (dot units incl. gap)")
        pl.line("lh", [-5, -4], [-5, -5], color=RED, width=9, name="Huffman (bits)")
        pl.line("freq", x, [100 * prob[k] for k in order], color=ORANGE, width=2.0,
                name="letter frequency (%)")
        pl.line("baudot", [-1, 40], [5, 5], color=GREEN, width=1.6, style="--", name="Baudot: 5")
        pl.set_xticks([(i, k) for i, k in enumerate(order)])
        pl.set_xlim(-0.7, max(len(order), 6) - 0.3)
        # average cost bars
        vals = [H, avg_h, 5.0, 7.5, avg_m, avg_m * C_TEL]
        names = ["entropy", "Huffman", "Baudot", "ITA2", "Morse", "Morse bits"]
        cols = [GRAY, RED, GREEN, ORANGE, NAVY, PURPLE]
        pa = self.plot("avg")
        pa.bars("b", np.arange(6), vals, width=0.62, colors=cols)
        for i, val in enumerate(vals):
            pa.text(f"t{i}", i, val, f"{val:.2f}", anchor=(0.5, 1.05), bold=True, size=9)
        pa.set_xticks([(i, n) for i, n in enumerate(names)])
        pa.set_xlim(-0.6, 5.6)
        # cumulative cost of the actual message
        pc = self.plot("cum")
        if letters:
            n = np.arange(1, len(letters) + 1)
            cm = np.cumsum([tg.morse_units(tg.MORSE[c]) * C_TEL for c in letters])
            ch = np.cumsum([len(code.get(c, "0" * 12)) for c in letters])
            ce = np.cumsum([-np.log2(prob.get(c, 1e-6)) for c in letters])
            pc.line("m", n, cm, color=NAVY, width=2.0, name="Morse (units × 0.539)")
            pc.line("h", n, ch, color=RED, width=2.0, name="Huffman bits")
            pc.line("e", n, ce, color=GRAY, width=1.6, style="--", name="information −log₂p")
            pc.set_xlim(0, len(letters) + 0.5)
            pc.set_ylim(0, max(cm[-1], ch[-1], ce[-1]) * 1.15 + 1)
        msg_m = sum(tg.morse_units(tg.MORSE[c]) for c in letters)
        self.readout(H=H, huff=avg_h, morse=avg_m, eff=100 * H / (avg_m * C_TEL), nlet=len(letters),
                     msgm=msg_m)
        self._top = order[:3]

    def story(self, p):
        eff = self.r.get("eff", 0)
        s = ("<p>Morse and his assistant Vail gave the commonest letters the shortest codes: E is a "
             "single dot, T a single dash. That is a <b>variable-length source code</b>, a century "
             "before Huffman proved how to build the best one. The blue and red bars rise together "
             "as the letters get rarer.</p>"
             "<p>But a Morse unit is not a free bit: the line must be on for 1 or 3 units, then off. "
             f"Shannon computed the capacity of that constrained channel: {v(C_TEL, '.3f')} bit per "
             f"unit. Measured fairly, Morse uses {v(self.r.get('morse', 0) * C_TEL, '.2f', 'bits')} of "
             f"capacity to carry {v(self.r.get('H', 0), '.2f', 'bits')} of information: "
             f"{v(eff, '.0f', '%')} efficient.</p>")
        if p.stats == "Your text" and eff < 70:
            s += ("<p>" + bad("Morse struggles here") + ": your text's statistics are not English, "
                  "and Vail's code was frozen in 1840s English. Huffman adapts to any source.</p>")
        elif p.stats == "Your text":
            s += "<p>Huffman's code was rebuilt for your text's own letter frequencies.</p>"
        else:
            s += ("<p>With the standard English table the numbers are the chapter's: entropy 4.17, "
                  "Huffman 4.20, Morse 9.07 units, about 85 % efficient. Switch the statistics to "
                  "<i>Your text</i> and type something strange.</p>")
        return "<h3>An 1840s compressor</h3>" + s + keybox(
            "Short codes for frequent symbols (Morse, Huffman, arithmetic coding) and the right "
            "measure of a channel (capacity, not symbol count): both ideas were born on the telegraph.")


# =============================================================================== 3. Kelvin's cable
PRESETS = {"1858": (3000, 5.9, 0.12), "1866": (3000, 2.1, 0.15)}


class KelvinCable(Experiment):
    title = "Kelvin's Atlantic cable"
    blurb = "A 3000 km RC line turns crisp pulses into slow swells. How fast can you key?"
    book = "sec:ch01:cables"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("The cable"),
        Slider("length", "Cable length", 200, 6000, 3000, step=50, unit="km"),
        Slider("R", "Conductor resistance R", 0.5, 8.0, 2.1, step=0.05, unit="Ω/km"),
        Slider("C", "Capacitance C", 0.05, 0.30, 0.15, step=0.005, unit="µF/km"),
        Button("p1858", "Load the 1858 cable"),
        Button("p1866", "Load the 1866 cable"),
        Heading("The operator"),
        Slider("wpm", "Keying speed", 0.5, 30, 4, step=0.5, unit="wpm",
               help="Cable code: about 19 elements per word, each a ±1 pulse"),
        Toggle("curb", "Curb signalling", False,
               help="Thomson's pre-emphasis: end each element with a short pulse of the opposite polarity"),
        Slider("cs", "Curb strength", 0.1, 1.0, 0.5, step=0.05,
               help="Amplitude of the opposite pulse (it lasts the last 30 % of the element)",
               enabled_if=lambda p: p.curb),
    ]
    plots = [
        Plot("rx", "Sent and received (the recorder trace scrolls while playing)",
             x="time (element periods)", y="voltage (sent = ±1)", xlim=(0, 16), ylim=(-1.35, 1.75),
             legend="tl", legend_cols=3),
        Plot("swing", "Reversals: received swing vs element duration", x="element duration T / τ",
             y="received swing / sent", xlim=(0.01, 10), ylim=(1e-4, 2), logx=True, logy=True,
             legend="br"),
        Plot("law", "Law of squares: speed vs length", x="cable length (km)",
             y="words per minute (10 % swing)", xlim=(200, 6000), ylim=(0.05, 500), logx=True,
             logy=True, legend="bl"),
    ]
    layout = [["rx", "rx"], ["swing", "law"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("tau", "Time constant τ = RCℓ²", "s", ".2f"),
        Readout("T", "Element duration", "s", ".2f"),
        Readout("swing", "Reversal swing", "%", ".1f", good=lambda x: x >= 10,
                help="Peak-to-peak swing of alternating elements, as a % of the full deflection of a long mark"),
        Readout("lim", "Speed limit (10 % rule)", "wpm", ".1f"),
    ]
    challenges = [
        Challenge("1866 cable (3000 km, 2.1 Ω/km, 0.15 µF/km): find the fastest keying that keeps a "
                  "10 % swing (within 0.5 wpm).",
                  lambda s: (abs(s.p.length - 3000) < 1 and abs(s.p.R - 2.1) < 0.01
                             and abs(s.p.C - 0.15) < 0.001 and not s.p.curb
                             and s.r.swing >= 10 and s.p.wpm >= s.r.lim - 0.5)),
        Challenge("Halve the cable and key four times faster: 1500 km of 1866 cable at 16 wpm or "
                  "more with a 10 % swing.",
                  lambda s: (s.p.length <= 1500 and abs(s.p.R - 2.1) < 0.01
                             and abs(s.p.C - 0.15) < 0.001 and s.p.wpm >= 16 and s.r.swing >= 10),
                  hint="Law of squares: half the length, a quarter of τ."),
        Challenge("Reach the 1866 service speed (8 wpm over the full 3000 km 1866 cable) with a 10 % "
                  "swing, using curb signalling.",
                  lambda s: (s.p.curb and s.p.length >= 3000 and abs(s.p.R - 2.1) < 0.01
                             and abs(s.p.C - 0.15) < 0.001 and s.p.wpm >= 8 and s.r.swing >= 10),
                  hint="The opposite pulse cuts the slow full-scale deflection, so the galvanometer can be set more sensitive: push the curb strength up."),
    ]
    nhist = 48

    def setup(self):
        self.bits = np.sign(self.rng.standard_normal(4000))
        self.t0 = 64.0                      # start with a history on the line (no switch-on transient)

    def on_p1858(self, p):
        for k, val in zip(("length", "R", "C"), PRESETS["1858"]):
            self.set_control(k, val)

    def on_p1866(self, p):
        for k, val in zip(("length", "R", "C"), PRESETS["1866"]):
            self.set_control(k, val)

    def element_wave(self, p, sub=10):
        """Sub-element levels of one +1 element (with curb if enabled)."""
        w = np.ones(sub)
        if p.curb:
            w[int(round(0.7 * sub)):] = -p.cs
        return w

    def periodic_swing(self, p, r):
        """Steady-state swing of alternating elements through H = exp(-sqrt(j w tau))."""
        M = 400
        w1 = self.element_wave(p)[(np.arange(M // 2) * 10) // (M // 2)]
        per = np.r_[w1, -w1]
        X = np.fft.fft(per)
        k = np.fft.fftfreq(M, 1 / M)                    # harmonic index of the 2T period
        H = np.exp(-np.sqrt(1j * np.abs(k) * np.pi / r))
        H = np.where(k >= 0, H, np.conj(H))
        H[0] = 1.0
        y = np.fft.ifft(X * H).real
        full = w1.mean()                       # received level of a long run of marks (DC)
        return (y.max() - y.min()) / 2 / full

    def update(self, p):
        tau = p.R * p.C * 1e-6 * p.length ** 2
        T = 60.0 / (19 * p.wpm)
        r = T / tau
        self._r = r
        sw = self.periodic_swing(p, r)
        # speed limit: largest wpm with swing >= 10 % (bisection on the element duration)
        lo, hi = 1e-3, 50.0
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            if self.periodic_swing(p, mid) >= 0.1:
                hi = mid
            else:
                lo = mid
        lim = 60.0 / (19 * hi * tau)
        self._lim_ratio = hi
        self._sw = sw
        self.draw(p)
        self.draw_side(p)
        self.readout(tau=tau, T=T, swing=100 * sw, lim=lim)

    def draw_side(self, p):
        r, sw, hi = self._r, self._sw, self._lim_ratio
        # swing curve
        rr = np.logspace(-2, 1, 90)
        if not hasattr(self, "_alt"):
            self._alt = tg.alt_swing(rr)
        ps = self.plot("swing")
        ps.line("plain", rr, self._alt, color=NAVY, width=2.0, name="plain keying")
        if p.curb:
            ps.line("curb", rr, [self.periodic_swing(p, x) for x in rr], color=GREEN, width=2.0,
                    name="with curb")
        ps.hline("ten", 0.1, color=RED, style="--", label="10 %: readable", label_pos=0.03)
        ps.scatter("now", [min(max(r, 0.0101), 9.9)], [max(sw, 1.1e-4)], color=ORANGE, size=14,
                   symbol="d", name="your cable now")
        # law of squares
        L = np.logspace(np.log10(200), np.log10(6000), 60)
        pl = self.plot("law")
        for key, (_, R, C), col in (("1858", PRESETS["1858"], RED), ("1866", PRESETS["1866"], NAVY)):
            pl.line(key, L, tg.wpm_limit(R, C, L), color=col, width=1.6, name=f"{key} design")
        pl.line("you", L, 60.0 / (19 * hi * p.R * p.C * 1e-6 * L ** 2), color=GREEN, width=2.4,
                name="your cable" + (" + curb" if p.curb else ""))
        pl.scatter("svc", [3000], [8], color=PURPLE, size=10, symbol="s", name="1866 in service, ≈8 wpm")
        pl.scatter("now", [p.length], [p.wpm], color=ORANGE, size=14, symbol="d")

    def draw(self, p):
        r = self._r
        sub = 10
        k0 = int(self.t0)
        start = max(0, k0 - self.nhist)
        lv = (self.bits[start:k0 + 17, None] * self.element_wave(p, sub)[None, :]).ravel()
        tt = np.linspace(0, 16, 700)                              # display window (elements)
        t_abs = (tt + (self.t0 - 16 if self.t0 > 16 else 0) - start) * r
        y = tg.cable_response(lv, r / sub, np.maximum(t_abs, 0))
        off = self.t0 - 16 if self.t0 > 16 else 0
        cut = tt <= 16
        pr = self.plot("rx")
        e_lo = int(np.floor(off))
        sent_x = np.arange(e_lo, e_lo + 18) - off
        bsent = self.bits[e_lo:e_lo + 18]
        if p.curb:
            xs = (np.repeat(np.arange(e_lo, e_lo + 18), sub) + np.tile(np.arange(sub) / sub, 18)) - off
            ys = np.repeat(bsent, sub) * np.tile(self.element_wave(p, sub), 18)
            pr.line("sent", np.r_[xs, xs[-1] + 1 / sub], ys, color=GRAY, width=1.2, step=True,
                    name="sent")
        else:
            pr.line("sent", np.r_[sent_x, sent_x[-1] + 1], bsent, color=GRAY, width=1.2, step=True,
                    name="sent")
        yy = y
        pr.line("rx", tt, yy, color=NAVY, width=2.4, name="received")
        pr.hband("read", -0.1, 0.1, color=RED, alpha=0.07)
        pr.hline("z", 0, color=GRAY, style="-", width=0.6)
        g = max(np.nanmax(np.abs(y)), 1e-6)
        if g < 0.25:
            pr.line("zoom", tt, yy / g * 0.9, color=ORANGE, width=1.2, style="--",
                    name=f"received × {0.9 / g:.0f} (galvanometer)")

    def tick(self, p):
        self.t0 += 0.25
        if self.t0 > 3500:
            self.t0 = 64.0
        self.draw(p)
        self.draw_side(p)

    def story(self, p):
        tau = self.r.get("tau", 1)
        sw = self.r.get("swing", 0)
        s = (f"<p>Kelvin (1855) modelled the cable as a resistance and a capacitance per kilometre. "
             f"Every time scale is set by τ = RCℓ², here {v(tau, '.2f', 's')}. Each "
             f"{v(60 / (19 * p.wpm), '.2f', 's')} element is {v(self._r, '.2f')} τ long.</p>")
        if sw >= 10:
            s += (f"<p>{good('Readable.')} Alternate elements still swing the far end by "
                  f"{v(sw, '.1f', '%')} of the sent voltage, enough for Thomson's mirror galvanometer.</p>")
        else:
            s += (f"<p>{bad('Unreadable.')} The tails of earlier pulses pile onto later ones and the "
                  f"swing collapses to {v(sw, '.2f', '%')}: <b>intersymbol interference</b>, 150 years "
                  "before the name. The dashed orange trace is the same signal magnified.</p>")
        s += ("<p>Double the length and RC doubles twice over: τ quadruples, the speed falls four "
              "times. There is no voltage slider, on purpose: raising the voltage raises the swells "
              "and their overlap alike. Whitehouse tried 2000 V in 1858 and destroyed the cable.</p>")
        if p.curb:
            s += ("<p>Curb signalling ends each element with an opposite kick that drains the line: "
                  "a hand-made pre-emphasis filter, the ancestor of every SerDes transmit FIR.</p>")
        return "<h3>Pushing honey through a hose</h3>" + s + keybox(
            "Speed ∝ 1/ℓ². More power does not cure dispersion; slower keying, a better medium, "
            "pulse shaping or equalisation does.")


# =============================================================================== 4. Chappe semaphore
ALPHABETS = {"Chappe (92 signs)": 92, "Admiralty shutters (64)": 64, "Edelcrantz shutters (1024)": 1024}


class Semaphore(Experiment):
    title = "Chappe's semaphore pipeline"
    blurb = "A bucket brigade of towers. The first sign is slow; the rest stream in."
    book = "sec:ch01:chappe"
    animate = True
    fps = 15
    controls = [
        Heading("The line"),
        IntSlider("n", "Stations", 2, 40, 15, help="Paris–Lille had about 15 over 230 km"),
        Slider("t", "Time per sign at each station", 10, 90, 30, step=1, unit="s",
               help="See the sign, set it on your own mechanism, have it confirmed"),
        Toggle("slow", "One slow station (fog, a tired operator)", False),
        Slider("ts", "Slow station's time", 30, 180, 90, step=5, unit="s",
               enabled_if=lambda p: p.slow),
        Heading("The message"),
        Choice("alpha", "Alphabet", list(ALPHABETS), style="menu"),
        Choice("code", "Coding", ["Codebook (2 signs/word)", "Spelled (5 signs/word)"], style="menu"),
        IntSlider("words", "Dispatch length", 5, 60, 30, unit="words"),
    ]
    plots = [
        Plot("st", "Space–time diagram: each line is one sign climbing the chain",
             x="time (minutes)", y="station", legend=None),
        Plot("vs", "Dispatch time vs number of stations", x="stations", y="minutes",
             xlim=(1, 41), legend="tl"),
    ]
    layout = [["st", "vs"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("lat", "Latency (first sign)", "min", ".1f"),
        Readout("rate", "Throughput", "bit/s", ".2f"),
        Readout("disp", "Whole dispatch", "min", ".1f"),
        Readout("wpm", "Words per minute", "", ".2f"),
    ]
    challenges = [
        Challenge("Paris–Strasbourg: deliver a 30-word codebook dispatch over 30 stations in under "
                  "40 minutes.",
                  lambda s: s.p.n >= 30 and s.p.words >= 30 and s.p.code.startswith("Codebook")
                  and s.r.disp < 40),
        Challenge("Show that the slowest station sets the pace: with one slow station at 90 s or "
                  "more, make the throughput fall below 0.08 bit/s.",
                  lambda s: s.p.slow and s.p.ts >= 90 and s.r.rate < 0.08),
        Challenge("Break 1 bit/s with 1790s technology.",
                  lambda s: s.r.rate >= 1.0,
                  hint="Bits per sign = log₂(alphabet size). Which tower had the biggest alphabet?"),
    ]

    def compute(self, p, n_st=None):
        n_st = n_st or p.n
        M = ALPHABETS[p.alpha]
        per_word = 2 if p.code.startswith("Codebook") else 5
        nsig = p.words * per_word
        slow_i = n_st // 2 if p.slow else None
        T = tg.pipeline_times(min(nsig, 400), n_st, p.t, p.ts if p.slow else None, slow_i)
        bottleneck = max(p.t, p.ts if p.slow else 0)
        return T, M, per_word, nsig, bottleneck

    def update(self, p):
        T, M, per_word, nsig, bott = self.compute(p)
        self._T = T
        lat = T[0, -1] / 60
        disp = T[-1, -1] / 60
        rate = np.log2(M) / bott
        self.readout(lat=lat, rate=rate, disp=disp, wpm=p.words / disp)
        self.now = 0.0
        self.draw(p, None)
        self.draw_vs(p)

    def draw_vs(self, p):
        T, M, per_word, nsig, bott = self.compute(p, 2)
        lat, disp = self.r.lat, self.r.disp
        # dispatch time vs stations (curve) and latency
        ns = np.arange(2, 41)
        dt, lt = [], []
        for n in ns:
            ext = (p.ts - p.t) if p.slow else 0.0
            lt.append((n * p.t + ext) / 60)
            dt.append(((n * p.t + ext) + (nsig - 1) * bott) / 60)
        pv = self.plot("vs")
        pv.line("disp", ns, dt, color=NAVY, width=2.2, name=f"whole dispatch ({p.words} words)")
        pv.line("lat", ns, lt, color=RED, width=2.0, style="--", name="latency of the first sign")
        pv.scatter("now", [p.n], [disp], color=ORANGE, size=13, symbol="d")
        pv.scatter("nowl", [p.n], [lat], color=ORANGE, size=10, symbol="d")
        pv.set_ylim(0, max(dt) * 1.15)

    def draw(self, p, now):
        T = self._T
        ps = self.plot("st")
        nshow = min(len(T), 24)
        stn = np.arange(1, T.shape[1] + 1)
        for k in range(24):
            if k < nshow:
                ps.line(f"s{k}", np.r_[T[k, 0] - p.t, T[k]] / 60, np.r_[0, stn],
                        color=[NAVY, BLUE, PURPLE][k % 3], width=1.6)
        ps.set_xlim(0, T[nshow - 1, -1] / 60 * 1.05)
        ps.set_ylim(0, T.shape[1] + 1.5)
        if p.slow:
            ps.hband("slow", T.shape[1] // 2 + 0.5, T.shape[1] // 2 + 1.5, color=RED, alpha=0.12)
        if now is not None:
            ps.vline("now", now / 60, color=ORANGE, style="-", width=1.6)
            # where is each sign now?
            xs, ys = [], []
            for k in range(nshow):
                row = np.r_[T[k, 0] - p.t, T[k]]
                i = np.searchsorted(row, now) - 1
                if 0 <= i < len(row) - 1:
                    xs.append(now / 60)
                    ys.append(i + (now - row[i]) / max(row[i + 1] - row[i], 1e-9))
            ps.scatter("pos", xs, ys, color=ORANGE, size=9, outline=None)
            arrived = int(np.sum(T[:nshow, -1] <= now))
            ps.text("cnt", T[nshow - 1, -1] / 60 * 0.03, T.shape[1] + 1.4,
                    f"{arrived} of {nshow} signs delivered", color=ORANGE, size=9, bold=True,
                    anchor=(0, 0))

    def tick(self, p):
        T = self._T
        nshow = min(len(T), 24)
        end = T[nshow - 1, -1]
        self.now = (getattr(self, "now", 0.0) + end / 150) % (end * 1.08)
        self.draw(p, self.now)
        self.draw_vs(p)

    def story(self, p):
        M = ALPHABETS[p.alpha]
        s = (f"<p>Each tower copies the sign of the one before it. The first sign needs "
             f"{v(p.n, 'd')} hops × {v(p.t, '.0f', 's')}: a latency of {v(self.r.get('lat', 0), '.1f', 'min')}. "
             f"But a new sign can leave every {v(p.t, '.0f', 's')} while the first is still climbing, "
             f"so the lines in the diagram stay parallel: <b>pipelining</b>.</p>"
             f"<p>Each sign is one of {v(M, 'd')} shapes: log₂{M} = {v(np.log2(M), '.2f', 'bits')} in "
             f"Hartley's sense, giving {v(self.r.get('rate', 0), '.2f', 'bit/s')}. ")
        if p.code.startswith("Codebook"):
            s += ("The codebook (page + line) packs a whole word into two signs: a dictionary "
                  "compressor shared by the two terminal directors.</p>")
        else:
            s += "Spelling words letter by letter costs about five signs each: 2.5 times slower.</p>"
        if p.slow:
            s += (f"<p>The slow station (red band) needs {v(p.ts, '.0f', 's')}: every sign queues "
                  "behind it. Throughput is set by the slowest pair of hands, whatever the others do.</p>")
        return "<h3>A bucket brigade of towers</h3>" + s + keybox(
            "Latency grows with the number of hops; throughput is set by the bottleneck. Internet "
            "routers, DSP pipelines and Chappe's towers all obey the same two rules. Press Play.")


# =============================================================================== 5. repeaters
class Regenerators(Experiment):
    title = "Repeaters vs regenerators"
    blurb = "Amplify a noisy signal 1000 times, or decide and re-send it 1000 times?"
    book = "sec:ch01:digital"
    animate = True
    fps = 8
    heavy = False
    controls = [
        LogSlider("spans", "Number of spans", 1, 10000, 100, unit="spans",
                  help="Each span = a length of line plus a repeater"),
        Slider("snr", "SNR at the end of each span", 5, 40, 20, step=0.5, unit="dB"),
        Button("again", "Restart the race"),
    ]
    plots = [
        Plot("wave", "The race: the same bits after k spans", x="time (bits)", y="amplitude",
             xlim=(0, 24), ylim=(-4.2, 5.4), legend="tl", legend_cols=3),
        Plot("mc", "Monte Carlo: error rate after k spans", x="spans", y="bit error rate",
             xlim=(0.5, 30.5), ylim=(1e-6, 0.6), logy=True, legend="tl"),
        Plot("an", "Analog: end-to-end SNR", x="number of spans", y="SNR (dB)", logx=True,
             xlim=(1, 10000), ylim=(-25, 50), legend="tr"),
        Plot("dg", "Digital: end-to-end BER", x="number of spans", y="bit error rate", logx=True,
             logy=True, xlim=(1, 10000), ylim=(1e-15, 0.6), legend="tl"),
    ]
    layout = [["wave", "mc"], ["an", "dg"]]
    readouts = [
        Readout("asnr", "Analog end-to-end SNR", "dB", ".1f", good=lambda x: x >= 30),
        Readout("dber", "Digital end-to-end BER", "", "sci", good=lambda x: x < 1e-6),
        Readout("need", "Per-span SNR for 40 dB analog", "dB", ".0f"),
        Readout("k", "Spans in the race so far", "", "int"),
    ]
    challenges = [
        Challenge("The digital cliff: with 1000 spans, find the lowest per-span SNR that still gives "
                  "a BER below 10⁻⁶ (within 0.5 dB).",
                  lambda s: (abs(s.p.spans - 1000) < 30 and s.r.dber < 1e-6
                             and tg.regenerator_ber(s.p.snr - 0.5, s.p.spans) >= 1e-6)),
        Challenge("Over 1000 spans or more, make the analog chain hopeless (SNR below −10 dB) while the "
                  "digital chain is still below 10⁻⁹.",
                  lambda s: s.p.spans >= 990 and s.r.asnr < -10 and s.r.dber < 1e-9),
        Challenge("A close race: find a per-span SNR where the regenerators do make errors after 30 "
                  "spans (Monte Carlo BER above 10⁻⁴) yet the analog chain makes 100 times more.",
                  lambda s: s.r.get("mcd", 0) > 1e-4 and s.r.get("mcratio", 0) >= 100,
                  hint="Somewhere between 11 and 14 dB. Wait for the Monte Carlo curves to finish."),
    ]
    nbits, sps = 24, 16

    def setup(self):
        self.k = 1
        self.mc_item = None
        self.seed = 1

    def on_again(self, p):
        self.k = 1
        self.seed += 1

    def update(self, p):
        self.mc_item = None
        self.draw_curves(p)
        self.draw_wave(p)

    def draw_curves(self, p):
        N = int(round(p.spans))
        Ns = np.unique(np.round(np.logspace(0, 4, 120)).astype(int))
        pa = self.plot("an")
        for s_, col in ((50, GREEN), (40, ORANGE), (30, GRAY)):
            pa.line(f"r{s_}", Ns, s_ - 10 * np.log10(Ns), color=col, width=1.3, style="--",
                    name=f"{s_} dB per span")
        pa.line("you", Ns, p.snr - 10 * np.log10(Ns), color=RED, width=2.4, name="yours")
        pa.scatter("now", [N], [p.snr - 10 * np.log10(N)], color=NAVY, size=13, symbol="d")
        pa.hline("ok", 30, color=GREEN, style="--", label="toll quality ≈ 30 dB", label_pos=0.55)
        pd = self.plot("dg")
        for s_, col in ((17, GREEN), (15, ORANGE), (13, GRAY)):
            pd.line(f"r{s_}", Ns, tg.regenerator_ber(s_, Ns), color=col, width=1.3, style="--",
                    name=f"{s_} dB per span")
        pd.line("you", Ns, np.maximum(tg.regenerator_ber(p.snr, Ns), 1.2e-15), color=NAVY, width=2.6,
                name="yours")
        pd.scatter("now", [N], [max(tg.regenerator_ber(p.snr, N), 1.2e-15)], color=RED, size=13,
                   symbol="d")
        asnr = p.snr - 10 * np.log10(N)
        dber = max(float(tg.regenerator_ber(p.snr, N)), 1e-300)
        self.readout(asnr=asnr, dber=dber, need=40 + 10 * np.log10(N), k=self.k)

    def draw_wave(self, p):
        rng = np.random.default_rng(self.seed)
        bits = rng.integers(0, 2, self.nbits)
        a = 2.0 * bits - 1
        x = np.repeat(a, self.sps)
        sig = np.sqrt(10 ** (-p.snr / 10))
        k = self.k
        noise = rng.standard_normal((k, len(x))) * sig
        analog = x + noise.sum(axis=0)
        dig = x.copy()
        for j in range(k):
            y = dig + noise[j]
            dec = np.sign(y.reshape(-1, self.sps)[:, self.sps // 2])
            dig = np.repeat(dec, self.sps)
        t = np.arange(len(x)) / self.sps
        pw = self.plot("wave")
        pw.line("sent", t, x + 2.6, color=GRAY, width=1.4, name="sent (+2.6)")
        pw.line("an", t, analog, color=RED, width=1.0, name="analog")
        pw.line("dg", t, dig - 0.0, color=NAVY, width=2.2, name="regenerated")
        nerr = int(np.sum(np.sign(analog.reshape(-1, self.sps)[:, self.sps // 2]) != a))
        derr = int(np.sum(dig[::self.sps] != a))
        pw.set_title(f"After {k} span{'s' if k > 1 else ''}: analog {nerr} errors, "
                     f"regenerated {derr}")
        self.readout(k=k)

    def tick(self, p):
        self.k = self.k + 1 if self.k < 30 else 1
        self.draw_wave(p)
        self.draw_curves(p)
        if self.mc_item is not None:
            self.draw_mc(p, self.mc_item)

    def background(self, p):
        rng = np.random.default_rng(7)
        n = 30000 if self.quick else 200000
        bits = rng.integers(0, 2, n)
        a = 2.0 * bits - 1
        sig = np.sqrt(10 ** (-p.snr / 10))
        xa = a.copy()
        xd = a.copy()
        errs_a, errs_d = np.zeros(30), np.zeros(30)
        for j in range(30):
            xa = xa + sig * rng.standard_normal(n)
            xd = np.sign(xd + sig * rng.standard_normal(n))
            errs_a[j] = np.mean((xa > 0) != bits)
            errs_d[j] = np.mean((xd > 0) != bits)
            if j % 3 == 2 or j == 29:
                yield (j + 1, errs_a[:j + 1].copy(), errs_d[:j + 1].copy(), n)

    def progress(self, p, item):
        self.mc_item = item
        self.draw_mc(p, item)

    def draw_mc(self, p, item):
        k, ea, ed, n = item
        floor = 0.5 / n
        x = np.arange(1, k + 1)
        pm = self.plot("mc")
        pm.line("a", x, np.maximum(ea, floor), color=RED, width=1.6, name="analog repeaters")
        pm.scatter("ap", x, np.maximum(ea, floor), color=RED, size=6)
        pm.line("d", x, np.maximum(ed, floor), color=NAVY, width=1.6, name="regenerators")
        pm.scatter("dp", x, np.maximum(ed, floor), color=NAVY, size=6)
        pm.line("th", x, np.maximum(tg.regenerator_ber(p.snr, x), floor), color=GREEN, width=1.6,
                style=":", name="theory (regenerators)")
        pm.hline("fl", floor, color=GRAY, style=":", label="simulation floor", label_pos=0.75)
        if k == 30:
            self.readout(mcratio=float(max(ea[-1], floor) / max(ed[-1], floor)), mcd=float(ed[-1]))

    def story(self, p):
        N = int(round(p.spans))
        asnr = self.r.get("asnr", 0)
        dber = self.r.get("dber", 0)
        s = (f"<p>Every span adds the same noise. An <b>analog repeater</b> amplifies signal and noise "
             f"alike, so after {v(N, 'd')} spans the noise has added up {v(N, 'd')} times: the end-to-end SNR is "
             f"{v(p.snr, '.1f')} − 10·log₁₀{N} = {v(asnr, '.1f', 'dB')}.</p>"
             f"<p>A <b>regenerator</b> decides each bit and re-sends a clean pulse. Noise never "
             f"accumulates; only the rare wrong decisions do, and they grow merely in proportion to N: "
             f"BER {v(st.sci(dber) if dber > 1e-12 else '&lt; 10⁻¹²')}.</p>")
        if asnr < 30 and dber < 1e-6:
            s += ("<p>" + good("Digital wins by a mile.") + " For 40 dB analog quality over this chain "
                  f"each span would need {v(40 + 10 * np.log10(N), '.0f', 'dB')}; the regenerators "
                  "manage with about 17.</p>")
        elif dber >= 1e-6:
            s += ("<p>" + bad("Over the cliff.") + " Below about 13 dB per span the regenerators "
                  "make so many decisions wrong that errors pile up fast. Digital fails abruptly; "
                  "analog fades gracefully.</p>")
        s += "<p>Press Play to race both chains span by span (top left).</p>"
        return "<h3>Why everything became digital</h3>" + s + keybox(
            "Regeneration converts noise into a tiny, controllable error rate per hop. That single "
            "fact, seen first in telegraph relays, is why the world's networks went digital.")


# =============================================================================== 6. loading coils
class LoadingCoils(Experiment):
    title = "Loading the telephone line"
    blurb = "Add inductance to a lossy pair: Heaviside's trick, and the cutoff it brings."
    book = "sec:ch01:heaviside"
    controls = [
        Choice("gauge", "Cable pair", list(tg.AWG), default="22 AWG", style="menu"),
        Slider("Lc", "Loading coil inductance", 0, 200, 88, step=1, unit="mH",
               help="0 = unloaded. The Bell standard H88 is 88 mH every 6000 ft (1.83 km)"),
        Slider("d", "Coil spacing", 0.5, 3.0, 1.83, step=0.01, unit="km"),
        Slider("len", "Line length", 1, 20, 8, step=0.5, unit="km"),
    ]
    plots = [
        Plot("att", "Attenuation per kilometre", x="frequency (kHz)", y="loss (dB/km)",
             xlim=(0, 8), ylim=(0, 3.0), legend="br"),
        Plot("vel", "Phase velocity (dispersion)", x="frequency (kHz)",
             y="velocity (1000 km/s)", xlim=(0, 8), ylim=(0, 130), legend="tr"),
        Plot("tot", "End-to-end loss of your line", x="frequency (kHz)", y="loss (dB)",
             xlim=(0, 8), ylim=(0, 40), legend="tl"),
    ]
    layout = [["att", "vel"], ["tot", "tot"]]
    readouts = [
        Readout("fc", "Cutoff frequency", "kHz", ".2f"),
        Readout("l1", "Loss at 1 kHz", "dB", ".1f", good=lambda x: x <= 8),
        Readout("l3", "Loss at 3 kHz", "dB", ".1f"),
        Readout("tilt", "Tilt 3 kHz − 300 Hz", "dB", ".1f", good=lambda x: abs(x) < 3),
    ]
    challenges = [
        Challenge("Make a 12 km, 24 AWG subscriber loop usable: under 8 dB at 1 kHz, with the "
                  "cutoff above 3.4 kHz.",
                  lambda s: s.p.gauge == "24 AWG" and s.p.len >= 12 and s.r.l1 <= 8 and s.r.fc > 3.4),
        Challenge("Flatten the voice band: on 15 km of 22 AWG, keep the 3 kHz − 300 Hz tilt within "
                  "±1 dB.",
                  lambda s: s.p.gauge == "22 AWG" and s.p.len >= 15 and abs(s.r.tilt) <= 1),
        Challenge("Overdo it: so much loading that the cutoff drops below 3 kHz and kills the voice "
                  "band's top.",
                  lambda s: s.p.Lc > 0 and s.r.fc < 3.0),
    ]
    L_pair, G_pair = 0.62e-3, 0.0

    def update(self, p):
        R, Cuf = tg.AWG[p.gauge]
        C = Cuf * 1e-6
        f = np.linspace(50, 8000, 500)
        g = tg.line_gamma(f, R, self.L_pair, self.G_pair, C)
        a_un = 20 / np.log(10) * g.real
        v_un = 2 * np.pi * f / g.imag
        if p.Lc > 0:
            a_ld, v_ld = tg.loaded_line(f, R, self.L_pair, self.G_pair, C, p.d, p.Lc * 1e-3)
            fc = float(tg.loading_cutoff(p.Lc * 1e-3, C, p.d))
        else:
            a_ld, v_ld, fc = a_un, v_un, np.inf
        a_ld = np.minimum(a_ld, 50)
        fk = f / 1e3
        pa = self.plot("att")
        pa.band("voice", 0.3, 3.4, color=GREEN, alpha=0.07)
        pa.line("un", fk, a_un, color=GRAY, width=1.8, name="unloaded")
        if p.Lc > 0:
            pa.line("ld", fk, a_ld, color=NAVY, width=2.4, name=f"{p.Lc:.0f} mH every {p.d:.2f} km")
            pa.vline("fc", fc / 1e3, color=RED, style="--", label=f"cutoff {fc / 1e3:.2f} kHz",
                     label_pos=0.4)
        pv = self.plot("vel")
        pv.band("voice", 0.3, 3.4, color=GREEN, alpha=0.07)
        pv.line("un", fk, v_un / 1e3, color=GRAY, width=1.8, name="unloaded")
        if p.Lc > 0:
            ok = f < 0.97 * fc
            pv.line("ld", fk[ok], v_ld[ok] / 1e3, color=NAVY, width=2.4, name="loaded")
        pt = self.plot("tot")
        pt.band("voice", 0.3, 3.4, color=GREEN, alpha=0.07)
        pt.line("un", fk, a_un * p.len, color=GRAY, width=1.8, name="unloaded")
        tot = a_ld * p.len
        if p.Lc > 0:
            pt.line("ld", fk, np.minimum(tot, 39.5), color=NAVY, width=2.6, name="loaded")
        pt.hline("8", 8, color=RED, style=":", label="≈ 8 dB loop-loss budget", label_pos=0.62)
        pt.text("vb", 1.85, 1.0, "voice band 0.3–3.4 kHz", color=GREEN, size=9, anchor=(0.5, 1))
        l1 = float(np.interp(1000, f, tot))
        l3 = float(np.interp(3000, f, tot))
        l03 = float(np.interp(300, f, tot))
        self.readout(fc=fc / 1e3 if np.isfinite(fc) else "none", l1=l1, l3=l3, tilt=l3 - l03)

    def story(self, p):
        fc = self.r.get("fc")
        s = ("<p>A long telephone pair is mostly resistance and capacitance, like Kelvin's cable: high "
             "voice frequencies are attenuated more and travel faster than low ones (grey), so speech "
             "is muffled and smeared. Heaviside (1887) saw the cure: the line is distortionless when "
             "L/R = C/G, so add <b>inductance</b>.</p>")
        if p.Lc > 0:
            s += (f"<p>Pupin and Campbell (1899–1900) added it in lumps: a {v(p.Lc, '.0f', 'mH')} coil "
                  f"every {v(p.d, '.2f', 'km')}. Below the cutoff, {v(fc, '.2f', 'kHz')}, the loss is "
                  "lower and flatter and the velocity nearly constant (navy). But a chain of lumps is "
                  "a low-pass ladder filter: above f_c = 1/(π√(L C d)) nothing gets through.</p>")
        else:
            s += "<p>Raise the coil inductance to load the line.</p>"
        s += ("<p>That cutoff is why DSL could not run on loaded loops: millions of coils had to be "
              "dug up before broadband could reach homes on them.</p>")
        return "<h3>Inductance to the rescue</h3>" + s + keybox(
            "Loading trades bandwidth for flatness: perfect inside the voice band, a brick wall just "
            "above it.")


# =============================================================================== 7. Shannon and the modem
MODEMS = [(1962, 300, "Bell 103"), (1976, 1200, "Bell 212A"), (1984, 2400, "V.22bis"),
          (1985, 9600, "V.32"), (1991, 14400, "V.32bis"), (1994, 28800, "V.34"),
          (1996, 33600, "V.34 (1996)")]


class ModemCapacity(Experiment):
    title = "Capacity: semaphore to modem"
    blurb = "Hartley counted symbols; Shannon added noise. Modems spent 35 years closing the gap."
    book = "sec:ch01:internet"
    controls = [
        Slider("B", "Line bandwidth", 0.5, 4.0, 3.1, step=0.05, unit="kHz",
               help="A telephone channel passes about 300–3400 Hz: 3.1 kHz"),
        Slider("snr", "Signal-to-noise ratio", 0, 50, 25, step=0.5, unit="dB"),
    ]
    plots = [
        Plot("hist", "Voiceband modems and the Shannon limit", x="year", y="bit rate (b/s)",
             xlim=(1960, 2000), ylim=(100, 3e5), logy=True, legend="tl"),
        Plot("cap", "Capacity vs SNR", x="SNR (dB)",
             y="capacity (kb/s)", xlim=(0, 50), ylim=(0, 70), legend="tl"),
    ]
    layout = [["hist", "cap"]]
    readouts = [
        Readout("C", "Shannon capacity", "kb/s", ".1f"),
        Readout("eta", "Spectral efficiency", "bit/s/Hz", ".2f"),
        Readout("v34", "V.34 (33.6 kb/s) as % of capacity", "%", ".0f"),
        Readout("lev", "Hartley's distinguishable levels", "", ".0f"),
    ]
    challenges = [
        Challenge("Find the SNR at which a 3.1 kHz line can just carry V.34's 33.6 kb/s (within "
                  "0.5 dB).",
                  lambda s: abs(s.p.B - 3.1) < 0.03 and abs(s.r.C - 33.6) < 0.55),
        Challenge("Trade SNR for bandwidth: carry 33.6 kb/s through only 2.2 kHz of bandwidth.",
                  lambda s: s.p.B <= 2.2 and s.r.C >= 33.6),
        Challenge("The semaphore's 0.22 bit/s is humbled: show a 3.1 kHz line at 0 dB SNR still "
                  "carries more than 3 kb/s.",
                  lambda s: abs(s.p.B - 3.1) < 0.03 and s.p.snr <= 0.01 and s.r.C > 3),
    ]

    def update(self, p):
        B = p.B * 1e3
        C = B * np.log2(1 + 10 ** (p.snr / 10))
        ph = self.plot("hist")
        yrs = [m[0] for m in MODEMS]
        rates = [m[1] for m in MODEMS]
        ph.line("m", yrs, rates, color=NAVY, width=1.6, name="standardised modems (approx. year)")
        ph.scatter("mp", yrs, rates, color=NAVY, size=8)
        for i, (y, r_, n) in enumerate(MODEMS):
            ph.text(f"n{i}", y + 0.6, r_, n, color=NAVY, size=8, anchor=(0, 0.5))
        ph.scatter("v90", [1998], [56000], color=ORANGE, size=10, symbol="d",
                   name="V.90 (digital PCM downstream)")
        ph.hline("C", C, color=RED, style="--", width=1.8,
                 label=f"Shannon: {C / 1e3:.1f} kb/s", label_pos=0.05)
        ph.hband("gap", min(C, 33600), max(C, 33600), color=RED if C < 33600 else GREEN, alpha=0.08)
        pc = self.plot("cap")
        s = np.linspace(0, 50, 201)
        pc.line("c", s, B * np.log2(1 + 10 ** (s / 10)) / 1e3, color=NAVY, width=2.4,
                name=f"Shannon, B = {p.B:.2f} kHz")
        if abs(p.B - 3.1) > 0.01:
            pc.line("c31", s, 3100 * np.log2(1 + 10 ** (s / 10)) / 1e3, color=GREEN, width=1.6,
                    style="--", name="telephone channel, 3.1 kHz")
        pc.hline("v34", 33.6, color=ORANGE, style=":", label="V.34 33.6 kb/s", label_pos=0.8)
        pc.scatter("now", [p.snr], [C / 1e3], color=RED, size=14, symbol="d", name="your line")
        self.readout(C=C / 1e3, eta=C / B, v34=100 * 33600 / C, lev=np.sqrt(1 + 10 ** (p.snr / 10)))

    def story(self, p):
        C = self.r.get("C", 0)
        s = (f"<p>Chappe's towers moved {v(0.22, '.2f', 'bit/s')}; Hartley (1928) said a line of "
             f"bandwidth B can carry 2B symbols per second, each one of M distinguishable levels. "
             f"Shannon (1948) made \"distinguishable\" exact: with noise, "
             f"C = B·log₂(1 + SNR) = {v(C, '.1f', 'kb/s')} for your {v(p.B, '.2f', 'kHz')} line at "
             f"{v(p.snr, '.1f', 'dB')}.</p>")
        if C * 1e3 >= 33600:
            s += (f"<p>V.34's 33.6 kb/s uses {v(self.r.get('v34', 0), '.0f', '%')} of it. After 35 years "
                  "of echo cancellers, trellis codes and precoding, modem designers had arrived within a "
                  "few dB of the limit.</p>")
        else:
            s += ("<p>" + bad("Below V.34.") + " No modem, however clever, can carry 33.6 kb/s over "
                  "this line: the red dashed limit sits below it.</p>")
        s += (f"<p>Hartley's formula 2B·log₂M gives exactly Shannon's answer if M = √(1 + SNR): "
              f"noise lets you tell apart about {v(self.r.get('lev', 0), '.0f')} amplitude levels.</p>")
        s += ("<p>V.90's 56 kb/s (orange) does not break the law: downstream it rides the network's "
              "64 kb/s PCM stream and crosses only one analog loop.</p>")
        return "<h3>The speed limit of a wire</h3>" + s + keybox(
            "Capacity grows linearly with bandwidth but only logarithmically with SNR: about 1 bit/s/Hz "
            "per 3 dB at high SNR.")


# =============================================================================== the lab
LAB = st.Lab(34, "The Telegraph and the Birth of Signalling", chapter=1,
             chapter_title="The Story of Telecommunication",
             experiments=[KeyClicks, MorseHuffman, KelvinCable, Semaphore, Regenerators,
                          LoadingCoils, ModemCapacity])

if __name__ == "__main__":
    st.run(LAB)
