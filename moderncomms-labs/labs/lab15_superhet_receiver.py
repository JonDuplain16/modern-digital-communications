"""Lab 15 · The Superheterodyne Receiver, and Why Radios Went Zero-IF   (Chapters 4 and 7)

Run it:      python labs/lab15_superhet_receiver.py
Self-test:   python labs/lab15_superhet_receiver.py --selftest

Armstrong's superheterodyne (1918) solved the problem that defeated every earlier receiver:
sharp selectivity *and* easy tuning. Convert every station to one fixed intermediate frequency
and build the sharp filter once, at the IF. A century later the same block diagram sits in
every spectrum analyser, and its weaknesses (the image, the fixed high-Q filter) are why
integrated radios went zero-IF and traded them for new ones (I/Q imbalance, DC offset). Seven
experiments: tune a medium-wave band, fight the image, buy selectivity, close the AGC loop,
convert twice, survive intermodulation, and finally go zero-IF.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.signal import ellip, freqs

import commlib as cl
from commlib import rf
from commlib import superhet as sh
from commlib.superhet import Station
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD)
from studio import v, keybox, good, bad

FS, N = 256e3, 16384                 # IF-strip simulation: +-128 kHz around the IF, 64 ms
IF = 455e3


# =============================================================================== shared helpers
def spec_dbm(z, fs=FS):
    """Spectrum of a complex record in dBm per bin (a tone of amplitude A shows A^2/2 into 50 ohm),
    Blackman-Harris window, frequency axis in kHz (fftshifted)."""
    n = len(z)
    w = np.blackman(n)
    Z = np.fft.fftshift(np.fft.fft(z * w)) / w.sum()
    P = np.abs(Z) ** 2 / 2 / 50 / 1e-3
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs)) / 1e3
    return f, 10 * np.log10(P + 1e-30)


def window(f, P, lo, hi, step=1):
    k = (f >= lo) & (f <= hi)
    return f[k][::step], P[k][::step]


def contributions(info, f_if_bw, order):
    """Level (dBm) of each station's carrier at the IF output (preselector x IF filter)."""
    out = []
    for d in info:
        s = d["station"]
        g = 20 * np.log10(max(d["pre_gain"], 1e-12))
        gi = 20 * np.log10(abs(sh.butter_lp(np.array([d["offset"]]), f_if_bw / 2, order))[0] + 1e-15)
        out.append((s, s.level + g + gi, d))
    return out


def run_receiver(stations, f_rf, inj, Q, stages, if_bw, if_order, n=N, f_if=IF, rng=1):
    f_lo = f_rf + f_if if inj == "high" else f_rf - f_if
    zb, za, info = sh.if_signal(stations, f_lo, f_if, FS, n, f_rf, Q, stages, if_bw, if_order,
                                rng=rng)
    audio, dc = sh.detect_am(za, FS)
    return f_lo, zb, za, info, audio / max(dc, 1e-15)


# =============================================================================== 1. tune the band
BAND = [Station(560e3, -52, "voice", seed=1), Station(600e3, -78, "tone", tone=400, seed=2),
        Station(710e3, -60, "music", seed=3), Station(880e3, -45, "voice", seed=4),
        Station(1000e3, -70, "tone", tone=1000, seed=5), Station(1010e3, -50, "music", seed=6),
        Station(1240e3, -64, "voice", seed=7), Station(1510e3, -42, "music", seed=8),
        Station(1700e3, -72, "tone", tone=700, seed=9)]


class TuneBand(Experiment):
    title = "Tune the superhet"
    blurb = "Turn the dial across a crowded medium-wave band and follow a station to the speaker."
    book = "sec:ch04:superhet"
    controls = [
        Heading("Front panel"),
        Slider("dial", "Dial", 530, 1700, 880, step=1, unit="kHz",
               help="Tunes the preselector and the local oscillator together (ganged)"),
        Heading("Inside the radio"),
        LogSlider("Q", "Preselector Q", 3, 150, 20,
                  help="Loaded Q of the tuned circuit(s) ahead of the mixer"),
        IntSlider("stages", "Tuned stages before the mixer", 1, 2, 1),
        Choice("inj", "LO injection", ["high", "low"],
               help="High side: LO = dial + 455 kHz (every broadcast radio). Low side: dial − 455"),
        Button("listen", "▶  Listen to the radio", primary=True),
    ]
    plots = [
        Plot("rf", "Antenna: the medium-wave band (grey) and after the preselector (navy)",
             x="frequency (kHz)", y="level (dBm)", xlim=(500, 2800), ylim=(-150, 5),
             legend="tr", legend_cols=2),
        SpectrumPlot("if", "At the IF: around 455 kHz after the mixer", x="offset from 455 kHz (kHz)",
                     y="level (dBm)", xlim=(-30, 30), ylim=(-150, -30), legend="tl", legend_cols=2),
        Plot("aud", "Audio out", x="time (ms)", y="audio (re carrier)", xlim=(0, 20),
             ylim=(-1.6, 1.9), legend=None),
    ]
    layout = [["rf", "rf"], ["if", "aud"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("lo", "Local oscillator", "kHz", ".0f"),
        Readout("img", "Image frequency", "kHz", ".0f"),
        Readout("who", "Loudest rival at the IF", "kHz", None),
        Readout("sir", "Wanted / interferer", "dB", ".1f", good=lambda x: x > 25),
    ]
    challenges = [
        Challenge("Let the image in: tuned to the weak 600 kHz station, make another station louder "
                  "at the IF than the one you tuned.",
                  lambda s: abs(s.p.dial - 600) <= 1 and s.r.sir < 0,
                  hint="Which station sits at 600 + 2 × 455 kHz?"),
        Challenge("Now rescue 600 kHz: wanted at least 25 dB above every interferer.",
                  lambda s: abs(s.p.dial - 600) <= 1 and s.r.sir > 25,
                  hint="Only filtering before the mixer can help: Q and the number of tuned stages."),
        Challenge("Low-side injection: find the dial setting whose image is the 560 kHz station.",
                  lambda s: s.p.inj == "low" and abs(s.p.dial - 1470) <= 1),
    ]

    def setup(self):
        run_receiver(BAND, 880e3, "high", 20, 1, 9e3, 6)          # warm the caches

    def compute(self, p, n=N):
        return run_receiver(BAND, p.dial * 1e3, p.inj, p.Q, p.stages, 9e3, 6, n=n)

    def update(self, p):
        f_lo, zb, za, info, audio = self.compute(p)
        f_img = f_lo + IF if p.inj == "high" else f_lo - IF
        # ---- RF view
        prf = self.plot("rf")
        xs = np.repeat([s.f / 1e3 for s in BAND], 3)
        ya = np.empty(len(xs))
        ya[0::3], ya[1::3], ya[2::3] = -150, [s.level for s in BAND], np.nan
        yp = ya.copy()
        g = [20 * np.log10(abs(sh.tuned_response(s.f, p.dial * 1e3, p.Q, p.stages))) for s in BAND]
        yp[1::3] = [s.level + gi for s, gi in zip(BAND, g)]
        prf.line("ant", xs, ya, color=GRAY, width=5, alpha=0.6, name="at the antenna")
        prf.line("pre", xs, yp, color=NAVY, width=2.4, name="after the preselector")
        ff = np.linspace(500, 2800, 700)
        H = 20 * np.log10(np.abs(sh.tuned_response(ff * 1e3, p.dial * 1e3, p.Q, p.stages)))
        prf.line("H", ff, -10 + H, color=GREEN, width=1.6, style="--",
                 name="preselector response (top = 0 dB)")
        prf.vline("dial", p.dial, color=GREEN, style="-", width=1.4, label="dial", label_pos=0.95)
        prf.vline("lo", f_lo / 1e3, color=ORANGE, style="--", label="LO", label_pos=0.95)
        prf.band("img", f_img / 1e3 - 6, f_img / 1e3 + 6, color=RED, alpha=0.18)
        prf.text("imgl", f_img / 1e3 + 8, -138, "image", color=RED, size=9, anchor=(0, 1))
        # ---- IF view
        f, Pb = spec_dbm(zb)
        _, Pa = spec_dbm(za)
        f1, P1 = window(f, Pb, -30, 30, 2)
        f2, P2 = window(f, Pa, -30, 30, 2)
        pi_ = self.plot("if")
        pi_.line("b", f1, P1, color=GRAY, width=1.0, name="mixer output")
        pi_.line("a", f2, P2, color=NAVY, width=1.8, name="after the IF filter")
        fh = np.linspace(-30, 30, 241)
        pi_.line("H", fh, -40 + 20 * np.log10(np.abs(sh.butter_lp(fh * 1e3, 4.5e3, 6)) + 1e-12),
                 color=GREEN, width=1.2, style="--")
        # ---- audio
        t = np.arange(len(audio)) / FS * 1e3
        k = (t >= 30) & (t <= 50.2)
        pa = self.plot("aud")
        pa.line("a", t[k] - 30, np.clip(audio[k], -1.55, 1.85), color=RED, width=1.6)
        # ---- who is where
        contrib = contributions(info, 9e3, 6)
        wanted = [c for c in contrib if abs(c[0].f - p.dial * 1e3) <= 3e3]
        others = [c for c in contrib if abs(c[0].f - p.dial * 1e3) > 3e3]
        wl = wanted[0][1] if wanted else -150.0
        if others:
            worst = max(others, key=lambda c: c[1])
            tag = "image" if abs(worst[0].f - f_img) <= 6e3 else "neighbour"
            who = f"{worst[0].f / 1e3:.0f} ({tag})"
            sir = min(wl - worst[1], 99.0)
        else:
            who, sir = "—", 99.0
        title = (f"Audio out: the {wanted[0][0].f / 1e3:.0f} kHz station" if wanted
                 else "Audio out: nothing on this frequency (noise)")
        pa.set_title(title)
        self.readout(lo=f_lo / 1e3, img=f_img / 1e3, who=who,
                     sir=(sir if sir < 60 else "> 60") if wanted else "no station")

    def on_listen(self, p):
        f_lo, zb, za, info, audio = self.compute(p, n=1 << 18)
        self.play_audio(np.clip(0.5 * audio[::8], -1, 1), FS / 8, f"{p.dial:.0f} kHz")

    def story(self, p):
        lo = self.r.get("lo", 0)
        s = (f"<p>Turning the dial moves two things at once: the preselector's tuned circuit and the "
             f"local oscillator, which sits {v(455, 'd', 'kHz')} {'above' if p.inj == 'high' else 'below'} "
             f"the dial at {v(lo, '.0f', 'kHz')}. The mixer makes |f − f_LO|, so <b>whatever station you "
             f"tune lands at 455 kHz</b>, where one fixed, sharp IF filter (dashed green, bottom left) does "
             f"the selecting.</p>"
             f"<p>The mixer cannot tell above from below: a station at {v(self.r.get('img', 0), '.0f', 'kHz')} "
             f"(red band) also lands at 455 kHz. That is the <b>image</b>, and only the preselector, "
             f"before the mixer, can stop it.</p>")
        sir = self.r.get("sir")
        if isinstance(sir, float) and sir < 10:
            s += (f"<p>{bad('Interference.')} {self.r.get('who', '')} arrives only "
                  f"{v(sir, '.1f', 'dB')} below your station at the IF. Press Listen.</p>")
        elif sir is not None and sir != "no station":
            s += (f"<p>{good('Clean reception')}: the loudest rival is "
                  f"{v(sir if isinstance(sir, str) else format(sir, '.0f'), 's', 'dB')} down.</p>")
        return "<h3>Translate everything to one frequency</h3>" + s + keybox(
            "Superhet = tunable LO + fixed IF filter. Selectivity is designed once, at the IF; the "
            "image must be rejected before the mixer.")


# =============================================================================== 2. image problem
IF_CHOICES = {"262 kHz (1920s)": 262e3, "455 kHz (AM radios)": 455e3, "1.6 MHz": 1.6e6,
              "10.7 MHz (FM IF)": 10.7e6}


class ImageProblem(Experiment):
    title = "The image problem"
    blurb = "A mirror-image station two IFs away lands exactly on top of yours."
    book = "sec:ch04:image"
    controls = [
        Slider("dial", "Wanted station", 530, 1700, 1000, step=10, unit="kHz"),
        Choice("ifc", "Intermediate frequency", list(IF_CHOICES), default="455 kHz (AM radios)",
               style="menu"),
        Heading("Preselector"),
        LogSlider("Q", "Loaded Q", 3, 200, 20),
        IntSlider("stages", "Tuned circuits", 1, 3, 1),
        Heading("The interferer"),
        Slider("lvl", "Image station strength", 0, 50, 30, step=1, unit="dB",
               help="Strength of the station on the image frequency, relative to the wanted one"),
        Button("listen", "▶  Listen", primary=True),
    ]
    plots = [
        Plot("rf", "Before the mixer: the LO is a mirror", x="frequency (MHz)",
             y="level (dB re wanted)", ylim=(-110, 65), legend="tr", legend_cols=1),
        SpectrumPlot("if", "After the mixer and IF filter: both on 455 kHz", x="offset from the IF (kHz)",
                     y="level (dBm)", xlim=(-12, 12), ylim=(-150, -55), legend="tl", legend_cols=2),
        Plot("irr", "Image rejection vs preselector Q", x="loaded Q", y="image rejection (dB)",
             xlim=(3, 200), ylim=(0, 120), logx=True, legend="tl"),
    ]
    layout = [["rf", "rf"], ["if", "irr"]]
    readouts = [
        Readout("img", "Image frequency", "kHz", ".0f"),
        Readout("irr", "Image rejection", "dB", ".1f"),
        Readout("rel", "Image at the IF", "dB re wanted", ".1f", good=lambda x: x < -30),
        Readout("sinad", "Audio SINAD (1 kHz tone)", "dB", ".1f", good=lambda x: x > 30),
    ]
    challenges = [
        Challenge("A +40 dB image station: reach 30 dB SINAD with a 455 kHz IF, using the lowest Q you can "
                  "(Q ≤ 40).",
                  lambda s: s.p.lvl >= 40 and "455" in s.p.ifc and s.p.Q <= 40.5 and s.r.sinad > 30,
                  hint="One circuit cannot do it even at Q = 200. Add an RF stage: a second tuned circuit."),
        Challenge("Keep a cheap Q = 20 single circuit and beat a +20 dB image (SINAD above 30 dB) by "
                  "choosing the IF instead.",
                  lambda s: abs(s.p.Q - 20) < 1 and s.p.stages == 1 and s.p.lvl >= 20 and s.r.sinad > 30),
        Challenge("Economy coil: 40 dB of image rejection at 1000 kHz from one circuit and a 455 kHz "
                  "IF, with a Q no higher than 75.",
                  lambda s: abs(s.p.dial - 1000) < 1 and s.p.stages == 1 and "455" in s.p.ifc
                  and s.r.irr >= 40 and s.p.Q <= 75),
    ]

    def stations(self, p):
        f_if = IF_CHOICES[p.ifc]
        f_img = p.dial * 1e3 + 2 * f_if
        return [Station(p.dial * 1e3, -70, "tone", tone=1000, seed=5),
                Station(f_img, -70 + p.lvl, "music", seed=8)], f_if, f_img

    def simulate(self, p, n=N):
        sts, f_if, f_img = self.stations(p)
        f_lo = p.dial * 1e3 + f_if
        zb, za, info = sh.if_signal(sts, f_lo, f_if, FS, n, p.dial * 1e3, p.Q, p.stages, 9e3, 6, rng=3)
        audio, dc = sh.detect_am(za, FS)
        return sts, f_if, f_img, f_lo, za, audio / max(dc, 1e-15)

    def update(self, p):
        sts, f_if, f_img, f_lo, za, audio = self.simulate(p)
        irr = float(sh.irr_db(p.dial * 1e3, f_img, p.Q, p.stages))
        sinad = sh.tone_sinad(audio, 1000, FS)
        # RF picture in MHz
        fw, fl, fi = p.dial / 1e3, f_lo / 1e6, f_img / 1e6
        span_hi = fi + 0.6 * (fi - fw) + 0.2
        ff = np.linspace(0.05, span_hi, 900)
        H = 20 * np.log10(np.abs(sh.tuned_response(ff * 1e6, p.dial * 1e3, p.Q, p.stages)))
        pr = self.plot("rf")
        pr.set_xlim(0, span_hi)
        pr.line("H", ff, H, color=GREEN, width=2.0, name="preselector response")
        pr.stems("w", [fw], [0], color=NAVY, base=-110, size=10, name="wanted")
        pr.stems("i", [fi], [p.lvl], color=GRAY, base=-110, size=9, name="image station at the antenna")
        pr.stems("ia", [fi], [p.lvl - irr], color=RED, base=-110, size=10, name="image after preselector")
        pr.vline("lo", fl, color=ORANGE, style="--", width=1.6, label="LO (the mirror)", label_pos=0.9)
        pr.line("arrow", [fw, fi], [40, 40], color=GRAY, width=1.0, style=":")
        pr.text("d1", (fw + fl) / 2, 41, f"{f_if / 1e3:g} kHz", color=GRAY, size=9, anchor=(0.5, 1))
        pr.text("d2", (fl + fi) / 2, 41, f"{f_if / 1e3:g} kHz", color=GRAY, size=9, anchor=(0.5, 1))
        # IF picture
        f, Pa = spec_dbm(za)
        f1, P1 = window(f, Pa, -12, 12)
        pi_ = self.plot("if")
        pi_.line("a", f1, P1, color=NAVY, width=1.6, name="IF output: wanted + image")
        sts0 = [sts[0]]
        _, za0, _ = sh.if_signal(sts0, f_lo, f_if, FS, N, p.dial * 1e3, p.Q, p.stages, 9e3, 6, rng=3)
        _, P0 = spec_dbm(za0)
        pi_.line("w", f1, window(f, P0, -12, 12)[1], color=GREEN, width=1.8, name="wanted alone", z=2)
        # IRR vs Q
        qs = np.logspace(np.log10(3), np.log10(200), 120)
        pq = self.plot("irr")
        for k, col in ((1, NAVY), (2, BLUE), (3, PURPLE)):
            pq.line(f"s{k}", qs, sh.irr_db(p.dial * 1e3, f_img, qs, k), color=col,
                    width=2.4 if k == p.stages else 1.0, name=f"{k} circuit{'s' if k > 1 else ''}")
        need = p.lvl + 30
        pq.hline("need", need, color=RED, style="--", label=f"needed for 30 dB: {need:.0f} dB",
                 label_pos=0.45)
        pq.scatter("now", [p.Q], [irr], color=ORANGE, size=14, symbol="d")
        self.readout(img=f_img / 1e3, irr=irr, rel=p.lvl - irr, sinad=sinad)

    def on_listen(self, p):
        *_, audio = self.simulate(p, n=1 << 18)
        self.play_audio(np.clip(0.5 * audio[::8], -1, 1), FS / 8, "the IF strip")

    def story(self, p):
        f_if = IF_CHOICES[p.ifc]
        rel = self.r.get("rel", 0)
        s = (f"<p>The LO sits {v(f_if / 1e3, 'g', 'kHz')} above your station. Anything the same distance "
             f"<i>above</i> the LO, at {v(self.r.get('img', 0) / 1e3, '.3f', 'MHz')}, is its mirror image: "
             f"the mixer turns both into exactly {v(f_if / 1e3, 'g', 'kHz')}. After the mixer they are one "
             f"signal (bottom left): no IF filter can separate them.</p>"
             f"<p>A tuned circuit attenuates the image by √(1 + Q²ρ²) with ρ = f_i/f_s − f_s/f_i; here "
             f"{v(self.r.get('irr', 0), '.1f', 'dB')}, leaving the image {v(rel, '.1f', 'dB')} relative "
             f"to your station.</p>")
        if rel > -30:
            s += ("<p>" + bad("Not enough.") + " Raise Q, add a tuned circuit (an RF amplifier stage), "
                  "or move the IF up so the image is further away in relative terms.</p>")
        else:
            s += "<p>" + good("The image is buried.") + "</p>"
        return "<h3>A mirror on the frequency axis</h3>" + s + keybox(
            "Rule 1 of IF choice: high enough that the preselector can reject the image. "
            "Rule 2 (next experiment): low enough that the IF filter can be sharp.")


# =============================================================================== 3. IF selectivity
class IFSelectivity(Experiment):
    title = "IF selectivity"
    blurb = "A neighbour 10 kHz away and 20 dB louder. How sharp must the IF filter be?"
    book = "sec:ch04:dynrange"
    controls = [
        Heading("IF filter"),
        IntSlider("order", "Filter order (poles)", 1, 10, 2),
        Slider("bw", "Bandwidth", 4, 16, 9, step=0.5, unit="kHz"),
        Heading("The band"),
        Choice("sp", "Channel spacing", ["10 kHz (Americas)", "9 kHz (Europe)"]),
        Slider("adj", "Neighbour strength", 0, 40, 20, step=1, unit="dB re wanted"),
        Toggle("both", "Neighbours on both sides", False),
        Button("listen", "▶  Listen", primary=True),
    ]
    plots = [
        SpectrumPlot("resp", "IF filter response and the channels", x="offset from the IF (kHz)",
                     y="response (dB)", xlim=(-30, 30), ylim=(-90, 5), legend=None),
        SpectrumPlot("ifs", "IF spectrum: before (grey) and after (navy) the filter",
                     x="offset from the IF (kHz)", y="level (dBm)", xlim=(-30, 30), ylim=(-150, -30),
                     legend="tl", legend_cols=2),
        SpectrumPlot("aud", "Audio spectrum: the 1 kHz test tone and what else got through",
                     x="audio frequency (kHz)", y="level (dB re tone)", xlim=(0, 5.2), ylim=(-80, 5),
                     legend=None),
    ]
    layout = [["resp", "ifs"], ["aud", "aud"]]
    readouts = [
        Readout("acs", "Rejection one channel away", "dB", ".1f", good=lambda x: x > 40),
        Readout("sf", "Shape factor (60 dB / 6 dB)", "", ".2f", good=lambda x: x < 2.5),
        Readout("tre", "Treble at 4 kHz audio", "dB", ".1f", good=lambda x: x > -3),
        Readout("sinad", "Audio SINAD", "dB", ".1f", good=lambda x: x > 25),
    ]
    challenges = [
        Challenge("Selectivity and fidelity: SINAD above 25 dB with the +20 dB neighbour, while 4 kHz "
                  "treble loses no more than 3 dB.",
                  lambda s: s.p.adj >= 20 and s.r.sinad > 25 and s.r.tre >= -3),
        Challenge("European crowding: 9 kHz spacing, +20 dB neighbours on both sides, SINAD above 22 dB.",
                  lambda s: s.p.sp.startswith("9") and s.p.both and s.p.adj >= 20 and s.r.sinad > 22,
                  hint="Something has to give: the treble."),
        Challenge("A mechanical-filter shape factor: 2.0 or better.",
                  lambda s: s.r.sf <= 2.0),
    ]
    Qpre = 40.0

    def stations(self, p):
        sp = 10e3 if p.sp.startswith("10") else 9e3
        sts = [Station(1000e3, -70, "tone", tone=1000, seed=5),
               Station(1000e3 + sp, -70 + p.adj, "music", seed=6)]
        if p.both:
            sts.append(Station(1000e3 - sp, -70 + p.adj, "voice", seed=4))
        return sts, sp

    def simulate(self, p, n=N):
        sts, sp = self.stations(p)
        f_lo = 1000e3 + IF
        zb, za, info = sh.if_signal(sts, f_lo, IF, FS, n, 1000e3, self.Qpre, 1, p.bw * 1e3, p.order,
                                    rng=4)
        audio, dc = sh.detect_am(za, FS)
        return zb, za, audio / max(dc, 1e-15), sp

    def update(self, p):
        zb, za, audio, sp = self.simulate(p)
        f = np.linspace(-30, 30, 601)
        H = 20 * np.log10(np.abs(sh.butter_lp(f * 1e3, p.bw * 1e3 / 2, p.order)) + 1e-12)
        pr = self.plot("resp")
        for k in range(-3, 4):
            c = k * sp / 1e3
            pr.band(f"c{k}", c - 4.5, c + 4.5, color=NAVY if k == 0 else (RED if (k == -1 or (k == 1 and p.both)) else GRAY),
                    alpha=0.16 if k == 0 or k == -1 or (k == 1 and p.both) else 0.05)
        pr.line("H", f, H, color=NAVY, width=2.4)
        acs = -float(20 * np.log10(abs(sh.butter_lp(np.array([sp]), p.bw * 1e3 / 2, p.order))[0]))
        pr.hline("acs", -acs, color=RED, style=":", label=f"next channel: −{acs:.0f} dB", label_pos=0.03)
        # shape factor: 60 dB width / 6 dB width (Butterworth closed form)
        w6 = (10 ** 0.6 - 1) ** (1 / (2 * p.order))
        w60 = (10 ** 6 - 1) ** (1 / (2 * p.order))
        tre = float(20 * np.log10(abs(sh.butter_lp(np.array([4e3]), p.bw * 1e3 / 2, p.order))[0]))
        fz, Pb = spec_dbm(zb)
        _, Pa = spec_dbm(za)
        f1, P1 = window(fz, Pb, -30, 30, 2)
        f2, P2 = window(fz, Pa, -30, 30, 2)
        pi_ = self.plot("ifs")
        pi_.line("b", f1, P1, color=GRAY, width=1.0, name="mixer output")
        pi_.line("a", f2, P2, color=NAVY, width=1.8, name="IF filter output")
        # audio spectrum (real, one-sided), relative to the tone
        A = np.abs(np.fft.rfft(audio * np.blackman(len(audio)))) ** 2
        fa = np.fft.rfftfreq(len(audio), 1 / FS) / 1e3
        k = fa <= 5.3
        Adb = 10 * np.log10(A[k] / A[k].max() + 1e-14)
        pa = self.plot("aud")
        pa.line("a", fa[k], Adb, color=NAVY, width=1.4, fill=-80, fill_alpha=0.12)
        pa.vline("t", 1.0, color=GREEN, style=":", label="test tone", label_pos=0.92)
        sinad = sh.tone_sinad(audio, 1000, FS)
        self.readout(acs=acs, sf=w60 / w6, tre=tre, sinad=sinad)

    def on_listen(self, p):
        zb, za, audio, sp = self.simulate(p, n=1 << 18)
        self.play_audio(np.clip(0.5 * audio[::8], -1, 1), FS / 8, "the receiver")

    def story(self, p):
        sinad = self.r.get("sinad", 0)
        s = (f"<p>Your station is the navy channel; its neighbour, {v(p.adj, '.0f', 'dB')} louder, sits "
             f"{v(10 if p.sp.startswith('10') else 9, 'd', 'kHz')} away. A {v(p.order, 'd')}-pole filter "
             f"{v(p.bw, '.1f', 'kHz')} wide knocks it down by {v(self.r.get('acs', 0), '.1f', 'dB')}.</p>")
        if sinad < 15:
            s += ("<p>" + bad("The neighbour wins.") + " Too much of it reaches the envelope detector, "
                  "whose output follows the <i>stronger</i> signal's envelope: its programme comes out of "
                  "your speaker (the hash in the audio spectrum), and your tone is pushed down.</p>")
        elif sinad < 25:
            s += "<p>Better, but the neighbour's programme is still audible under the tone.</p>"
        else:
            s += ("<p>" + good("Clean enough.") + " What remains is the neighbour's own sideband "
                  "splatter, right at the edge of your channel: no IF filter can remove that.</p>")
        s += (f"<p>The catch: AM's audio bandwidth is half the IF bandwidth. Narrow the filter for "
              f"selectivity and the treble goes ({v(self.r.get('tre', 0), '.1f', 'dB')} at 4 kHz). Only "
              f"<b>steeper skirts</b> (more poles: ceramic, crystal and mechanical filters) buy both.</p>")
        return "<h3>Buying selectivity</h3>" + s + keybox(
            "Each extra pole adds 6 dB/octave of skirt. Shape factor, not bandwidth, separates a good "
            "filter from a bad one.")


# =============================================================================== 4. AGC
class AGCLoop(Experiment):
    title = "AGC: the receiver's iris"
    blurb = "Hold the detector level steady over 100 dB of fading without flattening the programme."
    book = "sec:ch04:agc"
    controls = [
        Choice("scen", "Antenna signal", ["Steps +30/−40/+15 dB", "Rayleigh fading"]),
        LogSlider("det", "Detector time constant", 1, 200, 30, unit="ms"),
        LogSlider("att", "Attack time", 1, 1000, 60, unit="ms", help="How fast the gain falls when too loud"),
        LogSlider("dec", "Decay time", 1, 3000, 250, unit="ms", help="How fast the gain recovers when too quiet"),
        Toggle("on", "AGC on", True),
        Button("listen", "▶  Listen (voice programme)", primary=True),
    ]
    plots = [
        Plot("in", "Antenna level", x="time (s)", y="level (dB)", xlim=(0, 3), ylim=(-50, 45),
             legend=None),
        Plot("gain", "AGC gain", x="time (s)", y="gain (dB)", xlim=(0, 3), ylim=(-45, 50), legend=None),
        Plot("out", "Detector input: level held, modulation kept?", x="time (s)", y="envelope",
             xlim=(0, 3), ylim=(0, 3.6), legend="tl", legend_cols=2),
    ]
    layout = [["in", "gain"], ["out", "out"]]
    readouts = [
        Readout("spread", "Level held within (95 %)", "± dB", ".1f", good=lambda x: x < 3),
        Readout("mod", "Modulation kept", "%", ".0f", good=lambda x: x > 85),
        Readout("rec", "Recovery after the fade", "ms", ".0f"),
        Readout("ovs", "Overshoot after the jump", "dB", ".1f", good=lambda x: x < 6),
    ]
    challenges = [
        Challenge("The broadcast compromise: level held within ±3 dB and at least 90 % of the "
                  "modulation kept (step scenario).",
                  lambda s: s.p.scen.startswith("Steps") and s.p.on and s.r.spread < 3 and s.r.mod >= 90),
        Challenge("Eat the programme: a loop so fast that less than 30 % of the modulation survives.",
                  lambda s: s.p.on and s.r.mod < 30),
        Challenge("Ride the fades: Rayleigh fading held within ±4 dB with at least 70 % modulation.",
                  lambda s: s.p.scen.startswith("Ray") and s.p.on and s.r.spread < 4 and s.r.mod >= 70,
                  hint="Fades last tenths of a second: the loop must be fast, but not as fast as 40 Hz audio."),
    ]
    fs = 4000.0

    def scenario(self, p, fs, mod_sig=None, seed=1):
        n = int(3.0 * fs)
        t = np.arange(n) / fs
        if p.scen.startswith("Steps"):
            lvl = np.where(t < 0.6, 0, np.where(t < 1.5, 30, np.where(t < 2.2, -10, 5)))
            lvl = lvl + 4 * np.sin(2 * np.pi * 1.3 * t)
        else:
            rng = np.random.default_rng(seed)
            k = 4096
            X = np.zeros(k, complex)
            nd = 8                                    # Doppler bins (~2.7 Hz over 3 s)
            X[1:nd] = rng.standard_normal(nd - 1) + 1j * rng.standard_normal(nd - 1)
            X[-nd + 1:] = rng.standard_normal(nd - 1) + 1j * rng.standard_normal(nd - 1)
            X[0] = 0.4
            h = np.fft.ifft(X)
            h = np.interp(t, np.arange(k) / k * 3.0, np.abs(h))
            lvl = 20 * np.log10(h / np.sqrt(np.mean(h ** 2)) + 1e-6) + 10
        m = 0.6 * np.sin(2 * np.pi * 40 * t) if mod_sig is None else 0.6 * mod_sig[:n]
        rng = np.random.default_rng(seed + 7)
        noise = 10 ** (-45 / 20) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        env = np.abs(10 ** (lvl / 20) * (1 + m) + noise)
        return t, lvl, env, m

    def run_agc(self, p, env, fs):
        if not p.on:
            g = np.zeros(len(env))
            return env.copy(), g
        out, g, _ = sh.agc_loop(env, fs, p.det * 1e-3, p.att * 1e-3, p.dec * 1e-3)
        return out, g

    def update(self, p):
        fs = self.fs
        t, lvl, env, m = self.scenario(p, fs)
        out, g = self.run_agc(p, env, fs)
        per = int(fs / 40)                                       # one audio period
        ker = np.ones(per) / per
        sm = np.convolve(out, ker, mode="same")
        ldb = 20 * np.log10(np.maximum(sm, 1e-9))
        ok = (t > 0.3) & (t < 2.95)
        if p.scen.startswith("Steps"):           # judge the hold once each step has settled
            settled = ok & ~(((t > 0.6) & (t < 1.1)) | ((t > 1.5) & (t < 2.0)) | ((t > 2.2) & (t < 2.7)))
        else:
            settled = ok
        ref = np.median(ldb[ok])
        spread = float(np.percentile(np.abs(ldb[settled] - ref), 95))
        # modulation kept: projection of the output's relative ripple onto the programme
        rel = (out / np.maximum(sm, 1e-12) - 1)[ok]
        mm = m[ok]
        mod = 100 * float(np.clip(np.dot(rel, mm) / np.dot(mm, mm), 0, 1.2))
        if p.scen.startswith("Steps"):
            after = (t > 1.5) & (t < 2.15)
            bad_ = np.nonzero(np.abs(ldb[after] - ref) > 3)[0]
            rec = (bad_[-1] + 1) / fs * 1e3 if len(bad_) else 0.0
            ovs = float(np.max(ldb[(t > 0.6) & (t < 1.0)]) - ref)
        else:
            rec, ovs = "—", "—"
        pin = self.plot("in")
        pin.line("l", t, lvl, color=NAVY, width=1.8)
        pg = self.plot("gain")
        pg.line("g", t, g, color=ORANGE, width=1.8)
        po = self.plot("out")
        dec = 2
        po.line("o", t[::dec], np.clip(out[::dec], 0, 3.55), color=GREEN if mod > 85 else RED,
                width=0.8, name="detector input envelope")
        po.line("s", t, np.clip(sm, 0, 3.55), color=NAVY, width=2.0, name="average level")
        po.hband("tgt", 10 ** (-3 / 20), 10 ** (3 / 20), color=GREEN, alpha=0.08)
        if not p.on:
            po.set_ylim(0, 3.6)
        self.readout(spread=spread, mod=mod, rec=rec, ovs=ovs)

    def on_listen(self, p):
        fs = 8000.0
        from commlib.analog import voice_like
        vo = voice_like(fs, 3.0)
        t, lvl, env, m = self.scenario(p, fs, mod_sig=vo / np.max(np.abs(vo)))
        out, g = self.run_agc(p, env, fs)
        audio = out - np.convolve(out, np.ones(400) / 400, mode="same")
        audio = audio / max(np.percentile(np.abs(audio), 99.5), 1e-9) * 0.4 if p.on else audio / 10
        self.play_audio(np.clip(audio, -1, 1), fs, "the detector output")

    def story(self, p):
        mod = self.r.get("mod", 0)
        spread = self.r.get("spread", 0)
        s = ("<p>The antenna signal (top left) jumps by +30, −40 and +15 dB. The AGC measures the "
             "detector's average level and turns the RF and IF gain down when it is too high and up "
             "when too low (top right): fast <b>attack</b> so a strong signal cannot overload anything, "
             "slow <b>decay</b> so it does not pump during pauses.</p>")
        if not p.on:
            s += "<p>" + bad("AGC off:") + " the detector sees the full 100 dB of variation.</p>"
        elif mod < 50:
            s += (f"<p>{bad('Too fast.')} With a {v(p.det, '.0f', 'ms')} detector the loop follows the "
                  f"40 Hz programme itself and irons it flat: only {v(mod, '.0f', '%')} survives.</p>")
        elif spread > 4:
            s += (f"<p>{bad('Too slow.')} The level wanders ±{v(spread, '.1f', 'dB')}: listeners would "
                  "be chasing the volume knob, as they did before Wheeler's AVC (1925).</p>")
        else:
            s += (f"<p>{good('Iris adjusted.')} Level held within ±{v(spread, '.1f', 'dB')} with "
                  f"{v(mod, '.0f', '%')} of the modulation kept.</p>")
        s += ("<p>Press Listen with a fast decay: the background noise \"breathes\" up between words, "
              "the classic sound of a badly designed AGC.</p>")
        return "<h3>Fast attack, slow decay</h3>" + s + keybox(
            "An AGC is a feedback loop on the signal's <i>level</i>: time constants long compared with "
            "the audio, short compared with the fading.")


# =============================================================================== 5. double conversion
ARCHS = {"Single: 455 kHz IF": ("single", 0.455e6, None),
         "Single: 9 MHz IF": ("single", 9e6, None),
         "Double: 45 MHz up, then 455 kHz": ("double", 45e6, 0.455e6),
         "Double: 70 MHz up, then 10.7 MHz": ("double", 70e6, 10.7e6)}


@lru_cache(maxsize=4)
def _lpf():
    b, a = ellip(7, 0.2, 85, 2 * np.pi * 32e6, analog=True)
    return b, a


def lpf_db(f_hz):
    b, a = _lpf()
    _, h = freqs(b, a, worN=2 * np.pi * np.atleast_1d(np.asarray(f_hz, float)))
    return 20 * np.log10(np.abs(h) + 1e-12)


class DoubleConversion(Experiment):
    title = "Double conversion"
    blurb = "Shortwave, 1–30 MHz: convert up first, and the image problem disappears."
    book = "sec:ch04:image"
    controls = [
        Slider("f", "Tuned frequency", 1, 30, 14.2, step=0.05, unit="MHz"),
        Choice("arch", "Architecture", list(ARCHS), default="Single: 455 kHz IF", style="menu"),
        LogSlider("Q", "Tracking preselector Q", 10, 150, 40,
                  enabled_if=lambda p: ARCHS[p.arch][0] == "single"),
        Choice("if1f", "First-IF filter", ["Crystal roofing filter", "LC tank (Q = 50)"],
               enabled_if=lambda p: ARCHS[p.arch][0] == "double", style="menu"),
    ]
    plots = [
        Plot("plan", "Frequency plan: front end, LO and the first image", x="frequency (MHz)",
             y="front-end response (dB)", ylim=(-110, 15), legend="tr", legend_cols=2),
        Plot("plan2", "Second conversion: around the first IF", x="offset from the first IF (MHz)",
             y="first-IF filter (dB)", xlim=(-24, 4), ylim=(-110, 15), legend="bl"),
        Plot("irr", "First-image rejection across the shortwave band", x="tuned frequency (MHz)",
             y="image rejection (dB)", xlim=(1, 30), ylim=(0, 135), legend=None),
    ]
    layout = [["plan", "plan"], ["plan2", "irr"]]
    readouts = [
        Readout("img", "First image", "MHz", ".2f"),
        Readout("r1", "First-image rejection", "dB", ".1f", good=lambda x: x >= 60),
        Readout("r2", "Second-image rejection", "dB", ".1f", good=lambda x: x >= 60),
        Readout("ift", "IF feedthrough rejection", "dB", ".1f", good=lambda x: x >= 60),
    ]
    challenges = [
        Challenge("At least 80 dB of first-image rejection everywhere from 1 to 30 MHz.",
                  lambda s: s.r.minr >= 80),
        Challenge("Show single conversion's HF weakness: a 455 kHz IF, tuned above 25 MHz, with less "
                  "than 15 dB of image rejection.",
                  lambda s: s.p.arch.startswith("Single: 455") and s.p.f > 25 and s.r.r1 < 15),
        Challenge("Make the second conversion the weak link: second-image rejection under 30 dB.",
                  lambda s: isinstance(s.r.r2, float) and s.r.r2 < 30,
                  hint="A wide first-IF filter cannot stop an image only 2 × 455 kHz away."),
        Challenge("IF feedthrough: with a 9 MHz IF, tune where a 9 MHz signal leaks in with less than "
                  "20 dB of rejection.",
                  lambda s: s.p.arch.startswith("Single: 9") and s.r.ift < 20),
    ]

    def front_db(self, kind, f_if1, f_tuned, Q, fq):
        if kind == "single":
            return 20 * np.log10(np.abs(sh.tuned_response(fq, f_tuned, Q, 1)))
        return lpf_db(fq)

    def img_rej(self, kind, f_if1, f_tuned, Q):
        fi = f_tuned + 2 * f_if1
        return -float(self.front_db(kind, f_if1, f_tuned, Q, np.array([fi]))[0]) + \
            float(self.front_db(kind, f_if1, f_tuned, Q, np.array([f_tuned]))[0])

    def if1_resp(self, p, f_if1, off):
        if p.if1f.startswith("Crystal"):
            return 20 * np.log10(np.abs(sh.butter_lp(off, 7.5e3, 8)) + 1e-14)
        return 20 * np.log10(np.abs(sh.tuned_response(f_if1 + off, f_if1, 50, 1)))

    def update(self, p):
        kind, f_if1, f_if2 = ARCHS[p.arch]
        fw = p.f * 1e6
        f_lo = fw + f_if1
        fi = fw + 2 * f_if1
        top = max(fi * 1.15, 40e6) / 1e6
        fq = np.linspace(0.1, top, 1500) * 1e6
        H = self.front_db(kind, f_if1, fw, p.Q, fq)
        pp = self.plot("plan")
        pp.set_xlim(0, top)
        pp.line("H", fq / 1e6, H, color=GREEN, width=2.2,
                name="tracking preselector" if kind == "single" else "fixed 0–30 MHz low-pass")
        pp.band("hf", 1, 30, color=NAVY, alpha=0.05)
        r1 = self.img_rej(kind, f_if1, fw, p.Q)
        pp.stems("w", [p.f], [0], color=NAVY, base=-110, size=10, name="wanted")
        pp.stems("i", [fi / 1e6], [-r1], color=RED, base=-110, size=10, name="first image")
        pp.vline("lo", f_lo / 1e6, color=ORANGE, style="--", label="LO 1", label_pos=0.9)
        pp.vline("if", f_if1 / 1e6, color=PURPLE, style=":", label="IF 1", label_pos=0.75)
        ift = -float(self.front_db(kind, f_if1, fw, p.Q, np.array([f_if1]))[0])
        # second conversion
        p2 = self.plot("plan2")
        if kind == "double":
            off = np.linspace(-24e6, 4e6, 1400)
            R = self.if1_resp(p, f_if1, off)
            img2 = -2 * f_if2
            r2 = min(-float(self.if1_resp(p, f_if1, np.array([img2]))[0]), 100.0)  # ultimate rejection
            p2.line("R", off / 1e6, R, color=GREEN, width=2.2, name=p.if1f)
            p2.stems("w", [0], [0], color=NAVY, base=-110, size=9, name="wanted at IF 1")
            p2.stems("i", [img2 / 1e6], [-r2], color=RED, base=-110, size=9, name="second image")
            p2.vline("lo2", -f_if2 / 1e6, color=ORANGE, style="--", label="LO 2", label_pos=0.9)
            p2.set_title(f"Second conversion: {f_if1 / 1e6:g} MHz → {f_if2 / 1e6:g} MHz")
            p2.set_xlim(min(-2.6 * f_if2, -1.5e6) / 1e6, 0.6 * max(f_if2, 1e6) / 1e6)
        else:
            r2 = "—"
            p2.set_title("Second conversion: none in a single-conversion radio")
            p2.text("none", -10, -50, "single conversion:\nno second image", color=GRAY, size=11,
                    anchor=(0.5, 0.5))
            p2.set_xlim(-24, 4)
        # IRR across the band, all architectures
        fb = np.linspace(1, 30, 120)
        pr = self.plot("irr")
        mins = {}
        for i, (name, (k_, i1, _)) in enumerate(ARCHS.items()):
            curve = np.array([self.img_rej(k_, i1, x * 1e6, p.Q) for x in fb])
            mins[name] = curve.min()
            col = [RED, ORANGE, NAVY, PURPLE][i]
            pr.line(f"a{i}", fb, np.minimum(curve, 130), color=col,
                    width=2.8 if name == p.arch else 1.2)
            lab = name.split(",")[0].replace("Double: ", "").replace("Single: ", "")
            yl = [None, None, 124, 104][i]
            if yl is None:
                pr.text(f"t{i}", 29.5, min(curve[-1], 130) + 2, name.replace("Single: ", "single, "),
                        color=col, size=8.5, anchor=(1, 1), bold=name == p.arch)
            else:
                pr.text(f"t{i}", 1.5, yl, ("double, " + lab), color=col, size=8.5, anchor=(0, 1),
                        bold=name == p.arch)
        pr.scatter("now", [p.f], [min(r1, 130)], color=GREEN, size=14, symbol="d")
        self.readout(img=fi / 1e6, r1=r1, r2=r2, ift=ift, minr=mins[p.arch])

    def story(self, p):
        kind, f_if1, f_if2 = ARCHS[p.arch]
        s = (f"<p>Tuned to {v(p.f, '.2f', 'MHz')} with a {v(f_if1 / 1e6, 'g', 'MHz')} first IF, the image "
             f"is at {v(self.r.get('img', 0), '.2f', 'MHz')}.</p>")
        if kind == "single":
            s += ("<p>At shortwave a 455 kHz IF puts the image only 910 kHz away, a few percent of the "
                  "signal frequency: a tracking tuned circuit can barely tell them apart above 10 MHz. "
                  "Raising the IF helps, but a 9 MHz IF sits inside the band you want to tune, and a "
                  "strong station on 9 MHz leaks straight through.</p>")
        else:
            s += ("<p><b>Up-conversion</b> puts the first IF above the whole tuning range. The image is "
                  "then 90 MHz or more away, and a fixed 0–30 MHz low-pass filter removes every image "
                  "with no tracking at all. A narrow crystal <b>roofing filter</b> at the first IF then "
                  "protects the second conversion, whose own image is only 2 × IF₂ away.</p>")
        return "<h3>Convert twice</h3>" + s + keybox(
            "When one IF cannot satisfy both rules, use two: a high first IF for images, a low "
            "second IF for selectivity. Modern HF receivers and spectrum analysers all do this.")


# =============================================================================== 6. intermodulation
class Intermod(Experiment):
    title = "Intermod, blocking and SFDR"
    blurb = "Two strong stations you are not listening to manufacture a ghost on your channel."
    book = "sec:ch04:dynrange"
    controls = [
        Heading("The band"),
        Slider("pin", "Each strong interferer", -70, -5, -30, step=1, unit="dBm"),
        Slider("df", "Interferer spacing Δ", 10, 100, 40, step=5, unit="kHz",
               help="Tones at +Δ and +2Δ: their IM3 product 2f₁ − f₂ lands on your channel"),
        Slider("pw", "Wanted station", -130, -70, -100, step=1, unit="dBm"),
        Heading("Front end"),
        Slider("iip3", "Input intercept IIP3", -20, 40, 5, step=1, unit="dBm"),
        Slider("nf", "Noise figure", 3, 20, 8, step=0.5, unit="dB"),
        Slider("att", "Input attenuator", 0, 30, 0, step=1, unit="dB"),
    ]
    plots = [
        SpectrumPlot("spec", "Receiver output, referred to the antenna (simulated)",
                     x="offset from your channel (kHz)", y="level (dBm per 100 Hz)", ylim=(-160, 10),
                     legend="tl", legend_cols=3),
        Plot("ip3", "Two-tone test: fundamental and IM3 vs input", x="input per tone (dBm, at the antenna)",
             y="output (dBm, antenna-referred)", xlim=(-80, 40), ylim=(-170, 50), legend="tl"),
        Plot("atp", "Wanted SINR vs attenuator", x="attenuation (dB)", y="SINR (dB)", xlim=(0, 30),
             ylim=(-30, 40), legend="tr"),
    ]
    layout = [["spec", "spec"], ["ip3", "atp"]]
    readouts = [
        Readout("im3", "IM3 on your channel", "dBm", ".1f"),
        Readout("sinr", "Wanted SINR (2.4 kHz)", "dB", ".1f", good=lambda x: x > 10),
        Readout("sfdr", "SFDR (2.4 kHz)", "dB", ".1f", good=lambda x: x >= 100),
        Readout("des", "Blocking: wanted gain change", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Rescue the weak station with the attenuator alone: SINR at least 15 dB with the "
                  "default interferers (−30 dBm, IIP3 +5 dBm, NF 8 dB).",
                  lambda s: s.p.pin >= -30 and s.p.iip3 <= 5 and s.p.nf >= 8 and s.p.pw <= -100
                  and s.r.sinr >= 15),
        Challenge("Find the best attenuation for −20 dBm interferers (within 1 dB of the best SINR).",
                  lambda s: s.p.pin >= -20 and s.r.sinr >= s.r.best - 1 and s.p.att > 0),
        Challenge("Contest grade: an SFDR of at least 110 dB in 2.4 kHz.",
                  lambda s: s.r.sfdr >= 110,
                  hint="SFDR = ⅔(IIP3 − MDS); MDS = −174 + 10·log₁₀(2400) + NF."),
    ]
    fs, n, B = 1.6384e6, 16384, 2400.0

    def analytic(self, pin, pw, iip3, nf, att):
        mds = -174 + 10 * np.log10(self.B) + nf + att
        im3 = 3 * (pin - att) - 2 * iip3 + att
        sinr = pw - 10 * np.log10(10 ** (im3 / 10) + 10 ** (mds / 10))
        return mds, im3, sinr

    def update(self, p):
        mds, im3, sinr = self.analytic(p.pin, p.pw, p.iip3, p.nf, p.att)
        sfdr = 2 / 3 * (p.iip3 + p.att - mds)
        atts = np.arange(0, 30.5, 0.5)
        curve = np.array([self.analytic(p.pin, p.pw, p.iip3, p.nf, a)[2] for a in atts])
        # simulation (complex baseband at the front-end input)
        df = round(p.df * 1e3 / (self.fs / self.n)) * self.fs / self.n
        t = np.arange(self.n) / self.fs
        A = sh.dbm_to_amp(p.pin - p.att)
        Aw = sh.dbm_to_amp(p.pw - p.att)
        x = A * (np.exp(2j * np.pi * df * t) + np.exp(2j * np.pi * 2 * df * t + 0.7)) + Aw
        n0 = 1e-3 * 10 ** ((-174 + p.nf) / 10) * 50 * 2 * self.fs / 2   # input-referred noise
        x = x + np.sqrt(n0) * (self.rng.standard_normal(self.n) + 1j * self.rng.standard_normal(self.n)) / np.sqrt(2)
        y = sh.two_tone_cubic(x, 1.0, p.iip3)
        f, P = spec_dbm(y, self.fs)
        P = P + p.att
        lim = 3.6 * p.df
        ff, PP = window(f, P, -lim, lim)
        ps = self.plot("spec")
        ps.set_xlim(-lim, lim)
        ps.line("P", ff, PP, color=NAVY, width=1.4)
        ps.vline("ch", 0, color=GREEN, style=":", label="your channel", label_pos=0.92)
        ps.hline("im3", im3, color=RED, style="--", label=f"IM3 2f₁−f₂ = {im3:.0f} dBm", label_pos=0.6)
        ps.hline("mds", mds, color=GRAY, style=":")
        ps.text("mdsl", -0.97 * lim, mds + 2, f"noise in 2.4 kHz (MDS): {mds:.0f} dBm", color=GRAY,
                size=8.5, anchor=(0, 1), fill=True)
        # IP3 diagram (antenna referred)
        pin = np.linspace(-80, 40, 121)
        G = 0
        pi_ = self.plot("ip3")
        pi_.line("f", pin, pin + G, color=NAVY, width=2.0, name="fundamental")
        pi_.line("i", pin, 3 * pin - 2 * (p.iip3 + p.att) + G, color=RED, width=2.0, name="IM3 (3 dB per dB)")
        pi_.hline("mds", mds, color=GRAY, style=":", label="noise floor", label_pos=0.62)
        pi_.scatter("ipp", [p.iip3 + p.att], [p.iip3 + p.att], color=PURPLE, size=12, symbol="star",
                    name="intercept")
        pi_.scatter("now", [p.pin], [im3], color=ORANGE, size=13, symbol="d", name="your interferers")
        pa = self.plot("atp")
        pa.line("c", atts, np.clip(curve, -30, 40), color=NAVY, width=2.2, name="SINR")
        pa.scatter("now", [p.att], [np.clip(sinr, -30, 40)], color=ORANGE, size=13, symbol="d")
        best = float(curve.max())
        pa.vline("b", atts[int(np.argmax(curve))], color=GREEN, style="--", label="best", label_pos=0.9)
        # blocking: gain compression of the wanted by the two strong tones
        des = 20 * np.log10(abs(1 - 2 * 2 * A ** 2 / sh.dbm_to_amp(p.iip3) ** 2) + 1e-6)
        self.readout(im3=im3, sinr=sinr, sfdr=sfdr, des=des, best=best)

    def story(self, p):
        im3 = self.r.get("im3", 0)
        sinr = self.r.get("sinr", 0)
        s = (f"<p>Two strong stations at +{v(p.df, '.0f', 'kHz')} and +{v(2 * p.df, '.0f', 'kHz')} "
             f"pass through the front end's cubic nonlinearity and create a third-order product at "
             f"2f₁ − f₂: <b>exactly on your channel</b>, at {v(im3, '.0f', 'dBm')}. No filter after the "
             f"culprit can remove it.</p>"
             f"<p>IM3 rises 3 dB for every dB of input (red line, bottom left); the wanted signal only 1. "
             f"So an <b>attenuator</b> in front costs 1 dB of noise figure per dB but buys 2 dB of "
             f"intermodulation. Your SINR is {v(sinr, '.1f', 'dB')}; the best possible here is "
             f"{v(self.r.get('best', 0), '.1f', 'dB')}.</p>")
        if self.r.get("des", 0) < -1:
            s += (f"<p>{bad('Blocking:')} the strong tones also compress the receiver, shrinking your "
                  f"station by {v(-self.r.get('des', 0), '.1f', 'dB')}.</p>")
        return "<h3>Ghosts made inside the receiver</h3>" + s + keybox(
            "SFDR = ⅔(IIP3 − MDS). Every HF operator's \"ATT\" button exists because IM3 grows three "
            "times faster than the signal.")


# =============================================================================== 7. zero-IF vs low-IF
def blind_iq_correct(y):
    """Blind I/Q balance correction from second-order moments (circular signal assumed)."""
    i, q = y.real, y.imag
    pi_, pq = np.mean(i * i), np.mean(q * q)
    g = np.sqrt(pq / pi_)
    sphi = np.mean(i * q) / np.sqrt(pi_ * pq)
    phi = np.arcsin(np.clip(sphi, -0.99, 0.99))
    q2 = (q / g - i * np.sin(phi)) / np.cos(phi)
    return i + 1j * q2


@lru_cache(maxsize=4)
def _qpsk_signal(n, sps, seed):
    rng = np.random.default_rng(seed)
    con = cl.get_constellation("qpsk")
    nsym = n // sps + 40
    a = con.points[rng.integers(0, 4, nsym)]
    h = cl.rrc_taps(0.35, sps, 10)
    x = cl.shape(a, h, sps)[:n]
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), a, h


class ZeroIF(Experiment):
    title = "Zero-IF vs low-IF"
    blurb = "Integrated radios dropped the IF filter. The image came back as I/Q imbalance."
    book = "sec:ch07:zeroif"
    controls = [
        Choice("arch", "Architecture", ["Zero-IF", "Low-IF (+100 kHz)"], default="Low-IF (+100 kHz)"),
        Heading("Quadrature mixer"),
        Slider("g", "Gain imbalance", 0, 2, 0.5, step=0.05, unit="dB"),
        Slider("ph", "Phase error", 0, 10, 3, step=0.25, unit="°"),
        Slider("dc", "DC offset (LO leakage)", -60, -10, -40, step=1, unit="dBc"),
        Toggle("cal", "Digital I/Q correction (blind)", False),
        Heading("The band"),
        Slider("blk", "Blocker at the image position", 0, 50, 30, step=1, unit="dB",
               help="Strength of the neighbouring signal at −100 kHz, relative to the wanted"),
    ]
    plots = [
        SpectrumPlot("spec", "Spectrum after the quadrature mixer: yours (navy) vs a perfect one (grey)",
                     x="frequency (kHz)", y="power (dB re wanted)", xlim=(-250, 250), ylim=(-75, 60),
                     legend=None),
        ConstellationPlot("con", "Wanted QPSK after channel filtering", lim=1.8),
        Plot("irr", "Image rejection vs phase error", x="phase error (°)", y="IRR (dB)", xlim=(0, 10),
             ylim=(10, 70), legend="tr"),
    ]
    layout = [["spec", "spec"], ["con", "irr"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("irr", "Image rejection", "dB", ".1f"),
        Readout("img", "Image on the wanted", "dB re wanted", ".1f", good=lambda x: x < -25),
        Readout("mer", "MER", "dB", ".1f", good=lambda x: x > 20),
        Readout("dcr", "DC at the wanted", "", None),
    ]
    challenges = [
        Challenge("Low-IF with the +30 dB blocker: MER at least 20 dB by balancing the mixer better "
                  "(no digital correction).",
                  lambda s: s.p.arch.startswith("Low") and s.p.blk >= 30 and not s.p.cal and s.r.mer >= 20),
        Challenge("Same sloppy mixer (≥ 0.5 dB, ≥ 3°), same +30 dB blocker: get MER above 25 dB by "
                  "changing the architecture.",
                  lambda s: s.p.g >= 0.5 and s.p.ph >= 3 and s.p.blk >= 30 and s.r.mer > 25),
        Challenge("Calibration saves low-IF: 1 dB and 5° of imbalance, a +40 dB blocker, MER ≥ 20 dB.",
                  lambda s: s.p.arch.startswith("Low") and s.p.g >= 1 and s.p.ph >= 5 and s.p.blk >= 40
                  and s.r.mer >= 20),
        Challenge("Zero-IF's own enemy: with a perfect mixer, let DC offset alone push MER below 15 dB.",
                  lambda s: s.p.arch == "Zero-IF" and s.p.g == 0 and s.p.ph == 0 and s.r.mer < 15),
    ]
    fs, n, sps = 1.0e6, 1 << 15, 20

    def update(self, p):
        x, a, h = _qpsk_signal(self.n, self.sps, 3)
        b, _, _ = _qpsk_signal(self.n, self.sps, 4)
        f_if = 0.0 if p.arch == "Zero-IF" else 100e3
        t = np.arange(self.n) / self.fs
        f_blk = -100e3 if p.arch.startswith("Low") else -150e3
        wanted = x * np.exp(2j * np.pi * f_if * t)
        blocker = 10 ** (p.blk / 20) * b * np.exp(2j * np.pi * f_blk * t)
        r = wanted + blocker
        noise = 10 ** (-40 / 20) * (self.rng.standard_normal(self.n) + 1j * self.rng.standard_normal(self.n)) / np.sqrt(2)
        y = cl.iq_imbalance(r + noise, p.g, p.ph) + 10 ** (p.dc / 20)
        if p.cal:
            dcv = np.mean(y)
            y = blind_iq_correct(y - dcv) + dcv            # balance only; DC is left alone
        ideal = r + noise + 10 ** (p.dc / 20)
        fi, Pi = cl.welch_psd(ideal, self.fs, 2048)
        f, P = cl.welch_psd(y, self.fs, 2048)
        ref = np.max(Pi[np.abs(fi - f_if) < 15e3])
        ps = self.plot("spec")
        ps.line("ideal", fi / 1e3, Pi - ref, color=GRAY, width=1.0, name="perfect mixer")
        ps.line("y", f / 1e3, P - ref, color=NAVY, width=1.6, name="your mixer")
        ps.band("w", f_if / 1e3 - 34, f_if / 1e3 + 34, color=GREEN, alpha=0.10)
        ps.band("b", f_blk / 1e3 - 34, f_blk / 1e3 + 34, color=RED, alpha=0.07)
        ps.text("wl", f_if / 1e3, 56, "wanted", color=GREEN, size=9, anchor=(0.5, 0))
        ps.text("bl", f_blk / 1e3, 56, "blocker", color=RED, size=9, anchor=(0.5, 0))
        # receive: shift wanted to 0, matched filter, sample
        z = y * np.exp(-2j * np.pi * f_if * t)
        zf = np.convolve(z, h)
        d = len(h) - 1
        s_ = zf[d::self.sps][:len(a)]
        k = np.arange(20, min(len(s_), len(a)) - 20)
        s_ = s_[k]
        ak = a[k]
        gain = np.vdot(ak, s_) / np.vdot(ak, ak)
        s_ = s_ / gain
        err = s_ - ak
        mer = 10 * np.log10(np.mean(np.abs(ak) ** 2) / np.mean(np.abs(err) ** 2))
        pc = self.plot("con")
        pc.points("z", s_, color=NAVY, size=4, alpha=0.5)
        pc.ideal("i", cl.get_constellation("qpsk").points)
        # IRR: measured on a test tone through the same chain (with correction if enabled)
        if p.cal:
            irr = self.cal_irr(r + noise, p)
        else:
            irr = float(rf.irr_exact(p.g, p.ph)) if (p.g > 0 or p.ph > 0) else 80.0
        img = (p.blk - irr) if p.arch.startswith("Low") else -irr
        ph = np.linspace(0, 10, 101)
        pi_ = self.plot("irr")
        for i, (gg, col) in enumerate(((0.1, GREEN), (0.5, NAVY), (1.0, ORANGE), (2.0, RED))):
            pi_.line(f"g{i}", ph, np.minimum(rf.irr_exact(gg, ph), 69), color=col, width=1.4,
                     name=f"{gg:g} dB gain error")
        pi_.hline("need", p.blk + 20 if p.arch.startswith("Low") else 20, color=GRAY, style="--",
                  label="needed for 20 dB", label_pos=0.05)
        pi_.scatter("now", [p.ph], [min(irr, 69)], color=PURPLE, size=14, symbol="d",
                    name="yours" + (" (calibrated)" if p.cal else ""))
        dcr = "on the wanted" if p.arch == "Zero-IF" else "100 kHz away"
        self.readout(irr=irr, img=img, mer=mer, dcr=dcr)

    def cal_irr(self, r, p):
        """IRR after blind correction: estimate the correction on the received data, then apply
        the same correction to a test tone through the same imbalance."""
        y = cl.iq_imbalance(r, p.g, p.ph)
        i, q = y.real, y.imag
        pi_, pq = np.mean(i * i), np.mean(q * q)
        g = np.sqrt(pq / pi_)
        phi = np.arcsin(np.clip(np.mean(i * q) / np.sqrt(pi_ * pq), -0.99, 0.99))
        t = np.arange(4096) / self.fs
        yt = cl.iq_imbalance(np.exp(2j * np.pi * 50e3 * t), p.g, p.ph)
        yt = yt.real + 1j * (yt.imag / g - yt.real * np.sin(phi)) / np.cos(phi)
        T = np.fft.fft(yt)
        return float(min(10 * np.log10(np.abs(T[205]) ** 2 / max(np.abs(T[-205]) ** 2, 1e-30)), 80))

    def story(self, p):
        irr = self.r.get("irr", 0)
        s = ("<p>A quadrature mixer brings the band to baseband as I + jQ. If I and Q are not exactly "
             "equal in gain and 90° apart, every signal at +f leaks a mirror copy to −f, "
             f"{v(irr, '.1f', 'dB')} down (bottom right: IRR ≈ 4/(ε² + φ²)).</p>")
        if p.arch.startswith("Low"):
            s += (f"<p><b>Low-IF</b>: the wanted channel sits at +100 kHz, so DC offset and flicker noise "
                  f"are harmlessly far away. But the mirror of the {v(p.blk, '.0f', 'dB')} blocker at "
                  f"−100 kHz lands right on it, {v(self.r.get('img', 0), '.1f', 'dB')} relative to the "
                  "wanted. The superhet's image problem is back, solved now by balance instead of a "
                  "filter.</p>")
        else:
            s += ("<p><b>Zero-IF</b>: the wanted channel is centred on 0 Hz. Its image is its own "
                  "mirror, so imbalance only skews the constellation slightly, and blockers image onto "
                  "other blockers' positions. The price: DC offset and LO leakage land in the middle of "
                  "the channel (watch the constellation shift).</p>")
        if p.cal:
            s += ("<p>The digital correction measures I and Q powers and their correlation, then "
                  "undoes the imbalance: every modern RFIC (the B200's AD9364 included) runs a loop "
                  "like this.</p>")
        return "<h3>The image returns</h3>" + s + keybox(
            "Required IRR = blocker excess + SNR needed. 30 dB stronger neighbour + 15 dB SNR ⇒ 45 dB "
            "of image rejection: beyond analog matching, so calibrate.")


# =============================================================================== the lab
LAB = st.Lab(15, "The Superheterodyne Receiver", chapter=4,
             chapter_title="Analog Modulation and the Classic Radio (and Chapter 7)",
             experiments=[TuneBand, ImageProblem, IFSelectivity, AGCLoop, DoubleConversion,
                          Intermod, ZeroIF])

if __name__ == "__main__":
    st.run(LAB)
