"""Lab 01 · Complex Baseband, IQ Sampling and the SDR Receive Chain   (Chapters 2 and 7)

Run it:      python labs/lab01_baseband.py
Self-test:   python labs/lab01_baseband.py --selftest

Every radio you will ever build throws the carrier away as early as it can and does all its
thinking on a pair of low-rate signals, I and Q. Nine experiments make that trick concrete and
then walk down the receive chain of a real SDR (a B200-like zero-IF transceiver): the IQ helix,
why two ADCs, up- and down-conversion, the digital down-converter, DC and I/Q-imbalance
fingerprints, oscillator phase noise, ADC dynamic range, receiver sensitivity, and a live
waterfall of a crowded band.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as sps

import commlib as cl
from commlib import spectral as sp
from commlib import rf
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


def db(x, floor=1e-30):
    return 10 * np.log10(np.maximum(x, floor))


@lru_cache(maxsize=32)
def rrc(beta, sps_, span=10):
    return cl.rrc_taps(beta, sps_, span)


@lru_cache(maxsize=32)
def lowpass(ntaps, cutoff, fs):
    return sps.firwin(ntaps, cutoff, fs=fs)


def qpsk_symbols(n, seed):
    r = np.random.default_rng(seed)
    return (r.choice([-1.0, 1.0], n) + 1j * r.choice([-1.0, 1.0], n)) / np.sqrt(2)


def shaped(sym, sps_, beta=0.35, span=10):
    up = np.zeros(len(sym) * sps_, complex)
    up[::sps_] = sym
    return sps.fftconvolve(up, rrc(beta, sps_, span))


def fft_db(x, w=None):
    """Two-sided amplitude spectrum in dB re. a unit complex tone (fftshifted)."""
    w = np.blackman(len(x)) if w is None else w
    X = np.fft.fftshift(np.fft.fft(x * w)) / np.sum(w)
    return 20 * np.log10(np.abs(X) + 1e-9)


def mer_db(z, ref):
    """MER after the best complex gain (an ideal AGC + carrier phase), and that gain."""
    g = np.vdot(ref, z) / np.vdot(ref, ref)
    e = z - g * ref
    return 10 * np.log10(np.mean(np.abs(g * ref) ** 2) / max(np.mean(np.abs(e) ** 2), 1e-15)), g


# =============================================================================== 1. the IQ helix
SIGS = ["Complex tone", "Real cosine", "Two tones", "QPSK (RRC)"]


@lru_cache(maxsize=1)
def helix_qpsk():
    sym = qpsk_symbols(400, 4)
    x = shaped(sym, 50)                          # 4 Bd at 200 samples/s
    return x / np.sqrt(np.mean(np.abs(x) ** 2)) * 0.85


class IQHelix(Experiment):
    title = "The IQ helix"
    blurb = "A complex signal is a corkscrew in time: rotate it and see I, Q and the I/Q plane."
    book = "sec:complexenvelope"
    animate = True
    autoplay = True
    fps = 20
    controls = [
        Choice("sig", "Signal", SIGS, style="menu"),
        Slider("f", "Frequency", -4.0, 4.0, 1.0, step=0.25, unit="Hz",
               help="Rotation rate of the phasor. Negative = clockwise. For QPSK: a frequency offset"),
        Slider("az", "Viewing angle", 0, 90, 35, step=1, unit="°",
               help="0° = side view (time across), 90° = looking straight down the time axis"),
        Toggle("shadows", "Show the I and Q shadows", True),
    ]
    plots = [
        Plot("helix", "The signal in (time, I, Q)", x="", y="", xlim=(-2.25, 2.25),
             ylim=(-1.9, 2.5), aspect=True, grid=False, legend="tl", legend_cols=3),
        ConstellationPlot("iq", "The I/Q plane", lim=1.75),
        Plot("rails", "I(t) and Q(t): what the two ADCs see", x="time (s)", y="amplitude",
             xlim=(0, 2), ylim=(-1.9, 2.4), legend="tl", legend_cols=2),
        SpectrumPlot("spec", "Two-sided spectrum", x="frequency (Hz)", y="level (dB)",
                     xlim=(-6, 6), ylim=(-60, 5), legend=None),
    ]
    layout = [["helix", "helix", "iq"], ["rails", "rails", "spec"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("f", "Frequency", "Hz", ".2f"),
        Readout("rot", "Phasor turns", "", None),
        Readout("lines", "Spectral lines", "", None),
        Readout("sym", "Spectrum symmetric?", "", None),
    ]
    challenges = [
        Challenge("Make the phasor turn clockwise and look straight down the time axis (view ≥ 85°).",
                  lambda s: s.p.sig == "Complex tone" and s.p.f < 0 and s.p.az >= 85),
        Challenge("A real cosine is two counter-rotating phasors: make its two spectral lines 5 Hz apart.",
                  lambda s: s.p.sig == "Real cosine" and abs(abs(s.p.f) - 2.5) < 0.01),
        Challenge("QPSK: remove the frequency offset (f = 0) and look down the time axis (≥ 85°). "
                  "What shape is the trajectory?",
                  lambda s: s.p.sig == "QPSK (RRC)" and s.p.f == 0 and s.p.az >= 85),
    ]
    fs = 200.0

    def setup(self):
        self.k = 0
        h = self.plot("helix")
        h.pi.hideAxis("left")
        h.pi.hideAxis("bottom")

    def signal(self, p, t):
        if p.sig == "Complex tone":
            return np.exp(2j * np.pi * p.f * t)
        if p.sig == "Real cosine":
            return np.cos(2 * np.pi * p.f * t) + 0j
        if p.sig == "Two tones":
            return (np.exp(2j * np.pi * p.f * t) + 0.5 * np.exp(-2j * np.pi * 2.5 * t)) / 1.25
        q = helix_qpsk()
        n = np.round(t * self.fs).astype(int) + 600
        return q[n % len(q)] * np.exp(2j * np.pi * p.f * t)

    def proj(self, p, X, I, Q):
        a = np.deg2rad(p.az)
        e = np.deg2rad(18) * np.cos(a)
        sx = X * np.cos(a) + I * np.sin(a)
        depth = -X * np.sin(a) + I * np.cos(a)
        return sx, Q * np.cos(e) + depth * np.sin(e)

    def update(self, p):
        self.t = np.arange(0, 2.0, 1 / self.fs)
        self.z = self.signal(p, self.t)
        tl = np.arange(0, 20.48, 1 / 100)                 # long record for the spectrum
        if p.sig == "QPSK (RRC)":
            q = helix_qpsk()[::2][:len(tl)]                   # the same QPSK at 100 samples/s
            zl = q * np.exp(2j * np.pi * p.f * tl[:len(q)])
            f, P = sps.welch(zl, 100, nperseg=512, return_onesided=False)
            f, P = np.fft.fftshift(f), np.fft.fftshift(P)
            self.spec = (f, db(P / P.max()))
        else:
            zl = self.signal(p, tl)
            w = np.hanning(len(zl))
            Z = np.abs(np.fft.fftshift(np.fft.fft(zl * w, 8192))) ** 2
            f = np.fft.fftshift(np.fft.fftfreq(8192, 1 / 100))
            self.spec = (f, db(Z / Z.max()))
        self.draw(p)

    def tick(self, p):
        self.k += 1
        self.draw(p)

    def draw(self, p):
        t, z = self.t, self.z
        k = self.k % len(t)
        X = (t - 1.0) * 1.6
        h = self.plot("helix")
        ax = np.array([-1.85, 1.85])
        sx, sy = self.proj(p, ax, 0 * ax, 0 * ax)
        h.line("tax", sx, sy, color=GRAY, width=1.4)
        h.text("tl", sx[1], sy[1], " time →", color=GRAY, size=9, anchor=(0, 0.5))
        for nm, (I_, Q_) in {"I": ([0, 1.25], [0, 0]), "Q": ([0, 0], [0, 1.25])}.items():
            ax_x, ax_y = self.proj(p, np.full(2, -1.85), np.array(I_), np.array(Q_))
            h.line("a" + nm, ax_x, ax_y, color=GRAY, width=1.2, style=":")
            h.text("l" + nm, ax_x[1], ax_y[1], nm, color=GRAY, size=10, bold=True, anchor=(0.5, 1.0))
        if p.shadows:
            fx, fy = self.proj(p, X, z.real, np.full_like(X, -1.45))
            h.line("shI", fx, fy, color=RED, width=1.3, alpha=0.75, name="I(t) shadow")
            wx, wy = self.proj(p, X, np.full_like(X, 1.45), z.imag)
            h.line("shQ", wx, wy, color=GREEN, width=1.3, alpha=0.75, name="Q(t) shadow")
        hx, hy = self.proj(p, X, z.real, z.imag)
        h.line("hel", hx, hy, color=NAVY, width=2.6, name="I + jQ")
        h.scatter("dot", [hx[k]], [hy[k]], color=ORANGE, size=13)
        # I/Q plane
        pq = self.plot("iq")
        th = np.linspace(0, 2 * np.pi, 200)
        pq.line("uc", np.cos(th), np.sin(th), color=GRAY, width=1.0, style=":")
        pq.line("traj", z.real, z.imag, color=NAVY, width=1.6)
        if p.sig == "Real cosine":
            ph = 2 * np.pi * p.f * t[k]
            a, b = 0.5 * np.exp(1j * ph), 0.5 * np.exp(-1j * ph)
            pq.line("pa", [0, a.real], [0, a.imag], color=PURPLE, width=2.4)
            pq.line("pb", [0, b.real], [0, b.imag], color=TEAL, width=2.4)
            pq.line("pab", [a.real, a.real + b.real], [a.imag, a.imag + b.imag], color=TEAL, width=1.2,
                    style="--")
        pq.line("ph", [0, z[k].real], [0, z[k].imag], color=ORANGE, width=2.6)
        pq.scatter("pd", [z[k].real], [z[k].imag], color=ORANGE, size=12)
        # rails
        pr = self.plot("rails")
        pr.hline("z", 0, color=GRAY, style="-", width=0.6)
        pr.line("I", t, z.real, color=RED, width=2.0, name="I(t) = Re")
        pr.line("Q", t, z.imag, color=GREEN, width=2.0, name="Q(t) = Im")
        pr.vline("now", t[k], color=ORANGE, style="--", width=1.4)
        # spectrum
        f, P = self.spec
        ps = self.plot("spec")
        ps.line("P", f, P, color=NAVY, width=1.8, fill=-60, fill_alpha=0.12)
        ps.vline("zero", 0, color=GRAY, style=":")
        if p.sig == "Complex tone":
            rot = "anticlockwise" if p.f > 0 else "clockwise" if p.f < 0 else "not at all"
            lines, sym = "1", "no" if p.f else "yes"
        elif p.sig == "Real cosine":
            rot, lines, sym = "back and forth", "2" if p.f else "1", "yes"
        elif p.sig == "Two tones":
            rot, lines, sym = "two ways at once", "2", "no"
        else:
            rot = "spins with f" if p.f else "follows the symbols"
            lines, sym = "a band", "nearly" if p.f == 0 else "no"
        self.readout(f=p.f, rot=rot, lines=lines, sym=sym)

    def story(self, p):
        s = ("<p>Plot a complex signal against time and it becomes a <b>helix</b>: the in-phase part "
             "I is its shadow on the floor (red), the quadrature part Q its shadow on the wall "
             "(green). Turn the viewing angle: from the side you see a sine wave; looking down the "
             "time axis you see the phasor going round the I/Q plane.</p>")
        if p.sig == "Complex tone":
            s += (f"<p>e^(j2π·{p.f:g}·t) turns {v(abs(p.f), '.2f', 'Hz')} times a second, "
                  + ("clockwise: a <b>negative frequency</b>. It has a single spectral line at "
                     f"{p.f:g} Hz, which a real signal could never have." if p.f < 0 else
                     "anticlockwise. One spectral line, on one side only.") + "</p>")
        elif p.sig == "Real cosine":
            s += ("<p>A real cosine has no Q at all: the 'helix' is flat. In the I/Q plane it is two "
                  "half-length phasors turning in opposite directions (purple and teal) whose sum "
                  "always lands on the I axis. Hence two spectral lines, at +f and −f, with the "
                  "same height: real signals always have symmetric spectra.</p>")
        elif p.sig == "Two tones":
            s += (f"<p>Two tones, at {p.f:g} Hz and −2.5 Hz, unequal: the spectrum is lopsided. Only "
                  f"a complex signal can do that, which is why a receiver needs both I and Q to "
                  f"tell a signal above the carrier from one below it.</p>")
        else:
            s += ("<p>RRC-shaped QPSK: the phasor wanders between four constellation points. A "
                  "frequency offset makes the whole pattern spin, which is exactly what an untuned "
                  "receiver sees before carrier recovery (Chapter 10).</p>")
        return "<h3>Two real signals, one complex one</h3>" + s + keybox(
            "x̃ = I + jQ. A single complex number per sample carries amplitude and phase, and the "
            "sign of frequency.")


# =============================================================================== 2. real vs IQ sampling
class RealVsIQ(Experiment):
    title = "Why two ADCs: real vs IQ sampling"
    blurb = "Throw away Q and every signal gets a mirror image. Sample too slowly and they wrap."
    book = "sec:ch07:zeroif"
    controls = [
        Heading("Signals around the carrier"),
        Slider("f1", "Tone 1 offset", -480, 480, 100, step=5, unit="kHz"),
        Slider("f2", "Tone 2 offset", -480, 480, -250, step=5, unit="kHz"),
        Slider("a2", "Tone 2 level", -40, 0, -6, step=1, unit="dB"),
        Heading("Out-of-band interferer"),
        Toggle("intf", "Add an interferer", False,
               help="A signal outside ±500 kHz that the anti-alias filter failed to remove"),
        Slider("fi", "Interferer offset", -2500, 2500, 700, step=5, unit="kHz",
               enabled_if=lambda p: p.intf),
    ]
    plots = [
        SpectrumPlot("cx", "With I and Q: complex sampling at 1 MS/s", x="frequency offset (kHz)",
                     y="level (dB)", xlim=(-520, 520), ylim=(-70, 8), legend=None),
        SpectrumPlot("re", "I only: real sampling at 1 MS/s", x="frequency offset (kHz)",
                     y="level (dB)", xlim=(-520, 520), ylim=(-70, 8), legend=None),
        Plot("map", "Where a tone lands", x="true offset from the carrier (kHz)",
             y="where it appears (kHz)", xlim=(-2600, 2600), ylim=(-560, 680), legend="tl",
             legend_cols=2),
    ]
    layout = [["cx", "map"], ["re", "map"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("ncx", "Lines with I and Q", "", "int"),
        Readout("nre", "Lines with I only", "", "int"),
        Readout("app", "Interferer appears at", "kHz", ".0f"),
        Readout("sep", "I-only separation", "kHz", ".0f", good=lambda x: x > 20),
    ]
    challenges = [
        Challenge("Make the two tones collide in the I-only receiver while they stay 200 kHz or "
                  "more apart with I and Q.",
                  lambda s: abs(abs(s.p.f1) - abs(s.p.f2)) < 2 and abs(s.p.f1 - s.p.f2) >= 200),
        Challenge("Chapter 7: a B200 at 1 MS/s and an interferer at +700 kHz. Put tone 1 exactly where "
                  "it lands.",
                  lambda s: s.p.intf and abs(s.p.fi - 700) < 3 and abs(s.p.f1 + 300) < 3),
        Challenge("Find an interferer more than 1 MHz away that lands exactly on 0 Hz.",
                  lambda s: s.p.intf and abs(s.p.fi) > 1000 and abs(sp.alias_complex(s.p.fi, 1000)) < 3),
    ]
    fs, N = 1000.0, 4096

    def update(self, p):
        n = np.arange(self.N)
        a2 = 10 ** (p.a2 / 20)
        z = np.exp(2j * np.pi * p.f1 * n / self.fs) + a2 * np.exp(2j * np.pi * p.f2 * n / self.fs + 1.0)
        if p.intf:
            z = z + 0.7 * np.exp(2j * np.pi * p.fi * n / self.fs + 2.0)
        rng = np.random.default_rng(1)
        z = z + 3e-4 * (rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N))
        f = np.fft.fftshift(np.fft.fftfreq(self.N, 1 / self.fs))
        pc, pr = self.plot("cx"), self.plot("re")
        pc.line("S", f, fft_db(z), color=NAVY, width=1.8, fill=-70, fill_alpha=0.10)
        pr.line("S", f, fft_db(z.real + 0j), color=RED, width=1.8, fill=-70, fill_alpha=0.10)
        for pl in (pc, pr):
            pl.vline("c", 0, color=GRAY, style=":", label="carrier", label_pos=0.95)
        if p.intf:
            app = float(sp.alias_complex(p.fi, self.fs))
            pc.vline("ia", app, color=PURPLE, style="--", label="interferer", label_pos=0.85)
        else:
            app = None
        # the folding map
        x = np.linspace(-2600, 2600, 2601)
        pm = self.plot("map")
        pm.hband("nyq", -500, 500, color=GREEN, alpha=0.06)
        pm.line("cx", x, sp.alias_complex(x, self.fs), color=NAVY, width=2.0, name="I and Q: wraps")
        pm.line("re", x, sp.alias_real(x, self.fs), color=RED, width=1.6, style="--",
                name="I only: folds to |f|")
        pm.scatter("t1", [p.f1], [p.f1], color=GREEN, size=12, symbol="d", name="tone 1")
        pm.scatter("t2", [p.f2], [p.f2], color=ORANGE, size=12, symbol="d", name="tone 2")
        if p.intf:
            pm.scatter("ti", [p.fi], [app], color=PURPLE, size=14, symbol="star", name="interferer")
        lines_cx = 2 if abs(p.f1 - p.f2) > 2 else 1
        lines_re = (len({abs(round(p.f1)), abs(round(p.f2))}) * 2
                    - (1 if p.f1 == 0 else 0) - (1 if p.f2 == 0 else 0))
        if p.intf:
            lines_cx += 1
            lines_re += 2
        self.readout(ncx=lines_cx, nre=lines_re, app=app if app is not None else "—",
                     sep=abs(abs(p.f1) - abs(p.f2)))

    def story(self, p):
        r = self.r
        s = ("<p>A zero-IF receiver mixes the band down so the carrier sits at 0 Hz. With both I "
             "and Q (top) every tone keeps its sign: +100 kHz is above the carrier, −250 kHz below. "
             "Keep only I (bottom) and the signal is real, so its spectrum must be symmetric: every "
             "tone appears twice, at half the amplitude (−6 dB).</p>")
        if r.get("sep", 999) < 3:
            s += ("<p>" + bad("Collision.") + " Your two tones sit at ±the same offset: with I only "
                  "they land on top of each other and cannot be separated. With I and Q they are "
                  "still distinct. This is the image problem that the Q branch solves.</p>")
        if p.intf:
            s += (f"<p>The interferer at {v(p.fi, '.0f', 'kHz')} is outside the ±500 kHz that 1 MS/s "
                  f"complex sampling can represent, so it <b>wraps</b> to "
                  f"{v(r.get('app', 0), '.0f', 'kHz')} (right: the blue sawtooth). Real sampling would "
                  f"<b>fold</b> instead (red). Either way it is now indistinguishable from a real signal "
                  f"there: only the analog anti-alias filter can stop it.</p>")
        return "<h3>Mirrors and wraps</h3>" + s + keybox(
            "Complex sampling at fs covers −fs/2…fs/2 without mirrors. Out-of-band signals wrap: "
            "apparent f = ((f + fs/2) mod fs) − fs/2.")


# =============================================================================== 3. up/down conversion
@lru_cache(maxsize=1)
def updown_qpsk():
    sym = qpsk_symbols(240, 9)
    x = shaped(sym, 40)                                   # 500 Bd at 20 kHz
    return sym, x / np.sqrt(np.mean(np.abs(x) ** 2))


class UpDown(Experiment):
    title = "Up- and down-conversion"
    blurb = "Ride a carrier, come back down. What do the mixer, the filter and the LO have to get right?"
    book = "sec:complexenvelope"
    controls = [
        Choice("msg", "Baseband signal", ["Two tones (+100, −250 Hz)", "QPSK (500 Bd)"], style="menu"),
        Slider("fc", "Carrier frequency", 0.2, 6.0, 4.0, step=0.05, unit="kHz"),
        Heading("Receiver"),
        Slider("cut", "Low-pass cut-off", 200, 9000, 1000, step=50, unit="Hz"),
        Slider("phase", "LO phase error", -180, 180, 0, step=1, unit="°"),
        Slider("df", "LO frequency error", -20, 20, 0, step=0.5, unit="Hz"),
    ]
    plots = [
        SpectrumPlot("pb", "Real passband x(t) = Re{x̃·e^(j2πf_c t)}", x="frequency (kHz)",
                     y="level (dB)", xlim=(-10, 10), ylim=(-80, 8), legend=None),
        SpectrumPlot("mix", "After the mixer (grey) and the low-pass filter (green)",
                     x="frequency (kHz)", y="level (dB)", xlim=(-10, 10), ylim=(-80, 8), legend="tr"),
        ConstellationPlot("con", "Recovered complex envelope", lim=1.75),
        Plot("time", "I and Q: sent (thick, grey) and recovered", x="time (ms)", y="amplitude",
             xlim=(0, 40), ylim=(-2.0, 2.6), legend="tl", legend_cols=2),
    ]
    layout = [["pb", "con"], ["mix", "con"], ["time", "time"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("leak", "−2f_c copy after the filter", "dB", ".0f", good=lambda x: x < -40),
        Readout("rot", "Constellation rotation", "°", ".0f"),
        Readout("spin", "Rotation rate", "turns/s", ".1f"),
        Readout("mer", "Fidelity (MER)", "dB", ".1f", good=lambda x: x > 25),
    ]
    challenges = [
        Challenge("Let the −2f_c copy leak through: open the filter until the copy is less than 20 dB down.",
                  lambda s: s.r.leak > -20),
        Challenge("Rotate the recovered QPSK by exactly 45° using only the LO phase.",
                  lambda s: s.p.msg.startswith("QPSK") and abs(abs(s.p.phase) - 45) < 0.5 and s.p.df == 0),
        Challenge("Lower the carrier until the signal overlaps its own mirror and the QPSK is ruined "
                  "(MER below 10 dB) even with a perfect LO.",
                  lambda s: s.p.msg.startswith("QPSK") and s.p.phase == 0 and s.p.df == 0 and s.r.mer < 10,
                  hint="A real passband signal needs f_c larger than half its bandwidth."),
    ]
    fs, N = 20000.0, 9600

    def update(self, p):
        fs, N = self.fs, self.N
        t = np.arange(N) / fs
        if p.msg.startswith("Two"):
            xb = np.exp(2j * np.pi * 100 * t) + 0.5 * np.exp(-2j * np.pi * 250 * t)
        else:
            sym, xs = updown_qpsk()
            xb = xs[:N]
        fc = p.fc * 1e3
        xp = np.real(xb * np.exp(2j * np.pi * fc * t))
        mixed = 2 * xp * np.exp(-1j * (2 * np.pi * (fc + p.df) * t + np.deg2rad(p.phase)))
        h = lowpass(201, float(p.cut), fs)
        xr = sps.fftconvolve(mixed, h)[100:100 + N]
        f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs)) / 1e3
        w = np.blackman(N)
        spec = lambda x: 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x * w))) / np.sum(w) + 1e-9)
        ppb = self.plot("pb")
        ppb.line("P", f, spec(xp + 0j), color=NAVY, width=1.6, fill=-80, fill_alpha=0.12)
        ppb.vline("c1", p.fc, color=RED, style=":", label="+f_c", label_pos=0.95)
        ppb.vline("c2", -p.fc, color=RED, style=":", label="−f_c", label_pos=0.95)
        pm = self.plot("mix")
        Sm, Sr = spec(mixed), spec(xr)
        pm.line("m", f, Sm, color=GRAY, width=1.2, name="after the mixer")
        pm.line("r", f, Sr, color=GREEN, width=2.0, name="after the low-pass")
        fh = np.linspace(-10, 10, 801)
        _, H = sps.freqz(h, worN=fh * 1e3, fs=fs)
        pm.line("H", fh, 20 * np.log10(np.abs(H) + 1e-9), color=ORANGE, width=1.2, style="--",
                name="filter response")
        # leakage of the -2fc copy
        fk = f * 1e3
        P = np.abs(np.fft.fft(xr)) ** 2
        P = np.fft.fftshift(P)
        near0 = np.abs(fk) < 600
        near2 = np.abs(fk + 2 * fc + p.df) < 600
        leak = db(P[near2 & ~near0].sum() / max(P[near0].sum(), 1e-12)) if (near2 & ~near0).any() else 0.0
        pc = self.plot("con")
        th = np.linspace(0, 2 * np.pi, 100)
        pc.line("uc", np.cos(th), np.sin(th), color=GRAY, width=1.0, style=":")
        a0, a1 = 1200, N - 1200
        if p.msg.startswith("Two"):
            pc.line("ref", xb[a0:a0 + 400].real, xb[a0:a0 + 400].imag, color=GRAY, width=3.0, alpha=0.5)
            pc.line("rx", xr[a0:a0 + 400].real, xr[a0:a0 + 400].imag, color=NAVY, width=1.6)
            mer, g = mer_db(xr[a0:a1], xb[a0:a1])
        else:
            sym, _ = updown_qpsk()
            hr = rrc(0.35, 40, 10)
            mf = sps.fftconvolve(xr, hr)[len(hr) - 1:]          # symbol k now at index 40·k
            ks = np.arange(20, N // 40 - 20)
            z = mf[ks * 40]
            mer, g = mer_db(z, sym[ks])
            pc.points("z", z, color=NAVY, size=6, alpha=0.8)
            pc.ideal("ideal", np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2), color=GRAY)
        k = (t >= 0) & (t <= 0.041)
        pt = self.plot("time")
        pt.line("Ir", t[k] * 1e3, xb[k].real, color=GRAY, width=4.0, alpha=0.35)
        pt.line("Qr", t[k] * 1e3, xb[k].imag, color=GRAY, width=4.0, alpha=0.35)
        pt.line("I", t[k] * 1e3, xr[k].real, color=RED, width=1.8, name="I recovered")
        pt.line("Q", t[k] * 1e3, xr[k].imag, color=GREEN, width=1.8, name="Q recovered")
        rot = round(float(np.rad2deg(np.angle(g))), 1) + 0.0 if p.df == 0 else None
        self.readout(leak=leak, rot=rot if rot is not None else "spinning", spin=p.df, mer=mer)

    def story(self, p):
        r = self.r
        s = ("<p>The transmitter multiplies the complex envelope by e^(j2πf_c t) and keeps the real "
             "part: one copy at +f_c and its mirror at −f_c (top left). The receiver multiplies by "
             "2·e^(−j2πf_c t): the wanted copy drops to 0 Hz and the mirror lands at −2f_c, where "
             "the low-pass filter (orange) removes it.</p>")
        if r.get("leak", -99) > -20:
            s += ("<p>" + bad("The filter is too wide") + ": the −2f_c copy leaks through and "
                  "the recovered I and Q carry a ripple at twice the carrier.</p>")
        if p.phase != 0 and p.df == 0:
            s += (f"<p>The LO phase error of {v(p.phase, '.0f', '°')} rotates everything by the same "
                  f"angle: I and Q are mixed into each other, nothing is lost. Carrier recovery "
                  f"(Chapter 10) exists to undo exactly this.</p>")
        if p.df != 0:
            s += (f"<p>A frequency error of {v(p.df, '.1f', 'Hz')} makes the rotation grow with time: "
                  f"the constellation spins {v(abs(p.df), '.1f')} turns per second. Even 1 ppm at "
                  f"2.4 GHz is 2.4 kHz, so every receiver needs a frequency-locked loop.</p>")
        if p.fc * 1e3 < 700 and p.msg.startswith("QPSK"):
            s += ("<p>" + bad("Carrier too low") + ": the QPSK band (about ±340 Hz) overlaps its "
                  "own mirror image around 0 Hz in the real signal. No receiver can separate them.</p>")
        return "<h3>Up, down, and what can go wrong</h3>" + s + keybox(
            "x(t) = Re{x̃(t)·e^(j2πf_c t)}. Down-conversion: multiply by 2e^(−j2πf_c t) and low-pass. "
            "LO phase rotates, LO frequency spins.")


# =============================================================================== 4. DDC
@lru_cache(maxsize=1)
def ddc_capture():
    fs = 8e6
    r = np.random.default_rng(1)
    sym = qpsk_symbols(3000, 2)
    w = shaped(sym, 16)                                   # 500 kBd
    N = len(w)
    n = np.arange(N)
    nb = shaped(qpsk_symbols(N // 32 + 20, 3), 32)[:N] * np.sqrt(10)     # +10 dB neighbour, 250 kBd
    m = np.sin(2 * np.pi * 15e3 * n / fs)
    x = (w * np.exp(2j * np.pi * -2e6 * n / fs)
         + nb * np.exp(2j * np.pi * -2.75e6 * n / fs)
         + 0.5 * np.exp(2j * np.pi * 1.5e6 * n / fs)
         + 0.3 * np.exp(1j * (2 * np.pi * 3e6 * n / fs + 2 * np.pi * 75e3 * np.cumsum(m) / fs)))
    x = x + np.sqrt(1e-3 / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))
    return sym, x


class DigitalDownConverter(Experiment):
    title = "The digital down-converter"
    blurb = "Tune, filter, decimate: pull one channel out of an 8 MHz-wide capture."
    book = "sec:ch07:sdr"
    controls = [
        Slider("nco", "NCO frequency", -4.0, 4.0, -1.8, step=0.01, unit="MHz",
               help="The digital oscillator that moves the wanted channel to 0 Hz"),
        Choice("D", "Decimation", ["2", "4", "8"], "8", help="Output rate = 8 MS/s ÷ D"),
        IntSlider("taps", "Filter length", 15, 257, 129, step=2, unit="taps"),
        Slider("cut", "Filter cut-off", 100, 1000, 450, step=10, unit="kHz"),
    ]
    plots = [
        SpectrumPlot("wide", "Wideband capture, 8 MS/s", x="frequency (MHz)", y="level (dB)",
                     xlim=(-4, 4), ylim=(-75, 18), legend="tr"),
        SpectrumPlot("out", "After the DDC", x="frequency (kHz)", y="level (dB)", ylim=(-75, 18),
                     legend=None),
        ConstellationPlot("con", "Recovered QPSK symbols", lim=1.8),
    ]
    layout = [["wide", "wide"], ["out", "con"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("rate", "Output sample rate", "MS/s", ".2f"),
        Readout("rej", "Alias rejection", "dB", ".0f", good=lambda x: x < -50,
                help="Worst filter response at frequencies that fold onto the wanted channel"),
        Readout("gain", "Noise removed", "dB", ".1f", help="10·log₁₀(D): the processing gain"),
        Readout("mer", "QPSK MER", "dB", ".1f", good=lambda x: x > 20),
    ]
    challenges = [
        Challenge("Tune the NCO onto the QPSK channel: MER above 25 dB.",
                  lambda s: s.r.mer > 25),
        Challenge("Decimate by 8 with a filter of 25 taps or fewer: find the cut-off that lifts the "
                  "MER above 19 dB.",
                  lambda s: s.p.D == "8" and s.p.taps <= 25 and s.r.mer > 19,
                  hint="A short filter has a lazy skirt: move its cut-off to protect the channel."),
        Challenge("Show aliasing: on target, but let the strong neighbour fold into your channel "
                  "(MER below 12 dB at D = 8).",
                  lambda s: s.p.D == "8" and abs(s.p.nco + 2) < 0.02 and s.r.mer < 12),
    ]
    fs = 8e6

    def update(self, p):
        sym, x = ddc_capture()
        D = int(p.D)
        n = np.arange(len(x))
        mixed = x * np.exp(-2j * np.pi * p.nco * 1e6 * n / self.fs)
        h = lowpass(int(p.taps), float(p.cut) * 1e3, self.fs)
        y = sps.fftconvolve(mixed, h)[(len(h) - 1) // 2:][:len(x)][::D]
        fso = self.fs / D
        # recover the symbols
        s2 = 16 // D
        hr = rrc(0.35, s2, 10)
        mf = sps.fftconvolve(y, hr)[len(hr) - 1:]
        ks = np.arange(20, 2900)                         # TX + RX filter delays removed above
        z = mf[ks * s2]
        mer, g = mer_db(z, sym[ks])
        zn = z * np.exp(-1j * np.angle(g)) / np.sqrt(np.mean(np.abs(z) ** 2))   # AGC + phase only
        # plots
        pw = self.plot("wide")
        pw.psd("P", x, self.fs, 2048, scale=1e6, color=NAVY, width=1.4, name="capture", normalize=False)
        f = np.linspace(-4, 4, 1601)
        _, H = sps.freqz(h, worN=(f - p.nco) * 1e6, fs=self.fs)
        Hd = 20 * np.log10(np.abs(H) + 1e-9)
        Pw = cl.welch_psd(x, self.fs, 2048)[1]
        top = float(np.max(Pw)) + 3
        pw.line("H", f, Hd + top, color=ORANGE, width=1.6, style="--", name="DDC filter")
        pw.band("out", p.nco - fso / 2e6, p.nco + fso / 2e6, color=GREEN, alpha=0.10)
        pw.vline("nco", p.nco, color=RED, style="-", width=1.6, label="NCO", label_pos=0.06)
        pw.set_ylim(top - 95, top + 12)
        for lab, fx in [("QPSK", -2.0), ("neighbour", -2.75), ("carrier", 1.5), ("FM", 3.0)]:
            pw.text("t" + lab, fx, None, lab, color=PURPLE, size=8.5, anchor=(0.5, 0))
        po = self.plot("out")
        po.psd("P", y, fso, 1024, scale=1e3, color=GREEN, width=1.6, normalize=False)
        Py = cl.welch_psd(y, fso, 1024)[1]
        po.set_xlim(-fso / 2e3, fso / 2e3)
        po.set_ylim(float(Py.max()) - 85, float(Py.max()) + 8)
        pc = self.plot("con")
        pc.points("z", zn[:1500] * (1 + 1j) / np.sqrt(2), color=NAVY, size=4, alpha=0.5)
        pc.ideal("i", np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2), color=RED)
        # what lands in the wanted channel (±337.5 kHz) after decimation comes from f ≥ fso − 337.5 kHz
        fz = np.linspace(0, self.fs / 2, 4000)
        _, Hz = sps.freqz(h, worN=fz, fs=self.fs)
        stop = fz >= fso - 337.5e3
        rej = float(20 * np.log10(np.max(np.abs(Hz[stop])) + 1e-12)) if stop.any() else -200
        self.readout(rate=fso / 1e6, rej=rej, gain=10 * np.log10(D), mer=mer)

    def story(self, p):
        r = self.r
        D = int(p.D)
        s = (f"<p>An SDR digitises a wide slice of spectrum (8 MHz here) and selects a channel in the "
             f"FPGA or on your CPU. A digital down-converter does three things: an <b>NCO</b> "
             f"multiplies by e^(−j2πf_NCO·n/fs) to move the channel to 0 Hz, a <b>low-pass filter</b> "
             f"rejects everything else, and <b>decimation</b> keeps every {v(D, 'd')}th sample.</p>")
        if abs(p.nco + 2.0) > 0.05:
            s += (f"<p>The NCO is at {v(p.nco, '.2f', 'MHz')} but the QPSK sits at −2.00 MHz, so the "
                  f"constellation spins or vanishes: MER {v(r.get('mer', 0), '.1f', 'dB')}.</p>")
        else:
            s += (f"<p>On target. The filter's stop-band, {v(r.get('rej', 0), '.0f', 'dB')}, decides "
                  f"how much of the +10 dB neighbour 750 kHz away survives. After decimation it has "
                  f"nowhere to go but to <b>alias</b> into your band, because the output only spans "
                  f"±{4000 / D:.0f} kHz.</p>")
        s += (f"<p>Decimating by {D} discards {v(10 * np.log10(D), '.1f', 'dB')} of noise bandwidth: "
              f"the <b>processing gain</b> that lets a 12-bit ADC behave like a better one in a narrow "
              f"channel. Chapter 6 and Lab 17 make this efficient with polyphase and CIC filters.</p>")
        return "<h3>Tune, filter, decimate</h3>" + s + keybox(
            "Filter before you decimate: whatever the filter lets through beyond the new Nyquist "
            "band folds onto your signal.")


# =============================================================================== 5. zero-IF impairments
@lru_cache(maxsize=1)
def zeroif_signals():
    N, s_ = 1 << 15, 16
    q16 = cl.get_constellation("16qam")
    r = np.random.default_rng(31)
    sym = q16.points[r.integers(0, 16, N // s_ + 20)]
    sym = sym / np.sqrt(np.mean(np.abs(q16.points) ** 2))
    w = shaped(sym, s_, beta=0.25)[:N]
    nb = shaped(qpsk_symbols(N // 12 + 20, 32), 12, beta=0.25)[:N]      # its own clock: 1.67 MBd
    noise = (r.standard_normal(N) + 1j * r.standard_normal(N)) * np.sqrt(10 ** -3.5 / 2)
    return sym, w / np.sqrt(np.mean(np.abs(w) ** 2)), nb / np.sqrt(np.mean(np.abs(nb) ** 2)), noise


class ZeroIFImpairments(Experiment):
    title = "Zero-IF fingerprints: DC and I/Q images"
    blurb = "A neighbour's mirror image lands on your channel. How much image rejection do you need?"
    book = "sec:ch07:iq"
    controls = [
        Heading("Receiver imperfections"),
        Slider("g", "Gain imbalance", 0.0, 2.0, 0.5, step=0.05, unit="dB"),
        Slider("ph", "Phase imbalance", 0.0, 10.0, 3.0, step=0.1, unit="°"),
        Slider("dc", "DC offset", -60, 10, -5, step=1, unit="dBc"),
        Heading("Signals"),
        Slider("off", "Your channel offset", 0.5, 8.0, 3.0, step=0.05, unit="MHz"),
        Slider("nb", "Mirror neighbour level", -20, 40, 15, step=1, unit="dB",
               help="A stronger signal at exactly minus your offset"),
        Heading("Correction"),
        Toggle("dcfix", "Remove DC", False),
        Toggle("iqfix", "Blind I/Q correction", False),
    ]
    plots = [
        SpectrumPlot("spec", "What the zero-IF receiver delivers to its ADCs", x="frequency (MHz)",
                     y="PSD re. wanted (dB)", xlim=(-10, 10), ylim=(-45, 55), legend=None),
        ConstellationPlot("con", "Your 16-QAM after the receiver", lim=1.5),
        Plot("irr", "Image rejection vs phase imbalance", x="phase imbalance (°)",
             y="image rejection (dB)", xlim=(0, 10), ylim=(10, 75), legend="tr"),
    ]
    layout = [["spec", "spec"], ["con", "irr"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("irr", "Hardware IRR", "dB", ".1f"),
        Readout("irrm", "IRR after correction", "dB", ".1f"),
        Readout("img", "Neighbour's image vs you", "dB", ".1f", good=lambda x: x < -20),
        Readout("mer", "Your MER", "dB", ".1f", good=lambda x: x > 20),
    ]
    challenges = [
        Challenge("With a neighbour 30 dB stronger, get your MER above 20 dB without improving the "
                  "hardware (≥ 0.5 dB and ≥ 3°).",
                  lambda s: s.p.nb >= 30 and s.p.g >= 0.5 and s.p.ph >= 3 and s.r.mer > 20),
        Challenge("Phase error alone (gain error ≤ 0.02 dB): find the phase imbalance that gives an "
                  "IRR of 30 dB.",
                  lambda s: s.p.g <= 0.02 and abs(s.r.irr - 30) < 0.3),
        Challenge("Correct a terrible receiver (2 dB, 10°) to an IRR above 45 dB.",
                  lambda s: s.p.g >= 1.99 and s.p.ph >= 9.99 and s.r.irrm > 45),
    ]
    fs = 20e6

    def update(self, p):
        sym, w, nb, noise = zeroif_signals()
        N = len(w)
        n = np.arange(N)
        fo = p.off * 1e6 / self.fs
        x = w * np.exp(2j * np.pi * fo * n) + 10 ** (p.nb / 20) * nb * np.exp(-2j * np.pi * fo * n + 0.7j)
        y = cl.iq_imbalance(x, p.g, p.ph) + 10 ** (p.dc / 20) * (0.8 + 0.6j) + noise
        if p.iqfix:
            y, ge, pe = sp.blind_iq_correct(y)
        elif p.dcfix:
            y = y - np.mean(y)
        mu, nu, c = sp.iq_fit(y, x)
        irrm = float(db(abs(mu) ** 2 / max(abs(nu) ** 2, 1e-30)))
        irr = float(rf.irr_exact(p.g + 1e-9, p.ph + 1e-9))
        # demodulate the wanted channel
        z = y * np.exp(-2j * np.pi * fo * n)
        h = rrc(0.25, 16, 10)
        mf = sps.fftconvolve(z, h)[len(h) - 1:]
        ks = np.arange(20, N // 16 - 30)
        zz = mf[ks * 16]
        mer, g = mer_db(zz, sym[ks])
        # spectrum relative to the wanted signal's in-band PSD
        f, P = cl.welch_psd(y, self.fs, 1024)
        ref = self.wanted_ref()
        ps = self.plot("spec")
        ps.band("you", p.off - 0.78, p.off + 0.78, color=GREEN, alpha=0.12)
        ps.band("nbr", -p.off - 1.04, -p.off + 1.04, color=ORANGE, alpha=0.10)
        ps.line("P", f / 1e6, P - ref, color=NAVY, width=1.4)
        ps.text("ty", p.off, None, "you", color=GREEN, size=9, bold=True, anchor=(0.5, 0))
        ps.text("tn", -p.off, None, "neighbour", color=ORANGE, size=9, bold=True, anchor=(0.5, 0))
        if not (p.dcfix or p.iqfix):
            ps.text("tdc", 0.2, max(-40.0, float(P[np.argmin(np.abs(f))] - ref)) + 2, "DC / LO leakage",
                    color=RED, size=8.5, anchor=(0, 1))
        pc = self.plot("con")
        pc.points("z", zz / g * 1.0, color=NAVY, size=3, alpha=0.45)
        q = cl.get_constellation("16qam").points
        pc.ideal("i", q / np.sqrt(np.mean(np.abs(q) ** 2)), color=RED)
        # IRR curves
        phs = np.linspace(0.02, 10, 300)
        pi_ = self.plot("irr")
        for gd, col in [(0.0, NAVY), (0.1, GREEN), (0.5, ORANGE), (1.0, PURPLE), (2.0, RED)]:
            pi_.line(f"c{gd}", phs, rf.irr_exact(gd + 1e-9, phs), color=col, width=1.4,
                     name=f"{gd:g} dB gain error")
        need = p.nb + 20
        pi_.hline("need", need, color=GRAY, style="--", label=f"needed: neighbour + 20 dB = {need:.0f} dB",
                  label_pos=0.02)
        pi_.scatter("hw", [p.ph], [irr], color=RED, size=14, symbol="d")
        if p.iqfix:
            pi_.scatter("fx", [p.ph], [min(irrm, 74)], color=GREEN, size=14, symbol="star")
        self.readout(irr=irr, irrm=irrm, img=p.nb - irrm, mer=mer)

    @lru_cache(maxsize=1)
    def wanted_ref(self):
        sym, w, nb, noise = zeroif_signals()
        f, P = cl.welch_psd(w, self.fs, 1024)
        return float(np.median(P[np.abs(f) < 0.4e6]))

    def story(self, p):
        r = self.r
        s = (f"<p>A zero-IF receiver's I and Q branches are never perfectly matched: "
             f"{v(p.g, '.2f', 'dB')} of gain and {v(p.ph, '.1f', '°')} of phase imbalance. Each signal "
             f"leaks a mirror image at minus its frequency, {v(r.get('irr', 0), '.1f', 'dB')} down "
             f"(the <b>image-rejection ratio</b>). On its own that would be harmless.</p>"
             f"<p>The trouble is your neighbour. It sits at exactly −{p.off:.2f} MHz, "
             f"{v(p.nb, '.0f', 'dB')} stronger than you, so its image lands squarely on your channel, "
             f"only {v(-r.get('img', 0), '.0f', 'dB')} below you. Your MER: "
             f"{v(r.get('mer', 0), '.1f', 'dB')}. 16-QAM needs about 20 dB, hence the dashed line: "
             f"IRR ≥ neighbour + 20 dB.</p>")
        if p.iqfix:
            s += (f"<p>{good('Blind correction on')}: noise and modulated signals are <i>proper</i>, "
                  f"I and Q of equal power and uncorrelated. Measuring E[I²], E[Q²] and E[IQ] and "
                  f"forcing them back to that shape estimates and removes the imbalance: IRR now "
                  f"{v(r.get('irrm', 0), '.0f', 'dB')} (green star). The AD9364's background "
                  f"calibration does a version of this.</p>")
        elif p.dc > -30 and not p.dcfix:
            s += ("<p>The spike at 0 Hz is LO leakage and DC offset. Keep your channel away from it, "
                  "or remove the mean.</p>")
        return "<h3>Your neighbour's ghost</h3>" + s + keybox(
            "IRR = (1 + 2g·cos φ + g²)/(1 − 2g·cos φ + g²). Required IRR = power difference + "
            "the SNR you need.")


# =============================================================================== 6. phase noise
MODS = {"QPSK": "qpsk", "16-QAM": "16qam", "64-QAM": "64qam", "256-QAM": "256qam"}


class PhaseNoise(Experiment):
    title = "Oscillator phase noise"
    blurb = "The LO's phase wanders. A tracking loop follows the slow part; the fast part smears QAM."
    book = "sec:ch07:synth"
    animate = True
    fps = 8
    controls = [
        Choice("mod", "Modulation", list(MODS), "64-QAM", style="menu"),
        LogSlider("lw", "Oscillator linewidth", 1e-6, 1e-2, 2e-5, unit="× Rs",
                  help="3 dB linewidth of the Wiener phase noise, relative to the symbol rate"),
        Toggle("track", "Carrier tracking on", True),
        LogSlider("bl", "Tracking-loop bandwidth", 1e-4, 0.2, 5e-3, unit="× Rs",
                  enabled_if=lambda p: p.track),
        Slider("esn0", "Es/N₀", 10, 45, 35, step=0.5, unit="dB"),
    ]
    plots = [
        ConstellationPlot("con", "Constellation", lim=1.45),
        SpectrumPlot("skirt", "The oscillator's spectrum (one side)", x="offset from the carrier (× Rs)",
                     y="level (dBc per bin)", logx=True, xlim=(1e-4, 0.5), ylim=(-80, 3), legend="tr"),
        Plot("pn", "Phase of the oscillator", x="time (symbols)", y="phase (°)", xlim=(0, 3000),
             legend="tl", legend_cols=3),
    ]
    layout = [["con", "skirt"], ["con", "pn"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("rms", "Residual phase error", "° rms", ".2f"),
        Readout("evm", "EVM", "%", ".2f", good=lambda x: x < 3),
        Readout("ser", "Symbol error rate", "", "sci", good=lambda x: x < 1e-3),
        Readout("lim", "Phase-noise MER limit", "dB", ".1f"),
    ]
    challenges = [
        Challenge("64-QAM with a five-times-worse oscillator (linewidth ≥ 10⁻⁴ Rs): bring the EVM below 3 %.",
                  lambda s: s.p.mod == "64-QAM" and s.p.lw >= 1e-4 and s.p.track and s.r.evm < 3,
                  hint="A faster loop follows more of the phase wander, but lets in more noise in a real receiver."),
        Challenge("Break 256-QAM with phase noise alone: symbol errors above 10⁻² at Es/N₀ ≥ 40 dB.",
                  lambda s: s.p.mod == "256-QAM" and s.p.esn0 >= 40 and s.r.ser > 1e-2),
        Challenge("Switch tracking off: how clean must the oscillator be for 16-QAM to survive "
                  "3000 symbols (symbol errors below 10⁻³)?",
                  lambda s: not s.p.track and s.p.mod == "16-QAM" and s.r.ser < 1e-3,
                  hint="Without a loop the phase random walk only grows: think 10⁻⁶."),
    ]
    nsym = 3000

    def update(self, p):
        self.draw(p)

    def draw(self, p):
        rng = self.rng
        con = cl.get_constellation(MODS[p.mod])
        pts = con.points / np.sqrt(np.mean(np.abs(con.points) ** 2))
        idx = rng.integers(0, len(pts), self.nsym)
        s = pts[idx]
        th = np.cumsum(rng.standard_normal(self.nsym) * np.sqrt(2 * np.pi * p.lw))
        if p.track:
            a = 1 - np.exp(-2 * np.pi * p.bl)
            est = sps.lfilter([a], [1, a - 1], th)
            est = np.r_[0.0, est[:-1]]                       # the loop only knows the past
            res = th - est
        else:
            est = np.full_like(th, th[0])
            res = th - th[0]
        n0 = 10 ** (-p.esn0 / 10)
        y = s * np.exp(1j * res) + np.sqrt(n0 / 2) * (rng.standard_normal(self.nsym)
                                                      + 1j * rng.standard_normal(self.nsym))
        dec = np.argmin(np.abs(y[:, None] - pts[None, :]), axis=1)
        ser = float(np.mean(dec != idx))
        evm = 100 * np.sqrt(np.mean(np.abs(y - s) ** 2))
        rms = float(np.rad2deg(np.std(res)))
        pc = self.plot("con")
        pc.points("y", y, color=NAVY, size=3, alpha=0.4)
        pc.ideal("i", pts, color=RED, size=8 if len(pts) > 64 else 11)
        # oscillator spectrum (single sideband), from a longer record
        th2 = np.cumsum(np.random.default_rng(5).standard_normal(1 << 15) * np.sqrt(2 * np.pi * p.lw))
        f, P = sps.welch(np.exp(1j * th2), 1.0, nperseg=8192, return_onesided=False, window="blackmanharris")
        k = f > 0
        pk = self.plot("skirt")
        pk.line("L", f[k], db(P[k] / P.max()), color=NAVY, width=1.8, name="free-running oscillator")
        if p.track:
            pk.vline("bl", p.bl, color=GREEN, style="--", label="loop bandwidth: tracked below",
                     label_pos=0.08)
        pk.vline("lw", max(p.lw / 2, 1.1e-4), color=RED, style=":", label="half linewidth", label_pos=0.9)
        pp = self.plot("pn")
        nn = np.arange(self.nsym)
        thd, estd, resd = np.rad2deg(th), np.rad2deg(est), np.rad2deg(res)
        pp.line("th", nn, thd, color=GRAY, width=1.6, name="oscillator phase")
        if p.track:
            pp.line("est", nn, estd, color=GREEN, width=1.4, style="--", name="loop's estimate")
        pp.line("res", nn, resd, color=RED, width=1.2, name="residual error")
        lo = min(thd.min(), resd.min(), -5)
        hi = max(thd.max(), resd.max(), 5)
        pp.set_ylim(lo - 0.1 * (hi - lo), hi + 0.35 * (hi - lo))
        lim = -10 * np.log10(max(np.var(res), 1e-12))
        self.readout(rms=rms, evm=evm, ser=ser, lim=lim)

    def story(self, p):
        r = self.r
        s = ("<p>A real oscillator's phase drifts like a random walk (grey). Its spectrum grows "
             "<b>skirts</b> around the carrier (top right). The carrier-tracking loop follows the "
             "slow part of the wander (green dashed); what it cannot follow is left over (red) and "
             "rotates every symbol by a random angle.</p>")
        s += (f"<p>A rotation moves outer points further than inner ones, so dense QAM smears into "
              f"arcs, worst at the corners: {v(r.get('rms', 0), '.2f', '° rms')} of residual phase "
              f"caps the MER at {v(r.get('lim', 0), '.1f', 'dB')} however strong the signal. ")
        if not p.track:
            s += bad("Without tracking") + " the random walk keeps growing and the points smear into rings.</p>"
        else:
            s += (f"A wider loop (bandwidth {v(p.bl, '.2g', '× Rs')}) follows faster wander; in a real "
                  f"receiver it also lets in more thermal noise, so there is an optimum.</p>")
        s += ("<p>This is why 5G NR adds phase-tracking reference signals (PT-RS) at millimetre-wave "
              "carriers, and why 4096-QAM Wi-Fi 7 needs an excellent synthesiser.</p>")
        return "<h3>A wandering carrier</h3>" + s + keybox(
            "MER limit from phase noise ≈ 1/σ_φ² (σ in radians). 1° rms caps it at 35 dB.")


# =============================================================================== 7. ADC dynamic range
class ADCDynamicRange(Experiment):
    title = "ADC bits, back-off and processing gain"
    blurb = "How many bits? Depends on how strong, how weak, and how narrow your channel is."
    book = "sec:ch03:adcfloor"
    controls = [
        IntSlider("bits", "ADC resolution", 4, 16, 8, unit="bits"),
        Slider("lvl", "Signal level", -90, 3, -10, step=0.5, unit="dBFS"),
        LogSlider("bw", "Channel bandwidth", 0.1, 30, 10, unit="MHz",
                  help="Bandwidth kept by the digital channel filter (fs = 61.44 MS/s)"),
        Slider("dither", "Analog noise (dither)", 0.0, 3.0, 0.0, step=0.05, unit="LSB rms"),
    ]
    plots = [
        SpectrumPlot("spec", "ADC output spectrum (61.44 MS/s, I + jQ)", x="frequency (MHz)",
                     y="level (dBFS)", xlim=(-30.72, 30.72), ylim=(-160, 8), legend=None),
        Plot("snr", "SNR vs signal level", x="signal level (dBFS)", y="SNR (dB)", xlim=(-90, 5),
             ylim=(-10, 140), legend="tl"),
        Plot("wave", "Zoom: 48 samples of the I rail", x="sample", y="amplitude (LSB)", xlim=(0, 48),
             legend="tl", legend_cols=2),
    ]
    layout = [["spec", "spec"], ["snr", "wave"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("snr", "SNR, full band", "dB", ".1f"),
        Readout("snrc", "SNR in the channel", "dB", ".1f"),
        Readout("sfdr", "Spurious-free range", "dBc", ".1f"),
        Readout("dr", "Channel dynamic range", "dB", ".1f",
                help="Theory: 6.02·bits + 1.76 + 10·log₁₀(fs / B)"),
    ]
    challenges = [
        Challenge("Chapter 7's GSM case: 98 dB of dynamic range in a 200 kHz channel at 61.44 MS/s, "
                  "with as few bits as possible.",
                  lambda s: s.p.bits <= 12 and s.p.bw <= 0.2 and s.r.dr >= 98),
        Challenge("Tame the spurs: an 8-bit ADC with a −40 dBFS tone. Find the dither that lifts the "
                  "SFDR above 30 dBc (too little leaves spurs, too much is just noise).",
                  lambda s: s.p.bits == 8 and abs(s.p.lvl + 40) < 0.3 and s.r.sfdr > 30),
        Challenge("Clip it: push the signal past full scale until the full-band SNR collapses below 25 dB.",
                  lambda s: s.p.lvl > 0 and s.r.snr < 25),
    ]
    fs, N, k0 = 61.44, 1 << 14, 811

    def update(self, p):
        N, fs = self.N, self.fs
        n = np.arange(N)
        A = 10 ** (p.lvl / 20)
        lsb = 2 / 2 ** p.bits
        x = A * np.exp(2j * np.pi * self.k0 * n / N)
        rng = np.random.default_rng(3)
        d = p.dither * lsb * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        xq = sp.quantize(x + d, p.bits)
        # SNR from the error after removing the best-fit tone
        u = np.exp(2j * np.pi * self.k0 * n / N)
        a = np.vdot(u, xq) / N
        e = xq - a * u
        snr = float(db(abs(a) ** 2 / max(np.mean(np.abs(e) ** 2), 1e-30)))
        E = np.abs(np.fft.fft(e)) ** 2 / N ** 2
        fk = np.fft.fftfreq(N, 1 / fs)
        inch = np.abs(fk - fk[self.k0]) <= p.bw / 2
        snrc = float(db(abs(a) ** 2 / max(E[inch].sum(), 1e-30)))
        w = sps.windows.blackmanharris(N)
        X = np.abs(np.fft.fft(xq * w)) / np.sum(w)
        XdB = 20 * np.log10(X + 1e-12)
        mask = np.ones(N, bool)
        mask[(self.k0 - 6 + np.arange(13)) % N] = False
        sfdr = float(XdB[self.k0] - XdB[mask].max())
        dr = 6.02 * p.bits + 1.76 + 10 * np.log10(fs / p.bw)
        f = np.fft.fftshift(fk)
        ps = self.plot("spec")
        ps.band("ch", fk[self.k0] - p.bw / 2, fk[self.k0] + p.bw / 2, color=GREEN, alpha=0.12)
        ps.line("X", f, np.fft.fftshift(XdB), color=NAVY, width=1.0, alpha=0.55)
        floor_bin = -(6.02 * p.bits + 1.76) - 10 * np.log10(N / 2.0)
        ps.hline("qf", floor_bin, color=RED, style="--")
        ps.text("qft", -29.5, floor_bin + 4, "ideal quantisation floor (per bin)", color=RED, size=9,
                anchor=(0, 1), fill=True)
        ps.text("ch", fk[self.k0], None, "channel", color=GREEN, size=9, anchor=(0.5, 0))
        # SNR vs level
        L = np.linspace(-90, 5, 200)
        th = np.where(L <= 0, L + 6.02 * p.bits + 1.76, np.nan)
        pn = self.plot("snr")
        pn.line("full", L, th, color=NAVY, width=2.0, name="theory, full band")
        pn.line("chan", L, th + 10 * np.log10(fs / p.bw), color=GREEN, width=2.0,
                name=f"theory, in {p.bw:.3g} MHz")
        pn.vline("fs", 0, color=RED, style=":", label="full scale", label_pos=0.95)
        pn.scatter("m1", [p.lvl], [snr], color=NAVY, size=13, symbol="d", name="measured")
        pn.scatter("m2", [p.lvl], [snrc], color=GREEN, size=13, symbol="d")
        # zoom
        k = np.arange(48)
        pw = self.plot("wave")
        pw.line("a", k, (x + d).real[:48] / lsb, color=GRAY, width=2.4, alpha=0.6, name="analog input")
        pw.line("q", np.r_[k, 48] - 0.5, xq.real[:48] / lsb, color=RED, width=1.8,
                step=True, name="ADC output")
        amp = max(1.6 * (A + 3 * p.dither * lsb) / lsb, 2.5)
        pw.set_ylim(-amp, amp * 1.35)
        self.readout(snr=snr, snrc=snrc, sfdr=sfdr, dr=dr)

    def story(self, p):
        r = self.r
        s = (f"<p>An ideal {v(p.bits, 'd')}-bit ADC driven to full scale reaches "
             f"SNR = 6.02·b + 1.76 = {v(6.02 * p.bits + 1.76, '.1f', 'dB')}. Your tone is "
             f"{v(-p.lvl, '.1f', 'dB')} below full scale, so it gets {v(r.get('snr', 0), '.1f', 'dB')}: "
             f"every dB of back-off is a dB of SNR lost. Above full scale, clipping is far worse.</p>"
             f"<p>The quantisation noise spreads over the whole 61.44 MHz; your channel filter keeps "
             f"only {v(p.bw, '.3g', 'MHz')} of it. That <b>processing gain</b>, "
             f"{v(10 * np.log10(self.fs / p.bw), '.1f', 'dB')}, is why a modest ADC can serve a narrow "
             f"channel: in-channel SNR {v(r.get('snrc', 0), '.1f', 'dB')}.</p>")
        if p.lvl < -30 and p.dither == 0:
            s += ("<p>A small signal exercises only a few levels, and the error becomes a periodic "
                  "staircase: <b>spurs</b> (SFDR " + f"{v(r.get('sfdr', 0), '.0f', 'dBc')}" + "). A little "
                  "analog noise (dither) randomises the error into a flat floor.</p>")
        return "<h3>Bits, back-off, bandwidth</h3>" + s + keybox(
            "Channel dynamic range = 6.02·b + 1.76 + 10·log₁₀(fs/B). Chapter 7's GSM example needs "
            "98 dB in 200 kHz: 12 bits at 61.44 MS/s.")


# =============================================================================== 8. sensitivity
KNOWN = [  # (noise bandwidth Hz, sensitivity dBm, label)  -- Chapter 3's figure
    (2.046e6, -130, "GPS L1 C/A"), (125e3, -137, "LoRa SF12"), (125e3, -123, "LoRa SF7"),
    (200e3, -102, "GSM (TS 45.005)"), (9e6, -97, "LTE 10 MHz"), (20e6, -82, "Wi-Fi 6 Mb/s"),
    (20e6, -65, "Wi-Fi 54 Mb/s")]


class Sensitivity(Experiment):
    title = "Sensitivity and range"
    blurb = "−174 + 10·log B + NF + SNR: the most useful line in radio engineering."
    book = "sec:ch03:sensitivity"
    controls = [
        Heading("Receiver"),
        LogSlider("bw", "Bandwidth", 1, 100000, 200, unit="kHz"),
        Slider("nf", "Noise figure", 1, 15, 8, step=0.5, unit="dB"),
        Slider("snr", "Required SNR", -25, 30, 10, step=0.5, unit="dB",
               help="Negative for spread-spectrum and LoRa: processing gain lets them work below the noise"),
        Heading("Link"),
        Slider("ptx", "Transmit power", -30, 30, 0, step=1, unit="dBm"),
        LogSlider("fc", "Carrier frequency", 50, 6000, 915, unit="MHz"),
        Slider("gant", "Antenna gains (total)", 0, 40, 0, step=1, unit="dBi"),
        Slider("margin", "Fade margin", 0, 30, 0, step=1, unit="dB"),
    ]
    plots = [
        Plot("range", "Received power vs distance (free space)", x="distance (m)", y="power (dBm)",
             logx=True, xlim=(1, 1e6), ylim=(-170, 20), legend="tr"),
        BarPlot("bud", "Building the sensitivity", y="dBm"),
        Plot("known", "Real receivers vs the thermal floor", x="noise bandwidth (Hz)", y="power (dBm)",
             logx=True, xlim=(1e2, 1e9), ylim=(-165, -45), legend="tl"),
    ]
    layout = [["range", "range"], ["bud", "known"]]
    readouts = [
        Readout("floor", "Noise floor", "dBm", ".1f"),
        Readout("sens", "Sensitivity", "dBm", ".1f"),
        Readout("rng", "Free-space range", "km", ".2f"),
        Readout("m1k", "Margin at 1 km", "dB", ".1f", good=lambda x: x > 0),
    ]
    challenges = [
        Challenge("LoRa-style: 15 km of free space at 868 MHz with 14 dBm, no antenna gain and a "
                  "10 dB fade margin.",
                  lambda s: abs(s.p.fc - 868) < 10 and abs(s.p.ptx - 14) < 0.5 and s.p.gant == 0
                  and s.p.margin >= 10 and s.r.rng >= 15,
                  hint="LoRa demodulates below the noise: SF12 needs about −20 dB SNR."),
        Challenge("GPS arrives at about −130 dBm in 2 MHz. With a 2 dB noise figure, what SNR is that? "
                  "Make the sensitivity exactly −130 dBm (±0.5).",
                  lambda s: abs(s.p.bw - 2000) < 100 and abs(s.p.nf - 2) < 0.01 and abs(s.r.sens + 130) <= 0.5),
        Challenge("Point-to-point: 100 km at 2.4 GHz with 20 dBm or less (dishes allowed).",
                  lambda s: abs(s.p.fc - 2400) < 30 and s.p.ptx <= 20 and s.r.rng >= 100),
    ]

    def update(self, p):
        B = p.bw * 1e3
        floor = -174 + 10 * np.log10(B) + p.nf
        sens = floor + p.snr
        need = sens + p.margin
        d = np.logspace(0, 6, 400)
        prx = p.ptx + p.gant - cl.fspl_db(d, p.fc * 1e6)
        allowed = p.ptx + p.gant - need
        rng_m = 10 ** ((allowed - 20 * np.log10(p.fc * 1e6) + 147.55) / 20)
        pr = self.plot("range")
        pr.line("p", d, prx, color=NAVY, width=2.4, name="received power")
        pr.hline("s", sens, color=RED, style="--", label=f"sensitivity {sens:.1f} dBm", label_pos=0.02)
        pr.hline("f", floor, color=GRAY, style=":", label=f"noise floor {floor:.1f} dBm", label_pos=0.02)
        if p.margin > 0:
            pr.hband("m", sens, need, color=ORANGE, alpha=0.15)
        pr.vline("r", min(max(rng_m, 1.01), 9.9e5), color=GREEN, style="-", width=1.6,
                 label=f"range {rng_m / 1e3:.2f} km", label_pos=0.9)
        pb = self.plot("bud")
        steps = [("kT₀ = −174", -174.0, -174.0), ("+10·log B", -174.0, -174 + 10 * np.log10(B)),
                 ("+ NF", -174 + 10 * np.log10(B), floor), ("+ SNR", floor, sens)]
        for i, (lab, a, b) in enumerate(steps):
            lo, hi = min(a, b), max(a, b)
            if hi - lo < 0.5:
                hi = lo + 0.5
            pb.bars(f"b{i}", [i], [hi], width=0.6, colors=[[GRAY, NAVY, ORANGE, RED][i]], base=lo)
            pb.text(f"t{i}", i, max(a, b), f"{b:.0f}", anchor=(0.5, 1.05), bold=True, size=9)
        pb.set_xticks([(i, s[0]) for i, s in enumerate(steps)])
        pb.set_xlim(-0.6, 3.6)
        pb.set_ylim(-180, max(-60, sens + 15))
        pk = self.plot("known")
        Bs = np.logspace(2, 9, 100)
        pk.line("t", Bs, -174 + 10 * np.log10(Bs), color=NAVY, width=2.0, name="thermal floor kT₀B")
        pk.line("t5", Bs, -174 + 10 * np.log10(Bs) + 5, color=NAVY, width=1.0, style="--",
                name="+ 5 dB NF")
        for i, (b, lvl, lab) in enumerate(KNOWN):
            pk.scatter(f"k{i}", [b], [lvl], color=GRAY, size=8)
            pk.text(f"kt{i}", b, lvl, " " + lab, color=GRAY, size=8, anchor=(0, 0.5))
        pk.scatter("you", [B], [sens], color=RED, size=15, symbol="d", name="your receiver")
        m1k = p.ptx + p.gant - float(cl.fspl_db(1e3, p.fc * 1e6)) - need
        self.readout(floor=floor, sens=sens, rng=rng_m / 1e3, m1k=m1k)

    def story(self, p):
        r = self.r
        s = (f"<p>Every receiver's noise starts at kT₀ = −174 dBm per hertz. In "
             f"{v(p.bw, '.3g', 'kHz')} that becomes {v(-174 + 10 * np.log10(p.bw * 1e3), '.1f', 'dBm')}; "
             f"the receiver's noise figure adds {v(p.nf, '.1f', 'dB')}; the demodulator needs "
             f"{v(p.snr, '.1f', 'dB')} of SNR on top: sensitivity {v(r.get('sens', 0), '.1f', 'dBm')}.</p>")
        if p.snr < 0:
            s += ("<p>A negative SNR is not a typo: spread-spectrum systems (GPS, LoRa) despread "
                  "before deciding, so they work with the signal below the noise in the RF "
                  "bandwidth.</p>")
        s += (f"<p>In free space the received power falls 20 dB per decade of distance, so every "
              f"10 dB of sensitivity multiplies the range by about 3.16. Your link reaches "
              f"{v(r.get('rng', 0), '.2f', 'km')}"
              + (f" with a {v(p.margin, '.0f', 'dB')} fade margin" if p.margin else
                 ", optimistically: with no fade margin (Chapter 11)") + ".</p>")
        return "<h3>The most useful line in radio</h3>" + s + keybox(
            "P_min = −174 + 10·log₁₀(B) + NF + SNR_req dBm. Learn it by heart.")


# =============================================================================== 9. waterfall
MORSE = "-.-. --.-"          # CQ


def _morse_units(code):
    u = []
    for ch in code:
        if ch == ".":
            u += [1, 0]
        elif ch == "-":
            u += [1, 1, 1, 0]
        else:
            u += [0, 0]
    return np.array(u + [0] * 7, bool)


class Band:
    """A synthetic 2 MHz-wide capture: an FM station, twin CW beacons, a TDMA burst carrier, a
    frequency hopper, a chirp radar and a slow Morse beacon, in noise. Continuous across calls."""

    def __init__(self):
        self.n0 = 0
        self.morse = _morse_units(MORSE)
        r = np.random.default_rng(77)
        self.table = r.integers(0, 4, 100003)               # pseudo-random symbols and bits
        self.h = sps.firwin(63, 70e3, fs=2e6)               # smooths the TDMA symbols
        self.zi = np.zeros(len(self.h) - 1, complex)
        self.hphase = 0.0

    def next(self, N, dc=True, fs=2e6):
        t = (self.n0 + np.arange(N)) / fs
        self.n0 += N
        tw = 2 * np.pi
        fm = np.exp(1j * (tw * -600e3 * t - 75e3 * (0.6 * np.cos(tw * 1.1e3 * t) / 1.1e3
                                                  + 0.4 * np.cos(tw * 3.3e3 * t) / 3.3e3)))
        cw = 0.02 * (np.exp(1j * tw * 300e3 * t) + np.exp(1j * (tw * 305e3 * t + 1.0)))
        si = np.floor(t * 100e3).astype(np.int64)
        slot = np.floor(t / 577e-6).astype(np.int64) % 8
        qp = np.exp(1j * (np.pi / 2) * self.table[si % len(self.table)] + 1j * np.pi / 4)
        qp = qp * np.isin(slot, [0, 1, 4])
        qp, self.zi = sps.lfilter(self.h, 1.0, qp, zi=self.zi)
        tdma = 0.25 * qp * np.exp(1j * tw * 100e3 * t)
        hop = np.floor(t / 625e-6).astype(np.int64)
        fh = 450e3 + 10e3 * ((hop * 7919) % 50)
        bits = np.floor(t * 50e3).astype(np.int64)
        finst = fh + 20e3 * (2 * (self.table[(bits * 7) % len(self.table)] % 2) - 1)   # continuous-phase FSK
        ph = self.hphase + tw * np.cumsum(finst) / fs
        self.hphase = float(ph[-1] % tw)
        hp = 0.08 * np.exp(1j * ph)
        tau = t % 10e-3
        chirp = 0.04 * np.exp(1j * tw * (-950e3 * tau + 0.5 * (250e3 / 5e-3) * tau ** 2)) * (tau < 5e-3)
        unit = np.floor(t / 0.06).astype(np.int64) % len(self.morse)
        morse = 2e-3 * self.morse[unit] * np.exp(1j * tw * -200e3 * t)
        rng = np.random.default_rng(self.n0 % 100003)
        noise = np.sqrt(1e-5 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        x = fm + cw + tdma + hp + chirp + morse + noise
        if dc:
            x = x + 3e-3 * (1 + 0.5j)
        return x


class Waterfall(Experiment):
    title = "A live waterfall"
    blurb = "Watch a crowded band: tune the FFT to see hops, bursts, beacons and a radar chirp."
    book = "sec:ch02:stft"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Choice("nfft", "FFT size", ["256", "512", "1024", "2048", "4096"], "512", style="menu"),
        Choice("win", "Window", ["Rectangular", "Hann", "Blackman–Harris"], "Hann", style="menu"),
        IntSlider("avg", "FFTs averaged per row", 1, 16, 1),
        Slider("rng", "Colour range", 30, 90, 70, step=5, unit="dB"),
        Toggle("dc", "LO leakage (DC spur)", True),
        Toggle("hold", "Max hold", False),
        Toggle("labels", "Label the signals", True),
    ]
    plots = [
        ImagePlot("wf", "Waterfall (newest at the top)", x="frequency offset (kHz)", y="time ago (ms)"),
        SpectrumPlot("live", "Live spectrum: this row (grey) and the running average (blue)",
                     x="frequency offset (kHz)", y="level (dB)", xlim=(-1000, 1000), ylim=(-100, 8),
                     legend=None),
    ]
    layout = [["wf"], ["live"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("rbw", "Resolution bandwidth", "kHz", ".2f"),
        Readout("row", "Time per row", "ms", ".3f"),
        Readout("twins", "Beacons at +300/305 kHz", "", None),
        Readout("morse", "Morse beacon above noise", "dB", ".0f"),
    ]
    challenges = [
        Challenge("Resolve the twin beacons at +300 and +305 kHz (a 6 dB dip between them).",
                  lambda s: s.r.twins == "resolved"),
        Challenge("Catch individual hops: rows of 0.3 ms or less (each hop lasts 0.625 ms) with a "
                  "resolution bandwidth under 4 kHz.",
                  lambda s: s.r.row <= 0.3 and s.r.rbw < 4,
                  hint="Which window has the narrowest main lobe?"),
        Challenge("Find the faint Morse beacon at −200 kHz: make it stand 25 dB above the noise.",
                  lambda s: s.r.morse >= 25),
    ]
    fs, nrows = 2e6, 160

    def setup(self):
        self.band = Band()
        self.key = None

    def reset_buf(self, p):
        N = int(p.nfft)
        self.buf = np.full((self.nrows, N), -140.0)
        self.ema = None
        self.hold = None
        self.morse_hist = []
        self.key = (p.nfft, p.win, p.avg)

    def rows(self, p, k):
        N, avg = int(p.nfft), p.avg
        w = sp.window(p.win, N)
        x = self.band.next(N * avg * k, dc=p.dc, fs=self.fs).reshape(k, avg, N)
        X = np.abs(np.fft.fftshift(np.fft.fft(x * w, axis=2), axes=2)) ** 2 / np.sum(w) ** 2
        return 10 * np.log10(X.mean(axis=1) + 1e-16)

    def step(self, p):
        if self.key != (p.nfft, p.win, p.avg):
            self.reset_buf(p)
        N = int(p.nfft)
        k = max(1, min(8, 16384 // (N * p.avg)))
        R = self.rows(p, k)
        self.buf = np.vstack([R[::-1], self.buf[:-k]])
        lin = np.mean(10 ** (R / 10), axis=0)
        self.ema = lin if self.ema is None else 0.85 * self.ema + 0.15 * lin
        self.hold = R.max(axis=0) if self.hold is None else np.maximum(self.hold, R.max(axis=0))
        self.last = R[-1]

    def update(self, p):
        if self.key != (p.nfft, p.win, p.avg):
            self.reset_buf(p)
        for _ in range(3):
            self.step(p)
        self.draw(p)

    def tick(self, p):
        self.step(p)
        self.draw(p)

    def draw(self, p):
        N = int(p.nfft)
        f = np.fft.fftshift(np.fft.fftfreq(N, 1 / self.fs)) / 1e3
        row_ms = N * p.avg / self.fs * 1e3
        pw = self.plot("wf")
        top = 0.0
        pw.image("img", self.buf[::-1], x=(f[0], f[-1]), y=(self.nrows * row_ms, 0), cmap="heat",
                 levels=(top - p.rng, top), colorbar=True, cbar_label="dB")
        pw.set_xlim(-1000, 1000)
        pw.set_ylim(0, self.nrows * row_ms)
        pl = self.plot("live")
        pl.line("now", f, self.last, color=GRAY, width=1.0, alpha=0.8, name="this row")
        pl.line("avg", f, 10 * np.log10(self.ema + 1e-16), color=NAVY, width=1.8, name="average")
        if p.hold:
            pl.line("hold", f, self.hold, color=ORANGE, width=1.2, name="max hold")
        if p.labels:
            for key, fx, lab in [("fm", -600, "FM station"), ("cw", 302.5, "twin beacons"),
                                 ("td", 100, "TDMA bursts"), ("hp", 700, "frequency hopper"),
                                 ("ch", -825, "chirp radar"), ("mo", -200, "Morse")]:
                pl.text("l" + key, fx, None, lab, color=PURPLE, size=8.5, anchor=(0.5, 0))
        # measurements on the averaged spectrum
        S = 10 * np.log10(self.ema + 1e-16)
        floor = float(np.percentile(S, 15))          # signals cover over half the band
        a, b = np.argmin(np.abs(f - 300)), np.argmin(np.abs(f - 305))
        dip = min(S[a - 1:a + 2].max(), S[b - 1:b + 2].max()) - S[a:b + 1].min() if b - a >= 2 else 0.0
        twins = "resolved" if dip >= 6 else "merged"
        km = np.argmin(np.abs(f + 200))
        self.morse_hist.append(float(self.last[km - 1:km + 2].max()) - floor)
        self.morse_hist = self.morse_hist[-40:]
        rbw = sp.window_metrics(sp.window(p.win, N))["enbw"] * self.fs / N / 1e3
        self.floor = floor
        self.readout(rbw=rbw, row=row_ms, twins=twins, morse=max(self.morse_hist))

    def story(self, p):
        r = self.r
        s = (f"<p>A waterfall is a spectrogram that never stops: each new row is the spectrum of "
             f"the last {v(r.get('row', 0), '.3f', 'ms')} of samples. Your FFT of {p.nfft} points "
             f"gives bins {v(self.fs / int(p.nfft) / 1e3, '.2f', 'kHz')} apart and a resolution "
             f"bandwidth of {v(r.get('rbw', 0), '.2f', 'kHz')}.</p>")
        s += ("<p>This band is crowded. A wide <b>FM station</b>, two <b>CW beacons</b> only 5 kHz "
              "apart, <b>TDMA bursts</b> that switch on and off in 577 µs slots, a <b>frequency "
              "hopper</b> changing channel every 0.625 ms, a <b>chirp radar</b> sweeping every "
              "10 ms, and a faint <b>Morse</b> beacon calling CQ.</p>")
        s += ("<p>No single FFT size shows everything: long FFTs split the beacons and lift the "
              "Morse out of the noise but blur the hops into a smear; short ones catch every hop "
              "and burst but merge the beacons. Averaging lowers the noise's wobble, not its level.</p>"
              + ("<p>" + bad("Rectangular window") + ": the strong FM station's sidelobes now leak "
                 "across the whole band and raise the apparent floor by about 20 dB (Lab 35, "
                 "Experiment 2).</p>" if p.win == "Rectangular" else ""))
        return "<h3>A crowded band, live</h3>" + s + keybox(
            "Resolution bandwidth × time per row ≈ constant. The FFT-size knob on every SDR app is "
            "the uncertainty principle.")


# =============================================================================== the lab
LAB = st.Lab(1, "Complex Baseband and IQ Sampling", chapter=2,
             chapter_title="Signals, Spectra and Systems · The Radio Transceiver and the SDR",
             experiments=[IQHelix, RealVsIQ, UpDown, DigitalDownConverter, ZeroIFImpairments,
                          PhaseNoise, ADCDynamicRange, Sensitivity, Waterfall])

if __name__ == "__main__":
    st.run(LAB)
