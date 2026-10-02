"""Lab 35 · Fourier, Spectra and Bandwidth   (Chapter 2)

Run it:      python labs/lab35_fourier_spectra.py
Self-test:   python labs/lab35_fourier_spectra.py --selftest

A spectrum analyser, an SDR waterfall and the FFT inside a modem all show "the spectrum", but
each one is an estimate shaped by choices: which harmonics you keep, which window, how long you
look, what you call "bandwidth". Nine experiments build the intuition behind every spectral
measurement an engineer makes: Fourier series and Gibbs, windows and leakage, zero padding,
the uncertainty principle, convolution in slow motion, group delay, five bandwidths of one
signal, the spectrogram, and the two-tone test that characterises every amplifier.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as sps

import commlib as cl
from commlib import spectral as sp
import studio as st
from studio import (Experiment, Slider, IntSlider, Choice, Toggle, Heading, Plot, SpectrumPlot,
                    ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


def db(x, floor=1e-30):
    return 10 * np.log10(np.maximum(x, floor))


# =============================================================================== 1. Fourier series
WAVES = ["Square", "Triangle", "Sawtooth", "Pulse train", "Build your own"]


def target_wave(kind, t, duty=0.2):
    """The ideal waveform (unit period, peak 1). "Build your own" is compared with the square."""
    t = np.asarray(t, float)
    if kind in ("Square", "Build your own"):
        return np.where((t % 1) < 0.5, 1.0, -1.0)
    if kind == "Triangle":
        return 2 / np.pi * np.arcsin(np.sin(2 * np.pi * t))
    if kind == "Sawtooth":
        return 2 * (((t + 0.5) % 1) - 0.5)
    tt = ((t + 0.5) % 1) - 0.5
    return np.where(np.abs(tt) < duty / 2, 1.0, 0.0)


_SCRAMBLE = np.random.default_rng(35).uniform(-np.pi, np.pi, 100)


def series(p, nmax=None):
    """(c0, A, phi) for the current settings."""
    if p.wave == "Build your own":
        A = np.array([p.a1, p.a2, p.a3, p.a4, p.a5], float)
        c0, ph = 0.0, np.full(5, -np.pi / 2)
    else:
        c0, A, ph = sp.fourier_coeffs(p.wave, nmax or p.N, p.duty)
    if p.scramble:
        ph = ph + _SCRAMBLE[:len(ph)]
    return c0, A, ph


@lru_cache(maxsize=48)
def error_curves(wave, duty, sigma, scramble):
    """RMS error (%) and overshoot (% of the jump) of the partial sums for N = 1..99."""
    t = (np.arange(2000) + 0.5) / 2000
    tgt = target_wave(wave, t, duty)
    c0, A, ph = sp.fourier_coeffs(wave, 99, duty)
    if scramble:
        ph = ph + _SCRAMBLE[:99]
    C = np.cos(2 * np.pi * np.outer(t, np.arange(1, 100)) + ph) * A
    jump = tgt.max() - tgt.min()
    rms_t = np.sqrt(np.mean(tgt ** 2))
    err, over = np.zeros(99), np.zeros(99)
    for n in range(1, 100):
        g = np.sinc(np.arange(1, n + 1) / (n + 1)) if sigma else np.ones(n)
        x = c0 + C[:, :n] @ g
        err[n - 1] = 100 * np.sqrt(np.mean((x - tgt) ** 2)) / rms_t
        over[n - 1] = 100 * max(0.0, x.max() - tgt.max()) / jump
    return err, over


class FourierSeries(Experiment):
    title = "Fourier series builder"
    blurb = "Add harmonics one at a time and watch a square wave appear, ringing and all."
    book = "sec:ch02:fs"
    controls = [
        Choice("wave", "Target waveform", WAVES, style="menu"),
        IntSlider("N", "Highest harmonic N", 1, 99, 7,
                  help="Keep harmonics 1..N of the waveform's Fourier series",
                  enabled_if=lambda p: p.wave != "Build your own"),
        Slider("duty", "Pulse duty cycle", 0.05, 0.5, 0.2, step=0.01,
               enabled_if=lambda p: p.wave == "Pulse train"),
        Toggle("sigma", "Lanczos σ-smoothing", False,
               help="Taper the coefficients by sinc(k/(N+1)): a gentle low-pass on the series"),
        Toggle("scramble", "Scramble the harmonic phases", False,
               help="Same amplitudes (same line spectrum), random phases"),
        Heading("Build your own (sine harmonics)"),
        Slider("a1", "Harmonic 1", -1.5, 1.5, 1.0, step=0.01,
               enabled_if=lambda p: p.wave == "Build your own"),
        Slider("a2", "Harmonic 2", -1.5, 1.5, 0.0, step=0.01,
               enabled_if=lambda p: p.wave == "Build your own"),
        Slider("a3", "Harmonic 3", -1.5, 1.5, 0.0, step=0.01,
               enabled_if=lambda p: p.wave == "Build your own"),
        Slider("a4", "Harmonic 4", -1.5, 1.5, 0.0, step=0.01,
               enabled_if=lambda p: p.wave == "Build your own"),
        Slider("a5", "Harmonic 5", -1.5, 1.5, 0.0, step=0.01,
               enabled_if=lambda p: p.wave == "Build your own"),
    ]
    plots = [
        Plot("wave", "Partial Fourier sum and the target", x="time (periods)", y="amplitude",
             xlim=(-1, 1), ylim=(-1.75, 2.25), legend="tl", legend_cols=2),
        Plot("lines", "Line spectrum: harmonic amplitudes", x="harmonic number k",
             y="amplitude |A_k|", ylim=(0, 1.45), legend=None),
        Plot("err", "How fast does it converge?", x="highest harmonic N", y="error (%)",
             xlim=(0, 100), ylim=(0, 50), legend="tr"),
    ]
    layout = [["wave", "wave"], ["lines", "err"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("over", "Overshoot", "% of the jump", ".1f"),
        Readout("err", "RMS error", "%", ".1f", good=lambda x: x < 5),
        Readout("pow", "Power captured", "%", ".2f"),
        Readout("crest", "Crest factor", "", ".2f", help="Peak / rms of the partial sum"),
    ]
    challenges = [
        Challenge("Gibbs is stubborn: keep 49 or more harmonics of the square wave and still see "
                  "more than 8.5 % overshoot.",
                  lambda s: s.p.wave == "Square" and s.p.N >= 49 and not s.p.sigma and s.r.over > 8.5),
        Challenge("Tame the ringing: a square wave with N ≥ 15 and an overshoot below 1.5 %.",
                  lambda s: s.p.wave == "Square" and s.p.N >= 15 and s.r.over < 1.5,
                  hint="More harmonics never removes Gibbs. Changing how you weight them does."),
        Challenge("Build your own square from harmonics 1–5 only: RMS error below 27 % "
                  "(the best possible is 25.9 %).",
                  lambda s: s.p.wave == "Build your own" and s.r.err < 27,
                  hint="The square has only odd harmonics, with amplitudes 4/(πk)."),
        Challenge("Same line spectrum, different wave: scramble the phases of a 15-harmonic square "
                  "until its crest factor exceeds 1.7.",
                  lambda s: s.p.wave == "Square" and s.p.scramble and s.p.N == 15 and s.r.crest > 1.7),
    ]

    def update(self, p):
        build = p.wave == "Build your own"
        c0, A, ph = series(p)
        n = len(A)
        g = np.sinc(np.arange(1, n + 1) / (n + 1)) if p.sigma else np.ones(n)
        t = np.linspace(-1, 1, 2001)
        x = sp.fourier_sum(t, c0, A * g, ph)
        tgt = target_wave(p.wave, t, p.duty)
        pw = self.plot("wave")
        pw.hline("z", 0, color=GRAY, style="-", width=0.6)
        pw.line("tgt", t, tgt, color=GRAY, width=3.0, alpha=0.45,
                name="square (target)" if build else p.wave.lower())
        pw.line("x", t, x, color=NAVY, width=2.0,
                name=f"sum of {n} harmonic{'s' if n > 1 else ''}" + (" (σ-smoothed)" if p.sigma else ""))
        if not build and n <= 7 and not p.scramble:
            for k in range(1, n + 1):                         # the individual harmonics, faintly
                if A[k - 1] > 1e-9:
                    pw.line(f"h{k}", t, A[k - 1] * np.cos(2 * np.pi * k * t + ph[k - 1]),
                            color=[RED, ORANGE, GREEN, PURPLE, TEAL, GOLD, BLUE][(k - 1) % 7],
                            width=1.0, alpha=0.55)
        # metrics on one period
        t1 = (np.arange(2000) + 0.5) / 2000
        x1 = sp.fourier_sum(t1, c0, A * g, ph)
        tg1 = target_wave(p.wave, t1, p.duty)
        jump = tg1.max() - tg1.min()
        over = 100 * max(0.0, x1.max() - tg1.max()) / jump
        err = 100 * np.sqrt(np.mean((x1 - tg1) ** 2)) / np.sqrt(np.mean(tg1 ** 2))
        pwr = 100 * np.mean(x1 ** 2) / np.mean(tg1 ** 2)
        crest = np.max(np.abs(x1)) / np.sqrt(np.mean(x1 ** 2) + 1e-30)
        k = np.arange(1, n + 1)
        pl = self.plot("lines")
        if c0:
            pl.stems("dc", [0], [abs(c0)], color=GRAY, size=7)
        pl.stems("A", k, np.abs(A * g), color=NAVY, size=6, name="kept")
        if not build:
            _, Af, _ = sp.fourier_coeffs(p.wave, 40, p.duty)
            kk = np.arange(n + 1, 41)
            if len(kk):
                pl.stems("rest", kk, Af[n:40], color=GRAY, size=4)
        pl.set_xlim(-0.6, 40.6 if not build else 6.6)
        pe = self.plot("err")
        if not build:
            e, o = error_curves(p.wave, round(p.duty, 3), p.sigma, p.scramble)
            nn = np.arange(1, 100)
            pe.line("e", nn, e, color=NAVY, width=2.0, name="RMS error")
            pe.line("o", nn, o, color=RED, width=2.0, name="overshoot")
            pe.hline("g", 8.95, color=GRAY, style=":", label="Gibbs: 8.95 %", label_pos=0.6)
            pe.scatter("now", [n, n], [e[n - 1], o[n - 1]], color=ORANGE, size=11, symbol="d")
            pe.set_ylim(0, max(30, min(100, e[0] * 1.1)))
        else:
            pe.text("na", 50, 25, "(fixed set of 5 harmonics)", color=GRAY, anchor=(0.5, 0.5))
        self.readout(over=over, err=err, pow=pwr, crest=crest)

    def story(self, p):
        r = self.r
        if p.wave == "Build your own":
            s = (f"<p>You are mixing five sine harmonics by hand. The grey square is the target; your "
                 f"sum misses it by {v(r.get('err', 0), '.1f', '%')} rms. The best any five "
                 f"harmonics can do is the square's own Fourier coefficients, 4/(πk) for odd k and "
                 f"zero for even k: that is what "
                 f"<b>orthogonality</b> means. Each coefficient can be chosen on its own.</p>")
        elif p.wave == "Triangle":
            s = (f"<p>The triangle has no jumps, only corners. Its harmonics fall as 1/k², so "
                 f"{v(p.N, 'd')} harmonics already give {v(r.get('err', 0), '.2f', '%')} rms error "
                 f"and no visible ringing. <b>Smooth in time ⇔ fast-decaying spectrum.</b></p>")
        else:
            s = (f"<p>With {v(p.N, 'd')} harmonics the sum follows the {p.wave.lower()} everywhere "
                 f"except at the jumps, where it rings and overshoots by "
                 f"{v(r.get('over', 0), '.1f', '%')} of the jump. Add more harmonics: the ripples "
                 f"squeeze closer to the edge but the overshoot never drops below about 9 %. "
                 f"That is the <b>Gibbs phenomenon</b> (Wilbraham 1848, Gibbs 1899).</p>")
            if p.sigma:
                s += ("<p>" + good("σ-smoothing on.") + " Tapering the coefficients is a gentle "
                      "low-pass filter on the series: the edge is a little slower, the overshoot "
                      "almost vanishes. A brick-wall filter rings; a smooth roll-off does not. "
                      "Raised-cosine pulse shaping (Chapter 8) is the same idea.</p>")
        if p.scramble:
            s += (f"<p>The phases are scrambled. The line spectrum is identical, yet the wave looks "
                  f"nothing like a {('square' if p.wave != 'Triangle' else 'triangle')}: crest "
                  f"factor {v(r.get('crest', 0), '.2f')}. A magnitude spectrum throws the phase "
                  f"away; OFDM's peak-to-average problem lives right here.</p>")
        return "<h3>Harmonics, one at a time</h3>" + s + keybox(
            "A periodic signal is a sum of harmonics. Jumps need infinitely many; truncating them "
            "leaves a 9 % overshoot that only smoother weighting removes.")


# =============================================================================== 2. windows
WIN_NAMES = ["Rectangular", "Hann", "Blackman–Harris", "Flat-top"]
WIN_COLORS = {"Rectangular": RED, "Hann": ORANGE, "Blackman–Harris": GREEN, "Flat-top": PURPLE}


@lru_cache(maxsize=8)
def _win(name, N):
    return sp.window(name, N)


@lru_cache(maxsize=8)
def _winmet(name, N):
    return sp.window_metrics(_win(name, N))


class WindowsLeakage(Experiment):
    title = "Windows, leakage and scalloping"
    blurb = "A strong tone hides a weak neighbour behind its sidelobes. Pick the right window."
    book = "sec:windows"
    controls = [
        Choice("win", "Window", WIN_NAMES, "Hann", style="menu"),
        Slider("off", "Strong tone off-bin", 0.0, 0.5, 0.3, step=0.01, unit="bins",
               help="0 = exactly on a DFT bin, 0.5 = halfway between two bins"),
        Heading("Weak neighbour"),
        Slider("weak", "Weak tone level", -120, -10, -60, step=1, unit="dB"),
        Slider("sep", "Distance", 1.5, 40, 8, step=0.5, unit="bins"),
        Toggle("others", "Show the other windows", True),
    ]
    plots = [
        SpectrumPlot("spec", "Strong tone (0 dB) and a weak neighbour", x="frequency (DFT bins)",
                     y="level (dB)", xlim=(20, 90), ylim=(-150, 8), legend="tr"),
        Plot("scal", "Scalloping loss: reading a tone between bins",
             x="tone offset from the bin centre (bins)", y="amplitude read (dB)",
             xlim=(-0.5, 0.5), ylim=(-4.3, 0.5), legend="bl"),
        Plot("shape", "The window w[n]", x="sample n / N", y="w[n]", xlim=(0, 1),
             ylim=(-0.15, 1.15), legend=None),
    ]
    layout = [["spec", "spec"], ["scal", "shape"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("side", "Highest sidelobe", "dB", ".1f"),
        Readout("enbw", "Noise bandwidth (ENBW)", "bins", ".2f",
                help="Every bin collects noise from this many bins' worth of bandwidth"),
        Readout("read", "Strong tone reads", "dB", ".2f", good=lambda x: abs(x) < 0.1),
        Readout("margin", "Weak tone above leakage", "dB", ".1f", good=lambda x: x > 3),
    ]
    challenges = [
        Challenge("Spot a −80 dB tone only 6 bins from the strong one (margin above 6 dB).",
                  lambda s: s.p.weak <= -80 and s.p.sep <= 6 and s.r.margin > 6),
        Challenge("Read the strong tone to within 0.05 dB even when it falls halfway between bins.",
                  lambda s: s.p.off >= 0.495 and abs(s.r.read) < 0.05),
        Challenge("Find a weak tone that the rectangular window shows (margin > 3 dB) but the Hann "
                  "window hides (margin < 0 dB).",
                  lambda s: s.p.win == "Rectangular" and s.r.margin > 3 and s.exp.margins["Hann"] < 0,
                  hint="Hann's main lobe is twice as wide as the rectangle's: look very close in."),
    ]
    N, pad = 256, 16

    def spectrum(self, name, f1, f2, a2):
        """Padded spectrum (dB) of complex tones at f1, f2 (bins) through a window."""
        N = self.N
        w = _win(name, N)
        n = np.arange(N)
        x = np.exp(2j * np.pi * f1 * n / N) + a2 * np.exp(2j * np.pi * f2 * n / N + 1j)
        X = np.abs(np.fft.fft(x * w, self.pad * N)) / np.sum(w)
        return 20 * np.log10(X + 1e-12)

    def update(self, p):
        N, pad = self.N, self.pad
        f1 = 40 + p.off
        f2 = f1 + p.sep
        a2 = 10 ** (p.weak / 20)
        fb = np.arange(pad * N) / pad
        keep = (fb >= 18) & (fb <= 92)
        ps = self.plot("spec")
        self.margins = {}
        k2 = int(round(f2 * pad))
        for name in WIN_NAMES:
            S = self.spectrum(name, f1, f2, a2)
            Sl = self.spectrum(name, f1, f2, 0.0)              # strong tone alone: the leakage
            weak_read = p.weak + float(sp.scalloping_db(_win(name, N), [f2 - round(f2)])[0])
            self.margins[name] = weak_read - Sl[k2]
            if name == p.win:
                ps.line("sel", fb[keep], S[keep], color=NAVY, width=2.2, name=name, z=2)
                sel_S = S
            elif p.others:
                ps.line("o" + name, fb[keep], S[keep], color=WIN_COLORS[name],
                        width=1.0, alpha=0.55, name=name)
        kb = np.arange(18, 93)
        ps.scatter("bins", kb, sel_S[kb * pad], color=RED, size=5, name="the DFT's actual bins")
        ps.vline("weak", f2, color=GREEN, style=":", label=f"weak tone ({p.weak:.0f} dB)", label_pos=0.93)
        ps.hline("wl", p.weak, color=GREEN, style=":", width=0.8)
        # scalloping
        d = np.linspace(-0.5, 0.5, 201)
        pc = self.plot("scal")
        for name in WIN_NAMES:
            m = _winmet(name, 256)
            pc.line("s" + name, d, sp.scalloping_db(_win(name, 256), d),
                    color=WIN_COLORS[name],
                    width=2.6 if name == p.win else 1.3, name=f"{name} ({m['scallop']:.2f} dB)")
        read = float(sp.scalloping_db(_win(p.win, N), [p.off])[0])
        pc.scatter("now", [p.off], [read], color=RED, size=13, symbol="d")
        # window shape
        psh = self.plot("shape")
        nn = np.arange(N) / N
        for name in WIN_NAMES:
            if name == p.win:
                psh.line("w", nn, _win(name, N), color=NAVY, width=2.4, fill=0, fill_alpha=0.12)
            elif p.others:
                psh.line("w" + name, nn, _win(name, N), color=GRAY, width=1.0, alpha=0.6)
        m = _winmet(p.win, N)
        self.readout(side=m["sidelobe"], enbw=m["enbw"], read=read, margin=self.margins[p.win])

    def story(self, p):
        m = _winmet(p.win, self.N)
        mg = self.r.get("margin", 0)
        s = (f"<p>A DFT only sees N samples: it looks at the signal through a <b>window</b>, and "
             f"every tone appears as a copy of the window's spectrum. Its main lobe sets the "
             f"resolution; its sidelobes <b>leak</b> the strong tone's power across the band. The "
             f"{p.win} window's sidelobes peak at {v(m['sidelobe'], '.0f', 'dB')}.</p>")
        if mg > 3:
            s += (f"<p>The weak tone {v(p.sep, '.1f', 'bins')} away stands "
                  f"{v(mg, '.0f', 'dB')} above the leakage: " + good("visible") + ".</p>")
        else:
            s += (f"<p>The weak tone is " + bad("buried") + f" under the strong tone's skirt (margin "
                  f"{v(mg, '.0f', 'dB')}). A window with lower sidelobes would reveal it, at the cost "
                  f"of a wider main lobe.</p>")
        s += (f"<p>The red dots are the bins a plain DFT computes. A tone {v(p.off, '.2f', 'bins')} "
              f"off a bin is read {v(-self.r.get('read', 0), '.2f', 'dB')} low: the <b>picket-fence</b> "
              f"or scalloping loss (bottom left). The flat-top window is ugly for resolution but reads "
              f"amplitudes to 0.01 dB: spectrum analysers use it for power measurements.</p>")
        return "<h3>Seeing through a window</h3>" + s + keybox(
            "Main-lobe width vs sidelobe level vs amplitude accuracy: no window wins all three. "
            "Chapter 2 picks Blackman–Harris to see spurs 50 dB down.")


# =============================================================================== 3. zero padding
class ZeroPadding(Experiment):
    title = "Zero padding is not resolution"
    blurb = "Two close tones: more FFT points draw a smoother curve; only a longer look separates them."
    book = "sec:ch02:dft"
    controls = [
        Choice("N", "Record length N", ["32", "64", "128", "256", "512"], "64", style="menu",
               help="Number of real samples the DFT sees (fs = 1 kHz)"),
        Choice("pad", "Zero padding", ["×1", "×4", "×16", "×64"], "×16", style="menu"),
        Slider("df", "Tone spacing", 2, 60, 10, step=0.5, unit="Hz"),
        Choice("win", "Window", ["Rectangular", "Hann"], "Rectangular"),
    ]
    plots = [
        SpectrumPlot("spec", "What the DFT shows", x="frequency (Hz)", y="level (dB)",
                     xlim=(120, 330), ylim=(-50, 6), legend="tr"),
        Plot("time", "The record: two tones beating", x="time (ms)", y="amplitude",
             xlim=(0, 600), ylim=(-2.4, 3.0), legend="tl", legend_cols=2),
    ]
    layout = [["spec"], ["time"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bin", "Bin spacing fs/N", "Hz", ".2f"),
        Readout("grid", "Display grid", "Hz", ".2f"),
        Readout("T", "Observation time T", "ms", ".0f"),
        Readout("res", "Two tones?", "", None),
    ]
    fs, f1 = 1000.0, 200.3

    def analyse(self, N, pad, df, win):
        n = np.arange(N) - (N - 1) / 2            # both tones in phase at the record centre
        x = np.cos(2 * np.pi * self.f1 * n / self.fs) + np.cos(2 * np.pi * (self.f1 + df) * n / self.fs)
        w = np.ones(N) if win == "Rectangular" else np.hanning(N + 1)[:-1]
        M = N * pad
        X = np.abs(np.fft.rfft(x * w, M)) / np.sum(w) * 2
        f = np.fft.rfftfreq(M, 1 / self.fs)
        Xb = np.abs(np.fft.rfft(x * w)) / np.sum(w) * 2
        fb = np.fft.rfftfreq(N, 1 / self.fs)
        return f, 20 * np.log10(X + 1e-9), fb, 20 * np.log10(Xb + 1e-9)

    def resolved(self, f, S, df):
        lo, hi = self.f1 - 0.25 * df, self.f1 + 1.25 * df
        k = np.flatnonzero((f >= lo) & (f <= hi))
        if len(k) < 5:
            return False, 0.0
        seg = S[k]
        pk, _ = sps.find_peaks(seg)
        if len(pk) < 2:
            return False, 0.0
        top = pk[np.argsort(seg[pk])[-2:]]
        a, b = sorted(top)
        dip = min(seg[a], seg[b]) - seg[a:b + 1].min()
        return dip >= 3.0, float(dip)

    def nmin(self, df, win):
        for N in (32, 64, 128, 256, 512):
            f, S, _, _ = self.analyse(N, 64, df, win)
            if self.resolved(f, S, df)[0]:
                return N
        return None

    def update(self, p):
        N, pad = int(p.N), int(p.pad[1:])
        f, S, fb, Sb = self.analyse(N, pad, p.df, p.win)
        ps = self.plot("spec")
        ps.vline("t1", self.f1, color=GREEN, style=":")
        ps.vline("t2", self.f1 + p.df, color=GREEN, style=":", label="true tones", label_pos=0.08)
        ps.line("pad", f, S, color=NAVY, width=2.0, name=f"zero-padded ×{pad}" if pad > 1 else "DFT (no padding)")
        ps.stems("bins", fb, Sb, color=RED, base=-50, size=7, name=f"the {N} samples' own bins")
        ok, dip = self.resolved(f, S, p.df)
        # time view
        T = N / self.fs * 1e3
        tt = np.arange(0, 0.6, 1 / self.fs / 4)
        tc = (N - 1) / 2 / self.fs
        x = np.cos(2 * np.pi * self.f1 * (tt - tc)) + np.cos(2 * np.pi * (self.f1 + p.df) * (tt - tc))
        pt = self.plot("time")
        pt.band("rec", 0, T, color=GREEN, alpha=0.10)
        pt.line("all", tt * 1e3, x, color=GRAY, width=0.8, alpha=0.7, name="the signal (forever)")
        k = tt * 1e3 <= T
        pt.line("recl", tt[k] * 1e3, x[k], color=NAVY, width=1.6, name="what the DFT sees")
        pt.line("env", tt * 1e3, 2 * np.abs(np.cos(np.pi * p.df * (tt - tc))), color=RED, width=1.4,
                style="--", name=f"beat envelope (period {1e3 / p.df:.0f} ms)")
        self.readout(bin=self.fs / N, grid=self.fs / (N * pad), T=T,
                     res=f"resolved ({dip:.0f} dB dip)" if ok else "merged")

    @property
    def challenges(self):
        nm = self.nmin(25.0, "Rectangular")
        return [
            Challenge("Padding is not resolution: N = 64, tones 10 Hz apart, pad ×64. Still one peak?",
                      lambda s: s.p.N == "64" and s.p.pad == "×64" and s.p.df <= 10.5
                      and s.r.res == "merged"),
            Challenge("Resolve tones only 5 Hz apart with the Hann window (a dip of 3 dB or more).",
                      lambda s: s.p.win == "Hann" and s.p.df <= 5.05 and s.r.res.startswith("resolved"),
                      hint="Resolution is about 1/T: you need to watch for longer."),
            Challenge("Find the shortest record that resolves tones 25 Hz apart with the rectangular window.",
                      lambda s: s.p.win == "Rectangular" and abs(s.p.df - 25) < 0.3
                      and s.p.N == str(nm) and s.r.res.startswith("resolved"),
                      hint="Try the record lengths from the shortest up; padding does not matter."),
        ]

    def story(self, p):
        N, pad = int(p.N), int(p.pad[1:])
        T = N / self.fs
        res = self.r.get("res", "")
        s = (f"<p>The record lasts {v(T * 1e3, '.0f', 'ms')}, so the DFT's bins are "
             f"{v(1 / T, '.1f', 'Hz')} apart. Your tones are {v(p.df, '.1f', 'Hz')} apart: "
             + (good("resolved") if res.startswith("resolved") else bad("merged into one lump")) + ".</p>")
        if pad > 1:
            s += (f"<p>Zero padding ×{pad} computes {pad}× more points of the <i>same</i> curve "
                  f"(the blue line through the red stems). It interpolates. It cannot add "
                  f"information that the {N} samples do not contain.</p>")
        s += (f"<p>Look at the bottom plot: two tones beat with period 1/Δf = "
              f"{v(1e3 / p.df, '.0f', 'ms')}. If the record (green) is shorter than one beat, "
              f"nothing in the data says there are two tones. To resolve Δf you must observe for "
              f"about 1/Δf: <b>resolution = 1/T</b>.</p>")
        return "<h3>More points, same picture</h3>" + s + keybox(
            "Frequency resolution ≈ 1/(observation time). Zero padding only interpolates.")


# =============================================================================== 4. uncertainty
PULSES = ["Gaussian", "Raised cosine", "Half sine", "Triangle", "Rectangle"]
P_COL = {"Gaussian": NAVY, "Raised cosine": GREEN, "Half sine": ORANGE, "Triangle": PURPLE,
         "Rectangle": RED}


def pulse(kind, t, w):
    if kind == "Gaussian":
        return np.exp(-np.pi * t ** 2 / w ** 2)
    if kind == "Raised cosine":
        return np.where(np.abs(t) < w, 0.5 * (1 + np.cos(np.pi * t / w)), 0.0)
    if kind == "Half sine":
        return np.where(np.abs(t) < 0.75 * w, np.cos(np.pi * t / (1.5 * w)), 0.0)
    if kind == "Triangle":
        return np.maximum(0.0, 1 - np.abs(t) / w)
    return np.where(np.abs(t) < 0.5 * w, 1.0, 0.0)


@lru_cache(maxsize=256)
def pulse_analysis(kind, w):
    N, dt = 1 << 15, 1 / 256
    t = (np.arange(N) - N // 2) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, dt))
    x = pulse(kind, t, w)
    x = x / np.sqrt(np.sum(x ** 2) * dt)
    X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
    s_t, s_f = sp.rms_widths(t, x, f, X)
    E = np.abs(X) ** 2
    above = f[(E >= E.max() * 1e-4) & (f >= 0)]
    return t, x, f, X, s_t, s_f, float(above.max())


class Uncertainty(Experiment):
    title = "The uncertainty principle"
    blurb = "Short or narrowband: pick one. Only the Gaussian reaches the limit."
    book = "sec:ch02:uncertainty"
    controls = [
        Choice("kind", "Pulse shape", PULSES, "Rectangle", style="menu"),
        Slider("w", "Pulse width", 0.25, 2.0, 1.0, step=0.01, unit="s"),
        Toggle("all", "Show every pulse on the map", True),
    ]
    plots = [
        Plot("time", "Unit-energy pulse x(t)", x="time (s)", y="amplitude", xlim=(-2.5, 2.5),
             ylim=(-0.1, 2.9), legend="tr"),
        SpectrumPlot("spec", "Energy spectrum |X(f)|²", x="frequency (Hz)", y="level (dB)",
                     xlim=(0, 6), ylim=(-80, 4), legend=None),
        Plot("plane", "Duration vs bandwidth: the forbidden zone", x="rms duration σ_t (s)",
             y="rms bandwidth σ_f (Hz)", logx=True, logy=True, xlim=(0.04, 1.5), ylim=(0.04, 8),
             legend="bl"),
    ]
    layout = [["time", "plane"], ["spec", "plane"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("st", "RMS duration σ_t", "s", ".3f"),
        Readout("sf", "RMS bandwidth σ_f", "Hz", ".3f"),
        Readout("prod", "σ_t·σ_f relative to 1/4π", "×", ".3f", good=lambda x: x < 1.01),
        Readout("b40", "−40 dB bandwidth", "Hz", ".2f"),
    ]
    challenges = [
        Challenge("Reach the uncertainty limit: σ_t·σ_f within 1 % of 1/4π.",
                  lambda s: s.r.prod < 1.01),
        Challenge("Squeeze a Gaussian to σ_t ≤ 0.1 s and watch its bandwidth: σ_f must exceed 0.79 Hz.",
                  lambda s: s.p.kind == "Gaussian" and s.r.st <= 0.1 and s.r.sf > 0.79),
        Challenge("Find a pulse that is not Gaussian yet within 12 % of the limit.",
                  lambda s: s.p.kind != "Gaussian" and s.r.prod < 1.12,
                  hint="Smooth pulses with no corners do best."),
    ]

    def update(self, p):
        t, x, f, X, s_t, s_f, b40 = pulse_analysis(p.kind, round(p.w, 3))
        pt = self.plot("time")
        k = np.abs(t) <= 2.6
        pt.line("x", t[k], x[k], color=P_COL[p.kind], width=2.4, name=p.kind, fill=0, fill_alpha=0.12)
        pt.band("sig", -s_t, s_t, color=GRAY, alpha=0.10)
        pt.text("sl", s_t, float(x.max()) + 0.12, f" ±σ_t = ±{s_t:.2f} s", color=GRAY, size=9)
        E = np.abs(X) ** 2
        pf = self.plot("spec")
        kf = (f >= 0) & (f <= 6.2)
        pf.line("E", f[kf], db(E[kf] / E.max()), color=P_COL[p.kind], width=2.0)
        pf.vline("sf", s_f, color=GRAY, style="--", label="σ_f", label_pos=0.9)
        # the plane
        pp = self.plot("plane")
        xs = np.logspace(np.log10(0.04), np.log10(1.5), 100)
        lim = 1 / (4 * np.pi * xs)
        pp.line("forb", xs, lim, color=RED, width=0.5, alpha=0.3, fill=0.041, fill_alpha=0.10)
        pp.line("lim", xs, lim, color=GRAY, width=2.0, name="σ_t·σ_f = 1/4π (the limit)")
        pp.text("ftxt", 0.5, 0.05, "impossible", color=RED, size=10, bold=True)
        if p.all:
            for name in PULSES:
                if name == p.kind:
                    continue
                pts = [pulse_analysis(name, ww)[4:6] for ww in (0.25, 0.5, 1.0, 2.0)]
                a, b = zip(*pts)
                pp.line("c" + name, a, b, color=P_COL[name], width=1.2, alpha=0.6, style=":")
                pp.scatter("p" + name, a, b, color=P_COL[name], size=6, name=name)
        pp.scatter("now", [s_t], [s_f], color=P_COL[p.kind], size=16, symbol="d",
                   outline=None, name=f"{p.kind} (yours)")
        self.readout(st=s_t, sf=s_f, prod=4 * np.pi * s_t * s_f, b40=b40)

    def story(self, p):
        prod = self.r.get("prod", 1)
        s = (f"<p>Squeeze a pulse in time and its spectrum spreads: σ_t = {v(self.r.get('st', 0), '.3f', 's')} "
             f"and σ_f = {v(self.r.get('sf', 0), '.3f', 'Hz')}. Drag the width slider: the point "
             f"slides along a line parallel to the limit. Their product does not change.</p>")
        if p.kind == "Gaussian":
            s += ("<p>" + good("The Gaussian sits exactly on the limit") + ": σ_t·σ_f = 1/4π. No "
                  "other shape is as compact in both domains at once, which is why GMSK (GSM, "
                  "Bluetooth) and many radars use Gaussian shaping.</p>")
        elif p.kind == "Rectangle":
            s += ("<p>The rectangle is short, but its sinc spectrum decays so slowly (−6 dB per octave) "
                  "that σ_f is infinite; here it is limited only by the simulation's sample rate. "
                  "Corners cost bandwidth.</p>")
        else:
            s += (f"<p>The {p.kind.lower()} is {v(100 * (prod - 1), '.0f', '%')} above the limit. "
                  "Its corners (or the kinks where it meets zero) put power into high frequencies.</p>")
        return "<h3>Short or narrow, not both</h3>" + s + keybox(
            "σ_t·σ_f ≥ 1/(4π), with equality only for the Gaussian. The same maths is Heisenberg's.")


# =============================================================================== 5. convolution
X_KINDS = ["Rectangular pulse", "Two pulses", "Step", "Triangle"]
H_KINDS = ["RC low-pass", "Moving average", "Echo (two paths)", "Edge detector"]


class ConvolutionMovie(Experiment):
    title = "Convolution in slow motion"
    blurb = "Flip, slide, multiply, integrate: watch an output being built point by point."
    book = "sec:lti"
    animate = True
    autoplay = True
    fps = 20
    controls = [
        Choice("x", "Input x(t)", X_KINDS, style="menu"),
        Choice("h", "System h(t)", H_KINDS, style="menu"),
        Slider("w", "Width or delay of h(t)", 0.1, 2.0, 0.5, step=0.01, unit="s"),
        Slider("t", "Time t", -1.0, 4.5, 0.4, step=0.01, unit="s"),
        Slider("speed", "Playback speed", 0.1, 2.0, 0.6, step=0.05, unit="s/s"),
    ]
    plots = [
        Plot("slide", "Flip and slide: x(τ), h(t − τ) and their product", x="τ (s)",
             y="amplitude", xlim=(-1.5, 4.6), ylim=(-1.3, 2.3), legend="tl", legend_cols=3),
        Plot("out", "The output y(t) = ∫ x(τ) h(t − τ) dτ", x="t (s)", y="y(t)",
             xlim=(-1.5, 4.6), ylim=(-1.3, 1.8), legend="tl", legend_cols=2),
    ]
    layout = [["slide"], ["out"]]
    readouts = [
        Readout("t", "Time t", "s", ".2f"),
        Readout("y", "Output y(t)", "", ".3f"),
        Readout("peak", "Output peak", "", ".3f"),
        Readout("len", "Output duration", "s", ".2f"),
    ]
    challenges = [
        Challenge("Drag t to the moment the RC output peaks (rectangular input, within 0.03 s).",
                  lambda s: s.p.x == "Rectangular pulse" and s.p.h == "RC low-pass"
                  and abs(s.p.t - 1.0) < 0.03,
                  hint="The capacitor charges for as long as the input is on."),
        Challenge("Turn a rectangle into a triangle: convolve it with a moving average of the same width.",
                  lambda s: s.p.x == "Rectangular pulse" and s.p.h == "Moving average"
                  and abs(s.p.w - 1.0) < 0.02),
        Challenge("Detect both edges of each of the two pulses with an edge detector no wider than 0.2 s.",
                  lambda s: s.p.x == "Two pulses" and s.p.h == "Edge detector" and s.p.w <= 0.2),
    ]
    dt = 0.005

    def setup(self):
        self.tn = 0.4
        self._last_t = None

    def signals(self, p):
        dt = self.dt
        tau = np.arange(-1.5, 4.6 + dt / 2, dt)
        if p.x == "Rectangular pulse":
            x = ((tau >= 0) & (tau < 1)).astype(float)
        elif p.x == "Two pulses":
            x = ((tau >= 0) & (tau < 0.5)).astype(float) - 0.6 * ((tau >= 1.2) & (tau < 1.9))
        elif p.x == "Step":
            x = (tau >= 0).astype(float)
        else:
            x = np.maximum(0, 1 - np.abs(tau - 0.5) / 0.5)
        th = np.arange(0, 6, dt)
        w = p.w
        if p.h == "RC low-pass":
            h = np.exp(-th / w) / w
        elif p.h == "Moving average":
            h = (th < w) / w
        elif p.h == "Echo (two paths)":
            h = ((th < 0.05) + 0.6 * ((th >= w) & (th < w + 0.05))) / 0.05
        else:
            h = (2.0 * (th < w / 2) - 2.0 * ((th >= w / 2) & (th < w))) / w
        y = np.convolve(x, h)[:len(tau)] * dt
        return tau, x, th, h, y

    def update(self, p):
        if self._last_t is None or abs(p.t - self._last_t) > 1e-9:
            self.tn = p.t
        self._last_t = p.t
        self.cache = self.signals(p)
        self.draw(p)

    def tick(self, p):
        self.tn += p.speed / self.fps
        if self.tn > 4.5:
            self.tn = -1.0
        self.draw(p)

    def draw(self, p):
        tau, x, th, h, y = self.cache
        t0 = self.tn
        hs = np.interp(t0 - tau, th, h, left=0, right=0)          # h(t − τ)
        hmax = np.max(np.abs(h)) or 1.0
        prod = x * hs
        ps = self.plot("slide")
        ps.hline("z", 0, color=GRAY, style="-", width=0.7)
        ps.line("prod", tau, prod / hmax, color=GREEN, width=1.0, fill=0, fill_alpha=0.35,
                name="product (shaded area = y(t))")
        ps.line("x", tau, x, color=NAVY, width=2.4, name="x(τ)")
        ps.line("h", tau, hs / hmax, color=RED, width=2.0, name="h(t − τ), flipped and slid")
        ps.vline("t", t0, color=PURPLE, style="--", label=f"t = {t0:.2f} s", label_pos=0.08)
        yt = float(np.interp(t0, tau, y))
        po = self.plot("out")
        po.hline("z", 0, color=GRAY, style="-", width=0.7)
        po.line("full", tau, y, color=GRAY, width=1.2, style=":", name="the finished output")
        k = tau <= t0
        po.line("y", tau[k], y[k], color=GREEN, width=2.6, name="built so far")
        po.line("xin", tau, x, color=NAVY, width=1.0, alpha=0.35)
        po.scatter("now", [t0], [yt], color=RED, size=12)
        po.vline("t", t0, color=PURPLE, style="--")
        big = np.flatnonzero(np.abs(y) > 0.01 * max(np.abs(y).max(), 1e-9))
        dur = (tau[big[-1]] - tau[big[0]]) if len(big) else 0.0
        if p.x == "Step" and len(big):
            dur = tau[-1] - tau[big[0]]
        self.readout(t=t0, y=yt, peak=float(y[np.argmax(np.abs(y))]), len=dur)

    def story(self, p):
        s = ("<p>To find the output at one instant t: flip the impulse response, slide it to t "
             "(red), multiply it by the input (blue), and add up the product (the green area). "
             "That number is y(t). Press Play and the movie repeats it for every t.</p>")
        if p.h == "RC low-pass":
            s += (f"<p>An RC circuit remembers the past with a fading memory of time constant "
                  f"{v(p.w, '.2f', 's')}: the output rises while the input is on and decays after.</p>")
        elif p.h == "Moving average":
            s += (f"<p>A moving average adds up the last {v(p.w, '.2f', 's')} of input. Rectangle "
                  "into rectangle gives a ramp up and a ramp down: a triangle (a trapezoid if the "
                  "widths differ). In the frequency domain, sinc × sinc.</p>")
        elif p.h == "Echo (two paths)":
            s += (f"<p>Two paths, the second {v(p.w, '.2f', 's')} late and at 60 %: the output is the "
                  "input plus a delayed, weaker copy. This is multipath, and in Chapter 12 an "
                  "equaliser has to undo it.</p>")
        else:
            s += ("<p>The edge detector's response is a positive half and a negative half: it "
                  "subtracts the recent past from the more recent past. Flat input gives zero; "
                  "every edge gives a spike. A crude differentiator.</p>")
        s += (f"<p>The output lasts about as long as the input plus the impulse response: "
              f"{v(self.r.get('len', 0), '.2f', 's')} here.</p>")
        return "<h3>Flip, slide, multiply, add</h3>" + s + keybox(
            "y(t) = ∫ x(τ) h(t − τ) dτ. Every linear time-invariant system does exactly this, and "
            "in the frequency domain it is just Y(f) = X(f)·H(f).")


# =============================================================================== 6. group delay
SYSTEMS = ["Pure delays (τ_p, τ_g)", "Dispersion (quadratic phase)", "Chebyshev low-pass",
           "Bessel low-pass"]


class GroupDelay(Experiment):
    title = "Phase delay, group delay, dispersion"
    blurb = "The envelope and the carrier can arrive at different times, or even travel backwards."
    book = "sec:ch02:groupdelay"
    controls = [
        Choice("sys", "System", SYSTEMS, style="menu"),
        Slider("tg", "Group delay τ_g", 0.0, 2.0, 1.2, step=0.05, unit="s",
               enabled_if=lambda p: p.sys.startswith("Pure")),
        Slider("tp", "Phase delay τ_p", -0.5, 1.5, 0.3, step=0.05, unit="s",
               enabled_if=lambda p: p.sys.startswith("Pure")),
        Slider("beta", "Dispersion β", 0.0, 0.3, 0.05, step=0.005, unit="s²",
               enabled_if=lambda p: p.sys.startswith("Dispersion")),
        IntSlider("order", "Filter order", 2, 8, 5,
                  enabled_if=lambda p: "pass" in p.sys),
        Slider("fcut", "Filter cut-off", 0.5, 4.0, 1.5, step=0.05, unit="Hz",
               enabled_if=lambda p: "pass" in p.sys),
    ]
    plots = [
        Plot("t", "Input and output", x="time (s)", y="amplitude", xlim=(-1.5, 5), ylim=(-1.4, 1.75),
             legend="tl", legend_cols=3),
        Plot("mag", "Magnitude |H(f)|", x="frequency (Hz)", y="|H(f)| (dB)", ylim=(-40, 3), legend=None),
        Plot("gd", "Group delay τ_g(f) = −(1/2π) dφ/df", x="frequency (Hz)", y="delay (s)",
             legend="tr"),
    ]
    layout = [["t", "t"], ["mag", "gd"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("env", "Envelope arrives after", "s", ".2f"),
        Readout("width", "Pulse width", "× input", ".2f"),
        Readout("over", "Overshoot", "%", ".1f"),
        Readout("ripple", "Group-delay spread", "s", ".2f",
                help="Max − min group delay over the passband (pulse-shape distortion)"),
    ]
    challenges = [
        Challenge("Do what the ionosphere does to GPS: delay the envelope by at least 0.5 s while "
                  "advancing the carrier (τ_p < 0).",
                  lambda s: s.p.sys.startswith("Pure") and s.p.tg >= 0.5 and s.p.tp < 0),
        Challenge("Spread the pulse to at least three times its width with dispersion alone.",
                  lambda s: s.p.sys.startswith("Dispersion") and s.r.width >= 3),
        Challenge("Pass the pulse through a 6th-order (or higher) low-pass with under 1 % overshoot.",
                  lambda s: "pass" in s.p.sys and s.p.order >= 6 and s.r.over < 1,
                  hint="Overshoot comes from group delay that varies across the passband."),
        Challenge("Make a Chebyshev filter ring: overshoot above 10 %.",
                  lambda s: s.p.sys.startswith("Cheb") and s.r.over > 10,
                  hint="Ringing grows with the order: steeper skirts, bigger group-delay peak."),
    ]
    fc = 10.0

    def update(self, p):
        if "pass" in p.sys:
            return self.filters(p)
        fs, N = 200.0, 1 << 14
        t = (np.arange(N) - N // 4) / fs
        f = np.fft.fftfreq(N, 1 / fs)
        env = np.exp(-np.pi * (t / 0.6) ** 2)
        x = env * np.cos(2 * np.pi * self.fc * t)
        af = np.abs(f)
        if p.sys.startswith("Pure"):
            phi = 2 * np.pi * (self.fc * p.tp + (af - self.fc) * p.tg)
            tg_f = np.full_like(af, p.tg)
        else:
            d0 = 1.0
            phi = 2 * np.pi * (self.fc * d0 + (af - self.fc) * d0) + 2 * np.pi ** 2 * p.beta * (af - self.fc) ** 2
            tg_f = d0 + 2 * np.pi * p.beta * (af - self.fc)
        y = np.real(np.fft.ifft(np.fft.fft(x) * np.exp(-1j * phi * np.sign(f))))
        ey = np.abs(sps.hilbert(y))
        k = (t >= -1.5) & (t <= 5)
        pt = self.plot("t")
        pt.line("x", t[k], x[k], color=GRAY, width=1.0, alpha=0.8, name="input")
        pt.line("y", t[k], y[k], color=NAVY, width=1.2, name="output")
        pt.line("ey", t[k], ey[k], color=RED, width=2.0, style="--", name="output envelope")
        tenv = float(t[np.argmax(ey)])
        pt.vline("e0", 0, color=GRAY, style=":")
        pt.vline("e1", tenv, color=RED, style=":", label=f"envelope +{tenv:.2f} s", label_pos=0.06)
        w_in = np.sqrt(np.sum(t ** 2 * env ** 2) / np.sum(env ** 2))
        w_out = np.sqrt(np.sum((t - tenv) ** 2 * ey ** 2) / np.sum(ey ** 2))
        fp = np.linspace(0.5, 20, 400)
        pm = self.plot("mag")
        pm.line("m", fp, np.zeros_like(fp), color=NAVY, width=2.0)
        pm.band("sig", self.fc - 1.5, self.fc + 1.5, color=GREEN, alpha=0.10)
        pm.set_xlim(0, 20)
        pg = self.plot("gd")
        tgp = np.interp(fp, af[:N // 2], tg_f[:N // 2])
        pg.line("tg", fp, tgp, color=NAVY, width=2.2, name="group delay τ_g(f)")
        if p.sys.startswith("Pure"):
            phase_delay = (self.fc * p.tp + (fp - self.fc) * p.tg) / fp
            pg.line("tp", fp, phase_delay, color=ORANGE, width=1.6, style="--",
                    name="phase delay −φ(f)/(2πf)")
        pg.band("sig", self.fc - 1.5, self.fc + 1.5, color=GREEN, alpha=0.10)
        pg.vline("fc", self.fc, color=GRAY, style=":", label="carrier", label_pos=0.1)
        pg.set_xlim(0, 20)
        pg.set_ylim(-1.0, 3.2)
        band = np.abs(fp - self.fc) <= 1.5
        self.readout(env=tenv, width=w_out / w_in, over="—",
                     ripple=float(tgp[band].max() - tgp[band].min()))

    def filters(self, p):
        fs = 50.0
        t = np.arange(-1.5, 6, 1 / fs)
        x = ((t >= 0) & (t < 1.0)).astype(float)
        if p.sys.startswith("Cheb"):
            sos = sps.cheby1(p.order, 1.0, p.fcut, fs=fs, output="sos")
        else:
            sos = sps.bessel(p.order, p.fcut, fs=fs, norm="mag", output="sos")
        y = sps.sosfilt(sos, x)
        pt = self.plot("t")
        pt.line("x", t, x, color=GRAY, width=2.4, alpha=0.7, name="input pulse (1 s)")
        pt.line("y", t, y, color=NAVY, width=2.2, name=f"{p.sys.split()[0]} output")
        pt.hline("one", 1.0, color=GREEN, style=":", width=1.0)
        over = 100 * max(0.0, y.max() - 1.0)
        # envelope delay = time of 50 % crossing on the rising edge
        i50 = np.flatnonzero(y >= 0.5)
        tenv = float(t[i50[0]]) if len(i50) else 0.0
        f = np.linspace(0.02, 3 * p.fcut, 600)
        _, H = sps.sosfreqz(sos, worN=f, fs=fs)
        ph = np.unwrap(np.angle(H))
        tg = -np.gradient(ph, 2 * np.pi * f)
        pm = self.plot("mag")
        pm.line("m", f, 20 * np.log10(np.abs(H) + 1e-12), color=NAVY, width=2.0)
        pm.vline("fc", p.fcut, color=GRAY, style=":", label="cut-off", label_pos=0.15)
        pm.set_xlim(0, 3 * p.fcut)
        pg = self.plot("gd")
        pg.line("tg", f, tg, color=NAVY, width=2.2, name="group delay τ_g(f)")
        pg.vline("fc", p.fcut, color=GRAY, style=":")
        pg.set_xlim(0, 3 * p.fcut)
        pg.set_ylim(0, max(1.0, float(np.percentile(tg, 99)) * 1.2))
        pas = f <= 0.8 * p.fcut
        self.readout(env=tenv, width="—", over=over, ripple=float(tg[pas].max() - tg[pas].min()))

    def story(self, p):
        if p.sys.startswith("Pure"):
            s = (f"<p>A filter's phase φ(f) has two meanings. The carrier is delayed by the "
                 f"<b>phase delay</b> τ_p = −φ(f_c)/(2πf_c) = {v(p.tp, '.2f', 's')}; the envelope "
                 f"riding on it by the <b>group delay</b> τ_g = −(1/2π)dφ/df = {v(p.tg, '.2f', 's')}. "
                 f"Watch the red envelope arrive {v(self.r.get('env', 0), '.2f', 's')} late while the "
                 f"carrier under it has shifted by a different amount.</p>")
            if p.tp < 0 <= p.tg:
                s += ("<p>" + good("Carrier advanced, envelope delayed") + ": exactly what the "
                      "ionosphere does. At 10 TEC units it delays GPS L1 code by about 1.62 m and "
                      "advances the carrier phase by the same 1.62 m. Dual-frequency receivers "
                      "measure the difference and cancel the 1/f² term.</p>")
        elif p.sys.startswith("Dispersion"):
            s = (f"<p>Now the group delay changes linearly with frequency (slope 2πβ). The "
                 f"pulse's lower frequencies arrive at a different time from its upper ones, so it "
                 f"<b>spreads</b>: {v(self.r.get('width', 1), '.2f', '×')} its original width, and "
                 f"its peak falls. Energy is conserved, only rearranged in time.</p>"
                 "<p>This is chromatic dispersion in optical fibre (Chapter 24) and, done on "
                 "purpose, the chirp in a radar pulse compressor.</p>")
        else:
            cheb = p.sys.startswith("Cheb")
            s = (f"<p>A real low-pass filter of order {v(p.order, 'd')}. Its group delay (bottom "
                 f"right) " + ("peaks sharply near the cut-off, so the pulse's edge components "
                               "arrive late and pile up as <b>ringing</b>: "
                               if cheb else
                               "is almost flat across the passband, so all components arrive "
                               "together and the pulse keeps its shape: ")
                 + f"overshoot {v(self.r.get('over', 0), '.1f', '%')}.</p>"
                 + ("<p>Chebyshev buys a sharp magnitude cut-off with group-delay distortion. Bessel "
                    "(Thomson) filters do the opposite: maximally flat delay, a lazier skirt. "
                    "Oscilloscope front ends and data-acquisition filters are Bessel for this reason.</p>"))
        return "<h3>When does the signal arrive?</h3>" + s + keybox(
            "Flat |H| is not enough for a distortionless channel: the group delay must be flat too.")


# =============================================================================== 7. five bandwidths
@lru_cache(maxsize=64)
def rrc_qpsk(beta):
    rng = np.random.default_rng(7)
    sps_ = 8
    sym = cl.get_constellation("qpsk").modulate(cl.random_bits(2 * 16000, rng))
    up = np.zeros(len(sym) * sps_, complex)
    up[::sps_] = sym
    s = sps.fftconvolve(up, cl.rrc_taps(max(beta, 0.01), sps_, 16))
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


@lru_cache(maxsize=1)
def rect_bpsk_metrics():
    fr = np.linspace(-1000, 1000, 2_000_001)
    Sr = np.sinc(fr) ** 2
    return sp.bandwidth_metrics(fr, Sr, ref=1.0)


class FiveBandwidths(Experiment):
    title = "Five bandwidths of one signal"
    blurb = "3 dB, noise-equivalent, null-to-null, 99 % and −26 dB: same spectrum, five answers."
    book = "sec:bandwidth"
    controls = [
        Choice("sig", "Signal", ["RRC-filtered QPSK", "Rectangular BPSK"]),
        Slider("beta", "RRC roll-off β", 0.05, 1.0, 0.35, step=0.01,
               enabled_if=lambda p: p.sig.startswith("RRC")),
        Slider("comp", "PA compression", 0.0, 0.12, 0.0, step=0.005,
               help="Third-order compression y = x − c·x|x|² (spectral regrowth)"),
    ]
    plots = [
        SpectrumPlot("psd", "One spectrum, five bandwidths", x="frequency / symbol rate",
                     y="PSD (dB, in-band = 0)", xlim=(-2.5, 2.5), ylim=(-75, 10), legend="tr"),
        BarPlot("bars", "The five numbers", y="bandwidth (× symbol rate)"),
    ]
    layout = [["psd", "bars"]]
    col_stretch = [5, 2]
    readouts = [
        Readout("b3", "3 dB bandwidth", "× Rs", ".2f"),
        Readout("bn", "Noise-equivalent", "× Rs", ".2f"),
        Readout("obw", "99 % occupied", "× Rs", ".2f"),
        Readout("b26", "−26 dB bandwidth", "× Rs", ".2f"),
    ]
    challenges = [
        Challenge("Find the roll-off whose 99 % occupied bandwidth is 1.20 Rs (±0.01), with no compression.",
                  lambda s: s.p.sig.startswith("RRC") and s.p.comp == 0 and abs(s.r.obw - 1.20) <= 0.01),
        Challenge("Spectral regrowth: at β = 0.22, compress the PA until the −26 dB bandwidth "
                  "exceeds 1.4 Rs while the 3 dB bandwidth stays below 1.05 Rs.",
                  lambda s: s.p.sig.startswith("RRC") and abs(s.p.beta - 0.22) < 0.006
                  and s.r.b26 > 1.4 and s.r.b3 < 1.05),
        Challenge("Rectangular BPSK: same noise bandwidth (1 Rs), yet a 99 % bandwidth above 20 Rs.",
                  lambda s: s.p.sig.startswith("Rect") and s.r.obw > 20),
    ]

    def update(self, p):
        pp = self.plot("psd")
        if p.sig.startswith("RRC"):
            s = rrc_qpsk(round(p.beta, 2))
            y = s - p.comp * s * np.abs(s) ** 2
            f, S = sps.welch(y, fs=8, nperseg=2048, return_onesided=False, window="blackmanharris",
                             detrend=False)
            f, S = np.fft.fftshift(f), np.fft.fftshift(S)
            m = sp.bandwidth_metrics(f, S)
            ref = np.median(S[np.abs(f) < 0.3])
            nn = None
        else:
            # constant envelope: compression only scales it, the shape is sinc²
            f = np.linspace(-4, 4, 8001)
            S = np.sinc(f) ** 2
            m = rect_bpsk_metrics()
            ref = 1.0
            nn = 2.0
        Sd = db(S / ref)
        pp.band("b3", m["f3"][0], m["f3"][1], color=GREEN, alpha=0.16)
        pp.line("S", f, Sd, color=NAVY, width=1.8, name="PSD")
        bn = m["bn"]
        pp.line("bn", [-bn / 2, -bn / 2, bn / 2, bn / 2], [-75, 4, 4, -75], color=PURPLE, width=1.6,
                style="--", name=f"noise-equivalent rectangle ({bn:.2f})")
        if abs(m["lo"]) < 2.5:
            pp.vline("lo", m["lo"], color=ORANGE, style="--")
            pp.vline("hi", m["hi"], color=ORANGE, style="--", label="99 % power", label_pos=0.45)
        else:
            pp.text("obwt", 1.25, 7, f"99 % edges at ±{m['obw'] / 2:.1f} Rs →", color=ORANGE, size=9)
        pp.line("b26", [m["fx"][0], m["fx"][1]], [-26, -26], color=RED, width=3.0,
                name=f"−26 dB width ({m['bx']:.2f})")
        pp.text("b3t", 0, -6, "3 dB", color=GREEN, anchor=(0.5, 0.5), bold=True)
        if nn:
            pp.scatter("nulls", [-1, 1], [-70, -70], color=GRAY, size=11, symbol="t",
                       name="first nulls (2 Rs apart)")
        vals = [m["b3"], bn, nn if nn else 0.0, m["obw"], m["bx"]]
        names = ["3 dB", "noise", "nulls", "99 %", "−26 dB"]
        pb = self.plot("bars")
        cols = [GREEN, PURPLE, GRAY, ORANGE, RED]
        shown = [min(x, 3.4) for x in vals]
        pb.bars("b", np.arange(5), shown, width=0.65, colors=cols)
        for i, x in enumerate(vals):
            lab = "none" if (i == 2 and not nn) else f"{x:.2f}"
            pb.text(f"t{i}", i, shown[i], lab, anchor=(0.5, 1.05), bold=True, size=9)
        pb.set_xticks([(i, n) for i, n in enumerate(names)])
        pb.set_xlim(-0.6, 4.6)
        pb.set_ylim(0, 3.9)
        self.readout(b3=m["b3"], bn=bn, obw=m["obw"], b26=m["bx"])

    def story(self, p):
        r = self.r
        if p.sig.startswith("RRC"):
            s = (f"<p>Root-raised-cosine QPSK with β = {v(p.beta)}: the 3 dB and noise-equivalent "
                 f"widths are both close to the symbol rate ({v(r.get('b3', 0))} and "
                 f"{v(r.get('bn', 0))} Rs) while 99 % of the power fits in {v(r.get('obw', 0))} Rs. "
                 f"There are no nulls, so \"null-to-null\" means nothing here.</p>")
            if p.comp > 0:
                s += (f"<p>The compressing PA multiplies the signal by its own envelope, creating "
                      f"third-order products that spill into the neighbours: <b>spectral regrowth</b>. "
                      f"The 3 dB width barely moves; the −26 dB width jumps to {v(r.get('b26', 0))} Rs. "
                      f"That is why emission masks and ACLR are specified far down the skirt.</p>")
        else:
            s = (f"<p>Rectangular BPSK has a sinc² spectrum. Its noise bandwidth is exactly 1 Rs, "
                 f"its first nulls are 2 Rs apart, but the sidelobes fall so slowly that 99 % of the "
                 f"power needs {v(r.get('obw', 0), '.1f')} Rs. Constant envelope also makes it immune "
                 f"to PA compression: move that slider and nothing happens.</p>")
        s += ("<p>The noise-equivalent width sets your SNR; the occupied width is what the "
              "regulator measures. A bandwidth quoted without its definition is not a number.</p>")
        return "<h3>Which bandwidth?</h3>" + s + keybox(
            "Chapter 2's example: rectangular BPSK and RRC QPSK have the same noise bandwidth, "
            "and 99 % bandwidths twenty times apart.")


# =============================================================================== 8. spectrogram
@lru_cache(maxsize=2)
def stft_signal(kind):
    fs = 8000
    t = np.arange(0, 1.0, 1 / fs)
    if kind.startswith("Chirp"):
        return fs, (np.cos(2 * np.pi * (200 * t + 0.5 * 2600 * t ** 2))
                    + 0.6 * np.cos(2 * np.pi * 1500 * t) * ((t > 0.3) & (t < 0.36)))
    r = np.random.default_rng(5)
    syms = r.integers(0, 4, 50)
    fsk_f = np.repeat(np.array([800, 1200, 1600, 2000])[syms], int(0.02 * fs))[:len(t)]
    x = np.cos(2 * np.pi * np.cumsum(fsk_f) / fs) * ((t > 0.1) & (t < 0.7))
    return fs, x + 0.02 * r.standard_normal(len(x))


class Spectrogram(Experiment):
    title = "The spectrogram (STFT)"
    blurb = "Slide a window along the signal: short windows see when, long windows see what."
    book = "sec:ch02:stft"
    controls = [
        Choice("sig", "Signal", ["Chirp + 60 ms tone burst", "4-FSK burst (20 ms symbols)"],
               style="menu"),
        Choice("nper", "Window length", ["32", "64", "128", "256", "512", "1024"], "256",
               style="menu", help="Samples at 8 kHz"),
        Choice("win", "Window", ["Rectangular", "Hann", "Blackman–Harris"], "Hann", style="menu"),
        Slider("ovl", "Overlap", 0.0, 0.95, 0.85, step=0.05),
        Slider("tc", "Slice at time", 0.05, 0.95, 0.33, step=0.01, unit="s"),
    ]
    plots = [
        ImagePlot("sg", "Spectrogram (dB)", x="time (s)", y="frequency (kHz)"),
        SpectrumPlot("slice", "Spectrum at the slice time", x="frequency (kHz)",
                     y="level (dB)", xlim=(0, 3.2), ylim=(-60, 3), legend=None),
    ]
    layout = [["sg", "slice"]]
    col_stretch = [5, 3]
    readouts = [
        Readout("dur", "Window duration", "ms", ".1f"),
        Readout("rbw", "Frequency resolution", "Hz", ".1f", help="Noise bandwidth of one bin"),
        Readout("hop", "Time step", "ms", ".2f"),
        Readout("verdict", "Resolves", "", None),
    ]
    challenges = [
        Challenge("4-FSK: resolve both the tones (400 Hz apart) and the 20 ms symbols at once.",
                  lambda s: s.p.sig.startswith("4-FSK") and s.r.verdict == "tones and symbols"),
        Challenge("Chirp: catch the 60 ms burst's start and end sharply (window ≤ 8 ms). Now look at the chirp.",
                  lambda s: s.p.sig.startswith("Chirp") and s.r.dur <= 8.01),
        Challenge("Draw the chirp as a thin line: frequency resolution of 12 Hz or better.",
                  lambda s: s.p.sig.startswith("Chirp") and s.r.rbw <= 12),
    ]

    def update(self, p):
        fs, x = stft_signal(p.sig)
        N = int(p.nper)
        w = sp.window(p.win, N)
        nov = min(int(N * p.ovl), N - 1)
        f, t, S = sps.spectrogram(x, fs, window=w, nperseg=N, noverlap=nov, nfft=max(N, 256),
                                  detrend=False)
        Sd = db(S)
        Sd = Sd - Sd.max()
        keep = f <= 3200
        pg = self.plot("sg")
        hop = (N - nov) / fs
        pg.image("S", Sd[keep], x=(t[0] - hop / 2, t[-1] + hop / 2), y=(0, f[keep][-1] / 1e3),
                 cmap="heat", levels=(-60, 0), colorbar=True, cbar_label="dB")
        pg.vline("tc", p.tc, color=PURPLE, style="--", width=1.6, label="slice", label_pos=0.97)
        pg.set_xlim(0, 1)
        pg.set_ylim(0, 3.2)
        j = int(np.argmin(np.abs(t - p.tc)))
        ps = self.plot("slice")
        col = Sd[keep, j]
        ps.line("c", f[keep] / 1e3, col - col.max(), color=NAVY, width=1.8, fill=-60, fill_alpha=0.15)
        m = sp.window_metrics(w)
        rbw = m["enbw"] * fs / N
        dur = N / fs * 1e3
        lobe = m["mainlobe"] * fs / N
        tones = lobe < 0.75 * 400
        syms = dur <= 16.0
        verdict = ("tones and symbols" if tones and syms else "tones only" if tones
                   else "symbols only" if syms else "neither")
        if p.sig.startswith("Chirp"):
            verdict = ("burst edges" if dur <= 8.01 else "thin chirp" if rbw <= 20 else "a bit of both")
        self.readout(dur=dur, rbw=rbw, hop=hop * 1e3, verdict=verdict)

    def story(self, p):
        r = self.r
        dur, rbw = r.get("dur", 0), r.get("rbw", 0)
        s = (f"<p>The short-time Fourier transform slides a {v(dur, '.0f', 'ms')} window along "
             f"the signal and takes a DFT of each piece. Each pixel is a tile about "
             f"{v(dur, '.0f', 'ms')} wide and {v(rbw, '.0f', 'Hz')} tall: you cannot shrink both. "
             f"The tile area is fixed by the uncertainty principle.</p>")
        if p.sig.startswith("Chirp"):
            s += ("<p>A short window draws the 60 ms burst at 1.5 kHz with crisp start and end "
                  "but smears the chirp into a fat band (it moves 2.6 kHz per second, so even its "
                  "instantaneous frequency changes inside a long window). A long window draws a "
                  "thin chirp and smears the burst sideways in time.</p>")
        else:
            s += (f"<p>The 4-FSK symbols last 20 ms and the tones are 400 Hz apart. Your verdict: "
                  f"{v(r.get('verdict', ''))}. You need a window shorter than a symbol and a main "
                  f"lobe narrower than the tone spacing: 8–16 ms works.</p>")
        return "<h3>When versus what</h3>" + s + keybox(
            "Spectrogram resolution: Δt·Δf ≈ constant. Choose the window for the feature you need "
            "to see, which is exactly what an SDR waterfall's FFT-size knob does.")


# =============================================================================== 9. two-tone IP3
class TwoToneIP3(Experiment):
    title = "Two-tone test and IP3"
    blurb = "Two clean tones in, a forest of distortion out. Why IM3 is the product that hurts."
    book = "sec:ch02:nonlinear"
    controls = [
        Heading("Test signal"),
        Slider("pin", "Power per tone", -60, 10, -20, step=0.5, unit="dBm"),
        Slider("spacing", "Tone spacing", 1, 20, 10, step=1, unit="MHz"),
        Heading("Amplifier"),
        Slider("gain", "Small-signal gain", 0, 30, 15, step=0.5, unit="dB"),
        Slider("iip3", "Input IP3", -20, 30, 10, step=0.5, unit="dBm"),
        Slider("iip2", "Input IP2", 10, 80, 45, step=1, unit="dBm"),
    ]
    plots = [
        SpectrumPlot("spec", "Output spectrum (tones at 100 MHz and above)", x="frequency (MHz)",
                     y="output power (dBm)", xlim=(0, 340), ylim=(-150, 45), legend=None),
        Plot("ip", "Output vs input: slopes 1 and 3",
             x="input power per tone (dBm)", y="output power (dBm)", xlim=(-60, 30),
             ylim=(-160, 70), legend="tl"),
    ]
    layout = [["spec", "ip"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("delta", "Tone-to-IM3 ratio Δ", "dB", ".1f"),
        Readout("est", "IIP3 estimate P_in + Δ/2", "dBm", ".1f"),
        Readout("oip3", "Output IP3", "dBm", ".1f"),
        Readout("im3", "IM3 product at the output", "dBm", ".1f"),
    ]
    challenges = [
        Challenge("Chapter 2's example: −20 dBm tones with IM3 exactly 70 dB below them. What IIP3 does it take?",
                  lambda s: abs(s.p.pin + 20) < 0.3 and abs(s.r.delta - 70) < 0.5),
        Challenge("Overdrive: get close enough to the intercept that P_in + Δ/2 misjudges the "
                  "true IIP3 by more than 2 dB.",
                  lambda s: abs(s.r.est - s.p.iip3) > 2),
        Challenge("Two −25 dBm interferers reach your receiver. Keep their IM3 below −110 dBm "
                  "(referred to the input) with the lowest IIP3 that does it (within 1 dB).",
                  lambda s: abs(s.p.pin + 25) < 0.3 and s.r.im3 - s.p.gain < -110
                  and s.p.iip3 <= 18.5,
                  hint="Input-referred IM3 = 3·P_in − 2·IIP3."),
    ]
    fs, N, f1 = 1024.0, 8192, 100.0

    def amp(self, p):
        a1 = 10 ** (p.gain / 20)
        a3 = -4 / 3 * a1 / (2 * 10 ** (p.iip3 / 10))
        a2 = a1 / np.sqrt(2 * 10 ** (p.iip2 / 10))
        return lambda x: a1 * x + a2 * x ** 2 + a3 * x ** 3

    def run(self, amp, pin, f2):
        n = np.arange(self.N)
        A = np.sqrt(2 * 10 ** (pin / 10))
        x = A * (np.cos(2 * np.pi * self.f1 * n / self.fs) + np.cos(2 * np.pi * f2 * n / self.fs))
        Y = np.fft.rfft(amp(x)) / (self.N / 2)
        return 10 * np.log10(np.abs(Y) ** 2 / 2 + 1e-18)

    def update(self, p):
        amp = self.amp(p)
        f2 = self.f1 + p.spacing
        P = self.run(amp, p.pin, f2)
        f = np.fft.rfftfreq(self.N, 1 / self.fs)
        P = np.maximum(P, -145 + 0.0 * P)
        k = lambda fx: int(round(fx * self.N / self.fs))
        ps = self.plot("spec")
        ps.line("P", f, P, color=NAVY, width=1.4)
        im_lo, im_hi = 2 * self.f1 - f2, 2 * f2 - self.f1
        ps.scatter("tones", [self.f1, f2], [P[k(self.f1)], P[k(f2)]], color=NAVY, size=8)
        ps.scatter("im3", [im_lo, im_hi], [P[k(im_lo)], P[k(im_hi)]], color=RED, size=10, symbol="d")
        ps.text("tim3", im_lo, P[k(im_lo)] + 4, "IM3", color=RED, anchor=(1.0, 1.0), size=9, bold=True)
        ps.text("tim2", p.spacing, P[k(p.spacing)] + 4, "IM2", color=ORANGE, anchor=(0, 1.0), size=9)
        ps.text("th2", 2 * self.f1, P[k(2 * self.f1)] + 4, "2nd harmonics", color=ORANGE,
                anchor=(0.5, 1.0), size=9)
        ps.text("th3", 3 * self.f1, P[k(3 * self.f1)] + 4, "3rd-order", color=PURPLE,
                anchor=(0.5, 1.0), size=9)
        tone, im3 = P[k(self.f1)], P[k(im_lo)]
        delta = tone - im3
        # input-output sweep
        pins = np.arange(-60, min(22.6, p.iip3 - 2.9), 2.5)     # stop short of saturation
        tt, ii = [], []
        for q in pins:
            Q = self.run(amp, q, f2)
            tt.append(Q[k(self.f1)])
            ii.append(Q[k(im_lo)])
        tt, ii = np.array(tt), np.array(ii)
        pi_ = self.plot("ip")
        xs = np.linspace(-60, 30, 50)
        oip3 = p.iip3 + p.gain
        pi_.line("l1", xs, xs + p.gain, color=NAVY, width=1.2, style=":")
        pi_.line("l3", xs, 3 * xs - 2 * p.iip3 + p.gain, color=RED, width=1.2, style=":")
        pi_.scatter("mt", pins, tt, color=NAVY, size=7, name="tone (measured)")
        pi_.scatter("mi", pins[ii > -140], ii[ii > -140], color=RED, size=7, symbol="s",
                    name="IM3 (measured)")
        pi_.scatter("ip3", [p.iip3], [oip3], color=GREEN, size=18, symbol="star",
                    name=f"intercept ({p.iip3:.1f}, {oip3:.1f}) dBm")
        pi_.vline("now", p.pin, color=ORANGE, style="--", label="your drive", label_pos=0.08)
        pi_.set_xlim(-60, 30)
        self.readout(delta=delta, est=p.pin + delta / 2, oip3=oip3, im3=im3)

    def story(self, p):
        r = self.r
        d = r.get("delta", 0)
        s = (f"<p>Two equal tones at 100 and {100 + p.spacing:.0f} MHz go into an amplifier "
             f"y = a₁x + a₂x² + a₃x³. Out come harmonics, sum and difference products, and the "
             f"troublemakers: <b>third-order intermodulation</b> at 2f₁ − f₂ and 2f₂ − f₁ (red "
             f"diamonds), only {p.spacing:.0f} MHz from the tones, where no filter can reach them.</p>"
             f"<p>IM3 grows 3 dB for every 1 dB of drive; the tones only 1 dB. Their straight-line "
             f"extrapolations meet at the <b>third-order intercept</b>. Here the tones are "
             f"{v(d, '.1f', 'dB')} above IM3, so IIP3 ≈ P_in + Δ/2 = {v(r.get('est', 0), '.1f', 'dBm')}"
             f" (true: {p.iip3:.1f} dBm).</p>")
        if abs(r.get("est", 0) - p.iip3) > 1:
            s += ("<p>" + bad("Too close to compression") + ": the fundamental is no longer on its "
                  "slope-1 line, so the one-point estimate is wrong. Measure IP3 well below P1dB.</p>")
        return "<h3>Distortion you cannot filter</h3>" + s + keybox(
            "IM3 rises 3 dB per dB. Input-referred IM3 = 3·P_in − 2·IIP3: back off 1 dB, gain 2 dB of "
            "spurious-free range.")


# =============================================================================== the lab
LAB = st.Lab(35, "Fourier, Spectra and Bandwidth", chapter=2,
             chapter_title="Signals, Spectra and Systems",
             experiments=[FourierSeries, WindowsLeakage, ZeroPadding, Uncertainty, ConvolutionMovie,
                          GroupDelay, FiveBandwidths, Spectrogram, TwoToneIP3])

if __name__ == "__main__":
    st.run(LAB)
