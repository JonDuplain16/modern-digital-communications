"""Lab 16 · Sampling, PCM and Companding   (Chapter 5)

Run it:      python labs/lab16_pcm_companding.py
Self-test:   python labs/lab16_pcm_companding.py --selftest

Every digital system starts by turning a waveform into numbers: sample it in time, quantize it
in amplitude. Telephony did it first and brilliantly: 8000 samples a second, 8 bits each,
logarithmic companding so a whisper and a shout get the same relative accuracy, and 24 such
channels woven into a 1.544 Mb/s T1 line. Nine experiments rebuild that chain, then the modern
way to convert: oversample with a crude 1-bit quantizer and shape the noise out of the band.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

from commlib import pcm
from commlib.analog import voice_like
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD)
from studio import v, keybox, good, bad


# =============================================================================== helpers
def butter_mag_db(f, fc, order):
    """Butterworth low-pass magnitude (dB)."""
    return -10 * np.log10(1 + (np.asarray(f, float) / fc) ** (2 * order))


def line_spectrum_db(x, win=None):
    """One-sided spectrum (dB re the strongest bin) of a periodic record."""
    w = np.blackman(len(x)) if win is None else win
    X = np.abs(np.fft.rfft(x * w)) ** 2
    return 10 * np.log10(X / X.max() + 1e-20)


@lru_cache(maxsize=4)
def speech(fs, dur, seed=0):
    v_ = voice_like(fs, dur, f0=(105.0, 160.0))
    return v_ / np.sqrt(np.mean(v_ ** 2))           # unit rms


# =============================================================================== 1. aliasing
class Aliasing(Experiment):
    title = "Sampling and aliasing"
    blurb = "Above fs/2, two different tones give identical samples. The filter must come first."
    book = "sec:ch05:sampling"
    controls = [
        Slider("f", "Input tone", 0, 24000, 1000, step=10, unit="Hz"),
        Slider("fs", "Sample rate", 2000, 16000, 8000, step=100, unit="Hz"),
        Heading("Anti-alias filter (cut-off 3.4 kHz)"),
        Toggle("aa", "Filter before the sampler", False),
        IntSlider("order", "Filter order", 2, 24, 8, enabled_if=lambda p: p.aa),
        Button("listen", "▶  Listen to what comes out", primary=True),
    ]
    plots = [
        Plot("time", "The tone, its samples, and the slowest sine through them", x="time (ms)",
             y="amplitude", xlim=(0, 4), ylim=(-1.4, 1.9), legend="tl", legend_cols=3),
        Plot("fold", "Folding: apparent vs real frequency", x="input frequency (kHz)",
             y="apparent frequency (kHz)", xlim=(0, 24), ylim=(0, 8.6), legend=None),
        SpectrumPlot("spec", "Spectrum of the samples: copies every fs", x="frequency (kHz)",
                     y="level (dB)", xlim=(0, 24), ylim=(-90, 8), legend=None),
    ]
    layout = [["time", "time"], ["fold", "spec"]]
    readouts = [
        Readout("app", "Apparent frequency", "Hz", ".0f"),
        Readout("zone", "Nyquist zone", "", "int"),
        Readout("att", "Anti-alias attenuation", "dB", ".1f"),
        Readout("ok", "Verdict", "", None),
    ]
    challenges = [
        Challenge("Impersonation: at fs = 8 kHz, find a tone above 8 kHz whose samples are exactly "
                  "those of a 1 kHz tone.",
                  lambda s: abs(s.p.fs - 8000) < 1 and s.p.f > 8000 and abs(s.r.app - 1000) < 5),
        Challenge("Protect the voice band: make a 4.6 kHz tone (it would alias to 3.4 kHz at 8 kHz) "
                  "at least 40 dB down at the sampler output.",
                  lambda s: abs(s.p.f - 4600) < 6 and abs(s.p.fs - 8000) < 1 and s.p.aa and s.r.att >= 40,
                  hint="Only the filter can do it at 8 kHz: how steep must it be? (Or raise fs and watch the alias leave the band.)"),
        Challenge("Make a tone above 20 kHz turn into DC: apparent frequency 0 Hz.",
                  lambda s: s.p.f > 20000 and s.r.app < 5),
    ]

    def update(self, p):
        fs, f = p.fs, p.f
        fa = float(pcm.alias_frequency(f, fs))
        zone = int(pcm.nyquist_zone(f, fs)) if f > 0 else 1
        att = -float(butter_mag_db(f, 3400, p.order)) if p.aa else 0.0
        g = 10 ** (-att / 20)
        tc = np.linspace(0, 4.1e-3, 3000)
        ts = np.arange(0, 4.1e-3, 1 / fs)
        sgn = np.sign(np.cos(2 * np.pi * f * ts[1] if len(ts) > 1 else 1))
        # the alias has the same samples: cos(2π f n/fs) = cos(2π fa n/fs) (phase sign may flip)
        pt = self.plot("time")
        pt.line("x", tc * 1e3, g * np.cos(2 * np.pi * f * tc), color=GRAY, width=1.0,
                name=f"input {f / 1e3:.2f} kHz")
        pt.line("a", tc * 1e3, g * np.cos(2 * np.pi * fa * tc), color=NAVY, width=2.2, style="--",
                name=f"what the samples say: {fa / 1e3:.2f} kHz")
        pt.scatter("s", ts * 1e3, g * np.cos(2 * np.pi * f * ts), color=RED, size=9,
                   name="samples")
        fin = np.linspace(0, 24000, 2401)
        pf = self.plot("fold")
        pf.set_ylim(0, fs / 2e3 * 1.08)
        for k in range(1, 7):
            pf.band(f"z{k}", (k - 1) * fs / 2e3, k * fs / 2e3, color=GRAY if k % 2 == 0 else GREEN,
                    alpha=0.05)
        pf.line("fold", fin / 1e3, pcm.alias_frequency(fin, fs) / 1e3, color=NAVY, width=2.0)
        pf.scatter("now", [f / 1e3], [fa / 1e3], color=RED, size=14, symbol="d")
        ps = self.plot("spec")
        lv = 20 * np.log10(g + 1e-12)
        xs, ys = [], []
        for k in range(0, 4):
            for s_ in (-1, 1):
                fr = k * fs + s_ * f
                if 0 <= fr <= 24000:
                    xs.append(fr / 1e3)
                    ys.append(lv)
        ps.stems("img", xs, ys, color=GRAY, base=-90, size=7)
        ps.stems("orig", [f / 1e3], [lv], color=RED, base=-90, size=10)
        ps.stems("bb", [fa / 1e3], [lv], color=NAVY, base=-90, size=10)
        ps.band("nyq", 0, fs / 2e3, color=GREEN, alpha=0.10)
        ps.text("nyql", 0.3, 6, "← 0 … fs/2: what comes back out", color=GREEN, size=8.5,
                anchor=(0, 0))
        if p.aa:
            ff = np.linspace(0, 24000, 600)
            ps.line("aa", ff / 1e3, butter_mag_db(ff, 3400, p.order), color=ORANGE, width=1.6,
                    style="--")
        ok = "faithful" if f <= fs / 2 else ("aliased, but filtered" if att >= 40 else "ALIASED")
        self.readout(app=fa, zone=zone, att=att, ok=ok)

    def on_listen(self, p):
        fa = float(pcm.alias_frequency(p.f, p.fs))
        att = -float(butter_mag_db(p.f, 3400, p.order)) if p.aa else 0.0
        t = np.arange(int(1.2 * 16000)) / 16000
        self.play_audio(0.5 * 10 ** (-att / 20) * np.sin(2 * np.pi * fa * t), 16000,
                        f"a {fa:.0f} Hz tone")

    def story(self, p):
        fa = self.r.get("app", 0)
        s = (f"<p>The red dots are what the sampler keeps: {v(p.fs, '.0f', 'samples/s')}. Any tone at "
             f"f, fs − f, fs + f, 2fs − f … passes through exactly the same dots, so the "
             f"reconstruction can only give back the slowest one, in 0 … fs/2 "
             f"(= {v(p.fs / 2, '.0f', 'Hz')}).</p>")
        if p.f > p.fs / 2:
            s += (f"<p>{bad('Aliasing.')} Your {v(p.f, '.0f', 'Hz')} tone comes back as "
                  f"{v(fa, '.0f', 'Hz')}, folded about a multiple of fs/2 (left: the folding "
                  "triangle). Nothing after the sampler can tell, so the anti-alias filter must "
                  "come <i>before</i> it.</p>")
        else:
            s += f"<p>{good('Below fs/2')}: the samples describe only your tone.</p>"
        s += ("<p>Telephony samples at 8 kHz after filtering to about 3.4 kHz; the gap up to 4 kHz is "
              "the room a real filter needs to roll off.</p>")
        return "<h3>Two tones, one set of samples</h3>" + s + keybox(
            "Apparent frequency = |f − fs·round(f/fs)|. Filter first; sample second.")


# =============================================================================== 2. reconstruction
class Reconstruction(Experiment):
    title = "Reconstruction and ZOH droop"
    blurb = "A DAC holds each sample: staircase, sinc droop and images at every multiple of fs."
    book = "sec:ch05:recon"
    controls = [
        Slider("f", "Tone", 100, 3900, 3000, step=50, unit="Hz", help="fs = 8 kHz"),
        Choice("m", "Reconstruction", ["Zero-order hold (DAC)", "Linear interpolation", "Ideal sinc"],
               style="menu"),
        Choice("L", "Digital oversampling before the DAC", ["×1", "×2", "×4", "×8"]),
        Toggle("comp", "Inverse-sinc pre-compensation", False,
               enabled_if=lambda p: not p.m.startswith("Ideal")),
    ]
    plots = [
        Plot("time", "Samples and the reconstructed waveform", x="time (ms)", y="amplitude",
             xlim=(0, 2), ylim=(-1.6, 2.0), legend="tl", legend_cols=3),
        SpectrumPlot("spec", "Spectrum of the analog output: the tone, its images and the sinc envelope",
                     x="frequency (kHz)", y="level (dB re ideal tone)", xlim=(0, 40), ylim=(-70, 5),
                     legend="tr"),
    ]
    layout = [["time"], ["spec"]]
    readouts = [
        Readout("droop", "Droop at the tone", "dB", ".2f", good=lambda x: x > -0.1),
        Readout("img", "Strongest image", "dB re tone", ".1f", good=lambda x: x < -30),
        Readout("fimg", "…at", "kHz", ".2f"),
        Readout("err", "Waveform error", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Undo the droop: a 3 kHz tone through a ZOH DAC within 0.1 dB of its true level, "
                  "without oversampling.",
                  lambda s: s.p.m.startswith("Zero") and abs(s.p.f - 3000) < 1 and s.p.L == "×1"
                  and s.r.droop > -0.1),
        Challenge("Push the images away: ZOH, 3 kHz tone, strongest image below −25 dB.",
                  lambda s: s.p.m.startswith("Zero") and abs(s.p.f - 3000) < 1 and s.r.img < -25),
        Challenge("Smarter beats faster: images of a 3 kHz tone below −24 dB with no more than ×2 "
                  "oversampling, using a real (non-ideal) reconstruction.",
                  lambda s: not s.p.m.startswith("Ideal") and s.p.L in ("×1", "×2") and abs(s.p.f - 3000) < 1
                  and s.r.img < -24),
    ]
    fs, M, up = 8000.0, 256, 32

    def recon(self, p):
        fs, M = self.fs, self.M
        k = int(round(p.f / fs * M))
        f = k * fs / M
        L = int(p.L[1:])
        n = np.arange(M)
        amp = 1.0
        fs2 = fs * L
        if p.comp and not p.m.startswith("Ideal"):
            amp = 1 / np.sinc(f / fs2) if p.m.startswith("Zero") else 1 / np.sinc(f / fs2) ** 2
        x = amp * np.cos(2 * np.pi * f * n / fs)
        # ideal digital interpolation by L (periodic: zero-pad the spectrum)
        if L > 1:
            X = np.fft.rfft(x)
            xL = np.fft.irfft(X, M * L) * L
        else:
            xL = x
        U = self.up // L                                  # fine-grid points per output sample
        if p.m.startswith("Zero"):
            y = np.repeat(xL, U)
        elif p.m.startswith("Linear"):
            idx = np.arange(M * L * U) / U
            y = np.interp(idx, np.arange(M * L + 1), np.r_[xL, xL[0]])
        else:
            y = np.fft.irfft(np.fft.rfft(x), M * self.up) * self.up
        tf = np.arange(len(y)) / (fs * self.up)
        return f, x, y, tf, n / fs

    def update(self, p):
        f, x, y, tf, ts = self.recon(p)
        Y = np.abs(np.fft.rfft(y)) / (len(y) / 2)
        fr = np.fft.rfftfreq(len(y), 1 / (self.fs * self.up))
        k0 = int(round(f / self.fs * self.M))
        tone = Y[k0]
        droop = 20 * np.log10(tone + 1e-12)
        imgs = Y.copy()
        imgs[k0] = 0
        imgs[0] = 0
        ki = int(np.argmax(imgs))
        img = 20 * np.log10(imgs[ki] / max(tone, 1e-12) + 1e-12)
        # waveform error vs the true sine (delay-compensated by least squares on cos/sin)
        B = np.stack([np.cos(2 * np.pi * f * tf), np.sin(2 * np.pi * f * tf)], 1)
        c, *_ = np.linalg.lstsq(B, y, rcond=None)
        err = 10 * np.log10(np.mean((y - B @ c) ** 2) / np.mean((B @ c) ** 2) + 1e-20)
        pt = self.plot("time")
        k = tf <= 2.05e-3
        pt.line("true", tf[k] * 1e3, np.cos(2 * np.pi * f * tf[k]), color=GRAY, width=3, alpha=0.5,
                name="original")
        pt.line("y", tf[k] * 1e3, y[k], color=ORANGE if p.m.startswith("Zero") else NAVY, width=2.0,
                name="DAC output")
        ks = ts <= 2.05e-3
        pt.scatter("s", ts[ks] * 1e3, x[ks], color=RED, size=9, name="samples")
        Pdb = 20 * np.log10(Y + 1e-9)
        ps = self.plot("spec")
        kk = fr <= 40.5e3
        ps.stems("lines", fr[kk][Pdb[kk] > -70] / 1e3, Pdb[kk][Pdb[kk] > -70], color=NAVY,
                 base=-70, size=6, name="output lines")
        L = int(p.L[1:])
        ff = np.linspace(1, 40000, 800)
        env = np.sinc(ff / (self.fs * L))
        if p.m.startswith("Linear"):
            env = env ** 2
        if not p.m.startswith("Ideal"):
            ps.line("env", ff / 1e3, 20 * np.log10(np.abs(env) + 1e-9), color=ORANGE, width=1.4,
                    style="--", name="sinc envelope of the hold")
        ps.band("band", 0, self.fs / 2e3, color=GREEN, alpha=0.08)
        for m in range(1, 6):
            ps.vline(f"fs{m}", m * self.fs / 1e3, color=GRAY, style=":", width=0.8)
        if p.m.startswith("Ideal"):
            self.readout(droop=droop, img="none", fimg="—", err=max(err, -99.0))
        else:
            self.readout(droop=droop, img=img, fimg=fr[ki] / 1e3, err=err)

    def story(self, p):
        s = ("<p>Ideal reconstruction draws a sinc pulse through every sample. A real DAC just "
             "<b>holds</b> each value for one sample period: the staircase. Holding is a filter "
             "whose response is sinc(f/fs): it droops "
             f"{v(-self.r.get('droop', 0), '.2f', 'dB')} at your tone (3.9 dB at fs/2) and lets "
             "through images at fs ± f, 2fs ± f, … (stems).</p>")
        if p.comp:
            s += ("<p>The inverse-sinc pre-compensation boosts the samples by 1/sinc(f/fs) before the "
                  "DAC, cancelling the droop exactly; it does nothing for the images.</p>")
        if p.L != "×1":
            s += (f"<p>Oversampling {v(p.L)} interpolates digitally first, so the DAC's own images move "
                  "up to L·fs where a cheap analog filter removes them, and the droop flattens.</p>")
        return "<h3>Holding is filtering</h3>" + s + keybox(
            "ZOH response = sinc(f/fs): −3.92 dB at fs/2. Real DACs add an inverse-sinc FIR and "
            "oversample, then a gentle smoothing filter.")


# =============================================================================== 3. quantizer
class Quantizer(Experiment):
    title = "Quantization: bits and SQNR"
    blurb = "Six decibels per bit, one decibel per decibel of level, and the cliff of clipping."
    book = "sec:ch05:quant"
    controls = [
        IntSlider("bits", "Bits", 1, 16, 4),
        Slider("lvl", "Signal level (rms re full scale)", -70, 6, -3, step=0.5, unit="dBFS",
               help="0 dBFS here means rms = full scale / √2 (a full-scale sine)"),
        Choice("sig", "Signal", ["Sine", "Gaussian", "Speech-like"]),
    ]
    plots = [
        Plot("wave", "Input and quantized output (zoom)", x="time (samples)", y="amplitude (full scale = 1)",
             xlim=(0, 120), ylim=(-1.3, 1.6), legend="tl", legend_cols=2),
        Plot("hist", "Quantization error", x="error (LSB)", y="share of samples (%)", xlim=(-1.5, 1.5),
             legend=None),
        Plot("curve", "SQNR vs signal level", x="signal level (dBFS)", y="SQNR (dB)", xlim=(-70, 6),
             ylim=(-10, 105), legend="tl"),
    ]
    layout = [["wave", "wave"], ["hist", "curve"]]
    readouts = [
        Readout("sqnr", "SQNR", "dB", ".1f"),
        Readout("th", "6.02b + 1.76 + level", "dB", ".1f"),
        Readout("enob", "Effective bits", "", ".2f"),
        Readout("clip", "Samples clipped", "%", ".2f"),
    ]
    challenges = [
        Challenge("Optimal loading: 8-bit Gaussian signal, find the level that maximises SQNR (within "
                  "0.5 dB of the best).",
                  lambda s: s.p.sig == "Gaussian" and s.p.bits == 8 and s.r.sqnr >= s.r.best - 0.5,
                  hint="Too loud: overload noise from clipping. Too quiet: granular noise. The curve has a peak."),
        Challenge("A quiet talker at −30 dBFS still gets 40 dB SQNR (sine): use as few bits as possible "
                  "(12 or fewer).",
                  lambda s: s.p.sig == "Sine" and s.p.lvl <= -30 and s.r.sqnr >= 40 and s.p.bits <= 12),
        Challenge("Overload: clip more than 2 % of a Gaussian signal's samples.",
                  lambda s: s.p.sig == "Gaussian" and s.r.clip > 2),
    ]
    N = 1 << 14

    def signal(self, kind, lvl):
        n = np.arange(self.N)
        rms = 10 ** (lvl / 20) / np.sqrt(2)
        if kind == "Sine":
            return np.sqrt(2) * rms * np.sin(2 * np.pi * 0.0123456 * n + 0.3)
        if kind == "Gaussian":
            return rms * np.random.default_rng(3).standard_normal(self.N)
        return rms * speech(8000.0, self.N / 8000.0)[:self.N]

    def sqnr(self, kind, lvl, bits):
        x = self.signal(kind, lvl)
        return pcm.sqnr_db(x, pcm.quantize(x, bits))

    def update(self, p):
        x = self.signal(p.sig, p.lvl)
        xq = pcm.quantize(x, p.bits)
        s = pcm.sqnr_db(x, xq)
        clip = 100 * np.mean(np.abs(x) > 1)
        D = 2.0 / 2 ** p.bits
        pw = self.plot("wave")
        k = np.arange(130)
        pw.line("x", k, x[k], color=GRAY, width=2.4, alpha=0.6, name="input")
        pw.line("q", k, xq[k], color=NAVY, width=1.8, name=f"{p.bits}-bit output")
        pw.hband("fs", -1, 1, color=GREEN, alpha=0.04)
        if p.bits <= 6:
            for j in range(2 ** p.bits):
                pw.hline(f"lv{j}", -1 + D / 2 + j * D, color=GRAY, style=":", width=0.6)
        for j in range(2 ** p.bits if p.bits <= 6 else 0, 64):
            pass
        e = (xq - x) / D
        h, edges = np.histogram(np.clip(e, -1.49, 1.49), bins=60, range=(-1.5, 1.5))
        ph = self.plot("hist")
        ph.line("h", edges, 100 * h / len(e), color=NAVY, width=1.4, step=True, fill=0, fill_alpha=0.25)
        ph.band("u", -0.5, 0.5, color=GREEN, alpha=0.08)
        ph.set_ylim(0, max(100 * h.max() / len(e) * 1.2, 2))
        lv = np.arange(-70, 6.5, 2.0)
        cur = [self.sqnr(p.sig, L_, p.bits) for L_ in lv]
        pc = self.plot("curve")
        pc.line("th", lv, 6.02 * p.bits + 1.76 + lv, color=GRAY, width=1.2, style="--",
                name="6.02b + 1.76 + level")
        pc.line("c", lv, cur, color=NAVY, width=2.4, name=f"{p.bits} bits, {p.sig.lower()}")
        pc.scatter("now", [p.lvl], [s], color=ORANGE, size=14, symbol="d")
        best = max(cur + [s])
        if p.sig == "Gaussian":
            fine = np.arange(-30, 3, 0.5)
            best = max(self.sqnr(p.sig, L_, p.bits) for L_ in fine)
        self.readout(sqnr=s, th=6.02 * p.bits + 1.76 + p.lvl, enob=(s - 1.76) / 6.02, clip=clip,
                     best=best)

    def story(self, p):
        s_ = self.r.get("sqnr", 0)
        s = (f"<p>A {v(p.bits, 'd')}-bit quantizer has 2^{p.bits} = {v(2 ** p.bits, 'd')} levels; each "
             f"sample is rounded to the nearest. If the signal is busy the error is uniform within ±½ "
             f"step (bottom left) with power Δ²/12, which gives <b>6.02 dB per bit</b>: SQNR = "
             f"6.02b + 1.76 dB for a full-scale sine.</p>"
             f"<p>The noise is fixed but the signal is not: every dB quieter costs a dB of SQNR (the "
             f"45° line). You have {v(s_, '.1f', 'dB')}.</p>")
        if self.r.get("clip", 0) > 0.5:
            s += ("<p>" + bad("Clipping.") + " Peaks beyond full scale are flattened: overload noise "
                  "that no extra bit can fix.</p>")
        if p.sig != "Sine":
            s += ("<p>Noise-like signals have peaks: you must back off by a <b>loading factor</b> "
                  "(about 4σ), costing roughly 7 dB against a sine.</p>")
        return "<h3>Six decibels per bit</h3>" + s + keybox(
            "SQNR falls 1 dB per dB below full scale. A quiet telephone talker would need 13 uniform "
            "bits: the reason for companding (Experiment 5).")


# =============================================================================== 4. dither
class Dither(Experiment):
    title = "Dither"
    blurb = "Add noise on purpose, and the distortion disappears."
    book = "sec:ch05:dither"
    controls = [
        Slider("amp", "Sine amplitude", 0.1, 4.0, 0.9, step=0.05, unit="LSB"),
        Choice("d", "Dither", ["None", "RPDF", "TPDF", "Subtractive"],
               help="RPDF: uniform ±½ LSB. TPDF: triangular ±1 LSB. Subtractive: RPDF added before "
                    "and removed after the quantizer"),
        Button("listen", "▶  Listen to a fading tone", primary=True),
    ]
    plots = [
        SpectrumPlot("spec", "Output spectrum: harmonics or a noise floor?", x="frequency (× fs)",
                     y="level (dB re tone)", xlim=(0, 0.25), ylim=(-110, 5), legend=None),
        Plot("time", "Input and quantized output", x="time (samples)", y="amplitude (LSB)",
             xlim=(0, 120), ylim=(-4.8, 5.8), legend="tl", legend_cols=2),
        Plot("stats", "Error statistics vs a DC input", x="input (LSB)", y="error (LSB / LSB²)",
             xlim=(-1.5, 1.5), ylim=(-0.6, 0.6), legend="tl", legend_cols=2),
    ]
    layout = [["spec", "spec"], ["time", "stats"]]
    readouts = [
        Readout("sfdr", "Spurious-free dynamic range", "dB", ".0f", good=lambda x: x >= 60,
                help="Tone vs the largest spur of the average output (noise excluded); capped at 120"),
        Readout("noise", "Total error power", "LSB²", ".3f"),
        Readout("thd", "Harmonic distortion", "dB", ".0f"),
        Readout("out", "Output", "", None),
    ]
    challenges = [
        Challenge("Clean up a 0.9 LSB tone: no harmonics (SFDR ≥ 60 dB) and no noise modulation either.",
                  lambda s: s.p.d == "TPDF" and abs(s.p.amp - 0.9) < 0.03 and s.r.sfdr >= 60,
                  hint="Look at the variance curve, bottom right: which dither keeps it flat?"),
        Challenge("Vanish: with no dither, find an amplitude where the quantizer outputs nothing at all.",
                  lambda s: s.p.d == "None" and s.r.out == "silent"),
        Challenge("The best of both: harmonic-free (SFDR ≥ 60 dB) with error power below 0.1 LSB².",
                  lambda s: s.r.sfdr >= 60 and s.r.noise < 0.1),
    ]
    N = 1 << 14

    def setup(self):
        # mean and variance of the total error vs a DC input, for each dither type (as in the book)
        r = np.random.default_rng(5)
        xs = np.linspace(-1.5, 1.5, 241)
        M = 4000
        dr = r.random(M) - 0.5
        dt = r.random(M) - r.random(M)
        self.stats = {}
        for name, d, sub in [("None", np.zeros(M), False), ("RPDF", dr, False), ("TPDF", dt, False),
                             ("Subtractive", dr, True)]:
            e = np.round(xs[:, None] + d[None, :]) - (d[None, :] if sub else 0) - xs[:, None]
            self.stats[name] = (xs, e.mean(1), e.var(1))

    def run(self, p, n, amp, rng):
        t = np.arange(n)
        x = amp * np.sin(2 * np.pi * 331 / self.N * t)
        d = pcm.dither("RPDF" if p.d == "Subtractive" else p.d, n, 1.0, rng)
        y = np.round(x + d) - (d if p.d == "Subtractive" else 0)
        return x, y

    def update(self, p):
        rng = np.random.default_rng(1)
        x, y = self.run(p, self.N, p.amp, rng)
        e = y - x
        k0 = 331
        # displayed spectrum: power-averaged over 6 dither realisations (a smooth noise floor)
        w = np.blackman(self.N)
        X = np.abs(np.fft.rfft(y * w)) ** 2
        for _ in range(5):
            X = X + np.abs(np.fft.rfft(self.run(p, self.N, p.amp, rng)[1] * w)) ** 2
        f = np.fft.rfftfreq(self.N)
        silent = np.all(y == 0)
        # distortion = harmonics of the EXPECTED output E[y | x]: round(x) without dither, and
        # exactly x for RPDF, TPDF and subtractive dither (the dither linearises the mean)
        yE = np.round(x) if p.d == "None" else x
        Y = np.abs(np.fft.rfft(yE)) ** 2                 # coherent record: no window needed
        if silent:
            sfdr, thd, Pdb, out = 0.0, 0.0, np.full(len(X), -150.0), "silent"
        else:
            Pdb = 10 * np.log10(X / X[k0 - 2:k0 + 3].max() + 1e-20)
            spur = np.delete(Y, [0, k0]).max()
            sfdr = float(min(10 * np.log10(Y[k0] / max(spur, 1e-30)), 120.0))
            harm = sum(Y[h * k0] for h in range(2, 12) if h * k0 < len(Y))
            thd = float(max(10 * np.log10(harm / Y[k0] + 1e-30), -120.0))
            out = "distorted" if sfdr < 60 else "tone + hiss"
        ps = self.plot("spec")
        ps.line("P", f, np.maximum(Pdb, -110), color=NAVY, width=1.0)
        for h in range(2, 8):
            ps.vline(f"h{h}", h * k0 / self.N, color=RED, style=":", width=0.8,
                     label=f"{h}f" if h <= 5 else None, label_pos=0.95)
        pt = self.plot("time")
        k = np.arange(130)
        pt.line("x", k, x[k], color=GRAY, width=2.4, alpha=0.6, name="input")
        pt.line("y", k, y[k], color=NAVY, width=1.6, step=False, name="output")
        xs, mu, var = self.stats[p.d]
        pst = self.plot("stats")
        pst.line("mu", xs, mu, color=RED, width=2.0, name="mean of error")
        pst.line("var", xs, var, color=NAVY, width=2.0, name="variance")
        pst.hline("12", 1 / 12, color=GREEN, style=":", label="1/12", label_pos=0.03)
        self.readout(sfdr=sfdr, noise=float(np.mean(e ** 2)), thd=thd, out=out)

    def on_listen(self, p):
        fs = 8000
        n = int(3 * fs)
        t = np.arange(n)
        amp = np.linspace(4, 0.2, n)
        x = amp * np.sin(2 * np.pi * 440 / fs * t)
        d = pcm.dither("RPDF" if p.d == "Subtractive" else p.d, n, 1.0, np.random.default_rng())
        y = np.round(x + d) - (d if p.d == "Subtractive" else 0)
        self.play_audio(0.08 * y, fs, "a tone fading into the last bits")

    def story(self, p):
        s = ("<p>For a quiet, regular signal the quantization error is not noise at all: it is a "
             "deterministic function of the signal, so it repeats with it and shows up as "
             "<b>harmonics</b> (red lines): audible distortion, visible contouring.</p>")
        if p.d == "None":
            s += (f"<p>{bad('No dither.')} At {v(p.amp, '.2f', 'LSB')} the output is a crude square-ish "
                  "wave. Below half an LSB it vanishes completely.</p>")
        elif p.d == "RPDF":
            s += ("<p>RPDF dither (uniform, ±½ LSB) makes the error's <i>mean</i> independent of the "
                  "signal: the harmonics go. But its variance still depends on the input (bottom right): "
                  "noise modulation, audible on fades.</p>")
        elif p.d == "TPDF":
            s += ("<p>TPDF dither (triangular, ±1 LSB) makes both the mean and the variance constant: "
                  "a benign, signal-independent hiss at 3× the undithered power (¼ LSB²). Every audio "
                  "ADC and mastering chain uses it.</p>")
        else:
            s += ("<p>Subtractive dither is added before the quantizer and removed afterwards: the "
                  "error becomes exactly uniform, independent and only 1/12 LSB², but the receiver must "
                  "know the dither sequence.</p>")
        return "<h3>Noise as a cure</h3>" + s + keybox(
            "Dither trades a small rise in noise for the elimination of distortion. Press Listen and "
            "compare None with TPDF as the tone fades away.")


# =============================================================================== 5. companding
CODECS = ["8-bit uniform", "μ-law (G.711)", "A-law (G.711)", "13-bit uniform"]


def codec(x, name):
    if name.startswith("μ"):
        return pcm.g711_codec(x, "mu")[0]
    if name.startswith("A"):
        return pcm.g711_codec(x, "A")[0]
    b = 8 if name.startswith("8") else 13
    return pcm.quantize(x, b)


def dbm0_amp(level):
    """Peak amplitude (full scale 1) of a sine at ``level`` dBm0 (G.711: +3.17 dBm0 = full scale)."""
    return 10 ** ((level - 3.17) / 20)


class Companding(Experiment):
    title = "μ-law and A-law (G.711)"
    blurb = "Eight bits that sound like thirteen: logarithmic steps for a telephone voice."
    book = "sec:ch05:g711"
    controls = [
        Choice("c", "Codec", CODECS, default="μ-law (G.711)", style="menu"),
        Slider("lvl", "Talker level", -60, 3, -20, step=0.5, unit="dBm0",
               help="Sine level relative to the telephone reference; +3.17 dBm0 is the codec's full scale"),
        Choice("sig", "Signal", ["1 kHz sine", "Speech-like"]),
        Button("listen", "▶  Listen through the codec", primary=True),
    ]
    plots = [
        Plot("char", "Compressor characteristic: 8 chords of 16 steps", x="input magnitude (full scale = 1)",
             y="code / 128", xlim=(0, 1), ylim=(0, 1.05), legend="br"),
        Plot("sqnr", "SQNR vs talker level", x="level (dBm0)", y="SQNR (dB)", xlim=(-60, 3),
             ylim=(-5, 75), legend="tl"),
        Plot("wave", "A few milliseconds: the signal and 10 × its coding error", x="time (ms)",
             y="amplitude (full scale = 1)", xlim=(0, 4), legend="tl", legend_cols=2),
    ]
    layout = [["sqnr", "char"], ["wave", "wave"]]
    readouts = [
        Readout("sqnr", "SQNR", "dB", ".1f", good=lambda x: x >= 33),
        Readout("code", "Code word of the peak sample", "", None,
                help="sign | chord (3 bits) | step (4 bits), before the G.711 bit inversions"),
        Readout("rate", "Bit rate", "kb/s", ".0f"),
        Readout("range", "Levels with SQNR ≥ 33 dB", "dB wide", ".0f"),
    ]
    challenges = [
        Challenge("Toll quality for a quiet talker: SQNR of at least 33 dB at −30 dBm0 or quieter, with "
                  "only 8 bits.",
                  lambda s: s.p.lvl <= -30 and s.r.sqnr >= 33 and s.r.rate <= 64),
        Challenge("Find where 8-bit uniform PCM drops below 20 dB SQNR (sine, within 1 dB).",
                  lambda s: s.p.c == "8-bit uniform" and s.p.sig == "1 kHz sine" and 19 <= s.r.sqnr < 20.5),
        Challenge("Make the companded codec use its top chord (7): a peak sample in the loudest segment.",
                  lambda s: s.p.c.startswith(("μ", "A")) and s.r.code.split()[1] == "111"),
    ]
    fs, N = 8000.0, 4096

    def setup(self):
        lv = np.arange(-60, 3.5, 1.5)
        n = np.arange(self.N)
        self.curves = {}
        for c in CODECS:
            out = []
            for L in lv:
                x = dbm0_amp(L) * np.sin(2 * np.pi * 1020.3 / self.fs * n + 0.4)
                out.append(pcm.sqnr_db(x, codec(x, c)))
            self.curves[c] = (lv, np.array(out))

    def signal(self, p, n):
        t = np.arange(n) / self.fs
        if p.sig.startswith("1"):
            return dbm0_amp(p.lvl) * np.sin(2 * np.pi * 1020.3 * t + 0.4)
        sp = speech(self.fs, n / self.fs)[:n]
        return np.clip(dbm0_amp(p.lvl) / np.sqrt(2) * sp, -1, 1)   # same rms as the sine

    def update(self, p):
        x = self.signal(p, self.N)
        y = codec(x, p.c)
        s = pcm.sqnr_db(x, y)
        # code word of the largest sample
        i = int(np.argmax(np.abs(x)))
        if p.c.startswith(("μ", "A")):
            _, sg, ch, stp = pcm.g711_codec(np.array([x[i]]), "mu" if p.c.startswith("μ") else "A")
            code = f"{1 if sg[0] > 0 else 0} {int(ch[0]):03b} {int(stp[0]):04b}"
            rate = 64
        else:
            b = 8 if p.c.startswith("8") else 13
            q = int(np.clip(np.floor((x[i] + 1) / (2 / 2 ** b)), 0, 2 ** b - 1))
            code = format(q, f"0{b}b")
            rate = 8 * b
        lv, cur = self.curves[p.c]
        good_lv = lv[cur >= 33]
        rng_ = float(good_lv.max() - good_lv.min() + 1.5) if len(good_lv) else 0.0
        ps = self.plot("sqnr")
        for c, col in zip(CODECS, [GRAY, NAVY, RED, GREEN]):
            lvc, cc = self.curves[c]
            ps.line(c, lvc, cc, color=col, width=2.8 if c == p.c else 1.2, name=c)
        ps.hline("toll", 33, color=ORANGE, style=":", label="toll quality ≈ 33 dB", label_pos=0.7)
        ps.scatter("now", [p.lvl], [s], color=ORANGE, size=14, symbol="d")
        # characteristic
        pc = self.plot("char")
        xm = np.linspace(0, 1, 2000)
        if p.c.startswith("μ"):
            _, _, e, q = pcm.g711_codec(xm, "mu")
            pc.line("smooth", xm, pcm.mulaw(xm), color=GRAY, width=1.2, style="--", name="smooth μ = 255 law")
            ends = [(2 ** k - 1) / 255 for k in range(9)]
        elif p.c.startswith("A"):
            _, _, e, q = pcm.g711_codec(xm, "A")
            pc.line("smooth", xm, pcm.alaw(xm), color=GRAY, width=1.2, style="--", name="smooth A = 87.6 law")
            ends = [0] + [2 ** k / 128 for k in range(1, 8)] + [1]
        else:
            b = 8 if p.c.startswith("8") else 13
            e, q = None, None
            pc.line("smooth", xm, xm, color=GRAY, width=1.2, style="--", name="uniform")
            ends = []
        if e is not None:
            pc.line("g", xm, (16 * e + q) / 128, color=NAVY, width=2.2, name="G.711 segments")
            pc.scatter("ends", ends, np.arange(len(ends)) / 8, color=RED, size=8)
        pc.vline("pk", abs(x[i]), color=ORANGE, style="--", label="your peak", label_pos=0.15)
        # waveform
        t = np.arange(self.N) / self.fs * 1e3
        k = t <= 4.05
        pw = self.plot("wave")
        pk = max(np.max(np.abs(x)), 1e-6)
        pw.set_ylim(-1.3 * pk, 1.75 * pk)
        pw.line("x", t[k], x[k], color=GRAY, width=2.2, name="signal")
        pw.line("e", t[k], 10 * (y - x)[k], color=RED, width=1.4, name="coding error × 10")
        self.readout(sqnr=s, code=code, rate=rate, range=rng_)

    def on_listen(self, p):
        n = int(2.0 * self.fs)
        sp = speech(self.fs, 2.0)
        x = np.clip(dbm0_amp(p.lvl) / np.sqrt(2) * sp[:n], -1, 1)
        y = codec(x, p.c)
        g = 0.4 / max(np.max(np.abs(x)), 1e-6)
        self.play_audio(np.clip(g * y, -1, 1), self.fs, f"{p.c} at {p.lvl:.0f} dBm0 (normalised)")

    def story(self, p):
        rng_ = self.r.get("range", 0)
        s = ("<p>Speech levels vary by 40 dB between talkers and lines, and uniform PCM loses a dB of "
             "SQNR for every dB of level. <b>Companding</b> compresses the signal logarithmically "
             "before a uniform quantizer and expands it after: the step grows with the amplitude, so "
             "the <i>relative</i> error, and hence the SQNR, stays nearly constant.</p>")
        if p.c.startswith(("μ", "A")):
            s += (f"<p>G.711 approximates the curve with 8 straight chords of 16 steps (top right), each "
                  f"chord twice as coarse as the one below. Your codec keeps 33 dB or better over "
                  f"{v(rng_, '.0f', 'dB')} of level, with 8 bits at 64 kb/s. The code word "
                  f"{v(self.r.get('code', ''))} is sign | chord | step.</p>")
        else:
            s += (f"<p>Uniform PCM keeps 33 dB over only {v(rng_, '.0f', 'dB')} of level with "
                  f"{v(self.r.get('rate', 0), '.0f', 'kb/s')}.</p>")
        s += ("<p>μ-law (μ = 255) is used in North America and Japan, A-law (A = 87.6) everywhere else; "
              "international links convert between them.</p>")
        return "<h3>A logarithmic ruler</h3>" + s + keybox(
            "8-bit companded PCM matches 13-bit uniform PCM for quiet talkers. Those 5 saved bits "
            "made 64 kb/s the world's telephone channel.")


# =============================================================================== 6. delta modulation
class DeltaMod(Experiment):
    title = "Delta modulation and CVSD"
    blurb = "One bit per sample: up or down. Too small a step can't keep up; too big chatters."
    book = "sec:ch05:dpcm"
    controls = [
        LogSlider("step", "Step size δ", 0.005, 0.3, 0.08),
        Choice("fs", "Sample (= bit) rate", ["16 kb/s", "32 kb/s", "64 kb/s"], default="32 kb/s"),
        Toggle("cvsd", "Adaptive step (CVSD)", False,
               help="Continuously variable slope: the step grows during runs of identical bits"),
        Slider("f", "Tone frequency", 200, 2000, 800, step=50, unit="Hz"),
        Slider("A", "Tone amplitude", 0.1, 1.0, 0.8, step=0.05),
    ]
    plots = [
        Plot("stair", "Input, staircase and the low-pass-filtered output", x="time (ms)", y="amplitude",
             xlim=(0, 4), ylim=(-1.4, 1.8), legend="tl", legend_cols=3),
        Plot("curve", "SNR vs step size", x="step size δ", y="SNR (dB)", logx=True, xlim=(0.005, 0.3),
             ylim=(-10, 40), legend="tr"),
    ]
    layout = [["stair"], ["curve"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("snr", "SNR", "dB", ".1f"),
        Readout("slope", "Max slope / (δ·fs)", "", ".2f", good=lambda x: x < 1,
                help="Above 1 the staircase cannot keep up: slope overload"),
        Readout("rate", "Bit rate", "kb/s", ".0f"),
        Readout("state", "Regime", "", None),
    ]
    challenges = [
        Challenge("Find the best fixed step at 32 kb/s (SNR within 1 dB of the peak of the curve).",
                  lambda s: not s.p.cvsd and s.p.fs.startswith("32") and s.r.snr >= s.r.best - 1),
        Challenge("CVSD beats the best fixed step by at least 3 dB at 32 kb/s (800 Hz tone, amplitude 0.8).",
                  lambda s: s.p.cvsd and s.p.fs.startswith("32") and abs(s.p.f - 800) < 1
                  and abs(s.p.A - 0.8) < 0.01 and s.r.snr >= s.r.bestfix + 3,
                  hint="Find the best fixed step first (it sets the bar), then switch CVSD on and tune its minimum step."),
        Challenge("Show slope overload: the staircase falls behind (slope ratio above 2).",
                  lambda s: not s.p.cvsd and s.r.slope > 2),
    ]

    def sim(self, p, step, cvsd, fs):
        n = int(0.016 * fs)
        t = np.arange(n) / fs
        x = p.A * np.sin(2 * np.pi * p.f * t) * (0.6 + 0.4 * np.cos(2 * np.pi * 62.5 * t))
        if cvsd:
            _, est, _ = pcm.cvsd(x, step, 20 * step, run=3, beta=0.95, gain=step)
        else:
            _, est = pcm.delta_mod(x, step)
        sos = butter(4, 3400, fs=fs, output="sos")
        rec = sosfiltfilt(sos, est)
        a = n // 8
        snr = 10 * np.log10(np.mean(x[a:] ** 2) / np.mean((rec - x)[a:] ** 2))
        return t, x, est, rec, snr

    def update(self, p):
        fs = float(p.fs.split()[0]) * 1e3
        t, x, est, rec, snr = self.sim(p, p.step, p.cvsd, fs)
        steps = np.logspace(np.log10(0.005), np.log10(0.3), 26)
        key = (fs, p.f, p.A)
        if getattr(self, "_sweep_key", None) != key:          # the curve does not depend on δ
            self._sweep_key = key
            self._fixed = np.array([self.sim(p, s_, False, fs)[4] for s_ in steps])
            self._cv = None
        fixed = self._fixed
        ps = self.plot("stair")
        k = t <= 4.05e-3
        ps.line("x", t[k] * 1e3, x[k], color=GRAY, width=3, alpha=0.5, name="input")
        ps.line("e", np.r_[t[k], t[k][-1] + 1 / fs] * 1e3, est[k], color=RED, width=1.2, step=True,
                name="staircase")
        ps.line("r", t[k] * 1e3, rec[k], color=NAVY, width=2.0, name="after 3.4 kHz filter")
        pc = self.plot("curve")
        pc.line("fix", steps, fixed, color=NAVY, width=2.2, name="fixed step")
        if p.cvsd:
            if self._cv is None:
                self._cv = np.array([self.sim(p, s_, True, fs)[4] for s_ in steps])
            cv = self._cv
            pc.line("cv", steps, cv, color=GREEN, width=2.2, name="CVSD (minimum step)")
        pc.scatter("now", [p.step], [snr], color=ORANGE, size=14, symbol="d")
        slope = np.max(np.abs(np.diff(x))) * fs / (p.step * fs)
        state = "slope overload" if slope > 1.2 and not p.cvsd else (
            "granular noise" if slope < 0.5 and not p.cvsd else "balanced")
        if p.cvsd:
            state = "adaptive"
        self.readout(snr=snr, slope=slope, rate=fs / 1e3, state=state, best=float(fixed.max()),
                     bestfix=float(fixed.max()))

    def story(self, p):
        st_ = self.r.get("state", "")
        s = ("<p>Delta modulation sends <b>one bit per sample</b>: is the input above or below the "
             "running staircase? The staircase moves by a fixed step δ each sample, so it can climb at "
             "most δ·fs per second.</p>")
        if st_ == "slope overload":
            s += (f"<p>{bad('Slope overload.')} The signal climbs faster than δ·fs: the staircase falls "
                  f"behind on every steep slope (ratio {v(self.r.get('slope', 0), '.2f')}).</p>")
        elif st_ == "granular noise":
            s += (f"<p>{bad('Granular noise.')} The step is so large that the staircase chatters "
                  "around flat parts of the waveform.</p>")
        elif st_ == "adaptive":
            s += ("<p><b>CVSD</b> watches for runs of identical bits (a sign of overload) and grows the "
                  "step; it shrinks it again when the bits alternate. Bluetooth's original voice link "
                  "and many military radios use it at 16–64 kb/s.</p>")
        else:
            s += "<p>Balanced: the step just keeps up with the steepest slope.</p>"
        return "<h3>Up or down?</h3>" + s + keybox(
            "Fixed-step DM must trade slope overload against granular noise; adapting the step "
            "removes most of the trade.")


# =============================================================================== 7. sigma-delta
class SigmaDelta(Experiment):
    title = "Sigma-delta noise shaping"
    blurb = "A 1-bit comparator, a loop filter and oversampling: many bits out of one."
    book = "sec:ch05:sigmadelta"
    controls = [
        IntSlider("L", "Loop order", 1, 5, 2),
        Choice("osr", "Oversampling ratio", ["8", "16", "32", "64", "128", "256"], default="64",
               style="menu"),
        Slider("amp", "Input amplitude", -60, 0, -6, step=0.5, unit="dBFS"),
    ]
    plots = [
        SpectrumPlot("spec", "Output spectrum (log frequency) and the noise-transfer function",
                     x="frequency (× fs)", y="level (dB)", logx=True, xlim=(1e-4, 0.5), ylim=(-170, 10),
                     legend="tl", legend_cols=2),
        Plot("bits", "The bitstream and what it means", x="time (samples)", y="amplitude",
             xlim=(0, 200), ylim=(-1.4, 1.9), legend="tl", legend_cols=3),
        Plot("curve", "In-band SQNR vs oversampling", x="oversampling ratio", y="SQNR (dB)", logx=True,
             xlim=(8, 256), ylim=(0, 160), legend="br"),
    ]
    layout = [["spec", "spec"], ["bits", "curve"]]
    readouts = [
        Readout("sqnr", "In-band SQNR", "dB", ".1f"),
        Readout("enob", "Effective bits", "", ".1f", good=lambda x: x >= 14),
        Readout("th", "Linear-model prediction", "dB", ".1f"),
        Readout("stab", "Loop", "", None),
    ]
    challenges = [
        Challenge("Fourteen bits from one: at least 14 effective bits with a second-order loop.",
                  lambda s: s.p.L == 2 and s.r.enob >= 14),
        Challenge("The same 14 bits at an oversampling ratio of 64 or less.",
                  lambda s: int(s.p.osr) <= 64 and s.r.enob >= 14,
                  hint="Each order adds 6 dB per octave of OSR."),
        Challenge("Overload a high-order loop: make a 4th- or 5th-order modulator go unstable.",
                  lambda s: s.p.L >= 4 and s.r.stab == "UNSTABLE"),
    ]
    N, k0 = 8192, 5

    def update(self, p):
        osr = int(p.osr)
        n = np.arange(self.N)
        A = 10 ** (p.amp / 20)
        x = A * np.sin(2 * np.pi * self.k0 / self.N * n)
        b, a = pcm.ntf_coeffs(p.L, None if p.L <= 2 else 1.5)
        vv = pcm.sigma_delta(x, b, a)
        unstable = bool(np.any(np.isnan(vv)))
        vz = np.nan_to_num(vv)
        w = np.hanning(self.N)
        V = np.abs(np.fft.rfft(vz * w)) ** 2
        f = np.fft.rfftfreq(self.N)
        nb = self.N // (2 * osr)
        sigb = np.zeros(len(V), bool)
        sigb[self.k0 - 2:self.k0 + 3] = True
        inb = np.zeros(len(V), bool)
        inb[2:nb + 1] = True
        sq = 10 * np.log10(V[sigb].sum() / max(V[inb & ~sigb].sum(), 1e-30))
        ref = V[sigb].max()
        ps = self.plot("spec")
        ps.line("V", f[1:], 10 * np.log10(V[1:] / ref + 1e-20), color=NAVY, width=0.8, name="1-bit output")
        from scipy.signal import freqz
        wq, h = freqz(b, a, worN=f[1:] * 2 * np.pi)
        ntf = 20 * np.log10(np.abs(h) + 1e-12)
        ps.line("ntf", f[1:], ntf - 60, color=ORANGE, width=1.8, style="--", name="|NTF|² (shifted)")
        ps.band("sb", 1e-4, 0.5 / osr, color=GREEN, alpha=0.10)
        ps.text("sbl", 1.1e-4, 8, "signal band", color=GREEN, size=8.5, anchor=(0, 0))
        pb = self.plot("bits")
        k = np.arange(210)
        # what the bits mean: a moving average over one signal-band period
        avg = np.convolve(vz, np.ones(2 * osr) / (2 * osr), mode="same")
        pb.line("v", k, vz[k] * 0.9, color=GRAY, width=1.0, step=False, name="bits (±1)")
        pb.line("x", k, x[k], color=RED, width=2.4, name="input")
        pb.line("a", k, avg[k], color=NAVY, width=2.0, name="moving average")
        pc = self.plot("curve")
        os_ = np.array([8, 16, 32, 64, 128, 256])
        for L_, col in zip(range(1, 6), [ORANGE, NAVY, GREEN, PURPLE, RED]):
            pc.line(f"t{L_}", os_, pcm.sd_theory_sqnr(L_, os_, A), color=col,
                    width=2.4 if L_ == p.L else 1.0, name=f"order {L_} ({6 * L_ + 3} dB/oct)")
        pc.hline("cd", 98, color=GRAY, style=":", label="16 bits", label_pos=0.85)
        pc.scatter("now", [osr], [max(sq, 0.5)], color=ORANGE, size=14, symbol="d")
        th = float(pcm.sd_theory_sqnr(p.L, osr, A))
        self.readout(sqnr=sq, enob=(sq - 1.76) / 6.02, th=th,
                     stab="UNSTABLE" if unstable else "stable")

    def story(self, p):
        s = ("<p>A one-bit quantizer is terrible: its error is about as big as the signal. Put it inside "
             "a feedback loop with an integrator, though, and the output's <i>average</i> follows the "
             "input (navy curve over the bits), while the quantization noise is pushed toward high "
             f"frequencies by the noise-transfer function (1 − z⁻¹)^L (orange). With L = {v(p.L, 'd')} it "
             f"rises {v(20 * p.L, 'd', 'dB')} per decade.</p>"
             f"<p>A digital decimation filter then keeps only the green signal band, 1/{p.osr} of the "
             f"Nyquist range, and with it {v(self.r.get('sqnr', 0), '.1f', 'dB')} of SQNR: "
             f"{v(self.r.get('enob', 0), '.1f')} effective bits.</p>")
        if self.r.get("stab") == "UNSTABLE":
            s += ("<p>" + bad("Unstable.") + " Orders above 2 can overload: a large input drives the "
                  "integrators into runaway. Real designs keep |NTF| ≤ 1.5 (Lee's rule) and limit the "
                  "input to well below full scale.</p>")
        return "<h3>Pushing the noise out of the band</h3>" + s + keybox(
            "SQNR grows (6L + 3) dB per doubling of OSR. This is how a 1-bit comparator delivers the "
            "24-bit audio ADC in your phone.")


# =============================================================================== 8. jitter
class Jitter(Experiment):
    title = "Clock jitter limits SNR"
    blurb = "A wobbling sample clock turns a steep signal into noise, whatever your bits."
    book = "sec:ch05:jitter"
    controls = [
        LogSlider("f", "Input frequency", 1, 10000, 100, unit="MHz"),
        LogSlider("tj", "Clock jitter (rms)", 10, 10000, 200, unit="fs"),
        IntSlider("bits", "ADC resolution", 6, 18, 14, unit="bits"),
    ]
    plots = [
        Plot("lim", "SNR limit vs input frequency", x="input frequency (MHz)", y="SNR (dB)", logx=True,
             xlim=(1, 10000), ylim=(20, 120), legend="bl"),
        Plot("sim", "Simulated: the error made by the late and early samples", x="sample",
             y="error (LSB)", xlim=(0, 300), legend="tl", legend_cols=2),
    ]
    layout = [["lim"], ["sim"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("sj", "Jitter-limited SNR", "dB", ".1f"),
        Readout("sq", "Quantization SNR", "dB", ".1f"),
        Readout("tot", "Total SNR", "dB", ".1f"),
        Readout("enob", "Effective bits", "", ".2f"),
    ]
    challenges = [
        Challenge("Keep a 14-bit ADC within 1 dB of its ideal SNR at 100 MHz input: how good must the "
                  "clock be?",
                  lambda s: s.p.bits == 14 and abs(s.p.f - 100) < 3 and s.r.sq - s.r.tot <= 1),
        Challenge("Direct RF sampling at 3 GHz with at least 10 effective bits.",
                  lambda s: s.p.f >= 2950 and s.r.enob >= 10),
        Challenge("Make jitter, not the bits, the limit at 10 MHz with a 16-bit ADC (total SNR at least "
                  "10 dB below the quantization SNR).",
                  lambda s: s.p.bits == 16 and abs(s.p.f - 10) < 0.4 and s.r.sq - s.r.tot >= 10),
    ]

    def update(self, p):
        f = p.f * 1e6
        tj = p.tj * 1e-15
        sj = -20 * np.log10(2 * np.pi * f * tj)
        sq = 6.02 * p.bits + 1.76
        tot = -10 * np.log10(10 ** (-sj / 10) + 10 ** (-sq / 10))
        fr = np.logspace(0, 4, 200) * 1e6
        pl = self.plot("lim")
        for tt, col in ((1e-12, GRAY), (100e-15, GRAY), (50e-15, GRAY)):
            pl.line(f"r{tt}", fr / 1e6, -20 * np.log10(2 * np.pi * fr * tt), color=col, width=1.0,
                    style=":", name=None)
        pl.line("you", fr / 1e6, -20 * np.log10(2 * np.pi * fr * tj), color=RED, width=2.4,
                name=f"jitter {p.tj:.0f} fs rms")
        pl.hline("q", sq, color=NAVY, style="--", label=f"{p.bits}-bit ideal: {sq:.0f} dB", label_pos=0.6)
        pl.line("tot", fr / 1e6, -10 * np.log10(10 ** (2 * np.log10(2 * np.pi * fr * tj)) + 10 ** (-sq / 10)),
                color=GREEN, width=2.0, name="total")
        pl.scatter("now", [p.f], [tot], color=ORANGE, size=14, symbol="d")
        pl.text("r1", 1.2, -20 * np.log10(2 * np.pi * 1.2e6 * 1e-12) - 1, "1 ps", color=GRAY, size=8,
                anchor=(0, 0))
        # time-domain simulation: sample a sine at jittered instants (normalised frequency)
        n = 300
        rng = np.random.default_rng(2)
        fs = 4.1 * f                                        # some sample rate above 2f
        ts = np.arange(n) / fs
        dt = tj * rng.standard_normal(n)
        lsb = 2 / 2 ** p.bits
        xj = np.sin(2 * np.pi * f * (ts + dt))
        x0 = np.sin(2 * np.pi * f * ts)
        e_j = (xj - x0) / lsb
        e_q = (pcm.quantize(xj, p.bits) - xj) / lsb
        ps = self.plot("sim")
        ps.line("q", np.arange(n), e_q, color=NAVY, width=1.0, name="quantization error")
        ps.line("j", np.arange(n), e_j, color=RED, width=1.4, name="jitter error")
        m = max(1.0, float(np.max(np.abs(e_j))) * 1.15)
        ps.set_ylim(-m, 1.4 * m)
        self.readout(sj=sj, sq=sq, tot=tot, enob=(tot - 1.76) / 6.02)

    def story(self, p):
        sj = self.r.get("sj", 0)
        sq = self.r.get("sq", 0)
        s = (f"<p>A sample taken τ seconds late is wrong by τ times the signal's slope, and a sine's "
             f"slope grows with its frequency. With {v(p.tj, '.0f', 'fs')} of rms clock jitter at "
             f"{v(p.f, '.0f', 'MHz')} the error limits SNR to −20·log₁₀(2π f σ_t) = "
             f"{v(sj, '.1f', 'dB')}, whatever the resolution.</p>")
        if sj < sq - 6:
            s += (f"<p>{bad('Jitter-limited.')} Your {v(p.bits, 'd')} bits are wasted: the red error "
                  "dwarfs the quantization error. Buy a better clock, not a better ADC.</p>")
        else:
            s += f"<p>{good('Bits-limited')}: the clock is good enough for this resolution.</p>"
        s += ("<p>Note what does <i>not</i> appear: the sample rate. Undersampling a 1 GHz signal at "
              "100 MS/s is exactly as jitter-sensitive as sampling it at 3 GS/s.</p>")
        return "<h3>The clock is part of the converter</h3>" + s + keybox(
            "SNR_jitter = −20·log₁₀(2π f σ_t): 20 dB per decade of input frequency.")


# =============================================================================== 9. T1 / E1
class Frames(Experiment):
    title = "T1 and E1 frames"
    blurb = "24 or 30 voices, 8000 frames a second: the first digital multiplex."
    book = "sec:ch05:telephony"
    animate = True
    fps = 4
    controls = [
        Choice("sys", "System", ["T1 (North America)", "E1 (Europe)"]),
        IntSlider("ch", "Channel / timeslot", 0, 31, 1,
                  help="T1: channels 1–24. E1: timeslots 0–31 (0 = framing, 16 = signalling)"),
        IntSlider("fr", "Frame", 1, 16, 1, help="Frame number within the (super/multi)frame"),
        Toggle("rob", "Robbed-bit signalling (T1)", True,
               enabled_if=lambda p: p.sys.startswith("T1")),
    ]
    plots = [
        ImagePlot("map", "The superframe: one row per 125 µs frame", x="bit position in the frame",
                  y="frame"),
        Plot("ch", "Your channel: 8-bit μ-law samples, one per frame", x="frame", y="decoded sample",
             xlim=(0.4, 12.6), ylim=(-1.2, 1.6), legend=None),
    ]
    layout = [["map"], ["ch"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("rate", "Line rate", "Mb/s", ".3f"),
        Readout("eff", "Voice payload", "%", ".1f"),
        Readout("clean", "Clean data per channel", "kb/s", ".0f"),
        Readout("what", "Selected bits", "", None),
    ]
    challenges = [
        Challenge("Find a frame in which T1 steals the least significant bit of every channel.",
                  lambda s: s.p.sys.startswith("T1") and s.p.rob and s.p.fr in (6, 12)),
        Challenge("Select the E1 timeslot that carries signalling for all 30 voice channels.",
                  lambda s: s.p.sys.startswith("E1") and s.p.ch == 16),
        Challenge("Get a full 64 kb/s clean channel on T1.",
                  lambda s: s.p.sys.startswith("T1") and s.r.clean == 64,
                  hint="Signalling could also travel out of band (common-channel signalling, ISDN)."),
    ]

    def update(self, p):
        t1 = p.sys.startswith("T1")
        if t1:
            m, fbits = pcm.t1_superframe(p.rob)
            nfr, nbit = 12, 193
            ch = int(np.clip(p.ch, 1, 24))
            c0 = 1 + 8 * (ch - 1)
        else:
            m = pcm.e1_frame(16)
            nfr, nbit = 16, 256
            ch = int(np.clip(p.ch, 0, 31))
            c0 = 8 * ch
        fr = int(np.clip(p.fr, 1, nfr))
        img = m.astype(float).copy()
        img[:, c0:c0 + 8] = np.where(img[:, c0:c0 + 8] == 1, 4, img[:, c0:c0 + 8])
        img[fr - 1, c0:c0 + 8] = 5
        # a tuple (hashable: studio looks colormaps up by name first)
        cmap = ((0.0, "#c0392b"), (0.1999, "#c0392b"), (0.2, "#c9d9ea"), (0.3999, "#c9d9ea"),
                (0.4, "#e67e22"), (0.6999, "#e67e22"), (0.7, "#2e86c1"), (0.8999, "#2e86c1"),
                (0.9, "#f1c40f"), (1.0, "#f1c40f"))
        pm = self.plot("map")
        pm.image("m", img[::-1], x=(0, nbit), y=(0.5, nfr + 0.5), cmap=cmap, levels=(0, 5))
        pm.set_xlim(0, nbit)
        pm.set_ylim(0.5, nfr + 0.5)
        pm.set_yticks([(nfr - i, str(i + 1)) for i in range(nfr)])
        # the channel's samples: a voice-like signal, μ-law coded
        sp = speech(8000.0, 0.01)
        x = 0.5 * sp[:nfr] / np.max(np.abs(sp[:16]))
        y, sg, e, q = pcm.g711_codec(x, "mu")
        codes = [(1 if sg[i] > 0 else 0) * 128 + 16 * int(e[i]) + int(q[i]) for i in range(nfr)]
        yr = y.copy()
        if t1 and p.rob:
            for i in (5, 11):
                cw = codes[i] & ~1                               # LSB robbed (signalling bit = 0)
                s_, ee, qq = (1 if cw & 128 else -1), (cw >> 4) & 7, cw & 15
                yr[i] = pcm.g711_mu_decode(s_, ee, qq) / 8159
        pc = self.plot("ch")
        k = np.arange(1, nfr + 1)
        pc.set_xlim(0.4, nfr + 0.6)
        if not t1 and ch in (0, 16):
            pc.set_title(f"Timeslot {ch}: {'framing and alarms' if ch == 0 else 'signalling (abcd bits)'}"
                         " — no voice here")
            pc.bars("b", k, np.zeros(nfr), width=0.6, color=GRAY)
        else:
            pc.set_title(f"Channel {ch}: one 8-bit μ-law sample per frame — decoded (navy) vs original "
                         f"(green), code words above")
            pc.stems("y", k, yr, color=NAVY, size=9, name="decoded")
            pc.scatter("x", k, x, color=GREEN, size=11, outline=GREEN, name="original")
            for i in range(min(nfr, 16)):
                pc.text(f"c{i}", i + 1, 1.25, format(codes[i], "08b")[:1] + " " + format(codes[i], "08b")[1:4]
                        + " " + format(codes[i], "08b")[4:], color=RED if (t1 and p.rob and i in (5, 11)) else GRAY,
                        size=7.5, anchor=(0.5, 0.5))
            pc.vline("sel", fr, color=ORANGE, style="--", width=1.2)
        if t1:
            rate, eff = 193 * 8000 / 1e6, 100 * 24 * 64 / 1544
            clean = 56 if p.rob else 64
            if fr in (6, 12) and p.rob:
                what = "7 voice + 1 robbed"
            else:
                what = "8 voice bits"
        else:
            rate, eff = 2.048, 100 * 30 * 64 / 2048
            clean = 64 if ch not in (0, 16) else 0
            what = {0: "framing", 16: "signalling"}.get(ch, "8 voice bits")
        self.readout(rate=rate, eff=eff, clean=clean, what=what)

    def tick(self, p):
        nfr = 12 if p.sys.startswith("T1") else 16
        new = p.fr % nfr + 1
        self.set_control("fr", new)
        self.update(st.Params({**p, "fr": new}))

    def story(self, p):
        if p.sys.startswith("T1"):
            s = ("<p>A <b>T1</b> frame carries one 8-bit sample from each of 24 voice channels plus one "
                 "framing bit (red): 193 bits every 125 µs, 193 × 8000 = 1.544 Mb/s. Twelve frames make a "
                 "<b>D4 superframe</b>; the framing bits spell 100011011100 so the receiver can find frame "
                 "6 and frame 12.</p>")
            if p.rob:
                s += ("<p>In those two frames the least significant bit of every channel is "
                      "<b>robbed</b> (orange) for on-hook/off-hook signalling. Voice never notices; data "
                      "does, which is why a T1 channel is only 56 kb/s clean for data.</p>")
        else:
            s = ("<p>An <b>E1</b> frame has 32 timeslots of 8 bits: 256 bits every 125 µs = 2.048 Mb/s. "
                 "TS0 carries framing and alarms (red), TS16 the signalling for all 30 voice channels "
                 "(orange), so every voice channel is a clean 64 kb/s.</p>")
        return "<h3>Time-division multiplexing</h3>" + s + keybox(
            "8000 frames/s × 8 bits = 64 kb/s per voice: the DS0, the atom of the digital telephone "
            "network. Press Play to step through the frames.")


# =============================================================================== the lab
LAB = st.Lab(16, "Sampling, PCM and Companding", chapter=5,
             chapter_title="Sampling, Quantization and Digital Telephony",
             experiments=[Aliasing, Reconstruction, Quantizer, Dither, Companding, DeltaMod,
                          SigmaDelta, Jitter, Frames])

if __name__ == "__main__":
    st.run(LAB)
