"""Lab 14 · Analog Modulation: AM, SSB, FM and FM Stereo   (Chapter 4)

Run it:      python labs/lab14_analog_am_fm.py
Self-test:   python labs/lab14_analog_am_fm.py --selftest

Where radio began, and where every idea in digital communications first appeared: the
complex envelope, coherent versus envelope detection, trading bandwidth for SNR, threshold
effects and pilot-aided carrier recovery. Six experiments, several of which you can *hear*
(the Listen buttons use your speakers if Python can reach them; nothing is downloaded).
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.signal import hilbert, spectrogram
from scipy.special import jv

import commlib as cl
from commlib import analog as an
import studio as st
from studio import (Experiment, Slider, LogSlider, Choice, Toggle, Button, Heading, Plot,
                    SpectrumPlot, ConstellationPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD)
from studio import v, keybox, good, bad


# =============================================================================== messages
@lru_cache(maxsize=8)
def _voice(fs, dur):
    return an.voice_like(fs, dur)


def message(kind, f, fs, dur):
    """Test messages, peak-normalised to 1: a tone, two tones, or a synthetic vowel."""
    t = np.arange(int(round(fs * dur))) / fs
    if kind == "Tone":
        return np.sin(2 * np.pi * f * t)
    if kind == "Two tones":
        m = 0.6 * np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2.71 * f * t + 1.0)
        return m / np.max(np.abs(m))
    if kind == "Triangle sweep":
        return 2 * np.abs(2 * ((t * f) % 1) - 1) - 1
    return _voice(fs, dur).copy()


def snr_tone(y, f0, fs, trim=0.08):
    """SNR (dB) of a tone at f0 in y: least-squares fit of sin and cos, residual = noise."""
    n = len(y)
    a, b = int(trim * n), int((1 - trim) * n)
    t = np.arange(a, b) / fs
    B = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)], 1)
    seg = y[a:b]
    coef, *_ = np.linalg.lstsq(np.c_[B, np.ones(len(t))], seg, rcond=None)
    fit = B @ coef[:2]
    seg = seg - coef[2]                       # DC is not noise
    return 10 * np.log10(np.sum(fit ** 2) / max(np.sum((seg - fit) ** 2), 1e-30))


# =============================================================================== 1. AM
class AMEnvelope(Experiment):
    title = "AM and the envelope detector"
    blurb = "A diode, a capacitor and a resistor: the receiver in every AM radio."
    book = "sec:ch04:envdet"
    controls = [
        Heading("Message"),
        Choice("msg", "Message", ["Tone", "Two tones", "Voice"]),
        Slider("fm", "Tone frequency", 200, 3000, 500, step=10, unit="Hz",
               enabled_if=lambda p: p.msg != "Voice"),
        Heading("Transmitter"),
        Slider("mu", "Modulation index μ", 0.05, 1.5, 0.8, step=0.01,
               help="How deeply the message varies the carrier amplitude. Above 1 = overmodulation"),
        Heading("Receiver"),
        LogSlider("rc", "Detector time constant RC", 5, 5000, 60, unit="µs",
                  help="Too short: carrier ripple. Too long: the capacitor cannot follow the envelope down"),
        Button("listen", "▶  Listen to the detector output", primary=True),
    ]
    plots = [
        Plot("wave", "AM signal and what the envelope detector makes of it", x="time (ms)",
             y="amplitude", xlim=(0, 6), ylim=(-2.7, 3.3), legend="tl", legend_cols=3),
        SpectrumPlot("spec", "Spectrum: carrier and two sidebands", x="frequency (kHz)",
                     xlim=(34, 46), ylim=(-70, 3), legend=None),
        Plot("audio", "Recovered audio (after a 5 kHz audio filter)", x="time (ms)",
             y="amplitude", xlim=(0, 10), ylim=(-1.6, 2.1), legend="tl", legend_cols=2),
    ]
    layout = [["wave", "wave"], ["spec", "audio"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("eff", "Power efficiency", "% in sidebands", ".1f"),
        Readout("sinad", "Audio quality (SINAD)", "dB", ".1f", good=lambda x: x > 30),
        Readout("rcmax", "Largest RC without clipping", "µs", ".0f"),
        Readout("state", "Detector", "", None),
    ]
    challenges = [
        Challenge("Overmodulate: push μ past 1 and watch the envelope fold over (SINAD below 20 dB).",
                  lambda s: s.p.mu > 1.02 and s.r.sinad < 20),
        Challenge("Find the sweet spot: SINAD above 36.5 dB with a 500 Hz tone at μ = 0.8.",
                  lambda s: s.p.msg == "Tone" and abs(s.p.fm - 500) < 1 and abs(s.p.mu - 0.8) < 0.006
                  and s.r.sinad > 36.5,
                  hint="RC must be much longer than a carrier period (25 µs) but shorter than the clipping limit."),
        Challenge("Cause diagonal clipping: make RC so long that the detector cannot follow (μ ≤ 0.9).",
                  lambda s: s.p.mu <= 0.9 and s.r.state == "diagonal clipping"),
    ]
    fs, fc, dur = 400e3, 40e3, 0.05

    def chain(self, p, dur=None):
        dur = dur or self.dur
        m = message(p.msg, p.fm, self.fs, dur)
        s = an.am_modulate(m, p.mu, self.fc, self.fs)
        env = an.envelope_detector(s, self.fs, p.rc * 1e-6)
        audio = an.brickwall(env - np.mean(env), self.fs, 5000.0)
        return m, s, env, audio

    def update(self, p):
        m, s, env, audio = self.chain(p)
        t = np.arange(len(m)) / self.fs * 1e3
        w = t <= 6.2
        pw = self.plot("wave")
        pw.line("s", t[w], s[w], color=GRAY, width=0.6, alpha=0.8, name="AM signal")
        pw.line("ideal", t[w], np.abs(1 + p.mu * m[w]), color=GREEN, width=1.6, style="--",
                name="true envelope |1 + μm|")
        pw.line("env", t[w], env[w], color=RED, width=2.0, name="detector output")
        pw.hline("zero", 0, color=GRAY, style="-", width=0.6)
        ps = self.plot("spec")
        f, P = ps.psd("P", s, self.fs, 8192, scale=1e3, color=NAVY, width=1.6)
        ps.vline("fc", self.fc / 1e3, color=RED, style=":", label="carrier", label_pos=0.95)
        fmax_k = (p.fm if p.msg == "Tone" else 2.71 * p.fm if p.msg == "Two tones" else 4000) / 1e3
        half = max(2.0, 2.5 * fmax_k)
        ps.set_xlim(self.fc / 1e3 - half, self.fc / 1e3 + half)
        sinad, lag = an.align_sinad(audio, m, max_lag=400)
        pa = self.plot("audio")
        ta = t[:len(t) - lag]
        g = np.dot(audio[lag:], m[:len(m) - lag]) / np.dot(m, m)
        k = ta <= 10.2
        pa.line("m", ta[k], m[:len(ta)][k], color=GRAY, width=3.0, alpha=0.6, name="message")
        pa.line("a", ta[k], audio[lag:][k] / (g if abs(g) > 1e-6 else 1), color=NAVY, width=1.6,
                name="recovered (scaled)")
        pm = np.mean(m ** 2)
        eff = 100 * p.mu ** 2 * pm / (1 + p.mu ** 2 * pm)
        fmax = p.fm if p.msg == "Tone" else (2.71 * p.fm if p.msg == "Two tones" else 3000.0)
        rcmax = (np.sqrt(1 - p.mu ** 2) / (2 * np.pi * fmax * p.mu) * 1e6) if p.mu < 1 else 0.0
        if p.mu > 1.0:
            state = "overmodulated"
        elif p.rc > rcmax * 1.15:
            state = "diagonal clipping"
        elif p.rc < 50:
            state = "carrier ripple"
        else:
            state = "following"
        self.readout(eff=eff, sinad=sinad, rcmax=rcmax if p.mu < 1 else "—", state=state)

    def on_listen(self, p):
        m, s, env, audio = self.chain(p, dur=1.6)
        self.play_audio(audio[::10], self.fs / 10, "the detector output")

    def story(self, p):
        st_ = self.r.get("state", "")
        s = (f"<p>The carrier's amplitude follows the message: 1 + μ·m(t) with "
             f"μ = {v(p.mu)}. The receiver's diode charges a capacitor to each carrier peak and "
             f"the resistor lets it leak away between peaks. With RC = {v(p.rc, '.0f', 'µs')} "
             f"against a {v(1e6 / self.fc, '.0f', 'µs')} carrier period, the red trace "
             + ("follows the green envelope." if st_ == "following" else "struggles.") + "</p>")
        if st_ == "overmodulated":
            s += ("<p>" + bad("Overmodulated.") + " Where 1 + μm goes negative the envelope "
                  "folds back up: the detector outputs |1 + μm| and the audio distorts badly. A "
                  "coherent receiver would not care, but nobody builds AM receivers that way.</p>")
        elif st_ == "diagonal clipping":
            s += ("<p>" + bad("Diagonal clipping.") + " The capacitor discharges more slowly "
                  "than the envelope falls, so the output slides down a straight-ish line instead of "
                  "following it. The rule: RC ≤ √(1 − μ²) / (2π f_m μ).</p>")
        elif st_ == "carrier ripple":
            s += ("<p>The time constant is so short that the capacitor droops between carrier "
                  "peaks: a sawtooth of carrier ripple rides on the audio.</p>")
        else:
            s += "<p>" + good("Clean detection.") + " This is why a crystal set needs no battery.</p>"
        s += (f"<p>The price of simplicity: only {v(self.r.get('eff', 0), '.1f', '%')} of the "
              f"transmitted power is in the sidebands, the part that carries the message. The "
              f"rest is the carrier, there only so the envelope never crosses zero.</p>")
        return "<h3>Riding the envelope</h3>" + s + keybox(
            "Envelope detection needs 1/f_c ≪ RC ≪ 1/W and μ ≤ 1. Press Listen and change RC.")


# =============================================================================== 2. DSB / SSB
class DSBandSSB(Experiment):
    title = "DSB-SC and SSB"
    blurb = "Drop the carrier, then drop a sideband. What does the receiver have to get right?"
    book = "sec:ch04:ssb"
    controls = [
        Choice("scheme", "Modulation", ["AM", "DSB-SC", "SSB (upper)", "SSB (lower)"]),
        Choice("msg", "Message", ["Two tones", "Voice"]),
        Heading("Receiver local oscillator"),
        Slider("phase", "Phase error", 0, 180, 0, step=1, unit="°"),
        Slider("df", "Frequency error", -300, 300, 0, step=5, unit="Hz"),
        Button("listen", "▶  Listen to the receiver output", primary=True),
    ]
    plots = [
        SpectrumPlot("spec", "Transmitted spectrum", x="frequency (kHz)", xlim=(35, 45),
                     ylim=(-70, 3), legend=None),
        Plot("out", "Receiver output vs the message", x="time (ms)", y="amplitude",
             xlim=(0, 8), ylim=(-1.7, 2.3), legend="tl", legend_cols=2),
        Plot("lvl", "Output level vs LO phase error", x="phase error (°)", y="output power (dB)",
             xlim=(-4, 184), ylim=(-40, 6), legend="bl"),
    ]
    layout = [["spec", "out"], ["lvl", "lvl"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bw", "Bandwidth", "kHz", ".1f"),
        Readout("eff", "Power in sidebands", "%", ".0f"),
        Readout("lvl", "Output level", "dB", ".1f"),
        Readout("fid", "Waveform match", "%", ".0f", help="Correlation between output and message"),
    ]
    challenges = [
        Challenge("Make the DSB-SC output vanish with the wrong LO phase.",
                  lambda s: s.p.scheme == "DSB-SC" and s.r.lvl < -25),
        Challenge("Show SSB's tolerance: a 90° phase error and the output is still at full level.",
                  lambda s: s.p.scheme.startswith("SSB") and abs(s.p.phase - 90) <= 3
                  and s.r.lvl > -1),
        Challenge("Mistune an SSB receiver by 100 Hz or more with the voice message, then listen.",
                  lambda s: s.p.scheme.startswith("SSB") and abs(s.p.df) >= 100 and s.p.msg == "Voice",
                  hint="Every frequency in the voice moves by the same number of hertz: the "
                       "harmonics are no longer harmonic."),
    ]
    fs, fc, dur = 400e3, 40e3, 0.05

    def chain(self, p, dur=None):
        dur = dur or self.dur
        m = message(p.msg, 700.0, self.fs, dur)
        t = np.arange(len(m)) / self.fs
        if p.scheme == "AM":
            s = (1 + 0.8 * m) * np.cos(2 * np.pi * self.fc * t)
        elif p.scheme == "DSB-SC":
            s = m * np.cos(2 * np.pi * self.fc * t)
        else:
            ma = hilbert(m)
            if "lower" in p.scheme:
                ma = np.conj(ma)
            s = np.real(ma * np.exp(2j * np.pi * self.fc * t))
        lo = 2 * np.cos(2 * np.pi * (self.fc + p.df) * t + np.deg2rad(p.phase))
        y = an.brickwall(s * lo, self.fs, 5000.0)
        if p.scheme == "AM":
            y = (y - np.mean(y)) / 0.8
        return m, s, y, t

    def update(self, p):
        m, s, y, t = self.chain(p)
        ps = self.plot("spec")
        ps.psd("P", s, self.fs, 8192, scale=1e3, color=NAVY, width=1.6)
        ps.vline("fc", self.fc / 1e3, color=RED, style=":", label="carrier frequency", label_pos=0.95)
        k = t * 1e3 <= 8.2
        po = self.plot("out")
        po.line("m", t[k] * 1e3, m[k], color=GRAY, width=3.0, alpha=0.55, name="message")
        po.line("y", t[k] * 1e3, y[k], color=RED, width=1.6, name="receiver output")
        lvl = 10 * np.log10(np.mean(y[2000:-2000] ** 2) / np.mean(m ** 2) + 1e-12)
        c = np.dot(y[2000:-2000], m[2000:-2000]) / np.sqrt(
            np.dot(y[2000:-2000], y[2000:-2000]) * np.dot(m[2000:-2000], m[2000:-2000]) + 1e-30)
        ph = np.linspace(0, 180, 361)
        pl = self.plot("lvl")
        dsb = 20 * np.log10(np.abs(np.cos(np.deg2rad(ph))) + 1e-3)
        pl.line("dsb", ph, dsb, color=NAVY, width=2.0, name="AM / DSB-SC (coherent)")
        pl.line("ssb", ph, np.zeros_like(ph), color=GREEN, width=2.0, name="SSB")
        pl.scatter("now", [p.phase], [max(lvl, -40)], color=RED, size=13, symbol="d",
                   name="your receiver")
        W = 4.0 if p.msg == "Voice" else 1.9
        bw = W if p.scheme.startswith("SSB") else 2 * W
        eff = 100 * (0.64 * np.mean(m ** 2) / (1 + 0.64 * np.mean(m ** 2))) if p.scheme == "AM" else 100
        self.readout(bw=bw, eff=eff, lvl=lvl, fid=100 * abs(c))

    def on_listen(self, p):
        m, s, y, t = self.chain(p, dur=1.6)
        self.play_audio(y[::10], self.fs / 10, "the receiver output")

    def story(self, p):
        if p.scheme == "AM":
            s = ("<p>Plain AM: a carrier line in the middle and two mirror-image sidebands. Both "
                 "sidebands carry the same information, and the carrier carries none.</p>")
        elif p.scheme == "DSB-SC":
            s = ("<p>DSB-SC removes the carrier: every watt goes into the sidebands. But now the "
                 "envelope crosses zero, so the receiver must multiply by a local carrier of exactly "
                 "the right phase. Its output is m(t)·cos φ: at 90° it disappears entirely.</p>")
        else:
            s = ("<p>SSB sends a single sideband: half the bandwidth of AM, all the power in the "
                 "message. A phase error φ turns the output into m·cos φ ± m̂·sin φ, where m̂ is the "
                 "Hilbert transform: a different waveform with the <b>same</b> magnitude spectrum. "
                 "The ear barely notices phase, so SSB voice works with free-running oscillators.</p>")
        if abs(p.df) > 0 and p.scheme.startswith("SSB"):
            s += (f"<p>The {v(abs(p.df), '.0f', 'Hz')} frequency error shifts every audio component "
                  f"by the same amount. Harmonics of the voice are no longer harmonic: the classic "
                  f"\"Donald Duck\" sound of a mistuned SSB receiver. Press Listen.</p>")
        elif abs(p.df) > 0:
            s += ("<p>A frequency error makes the phase error rotate continuously: the DSB output "
                  "beats up and down at the error frequency.</p>")
        return "<h3>Carrier optional</h3>" + s + keybox(
            "DSB and AM fade with LO phase as cos φ; SSB keeps its level but needs an accurate frequency.")


# =============================================================================== 3. FM in time
class FMTime(Experiment):
    title = "FM: frequency follows the message"
    blurb = "Constant amplitude, wandering frequency, seen three ways."
    book = "sec:ch04:angle"
    controls = [
        Choice("msg", "Message", ["Tone", "Triangle sweep", "Voice"]),
        Slider("fm", "Message frequency", 0.2, 5.0, 0.3, step=0.05, unit="kHz",
               enabled_if=lambda p: p.msg != "Voice"),
        Slider("dev", "Peak deviation Δf", 1, 50, 10, step=0.5, unit="kHz"),
    ]
    plots = [
        Plot("inst", "Instantaneous frequency f_c + Δf·m(t)", x="time (ms)", y="frequency (kHz)",
             xlim=(0, 10), ylim=(40, 160), legend=None),
        Plot("wave", "50 µs of the FM signal at the message peak and trough", x="time (µs)",
             y="amplitude", xlim=(0, 50), ylim=(-1.4, 2.0), legend="tl", legend_cols=2),
        ImagePlot("sg", "Spectrogram: where the power is, over time", x="time (ms)",
                  y="frequency (kHz)"),
    ]
    layout = [["inst", "wave"], ["sg", "sg"]]
    readouts = [
        Readout("dev", "Peak deviation", "kHz", ".1f"),
        Readout("beta", "Modulation index β = Δf / f_m", "", ".2f"),
        Readout("bw", "Carson bandwidth", "kHz", ".1f"),
        Readout("env", "Amplitude variation", "%", ".1f"),
    ]
    challenges = [
        Challenge("Make the frequency swing exactly from 80 to 120 kHz.",
                  lambda s: abs(s.p.dev - 20) < 0.3),
        Challenge("Narrowband FM: a β below 0.5 with a tone of at least 2 kHz.",
                  lambda s: s.p.msg == "Tone" and s.p.fm >= 2 and s.r.beta < 0.5),
        Challenge("Need a wide channel? Make the Carson bandwidth exceed 100 kHz.",
                  lambda s: s.r.bw > 100),
    ]
    fs, fc, dur = 1e6, 100e3, 0.02

    def update(self, p):
        m = message(p.msg, p.fm * 1e3, self.fs, self.dur)
        t = np.arange(len(m)) / self.fs
        dev = p.dev * 1e3
        phase = 2 * np.pi * self.fc * t + 2 * np.pi * dev * np.cumsum(m) / self.fs
        x = np.cos(phase)
        pi_ = self.plot("inst")
        k = t <= 0.0102
        pi_.hband("swing", (self.fc - dev) / 1e3, (self.fc + dev) / 1e3, color=GREEN, alpha=0.10)
        pi_.hline("fc", self.fc / 1e3, color=GRAY, style=":", label="carrier 100 kHz", label_pos=0.85)
        pi_.line("f", t[k] * 1e3, (self.fc + dev * m[k]) / 1e3, color=NAVY, width=2.2)
        pi_.set_ylim(min(40, (self.fc - dev) / 1e3 - 10), max(160, (self.fc + dev) / 1e3 + 10))
        # 50 µs windows at the message maximum and minimum
        n50 = int(50e-6 * self.fs)
        imax, imin = int(np.argmax(m[:-n50])), int(np.argmin(m[:-n50]))
        tw = np.arange(n50) / self.fs * 1e6
        pw = self.plot("wave")
        pw.line("hi", tw, x[imax:imax + n50], color=RED, width=1.8,
                name=f"at the peak: {(self.fc + dev * m[imax]) / 1e3:.0f} kHz")
        pw.line("lo", tw, x[imin:imin + n50], color=NAVY, width=1.8,
                name=f"at the trough: {(self.fc + dev * m[imin]) / 1e3:.0f} kHz")
        f, tt, S = spectrogram(x, self.fs, nperseg=512, noverlap=384)
        keep = f <= 200e3
        Sdb = 10 * np.log10(S[keep] / S.max() + 1e-12)
        pg_ = self.plot("sg")
        pg_.image("sg", Sdb, x=(tt[0] * 1e3, tt[-1] * 1e3), y=(0, 200), cmap="heat",
                  levels=(-50, 0), colorbar=True, cbar_label="dB")
        W = p.fm if p.msg != "Voice" else 4.0
        beta = p.dev / W if p.msg == "Tone" else None
        env = 100 * np.std(np.abs(hilbert(x))[500:-500])
        self.readout(dev=p.dev, beta=beta if beta is not None else "—",
                     bw=2 * (p.dev + W), env=env)

    def story(self, p):
        s = (f"<p>In FM the message moves the carrier's <b>frequency</b>, not its amplitude. "
             f"The frequency swings {v(p.dev, '.1f', 'kHz')} either side of 100 kHz (top left): when "
             f"the message peaks the waveform is squeezed (red), in the troughs it is stretched (blue).</p>"
             "<p>The amplitude never changes. That is FM's superpower: amplitude noise and a "
             "saturating power amplifier do no harm, which is why every FM broadcast transmitter "
             "runs its final stage flat out.</p>"
             "<p>The spectrogram shows the power sliding up and down in frequency. The total "
             "bandwidth is roughly twice the swing plus twice the message bandwidth: "
             f"<b>Carson's rule</b>, here {v(2 * (p.dev + (p.fm if p.msg != 'Voice' else 4.0)), '.0f', 'kHz')}.</p>")
        return "<h3>A wandering frequency</h3>" + s + keybox(
            "Instantaneous frequency = f_c + Δf·m(t). Bandwidth ≈ 2(Δf + W).")


# =============================================================================== 4. Bessel
class FMBessel(Experiment):
    title = "FM spectrum: Bessel and Carson"
    blurb = "A single tone produces infinitely many sidebands. How many matter?"
    book = "sec:ch04:nbfm"
    controls = [
        Slider("beta", "Modulation index β", 0.0, 12.0, 1.0, step=0.01,
               help="β = peak deviation / tone frequency"),
        Slider("fm", "Tone frequency", 0.5, 15.0, 1.0, step=0.1, unit="kHz"),
        Toggle("db", "Spectrum in dB", False),
    ]
    plots = [
        Plot("lines", "Spectral lines of tone-modulated FM", x="frequency offset from the carrier (kHz)",
             y="line amplitude |Jₙ(β)|", legend="tr"),
        Plot("bessel", "Bessel functions Jₙ(β): the line amplitudes", x="β", y="Jₙ(β)",
             xlim=(0, 12), ylim=(-0.5, 1.05), legend="tr", legend_cols=2),
        Plot("pow", "Power captured vs bandwidth", x="bandwidth (kHz)", y="power inside (%)",
             ylim=(0, 105), legend="br"),
    ]
    layout = [["lines", "lines"], ["bessel", "pow"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("dev", "Peak deviation", "kHz", ".2f"),
        Readout("carson", "Carson bandwidth", "kHz", ".1f"),
        Readout("inside", "Power inside Carson", "%", ".1f"),
        Readout("carrier", "Power left in the carrier", "%", ".1f"),
    ]
    challenges = [
        Challenge("Make the carrier vanish completely (the first Bessel null).",
                  lambda s: abs(jv(0, s.p.beta)) < 0.02 and s.p.beta < 3),
        Challenge("Find the second carrier null.",
                  lambda s: abs(jv(0, s.p.beta)) < 0.02 and 4.5 < s.p.beta < 6.5),
        Challenge("Broadcast FM: 75 kHz deviation with a 15 kHz tone.",
                  lambda s: abs(s.p.beta * s.p.fm - 75) < 1 and s.p.fm >= 14.9),
        Challenge("Narrowband FM: only one pair of sidebands above 1 % of the power.",
                  lambda s: jv(2, s.p.beta) ** 2 < 0.01 and s.p.beta > 0.05),
    ]

    def update(self, p):
        beta, fm = p.beta, p.fm
        N, per = 4096, 16
        n = np.arange(N)
        z = np.exp(1j * beta * np.sin(2 * np.pi * per * n / N))
        Z = np.fft.fft(z) / N
        nmax = int(max(6, beta + 6))
        k = np.arange(-nmax, nmax + 1)
        meas = np.abs(Z[(k * per) % N])
        theo = np.abs(jv(k, beta))
        carson = 2 * (beta + 1) * fm
        pl = self.plot("lines")
        yv = (lambda a: 20 * np.log10(a + 1e-6)) if p.db else (lambda a: a)
        base = -60 if p.db else 0
        pl.band("carson", -carson / 2, carson / 2, color=GREEN, alpha=0.10)
        pl.stems("side", (k * fm)[k != 0], yv(meas[k != 0]), color=NAVY, base=base,
                 name="sidebands (simulated FFT)")
        pl.stems("car", [0], [yv(meas[nmax])], color=RED, base=base, size=9, name="carrier")
        pl.scatter("th", k * fm, yv(theo), color=GOLD, size=12, outline=ORANGE,
                   name="Bessel prediction |Jₙ(β)|")
        span = (nmax + 1) * fm
        pl.set_xlim(-span, span)
        pl.set_ylim(*((-60, 3) if p.db else (0, 1.08)))
        pl.set_labels(y="line level (dB)" if p.db else "line amplitude |Jₙ(β)|")
        pl.text("cl", carson / 2, -57 if p.db else 0.02, f" Carson {carson:.1f} kHz", color=GREEN,
                size=9, anchor=(0, 1))
        # Bessel curves
        b = np.linspace(0, 12, 601)
        pb = self.plot("bessel")
        for i, c in enumerate([RED, NAVY, GREEN, ORANGE]):
            pb.line(f"J{i}", b, jv(i, b), color=c, width=1.8, name=f"J{i} ({'carrier' if i == 0 else f'pair {i}'})")
        pb.hline("z", 0, color=GRAY, style="-", width=0.7)
        pb.vline("now", beta, color=PURPLE, style="--", label=f"β = {beta:.2f}", label_pos=0.08)
        # cumulative power vs bandwidth
        kk = np.arange(0, 40)
        cum = np.cumsum(np.where(kk == 0, 1, 2) * jv(kk, beta) ** 2) * 100
        pp = self.plot("pow")
        pp.line("cum", 2 * kk * fm, cum, color=NAVY, width=2.0, step=False)
        pp.scatter("cumd", 2 * kk * fm, cum, color=NAVY, size=6)
        pp.hline("98", 98, color=GREEN, style=":", label="98 %", label_pos=0.05)
        pp.vline("cs", carson, color=GREEN, style="--", label="Carson", label_pos=0.5)
        pp.set_xlim(0, max(4 * fm, 2.2 * carson))
        inside = 100 * np.sum(theo[np.abs(k) * fm <= carson / 2 + 1e-9] ** 2)
        self.readout(dev=beta * fm, carson=carson, inside=inside, carrier=100 * jv(0, beta) ** 2)

    def story(self, p):
        b = p.beta
        s = (f"<p>Modulate a carrier with a single {v(p.fm, '.1f', 'kHz')} tone and the spectrum is a "
             f"comb of lines {v(p.fm, '.1f', 'kHz')} apart, in principle forever. Line n has amplitude "
             f"Jₙ(β), a Bessel function of the modulation index β = {v(b)}.</p>")
        if b < 0.5:
            s += ("<p><b>Narrowband FM:</b> just a carrier and one pair of sidebands, exactly like AM "
                  "but with the sidebands in quadrature. Same bandwidth as AM.</p>")
        elif abs(jv(0, b)) < 0.05:
            s += ("<p>" + good("The carrier has vanished") + ": all the power is in the sidebands. "
                  "Engineers calibrate deviation meters exactly this way: raise the deviation until "
                  "the carrier line nulls on a spectrum analyser, at β = 2.405, 5.520, 8.654…</p>")
        else:
            s += ("<p>As β grows, power moves out of the carrier into more and more sidebands, but "
                  "the total never changes: FM has constant power. Notice how quickly the lines "
                  "beyond β + 1 die away.</p>")
        s += (f"<p>That is why <b>Carson's rule</b>, B ≈ 2(β + 1)·f_m, works: the shaded band holds "
              f"{v(self.r.get('inside', 0), '.1f', '%')} of the power.</p>")
        return "<h3>Infinitely many sidebands</h3>" + s + keybox(
            "Σ Jₙ²(β) = 1 for every β: FM moves power around, it never adds any.")


# =============================================================================== 5. threshold
class FMThreshold(Experiment):
    title = "FM noise, clicks and the threshold"
    blurb = "Trade bandwidth for SNR, until the noise wraps the phase around the origin."
    book = "sec:ch04:threshold"
    animate = True
    fps = 8
    controls = [
        Slider("cnr", "Carrier-to-noise ratio", 0, 30, 18, step=0.5, unit="dB",
               help="Measured in the Carson bandwidth 2(β+1)·f_m"),
        Slider("beta", "Modulation index β", 1, 10, 5, step=0.1),
        Button("listen", "▶  Listen (1 kHz tone + noise)", primary=True),
    ]
    plots = [
        ConstellationPlot("ph", "Received phasor (complex envelope)", lim=2.2),
        Plot("aud", "Discriminator output", x="time (ms)", y="output",
             xlim=(0, 10), ylim=(-3, 4.0), legend="tl", legend_cols=2),
        Plot("curve", "Output SNR vs CNR", x="CNR in the Carson bandwidth (dB)",
             y="audio SNR (dB)", xlim=(0, 30), ylim=(-10, 75), legend="tl"),
    ]
    layout = [["ph", "aud"], ["ph", "curve"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("snr", "Audio SNR", "dB", ".1f"),
        Readout("theory", "Above-threshold theory", "dB", ".1f"),
        Readout("clicks", "Clicks per second", "", ".0f", good=lambda x: x < 1),
        Readout("adv", "FM advantage over SSB", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Hear the clicks: drop below threshold until there are more than 20 clicks per second.",
                  lambda s: s.r.clicks > 20),
        Challenge("Beat SSB at the same power by 20 dB or more (stay above threshold).",
                  lambda s: s.r.adv >= 20 and s.r.clicks < 2,
                  hint="FM's advantage grows as 1.5·β², but a bigger β needs a higher CNR to stay above threshold."),
        Challenge("Find the threshold: measured SNR 2–4 dB below the theory line.",
                  lambda s: 2 <= s.r.theory - s.r.snr <= 4),
    ]
    fs, fm, N = 96e3, 1e3, 96 * 340        # an integer number of tone periods (340 ms)

    def simulate(self, cnr, beta, N, rng):
        t = np.arange(N) / self.fs
        z = np.exp(1j * beta * np.sin(2 * np.pi * self.fm * t))
        bt = 2 * (beta + 1) * self.fm
        sigma2 = self.fs / (10 ** (cnr / 10) * bt)
        n = np.sqrt(sigma2 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        y = an.brickwall(z + n, self.fs, bt / 2)                     # IF filter
        d = an.fm_discriminator_hz(y, self.fs) / (beta * self.fm)     # ≈ cos(2π f_m t)
        audio = an.brickwall(d, self.fs, 1.05 * self.fm)
        e = np.unwrap(np.angle(y * np.conj(z)))
        slips = np.count_nonzero(np.diff(np.round(e / (2 * np.pi))))
        return y, d, audio, slips / (N / self.fs)

    def update(self, p):
        self.sweep = []
        self.draw(p)

    def tick(self, p):
        self.draw(p)
        self.plot_sweep(p)

    def draw(self, p):
        y, d, audio, cps = self.simulate(p.cnr, p.beta, self.N, self.rng)
        snr = snr_tone(audio, self.fm, self.fs)
        gamma = 10 ** (p.cnr / 10) * 2 * (p.beta + 1)
        th = 10 * np.log10(3 * p.beta ** 2 * (p.beta + 1) * 10 ** (p.cnr / 10))
        ph = self.plot("ph")
        a = np.linspace(0, 2 * np.pi, 200)
        ph.points("y", y[-2500:], color=NAVY, size=3, alpha=0.35)
        ph.line("circ", np.cos(a), np.sin(a), color=GREEN, width=1.5)
        ph.scatter("o", [0], [0], color=RED, size=10, symbol="+")
        t = np.arange(len(d)) / self.fs * 1e3
        k = t <= 10.1
        pa = self.plot("aud")
        pa.line("raw", t[k], np.clip(d[k], -3, 4), color=GRAY, width=1.0, alpha=0.8,
                name="raw discriminator")
        pa.line("a", t[k], audio[k], color=NAVY, width=2.0, name="after the audio filter")
        self.readout(snr=snr, theory=th, clicks=cps, adv=snr - 10 * np.log10(gamma))
        pc = self.plot("curve")
        c = np.linspace(0, 30, 61)
        pc.line("th", c, 10 * np.log10(3 * p.beta ** 2 * (p.beta + 1)) + c, color=NAVY, width=2.0,
                name="FM theory (above threshold)")
        pc.line("ssb", c, c + 10 * np.log10(2 * (p.beta + 1)), color=GRAY, width=1.6, style="--",
                name="SSB / DSB at the same power")
        pc.scatter("now", [p.cnr], [snr], color=ORANGE, size=14, symbol="d", name="right now")
        self.plot_sweep(p)

    def plot_sweep(self, p):
        sw = getattr(self, "sweep", [])
        pc = self.plot("curve")
        if sw:
            x, yv = zip(*sw)
            pc.scatter("sim", x, yv, color=RED, size=8, outline=RED, name="simulated sweep")

    def background(self, p):
        rng = np.random.default_rng(5)
        for cnr in np.arange(0, 30.1, 2.0 if not self.quick else 6.0):
            y, d, audio, cps = self.simulate(cnr, p.beta, self.N, rng)
            yield (cnr, snr_tone(audio, self.fm, self.fs))

    def progress(self, p, item):
        self.sweep.append(item)
        self.plot_sweep(p)

    def on_listen(self, p):
        rng = np.random.default_rng()
        y, d, audio, cps = self.simulate(p.cnr, p.beta, 96 * 1500, rng)
        self.play_audio(np.clip(an.brickwall(d, self.fs, 4000.0), -4, 4)[::2], self.fs / 2,
                        "the FM receiver output")

    def story(self, p):
        cps = self.r.get("clicks", 0)
        s = ("<p>The blue cloud is the received complex envelope: the signal is a point running "
             "round the green circle; noise blurs it. The discriminator measures how fast the "
             "angle turns.</p>")
        if cps > 2:
            s += (f"<p>{bad('Below threshold.')} The noise is now big enough to drag the phasor "
                  f"<i>around the origin</i> (red cross) {v(cps, '.0f')} times a second. Each trip "
                  f"adds a full 2π of phase in an instant: the spikes in the raw output, heard as "
                  f"clicks. The SNR falls off a cliff below the theory line.</p>")
        else:
            s += (f"<p>{good('Above threshold.')} Noise only nudges the phase. FM converts its "
                  f"wide bandwidth into SNR: theory gives 3β²(β+1)·CNR, here "
                  f"{v(self.r.get('theory', 0), '.1f', 'dB')}, and you beat SSB at the same power by "
                  f"{v(self.r.get('adv', 0), '.1f', 'dB')}.</p>")
        s += ("<p>The red circles sweep the CNR in the background for your β. Increase β: the curve "
              "rises, and the knee moves right. Wideband FM is magnificent, as long as you can "
              "afford the CNR.</p>")
        return "<h3>Bandwidth for SNR, until the knee</h3>" + s + keybox(
            "FM threshold ≈ 10 dB CNR. Below it, clicks; above it, a 1.5β² advantage over SSB.")


# =============================================================================== 6. stereo
class FMStereo(Experiment):
    title = "FM stereo multiplex"
    blurb = "Two channels through one FM transmitter, compatible with every mono radio since 1961."
    book = "sec:ch04:stereo"
    controls = [
        Choice("prog", "Programme", ["Left only", "Right only", "Both"]),
        Slider("perr", "Decoder pilot phase error", 0, 90, 2, step=0.5, unit="°"),
        Toggle("stereo", "Stereo decoder on", True),
        Slider("cnr", "Carrier-to-noise ratio", 15, 50, 40, step=0.5, unit="dB"),
        Toggle("rds", "Add the RDS data subcarrier", True),
        Button("listen", "▶  Listen in stereo", primary=True),
    ]
    plots = [
        SpectrumPlot("mpx", "Demodulated multiplex (MPX) spectrum", x="frequency (kHz)",
                     xlim=(0, 62), ylim=(-75, 8), legend=None),
        Plot("lr", "Recovered left and right channels", x="time (ms)", y="amplitude",
             xlim=(0, 5), ylim=(-1.4, 2.0), legend="tl", legend_cols=2),
        BarPlot("lev", "Where each tone ends up", y="level (dB)", ylim=(-70, 8)),
    ]
    layout = [["mpx", "mpx"], ["lr", "lev"]]
    readouts = [
        Readout("sep", "Stereo separation", "dB", ".1f", good=lambda x: x > 30),
        Readout("snr", "Audio SNR", "dB", ".1f"),
        Readout("bw", "Carson bandwidth", "kHz", ".0f"),
        Readout("mode", "Receiver", "", None),
    ]
    challenges = [
        Challenge("Broadcast quality is about 40 dB of separation. Find the largest pilot phase "
                  "error that still achieves it (10° or more).",
                  lambda s: s.p.stereo and s.r.sep > 40 and s.p.perr >= 10),
        Challenge("Ruin it: a pilot phase error that drops the separation below 10 dB.",
                  lambda s: s.p.stereo and s.r.sep < 10 and s.p.perr > 0),
        Challenge("Become a 1961 mono radio: switch the decoder off and hear L + R in both ears.",
                  lambda s: not s.p.stereo),
    ]
    fs, dur, fl, fr = 400e3, 0.04, 1000.0, 2500.0

    def chain(self, p, dur=None, rng=None):
        dur = dur or self.dur
        rng = rng if rng is not None else self.rng
        t = np.arange(int(self.fs * dur)) / self.fs
        L = np.sin(2 * np.pi * self.fl * t) if p.prog != "Right only" else 0 * t
        R = np.sin(2 * np.pi * self.fr * t) if p.prog != "Left only" else 0 * t
        mpx = an.stereo_mpx(L, R, self.fs, pilot=0.1, rds=0.05 if p.rds else 0.0, rng=rng)
        z = an.fm_complex(mpx, self.fs, 75e3)
        z = cl.awgn(z, p.cnr - 10 * np.log10(self.fs / 256e3), rng)
        rx = an.fm_discriminator_hz(z, self.fs) / 75e3
        Lh, Rh = an.stereo_decode(rx, self.fs, p.perr, p.stereo)
        return rx, Lh, Rh, t

    def update(self, p):
        rx, Lh, Rh, t = self.chain(p)
        pm = self.plot("mpx")
        pm.psd("P", rx, self.fs, 4096, scale=1e3, color=NAVY, width=1.4, onesided=True)
        pm.band("lpr", 0.03, 15, color=GREEN, alpha=0.08)
        pm.band("lmr", 23, 53, color=PURPLE, alpha=0.07)
        pm.text("t1", 1, 7, "L+R (mono)", color=GREEN, size=9, anchor=(0, 0))
        pm.text("t2", 24, 7, "L−R on a suppressed 38 kHz carrier", color=PURPLE, size=9,
                anchor=(0, 0))
        pm.vline("pil", 19, color=RED, style=":", label="19 kHz pilot", label_pos=0.7)
        if p.rds:
            pm.vline("rds", 57, color=ORANGE, style=":", label="RDS 57 kHz", label_pos=0.7)
        pl = self.plot("lr")
        a0 = int(0.01 * self.fs)
        tt = t[a0:a0 + int(0.0051 * self.fs)]
        pl.line("L", (tt - tt[0]) * 1e3, Lh[a0:a0 + len(tt)], color=NAVY, width=2.0, name="left out")
        pl.line("R", (tt - tt[0]) * 1e3, Rh[a0:a0 + len(tt)], color=RED, width=2.0, name="right out")
        lev = lambda x, f: 20 * np.log10(an.tone_level(x, f, self.fs) + 1e-6)
        vals, labs, cols = [], [], []
        seps = []
        if p.prog != "Right only":
            a, b = lev(Lh, self.fl), lev(Rh, self.fl)
            vals += [a, b]
            labs += ["1 kHz → L", "1 kHz → R"]
            cols += [NAVY, ORANGE]
            seps.append(a - b)
        if p.prog != "Left only":
            a, b = lev(Rh, self.fr), lev(Lh, self.fr)
            vals += [a, b]
            labs += ["2.5 kHz → R", "2.5 kHz → L"]
            cols += [RED, ORANGE]
            seps.append(a - b)
        pb = self.plot("lev")
        x = np.arange(len(vals))
        pb.bars("b", x, np.maximum(vals, -68), width=0.6, colors=cols, base=-70)
        pb.set_xticks([(i, s) for i, s in enumerate(labs)])
        pb.set_xlim(-0.6, len(vals) - 0.4)
        for i, val in enumerate(vals):
            pb.text(f"v{i}", i, max(val, -68), f"{int(round(val))} dB", anchor=(0.5, 1.05), bold=True, size=9)
        sig = Lh if p.prog != "Right only" else Rh
        f0 = self.fl if p.prog != "Right only" else self.fr
        audio_snr = snr_tone(sig, f0, self.fs) if p.prog != "Both" else None
        self.readout(sep=min(seps), snr=audio_snr if audio_snr is not None else "—",
                     bw=2 * (75 + 53), mode="stereo" if p.stereo else "mono (L+R)")

    def on_listen(self, p):
        rx, Lh, Rh, t = self.chain(p, dur=1.2, rng=np.random.default_rng())
        stereo = np.stack([Lh[::10], Rh[::10]], 1)
        self.play_audio(stereo, self.fs / 10, "left and right")

    def story(self, p):
        sep = self.r.get("sep", 0)
        s = ("<p>The stereo signal had to fit into an existing FM channel without upsetting mono "
             "radios. The trick, standardised in 1961: send <b>L+R</b> as ordinary audio (a mono "
             "radio hears just that), and <b>L−R</b> as DSB-SC on a 38 kHz subcarrier, above "
             "the audio band.</p>"
             "<p>DSB-SC needs a perfectly phased carrier at the receiver (Experiment 2!). So the "
             "transmitter adds a small <b>pilot</b> at exactly half the subcarrier, 19 kHz. The "
             "receiver doubles its frequency to rebuild the 38 kHz carrier, demodulates L−R, and "
             "forms L = (S + D)/2, R = (S − D)/2.</p>")
        if not p.stereo:
            s += "<p>Decoder off: both speakers get L+R, exactly what a mono radio plays.</p>"
        elif p.perr > 0:
            s += (f"<p>A {v(p.perr, '.1f', '°')} error in the regenerated carrier scales the "
                  f"difference signal by cos φ, and L leaks into R. Separation is now "
                  f"{v(sep, '.1f', 'dB')} (theory: 20·log₁₀ cot²(φ/2)).</p>")
        else:
            s += f"<p>With a phase-perfect carrier the separation is {v(sep, '.1f', 'dB')}.</p>"
        return "<h3>A compatible multiplex</h3>" + s + keybox(
            "Pilot at f/2, subcarrier at f: pilot-aided coherent detection, 1961 style.")


# =============================================================================== the lab
LAB = st.Lab(14, "Analog Modulation: AM, SSB, FM and Stereo", chapter=4,
             chapter_title="Analog Modulation and the Classic Radio",
             experiments=[AMEnvelope, DSBandSSB, FMTime, FMBessel, FMThreshold, FMStereo])

if __name__ == "__main__":
    st.run(LAB)
