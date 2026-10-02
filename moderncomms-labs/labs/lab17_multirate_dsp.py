"""Lab 17 · Digital Filters and Multirate DSP   (Chapter 6)

Run it:      python labs/lab17_multirate_dsp.py
Self-test:   python labs/lab17_multirate_dsp.py --selftest

A software radio is mostly filters running at different sample rates. Drag poles and zeros
around the z-plane, design FIR and IIR filters and break them with fixed-point coefficients,
watch a tone alias when you throw samples away, split a wideband capture into channels with a
polyphase filter bank, build Hogenauer's multiplier-free CIC, hear (well, see) an NCO's spurs,
rotate vectors with nothing but shifts and adds, and finally run a complete digital
down-converter that pulls one QPSK channel out of a crowded 16 MS/s capture.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy import signal as sg

import commlib as cl
from commlib import multirate as mr
import studio as st
from studio import (Experiment, Slider, IntSlider, Choice, Toggle, Button, Heading, Plot,
                    SpectrumPlot, ConstellationPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
def fir_db(h, n=4096):
    """(f in cycles/sample on [0, 0.5], |H| in dB) of an FIR filter via a zero-padded FFT."""
    H = np.fft.rfft(h, 2 * n)
    return np.fft.rfftfreq(2 * n), 20 * np.log10(np.abs(H) + 1e-12)


def unit_circle(p, key="uc"):
    a = np.linspace(0, 2 * np.pi, 361)
    p.line(key, np.cos(a), np.sin(a), color=GRAY, width=1.2)
    p.hline(key + "x", 0, color=GRAY, style="-", width=0.6)
    p.vline(key + "y", 0, color=GRAY, style="-", width=0.6)


def tone_spectrum(x, fs, nfft=None):
    """Blackman-Harris amplitude spectrum in dB (peak 0 dB ≈ a full-scale tone), one-sided for
    real x, two-sided (shifted) for complex x."""
    n = nfft or len(x)
    x = x[:n]
    w = sg.windows.blackmanharris(n)
    X = np.fft.fft(x * w) / np.sum(w)
    if np.iscomplexobj(x):
        f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
        return f, 20 * np.log10(np.abs(np.fft.fftshift(X)) + 1e-12)
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, 20 * np.log10(2 * np.abs(X[:len(f)]) + 1e-12)


def level_at(f, P, f0, df):
    """Peak level (dB) of the spectrum P within ±df of f0."""
    m = np.abs(f - f0) <= df
    return float(P[m].max()) if m.any() else -200.0


# =============================================================================== 1. pole-zero
class PoleZero(Experiment):
    title = "Pole–zero playground"
    blurb = "Place a pair of poles and a pair of zeros; the frequency response follows the geometry."
    book = "sec:ch06:ztrans"
    controls = [
        Heading("Poles (a conjugate pair)"),
        Slider("rp", "Pole radius r", 0.0, 1.1, 0.70, step=0.005,
               help="Distance of the poles from the origin. r ≥ 1 is unstable"),
        Slider("fp", "Pole angle", 0.0, 0.5, 0.10, step=0.002, unit="× f_s",
               help="The frequency the poles sit at: angle / 2π"),
        Heading("Zeros (a conjugate pair)"),
        Slider("rz", "Zero radius", 0.0, 1.5, 1.0, step=0.005,
               help="Zeros on the unit circle (radius 1) give perfect nulls"),
        Slider("fz", "Zero angle", 0.0, 0.5, 0.30, step=0.002, unit="× f_s"),
        Heading("Probe"),
        Slider("probe", "Probe frequency", 0.0, 0.5, 0.06, step=0.002, unit="× f_s",
               help="The point e^jω on the unit circle where the response is evaluated"),
    ]
    plots = [
        Plot("zp", "The z-plane: |H| = (zero distances) / (pole distances)", x="real part",
             y="imaginary part", xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), aspect=True, legend=None),
        Plot("mag", "Magnitude response", x="frequency (cycles/sample)", y="gain (dB)",
             xlim=(0, 0.5), ylim=(-60, 50), legend=None),
        Plot("imp", "Impulse response h[n]", x="sample n", y="amplitude", xlim=(-1, 60),
             legend=None),
    ]
    layout = [["zp", "mag"], ["zp", "imp"]]
    col_stretch = [5, 6]
    readouts = [
        Readout("probe", "Gain at the probe", "dB", ".1f"),
        Readout("peak", "Peak gain", "dB", ".1f"),
        Readout("dc_ratio", "DC gain over gain at 0.25", "dB", ".1f"),
        Readout("state", "Stability", "", None),
    ]
    challenges = [
        Challenge("Build the book's notch: a null at 0.15 cycles/sample whose −3 dB width is "
                  "below 0.02 cycles/sample, with stable poles.",
                  lambda s: abs(s.p.fz - 0.15) < 0.003 and s.r.notch is not None
                  and s.r.notch < 0.02 and s.p.rp < 1,
                  hint="Zeros on the circle at 0.15; poles at the same angle, just inside the circle. "
                       "Notch width ≈ (1 − r)/π."),
        Challenge("Make a resonator: a peak of at least 30 dB at 0.25 cycles/sample, still stable.",
                  lambda s: s.r.peak >= 30 and abs(s.r.fpeak - 0.25) < 0.01 and s.p.rp < 1),
        Challenge("Turn it into a low-pass: zeros at z = −1 and at least 30 dB more gain at DC "
                  "than at 0.25 cycles/sample, stable.",
                  lambda s: abs(s.p.fz - 0.5) < 0.003 and abs(s.p.rz - 1) < 0.006
                  and s.r.dc_ratio >= 30 and s.p.rp < 1,
                  hint="Put the poles on the positive real axis (angle 0) and move them towards z = 1."),
    ]

    @staticmethod
    def coeffs(p):
        b = np.real(np.poly(p.rz * np.exp(2j * np.pi * p.fz * np.array([1, -1]))))
        a = np.real(np.poly(p.rp * np.exp(2j * np.pi * p.fp * np.array([1, -1]))))
        return b, a

    def update(self, p):
        b, a = self.coeffs(p)
        f = np.linspace(0, 0.5, 2001)
        z = np.exp(2j * np.pi * f)
        H = np.polyval(b, z) / np.polyval(a, z)          # b, a in descending powers of z
        Hdb = 20 * np.log10(np.abs(H) + 1e-12)
        poles = p.rp * np.exp(2j * np.pi * p.fp * np.array([1, -1]))
        zeros = p.rz * np.exp(2j * np.pi * p.fz * np.array([1, -1]))
        e = np.exp(2j * np.pi * p.probe)
        dz, dp = np.abs(e - zeros), np.abs(e - poles)
        gprobe = 20 * np.log10(np.prod(dz) / max(np.prod(dp), 1e-12) + 1e-12)
        # z-plane
        pz = self.plot("zp")
        unit_circle(pz)
        segs = lambda pts: (np.ravel([[q.real, e.real, np.nan] for q in pts]),
                            np.ravel([[q.imag, e.imag, np.nan] for q in pts]))
        x, y = segs(zeros)
        pz.line("vz", x, y, color=RED, width=1.4, style="--")
        x, y = segs(poles)
        pz.line("vp", x, y, color=NAVY, width=1.4, style="--")
        pz.scatter("zeros", zeros.real, zeros.imag, color=RED, size=13, outline=RED)
        pz.scatter("poles", poles.real, poles.imag, color=NAVY, size=15, symbol="x")
        pz.scatter("probe", [e.real], [e.imag], color=GREEN, size=11)
        pz.text("pl", e.real * 1.12, e.imag * 1.12, "e^jω", color=GREEN, anchor=(0.5, 0.5),
                size=9, bold=True)
        # magnitude response
        pm = self.plot("mag")
        pm.line("H", f, np.clip(Hdb, -80, 80), color=NAVY, width=2.2, fill=-60, fill_alpha=0.08)
        pm.vline("fp", p.fp, color=NAVY, style=":", width=1.0, label="poles", label_pos=0.95)
        pm.vline("fz", p.fz, color=RED, style=":", width=1.0, label="zeros", label_pos=0.88)
        pm.scatter("pr", [p.probe], [np.clip(gprobe, -59, 49)], color=GREEN, size=12, symbol="d")
        # impulse response
        n = np.arange(61)
        imp = np.zeros(61)
        imp[0] = 1
        h = sg.lfilter(b, a, imp)
        pi_ = self.plot("imp")
        pi_.hline("z", 0, color=GRAY, style="-", width=0.8)
        pi_.stems("h", n, np.clip(h, -1e6, 1e6), color=NAVY, size=5)
        m = max(1.2, float(np.max(np.abs(h[:60]))) * 1.15)
        pi_.set_ylim(-min(m, 1e6), min(m, 1e6))
        # metrics
        k = int(np.argmax(Hdb))
        peak, fpeak = float(Hdb[k]), float(f[k])
        notch = None
        j = int(np.argmin(Hdb))
        ref = float(np.median(Hdb))
        if Hdb[j] < ref - 20:
            lo, hi = j, j
            while lo > 0 and Hdb[lo] < ref - 3:
                lo -= 1
            while hi < len(f) - 1 and Hdb[hi] < ref - 3:
                hi += 1
            notch = float(f[hi] - f[lo])
        H0 = abs(np.polyval(b, 1.0) / np.polyval(a, 1.0))
        Hq = abs(np.polyval(b, 1j) / np.polyval(a, 1j))
        dc_ratio = 20 * np.log10((H0 + 1e-12) / (Hq + 1e-12))
        state = "stable" if p.rp < 1 else ("on the edge" if p.rp < 1.0001 else "UNSTABLE")
        self.readout(probe=gprobe, peak=peak, fpeak=fpeak, notch=notch,
                     dc_ratio=float(np.clip(dc_ratio, -200, 200)), state=state,
                     dz=float(np.prod(dz)), dp=float(np.prod(dp)))

    def story(self, p):
        r = self.r
        s = (f"<p>Every frequency is a point e<sup>jω</sup> on the unit circle (green). The gain there is "
             f"the product of its distances to the zeros (red dashes, {v(r.get('dz', 0), '.2f')}) "
             f"divided by the product of its distances to the poles (blue dashes, "
             f"{v(r.get('dp', 0), '.2f')}): {v(r.get('probe', 0), '.1f', 'dB')} at the probe. "
             f"Slide the probe and watch the dashes stretch.</p>")
        if p.rp >= 1:
            s += ("<p>" + bad("Unstable.") + " A pole on or outside the circle makes the impulse "
                  "response grow (or ring) forever: every IIR design must keep its poles strictly "
                  "inside.</p>")
        elif p.rp > 0.9:
            s += (f"<p>The poles are close to the circle (r = {v(p.rp, '.3f')}), so the response "
                  f"peaks sharply near {v(p.fp, '.3f')} cycles/sample and the impulse response rings "
                  f"for about 1/(1 − r) ≈ {v(1 / max(1 - p.rp, 1e-3), '.0f')} samples. The −3 dB width "
                  f"of the peak is about (1 − r)/π.</p>")
        else:
            s += ("<p>Poles near the origin barely matter; move them towards the circle to make "
                  "a resonance, and put zeros on the circle to dig perfect nulls.</p>")
        if r.get("notch") is not None:
            s += (f"<p>The zeros dig a null at {v(p.fz, '.3f')} cycles/sample, "
                  f"{v(r['notch'], '.3f')} wide at −3 dB. A notch filter is exactly this: zeros on "
                  f"the circle and poles right behind them, so that everywhere else the two distances "
                  f"cancel.</p>")
        return "<h3>Geometry is the response</h3>" + s + keybox(
            "|H(e<sup>jω</sup>)| = Π|e<sup>jω</sup> − zₖ| / Π|e<sup>jω</sup> − pₖ|. Poles lift, zeros dig, and the closer "
            "they sit to the circle the sharper the effect.")


# =============================================================================== 2. FIR design
@lru_cache(maxsize=256)
def design_fir(method, ntaps, fp, dfw, beta):
    fs_ = fp + dfw
    fc = fp + dfw / 2
    if method == "Parks–McClellan":
        try:
            return sg.remez(ntaps, [0, fp, min(fs_, 0.499), 0.5], [1, 0], weight=[1, 10], fs=1.0,
                            maxiter=60)
        except Exception:
            return sg.firwin(ntaps, fc, window=("kaiser", 6.0), fs=1.0)
    win = {"Kaiser window": ("kaiser", beta), "Hamming window": "hamming",
           "Rectangular window": "boxcar", "Blackman window": "blackman"}[method]
    return sg.firwin(ntaps, fc, window=win, fs=1.0)


def fir_metrics(h, fp, dfw):
    f, H = fir_db(h)
    pb = H[f <= fp]
    sb = H[f >= fp + dfw]
    return f, H, float(-sb.max()) if len(sb) else 0.0, float(pb.max() - pb.min()) if len(pb) else 0.0


def kaiser_taps(A, dfw):
    return int(np.ceil((A - 8) / (2.285 * 2 * np.pi * dfw))) + 1


class FIRDesign(Experiment):
    title = "FIR design: windows vs equiripple"
    blurb = "Meet a stopband specification with as few taps as you can, then round the coefficients."
    book = "sec:ch06:fir"
    controls = [
        Heading("Specification"),
        Slider("fp", "Passband edge", 0.02, 0.30, 0.10, step=0.005, unit="× f_s"),
        Slider("dfw", "Transition width", 0.01, 0.12, 0.04, step=0.002, unit="× f_s"),
        Slider("spec", "Required attenuation", 20, 100, 60, step=1, unit="dB"),
        Heading("Design"),
        Choice("method", "Method", ["Rectangular window", "Hamming window", "Blackman window",
                                    "Kaiser window", "Parks–McClellan"], "Hamming window",
               style="menu"),
        IntSlider("ntaps", "Number of taps", 7, 201, 41, step=2),
        Slider("beta", "Kaiser β", 0.0, 14.0, 5.65, step=0.05,
               help="Larger β: lower sidelobes, wider transition", enabled_if=lambda p: p.method == "Kaiser window"),
        Heading("Fixed point"),
        Toggle("quant", "Round the coefficients", False),
        IntSlider("cbits", "Coefficient word length", 4, 24, 12, unit="bits",
                  enabled_if=lambda p: p.quant),
    ]
    plots = [
        SpectrumPlot("mag", "Magnitude response against the specification",
                     x="frequency (cycles/sample)", y="gain (dB)", xlim=(0, 0.5), ylim=(-130, 12),
                     legend="tr"),
        Plot("imp", "Impulse response (symmetric = linear phase)", x="tap (centred)",
             y="coefficient", legend=None),
        Plot("pb", "Passband ripple, close up", x="frequency (cycles/sample)", y="gain (dB)",
             ylim=(-1.2, 0.8), legend=None),
    ]
    layout = [["mag", "mag"], ["imp", "pb"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("att", "Stopband attenuation", "dB", ".1f"),
        Readout("ripple", "Passband ripple", "dB p-p", ".3f"),
        Readout("kaiser", "Kaiser's estimate for this spec", "taps", "int"),
        Readout("macs", "Multiplies per output (folded)", "", "int"),
    ]
    challenges = [
        Challenge("Meet the default specification (60 dB, transition 0.10 → 0.14) with a window "
                  "design.",
                  lambda s: "window" in s.p.method and s.r.att >= 60 and abs(s.p.fp - 0.1) < 0.003
                  and abs(s.p.dfw - 0.04) < 0.0015 and not s.p.quant,
                  hint="Only the Kaiser window can reach 60 dB; its β sets the attenuation, the "
                       "number of taps the transition width."),
        Challenge("Meet the same specification with Parks–McClellan using 71 taps or fewer "
                  "(the Kaiser window needs about 95).",
                  lambda s: s.p.method == "Parks–McClellan" and s.r.att >= 60 and s.p.ntaps <= 71
                  and abs(s.p.fp - 0.1) < 0.003 and abs(s.p.dfw - 0.04) < 0.0015 and not s.p.quant),
        Challenge("Round an equiripple design to 10 bits or fewer and still keep 50 dB of "
                  "stopband.",
                  lambda s: s.p.quant and s.p.cbits <= 10 and s.r.att >= 50
                  and s.p.method == "Parks–McClellan",
                  hint="Each coefficient bit buys about 6 dB of attenuation."),
        Challenge("See Gibbs: a rectangular window with at least 151 taps still cannot beat 35 dB.",
                  lambda s: s.p.method == "Rectangular window" and s.p.ntaps >= 151 and s.r.att < 35),
    ]

    def update(self, p):
        h0 = design_fir(p.method, int(p.ntaps), round(p.fp, 4), round(p.dfw, 4), round(p.beta, 3))
        h = mr.quantize_coefs(h0, int(p.cbits)) if p.quant else h0
        f, H, att, ripple = fir_metrics(h, p.fp, p.dfw)
        pm = self.plot("mag")
        fsb = p.fp + p.dfw
        xs = np.array([fsb, 0.5])
        pm.fill_between("forbid", xs, [-p.spec, -p.spec], [12, 12], color=RED, alpha=0.10)
        pm.line("specl", [fsb, 0.5], [-p.spec, -p.spec], color=RED, width=1.4, style="--",
                name=f"spec: {p.spec:.0f} dB")
        pm.band("trans", p.fp, fsb, color=GRAY, alpha=0.10)
        if p.quant:
            f0, H0 = fir_db(h0)
            pm.line("H0", f0, H0, color=GRAY, width=1.2, name="unrounded")
        pm.line("H", f, H, color=NAVY, width=2.0, name=f"{p.method}, {p.ntaps} taps"
                + (f", {p.cbits}-bit" if p.quant else ""))
        pm.hline("att", -att, color=ORANGE, style=":", width=1.2,
                 label=f"achieved {att:.1f} dB", label_pos=0.8)
        n = np.arange(len(h)) - (len(h) - 1) / 2
        pi_ = self.plot("imp")
        pi_.hline("z", 0, color=GRAY, style="-", width=0.8)
        pi_.stems("h", n, h, color=NAVY, size=4 if len(h) > 80 else 6, width=1.2)
        pi_.set_xlim(n[0] - 1, n[-1] + 1)
        pi_.set_ylim(-0.25 * np.max(np.abs(h)) - 0.01, 1.15 * np.max(np.abs(h)))
        pp = self.plot("pb")
        m = f <= p.fp * 1.05
        pp.band("pbb", 0, p.fp, color=GREEN, alpha=0.08)
        pp.line("H", f[m], H[m], color=NAVY, width=2.0)
        pp.set_xlim(0, p.fp * 1.05)
        self.readout(att=att, ripple=ripple, kaiser=kaiser_taps(p.spec, p.dfw),
                     macs=(len(h) + 1) // 2)

    def story(self, p):
        att = self.r.get("att", 0)
        need = self.r.get("kaiser", 0)
        ok = att >= p.spec
        s = (f"<p>The specification: pass everything up to {v(p.fp, '.3f')} cycles/sample, "
             f"attenuate everything above {v(p.fp + p.dfw, '.3f')} by {v(p.spec, '.0f', 'dB')}. "
             f"Your {v(p.ntaps, 'd')}-tap design reaches {v(att, '.1f', 'dB')}: "
             + (good("specification met.") if ok else bad("not yet.")) + "</p>")
        if "window" in p.method:
            s += ("<p>A <b>window design</b> truncates the ideal sinc response. The window's "
                  "sidelobes fix the attenuation (rectangular ≈ 21 dB, Hamming ≈ 53, Blackman ≈ 74, "
                  "Kaiser: whatever β asks for); more taps only narrow the transition. That is why "
                  "the rectangular window never improves: the Gibbs ripple stays put.</p>")
        else:
            s += ("<p><b>Parks–McClellan</b> spreads the error evenly (equiripple) across both "
                  "bands, the optimum for a given length: typically 10–20 % shorter than a Kaiser "
                  "design for the same specification.</p>")
        s += (f"<p>Kaiser's formula says this specification needs about {v(need, 'd')} taps: "
              f"cost grows with attenuation and inversely with the <i>relative</i> transition "
              f"width. Halve the transition and you double the filter.</p>")
        if p.quant:
            s += (f"<p>Rounding to {v(p.cbits, 'd')} bits adds a random-looking error to every "
                  f"coefficient: roughly 6 dB of stopband per bit (grey: the unrounded design).</p>")
        return "<h3>Paying for steepness</h3>" + s + keybox(
            "Taps ≈ (A − 8) / (2.285 · 2π Δf). Attenuation is bought with sidelobe control, "
            "steepness with length.")


# =============================================================================== 3. IIR
FAMILIES = ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"]


@lru_cache(maxsize=256)
def design_iir(family, order, fc, rp, rs):
    kw = dict(output="zpk", fs=1.0)
    if family == "Butterworth":
        return sg.butter(order, fc, **kw)
    if family == "Chebyshev I":
        return sg.cheby1(order, rp, fc, **kw)
    if family == "Chebyshev II":
        return sg.cheby2(order, rs, min(fc * 1.25, 0.49), **kw)
    return sg.ellip(order, rp, rs, fc, **kw)


def quantize_iir(z, p, k, struct, bits):
    """Return (sos or None, b, a, poles) after rounding coefficients to `bits`."""
    if struct == "Floating point":
        sos = sg.zpk2sos(z, p, k)
        return sos, None, None, p
    if struct == "Direct form":
        b, a = sg.zpk2tf(z, p, k)
        aq = mr.quantize_coefs(a, bits)
        bq = mr.quantize_coefs(b, bits)
        aq[0] = 1.0 if abs(aq[0] - 1) < 0.5 else aq[0]
        return None, bq, aq, np.roots(aq)
    sos = sg.zpk2sos(z, p, k)
    sq = sos.copy()
    for i in range(len(sq)):
        sq[i, :3] = mr.quantize_coefs(sos[i, :3], bits)
        sq[i, 4:] = mr.quantize_coefs(sos[i, 4:], bits, full_scale=2.0)
    poles = np.concatenate([np.roots(r[3:]) for r in sq])
    return sq, None, None, poles


class IIRFilters(Experiment):
    title = "IIR filters: sharp, cheap, fragile"
    blurb = "Butterworth to elliptic, their group delay, and what fixed-point coefficients do to them."
    book = "sec:ch06:iir"
    controls = [
        Heading("Design"),
        Choice("family", "Family", FAMILIES, "Butterworth", style="menu"),
        IntSlider("order", "Order", 1, 12, 6),
        Slider("fc", "Cut-off frequency", 0.02, 0.40, 0.20, step=0.005, unit="× f_s"),
        Slider("rp", "Passband ripple", 0.1, 3.0, 1.0, step=0.1, unit="dB",
               enabled_if=lambda p: p.family in ("Chebyshev I", "Elliptic")),
        Slider("rs", "Stopband attenuation", 20, 100, 60, step=1, unit="dB",
               enabled_if=lambda p: p.family in ("Chebyshev II", "Elliptic")),
        Heading("Fixed-point implementation"),
        Choice("struct", "Structure", ["Floating point", "Direct form", "Biquad cascade"]),
        IntSlider("bits", "Coefficient word length", 4, 24, 16, unit="bits",
                  enabled_if=lambda p: p.struct != "Floating point"),
        Toggle("fir", "Compare: 61-tap linear-phase FIR", False),
    ]
    plots = [
        SpectrumPlot("mag", "Magnitude response", x="frequency (cycles/sample)", y="gain (dB)",
                     xlim=(0, 0.5), ylim=(-110, 8), legend="bl"),
        Plot("gd", "Group delay: how long each frequency is held up", x="frequency (cycles/sample)",
             y="delay (samples)", xlim=(0, 0.5), legend="tr"),
        Plot("pz", "Poles (×) and zeros (o)", x="real part", y="imaginary part",
             xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect=True, legend=None),
    ]
    layout = [["mag", "pz"], ["gd", "gd"]]
    col_stretch = [3, 2]
    row_stretch = [3, 2]
    readouts = [
        Readout("att", "Attenuation at f_c + 0.05", "dB", ".1f"),
        Readout("gdvar", "Group-delay spread in passband", "samples", ".1f"),
        Readout("rmax", "Largest pole radius", "", ".4f", good=lambda x: x < 1),
        Readout("mults", "Multiplies per sample", "", "int"),
    ]
    challenges = [
        Challenge("Get 60 dB of attenuation just 0.05 above a 0.2 cut-off with order 6 or less.",
                  lambda s: s.r.att >= 60 and s.p.order <= 6 and abs(s.p.fc - 0.2) < 0.003
                  and s.r.rmax < 1,
                  hint="Only one family ripples in both bands to buy the steepest skirt."),
        Challenge("Break it: an 8th-order elliptic at cut-off 0.05 in direct form, unstable with "
                  "16-bit (or longer) coefficients.",
                  lambda s: s.p.family == "Elliptic" and s.p.order == 8 and abs(s.p.fc - 0.05) < 0.003
                  and s.p.struct == "Direct form" and s.p.bits >= 16 and s.r.rmax >= 1),
        Challenge("Save it: the same filter as a biquad cascade, stable with 8-bit coefficients "
                  "and still 50 dB down at f_c + 0.05.",
                  lambda s: s.p.family == "Elliptic" and s.p.order == 8 and abs(s.p.fc - 0.05) < 0.003
                  and s.p.struct == "Biquad cascade" and s.p.bits <= 8 and s.r.rmax < 1
                  and s.r.att >= 50),
        Challenge("Keep the passband group delay within 2 samples while reaching 30 dB at f_c + 0.05.",
                  lambda s: s.r.gdvar <= 2 and s.r.att >= 30 and s.r.rmax < 1,
                  hint="Poles near the circle mean long, uneven delay. Which family keeps its "
                       "passband poles furthest from the circle? Try a high cut-off."),
    ]

    def update(self, p):
        z, pp, k = design_iir(p.family, int(p.order), round(p.fc, 4), round(p.rp, 2), round(p.rs, 1))
        sos, b, a, poles = quantize_iir(z, pp, k, p.struct, int(p.bits))
        n = 2048
        if sos is not None:
            w, H = sg.sosfreqz(sos, worN=n, fs=1.0)
            gd = np.zeros(n)
            for r_ in sos:
                with np.errstate(all="ignore"):
                    gd += sg.group_delay((r_[:3], r_[3:]), w=w, fs=1.0)[1]
            zeros = np.concatenate([np.roots(r_[:3]) if np.any(r_[:3]) else [] for r_ in sos])
        else:
            w, H = sg.freqz(b, a, worN=n, fs=1.0)
            with np.errstate(all="ignore"):
                gd = sg.group_delay((b, a), w=w, fs=1.0)[1]
            zeros = np.roots(b) if np.any(b) else np.array([])
        Hdb = 20 * np.log10(np.abs(H) + 1e-12)
        rmax = float(np.max(np.abs(poles))) if len(poles) else 0.0
        stable = rmax < 1
        pm = self.plot("mag")
        if p.struct != "Floating point":
            w0, H0 = sg.sosfreqz(sg.zpk2sos(z, pp, k), worN=n, fs=1.0)
            pm.line("H0", w0, 20 * np.log10(np.abs(H0) + 1e-12), color=GRAY, width=1.2,
                    name="floating point")
        if p.fir:
            hf = sg.firwin(61, p.fc, fs=1.0)
            ff, Hf = fir_db(hf, 1024)
            pm.line("fir", ff, Hf, color=GREEN, width=1.4, style="--", name="61-tap FIR")
        pm.line("H", w, np.clip(Hdb, -140, 60), color=NAVY if stable else RED, width=2.0,
                name=f"{p.family}, order {p.order}" + ("" if p.struct == "Floating point"
                                                     else f", {p.bits}-bit {p.struct.lower()}"))
        pm.vline("fc", p.fc, color=GRAY, style=":", width=1.0)
        pm.vline("f5", min(p.fc + 0.05, 0.5), color=ORANGE, style=":", width=1.0,
                 label="f_c + 0.05", label_pos=0.9)
        pg_ = self.plot("gd")
        pb = w <= p.fc * 0.95
        gfin = gd[np.isfinite(gd)]
        top = float(np.percentile(gfin, 99.5)) if gfin.size else 10
        if p.fir:
            pg_.hline("fir", 30, color=GREEN, style="--", width=1.4,
                      label="FIR: 30 samples, constant", label_pos=0.6)
        if stable:
            pg_.line("gd", w, np.clip(gd, 0, 500), color=NAVY, width=2.0, fill=0, fill_alpha=0.08)
        else:
            pg_.text("unst", 0.25, None, "unstable: no meaningful group delay", color=RED,
                     anchor=(0.5, 0), size=10, bold=True)
        pg_.band("pbb", 0, p.fc, color=GREEN, alpha=0.06)
        pg_.set_ylim(0, max(10, min(400, top * 1.15), 35 if p.fir else 0))
        # pole-zero map
        pz = self.plot("pz")
        unit_circle(pz)
        if p.struct != "Floating point":
            pz.scatter("p0", pp.real, pp.imag, color=GRAY, size=12, symbol="x")
        zz = np.asarray(zeros)
        zz = zz[np.abs(zz) < 50] if zz.size else zz
        pz.scatter("z", zz.real if zz.size else [], zz.imag if zz.size else [], color=RED,
                   size=10, outline=RED)
        pz.scatter("p", poles.real, poles.imag, color=NAVY if stable else RED, size=13, symbol="x")
        if p.struct != "Floating point" and p.fc < 0.1:
            pz.set_xlim(0.55, 1.15)
            pz.set_ylim(-0.3, 0.3)
        else:
            pz.set_xlim(-1.3, 1.3)
            pz.set_ylim(-1.3, 1.3)
        f5 = min(p.fc + 0.05, 0.499)
        att = -float(np.interp(f5, w, Hdb)) if stable else 0.0
        gdv = float(np.ptp(gd[pb])) if stable and pb.any() else np.inf
        nsec = int(np.ceil(p.order / 2))
        mults = 5 * nsec if p.struct != "Direct form" else 2 * p.order + 1
        self.readout(att=att, gdvar=gdv if np.isfinite(gdv) else 999.0, rmax=rmax, mults=mults)

    def story(self, p):
        r = self.r
        fam = {"Butterworth": "is maximally flat: no ripple anywhere, the gentlest skirt",
               "Chebyshev I": "ripples in the passband to buy a steeper skirt",
               "Chebyshev II": "has a flat passband, ripples in the stopband",
               "Elliptic": "ripples in both bands: the steepest skirt any filter of this order can have"}
        s = (f"<p>An order-{v(p.order, 'd')} {p.family} filter {fam[p.family]}. It costs about "
             f"{v(r.get('mults', 0), 'd')} multiplies per sample, against 31 for the 61-tap FIR: "
             f"feedback reuses every coefficient.</p>"
             f"<p>The price is phase. Frequencies near the cut-off are held up longer (bottom), a "
             f"spread of {v(min(r.get('gdvar', 0), 999), '.1f', 'samples')} across the passband that "
             f"smears pulses.</p>")
        if p.struct == "Direct form" and r.get("rmax", 0) >= 1:
            s += ("<p>" + bad("Unstable!") + f" Rounding the coefficients of one big denominator "
                  f"polynomial to {v(p.bits, 'd')} bits moved its tightly clustered roots across the "
                  f"unit circle (grey × = where they should be). High-order polynomials are "
                  f"exquisitely sensitive to their coefficients.</p>")
        elif p.struct == "Biquad cascade":
            s += (f"<p>As a cascade of {v(int(np.ceil(p.order / 2)), 'd')} second-order sections, "
                  f"each pole pair depends on only two coefficients, so rounding moves each a little "
                  f"and independently. This is why every practical IIR filter is built from biquads.</p>")
        elif p.struct == "Direct form":
            s += ("<p>Direct form looks fine here; now try a narrow filter (cut-off 0.05) at high "
                  "order, where the poles crowd together near z = 1.</p>")
        return "<h3>Feedback buys steepness</h3>" + s + keybox(
            "IIR: steep and cheap, but non-linear phase and coefficient-sensitive. Build them as "
            "biquads; keep FIR for data paths that need linear phase.")


# =============================================================================== 4. decimation
class RateChange(Experiment):
    title = "Decimation, interpolation, aliasing"
    blurb = "Throw away samples and the spectrum folds; insert zeros and it repeats."
    book = "sec:ch06:multirate"
    controls = [
        Choice("mode", "Operation", ["Decimate ↓M", "Interpolate ↑L"]),
        IntSlider("M", "Rate-change factor", 2, 8, 4),
        Heading("Signal (48 kS/s input)"),
        Slider("f2", "Interferer frequency", 0.0, 24.0, 9.0, step=0.05, unit="kHz",
               enabled_if=lambda p: p.mode.startswith("Dec"),
               help="A second tone. The wanted tone is fixed at 1 kHz"),
        Slider("a2", "Interferer level", -40, 0, 0, step=1, unit="dB",
               enabled_if=lambda p: p.mode.startswith("Dec")),
        Heading("Filter"),
        Toggle("filt", "Anti-alias / anti-image filter", False),
        IntSlider("taps", "Filter length", 7, 127, 31, step=2, unit="taps",
                  enabled_if=lambda p: p.filt),
    ]
    plots = [
        SpectrumPlot("inp", "Before the rate change", x="frequency (kHz)", y="level (dB)",
                     ylim=(-120, 12), legend="tr"),
        SpectrumPlot("out", "After the rate change", x="frequency (kHz)", y="level (dB)",
                     ylim=(-120, 12), legend="tr"),
        Plot("time", "In time", x="time (ms)", y="amplitude", ylim=(-1.5, 2.3), legend="tl",
             legend_cols=3),
    ]
    layout = [["inp", "out"], ["time", "time"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("alias", "Where the interferer lands", "kHz", ".2f"),
        Readout("rej", "Alias / image level", "dB rel. wanted", ".1f",
                good=lambda x: x < -60),
        Readout("rate", "Output rate", "kS/s", ".1f"),
        Readout("cost", "Multiplies per output", "", None),
    ]
    challenges = [
        Challenge("No filter, M = 4: tune the interferer so its alias lands exactly on the 1 kHz "
                  "tone.",
                  lambda s: s.p.mode.startswith("Dec") and not s.p.filt and s.p.M == 4
                  and abs(s.r.alias - 1.0) < 0.06 and s.p.f2 > 6.5,
                  hint="After ↓4 the new sample rate is 12 kHz: everything folds about multiples of 12 kHz."),
        Challenge("Filter on, M = 4, interferer at 9 kHz: push its alias 60 dB below the wanted tone "
                  "with 29 taps or fewer.",
                  lambda s: s.p.mode.startswith("Dec") and s.p.filt and s.p.M == 4
                  and abs(s.p.f2 - 9) < 0.06 and s.p.a2 >= -0.5 and s.r.rej <= -60 and s.p.taps <= 29),
        Challenge("Interpolate by 4 and suppress every image by at least 50 dB.",
                  lambda s: s.p.mode.startswith("Int") and s.p.M == 4 and s.r.rej <= -50),
    ]
    fs = 48.0          # kHz

    @staticmethod
    @lru_cache(maxsize=64)
    def taps_for(M, ntaps):
        """Low-pass with its transition centred just inside the new Nyquist frequency."""
        return sg.firwin(ntaps, 0.5 / M * 0.85, window=("kaiser", 7.0), fs=1.0)

    def update(self, p):
        if p.mode.startswith("Dec"):
            self.decimate(p)
        else:
            self.interpolate(p)

    def decimate(self, p):
        M, fs = int(p.M), self.fs
        n = 8192 * M + 600
        t = np.arange(n) / fs
        a2 = 10 ** (p.a2 / 20)
        wanted = np.sin(2 * np.pi * 1.0 * t)
        intf = a2 * np.sin(2 * np.pi * p.f2 * t + 0.4)
        x = wanted + intf
        h = self.taps_for(M, int(p.taps)) if p.filt else np.array([1.0])
        y = mr.polyphase_decimate(x, h, M)[300 // M + 1:][:8192]
        fo = fs / M
        fa = abs(((p.f2 + fo / 2) % fo) - fo / 2)            # folded frequency
        f_in, P_in = tone_spectrum(x[:8192 * M], fs)
        f_out, P_out = tone_spectrum(y, fo)
        pi_ = self.plot("inp")
        pi_.set_xlim(0, fs / 2)
        pi_.band("keep", 0, fo / 2, color=GREEN, alpha=0.08)
        pi_.text("keepl", fo / 4, 10, "survives", color=GREEN, anchor=(0.5, 0), size=8.5)
        if p.filt:
            ff, Hh = fir_db(h, 2048)
            pi_.line("h", ff * fs, Hh, color=ORANGE, width=1.6, style="--", name="filter response")
        pi_.line("P", f_in, P_in, color=NAVY, width=1.6, name="input", fill=-120, fill_alpha=0.10)
        pi_.vline("nyq", fo / 2, color=RED, style=":", width=1.2, label="new Nyquist", label_pos=0.08)
        po = self.plot("out")
        po.set_xlim(0, fo / 2)
        po.line("P", f_out, P_out, color=NAVY, width=1.6, name="output", fill=-120, fill_alpha=0.10)
        po.vline("al", fa, color=RED, style="--", width=1.2, label="alias", label_pos=0.08)
        rej = level_at(f_out, P_out, fa, 0.06) - level_at(f_out, P_out, 1.0, 0.06)
        if abs(fa - 1.0) < 0.1:
            rej = 20 * np.log10(abs(self._alias_gain(h, p.f2, M) * a2) + 1e-12)
        # time view: the interferer alone, its kept samples and the alias they trace
        pt = self.plot("time")
        tt = np.arange(0, 2.0, 1 / (fs * 8))
        pt.line("c", tt, a2 * np.sin(2 * np.pi * p.f2 * tt + 0.4), color=GRAY, width=1.0,
                name="interferer (input)")
        ts = np.arange(0, 2.0, M / fs)
        ys = a2 * np.sin(2 * np.pi * p.f2 * ts + 0.4)
        sign = 1 if ((p.f2 % fo) < fo / 2) else -1
        pt.line("al", tt, a2 * np.sin(2 * np.pi * sign * fa * tt + 0.4), color=RED, width=2.0,
                style="--", name=f"the samples trace {fa:.2f} kHz")
        pt.stems("s", ts, ys, color=RED, size=7, name="samples kept")
        pt.set_xlim(0, 2.0)
        self.readout(alias=fa, rej=float(np.clip(rej, -150, 20)), rate=fo,
                     cost=f"{len(h)} (direct: {len(h) * M})" if p.filt else "0")

    @staticmethod
    def _alias_gain(h, f, M, fs=48.0):
        return abs(np.sum(h * np.exp(-2j * np.pi * f / fs * np.arange(len(h)))))

    def interpolate(self, p):
        L, fs = int(p.M), self.fs
        fi = fs / L                                          # input rate
        n = 4096
        t = np.arange(n) / fi
        f_b = round(0.3 * fi, 2)
        x = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * f_b * t + 0.7)
        up = np.zeros(n * L)
        up[::L] = x
        if p.filt:
            h = L * self.taps_for(L, int(p.taps))
            y = mr.polyphase_interpolate(x, h, L)
        else:
            h = np.array([1.0])
            y = up * L
        f_in, P_in = tone_spectrum(up[:4096 * L // 2 * 2], fs)
        y = y[200:200 + 4096 * L // 2 * 2]
        f_out, P_out = tone_spectrum(y, fs)
        P_in = P_in + 20 * np.log10(L)
        pi_ = self.plot("inp")
        pi_.set_xlim(0, fs / 2)
        pi_.band("keep", 0, fi / 2, color=GREEN, alpha=0.08)
        pi_.text("keepl", fi / 4, 10, "original band", color=GREEN, anchor=(0.5, 0), size=8.5)
        if p.filt:
            ff, Hh = fir_db(h / L, 2048)
            pi_.line("h", ff * fs, Hh, color=ORANGE, width=1.6, style="--", name="filter response")
        pi_.line("P", f_in, P_in, color=NAVY, width=1.6, name="zero-stuffed")
        for k in range(1, L):
            pi_.vline(f"img{k}", k * fi, color=RED, style=":", width=1.0,
                      label="images" if k == 1 else None, label_pos=0.08)
        pi_.vline("nyq", fi / 2, color=GRAY, style=":", width=1.0)
        po = self.plot("out")
        po.set_xlim(0, fs / 2)
        po.line("P", f_out, P_out, color=NAVY, width=1.6, name="interpolated output")
        po.vline("nyq", fi / 2, color=GRAY, style=":", width=1.0)
        img = []
        for k in range(1, L + 1):
            for f0 in (1.0, f_b):
                for fimg in (k * fi - f0, k * fi + f0):
                    if 0 < fimg < fs / 2:
                        img.append(level_at(f_out, P_out, fimg, 0.08))
        rej = (max(img) if img else -150) - level_at(f_out, P_out, 1.0, 0.06)
        pt = self.plot("time")
        tt = np.arange(len(y)) / fs
        k = tt <= 2.0
        tu = np.arange(len(up)) / fs
        ku = tu <= 2.0
        pt.stems("s", tu[ku][::L], up[ku][::L], color=RED, size=7, name="input samples")
        pt.scatter("z", tu[ku][np.arange(len(tu[ku])) % L != 0], 0 * tu[ku][np.arange(len(tu[ku])) % L != 0],
                   color=GRAY, size=5, name="inserted zeros")
        if p.filt:
            d = (len(h) - 1) / 2 - 200
            pt.line("y", tt[k] - d / fs, y[k], color=NAVY, width=2.0, name="filtered output")
        pt.set_xlim(0, 2.0)
        self.readout(alias=None, rej=float(np.clip(rej, -150, 20)), rate=fs,
                     cost=f"{int(np.ceil(len(h) / L))} (direct: {len(h)})" if p.filt else "0")

    def story(self, p):
        M = int(p.M)
        if p.mode.startswith("Dec"):
            fo = self.fs / M
            fa = self.r.get("alias", 0) or 0
            s = (f"<p>Keeping one sample in {v(M, 'd')} lowers the rate to {v(fo, '.1f', 'kS/s')}, so "
                 f"the new Nyquist frequency is {v(fo / 2, '.1f', 'kHz')}. Anything above it does not "
                 f"disappear: it <b>folds</b>. Your {v(p.f2, '.2f', 'kHz')} interferer lands at "
                 f"{v(fa, '.2f', 'kHz')}, and the red samples (bottom) trace that slower sine "
                 f"perfectly. After decimation nothing can tell them apart from a real tone.</p>")
            if p.filt:
                s += (f"<p>The low-pass filter removes the interferer <i>before</i> the samples are "
                      f"thrown away. In polyphase form it computes only the outputs you keep: "
                      f"{v(p.taps, 'd')} multiplies per output instead of {v(p.taps * M, 'd')}.</p>")
            else:
                s += ("<p>" + bad("No filter:") + " aliasing is permanent. Switch the filter on.</p>")
        else:
            s = (f"<p>To raise the rate by {v(M, 'd')}, insert {v(M - 1, 'd')} zeros between samples "
                 f"(grey dots). The spectrum does not change shape, it just repeats: "
                 f"{v(M - 1, 'd')} <b>images</b> appear around multiples of the old sample rate.</p>"
                 "<p>A low-pass filter with gain L keeps the original band and removes the images, "
                 "filling in the zeros with the right values. A pulse-shaping filter at several "
                 "samples per symbol is exactly this operation.</p>")
        return "<h3>Changing the sample rate</h3>" + s + keybox(
            "Filter before you decimate, filter after you interpolate, and run the filter at the "
            "low rate (polyphase).")


# =============================================================================== 5. channelizer
class Channelizer(Experiment):
    title = "Polyphase channelizer"
    blurb = "One prototype filter and one FFT split a wideband capture into eight channels."
    book = "sec:ch06:filterbanks"
    controls = [
        Heading("Prototype filter"),
        IntSlider("T", "Taps per branch", 1, 16, 2,
                  help="The prototype has 8 × this many taps. 1 tap per branch = a plain FFT"),
        Choice("win", "Window", ["Rectangular", "Hamming", "Kaiser (β = 9)"], "Hamming"),
        Slider("bw", "Prototype bandwidth", 0.6, 1.5, 1.0, step=0.01, unit="ch.",
               help="Two-sided bandwidth of each channel's filter in channel spacings"),
        Heading("Input"),
        Slider("lvl", "Strong signal in channel 5", -10, 40, 10, step=1, unit="dB"),
        IntSlider("sel", "Channel to inspect", 0, 7, 4),
    ]
    plots = [
        SpectrumPlot("inp", "Wideband input (8 channels)", x="frequency (channel spacings)",
                     y="level (dB)", xlim=(-0.5, 7.5), ylim=(-90, 50), legend=None),
        SpectrumPlot("proto", "One channel's filter", x="offset from channel centre (spacings)",
                     y="gain (dB)", xlim=(-3, 3), ylim=(-120, 5), legend=None),
        BarPlot("bars", "Power in each output channel", x="channel", y="level (dB)",
                ylim=(-90, 50)),
        SpectrumPlot("sel", "Inspected channel output", x="frequency (output samples)",
                     y="level (dB)", xlim=(-0.5, 0.5), ylim=(-90, 50), legend=None),
    ]
    layout = [["inp", "inp"], ["proto", "bars"], ["sel", "bars"]]
    row_stretch = [4, 3, 3]
    readouts = [
        Readout("leak", "Leakage into channel 4", "dB rel. ch. 5", ".1f", good=lambda x: x < -50),
        Readout("acr", "Rejection at the next channel", "dB", ".1f"),
        Readout("cost", "Multiplies per input sample", "", ".1f"),
        Readout("brute", "…with 8 separate DDCs", "", ".0f"),
    ]
    challenges = [
        Challenge("With the channel-5 signal 30 dB above the rest, keep its leakage into the empty "
                  "channel 4 below −55 dB.",
                  lambda s: s.p.lvl >= 30 and s.r.leak < -55,
                  hint="Window choice sets the sidelobes, taps per branch the steepness."),
        Challenge("Find the cheapest bank (fewest taps per branch) that keeps the leakage below −40 dB.",
                  lambda s: s.r.leak < -40 and s.p.T <= 4,
                  hint="Four taps per branch can do it, if the window and bandwidth are right."),
        Challenge("See why a plain FFT is a poor channelizer: 1 tap per branch, rectangular window, "
                  "leakage above −25 dB.",
                  lambda s: s.p.T == 1 and s.p.win == "Rectangular" and s.r.leak > -25),
    ]
    M = 8
    nsym = 3000

    def setup(self):
        rng = np.random.default_rng(6)
        M, sps = self.M, 2 * self.M
        t = cl.rrc_taps(0.25, sps, 10)
        comps = {}
        n = None
        for k, lev in {1: 0, 2: -6, 3: -20, 5: 0, 6: -12}.items():
            s = (rng.choice([-1, 1], self.nsym) + 1j * rng.choice([-1, 1], self.nsym)) / np.sqrt(2)
            b = cl.shape(s, t, sps)
            if n is None:
                n = np.arange(len(b))
            comps[k] = 10 ** (lev / 20) * b[:len(n)] * np.exp(2j * np.pi * k * n / M)
        noise = 10 ** (-60 / 20) * (rng.standard_normal(len(n)) + 1j * rng.standard_normal(len(n))) / np.sqrt(2)
        self.base = sum(c for k, c in comps.items() if k != 5) + noise
        self.c5 = comps[5]
        self.ref_pow = np.mean(np.abs(comps[1]) ** 2)
        self.p1max = float(np.max(cl.welch_psd(comps[1], 1.0, 2048)[1]))

    @staticmethod
    @lru_cache(maxsize=128)
    def proto(M, T, win, bw):
        N = M * T
        w = {"Rectangular": "boxcar", "Hamming": "hamming", "Kaiser (β = 9)": ("kaiser", 9.0)}[win]
        if T == 1 and win == "Rectangular":
            return np.ones(M) / M
        return sg.firwin(N, min(bw / M, 0.999), window=w, fs=2.0) if N > 1 else np.ones(1)

    def update(self, p):
        M = self.M
        g5 = 10 ** (p.lvl / 20)
        x = self.base + g5 * self.c5
        h = self.proto(M, int(p.T), p.win, round(p.bw, 3))
        Y = mr.pfb_channelizer(x, h / np.sum(h), M)
        Y = Y[4 * p.T + 4:]
        pw = 10 * np.log10(np.mean(np.abs(Y) ** 2, axis=0) / self.ref_pow + 1e-15)
        # input spectrum (channel-spacing units: f * M), shown 0..8
        f, P = cl.welch_psd(x, 1.0, 2048)
        ff = (f * M) % M
        o = np.argsort(ff)
        Pdb = P - self.p1max
        pi_ = self.plot("inp")
        ffo = ff[o]
        ffo = np.where(ffo > M - 0.5, ffo - M, ffo)
        o2 = np.argsort(ffo)
        pi_.line("P", ffo[o2], Pdb[o][o2], color=NAVY, width=1.4, fill=-90, fill_alpha=0.08)
        for c in range(M):
            pi_.vline(f"g{c}", c + 0.5, color=GRAY, style=":", width=0.8)
            pi_.text(f"n{c}", c, 47, str(c), color=RED if c == p.sel else GRAY, anchor=(0.5, 0),
                     size=10 if c == p.sel else 8.5, bold=c == p.sel)
        pi_.band("selb", p.sel - 0.5, p.sel + 0.5, color=ORANGE, alpha=0.10)
        # prototype response in channel-spacing units
        nf = 1 << 14
        Hh = np.abs(np.fft.fft(h, nf))
        Hh = 20 * np.log10(Hh / Hh.max() + 1e-12)
        fh = np.fft.fftfreq(nf) * M
        o = np.argsort(fh)
        pp = self.plot("proto")
        pp.band("own", -0.5, 0.5, color=GREEN, alpha=0.08)
        pp.band("nb1", 0.5, 1.5, color=RED, alpha=0.06)
        pp.band("nb2", -1.5, -0.5, color=RED, alpha=0.06)
        pp.line("H", fh[o], Hh[o], color=NAVY, width=2.0)
        acr = -float(np.max(Hh[(np.abs(fh) >= 0.75) & (np.abs(fh) <= 1.25)]))
        pp.hline("acr", -acr, color=ORANGE, style=":", width=1.2, label=f"next channel: −{acr:.0f} dB",
                 label_pos=0.65)
        # bars
        pb = self.plot("bars")
        truth = {1: 0, 2: -6, 3: -20, 5: p.lvl, 6: -12}
        cols = [ORANGE if c == p.sel else (NAVY if c in truth else GRAY) for c in range(M)]
        pb.bars("b", np.arange(M), np.maximum(pw, -88), width=0.62, colors=cols, base=-90)
        pb.scatter("t", list(truth.keys()), list(truth.values()), color=GREEN, size=11, symbol="d")
        pb.set_xticks([(c, str(c)) for c in range(M)])
        pb.set_xlim(-0.6, M - 0.4)
        for c in range(M):
            pb.text(f"v{c}", c, max(pw[c], -88), f"{int(round(pw[c])):d}".replace("-", "−"),
                    anchor=(0.5, 1.05), size=8)
        # selected channel output spectrum
        fy, Py = cl.welch_psd(Y[:, p.sel], 1.0, 256)
        ps = self.plot("sel")
        ref1 = float(np.max(cl.welch_psd(Y[:, 1], 1.0, 256)[1]))
        ps.line("P", fy, Py - ref1, color=ORANGE, width=1.8, fill=-90, fill_alpha=0.10)
        ps.set_title(f"Channel {p.sel} output, at 1/8 of the input rate")
        leak = pw[4] - pw[5]
        T = int(p.T)
        self.readout(leak=float(leak), acr=acr, cost=T + 0.5 * np.log2(M),
                     brute=float(M * (T + 1)))

    def story(self, p):
        leak = self.r.get("leak", 0)
        s = (f"<p>Eight channels sit side by side in one capture. A brute-force receiver would run "
             f"eight mixers and eight filters at the full rate. The <b>polyphase filter bank</b> "
             f"splits one prototype low-pass into 8 branches of {v(p.T, 'd')} taps, runs them at the "
             f"output rate, and lets one 8-point FFT do all the mixing at once.</p>")
        if p.T == 1 and p.win == "Rectangular":
            s += ("<p>With one tap per branch and no window this is just a block FFT. Each bin's "
                  "response is a sinc with −13 dB sidelobes, so the strong signal in channel 5 "
                  f"spills into its neighbours: {v(leak, '.0f', 'dB')} into channel 4.</p>")
        else:
            s += (f"<p>The prototype (bottom left) decides everything: its rejection at the next "
                  f"channel centre is {v(self.r.get('acr', 0), '.0f', 'dB')}, and the strong signal "
                  f"leaks {v(leak, '.0f', 'dB')} into the empty channel 4. More taps per branch make "
                  f"the skirt steeper; the window sets how deep the sidelobes go.</p>")
        if leak < -50:
            s += ("<p>The last −57 dB or so is not the bank's fault: it is the neighbour's own "
                  "transmitted spectrum spilling into channel 4. No receive filter can remove it.</p>")
        return "<h3>Eight receivers for the price of one</h3>" + s + keybox(
            "Channelizer = polyphase filter (M branches × T taps) + M-point FFT: about T + ½·log₂M "
            "multiplies per input sample for all M channels.")


# =============================================================================== 6. CIC
class CIC(Experiment):
    title = "CIC decimator and compensator"
    blurb = "Hogenauer's filter: decimation with nothing but adders, and the droop it leaves behind."
    book = "sec:ch06:cic"
    controls = [
        Heading("CIC"),
        IntSlider("R", "Decimation R", 2, 64, 16),
        IntSlider("N", "Stages N", 1, 6, 4),
        Choice("D", "Differential delay D", ["1", "2"]),
        Slider("fpb", "Passband edge", 0.05, 0.25, 0.20, step=0.005, unit="× f_out",
               help="Highest wanted frequency, as a fraction of the CIC output rate"),
        Heading("Compensator"),
        Toggle("comp", "Droop compensator", False),
        IntSlider("ctaps", "Compensator length", 7, 63, 15, step=2, unit="taps",
                  enabled_if=lambda p: p.comp),
        Heading("Fixed point"),
        IntSlider("bin", "Input word length", 8, 16, 12, unit="bits"),
        IntSlider("width", "Register width", 8, 64, 28, unit="bits",
                  help="Every integrator and comb register, two's complement, wrapping on overflow"),
    ]
    plots = [
        SpectrumPlot("resp", "CIC response: nulls on the bands that alias", x="frequency (× f_out)",
                     y="gain (dB)", xlim=(0, 4), ylim=(-140, 8), legend="tr"),
        Plot("pb", "Passband droop", x="frequency (× f_out)", y="gain (dB)", xlim=(0, 0.3),
             ylim=(-5, 1.5), legend="bl"),
        Plot("reg", "Integer CIC at work", x="output sample", y="value (normalised)",
             ylim=(-1.6, 2.3), legend="tl", legend_cols=2),
    ]
    layout = [["resp", "resp"], ["pb", "reg"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("droop", "Droop at the passband edge", "dB", ".2f"),
        Readout("rej", "Worst alias rejection", "dB", ".1f", good=lambda x: x >= 80),
        Readout("need", "Register width needed", "bits", "int"),
        Readout("exact", "Integer output", "", None),
    ]
    challenges = [
        Challenge("The book's case (R = 16, N = 4, passband 0.2): flatten the passband to within "
                  "0.1 dB with the shortest compensator you can (≤ 31 taps).",
                  lambda s: s.p.R == 16 and s.p.N == 4 and s.p.comp and s.p.fpb >= 0.199
                  and s.r.ripple is not None and s.r.ripple < 0.1 and s.p.ctaps <= 31),
        Challenge("Reach 80 dB of alias rejection for a passband of 0.1 with the fewest stages "
                  "(no more than 5).",
                  lambda s: abs(s.p.fpb - 0.1) < 0.003 and s.r.rej >= 80 and s.p.N <= 5
                  and s.p.D == "1"),
        Challenge("Hogenauer's trick: find the narrowest register that is still bit-exact for "
                  "R = 32, N = 3, D = 1 and 12-bit input.",
                  lambda s: s.p.R == 32 and s.p.N == 3 and s.p.D == "1" and s.p.bin == 12
                  and s.p.width == 27 and s.r.exact == "bit-exact",
                  hint="B_out = B_in + N·log₂(R·D). Narrower and the output wraps; wider and you "
                       "waste flip-flops."),
    ]

    def update(self, p):
        R, N, D = int(p.R), int(p.N), int(p.D)
        f = np.linspace(1e-6, 4, 4001)                         # in units of f_out
        H = mr.cic_response(f / R, R, N, D)
        Hdb = 20 * np.log10(H + 1e-14)
        Hdb[f > R / 2] = np.nan                                # beyond the input Nyquist
        pr = self.plot("resp")
        for k in range(1, 5):
            pr.band(f"al{k}", k - p.fpb, k + p.fpb, color=RED, alpha=0.08)
        pr.band("pass", 0, p.fpb, color=GREEN, alpha=0.10)
        pr.line("H", f, Hdb, color=NAVY, width=2.0, name=f"CIC R = {R}, N = {N}, D = {D}")
        ks = np.arange(1, R // 2 + 1)
        rej_each = []
        for k in ks:
            ff = np.linspace(max(k - p.fpb, 1e-6), min(k + p.fpb, R / 2), 60)
            rej_each.append(-20 * np.log10(np.max(mr.cic_response(ff / R, R, N, D)) + 1e-15))
        rej = float(min(rej_each)) if rej_each else 0.0
        pr.hline("rej", -rej, color=RED, style=":", width=1.2, label=f"worst alias: −{rej:.0f} dB",
                 label_pos=0.62)
        # passband and compensation
        fpz = np.linspace(1e-6, 0.3, 400)
        Hc = mr.cic_response(fpz / R, R, N, D)
        pp = self.plot("pb")
        pp.band("pass", 0, p.fpb, color=GREEN, alpha=0.08)
        pp.line("cic", fpz, 20 * np.log10(Hc), color=GRAY, width=1.8, name="CIC alone")
        droop = -20 * np.log10(mr.cic_response(p.fpb / R, R, N, D))
        ripple = None
        if p.comp:
            hc = mr.cic_comp_taps(R, N, int(p.ctaps), p.fpb, D=D)
            _, Hk = sg.freqz(hc, worN=fpz, fs=1.0)
            tot = 20 * np.log10(np.abs(Hc * Hk) + 1e-12)
            pp.line("comp", fpz, 20 * np.log10(np.abs(Hk) + 1e-12), color=GREEN, width=1.4,
                    style="--", name="compensator")
            pp.line("tot", fpz, tot, color=NAVY, width=2.4, name="CIC × compensator")
            m = fpz <= p.fpb
            ripple = float(np.ptp(tot[m]))
        # integer simulation: DC + tone near full scale
        B = int(p.bin)
        need = mr.cic_bits(B, R, N, D)
        nout = 160
        n = np.arange(R * (nout + 8))
        full = 2 ** (B - 1) - 1
        xin = np.round(full * (0.45 + 0.5 * np.sin(2 * np.pi * 0.07 / R * n * 1.0 + 0.3))).astype(np.int64)
        ref = mr.cic_decimate(xin, R, N, D)
        out = mr.cic_decimate(xin, R, N, D, width=int(p.width))
        g = float((R * D) ** N) * full
        ok = np.array_equal(ref, out)
        # integrator 1 register contents (wrapped) at the output instants
        w = int(p.width)
        mod = 2 ** min(w, 62)
        integ = xin.copy()
        for _ in range(N):                                   # the last integrator grows fastest
            integ = (np.cumsum(integ) + mod // 2) % mod - mod // 2 if w < 63 else np.cumsum(integ)
        integ = integ[R - 1::R][:nout]
        wrapped = integ / (mod / 2) if w < 63 else integ / max(abs(integ).max(), 1)
        k = np.arange(nout)
        prg = self.plot("reg")
        prg.line("int", k, wrapped, color=GRAY, width=1.2, name=f"integrator {N} (wraps)")
        prg.line("ref", k, ref[:nout] / g, color=GREEN, width=4.0, alpha=0.5, name="exact output")
        prg.line("out", k, np.clip(out[:nout] / g, -1.55, 2.2), color=NAVY if ok else RED, width=1.8,
                 name=f"{w}-bit registers")
        prg.set_xlim(0, nout)
        wraps = int(np.sum(np.abs(np.diff(wrapped)) > 1.0))
        self.readout(droop=droop, rej=rej, need=need, ripple=ripple, wraps=wraps,
                     exact="bit-exact" if ok else "WRAPPED: wrong")

    def story(self, p):
        R, N, D = int(p.R), int(p.N), int(p.D)
        r = self.r
        s = (f"<p>The CIC is {v(N, 'd')} accumulators at the input rate, a ↓{v(R, 'd')}, and "
             f"{v(N, 'd')} differencers at the output rate: no multipliers at all. Its response is "
             f"(sin πfRD / RD sin πf)ᴺ, with nulls exactly where the bands that would alias onto DC sit "
             f"(red). Deep at the centres, only {v(r.get('rej', 0), '.0f', 'dB')} at their worst "
             f"edges for a passband of {v(p.fpb, '.2f')}·f_out.</p>")
        s += (f"<p>The price is <b>droop</b>: {v(r.get('droop', 0), '.2f', 'dB')} at the passband "
              f"edge. " + ("A short FIR at the low rate with an inverse-sinc passband flattens it "
                           f"to {v(r.get('ripple') or 0, '.2f', 'dB')} of ripple." if p.comp else
                           "Switch on the compensator.") + "</p>")
        need = r.get("need", 0)
        if r.get("exact") == "bit-exact":
            wr = r.get("wraps", 0)
            s += ((f"<p>The last integrator overflowed and wrapped {v(wr, 'd')} times in this window "
                   f"(grey), yet the output is " if wr else
                   "<p>The integrators would overflow given time (make the registers narrower to see "
                   "them wrap), yet the output stays ")
                  + f"{good('exactly right')}: two's-complement arithmetic is modular, and the combs "
                  f"undo every wrap, provided each register has at least B_in + N·log₂(RD) = "
                  f"{v(need, 'd')} bits.</p>")
        else:
            s += (f"<p>{bad('Too narrow.')} The true output needs {v(need, 'd')} bits; with "
                  f"{v(p.width, 'd')} the modular arithmetic can no longer represent it and the "
                  f"result wraps.</p>")
        return "<h3>Decimation with adders</h3>" + s + keybox(
            "A CIC does the big, early decimation for free; a compensating FIR at the low rate fixes "
            "its droop and does the sharp filtering.")


# =============================================================================== 7. NCO
FS_NCO = 245.76e6


class NCO(Experiment):
    title = "NCO: phase bits, spurs, dither"
    blurb = "A phase accumulator and a sine table: where the spurs come from and how to hide them."
    book = "sec:ch06:nco"
    controls = [
        Slider("f", "Tuning frequency", 0.001, 0.499, 0.1235, step=0.0001, unit="× f_s"),
        IntSlider("acc", "Accumulator width", 8, 32, 32, unit="bits",
                  help="Sets the frequency resolution f_s / 2^B"),
        IntSlider("P", "Phase bits to the table", 4, 16, 10, unit="bits",
                  help="Only the top P accumulator bits address the sine table"),
        IntSlider("A", "Table amplitude bits", 4, 18, 14, unit="bits"),
        Toggle("dith", "Phase dither", False,
               help="Add a random value below one phase LSB before truncation"),
    ]
    plots = [
        SpectrumPlot("spec", "NCO output spectrum", x="frequency (cycles/sample)", y="level (dBc)",
                     xlim=(-0.5, 0.5), ylim=(-160, 8), legend=None),
        Plot("perr", "Phase-truncation error (first 300 samples)", x="sample", y="error (table LSBs)",
             xlim=(0, 300), ylim=(-1.2, 0.3), legend=None),
        Plot("rule", "Spur-free range vs phase bits", x="phase bits P", y="SFDR (dBc)",
             xlim=(3.5, 16.5), ylim=(0, 120), legend="tl"),
    ]
    layout = [["spec", "spec"], ["perr", "rule"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("sfdr", "Spur-free dynamic range", "dBc", ".1f", good=lambda x: x >= 72),
        Readout("res", "Frequency resolution", "Hz", lambda x: f"{x:.3g}"),
        Readout("ferr", "Frequency error", "Hz", lambda x: f"{x:.3g}"),
        Readout("rom", "Table size", "kbit", ".1f"),
    ]
    challenges = [
        Challenge("The book's 75 dB receiver: at least 72 dBc SFDR with a table of 12 phase bits "
                  "and 14 amplitude bits or fewer, no dither.",
                  lambda s: s.r.sfdr >= 72 and s.p.P <= 12 and s.p.A <= 14 and not s.p.dith),
        Challenge("Keep 10 phase bits but use dither to push every spur below −75 dBc.",
                  lambda s: s.p.P == 10 and s.p.dith and s.r.sfdr >= 75),
        Challenge("Find a tuning frequency that an 8-bit-phase NCO makes with no truncation spurs "
                  "at all (SFDR above 80 dBc, no dither).",
                  lambda s: s.p.P == 8 and not s.p.dith and s.r.sfdr > 80,
                  hint="Truncation is harmless when the low accumulator bits of the tuning word are zero."),
        Challenge("Tune to 0.1 cycles/sample with less than 100 Hz of error at 245.76 MHz, using the "
                  "shortest accumulator that can do it.",
                  lambda s: abs(s.p.f - 0.1) < 5e-5 and s.r.ferr < 100 and s.p.acc <= 19,
                  hint="The error depends on the fractional part of 0.1 × 2^B, not just the resolution."),
    ]
    n = 1 << 14

    @lru_cache(maxsize=64)
    def rule_curve(self, f, acc, A):
        out = []
        for P in range(4, 17):
            z, *_ = mr.nco(self.n // 2, f, acc, min(P, acc), A)
            out.append(mr.nco_sfdr(z)[0])
        return np.array(out)

    def update(self, p):
        P = min(int(p.P), int(p.acc))
        z, ftw, fa = mr.nco(self.n, p.f, int(p.acc), P, int(p.A), dither=p.dith,
                            rng=self.rng if p.dith else None)
        sfdr, fsp, f, Pdb = mr.nco_sfdr(z)
        o = np.argsort(f)
        ps = self.plot("spec")
        ps.line("P", f[o], Pdb[o], color=NAVY, width=1.0)
        ps.hline("sf", -sfdr, color=RED, style="--", width=1.2, label=f"worst spur −{sfdr:.1f} dBc",
                 label_pos=0.03)
        ps.scatter("sp", [fsp], [-sfdr], color=RED, size=11, symbol="d")
        ps.vline("c", fa, color=GREEN, style=":", width=1.0)
        # phase error in table LSBs
        B = int(p.acc)
        k = np.arange(300, dtype=np.int64)
        accv = (k * ftw) % (1 << B)
        exact = accv / 2.0 ** (B - P)
        trunc = np.floor(exact) if not p.dith else np.floor(exact + self.rng.random(300))
        pe = self.plot("perr")
        pe.line("e", k, trunc - exact, color=NAVY, width=1.4, step=False)
        pe.set_ylim(-1.2 if not p.dith else -1.2, 0.3 if not p.dith else 1.2)
        # 6 dB per bit
        curve = self.rule_curve(round(p.f, 6), B, int(p.A))
        pr = self.plot("rule")
        Ps = np.arange(4, 17)
        pr.line("th", Ps, 6.02 * Ps, color=GRAY, width=1.4, style="--", name="6.02 · P")
        pr.line("cur", Ps, curve, color=NAVY, width=1.8, name="this tuning word")
        pr.scatter("cur#p", Ps, curve, color=NAVY, size=6)
        pr.hline("amp", 6.02 * p.A + 1.76, color=ORANGE, style=":", width=1.2,
                 label="amplitude-bit floor", label_pos=0.6)
        pr.scatter("now", [P], [sfdr], color=RED, size=14, symbol="d", name="now")
        res = FS_NCO / 2 ** B
        self.readout(sfdr=sfdr, res=res, ferr=abs(fa - p.f) * FS_NCO,
                     rom=2 ** P * int(p.A) / 1024)

    def story(self, p):
        r = self.r
        s = (f"<p>Every clock the accumulator adds the tuning word; its top {v(p.P, 'd')} bits "
             f"look up the sine. Throwing away the low bits is a phase error that is "
             f"<i>periodic</i> (the sawtooth, bottom left), and a periodic error makes discrete "
             f"<b>spurs</b>, here {v(r.get('sfdr', 0), '.1f', 'dB')} below the carrier. The rule of thumb "
             f"is 6.02 dB per phase bit (bottom right).</p>")
        if p.dith:
            s += ("<p>With <b>dither</b> a random amount under one LSB is added before truncation, "
                  "so the error is no longer periodic: the spurs dissolve into a flat noise floor. "
                  "The total error power is a little higher, but no single line can mask a weak signal.</p>")
        s += (f"<p>The {v(p.acc, 'd')}-bit accumulator gives a resolution of "
              f"{v(r.get('res', 0), '.3g', 'Hz')} at 245.76 MHz, and frequency hops are "
              f"phase-continuous and instant: the reason NCOs drive every DDC, frequency hopper and "
              f"chirp generator.</p>")
        return "<h3>A clock that counts phase</h3>" + s + keybox(
            "f₀ = F·f_s / 2^B. Spurs ≈ 6.02·P dBc from phase truncation; dither trades them for noise.")


# =============================================================================== 8. CORDIC
class CORDIC(Experiment):
    title = "CORDIC: rotation by shifts"
    blurb = "Rotate a vector, or measure its angle, with nothing but shifts and adds."
    book = "sec:ch06:cordic"
    controls = [
        Choice("mode", "Mode", ["Rotation (make cos, sin)", "Vectoring (measure the angle)"],
               style="menu"),
        Slider("ang", "Target angle", -180, 180, 57, step=0.5, unit="°"),
        IntSlider("n", "Iterations", 1, 24, 6),
        IntSlider("bits", "Datapath width", 10, 32, 24, unit="bits"),
        Toggle("pre", "±90° pre-rotation", True,
               help="Swap and negate first, so that angles beyond ±99.9° converge"),
    ]
    plots = [
        Plot("traj", "The vector's path (each step: ± arctan 2⁻ⁱ)", x="x", y="y",
             xlim=(-1.9, 1.9), ylim=(-1.9, 1.9), aspect=True, legend=None),
        Plot("res", "Remaining angle after each iteration", x="iteration i", y="|residual angle| (rad)",
             logy=True, ylim=(1e-8, 4), legend="bl"),
        Plot("acc", "Worst error over 2000 random angles", x="iterations", y="max error (rad)",
             logy=True, xlim=(0.5, 24.5), ylim=(1e-8, 4), legend="tr"),
    ]
    layout = [["traj", "res"], ["traj", "acc"]]
    col_stretch = [5, 6]
    readouts = [
        Readout("err", "Error for this angle", "rad", "sci"),
        Readout("maxerr", "Worst error (any angle)", "rad", "sci", good=lambda x: x < 3.1e-5),
        Readout("gain", "Gain 1/K", "", ".5f"),
        Readout("adders", "Adders (pipelined)", "", "int"),
    ]
    challenges = [
        Challenge("16-bit accuracy (worst error below 3.1×10⁻⁵ rad) on a datapath no wider than "
                  "22 bits, with as few iterations as possible (17 or fewer).",
                  lambda s: s.r.maxerr < 3.1e-5 and s.p.n <= 17 and s.p.bits <= 22,
                  hint="About one bit per iteration, plus log₂(n) + 2 guard bits in the datapath."),
        Challenge("Watch an 18-bit datapath stall: run at least 20 iterations and see the error "
                  "stay above 10⁻⁴ rad.",
                  lambda s: s.p.bits == 18 and s.p.n >= 20 and s.r.maxerr > 1e-4),
        Challenge("Switch off the pre-rotation and find the largest angle that still converges "
                  "(error below 0.01 rad, |angle| ≥ 95°).",
                  lambda s: not s.p.pre and abs(s.p.ang) >= 95 and s.r.err < 0.01),
    ]

    @staticmethod
    @lru_cache(maxsize=64)
    def worst_curve(bits, pre):
        th = np.random.default_rng(2).uniform(-np.pi, np.pi, 2000)
        out = []
        for n in range(1, 25):
            c, s = mr.cordic(th, n, bits, prerotate=pre)
            out.append(np.max(np.abs(np.angle(np.exp(1j * (np.arctan2(s, c) - th))))))
        return np.array(out)

    def update(self, p):
        n = int(p.n)
        th = np.deg2rad(p.ang)
        vect = p.mode.startswith("Vector")
        G = mr.cordic_gain(n)
        # trajectory (floating point), with the optional pre-rotation shown as the first hop
        if vect:
            x0, y0 = 0.8 * np.cos(th), 0.8 * np.sin(th)
            pre = []
            if p.pre and x0 < 0:
                pre = [(x0, y0)]
                x0, y0 = -x0, -y0
            X, Y, Z = mr.cordic_trajectory(0.0, n, "vector", x0, y0)
            est = Z[-1] + (np.pi * np.sign(th) if pre else 0.0)
            err = abs(np.angle(np.exp(1j * (est - th))))
            resid = np.abs(np.arctan2(Y, X))
        else:
            pre = []
            t2 = th
            if p.pre and abs(th) > np.pi / 2:
                pre = [(1.0, 0.0)]
                t2 = th - np.pi * np.sign(th)
            X, Y, Z = mr.cordic_trajectory(t2, n, "rotate", 1.0, 0.0)
            if pre:
                X, Y = -X, -Y
                X = np.r_[1.0, X]
                Y = np.r_[0.0, Y]
                Z = np.r_[Z[0], Z]
            c, s_ = mr.cordic(np.array([th]), n, int(p.bits), prerotate=p.pre)
            err = float(abs(np.angle(np.exp(1j * (np.arctan2(s_[0], c[0]) - th)))))
            resid = np.abs(Z)
        pt = self.plot("traj")
        a = np.linspace(0, 2 * np.pi, 361)
        pt.line("c1", np.cos(a), np.sin(a), color=GRAY, width=1.0, style=":")
        pt.line("cg", G * np.cos(a), G * np.sin(a), color=GREEN, width=1.0, style="--")
        R = G * (0.8 if vect else 1.0)
        pt.line("tgt", [0, R * np.cos(th)], [0, R * np.sin(th)], color=RED, width=1.6, style="--")
        if vect:
            pt.line("xax", [0, R], [0, 0], color=RED, width=1.0, style=":")
        pt.line("path", X, Y, color=NAVY, width=1.8)
        pt.scatter("pts", X, Y, color=NAVY, size=7)
        pt.scatter("end", [X[-1]], [Y[-1]], color=ORANGE, size=12, symbol="d")
        for i in range(min(len(X), 7)):
            pt.text(f"i{i}", X[i] + 0.05, Y[i] + 0.04, str(i), color=NAVY, size=8, anchor=(0, 1))
        # residual angle per iteration
        pr = self.plot("res")
        it = np.arange(len(resid))
        pr.line("bound", np.arange(1, 26), np.arctan(2.0 ** -(np.arange(1, 26) - 1.0)), color=GRAY,
                width=1.2, style="--", name="bound arctan 2⁻⁽ⁱ⁻¹⁾")
        pr.line("r", it, np.maximum(resid, 1e-9), color=NAVY, width=1.6, name="this angle")
        pr.scatter("r#p", it, np.maximum(resid, 1e-9), color=NAVY, size=7)
        pr.set_xlim(-0.5, max(8, len(resid)) + 0.5)
        # worst-case curve
        wc = self.worst_curve(int(p.bits), bool(p.pre))
        wf = self.worst_curve(None, bool(p.pre))
        pa = self.plot("acc")
        k = np.arange(1, 25)
        pa.line("f", k, np.maximum(wf, 1e-9), color=GRAY, width=1.4, style="--", name="floating point")
        pa.line("b", k, np.maximum(wc, 1e-9), color=NAVY, width=2.0, name=f"{p.bits}-bit datapath")
        pa.hline("16", 3.1e-5, color=GREEN, style=":", width=1.2, label="16-bit accuracy", label_pos=0.05)
        pa.scatter("now", [n], [max(wc[n - 1], 1e-9)], color=RED, size=13, symbol="d")
        if vect:
            err_show = err
        else:
            err_show = err
        self.readout(err=float(err_show), maxerr=float(wc[n - 1]), gain=G, adders=3 * n)

    def story(self, p):
        n = int(p.n)
        r = self.r
        if p.mode.startswith("Vector"):
            s = ("<p><b>Vectoring mode</b> rotates the vector onto the x axis, one elementary angle "
                 "at a time, adding up the angles it used: the result is the vector's phase (and, "
                 "times K, its length). That is a phase detector, an FM demodulator and an AGC's "
                 "magnitude estimate in one pass.</p>")
        else:
            s = (f"<p><b>Rotation mode</b> turns (1, 0) towards {v(p.ang, '.1f', '°')} by steps of "
                 f"±arctan 2⁻ⁱ: 45°, 26.6°, 14.0°, 7.1°… Because the tangent of each step is a power "
                 f"of two, each step is two shifts and three adds. Start from (K, 0) instead and the "
                 f"vector ends exactly on (cos θ, sin θ): an NCO without a table.</p>")
        s += (f"<p>Every step stretches the vector by √(1 + 2⁻²ⁱ), whichever way it turns, so the "
              f"total gain is a constant, {v(r.get('gain', 1), '.5f')} after {v(n, 'd')} iterations "
              f"(green circle), removed by one multiply at the end.</p>"
              f"<p>Accuracy grows by about one bit per iteration until the {v(p.bits, 'd')}-bit "
              f"datapath runs out of resolution: worst error now {v(r.get('maxerr', 0), '.1e', 'rad')}.</p>")
        if not p.pre:
            s += ("<p>Without pre-rotation the elementary angles add up to only 99.9°: beyond that "
                  "the algorithm cannot reach the target.</p>")
        return "<h3>Volder's navigation computer</h3>" + s + keybox(
            "CORDIC: one bit of accuracy per iteration, three adders per iteration, no multipliers.")


# =============================================================================== 9. DDC
class DDC(Experiment):
    title = "A complete digital down-converter"
    blurb = "NCO → CIC → compensating FIR → matched filter: one channel out of a 16 MS/s capture."
    book = "sec:ch06:ddc"
    animate = True
    autoplay = True
    fps = 8
    controls = [
        Heading("NCO"),
        Slider("tune", "NCO frequency", 2.8, 3.6, 3.15, step=0.0005, unit="MHz",
               help="The wanted 250 kBd channel sits at +3.2 MHz"),
        Heading("Decimation chain"),
        IntSlider("N", "CIC stages (R = 8)", 1, 5, 4),
        Toggle("comp", "Droop-compensating FIR", True,
               help="Off: a plain 63-tap low-pass that ignores the CIC droop"),
        Heading("The neighbourhood"),
        Slider("nlev", "Neighbour above wanted", 0, 45, 10, step=1, unit="dB"),
        Slider("noff", "Neighbour offset (below)", 0.3, 1.2, 0.6, step=0.01, unit="MHz"),
        Toggle("persist", "Keep constellation history", True),
    ]
    plots = [
        SpectrumPlot("cap", "16 MS/s capture", x="frequency (MHz)", y="level (dB)",
                     xlim=(-8, 8), ylim=(-100, 10), legend=None),
        SpectrumPlot("cic", "After NCO + CIC (2 MS/s)", x="frequency (kHz)", y="level (dB)",
                     xlim=(-1000, 1000), ylim=(-100, 10), legend=None),
        SpectrumPlot("out", "After FIR ↓2 + matched filter (1 MS/s)", x="frequency (kHz)",
                     y="level (dB)", xlim=(-500, 500), ylim=(-100, 10), legend=None),
        ConstellationPlot("con", "Recovered QPSK symbols", lim=1.7),
    ]
    layout = [["cap", "con"], ["cic", "out"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("evm", "EVM", "dB", ".1f", good=lambda x: x < -25),
        Readout("cfo", "Residual frequency offset", "kHz", ".2f"),
        Readout("mps", "Multiplies per input sample", "", ".2f"),
        Readout("rates", "Rates", "MS/s", None),
    ]
    challenges = [
        Challenge("Tune the NCO onto the channel: residual offset below 1 kHz and EVM better than −25 dB.",
                  lambda s: abs(s.r.cfo) < 1 and s.r.evm < -25),
        Challenge("Make the neighbour 40 dB stronger than the wanted channel and still keep EVM "
                  "better than −25 dB.",
                  lambda s: s.p.nlev >= 40 and s.r.evm < -25 and abs(s.r.cfo) < 1),
        Challenge("Bring the 40 dB neighbour to within 0.4 MHz and find a chain that still gets EVM "
                  "better than −18 dB.",
                  lambda s: s.p.nlev >= 40 and s.p.noff <= 0.4 and s.r.evm < -18 and abs(s.r.cfo) < 1),
    ]
    fs = 16e6
    sps = 64
    nsym = 360

    def setup(self):
        self.n0 = 0
        self.hist = np.zeros(0, complex)
        self.rrc_tx = cl.rrc_taps(0.35, self.sps, 32)
        self.rrc_rx = cl.rrc_taps(0.35, 4, 16)
        self.qpsk = cl.get_constellation("qpsk")

    @staticmethod
    @lru_cache(maxsize=16)
    def cfir(N, comp):
        fo2 = np.linspace(0, 0.5, 256)
        if comp:
            pb = fo2 <= 0.09
            cic2 = mr.cic_response(np.maximum(fo2[pb], 1e-9) / 8, 8, N)
            return sg.firwin2(63, np.r_[fo2[pb], 0.16, 0.5] * 2, np.r_[1 / cic2, 0, 0], fs=2.0)
        return sg.firwin(63, 0.125 * 2, fs=2.0)

    def block(self, p):
        n = np.arange(self.n0, self.n0 + self.nsym * self.sps)
        self.n0 += len(n)
        sym = self.qpsk.modulate(cl.random_bits(2 * self.nsym, self.rng))
        nb = self.qpsk.modulate(cl.random_bits(2 * self.nsym, self.rng))
        L = len(n)
        bb = cl.shape(sym, self.rrc_tx, self.sps)[:L]
        nbw = cl.shape(nb, self.rrc_tx, self.sps)[:L]
        fw = 3.2e6
        cap = (bb * np.exp(2j * np.pi * fw / self.fs * n)
               + 10 ** (p.nlev / 20) * nbw * np.exp(2j * np.pi * (fw - p.noff * 1e6) / self.fs * n)
               + 2 * np.exp(2j * np.pi * -1.1e6 / self.fs * n))
        sc = 0.9 / max(1.0, 10 ** (p.nlev / 20) + 3)       # keep the ADC from clipping
        cap = sc * cap + 2e-5 * (self.rng.standard_normal(L) + 1j * self.rng.standard_normal(L))
        # stage 1: NCO (32-bit accumulator, 14 phase bits, 16-bit table)
        lo, _, fa = mr.nco(L, p.tune * 1e6 / self.fs, 32, 14, 16, start=int(self.n0 - L))
        mixed = cap * np.conj(lo)
        # stage 2: 16-bit fixed-point CIC, R = 8
        N = int(p.N)
        s16 = 2 ** 15 - 1
        ci = mr.cic_decimate(np.round(mixed.real * s16).astype(np.int64), 8, N)
        cq = mr.cic_decimate(np.round(mixed.imag * s16).astype(np.int64), 8, N)
        cic_out = (ci + 1j * cq) / (s16 * 8 ** N * sc)
        # stage 3: compensating FIR, ↓2 (polyphase)
        h = self.cfir(N, bool(p.comp))
        y1 = mr.polyphase_decimate(cic_out, h, 2)
        # stage 4: RRC matched filter at 4 samples/symbol
        z = np.convolve(y1, self.rrc_rx)
        return cap / sc, cic_out, z, sym, fa

    def draw(self, p, accumulate):
        cap, cic_out, z, sym, fa = self.block(p)
        # symbol timing and alignment: fixed chain delay; search a small window
        best = None
        for ph in range(4):
            zs = z[ph::4]
            c = np.abs(np.correlate(zs[:self.nsym], sym[:self.nsym // 2], "valid")[:60])
            k = int(np.argmax(c))
            if best is None or c[k] > best[0]:
                best = (c[k], ph, k)
        _, ph, lag = best
        zs = z[ph::4][lag:lag + self.nsym]
        ref = sym[:len(zs)]
        keep = slice(12, len(zs) - 12)
        zs, ref = zs[keep], ref[keep]
        g = np.vdot(ref, zs) / np.vdot(ref, ref)
        zn = zs / g if abs(g) > 1e-9 else zs
        evm = 10 * np.log10(np.mean(np.abs(zn - ref) ** 2) / np.mean(np.abs(ref) ** 2) + 1e-12)
        cfo = (3.2e6 - fa * self.fs) / 1e3
        # constellation: rotate by the block's mean phase only (residual offsets spin)
        rms = np.sqrt(np.mean(np.abs(zs) ** 2)) + 1e-12
        show = zs / rms * np.exp(-1j * np.angle(g)) if abs(g) > 1e-9 else zs / rms
        self.hist = np.r_[self.hist, show][-1500:] if (accumulate and p.persist) else show
        pc = self.plot("con")
        pc.points("z", self.hist, color=NAVY, size=3, alpha=0.5)
        pc.ideal("i", self.qpsk.points, color=RED)
        # spectra
        pa = self.plot("cap")
        f, P = cl.welch_psd(cap, self.fs, 2048)
        pa.line("P", f / 1e6, P - P.max(), color=NAVY, width=1.2, fill=-100, fill_alpha=0.06)
        pa.band("w", 3.2 - 0.17, 3.2 + 0.17, color=GREEN, alpha=0.15)
        pa.vline("nco", p.tune, color=RED, style="--", width=1.4, label="NCO", label_pos=0.9)
        pa.text("wl", 3.2, 8, "wanted", color=GREEN, anchor=(0.5, 0), size=8.5)
        pb = self.plot("cic")
        f2, P2 = cl.welch_psd(cic_out, 2e6, 1024)
        pb.line("P", f2 / 1e3, P2 - P2.max(), color=TEAL, width=1.4, fill=-100, fill_alpha=0.06)
        h = self.cfir(int(p.N), bool(p.comp))
        ff, Hh = fir_db(h, 1024)
        fH = np.r_[-ff[::-1], ff] * 2e3
        HH = np.r_[Hh[::-1], Hh]
        pb.line("H", fH, HH, color=ORANGE, width=1.4, style="--")
        pb.vline("ny", 500, color=GRAY, style=":", width=1.0)
        pb.vline("ny2", -500, color=GRAY, style=":", width=1.0)
        po = self.plot("out")
        f3, P3 = cl.welch_psd(z, 1e6, 512)
        po.line("P", f3 / 1e3, P3 - P3.max(), color=GREEN, width=1.6, fill=-100, fill_alpha=0.08)
        N = int(p.N)
        mps = 4 + 2 * 63 / 16 + 2 * len(self.rrc_rx) / 16     # mixer + CFIR + RRC, per input sample
        self.readout(evm=float(evm), cfo=float(cfo), mps=mps,
                     rates="16 → 2 → 1")

    def update(self, p):
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        r = self.r
        cfo = r.get("cfo", 0)
        s = ("<p>The capture (top left) holds the wanted 250 kBd QPSK channel at +3.2 MHz, a "
             f"neighbour {v(p.nlev, '.0f', 'dB')} stronger {v(p.noff, '.2f', 'MHz')} below it, and a "
             "carrier at −1.1 MHz. The DDC <b>mixes</b> the wanted channel to 0 Hz with the NCO, "
             "decimates by 8 with a <b>CIC</b> (adders only), corrects the CIC's droop and "
             "decimates by 2 with a 63-tap <b>FIR</b>, and finishes with the RRC <b>matched filter</b> "
             "at 4 samples per symbol.</p>")
        if abs(cfo) >= 1:
            s += (f"<p>{bad('Mistuned')} by {v(cfo, '.1f', 'kHz')}: the constellation spins, because a "
                  f"frequency error is a phase that grows with time. Slide the NCO to 3.2 MHz.</p>")
        else:
            s += (f"<p>Tuned. EVM {v(r.get('evm', 0), '.1f', 'dB')}: "
                  + (good("a clean constellation.") if r.get("evm", 0) < -25 else
                     "the neighbour is getting through: look at what the CIC and FIR let past.") + "</p>")
        s += ("<p>Only the mixer and the CIC integrators run at 16 MS/s; every sharp filter runs after "
              "most of the decimation, where it is cheap. That is the whole art of the DDC.</p>")
        return "<h3>Push arithmetic to the lowest rate</h3>" + s + keybox(
            "NCO → CIC ↓R → CFIR ↓2 → PFIR/RRC: the architecture inside every SDR's FPGA.")


# =============================================================================== the lab
LAB = st.Lab(17, "Digital Filters and Multirate DSP", chapter=6,
             chapter_title="Digital Filters and Multirate Processing",
             experiments=[PoleZero, FIRDesign, IIRFilters, RateChange, Channelizer, CIC, NCO,
                          CORDIC, DDC])

if __name__ == "__main__":
    st.run(LAB)
