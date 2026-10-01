"""Lab 03 · Pulse Shaping, the Nyquist Criterion and the Eye Diagram   (Chapter 8)

Run it:      python labs/lab03_pulse_shaping.py
Self-test:   python labs/lab03_pulse_shaping.py --selftest

Symbols are numbers; the wire and the air want waveforms. The pulse you choose decides how
much spectrum you occupy, how much you leak into your neighbours, how much your symbols
smear into each other, and how gracefully the receiver survives noise and timing error.
Eight experiments, from a single pulse to a live oscilloscope eye and a Monte Carlo BER run.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve, lfilter

import commlib as cl
from commlib import linecodes as lc
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, EyePlot, ConstellationPlot, BERPlot, BarPlot,
                    Readout, Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD)
from studio import v, eq, keybox, good, bad


# =============================================================================== shared helpers
@lru_cache(maxsize=64)
def rrc(beta, sps, span):
    """Unit-energy root-raised-cosine taps (cached: the slider revisits the same values)."""
    return cl.rrc_taps(max(beta, 1e-3), sps, span)


@lru_cache(maxsize=64)
def rc_unit(beta, sps, span):
    """Raised-cosine taps normalised to unit energy (so Es is the same as for RRC)."""
    h = cl.rc_taps(beta, sps, span)
    return h / np.sqrt(np.sum(h ** 2))


def rc_pulse(t, beta):
    """The raised-cosine pulse p(t) at arbitrary times t (in symbol periods), peak 1."""
    t = np.asarray(t, float)
    h = np.sinc(t)
    if beta > 0:
        den = 1 - (2 * beta * t) ** 2
        sing = np.isclose(den, 0, atol=1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(sing, np.pi / 4 * np.sinc(1 / (2 * beta)),
                         h * np.cos(np.pi * beta * t) / np.where(sing, 1, den))
    return h


def rc_spectrum(f, beta):
    """Raised-cosine spectrum P(f) (f in multiples of Rs), P(0) = 1."""
    a = np.abs(f)
    out = np.zeros_like(a)
    out[a <= (1 - beta) / 2] = 1
    if beta > 0:
        m = (a > (1 - beta) / 2) & (a <= (1 + beta) / 2)
        out[m] = 0.5 * (1 + np.cos(np.pi / beta * (a[m] - (1 - beta) / 2)))
    return out


def pam_levels(M):
    """Unit-average-energy PAM levels."""
    return cl.get_constellation(f"{M}pam").points.real


def one_pole(x, f3db, fs):
    """A first-order RC low-pass channel with −3 dB bandwidth f3db (same units as fs)."""
    a = np.exp(-2 * np.pi * f3db / fs)
    return lfilter([1 - a], [1, -a], x)


def eye_metrics(y, sym, d, sps, levels, nphase=64):
    """Measure an eye from the receiver waveform y, the transmitted symbols and the
    sampling index d of symbol 0. Returns dict(height, width, phases, open, mer, best).

    height = worst inner opening / ideal level spacing, at the best phase (can be < 0);
    open[φ] = the same opening as a function of sampling phase (the "bathtub")."""
    M = len(levels)
    spacing = levels[1] - levels[0]
    K = len(sym)
    phases = np.linspace(-0.5, 0.5, nphase, endpoint=False)
    k = np.arange(8, K - 8)
    pos = d + k[:, None] * sps + phases[None, :] * sps
    pos = np.clip(pos, 0, len(y) - 2)
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    S = y[i0] * (1 - w) + y[i0 + 1] * w                      # (K', nphase)
    lev_idx = np.searchsorted(levels, sym[k] - 1e-9)
    opening = np.full(nphase, np.inf)
    for j in range(M - 1):
        lo, hi = S[lev_idx == j], S[lev_idx == j + 1]
        if len(lo) == 0 or len(hi) == 0:
            continue
        opening = np.minimum(opening, hi.min(axis=0) - lo.max(axis=0))
    opening = opening / spacing
    best = int(np.argmax(opening))
    err = S[:, best] - levels[lev_idx]
    mer = 10 * np.log10(np.mean(levels ** 2) / max(np.mean(err ** 2), 1e-12))
    return dict(height=float(opening[best]), width=float(np.mean(opening > 0)),
                phases=phases, open=opening, mer=mer, best=best, samples=S[:, best],
                sigma=float(np.std(err)))


def ber_from_sigma(M, sigma, spacing):
    """Gaussian estimate of the Gray-coded PAM bit error rate from the rms sample error."""
    if sigma <= 0:
        return 0.0
    return float(2 * (M - 1) / (M * np.log2(M)) * cl.qfunc(spacing / (2 * sigma)))


# =============================================================================== 1. pulses
_ACLR_SPAN_22 = None


def _aclr(h, edge, nfft=1 << 14, sps=16):
    H = np.abs(np.fft.fft(h, nfft)) ** 2
    f = np.abs(np.fft.fftfreq(nfft, 1 / sps))
    return 10 * np.log10(np.sum(H[(f > edge) & (f <= 3 * edge)]) / np.sum(H[f <= edge]) + 1e-30)


def _min_span_022():
    global _ACLR_SPAN_22
    if _ACLR_SPAN_22 is None:
        _ACLR_SPAN_22 = next(s for s in range(2, 80, 2) if _aclr(rrc(0.22, 16, s), 0.61) < -45)
    return _ACLR_SPAN_22


class PulseSpectra(Experiment):
    title = "Pulses and their spectra"
    blurb = "Pick a pulse; see the bandwidth it occupies and how fast its tails die."
    book = "sec:ch08:rc"
    controls = [
        Choice("pulse", "Pulse shape", ["Rectangular", "Sinc (ideal)", "Raised cosine",
                                        "Root raised cosine"], "Raised cosine"),
        Slider("beta", "Roll-off β", 0.0, 1.0, 0.35, step=0.01,
               help="Excess bandwidth: the pulse occupies (1+β)·Rs",
               enabled_if=lambda p: "cosine" in p.pulse),
        IntSlider("span", "Filter length", 2, 32, 8, step=2, unit="symbols",
                  help="Real filters are truncated; shorter filters leak more power next door"),
        Toggle("compare", "Overlay the other pulses", False),
    ]
    plots = [
        Plot("time", "Impulse response p(t)", x="time (symbol periods T)", y="amplitude",
             xlim=(-6, 6), ylim=(-0.3, 1.1), legend="tr"),
        SpectrumPlot("spec", "Power spectrum |P(f)|²", x="frequency (multiples of Rs)",
                     xlim=(-2.2, 2.2), ylim=(-90, 5), legend=None),
        Plot("tail", "How fast the tails die: |p(t)| in dB", x="time (symbol periods T)",
             y="|p(t)| (dB)", xlim=(0, 16), ylim=(-100, 3)),
    ]
    layout = [["time", "spec"], ["tail", "tail"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bw", "99 % power bandwidth", "× Rs", ".2f",
                help="Two-sided bandwidth holding 99 % of the power, in multiples of the symbol rate"),
        Readout("excess", "Excess bandwidth", "%", ".0f"),
        Readout("aclr", "Adjacent-channel leakage", "dB", ".1f", good=lambda x: x < -45,
                help="Power in the next channel up relative to your own (ACLR)"),
        Readout("tail", "Tail near t = 3T", "dB", ".0f", good=lambda x: x < -40),
    ]

    @property
    def challenges(self):
        n = _min_span_022()
        return [
            Challenge("Choose a roll-off that uses exactly 25 % excess bandwidth (a DVB-S2 option).",
                      lambda s: "cosine" in s.p.pulse and abs(s.p.beta - 0.25) < 0.006),
            Challenge("Make the tail near t = 3T at least 40 dB below the peak, with a pulse that "
                      "is still band-limited.",
                      lambda s: s.p.pulse != "Rectangular" and s.r.tail < -40,
                      hint="Larger roll-off = smoother spectrum = faster-dying tails."),
            Challenge(f"With an RRC at β = 0.22 (3G/UMTS), push the leakage below −45 dB using a "
                      f"filter no longer than {n} symbols.",
                      lambda s: (s.p.pulse == "Root raised cosine" and abs(s.p.beta - 0.22) < 0.006
                                 and s.r.aclr < -45 and s.p.span <= n),
                      hint="Lengthen the filter one step at a time and watch the ACLR readout."),
        ]

    sps = 16

    def taps(self, kind, beta, span):
        t = np.arange(-span * self.sps // 2, span * self.sps // 2 + 1) / self.sps
        if kind == "Rectangular":
            h = np.where(np.abs(t) < 0.5, 1.0, 0.0)
            h[np.isclose(np.abs(t), 0.5)] = 0.5
        elif kind == "Sinc (ideal)":
            h = np.sinc(t)
        elif kind == "Raised cosine":
            h = cl.rc_taps(beta, self.sps, span)
        else:
            h = rrc(beta, self.sps, span)
            h = h / h.max()
        return t, h

    def edge(self, kind, beta):
        """Nominal one-sided channel edge (× Rs) used for the ACLR measurement."""
        return {"Rectangular": 1.0, "Sinc (ideal)": 0.5}.get(kind, (1 + beta) / 2)

    def analyse(self, kind, beta, span):
        t, h = self.taps(kind, beta, span)
        nfft = 1 << 14
        H = np.abs(np.fft.fft(h, nfft)) ** 2
        f = np.fft.fftfreq(nfft, 1 / self.sps)
        order = np.argsort(np.abs(f))
        c = np.cumsum(H[order]) / H.sum()
        f99 = np.abs(f[order][np.searchsorted(c, 0.99)])
        fs_, Hs = np.fft.fftshift(f), np.fft.fftshift(H)
        Pdb = 10 * np.log10(Hs / Hs.max() + 1e-12)
        return t, h, fs_, Pdb, 2 * f99, _aclr(h, self.edge(kind, beta), nfft, self.sps)

    def update(self, p):
        beta = p.beta if "cosine" in p.pulse else 0.0
        t, h, f, Pdb, bw99, aclr = self.analyse(p.pulse, beta, p.span)
        pt, ps, pl = self.plot("time"), self.plot("spec"), self.plot("tail")
        others = [k for k in ("Rectangular", "Sinc (ideal)", "Raised cosine", "Root raised cosine")
                  if k != p.pulse]
        if p.compare:
            for i, k in enumerate(others):
                b = p.beta if "cosine" in k else 0.0
                t2, h2, f2, P2, *_ = self.analyse(k, b, p.span)
                col = [GRAY, ORANGE, PURPLE][i]
                pt.line("o" + str(i), t2, h2, color=col, width=1.2, alpha=0.7, name=k)
                ps.line("o" + str(i), f2, P2, color=col, width=1.0, alpha=0.7, name=k)
                pl.line("o" + str(i), t2[t2 >= 0], 20 * np.log10(np.abs(h2[t2 >= 0]) + 1e-12),
                        color=col, width=1.0, alpha=0.6)
        pt.hline("zero", 0, color=GRAY, style="-", width=0.8)
        pt.line("h", t, h, color=NAVY, width=2.4, name=p.pulse)
        k = np.arange(-6, 7)
        k = k[(k != 0) & (np.abs(k) <= p.span // 2)]
        hk = np.interp(k, t, h)
        pt.scatter("zc", k, hk, color=RED, size=8, name="at the neighbours' instants")
        # spectrum
        e = self.edge(p.pulse, beta)
        ps.band("occ", -e, e, color=GREEN, alpha=0.10)
        ps.band("adj", e, 3 * e, color=RED, alpha=0.06)
        ps.text("adjlab", e + 0.05, -8, "adjacent\nchannel", color=RED, size=8.5)
        ps.vline("nyq-", -0.5, color=GRAY, style=":")
        ps.vline("nyq+", 0.5, color=GRAY, style=":", label="Rs/2", label_pos=0.06)
        ps.line("P", f, Pdb, color=NAVY, width=2.0, name=p.pulse)
        # tails
        m = t >= 0
        pl.line("tail", t[m], 20 * np.log10(np.abs(h[m]) + 1e-12), color=NAVY, width=1.6)
        near3 = (np.abs(t) >= 2.5) & (np.abs(t) <= 3.5)
        tail = 20 * np.log10(np.max(np.abs(h[near3])) + 1e-12) if near3.any() else -120
        pl.hline("t3", tail, color=RED, style="--", label=f"tail near 3T: {tail:.0f} dB", label_pos=0.6)
        pl.vline("end", p.span / 2, color=ORANGE, style=":", label="filter ends", label_pos=0.85)
        excess = 100 * beta if "cosine" in p.pulse else (0 if "Sinc" in p.pulse else None)
        self.readout(bw=bw99, excess=excess if excess is not None else "—", aclr=aclr,
                     tail=max(tail, -120))

    def story(self, p):
        beta = p.beta
        if p.pulse == "Rectangular":
            body = (f"<p>The rectangular pulse is what a switch makes: trivial to build, and "
                    f"terrible for your neighbours. Its spectrum is a <b>sinc</b> whose sidelobes "
                    f"fall only 6 dB per octave, so 99 % of the power needs "
                    f"{v(self.r.get('bw', 0))}× the symbol rate, and the adjacent channel "
                    f"still sees {v(self.r.get('aclr', 0), '.0f', 'dB')}.</p>"
                    f"<p>Note the red dots: a rectangle is zero at every other symbol's sampling "
                    f"instant, so it causes no ISI. It is a Nyquist pulse; it just wastes spectrum.</p>")
        elif p.pulse == "Sinc (ideal)":
            body = ("<p>The sinc is Nyquist's ideal: a perfect brick-wall spectrum exactly "
                    "Rs/2 wide on each side, the narrowest possible zero-ISI pulse. "
                    "Look at its tails though: they decay only as 1/t. Truncate it and the "
                    "brick wall grows skirts; sample it a little late and the ISI from "
                    "hundreds of neighbours adds up.</p>")
        elif p.pulse == "Raised cosine":
            body = (f"<p>Your roll-off of {v(beta)} means the signal occupies "
                    f"{v(1 + beta)}× the symbol rate: {v(100 * beta, '.0f')} % more than "
                    f"Nyquist's minimum. In exchange, the spectrum's edges are a smooth "
                    f"cosine instead of a cliff, and the tails die as 1/t³.</p>"
                    "<p>The red dots sit exactly on zero: every other symbol's pulse passes "
                    "through zero at <i>your</i> sampling instant. That is the zero-ISI property.</p>")
        else:
            body = (f"<p>The <b>root</b> raised cosine has the square root of the RC spectrum. "
                    f"Put one at the transmitter and an identical one at the receiver and the "
                    f"two multiply to a full raised cosine.</p>"
                    "<p>Notice the red dots are <b>not</b> zero: the RRC alone is not a Nyquist "
                    "pulse. Only the matched pair is. Experiment 5 shows what happens if you forget.</p>")
        trunc = (f"<p>The filter is {v(p.span, 'd')} symbols long. Shorten it and watch the "
                 f"spectral floor rise into the shaded adjacent channel: real ACLR budgets "
                 f"(3GPP asks about 45 dB of a base station) are set as much by filter length "
                 f"as by β.</p>")
        return ("<h3>Shaping a symbol</h3>" + body + trunc +
                keybox("Bandwidth, tails and filter length trade against each other. "
                       "Small β is spectrally efficient but needs long filters and precise timing."))


# =============================================================================== 2. Nyquist
class NyquistZeroISI(Experiment):
    title = "Nyquist: zero ISI"
    blurb = "Overlapping pulses that still don't interfere, and how timing error breaks the spell."
    book = "sec:ch08:nyquist"
    controls = [
        Choice("pulse", "End-to-end pulse", ["Raised cosine", "Gaussian (not Nyquist)",
                                             "RC filtered twice"], style="menu"),
        Slider("beta", "Roll-off β", 0.0, 1.0, 0.35, step=0.01,
               enabled_if=lambda p: p.pulse != "Gaussian (not Nyquist)"),
        Slider("eps", "Sampling offset ε", -0.5, 0.5, 0.0, step=0.01, unit="T",
               help="Sample ε symbol periods away from the ideal instant"),
        Choice("data", "Symbols", ["Random", "Alternating", "Single"]),
        Button("again", "New random symbols"),
    ]
    plots = [
        Plot("sum", "Each symbol's pulse (thin) and the received waveform (thick)",
             x="time (symbol periods T)", y="amplitude", xlim=(-5.5, 5.5), ylim=(-1.9, 2.5),
             legend="tl", legend_cols=3),
        Plot("fold", "Folded spectrum: Σ P(f − m·Rs)", x="frequency (multiples of Rs)",
             y="P(f) (normalised)", xlim=(-1.5, 1.5), ylim=(-0.05, 1.7)),
        BarPlot("isi", "What the sampler sees from each symbol: p(kT + ε)",
                x="symbol index k (0 = the wanted symbol)", y="contribution", xlim=(-6.6, 6.6),
                ylim=(-0.35, 1.1)),
    ]
    layout = [["sum", "sum"], ["fold", "isi"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("isi", "Worst-case ISI", "% of signal", ".0f", good=lambda x: x < 10),
        Readout("open", "Worst-case eye opening", "%", ".0f", good=lambda x: x > 50),
        Readout("ripple", "Folded-spectrum ripple", "%", ".1f", good=lambda x: x < 1),
    ]
    challenges = [
        Challenge("With a quarter-symbol timing error (|ε| ≥ 0.25 T), keep the worst-case ISI "
                  "below 40 %.",
                  lambda s: abs(s.p.eps) >= 0.249 and s.r.isi < 40,
                  hint="Faster-dying tails mean fewer neighbours leak in: raise β."),
        Challenge("Break Nyquist: pick a pulse whose folded spectrum is not flat, and see ISI "
                  "appear even with perfect timing.",
                  lambda s: s.r.ripple > 5 and abs(s.p.eps) < 0.005 and s.r.isi > 2),
        Challenge("Try the brick wall: β = 0 with a timing error of 0.1 T. How bad is it?",
                  lambda s: s.p.pulse == "Raised cosine" and s.p.beta <= 0.005 and abs(s.p.eps) >= 0.099),
    ]

    def setup(self):
        self.sym = np.sign(self.rng.standard_normal(11))

    def on_again(self, p):
        self.sym = np.sign(self.rng.standard_normal(11))

    def pulse(self, t, p):
        if p.pulse == "Raised cosine":
            return rc_pulse(t, p.beta)
        if p.pulse == "Gaussian (not Nyquist)":
            return np.exp(-np.pi ** 2 * np.asarray(t) ** 2 / (4 * np.log(2)))
        # RC * RC: convolve on a fine grid, normalise the peak
        key = round(p.beta, 3)
        if getattr(self, "_rr_key", None) != key:
            dt = 1 / 32
            tt = np.arange(-24, 24 + dt / 2, dt)
            g = np.convolve(rc_pulse(tt, p.beta), rc_pulse(tt, p.beta), mode="same") * dt
            self._rr = (tt, g / g.max())
            self._rr_key = key
        tt, g = self._rr
        return np.interp(t, tt, g, left=0, right=0)

    def spectrum(self, f, p):
        if p.pulse == "Raised cosine":
            return rc_spectrum(f, p.beta)
        if p.pulse == "Gaussian (not Nyquist)":
            return np.exp(-np.log(2) * (np.asarray(f) / 0.5) ** 2)
        return rc_spectrum(f, p.beta) ** 2

    def update(self, p):
        if p.data == "Random":
            a = self.sym
        elif p.data == "Alternating":
            a = np.array([(-1.0) ** k for k in range(11)])
        else:
            a = np.zeros(11)
            a[5] = 1.0
        ks = np.arange(-5, 6)
        t = np.linspace(-6, 6, 1201)
        ps = self.plot("sum")
        total = np.zeros_like(t)
        cols = [BLUE, ORANGE, GREEN, PURPLE, GOLD, RED]
        for i, (k, ak) in enumerate(zip(ks, a)):
            if ak == 0:
                continue
            y = ak * self.pulse(t - k, p)
            total += y
            ps.line(f"p{i}", t, y, color=cols[i % len(cols)], width=1.0, alpha=0.55)
        for k in ks:
            ps.vline(f"g{k}", k, color=GRAY, style=":", width=0.7)
        ps.line("sum", t, total, color=NAVY, width=2.6, name="received waveform")
        ts = ks + p.eps
        ys = np.interp(ts, t, total)
        ps.scatter("ideal", ks, a, color=GREEN, size=11, outline=GREEN, name="symbol sent")
        ps.scatter("samp", ts, ys, color=RED, size=8, name="sampled value")
        # folded spectrum
        f = np.linspace(-1.5, 1.5, 1501)
        pf = self.plot("fold")
        tot = np.zeros_like(f)
        for m in range(-3, 4):
            c = self.spectrum(f - m, p)
            tot += c
            if m:
                pf.line(f"c{m}", f, c, color=GRAY, width=1.0, alpha=0.7,
                        name="shifted copies" if m == 1 else None)
        pf.line("P", f, self.spectrum(f, p), color=NAVY, width=2.2, name="P(f)")
        pf.line("tot", f, tot, color=RED, width=2.4, name="sum of copies")
        inner = np.abs(f) <= 0.5
        ripple = 100 * (tot[inner].max() - tot[inner].min()) / tot[inner].mean()
        # ISI taps
        k = np.arange(-30, 31)
        c = self.pulse(k + p.eps, p)
        main = c[30]
        D = (np.sum(np.abs(c)) - abs(main)) / max(abs(main), 1e-9)
        kk = np.arange(-6, 7)
        cc = c[24:37]
        pb = self.plot("isi")
        pb.bars("b", kk, cc, width=0.6, colors=[NAVY if x == 0 else RED for x in kk])
        pb.set_xticks([(x, str(x)) for x in kk])
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        self.readout(isi=100 * D, open=max(0.0, 100 * (1 - D)), ripple=ripple)

    def story(self, p):
        isi = self.r.get("isi", 0)
        if p.pulse == "Raised cosine" and abs(p.eps) < 0.005:
            s = ("<p>Every symbol's pulse (thin curves) overlaps its neighbours heavily, yet at "
                 "each sampling instant (dotted lines) all the others pass exactly through zero. "
                 "The red samples land right on the green symbols: <span class='good'>zero ISI</span>.</p>"
                 "<p>The folded spectrum shows why: copies of P(f) shifted by the symbol rate add up "
                 "to a perfectly flat line. Flat folded spectrum ⇔ zero ISI. That is the "
                 "<b>Nyquist criterion</b>.</p>")
        elif p.pulse == "Raised cosine":
            s = (f"<p>You are sampling {v(abs(p.eps))} T {'late' if p.eps > 0 else 'early'}. "
                 f"The neighbours are no longer at their zero crossings, and their leftover "
                 f"values (red bars) add up to as much as {v(isi, '.0f')} % of the wanted "
                 f"symbol in the worst case.</p>"
                 f"<p>Larger β makes the tails die faster, so fewer neighbours matter: compare "
                 f"β = 0.1 with β = 1 at the same offset.</p>")
        else:
            s = (f"<p>This pulse is <span class='bad'>not Nyquist</span>: its shifted spectra do not "
                 f"add to a flat line (ripple {v(self.r.get('ripple', 0), '.0f')} %), so even with "
                 f"perfect timing the neighbours leak into each sample (red bars at k ≠ 0).</p>"
                 + ("<p>Gaussian pulses like this are used on purpose in GMSK (GSM, Bluetooth): "
                    "the receiver must then undo the ISI with an equaliser or a Viterbi detector.</p>"
                    if "Gaussian" in p.pulse else
                    "<p>Filtering a raised cosine again (RC at both ends) is a classic bug: two "
                    "Nyquist filters in a row are not Nyquist. Split it into two RRCs instead.</p>"))
        return "<h3>Overlap without interference</h3>" + s + keybox(
            "Nyquist (1928): p(kT) = 0 for every k ≠ 0  ⇔  Σ P(f − m/T) is flat.")


# =============================================================================== 3. live eye
class LiveEye(Experiment):
    title = "Live eye diagram"
    blurb = "An oscilloscope in persistence mode: watch noise, jitter and bandwidth close the eye."
    book = "sec:ch08:eyes"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Heading("Transmitter and receiver"),
        Choice("M", "Signal", ["2-PAM (NRZ)", "4-PAM"]),
        Choice("filters", "Filters", ["RRC → RRC (matched)", "RC → nothing", "RRC → nothing"],
               style="menu"),
        Slider("beta", "Roll-off β", 0.05, 1.0, 0.35, step=0.01),
        Heading("Impairments"),
        Slider("esn0", "Es/N0", 0.0, 40.0, 25.0, step=0.5, unit="dB"),
        Slider("jitter", "Timing jitter (rms)", 0.0, 15.0, 0.0, step=0.5, unit="% UI"),
        Toggle("chan", "Band-limited channel", False),
        LogSlider("bw", "Channel bandwidth", 0.15, 3.0, 0.5, unit="× Rs",
                  help="−3 dB bandwidth of a first-order RC channel", enabled_if=lambda p: p.chan),
        Toggle("persist", "Persistence", True),
    ]
    plots = [
        EyePlot("eye", "Eye diagram (two symbol periods)", yrange=(-1.7, 1.7)),
        Plot("hist", "Values at the best sampling instant", x="amplitude", y="count",
             legend=None, xlim=(-1.7, 1.7)),
        Plot("tub", "Eye opening vs sampling phase", x="sampling phase (UI)",
             y="opening (% of ideal)", xlim=(-0.5, 0.5), ylim=(-100, 105), legend=None),
    ]
    layout = [["eye", "hist"], ["eye", "tub"]]
    col_stretch = [5, 3]
    readouts = [
        Readout("height", "Eye height", "%", ".0f", good=lambda x: x > 50),
        Readout("width", "Eye width", "% UI", ".0f", good=lambda x: x > 50),
        Readout("mer", "SNR at the sampler", "dB", ".1f"),
        Readout("ber", "Estimated BER", "", "sci", good=lambda x: x < 1e-6),
    ]
    challenges = [
        Challenge("Squeeze the channel to 0.25 Rs or less and still keep the eye at least 50 % open.",
                  lambda s: s.p.chan and s.p.bw <= 0.25 and s.r.height >= 50,
                  hint="There is a sweet spot in β: too small and the tails ring, too large and the "
                       "channel cuts off the extra bandwidth. Raise Es/N0 too."),
        Challenge("Close the eye horizontally with jitter alone: width below 50 % at Es/N0 ≥ 30 dB, "
                  "no channel.",
                  lambda s: s.p.jitter > 0 and s.r.width < 50 and s.p.esn0 >= 30 and not s.p.chan),
        Challenge("4-PAM: keep all three eyes at least 50 % open with Es/N0 of 19 dB or less.",
                  lambda s: s.p.M == "4-PAM" and s.p.esn0 <= 19 and s.r.height >= 50),
        Challenge("See pure ISI: choose RRC → nothing with Es/N0 ≥ 35 dB. The traces no longer "
                  "meet at single points.",
                  lambda s: s.p.filters == "RRC → nothing" and s.p.esn0 >= 35),
    ]
    sps = 16
    nsym = 400

    def make_frame(self, p):
        M = 2 if p.M.startswith("2") else 4
        lev = pam_levels(M)
        a = lev[self.rng.integers(0, M, self.nsym)]
        span = 12
        if p.filters.startswith("RC"):
            tx, rx = rc_unit(round(p.beta, 3), self.sps, span), np.array([1.0])
        elif p.filters == "RRC → nothing":
            tx, rx = rrc(round(p.beta, 3), self.sps, span), np.array([1.0])
        else:
            tx = rx = rrc(round(p.beta, 3), self.sps, span)
        x = cl.shape(a, tx, self.sps).real
        g = np.convolve(tx, rx)
        if p.chan:
            x = one_pole(x, p.bw, self.sps)
            g = one_pole(np.r_[g, np.zeros(8 * self.sps)], p.bw, self.sps)
        sigma = np.sqrt(1 / (2 * 10 ** (p.esn0 / 10)))
        x = x + sigma * self.rng.standard_normal(len(x))
        y = np.convolve(x, rx)
        d = int(np.argmax(g))
        y = y / (g[d] * lev.max())
        return y, a / lev.max(), d, lev / lev.max(), M

    def draw(self, p, accumulate):
        y, a, d, lev, M = self.make_frame(p)
        pe = self.plot("eye")
        start = d - self.sps + 10 * self.sps
        pe.eye("eye", y[:d + (self.nsym - 10) * self.sps], self.sps, n_sym=2, offset=start,
               yrange=(-1.7, 1.7), accumulate=accumulate and p.persist, decay=0.88,
               jitter=p.jitter / 100, rng=self.rng)
        m = eye_metrics(y, a, d, self.sps, lev)
        if p.jitter > 0:
            # Jitter moves the sampling instant by up to about ±2.5σ: the usable opening at
            # phase φ is the worst opening within that window (it carves the bathtub walls).
            ph = m["phases"]
            win = np.abs(ph[:, None] - ph[None, :]) <= 2.5 * p.jitter / 100
            op = np.where(win, m["open"][None, :], np.inf).min(axis=1)
            m.update(open=op, best=int(np.argmax(op)), height=float(op.max()),
                     width=float(np.mean(op > 0)))
        # smooth the readouts over frames while streaming so the numbers are steady
        sm = getattr(self, "_sm", None)
        if accumulate and sm is not None:
            for k in ("height", "width", "mer", "sigma"):
                m[k] = 0.75 * sm[k] + 0.25 * m[k]
            m["open"] = 0.75 * sm["open"] + 0.25 * m["open"]
        self._sm = {k: m[k] for k in ("height", "width", "mer", "sigma", "open")}
        ph = m["phases"]
        pt = self.plot("tub")
        pt.hline("z", 0, color=GRAY, style="-", width=0.8)
        pt.line("op", ph, 100 * np.clip(m["open"], -1, 1.05), color=NAVY, width=2.2, fill=0,
                fill_alpha=0.12)
        pt.vline("best", ph[m["best"]], color=RED, style="--", label="best phase", label_pos=0.1)
        pe.vline("best", 1 + ph[m["best"]], color=RED, style="--", width=1.0)
        # histogram of the samples at the best phase
        h, edges = np.histogram(m["samples"], bins=120, range=(-1.7, 1.7))
        pp = self.plot("hist")
        pp.line("h", edges, h, color=NAVY, width=1.4, step=True, fill=0, fill_alpha=0.25)
        for i, L in enumerate(lev):
            pp.vline(f"l{i}", L, color=GREEN, style=":", width=1.0)
        spacing = lev[1] - lev[0]
        ber = ber_from_sigma(M, m["sigma"], spacing)
        self.readout(height=100 * m["height"], width=100 * m["width"], mer=m["mer"], ber=ber)

    def update(self, p):
        self.plot("eye").reset_persistence()
        self._sm = None
        self.draw(p, accumulate=False)

    def tick(self, p):
        self.draw(p, accumulate=True)

    def story(self, p):
        h, w = self.r.get("height", 0), self.r.get("width", 0)
        state = (good("wide open") if h > 70 else "partly closed" if h > 20
                 else bad("closed") if h <= 0 else bad("nearly closed"))
        s = (f"<p>Each trace is two symbol periods of the received waveform, all drawn on top of "
             f"each other like a scope in persistence mode (brighter = more traces). "
             f"Right now the eye is {state}: {v(h, '.0f')} % of its ideal height and "
             f"{v(w, '.0f')} % of a symbol wide.</p>"
             "<p><b>Vertical</b> opening is your noise margin; <b>horizontal</b> opening is your "
             "timing margin. The receiver samples at the widest point (red line, bottom right).</p>")
        tips = []
        if p.chan:
            tips.append(f"The channel's {v(p.bw, '.2f')}·Rs bandwidth smears each pulse into the "
                        f"next symbols: the traces no longer converge.")
        if p.jitter > 0:
            tips.append(f"{v(p.jitter, '.1f')} % UI of rms jitter shakes every trace sideways: "
                        f"the eye's corners fill in and its width shrinks.")
        if p.esn0 < 15:
            tips.append("At this Es/N0 the noise fills the eye from top and bottom.")
        if p.filters == "RRC → nothing":
            tips.append("An RRC alone is not a Nyquist pulse: without the matched RRC at the "
                        "receiver, ISI remains even with no noise.")
        if tips:
            s += "<p>" + " ".join(tips) + "</p>"
        return "<h3>Reading an eye</h3>" + s + keybox(
            "Press Play/Pause to freeze the scope. In SerDes compliance testing, a "
            "hexagonal <b>mask</b> must fit inside the eye.")


# =============================================================================== 4. matched filter
class MatchedFilter(Experiment):
    title = "The matched filter"
    blurb = "Why the receive filter should look like the transmitted pulse, reversed."
    book = "sec:ch08:mf"
    controls = [
        Choice("rx", "Receive filter", ["Matched (RRC, β = 0.35)", "Moving average",
                                        "Low-pass (brick-wall)", "Gaussian low-pass",
                                        "RRC, other roll-off"]),
        Slider("width", "Averaging length", 0.25, 2.0, 1.0, step=0.05, unit="T",
               enabled_if=lambda p: p.rx == "Moving average"),
        Slider("fc", "Cut-off frequency", 0.2, 1.5, 0.6, step=0.01, unit="× Rs",
               enabled_if=lambda p: "low-pass" in p.rx.lower()),
        Slider("brx", "Receiver roll-off", 0.05, 1.0, 0.7, step=0.01,
               enabled_if=lambda p: p.rx == "RRC, other roll-off"),
        Slider("esn0", "Es/N0", 0.0, 30.0, 12.0, step=0.5, unit="dB"),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("imp", "Transmit pulse p(t) and receive filter h(−t)", x="time (symbol periods T)",
             y="amplitude (peak 1)", xlim=(-4, 4), ylim=(-0.35, 1.15)),
        SpectrumPlot("freq", "What the filter lets through", x="frequency (multiples of Rs)",
                     xlim=(0, 1.6), ylim=(-50, 4)),
        Plot("out", "Filter output for a burst of ±1 symbols", x="time (symbol periods T)",
             y="amplitude", xlim=(-0.5, 14.5), ylim=(-2.2, 3.4), legend="tl", legend_cols=2),
        BarPlot("bud", "SNR at the sampler", x="", y="dB"),
    ]
    readouts = [
        Readout("loss", "SNR loss vs matched", "dB", ".2f", good=lambda x: x > -0.5),
        Readout("isi", "Residual ISI", "dB", ".1f", good=lambda x: x < -30),
        Readout("sinr", "SINR at the sampler", "dB", ".1f"),
        Readout("ber", "BER (2-PAM)", "", "sci"),
    ]
    challenges = [
        Challenge("Get within 0.5 dB of the matched filter with a filter that is NOT matched.",
                  lambda s: not s.p.rx.startswith("Matched") and s.r.loss > -0.5,
                  hint="Try the low-pass filters with a cut-off near the signal's band edge."),
        Challenge("Find the moving-average length that loses less than 0.6 dB.",
                  lambda s: s.p.rx == "Moving average" and s.r.loss > -0.6),
        Challenge("Make ISI, not noise, the limit: residual ISI stronger than the noise at the sampler.",
                  lambda s: s.r.isi > -(s.p.esn0 + s.r.loss),
                  hint="A wide low-pass filter passes the TX pulse untouched, and an RRC alone is not Nyquist."),
    ]
    sps = 16
    beta_tx = 0.35

    def setup(self):
        self.p_tx = rrc(self.beta_tx, self.sps, 16)
        self.burst = np.sign(self.rng.standard_normal(14))
        self.noise_seed = 1

    def on_again(self, p):
        self.noise_seed += 1
        self.burst = np.sign(self.rng.standard_normal(14))

    def rx_filter(self, p):
        s = self.sps
        if p.rx.startswith("Matched"):
            return self.p_tx
        if p.rx == "Moving average":
            return np.ones(max(1, int(round(p.width * s))))
        if p.rx == "Low-pass (brick-wall)":
            from scipy.signal import firwin
            return firwin(16 * s + 1, min(p.fc, 7.9) / (s / 2), window=("kaiser", 6))
        if p.rx == "Gaussian low-pass":
            sig = np.sqrt(np.log(2)) / (2 * np.pi * p.fc)
            t = np.arange(-6 * s, 6 * s + 1) / s
            return np.exp(-t ** 2 / (2 * sig ** 2))
        return rrc(round(p.brx, 3), s, 16)

    def update(self, p):
        s = self.sps
        ptx = self.p_tx
        h = self.rx_filter(p)
        g = np.convolve(ptx, h)
        d = int(np.argmax(np.abs(g)))
        eta = g[d] ** 2 / (np.sum(ptx ** 2) * np.sum(h ** 2))
        loss = 10 * np.log10(eta)
        idx = np.arange(d % s, len(g), s)
        isi_lin = (np.sum(g[idx] ** 2) - g[d] ** 2) / g[d] ** 2
        isi = 10 * np.log10(isi_lin + 1e-12)
        snr = 10 ** (p.esn0 / 10) * eta
        sinr = 1 / (1 / snr + isi_lin)
        ber = float(cl.qfunc(np.sqrt(2 * sinr))) if np.isfinite(sinr) else 0.0
        # impulse responses
        pi_ = self.plot("imp")
        t1 = (np.arange(len(ptx)) - (len(ptx) - 1) / 2) / s
        t2 = (np.arange(len(h)) - (len(h) - 1) / 2) / s
        pi_.hline("z", 0, color=GRAY, style="-", width=0.8)
        pi_.line("p", t1, ptx / ptx.max(), color=NAVY, width=2.4, name="transmit pulse p(t)")
        pi_.line("h", t2, h[::-1] / np.max(np.abs(h)), color=RED, width=2.0, style="--",
                 name="receive filter h(−t)")
        # spectra
        nfft = 8192
        f = np.fft.rfftfreq(nfft, 1 / s)
        P = np.abs(np.fft.rfft(ptx, nfft)) ** 2
        H = np.abs(np.fft.rfft(h, nfft)) ** 2
        pf = self.plot("freq")
        pf.line("H", f, 10 * np.log10(H / H.max() + 1e-12), color=RED, width=1.8, fill=-60,
                fill_alpha=0.10, name="|H(f)|²: noise let in")
        pf.line("P", f, 10 * np.log10(P / P.max() + 1e-12), color=NAVY, width=2.2,
                name="|P(f)|²: signal")
        # a noisy burst through the filter
        rng = np.random.default_rng(self.noise_seed)
        x = cl.shape(self.burst, ptx, s).real
        sigma = np.sqrt(1 / (2 * 10 ** (p.esn0 / 10)))
        noisy = x + sigma * rng.standard_normal(len(x))
        yc = np.convolve(x, h) / g[d]
        yn = np.convolve(noisy, h) / g[d]
        t = (np.arange(len(yc)) - d) / s
        po = self.plot("out")
        po.line("yn", t, yn, color=GRAY, width=1.0, alpha=0.9, name="with noise")
        po.line("yc", t, yc, color=NAVY, width=2.0, name="noise-free")
        k = np.arange(len(self.burst))
        po.scatter("ideal", k, self.burst, color=GREEN, size=10, outline=GREEN, name="symbol sent")
        po.scatter("samp", k, yn[d + k * s], color=RED, size=7, name="sampled")
        # budget bars
        pb = self.plot("bud")
        vals = [p.esn0, 10 * np.log10(snr), 10 * np.log10(sinr)]
        pb.bars("b", [0, 1, 2], vals, width=0.6, colors=[GREEN, NAVY, RED])
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.1f} dB", anchor=(0.5, 1.05), bold=True)
        pb.set_xticks([(0, "matched"), (1, "yours: noise"), (2, "yours: noise+ISI")])
        pb.set_ylim(min(0, min(vals)) - 1, max(vals) * 1.25 + 2)
        pb.set_xlim(-0.6, 2.6)
        self.readout(loss=loss, isi=isi, sinr=10 * np.log10(sinr), ber=ber)

    def story(self, p):
        loss = self.r.get("loss", 0)
        s = ("<p>Noise is white: it has the same strength at every frequency. The signal is not: "
             "its energy lives where |P(f)|² is big. The best receive filter therefore weights each "
             "frequency by how much signal it carries: <b>H(f) = P*(f)</b>. In time, that is the "
             "transmit pulse reversed, h(t) = p(−t): the <b>matched filter</b>.</p>")
        if p.rx.startswith("Matched"):
            s += (f"<p>Matched: the dashed red filter is the blue pulse's mirror image. Every bit of "
                  f"signal energy is collected and no extra noise is admitted, so the SNR at the "
                  f"sampler equals Es/N0 = {v(p.esn0, '.1f', 'dB')}. Nothing can do better "
                  f"(Cauchy–Schwarz).</p>")
        else:
            s += (f"<p>Your filter delivers {v(loss, '.2f', 'dB')} relative to the matched bound. "
                  + ("A wide filter lets in noise from frequencies with no signal (red shading "
                     "beyond the blue curve). " if loss < -0.5 else
                     "That is close: near-matched filters cost very little SNR. ")
                  + "Watch the residual ISI too: a filter can be good for noise and bad for "
                    "ISI, because only the matched RRC pair forms a Nyquist pulse.</p>")
        return "<h3>Collect the signal, reject the noise</h3>" + s + keybox(
            "Matched filter output SNR = 2E/N₀, the best possible. RRC at both ends is matched "
            "<i>and</i> Nyquist at once.")


# =============================================================================== 5. TX/RX chains
class FilterChains(Experiment):
    title = "Matched and mismatched chains"
    blurb = "Mix and match transmit and receive filters; watch the constellation."
    book = "sec:ch08:rc"
    controls = [
        Heading("Transmitter"),
        Choice("txf", "Transmit filter", ["RRC", "RC", "Rectangular"]),
        Slider("btx", "TX roll-off", 0.05, 1.0, 0.35, step=0.01,
               enabled_if=lambda p: p.txf != "Rectangular"),
        Heading("Receiver"),
        Choice("rxf", "Receive filter", ["RRC", "RC", "Nothing", "Integrate & dump"]),
        Toggle("lock", "Same roll-off as the transmitter", True),
        Slider("brx", "RX roll-off", 0.05, 1.0, 0.20, step=0.01,
               enabled_if=lambda p: not p.lock and p.rxf in ("RRC", "RC")),
        Heading("Signal"),
        Choice("mod", "Modulation", ["QPSK", "16-QAM", "64-QAM"]),
        Slider("esn0", "Es/N0", 10.0, 50.0, 35.0, step=0.5, unit="dB"),
    ]
    plots = [
        ConstellationPlot("const", "Constellation at the sampler", lim=1.45),
        Plot("pulse", "End-to-end pulse g(t) = transmit ∗ receive", x="time (symbol periods T)",
             y="g(t) / g(0)", xlim=(-5.5, 5.5), ylim=(-0.3, 1.15)),
        EyePlot("eye", "Eye (in-phase rail)", yrange=(-1.6, 1.6)),
    ]
    layout = [["const", "pulse"], ["const", "eye"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("isi", "Residual ISI", "dB", ".1f", good=lambda x: x < -30),
        Readout("loss", "SNR loss vs matched", "dB", ".1f", good=lambda x: x > -0.5),
        Readout("mer", "MER", "dB", ".1f", good=lambda x: x > 30),
        Readout("evm", "EVM", "%", ".2f"),
    ]
    challenges = [
        Challenge("Find a pair that is Nyquist (ISI below −40 dB) but NOT matched (loss worse than 1 dB).",
                  lambda s: s.r.isi < -40 and s.r.loss < -1),
        Challenge("Unlock the roll-offs: with 64-QAM, let RX β differ from TX β by at least 0.1 and "
                  "keep MER above 30 dB.",
                  lambda s: (s.p.mod == "64-QAM" and not s.p.lock and s.p.txf == "RRC"
                             and s.p.rxf == "RRC" and abs(s.p.brx - s.p.btx) >= 0.099 and s.r.mer > 30)),
        Challenge("The classic bug: RC at both ends. Watch 16-QAM fall apart (MER below 20 dB "
                  "at Es/N0 ≥ 40 dB).",
                  lambda s: s.p.txf == "RC" and s.p.rxf == "RC" and s.p.mod == "16-QAM"
                  and s.p.esn0 >= 40 and s.r.mer < 20),
    ]
    sps = 8
    nsym = 1500

    def filt(self, kind, beta):
        s = self.sps
        if kind == "RRC":
            return rrc(round(beta, 3), s, 16)
        if kind == "RC":
            return rc_unit(round(beta, 3), s, 16)
        if kind in ("Rectangular", "Integrate & dump"):
            return np.ones(s) / np.sqrt(s)
        return np.array([1.0])

    def update(self, p):
        s = self.sps
        con = cl.get_constellation({"QPSK": "qpsk", "16-QAM": "16qam", "64-QAM": "64qam"}[p.mod])
        a = con.points[self.rng.integers(0, con.M, self.nsym)]
        tx = self.filt(p.txf, p.btx)
        rx = self.filt(p.rxf, p.btx if p.lock else p.brx)
        g = np.convolve(tx, rx)
        d = int(np.argmax(np.abs(g)))
        x = cl.shape(a, tx, s)
        n0 = 10 ** (-p.esn0 / 10)
        x = x + np.sqrt(n0 / 2) * (self.rng.standard_normal(len(x)) + 1j * self.rng.standard_normal(len(x)))
        y = np.convolve(x, rx) / g[d]
        z = y[d::s][:self.nsym]
        err = z[20:-20] - a[20:-20]
        mer = 10 * np.log10(1 / np.mean(np.abs(err) ** 2))
        idx = np.arange(d % s, len(g), s)
        isi = 10 * np.log10((np.sum(np.abs(g[idx]) ** 2) - abs(g[d]) ** 2) / abs(g[d]) ** 2 + 1e-12)
        loss = 10 * np.log10(abs(g[d]) ** 2 / (np.sum(tx ** 2) * np.sum(rx ** 2)))
        pc = self.plot("const")
        pc.points("z", z[20:-20], color=NAVY, size=3, alpha=0.5)
        pc.ideal("ideal", con.points, color=RED)
        pp = self.plot("pulse")
        t = (np.arange(len(g)) - d) / s
        gn = g / g[d]
        pp.hline("z", 0, color=GRAY, style="-", width=0.8)
        pp.line("g", t, gn, color=NAVY, width=2.2)
        k = np.arange(-5, 6)
        gk = np.interp(k, t, gn)
        pp.stems("isi", k[k != 0], gk[k != 0], color=RED, size=7, name="ISI at other instants")
        pp.stems("cur", [0], [1.0], color=GREEN, size=9, name="main cursor")
        pe = self.plot("eye")
        pe.eye("eye", y.real[:d + (self.nsym - 20) * s] / np.max(np.abs(con.points.real)), s,
               n_sym=2, offset=d - s + 20 * s, yrange=(-1.6, 1.6))
        self.readout(isi=isi, loss=loss, mer=mer, evm=100 * np.sqrt(np.mean(np.abs(err) ** 2)))

    def story(self, p):
        isi, loss = self.r.get("isi", 0), self.r.get("loss", 0)
        nyq = isi < -40
        matched = loss > -0.3
        if nyq and matched:
            verdict = good("Nyquist and matched: the textbook optimum.")
        elif nyq:
            verdict = ("Nyquist (no ISI) but " + bad("not matched") +
                       f": you lose {v(-loss, '.1f', 'dB')} of SNR to extra noise.")
        elif matched:
            verdict = "Matched for noise but " + bad("not Nyquist") + ": ISI blurs every point."
        else:
            verdict = bad("Neither matched nor Nyquist.")
        s = (f"<p>{verdict}</p>"
             "<p>The top-right plot is the whole chain's pulse. Zero-ISI needs it to cross zero "
             "at every other symbol instant (the red stems). The cloud size in the constellation "
             "is noise plus that ISI; MER measures both together.</p>")
        if p.rxf == "Nothing":
            s += ("<p>With no receive filter, all the noise in the simulation bandwidth "
                  f"({self.sps}× the symbol rate) reaches the sampler: hence the SNR loss.</p>")
        return "<h3>Two filters, one pulse</h3>" + s + keybox(
            "Split the raised cosine: RRC at the transmitter, the same RRC at the receiver.")


# =============================================================================== 6. BER Monte Carlo
class BERMonteCarlo(Experiment):
    title = "BER: simulation meets theory"
    blurb = "A live Monte Carlo run against the matched-filter theory curve."
    book = "sec:ch08:binary"
    heavy = True
    controls = [
        Choice("mod", "Modulation", ["2-PAM (BPSK)", "QPSK", "4-PAM", "16-QAM"]),
        Choice("rx", "Receiver", ["Matched RRC", "Integrate & dump"]),
        Slider("eps", "Timing offset", 0.0, 0.5, 0.0, step=0.01, unit="T"),
        Slider("beta", "Roll-off β", 0.1, 1.0, 0.35, step=0.05),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [BERPlot("ber", "Bit error rate vs Eb/N0", x="Eb/N0 (dB)", ylim=(1e-6, 0.5),
                     xlim=(-0.5, 16.5), legend="bl")]
    readouts = [
        Readout("penalty", "Penalty at BER 10⁻³", "dB", ".2f", good=lambda x: x < 0.3),
        Readout("bits", "Bits simulated", "", "int"),
        Readout("points", "Points done", "", None),
    ]
    challenges = [
        Challenge("Watch simulation land on theory: QPSK, matched filter, perfect timing, penalty under 0.3 dB.",
                  lambda s: s.p.mod == "QPSK" and s.p.rx == "Matched RRC" and s.p.eps == 0
                  and s.r.penalty is not None and s.r.penalty < 0.3),
        Challenge("Find a timing offset that costs about 1 dB (0.7–1.3 dB) at BER 10⁻³.",
                  lambda s: s.r.penalty is not None and 0.7 <= s.r.penalty <= 1.3 and s.p.eps > 0
                  and s.p.rx == "Matched RRC"),
        Challenge("Measure the integrate-and-dump receiver's penalty (more than 0.5 dB).",
                  lambda s: s.p.rx == "Integrate & dump" and s.r.penalty is not None
                  and s.r.penalty > 0.5 and s.p.eps == 0),
    ]
    sps = 16

    def setup(self):
        self.ghost = None
        self.run_id = 0

    def on_rerun(self, p):
        self.run_id += 1

    def spec(self, mod):
        return {"2-PAM (BPSK)": ("bpsk", 1), "QPSK": ("qpsk", 2), "4-PAM": ("4pam", 2),
                "16-QAM": ("16qam", 4)}[mod]

    def theory(self, mod, ebn0):
        g = 10 ** (ebn0 / 10)
        if mod in ("2-PAM (BPSK)", "QPSK"):
            return cl.ber_bpsk(ebn0)
        if mod == "4-PAM":
            return 0.75 * cl.qfunc(np.sqrt(0.8 * g))
        return cl.ber_mqam_gray(ebn0 + 10 * np.log10(4), 16)

    def update(self, p):
        if getattr(self, "sim", None) and self.sim["done"] and len(self.sim["x"]) > 2:
            self.ghost = dict(self.sim)
        self.sim = dict(x=[], y=[], bits=0, done=False, cur=None)
        x = np.linspace(0, 16, 161)
        pb = self.plot("ber")
        pb.theory("th", x, self.theory(p.mod, x), name="theory (matched, perfect timing)")
        if self.ghost is not None:
            pb.sim("ghost", self.ghost["x"], self.ghost["y"], color=GRAY, size=7,
                   name="previous run")
        pb.hline("e3", 1e-3, color=GRAY, style=":", width=0.8)
        self.readout(penalty=None, bits=0, points="0")

    def background(self, p):
        name, k = self.spec(p.mod)
        con = cl.get_constellation(name)
        s = self.sps
        h = rrc(round(p.beta, 3), s, 12)
        rx = h if p.rx == "Matched RRC" else np.ones(s) / np.sqrt(s)
        g = np.convolve(h, rx)
        d = int(np.argmax(np.abs(g))) + int(round(p.eps * s))
        thorough = p.effort == "Thorough" and not self.quick
        target = 30 if self.quick else (300 if thorough else 100)
        max_bits = 4e4 if self.quick else (3e6 if thorough else 4e5)
        floor = 1e-4 if self.quick else (2e-6 if thorough else 1e-5)
        rng = np.random.default_rng(self.run_id * 7919 + 13)
        real = con.points.imag.max() < 1e-9
        for ebn0 in np.arange(0, 17, 1.0):
            esn0 = ebn0 + 10 * np.log10(k)
            n0 = 10 ** (-esn0 / 10)
            errs = bits = 0
            while errs < target and bits < max_bits:
                nsym = 8000
                b = cl.random_bits(k * nsym, rng)
                a = con.modulate(b)
                xx = cl.shape(a, h, s)
                noise = rng.standard_normal(len(xx)) + 1j * rng.standard_normal(len(xx))
                if real:
                    noise = noise.real + 0j          # real signalling: noise in one dimension
                yy = fftconvolve(xx + np.sqrt(n0 / 2) * noise, rx)
                z = yy[d::s][:nsym] / g[np.argmax(np.abs(g))]
                bh = con.demodulate(z[10:-10])
                errs += int(np.sum(bh != b[k * 10:k * (nsym - 10)]))
                bits += len(bh)
                yield dict(ebn0=ebn0, ber=errs / bits, bits=bits, final=False)
            ber = errs / bits
            yield dict(ebn0=ebn0, ber=ber, bits=bits, final=True)
            if ber < floor or errs == 0:
                break
        yield dict(done=True)

    def progress(self, p, it):
        sim = self.sim
        if it.get("done"):
            sim["done"] = True
        elif it["final"]:
            sim["x"].append(it["ebn0"])
            sim["y"].append(max(it["ber"], 1e-9))
            sim["bits"] += it["bits"]
            sim["cur"] = None
        else:
            sim["cur"] = (it["ebn0"], max(it["ber"], 1e-9), it["bits"])
        pb = self.plot("ber")
        if sim["x"]:
            pb.sim("sim", sim["x"], sim["y"], color=RED, name="simulation")
        if sim["cur"] is not None:
            pb.scatter("cur", [sim["cur"][0]], [sim["cur"][1]], color=ORANGE, size=12,
                       symbol="d")
        elif pb.item("cur") is not None:
            pb.item("cur").setVisible(False)
        pen = self.penalty(p)
        cur_bits = sim["bits"] + (sim["cur"][2] if sim["cur"] else 0)
        self.readout(penalty=pen, bits=cur_bits,
                     points=f"{len(sim['x'])}" + (" ✓" if sim["done"] else " …"))

    def penalty(self, p):
        x, y = np.array(self.sim["x"]), np.array(self.sim["y"])
        if len(x) < 2 or y.min() > 1e-3 or y.max() < 1e-3:
            return None
        ly = np.log10(y)
        i = np.where(ly <= -3)[0][0]
        if i == 0:
            return None
        x3 = np.interp(-3, [ly[i], ly[i - 1]], [x[i], x[i - 1]])
        xt = np.linspace(0, 16, 1601)
        t3 = np.interp(-3, np.log10(self.theory(p.mod, xt))[::-1], xt[::-1])
        return float(x3 - t3)

    def story(self, p):
        pen = self.r.get("penalty")
        s = ("<p>Each red circle is a real experiment: random bits, pulse shaping, Gaussian noise, "
             "the receive filter, a sampler and a decision. The orange diamond is the point being "
             "simulated now; it settles as errors accumulate. A point is finished after "
             "enough errors (100 for Quick) for a trustworthy estimate.</p>")
        if pen is not None:
            s += (f"<p>At BER 10⁻³ your receiver needs {v(pen, '.2f', 'dB')} more Eb/N0 than the "
                  f"ideal matched-filter receiver.</p>")
        if p.eps > 0:
            s += ("<p>The timing offset makes neighbours leak into each sample (ISI) and lowers the "
                  "wanted sample's amplitude: both cost Eb/N0.</p>")
        if p.mod == "QPSK":
            s += ("<p>QPSK has the same BER versus Eb/N0 as BPSK: it is two BPSKs on cosine and "
                  "sine, sharing the bandwidth.</p>")
        return "<h3>The number that matters</h3>" + s + keybox(
            "Change any control and the run restarts. The previous run stays in grey for comparison.")


# =============================================================================== 7. NRZ vs PAM-4
class NRZvsPAM4(Experiment):
    title = "NRZ versus PAM-4"
    blurb = "Same bit rate, same lossy backplane: two bits per symbol or one?"
    book = "sec:ch08:mpam"
    animate = True
    fps = 10
    controls = [
        Slider("loss", "Channel loss at Rb/2", 0.0, 40.0, 18.0, step=0.5, unit="dB",
               help="Insertion loss of the backplane at the NRZ Nyquist frequency"),
        Slider("snr", "Peak signal / rms noise", 15.0, 45.0, 30.0, step=0.5, unit="dB"),
        Slider("jitter", "Random jitter (rms)", 0.0, 6.0, 0.0, step=0.1, unit="% of a bit",
               help="The same jitter in seconds is a smaller fraction of a PAM-4 symbol"),
        Toggle("ffe", "5-tap feed-forward equaliser", True,
               help="Symbol-spaced FFE (1 pre-cursor, 3 post-cursor taps), zero-forcing design"),
        Toggle("persist", "Persistence", True),
    ]
    plots = [
        EyePlot("nrz", "NRZ eye (1 bit per symbol)", yrange=(-1.5, 1.5)),
        EyePlot("pam", "PAM-4 eye (2 bits per symbol)", yrange=(-1.5, 1.5)),
        SpectrumPlot("chan", "Backplane insertion loss", x="frequency (multiples of the bit rate Rb)",
                     y="|H(f)| (dB)", xlim=(0, 1.0), ylim=(-50, 2), legend=None),
    ]
    layout = [["nrz", "pam"], ["chan", "chan"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("nrz", "NRZ eye opening", "% of swing", ".0f"),
        Readout("pam", "PAM-4 eye opening", "% of swing", ".0f"),
        Readout("lpam", "Loss at Rb/4", "dB", ".1f"),
        Readout("win", "Wider eye", "", None),
    ]
    challenges = [
        Challenge("With the equaliser on, raise the loss until PAM-4's eye overtakes NRZ's.",
                  lambda s: s.r.win == "PAM-4" and s.p.ffe),
        Challenge("Close the NRZ eye completely while the PAM-4 eye stays open.",
                  lambda s: s.r.nrz <= 0 and s.r.pam > 0),
        Challenge("Switch the equaliser off: how much loss can bare NRZ take? Keep it open at 10 dB or more.",
                  lambda s: not s.p.ffe and s.p.loss >= 10 and s.r.nrz > 0),
    ]
    sps_bit = 16
    nbits = 1200

    def chan(self, loss):
        key = round(loss, 2)
        if getattr(self, "_ck", None) != key:
            self._h = lc.backplane(self.sps_bit, loss, 0.5, nfft=1 << 13, length=40 * self.sps_bit,
                                   skin_frac=0.15)
            self._ck = key
        return self._h

    @staticmethod
    def ffe(x, w, s):
        """Symbol-spaced FFE with one pre-cursor tap: sum_i w_i x[n - (i-1) s]. The cursor
        stays where it was, so the sampling index of the unequalised pulse still applies."""
        out = np.zeros(len(x))
        for i, wi in enumerate(w):
            sh = (i - 1) * s
            if sh >= 0:
                out[sh:] += wi * x[:len(x) - sh]
            else:
                out[:sh] += wi * x[-sh:]
        return out

    def link(self, p, M):
        s = self.sps_bit * (1 if M == 2 else 2)
        nsym = self.nbits // (1 if M == 2 else 2)
        lev = np.linspace(-1, 1, M)
        a = lev[self.rng.integers(0, M, nsym)]
        h = self.chan(p.loss)
        y = fftconvolve(np.repeat(a, s), h)[:nsym * s]
        y = y + 10 ** (-p.snr / 20) * self.rng.standard_normal(len(y))   # noise at the receiver
        pr = np.convolve(h, np.ones(s))                                   # single-symbol response
        if p.ffe:
            w, _, _ = lc.ffe_zf(pr, s, ntaps=5, pre=1)
            y, pr = self.ffe(y, w, s), self.ffe(pr, w, s)
        d = int(np.argmax(pr))
        return y, a, d, s, lev            # volts as received: long runs settle at ±1

    def measure(self, y, a, d, s, lev, jit):
        m = eye_metrics(y, a, d, s, lev)
        if jit > 0:
            ph = m["phases"]
            win = np.abs(ph[:, None] - ph[None, :]) <= 2.5 * jit
            op = np.where(win, m["open"][None, :], np.inf).min(axis=1)
            m["height"] = float(op.max())
        # absolute opening as a fraction of the full swing (2)
        return m["height"] * (lev[1] - lev[0]) / 2

    def draw(self, p, accumulate):
        out = {}
        for key, M in (("nrz", 2), ("pam", 4)):
            y, a, d, s, lev = self.link(p, M)
            jit = p.jitter / 100 / (1 if M == 2 else 2)        # in this scheme's UI
            pe = self.plot(key)
            pe.eye(key, y, s, n_sym=2, offset=d - s + 30 * s, yrange=(-1.5, 1.5),
                   accumulate=accumulate and p.persist, decay=0.85, jitter=jit, rng=self.rng)
            out[key] = self.measure(y, a, d, s, lev, jit)
        h = self.chan(p.loss)
        nfft = 1 << 13
        f = np.fft.rfftfreq(nfft, 1 / self.sps_bit)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, nfft)) + 1e-9)
        pc = self.plot("chan")
        pc.line("H", f, H, color=NAVY, width=2.2)
        lq = float(np.interp(0.25, f, H))
        lh = float(np.interp(0.5, f, H))
        pc.vline("q", 0.25, color=PURPLE, style="--", label=f"PAM-4 Nyquist: {lq:.1f} dB", label_pos=0.15)
        pc.vline("h", 0.5, color=ORANGE, style="--", label=f"NRZ Nyquist: {lh:.1f} dB", label_pos=0.35)
        win = "PAM-4" if out["pam"] > out["nrz"] else "NRZ"
        if out["pam"] <= 0 and out["nrz"] <= 0:
            win = "neither"
        self.readout(nrz=100 * out["nrz"], pam=100 * out["pam"], lpam=-lq, win=win)

    def update(self, p):
        for k in ("nrz", "pam"):
            self.plot(k).reset_persistence()
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        nrz, pam = self.r.get("nrz", 0), self.r.get("pam", 0)
        s = ("<p>Both links carry the same bit rate over the same backplane. NRZ sends one bit per "
             "symbol, so it needs twice the symbol rate and sees the loss at Rb/2. PAM-4 sends two "
             "bits per symbol at half the rate, so it only sees the (smaller) loss at Rb/4.</p>"
             "<p>The price: PAM-4 squeezes three eyes into the same voltage swing, each a third "
             "the height. That is a 9.5 dB SNR penalty before the channel does anything.</p>")
        if pam > nrz:
            s += (f"<p>At {v(p.loss, '.1f', 'dB')} of loss {good('PAM-4 wins')}: to undo the loss at "
                  f"Rb/2 the equaliser must boost high frequencies so hard that it amplifies the noise "
                  f"more than PAM-4's level penalty costs. This is why 400G Ethernet, PCIe 6.0 and "
                  f"GDDR6X moved to PAM-4.</p>")
        elif p.ffe:
            s += ("<p>The equaliser subtracts scaled copies of the neighbouring symbols, cancelling "
                  "ISI, but every dB of high-frequency boost also boosts noise. Raise the loss and "
                  "watch which eye survives longer.</p>")
        else:
            s += ("<p>Without an equaliser the ISI closes both eyes quickly. Real SerDes receivers "
                  "never run without one: switch it back on.</p>")
        return "<h3>Two bits per symbol or one?</h3>" + s + keybox(
            "Multilevel signalling trades SNR for bandwidth: worth it when the channel's loss "
            "grows faster with frequency than the noise does.")


# =============================================================================== 8. FTN
class FasterThanNyquist(Experiment):
    title = "Faster than Nyquist"
    blurb = "Pack the pulses closer than Nyquist allows and see what you pay."
    book = "sec:ch08:ftn"
    animate = True
    fps = 10
    controls = [
        Slider("tau", "Symbol spacing τ", 0.5, 1.0, 1.0, step=0.01, unit="T",
               help="Symbols are sent every τT instead of every T"),
        Slider("beta", "Roll-off β", 0.1, 1.0, 0.3, step=0.05),
        Slider("esn0", "Es/N0", 5.0, 40.0, 40.0, step=0.5, unit="dB"),
    ]
    plots = [
        EyePlot("eye", "Binary eye with symbols every τT", yrange=(-2.0, 2.0)),
        BarPlot("taps", "Neighbour leakage g(kτT)", x="neighbour k", y="contribution",
                xlim=(-6.6, 6.6), ylim=(-0.4, 1.1)),
    ]
    layout = [["eye", "taps"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("gain", "Extra throughput", "%", ".0f"),
        Readout("height", "Eye height", "%", ".0f", good=lambda x: x > 0),
        Readout("isi", "Worst-case ISI", "%", ".0f"),
    ]
    challenges = [
        Challenge("Squeeze in 20 % more data (τ ≤ 0.83) and look at the eye.",
                  lambda s: s.p.tau <= 0.834),
        Challenge("Find the spacing where the binary eye just closes (height ≤ 0) with no noise.",
                  lambda s: s.r.height <= 0 and s.p.esn0 >= 35),
    ]
    base = 32

    def draw(self, p, acc):
        step = int(round(self.base * p.tau))
        h = rrc(round(p.beta, 3), self.base, 16)
        g = np.convolve(h, h)
        c = len(g) // 2
        n = 600
        a = np.sign(self.rng.standard_normal(n))
        up = np.zeros(n * step)
        up[::step] = a
        x = fftconvolve(up, h)
        x = x + np.sqrt(1 / (2 * 10 ** (p.esn0 / 10))) * self.rng.standard_normal(len(x))
        y = fftconvolve(x, h)
        d = len(h) - 1
        pe = self.plot("eye")
        pe.eye("eye", y[:d + (n - 20) * step], step, n_sym=2, offset=d - step + 20 * step,
               yrange=(-2.0, 2.0),
               accumulate=acc, decay=0.85)
        k = np.arange(-6, 7)
        gk = np.interp(c + k * step, np.arange(len(g)), g) / g[c]
        lev = np.array([-1.0, 1.0])
        m = eye_metrics(y, a, d, step, lev, nphase=32)
        isi = (np.sum(np.abs(np.interp(c + np.arange(-40, 41) * step, np.arange(len(g)), g,
                                       left=0, right=0))) - g[c]) / g[c]
        pb = self.plot("taps")
        pb.bars("b", k, gk, width=0.6, colors=[NAVY if x == 0 else RED for x in k])
        pb.set_xticks([(x, str(x)) for x in k])
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        self.readout(gain=100 * (1 / p.tau - 1), height=100 * m["height"], isi=100 * isi)

    def update(self, p):
        self.plot("eye").reset_persistence()
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        s = (f"<p>You are sending a symbol every {v(p.tau)} T: {v(100 * (1 / p.tau - 1), '.0f')} % "
             f"more symbols per second in exactly the same bandwidth. Nyquist said you cannot do "
             f"that <i>without ISI</i>, and the red bars show the ISI arriving.</p>"
             "<p>Mazo (1975) found the surprise: for binary sinc pulses you can go down to "
             "τ ≈ 0.802 before the minimum distance between sequences shrinks, so a maximum-likelihood "
             "sequence detector loses nothing, even though the eye looks closed.</p>")
        return "<h3>Breaking the speed limit</h3>" + s + keybox(
            "Faster-than-Nyquist trades receiver complexity (a trellis or iterative detector) for "
            "throughput: a 6G candidate idea.")


# =============================================================================== the lab
LAB = st.Lab(3, "Pulse Shaping, Nyquist and the Eye", chapter=8,
             chapter_title="Baseband Transmission and Pulse Shaping",
             experiments=[PulseSpectra, NyquistZeroISI, LiveEye, MatchedFilter, FilterChains,
                          BERMonteCarlo, NRZvsPAM4, FasterThanNyquist])

if __name__ == "__main__":
    st.run(LAB)
