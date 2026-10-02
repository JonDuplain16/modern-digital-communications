"""Lab 07 · OFDM: From the IFFT to the 5G NR Resource Grid   (Chapter 17)

Run it:      python labs/lab07_ofdm.py
Self-test:   python labs/lab07_ofdm.py --selftest

A wideband channel is hard to equalise. OFDM's answer is not to have a wideband signal at all:
split the band into hundreds of narrow subcarriers, each so narrow that the channel looks flat
across it, and let an FFT do the work. Eight experiments build the idea from overlapping sincs
to the 5G NR numerologies, breaking it on purpose along the way (short prefixes, frequency
offsets, sparse pilots, power-amplifier peaks) and fixing it again.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import ofdm as co
from commlib import ofdmadv as oa
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, BarPlot, ImagePlot,
                    Readout, Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

# =============================================================================== shared helpers
FS = oa.FS_LTE                 # 15.36 MS/s: the LTE / NR 10 MHz sampling rate
NFFT, NUSED = 1024, 600        # 1024-point FFT, 600 active 15 kHz subcarriers (9 MHz)
CFG = co.OFDMConfig(NFFT, NUSED, 72)
KSIG = CFG.k.astype(float)     # signed subcarrier indices (DC left empty)
QPSK, Q16, Q64 = (cl.get_constellation(n) for n in ("qpsk", "16qam", "64qam"))
CONS = {"QPSK": QPSK, "16-QAM": Q16, "64-QAM": Q64}
# Es/N0 that gives an uncoded symbol error rate near 10^-3 (rule of thumb for the threshold
# lines in the plots): QPSK ~ 9.8 dB, 16-QAM ~ 16.5 dB, 64-QAM ~ 22.5 dB.
THRESH = {"QPSK": 9.8, "16-QAM": 16.5, "64-QAM": 22.5}


def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-30))


def qam(con, n, rng):
    return con.modulate(cl.random_bits(con.k * n, rng))


def channel_taps(kind, seed, d2_us=3.0, a2_db=-3.0):
    """A static unit-energy channel: a two-ray echo or one draw of an LTE TDL profile."""
    rng = np.random.default_rng(seed)
    if kind == "Two-ray":
        d = int(round(d2_us * 1e-6 * FS))
        h = np.zeros(d + 1, complex)
        h[0] = 1.0
        h[d] += 10 ** (a2_db / 20) * np.exp(2j * np.pi * rng.random())
    elif kind == "Flat":
        h = np.array([np.exp(2j * np.pi * rng.random())])
    else:
        h = oa.tdl_static_taps(kind, FS, rng)
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def max_delay_us(kind, d2_us=3.0):
    if kind == "Two-ray":
        return round(d2_us * 1e-6 * FS) / FS * 1e6
    if kind == "Flat":
        return 0.0
    return cl.TDL_PROFILES[kind][0][-1] / 1e3


def ccdf_fast(vals, grid):
    s = np.sort(np.asarray(vals))
    return 1.0 - np.searchsorted(s, grid, side="right") / max(len(s), 1)


def papr_at(grid, cc, p=1e-3):
    """PAPR0 where the CCDF crosses p (linear interpolation on log10)."""
    lc = np.log10(np.maximum(cc, 1e-9))
    if cc[0] < p:
        return float(grid[0])
    i = np.argmax(cc < p)
    if i == 0:
        return float(grid[-1])
    return float(np.interp(np.log10(p), [lc[i], lc[i - 1]], [grid[i], grid[i - 1]]))


# =============================================================================== 1. orthogonality
class Orthogonality(Experiment):
    title = "Orthogonal subcarriers"
    blurb = "Spectra that overlap completely, yet do not interfere. Until the receiver drifts."
    book = "sec:ch17:dft"
    controls = [
        Heading("The comb of subcarriers"),
        IntSlider("K", "Subcarriers shown", 2, 8, 5),
        Slider("eps", "Receiver frequency offset ε", -0.5, 0.5, 0.0, step=0.005, unit="× Δf",
               help="How far the receiver's FFT grid is shifted from the transmitter's, in "
                    "subcarrier spacings (a carrier frequency offset)"),
        Heading("One correlation, in time"),
        IntSlider("k1", "Transmitted subcarrier", 1, 6, 2,
                  help="Subcarrier k completes exactly k cycles in one symbol"),
        IntSlider("k2", "Receiver's template subcarrier", 1, 6, 3,
                  help="The FFT bin the receiver is computing: a correlation with this sinusoid"),
    ]
    plots = [
        Plot("spec", "Subcarrier spectra and what the FFT samples",
             x="frequency (multiples of Δf = 1/T)", y="amplitude", xlim=(-1.6, 8.6),
             ylim=(-0.4, 1.45), legend="tl", legend_cols=3),
        Plot("time", "Correlating over one symbol: Re{ transmitted × template* }",
             x="time (fraction of the symbol T)", y="product", xlim=(0, 1), ylim=(-1.25, 1.6),
             legend="tl", legend_cols=2),
        BarPlot("leak", "Where a subcarrier's power lands", x="FFT bin, relative to the wanted one",
                y="power (dB)", xlim=(-6.6, 6.6), ylim=(-60, 4)),
    ]
    layout = [["spec", "spec"], ["time", "leak"]]
    row_stretch = [5, 4]
    readouts = [
        Readout("sir", "Signal-to-ICI ratio", "dB", ".1f", good=lambda x: x > 30,
                help="Wanted power over the power leaked in from all the other subcarriers"),
        Readout("loss", "Wanted sample", "dB", ".2f", help="Loss of the wanted subcarrier: sinc²(ε)"),
        Readout("nb", "Leak into the nearest bin", "dB", ".1f"),
        Readout("corr", "Template correlation", "%", ".0f",
                help="|average of transmitted × template*| over one symbol"),
    ]
    challenges = [
        Challenge("Find the offset at which the nearest neighbour receives exactly −20 dB "
                  "(within ±0.3 dB).",
                  lambda s: abs(s.r.nb + 20) < 0.3,
                  hint="Leakage into bin ±1 is sinc²(1 − |ε|). Small offsets are enough."),
        Challenge("Find the largest offset (to the nearest 0.005) that keeps the signal-to-ICI "
                  "ratio at 30 dB or better.",
                  lambda s: s.r.sir >= 30 and abs(s.p.eps) >= 0.0149,
                  hint="SIR ≈ 3/(πε)²: about 1.7 % of a subcarrier spacing."),
        Challenge("Correlate two different subcarriers and break their orthogonality: make the "
                  "template correlation exceed 50 %.",
                  lambda s: s.p.k1 != s.p.k2 and s.r.corr > 50,
                  hint="Different whole numbers of cycles average to zero, unless the receiver's "
                       "frequency is off."),
    ]

    def update(self, p):
        K, eps = int(p.K), p.eps
        f = np.linspace(-1.6, 8.6, 1400)
        ps = self.plot("spec")
        cols = [NAVY, BLUE, TEAL, GREEN, GOLD, ORANGE, PURPLE, GRAY]
        total = np.zeros_like(f)
        for k in range(K):
            s = np.sinc(f - k)
            total += s
            ps.line(f"s{k}", f, s, color=cols[k % len(cols)], width=1.6, alpha=0.85,
                    name="subcarrier spectra" if k == 0 else None)
        ps.hline("z", 0, color=GRAY, style="-", width=0.7)
        m = np.arange(-1, K + 1)
        for i, mm in enumerate(m):
            ps.vline(f"g{i}", mm + eps, color=GRAY, style=":", width=0.9)
        fs_ = m + eps
        samples = np.array([np.sum(np.sinc(x - np.arange(K))) for x in fs_])
        ideal = ((m >= 0) & (m < K)).astype(float)
        ps.scatter("ideal", m, ideal, color=GREEN, size=17, outline=GREEN, name="what was sent")
        ps.scatter("samp", fs_, samples, color=RED, size=8, name="what the FFT sees")
        # time-domain correlation of one transmitted subcarrier with one template
        t = np.linspace(0, 1, 801)
        tx = np.exp(2j * np.pi * p.k1 * t)
        tpl = np.exp(2j * np.pi * (p.k2 + eps) * t)
        prod = tx * np.conj(tpl)
        corr = abs(np.mean(prod[:-1]))
        pt = self.plot("time")
        pt.line("tx", t, tx.real, color=NAVY, width=1.2, alpha=0.55, name=f"subcarrier {p.k1}")
        pt.line("tp", t, tpl.real, color=ORANGE, width=1.2, alpha=0.55,
                name=f"template {p.k2}" + (f" {eps:+.3f}" if eps else ""))
        pt.line("pr", t, prod.real, color=RED, width=2.0, fill=0, fill_alpha=0.18)
        pt.hline("mean", np.mean(prod.real[:-1]), color=GREEN, style="--",
                 label=f"average = {np.mean(prod.real[:-1]):+.2f}", label_pos=0.02)
        # leakage of one subcarrier into the bins around it (Dirichlet kernel, N = 64)
        Nn = 64
        el = np.arange(-6, 7)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = np.sin(np.pi * (el + eps)) / (Nn * np.sin(np.pi * (el + eps) / Nn))
        S = np.where(np.isclose(el + eps, 0), 1.0, S)
        L = db(np.abs(S) ** 2)
        pb = self.plot("leak")
        pb.bars("b", el, np.maximum(L, -59), width=0.6, base=-60,
                colors=[NAVY if x == 0 else RED for x in el])
        pb.set_xticks([(x, str(x)) for x in el])
        s2 = np.sinc(eps) ** 2
        sir = float(oa.sir_cfo_db(eps)) if abs(eps) > 1e-6 else None
        nb = float(max(L[5], L[7]))
        self.readout(sir=sir if sir is not None else "∞", loss=10 * np.log10(s2),
                     nb=nb if abs(eps) > 1e-6 else "−∞", corr=100 * corr)

    def story(self, p):
        eps = p.eps
        s = ("<p>Each subcarrier is a sinusoid that completes a <b>whole number of cycles</b> in "
             "one symbol of length T. Chopped to that length, its spectrum is a sinc, and every "
             "sinc passes through zero at all the <i>other</i> multiples of 1/T. Sample the "
             "spectrum exactly there (the FFT does) and each sample sees one subcarrier only: "
             "the spectra overlap completely and still do not interfere. That is "
             "<b>orthogonality</b>, and it is why OFDM needs no guard bands.</p>")
        if abs(eps) < 1e-6:
            s += ("<p>Right now the red samples sit on the green targets. In the time plot, "
                  f"subcarrier {p.k1} against template {p.k2}: "
                  + ("the product is a pure positive signal: correlation 100 %." if p.k1 == p.k2 else
                     f"the product has {abs(p.k1 - p.k2)} whole cycle(s), so it averages to exactly "
                     "zero.") + "</p>")
        else:
            s += (f"<p>Your receiver is {v(abs(eps) * 100, '.1f', '%')} of a subcarrier spacing off. "
                  f"The FFT now samples each sinc off its peak (the wanted sample loses "
                  f"{v(-self.r.get('loss', 0), '.2f', 'dB')}) and on its neighbours' slopes, so "
                  f"every subcarrier leaks into every other: <b>inter-carrier interference</b>. "
                  f"The signal-to-ICI ratio is {v(self.r.get('sir', 0), '.1f', 'dB')} "
                  f"({'enough for 256-QAM' if self.r.get('sir', 0) > 30 else 'too little for dense QAM'}).</p>")
        s += ("<p>For an LTE phone at 2 GHz with 15 kHz spacing, a crystal error of just 0.75 ppm "
              "is already a tenth of a subcarrier. That is why every OFDM receiver estimates and "
              "removes the frequency offset first (Experiment 5).</p>")
        return "<h3>Overlap without interference</h3>" + s + keybox(
            "Spacing 1/T makes the subcarriers orthogonal. A frequency offset ε breaks it: "
            "SIR ≈ 3/(πε)².")


# =============================================================================== 2. symbol builder
class SymbolBuilder(Experiment):
    title = "Building a symbol: IFFT + CP"
    blurb = "Numbers on subcarriers in, one time-domain symbol out; then copy its tail to the front."
    book = "sec:ch17:cp"
    controls = [
        Heading("Frequency domain"),
        Choice("N", "FFT size", ["16", "64", "256"], "64",
               help="802.11a uses 64 at 20 MS/s: 312.5 kHz subcarriers"),
        Choice("data", "What the subcarriers carry",
               ["QPSK", "16-QAM", "One subcarrier", "All in phase", "Zadoff–Chu (CAZAC)"],
               style="menu"),
        IntSlider("hk", "Highlight subcarrier k", 1, 8, 2),
        Button("again", "New random data"),
        Heading("Cyclic prefix"),
        Slider("cp", "Prefix length", 0, 25, 7.0, step=0.5, unit="% of T",
               help="LTE's normal prefix is about 7 % of the symbol, 802.11a's 25 %"),
    ]
    plots = [
        Plot("freq", "Step 1: one complex number per subcarrier (Xₖ)", x="subcarrier index k",
             y="value", ylim=(-1.9, 2.1), legend="tl", legend_cols=3),
        Plot("time", "Step 2–3: IFFT, then the cyclic prefix (real part shown)", x="time (µs)",
             y="amplitude", legend="tl", legend_cols=3),
        Plot("env", "The envelope: instantaneous power relative to the mean", x="time (µs)",
             y="power (dB)", ylim=(-25, 22), legend="tl", legend_cols=2),
    ]
    layout = [["freq", "env"], ["time", "time"]]
    row_stretch = [4, 5]
    readouts = [
        Readout("df", "Subcarrier spacing", "kHz", ".1f"),
        Readout("T", "Useful symbol", "µs", ".2f"),
        Readout("tcp", "Prefix", "µs", ".2f"),
        Readout("papr", "Peak-to-average (this symbol)", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Build 802.11a: a 64-point FFT at 20 MS/s with a 0.8 µs prefix.",
                  lambda s: s.p.N == "64" and abs(s.r.tcp - 0.8) < 0.01),
        Challenge("Build the worst possible symbol: a peak at least 15 dB above the mean.",
                  lambda s: s.r.papr >= 15,
                  hint="What happens at t = 0 if every subcarrier starts with the same phase?"),
        Challenge("Use every subcarrier and still keep the peak within 3 dB of the mean.",
                  lambda s: s.p.data not in ("One subcarrier",) and s.r.papr < 3,
                  hint="Some sequences have a constant magnitude in both domains (used for "
                       "LTE/NR reference signals)."),
    ]
    fs = 20e6
    over = 8

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def make_grid(self, p, Nn, nu, ks):
        rng = np.random.default_rng(self.seed)
        if p.data == "QPSK":
            return qam(QPSK, nu, rng)
        if p.data == "16-QAM":
            return qam(Q16, nu, rng)
        if p.data == "One subcarrier":
            X = np.zeros(nu, complex)
            X[np.argmin(np.abs(ks - min(int(p.hk), nu // 2)))] = np.sqrt(nu)
            return X
        if p.data == "All in phase":
            return np.ones(nu, complex)
        n = np.arange(nu)
        q = 1 if nu % 2 == 0 else 1
        return np.exp(-1j * np.pi * q * n * n / nu) if nu % 2 == 0 else \
            np.exp(-1j * np.pi * q * n * (n + 1) / nu)

    def update(self, p):
        Nn = int(p.N)
        nu = {16: 12, 64: 52, 256: 208}[Nn]
        cfg = co.OFDMConfig(Nn, nu, 0)
        ks = cfg.k
        X = self.make_grid(p, Nn, nu, ks)
        ncp = int(round(p.cp / 100 * Nn))
        M = Nn * self.over
        Xo = np.zeros(M, complex)
        Xo[np.mod(ks, M)] = X
        x = np.fft.ifft(Xo) * M / np.sqrt(nu)                 # unit mean power, oversampled
        ncpo = ncp * self.over
        xcp = np.concatenate([x[M - ncpo:], x]) if ncpo else x
        dt = 1 / self.fs / self.over * 1e6                     # µs per oversampled sample
        t = (np.arange(len(xcp)) - ncpo) * dt
        T = Nn / self.fs * 1e6
        hk = min(int(p.hk), nu // 2)
        kh = ks[np.argmin(np.abs(ks - hk))]
        # frequency-domain stems
        pf = self.plot("freq")
        pf.hline("z", 0, color=GRAY, style="-", width=0.7)
        pf.stems("re", ks, X.real / max(1.0, np.max(np.abs(X)) / 1.4), color=NAVY, size=5,
                 name="real part")
        pf.stems("im", ks + 0.25, X.imag / max(1.0, np.max(np.abs(X)) / 1.4), color=ORANGE,
                 size=5, name="imaginary part")
        ih = np.flatnonzero(ks == kh)
        pf.scatter("hk", [kh], [X.real[ih[0]] / max(1.0, np.max(np.abs(X)) / 1.4)], color=RED,
                   size=12, symbol="d", name=f"subcarrier {kh}")
        pf.set_xlim(-Nn / 2 - 1, Nn / 2 + 1)
        # the time-domain symbol
        ptm = self.plot("time")
        lim = 1.2 * np.max(np.abs(xcp.real)) + 0.3
        if ncpo:
            ptm.band("cp", t[0], 0, color=ORANGE, alpha=0.13)
            ptm.band("tail", T - ncp / self.fs * 1e6, T, color=GREEN, alpha=0.13)
            ptm.text("cpl", t[0], -1.02 * lim, "prefix", color=ORANGE, size=9, anchor=(0, 0))
            ptm.text("tll", T, -1.02 * lim, "copied from here", color=GREEN, size=9,
                     anchor=(1, 0))
        else:
            ptm.text("nocp", t[0] + 0.02 * T, 0.92 * lim, "no prefix", color=GRAY, size=9)
        ptm.line("x", t, xcp.real, color=NAVY, width=1.6, name="symbol, real part")
        comp = np.exp(2j * np.pi * kh * (np.arange(len(xcp)) - ncpo) / M) * \
            X[ih[0]] / np.sqrt(nu)
        ptm.line("c", t, 4 * comp.real, color=RED, width=1.3, style="--",
                 name=f"subcarrier {kh} alone (× 4)")
        idx = np.arange(0, len(xcp), self.over)
        ptm.scatter("smp", t[idx], xcp.real[idx], color=NAVY, size=4, name="the N samples sent")
        ptm.set_xlim(t[0] - 0.02 * T, T * 1.02)
        ptm.set_ylim(-lim, 1.35 * lim)
        # the envelope
        pw = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
        pe = self.plot("env")
        te = np.arange(M) * dt
        pe.line("p", te, db(pw), color=NAVY, width=1.3, fill=-25, fill_alpha=0.10,
                name="|x(t)|² / mean")
        pe.hline("mean", 0, color=GRAY, style="--", label="mean power", label_pos=0.45)
        i = int(np.argmax(pw))
        pe.scatter("pk", [te[i]], [db(pw[i])], color=RED, size=10, symbol="t",
                   name=f"peak {db(pw[i]):.1f} dB")
        pe.set_xlim(0, T)
        self.readout(df=self.fs / Nn / 1e3, T=T, tcp=ncp / self.fs * 1e6, papr=float(db(pw.max())))

    def story(self, p):
        Nn = int(p.N)
        s = (f"<p><b>Step 1</b> (top left): put one complex number on each of the active "
             f"subcarriers of a {v(Nn, 'd')}-point FFT. <b>Step 2</b>: one inverse FFT turns them "
             f"into {v(Nn, 'd')} time samples (the dots), which are the sum of all the subcarrier "
             f"sinusoids, each a whole number of cycles long: the dashed red curve is subcarrier "
             f"{int(p.hk)} on its own (magnified four times).</p>")
        tcp = self.r.get("tcp", 0)
        if tcp > 0:
            s += (f"<p><b>Step 3</b>: copy the last {v(tcp, '.2f', 'µs')} (green) to the front "
                  f"(orange). The symbol now looks periodic to any receiver whose FFT window starts "
                  f"anywhere inside the prefix, which is what absorbs multipath echoes "
                  f"(Experiment 3). It costs {v(100 * tcp / (tcp + self.r.get('T', 1)), '.0f', '%')} "
                  f"of the airtime.</p>")
        else:
            s += "<p><b>Step 3</b> is skipped: with no prefix, every echo will cause interference.</p>"
        papr = self.r.get("papr", 0)
        if p.data == "All in phase":
            s += (f"<p>All subcarriers in phase add up coherently once per symbol: a spike "
                  f"{v(papr, '.1f', 'dB')} above the mean, 10·log₁₀(number of subcarriers). A power "
                  f"amplifier must be backed off by that much to pass it undistorted.</p>")
        elif p.data.startswith("Zadoff"):
            s += (f"<p>A Zadoff–Chu sequence has constant magnitude in frequency <i>and</i> time "
                  f"(a CAZAC sequence), so the envelope is almost flat ({v(papr, '.1f', 'dB')} "
                  f"between samples). LTE and NR use them for synchronisation and random access.</p>")
        else:
            s += (f"<p>With random data the samples look like Gaussian noise: this symbol's peak is "
                  f"{v(papr, '.1f', 'dB')} above its mean. Press <i>New random data</i> a few times: "
                  f"the peaks wander, and occasionally one is huge (Experiment 7).</p>")
        return "<h3>IFFT, then prefix</h3>" + s + keybox(
            "Transmitter = IFFT + cyclic prefix. Receiver = drop the prefix + FFT. Everything else "
            "in OFDM is detail.")


# =============================================================================== 3. CP vs multipath
class CPvsMultipath(Experiment):
    title = "Cyclic prefix vs multipath"
    blurb = "Echoes longer than the prefix leak one symbol into the next. Find the shortest safe prefix."
    book = "sec:ch17:cp"
    controls = [
        Heading("Channel"),
        Choice("chan", "Channel", ["Two-ray", "EPA", "EVA", "ETU"], "EVA",
               help="3GPP LTE models: pedestrian (0.41 µs), vehicular (2.51 µs), "
                    "typical urban (5 µs)"),
        Slider("d2", "Echo delay", 0.0, 10.0, 6.0, step=0.1, unit="µs",
               enabled_if=lambda p: p.chan == "Two-ray"),
        Slider("a2", "Echo strength", -20, 0, -3, step=0.5, unit="dB",
               enabled_if=lambda p: p.chan == "Two-ray"),
        Button("again", "New channel draw"),
        Heading("Transmitter and receiver"),
        IntSlider("ncp", "Cyclic prefix", 0, 144, 72, step=4, unit="samples",
                  help="At 15.36 MS/s; LTE's normal prefix is 72 samples (4.69 µs)"),
        Slider("snr", "SNR per subcarrier", 10, 50, 40, step=0.5, unit="dB"),
    ]
    plots = [
        Plot("cir", "Channel impulse response vs the prefix", x="delay (µs)",
             y="path power (dB)", xlim=(-0.3, 10.5), ylim=(-32, 6), legend="tr"),
        ConstellationPlot("const", "16-QAM after the one-tap equaliser", lim=1.6),
        Plot("curve", "Interference + noise vs prefix length", x="cyclic prefix (µs)",
             y="relative to signal (dB)", xlim=(0, 9.6), ylim=(-55, 3), legend="tr"),
    ]
    layout = [["cir", "const"], ["curve", "curve"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("dmax", "Longest echo", "µs", ".2f"),
        Readout("tcp", "Prefix", "µs", ".2f"),
        Readout("evm", "Interference + noise", "dB", ".1f", good=lambda x: x < -25),
        Readout("ovh", "Prefix overhead", "%", ".1f"),
    ]
    challenges = [
        Challenge("EVA at 40 dB SNR: stay within 0.5 dB of the noise floor with a prefix SHORTER "
                  "than the longest echo (2.51 µs).",
                  lambda s: (s.p.chan == "EVA" and s.p.snr >= 40 and s.r.tcp < 2.5
                             and s.r.evm < -s.p.snr + 0.5),
                  hint="EVA's last paths are 12–17 dB weaker than the first: a slightly short "
                       "prefix lets only a little of their energy leak."),
        Challenge("Two-ray: put a 0 dB echo 2 µs beyond the prefix and see the floor stay above "
                  "−20 dB even at 50 dB SNR.",
                  lambda s: (s.p.chan == "Two-ray" and s.p.a2 >= -0.01 and s.p.snr >= 50
                             and s.p.d2 - s.r.tcp >= 1.95 and s.r.evm > -20)),
        Challenge("ETU at 40 dB SNR: LTE's normal prefix is shorter than ETU's 5 µs. Find a prefix "
                  "that gets within 0.3 dB of the noise floor.",
                  lambda s: s.p.chan == "ETU" and s.p.snr >= 40 and s.r.evm < -s.p.snr + 0.3,
                  hint="ETU's last path arrives at 5 µs, 77 samples. That is what LTE's extended "
                       "prefix (16.7 µs) is for in very large cells."),
    ]
    nsym = 6

    def setup(self):
        self.seed = 3
        self._curve_key = None

    def on_again(self, p):
        self.seed += 1

    def floors(self, g, h, ncp):
        cfg = co.OFDMConfig(NFFT, NUSED, int(ncp))
        x = co.ofdm_modulate(g, cfg)
        y = np.convolve(x, h)[:len(x)]
        Y = co.ofdm_demodulate(y, cfg)
        H = np.fft.fft(h, NFFT)[cfg.active]
        err = Y[1:] - H * g[1:]
        return Y, H, float(np.mean(np.abs(err) ** 2) / np.mean(np.abs(H * g[1:]) ** 2))

    def update(self, p):
        h = channel_taps(p.chan, self.seed, p.d2, p.a2)
        g = qam(Q16, NUSED * self.nsym, np.random.default_rng(7)).reshape(self.nsym, NUSED)
        Y, H, isi = self.floors(g, h, p.ncp)
        n0 = 10 ** (-p.snr / 10)
        rng = np.random.default_rng(self.seed + 1000)
        W = np.sqrt(n0 / 2) * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
        Z = (Y[1:] + W[1:]) / H
        tot = isi + n0
        dmax = max_delay_us(p.chan, p.d2)
        tcp = p.ncp / FS * 1e6
        # impulse response
        pc = self.plot("cir")
        d_us = np.arange(len(h)) / FS * 1e6
        nz = np.abs(h) > 1e-9
        P = db(np.abs(h[nz]) ** 2)
        inside = d_us[nz] <= tcp + 1e-9
        pc.band("cp", 0, tcp, color=GREEN, alpha=0.12)
        pc.text("cpl", 0.1, 5.6, f"prefix {tcp:.2f} µs", color=GREEN, size=9, anchor=(0, 0))
        pc.stems("in", d_us[nz][inside], P[inside], color=NAVY, base=-32, name="inside the prefix")
        if (~inside).any():
            pc.stems("out", d_us[nz][~inside], P[~inside], color=RED, base=-32,
                     name="too late: ISI")
        # constellation
        pk = self.plot("const")
        z = Z.ravel()
        pk.points("z", z[:3000], color=NAVY, size=3, alpha=0.5)
        pk.ideal("ideal", Q16.points, color=RED)
        # floor vs CP length (cached per channel)
        key = (p.chan, round(p.d2, 2), round(p.a2, 2), self.seed)
        if key != self._curve_key:
            grid = np.arange(0, 148, 8)
            self._curve = (grid, np.array([self.floors(g, h, c)[2] for c in grid]))
            self._curve_key = key
        grid, fl = self._curve
        pv = self.plot("curve")
        pv.line("isi", grid / FS * 1e6, db(fl + 1e-6), color=GRAY, width=1.4, style="--",
                name="echo interference alone")
        pv.line("tot", grid / FS * 1e6, db(fl + n0), color=NAVY, width=2.2,
                name="interference + noise")
        pv.hline("n0", -p.snr, color=GREEN, style=":", label="noise floor", label_pos=0.05)
        pv.vline("dmax", dmax, color=ORANGE, style=":", label="longest echo", label_pos=0.55)
        pv.vline("lte", 72 / FS * 1e6, color=GRAY, style=":", width=0.9, label="LTE normal",
                 label_pos=0.93)
        pv.scatter("now", [tcp], [db(tot)], color=RED, size=13, symbol="d", name="your prefix")
        self.readout(dmax=dmax, tcp=tcp, evm=float(db(tot)),
                     ovh=100 * p.ncp / (NFFT + p.ncp))
        self._isi = isi

    def story(self, p):
        tcp, dmax = self.r.get("tcp", 0), self.r.get("dmax", 0)
        isi_db = float(db(getattr(self, "_isi", 0) + 1e-6))
        s = (f"<p>The channel smears each symbol over {v(dmax, '.2f', 'µs')} (top left). As long as "
             f"the prefix ({v(tcp, '.2f', 'µs')}) is longer, every echo of the previous symbol dies "
             f"out inside the prefix the receiver throws away, and what is left looks like a "
             f"<i>circular</i> convolution: each subcarrier is simply multiplied by one complex "
             f"number H<sub>k</sub>, undone by one division.</p>")
        if tcp + 1e-9 < dmax:
            s += (f"<p>{bad('The prefix is too short.')} The red paths arrive after it ends: the "
                  f"tail of one symbol spills into the next (ISI) and the FFT window no longer holds "
                  f"whole cycles of the echoes (ICI). Their interference sits at "
                  f"{v(isi_db, '.1f', 'dB')}, a floor no amount of transmit power can push down "
                  f"(grey dashed curve).</p>")
        else:
            s += (f"<p>{good('The prefix covers the channel.')} The constellation is limited by "
                  f"noise only, amplified on the subcarriers that sit in a fade (the scattered "
                  f"points).</p>")
        s += ("<p>The price is airtime: LTE spends 6.7 % on the prefix, 802.11a 20 %, and a DVB-T2 "
              "single-frequency network up to 25 % because its echoes are other transmitters "
              "tens of kilometres away.</p>")
        return "<h3>The prefix absorbs the echoes</h3>" + s + keybox(
            "Prefix ≥ channel memory ⇒ Yₖ = Hₖ·Xₖ + Wₖ. Shorter ⇒ an interference floor.")


# =============================================================================== 4. one-tap equaliser
class OneTapEqualizer(Experiment):
    title = "The one-tap equaliser"
    blurb = "One complex division per subcarrier undoes a 20 dB-deep frequency-selective channel."
    book = "sec:ch17:cp"
    controls = [
        Choice("chan", "Channel", ["Flat", "EPA", "EVA", "ETU"], "EVA"),
        Button("again", "New channel draw"),
        Choice("mod", "Modulation", ["QPSK", "16-QAM", "64-QAM"], "16-QAM"),
        Slider("snr", "Average SNR per subcarrier", 0, 40, 20, step=0.5, unit="dB"),
        Choice("eq", "Equaliser", ["None", "Zero-forcing", "MMSE"], "Zero-forcing",
               help="ZF divides by Hₖ. MMSE divides by Hₖ but backs off where |Hₖ|² is "
                    "comparable to the noise."),
    ]
    plots = [
        Plot("snrk", "SNR on each subcarrier", x="frequency (MHz)", y="SNR (dB)", xlim=(-4.6, 4.6),
             ylim=(-20, 52), legend="tl", legend_cols=2),
        ConstellationPlot("rx", "Received Yₖ (before equalising)", lim=2.6),
        ConstellationPlot("eq", "After the one-tap equaliser", lim=1.65),
    ]
    layout = [["snrk", "snrk"], ["rx", "eq"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("fade", "Deepest fade", "dB", ".1f"),
        Readout("below", "Subcarriers below threshold", "%", ".1f", good=lambda x: x < 5),
        Readout("evm", "EVM after equalising", "dB", ".1f"),
        Readout("ber", "Uncoded BER", "", "sci", good=lambda x: x < 1e-3),
    ]
    challenges = [
        Challenge("EVA with 16-QAM: get the uncoded BER below 10⁻³.",
                  lambda s: s.p.chan == "EVA" and s.p.mod == "16-QAM" and s.r.ber < 1e-3,
                  hint="In fading, BER falls only as 1/SNR: the deepest fades dominate."),
        Challenge("At an SNR of 10 dB or less on EVA, show that MMSE beats zero-forcing's EVM "
                  "by at least 2 dB.",
                  lambda s: (s.p.eq == "MMSE" and s.p.chan == "EVA" and s.p.snr <= 10
                             and s.exp.evm_zf - s.r.evm >= 2),
                  hint="Zero-forcing divides the noise by |Hₖ|² on the faded subcarriers."),
        Challenge("64-QAM on ETU: keep at least 95 % of the subcarriers above the 64-QAM threshold.",
                  lambda s: s.p.mod == "64-QAM" and s.p.chan == "ETU" and s.r.below <= 5),
    ]
    nsym = 8

    def setup(self):
        self.seed = 11

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        con = CONS[p.mod]
        rng = np.random.default_rng(self.seed)
        h = channel_taps(p.chan, self.seed)
        H = np.fft.fft(h, NFFT)[CFG.active]
        bits = cl.random_bits(con.k * NUSED * self.nsym, np.random.default_rng(5))
        X = con.modulate(bits).reshape(self.nsym, NUSED)
        n0 = 10 ** (-p.snr / 10)
        nrng = np.random.default_rng(self.seed + 99)
        W = np.sqrt(n0 / 2) * (nrng.standard_normal(X.shape) + 1j * nrng.standard_normal(X.shape))
        Y = H * X + W
        if p.eq == "None":
            Z = Y
        elif p.eq == "Zero-forcing":
            Z = Y / H
        else:
            Z = Y * np.conj(H) / (np.abs(H) ** 2 + n0)
        evm = db(np.mean(np.abs(Z - X) ** 2) / np.mean(np.abs(X) ** 2))
        self.evm_zf = float(db(np.mean(np.abs(Y / H - X) ** 2) / np.mean(np.abs(X) ** 2)))
        # MMSE output is biased toward zero on faded subcarriers: unbias before deciding
        Zd = Z / (np.abs(H) ** 2 / (np.abs(H) ** 2 + n0)) if p.eq == "MMSE" else Z
        ber = float(np.mean(con.demodulate(Zd.ravel()) != bits))
        snrk = p.snr + db(np.abs(H) ** 2)
        thr = THRESH[p.mod]
        fmhz = KSIG * 15e-3
        ps = self.plot("snrk")
        ps.line("s", fmhz, snrk, color=NAVY, width=1.8, name="SNR of subcarrier k")
        bad_ = np.where(snrk < thr, snrk, np.nan)
        ps.line("bad", fmhz, bad_, color=RED, width=2.6, name=f"below the {p.mod} threshold")
        ps.hline("thr", thr, color=RED, style="--", label=f"{p.mod} needs ≈ {thr:.1f} dB",
                 label_pos=0.82)
        ps.hline("avg", p.snr, color=GRAY, style=":", label="average", label_pos=0.6)
        ps.set_ylim(p.snr - 35, p.snr + 17)
        pr = self.plot("rx")
        pr.points("y", Y[:3].ravel(), color=ORANGE, size=3, alpha=0.45)
        pr.ideal("ideal", con.points, color=GRAY)
        pe = self.plot("eq")
        pe.points("z", Z[:4].ravel(), color=NAVY, size=3, alpha=0.5)
        pe.ideal("ideal", con.points, color=RED)
        self.readout(fade=float(db(np.min(np.abs(H) ** 2))), below=100 * np.mean(snrk < thr),
                     evm=float(evm), ber=ber if ber > 0 else 1.0 / bits.size / 3)
        self._ber0 = ber == 0

    def story(self, p):
        fade, below = self.r.get("fade", 0), self.r.get("below", 0)
        s = ("<p>With an adequate prefix, the channel multiplies subcarrier k by one complex "
             "number Hₖ: a gain and a rotation. The received points (bottom left) are therefore "
             "scaled and rotated copies of the constellation, a different copy on every "
             "subcarrier, so together they are a meaningless smear.</p>")
        if p.eq == "None":
            s += "<p>Without an equaliser there is no constellation left to decide on.</p>"
        else:
            s += (f"<p>Divide each subcarrier by its own Hₖ and the constellation snaps back. "
                  f"The catch is in the top plot: the channel puts some subcarriers into fades as "
                  f"deep as {v(fade, '.1f', 'dB')}, and dividing restores the signal <i>and the "
                  f"noise with it</i>. {v(below, '.1f', '%')} of the subcarriers sit below the "
                  f"{p.mod} threshold.</p>")
            if p.eq == "MMSE":
                s += ("<p><b>MMSE</b> refuses to blow up the noise on the faded subcarriers: it "
                      "divides by |Hₖ|² + N₀ instead of |Hₖ|², which minimises the total error.</p>")
        s += ("<p>Uncoded, those few bad subcarriers set the error rate, which falls only as 1/SNR "
              "(like flat Rayleigh fading). A code spread across the band repairs them from the "
              "good ones: that is coded OFDM, the subject of Lab 28.</p>")
        return "<h3>One division per subcarrier</h3>" + s + keybox(
            "Equalising OFDM costs one complex multiply per subcarrier, whatever the delay spread. "
            "The faded subcarriers are the problem, not the ISI.")


# =============================================================================== 5. Schmidl & Cox
class SchmidlCox(Experiment):
    title = "Acquisition: Schmidl & Cox"
    blurb = "A preamble with two identical halves finds both the symbol start and the frequency offset."
    book = "sec:ch17:sync"
    controls = [
        Slider("eps", "Carrier frequency offset", -1.5, 1.5, 0.3, step=0.01, unit="× Δf",
               help="In subcarrier spacings. 1 Δf = 15 kHz in LTE"),
        Slider("snr", "SNR", -5, 30, 20, step=0.5, unit="dB"),
        Toggle("fix", "Correct the estimated offset", True,
               help="De-rotate by the estimate, then let pilots remove the leftover common phase"),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("metric", "Timing metric M(d) = |P(d)|² / R(d)²", x="sample index d",
             y="M(d)", ylim=(-0.05, 1.3), legend="tr"),
        Plot("est", "Estimated vs true offset", x="true offset ε (× Δf)",
             y="estimate ε̂ (× Δf)", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), legend="tl"),
        ConstellationPlot("const", "16-QAM data after the FFT", lim=1.7),
    ]
    layout = [["metric", "metric"], ["est", "const"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("terr", "Timing", "", None),
        Readout("ehat", "Offset estimate", "× Δf", ".3f"),
        Readout("res", "Residual offset", "× Δf", ".3f", good=lambda x: abs(x) < 0.02),
        Readout("evm", "Data EVM", "dB", ".1f", good=lambda x: x < -20),
    ]
    challenges = [
        Challenge("At an SNR of 0 dB or less, still estimate the offset to within 0.02 subcarrier "
                  "spacings.",
                  lambda s: s.p.snr <= 0 and abs(s.r.res) < 0.02,
                  hint="The estimate averages 128 sample pairs; try New noise if you are unlucky."),
        Challenge("Push the offset past one subcarrier spacing and watch the estimate wrap "
                  "around (residual error above 1.5 Δf).",
                  lambda s: abs(s.r.res) > 1.5,
                  hint="The phase of P(d) only knows ε modulo 2: the integer part needs a second "
                       "training symbol."),
        Challenge("Turn the correction off and find how small an offset already wrecks 16-QAM "
                  "(EVM worse than −10 dB with |ε| ≤ 0.02).",
                  lambda s: (not s.p.fix) and abs(s.p.eps) <= 0.02 and s.r.evm > -10),
    ]
    Nn, nu, ncp, nd = 256, 200, 16, 8
    lead = 300

    def setup(self):
        self.seed = 21
        rng = np.random.default_rng(4)
        cfg = co.OFDMConfig(self.Nn, self.nu, self.ncp)
        X = np.zeros(self.Nn, complex)
        ev = cfg.active[(cfg.k % 2) == 0]                     # even subcarriers only
        X[ev] = QPSK.points[rng.integers(0, 4, len(ev))] * np.sqrt(2)
        pre = np.fft.ifft(X) * np.sqrt(self.Nn)               # two identical halves
        self.pre = np.concatenate([pre[-self.ncp:], pre])
        self.g = qam(Q16, self.nu * self.nd, rng).reshape(self.nd, self.nu)
        self.data = co.ofdm_modulate(self.g, cfg)
        self.cfg = cfg

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        Nn, L = self.Nn, self.Nn // 2
        rng = np.random.default_rng(self.seed)
        sig = np.concatenate([np.zeros(self.lead), self.pre, self.data, np.zeros(120)])
        n = np.arange(len(sig))
        pw = np.mean(np.abs(self.data) ** 2)
        n0 = pw / 10 ** (p.snr / 10)
        r = sig * np.exp(2j * np.pi * p.eps / Nn * n)
        r = r + np.sqrt(n0 / 2) * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))
        M, P = cl.schmidl_cox_metric(r, L)
        d = int(np.argmax(M))
        ehat = float(np.angle(P[d]) / np.pi)
        start = self.lead + self.ncp                          # preamble body starts here
        if start - self.ncp <= d <= start:
            terr = "inside the prefix ✓"
        else:
            terr = f"{min(abs(d - start), abs(d - start + self.ncp))} samples off"
        res = p.eps - ehat if p.fix else p.eps
        y = r * np.exp(-2j * np.pi * ehat / Nn * n) if p.fix else r
        d0 = self.lead + len(self.pre)
        Y = co.ofdm_demodulate(y[d0:d0 + self.nd * (Nn + self.ncp)], self.cfg)
        if p.fix:                     # pilots on every 12th subcarrier track the leftover rotation
            pil = np.zeros_like(self.g)
            pil[:, ::12] = self.g[:, ::12]
            Y, _ = oa.cpe_correct(Y, pil)
        evm = db(np.mean(np.abs(Y - self.g) ** 2) / np.mean(np.abs(self.g) ** 2))
        pm = self.plot("metric")
        dd = np.arange(len(M))
        pm.band("pl", start - self.ncp, start, color=GREEN, alpha=0.18)
        pm.line("M", dd, M, color=NAVY, width=1.6, name="metric")
        pm.vline("true", start, color=GREEN, style=":", label="preamble starts", label_pos=0.88)
        pm.scatter("pk", [d], [M[d]], color=RED, size=11, symbol="d", name="detected peak")
        pm.set_xlim(0, 1000)
        pe = self.plot("est")
        x = np.linspace(-1.6, 1.6, 641)
        wrap = (x + 1) % 2 - 1
        wrap[np.abs(np.diff(np.r_[wrap[0], wrap])) > 1] = np.nan
        pe.line("id", x, x, color=GRAY, style="--", width=1.0, name="perfect")
        pe.line("w", x, wrap, color=NAVY, width=2.0, name="what the phase can tell")
        pe.scatter("now", [p.eps], [ehat], color=RED, size=13, symbol="d", name="this estimate")
        pc = self.plot("const")
        pc.points("y", Y.ravel()[:1600], color=NAVY, size=3, alpha=0.5)
        pc.ideal("ideal", Q16.points, color=RED)
        self.readout(terr=terr, ehat=ehat, res=res, evm=float(evm))

    def story(self, p):
        ehat, res = self.r.get("ehat", 0), self.r.get("res", 0)
        s = ("<p>The preamble uses only the even subcarriers, so in time it is two identical halves. "
             "The receiver multiplies the signal by a copy of itself delayed by half a symbol and "
             "sums: noise averages out, the preamble's halves line up, and the metric M(d) jumps "
             "towards 1. With a cyclic prefix it becomes a plateau (green): any start inside it is "
             "an ISI-free window.</p>")
        s += (f"<p>A frequency offset rotates the second half relative to the first by π·ε, so the "
              f"<b>angle</b> of the correlation at the peak is the offset: here "
              f"ε̂ = {v(ehat, '.3f')} Δf against a true {v(p.eps, '.3f')} Δf.</p>")
        if abs(p.eps) > 1:
            s += (f"<p>{bad('Ambiguous!')} An angle only lives in (−π, π], so offsets beyond one "
                  f"subcarrier spacing wrap around (the sawtooth). Real systems add a second "
                  f"training symbol, or search over integer offsets, to resolve it.</p>")
        elif not p.fix:
            s += (f"<p>{bad('Correction off.')} The offset keeps rotating every symbol by "
                  f"2π·ε·(N + N<sub>cp</sub>)/N, so the constellation smears into rings within a few "
                  f"symbols, even when the ICI itself is tiny.</p>")
        else:
            s += (f"<p>After correction the residual is {v(res, '.3f', 'Δf')}; pilots on every 12th "
                  f"subcarrier remove the slow common rotation it leaves. Every OFDM standard "
                  f"does this first: Wi-Fi's short and long training fields, LTE/NR's PSS/SSS and "
                  f"DVB-T2's P1 symbol are all relatives of this idea.</p>")
        return "<h3>Two halves find the clock and the carrier</h3>" + s + keybox(
            "Timing from the metric's peak, frequency from its phase: ε̂ = ∠P(d̂)/π.")


# =============================================================================== 6. pilots
class CombPilots(Experiment):
    title = "Pilots and channel estimation"
    blurb = "Known symbols every few subcarriers sample H(f). Space them too far and it aliases."
    book = "sec:ch17:grid"
    heavy = True
    controls = [
        Heading("Channel"),
        Choice("chan", "Channel", ["EPA", "EVA", "ETU"], "EVA"),
        Button("again", "New channel draw"),
        Slider("snr", "SNR", 0, 40, 20, step=0.5, unit="dB"),
        Heading("Pilots"),
        Choice("dp", "Pilot every", ["2", "3", "4", "6", "8", "12", "16", "24"], "6",
               help="Pilot spacing Dƒ in subcarriers (comb pattern, every symbol)"),
        Choice("interp", "Interpolation", ["Linear", "Nearest pilot"]),
    ]
    plots = [
        Plot("H", "Channel and its estimate (first 300 subcarriers)", x="subcarrier index",
             y="|H| (dB)", xlim=(-301, 0), ylim=(-45, 12), legend="bl", legend_cols=3),
        ConstellationPlot("const", "16-QAM equalised with the estimate", lim=1.7),
        BERPlot("ber", "Uncoded 16-QAM over the channel (Monte Carlo)", x="SNR per subcarrier (dB)",
                ylim=(1e-5, 0.5), xlim=(-1, 41), legend="bl"),
    ]
    layout = [["H", "H"], ["const", "ber"]]
    row_stretch = [2, 3]
    col_stretch = [2, 3]
    readouts = [
        Readout("nyq", "Nyquist limit on spacing", "subcarriers", ".1f"),
        Readout("mse", "Estimation error", "dB", ".1f", good=lambda x: x < -20),
        Readout("pen", "SNR lost to estimation", "dB", ".2f", good=lambda x: x < 1,
                help="Effective SNR with the estimate vs with perfect channel knowledge: "
                     "10·log₁₀((N₀ + MSE)/N₀)"),
        Readout("ovh", "Pilot overhead", "%", ".1f"),
    ]
    challenges = [
        Challenge("ETU at 20 dB SNR: keep the estimation error below −15 dB with pilots no denser "
                  "than every 4th subcarrier.",
                  lambda s: s.p.chan == "ETU" and s.p.snr <= 20 and int(s.p.dp) >= 4 and s.r.mse < -15),
        Challenge("Space the pilots beyond the Nyquist limit and watch the estimate fall apart "
                  "(error above −9 dB).",
                  lambda s: int(s.p.dp) > s.r.nyq and s.r.mse > -9,
                  hint="Dƒ ≤ 1/(τ_max·Δf): the comb must sample H(f) faster than it changes."),
        Challenge("EVA at 30 dB SNR: lose less than 2.5 dB to estimation while spending at most "
                  "1/6 of the subcarriers on pilots.",
                  lambda s: (s.p.chan == "EVA" and s.p.snr >= 30 and int(s.p.dp) >= 6
                             and s.r.pen < 2.5),
                  hint="At high SNR the interpolation's bias, not the noise, sets the error."),
    ]
    nsym = 2

    def setup(self):
        self.seed = 31
        self.ghost = None
        self.sim = None

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def estimate(Yp, kp, method):
        if method == "Linear":
            return oa.chest_ls_interp(Yp, kp, KSIG)
        idx = np.argmin(np.abs(KSIG[:, None] - kp[None, :]), axis=1)
        return Yp[idx]

    def update(self, p):
        Dp = int(p.dp)
        pidx = np.arange(0, NUSED, Dp)
        kp = KSIG[pidx]
        mask = np.zeros(NUSED, bool)
        mask[pidx] = True
        rng = np.random.default_rng(self.seed)
        H = oa.tdl_freq_response(p.chan, KSIG, NFFT, FS, rng)
        X = qam(Q16, NUSED * self.nsym, np.random.default_rng(3)).reshape(self.nsym, NUSED)
        X[:, mask] = QPSK.points[0]
        n0 = 10 ** (-p.snr / 10)
        nr = np.random.default_rng(self.seed + 7)
        Y = H * X + np.sqrt(n0 / 2) * (nr.standard_normal(X.shape) + 1j * nr.standard_normal(X.shape))
        Hls = (Y[:, mask] / X[:, mask]).mean(axis=0)         # average the two pilot symbols
        Hh = self.estimate(Hls, kp, p.interp)
        mse = db(np.mean(np.abs(Hh - H) ** 2))
        Z = (Y / Hh)[:, ~mask]
        pen = 10 * np.log10((n0 + np.mean(np.abs(Hh - H) ** 2)) / n0)
        dmax = cl.TDL_PROFILES[p.chan][0][-1] * 1e-9
        nyq = 1 / (dmax * 15e3)
        sel = KSIG < 0
        ph = self.plot("H")
        ph.line("H", KSIG[sel], db(np.abs(H[sel]) ** 2), color=NAVY, width=2.4, name="true channel")
        ph.line("Hh", KSIG[sel], db(np.abs(Hh[sel]) ** 2), color=RED, width=1.5, name="estimate")
        ps = kp < 0
        ph.scatter("p", kp[ps], db(np.abs(Hls[ps]) ** 2), color=ORANGE, size=7,
                   name="LS at the pilots")
        pc = self.plot("const")
        pc.points("z", Z.ravel()[:2400], color=NAVY, size=3, alpha=0.5)
        pc.ideal("ideal", Q16.points, color=RED)
        pb = self.plot("ber")
        if self.sim is not None and self.sim.get("done") and len(self.sim["x"]) > 2:
            self.ghost = dict(self.sim)
        self.sim = dict(x=[], est=[], perf=[], done=False)
        if self.ghost is not None:
            pb.sim("ghost", self.ghost["x"], self.ghost["est"], color=GRAY, size=7,
                   name="previous setting")
        pb.vline("now", max(p.snr, 0.01), color=GRAY, style=":", width=0.9)
        self.readout(nyq=nyq, mse=float(mse), pen=float(pen), ovh=100 / Dp)

    def background(self, p):
        Dp = int(p.dp)
        pidx = np.arange(0, NUSED, Dp)
        kp = KSIG[pidx]
        mask = np.zeros(NUSED, bool)
        mask[pidx] = True
        rng = np.random.default_rng(1234)
        B = 12 if self.quick else 80
        for snr in np.arange(0, 41, 5.0):
            n0 = 10 ** (-snr / 10)
            H = oa.tdl_freq_response(p.chan, KSIG, NFFT, FS, rng, n_draws=B)
            bits = cl.random_bits(4 * B * NUSED, rng)
            X = Q16.modulate(bits).reshape(B, NUSED)
            X[:, mask] = QPSK.points[0]
            W = np.sqrt(n0 / 2) * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
            Y = H * X + W
            Hls = Y[:, mask] / X[:, mask]
            Hh = np.array([self.estimate(Hls[b], kp, p.interp) for b in range(B)])
            b2 = bits.reshape(B, NUSED, 4)[:, ~mask].ravel()
            e_est = np.mean(Q16.demodulate((Y / Hh)[:, ~mask].ravel()) != b2)
            e_perf = np.mean(Q16.demodulate((Y / H)[:, ~mask].ravel()) != b2)
            yield dict(snr=snr, est=max(e_est, 1e-7), perf=max(e_perf, 1e-7))
        yield dict(done=True)

    def progress(self, p, it):
        if it.get("done"):
            self.sim["done"] = True
            return
        self.sim["x"].append(it["snr"])
        self.sim["est"].append(it["est"])
        self.sim["perf"].append(it["perf"])
        pb = self.plot("ber")
        pb.sim("perf", self.sim["x"], self.sim["perf"], color=GREEN, name="perfect channel knowledge")
        pb.sim("est", self.sim["x"], self.sim["est"], color=RED,
               name=f"pilots every {p.dp}, {p.interp.lower()}")

    def story(self, p):
        nyq, mse = self.r.get("nyq", 1), self.r.get("mse", 0)
        s = ("<p>The receiver needs Hₖ on every subcarrier, but knows the transmitted value only on "
             "the <b>pilots</b>: there, Hₖ = Yₖ/Xₖ (orange dots, noisy). Everything in between is "
             "interpolation. H(f) is the Fourier transform of the impulse response, so it wiggles "
             "faster the longer the echoes are: a channel of length τ needs a pilot at least every "
             f"1/(τ·Δf) subcarriers, here {v(nyq, '.1f')}. It is the sampling theorem, turned "
             "sideways.</p>")
        if int(p.dp) > nyq:
            s += (f"<p>{bad('Under-sampled.')} With a pilot every {v(int(p.dp), 'd')} subcarriers the "
                  f"comb cannot follow the channel's ripples: the red estimate cuts across the fades "
                  f"(error {v(mse, '.1f', 'dB')}) and the BER floors no matter the SNR.</p>")
        else:
            s += (f"<p>The comb samples the channel fast enough. The estimation error "
                  f"({v(mse, '.1f', 'dB')}) is then set by the noise on the pilots and the bias of "
                  f"{p.interp.lower()} interpolation in the deepest fades. Denser pilots average more "
                  f"noise but steal data subcarriers: {v(100 / int(p.dp), '.1f', '%')} here.</p>")
        s += ("<p>The BER curves (bottom right) run in the background: random channels, red with your "
              "pilots, green with perfect knowledge. Note how slowly even the green one falls: "
              "uncoded OFDM in fading is poor, which is why it always comes with a code.</p>")
        return "<h3>Sampling the channel</h3>" + s + keybox(
            "Pilot spacing Dƒ ≤ 1/(τ_max·Δf) in frequency and Dₜ ≤ 1/(2·f_D·T) in time.")


# =============================================================================== 7. PAPR
class PAPR(Experiment):
    title = "PAPR: OFDM vs DFT-s-OFDM"
    blurb = "Sums of many subcarriers have huge peaks. Spreading with a DFT first tames them."
    book = "sec:ch17:dfts"
    controls = [
        Choice("nu", "Active subcarriers", ["52", "300", "1200"], "300"),
        Choice("mod", "Modulation", ["QPSK", "16-QAM", "π/2-BPSK"], "16-QAM",
               help="π/2-BPSK: NR's low-PAPR uplink option for coverage-limited phones"),
        Toggle("fdss", "Spectral shaping on DFT-s-OFDM (FDSS)", False,
               help="Frequency-domain spectral shaping: weight the DFT outputs by |1 + e^(−j2πk/M)|, "
                    "i.e. filter the symbols with [1, 1]. NR allows it with π/2-BPSK"),
        Choice("show", "Envelope shown", ["CP-OFDM", "DFT-s-OFDM", "Single carrier"]),
        Button("again", "New data"),
    ]
    plots = [
        Plot("env", "Instantaneous power over three symbols", x="time (symbols)",
             y="|x|² / mean (dB)", xlim=(0, 3), ylim=(-25, 16), legend="tl", legend_cols=2),
        Plot("ccdf", "How often the peaks exceed a level (CCDF of PAPR)", x="PAPR₀ (dB)",
             y="P(PAPR > PAPR₀)", xlim=(0, 14.5), ylim=(1e-3, 1), logy=True, legend="tr"),
        Plot("pa", "What the back-off costs a class-B amplifier", x="back-off from saturation (dB)",
             y="efficiency (%)", xlim=(0, 14), ylim=(0, 85), legend="tr"),
    ]
    layout = [["env", "env"], ["ccdf", "pa"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("ofdm", "CP-OFDM PAPR at 10⁻³", "dB", ".1f"),
        Readout("dfts", "DFT-s-OFDM", "dB", ".1f"),
        Readout("sc", "Single carrier", "dB", ".1f"),
        Readout("gain", "Efficiency gain (DFT-s vs OFDM)", "×", ".2f"),
    ]
    challenges = [
        Challenge("Get DFT-s-OFDM's PAPR at 10⁻³ below 4 dB.",
                  lambda s: s.r.dfts < 4,
                  hint="Two tricks together: a modulation that never jumps through the origin, "
                       "and spectral shaping."),
        Challenge("Make DFT-s-OFDM run the amplifier at least twice as efficiently as CP-OFDM.",
                  lambda s: s.r.gain >= 2),
        Challenge("Show that CP-OFDM's peaks hardly depend on the modulation: with 1200 "
                  "subcarriers, π/2-BPSK, still above 10.5 dB.",
                  lambda s: s.p.nu == "1200" and s.p.mod == "π/2-BPSK" and s.r.ofdm > 10.5),
    ]
    grid = np.linspace(0, 14.5, 291)

    def setup(self):
        self.cache = {}
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def symbols(mod, n, rng):
        if mod == "π/2-BPSK":
            b = 1 - 2 * rng.integers(0, 2, n)
            return b * np.exp(1j * np.pi / 2 * (np.arange(n) % 2)) * np.exp(1j * np.pi / 4)
        return qam(CONS[mod], n, rng)

    @staticmethod
    def dfts(sym, M, Nf, fdss=False):
        """Localised DFT-s-OFDM: an M-point DFT, mapped (centred, contiguous) onto Nf bins."""
        s = sym.reshape(-1, M)
        S = np.fft.fft(s, axis=1) / np.sqrt(M)
        if fdss:
            W = 1 + np.exp(-2j * np.pi * np.arange(M) / M)
            S = S * W / np.sqrt(np.mean(np.abs(W) ** 2))
        S = np.fft.fftshift(S, axes=1)
        X = np.zeros((len(s), Nf), complex)
        X[:, (np.arange(M) - M // 2) % Nf] = S
        return (np.fft.ifft(X, axis=1) * np.sqrt(Nf)).ravel()

    def waveforms(self, nu, mod, nsym, rng, fdss=False):
        nf = {52: 64, 300: 512, 1200: 2048}[nu]
        cfg = co.OFDMConfig(4 * nf, nu, 0)
        s1 = self.symbols(mod, nu * nsym, rng)
        x_o = co.ofdm_modulate(s1.reshape(nsym, nu), cfg)
        s2 = self.symbols(mod, nu * nsym, rng)
        x_d = self.dfts(s2, nu, 4 * nf, fdss)
        s3 = self.symbols(mod, nu * (nsym + 2), rng)
        x_s = cl.shape(s3, cl.rrc_taps(0.22, 4, 10), 4)[4 * nu:4 * nu * (nsym + 1)]
        return dict(o=x_o, d=x_d, s=x_s), 4 * nf, 4 * nu

    def update(self, p):
        nu = int(p.nu)
        rng = np.random.default_rng(self.seed)
        w, L, Ls = self.waveforms(nu, p.mod, 3, rng, p.fdss)
        key = {"CP-OFDM": "o", "DFT-s-OFDM": "d", "Single carrier": "s"}[p.show]
        x = w[key]
        Lx = L if key != "s" else Ls
        pw = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
        t = np.arange(len(x)) / Lx
        col = {"o": NAVY, "d": RED, "s": GRAY}[key]
        pe = self.plot("env")
        pe.line("p", t, db(pw), color=col, width=1.0, fill=-25, fill_alpha=0.10,
                name=p.show)
        pe.hline("m", 0, color=GRAY, style="--", label="mean", label_pos=0.97)
        for s in range(3):
            seg = pw[s * Lx:(s + 1) * Lx]
            i = int(np.argmax(seg))
            pe.scatter(f"pk{s}", [t[s * Lx + i]], [db(seg[i])], color=ORANGE, size=10, symbol="t",
                       name="peak of each symbol" if s == 0 else None)
        ck = (nu, p.mod, p.fdss)
        self.cur = self.cache.get(ck)
        self.draw_ccdf(p)

    def draw_ccdf(self, p):
        pc = self.plot("ccdf")
        pc.hline("e3", 1e-3 * 1.02, color=GRAY, style=":", width=0.8)
        cur = self.cur
        vals = {}
        if cur is not None:
            for k, col, nm in (("o", NAVY, "CP-OFDM"), ("d", RED, "DFT-s-OFDM"),
                               ("s", GRAY, "single carrier (RRC 0.22)")):
                cc = ccdf_fast(cur[k], self.grid)
                pc.line(k, self.grid, np.where(cc > 0, cc, np.nan), color=col, width=2.0, name=nm)
                vals[k] = papr_at(self.grid, cc)
        pa = self.plot("pa")
        bo = np.linspace(0, 14, 200)
        pa.line("eta", bo, 78.5 * 10 ** (-bo / 20), color=GREEN, width=2.0,
                name="ideal class B: 78.5 % × 10^(−BO/20)")
        for k, col in (("o", NAVY), ("d", RED), ("s", GRAY)):
            if k in vals:
                e = 78.5 * 10 ** (-vals[k] / 20)
                pa.scatter("m" + k, [vals[k]], [e], color=col, size=12, symbol="d")
                below = k == "s"
                pa.text("t" + k, vals[k] + 0.25, e - 1.5 if below else e + 2.5, f"{e:.0f} %",
                        color=col, size=9, anchor=(0, 0) if below else (0, 1))
        if vals:
            self.readout(ofdm=vals["o"], dfts=vals["d"], sc=vals["s"],
                         gain=10 ** ((vals["o"] - vals["d"]) / 20))
        else:
            self.readout(ofdm=None, dfts=None, sc=None, gain=None)

    def background(self, p):
        nu = int(p.nu)
        ck = (nu, p.mod, p.fdss)
        if ck in self.cache:
            return
        rng = np.random.default_rng(77)
        target = 300 if self.quick else 2000
        chunk = {52: 1000, 300: 250, 1200: 60}[nu] if not self.quick else 150
        acc = {"o": [], "d": [], "s": []}
        done = 0
        while done < target:
            w, L, Ls = self.waveforms(nu, p.mod, chunk, rng, p.fdss)
            acc["o"].append(co.papr_db(w["o"], L))
            acc["d"].append(co.papr_db(w["d"], L))
            acc["s"].append(co.papr_db(w["s"], Ls))
            done += chunk
            yield (ck, {k: np.concatenate(v_) for k, v_ in acc.items()}, done >= target)

    def progress(self, p, item):
        ck, vals, final = item
        if final:
            self.cache[ck] = vals
        if ck == (int(p.nu), p.mod, p.fdss):
            self.cur = vals
            self.draw_ccdf(p)

    def story(self, p):
        o, d = self.r.get("ofdm"), self.r.get("dfts")
        s = ("<p>An OFDM sample is the sum of hundreds of independent subcarriers, so by the central "
             "limit theorem it is nearly Gaussian: mostly modest, occasionally huge. A power "
             "amplifier must be backed off from saturation by the PAPR it has to pass cleanly, and "
             "back-off is wasted battery (right).</p>")
        if o is not None and d is not None:
            s += (f"<p>Here CP-OFDM exceeds {v(o, '.1f', 'dB')} once in a thousand symbols, "
                  f"DFT-spread OFDM only {v(d, '.1f', 'dB')}. The DFT spreads each data symbol over "
                  f"all the subcarriers, so after the IFFT the signal is a single carrier again, "
                  f"with a single carrier's gentle envelope: {v(o - d, '.1f', 'dB')} less back-off, "
                  f"{v(self.r.get('gain', 1), '.2f')}× the amplifier efficiency.</p>")
        else:
            s += "<p>The CCDF curves are being measured in the background…</p>"
        if p.mod == "π/2-BPSK":
            s += ("<p>π/2-BPSK rotates every other symbol by 90°, so consecutive symbols never "
                  "jump through the origin. "
                  + ("With spectral shaping, which smooths the transitions between symbols, the "
                     "DFT-spread envelope is almost constant. " if p.fdss else
                     "Turn on spectral shaping (FDSS) to smooth the transitions between symbols too. ")
                  + "NR added both for phones at the cell edge.</p>")
        s += ("<p>This is why LTE's uplink is SC-FDMA (DFT-s-OFDM) and NR keeps it as the coverage "
              "option: a phone's amplifier and battery care about every decibel; a base station "
              "can afford CP-OFDM and some crest-factor reduction (Lab 28).</p>")
        return "<h3>Peaks cost power</h3>" + s + keybox(
            "CP-OFDM: about 10–11 dB of PAPR at 10⁻³ whatever the data. DFT-s-OFDM: a few dB less, "
            "at the cost of flexible frequency scheduling.")


# =============================================================================== 8. numerology
def nr_cp_us(mu, extended=False):
    mu = int(mu)
    if extended:
        return 512 * 64 / (480e3 * 4096) / 2 ** mu * 1e6
    return co.nr_numerology(mu)["cp_us"]


class Numerology(Experiment):
    title = "5G NR numerology explorer"
    blurb = "One knob, μ: wider subcarriers fight Doppler and phase noise, narrower ones echoes."
    book = "sec:ch17:design"
    controls = [
        IntSlider("mu", "Numerology μ", 0, 6, 0, help="Subcarrier spacing 15·2^μ kHz"),
        Toggle("ext", "Extended cyclic prefix", False, enabled_if=lambda p: p.mu == 2,
               help="Only μ = 2 (60 kHz) has an extended-CP option (12 symbols per slot)"),
        Heading("The deployment"),
        LogSlider("fc", "Carrier frequency", 0.6, 40.0, 3.5, unit="GHz", fmt=".1f"),
        Slider("speed", "Speed", 0, 500, 60, step=5, unit="km/h"),
        LogSlider("tau", "RMS delay spread", 10, 3000, 300, unit="ns", fmt=".0f",
                  help="Exponential power-delay profile with this rms spread"),
        Slider("snr", "SNR", 0, 40, 25, step=0.5, unit="dB"),
    ]
    plots = [
        ImagePlot("slots", "One millisecond of each numerology (slots of 14 symbols)",
                  x="time (µs)", y=""),
        Plot("eff", "Net spectral efficiency vs subcarrier spacing", x="subcarrier spacing (kHz)",
             y="bits/s/Hz", logx=True, xlim=(7, 1500), legend="tr"),
        Plot("loss", "Where the efficiency goes", x="subcarrier spacing (kHz)", y="loss (%)",
             logx=True, xlim=(7, 1500), ylim=(0, 105), legend="br"),
    ]
    layout = [["slots", "slots"], ["eff", "loss"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("scs", "Subcarrier spacing", "kHz", ".0f"),
        Readout("cp", "Cyclic prefix", "µs", ".3f"),
        Readout("sir", "Doppler signal-to-ICI", "dB", ".1f", good=lambda x: x > 30),
        Readout("eff", "Net efficiency", "b/s/Hz", ".2f"),
    ]
    challenges = [
        Challenge("A 28 GHz small cell (rms delay spread ≤ 50 ns) serving users at 120 km/h or "
                  "faster: pick the μ with the highest net efficiency.",
                  lambda s: (abs(s.p.fc - 28) < 1.5 and s.p.speed >= 120 and s.p.tau <= 50
                             and s.p.mu == s.exp.best_mu)),
        Challenge("300 ns of rms delay spread: find the largest μ whose prefix keeps the ISI "
                  "share below 1 %.",
                  lambda s: (abs(s.p.tau - 300) < 15 and s.exp.isi_pct < 1
                             and s.exp.isi_next >= 1 and not s.p.ext),
                  hint="ISI ≈ exp(−T_cp/τ_rms): the prefix should be about 4.6 rms spreads."),
        Challenge("μ = 2 with 1 µs of rms delay spread: show the extended prefix beats the "
                  "normal one.",
                  lambda s: (s.p.mu == 2 and s.p.ext and s.p.tau >= 950
                             and s.r.eff > s.exp.eff_normal + 0.01)),
    ]

    def model(self, scs_hz, tcp_s, p):
        T = 1 / scs_hz
        fd = p.speed / 3.6 * p.fc * 1e9 / 299792458.0
        p_ici = (np.pi * fd * T) ** 2 / 6
        p_isi = np.exp(-tcp_s / (p.tau * 1e-9))
        snr = 10 ** (p.snr / 10)
        sinr = 1 / (1 / snr + p_ici + p_isi)
        eta = T / (T + tcp_s)
        return eta * np.log2(1 + sinr), 1 - eta, p_ici, p_isi, snr

    def update(self, p):
        mu = int(p.mu)
        ext = bool(p.ext and mu == 2)
        d = co.nr_numerology(mu)
        scs = d["scs_khz"] * 1e3
        tcp = nr_cp_us(mu, ext) * 1e-6
        # slot picture: rows mu = 0..6, columns 1 µs
        cols = 2000
        tt = (np.arange(cols) + 0.5) / cols * 1000
        img = np.zeros((7, cols))
        for m in range(7):
            slot = 1000 / 2 ** m
            par = (tt // slot) % 2
            row = 6 - m
            if m == mu:
                img[row] = np.where(par == 0, 0.55, 0.72)
                if m <= 4:
                    nsym = 12 if ext else 14
                    sym = (tt % slot) / slot * nsym
                    img[row][(sym % 1) < 0.06 * (1 + m * 0.6)] = 0.4
            else:
                img[row] = np.where(par == 0, 0.13, 0.22)
        pl = self.plot("slots")
        pl.image("img", img, x=(0, 1000), y=(-0.5, 6.5), cmap="heat", levels=(0, 1))
        pl.set_yticks([(6 - m, f"μ={m} · {15 * 2 ** m} kHz") for m in range(7)])
        pl.set_xlim(0, 1000)
        pl.set_ylim(-0.5, 6.5)
        # efficiency curves vs SCS with the CP scaling like NR (CP = 7 % of T)
        sg = np.logspace(np.log10(7e3), np.log10(1.5e6), 300)
        tcp_g = 144 / 2048 / sg
        eff, cpo, ici, isi, snr = self.model(sg, tcp_g, p)
        pe = self.plot("eff")
        pe.line("c", sg / 1e3, eff, color=NAVY, width=2.2, name="normal prefix (≈ 7 % of T)")
        mus = np.arange(7)
        effs = np.array([self.model(15e3 * 2.0 ** m, nr_cp_us(m) * 1e-6, p)[0] for m in mus])
        pe.scatter("nr", 15 * 2.0 ** mus, effs, color=GRAY, size=9, outline=NAVY, name="NR μ = 0…6")
        me, cpl, icl, isl, _ = self.model(scs, tcp, p)
        pe.scatter("now", [scs / 1e3], [me], color=RED, size=14, symbol="d", name="your choice")
        pe.set_ylim(0, max(1.0, 1.25 * np.max(effs)))
        tot = 1 / snr + ici + isi
        pls = self.plot("loss")
        pls.line("cp", sg / 1e3, 100 * cpo, color=GRAY, width=1.8, style="--", name="prefix overhead")
        pls.line("ici", sg / 1e3, 100 * ici / tot, color=ORANGE, width=2.0,
                 name="Doppler ICI share of the noise")
        pls.line("isi", sg / 1e3, 100 * isi / tot, color=PURPLE, width=2.0,
                 name="echo ISI share of the noise")
        pls.vline("now", scs / 1e3, color=RED, style=":", label=f"{scs / 1e3:.0f} kHz",
                  label_pos=0.6)
        ticks = [(np.log10(15 * 2 ** m), f"{15 * 2 ** m}") for m in range(7)]
        pls.set_xticks(ticks)
        pe.set_xticks(ticks)
        self.best_mu = int(np.argmax(effs))
        self.isi_pct = 100 * isl
        self.isi_next = 100 * float(np.exp(-nr_cp_us(min(mu + 1, 6)) * 1e-6 / (p.tau * 1e-9))) \
            if mu < 6 else 100.0
        self.eff_normal = float(self.model(scs, nr_cp_us(mu) * 1e-6, p)[0])
        fd = p.speed / 3.6 * p.fc * 1e9 / 299792458.0
        sir = 10 * np.log10(6 / max((np.pi * fd / scs) ** 2, 1e-12)) if fd > 0 else 99.0
        self.readout(scs=scs / 1e3, cp=tcp * 1e6,
                     sir=min(sir, 99.0), eff=float(me))
        self.fd = fd

    def story(self, p):
        mu = int(p.mu)
        d = co.nr_numerology(mu)
        cp = self.r.get("cp", 0)
        s = (f"<p>NR fixes one rule, Δf = 15·2<sup>μ</sup> kHz, and lets everything else scale: at "
             f"μ = {v(mu, 'd')} the subcarriers are {v(d['scs_khz'], '.0f', 'kHz')} apart, a symbol "
             f"lasts {v(d['useful_symbol_us'], '.2f', 'µs')} plus a {v(cp, '.2f', 'µs')} prefix, and "
             f"a 14-symbol slot takes {v(d['slot_ms'] * 1000, 'g', 'µs')} (top: the highlighted row). "
             f"Shorter slots mean lower latency.</p>")
        s += (f"<p>The prefix absorbs echoes up to {v(cp * 300, '.0f', 'm')} of path difference. At "
              f"{v(p.fc, '.1f', 'GHz')} and {v(p.speed, '.0f', 'km/h')} the Doppler is "
              f"{v(self.fd, '.0f', 'Hz')}, {v(100 * self.fd / (d['scs_khz'] * 1e3), '.2f', '%')} of a "
              f"subcarrier. Wide spacing shrinks the Doppler and phase-noise ICI (orange) but makes "
              f"the prefix shorter in time, so echoes leak (purple). The best μ (peak of the navy "
              f"curve) balances the two: here μ = {v(self.best_mu, 'd')}.</p>")
        s += ("<p>That is why FR1 cells (sub-6 GHz, kilometres wide) use 15–30 kHz and FR2 "
              "millimetre-wave cells (small, fast-varying, noisy oscillators) use 120 kHz.</p>")
        return "<h3>One family, scaled by powers of two</h3>" + s + keybox(
            "Every NR numerology keeps T·Δf = 1 and the prefix ≈ 7 % of T: choosing μ trades echo "
            "tolerance for Doppler tolerance.")


# =============================================================================== the lab
LAB = st.Lab(7, "OFDM: From the IFFT to the NR Grid", chapter=17,
             chapter_title="OFDM and Multicarrier",
             experiments=[Orthogonality, SymbolBuilder, CPvsMultipath, OneTapEqualizer,
                          SchmidlCox, CombPilots, PAPR, Numerology])

if __name__ == "__main__":
    st.run(LAB)
