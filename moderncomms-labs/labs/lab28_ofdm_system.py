"""Lab 28 · OFDM as a System: Estimation, ICI, Coding, Peaks, Spectrum, DMT and Radar   (Chapter 17)

Run it:      python labs/lab28_ofdm_system.py
Self-test:   python labs/lab28_ofdm_system.py --selftest

Lab 7 built the OFDM link. This lab takes the system view: the engineering trades around the
idea that make it work in a real product. Estimate hundreds of channel coefficients from a few
pilots; keep subcarriers orthogonal when oscillators wander and users move; turn deep fades into
diversity with a code; tame a waveform whose peaks sit 10 dB above its average; keep its spectrum
out of the neighbours' channels; load bits tone by tone on a copper pair; and, finally, use the
very same grid as a radar.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from scipy.special import erfc

import commlib as cl
from commlib import ofdm as co
from commlib import ofdmadv as oa
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, BarPlot, ImagePlot,
                    Readout, Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

# =============================================================================== shared helpers
FS = oa.FS_LTE                 # 15.36 MS/s, the LTE / NR 10 MHz sampling rate
NFFT, NUSED = 1024, 600
KSIG = co.OFDMConfig(NFFT, NUSED, 72).k.astype(float)
QPSK, Q16, Q64, Q256 = (cl.get_constellation(n) for n in ("qpsk", "16qam", "64qam", "256qam"))
C0 = 299_792_458.0


def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-30))


def qam(con, n, rng):
    return con.modulate(cl.random_bits(con.k * n, rng))


def cn(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def ccdf_fast(vals, grid):
    s = np.sort(np.asarray(vals))
    return 1.0 - np.searchsorted(s, grid, side="right") / max(len(s), 1)


def level_at(grid, cc, p):
    lc = np.log10(np.maximum(cc, 1e-9))
    if cc[0] < p:
        return float(grid[0])
    i = int(np.argmax(cc < p))
    if i == 0:
        return float(grid[-1])
    return float(np.interp(np.log10(p), [lc[i], lc[i - 1]], [grid[i], grid[i - 1]]))


# =============================================================================== 1. estimators
class EstimatorShootout(Experiment):
    title = "Channel estimation: LS, DFT, LMMSE"
    blurb = "Three ways to fill in the channel between the pilots, and what each one knows."
    book = "sec:ch17:grid"
    controls = [
        Choice("chan", "Channel", ["EPA", "EVA", "ETU"], "EVA"),
        Button("again", "New channel draw"),
        Choice("dp", "Pilot every", ["2", "3", "4", "6", "8", "12", "16", "24"], "6",
               help="Pilot spacing in subcarriers (LTE 10 MHz: 1024-point FFT, 600 subcarriers)"),
        Slider("snr", "Pilot SNR", 0, 40, 10, step=1, unit="dB"),
        Toggle("showdft", "Show the DFT estimate", True),
    ]
    plots = [
        Plot("H", "Channel and estimates (first 240 subcarriers)", x="subcarrier index",
             y="|H| (dB)", xlim=(-301, -60), ylim=(-32, 14), legend="tl", legend_cols=5),
        BarPlot("bars", "Estimation error at this SNR (this draw)", y="MSE (dB)"),
        Plot("mse", "Average error vs SNR (30 random channels)", x="pilot SNR (dB)",
             y="MSE (dB)", xlim=(-1, 41), ylim=(-55, 5), legend="bl"),
    ]
    layout = [["H", "H"], ["bars", "mse"]]
    row_stretch = [1, 1]
    col_stretch = [2, 3]
    readouts = [
        Readout("nyq", "Nyquist limit on spacing", "subcarriers", ".1f"),
        Readout("lin", "Linear interpolation", "dB", ".1f"),
        Readout("dft", "DFT-based", "dB", ".1f"),
        Readout("mmse", "LMMSE", "dB", ".1f", good=lambda x: x < -20),
    ]
    challenges = [
        Challenge("At 0 dB pilot SNR, make LMMSE beat linear interpolation by at least 10 dB.",
                  lambda s: s.p.snr <= 0 and s.r.lin - s.r.mmse >= 10,
                  hint="LMMSE averages many noisy pilots, weighted by the channel's known "
                       "frequency correlation. It gains most when there are many pilots per "
                       "coherence bandwidth."),
        Challenge("Starve the estimators: ETU with pilots beyond the Nyquist limit, linear "
                  "interpolation error above −5 dB. How much does LMMSE salvage?",
                  lambda s: s.p.chan == "ETU" and int(s.p.dp) > s.r.nyq and s.r.lin > -5),
        Challenge("Find the DFT estimator's floor: at 35 dB SNR or more on EVA, see it trail "
                  "LMMSE by at least 5 dB.",
                  lambda s: s.p.chan == "EVA" and s.p.snr >= 35 and s.r.dft - s.r.mmse >= 5,
                  hint="EVA's path delays do not fall on the DFT estimator's delay grid: their "
                       "energy leaks."),
    ]
    maxdel = 72

    def setup(self):
        self.seed = 41
        self._W = {}
        self.curves = {}

    def on_again(self, p):
        self.seed += 1

    def weights(self, chan, dp, snr):
        key = (chan, dp, snr)
        if key not in self._W:
            d, pw = oa.tdl_pdp(chan, FS)
            kp = KSIG[np.arange(0, NUSED, dp)]
            if len(self._W) > 40:
                self._W.clear()
            self._W[key] = oa.lmmse_weights(kp, KSIG, 10 ** (-snr / 10), d, pw, NFFT)
        return self._W[key]

    def estimates(self, Hp, kp, W):
        lin = oa.chest_ls_interp(Hp, kp, KSIG)
        dft = oa.chest_dft(Hp, kp, KSIG, NFFT, self.maxdel)
        return lin, dft, W @ Hp

    def update(self, p):
        dp = int(p.dp)
        pidx = np.arange(0, NUSED, dp)
        kp = KSIG[pidx]
        rng = np.random.default_rng(self.seed)
        H = oa.tdl_freq_response(p.chan, KSIG, NFFT, FS, rng)
        n0 = 10 ** (-p.snr / 10)
        Hls = H + np.sqrt(n0) * cn(NUSED, np.random.default_rng(self.seed + 5))
        Hp = Hls[pidx]
        lin, dft, mm = self.estimates(Hp, kp, self.weights(p.chan, dp, p.snr))
        mse = [np.mean(np.abs(Hp - H[pidx]) ** 2)] + [np.mean(np.abs(e - H) ** 2) for e in (lin, dft, mm)]
        mse = [float(db(m)) for m in mse]
        sel = KSIG <= -60
        ph = self.plot("H")
        ph.line("H", KSIG[sel], db(np.abs(H[sel]) ** 2), color=GRAY, width=4.0, alpha=0.6,
                name="true")
        ps = kp <= -60
        ph.scatter("p", kp[ps], db(np.abs(Hp[ps]) ** 2), color=ORANGE, size=6, name="LS pilots")
        ph.line("lin", KSIG[sel], db(np.abs(lin[sel]) ** 2), color=RED, width=1.3, name="linear")
        if p.showdft:
            ph.line("dft", KSIG[sel], db(np.abs(dft[sel]) ** 2), color=GREEN, width=1.3, name="DFT")
        ph.line("mm", KSIG[sel], db(np.abs(mm[sel]) ** 2), color=NAVY, width=2.0, style="--",
                name="LMMSE")
        pb = self.plot("bars")
        cols = [ORANGE, RED, GREEN, NAVY]
        pb.bars("b", np.arange(4), np.maximum(mse, -59), width=0.6, colors=cols, base=-60)
        for i, m in enumerate(mse):
            pb.text(f"t{i}", i, max(m, -59), f"{m:.1f}", anchor=(0.5, 1.1), bold=True, size=9)
        pb.set_xticks([(0, "LS at pilots"), (1, "linear"), (2, "DFT"), (3, "LMMSE")])
        pb.set_xlim(-0.6, 3.6)
        pb.set_ylim(-60, 5)
        pm = self.plot("mse")
        pm.vline("now", p.snr, color=GRAY, style=":")
        pm.line("n0", [0, 40], [0, -40], color=GRAY, width=1.6, style="--", name="pilot noise N₀")
        self.cur_key = (p.chan, dp)
        self.draw_curves(p)
        dmax = cl.TDL_PROFILES[p.chan][0][-1] * 1e-9
        self.readout(nyq=1 / (dmax * 15e3), lin=mse[1], dft=mse[2], mmse=mse[3])

    def draw_curves(self, p):
        c = self.curves.get(self.cur_key)
        if not c:
            return
        pm = self.plot("mse")
        x = c["x"]
        for k, col, nm in (("lin", RED, "linear"), ("dft", GREEN, "DFT-based"), ("mm", NAVY, "LMMSE")):
            pm.line(k, x, c[k], color=col, width=2.0, name=nm)
            pm.scatter(k + "d", x, c[k], color=col, size=6)

    def background(self, p):
        key = (p.chan, int(p.dp))
        if key in self.curves and self.curves[key].get("done"):
            return
        dp = int(p.dp)
        pidx = np.arange(0, NUSED, dp)
        kp = KSIG[pidx]
        d, pw = oa.tdl_pdp(p.chan, FS)
        rng = np.random.default_rng(99)
        draws = 6 if self.quick else 30
        out = dict(x=[], lin=[], dft=[], mm=[])
        for snr in np.arange(0, 41, 5.0):
            n0 = 10 ** (-snr / 10)
            W = oa.lmmse_weights(kp, KSIG, n0, d, pw, NFFT)
            H = oa.tdl_freq_response(p.chan, KSIG, NFFT, FS, rng, n_draws=draws)
            acc = np.zeros(3)
            for t in range(draws):
                Hp = (H[t] + np.sqrt(n0) * cn(NUSED, rng))[pidx]
                est = self.estimates(Hp, kp, W)
                acc += [np.mean(np.abs(e - H[t]) ** 2) for e in est]
            out["x"].append(snr)
            for k, a in zip(("lin", "dft", "mm"), acc / draws):
                out[k].append(float(db(a)))
            yield key, dict(out)
        out["done"] = True
        yield key, out

    def progress(self, p, item):
        key, out = item
        self.curves[key] = out
        if key == getattr(self, "cur_key", None):
            self.draw_curves(p)

    def story(self, p):
        r = self.r
        s = ("<p>At the pilots, the receiver measures Hₖ directly but noisily (orange dots, error "
             "≈ N₀). Everywhere else it must <i>infer</i> Hₖ, and the three estimators differ in what "
             "they assume:</p>"
             "<p><b>Linear interpolation</b> assumes nothing: it joins the dots, noise and all, and "
             "cuts corners in the fades. <b>DFT-based</b> estimation assumes the impulse response "
             "fits inside the prefix: it transforms the pilots to the delay domain, keeps the first "
             "few taps and throws the rest (pure noise) away. <b>LMMSE</b> knows the channel's "
             "power-delay profile and the noise level, and weighs every pilot accordingly.</p>")
        if r.get("lin") is not None:
            s += (f"<p>On this draw: linear {v(r['lin'], '.1f', 'dB')}, DFT {v(r['dft'], '.1f', 'dB')}, "
                  f"LMMSE {v(r['mmse'], '.1f', 'dB')}. ")
            if int(p.dp) > r.get("nyq", 99):
                s += bad("The pilots are too sparse for this channel: nothing can recover what was "
                         "never sampled.") + "</p>"
            else:
                s += ("The average curves (right) show the pattern: LMMSE wins everywhere; linear "
                      "interpolation and DFT both floor at high SNR, from bias rather than "
                      "noise.</p>")
        return "<h3>Filling in between the pilots</h3>" + s + keybox(
            "Ĥ = R<sub>hp</sub>(R<sub>pp</sub> + N₀I)⁻¹·Ĥ<sub>p</sub>: the more you know about the "
            "channel's statistics, the fewer pilots you need.")


# =============================================================================== 2. ICI
EVM_LIMIT = {"16-QAM": 12.5, "64-QAM": 8.0, "256-QAM": 3.5}   # 3GPP TS 38.104 base-station EVM


class ICIImpairments(Experiment):
    title = "Frequency offset, Doppler, phase noise"
    blurb = "Three ways to lose orthogonality, and the one part of it that pilots can fix."
    book = "sec:ch17:sync"
    controls = [
        Choice("imp", "Impairment", ["Frequency offset", "Doppler (Jakes)", "Phase noise"],
               "Phase noise", style="menu"),
        LogSlider("x", "Size: ε, f_D·T or β·T", 0.0005, 0.3, 0.01, fmt=".4f",
                  help="Normalised to the subcarrier spacing: CFO ε, maximum Doppler × symbol "
                       "time, or oscillator 3-dB linewidth × symbol time"),
        Choice("mod", "Modulation", ["16-QAM", "64-QAM", "256-QAM"], "64-QAM"),
        Toggle("cpe", "Pilot phase and gain correction", True,
               help="Every 12th subcarrier is a pilot; each symbol is de-rotated (and scaled) by "
                    "their average"),
        Button("again", "New realisation"),
    ]
    plots = [
        ConstellationPlot("const", "Constellation after the FFT", lim=1.55),
        Plot("sir", "Signal-to-ICI ratio: theory (lines) and this run", x="ε, f_D·T or β·T",
             y="SIR (dB)", logx=True, xlim=(0.0005, 0.3), ylim=(0, 62), legend="tr"),
        Plot("cpe", "Common phase error of each OFDM symbol", x="OFDM symbol",
             y="phase (°)", xlim=(-0.5, 39.5), legend="tl"),
    ]
    layout = [["const", "sir"], ["cpe", "sir"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("sir", "Measured SIR", "dB", ".1f"),
        Readout("th", "Theory", "dB", ".1f"),
        Readout("evm", "EVM", "%", ".1f"),
        Readout("ok", "Supports", "", None),
    ]
    challenges = [
        Challenge("Phase noise on 64-QAM: find the largest linewidth (β·T ≥ 0.005) that still "
                  "meets the 8 % EVM limit with the common phase corrected.",
                  lambda s: (s.p.imp == "Phase noise" and s.p.cpe and s.p.mod == "64-QAM"
                             and s.p.x >= 0.005 and s.r.evm <= 8.0),
                  hint="With the CPE removed, the residual ICI is about πβT/3."),
        Challenge("Switch the pilot correction off with phase noise as small as β·T ≤ 0.002 and "
                  "watch the EVM exceed 15 %.",
                  lambda s: (s.p.imp == "Phase noise" and not s.p.cpe and s.p.x <= 0.002
                             and s.r.evm > 15)),
        Challenge("Doppler: find the f_D·T at which 256-QAM just stops working (SIR between 27 and "
                  "29 dB).",
                  lambda s: s.p.imp == "Doppler (Jakes)" and s.p.cpe and 27 <= s.r.sir <= 29,
                  hint="Theory: SIR ≈ 6/(π·f_D·T)²."),
    ]
    N, nu, ncp, nsym = 256, 200, 16, 40

    def setup(self):
        self.seed = 51
        self.cfg = co.OFDMConfig(self.N, self.nu, self.ncp)

    def on_again(self, p):
        self.seed += 1

    def theory(self, imp, x):
        x = np.asarray(x, float)
        if imp == "Frequency offset":
            return oa.sir_cfo_db(x)
        if imp == "Doppler (Jakes)":
            return oa.sir_doppler_db(x)
        return -db(np.pi * x / 3)

    def update(self, p):
        con = {"16-QAM": Q16, "64-QAM": Q64, "256-QAM": Q256}[p.mod]
        rng = np.random.default_rng(self.seed)
        X = qam(con, self.nu * self.nsym, np.random.default_rng(7)).reshape(self.nsym, self.nu)
        x = co.ofdm_modulate(X, self.cfg)
        n = np.arange(len(x))
        if p.imp == "Frequency offset":
            y = x * np.exp(2j * np.pi * p.x / self.N * n)
        elif p.imp == "Doppler (Jakes)":
            y = x * cl.jakes_process(len(x), p.x / self.N, n_sin=24, rng=rng)
        else:
            y = x * cl.phase_noise(len(x), p.x / self.N, rng=rng)
        Y = co.ofdm_demodulate(y, self.cfg)
        pil = np.zeros_like(X)
        pil[:, ::12] = X[:, ::12]
        g = np.sum(Y * np.conj(pil), axis=1) / np.sum(np.abs(pil) ** 2, axis=1)
        cpe_deg = np.degrees(np.angle(g))
        Z = Y / g[:, None] if p.cpe else Y
        # SIR after removing each symbol's common gain (what an ideal one-tap correction leaves)
        gt = np.sum(Y * np.conj(X), axis=1) / np.sum(np.abs(X) ** 2, axis=1)
        sir = float(db(np.mean(np.abs(gt[:, None] * X) ** 2) / np.mean(np.abs(Y - gt[:, None] * X) ** 2)))
        evm = 100 * np.sqrt(np.mean(np.abs(Z - X) ** 2) / np.mean(np.abs(X) ** 2))
        th = float(self.theory(p.imp, p.x))
        pc = self.plot("const")
        pc.points("z", Z[:, 1::2].ravel()[:2500], color=NAVY, size=2.5, alpha=0.45)
        pc.ideal("i", con.points, color=RED, size=7)
        ps = self.plot("sir")
        xg = np.logspace(np.log10(0.0005), np.log10(0.3), 200)
        for imp, col, nm in (("Frequency offset", NAVY, "CFO: sinc²ε/(1 − sinc²ε)"),
                             ("Doppler (Jakes)", RED, "Doppler: 6/(π f_D T)²"),
                             ("Phase noise", GREEN, "phase noise, CPE removed: 3/(πβT)")):
            ps.line(imp, xg, np.minimum(self.theory(imp, xg), 70), color=col,
                    width=2.6 if imp == p.imp else 1.2, alpha=1.0 if imp == p.imp else 0.5, name=nm)
        for k, (m, lim) in enumerate(EVM_LIMIT.items()):
            ps.hline(f"l{k}", -20 * np.log10(lim / 100), color=GRAY, style=":", width=0.9,
                     label=f"{m} ({lim:g} % EVM)", label_pos=0.72)
        ps.scatter("now", [p.x], [min(sir, 61)], color=ORANGE, size=14, symbol="d",
                   name="this run")
        pp = self.plot("cpe")
        pp.hline("z", 0, color=GRAY, style="-", width=0.7)
        pp.line("c", np.arange(self.nsym), cpe_deg, color=PURPLE, width=1.8,
                name="common phase (from the pilots)")
        pp.scatter("cd", np.arange(self.nsym), cpe_deg, color=PURPLE, size=5)
        lim = max(5.0, 1.2 * np.max(np.abs(cpe_deg)))
        pp.set_ylim(-lim, lim)
        ok = [m for m, l_ in EVM_LIMIT.items() if evm <= l_]
        self.readout(sir=min(sir, 99.0), th=min(th, 99.0), evm=float(evm),
                     ok=ok[-1] if ok else ("QPSK only" if evm <= 17.5 else "nothing"))

    def story(self, p):
        x, sir, evm = p.x, self.r.get("sir", 0), self.r.get("evm", 0)
        if p.imp == "Frequency offset":
            s = (f"<p>A frequency offset of ε = {v(x, '.4f')} subcarrier spacings shifts every "
                 f"subcarrier off its FFT bin by the same amount. Part of the damage is common to all "
                 f"subcarriers: a rotation that grows from symbol to symbol (purple, bottom right) and "
                 f"a small loss of amplitude. Pilots remove that. The rest leaks into the "
                 f"neighbours: ICI at {v(sir, '.1f', 'dB')}, matching sinc²ε/(1 − sinc²ε).</p>")
        elif p.imp == "Doppler (Jakes)":
            s = (f"<p>Doppler spread is a frequency offset that differs for every path: the channel "
                 f"changes <i>within</i> a symbol. With f_D·T = {v(x, '.4f')}, the average change "
                 f"per symbol is a common gain the pilots can track, but the variation inside "
                 f"the symbol leaks as ICI: {v(sir, '.1f', 'dB')}, about 3 dB kinder than a CFO "
                 f"of the same size (6/(πf_DT)² instead of 3/(πε)²).</p>")
        else:
            s = (f"<p>Oscillator phase noise is a random walk of the carrier phase (linewidth × T = "
                 f"{v(x, '.4f')}). Its average over each symbol is the <b>common phase error</b>: "
                 f"the whole constellation rotates together (purple trace). Pilots measure and remove "
                 f"it. What they cannot remove is the wobble <i>within</i> a symbol, which spreads "
                 f"energy into the neighbours: an ICI floor ≈ πβT/3.</p>")
        if not p.cpe:
            s += (f"<p>{bad('Correction off:')} the common rotation alone spins the constellation "
                  f"into arcs (EVM {v(evm, '.1f', '%')}).</p>")
        else:
            s += (f"<p>After pilot correction the EVM is {v(evm, '.1f', '%')}: good enough for "
                  f"{v(self.r.get('ok', '—'))}.</p>")
        s += ("<p>Real numbers (Chapter 17's 28 GHz budget): a 0.1 ppm crystal error is "
              "ε = 0.023 at 120 kHz spacing but 0.19 at 15 kHz. Millimetre-wave oscillators are "
              "noisy too, which is why FR2 uses wide subcarriers and dedicated phase-tracking "
              "pilots (PT-RS).</p>")
        return "<h3>Losing orthogonality</h3>" + s + keybox(
            "Common errors (one rotation per symbol) are cheap to fix with pilots; ICI (energy "
            "leaking between subcarriers) is a floor you avoid with wider subcarrier spacing.")


# =============================================================================== 3. coded OFDM
def ber_awgn_bpsk(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * erfc(np.sqrt(g))


def ber_rayleigh_bpsk(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


class CodedOFDM(Experiment):
    title = "Coded OFDM and interleaving"
    blurb = "A code spread across the band turns a fade on one subcarrier into a correctable error."
    book = "sec:ch17:coded"
    heavy = True
    controls = [
        Choice("chan", "Channel", ["AWGN", "EPA", "EVA", "ETU"], "ETU"),
        Toggle("coded", "Rate-½ code (K = 7)", False),
        Toggle("inter", "Interleave across the band", True,
               enabled_if=lambda p: p.coded),
        Slider("ebn0", "Eb/N0 for the snapshot", 0, 20, 8, step=0.5, unit="dB"),
        Choice("effort", "Monte Carlo effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [
        BERPlot("ber", "BER: QPSK on 600 subcarriers, one codeword per symbol",
                x="Eb/N0 (dB)", ylim=(1e-5, 0.5), xlim=(-0.5, 20.5), legend="bl"),
        Plot("snap", "One OFDM symbol: channel and where the bit errors land", x="subcarrier",
             y="|Hₖ|² (dB)", xlim=(-302, 302), ylim=(-32, 16), legend="tr", legend_cols=2),
    ]
    layout = [["ber", "snap"]]
    col_stretch = [5, 6]
    readouts = [
        Readout("raw", "Channel bit errors", "", "sci"),
        Readout("dec", "After decoding", "", "sci", good=lambda x: x < 1e-4),
        Readout("need", "Eb/N0 for 10⁻³ (simulated)", "dB", ".1f"),
        Readout("pts", "Monte Carlo", "", None),
    ]
    challenges = [
        Challenge("ETU: make the link reach BER 10⁻³ at an Eb/N0 of 8.5 dB or less (uncoded "
                  "needs about 24 dB).",
                  lambda s: (s.p.chan == "ETU" and s.p.coded and s.r.need is not None
                             and s.r.need <= 8.5),
                  hint="A code can only use diversity it can see: adjacent coded bits sit on "
                       "adjacent (equally faded) subcarriers unless something spreads them."),
        Challenge("Measure what the interleaver is worth: ETU, coded, no interleaver, needing 9 dB "
                  "or more for 10⁻³.",
                  lambda s: (s.p.chan == "ETU" and s.p.coded and not s.p.inter
                             and s.r.need is not None and s.r.need >= 9)),
        Challenge("Short delay spread, less diversity: show that interleaved EPA needs at least "
                  "3 dB more than interleaved ETU (about 8 dB).",
                  lambda s: (s.p.chan == "EPA" and s.p.coded and s.p.inter
                             and s.r.need is not None and s.r.need >= 11)),
    ]
    B_snap = 6

    def setup(self):
        self.code = cl.ConvCode()
        self.ninfo = NUSED - (self.code.K - 1)
        self.perm = np.random.default_rng(17).permutation(2 * NUSED)
        self.ghost = None
        self.sim = None
        self.run_id = 0

    def on_rerun(self, p):
        self.run_id += 1

    def chan(self, profile, B, rng):
        if profile == "AWGN":
            return np.ones((B, NUSED), complex)
        return oa.tdl_freq_response(profile, KSIG, NFFT, FS, rng, n_draws=B)

    def block(self, p, ebn0, B, rng):
        """Simulate B OFDM symbols (one codeword each). Returns errors, bits, raw errors, H, raw
        error mask and decoded error count of the first symbol."""
        coded, inter = p.coded, p.coded and p.inter
        rate = self.ninfo / (2 * NUSED) if coded else 1.0
        esn0 = ebn0 + 10 * np.log10(2 * rate)
        n0 = 10 ** (-esn0 / 10)
        H = self.chan(p.chan, B, rng)
        if coded:
            u = rng.integers(0, 2, (B, self.ninfo))
            c = self.code.encode_batch(u)
            ci = c[:, self.perm] if inter else c
        else:
            ci = rng.integers(0, 2, (B, 2 * NUSED))
        s = QPSK.modulate(ci.ravel()).reshape(B, NUSED)
        y = H * s + np.sqrt(n0) * cn((B, NUSED), rng)
        llr = QPSK.llr(y.ravel(), n0, h=H.ravel()).reshape(B, 2 * NUSED)
        rawmask = (llr < 0) != ci
        if coded:
            if inter:
                l2 = np.empty_like(llr)
                l2[:, self.perm] = llr
                llr = l2
            uh = self.code.decode_batch(llr)
            err = uh != u
            return int(err.sum()), u.size, int(rawmask.sum()), rawmask.size, H, rawmask, err
        return int(rawmask.sum()), rawmask.size, int(rawmask.sum()), rawmask.size, H, rawmask, None

    def update(self, p):
        rng = np.random.default_rng(1000 + self.run_id)
        e, n, re, rn, H, raw, err = self.block(p, p.ebn0, self.B_snap, rng)
        ps = self.plot("snap")
        hk = db(np.abs(H[0]) ** 2)
        ps.line("H", KSIG, hk, color=NAVY, width=1.6, fill=-32, fill_alpha=0.10, name="|Hₖ|²")
        sub = np.flatnonzero(raw[0].reshape(NUSED, 2).any(axis=1))
        ps.scatter("raw", KSIG[sub], hk[sub], color=RED, size=6, name="channel bit errors")
        if err is not None:
            nd = int(err[0].sum())
            ps.text("dec", -295, -31, f"after decoding: {nd} bit error{'s' if nd != 1 else ''} "
                    f"in this symbol", color=GREEN if nd == 0 else RED, size=9.5, bold=True,
                    anchor=(0, 1))
        # BER plot reset
        pb = self.plot("ber")
        if self.sim is not None and self.sim.get("done") and len(self.sim["x"]) > 2:
            self.ghost = dict(self.sim)
        self.sim = dict(x=[], y=[], done=False)
        eb = np.linspace(0, 20, 201)
        pb.theory("awgn", eb, ber_awgn_bpsk(eb), color=GRAY, name="uncoded, AWGN")
        pb.theory("ray", eb, ber_rayleigh_bpsk(eb), color=GRAY, style="--",
                  name="uncoded, flat Rayleigh")
        if self.ghost is not None:
            pb.sim("ghost", self.ghost["x"], self.ghost["y"], color=GRAY, size=7, name="previous run")
        pb.vline("now", p.ebn0, color=ORANGE, style=":", width=1.0)
        pb.hline("e3", 1e-3, color=GRAY, style=":", width=0.8)
        self.readout(raw=re / rn, dec=(e / n) if p.coded else re / rn, need=None, pts="…")

    def background(self, p):
        thorough = p.effort == "Thorough" and not self.quick
        B = 30 if self.quick else 60
        target = 20 if self.quick else (300 if thorough else 80)
        maxbits = 3e4 if self.quick else (4e6 if thorough else 6e5)
        rng = np.random.default_rng(7 + 31 * self.run_id)
        for eb in np.arange(0, 20.1, 2.0 if self.quick else 1.0):
            errs = bits = 0
            while errs < target and bits < maxbits:
                e, n, *_ = self.block(p, eb, B, rng)
                errs += e
                bits += n
            ber = errs / bits
            yield dict(x=float(eb), y=max(ber, 1e-7))
            if ber < (2e-5 if not self.quick else 1e-3) or errs == 0:
                break
        yield dict(done=True)

    def progress(self, p, it):
        if it.get("done"):
            self.sim["done"] = True
        else:
            self.sim["x"].append(it["x"])
            self.sim["y"].append(it["y"])
        pb = self.plot("ber")
        lab = ("coded" if p.coded else "uncoded") + (", interleaved" if p.coded and p.inter else "")
        pb.line("simln", self.sim["x"], self.sim["y"], color=RED, width=1.4)
        pb.scatter("sim", self.sim["x"], self.sim["y"], color=RED, size=9, name=f"{lab}, {p.chan}")
        need = self.need()
        self.readout(need=need, pts=f"{len(self.sim['x'])}" + (" ✓" if self.sim["done"] else " …"))

    def need(self):
        x, y = np.array(self.sim["x"]), np.array(self.sim["y"])
        if len(x) < 2 or y.min() > 1e-3 or y.max() < 1e-3:
            return None
        ly = np.log10(y)
        i = int(np.argmax(ly <= -3))
        if i == 0:
            return float(x[0])
        return float(np.interp(-3, [ly[i], ly[i - 1]], [x[i], x[i - 1]]))

    def story(self, p):
        need = self.r.get("need")
        s = ("<p>Each subcarrier is a flat channel with its own gain |Hₖ|² (right). Uncoded, the "
             "errors (red dots) pile up in the fades, and the error rate follows the flat-Rayleigh "
             "curve: 10⁻³ needs about 24 dB.</p>")
        if p.coded and p.inter:
            s += ("<p>With the code <b>and</b> the interleaver, neighbouring coded bits travel on "
                  "subcarriers far apart, in independent fades. The Viterbi decoder, fed soft "
                  "values that know each subcarrier's gain, sees a few unreliable bits scattered "
                  "among many good ones and repairs them. That is <b>frequency diversity</b>.</p>")
        elif p.coded:
            s += ("<p>The code alone helps less than you would hope: consecutive coded bits sit on "
                  "neighbouring subcarriers, inside the <i>same</i> fade, so the decoder meets a "
                  "burst of bad bits longer than it can bridge. Switch on the interleaver.</p>")
        if p.chan == "EPA":
            s += ("<p>EPA's short delay spread makes H(f) vary slowly: fewer independent fades fit "
                  "in the band, so there is less diversity to collect.</p>")
        if need is not None:
            s += f"<p>Simulated: 10⁻³ at {v(need, '.1f', 'dB')} Eb/N0.</p>"
        s += ("<p>This is COFDM, the reason DAB, DVB-T, Wi-Fi and LTE never transmit OFDM without "
              "a code and an interleaver.</p>")
        return "<h3>Diversity from a code</h3>" + s + keybox(
            "Uncoded OFDM in fading is Rayleigh: BER ∝ 1/SNR. Coding + interleaving + soft "
            "decisions recover the band's frequency diversity.")


# =============================================================================== 4. clipping and the PA
class ClipAndPA(Experiment):
    title = "Clipping, filtering and the PA"
    blurb = "Shave the peaks yourself before the amplifier does it for you, badly."
    book = "sec:ch17:papr"
    controls = [
        Heading("Crest-factor reduction"),
        Toggle("cfr", "Clip and filter (CFR)", True),
        Slider("clip", "Clip level above RMS", 2, 9, 5, step=0.25, unit="dB",
               enabled_if=lambda p: p.cfr),
        IntSlider("iters", "Clip–filter iterations", 1, 8, 3, enabled_if=lambda p: p.cfr),
        Heading("Power amplifier"),
        Slider("ibo", "Input back-off", 0, 12, 4, step=0.25, unit="dB",
               help="Mean input power below the amplifier's saturation"),
        Choice("pa", "PA characteristic", ["Soft (p = 2)", "Firm (p = 3)", "With DPD (p = 10)"],
               "With DPD (p = 10)", style="menu",
               help="Rapp model smoothness. p = 10 is close to an ideal limiter, which is what "
                    "a PA linearised by digital predistortion looks like"),
    ]
    plots = [
        Plot("ccdf", "Peaks: CCDF of the symbol PAPR", x="PAPR₀ (dB)", y="P(PAPR > PAPR₀)",
             logy=True, xlim=(3, 13), ylim=(2e-3, 1), legend="bl"),
        SpectrumPlot("spec", "Spectrum after the PA", x="frequency (subcarrier spacings)",
                     y="PSD (dB)", xlim=(-512, 512), ylim=(-85, 5), legend="tl"),
        ConstellationPlot("const", "16-QAM after the PA", lim=1.5),
        Plot("trade", "ACLR and EVM vs back-off (your CFR setting)", x="input back-off (dB)",
             y="dB", xlim=(0, 12), ylim=(10, 75), legend="br"),
    ]
    layout = [["ccdf", "spec", "spec"], ["const", "trade", "trade"]]
    readouts = [
        Readout("papr", "PAPR at 10⁻²", "dB", ".1f"),
        Readout("evm", "EVM after the PA", "dB", ".1f", good=lambda x: x < -25),
        Readout("aclr", "ACLR", "dB", ".1f", good=lambda x: x >= 45),
        Readout("eff", "Class-B efficiency", "%", ".0f"),
    ]
    challenges = [
        Challenge("Meet a 45 dB ACLR and 16-QAM's EVM limit (12.5 %, −18 dB) with an input "
                  "back-off of 5 dB or less.",
                  lambda s: s.r.aclr >= 45 and s.r.evm <= -18 and s.p.ibo <= 5,
                  hint="Clip a little below the level where the PA would clip anyway."),
        Challenge("Without CFR, find how much back-off a 45 dB ACLR needs.",
                  lambda s: not s.p.cfr and s.r.aclr >= 45,
                  hint="Uncontrolled peaks hit the PA's ceiling and splatter into the next channel."),
        Challenge("Overdo it: clip so hard that the CFR alone costs more than −20 dB of EVM.",
                  lambda s: s.p.cfr and s.exp.evm_cfr > -20),
    ]
    nf, nu, L, nsym = 256, 200, 4, 320

    def setup(self):
        rng = np.random.default_rng(28)
        self.cfg = co.OFDMConfig(self.L * self.nf, self.nu, 0)
        self.g = qam(Q16, self.nu * self.nsym, rng).reshape(self.nsym, self.nu)
        self.x = co.ofdm_modulate(self.g, self.cfg).reshape(self.nsym, -1)
        self.rms = np.sqrt(np.mean(np.abs(self.x) ** 2))
        self.grid = np.linspace(3, 13, 201)
        self._cfr_key = None
        self.trade = {}
        self.c0 = ccdf_fast(co.papr_db(self.x.ravel(), self.L * self.nf), self.grid)

    def pa(self, y, ibo, p_):
        z = cl.rapp_pa(y.ravel() / self.rms, sat=10 ** (ibo / 20), p=p_).reshape(y.shape)
        return z * self.rms

    def pexp(self, p):
        return {"Soft (p = 2)": 2.0, "Firm (p = 3)": 3.0}.get(p.pa, 10.0)

    def cfr(self, p):
        key = (p.cfr, round(p.clip, 3), int(p.iters))
        if key != self._cfr_key:
            self._y = oa.clip_filter(self.x, self.cfg, p.clip, int(p.iters)) if p.cfr else self.x
            self._cfr_key = key
        return self._y

    def measure(self, z):
        return oa.bussgang_evm_db(z, self.g, self.cfg, self.g), oa.aclr_symbols(z, self.cfg)

    def update(self, p):
        y = self.cfr(p)
        pe = self.pexp(p)
        z = self.pa(y, p.ibo, pe)
        evm, aclr = self.measure(z)
        self.evm_cfr = float(oa.bussgang_evm_db(y, self.g, self.cfg, self.g)) if p.cfr else -99.0
        # CCDF
        pc = self.plot("ccdf")
        c0 = self.c0
        pc.line("orig", self.grid, np.where(c0 > 0, c0, np.nan), color=NAVY, width=2.0,
                name="original")
        papr = level_at(self.grid, c0, 1e-2)
        if p.cfr:
            c1 = ccdf_fast(co.papr_db(y.ravel(), self.L * self.nf), self.grid)
            pc.line("cfr", self.grid, np.where(c1 > 0, c1, np.nan), color=GREEN, width=2.2,
                    name=f"after CFR (EVM {self.evm_cfr:.0f} dB)")
            papr = level_at(self.grid, c1, 1e-2)
        pc.hline("e2", 1e-2, color=GRAY, style=":", width=0.8)
        pc.vline("ibo", p.ibo, color=ORANGE, style="--", label="PA back-off", label_pos=0.15)
        # spectrum
        ps = self.plot("spec")
        fb = np.arange(-self.L * self.nf // 2, self.L * self.nf // 2)

        def spec(w):
            P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(w, axis=1)) ** 2, axis=0))
            return db(P / np.median(P[np.abs(fb) < 60]))
        if p.cfr:
            rk = (round(p.ibo, 3), pe)
            if getattr(self, "_refk", None) != rk:
                self._ref = spec(self.pa(self.x, p.ibo, pe))
                self._refk = rk
            ps.line("ref", fb, self._ref, color=GRAY, width=1.0, name="no CFR, same back-off")
        ps.line("now", fb, spec(z), color=GREEN if p.cfr else NAVY, width=1.4,
                name="your transmitter")
        g = 0.05 * self.nu
        ps.band("adjr", self.nu / 2 + g, self.nu * 1.5 + g, color=RED, alpha=0.06)
        ps.band("adjl", -self.nu * 1.5 - g, -self.nu / 2 - g, color=RED, alpha=0.06)
        ps.text("al", self.nu / 2 + g + 5, 0, "adjacent channel", color=RED, size=8.5, anchor=(0, 0))
        # constellation
        Zf = np.fft.fft(z[:12], axis=1)[:, self.cfg.active]
        a = np.sum(Zf * np.conj(self.g[:12])) / np.sum(np.abs(self.g[:12]) ** 2)
        pk = self.plot("const")
        pk.points("z", (Zf / a).ravel(), color=GREEN if p.cfr else NAVY, size=2.5, alpha=0.45)
        pk.ideal("i", Q16.points / np.sqrt(np.mean(np.abs(Q16.points) ** 2)), color=RED, size=8)
        # efficiency at the output back-off
        obo = p.ibo - float(db(np.mean(np.abs(z) ** 2) / self.rms ** 2))   # output below saturation
        eff = 78.5 * 10 ** (-max(obo, 0) / 20)
        self.trade_key = (self._cfr_key, pe)
        self.draw_trade(p)
        self.readout(papr=papr, evm=float(evm), aclr=float(aclr), eff=float(eff))

    def draw_trade(self, p):
        t = self.trade.get(self.trade_key)
        pt = self.plot("trade")
        pt.hline("t45", 45, color=GRAY, style=":", label="ACLR target 45 dB", label_pos=0.02)
        pt.vline("now", p.ibo, color=ORANGE, style="--")
        if not t:
            return
        pt.line("aclr", t["x"], t["aclr"], color=NAVY, width=2.0, name="ACLR")
        pt.line("evm", t["x"], t["evm"], color=PURPLE, width=2.0, style="--", name="−EVM")
        pt.scatter("ad", t["x"], t["aclr"], color=NAVY, size=5)
        pt.scatter("ed", t["x"], t["evm"], color=PURPLE, size=5)

    def background(self, p):
        key = self.trade_key
        if key in self.trade:
            return
        y = self.cfr(p)
        pe = self.pexp(p)
        out = dict(x=[], aclr=[], evm=[])
        for ibo in np.arange(0, 12.1, 1.0 if not self.quick else 3.0):
            z = self.pa(y[::2], ibo, pe)
            evm = oa.bussgang_evm_db(z, self.g[::2], self.cfg, self.g[::2])
            out["x"].append(ibo)
            out["aclr"].append(float(oa.aclr_symbols(z, self.cfg)))
            out["evm"].append(float(-evm))
            yield key, dict(out)
        if len(self.trade) > 30:
            self.trade.clear()
        self.trade[key] = out

    def progress(self, p, item):
        key, out = item
        if key == self.trade_key:
            self.trade[key] = out
            self.draw_trade(p)

    def story(self, p):
        r = self.r
        s = ("<p>An OFDM signal crosses 10 dB above its average now and then. A power amplifier "
             "cannot follow: near saturation it flattens the peaks itself, and a clipped peak is a "
             "spectral splash that lands in the neighbours' channel (red bands). Backing off "
             "avoids it, at the price of efficiency.</p>")
        if p.cfr:
            s += (f"<p><b>Crest-factor reduction</b> does the clipping deliberately, at "
                  f"{v(p.clip, '.2f', 'dB')} above RMS, then filters away the splatter it creates "
                  f"and repeats ({v(int(p.iters), 'd')} times). The peaks come down (green CCDF), "
                  f"the distortion stays <i>inside</i> the channel as an EVM of "
                  f"{v(self.evm_cfr, '.1f', 'dB')}, which the link budget can afford, rather than "
                  f"outside it, which the regulator cannot.</p>")
        else:
            s += "<p>No CFR: every peak goes to the amplifier unprocessed.</p>"
        s += (f"<p>Through the PA at {v(p.ibo, '.2f', 'dB')} back-off: EVM "
              f"{v(r.get('evm', 0), '.1f', 'dB')}, ACLR {v(r.get('aclr', 0), '.1f', 'dB')} "
              f"({good('meets 45 dB') if r.get('aclr', 0) >= 45 else bad('fails 45 dB')}), and an "
              f"ideal class-B stage would run at {v(r.get('eff', 0), '.0f', '%')} efficiency. "
              f"Every base station radio does this, usually with digital predistortion too.</p>")
        return "<h3>Clip on purpose, filter, then amplify</h3>" + s + keybox(
            "CFR trades in-band EVM, which the link can absorb, for out-of-band emission, which "
            "nobody will forgive, and so lets the PA run closer to saturation.")


# =============================================================================== 5. spectral containment
class SpectralContainment(Experiment):
    title = "WOLA and filtered OFDM"
    blurb = "Rectangular symbols have slowly decaying sidelobes. Soften the edges or filter them."
    book = "sec:ch17:oob"
    controls = [
        Slider("wola", "WOLA taper", 0.1, 4.6, 0.3, step=0.1, unit="µs",
               help="Raised-cosine ramp at each symbol edge, overlapping the neighbour's prefix"),
        Slider("filt", "Filter length", 0.5, 66, 33, step=0.5, unit="µs",
               help="Windowed-sinc low-pass applied to the whole waveform"),
        Heading("Channel (for the in-band cost)"),
        Slider("echo", "Echo delay", 0.0, 4.6, 2.0, step=0.1, unit="µs",
               help="A second path 3 dB down. The prefix is 4.69 µs"),
    ]
    plots = [
        SpectrumPlot("psd", "Spectrum: 600 subcarriers (9 MHz) in a 10 MHz channel",
                     x="frequency (MHz)", y="PSD (dB rel. in-band)", xlim=(-14, 14),
                     ylim=(-110, 6), legend="bl"),
        Plot("win", "What each technique does at a symbol boundary", x="time (µs)",
             y="weight", xlim=(-11, 7), ylim=(-0.35, 1.7), legend="tr", legend_cols=2),
        BarPlot("evm", "In-band cost: EVM with the echo", y="EVM (dB)"),
    ]
    layout = [["psd", "psd"], ["win", "evm"]]
    row_stretch = [3, 2]
    col_stretch = [3, 2]
    readouts = [
        Readout("a0", "ACLR, plain CP-OFDM", "dB", ".1f"),
        Readout("a1", "ACLR, WOLA", "dB", ".1f", good=lambda x: x >= 45),
        Readout("a2", "ACLR, filtered", "dB", ".1f", good=lambda x: x >= 45),
        Readout("ew", "EVM, WOLA", "dB", lambda x: "< −60" if x <= -60 else f"{x:.1f}".replace("-", "−")),
    ]
    challenges = [
        Challenge("Reach 60 dB of ACLR with a WOLA taper of 0.8 µs or less.",
                  lambda s: s.r.a1 >= 60 and s.p.wola <= 0.8),
        Challenge("Filtered OFDM: how short can the filter be? Keep 70 dB of ACLR with a filter of "
                  "1 µs or less.",
                  lambda s: s.r.a2 >= 70 and s.p.filt <= 1.0),
        Challenge("Eat the prefix: with a 3 µs echo, make the WOLA taper long enough that its EVM "
                  "is worse than −30 dB.",
                  lambda s: s.p.echo >= 3 and s.r.ew > -30,
                  hint="The taper overlaps the next symbol's prefix; the echo needs that prefix too."),
    ]
    Lx, ncp, nsym = 2, 72, 44

    def setup(self):
        rng = np.random.default_rng(13)
        self.fs = FS * self.Lx
        self.N = NFFT * self.Lx
        self.Ncp = self.ncp * self.Lx
        self.g = qam(Q16, NUSED * self.nsym, rng).reshape(self.nsym, NUSED)
        self.cfg = co.OFDMConfig(self.N, NUSED, self.Ncp)
        self.x_cp = co.ofdm_modulate(self.g, self.cfg)
        self.a0 = self.aclr(self.x_cp)
        self.f0, self.p0 = self.spec(self.x_cp)
        self._fk = None

    def spec(self, x):
        f, pw = oa.psd(x, self.fs, 4096)
        return f / 1e6, db(pw / np.median(pw[np.abs(f) < 3e6]))

    def aclr(self, x):
        return float(min(oa.aclr_db(x, self.fs, 9e6, 10e6, 9e6, 4096), 100.0))

    def evm(self, x, echo_us, tx_delay=0):
        d = int(round(echo_us * 1e-6 * self.fs))
        h = np.zeros(d + 1, complex)
        h[0] = 1
        h[d] += 10 ** (-3 / 20)
        y = np.convolve(x, h)[tx_delay:tx_delay + self.nsym * self.cfg.sym_len]
        Y = co.ofdm_demodulate(y, self.cfg)
        H = np.fft.fft(h, self.N)[self.cfg.active]
        Z = Y[2:-2] / H
        return max(float(db(np.mean(np.abs(Z - self.g[2:-2]) ** 2) / np.mean(np.abs(self.g) ** 2))),
                   -60.0)

    def update(self, p):
        W = max(2, int(round(p.wola * 1e-6 * self.fs)))
        x_w = oa.wola_ofdm(self.g, self.N, self.Ncp, W)
        ntap = int(round(p.filt * 1e-6 * self.fs)) | 1
        fk = ntap
        if fk != self._fk:
            self._xf = oa.filtered_ofdm(self.x_cp, self.fs, (NUSED + 4) * 15e3, ntap)
            self._fk = fk
        x_f = self._xf
        a1, a2 = self.aclr(x_w), self.aclr(x_f)
        pp = self.plot("psd")
        pp.band("occ", -4.5, 4.5, color=GREEN, alpha=0.07)
        pp.band("adjr", 5.5, 14.5, color=RED, alpha=0.05)
        pp.band("adjl", -14.5, -5.5, color=RED, alpha=0.05)
        pp.vline("e1", 5, color=GRAY, style=":")
        pp.vline("e2", -5, color=GRAY, style=":")
        pp.line("cp", self.f0, self.p0, color=RED, width=1.1, name=f"CP-OFDM ({self.a0:.0f} dB)")
        f1, p1 = self.spec(x_w)
        pp.line("w", f1, p1, color=PURPLE, width=1.3, name=f"WOLA {p.wola:.1f} µs ({a1:.0f} dB)")
        f2, p2 = self.spec(x_f)
        pp.line("f", f2, p2, color=NAVY, width=1.3, name=f"filtered {p.filt:.0f} µs ({a2:.0f} dB)")
        pp.text("adj", 6.0, 1, "adjacent channel", color=RED, size=8.5, anchor=(0, 0))
        # time view: window at a boundary and the filter's impulse response
        pw_ = self.plot("win")
        tcp = self.Ncp / self.fs * 1e6
        Tw = p.wola
        t = np.linspace(-12, 12, 1201)
        ramp_up = np.clip((t + tcp) / Tw, 0, 1)
        up = 0.5 * (1 - np.cos(np.pi * ramp_up))
        down = 0.5 * (1 + np.cos(np.pi * np.clip((t - (-tcp + Tw) + Tw) / Tw, 0, 1)))
        pw_.band("cp", -tcp, 0, color=GREEN, alpha=0.10)
        pw_.text("cpl", -tcp + 0.1, -0.3, "prefix of symbol 2", color=GREEN, size=8.5, anchor=(0, 1))
        pw_.line("dn", t, down, color=PURPLE, width=2.0, style="--", name="WOLA: symbol 1 fades out")
        pw_.line("up", t, up, color=PURPLE, width=2.0, name="WOLA: symbol 2 fades in")
        n = np.arange(ntap) - ntap // 2
        bw = (NUSED + 4) * 15e3 / 2
        hf = 2 * bw / self.fs * np.sinc(2 * bw / self.fs * n) * np.hanning(ntap) ** 0.6
        pw_.line("h", n / self.fs * 1e6 - tcp, hf / hf.max(), color=NAVY, width=1.4,
                 name="filter's impulse response")
        pw_.vline("echo", -tcp + p.echo, color=RED, style=":", label="echo", label_pos=0.6)
        # in-band cost
        e0 = self.evm(self.x_cp, p.echo)
        ew = self.evm(x_w, p.echo)
        ef = self.evm(x_f, p.echo)
        pb = self.plot("evm")
        vals = [e0, ew, ef]
        pb.bars("b", [0, 1, 2], vals, width=0.6, base=-62, colors=[RED, PURPLE, NAVY])
        for i, x_ in enumerate(vals):
            pb.text(f"t{i}", i, x_, "< −60" if x_ <= -60 else f"{x_:.0f}", anchor=(0.5, 1.1),
                    bold=True, size=9)
        pb.set_xticks([(0, "CP-OFDM"), (1, "WOLA"), (2, "filtered")])
        pb.set_xlim(-0.6, 2.6)
        pb.set_ylim(-62, 0)
        self.readout(a0=self.a0, a1=a1, a2=a2, ew=ew)
        self.ef = ef

    def story(self, p):
        r = self.r
        s = ("<p>Every OFDM symbol is a burst of sinusoids switched on and off abruptly, so its "
             "spectrum is a sum of sincs whose sidelobes fall only as 1/f: plain CP-OFDM leaks "
             f"into the next channel at {v(r.get('a0', 0), '.0f', 'dB')} (red curve), far short of "
             "what a base station must meet.</p>")
        s += (f"<p><b>WOLA</b> softens the switch: each symbol fades in and out over "
              f"{v(p.wola, '.1f', 'µs')}, overlapping its neighbour inside the prefix (bottom left). "
              f"ACLR: {v(r.get('a1', 0), '.0f', 'dB')}. <b>Filtered OFDM</b> passes the whole "
              f"waveform through a sharp {v(p.filt, '.0f', 'µs')} low-pass: "
              f"{v(r.get('a2', 0), '.0f', 'dB')}.</p>")
        s += (f"<p>Neither is free. Both spend part of the prefix: the taper and the filter's "
              f"transient both smear across the symbol boundary, and so does the echo "
              f"({v(p.echo, '.1f', 'µs')}). When they add up to more than the 4.69 µs prefix, "
              f"interference appears in-band (bars). 5G NR leaves the choice to the "
              f"implementer: the standard specifies the emission mask, not the method.</p>")
        return "<h3>Taming the sidelobes</h3>" + s + keybox(
            "Soft symbol edges (WOLA) or a filter (f-OFDM) buy tens of dB of spectral containment "
            "for a slice of the cyclic prefix.")


# =============================================================================== 6. DSL bit loading
TONES = np.arange(33, 512)
FDMT = TONES * 4312.5


def bridged_tap_db(f, length_m, att_db_per_km_at_1mhz=23.0):
    """Insertion loss (dB, negative) of an open-ended bridged tap: H = 1 / (1 + tanh(γl)/2),
    with the cable's attenuation and a velocity factor of 0.67."""
    alpha = att_db_per_km_at_1mhz * np.sqrt(f / 1e6) / 8.686 / 1e3       # Np/m
    beta = 2 * np.pi * f / (0.67 * C0)
    H = 1 / (1 + np.tanh((alpha + 1j * beta) * length_m) / 2)
    return 20 * np.log10(np.abs(H))


class BitLoading(Experiment):
    title = "DMT on copper: bit loading"
    blurb = "On a wire that never changes, give every tone exactly the bits its SNR can carry."
    book = "sec:ch17:coded"
    controls = [
        Heading("The line"),
        Slider("loop", "Loop length", 0.3, 6.0, 3.0, step=0.1, unit="km"),
        Toggle("bt", "Bridged tap", False,
               help="An unused, open-ended branch of cable left connected to the line"),
        Slider("btl", "Tap length", 30, 500, 150, step=10, unit="m", enabled_if=lambda p: p.bt),
        Heading("Noise"),
        Slider("rfi", "AM radio ingress", 0, 50, 30, step=1, unit="dB"),
        Slider("rfif", "Ingress frequency", 0.3, 2.1, 1.35, step=0.01, unit="MHz"),
        Heading("Modem"),
        Slider("gap", "SNR gap Γ", 6, 18, 11.8, step=0.2, unit="dB",
               help="9.8 dB (uncoded, 10⁻⁷) + 6 dB margin − 4 dB coding gain"),
        Choice("alloc", "Power", ["Flat PSD (mask)", "Water-filling"]),
    ]
    plots = [
        Plot("snr", "SNR per tone", x="frequency (MHz)", y="SNR (dB)", xlim=(0, 2.25),
             ylim=(-10, 75), legend="tr"),
        Plot("bits", "Bits on each tone", x="frequency (MHz)", y="bits", xlim=(0, 2.25),
             ylim=(0, 17), legend="tr"),
        Plot("water", "Water-filling: Γ/gₖ and the water level", x="frequency (MHz)",
             y="Γ/gₖ (log)", logy=True, xlim=(0, 2.25), ylim=(1e-7, 1e2), legend="tl"),
        Plot("reach", "Rate versus reach", x="loop length (km)", y="line rate (Mb/s)",
             xlim=(0, 6.2), ylim=(0, 32), legend="tr"),
    ]
    layout = [["snr", "bits"], ["water", "reach"]]
    readouts = [
        Readout("rate", "Line rate", "Mb/s", ".2f"),
        Readout("tones", "Tones in use", "", "int"),
        Readout("cap", "Capacity with the gap", "Mb/s", ".2f"),
        Readout("wf", "Water-filling gain", "%", ".1f"),
    ]
    challenges = [
        Challenge("Deliver 10 Mb/s on the longest loop you can (2.3 km or more), default gap.",
                  lambda s: s.p.loop >= 2.3 and s.r.rate >= 10 and abs(s.p.gap - 11.8) < 0.05),
        Challenge("Add a bridged tap whose first notch falls at 1 MHz (±5 %), and read off what it "
                  "costs.",
                  lambda s: s.p.bt and 0.95e6 <= 0.67 * C0 / (4 * s.p.btl) <= 1.05e6,
                  hint="A tap of length l notches the line where it is a quarter wavelength long: "
                       "f = v/(4l), with v ≈ 0.67c in the cable."),
        Challenge("Find a loop where water-filling adds more than 30 % over a flat PSD.",
                  lambda s: s.r.wf > 30,
                  hint="Water-filling helps most when only a few tones are usable."),
    ]

    def snr(self, p, L=None, bt=None):
        L = p.loop if L is None else L
        s = oa.dsl_snr_db(FDMT, L, rfi_db=p.rfi, rfi_hz=p.rfif * 1e6)
        if (p.bt if bt is None else bt):
            s = s + bridged_tap_db(FDMT, p.btl)
        return s

    def load(self, snr_db, gap, alloc):
        s = 10 ** ((snr_db - gap) / 10)
        pw, mu = oa.waterfill_bisect(1 / s, len(TONES) * 1.0)
        if alloc == "Water-filling":
            b = np.clip(np.floor(np.log2(1 + pw * s)), 0, 15)
            b[b < 2] = 0
        else:
            b = oa.dmt_bit_loading(snr_db, gap)
        return b.astype(int), s, pw, mu

    def update(self, p):
        sdb = self.snr(p)
        b, s, pw, mu = self.load(sdb, p.gap, p.alloc)
        bflat, *_ = self.load(sdb, p.gap, "Flat PSD (mask)")
        bwf, *_ = self.load(sdb, p.gap, "Water-filling")
        rate = 4000 * b.sum() / 1e6
        cap = 4000 * np.sum(np.log2(1 + s)) / 1e6
        fm = FDMT / 1e6
        ps = self.plot("snr")
        ps.line("s", fm, sdb, color=NAVY, width=1.8, name="SNR per tone")
        ps.hline("gap", p.gap, color=RED, style="--", label="gap Γ", label_pos=0.02)
        pb = self.plot("bits")
        pb.line("b", np.r_[fm, fm[-1] + 0.0043], b, color=NAVY, width=1.6, step=True, fill=0,
                fill_alpha=0.18, name=f"{p.alloc.split(' (')[0]}: {rate:.2f} Mb/s")
        if p.bt:
            bnb, *_ = self.load(self.snr(p, bt=False), p.gap, p.alloc)
            pb.line("nb", np.r_[fm, fm[-1] + 0.0043], bnb, color=GRAY, width=1.0, step=True,
                    name="without the tap")
        pw_ = self.plot("water")
        inv = 1 / s
        pw_.line("inv", fm, np.minimum(inv, 1e2), color=NAVY, width=1.5, name="Γ/gₖ (the vessel)")
        pw_.fill_between("wat", fm, np.minimum(inv, mu), np.full_like(fm, mu), color=BLUE, alpha=0.4)
        pw_.hline("mu", mu, color=BLUE, style="--", label="water level", label_pos=0.75)
        # rate vs reach
        Ls = np.linspace(0.3, 6.0, 40)
        rr = [4000 * self.load(self.snr(p, L=L_), p.gap, p.alloc)[0].sum() / 1e6 for L_ in Ls]
        pr = self.plot("reach")
        pr.line("rr", Ls, rr, color=NAVY, width=2.2, name=p.alloc.split(" (")[0])
        for L_, ref in ((1.0, 26.7), (3.0, 5.6), (5.0, 1.0)):
            pr.scatter(f"bk{L_}", [L_], [ref], color=GRAY, size=9, outline=GRAY,
                       name="Chapter 17 example" if L_ == 1.0 else None)
        pr.scatter("now", [p.loop], [rate], color=RED, size=13, symbol="d", name="your loop")
        self.rate_nobt = 4000 * self.load(self.snr(p, bt=False), p.gap, p.alloc)[0].sum() / 1e6
        rf, rw = 4000 * bflat.sum() / 1e6, 4000 * bwf.sum() / 1e6
        self.readout(rate=rate, tones=int(np.count_nonzero(b)), cap=cap,
                     wf=100 * (rw / rf - 1) if rf > 0 else 0.0)

    def story(self, p):
        r = self.r
        s = (f"<p>ADSL2+ is OFDM on a telephone pair (called DMT): 479 downstream tones 4.3125 kHz "
             f"apart, 4000 symbols a second. The copper loses more at high frequencies the longer "
             f"the loop, so on {v(p.loop, '.1f', 'km')} the SNR (top left) slides down until it "
             f"meets the gap Γ. Because the line hardly changes, the modem measures it once and "
             f"<b>loads</b> each tone with ⌊log₂(1 + SNR/Γ)⌋ bits, up to 15.</p>"
             f"<p>Total: {v(r.get('rate', 0), '.2f', 'Mb/s')} on {v(r.get('tones', 0), 'd')} tones.")
        if p.rfi > 5:
            s += (f" The notch at {v(p.rfif, '.2f', 'MHz')} is an AM broadcaster leaking into the "
                  f"wire: the loading simply steps around it.")
        s += "</p>"
        if p.bt:
            s += (f"<p>The bridged tap, an open branch of {v(p.btl, '.0f', 'm')}, reflects the "
                  f"signal back a quarter-wavelength out of phase: notches at about "
                  f"{v(0.67 * C0 / (4 * p.btl) / 1e6, '.2f', 'MHz')} and odd multiples. Cost: "
                  f"{v(self.rate_nobt - r.get('rate', 0), '.2f', 'Mb/s')}.</p>")
        s += (f"<p>Water-filling would pour power into the tones where it buys most bits "
              f"({v(r.get('wf', 0), '.1f', '%')} more here), but DSL must respect a fixed PSD mask, "
              f"so in practice only the bits adapt. The steep rate-versus-reach curve is why "
              f"operators pushed fibre ever closer to the home.</p>")
        return "<h3>Bits where the SNR is</h3>" + s + keybox(
            "bₖ = ⌊log₂(1 + SNRₖ/Γ)⌋: adapt every tone to the line, once, and run at the capacity "
            "minus a gap.")


# =============================================================================== 7. OFDM radar
class OFDMRadar(Experiment):
    title = "OFDM as a radar"
    blurb = "Divide out the data and two FFTs turn the echo into a range–Doppler map."
    book = "sec:ch17:systems"
    controls = [
        Heading("Targets"),
        Slider("r1", "Target 1 range", 5, 190, 40, step=0.5, unit="m"),
        Slider("v1", "Target 1 velocity", -150, 150, 12, step=1, unit="m/s"),
        Slider("r2", "Target 2 range", 5, 190, 95, step=0.5, unit="m"),
        Slider("v2", "Target 2 velocity", -150, 150, -25, step=1, unit="m/s"),
        Slider("a2", "Target 2 echo strength", -30, 0, -6, step=1, unit="dB"),
        Heading("Waveform at 28 GHz"),
        Choice("scs", "Subcarrier spacing", ["15 kHz", "30 kHz", "60 kHz", "120 kHz", "240 kHz"],
               "120 kHz", help="512 subcarriers × 128 symbols. Bandwidth = 512 × spacing; "
                               "symbol time ≈ 1.07 / spacing"),
        Slider("snr", "SNR per resource element", -40, 20, -10, step=1, unit="dB"),
        Toggle("win", "Hann window (lower sidelobes)", True),
    ]
    plots = [
        ImagePlot("map", "Range–Doppler map", x="velocity (m/s)", y="range (m)"),
        Plot("rcut", "Range cut", x="range (m)", y="power (dB)",
             xlim=(0, 200), ylim=(-60, 8), legend=None),
        Plot("vcut", "Doppler cut", x="velocity (m/s)",
             y="power (dB)", ylim=(-60, 8), legend=None),
    ]
    layout = [["map", "rcut"], ["map", "vcut"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("dr", "Range resolution", "m", ".2f"),
        Readout("dv", "Velocity resolution", "m/s", ".2f"),
        Readout("vmax", "Unambiguous velocity", "± m/s", ".0f"),
        Readout("found", "Targets found", "", None),
    ]
    challenges = [
        Challenge("Resolve two targets only 3 m apart in range at the same speed.",
                  lambda s: (abs(s.p.r1 - s.p.r2) <= 3 and abs(s.p.v1 - s.p.v2) < 0.5
                             and s.exp.n_found == 2),
                  hint="Range resolution is c/(2·bandwidth): widen the subcarriers."),
        Challenge("Detect target 1 at a per-element SNR of −25 dB or lower.",
                  lambda s: s.p.snr <= -25 and s.exp.found1,
                  hint="The map integrates 512 × 128 elements: 48 dB of processing gain."),
        Challenge("Make target 2 fast enough to alias: it appears at the wrong velocity.",
                  lambda s: abs(s.p.v2) > s.r.vmax and s.p.a2 > -20,
                  hint="Lower the subcarrier spacing: longer symbols sample the Doppler more slowly."),
    ]
    M, Ns, fc = 512, 128, 28e9

    def setup(self):
        rng = np.random.default_rng(30)
        self.X = QPSK.points[rng.integers(0, 4, (self.M, self.Ns))]
        self.seed = 3

    def update(self, p):
        df = float(p.scs.split()[0]) * 1e3
        Tsym = 1 / df * (1 + 144 / 2048)
        dR = C0 / (2 * self.M * df)
        dv = C0 / (2 * self.fc * self.Ns * Tsym)
        vmax = self.Ns / 2 * dv
        rng = np.random.default_rng(self.seed)
        tg = [(p.r1, p.v1, 1.0), (p.r2, p.v2, 10 ** (p.a2 / 20))]
        Y = oa.radar_echo(self.X, tg, df, Tsym, self.fc, n0=10 ** (-p.snr / 10), rng=rng)
        P = oa.radar_map(Y, self.X, window=p.win)
        P = P / P.max()
        rr = np.arange(self.M) * dR
        vv = (np.arange(self.Ns) - self.Ns // 2) * dv
        sel = rr <= 200 + dR
        Pd = db(P[sel])
        pm = self.plot("map")
        pm.image("img", Pd, x=(vv[0] - dv / 2, vv[-1] + dv / 2),
                 y=(rr[0] - dR / 2, rr[sel][-1] + dR / 2), cmap="heat", levels=(-45, 0),
                 colorbar=True, cbar_label="dB")
        th = np.linspace(0, 2 * np.pi, 41)
        for k, (vv_, r_) in enumerate(((p.v1, p.r1), (p.v2, p.r2))):
            pm.line(f"ring{k}", vv_ + 9 * np.cos(th), r_ + 7 * np.sin(th), color=GREEN, width=1.6,
                    name="true position" if k == 0 else None)
        pm.set_xlim(max(-vmax, -160), min(vmax, 160))
        pm.set_ylim(0, 200)
        # detection: local maxima 13 dB above the median noise floor
        floor = np.median(P)
        loc = (P == maximum_filter(P, size=5)) & (P > 20 * floor) & (P > 10 ** (-3.5))
        peaks = np.argwhere(loc)
        det = [(rr[i], vv[j]) for i, j in peaks]

        def near(r, v_):
            vw = (v_ + vmax) % (2 * vmax) - vmax
            return any(abs(a - r) <= 1.5 * dR + 0.5 and abs(b - vw) <= 1.5 * dv + 0.5 for a, b in det)
        self.found1 = near(p.r1, p.v1)
        found2 = near(p.r2, p.v2)
        aliased = abs(p.v2) > vmax
        self.n_found = int(self.found1) + int(found2 and not aliased)
        # cuts through the strongest cell (titles name the cell; dotted lines mark the targets)
        i, j = np.unravel_index(np.argmax(P), P.shape)
        pr = self.plot("rcut")
        pr.set_title(f"Range cut at v = {vv[j]:+.1f} m/s")
        pr.line("r", rr[sel], db(P[sel, j]), color=NAVY, width=1.6)
        for k, (r_, c_) in enumerate(((p.r1, GREEN), (p.r2, ORANGE))):
            pr.vline(f"t{k}", r_, color=c_, style=":", width=1.2, label=f"T{k + 1}",
                     label_pos=0.96 - 0.1 * k)
        pv = self.plot("vcut")
        pv.set_title(f"Doppler cut at R = {rr[i]:.1f} m")
        pv.line("v", vv, db(P[i]), color=NAVY, width=1.6)
        pv.set_xlim(max(-vmax, -160), min(vmax, 160))
        for k, (v_, c_) in enumerate(((p.v1, GREEN), (p.v2, ORANGE))):
            if abs(v_) <= vmax:
                pv.vline(f"t{k}", v_, color=c_, style=":", width=1.2, label=f"T{k + 1}",
                         label_pos=0.96 - 0.1 * k)
        n_det = len(det)
        self.readout(dr=dR, dv=dv, vmax=vmax,
                     found=f"{self.n_found} of 2" + (" (+ alias)" if aliased and found2 is False
                                                      and n_det > self.n_found else ""))

    def story(self, p):
        df = float(p.scs.split()[0])
        r = self.r
        s = ("<p>The transmitter knows every symbol it sent, so it can divide the echo by the data. "
             "What is left, for each target, is a phase ramp across subcarriers (from its delay, "
             "i.e. range) and a phase ramp across symbols (from its Doppler, i.e. velocity). An "
             "inverse FFT along frequency and an FFT along time turn both ramps into a peak.</p>")
        s += (f"<p>With {v(df, '.0f', 'kHz')} spacing the 512 subcarriers span "
              f"{v(512 * df / 1e3, '.1f', 'MHz')}: range cells of {v(r.get('dr', 0), '.2f', 'm')} "
              f"(c/2B). The 128 symbols last {v(128 / df * 1.07, '.2f', 'ms')}: velocity cells of "
              f"{v(r.get('dv', 0), '.2f', 'm/s')}, up to ±{v(r.get('vmax', 0), '.0f', 'm/s')} before "
              f"the Doppler wraps around. Bandwidth buys range resolution; dwell time buys velocity "
              f"resolution: the same trade-off as the numerology.</p>")
        s += ("<p>The two cuts pass through the strongest cell of the map; their titles give that "
              "cell's velocity and range, which sit on the grid, so they differ from the slider "
              "values by up to half a cell. The dotted lines mark the true targets: "
              "<b>T1</b> green, <b>T2</b> orange.</p>")
        if p.snr < -15:
            s += (f"<p>At {v(p.snr, '.0f', 'dB')} per element the echo is invisible in any single "
                  f"subcarrier, yet the map integrates 65 536 of them: about 48 dB of processing "
                  f"gain.</p>")
        s += ("<p>This is integrated sensing and communication (ISAC), studied by 3GPP from "
              "Release 19: the base station's own downlink doubles as the radar.</p>")
        return "<h3>A radar for free</h3>" + s + keybox(
            "ΔR = c/(2·N·Δf),  Δv = λ/(2·N_sym·T_sym): bandwidth sets range resolution, dwell time "
            "sets velocity resolution.")


# =============================================================================== the lab
LAB = st.Lab(28, "OFDM as a System", chapter=17, chapter_title="OFDM and Multicarrier",
             experiments=[EstimatorShootout, ICIImpairments, CodedOFDM, ClipAndPA,
                          SpectralContainment, BitLoading, OFDMRadar])

if __name__ == "__main__":
    st.run(LAB)
