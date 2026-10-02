"""Lab 12 · Frontiers: Delay–Doppler, ISAC, RIS and Learned Receivers   (Chapter 25)

Run it:      python labs/lab12_frontiers.py
Self-test:   python labs/lab12_frontiers.py --selftest

Six ideas that dominate 6G research, each small enough to understand completely and each with an
honest note about what it leaves out. Watch a doubly dispersive channel turn into a few still
points in the delay–Doppler plane, race OTFS against OFDM on a fast train, see why phase noise
forces wider subcarriers at high frequencies, use an OFDM frame as a radar, bend a wave with a
passive surface, and train two neural networks live: a demapper that learns a distorted
constellation, and an autoencoder that invents its own.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import sixg, rf
from commlib import frontier as fr
from commlib import modzoo as mz
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ConstellationPlot, BERPlot, BarPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

C0 = sixg.C0
QPSK = cl.get_constellation("qpsk")
Q16 = cl.get_constellation("16qam")
SCS = {"15 kHz": 15e3, "30 kHz": 30e3, "60 kHz": 60e3, "120 kHz": 120e3}


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def cgauss(rng, shape):
    """Unit-power circular complex Gaussian samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def region_stops(theme_name):
    """Six soft colours for decision-region maps (tuples, as studio expects)."""
    if theme_name == "dark":
        return ((0, "#1E2B3A"), (0.2, "#3A2624"), (0.4, "#1F352A"), (0.6, "#3B2E20"),
                (0.8, "#2F2539"), (1.0, "#2B3036"))
    return ((0, "#DCE6F0"), (0.2, "#F3DCD9"), (0.4, "#DCEFE4"), (0.6, "#F6E6D6"),
            (0.8, "#E6DCEF"), (1.0, "#E4E4E4"))


def colour_regions(idx, n_classes):
    """Greedy graph colouring of a decision-region index map with 6 colours, so that touching
    regions differ (returns a map of colour numbers 0..5)."""
    adj = {i: set() for i in range(n_classes)}
    for a_, b_ in ((idx[:, 1:], idx[:, :-1]), (idx[1:, :], idx[:-1, :])):
        m = a_ != b_
        for u, w in set(zip(a_[m].tolist(), b_[m].tolist())):
            adj[u].add(w)
            adj[w].add(u)
    col = {}
    for u in sorted(adj, key=lambda q: -len(adj[q])):
        used = {col[w] for w in adj[u] if w in col}
        col[u] = next((k for k in range(6) if k not in used), 0)
    lut = np.array([col.get(i, 0) for i in range(n_classes)])
    return lut[idx]


def rayleigh_qpsk_ber(snr_db):
    """Uncoded QPSK on a single Rayleigh path: the 'no diversity' reference."""
    g = 10 ** (np.asarray(snr_db, float) / 10) / 2
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


# =============================================================================== 1. DD channel
class DelayDopplerView(Experiment):
    title = "The delay–Doppler view"
    blurb = "A fast channel that churns in time and frequency is four still points in delay–Doppler."
    book = "sec:ch25:waveforms"
    animate = True
    autoplay = True
    fps = 15
    controls = [
        Heading("The mobile"),
        Slider("speed", "Speed", 0, 1000, 500, step=10, unit="km/h",
               help="500 km/h: a high-speed train; 1000 km/h: an aircraft"),
        LogSlider("fc", "Carrier frequency", 0.7, 40, 4.0, unit="GHz", fmt=".2f"),
        Heading("The OFDM / OTFS grid (64 × 32)"),
        Choice("scs", "Subcarrier spacing", list(SCS), "30 kHz"),
        Toggle("frac", "Fractional Doppler", True,
               help="Off: every path's Doppler is rounded to a whole Doppler bin of the grid"),
    ]
    plots = [
        ImagePlot("tf", "What OFDM sees: |H(f, t)| over one frame", x="OFDM symbol (time)",
                  y="subcarrier (frequency)"),
        ImagePlot("dd", "What OTFS sees: |h(τ, ν)|, the response to one DD symbol",
                  x="Doppler bin", y="delay bin"),
        Plot("hist", "One subcarrier over time, against the four delay–Doppler taps",
             x="time (ms)", y="gain (dB)", ylim=(-32, 12), legend="tl", legend_cols=2),
    ]
    layout = [["tf", "dd"], ["hist", "hist"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("nu", "Maximum Doppler", "Hz", ",.0f"),
        Readout("nuT", "Doppler / subcarrier spacing", "", ".3f"),
        Readout("sir", "OFDM signal-to-ICI", "dB", ".1f", good=lambda x: x >= 25),
        Readout("leak", "DD energy outside the path cells", "%", ".0f"),
    ]
    challenges = [
        Challenge("Size the grid for the 500 km/h train at 4 GHz or more: pick a subcarrier spacing "
                  "that gives OFDM at least 25 dB of signal-to-ICI.",
                  lambda s: s.p.speed >= 499 and s.p.fc >= 3.99 and s.exp.sir_num >= 25,
                  hint="The ICI bound is (πν_max T)²/3: halve T (double Δf) and gain 6 dB."),
        Challenge("Overflow the grid: at 28 GHz push the fastest path beyond the ±16 Doppler bins "
                  "this 32-symbol frame can represent.",
                  lambda s: s.p.fc >= 27 and s.exp.kmax > 16,
                  hint="Doppler bins are Δf/N wide: small subcarrier spacing, high speed."),
        Challenge("With fractional Doppler on, find a speed where at least 45 % of the DD energy "
                  "leaks out of the four path cells.",
                  lambda s: s.p.frac and s.r.leak >= 45,
                  hint="Leakage is worst when a path sits halfway between two Doppler bins."),
    ]
    M, N = 64, 32
    DELAYS = np.array([0, 3, 7, 12])
    GAINS = np.array([1.0, 0.7, 0.5, 0.35]) * np.exp(1j * np.array([0.0, 1.3, 2.4, 4.0]))
    COSA = np.array([1.0, 0.45, -0.8, -0.27])            # cos(arrival angle) of each path

    def setup(self):
        self.t0 = 0.0
        X = np.zeros((self.M, self.N), complex)
        X[0, 0] = 1
        self.s = sixg.otfs_modulate(X)

    def paths(self, p):
        df = SCS[p.scs]
        numax = p.speed / 3.6 * p.fc * 1e9 / C0
        nu = numax * self.COSA                                    # Hz
        k = nu * self.N / df                                      # in Doppler bins
        if not p.frac:
            k = np.round(k)
            nu = k * df / self.N
        return df, numax, nu, k

    def update(self, p):
        df, numax, nu, k = self.paths(p)
        # delay-Doppler response: exact OTFS chain on the 64 x 32 grid (one prefix per frame)
        n = np.arange(self.M * self.N)
        fs = self.M * df
        r = np.zeros_like(self.s)
        for lp, vp, hp in zip(self.DELAYS, nu / fs, self.GAINS):
            r += hp * np.exp(2j * np.pi * vp * n) * np.roll(self.s, int(lp))
        Y = np.abs(sixg.otfs_demodulate(r, self.M, self.N))
        E = Y ** 2
        cells = np.zeros_like(E, bool)
        for lp, kp in zip(self.DELAYS, k):
            cells[int(lp), int(np.round(kp)) % self.N] = True
        self.leak = 100 * (1 - E[cells].sum() / E.sum())
        self.Ydd = np.fft.fftshift(Y, axes=1)[:16]
        self.kmax = float(np.max(np.abs(k)))
        self.numax = numax
        nuT = numax / df
        self.sir_num = -10 * np.log10((np.pi * nuT) ** 2 / 3) if nuT > 1e-6 else np.inf
        self.draw(p)

    def tick(self, p):
        df = SCS[p.scs]
        self.t0 += 0.25 * self.N / df                            # a quarter of a frame per frame
        self.draw(p)

    def draw(self, p):
        df, numax, nu, k = self.paths(p)
        T = 1 / df
        kk = np.arange(self.M)[:, None]
        tn = self.t0 + np.arange(self.N)[None, :] * T
        H = sum(hp * np.exp(2j * np.pi * vp * tn) * np.exp(-2j * np.pi * kk * lp / self.M)
                for lp, vp, hp in zip(self.DELAYS, nu, self.GAINS))
        pt = self.plot("tf")
        pt.image("img", np.abs(H), x=(-0.5, self.N - 0.5), y=(-0.5, self.M - 0.5), cmap="heat",
                 levels=(0, 2.2), colorbar=True, cbar_label="|H|")
        pt.set_xlim(-0.5, self.N - 0.5)
        pt.set_ylim(-0.5, self.M - 0.5)
        pt.hline("k0", 10, color=RED, style="--", width=1.2)
        pd = self.plot("dd")
        pd.image("img", self.Ydd, x=(-self.N / 2 - 0.5, self.N / 2 - 0.5), y=(-0.5, 15.5),
                 cmap="heat", levels=(0, 1.0), colorbar=True, cbar_label="|h|")
        kw = (np.asarray(k) + self.N / 2) % self.N - self.N / 2
        pd.scatter("true", kw, self.DELAYS, color=RED, size=14, symbol="o", outline=RED)
        kl = min(self.N / 2, max(6.0, self.kmax + 2.5))
        pd.set_xlim(-kl - 0.5, kl - 0.5)
        pd.set_ylim(-0.5, 14.5)
        # one subcarrier over the last four frames, against the constant DD taps
        tt = self.t0 + np.linspace(-3 * self.N * T, self.N * T, 600)
        h0 = sum(hp * np.exp(2j * np.pi * vp * tt) * np.exp(-2j * np.pi * 10 * lp / self.M)
                 for lp, vp, hp in zip(self.DELAYS, nu, self.GAINS))
        ph = self.plot("hist")
        x = (tt - self.t0) * 1e3
        ph.line("h0", x, 20 * np.log10(np.abs(h0) + 1e-6), color=NAVY, width=2.0,
                name="|H| on subcarrier 10 (OFDM)")
        for i, (hp, col) in enumerate(zip(self.GAINS, (RED, ORANGE, GREEN, PURPLE))):
            ph.line(f"tap{i}", [x[0], x[-1]], [20 * np.log10(abs(hp))] * 2, color=col, width=1.6,
                    style="--", name="DD taps (OTFS)" if i == 0 else None)
        ph.band("frame", 0, self.N * T * 1e3, color=GREEN, alpha=0.08)
        ph.set_xlim(x[0], x[-1])
        nuT = numax / df
        self.readout(nu=numax, nuT=nuT,
                     sir=self.sir_num if np.isfinite(self.sir_num) else "∞",
                     leak=self.leak)

    def story(self, p):
        df = SCS[p.scs]
        nuT = self.numax / df
        s = (f"<p>At {v(p.speed, '.0f', 'km/h')} on {v(p.fc, '.2f', 'GHz')} the fastest path is "
             f"Doppler-shifted by {v(self.numax, ',.0f', 'Hz')}: {v(nuT, '.3f')} of a "
             f"{v(p.scs)} subcarrier spacing. Press <b>Play</b> and watch the two pictures on top: "
             f"the time–frequency response OFDM must track (left) churns from frame to frame, "
             f"while the delay–Doppler response (right) is four bright cells that never move. "
             f"Bottom: one subcarrier fades by tens of dB in milliseconds; the four DD taps are "
             f"flat lines.</p>")
        if self.kmax > 16:
            s += (f"<p>{bad('The grid is too coarse:')} the fastest path needs "
                  f"{v(self.kmax, '.1f')} Doppler bins but a 32-symbol frame resolves only ±16, so "
                  f"it wraps around (aliases). Use wider subcarriers or a lower carrier.</p>")
        elif p.frac and self.leak > 10:
            s += (f"<p>With <b>fractional Doppler</b> a path that falls between two Doppler bins "
                  f"smears along its delay row: {v(self.leak, '.0f', '%')} of the energy leaks out of "
                  f"the path cells. Turn the toggle off to snap the paths onto the grid.</p>")
        else:
            s += ("<p>Every path sits on a grid cell, so the DD channel is exactly sparse: one pilot "
                  "and a small guard region would reveal all of it at once (Lab 32).</p>")
        s += (f"<p>For OFDM the same Doppler is interference: the ICI bound (πν<sub>max</sub>T)²/3 "
              f"gives a signal-to-ICI ratio of "
              f"{v(self.sir_num if np.isfinite(self.sir_num) else 99, '.1f', 'dB')}"
              + (". " + good("Enough for 64-QAM.") if self.sir_num >= 25 else
                 ". " + bad("Marginal for 64-QAM.")) + "</p>")
        return "<h3>Two views of one channel</h3>" + s + keybox(
            "Delay and Doppler are set by geometry and change slowly; H(f, t), their 2-D Fourier "
            "transform, changes every coherence time. OTFS puts its data where the channel stands still.")


# =============================================================================== 2. OTFS vs OFDM
GRID = 16
A_OTFS = sixg.otfs_matrix(GRID, GRID)


def ofdm_frame(l, nu, h, x, n0, rng, ici_aware, cp=4):
    """16 CP-OFDM symbols of 16 subcarriers: one-tap or ICI-aware LMMSE (commlib.frontier)."""
    return fr.ofdm_dd_frame(GRID, GRID, cp, l, nu, h, x, n0, rng, ici_aware)


def otfs_frame(l, nu, h, x, n0, rng):
    MN = GRID * GRID
    He = A_OTFS.conj().T @ sixg.dd_channel_matrix(l, nu, h, MN) @ A_OTFS
    y = He @ x + np.sqrt(n0) * cgauss(rng, MN)
    return sixg.lmmse(He, y, n0)


class OTFSvsOFDM(Experiment):
    title = "OTFS versus OFDM on a fast train"
    blurb = "Same channel, same SNR, same 16 × 16 grid: one-tap OFDM against delay–Doppler OTFS."
    book = "sec:ch25:waveforms"
    animate = True
    fps = 8
    controls = [
        Heading("Channel"),
        Slider("speed", "Speed", 0, 1000, 300, step=10, unit="km/h"),
        Choice("fc", "Carrier", ["3.5 GHz", "28 GHz"], "28 GHz"),
        Choice("scs", "Subcarrier spacing", list(SCS), "30 kHz"),
        IntSlider("paths", "Number of paths", 1, 6, 4,
                  help="Rayleigh paths with delays up to 3 samples and Jakes-distributed Dopplers"),
        Slider("snr", "SNR", 0, 30, 20, step=1, unit="dB"),
        Heading("OFDM receiver"),
        Choice("eq", "OFDM equaliser", ["One-tap", "ICI-aware LMMSE"]),
    ]
    plots = [
        ConstellationPlot("ofdm", "OFDM after equalisation (red: wrong)", lim=1.5),
        ConstellationPlot("otfs", "OTFS after LMMSE detection (red: wrong)", lim=1.5),
        BERPlot("ber", "Bit error rate: curves (background) and your live count", x="SNR (dB)",
                xlim=(0, 30), ylim=(1e-5, 0.5), legend="bl"),
    ]
    layout = [["ofdm", "ber"], ["otfs", "ber"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("nuT", "Doppler / subcarrier spacing", "", ".2f"),
        Readout("ofdm", "OFDM BER (live)", "", "sci"),
        Readout("otfs", "OTFS BER (live)", "", "sci"),
        Readout("frames", "Frames counted", "", "int"),
    ]
    challenges = [
        Challenge("Press Play: at 25 dB or more, count 30 frames in which OTFS makes at least ten "
                  "times fewer errors than one-tap OFDM.",
                  lambda s: (s.r.frames >= 30 and s.p.snr >= 25 and s.p.eq == "One-tap"
                             and s.exp.err_ofdm >= 10 * (s.exp.err_otfs + 1)),
                  hint="Uncoded one-tap OFDM has no diversity; OTFS collects all the paths."),
        Challenge("Find OFDM's ICI floor: one-tap OFDM above 1 % BER at 30 dB SNR (20 frames).",
                  lambda s: (s.p.snr >= 30 and s.p.eq == "One-tap" and s.r.frames >= 20
                             and s.r.ofdm > 0.01),
                  hint="Raise the Doppler relative to the subcarrier spacing."),
        Challenge("Give OTFS nothing to harvest: with a single path, show OTFS no better than "
                  "ICI-aware OFDM (live BERs within a factor of 2, ≥ 20 frames, 15 dB).",
                  lambda s: (s.p.paths == 1 and s.p.eq == "ICI-aware LMMSE" and s.r.frames >= 20
                             and abs(s.p.snr - 15) < 0.5 and s.r.otfs > 0
                             and 0.5 <= s.r.otfs / max(s.r.ofdm, 1e-9) <= 2)),
    ]
    SNRS = np.arange(0, 31, 5.0)

    def setup(self):
        self.curves = {}
        self.curve_key = None
        self.curve_done = False
        self.err_ofdm = self.err_otfs = self.bits = 0
        self.frames = 0

    def numax(self, p):
        fc = 3.5e9 if p.fc.startswith("3.5") else 28e9
        nu_hz = p.speed / 3.6 * fc / C0
        return nu_hz / SCS[p.scs]                                 # in subcarrier spacings

    def make_frame(self, p, snr_db, rng):
        nT = self.numax(p)
        l, nu, h = sixg.random_dd_paths(p.paths, 3, nT / GRID, rng)
        b = cl.random_bits(2 * GRID * GRID, rng)
        x = QPSK.modulate(b)
        n0 = 10 ** (-snr_db / 10)
        zo = ofdm_frame(l, nu, h, x, n0, rng, p.eq != "One-tap")
        zt = otfs_frame(l, nu, h, x, n0, rng)
        return b, x, zo, zt

    def live(self, p, accumulate):
        if not accumulate:
            self.err_ofdm = self.err_otfs = self.bits = 0
            self.frames = 0
        b, x, zo, zt = self.make_frame(p, p.snr, self.rng)
        self.err_ofdm += int(np.sum(QPSK.demodulate(zo) != b))
        self.err_otfs += int(np.sum(QPSK.demodulate(zt) != b))
        self.bits += len(b)
        self.frames += 1
        self.last = (x, zo, zt)

    def draw(self, p):
        x, zo, zt = self.last
        for key, z in (("ofdm", zo), ("otfs", zt)):
            pc = self.plot(key)
            wrong = QPSK.decide(z) != x
            zc = np.clip(z.real, -1.45, 1.45) + 1j * np.clip(z.imag, -1.45, 1.45)
            pc.points("ok", zc[~wrong], color=NAVY, size=4, alpha=0.5)
            pc.points("bad", zc[wrong], color=RED, size=6, alpha=0.9)
            pc.ideal("ideal", QPSK.points)
        pb = self.plot("ber")
        s = np.linspace(0, 30, 61)
        pb.theory("ray", s, rayleigh_qpsk_ber(s), color=GRAY, style="--",
                  name="one Rayleigh path (no diversity)")
        if self.curves:
            xs = sorted(self.curves)
            yo = [max(self.curves[k][0], 1e-6) for k in xs]
            yt = [self.curves[k][1] for k in xs]
            pb.sim("cofdm", xs, yo, color=ORANGE, name="OFDM, " + ("one-tap" if self.p.eq == "One-tap" else "ICI-aware"))
            xt = [k for k, y in zip(xs, yt) if y > 0]
            if xt:
                pb.sim("cotfs", xt, [y for y in yt if y > 0], color=NAVY, name="OTFS LMMSE")
        lo, lt = self.err_ofdm / self.bits, self.err_otfs / self.bits
        if lo > 0:
            pb.scatter("lo", [p.snr], [lo], color=ORANGE, size=15, symbol="d", name="live OFDM")
        if lt > 0:
            pb.scatter("lt", [p.snr], [lt], color=RED, size=15, symbol="d", name="live OTFS")
        pb.vline("now", p.snr, color=GRAY, style=":")
        self.readout(nuT=self.numax(p), ofdm=lo if self.err_ofdm else 0.0,
                     otfs=lt if self.err_otfs else 0.0, frames=self.frames)

    def update(self, p):
        key = (p.speed, p.fc, p.scs, p.paths, p.eq)
        if key != self.curve_key:
            self.curves, self.curve_key, self.curve_done = {}, key, False
        self.live(p, False)
        self.draw(p)

    def tick(self, p):
        self.live(p, True)
        self.draw(p)

    def background(self, p):
        if self.curve_done:
            return
        key = self.curve_key
        rng = np.random.default_rng(25)
        trials = 2 if self.quick else 24
        for snr in self.SNRS:
            eo = et = nb = 0
            for _ in range(trials):
                b, x, zo, zt = self.make_frame(p, snr, rng)
                eo += int(np.sum(QPSK.demodulate(zo) != b))
                et += int(np.sum(QPSK.demodulate(zt) != b))
                nb += len(b)
            yield dict(key=key, snr=snr, ofdm=eo / nb, otfs=et / nb)
        yield dict(key=key, done=True)

    def progress(self, p, item):
        if item["key"] != self.curve_key:
            return
        if item.get("done"):
            self.curve_done = True
        else:
            self.curves[item["snr"]] = (item["ofdm"], item["otfs"])
        self.draw(p)

    def story(self, p):
        nT = self.numax(p)
        s = (f"<p>A 16 × 16 frame of QPSK crosses {v(p.paths, 'd')} path"
             f"{'s' if p.paths > 1 else ''} whose Dopplers reach {v(nT, '.2f')} subcarrier spacings "
             f"({v(p.speed, '.0f', 'km/h')} at {v(p.fc)} with {v(p.scs)} subcarriers). Press "
             f"<b>Play</b> to stream frames; the diamonds on the right are your running counts, the "
             f"circles a background sweep over SNR.</p>")
        if p.eq == "One-tap":
            s += ("<p><b>One-tap OFDM</b> divides each subcarrier by the symbol's average channel. "
                  "It ignores the energy Doppler leaks into the neighbours, so its errors level off "
                  "into an <b>ICI floor</b>, and because each subcarrier sees one flat-fading gain it "
                  "falls only as 1/SNR (grey dashed line) before that.</p>")
        else:
            s += ("<p>The <b>ICI-aware</b> equaliser knows the full 16 × 16 channel matrix of each "
                  "symbol and removes the floor, but each symbol is still a single fading "
                  "observation: no diversity across the frame.</p>")
        s += ("<p><b>OTFS</b> spreads every symbol over the whole frame, so each one sees every "
              "path, and a joint LMMSE detector (a 256 × 256 solve) collects that diversity: the "
              "curve falls much faster. Try one path: there is nothing to collect and the gap "
              "closes.</p>")
        return "<h3>Doppler as interference, Doppler as diversity</h3>" + s + keybox(
            "An unfair race, honestly labelled: OTFS solves a 256 × 256 system per frame, OFDM "
            "sixteen 16 × 16 ones, and both are uncoded. Coding across subcarriers closes much of the gap.")


# =============================================================================== 3. phase noise
_QAM_NEED = {"16-QAM": 17.0, "64-QAM": 25.0, "256-QAM": 31.0}


@lru_cache(maxsize=64)
def ceiling_curve(fc, l1m):
    scs = np.logspace(np.log10(15e3), np.log10(30e6), 48)
    return scs, np.array([fr.pn_ici_snr_db(fc, x, l1m) for x in scs])


@lru_cache(maxsize=256)
def min_scs(fc, l1m, need):
    """Smallest subcarrier spacing (Hz) whose phase-noise SNR ceiling reaches ``need`` dB."""
    lo, hi = np.log10(15e3), np.log10(60e6)
    if fr.pn_ici_snr_db(fc, 10 ** hi, l1m) < need:
        return np.inf
    for _ in range(14):
        mid = (lo + hi) / 2
        if fr.pn_ici_snr_db(fc, 10 ** mid, l1m) >= need:
            hi = mid
        else:
            lo = mid
    return 10 ** hi


class PhaseNoiseNumerology(Experiment):
    title = "Phase noise sets the numerology"
    blurb = "Why 6G at 140 GHz needs subcarriers hundreds of kilohertz wide."
    book = "sec:ch25:spectrum"
    animate = True
    fps = 10
    controls = [
        LogSlider("fc", "Carrier frequency", 3.5, 300, 28, unit="GHz", fmt=".1f"),
        LogSlider("scs", "Subcarrier spacing", 15, 15360, 120, unit="kHz", fmt=",.0f",
                  help="NR: 15·2^μ kHz; 120 kHz is FR2, 960 kHz the 52.6–71 GHz band"),
        Slider("l1m", "Synthesiser noise at 1 MHz", -110, -90, -100, step=1, unit="dBc/Hz",
               help="Referred to 28 GHz; multiplied up to fc it gains 20·log10(fc/28 GHz)"),
        Choice("qam", "Constellation", list(_QAM_NEED), "64-QAM", style="menu"),
    ]
    plots = [
        Plot("pn", "Phase noise L(f) and the part that becomes ICI", x="offset frequency (Hz)",
             y="dBc/Hz", logx=True, xlim=(1e3, 1e9), ylim=(-175, -40), legend="tr"),
        Plot("ceil", "SNR ceiling from phase-noise ICI", x="subcarrier spacing (kHz)", y="SNR (dB)",
             logx=True, xlim=(15, 30000), ylim=(0, 70), legend="tl", legend_cols=2),
        ConstellationPlot("con", "One frame after the FFT (8 × 256 subcarriers)", lim=1.35),
    ]
    layout = [["pn", "con"], ["ceil", "con"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("ceil", "SNR ceiling (theory)", "dB", ".1f"),
        Readout("mer", "Measured MER", "dB", ".1f"),
        Readout("rms", "RMS phase error", "°", ".2f"),
        Readout("ok", "Verdict", "", None),
    ]
    challenges = [
        Challenge("At 140 GHz, find the smallest subcarrier spacing that still gives 64-QAM its "
                  "25 dB (within 15 % of the minimum).",
                  lambda s: (abs(s.p.fc - 140) < 3 and s.p.qam == "64-QAM" and s.exp.ceil_num >= 25
                             and s.p.scs * 1e3 <= 1.15 * s.exp.min_scs)),
        Challenge("LTE-style 15 kHz subcarriers: find the highest carrier at which 64-QAM still "
                  "gets its 25 dB (within 10 % of the limit).",
                  lambda s: (abs(s.p.scs - 15) < 1 and s.p.qam == "64-QAM" and s.exp.ceil_num >= 25
                             and s.p.fc >= 0.9 * s.exp.fc_limit),
                  hint="The ceiling falls 6 dB for every doubling of the carrier."),
        Challenge("Make 256-QAM work at 140 GHz (ceiling ≥ 31 dB) with subcarriers no wider than "
                  "1 MHz.",
                  lambda s: (abs(s.p.fc - 140) < 3 and s.exp.ceil_num >= 31 and s.p.scs <= 1000),
                  hint="You need both: a wider spacing and a quieter synthesiser."),
    ]
    NFFT, NSYM = 256, 8

    def setup(self):
        self.ceil_num = 0.0
        self.min_scs = np.inf
        self.fc_limit = 0.0

    def make_frame(self, p):
        rng = self.rng
        fc, scs = p.fc * 1e9, p.scs * 1e3
        con = cl.get_constellation(p.qam.lower().replace("-", ""))
        n = self.NFFT * self.NSYM
        fs = self.NFFT * scs
        fm = np.logspace(np.log10(fs / n), np.log10(fs / 2), 200)
        phi = rf.phase_noise_from_mask(n, fs, fm, fr.pn_psd_dbc(fm, fc, p.l1m), rng)
        X = con.points[rng.integers(0, con.M, (self.NSYM, self.NFFT))]
        s = np.fft.ifft(X, axis=1) * np.sqrt(self.NFFT)
        r = s * np.exp(1j * phi.reshape(self.NSYM, self.NFFT))
        Y = np.fft.fft(r, axis=1) / np.sqrt(self.NFFT)
        c = np.sum(Y * X.conj(), axis=1)                    # pilots remove the common phase
        Y = Y * np.exp(-1j * np.angle(c))[:, None]
        mer = 10 * np.log10(np.sum(np.abs(X) ** 2) / max(np.sum(np.abs(Y - X) ** 2), 1e-30))
        return con, X, Y, mer

    def draw(self, p, accumulate):
        fc, scs = p.fc * 1e9, p.scs * 1e3
        need = _QAM_NEED[p.qam]
        con, X, Y, mer = self.make_frame(p)
        sm = getattr(self, "_mer", None)
        self._mer = mer if (not accumulate or sm is None) else 0.7 * sm + 0.3 * mer
        pc = self.plot("con")
        pc.points("y", Y.ravel(), color=NAVY, size=4, alpha=0.5)
        pc.ideal("i", con.points, color=RED, size=7)
        f = np.logspace(3, 9, 400)
        L = fr.pn_psd_dbc(f, fc, p.l1m)
        w = 1 - np.sinc(f / scs) ** 2
        pp = self.plot("pn")
        pp.line("L", f, L, color=NAVY, width=2.2, name="L(f)")
        pp.line("ici", f, L + 10 * np.log10(np.maximum(w, 1e-12)), color=RED, width=1.6,
                style="--", fill=-175, fill_alpha=0.12, name="the part that leaks into neighbours")
        pp.vline("scs", scs, color=ORANGE, style=":", label="Δf", label_pos=0.9)
        pc_ = self.plot("ceil")
        for f0, col in ((3.5e9, GRAY), (28e9, BLUE), (140e9, ORANGE), (300e9, PURPLE)):
            x, y = ceiling_curve(f0, float(p.l1m))
            pc_.line(f"c{f0}", x / 1e3, y, color=col, width=1.2, alpha=0.7, name=f"{f0 / 1e9:g} GHz")
        fkey = float(f"{fc:.2g}")
        x, y = ceiling_curve(fkey, float(p.l1m))
        pc_.line("cur", x / 1e3, y, color=NAVY, width=2.6, name="your carrier")
        ceil = fr.pn_ici_snr_db(fc, scs, p.l1m)
        pc_.scatter("now", [p.scs], [ceil], color=RED, size=13, symbol="d")
        pc_.hline("need", need, color=GREEN, style="--", label=f"{p.qam} needs ≈ {need:.0f} dB",
                  label_pos=0.62)
        self.ceil_num = ceil
        self.min_scs = min_scs(fkey, float(p.l1m), need)
        rms = rf.rms_phase_deg(f, L)
        verdict = f"{p.qam} ✓" if ceil >= need else f"{p.qam} ✗"
        self.fc_limit = 1e-9 * fc * 10 ** ((ceil - need) / 20)
        self.readout(ceil=ceil, mer=self._mer, rms=rms, ok=verdict)

    def update(self, p):
        self._mer = None
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        need = _QAM_NEED[p.qam]
        c = self.ceil_num
        s = (f"<p>An oscillator is never a perfect sinusoid: its phase wanders, and the spectrum "
             f"L(f) (top left) has skirts. Multiplying a synthesiser up to {v(p.fc, '.1f', 'GHz')} "
             f"raises those skirts by 20·log10(f<sub>c</sub>/28 GHz) = "
             f"{v(20 * np.log10(p.fc / 28), '+.1f', 'dB')}.</p>"
             f"<p>Slow phase wander rotates a whole OFDM symbol (the <b>common phase error</b>, "
             f"undone by pilots); wander faster than about the subcarrier spacing smears each "
             f"subcarrier into its neighbours (red dashed part). With {v(p.scs, ',.0f', 'kHz')} "
             f"subcarriers the ceiling is {v(c, '.1f', 'dB')}: "
             + (good(f"enough for {p.qam}.") if c >= need else bad(f"not enough for {p.qam} (≈ {need:.0f} dB).")))
        if np.isfinite(self.min_scs):
            s += f" It needs at least {v(self.min_scs / 1e3, ',.0f', 'kHz')} here."
        s += "</p>"
        return "<h3>Wider subcarriers for higher carriers</h3>" + s + keybox(
            "Phase noise grows with carrier frequency; ICI falls as the subcarrier spacing grows. "
            "That is why NR went to 120 kHz in FR2 and 960 kHz above 52.6 GHz, and why sub-THz "
            "6G looks at single-carrier waveforms.")


# =============================================================================== 4. OFDM radar
NUMER = {"15 kHz (μ = 0)": 0, "30 kHz (μ = 1)": 1, "60 kHz (μ = 2)": 2, "120 kHz (μ = 3)": 3}


class OFDMRadar(Experiment):
    title = "Your OFDM frame is a radar"
    blurb = "Divide out the data, take two FFTs, and every echo becomes a dot at its range and speed."
    book = "sec:ch25:isac"
    controls = [
        Heading("Targets"),
        Slider("r1", "Target 1 range", 2, 150, 40, step=0.5, unit="m"),
        Slider("v1", "Target 1 speed", -60, 60, 12, step=0.5, unit="m/s"),
        Slider("r2", "Target 2 range", 2, 150, 90, step=0.5, unit="m"),
        Slider("v2", "Target 2 speed", -60, 60, -25, step=0.5, unit="m/s"),
        Slider("a2", "Target 2 echo", -30, 0, -6, step=1, unit="dB"),
        Heading("Waveform at 28 GHz"),
        Choice("mu", "Numerology", list(NUMER), "120 kHz (μ = 3)", style="menu"),
        Choice("M", "Subcarriers", ["256", "512", "1024", "2048"], "512"),
        Choice("N", "Symbols", ["32", "64", "128", "256"], "128"),
        Slider("snr", "SNR per resource elem.", -40, 20, -10, step=1, unit="dB"),
        Toggle("win", "Hann window", True),
        Button("again", "New data and noise"),
    ]
    plots = [
        ImagePlot("rd", "Range–Doppler map (dB)", x="radial speed (m/s)", y="range (m)"),
        Plot("rng", "Range profile through target 1's speed", x="range (m)", y="dB",
             ylim=(-60, 3), legend=None),
        Plot("vel", "Speed profile through target 1's range", x="radial speed (m/s)", y="dB",
             ylim=(-60, 3), legend=None),
    ]
    layout = [["rd", "rng"], ["rd", "vel"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("dR", "Range resolution c/2B", "m", ".2f"),
        Readout("dv", "Speed resolution λ/2NT", "m/s", ".2f"),
        Readout("vmax", "Unambiguous speed ±", "m/s", ".0f"),
        Readout("snr1", "Target 1 above the map floor", "dB", ".1f", good=lambda x: x >= 10),
    ]
    challenges = [
        Challenge("Resolve two cars 2 m apart at the same speed: a dip of at least 3 dB between them.",
                  lambda s: (abs(s.p.r1 - s.p.r2) <= 2.05 and abs(s.p.v1 - s.p.v2) <= 0.5
                             and s.exp.dip >= 3),
                  hint="Range resolution is c/2B: more subcarriers. The window widens the peaks."),
        Challenge("Detect target 1 at −40 dB SNR per resource element, at least 10 dB above the floor.",
                  lambda s: s.p.snr <= -40 and s.r.snr1 >= 10,
                  hint="The two FFTs give a processing gain of 10·log10(M·N)."),
        Challenge("Make target 2 alias: pick a numerology whose unambiguous speed is below |v₂| and "
                  "watch it appear on the wrong side.",
                  lambda s: s.r.vmax < abs(s.p.v2) - 1,
                  hint="v_max = λ/(4T): long symbols (narrow subcarriers) see less speed."),
    ]
    fc = 28e9

    def setup(self):
        self.seed = 1
        self.dip = 0.0

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        nr = cl.nr_numerology(NUMER[p.mu])
        df = nr["scs_khz"] * 1e3
        Ts = (nr["useful_symbol_us"] + nr["cp_us"]) * 1e-6
        M, N = int(p.M), int(p.N)
        lam = C0 / self.fc
        X = QPSK.points[rng.integers(0, 4, (M, N))]
        m = np.arange(M)[:, None]
        n = np.arange(N)[None, :]
        H = np.zeros((M, N), complex)
        for R, vel, a in ((p.r1, p.v1, 1.0), (p.r2, p.v2, 10 ** (p.a2 / 20))):
            H += a * np.exp(-2j * np.pi * m * df * 2 * R / C0) * np.exp(2j * np.pi * n * Ts * 2 * vel / lam)
        Y = X * H + np.sqrt(10 ** (-p.snr / 10)) * cgauss(rng, (M, N))
        Z = Y / X                                                 # divide out the data
        wr = np.hanning(M)[:, None] if p.win else np.ones((M, 1))
        wd = np.hanning(N)[None, :] if p.win else np.ones((1, N))
        RD = np.fft.fftshift(np.fft.fft(np.fft.ifft(Z * wr * wd, axis=0), axis=1), axes=1)
        P = np.abs(RD) ** 2
        Pdb = 10 * np.log10(P / P.max() + 1e-30)
        dR = C0 / (2 * M * df)
        dv = lam / (2 * N * Ts)
        vmax = lam / (4 * Ts)
        rmax_show = min(160.0, C0 / (2 * df))
        nr_show = min(M, int(np.ceil(rmax_show / dR)) + 1)
        vax = (np.arange(N) - N / 2) * dv
        vshow = min(vmax, 70.0)
        cols = np.where(np.abs(vax) <= vshow)[0]
        img = np.clip(Pdb[:nr_show][:, cols], -50, 0)
        pr = self.plot("rd")
        pr.image("img", img, x=(vax[cols[0]] - dv / 2, vax[cols[-1]] + dv / 2),
                 y=(-dR / 2, (nr_show - 0.5) * dR), cmap="heat", levels=(-50, 0), colorbar=True,
                 cbar_label="dB")
        pr.set_xlim(-vshow, vshow)
        pr.set_ylim(0, rmax_show)
        tv = [((vv + vmax) % (2 * vmax)) - vmax for vv in (p.v1, p.v2)]
        pr.scatter("tgt", tv, [p.r1, p.r2], color=GREEN, size=18, symbol="o", outline=GREEN)
        pr.text("t1", tv[0], p.r1, "  1", color=GREEN, anchor=(0, 0.5), size=9, bold=True)
        pr.text("t2", tv[1], p.r2, "  2", color=GREEN, anchor=(0, 0.5), size=9, bold=True)
        # profiles through target 1 (zero-padded for smooth curves)
        c1 = int(np.argmin(np.abs(vax - tv[0])))
        pad = 8
        zr = np.fft.ifft(np.fft.fft(Z * wr * wd, axis=1)[:, (c1 + N // 2) % N], pad * M)
        rr = np.arange(pad * M) * dR / pad
        sel = rr <= rmax_show
        prof = np.abs(zr[sel]) ** 2
        prof_db = 10 * np.log10(prof / prof.max() + 1e-30)
        pg = self.plot("rng")
        pg.line("p", rr[sel], prof_db, color=NAVY, width=1.8)
        pg.vline("r1", p.r1, color=GREEN, style=":")
        pg.vline("r2", p.r2, color=GREEN, style=":")
        pg.set_xlim(0, rmax_show)
        # dip between the two targets when they share a speed column
        self.dip = 0.0
        if abs(p.v1 - p.v2) <= 0.5 and abs(p.r1 - p.r2) > 0.05:
            a_, b_ = sorted((p.r1, p.r2))
            i0, i1 = np.searchsorted(rr, [a_, b_])
            i0, i1 = max(i0, 0), min(i1, len(prof_db) - 1)
            if i1 > i0 + 1:
                seg = prof_db[i0:i1 + 1]
                lo_pk = min(prof_db[max(i0 - 2, 0):i0 + 3].max(), prof_db[i1 - 2:i1 + 3].max())
                self.dip = float(max(0.0, lo_pk - seg.min()))
        r1b = int(np.clip(round(p.r1 / dR), 0, M - 1))
        vline = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifft(Z * wr, axis=0)[r1b] * wd.ravel(),
                                                  pad * N)))
        vv = (np.arange(pad * N) - pad * N / 2) * dv / pad
        vdb = 20 * np.log10(vline / vline.max() + 1e-15)
        pvv = self.plot("vel")
        pvv.line("p", vv, vdb, color=NAVY, width=1.8)
        pvv.vline("v1", tv[0], color=GREEN, style=":")
        pvv.set_xlim(-vshow, vshow)
        # target 1 above the floor
        pk = P[r1b, c1]
        floor = np.median(P)
        snr1 = 10 * np.log10(pk / floor) if floor > 0 else 99.0
        self.gain = 10 * np.log10(M * N)
        self.tcp_range = C0 * nr["cp_us"] * 1e-6 / 2
        self.rmax = C0 / (2 * df)
        self.readout(dR=dR, dv=dv, vmax=vmax, snr1=snr1)

    def story(self, p):
        r = self.r
        s = (f"<p>A 28 GHz base station sends {v(int(p.M), 'd')} subcarriers × "
             f"{v(int(p.N), 'd')} symbols of ordinary QPSK data. Every echo returns that grid "
             f"multiplied by a phase ramp across subcarriers (its delay) and a phase ramp across "
             f"symbols (its Doppler). Divide by the known data and the ramps remain: an IFFT over "
             f"subcarriers finds the range, an FFT over symbols the speed.</p>"
             f"<p>Bandwidth sets the range resolution ({v(r.get('dR', 0), '.2f', 'm')}), dwell "
             f"time the speed resolution ({v(r.get('dv', 0), '.2f', 'm/s')}), and the two FFTs "
             f"add {v(self.gain, '.0f', 'dB')} of processing gain: at {v(p.snr, '.0f', 'dB')} per "
             f"resource element target 1 still stands {v(r.get('snr1', 0), '.0f', 'dB')} above the "
             f"floor.</p>")
        far = max(p.r1, p.r2)
        if far > self.tcp_range:
            s += (f"<p>{bad('Caution:')} a target beyond {v(self.tcp_range, '.0f', 'm')} returns "
                  f"after the cyclic prefix has ended, so in a real receiver it would leak between "
                  f"symbols (this model ignores that).</p>")
        if r.get("vmax", 999) < max(abs(p.v1), abs(p.v2)):
            s += ("<p>One target moves faster than the unambiguous speed λ/(4T): its Doppler phase "
                  "turns by more than π per symbol and it <b>aliases</b> to the wrong side "
                  "of the map (green circle shows where it lands).</p>")
        return "<h3>Sensing for free</h3>" + s + keybox(
            "ΔR = c/(2B), Δv = λ/(2NT), v_max = ±λ/(4T), processing gain M·N: the radar is designed "
            "by choosing the communication numerology.")


# =============================================================================== 5. RIS
class RISSquareLaw(Experiment):
    title = "RIS: power grows as N squared"
    blurb = "A passive surface of N phase-shifting elements adds its echoes coherently."
    book = "sec:ch25:mimo"
    controls = [
        LogSlider("N", "RIS elements", 1, 4096, 64, unit="", fmt=",.0f"),
        Choice("bits", "Phase control", ["1 bit", "2 bits", "3 bits", "continuous"], "2 bits"),
        Slider("direct", "Direct path level", -80, 0, -20, step=1, unit="dB",
               help="Relative to one element's cascaded path, which is −60 dB"),
        Toggle("on", "RIS configured (phases aligned)", True,
               help="Off: the elements keep random phases, as an unconfigured surface would"),
        Button("again", "New channel"),
    ]
    plots = [
        Plot("walk", "The received phasor: direct path, then every element's echo added in turn",
             x="in-phase", y="quadrature", aspect=True, legend="tl"),
        Plot("pw", "Received power against RIS size", x="RIS elements N", y="power (dB)",
             logx=True, xlim=(1, 4096), ylim=(-85, 30), legend="tl", legend_cols=2),
        BarPlot("q", "Coherent-gain loss of b-bit phases", y="loss (dB)"),
    ]
    layout = [["walk", "walk"], ["pw", "q"]]
    col_stretch = [3, 2]
    row_stretch = [2, 3]
    readouts = [
        Readout("pr", "Received power", "dB", ".1f"),
        Readout("gain", "Gain over the direct path", "dB", ".1f"),
        Readout("loss", "Phase-quantisation loss", "dB", ".2f", good=lambda x: x < 1),
        Readout("neq", "N to match the direct path", "", ",.0f"),
    ]
    challenges = [
        Challenge("Double the received amplitude (+6 dB over the direct path at −20 dB) with 2-bit "
                  "phases and no more than 15 % more elements than needed.",
                  lambda s: (abs(s.p.direct + 20) < 0.5 and s.p.bits == "2 bits" and s.p.on
                             and s.exp.gain_mean >= 6 and s.p.N <= 1.15 * s.exp.n6)),
        Challenge("Find the phase resolution beyond which quantisation costs less than 0.25 dB.",
                  lambda s: s.p.bits == "3 bits" and s.p.on),
        Challenge("Leave the surface unconfigured (random phases) and find how many elements it takes "
                  "just to add 1 dB to the −20 dB direct path.",
                  lambda s: (not s.p.on and abs(s.p.direct + 20) < 0.5 and s.exp.gain_mean >= 1
                             and s.p.N <= 1.2 * s.exp.n1_random)),
    ]
    SIG = 10 ** (-60 / 20)                     # rms amplitude of one element's cascaded path

    def setup(self):
        self.seed = 4

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def sb(bits):
        return 1.0 if bits == "continuous" else float(np.sinc(2.0 ** -int(bits[0])))

    def mean_power(self, N, d, bits, on):
        """Analytic mean received power for aligned (b-bit) or random phases."""
        N = np.asarray(N, float)
        sig = self.SIG
        if not on:
            return d ** 2 + N * sig ** 2
        s = self.sb(bits)
        a = np.sqrt(np.pi) / 2 * sig                   # E|g| for a Rayleigh cascaded gain
        return (d + s * N * a) ** 2 + N * sig ** 2 * (1 - s ** 2 * np.pi / 4)

    def update(self, p):
        N = int(round(p.N))
        d = 10 ** (p.direct / 20)
        rng = np.random.default_rng(self.seed)
        g = self.SIG * cgauss(rng, 4096)[:N]
        dph = rng.uniform(0, 2 * np.pi)
        dvec = d * np.exp(1j * dph)
        if p.on:
            th = dph - np.angle(g)
            if p.bits != "continuous":
                q = 2 * np.pi / 2 ** int(p.bits[0])
                th = q * np.round(th / q)
        else:
            th = rng.uniform(0, 2 * np.pi, N)
        contrib = g * np.exp(1j * th)
        path = dvec + np.r_[0, np.cumsum(contrib)]
        # rotate so that the direct path points right
        rot = np.exp(-1j * dph)
        path = path * rot
        pw = self.plot("walk")
        pw.line("dir", [0, d], [0, 0], color=RED, width=3.0, name="direct path")
        pw.line("ris", path.real, path.imag, color=NAVY, width=1.6,
                name=f"{N} RIS echoes, added in turn")
        if N <= 64:
            pw.scatter("pts", path.real[1:], path.imag[1:], color=NAVY, size=4)
        pw.scatter("end", [path[-1].real], [path[-1].imag], color=GREEN, size=12, symbol="star",
                   name="total")
        pw.line("tot", [0, path[-1].real], [0, path[-1].imag], color=GREEN, width=1.4, style="--")
        span = max(abs(path).max(), d) * 1.15 + 1e-9
        pw.set_xlim(-0.05 * span, 1.05 * span)
        pw.set_ylim(-0.12 * span, 0.12 * span)
        # power curves
        Ns = np.unique(np.round(np.logspace(0, np.log10(4096), 120)))
        pp = self.plot("pw")
        pp.line("ideal", Ns, db(self.mean_power(Ns, d, "continuous", True)), color=NAVY,
                width=2.2, name="aligned, continuous")
        if p.bits != "continuous":
            pp.line("quant", Ns, db(self.mean_power(Ns, d, p.bits, True)), color=ORANGE, width=2.0,
                    name=f"aligned, {p.bits}")
        pp.line("rand", Ns, db(self.mean_power(Ns, d, p.bits, False)), color=GRAY, width=1.6,
                name="random phases")
        pp.line("n2", Ns, db(Ns ** 2 * (np.sqrt(np.pi) / 2 * self.SIG) ** 2), color=PURPLE,
                width=1.0, style=":", name="N² law")
        pp.hline("d", p.direct, color=RED, style="--", label="direct path alone", label_pos=0.72)
        prx = abs(path[-1]) ** 2
        pp.scatter("now", [N], [db(prx)], color=RED, size=13, symbol="d", name="this channel")
        # quantisation-loss bars
        pq = self.plot("q")
        losses = [fr.ris_phase_loss_db(b) for b in (1, 2, 3, 4)]
        cur = {"1 bit": 0, "2 bits": 1, "3 bits": 2}.get(p.bits, 4)
        pq.bars("b", [0, 1, 2, 3], losses, width=0.6,
                colors=[RED if i == cur else NAVY for i in range(4)])
        for i, L in enumerate(losses):
            pq.text(f"t{i}", i, L, f"{L:.2f} dB", anchor=(0.5, 1.05), size=8.5, bold=True)
        pq.set_xticks([(0, "1 bit"), (1, "2 bits"), (2, "3 bits"), (3, "4 bits")])
        pq.set_xlim(-0.6, 3.6)
        pq.set_ylim(0, 4.8)
        # readouts (analytic means are what the challenges use)
        self.gain_mean = float(db(self.mean_power(N, d, p.bits, p.on)) - p.direct)
        a = np.sqrt(np.pi) / 2 * self.SIG
        s = self.sb(p.bits)
        self.n6 = d / (s * a)                                      # amplitude doubling
        self.n1_random = (10 ** 0.1 - 1) * d ** 2 / self.SIG ** 2
        loss = 0.0 if p.bits == "continuous" else fr.ris_phase_loss_db(int(p.bits[0]))
        self.readout(pr=float(db(prx)), gain=float(db(prx) - p.direct),
                     loss=loss if p.on else "—", neq=d / (s * a))

    def story(self, p):
        N = int(round(p.N))
        r = self.r
        if p.on:
            s = (f"<p>Each element re-radiates a tiny echo (about −60 dB, a double path loss), but "
                 f"the controller sets every element's phase so that all {v(N, ',d')} echoes "
                 f"arrive in step with the direct path: at the top they line up head to tail into "
                 f"a straight arrow. Amplitude grows as N, so power grows as <b>N²</b>: 20 dB per "
                 f"decade of elements (bottom left).</p>")
            if p.bits != "continuous":
                s += (f"<p>With only {v(p.bits)} of phase control every echo is a little off "
                      f"(the arrow wiggles): the coherent sum loses "
                      f"{v(r.get('loss', 0) if isinstance(r.get('loss'), float) else 0, '.2f', 'dB')}, "
                      f"the sinc law 20·log10 sinc(2<sup>−b</sup>).</p>")
        else:
            s = ("<p>Unconfigured, the echoes add with random phases: a random walk (top), "
                 "power growing only as N (10 dB per decade, grey). A surface is only useful when "
                 "somebody computes and sets its phases.</p>")
        s += (f"<p>To match the direct path the surface needs about "
              f"{v(r.get('neq', 0), ',.0f')} elements. Against a blocked direct path a few hundred "
              f"suffice; against an open line of sight it takes thousands.</p>")
        return "<h3>Coherent, passive, and big</h3>" + s + keybox(
            "An RIS gains N² but starts from a double path loss: hundreds to thousands of elements, "
            "and 2-bit phases (0.9 dB) are usually enough.")


# =============================================================================== 6. learned demapper
def checker_colour(con):
    """Colour index (0..3) per constellation label from its position on the square grid."""
    pts = con._lut
    lv = np.unique(np.round(pts.real, 6))
    i = np.searchsorted(lv, np.round(pts.real, 6))
    j = np.searchsorted(lv, np.round(pts.imag, 6))
    return (i % 2) + 2 * (j % 2)


_CHK16 = checker_colour(Q16)


def features(y):
    return np.stack([y.real, y.imag, np.abs(y), np.abs(y) ** 2], axis=1)


class LearnedDemapper(Experiment):
    title = "A demapper that learns"
    blurb = "Train a tiny neural network live to detect 16-QAM through a saturating PA and phase noise."
    book = "sec:ch25:ai"
    animate = True
    fps = 12
    controls = [
        Heading("The impaired transmitter"),
        Slider("sat", "PA saturation level", 0.6, 2.0, 1.0, step=0.05, unit="× rms",
               help="Rapp PA: amplitudes approaching this level are compressed"),
        Slider("pn", "Phase noise (rms)", 0, 10, 4, step=0.5, unit="°"),
        Slider("snr", "Es/N0", 10, 30, 20, step=0.5, unit="dB"),
        Heading("The network (Play = train)"),
        Choice("hidden", "Hidden neurons", ["8", "16", "32", "64"], "32"),
        LogSlider("lr", "Learning rate", 3e-4, 3e-2, 3e-3, unit="", fmt=".4f"),
        Button("reset", "Reset to the textbook grid"),
    ]
    plots = [
        Plot("map", "Received symbols and the learned decision regions", x="in-phase",
             y="quadrature", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), legend=None, grid=False),
        Plot("curve", "Learning curve: symbol error rate", x="training iterations", y="SER",
             logy=True, ylim=(1e-4, 1), legend="tr"),
        BarPlot("bars", "Symbol error rate by detector", y="SER (%)"),
    ]
    layout = [["map", "curve"], ["map", "bars"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("it", "Iterations", "", "int"),
        Readout("md", "SER, minimum distance", "", "sci"),
        Readout("nn", "SER, learned", "", "sci"),
        Readout("cent", "SER, nearest centroid", "", "sci"),
    ]
    challenges = [
        Challenge("Train until the network makes at most half the errors of the minimum-distance "
                  "detector.",
                  lambda s: s.r.it > 0 and s.r.nn <= 0.5 * s.r.md and s.r.md > 0,
                  hint="Press Play and wait: the regions bend to follow the compressed corners."),
        Challenge("Starve it: with only 8 hidden neurons, still beat minimum distance.",
                  lambda s: s.p.hidden == "8" and s.r.it > 0 and s.r.nn < s.r.md),
        Challenge("Drive the PA hard (saturation ≤ 0.8) at Es/N0 ≤ 20 dB and train the learned SER "
                  "below 4 %.",
                  lambda s: s.p.sat <= 0.8 and s.p.snr <= 20 and s.r.it > 0 and s.r.nn < 0.04),
    ]
    NEVAL = 20000

    def setup(self):
        self.net = None
        self.key = None
        self.hist_it, self.hist_ser = [], []
        self.its = 0

    def on_reset(self, p):
        self.net = None

    def impaired(self, idx, p, rng):
        x = cl.rapp_pa(Q16._lut[idx], sat=p.sat, p=2.0)
        x = x * np.exp(1j * np.deg2rad(p.pn) * rng.standard_normal(len(x)))
        return x + np.sqrt(10 ** (-p.snr / 10)) * cgauss(rng, len(x))

    def update(self, p):
        if self.net is None or self.key != p.hidden:
            self.net = fr.MLP([4, int(p.hidden), 16], np.random.default_rng(12))
            # start from the textbook: pre-train on the ideal grid with a little noise
            r0 = np.random.default_rng(3)
            for _ in range(250):
                idx = r0.integers(0, 16, 256)
                y0 = Q16._lut[idx] + 0.09 * cgauss(r0, 256)
                self.net.train_step(features(y0), idx, lr=1e-2)
            self.key = p.hidden
            self.hist_it, self.hist_ser = [], []
            self.its = 0
        rng = np.random.default_rng(7)
        self.ev_idx = rng.integers(0, 16, self.NEVAL)
        self.ev_y = self.impaired(self.ev_idx, p, rng)
        self.ev_f = features(self.ev_y)
        self.ser_md = float(np.mean(Q16.nearest(self.ev_y) != self.ev_idx))
        # nearest-centroid detector: cluster means measured from 4000 pilots
        pidx = np.repeat(np.arange(16), 250)
        py = self.impaired(pidx, p, rng)
        self.cent = np.array([py[pidx == k].mean() for k in range(16)])
        d = np.abs(self.ev_y[:, None] - self.cent[None, :])
        self.ser_cent = float(np.mean(np.argmin(d, 1) != self.ev_idx))
        g = np.linspace(-1.6, 1.6, 121)
        Xg, Yg = np.meshgrid(g, g)
        self.grid_f = features((Xg + 1j * Yg).ravel())
        self.draw(p)

    def train(self, p, steps=30):
        rng = self.rng
        for _ in range(steps):
            idx = rng.integers(0, 16, 256)
            self.net.train_step(features(self.impaired(idx, p, rng)), idx, lr=p.lr)
        self.its += steps

    def tick(self, p):
        self.train(p)
        self.draw(p)

    def draw(self, p):
        pred = np.argmax(self.net.forward(self.ev_f), 1)
        ser_nn = float(np.mean(pred != self.ev_idx))
        if not self.hist_it or self.hist_it[-1] != self.its:
            self.hist_it.append(self.its)
            self.hist_ser.append(max(ser_nn, 1e-4))
        reg = np.argmax(self.net.forward(self.grid_f), 1).reshape(121, 121)
        pm = self.plot("map")
        pm.image("reg", _CHK16[reg], x=(-1.6, 1.6), y=(-1.6, 1.6),
                 cmap=region_stops(pm.theme.name), levels=(0, 5))
        k = 2500
        yy = self.ev_y[:k]
        wrong = pred[:k] != self.ev_idx[:k]
        pm.scatter("ok", yy.real[~wrong], yy.imag[~wrong], color=NAVY, size=3, alpha=0.4)
        pm.scatter("bad", yy.real[wrong], yy.imag[wrong], color=RED, size=5, alpha=0.9)
        pm.scatter("ideal", Q16.points.real, Q16.points.imag, color=GRAY, size=11, symbol="+",
                   alpha=1.0, z=5)
        for t in (-2 / np.sqrt(10), 0, 2 / np.sqrt(10)):
            pm.vline(f"gv{t:.2f}", t, color=GRAY, style=":", width=0.8)
            pm.hline(f"gh{t:.2f}", t, color=GRAY, style=":", width=0.8)
        pc = self.plot("curve")
        xs = np.maximum(np.array(self.hist_it, float), 0)
        pc.line("nn", xs, self.hist_ser, color=NAVY, width=2.2, name="learned")
        pc.hline("md", max(self.ser_md, 1.1e-4), color=RED, style="--", label="minimum distance",
                 label_pos=0.65)
        pc.hline("ce", max(self.ser_cent, 1.1e-4), color=ORANGE, style=":", label="nearest centroid",
                 label_pos=0.3)
        pc.set_xlim(0, max(600, self.its * 1.1))
        pb = self.plot("bars")
        vals = [100 * self.ser_md, 100 * self.ser_cent, 100 * ser_nn]
        pb.bars("b", [0, 1, 2], vals, width=0.6, colors=[RED, ORANGE, NAVY])
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.2f} %", anchor=(0.5, 1.05), size=8.5, bold=True)
        pb.set_xticks([(0, "min. distance"), (1, "centroid"), (2, "learned")])
        pb.set_xlim(-0.6, 2.6)
        pb.set_ylim(0, max(vals) * 1.3 + 0.2)
        self.readout(it=self.its, md=self.ser_md, nn=ser_nn, cent=self.ser_cent)

    def story(self, p):
        r = self.r
        s = (f"<p>The PA squashes the outer points towards the centre (saturation "
             f"{v(p.sat, '.2f', '× rms')}) and {v(p.pn, '.1f', '°')} of phase noise smears every "
             f"point into an arc. The textbook detector still uses the square grid (dotted lines) "
             f"and errs on {v(100 * r.get('md', 0), '.2f', '%')} of symbols.</p>"
             f"<p>The network starts out as that textbook detector (it was pre-trained on the ideal "
             f"grid) and then sees only examples: received samples and the symbols that were "
             f"sent. Press <b>Play</b> and watch the coloured regions bend to follow the clusters; "
             f"after {v(r.get('it', 0), ',d')} iterations it errs on "
             f"{v(100 * r.get('nn', 0), '.2f', '%')}.</p>")
        if r.get("nn", 1) < r.get("md", 0):
            s += (f"<p>{good('It beats the textbook detector.')} But look at the orange bar: simply "
                  f"measuring where each cluster's centre moved (a model-based fix) gets "
                  f"{v(100 * r.get('cent', 0), '.2f', '%')}. Learning earns its keep only when the "
                  f"distortion is hard to model, like the arcs of phase noise.</p>")
        return "<h3>Learning the decision regions</h3>" + s + keybox(
            "A learned receiver is a function fitted to data. Always compare it with a good "
            "model-based baseline, not just with a detector that ignores the impairment.")


# =============================================================================== 7. autoencoder
_REF = {4: ("QPSK", lambda: cl.get_constellation("qpsk").points),
        8: ("8-PSK", lambda: cl.get_constellation("8psk").points),
        16: ("16-QAM", lambda: cl.get_constellation("16qam").points),
        32: ("32-APSK", lambda: mz.constellation("32apsk").points)}


def dmin(pts):
    d = np.abs(pts[:, None] - pts[None, :])
    return d[~np.eye(len(pts), dtype=bool)].min()


def es_at_ser(pts, target=1e-3):
    """Es/N0 (dB) at which the union-bound SER of ``pts`` (unit average energy) reaches target."""
    es = np.arange(0, 32, 0.25)
    y = np.log10(union_ser(pts, es) + 1e-300)
    return float(np.interp(-np.log10(target), -y, es))


def union_ser(pts, esn0_db):
    """Union bound on the SER of equiprobable points (scaled as given) at Es/N0 = esn0_db, where
    Es is 1 (the caller normalises)."""
    d = np.abs(pts[:, None] - pts[None, :])[~np.eye(len(pts), dtype=bool)]
    n0 = 10 ** (-np.atleast_1d(np.asarray(esn0_db, float)) / 10)
    out = np.sum(cl.qfunc(d[None, :] / np.sqrt(2 * n0)[:, None]), axis=1) / len(pts)
    return np.minimum(out, 1.0)


class Autoencoder(Experiment):
    title = "An autoencoder invents a constellation"
    blurb = "Let transmitter and receiver learn together: points and decoder trained end to end."
    book = "sec:ch25:ai"
    animate = True
    fps = 12
    controls = [
        Choice("M", "Messages", ["4", "8", "16", "32"], "16"),
        Slider("esn0", "Training Es/N0", 4, 20, 14, step=0.5, unit="dB"),
        Toggle("peak", "Peak-power limit (instead of average)", False,
               help="Normalise the points so that the largest has unit power, as a saturated PA demands"),
        Button("reset", "Start again from random points"),
    ]
    plots = [
        Plot("pts", "Learned points and the decoder's regions", x="in-phase", y="quadrature",
             xlim=(-1.9, 1.9), ylim=(-1.9, 1.9), legend="tl", grid=False),
        Plot("dm", "Gain over the textbook constellation", x="training iterations",
             y="gain (dB)", ylim=(-12, 3), legend=None),
        BERPlot("ser", "Symbol error rate (union bound)", x="Es/N0 (dB)", y="SER",
                xlim=(0, 24), ylim=(1e-6, 1), legend="bl"),
    ]
    layout = [["pts", "dm"], ["pts", "ser"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("it", "Iterations", "", "int"),
        Readout("gain", "Gain over the textbook", "dB", "+.2f", good=lambda x: x > 0,
                help="Average-power limit: Es/N0 saved at SER 10⁻³. Peak limit: d_min gain at equal peak"),
        Readout("papr", "Peak / average power", "dB", ".2f"),
        Readout("loss", "Training loss", "", ".3f"),
    ]
    challenges = [
        Challenge("Four messages: train until it rediscovers QPSK (within 0.05 dB).",
                  lambda s: s.p.M == "4" and not s.p.peak and s.r.it > 0 and s.r.gain >= -0.05,
                  hint="At high Es/N0 the errors vanish and so does the push: train at a low Es/N0."),
        Challenge("Sixteen messages: beat 16-QAM by at least 0.2 dB at SER 10⁻³.",
                  lambda s: s.p.M == "16" and not s.p.peak and s.r.gain >= 0.2,
                  hint="Hexagonal packing beats the square grid. Be patient, and train near the SNR "
                       "where you want to win."),
        Challenge("Peak-limited, eight messages: find a constellation better than 8-PSK "
                  "(positive d_min gain at the same peak power).",
                  lambda s: s.p.M == "8" and s.p.peak and s.r.gain > 0.05,
                  hint="One point in the middle, seven on the rim."),
    ]

    def setup(self):
        self.P = None
        self.key = None

    def on_reset(self, p):
        self.P = None

    def init(self, p):
        M = int(p.M)
        rng = np.random.default_rng(5 + M)
        self.P = rng.standard_normal((M, 2)) * 0.5
        self.net = fr.MLP([2, 64, 64, M], rng)
        self.opt = fr.Adam(self.P.shape)
        self.its = 0
        self.hist_it, self.hist_g = [], []
        self.loss = np.log(M)
        self.key = (p.M, p.peak)

    def norm(self, P, peak):
        if peak:
            r = np.sqrt(np.sum(P ** 2, 1))
            k = int(np.argmax(r))
            return P / r[k], r[k], k
        s = np.sqrt(np.mean(np.sum(P ** 2, 1)))
        return P / s, s, None

    def train(self, p, steps):
        M = len(self.P)
        rng = self.rng
        sigma = np.sqrt(10 ** (-p.esn0 / 10) / 2)
        for _ in range(steps):
            lr = 3e-3 if self.its < 4000 else 1e-3 if self.its < 9000 else 4e-4
            Pn, s, k = self.norm(self.P, p.peak)
            idx = rng.integers(0, M, 512)
            y = Pn[idx] + sigma * rng.standard_normal((512, 2))
            loss, gy = self.net.train_step(y, idx, lr=lr)
            gPn = np.zeros_like(self.P)
            np.add.at(gPn, idx, gy)
            if k is None:
                gP = gPn / s - self.P * np.sum(gPn * self.P) / (s ** 3 * M)
            else:
                gP = gPn / s
                gP[k] -= self.P[k] * np.sum(gPn * self.P) / s ** 3
            self.opt.step(self.P, gP, lr)
            self.its += 1
            self.loss = 0.95 * self.loss + 0.05 * loss

    def points(self, p):
        Pn, _, _ = self.norm(self.P, p.peak)
        return Pn[:, 0] + 1j * Pn[:, 1]

    def reference(self, p):
        name, fn = _REF[int(p.M)]
        ref = fn()
        if p.peak:
            ref = ref / np.max(np.abs(ref))
        return name, ref

    def draw(self, p):
        pts = self.points(p)
        name, ref = self.reference(p)
        if p.peak:                       # same peak power: compare distances directly
            gain = 20 * np.log10(dmin(pts) / dmin(ref))
        else:                            # same average power: Es/N0 needed for SER 10^-3
            gain = es_at_ser(ref / np.sqrt(np.mean(np.abs(ref) ** 2))) - es_at_ser(pts)
        if not self.hist_it or self.hist_it[-1] != self.its:
            self.hist_it.append(self.its)
            self.hist_g.append(gain)
        g = np.linspace(-1.9, 1.9, 111)
        Xg, Yg = np.meshgrid(g, g)
        reg = np.argmax(self.net.forward(np.stack([Xg.ravel(), Yg.ravel()], 1)), 1).reshape(111, 111)
        pp = self.plot("pts")
        pp.image("reg", colour_regions(reg, len(pts)), x=(-1.9, 1.9), y=(-1.9, 1.9),
                 cmap=region_stops(pp.theme.name), levels=(0, 5))
        th = np.linspace(0, 2 * np.pi, 200)
        pp.line("unit", np.cos(th), np.sin(th), color=GRAY, width=0.8, style=":")
        pp.scatter("ref", ref.real, ref.imag, color=GRAY, size=10, symbol="x", alpha=0.9,
                   name=name)
        pp.scatter("pts", pts.real, pts.imag, color=NAVY, size=11, alpha=1.0, z=5, name="learned")
        pd = self.plot("dm")
        pd.line("g", self.hist_it, self.hist_g, color=NAVY, width=2.0)
        pd.hline("z", 0, color=RED, style="--", label=f"{name}", label_pos=0.05)
        pd.set_xlim(0, max(1000, self.its * 1.1))
        ps = self.plot("ser")
        es = np.linspace(0, 24, 49)
        ps.theory("ref", es, union_ser(ref, es), color=GRAY, style="--", name=name)
        ps.theory("lrn", es, union_ser(pts, es), color=NAVY, name="learned")
        ps.vline("tr", p.esn0, color=ORANGE, style=":", label="training Es/N0", label_pos=0.05)
        papr = 10 * np.log10(np.max(np.abs(pts) ** 2) / np.mean(np.abs(pts) ** 2))
        self.readout(it=self.its, gain=gain, papr=papr, loss=self.loss)

    def update(self, p):
        if self.P is None or self.key != (p.M, p.peak):
            self.init(p)
        self.draw(p)

    def tick(self, p):
        self.train(p, 18)
        self.draw(p)

    def story(self, p):
        r = self.r
        name, _ = self.reference(p)
        s = (f"<p>O'Shea and Hoydis (2017) noticed that a link is an <b>autoencoder</b>: a network "
             f"maps each of the {v(int(p.M), 'd')} messages to a point (the transmitter), noise is "
             f"added, and a second network guesses the message (the receiver). Train both together "
             f"to minimise errors and the points arrange themselves. Press <b>Play</b> and watch "
             f"random dots spread out.</p>")
        if p.peak:
            s += ("<p>With a <b>peak-power limit</b> (all points inside the dotted unit circle, as a "
                  "saturated amplifier demands), rings win: the network discovers that a point in the "
                  "centre costs no peak power.</p>")
        else:
            s += (f"<p>With an average-power limit it tends to a <b>hexagonal</b> packing, the "
                  f"densest in the plane, which is why it can beat {name}'s square grid by a "
                  f"fraction of a dB.</p>")
        g = r.get("gain", -9)
        s += (f"<p>Right now its minimum distance is {v(g, '+.2f', 'dB')} relative to {name} "
              f"(peak/average {v(r.get('papr', 0), '.2f', 'dB')}). "
              + (good("It has found something better than the textbook. ") if g > 0.05 else "")
              + "The gain is real but small: the textbook constellations were already good.</p>")
        return "<h3>Learning the transmitter too</h3>" + s + keybox(
            "End-to-end learning finds shaping gains of a fraction of a dB on AWGN; its promise lies "
            "in channels and hardware too awkward to model by hand.")


# =============================================================================== the lab
LAB = st.Lab(12, "Frontiers: Delay–Doppler, ISAC, RIS and Learned Receivers", chapter=25,
             chapter_title="The Road to 6G",
             experiments=[DelayDopplerView, OTFSvsOFDM, PhaseNoiseNumerology, OFDMRadar,
                          RISSquareLaw, LearnedDemapper, Autoencoder])

if __name__ == "__main__":
    st.run(LAB)
