"""Lab 02 · Digital Modulation and Optimal Detection   (Chapter 9)

Run it:      python labs/lab02_modulation.py
Self-test:   python labs/lab02_modulation.py --selftest

A modulator turns bits into points; a detector draws the best possible borders between them.
Nine experiments walk from the constellation zoo (PSK, QAM, APSK, cross QAM and their Gray
labels) through signal space, decision regions and MAP detection, live Monte Carlo error
rates, soft LLRs, orthogonal FSK and noncoherent detection, the bandwidth–power plane with
its Shannon limit, and finally what Rayleigh fading does to all of it.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import modzoo as mz
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
ZOO = ["BPSK", "QPSK", "8-PSK", "16-PSK", "4-PAM", "8-PAM", "16-QAM", "32-cross QAM", "64-QAM",
       "256-QAM", "16-APSK", "32-APSK"]
LABELLINGS = {"Gray": "gray", "Natural binary": "natural", "Random": "random"}


@lru_cache(maxsize=96)
def const(name, ratio=None, labels="gray"):
    """Labelled unit-energy constellation by display name (cached)."""
    key = mz.NAMES[name]
    r = None if "APSK" not in name or ratio is None else round(float(ratio), 3)
    return mz.constellation(key, r, labels)


def region_colors(theme_name):
    if theme_name == "dark":
        return ((0, "#1E2B3A"), (0.2, "#3A2624"), (0.4, "#1F352A"), (0.6, "#3B2E20"),
                (0.8, "#2F2539"), (1.0, "#2B3036"))
    return ((0, "#DCE6F0"), (0.2, "#F3DCD9"), (0.4, "#DCEFE4"), (0.6, "#F6E6D6"),
            (0.8, "#E6DCEF"), (1.0, "#E4E4E4"))


@lru_cache(maxsize=48)
def region_map(name, ratio, lim=3.2, n=321):
    """Decision-region map: nearest-point index on an n×n grid, recoloured for contrast."""
    c = const(name, ratio)
    g = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(g, g)
    idx = c.nearest((X + 1j * Y).ravel()).reshape(X.shape)
    # greedy graph colouring of the cells with 6 colours so that touching cells differ
    adj = {i: set() for i in range(c.M)}
    for a_, b_ in ((idx[:, 1:], idx[:, :-1]), (idx[1:, :], idx[:-1, :])):
        m = a_ != b_
        for u, w in set(zip(a_[m].tolist(), b_[m].tolist())):
            adj[u].add(w)
            adj[w].add(u)
    col = {}
    for u in sorted(adj, key=lambda q: -len(adj[q])):
        used = {col[w] for w in adj[u] if w in col}
        col[u] = next(k for k in range(6) if k not in used) if len(used) < 6 else 0
    lut = np.array([col.get(i, 0) for i in range(c.M)])
    return lut[idx]


def bitstr(label, k):
    return format(int(label), f"0{k}b")


def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


# =============================================================================== 1. gallery
class Gallery(Experiment):
    title = "Constellation gallery"
    blurb = "Twelve constellations that carry most of the world's radio traffic, with their labels."
    book = "sec:ch09:constellations"
    controls = [
        Choice("name", "Constellation", ZOO, "16-QAM", style="menu"),
        Slider("ratio", "APSK ring ratio γ", 1.5, 4.0, 2.85, step=0.01,
               help="Outer-ring radius over inner-ring radius (DVB-S2 uses about 2.6–3.2)",
               enabled_if=lambda p: "APSK" in p.name),
        Choice("labels", "Bit labelling", list(LABELLINGS), "Gray"),
        Toggle("show_bits", "Show bit labels", True),
        Toggle("links", "Link nearest neighbours", True),
    ]
    plots = [
        Plot("pts", "Constellation", x="in-phase I", y="quadrature Q", xlim=(-1.65, 1.65),
             ylim=(-1.65, 1.65), aspect=True, legend=None),
        Plot("map", "The whole zoo: distance per bit against peak power",
             x="peak / average power (dB)", y="d²min / Eb (dB)", xlim=(-0.4, 4.6),
             ylim=(-4.0, 7.5), legend=None),
    ]
    layout = [["pts", "map"]]
    col_stretch = [5, 4]
    readouts = [
        Readout("k", "Bits per symbol", "", "int"),
        Readout("d2", "d²min / Es", "", ".3f", help="Squared minimum distance at unit average energy"),
        Readout("papr", "Peak / average", "dB", ".2f", good=lambda x: x < 1.0),
        Readout("flip", "Bits flipped per neighbour error", "", ".2f", good=lambda x: x <= 1.001,
                help="Average label bits that differ between nearest neighbours (1.00 = Gray)"),
    ]
    challenges = [
        Challenge("Tune 16-APSK's ring ratio to maximise its minimum distance (d²min within "
                  "1 % of the best possible).",
                  lambda s: s.p.name == "16-APSK" and s.r.d2 >= 0.99 * 0.34197,
                  hint="Watch d²min as you slide γ: the inner ring, the outer ring and the gap "
                       "between them all compete."),
        Challenge("Find the one constellation here whose labels cannot all be Gray, even with "
                  "the best labelling.",
                  lambda s: s.p.labels == "Gray" and s.r.flip > 1.01,
                  hint="A square grid with its corners cut off."),
        Challenge("Make a nearest-neighbour slip cost more than 2 bits on average.",
                  lambda s: s.r.flip > 2.0),
        Challenge("Beat 32-cross QAM's peak/average power by at least 0.1 dB with another "
                  "32-point constellation.",
                  lambda s: s.p.name == "32-APSK" and s.r.papr <= 2.30 - 0.1),
    ]

    def update(self, p):
        c = const(p.name, p.ratio, LABELLINGS[p.labels])
        pts = c.points
        pp = self.plot("pts")
        pp.set_title(f"{p.name}: {c.M} points, {c.k} bit{'s' if c.k > 1 else ''} each")
        pp.hline("h0", 0, color=GRAY, style="-", width=0.6)
        pp.vline("v0", 0, color=GRAY, style="-", width=0.6)
        if "APSK" in p.name:
            th = np.linspace(0, 2 * np.pi, 200)
            for i, r in enumerate(np.unique(np.round(np.abs(pts), 5))):
                pp.line(f"ring{i}", r * np.cos(th), r * np.sin(th), color=GRAY, width=0.8, style=":")
        i, j, dmin = mz.nearest_neighbours(pts)
        if p.links:
            xs = np.column_stack([pts[i].real, pts[j].real, np.full(len(i), np.nan)]).ravel()
            ys = np.column_stack([pts[i].imag, pts[j].imag, np.full(len(i), np.nan)]).ravel()
            flips = np.array([bin(int(a ^ b)).count("1") for a, b in zip(c.labels[i], c.labels[j])])
            one = np.repeat(flips == 1, 3)
            if one.any():
                pp.line("nn1", np.where(one, xs, np.nan), np.where(one, ys, np.nan), color=GREEN,
                        width=1.6, alpha=0.7)
            if (~one).any():
                pp.line("nn2", np.where(~one, xs, np.nan), np.where(~one, ys, np.nan), color=RED,
                        width=2.2, alpha=0.8)
        size = 11 if c.M <= 16 else 8 if c.M <= 64 else 5
        pp.scatter("pts", pts.real, pts.imag, color=NAVY, size=size, alpha=1.0, z=3)
        if p.show_bits and c.M <= 32:
            fs = 8.5 if c.M <= 16 else 7.0
            for n, (z, lab) in enumerate(zip(pts, c.labels)):
                pp.text(f"lab{n}", z.real, z.imag + 0.07, bitstr(lab, c.k), color=RED,
                        anchor=(0.5, 1.0), size=fs)
        # the zoo map
        pm = self.plot("map")
        groups = {}
        for nm in ZOO:
            cc = const(nm, 2.85 if nm == "16-APSK" else (2.84 if nm == "32-APSK" else None))
            key = (round(mz.papr_db(cc), 2), round(float(db10(cc.dmin ** 2 * cc.k)), 2))
            groups.setdefault(key, []).append(nm)
        xs = [k_[0] for k_ in groups]
        ys = [k_[1] for k_ in groups]
        for n, (k_, names) in enumerate(groups.items()):
            pm.text(f"t{n}", k_[0] + 0.09, k_[1], " = ".join(names), color=GRAY, anchor=(0, 0.5),
                    size=8)
        cur = (mz.papr_db(c), db10(dmin ** 2 * c.k))
        pm.scatter("cur", [cur[0]], [cur[1]], color=RED, size=15, z=5)
        self.readout(k=c.k, d2=dmin ** 2, papr=mz.papr_db(c), flip=mz.nn_bit_flips(c))

    def story(self, p):
        r = self.r
        k, d2, papr, flip = r.get("k", 4), r.get("d2", 0.4), r.get("papr", 0), r.get("flip", 1)
        fam = ("PSK" if "PSK" in p.name and "APSK" not in p.name else "PAM" if "PAM" in p.name
               else "APSK" if "APSK" in p.name else "QAM")
        txt = {
            "PSK": f"All {2 ** k} points sit on one circle, so the peak power equals the average "
                   f"({v(papr, '.2f', 'dB')}). The price: neighbours crowd together as M grows, "
                   f"and every doubling of M costs about 6 dB once past QPSK.",
            "PAM": "One dimension only: levels on a line. Every extra bit needs four times the "
                   "energy for the same spacing: the 6 dB-per-bit exchange rate of amplitude levels.",
            "QAM": f"QAM fills the plane with a grid: two PAM signals on I and Q. Its corners stand "
                   f"{v(papr, '.2f', 'dB')} above the average power, which the amplifier must "
                   f"accommodate, but it packs points more efficiently than PSK.",
            "APSK": f"Rings of PSK, as in DVB-S2. The ring ratio γ = {v(p.ratio)} trades the "
                    f"spacing inside the rings against the gap between them; the peak/average is "
                    f"only {v(papr, '.2f', 'dB')}, a gift to a saturated satellite amplifier.",
        }[fam]
        lab = (good("Every nearest-neighbour pair differs in exactly one bit: Gray labelling.")
               if flip <= 1.001 else
               bad(f"A nearest-neighbour slip costs {flip:.2f} bits on average.") +
               (" No labelling can make a cross constellation fully Gray: count the neighbours of "
                "a point next to a missing corner." if "cross" in p.name and p.labels == "Gray" else ""))
        return ("<h3>Points, distances, labels</h3>"
                f"<p>{txt}</p>"
                f"<p>At unit average energy the closest pair is {v(np.sqrt(d2), '.3f')} apart "
                f"(d²min = {v(d2, '.3f')}). Green links join nearest neighbours whose labels differ "
                f"in one bit, red ones more. {lab}</p>"
                "<p>The right-hand map ranks the whole zoo by distance <i>per bit</i> against the "
                "peak power the amplifier must handle: up and to the left is better.</p>"
                + keybox("d_min sets the error rate at high SNR; the peak/average sets the "
                         "amplifier back-off; the labels decide how many bits one slip costs."))


# =============================================================================== 2. signal space
class SignalSpace(Experiment):
    title = "From waveforms to points"
    blurb = "A correlator bank turns a noisy passband waveform into a point in the plane."
    book = "sec:ch09:signalspace"
    controls = [
        Choice("mod", "Modulation", ["BPSK", "QPSK", "8-PSK", "16-QAM"], "QPSK"),
        IntSlider("fc", "Carrier cycles per symbol", 1, 8, 3,
                  help="Carrier frequency in units of the symbol rate"),
        Slider("esn0", "Es/N0", 0.0, 30.0, 12.0, step=0.5, unit="dB"),
        Slider("win", "Correlator integration window", 0.1, 1.0, 1.0, step=0.01, unit="T",
               help="Integrate over only part of the symbol: less signal, proportionally less noise"),
        Button("again", "New symbols"),
    ]
    plots = [
        Plot("wave", "Six symbols in passband: transmitted (navy) and received (grey)",
             x="time (symbol periods)", y="amplitude", xlim=(0, 6), ylim=(-5.2, 6.2), legend=None),
        Plot("corr", "Inside the correlators (first symbol)", x="time within the symbol (T)",
             y="running integral", xlim=(0, 1), ylim=(-1.6, 1.6), legend="tl", legend_cols=2),
        ConstellationPlot("pts", "Correlator outputs (r₁, r₂) for 2000 symbols", lim=1.75),
    ]
    layout = [["wave", "wave"], ["corr", "pts"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("var", "Noise variance per dimension", "", ".4f"),
        Readout("th", "Theory N0/2 (× window)", "", ".4f"),
        Readout("loss", "SNR lost to the window", "dB", ".2f", good=lambda x: x > -0.1),
        Readout("ser", "Symbol error rate", "", "sci"),
    ]
    challenges = [
        Challenge("Find the integration window that throws away exactly 1 dB of SNR (±0.05 dB).",
                  lambda s: abs(s.r.loss + 1.0) < 0.05,
                  hint="Signal grows with the window, noise power too: the SNR scales as the window."),
        Challenge("Keep 16-QAM below 1 % symbol errors with Es/N0 no higher than 17 dB.",
                  lambda s: s.p.mod == "16-QAM" and s.p.esn0 <= 17 and s.r.ser < 0.01),
        Challenge("Show that the carrier frequency is irrelevant: with one cycle per symbol, keep "
                  "QPSK's SER below 10⁻² at 10 dB.",
                  lambda s: s.p.fc == 1 and s.p.mod == "QPSK" and abs(s.p.esn0 - 10) < 0.3
                  and s.r.ser < 1e-2),
    ]
    ns = 32
    nsym = 2000

    def setup(self):
        self.seed = 3

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        c = const(p.mod)
        a = c.points[rng.integers(0, c.M, self.nsym)]
        ns = self.ns
        t = np.arange(ns) / ns
        phi1 = np.sqrt(2) * np.cos(2 * np.pi * p.fc * t)
        phi2 = -np.sqrt(2) * np.sin(2 * np.pi * p.fc * t)
        wave = a.real[:, None] * phi1 + a.imag[:, None] * phi2
        n0 = 10 ** (-p.esn0 / 10)
        r = wave + np.sqrt(n0 / 2 * ns) * rng.standard_normal(wave.shape)
        m = max(1, int(round(p.win * ns)))
        dt = 1 / ns
        r1 = (r[:, :m] @ phi1[:m]) * dt
        r2 = (r[:, :m] @ phi2[:m]) * dt
        gain = np.sum(phi1[:m] ** 2) * dt                       # ≈ window (signal scale)
        z = (r1 + 1j * r2) / gain
        err = z - a
        var_meas = np.var(err.real) * gain ** 2
        ser = float(np.mean(c.decide(z) != a))
        # plots
        pw = self.plot("wave")
        tt = np.arange(6 * ns) / ns
        pw.line("r", tt, r[:6].ravel(), color=GRAY, width=0.8, name="received r(t)")
        pw.line("s", tt, wave[:6].ravel(), color=NAVY, width=2.0, name="transmitted s(t)")
        for k in range(1, 6):
            pw.vline(f"b{k}", k, color=GRAY, style=":", width=0.7)
        for k in range(6):
            pw.text(f"sym{k}", k + 0.5, 4.4, f"a = {a[k].real:+.2f}{a[k].imag:+.2f}j", color=NAVY,
                    anchor=(0.5, 1.0), size=8.5, fill=True)
        pc = self.plot("corr")
        run1 = np.cumsum(r[0] * phi1) * dt
        run2 = np.cumsum(r[0] * phi2) * dt
        pc.line("i1", t + dt, run1, color=NAVY, width=2.0, name="∫ r·φ₁ (→ I)")
        pc.line("i2", t + dt, run2, color=RED, width=2.0, name="∫ r·φ₂ (→ Q)")
        pc.hline("a1", a[0].real, color=NAVY, style=":", width=1.0)
        pc.hline("a2", a[0].imag, color=RED, style=":", width=1.0)
        pc.vline("w", min(p.win, 0.985), color=ORANGE, style="--", label="sample here", label_pos=0.08)
        pp = self.plot("pts")
        pp.points("z", z, color=NAVY, size=3, alpha=0.35)
        bad_ = c.decide(z) != a
        pp.points("e", z[bad_], color=RED, size=5, alpha=0.9)
        pp.ideal("ideal", c.points, color=RED)
        loss = 10 * np.log10(gain)
        self.readout(var=var_meas, th=n0 / 2 * gain, loss=0.0 if abs(loss) < 0.005 else loss, ser=ser)

    def story(self, p):
        loss = self.r.get("loss", 0)
        s = ("<p>The waveform on top is a mess: a carrier at "
             f"{v(p.fc, 'd')} cycles per symbol, its amplitude and phase set by each symbol, buried "
             "in noise. Multiply it by the two basis functions φ₁ = √2·cos and φ₂ = −√2·sin, "
             "integrate over the symbol (bottom left), and two numbers come out: the in-phase and "
             "quadrature coordinates of a point (bottom right).</p>"
             "<p>Nothing is lost on the way. The noise outside the span of the two basis "
             "functions is irrelevant to the decision (the <b>theorem of irrelevance</b>), and "
             "the noise that remains is Gaussian with variance N0/2 in each direction: compare "
             "the two readouts.</p>")
        if loss < -0.05:
            s += (f"<p>You integrate over only {v(p.win)} T, so you collect that fraction of the "
                  f"signal energy and the noise shrinks less: the SNR drops by "
                  f"{v(-loss, '.2f', 'dB')} and the clouds swell. The matched filter (the full "
                  f"correlator) is optimal.</p>")
        return "<h3>Waveforms become points</h3>" + s + keybox(
            "A correlator bank (equivalently, a matched filter sampled once per symbol) reduces "
            "any waveform channel to points plus Gaussian noise. Everything after this works on points.")


# =============================================================================== 3. decision regions
REGION_SCHEMES = ["BPSK", "QPSK", "8-PSK", "16-QAM", "32-cross QAM", "64-QAM", "16-APSK", "32-APSK"]


class DecisionRegions(Experiment):
    title = "Noise and decision regions"
    blurb = "Pick the nearest point: the ML detector's borders, its errors and the union bound."
    book = "sec:ch09:detection"
    animate = True
    fps = 8
    controls = [
        Choice("name", "Constellation", REGION_SCHEMES, "16-QAM", style="menu"),
        Slider("esn0", "Es/N0", 0.0, 35.0, 14.0, step=0.25, unit="dB"),
        Toggle("regions", "Shade the decision regions", True),
        Toggle("arrows", "Show where errors came from", True),
    ]
    plots = [
        Plot("pts", "Received symbols (red: decided wrongly)", x="in-phase I", y="quadrature Q",
             xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect=True, legend=None, grid=False),
        BERPlot("ser", "Symbol error rate: exact, bounds and your measurement", x="Es/N0 (dB)",
                y="symbol error rate", ylim=(1e-6, 2.0), xlim=(0, 35), legend="bl"),
    ]
    layout = [["pts", "ser"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("meas", "Measured SER", "", "sci"),
        Readout("th", "Exact SER", "", "sci"),
        Readout("ub", "Union bound / exact", "×", ".2f"),
        Readout("n", "Symbols counted", "", "int"),
    ]
    challenges = [
        Challenge("Press Play and count at least 20 000 symbols: land within 20 % of the exact "
                  "SER (with at least 50 errors).",
                  lambda s: s.r.n >= 20000 and s.exp.errs >= 50 and abs(s.r.meas / s.r.th - 1) < 0.2),
        Challenge("Find where the union bound is useless: at least twice the true SER.",
                  lambda s: s.r.ub >= 2.0),
        Challenge("Find the Es/N0 at which 8-PSK makes one symbol error in a thousand (exact SER "
                  "within ±10 % of 10⁻³).",
                  lambda s: s.p.name == "8-PSK" and abs(s.r.th / 1e-3 - 1) < 0.1),
    ]
    nsym = 2500

    def setup(self):
        self.errs = 0
        self.count = 0

    def draw(self, p, accumulate):
        c = const(p.name)
        rng = self.rng
        a_idx = rng.integers(0, c.M, self.nsym)
        a = c.points[a_idx]
        y, _ = cl.awgn_esn0(a, p.esn0, rng=rng, es=1.0)
        wrong = c.decide(y) != a
        if not accumulate:
            self.errs = self.count = 0
        self.errs += int(np.sum(wrong))
        self.count += self.nsym
        pp = self.plot("pts")
        if p.regions:
            pp.image("reg", region_map(p.name, None), x=(-3.2, 3.2), y=(-3.2, 3.2),
                     cmap=region_colors(pp.theme.name), levels=(0, 5))
        pp.scatter("ok", y[~wrong].real, y[~wrong].imag, color=NAVY, size=3, alpha=0.35)
        pp.scatter("bad", y[wrong].real, y[wrong].imag, color=RED, size=6, alpha=0.95, z=4)
        if p.arrows and wrong.any():
            w = np.where(wrong)[0][:150]
            xs = np.column_stack([a[w].real, y[w].real, np.full(len(w), np.nan)]).ravel()
            ys = np.column_stack([a[w].imag, y[w].imag, np.full(len(w), np.nan)]).ravel()
            pp.line("arr", xs, ys, color=RED, width=0.9, alpha=0.55)
        pp.scatter("pts", c.points.real, c.points.imag, color=GREEN if pp.theme.name == "dark"
                   else "#000000", size=7, alpha=1.0, z=5)
        # error-rate curves
        key = mz.NAMES[p.name]
        es = np.linspace(0, 35, 176)
        exact = mz.ser_theory(key, es)
        pb = self.plot("ser")
        pb.theory("ub", es, np.minimum(mz.union_bound_ser(c, es), 2.0), color=ORANGE, style="--",
                  name="full union bound")
        pb.theory("nn", es, np.minimum(mz.nn_approx_ser(c, es), 2.0), color=GREEN, style=":",
                  name="nearest-neighbour approx.")
        exact_name = "exact" if key in ("bpsk", "qpsk", "8psk", "16qam", "64qam") else "nearest-neighbour"
        pb.theory("ex", es, exact, color=NAVY, name=exact_name if key in ("bpsk", "qpsk", "8psk",
                                                                          "16qam", "64qam") else None)
        meas = self.errs / self.count
        if self.errs > 0:
            pb.sim("me", [p.esn0], [meas], color=RED, name="your measurement")
        pb.vline("cur", p.esn0, color=RED, style=":", width=1.0)
        th = float(mz.ser_theory(key, p.esn0)[0])
        ub = float(mz.union_bound_ser(c, p.esn0)[0])
        self.readout(meas=meas if self.errs else 0.0, th=th, ub=ub / max(th, 1e-300), n=self.count)

    def update(self, p):
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        n, e = self.r.get("n", 0), self.errs
        ub = self.r.get("ub", 1)
        s = ("<p>With equally likely symbols and Gaussian noise, the best possible detector "
             "simply picks the <b>nearest</b> constellation point. Its decision regions are the "
             "shaded cells: pie slices for PSK, a chessboard for QAM (so the detector is two "
             "independent slicers), wedges and rings for APSK.</p>"
             f"<p>Each red dot crossed into a neighbour's cell; the thin red lines show where it "
             f"started. So far {v(e, 'd')} errors in {v(n, 'd')} symbols. Press <b>Play</b> to keep "
             f"counting: the red circle on the right settles onto the exact curve.</p>")
        if ub > 1.5:
            s += (f"<p>The union bound adds the probability of crossing <i>each</i> border "
                  f"separately, double-counting the corners: here it overstates the error rate "
                  f"{v(ub, '.1f')}×. At high SNR the corners become irrelevant and all three curves "
                  f"merge.</p>")
        else:
            s += ("<p>At this SNR the union bound and the nearest-neighbour approximation hug the "
                  "exact curve: only the closest border matters, and only d_min and the number of "
                  "neighbours count.</p>")
        return "<h3>Pick the nearest point</h3>" + s + keybox(
            "ML in AWGN = minimum distance = Voronoi cells. P_s ≈ N_min·Q(d_min / √(2N0)).")


# =============================================================================== 4. MAP vs ML
class MapDetection(Experiment):
    title = "Priors move the boundary"
    blurb = "When one symbol is more likely, the best threshold leaves the midpoint."
    book = "sec:ch09:detection"
    controls = [
        Slider("p0", "Probability that −1 is sent", 0.5, 0.99, 0.8, step=0.01),
        Slider("esn0", "Es/N0", -5.0, 15.0, 2.0, step=0.25, unit="dB"),
        Slider("thr", "Your threshold", -1.5, 1.5, 0.0, step=0.005,
               help="Decide +1 when the sample is above this level"),
        Toggle("snap", "Snap my threshold to MAP", False),
    ]
    plots = [
        Plot("pdf", "Weighted likelihoods p·f(y | symbol)", x="received sample y",
             y="density", xlim=(-4, 4), legend="tr"),
        Plot("pe", "Error probability against threshold", x="threshold", y="error probability",
             xlim=(-1.5, 1.5), logy=True, legend="tr"),
    ]
    layout = [["pdf", "pe"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("pe", "Your error rate", "", "sci"),
        Readout("map", "MAP error rate", "", "sci"),
        Readout("ml", "ML (midpoint) error rate", "", "sci"),
        Readout("excess", "Your excess over MAP", "%", ".1f", good=lambda x: x < 1.0),
    ]
    challenges = [
        Challenge("Find the MAP threshold by hand (without snapping): within 1 % of the minimum "
                  "error rate, with p(−1) ≥ 0.8 and Es/N0 ≤ 4 dB.",
                  lambda s: not s.p.snap and s.p.p0 >= 0.8 and s.p.esn0 <= 4 and abs(s.p.thr) > 0.05
                  and s.r.excess < 1.0),
        Challenge("Find a setting where the ML detector makes at least three times as many "
                  "errors as MAP.",
                  lambda s: s.r.ml >= 3 * s.r.map),
        Challenge("Squeeze the MAP shift below 0.05 with p(−1) ≥ 0.9. Does ML now cost nothing?",
                  lambda s: s.p.p0 >= 0.9 and abs(s.exp.t_map) < 0.05),
    ]

    def setup(self):
        self.t_map = 0.0

    def pe(self, t, p0, sig):
        return p0 * cl.qfunc((t + 1) / sig) + (1 - p0) * cl.qfunc((1 - t) / sig)

    def update(self, p):
        sig = np.sqrt(1 / (2 * 10 ** (p.esn0 / 10)))
        p0, p1 = p.p0, 1 - p.p0
        t_map = sig ** 2 / 2 * np.log(p0 / p1)
        self.t_map = t_map
        thr = t_map if p.snap else p.thr
        y = np.linspace(-4, 4, 801)
        f0 = p0 * np.exp(-(y + 1) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)
        f1 = p1 * np.exp(-(y - 1) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)
        pd = self.plot("pdf")
        top = max(f0.max(), f1.max())
        pd.set_ylim(0, top * 1.35)
        e0 = np.where(y > thr, f0, 0)
        e1 = np.where(y < thr, f1, 0)
        pd.fill_between("e0", y, 0 * y, e0, color=NAVY, alpha=0.30)
        pd.fill_between("e1", y, 0 * y, e1, color=RED, alpha=0.30)
        pd.line("f0", y, f0, color=NAVY, width=2.2, name=f"p₀·f(y | −1),  p₀ = {p0:.2f}")
        pd.line("f1", y, f1, color=RED, width=2.2, name=f"p₁·f(y | +1),  p₁ = {p1:.2f}")
        pd.vline("ml", 0, color=GRAY, style=":", label="ML", label_pos=0.95)
        pd.vline("map", t_map, color=GREEN, style="--", label="MAP", label_pos=0.86)
        pd.vline("you", thr, color=ORANGE, style="-", width=2.0, label="yours", label_pos=0.77)
        tt = np.linspace(-1.5, 1.5, 601)
        pe_curve = self.pe(tt, p0, sig)
        pp = self.plot("pe")
        lo = max(pe_curve.min() / 3, 1e-12)
        pp.set_ylim(lo, min(1.0, pe_curve.max() * 3))
        pp.line("pe", tt, pe_curve, color=NAVY, width=2.2)
        pe_you, pe_map, pe_ml = self.pe(thr, p0, sig), self.pe(t_map, p0, sig), self.pe(0.0, p0, sig)
        pp.scatter("ml", [0], [pe_ml], color=GRAY, size=10, name="ML")
        pp.scatter("map", [t_map], [pe_map], color=GREEN, size=12, name="MAP")
        pp.scatter("you", [thr], [pe_you], color=ORANGE, size=10, symbol="d", name="yours")
        self.readout(pe=pe_you, map=pe_map, ml=pe_ml, excess=100 * (pe_you / pe_map - 1))

    def story(self, p):
        t = self.t_map
        s = ("<p>Two symbols, ±1, in Gaussian noise. Each curve is a likelihood weighted by how "
             "often that symbol is sent. The best (MAP) detector decides for whichever weighted "
             "curve is higher, so its threshold sits where they <b>cross</b>; the shaded tails on "
             "the wrong side of your threshold are the errors.</p>")
        if p.p0 > 0.505:
            s += (f"<p>With −1 sent {v(100 * p.p0, '.0f')} % of the time, the crossing moves "
                  f"towards +1, to {v(t, '.3f')}: a sample near zero is more likely a noisy −1 than "
                  f"a rare +1. The shift is σ²/2·ln(p₀/p₁): it shrinks as the SNR rises, but so "
                  f"does σ, and measured in noise standard deviations it does not vanish. That is "
                  f"why ML still costs {v(self.r.get('ml', 1) / max(self.r.get('map', 1), 1e-300), '.2f')}× "
                  f"the MAP error rate here.</p>")
        else:
            s += "<p>With equal priors MAP and ML coincide: the threshold is the midpoint.</p>"
        return "<h3>MAP versus maximum likelihood</h3>" + s + keybox(
            "Real receivers almost always assume equal priors (ML): coded bits are close to "
            "equiprobable, and the decoder, not the detector, uses prior knowledge.")


# =============================================================================== 5. BER curves
BER_SCHEMES = ["BPSK", "QPSK", "8-PSK", "16-PSK", "16-QAM", "64-QAM", "256-QAM", "16-APSK",
               "32-APSK"]


class BERCurves(Experiment):
    title = "BER: Monte Carlo vs theory"
    blurb = "Simulate any constellation live and watch it land on its closed form."
    book = "sec:ch09:errorprob"
    heavy = True
    controls = [
        Choice("name", "Constellation", BER_SCHEMES, "16-QAM", style="menu"),
        Choice("labels", "Bit labelling", ["Gray", "Natural binary"], "Gray"),
        Toggle("family", "Show the other schemes' theory", True),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [BERPlot("ber", "Bit error rate against Eb/N0", x="Eb/N0 (dB)", ylim=(1e-6, 0.5),
                     xlim=(-1, 32), legend="tr")]
    readouts = [
        Readout("req", "Eb/N0 for BER 10⁻⁵ (theory)", "dB", ".2f"),
        Readout("pen", "Measured − theory at 10⁻³", "dB", ".2f"),
        Readout("bits", "Bits simulated", "", "int"),
        Readout("pts", "Points done", "", None),
    ]
    challenges = [
        Challenge("Confirm QPSK = BPSK: simulate Gray QPSK and land within 0.3 dB of the BPSK "
                  "curve at 10⁻³.",
                  lambda s: s.p.name == "QPSK" and s.p.labels == "Gray" and s.r.pen is not None
                  and abs(s.r.pen) < 0.3),
        Challenge("Measure what natural labelling costs 16-QAM at 10⁻³ (more than 0.2 dB).",
                  lambda s: s.p.name == "16-QAM" and s.p.labels == "Natural binary"
                  and s.r.pen is not None and s.r.pen > 0.2),
        Challenge("Find a 16-point constellation that needs at least 3 dB more than 16-QAM at 10⁻⁵.",
                  lambda s: s.p.name in ("16-PSK",) and s.r.req > 13.43 + 3),
    ]

    def setup(self):
        self.run_id = 0
        self.sim = dict(x=[], y=[], bits=0, done=False, cur=None)

    def on_rerun(self, p):
        self.run_id += 1

    def theory(self, name, eb):
        return mz.ber_theory(mz.NAMES[name], eb)

    def update(self, p):
        self.sim = dict(x=[], y=[], bits=0, done=False, cur=None)
        eb = np.linspace(-1, 32, 199)
        pb = self.plot("ber")
        if p.family:
            xs = []
            for i, nm in enumerate(BER_SCHEMES):
                if nm != p.name:
                    pb.theory(f"f{i}", eb, self.theory(nm, eb), color=GRAY, width=1.0,
                              name="other schemes" if not xs else None)
                    if nm == "QPSK" and p.name != "BPSK":
                        xs.append(None)
                        continue
                    xs.append((mz.ebn0_required(mz.NAMES[nm], 1e-5), i,
                               "BPSK = QPSK" if nm == "BPSK" and p.name != "QPSK" else nm))
            pts = sorted(x for x in xs if x is not None)
            for n_, (x5, i, lab) in enumerate(pts):
                yl = 10 ** (-5.25 - 0.3 * (n_ % 3))
                pb.text(f"fl{i}", x5 + 0.25, yl, lab, color=GRAY, size=7.5, anchor=(0, 0.5))
        pb.theory("th", eb, self.theory(p.name, eb), color=NAVY, width=2.4,
                  name=f"{p.name} theory (Gray)")
        pb.hline("e5", 1e-5, color=GRAY, style=":", width=0.8)
        pb.hline("e3", 1e-3, color=GRAY, style=":", width=0.8)
        self.readout(req=mz.ebn0_required(mz.NAMES[p.name], 1e-5), pen=None, bits=0, pts="0")

    def background(self, p):
        c = const(p.name, None, "gray" if p.labels == "Gray" else "natural")
        rng = np.random.default_rng(1000 + self.run_id)
        thorough = p.effort == "Thorough" and not self.quick
        target = 25 if self.quick else (300 if thorough else 100)
        max_bits = 6e4 if self.quick else (4e6 if thorough else 6e5)
        floor = 3e-4 if self.quick else (3e-6 if thorough else 2e-5)
        start = int(np.floor(max(0.0, mz.ebn0_required(mz.NAMES[p.name], 0.1))))
        for eb in np.arange(start, 33, 1.0):
            es = eb + 10 * np.log10(c.k)
            errs = bits = 0
            while errs < target and bits < max_bits:
                b = cl.random_bits(c.k * 20000, rng)
                y, _ = cl.awgn_esn0(c.modulate(b), es, rng=rng, es=1.0)
                errs += int(np.sum(c.demodulate(y) != b))
                bits += len(b)
                yield dict(eb=eb, ber=errs / bits, bits=bits, final=False)
            yield dict(eb=eb, ber=errs / bits, bits=bits, final=True)
            if errs / bits < floor or errs == 0:
                break
        yield dict(done=True)

    def progress(self, p, it):
        sim = self.sim
        if it.get("done"):
            sim["done"] = True
        elif it["final"]:
            sim["x"].append(it["eb"])
            sim["y"].append(max(it["ber"], 1e-9))
            sim["bits"] += it["bits"]
            sim["cur"] = None
        else:
            sim["cur"] = (it["eb"], max(it["ber"], 1e-9), it["bits"])
        pb = self.plot("ber")
        if sim["x"]:
            pb.sim("sim", sim["x"], sim["y"], color=RED, name=f"simulation ({p.labels.lower()})")
        if sim["cur"] is not None:
            pb.scatter("cur", [sim["cur"][0]], [sim["cur"][1]], color=ORANGE, size=12, symbol="d")
        elif pb.item("cur") is not None:
            pb.item("cur").setVisible(False)
        cur_bits = sim["bits"] + (sim["cur"][2] if sim["cur"] else 0)
        self.readout(pen=self.penalty(p), bits=cur_bits,
                     pts=f"{len(sim['x'])}" + (" ✓" if sim["done"] else " …"))

    def penalty(self, p):
        x, y = np.array(self.sim["x"]), np.array(self.sim["y"])
        if len(x) < 2 or y.min() > 1e-3 or y.max() < 1e-3:
            return None
        ly = np.log10(y)
        i = np.where(ly <= -3)[0][0]
        if i == 0:
            return None
        x3 = np.interp(-3, [ly[i], ly[i - 1]], [x[i], x[i - 1]])
        ref = "BPSK" if p.name == "QPSK" else p.name
        return float(x3 - mz.ebn0_required(mz.NAMES[ref], 1e-3))

    def story(self, p):
        pen, req = self.r.get("pen"), self.r.get("req", 0)
        s = ("<p>Each red circle is a real experiment: random bits, the mapper, Gaussian noise "
             "and a minimum-distance detector, run until about 100 errors have been counted. "
             "The orange diamond is the point being simulated now.</p>"
             f"<p>{p.name} needs {v(req, '.1f', 'dB')} of Eb/N0 for one error in 10⁵ bits. "
             "Grey curves are the rest of the zoo: QPSK sits exactly on BPSK (two BPSKs in "
             "quadrature), 16-QAM beats 16-PSK by about 4 dB, and each extra two bits per symbol "
             "costs QAM about 4–5 dB.</p>")
        if pen is not None:
            s += f"<p>Measured minus theory at 10⁻³: {v(pen, '.2f', 'dB')}.</p>"
        if p.labels != "Gray":
            s += ("<p>Natural binary labels: a symbol error between the two middle levels flips "
                  "two bits instead of one, so the BER rises by roughly a third at every SNR, a few "
                  "tenths of a dB. Gray labelling is free, which is why every standard uses it.</p>")
        if "APSK" in p.name:
            s += ("<p>APSK's theory here is the nearest-neighbour approximation (no closed form), "
                  "so expect the simulation to sit slightly off it at low SNR.</p>")
        return "<h3>Error rates you can trust</h3>" + s + keybox(
            "Change any control and the run restarts. Gray labels make BER ≈ SER / k.")


# =============================================================================== 6. LLRs
class SoftBits(Experiment):
    title = "Soft bits: LLRs"
    blurb = "Not just which point, but how sure: the log-likelihood ratio of every bit."
    book = "sec:ch09:soft"
    controls = [
        Choice("name", "Constellation", ["QPSK", "8-PSK", "16-QAM", "64-QAM"], "16-QAM"),
        Slider("esn0", "Es/N0", -5.0, 25.0, 5.0, step=0.25, unit="dB"),
        Heading("The received sample"),
        Slider("yi", "In-phase I", -1.5, 1.5, 0.30, step=0.005),
        Slider("yq", "Quadrature Q", -1.5, 1.5, -0.45, step=0.005),
        IntSlider("bit", "Bit to inspect", 1, 6, 2, help="Bit position in the label (1 = first)"),
    ]
    plots = [
        Plot("map", "LLR of the chosen bit over the plane", x="in-phase I", y="quadrature Q",
             xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect=True, legend=None, grid=False),
        Plot("cut", "LLR along I through the sample (dashed: max-log)",
             x="in-phase I", y="LLR", xlim=(-1.5, 1.5), ylim=(-30, 30), legend="bl", legend_cols=6),
        BarPlot("bars", "This sample: LLR per bit (% = chance it is wrong)",
                x="bit position", y="LLR", ylim=(-30, 30)),
    ]
    layout = [["map", "cut"], ["map", "bars"]]
    col_stretch = [5, 4]
    readouts = [
        Readout("hard", "Hard decision", "", None),
        Readout("llr", "LLR of chosen bit", "", ".2f"),
        Readout("pw", "Chance that bit is wrong", "%", ".1f"),
        Readout("ml", "Max-log error", "", ".2f", good=lambda x: x < 0.5,
                help="|exact − max-log| for the chosen bit, in LLR units"),
    ]
    challenges = [
        Challenge("Reproduce the book's worked example: 16-QAM at 13 dB with the sample at "
                  "I = 0.16, Q = −0.76, and inspect the bit that is only about 96 % sure.",
                  lambda s: s.p.name == "16-QAM" and abs(s.p.esn0 - 13) < 0.3
                  and abs(s.p.yi - 0.158) < 0.02 and abs(s.p.yq + 0.759) < 0.02
                  and 2 < s.r.pw < 7),
        Challenge("Find a sample where one bit is a coin toss (|LLR| < 0.2) and another is "
                  "near-certain (|LLR| > 15).",
                  lambda s: s.exp.minabs < 0.2 and s.exp.maxabs > 15),
        Challenge("Catch max-log out: find a point where it misses the exact LLR by more than 1.",
                  lambda s: s.r.ml > 1.0,
                  hint="Low SNR, near a decision boundary of the chosen bit."),
    ]

    def setup(self):
        self.minabs, self.maxabs = 0.0, 0.0

    @staticmethod
    @lru_cache(maxsize=32)
    def llr_grid(name, esn0, bit, n=121):
        """Exact LLR of one bit on an n×n grid (vectorised log-sum-exp)."""
        c = const(name)
        g = np.linspace(-1.5, 1.5, n)
        X, Y = np.meshgrid(g, g)
        y = (X + 1j * Y).ravel()
        P = c._lut
        m = -np.abs(y[:, None] - P[None, :]) ** 2 * 10 ** (esn0 / 10)
        b = c.bit_matrix[:, bit].astype(bool)
        def lse(a):
            mx = a.max(axis=1, keepdims=True)
            return mx[:, 0] + np.log(np.sum(np.exp(a - mx), axis=1))
        return (lse(m[:, ~b]) - lse(m[:, b])).reshape(X.shape)

    def update(self, p):
        c = const(p.name)
        k = c.k
        bit = min(p.bit, k) - 1
        n0 = 10 ** (-p.esn0 / 10)
        y = np.array([p.yi + 1j * p.yq])
        ex = c.llr(y, n0, exact=True)
        mlg = c.llr(y, n0, exact=False)
        hard = c.bit_matrix[c.nearest(y)[0]]
        # the map
        pm = self.plot("map")
        Lg = self.llr_grid(p.name, round(p.esn0, 2), bit)
        lim = float(np.clip(np.percentile(np.abs(Lg), 90), 2.0, 40.0))
        mid = pm.theme.plot_bg
        cm = ((0, "#C0661A"), (0.5, mid), (1.0, "#1B3A5C")) if pm.theme.name != "dark" else \
            ((0, "#F5A35C"), (0.5, mid), (1.0, "#8AB8E8"))
        pm.image("llr", np.clip(Lg, -lim, lim), x=(-1.5, 1.5), y=(-1.5, 1.5), cmap=cm,
                 levels=(-lim, lim), colorbar=True, cbar_label="LLR")
        b = c.bit_matrix[:, bit]
        # label points: lookup per label index
        P = c._lut
        z0, z1 = P[b == 0], P[b == 1]
        pm.scatter("p0", z0.real, z0.imag, color=NAVY, size=11, alpha=1.0, z=4)
        pm.scatter("p1", z1.real, z1.imag, color=ORANGE, size=11, alpha=1.0, z=4)
        n0p = z0[np.argmin(np.abs(z0 - y[0]))]
        n1p = z1[np.argmin(np.abs(z1 - y[0]))]
        pm.line("to0", [y[0].real, n0p.real], [y[0].imag, n0p.imag], color=NAVY, width=2.0)
        pm.line("to1", [y[0].real, n1p.real], [y[0].imag, n1p.imag], color=ORANGE, width=2.0,
                style="--")
        pm.scatter("y", [p.yi], [p.yq], color=RED, size=16, symbol="star", z=6)
        if c.M <= 16:
            for i, (z, lab) in enumerate(zip(c.points, c.labels)):
                pm.text(f"l{i}", z.real, z.imag + 0.07, bitstr(lab, k), size=7.5,
                        anchor=(0.5, 1.0), fill=True)
        # the cut along I
        xi = np.linspace(-1.5, 1.5, 301)
        yy = xi + 1j * p.yq
        E = c.llr(yy, n0, exact=True).reshape(-1, k)
        Mx = c.llr(yy, n0, exact=False).reshape(-1, k)
        pc = self.plot("cut")
        cols = [NAVY, RED, GREEN, ORANGE, PURPLE, BLUE]
        for j in range(k):
            w = 2.6 if j == bit else 1.2
            pc.line(f"e{j}", xi, np.clip(E[:, j], -60, 60), color=cols[j], width=w, name=f"bit {j + 1}")
            pc.line(f"m{j}", xi, np.clip(Mx[:, j], -60, 60), color=cols[j], width=1.0, style="--")
        pc.hline("z", 0, color=GRAY, style="-", width=0.7)
        pc.vline("y", p.yi, color=RED, style=":", width=1.2)
        ym = float(np.clip(np.max(np.abs(E)), 3, 60))
        pc.set_ylim(-1.35 * ym, 1.1 * ym)
        # bars
        pb = self.plot("bars")
        xb = np.arange(1, k + 1)
        yb = float(np.clip(np.max(np.abs(ex)), 2, 60)) * 1.35
        pb.set_ylim(-yb, yb)
        pb.bars("b", xb, ex, width=0.6,
                colors=[RED if j == bit else (NAVY if ex[j] >= 0 else ORANGE) for j in range(k)])
        pb.hline("z", 0, color=GRAY, style="-", width=0.8)
        for j in range(k):
            pw = 100 / (1 + np.exp(min(abs(ex[j]), 700)))
            pb.text(f"t{j}", j + 1, ex[j], f"{pw:.1f} %" if pw >= 0.05 else "≈0 %",
                    anchor=(0.5, 1.1 if ex[j] >= 0 else -0.1), size=8.5, bold=(j == bit))
        pb.set_xticks([(j + 1, f"b{j + 1}") for j in range(k)])
        pb.set_xlim(0.3, k + 0.7)
        self.minabs = float(np.min(np.abs(ex)))
        self.maxabs = float(np.max(np.abs(ex)))
        L = float(ex[bit])
        self.readout(hard=" ".join(str(int(x)) for x in hard), llr=L,
                     pw=100 / (1 + np.exp(min(abs(L), 700))), ml=float(abs(ex[bit] - mlg[bit])))

    def story(self, p):
        L, pw = self.r.get("llr", 0), self.r.get("pw", 50)
        b = min(p.bit, const(p.name).k)
        s = ("<p>A hard detector says only <i>which</i> point is nearest. A soft one says, for each "
             "bit, how much more likely a 0 is than a 1: the <b>log-likelihood ratio</b> "
             "L = ln P(b=0 | y) / P(b=1 | y). Its sign is the decision, its size the confidence.</p>"
             f"<p>Bit {v(b, 'd')} of your sample has L = {v(L, '.2f')}, so it is wrong with "
             f"probability 1/(1 + e^|L|) = {v(pw, '.1f', '%')}. On the map blue means 'probably 0', "
             "orange 'probably 1'. <b>Max-log</b> keeps only the nearest point with a 0 (solid "
             "line) and with a 1 (dashed): L ≈ (d₁² − d₀²)/N0, piecewise linear.</p>")
        if p.esn0 < 6:
            s += "<p>At low SNR the exact LLR rounds max-log's corners near the boundaries.</p>"
        return "<h3>How sure is each bit?</h3>" + s + keybox(
            "Soft decisions are worth about 2 dB to a decoder. Bit positions in a QAM symbol are "
            "not equally reliable: compare the bars.")


# =============================================================================== 7. FSK
class OrthogonalFSK(Experiment):
    title = "Orthogonal FSK, noncoherently"
    blurb = "Trade bandwidth for power with M tones, and detect them without the carrier phase."
    book = "sec:ch09:noncoherent"
    animate = True
    fps = 6
    controls = [
        Choice("M", "Number of tones M", ["2", "4", "8", "16", "32", "64"], "2", style="menu"),
        Choice("det", "Receiver", ["Coherent", "Noncoherent"], "Noncoherent"),
        Slider("ebn0", "Eb/N0", 0.0, 14.0, 8.0, step=0.25, unit="dB"),
        Toggle("refs", "Show BPSK and DPSK references", True),
    ]
    plots = [
        BarPlot("bank", "Correlator bank, one symbol (green: the tone sent)",
                x="tone m", y="correlator output"),
        SpectrumPlot("tones", "The M tones and the bandwidth they need", x="frequency (× 1/T)",
                     y="power (dB)", ylim=(-40, 3), legend=None),
        BERPlot("ber", "Bit error rate", x="Eb/N0 (dB)", ylim=(1e-7, 0.5), xlim=(0, 14), legend="bl"),
    ]
    layout = [["bank", "ber"], ["tones", "ber"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("th", "BER (theory)", "", "sci"),
        Readout("meas", "BER (measured)", "", "sci"),
        Readout("eta", "Spectral efficiency", "b/s/Hz", ".3f"),
        Readout("req", "Eb/N0 for 10⁻⁵", "dB", ".2f"),
    ]
    challenges = [
        Challenge("Beat coherent BPSK (9.6 dB at 10⁻⁵) with a noncoherent receiver, using the "
                  "fewest tones that do it.",
                  lambda s: s.p.det == "Noncoherent" and s.p.M == "8"),
        Challenge("Get the theoretical BER below 10⁻⁵ at an Eb/N0 of 7 dB or less.",
                  lambda s: s.p.ebn0 <= 7 and s.r.th < 1e-5),
        Challenge("Press Play, simulate at least 200 000 bits and land within 10 % of theory.",
                  lambda s: s.exp.bits >= 200000 and s.exp.errs >= 200
                  and abs(s.r.meas / s.r.th - 1) < 0.10,
                  hint="Pick an Eb/N0 where errors are common enough to count."),
    ]

    def setup(self):
        self.errs = 0
        self.bits = 0

    def draw(self, p, acc):
        M = int(p.M)
        k = int(np.log2(M))
        coh = p.det == "Coherent"
        es = k * 10 ** (p.ebn0 / 10)
        rng = self.rng
        n = max(200, 20000 // M)
        sym = rng.integers(0, M, n)
        theta = np.zeros(n) if coh else rng.uniform(0, 2 * np.pi, n)
        R = (rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))) / np.sqrt(2)
        R[np.arange(n), sym] += np.sqrt(es) * np.exp(1j * theta)
        stat = R.real if coh else np.abs(R)
        det = np.argmax(stat, axis=1)
        x = det ^ sym
        nerr = int(np.sum([bin(int(t)).count("1") for t in x[x > 0]]))
        if not acc:
            self.errs = self.bits = 0
        self.errs += nerr
        self.bits += n * k
        # bank for the first symbol
        pb = self.plot("bank")
        m = np.arange(M)
        cols = [GREEN if j == sym[0] else (RED if j == det[0] else NAVY) for j in m]
        top = max(4.0, np.sqrt(es) * 1.6)
        pb.bars("b", m, stat[0], width=0.7, colors=cols)
        pb.set_ylim(-top * 0.6 if coh else 0, top)
        pb.set_xlim(-0.7, M - 0.3)
        pb.set_labels(y="Re{rₘ}" if coh else "|rₘ|")
        pb.hline("z", 0, color=GRAY, style="-", width=0.7)
        verdict = "correct" if det[0] == sym[0] else "WRONG"
        pb.text("v", (M - 1) / 2, None, f"sent tone {sym[0]}, decided {det[0]}: {verdict}",
                color=GREEN if verdict == "correct" else RED, anchor=(0.5, 0), size=9)
        stepk = max(1, M // 16)
        pb.set_xticks([(j, str(j)) for j in range(0, M, stepk)])
        # tones
        sp = 0.5 if coh else 1.0
        f = np.linspace(-1.5, (M - 1) * sp + 1.5, 900)
        pt = self.plot("tones")
        tot = np.zeros_like(f)
        for j in range(min(M, 64)):
            s_ = np.sinc(f - j * sp) ** 2
            tot += s_
            if M <= 16:
                pt.line(f"t{j}", f, 10 * np.log10(s_ + 1e-6), color=GRAY, width=0.9)
        pt.line("tot", f, 10 * np.log10(tot / tot.max() + 1e-6), color=NAVY, width=2.0)
        pt.set_xlim(f[0], f[-1])
        W = M * sp
        pt.band("bw", -sp / 2, (M - 1) * sp + sp / 2, color=GREEN, alpha=0.10)
        # BER
        eb = np.linspace(0, 14, 113)
        pr = self.plot("ber")
        curves = self.fsk_curves(coh)
        for i, mm in enumerate([2, 4, 8, 16, 32, 64]):
            if mm != M:
                pr.theory(f"o{i}", eb, curves[i], color=GRAY, width=0.9,
                          name="other M" if i == (0 if M != 2 else 1) else None)
        pr.theory("th", eb, curves[[2, 4, 8, 16, 32, 64].index(M)], color=NAVY, width=2.4,
                  name=f"{M}-FSK {'coherent' if coh else 'noncoherent'}")
        if p.refs:
            pr.theory("bpsk", eb, cl.ber_bpsk(eb), color=GREEN, style="--", name="coherent BPSK")
            pr.theory("dpsk", eb, mz.ber_dbpsk(eb), color=ORANGE, style="--", name="DBPSK")
        meas = self.errs / self.bits
        if self.errs:
            pr.sim("me", [p.ebn0], [max(meas, 1e-9)], color=RED, name="measured")
        pr.vline("cur", p.ebn0, color=RED, style=":", width=1.0)
        th = float(mz.ber_orth(p.ebn0, M, coh)[0])
        self.readout(th=th, meas=meas if self.errs else 0.0, eta=k / W,
                     req=self.req(M, coh))

    @staticmethod
    @lru_cache(maxsize=4)
    def fsk_curves(coh):
        eb = np.linspace(0, 14, 113)
        return [mz.ber_orth(eb, m, coh) for m in (2, 4, 8, 16, 32, 64)]

    @staticmethod
    @lru_cache(maxsize=16)
    def req(M, coh):
        return mz.ebn0_required(None, 1e-5, lambda e: mz.ber_orth(e, M, coh))

    def update(self, p):
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        M = int(p.M)
        coh = p.det == "Coherent"
        req, eta = self.r.get("req", 0), self.r.get("eta", 0)
        s = (f"<p>Each symbol is one of {v(M, 'd')} tones; the receiver correlates against all of "
             f"them and picks the strongest (top left, one symbol per frame while playing). "
             f"Orthogonal signals never interfere, so adding tones does not crowd anybody: it "
             f"<i>lowers</i> the Eb/N0 needed, here {v(req, '.1f', 'dB')} for 10⁻⁵. The price is "
             f"bandwidth: {v(eta, '.3f')} bit/s per Hz (bottom left).</p>")
        if coh:
            s += ("<p>The coherent receiver knows each tone's phase, so tones may be as close as "
                  "1/(2T). As M → ∞ it approaches the Shannon limit of −1.6 dB.</p>")
        else:
            s += ("<p>The <b>noncoherent</b> receiver compares only the energies |rₘ|², never the "
                  "phase: no carrier recovery at all, at a cost of about 1 dB and tone spacing 1/T. "
                  "This is the power-hungry-for-bandwidth corner exploited by pagers, deep-space "
                  "links and (with chirps) LoRa.</p>")
        return "<h3>More tones, less power</h3>" + s + keybox(
            "Binary FSK is 3 dB behind BPSK (orthogonal, not antipodal); DPSK, the noncoherent "
            "cousin of BPSK, is within 1 dB of it.")


# =============================================================================== 8. efficiency plane
PLANE = (["BPSK", "QPSK", "8-PSK", "16-PSK", "32-PSK", "16-QAM", "64-QAM", "256-QAM",
          "1024-QAM", "4096-QAM"]
         + [f"{m}-FSK" for m in (2, 4, 8, 16, 32, 64, 256, 1024)])


@lru_cache(maxsize=64)
def plane_point(name, target):
    """(Eb/N0 required, bits per symbol, eta at beta=0) for a scheme at BER ``target``."""
    if name.endswith("FSK"):
        M = int(name.split("-")[0])
        k = np.log2(M)
        return mz.ebn0_required(None, target, lambda e: mz.ber_orth(e, M, True)), k, 2 * k / M
    if name in ("BPSK", "QPSK"):
        return mz.ebn0_required("bpsk", target), (1 if name == "BPSK" else 2), \
            (1.0 if name == "BPSK" else 2.0)
    M = int(name.split("-")[0])
    k = np.log2(M)
    if "PSK" in name:
        f = lambda e: mz.ser_psk_exact(e + 10 * np.log10(k), M) / k
    else:
        f = lambda e: cl.ber_mqam_gray(e + 10 * np.log10(k), M)
    return mz.ebn0_required(None, target, f), k, k


class EfficiencyPlane(Experiment):
    title = "The bandwidth–power plane"
    blurb = "Every modulation is a point; Shannon draws the border of the possible."
    book = "sec:ch09:tradeoff"
    controls = [
        Choice("name", "Modulation", PLANE, "16-QAM", style="menu"),
        Slider("beta", "Roll-off β", 0.0, 1.0, 0.0, step=0.01,
               help="Bandwidth W = Rs(1 + β); FSK ignores it"),
        LogSlider("target", "Target bit error rate", 1e-7, 1e-2, 1e-5),
        Slider("gain", "Coding gain", 0.0, 10.0, 0.0, step=0.1, unit="dB",
               help="Moves the point left by the gain of an error-correcting code (rate ignored)"),
        Heading("A real link"),
        Toggle("link", "Overlay a link budget", False),
        Slider("cn0", "C/N0", 60.0, 110.0, 90.0, step=0.5, unit="dB-Hz",
               enabled_if=lambda p: p.link),
        LogSlider("bw", "Channel bandwidth", 0.1, 500.0, 36.0, unit="MHz",
                  enabled_if=lambda p: p.link),
    ]
    plots = [Plot("plane", "Spectral efficiency against the Eb/N0 it costs",
                  x="Eb/N0 needed (dB)", y="spectral efficiency (b/s/Hz)", xlim=(-3, 40),
                  ylim=(0.015, 16), logy=True, legend="br")]
    readouts = [
        Readout("eta", "Spectral efficiency", "b/s/Hz", ".3f"),
        Readout("req", "Eb/N0 needed", "dB", ".2f"),
        Readout("gap", "Gap to Shannon", "dB", ".2f", good=lambda x: x < 3),
        Readout("rate", "Link throughput", "Mb/s", ".1f"),
    ]
    challenges = [
        Challenge("The book's transponder: 36 MHz, C/N0 = 90 dB-Hz, β = 0.2. Choose the fastest "
                  "uncoded PSK/QAM that closes the link.",
                  lambda s: s.p.link and abs(s.p.cn0 - 90) < 0.6 and abs(s.p.bw - 36) < 1.5
                  and abs(s.p.beta - 0.2) < 0.015 and s.p.gain == 0 and s.p.name == "QPSK"
                  and s.r.rate > 0),
        Challenge("Get within 2 dB of Shannon (a modern code is needed).",
                  lambda s: s.r.gap < 2.0),
        Challenge("Find an uncoded scheme in the power-limited region (below 1 b/s/Hz) that needs "
                  "less Eb/N0 than QPSK at the same target.",
                  lambda s: s.r.eta < 1 and s.p.gain == 0 and s.r.req < s.exp.qpsk_req - 0.05),
    ]

    def setup(self):
        self.qpsk_req = 9.6

    def update(self, p):
        tgt = float(10 ** round(np.log10(p.target), 2))
        pl = self.plot("plane")
        eta = np.geomspace(0.012, 16, 300)
        pl.line("sh", mz.shannon_ebn0_db(eta), eta, color=NAVY, width=2.4, name="Shannon limit")
        pl.vline("ln2", 10 * np.log10(np.log(2)), color=NAVY, style=":", label="−1.59 dB",
                 label_pos=0.05)
        pl.hband("bwl", 1, 16, color=GREEN, alpha=0.05)
        pl.text("bwlab", 28, 1.25, "bandwidth-limited", color=GREEN, size=8.5, anchor=(0, 1))
        pl.text("pwlab", 14, 0.03, "power-limited", color=ORANGE, size=8.5, anchor=(0, 0))
        pl.text("un", 1.0, 6.5, "unattainable", color=NAVY, size=9)
        fam = {"PSK": ([], [], BLUE, "o"), "QAM": ([], [], GREEN, "s"), "FSK": ([], [], ORANGE, "t")}
        for n_ in PLANE:
            req, k, e0 = plane_point(n_, tgt)
            e = e0 if n_.endswith("FSK") else e0 / (1 + p.beta)
            key = "FSK" if n_.endswith("FSK") else ("QAM" if "QAM" in n_ else "PSK")
            fam[key][0].append(req)
            fam[key][1].append(e)
            lab = n_.split("-")[0] if "-" in n_ else ("2" if n_ == "BPSK" else "4")
            if key == "FSK":
                pl.text(f"lab{n_}", req - 0.4, e * (0.86 if lab in ("2", "4") else 1.0), lab,
                        color=fam[key][2], size=7.5, anchor=(1, 0.5 if lab not in ("2", "4") else 0))
            else:
                pl.text(f"lab{n_}", req + 0.4, e, lab, color=fam[key][2], size=7.5, anchor=(0, 0.5))
        for key, (xs, ys, col, sym) in fam.items():
            name = {"PSK": "M-PSK", "QAM": "M-QAM", "FSK": "coherent M-FSK"}[key]
            pl.line(f"f{key}", xs, ys, color=col, width=0.9)
            pl.scatter(f"s{key}", xs, ys, color=col, size=7, symbol=sym, name=name)
        req, k, e0 = plane_point(p.name, tgt)
        eta_cur = e0 if p.name.endswith("FSK") else e0 / (1 + p.beta)
        x_cur = req - p.gain
        sh = float(mz.shannon_ebn0_db(eta_cur))
        pl.line("gapline", [sh, x_cur], [eta_cur, eta_cur], color=RED, width=1.6, style="--")
        if p.gain > 0:
            pl.line("gainline", [req, x_cur], [eta_cur, eta_cur], color=PURPLE, width=2.0)
            pl.scatter("uncoded", [req], [eta_cur], color=PURPLE, size=9, outline=PURPLE)
        pl.scatter("cur", [x_cur], [eta_cur], color=RED, size=16, z=6, name="your choice")
        self.qpsk_req = plane_point("QPSK", tgt)[0]
        rate = None
        if p.link:
            avail = p.cn0 - 10 * np.log10(eta * p.bw * 1e6)
            pl.line("link", avail, eta, color=PURPLE, width=2.0, style="-",
                    name=f"link: {p.cn0:.0f} dB-Hz in {p.bw:.0f} MHz")
            ok = x_cur <= p.cn0 - 10 * np.log10(eta_cur * p.bw * 1e6)
            rate = eta_cur * p.bw if ok else 0.0
        self.readout(eta=eta_cur, req=x_cur, gap=x_cur - sh, rate=rate if p.link else "—")

    def story(self, p):
        eta, req, gap = self.r.get("eta", 1), self.r.get("req", 0), self.r.get("gap", 0)
        region = "bandwidth-limited" if eta >= 1 else "power-limited"
        s = (f"<p>{p.name} delivers {v(eta, '.2f')} b/s per Hz and needs {v(req, '.1f', 'dB')} of "
             f"Eb/N0 for your target error rate: a point in the {region} region. Shannon's limit "
             f"(navy) says no scheme, however clever, can operate to its left; you are "
             f"{v(gap, '.1f', 'dB')} away (red dashes).</p>"
             "<p>PSK and QAM climb upwards at a cost of power; orthogonal FSK walks left towards "
             "−1.59 dB at a cost of bandwidth. Uncoded points all sit 7–9 dB from the curve: "
             "that gap is the territory error-correcting codes reclaim (try the coding-gain "
             "slider: LDPC and polar codes get within 1–2 dB).</p>")
        if p.link:
            rate = self.r.get("rate", 0)
            s += (f"<p>The purple curve is your link: Eb/N0 available = C/N0 − 10·log₁₀(η·W). "
                  f"Points to its left close the link. "
                  + (good(f"Yours does: {rate:.1f} Mb/s.") if rate else bad("Yours does not.")) +
                  "</p>")
        return "<h3>Two currencies: hertz and joules</h3>" + s + keybox(
            "η &lt; log₂(1 + η·Eb/N0)  ⇔  Eb/N0 &gt; (2^η − 1)/η. At η → 0 the limit is ln 2 = −1.59 dB.")


# =============================================================================== 9. fading
class Fading(Experiment):
    title = "Rayleigh fading and diversity"
    blurb = "Deep fades turn the waterfall into a slope; diversity bends it back."
    book = "sec:ch09:fading"
    animate = True
    autoplay = True
    fps = 10
    controls = [
        Choice("mod", "Modulation", ["BPSK", "16-QAM"], "BPSK"),
        Slider("ebn0", "Average Eb/N0 per branch", 0.0, 40.0, 15.0, step=0.5, unit="dB"),
        Choice("L", "Receive branches (MRC)", ["1", "2", "4"], "1"),
        LogSlider("fd", "Fading rate (Doppler × T)", 0.0005, 0.02, 0.003),
        Toggle("persist", "Keep counting while playing", True),
    ]
    plots = [
        Plot("fade", "Channel power seen by the detector (thin: each branch, thick: combined)",
             x="time (symbols)", y="|h|² (dB re mean)", ylim=(-38, 12), legend="bl", legend_cols=6),
        ConstellationPlot("const", "Equalised symbols (red: errors)", lim=1.8),
        BERPlot("ber", "Bit error rate", x="average Eb/N0 per branch (dB)", ylim=(1e-6, 0.5),
                xlim=(0, 40), legend="tr"),
    ]
    layout = [["fade", "fade"], ["const", "ber"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("fad", "BER in fading (theory)", "", "sci"),
        Readout("awgn", "BER without fading", "", "sci"),
        Readout("meas", "BER measured", "", "sci"),
        Readout("deep", "Time > 10 dB below mean", "%", ".2f", good=lambda x: x < 1),
    ]
    challenges = [
        Challenge("Reach BER 10⁻³ in fading with no more than 12 dB per branch.",
                  lambda s: s.p.ebn0 <= 12 and s.r.fad <= 1e-3),
        Challenge("With one branch and BPSK, find the average Eb/N0 that gives BER 10⁻³ "
                  "(theory within ±25 %).",
                  lambda s: s.p.L == "1" and s.p.mod == "BPSK" and abs(s.r.fad / 1e-3 - 1) < 0.25),
        Challenge("Make deep fades rare: the combined channel more than 10 dB below its mean "
                  "less than 0.5 % of the time (measured over at least 20 000 symbols).",
                  lambda s: s.exp.count >= 20000 and s.r.deep < 0.5),
    ]
    win = 3000
    step = 250
    total = 1 << 15

    def setup(self):
        self.pos = 0
        self.errs = self.bits = 0
        self.count = self.deep = 0

    @staticmethod
    @lru_cache(maxsize=8)
    def fading(fd, n, seed):
        rng = np.random.default_rng(seed)
        return np.array([cl.jakes_process(n, fd, rng=rng) for _ in range(4)])

    def draw(self, p, acc):
        L = int(p.L)
        H = self.fading(round(p.fd, 5), self.total, 4)[:L]
        if not acc:
            self.pos = 0
            self.errs = self.bits = self.count = self.deep = 0
        else:
            self.pos = (self.pos + self.step) % (self.total - self.win)
        sl = slice(self.pos, self.pos + self.win)
        h = H[:, sl]
        c = const(p.mod)
        rng = self.rng
        b = cl.random_bits(c.k * self.win, rng)
        a = c.modulate(b)
        es = 10 ** ((p.ebn0 + 10 * np.log10(c.k)) / 10)
        n0 = 1 / es
        noise = np.sqrt(n0 / 2) * (rng.standard_normal((L, self.win)) + 1j * rng.standard_normal((L, self.win)))
        y = h * a[None, :] + noise
        g = np.sum(np.abs(h) ** 2, axis=0)
        z = np.sum(np.conj(h) * y, axis=0) / g          # MRC then equalise
        bh = c.demodulate(z)
        new = slice(self.win - self.step, self.win) if acc else slice(0, self.win)
        bb = b.reshape(-1, c.k)
        bhh = bh.reshape(-1, c.k)
        self.errs += int(np.sum(bhh[new] != bb[new]))
        self.bits += bb[new].size
        gn = g / L
        self.deep += int(np.sum(gn[new] < 0.1))
        self.count += len(gn[new])
        t = np.arange(self.pos, self.pos + self.win)
        pf = self.plot("fade")
        cols = [GRAY, BLUE, PURPLE, TEAL]
        for i in range(L):
            pf.line(f"b{i}", t, 10 * np.log10(np.abs(h[i]) ** 2 + 1e-6), color=cols[i], width=0.9,
                    alpha=0.8, name=f"branch {i + 1}" if L > 1 else None)
        pf.line("g", t, 10 * np.log10(gn + 1e-6), color=NAVY, width=2.2,
                name="combined (MRC)" if L > 1 else "channel")
        pf.hline("deep", -10, color=RED, style="--", label="10 dB fade", label_pos=0.02)
        wrong = np.any(bhh != bb, axis=1)
        if wrong.any():
            tw = t[wrong]
            pf.scatter("err", tw, np.full(len(tw), 8.0), color=RED, size=6, symbol="t",
                       name="symbol errors")
        pf.set_xlim(t[0], t[-1])
        pc = self.plot("const")
        pc.points("ok", z[~wrong], color=NAVY, size=3, alpha=0.35)
        pc.points("bad", z[wrong], color=RED, size=5, alpha=0.9)
        pc.ideal("ideal", c.points, color=GREEN)
        # theory
        eb = np.linspace(0, 40, 161)
        fn = (lambda e: cl.ber_bpsk(e)) if p.mod == "BPSK" else (lambda e: mz.ber_theory("16qam", e))
        pb = self.plot("ber")
        pb.theory("awgn", eb, fn(eb), color=GRAY, style="--", name="AWGN")
        curves = self.curves(p.mod)
        for i, LL in enumerate((1, 2, 4)):
            pb.theory(f"r{LL}", eb, curves[i], color=[RED, ORANGE, GREEN][i],
                      width=2.6 if LL == L else 1.2, name=f"Rayleigh, {LL} branch{'es' if LL > 1 else ''}")
        pb.theory("inv", eb, 1 / (4 * 10 ** (eb / 10)), color=GRAY, style=":", width=0.9,
                  name="1 / (4·Eb/N0)" if p.mod == "BPSK" else None)
        th = float(np.interp(p.ebn0, eb, curves[(1, 2, 4).index(L)]))
        meas = self.errs / max(self.bits, 1)
        if self.errs:
            pb.sim("me", [p.ebn0], [max(meas, 1e-9)], color=RED, name="measured")
        pb.vline("cur", p.ebn0, color=RED, style=":", width=1.0)
        self.readout(fad=th, awgn=float(np.ravel(fn(p.ebn0))[0]), meas=meas if self.errs else 0.0,
                     deep=100 * self.deep / max(self.count, 1))

    @staticmethod
    @lru_cache(maxsize=4)
    def curves(mod):
        eb = np.linspace(0, 40, 161)
        if mod == "BPSK":
            return [mz.ber_rayleigh_mrc(eb, L) for L in (1, 2, 4)]
        fn = lambda e: mz.ber_theory("16qam", e)
        return [mz.ber_fading_mrc(fn, eb, L) for L in (1, 2, 4)]

    def update(self, p):
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, p.persist)

    def story(self, p):
        fad, awgn = self.r.get("fad", 1e-3), self.r.get("awgn", 1e-6)
        L = int(p.L)
        s = ("<p>The top trace is the channel's power gain as the terminal moves: mostly near "
             "its mean, but every so often it plunges 20 or 30 dB into a <b>deep fade</b>. The red "
             "triangles mark symbol errors: they cluster in the fades, and the constellation "
             "bottom-left shows why: when the channel is weak, noise is enormous after "
             "equalisation.</p>"
             f"<p>At {v(p.ebn0, '.1f', 'dB')} the same signal would make "
             f"{v(st.sci(awgn) if awgn > 1e-12 else 'fewer than 10⁻¹²')} errors per bit without fading, "
             f"but {v(st.sci(fad))} with it. The waterfall has become a straight line falling "
             f"only one decade per 10 dB: averaging over fades, the error rate is set by how often "
             f"the channel is weak, not by the noise.</p>")
        if L > 1:
            s += (f"<p>With {v(L, 'd')} independently fading branches combined (maximal-ratio "
                  f"combining), all of them must fade at once to cause an error, and the slope "
                  f"steepens to {v(L, 'd')} decades per 10 dB: <b>diversity order</b> {L}.</p>")
        else:
            s += "<p>Add receive branches and watch the thick line stop diving.</p>"
        return "<h3>When the channel fades</h3>" + s + keybox(
            "In fading, diversity (antennas, frequency, time, multipath) matters more than "
            "Euclidean distance. BPSK at 10⁻³: 6.8 dB in AWGN, 24 dB in Rayleigh, 11 dB with two "
            "branches.")


# =============================================================================== the lab
LAB = st.Lab(2, "Digital Modulation and Optimal Detection", chapter=9,
             chapter_title="Digital Modulation and Optimal Detection",
             experiments=[Gallery, SignalSpace, DecisionRegions, MapDetection, BERCurves,
                          SoftBits, OrthogonalFSK, EfficiencyPlane, Fading])

if __name__ == "__main__":
    st.run(LAB)
