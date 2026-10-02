"""Lab 26 · The Cellular Concept: Reuse, Traffic, Random Access, Coverage and Scheduling   (Chapter 20)

Run it:      python labs/lab26_cellular_system.py
Self-test:   python labs/lab26_cellular_system.py --selftest

A cellular network is a machine for reusing a scarce spectrum as many times per square kilometre
as interference allows, and for sharing each reuse among users who arrive at random, contend for
access, fade, and drive from cell to cell. Nine experiments, each a small live simulator: reuse
patterns and SIR maps, a switchboard that obeys Erlang, ALOHA packets colliding on screen, a
backlog that runs away, Wi-Fi's DCF, a RACH storm, Poisson-placed base stations, a scheduler
riding fades, and a handover that ping-pongs. All numbers come from commlib.cellular and
commlib.cellsim, the same code as the Chapter 20 figures.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import numpy as np
from scipy.ndimage import zoom

from commlib import cellular as cel
from commlib import cellsim as cs
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

PALE_STOPS = ((0, "#1B3A5C"), (0.35, "#2E86C1"), (0.6, "#48C9B0"), (0.8, "#F4D03F"), (1.0, "#FDFEFE"))


def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def undb(x):
    return 10 ** (np.asarray(x, float) / 10)


# =============================================================================== 1. reuse and SIR
REUSE = {"N = 1": (1, 0), "N = 3": (1, 1), "N = 4": (2, 0), "N = 7": (2, 1), "N = 9": (3, 0),
         "N = 12": (2, 2), "N = 13": (3, 1), "N = 19": (3, 2), "N = 21": (4, 1)}


class HexReuse(Experiment):
    title = "Clusters, reuse and SIR"
    blurb = "Reuse each channel set every N cells: how far apart is far enough?"
    book = "sec:ch20:cellular"
    controls = [
        Choice("N", "Cluster size", list(REUSE), "N = 7", style="menu"),
        Slider("n", "Path-loss exponent", 2.5, 5.0, 4.0, step=0.05),
        Choice("sec", "Antennas", ["Omni", "3 sectors"]),
        Slider("sigma", "Shadowing σ", 0.0, 12.0, 8.0, step=0.5, unit="dB"),
        Slider("target", "SIR target", -5.0, 25.0, 18.0, step=0.5, unit="dB",
               help="AMPS analog FM voice needed about 18 dB"),
    ]
    plots = [
        ImagePlot("map", "Downlink SIR without shadowing (dB)", x="", y="", aspect=True),
        Plot("cdf", "SIR over the cell's locations", x="downlink SIR (dB)", y="CDF",
             xlim=(-10, 50), ylim=(0, 1.02), legend="tl"),
    ]
    layout = [["map", "cdf"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("dr", "Reuse ratio D/R", "", ".2f"),
        Readout("form", "Edge SIR (formula)", "dB", ".1f"),
        Readout("med", "Median SIR", "dB", ".1f"),
        Readout("cov", "Locations above target", "%", ".0f", good=lambda x: x >= 80),
    ]
    challenges = [
        Challenge("With n = 3.5, find the smallest cluster whose edge-SIR formula reaches 18 dB.",
                  lambda s: abs(s.p.n - 3.5) < 0.03 and s.p.N == "N = 12" and s.p.sec == "Omni",
                  hint="(3N)^(n/2)/6 ≥ 63: try the valid sizes in turn."),
        Challenge("AMPS with 8 dB shadowing and n = 4: meet 18 dB at 80 % of locations with N = 7.",
                  lambda s: (s.p.N == "N = 7" and s.p.sigma >= 8 and abs(s.p.n - 4) < 0.03
                             and s.p.target >= 18 and s.r.cov >= 79.5)),
        Challenge("Reuse 1 like LTE: N = 1 with sectors, and a median SIR above 5 dB.",
                  lambda s: s.p.N == "N = 1" and s.p.sec == "3 sectors" and s.r.med > 5,
                  hint="Faster-decaying path loss isolates cells better."),
    ]

    def setup(self):
        q, r = cel.hex_axial_grid(4)
        x, y = cel.axial_to_xy(q, r)
        self.q, self.r_ = q, r
        self.sites = x + 1j * y
        self.xs = np.linspace(-4.2, 4.2, 96)
        self.ys = np.linspace(-3.6, 3.6, 84)
        self._mk = None
        self._ck = None

    def update(self, p):
        i, j = REUSE[p.N]
        N = cel.cluster_size(i, j)
        lab = cel.reuse_labels(self.q, self.r_, i, j) if N > 1 else np.zeros(len(self.q), int)
        sectors = 3 if p.sec == "3 sectors" else 1
        key = (i, j, round(p.n, 3), sectors)
        if key != self._mk:
            if sectors == 1:
                m = cs.sir_map(self.xs, self.ys, self.sites, lab, p.n, 1)
                self.img = zoom(np.clip(m, -10, 45), 2, order=1)
            else:                            # three sectors cost 3x: a coarser grid
                xs, ys = np.linspace(-4.2, 4.2, 66), np.linspace(-3.6, 3.6, 58)
                m = cs.sir_map(xs, ys, self.sites, lab, p.n, 3)
                self.img = zoom(np.clip(m, -10, 45), 3, order=1)
            self._mk = key
        pm = self.plot("map")
        pm.image("img", np.clip(self.img, -5, 35), x=(self.xs[0], self.xs[-1]),
                 y=(self.ys[0], self.ys[-1]), cmap=PALE_STOPS, levels=(-5, 35), colorbar=True,
                 cbar_label="SIR (dB)")
        vis = (np.abs(self.sites.real) < 4.6) & (np.abs(self.sites.imag) < 4.0)
        hx, hy = cs.hex_outlines(self.sites[vis])
        pm.line("hex", hx, hy, color=GRAY, width=0.8)
        co = vis & (lab == lab[0])
        pm.scatter("co", self.sites.real[co], self.sites.imag[co], color=RED, size=14,
                   outline=RED, symbol="o")
        pm.scatter("me", [0], [0], color=RED, size=8, symbol="s")
        if N <= 13:
            sel = np.flatnonzero(vis & (np.abs(self.sites.real) < 4.0) & (np.abs(self.sites.imag) < 3.4))
            for k in range(70):
                if k < len(sel):
                    s_ = self.sites[sel[k]]
                    pm.text(f"l{k}", s_.real, s_.imag + 0.45, str(lab[sel[k]] + 1), color=GRAY,
                            size=7.5, anchor=(0.5, 0.5))
        pm.set_xlim(self.xs[0], self.xs[-1])
        pm.set_ylim(self.ys[0], self.ys[-1])
        pm.set_xticks([])
        pm.set_yticks([])
        # CDF with shadowing
        ck = key + (round(p.sigma, 2),)
        if ck != self._ck:
            rng = np.random.default_rng(7)
            self.v_sh = np.sort(cs.sir_samples(i, j, p.n, p.sigma, sectors, 5000, rng))
            self.v_0 = np.sort(cs.sir_samples(i, j, p.n, 0.0, sectors, 5000, np.random.default_rng(8)))
            self._ck = ck
        F = (np.arange(len(self.v_sh)) + 1) / len(self.v_sh)
        F0 = (np.arange(len(self.v_0)) + 1) / len(self.v_0)
        pc = self.plot("cdf")
        pc.line("no", self.v_0, F0, color=GRAY, style="--", width=1.6, name="no shadowing")
        pc.line("sh", self.v_sh, F, color=NAVY, width=2.4, name=f"σ = {p.sigma:.1f} dB")
        Q = np.sqrt(3 * N)
        form = float(db(Q ** p.n / (2 if sectors == 3 else 6)))
        pc.vline("form", form, color=ORANGE, style=":", label="edge formula", label_pos=0.5)
        pc.vline("tgt", p.target, color=RED, style="--", label=f"target {p.target:.0f} dB",
                 label_pos=0.9)
        cov = 100 * float(np.mean(self.v_sh >= p.target))
        self.readout(dr=Q, form=form, med=float(np.median(self.v_sh)), cov=cov)

    def story(self, p):
        r = self.r
        i, j = REUSE[p.N]
        N = cel.cluster_size(i, j)
        s = (f"<p>Move {v(i, 'd')} cells along one axis, turn 60°, move {v(j, 'd')}: that is where "
             f"the same channels are used again. The cluster size is N = i² + ij + j² = "
             f"{v(N, 'd')}, and co-channel cells (red rings) sit D = R·√(3N) = "
             f"{v(r.get('dr', 0), '.2f')} cell radii apart. Each cell gets 1/{N} of the spectrum.</p>"
             f"<p>With six interferers at distance D the edge SIR is about (D/R)<sup>n</sup>/6 = "
             f"{v(r.get('form', 0), '.1f', 'dB')}. The map shows the geometric truth (darkest at "
             f"cell edges); shadowing then spreads it into a distribution (right): "
             f"{v(r.get('cov', 0), '.0f', '%')} of locations reach {v(p.target, '.0f', 'dB')}.</p>")
        if p.sec == "3 sectors":
            s += ("<p>Sectors: each 120° antenna sees only the interferers in front of it (two of "
                  "the six), about 5 dB of SIR, paid for in trunking efficiency (next "
                  "experiment).</p>")
        if N == 1:
            s += ("<p>Reuse 1 looks hopeless for 1980s analog FM, yet LTE and NR do exactly this: "
                  "their coding, link adaptation and HARQ work down to about −5 dB SINR, and using "
                  "every channel in every cell wins.</p>")
        return "<h3>Far enough apart</h3>" + s + keybox(
            "N = i² + ij + j², D/R = √(3N), SIR ≈ (3N)<sup>n/2</sup>/6. Fast-decaying path loss is "
            "a friend of reuse.")


# =============================================================================== 2. Erlang
_CAPC = {}


def capacity_curve(g):
    """(C, carried erlangs per channel at blocking g) for C = 1..150, from an Erlang-B table
    built by the recursion over C on a fine grid of offered traffic (fast and cached)."""
    if g not in _CAPC:
        A = np.logspace(-3, np.log10(300), 4000)
        B = np.ones_like(A)
        Cg = np.arange(1, 151)
        cap = np.empty(len(Cg))
        for c in Cg:
            B = A * B / (c + A * B)
            cap[c - 1] = np.interp(np.log(g), np.log(np.maximum(B, 1e-300)), A)   # B rises with A
        _CAPC[g] = (Cg, cap * (1 - g) / Cg)
    return _CAPC[g]


class Erlang(Experiment):
    title = "Erlang: a live switchboard"
    blurb = "Calls arrive at random and hold a channel; how many channels does a cell need?"
    book = "sec:ch20:traffic"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        IntSlider("C", "Channels", 1, 150, 57),
        LogSlider("A", "Offered traffic", 0.5, 150, 40, unit="E", fmt=".1f",
                  help="Erlangs: call arrival rate × mean holding time"),
        Slider("gos", "Grade of service (blocking)", 0.5, 10.0, 2.0, step=0.1, unit="%"),
        Choice("sec", "Split the channels into sectors", ["1", "3", "6"], "1"),
    ]
    plots = [
        Plot("occ", "Busy channels (live)", x="time (mean holding times)", y="busy channels",
             xlim=(0, 10), legend="tl", legend_cols=3),
        Plot("eb", "Erlang B", x="offered traffic (E)", y="blocking", logx=True, logy=True,
             xlim=(0.5, 200), ylim=(1e-4, 1), legend="tl"),
        Plot("trunk", "Trunking efficiency at your GoS", x="channels", y="carried E per channel",
             xlim=(0, 150), ylim=(0, 1), legend="br"),
        Plot("ec", "Erlang C: if callers wait instead", x="utilisation A/C", y="P(wait)",
             xlim=(0, 1), ylim=(0, 1.02), legend="tl"),
    ]
    layout = [["occ", "occ", "occ"], ["eb", "trunk", "ec"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("bth", "Blocking (Erlang B)", "%", ".2f"),
        Readout("bme", "Measured", "%", ".2f"),
        Readout("cap", "Capacity at your GoS", "E", ".1f"),
        Readout("loss", "Trunking loss of sectors", "%", ".0f"),
    ]
    challenges = [
        Challenge("AMPS: offer a 57-channel cell exactly the traffic it carries at 2 % blocking "
                  "(± 0.5 E).",
                  lambda s: (s.p.C == 57 and abs(s.p.gos - 2) < 0.05
                             and abs(s.p.A - s.r.cap) <= 0.5)),
        Challenge("Price the sectors: split 57 channels three ways and lose more than 20 % of the "
                  "carried traffic at 2 %.",
                  lambda s: (s.p.C == 57 and s.p.sec == "3" and abs(s.p.gos - 2) < 0.05
                             and s.r.loss > 20)),
        Challenge("Queue instead: with 10 channels, find the load at which half of the callers "
                  "must wait (Erlang C 50 % ± 2 %).",
                  lambda s: s.p.C == 10 and abs(s.exp.pw - 0.5) <= 0.02),
    ]
    dt = 0.12

    def reset_sim(self, p):
        rng = self.rng
        k = int(rng.choice(p.C + 1, p=cs.erlang_occupancy(p.A, p.C)))
        self.ends = np.full(p.C, -np.inf)
        self.ends[:k] = rng.exponential(1.0, k)
        self.calls = [(0.0, e) for e in self.ends[:k]]        # (start, end) of carried calls
        self.t = 0.0
        self.next_arr = rng.exponential(1 / p.A)
        self.blk = []
        self.n_arr = self.n_blk = 0

    def step(self, p):
        rng = self.rng
        t1 = self.t + self.dt
        while self.next_arr < t1:
            ta = self.next_arr
            self.n_arr += 1
            free = np.flatnonzero(self.ends <= ta)
            if len(free):
                e = ta + rng.exponential(1.0)
                self.ends[free[0]] = e
                self.calls.append((ta, e))
            else:
                self.n_blk += 1
                self.blk.append(ta)
            self.next_arr = ta + rng.exponential(1 / p.A)
        self.t = t1
        lo = self.t - 10
        self.calls = [c for c in self.calls if c[1] > lo]
        self.blk = [b for b in self.blk if b >= lo]

    def draw(self, p):
        lo = max(0.0, self.t - 10)
        tg = np.linspace(lo, max(self.t, lo + 1e-6), 500)
        if self.calls:
            c = np.array(self.calls)
            busy = np.sum((c[:, 0][None, :] <= tg[:, None]) & (c[:, 1][None, :] > tg[:, None]), axis=1)
        else:
            busy = np.zeros(len(tg))
        po = self.plot("occ")
        po.line("occ", tg - lo, busy, color=NAVY, width=1.8, fill=0, fill_alpha=0.15,
                name="busy channels")
        po.hline("C", p.C, color=RED, style="--", width=1.6, label="all channels busy",
                 label_pos=0.01)
        po.hline("A", p.A, color=GREEN, style=":", width=1.4, label="mean load A", label_pos=0.01)
        if self.blk:
            po.scatter("blk", np.array(self.blk) - lo, np.full(len(self.blk), p.C * 1.06 + 0.5),
                       color=RED, size=9, symbol="x", name="blocked call")
        po.set_ylim(0, p.C * 1.3 + 2)
        bme = 100 * self.n_blk / self.n_arr if self.n_arr else 0.0
        self.readout(bme=bme)
        pe = self.plot("eb")
        pe.scatter("me", [p.A], [max(bme / 100, 1.2e-4)], color=ORANGE, size=12, symbol="d",
                   name="measured")

    def update(self, p):
        self.reset_sim(p)
        g = p.gos / 100
        nsec = int(p.sec)
        A = np.logspace(np.log10(0.5), np.log10(200), 300)
        pe = self.plot("eb")
        Cs = max(1, p.C // nsec)
        if nsec > 1:
            pe.line("sec", A, np.maximum(cel.erlang_b(A, Cs), 1e-9), color=ORANGE, style="--",
                    width=1.4, name=f"C = {Cs} (one sector)")
        pe.line("C", A, np.maximum(cel.erlang_b(A, p.C), 1e-9), color=NAVY, width=2.4,
                name=f"C = {p.C}")
        bth = float(cel.erlang_b(p.A, p.C))
        pe.scatter("th", [p.A], [max(bth, 1.2e-4)], color=RED, size=10, name="your load")
        pe.hline("gos", g, color=GRAY, style=":", label=f"GoS {p.gos:.1f} %", label_pos=0.02)
        cap = cel.erlang_b_capacity(p.C, g)
        cap_sec = nsec * cel.erlang_b_capacity(Cs, g)
        pt = self.plot("trunk")
        self._eff = capacity_curve(round(g, 5))
        self._eff_key = (round(g, 4),)
        pt.line("eff", *self._eff, color=NAVY, width=2.2, name="carried / channels")
        pt.scatter("me", [p.C], [cap * (1 - g) / p.C], color=RED, size=11, name="one trunk")
        if nsec > 1:
            pt.scatter("sec", [Cs], [cel.erlang_b_capacity(Cs, g) * (1 - g) / Cs], color=ORANGE,
                       size=11, symbol="d", name="each sector")
        pc = self.plot("ec")
        rho = np.linspace(0.005, 0.995, 200)
        for c, col in ((1, GRAY), (10, BLUE)):
            if c != p.C:
                pc.line(f"c{c}", rho, cel.erlang_c(rho * c, c), color=col, width=1.2, style="--",
                        name=f"C = {c}")
        pc.line("cur", rho, cel.erlang_c(rho * p.C, p.C), color=NAVY, width=2.4, name=f"C = {p.C}")
        self.pw = float(cel.erlang_c(p.A, p.C)) if p.A < p.C else 1.0
        if p.A < p.C:
            pc.scatter("pt", [p.A / p.C], [self.pw], color=RED, size=11)
        self.cap, self.cap_sec = cap, cap_sec
        self.readout(bth=100 * bth, cap=cap,
                     loss=100 * (1 - cap_sec / cap) if nsec > 1 else "—")
        self.draw(p)

    def tick(self, p):
        self.step(p)
        self.draw(p)
        # untouched plots would be hidden: redraw the static ones too
        self._redraw_static(p)

    def _redraw_static(self, p):
        g = p.gos / 100
        nsec = int(p.sec)
        A = np.logspace(np.log10(0.5), np.log10(200), 300)
        Cs = max(1, p.C // nsec)
        pe = self.plot("eb")
        if nsec > 1:
            pe.line("sec", A, np.maximum(cel.erlang_b(A, Cs), 1e-9), color=ORANGE, style="--",
                    width=1.4, name=f"C = {Cs} (one sector)")
        pe.line("C", A, np.maximum(cel.erlang_b(A, p.C), 1e-9), color=NAVY, width=2.4,
                name=f"C = {p.C}")
        bth = float(cel.erlang_b(p.A, p.C))
        pe.scatter("th", [p.A], [max(bth, 1.2e-4)], color=RED, size=10, name="your load")
        pe.hline("gos", g, color=GRAY, style=":", label=f"GoS {p.gos:.1f} %", label_pos=0.02)
        pt = self.plot("trunk")
        self._eff = capacity_curve(round(g, 5))
        pt.line("eff", *self._eff, color=NAVY, width=2.2, name="carried / channels")
        pt.scatter("me", [p.C], [self.cap * (1 - g) / p.C], color=RED, size=11, name="one trunk")
        if nsec > 1:
            pt.scatter("sec", [Cs], [cel.erlang_b_capacity(Cs, g) * (1 - g) / Cs], color=ORANGE,
                       size=11, symbol="d", name="each sector")
        pc = self.plot("ec")
        rho = np.linspace(0.005, 0.995, 200)
        for c, col in ((1, GRAY), (10, BLUE)):
            if c != p.C:
                pc.line(f"c{c}", rho, cel.erlang_c(rho * c, c), color=col, width=1.2, style="--",
                        name=f"C = {c}")
        pc.line("cur", rho, cel.erlang_c(rho * p.C, p.C), color=NAVY, width=2.4, name=f"C = {p.C}")
        if p.A < p.C:
            pc.scatter("pt", [p.A / p.C], [self.pw], color=RED, size=11)

    def story(self, p):
        r = self.r
        s = (f"<p>Calls arrive at random (Poisson) and each holds a channel for an exponential time. "
             f"With {v(p.A, '.1f', 'erlangs')} offered to {v(p.C, 'd')} channels the busy count "
             f"(top) wanders around {v(p.A, '.0f')}; whenever it hits the red line an arriving call "
             f"is lost (✕). <b>Erlang B</b> predicts {v(r.get('bth', 0), '.2f', '%')} blocking; the "
             f"live switchboard has measured {v(r.get('bme', 0), '.2f', '%')} so far.</p>"
             f"<p>At a {v(p.gos, '.1f', '%')} grade of service these channels carry "
             f"{v(r.get('cap', 0), '.1f', 'E')}. Big trunks are efficient (bottom middle): ")
        if p.sec != "1":
            s += (f"split into {v(int(p.sec), 'd')} sectors the same channels carry "
                  f"{v(self.cap_sec, '.1f', 'E')}, a {v(r.get('loss', 0), '.0f', '%')} trunking "
                  f"loss. Sectors pay off only if their SIR gain buys a tighter reuse.</p>")
        else:
            s += "one trunk is as good as it gets.</p>"
        s += ("<p>If callers wait instead of being blocked, <b>Erlang C</b> gives the chance of "
              "waiting (bottom right): it climbs steeply as A approaches C.</p>")
        return "<h3>Agner Erlang's switchboard</h3>" + s + keybox(
            "B(A, C) = (A<sup>C</sup>/C!) / Σ A<sup>k</sup>/k!. Book: 57 AMPS channels carry 46.8 E "
            "at 2 %; three sectors of 19 only 37.0 E.")


# =============================================================================== 3. ALOHA live
class AlohaLive(Experiment):
    title = "ALOHA and CSMA, live"
    blurb = "Send whenever you like and hope: watch packets collide and the throughput peak."
    book = "sec:ch20:random"
    animate = True
    autoplay = True
    fps = 3
    controls = [
        Choice("proto", "Protocol", ["Pure ALOHA", "Slotted ALOHA"]),
        LogSlider("G", "Offered load G", 0.05, 10, 0.3, unit="pkt/T", fmt=".2f",
                  help="Transmission attempts per packet time"),
        LogSlider("a", "CSMA propagation delay a", 0.001, 1.0, 0.01, fmt=".3f",
                  help="Propagation delay as a fraction of the packet time (for the CSMA curves)"),
    ]
    plots = [
        Plot("tl", "Forty packet times on the channel (each row one station)",
             x="time (packet times)", y="station", xlim=(0, 40), ylim=(-0.8, 12.8), legend="tr",
             legend_cols=2),
        Plot("s", "Throughput versus offered load", x="offered load G", y="throughput S",
             logx=True, xlim=(0.01, 100), ylim=(0, 1.0), legend="tl"),
    ]
    layout = [["tl"], ["s"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("th", "Throughput S (theory)", "", ".3f"),
        Readout("me", "Measured", "", ".3f"),
        Readout("ok", "Packets that get through", "%", ".1f"),
        Readout("csma", "Nonpersistent CSMA peak", "", ".3f"),
    ]
    challenges = [
        Challenge("Tune slotted ALOHA to its peak: theoretical throughput within 1 % of 1/e.",
                  lambda s: s.p.proto == "Slotted ALOHA" and s.r.th >= 0.99 / np.e),
        Challenge("Overload pure ALOHA until fewer than 1 packet in 50 gets through.",
                  lambda s: s.p.proto == "Pure ALOHA" and s.r.ok < 2.0),
        Challenge("Find the propagation delay at which carrier sensing is no better than slotted "
                  "ALOHA (CSMA peak within 0.02 of 1/e).",
                  lambda s: abs(s.r.csma - 1 / np.e) <= 0.02,
                  hint="Listening only helps if you hear the others in time."),
    ]

    def setup(self):
        self.acc_s = 0
        self.acc_t = 0.0

    def frame_events(self, p):
        slotted = p.proto == "Slotted ALOHA"
        t, ok = cs.aloha_events(p.G, 40, slotted, self.rng)
        lane = self.rng.integers(0, 13, len(t))
        return t, ok, lane

    def draw(self, p, accumulate):
        t, ok, lane = self.frame_events(p)
        if not accumulate:
            self.acc_s, self.acc_t = 0, 0.0
        self.acc_s += int(ok.sum())
        self.acc_t += 40.0
        ptl = self.plot("tl")
        for key, m, col, nm in (("ok", ok, GREEN, "got through"), ("bad", ~ok, RED, "collided")):
            if not m.any():
                continue
            xs = np.c_[t[m], t[m] + 1, np.full(m.sum(), np.nan)].ravel()
            ys = np.c_[lane[m], lane[m], np.full(m.sum(), np.nan)].ravel()
            ptl.line(key, xs if len(xs) else [np.nan], ys if len(ys) else [np.nan], color=col,
                     width=7, name=nm)
        if p.proto == "Slotted ALOHA":
            ptl.line("slots", np.repeat(np.arange(0, 41), 3)[:-1],
                     np.tile([-0.8, 12.8, np.nan], 41)[:-1], color=GRAY, width=0.5, alpha=0.6)
        G = np.logspace(-2, 2, 400)
        ps = self.plot("s")
        ps.line("pure", G, cel.aloha_throughput(G), color=GRAY, width=1.8, name="pure ALOHA")
        ps.line("slot", G, cel.aloha_throughput(G, True), color=NAVY, width=1.8, name="slotted ALOHA")
        ps.line("np", G, cel.csma_throughput(G, p.a, "nonpersistent"), color=GREEN, width=1.6,
                name=f"nonpersistent CSMA, a = {p.a:.3f}")
        ps.line("1p", G, cel.csma_throughput(G, p.a, "1-persistent"), color=ORANGE, width=1.4,
                style="--", name=f"1-persistent CSMA, a = {p.a:.3f}")
        th = float(cel.aloha_throughput(p.G, p.proto == "Slotted ALOHA"))
        me = self.acc_s / self.acc_t
        ps.scatter("th", [p.G], [th], color=NAVY, size=11, name="your load (theory)")
        ps.scatter("me", [p.G], [me], color=RED, size=13, symbol="d", name="measured (live)")
        peak = float(np.nanmax(cel.csma_throughput(np.logspace(-2, 3, 2000), p.a, "nonpersistent")))
        self.readout(th=th, me=me, ok=100 * th / p.G, csma=peak)

    def update(self, p):
        self.draw(p, False)

    def tick(self, p):
        self.draw(p, True)

    def story(self, p):
        r = self.r
        slotted = p.proto == "Slotted ALOHA"
        s = (f"<p>Each bar is a packet: stations transmit whenever they have something to send, "
             f"{v(p.G, '.2f')} attempts per packet time on average. A packet survives only if nobody "
             f"else's overlaps it (green); otherwise both are lost (red). ")
        if slotted:
            s += ("With <b>slots</b> (grey lines) packets either coincide exactly or not at all, "
                  "halving the vulnerable period: S = G·e<sup>−G</sup>, peaking at 1/e = 0.368.</p>")
        else:
            s += ("In <b>pure ALOHA</b> a packet is vulnerable for two packet times: "
                  "S = G·e<sup>−2G</sup>, peaking at 1/(2e) = 0.184 when G = 0.5.</p>")
        s += (f"<p>Theory {v(r.get('th', 0), '.3f')}, measured {v(r.get('me', 0), '.3f')}: only "
              f"{v(r.get('ok', 0), '.1f', '%')} of packets get through. Push G past the peak and "
              f"throughput collapses: more trying, less success.</p>"
              f"<p><b>Listening first</b> (CSMA, green/orange) does far better while the "
              f"propagation delay is a small fraction of a packet: peak "
              f"{v(r.get('csma', 0), '.2f')} at a = {v(p.a, '.3f')}.</p>")
        return "<h3>Talk whenever you like</h3>" + s + keybox(
            "ALOHAnet (Abramson, Hawaii, 1971) started packet radio; Ethernet added carrier sense "
            "and collision detection; Wi-Fi added collision avoidance.")


# =============================================================================== 4. ALOHA instability
class AlohaStability(Experiment):
    title = "Slotted ALOHA's instability"
    blurb = "Below capacity on average, yet the backlog can run away and never come back."
    book = "sec:ch20:random"
    animate = True
    autoplay = True
    fps = 10
    controls = [
        Slider("lam", "New packets per slot λ", 0.05, 0.40, 0.30, step=0.01),
        Slider("q", "Fixed retransmission probability", 0.01, 0.5, 0.15, step=0.01),
        Button("again", "Empty the backlogs"),
    ]
    plots = [
        Plot("bl", "Backlogged packets, slot by slot", x="slot", y="backlog n", legend="tl"),
        Plot("dr", "Drift: departures versus arrivals (fixed q)", x="backlog n",
             y="packets per slot", ylim=(0, 0.45), legend="tr"),
    ]
    layout = [["bl", "dr"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("nf", "Backlog, fixed q", "", "int"),
        Readout("np_", "Backlog, pseudo-Bayesian", "", "int"),
        Readout("sf", "Throughput, fixed q", "", ".3f"),
        Readout("sp", "Throughput, pseudo-Bayesian", "", ".3f"),
    ]
    challenges = [
        Challenge("At λ = 0.30 or more, watch the fixed scheme's backlog run away past 100.",
                  lambda s: s.p.lam >= 0.299 and s.r.nf > 100),
        Challenge("Make the fixed scheme survive λ = 0.30: backlog still below 20 after 5000 slots "
                  "(press 'Empty the backlogs' after changing q).",
                  lambda s: s.p.lam >= 0.299 and s.exp.slot >= 5000 and s.r.nf < 20,
                  hint="Where must the drift curve's unstable crossing be?"),
        Challenge("Let the pseudo-Bayesian controller carry λ = 0.35: throughput within 0.02 of λ "
                  "after 3000 slots.",
                  lambda s: (s.p.lam >= 0.349 and s.exp.slot >= 3000
                             and abs(s.r.sp - s.p.lam) <= 0.02)),
    ]
    per_tick = 60

    def setup(self):
        self.reset()

    def reset(self):
        self.st_f = dict(n=0, nhat=0.0)
        self.st_p = dict(n=0, nhat=0.0)
        self.tr_f, self.tr_p = [0], [0]
        self.sf = self.sp = 0
        self.slot = 0
        self.r_ = np.random.default_rng(3)

    def on_again(self, p):
        self.reset()

    def run(self, p, n):
        for _ in range(n):
            self.sf += cs.aloha_backlog_step(self.st_f, p.lam, p.q, self.r_, "fixed")
            self.sp += cs.aloha_backlog_step(self.st_p, p.lam, p.q, self.r_, "pb")
            self.slot += 1
            self.tr_f.append(self.st_f["n"])
            self.tr_p.append(self.st_p["n"])

    def draw(self, p):
        x = np.arange(len(self.tr_f))
        pb = self.plot("bl")
        pb.line("f", x, self.tr_f, color=RED, width=1.4, name=f"fixed q = {p.q:.2f}")
        pb.line("p", x, self.tr_p, color=NAVY, width=1.4, name="pseudo-Bayesian q = 1/n̂")
        pb.set_xlim(0, max(1000, len(x)))
        pb.set_ylim(0, max(40, 1.25 * max(max(self.tr_f), max(self.tr_p))))
        # drift (Poisson approximation of the attempts): G = λ + n q
        Gs = np.linspace(0.01, 6, 2000)
        cr = np.flatnonzero(np.diff(np.sign(Gs * np.exp(-Gs) - p.lam)) != 0)
        n_u = (Gs[cr[-1]] - p.lam) / p.q if len(cr) else 50.0
        nmax = max(30.0, 2.5 * n_u, 1.2 * self.st_f["n"])
        n = np.linspace(0, nmax, 600)
        Gn = p.lam + n * p.q
        dep = Gn * np.exp(-Gn)
        pd = self.plot("dr")
        pd.line("dep", n, dep, color=NAVY, width=2.2, name="departures (G e⁻ᴳ)")
        pd.line("arr", n, np.full(len(n), p.lam), color=RED, width=1.8, style="--",
                name=f"arrivals λ = {p.lam:.2f}")
        cross = np.flatnonzero(np.diff(np.sign(dep - p.lam)) != 0)
        pd.scatter("eq", n[cross], dep[cross], color=ORANGE, size=11, outline=ORANGE,
                   name="equilibria")
        nf = self.st_f["n"]
        Gf = p.lam + nf * p.q
        pd.scatter("now", [nf], [Gf * np.exp(-Gf)], color=RED, size=13, symbol="d",
                   name="fixed scheme now")
        pd.set_xlim(0, nmax)
        s = max(self.slot, 1)
        self.readout(nf=self.st_f["n"], np_=self.st_p["n"], sf=self.sf / s, sp=self.sp / s)

    def update(self, p):
        self.draw(p)

    def tick(self, p):
        if self.slot < 20000:
            self.run(p, self.per_tick)
        self.draw(p)

    def story(self, p):
        r = self.r
        s = (f"<p>New packets arrive at {v(p.lam, '.2f')} per slot, below slotted ALOHA's capacity "
             f"of 1/e = 0.368. Backlogged stations retransmit with probability q each slot. Right: "
             f"the departure rate (navy) against the arrival rate (red). Where departures exceed "
             f"arrivals the backlog shrinks, elsewhere it grows; the orange points are the "
             f"equilibria.</p>")
        if r.get("nf", 0) > 80:
            s += (f"<p>The fixed scheme (red) has crossed the unstable equilibrium: every extra "
                  f"backlogged station adds retransmissions that cause more collisions, and the "
                  f"backlog ({v(r.get('nf', 0), 'd')}) grows without bound while its throughput "
                  f"falls to {v(r.get('sf', 0), '.3f')}.</p>")
        else:
            s += ("<p>A burst of collisions can push the fixed scheme past the middle crossing; "
                  "from there the backlog only grows. Wait, or raise λ.</p>")
        s += (f"<p>Rivest's <b>pseudo-Bayesian</b> controller (navy) estimates the backlog n̂ and "
              f"retransmits with q = 1/n̂, keeping G near 1: stable for any λ below 1/e "
              f"(throughput {v(r.get('sp', 0), '.3f')}).</p>")
        return "<h3>Stable on average, unstable in fact</h3>" + s + keybox(
            "Random access needs backlog-aware retransmission control: exponential backoff in "
            "Ethernet and Wi-Fi, backoff indicators in the cellular RACH.")


# =============================================================================== 5. Bianchi DCF
class BianchiDCF(Experiment):
    title = "Wi-Fi's DCF: Bianchi's model"
    blurb = "Many saturated stations, binary exponential backoff: how much air carries data?"
    book = "sec:ch20:random"
    heavy = True
    controls = [
        IntSlider("n", "Saturated stations", 1, 60, 10),
        Choice("W", "Minimum contention window", ["8", "16", "32", "64", "128"], "16"),
        IntSlider("m", "Backoff doublings m", 0, 7, 6),
        Choice("acc", "Access", ["Basic", "RTS/CTS"]),
        Slider("pay", "Payload", 100, 2304, 1500, step=4, unit="bytes"),
        Choice("rate", "Data rate", ["6 Mb/s", "24 Mb/s", "54 Mb/s"], "54 Mb/s"),
    ]
    plots = [
        Plot("thr", "Saturation throughput", x="stations n", y="MAC throughput (Mb/s)",
             xlim=(0, 61), legend="br"),
        Plot("pr", "Collision and attempt probabilities", x="stations n", y="probability",
             xlim=(0, 61), ylim=(0, 0.8), legend="br"),
    ]
    layout = [["thr", "pr"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("thr", "Throughput", "Mb/s", ".1f"),
        Readout("eff", "Efficiency", "%", ".0f"),
        Readout("p", "Collision probability p", "", ".3f"),
        Readout("tau", "Attempt probability τ", "", ".4f"),
    ]
    challenges = [
        Challenge("50 stations, basic access, 1500-byte frames at 54 Mb/s: pick the minimum "
                  "contention window that maximises throughput.",
                  lambda s: (s.p.n == 50 and s.p.acc == "Basic" and abs(s.p.pay - 1500) < 3
                             and s.p.rate == "54 Mb/s" and s.exp.best_W)),
        Challenge("Short frames (200 bytes or less): show that RTS/CTS loses to basic access even "
                  "with 60 stations.",
                  lambda s: s.p.pay <= 200 and s.p.n == 60 and s.exp.rts_thr < s.exp.basic_thr),
        Challenge("Squeeze the window (CWmin 8, no doublings) and watch 30 stations collide more "
                  "than half the time.",
                  lambda s: s.p.n >= 30 and s.p.W == "8" and s.p.m == 0 and s.r.p > 0.5),
    ]
    ns = np.arange(1, 61)

    def times(self, p):
        rate = int(p.rate.split()[0])
        return cs.dcf_times(int(p.pay), rate, min(24, rate))

    def curve(self, p, W, access):
        slot, pl, bas, rts = self.times(p)
        T = bas if access == "Basic" else rts
        rate = int(p.rate.split()[0])
        return np.array([cel.bianchi_throughput(n, W, p.m, slot, payload_time=pl, **T)[0] * rate
                         for n in self.ns])

    def update(self, p):
        W = int(p.W)
        rate = int(p.rate.split()[0])
        cb = self.curve(p, W, "Basic")
        cr = self.curve(p, W, "RTS/CTS")
        pt = self.plot("thr")
        pt.line("b", self.ns, cb, color=NAVY, width=2.6 if p.acc == "Basic" else 1.4,
                name=f"basic, CWmin {W}")
        pt.line("r", self.ns, cr, color=RED, width=2.6 if p.acc != "Basic" else 1.4,
                name=f"RTS/CTS, CWmin {W}")
        pt.vline("n", p.n, color=PURPLE, style=":")
        top = max(cb.max(), cr.max())
        pt.set_ylim(0, 1.15 * top)
        tp = [cel.bianchi_tau(n, W, p.m) for n in self.ns]
        pp = self.plot("pr")
        pp.line("p", self.ns, [x[1] for x in tp], color=NAVY, width=2.2, name="collision p")
        pp.line("t", self.ns, [5 * x[0] for x in tp], color=ORANGE, width=2.0, name="5 × attempt τ")
        pp.vline("n", p.n, color=PURPLE, style=":")
        cur = cb if p.acc == "Basic" else cr
        tau, pc = cel.bianchi_tau(p.n, W, p.m)
        self.basic_thr, self.rts_thr = float(cb[p.n - 1]), float(cr[p.n - 1])
        slot, pl, bas, rts = self.times(p)
        best = max((cel.bianchi_throughput(p.n, w, p.m, slot, payload_time=pl, **bas)[0], w)
                   for w in (8, 16, 32, 64, 128))[1]
        self.best_W = best == W
        self.sim = dict(x=[], y=[])
        self.readout(thr=float(cur[p.n - 1]), eff=100 * float(cur[p.n - 1]) / rate, p=pc, tau=tau)

    def background(self, p):
        slot, pl, bas, rts = self.times(p)
        T = bas if p.acc == "Basic" else rts
        rate = int(p.rate.split()[0])
        rng = np.random.default_rng(11)
        ev = 1500 if self.quick else 8000
        for n in (1, 2, 5, 10, 20, 35, 50, 60):
            s, _ = cel.dcf_simulate(n, int(p.W), p.m, slot, payload_time=pl, events=ev, rng=rng, **T)
            yield dict(n=n, s=s * rate)

    def progress(self, p, it):
        self.sim["x"].append(it["n"])
        self.sim["y"].append(it["s"])
        self.plot("thr").scatter("sim", self.sim["x"], self.sim["y"],
                                 color=NAVY if p.acc == "Basic" else RED, size=9, outline=ORANGE,
                                 name="slot-level simulation")

    def story(self, p):
        r = self.r
        s = (f"<p>{v(p.n, 'd')} stations always have a {v(p.pay, '.0f', 'byte')} frame to send. Each "
             f"counts down a random backoff from a window of {v(int(p.W), 'd')} slots, doubling it "
             f"after every collision (up to {v(p.m, 'd')} times). Bianchi's insight: every attempt "
             f"collides with the same probability p, which makes the whole thing a two-equation "
             f"fixed point. Here p = {v(r.get('p', 0), '.3f')}.</p>"
             f"<p>Throughput: {v(r.get('thr', 0), '.1f', 'Mb/s')}, only "
             f"{v(r.get('eff', 0), '.0f', '%')} of the {p.rate} PHY rate, because preambles, "
             f"inter-frame spaces, ACKs and idle backoff slots all take air time. ")
        if p.acc == "RTS/CTS":
            s += ("RTS/CTS makes collisions short (two tiny frames instead of a full data frame), "
                  "which pays with many stations and long frames, and costs with short ones.</p>")
        else:
            s += ("Collisions waste a whole data frame; RTS/CTS would shorten them.</p>")
        return "<h3>Listen, wait, back off</h3>" + s + keybox(
            "τ = 2(1−2p)/((1−2p)(W+1) + pW(1−(2p)<sup>m</sup>)), p = 1 − (1−τ)<sup>n−1</sup>. "
            "Markers: a slot-level simulation agrees.")


# =============================================================================== 6. RACH
class RACH(Experiment):
    title = "RACH collisions and barring"
    blurb = "Thousands of devices wake at once and race for 54 preambles."
    book = "sec:ch20:rach"
    heavy = True
    controls = [
        Heading("One RACH opportunity"),
        IntSlider("M", "Preambles", 8, 64, 54),
        IntSlider("k", "Devices contending", 1, 240, 20),
        Heading("A burst of devices"),
        LogSlider("ndev", "Devices", 1000, 30000, 10000, fmt=",.0f"),
        Choice("arr", "Activation", ["10 s burst", "Uniform, 60 s"]),
        Slider("acb", "Access-class barring: pass probability", 0.05, 1.0, 1.0, step=0.01),
    ]
    plots = [
        Plot("one", "One opportunity: k devices pick among M preambles", x="devices contending k",
             y="devices", xlim=(0, 240), legend="tl"),
        Plot("time", "Preambles per opportunity during the burst", x="time (s)",
             y="devices per opportunity", legend="tr"),
        Plot("sw", "Access success versus number of devices", x="devices", y="success probability",
             logx=True, xlim=(1000, 30000), ylim=(0, 1.05), legend="bl"),
    ]
    layout = [["one", "sw"], ["time", "time"]]
    readouts = [
        Readout("succ1", "Successes in one opportunity", "", ".1f"),
        Readout("coll", "Collision probability", "%", ".0f"),
        Readout("succ", "Burst: access success", "%", ".1f", good=lambda x: x >= 95),
        Readout("delay", "Mean access delay", "ms", ".0f"),
    ]
    challenges = [
        Challenge("For M = 54, find the number of contenders that maximises the successes (± 2).",
                  lambda s: s.p.M == 54 and abs(s.p.k - 53.5) <= 2,
                  hint="Too few contenders waste preambles, too many collide."),
        Challenge("30 000 devices in a 10 s burst: rescue the access success above 95 % with "
                  "barring.",
                  lambda s: (s.p.ndev >= 29500 and s.p.arr == "10 s burst" and s.p.acb < 1
                             and s.r.succ is not None and s.r.succ >= 95)),
        Challenge("…and find the gentlest barring that does it: pass probability of 0.5 or more.",
                  lambda s: (s.p.ndev >= 29500 and s.p.arr == "10 s burst" and s.p.acb >= 0.5
                             and s.p.acb < 1 and s.r.succ is not None and s.r.succ >= 95)),
    ]

    def update(self, p):
        k = np.arange(1, 241)
        po = self.plot("one")
        succ = cel.rach_success(k, p.M)
        po.line("s", k, succ, color=GREEN, width=2.4, name="succeed")
        po.line("c", k, k - succ, color=RED, width=1.8, style="--", name="collide")
        po.vline("M", p.M, color=GRAY, style=":", label="k = M", label_pos=0.45)
        s1 = float(cel.rach_success(p.k, p.M))
        po.scatter("pt", [p.k], [s1], color=NAVY, size=12, symbol="d", name="your k")
        po.set_ylim(0, 245)
        self.sweep = dict(x=[], a=[], b=[])
        self.readout(succ1=s1, coll=100 * (1 - s1 / p.k), succ=None, delay=None)

    def background(self, p):
        beta = p.arr == "10 s burst"
        spread = 10.0 if beta else 60.0
        nd = int(round(p.ndev))
        if self.quick:
            nd = min(nd, 3000)
        res = cs.rach_sim(nd, spread, beta, M=p.M, acb=p.acb, rng=np.random.default_rng(5))
        yield dict(kind="run", res=res)
        nds = [1000, 3000, 10000, 30000] if self.quick else [1000, 2000, 5000, 10000, 15000, 20000, 30000]
        for n in nds:
            a = cs.rach_sim(n, spread, beta, M=p.M, acb=1.0, rng=np.random.default_rng(6))["success"]
            b = (cs.rach_sim(n, spread, beta, M=p.M, acb=p.acb, rng=np.random.default_rng(6))["success"]
                 if p.acb < 1 else a)
            yield dict(kind="sw", n=n, a=a, b=b)

    def progress(self, p, it):
        if it["kind"] == "run":
            res = it["res"]
            sc, cc = res["per_ro"]
            t = np.arange(len(sc)) * 5e-3
            w = 40                                   # 0.2 s moving average
            ker = np.ones(w) / w
            pt = self.plot("time")
            pt.line("tot", t, np.convolve(sc + cc, ker, "same"), color=NAVY, width=1.8,
                    name="attempting")
            pt.line("s", t, np.convolve(sc, ker, "same"), color=GREEN, width=2.0, name="succeeding")
            pt.hline("cap", p.M / np.e, color=GRAY, style=":", label="M/e", label_pos=0.85)
            pt.set_xlim(0, max(t[-1], 1.0))
            pt.set_ylim(0, max(5.0, 1.15 * float(np.max(np.convolve(sc + cc, ker, "same")))))
            self.readout(succ=100 * res["success"], delay=1e3 * res["mean_delay"]
                         if np.isfinite(res["mean_delay"]) else None)
        else:
            sw = self.sweep
            sw["x"].append(it["n"])
            sw["a"].append(it["a"])
            sw["b"].append(it["b"])
            ps = self.plot("sw")
            ps.line("a", sw["x"], sw["a"], color=RED, width=2.0, name="no barring")
            ps.scatter("as", sw["x"], sw["a"], color=RED, size=7)
            if p.acb < 1:
                ps.line("b", sw["x"], sw["b"], color=GREEN, width=2.0,
                        name=f"barring, pass {p.acb:.2f}")
                ps.scatter("bs", sw["x"], sw["b"], color=GREEN, size=7)
            ps.vline("now", p.ndev, color=PURPLE, style=":")

    def story(self, p):
        r = self.r
        s = (f"<p>A device that wants to connect picks one of {v(p.M, 'd')} preambles at random. If "
             f"nobody else picked it, it gets through; otherwise the collision is discovered only "
             f"later. With {v(p.k, 'd')} contenders, {v(r.get('succ1', 0), '.1f')} succeed on "
             f"average: at most about M/e ≈ {v(p.M / np.e, '.0f')}, however many try (top left).</p>"
             f"<p>Now {v(p.ndev, ',.0f')} devices wake up "
             f"{'within 10 seconds' if p.arr.startswith('10') else 'spread over a minute'} "
             f"(a power cut ends, a train arrives). Collided devices back off and try "
             f"again, adding to the next opportunities. ")
        if r.get("succ") is not None:
            s += (f"Result: {v(r.get('succ', 0), '.1f', '%')} get access, after "
                  f"{v(r.get('delay') or 0, '.0f', 'ms')} on average. ")
        if p.acb < 1:
            s += (f"<b>Access-class barring</b> lets each device try only with probability "
                  f"{v(p.acb, '.2f')} and makes the rest wait: fewer collisions, more delay.</p>")
        else:
            s += "When the attempts exceed M/e per opportunity, the channel jams.</p>"
        return "<h3>A race for preambles</h3>" + s + keybox(
            "Massive IoT access is ALOHA again. Barring, backoff indicators and early data "
            "transmission keep it stable.")


# =============================================================================== 7. PPP coverage
def ppp_cov(T, alpha, snr1, lam, npts=1500):
    """Vectorised Andrews-Baccelli-Ganti coverage with noise (same integral as
    commlib.cellular.abg_coverage): T (n,), lam (m,) -> (n, m)."""
    T = np.atleast_1d(np.asarray(T, float))
    lam = np.atleast_1d(np.asarray(lam, float))
    rho = cel.abg_rho(T, alpha)
    v = (np.arange(npts) + 0.5) / npts
    out = np.empty((len(T), len(lam)))
    for i, (t, r_) in enumerate(zip(T, rho)):
        sc = 1.0 / (np.pi * lam[:, None] * (1 + r_))
        vv = sc * v[None, :] / (1 - v[None, :])
        jac = sc / (1 - v[None, :]) ** 2
        f = np.exp(-np.pi * lam[:, None] * vv * (1 + r_) - t / snr1 * vv ** (alpha / 2))
        out[i] = np.pi * lam * np.sum(f * jac, axis=1) / npts
    return out


class PPPCoverage(Experiment):
    title = "Stochastic geometry: PPP coverage"
    blurb = "Base stations scattered at random: coverage that does not depend on density."
    book = "sec:ch20:capacity"
    heavy = True
    controls = [
        LogSlider("lam", "Base-station density", 0.1, 100, 5, unit="/km²", fmt=".2f"),
        Slider("alpha", "Path-loss exponent α", 2.5, 5.0, 4.0, step=0.05),
        Slider("T", "SINR threshold T", -10.0, 20.0, 0.0, step=0.5, unit="dB"),
        Slider("snr", "SNR at 300 m", -10.0, 40.0, 10.0, step=1.0, unit="dB",
               help="Sets the transmit power relative to the noise"),
        Button("again", "New deployment"),
    ]
    plots = [
        ImagePlot("map", "SINR map (no fading)", x="km", y="km", aspect=True),
        Plot("cov", "Coverage P[SINR > T], Rayleigh fading", x="T (dB)", y="coverage",
             xlim=(-10, 20), ylim=(0, 1.02), legend="bl", legend_cols=2),
        Plot("dens", "Densification at your threshold", x="base stations per km²", y="coverage",
             logx=True, xlim=(0.1, 100), ylim=(0, 1.02), legend="br"),
    ]
    layout = [["map", "cov"], ["map", "dens"]]
    readouts = [
        Readout("pc", "Coverage (analysis)", "", ".3f"),
        Readout("sim", "Coverage (simulation)", "", ".3f"),
        Readout("lim", "Interference-limited value", "", ".3f"),
        Readout("se", "Mean spectral efficiency", "b/s/Hz", ".2f"),
    ]
    challenges = [
        Challenge("Check Andrews–Baccelli–Ganti: α = 4, T = 0 dB, interference-limited (SNR ≥ 30 dB): "
                  "coverage 1/(1 + π/4) = 0.56.",
                  lambda s: (abs(s.p.alpha - 4) < 0.03 and abs(s.p.T) < 0.25 and s.p.snr >= 30
                             and abs(s.r.pc - 0.56) < 0.01)),
        Challenge("With 0 dB SNR at 300 m, find the density where coverage at 0 dB comes within "
                  "0.02 of its interference-limited value (± 50 %).",
                  lambda s: (s.p.snr <= 0.5 and abs(s.p.T) < 0.25 and s.exp.dens_ok),
                  hint="Beyond this density, more cells add capacity per km² but not coverage."),
        Challenge("Steeper path loss isolates cells: coverage above 0.5 at T = 5 dB.",
                  lambda s: s.p.T >= 5 and s.r.pc > 0.5),
    ]

    def setup(self):
        self.seed = 21
        self._hex = {}

    def on_again(self, p):
        self.seed += 1

    def snr1(self, p):
        return float(undb(p.snr)) * 300.0 ** p.alpha            # SNR at 1 m (metre units)

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        side = np.sqrt(60 / p.lam)                               # km, about 60 sites in view
        bs = cel.ppp_drop(p.lam, 1.7 * side, rng)
        g = np.linspace(-side / 2, side / 2, 96)
        X, Y = np.meshgrid(g, g)
        d = np.maximum(np.abs((X + 1j * Y)[..., None] - bs[None, None, :]), 0.01)   # km
        pw = undb(p.snr) * (0.3 / d) ** p.alpha
        S = pw.max(axis=-1)
        I = pw.sum(axis=-1) - S
        sinr = db(S / (I + 1))
        pm = self.plot("map")
        pm.image("img", np.clip(sinr, -10, 30), x=(g[0], g[-1]), y=(g[0], g[-1]), cmap=PALE_STOPS,
                 levels=(-10, 30), colorbar=True, cbar_label="SINR (dB)")
        sel = (np.abs(bs.real) < side / 2) & (np.abs(bs.imag) < side / 2)
        pm.scatter("bs", bs.real[sel], bs.imag[sel], color=RED, size=8, symbol="t1")
        pm.set_xlim(g[0], g[-1])
        pm.set_ylim(g[0], g[-1])
        TdB = np.linspace(-10, 20, 61)
        T = undb(TdB)
        lam_m = p.lam * 1e-6
        pc_n = ppp_cov(T, p.alpha, self.snr1(p), [lam_m])[:, 0]
        pc_i = 1 / (1 + cel.abg_rho(T, p.alpha))
        pv = self.plot("cov")
        pv.line("i", TdB, pc_i, color=GRAY, style="--", width=1.6, name="interference only")
        pv.line("n", TdB, pc_n, color=NAVY, width=2.4, name="with noise (analysis)")
        ka = round(p.alpha, 2)
        if ka not in self._hex:
            h = cs.hex_rayleigh_sinr(p.alpha, 8000, np.random.default_rng(2))
            self._hex[ka] = np.array([(h > t).mean() for t in T])
        pv.line("hex", TdB, self._hex[ka], color=ORANGE, style=":", width=1.8,
                name="hexagonal grid (interference only)")
        pv.vline("T", p.T, color=PURPLE, style=":")
        lams = np.logspace(-1, 2, 40)
        allc = ppp_cov([undb(p.T)], p.alpha, self.snr1(p),
                       np.r_[lam_m, p.lam / 1.5 * 1e-6, lams * 1e-6])[0]
        pc, lo, pd_ = float(allc[0]), float(allc[1]), allc[2:]
        lim = float(1 / (1 + cel.abg_rho([undb(p.T)], p.alpha)[0]))
        pd = self.plot("dens")
        pd.line("d", lams, pd_, color=NAVY, width=2.4, name="with noise")
        pd.hline("lim", lim, color=GRAY, style="--", label="interference-limited", label_pos=0.02)
        pd.scatter("now", [p.lam], [pc], color=RED, size=12, symbol="d", name="your density")
        self.dens_ok = pc >= lim - 0.02 and lo < lim - 0.02
        self.sim = dict(x=[], y=[])
        self.readout(pc=pc, lim=lim, sim=None, se=None)

    def background(self, p):
        rng = np.random.default_rng(self.seed + 100)
        trials = 200 if self.quick else 1500
        s = cel.ppp_sinr_samples(p.lam * 1e-6, p.alpha, snr=self.snr1(p), trials=trials, rng=rng)
        yield dict(s=s)

    def progress(self, p, it):
        s = it["s"]
        TdB = np.arange(-10, 20.1, 2.5)
        cov = [(s > undb(t)).mean() for t in TdB]
        pv = self.plot("cov")
        pv.scatter("mc", TdB, cov, color=RED, size=9, outline=RED, name="PPP simulation")
        self.readout(sim=float((s > undb(p.T)).mean()), se=float(np.mean(np.log2(1 + s))))

    def story(self, p):
        r = self.r
        s = (f"<p>Real base stations are not on a hexagonal grid. Model them as a <b>Poisson point "
             f"process</b>: {v(p.lam, '.2f')} per km², placed independently at random (red "
             f"triangles), each user served by the nearest. With Rayleigh fading, Andrews, Baccelli "
             f"and Ganti (2011) found the coverage in closed form: here "
             f"{v(r.get('pc', 0), '.3f')} at T = {v(p.T, '.1f', 'dB')}.</p>"
             f"<p>The surprise: when interference dominates, coverage does <b>not depend on "
             f"density</b> ({v(r.get('lim', 0), '.3f')}): halve the distances and both the wanted "
             f"signal and every interferer grow by the same factor. Only noise breaks the "
             f"symmetry (bottom right): sparse networks are noise-limited, dense ones hit the "
             f"interference ceiling.</p>")
        return "<h3>Random cells, exact answers</h3>" + s + keybox(
            "α = 4, no noise: P<sub>c</sub> = 1/(1 + √T(π/2 − arctan(1/√T))). The hexagonal grid "
            "(orange) is the optimistic bound, the PPP the pessimistic one.")


# =============================================================================== 8. scheduling
class PFScheduler(Experiment):
    title = "Proportional fair scheduling"
    blurb = "Serve whoever's channel is relatively best: throughput and fairness, together."
    book = "sec:ch20:scheduling"
    animate = True
    autoplay = True
    fps = 8
    controls = [
        IntSlider("K", "Users", 2, 20, 8),
        Choice("pol", "Scheduler", ["Round robin", "Max rate", "α-fair"], "Round robin"),
        Slider("alpha", "Fairness exponent α", 0.0, 5.0, 1.0, step=0.05,
               help="0 = max rate, 1 = proportional fair, large = max-min fair",
               enabled_if=lambda p: p.pol == "α-fair"),
        Slider("best", "Best user's mean SNR", 0.0, 30.0, 20.0, step=1.0, unit="dB"),
        Slider("edge", "Edge user's mean SNR", -10.0, 30.0, -5.0, step=1.0, unit="dB"),
        Choice("speed", "Fading", ["Slow", "Fast"]),
    ]
    plots = [
        ImagePlot("rates", "Rates the users could get (colour); who is served (red)",
                  x="slot (recent)", y="user (1 = best)"),
        BarPlot("thr", "Throughput per user so far", x="user", y="b/s/Hz"),
        Plot("tr", "The fairness–throughput trade-off", x="Jain's fairness index",
             y="cell throughput (b/s/Hz)", xlim=(0, 1.02), legend="bl"),
    ]
    layout = [["rates", "rates"], ["thr", "tr"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("cell", "Cell throughput", "b/s/Hz", ".2f"),
        Readout("jain", "Jain's index", "", ".2f"),
        Readout("edge", "Edge user", "b/s/Hz", ".3f"),
        Readout("gain", "Gain over round robin", "×", ".2f"),
    ]
    challenges = [
        Challenge("Proportional fair (α = 1): give the edge user more than its round-robin share "
                  "and the cell more than round robin.",
                  lambda s: (s.p.pol == "α-fair" and abs(s.p.alpha - 1) < 0.03 and s.exp.n >= 1500
                             and s.exp.edge_gain > 1.0 and s.r.gain > 1.0)),
        Challenge("Starve the edge: a scheduler whose Jain index falls below 0.3.",
                  lambda s: s.exp.n >= 1500 and s.r.jain < 0.3),
        Challenge("Pure multiuser diversity: equal users (best = edge), at least 1.6× round robin.",
                  lambda s: (s.p.best == s.p.edge and s.exp.n >= 1500 and s.r.gain >= 1.6),
                  hint="More users means a better best channel in every slot."),
    ]
    W = 120

    def setup(self):
        self.key = None

    def reset(self, p):
        r = np.random.default_rng(40)
        self.r_ = r
        self.h = (r.standard_normal(p.K) + 1j * r.standard_normal(p.K)) / np.sqrt(2)
        self.avg = np.full(p.K, 1e-3)
        self.served = np.zeros(p.K)
        self.ratesum = np.zeros(p.K)
        self.n = 0
        self.img = np.zeros((p.K, self.W))
        self.sel = np.full(self.W, -1)
        self.means = undb(np.linspace(p.best, p.edge, p.K))

    def run(self, p, slots):
        a = 0.98 if p.speed == "Slow" else 0.7
        K = p.K
        for _ in range(slots):
            w = (self.r_.standard_normal(K) + 1j * self.r_.standard_normal(K)) / np.sqrt(2)
            self.h = a * self.h + np.sqrt(1 - a * a) * w
            r = np.log2(1 + self.means * np.abs(self.h) ** 2)
            if p.pol == "Round robin":
                k = self.n % K
            elif p.pol == "Max rate":
                k = int(np.argmax(r))
            else:
                k = int(np.argmax(r / self.avg ** p.alpha))
            inst = np.zeros(K)
            inst[k] = r[k]
            self.avg = 0.99 * self.avg + 0.01 * inst
            self.served += inst
            self.ratesum += r
            self.n += 1
            self.img = np.roll(self.img, -1, axis=1)
            self.img[:, -1] = r
            self.sel = np.roll(self.sel, -1)
            self.sel[-1] = k

    def tradeoff(self, p):
        key = (p.K, p.best, p.edge, p.speed)
        if getattr(self, "_tk", None) != key:
            r = np.random.default_rng(41)
            a = 0.98 if p.speed == "Slow" else 0.7
            S = 800
            h = np.empty((S, p.K), complex)
            x = (r.standard_normal(p.K) + 1j * r.standard_normal(p.K)) / np.sqrt(2)
            for s in range(S):
                x = a * x + np.sqrt(1 - a * a) * (r.standard_normal(p.K) + 1j * r.standard_normal(p.K)) / np.sqrt(2)
                h[s] = x
            rates = np.log2(1 + undb(np.linspace(p.best, p.edge, p.K))[None, :] * np.abs(h) ** 2)
            pts = []
            for al in (0, 0.25, 0.5, 1.0, 2.0, 5.0):
                thr = cel.schedule(rates, "alpha", alpha=al, tc=100)
                pts.append((cel.jain(thr), thr.sum(), al))
            rr = cel.schedule(rates, "rr")
            self._tr = (np.array(pts), (cel.jain(rr), rr.sum()))
            self._tk = key
        return self._tr

    def draw(self, p):
        pr = self.plot("rates")
        top = float(np.log2(1 + undb(max(p.best, p.edge)) * 4))
        pr.image("img", self.img, x=(0, self.W), y=(0.5, p.K + 0.5), cmap="heat", levels=(0, top),
                 colorbar=True, cbar_label="b/s/Hz")
        ok = self.sel >= 0
        pr.scatter("sel", np.arange(self.W)[ok] + 0.5, self.sel[ok] + 1, color=RED, size=6,
                   symbol="s")
        pr.set_xlim(0, self.W)
        pr.set_ylim(0.5, p.K + 0.5)
        n = max(self.n, 1)
        thr = self.served / n
        rr_share = self.ratesum / n / p.K
        pb = self.plot("thr")
        k = np.arange(1, p.K + 1)
        pb.bars("rr", k - 0.2, rr_share, width=0.38, color=GRAY)
        pb.bars("cur", k + 0.2, thr, width=0.38, color=NAVY)
        pb.set_xlim(0.4, p.K + 0.6)
        pb.set_ylim(0, 1.3 * max(thr.max(), rr_share.max(), 1e-3))
        pb.set_xticks([(i, str(i)) for i in k])
        pb.text("leg", 0.5, 1.25 * max(thr.max(), rr_share.max(), 1e-3),
                "navy: this scheduler   grey: round-robin share", size=8.5, anchor=(0, 0))
        pts, rr = self.tradeoff(p)
        pt = self.plot("tr")
        pt.line("c", pts[:, 0], pts[:, 1], color=NAVY, width=1.8, name="α-fair, α = 0 … 5")
        pt.scatter("cp", pts[:, 0], pts[:, 1], color=NAVY, size=7)
        for i, (j_, s_, al) in enumerate(pts):
            pt.text(f"a{i}", j_, s_, f" α={al:g}", size=8, anchor=(0, 1))
        pt.scatter("rr", [rr[0]], [rr[1]], color=GRAY, size=11, symbol="s", name="round robin")
        pt.scatter("now", [cel.jain(thr) if thr.sum() > 0 else 0], [thr.sum()], color=RED, size=13,
                   symbol="d", name="you, live")
        pt.set_ylim(0, 1.2 * max(pts[:, 1].max(), thr.sum(), 0.1))
        rrc = float(rr_share.sum())
        self.edge_gain = float(thr[-1] / max(rr_share[-1], 1e-9))
        self.readout(cell=float(thr.sum()), jain=float(cel.jain(thr)) if thr.sum() > 0 else 0.0,
                     edge=float(thr[-1]), gain=float(thr.sum() / max(rrc, 1e-9)))

    def update(self, p):
        self.reset(p)
        self.run(p, 150)
        self.draw(p)

    def tick(self, p):
        if self.n < 20000:
            self.run(p, 25)
        self.draw(p)

    def story(self, p):
        r = self.r
        s = (f"<p>{v(p.K, 'd')} users with mean SNRs from {v(p.best, '.0f', 'dB')} down to "
             f"{v(p.edge, '.0f', 'dB')}, each fading independently (top: the rate each could get, "
             f"slot by slot; red: who is served). ")
        if p.pol == "Max rate" or (p.pol == "α-fair" and p.alpha < 0.2):
            s += ("<b>Max rate</b> serves the best instantaneous channel: maximum cell throughput, "
                  "and the edge users starve.</p>")
        elif p.pol == "Round robin":
            s += ("<b>Round robin</b> takes turns regardless of the channel: perfectly fair in time, "
                  "but it serves users in their fades as often as in their peaks.</p>")
        else:
            s += (f"The <b>α-fair</b> scheduler serves the largest r/R̄<sup>α</sup>: with α = 1, "
                  f"<b>proportional fair</b>, each user gets the slots where its channel is "
                  f"relatively best, riding its own peaks. ")
            s += ("Higher α pushes towards equal throughput, lower α towards max rate.</p>")
        s += (f"<p>Cell throughput {v(r.get('cell', 0), '.2f', 'b/s/Hz')}, "
              f"{v(r.get('gain', 0), '.2f', '×')} round robin; Jain's index "
              f"{v(r.get('jain', 0), '.2f')}. That gain over round robin is <b>multiuser "
              f"diversity</b>: with many independent fades, someone is always on a peak.</p>")
        return "<h3>Riding the peaks</h3>" + s + keybox(
            "Book example: A has R̄ = 8, r = 10; B has R̄ = 1, r = 1.5 Mb/s. Max rate serves A; PF "
            "serves B (1.5 > 1.25).")


# =============================================================================== 9. handover
class Handover(Experiment):
    title = "Handover: hysteresis and TTT"
    blurb = "Drive between two sites through shadowing: when should the phone switch cells?"
    book = "sec:ch20:mobility"
    heavy = True
    controls = [
        Slider("hyst", "Hysteresis", 0.0, 10.0, 1.0, step=0.5, unit="dB"),
        Slider("ttt", "Time-to-trigger", 0.0, 2560.0, 0.0, step=40.0, unit="ms"),
        Slider("speed", "Speed", 1.0, 40.0, 15.0, step=1.0, unit="m/s"),
        Slider("sigma", "Shadowing σ", 0.0, 12.0, 8.0, step=0.5, unit="dB"),
        Button("again", "Another drive"),
    ]
    plots = [
        Plot("drive", "Filtered RSRP along the route (2 km between the sites)", x="position (m)",
             y="RSRP (dB, relative)", xlim=(0, 2000), legend="tr", legend_cols=3),
        Plot("ho", "Handovers per drive (mean of many)", x="hysteresis (dB)", y="handovers",
             logy=True, xlim=(0, 10), ylim=(0.5, 100), legend="tr"),
        Plot("late", "The price: route on a cell > 3 dB weaker", x="hysteresis (dB)", y="% of route",
             xlim=(0, 10), legend="tl"),
    ]
    layout = [["drive", "drive"], ["ho", "late"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("n", "Handovers this drive", "", "int"),
        Readout("pp", "Ping-pongs", "", "int", good=lambda x: x == 0),
        Readout("worse", "Route on a weaker cell", "%", ".1f", good=lambda x: x < 5),
        Readout("mean", "Mean handovers per drive", "", ".1f"),
    ]
    challenges = [
        Challenge("Remove all protection (0 dB, 0 ms) and count at least 20 handovers on one drive.",
                  lambda s: s.p.hyst == 0 and s.p.ttt == 0 and s.r.n >= 20),
        Challenge("Time-to-trigger alone (0 dB hysteresis): bring the mean below 3 handovers per "
                  "drive.",
                  lambda s: s.p.hyst == 0 and s.r.mean is not None and s.r.mean < 3),
        Challenge("Tune it like a network: mean below 2 handovers with under 5 % of the route on a "
                  "weaker cell (mean over drives).",
                  lambda s: (s.r.mean is not None and s.r.mean < 2 and s.exp.mean_worse is not None
                             and s.exp.mean_worse < 5)),
    ]

    def setup(self):
        self.seed = 31

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        ttt_m = p.ttt * 1e-3 * p.speed
        d = cs.handover_drive(p.hyst, ttt_m, rng, sigma=p.sigma, pingpong_m=5 * p.speed)
        off = max(d["f1"].max(), d["f2"].max())
        f1, f2 = d["f1"] - off, d["f2"] - off
        pd = self.plot("drive")
        pd.line("f1", d["x"], f1, color=NAVY, width=1.2, name="cell 1")
        pd.line("f2", d["x"], f2, color=GREEN, width=1.2, name="cell 2")
        serv = np.where(d["serving"] == 0, f1, f2)
        pd.line("srv", d["x"], serv, color=RED, width=2.4, alpha=0.6, name="serving")
        lo = float(min(f1.min(), f2.min()))
        hp = d["ho_positions"]
        if len(hp):
            pd.scatter("ho", hp, np.full(len(hp), lo - 4), color=ORANGE, size=10, symbol="t",
                       name="handover")
        pd.set_ylim(lo - 8, 12)
        self.curve = dict(h=[], n=[], w=[])
        self.mean_worse = None
        self.readout(n=d["count"], pp=d["pingpong"], worse=100 * d["worse"], mean=None)

    def background(self, p):
        ttt_m = p.ttt * 1e-3 * p.speed
        drives = 6 if self.quick else 40
        hs = [p.hyst] + [h for h in np.arange(0, 10.5, 1.0) if abs(h - p.hyst) > 1e-9]
        for h in hs:
            rng = np.random.default_rng(100)
            res = [cs.handover_drive(h, ttt_m, rng, sigma=p.sigma, pingpong_m=5 * p.speed)
                   for _ in range(drives)]
            yield dict(h=h, n=np.mean([r["count"] for r in res]),
                       w=100 * np.mean([r["worse"] for r in res]), cur=(h == p.hyst))

    def progress(self, p, it):
        c = self.curve
        c["h"].append(it["h"])
        c["n"].append(it["n"])
        c["w"].append(it["w"])
        o = np.argsort(c["h"])
        h, n, w = np.array(c["h"])[o], np.array(c["n"])[o], np.array(c["w"])[o]
        ph = self.plot("ho")
        ph.line("n", h, np.maximum(n, 0.5), color=NAVY, width=2.0, name=f"TTT {p.ttt:.0f} ms")
        ph.scatter("ns", h, np.maximum(n, 0.5), color=NAVY, size=7)
        pl = self.plot("late")
        pl.line("w", h, w, color=RED, width=2.0, name=f"TTT {p.ttt:.0f} ms")
        pl.scatter("ws", h, w, color=RED, size=7)
        pl.set_ylim(0, max(10.0, 1.2 * w.max()))
        if it["cur"]:
            ph.scatter("now", [it["h"]], [max(it["n"], 0.5)], color=ORANGE, size=13, symbol="d")
            pl.scatter("now", [it["h"]], [it["w"]], color=ORANGE, size=13, symbol="d")
            self.mean_worse = float(it["w"])
            self.readout(mean=float(it["n"]))

    def story(self, p):
        r = self.r
        s = (f"<p>The phone drives from site 1 to site 2 at {v(p.speed, '.0f', 'm/s')}, measuring "
             f"both cells through {v(p.sigma, '.1f', 'dB')} of shadowing. Near the boundary the two "
             f"traces cross again and again. It hands over when the other cell is better by "
             f"{v(p.hyst, '.1f', 'dB')} (<b>hysteresis</b>) for {v(p.ttt, '.0f', 'ms')} "
             f"(<b>time-to-trigger</b>).</p>"
             f"<p>This drive: {v(r.get('n', 0), 'd')} handovers, {v(r.get('pp', 0), 'd')} of them "
             f"<b>ping-pongs</b> (back within 5 s), and "
             f"{v(r.get('worse', 0), '.1f', '%')} of the route on a cell more than 3 dB weaker "
             f"than the best. ")
        if r.get("n", 0) > 6:
            s += "Every one costs signalling and an interruption.</p>"
        else:
            s += "Fewer handovers, but each one comes later.</p>"
        s += ("<p>Bottom: averaged over many drives, more hysteresis kills the ping-pong (left) "
              "but keeps the user on a fading cell longer (right), which ends in radio-link "
              "failures if overdone.</p>")
        return "<h3>Not too early, not too late</h3>" + s + keybox(
            "Typical settings: 2–4 dB of hysteresis and a few hundred milliseconds of TTT; "
            "networks tune them per neighbour (mobility robustness optimisation).")


# =============================================================================== the lab
LAB = st.Lab(26, "The Cellular Concept", chapter=20,
             chapter_title="Multiple Access and the Cellular Concept",
             experiments=[HexReuse, Erlang, AlohaLive, AlohaStability, BianchiDCF, RACH,
                          PPPCoverage, PFScheduler, Handover])

if __name__ == "__main__":
    st.run(LAB)
