"""Lab 11 · Air Interfaces: Cells, Trunks, Schedulers and Contention   (Chapters 20-22)

Run it:      python labs/lab11_air_interfaces.py
Self-test:   python labs/lab11_air_interfaces.py --selftest

A physical layer moves bits over one link; an air interface shares a scarce spectrum among
thousands of users: in space (cells and reuse), in time and frequency (TDMA, OFDMA), by
statistics (trunking) and by politeness (Wi-Fi's listen-before-talk). Seven experiments walk
up that stack: the reuse map and its interference, a live telephone switchboard obeying
Erlang's law, an OFDMA scheduler serving users who walk around the cell, NR link adaptation
with HARQ on a fading channel, Wi-Fi stations contending for the air, the EVM budget of
4096-QAM, and a side-by-side tour of GSM, UMTS, LTE, NR, Wi-Fi, Bluetooth and LoRa.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import colorsys
import heapq
from functools import lru_cache

import numpy as np
from scipy.special import j0

import commlib as cl
from commlib import airif as ai
from commlib import cellular as cel
from commlib import iot
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, ImagePlot, BarPlot,
                    Readout, Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

USER_COLS = ["#2E86C1", "#C0392B", "#1E7B4F", "#C0661A", "#6C3483", "#117A65", "#B7950B",
             "#1B3A5C", "#E67E22", "#8E44AD", "#16A085", "#7F8C8D"]


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def cat_cmap(colors):
    """A stepped colormap for an integer image with values 0..n-1 (use levels=(-0.5, n-0.5))."""
    n = len(colors)
    stops = []
    for k, c in enumerate(colors):
        stops.append((k / n, c))
        stops.append(((k + 1) / n - 1e-4 if k < n - 1 else 1.0, c))
    return tuple(stops)


def pastels(n, dark=False, offset=0.0):
    """n distinct soft colours (hex), for maps and grids."""
    out = []
    for k in range(n):
        h = (offset + k / max(n, 1)) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, 0.30 if dark else 0.84, 0.35 if dark else 0.55)
        out.append("#%02X%02X%02X" % (int(255 * r), int(255 * g), int(255 * b)))
    return out


def hex_outline(centres, R=1.0):
    """One NaN-separated polyline drawing the outline of every hexagon in ``centres``."""
    ang = np.pi / 6 + np.arange(7) * np.pi / 3
    xs, ys = [], []
    for c in centres:
        v_ = c + R * np.exp(1j * ang)
        xs += list(v_.real) + [np.nan]
        ys += list(v_.imag) + [np.nan]
    return np.array(xs), np.array(ys)


# =============================================================================== 1. reuse and SIR
ASP = 1.15             # height/width of the reuse map
CLUSTERS = {N: (i, j) for N, i, j in cel.valid_cluster_sizes(21)}


def sector_gain_db(theta_deg, bw=65.0, am=20.0):
    """3GPP-style sector antenna pattern: −min(12(θ/θ3dB)², A_m) (Chapter 20)."""
    th = (np.asarray(theta_deg) + 180) % 360 - 180
    return -np.minimum(12 * (th / bw) ** 2, am)


@lru_cache(maxsize=512)
def sir_samples(N, n_exp, sigma, sectors, users=3000):
    """Downlink SIR (dB) of users dropped uniformly in the centre cell against two tiers of
    co-channel cells (as Figure ch20_sir_reuse), log-normal shadowing on every link."""
    i, j = CLUSTERS[N]
    r_ = np.random.default_rng(7)
    u = cel.drop_in_hex(users, rng=r_, rmin=0.05)
    if sectors == 3:
        u = u[np.abs(np.angle(u, deg=True)) < 60]
    if N == 1:
        q, rr = cel.hex_axial_grid(4)
        x, y = cel.axial_to_xy(q, rr)
        bs = (x + 1j * y)[1:]
    else:
        bs = cel.cochannel_centres(i, j, tiers=2)
    z1 = r_.standard_normal(len(u))
    z2 = r_.standard_normal((len(u), len(bs)))
    S = np.abs(u) ** -n_exp * 10 ** (sigma * z1 / 10)
    d = u[:, None] - bs[None, :]
    g = np.abs(d) ** -n_exp * 10 ** (sigma * z2 / 10)
    if sectors == 3:
        S = S * 10 ** (sector_gain_db(np.angle(u, deg=True)) / 10)
        g = g * 10 ** (sector_gain_db(np.angle(d, deg=True)) / 10)
    return np.sort(db(S / g.sum(axis=1)))


@lru_cache(maxsize=32)
def reuse_map(N, dark):
    """Pixel map of the channel-set labels of a hexagonal lattice for cluster size N."""
    i, j = CLUSTERS[N]
    D = np.sqrt(3 * N)
    L = max(3.0, 1.15 * D + 1.2)
    nx = 320
    ny = int(nx * ASP)
    xs = np.linspace(-L, L, nx)
    ys = np.linspace(-ASP * L, ASP * L, ny)
    X, Y = np.meshgrid(xs, ys)
    xc = X / np.sqrt(3) - Y / 3
    zc = Y / 1.5
    yc = -xc - zc
    rx, ry, rz = np.round(xc), np.round(yc), np.round(zc)
    dx, dy, dz = np.abs(rx - xc), np.abs(ry - yc), np.abs(rz - zc)
    m1 = (dx > dy) & (dx > dz)
    rx[m1] = -ry[m1] - rz[m1]
    m2 = ~m1 & (dy > dz)
    rz2 = np.where(~m1 & ~m2, -rx - ry, rz)
    q, r = rx.astype(int), rz2.astype(int)
    s = np.mod((i + j) * q + j * r, N)
    t = np.mod(-j * q + i * r, N)
    key = s * N + t
    uniq, inv = np.unique(key, return_inverse=True)
    lab = inv.reshape(key.shape)
    lab0 = int(np.searchsorted(uniq, 0))
    # relabel so that the centre cell's channel set is category 0
    lab = np.where(lab == lab0, 0, np.where(lab < lab0, lab + 1, lab))
    cols = ["#A8473A" if dark else "#E57361"] + pastels(max(N - 1, 1), dark, 0.55)[:max(N - 1, 0)]
    # hexagons in view and their labels (for outlines and numbers)
    qq, rr = cel.hex_axial_grid(int(L / 1.5) + 3)
    cx, cy = cel.axial_to_xy(qq, rr)
    keep = (np.abs(cx) < L + 1) & (np.abs(cy) < ASP * L + 1)
    cen = (cx + 1j * cy)[keep]
    ks = np.mod((i + j) * qq[keep] + j * rr[keep], N) * N + np.mod(-j * qq[keep] + i * rr[keep], N)
    cl_ = np.searchsorted(uniq, ks)
    cl_ = np.where(cl_ == lab0, 0, np.where(cl_ < lab0, cl_ + 1, cl_))
    return dict(img=lab.astype(float), L=L, cols=cols, cen=cen, lab=cl_, D=D)


class ReuseSIR(Experiment):
    title = "Frequency reuse and SIR"
    blurb = "Reuse the same channels in cells far enough apart: how far is far enough?"
    book = "sec:ch20:interference"
    controls = [
        Heading("Reuse plan"),
        Choice("N", "Cluster size N", [str(k) for k in CLUSTERS], "7",
               help="N = i² + ij + j²: the cells sharing one set of channels are D = √(3N)·R apart"),
        Choice("sect", "Base-station antennas", ["Omni", "3 sectors"],
               help="Three 120° sectors per site: each sector hears only about a third of the "
                    "interferers"),
        IntSlider("total", "Channels in the system", 50, 800, 395, step=5,
                  help="AMPS gave each operator 395 voice channels (plus 21 control channels)"),
        Heading("Propagation"),
        Slider("n", "Path-loss exponent n", 2.5, 5.0, 3.5, step=0.05,
               help="2 in free space; 3–4 in typical macro cells"),
        Slider("sigma", "Shadowing σ", 0.0, 10.0, 6.0, step=0.5, unit="dB",
               help="Log-normal shadowing, independent on every link"),
    ]
    plots = [
        Plot("map", "The reuse pattern (cells with the same colour share channels)", x="", y="",
             aspect=True, grid=False, legend=None),
        Plot("cdf", "Downlink SIR over the centre cell", x="SIR (dB)",
             y="fraction of users below", xlim=(-10, 55), ylim=(0, 1.02), legend="br"),
        Plot("vsN", "Reuse buys SIR, and costs channels", x="cluster size N", y="SIR (dB)",
             xlim=(0, 22), ylim=(-15, 45), legend="tl"),
    ]
    layout = [["map", "cdf"], ["map", "vsN"]]
    col_stretch = [5, 4]
    readouts = [
        Readout("form", "Edge SIR, formula", "dB", ".1f",
                help="(D/R)ⁿ / 6 for omni cells, (D/R)ⁿ / 2 for three sectors"),
        Readout("p5", "Worst 5 % of users below", "dB", ".1f", good=lambda x: x >= 18),
        Readout("med", "Median SIR", "dB", ".1f"),
        Readout("chan", "Channels per cell", "", "int"),
    ]
    challenges = [
        Challenge("AMPS needed 18 dB. With n = 4, find the smallest cluster whose formula SIR "
                  "reaches it.",
                  lambda s: abs(s.p.n - 4) < 0.03 and s.p.N == "7" and s.p.sect == "Omni",
                  hint="Formula SIR = (3N)^(n/2)/6. Try N = 4 and N = 7."),
        Challenge("With 8 dB of shadowing, lift the worst 5 % of users above 8 dB using at most "
                  "N = 12.",
                  lambda s: s.p.sigma >= 8 and int(s.p.N) <= 12 and s.r.p5 >= 8,
                  hint="Sectors cut the number of interferers and add front-to-back isolation."),
        Challenge("Run reuse 1, like LTE and NR, and see the worst 5 % of users fall below 0 dB.",
                  lambda s: s.p.N == "1" and s.r.p5 < 0),
        Challenge("Keep N = 7 but sectorise: gain at least 3 dB for the worst 5 % of users over "
                  "omni cells.",
                  lambda s: s.p.N == "7" and s.p.sect == "3 sectors" and s.r.p5 >= s.exp.ref7 + 3,
                  hint="Compare at the same n and σ."),
    ]

    def update(self, p):
        N = int(p.N)
        sect = 3 if p.sect == "3 sectors" else 1
        dark = self.plot("map").theme.name == "dark"
        M = reuse_map(N, dark)
        pm = self.plot("map")
        L = M["L"]
        pm.image("img", M["img"], x=(-L, L), y=(-ASP * L, ASP * L),
                 cmap=cat_cmap(M["cols"]), levels=(-0.5, len(M["cols"]) - 0.5))
        ox, oy = hex_outline(M["cen"])
        pm.line("hex", ox, oy, color=pm.theme.plot_bg, width=1.2)
        D = M["D"]
        # numbers on the cells near the centre
        near = np.argsort(np.abs(M["cen"]))[:min(len(M["cen"]), max(19, int(2.2 * N) + 7))]
        for k, idx in enumerate(near):
            c = M["cen"][idx]
            if abs(c.real) < L - 0.6 and abs(c.imag) < ASP * L - 0.6:
                pm.text(f"n{k}", c.real, c.imag, str(M["lab"][idx] + 1), anchor=(0.5, 0.5),
                        size=8, bold=M["lab"][idx] == 0,
                        color=None if M["lab"][idx] else "#FFFFFF")
        # first-tier interferers
        if N > 1:
            i, j = CLUSTERS[N]
            tier = cel.cochannel_centres(i, j, tiers=1)
        else:
            tier = np.sqrt(3) * np.exp(1j * (np.pi / 6 + np.arange(6) * np.pi / 3))
        lx, ly = [], []
        for c in tier:
            lx += [0, c.real, np.nan]
            ly += [0, c.imag, np.nan]
        pm.line("int", lx, ly, color=NAVY, width=1.6, style="--")
        pm.scatter("bs", [0], [0], color=NAVY, size=11, symbol="t")
        c0 = tier[np.argmax(tier.real + 0.01 * tier.imag)]
        pm.text("D", c0.real / 2, c0.imag / 2 + 0.25, f"D = {D:.2f} R", color=NAVY, size=9,
                bold=True, anchor=(0.5, 0), fill=True)
        pm.set_xlim(-L, L)
        pm.set_ylim(-ASP * L, ASP * L)
        pm.set_xticks([])
        pm.set_yticks([])
        # CDF
        s = sir_samples(N, round(p.n, 3), round(p.sigma, 2), sect)
        cdf = np.arange(1, len(s) + 1) / len(s)
        pc = self.plot("cdf")
        for k, Nr in enumerate((1, 3, 7, 21)):
            if Nr == N:
                continue
            sr = sir_samples(Nr, round(p.n, 3), round(p.sigma, 2), sect)
            pc.line(f"r{Nr}", sr, np.arange(1, len(sr) + 1) / len(sr), color=GRAY, width=1.0,
                    alpha=0.7)
            pc.text(f"rt{Nr}", float(np.median(sr)), 0.5, f"N={Nr}", color=GRAY, size=8,
                    anchor=(1.05, 0.5))
        pc.line("cur", s, cdf, color=NAVY, width=2.6, name=f"N = {N}, {p.sect.lower()}")
        pc.vline("18", 18, color=GREEN, style=":", label="18 dB", label_pos=0.9)
        p5 = float(np.percentile(s, 5))
        pc.scatter("p5", [p5], [0.05], color=RED, size=11, symbol="d", name="worst 5 %")
        # SIR versus N
        Ns = np.linspace(1, 21, 200)
        nint = 6 if sect == 1 else 2
        pv = self.plot("vsN")
        pv.line("f", Ns, db(cel.reuse_ratio(Ns) ** p.n / nint), color=NAVY, width=2.0,
                name=f"(D/R)ⁿ/{nint}")
        valid = list(CLUSTERS)
        sims = [float(np.percentile(sir_samples(k, round(p.n, 3), round(p.sigma, 2), sect), 5))
                for k in valid]
        pv.scatter("sim", valid, sims, color=RED, size=8, name="simulated worst 5 %")
        pv.hline("18", 18, color=GREEN, style=":", label="AMPS target 18 dB", label_pos=0.62)
        form = float(db(cel.reuse_ratio(N) ** p.n / nint))
        pv.scatter("now", [N], [form], color=ORANGE, size=15, symbol="d", outline=ORANGE)
        pv.set_xticks([(k, str(k)) for k in valid])
        self.ref7 = float(np.percentile(sir_samples(7, round(p.n, 3), round(p.sigma, 2), 1), 5))
        self.readout(form=form, p5=p5, med=float(np.median(s)), chan=p.total // N)

    def story(self, p):
        N = int(p.N)
        D = np.sqrt(3 * N)
        r = self.r
        s = (f"<p>Every colour on the map is one set of channels; the numbered cells marked "
             f"<b>1</b> (red) all use the same set as the centre cell. With N = {v(N, 'd')} the "
             f"nearest of them are D = √(3N)·R = {v(D, '.2f')} cell radii away (dashed lines): "
             f"their signals arrive {v(10 * p.n * np.log10(max(D - 1, 0.5)), '.0f', 'dB')} or so "
             f"weaker than the wanted one at the cell edge.</p>")
        s += (f"<p>The formula says {v(r.get('form', 0), '.1f', 'dB')}; the simulation, with "
              f"{v(p.sigma, '.0f', 'dB')} of shadowing, says the worst 5 % of users sit below "
              f"{v(r.get('p5', 0), '.1f', 'dB')}. ")
        if p.sigma > 3:
            s += ("Shadowing spreads the curve: some users stand behind a building on the wanted "
                  "path and in the open towards an interferer. ")
        s += (f"The price of distance: each cell gets only {v(p.total // N, 'd')} of the "
              f"{v(p.total, 'd')} channels.</p>")
        if N == 1:
            s += ("<p>Reuse 1 hands every cell the whole band, so the edge users fight their "
                  "neighbours at about 0 dB. CDMA, LTE and NR live there on purpose and recover "
                  "with coding, power control, scheduling and interference coordination.</p>")
        elif p.sect == "3 sectors":
            s += ("<p>Three sectors per site: each sector antenna sees only the interferers in "
                  "front of it, and the pattern's 20 dB front-to-back ratio mutes the rest. GSM "
                  "used this to run N = 3 or 4 instead of AMPS's 7.</p>")
        return "<h3>Distance is the only shield</h3>" + s + keybox(
            "SIR ≈ (D/R)ⁿ / 6 = (3N)^(n/2) / 6: a larger cluster buys interference margin "
            "and costs capacity per cell. Every generation since has fought to shrink N.")


# =============================================================================== 2. Erlang B
GOS = {"1 %": 0.01, "2 %": 0.02, "5 %": 0.05}


def _min_c_util(target=0.8, gos=0.02):
    for C in range(1, 400):
        if cel.erlang_b_capacity(C, gos) * (1 - gos) / C >= target:
            return C
    return 400


C80 = _min_c_util()


@lru_cache(maxsize=8)
def trunk_efficiency(gos):
    """Utilisation (%) of a single trunk of C = 1..120 channels at blocking ``gos``: one pass of
    the Erlang B recursion over a fine traffic grid gives every C at once."""
    A = np.logspace(-3, np.log10(160), 6000)
    B = np.ones_like(A)
    out = []
    for c in range(1, 121):
        B = A * B / (c + A * B)
        out.append(np.interp(np.log(gos), np.log(np.maximum(B, 1e-300)), A) * (1 - gos) / c)
    return 100 * np.array(out)


class ErlangSwitchboard(Experiment):
    title = "Trunking: a live switchboard"
    blurb = "Random calls on a bundle of channels: Erlang's law, watched as it happens."
    book = "sec:ch20:traffic"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("The trunk"),
        IntSlider("C", "Channels", 1, 60, 10, help="Lines in the trunk (traffic channels in a cell)"),
        IntSlider("k", "Split into separate groups", 1, 6, 1,
                  help="Divide the channels into k groups, each serving 1/k of the callers "
                       "(like sectors that cannot borrow from each other)"),
        Heading("The callers"),
        LogSlider("A", "Offered traffic", 0.2, 60, 5.0, unit="erlangs", fmt=".2f",
                  help="Erlangs = call arrival rate × mean call duration"),
        Slider("hold", "Mean call duration", 1, 10, 3, step=0.5, unit="min"),
        Choice("gos", "Grade of service", list(GOS), "2 %",
               help="The blocking probability the network is designed for"),
    ]
    plots = [
        ImagePlot("board", "The switchboard: each row is a channel, each block a call",
                  x="time (mean call durations, 0 = now)", y="channel"),
        Plot("erl", "Erlang B: blocking vs offered traffic", x="offered traffic (erlangs)",
             y="blocking probability", logx=True, logy=True, xlim=(0.2, 100), ylim=(1e-4, 1),
             legend="tl"),
        Plot("eff", "Big trunks are efficient", x="channels C",
             y="channels busy at the GoS (%)", xlim=(0, 120), ylim=(0, 100), legend="br"),
    ]
    layout = [["board", "board"], ["erl", "eff"]]
    row_stretch = [4, 3]
    readouts = [
        Readout("B", "Blocking (Erlang B)", "%", ".2f"),
        Readout("sim", "Blocking, 20 000 simulated calls", "%", ".2f"),
        Readout("cap", "Traffic carried at the GoS", "erlangs", ".1f"),
        Readout("util", "Channel utilisation at the GoS", "%", ".0f", good=lambda x: x >= 80),
    ]
    challenges = [
        Challenge("A GSM cell with three carriers has 22 traffic channels. Find the most "
                  "traffic it carries at 2 % blocking (within 0.3 erlang).",
                  lambda s: (s.p.C == 22 and s.p.k == 1 and s.p.gos == "2 %"
                             and 14.6 <= s.p.A and s.r.B <= 2.0),
                  hint="Raise the traffic until the blocking readout just reaches 2 %."),
        Challenge(f"Find the smallest single trunk that is at least 80 % busy at 2 % blocking.",
                  lambda s: s.p.C == C80 and s.p.k == 1 and s.p.gos == "2 %"),
        Challenge("Split 30 channels into 3 groups offered 21 erlangs in all, and watch blocking "
                  "grow more than five-fold compared with one trunk.",
                  lambda s: (s.p.C == 30 and s.p.k == 3 and abs(s.p.A - 21) < 0.6
                             and s.r.B > 5 * s.exp.B_one),
                  hint="Same channels, same callers: only the pooling changes."),
    ]
    W = 160
    dt = 0.025            # mean holding times per board column

    def setup(self):
        self.reset_board(None)

    def reset_board(self, p):
        self.t = 0.0
        self.hist = np.zeros((60, self.W))
        self.ends = np.full(60, -1.0)
        self.ids = np.zeros(60, int)
        self.calls = 0
        self.blocked = []

    def groups(self, p):
        C, k = p.C, min(p.k, p.C)
        per = C // k
        return [(g * per, (g + 1) * per) for g in range(k)], per * k

    def advance(self, p, ncol):
        grp, Cu = self.groups(p)
        rng = self.rng
        for _ in range(ncol):
            t0 = self.t
            na = rng.poisson(p.A * self.dt)
            for ta in np.sort(t0 + self.dt * rng.random(na)):
                a, b = grp[rng.integers(0, len(grp))]
                free = np.flatnonzero(self.ends[a:b] <= ta)
                self.calls += 1
                if len(free):
                    ch = a + free[0]
                    self.ends[ch] = ta + rng.exponential(1.0)
                    self.ids[ch] = 1 + self.calls % 5
                else:
                    self.blocked.append(ta)
            self.t = t0 + self.dt
            col = np.where(self.ends > t0 + self.dt / 2, self.ids, 0)
            col[Cu:] = 0
            self.hist[:, :-1] = self.hist[:, 1:]
            self.hist[:, -1] = col
        self.blocked = [b for b in self.blocked if b > self.t - self.W * self.dt]

    def long_run(self, p):
        """Event-driven loss system: 20 000 calls (fewer in the self-test)."""
        grp, _ = self.groups(p)
        n = 4000 if self.quick else 20000
        rng = np.random.default_rng(11)
        arr = np.cumsum(rng.exponential(1 / p.A, n))
        hold = rng.exponential(1.0, n)
        g = rng.integers(0, len(grp), n)
        heaps = [[] for _ in grp]
        cap = [b - a for a, b in grp]
        blk = 0
        for ta, h, gi in zip(arr, hold, g):
            hp = heaps[gi]
            while hp and hp[0] <= ta:
                heapq.heappop(hp)
            if len(hp) < cap[gi]:
                heapq.heappush(hp, ta + h)
            else:
                blk += 1
        return blk / n

    def update(self, p):
        self.reset_board(p)
        self.ends[:] = -1
        self.advance(p, self.W + 40)
        gos = GOS[p.gos]
        grp, Cu = self.groups(p)
        k = len(grp)
        per = Cu // k
        B = float(cel.erlang_b(p.A / k, per))
        self.B_one = float(cel.erlang_b(p.A, p.C))
        cap = k * cel.erlang_b_capacity(per, gos)
        util = 100 * cap * (1 - gos) / max(Cu, 1)
        self.sim = self.long_run(p)
        self.cur = dict(B=B, k=k, per=per, Cu=Cu, util=util, gos=gos)
        self.readout(B=100 * B, sim=100 * self.sim, cap=cap, util=util)
        self.draw_curves(p)
        self.draw_board(p)

    def draw_curves(self, p):
        B, k, per, Cu, util, gos = (self.cur[x] for x in ("B", "k", "per", "Cu", "util", "gos"))
        A = np.logspace(np.log10(0.2), 2, 300)
        pe = self.plot("erl")
        for c in (2, 5, 10, 20, 50):
            if c == p.C and k == 1:
                continue
            pe.line(f"c{c}", A, cel.erlang_b(A, c), color=GRAY, width=1.0, alpha=0.6)
            ax = A[np.argmin(np.abs(cel.erlang_b(A, c) - 2e-4))]
            pe.text(f"t{c}", max(ax, 0.32), 2e-4, f"C={c}", color=GRAY, size=8, anchor=(0.5, 1))
        lab = f"C = {p.C}" if k == 1 else f"{k} groups of {per}"
        pe.line("cur", A, cel.erlang_b(A / k, per), color=NAVY, width=2.6, name=lab)
        pe.hline("gos", gos, color=GREEN, style=":", label=f"GoS {p.gos}", label_pos=0.82)
        pe.scatter("now", [p.A], [max(B, 1.2e-4)], color=RED, size=13, symbol="d", name="you")
        if self.sim > 0:
            pe.scatter("sim", [p.A], [self.sim], color=ORANGE, size=9, outline=ORANGE,
                       name="simulated")
        # trunking efficiency
        Cs = np.arange(1, 121)
        pf = self.plot("eff")
        effc = trunk_efficiency(gos)
        pf.line("e", Cs, effc, color=NAVY, width=2.2, name=f"one trunk, GoS {p.gos}")
        pf.scatter("now", [Cu], [util], color=RED, size=13, symbol="d",
                   name="yours" if k == 1 else f"yours: {k} groups")
        pf.hline("80", 80, color=GREEN, style=":", width=1.0)

    def tick(self, p):
        self.advance(p, 3)
        self.draw_curves(p)
        self.draw_board(p)

    def draw_board(self, p):
        C = p.C
        pb = self.plot("board")
        dark = pb.theme.name == "dark"
        cols = ["#1E252E" if dark else "#EEF2F7"] + (pastels(5, dark=False, offset=0.08) if not dark
                                                    else pastels(5, dark=True, offset=0.08))
        img = self.hist[:C]
        span = self.W * self.dt
        pb.image("img", img, x=(-span, 0), y=(0, C), cmap=cat_cmap(cols), levels=(-0.5, 5.5))
        grp, _ = self.groups(p)
        for g, (a, b) in enumerate(grp[1:]):
            pb.hline(f"g{g}", a, color=NAVY, style="-", width=1.6)
        bl = np.array(self.blocked) - self.t
        if len(bl):
            pb.scatter("blk", bl, np.full(len(bl), C + 0.55), color=RED, size=10, symbol="x")
        else:
            pb.scatter("blk", [], [], color=RED)
        busy = int(np.sum(self.hist[:C, -1] > 0))
        pb.set_title(f"The switchboard: {busy} of {C} channels busy now (✕ = blocked call)")
        pb.set_xlim(-span, 0)
        pb.set_ylim(0, C + 1.6)
        step = max(1, C // 10)
        pb.set_yticks([(c + 0.5, str(c + 1)) for c in range(0, C, step)])

    def story(self, p):
        r = self.r
        gos = GOS[p.gos]
        s = (f"<p>Calls arrive at random (Poisson) and last a random time, on average "
             f"{v(p.hold, '.1f', 'min')}: {v(p.A, '.2f')} erlangs means "
             f"{v(60 * p.A / p.hold, '.0f')} calls per hour. Each coloured block on the board is "
             f"one call holding one channel; a red ✕ marks a caller who found every line busy.</p>")
        s += (f"<p>Erlang's 1917 formula gives the blocking without simulating anything: "
              f"{v(r.get('B', 0), '.2f', '%')}, and 20 000 simulated calls agree "
              f"({v(r.get('sim', 0), '.2f', '%')}). At the {p.gos} grade of service this trunk "
              f"carries {v(r.get('cap', 0), '.1f')} erlangs: its channels are busy only "
              f"{v(r.get('util', 0), '.0f', '%')} of the time.</p>")
        if p.k > 1:
            s += (f"<p>Split into {v(p.k, 'd')} groups that cannot help each other, the same "
                  f"channels block {v(r.get('B', 0) / max(100 * self.B_one, 1e-9), '.1f')}× as "
                  f"often as one pooled trunk: one group overflows while another sits idle. "
                  f"This <b>trunking loss</b> is the hidden cost of sectors and of dividing "
                  f"spectrum between operators.</p>")
        else:
            s += ("<p>Drag the channel count up at a fixed grade of service and watch the "
                  "utilisation climb (bottom right): a big trunk smooths out the randomness. "
                  "Ten channels carry about 5 erlangs at 2 %; a hundred carry about 88.</p>")
        return "<h3>Erlang's law</h3>" + s + keybox(
            "B(A, C) = (A^C/C!) / Σₖ Aᵏ/k!: blocking depends only on the offered traffic and the "
            "number of channels, and large pooled trunks are far more efficient than small ones.")


# =============================================================================== 3. OFDMA scheduling
POLICIES = ["Round robin", "Max C/I", "Proportional fair"]


class OFDMAScheduling(Experiment):
    title = "OFDMA scheduling, live"
    blurb = "Users walk around a cell; the scheduler hands out resource blocks every slot."
    book = "sec:ch20:scheduling"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("Scheduler"),
        Choice("pol", "Policy", POLICIES, "Proportional fair", style="menu"),
        Slider("alpha", "Fairness exponent α", 0.0, 3.0, 1.0, step=0.05,
               help="PF metric: instantaneous rate / (average rate)^α. α = 0 is max-rate, "
                    "α = 1 classic PF, large α approaches max-min fairness",
               enabled_if=lambda p: p.pol == "Proportional fair"),
        Heading("Cell"),
        IntSlider("U", "Active users", 2, 12, 8),
        Slider("edge", "SNR at the cell edge", -5, 20, 0, step=0.5, unit="dB"),
        Slider("speed", "User speed", 1, 120, 30, step=1, unit="km/h",
               help="Sets the Doppler (carrier 3.5 GHz): how fast each user's fading changes"),
        Toggle("walk", "Users walk around", True),
        Button("drop", "New user drop"),
    ]
    plots = [
        Plot("cell", "The cell (users sized by their share)", x="", y="", aspect=True,
             grid=False, legend=None, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2)),
        ImagePlot("grid", "Who gets each resource block (last 60 slots)",
                  x="slot (0 = now)", y="resource block"),
        BarPlot("thr", "Throughput per user", x="user", y="throughput (Mb/s)", legend="tr"),
        Plot("trade", "The trade-off", x="Jain's fairness index", y="cell throughput (b/s/Hz)",
             xlim=(0, 1.05), legend=None),
    ]
    layout = [["cell", "grid"], ["thr", "trade"]]
    col_stretch = [2, 3]
    readouts = [
        Readout("tot", "Cell throughput", "b/s/Hz", ".2f"),
        Readout("jain", "Jain's fairness", "", ".2f", good=lambda x: x >= 0.6),
        Readout("worst", "Worst user", "Mb/s", ".2f"),
        Readout("vsmax", "Of max C/I's throughput", "%", ".0f"),
    ]
    challenges = [
        Challenge("Edge SNR of 10 dB or less: with proportional fair, keep at least 75 % of max "
                  "C/I's throughput with a Jain index of 0.8 or more.",
                  lambda s: (s.p.pol == "Proportional fair" and s.p.edge <= 10 and s.p.U >= 6
                             and s.r.vsmax >= 75 and s.r.jain >= 0.8),
                  hint="Raise the edge SNR towards 10 dB, then tune α: 0 is max C/I, large α "
                       "chases fairness."),
        Challenge("Starve the cell edge: make the worst user get under 1 % of the mean "
                  "throughput.",
                  lambda s: s.exp.worst_share < 0.01,
                  hint="Max C/I with a weak edge."),
        Challenge("With a 0 dB edge, find an α where proportional fair beats round robin's "
                  "Jain index by 0.25 or more and still carries more traffic.",
                  lambda s: (s.p.pol == "Proportional fair" and s.p.edge <= 0
                             and s.exp.stats[2]["jain"] >= s.exp.stats[0]["jain"] + 0.25
                             and s.exp.stats[2]["tot"] > s.exp.stats[0]["tot"])),
    ]
    NRB = 24
    WIN = 60
    TAU = np.arange(6) * 0.4e-6           # tap delays (s): a 2 µs delay spread
    PDP = np.exp(-np.arange(6) / 2.0)
    SLOT = 0.5e-3

    def setup(self):
        self.on_drop(None)

    def on_drop(self, p):
        r = self.rng
        n = 12
        rad = np.sqrt(r.uniform(0.15 ** 2, 1, n))
        self.pos = rad * np.exp(2j * np.pi * r.random(n))
        self.vel = np.exp(2j * np.pi * r.random(n))
        self.g = (r.standard_normal((n, 6)) + 1j * r.standard_normal((n, 6))) * \
            np.sqrt(self.PDP / self.PDP.sum() / 2)

    def reset_stats(self, U):
        self.avg = np.full((3, U), 1e-3)
        self.served = np.zeros((3, U))
        self.nslot = 0
        self.slotidx = 0
        self.gridimg = np.zeros((self.NRB, self.WIN))

    def mean_snr_db(self, p, U):
        return np.minimum(p.edge + 35 * np.log10(1 / np.abs(self.pos[:U])), 35.0)

    def run(self, p, nslots):
        U = p.U
        r = self.rng
        fd = p.speed / 3.6 / (3e8 / 3.5e9)
        rho = float(j0(2 * np.pi * fd * self.SLOT))
        f = np.arange(self.NRB) * 360e3
        E = np.exp(-2j * np.pi * np.outer(self.TAU, f))
        sd = np.sqrt(self.PDP / self.PDP.sum() / 2)
        snr0 = 10 ** (self.mean_snr_db(p, U) / 10)
        alpha = p.alpha
        rbi = np.arange(self.NRB)
        show = POLICIES.index(p.pol)
        for _ in range(nslots):
            w = (r.standard_normal((12, 6)) + 1j * r.standard_normal((12, 6))) * sd
            self.g = rho * self.g + np.sqrt(1 - rho ** 2) * w
            H2 = np.abs(self.g[:U] @ E) ** 2
            R = np.minimum(np.log2(1 + snr0[:, None] * H2 / 2), 7.4)
            picks = [(self.slotidx * self.NRB + rbi) % U,
                     np.argmax(R, axis=0),
                     np.argmax(R / self.avg[2][:, None] ** alpha, axis=0)]
            for k, pk in enumerate(picks):
                inst = np.bincount(pk, weights=R[pk, rbi], minlength=U)
                self.avg[k] = 0.98 * self.avg[k] + 0.02 * inst
                self.served[k] += inst
            self.gridimg[:, :-1] = self.gridimg[:, 1:]
            self.gridimg[:, -1] = picks[show]
            self.slotidx += 1
            self.nslot += 1

    def walk(self, p):
        if not p.walk:
            return
        step = 0.012
        self.pos = self.pos + step * self.vel
        a = np.abs(self.pos)
        out = (a > 0.98) | (a < 0.12)
        if out.any():
            ang = np.angle(self.pos[out])
            inward = np.where(a[out] > 0.98, ang + np.pi, ang)
            self.vel[out] = np.exp(1j * (inward + self.rng.uniform(-1.0, 1.0, out.sum())))
            self.pos[out] = np.clip(a[out], 0.13, 0.97) * np.exp(1j * ang)

    def update(self, p):
        self.reset_stats(p.U)
        self.run(p, 60 if self.quick else 300)
        self.draw(p)

    def tick(self, p):
        self.walk(p)
        self.run(p, 6)
        self.draw(p)

    def draw(self, p):
        U = p.U
        thr = self.served / max(self.nslot, 1) * 0.36      # Mb/s: each RB is 360 kHz
        tot = self.served.sum(axis=1) / max(self.nslot, 1) / self.NRB
        self.stats = [dict(tot=float(tot[k]), jain=float(cel.jain(thr[k] + 1e-12))) for k in range(3)]
        k = POLICIES.index(p.pol)
        mine = thr[k]
        share = mine / max(mine.sum(), 1e-12)
        self.worst_share = float(mine.min() / max(mine.mean(), 1e-12))
        # cell map
        pc = self.plot("cell")
        th = np.linspace(0, 2 * np.pi, 200)
        pc.line("edge", np.cos(th), np.sin(th), color=GRAY, width=1.4, style="--")
        pc.scatter("bs", [0], [0], color=NAVY, size=14, symbol="t")
        snr = self.mean_snr_db(p, U)
        for u in range(12):
            if u < U:
                z = self.pos[u]
                pc.scatter(f"u{u}", [z.real], [z.imag], color=USER_COLS[u],
                           size=8 + 60 * share[u] ** 0.7)
                pc.text(f"t{u}", z.real + 0.06, z.imag + 0.06, f"{u + 1}", size=8.5,
                        color=USER_COLS[u], bold=True, anchor=(0, 1))
        pc.text("snr", -1.15, -1.12, f"edge SNR {p.edge:.0f} dB · mean SNR "
                f"{snr.min():.0f}…{snr.max():.0f} dB", size=8.5, anchor=(0, 1), color=GRAY)
        pc.set_xticks([])
        pc.set_yticks([])
        # grid
        pg_ = self.plot("grid")
        pg_.image("img", self.gridimg, x=(-self.WIN, 0), y=(0, self.NRB),
                  cmap=cat_cmap(USER_COLS), levels=(-0.5, 11.5))
        pg_.set_xlim(-self.WIN, 0)
        pg_.set_ylim(0, self.NRB)
        # throughput bars
        pb = self.plot("thr")
        x = np.arange(1, U + 1)
        pb.bars("b", x, mine, width=0.62, colors=USER_COLS[:U], name=p.pol)
        if k != 0:
            pb.scatter("rr", x, thr[0], color=GRAY, size=9, outline=GRAY, name="round robin")
        top = 1.3 * max(mine.max(), thr[0].max(), np.median(thr[1]), 0.1)
        if k != 1:
            pb.scatter("mx", x, np.minimum(thr[1], 0.97 * top), color=ORANGE, size=10,
                       symbol="d", name="max C/I (clipped)" if thr[1].max() > top else "max C/I")
        pb.set_xticks([(i, str(i)) for i in x])
        pb.set_xlim(0.4, U + 0.6)
        pb.set_ylim(0, top)
        # trade-off
        pt = self.plot("trade")
        cols = [GRAY, ORANGE, NAVY]
        for i, nm in enumerate(POLICIES):
            st_ = self.stats[i]
            pt.scatter(f"p{i}", [st_["jain"]], [st_["tot"]], color=cols[i],
                       size=18 if i == k else 11, symbol="d" if i == k else "o")
            pt.text(f"l{i}", st_["jain"], st_["tot"], " " + nm, color=cols[i], size=8.5,
                    anchor=(0, 1.0 if i != 1 else 0.0), bold=i == k)
        ymax = max(s_["tot"] for s_ in self.stats)
        pt.set_ylim(0, 1.35 * ymax + 0.1)
        self.readout(tot=self.stats[k]["tot"], jain=self.stats[k]["jain"],
                     worst=float(mine.min()), vsmax=100 * self.stats[k]["tot"] / max(self.stats[1]["tot"], 1e-9))

    def story(self, p):
        r = self.r
        s = (f"<p>{v(p.U, 'd')} users share {self.NRB} resource blocks. Every 0.5 ms slot the "
             f"scheduler gives each block to one user (the coloured columns, newest on the "
             f"right). Each user sees its own frequency-selective fading, changing at "
             f"{v(p.speed, '.0f', 'km/h')}, on top of its distance from the mast.</p>")
        if p.pol == "Round robin":
            s += ("<p><b>Round robin</b> takes turns blindly: perfectly fair in time, but it "
                  "often serves a user in a fade while another sits on a peak.</p>")
        elif p.pol == "Max C/I":
            s += (f"<p><b>Max C/I</b> always picks the best channel: the most bits per slot, "
                  f"but the users near the mast win almost every block and the far ones "
                  f"starve (worst user {v(r.get('worst', 0), '.2f', 'Mb/s')}).</p>")
        else:
            s += (f"<p><b>Proportional fair</b> (α = {v(p.alpha, '.2f')}) schedules each user "
                  f"when its channel is good <i>relative to its own average</i>: everyone rides "
                  f"their own fading peaks. That is <b>multi-user diversity</b>: here it keeps "
                  f"{v(r.get('vsmax', 0), '.0f', '%')} of max C/I's throughput with a Jain index "
                  f"of {v(r.get('jain', 0), '.2f')}.</p>")
        return "<h3>Riding the peaks</h3>" + s + keybox(
            "Fading is not only a problem: with many users, someone is always on a peak. A "
            "scheduler that picks rate / average-rate^α trades total throughput against fairness.")


# =============================================================================== 4. link adaptation
class LinkAdaptation(Experiment):
    title = "Link adaptation and HARQ"
    blurb = "Pick the MCS from a stale CQI, let HARQ mop up: how close to Shannon?"
    book = "sec:ch21:nrcoding"
    controls = [
        Heading("Channel"),
        Slider("snr", "Mean SNR", -5, 30, 12, step=0.5, unit="dB"),
        Slider("speed", "Speed", 1, 150, 30, step=1, unit="km/h",
               help="Rayleigh fading at 3.5 GHz: 30 km/h is about 100 Hz of Doppler"),
        Heading("Link adaptation"),
        IntSlider("delay", "CQI report delay", 0, 20, 4, unit="slots",
                  help="Age of the channel report when the MCS is chosen (0.5 ms slots)"),
        Slider("margin", "Back-off margin", 0, 10, 0, step=0.25, unit="dB",
               help="Pick the MCS for an SNR this much lower than reported"),
        Toggle("olla", "Outer-loop correction (OLLA)", False,
               help="Nudge the margin up on every NACK, down on every ACK, to hold 10 % BLER"),
        Heading("HARQ"),
        Choice("comb", "Retransmissions", ["Chase", "Incremental red."],
               help="Chase repeats the same bits; incremental redundancy sends new parity"),
        IntSlider("maxtx", "Maximum transmissions", 1, 4, 4),
        Button("again", "New fading trace"),
    ]
    plots = [
        Plot("trace", "The channel and the CQI the scheduler sees (first 300 ms)",
             x="time (ms)", y="SNR (dB)", xlim=(0, 300), legend="tl", legend_cols=2),
        Plot("stair", "NR MCS table: what each MCS needs", x="SNR (dB)",
             y="spectral efficiency (b/s/Hz)", xlim=(-10, 30), ylim=(0, 7.5), legend="br"),
        Plot("mcs", "MCS chosen (red ✕ = NACK)", x="time (ms)", y="MCS index",
             xlim=(0, 300), ylim=(-1, 30), legend=None),
        BarPlot("hist", "When was each block decoded?", y="transport blocks (%)"),
    ]
    layout = [["trace", "stair"], ["mcs", "hist"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("tput", "Throughput", "b/s/Hz", ".2f"),
        Readout("shan", "Of the Shannon capacity", "%", ".0f", good=lambda x: x >= 70),
        Readout("bler1", "First-transmission BLER", "%", ".1f"),
        Readout("resid", "Residual BLER", "%", ".2f", good=lambda x: x < 1),
    ]
    challenges = [
        Challenge("By hand (outer loop off), set the back-off so the first-transmission BLER "
                  "lands between 8 and 12 % with an 8-slot CQI delay.",
                  lambda s: (not s.p.olla and s.p.delay >= 8 and 8 <= s.r.bler1 <= 12
                             and abs(s.p.snr - 12) < 0.01 and s.p.speed >= 30),
                  hint="Keep the mean SNR at 12 dB and the speed at 30 km/h or more."),
        Challenge("At 120 km/h with a 10-slot delay, get at least 53 % of Shannon.",
                  lambda s: s.p.speed >= 120 and s.p.delay >= 10 and s.r.shan >= 53,
                  hint="Stale CQI at high speed: HARQ with incremental redundancy turns "
                       "over-optimism into extra parity instead of lost blocks."),
        Challenge("No HARQ (one transmission): hold the residual BLER under 5 % at 30 km/h or "
                  "faster, and see what it costs in throughput.",
                  lambda s: s.p.maxtx == 1 and s.r.resid < 5 and s.p.speed >= 30),
    ]
    T = 2000
    GAP = 2.0
    SLOT = 0.5e-3

    def setup(self):
        self.seed = 3

    def on_again(self, p):
        self.seed += 1

    def fading(self, p):
        rr = np.random.default_rng(self.seed)
        fd = p.speed / 3.6 / (3e8 / 3.5e9)
        t = np.arange(self.T) * self.SLOT
        n = 16
        al = 2 * np.pi * (np.arange(n) + rr.random(n)) / n
        ph = 2 * np.pi * rr.random(n)
        h = np.exp(1j * (2 * np.pi * fd * np.cos(al)[:, None] * t[None, :] + ph[:, None])).sum(0)
        return np.abs(h) ** 2 / n, rr

    def simulate(self, p):
        g2, rr = self.fading(p)
        snr_lin = 10 ** (p.snr / 10) * g2
        snr_db = db(snr_lin)
        se = ai.spectral_efficiency(ai.NR_MCS64)
        req = ai.required_snr_db(se, self.GAP)
        g = 10 ** (self.GAP / 10)
        u = rr.random(self.T)
        ir = p.comb.startswith("Incr")
        olla = 0.0
        k = 0
        acc = 0.0
        cur = 0
        delivered = 0.0
        first = first_err = 0
        done_at = []
        mcs = np.zeros(self.T, int)
        rep = np.zeros(self.T)
        nack = []
        for t in range(self.T):
            if k == 0:
                rep[t] = snr_db[max(t - p.delay, 0)] + olla - p.margin
                cur = max(ai.select_entry(rep[t], req), 0)
                acc = 0.0
            else:
                rep[t] = rep[t - 1]
            k += 1
            s = snr_lin[t]
            if ir:
                acc += np.log2(1 + s / g)
                eff = db(g * (2.0 ** acc - 1))
            else:
                acc += s
                eff = db(acc)
            pe = float(ai.bler_logistic(eff, req[cur]))
            ok = u[t] > pe
            mcs[t] = cur
            if k == 1:
                first += 1
                first_err += not ok
                if p.olla:
                    olla += 0.5 * 0.1 / 0.9 if ok else -0.5
            if not ok:
                nack.append(t)
            if ok:
                delivered += se[cur]
                done_at.append(k)
                k = 0
            elif k >= p.maxtx:
                done_at.append(0)
                k = 0
        done_at = np.array(done_at)
        return dict(snr_db=snr_db, rep=rep, mcs=mcs, nack=np.array(nack, int), se=se, req=req,
                    tput=delivered / self.T, shan=float(np.mean(np.log2(1 + snr_lin))),
                    bler1=first_err / max(first, 1), resid=float(np.mean(done_at == 0)) if len(done_at) else 0.0,
                    hist=[float(np.mean(done_at == k_)) for k_ in (1, 2, 3, 4)] + [float(np.mean(done_at == 0))])

    def update(self, p):
        R = self.simulate(p)
        tm = np.arange(self.T) * self.SLOT * 1e3
        w = tm <= 300
        pt = self.plot("trace")
        pt.line("snr", tm[w], R["snr_db"][w], color=GRAY, width=1.2, name="SNR now")
        pt.line("rep", tm[w], R["rep"][w], color=ORANGE, width=1.6, style="--",
                name="SNR the scheduler uses")
        pt.set_ylim(min(-15, p.snr - 25), max(40, p.snr + 22))
        pm = self.plot("mcs")
        pm.line("m", tm[w], R["mcs"][w], color=NAVY, width=1.8)
        nk = R["nack"][R["nack"] < w.sum()]
        if len(nk):
            pm.scatter("n", tm[nk], R["mcs"][nk], color=RED, size=8, symbol="x")
        else:
            pm.scatter("n", [], [], color=RED)
        # staircase
        ps = self.plot("stair")
        x = np.linspace(-10, 30, 401)
        ps.line("sh", x, np.log2(1 + 10 ** (x / 10)), color=GRAY, width=1.4, style="--",
                name="Shannon")
        idx = np.searchsorted(R["req"], x, side="right") - 1
        ps.line("st", x, np.where(idx >= 0, R["se"][np.clip(idx, 0, None)], 0) * 0.9,
                color=NAVY, width=2.0, name="best MCS × 0.9")
        q = np.array([qm for qm, _ in ai.NR_MCS64])
        for qm, col, nm in ((2, GREEN, "QPSK"), (4, ORANGE, "16QAM"), (6, RED, "64QAM")):
            m = q == qm
            ps.scatter(f"q{qm}", R["req"][m], R["se"][m], color=col, size=6, name=nm)
        ps.scatter("op", [p.snr], [R["tput"]], color=PURPLE, size=16, symbol="d",
                   name="you (mean SNR)")
        # outcome histogram
        ph = self.plot("hist")
        hv = 100 * np.array(R["hist"])
        ph.bars("h", np.arange(5), hv, width=0.62, colors=[GREEN, TEAL, BLUE, PURPLE, RED])
        for i, val in enumerate(hv):
            ph.text(f"t{i}", i, val, f"{val:.1f}", anchor=(0.5, 1.05), size=8.5, bold=True)
        ph.set_xticks([(0, "1st tx"), (1, "2nd"), (2, "3rd"), (3, "4th"), (4, "lost")])
        ph.set_xlim(-0.6, 4.6)
        ph.set_ylim(0, 118)
        self.res = R
        self.readout(tput=R["tput"], shan=100 * R["tput"] / max(R["shan"], 1e-9),
                     bler1=100 * R["bler1"], resid=100 * R["resid"])

    def story(self, p):
        r = self.r
        R = getattr(self, "res", None)
        if R is None:
            return ""
        s = (f"<p>The phone reports its SNR; {v(p.delay, 'd')} slots "
             f"({v(0.5 * p.delay, '.1f', 'ms')}) later the base station picks the highest MCS "
             f"whose requirement the report meets (the dashed orange line is what it believes). "
             f"At {v(p.speed, '.0f', 'km/h')} the fading has moved on by then, so some "
             f"choices are too greedy (red ✕: NACK) and some too timid.</p>")
        s += (f"<p>Result: {v(r.get('tput', 0), '.2f', 'b/s/Hz')}, "
              f"{v(r.get('shan', 0), '.0f', '%')} of the channel's Shannon capacity, with "
              f"{v(r.get('bler1', 0), '.1f', '%')} of first transmissions failing. ")
        if p.maxtx > 1:
            s += (f"HARQ ({'incremental redundancy' if p.comb.startswith('Incr') else 'chase combining'}, "
                  f"up to {p.maxtx} transmissions) rescues most of them: only "
                  f"{v(r.get('resid', 0), '.2f', '%')} are lost.</p>")
        else:
            s += "With no HARQ every failed block is lost to the higher layers.</p>"
        if p.olla:
            s += ("<p>The <b>outer loop</b> shifts the reported SNR down by 0.5 dB on each NACK "
                  "and up by 0.056 dB on each ACK: it settles where NACKs are one in ten, "
                  "whatever the delay and speed. At high speed that means a large back-off, "
                  "and with fast HARQ it can cost more throughput than it saves.</p>")
        return "<h3>Aim for 10 % and let HARQ catch the rest</h3>" + s + keybox(
            "Link adaptation deliberately runs the first transmission at about 10 % BLER: being a "
            "little greedy and retransmitting beats being safe, as long as HARQ is fast.")


# =============================================================================== 5. Wi-Fi DCF
class WiFiContention(Experiment):
    title = "Wi-Fi contention, live"
    blurb = "Stations count down random backoffs; two reaching zero together collide."
    book = "sec:ch20:random"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("Stations"),
        IntSlider("n", "Saturated stations", 1, 50, 5, help="Every station always has a packet"),
        Choice("cw", "Minimum contention window", ["8", "16", "32", "64", "128"], "16",
               help="Backoff drawn from 0..CW−1 slots; doubled after each collision up to 1024"),
        Toggle("rts", "RTS/CTS handshake", False,
               help="A short RTS/CTS exchange reserves the medium: collisions cost less time"),
        Heading("Packets"),
        Choice("rate", "PHY rate", ["6", "24", "54"], "54", help="802.11a/g rate in Mb/s"),
        IntSlider("payload", "Payload", 100, 2300, 1500, step=50, unit="bytes"),
    ]
    plots = [
        Plot("tl", "The air, slot by slot (green = success, red = collision)",
             x="time (ms, 0 = now)", y="station", legend=None),
        Plot("thr", "Throughput vs number of stations", x="stations", y="throughput (Mb/s)",
             xlim=(0, 51), legend="tr"),
        Plot("col", "Collision probability per attempt", x="stations", y="probability",
             xlim=(0, 51), ylim=(0, 1), legend="br"),
    ]
    layout = [["tl", "tl"], ["thr", "col"]]
    row_stretch = [4, 3]
    readouts = [
        Readout("thr", "Throughput (simulated)", "Mb/s", ".1f"),
        Readout("eff", "Efficiency", "% of PHY rate", ".0f"),
        Readout("pc", "Collision probability", "%", ".0f", good=lambda x: x < 30),
        Readout("bi", "Bianchi's model", "Mb/s", ".1f"),
    ]
    challenges = [
        Challenge("With 30 or more stations, keep the collision probability under 30 % "
                  "(basic access).",
                  lambda s: s.p.n >= 30 and not s.p.rts and s.exp.pc_th < 0.30,
                  hint="Collisions fall as the contention window grows."),
        Challenge("Find a setting where RTS/CTS beats basic access by 5 % or more (Bianchi).",
                  lambda s: s.p.rts and s.exp.th_rts > 1.05 * s.exp.th_basic,
                  hint="RTS/CTS pays when collisions are frequent and packets long."),
        Challenge("Show the cost of politeness: with 2 stations, pick a CWmin that loses at "
                  "least 10 % throughput compared with CWmin = 16.",
                  lambda s: s.p.n == 2 and s.exp.th_now < 0.9 * s.exp.th_cw16),
    ]
    WIN = 8e-3

    def times(self, p):
        return ai.dcf_times(payload=p.payload, rate=int(p.rate), rtscts=p.rts)

    def bianchi(self, n, p, rts=None, W=None):
        slot, pl, Ts, Tc = ai.dcf_times(payload=p.payload, rate=int(p.rate),
                                        rtscts=p.rts if rts is None else rts)
        W = int(p.cw) if W is None else W
        m = int(np.log2(1024 / W))
        S, tau, pc = cel.bianchi_throughput(n, W=W, m=m, slot=slot, Ts=Ts, Tc=Tc, payload_time=pl)
        return S * int(p.rate), pc

    def reset_sim(self, p):
        W = int(p.cw)
        self.cw = np.full(p.n, W)
        self.bo = self.rng.integers(0, W, p.n)
        self.t = 0.0
        self.useful = 0.0
        self.att = self.coll = 0
        self.ev = []

    def step(self, p, nev):
        slot, pl, Ts, Tc = self.times(p)
        W = int(p.cw)
        for _ in range(nev):
            k = self.bo.min()
            self.t += k * slot
            self.bo -= k
            tx = np.flatnonzero(self.bo == 0)
            self.att += len(tx)
            okk = len(tx) == 1
            dur = Ts if okk else Tc
            if okk:
                self.useful += pl
                self.cw[tx] = W
            else:
                self.coll += len(tx)
                self.cw[tx] = np.minimum(2 * self.cw[tx], 1024)
            self.ev.append((self.t, self.t + dur, tx, okk))
            self.t += dur
            self.bo[tx] = self.rng.integers(0, self.cw[tx])
        cut = self.t - self.WIN
        self.ev = [e for e in self.ev if e[1] > cut]

    def update(self, p):
        self.reset_sim(p)
        self.step(p, 400 if self.quick else 1500)
        ns = np.arange(1, 51)
        th = np.array([self.bianchi(n, p) for n in ns])
        alt = np.array([self.bianchi(n, p, rts=not p.rts)[0] for n in ns])
        self.curves = (ns, th, alt)
        self.pc_th = float(th[p.n - 1, 1])
        self.th_now = float(th[p.n - 1, 0])
        self.th_basic = float(self.bianchi(p.n, p, rts=False)[0])
        self.th_rts = float(self.bianchi(p.n, p, rts=True)[0])
        self.th_cw16 = float(self.bianchi(p.n, p, W=16)[0])
        self.bi = self.th_now
        self.draw(p)

    def draw_theory(self, p):
        ns, th, alt = self.curves
        pt = self.plot("thr")
        pt.line("th", ns, th[:, 0], color=NAVY, width=2.2,
                name="Bianchi, " + ("RTS/CTS" if p.rts else "basic"))
        pt.line("alt", ns, alt, color=GRAY, width=1.4, style="--",
                name="Bianchi, " + ("basic" if p.rts else "RTS/CTS"))
        pt.set_ylim(0, 1.25 * max(th[:, 0].max(), alt.max()))
        pc = self.plot("col")
        pc.line("th", ns, th[:, 1], color=NAVY, width=2.2, name="Bianchi")

    def tick(self, p):
        self.step(p, 12)
        self.draw(p)

    def draw(self, p):
        self.draw_theory(p)
        rate = int(p.rate)
        thr = self.useful / max(self.t, 1e-12) * rate
        pcol = self.coll / max(self.att, 1)
        pl_ = self.plot("tl")
        okx, oky, bx, by, mx, my = [], [], [], [], [], []
        now = self.t
        n = p.n
        for t0, t1, tx, okk in self.ev:
            a, b = (t0 - now) * 1e3, (t1 - now) * 1e3
            for s in tx:
                (okx if okk else bx).extend([a, b, np.nan])
                (oky if okk else by).extend([s + 1, s + 1, np.nan])
            mx.extend([a, b, np.nan])
            my.extend([0, 0, np.nan])
        wpx = float(np.clip(260 / max(n, 1), 2, 10))
        for key, xs, ys, col in (("ok", okx, oky, GREEN), ("bad", bx, by, RED),
                                 ("med", mx, my, NAVY)):
            if xs:
                pl_.line(key, xs, ys, color=col, width=wpx)
        pl_.set_xlim(-self.WIN * 1e3, 0)
        pl_.set_ylim(-1, n + 1)
        step = max(1, n // 10)
        pl_.set_yticks([(0, "air")] + [(s, str(s)) for s in range(1, n + 1, step)])
        pt = self.plot("thr")
        pt.scatter("sim", [p.n], [thr], color=RED, size=13, symbol="d", name="simulation")
        pc = self.plot("col")
        pc.scatter("sim", [p.n], [pcol], color=RED, size=13, symbol="d", name="simulation")
        self.readout(thr=thr, eff=100 * thr / rate, pc=100 * pcol, bi=self.bi)

    def story(self, p):
        r = self.r
        slot, pl, Ts, Tc = self.times(p)
        s = (f"<p>No base station decides who talks: each of the {v(p.n, 'd')} stations waits "
             f"for the air to go quiet, then counts down a random number of 9 µs slots between "
             f"0 and its contention window. Whoever reaches zero first transmits "
             f"({v(Ts * 1e6, '.0f', 'µs')} for a {p.payload}-byte packet at {p.rate} Mb/s, with "
             f"its acknowledgement). Two reaching zero in the same slot collide (red), both "
             f"double their window and try again.</p>")
        s += (f"<p>Right now {v(r.get('pc', 0), '.0f', '%')} of attempts collide and the stations "
              f"together get {v(r.get('thr', 0), '.1f', 'Mb/s')}, "
              f"{v(r.get('eff', 0), '.0f', '%')} of the PHY rate. Bianchi's Markov-chain model "
              f"(2000) predicts {v(r.get('bi', 0), '.1f', 'Mb/s')}.</p>")
        if p.n == 1:
            s += ("<p>Even alone a station loses time: the average backoff, the inter-frame "
                  "spaces, the preamble and the ACK are paid on every packet.</p>")
        if p.rts:
            s += (f"<p>With RTS/CTS a collision wastes only a short RTS "
                  f"({v(Tc * 1e6, '.0f', 'µs')} instead of a whole data frame), at the price "
                  f"of the handshake on every success.</p>")
        return "<h3>Listen before talk</h3>" + s + keybox(
            "CSMA/CA is fair and needs no coordinator, but collisions grow with the number of "
            "stations. 802.11ax/be borrow the cellular answer: scheduled OFDMA via trigger frames.")


# =============================================================================== 6. EVM budget
MODS = {"64-QAM": ("64qam", 64), "256-QAM": ("256qam", 256), "1024-QAM": ("1024qam", 1024),
        "4096-QAM": ("4096qam", 4096)}


@lru_cache(maxsize=8)
def constellation(name):
    return cl.get_constellation(name).points


class EVMBudget(Experiment):
    title = "4096-QAM: the EVM budget"
    blurb = "Transmitter error, phase noise and receiver noise add up: who sets the limit?"
    book = "sec:ch22:special"
    controls = [
        Choice("mod", "Modulation", list(MODS), "4096-QAM", style="menu"),
        Heading("Transmitter"),
        Slider("evm", "Transmitter EVM", -48, -20, -36, step=0.5, unit="dB",
               help="Error-vector magnitude of the transmitter alone (PA, DAC, IQ imbalance)"),
        Slider("pn", "Phase noise (rms)", 0.0, 2.0, 0.3, step=0.05, unit="°",
               help="Integrated oscillator phase noise of transmitter plus receiver"),
        Heading("Receiver"),
        Slider("snr", "Receiver SNR", 20, 55, 42, step=0.5, unit="dB"),
        Toggle("zoom", "Zoom on one corner", True),
        Button("again", "New symbols"),
    ]
    plots = [
        ConstellationPlot("const", "Received constellation", lim=1.3),
        BarPlot("bud", "The SNR budget: each impairment alone, and together", y="SNR (dB)"),
        BERPlot("ber", "Uncoded BER vs receiver SNR", x="receiver SNR (dB)", ylim=(1e-7, 0.5),
                xlim=(20, 55), legend="bl"),
    ]
    layout = [["const", "bud"], ["const", "ber"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("eff", "Effective SNR", "dB", ".1f"),
        Readout("meas", "Measured EVM", "dB", ".1f"),
        Readout("ber", "Uncoded BER", "", "sci"),
        Readout("mcs", "Best Wi-Fi 7 MCS", "", None),
    ]
    challenges = [
        Challenge("With a transmitter at exactly the 802.11be limit (−38 dB), find the receiver "
                  "SNR that gives an effective SNR of 35 dB (± 0.3).",
                  lambda s: abs(s.p.evm + 38) < 0.01 and abs(s.r.eff - 35) < 0.3 and s.p.pn < 0.01,
                  hint="Phase noise to zero. 1/SNR_eff = EVM² + 1/SNR_rx."),
        Challenge("Phase-noise budget: with TX EVM −42 dB and SNR 50 dB, find the largest rms "
                  "phase noise that keeps the effective SNR at 35 dB or better.",
                  lambda s: (abs(s.p.evm + 42) < 0.01 and abs(s.p.snr - 50) < 0.01
                             and s.r.eff >= 35 and s.p.pn >= 0.95),
                  hint="A phase error of σ radians acts like an SNR of 1/σ²."),
        Challenge("Unlock MCS 13 (4096-QAM 5/6): effective SNR ≥ 43 dB and a transmitter "
                  "inside its −38 dB EVM limit.",
                  lambda s: isinstance(s.r.mcs, str) and s.r.mcs.startswith("13")),
    ]

    def setup(self):
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        name, M = MODS[p.mod]
        pts = constellation(name)
        rr = np.random.default_rng(self.seed)
        lev = np.unique(np.round(pts.real, 9))
        if p.zoom and M >= 256:
            corner = pts[(pts.real >= lev[-8] - 1e-9) & (pts.imag >= lev[-8] - 1e-9)]
            sym = corner[rr.integers(0, len(corner), 2500)]
            lo = lev[-8] - (lev[1] - lev[0])
            hi = lev[-1] + (lev[1] - lev[0])
            ideal = corner
        else:
            sym = pts[rr.integers(0, M, 6000)]
            lo, hi = -1.3, 1.3
            ideal = pts
        n = len(sym)
        cn = lambda: (rr.standard_normal(n) + 1j * rr.standard_normal(n)) / np.sqrt(2)
        e_tx = 10 ** (p.evm / 20)
        sig_pn = np.deg2rad(p.pn)
        x = sym * np.exp(1j * sig_pn * rr.standard_normal(n)) + e_tx * cn()
        y = x + 10 ** (-p.snr / 20) * cn()
        meas = 10 * np.log10(np.mean(np.abs(y - sym) ** 2) / np.mean(np.abs(sym) ** 2))
        snr_pn = -10 * np.log10(max(sig_pn ** 2, 1e-12))
        eff = float(ai.combine_snr_db(-p.evm, p.snr, snr_pn))
        pc = self.plot("const")
        pc.points("y", y, color=NAVY, size=3, alpha=0.45)
        pc.ideal("ideal", ideal, color=RED, size=7 if len(ideal) <= 64 else 3)
        pc.set_xlim(lo, hi)
        pc.set_ylim(lo, hi)
        pc.set_title("Received constellation" + (" (top-right corner, 64 of the points)"
                                                 if p.zoom and M >= 256 else ""))
        # budget bars
        pb = self.plot("bud")
        vals = [-p.evm, min(snr_pn, 80.0), p.snr, eff]
        names = ["TX EVM", "phase noise", "RX noise", "together"]
        pb.bars("b", np.arange(4), vals, width=0.6, colors=[ORANGE, PURPLE, BLUE, NAVY])
        for i, val in enumerate(vals):
            pb.text(f"t{i}", i, val, f"{val:.1f} dB" if val < 79.9 else "—", anchor=(0.5, 1.05),
                    bold=True, size=9)
        pb.set_xticks([(i, s_) for i, s_ in enumerate(names)])
        pb.set_xlim(-0.6, 3.6)
        pb.set_ylim(0, 75)
        need = ai.WIFI_SNR_REQ[13 if M == 4096 else 11 if M == 1024 else 9 if M == 256 else 7]
        pb.hline("need", need, color=RED, style="--")
        pb.text("needt", 3.55, need, f"top {p.mod} MCS needs {need:.0f} dB", color=RED, size=8.5,
                anchor=(1, 1.1))
        # BER
        x_ = np.linspace(20, 55, 141)
        pber = self.plot("ber")
        pber.theory("ideal", x_, cl.ber_mqam_gray(x_, M), color=GRAY, style="--",
                    name="perfect transmitter")
        effx = ai.combine_snr_db(-p.evm, x_, snr_pn)
        pber.theory("real", x_, cl.ber_mqam_gray(effx, M), color=NAVY,
                    name="with your TX EVM and phase noise")
        ber = float(cl.ber_mqam_gray(eff, M))
        pber.scatter("now", [p.snr], [max(ber, 1e-7)], color=RED, size=13, symbol="d")
        floor = float(cl.ber_mqam_gray(ai.combine_snr_db(-p.evm, 200.0, snr_pn), M))
        if floor > 1e-7:
            pber.hline("floor", floor, color=ORANGE, style=":", label="error floor",
                       label_pos=0.8)
        else:
            pber.hline("floor", 1e-9, color=ORANGE, style=":")
        ok = [m for m in range(14) if ai.WIFI_SNR_REQ[m] <= eff and p.evm <= ai.WIFI_EVM_DB[m]]
        best = f"{max(ok)}: {ai.WIFI_MCS_NAME[max(ok)]}" if ok else "none"
        self.readout(eff=eff, meas=meas, ber=ber, mcs=best)

    def story(self, p):
        r = self.r
        eff = r.get("eff", 0)
        s = (f"<p>Three independent impairments add as powers: 1/SNR<sub>eff</sub> = EVM² + "
             f"σ<sub>φ</sub>² + 1/SNR<sub>rx</sub>. Your transmitter alone would allow "
             f"{v(-p.evm, '.1f', 'dB')}, the phase noise {v(-20 * np.log10(max(np.deg2rad(p.pn), 1e-6)), '.1f', 'dB')}, "
             f"the receiver {v(p.snr, '.1f', 'dB')}; together {v(eff, '.1f', 'dB')}. The worst "
             f"one always dominates, so the budget is a negotiation between the PA designer, "
             f"the synthesizer designer and the antenna/LNA designer.</p>")
        if p.zoom:
            s += ("<p>The zoom shows one corner of the constellation: in 4096-QAM the points sit "
                  "only about 0.04 apart (unit average power), so phase noise smears the outer "
                  "points into arcs while additive noise makes round clouds.</p>")
        s += (f"<p>802.11be allows a 4096-QAM transmitter −38 dB of EVM; even with a perfect "
              f"receiver that caps the effective SNR near 38 dB. The best MCS you can run now is "
              f"{v(str(r.get('mcs', '—')))}.</p>")
        return "<h3>Every decibel is spoken for</h3>" + s + keybox(
            "Dense constellations are limited by the transmitter as much as by the link: "
            "Wi-Fi 7's 4096-QAM needs a cleaner PA and synthesizer, not just a closer access point.")


# =============================================================================== 7. explorer
SYSTEMS = {
    "GSM (2G)": dict(bw=0.2, rate=0.2708, sym=3.69, year=1991, access="TDMA/FDMA",
                     mod="GMSK", note="8 timeslots per 200 kHz carrier, 4.615 ms frame",
                     dim=("time slot", "carrier")),
    "UMTS (3G)": dict(bw=5.0, rate=14.4, sym=0.26, year=2001, access="CDMA",
                      mod="QPSK→16QAM (HSDPA)", note="3.84 Mchip/s, codes share time and "
                                                    "frequency", dim=("time (slots)", "code")),
    "LTE 20 MHz (4G)": dict(bw=20.0, rate=300.0, sym=71.4, year=2009, access="OFDMA",
                            mod="64QAM, 4 layers", note="100 RBs × 15 kHz, 1 ms subframes",
                            dim=("slot (0.5 ms)", "resource block")),
    "NR 100 MHz (5G)": dict(bw=100.0, rate=2337.0, sym=35.7, year=2019, access="OFDMA",
                            mod="256QAM, 4 layers", note="273 RBs × 30 kHz, 0.5 ms slots",
                            dim=("slot (0.5 ms)", "resource block")),
    "Wi-Fi 6, 80 MHz": dict(bw=80.0, rate=1201.0, sym=13.6, year=2019, access="CSMA/CA + OFDMA",
                            mod="1024QAM, 2 streams", note="980 data tones × 78.125 kHz",
                            dim=("time (PPDUs)", "resource unit")),
    "Wi-Fi 7, 320 MHz": dict(bw=320.0, rate=5765.0, sym=13.6, year=2024, access="CSMA/CA + OFDMA",
                             mod="4096QAM, 2 streams", note="3920 data tones, multi-link",
                             dim=("time (PPDUs)", "resource unit")),
    "Bluetooth LE 1M": dict(bw=2.0, rate=1.0, sym=1.0, year=2010, access="FHSS/TDMA",
                            mod="GFSK", note="40 channels × 2 MHz, adaptive hopping",
                            dim=("connection event", "channel")),
    "LoRa SF7, 125 kHz": dict(bw=0.125, rate=0.00547, sym=1024.0, year=2015, access="ALOHA",
                              mod="chirp spread spectrum", note="2^SF chips per symbol",
                              dim=("time (s)", "channel")),
}
METRICS = {"Peak rate": ("rate", "Mb/s"), "Channel bandwidth": ("bw", "MHz"),
           "Spectral efficiency": ("se", "b/s/Hz"), "Symbol duration": ("sym", "µs")}


@lru_cache(maxsize=16)
def system_psd(name, npts=1201):
    """(f in MHz, PSD in dB) of one carrier of a system: a teaching sketch with the right width
    and roll-off (GMSK and GFSK simulated, WCDMA RRC 0.22, OFDM as a sum of sinc² tones)."""
    S = SYSTEMS[name]
    bw = S["bw"]
    if name.startswith("GSM") or name.startswith("Bluetooth"):
        rs = 0.2708 if name.startswith("GSM") else 1.0
        bt, h = (0.3, 0.5) if name.startswith("GSM") else (0.5, 0.5)
        r_ = np.random.default_rng(1)
        x, _ = iot.gfsk_mod(r_.integers(0, 2, 6000), sps=16, bt=bt, h=h)
        Xf = np.abs(np.fft.fftshift(np.fft.fft(x[:len(x) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024), axis=1),
                                    axes=1)) ** 2
        P = Xf.mean(0)
        f = np.fft.fftshift(np.fft.fftfreq(1024, 1 / 16)) * rs
        fx = np.linspace(-2 * bw, 2 * bw, npts)
        return fx, db(np.interp(fx, f, P / P.max()))
    fx = np.linspace(-0.75 * bw, 0.75 * bw, npts)
    if name.startswith("UMTS"):
        a = np.abs(fx) / 3.84
        beta = 0.22
        Pf = np.where(a <= (1 - beta) / 2, 1.0,
                      np.where(a <= (1 + beta) / 2,
                               0.5 * (1 + np.cos(np.pi / beta * (a - (1 - beta) / 2))), 0.0))
        return fx, np.maximum(db(Pf), -60)
    if name.startswith("LoRa"):
        Pf = 1 / (1 + (np.abs(fx) / (bw / 2)) ** 40)
        return fx, np.maximum(db(Pf), -60)
    occ = {"LTE": 18.0, "NR 1": 98.28, "Wi-Fi 6": 77.5, "Wi-Fi 7": 310.0}
    key = next(k for k in occ if name.startswith(k))
    o = occ[key]
    df = {"LTE": 0.015, "NR 1": 0.03, "Wi-Fi 6": 0.078125, "Wi-Fi 7": 0.078125}[key]
    fk = np.linspace(-o / 2, o / 2, 400)
    w = max(df, o / 399)              # representative tones: never narrower than their spacing
    Pf = (np.sinc((fx[:, None] - fk[None, :]) / w) ** 2).sum(1)
    Pf /= Pf[npts // 2]
    return fx, np.maximum(db(Pf), -60)


def nice(x):
    """Short human number: 2 337, 14.4, 0.271, 0.00547."""
    return f"{x:,.0f}".replace(",", " ") if x >= 100 else f"{x:.3g}"


class AirInterfaceTour(Experiment):
    title = "Air interfaces side by side"
    blurb = "GSM to Wi-Fi 7 and LoRa: how each one carves up spectrum, time and users."
    book = "sec:ch21:compare"
    controls = [
        Choice("sys", "System", list(SYSTEMS), "LTE 20 MHz (4G)", style="menu"),
        Choice("cmp", "Compare with", ["nothing"] + list(SYSTEMS), "GSM (2G)", style="menu"),
        IntSlider("users", "Active users", 1, 12, 4,
                  help="How the system shares itself among this many users (frame picture)"),
        Choice("metric", "Bar chart", list(METRICS), "Peak rate", style="menu"),
    ]
    plots = [
        SpectrumPlot("spec", "One carrier's spectrum, to scale", x="frequency (MHz)",
                     y="PSD (dB)", ylim=(-55, 8), legend="tr"),
        ImagePlot("frame", "How users share it", x="time", y="resource"),
        BarPlot("bars", "All systems, one number", y="log scale"),
    ]
    layout = [["spec", "frame"], ["bars", "bars"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("bw", "Channel bandwidth", "MHz", lambda x: f"{x:g}"),
        Readout("rate", "Peak rate", "Mb/s", nice),
        Readout("se", "Peak spectral efficiency", "b/s/Hz", ".2f"),
        Readout("sym", "Symbol duration", "µs", nice),
    ]
    challenges = [
        Challenge("Find the system whose symbols last longer than a millisecond, and say why.",
                  lambda s: s.p.sys.startswith("LoRa"),
                  hint="Long symbols buy energy per symbol: sensitivity far below the noise."),
        Challenge("Put GSM and NR on the same axis: one NR carrier is 500 GSM carriers wide.",
                  lambda s: {s.p.sys, s.p.cmp} == {"GSM (2G)", "NR 100 MHz (5G)"}),
        Challenge("Find the champion of peak spectral efficiency (b/s/Hz).",
                  lambda s: s.p.sys == max(SYSTEMS, key=lambda k: SYSTEMS[k]["rate"] / SYSTEMS[k]["bw"]),
                  hint="Use the bar chart on 'Spectral efficiency'."),
    ]

    def make_frame(self, p):
        """(image, x extent, y extent, x label, y label): user index (1..12) or 0 = empty,
        13 = control/overhead."""
        name, U = p.sys, p.users
        r_ = np.random.default_rng(5)
        if name.startswith("GSM"):
            img = np.zeros((4, 16))
            for u in range(U):
                c, ts = (u // 7), 1 + u % 7
                img[c, ts::8] = u + 1
            img[0, 0::8] = 13                      # BCCH/control timeslot 0 on carrier 0
            return img, (0, 16), (0, 4), "time slot (2 TDMA frames of 4.615 ms)", "200 kHz carrier"
        if name.startswith("UMTS"):
            img = np.zeros((16, 30))
            for u in range(U):
                img[u + 1, :] = u + 1
            img[0, :] = 13
            return img, (0, 30), (0, 16), "slot (2 frames of 10 ms)", "OVSF code (SF 16)"
        if name.startswith("LTE") or name.startswith("NR"):
            nrb, nsl = 25, 20
            img = np.zeros((nrb, nsl))
            for t in range(nsl):
                cuts = np.sort(r_.choice(np.arange(1, nrb), U - 1, replace=False)) if U > 1 else []
                edges = np.r_[0, cuts, nrb]
                users = r_.permutation(U)
                for k in range(U):
                    img[edges[k]:edges[k + 1], t] = users[k] + 1
            img[:, ::2 if name.startswith("LTE") else 1] = np.where(
                np.arange(nrb)[:, None] % 6 == 0, 13, img[:, ::2 if name.startswith("LTE") else 1])
            return img, (0, nsl), (0, nrb), "slot (0.5 ms)", "resource block (25 shown)"
        if name.startswith("Wi-Fi"):
            nru, nt = 9, 24
            img = np.zeros((nru, nt))
            t = 0
            while t < nt:
                gap = int(r_.integers(1, 3))
                t += gap
                if t >= nt:
                    break
                ln = 3
                if U > 1:
                    users = r_.permutation(U)[:min(U, nru)]
                    per = nru // len(users)
                    for k, u in enumerate(users):
                        hi_ = nru if k == len(users) - 1 else (k + 1) * per
                        img[k * per:hi_, t:t + ln] = u + 1
                else:
                    img[:, t:t + ln] = 1
                img[:, max(t - 1, 0)] = np.where(img[:, max(t - 1, 0)] == 0, 13, img[:, max(t - 1, 0)])
                t += ln
            return img, (0, nt), (0, nru), "time (preamble + PPDU)", "26-tone resource unit"
        if name.startswith("Bluetooth"):
            img = np.zeros((40, 24))
            for u in range(U):
                ch = (u * 7 + 3) % 37
                for t in range(u % 3, 24, 3):
                    img[ch, t] = u + 1
                    ch = (ch + 9 + u) % 37
            for c in (37, 38, 39):
                img[c, ::8] = 13
            return img, (0, 24), (0, 40), "connection event", "RF channel (2 MHz)"
        img = np.zeros((8, 60))
        for u in range(U):
            for t0 in r_.integers(0, 58, 2):
                c = r_.integers(0, 8)
                img[c, t0:t0 + 1] = u + 1
        return img, (0, 60), (0, 8), "time (s)", "125 kHz channel"

    def update(self, p):
        S = SYSTEMS[p.sys]
        ps = self.plot("spec")
        f, P = system_psd(p.sys)
        ps.line("a", f, P, color=NAVY, width=2.2, fill=-60, fill_alpha=0.15, name=p.sys)
        span = 0.75 * S["bw"] if not (p.sys.startswith("GSM") or p.sys.startswith("Blue")) else 2 * S["bw"]
        if p.cmp != "nothing" and p.cmp != p.sys:
            f2, P2 = system_psd(p.cmp)
            ps.line("b", f2, P2, color=RED, width=1.8, fill=-60, fill_alpha=0.10, name=p.cmp)
            S2 = SYSTEMS[p.cmp]
            span = max(span, 0.75 * S2["bw"] if not (p.cmp.startswith("GSM") or p.cmp.startswith("Blue"))
                       else 2 * S2["bw"])
            ratio = S["bw"] / S2["bw"]
            ps.set_title(f"To scale: {p.sys.split(' (')[0]} is {ratio:.3g}× as wide as "
                         f"{p.cmp.split(' (')[0]}")
        else:
            ps.set_title("One carrier's spectrum, to scale")
        ps.set_xlim(-span, span)
        # frame picture
        img, xe, ye, xl, yl = self.make_frame(p)
        pf = self.plot("frame")
        dark = pf.theme.name == "dark"
        cols = ["#1E252E" if dark else "#F1F4F8"] + USER_COLS + ["#555F6B" if dark else "#B8C0CA"]
        pf.image("img", img, x=xe, y=ye, cmap=cat_cmap(cols), levels=(-0.5, 13.5))
        pf.set_labels(x=xl, y=yl)
        pf.set_xlim(*xe)
        pf.set_ylim(*ye)
        pf.set_title(f"How {p.users} user{'s' if p.users > 1 else ''} share it: "
                     f"{S['access']} (grey = control)")
        # bars
        key, unit = METRICS[p.metric]
        names = list(SYSTEMS)
        vals = np.array([SYSTEMS[k]["rate"] / SYSTEMS[k]["bw"] if key == "se" else SYSTEMS[k][key]
                         for k in names])
        lv = np.log10(vals)
        base = np.floor(lv.min()) - 0.3
        pb = self.plot("bars")
        pb.bars("b", np.arange(len(names)), lv, base=base, width=0.62,
                colors=[RED if k == p.sys else (ORANGE if k == p.cmp else GRAY) for k in names])
        for i, (k, val) in enumerate(zip(names, vals)):
            pb.text(f"t{i}", i, lv[i], nice(val), anchor=(0.5, 1.05),
                    size=8.5, bold=k == p.sys)
        pb.set_xticks([(i, k.replace(" (", " (").split(",")[0].split(" (")[0])
                       for i, k in enumerate(names)])
        ticks = np.arange(np.ceil(base), np.ceil(lv.max()) + 1)
        pb.set_yticks([(t, f"{10 ** t:g}") for t in ticks])
        pb.set_ylim(base, lv.max() + 0.8)
        pb.set_xlim(-0.6, len(names) - 0.4)
        pb.set_title(f"All systems: {p.metric.lower()} ({unit}, log scale)")
        self.readout(bw=S["bw"], rate=S["rate"], se=S["rate"] / S["bw"], sym=S["sym"])

    def story(self, p):
        S = SYSTEMS[p.sys]
        rate_txt = nice(S["rate"])
        s = (f"<p><b>{p.sys}</b> ({S['year']}): {S['access']}, {S['mod']}; {S['note']}. "
             f"A carrier {v(S['bw'], 'g', 'MHz')} wide delivers up to "
             f"{v(rate_txt, unit='Mb/s')}, "
             f"or {v(S['rate'] / S['bw'], '.2f', 'b/s/Hz')}.</p>")
        acc = S["access"]
        if acc.startswith("TDMA"):
            s += ("<p>GSM gives each user one timeslot in eight on a 200 kHz carrier; more users "
                  "need more carriers (frequency planning with reuse 3–4).</p>")
        elif acc == "CDMA":
            s += ("<p>In CDMA every user occupies the whole 5 MHz all the time; they are told "
                  "apart by orthogonal spreading codes, so capacity is set by interference, "
                  "not by slots.</p>")
        elif acc == "OFDMA":
            s += ("<p>OFDMA hands out resource blocks in both frequency and time, anew every "
                  "slot, so the scheduler can follow each user's channel (experiment 3).</p>")
        elif acc.startswith("CSMA"):
            s += ("<p>Wi-Fi listens before talking: transmissions are bursts separated by "
                  "contention gaps (grey preambles), and since Wi-Fi 6 one burst can be split "
                  "into resource units for several users.</p>")
        elif acc.startswith("FHSS"):
            s += ("<p>Bluetooth hops: each connection jumps between 37 data channels on every "
                  "event, dodging interference, with three advertising channels (grey).</p>")
        else:
            s += ("<p>LoRa devices just transmit (ALOHA) on one of a few channels: rare, long, "
                  "very robust packets, perfect for a sensor that wakes once an hour.</p>")
        return "<h3>Many answers to one question</h3>" + s + keybox(
            "Every air interface divides the same three resources, time, frequency and code (or "
            "space), among users. The symbol lengths span six decades.")


# =============================================================================== the lab
LAB = st.Lab(11, "Air Interfaces: Cells, Trunks, Schedulers and Contention", chapter=20,
             chapter_title="Multiple Access and the Cellular Concept",
             experiments=[ReuseSIR, ErlangSwitchboard, OFDMAScheduling, LinkAdaptation,
                          WiFiContention, EVMBudget, AirInterfaceTour])

if __name__ == "__main__":
    st.run(LAB)
