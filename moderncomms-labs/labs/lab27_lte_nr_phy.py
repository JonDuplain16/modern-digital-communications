"""Lab 27 · Inside an LTE/NR Downlink: Grid, Cell Search, PDSCH and HARQ   (Chapter 21)

Run it:      python labs/lab27_lte_nr_phy.py
Self-test:   python labs/lab27_lte_nr_phy.py --selftest

When a phone is switched on it knows nothing: not the timing, not its own crystal's error, not
which of 504 cells it hears. Within a few hundred milliseconds it has found a cell, read its
identity and is decoding data protected by CRCs, an LDPC code, rate matching, QAM and OFDM,
with hybrid ARQ quietly combining retransmissions underneath. Eight experiments build that
physical layer from commlib.ltephy: the resource grid of LTE and NR, the synchronisation
sequences, a two-cell search, the PDSCH chain step by step, HARQ with chase combining and
incremental redundancy, CQI-driven throughput, the TS 38.306 peak-rate formula, and why the
uplink is DFT-spread: the PAPR of OFDMA against DFT-s-OFDM.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

import threading
from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import airif as ai
from commlib import ltephy as lp
from commlib import ofdm as co
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, ConstellationPlot, BERPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def cat_cmap(colors):
    """Stepped colormap for an integer image 0..n-1 (use levels=(-0.5, n-0.5))."""
    n = len(colors)
    stops = []
    for k, c in enumerate(colors):
        stops.append((k / n, c))
        stops.append(((k + 1) / n - 1e-4 if k < n - 1 else 1.0, c))
    return tuple(stops)


RE_NAMES = ["data (PDSCH)", "control (PDCCH)", "CRS / DMRS", "other RS", "PSS", "SSS", "PBCH",
            "empty / guard"]
RE_SHORT = ["data", "ctrl", "RS", "oth. RS", "PSS", "SSS", "PBCH", "empty"]
RE_LIGHT = ["#DDF0E5", "#AED6F1", "#1B3A5C", "#48C9B0", "#E74C3C", "#F5B041", "#A569BD", "#FFFFFF"]
RE_DARK = ["#1F3A2E", "#24507A", "#8AB8E8", "#2BA88F", "#E0604F", "#E0A040", "#9B6BC0", "#11151A"]


def re_colors(theme_name):
    return RE_DARK if theme_name == "dark" else RE_LIGHT


# ------------------------------------------------------------------ grid builders
def lte_grid(nrb, cfi, ports, sf, pci):
    """(12·nrb, 14) labels of an LTE FDD subframe (TS 36.211, normal CP), categories above."""
    G = np.zeros((12 * nrb, 14), int)
    G[:, :cfi] = 1
    vsh = pci % 6
    k = np.arange(12 * nrb)
    for slot in (0, 1):
        for l, (v0, v1) in ((0, (0, 3)), (4, (3, 0))):
            sym = 7 * slot + l
            G[(k % 6) == (v0 + vsh) % 6, sym] = 2
            if ports >= 2:
                G[(k % 6) == (v1 + vsh) % 6, sym] = 3
        if ports == 4:
            sym = 7 * slot + 1
            v2 = 3 * (slot % 2)
            G[((k % 6) == (v2 + vsh) % 6) | ((k % 6) == (v2 + 3 + vsh) % 6), sym] = 3
    c0 = 6 * nrb - 36
    mid = slice(c0, c0 + 72)
    if sf in (0, 5):
        G[mid, 5:7] = 7
        G[c0 + 5:c0 + 67, 6] = 4
        G[c0 + 5:c0 + 67, 5] = 5
    if sf == 0:
        blk = G[mid, 7:11]
        blk[blk == 0] = 6
        # the PBCH is always rate-matched around the CRS of four ports (symbols 7 and 8)
        crs4 = (np.arange(72) + c0 - vsh) % 3 == 0
        for s_ in (0, 1):
            col = blk[:, s_]
            col[crs4 & (col == 6)] = 7
        G[mid, 7:11] = blk
    return G


DMRS_POS = {0: [2], 1: [2, 11], 2: [2, 7, 11], 3: [2, 5, 8, 11]}   # TS 38.211 Table 7.4.1.1.2-3


def nr_grid(nrb, coreset, add_dmrs, ptrs, csirs, ssb):
    """(12·nrb, 14) labels of one NR downlink slot: CORESET (lower 48 RBs), type-A DMRS
    (configuration type 1, one CDM group), PT-RS (every 2nd RB), a 4-port CSI-RS in
    symbol 12, and optionally one SS/PBCH block (20 RBs × 4 symbols, centred)."""
    G = np.zeros((12 * nrb, 14), int)
    G[:12 * min(nrb, 48), :coreset] = 1
    k = np.arange(12 * nrb)
    for s_ in DMRS_POS[add_dmrs]:
        G[k % 2 == 0, s_] = 2
    if ptrs:
        rows = 24 * np.arange((nrb + 1) // 2) + 1
        for s_ in range(3, 14):
            col = G[:, s_]
            r_ = rows[rows < 12 * nrb]
            col[r_[col[r_] == 0]] = 3
    if csirs:
        rows = (12 * np.arange(nrb)[:, None] + np.array([3, 4, 9, 10])[None, :]).ravel()
        G[rows, 12] = np.where(G[rows, 12] == 0, 3, G[rows, 12])
    if ssb and nrb >= 20:
        c0 = 6 * nrb - 120
        b = slice(c0, c0 + 240)
        G[b, 2:6] = 6
        G[b, 2] = 7
        G[c0 + 56:c0 + 183, 2] = 4
        G[c0 + 48:c0 + 192, 4] = 7
        G[c0 + 56:c0 + 183, 4] = 5
    return G


def nearest_bw(mu, bw):
    allowed = ai.nr_bandwidths(mu)
    return min(allowed, key=lambda b: abs(b - bw))


# =============================================================================== 1. resource grid
class ResourceGrid(Experiment):
    title = "The resource grid builder"
    blurb = "Lay out an LTE subframe or an NR slot and count what is left for data."
    book = "sec:ch21:lte"
    controls = [
        Choice("sys", "Standard", ["LTE", "NR"]),
        Heading("LTE (TS 36.211)"),
        Choice("lbw", "Bandwidth", ["1.4", "3", "5", "10", "15", "20"], "1.4",
               help="MHz; 6, 15, 25, 50, 75 or 100 resource blocks", enabled_if=lambda p: p.sys == "LTE"),
        IntSlider("cfi", "Control symbols (CFI)", 1, 3, 2, enabled_if=lambda p: p.sys == "LTE"),
        Choice("ports", "CRS antenna ports", ["1", "2", "4"], "1", enabled_if=lambda p: p.sys == "LTE"),
        IntSlider("sf", "Subframe", 0, 9, 0, enabled_if=lambda p: p.sys == "LTE"),
        IntSlider("pci", "Physical cell ID", 0, 503, 301, enabled_if=lambda p: p.sys == "LTE",
                  help="The CRS lattice shifts by PCI mod 6 subcarriers"),
        Heading("NR (TS 38.211)"),
        Choice("mu", "Subcarrier spacing", ["15 kHz", "30 kHz", "60 kHz", "120 kHz"], "30 kHz",
               style="menu", enabled_if=lambda p: p.sys == "NR"),
        Choice("nbw", "Bandwidth", ["5", "10", "20", "40", "50", "100", "200", "400"], "100",
               style="menu", help="MHz; snapped to the nearest bandwidth defined for the spacing",
               enabled_if=lambda p: p.sys == "NR"),
        IntSlider("coreset", "CORESET symbols", 1, 3, 2, enabled_if=lambda p: p.sys == "NR"),
        Choice("dmrs", "Additional DMRS symbols", ["0", "1", "2", "3"], "1", style="menu",
               help="More DMRS symbols track faster channels", enabled_if=lambda p: p.sys == "NR"),
        Toggle("ptrs", "Phase-tracking RS (PT-RS)", True, enabled_if=lambda p: p.sys == "NR"),
        Toggle("csirs", "CSI-RS (4 ports)", True, enabled_if=lambda p: p.sys == "NR"),
        IntSlider("slot", "Slot in the frame", 0, 79, 0, enabled_if=lambda p: p.sys == "NR",
                  help="SS/PBCH blocks occupy the first slots of every other half-frame"),
        Heading("View"),
        Toggle("whole", "Show the whole carrier", False),
    ]
    plots = [
        ImagePlot("grid", "Resource grid", x="OFDM symbol", y="subcarrier"),
        BarPlot("mix", "RE shares (colours as in the grid)", y="share (%)"),
        ImagePlot("strip", "10 ms frame: orange = sync, ring = shown",
                  x="subframe / slot", y=""),
    ]
    layout = [["grid", "mix"], ["grid", "strip"]]
    col_stretch = [3, 2]
    row_stretch = [3, 1]
    readouts = [
        Readout("nre", "Resource elements", "per subframe/slot", "int"),
        Readout("oh", "Overhead", "%", ".1f", good=lambda x: x < 15),
        Readout("data", "Data REs per ms", "", "int"),
        Readout("rate", "1 layer, 256QAM R 0.93", "Mb/s", ".1f"),
    ]
    challenges = [
        Challenge("Reproduce Chapter 21's count: LTE 20 MHz, one control symbol, four CRS ports, "
                  "no sync signals: 13 600 data REs per ms.",
                  lambda s: s.p.sys == "LTE" and s.r.data == 13600),
        Challenge("Move the CRS lattice onto PCI 301's (same subcarriers) with a different cell ID.",
                  lambda s: s.p.sys == "LTE" and s.p.pci != 301 and s.p.pci % 6 == 301 % 6,
                  hint="The shift is PCI mod 6: neighbours plan their PCIs so their pilots differ."),
        Challenge("NR, 100 MHz at 30 kHz: trim the overhead below 6 % for a slow, quiet channel.",
                  lambda s: (s.p.sys == "NR" and s.p.mu == "30 kHz" and s.p.nbw == "100"
                             and s.r.oh < 6)),
        Challenge("NR: catch a slot carrying an SS/PBCH block at 120 kHz.",
                  lambda s: s.p.sys == "NR" and s.p.mu == "120 kHz" and s.exp.has_ssb),
    ]

    def ssb_slots(self, mu):
        """Slots of the 10 ms frame holding SSBs (first half-frame, cases A/C/D)."""
        L = {0: 4, 1: 8, 2: 8, 3: 64}[mu]
        if mu == 3:          # case D: 64 SSBs in 4 groups of 8 slots, gaps of 2 slots
            return sorted({g * 10 + s for g in range(4) for s in range(8)})
        return list(range(L // 2))

    def update(self, p):
        theme = self.plot("grid").theme.name
        cols = re_colors(theme)
        if p.sys == "LTE":
            nrb = ai.LTE_RB[float(p.lbw)] if p.lbw == "1.4" else ai.LTE_RB[int(p.lbw)]
            G = lte_grid(nrb, p.cfi, int(p.ports), p.sf, p.pci)
            per_ms = 1
            nunits, cur = 10, p.sf
            sync_units = [0, 5]
            unit = "subframe"
            title = f"LTE {p.lbw} MHz ({nrb} RBs), subframe {p.sf}, PCI {p.pci}, CFI {p.cfi}"
            self.has_ssb = p.sf in (0, 5)
        else:
            mu = ["15 kHz", "30 kHz", "60 kHz", "120 kHz"].index(p.mu)
            bw = nearest_bw(mu, float(p.nbw))
            nrb = ai.NR_MAX_PRB[(mu, bw)]
            nunits = 10 * 2 ** mu
            cur = min(p.slot, nunits - 1)
            sync_units = self.ssb_slots(mu)
            self.has_ssb = cur in sync_units
            G = nr_grid(nrb, p.coreset, int(p.dmrs), p.ptrs, p.csirs, self.has_ssb)
            per_ms = 2 ** mu
            unit = "slot"
            title = (f"NR {bw} MHz at {p.mu} ({nrb} RBs), slot {cur}"
                     + (" with an SS/PBCH block" if self.has_ssb else ""))
        n_sc = G.shape[0]
        # view window
        if p.whole or nrb <= 25:
            lo, hi = 0, n_sc
        else:
            lo = 12 * (nrb // 2 - 12)
            hi = lo + 300
        pg = self.plot("grid")
        pg.image("img", G[lo:hi].astype(float), x=(-0.5, 13.5), y=(lo, hi), cmap=cat_cmap(cols),
                 levels=(-0.5, 7.5))
        # symbol and RB separators
        vx, vy = [], []
        for x in np.arange(0.5, 13.5, 1.0):
            vx += [x, x, np.nan]
            vy += [lo, hi, np.nan]
        pg.line("sym", vx, vy, color=pg.theme.plot_bg, width=1.0)
        if hi - lo <= 300:
            hx, hy = [], []
            for y in range(lo - lo % 12 + 12, hi, 12):
                hx += [-0.5, 13.5, np.nan]
                hy += [y, y, np.nan]
            if hx:
                pg.line("rb", hx, hy, color=GRAY, width=0.8)
        if p.sys == "LTE":
            pg.vline("slot", 6.5, color=NAVY, style="-", width=2.0)
        pg.set_xlim(-0.5, 13.5)
        pg.set_ylim(lo, hi)
        pg.set_xticks([(i, str(i % 7) if p.sys == "LTE" else str(i)) for i in range(14)])
        pg.set_title(title + ("" if (p.whole or nrb <= 25) else " — central 25 RBs"))
        # mix
        cnt = np.array([np.sum(G == i) for i in range(8)])
        share = 100 * cnt / G.size
        pm = self.plot("mix")
        pm.bars("b", np.arange(8), share, width=0.62, colors=[c if c not in ("#FFFFFF", "#11151A")
                                                               else GRAY for c in cols])
        for i, s_ in enumerate(share):
            pm.text(f"t{i}", i, s_, f"{s_:.1f}" if s_ > 0 else "", anchor=(0.5, 1.05), size=8.5)
        pm.set_xticks([(i, RE_SHORT[i]) for i in range(8)])
        pm.set_xlim(-0.6, 7.6)
        pm.set_ylim(0, 112)
        # frame strip
        ps = self.plot("strip")
        strip = np.zeros((1, nunits))
        strip[0, sync_units] = 1
        sc = ["#E3ECF6" if theme != "dark" else "#24303D", "#F5B041"]
        ps.image("s", strip, x=(-0.5, nunits - 0.5), y=(0, 1), cmap=cat_cmap(sc), levels=(-0.5, 1.5))
        ps.scatter("cur", [cur], [0.5], color=RED, size=16 if nunits <= 20 else 11, outline=RED,
                   symbol="o")
        ps.set_xlim(-0.5, nunits - 0.5)
        ps.set_ylim(0, 1)
        ps.set_yticks([])
        step = 1 if nunits <= 20 else (4 if nunits <= 40 else 8)
        ps.set_xticks([(i, str(i)) for i in range(0, nunits, step)])
        ps.set_labels(x=f"{unit} in the 10 ms frame")
        ndata = int(cnt[0])
        self.readout(nre=G.size, oh=100 * (1 - ndata / G.size), data=ndata * per_ms,
                     rate=ndata * per_ms * 1000 * 8 * 948 / 1024 / 1e6)

    def story(self, p):
        r = self.r
        if p.sys == "LTE":
            s = ("<p>An LTE resource block is 12 subcarriers of 15 kHz for one 0.5 ms slot; a "
                 "1 ms subframe has 14 OFDM symbols (the thick line splits its two slots). One "
                 "subcarrier for one symbol is a <b>resource element</b>, the atom of the air "
                 "interface.</p>"
                 f"<p>Before any user data, the first {v(p.cfi, 'd')} symbol(s) carry control "
                 f"(PDCCH), the cell-specific reference signals (strong blue) form a diagonal lattice "
                 f"every sixth subcarrier, shifted by PCI mod 6 = {v(p.pci % 6, 'd')}, ")
            s += ("and subframes 0 and 5 carry the PSS and SSS on the central 62 subcarriers "
                  "(subframe 0 also the PBCH). " if p.sf in (0, 5) else
                  "and this subframe carries no synchronisation signals. ")
            s += (f"In all, {v(r.get('oh', 0), '.1f', '%')} of the grid is overhead.</p>")
        else:
            s = ("<p>NR keeps the 12-subcarrier resource block but lets the subcarrier spacing "
                 "scale as 15·2<sup>μ</sup> kHz; a slot is always 14 symbols, so it shrinks to "
                 f"{v(1 / 2 ** ['15 kHz', '30 kHz', '60 kHz', '120 kHz'].index(p.mu), 'g', 'ms')} at "
                 f"{p.mu}.</p>"
                 "<p>NR dropped LTE's always-on CRS: every pilot is now per user and on demand. "
                 "DMRS (strong blue) only where data is sent, PT-RS to track phase noise at high "
                 "frequency, CSI-RS when the network wants a channel report, and the SS/PBCH "
                 "block (red/orange/purple) only in a few slots every 20 ms. That 'lean carrier' "
                 f"is why the overhead here is {v(r.get('oh', 0), '.1f', '%')}.</p>")
        return "<h3>Counting resource elements</h3>" + s + keybox(
            "Peak rate = data REs per second × bits per RE × layers. Every pilot, control symbol "
            "and sync signal is a resource element not carrying your data.")


# =============================================================================== 2. sync sequences
@lru_cache(maxsize=1)
def sss_matrix():
    """(2·168, 62) SSS sequences for N_ID2 = 0 (subframe 0 then subframe 5) and their |corr|."""
    S = np.array([lp.sss_sequence(n1, 0, 0) for n1 in range(168)] +
                 [lp.sss_sequence(n1, 0, 5) for n1 in range(168)])
    return S, np.abs(S @ S.T) / 62


@lru_cache(maxsize=1)
def pss_wave():
    return lp.pss_waveforms()


def pss_corr(x, ref, segments):
    """Segmented correlation of x with ref at every lag (non-coherent sum of |.|²)."""
    L = len(ref) // segments
    out = 0
    for h in range(segments):
        r = np.zeros(len(ref), complex)
        r[h * L:(h + 1) * L] = ref[h * L:(h + 1) * L]
        out = out + np.abs(np.correlate(x, r, "full")) ** 2
    return out


class SyncSequences(Experiment):
    title = "PSS and SSS: why they work"
    blurb = "Zadoff–Chu and m-sequences: sharp peaks, low cross-talk, and the CFO trap."
    book = "sec:ch21:lte"
    controls = [
        IntSlider("nid2", "Transmitted N_ID(2)", 0, 2, 1,
                  help="Selects the Zadoff–Chu root 25, 29 or 34"),
        Slider("cfo", "Frequency offset", -15, 15, 0, step=0.25, unit="kHz",
               help="The phone's crystal error: 1 ppm at 2 GHz is 2 kHz"),
        Choice("seg", "Coherent segments", ["1", "2", "4"], "1",
               help="Split the 128-sample correlation into pieces added non-coherently"),
    ]
    plots = [
        Plot("lag", "PSS correlation against the three hypotheses", x="lag (samples)",
             y="|correlation|² (dB, 0 = ideal peak)", xlim=(-127, 127), ylim=(-40, 5), legend="tl",
             legend_cols=3),
        Plot("cfo", "Correct-root peak vs frequency offset", x="frequency offset (kHz)",
             y="peak (dB re no offset)", xlim=(-15, 15), ylim=(-25, 2), legend="bl"),
        ImagePlot("sss", "|SSS correlation|: 168 IDs × 2 half-frames", x="hypothesis",
                  y="hypothesis"),
    ]
    layout = [["lag", "lag"], ["cfo", "sss"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("peak", "Correct-root peak", "dB", ".1f", good=lambda x: x > -1),
        Readout("terr", "Timing error", "samples", "int", good=lambda x: x == 0),
        Readout("margin", "Margin over wrong roots", "dB", ".1f", good=lambda x: x > 6),
        Readout("sssmax", "Worst SSS cross-corr.", "", ".2f"),
    ]
    challenges = [
        Challenge("One coherent segment: find the offset where the timing jumps — the biggest "
                  "peak lands tens of samples away from the truth.",
                  lambda s: s.p.seg == "1" and abs(s.r.terr) > 10,
                  hint="A frequency offset on a Zadoff–Chu sequence looks exactly like a delay."),
        Challenge("Keep the peak loss under 1 dB at 12 kHz of offset or more.",
                  lambda s: abs(s.p.cfo) >= 12 and s.r.peak > -1,
                  hint="Shorter coherent pieces turn less under a frequency offset."),
        Challenge("With 2 segments, find the largest offset (to 0.5 kHz) that keeps the peak "
                  "within 1 dB.",
                  lambda s: s.p.seg == "2" and s.r.peak >= -1 and s.exp.peak_more < -1),
    ]

    def peaks(self, nid2, cfo_hz, seg):
        P = pss_wave()
        n = np.arange(128)
        x = P[nid2] * np.exp(2j * np.pi * cfo_hz * n / lp.FS)
        ref0 = pss_corr(P[nid2], P[nid2], seg).max()
        cs = [pss_corr(x, P[i], seg) for i in range(3)]
        return [db(c / ref0) for c in cs]

    def update(self, p):
        seg = int(p.seg)
        cs = self.peaks(p.nid2, p.cfo * 1e3, seg)
        lags = np.arange(-127, 128)
        pl = self.plot("lag")
        cols = [NAVY, RED, GREEN]
        for i in range(3):
            pl.line(f"c{i}", lags, np.maximum(cs[i], -60), color=cols[i],
                    width=2.4 if i == p.nid2 else 1.2,
                    name=f"N_ID(2) = {i} (u = {lp.PSS_ROOTS[i]})" + (" ✓" if i == p.nid2 else ""))
        peak = float(cs[p.nid2].max())
        terr = int(np.argmax(cs[p.nid2])) - 127
        wrong = max(float(cs[i].max()) for i in range(3) if i != p.nid2)
        self.peak_more = float(self.peaks(p.nid2, (abs(p.cfo) + 0.5) * 1e3, seg)[p.nid2].max())
        # peak vs CFO
        pc = self.plot("cfo")
        f = np.linspace(-15, 15, 61)
        for k, (sg, col) in enumerate(((1, NAVY), (2, ORANGE), (4, GREEN))):
            vals = [self.peaks(p.nid2, ff * 1e3, sg)[p.nid2].max() for ff in f]
            pc.line(f"s{sg}", f, vals, color=col, width=2.4 if sg == seg else 1.2,
                    name=f"{sg} segment{'s' if sg > 1 else ''}")
        pc.scatter("now", [p.cfo], [peak], color=RED, size=13, symbol="d")
        pc.vline("ss", 7.5, color=GRAY, style=":", label="½ subcarrier", label_pos=0.1)
        pc.vline("ss2", -7.5, color=GRAY, style=":")
        # SSS
        S, Cm = sss_matrix()
        ps = self.plot("sss")
        ps.image("m", Cm, x=(0, 336), y=(0, 336), levels=(0, 1), colorbar=True)
        ps.set_xlim(0, 336)
        ps.set_ylim(0, 336)
        off = Cm[~np.eye(336, dtype=bool)]
        self.readout(peak=peak, terr=terr, margin=peak - wrong, sssmax=float(off.max()))

    def story(self, p):
        r = self.r
        s = ("<p>The <b>PSS</b> is a length-63 Zadoff–Chu sequence (middle element punctured for "
             "DC), one of three roots. Zadoff–Chu sequences have constant amplitude and an ideal "
             "periodic autocorrelation, so the right hypothesis gives one sharp spike (thick "
             "line) while the other two stay low at every lag: a phone can find the timing "
             "without knowing anything else.</p>")
        if abs(p.cfo) > 0.1:
            s += (f"<p>But a frequency offset of {v(p.cfo, '.2f', 'kHz')} rotates the phase "
                  f"by {v(abs(p.cfo) * 1e3 * 128 / lp.FS * 360, '.0f', '°')} across the 67 µs "
                  f"symbol; a coherent correlation then adds up vectors pointing every which "
                  f"way. The correct peak is now {v(r.get('peak', 0), '.1f', 'dB')}. Zadoff–Chu "
                  f"sequences have a nasty twist: a frequency offset looks like a time shift, so "
                  f"past about 9 kHz a side peak wins and the timing is off by "
                  f"{v(abs(r.get('terr', 0)), 'd')} samples. Splitting into {p.seg} coherent "
                  f"piece(s), added non-coherently, limits the rotation inside each piece (at "
                  f"the price of a little noise performance).</p>")
        s += (f"<p>The <b>SSS</b> (bottom right) interleaves two length-31 m-sequences: 336 "
              f"hypotheses (168 identities × which half-frame) with a bright diagonal and "
              f"cross-correlations up to {v(r.get('sssmax', 0), '.2f')}. PSS and SSS together give "
              f"PCI = 3·N_ID(1) + N_ID(2): 504 cells.</p>")
        return "<h3>Sequences built to be found</h3>" + s + keybox(
            "Good sync sequences: flat spectrum, one sharp autocorrelation peak, low "
            "cross-correlation, and tolerance to the frequency error of a cold crystal.")


# =============================================================================== 3. cell search
PCI_A, PCI_B = 301, 77
HF = 9600


@lru_cache(maxsize=4)
def cell_wave(pci):
    """Four 10 ms frames of a fully loaded cell."""
    r = np.random.default_rng(pci)
    return lp.ofdm_mod(np.array([lp.subframe_grid(pci, sf % 10, r)[0] for sf in range(40)]))


@lru_cache(maxsize=1)
def sss_bank():
    """SSS sequences for every (N_ID2, subframe, N_ID1): shape (3, 2, 168, 62)."""
    return np.array([[[lp.sss_sequence(n1, n2, sf) for n1 in range(168)] for sf in (0, 5)]
                     for n2 in range(3)])


def sss_metric(y, t_pss, nid2, cfo):
    """Vectorised version of ltephy.sss_detect: metric (2, 168) for the SSS before a PSS."""
    t = np.arange(t_pss - 300, t_pss + 140)
    rc = y[t] * np.exp(-2j * np.pi * cfo * t / lp.FS)
    k = lp._BINS[lp.sync_subcarriers()]
    o = 300
    Yp = (np.fft.fft(rc[o:o + 128]) / np.sqrt(128))[k]
    ts = o - 128 - lp.CP[6]
    Ys = (np.fft.fft(rc[ts:ts + 128]) / np.sqrt(128))[k]
    H = Yp / lp.pss_sequence(nid2)
    H = np.convolve(H, np.ones(31), "same") / np.convolve(np.ones(62), np.ones(31), "same")
    z = Ys * np.conj(H)
    return np.real(sss_bank()[nid2] @ z)


class CellSearch(Experiment):
    title = "Cell search on two cells"
    blurb = "A phone wakes up hearing two cells: find timing, frequency offset and both IDs."
    book = "sec:ch21:lte"
    controls = [
        Heading("What the phone hears"),
        Slider("snr", "SNR (cell A)", -12, 20, 3, step=0.5, unit="dB"),
        Slider("cfo", "Crystal frequency offset", -12, 12, 4, step=0.25, unit="kHz"),
        Slider("rel", "Cell B relative power", -15, 0, -6, step=0.5, unit="dB"),
        Slider("dly", "Cell B timing offset", 50, 4950, 365, step=5, unit="µs"),
        Heading("Search"),
        Choice("seg", "Coherent PSS segments", ["1", "2", "4"], "2"),
        Choice("nhf", "Half-frames averaged", ["1", "2", "4"], "4"),
        Button("again", "New noise and phases"),
    ]
    plots = [
        Plot("pss", "PSS correlation folded onto one 5 ms half-frame", x="time (ms)",
             y="normalised metric", xlim=(0, 5), legend="tr", legend_cols=3),
        Plot("sss", "SSS metric for each detected cell", x="N_ID(1) hypothesis",
             y="metric (normalised)", xlim=(0, 167), legend="tr"),
        Plot("zoom", "Zoom on the two peaks (each scaled to its own top)",
             x="time offset from the peak (µs)", y="metric / peak", ylim=(-0.05, 1.1),
             legend=None),
    ]
    layout = [["pss", "pss"], ["sss", "zoom"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("found", "Cells found", "", None),
        Readout("cfo", "CFO estimate", "kHz", ".2f"),
        Readout("err", "CFO error", "Hz", ".0f", good=lambda x: abs(x) < 500),
        Readout("ok", "Both correct?", "", None),
    ]
    challenges = [
        Challenge("Find both cells at an SNR of −4 dB or lower.",
                  lambda s: s.p.snr <= -4 and s.r.ok == "yes",
                  hint="Average more half-frames: the PSS repeats every 5 ms."),
        Challenge("One coherent segment cannot even measure the offset: make the search fail "
                  "with an offset of 3 kHz or less.",
                  lambda s: s.p.seg == "1" and abs(s.p.cfo) <= 3 and s.r.ok != "yes",
                  hint="Without the offset estimate the SSS test is done on a rotating signal."),
        Challenge("Find cell B even when it is 9 dB weaker than cell A.",
                  lambda s: s.p.rel <= -9 and s.r.ok == "yes",
                  hint="Cell A's own data look like noise to cell B. Try other timing offsets "
                       "and 'New noise': near the limit, detection is a matter of luck."),
    ]

    def setup(self):
        self.seed = 5

    def on_again(self, p):
        self.seed += 1

    def receive(self, p, nhf):
        r = np.random.default_rng(self.seed)
        n = (nhf + 1) * HF + 2 * lp.NFFT
        y = np.zeros(n, complex)
        dB = int(round(p.dly * 1e-6 * lp.FS))
        for pci, d, g in ((PCI_A, 0, 1.0), (PCI_B, dB, 10 ** (p.rel / 20))):
            w = cell_wave(pci)
            y += g * np.exp(2j * np.pi * r.random()) * w[2 * HF - d:2 * HF - d + n]
        y *= np.exp(2j * np.pi * p.cfo * 1e3 * np.arange(n) / lp.FS)
        s2 = 72 / 128 * 10 ** (-p.snr / 10)
        return y + np.sqrt(s2 / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))

    def update(self, p):
        nhf, seg = int(p.nhf), int(p.seg)
        y = self.receive(p, nhf)
        m, sg = lp.pss_search(y[:nhf * HF + lp.NFFT], seg)
        mf = sum(m[:, k * HF:(k + 1) * HF] for k in range(nhf)) / nhf
        found = []
        mm = mf.copy()
        sss_curves = []
        for c in range(2):
            i, t = np.unravel_index(np.argmax(mm), mm.shape)
            if found:
                cfo = found[0]["cfo"]
            elif seg > 1:
                prod = sum(np.sum(sg[i, 1:, t + k * HF] * np.conj(sg[i, :-1, t + k * HF]))
                           for k in range(nhf))
                cfo = np.angle(prod) / (2 * np.pi * (lp.NFFT // seg) / lp.FS)
            else:
                cfo = 0.0
            acc = np.zeros((2, 168))
            for k in range(1, nhf + 1):
                mk = sss_metric(y, t + k * HF, i, cfo)
                acc += mk if k % 2 else mk[::-1]
            sf_i, n1 = np.unravel_index(np.argmax(acc), acc.shape)
            found.append(dict(pci=3 * int(n1) + int(i), t=int(t), cfo=float(cfo), metric=float(mf[i, t]),
                              nid2=int(i)))
            sss_curves.append(acc[sf_i] / max(np.abs(acc).max(), 1e-12))
            mm[i, (t + np.arange(-200, 200)) % HF] = 0
        tm = np.arange(HF) / lp.FS * 1e3
        pp = self.plot("pss")
        cols = [NAVY, RED, GREEN]
        for i in range(3):
            pp.line(f"m{i}", tm, mf[i], color=cols[i], width=1.2, name=f"N_ID(2) = {i}")
        ymax = max(mf.max(), 0.05)
        for c, fnd in enumerate(found):
            okc = fnd["pci"] in (PCI_A, PCI_B)
            pp.text(f"f{c}", fnd["t"] / lp.FS * 1e3, fnd["metric"],
                    f" PCI {fnd['pci']} {'✓' if okc else '✗'}", color=GREEN if okc else RED,
                    size=9.5, bold=True, anchor=(0, 1))
        pp.set_ylim(0, 1.3 * ymax)
        ps = self.plot("sss")
        for c, (fnd, cur) in enumerate(zip(found, sss_curves)):
            ps.line(f"s{c}", np.arange(168), cur, color=[NAVY, ORANGE][c], width=1.6,
                    name=f"cell {c + 1}: N_ID(1) = {fnd['pci'] // 3}")
        ps.set_ylim(-0.6, 1.4)
        pz = self.plot("zoom")
        us = (np.arange(-120, 121)) / lp.FS * 1e6
        for c, fnd in enumerate(found):
            idx = (fnd["t"] + np.arange(-120, 121)) % HF
            seg_ = mf[fnd["nid2"], idx]
            pz.line(f"z{c}", us, seg_ / max(seg_.max(), 1e-12), color=[NAVY, ORANGE][c], width=2.0)
        pz.set_xlim(us[0], us[-1])
        truth = {PCI_A, PCI_B}
        ok = {f_["pci"] for f_ in found} == truth
        self.found = found
        self.readout(found=", ".join(str(f_["pci"]) for f_ in found), cfo=found[0]["cfo"] / 1e3,
                     err=found[0]["cfo"] - p.cfo * 1e3, ok="yes" if ok else "no")

    def story(self, p):
        r = self.r
        s = (f"<p>The phone hears cell A (PCI {PCI_A}) and cell B (PCI {PCI_B}), "
             f"{v(-p.rel, '.1f', 'dB')} weaker and {v(p.dly, '.0f', 'µs')} later, through a "
             f"crystal that is {v(p.cfo, '.2f', 'kHz')} off. Step 1: correlate with the three "
             f"PSS waveforms and fold {p.nhf} half-frame(s) of 5 ms on top of each other: two "
             f"peaks rise above the other cells' data (top).</p>"
             f"<p>Step 2: the phase between the PSS segments measures the frequency offset: "
             f"{v(r.get('cfo', 0), '.2f', 'kHz')}, off by {v(r.get('err', 0), '.0f', 'Hz')}. "
             f"Step 3: remove it, use the PSS as a channel estimate and test the 2 × 168 SSS "
             f"hypotheses (bottom left): the winner gives N_ID(1), the PCI and the 10 ms frame "
             f"boundary.</p>")
        if r.get("ok") == "yes":
            s += f"<p>{good('Both cells found.')} Real phones do this in a few tens of milliseconds.</p>"
        else:
            s += (f"<p>{bad('The search failed')}: a noise or data peak won somewhere. More "
                  f"half-frames averaged or more coherent segments usually rescue it.</p>")
        return "<h3>From nothing to a cell ID</h3>" + s + keybox(
            "Cell search = PSS (timing, N_ID(2)) → CFO → SSS (N_ID(1), frame timing). The CFO is "
            "the phone's own crystal, so it is estimated once and reused for every cell.")


# =============================================================================== 4. PDSCH chain
_CODE_LOCK = threading.Lock()
_CODE = []


def ldpc_code():
    """The (1152, 384) LDPC code (built once, about a second; warmed in the background)."""
    with _CODE_LOCK:
        if not _CODE:
            _CODE.append(cl.LDPCCode(n=1152, rate=1 / 3, seed=1))   # NR-style circular buffer
        return _CODE[0]


threading.Thread(target=ldpc_code, daemon=True).start()


PCI, SF, CFI, RNTI = 301, 1, 2, 0x1234
MCS = {"QPSK, R≈0.23 (1 CB)": (2, 1), "QPSK, R≈0.46 (2 CBs)": (2, 2),
       "16QAM, R≈0.46 (4 CBs)": (4, 4), "64QAM, R≈0.46 (6 CBs)": (6, 6),
       "QPSK, R≈0.70 (3 CBs)": (2, 3), "64QAM, R≈0.70 (9 CBs)": (6, 9)}


@lru_cache(maxsize=1)
def chain_consts():
    code = ldpc_code()
    mask = lp.pdsch_re_mask(PCI, SF, CFI)
    nre = int(mask.sum())
    scr = lp.gold_sequence(RNTI * 2 ** 14 + (2 * SF // 2) * 2 ** 9 + PCI, nre * 6)
    con = {q: cl.get_constellation(n) for q, n in ((2, "qpsk"), (4, "16qam"), (6, "64qam"))}
    return code, mask, nre, scr, con, code.k - 24


@lru_cache(maxsize=1)
def fmap():
    """Frequency-first RE order (as TS 36.211: subcarriers within a symbol, then the next
    symbol): position j of the symbol stream goes to the fmap()[j]-th RE of grid[mask]."""
    rows, cols = np.nonzero(chain_consts()[1])
    return np.lexsort((rows, cols))


def tb_size(C):
    return C * (ldpc_code().k - 24) - 24


def pdsch_encode(tb, Qm, C):
    code, mask, nre, scr, con, kcb = chain_consts()
    a = lp.crc_attach(tb, lp.CRC24A)
    blocks = np.array([lp.crc_attach(a[c * kcb:(c + 1) * kcb], lp.CRC24B) for c in range(C)])
    cw = code.encode(blocks).reshape(C, -1)
    return np.concatenate([cw[:, code.info], cw[:, code.pivots]], axis=1), nre * Qm // C


def pdsch_modulate(cbufs, E, Qm, rv):
    code, mask, nre, scr, con, kcb = chain_consts()
    e = np.concatenate([lp.bit_interleave(lp.rate_match(cb, E, rv), Qm) for cb in cbufs])
    sym = con[Qm].modulate(e ^ scr[:len(e)])
    dat = np.empty_like(sym)
    dat[fmap()] = sym                          # frequency-first mapping spreads every CB
    G, _ = lp.subframe_grid(PCI, SF, np.random.default_rng(1), cfi=CFI, data=dat)
    return G


def channel(x, snr_db, profile, r):
    y = x if profile == "AWGN" else cl.tdl_channel(x, lp.FS, profile, fd_hz=5.0, rng=r)[:len(x)]
    n0 = 10 ** (-snr_db / 10)
    return y + np.sqrt(n0 / 2) * (r.standard_normal(len(y)) + 1j * r.standard_normal(len(y))), n0


def pdsch_demod(y, n0, Qm, C, E, rv, soft=None, H=None):
    code, mask, nre, scr, con, kcb = chain_consts()
    Y = lp.ofdm_demod(y)
    H = lp.crs_channel_estimate(Y, PCI, SF) if H is None else H
    o = fmap()
    llr = con[Qm].llr(Y[mask][o], n0, h=H[mask][o])
    llr = llr * (1 - 2.0 * scr[:len(llr)])
    soft = np.zeros((C, code.n)) if soft is None else soft
    for c in range(C):
        soft[c] = lp.rate_recover(lp.bit_deinterleave(llr[c * E:(c + 1) * E], Qm), code.n, rv, soft[c])
    return soft, Y, H


def pdsch_decode(soft, C, iters=25):
    code, mask, nre, scr, con, kcb = chain_consts()
    lcw = np.empty_like(soft)
    lcw[:, code.info], lcw[:, code.pivots] = soft[:, :code.k], soft[:, code.k:]
    d, used = lp.ldpc_decode_batch(code, lcw, iters=iters)
    blocks = d[:, code.info]
    cb_ok = np.array([lp.crc_ok(b, lp.CRC24B) for b in blocks])
    a = np.concatenate([b[:kcb] for b in blocks])
    return a[:-24], cb_ok, lp.crc_ok(a, lp.CRC24A), used


class PDSCHChain(Experiment):
    title = "The PDSCH chain, step by step"
    blurb = "CRC → LDPC → rate matching → QAM → OFDM → channel → and all the way back."
    book = "sec:ch21:nrcoding"
    controls = [
        Heading("Transmitter"),
        Choice("mcs", "Modulation and coding", list(MCS), "16QAM, R≈0.46 (4 CBs)", style="menu"),
        Choice("rv", "Redundancy version", ["0", "1", "2", "3"], "0",
               help="Where in the circular buffer the transmission starts reading"),
        Heading("Channel"),
        Slider("snr", "SNR per resource element", -5, 30, 15, step=0.5, unit="dB"),
        Choice("prof", "Channel (TS 36.101)", ["AWGN", "EPA", "EVA", "ETU"], "EVA", style="menu",
               help="Extended pedestrian / vehicular / typical-urban multipath profiles"),
        Heading("Receiver"),
        Toggle("perfect", "Perfect channel knowledge", False,
               help="Use the true channel instead of the CRS-based estimate"),
        IntSlider("iters", "LDPC iterations", 1, 40, 25),
        Button("again", "New data and channel"),
    ]
    plots = [
        BarPlot("pipe", "Bits at every stage (one 1 ms transport block)", y="bits (log scale)"),
        ConstellationPlot("const", "Received (grey), equalised (blue)", lim=1.6),
        Plot("buf", "Code block 1's soft circular buffer (LLR = 0: never sent)",
             x="buffer position", y="LLR", xlim=(0, 1152), legend=None),
        Plot("chan", "Channel across the subcarriers (symbol 7)", x="subcarrier",
             y="|H| (dB)", xlim=(0, 71), ylim=(-25, 12), legend="bl"),
    ]
    layout = [["pipe", "const"], ["buf", "chan"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("tbs", "Transport block", "bits", "int"),
        Readout("rate", "Code rate K/E", "", ".2f"),
        Readout("cbs", "Code-block CRCs", "", None),
        Readout("tb", "Transport-block CRC", "", None),
    ]
    challenges = [
        Challenge("64QAM R≈0.46 over EVA: decode the transport block at 16 dB or less.",
                  lambda s: (s.p.mcs.startswith("64QAM, R≈0.46") and s.p.prof == "EVA"
                             and s.p.snr <= 16 and s.r.tb == "PASS")),
        Challenge("Catch a split decision: some code blocks pass and at least one fails.",
                  lambda s: "✓" in s.r.cbs and "✗" in s.r.cbs,
                  hint="Lower the SNR slowly with a multi-block MCS; each block has its own CRC."),
        Challenge("ETU's long echoes outrun the CRS: find a case where perfect channel knowledge "
                  "decodes but the estimate does not.",
                  lambda s: s.p.prof == "ETU" and s.p.perfect and s.exp.est_fail,
                  hint="Compare at the same SNR: tick and untick the switch."),
    ]

    def setup(self):
        self.seed = 11

    def on_again(self, p):
        self.seed += 1

    def run_once(self, p, perfect):
        Qm, C = MCS[p.mcs]
        rv = int(p.rv)
        r = np.random.default_rng(self.seed)
        tb = r.integers(0, 2, tb_size(C)).astype(np.int8)
        cbufs, E = pdsch_encode(tb, Qm, C)
        x = lp.ofdm_mod(pdsch_modulate(cbufs, E, Qm, rv))
        rch = np.random.default_rng(self.seed + 1000)
        if p.prof == "AWGN":
            y0 = x
        else:
            y0 = cl.tdl_channel(x, lp.FS, p.prof, fd_hz=5.0, rng=rch)[:len(x)]
        n0 = 10 ** (-p.snr / 10)
        y = y0 + np.sqrt(n0 / 2) * (r.standard_normal(len(x)) + 1j * r.standard_normal(len(x)))
        X = lp.ofdm_demod(x)                       # the true channel, for 'perfect knowledge'
        Y0 = lp.ofdm_demod(y0)
        Htrue = np.where(np.abs(X) > 1e-9, Y0 / np.where(np.abs(X) > 1e-9, X, 1), 0)
        soft, Y, H = pdsch_demod(y, n0, Qm, C, E, rv, H=Htrue if perfect else None)
        dec, cb_ok, tb_ok, used = pdsch_decode(soft.copy(), C, iters=p.iters)
        return dict(tb=tb, cbufs=cbufs, E=E, soft=soft, Y=Y, H=H, Htrue=Htrue, cb_ok=cb_ok,
                    tb_ok=tb_ok, used=used, C=C, Qm=Qm)

    def update(self, p):
        code, mask, nre, scr, con, kcb = chain_consts()
        R = self.run_once(p, p.perfect)
        self.est_fail = False
        if p.prof == "ETU" and p.perfect and R["tb_ok"]:
            self.est_fail = not self.run_once(p, False)["tb_ok"]
        C, Qm, E = R["C"], R["Qm"], R["E"]
        tbs = len(R["tb"])
        # pipeline bars
        stages = [("TB", tbs), ("+CRC", tbs + 24), (f"{C} CBs", C * code.k),
                  ("LDPC", C * code.n), ("rate-matched", C * E), ("QAM sym.", nre)]
        pb = self.plot("pipe")
        vals = np.log10([s_[1] for s_ in stages])
        pb.bars("b", np.arange(6), vals, base=2.0, width=0.62,
                colors=[NAVY, NAVY, BLUE, PURPLE, ORANGE, GREEN])
        for i, (nm, val) in enumerate(stages):
            pb.text(f"t{i}", i, vals[i], f"{val:,}".replace(",", " "), anchor=(0.5, 1.05), size=9,
                    bold=True)
        pb.set_xticks([(i, nm) for i, (nm, _) in enumerate(stages)])
        pb.set_yticks([(k_, f"{10 ** k_:g}") for k_ in (2, 3, 4, 5)])
        pb.set_ylim(2, 4.9)
        pb.set_xlim(-0.6, 5.6)
        # constellation
        pc = self.plot("const")
        Y, H = R["Y"], R["H"]
        pc.points("raw", Y[mask][:800] / np.sqrt(np.mean(np.abs(Y[mask]) ** 2)), color=GRAY,
                  size=2, alpha=0.35)
        pc.points("eq", (Y[mask] / np.where(np.abs(H[mask]) > 1e-6, H[mask], 1e-6))[:800],
                  color=NAVY, size=3, alpha=0.6)
        pc.ideal("ideal", con[Qm].points, color=RED, size=8 if Qm < 6 else 5)
        # circular buffer
        pbuf = self.plot("buf")
        s0 = R["soft"][0]
        k0 = lp.rv_start(int(p.rv), code.n)
        pbuf.band("sys", 0, code.k, color=GREEN, alpha=0.10)
        lim = max(8.0, float(np.percentile(np.abs(s0), 99)) * 1.25)
        pbuf.text("syst", code.k / 2, lim * 1.25, "systematic", color=GREEN, size=9, anchor=(0.5, 0))
        pbuf.text("part", (code.k + code.n) / 2, lim * 1.25, "parity", color=PURPLE, size=9,
                  anchor=(0.5, 0))
        end = k0 + E
        if end <= code.n:
            pbuf.band("rd", k0, end, color=ORANGE, alpha=0.12)
            pbuf.band("rd2", 0, 0, color=ORANGE, alpha=0.0)
        else:
            pbuf.band("rd", k0, code.n, color=ORANGE, alpha=0.12)
            pbuf.band("rd2", 0, end - code.n, color=ORANGE, alpha=0.12)
        pbuf.vline("k0", k0, color=ORANGE, style="--", label=f"RV {p.rv} starts here (k₀ = {k0})",
                   label_pos=0.08)
        pbuf.line("llr", np.arange(code.n), np.clip(s0, -lim, lim), color=NAVY, width=1.0,
                  name="soft bits (0 = never sent)")
        pbuf.set_ylim(-lim, lim * 1.45)
        # channel
        pch = self.plot("chan")
        if R["Htrue"] is not None:
            pch.line("ht", np.arange(72), db(np.abs(R["Htrue"][:, 7]) ** 2), color=GRAY, width=2.4,
                     name="true channel")
        Hest = lp.crs_channel_estimate(Y, PCI, SF)
        pch.line("he", np.arange(72), db(np.abs(Hest[:, 7]) ** 2), color=ORANGE, width=1.6,
                 style="--", name="CRS estimate")
        rows, _ = lp.crs_port0(PCI, 2 * SF + 1, 0)
        pch.scatter("pil", rows, db(np.abs(Hest[rows, 7]) ** 2), color=RED, size=7,
                    name="pilots")
        cbs = "".join("✓" if o else "✗" for o in R["cb_ok"])
        self.res = R
        self.readout(tbs=tbs, rate=code.k / E, cbs=cbs, tb="PASS" if R["tb_ok"] else "FAIL")

    def story(self, p):
        R = getattr(self, "res", None)
        if R is None:
            return ""
        code = ldpc_code()
        C, E, Qm = R["C"], R["E"], R["Qm"]
        s = (f"<p><b>Transmitter.</b> A {v(len(R['tb']), 'd')}-bit transport block gets a 24-bit "
             f"CRC, is cut into {v(C, 'd')} code block(s) of {code.k - 24} bits each with its "
             f"own CRC, and each is LDPC-encoded at rate 1/3 into a <b>circular buffer</b> of "
             f"{code.n} bits, systematic bits first. Rate matching reads E = {v(E, 'd')} bits "
             f"per block starting at redundancy version {p.rv}'s position (orange): code rate "
             f"{v(code.k / E, '.2f')}. The bits are interleaved, scrambled, mapped to "
             f"{2 ** Qm}-QAM and placed on the resource elements around the control region and "
             f"the CRS.</p>")
        s += (f"<p><b>Receiver.</b> Channel estimate from the CRS pilots (red dots, right), "
              f"equalise, soft bits (LLRs), descramble, de-interleave, add them back into the "
              f"circular buffer (bottom left: zeros where nothing was sent), LDPC decode, check "
              f"the CRCs: {v(self.r.get('cbs', ''))} → {v(self.r.get('tb', ''))}.</p>")
        if p.prof == "ETU":
            s += ("<p>ETU has 5 µs of delay spread, about the cyclic prefix, and the CRS sit "
                  "every 6 subcarriers (90 kHz): the estimate cannot follow the fast ripple in "
                  "frequency, so the equalised points stay blurred even at high SNR.</p>")
        return "<h3>One transport block's journey</h3>" + s + keybox(
            "Every stage exists for a reason: CRCs for HARQ, the circular buffer for any code "
            "rate, interleaving and scrambling against bursts and interference, pilots for "
            "the equaliser.")


# =============================================================================== 5. HARQ
HARQ_MCS = {"QPSK, R≈0.70 (3 CBs)": (2, 3), "16QAM, R≈0.46 (4 CBs)": (4, 4)}


class HARQCombining(Experiment):
    title = "HARQ: chase vs incremental redundancy"
    blurb = "Same bits again, or new parity? Watch the circular buffer fill and the BLER fall."
    book = "sec:ch21:nrcoding"
    heavy = True
    controls = [
        IntSlider("k", "Transmissions so far", 1, 4, 1,
                  help="Chase repeats RV 0; incremental redundancy sends RV 0, 2, 3, 1"),
        Choice("mcs", "Modulation and coding", list(HARQ_MCS), style="menu"),
        Choice("effort", "Effort", ["Quick", "Thorough"]),
        Button("rerun", "Run again", primary=True),
    ]
    plots = [
        ImagePlot("cov", "Circular buffer coverage after k transmissions (one code block)",
                  x="buffer position", y=""),
        BERPlot("bler", "Residual BLER after k transmissions", x="SNR per RE (dB)",
                y="block error rate", ylim=(5e-3, 1.2), xlim=(-2, 10), legend="bl"),
        Plot("tput", "Throughput (one TB per 1 ms transmission)", x="SNR per RE (dB)",
             y="throughput (Mb/s)", xlim=(-2, 10), legend="tl"),
    ]
    layout = [["cov", "cov"], ["bler", "tput"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("rch", "Chase: code rate", "", ".2f"),
        Readout("rir", "IR: code rate", "", ".2f"),
        Readout("gain", "IR gain at 10 % BLER", "dB", ".1f", good=lambda x: x > 1),
        Readout("pts", "Points done", "", None),
    ]
    challenges = [
        Challenge("Send enough redundancy versions that IR has transmitted every parity bit "
                  "(code rate ≤ 1/3).",
                  lambda s: s.r.rir <= 0.334),
        Challenge("Measure an IR gain of at least 2 dB at 10 % residual BLER.",
                  lambda s: s.r.gain is not None and s.r.gain >= 2,
                  hint="Look after two transmissions."),
        Challenge("Find where chase combining catches up: with 16QAM R≈0.46, an IR gain under "
                  "1 dB after two transmissions.",
                  lambda s: (s.p.mcs.startswith("16QAM") and s.p.k == 2 and s.r.gain is not None
                             and s.r.gain < 1)),
    ]

    def setup(self):
        self.run_id = 0
        self.res = {}

    def on_rerun(self, p):
        self.run_id += 1

    def coverage(self, p):
        code = ldpc_code()
        Qm, C = HARQ_MCS[p.mcs]
        nre = chain_consts()[2]
        E = nre * Qm // C
        img = np.zeros((2, code.n))
        for mode, row in (("chase", 1), ("ir", 0)):
            rvs = (0, 2, 3, 1) if mode == "ir" else (0, 0, 0, 0)
            cnt = np.zeros(code.n)
            for t in range(p.k):
                np.add.at(cnt, (lp.rv_start(rvs[t], code.n) + np.arange(E)) % code.n, 1)
            img[row] = np.minimum(cnt, 3)
        rate_ch = code.k / E
        rate_ir = code.k / min(p.k * E, code.n) if p.k * E < code.n else code.k / code.n
        rate_ir = code.k / (p.k * E) if p.k * E <= code.n else code.k / code.n
        return img, E, rate_ch, rate_ir

    def update(self, p):
        img, E, rch, rir = self.coverage(p)
        pc = self.plot("cov")
        dark = pc.theme.name == "dark"
        cols = (["#1E252E", "#2E6DA4", "#5DADE2", "#AED6F1"] if dark else
                ["#F1F4F8", "#AED6F1", "#5D9CCF", "#1B3A5C"])
        pc.image("c", img, x=(0, 1152), y=(0, 2), cmap=cat_cmap(cols), levels=(-0.5, 3.5))
        pc.set_yticks([(0.5, "IR"), (1.5, "chase")])
        pc.vline("sys", 384, color=GREEN, style="-", width=2.0, label="systematic | parity",
                 label_pos=0.5)
        pc.set_xlim(0, 1152)
        pc.set_ylim(0, 2)
        pc.set_title(f"Circular buffer after {p.k} transmission(s): pale = sent once, dark = "
                     f"sent 3+ times, white = never sent")
        self.key = (p.mcs, p.effort, self.run_id)
        self.res.setdefault(self.key, {})
        self.draw_results(p)
        self.readout(rch=rch, rir=rir)

    def draw_results(self, p):
        R = self.res.get(self.key, {})
        pb, pt = self.plot("bler"), self.plot("tput")
        k = p.k
        gain = None
        th = {}
        for mode, col in (("chase", NAVY), ("ir", RED)):
            pts = sorted(R.get(mode, {}).items())
            if pts:
                x = np.array([a for a, _ in pts])
                y = np.array([b["bl"][k - 1] for _, b in pts])
                pb.sim(mode, x, np.maximum(y, 6e-3), color=col,
                       name=("chase" if mode == "chase" else "incremental redundancy") + f", {k} tx")
                y1 = np.array([b["bl"][0] for _, b in pts])
                pb.sim(mode + "1", x, np.maximum(y1, 6e-3), color=GRAY, size=5, connect=True,
                       name="first transmission" if mode == "chase" else None)
                pt.line(mode, x, [b["tp"] for _, b in pts], color=col, width=2.2,
                        name="chase" if mode == "chase" else "IR")
                pt.scatter(mode + "p", x, [b["tp"] for _, b in pts], color=col, size=7)
                lg = np.log10(np.maximum(y, 1e-3))
                if lg.max() > -1 > lg.min():
                    i = np.flatnonzero(lg <= -1)[0]
                    th[mode] = float(np.interp(-1, [lg[i], lg[i - 1]], [x[i], x[i - 1]])) if i else None
        pb.hline("t10", 0.1, color=GRAY, style=":", label="10 %", label_pos=0.9)
        if th.get("chase") is not None and th.get("ir") is not None:
            gain = th["chase"] - th["ir"]
        elif k == 1 and th.get("chase") is not None:
            gain = 0.0
        nd = sum(len(R.get(m_, {})) for m_ in ("chase", "ir"))
        tot = getattr(self, "npts", 0) or nd
        self.readout(gain=gain, pts=f"{nd}" + (" ✓" if R.get("done") else " …"))

    def background(self, p):
        Qm, C = HARQ_MCS[p.mcs]
        thorough = p.effort == "Thorough" and not self.quick
        n_tb = 3 if self.quick else (32 if thorough else 12)
        snrs = np.arange(-2, 10.1, 3.0 if self.quick else 1.5)
        key = (p.mcs, p.effort, self.run_id)
        for s_ in snrs:
            for mode in ("chase", "ir"):
                r = np.random.default_rng(int(1000 + 10 * s_) + 7919 * self.run_id)
                yield dict(key=key, mode=mode, snr=float(s_),
                           res=self.harq(s_, mode, n_tb, Qm, C, r))
        yield dict(key=key, done=True)

    def harq(self, snr_db, mode, n_tb, Qm, C, r, max_tx=4):
        code = ldpc_code()
        rvs = (0, 2, 3, 1) if mode == "ir" else (0, 0, 0, 0)
        data = []
        for _ in range(n_tb):
            tb = r.integers(0, 2, tb_size(C)).astype(np.int8)
            data.append(pdsch_encode(tb, Qm, C))
        soft = [None] * n_tb
        done_at = np.zeros(n_tb, int)
        for k in range(max_tx):
            act = np.flatnonzero(done_at == 0)
            if len(act) == 0:
                break
            for t in act:
                cbufs, E = data[t]
                y, n0 = channel(lp.ofdm_mod(pdsch_modulate(cbufs, E, Qm, rvs[k])), snr_db, "AWGN", r)
                soft[t] = pdsch_demod(y, n0, Qm, C, E, rvs[k], soft[t])[0]
            lcw = np.concatenate([soft[t] for t in act])
            full = np.empty_like(lcw)
            full[:, code.info], full[:, code.pivots] = lcw[:, :code.k], lcw[:, code.k:]
            d, _ = lp.ldpc_decode_batch(code, full, iters=20)
            blocks = d[:, code.info].reshape(len(act), C, -1)
            kcb = code.k - 24
            for j, t in enumerate(act):
                if lp.crc_ok(np.concatenate([b[:kcb] for b in blocks[j]])):
                    done_at[t] = k + 1
        bl = [1 - float(np.mean((done_at > 0) & (done_at <= k))) for k in range(1, max_tx + 1)]
        ntx = np.where(done_at > 0, done_at, max_tx)
        tp = float(np.sum(done_at > 0) * tb_size(C) / np.sum(ntx) / 1e3)
        return dict(bl=bl, tp=tp)

    def progress(self, p, it):
        R = self.res.setdefault(it["key"], {})
        if it.get("done"):
            R["done"] = True
        else:
            R.setdefault(it["mode"], {})[it["snr"]] = it["res"]
        if it["key"] == self.key:
            self.draw_results(p)

    def story(self, p):
        r = self.r
        s = ("<p>When a transport block fails, the phone keeps its soft bits and sends a NACK. "
             "With <b>chase combining</b> the retransmission repeats the very same coded bits "
             "(RV 0): the soft values add, which is worth 3 dB per doubling, but the code stays "
             f"the punctured rate-{r.get('rch', 0):.2f} code. With <b>incremental redundancy</b> "
             "the base station reads further round the circular buffer (RV 2, 3, 1): new parity "
             f"bits, so after {p.k} transmission(s) the decoder sees a rate-"
             f"{v(r.get('rir', 0), '.2f')} code.</p>")
        g = r.get("gain")
        if g is not None and p.k > 1:
            s += (f"<p>Measured: IR reaches 10 % residual BLER {v(g, '.1f', 'dB')} earlier than "
                  f"chase after {p.k} transmissions. A lower-rate code is worth more than the "
                  f"same SNR spent on repetition, which is why LTE and NR use IR with the RV "
                  f"order 0, 2, 3, 1.</p>")
        else:
            s += ("<p>Each point is a Monte Carlo run of real transport blocks through the "
                  "full chain of experiment 4 (AWGN). Move the transmissions slider to compare "
                  "after 2, 3 and 4 tries.</p>")
        return "<h3>Retransmit smarter, not louder</h3>" + s + keybox(
            "HARQ with incremental redundancy turns every NACK into extra parity: the code rate "
            "adapts to the channel after the fact.")


# =============================================================================== 6. CQI and throughput
class CQIThroughput(Experiment):
    title = "CQI, MCS and throughput"
    blurb = "From an SNR report to a CQI, a modulation, a code rate and megabits per second."
    book = "sec:ch21:lte"
    controls = [
        Slider("snr", "Reported SINR", -10, 25, 8, step=0.5, unit="dB"),
        Slider("gap", "Implementation gap to Shannon", 0, 6, 2, step=0.25, unit="dB",
               help="How far a real receiver sits from capacity (coding, estimation, …)"),
        Heading("Carrier"),
        Choice("bw", "LTE bandwidth", ["1.4", "3", "5", "10", "15", "20"], "20", style="menu",
               help="MHz"),
        Choice("layers", "MIMO layers", ["1", "2", "4"], "2"),
        Heading("HARQ panel"),
        Slider("r0", "First-transmission rate", 0.5, 5.5, 3.0, step=0.1, unit="b/s/Hz",
               help="A fixed, deliberately optimistic MCS; HARQ recovers the failures"),
    ]
    plots = [
        Plot("cqi", "The LTE CQI table against Shannon", x="SINR (dB)",
             y="spectral efficiency (b/s/Hz)", xlim=(-10, 25), ylim=(0, 7.2), legend="tl"),
        Plot("mbps", "Throughput of the carrier at 10 % BLER", x="SINR (dB)", y="throughput (Mb/s)",
             xlim=(-10, 25), legend="tl"),
        Plot("harq", "HARQ at a fixed first-transmission rate", x="SNR (dB)",
             y="throughput (b/s/Hz)", xlim=(-10, 20), ylim=(0, 6), legend="tl"),
    ]
    layout = [["cqi", "harq"], ["mbps", "harq"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("cqi", "CQI", "", None),
        Readout("mod", "Modulation, code rate", "", None),
        Readout("se", "Spectral efficiency", "b/s/Hz", ".2f"),
        Readout("tput", "Throughput", "Mb/s", ".1f"),
    ]
    challenges = [
        Challenge("Find the lowest SINR where 64QAM appears (CQI 10), within 0.25 dB.",
                  lambda s: s.r.cqi == "10" and s.exp.cqi_below != "10"),
        Challenge("Reach 100 Mb/s on a 10 MHz carrier with two layers.",
                  lambda s: s.p.bw == "10" and s.p.layers == "2" and s.r.tput >= 100),
        Challenge("HARQ panel: find an SNR where incremental redundancy beats chase combining "
                  "by 0.5 b/s/Hz or more.",
                  lambda s: s.exp.ir_minus_chase >= 0.5),
    ]
    OH = 13600 / 16800          # Chapter 21: 20 MHz, CFI 1, 4 CRS ports

    def update(self, p):
        se = ai.spectral_efficiency(ai.LTE_CQI)
        req = ai.required_snr_db(se, p.gap)
        x = np.linspace(-10, 25, 701)
        idx = np.searchsorted(req, x, side="right") - 1
        stair = np.where(idx >= 0, se[np.clip(idx, 0, None)], 0) * 0.9
        pc = self.plot("cqi")
        pc.line("sh", x, np.log2(1 + 10 ** (x / 10)), color=GRAY, width=1.4, style="--",
                name="Shannon")
        pc.line("st", x, stair, color=NAVY, width=2.2, name="best CQI × 0.9 (10 % BLER)")
        q = np.array([a for a, _ in ai.LTE_CQI])
        for qm, col, nm in ((2, GREEN, "QPSK"), (4, ORANGE, "16QAM"), (6, RED, "64QAM")):
            m = q == qm
            pc.scatter(f"q{qm}", req[m], se[m], color=col, size=8, name=nm)
        i = ai.select_entry(p.snr, req)
        i_b = ai.select_entry(p.snr - 0.25, req)
        self.cqi_below = str(i_b + 1) if i_b >= 0 else "0"
        cur = se[i] * 0.9 if i >= 0 else 0.0
        pc.vline("now", p.snr, color=PURPLE, style=":")
        pc.scatter("now", [p.snr], [cur], color=PURPLE, size=15, symbol="d")
        nrb = ai.LTE_RB[float(p.bw)] if p.bw == "1.4" else ai.LTE_RB[int(p.bw)]
        L = int(p.layers)
        res = 12 * nrb * 14 * 1000 * self.OH * L
        pm = self.plot("mbps")
        pm.line("t", x, stair * res / 1e6, color=NAVY, width=2.2, name=f"{p.bw} MHz, {L} layer(s)")
        pm.line("sh", x, np.log2(1 + 10 ** (x / 10)) * res / 1e6, color=GRAY, width=1.2,
                style="--", name="Shannon, same REs")
        tput = cur * res / 1e6
        pm.scatter("now", [p.snr], [tput], color=PURPLE, size=15, symbol="d")
        pm.set_ylim(0, 1.15 * 6.2 * res / 1e6)
        ph = self.plot("harq")
        s2 = np.linspace(-10, 20, 301)
        vals = {}
        for sch, col, nm in (("none", GRAY, "no HARQ"), ("chase", NAVY, "chase (4 tx)"),
                             ("ir", RED, "incremental redundancy (4 tx)")):
            vals[sch] = ai.harq_throughput(s2, p.r0, p.gap, sch)
            ph.line(sch, s2, vals[sch], color=col, width=2.0, name=nm)
        ph.line("cap", s2, np.log2(1 + 10 ** (s2 / 10) / 10 ** (p.gap / 10)), color=GRAY,
                width=1.0, style=":", name="capacity with the gap")
        ph.vline("now", p.snr, color=PURPLE, style=":")
        ph.set_ylim(0, max(2.0, 1.6 * p.r0))
        k_ = int(np.argmin(np.abs(s2 - p.snr)))
        self.ir_minus_chase = float(vals["ir"][k_] - vals["chase"][k_])
        for sch, col in (("chase", NAVY), ("ir", RED)):
            ph.scatter(sch + "p", [p.snr], [vals[sch][k_]], color=col, size=11, symbol="d")
        if i >= 0:
            qm, rr = ai.LTE_CQI[i]
            mod = f"{ai.QAM_NAME[qm]}, {rr / 1024:.2f}"
        else:
            mod = "out of range"
        self.readout(cqi=str(i + 1) if i >= 0 else "0", mod=mod, se=se[i] if i >= 0 else 0.0,
                     tput=tput)

    def story(self, p):
        r = self.r
        s = (f"<p>The phone measures {v(p.snr, '.1f', 'dB')} and reports CQI {v(r.get('cqi', '0'))}: "
             f"the highest table entry it could receive at 10 % BLER. That is "
             f"{v(r.get('mod', ''))} ({v(r.get('se', 0), '.2f', 'b/s/Hz')}). The table's 15 steps "
             f"climb about 1.9 dB each, a fixed gap of {v(p.gap, '.2f', 'dB')} to the right of "
             f"Shannon (top left).</p>"
             f"<p>On a {p.bw} MHz carrier with {p.layers} layer(s), after control and reference "
             f"signals ({100 * (1 - self.OH):.0f} % overhead), that is "
             f"{v(r.get('tput', 0), '.1f', 'Mb/s')}.</p>"
             f"<p>The right panel holds the MCS fixed at {v(p.r0, '.1f', 'b/s/Hz')}: below its "
             f"threshold the first transmission fails, and HARQ decides what survives. Chase "
             f"combining only adds SNR; incremental redundancy adds parity, so at low SNR it "
             f"keeps delivering.</p>")
        return "<h3>Reporting, choosing, delivering</h3>" + s + keybox(
            "CQI → MCS is a lookup table; each step of about 2 dB buys about 0.37 b/s/Hz, the "
            "slope of Shannon's log₂(1 + SNR).")


# =============================================================================== 7. peak rate
SCS_OPT = ["15 kHz", "30 kHz", "60 kHz", "120 kHz"]
QM_OPT = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8, "1024QAM": 10}


class PeakRate(Experiment):
    title = "Peak-rate calculator (TS 38.306)"
    blurb = "Layers × bits × code rate × REs per second: where 5G's gigabits come from."
    book = "sec:ch21:nrcoding"
    controls = [
        Heading("Carrier"),
        Choice("mu", "Subcarrier spacing", SCS_OPT, "30 kHz", style="menu",
               help="15/30/60 kHz are FR1; 120 kHz is FR2 (millimetre wave)"),
        Choice("bw", "Bandwidth", ["5", "10", "20", "40", "50", "100", "200", "400"], "100",
               style="menu", help="MHz; snapped to the nearest bandwidth defined for the spacing"),
        Choice("dir", "Direction", ["DL", "UL"]),
        Heading("The phone"),
        IntSlider("layers", "MIMO layers", 1, 8, 2),
        Choice("qm", "Highest modulation", list(QM_OPT), "256QAM", style="menu"),
        Slider("f", "Scaling factor f", 0.4, 1.0, 1.0, step=0.05,
               help="Signalled per band combination: 1, 0.8, 0.75 or 0.4"),
        Heading("Deployment"),
        IntSlider("ncc", "Aggregated carriers", 1, 16, 1),
        Toggle("tdd", "TDD DDDSU (≈74 % downlink)", False, enabled_if=lambda p: p.dir == "DL"),
    ]
    plots = [
        BarPlot("chain", "The formula, factor by factor", y="rate (b/s, log scale)"),
        Plot("vsbw", "Peak rate vs bandwidth: the spacing hardly matters", x="bandwidth (MHz)",
             y="peak rate (Gb/s)", logx=True, logy=True, xlim=(5, 400), ylim=(0.01, 100),
             legend="tl"),
        BarPlot("cmp", "Against the milestones", y="peak rate (Gb/s, log scale)"),
    ]
    layout = [["chain", "chain"], ["vsbw", "cmp"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("rate", "Peak rate", "Gb/s", ".2f"),
        Readout("nprb", "Resource blocks", "", "int"),
        Readout("res", "REs per second", "M", ".1f"),
        Readout("se", "Spectral efficiency", "b/s/Hz", ".1f"),
    ]
    challenges = [
        Challenge("Reproduce Chapter 21's FR1 example: 100 MHz at 30 kHz, 4 layers, 256QAM, "
                  "f = 1: 2.34 Gb/s.",
                  lambda s: (s.p.mu == "30 kHz" and s.exp.bw == 100 and s.p.layers == 4
                             and s.p.qm == "256QAM" and abs(s.r.rate - 2.34) < 0.01 and s.p.ncc == 1
                             and s.p.dir == "DL")),
        Challenge("Reproduce the FR2 example: 400 MHz at 120 kHz, 2 layers, 64QAM: 3.23 Gb/s.",
                  lambda s: (s.p.mu == "120 kHz" and s.exp.bw == 400 and s.p.layers == 2
                             and s.p.qm == "64QAM" and abs(s.r.rate - 3.23) < 0.01 and s.p.ncc == 1)),
        Challenge("Meet IMT-2020's 20 Gb/s downlink target with a single carrier.",
                  lambda s: s.p.ncc == 1 and s.p.dir == "DL" and s.r.rate >= 20,
                  hint="Only the widest millimetre-wave carrier, eight layers and 1024QAM get "
                       "there: a specification number, not a phone."),
    ]

    def update(self, p):
        mu = SCS_OPT.index(p.mu)
        bw = nearest_bw(mu, float(p.bw))
        self.bw = bw
        nprb = ai.NR_MAX_PRB[(mu, bw)]
        fr = "FR2" if mu == 3 else "FR1"
        oh = ai.NR_OVERHEAD[(fr, p.dir)]
        qm = QM_OPT[p.qm]
        if p.dir == "UL" and qm > 8:
            qm = 8
        res = 12 * nprb / ai.nr_symbol_time(mu)
        tdd = 0.74 if (p.tdd and p.dir == "DL") else 1.0
        steps = [("REs/s", res), (f"× {p.layers} layers", p.layers), (f"× Qm {qm}", qm),
                 ("× R 948/1024", 948 / 1024), (f"× (1 − OH {oh:.2f})", 1 - oh),
                 (f"× f {p.f:.2f}", p.f), (f"× {p.ncc} carrier{'s' if p.ncc > 1 else ''}", p.ncc),
                 ("× TDD 0.74" if tdd < 1 else "× FDD 1", tdd)]
        run = 1.0
        hs = []
        for _, fac in steps:
            run *= fac
            hs.append(run)
        rate = run
        pc = self.plot("chain")
        lv = np.log10(hs)
        base = 6.0
        pc.bars("b", np.arange(len(steps)), lv, base=base, width=0.62,
                colors=[NAVY] + [GREEN if f_ > 1 else (RED if f_ < 1 else GRAY) for _, f_ in steps[1:]])
        for i, h in enumerate(hs):
            lab = f"{h / 1e9:.2f} G" if h >= 1e9 else f"{h / 1e6:.0f} M"
            pc.text(f"t{i}", i, lv[i], lab, anchor=(0.5, 1.05), size=9, bold=i == len(steps) - 1)
        pc.set_xticks([(i, nm) for i, (nm, _) in enumerate(steps)])
        sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        pc.set_yticks([(k, "10" + str(k).translate(sup)) for k in range(6, 13)])
        pc.set_ylim(base, max(11.3, lv.max() + 0.5))
        pc.set_xlim(-0.6, len(steps) - 0.4)
        # vs bandwidth
        pv = self.plot("vsbw")
        cols = [GREEN, NAVY, ORANGE, RED]
        for m_ in range(4):
            bws = ai.nr_bandwidths(m_)
            frm = "FR2" if m_ == 3 else "FR1"
            r_ = [ai.nr_peak_rate(p.layers, qm, ai.NR_MAX_PRB[(m_, b)], m_,
                                  ai.NR_OVERHEAD[(frm, p.dir)], p.f) * p.ncc * tdd / 1e9 for b in bws]
            pv.line(f"m{m_}", bws, r_, color=cols[m_], width=2.4 if m_ == mu else 1.3,
                    name=SCS_OPT[m_])
            pv.scatter(f"p{m_}", bws, r_, color=cols[m_], size=5)
        pv.scatter("now", [bw], [rate / 1e9], color=PURPLE, size=16, symbol="d")
        # milestones
        pm = self.plot("cmp")
        names = ["LTE Cat 4", "LTE Cat 16", "IMT-Adv.", "yours", "IMT-2020"]
        vals = np.array([0.15, 1.0, 1.0, rate / 1e9, 20.0])
        lvv = np.log10(np.maximum(vals, 1e-3))
        pm.bars("b", np.arange(5), lvv, base=-2, width=0.6, colors=[GRAY, GRAY, ORANGE, PURPLE, RED])
        for i, val in enumerate(vals):
            pm.text(f"t{i}", i, lvv[i], f"{val:.3g}", anchor=(0.5, 1.05), size=9, bold=i == 3)
        pm.set_xticks([(i, n_) for i, n_ in enumerate(names)])
        pm.set_yticks([(k, f"{10 ** k:g}") for k in (-2, -1, 0, 1, 2)])
        pm.set_ylim(-2, 2.6)
        pm.set_xlim(-0.6, 4.6)
        self.readout(rate=rate / 1e9, nprb=nprb, res=res / 1e6,
                     se=rate / (bw * 1e6 * p.ncc))

    def story(self, p):
        r = self.r
        mu = SCS_OPT.index(p.mu)
        s = (f"<p>A {self.bw} MHz carrier at {p.mu} holds {v(r.get('nprb', 0), 'd')} resource "
             f"blocks of 12 subcarriers; each OFDM symbol (with its prefix) lasts "
             f"{v(ai.nr_symbol_time(mu) * 1e6, '.2f', 'µs')}, so the carrier offers "
             f"{v(r.get('res', 0), '.1f')} million resource elements per second. Every factor "
             f"after that multiplies (top): layers, bits per symbol, the top code rate 948/1024, "
             f"one minus the control/pilot overhead, and the capability scaling.</p>"
             f"<p>Result: {v(r.get('rate', 0), '.2f', 'Gb/s')}, "
             f"{v(r.get('se', 0), '.1f', 'b/s/Hz')}. ")
        if r.get("rate", 0) >= 20:
            s += good("That meets the IMT-2020 peak requirement") + ", on paper.</p>"
        else:
            s += ("Phones advertise this number; what users see is set by the SINR and the "
                  "load (experiments 4–6).</p>")
        return "<h3>Where the gigabits come from</h3>" + s + keybox(
            "R ≈ ν · Q<sub>m</sub> · f · R<sub>max</sub> · 12·N<sub>PRB</sub>/T<sub>s</sub> · (1 − OH), "
            "summed over carriers (TS 38.306).")


# =============================================================================== 8. PAPR
PAPR_MODS = ["QPSK", "16QAM", "64QAM", "π/2-BPSK"]


def papr_symbols(mod, n, rng):
    """n unit-power data symbols; π/2-BPSK rotates every other symbol by 90° (TS 38.211 5.1.1)."""
    if mod == "π/2-BPSK":
        b = 1 - 2 * rng.integers(0, 2, n)
        return b * np.exp(1j * np.pi / 2 * (np.arange(n) % 2)) * np.exp(1j * np.pi / 4)
    con = cl.get_constellation({"QPSK": "qpsk", "16QAM": "16qam", "64QAM": "64qam"}[mod])
    return con.modulate(cl.random_bits(con.k * n, rng))


def ofdma_wave(sym, M, nfft):
    """OFDMA: the data symbols go straight onto M adjacent subcarriers centred on DC."""
    s = sym.reshape(-1, M)
    X = np.zeros((len(s), nfft), complex)
    X[:, (np.arange(M) - M // 2) % nfft] = s
    return np.fft.ifft(X, axis=1) * np.sqrt(nfft)


def dfts_wave(sym, M, nfft, fdss=False):
    """DFT-s-OFDM (SC-FDMA) with the contiguous (localized) mapping of LTE/NR: an M-point DFT,
    optional spectral shaping, then the same M adjacent subcarriers as OFDMA."""
    if not fdss:
        cfg = co.OFDMConfig(nfft, M, 0)
        return co.dft_s_ofdm_modulate(sym, cfg, contiguous=True).reshape(-1, nfft)
    s = sym.reshape(-1, M)
    S = np.fft.fft(s, axis=1) / np.sqrt(M)
    W = 1 + np.exp(-2j * np.pi * np.arange(M) / M)             # the [1, 1] filter of FDSS
    S = np.fft.fftshift(S * W / np.sqrt(np.mean(np.abs(W) ** 2)), axes=1)
    X = np.zeros((len(s), nfft), complex)
    X[:, (np.arange(M) - M // 2) % nfft] = S
    return np.fft.ifft(X, axis=1) * np.sqrt(nfft)


def papr_rows(x):
    pw = np.abs(x) ** 2
    return db(pw.max(axis=1) / pw.mean(axis=1))


def ccdf_fast(vals, grid):
    s = np.sort(np.asarray(vals))
    return 1.0 - np.searchsorted(s, grid, side="right") / max(len(s), 1)


def papr_at(grid, cc, p=1e-3):
    """PAPR₀ where the CCDF crosses p (interpolated on a log scale)."""
    if cc[0] < p:
        return float(grid[0])
    i = int(np.argmax(cc < p))
    if i == 0:
        return float(grid[-1])
    lc = np.log10(np.maximum(cc, 1e-9))
    return float(np.interp(np.log10(p), [lc[i], lc[i - 1]], [grid[i], grid[i - 1]]))


class UplinkPAPR(Experiment):
    title = "PAPR: OFDMA vs DFT-s-OFDM"
    blurb = "Why the phone transmits single-carrier: peaks, back-off and battery."
    book = "sec:ch17:dfts"
    controls = [
        Heading("Data"),
        Choice("mod", "Modulation", PAPR_MODS, "16QAM",
               help="π/2-BPSK is NR's uplink option for coverage-limited phones (DFT-s-OFDM only)"),
        IntSlider("nprb", "Allocation", 1, 50, 25, unit="PRB",
                  help="Resource blocks of 12 subcarriers given to this transmission "
                       "(25 PRB = 300 subcarriers = 4.5 MHz at 15 kHz)"),
        Toggle("fdss", "Spectral shaping on DFT-s-OFDM (FDSS)", False,
               help="Frequency-domain spectral shaping: weight the DFT outputs by "
                    "|1 + e^(−j2πk/M)|, i.e. filter the symbols with [1, 1]. NR allows it with "
                    "π/2-BPSK"),
        Heading("Measurement"),
        Choice("L", "Oversampling", [1, 2, 4, 8], 4, labels=["1×", "2×", "4×", "8×"],
               help="IFFT size = oversampling × allocated subcarriers. At 1× you only see the "
                    "samples, not the analog peaks between them"),
        Button("again", "New data"),
    ]
    plots = [
        Plot("ccdf", "How often the peaks exceed a level (CCDF of PAPR)", x="PAPR₀ (dB)",
             y="P(PAPR > PAPR₀)", xlim=(0, 13), ylim=(1e-3, 1), logy=True, legend="bl"),
        Plot("pa", "What the back-off costs a class-B amplifier", x="back-off from saturation (dB)",
             y="efficiency (%)", xlim=(0, 13), ylim=(0, 85), legend="tr"),
        Plot("env", "Instantaneous power: a close-up of 24 data-symbol periods",
             x="time (data-symbol periods, T_sym / M)", y="|x|² / mean (dB)", xlim=(0, 24),
             ylim=(-25, 22), legend="tl", legend_cols=3),
    ]
    layout = [["ccdf", "pa"], ["env", "env"]]
    row_stretch = [3, 2]
    col_stretch = [3, 2]
    readouts = [
        Readout("ofdma", "OFDMA PAPR at 10⁻³", "dB", ".1f"),
        Readout("dfts", "DFT-s-OFDM PAPR at 10⁻³", "dB", ".1f"),
        Readout("adv", "DFT-s advantage", "dB", ".1f"),
        Readout("gain", "PA efficiency gain", "×", ".2f"),
    ]
    challenges = [
        Challenge("Fall into the sampling trap: make DFT-s-OFDM's measured PAPR read below 1 dB.",
                  lambda s: s.r.dfts < 1,
                  hint="Measure at the symbol rate with a constant-envelope constellation: the "
                       "samples are then the symbols themselves, and the peaks hide between them."),
        Challenge("With at least 4× oversampling and no spectral shaping, win 4.5 dB or more "
                  "over OFDMA.",
                  lambda s: s.p.L >= 4 and not s.p.fdss and s.r.adv >= 4.5,
                  hint="Pick a constellation whose consecutive symbols never pass through the "
                       "origin."),
        Challenge("Get DFT-s-OFDM's PAPR at 10⁻³ under 3 dB, measured at 4× or more.",
                  lambda s: s.p.L >= 4 and s.r.dfts < 3,
                  hint="NR's coverage recipe: two tricks together."),
    ]
    grid = np.linspace(0, 13, 261)

    def setup(self):
        self.cache = {}
        self.cur = None
        self.seed = 1

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def cache_key(p):
        return (p.mod, int(p.nprb), int(p.L), bool(p.fdss))

    @staticmethod
    def waves(p, nsym, rng):
        M = 12 * int(p.nprb)
        nfft = int(p.L) * M
        xo = ofdma_wave(papr_symbols(p.mod, M * nsym, rng), M, nfft)
        xd = dfts_wave(papr_symbols(p.mod, M * nsym, rng), M, nfft, p.fdss)
        return xo, xd

    def update(self, p):
        M, L = 12 * int(p.nprb), int(p.L)
        nsym = int(np.ceil(24 / M)) + 1
        xo, xd = self.waves(p, nsym, np.random.default_rng(self.seed))
        pe = self.plot("env")
        n = 24 * L + 1
        t = np.arange(n) / L
        for k, (x, col, nm) in enumerate(((xo, NAVY, "OFDMA (downlink)"),
                                          (xd, RED, "DFT-s-OFDM (uplink)"))):
            xr = x.ravel()
            pw = db(np.abs(xr[:n]) ** 2 / np.mean(np.abs(xr) ** 2))
            pe.line(f"p{k}", t, pw, color=col, width=2.0 if k else 1.4, name=nm)
            if k == 1 and L > 1:
                pe.scatter("s1", t[::L], pw[::L], color=RED, size=6,
                           name="DFT-s samples at 1× (the data symbols)")
        pe.hline("m", 0, color=GRAY, style="--", label="mean", label_pos=0.02)
        self.cur = self.cache.get(self.cache_key(p))
        self.draw_ccdf(p)

    def draw_ccdf(self, p):
        pc = self.plot("ccdf")
        pc.hline("e3", 1e-3 * 1.02, color=GRAY, style=":", width=0.8)
        vals = {}
        if self.cur is not None:
            for k, col, nm in (("o", NAVY, "OFDMA (downlink)"), ("d", RED, "DFT-s-OFDM (uplink)")):
                cc = ccdf_fast(self.cur[k], self.grid)
                pc.line(k, self.grid, np.where(cc > 0, cc, np.nan), color=col, width=2.0, name=nm)
                vals[k] = papr_at(self.grid, cc)
        pa = self.plot("pa")
        bo = np.linspace(0, 13, 200)
        pa.line("eta", bo, 78.5 * 10 ** (-bo / 20), color=GREEN, width=2.0,
                name="class B: 78.5 % × 10^(−BO/20)")
        for k, col in (("o", NAVY), ("d", RED)):
            if k in vals:
                e = 78.5 * 10 ** (-vals[k] / 20)
                pa.scatter("m" + k, [vals[k]], [e], color=col, size=12, symbol="d")
                pa.text("t" + k, vals[k] + 0.25, e + 2.5, f"{e:.0f} %", color=col, size=9)
        if vals:
            self.readout(ofdma=vals["o"], dfts=vals["d"], adv=vals["o"] - vals["d"],
                         gain=10 ** ((vals["o"] - vals["d"]) / 20))
        else:
            nan = float("nan")
            self.readout(ofdma=nan, dfts=nan, adv=nan, gain=nan)

    def background(self, p):
        ck = self.cache_key(p)
        if ck in self.cache:
            return
        rng = np.random.default_rng(77)
        target = 300 if self.quick else 3000
        M = 12 * int(p.nprb)
        chunk = max(50, min(1000, 120000 // (M * int(p.L))))
        acc = {"o": [], "d": []}
        done = 0
        while done < target:
            xo, xd = self.waves(p, chunk, rng)
            acc["o"].append(papr_rows(xo))
            acc["d"].append(papr_rows(xd))
            done += chunk
            yield ck, {k: np.concatenate(v_) for k, v_ in acc.items()}, done >= target

    def progress(self, p, item):
        ck, vals, final = item
        if final:
            self.cache[ck] = vals
        if ck == self.cache_key(p):
            self.cur = vals
            self.draw_ccdf(p)

    def story(self, p):
        r = self.r
        o, d = r.get("ofdma", float("nan")), r.get("dfts", float("nan"))
        M = 12 * int(p.nprb)
        s = (f"<p>Your {v(int(p.nprb), 'd')} resource blocks are {v(M, 'd')} subcarriers. In "
             "OFDMA (the downlink) each carries its own symbol, so a sample is the sum of "
             f"{M} independent phasors: nearly Gaussian, mostly modest, occasionally huge. "
             f"DFT-s-OFDM (SC-FDMA, the LTE uplink) first spreads the symbols with a {M}-point "
             "DFT; after the IFFT the signal is the symbol stream again, a single carrier that "
             "only rings between symbols (the close-up below: red dots are the data symbols, "
             "the red line what the amplifier really sees).</p>")
        if np.isfinite(o) and np.isfinite(d):
            s += (f"<p>Once in a thousand symbols OFDMA peaks {v(o, '.1f', 'dB')} above its mean, "
                  f"DFT-s-OFDM {v(d, '.1f', 'dB')}. The amplifier must be backed off by that "
                  f"much to pass the peaks cleanly, and an ideal class-B stage's efficiency falls "
                  f"as 10^(−BO/20): {v(o - d, '.1f', 'dB')} less back-off is "
                  f"{v(10 ** ((o - d) / 20), '.2f', '×')} the efficiency (right): battery life, "
                  "or a few dB more power at the cell edge.</p>")
        else:
            s += "<p>The CCDF curves are being measured in the background…</p>"
        if int(p.L) == 1:
            s += ("<p>" + bad("Careful: 1× sampling.") + " With the IFFT no larger than the "
                  "allocation you see only the sample instants; the analog waveform peaks between "
                  "them. For DFT-s-OFDM with QPSK the samples are the symbols themselves, so the "
                  "PAPR reads 0 dB. Measure at 4× or more.</p>")
        elif p.mod == "π/2-BPSK":
            s += ("<p>π/2-BPSK rotates every other symbol by 90°, so consecutive symbols never "
                  "jump through the origin. "
                  + ("With spectral shaping, which smooths the transitions, the envelope is nearly "
                     "constant: NR's recipe for phones at the cell edge. " if p.fdss else
                     "Turn on spectral shaping (FDSS) to smooth the transitions as well. ")
                  + "In OFDMA the same trick is useless: the subcarriers do not know about it.</p>")
        else:
            s += ("<p>Notice how little OFDMA cares about the constellation or the allocation: "
                  "beyond a few PRBs the central limit theorem sets its peaks. DFT-s-OFDM "
                  "inherits the constellation's own peaks, so it gains most with QPSK and least "
                  "with 64QAM.</p>")
        return "<h3>Peaks cost battery</h3>" + s + keybox(
            "OFDMA needs about 10–11 dB of back-off whatever the data; DFT-spreading gives the "
            "phone back 2–3 dB (more with π/2-BPSK), which is why LTE's uplink is SC-FDMA and NR "
            "keeps DFT-s-OFDM for coverage.")


# =============================================================================== the lab
LAB = st.Lab(27, "Inside an LTE/NR Downlink", chapter=21,
             chapter_title="Cellular Generations: AMPS to 5G",
             experiments=[ResourceGrid, SyncSequences, CellSearch, PDSCHChain, HARQCombining,
                          CQIThroughput, PeakRate, UplinkPAPR])

if __name__ == "__main__":
    st.run(LAB)
