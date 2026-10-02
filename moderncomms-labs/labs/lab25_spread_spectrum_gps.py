"""Lab 25 · Spread Spectrum, CDMA and GPS   (Chapter 18)

Run it:      python labs/lab25_spread_spectrum_gps.py
Self-test:   python labs/lab25_spread_spectrum_gps.py --selftest

Spread spectrum deliberately uses far more bandwidth than the data needs, and spends that
excess on robustness: against jammers, against multipath, against other users, and to
measure time to a few nanoseconds. Eight experiments: the pseudo-noise codes that make it
work, a DSSS link under a tone jammer, frequency hopping against Wi-Fi and a smart jammer,
the RAKE receiver that turns multipath into diversity, the near–far problem of CDMA and the
detectors that fix it, and finally a GPS receiver: acquiring a signal 20 dB below the noise,
tracking it with a DLL and a Costas loop, and solving for position.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache
from math import comb

import numpy as np

import commlib as cl
from commlib import spread as sp
from commlib import spreadkit as sk
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, ConstellationPlot, BERPlot, ImagePlot, BarPlot, Readout,
                    Challenge, NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-300))


def undb(x):
    return 10 ** (np.asarray(x, float) / 10)


# =============================================================================== 1. codes
FAMILIES = ["m-sequence", "Gold", "Small Kasami", "GPS C/A", "Random ±1"]


def family_degree(fam, m):
    """The nearest degree for which the family exists."""
    if fam == "Gold" and m % 4 == 0:
        return m + 1 if m < 10 else m - 1
    if fam == "Small Kasami" and m % 2:
        return m + 1 if m < 10 else m - 1
    if fam == "GPS C/A":
        return 10
    return m


@lru_cache(maxsize=32)
def code_set(fam, m):
    """Bipolar code family (rows) for a family name and degree."""
    if fam == "m-sequence":
        u = sp.bipolar(sp.msequence(m))
        return np.array([u, u[::-1]])                  # the reversed m-sequence is another one
    if fam == "Gold":
        return sp.bipolar(sp.gold_codes(m))
    if fam == "Small Kasami":
        return sp.bipolar(sp.kasami_small(m))
    if fam == "GPS C/A":
        return np.array([sp.bipolar(sp.gps_ca_code(k)) for k in range(1, 33)])
    r = np.random.default_rng(m)
    return r.choice([-1.0, 1.0], (40, 2 ** m - 1))


def family_bound(fam, m):
    if fam in ("Gold", "GPS C/A"):
        return sp.gold_bound(m)
    if fam == "Small Kasami":
        return 2 ** (m // 2) + 1
    return None


class PNCodes(Experiment):
    title = "PN, Gold and Kasami codes"
    blurb = "Codes that look like noise, correlate like a spike, and ignore each other."
    book = "sec:ch18:pn"
    controls = [
        Choice("fam", "Code family", FAMILIES, "Gold", style="menu"),
        IntSlider("m", "Shift-register length m", 3, 10, 7,
                  help="Code length N = 2^m − 1 chips (Gold needs m not a multiple of 4, "
                       "small Kasami an even m; the nearest valid m is used)"),
        IntSlider("a", "Code A (index or PRN)", 0, 31, 2),
        IntSlider("b", "Code B (index or PRN)", 0, 31, 9),
    ]
    plots = [
        Plot("chips", "The first 64 chips of code A", x="chip", y="value", xlim=(-0.5, 63.5),
             ylim=(-1.6, 1.9), legend=None),
        Plot("auto", "Autocorrelation of code A (periodic)", x="shift (chips)",
             y="correlation / N", ylim=(-0.25, 1.1), legend=None),
        Plot("cross", "Cross-correlation of A with B", x="shift (chips)", y="correlation / N",
             legend=None),
        BarPlot("hist", "Which cross-correlation values occur", x="correlation value",
                y="how often (%)"),
    ]
    layout = [["chips", "chips"], ["auto", "cross"], ["auto", "hist"]]
    row_stretch = [2, 3, 2]
    readouts = [
        Readout("N", "Code length", "chips", "int"),
        Readout("side", "Worst autocorrelation sidelobe", "% of N", ".1f"),
        Readout("cc", "Worst cross-correlation", "% of N", ".1f", good=lambda x: x < 10),
        Readout("bound", "Theory bound", "", None),
    ]
    challenges = [
        Challenge("Show why plain m-sequences make poor CDMA codes: an m-sequence pair of 127 "
                  "chips or more whose cross-correlation beats the Gold bound for that length.",
                  lambda s: s.p.fam == "m-sequence" and s.r.N >= 127
                  and s.r.cc > 100 * sp.gold_bound(s.exp.m_eff) / s.r.N,
                  hint="Gold's bound is about 2^((m+2)/2)/N; try m = 7 and m = 9."),
        Challenge("Choose a family and length (63 chips or more) whose worst cross-correlation "
                  "stays below 10 % of the length.",
                  lambda s: s.r.N >= 63 and s.r.cc < 10 and s.p.fam != "m-sequence"
                  and s.exp.distinct),
        Challenge("Random codes carry no guarantee: find a random 1023-chip code whose "
                  "autocorrelation sidelobe exceeds 11 % (an m-sequence's is 0.1 %).",
                  lambda s: s.p.fam == "Random ±1" and s.r.N >= 1023 and s.r.side > 11,
                  hint="Step through code A."),
    ]

    def setup(self):
        self.distinct = True
        self.m_eff = 7

    def update(self, p):
        m = family_degree(p.fam, p.m)
        self.m_eff = m
        C = code_set(p.fam, m)
        n = len(C)
        if p.fam == "GPS C/A":
            ia, ib = (p.a % 32), (p.b % 32)
            la, lb = f"PRN {ia + 1}", f"PRN {ib + 1}"
        else:
            ia, ib = p.a % n, p.b % n
            la, lb = f"code {ia}", f"code {ib}"
        self.distinct = ia != ib
        A, B = C[ia], C[ib]
        N = len(A)
        pc = self.plot("chips")
        pc.line("c", np.repeat(np.arange(min(64, N) + 1), 2)[1:-1] - 0.5,
                np.repeat(A[:min(64, N)], 2), color=NAVY, width=2.0)
        pc.set_title(f"The first {min(64, N)} chips of {la} ({N} chips per period)")
        ra = sp.pcorr(A) / N
        rb = sp.pcorr(A, B) / N
        sh = np.arange(N)
        cen = (sh + N // 2) % N - N // 2
        order = np.argsort(cen)
        pa = self.plot("auto")
        pa.line("r", cen[order], ra[order], color=NAVY, width=1.6)
        pa.scatter("pk", [0], [1.0], color=RED, size=10)
        pa.set_xlim(-N // 2, N // 2)
        side = float(np.max(np.abs(ra[1:]))) if N > 1 else 0.0
        pa.hline("s", side, color=RED, style=":", width=1.2)
        px = self.plot("cross")
        px.set_title(f"Cross-correlation of {la} with {lb}"
                     + (" (red: ± bound)" if family_bound(p.fam, m) is not None else ""))
        px.line("r", cen[order], rb[order], color=ORANGE, width=1.4, name=f"{la} × {lb}")
        bnd = family_bound(p.fam, m)
        if bnd is not None:
            px.hline("b1", bnd / N, color=RED, style="--", width=1.2)
            px.hline("b2", -bnd / N, color=RED, style="--", width=1.2)
        px.set_xlim(-N // 2, N // 2)
        lim = max(0.3, float(np.max(np.abs(rb))) * 1.3)
        px.set_ylim(-lim, lim * 1.15)
        vals, cnts = np.unique(np.round(rb * N).astype(int), return_counts=True)
        ph = self.plot("hist")
        if len(vals) <= 25:
            ph.bars("h", np.arange(len(vals)), 100 * cnts / N, width=0.7, color=ORANGE)
            ph.set_xticks([(i, str(val)) for i, val in enumerate(vals)])
            ph.set_xlim(-0.7, len(vals) - 0.3)
            ph.set_title(f"Cross-correlation takes only {len(vals)} value"
                         f"{'s' if len(vals) > 1 else ''}")
        else:
            h, e = np.histogram(rb * N, bins=25)
            ph.bars("h", np.arange(25), 100 * h / N, width=0.85, color=ORANGE)
            ph.set_xticks([(0, f"{e[0]:.0f}"), (12, f"{e[12]:.0f}"), (24, f"{e[-1]:.0f}")])
            ph.set_xlim(-0.7, 24.7)
            ph.set_title(f"Cross-correlation spreads over {len(vals)} values")
        if len(vals) <= 25:
            ph.set_ylim(0, max(100 * cnts / N) * 1.2)
        else:
            ph.set_ylim(0, 100 * h.max() / N * 1.2)
        cc = float(np.max(np.abs(rb))) if self.distinct else float("nan")
        self._st = dict(m=m, N=N, nvals=len(vals), fam=p.fam)
        self.readout(N=N, side=100 * side, cc=100 * cc if self.distinct else "—",
                     bound=(f"{bnd} = {100 * bnd / N:.1f} %" if bnd is not None else "—"))

    def story(self, p):
        stt = getattr(self, "_st", None)
        if stt is None:
            return ""
        s = ""
        if stt["m"] != p.m and p.fam in ("Gold", "Small Kasami"):
            s += (f"<p class='muted'>{p.fam} sets do not exist for m = {p.m}; showing "
                  f"m = {stt['m']}.</p>")
        if p.fam == "m-sequence":
            s += ("<p>A maximal-length shift register cycles through all 2^m − 1 non-zero states, "
                  "so its output is balanced and its periodic autocorrelation is perfect: N at "
                  "zero shift, exactly −1 everywhere else. That spike is what lets a receiver "
                  "find the code's timing.</p><p>But two different m-sequences can correlate "
                  "badly with each other (code B here is A played backwards), which is no good "
                  "when many users share a channel.</p>")
        elif p.fam in ("Gold", "GPS C/A"):
            s += ("<p>Gold (1967) XORed two carefully chosen m-sequences (a <b>preferred pair</b>) "
                  "with every relative shift, getting N + 2 codes whose cross-correlation takes "
                  f"only three values, all below t(m). The histogram shows {v(stt['nvals'], 'd')} "
                  "distinct values. ")
            if p.fam == "GPS C/A":
                s += ("Every GPS satellite transmits one of these 1023-chip Gold codes, a "
                      "millisecond long: the receiver separates them by correlation alone.")
            s += "</p>"
        elif p.fam == "Small Kasami":
            s += ("<p>The small Kasami set trades the number of codes (only 2^(m/2)) for even "
                  "lower cross-correlation, 2^(m/2) + 1, which meets the Welch bound: no family "
                  "of that size can do better.</p>")
        else:
            s += ("<p>Random ±1 codes look noise-like, but their correlations are only "
                  "statistically small (about √N): the worst sidelobe and cross-correlation are "
                  "several times larger than for designed codes, and not guaranteed.</p>")
        return "<h3>Noise you can recognise</h3>" + s + keybox(
            "Good spreading codes: sharp autocorrelation (timing) and low cross-correlation "
            "(many users). Gold and Kasami give both, with guarantees.")


# =============================================================================== 2. DSSS
GPS_CHOICES = ["15", "31", "63", "127", "255", "1023"]


class DSSSJammer(Experiment):
    title = "DSSS against a jammer"
    blurb = "Spread, jam, despread: the jammer is smeared out while your signal comes back."
    book = "sec:ch18:jamming"
    controls = [
        Choice("gp", "Processing gain (chips per bit)", GPS_CHOICES, "63", style="menu"),
        Slider("js", "Jammer-to-signal ratio J/S", -10, 40, 5, step=0.5, unit="dB"),
        Choice("jam", "Jammer", ["Tone", "Narrowband noise (2 %)"]),
        Slider("ebn0", "Eb/N0 (thermal noise)", 0, 20, 10, step=0.5, unit="dB"),
        Button("again", "New data"),
    ]
    plots = [
        SpectrumPlot("rx", "Received: the jammer towers over the spread signal",
                     x="frequency (× chip rate)", xlim=(-0.5, 0.5), ylim=(-60, 45), legend="tr"),
        SpectrumPlot("ds", "After despreading: data collapses, jammer spreads",
                     x="frequency (× chip rate)", xlim=(-0.5, 0.5), ylim=(-60, 45), legend="tr"),
        BERPlot("ber", "Bit error rate", x="Eb/N0 (dB)", xlim=(0, 20), ylim=(1e-6, 0.5),
                legend="bl"),
    ]
    layout = [["rx", "ber"], ["ds", "ber"]]
    readouts = [
        Readout("gpdb", "Processing gain", "dB", ".1f"),
        Readout("sim", "Simulated BER", "", "sci"),
        Readout("th", "Theory BER", "", "sci", good=lambda x: x < 1e-3),
        Readout("margin", "Jamming margin", "dB", ".1f",
                help="Gp − required Eb/J0 (9.6 dB for BER 10⁻⁵) − 2 dB implementation loss"),
    ]
    challenges = [
        Challenge("Survive a jammer 20 dB stronger than your signal with a BER below 10⁻³.",
                  lambda s: s.p.js >= 20 and s.r.th < 1e-3,
                  hint="More chips per bit: processing gain has to beat J/S."),
        Challenge("Find the least processing gain that keeps BER below 10⁻³ at J/S = 15 dB and "
                  "Eb/N0 = 12 dB.",
                  lambda s: abs(s.p.js - 15) < 0.3 and abs(s.p.ebn0 - 12) < 0.3 and s.r.th < 1e-3
                  and s.exp.next_lower_fails),
        Challenge("Overwhelm it: make the jamming margin negative and watch the BER collapse.",
                  lambda s: s.r.margin < 0 and s.p.js > s.r.gpdb),
    ]

    def setup(self):
        self.seed = 1
        self.next_lower_fails = False

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def theory(ebn0_db, js_db, gp):
        return cl.qfunc(np.sqrt(2 / (1 / undb(ebn0_db) + undb(js_db) / gp)))

    def update(self, p):
        gp = int(p.gp)
        rng = np.random.default_rng(self.seed)
        nb = max(60, 150_000 // gp)
        b = rng.choice([-1.0, 1.0], nb)
        pn = rng.choice([-1.0, 1.0], (nb, gp))
        tx = (b[:, None] * pn).ravel()
        k = np.arange(len(tx))
        J = undb(p.js)
        if p.jam == "Tone":
            jam = np.sqrt(J) * np.exp(1j * (2 * np.pi * 0.071 * k + 0.3))
        else:
            w = rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx))
            Wf = np.fft.fft(w)
            f = np.fft.fftfreq(len(tx))
            Wf[np.abs(f - 0.071) > 0.01] = 0
            jam = np.fft.ifft(Wf)
            jam *= np.sqrt(J / np.mean(np.abs(jam) ** 2))
        N0 = gp / undb(p.ebn0)
        n = np.sqrt(N0 / 2) * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        rx = tx + jam + n
        des = rx * pn.ravel()
        z = np.real(des.reshape(nb, gp).sum(axis=1))
        ber = float(np.mean(np.sign(z) != b))
        th = float(self.theory(p.ebn0, p.js, gp))
        idx = GPS_CHOICES.index(p.gp)
        self.next_lower_fails = idx == 0 or self.theory(p.ebn0, p.js, int(GPS_CHOICES[idx - 1])) >= 1e-3
        m = min(len(tx), 65536)
        pr = self.plot("rx")
        pr.psd("rx", rx[:m], 1.0, 1024, color=GRAY, width=1.2, name="received", normalize=False)
        pr.psd("tx", tx[:m] + 0j, 1.0, 1024, color=NAVY, width=1.8, name="your spread signal",
               normalize=False)
        pd = self.plot("ds")
        pd.psd("ds", des[:m], 1.0, 1024, color=RED, width=1.4, name="after despreading",
               normalize=False)
        # the despread data as a narrow line: bits repeat for gp chips
        dat = np.repeat(b[:m // gp + 1], gp)[:m] + 0j
        pd.psd("dat", dat, 1.0, 1024, color=NAVY, width=1.8, name="the data",
               normalize=False)
        e = np.linspace(0, 20, 81)
        pb = self.plot("ber")
        pb.theory("nj", e, cl.ber_bpsk(e), color=GRAY, style=":", name="no jammer")
        for jj, col in [(p.js - 10, GREEN), (p.js, NAVY), (p.js + 10, RED)]:
            pb.theory(f"j{jj}", e, self.theory(e, jj, gp), color=col,
                      width=2.4 if jj == p.js else 1.4, name=f"J/S = {jj:g} dB")
        pb.sim("sim", [p.ebn0], [max(ber, 1e-7)], color=ORANGE, name=f"simulated ({nb} bits)")
        gpdb = 10 * np.log10(gp)
        self.readout(gpdb=gpdb, sim=ber, th=th, margin=gpdb - 9.6 - 2.0 - p.js)

    def story(self, p):
        gpdb, mg = self.r.get("gpdb", 0), self.r.get("margin", 0)
        s = (f"<p>Each bit is multiplied by {v(int(p.gp), 'd')} pseudo-random chips: the signal "
             f"is spread {v(gpdb, '.1f', 'dB')} wider and sits low (navy, top). The jammer puts "
             f"{v(p.js, '.0f', 'dB')} more power than you into one spot.</p>"
             "<p>The receiver multiplies by the same chips again. Your signal collapses back to "
             "the narrow data spectrum (navy, bottom), while the jammer, which never saw the code, "
             "is spread over the whole band (red): only 1/Gp of its power lands on each bit. "
             "That is <b>processing gain</b>.</p>")
        s += ((good(f"Jamming margin {mg:.1f} dB:") + " the link survives the jammer.")
              if mg >= 0 else (bad(f"Jamming margin {mg:.1f} dB:") + " the jammer wins."))
        return "<h3>Hiding in plain sight</h3>" + s + keybox(
            "Effective Eb/J0 = Gp / (J/S). Jamming margin = Gp − (Eb/J0)required − losses.")


# =============================================================================== 3. frequency hopping
NCH, NHOP = 79, 48
WIFI = {"Wi-Fi channel 1": 0, "Wi-Fi channel 6": 25, "Wi-Fi channel 11": 50}


class FrequencyHopping(Experiment):
    title = "Frequency hopping"
    blurb = "Jump around the band. A jammer can hit only the hops that land on it."
    book = "sec:ch18:fhss"
    controls = [
        Choice("intf", "Interference", ["Wi-Fi network", "Partial-band jammer"]),
        Choice("wifi", "Wi-Fi channel", list(WIFI), "Wi-Fi channel 6",
               enabled_if=lambda p: p.intf == "Wi-Fi network"),
        Toggle("afh", "Adaptive hopping (avoid bad channels)", False,
               enabled_if=lambda p: p.intf == "Wi-Fi network"),
        Heading("Partial-band jammer"),
        LogSlider("rho", "Fraction of the band jammed ρ", 0.01, 1.0, 0.3, unit="", fmt=".2f",
                  enabled_if=lambda p: p.intf != "Wi-Fi network"),
        Slider("ebj", "Eb/J0", 0, 40, 15, step=0.5, unit="dB",
               help="Bit energy over the jammer's power density averaged over the whole band"),
        IntSlider("L", "Hops per bit (diversity)", 1, 7, 1, step=2,
                  help="Fast hopping: each bit is sent on L hops and decided by majority"),
        Button("again", "New hop pattern"),
    ]
    plots = [
        ImagePlot("map", "Time–frequency map: hops (navy), interference (orange), hits (red)",
                  x="time (ms)", y="channel"),
        Plot("rho", "BER vs the jammer's band fraction ρ", x="fraction of band jammed ρ",
             y="bit error rate", logx=True, logy=True, xlim=(0.003, 1), ylim=(1e-7, 0.6),
             legend="br"),
        Plot("ber", "BER vs Eb/J0", x="Eb/J0 (dB)", y="bit error rate", logy=True, xlim=(0, 40),
             ylim=(1e-7, 0.6), legend="bl"),
    ]
    layout = [["map", "map"], ["rho", "ber"]]
    readouts = [
        Readout("hit", "Hops hit", "%", ".1f", good=lambda x: x < 5),
        Readout("ber", "Bit error rate", "", "sci", good=lambda x: x < 1e-3),
        Readout("worst", "Jammer's best ρ", "", ".3f"),
        Readout("wber", "BER against the best ρ", "", "sci"),
    ]
    challenges = [
        Challenge("Bluetooth beside Wi-Fi: get the share of hops hit below 2 %.",
                  lambda s: s.p.intf == "Wi-Fi network" and s.r.hit < 2,
                  hint="Bluetooth 1.2 added adaptive frequency hopping for exactly this."),
        Challenge("Be the smart jammer: at Eb/J0 = 20 dB and one hop per bit, choose ρ within "
                  "10 % of the worst case for the link.",
                  lambda s: s.p.intf != "Wi-Fi network" and abs(s.p.ebj - 20) < 0.3 and s.p.L == 1
                  and abs(s.p.rho / s.r.worst - 1) < 0.1),
        Challenge("Beat the smart jammer: keep even the worst-case BER below 10⁻⁴ with Eb/J0 of "
                  "20 dB or less.",
                  lambda s: s.p.ebj <= 20 and s.r.wber < 1e-4,
                  hint="Diversity: spread each bit over several hops."),
    ]

    def setup(self):
        self.seed = 3

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        if p.intf == "Wi-Fi network":
            lo = WIFI[p.wifi]
            bad_ch = set(range(lo, lo + 22))
            rho = len(bad_ch) / NCH
            excl = bad_ch if p.afh else ()
            hops = sp.hop_pattern(NHOP, NCH, rng=rng, exclude=excl)
            rho_eff = 0.0 if p.afh else rho
        else:
            nj = max(1, int(round(p.rho * NCH)))
            lo = int(np.random.default_rng(7).integers(0, NCH - nj + 1))
            bad_ch = set(range(lo, lo + nj))
            hops = sp.hop_pattern(NHOP, NCH, rng=rng)
            rho_eff = p.rho
        img = np.zeros((NCH, NHOP))
        for c in bad_ch:
            img[c, :] = 1
        hit = 0
        for t, c in enumerate(hops):
            img[c, t] = 3 if c in bad_ch else 2
            hit += c in bad_ch
        pm = self.plot("map")
        cmap = ((0, "#F4F6F7"), (0.33, "#F5CBA7"), (0.66, "#1B3A5C"), (1.0, "#C0392B"))
        if pm.theme.name == "dark":
            cmap = ((0, "#1A1F26"), (0.33, "#6E4B2A"), (0.66, "#6EC3FF"), (1.0, "#F1948A"))
        pm.image("m", img, x=(0, NHOP * 0.625), y=(2402, 2402 + NCH), cmap=cmap, levels=(0, 3))
        pm.set_xlim(0, NHOP * 0.625)
        pm.set_ylim(2402, 2402 + NCH)
        pm.set_labels(x="time (ms), 1600 hops per second", y="frequency (MHz)")
        # analytic BER curves
        r = np.logspace(np.log10(0.003), 0, 120)
        pr = self.plot("rho")
        for L, col in [(1, NAVY), (3, GREEN), (5, ORANGE), (7, PURPLE)]:
            if L == p.L or L == 1:
                pr.line(f"L{L}", r, np.maximum(sk.fh_ber(r, p.ebj, L), 1e-12), color=col,
                        width=2.4 if L == p.L else 1.2, name=f"{L} hop{'s' if L > 1 else ''}/bit")
        wr, wb = sk.fh_worst_rho(p.ebj, p.L)
        pr.scatter("worst", [wr], [max(wb, 1e-12)], color=RED, size=14, symbol="d",
                   name="jammer's best ρ")
        b_now = sk.fh_ber(max(rho_eff, 1e-6), p.ebj, p.L) if rho_eff > 0 else 0.0
        if rho_eff > 0:
            pr.vline("now", rho_eff, color=GRAY, style=":", width=1.4, label="your ρ",
                     label_pos=0.9)
        e = np.linspace(0, 40, 81)
        pb = self.plot("ber")
        pb.line("full", e, np.maximum([sk.fh_ber(1.0, x, 1) for x in e], 1e-12), color=GRAY,
                width=1.4, style="--", name="broadband jammer (ρ = 1)")
        pb.line("w1", e, np.maximum([sk.fh_worst_rho(x, 1)[1] for x in e], 1e-12), color=NAVY,
                width=1.6, name="worst-case ρ, 1 hop/bit")
        if p.L > 1:
            pb.line("wL", e, np.maximum([sk.fh_worst_rho(x, p.L, n=120)[1] for x in e], 1e-12),
                    color=GREEN, width=2.4, name=f"worst-case ρ, {p.L} hops/bit")
        pb.line("inv", e, np.minimum(0.368 / undb(e), 0.5), color=RED, style=":", width=1.2,
                name="e⁻¹ / (Eb/J0)")
        pb.vline("now", p.ebj, color=GRAY, style=":", width=1.2)
        self._st = dict(rho=rho_eff)
        self.readout(hit=100 * hit / NHOP, ber=max(float(b_now), 1e-15) if rho_eff > 0 else 0.0,
                     worst=wr, wber=wb)

    def story(self, p):
        if p.intf == "Wi-Fi network":
            s = ("<p>Bluetooth hops among 79 one-megahertz channels 1600 times a second. A Wi-Fi "
                 "network next door occupies about 22 of them (orange band): blind hopping lands "
                 f"there on roughly a quarter of the hops. Here {v(self.r.get('hit', 0), '.0f', '%')} "
                 f"of hops are hit (red).</p>")
            s += ("<p>" + good("Adaptive hopping") + " has marked those channels as bad and "
                  "simply never visits them.</p>" if p.afh else
                  "<p>Switch on adaptive hopping (Bluetooth 1.2, 2003): the device measures which "
                  "channels lose packets and removes them from the hop set.</p>")
        else:
            s = (f"<p>A jammer with a fixed power can spread it thinly over the whole band, or "
                 f"concentrate it on a fraction ρ = {v(p.rho, '.2f')} of the channels. Concentrated, "
                 f"it hits fewer hops but wipes out each one. Against non-coherent FSK the "
                 f"jammer's best choice is ρ = {v(self.r.get('worst', 0), '.3f')} (red diamond), "
                 f"and then the BER falls only as e⁻¹/(Eb/J0): every 10 dB of extra power buys "
                 f"only a factor of 10.</p>")
            s += (f"<p>With {v(p.L, 'd')} hops per bit, the jammer must hit most of a bit's hops "
                  f"to cause an error: the green curve falls steeply again. That is why military "
                  f"hoppers use fast hopping and coding across hops.</p>" if p.L > 1 else
                  "<p>Raise the hops per bit to fight back.</p>")
        return "<h3>A moving target</h3>" + s + keybox(
            "FH processing gain is statistical: interference costs the hops it hits. Diversity and "
            "coding across hops turn that into reliability.")


# =============================================================================== 4. RAKE
_RAKE_MC = {}
DELAYS = np.array([0, 3, 7, 12])
POW_DB = np.array([0.0, -2.0, -4.0, -6.0])


def mrc_rayleigh(gb_db, powers):
    """BPSK BER with MRC over independent Rayleigh paths with distinct mean powers
    (normalised to sum 1), average Eb/N0 gb_db (array)."""
    pw = np.asarray(powers, float) / np.sum(powers)
    g = undb(gb_db)[:, None] * pw[None, :]
    out = np.zeros(len(g))
    for k in range(len(pw)):
        pik = np.ones(len(g))
        for i in range(len(pw)):
            if i != k:
                pik = pik * g[:, k] / (g[:, k] - g[:, i])
        out += pik * 0.5 * (1 - np.sqrt(g[:, k] / (1 + g[:, k])))
    return np.clip(out, 1e-12, 0.5)


class Rake(Experiment):
    title = "The RAKE receiver"
    blurb = "Multipath echoes are separate copies of your signal. Collect them all."
    book = "sec:ch18:rake"
    controls = [
        IntSlider("paths", "Resolvable paths", 1, 4, 3,
                  help="Echoes at 0, 3, 7 and 12 chips with 0, −2, −4, −6 dB mean power"),
        IntSlider("fing", "RAKE fingers", 1, 4, 1),
        Choice("comb", "Combining", ["Maximal ratio", "Equal gain", "Selection"]),
        Slider("ebn0", "Average Eb/N0", 0, 25, 10, step=0.5, unit="dB"),
        Button("again", "New fading channel"),
    ]
    plots = [
        Plot("srch", "Searcher: correlation vs delay (this fade)", x="delay (chips)",
             y="|correlation|", xlim=(-1, 15), legend="tr"),
        ConstellationPlot("z", "Decision values: strongest finger (gray) vs RAKE (navy)", lim=3.0),
        BERPlot("ber", "BER over Rayleigh multipath", x="average Eb/N0 (dB)", xlim=(0, 25),
                ylim=(1e-6, 0.5), legend="bl"),
    ]
    layout = [["srch", "z"], ["ber", "ber"]]
    readouts = [
        Readout("energy", "Energy collected", "%", ".0f"),
        Readout("ber", "BER (your RAKE)", "", "sci", good=lambda x: x < 1e-3),
        Readout("div", "Diversity order", "", "int"),
        Readout("gain", "Gain over 1 finger at 10⁻³", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Collect at least 95 % of the multipath energy.",
                  lambda s: s.r.energy >= 95),
        Challenge("Reach BER 10⁻³ at an average Eb/N0 of 12 dB or less over 3 or more paths.",
                  lambda s: s.p.paths >= 3 and s.p.ebn0 <= 12 and s.r.ber <= 1e-3),
        Challenge("Compare combiners: with 4 paths and 4 fingers, find one that loses more than "
                  "1 dB against maximal ratio.",
                  lambda s: s.p.paths == 4 and s.p.fing == 4 and s.exp.loss_vs_mrc > 1),
    ]

    def setup(self):
        self.seed = 2
        self.loss_vs_mrc = 0.0

    def on_again(self, p):
        self.seed += 1

    @staticmethod
    def mc_ber(eb, powers, nf, comb_, n=20000, seed=5):
        """Symbol-level Monte Carlo of BPSK over Rayleigh paths with nf fingers on the strongest
        mean paths, for a vector of Eb/N0 values (memoised: it is deterministic)."""
        key = (tuple(np.round(np.atleast_1d(eb), 4)), tuple(np.round(powers, 6)), nf, comb_, n,
               seed)
        if key not in _RAKE_MC:
            _RAKE_MC[key] = Rake._mc(eb, powers, nf, comb_, n, seed)
        return _RAKE_MC[key]

    @staticmethod
    def _mc(eb, powers, nf, comb_, n, seed):
        r = np.random.default_rng(seed)
        pw = np.asarray(powers, float) / np.sum(powers)
        L = len(pw)
        h = (r.standard_normal((L, n)) + 1j * r.standard_normal((L, n))) * np.sqrt(pw[:, None] / 2)
        b = r.choice([-1.0, 1.0], n)
        w = r.standard_normal((L, n)) + 1j * r.standard_normal((L, n))
        out = []
        for e in np.atleast_1d(eb):
            y = h * b + np.sqrt(1 / (2 * undb(e))) * w
            yf, hf = y[:nf], h[:nf]
            if comb_ == "Maximal ratio":
                z = np.sum(np.conj(hf) * yf, axis=0).real
            elif comb_ == "Equal gain":
                z = np.sum(np.exp(-1j * np.angle(hf)) * yf, axis=0).real
            else:
                k = np.argmax(np.abs(hf), axis=0)
                z = (np.conj(hf[k, np.arange(n)]) * yf[k, np.arange(n)]).real
            out.append(np.mean(np.sign(z) != b))
        return np.array(out)

    def update(self, p):
        nf = min(p.fing, p.paths)
        pw = undb(POW_DB[:p.paths])
        # chip-level demonstration on one fading realisation
        rng = np.random.default_rng(self.seed)
        Lc, nsym = 64, 600
        code = rng.choice([-1.0, 1.0], Lc)
        g = (rng.standard_normal(p.paths) + 1j * rng.standard_normal(p.paths)) * np.sqrt(
            pw / pw.sum() / 2)
        b = rng.choice([-1.0, 1.0], nsym)
        tx = sp.spread(b, code)
        rx = np.zeros(len(tx) + 16, complex)
        for d, gg in zip(DELAYS[:p.paths], g):
            rx[d:d + len(tx)] += gg * tx
        sig = np.sqrt(Lc / (2 * undb(p.ebn0)))
        rx += sig * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
        lags = np.arange(-1, 15)
        pil = sp.spread(b[1:41], code)
        srch = [np.abs(np.vdot(pil, rx[Lc + l:Lc + l + 40 * Lc])) / (40 * Lc) if l >= 0 else
                np.abs(np.vdot(pil, rx[Lc + l:Lc + l + 40 * Lc])) / (40 * Lc) for l in lags]
        strongest = np.argsort(-np.abs(g))[:nf]          # fingers on the strongest paths now
        fd = list(DELAYS[strongest])
        ps = self.plot("srch")
        ps.stems("s", lags, srch, color=GRAY, size=6, name="searcher output")
        ps.scatter("f", fd, [srch[list(lags).index(d)] for d in fd], color=RED, size=14,
                   symbol="d", name="fingers")
        ps.set_ylim(0, max(srch) * 1.3 + 1e-3)
        zc, fingers = sp.rake_combine(rx, code, nsym, fd,
                                      g[strongest] if p.comb == "Maximal ratio" else None)
        if p.comb == "Equal gain":
            zc = np.sum(np.exp(-1j * np.angle(g[strongest]))[:, None] * fingers, axis=0)
        elif p.comb == "Selection":
            zc = np.conj(g[strongest[0]]) * fingers[0]
        z1 = np.conj(g[strongest[0]]) * fingers[0]
        sc = 1 / max(np.mean(np.abs(zc.real)), 1e-9)
        pz = self.plot("z")
        pz.points("one", z1 * sc, color=GRAY, size=4, alpha=0.5)
        pz.points("rake", zc * sc, color=NAVY, size=4, alpha=0.6)
        pz.ideal("ref", [-1, 1])
        # BER curves
        e = np.linspace(0, 25, 26)
        pb = self.plot("ber")
        for L, col in [(1, GRAY), (2, BLUE), (3, GREEN), (4, ORANGE)]:
            pb.theory(f"t{L}", e, mrc_rayleigh(e, undb(POW_DB[:L])) if L > 1 else
                      mrc_rayleigh(e, [1.0]), color=col, width=1.2, style=":",
                      name=f"MRC, {L} path{'s' if L > 1 else ''} (theory)")
        sim = self.mc_ber(e, pw, nf, p.comb)
        pb.sim("sim", e[sim > 0], sim[sim > 0], color=RED,
               name=f"{p.comb}, {nf} of {p.paths} paths (sim.)")
        pb.vline("now", p.ebn0, color=GRAY, style=":", width=1.2)
        ber_now = float(np.interp(p.ebn0, e, sim))
        one = mrc_rayleigh(e, [1.0])

        def at3(curve):
            ok = np.nonzero(curve <= 1e-3)[0]
            if not len(ok) or ok[0] == 0:
                return None
            i = ok[0]
            return float(np.interp(-3, np.log10([curve[i], curve[i - 1]]), [e[i], e[i - 1]]))

        a1, aS = at3(one), at3(np.maximum(sim, 1e-9))
        gain = (a1 - aS) if (a1 is not None and aS is not None) else None
        if p.paths == 4 and nf == 4 and p.comb != "Maximal ratio":
            mrc = self.mc_ber(e, pw, nf, "Maximal ratio")
            am = at3(np.maximum(mrc, 1e-9))
            self.loss_vs_mrc = (aS - am) if (aS is not None and am is not None) else 0.0
        else:
            self.loss_vs_mrc = 0.0
        self.readout(energy=100 * pw[np.argsort(-pw)[:nf]].sum() / pw.sum(), ber=ber_now,
                     div=nf, gain=gain if gain is not None else "—")

    def story(self, p):
        nf = min(p.fing, p.paths)
        s = (f"<p>The signal reaches you along {v(p.paths, 'd')} paths, a few chips apart. "
             f"Because the spreading code decorrelates after one chip, the receiver can separate "
             f"them: the searcher (top left) finds a peak at each delay. Each <b>finger</b> "
             f"despreads one path; the RAKE adds them up, like the teeth of a garden rake.</p>"
             f"<p>The paths fade independently, so it is unlikely they are all weak at once: with "
             f"{v(nf, 'd')} finger{'s' if nf > 1 else ''} the error rate falls with diversity "
             f"order {v(nf, 'd')} (the BER curves get steeper). Top right: the gray cloud of a "
             f"single finger smears across zero when that path fades; the navy RAKE output "
             f"stays clear.</p>")
        if p.comb != "Maximal ratio":
            s += ("<p>Maximal-ratio combining weights each finger by its own path gain, the "
                  "matched filter for the whole channel. Equal-gain and selection combining are "
                  "simpler and lose a little.</p>")
        return "<h3>Multipath as a gift</h3>" + s + keybox(
            "Spread spectrum turns multipath into diversity: the RAKE (Price and Green, 1958) "
            "collects every resolvable echo.")


# =============================================================================== 5. near-far
DETECTORS = ["Matched filter", "Decorrelator", "MMSE", "SIC"]
DET_KEY = {"Matched filter": "conventional", "Decorrelator": "decorrelator", "MMSE": "mmse",
           "SIC": "sic"}
EXC = np.arange(-10, 31, 2.5)


@lru_cache(maxsize=1)
def gold31():
    return sp.bipolar(sp.gold_codes(5)) / np.sqrt(31)


def stats_weak(y, S, A, method, sigma2):
    """Soft statistic for user 0 (the weak one) under each detector."""
    z = S.T @ y
    R = S.T @ S
    if method == "conventional":
        return z[0]
    if method == "decorrelator":
        return np.linalg.solve(R, z)[0]
    if method == "mmse":
        return np.linalg.solve(R + sigma2 * np.diag(1 / A ** 2), z)[0]
    resid = y.copy()
    for k in np.argsort(-A):
        bk = np.sign(S[:, k] @ resid)
        if k == 0:
            return S[:, 0] @ resid
        resid = resid - np.outer(S[:, k] * A[k], bk)
    return S[:, 0] @ resid


class NearFar(Experiment):
    title = "CDMA: near–far and multiuser detection"
    blurb = "One loud user drowns a quiet one. Power control or smarter detectors fix it."
    book = "sec:ch18:mud"
    heavy = False
    controls = [
        IntSlider("K", "Users", 2, 12, 6),
        Slider("exc", "Power excess of the other users", -10, 30, 5, step=0.5, unit="dB"),
        Slider("ebn0", "Weak user's Eb/N0", 2, 14, 8, step=0.5, unit="dB"),
        Choice("det", "Detector", DETECTORS),
        Toggle("pc", "Perfect power control", False,
               help="Every user is received at the same power: the excess is forced to 0 dB"),
    ]
    plots = [
        Plot("ber", "Weak user's BER vs the others' power excess (Gold-31 codes, random code phases)",
             x="power excess of the other users (dB)", y="bit error rate", logy=True,
             xlim=(-10, 30), ylim=(1e-5, 0.6), legend="bl", legend_cols=2),
        BarPlot("pow", "Received powers", x="user", y="power (dB)"),
        Plot("hist", "Weak user's output (data sign removed; below 0 = error)",
             x="decision statistic (normalised)", y="count", xlim=(-3.5, 3.5), legend=None),
    ]
    layout = [["ber", "ber"], ["pow", "hist"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("ber", "Weak user's BER", "", "sci", good=lambda x: x < 1e-2),
        Readout("single", "Alone in the cell", "", "sci"),
        Readout("loss", "Near–far penalty", "×", ".1f",
                help="BER with the other users / BER alone"),
        Readout("rho", "Worst code correlation", "", ".2f"),
    ]
    challenges = [
        Challenge("See the near–far problem: with the matched filter, push the weak user's BER "
                  "above 10 %.",
                  lambda s: s.p.det == "Matched filter" and not s.p.pc and s.r.ber > 0.1),
        Challenge("Keep 6 or more users with a 20 dB power excess, and the weak user's BER "
                  "within a factor of 3 of being alone.",
                  lambda s: s.p.K >= 6 and s.p.exc >= 20 and not s.p.pc and s.r.loss < 3,
                  hint="A detector that knows all the codes can null the interferers."),
        Challenge("Find where the decorrelator's noise enhancement makes it worse than the "
                  "matched filter.",
                  lambda s: s.exp.decor_worse,
                  hint="Weak interferers: low or negative excess, many users."),
    ]

    def setup(self):
        self.curves = {}
        self.decor_worse = False
        self.seed = 4

    def sim(self, K, exc_db, ebn0, method, nb, seed):
        r = np.random.default_rng(seed)
        G = gold31()
        idx = r.choice(len(G), K, replace=False)
        S = np.array([np.roll(G[i], int(r.integers(0, 31))) for i in idx]).T   # chip-aligned, random code phases
        A = np.r_[1.0, np.full(K - 1, undb(exc_db / 2))]
        sigma = np.sqrt(1 / (2 * undb(ebn0)))
        bb = r.choice([-1.0, 1.0], (K, nb))
        y = S @ (A[:, None] * bb) + sigma * r.standard_normal((S.shape[0], nb))
        z = stats_weak(y, S, A, DET_KEY[method], sigma ** 2)
        return float(np.mean(np.sign(z) != bb[0])), z * bb[0], S

    def update(self, p):
        exc = 0.0 if p.pc else p.exc
        nb = 4000 if self.quick else 20000
        ber, zz, S = self.sim(p.K, exc, p.ebn0, p.det, nb, self.seed)
        single = float(cl.ber_bpsk(p.ebn0))
        bm, _, _ = self.sim(p.K, exc, p.ebn0, "Matched filter", nb, self.seed)
        bd, _, _ = self.sim(p.K, exc, p.ebn0, "Decorrelator", nb, self.seed)
        self.decor_worse = bd > 1.5 * bm and bm > 0
        R = S.T @ S
        rho = float(np.max(np.abs(R - np.eye(p.K))))
        pb = self.plot("ber")
        pb.hline("single", single, color=GRAY, style=":", width=1.4, label="alone in the cell",
                 label_pos=0.6)
        pb.vline("nowl", exc, color=GRAY, style=":", width=1.2)
        pb.scatter("now", [exc], [max(ber, 1e-5)], color=RED, size=15, symbol="d",
                   name="you")
        key = (p.K, p.ebn0)
        cols = {"Matched filter": RED, "Decorrelator": NAVY, "MMSE": GREEN, "SIC": ORANGE}
        for d in DETECTORS:
            c = self.curves.get((key, d))
            if c is not None:
                pb.line("c" + d, EXC[:len(c)], np.where(np.asarray(c) > 0, c, np.nan), color=cols[d],
                        width=2.4 if d == p.det else 1.2, name=d)
        pw = self.plot("pow")
        A = np.r_[0.0, np.full(p.K - 1, exc)]
        pw.bars("p", np.arange(1, p.K + 1), A, base=min(-12, A.min() - 2), width=0.7,
                colors=[RED] + [GRAY] * (p.K - 1))
        pw.set_ylim(min(-12, A.min() - 2), max(32, A.max() + 3))
        pw.set_xticks([(1, "weak")] + [(k, str(k)) for k in range(2, p.K + 1)])
        ph = self.plot("hist")
        zn = zz / max(np.median(np.abs(zz)), 1e-9)
        h, e = np.histogram(np.clip(zn, -3.4, 3.4), bins=80, range=(-3.5, 3.5))
        ph.line("h", e, h, step=True, color=NAVY, width=1.4, fill=0, fill_alpha=0.25,
                name=None)
        ph.band("err", -3.5, 0, color=RED, alpha=0.08)
        ph.text("et", -3.3, None, "errors", color=RED, size=9, anchor=(0, 0))
        self.readout(ber=ber, single=single, loss=ber / single, rho=rho)

    def background(self, p):
        key = (p.K, p.ebn0)
        nb = 2000 if self.quick else 40000
        for d in DETECTORS:
            if (key, d) in self.curves:
                continue
            out = []
            for i, x in enumerate(EXC):
                b, _, _ = self.sim(p.K, x, p.ebn0, d, nb, self.seed)
                out.append(b)
                yield (key, d, list(out))

    def progress(self, p, item):
        key, d, c = item
        self.curves[key, d] = np.array(c)
        cols = {"Matched filter": RED, "Decorrelator": NAVY, "MMSE": GREEN, "SIC": ORANGE}
        self.plot("ber").line("c" + d, EXC[:len(c)], np.where(np.asarray(c) > 0, c, np.nan), color=cols[d],
                              width=2.4 if d == p.det else 1.2, name=d)

    def story(self, p):
        exc = 0.0 if p.pc else p.exc
        s = (f"<p>{v(p.K, 'd')} users share the channel, separated only by their Gold codes, "
             f"whose cross-correlation is small but not zero (up to {v(self.r.get('rho', 0), '.2f')}). "
             f"The other users arrive {v(exc, '.1f', 'dB')} stronger than the weak one: a phone at "
             f"the cell edge versus phones next to the tower.</p>")
        if p.det == "Matched filter" and not p.pc:
            s += ("<p>The matched filter sees each interferer leak through with its "
                  "correlation times its amplitude. Loud enough, and the leak exceeds the wanted "
                  "signal: the <b>near–far problem</b>. IS-95 and UMTS solved it with fast power "
                  "control, 800 to 1500 updates per second.</p>")
        elif p.pc:
            s += ("<p>" + good("Power control") + " makes every user arrive at the same power, so "
                  "each one only sees the small residual interference of the others.</p>")
        else:
            s += {"Decorrelator": "<p>The <b>decorrelator</b> inverts the code correlation matrix: "
                                  "it nulls the interferers completely, whatever their power, at "
                                  "the price of some noise enhancement.</p>",
                  "MMSE": "<p>The <b>MMSE</b> detector balances interference nulling against "
                          "noise enhancement: never worse than either.</p>",
                  "SIC": "<p><b>Successive cancellation</b> decodes the strongest user first, "
                         "subtracts it, and repeats. Here the loud users are easy to decode, so "
                         "their removal is nearly perfect.</p>"}[p.det]
        return "<h3>The loud and the quiet</h3>" + s + keybox(
            "CDMA is interference-limited: control the powers, or detect the users jointly.")


# =============================================================================== 6. GPS acquisition
FS_G = 2.046e6
SATS = [dict(prn=7, cn0=44, doppler=1750.0, code_phase=301.4),
        dict(prn=12, cn0=42, doppler=-2200.0, code_phase=700.0),
        dict(prn=21, cn0=38, doppler=3150.0, code_phase=88.0),
        dict(prn=3, cn0=46, doppler=-450.0, code_phase=990.0)]
PRN_CHOICES = ["PRN 7 (44 dB-Hz)", "PRN 12 (42 dB-Hz)", "PRN 21 (38 dB-Hz)", "PRN 3 (46 dB-Hz)",
               "PRN 30 (not in view)"]


@lru_cache(maxsize=8)
def gps_signal(shift_db, ms):
    sats = [dict(s, cn0=s["cn0"] + shift_db, bit_offset_ms=5) for s in SATS]
    x, _ = sp.gps_baseband(sats, FS_G, ms * 1e-3 + 0.002, rng=12)
    return x


class Acquisition(Experiment):
    title = "GPS acquisition"
    blurb = "Find a signal 20 dB below the noise: search every code phase and Doppler."
    book = "sec:ch18:acq"
    controls = [
        Choice("prn", "Satellite to search", PRN_CHOICES, style="menu"),
        Slider("shift", "Signal strength change (all satellites)", -15, 5, 0, step=1, unit="dB",
               help="Indoors, under trees or in a car, GPS signals lose 5 to 30 dB"),
        IntSlider("tc", "Coherent integration", 1, 10, 1, unit="ms",
                  help="Longer coherent sums gain SNR but need finer Doppler bins (500 Hz / T)"),
        IntSlider("kn", "Non-coherent sums", 1, 20, 4,
                  help="Sum the power of K coherent blocks (total time is limited to 40 ms)"),
    ]
    plots = [
        ImagePlot("grid", "Search grid: correlation power over code phase and Doppler",
                  x="code phase (chips)", y="Doppler (kHz)"),
        Plot("slice", "Slice through the peak's Doppler bin", x="code phase (chips)",
             y="power / noise mean", xlim=(0, 1023), legend=None),
        Plot("pd", "Detection probability (Pfa = 10⁻⁶ per cell)", x="C/N0 (dB-Hz)",
             y="probability of detection", xlim=(20, 50), ylim=(0, 1.05), legend="br"),
    ]
    layout = [["grid", "grid"], ["slice", "pd"]]
    row_stretch = [3, 2]
    readouts = [
        Readout("metric", "Peak / mean", "", ".1f"),
        Readout("det", "Detected?", "", None),
        Readout("dop", "Doppler error", "Hz", ".0f"),
        Readout("code", "Code phase error", "chips", ".2f"),
    ]
    challenges = [
        Challenge("Weaken all signals by 6 dB or more and still detect PRN 21 (32 dB-Hz).",
                  lambda s: s.p.prn.startswith("PRN 21") and s.p.shift <= -6
                  and s.r.det.startswith("yes"),
                  hint="Integrate longer: coherently first, then non-coherently."),
        Challenge("Fool the receiver: find settings where it 'detects' PRN 30, a satellite that "
                  "is not even in view.",
                  lambda s: s.p.prn.startswith("PRN 30") and s.r.det.startswith("yes"),
                  hint="Long integration also integrates the cross-correlation with strong "
                       "satellites (PRN 3 is 46 dB-Hz)."),
        Challenge("Use the full 40 ms with a coherent time of 5 ms or more and get a "
                  "peak/mean above 60 on PRN 7.",
                  lambda s: s.p.prn.startswith("PRN 7") and s.p.tc >= 5 and s.exp.total >= 40
                  and s.r.metric > 60),
    ]
    heavy = True

    def setup(self):
        self.res = None
        self.total = 4

    def config(self, p):
        T = p.tc
        K = max(1, min(p.kn, 40 // T))
        step = 500.0 / T
        dop = np.arange(-5000, 5000 + step / 2, step)
        prn = int(p.prn.split()[1])
        return T, K, dop, prn

    def update(self, p):
        T, K, dop, prn = self.config(p)
        self.total = T * K
        self.res = dict(key=(p.prn, p.shift, T, K), rows=0, grid=np.zeros((len(dop), 2046)))
        pg = self.plot("grid")
        pg.set_xlim(0, 1023)
        pg.set_ylim(-5.25, 5.25)
        pg.image("g", np.zeros((len(dop), 341)), x=(0, 1023), y=(dop[0] / 1e3 - step_half(dop),
                 dop[-1] / 1e3 + step_half(dop)), cmap="heat", levels=(0, 1))
        c = np.arange(20, 50.1, 0.5)
        pp = self.plot("pd")
        for (tc, kk, col, lab) in [(1e-3, 1, GRAY, "1 ms"), (1e-3 * T, K, RED, f"yours: {T} ms × {K}"),
                                   (10e-3, 4, GREEN, "10 ms × 4")]:
            pp.line(f"p{lab}", c, sp.acq_pd(c, tc, kk, pfa=1e-6, loss_db=1.5), color=col,
                    width=2.4 if col == RED else 1.4, name=lab)
        s = next((x for x in SATS if x["prn"] == prn), None)
        if s is not None:
            pp.vline("cn0", s["cn0"] + p.shift, color=NAVY, style=":", width=1.4,
                     label=f"PRN {prn}", label_pos=0.1)
        self.readout(metric="…", det="searching…", dop="—", code="—")

    def background(self, p):
        T, K, dop, prn = self.config(p)
        x = gps_signal(p.shift, max(T * K, 1))
        chunk = max(1, len(dop) // 12)
        for i in range(0, len(dop), chunk):
            r = sp.acquire(x, FS_G, prn, dop[i:i + chunk], T, K)
            yield (i, r["grid"])

    def progress(self, p, item):
        i, g = item
        T, K, dop, prn = self.config(p)
        res = self.res
        if res is None or res["key"] != (p.prn, p.shift, T, K):
            return
        res["grid"][i:i + len(g)] = g
        res["rows"] = max(res["rows"], i + len(g))
        G = res["grid"]
        done = res["rows"] >= len(dop)
        filled = G[:res["rows"]]
        mean = filled.mean() if filled.size else 1
        pooled = G[:, :2046 // 6 * 6].reshape(len(dop), -1, 6).max(axis=2) / mean
        pg = self.plot("grid")
        top = max(10.0, float(pooled.max()))
        pg.image("g", pooled, x=(0, 1023), y=(dop[0] / 1e3 - step_half(dop),
                 dop[-1] / 1e3 + step_half(dop)), cmap="heat", levels=(0, top), colorbar=True,
                 cbar_label="power / mean")
        ii, kk = np.unravel_index(np.argmax(filled), filled.shape)
        cp = (-kk * sp.CA_RATE / FS_G) % 1023
        code_axis = (np.arange(2046) * sp.CA_RATE / FS_G)
        ps = self.plot("slice")
        ps.line("s", code_axis, G[ii] / mean, color=NAVY, width=1.0)
        from scipy.stats import gamma
        thr = gamma.isf(1e-3 / G.size, K) / K
        ps.hline("thr", thr, color=RED, style="--", width=1.3, label=f"threshold {thr:.1f}",
                 label_pos=0.02)
        ps.set_ylim(0, max(thr * 1.5, float(G[ii].max() / mean) * 1.1))
        pg.scatter("pk", [kk * sp.CA_RATE / FS_G], [dop[ii] / 1e3], color=GREEN, size=26,
                   outline=GREEN)
        metric = float(G.max() / mean)
        s = next((x for x in SATS if x["prn"] == prn), None)
        if done:
            det = "yes" if metric > thr else "no"
            if s is not None:
                derr = dop[ii] - s["doppler"]
                cerr = ((cp - s["code_phase"] + 511.5) % 1023) - 511.5
            self.readout(metric=metric, det=det + (" ✓" if (s is not None) == (det == "yes") else " ✗"),
                         dop=derr if s is not None else "—",
                         code=cerr if s is not None else "—")
        else:
            self.readout(metric=metric, det=f"searching… {100 * res['rows'] / len(dop):.0f} %")

    def story(self, p):
        T, K, dop, prn = self.config(p)
        s = ("<p>A GPS signal arrives at about −130 dBm, some 20 dB <i>below</i> the noise in its "
             "2 MHz band. The receiver cannot see it; it can only correlate. It does not know "
             f"the code phase (1023 possibilities) or the Doppler shift (±5 kHz), so it tests "
             f"every combination: {v(len(dop), 'd')} Doppler bins × 2046 code phases, each one "
             f"an FFT correlation.</p>"
             f"<p>Each cell integrates {v(T, 'd')} ms coherently and sums {v(K, 'd')} blocks "
             f"non-coherently ({v(T * K, 'd')} ms in all). The coherent gain is 10·log₁₀(C/N0·T): "
             f"longer is better, until navigation-bit flips every 20 ms and the finer Doppler "
             f"grid (500 Hz / T) make it expensive. That is why indoor receivers lean on "
             f"<b>assisted GPS</b>: the network tells them roughly where to look.</p>")
        if prn == 30:
            s += ("<p>PRN 30 is not in view, so any detection is a <b>false alarm</b>: noise, or the "
                  "small cross-correlation of the PRN 30 replica with a strong satellite's code, "
                  "which grows with integration time just like a real signal.</p>")
        return "<h3>Searching for a needle</h3>" + s + keybox(
            "Acquisition = a 2-D search over code phase and Doppler; sensitivity grows with "
            "integration time.")


def step_half(dop):
    return (dop[1] - dop[0]) / 2e3 if len(dop) > 1 else 0.25


# =============================================================================== 7. tracking
class Tracking(Experiment):
    title = "DLL and PLL tracking"
    blurb = "Once found, keep the code and the carrier locked, millisecond after millisecond."
    book = "sec:ch18:tracking"
    animate = True
    autoplay = True
    fps = 20
    controls = [
        Heading("Signal"),
        Slider("cn0", "C/N0", 20, 50, 42, step=0.5, unit="dB-Hz"),
        Slider("acc", "Line-of-sight acceleration", 0, 6, 0, step=0.1, unit="g"),
        Heading("Loops"),
        LogSlider("dll", "DLL bandwidth", 0.2, 10, 2, unit="Hz", fmt=".2g"),
        LogSlider("pll", "PLL bandwidth", 2, 40, 15, unit="Hz", fmt=".3g"),
        Slider("d", "Early–late spacing", 0.1, 1.0, 1.0, step=0.05, unit="chip"),
        Toggle("fll", "FLL assist (10 Hz)", True),
        Slider("ferr", "Doppler error at hand-over", 0, 250, 120, step=5, unit="Hz",
               help="The acquisition's Doppler bin is only this accurate"),
        Button("reacq", "Hand over from acquisition again", primary=True),
    ]
    plots = [
        Plot("code", "Code tracking error (DLL)", x="time (s)", y="error (chips)",
             ylim=(-0.6, 0.6), legend="tr"),
        Plot("ph", "Carrier phase error (Costas PLL)", x="time (s)", y="error (degrees)",
             ylim=(-100, 100), legend="tr"),
        ConstellationPlot("iq", "Prompt correlator, last 300 ms", lim=1.0),
        Plot("bits", "In-phase prompt: the 50 b/s navigation bits appear", x="time (s)",
             y="I_P (normalised)", ylim=(-1.6, 1.6), legend=None),
    ]
    layout = [["code", "iq"], ["ph", "iq"], ["bits", "bits"]]
    col_stretch = [2, 1]
    readouts = [
        Readout("crms", "Code noise", "m rms", ".2f", good=lambda x: x < 3),
        Readout("prms", "Phase error", "° rms", ".1f", good=lambda x: x < 15),
        Readout("lock", "Phase lock", "", None),
        Readout("ferr", "Frequency error", "Hz", ".1f"),
    ]
    challenges = [
        Challenge("Stay in phase lock at 26 dB-Hz or less (a weak indoor signal).",
                  lambda s: s.p.cn0 <= 26 and s.exp.locked and s.exp.hist_len > 400,
                  hint="Lock first at a strong level, then lower C/N0; narrow the PLL and switch "
                       "the FLL off once locked (its own noise is large on weak signals)."),
        Challenge("Track a 3 g manoeuvre (or harder) with less than 15° rms phase error.",
                  lambda s: s.p.acc >= 3 and s.exp.locked and s.r.prms < 15 and s.exp.hist_len > 400,
                  hint="Dynamics need a wide loop; noise wants a narrow one."),
        Challenge("Survey-grade code: less than 1 m rms code noise at 45 dB-Hz or less.",
                  lambda s: s.p.cn0 <= 45 and s.r.crms < 1 and s.exp.hist_len > 400,
                  hint="Narrow correlator spacing and a narrow DLL."),
    ]

    def setup(self):
        self.loop = None
        self.hist = None
        self.locked = False
        self.hist_len = 0
        self.reset_loop(self.p)

    def reset_loop(self, p):
        self.loop = sk.TrackingLoop(cn0=p.cn0, dll_bn=p.dll, pll_bn=p.pll,
                                    fll_bn=10.0 if p.fll else 0.0, spacing=p.d, code_err0=0.3,
                                    freq_err0=p.ferr, accel_g=p.acc, seed=int(p.ferr * 7 + 1))
        self.hist = {k: np.empty(0) for k in ("t", "code_err", "phase_err", "f_err", "IP", "QP")}

    def on_reacq(self, p):
        self.reset_loop(p)

    def configure(self, p):
        L = self.loop
        L.cn0, L.dll_bn, L.pll_bn = p.cn0, p.dll, p.pll
        L.fll_bn = 10.0 if p.fll else 0.0
        L.spacing, L.accel_g = p.d, p.acc

    def step(self, p, n_ms):
        self.configure(p)
        out = self.loop.run(n_ms)
        for k in self.hist:
            self.hist[k] = np.r_[self.hist[k], out[k]][-1500:]

    def update(self, p):
        if self.loop is None:
            self.reset_loop(p)
        if len(self.hist["t"]) == 0:
            self.step(p, 300)
        self.draw(p)

    def tick(self, p):
        self.step(p, 50)
        self.draw(p)

    def draw(self, p):
        h = self.hist
        t = h["t"]
        self.hist_len = len(t)
        # statistics on the last 400 ms
        k = slice(-400, None)
        ce = h["code_err"][k]
        pe = (h["phase_err"][k] + np.pi / 2) % np.pi - np.pi / 2       # Costas: modulo 180°
        crms = float(np.std(ce)) * 293.05
        prms = float(np.degrees(np.sqrt(np.mean(pe ** 2))))
        # the simulation knows the truth: call it locked when the phase error stays small
        # (a real receiver uses the noisier indicator mean(I² − Q²) / mean(I² + Q²))
        self.locked = prms < 30 and abs(float(np.mean(ce))) < 0.5
        pc = self.plot("code")
        pc.line("ce", t, h["code_err"], color=NAVY, width=1.4, name="true code error")
        pc.hline("z", 0, color=GRAY, style="-", width=0.8)
        pc.set_xlim(t[0], t[-1] + 1e-3)
        lim = max(0.6, min(2.0, float(np.max(np.abs(h["code_err"])) * 1.2)))
        pc.set_ylim(-lim, lim)
        pp = self.plot("ph")
        pw = np.degrees((h["phase_err"] + np.pi / 2) % np.pi - np.pi / 2)
        pp.line("pe", t, pw, color=RED, width=1.2, name="phase error (mod 180°)")
        pp.hband("ok", -15, 15, color=GREEN, alpha=0.08)
        pp.set_xlim(t[0], t[-1] + 1e-3)
        A = np.sqrt(2 * undb(p.cn0) * 1e-3)
        n = min(300, len(t))
        pq = self.plot("iq")
        pq.points("iq", (h["IP"][-n:] + 1j * h["QP"][-n:]) / (A + 2), color=NAVY, size=4,
                  alpha=0.55)
        pq.ideal("ref", [-A / (A + 2), A / (A + 2)])
        pb = self.plot("bits")
        pb.line("ip", t, h["IP"] / (A + 2), color=NAVY, width=1.0)
        pb.line("qp", t, h["QP"] / (A + 2), color=GRAY, width=0.8, alpha=0.6)
        pb.set_xlim(t[0], t[-1] + 1e-3)
        ferr = float(np.mean(h["f_err"][-100:]))
        self.readout(crms=crms, prms=prms, lock="✓ locked" if self.locked else "✗ not locked",
                     ferr=ferr)

    def story(self, p):
        s = ("<p>The receiver now runs two feedback loops on 1 ms correlations. The <b>delay-lock "
             "loop</b> compares an early and a late replica of the code and nudges the timing "
             "toward the peak; its error, times 293 m per chip, is your pseudorange noise. The "
             "<b>Costas loop</b> keeps the local carrier in phase, ignoring the 180° flips of the "
             "50 b/s navigation data, which then appear as clean ±1 steps on I (bottom).</p>")
        s += (f"<p>Loop bandwidth is the classic trade-off. A {v(p.pll, '.3g', 'Hz')} PLL lets in "
              f"noise in proportion to its bandwidth, but follows dynamics only if it is wide "
              f"enough: {v(p.acc, '.1f', 'g')} of acceleration ramps the Doppler by "
              f"{v(p.acc * sk.ACCEL_HZ_PER_G, '.0f', 'Hz/s')}. ")
        s += (good("Locked.") if str(self.r.get("lock", "")).startswith("✓") else
              bad("Not locked: ") + "press Hand over to retry, or widen the loop / add FLL assist.")
        s += "</p>"
        return "<h3>Holding on</h3>" + s + keybox(
            "Narrow loops for weak signals, wide loops for dynamics. Carrier aiding lets the DLL "
            "be narrow even when the receiver moves.")


# =============================================================================== 8. position fix
USER = sp.lla_to_ecef(45.4215, -75.6972, 70.0)
GEOMS = {
    "Good: spread over the sky": ([30, 0, 120, 240, 72, 288, 180, 300], [85, 15, 20, 25, 22, 18, 55, 50]),
    "Poor: clustered": ([20, 35, 50, 40, 28, 45, 33, 55], [30, 45, 60, 20, 70, 38, 52, 25]),
    "Urban canyon (high only)": ([10, 100, 190, 280, 60, 230, 150, 330], [62, 68, 65, 70, 80, 75, 72, 66]),
}


def best_subset(P, vis, n):
    """The n visible satellites with the lowest PDOP (exhaustive over small sets, as a receiver
    with few channels would choose)."""
    from itertools import combinations
    vis = list(vis)
    if len(vis) <= n:
        return np.array(vis, int)
    best, bp = None, np.inf
    for c in combinations(vis, n):
        try:
            d = sp.dop(P[list(c)], USER)["PDOP"]
        except np.linalg.LinAlgError:
            continue
        if d < bp:
            best, bp = c, d
    return np.array(best if best is not None else vis[:n], int)


class PositionFix(Experiment):
    title = "Position fix and DOP"
    blurb = "Four pseudoranges, four unknowns. Geometry decides how good the answer is."
    book = "sec:ch18:errors"
    controls = [
        Choice("geo", "Sky", list(GEOMS) + ["Live constellation"], style="menu"),
        Slider("hour", "Time of day", 0, 12, 3, step=0.1, unit="h",
               enabled_if=lambda p: p.geo == "Live constellation"),
        Slider("mask", "Elevation mask", 0, 40, 10, step=1, unit="°",
               enabled_if=lambda p: p.geo == "Live constellation"),
        IntSlider("nsat", "Satellites used (at most)", 4, 8, 6),
        Slider("sig", "Pseudorange error σ (UERE)", 0.3, 10, 3, step=0.1, unit="m"),
        Slider("bias", "Receiver clock error", -1000, 1000, 300, step=10, unit="µs",
               help="A cheap quartz clock can be off by milliseconds; the fix solves for it"),
    ]
    plots = [
        Plot("sky", "Sky plot (centre = overhead, ring = horizon)", legend=None, grid=False,
             aspect=True, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15)),
        Plot("cloud", "400 fixes: horizontal error", x="east error (m)", y="north error (m)",
             aspect=True, legend="tr"),
        Plot("conv", "Gauss–Newton from the centre of the Earth", x="iteration",
             y="position error (m)", logy=True, xlim=(-0.3, 8.3), ylim=(1e-3, 1e8), legend=None),
        BarPlot("dop", "Dilution of precision", y="DOP"),
    ]
    layout = [["sky", "cloud"], ["conv", "dop"]]
    readouts = [
        Readout("pdop", "PDOP", "", ".2f", good=lambda x: x < 2.5),
        Readout("hrms", "Horizontal error", "m rms", ".1f"),
        Readout("vrms", "Vertical error", "m rms", ".1f"),
        Readout("clk", "Clock error solved", "µs", ".1f"),
    ]
    challenges = [
        Challenge("With only 4 satellites, get the PDOP below 2.5.",
                  lambda s: s.p.nsat == 4 and s.r.pdop < 2.5,
                  hint="Spread them out: one overhead and three around the horizon."),
        Challenge("Urban canyon: see the vertical error grow above 3 × the horizontal error.",
                  lambda s: s.p.geo.startswith("Urban") and s.r.vrms > 3 * s.r.hrms),
        Challenge("Live sky: raise the mask until fewer than 5 satellites are left, but keep "
                  "PDOP below 6.",
                  lambda s: s.p.geo == "Live constellation" and s.exp.nused == 4
                  and s.exp.nvis == 4 and s.r.pdop < 6),
    ]

    def setup(self):
        self.nused = 0
        self.nvis = 0

    def sats(self, p):
        if p.geo == "Live constellation":
            P = sp.gps_constellation_ecef(np.array([p.hour * 3600.0]))[:, :, 0]
            az, el = sp.azel(P.T, USER)
            vis = np.where(el > p.mask)[0]
            sel = best_subset(P, vis, p.nsat)
            return P[sel], az[sel], el[sel], az[vis], el[vis], len(vis)
        azs, els = GEOMS[p.geo]
        az, el = np.array(azs[:p.nsat], float), np.array(els[:p.nsat], float)
        return sk.sats_from_azel(USER, az, el), az, el, az, el, p.nsat

    def update(self, p):
        S, az, el, azv, elv, nvis = self.sats(p)
        self.nvis = nvis
        self.nused = len(S)
        psky = self.plot("sky")
        psky.pi.hideAxis("left")
        psky.pi.hideAxis("bottom")
        th = np.linspace(0, 2 * np.pi, 120)
        for i, e in enumerate((0, 30, 60)):
            r = 1 - e / 90
            psky.line(f"ring{i}", r * np.sin(th), r * np.cos(th), color=GRAY, width=1.0,
                      alpha=0.6)
            if e:
                psky.text(f"rl{i}", 0.02, r, f"{e}°", color=GRAY, size=8, anchor=(0, 1))
        for i, (lab, a) in enumerate((("N", 0), ("E", 90), ("S", 180), ("W", 270))):
            psky.text(f"c{i}", 1.09 * np.sin(np.radians(a)), 1.09 * np.cos(np.radians(a)), lab,
                      color=GRAY, size=9, bold=True, anchor=(0.5, 0.5))
        if p.geo == "Live constellation":
            rm = 1 - p.mask / 90
            psky.line("mask", rm * np.sin(th), rm * np.cos(th), color=ORANGE, width=1.6,
                      style="--")
            rv = 1 - elv / 90
            psky.scatter("vis", rv * np.sin(np.radians(azv)), rv * np.cos(np.radians(azv)),
                         color=GRAY, size=10)
        r = 1 - el / 90
        psky.scatter("used", r * np.sin(np.radians(az)), r * np.cos(np.radians(az)), color=NAVY,
                     size=15)
        psky.set_xlim(-1.15, 1.15)
        psky.set_ylim(-1.15, 1.15)
        if len(S) < 4:
            self.readout(pdop="—", hrms="—", vrms="—", clk="—")
            self.plot("cloud").set_title("Fewer than 4 satellites: no fix")
            return
        d = sp.dop(S, USER)
        enu, dclk = sk.fix_cloud(S, USER, p.sig, 400, rng=3)
        pcl = self.plot("cloud")
        pcl.set_title("400 fixes: horizontal error")
        pcl.scatter("pts", enu[:, 0], enu[:, 1], color=NAVY, size=3, alpha=0.5)
        hr = float(np.sqrt(np.mean(enu[:, 0] ** 2 + enu[:, 1] ** 2)))
        pcl.line("c2", 2 * hr * np.cos(th), 2 * hr * np.sin(th), color=RED, width=1.6,
                 name=f"2DRMS = {2 * hr:.1f} m")
        lim = max(5.0, 2.6 * hr)
        pcl.set_xlim(-lim, lim)
        pcl.set_ylim(-lim, lim)
        # Gauss-Newton convergence of one noisy fix with the clock bias
        bias_m = p.bias * 1e-6 * sp.C_LIGHT
        rho = sp.pseudoranges(S, USER, bias_m, p.sig, rng=5)
        x, hist, _ = sp.solve_position(S, rho, n_iter=8, tol=0)
        errs = [max(np.linalg.norm(h_[:3] - USER), 1e-3) for h_ in hist]
        pv = self.plot("conv")
        pv.line("e", np.arange(len(errs)), errs, color=NAVY, width=2.0)
        pv.scatter("ep", np.arange(len(errs)), errs, color=NAVY, size=8)
        pv.hline("sig", p.sig, color=RED, style=":", width=1.2, label="σ UERE", label_pos=0.75)
        pd = self.plot("dop")
        keys = ["GDOP", "PDOP", "HDOP", "VDOP", "TDOP"]
        vals = [float(d[k]) for k in keys]
        pd.bars("b", np.arange(5), vals, width=0.6, colors=[GRAY, NAVY, GREEN, ORANGE, PURPLE])
        for i, val in enumerate(vals):
            pd.text(f"t{i}", i, min(val, 28.5), f"{val:.2f}", anchor=(0.5, 1.05), size=9, bold=True)
        pd.set_xticks([(i, k) for i, k in enumerate(keys)])
        pd.set_xlim(-0.6, 4.6)
        pd.set_ylim(0, min(30, max(vals) * 1.2 + 0.3))
        self.readout(pdop=float(d["PDOP"]), hrms=hr, vrms=float(np.sqrt(np.mean(enu[:, 2] ** 2))),
                     clk=float(x[3] / sp.C_LIGHT * 1e6))

    def story(self, p):
        pd = self.r.get("pdop", 0)
        s = ("<p>Each satellite says when it sent its signal; the receiver notes when it arrived. "
             "The difference times c is a <b>pseudorange</b>: range plus the receiver's clock "
             f"error, here {v(p.bias, '.0f', 'µs')} (about {v(abs(p.bias) * 0.2998, '.0f', 'km')}!). "
             "Four unknowns (x, y, z, clock) need four satellites. The solver starts at the "
             "centre of the Earth and converges in a handful of iterations (bottom left), "
             "recovering the clock error exactly.</p>")
        if isinstance(pd, float):
            s += (f"<p>Geometry multiplies the {v(p.sig, '.1f', 'm')} range error by the "
                  f"<b>dilution of precision</b>: PDOP = {v(pd, '.2f')}. Satellites spread over "
                  "the sky give small DOP; clustered ones, or only high ones as in a street "
                  "canyon, make the solution wobble, especially vertically because all "
                  "satellites are above you.</p>")
        else:
            s += "<p>" + bad("Fewer than four satellites: no fix.") + "</p>"
        return "<h3>Where am I, and what time is it?</h3>" + s + keybox(
            "Position error ≈ DOP × UERE. GPS is a clock-comparison system: every receiver is "
            "also a precise clock.")


# =============================================================================== the lab
LAB = st.Lab(25, "Spread Spectrum, CDMA and GPS", chapter=18,
             chapter_title="Spread Spectrum, CDMA and Satellite Navigation",
             experiments=[PNCodes, DSSSJammer, FrequencyHopping, Rake, NearFar, Acquisition,
                          Tracking, PositionFix])

if __name__ == "__main__":
    st.run(LAB)
