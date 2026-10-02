"""Lab 32 · Toward 6G: OTFS and AFDM, Embedded Pilots, Near Field, ISAC, RIS and Sub-THz   (Chapter 25)

Run it:      python labs/lab32_6g_waveforms.py
Self-test:   python labs/lab32_6g_waveforms.py --selftest

The deeper companion to Lab 12, built on commlib.sixg and commlib.frontier, the modules behind
Chapter 25's figures. Race OFDM, OTFS and AFDM through fractional Doppler and look inside their
effective channel matrices; estimate a whole delay–Doppler channel from one pilot; focus a large
array on a point instead of a direction; turn data symbols into a radar and see which
constellations blind it; decide between a reflecting surface and a relay; and close a 300 GHz link
through the absorption lines of oxygen and water.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache

import numpy as np

import commlib as cl
from commlib import sixg
from commlib import frontier as fr
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, BERPlot, BarPlot, ImagePlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

C0 = sixg.C0
QPSK = cl.get_constellation("qpsk")


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(np.asarray(x, float), 1e-300))


def cgauss(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def waterfall(plot, key, items, total_name, total_color=NAVY, label_size=8.5):
    """A decibel 'bank statement' (as in Lab 18): floating bars from the running balance."""
    run = 0.0
    names = []
    for i, (nm, val) in enumerate(items):
        lo, hi = sorted([run, run + val])
        plot.bars(f"{key}{i}", [i], [hi], base=lo, width=0.62, color=GREEN if val >= 0 else RED)
        plot.text(f"{key}t{i}", i, hi, f"{val:+.1f}", anchor=(0.5, 1.05), size=label_size, bold=True)
        if i < len(items) - 1:
            plot.line(f"{key}c{i}", [i + 0.31, i + 0.69], [run + val] * 2, color=GRAY, width=1.0)
        run += val
        names.append(nm)
    n = len(items)
    lo, hi = sorted([0.0, run])
    plot.bars(f"{key}T", [n], [hi], base=lo, width=0.62, color=total_color)
    plot.text(f"{key}tT", n, hi, f"{run:.1f}", anchor=(0.5, 1.05), size=label_size + 0.5, bold=True,
              color=total_color)
    plot.set_xticks([(i, s) for i, s in enumerate(names + [total_name])])
    plot.set_xlim(-0.6, n + 0.6)
    return run


# =============================================================================== 1. waveforms
G16 = 16
MN = G16 * G16
A_OTFS = sixg.otfs_matrix(G16, G16)
LMAX = 3


@lru_cache(maxsize=32)
def afdm_matrix_for(alpha):
    return sixg.afdm_matrix(MN, sixg.afdm_c1(MN, alpha, guard=1), c2=1 / (MN ** 2 * np.pi))


SCHEMES = {"ofdm1": ("OFDM, one-tap", GRAY), "ofdmL": ("OFDM, ICI-aware", ORANGE),
           "otfs": ("OTFS, LMMSE", NAVY), "afdm": ("AFDM, LMMSE", GREEN)}


def waveform_trial(nu_sc, paths, snr_db, rng, frac=True):
    """One 16 × 16 frame through a random doubly dispersive channel for all four receivers.
    nu_sc: maximum Doppler in OFDM subcarrier spacings. Returns bit errors per scheme."""
    n0 = 10 ** (-snr_db / 10)
    numax = nu_sc / G16
    l, nu, h = sixg.random_dd_paths(paths, LMAX, numax, rng, fractional=frac, scale=1 / MN)
    b = cl.random_bits(2 * MN, rng)
    x = QPSK.modulate(b)
    out = {}
    for key, ici in (("ofdm1", False), ("ofdmL", True)):
        out[key] = int(np.sum(QPSK.demodulate(fr.ofdm_dd_frame(G16, G16, LMAX + 1, l, nu, h, x, n0,
                                                               rng, ici)) != b))
    Ht = sixg.dd_channel_matrix(l, nu, h, MN)
    w = np.sqrt(n0) * cgauss(rng, MN)
    alpha = int(np.ceil(numax * MN))
    for key, A in (("otfs", A_OTFS), ("afdm", afdm_matrix_for(alpha).conj().T)):
        He = A.conj().T @ Ht @ A
        out[key] = int(np.sum(QPSK.demodulate(sixg.lmmse(He, He @ x + A.conj().T @ w, n0)) != b))
    return out, len(b)


class WaveformRace(Experiment):
    title = "OFDM, OTFS and AFDM under Doppler"
    blurb = "Three ways to modulate a fast channel, and the effective channel matrix each one sees."
    book = "sec:ch25:waveforms"
    controls = [
        Slider("nu", "Maximum Doppler", 0.0, 2.0, 0.3, step=0.05, unit="× Δf",
               help="In OFDM subcarrier spacings of the 16-subcarrier symbol"),
        IntSlider("paths", "Number of paths", 1, 6, 4),
        Toggle("frac", "Fractional Doppler", True),
        Heading("Look inside"),
        Choice("view", "Effective channel", ["OFDM", "AFDM", "OTFS"], "AFDM"),
        Button("again", "New channel"),
    ]
    plots = [
        ImagePlot("G", "Effective channel |G|", x="input index", y="output index"),
        BERPlot("ber", "Bit error rate (QPSK, 16 × 16, LMMSE where noted)", x="SNR (dB)",
                xlim=(0, 30), ylim=(1e-5, 0.5), legend="bl"),
    ]
    layout = [["G", "ber"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("ici", "OFDM leakage off the diagonal", "dB", ".1f", good=lambda x: x < -20),
        Readout("c1", "AFDM chirp rate c₁", "", ".4f"),
        Readout("cond", "AFDM full-diversity condition", "", None),
        Readout("b20", "BER at 20 dB: OTFS / AFDM", "", None),
    ]
    challenges = [
        Challenge("Keep OFDM's leakage off the diagonal below −20 dB with a Doppler of at least "
                  "0.1 Δf.",
                  lambda s: s.r.ici < -20 and s.p.nu >= 0.1 - 1e-9,
                  hint="Inter-carrier interference ≈ (πν_max T)²/3: it rises 6 dB per doubling."),
        Challenge("Overload the chirps: push the Doppler until AFDM's full-diversity condition fails.",
                  lambda s: s.r.cond == "violated"),
        Challenge("Watch OTFS and AFDM tie: with ν ≥ 0.3 Δf and 4 paths, their 20 dB BERs within a "
                  "factor of 2 (let the sweep finish).",
                  lambda s: (s.p.nu >= 0.3 and s.p.paths >= 4 and s.exp.done and s.exp.b_otfs > 0
                             and 0.5 <= s.exp.b_afdm / s.exp.b_otfs <= 2)),
    ]
    SNRS = np.arange(0, 31, 5.0)

    def setup(self):
        self.seed = 2
        self.curves = {}
        self.key = None
        self.done = False
        self.b_otfs = self.b_afdm = 0.0

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        numax = p.nu / G16
        l, nu, h = sixg.random_dd_paths(p.paths, LMAX, numax, rng, fractional=p.frac, scale=1 / MN)
        Ht = sixg.dd_channel_matrix(l, nu, h, MN)
        alpha = int(np.ceil(numax * MN))
        if p.view == "OFDM":
            F = sixg.dft_matrix(G16)
            # the 16-subcarrier symbol of the frame's first OFDM symbol (no CP: circular here)
            Hs = sum(hp * np.exp(2j * np.pi * vp * np.arange(G16))[:, None] * np.roll(np.eye(G16), int(lp), 0)
                     for lp, vp, hp in zip(l, nu, h))
            G = F @ Hs @ F.conj().T
            title = "OFDM: F H Fᴴ (one 16-subcarrier symbol)"
        elif p.view == "AFDM":
            A = afdm_matrix_for(alpha)
            G = A @ Ht @ A.conj().T
            title = "AFDM: A H Aᴴ (256 chirps)"
        else:
            G = A_OTFS.conj().T @ Ht @ A_OTFS
            title = "OTFS: the delay–Doppler effective channel (256 bins)"
        Gm = np.abs(G)
        n = G.shape[0]
        pg = self.plot("G")
        pg.set_title(title)
        pg.image("img", Gm / max(Gm.max(), 1e-12), x=(-0.5, n - 0.5), y=(-0.5, n - 0.5),
                 cmap="heat", levels=(0, 0.6), colorbar=True, cbar_label="|G|")
        pg.set_xlim(-0.5, n - 0.5)
        pg.set_ylim(-0.5, n - 0.5)
        # OFDM leakage for the readout (always)
        F = sixg.dft_matrix(G16)
        Hs = sum(hp * np.exp(2j * np.pi * vp * np.arange(G16))[:, None] * np.roll(np.eye(G16), int(lp), 0)
                 for lp, vp, hp in zip(l, nu, h))
        Go = F @ Hs @ F.conj().T
        dpow = np.sum(np.abs(np.diag(Go)) ** 2)
        ici = db((np.sum(np.abs(Go) ** 2) - dpow) / max(dpow, 1e-30))
        c1 = sixg.afdm_c1(MN, alpha, guard=1)
        cond = (LMAX + 1) * (2 * (alpha + 1) + 1) <= MN
        key = (p.nu, p.paths, p.frac)
        if key != self.key:
            self.key, self.curves, self.done = key, {}, False
        self.c1 = c1
        self.draw_ber(p)
        self.readout(ici=max(ici, -80.0), c1=c1, cond="met" if cond else "violated")

    def draw_ber(self, p):
        pb = self.plot("ber")
        if self.curves:
            xs = sorted(self.curves)
            for k, (name, col) in SCHEMES.items():
                pts = [(x, self.curves[x][k]) for x in xs if self.curves[x][k] > 0]
                if pts:
                    pb.sim(k, [a for a, _ in pts], [b for _, b in pts], color=col, name=name)
        b20 = self.curves.get(20.0)
        if b20 is not None:
            self.b_otfs, self.b_afdm = b20["otfs"], b20["afdm"]
            txt = f"{st.sci(b20['otfs'])} / {st.sci(b20['afdm'])}" if b20["otfs"] > 0 else "0 / " + st.sci(b20["afdm"])
        else:
            txt = "…"
        self.readout(b20=txt)

    def background(self, p):
        if self.done:
            return
        key = self.key
        rng = np.random.default_rng(25)
        trials = 2 if self.quick else 14
        for snr in self.SNRS:
            errs = {k: 0 for k in SCHEMES}
            nb = 0
            for _ in range(trials):
                e, n = waveform_trial(p.nu, p.paths, snr, rng, p.frac)
                for k in errs:
                    errs[k] += e[k]
                nb += n
            yield dict(key=key, snr=float(snr), ber={k: errs[k] / nb for k in errs})
        yield dict(key=key, done=True)

    def progress(self, p, item):
        if item["key"] != self.key:
            return
        if item.get("done"):
            self.done = True
        else:
            self.curves[item["snr"]] = item["ber"]
        self.draw_ber(p)

    def story(self, p):
        r = self.r
        s = (f"<p>Four receivers, one channel: {v(p.paths, 'd')} paths with Dopplers up to "
             f"{v(p.nu, '.2f')} subcarrier spacings. The picture on the left is the "
             f"<b>effective channel</b> each waveform presents to its detector: the matrix that maps "
             f"transmitted symbols to received ones.</p>")
        if p.view == "OFDM":
            s += (f"<p>OFDM wants a diagonal (one tap per subcarrier). Doppler spreads energy to the "
                  f"neighbouring diagonals: {v(r.get('ici', -80), '.1f', 'dB')} of leakage, the "
                  f"inter-carrier interference a one-tap equaliser cannot remove.</p>")
        elif p.view == "AFDM":
            s += (f"<p>AFDM's chirps (c₁ = {v(self.c1, '.4f')}) turn each path into its own "
                  f"<b>separate diagonal</b>: delays land in different ranges, Dopplers shift within "
                  f"them. The condition (ℓ<sub>max</sub>+1)(2(α<sub>max</sub>+ξ)+1) ≤ N is "
                  + (good("met: every path is resolved.") if r.get("cond") == "met" else
                     bad("violated: the diagonals overlap.")) + "</p>")
        else:
            s += ("<p>OTFS works on the 2-D delay–Doppler grid: each path is a 2-D shift, so the "
                  "matrix is a sparse set of shifted blocks. Fractional Doppler blurs each block "
                  "along Doppler.</p>")
        s += ("<p>On the right, a background sweep: the one-tap OFDM curve floors, the ICI-aware "
              "one has no diversity, and OTFS and AFDM, detected jointly over the frame, fall "
              "steeply and almost together.</p>")
        return "<h3>Three waveforms, one channel</h3>" + s + keybox(
            "OTFS and AFDM perform almost identically; the differences lie in pilots, complexity, "
            "MIMO integration and how gracefully they bolt onto an OFDM modem.")


# =============================================================================== 2. embedded pilot
_LP = G16 // 2                                       # pilot delay row
_GUARD = np.zeros((G16, G16), bool)
_GUARD[_LP - LMAX:_LP + LMAX + 1, :] = True
_DATA = np.flatnonzero(~_GUARD.ravel(order="F"))
_PIL = _LP                                           # (delay _LP, Doppler bin 0), column-major
_MEAS = np.zeros((G16, G16), bool)
_MEAS[_LP:_LP + LMAX + 1, :] = True                  # where the pilot's echoes land (data-free)
_GVEC = np.flatnonzero(_MEAS.ravel(order="F"))
_KC = 5
_CAND = [(l_, k_) for l_ in range(LMAX + 1) for k_ in range(-_KC, _KC + 1)]
_APIL = A_OTFS[:, _PIL]


def _resp(l_, k_):
    """DD response of an on-grid unit path (delay l_, Doppler bin k_) to the pilot (matrix-vector)."""
    n = np.arange(MN)
    s = np.exp(2j * np.pi * k_ / MN * n) * np.roll(_APIL, l_)
    return A_OTFS.conj().T @ s


_RESP = np.array([_resp(l_, k_) for l_, k_ in _CAND]).T            # (MN, n_cand)


def pilot_trial(snr_db, pilot_db, frac, span, rng, want=False):
    n0 = 10 ** (-snr_db / 10)
    l, nu, h = sixg.random_dd_paths(4, LMAX, span / MN, rng, fractional=frac, scale=1 / MN)
    He = A_OTFS.conj().T @ sixg.dd_channel_matrix(l, nu, h, MN) @ A_OTFS
    x = np.zeros(MN, complex)
    b = cl.random_bits(2 * len(_DATA), rng)
    x[_DATA] = QPSK.modulate(b)
    ap = np.sqrt(10 ** (pilot_db / 10))
    x[_PIL] = ap
    y = He @ x + np.sqrt(n0) * cgauss(rng, MN)
    yg = y[_GVEC] / ap
    R = _RESP[_GVEC]
    keep = np.abs(R.conj().T @ yg) > 3 * np.sqrt(n0) / ap * np.sqrt(np.sum(np.abs(R) ** 2, 0))
    if not keep.any():
        keep[np.argmax(np.abs(R.conj().T @ yg))] = True
    g = np.linalg.lstsq(R[:, keep], yg, rcond=None)[0]
    idx = np.flatnonzero(keep)
    nn = np.arange(MN)
    Hhat_t = np.zeros((MN, MN), complex)
    for l_ in range(LMAX + 1):                       # group the taps by delay: one shift each
        dl = np.zeros(MN, complex)
        for gi, j in zip(g, idx):
            if _CAND[j][0] == l_:
                dl += gi * np.exp(2j * np.pi * _CAND[j][1] / MN * nn)
        if np.any(dl):
            Hhat_t += dl[:, None] * np.roll(np.eye(MN), l_, axis=0)
    Hhe = A_OTFS.conj().T @ Hhat_t @ A_OTFS
    errs = {}
    for key, H_ in (("perfect", He), ("pilot", Hhe)):
        yd = y - H_[:, _PIL] * x[_PIL]
        xh = sixg.lmmse(H_[:, _DATA], yd, n0)
        errs[key] = int(np.sum(QPSK.demodulate(xh) != b))
    nmse = float(np.sum(np.abs(Hhe - He) ** 2) / np.sum(np.abs(He) ** 2))
    if want:
        return errs, len(b), nmse, dict(y=y, He=He, Hhe=Hhe, l=l, nu=nu, h=h, idx=idx, g=g)
    return errs, len(b), nmse


class EmbeddedPilot(Experiment):
    title = "One pilot reveals the channel"
    blurb = "An OTFS embedded pilot and a guard band estimate every path at once."
    book = "sec:ch25:waveforms"
    controls = [
        Slider("pilot", "Pilot power above a data symbol", 0, 35, 20, step=1, unit="dB"),
        Slider("span", "Maximum Doppler", 0.5, 4.0, 2.0, step=0.1, unit="bins",
               help="In Doppler bins of the 16 × 16 grid (1/(NT))"),
        Toggle("frac", "Fractional Doppler", True),
        Slider("snr", "Data SNR", 0, 30, 15, step=1, unit="dB"),
        Button("again", "New frame"),
    ]
    plots = [
        ImagePlot("rx", "Received DD grid (guard rows between the lines)", x="Doppler bin",
                  y="delay bin"),
        ImagePlot("est", "Estimated pilot response", x="Doppler bin", y="delay bin"),
        BERPlot("ber", "Bit error rate: perfect CSI against the pilot", x="data SNR (dB)",
                xlim=(0, 30), ylim=(1e-5, 0.5), legend="bl"),
    ]
    layout = [["rx", "est"], ["ber", "ber"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("nmse", "Channel-estimate error (NMSE)", "dB", ".1f", good=lambda x: x < -20),
        Readout("taps", "Taps detected", "", "int"),
        Readout("ovh", "Guard overhead", "% of the grid", ".0f"),
        Readout("pen", "Pilot penalty at 20 dB", "×", ".1f"),
    ]
    challenges = [
        Challenge("With integer Doppler, bring the channel-estimate error below −25 dB.",
                  lambda s: not s.p.frac and s.r.nmse < -25,
                  hint="Integer Doppler fits the on-grid model exactly; then it is only pilot power."),
        Challenge("Fractional Doppler: find that more pilot power stops helping: NMSE stuck above "
                  "−20 dB with a pilot of 30 dB or more.",
                  lambda s: s.p.frac and s.p.pilot >= 30 and s.r.nmse > -20 and s.p.span >= 1.5),
        Challenge("Cheap pilot: keep the pilot-aided BER within 2× of perfect CSI at 20 dB "
                  "(sweep finished) with the pilot no more than 15 dB above a data symbol.",
                  lambda s: (s.p.pilot <= 15 and s.exp.done and s.r.pen is not None
                             and isinstance(s.r.pen, float) and s.r.pen <= 2)),
    ]
    SNRS = np.arange(0, 31, 5.0)

    def setup(self):
        self.seed = 3
        self.curves = {}
        self.key = None
        self.done = False

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        errs, nb, nmse, d = pilot_trial(p.snr, p.pilot, p.frac, p.span, rng, want=True)
        Y = np.abs(d["y"]).reshape(G16, G16, order="F")
        Ys = np.roll(Y, G16 // 2, axis=1)
        pr = self.plot("rx")
        pr.image("img", Ys / Ys.max(), x=(-G16 / 2 - 0.5, G16 / 2 - 0.5), y=(-0.5, G16 - 0.5),
                 cmap="heat", levels=(0, 0.5), colorbar=True)
        for k, yv in (("a", _LP - LMAX - 0.5), ("b", _LP + LMAX + 0.5)):
            pr.hline(k, yv, color=GREEN, style="--", width=1.6)
        pr.set_xlim(-G16 / 2 - 0.5, G16 / 2 - 0.5)
        pr.set_ylim(-0.5, G16 - 0.5)
        E = np.abs(d["Hhe"][:, _PIL]).reshape(G16, G16, order="F")
        T = np.abs(d["He"][:, _PIL])
        Es = np.roll(E, G16 // 2, axis=1)
        pe = self.plot("est")
        pe.image("img", Es / max(T.max(), 1e-12), x=(-G16 / 2 - 0.5, G16 / 2 - 0.5),
                 y=(-0.5, G16 - 0.5), cmap="heat", levels=(0, 1), colorbar=True)
        kt = d["nu"] * MN
        pe.scatter("true", (kt + G16 / 2) % G16 - G16 / 2, d["l"] + _LP, color=GREEN, size=14,
                   symbol="o", outline=GREEN)
        pe.set_xlim(-G16 / 2 - 0.5, G16 / 2 - 0.5)
        pe.set_ylim(-0.5, G16 - 0.5)
        key = (p.pilot, p.span, p.frac)
        if key != self.key:
            self.key, self.curves, self.done = key, {}, False
        self.draw_ber(p)
        self.readout(nmse=float(db(nmse)), taps=len(d["idx"]),
                     ovh=100 * (2 * LMAX + 1) / G16)

    def draw_ber(self, p):
        pb = self.plot("ber")
        xs = sorted(self.curves)
        for k, name, col in (("perfect", "perfect channel knowledge", NAVY),
                             ("pilot", "embedded-pilot estimate", RED)):
            pts = [(x, self.curves[x][k]) for x in xs if self.curves[x][k] > 0]
            if pts:
                pb.sim(k, [a for a, _ in pts], [b for _, b in pts], color=col, name=name)
        pb.vline("now", p.snr, color=GRAY, style=":")
        c = self.curves.get(20.0)
        pen = None
        if c is not None:
            pen = c["pilot"] / c["perfect"] if c["perfect"] > 0 else (1.0 if c["pilot"] == 0 else 99.0)
        self.readout(pen=pen)

    def background(self, p):
        if self.done:
            return
        key = self.key
        rng = np.random.default_rng(8)
        trials = 1 if self.quick else 10
        for snr in self.SNRS:
            e = {"perfect": 0, "pilot": 0}
            nb = 0
            for _ in range(trials):
                er, n, _ = pilot_trial(snr, p.pilot, p.frac, p.span, rng)
                for k in e:
                    e[k] += er[k]
                nb += n
            yield dict(key=key, snr=float(snr), ber={k: e[k] / nb for k in e})
        yield dict(key=key, done=True)

    def progress(self, p, item):
        if item["key"] != self.key:
            return
        if item.get("done"):
            self.done = True
        else:
            self.curves[item["snr"]] = item["ber"]
        self.draw_ber(p)

    def story(self, p):
        r = self.r
        s = (f"<p>One strong pilot sits in the middle of a band of empty <b>guard</b> rows (between "
             f"the green lines, top left). Because the DD channel is a handful of shifts, the "
             f"received guard region <i>is</i> the channel: each path paints a copy of the pilot at "
             f"its delay and Doppler. The receiver keeps the cells above a threshold, fits their "
             f"gains, rebuilds the whole channel and detects the data around it.</p>")
        if p.frac:
            s += (f"<p>With <b>fractional Doppler</b> each path smears along its Doppler row; the "
                  f"on-grid fit misses part of it, so the error ({v(r.get('nmse', 0), '.1f', 'dB')}) "
                  f"stops improving however strong the pilot: the BER curve flattens.</p>")
        else:
            s += (f"<p>With <b>integer Doppler</b> the on-grid model is exact, and the estimate "
                  f"improves with pilot power: NMSE {v(r.get('nmse', 0), '.1f', 'dB')}.</p>")
        s += (f"<p>The price is the guard: {v(r.get('ovh', 0), '.0f', '%')} of this tiny grid, about "
              f"8 % of the book's 512 × 64 train grid.</p>")
        return "<h3>The channel in one snapshot</h3>" + s + keybox(
            "Sparse in delay–Doppler means cheap to estimate: one pilot and a guard region, valid "
            "for the whole frame, whatever the speed.")


# =============================================================================== 3. near field
class NearFieldFocus(Experiment):
    title = "Beamfocusing in the near field"
    blurb = "A large array can focus on a point, not just a direction, inside its Rayleigh distance."
    book = "sec:ch25:mimo"
    controls = [
        Choice("nel", "Array elements (λ/2 spacing)", ["64", "128", "256", "512"], "256"),
        LogSlider("fc", "Carrier frequency", 3.5, 140, 28, unit="GHz", fmt=".1f"),
        Slider("rf", "Focus distance", 1, 60, 6, step=0.5, unit="m"),
        Slider("ang", "Focus angle", -60, 60, 20, step=1, unit="°"),
        Choice("mode", "Beamformer", ["Focused", "Far-field beam"], "Focused"),
    ]
    plots = [
        ImagePlot("map", "Normalised gain over the floor (dB)", x="x (m)", y="y (m)"),
        Plot("dof", "Gain along the focus direction", x="range (m)", y="normalised gain (dB)",
             logx=True, xlim=(0.5, 3000), ylim=(-25, 2), legend="bl"),
    ]
    layout = [["map", "dof"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("D", "Aperture D", "m", ".2f"),
        Readout("dR", "Rayleigh distance 2D²/λ", "m", ",.0f"),
        Readout("depth", "3 dB depth of focus", "m", None),
        Readout("far", "Far-field beam at the focus", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Separate two users on the same bearing: focus closer than 6 m with a 3 dB depth "
                  "of focus under 1 m.",
                  lambda s: s.p.rf < 6 and s.exp.depth_num < 1.0 and s.p.mode == "Focused"),
        Challenge("Push the focus out until its far edge runs off to infinity (it becomes a beam), "
                  "within 15 % of where that first happens.",
                  lambda s: (s.p.mode == "Focused" and np.isinf(s.exp.far_edge)
                             and s.p.rf <= 1.15 * s.exp.first_inf)),
        Challenge("Make an ordinary far-field beam good enough (within 1 dB of focusing) for a user "
                  "only 6 m away.",
                  lambda s: abs(s.p.rf - 6) < 0.3 and s.r.far >= -1,
                  hint="The near field ends at 2D²/λ: shrink the aperture in wavelengths, or the frequency."),
    ]

    def setup(self):
        self.depth_num = np.inf
        self.far_edge = np.inf
        self.first_inf = np.inf

    def gain_along(self, w, nel, d, fc, th, r):
        x = sixg.ula_positions(nel, d)
        px, py = r[:, None] * np.sin(th), r[:, None] * np.cos(th)
        dist = np.sqrt((px - x[None, :]) ** 2 + py ** 2)
        a = np.exp(-2j * np.pi * (dist - r[:, None]) / (C0 / fc))
        return np.abs(a @ w) ** 2 / nel ** 2

    def update(self, p):
        nel = int(p.nel)
        fc = p.fc * 1e9
        lam = C0 / fc
        d = lam / 2
        D = nel * d
        th = np.deg2rad(p.ang)
        dR = float(sixg.rayleigh_distance(D, fc))
        w_near = np.conj(sixg.nearfield_response(nel, d, fc, p.rf, th))
        w_far = np.conj(sixg.farfield_response(nel, d, fc, th))
        w = w_near if p.mode == "Focused" else w_far
        # field map, extent scaled to the aperture and focus
        ymax = max(2.2 * p.rf, 2 * D, 8.0)
        half = max(1.2 * D / 2 + 1, ymax * abs(np.sin(th)) * 1.05 + 0.5, 4.0)
        nx, ny = 72, 72
        xg = np.linspace(-half, half, nx, dtype=np.float32)
        yg = np.linspace(0.2, ymax, ny, dtype=np.float32)
        X, Y = np.meshgrid(xg, yg)
        xe = sixg.ula_positions(nel, d).astype(np.float32)
        ph = (np.float32(2 * np.pi / lam) * np.sqrt((X.ravel()[:, None] - xe[None, :]) ** 2
                                                   + Y.ravel()[:, None] ** 2))
        wc = w.astype(np.complex64)
        # sum_n w_n exp(-j ph_n) as two real matrix products (fast in float32)
        re = np.cos(ph) @ wc.real + np.sin(ph) @ wc.imag
        im = np.cos(ph) @ wc.imag - np.sin(ph) @ wc.real
        P = ((re ** 2 + im ** 2) / nel ** 2).reshape(ny, nx)
        pm = self.plot("map")
        pm.image("img", np.clip(db(P + 1e-9), -25, 0), x=(-half, half), y=(0.2, ymax), cmap="heat",
                 levels=(-25, 0), colorbar=True, cbar_label="dB")
        pm.line("arr", [-D / 2, D / 2], [0.2, 0.2], color=GREEN, width=6)
        pm.scatter("foc", [p.rf * np.sin(th)], [p.rf * np.cos(th)], color=GREEN, size=16,
                   symbol="+")
        pm.set_xlim(-half, half)
        pm.set_ylim(0.2, ymax)
        # gain along the focus direction
        r = np.logspace(np.log10(0.5), np.log10(3000), 320)
        pd = self.plot("dof")
        g_now = db(self.gain_along(w_near, nel, d, fc, th, r))
        g_far = db(self.gain_along(w_far, nel, d, fc, th, r))
        pd.line("near", r, g_now, color=NAVY, width=2.4, name=f"focused at {p.rf:g} m")
        pd.line("far", r, g_far, color=RED, width=2.0, style="--", name="far-field beam")
        pd.hline("m3", -3, color=GRAY, style=":", label="−3 dB", label_pos=0.02)
        pd.vline("dR", dR, color=PURPLE, style=":", label="Rayleigh distance", label_pos=0.12)
        near, far = fr.depth_of_focus(dR, p.rf)
        pd.band("dof", near, min(far, 3000), color=GREEN, alpha=0.10)
        # measured 3 dB region around the focus
        above = g_now >= -3
        i0 = int(np.argmin(np.abs(r - p.rf)))
        lo = i0
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(r) - 1 and above[hi + 1]:
            hi += 1
        m_near, m_far = r[lo], (np.inf if hi == len(r) - 1 else r[hi])
        self.depth_num = m_far - m_near
        self.far_edge = m_far
        # where the measured far edge first diverges (cached per array)
        self.first_inf = self.first_divergence(nel, p.fc, p.ang)
        far_at_focus = float(db(self.gain_along(w_far, nel, d, fc, th, np.array([p.rf])))[0])
        self.dR = dR
        self.an = (near, far)
        self.readout(D=D, dR=dR,
                     depth=(f"{m_near:.1f} – ∞" if np.isinf(m_far) else f"{m_near:.1f} – {m_far:.1f}"),
                     far=far_at_focus)

    @lru_cache(maxsize=64)
    def first_divergence(self, nel, fcg, ang):
        fc = fcg * 1e9
        d = C0 / fc / 2
        th = np.deg2rad(ang)
        dR = float(sixg.rayleigh_distance(nel * d, fc))
        far = np.array([3000.0])

        def diverged(rf):
            w = np.conj(sixg.nearfield_response(nel, d, fc, rf, th))
            return db(self.gain_along(w, nel, d, fc, th, far))[0] >= -3

        lo, hi = 1.0, 60.0
        if not diverged(hi):
            return np.inf
        if diverged(lo):
            return lo
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if diverged(mid) else (mid, hi)
        return hi

    def story(self, p):
        r = self.r
        near, far = self.an
        s = (f"<p>The {v(int(p.nel), 'd')}-element array is {v(r.get('D', 0), '.2f', 'm')} wide "
             f"(green bar). Its <b>Rayleigh distance</b> 2D²/λ is "
             f"{v(r.get('dR', 0), ',.0f', 'm')}: closer than that, the wavefront across the array "
             f"is noticeably curved.</p>")
        if p.mode == "Focused":
            s += (f"<p>Matching the exact spherical phases of the point at {v(p.rf, '.1f', 'm')} "
                  f"concentrates energy in a <b>spot</b>, not a wedge: the gain along the bearing "
                  f"(right) rises and falls again around the focus, measured 3 dB region "
                  f"{v(r.get('depth', '—'))}. The book's estimate d<sub>R</sub>r<sub>F</sub>/(d<sub>R</sub> ± 10r<sub>F</sub>) "
                  f"gives {v(near, '.1f')} to {v(far, '.1f') if np.isfinite(far) else v('∞')} m "
                  f"(green band).</p>")
        else:
            s += ("<p>A far-field beam matches only a direction: a wedge that never ends, reaching "
                  "full gain only near the Rayleigh distance (red dashed curve).</p>")
        s += (f"<p>A far-field codebook beam aimed at the same user delivers "
              f"{v(r.get('far', 0), '.1f', 'dB')} there: near-field codebooks must sample range as "
              f"well as angle.</p>")
        return "<h3>Focus on a point</h3>" + s + keybox(
            "Inside d_R = 2D²/λ, beams become spots: users at the same angle but different "
            "distances can be served separately.")


# =============================================================================== 4. ISAC
CONS = {"16-PSK": "16psk", "QPSK": "qpsk", "16-QAM": "16qam", "64-QAM": "64qam", "256-QAM": "256qam",
        "Gaussian": None}


class ISACTradeoff(Experiment):
    title = "ISAC: the data blinds the radar"
    blurb = "The same OFDM frame carries data and senses cars; the constellation sets the floor."
    book = "sec:ch25:isac"
    controls = [
        Heading("Waveform (28 GHz, 120 kHz, 64 symbols)"),
        Choice("bw", "Bandwidth", ["99 MHz", "198 MHz", "396 MHz"], "198 MHz"),
        Choice("con", "Data constellation", list(CONS), "QPSK", style="menu"),
        Choice("rx", "Radar receiver", ["Divide by data", "Correlate"], "Divide by data",
               help="Divide: Y/X (reciprocal filter). Correlate: Y·X* (matched filter)"),
        Heading("Scene"),
        Slider("sep", "Gap between the two cars", 0.2, 5.0, 1.0, step=0.1, unit="m"),
        Slider("snr", "SNR per resource element", -30, 10, -10, step=1, unit="dB"),
        Button("again", "New data"),
    ]
    plots = [
        ImagePlot("rd", "Range–Doppler map (dB)", x="radial speed (m/s)", y="range (m)"),
        Plot("cut", "Range profile at 15 m/s (the two cars)", x="range (m)", y="dB",
             xlim=(30, 42), ylim=(-45, 3), legend=None),
        Plot("ped", "One target, one symbol, correlation", x="range bin (M = 1024)",
             y="dB", xlim=(0, 400), ylim=(-56, 3), legend=None),
    ]
    layout = [["rd", "cut"], ["rd", "ped"]]
    col_stretch = [3, 2]
    readouts = [
        Readout("dR", "Range resolution", "m", ".2f"),
        Readout("floor", "Data floor (κ−1)/M, theory", "dB", None),
        Readout("noise", "Noise enhancement E[1/|X|²]", "dB", None),
        Readout("meas", "Measured data floor", "dB", None),
    ]
    challenges = [
        Challenge("Resolve the two cars 1 m apart or closer (a dip of at least 3 dB between them).",
                  lambda s: s.p.sep <= 1.0 + 1e-9 and s.exp.dip >= 3),
        Challenge("Carry 4 bits per symbol with no data-induced floor at all.",
                  lambda s: s.p.con == "16-PSK",
                  hint="The floor comes from amplitude fluctuations: kurtosis κ − 1."),
        Challenge("Find the data that costs the dividing receiver at least 5 dB of noise enhancement.",
                  lambda s: s.p.rx == "Divide by data" and s.exp.enh >= 5),
    ]
    fc, df, N = 28e9, 120e3, 64

    def setup(self):
        self.seed = 3
        self.dip = 0.0
        self.enh = 0.0

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rng = np.random.default_rng(self.seed)
        M = {"99 MHz": 825, "198 MHz": 1650, "396 MHz": 3300}[p.bw]
        N = self.N
        Ts = 1 / self.df * 1.0703
        lam = C0 / self.fc
        name = CONS[p.con]
        if name is None:
            X = cgauss(rng, (M, N))
            pts = None
        else:
            c = cl.get_constellation(name)
            X = c.points[rng.integers(0, c.M, (M, N))]
            pts = c.points
        a_ped = 10 ** (-25 / 20)
        targets = [(12.0, 1.4, a_ped), (35.0, 15.0, 1.0), (35.0 + p.sep, 15.0, 0.8),
                   (60.0, -22.0, 0.6), (48.0, 0.0, 1.0)]
        m = np.arange(M)[:, None]
        n = np.arange(N)[None, :]
        Rm = np.hstack([a * np.exp(-2j * np.pi * m * self.df * 2 * R / C0) for R, vel, a in targets])
        Dn = np.vstack([np.exp(2j * np.pi * n * Ts * 2 * vel / lam) for R, vel, a in targets])
        H = Rm @ Dn                                          # separable: a sum of rank-one ramps
        Y = X * H + np.sqrt(10 ** (-p.snr / 10)) * cgauss(rng, (M, N))
        Z = Y / X if p.rx == "Divide by data" else Y * X.conj()
        wr = np.hanning(M)[:, None]
        wd = np.hanning(N)[None, :]
        Zr = np.fft.ifft(Z * wr, axis=0)
        RD = np.fft.fftshift(np.fft.fft(Zr * wd, axis=1), axes=1)
        P = np.abs(RD) ** 2
        Pdb = 10 * np.log10(P / P.max() + 1e-30)
        dR = C0 / (2 * M * self.df)
        dv = lam / (2 * N * Ts)
        vax = (np.arange(N) - N / 2) * dv
        nr = int(80 / dR) + 1
        cols = np.where(np.abs(vax) <= 40)[0]
        pr = self.plot("rd")
        pr.image("img", np.clip(Pdb[:nr][:, cols], -50, 0),
                 x=(vax[cols[0]] - dv / 2, vax[cols[-1]] + dv / 2), y=(-dR / 2, (nr - 0.5) * dR),
                 cmap="heat", levels=(-50, 0), colorbar=True, cbar_label="dB")
        pr.set_xlim(-40, 40)
        pr.set_ylim(0, 80)
        # zero-padded range profiles through two speed columns
        pad = 8
        Zd = np.fft.fft(Z * wd, axis=1)

        def profile(vel):
            col = int(np.round(vel / dv)) % N
            z = np.fft.ifft(Zd[:, col] * wr.ravel(), pad * M)
            rr = np.arange(pad * M) * dR / pad
            return rr, np.abs(z) ** 2

        rr, pc = profile(15.0)
        sel = (rr >= 28) & (rr <= 44)
        ref = P.max()
        pcd = 10 * np.log10(pc / (pc[sel].max()) + 1e-30)
        pg = self.plot("cut")
        pg.line("p", rr[sel], pcd[sel], color=NAVY, width=2.0)
        pg.vline("a", 35.0, color=GREEN, style=":")
        pg.vline("b", 35.0 + p.sep, color=GREEN, style=":")
        a_, b_ = 35.0, 35.0 + p.sep
        i0, i1 = np.searchsorted(rr, [a_, b_])
        seg = pcd[i0:i1 + 1]
        pk = min(pcd[max(i0 - 3, 0):i0 + 4].max(), pcd[max(i1 - 3, 0):i1 + 4].max())
        self.dip = float(max(0.0, pk - seg.min())) if len(seg) > 2 else 0.0
        # the book's trade-off picture: one target in range bin 100, correlation, 20 frames
        Mm = 1024
        r2 = np.random.default_rng(11)
        acc = {}
        for nm, pp_ in (("ref", cl.get_constellation("16psk").points), ("cur", pts)):
            a_ = np.zeros(Mm)
            for _ in range(20):
                Xc = cgauss(r2, Mm) if pp_ is None else pp_[r2.integers(0, len(pp_), Mm)]
                prof = np.abs(np.fft.ifft(np.abs(Xc) ** 2 * np.exp(-2j * np.pi * np.arange(Mm) * 100 / Mm))) ** 2
                a_ += prof / prof.max()
            acc[nm] = 10 * np.log10(a_ / 20 + 1e-6)
        pq = self.plot("ped")
        bins = np.arange(Mm)
        pq.line("ref", bins, np.maximum(acc["ref"], -54), color=GREEN, width=1.4, name="16-PSK")
        pq.line("cur", bins, np.maximum(acc["cur"], -54), color=NAVY, width=1.2, name=p.con)
        mfl = float(10 * np.log10(np.mean(10 ** (acc["cur"][150:900] / 10))))
        self.mfloor = mfl
        if pts is None:
            kappa, enh = 2.0, np.inf
        else:
            kappa = np.mean(np.abs(pts) ** 4)
            enh = 10 * np.log10(np.mean(1 / np.abs(pts) ** 2))
        self.enh = enh
        fl = fr.isac_floor_db(pts, Mm) if pts is not None else 10 * np.log10(1.0 / Mm)
        self.readout(dR=dR, floor=("−∞" if not np.isfinite(fl) else f"{fl:.1f}"),
                     noise=("∞" if not np.isfinite(enh) else f"{max(enh, 0.0):.1f}"),
                     meas=("none" if self.mfloor < -55 else f"{self.mfloor:.1f}"))
        self.kappa = kappa

    def story(self, p):
        r = self.r
        s = (f"<p>Five targets: a pedestrian at 12 m, a parked car at 48 m, two cars at 35 m and "
             f"{v(35 + p.sep, '.1f', 'm')} moving together at 15 m/s, and an oncoming car. With "
             f"{v(p.bw)} the range resolution is {v(r.get('dR', 0), '.2f', 'm')} (about twice that "
             f"with the Hann window): {('the cars are resolved' if self.dip >= 3 else 'the cars merge')}"
             f" (top right).</p>")
        s += (f"<p>Bottom right: a <b>correlation</b> receiver (multiply by the conjugate data) "
              f"sees one target through one OFDM symbol. With {v(p.con)} the symbol amplitudes "
              f"fluctuate (kurtosis κ = {v(self.kappa, '.2f')}) and that fluctuation spreads a "
              f"floor (κ−1)/M = {v(r.get('floor', '—'))} dB across every range bin, where a weak "
              f"target would drown. Constant-modulus 16-PSK (green) has none.</p>")
        if p.rx == "Correlate":
            s += ("<p>The map uses the correlation receiver too; here it also averages over the "
                  "64 symbols, which pushes the floor down by another 18 dB.</p>")
        else:
            s += (f"<p><b>Dividing</b> by the data removes it completely, whatever the constellation, "
                  f"but it amplifies the noise on subcarriers that carried small symbols: "
                  f"E[1/|X|²] = {v(r.get('noise', '—'))} dB of noise enhancement here "
                  f"(infinite for Gaussian data).</p>")
        return "<h3>The deterministic–random trade-off</h3>" + s + keybox(
            "The best communication signal is the worst radar signal: every ISAC system sits "
            "somewhere on that trade-off, usually by dedicating some resources to sensing.")


# =============================================================================== 5. RIS vs relay
FREQS = {"3.5 GHz": 3.5e9, "28 GHz": 28e9, "140 GHz": 140e9}


class RISvsRelay(Experiment):
    title = "RIS, relay or nothing?"
    blurb = "A passive surface, a decode-and-forward relay and the direct path, in one geometry."
    book = "sec:ch25:mimo"
    controls = [
        Heading("Geometry (Tx at 0 m, Rx at 100 m)"),
        Slider("x", "Helper position along the link", 1, 99, 95, step=1, unit="m"),
        Slider("off", "Helper offset from the link", 1, 30, 5, step=1, unit="m"),
        Slider("block", "Direct-path blockage", 0, 40, 20, step=1, unit="dB"),
        Heading("Radio"),
        Choice("fc", "Carrier", list(FREQS), "28 GHz"),
        LogSlider("N", "RIS elements", 100, 100000, 2500, unit="", fmt=",.0f"),
        Choice("bits", "RIS phase control", ["1 bit", "2 bits", "continuous"], "2 bits"),
        Slider("pt", "Transmit power", 0, 40, 20, step=1, unit="dBm"),
    ]
    plots = [
        Plot("geo", "The scene", x="along the link (m)", y="offset (m)", xlim=(-5, 105),
             ylim=(-12, 34), legend="tr", grid=False),
        Plot("rate", "Spectral efficiency against RIS size", x="RIS elements N",
             y="b/s/Hz", logx=True, xlim=(100, 100000), ylim=(0, 16), legend="tl"),
        Plot("need", "Elements for the RIS path to equal an open direct path",
             x="helper position along the link (m)", y="elements", logy=True, xlim=(0, 100),
             ylim=(30, 3e5), legend="tr"),
    ]
    layout = [["geo", "rate"], ["need", "rate"]]
    col_stretch = [1, 1]
    readouts = [
        Readout("rd", "Direct", "b/s/Hz", ".2f"),
        Readout("rr", "With the RIS", "b/s/Hz", ".2f"),
        Readout("rl", "Half-duplex relay", "b/s/Hz", ".2f"),
        Readout("win", "Winner", "", None),
    ]
    challenges = [
        Challenge("Move the helper to the middle of the link (40–60 m) and still make the RIS beat "
                  "the relay.",
                  lambda s: 40 <= s.p.x <= 60 and s.r.win == "RIS",
                  hint="In the middle d₁·d₂ is largest: you need many more elements."),
        Challenge("At 28 GHz with 20 dB blockage, find the smallest RIS (within 15 %) that beats "
                  "the relay at 95 m.",
                  lambda s: (s.p.fc == "28 GHz" and abs(s.p.block - 20) < 0.5 and abs(s.p.x - 95) < 0.5
                             and s.r.win == "RIS" and s.p.N <= 1.15 * s.exp.n_beat)),
        Challenge("Find a setting where a relay beats even a 10 000-element RIS.",
                  lambda s: s.p.N >= 9999 and s.r.win == "relay",
                  hint="Low rates favour the relay: little power, a high carrier."),
    ]
    D = 100.0
    GT = 10 ** 1.5                      # 15 dBi base-station antenna towards the user and helper
    BW = 100e6

    def rates(self, p, N):
        fc = FREQS[p.fc]
        lam = C0 / fc
        d1 = np.hypot(p.x, p.off)
        d2 = np.hypot(self.D - p.x, p.off)
        noise_w = 10 ** ((-174 + 10 * np.log10(self.BW) + 7) / 10) / 1e3
        pt = 10 ** (p.pt / 10) / 1e3
        g_dir = sixg.fspl_gain(self.D, fc, gt=self.GT) * 10 ** (-p.block / 10)
        s = 1.0 if p.bits == "continuous" else float(np.sinc(2.0 ** -int(p.bits[0])))
        g_ris = sixg.ris_gain(np.asarray(N, float), fc, d1, d2, gt=self.GT) * s ** 2
        snr_dir = pt * g_dir / noise_w
        snr_ris = pt * (np.sqrt(g_dir) + np.sqrt(g_ris)) ** 2 / noise_w
        # DF relay with the same antenna gain and power, half duplex
        snr_sr = pt * sixg.fspl_gain(d1, fc, gt=self.GT, gr=self.GT) / noise_w
        snr_rd = pt * sixg.fspl_gain(d2, fc, gt=self.GT) / noise_w
        r_rel = 0.5 * min(np.log2(1 + snr_sr), np.log2(1 + snr_rd + snr_dir))
        return np.log2(1 + snr_dir), np.log2(1 + snr_ris), r_rel, d1, d2, lam

    def update(self, p):
        Ns = np.logspace(2, 5, 160)
        r_d, r_ris, r_rel, d1, d2, lam = self.rates(p, Ns)
        pr = self.plot("rate")
        pr.line("ris", Ns, r_ris, color=NAVY, width=2.4, name="direct + RIS")
        pr.hline("rel", r_rel, color=ORANGE, style="-", width=2.0, label="half-duplex DF relay",
                 label_pos=0.62)
        pr.hline("dir", r_d, color=RED, style="--", label="direct (blocked)", label_pos=0.35)
        rd, rr, rl, *_ = self.rates(p, p.N)
        rr = float(rr)
        pr.scatter("now", [p.N], [rr], color=RED, size=13, symbol="d")
        beat = np.where(r_ris > r_rel)[0]
        self.n_beat = float(Ns[beat[0]]) if len(beat) else np.inf
        # geometry
        pg = self.plot("geo")
        pg.line("dl", [0, self.D], [0, 0], color=RED, width=2.0, style="--",
                name=f"direct path (−{p.block:.0f} dB)")
        pg.line("h1", [0, p.x], [0, p.off], color=NAVY, width=1.6, name="via the helper")
        pg.line("h2", [p.x, self.D], [p.off, 0], color=NAVY, width=1.6)
        pg.scatter("tx", [0], [0], color=NAVY, size=14, symbol="t")
        pg.scatter("rx", [self.D], [0], color=GREEN, size=14, symbol="s")
        pg.scatter("h", [p.x], [p.off], color=ORANGE, size=16, symbol="d")
        side = np.sqrt(p.N) * lam / 2
        pg.text("ht", p.x, p.off + 2, f"RIS {side * 100:.0f} cm square\nor relay", color=ORANGE,
                anchor=(0.5, 1), size=8.5)
        pg.text("txt", 0, -2, "BS", anchor=(0.5, 0), size=9, bold=True)
        pg.text("rxt", self.D, -2, "user", anchor=(0.5, 0), size=9, bold=True)
        if p.block > 0:
            pg.line("wall", [50, 50], [-6, 6], color=GRAY, width=7)
        # elements needed vs position
        xr = np.linspace(1, 99, 197)
        pn = self.plot("need")
        for f_, col in ((3.5e9, BLUE), (28e9, NAVY), (140e9, ORANGE)):
            l_ = C0 / f_
            neq = 4 * np.hypot(xr, p.off) * np.hypot(self.D - xr, p.off) / (l_ * self.D)
            pn.line(f"n{f_}", xr, neq, color=col, width=2.0 if f_ == FREQS[p.fc] else 1.2,
                    name=f"{f_ / 1e9:g} GHz")
        pn.vline("x", p.x, color=GRAY, style=":")
        win = "RIS" if rr > max(rl, rd) + 1e-9 else ("relay" if rl > rd else "direct")
        self.neq = 4 * d1 * d2 / (lam * self.D)
        self.side = side
        self.readout(rd=float(rd), rr=rr, rl=float(rl), win=win)

    def story(self, p):
        r = self.r
        s = (f"<p>A {v(p.fc)} base station serves a user 100 m away through a wall that costs "
             f"{v(p.block, '.0f', 'dB')}. A helper stands at {v(p.x, '.0f', 'm')} along the link: "
             f"either an RIS of {v(p.N, ',.0f')} elements (a {v(self.side * 100, '.0f', 'cm')} "
             f"panel) or a decode-and-forward relay with the same antenna gain.</p>"
             f"<p>The RIS path follows the <b>product-distance law</b>: gain ∝ (N·A)²/(d₁d₂)². To "
             f"equal an unobstructed direct path here it needs about {v(self.neq, ',.0f')} elements "
             f"(bottom left): near either end d₁·d₂ is small; in the middle it explodes.</p>")
        s += (f"<p>Right now: direct {v(r.get('rd', 0), '.2f')}, RIS {v(r.get('rr', 0), '.2f')}, relay "
              f"{v(r.get('rl', 0), '.2f')} b/s/Hz. The relay amplifies but, half duplex, spends half "
              f"its time listening, so its rate is halved; the passive surface adds no noise and works "
              f"in full duplex. The RIS wins at high rates, the relay at low ones.</p>")
        return "<h3>Product distance and the half-duplex tax</h3>" + s + keybox(
            "An RIS needs hundreds to thousands of elements and belongs near one end of the link; "
            "3GPP's first answer was the powered, network-controlled repeater.")


# =============================================================================== 6. sub-THz budget
class SubTHzBudget(Experiment):
    title = "A sub-terahertz link budget"
    blurb = "Free space, molecular absorption and antenna size from 100 to 400 GHz."
    book = "sec:ch25:spectrum"
    controls = [
        Slider("f", "Carrier frequency", 100, 400, 140, step=1, unit="GHz"),
        LogSlider("d", "Distance", 1, 2000, 100, unit="m", fmt=",.0f"),
        Slider("pt", "Transmit power", -10, 20, 0, step=1, unit="dBm"),
        Choice("ant", "Antennas", ["30 dBi each end", "1 cm² apertures"], "30 dBi each end",
               style="menu", help="Fixed gain: the aperture shrinks as f rises. Fixed aperture: "
                                  "the gain grows as f²"),
        Choice("bw", "Bandwidth", ["10 GHz", "25 GHz", "50 GHz"], "50 GHz"),
        Slider("nf", "Receiver noise figure", 6, 15, 10, step=0.5, unit="dB"),
        Slider("rho", "Water vapour", 0, 20, 7.5, step=0.5, unit="g/m³"),
        Slider("rain", "Rain", 0, 20, 0, step=1, unit="dB/km"),
    ]
    plots = [
        Plot("gam", "Specific attenuation of clear air (approx. ITU-R P.676)", x="frequency (GHz)",
             y="dB/km", logy=True, xlim=(100, 400), ylim=(0.05, 200), legend="br"),
        BarPlot("bud", "The link budget (dB / dBm)", y="dB / dBm"),
        Plot("cap", "Shannon rate against distance", x="distance (m)", y="rate (Gb/s)", logx=True,
             logy=True, xlim=(1, 2000), ylim=(0.1, 2000), legend="tr"),
    ]
    layout = [["gam", "cap"], ["bud", "bud"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("fspl", "Free-space loss", "dB", ".1f"),
        Readout("abs", "Absorption", "dB", ".2f"),
        Readout("snr", "SNR", "dB", ".1f", good=lambda x: x >= 0),
        Readout("rate", "Shannon rate", "Gb/s", ".1f"),
    ]
    challenges = [
        Challenge("Find a frequency where a 1 km link loses more than 20 dB to the air alone.",
                  lambda s: s.p.d >= 990 and s.r.abs > 20,
                  hint="Look for the water lines at 183 and 325 GHz."),
        Challenge("The book's kiosk: 300 GHz, 10 m, 0 dBm, 30 dBi antennas, 50 GHz: reach about "
                  "12 dB of SNR (±1 dB).",
                  lambda s: (abs(s.p.f - 300) < 1 and abs(s.p.d - 10) < 0.5 and abs(s.p.pt) < 0.5
                             and s.p.ant.startswith("30") and s.p.bw == "50 GHz"
                             and abs(s.r.snr - 12) <= 1)),
        Challenge("Fixed 1 cm² apertures: carry 20 Gb/s over 200 m or more with at most 10 dBm "
                  "and 50 GHz of bandwidth.",
                  lambda s: (s.p.ant.startswith("1 cm") and s.p.d >= 199 and s.p.pt <= 10
                             and s.p.bw == "50 GHz" and s.r.rate >= 20),
                  hint="With fixed apertures, higher frequency means more gain at both ends."),
    ]

    def budget(self, p, f_ghz, d):
        lam = C0 / (f_ghz * 1e9)
        fspl = 20 * np.log10(4 * np.pi * d / lam)
        go, gw = fr.gas_atten_db_km(f_ghz, p.rho)
        gamma = np.asarray(go) + np.asarray(gw)
        absorb = gamma * d / 1e3
        rain = p.rain * d / 1e3
        if p.ant.startswith("30"):
            g = 30.0 + 0 * np.asarray(f_ghz, float)
        else:
            g = 10 * np.log10(4 * np.pi * 1e-4 * 0.7 / lam ** 2)
        bw = float(p.bw.split()[0]) * 1e9
        noise = -174 + 10 * np.log10(bw) + p.nf
        pr = p.pt + 2 * g - fspl - absorb - rain - 3.0
        snr = pr - noise
        rate = bw * np.log2(1 + 10 ** (snr / 10)) / 1e9
        return dict(fspl=fspl, absorb=absorb, rain=rain, g=g, noise=noise, pr=pr, snr=snr, rate=rate)

    def update(self, p):
        f = np.linspace(100, 400, 601)
        go, gw = fr.gas_atten_db_km(f, p.rho)
        pg = self.plot("gam")
        pg.line("tot", f, go + gw, color=NAVY, width=2.2, name="oxygen + water vapour")
        pg.line("w", f, gw, color=BLUE, width=1.2, style="--", name="water vapour")
        pg.line("o", f, go, color=ORANGE, width=1.2, style=":", name="oxygen")
        for a, b, col in ((110, 170, PURPLE), (252, 325, GREEN)):
            pg.band(f"b{a}", a, b, color=col, alpha=0.07)
        pg.text("bd", 140, None, "D-band", color=PURPLE, anchor=(0.5, 0), size=8.5)
        pg.text("b3", 288, None, "802.15.3d", color=GREEN, anchor=(0.5, 0), size=8.5)
        g_now = sum(fr.gas_atten_db_km(p.f, p.rho))
        pg.scatter("now", [p.f], [g_now], color=RED, size=13, symbol="d")
        b = self.budget(p, p.f, p.d)
        pb = self.plot("bud")
        items = [("Tx power", p.pt), ("Tx antenna", float(b["g"])), ("free space", -float(b["fspl"])),
                 ("absorption", -float(b["absorb"]))]
        if p.rain > 0:
            items.append(("rain", -float(b["rain"])))
        items += [("pointing etc.", -3.0), ("Rx antenna", float(b["g"]))]
        tot = waterfall(pb, "w", items, "received")
        pb.hline("noise", b["noise"], color=ORANGE, style="--",
                 label=f"noise in {p.bw}: {b['noise']:.1f} dBm", label_pos=0.01)
        lo = min(tot, float(b["noise"]), -float(b["fspl"]) + p.pt) - 15
        pb.set_ylim(lo, max(40.0, p.pt + float(b["g"])) + 15)
        dd = np.logspace(0, np.log10(2000), 200)
        pc = self.plot("cap")
        for f_, col in ((140, PURPLE), (300, GREEN)):
            if abs(p.f - f_) > 2:
                pc.line(f"c{f_}", dd, np.maximum(self.budget(p, f_, dd)["rate"], 1e-3), color=col,
                        width=1.2, alpha=0.7, name=f"{f_} GHz")
        pc.line("cur", dd, np.maximum(self.budget(p, p.f, dd)["rate"], 1e-3), color=NAVY, width=2.4,
                name=f"{p.f:.0f} GHz (yours)")
        pc.scatter("now", [p.d], [max(float(b["rate"]), 1e-3)], color=RED, size=13, symbol="d")
        pc.hline("100", 100, color=GRAY, style=":", label="100 Gb/s", label_pos=0.05)
        self.b = b
        self.readout(fspl=float(b["fspl"]), abs=float(b["absorb"]), snr=float(b["snr"]),
                     rate=float(b["rate"]))

    def story(self, p):
        b = self.b
        s = (f"<p>At {v(p.f, '.0f', 'GHz')} the wavelength is {v(C0 / p.f / 1e6, '.2f', 'mm')}. Over "
             f"{v(p.d, ',.0f', 'm')} free space costs {v(float(b['fspl']), '.1f', 'dB')} between "
             f"isotropic antennas, while the air itself absorbs only "
             f"{v(float(b['absorb']), '.2f', 'dB')} (Beer–Lambert: linear in dB with distance). "
             f"Molecules of oxygen and water have rotational lines at 119, 183 and 325 GHz; between "
             f"them lie windows around 140 and 300 GHz.</p>")
        if p.ant.startswith("30"):
            s += ("<p>With <b>fixed-gain</b> antennas the free-space term grows 20 dB per decade of "
                  "frequency, which is really a statement about each antenna's shrinking "
                  "aperture. Switch to fixed 1 cm² apertures and the trend reverses.</p>")
        else:
            s += (f"<p>With <b>fixed 1 cm² apertures</b> each antenna's gain is "
                  f"{v(float(b['g']), '.1f', 'dBi')} and grows as f², so the link improves with "
                  f"frequency until the absorption lines bite. The real problem is pointing a "
                  f"pencil beam and surviving a hand or head walking through it.</p>")
        s += (f"<p>Received {v(float(b['pr']), '.1f', 'dBm')} against "
              f"{v(float(b['noise']), '.1f', 'dBm')} of noise: SNR {v(float(b['snr']), '.1f', 'dB')}, "
              f"Shannon {v(float(b['rate']), '.1f', 'Gb/s')} in {v(p.bw)}.</p>")
        return "<h3>The terahertz frontier</h3>" + s + keybox(
            "Sub-THz is limited not by the air but by directivity: tiny apertures, enormous gain, "
            "beams a degree wide that must be pointed, tracked and re-found after blockage.")


# =============================================================================== the lab
LAB = st.Lab(32, "Toward 6G: Waveforms, Near Field, ISAC, RIS and Sub-THz", chapter=25,
             chapter_title="The Road to 6G",
             experiments=[WaveformRace, EmbeddedPilot, NearFieldFocus, ISACTradeoff, RISvsRelay,
                          SubTHzBudget])

if __name__ == "__main__":
    st.run(LAB)
