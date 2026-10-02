"""Lab 31 · Wi-Fi, Bluetooth LE, 802.15.4, LoRa and RFID: the Unlicensed Radios   (Chapter 22)

Run it:      python labs/lab31_wifi_ble_lora.py
Self-test:   python labs/lab31_wifi_ble_lora.py --selftest

The unlicensed bands host the most-used radios on Earth, and each is a different answer to
the same question: how do you move bits when nobody owns the spectrum? Wi-Fi answers with
wide OFDM channels and a clever MAC; Bluetooth Low Energy with a constant-envelope radio that
sleeps 99.9 % of the time; Zigbee and Thread with spreading chips; LoRa with chirps that work
20 dB below the noise; RFID by not having a transmitter at all. Nine experiments built on
commlib.iot take each one apart.
"""
import _path  # noqa: F401  (makes commlib and studio importable)

from functools import lru_cache
from math import ceil

import numpy as np
from scipy.signal import firwin

import commlib as cl
from commlib import airif as ai
from commlib import iot
import studio as st
from studio import (Experiment, Slider, LogSlider, IntSlider, Choice, Toggle, Button, Heading,
                    Plot, SpectrumPlot, EyePlot, BERPlot, ImagePlot, BarPlot, Readout, Challenge,
                    NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL)
from studio import v, keybox, good, bad

STA_COLS = ["#2E86C1", "#C0392B", "#1E7B4F", "#C0661A", "#6C3483", "#117A65", "#B7950B",
            "#1B3A5C", "#E67E22", "#8E44AD"]


# =============================================================================== shared helpers
def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def cn(r, n):
    return (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)


# =============================================================================== 1. 802.11 preamble
L64 = np.zeros(64, complex)
for _k, _v in zip(range(-26, 27), iot.wifi_ltf_freq()):
    L64[_k % 64] = _v
LTF_T = np.fft.ifft(L64) * 64 / np.sqrt(52)
USED = np.array([k % 64 for k in range(-26, 27) if k != 0])
KS = np.array([k for k in range(-26, 27) if k != 0])
CHANNELS = {"None (cable)": np.array([1.0]),
            "Indoor (3 echoes)": np.array([1, 0, 0.45 * np.exp(1.1j), 0, 0, 0.2j]),
            "Strong echo": np.array([1, 0, 0, 0, 0, 0, 0, 0.8 * np.exp(2.0j), 0, 0, 0.3])}


def wifi_sync(x, thr=0.5, avg=True):
    """Packet detection (delay-16 metric), coarse CFO (STF), timing (LTF cross-correlation),
    fine CFO (two LTF copies) and LS channel estimate. Returns a dict or None."""
    m, P = iot.delay_correlate(x, 16, 48)
    above = np.flatnonzero(m > thr)
    if len(above) == 0 or above[0] + 400 + 128 > len(x):
        return dict(m=m, k0=None)
    k0 = above[0]
    cfo1 = -np.angle(P[k0 + 40]) / (2 * np.pi * 16) * 20e6
    n = np.arange(len(x))
    xc = x * np.exp(-2j * np.pi * cfo1 / 20e6 * n)
    xcorr = np.abs(np.convolve(xc, np.conj(LTF_T[::-1]), "valid"))
    win = slice(k0 + 100, min(k0 + 400, len(xcorr)))
    t1 = win.start + int(np.argmax(xcorr[win]))
    if t1 - 64 >= 0 and xcorr[t1 - 64] > 0.6 * xcorr[t1]:
        t1 -= 64
    a, b = xc[t1:t1 + 64], xc[t1 + 64:t1 + 128]
    cfo2 = -np.angle(np.sum(a * np.conj(b))) / (2 * np.pi * 64) * 20e6
    a = a * np.exp(-2j * np.pi * cfo2 / 20e6 * np.arange(64))
    b = b * np.exp(-2j * np.pi * cfo2 / 20e6 * (np.arange(64) + 64))
    F = (np.fft.fft(a) + np.fft.fft(b)) / 2 if avg else np.fft.fft(a)
    H = F / (64 / np.sqrt(52))
    Hest = np.zeros(64, complex)
    Hest[USED] = H[USED] / L64[USED]
    return dict(k0=k0, cfo=cfo1 + cfo2, cfo1=cfo1, t1=t1, H=Hest, m=m, xcorr=xcorr)


class PreambleSync(Experiment):
    title = "802.11 preamble: find, tune, measure"
    blurb = "Detect a packet, measure its frequency offset and the channel, in 16 µs."
    book = "sec:ch22:ppdu"
    controls = [
        Heading("Channel"),
        Slider("snr", "SNR", -5, 35, 10, step=0.5, unit="dB"),
        Slider("cfo", "Carrier frequency offset", -600, 600, 80, step=5, unit="kHz",
               help="±20 ppm at 5 GHz is ±100 kHz per device; two devices can be 200 kHz apart"),
        Choice("mp", "Multipath", list(CHANNELS), "Indoor (3 echoes)", style="menu"),
        Heading("Receiver"),
        Slider("thr", "Detection threshold", 0.2, 0.95, 0.5, step=0.01,
               help="On the normalised delay-16 autocorrelation |P|²/R²"),
        Toggle("avg", "Average the two LTF symbols", True),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("pkt", "The received packet", x="time (µs)", y="|r|", xlim=(0, 42), legend=None),
        Plot("met", "Delay-16 autocorrelation: the STF plateau", x="time (µs)",
             y="|P|² / R²", xlim=(0, 42), ylim=(0, 1.25), legend=None),
        Plot("lx", "Cross-correlation with the known LTF symbol", x="time (µs)",
             y="|xcorr|", xlim=(0, 42), legend=None),
        Plot("H", "Channel estimate from the LTF", x="subcarrier", y="|H|² (dB)",
             xlim=(-27, 27), ylim=(-20, 12), legend="bl", legend_cols=2),
    ]
    layout = [["pkt", "met"], ["lx", "H"]]
    readouts = [
        Readout("det", "Packet found at", "µs", None, help="The packet truly starts at 10.0 µs"),
        Readout("cfo", "CFO estimate", "kHz", None),
        Readout("err", "CFO error", "Hz", None),
        Readout("nmse", "Channel-estimate error", "dB", None),
    ]
    challenges = [
        Challenge("Push the offset beyond ±156 kHz, where the LTF alone aliases, and still "
                  "estimate it within 1 kHz.",
                  lambda s: abs(s.p.cfo) > 160 and s.exp.err_hz is not None and abs(s.exp.err_hz) < 1000,
                  hint="The coarse STF estimate (16-sample period) is unambiguous to ±625 kHz."),
        Challenge("Detect the packet within 1 µs of its start at an SNR of 0 dB or less.",
                  lambda s: s.p.snr <= 0 and s.exp.det_us is not None and abs(s.exp.det_us - 10) <= 1,
                  hint="The threshold trades missed packets against false alarms on noise."),
        Challenge("Strong echo: get the channel-estimate error below −25 dB.",
                  lambda s: s.p.mp == "Strong echo" and s.exp.nmse_db is not None and s.exp.nmse_db < -25),
    ]

    def setup(self):
        self.seed = 3

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        r = np.random.default_rng(self.seed)
        h = CHANNELS[p.mp]
        stf, ltf = iot.wifi_legacy_preamble()
        pkt = np.concatenate([stf, ltf, cn(r, 320)])
        lead = 200
        x = np.concatenate([np.zeros(lead), np.convolve(pkt, h)[:len(pkt)], np.zeros(120)])
        x = x * np.exp(2j * np.pi * p.cfo * 1e3 / 20e6 * np.arange(len(x)))
        x = x + 10 ** (-p.snr / 20) * cn(r, len(x))
        s = wifi_sync(x, p.thr, p.avg)
        t = np.arange(len(x)) / 20.0
        pp = self.plot("pkt")
        top = 1.3 * np.abs(x).max()
        for (a_, b_, lab, c) in ((200, 360, "L-STF", GREEN), (360, 520, "L-LTF", ORANGE),
                                 (520, 840, "SIG + data", GRAY)):
            pp.band(f"b{lab}", a_ / 20, b_ / 20, color=c, alpha=0.12)
            pp.text(f"t{lab}", (a_ + b_) / 40, 0.98 * top, lab, color=c, size=9, anchor=(0.5, 0),
                    bold=True)
        pp.line("x", t, np.abs(x), color=NAVY, width=1.0)
        pp.set_ylim(0, top)
        pm = self.plot("met")
        pm.line("m", np.arange(len(s["m"])) / 20, s["m"], color=NAVY, width=1.8)
        pm.hline("thr", p.thr, color=RED, style="--", label="threshold", label_pos=0.85)
        pl = self.plot("lx")
        H = None
        self.err_hz = self.det_us = self.nmse_db = None
        if s["k0"] is not None:
            self.det_us = s["k0"] / 20
            pm.vline("det", self.det_us, color=GREEN, style=":", label="detected", label_pos=0.1)
            pl.line("xc", np.arange(len(s["xcorr"])) / 20, s["xcorr"], color=GREEN, width=1.6)
            pl.vline("t1", s["t1"] / 20, color=RED, style=":", label="LTF symbol 1", label_pos=0.85)
            self.err_hz = s["cfo"] - p.cfo * 1e3
            Htrue = np.fft.fft(h, 64)
            ph = np.angle(np.sum(s["H"][USED] * np.conj(Htrue[USED])))
            mse = np.mean(np.abs(s["H"][USED] * np.exp(-1j * ph) - Htrue[USED]) ** 2) / \
                np.mean(np.abs(Htrue[USED]) ** 2)
            self.nmse_db = float(db(mse))
            H = s["H"]
        pH = self.plot("H")
        Htrue = np.fft.fft(h, 64)
        pH.line("true", KS, db(np.abs(Htrue[KS % 64]) ** 2), color=GRAY, width=2.6,
                name="true |H|²")
        if H is not None:
            pH.scatter("est", KS, db(np.abs(H[KS % 64]) ** 2), color=ORANGE, size=7,
                       name="LTF estimate")
        f = lambda val, fmt: "—" if val is None else format(val, fmt).replace("-", "−")
        self.readout(det=f(self.det_us, ".1f"), cfo=f(None if s["k0"] is None else s["cfo"] / 1e3, ".2f"),
                     err=f(self.err_hz, ".0f"), nmse=f(self.nmse_db, ".1f"))

    def story(self, p):
        r = self.r
        s = ("<p>Every 802.11 OFDM packet opens with the <b>L-STF</b>: ten repetitions of a "
             "0.8 µs pattern. Correlating the signal with itself 16 samples later gives a "
             "plateau (top right) whatever the channel and the frequency offset: that is how a "
             "receiver spots a packet without knowing anything about it.</p>")
        if r.get("det") == "—":
            s += f"<p>{bad('No packet detected')}: the plateau never crossed the threshold.</p>"
        else:
            s += (f"<p>The phase of that correlation is −2π·Δf·0.8 µs, so the STF also measures "
                  f"the offset (coarsely, up to ±625 kHz). The <b>L-LTF</b>'s two identical 3.2 µs "
                  f"symbols refine it: estimate {v(r.get('cfo', '—'), unit='kHz')} against "
                  f"{v(p.cfo, '.0f', 'kHz')} true, error {v(r.get('err', '—'), unit='Hz')}. Dividing "
                  f"the LTF's FFT by the known sequence gives the channel on 52 subcarriers "
                  f"(bottom right), {v(r.get('nmse', '—'), unit='dB')} from the truth.</p>")
        if abs(p.cfo) > 156:
            s += ("<p>Beyond ±156 kHz the LTF's 64-sample spacing aliases; only the two-stage "
                  "estimate (STF first, LTF second) gets it right.</p>")
        return "<h3>Sixteen microseconds of preamble</h3>" + s + keybox(
            "Short repetitions for detection and coarse frequency, long ones for fine frequency "
            "and the channel: the same recipe opens LTE, NR and DVB frames.")


# =============================================================================== 2. MAC efficiency
def exchange_parts(phy, msdu=1500, n_agg=1, t_pre=40e-6, mac_overhead=38, ack_rate=24.0):
    """The pieces of one EDCA exchange (s), mirroring commlib.iot.mac_exchange."""
    p = iot.WIFI5G
    aifs = p["sifs"] + 3 * p["slot"]
    backoff = p["cwmin"] / 2 * p["slot"]
    mpdu = msdu + mac_overhead
    if n_agg > 1:
        sub = 4 + mpdu
        sub += (-sub) % 4
        nmax = (5.484e-3 - t_pre) * phy * 1e6 / 8
        n_agg = int(max(1, min(n_agg, nmax // sub)))
        nbytes = n_agg * sub
        t_ack = iot.legacy_ppdu_time(32, ack_rate)
    else:
        nbytes = mpdu
        t_ack = iot.legacy_ppdu_time(14, ack_rate)
    t_pay = nbytes * 8 / (phy * 1e6)
    parts = [("AIFS", aifs), ("backoff", backoff), ("preamble", t_pre), ("payload", t_pay),
             ("SIFS", p["sifs"]), ("ACK" if n_agg == 1 else "Block Ack", t_ack)]
    T = sum(x for _, x in parts)
    return parts, n_agg, n_agg * msdu * 8 / T / 1e6


class MACAggregation(Experiment):
    title = "MAC efficiency and A-MPDU"
    blurb = "A 600 Mb/s PHY that delivers 56 Mb/s, and the trick that fixes it."
    book = "sec:ch22:mac"
    controls = [
        LogSlider("phy", "PHY rate", 6, 2400, 300, unit="Mb/s", fmt=".0f"),
        LogSlider("agg", "A-MPDU length", 1, 256, 1, unit="MPDUs", fmt=".0f",
                  help="MPDUs aggregated into one PPDU, acknowledged by one Block Ack"),
        IntSlider("msdu", "Packet size", 64, 1500, 1500, step=4, unit="bytes"),
        Choice("pre", "Preamble", ["20 µs (11a/g)", "40 µs (11n/ac/ax)"], "40 µs (11n/ac/ax)",
               style="menu"),
    ]
    plots = [
        Plot("tl", "One channel access, to scale", x="time (µs)", y="", legend=None),
        Plot("thr", "MAC throughput vs PHY rate", x="PHY rate (Mb/s)", y="throughput (Mb/s)",
             logx=True, logy=True, xlim=(6, 2400), ylim=(1, 3000), legend="tl"),
        Plot("eff", "Efficiency vs aggregation at your PHY rate", x="A-MPDU length",
             y="MAC efficiency (%)", logx=True, xlim=(1, 256), ylim=(0, 105), legend=None),
    ]
    layout = [["tl", "tl"], ["thr", "eff"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("thr", "Throughput", "Mb/s", ".1f"),
        Readout("eff", "Efficiency", "%", ".1f", good=lambda x: x >= 80),
        Readout("pay", "Payload airtime", "µs", ".1f"),
        Readout("oh", "Fixed overhead", "µs", ".0f"),
    ]
    challenges = [
        Challenge("Reproduce Chapter 22: one 1500-byte MPDU per access at 600 Mb/s gives about "
                  "56 Mb/s.",
                  lambda s: abs(s.p.phy - 600) < 15 and round(s.p.agg) == 1 and s.p.msdu == 1500
                  and abs(s.r.thr - 55.8) < 1.5),
        Challenge("At 1200 Mb/s or more, reach 80 % efficiency with the shortest A-MPDU you can "
                  "(1500-byte packets).",
                  lambda s: (s.p.phy >= 1190 and s.r.eff >= 80 and s.p.msdu == 1500
                             and s.exp.eff_less < 80)),
        Challenge("Small packets hurt: with 100-byte packets, see even an A-MPDU of 64 stay "
                  "under 50 % at 600 Mb/s.",
                  lambda s: s.p.msdu <= 100 and round(s.p.agg) >= 64 and s.p.phy >= 585 and s.r.eff < 50),
    ]

    def tpre(self, p):
        return 20e-6 if p.pre.startswith("20") else 40e-6

    def update(self, p):
        n = int(round(p.agg))
        parts, n_eff, thr = exchange_parts(p.phy, p.msdu, n, self.tpre(p))
        self.eff_less = exchange_parts(p.phy, p.msdu, max(n - 1, 1), self.tpre(p))[2] / p.phy * 100 \
            if n > 1 else -1
        pt = self.plot("tl")
        x0 = 0.0
        cols = [GRAY, ORANGE, PURPLE, GREEN, GRAY, BLUE]
        for i, (nm, d) in enumerate(parts):
            d_us = d * 1e6
            pt.bars(f"b{i}", [x0 + d_us / 2], [1.0], width=d_us, color=cols[i])
            up = i % 2 == 0
            pt.text(f"t{i}", x0 + d_us / 2, 1.05 if up else -0.05, f"{nm}\n{d_us:.1f} µs",
                    anchor=(0.5, 1) if up else (0.5, 0), size=8.5, color=cols[i],
                    bold=nm == "payload")
            x0 += d_us
        pt.set_xlim(-0.01 * x0, 1.01 * x0)
        pt.set_ylim(-0.6, 1.8)
        pt.set_yticks([])
        pt.set_title(f"One channel access: {n_eff} MPDU{'s' if n_eff > 1 else ''} of {p.msdu} bytes "
                     f"in {x0:.0f} µs")
        rates = np.logspace(np.log10(6), np.log10(2400), 80)
        pr = self.plot("thr")
        pr.line("phy", rates, rates, color=GRAY, width=1.0, style=":", name="PHY rate")
        for na, col in ((1, RED), (8, ORANGE), (64, NAVY), (256, GREEN)):
            th = [exchange_parts(r_, p.msdu, na, self.tpre(p))[2] for r_ in rates]
            pr.line(f"a{na}", rates, th, color=col, width=2.4 if na == n else 1.3,
                    name="one MPDU" if na == 1 else f"A-MPDU {na}")
        pr.scatter("now", [p.phy], [thr], color=PURPLE, size=15, symbol="d")
        ns = np.unique(np.round(np.logspace(0, np.log10(256), 60)).astype(int))
        pe = self.plot("eff")
        pe.line("e", ns, [exchange_parts(p.phy, p.msdu, k, self.tpre(p))[2] / p.phy * 100 for k in ns],
                color=NAVY, width=2.2)
        pe.scatter("now", [n], [thr / p.phy * 100], color=PURPLE, size=15, symbol="d")
        pe.hline("80", 80, color=GREEN, style=":", label="80 %", label_pos=0.03)
        pay = dict(parts)["payload"]
        self.readout(thr=thr, eff=100 * thr / p.phy, pay=pay * 1e6,
                     oh=(sum(d for _, d in parts) - pay) * 1e6)

    def story(self, p):
        r = self.r
        s = (f"<p>Every channel access pays a fixed toll: AIFS, an average backoff, the "
             f"preamble, a SIFS and the acknowledgement, {v(r.get('oh', 0), '.0f', 'µs')} in all, "
             f"whatever the PHY rate. At {v(p.phy, '.0f', 'Mb/s')} the payload itself takes only "
             f"{v(r.get('pay', 0), '.1f', 'µs')} (top: the green bar). Hence "
             f"{v(r.get('thr', 0), '.1f', 'Mb/s')}, {v(r.get('eff', 0), '.1f', '%')} of the PHY "
             f"rate.</p>"
             "<p><b>A-MPDU aggregation</b> (802.11n onwards) packs up to 64 (Wi-Fi 6: 256) packets "
             "into one transmission with one Block Ack: the toll is paid once. That is why the "
             "jump from 54 Mb/s to gigabit Wi-Fi needed a new MAC as much as a new PHY.</p>")
        return "<h3>The fixed toll</h3>" + s + keybox(
            "Efficiency = payload time / (payload time + overhead). When the PHY gets faster, "
            "only bigger bursts keep the overhead small.")


# =============================================================================== 3. rate anomaly
class RateAnomaly(Experiment):
    title = "The rate anomaly and airtime fairness"
    blurb = "One slow station in the room slows everyone. Equal time instead of equal turns?"
    book = "sec:ch22:mac"
    controls = [
        Heading("Fast stations"),
        IntSlider("nf", "Number", 1, 5, 1),
        LogSlider("fast", "PHY rate", 24, 1200, 600, unit="Mb/s", fmt=".0f"),
        Choice("aggf", "A-MPDU", ["1", "8", "64"], "1"),
        Heading("Slow stations"),
        IntSlider("ns", "Number", 0, 5, 0),
        LogSlider("slow", "PHY rate", 1, 100, 6, unit="Mb/s", fmt=".1f",
                  help="A distant phone, or an old 802.11b/g device"),
        Heading("Access point"),
        Toggle("atf", "Airtime fairness", False,
               help="Share the medium by time rather than by transmission opportunities"),
    ]
    plots = [
        BarPlot("thr", "Throughput per station: red = equal turns, green = airtime fair", x="",
                y="throughput (Mb/s)", legend=None),
        BarPlot("air", "Share of the airtime", x="", y="airtime (%)", legend=None),
        Plot("tl", "One round of the medium (each station once, by DCF)", x="time (µs)",
             y="", legend=None),
    ]
    layout = [["thr", "air"], ["tl", "tl"]]
    row_stretch = [3, 1]
    readouts = [
        Readout("dcf", "Total, equal turns", "Mb/s", ".1f"),
        Readout("atf", "Total, airtime fair", "Mb/s", ".1f"),
        Readout("alone", "Fast station alone", "Mb/s", ".1f"),
        Readout("fastnow", "Fast station now", "Mb/s", ".1f"),
    ]
    challenges = [
        Challenge("Add one 6 Mb/s station and watch a 600 Mb/s station fall below 10 % of what "
                  "it gets alone (no airtime fairness).",
                  lambda s: (s.p.ns >= 1 and s.p.slow <= 6.5 and not s.p.atf and s.p.fast >= 590
                             and s.r.fastnow < 0.1 * s.r.alone)),
        Challenge("With slow stations present, switch on airtime fairness and at least triple "
                  "the total throughput.",
                  lambda s: s.p.ns >= 1 and s.p.atf and s.r.atf >= 3 * s.r.dcf),
        Challenge("Find the slow rate at which the anomaly costs less than 20 % (equal turns at "
                  "least 80 % of airtime fair), with one slow and one 600 Mb/s station.",
                  lambda s: (s.p.ns == 1 and s.p.nf == 1 and abs(s.p.fast - 600) < 15
                             and s.r.dcf >= 0.8 * s.r.atf and s.p.slow < 100)),
    ]

    def update(self, p):
        nagg = int(p.aggf)
        sta = [("fast", p.fast, nagg)] * p.nf + [("slow", p.slow, 1)] * p.ns
        T, B = [], []
        for _, rate, na in sta:
            parts, n_eff, thr = exchange_parts(rate, 1500, na)
            T.append(sum(d for _, d in parts))
            B.append(n_eff * 12000)
        T, B = np.array(T), np.array(B)
        N = len(sta)
        dcf = B / T.sum() / 1e6
        atf = B / T / N / 1e6
        air_dcf = 100 * T / T.sum()
        alone = B[0] / T[0] / 1e6
        x = np.arange(N)
        pb = self.plot("thr")
        cols = [BLUE if k == "fast" else ORANGE for k, _, _ in sta]
        pb.bars("d", x - 0.2, dcf, width=0.38, color=RED, name="equal turns (DCF)")
        pb.bars("a", x + 0.2, atf, width=0.38, color=GREEN, name="airtime fairness")
        labels = [f"F{i + 1} {p.fast:.0f}" for i in range(p.nf)] + \
                 [f"S{i + 1} {p.slow:.3g}" for i in range(p.ns)]
        pb.set_xticks([(i, l_) for i, l_ in enumerate(labels)])
        pb.set_xlim(-0.6, N - 0.4)
        pb.set_ylim(0, 1.3 * max(dcf.max(), atf.max(), 1e-3))
        pa = self.plot("air")
        share = air_dcf if not p.atf else np.full(N, 100 / N)
        pa.bars("s", x, share, width=0.6, colors=cols)
        for i, sv in enumerate(share):
            pa.text(f"t{i}", i, sv, f"{sv:.0f} %", anchor=(0.5, 1.05), size=9, bold=True)
        pa.set_xticks([(i, l_.split()[0]) for i, l_ in enumerate(labels)])
        pa.set_xlim(-0.6, N - 0.4)
        pa.set_ylim(0, 115)
        pa.set_title("Share of the airtime: " + ("airtime fairness" if p.atf else "equal turns (DCF)"))
        pt = self.plot("tl")
        x0 = 0.0
        for i, (k, rate, _) in enumerate(sta):
            d = T[i] * 1e6
            pt.bars(f"b{i}", [x0 + d / 2], [1.0], width=d * 0.98, color=cols[i])
            if d > 0.06 * T.sum() * 1e6:
                pt.text(f"l{i}", x0 + d / 2, 0.5, labels[i].split()[0], anchor=(0.5, 0.5),
                        size=9, color="#FFFFFF", bold=True)
            x0 += d
        pt.set_xlim(0, x0)
        pt.set_ylim(0, 1.05)
        pt.set_yticks([])
        tot_dcf, tot_atf = float(dcf.sum()), float(atf.sum())
        self.readout(dcf=tot_dcf, atf=tot_atf, alone=alone,
                     fastnow=float(atf[0] if p.atf else dcf[0]))

    def story(self, p):
        r = self.r
        s = ("<p>The DCF gives every station an equal chance to transmit, not equal time on the "
             "air. A slow station's packet occupies the medium many times longer than a fast "
             "one's (bottom: one round of turns, to scale).</p>")
        if p.ns:
            s += (f"<p>So the fast station's throughput collapses from "
                  f"{v(r.get('alone', 0), '.1f', 'Mb/s')} alone to "
                  f"{v(r.get('fastnow', 0), '.1f', 'Mb/s')}: everyone ends up at roughly the slow "
                  f"station's pace. This is the <b>rate anomaly</b> (Heusse et al., 2003). "
                  f"<b>Airtime fairness</b> in the access point gives each station an equal share "
                  f"of time instead: total {v(r.get('atf', 0), '.1f', 'Mb/s')} versus "
                  f"{v(r.get('dcf', 0), '.1f', 'Mb/s')}.</p>")
        else:
            s += "<p>Add a slow station (a distant phone, an old 802.11g laptop) and watch.</p>"
        return "<h3>Equal turns, unequal time</h3>" + s + keybox(
            "Under DCF all stations converge to the throughput of the slowest. Fairness in "
            "airtime, not in opportunities, keeps a fast network fast.")


# =============================================================================== 4. Minstrel
class MinstrelLive(Experiment):
    title = "Rate adaptation (Minstrel), live"
    blurb = "No CQI in Wi-Fi: the transmitter learns which MCS works by trying."
    book = "sec:ch22:realworld"
    animate = True
    autoplay = True
    fps = 12
    controls = [
        Heading("Minstrel"),
        Slider("look", "Look-around fraction", 0.0, 0.5, 0.10, step=0.01,
               help="Share of packets sent at a random other rate to keep the statistics fresh"),
        Choice("upd", "Update interval", ["20 ms", "100 ms", "500 ms"], "100 ms"),
        Slider("ewma", "EWMA weight of new statistics", 0.05, 1.0, 0.25, step=0.05),
        Heading("Channel"),
        Slider("snr", "Mean SNR", 10, 40, 24, step=0.5, unit="dB"),
        Slider("swing", "Walking swing", 0, 15, 9, step=0.5, unit="dB",
               help="How much the slow SNR varies as the station walks through the room"),
        Button("reset", "Restart"),
    ]
    plots = [
        Plot("snr", "SNR seen by the station (grey) and the MCS chosen (blue)", x="time (s)",
             y="SNR (dB) / MCS × 3", xlim=(-6, 0), ylim=(-5, 50), legend="tl", legend_cols=3),
        Plot("gp", "Goodput (0.1 s average)", x="time (s)", y="goodput (Mb/s)", xlim=(-6, 0),
             ylim=(0, 650), legend="tl", legend_cols=3),
        BarPlot("prob", "Success estimates (red = in use)", x="MCS", y="success probability",
                legend=None),
    ]
    layout = [["snr", "prob"], ["gp", "prob"]]
    col_stretch = [3, 1]
    readouts = [
        Readout("got", "Minstrel goodput", "Mb/s", ".0f"),
        Readout("orc", "Oracle", "Mb/s", ".0f"),
        Readout("pct", "Of the oracle", "%", ".0f", good=lambda x: x >= 75),
        Readout("fix", "Fixed MCS 6", "Mb/s", ".0f"),
    ]
    challenges = [
        Challenge("Reach 77 % of the oracle's goodput.",
                  lambda s: s.r.pct >= 77,
                  hint="Faster updates and a heavier EWMA weight react quicker to the walk."),
        Challenge("Switch off looking around and watch Minstrel get stuck below 50 % of the "
                  "oracle.",
                  lambda s: s.p.look == 0 and s.r.pct < 50,
                  hint="After a fade it never discovers that faster rates work again (Restart helps)."),
        Challenge("With a 30 dB mean SNR and 12 dB swings, beat the best fixed MCS by 20 %.",
                  lambda s: s.p.snr >= 30 and s.p.swing >= 12 and s.r.got >= 1.2 * s.exp.best_fixed,
                  hint="React faster: shorter updates, more weight on new statistics."),
    ]
    W = 6000
    RATE = ai.wifi_rate_mbps(np.arange(12), 80)

    def setup(self):
        self.on_reset(None)

    def on_reset(self, p):
        self.t = 0
        self.prob = np.full(12, 0.5)
        self.succ = np.zeros(12)
        self.att = np.zeros(12)
        self.cur = 0
        self.walk = 0.0
        self.fc = 0j
        n = self.W
        self.h_snr = np.full(n, np.nan)
        self.h_mcs = np.full(n, np.nan)
        self.h_got = np.zeros(n)
        self.h_orc = np.zeros(n)
        self.h_fix = np.zeros(n)
        self.h_best = np.zeros((n, 12))

    def run(self, p, n):
        r = self.rng
        upd = int(p.upd.split()[0])
        snrs = np.empty(n)
        mcs = np.empty(n)
        got = np.empty(n)
        orc = np.empty(n)
        fix = np.empty(n)
        best = np.empty((n, 12))
        u = r.random((n, 3))
        rnd = r.integers(0, 12, n)
        w = (r.standard_normal(n) + 1j * r.standard_normal(n)) / np.sqrt(2)
        a = np.exp(-1 / 60)
        K = 4.0
        for i in range(n):
            tt = (self.t + i) / 1000
            self.walk += 0.03 * r.standard_normal()
            self.walk *= 0.999
            self.fc = a * self.fc + np.sqrt(1 - a * a) * w[i]
            fast = np.abs(np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * self.fc) ** 2
            snr = p.snr + p.swing * np.sin(2 * np.pi * tt / 6.0 + 0.6) + self.walk + 10 * np.log10(fast)
            per = ai.wifi_per(snr, np.arange(12))
            m = rnd[i] if u[i, 0] < p.look else self.cur
            ok = u[i, 1] > per[m]
            self.att[m] += 1
            self.succ[m] += ok
            snrs[i], mcs[i] = snr, m
            got[i] = self.RATE[m] * ok
            best[i] = (1 - per) * self.RATE
            orc[i] = best[i].max()
            fix[i] = self.RATE[6] * (u[i, 2] > per[6])
            if (self.t + i) % upd == upd - 1:
                upd_m = self.att > 0
                self.prob[upd_m] = (1 - p.ewma) * self.prob[upd_m] + p.ewma * self.succ[upd_m] / self.att[upd_m]
                self.succ[:] = 0
                self.att[:] = 0
                self.cur = int(np.argmax(self.prob * self.RATE * (self.prob > 0.1)))
        self.t += n
        for arr, new in ((self.h_snr, snrs), (self.h_mcs, mcs), (self.h_got, got),
                         (self.h_orc, orc), (self.h_fix, fix)):
            arr[:-n] = arr[n:]
            arr[-n:] = new
        self.h_best[:-n] = self.h_best[n:]
        self.h_best[-n:] = best

    def update(self, p):
        self.on_reset(p)
        self.run(p, 1500 if self.quick else self.W)
        self.draw(p)

    def tick(self, p):
        self.run(p, 60)
        self.draw(p)

    def draw(self, p):
        t = (np.arange(self.W) - self.W) / 1000
        valid = np.isfinite(self.h_snr)
        sm = lambda x: np.convolve(x, np.ones(100) / 100)[:len(x)]      # trailing average
        ps = self.plot("snr")
        st_ = 4
        ps.line("snr", t[valid][::st_], self.h_snr[valid][::st_], color=GRAY, width=1.0, name="SNR")
        ps.line("mcs", t[valid][::st_], 3 * sm(np.nan_to_num(self.h_mcs))[valid][::st_], color=NAVY,
                width=2.2, name="mean MCS × 3")
        pg = self.plot("gp")
        nv = int(valid.sum())
        got, orc, fix = (sm(x)[valid] for x in (self.h_got, self.h_orc, self.h_fix))
        pg.line("orc", t[valid][::st_], orc[::st_], color=GREEN, width=1.8, name="oracle")
        pg.line("got", t[valid][::st_], got[::st_], color=NAVY, width=2.2, name="Minstrel")
        pg.line("fix", t[valid][::st_], fix[::st_], color=RED, width=1.4, name="fixed MCS 6")
        pp = self.plot("prob")
        pp.bars("p", np.arange(12), self.prob, width=0.62,
                colors=[RED if k == self.cur else NAVY for k in range(12)])
        pp.set_xticks([(k, str(k)) for k in range(12)])
        pp.set_ylim(0, 1.1)
        pp.set_xlim(-0.6, 11.6)
        g, o, f = (float(np.mean(x[valid])) for x in (self.h_got, self.h_orc, self.h_fix))
        bests = self.h_best[valid].mean(axis=0) if nv else np.zeros(12)
        self.best_fixed = float(bests.max())
        self.readout(got=g, orc=o, pct=100 * g / max(o, 1e-9), fix=f)

    def story(self, p):
        r = self.r
        s = ("<p>Wi-Fi has no channel-quality report: the transmitter only learns whether each "
             "packet was acknowledged. Linux's <b>Minstrel</b> keeps a running success "
             "probability per rate (right; red = the rate in use), picks the rate with the best "
             f"expected throughput every {p.upd}, and spends {v(100 * p.look, '.0f', '%')} of "
             "packets <i>looking around</i> at other rates.</p>"
             f"<p>The station walks through a room (grey: slow swings of "
             f"±{v(p.swing, '.1f', 'dB')} plus Rician fading). Minstrel delivers "
             f"{v(r.get('got', 0), '.0f', 'Mb/s')}, {v(r.get('pct', 0), '.0f', '%')} of an oracle "
             f"that knows the SNR of every packet; a fixed MCS gets "
             f"{v(r.get('fix', 0), '.0f', 'Mb/s')}.</p>")
        if p.look == 0:
            s += ("<p>Without looking around, a rate that failed in a fade is never tried again: "
                  "the statistics go stale and Minstrel gets stuck low.</p>")
        return "<h3>Learning by trying</h3>" + s + keybox(
            "Rate adaptation by trial and error: exploit the best-known rate, explore the others "
            "a little, forget old evidence at the right speed. A bandit problem in every laptop.")


# =============================================================================== 5. BLE GFSK
BTS = {"0.3": 0.3, "0.5": 0.5, "1.0": 1.0}


def ble_ber_sim(ebn0_db, h, bt, n, r, sps=8):
    bits = r.integers(0, 2, n)
    x, _ = iot.gfsk_mod(bits, sps=sps, bt=bt, h=h)
    y = x + np.sqrt(sps / 10 ** (ebn0_db / 10)) * cn(r, len(x))
    y = np.convolve(y, firwin(4 * sps + 1, 1.6 / sps), "same")
    fd = np.r_[0, iot.fm_discriminator(y)]
    dec = np.add.reduceat(fd, np.arange(0, len(fd), sps)) > 0
    return float(np.mean(dec[2:-2] != bits[2:-2])), fd


class BLEGFSK(Experiment):
    title = "Bluetooth LE: GFSK"
    blurb = "A constant-envelope FM radio that a cheap chip can send and a discriminator can hear."
    book = "sec:ch22:ble"
    heavy = False
    controls = [
        Slider("h", "Modulation index h", 0.3, 1.0, 0.5, step=0.01,
               help="Peak frequency deviation = h/2 × the symbol rate (BLE: 0.45–0.55)"),
        Choice("bt", "Gaussian filter BT", list(BTS), "0.5",
               help="Bandwidth-time product: smaller = narrower spectrum, more ISI"),
        Choice("phy", "PHY", ["LE 1M", "LE 2M"]),
        Slider("ebn0", "Eb/N0", 0, 25, 14, step=0.5, unit="dB"),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("f", "Instantaneous frequency of 16 bits", x="time (bits)", y="frequency (kHz)",
             xlim=(0, 16), legend="tl", legend_cols=2),
        SpectrumPlot("s", "Spectrum and the 2 MHz channel", x="frequency (MHz)", y="PSD (dB)",
                     xlim=(-4, 4), ylim=(-60, 5), legend=None),
        EyePlot("eye", "Discriminator output eye", yrange=(-1.6, 1.6)),
        BERPlot("ber", "Bit error rate vs Eb/N0", x="Eb/N0 (dB)", xlim=(0, 22), ylim=(1e-5, 0.5),
                legend="tr"),
    ]
    layout = [["f", "s"], ["eye", "ber"]]
    readouts = [
        Readout("bw", "99 % bandwidth", "MHz", ".2f", good=lambda x: x <= 2.0),
        Readout("dev", "Peak deviation", "kHz", ".0f"),
        Readout("ber", "BER (simulated)", "", "sci", good=lambda x: x < 1e-3),
        Readout("need", "Eb/N0 for 10⁻³", "dB", None),
    ]
    challenges = [
        Challenge("Keep the BER below 10⁻³ at an Eb/N0 of 11 dB or less.",
                  lambda s: s.p.ebn0 <= 11 and s.r.ber < 1e-3,
                  hint="A larger modulation index spreads the two tones further apart."),
        Challenge("Squeeze h or BT until the BER exceeds 10⁻³ at 14 dB.",
                  lambda s: s.p.ebn0 >= 14 and s.r.ber > 1e-3),
        Challenge("Find a setting whose 99 % bandwidth exceeds the 2 MHz LE 1M channel.",
                  lambda s: s.p.phy == "LE 1M" and s.r.bw > 2.0),
    ]
    SPS = 8

    def setup(self):
        self.seed = 5
        self.curve = {}

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        rs = 1.0 if p.phy == "LE 1M" else 2.0
        bt = BTS[p.bt]
        r = np.random.default_rng(self.seed)
        sps = 16
        bits = r.integers(0, 2, 3000)
        x, f = iot.gfsk_mod(bits, sps=sps, bt=bt, h=p.h)
        tt = np.arange(16 * sps) / sps
        pf = self.plot("f")
        nrz = np.repeat(2 * bits[:16] - 1.0, sps) * p.h / 2 * rs * 1e3
        pf.line("nrz", tt, nrz, color=GRAY, width=1.2, name="bits × h/2 (no filter)")
        pf.line("g", tt, f[:16 * sps] * rs * 1e3, color=NAVY, width=2.4, name=f"Gaussian, BT = {p.bt}")
        lim = 0.55 * rs * 1e3
        pf.set_ylim(-1.5 * lim, 1.6 * lim)
        # spectrum
        X = np.abs(np.fft.fft(x[:len(x) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024),
                              axis=1)) ** 2
        P = np.fft.fftshift(X.mean(0))
        fr = np.fft.fftshift(np.fft.fftfreq(1024, 1 / sps)) * rs
        ps = self.plot("s")
        ps.band("ch", -1, 1, color=GREEN, alpha=0.10)
        ps.line("p", fr, db(P / P.max()), color=NAVY, width=2.0)
        o = np.argsort(np.abs(fr))
        c = np.cumsum(P[o]) / P.sum()
        bw99 = 2 * np.abs(fr[o][np.searchsorted(c, 0.99)])
        ps.vline("b1", bw99 / 2, color=RED, style="--", label="99 %", label_pos=0.85)
        ps.vline("b2", -bw99 / 2, color=RED, style="--")
        # BER and eye
        ber, fd = ble_ber_sim(p.ebn0, p.h, bt, 6000 if self.quick else 20000, r, self.SPS)
        pe = self.plot("eye")
        pe.eye("eye", fd[:4000] / (p.h / 2 / self.SPS), self.SPS, n_sym=2, offset=self.SPS // 2,
               yrange=(-1.6, 1.6))
        pb = self.plot("ber")
        x_ = np.linspace(0, 22, 111)
        pb.theory("fsk", x_, 0.5 * np.exp(-10 ** (x_ / 10) / 2), color=GRAY, style="--",
                  name="noncoherent BFSK ½e^(−Eb/2N0)")
        pb.theory("msk", x_, cl.ber_bpsk(x_), color=GREEN, style=":", name="coherent MSK/BPSK")
        key = (round(p.h, 2), p.bt)
        if key not in self.curve:
            e_ = np.arange(2, 21, 2.0)
            rr = np.random.default_rng(99)
            self.curve[key] = (e_, np.array([ble_ber_sim(e, p.h, bt, 4000 if self.quick else 12000,
                                                         rr, self.SPS)[0] for e in e_]))
        e_, b_ = self.curve[key]
        okm = b_ > 0
        pb.sim("sim", e_[okm], b_[okm], color=NAVY, name="discriminator, simulated")
        pb.scatter("now", [p.ebn0], [max(ber, 2e-5)], color=RED, size=13, symbol="d")
        lg = np.log10(np.maximum(b_, 1e-6))
        need = "—"
        if lg.max() > -3 > lg.min():
            i = np.flatnonzero(lg <= -3)[0]
            if i:
                need = f"{np.interp(-3, [lg[i], lg[i - 1]], [e_[i], e_[i - 1]]):.1f}"
        self.readout(bw=bw99, dev=p.h / 2 * rs * 1e3, ber=ber, need=need)

    def story(self, p):
        r = self.r
        rs = 1.0 if p.phy == "LE 1M" else 2.0
        s = (f"<p>BLE sends bits by shifting frequency: ±{v(r.get('dev', 0), '.0f', 'kHz')} around "
             f"the carrier for {p.phy} (h = {v(p.h)}). A Gaussian filter (BT = {p.bt}) rounds the "
             f"steps (top left) so the spectrum fits the 2 MHz channel: 99 % of the power within "
             f"{v(r.get('bw', 0), '.2f', 'MHz')}.</p>"
             "<p>The envelope is constant, so the transmitter can use a saturated, efficient "
             "power amplifier, and the receiver can be a limiter plus a <b>frequency "
             "discriminator</b> (bottom left: its output, sampled once per bit). That simplicity "
             f"costs sensitivity: this receiver needs {v(r.get('need', '—'), unit='dB')} of "
             "Eb/N0 for 10⁻³, a few dB more than coherent detection would.</p>")
        return "<h3>FM for bits</h3>" + s + keybox(
            "GFSK trades a few dB of sensitivity for a radio that costs cents and sips power: "
            "the right trade for a device that talks for a millisecond a second.")


# =============================================================================== 6. BLE battery
CELLS = {"CR2032 (225 mAh)": 225.0, "CR2477 (1000 mAh)": 1000.0, "2 × AA (2500 mAh)": 2500.0}


class BLEBattery(Experiment):
    title = "A coin cell's life"
    blurb = "Microcoulombs per advertisement, microamps asleep: years on one battery."
    book = "sec:ch22:ble"
    controls = [
        Heading("Advertising"),
        LogSlider("iv", "Advertising interval", 0.02, 10.24, 0.1, unit="s", fmt=".2f"),
        IntSlider("pl", "Payload", 0, 31, 31, unit="bytes"),
        IntSlider("nch", "Advertising channels", 1, 3, 3),
        Choice("phy", "PHY", ["LE 1M", "LE 2M"]),
        Heading("Hardware"),
        Slider("itx", "Radio current (TX)", 3, 20, 6, step=0.5, unit="mA"),
        LogSlider("isl", "Sleep current", 0.2, 20, 2, unit="µA", fmt=".2f"),
        Choice("cell", "Battery", list(CELLS), "CR2032 (225 mAh)", style="menu"),
    ]
    plots = [
        Plot("ev", "One advertising event: current drawn", x="time (ms)", y="current (mA)",
             legend=None),
        Plot("life", "Battery life vs advertising interval", x="interval (s)", y="life (years)",
             logx=True, logy=True, xlim=(0.02, 10.24), ylim=(0.01, 50), legend="tl"),
        BarPlot("q", "Where the charge goes, per interval", y="charge (µC)"),
    ]
    layout = [["ev", "ev"], ["life", "q"]]
    row_stretch = [2, 3]
    readouts = [
        Readout("q", "Charge per event", "µC", ".1f"),
        Readout("iavg", "Average current", "µA", ".1f"),
        Readout("life", "Battery life", "years", ".2f", good=lambda x: x >= 2),
        Readout("tpkt", "Air time per packet", "µs", ".0f"),
    ]
    challenges = [
        Challenge("Reproduce Chapter 22: a CR2032 beacon with 31-byte adverts on three channels "
                  "lasting about 2.0 years.",
                  lambda s: (s.p.pl == 31 and s.p.nch == 3 and s.p.cell.startswith("CR2032")
                             and abs(s.r.life - 2.03) < 0.1 and abs(s.p.itx - 6) < 0.01)),
        Challenge("Keep a 100 ms interval (snappy discovery) and still last a year on a CR2032.",
                  lambda s: s.p.iv <= 0.105 and s.p.cell.startswith("CR2032") and s.r.life >= 1,
                  hint="Fewer channels, shorter packets, the 2M PHY, a frugal radio."),
        Challenge("Find the interval beyond which sleeping costs more charge than advertising.",
                  lambda s: s.exp.sleep_share > 0.5 and s.exp.sleep_share_less <= 0.5,
                  hint="Move the interval slowly; watch the bars."),
    ]

    def charge(self, p, iv=None):
        rate = 1e6 if p.phy == "LE 1M" else 2e6
        q, t = ai.ble_adv_charge(payload=p.pl, i_tx=p.itx * 1e-3, rate=rate, n_ch=p.nch)
        return q, t

    def update(self, p):
        q, tpkt = self.charge(p)
        cap = CELLS[p.cell]
        isl = p.isl * 1e-6
        life = float(ai.battery_life_years(q, p.iv, isl, cap))
        iavg = q / p.iv + isl
        # event waveform
        pe = self.plot("ev")
        t, i_ = [0.0], [p.isl * 1e-3]
        x0 = 0.2
        t += [x0, x0]
        i_ += [p.isl * 1e-3, 3.0]
        x0 += 0.4
        t += [x0, x0]
        i_ += [3.0, p.isl * 1e-3]
        for k in range(p.nch):
            a = x0 + 0.15
            b = a + 0.15 + tpkt * 1e3
            t += [a, a, b, b]
            i_ += [p.isl * 1e-3, p.itx, p.itx, p.isl * 1e-3]
            x0 = b + 0.25
        t += [x0 + 0.3]
        i_ += [p.isl * 1e-3]
        pe.line("i", t, i_, color=NAVY, width=2.0, fill=0, fill_alpha=0.2)
        pe.text("cpu", 0.4, 3.2, "CPU", color=GRAY, size=8.5, anchor=(0.5, 0))
        pe.set_ylim(0, max(p.itx, 3) * 1.3)
        pe.set_xlim(0, x0 + 0.3)
        pe.set_title(f"One advertising event ({p.nch} packet{'s' if p.nch > 1 else ''} of "
                     f"{tpkt * 1e6:.0f} µs), then {p.iv * 1e3:.0f} ms asleep at {p.isl:.2g} µA")
        # life curve
        ivs = np.logspace(np.log10(0.02), np.log10(10.24), 200)
        pl = self.plot("life")
        pl.line("l", ivs, ai.battery_life_years(q, ivs, isl, cap), color=NAVY, width=2.2,
                name=p.cell.split(" (")[0])
        pl.line("cap", ivs, cap * 1e-3 / isl / 24 / 365 * np.ones_like(ivs), color=GRAY, style=":",
                width=1.0, name="sleep-only limit")
        pl.hline("10", 10, color=RED, style="--", label="10 y: self-discharge", label_pos=0.62)
        pl.scatter("now", [p.iv], [life], color=PURPLE, size=15, symbol="d")
        # charge bars per interval
        qtx = q - 0.4e-3 * 3e-3
        qcpu = 0.4e-3 * 3e-3
        qsl = isl * p.iv
        self.sleep_share = qsl / (q + qsl)
        ivl = p.iv / 1.03
        self.sleep_share_less = isl * ivl / (q + isl * ivl)
        pq = self.plot("q")
        vals = np.array([qtx, qcpu, qsl]) * 1e6
        pq.bars("b", np.arange(3), vals, width=0.6, colors=[BLUE, GRAY, PURPLE])
        for k, val in enumerate(vals):
            pq.text(f"t{k}", k, val, f"{val:.1f}", anchor=(0.5, 1.05), size=9, bold=True)
        pq.set_xticks([(0, "radio"), (1, "CPU"), (2, "sleep")])
        pq.set_xlim(-0.6, 2.6)
        pq.set_ylim(0, 1.25 * vals.max() + 0.1)
        self.readout(q=q * 1e6, iavg=iavg * 1e6, life=life, tpkt=tpkt * 1e6)

    def story(self, p):
        r = self.r
        s = (f"<p>A beacon wakes, runs its CPU for a moment, transmits the same "
             f"{p.pl + 16}-byte advertisement on {p.nch} channel(s) "
             f"({v(r.get('tpkt', 0), '.0f', 'µs')} each at {p.phy}) and goes back to sleep. "
             f"Each event costs {v(r.get('q', 0), '.1f', 'µC')}. Every "
             f"{v(p.iv * 1e3, '.0f', 'ms')} that adds up to an average of "
             f"{v(r.get('iavg', 0), '.1f', 'µA')}, and a {p.cell} lasts "
             f"{v(r.get('life', 0), '.2f', 'years')}.</p>")
        if self.sleep_share > 0.5:
            s += ("<p>At this interval the <b>sleep current</b> dominates (right): advertising "
                  "less often barely helps any more. Real cells also lose a few per cent a year to "
                  "self-discharge, so ten years is a practical ceiling.</p>")
        else:
            s += ("<p>Here the radio dominates: halve the advertising rate and the battery life "
                  "almost doubles.</p>")
        return "<h3>Energy is the real spec</h3>" + s + keybox(
            "Life ≈ capacity / (charge per event / interval + sleep current). BLE wins by making "
            "both terms tiny: short packets, fast radios, deep sleep.")


# =============================================================================== 7. 802.15.4
TAB = iot.ieee802154_chips()
BCH = 2.0 * TAB - 1


def zb_ser(snr_db, n_sym, r, sps=4):
    """SER of the 2.4 GHz O-QPSK PHY with a 16-way correlation receiver; snr_db is the SNR in
    the chip-rate bandwidth (2 MHz)."""
    sym = r.integers(0, 16, n_sym)
    x = iot.oqpsk_halfsine(TAB[sym].ravel(), sps=sps)
    y = x + np.sqrt(sps / 10 ** (snr_db / 10)) * cn(r, len(x))
    pls = np.sin(np.pi * np.arange(2 * sps) / (2 * sps))
    mi = np.convolve(y.real, pls[::-1])[2 * sps - 1::2 * sps][:16 * n_sym]
    mq = np.convolve(y.imag, pls[::-1])[3 * sps - 1::2 * sps][:16 * n_sym]
    chips = np.empty(32 * n_sym)
    chips[0::2], chips[1::2] = mi, mq
    corr = chips.reshape(n_sym, 32) @ BCH.T
    return float(np.mean(np.argmax(corr, axis=1) != sym)), corr, sym


class Zigbee154(Experiment):
    title = "802.15.4: chips and O-QPSK"
    blurb = "Zigbee, Thread and Matter: 4 bits become 32 chips, sent with a constant envelope."
    book = "sec:ch22:154"
    heavy = True
    controls = [
        IntSlider("sym", "Symbol sent", 0, 15, 3),
        Slider("snr", "SNR in the 2 MHz chip bandwidth", -15, 10, -4, step=0.5, unit="dB"),
        Toggle("oq", "Offset the Q chips (O-QPSK)", True,
               help="Off: plain QPSK with half-sine pulses, whose envelope dips to zero"),
        Button("again", "New noise"),
    ]
    plots = [
        Plot("wave", "I, Q and the envelope", x="time (chips)", y="amplitude", xlim=(0, 32),
             ylim=(-3.6, 3.6), legend="tl", legend_cols=3),
        ImagePlot("cm", "Correlation between the 16 chip sequences", x="symbol", y="symbol"),
        BarPlot("corr", "Correlator outputs for one received symbol", x="symbol hypothesis",
                y="correlation (norm.)"),
        BERPlot("ser", "Symbol error rate", x="SNR in the chip bandwidth (dB)", y="SER",
                xlim=(-16, 4), ylim=(1e-4, 1), legend="bl"),
    ]
    layout = [["wave", "cm"], ["corr", "ser"]]
    readouts = [
        Readout("dec", "Decision", "", None),
        Readout("xc", "Worst cross-correlation", "", ".2f"),
        Readout("pg", "Processing gain", "dB", ".1f"),
        Readout("ser", "SER now (2000 symbols)", "", "sci", good=lambda x: x < 1e-2),
    ]
    challenges = [
        Challenge("Find the SNR where the symbol error rate drops to 1 % (± 0.5 dB).",
                  lambda s: s.r.ser <= 0.012 and s.exp.ser_less > 0.008,
                  hint="Below 0 dB! The 32-chip correlation buys about 9 dB."),
        Challenge("Remove the offset and see the envelope fall to zero between chips.",
                  lambda s: not s.p.oq and s.exp.env_min < 0.1),
        Challenge("Send symbol 0 and find its most similar neighbour (correlation ≥ 0.25 in "
                  "magnitude).",
                  lambda s: s.p.sym == 0 and s.exp.max_other >= 0.25),
    ]

    def setup(self):
        self.seed = 9
        self.res = {}

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        sps = 16
        chips = TAB[[p.sym, (p.sym + 5) % 16]].ravel()
        x = iot.oqpsk_halfsine(chips, sps=sps)
        if not p.oq:
            c = 2 * chips - 1.0
            pl = np.sin(np.pi * np.arange(2 * sps) / (2 * sps))
            i_ = np.zeros(len(c) * sps + 2 * sps)
            q_ = np.zeros_like(i_)
            for k in range(len(c) // 2):
                i_[2 * k * sps:2 * k * sps + 2 * sps] += c[2 * k] * pl
                q_[2 * k * sps:2 * k * sps + 2 * sps] += c[2 * k + 1] * pl
            x = i_ + 1j * q_
        t = np.arange(len(x)) / sps
        pw = self.plot("wave")
        m = (t >= 1) & (t <= 31)
        pw.line("i", t, x.real + 1.3, color=NAVY, width=1.4, name="I (+1.3)")
        pw.line("q", t, x.imag - 1.3, color=RED, width=1.4, name="Q (−1.3)")
        pw.line("e", t, np.abs(x) - 3.0, color=GREEN, width=2.0, name="|envelope| (−3)")
        self.env_min = float(np.abs(x[m]).min())
        pw.set_yticks([])
        C = BCH @ BCH.T / 32
        pc = self.plot("cm")
        cmap = ((0, "#C0392B"), (0.5, "#F4F6F8"), (1, "#1B3A5C"))
        pc.image("c", C, x=(-0.5, 15.5), y=(-0.5, 15.5), cmap=cmap, levels=(-1, 1), colorbar=True)
        pc.set_xlim(-0.5, 15.5)
        pc.set_ylim(-0.5, 15.5)
        off = C[~np.eye(16, dtype=bool)]
        self.max_other = float(np.abs(np.delete(C[p.sym], p.sym)).max())
        # one received symbol
        r = np.random.default_rng(self.seed)
        sps2 = 4
        xs = iot.oqpsk_halfsine(TAB[p.sym], sps=sps2)
        y = xs + np.sqrt(sps2 / 10 ** (p.snr / 10)) * cn(r, len(xs))
        pls = np.sin(np.pi * np.arange(2 * sps2) / (2 * sps2))
        mi = np.convolve(y.real, pls[::-1])[2 * sps2 - 1::2 * sps2][:16]
        mq = np.convolve(y.imag, pls[::-1])[3 * sps2 - 1::2 * sps2][:16]
        ch = np.empty(32)
        ch[0::2], ch[1::2] = mi, mq
        corr = BCH @ ch
        corr = corr / max(np.abs(corr).max(), 1e-9)
        dec = int(np.argmax(corr))
        pb = self.plot("corr")
        pb.bars("b", np.arange(16), corr, width=0.6,
                colors=[GREEN if k == p.sym else (RED if k == dec else GRAY) for k in range(16)])
        pb.set_xticks([(k, str(k)) for k in range(16)])
        pb.set_ylim(-1.1, 1.25)
        pb.set_xlim(-0.6, 15.6)
        # SER now and the curve
        ser, _, _ = zb_ser(p.snr, 600 if self.quick else 2000, r)
        self.ser_less = zb_ser(p.snr - 0.5, 600 if self.quick else 2000, np.random.default_rng(self.seed + 7))[0]
        ps = self.plot("ser")
        ps.scatter("now", [p.snr], [max(ser, 1.2e-4)], color=RED, size=13, symbol="d")
        pts = sorted(self.res.items())
        if pts:
            ps.sim("sim", [a for a, _ in pts], [max(b, 1.2e-4) for _, b in pts], color=NAVY,
                   name="simulated (Monte Carlo)")
        ps.vline("zero", 0, color=GRAY, style=":", label="0 dB", label_pos=0.9)
        self.readout(dec=f"{dec} {'✓' if dec == p.sym else '✗'}", xc=float(np.abs(off).max()),
                     pg=10 * np.log10(32 / 4), ser=ser)

    def background(self, p):
        if len(self.res) >= 10:
            return
        for s_ in np.arange(-16, 3, 2.0):
            r = np.random.default_rng(int(100 + s_))
            yield (float(s_), zb_ser(s_, 800 if self.quick else 4000, r)[0])

    def progress(self, p, item):
        self.res[item[0]] = item[1]
        pts = sorted(self.res.items())
        self.plot("ser").sim("sim", [a for a, _ in pts], [max(b, 1.2e-4) for _, b in pts],
                             color=NAVY, name="simulated (Monte Carlo)")

    def story(self, p):
        r = self.r
        s = ("<p>The 2.4 GHz 802.15.4 PHY maps each 4-bit symbol to one of 16 sequences of 32 "
             "chips (2 Mchip/s, so 250 kb/s). Even chips go on I, odd chips on Q delayed by one "
             "chip, each a half-sine pulse: that is <b>O-QPSK</b>, which is exactly MSK. "
             + ("The envelope never touches zero (green)." if p.oq else
                "Without the offset, I and Q change sign together and the envelope crashes to "
                "zero: a PA nightmare.") + "</p>")
        s += (f"<p>The receiver correlates the 32 received chips with all 16 sequences and picks "
              f"the largest (bottom left: green = sent, red = a wrong winner). The sequences are "
              f"nearly orthogonal (worst cross-correlation {v(r.get('xc', 0), '.2f')}, top right), "
              f"and 32 chips for 4 bits is {v(r.get('pg', 0), '.1f', 'dB')} of processing gain, "
              f"so it works at {v(p.snr, '.1f', 'dB')} SNR: SER {v(r.get('ser', 0), '.3f')}.</p>")
        return "<h3>Spreading for robustness</h3>" + s + keybox(
            "DSSS with a 16-ary code: modest rate, a constant-envelope waveform and decoding "
            "below the noise floor, ideal for battery-powered meshes.")


# =============================================================================== 8. LoRa chirps
class LoRaChirps(Experiment):
    title = "LoRa: chirps and the dechirp"
    blurb = "Every symbol is a sweep; multiply by a down-chirp and it becomes a single tone."
    book = "sec:ch22:lora"
    heavy = True
    controls = [
        IntSlider("sf", "Spreading factor SF", 7, 12, 7),
        Choice("bw", "Bandwidth", ["125", "250", "500"], "125", help="kHz"),
        Slider("snr", "SNR in the bandwidth", -25, 5, -5, step=0.5, unit="dB"),
        IntSlider("sym", "Data symbol", 0, 4095, 100, help="Clipped to 2^SF − 1"),
        Button("again", "New noise"),
    ]
    plots = [
        ImagePlot("spec", "Spectrogram: 4 preamble up-chirps, then 4 data symbols",
                  x="time (symbols)", y="frequency (× BW)"),
        Plot("fft", "Dechirp + FFT of your symbol", x="FFT bin", y="|Y[k]|² / M", legend=None),
        BERPlot("ser", "Symbol error rate", x="SNR in the bandwidth (dB)", y="SER",
                xlim=(-25, 0), ylim=(1e-4, 1), legend="bl"),
    ]
    layout = [["spec", "spec"], ["fft", "ser"]]
    row_stretch = [1, 1]
    readouts = [
        Readout("ts", "Symbol time", "ms", ".3f"),
        Readout("rb", "Bit rate (CR 4/5)", "b/s", ".0f"),
        Readout("dec", "Decided symbol", "", None),
        Readout("pg", "Processing gain", "dB", ".1f"),
    ]
    challenges = [
        Challenge("Decode a symbol correctly at −15 dB SNR or below.",
                  lambda s: s.p.snr <= -15 and "✓" in s.r.dec,
                  hint="Each SF step adds 3 dB of processing gain (and halves the rate)."),
        Challenge("Find the slowest LoRa: a symbol longer than 30 ms.",
                  lambda s: s.r.ts > 30),
        Challenge("At SF7 and −8 dB, watch the decision fail (try New noise).",
                  lambda s: s.p.sf == 7 and s.p.snr <= -8 and "✗" in s.r.dec),
    ]

    def setup(self):
        self.seed = 2
        self.sims = {}

    def on_again(self, p):
        self.seed += 1

    def update(self, p):
        M = 2 ** p.sf
        bw = float(p.bw) * 1e3
        sym = min(p.sym, M - 1)
        osr = 2
        data = [sym, M // 2, M - 20 if M > 20 else 1, M // 4]
        x = np.concatenate([iot.lora_symbols(p.sf, [0] * 4, osr), iot.lora_symbols(p.sf, data, osr)])
        nper = max(32, M // 8)
        frames = x[:len(x) // nper * nper].reshape(-1, nper)
        S = np.abs(np.fft.fftshift(np.fft.fft(frames * np.hanning(nper), axis=1), axes=1)) ** 2
        S = db(S / S.max()).T
        ps = self.plot("spec")
        ps.image("s", np.clip(S, -30, 0), x=(0, 8), y=(-osr / 2, osr / 2), levels=(-30, 0))
        ps.set_xlim(0, 8)
        ps.set_ylim(-0.75, 0.75)
        ps.vline("d", 4, color=RED, style="--", label="data", label_pos=0.95)
        r = np.random.default_rng(self.seed)
        s = iot.lora_symbols(p.sf, [sym])
        y = s + 10 ** (-p.snr / 20) * cn(r, M)
        d, X = iot.lora_demod(y, p.sf)
        pf = self.plot("fft")
        pf.line("X", np.arange(M), X[0] ** 2 / M, color=NAVY, width=1.0)
        pf.scatter("pk", [sym], [X[0, sym] ** 2 / M], color=GREEN, size=11, outline=GREEN)
        if d[0] != sym:
            pf.scatter("bad", [d[0]], [X[0, d[0]] ** 2 / M], color=RED, size=11, symbol="x")
        pf.set_xlim(0, M - 1)
        pf.set_ylim(0, 1.2 * (X[0] ** 2 / M).max())
        pser = self.plot("ser")
        snr = np.linspace(-25, 0, 51)
        cols = [NAVY, RED, GREEN, ORANGE, PURPLE, TEAL]
        for k, sf in enumerate(range(7, 13)):
            pser.theory(f"t{sf}", snr, iot.lora_ser_theory(sf, snr, ngrid=1500), color=cols[k],
                        width=2.4 if sf == p.sf else 1.0, name=f"SF{sf}" if sf in (7, 12) or sf == p.sf else None)
        pts = sorted(self.sims.get(p.sf, {}).items())
        if pts:
            pser.sim("sim", [a for a, _ in pts], [max(b, 1.2e-4) for _, b in pts],
                     color=cols[p.sf - 7], name=f"SF{p.sf} simulated")
        pser.vline("now", p.snr, color=GRAY, style=":")
        self.readout(ts=M / bw * 1e3, rb=iot.lora_bitrate(p.sf, bw), pg=10 * np.log10(M),
                     dec=f"{d[0]} {'✓' if d[0] == sym else '✗'}")

    def background(self, p):
        req = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}[p.sf]
        have = self.sims.setdefault(p.sf, {})
        for s_ in np.arange(round(req) - 5, round(req) + 2, 1.0):
            if s_ in have:
                continue
            n = 300 if self.quick else max(400, 400000 // 2 ** p.sf)
            yield (p.sf, float(s_), iot.lora_ser_sim(p.sf, s_, n, rng=int(p.sf * 10 + s_ + 40)))

    def progress(self, p, item):
        sf, s_, ser = item
        self.sims.setdefault(sf, {})[s_] = ser
        if sf == p.sf:
            pts = sorted(self.sims[sf].items())
            cols = [NAVY, RED, GREEN, ORANGE, PURPLE, TEAL]
            self.plot("ser").sim("sim", [a for a, _ in pts], [max(b, 1.2e-4) for _, b in pts],
                                 color=cols[sf - 7], name=f"SF{sf} simulated")

    def story(self, p):
        r = self.r
        M = 2 ** p.sf
        s = (f"<p>A LoRa symbol is a <b>chirp</b>: a sweep across the {p.bw} kHz channel lasting "
             f"2<sup>SF</sup> = {v(M, 'd')} chips ({v(r.get('ts', 0), '.3f', 'ms')}). The data value "
             f"is where the sweep starts: it wraps around (top, after the dashed line).</p>"
             f"<p>The receiver multiplies by a down-chirp, which turns each symbol into a pure "
             f"tone, then takes an FFT: all {M} chips pile their energy into one bin "
             f"({v(r.get('pg', 0), '.1f', 'dB')} of gain), standing out of noise that is "
             f"{v(-p.snr, '.1f', 'dB')} stronger than the signal. Result: {v(r.get('dec', ''))}.</p>"
             "<p>This is noncoherent M-ary orthogonal signalling: each SF step buys about 2.5 dB of "
             "sensitivity and halves the bit rate.</p>")
        return "<h3>Below the noise floor</h3>" + s + keybox(
            "Dechirp + FFT is the cheapest optimal receiver there is: one multiply and one FFT per "
            "symbol, robust to Doppler and offsets, and 20 dB below the noise at SF12.")


# =============================================================================== 9. budgets
class RangeBudgets(Experiment):
    title = "LoRa air time, range, and RFID"
    blurb = "Duty cycles, kilometres of range, and a tag with no battery at all."
    book = "sec:ch22:lora"
    controls = [
        Heading("LoRa (868 MHz, 125 kHz)"),
        IntSlider("sf", "Spreading factor", 7, 12, 7),
        IntSlider("pl", "Payload", 1, 51, 20, unit="bytes"),
        Slider("ptx", "Transmit power", 2, 20, 14, step=1, unit="dBm"),
        Slider("n", "Path-loss exponent", 2.0, 4.5, 3.5, step=0.05,
               help="2.8 suburban, 3.5 urban (beyond 1 m of free space)"),
        Slider("margin", "Fade margin", 0, 20, 10, step=0.5, unit="dB"),
        Heading("UHF RFID (915 MHz)"),
        Slider("eirp", "Reader EIRP", 20, 36, 36, step=0.5, unit="dBm",
               help="36 dBm (4 W) is the FCC limit"),
        Slider("tag", "Tag chip sensitivity", -25, -10, -22, step=0.5, unit="dBm"),
    ]
    plots = [
        Plot("toa", "LoRa time on air", x="spreading factor", y="time on air (ms)", logy=True,
             xlim=(6.6, 12.4), ylim=(10, 5000), legend="tl"),
        Plot("rng", "LoRa range with your margin", x="spreading factor", y="range (km)",
             logy=True, xlim=(6.6, 12.4), ylim=(0.1, 50), legend="tl"),
        Plot("rfid", "UHF RFID: forward and reverse links", x="distance (m)", y="power (dBm)",
             logx=True, xlim=(0.3, 40), ylim=(-100, 30), legend="tr"),
    ]
    layout = [["toa", "rng"], ["rfid", "rfid"]]
    readouts = [
        Readout("toa", "Time on air", "ms", ".1f"),
        Readout("ph", "Packets/hour at 1 % duty", "", ".0f"),
        Readout("km", "LoRa range", "km", ".2f"),
        Readout("m", "RFID read range", "m", ".1f"),
    ]
    challenges = [
        Challenge("Reproduce Chapter 22: 20 bytes at SF12 take about 1.32 s on air.",
                  lambda s: s.p.sf == 12 and s.p.pl == 20 and abs(s.r.toa - 1319) < 15),
        Challenge("Stay inside the US915 400 ms dwell limit and still reach 1.5 km (urban, n = 3.5, "
                  "10 dB margin).",
                  lambda s: (s.r.toa <= 400 and s.r.km >= 1.5 and s.p.n >= 3.49 and s.p.margin >= 10),
                  hint="The highest spreading factor that fits the dwell time, then more power."),
        Challenge("RFID: reach 10 m read range at a reader EIRP of 30 dBm or less.",
                  lambda s: s.p.eirp <= 30 and s.r.m >= 10,
                  hint="The tag chip's sensitivity is everything: the forward link limits range."),
    ]
    F = 868e6

    def update(self, p):
        sfs = np.arange(7, 13)
        pt = self.plot("toa")
        for pl_, col in ((10, GREEN), (p.pl, NAVY), (51, ORANGE)):
            key = "cur" if pl_ == p.pl else f"p{pl_}"
            pt.line(key, sfs, [iot.lora_time_on_air(pl_, s_) * 1e3 for s_ in sfs], color=col,
                    width=2.4 if key == "cur" else 1.0, name=f"{pl_} bytes")
            pt.scatter(key + "m", sfs, [iot.lora_time_on_air(pl_, s_) * 1e3 for s_ in sfs], color=col,
                       size=6 if key != "cur" else 8)
        toa = iot.lora_time_on_air(p.pl, p.sf)
        pt.hline("dw", 400, color=RED, style="--", label="US915 400 ms dwell", label_pos=0.03)
        pt.scatter("now", [p.sf], [toa * 1e3], color=PURPLE, size=16, symbol="d")
        pl0 = 20 * np.log10(4 * np.pi * self.F / 3e8)
        mapl = p.ptx + 3 - np.array([iot.lora_sensitivity_dbm(s_) for s_ in sfs]) - p.margin
        dkm = 10 ** ((mapl - pl0) / (10 * p.n)) / 1e3
        pr = self.plot("rng")
        pr.line("r", sfs, dkm, color=NAVY, width=2.2, name=f"n = {p.n:.2f}, {p.ptx:.0f} dBm")
        pr.scatter("rm", sfs, dkm, color=NAVY, size=7)
        km = float(dkm[p.sf - 7])
        pr.scatter("now", [p.sf], [km], color=PURPLE, size=16, symbol="d")
        # RFID
        d = np.logspace(np.log10(0.3), np.log10(40), 300)
        lam = 3e8 / 915e6
        fwd = p.eirp + 2.15 - 3 - 20 * np.log10(4 * np.pi * d / lam)
        rev = iot.rfid_backscatter_dbm(d, p.eirp, 915e6)
        pf = self.plot("rfid")
        pf.line("f", d, fwd, color=NAVY, width=2.2, name="power reaching the tag")
        pf.line("b", d, rev, color=ORANGE, width=2.2, name="backscatter at the reader")
        pf.hline("ts", p.tag, color=NAVY, style="--", label=f"tag wakes at {p.tag:.1f} dBm",
                 label_pos=0.02)
        pf.hline("rs", -85, color=ORANGE, style="--", label="reader sensitivity ≈ −85 dBm",
                 label_pos=0.02)
        dmax = lam / (4 * np.pi) * 10 ** ((p.eirp + 2.15 - 3 - p.tag) / 20)
        pf.vline("dm", dmax, color=GREEN, style=":", label=f"read range {dmax:.1f} m",
                 label_pos=0.45)
        self.readout(toa=toa * 1e3, ph=3600 * 0.01 / toa, km=km, m=dmax)

    def story(self, p):
        r = self.r
        s = (f"<p><b>LoRa.</b> {p.pl} bytes at SF{p.sf} take {v(r.get('toa', 0), '.1f', 'ms')} on "
             f"air. Europe's 868 MHz rules allow a 1 % duty cycle: at most "
             f"{v(r.get('ph', 0), '.0f')} such packets an hour. With {p.ptx:.0f} dBm, a 3 dBi "
             f"gateway antenna and {p.margin:.0f} dB of fade margin the link reaches "
             f"{v(r.get('km', 0), '.2f', 'km')} (path-loss exponent {p.n:.2f}). Each SF step: "
             f"2.5 dB more budget, twice the air time.</p>"
             f"<p><b>Passive RFID.</b> The tag has no battery: it harvests the reader's carrier and "
             f"answers by switching its antenna's load (backscatter). The tag must collect "
             f"{v(p.tag, '.1f', 'dBm')} to wake up, so the read range is "
             f"{v(r.get('m', 0), '.1f', 'm')}, set by the forward link (blue); the backscatter "
             f"(orange) falls as 1/d⁴ but readers hear it far beyond that.</p>")
        return "<h3>Kilometres for bytes, metres for free</h3>" + s + keybox(
            "Low-power links are budgets of energy and time as much as of decibels: air time "
            "limits LoRa's capacity, harvested power limits RFID's range.")


# =============================================================================== the lab
LAB = st.Lab(31, "Wi-Fi, Bluetooth LE, 802.15.4, LoRa and RFID", chapter=22,
             chapter_title="Wi-Fi, Bluetooth and IoT",
             experiments=[PreambleSync, MACAggregation, RateAnomaly, MinstrelLive, BLEGFSK,
                          BLEBattery, Zigbee154, LoRaChirps, RangeBudgets])

if __name__ == "__main__":
    st.run(LAB)
