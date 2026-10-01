#!/usr/bin/env python3
"""Chapter 11 hardware lab: a correlation channel sounder with the B200.

TX: a Zadoff-Chu sequence of length N (default 255, root 7), transmitted periodically.
    Its periodic autocorrelation is a perfect impulse, so the receive correlation is
    directly the channel impulse response, one estimate per period.
RX: matched filter (time-reversed conjugate ZC) -> |.|^2 -> one vector per period
    -> exponential averaging -> power-delay profile (PDP) display and console report.

Delay resolution = 1 / rate (50 ns at 20 MS/s, i.e. ~15 m of path-length difference).
Maximum unambiguous excess delay = N / rate (12.75 us at 20 MS/s).

    python3 gr03_channel_sounder.py --sim --multipath          # 3-tap software channel
    python3 gr03_channel_sounder.py --rate 20e6 --freq 2.45e9   # antennas, indoors
    python3 gr03_channel_sounder.py --sim --nogui --duration 5 --out ../data/pdp.npy

Cabled loopback (with >= 30 dB attenuation) shows a single tap: that is your system's
own impulse response, including the B200's filters. Subtract it when you use antennas.
"""
import numpy as np
from gnuradio import blocks, gr
from gnuradio import filter as grfilter

import grcommon as gc


def zadoff_chu(u, N):
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1 + (N % 2 == 0) * -1) / N)


class PdpReporter(gr.sync_block):
    """Averages PDP vectors, aligns them to the strongest tap and reports the taps."""

    def __init__(self, N, rate, alpha=0.1, out=None):
        gr.sync_block.__init__(self, name="pdp_reporter", in_sig=[(np.float32, N)], out_sig=[(np.float32, N)])
        self.N, self.rate, self.alpha, self.out = N, rate, alpha, out
        self.avg = None
        self.count = 0

    def work(self, input_items, output_items):
        for v in input_items[0]:
            self.avg = v.copy() if self.avg is None else (1 - self.alpha) * self.avg + self.alpha * v
            self.count += 1
        k = int(np.argmax(self.avg))
        aligned = np.roll(self.avg, -k + self.N // 8)       # main tap at N/8 for display
        output_items[0][:] = aligned / (aligned.max() + 1e-20)
        if self.count % 2000 < len(input_items[0]):
            rel = 10 * np.log10(aligned / aligned.max() + 1e-20)
            taps = np.where(rel > -20)[0]
            desc = ", ".join(f"{(t - self.N // 8) / self.rate * 1e9:+.0f} ns: {rel[t]:.1f} dB" for t in taps[:8])
            tau = (np.arange(self.N) - self.N // 8) / self.rate
            w = aligned * (rel > -20)          # threshold out the noise floor
            p = w / w.sum()
            mean = np.sum(p * tau)
            rms = np.sqrt(np.sum(p * (tau - mean) ** 2))
            print(f"[PDP] taps within 20 dB: {desc}   RMS delay spread {rms * 1e9:.0f} ns", flush=True)
            if self.out:
                np.save(self.out, aligned)
        return len(output_items[0])


class Sounder(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "gr03 channel sounder")
        self.a = a
        N = a.length
        zc = zadoff_chu(a.root, N).astype(np.complex64) * a.amplitude
        src = blocks.vector_source_c(zc.tolist(), True)
        self.corr_in = blocks.multiply_const_cc(1.0)
        gc.radio_link(self, a, src, self.corr_in, tx_power=a.amplitude ** 2)
        mf = grfilter.fir_filter_ccc(1, np.conj(zc[::-1] / a.amplitude / N).tolist())
        mag = blocks.complex_to_mag_squared(1)
        s2v = blocks.stream_to_vector(gr.sizeof_float, N)
        self.rep = PdpReporter(N, a.rate, out=a.out)
        self.connect(self.corr_in, mf, mag, s2v, self.rep)
        self.pdp = self.rep
        if a.nogui:
            self.connect(self.rep, blocks.null_sink(gr.sizeof_float * N))

    def gui(self):
        from gnuradio import qtgui
        a, N = self.a, self.a.length
        us = 1e6 / a.rate
        vs = qtgui.vector_sink_f(N, -N // 8 * us, us, "Excess delay (us)", "Relative power (dB)",
                                 "Power-delay profile (averaged)", 1, None)
        vs.set_y_axis(-40, 3)
        log = blocks.nlog10_ff(10, N, 0)
        self.connect(self.rep, log, vs)
        return [gc.qt(vs)]


def main():
    p = gc.base_parser(__doc__, freq=915e6, rate=10e6, tx_gain=30, rx_gain=30)
    p.add_argument("--length", type=int, default=255, help="Zadoff-Chu length (odd)")
    p.add_argument("--root", type=int, default=7, help="ZC root, coprime with length")
    p.add_argument("--amplitude", type=float, default=0.5)
    p.add_argument("--out", default=None, help="save the averaged PDP to this .npy file")
    a = p.parse_args()
    tb = Sounder(a)
    gc.run(tb, a, None if a.nogui else tb.gui(), "gr03 - channel sounder")


if __name__ == "__main__":
    main()
