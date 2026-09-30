#!/usr/bin/env python3
"""Chapters 4 and 6 hardware lab: a complete QPSK link through the B200.

TX: known pseudo-random bytes -> differential QPSK -> RRC (beta 0.35) -> B200 TX/RX port
RX: B200 RX2 port -> AGC -> FLL band-edge (coarse CFO) -> polyphase clock sync (timing)
    -> CMA equalizer -> Costas loop (fine phase) -> slicer -> differential decode -> BER

Cabled loopback: TX/RX -> 30 dB (or more) attenuator -> RX2.   NEVER connect without attenuation.

    python3 gr02_psk_link.py --sim                          # software channel first
    python3 gr02_psk_link.py --sim --cfo 2000 --multipath --ppm 50 --snr 15
    python3 gr02_psk_link.py --cfo 2000                     # hardware, 2 kHz injected offset
    python3 gr02_psk_link.py --sim --nogui --duration 10    # headless BER measurement

Differential encoding removes the Costas loop's 90-degree phase ambiguity at a cost of
roughly 2x the bit errors near threshold (Chapter 4). The GUI shows the constellation
after each receiver stage so you can see what every block contributes.
"""
import numpy as np
from gnuradio import analog, blocks, digital, gr
from gnuradio import filter as grfilter
from gnuradio.filter import firdes

import grcommon as gc


def qpsk_rotational():
    """QPSK with points in 90-degree order so that differential coding cancels rotations."""
    pts = [0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j]
    return digital.constellation_rect(pts, [0, 1, 2, 3], 4, 2, 2, 1, 1).base()


class PskLink(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "gr02 QPSK link")
        self.a = a
        sps, nfilts, beta = a.sps, 32, 0.35
        const = qpsk_rotational()
        payload = gc.lfsr_bytes(1024)
        self.ref_bits = np.unpackbits(payload)

        # ---------------- transmitter
        src = blocks.vector_source_b(payload.tolist(), True)
        mod = digital.generic_mod(constellation=const, differential=True, samples_per_symbol=sps,
                                  pre_diff_code=True, excess_bw=beta, verbose=False, log=False, truncate=False)
        amp = blocks.multiply_const_cc(a.amplitude)
        self.connect(src, mod, amp)

        # ---------------- receiver
        self.agc = analog.agc_cc(1e-3, 1.0, 1.0); self.agc.set_max_gain(65536)
        gc.radio_link(self, a, amp, self.agc, tx_power=a.amplitude ** 2 / sps)
        self.fll = digital.fll_band_edge_cc(sps, beta, 44, 2 * np.pi / 100 * a.fll_bw)
        rrc = firdes.root_raised_cosine(nfilts, nfilts, 1.0 / sps, beta, 11 * sps * nfilts)
        self.clock = digital.pfb_clock_sync_ccf(sps, 2 * np.pi / 100 * a.timing_bw, rrc, nfilts, nfilts / 2, 1.5, 2)
        alg = digital.adaptive_algorithm_cma(const, a.cma_mu, 1.0).base()
        self.eq = digital.linear_equalizer(15, 2, alg, True, [], "")
        self.costas = digital.costas_loop_cc(2 * np.pi / 100 * a.phase_bw, 4, False)
        dec = digital.constellation_decoder_cb(const)
        diff = digital.diff_decoder_bb(4)
        unmap = digital.map_bb([0, 1, 2, 3])
        unpack = blocks.unpack_k_bits_bb(2)
        self.ber = gc.BerCounter(self.ref_bits, label="QPSK BER")
        self.connect(self.agc, self.fll, self.clock, self.eq, self.costas, dec, diff, unmap, unpack, self.ber)

    def gui(self):
        from gnuradio import qtgui
        from gnuradio.fft import window
        a = self.a
        cs = qtgui.const_sink_c(1024, "Constellation after each stage", 4, None)
        cs.set_update_time(0.1); cs.enable_grid(True)
        for i, lab in enumerate(["AGC input (raw)", "clock sync (2 sps)", "CMA equalizer", "Costas loop"]):
            cs.set_line_label(i, lab)
        cs.set_x_axis(-2, 2); cs.set_y_axis(-2, 2)
        keep = blocks.keep_one_in_n(gr.sizeof_gr_complex, a.sps)
        self.connect(self.agc, keep, (cs, 0))
        self.connect(self.clock, (cs, 1)); self.connect(self.eq, (cs, 2)); self.connect(self.costas, (cs, 3))
        fs = qtgui.freq_sink_c(2048, window.WIN_BLACKMAN_hARRIS, 0, a.rate, "RX spectrum before / after FLL", 2, None)
        fs.set_line_label(0, "before FLL"); fs.set_line_label(1, "after FLL")
        self.connect(self.agc, (fs, 0)); self.connect(self.fll, (fs, 1))
        ts = qtgui.time_sink_f(500, 1.0, "Costas loop frequency estimate (rad/sample)", 1, None)
        fq = blocks.probe_signal_f()
        self.connect((self.costas, 1), ts)
        return [gc.qt(cs), gc.qt(fs), gc.qt(ts),
                gc.slider("Costas loop bandwidth (x 2pi/100)", 0.1, 5, 0.1, a.phase_bw,
                          lambda v: self.costas.set_loop_bandwidth(2 * np.pi / 100 * v))]


def main():
    p = gc.base_parser(__doc__, freq=915e6, rate=1e6, tx_gain=30, rx_gain=30)
    p.add_argument("--sps", type=int, default=4, help="samples per symbol")
    p.add_argument("--amplitude", type=float, default=0.5, help="digital TX amplitude (0-1)")
    p.add_argument("--fll-bw", type=float, default=0.5, help="FLL loop bandwidth (x 2pi/100)")
    p.add_argument("--timing-bw", type=float, default=1.0, help="clock sync loop bandwidth (x 2pi/100)")
    p.add_argument("--phase-bw", type=float, default=1.0, help="Costas loop bandwidth (x 2pi/100)")
    p.add_argument("--cma-mu", type=float, default=1e-3, help="CMA step size")
    a = p.parse_args()
    tb = PskLink(a)
    gc.run(tb, a, None if a.nogui else tb.gui(), "gr02 - QPSK link")


if __name__ == "__main__":
    main()
