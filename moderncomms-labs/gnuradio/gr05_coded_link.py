#!/usr/bin/env python3
"""Chapter 8 hardware lab: measure real coding gain with a K=7 convolutional code.

TX: frames = [63-symbol BPSK m-sequence preamble | 22 blocks of (16 pilots + <=48 data)]
    carrying 1012 coded BPSK symbols (500 info bits + 6 tail bits through the (133,171)
    rate-1/2 code from commlib), RRC beta 0.35.
RX: AGC -> FLL band-edge (coarse CFO) -> polyphase clock sync (1 sample/symbol)
    -> FrameDecoder: preamble correlation, pilot-aided phase tracking (interpolated between
       pilot blocks), Es/N0 estimate, raw channel BER, soft-decision Viterbi (commlib.ConvCode).

Why pilots instead of a Costas loop? The code works at Es/N0 near 0 dB, where a
decision-directed loop cycle-slips constantly. Pilot-aided estimation is what DVB-S2,
5G NR (DM-RS/PT-RS) and satellite modems use at low SNR: see Chapter 4. Try
--pilot-spacing to trade overhead against tracking.

The console shows, per second: estimated Es/N0, uncoded (channel) BER and decoded BER,
so you can place your hardware operating point on the curves from Lab 8.

    python3 gr05_coded_link.py --sim --snr -1        # near the waterfall of the code
    python3 gr05_coded_link.py --sim --snr 2 --cfo 500
    python3 gr05_coded_link.py --tx-gain 10 --rx-gain 20     # hardware; lower gains to add errors

In --sim, --snr is SNR per sample; with 4 samples/symbol Es/N0 = SNR + 6 dB. Each BPSK
symbol carries one coded bit = 1/2 information bit, so Eb/N0 = Es/N0 - 10 log10(1/2) = Es/N0 + 3 dB.
"""
import time

import numpy as np
from gnuradio import analog, blocks, digital, gr
from gnuradio.filter import firdes

import grcommon as gc
import commlib as cl

N_INFO = 500


def mseq63():
    """Length-63 maximal-length sequence (x^6 + x + 1) as +-1 values."""
    reg = [1, 0, 0, 0, 0, 0]
    out = []
    for _ in range(63):
        out.append(reg[-1])
        fb = reg[5] ^ reg[0]
        reg = [fb] + reg[:-1]
    return 1.0 - 2.0 * np.array(out)


N_PILOT = 16


def build_frame(preamble, coded, block_data):
    """Return (frame symbols, data index array, list of (pilot index array, pilot values))."""
    pil = preamble[:N_PILOT]
    data = 1.0 - 2.0 * coded
    sym, data_idx, pilots = list(preamble), [], [(np.arange(len(preamble)), preamble)]
    for b in range(0, len(data), block_data):
        pos = len(sym)
        pilots.append((np.arange(pos, pos + N_PILOT), pil))
        sym.extend(pil)
        chunk = data[b:b + block_data]
        data_idx.extend(range(len(sym), len(sym) + len(chunk)))
        sym.extend(chunk)
    return np.array(sym), np.array(data_idx), pilots


class FrameDecoder(gr.sync_block):
    def __init__(self, preamble, info_bits, coded_bits, code, frame, data_idx, pilots):
        gr.sync_block.__init__(self, name="frame_decoder", in_sig=[np.complex64], out_sig=None)
        self.pre = preamble.astype(complex)
        self.info, self.coded, self.code = info_bits, coded_bits, code
        self.data_idx, self.pilots = data_idx, pilots
        self.flen = len(frame)
        self.buf = np.zeros(0, dtype=complex)
        self.frames = self.raw_err = self.raw_bits = self.dec_err = self.dec_bits = self.fer = 0
        self.esn0 = []
        self.warmup = 40
        self.last = time.time()

    def work(self, input_items, output_items):
        self.buf = np.concatenate([self.buf, input_items[0]])
        P = len(self.pre)
        while len(self.buf) >= 2 * self.flen:
            seg = self.buf[:self.flen + P]
            c = np.correlate(seg, self.pre, mode="valid")
            k = int(np.argmax(np.abs(c)))
            peak = np.abs(c[k]) / (np.sqrt(np.sum(np.abs(seg[k:k + P]) ** 2) * P) + 1e-12)
            if peak < 0.6:                             # no preamble here: slide on
                self.buf = self.buf[self.flen:]
                continue
            frame = self.buf[k:k + self.flen]
            self.buf = self.buf[k + self.flen:]
            # pilot-aided phase: one estimate per pilot group, unwrapped, linearly interpolated
            centers, phases = [], []
            for idx, val in self.pilots:
                centers.append(idx.mean())
                phases.append(np.angle(np.sum(frame[idx] * np.conj(val))))
            ph = np.interp(np.arange(self.flen), centers, np.unwrap(phases))
            y = (frame * np.exp(-1j * ph))[self.data_idx].real
            amp = np.mean(np.abs(y))
            y = y / (amp + 1e-12)
            s = 1.0 - 2.0 * self.coded
            n0 = 2 * np.mean((y - s) ** 2) + 1e-9      # complex-equivalent noise variance
            self.esn0.append(10 * np.log10(1.0 / n0))
            self.raw_err += int(np.sum((y < 0) != self.coded.astype(bool)))
            self.raw_bits += len(y)
            uh = self.code.decode(4 * y / n0)          # LLR = 2y/sigma^2 with sigma^2 = n0/2
            e = int(np.sum(uh != self.info))
            self.dec_err += e; self.dec_bits += len(uh); self.fer += e > 0
            self.frames += 1
            if self.frames == self.warmup:              # discard frames received while loops acquire
                self.raw_err = self.raw_bits = self.dec_err = self.dec_bits = self.fer = 0
        now = time.time()
        if now - self.last > 1.0 and self.frames:
            es = np.median(self.esn0[-50:])
            print(f"[coded link] frames {self.frames:5d}  Es/N0 ~ {es:5.1f} dB (Eb/N0 ~ {es + 3:4.1f})  "
                  f"raw BER {self.raw_err / max(self.raw_bits, 1):.2e}  "
                  f"decoded BER {self.dec_err / max(self.dec_bits, 1):.2e}  FER {self.fer / max(self.frames - self.warmup, 1):.3f}",
                  flush=True)
            self.last = now
        return len(input_items[0])


class CodedLink(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "gr05 coded link")
        self.a = a
        sps, beta, nfilts = a.sps, 0.35, 32
        code = cl.ConvCode()
        rng = np.random.default_rng(55)
        info = rng.integers(0, 2, N_INFO).astype(np.int8)
        coded = code.encode(info)
        pre = mseq63()
        frame, data_idx, pilots = build_frame(pre, coded, a.pilot_spacing)
        frame = frame.astype(complex)
        h = cl.rrc_taps(beta, sps, 12)
        reps = 8                                       # periodic waveform: shape a few frames, drop edges
        wave = cl.shape(np.tile(frame, reps), h, sps)
        L = len(frame) * sps
        start = len(h) // 2 + 2 * L
        one_period = wave[start:start + L]             # a steady-state period of the periodic signal
        src = blocks.vector_source_c((a.amplitude * one_period).astype(np.complex64).tolist(), True)

        self.agc = analog.agc_cc(1e-3, 1.0, 1.0); self.agc.set_max_gain(65536)
        gc.radio_link(self, a, src, self.agc, tx_power=a.amplitude ** 2 / sps)
        fll = digital.fll_band_edge_cc(sps, beta, 44, 2 * np.pi / 100 * a.fll_bw)
        rrc = firdes.root_raised_cosine(nfilts, nfilts, 1.0 / sps, beta, 11 * sps * nfilts)
        self.clock = digital.pfb_clock_sync_ccf(sps, 2 * np.pi / 100 * a.timing_bw, rrc, nfilts, nfilts / 2, 1.5, 1)
        self.dec = FrameDecoder(pre, info, coded, code, frame, data_idx, pilots)
        self.connect(self.agc, fll, self.clock, self.dec)

    def gui(self):
        from gnuradio import qtgui
        cs = qtgui.const_sink_c(1024, "Symbols after clock sync (before pilot phase correction)", 1, None)
        cs.set_x_axis(-2, 2); cs.set_y_axis(-2, 2)
        self.connect(self.clock, cs)
        return [gc.qt(cs)]


def main():
    p = gc.base_parser(__doc__, freq=915e6, rate=250e3, tx_gain=20, rx_gain=25)
    p.add_argument("--sps", type=int, default=4, help="samples per symbol")
    p.add_argument("--amplitude", type=float, default=0.5, help="digital TX amplitude (0-1)")
    p.add_argument("--timing-bw", type=float, default=0.3, help="clock sync loop bandwidth (x 2pi/100)")
    p.add_argument("--fll-bw", type=float, default=0.05, help="FLL loop bandwidth (x 2pi/100); narrow = less frequency jitter")
    p.add_argument("--pilot-spacing", type=int, default=48, help="data symbols between 16-symbol pilot groups")
    a = p.parse_args()
    tb = CodedLink(a)
    gc.run(tb, a, None if a.nogui else tb.gui(), "gr05 coded BPSK link")


if __name__ == "__main__":
    main()
