#!/usr/bin/env python3
"""Chapter 17 hardware lab: an 802.11a-like packet OFDM link through the B200.

Uses GNU Radio's reference OFDM transceiver (gr-digital ofdm_tx / ofdm_rx):
64-point FFT, 16-sample cyclic prefix, 48 data + 4 pilot subcarriers, Schmidl & Cox
synchronization, BPSK header, BPSK/QPSK/8-PSK payload and a CRC-32 per packet.

TX: packets [seq number | pseudo-random bytes] -> tagged stream -> ofdm_tx -> B200 (bursts)
RX: B200 -> ofdm_rx -> CRC-checked packets -> packet error rate and throughput report

    python3 gr04_ofdm_link.py --sim                                  # software channel
    python3 gr04_ofdm_link.py --sim --multipath --cfo 5000 --snr 18 --bps 3
    python3 gr04_ofdm_link.py --rate 2e6 --bps 2                     # hardware loopback
    python3 gr04_ofdm_link.py --sim --nogui --duration 8

Things to try (Chapter 17): raise --bps to 3 (8-PSK) and find the SNR where the
packet error rate collapses; add --multipath and note that the link still works
without an equalizer because the 16-sample CP (1.6 us at 10 MS/s) exceeds the channel
length; push --cfo past half a subcarrier spacing (rate/64/2) and watch sync fail.
"""
import time

import numpy as np
import pmt
from gnuradio import blocks, digital, gr, pdu

import grcommon as gc

TAG = "packet_len"


class PacketCounter(gr.basic_block):
    """Receives CRC-checked PDUs, checks content and sequence numbers, reports PER."""

    def __init__(self, payloads, sent_counter):
        gr.basic_block.__init__(self, name="packet_counter", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self.handle)
        self.payloads = payloads
        self.sent = sent_counter
        self.ok = self.bad = 0
        self.last_seq = None
        self.missing = 0
        self.t0 = time.time(); self.last = 0.0

    def handle(self, msg):
        data = np.array(pmt.u8vector_elements(pmt.cdr(msg)), dtype=np.uint8)
        seq = int(data[0]) if len(data) else -1
        if 0 <= seq < len(self.payloads) and np.array_equal(data, self.payloads[seq]):
            self.ok += 1
            if self.last_seq is not None:
                self.missing += (seq - self.last_seq - 1) % len(self.payloads)
            self.last_seq = seq
        else:
            self.bad += 1
        now = time.time()
        if now - self.last > 1.0:
            total = self.ok + self.missing + self.bad
            per = 1 - self.ok / max(total, 1)
            tput = self.ok * len(self.payloads[0]) * 8 / (now - self.t0) / 1e3
            print(f"[OFDM] received {self.ok} good packets, {self.missing} lost, {self.bad} corrupt "
                  f"-> PER {per:.3f}, goodput {tput:.1f} kb/s", flush=True)
            self.last = now


class OfdmLink(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "gr04 OFDM link")
        self.a = a
        n_pkt, L = 64, a.payload
        rng = np.random.default_rng(1234)
        self.payloads = [np.concatenate([[i], rng.integers(0, 256, L - 1)]).astype(np.uint8) for i in range(n_pkt)]
        stream = np.concatenate(self.payloads)

        # ---------------- transmitter
        src = blocks.vector_source_b(stream.tolist(), True)
        s2ts = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, L, TAG)
        tx = digital.ofdm_tx(fft_len=64, cp_len=16, packet_length_tag_key=TAG,
                             bps_header=1, bps_payload=a.bps, rolloff=0, scramble_bits=True)
        # ofdm_tx output has mean power ~150 (un-normalized IFFT); scale to the requested RMS
        amp = blocks.multiply_const_cc(a.amplitude / np.sqrt(150.0))
        # Zero-pad 160 samples after every packet: real bursts are separated in time, and the
        # gap lets the Schmidl & Cox detector reset between packets.
        gap = digital.burst_shaper_cc([], 0, 160, False, TAG)
        self.connect(src, s2ts, tx, gap, amp)

        # ---------------- receiver
        rx = digital.ofdm_rx(fft_len=64, cp_len=16, frame_length_tag_key="frame_len",
                             packet_length_tag_key=TAG, bps_header=1, bps_payload=a.bps,
                             scramble_bits=True)
        self.rx_in = blocks.multiply_const_cc(1.0)
        if a.sim:
            # Strip the TX 'packet_len' tags: over the air they would not exist, and if they
            # reach ofdm_rx they collide with the tags its header parser creates.
            gate = blocks.tag_gate(gr.sizeof_gr_complex, False)
            self.connect(amp, gate)
            gc.radio_link(self, a, gate, self.rx_in, tx_power=a.amplitude ** 2)
        else:
            from gnuradio import uhd
            snk = uhd.usrp_sink(a.args, uhd.stream_args(cpu_format="fc32", channels=[0]), TAG)
            snk.set_samp_rate(a.rate); snk.set_center_freq(a.freq, 0)
            snk.set_gain(a.tx_gain, 0); snk.set_antenna("TX/RX", 0)
            self.connect(amp, snk)
            self.connect(gc.usrp_source(a), self.rx_in)

        self.connect(self.rx_in, rx)
        to_pdu = pdu.tagged_stream_to_pdu(gr.types.byte_t, TAG)
        self.counter = PacketCounter(self.payloads, None)
        self.connect(rx, to_pdu)
        self.msg_connect(to_pdu, "pdus", self.counter, "pdus")

    def gui(self):
        from gnuradio import qtgui
        from gnuradio.fft import window
        fs = qtgui.freq_sink_c(1024, window.WIN_BLACKMAN_hARRIS, 0, self.a.rate, "Received OFDM spectrum", 1, None)
        ws = qtgui.waterfall_sink_c(1024, window.WIN_BLACKMAN_hARRIS, 0, self.a.rate, "Waterfall", 1, None)
        self.connect(self.rx_in, fs); self.connect(self.rx_in, ws)
        return [gc.qt(fs), gc.qt(ws)]


def main():
    p = gc.base_parser(__doc__, freq=915e6, rate=2e6, tx_gain=30, rx_gain=30)
    p.add_argument("--bps", type=int, default=2, choices=[1, 2, 3], help="payload bits/subcarrier: 1=BPSK 2=QPSK 3=8PSK")
    p.add_argument("--payload", type=int, default=96, help="payload bytes per packet")
    p.add_argument("--amplitude", type=float, default=0.2, help="TX RMS amplitude (OFDM PAPR ~10 dB: peaks reach ~3x RMS)")
    a = p.parse_args()
    tb = OfdmLink(a)
    gc.run(tb, a, None if a.nogui else tb.gui(), "gr04 - OFDM link")


if __name__ == "__main__":
    main()
