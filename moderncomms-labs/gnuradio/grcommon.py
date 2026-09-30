"""Shared helpers for the course's GNU Radio 3.10 hardware examples.

Every script builds its flowgraph from these pieces so the USRP B200 and the
software channel used by --sim are interchangeable. Scripts are plain Python
(the same code GNU Radio Companion generates) so you can read them top to bottom.
"""
import argparse
import os
import sys
import time

import numpy as np
from gnuradio import blocks, channels, gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def base_parser(desc, freq=915e6, rate=1e6, tx_gain=30.0, rx_gain=35.0):
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--args", default="type=b200", help="UHD device arguments")
    p.add_argument("--freq", type=float, default=freq, help="RF center frequency (Hz)")
    p.add_argument("--rate", type=float, default=rate, help="sample rate (S/s)")
    p.add_argument("--tx-gain", type=float, default=tx_gain, help="B200 TX gain 0-89.75 dB (keep low!)")
    p.add_argument("--rx-gain", type=float, default=rx_gain, help="B200 RX gain 0-76 dB")
    p.add_argument("--sim", action="store_true", help="replace the USRP with a software channel model")
    p.add_argument("--snr", type=float, default=20.0, help="--sim only: SNR per sample (dB)")
    p.add_argument("--cfo", type=float, default=0.0, help="injected carrier offset (Hz); hardware: RX LO offset")
    p.add_argument("--ppm", type=float, default=0.0, help="--sim only: sample-clock offset (ppm)")
    p.add_argument("--multipath", action="store_true", help="--sim only: add a 3-tap multipath channel")
    p.add_argument("--nogui", action="store_true", help="no Qt GUI; print statistics to the console")
    p.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = run until Ctrl-C/close)")
    return p


def usrp_source(a, antenna="RX2", dc_corr=True):
    from gnuradio import uhd
    src = uhd.usrp_source(a.args, uhd.stream_args(cpu_format="fc32", channels=[0]))
    src.set_samp_rate(a.rate)
    src.set_center_freq(uhd.tune_request(a.freq - a.cfo), 0)
    src.set_gain(a.rx_gain, 0)
    src.set_antenna(antenna, 0)
    src.set_auto_dc_offset(dc_corr, 0)
    src.set_auto_iq_balance(True, 0)
    return src


def usrp_sink(a, antenna="TX/RX"):
    from gnuradio import uhd
    snk = uhd.usrp_sink(a.args, uhd.stream_args(cpu_format="fc32", channels=[0]), "")
    snk.set_samp_rate(a.rate)
    snk.set_center_freq(uhd.tune_request(a.freq), 0)
    snk.set_gain(a.tx_gain, 0)
    snk.set_antenna(antenna, 0)
    return snk


def sim_channel(a, signal_power=1.0):
    """Software stand-in for TX -> cable/air -> RX: CFO, clock offset, multipath, AWGN."""
    taps = [1.0, 0.0, 0.35 + 0.25j, 0.0, -0.15] if a.multipath else [1.0]
    nv = float(np.sqrt(signal_power / 10 ** (a.snr / 10)))
    return channels.channel_model(noise_voltage=nv, frequency_offset=a.cfo / a.rate,
                                  epsilon=1.0 + a.ppm * 1e-6, taps=taps, noise_seed=42, block_tags=False)


def radio_link(tb, a, tx_out, rx_in, tx_power=1.0):
    """Connect a TX signal to an RX input through hardware or the simulated channel."""
    if a.sim:
        thr = blocks.throttle(gr.sizeof_gr_complex, a.rate, True)
        ch = sim_channel(a, tx_power)
        tb.connect(tx_out, thr, ch, rx_in)
    else:
        tb.connect(tx_out, usrp_sink(a))
        tb.connect(usrp_source(a), rx_in)


def lfsr_bytes(n, seed=0xACE1):
    """Deterministic pseudo-random payload both ends can regenerate."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n, dtype=np.uint8)


class BerCounter(gr.sync_block):
    """Measures BER of a received bit stream against a known periodic bit pattern.

    Alignment (and, for BPSK/QPSK without differential coding, polarity) is found by
    circular correlation over one period every `window` bits.
    """

    def __init__(self, ref_bits, window=200_000, label="BER"):
        gr.sync_block.__init__(self, name="ber_counter", in_sig=[np.uint8], out_sig=None)
        self.ref = np.asarray(ref_bits, dtype=np.int8)
        self.P = len(self.ref)
        self.buf = np.zeros(0, dtype=np.int8)
        self.window, self.label = window, label
        self.errors = self.bits = 0
        self.last = 0.0
        self.ref_fft = np.conj(np.fft.fft(1 - 2 * self.ref.astype(float)))

    def work(self, input_items, output_items):
        x = input_items[0].astype(np.int8) & 1
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= self.window:
            blk, self.buf = self.buf[:self.window], self.buf[self.window:]
            seg = blk[:self.P]
            c = np.fft.ifft(np.fft.fft(1 - 2 * seg.astype(float)) * self.ref_fft).real
            k = int(np.argmax(np.abs(c)))
            inv = c[k] < 0
            ref = np.roll(self.ref, k)             # seg[m] = ref[m - k]
            ref_t = np.tile(ref, len(blk) // self.P + 1)[:len(blk)]
            e = int(np.sum((blk ^ inv) != ref_t))
            self.errors += e
            self.bits += len(blk)
            now = time.time()
            if now - self.last > 1.0:
                print(f"[{self.label}] block BER {e/len(blk):.2e}   cumulative {self.errors/self.bits:.2e} "
                      f"({self.bits/1e6:.2f} Mbit)", flush=True)
                self.last = now
        return len(input_items[0])


def run(tb, a, gui_widgets=None, title="Modern Digital Communications"):
    """Start the flowgraph with or without a Qt window."""
    if a.nogui or not gui_widgets:
        tb.start()
        try:
            if a.duration > 0:
                time.sleep(a.duration)
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        tb.stop(); tb.wait()
        return
    from PyQt5 import Qt
    app = Qt.QApplication.instance() or Qt.QApplication(sys.argv)
    w = Qt.QWidget(); w.setWindowTitle(title)
    lay = Qt.QGridLayout(w)
    for i, wid in enumerate(gui_widgets):
        r, cspan = (i // 2, 1) if len(gui_widgets) > 1 else (0, 2)
        lay.addWidget(wid, r, i % 2, 1, cspan)
    w.resize(1200, 800); w.show()
    tb.start()
    if a.duration > 0:
        Qt.QTimer.singleShot(int(a.duration * 1000), w.close)
    app.aboutToQuit.connect(lambda: (tb.stop(), tb.wait()))
    app.exec_()


def qt(block):
    """Wrap a gr-qtgui sink's widget for layout."""
    import sip
    from PyQt5 import Qt
    return sip.wrapinstance(block.qwidget(), Qt.QWidget)


def slider(label, lo, hi, step, value, callback):
    from gnuradio.qtgui import Range, RangeWidget
    from PyQt5 import QtCore
    return RangeWidget(Range(lo, hi, step, value, 200), callback, label, "counter_slider", float,
                       QtCore.Qt.Horizontal)
