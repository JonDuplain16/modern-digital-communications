#!/usr/bin/env python3
"""Chapter 1 hardware lab: live spectrum/waterfall and IQ recording with the B200.

    python3 gr01_spectrum_iq_capture.py --freq 100e6 --rate 2e6            # FM band
    python3 gr01_spectrum_iq_capture.py --freq 915e6 --out ../data/capture.cfile --duration 5
    python3 gr01_spectrum_iq_capture.py --no-dc-corr                       # see the raw DC spur
    python3 gr01_spectrum_iq_capture.py --sim --nogui --duration 3 --out /tmp/x.cfile

Receive-only: this script never transmits. Recordings are complex64 (.cfile) with
a SigMF .sigmf-meta sidecar; load them with commlib.read_cfile (Lab 1, section 5).
"""
import numpy as np
from gnuradio import analog, blocks, gr

import grcommon as gc


class SpectrumCapture(gr.top_block):
    def __init__(self, a):
        gr.top_block.__init__(self, "gr01 spectrum and IQ capture")
        self.a = a
        if a.sim:
            # stand-in: two tones, a DC spur, an IQ-imbalance image and noise
            t1 = analog.sig_source_c(a.rate, analog.GR_COS_WAVE, 0.12 * a.rate, 0.3)
            t2 = analog.sig_source_c(a.rate, analog.GR_COS_WAVE, -0.31 * a.rate, 0.03)
            img = analog.sig_source_c(a.rate, analog.GR_COS_WAVE, -0.12 * a.rate, 0.3 * 10 ** (-30 / 20))
            dc = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0.0 if not a.no_dc_corr else 0.05)
            nz = analog.noise_source_c(analog.GR_GAUSSIAN, 10 ** (-a.snr / 20), 1)
            add = blocks.add_vcc(1)
            for i, s in enumerate([t1, t2, img, dc, nz]):
                self.connect(s, (add, i))
            self.src = blocks.throttle(gr.sizeof_gr_complex, a.rate, True)
            self.connect(add, self.src)
        else:
            self.src = gc.usrp_source(a, dc_corr=not a.no_dc_corr)
        if a.out:
            self.connect(self.src, blocks.head(gr.sizeof_gr_complex, int(a.rate * a.duration))
                         if a.duration > 0 else blocks.copy(gr.sizeof_gr_complex),
                         blocks.file_sink(gr.sizeof_gr_complex, a.out, False))
            from commlib.iq import write_sigmf_meta
            write_sigmf_meta(a.out.rsplit(".", 1)[0] + ".sigmf-meta", a.rate, a.freq,
                             "gr01 capture" + (" (simulated)" if a.sim else ""))
        # console power meter
        self.probe = blocks.probe_signal_f()
        self.connect(self.src, blocks.complex_to_mag_squared(),
                     blocks.moving_average_ff(int(a.rate // 10), 1.0 / int(a.rate // 10), 4000), self.probe)

    def set_freq(self, f):
        self.a.freq = f
        if not self.a.sim:
            from gnuradio import uhd
            self.src.set_center_freq(uhd.tune_request(f), 0)
        if hasattr(self, "fsink"):
            self.fsink.set_frequency_range(f, self.a.rate)
            self.wsink.set_frequency_range(f, self.a.rate)

    def set_gain(self, g):
        if not self.a.sim:
            self.src.set_gain(g, 0)

    def gui(self):
        from gnuradio import qtgui
        from gnuradio.fft import window
        a = self.a
        self.fsink = qtgui.freq_sink_c(4096, window.WIN_BLACKMAN_hARRIS, a.freq, a.rate, "Spectrum", 1, None)
        self.fsink.set_update_time(0.05); self.fsink.enable_grid(True); self.fsink.set_fft_average(0.2)
        self.wsink = qtgui.waterfall_sink_c(2048, window.WIN_BLACKMAN_hARRIS, a.freq, a.rate, "Waterfall", 1, None)
        self.connect(self.src, self.fsink); self.connect(self.src, self.wsink)
        return [gc.qt(self.fsink), gc.qt(self.wsink),
                gc.slider("Center frequency (Hz)", 70e6, 6e9, 100e3, a.freq, self.set_freq),
                gc.slider("RX gain (dB)", 0, 76, 1, a.rx_gain, self.set_gain)]


def main():
    p = gc.base_parser(__doc__, freq=915e6, rate=1e6)
    p.add_argument("--out", default="", help="record IQ to this .cfile (with --duration: fixed length)")
    p.add_argument("--no-dc-corr", action="store_true", help="disable UHD automatic DC offset correction")
    a = p.parse_args()
    tb = SpectrumCapture(a)
    widgets = None if a.nogui else tb.gui()
    if a.nogui:
        import threading, time
        def meter():
            while True:
                time.sleep(1.0)
                print(f"mean power {10*np.log10(tb.probe.level()+1e-20):6.1f} dBFS", flush=True)
        threading.Thread(target=meter, daemon=True).start()
    gc.run(tb, a, widgets, "gr01 - spectrum and IQ capture")


if __name__ == "__main__":
    main()
