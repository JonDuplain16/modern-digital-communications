# Modern Digital Communications — From the Telegraph to 6G (Second Edition)

A college-textbook-length book on communication engineering, written to be an enjoyable read:
25 chapters and 2 appendices told through stories, analogies, real photographs and a figure on
almost every page — with all the mathematics still there for those who want it — plus 36
interactive simulation labs.

**Read the book:** [`book/main.pdf`](book/main.pdf) — fully hyperlinked: clickable table of contents,
PDF bookmarks, cross-references, index, references and image credits, and a "Contents" link in
every page footer.

| Part | Chapters |
|------|----------|
| I. Foundations | 1 The Story of Telecommunication · 2 Signals, Spectra and Systems · 3 Random Signals and Noise · 4 Analog Modulation and the Classic Radio · 5 Sampling, Quantization and Digital Telephony |
| II. DSP for Radio | 6 Digital Filters and Multirate Processing · 7 The Radio Transceiver and the Software-Defined Radio |
| III. Digital Transmission | 8 Baseband Transmission and Pulse Shaping · 9 Digital Modulation and Optimal Detection · 10 Synchronization · 11 The Wireless Channel · 12 Equalization |
| IV. Information and Coding | 13 Information Theory · 14 Classical Error-Control Codes · 15 Turbo, LDPC and Polar Codes · 16 Source Coding |
| V. Systems | 17 OFDM · 18 Spread Spectrum, CDMA and GNSS · 19 MIMO and Antenna Arrays · 20 Multiple Access and the Cellular Concept · 21 Cellular Generations: AMPS to 5G · 22 Wi-Fi, Bluetooth and IoT · 23 Satellite Communications · 24 Wireline and Optical · 25 The Road to 6G |
| Appendices | A Mathematical Reference · B Using the Companion Labs |

## Interactive labs
`moderncomms-labs/` holds 36 live desktop labs (PySide6 + pyqtgraph), all with the same look:
choose an experiment, drag sliders, and plots, readouts and a plain-language "What's going on"
panel update instantly; "Try this" challenges tick themselves off as you reach them.

```bash
pip install -r moderncomms-labs/requirements.txt
python moderncomms-labs/labs/launcher.py                 # gallery of all labs
python moderncomms-labs/labs/lab03_pulse_shaping.py      # or run one lab directly (--exp N, --dark)
```

See the [labs README](moderncomms-labs/README.md) and Appendix B of the book. `commlib/` is the
shared DSP/comms library used by both the labs and the book's figures.

## Building
- Book: `cd book && ./build.sh` (`./build.sh figs` regenerates every figure first). Needs TeX Live
  or MiKTeX and Python with numpy/scipy/matplotlib.
- One chapter in isolation: `bash book/build_chapter.sh ch08`; visual coverage:
  `python book/tools/check_visuals.py book/_chapbuild/ch08_only.pdf`.
- Labs: `python moderncomms-labs/tests/selftest_labs.py` (add `--dark`); library tests:
  `python moderncomms-labs/tests/test_commlib.py`.

Photographs are from Wikimedia Commons under free licenses (see the book's Image Credits).
See [`CLAUDE.md`](CLAUDE.md) for project history, conventions and status.
