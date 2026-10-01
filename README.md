# Modern Digital Communications — From the Telegraph to 6G

A college-textbook-length book on communication engineering (≈1,300 pages, 25 chapters + 2 appendices)
with a companion library and 36 Python simulation labs.

**Read the book:** [`book/main.pdf`](book/main.pdf) — fully hyperlinked: clickable table of contents,
PDF bookmarks, cross-references, index, and a "Contents" link in every page footer.

| Part | Chapters |
|------|----------|
| I. Foundations | 1 The Story of Telecommunication · 2 Signals, Spectra and Systems · 3 Random Signals and Noise · 4 Analog Modulation and the Classic Radio · 5 Sampling, Quantization and Digital Telephony |
| II. DSP for Radio | 6 Digital Filters and Multirate Processing · 7 The Radio Transceiver and the Software-Defined Radio |
| III. Digital Transmission | 8 Baseband Transmission and Pulse Shaping · 9 Digital Modulation and Optimal Detection · 10 Synchronization · 11 The Wireless Channel · 12 Equalization |
| IV. Information and Coding | 13 Information Theory · 14 Classical Error-Control Codes · 15 Turbo, LDPC and Polar Codes · 16 Source Coding |
| V. Systems | 17 OFDM · 18 Spread Spectrum, CDMA and GNSS · 19 MIMO and Antenna Arrays · 20 Multiple Access and the Cellular Concept · 21 Cellular Generations: AMPS to 5G · 22 Wi-Fi, Bluetooth and IoT · 23 Satellite Communications · 24 Wireline and Optical · 25 The Road to 6G |
| Appendices | A Mathematical Reference · B Using the Companion Labs |

## Labs
`moderncomms-labs/` — start with [`labs/lab00_index.ipynb`](moderncomms-labs/labs/lab00_index.ipynb)
and the [labs README](moderncomms-labs/README.md). Every lab is a pre-executed Jupyter notebook
(viewable directly on GitHub) generated from a `.py` source; `commlib/` is the shared DSP/comms
library used by both the labs and the book's figures. Appendix B of the book is the full guide.

```bash
pip install -r moderncomms-labs/requirements.txt
jupyter lab moderncomms-labs/labs
```

## Building
- Book: `cd book && ./build.sh` (`./build.sh figs` regenerates every figure first). Needs a LaTeX
  distribution (TeX Live or MiKTeX) and Python with numpy/scipy/matplotlib.
- One chapter in isolation: `bash book/build_chapter.sh ch08`.
- Labs: `python moderncomms-labs/tests/build_notebooks.py`; tests: `python moderncomms-labs/tests/test_commlib.py`.

See [`CLAUDE.md`](CLAUDE.md) for project history, conventions and status.
