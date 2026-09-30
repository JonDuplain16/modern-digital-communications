# Modern Digital Communications — Labs and SDR Examples

Companion code for the course text *Modern Digital Communications: A Hands-On Course
with Python and SDR*. Twelve interactive Jupyter labs (one per chapter), a small shared
DSP library, and five GNU Radio 3.10 flowgraphs for the Ettus USRP B200.

```
commlib/     shared library: modulation, filters, channels, sync, equalizers,
             OFDM, coding (Viterbi, LDPC, polar), MIMO, IQ file I/O
labs/        labNN_*.ipynb (pre-executed, with figures) + labNN_*.py sources (jupytext)
gnuradio/    grNN_*.py hardware flowgraphs + grcommon.py helpers
tests/       test_commlib.py, build_notebooks.py (rebuild/execute all labs)
data/        IQ captures from gr01 land here
```

## 1. Labs

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab labs/
```
Open a notebook and choose *Run All*. Each `interact(...)` panel becomes live sliders.
The notebooks ship pre-executed, so you can read every figure before running anything.

| Lab | Topic | Interactive highlights |
|-----|-------|------------------------|
| 01 | Complex baseband, IQ, SDR receive chain | IQ imbalance image, sensitivity / link budget |
| 02 | Signal space and modulation | Any constellation vs Eb/N0, Gray vs natural, LLRs |
| 03 | Pulse shaping, Nyquist, matched filter | Roll-off vs spectrum and eye |
| 04 | Synchronization | PLL bandwidth, Gardner timing, ZC frame sync, full burst RX |
| 05 | Wireless channels | Doppler spectrum, TDL profiles, fading BER, channel sounding |
| 06 | Equalization | ZF/MMSE/LMS/CMA/DFE on ISI channels |
| 07 | OFDM | CP, CFO/ICI, Schmidl-Cox, pilot estimation, PAPR, NR numerology |
| 08 | Information theory, convolutional codes | Shannon limit, Hamming, Viterbi hard/soft, puncturing |
| 09 | LDPC and polar codes | BP vs min-sum, polarization, SC vs CA-SCL |
| 10 | MIMO | Diversity slope, capacity, ZF/MMSE/SIC/ML, beamforming, massive MIMO |
| 11 | Air interfaces (NR, Wi-Fi 7) | OFDMA schedulers, DFT-s-OFDM PAPR, MCS/HARQ, 4096-QAM EVM |
| 12 | Frontiers | OTFS vs OFDM, OFDM radar map, learned demapper |

## 2. GNU Radio hardware examples

Tested with GNU Radio 3.10.9 / UHD 4 on Ubuntu 24.04. Every script runs **without hardware**
first: `--sim` replaces the B200 with `channels.channel_model` (CFO, clock offset,
multipath, AWGN), and `--nogui` prints statistics instead of opening Qt windows.

| Script | Chapter | What it shows | Try |
|--------|---------|---------------|-----|
| `gr01_spectrum_iq_capture.py` | 1 | Live spectrum/waterfall; records `.cfile` + SigMF | `--freq 100e6 --rate 2e6 --out ../data/capture.cfile` |
| `gr02_psk_link.py` | 4, 6 | QPSK link: AGC, FLL, polyphase clock sync, CMA, Costas, live BER | `--sim --cfo 2000 --multipath --ppm 50 --snr 15` |
| `gr03_channel_sounder.py` | 5 | Zadoff-Chu sounder: power-delay profile, RMS delay spread | `--sim --multipath` |
| `gr04_ofdm_link.py` | 7 | 802.11a-like packet OFDM: PER and goodput | `--sim --multipath --cfo 5000 --snr 20` |
| `gr05_coded_link.py` | 8 | Framed BPSK + K=7 Viterbi (commlib): raw vs decoded BER | `--sim --snr -3` |

Results measured in simulation while preparing this release: gr02 converged to BER < 1e-4
with 2 kHz CFO at 18 dB; gr03 recovered the 3-tap profile (88 ns RMS delay spread);
gr04 delivered 100% of packets at 20 dB and ~99% with multipath + 5 kHz CFO;
gr05 at Es/N0 = 2.9 dB measured raw BER 2.4e-2 (theory 2.3e-2) and decoded BER 0.

### Hardware wiring (B200, cabled loopback)
`TX/RX` → **30 dB attenuator (60 dB is safer)** → `RX2`. Never connect TX to RX directly.
Default frequency is 915 MHz (ISM, ITU Region 2); in Region 1 use `--freq 433.92e6` or `868e6`.
A single B200 shares one reference for TX and RX, so loopback CFO is ~0: use `--cfo` to inject
an offset (implemented as an RX LO offset). Start with low `--tx-gain` and raise it slowly.

### Installing GNU Radio
* **Recommended:** [radioconda](https://github.com/ryanvolz/radioconda) (GNU Radio + UHD + NumPy
  in one environment), then `pip install -r requirements.txt` inside it.
* **Ubuntu/Debian:** `sudo apt install gnuradio uhd-host && sudo uhd_images_downloader`.
  Distribution GNU Radio packages are built against the system NumPy 1.x. If `pip` has
  installed NumPy 2.x into the same Python, `from gnuradio import gr` fails with an
  "initialization failed" / NumPy ABI error. Fix: run the flowgraphs with the system
  packages first, e.g. `PYTHONPATH=/usr/lib/python3/dist-packages python3 gr02_psk_link.py`,
  or keep the labs in a separate virtual environment.
* Check the radio: `uhd_find_devices` and `uhd_usrp_probe`.

## 3. Tests and rebuilding

```bash
python tests/test_commlib.py                 # library self-test (BER vs theory, loops lock, codes decode)
python tests/build_notebooks.py              # regenerate + execute every lab from labs/*.py
python tests/build_notebooks.py lab07        # just one
```
Edit the `labs/*.py` sources (jupytext percent format) and rebuild, or edit the notebooks directly.

## License
Course material for teaching use. commlib and the flowgraphs may be reused freely with attribution.
