# Modern Digital Communications — Lab Studio and SDR Examples

Companion code for the textbook *Modern Digital Communications*: thirty-six **interactive
desktop labs** (the *Lab Studio*), the shared DSP library they are built on, and five GNU Radio
3.10 flowgraphs for the Ettus USRP B200. **Start with the launcher: `python labs/launcher.py`.**

```
commlib/     shared DSP/comms library used by the labs AND the book figures: modulation, filters,
             channels, sync, equalizers, OFDM, coding (Viterbi, LDPC, polar, block codes, GF, turbo),
             MIMO, satellite links, line codes, information theory, CPM/EVM, propagation, source
             coding, spread spectrum/GNSS, cellular, an LTE/NR PHY, wireline/optical, Wi-Fi/BLE/LoRa,
             6G waveforms, RF transceiver models
studio/      the Lab Studio framework (PySide6 + pyqtgraph): window, controls, live plots, self-test,
             launcher. API: studio/README.md
labs/        labNN_*.py: one studio app per lab, plus launcher.py (the gallery)
gnuradio/    grNN_*.py hardware flowgraphs + grcommon.py helpers
tests/       test_commlib.py (library), selftest_labs.py (every lab, headless, with screenshots)
data/        IQ captures from gr01 land here
LAB_STYLE_GUIDE.md   how to write or edit a lab
```

## 1. Installing and running the labs

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # numpy, scipy, matplotlib, PySide6, pyqtgraph
python labs/launcher.py            # the Lab Studio gallery (same as: python -m studio)
```

The launcher lists every lab grouped by the book's parts, with a search box; **Launch** opens a
lab in its own window (you can keep several open). To run one lab directly:

```bash
python labs/lab03_pulse_shaping.py              # open the lab
python labs/lab03_pulse_shaping.py --exp 4      # open on experiment 4
python labs/lab03_pulse_shaping.py --dark       # dark theme (remembered)
```

Every lab looks and works the same way: the **experiments** of the lab on the left with their
**controls** (sliders, menus, switches, text fields; double-click a slider to reset it, or type
a value into its box); **live plots** and big **readouts** in the middle that update as you move
a control; on the right, **What's going on** (a short explanation tied to your current settings),
**Try this** challenges that tick themselves when you achieve them, and a **Read more in the book**
link that opens the PDF at the matching section. Streaming experiments have a **Play** button.
Keyboard: Ctrl+1…9 experiments, PgUp/PgDn, Ctrl+P play/pause, Ctrl+R reset, Ctrl+E export PNG,
Ctrl+D dark/light, F1 hide/show the notes. Some labs can play audio (optional `sounddevice`;
they fall back to the operating system's player).

| Lab | Chapter(s) | Topic and experiments | File |
|-----|-----------|-----------------------|------|
| 34 | 1 | **The Telegraph and the Birth of Signalling** — Morse keying, Morse vs Huffman, Kelvin's cable, semaphore pipeline, regenerators, loading coils, capacity | `labs/lab34_telegraph_history.py` |
| 01 | 2, 7 | **Complex Baseband and IQ Sampling** — The IQ helix, real vs IQ sampling, up/down-conversion, the DDC, zero-IF images, phase noise, ADC bits, sensitivity, a live waterfall | `labs/lab01_baseband.py` |
| 35 | 2 | **Fourier, Spectra and Bandwidth** — Fourier series and Gibbs, windows and scalloping, zero padding, uncertainty, convolution in slow motion, group delay, five bandwidths, STFT, IP3 | `labs/lab35_fourier_spectra.py` |
| 33 | 3 | **Noise and Detection** — Why noise is Gaussian, Q-function tails, Rayleigh/Rice, filtered and bandpass noise, Friis, Y-factor, ROC curves, signals below the noise | `labs/lab33_noise_detection.py` |
| 14 | 4 | **Analog Modulation: AM, SSB, FM and Stereo** — Envelope detector, DSB-SC and SSB, FM in time, Bessel sidebands and Carson, FM threshold and clicks, stereo multiplex (with audio) | `labs/lab14_analog_am_fm.py` |
| 15 | 4, 7 | **The Superheterodyne Receiver** — Tune the band, images, IF selectivity, AGC, double conversion, IM3/SFDR, zero-IF vs low-IF | `labs/lab15_superhet_receiver.py` |
| 16 | 5 | **Sampling, PCM and Companding** — Aliasing, ZOH, SQNR, dither, G.711, DM/CVSD, sigma-delta, jitter, T1/E1 | `labs/lab16_pcm_companding.py` |
| 17 | 6 | **Digital Filters and Multirate DSP** — Pole–zero, FIR/IIR and fixed point, aliasing, PFB channelizer, CIC, NCO spurs, CORDIC, a live DDC | `labs/lab17_multirate_dsp.py` |
| 36 | 7 | **The RF Transceiver** — Two-tone IIP3, line-up and SFDR, reciprocal mixing, I/Q image, PA ACLR/EVM, Doherty, CFR, DPD | `labs/lab36_rf_transceiver.py` |
| 03 | 8 | **Pulse Shaping, Nyquist and the Eye** — Pulses and spectra, zero ISI, live eye diagrams, the matched filter, TX/RX chains, BER Monte Carlo, NRZ vs PAM-4, faster than Nyquist | `labs/lab03_pulse_shaping.py` |
| 13 | 8 | **Line Codes, Jitter and SerDes Eyes** — Typed line codes and PSDs, baseline wander, scramblers, 8b/10b, jitter bathtub, PAM-4 with CTLE/FFE/DFE, duobinary | `labs/lab13_line_codes_eyes.py` |
| 02 | 9 | **Digital Modulation and Optimal Detection** — Constellation zoo, signal space, decision regions and MAP, BER Monte Carlo, LLRs, noncoherent FSK, Shannon plane, Rayleigh fading | `labs/lab02_modulation.py` |
| 21 | 9 | **Constant Envelope, PAPR and EVM** — CPM phase and spectra, MSK = OQPSK (Laurent), three receivers, PAPR, PA regrowth, EVM signatures and budget | `labs/lab21_cpm_evm.py` |
| 04 | 10 | **Synchronization: Carrier, Timing and Frame** — Impairment diagnosis, PLL steps and slips, S-curves, live Costas and timing loops, frequency estimators vs CRB, Farrow, frame sync, Schmidl–Cox and PSS cell search | `labs/lab04_synchronization.py` |
| 05 | 11 | **The Wireless Channel** — Path loss and shadowing, room standing waves, Doppler speckle, Rayleigh/Rice/Nakagami, echoes as notches, 3GPP TDL models, diversity, channel sounding | `labs/lab05_channels.py` |
| 22 | 11 | **Propagation and Link Budgets** — Friis, the decibel bank account, Fresnel zones and knife edges, two-ray, Hata/COST-231/38.901, rain and oxygen, edge vs area coverage, a coverage map | `labs/lab22_propagation_link_budget.py` |
| 06 | 12 | **Equalization** — ISI, ZF vs MMSE, length and delay, live LMS, LMS vs RLS, blind CMA, DFE error bursts, the Viterbi equalizer, SerDes CTLE/FFE/DFE | `labs/lab06_equalization.py` |
| 19 | 13 | **Information Theory** — Entropy of your own text, live Huffman trees, Blahut–Arimoto, Shannon's limit, real constellations, short packets, water-filling and rate–distortion | `labs/lab19_information_theory.py` |
| 08 | 14 | **Convolutional Codes and the Viterbi Algorithm** — Shannon's gap, shift-register encoder, trellis and d_free, Viterbi step by step, soft vs hard, puncturing, traceback, bursts and interleaving | `labs/lab08_convolutional.py` |
| 20 | 14 | **Block Codes: Hamming to Reed–Solomon** — Hamming Venn diagram, SECDED, CRC calculator and detection, GF(2^m), Reed–Solomon errors/erasures, interleaving, coding gain | `labs/lab20_block_codes.py` |
| 09 | 15 | **LDPC and Polar Codes** — Tanner graphs and girth, live belief propagation, min-sum variants, early stopping, polarization, SC vs CA-SCL, finite-length limits | `labs/lab09_ldpc_polar.py` |
| 23 | 15 | **Turbo Codes and EXIT Charts** — LTE turbo encoder and QPP, BCJR vs max-log, iterations, EXIT tunnel, BEC density evolution, coupling wave, error floors | `labs/lab23_turbo_exit.py` |
| 24 | 16 | **Source Coding** — Huffman vs arithmetic coding, LZ77, Lloyd–Max, the DCT, a working JPEG, an LPC vocoder you can hear, psychoacoustic masking and motion search | `labs/lab24_source_coding.py` |
| 07 | 17 | **OFDM: From the IFFT to the NR Grid** — Orthogonal subcarriers, IFFT + CP, CP vs multipath, the one-tap equaliser, Schmidl & Cox, pilots, PAPR vs DFT-s-OFDM, NR numerology | `labs/lab07_ofdm.py` |
| 28 | 17 | **OFDM as a System** — LS/DFT/LMMSE channel estimation, CFO/Doppler/phase noise, coded OFDM, clipping and the PA, WOLA/f-OFDM, DMT bit loading, OFDM radar | `labs/lab28_ofdm_system.py` |
| 25 | 18 | **Spread Spectrum, CDMA and GPS** — Gold codes, a DSSS link under jamming, frequency hopping, the RAKE, near–far and multiuser detection, and a GPS receiver from acquisition to position fix | `labs/lab25_spread_spectrum_gps.py` |
| 10 | 19 | **MIMO: Diversity, Capacity, Detection** — Fading and combining live, SC/EGC/MRC slopes, Alamouti, water-filling, ergodic/outage capacity, ZF/MMSE/SIC/ML, DMT, massive MIMO, MU precoding | `labs/lab10_mimo.py` |
| 29 | 19 | **Antenna Arrays and Beamforming** — Steering and grating lobes, planar panel EIRP, beam squint vs TTD, Bartlett/MVDR/MUSIC, SSB sweep and Type I codebook, OMP hybrid precoding, LOS MIMO | `labs/lab29_arrays_beamforming.py` |
| 11 | 20, 21, 22 | **Air Interfaces: Cells, Trunks, Schedulers** — Reuse and SIR, a live switchboard, OFDMA scheduling, link adaptation and HARQ, Wi-Fi contention, the 4096-QAM EVM budget, air interfaces side by side | `labs/lab11_air_interfaces.py` |
| 26 | 20 | **The Cellular Concept** — Reuse and SIR maps, live Erlang switchboard, ALOHA/CSMA live, ALOHA instability, Bianchi DCF, RACH and barring, PPP coverage, PF scheduling, handover | `labs/lab26_cellular_system.py` |
| 27 | 21 | **Inside an LTE/NR Downlink** — Resource grid builder, PSS/SSS and two-cell search, the PDSCH chain step by step, chase vs IR HARQ, CQI/MCS throughput, TS 38.306 peak rate | `labs/lab27_lte_nr_phy.py` |
| 31 | 22 | **Wi-Fi, Bluetooth LE, 802.15.4, LoRa and RFID** — 802.11 preamble, A-MPDU efficiency, rate anomaly, live Minstrel, BLE GFSK, coin-cell life, 802.15.4 O-QPSK, LoRa chirps, air time and range, RFID | `labs/lab31_wifi_ble_lora.py` |
| 18 | 23 | **Satellite Links** — Link budget as a decibel bank account, GEO vs LEO, a live LEO pass, rain fade and availability, ACM vs CCM, TWTA with 16QAM vs 16APSK, Voyager | `labs/lab18_satellite_link.py` |
| 30 | 24 | **Wireline and Optical** — Loop loss and Z0, loading coils and taps, DSL rate vs reach, vectoring, IM/DD vs coherent, live CMA, GN link planning, PON | `labs/lab30_wireline_optical.py` |
| 12 | 25 | **Frontiers: Delay–Doppler, ISAC, RIS, Learning** — Delay–Doppler view, OTFS vs OFDM on a fast train, phase noise and numerology, OFDM radar, RIS N² law, a learned demapper and autoencoder | `labs/lab12_frontiers.py` |
| 32 | 25 | **Toward 6G: Waveforms, Near Field, ISAC** — OFDM/OTFS/AFDM under Doppler, embedded pilot, near-field focusing, ISAC trade-off, RIS vs relay, sub-THz budget | `labs/lab32_6g_waveforms.py` |

### Self-test and editing a lab

```bash
python tests/selftest_labs.py                     # every lab, headless: defaults + random settings
python tests/selftest_labs.py lab03 lab14         # some labs
python tests/selftest_labs.py --random 2          # fewer random settings (faster)
python labs/lab03_pulse_shaping.py --selftest     # one lab; add --dark for the dark theme
python tests/test_commlib.py                      # library self-tests (after any commlib change)
```

The self-test opens each experiment, runs it at the default settings (including background
Monte Carlo jobs in quick mode and a few animation frames) and at random settings, fails on
exceptions, infinite values or slow updates, and saves a screenshot of every experiment to
`tests/screens/labNN/`. Look at them: layout bugs are bugs.

To write or change a lab, read `LAB_STYLE_GUIDE.md` (conventions, checklist) and
`studio/README.md` (API). A lab is one declarative Python file; all signal processing lives in
`commlib` so the labs and the book figures agree.

## 2. GNU Radio hardware examples

Tested with GNU Radio 3.10.9 / UHD 4 on Ubuntu 24.04. Every script runs **without hardware**
first: `--sim` replaces the B200 with `channels.channel_model` (CFO, clock offset,
multipath, AWGN), and `--nogui` prints statistics instead of opening Qt windows.

| Script | Book chapter | What it shows | Try |
|--------|---------|---------------|-----|
| `gr01_spectrum_iq_capture.py` | 2, 7 | Live spectrum/waterfall; records `.cfile` + SigMF | `--freq 100e6 --rate 2e6 --out ../data/capture.cfile` |
| `gr02_psk_link.py` | 9, 10, 12 | QPSK link: AGC, FLL, polyphase clock sync, CMA, Costas, live BER | `--sim --cfo 2000 --multipath --ppm 50 --snr 15` |
| `gr03_channel_sounder.py` | 11 | Zadoff-Chu sounder: power-delay profile, RMS delay spread | `--sim --multipath` |
| `gr04_ofdm_link.py` | 17 | 802.11a-like packet OFDM: PER and goodput | `--sim --multipath --cfo 5000 --snr 20` |
| `gr05_coded_link.py` | 14 | Framed BPSK + K=7 Viterbi (commlib): raw vs decoded BER | `--sim --snr -3` |

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

## License
Course material for teaching use. commlib, studio and the flowgraphs may be reused freely with
attribution.
