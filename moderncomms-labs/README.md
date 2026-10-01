# Modern Digital Communications — Labs and SDR Examples

Companion code for the textbook *Modern Digital Communications*. Thirty-six interactive Python
simulation labs (plus a start-here index), a small shared DSP library, and five GNU Radio 3.10
flowgraphs for the Ettus USRP B200. **Start with `labs/lab00_index.ipynb`.**

```
commlib/     shared library: modulation, filters, channels, sync, equalizers (+ eqadv), OFDM (+ ofdmadv),
             coding (Viterbi, LDPC, polar; blockcodes, gf), MIMO, IQ file I/O, satellite links, line codes,
             information theory, CPM/EVM, propagation, source coding, spread spectrum/GNSS, turbo codes and
             EXIT/DE, cellular systems, an LTE/NR PHY (ltephy), wireline/optical, Wi-Fi/BLE/LoRa (iot), 6G
             waveforms (sixg), RF transceiver models (rf), and labkit (the common look-and-feel, widgets and self-checks used by every lab)
labs/        labNN_*.ipynb (pre-executed, with figures) + labNN_*.py sources (jupytext percent format)
gnuradio/    grNN_*.py hardware flowgraphs + grcommon.py helpers
tests/       test_commlib.py, build_notebooks.py (rebuild/execute labs), dump_figs.py (figure review)
data/        IQ captures from gr01 land here
```

## 1. Labs

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab labs/
```
Open a notebook and choose *Restart Kernel and Run All Cells*. Panels built with `interact` become
live sliders; in the shipped (pre-executed) notebooks they are rendered once at their defaults.

Every lab follows the same template: a title block (chapter, what you will learn, prerequisites, time,
roadmap); numbered sections that start with the physics and the key equations, then code, then a
**What you should see** paragraph; **Try it yourself** questions with `lk.check(...)` self-checks that
print PASS / FAIL / TODO; and closing **Key takeaways**, **Going further (USRP B200 / GNU Radio)** and
graded **Exercises**. Run times below are for the full notebook on a desktop PC.

| Lab | Chapter(s) | Topic | Interactive highlights | Run time |
|-----|-----------|-------|------------------------|---------:|
| 00 | — | Start here: setup, environment check, index | — | 1 s |
| 01 | 2, 7 | Complex baseband, IQ sampling, the SDR receive chain | real vs IQ sampling, IQ imbalance, sensitivity | 3 s |
| 02 | 9 | Digital modulation and optimal detection (PSK/QAM/APSK, FSK, DPSK, MSK/GMSK) | any constellation vs Eb/N0, GMSK BT | 5 s |
| 03 | 8 | Pulse shaping, Nyquist criterion, matched filter, eyes, ACLR | folded spectrum, eye explorer | 3 s |
| 04 | 10 | Carrier, timing and frame synchronization; MCRB | PLL playground, burst receiver | 6 s |
| 05 | 11 | Path loss, fading, Doppler, level crossings, coherence bandwidth, sounding | TDL time-frequency response | 10 s |
| 06 | 12 | ZF/MMSE, LMS/NLMS/RLS, CMA, DFE, MLSE | null depth vs ZF/MMSE | 5 s |
| 07 | 17 | OFDM: CP, ICI, Schmidl–Cox, pilots, PAPR, NR numerology | CP vs delay spread | 21 s |
| 08 | 13, 14 | Capacity, Hamming, CRC, convolutional codes, puncturing, interleaving | Viterbi frame viewer | 35 s |
| 09 | 15 | LDPC (PEG, BP, min-sum) and polar codes (SC, CA-SCL) | iterations histogram | 55 s |
| 10 | 19 | MIMO diversity, capacity, detection, beamforming, massive MIMO precoding | water-filling, beam steering | 12 s |
| 11 | 20–22 | Frequency reuse, Erlang B, NR grid, scheduling, link adaptation, CSMA/CA, 4096-QAM | PF scheduler, PAPR, MCS/HARQ | 6 s |
| 12 | 25 | OTFS vs OFDM, OFDM radar (ISAC), RIS, learned demapper | range-Doppler map, RIS | 11 s |
| 13 | 8 | Line codes and PSDs, PRBS/scramblers, 8b/10b, jitter and bathtubs, NRZ vs PAM-4 + FFE, duobinary | jitter budget, backplane loss | 8 s |
| 14 | 4 | AM/DSB/SSB, envelope and coherent detection, FM, Carson, FM threshold, FM stereo | AM index and RC, tone FM | 4 s |
| 15 | 4, 7 | Superheterodyne receiver, images, IF selectivity, AGC, zero-IF and low-IF | preselector Q and IF order, low-IF image | 2 s |
| 16 | 5 | Sampling, ZOH, quantization, dither, μ-law/A-law, delta and sigma-delta, T1 | aliasing, delta-mod step | 2 s |
| 17 | 6 | FIR/IIR design, polyphase, CIC + compensation, NCO, multi-stage DDC | FIR window vs Parks–McClellan | 2 s |
| 18 | 23 | Satellite link budgets, GEO/LEO geometry and Doppler, P.618 rain, ACM vs CCM, TWTA | DTH budget, orbit explorer | 3 s |
| 19 | 13 | Entropy of text and images, Huffman, Blahut–Arimoto, CM/BICM capacity, finite blocklength, water-filling | Markov source, block Huffman, constellation capacity, DSL loading | 5 s |
| 20 | 14 | Hamming/SECDED, CRC calculator and detection test, GF(2^m), BCH and Reed–Solomon step by step, bursts and interleaving | syndrome decoder, CRC, GF table, RS errors/erasures, interleaver depth | 9 s |
| 21 | 9 | MSK/GMSK/GFSK, 99% bandwidth, Laurent receiver vs differential/discriminator, PAPR CCDF, EVM signatures and budget | BT and h, GMSK eye, impairment explorer | 33 s |
| 22 | 11 | Antennas, Fresnel/knife-edge, two-ray, Hata/COST-231/TR 38.901 + O2I, gas and rain, Jakes–Reudink coverage, link-budget planner | path profile, models, coverage planner | 13 s |
| 23 | 15 | LTE turbo code (RSC, QPP), BCJR vs brute force, log-MAP vs max-log-MAP, BER per iteration, weight-2 error floor, EXIT charts, BEC density evolution incl. spatial coupling | block length/decoder/iterations, EXIT SNR, DE ensemble and ε | 58 s |
| 24 | 16 | Huffman/canonical, adaptive arithmetic coder, LPC vocoder, toy JPEG, masking threshold and shaped noise | arithmetic orders, LPC pitch, JPEG quality, noise offset | 18 s |
| 25 | 18 | m-sequences/Gold/Kasami/C/A, DSSS vs jammer, Rake, near–far + MUD, IS-95 capacity, GPS acquisition, tracking, position fix | jammer, near–far, acquisition grid, DOP | 8 s |
| 26 | 20 | Hexagonal reuse and SIR (sectors, shadowing), Erlang B/C + call simulator, ALOHA/CSMA, ALOHA instability, RACH, PPP coverage, PF/α-fair scheduling | cluster, SIR layout, trunking, stability, PPP, scheduler | 6 s |
| 27 | 21 | LTE grid and overhead, PSS/SSS cell search with CFO (two cells), PDSCH chain: CRC24, segmentation, LDPC, circular-buffer RM, scrambling, QAM, OFDM, CRS chest; BLER per MCS; HARQ chase vs IR | grid/PCI, cell search, PDSCH chain | 50 s |
| 28 | 17 | LS/DFT/LMMSE channel estimation, CFO/Doppler/phase-noise ICI, coded OFDM, CFR + PA (EVM/ACLR), WOLA/f-OFDM, DMT bit loading, OFDM radar | pilots, phase noise, CFR/PA, spectra, DSL reach, radar | 22 s |
| 29 | 19 | Planar arrays and EIRP, beam squint (PS vs TTD), Bartlett/MVDR/MUSIC, SSB sweep + Type I codebook, hybrid precoding by OMP, LOS MIMO | panel, squint, DOA, beams, hybrid, LOS spacing | 3 s |
| 30 | 24 | RLGC loops, loading coils, bridged taps, DSL loading and reach, FEXT vectoring, IM/DD vs coherent over dispersive fibre, coherent DSP chain, OSNR/GN and PON budgets | loop, DSL, vectoring, fibre, coherent DSP, link plan | 3 s |
| 31 | 22 | 802.11 preamble sync and LTF chest, MAC efficiency/aggregation/rate anomaly, Minstrel-like RA, BLE GFSK + battery, 802.15.4 O-QPSK, LoRa SER/ToA/range, RFID | preamble, MAC, Minstrel, BLE, Zigbee, LoRa | 5 s |
| 32 | 25 | DD channels and OTFS sizing, OFDM vs OTFS vs AFDM, embedded-pilot OTFS estimation, near-field focusing, ISAC range–Doppler and sidelobe floors, RIS sizing | grid, Doppler, pilot, focus, ISAC, RIS | 17 s |
| 33 | 3 | Gaussian/Q tails, Rayleigh/Rice, filtered noise and B_N, bandpass noise, spectral correlation, Friis cascades, Y-factor, ROC curves | samples, filter, line-up, Y-factor, ROC | 10 s |
| 34 | 1 | Morse keying and key clicks, Morse vs Huffman and Shannon's telegraph capacity, Atlantic cable law of squares, repeaters vs regenerators | keying, text, cable, repeaters | 2 s |
| 35 | 2 | Windows/scalloping/zero padding, uncertainty, group delay and dispersion, spectrograms, five bandwidths, two-tone IP3 | window, pulse width, delays, STFT, bandwidth, IP3 | 2 s |
| 36 | 7 | Two-tone IIP3 + cascade check, receiver line-up and SFDR, phase noise from an L(f) mask and reciprocal mixing, PA classes and Doherty, memory-polynomial DPD (ILA) | IIP3, line-up, phase noise, PA, DPD | 5 s |

### Writing or editing a lab
Edit `labs/labNN_*.py` (jupytext percent format: `# %%` code cells, `# %% [markdown]` text cells), then
rebuild and review:

```bash
python tests/build_notebooks.py lab07           # one lab (or several: lab07 lab13); no argument = all
python tests/dump_figs.py lab07_ofdm out.png     # stitch the lab's figures into one PNG to eyeball
python tests/test_commlib.py                     # library self-tests (run after any commlib change)
```
`build_notebooks.py` executes in-process (no Jupyter kernel), captures prints, figures and labkit tables
in order, records each cell's run time, and exits non-zero if a lab fails. It works on Linux, macOS and
Windows (Git Bash or PowerShell). It caps BLAS threads at 2: on many-core machines OpenBLAS thread
start-up can make the small matrix solves in these labs 100× slower, so the labs set
`OPENBLAS_NUM_THREADS=2` in their first cell as well.

New labs should start from any existing lab: use `from commlib import labkit as lk`, `rng = lk.setup(...)`,
`lk.fig/lk.show` for figures, `lk.interact` + `lk.slider/lk.islider/lk.choice` for controls, `lk.table` for
results, `lk.check` for self-checks and `lk.summary()` at the end.

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

## 3. Tests and rebuilding

```bash
python tests/test_commlib.py                 # library self-test (BER vs theory, loops lock, codes decode)
python tests/build_notebooks.py              # regenerate + execute every lab from labs/*.py
python tests/build_notebooks.py lab07        # just one
bash tests/build_all.sh                      # one process per lab (isolates failures)
```
Edit the `labs/*.py` sources (jupytext percent format) and rebuild, or edit the notebooks directly.

## License
Course material for teaching use. commlib and the flowgraphs may be reused freely with attribution.
