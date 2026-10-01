"""The list of labs shown by the launcher, grouped by the book's parts.

Each entry: (number, file stem, chapters, title, one-line description). A lab counts as
"interactive" (launchable) when its file calls ``studio.run``; the others are still Jupyter
notebooks waiting to be converted and are listed as "coming soon".
"""
import os

PARTS = [
    ("Part I · Foundations", range(1, 6)),
    ("Part II · Digital Signal Processing for Radio", range(6, 8)),
    ("Part III · Digital Transmission", range(8, 13)),
    ("Part IV · Information and Coding", range(13, 17)),
    ("Part V · Systems", range(17, 26)),
]

LABS = [
    (34, "lab34_telegraph_history", (1,), "The Telegraph and the Birth of Signalling",
     "Morse keying and key clicks, Morse vs Huffman, the Atlantic cable's law of squares, repeaters"),
    (35, "lab35_fourier_spectra", (2,), "Fourier, Spectra and Bandwidth",
     "Windows and leakage, uncertainty, group delay, spectrograms, five bandwidths, two-tone IP3"),
    (1, "lab01_baseband", (2, 7), "Complex Baseband and IQ Sampling",
     "Real vs IQ sampling, the SDR receive chain, IQ imbalance, sensitivity"),
    (33, "lab33_noise_detection", (3,), "Noise and Detection",
     "Gaussian tails, Rayleigh/Rice, filtered noise, Friis cascades, Y-factor, ROC curves"),
    (14, "lab14_analog_am_fm", (4,), "Analog Modulation: AM, SSB, FM and Stereo",
     "Envelope detector, DSB/SSB, Bessel sidebands and Carson, FM threshold and clicks, stereo MPX"),
    (15, "lab15_superhet_receiver", (4, 7), "The Superheterodyne Receiver",
     "Images, IF selectivity, AGC, zero-IF and low-IF"),
    (16, "lab16_pcm_companding", (5,), "Sampling, PCM and Companding",
     "Aliasing, quantization and dither, μ-law/A-law, delta and sigma-delta, T1"),
    (17, "lab17_multirate_dsp", (6,), "Digital Filters and Multirate DSP",
     "FIR/IIR design, polyphase, CIC and compensation, NCO, multi-stage DDC"),
    (36, "lab36_rf_transceiver", (7,), "The RF Transceiver",
     "IIP3 and cascades, receiver line-up, phase noise, PA classes and Doherty, DPD"),
    (3, "lab03_pulse_shaping", (8,), "Pulse Shaping, Nyquist and the Eye",
     "Raised cosine, zero ISI, live eye diagrams, matched filter, BER Monte Carlo, NRZ vs PAM-4"),
    (13, "lab13_line_codes_eyes", (8,), "Line Codes, Jitter and SerDes Eyes",
     "Line-code PSDs, scramblers, 8b/10b, jitter and bathtubs, NRZ vs PAM-4 with FFE, duobinary"),
    (2, "lab02_modulation", (9,), "Digital Modulation and Optimal Detection",
     "PSK/QAM/APSK constellations vs Eb/N0, FSK, DPSK, MSK/GMSK"),
    (21, "lab21_cpm_evm", (9,), "CPM, Spectra and EVM",
     "MSK/GMSK/GFSK, 99 % bandwidth, PAPR, EVM signatures and budgets"),
    (4, "lab04_synchronization", (10,), "Synchronization",
     "PLL playground, carrier and timing recovery, frame sync, burst receiver"),
    (5, "lab05_channels", (11,), "The Wireless Channel",
     "Path loss, fading, Doppler, level crossings, coherence bandwidth, sounding"),
    (22, "lab22_propagation_link_budget", (11,), "Propagation and Link Budgets",
     "Fresnel and knife-edge, two-ray, Hata/COST-231/38.901, rain, coverage planner"),
    (6, "lab06_equalization", (12,), "Equalization",
     "ZF/MMSE, LMS/NLMS/RLS, CMA, DFE, MLSE"),
    (19, "lab19_information_theory", (13,), "Information Theory",
     "Entropy, Huffman, Blahut–Arimoto, constellation capacity, water-filling"),
    (8, "lab08_convolutional", (13, 14), "Convolutional Codes and Viterbi",
     "Capacity, Hamming, CRC, convolutional codes, puncturing, interleaving"),
    (20, "lab20_block_codes", (14,), "Block Codes: Hamming to Reed–Solomon",
     "Syndromes, CRC, GF(2^m), BCH and Reed–Solomon step by step, interleaving"),
    (9, "lab09_ldpc_polar", (15,), "LDPC and Polar Codes",
     "PEG construction, belief propagation, min-sum, SC and CA-SCL decoding"),
    (23, "lab23_turbo_exit", (15,), "Turbo Codes and EXIT Charts",
     "LTE turbo code, BCJR, log-MAP vs max-log, EXIT charts, density evolution"),
    (24, "lab24_source_coding", (16,), "Source Coding",
     "Huffman and arithmetic coding, LPC vocoder, toy JPEG, perceptual masking"),
    (7, "lab07_ofdm", (17,), "OFDM Basics",
     "Cyclic prefix, ICI, Schmidl–Cox, pilots, PAPR, NR numerology"),
    (28, "lab28_ofdm_system", (17,), "An OFDM System",
     "Channel estimation, phase noise, coded OFDM, CFR and PA, DMT bit loading, OFDM radar"),
    (25, "lab25_spread_spectrum_gps", (18,), "Spread Spectrum and GPS",
     "m-sequences and Gold codes, DSSS vs jammer, Rake, near–far, GPS acquisition and fix"),
    (10, "lab10_mimo", (19,), "MIMO",
     "Diversity, capacity, detection, beamforming, massive-MIMO precoding"),
    (29, "lab29_arrays_beamforming", (19,), "Antenna Arrays and Beamforming",
     "Planar arrays, beam squint, MUSIC/MVDR, codebooks, hybrid precoding"),
    (26, "lab26_cellular_system", (20,), "The Cellular Concept",
     "Reuse and SIR, Erlang B/C, ALOHA stability, PPP coverage, scheduling"),
    (11, "lab11_air_interfaces", (20, 21, 22), "Air Interfaces",
     "Reuse, Erlang, NR grid, scheduling, link adaptation, CSMA/CA"),
    (27, "lab27_lte_nr_phy", (21,), "The LTE/NR Physical Layer",
     "Resource grid, PSS/SSS cell search, PDSCH chain, BLER per MCS, HARQ"),
    (31, "lab31_wifi_ble_lora", (22,), "Wi-Fi, Bluetooth and LoRa",
     "802.11 preamble, MAC efficiency, rate adaptation, BLE, 802.15.4, LoRa"),
    (18, "lab18_satellite_link", (23,), "Satellite Links",
     "Link budgets, GEO/LEO geometry and Doppler, rain fade, ACM, TWTA back-off"),
    (30, "lab30_wireline_optical", (24,), "Wireline and Optical",
     "Loops and DSL, vectoring, fibre dispersion, coherent DSP, PON budgets"),
    (12, "lab12_frontiers", (25,), "Frontiers",
     "OTFS vs OFDM, OFDM radar, RIS, learned demapper"),
    (32, "lab32_6g_waveforms", (25,), "6G Waveforms",
     "OTFS and AFDM, near-field focusing, ISAC range–Doppler, RIS sizing"),
]

LABS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs"))


def lab_path(stem):
    return os.path.join(LABS_DIR, stem + ".py")


def is_interactive(stem):
    """True when the lab has been converted to a studio script."""
    try:
        with open(lab_path(stem), encoding="utf-8") as fh:
            txt = fh.read()
        return "studio.run(" in txt or "st.run(" in txt
    except OSError:
        return False


def by_part():
    """[(part title, [entries sorted by chapter then number])]"""
    out = []
    for title, chapters in PARTS:
        items = [e for e in LABS if e[2][0] in chapters]
        items.sort(key=lambda e: (e[2][0], e[0]))
        out.append((title, items))
    return out
