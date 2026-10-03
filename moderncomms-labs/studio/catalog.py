"""The list of labs shown by the launcher, grouped by the book's parts.

Each entry: (number, file stem, chapters, title, one-line description). A lab counts as
"interactive" (launchable) when its file calls ``studio.run`` (detected automatically);
anything else would be listed as "coming soon". All 36 labs are studio apps.
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
     "Morse keying, Morse vs Huffman, Kelvin's cable, semaphore pipeline, regenerators, loading coils, capacity"),
    (35, "lab35_fourier_spectra", (2,), "Fourier, Spectra and Bandwidth",
     "Fourier series and Gibbs, windows and scalloping, zero padding, uncertainty, convolution in slow motion, group delay, five bandwidths, STFT, IP3"),
    (1, "lab01_baseband", (2, 7), "Complex Baseband and IQ Sampling",
     "The IQ helix, real vs IQ sampling, up/down-conversion, the DDC, zero-IF images, phase noise, ADC bits, sensitivity, a live waterfall"),
    (33, "lab33_noise_detection", (3,), "Noise and Detection",
     "Why noise is Gaussian, Q-function tails, Rayleigh/Rice, filtered and bandpass noise, Friis, Y-factor, ROC curves, signals below the noise"),
    (14, "lab14_analog_am_fm", (4,), "Analog Modulation: AM, SSB, FM and Stereo",
     "Envelope detector, DSB-SC and SSB, FM in time, Bessel sidebands and Carson, FM threshold and clicks, stereo multiplex (with audio)"),
    (15, "lab15_superhet_receiver", (4, 7), "The Superheterodyne Receiver",
     "Tune the band, images, IF selectivity, AGC, double conversion, IM3/SFDR, zero-IF vs low-IF"),
    (16, "lab16_pcm_companding", (5,), "Sampling, PCM and Companding",
     "Aliasing, ZOH, SQNR, dither, G.711, DM/CVSD, sigma-delta, jitter, T1/E1"),
    (17, "lab17_multirate_dsp", (6,), "Digital Filters and Multirate DSP",
     "Pole–zero, FIR/IIR and fixed point, aliasing, PFB channelizer, CIC, NCO spurs, CORDIC, a live DDC"),
    (36, "lab36_rf_transceiver", (7,), "The RF Transceiver",
     "Two-tone IIP3, line-up and SFDR, reciprocal mixing, I/Q image, PA ACLR/EVM, Doherty, CFR, DPD"),
    (3, "lab03_pulse_shaping", (8,), "Pulse Shaping, Nyquist and the Eye",
     "Pulses and spectra, zero ISI, live eye diagrams, the matched filter, TX/RX chains, BER Monte Carlo, NRZ vs PAM-4, faster than Nyquist"),
    (13, "lab13_line_codes_eyes", (8,), "Line Codes, Jitter and SerDes Eyes",
     "Typed line codes and PSDs, baseline wander, scramblers, 8b/10b, jitter bathtub, PAM-4 with CTLE/FFE/DFE, duobinary"),
    (2, "lab02_modulation", (9,), "Digital Modulation and Optimal Detection",
     "Constellation zoo, signal space, decision regions and MAP, BER Monte Carlo, LLRs, noncoherent FSK, Shannon plane, Rayleigh fading"),
    (21, "lab21_cpm_evm", (9,), "Constant Envelope, PAPR and EVM",
     "CPM phase and spectra, MSK = OQPSK (Laurent), three receivers, PAPR, PA regrowth, EVM signatures and budget"),
    (4, "lab04_synchronization", (10,), "Synchronization: Carrier, Timing and Frame",
     "Impairment diagnosis, PLL steps and slips, S-curves, live Costas and timing loops, frequency estimators vs CRB, Farrow, frame sync, Schmidl–Cox and PSS cell search"),
    (5, "lab05_channels", (11,), "The Wireless Channel",
     "Path loss and shadowing, room standing waves, Doppler speckle, Rayleigh/Rice/Nakagami, echoes as notches, 3GPP TDL models, diversity, channel sounding"),
    (22, "lab22_propagation_link_budget", (11,), "Propagation and Link Budgets",
     "Friis, the decibel bank account, Fresnel zones and knife edges, two-ray, Hata/COST-231/38.901, rain and oxygen, edge vs area coverage, a coverage map"),
    (6, "lab06_equalization", (12,), "Equalization",
     "ISI, ZF vs MMSE, length and delay, live LMS, LMS vs RLS, blind CMA, DFE error bursts, the Viterbi equalizer, SerDes CTLE/FFE/DFE"),
    (19, "lab19_information_theory", (13,), "Information Theory",
     "Entropy of your own text, live Huffman trees, Blahut–Arimoto, Shannon's limit, real constellations, short packets, water-filling and rate–distortion"),
    (8, "lab08_convolutional", (14,), "Convolutional Codes and the Viterbi Algorithm",
     "Shannon's gap, shift-register encoder, trellis and d_free, Viterbi step by step, soft vs hard, puncturing, traceback, bursts and interleaving"),
    (20, "lab20_block_codes", (14,), "Block Codes: Hamming to Reed–Solomon",
     "Hamming Venn diagram, SECDED, CRC calculator and detection, GF(2^m), Reed–Solomon errors/erasures, interleaving, coding gain"),
    (9, "lab09_ldpc_polar", (15,), "LDPC and Polar Codes",
     "Tanner graphs and girth, live belief propagation, min-sum variants, early stopping, polarization, SC vs CA-SCL, finite-length limits"),
    (23, "lab23_turbo_exit", (15,), "Turbo Codes and EXIT Charts",
     "LTE turbo encoder and QPP, BCJR vs max-log, iterations, EXIT tunnel, BEC density evolution, coupling wave, error floors"),
    (24, "lab24_source_coding", (16,), "Source Coding",
     "Huffman vs arithmetic coding, LZ77, Lloyd–Max, the DCT, a working JPEG, an LPC vocoder you can hear, psychoacoustic masking and motion search"),
    (7, "lab07_ofdm", (17,), "OFDM: From the IFFT to the NR Grid",
     "Orthogonal subcarriers, IFFT + CP, CP vs multipath, the one-tap equaliser, Schmidl & Cox, pilots, PAPR vs DFT-s-OFDM, NR numerology"),
    (28, "lab28_ofdm_system", (17,), "OFDM as a System",
     "LS/DFT/LMMSE channel estimation, CFO/Doppler/phase noise, coded OFDM, clipping and the PA, WOLA/f-OFDM, DMT bit loading, OFDM radar"),
    (25, "lab25_spread_spectrum_gps", (18,), "Spread Spectrum, CDMA and GPS",
     "Gold codes, a DSSS link under jamming, frequency hopping, the RAKE, near–far and multiuser detection, and a GPS receiver from acquisition to position fix"),
    (10, "lab10_mimo", (19,), "MIMO: Diversity, Capacity, Detection",
     "Fading and combining live, SC/EGC/MRC slopes, Alamouti, water-filling, ergodic/outage capacity, ZF/MMSE/SIC/ML, DMT, massive MIMO, MU precoding"),
    (29, "lab29_arrays_beamforming", (19,), "Antenna Arrays and Beamforming",
     "Steering and grating lobes, planar panel EIRP, beam squint vs TTD, Bartlett/MVDR/MUSIC, SSB sweep and Type I codebook, OMP hybrid precoding, LOS MIMO"),
    (26, "lab26_cellular_system", (20,), "The Cellular Concept",
     "Reuse and SIR maps, live Erlang switchboard, ALOHA/CSMA live, ALOHA instability, Bianchi DCF, RACH and barring, PPP coverage, PF scheduling, handover"),
    (11, "lab11_air_interfaces", (20, 21, 22), "Air Interfaces: Cells, Trunks, Schedulers",
     "Reuse and SIR, a live switchboard, OFDMA scheduling, link adaptation and HARQ, Wi-Fi contention, the 4096-QAM EVM budget, air interfaces side by side"),
    (27, "lab27_lte_nr_phy", (21,), "Inside an LTE/NR Downlink",
     "Resource grid builder, PSS/SSS and two-cell search, the PDSCH chain step by step, chase vs IR HARQ, CQI/MCS throughput, TS 38.306 peak rate"),
    (31, "lab31_wifi_ble_lora", (22,), "Wi-Fi, Bluetooth LE, 802.15.4, LoRa and RFID",
     "802.11 preamble, A-MPDU efficiency, rate anomaly, live Minstrel, BLE GFSK, coin-cell life, 802.15.4 O-QPSK, LoRa chirps, air time and range, RFID"),
    (18, "lab18_satellite_link", (23,), "Satellite Links",
     "Link budget as a decibel bank account, GEO vs LEO, a live LEO pass, rain fade and availability, ACM vs CCM, TWTA with 16QAM vs 16APSK, Voyager"),
    (30, "lab30_wireline_optical", (24,), "Wireline and Optical",
     "Loop loss and Z0, loading coils and taps, DSL rate vs reach, vectoring, IM/DD vs coherent, live CMA, GN link planning, PON"),
    (12, "lab12_frontiers", (25,), "Frontiers: Delay–Doppler, ISAC, RIS, Learning",
     "Delay–Doppler view, OTFS vs OFDM on a fast train, phase noise and numerology, OFDM radar, RIS N² law, a learned demapper and autoencoder"),
    (32, "lab32_6g_waveforms", (25,), "Toward 6G: Waveforms, Near Field, ISAC",
     "OFDM/OTFS/AFDM under Doppler, embedded pilot, near-field focusing, ISAC trade-off, RIS vs relay, sub-THz budget"),
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
