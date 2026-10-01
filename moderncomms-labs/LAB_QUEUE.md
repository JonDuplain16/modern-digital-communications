# Labs requested by chapter authors (coordinator-assigned numbers)
| # | file | chapter | spec |
|---|------|---------|------|
| 13 | lab13_line_codes_eyes.py | 8 | line codes + PSDs, scramblers, 8b/10b RD, eyes/bathtub with RJ/DJ, NRZ vs PAM-4 over lossy channel with FFE, precoded duobinary (helpers in book/figscripts/ch08_figs.py) |
| 14 | lab14_analog_am_fm.py | 4 | AM/DSB/SSB/FM mod/demod, stereo MPX |
| 15 | lab15_superhet_receiver.py | 4,7 | superhet sim, image problem, AGC, zero-IF comparison |
| 16 | lab16_pcm_companding.py | 5 | sampling, quantization, companding, delta/sigma-delta, T1 |
| 17 | lab17_multirate_dsp.py | 6 | FIR/IIR, polyphase, CIC, NCO/DDC |
| 18 | lab18_satellite_link.py | 23 | link budget, GEO vs LEO geometry/Doppler, rain fade, ACM, TWTA backoff |
| 19 | lab19_information_theory.py | 13 | entropy, Huffman, Blahut–Arimoto, constrained capacity, water-filling |
| 20 | lab20_block_codes.py | 14 | Hamming/SECDED, CRC calculator, GF(2^m), RS with bursts, interleaving |
| 21 | lab21_cpm_evm.py | 9 | MSK/GMSK/GFSK gen+detect, spectra/99% BW, PAPR CCDF, EVM under impairments (reuse ch09_figs.py gmsk_baseband + impairments) |
| 22 | lab22_propagation_link_budget.py | 11 | link-budget & coverage planner: Friis, Hata/COST-231, 38.901 LOS/NLOS/O2I, knife-edge, two-ray, P.838 rain, Jakes–Reudink coverage (helpers in ch11_figs.py) |
| 23 | lab23_turbo_exit.py | 15 | turbo enc/BCJR dec, per-iteration BER, EXIT chart, BEC density evolution |
| 24 | lab24_source_coding.py | 16 | Huffman/canonical vs adaptive arithmetic on text, LPC analysis/resynthesis at new pitch, toy JPEG quality slider bpp/PSNR, masking-threshold shaped noise (helpers in ch16_figs.py) |
| 25 | lab25_spread_spectrum_gps.py | 18 | PN/Gold codes, DSSS w/ jammer, RAKE, near–far, GPS C/A FFT acquisition, DLL/PLL, position fix (commlib.spread) |
| 26 | lab26_cellular_system.py | 20 | hex-grid SIR maps, Erlang, ALOHA/CSMA sim, PPP coverage, PF scheduler |
| 27 | lab27_lte_nr_phy.py | 21 | resource grid, PSS/SSS cell search, PDSCH chain |
| 28 | lab28_ofdm_system.py | 17 | LMMSE/DFT/LS chest, Doppler+phase-noise ICI & CPE fix, coded OFDM ± BICM on EPA/ETU, clipping+filtering w/ Rapp PA ACLR/EVM, WOLA/f-OFDM spectra, DSL bit loading, OFDM radar range–Doppler (helpers in ch17_figs.py) |
| 29 | lab29_arrays_beamforming.py | 19 | planar array + EIRP calc, beam squint PS vs TTD, MUSIC/MVDR DOA, NR Type I codebook + SSB sweep, hybrid precoding by OMP, LOS-MIMO spacing (helpers in ch19_figs.py) |
| 30 | lab30_wireline_optical.py | 24 | loop loss & DSL rate vs reach, crosstalk/vectoring, fibre CD + DSP compensation, coherent DP-QPSK chain, OSNR budget |
| 31 | lab31_wifi_ble_lora.py | 22 | 802.11 preamble detect/CFO/LTF chest, MAC efficiency + aggregation + rate anomaly, Minstrel-like RA, BLE GFSK + battery calc, 802.15.4 O-QPSK, LoRa mod/demod SER vs theory + ToA/duty/link budget, RFID range (commlib/iot.py) |
| 32 | lab32_6g_waveforms.py | 25 | OTFS vs OFDM, near-field focusing, ISAC range-Doppler, RIS |
