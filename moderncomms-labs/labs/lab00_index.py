# %% [markdown]
# # Lab 0 — Start Here: Index of the Modern Digital Communications Labs
#
# Welcome. These notebooks are the hands-on half of *Modern Digital Communications*. Each one is a
# self-contained Python simulation of the ideas in one or two chapters of the book: you change a parameter, the
# physics answers. Nothing here needs hardware; where an Ettus USRP B200 and GNU Radio can take an experiment
# over the air, the lab ends with a **Going further** section that says how.
#
# ## 1. Set up (once)
#
# ```bash
# cd moderncomms-labs
# python -m venv .venv
# source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
# pip install -r requirements.txt      # numpy, scipy, matplotlib, ipywidgets, jupyterlab, jupytext
# jupyter lab labs/
# ```
#
# Then open any `labNN_*.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**. Every notebook is
# shipped already executed, so you can read all the figures before running anything.
#
# ## 2. How every lab is organised
#
# * **Title block**: which chapter it accompanies, what you will learn, prerequisites, time needed and a roadmap.
# * **Numbered sections**: a short physical explanation with the key equations, then code, then a
#   **What you should see** paragraph that tells you how to read the result.
# * **Interactive panels** (marked in the roadmap): sliders and menus built with `ipywidgets`. In the pre-built
#   notebook they are shown at their default settings; run the notebook to move them.
# * **Try it yourself** boxes: a question, a line `answer = None` for you to fill in, and a self-check that prints
#   `PASS`, `FAIL` (with a hint) or `TODO`.
# * **Key takeaways**, **Going further (USRP B200 / GNU Radio)** and graded **Exercises** (warm-up, core, stretch).
# * The last cell prints the run time and your self-check score.
#
# All labs share the small library `commlib/` (modulation, filters, channels, synchronisation, equalisers, OFDM,
# codes, MIMO, satellite links, line codes, information theory, block codes, CPM, propagation, source coding, spread spectrum) and `commlib/labkit.py` (the common look, widgets and self-checks). The
# library is short and written to be read: when a lab calls `cl.gardner_sync`, open `commlib/sync.py` and look.

# %%
import os, sys, glob, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")   # small matrices: avoid BLAS thread thrashing
sys.path[:0] = [p for p in (os.path.abspath(".."), os.path.abspath("."))
                if os.path.isdir(os.path.join(p, "commlib"))]
import numpy as np
import commlib as cl
from commlib import labkit as lk

rng = lk.setup(seed=0, lab="00")

# %% [markdown]
# ## 3. Check your environment
#
# The cell below imports everything the labs need and runs a few library self-tests (BER versus theory, a timing
# loop locking, a Viterbi and an LDPC decode). Everything should print PASS.

# %%
rows = []
for mod in ["numpy", "scipy", "matplotlib", "ipywidgets", "jupytext", "nbformat"]:
    try:
        m = __import__(mod)
        rows.append([mod, getattr(m, "__version__", "?"), "ok"])
    except ImportError:
        rows.append([mod, "-", "missing (pip install -r requirements.txt)"])
lk.table(rows, ["package", "version", "status"], title=f"Python {sys.version.split()[0]}, commlib {cl.__version__}")

c = cl.get_constellation("16qam")
bits = cl.random_bits(4 * 100_000, rng)
y, _ = cl.awgn_esn0(c.modulate(bits), 14.0, rng=rng)
lk.check("16-QAM BER at Es/N0 = 14 dB matches theory", np.mean(c.demodulate(y) != bits), cl.ber_mqam_gray(14.0, 16), rtol=0.25)
cc = cl.ConvCode()
u = cl.random_bits(500, rng)
lk.check("K=7 Viterbi decodes a noiseless codeword", None, cond=np.array_equal(cc.decode(1 - 2.0 * cc.encode(u)), u))
code = cl.LDPCCode(n=96, rate=0.5, seed=0)
cw = code.encode(cl.random_bits(code.k, rng))
lk.check("LDPC encoder produces valid codewords", None, cond=code.syndrome_ok(cw))

# %% [markdown]
# ## 4. The labs
#
# | Lab | Notebook | Chapter(s) | Topics |
# |---|---|---|---|
# | 1 | `lab01_baseband` | 2, 7 | Complex baseband, real vs IQ sampling, DDC, IQ imbalance and blind correction, phase noise, ADC dynamic range, sensitivity |
# | 2 | `lab02_modulation` | 9 | Constellations (PSK/QAM/APSK), signal space, ML regions, BER vs theory, Gray coding, union bound, FSK/DPSK, MSK/GMSK, LLRs |
# | 3 | `lab03_pulse_shaping` | 8 | Raised cosine and RRC, Nyquist criterion, ISI, matched filter, eye diagrams, timing sensitivity, ACLR vs filter length, FTN |
# | 4 | `lab04_synchronization` | 10 | Sync impairments, M-th power and data-aided CFO vs MCRB, PLL jitter and acquisition, Gardner timing, Zadoff–Chu frame sync, burst receiver |
# | 5 | `lab05_channels` | 11 | Path loss, two-ray, shadowing, link budget, Rayleigh/Rician fading, Doppler, level crossings, TDL models, coherence bandwidth, sounding |
# | 6 | `lab06_equalization` | 12 | ZF/MMSE, noise enhancement, length and delay, LMS/NLMS/RLS, CMA, DFE, MLSE |
# | 7 | `lab07_ofdm` | 17 | Orthogonality, cyclic prefix, CFO/ICI, Schmidl–Cox, pilot channel estimation, PAPR and DFT-s-OFDM, NR numerology |
# | 8 | `lab08_convolutional` | 13, 14 | Capacity, Hamming, CRC, convolutional codes and Viterbi, puncturing, interleaving |
# | 9 | `lab09_ldpc_polar` | 15 | PEG LDPC, sum-product and min-sum, block length, polarisation, SC and CA-SCL |
# | 10 | `lab10_mimo` | 19 | MRC and Alamouti, MIMO capacity, water-filling, ZF/MMSE/SIC/ML, beamforming, massive MIMO precoding |
# | 11 | `lab11_air_interfaces` | 20–22 | Frequency reuse, Erlang B, NR grid, OFDMA scheduling, DFT-s-OFDM, link adaptation, CSMA/CA, 4096-QAM EVM |
# | 12 | `lab12_frontiers` | 25 | OTFS vs OFDM, OFDM radar (ISAC), RIS, learned demapper |
# | 13 | `lab13_line_codes_eyes` | 8 | Line codes and PSDs, PRBS and scramblers, 8b/10b, jitter and bathtubs, NRZ vs PAM-4 with FFE, duobinary |
# | 14 | `lab14_analog_am_fm` | 4 | AM and envelope detection, DSB/SSB, FM spectra and Carson, discriminator and PLL, FM threshold, FM stereo |
# | 15 | `lab15_superhet_receiver` | 4, 7 | Superhet chain, image rejection, IF selectivity, AGC, zero-IF and low-IF with IQ imbalance |
# | 16 | `lab16_pcm_companding` | 5 | Aliasing, ZOH, SQNR, dither, μ-law/A-law, delta modulation, sigma-delta, T1 framing |
# | 17 | `lab17_multirate_dsp` | 6 | FIR/IIR design, polyphase decimation/interpolation, CIC and compensation, NCO, multi-stage DDC |
# | 18 | `lab18_satellite_link` | 23 | DTH and Ka link budgets, GEO look angles, LEO passes and Doppler, P.618 rain, ACM vs CCM, TWTA and APSK |
# | 19 | `lab19_information_theory` | 13 | Entropy of English and images, Markov sources, Huffman and block coding, Blahut–Arimoto, hard vs soft BPSK, CM and BICM capacity of QAM, finite blocklength, water-filling and bit loading |
# | 20 | `lab20_block_codes` | 14 | Hamming and syndrome decoding, (72,64) SECDED, CRC calculator and detection test, GF(2^m), BCH and Reed–Solomon step by step, erasures, burst interleaving |
# | 21 | `lab21_cpm_evm` | 9 | MSK/GMSK/GFSK, 99% bandwidth, MSK as OQPSK and the Laurent receiver, differential and discriminator detection, PAPR CCDF, EVM signatures and budgets |
# | 22 | `lab22_propagation_link_budget` | 11 | Antenna gain, Fresnel clearance and knife-edge, two-ray, Hata/COST-231/TR 38.901 and O2I, gas and rain, Jakes–Reudink coverage, link-budget planner |
# | 24 | `lab24_source_coding` | 16 | Huffman and canonical codes, adaptive arithmetic coding, LPC analysis and vocoder, toy baseline JPEG, masking threshold and shaped noise |
# | 25 | `lab25_spread_spectrum_gps` | 18 | m-sequences, Gold/Kasami and C/A codes, DSSS against a jammer, Rake, near–far and MUD, CDMA capacity, GPS acquisition, tracking, nav bits, position fix |
# | 28 | `lab28_ofdm_system` | 17 | LS/DFT/LMMSE channel estimation, ICI from CFO, Doppler and phase noise, coded OFDM, CFR and PA (EVM/ACLR), WOLA and filtered OFDM, DMT bit loading, OFDM radar |
#
# **Suggested paths.** *A first course in digital communications:* 1 → 3 → 2 → 4 → 5 → 6 → 7 → 8.
# *Wireless systems:* 5 → 22 → 7 → 28 → 10 → 11 → 25 → 12. *Radio and DSP fundamentals:* 14 → 15 → 16 → 17 → 1 → 21.
# *Information and coding:* 19 → 8 → 20 → 9 → 24. *Wireline/SerDes:* 3 → 13 → 6 → 28 (DMT). *Space and navigation:* 18 and 25 after 2, 5 and 8.
#
# The table below is generated from the notebooks themselves (title and the measured run time of the pre-built copy).

# %%
rows = []
for nb_path in sorted(glob.glob("lab[0-9][0-9]_*.ipynb")):
    name = os.path.basename(nb_path)[:-6]
    if name.startswith("lab00"):
        continue
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)
    title = next((l_.lstrip("# ").strip() for c_ in nb["cells"] if c_["cell_type"] == "markdown"
                  for l_ in "".join(c_["source"]).splitlines() if l_.startswith("# ")), name)
    runtime = sum(c_.get("metadata", {}).get("mdc_runtime_s", 0) for c_ in nb["cells"])
    n_fig = sum(1 for c_ in nb["cells"] for o in c_.get("outputs", []) if "image/png" in o.get("data", {}))
    rows.append([name, title.split("—", 1)[-1].strip(), n_fig, f"{runtime:.0f} s" if runtime else "not built"])
lk.table(rows, ["notebook", "title", "figures", "run time"], title="Labs in this folder")

# %% [markdown]
# ## 5. Rebuilding the notebooks
#
# The `.ipynb` files are generated from the `labNN_*.py` sources (jupytext "percent" format, readable and diff-able
# as plain Python). After editing a source:
#
# ```bash
# python tests/build_notebooks.py lab07         # convert + execute one lab (or several: lab07 lab13)
# python tests/build_notebooks.py               # all labs, with a timing summary
# python tests/test_commlib.py                  # library self-tests
# python tests/dump_figs.py lab07_ofdm out.png  # stitch a lab's figures into one image for review
# ```
#
# ## 6. Hardware (secondary): USRP B200 and GNU Radio
#
# The `gnuradio/` folder has five GNU Radio 3.10 flowgraphs (spectrum and IQ capture, a QPSK link, a channel sounder,
# an OFDM link and a coded link). Every one runs without hardware first (`--sim`). See `README.md` for installation
# notes (radioconda is the easy route) and for safe cabled-loopback wiring: **never connect TX directly to RX; use
# at least 30 dB of attenuation.**

# %%
lk.summary()
