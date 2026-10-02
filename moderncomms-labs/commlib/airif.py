"""commlib.airif -- air-interface tables and system-level models (Chapters 20-22).

Import explicitly:  ``from commlib import airif as ai``

Everything here is a *table from a standard* or a *teaching model* that the book's figure
scripts (book/figscripts/ch20_figs.py, ch21_figs.py, ch22_figs.py) and Labs 11, 27 and 31
share, so that the numbers in the labs and in the chapters agree.

Contents
--------
* NR MCS table (TS 38.214 Table 5.1.3.1-1, 64QAM) and the LTE CQI table (TS 36.213
  Table 7.2.3-1); the "Shannon plus a gap" SNR each entry needs; a logistic BLER model.
* The hybrid-ARQ throughput model of Chapter 21 (none / chase combining / incremental
  redundancy).
* The NR peak-rate formula of TS 38.306 and the maximum RB counts of TS 38.101-1/-2.
* 802.11 rate tables (Chapter 22's rate-range model), the logistic PER model used for the
  Minstrel figure, and the transmitter EVM limits per MCS (IEEE 802.11-2020 / 802.11be).
* 802.11 DCF timing for Bianchi's model (Chapter 20) and the BLE advertising-charge model
  (Chapter 22).
"""
from __future__ import annotations

import numpy as np

# ===================================================================== NR / LTE link tables
# TS 38.214 Table 5.1.3.1-1 (MCS index table 1, up to 64QAM): (Qm, code rate x 1024)
NR_MCS64 = [(2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526),
            (2, 602), (2, 679), (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616),
            (4, 658), (6, 438), (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719),
            (6, 772), (6, 822), (6, 873), (6, 910), (6, 948)]

# TS 36.213 Table 7.2.3-1 (4-bit CQI, CQI 1..15): (Qm, code rate x 1024)
LTE_CQI = [(2, 78), (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 490),
           (4, 616), (6, 466), (6, 567), (6, 666), (6, 772), (6, 873), (6, 948)]

QAM_NAME = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM", 10: "1024QAM",
            12: "4096QAM"}


def spectral_efficiency(table):
    """Bits per resource element Qm * R of each table entry."""
    return np.array([q * r / 1024 for q, r in table])


def required_snr_db(se, gap_db=2.0):
    """SNR (dB) at which an entry of spectral efficiency ``se`` reaches its BLER target in the
    "Shannon plus a gap" abstraction: 10 log10(2^se - 1) + gap."""
    return 10 * np.log10(2.0 ** np.asarray(se, float) - 1) + gap_db


def bler_logistic(snr_db, req_db, target=0.1, slope_db=0.8):
    """Smooth BLER waterfall that equals ``target`` at ``req_db`` and falls by a factor of
    about e every ``slope_db`` dB (a common link-to-system abstraction)."""
    a = 1.0 / slope_db
    b = np.log(1 / target - 1)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(a * (np.asarray(snr_db, float) - req_db) + b))


def select_entry(snr_db, req_db):
    """Index of the highest table entry whose requirement is met (-1 if none)."""
    return int(np.searchsorted(np.asarray(req_db), snr_db, side="right")) - 1


def harq_throughput(snr_db, rate, gap_db=2.0, scheme="ir", max_tx=4, width=0.12):
    """Chapter 21's HARQ model: a first transmission at ``rate`` b/s/Hz; after k transmissions
    the decoder succeeds when the accumulated capacity exceeds ``rate`` (smooth logistic of
    width ``width``). Chase combining adds SNR (log2(1 + k snr/gap)); incremental redundancy
    adds mutual information (k log2(1 + snr/gap)). Returns throughput in b/s/Hz."""
    snr = 10 ** (np.asarray(snr_db, float) / 10)
    g = 10 ** (gap_db / 10)
    kmax = 1 if scheme == "none" else max_tx
    en = np.zeros_like(snr)
    succ = np.zeros_like(snr)
    pall = np.ones_like(snr)
    for k in range(1, kmax + 1):
        cap = k * np.log2(1 + snr / g) if scheme == "ir" else np.log2(1 + k * snr / g)
        with np.errstate(over="ignore"):
            pf = 1 / (1 + np.exp((cap - rate) / width))
        en += pall
        succ += pall * (1 - pf)
        pall = pall * pf
    return rate * succ / en


# ===================================================================== NR peak rate
NR_SCS_KHZ = {0: 15, 1: 30, 2: 60, 3: 120, 4: 240}

# Maximum transmission bandwidth configuration N_RB (TS 38.101-1 Table 5.3.2-1, FR1;
# TS 38.101-2 Table 5.3.2-1, FR2), keyed by (mu, bandwidth in MHz).
NR_MAX_PRB = {
    (0, 5): 25, (0, 10): 52, (0, 15): 79, (0, 20): 106, (0, 25): 133, (0, 30): 160,
    (0, 40): 216, (0, 50): 270,
    (1, 5): 11, (1, 10): 24, (1, 15): 38, (1, 20): 51, (1, 25): 65, (1, 30): 78, (1, 40): 106,
    (1, 50): 133, (1, 60): 162, (1, 70): 189, (1, 80): 217, (1, 90): 245, (1, 100): 273,
    (2, 10): 11, (2, 15): 18, (2, 20): 24, (2, 25): 31, (2, 30): 38, (2, 40): 51, (2, 50): 65,
    (2, 60): 79, (2, 70): 93, (2, 80): 107, (2, 90): 121, (2, 100): 135,
    (3, 50): 32, (3, 100): 66, (3, 200): 132, (3, 400): 264,
}
LTE_RB = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}

# Overhead OH of TS 38.306 4.1.2: (FR, direction) -> fraction
NR_OVERHEAD = {("FR1", "DL"): 0.14, ("FR1", "UL"): 0.08, ("FR2", "DL"): 0.18, ("FR2", "UL"): 0.10}


def nr_bandwidths(mu):
    """Channel bandwidths (MHz) defined for numerology mu, ascending."""
    return sorted(b for (m, b) in NR_MAX_PRB if m == mu)


def nr_symbol_time(mu):
    """Average OFDM symbol duration including the cyclic prefix, T_s = 1e-3 / (14 * 2^mu)."""
    return 1e-3 / (14 * 2 ** mu)


def nr_peak_rate(layers, qm, n_prb, mu, overhead, scaling=1.0, rmax=948 / 1024):
    """Approximate maximum data rate (b/s) of one NR carrier, TS 38.306 Section 4.1.2."""
    return layers * qm * scaling * rmax * 12 * n_prb / nr_symbol_time(mu) * (1 - overhead)


# ===================================================================== 802.11 rate tables
# Chapter 22's rate-range model, MCS 0..13 (BPSK 1/2 ... 4096-QAM 5/6): SNR needed (dB),
# information bits per data subcarrier, data subcarriers per stream (802.11ax/be numerology).
WIFI_SNR_REQ = np.array([2, 5, 9, 11, 15, 18, 20, 25, 29, 31, 34, 37, 40, 43], float)
WIFI_BITS = np.array([0.5, 1, 1.5, 2, 3, 4, 4.5, 5, 6, 6.67, 7.5, 8.33, 9, 10])
WIFI_NSD = {20: 234, 40: 468, 80: 980, 160: 1960, 320: 3920}
WIFI_MCS_NAME = ["BPSK 1/2", "QPSK 1/2", "QPSK 3/4", "16QAM 1/2", "16QAM 3/4", "64QAM 2/3",
                 "64QAM 3/4", "64QAM 5/6", "256QAM 3/4", "256QAM 5/6", "1024QAM 3/4",
                 "1024QAM 5/6", "4096QAM 3/4", "4096QAM 5/6"]
# Transmitter constellation-error (EVM) limits per MCS, dB (IEEE 802.11-2020 Tables 17-18,
# 21-37, 27-49 and 802.11be-2024 for 4096-QAM).
WIFI_EVM_DB = np.array([-5, -10, -13, -16, -19, -22, -25, -27, -30, -32, -35, -35, -38, -38],
                       float)


def wifi_rate_mbps(mcs, bw_mhz, nss=1, gi_us=0.8):
    """802.11ax/be PHY rate (Mb/s): N_SD * bits * N_ss / (12.8 us + GI)."""
    return WIFI_NSD[bw_mhz] * WIFI_BITS[np.asarray(mcs)] * nss / (12.8 + gi_us)


def wifi_per(snr_db, mcs):
    """Logistic packet-error model of Chapter 22's Minstrel figure."""
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(1.6 * (np.asarray(snr_db, float) - WIFI_SNR_REQ[np.asarray(mcs)])))


def combine_snr_db(*snrs_db):
    """Independent impairments add as powers: 1/SNR = sum 1/SNR_i (all in dB)."""
    inv = sum(10 ** (-np.asarray(s, float) / 10) for s in snrs_db)
    return -10 * np.log10(inv)


# ===================================================================== 802.11 DCF timing
def ofdm_ppdu_us(nbytes, rate_mbps):
    """802.11a/g OFDM PPDU duration (us): 20 us preamble+SIGNAL, 16 service + 6 tail bits."""
    ndbps = rate_mbps * 4
    return 20 + 4 * np.ceil((16 + 8 * nbytes + 6) / ndbps)


def dcf_times(payload=1500, rate=54, basic=24, rtscts=False):
    """(slot, payload_time, Ts, Tc) in seconds for Bianchi's model (Chapter 20's numbers)."""
    slot, sifs = 9e-6, 16e-6
    difs = sifs + 2 * slot
    data = ofdm_ppdu_us(payload + 34, rate) * 1e-6
    ack = ofdm_ppdu_us(14, basic) * 1e-6
    rts = ofdm_ppdu_us(20, basic) * 1e-6
    cts = ofdm_ppdu_us(14, basic) * 1e-6
    pl = payload * 8 / (rate * 1e6)
    if rtscts:
        Ts = rts + sifs + cts + sifs + data + sifs + ack + difs
        Tc = rts + difs + sifs + cts
    else:
        Ts = data + sifs + ack + difs
        Tc = data + difs + sifs + ack
    return slot, pl, Ts, Tc


# ===================================================================== Bluetooth LE energy
def ble_adv_charge(payload=31, i_tx=6.0e-3, i_cpu=3.0e-3, t_cpu=0.4e-3, ramp=0.15e-3,
                   rate=1e6, n_ch=3):
    """Charge (C) of one advertising event and the packet air time (s): n_ch packets of
    1 + 4 + 2 + 6 + payload + 3 bytes, each with a radio ramp, plus a burst of CPU activity."""
    nbytes = 1 + 4 + 2 + 6 + payload + 3
    t_pkt = nbytes * 8 / rate
    return n_ch * (t_pkt + ramp) * i_tx + t_cpu * i_cpu, t_pkt


def battery_life_years(charge_per_event, interval_s, i_sleep=2e-6, capacity_mah=225.0):
    """Battery life (years) of a device that spends ``charge_per_event`` every ``interval_s``."""
    iavg = charge_per_event / np.asarray(interval_s, float) + i_sleep
    return capacity_mah * 1e-3 / iavg / 24 / 365
