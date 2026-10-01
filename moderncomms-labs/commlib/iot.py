"""Wi-Fi, Bluetooth and IoT physical-layer helpers (Chapter 22).

Import explicitly:  ``from commlib import iot``

Contents
--------
* LoRa chirp spread spectrum: vectorised modulator/demodulator, exact noncoherent SER,
  time-on-air (Semtech AN1200.13 formula), sensitivity estimate.
* GFSK (Bluetooth BR / BLE): modulator and limiter-discriminator demodulator.
* IEEE 802.15.4 2.4 GHz O-QPSK DSSS: the 16 chip sequences and a half-sine O-QPSK modulator.
* IEEE 802.11 legacy preamble (L-STF, L-LTF) at 20 MS/s and a delay-and-correlate detector.
* IEEE 802.11 MAC timing: efficiency of a single MPDU versus an A-MPDU exchange.
* Passive UHF RFID forward/reverse link range.

All functions are teaching models: they follow the standards' numerology but omit details
(interleaving, whitening, coding) that do not change the point being made.
"""
import numpy as np
from math import ceil

# ============================================================== LoRa (chirp spread spectrum)

def lora_symbols(sf, symbols, osr=1, down=False):
    """Complex baseband LoRa chirps for an array of symbol values (0..2^sf-1).

    Each symbol is the base up-chirp sweeping -BW/2..+BW/2 in 2^sf chips, cyclically shifted by
    the symbol value. Sampled at osr samples per chip. Returns a 1-D array of length
    len(symbols)*2^sf*osr. ``down=True`` returns conjugate (down) chirps."""
    M = 2 ** sf
    n = np.arange(M * osr) / osr                           # time in chips
    s = np.atleast_1d(np.asarray(symbols))[:, None]
    k = (n[None, :] + s) % M
    x = np.exp(2j * np.pi * (k ** 2 / (2 * M) - k / 2))
    x = np.conj(x) if down else x
    return x.ravel()


def lora_demod(x, sf, osr=1):
    """Dechirp (multiply by conjugate base chirp) and FFT, symbol by symbol.

    Returns (decided symbols, |FFT| matrix of shape (n_sym, 2^sf))."""
    M = 2 ** sf
    base = lora_symbols(sf, [0], osr=1)
    y = np.asarray(x)[::osr]
    n_sym = len(y) // M
    y = y[:n_sym * M].reshape(n_sym, M) * np.conj(base)[None, :]
    X = np.abs(np.fft.fft(y, axis=1))
    return np.argmax(X, axis=1), X


def lora_ser_sim(sf, snr_db, n_sym=2000, rng=None):
    """Monte Carlo symbol error rate. snr_db is the SNR in the signal bandwidth BW
    (one complex sample per chip, unit-power chirp)."""
    rng = np.random.default_rng(rng)
    M = 2 ** sf
    s = rng.integers(0, M, n_sym)
    x = lora_symbols(sf, s)
    sig = 10 ** (-snr_db / 20) / np.sqrt(2)
    y = x + sig * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    d, _ = lora_demod(y, sf)
    return np.mean(d != s)


def lora_ser_theory(sf, snr_db, ngrid=4000):
    """Exact SER of noncoherent M-ary orthogonal detection (M = 2^sf) with per-chip SNR snr_db,
    by numerical integration over the Rician-distributed correct bin."""
    from scipy.special import i0e
    M = 2 ** sf
    out = []
    for snr in np.atleast_1d(snr_db):
        g = M * 10 ** (snr / 10)                   # symbol SNR Es/N0 after the FFT gain
        # normalised: noise bins |N|^2 ~ Exp(1); correct bin r^2 with noncentrality 2g (in units of sigma^2=1/2 each dim)
        A = np.sqrt(2 * g)
        r = np.linspace(0, A + 12, ngrid)
        pdf = r * np.exp(-(r - A) ** 2 / 2) * i0e(A * r)      # Rician, sigma = 1
        cdf_other = (1 - np.exp(-r ** 2 / 2)) ** (M - 1)
        pc = np.trapezoid(pdf * cdf_other, r)
        out.append(1 - pc)
    return np.array(out)


LORA_SNR_REQ = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}   # dB, Semtech SX127x datasheets


def lora_bitrate(sf, bw=125e3, cr=1):
    """Raw bit rate SF * BW/2^SF * 4/(4+cr)."""
    return sf * bw / 2 ** sf * 4 / (4 + cr)


def lora_sensitivity_dbm(sf, bw=125e3, nf_db=6.0, snr_req_db=None):
    snr = LORA_SNR_REQ[sf] if snr_req_db is None else snr_req_db
    return -174 + 10 * np.log10(bw) + nf_db + snr


def lora_time_on_air(pl, sf, bw=125e3, cr=1, n_preamble=8, crc=True, implicit_header=False, de=None):
    """Time on air (s) of a LoRa packet, Semtech AN1200.13 / SX1276 datasheet formula."""
    if de is None:
        de = 1 if (2 ** sf / bw) > 16e-3 else 0     # low-data-rate optimisation for long symbols
    ts = 2 ** sf / bw
    t_pre = (n_preamble + 4.25) * ts
    num = 8 * pl - 4 * sf + 28 + 16 * int(crc) - 20 * int(implicit_header)
    n_pay = 8 + max(ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return t_pre + n_pay * ts


# ============================================================== GFSK (Bluetooth)

def gaussian_taps(bt, sps, span=4):
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    a = np.sqrt(np.log(2) / 2) / bt
    g = np.sqrt(np.pi) / a * np.exp(-(np.pi * t / a) ** 2)
    return g / g.sum()


def gfsk_mod(bits, sps=8, bt=0.5, h=0.5):
    """GFSK baseband: NRZ -> Gaussian filter -> FM with modulation index h.

    Returns (complex baseband, instantaneous frequency in units of the symbol rate)."""
    nrz = np.repeat(2 * np.asarray(bits) - 1.0, sps)
    f = np.convolve(nrz, gaussian_taps(bt, sps), mode="same") * h / 2   # peak deviation h/2 * Rs
    phase = 2 * np.pi * np.cumsum(f) / sps
    return np.exp(1j * phase), f


def fm_discriminator(x):
    """Instantaneous frequency (cycles/sample) by the phase-difference discriminator."""
    return np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi)


# ============================================================== IEEE 802.15.4 O-QPSK (2.4 GHz)

_C0 = "11011001110000110101001000101110"


def ieee802154_chips():
    """The 16 x 32 chip table of the 2.4 GHz O-QPSK PHY (IEEE 802.15.4-2020, Table 12-1):
    symbols 1-7 are right-cyclic shifts of symbol 0 by 4 chips; symbols 8-15 invert the
    odd-indexed chips of symbols 0-7."""
    c0 = np.array([int(c) for c in _C0])
    tab = [np.roll(c0, 4 * k) for k in range(8)]
    odd = np.zeros(32, int); odd[1::2] = 1
    tab += [t ^ odd for t in tab[:8]]
    return np.array(tab)


def oqpsk_halfsine(chips, sps=8):
    """Half-sine O-QPSK: even chips on I, odd chips on Q delayed by one chip (Tc).
    Each pulse spans 2 Tc. Returns complex baseband at sps samples per chip."""
    c = 2 * np.asarray(chips) - 1.0
    ci, cq = c[0::2], c[1::2]
    p = np.sin(np.pi * np.arange(2 * sps) / (2 * sps))
    n = len(c) * sps + 2 * sps
    i = np.zeros(n); q = np.zeros(n)
    for k, v in enumerate(ci):
        i[2 * k * sps:2 * k * sps + 2 * sps] += v * p
    for k, v in enumerate(cq):
        q[(2 * k + 1) * sps:(2 * k + 1) * sps + 2 * sps] += v * p
    return i + 1j * q


# ============================================================== IEEE 802.11 legacy preamble

_STF = {-24: 1, -20: -1, -16: 1, -12: -1, -8: -1, -4: 1, 4: -1, 8: -1, 12: 1, 16: 1, 20: 1, 24: 1}
_LTF = [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
        1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1]


def wifi_ltf_freq():
    """L-LTF frequency-domain sequence on subcarriers -26..26 (index 0 = DC)."""
    return np.array(_LTF, float)


def wifi_legacy_preamble():
    """L-STF (8 us) + L-LTF (8 us) at 20 MS/s, as in IEEE 802.11-2020 Clause 17 (no windowing).
    Returns (stf[160], ltf[160])."""
    X = np.zeros(64, complex)
    for k, v in _STF.items():
        X[k % 64] = np.sqrt(13 / 6) * v * (1 + 1j)
    s = np.fft.ifft(X) * 64 / np.sqrt(52)
    stf = np.tile(s, 3)[:160]                      # ten 0.8 us repetitions of a 16-sample pattern
    L = np.zeros(64, complex)
    for k, v in zip(range(-26, 27), _LTF):
        L[k % 64] = v
    l = np.fft.ifft(L) * 64 / np.sqrt(52)
    ltf = np.concatenate([l[-32:], l, l])          # 1.6 us GI2 + two 3.2 us symbols
    return stf, ltf


def delay_correlate(r, D=16, L=48):
    """Schmidl-Cox-style metric for a periodic preamble: |P|^2 / R^2 with
    P(n) = sum r[n+k] r*[n+k+D], R(n) = sum |r[n+k+D]|^2 over a window L. Returns (metric, P)."""
    c = r[:-D] * np.conj(r[D:])
    e = np.abs(r[D:]) ** 2
    k = np.ones(L)
    P = np.convolve(c, k, "valid")
    R = np.convolve(e, k, "valid")
    return np.abs(P) ** 2 / np.maximum(R, 1e-12) ** 2, P


# ============================================================== IEEE 802.11 MAC efficiency

WIFI5G = dict(slot=9e-6, sifs=16e-6, cwmin=15)


def legacy_ppdu_time(nbytes, rate_mbps=24.0):
    """Duration of a legacy OFDM (non-HT) PPDU: 20 us preamble + ceil((16+8L+6)/Ndbps) symbols of 4 us."""
    ndbps = rate_mbps * 4
    return 20e-6 + 4e-6 * ceil((16 + 8 * nbytes + 6) / ndbps)


def mac_exchange(phy_mbps, msdu=1500, n_agg=1, t_preamble=40e-6, mac_overhead=38, ack_rate=24.0,
                 aifsn=3, p=WIFI5G, max_ppdu=5.484e-3, symbol=None):
    """One uncontended EDCA exchange: AIFS + mean backoff + PPDU + SIFS + (Block)Ack.

    Returns (throughput in Mb/s, MAC efficiency). n_agg MPDUs are aggregated (A-MPDU; each subframe
    adds a 4-byte delimiter and padding to 4 bytes); n_agg=1 means a single MPDU acknowledged by an
    ACK. ``symbol`` (s) rounds the payload up to whole OFDM symbols if given. The PPDU is truncated to
    ``max_ppdu`` by reducing n_agg."""
    aifs = p["sifs"] + aifsn * p["slot"]
    backoff = p["cwmin"] / 2 * p["slot"]
    mpdu = msdu + mac_overhead
    if n_agg > 1:
        sub = 4 + mpdu
        sub += (-sub) % 4
        nbytes_max = (max_ppdu - t_preamble) * phy_mbps * 1e6 / 8
        n_agg = int(max(1, min(n_agg, nbytes_max // sub)))
        nbytes = n_agg * sub
        t_ack = legacy_ppdu_time(32, ack_rate)       # compressed Block Ack, 64-bit bitmap
    else:
        nbytes = mpdu
        t_ack = legacy_ppdu_time(14, ack_rate)
    t_pay = nbytes * 8 / (phy_mbps * 1e6)
    if symbol:
        t_pay = ceil(t_pay / symbol) * symbol
    T = aifs + backoff + t_preamble + t_pay + p["sifs"] + t_ack
    thr = n_agg * msdu * 8 / T / 1e6
    return thr, thr / phy_mbps


# ============================================================== passive RFID

def rfid_forward_range(eirp_w, f_hz, g_tag=1.64, p_th_w=10e-6, tau=1.0, pol_loss=1.0):
    """Tag-power-limited read range (m): d = lambda/(4 pi) sqrt(EIRP G_tag tau / (P_th L_pol))."""
    lam = 3e8 / f_hz
    return lam / (4 * np.pi) * np.sqrt(eirp_w * g_tag * tau / (p_th_w * pol_loss))


def rfid_backscatter_dbm(d, eirp_dbm, f_hz, g_tag_dbi=2.15, g_rx_dbi=6.0, mod_loss_db=8.0):
    """Backscattered power at the reader (dBm): two-way radar-like 1/d^4 link."""
    lam = 3e8 / f_hz
    fspl = 20 * np.log10(4 * np.pi * np.asarray(d) / lam)
    return eirp_dbm + 2 * g_tag_dbi + g_rx_dbi - 2 * fspl - mod_loss_db


if __name__ == "__main__":
    # quick self-checks
    ch = ieee802154_chips()
    x = bits = None
    for sf in (7, 12):
        print("LoRa SF", sf, "Ts %.3f ms" % (2 ** sf / 125e3 * 1e3), "Rb %.0f b/s" % lora_bitrate(sf),
              "sens %.1f dBm" % lora_sensitivity_dbm(sf), "ToA(20B) %.1f ms" % (lora_time_on_air(20, sf) * 1e3))
    print("SER theory SF7 @-7.5dB", lora_ser_theory(7, -7.5), "sim", lora_ser_sim(7, -7.5, 4000, 1))
    stf, ltf = wifi_legacy_preamble()
    print("STF power", np.mean(abs(stf) ** 2), "LTF power", np.mean(abs(ltf) ** 2))
    for r in (54, 600, 2400):
        print(r, "Mb/s single:", mac_exchange(r, n_agg=1, t_preamble=20e-6 if r == 54 else 40e-6),
              "A-MPDU64:", mac_exchange(r, n_agg=64))
    print("RFID range 4W EIRP:", rfid_forward_range(4, 915e6))
