"""Sanity tests for commlib. Run:  python -m pytest -q tests/  (or python tests/test_commlib.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import commlib as cl  # noqa: E402

RNG = np.random.default_rng(0)


def test_qam_ber_matches_theory():
    for name, M in [("qpsk", 4), ("16qam", 16), ("64qam", 64)]:
        c = cl.get_constellation(name)
        bits = cl.random_bits(c.k * 200_000, RNG)
        s = c.modulate(bits)
        esn0 = 12.0 if M > 4 else 8.0
        y, n0 = cl.awgn_esn0(s, esn0, rng=RNG)
        ber = np.mean(c.demodulate(y) != bits)
        th = cl.ber_mqam_gray(esn0, M)
        assert 0.7 < ber / th < 1.3, (name, ber, th)


def test_llr_sign_consistency():
    c = cl.get_constellation("16qam")
    bits = cl.random_bits(4000, RNG)
    y, n0 = cl.awgn_esn0(c.modulate(bits), 30, rng=RNG)
    assert np.all((c.llr(y, n0) < 0) == bits.astype(bool))


def test_rrc_nyquist():
    sps = 8
    h = cl.rrc_taps(0.35, sps, 12)
    p = np.convolve(h, h)
    mid = len(p) // 2
    assert abs(p[mid] - 1) < 1e-3
    for k in range(1, 6):
        assert abs(p[mid + k * sps]) < 5e-3


def _psk_waveform(n, sps, delay, beta=0.35):
    c = cl.get_constellation("qpsk")
    bits = cl.random_bits(2 * n, RNG)
    s = c.modulate(bits)
    h = cl.rrc_taps(beta, sps, 10)
    x = cl.shape(s, h, sps)
    x = cl.fractional_delay(np.concatenate([x, np.zeros(20)]), delay)
    y = cl.matched_filter(x, h)
    return c, s, y, h


def test_gardner_converges():
    sps = 4
    c, s, y, h = _psk_waveform(4000, sps, 1.37)
    y, _ = cl.awgn_esn0(y, 25, sps=sps, rng=RNG)
    syms, e, tau = cl.gardner_sync(y, sps, bn=0.01)
    tail = syms[-1100:-100]   # skip the filter ramp-down at the very end
    evm = np.sqrt(np.mean(np.abs(tail - c.decide(tail)) ** 2))
    assert evm < 0.1, evm


def test_mm_converges():
    sps = 4
    c, s, y, h = _psk_waveform(4000, sps, 2.6)
    syms, e = cl.mm_sync(y, sps, c, bn=0.01)
    tail = syms[-1100:-100]
    evm = np.sqrt(np.mean(np.abs(tail - c.decide(tail)) ** 2))
    assert evm < 0.05, evm


def test_pll_removes_cfo():
    c = cl.get_constellation("qpsk")
    s = c.modulate(cl.random_bits(8000, RNG))
    y = cl.apply_cfo(s, 0.002, 0.7)
    y, _ = cl.awgn_esn0(y, 20, rng=RNG)
    z, ph = cl.pll_dd(y, c, bn=0.02)
    tail = z[-1000:]
    assert np.mean(np.abs(tail - c.decide(tail)) ** 2) < 0.05
    f = cl.cfo_power_estimate(y, 4)
    assert abs(f - 0.002) < 2e-4


def test_equalizers():
    c = cl.get_constellation("qpsk")
    s = c.modulate(cl.random_bits(2 * 6000, RNG))
    h = np.array([1, 0.45 + 0.3j, -0.2])
    y = np.convolve(s, h)[:len(s)]
    y, n0 = cl.awgn_esn0(y, 22, rng=RNG, es=1.0)
    w, d = cl.mmse_fir(h, 21, n0)
    z = cl.apply_fir(y, w, d)
    assert np.mean(np.abs(z[100:-100] - s[100:-100]) ** 2) < 0.05
    out, err, _ = cl.lms_equalizer(y, s[:500], L=21, mu=0.01, constellation=c)
    assert np.mean(err[-1000:]) < 0.08
    out, cost, _ = cl.cma_equalizer(y, L=21, mu=2e-3)
    assert np.mean(cost[-1000:]) < np.mean(cost[:200])


def test_ofdm_roundtrip_and_ls():
    cfg = cl.OFDMConfig(64, 48, 16)
    c = cl.get_constellation("16qam")
    grid = c.modulate(cl.random_bits(4 * 48 * 10, RNG)).reshape(10, 48)
    x = cl.ofdm_modulate(grid, cfg)
    assert np.allclose(cl.ofdm_demodulate(x, cfg), grid)
    h = np.array([0.9, 0, 0.3j, 0, 0, -0.2])
    y = np.convolve(x, h)[:len(x)]
    Y = cl.ofdm_demodulate(y, cfg)
    Htrue = np.fft.fft(h, 64)[cfg.active]
    mask = cl.comb_pilot_mask(10, 48, 3)
    Hh = cl.ls_channel_estimate(Y, grid, mask)
    assert np.mean(np.abs(Hh - Htrue) ** 2) < 1e-2


def test_viterbi():
    cc = cl.ConvCode()
    u = cl.random_bits(2000, RNG)
    x = 1 - 2 * cc.encode(u).astype(float)
    sigma = np.sqrt(1 / (2 * cc.rate * 10 ** (3 / 10)))  # Eb/N0 = 3 dB
    y = x + sigma * RNG.standard_normal(len(x))
    uh = cc.decode(2 * y / sigma ** 2)
    assert np.mean(uh != u) < 1e-2


def test_ldpc():
    code = cl.LDPCCode(n=480, rate=0.5, dv=3, seed=2)
    u = cl.random_bits(code.k, RNG)
    cw = code.encode(u)
    assert code.syndrome_ok(cw)
    sigma = np.sqrt(1 / (2 * code.rate * 10 ** (3 / 10)))
    y = (1 - 2 * cw) + sigma * RNG.standard_normal(code.n)
    for m in ("minsum", "spa"):
        ch = code.decode(2 * y / sigma ** 2, method=m)
        assert np.mean(code.info_bits(ch) != u) < 0.02, m


def test_polar():
    pc = cl.PolarCode(256, 128, design_snr_db=2.0, crc_poly=cl.CRC11_5G)
    ok = 0
    for t in range(20):
        u = cl.random_bits(128, RNG)
        x = 1 - 2 * pc.encode(u).astype(float)
        sigma = np.sqrt(1 / (2 * 0.5 * 10 ** (3.0 / 10)))
        y = x + sigma * RNG.standard_normal(len(x))
        ok += np.array_equal(pc.decode(2 * y / sigma ** 2, L=8), u)
    assert ok >= 17, ok
    # noiseless SC
    pc0 = cl.PolarCode(64, 32)
    u = cl.random_bits(32, RNG)
    assert np.array_equal(pc0.decode(10 * (1 - 2 * pc0.encode(u).astype(float))), u)


def test_mimo():
    c = cl.get_constellation("qpsk")
    errs = {"zf": 0, "mmse": 0, "ml": 0}
    n0 = 10 ** (-15 / 10)
    for _ in range(300):
        H = cl.rayleigh_mimo(4, 2, rng=RNG)
        x = c.points[RNG.integers(0, 4, 2)]
        y = H @ x + np.sqrt(n0 / 2) * (RNG.standard_normal(4) + 1j * RNG.standard_normal(4))
        errs["zf"] += np.sum(c.decide(cl.zf_detect(H, y)) != x)
        errs["mmse"] += np.sum(c.decide(cl.mmse_detect(H, y, n0)) != x)
        errs["ml"] += np.sum(cl.ml_detect(H, y, c.points) != x)
    assert errs["ml"] <= errs["zf"] + 2
    s = c.modulate(cl.random_bits(400, RNG))
    tx = cl.alamouti_encode(s)
    h = np.array([0.3 + 0.8j, -0.5 + 0.1j])
    r = h @ tx
    assert np.allclose(cl.alamouti_decode(r, h), s)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
