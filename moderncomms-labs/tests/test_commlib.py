"""Sanity tests for commlib. Run:  python -m pytest -q tests/  (or python tests/test_commlib.py)"""
import os
import sys
import traceback

# Small matrices + many-core machines: OpenBLAS thread start-up and contention can make a
# 256x256 solve 100x slower. A few threads are plenty for these labs (set before NumPy loads).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

try:                                   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("MPLBACKEND", "Agg")

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


def test_viterbi_batch_matches_single():
    cc = cl.ConvCode()
    u = RNG.integers(0, 2, (8, 300))
    c = cc.encode_batch(u)
    assert np.array_equal(c, np.array([cc.encode(x) for x in u]))
    y = (1 - 2.0 * c) + 0.9 * RNG.standard_normal(c.shape)
    assert np.array_equal(cc.decode_batch(y), np.array([cc.decode(r) for r in y]))


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


def test_ldpc_peg_has_no_4_cycles():
    for n, seed in [(576, 3), (480, 2), (96, 0)]:
        H = cl.LDPCCode(n=n, rate=0.5, dv=3, seed=seed).H.astype(int)
        overlap = H @ H.T
        np.fill_diagonal(overlap, 0)
        assert overlap.max() <= 1, (n, int((overlap > 1).sum() // 2))
        assert np.all(H.sum(axis=0) == 3)


def test_labkit_helpers():
    from commlib import labkit as lk
    rng = lk.setup(seed=1, quiet=True)
    assert isinstance(rng, np.random.Generator)
    assert lk.check("pass", 1.02, 1.0, rtol=0.05)
    assert not lk.check("fail", 2.0, 1.0)
    assert not lk.check("todo", None, 1.0)
    assert lk.check("cond", None, cond=True)
    lk.table([[1, 2.5], [3, 4.25]], ["a", "b"], title="t")
    ber = lk.ber_mc(lambda b: (5, 1000), min_errors=20)
    assert abs(ber - 5e-3) < 1e-12
    assert abs(lk.db(100) - 20) < 1e-12 and abs(lk.undb(3) - 1.9953) < 1e-3


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


def test_infotheory():
    from commlib import infotheory as it
    assert abs(it.entropy([0.5, 0.25, 0.125, 0.125]) - 1.75) < 1e-12
    assert abs(it.markov_entropy_rate([[0.9, 0.1], [0.3, 0.7]]) - 0.572) < 1e-3
    code = it.huffman_code(dict(a=0.4, b=0.2, c=0.2, d=0.1, e=0.1))
    assert abs(sum(p * len(code[s]) for s, p in dict(a=0.4, b=0.2, c=0.2, d=0.1, e=0.1).items()) - 2.2) < 1e-12
    C, p, _, _ = it.blahut_arimoto(np.array([[1.0, 0.0], [0.5, 0.5]]), 5000)
    assert abs(C - float(it.z_capacity(0.5))) < 1e-9
    assert abs(it.biawgn_capacity(1.0) - 0.7215) < 1e-3                # Chapter 13 worked example
    assert abs(it.qam_cm(16, 10.0) - it.mi_2d_mc(cl.get_constellation("16qam"), 10.0, 40000)) < 0.02
    assert abs(it.fbl_snr_penalty_db(512, 0.5, 1e-5) - 1.83) < 0.02      # URLLC example
    pw, mu = it.waterfill(np.array([0.1, 0.2, 0.5, 1.0]), 1.0)
    assert abs(pw.sum() - 1.0) < 1e-12 and np.all(pw >= 0)


def test_blockcodes():
    from commlib import blockcodes as bc
    h = bc.hamming_code(3)
    c = h.encode([1, 0, 1, 1])
    assert list(c) == [1, 0, 1, 1, 0, 1, 0]                            # Chapter 14 worked example
    for j in range(7):
        r = c.copy(); r[j] ^= 1
        assert np.array_equal(h.decode(r)[0], [1, 0, 1, 1])
    assert h.dmin() == 3 and bc.extended_hamming_code(3).dmin() == 4
    s = bc.hsiao_secded_72_64()
    u = RNG.integers(0, 2, 64); cw = s.encode(u)
    assert np.all(s.H.sum(axis=0) % 2 == 1)
    r = cw.copy(); r[[3, 40]] ^= 1
    assert s.decode(r)[2] == "detected"
    r = cw.copy(); r[17] ^= 1
    m, _, st = s.decode(r)
    assert st == "corrected" and np.array_equal(m, u)
    assert list(bc.crc_append([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1])[-4:]) == [1, 1, 1, 0]
    x = np.arange(24)
    assert np.array_equal(bc.block_deinterleave(bc.block_interleave(x, 4, 6), 4, 6), x)


def test_cpm():
    from commlib import cpm
    bits = RNG.integers(0, 2, 20000)
    x, _ = cpm.gmsk_baseband(bits, 16, 0.3)
    assert abs(cpm.occupied_bandwidth(x, 16) - 0.91) < 0.03            # Chapter 9: 99% BW of GMSK 0.3
    assert np.allclose(np.abs(x), 1)
    c = RNG.integers(0, 2, 4000)
    xm, _ = cpm.gmsk_baseband(cpm.msk_precode(c), 8, None)
    rx = cpm.LaurentReceiver(8, None); rx.calibrate(xm[:4000], c[:500])
    assert np.mean(rx.detect(xm, len(c))[:-3] != c[:-3]) == 0
    assert np.mean((cpm.differential_detect(xm, 8) > 0) != cpm.msk_precode(c)) == 0
    assert abs(cpm.evm_budget_db(-41.2, -45, -38, -40) - (-34.4)) < 0.1  # 1024-QAM budget example


def test_propagation():
    from commlib import propagation as pr
    assert abs(float(pr.dish_gain_dbi(0.6, 12e9)) - 35.7) < 0.1
    assert abs(float(pr.knife_edge_loss(0.0)) - 6.02) < 0.01
    assert abs(pr.radius_for_mapl(130.7, lambda d: pr.cost231(1800, 30, 1.5, d, 0)) - 0.70) < 0.01
    m, pe = pr.edge_margin_for_area(0.95, 3.5, 8)
    assert abs(m - 8.7) < 0.05 and abs(pe - 0.862) < 0.005
    k, a = pr.rain_k_alpha(20.0)
    assert abs(float(k) - 0.092) < 0.002 and abs(float(a) - 1.057) < 0.005
    assert abs(float(pr.o2i_loss_db(28, "high")) - 37.9) < 0.1


def test_sourcecoding():
    from commlib import sourcecoding as sc
    text = "the quick brown fox jumps over the lazy dog " * 30
    ac = sc.ArithmeticCoder(order=1)
    bits = ac.encode(text)
    assert sc.ArithmeticCoder(order=1, alphabet=ac.alphabet).decode(bits, len(text)) == text
    canon = sc.canonical_huffman({"a": 2, "b": 2, "c": 2, "d": 3, "e": 3})
    assert sc.huffman_decode(sc.huffman_encode("abcde", canon), canon) == list("abcde")
    lo, hi, cw = sc.arith_interval("abac", {"a": 0.6, "b": 0.3, "c": 0.1})
    assert abs((hi - lo) - 0.0108) < 1e-12 and lo <= int(cw, 2) / 2 ** len(cw) < hi
    a, k, E = sc.levinson(np.array([1.0, 0.8, 0.5]), 2)
    assert np.allclose(a, [1, -1.11111, 0.38889], atol=1e-4)
    img = np.tile(np.linspace(0, 255, 64), (64, 1))
    rec, nbits = sc.toy_jpeg(img, 75)
    assert sc.psnr(img, rec) > 35 and nbits < 64 * 64 * 2


def test_ofdmadv():
    from commlib import ofdm as co, ofdmadv as oa
    N, Nu = 256, 200
    cfg = co.OFDMConfig(N, Nu, 32)
    k = cfg.k.astype(float)
    d, p = oa.tdl_pdp("EPA", 3.84e6)
    H = oa.tdl_freq_response("EPA", k, N, 3.84e6, RNG)
    pidx = np.arange(0, Nu, 4)
    Hl, Hd, Hm = oa.chest_all(H[pidx], k[pidx], k, 1e-6, d, p, N, 32)
    assert np.mean(np.abs(Hm - H) ** 2) < 1e-3
    assert abs(float(oa.sir_cfo_db(0.023)) - 27.5) < 0.3                  # 28 GHz budget example
    f = np.arange(33, 512) * 4312.5
    assert abs(4000 * oa.dmt_bit_loading(oa.dsl_snr_db(f, 3.0)).sum() / 1e6 - 5.64) < 0.05
    X = np.exp(2j * np.pi * RNG.random((128, 32)))
    P = oa.radar_map(oa.radar_echo(X, [(30.0, 0.0, 1.0)], 120e3, 8.9e-6, 28e9), X)
    assert np.unravel_index(np.argmax(P), P.shape)[0] == round(30.0 / (3e8 / (2 * 128 * 120e3)))


def test_ltephy():
    from commlib import ltephy as lp
    msg = np.unpackbits(np.frombuffer(b"123456789", np.uint8))
    assert int("".join(map(str, lp.crc_attach(msg)[-24:])), 2) == 0xCDE703     # CRC-24/LTE-A check value
    assert lp.crc_ok(lp.crc_attach(msg, lp.CRC24B), lp.CRC24B)
    pci = 301
    x = lp.ofdm_mod(np.array([lp.subframe_grid(pci, sf, RNG)[0] for sf in range(6)]))
    r = np.r_[np.zeros(333), x] * np.exp(2j * np.pi * 2000 * np.arange(len(x) + 333) / lp.FS)
    m, seg = lp.pss_search(r[:6000])
    i, t = np.unravel_index(np.argmax(m), m.shape)
    assert i == pci % 3 and t == 333 + 823 + 9
    f = np.angle(np.sum(seg[i, 1, t] * np.conj(seg[i, 0, t]))) / (2 * np.pi * 64 / lp.FS)
    assert abs(f - 2000) < 200
    assert lp.sss_detect(r, t, i, f)[:2] == (pci // 3, 0)
    cb = RNG.integers(0, 2, 300)
    e = lp.rate_match(cb, 500, 2)
    soft = lp.rate_recover(1 - 2.0 * e, 300, 2)
    assert np.all((soft < 0) == cb.astype(bool))
    assert np.array_equal(lp.bit_deinterleave(lp.bit_interleave(e, 4), 4), e)
    code = cl.LDPCCode(n=96, rate=0.5, seed=0)
    u = RNG.integers(0, 2, (5, code.k))
    cw = code.encode(u).reshape(5, -1)
    d, _ = lp.ldpc_decode_batch(code, 4 * (1 - 2.0 * cw) + RNG.standard_normal(cw.shape))
    assert np.array_equal(d, cw)


def test_rf():
    from commlib import rf
    st = [("switch", -2.5, 2.5, None), ("LNA", 20, 1.5, -10), ("mixer", 15, 10, 5), ("BB", 30, 20, 15)]
    c = rf.cascade(st)[-1]
    assert abs(c[2] - 4.36) < 0.01 and abs(c[3] - (-19.0)) < 0.05                 # Chapter 7 worked example
    stage = rf.poly_stage(10, 5.0)
    assert abs(rf.two_tone(stage, -30)[2] - 5.0) < 0.1                             # two-tone recovers the IIP3
    sys_ = lambda x: rf.poly_stage(30, 15)(rf.poly_stage(15, 5)(rf.poly_stage(20, -10)(rf.poly_stage(-2.5)(x))))
    assert abs(rf.two_tone(sys_, -50)[2] - (-19.0)) < 0.2                           # cascade formula (coherent IM3)
    assert abs(rf.irr_exact(0.1, 1.0) - 39.6) < 0.1
    fs, n = 1e6, 1 << 18
    mf, md = np.array([1e2, 1e5]), np.array([-80.0, -80.0])
    phi = rf.phase_noise_from_mask(n, fs, mf, md, RNG)
    assert abs(np.rad2deg(np.std(phi)) - rf.rms_phase_deg(np.array([0, fs / 2]), [-80, -80])) < 0.1 * np.rad2deg(np.std(phi))
    v = np.linspace(0.01, 1, 50)
    assert np.allclose(rf.eff_doherty(np.array([0.5, 1.0])), np.pi / 4)


def test_analog():
    from commlib import analog as an
    fs = 400e3
    t = np.arange(20000) / fs
    x = (1 + 0.5 * np.sin(2 * np.pi * 500 * t)) * np.cos(2 * np.pi * 40e3 * t)
    rc = 1e-4
    a = np.exp(-1 / (rc * fs))
    ref = np.empty_like(x)
    v = 0.0
    for i, xi in enumerate(x):                 # the recursion the vectorised detector implements
        v = max(xi, a * v)
        ref[i] = v
    assert np.max(np.abs(an.envelope_detector(x, fs, rc) - ref)) < 1e-9
    fsb = 96e3
    tb = np.arange(96 * 100) / fsb
    z = np.exp(1j * 3 * np.sin(2 * np.pi * 1e3 * tb))
    f = an.fm_discriminator_hz(z, fsb)
    assert abs(np.max(f) - 3e3) < 30            # peak deviation = beta * fm
    L = np.sin(2 * np.pi * 1e3 * t)
    Lh, Rh = an.stereo_decode(an.stereo_mpx(L, 0 * L, fs), fs)
    assert 20 * np.log10(an.tone_level(Lh, 1e3, fs) / an.tone_level(Rh, 1e3, fs)) > 40


def test_fading():
    from commlib import fading as fdg
    e = np.array([0.0, 10.0, 20.0, 30.0])
    assert np.allclose(fdg.ber_bpsk_diversity(e), cl.ber_bpsk_rayleigh(e), rtol=1e-6)
    assert np.allclose(fdg.ber_bpsk_diversity(e, combining="sc"), cl.ber_bpsk_rayleigh(e), rtol=1e-3)
    assert np.allclose(fdg.ber_bpsk_diversity(e, m=1.0), cl.ber_bpsk_rayleigh(e), rtol=1e-6)
    # two-branch MRC, exact closed form ((1-mu)/2)^2 (2 + mu)
    g = 10 ** (e / 10)
    mu = np.sqrt(g / (1 + g))
    assert np.allclose(fdg.ber_bpsk_diversity(e, L=2), ((1 - mu) / 2) ** 2 * (2 + mu), rtol=1e-6)
    r = np.linspace(0, 6, 30001)
    for kind, K, m in [("rayleigh", 0, 1), ("rician", 6.0, 1), ("nakagami", 0, 3.0)]:
        pdf = fdg.envelope_pdf(r, kind, K=K, m=m)
        assert abs(np.sum(pdf) * (r[1] - r[0]) - 1) < 1e-3
        assert abs(np.sum(r ** 2 * pdf) * (r[1] - r[0]) - 1) < 1e-3
    assert abs(fdg.power_cdf(0.01) - 0.00995) < 1e-4            # 20 dB fade: 1 % of the time
    h = fdg.SoSFader(1, 32, rng=3).gains(np.arange(200_000) / 1e4, 100.0)[:, 0]
    assert abs(np.mean(np.abs(h) ** 2) - 1) < 0.1
    st_ = fdg.fade_stats(20 * np.log10(np.abs(h)), -10, 1e4)
    assert abs(st_["rate"] / fdg.lcr_rayleigh(10 ** -0.5, 100) - 1) < 0.15
    F = fdg.shadowing_field(96, 96, 10, 50, 8.0, rng=1)
    assert abs(F.std() - 8) < 1e-6
    d, p = cl.TDL_PROFILES["EPA"]
    assert fdg.coherence_bandwidth(np.array(d) * 1e-9, p) > 2e6


def test_modzoo():
    from commlib import modzoo as mz
    assert abs(mz.ebn0_required("qpsk", 1e-5) - 9.59) < 0.05
    assert abs(mz.ebn0_required("16qam", 1e-5) - 13.43) < 0.1
    assert abs(float(mz.shannon_ebn0_db(2.0)) - 1.76) < 0.01
    assert np.allclose(mz.ser_psk_exact([12.0], 8), cl.ser_mpsk(12.0, 8), rtol=1e-3)
    c = mz.constellation("16apsk")
    assert c.M == 16 and abs(np.mean(np.abs(c.points) ** 2) - 1) < 1e-9
    assert mz.nn_bit_flips(c) <= 1.0 + 1e-9                 # binary switching finds a Gray map
    assert mz.nn_bit_flips(mz.constellation("32cross")) > 1.0
    # large-M noncoherent FSK: integral form agrees with the alternating sum where both work
    a = mz.ser_orth_noncoherent([0.0, 5.0, 10.0], 16)
    g = 10 ** (np.array([0.0, 5.0, 10.0]) / 10)
    from scipy.special import comb
    ref = sum((-1) ** (n + 1) * comb(15, n) / (n + 1) * np.exp(-n * g / (n + 1)) for n in range(1, 16))
    assert np.allclose(a, ref, rtol=1e-3)
    assert np.allclose(mz.ber_fading_mrc(cl.ber_bpsk, [11.1], 2), mz.ber_rayleigh_mrc(11.1, 2), rtol=0.02)


def test_synckit():
    from commlib import synckit as sk
    rng = np.random.default_rng(1)
    q = cl.get_constellation("qpsk")
    a = q.points[rng.integers(0, 4, 3000)]
    loop = sk.CarrierLoop(0.02, detector="dd", M=4)
    loop.run(cl.apply_cfo(a[:1500], 0.003, 1.0))
    loop.run(cl.apply_cfo(a, 0.003, 1.0)[1500:])           # resumable
    assert abs(loop.freq - 0.003) < 2e-4
    h = cl.rrc_taps(0.35, 4, 12)
    x = cl.matched_filter(cl.shape(a, h, 4), h)
    x = cl.fractional_delay(np.r_[x, np.zeros(10)], 1.7)
    for ted in ("gardner", "mm", "el"):
        tl = sk.TimingLoop(4, 0.01, ted=ted)
        taus = np.concatenate([tl.push(x[i:i + 800])[2] for i in range(0, len(x), 800)])
        assert abs(np.median(taus[-500:-50]) - 1.7) < 0.15, ted
    z = np.exp(2j * np.pi * 0.01 * np.arange(64))
    for f in (sk.est_kay, sk.est_fitz, sk.est_lr, sk.est_periodogram):
        assert abs(f(z) - 0.01) < 1e-4, f.__name__
    assert np.allclose(sk.interp_taps(0.3, "cubic").sum(), 1.0)
    p = sk.nr_pss(0)
    assert len(p) == 127 and set(np.unique(p)) == {-1.0, 1.0}
    e = sk.pll_step_response(np.array([0.0, 50.0]), 0.707, "phase")
    assert abs(e[0] - 1) < 1e-9 and abs(e[1]) < 1e-6



def test_telegraph():
    from commlib import telegraph as tg
    assert abs(tg.telegraph_capacity() - 0.539) < 1e-3          # Shannon 1948
    p = tg.letter_stats()
    assert abs(sum(p[k] * tg.morse_units(tg.MORSE[k]) for k in p) - 9.07) < 0.01
    assert tg.keyed_envelope("PARIS", 1)[1] == 50               # the PARIS standard
    assert abs(tg.wpm_limit(2.1, 0.15, 3000) - 4.64) < 0.05      # Chapter 1 worked example
    assert abs(tg.alt_swing([0.24])[0] - 0.10) < 0.01           # 10 % swing at T = 0.24 tau
    v = tg.cable_response([1.0] * 40, 1.0, np.array([39.5]))   # long mark settles to 1
    assert abs(v[0] - 1) < 0.1
    T = tg.pipeline_times(3, 15, 30.0)
    assert T[0, -1] == 450 and T[2, -1] == 510                  # latency 15 hops, then 1 per 30 s
    assert abs(tg.regenerator_ber(17, 1000) - 7.2e-10) < 0.3e-10
    assert abs(tg.loading_cutoff(88e-3, 0.0516e-6, 1.829) - 3493) < 5


def test_pcm():
    from commlib import pcm
    n = np.arange(1 << 15)
    x = 0.999 * np.sin(2 * np.pi * 0.0123456 * n)
    for b in (8, 12):
        assert abs(pcm.sqnr_db(x, pcm.quantize(x, b)) - (6.02 * b + 1.76)) < 0.5
    xi = np.arange(-8159, 8160, 7)
    y = pcm.g711_mu_decode(*pcm.g711_mu_encode(xi))
    assert np.max(np.abs(y - xi)) <= 128                         # largest chord step / 2
    xa = np.arange(-4095, 4096, 5)
    assert np.max(np.abs(pcm.g711_a_decode(*pcm.g711_a_encode(xa)) - xa)) <= 64
    assert np.allclose(pcm.imulaw(pcm.mulaw(x)), x) and np.allclose(pcm.ialaw(pcm.alaw(x)), x)
    assert pcm.alias_frequency(7000, 8000) == 1000 and pcm.nyquist_zone(7000, 8000) == 2
    b, a = pcm.ntf_coeffs(2)
    v = pcm.sigma_delta(0.5 * np.sin(2 * np.pi * 5 / 8192 * np.arange(8192)), b, a)
    assert set(np.unique(v)) <= {-1.0, 1.0} and abs(np.mean(v)) < 0.01
    _, est = pcm.delta_mod(np.zeros(100), 0.1)
    assert np.max(np.abs(est)) <= 0.1 + 1e-12                    # idles between ±δ
    sf, fb = pcm.t1_superframe()
    assert sf.shape == (12, 193) and np.sum(sf == 2) == 48 and fb.sum() == 6


def test_superhet():
    from commlib import superhet as sh
    assert abs(sh.irr_db(1000e3, 1910e3, 40, 1) - 34.8) < 0.2   # Chapter 4 worked example (35 dB)
    assert abs(abs(sh.butter_lp(np.array([1000.0]), 1000.0, 5))[0] - 2 ** -0.5) < 1e-9
    fs, n = 256e3, 16384
    st = [sh.Station(1000e3, -70, "tone", tone=1000, seed=5)]
    zb, za, info = sh.if_signal(st, 1455e3, 455e3, fs, n, 1000e3, 30, 1, 9e3, 6, rng=1)
    audio, dc = sh.detect_am(za, fs)
    assert sh.tone_sinad(audio, 1000, fs) > 35 and info[0]["offset"] == 0
    env = np.r_[np.ones(4000), 10 * np.ones(4000)]
    out, g, _ = sh.agc_loop(env, 4000.0, 0.01, 0.02, 0.1)
    assert abs(np.mean(out[-500:]) - 1) < 0.05 and abs(g[-1] + 20) < 1   # 20 dB taken out


def test_fectools():
    from commlib import fectools as ft
    rng = np.random.default_rng(5)
    cc = cl.ConvCode()
    u = rng.integers(0, 2, (20, 60))
    c = cc.encode_batch(u)
    s = ft.bpsk_sigma(3.0, 0.5)
    L = 2 * (1 - 2.0 * c + s * rng.standard_normal(c.shape)) / s ** 2
    assert np.array_equal(ft.viterbi_window(cc, L)[:, :60], cc.decode_batch(L))
    assert ft.viterbi_window(cc, 1 - 2.0 * c, 35)[:, :60].tolist() == u.tolist()
    d, A, B = ft.distance_spectrum(cc, 14)
    assert d[0] == 10 and A[0] == 11 and B[0] == 36
    assert [ft.punctured_dfree(cc, p) for p in ([1, 1], [1, 1, 1, 0], [1, 1, 1, 0, 0, 1])] == [10, 6, 5]
    c3 = cl.ConvCode(3, (0o7, 0o5))
    rx = c3.encode([1, 0, 1, 1]).reshape(-1, 2)
    rx[1, 1] ^= 1
    tr = ft.viterbi_trace(c3, rx)
    assert tr["bits"][:4].tolist() == [1, 0, 1, 1] and tr["pm"][-1, 0] == 1
    code = cl.LDPCCode(96, 0.5, 3, seed=0)
    bp = ft.TannerBP(code.H)
    cw = code.encode(rng.integers(0, 2, code.k))
    for m in ("spa", "ms", "nms", "oms"):
        h, it = bp.decode(ft.bpsk_llr(cw[None], 6.0, 0.5, rng), 30, m)
        assert np.array_equal(h[0], cw), m
    assert ft.count_4cycles(code.H) == 0 and ft.tanner_girth(code.H) >= 6
    assert np.allclose(ft.polar_bec_z(32, 0.3), cl.PolarCode._bhat(32, 0.3))
    assert abs(ft.biawgn_limit_db(0.5) - 0.187) < 0.01
    assert abs(float(ft.J_fast(2.0)) - 0.4824) < 0.01
    e = np.array([0b100000111 << 5, 0b101], dtype=np.uint64)
    assert ft.gf2_mod_many(e, 0b100000111).tolist() == [0, 5]
    pi = ft.s_random_interleaver(64, 4, rng)
    assert sorted(pi.tolist()) == list(range(64))


def test_spectral():
    from commlib import spectral as sp
    m = {n: sp.window_metrics(sp.window(n, 256)) for n in sp.WINDOWS}
    assert abs(m["Rectangular"]["scallop"] - 3.92) < 0.01 and abs(m["Hann"]["enbw"] - 1.5) < 1e-6
    assert abs(m["Hann"]["sidelobe"] + 31.5) < 0.2 and m["Blackman–Harris"]["sidelobe"] < -90
    assert m["Flat-top"]["scallop"] < 0.02                           # Chapter 2's scalloping figures
    c0, A, ph = sp.fourier_coeffs("Square", 199)
    x = sp.fourier_sum(np.linspace(0, 1, 4001), c0, A, ph)
    assert abs((x.max() - 1) / 2 - 0.0895) < 0.003                   # Gibbs: 8.95 % of the jump
    assert abs(sp.fourier_sum([0.25], *sp.fourier_coeffs("Triangle", 99))[0] - 1) < 0.01
    assert sp.alias_complex(700e3, 1e6) == -300e3 and sp.alias_real(-700e3, 1e6) == 300e3
    rng = np.random.default_rng(3)
    z = rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)
    y = cl.iq_imbalance(z, 1.0, 6.0) + 0.1
    yc, g, p = sp.blind_iq_correct(y)
    assert abs(g - 1.0) < 0.05 and abs(p - 6.0) < 0.5
    mu, nu, _ = sp.iq_fit(yc, z)
    assert 10 * np.log10(abs(mu) ** 2 / abs(nu) ** 2) > 40
    tone = 0.999 * np.exp(2j * np.pi * 811 / 16384 * np.arange(16384))
    e = sp.quantize(tone, 10) - tone
    assert abs(10 * np.log10(np.mean(np.abs(tone) ** 2) / np.mean(np.abs(e) ** 2)) - 61.96) < 0.5
    t = (np.arange(1 << 15) - (1 << 14)) / 256
    f = np.fft.fftshift(np.fft.fftfreq(1 << 15, 1 / 256))
    g_ = np.exp(-np.pi * t ** 2)
    st_, sf_ = sp.rms_widths(t, g_, f, np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g_))))
    assert abs(4 * np.pi * st_ * sf_ - 1) < 1e-3                    # the Gaussian meets the limit
    sym = rng.choice([-1.0, 1.0], 300 * 32)
    up = np.zeros(len(sym) * 8)
    up[::8] = sym
    s = np.convolve(up, cl.rrc_taps(0.35, 8, 8))[:300 * 256] * np.exp(2j * np.pi * 10 * np.arange(300 * 256) / 256)
    a, nc, cj, P = sp.spectral_coherence(s, 256)
    assert abs(abs(a[np.nanargmax(nc)]) - 1 / 8) < 1e-9 and abs(a[np.argmax(cj)] - 20 / 256) < 1e-9


def test_noise():
    from commlib import noise as nz
    assert abs(nz.Qinv(1e-12) - 7.034) < 0.01 and abs(nz.Q(3.0) - 1.35e-3) < 1e-5
    st_ = [("switch", -0.5, 0.5), ("SAW", -1.5, 1.5), ("LNA", 18, 1.0), ("mixer", 8, 10),
           ("BB", 30, 12), ("ADC", 0, 27)]
    assert abs(nz.friis(st_)["nf"] - 3.58) < 0.01                    # Chapter 3's handset line-up
    assert abs(nz.butterworth_neb_ratio(2) - 1.1107) < 1e-3
    fr = np.linspace(0, 1000, 200001)
    assert abs(nz.neb_hz(fr, 1 / (1 + (fr / 50) ** 2)) / 50 - np.arctan(20)) < 1e-4   # RC to 1 kHz
    F = nz.yfactor_f(10 ** 1.35, 15.0, 8.0, 20.0)
    assert abs(10 * np.log10(F) - 1.54) < 0.02                        # Chapter 3's Y-factor example
    assert abs(nz.pd_coherent(1e-6, 10 ** 1.27) - nz.Q(nz.Qinv(1e-6) - np.sqrt(2 * 10 ** 1.27))) < 1e-12
    assert 0.9 < nz.pd_coherent(1e-6, 10 ** 1.27) < 0.92               # radar spec met at ~12.7 dB
    assert nz.pd_envelope(1e-6, 10 ** 1.27) < nz.pd_coherent(1e-6, 10 ** 1.27)
    assert nz.pd_energy(1e-6, 10 ** 1.27, 16) < nz.pd_envelope(1e-6, 10 ** 1.27)
    r = np.linspace(0, 6, 60001)
    assert abs(np.trapezoid(nz.rice_pdf(r, 3.0), r) - 1) < 1e-4
    assert abs(nz.rice_cdf(np.sqrt(0.1), 0.0) - (1 - np.exp(-0.1))) < 1e-6


def test_multirate():
    from scipy import signal as sg
    from commlib import multirate as mr
    r = np.random.default_rng(17)
    x = r.standard_normal(1003) + 1j * r.standard_normal(1003)
    h = sg.firwin(61, 0.1)
    for M in (2, 3, 8):                                   # polyphase = filter then keep every M-th
        a, b = np.convolve(x, h)[::M], mr.polyphase_decimate(x, h, M)
        assert len(a) == len(b) and np.max(np.abs(a - b)) < 1e-12
    up = np.zeros(4 * len(x), complex)
    up[::4] = x
    a, b = np.convolve(up, h), mr.polyphase_interpolate(x, h, 4)
    n = min(len(a), len(b))
    assert np.max(np.abs(a[:n] - b[:n])) < 1e-12
    xi = r.integers(-2 ** 11, 2 ** 11, 4000)              # Hogenauer: wrap-around is harmless...
    B = mr.cic_bits(12, 16, 4)
    assert B == 28
    assert np.array_equal(mr.cic_decimate(xi, 16, 4), mr.cic_decimate(xi, 16, 4, width=B))
    assert mr.cic_decimate(np.full(800, 1000), 16, 4)[-1] == 1000 * 16 ** 4
    assert abs(-20 * np.log10(mr.cic_response(0.2 / 16, 16, 4)) - 2.3) < 0.05   # Ch. 6 droop
    z, ftw, fa = mr.nco(1 << 14, 0.1234567, 32, 12, 16)
    assert abs(mr.nco_sfdr(z)[0] - 72) < 2                # about 6 dB per phase bit
    th = r.uniform(-np.pi, np.pi, 2000)
    c, s = mr.cordic(th, 24)
    assert np.max(np.abs(np.angle(np.exp(1j * (np.arctan2(s, c) - th))))) < 1e-6
    c, s = mr.cordic(th, 16, bits=18)                     # an 18-bit datapath stalls near 1.5e-4
    assert np.max(np.abs(np.angle(np.exp(1j * (np.arctan2(s, c) - th))))) > 1e-4
    assert abs(mr.cordic_gain(30) - 1.64676) < 1e-4
    assert mr.quantize_coefs([0.3, -0.7], 4).tolist() == [0.25, -0.75]


def test_serdes():
    from commlib import serdes as sd
    from commlib import linecodes as lc
    bits = np.r_[1, np.zeros(12, int), 1, 1, np.zeros(8, int)]
    s, v = sd.b8zs(bits)
    assert v.sum() == 4                                                # two 8-zero substitutions
    z = lc.run_lengths(np.abs(s))
    assert z.max() <= 7                                                # B8ZS: at most 7 zeros
    h, hv = sd.hdb3(bits)
    zr = [len(t) for t in "".join(str(abs(q)) for q in h).split("1") if t]
    assert max(zr) <= 3                                                # HDB3: at most 3 zeros
    assert set(np.unique(sd.hdb3(np.zeros(40, int))[0])) <= {-1, 0, 1}
    assert abs(np.sum(sd.hdb3(np.zeros(400, int))[0])) <= 2          # violations alternate: no DC
    line = sd.enc64b66b(np.zeros(640, np.int8))
    assert len(line) == 660 and np.all(line[::66] == 0) and np.all(line[1::66] == 1)
    tj = sd.total_jitter(1e-12, 0.018, 0.15)
    assert abs(tj - 0.40) < 0.01                                       # Chapter 8's 25G budget
    x = np.linspace(0, 1, 2001)
    ber = sd.dual_dirac_ber(x, 0.018, 0.15)
    w = x[ber < 1e-12]
    assert abs((w.max() - w.min()) - (1 - tj)) < 0.01
    assert abs(20 * np.log10(abs(sd.ctle_response(np.array([1.0]), 10, 1.0)[0])) - 10) < 1e-6
    d = np.array([1, 0, 1, 1, 0, 0, 1])                                # Chapter 8's worked example
    b = sd.duobinary_precode(d)
    assert b.tolist() == [1, 1, 0, 1, 1, 1, 0]
    y = sd.partial_response(2.0 * b - 1, "1+D")
    assert y.tolist() == [0, 2, 0, 0, 2, 2, 0]
    assert sd.duobinary_decode_precoded(y).tolist() == d.tolist()


def test_frontier():
    """commlib.frontier reproduces Chapter 25's worked numbers."""
    from commlib import frontier as fr
    fr._selftest()
    assert abs(sum(fr.gas_atten_db_km(140.0)) - 1.07) < 0.1           # 'about 1 dB/km at 140 GHz'
    assert 4 < sum(fr.gas_atten_db_km(300.0)) < 7                      # 'about 5 dB/km at 300 GHz'


def test_airif():
    from commlib import airif as ai
    assert abs(ai.nr_peak_rate(4, 8, 273, 1, 0.14) / 1e9 - 2.34) < 0.01     # Chapter 21 FR1 example
    assert abs(ai.nr_peak_rate(2, 6, 264, 3, 0.18) / 1e9 - 3.23) < 0.01     # Chapter 21 FR2 example
    assert ai.NR_MAX_PRB[(1, 100)] == 273 and ai.NR_MAX_PRB[(3, 400)] == 264
    assert abs(ai.spectral_efficiency(ai.LTE_CQI)[-1] - 5.5547) < 1e-3
    assert len(ai.NR_MCS64) == 29
    assert abs(ai.wifi_rate_mbps(11, 80) - 600.25) < 0.01
    assert abs(ai.combine_snr_db(38, 40) - 35.88) < 0.01
    q, t = ai.ble_adv_charge()
    assert abs(q * 1e6 - 10.67) < 0.01 and abs(t * 1e6 - 376) < 1e-6        # Chapter 22 beacon
    assert abs(ai.battery_life_years(q, 1.0) - 2.03) < 0.01
    assert abs(ai.bler_logistic(7.0, 7.0) - 0.1) < 1e-9
    req = ai.required_snr_db(ai.spectral_efficiency(ai.LTE_CQI))
    assert np.all(np.diff(req) > 0) and ai.select_entry(-20, req) == -1
    hi = ai.harq_throughput(np.array([0.0]), 3.0, scheme="ir")[0]
    lo = ai.harq_throughput(np.array([0.0]), 3.0, scheme="chase")[0]
    assert hi > lo > ai.harq_throughput(np.array([0.0]), 3.0, scheme="none")[0]
    slot, pl, Ts, Tc = ai.dcf_times()
    assert slot == 9e-6 and Ts > pl > 0 and Tc > 0


def test_mimokit():
    from commlib import mimokit as mk
    g = 10 ** (np.array([5.0, 15.0, 25.0]) / 10)
    # Craig/MGF MRC with equal eigenvalues equals the closed form; SC integral is stable
    assert np.allclose(mk.ber_bpsk_mrc_eig(g, np.ones(2)), mk.ber_bpsk_mrc_iid(g, 2), rtol=1e-6)
    assert 0 < mk.ber_bpsk_sc_iid(10 ** 3.5, 8) < 1e-20
    # sample-based BER with the exact tail matches theory for MRC
    h = mk.correlated_branches(2, 60000, 0.5, np.random.default_rng(1))
    b = mk.ber_bpsk_from_gains(mk.combine_gain(h, "mrc"), g,
                               mk.gain_cdf_asymptote("mrc", mk.exp_corr(2, 0.5)), 2)
    assert np.all(np.abs(np.log10(b / mk.ber_bpsk_mrc_corr(g, 2, 0.5))) < 0.25)   # within ~1.8x
    # water-filling agrees with commlib.mimo.waterfill
    p, mu, k = mk.waterfill_level([2.0, 1.0, 0.1], 1.0)
    assert np.allclose(p, cl.waterfill([2.0, 1.0, 0.1], 1.0)) and k == 2 and abs(mu - 1.25) < 1e-9
    # batched detectors agree with the single-vector ones
    qp = cl.get_constellation("qpsk").points
    r = np.random.default_rng(2)
    H = cl.rayleigh_mimo(2, 2, n=50, rng=r)
    X = qp[r.integers(0, 4, (50, 2))]
    Y = np.einsum("nij,nj->ni", H, X) + 0.1 * (r.standard_normal((50, 2)) + 1j * r.standard_normal((50, 2)))
    assert np.all(mk.batch_ml(H, Y, mk.ml_candidates(qp, 2)) ==
                  np.array([cl.ml_detect(H[i], Y[i], qp) for i in range(50)]))
    assert np.all(mk.batch_mmse_sic(H, Y, 0.02, qp) ==
                  np.array([cl.mmse_sic_detect(H[i], Y[i], 0.02, qp) for i in range(50)]))
    # DMT corners and LOS-MIMO orthogonality at the Rayleigh spacing
    assert mk.dmt_value(1.0, 2, 2) == 1.0 and mk.dmt_value(0.0, 4, 4) == 16
    lam = 3e8 / 80e9
    sv = np.linalg.svd(mk.los_channel(2, mk.los_opt_spacing(2, 1000, lam), 1000, lam), compute_uv=False)
    assert abs(sv[0] - sv[1]) < 1e-3
    # array factor: uniform 16-element array has 12 dB of gain and -13.3 dB sidelobes
    th = np.radians(np.linspace(-90, 90, 3601))
    af = 10 * np.log10(mk.ula_af(np.ones(16), th))
    _, hp, psl = mk.beam_metrics(th, af)
    assert abs(af.max() - 10 * np.log10(16)) < 0.01 and abs(psl + 13.2) < 0.3


def test_cellsim():
    from commlib import cellsim as cs, cellular as cel
    # Erlang occupancy: the last state is Erlang B
    assert abs(cs.erlang_occupancy(10.0, 10)[-1] - cel.erlang_b(10.0, 10)) < 1e-12
    # SIR samples reproduce the book: N = 7, n = 4, 8 dB -> about 60 % above 18 dB
    v = cs.sir_samples(2, 1, 4.0, 8.0, 1, 12000, np.random.default_rng(7))
    assert 0.55 < np.mean(v >= 18) < 0.68
    # slotted-ALOHA event simulation near 1/e at G = 1
    _, ok = cs.aloha_events(1.0, 20000, True, np.random.default_rng(3))
    assert abs(ok.sum() / 20000 - 1 / np.e) < 0.02
    # handovers: no protection gives many, hysteresis + TTT gives few
    r = np.random.default_rng(31)
    a = cs.handover_drive(0.0, 0.0, r)["count"]
    b = cs.handover_drive(4.0, 19.2, np.random.default_rng(31))["count"]
    assert a > 10 and b <= 4
    # RACH: barring rescues a 30 000-device burst
    s1 = cs.rach_sim(30000, 10, True, acb=1.0)["success"]
    s2 = cs.rach_sim(30000, 10, True, acb=0.3)["success"]
    assert s1 < 0.6 and s2 > 0.95


def test_sourcekit():
    from commlib import sourcekit as sk
    # Lloyd-Max for N(0,1): the classic table (Max 1960): 1 bit 4.40 dB, 2 bits 9.30 dB
    assert abs(10 * np.log10(1 / sk.lloyd_max(2)[1]) - 4.40) < 0.02
    assert abs(10 * np.log10(1 / sk.lloyd_max(4)[1]) - 9.30) < 0.02
    # uniform pdf: the uniform quantiser is optimal, 6.02 dB per bit
    assert abs(10 * np.log10(1 / sk.uniform_quantizer(8, "uniform")[1]) - 18.06) < 0.05
    # entropy-coded uniform quantiser approaches the 1.53 dB gap at high rate
    R, mse = sk.ecsq(0.05)
    assert abs(sk.gaussian_rd_snr_db(R) - 10 * np.log10(1 / mse) - 1.53) < 0.05
    D, Rk, th = sk.reverse_waterfill([4, 2, 1, 0.5, 0.25], 1.0)
    assert np.allclose(D, 0.2) and abs(th - 0.2) < 1e-6
    text = "abcabcabcabc xyz abcabc"
    toks = sk.lz77_parse(text, 64, 18)
    rebuilt = []
    for pos, L, d in toks:                               # decode the parse
        for _ in range(L):
            rebuilt.append(text[pos] if d == 0 else rebuilt[len(rebuilt) - d])
    assert "".join(rebuilt) == text and any(d for _, _, d in toks)
    r = np.random.default_rng(0)
    a = r.random((64, 64)) * 255
    b = np.roll(a, (2, 3), (0, 1))
    mv, pred, _ = sk.block_motion(b, a, 16, 4)
    assert tuple(mv[1, 1]) == (-2, -3) and np.allclose(pred[16:48, 16:48], b[16:48, 16:48])
    blk = r.random((8, 8))
    assert np.allclose(sk.idct8(sk.dct8(blk)), blk) and len(sk.zigzag_indices(8)) == 64


def test_spreadkit():
    from commlib import spreadkit as sk
    from commlib import spread as sp
    # partial-band jamming of FH/BFSK: worst case rho = 2/(Eb/J0), BER = e^-1/(Eb/J0)
    rho, ber = sk.fh_worst_rho(20.0, 1)
    assert abs(rho - 0.02) < 0.002 and abs(ber - np.exp(-1) / 100) < 3e-4
    assert sk.fh_worst_rho(20.0, 5)[1] < 1e-4                 # diversity restores exponential decay
    # tracking loop: locks and tracks at 45 dB-Hz with small code and phase error
    loop = sk.TrackingLoop(cn0=45, dll_bn=2, pll_bn=15, fll_bn=10, freq_err0=100, seed=1)
    o = loop.run(1200)
    pe = (o["phase_err"][-400:] + np.pi / 2) % np.pi - np.pi / 2
    assert np.std(o["code_err"][-400:]) < 0.02 and np.degrees(np.sqrt(np.mean(pe ** 2))) < 10
    # geometry helpers: satellites at the requested elevations, linearised errors ~ DOP x sigma
    u = sp.lla_to_ecef(45.0, -75.0, 100.0)
    S = sk.sats_from_azel(u, [0, 90, 180, 270, 45], [20, 30, 40, 50, 85])
    az, el = sp.azel(S.T, u)
    assert np.allclose(el, [20, 30, 40, 50, 85], atol=1e-6)
    enu, _ = sk.fix_cloud(S, u, 1.0, 20000, rng=2)
    assert abs(np.sqrt(np.mean(np.sum(enu ** 2, axis=1))) / sp.dop(S, u)["PDOP"] - 1) < 0.03


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    print(f"{failed} failure(s)" if failed else "all tests passed")
    sys.exit(1 if failed else 0)
