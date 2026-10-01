"""commlib.ltephy -- a compact LTE/NR downlink physical layer for simulation.

Companion module for Chapter 21 (Cellular Generations) and Lab 27. Import explicitly:
``from commlib import ltephy as lp``.

What is standard and what is simplified
---------------------------------------
* Frame structure, PSS (Zadoff-Chu, roots 25/29/34), SSS (interleaved m-sequences), the
  cell-specific reference signals of antenna port 0 and the length-31 Gold scrambling sequence
  follow 3GPP TS 36.211 (Sections 6.10.1, 6.11 and 7.2) for a 1.4 MHz carrier (6 resource
  blocks, 128-point FFT at 1.92 MS/s, normal cyclic prefix).
* CRC24A / CRC24B and code-block segmentation follow TS 36.212 / 38.212 in spirit (one CRC on the
  transport block, one per code block).
* Channel coding uses commlib's PEG LDPC code (not the NR base graphs) with an NR-style circular
  buffer (systematic bits first) and the four NR redundancy-version starting points
  k0 = {0, 17, 33, 56}/66 of the buffer (TS 38.212 Table 5.4.2.1-2, base graph 1).
* ``ldpc_decode_batch`` is a batched normalised min-sum decoder for ``commlib.LDPCCode``.

All grids are (72 subcarriers, 14 OFDM symbols) for one 1 ms subframe; subcarrier index 0 is
the lowest frequency (-36 x 15 kHz) and DC is not part of the grid, as in LTE.
"""
from __future__ import annotations

import functools

import numpy as np

from .coding import crc_remainder

__all__ = ["N_RB", "N_SC", "NFFT", "FS", "CP", "SF_LEN", "PSS_ROOTS", "CRC24A", "CRC24B",
           "gold_sequence", "pss_sequence", "sss_sequence", "sync_subcarriers", "crs_port0",
           "subframe_grid", "ofdm_mod", "ofdm_demod", "pss_waveforms", "pss_search",
           "sss_detect", "crc_attach", "crc_ok", "rate_match", "rate_recover",
           "rv_start", "bit_interleave", "bit_deinterleave", "ldpc_decode_batch",
           "pdsch_re_mask", "crs_channel_estimate"]

N_RB, N_SC = 6, 72                     # 1.4 MHz carrier
NFFT, FS = 128, 1.92e6                 # 15 kHz subcarriers
CP = np.array([10, 9, 9, 9, 9, 9, 9] * 2)
SF_LEN = int(np.sum(CP) + 14 * NFFT)   # 1920 samples = 1 ms
PSS_ROOTS = (25, 29, 34)
CRC24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]  # 0x1864CFB
CRC24B = [1, 1] + [0] * 16 + [1, 1, 0, 0, 0, 1, 1]  # 0x1800063
_SIGNED = np.r_[np.arange(-36, 0), np.arange(1, 37)]        # grid row -> signed subcarrier
_BINS = np.mod(_SIGNED, NFFT)


# ----------------------------------------------------------------------------- sequences
def gold_sequence(c_init, n):
    """Length-31 Gold pseudo-random sequence c(0..n-1) of TS 36.211 Sec. 7.2 (Nc = 1600)."""
    return _gold(int(c_init), int(n)).copy()


@functools.lru_cache(maxsize=4096)
def _gold(c_init, n):
    Nc = 1600
    L = n + Nc + 31
    x1 = np.zeros(L, np.int8); x2 = np.zeros(L, np.int8)
    x1[0] = 1
    x2[:31] = [(int(c_init) >> i) & 1 for i in range(31)]
    for i in range(L - 31):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return (x1[Nc:Nc + n] ^ x2[Nc:Nc + n]).astype(np.int8)


def pss_sequence(nid2):
    """62-element PSS d_u(n), u = 25, 29, 34 for N_ID^(2) = 0, 1, 2 (TS 36.211 6.11.1.1)."""
    u = PSS_ROOTS[nid2]
    n = np.arange(62)
    m = np.where(n < 31, n * (n + 1), (n + 1) * (n + 2))
    return np.exp(-1j * np.pi * u * m / 63)


def _msequence(taps):
    x = np.zeros(31, np.int8); x[4] = 1
    for i in range(26):
        x[i + 5] = np.bitwise_xor.reduce(x[[i + t for t in taps]])
    return 1 - 2 * x.astype(int)


_S, _C, _Z = _msequence([2, 0]), _msequence([3, 0]), _msequence([4, 2, 1, 0])


def sss_sequence(nid1, nid2, subframe=0):
    """62-element SSS of TS 36.211 6.11.2.1 for subframe 0 or 5 (values +-1)."""
    qp = nid1 // 30
    q = (nid1 + qp * (qp + 1) // 2) // 30
    mp = nid1 + q * (q + 1) // 2
    m0 = mp % 31
    m1 = (m0 + mp // 31 + 1) % 31
    n = np.arange(31)
    s0, s1 = _S[(n + m0) % 31], _S[(n + m1) % 31]
    c0, c1 = _C[(n + nid2) % 31], _C[(n + nid2 + 3) % 31]
    z0, z1 = _Z[(n + m0 % 8) % 31], _Z[(n + m1 % 8) % 31]
    d = np.empty(62)
    if subframe == 0:
        d[0::2], d[1::2] = s0 * c0, s1 * c1 * z0
    else:
        d[0::2], d[1::2] = s1 * c0, s0 * c1 * z1
    return d


def sync_subcarriers():
    """Grid rows of the 62 PSS/SSS subcarriers (k = n - 31 relative to DC, DC excluded)."""
    return np.r_[np.arange(5, 36), np.arange(36, 67)]


def crs_port0(pci, ns, l):
    """(rows, values) of the port-0 cell-specific reference signal in OFDM symbol l (0 or 4)
    of slot ns (TS 36.211 6.10.1, normal CP)."""
    c_init = 2 ** 10 * (7 * (ns + 1) + l + 1) * (2 * pci + 1) + 2 * pci + 1
    c = gold_sequence(c_init, 4 * 110)
    m = np.arange(2 * N_RB) + 110 - N_RB
    r = ((1 - 2.0 * c[2 * m]) + 1j * (1 - 2.0 * c[2 * m + 1])) / np.sqrt(2)
    v = 0 if l == 0 else 3
    rows = 6 * np.arange(2 * N_RB) + (v + pci % 6) % 6
    return rows, r


def pdsch_re_mask(pci, subframe, cfi=2):
    """Boolean (72, 14) mask of resource elements free for PDSCH (outside control, CRS, PSS/SSS
    and the reserved subcarriers around them; PBCH is not modelled)."""
    M = np.ones((N_SC, 14), bool)
    M[:, :cfi] = False
    for ns_ in (0, 1):
        for l in (0, 4):
            rows, _ = crs_port0(pci, 2 * subframe + ns_, l)
            M[rows, 7 * ns_ + l] = False
    if subframe in (0, 5):
        M[:, 5:7] = False
    return M


def subframe_grid(pci, subframe, rng, cfi=2, data=None, load=1.0):
    """One subframe of a cell: QPSK filler (or ``data`` on the PDSCH mask), port-0 CRS, and
    PSS/SSS in subframes 0 and 5. Returns (grid, labels) where labels marks the RE type:
    0 PDSCH, 1 control, 2 CRS, 3 PSS, 4 SSS, 5 reserved/empty."""
    G = np.zeros((N_SC, 14), complex)
    lab = np.full((N_SC, 14), 5, int)
    qpsk = lambda n: ((1 - 2.0 * rng.integers(0, 2, n)) + 1j * (1 - 2.0 * rng.integers(0, 2, n))) / np.sqrt(2)
    on = rng.random((N_SC, 14)) < load
    G[:, :cfi] = qpsk(N_SC * cfi).reshape(N_SC, cfi); lab[:, :cfi] = 1
    M = pdsch_re_mask(pci, subframe, cfi)
    if data is None:
        G[M & on] = qpsk(np.sum(M & on))
    else:
        G[M] = data
    lab[M] = 0
    for ns_ in (0, 1):
        for l in (0, 4):
            rows, r = crs_port0(pci, 2 * subframe + ns_, l)
            G[rows, 7 * ns_ + l] = r; lab[rows, 7 * ns_ + l] = 2
    if subframe in (0, 5):
        k = sync_subcarriers()
        G[:, 5:7] = 0; lab[:, 5:7] = 5
        G[k, 6] = pss_sequence(pci % 3); lab[k, 6] = 3
        G[k, 5] = sss_sequence(pci // 3, pci % 3, subframe); lab[k, 5] = 4
    return G, lab


# ----------------------------------------------------------------------------- OFDM
def ofdm_mod(grids):
    """(n_sf, 72, 14) or (72, 14) grids -> time samples with LTE normal cyclic prefixes."""
    grids = np.asarray(grids)
    grids = grids[None] if grids.ndim == 2 else grids
    out = []
    for G in grids:
        X = np.zeros((14, NFFT), complex)
        X[:, _BINS] = G.T
        x = np.fft.ifft(X, axis=1) * np.sqrt(NFFT)
        out += [np.r_[x[l, -CP[l]:], x[l]] for l in range(14)]
    return np.concatenate(out)


def ofdm_demod(x, n_sf=1):
    """Time samples (starting at a subframe boundary) -> (n_sf, 72, 14) grids."""
    out = np.zeros((n_sf, N_SC, 14), complex)
    for s in range(n_sf):
        p = s * SF_LEN
        for l in range(14):
            p += CP[l]
            out[s, :, l] = (np.fft.fft(x[p:p + NFFT]) / np.sqrt(NFFT))[_BINS]
            p += NFFT
    return out[0] if n_sf == 1 else out


# ----------------------------------------------------------------------------- cell search
def pss_waveforms():
    """Time-domain (128-sample, no CP) PSS symbols for N_ID^(2) = 0, 1, 2."""
    k = sync_subcarriers()
    out = []
    for nid2 in range(3):
        X = np.zeros(NFFT, complex)
        X[_BINS[k]] = pss_sequence(nid2)
        out.append(np.fft.ifft(X) * np.sqrt(NFFT))
    return np.array(out)


def pss_search(r, halves=2):
    """Correlate r against the three PSS waveforms.  The correlation is split into ``halves``
    coherent segments combined non-coherently, which tolerates a frequency offset of several
    kHz.  Returns (metric (3, n), seg (3, halves, n)) with metric[i, t] the energy for a PSS
    starting at sample t and seg the complex segment correlations (for CFO estimation)."""
    P = pss_waveforms()
    L = NFFT // halves
    n = len(r) - NFFT + 1
    seg = np.zeros((3, halves, n), complex)
    nf = 1 << int(np.ceil(np.log2(len(r) + NFFT)))
    R = np.fft.fft(r, nf)
    for i in range(3):
        for h in range(halves):
            p = np.zeros(NFFT, complex); p[h * L:(h + 1) * L] = P[i, h * L:(h + 1) * L]
            c = np.fft.ifft(R * np.conj(np.fft.fft(p, nf)))
            seg[i, h] = c[:n]
    metric = np.sum(np.abs(seg) ** 2, axis=1)
    norm = np.convolve(np.abs(r) ** 2, np.ones(NFFT), "valid")[:n] + 1e-12
    return metric / norm, seg


def sss_detect(r, t_pss, nid2, cfo_hz=0.0, smooth=31):
    """Detect N_ID^(1) and the half-frame (subframe 0 or 5) from the SSS one symbol before a
    PSS whose useful part starts at sample ``t_pss``. The PSS gives the channel estimate,
    smoothed by a moving average over ``smooth`` subcarriers (1 = raw per-subcarrier LS).
    Returns (nid1, subframe, metric array (2, 168))."""
    t = np.arange(len(r))
    rc = r * np.exp(-2j * np.pi * cfo_hz * t / FS)
    k = _BINS[sync_subcarriers()]
    Yp = (np.fft.fft(rc[t_pss:t_pss + NFFT]) / np.sqrt(NFFT))[k]
    ts = t_pss - NFFT - CP[6]
    Ys = (np.fft.fft(rc[ts:ts + NFFT]) / np.sqrt(NFFT))[k]
    H = Yp / pss_sequence(nid2)
    if smooth > 1:
        H = np.convolve(H, np.ones(smooth), "same") / np.convolve(np.ones(62), np.ones(smooth), "same")
    z = Ys * np.conj(H)
    m = np.array([[np.real(np.sum(z * sss_sequence(n1, nid2, sf))) for n1 in range(168)] for sf in (0, 5)])
    sf_i, n1 = np.unravel_index(np.argmax(m), m.shape)
    return int(n1), (0, 5)[sf_i], m


# ----------------------------------------------------------------------------- coding chain
def crc_attach(bits, poly=CRC24A):
    return np.r_[np.asarray(bits, np.int8), crc_remainder(bits, poly)].astype(np.int8)


def crc_ok(bits, poly=CRC24A):
    return not crc_remainder(np.asarray(bits, np.int8), poly).any()


def rv_start(rv, Ncb):
    """NR-style starting position k0 of redundancy version rv in a circular buffer of Ncb bits."""
    return int(np.floor((0, 17, 33, 56)[rv] * Ncb / 66))


def rate_match(cbuf, E, rv):
    """Read E bits from the circular buffer starting at k0(rv), wrapping (repetition) if E > Ncb."""
    Ncb = len(cbuf)
    return np.asarray(cbuf)[(rv_start(rv, Ncb) + np.arange(E)) % Ncb]


def rate_recover(llr, Ncb, rv, soft=None):
    """Accumulate received LLRs into a soft circular buffer (incremental redundancy / chase
    combining happen automatically: repeated positions add)."""
    soft = np.zeros(Ncb) if soft is None else soft
    np.add.at(soft, (rv_start(rv, Ncb) + np.arange(len(llr))) % Ncb, llr)
    return soft


def bit_interleave(e, Qm):
    """NR row-column bit interleaver (TS 38.212 5.4.2.2): write in Qm rows, read by columns."""
    return np.asarray(e).reshape(Qm, -1).T.reshape(-1)


def bit_deinterleave(f, Qm):
    return np.asarray(f).reshape(-1, Qm).T.reshape(-1)


def ldpc_decode_batch(code, llr, iters=25, alpha=0.8):
    """Batched normalised min-sum for a ``commlib.LDPCCode``. llr: (B, n). Returns (bits (B, n),
    iterations used per frame). Frames stop updating once their syndrome is satisfied."""
    Lch = np.atleast_2d(np.asarray(llr, float))
    B = Lch.shape[0]
    slot, valid = code.slot, code.slot >= 0
    sl = np.maximum(slot, 0)
    r = np.zeros((B, code.E))
    L = Lch.copy()
    done = np.zeros(B, bool)
    used = np.full(B, iters)
    Ht = code.H.T.astype(np.float32)
    ar = np.arange(code.dcmax)[None, None, :]
    perm = np.argsort(code.edge_v, kind="stable")                 # edges grouped by variable node
    starts = np.searchsorted(code.edge_v[perm], np.arange(code.n))
    for it in range(1, iters + 1):
        act = ~done
        q = L[act][:, code.edge_v] - r[act]
        Q = q[:, sl]
        mag = np.where(valid, np.abs(Q), np.inf)
        sgn = np.where(valid, np.where(Q < 0, -1.0, 1.0), 1.0)
        tot = np.prod(sgn, axis=2, keepdims=True)
        i1 = np.argmin(mag, axis=2)
        m1 = np.take_along_axis(mag, i1[..., None], 2)
        mag2 = mag.copy()
        np.put_along_axis(mag2, i1[..., None], np.inf, 2)
        m2 = mag2.min(axis=2, keepdims=True)
        R = alpha * tot * sgn * np.where(ar == i1[..., None], m2, m1)
        rn = np.zeros((int(act.sum()), code.E))
        rn[:, slot[valid]] = R[:, valid]
        r[act] = rn
        L[act] = Lch[act] + np.add.reduceat(rn[:, perm], starts, axis=1)
        c = (L < 0).astype(np.float32)
        ok = ~(np.rint(c @ Ht).astype(np.int64) % 2).any(axis=1)
        newly = ok & ~done
        used[newly] = it
        done |= ok
        if done.all():
            break
    return (L < 0).astype(np.int8), used


def crs_channel_estimate(Y, pci, subframe):
    """LS channel estimates at the port-0 CRS, linear interpolation across frequency in each
    CRS symbol and then across time (held constant beyond the first/last CRS symbol)."""
    syms, Hs = [], []
    rows_all = np.arange(N_SC)
    for ns_ in (0, 1):
        for l in (0, 4):
            rows, ref = crs_port0(pci, 2 * subframe + ns_, l)
            h = Y[rows, 7 * ns_ + l] / ref
            Hs.append(np.interp(rows_all, rows, h.real) + 1j * np.interp(rows_all, rows, h.imag))
            syms.append(7 * ns_ + l)
    Hs = np.array(Hs)
    H = np.empty((N_SC, 14), complex)
    for k in range(N_SC):
        H[k] = np.interp(np.arange(14), syms, Hs[:, k].real) + 1j * np.interp(np.arange(14), syms, Hs[:, k].imag)
    return H
