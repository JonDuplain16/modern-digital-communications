"""Figures for Chapter 15: Turbo, LDPC and Polar Codes.

Every curve is simulated or computed here.  Turbo/BCJR/EXIT/density-evolution helpers live in
commlib/turbo.py (shared with Lab 23); batched LDPC and polar decoders, the finite-blocklength
normal approximation for the binary-input AWGN channel and the Gaussian-approximation polar
construction are defined below.  Monte Carlo sizes are chosen so that each figure runs in a
minute or two on a laptop.

Run:  python ch15_figs.py [name ...]
"""
import sys, time
from figstyle import *
from scipy.stats import norm
from scipy.optimize import brentq
import commlib as cl
from commlib import turbo as tb
from commlib.coding import _gf2_rref

db = lambda x: 10 * np.log10(x)
lin = lambda x: 10 ** (np.asarray(x, dtype=float) / 10)
RES = {}                       # in-memory cache shared between figures in one run


# ============================================================================ channel helpers
def sigma_of(ebn0_db, R):
    return np.sqrt(1 / (2 * R * lin(ebn0_db)))


def awgn_llr(bits, ebn0_db, R, rng):
    s = sigma_of(ebn0_db, R)
    y = (1 - 2.0 * bits) + s * rng.standard_normal(np.shape(bits))
    return 2 * y / s ** 2


_GH_T, _GH_W = np.polynomial.hermite.hermgauss(100)


def biawgn_CV(sigma):
    """Capacity (bits) and dispersion (bits^2) of the BPSK-input AWGN channel."""
    y = 1 + np.sqrt(2) * sigma * _GH_T
    i = 1 - np.logaddexp(0, -2 * y / sigma ** 2) / np.log(2)
    w = _GH_W / np.sqrt(np.pi)
    C = np.sum(w * i)
    return C, np.sum(w * (i - C) ** 2)


def shannon_ebn0_biawgn(R):
    s = brentq(lambda s: biawgn_CV(s)[0] - R, 0.05, 20)
    return db(1 / (2 * R * s ** 2))


def na_ebn0(n, k, eps):
    """Eb/N0 (dB) at which the normal approximation allows k bits in n BPSK uses at FER eps."""
    R = k / n
    f = lambda s: n * biawgn_CV(s)[0] - np.sqrt(n * biawgn_CV(s)[1]) * norm.isf(eps) \
        + 0.5 * np.log2(n) - k
    s = brentq(f, 0.05, 20)
    return db(1 / (2 * R * s ** 2))



# ============================================================================ optional result cache
# Results are cached in cache/ch15_simcache.pkl by default (override with CH15_CACHE=<file.pkl>);
# delete it to recompute everything (a cold run takes ~50 min). Original note: cache Monte Carlo results between runs (handy when only the
# plotting changes). Without it every figure is recomputed from scratch.
import os, pickle, hashlib, functools
_CACHE_FILE = os.environ.get("CH15_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ch15_simcache.pkl"))


def _fp(code):
    h = hashlib.sha1()
    for attr in ("K", "N", "n", "puncture", "r"):
        h.update(repr(getattr(code, attr, None)).encode())
    for attr in ("pi", "H", "info"):
        v = getattr(code, attr, None)
        if v is not None:
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def _cached(fn):
    @functools.wraps(fn)
    def wrap(code, *a, **k):
        if not _CACHE_FILE:
            return fn(code, *a, **k)
        args = [np.round(np.asarray(x, float), 4).tolist() if isinstance(x, np.ndarray) else x for x in a]
        key = fn.__name__ + repr((_fp(code), args, sorted(k.items())))
        store = {}
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "rb") as f:
                store = pickle.load(f)
        if key not in store:
            store[key] = fn(code, *a, **k)
            with open(_CACHE_FILE, "wb") as f:
                pickle.dump(store, f)
        return store[key]
    return wrap


# ============================================================================ batched LDPC
class BatchLDPC:
    """Batched BP decoders for any parity-check matrix (all-zero-codeword simulations).

    methods: 'spa' (sum-product, tanh rule), 'ms' (min-sum), 'nms' (normalised, alpha),
    'oms' (offset, beta).  schedule: 'flood' or 'layered' (layers = list of row index arrays
    in which every column appears at most once)."""

    def __init__(self, H, layers=None):
        H = np.asarray(H) % 2
        self.H = H
        self.m, self.n = H.shape
        ce, ev = np.nonzero(H)
        self.ce, self.ev = ce, ev
        self.E = len(ce)
        dc = np.bincount(ce, minlength=self.m)
        self.dcmax = dc.max()
        slot = -np.ones((self.m, self.dcmax), dtype=int)
        pos = np.zeros(self.m, dtype=int)
        for e, c in enumerate(ce):
            slot[c, pos[c]] = e
            pos[c] += 1
        self.slot = slot
        self.regular = bool(np.all(slot >= 0)) and np.all(np.diff(ce) >= 0)
        self.layers = layers
        R, _ = _gf2_rref(H.astype(np.int8))
        self.k = self.n - R.shape[0]

    @property
    def rate(self):
        return self.k / self.n

    def _check(self, Q, valid, method, alpha, beta):
        """Q: (B, rows, dcmax) incoming messages (invalid slots arbitrary). Returns R."""
        if method == "spa":
            t = np.tanh(np.clip(Q, -40, 40) / 2)
            t = np.where(valid, t, 1.0)
            t = np.where(np.abs(t) < 1e-15, 1e-15, t)
            prod = np.prod(t, axis=2, keepdims=True)
            return 2 * np.arctanh(np.clip(prod / t, -1 + 1e-15, 1 - 1e-15))
        mag = np.where(valid, np.abs(Q), np.inf)
        neg = valid & (Q < 0)
        sgn = 1.0 - 2.0 * neg
        tot = 1.0 - 2.0 * (neg.sum(axis=2, keepdims=True) & 1)
        two = np.partition(mag, 1, axis=2)
        m1, m2 = two[:, :, :1], two[:, :, 1:2]
        mm = np.where(mag == m1, m2, m1)
        if method == "nms":
            mm = alpha * mm
        elif method == "oms":
            mm = np.maximum(mm - beta, 0)
        return tot * sgn * mm

    def decode(self, Lch, iters=50, method="spa", alpha=0.75, beta=0.5, schedule="flood"):
        """Returns (hard decisions (B, n), iterations used per frame (B,))."""
        Lch = np.atleast_2d(Lch).astype(float)
        B = Lch.shape[0]
        r = np.zeros((B, self.E))
        L = Lch.copy()
        used = np.full(B, iters)
        active = np.arange(B)
        out = np.zeros((B, self.n), dtype=np.int8)
        for it in range(1, iters + 1):
            if schedule == "flood":
                q = L[:, self.ev] - r
                sl = self.slot
                if self.regular:
                    R = self._check(q.reshape(q.shape[0], self.m, self.dcmax), True, method, alpha, beta)
                    r = R.reshape(q.shape[0], self.E)
                else:
                    valid = (sl >= 0)[None]
                    Q = q[:, np.maximum(sl, 0)]
                    R = self._check(Q, valid, method, alpha, beta)
                    r = np.zeros_like(r)
                    r[:, sl[sl >= 0]] = R[:, sl >= 0]
                L = Lch + (self._ev_mat().T @ r.T).T
            else:
                for rows in self.layers:
                    sl = self.slot[rows]
                    valid = (sl >= 0)[None]
                    e = sl[sl >= 0]
                    q = L[:, self.ev[e]] - r[:, e]
                    Q = np.zeros((L.shape[0],) + sl.shape)
                    Q[:, sl >= 0] = q
                    R = self._check(Q, valid, method, alpha, beta)
                    rn = R[:, sl >= 0]
                    L[:, self.ev[e]] = q + rn
                    r[:, e] = rn
            hard = (L < 0).astype(np.int8)
            ok = ~((self._Ht().T @ hard.T.astype(np.int32)).T % 2).any(axis=1)
            if ok.any():
                out[active[ok]] = hard[ok]
                used[active[ok]] = it
                keep = ~ok
                active, L, Lch, r = active[keep], L[keep], Lch[keep], r[keep]
                if len(active) == 0:
                    break
        if len(active):
            out[active] = (L < 0).astype(np.int8)
        return out, used

    def _ev_mat(self):
        if not hasattr(self, "_EVM"):
            from scipy.sparse import csr_matrix
            self._EVM = csr_matrix((np.ones(self.E), (np.arange(self.E), self.ev)),
                                   shape=(self.E, self.n))
        return self._EVM

    def _Ht(self):
        if not hasattr(self, "_HT"):
            from scipy.sparse import csr_matrix
            self._HT = csr_matrix(self.H.T.astype(np.int32))
        return self._HT


def qc_expand(S, Z):
    """Lift a base/shift matrix S (entries -1 = zero block, s >= 0 = identity shifted by s)."""
    mb, nb = S.shape
    H = np.zeros((mb * Z, nb * Z), dtype=np.int8)
    I = np.arange(Z)
    for i in range(mb):
        for j in range(nb):
            if S[i, j] >= 0:
                H[i * Z + I, j * Z + (I + S[i, j]) % Z] = 1
    return H


def qc_girth8_shifts(mb, nb, Z, rng, tries=20000):
    """Random circulant shifts for an all-ones mb x nb base matrix with no 4- or 6-cycles."""
    best = None
    for _ in range(tries):
        S = rng.integers(0, Z, (mb, nb))
        S[0, :] = 0
        S[:, 0] = 0
        bad = 0
        for i in range(mb):
            for j in range(mb):
                if i == j:
                    continue
                d = S[i] - S[j]
                bad += np.sum((d[:, None] - d[None, :]) % Z == 0) - nb
        if bad == 0:
            # 6-cycles: rows i,j,k distinct, columns a,b,c distinct
            ok = True
            import itertools
            for i, j, k in itertools.permutations(range(mb), 3):
                for a, b, c in itertools.permutations(range(nb), 3):
                    if (S[i, a] - S[j, a] + S[j, b] - S[k, b] + S[k, c] - S[i, c]) % Z == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return S
            best = S
    return best


@_cached
def simulate_ldpc(dec, ebn0s, method, schedule="flood", max_frames=2000, min_ferr=80, B=250,
                  iters=50, seed=1, **kw):
    ber, fer, avg_it = [], [], []
    for e in ebn0s:
        rng = np.random.default_rng(seed + int(1000 + 100 * e))
        be = fe = nf = 0
        its = []
        while nf < max_frames and fe < min_ferr:
            Lch = awgn_llr(np.zeros((B, dec.n), np.int8), e, dec.rate, rng)
            hard, used = dec.decode(Lch, iters, method, schedule=schedule, **kw)
            errs = hard.sum(axis=1)
            be += errs.sum(); fe += (errs > 0).sum(); nf += B
            its.append(used)
        ber.append(be / (nf * dec.n)); fer.append(fe / nf)
        avg_it.append(np.concatenate(its).mean())
    return np.array(ber), np.array(fer), np.array(avg_it)


# ============================================================================ batched polar
def ga_construction(N, design_ebn0_db, R):
    """Gaussian-approximation reliabilities (larger mean LLR = more reliable), commlib order."""
    def phi(x):
        x = np.maximum(x, 1e-12)
        return np.where(x < 10, np.exp(-0.4527 * x ** 0.86 + 0.0218),
                        np.sqrt(np.pi / x) * np.exp(-x / 4) * (1 - 10 / (7 * x)))

    xs = np.logspace(-6, 3.5, 4000)
    ph = np.maximum(phi(xs), 1e-300)

    def phiinv(y):
        return np.interp(-np.log(np.maximum(y, 1e-300)), -np.log(ph), xs)

    def rec(Nn, m):
        if Nn == 1:
            return np.array([m])
        mminus = phiinv(1 - (1 - phi(m)) ** 2)
        return np.concatenate([rec(Nn // 2, mminus), rec(Nn // 2, 2 * m)])

    s = sigma_of(design_ebn0_db, R)
    return rec(N, 2 / s ** 2)


def _f(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BatchPolar:
    """Batched SC / SC-list / CRC-aided SCL decoder (min-sum f, path metric of Balatsoukas-Stimming
    et al.), with the rate-0 subtree shortcut. Encoding via commlib.PolarCode.transform."""

    def __init__(self, N, K, design_ebn0_db=1.0, crc_poly=None):
        self.N, self.K = N, K
        self.crc = None if crc_poly is None else np.asarray(crc_poly, np.int8)
        self.r = 0 if crc_poly is None else len(crc_poly) - 1
        rel = ga_construction(N, design_ebn0_db, K / N)
        self.info = np.sort(np.argsort(-rel)[:K + self.r])
        self.frozen = np.ones(N, bool)
        self.frozen[self.info] = False
        if self.crc is not None:
            self.Gcrc = np.array([cl.crc_remainder(np.eye(K, dtype=np.int8)[i], self.crc)
                                  for i in range(K)], dtype=np.int32)
        G = np.array([cl.PolarCode.transform(np.eye(N, dtype=np.int8)[i]) for i in range(N)])
        self.G = G.astype(np.int32)
        self._allfrozen = {}

    def encode(self, msg):
        msg = np.atleast_2d(msg).astype(np.int32)
        if self.crc is not None:
            msg = np.concatenate([msg, msg @ self.Gcrc % 2], axis=1)
        u = np.zeros((msg.shape[0], self.N), np.int32)
        u[:, self.info] = msg
        return (u @ self.G % 2).astype(np.int8)

    def _node(self, alpha, fz, pm, L):
        B, Lc, n = alpha.shape
        if fz.all():                                   # rate-0 subtree: all zeros
            pm = pm + np.sum(np.where(alpha < 0, -alpha, 0.0), axis=2)
            z = np.zeros((B, Lc, n), np.int8)
            return z, z, pm, np.broadcast_to(np.arange(Lc), (B, Lc))
        if n == 1:
            a = alpha[:, :, 0]
            p0 = pm + np.where(a < 0, -a, 0.0)
            p1 = pm + np.where(a >= 0, a, 0.0)
            cand = np.concatenate([p0, p1], axis=1)
            keep = np.argsort(cand, axis=1, kind="stable")[:, :L]
            parent = keep % Lc
            bit = (keep >= Lc).astype(np.int8)[:, :, None]
            return bit, bit, np.take_along_axis(cand, keep, 1), parent
        h = n // 2
        aL = _f(alpha[:, :, :h], alpha[:, :, h:])
        bL, uL, pm, i1 = self._node(aL, fz[:h], pm, L)
        alpha = np.take_along_axis(alpha, i1[:, :, None], 1)
        aR = alpha[:, :, h:] + (1 - 2.0 * bL) * alpha[:, :, :h]
        bR, uR, pm, i2 = self._node(aR, fz[h:], pm, L)
        bL = np.take_along_axis(bL, i2[:, :, None], 1)
        uL = np.take_along_axis(uL, i2[:, :, None], 1)
        return (np.concatenate([bL ^ bR, bR], 2), np.concatenate([uL, uR], 2), pm,
                np.take_along_axis(i1, i2, 1))

    def decode(self, llr, L=1):
        llr = np.atleast_2d(llr)
        B = llr.shape[0]
        alpha = np.repeat(llr[:, None, :], L, axis=1)
        pm = np.full((B, L), np.inf)
        pm[:, 0] = 0.0
        _, u, pm, _ = self._node(alpha, self.frozen, pm, L)
        cand = u[:, :, self.info].astype(np.int32)                # (B, L, K + r)
        order = np.argsort(pm, axis=1)
        if self.crc is None:
            best = order[:, 0]
            return cand[np.arange(B), best, :self.K]
        msg, chk = cand[:, :, :self.K], cand[:, :, self.K:]
        ok = ~((msg @ self.Gcrc % 2) != chk).any(axis=2) & np.isfinite(pm)
        okso = np.take_along_axis(ok, order, 1)
        first = np.where(okso.any(axis=1), np.argmax(okso, axis=1), 0)
        best = order[np.arange(B), first]
        return msg[np.arange(B), best]


@_cached
def simulate_polar(pc, ebn0s, L, max_frames=6000, min_ferr=80, B=400, seed=7):
    fer = []
    R = pc.K / pc.N
    for e in ebn0s:
        rng = np.random.default_rng(seed + int(1000 + 100 * e))
        fe = nf = 0
        while nf < max_frames and fe < min_ferr:
            msg = rng.integers(0, 2, (B, pc.K))
            x = pc.encode(msg)
            dec = pc.decode(awgn_llr(x, e, R, rng), L)
            fe += np.any(dec != msg, axis=1).sum(); nf += B
        fer.append(fe / nf)
    return np.array(fer)


# ============================================================================ turbo helpers
QPP = {40: (3, 10), 128: (15, 32), 256: (15, 32), 1024: (31, 64), 6144: (263, 480)}


@_cached
def simulate_turbo(tc, ebn0s, iters=8, max_frames=2000, min_ferr=80, min_berr=400, B=200,
                   seed=3, maxlog=False, scale=1.0):
    """Returns ber[iteration, snr], fer[iteration, snr]."""
    ber = np.zeros((iters, len(ebn0s))); fer = np.zeros_like(ber)
    for j, e in enumerate(ebn0s):
        rng = np.random.default_rng(seed + int(1000 + 100 * e))
        be = np.zeros(iters); fe = np.zeros(iters); nf = 0
        while nf < max_frames and (fe[-1] < min_ferr or be[-1] < min_berr):
            u = rng.integers(0, 2, (B, tc.K))
            x = tc.flatten(tc.encode(u))
            hist = tc.decode(awgn_llr(x, e, tc.rate, rng), iters=iters, record=True,
                             maxlog=maxlog, scale=scale)
            for i, h in enumerate(hist):
                ne = (h != u).sum(axis=1)
                be[i] += ne.sum(); fe[i] += (ne > 0).sum()
            nf += B
        ber[:, j] = be / (nf * tc.K); fer[:, j] = fe / nf
    return ber, fer


@_cached
def weight2_codewords(tc, dmax_sep=7 * 8):
    """Low-weight codewords of a PCCC generated by weight-2 inputs (the patterns that cause the
    error floor of turbo codes with recursive constituents).  Returns (total weights, info weight 2)."""
    K, pi = tc.K, tc.pi
    inv = np.argsort(pi)
    pairs = set()
    for d in range(7, dmax_sep + 1, 7):
        for i in range(K - d):
            pairs.add((i, i + d))                      # short in encoder 1
            a, b = pi[i], pi[i + d]                    # short in encoder 2's input order
            pairs.add((min(a, b), max(a, b)))
    pairs = np.array(sorted(pairs))
    B = len(pairs)
    u = np.zeros((B, K), np.int8)
    u[np.arange(B), pairs[:, 0]] = 1
    u[np.arange(B), pairs[:, 1]] = 1
    w = []
    for s in range(0, B, 4000):
        cw = tc.flatten(tc.encode(u[s:s + 4000]))
        w.append(cw.sum(axis=1))
    return np.concatenate(w)


# ============================================================================ FIGURES
def turbo_iters():
    """BER of the LTE K=1024 turbo code vs iteration; log-MAP and max-log-MAP."""
    K = 1024
    tc = tb.TurboCode(K, tb.qpp_interleaver(K, *QPP[K]))
    eb = np.arange(-0.2, 1.61, 0.2)
    t0 = time.time()
    ber, fer = simulate_turbo(tc, eb, iters=8, max_frames=1200, min_berr=300, min_ferr=60)
    berm, _ = simulate_turbo(tc, eb, iters=8, max_frames=600, min_berr=300, min_ferr=60,
                             maxlog=True, seed=11)
    bers, _ = simulate_turbo(tc, eb, iters=8, max_frames=600, min_berr=300, min_ferr=60,
                             maxlog=True, scale=0.7, seed=12)
    print(f"  turbo sims {time.time() - t0:.0f}s")
    RES["turbo1024_fer"] = (eb, fer[-1])
    fig, ax = plt.subplots(figsize=(W1, 3.3))
    ebf = np.linspace(-1, 4, 200)
    ax.semilogy(ebf, cl.ber_bpsk(ebf), color=GRAY, lw=1.0, label="uncoded BPSK")
    cols = {0: GRAY, 1: ORANGE, 2: GREEN, 3: PURPLE, 5: "#2E86C1", 7: NAVY}
    for i in [0, 1, 2, 3, 5, 7]:
        b = ber[i]; ok = b > 0
        ax.semilogy(eb[ok], b[ok], "o-", ms=3, color=cols[i],
                    label=f"{i + 1} iteration" + ("s" if i else ""))
    ok = berm[-1] > 0
    ax.semilogy(eb[ok], berm[-1][ok], "s--", ms=3, color=ACCENT, lw=1.1, label="8 it., max-log-MAP")
    ok = bers[-1] > 0
    ax.semilogy(eb[ok], bers[-1][ok], "^:", ms=3, color=ACCENT, lw=1.1,
                label="8 it., max-log, extr. scaled 0.7")
    lim = shannon_ebn0_biawgn(tc.rate)
    ax.axvline(lim, color=ACCENT, ls=":", lw=1.0)
    ax.text(lim + 0.04, 2e-6, f"BPSK limit\nR = 1/3\n{lim:.2f} dB", fontsize=7, color=ACCENT)
    ax.set_xlim(-0.8, 2.2); ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    save(fig, "ch15_turbo_iters")
    for i in [0, 3, 7]:
        print("  iter", i + 1, np.round(np.log10(np.maximum(ber[i], 1e-9)), 2))
    print("  maxlog", np.round(np.log10(np.maximum(berm[-1], 1e-9)), 2))
    print("  maxlog scaled", np.round(np.log10(np.maximum(bers[-1], 1e-9)), 2))


def turbo_length():
    """FER of LTE-style turbo codes (QPP interleavers) at several block lengths, 8 iterations."""
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    specs = [(40, np.arange(0, 4.01, 0.5), 3000, NAVY),
             (256, np.arange(0, 2.51, 0.25), 2000, GREEN),
             (1024, np.arange(0, 1.41, 0.2), 1200, ORANGE),
             (6144, np.arange(0.0, 0.61, 0.1), 300, ACCENT)]
    t0 = time.time()
    for K, eb, mf, c in specs:
        tc = tb.TurboCode(K, tb.qpp_interleaver(K, *QPP[K]))
        B = 200 if K <= 1024 else 50
        _, fer = simulate_turbo(tc, eb, iters=8, max_frames=mf, min_ferr=60, min_berr=0, B=B)
        f = fer[-1]; ok = f > 0
        ax.semilogy(eb[ok], f[ok], "o-", ms=3, color=c, label=f"K = {K}")
        RES[f"turbo{K}_fer"] = (eb, f)
        na = na_ebn0(tc.n, K, 1e-2)
        ax.plot([na], [1e-2], marker="*", ms=8, color=c, mec="k", mew=0.4)
        print(f"  K={K} done {time.time() - t0:.0f}s  NA(1e-2)={na:.2f} dB  fer={np.round(f, 4)}")
    lim = shannon_ebn0_biawgn(1 / 3)
    ax.axvline(lim, color=GRAY, ls=":", lw=1.0)
    ax.text(lim + 0.05, 3e-4, f"BPSK limit, R=1/3 ({lim:.2f} dB)", fontsize=7, color=GRAY, rotation=90)
    ax.plot([], [], "*", color=GRAY, ms=8, mec="k", mew=0.4,
            label="normal-approx. bound at FER $10^{-2}$")
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("frame error rate")
    ax.set_ylim(1e-3, 1); ax.set_xlim(-0.8, 4)
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "ch15_turbo_length")


def turbo_floor():
    """Error floor: QPP vs random interleaver at K = 1024, with weight-2 asymptotes."""
    K = 1024
    codes = [("QPP (31, 64)", tb.TurboCode(K, tb.qpp_interleaver(K, *QPP[K])), NAVY),
             ("random", tb.TurboCode(K, np.random.default_rng(5).permutation(K)), ACCENT)]
    eb = np.arange(0.0, 2.01, 0.25)
    ebf = np.linspace(0, 3.5, 200)
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    t0 = time.time()
    for name, tc, c in codes:
        ber, _ = simulate_turbo(tc, eb, iters=8, max_frames=1200, min_berr=200, min_ferr=40, seed=21)
        b = ber[-1]; ok = b > 0
        ax.semilogy(eb[ok], b[ok], "o-", ms=3, color=c, label=f"{name}: simulated")
        w = weight2_codewords(tc)
        dmin = w.min()
        low = w[w <= dmin + 6]
        asym = sum((2 / K) * cl.qfunc(np.sqrt(2 * tc.rate * d * lin(ebf))) for d in low)
        ax.semilogy(ebf, asym, "--", color=c, lw=1.0,
                    label=f"{name}: weight-2 floor estimate ($d$ = {dmin}, {np.sum(w == dmin)} word" + ("s)" if np.sum(w == dmin) > 1 else ")"))
        print(f"  {name}: dmin(w2)={dmin}, counts {np.bincount(low)[dmin:]}, t={time.time() - t0:.0f}s,"
              f" ber={np.round(np.log10(np.maximum(b, 1e-9)), 2)}")
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.set_ylim(1e-9, 0.1); ax.set_xlim(0, 3.5)
    ax.annotate("waterfall", xy=(0.55, 1e-3), xytext=(1.1, 1e-2), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("error floor", xy=(1.9, 2e-6), xytext=(2.4, 2e-5), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.legend(fontsize=6.8, loc="upper right")
    save(fig, "ch15_turbo_floor")


def exit_chart():
    """EXIT chart of the rate-1/3 LTE turbo code, with a decoding trajectory."""
    rsc = tb.RSC()
    IA = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.05))
    t0 = time.time()
    cols = {-1.0: GRAY, -0.5: PURPLE, 0.0: GREEN, 0.5: ORANGE, 1.0: NAVY}
    curves = {}
    for eb, c in cols.items():
        IE = tb.exit_curve_rsc(rsc, sigma_of(eb, 1 / 3), IA, K=10000, reps=16,
                               rng=np.random.default_rng(int(10 * eb) + 50))
        curves[eb] = IE
        ax[0].plot(IA, IE, color=c, label=fr"$E_b/N_0$ = {eb:+.1f} dB")
    ax[0].plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
    ax[0].set_xlabel(r"$I_A$ (a-priori information)"); ax[0].set_ylabel(r"$I_E$ (extrinsic information)")
    ax[0].set_title("(a) Transfer curves of one RSC decoder")
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    print(f"  curves {time.time() - t0:.0f}s")
    a = ax[1]
    eb = 0.5
    IE = curves[eb]
    a.plot(IA, IE, color=NAVY, label="decoder 1")
    a.plot(IE, IA, color=ACCENT, label="decoder 2 (axes swapped)")
    IEc = curves[-0.5]
    a.plot(IA, IEc, color=GRAY, lw=0.9, ls="--", label="both at $-0.5$ dB (tunnel closed)")
    a.plot(IEc, IA, color=GRAY, lw=0.9, ls="--")
    # trajectory from a real decoder
    K = 10000
    tc = tb.TurboCode(K, np.random.default_rng(9).permutation(K))
    rng = np.random.default_rng(4)
    u = rng.integers(0, 2, (6, K))
    x = tc.flatten(tc.encode(u))
    _, traj = tc.decode(awgn_llr(x, eb, tc.rate, rng), iters=10, Linfo=u)
    X, Y = [0.0], [0.0]
    for j, (ia, ie) in enumerate(traj):
        if j % 2 == 0:     # decoder 1: vertical step to its extrinsic output
            X.append(X[-1]); Y.append(ie)
        else:              # decoder 2: horizontal step
            X.append(ie); Y.append(Y[-1])
        if ie > 0.999:
            break
    a.plot(X, Y, "-", color="k", lw=0.8, marker=".", ms=3, label="measured trajectory, K = 10000")
    a.set_xlabel(r"$I_{A1} = I_{E2}$"); a.set_ylabel(r"$I_{E1} = I_{A2}$")
    a.set_title(fr"(b) EXIT chart at $E_b/N_0$ = {eb:+.1f} dB")
    a.legend(fontsize=6.6, loc="lower right"); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15_exit")
    print(f"  total {time.time() - t0:.0f}s; trajectory steps {len(X)}")


def checknode():
    """The check-node function: tanh rule vs min-sum."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    x = np.linspace(0.01, 6, 400)
    phi = -np.log(np.tanh(x / 2))
    a = ax[0]
    a.plot(x, phi, color=NAVY, label=r"$\phi(x) = -\ln\tanh(x/2)$")
    a.plot(x, x, color="k", lw=0.5, ls=":")
    a.set_xlim(0, 6); a.set_ylim(0, 6); a.set_aspect("equal")
    a.set_xlabel("$x$"); a.set_ylabel(r"$\phi(x)$")
    a.set_title(r"(a) $\phi$ is its own inverse")
    a.legend(fontsize=7.5)
    a = ax[1]
    rng = np.random.default_rng(2)
    # messages into a degree-6 check from a moderately reliable population
    q = rng.normal(3.5, 2.5, (4000, 5))
    t = np.prod(np.tanh(q / 2), axis=1)
    exact = 2 * np.arctanh(np.clip(t, -1 + 1e-12, 1 - 1e-12))
    ms = np.prod(np.sign(q), axis=1) * np.min(np.abs(q), axis=1)
    a.plot(ms, exact, ".", ms=1.5, color=NAVY, alpha=0.4, label="sum-product vs min-sum")
    lim = 6
    a.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":", label="equal")
    a.plot([-lim, lim], [-0.75 * lim, 0.75 * lim], color=ACCENT, lw=1.0, label=r"normalised, $\alpha = 0.75$")
    xx = np.linspace(-lim, lim, 200)
    a.plot(xx, np.sign(xx) * np.maximum(np.abs(xx) - 0.5, 0), color=GREEN, lw=1.0, label=r"offset, $\beta = 0.5$")
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    a.set_xlabel("min-sum output"); a.set_ylabel("exact (tanh-rule) output")
    a.set_title(r"(b) Degree-6 check, 4000 random inputs")
    a.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "ch15_checknode")


def _qc_code(Z=384, seed=4):
    rng = np.random.default_rng(seed)
    S = qc_girth8_shifts(3, 6, Z, rng)
    H = qc_expand(S, Z)
    layers = [np.arange(i * Z, (i + 1) * Z) for i in range(3)]
    return BatchLDPC(H, layers), S


def ldpc_decoders():
    """BER/FER of a (3,6)-regular QC-LDPC code (n = 2304) with four check-node rules, and
    iterations: flooding vs layered."""
    dec, S = _qc_code()
    print("  shifts", S.tolist(), "k", dec.k, "rate", round(dec.rate, 4))
    eb = np.arange(0.75, 2.51, 0.25)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    t0 = time.time()
    cfg = [("spa", {}, NAVY, "sum-product"), ("ms", {}, ACCENT, "min-sum"),
           ("nms", dict(alpha=0.75), GREEN, r"normalised MS, $\alpha$=0.75"),
           ("oms", dict(beta=0.5), ORANGE, r"offset MS, $\beta$=0.5")]
    a = ax[0]
    for m, kw, c, lab in cfg:
        ber, fer, it = simulate_ldpc(dec, eb, m, max_frames=1500, min_ferr=60, **kw)
        ok = fer > 0
        a.semilogy(eb[ok], fer[ok], "o-", ms=3, color=c, label=lab)
        okb = ber > 0
        a.semilogy(eb[okb], ber[okb], "--", color=c, lw=0.9)
        RES[f"ldpc2304_{m}"] = (eb, fer)
        print(f"  {m}: {time.time() - t0:.0f}s fer {np.round(fer, 4)}")
    lim = shannon_ebn0_biawgn(dec.rate)
    a.axvline(lim, color=GRAY, ls=":", lw=1.0)
    a.axvline(1.11, color=PURPLE, ls=":", lw=1.0)
    a.set_xlabel(r"$E_b/N_0$ (dB)"); a.set_ylabel("FER (solid), BER (dashed)")
    a.set_ylim(1e-6, 1); a.set_xlim(0, 2.6)
    a.set_title("(a) Check-node rules, 50 iterations")
    a.legend(fontsize=6.3, loc="lower left")
    a = ax[1]
    eb2 = np.arange(1.0, 3.01, 0.25)
    for sch, c, lab in [("flood", NAVY, "flooding"), ("layered", ACCENT, "layered (3 layers)")]:
        _, fer, it = simulate_ldpc(dec, eb2, "nms", schedule=sch, max_frames=500, min_ferr=10**9,
                                   alpha=0.75, seed=77)
        a.plot(eb2, it, "o-", ms=3, color=c, label=lab)
    a.set_xlabel(r"$E_b/N_0$ (dB)"); a.set_ylabel("mean iterations (max 50)")
    a.set_title("(b) Schedules, normalised MS")
    a.legend(fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15_ldpc_decoders")
    print(f"  total {time.time() - t0:.0f}s")


def de_bec():
    """Density evolution on the BEC, and finite-length simulations."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.95))
    a = ax[0]
    x = np.linspace(0, 0.5, 400)
    thr = tb.bec_threshold([0, 0, 1], [0, 0, 0, 0, 0, 1])
    for eps, c in [(0.40, NAVY), (thr, GREEN), (0.45, ACCENT)]:
        f = eps * (1 - (1 - x) ** 5) ** 2
        a.plot(x, f, color=c, label=fr"$\varepsilon$ = {eps:.4f}" if eps == thr else fr"$\varepsilon$ = {eps:.2f}")
    a.plot(x, x, color="k", lw=0.6)
    # staircase trajectories
    for eps, c in [(0.40, NAVY), (0.45, ACCENT)]:
        xx = eps; X = [xx]; Y = [0]
        for _ in range(60):
            fx = eps * (1 - (1 - xx) ** 5) ** 2
            X += [xx, fx]; Y += [fx, fx]
            xx = fx
        a.plot(X[1:], Y[1:], color=c, lw=0.6, alpha=0.8)
    a.set_xlim(0, 0.46); a.set_ylim(0, 0.46); a.set_aspect("equal")
    a.set_xlabel(r"$x_\ell$ (erasure prob. of variable-to-check message)")
    a.set_ylabel(r"$x_{\ell+1} = \varepsilon\lambda(1-\rho(1-x_\ell))$")
    a.set_title("(a) (3,6) ensemble on the BEC")
    a.legend(fontsize=7, loc="upper left")
    # finite-length peeling decoder on random QC lifts of the all-ones 3x6 base graph
    a = ax[1]
    epss = np.linspace(0.33, 0.5, 18)
    t0 = time.time()
    for Z, c, frames in [(50, NAVY, 400), (500, GREEN, 120), (5000, ORANGE, 24), (20000, ACCENT, 4)]:
        rng = np.random.default_rng(Z)
        S = rng.integers(0, Z, (3, 6))
        n = 6 * Z
        # edge lists directly for large Z
        rows, cols = [], []
        I = np.arange(Z)
        for i in range(3):
            for j in range(6):
                rows.append(i * Z + I); cols.append(j * Z + (I + S[i, j]) % Z)
        ce = np.concatenate(rows); ev = np.concatenate(cols)
        order = np.argsort(ce, kind="stable")
        ce, ev = ce[order], ev[order]
        ber = []
        for eps in epss:
            er = rng.random((frames, n)) < eps
            known = ~er
            for _ in range(2000):
                # each check: number of erased neighbours; a check with exactly one erased
                # neighbour resolves it
                ek = ~known[:, ev]                                  # (F, E)
                cnt = np.add.reduceat(ek, np.arange(0, 3 * n, 6), axis=1)   # checks have degree 6
                single = np.repeat(cnt == 1, 6, axis=1) & ek
                newly = np.zeros_like(known)
                fidx, eidx = np.nonzero(single)
                newly[fidx, ev[eidx]] = True
                if not newly.any():
                    break
                known |= newly
            ber.append((~known).mean())
        ber = np.array(ber); ok = ber > 0
        a.semilogy(epss[ok], ber[ok], "o-", ms=2.5, color=c, label=f"n = {n}")
        print(f"  Z={Z} {time.time() - t0:.0f}s")
    a.axvline(thr, color=GRAY, ls="--", lw=1.0)
    a.text(thr - 0.012, 2e-5, f"BP threshold {thr:.4f}", rotation=90, fontsize=7, color=GRAY)
    a.axvline(0.5, color=ACCENT, ls=":", lw=1.0)
    a.text(0.488, 2e-5, "capacity", rotation=90, fontsize=7, color=ACCENT)
    a.set_xlabel(r"channel erasure probability $\varepsilon$")
    a.set_ylabel("residual erasure rate")
    a.set_title("(b) Peeling decoder, (3,6) QC lifts")
    a.set_ylim(1e-5, 1); a.legend(fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch15_de_bec")


def de_awgn():
    """Monte Carlo density evolution of the (3,6) ensemble on the BI-AWGN channel."""
    rng = np.random.default_rng(0)
    Npop = 200000
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))

    def run(sigma, iters, keep=()):
        mu = 2 / sigma ** 2
        L0 = lambda: mu + np.sqrt(2 * mu) * rng.standard_normal(Npop)      # all-zero codeword
        v = L0()
        snaps, pe = {}, []
        for it in range(1, iters + 1):
            # check update: 5 other inputs
            idx = rng.integers(0, Npop, (Npop, 5))
            t = np.prod(np.tanh(np.clip(v[idx], -40, 40) / 2), axis=1)
            c = 2 * np.arctanh(np.clip(t, -1 + 1e-15, 1 - 1e-15))
            # variable update: channel + 2 other check messages; APP: + 3
            j = rng.integers(0, Npop, (Npop, 3))
            app = L0() + c[j].sum(axis=1)
            v = app - c[j[:, 2]]
            pe.append(np.mean(app < 0) + 0.5 * np.mean(app == 0))
            if it in keep:
                snaps[it] = v.copy()
        return snaps, np.array(pe)

    a = ax[0]
    sig = 0.82
    snaps, _ = run(sig, 40, keep=(1, 6, 10, 12, 13))
    bins = np.linspace(-10, 50, 150)
    for (it, v), c in zip(snaps.items(), [GRAY, PURPLE, GREEN, ORANGE, NAVY]):
        h, e = np.histogram(v, bins=bins)
        h = h / (len(v) * (e[1] - e[0]))
        a.plot(0.5 * (e[1:] + e[:-1]), h, color=c, lw=1.0, label=f"iteration {it}")
    a.set_xlabel("variable-to-check LLR"); a.set_ylabel("density")
    a.set_title(fr"(a) Message densities, $\sigma$ = {sig}")
    a.legend(fontsize=7); a.set_xlim(-10, 50)
    a = ax[1]
    for s, c in [(0.84, NAVY), (0.87, GREEN), (0.89, ORANGE), (0.92, ACCENT)]:
        _, pe = run(s, 80)
        eb = db(1 / (2 * 0.5 * s ** 2))
        a.semilogy(np.arange(1, 81), np.maximum(pe, 1e-7), color=c,
                   label=fr"$\sigma$ = {s} ({eb:.2f} dB)")
    a.set_xlabel("iteration"); a.set_ylabel("bit error probability")
    a.set_title("(b) Above and below the threshold")
    a.set_ylim(1e-6, 0.2); a.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch15_de_awgn")


def polarization():
    """Channel polarization on the BEC(0.5)."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    a = ax[0]
    Zs = [np.array([0.5])]
    for lev in range(10):
        z = Zs[-1]
        Zs.append(np.concatenate([2 * z - z ** 2, z ** 2]))
    for lev, z in enumerate(Zs):
        a.plot(np.full(len(z), lev) + np.random.default_rng(lev).uniform(-0.18, 0.18, len(z)),
               1 - z, ".", ms=1.6 if lev > 5 else 4, color=NAVY, alpha=0.6)
    a.set_xlabel(r"polarization level $n$ ($N = 2^n$ bit-channels)")
    a.set_ylabel(r"capacity $I(W_N^{(i)}) = 1 - Z$")
    a.set_title(r"(a) Splitting BEC(0.5), level by level")
    a.set_xticks(range(0, 11, 2))
    a = ax[1]
    for n, c in [(4, GRAY), (6, PURPLE), (8, GREEN), (10, ORANGE), (14, ACCENT), (20, NAVY)]:
        z = cl.PolarCode._bhat(2 ** n, 0.5) if n <= 14 else None
        if z is None:
            z = np.array([0.5])
            for _ in range(n):
                z = np.concatenate([2 * z - z ** 2, z ** 2])
        zs = np.sort(1 - z)[::-1]
        a.plot(np.linspace(0, 1, len(zs)), zs, color=c, label=fr"$N = 2^{{{n}}}$")
        good = np.mean(z < 1e-3); bad = np.mean(z > 1 - 1e-3)
        print(f"  N=2^{n}: good {good:.3f} bad {bad:.3f} mediocre {1 - good - bad:.3f}")
    a.axvline(0.5, color="k", lw=0.6, ls=":")
    a.set_xlabel("fraction of bit-channels (sorted)"); a.set_ylabel("capacity")
    a.set_title("(b) Polarization sharpens with $N$")
    a.legend(fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch15_polarization")


def polar_decoders():
    """Polar (1024, 512): SC, SCL, CA-SCL vs the normal approximation."""
    N, K = 1024, 512
    eb = np.arange(1.0, 3.01, 0.25)
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    t0 = time.time()
    runs = [("SC", None, 1, GRAY), ("SCL, L = 8", None, 8, ORANGE),
            ("CA-SCL, L = 8, CRC-11", cl.CRC11_5G, 8, GREEN),
            ("CA-SCL, L = 32, CRC-11", cl.CRC11_5G, 32, NAVY)]
    for lab, crc, L, c in runs:
        pc = BatchPolar(N, K, design_ebn0_db=2.0, crc_poly=crc)
        fer = simulate_polar(pc, eb, L, max_frames=4000 if L < 32 else 1500, min_ferr=60,
                             B=400 if L < 32 else 150)
        ok = fer > 0
        ax.semilogy(eb[ok], fer[ok], "o-", ms=3, color=c, label=lab)
        print(f"  {lab}: {time.time() - t0:.0f}s {np.round(fer, 4)}")
    ebf = np.linspace(0.5, 3, 30)
    epsg = np.logspace(-4, -0.3, 25)
    na = [na_ebn0(N, K, e) for e in epsg]
    ax.semilogy(na, epsg, "--", color=ACCENT, lw=1.2, label="normal approximation, $n$ = 1024, $k$ = 512")
    lim = shannon_ebn0_biawgn(0.5)
    ax.axvline(lim, color=GRAY, ls=":", lw=1.0)
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("frame error rate")
    ax.set_ylim(1e-4, 1); ax.set_xlim(0, 3.1)
    ax.legend(fontsize=7.2, loc="upper right")
    save(fig, "ch15_polar_decoders")


def short_compare():
    """k = 128, n ~ 256: polar CA-SCL vs LDPC vs turbo vs convolutional vs the bound."""
    fig, ax = plt.subplots(figsize=(W1, 3.2))
    eb = np.arange(0.5, 4.51, 0.5)
    t0 = time.time()
    res = {}
    # polar
    for L, c, lab in [(8, NAVY, "polar (256,128), CA-SCL L = 8"), (32, PURPLE, "polar (256,128), CA-SCL L = 32")]:
        pc = BatchPolar(256, 128, design_ebn0_db=2.5, crc_poly=cl.CRC11_5G)
        fer = simulate_polar(pc, eb, L, max_frames=8000, min_ferr=60, B=500)
        res[lab] = (fer, c)
        print(f"  {lab} {time.time() - t0:.0f}s")
    # LDPC: (3,6) PEG from commlib, n = 256
    code = cl.LDPCCode(n=256, rate=0.5, dv=3, seed=2)
    dec = BatchLDPC(code.H)
    _, fer, _ = simulate_ldpc(dec, eb, "spa", max_frames=8000, min_ferr=60, B=500, iters=100)
    res[f"LDPC (256,{dec.k}), (3,6) PEG, BP"] = (fer, GREEN)
    print(f"  ldpc {time.time() - t0:.0f}s")
    # turbo K = 128 punctured to rate ~1/2
    tc = tb.TurboCode(128, tb.qpp_interleaver(128, *QPP[128]), puncture=True)
    _, fert = simulate_turbo(tc, eb, iters=8, max_frames=6000, min_ferr=60, min_berr=0, B=500)
    res[f"turbo ({tc.n},128), LTE-type, 8 it."] = (fert[-1], ORANGE)
    print(f"  turbo {time.time() - t0:.0f}s")
    # convolutional K = 7 (133,171), terminated: n = 268
    cc = cl.ConvCode()
    fer = []
    for e in eb:
        rng = np.random.default_rng(int(100 * e))
        fe = nf = 0
        while nf < 8000 and fe < 60:
            u = rng.integers(0, 2, (500, 128))
            x = cc.encode_batch(u)
            d = cc.decode_batch(awgn_llr(x, e, 128 / x.shape[1], rng))
            fe += np.any(d != u, axis=1).sum(); nf += 500
        fer.append(fe / nf)
    res["convolutional K = 7, (268,128), Viterbi"] = (np.array(fer), ACCENT)
    print(f"  conv {time.time() - t0:.0f}s")
    for lab, (f, c) in res.items():
        ok = f > 0
        ax.semilogy(eb[ok], f[ok], "o-", ms=3, color=c, label=lab)
    RES["short"] = (eb, {k: v[0] for k, v in res.items()})
    epsg = np.logspace(-4, -0.3, 25)
    ax.semilogy([na_ebn0(256, 128, e) for e in epsg], epsg, "--", color="k", lw=1.1,
                label="normal approximation, (256,128)")
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("frame error rate")
    ax.set_ylim(1e-4, 1); ax.set_xlim(0.4, 4.6)
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "ch15_short_compare")
    for lab, (f, c) in res.items():
        with np.errstate(divide="ignore"):
            print("   ", lab, np.round(np.interp(-3, np.log10(np.maximum(f[::-1], 1e-9)), eb[::-1]), 2), "dB @1e-3")


def gap_length():
    """Required Eb/N0 at FER 1e-2 vs information block length: bounds and simulated codes."""
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    ks = np.unique(np.round(np.logspace(np.log10(32), 5, 40)).astype(int))
    for R, c in [(1 / 3, NAVY), (1 / 2, GREEN), (3 / 4, ORANGE)]:
        req = [na_ebn0(int(round(k / R)), k, 1e-2) for k in ks]
        ax.semilogx(ks, req, color=c, label=f"normal approx., R = {R:.2g}")
        ax.axhline(shannon_ebn0_biawgn(R), color=c, ls=":", lw=0.9)
    # simulated points (FER = 1e-2), rate 1/3 turbo codes
    for K in [40, 256, 1024, 6144]:
        key = f"turbo{K}_fer"
        if key not in RES:
            continue
        e, f = RES[key]
        ok = f > 0
        x = np.interp(-2, np.log10(f[ok])[::-1], e[ok][::-1])
        ax.plot(K, x, "o", color=NAVY, mfc="white", ms=5)
    ax.plot([], [], "o", color=NAVY, mfc="white", label="LTE-type turbo, R = 1/3 (simulated)")
    if "short" in RES:
        e, d = RES["short"]
        mk = {"polar": ("s", GREEN), "LDPC": ("^", GREEN), "turbo": ("D", GREEN), "conv": ("v", GREEN)}
        for lab, f in d.items():
            ok = f > 0
            x = np.interp(-2, np.log10(f[ok])[::-1], e[ok][::-1])
            key = lab.split()[0]
            m, c = mk.get(key, ("x", GREEN))
            if "L = 32" in lab:
                continue
            ax.plot(128, x, m, color=c, mfc="white", ms=5, label=f"R ~ 1/2, k = 128: {key}")
    for kk in ["ldpc2304_spa"]:
        if kk in RES:
            e, f = RES[kk]
            ok = f > 0
            x = np.interp(-2, np.log10(f[ok])[::-1], e[ok][::-1])
            ax.plot(1152, x, "^", color=GREEN, ms=5, label="R ~ 1/2, k ~ 1152: (3,6) QC-LDPC, BP")
    ax.set_xlabel("information bits $k$"); ax.set_ylabel(r"required $E_b/N_0$ (dB) at FER $10^{-2}$")
    ax.set_ylim(-1, 5); ax.set_xlim(30, 1e5)
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "ch15_gap_length")


def harq():
    """HARQ throughput in Rayleigh block fading: no HARQ, Chase combining, incremental redundancy."""
    rng = np.random.default_rng(1)
    snr_db = np.arange(-10, 31, 1.0)
    R0 = 4.0            # bits per channel use of the first transmission
    T = 4
    Nmc = 40000
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    out = {k: [] for k in ("arq", "cc", "ir")}
    erg = []
    for s in snr_db:
        g = lin(s) * rng.exponential(size=(Nmc, T))
        erg.append(np.mean(np.log2(1 + g[:, 0])))
        for kind in out:
            if kind == "arq":
                ok = np.log2(1 + g) >= R0
            elif kind == "cc":
                ok = np.log2(1 + np.cumsum(g, axis=1)) >= R0
            else:
                ok = np.cumsum(np.log2(1 + g), axis=1) >= R0
            first = np.where(ok.any(axis=1), np.argmax(ok, axis=1) + 1, T)
            succ = ok.any(axis=1)
            out[kind].append(R0 * succ.mean() / first.mean())
    ax.plot(snr_db, erg, color="k", lw=1.0, ls=":", label="ergodic capacity")
    ax.plot(snr_db, out["ir"], color=NAVY, label="HARQ, incremental redundancy")
    ax.plot(snr_db, out["cc"], color=GREEN, label="HARQ, Chase combining")
    ax.plot(snr_db, out["arq"], color=ACCENT, label="plain ARQ (no combining)")
    ax.set_xlabel("average SNR (dB)"); ax.set_ylabel("throughput (bits per channel use)")
    ax.set_ylim(0, 4.3); ax.set_xlim(-10, 30)
    ax.legend(fontsize=7.2, loc="upper left")
    save(fig, "ch15_harq")
    i = list(snr_db).index(10.0)
    print("  at 10 dB:", {k: round(v[i], 2) for k, v in out.items()}, "erg", round(erg[i], 2))


def spatial_coupling():
    """Threshold saturation: DE for the (3,6,L,3) coupled ensemble on the BEC."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    a = ax[0]
    L = 64
    prof, snaps = tb.de_bec_coupled(0.47, 3, 6, L, w=3, iters=3000, record_every=40)
    cmap = plt.get_cmap("viridis")
    show = snaps[::2][:12]
    for i, s in enumerate(show):
        a.plot(np.arange(L), s, color=cmap(i / max(1, len(show) - 1)), lw=1.0)
    a.set_xlabel("position in the coupled chain"); a.set_ylabel("erasure probability")
    a.set_title(r"(a) Decoding wave, $\varepsilon$ = 0.47, L = 64")
    a = ax[1]
    Ls = [4, 8, 16, 32, 64]
    thr = []
    for Lc in Ls:
        lo, hi = 0.40, 0.90
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            p = tb.de_bec_coupled(mid, 3, 6, Lc, w=3, iters=20000, tol=1e-13)
            if p.max() < 1e-8:
                lo = mid
            else:
                hi = mid
        thr.append(lo)
        print(f"  L={Lc}: threshold {lo:.4f}")
    rate = [0.5 - 0.5 * (3 + 1 - 2 * sum((i / 3) ** 6 for i in range(4))) / Lc for Lc in Ls]
    a.plot(Ls, thr, "o-", color=NAVY, label="coupled BP threshold")
    a.plot(Ls, 1 - np.array(rate), "s--", color=ACCENT, ms=3, label="Shannon limit $1-R_L$")
    a.axhline(0.4294, color=GRAY, ls=":", label="uncoupled BP, 0.4294")
    a.axhline(0.4881, color=GREEN, ls="--", lw=1.0, label="(3,6) MAP threshold, 0.4881")
    a.set_xscale("log", base=2)
    a.set_xlabel("chain length L"); a.set_ylabel(r"threshold $\varepsilon^*$")
    a.set_title("(b) Threshold saturation, (3,6,L,3)")
    a.legend(fontsize=6.6, loc="center right")
    a.set_ylim(0.42, 0.9)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch15_spatial_coupling")


def nr_basegraph():
    """5G NR LDPC base-graph selection (TS 38.212 Sec. 7.2.2) and code-block sizes."""
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    A = np.logspace(np.log10(24), np.log10(20000), 600)
    R = np.linspace(0.02, 0.95, 400)
    AA, RR = np.meshgrid(A, R)
    bg2 = (AA <= 292) | ((AA <= 3824) & (RR <= 0.67)) | (RR <= 0.25)
    ax.contourf(AA, RR, bg2.astype(float), levels=[-0.5, 0.5, 1.5], colors=[NAVY, ORANGE], alpha=0.25)
    ax.contour(AA, RR, bg2.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    ax.set_xscale("log")
    ax.text(8200, 0.50, "base graph 1\n(46 x 68,\n$K \\leq$ 8448 per\ncode block,\nmother rate 1/3)",
            fontsize=7.5, ha="center", color=NAVY)
    ax.text(110, 0.45, "base graph 2\n(42 x 52, $K \\leq$ 3840,\nmother rate 1/5)", fontsize=7.5,
            ha="center", color=ORANGE)
    ax.text(8000, 0.12, "base graph 2 (R $\\leq$ 1/4)", fontsize=7.5, ha="center", color=ORANGE)
    ax.set_xlabel("transport block size $A$ (bits)"); ax.set_ylabel("code rate $R$")
    ax.set_xlim(24, 20000); ax.set_ylim(0.02, 0.95)
    ax.axvline(292, color=GRAY, lw=0.6, ls=":"); ax.axvline(3824, color=GRAY, lw=0.6, ls=":")
    ax.text(310, 0.04, "A = 292", fontsize=7, color=GRAY)
    ax.text(4000, 0.04, "A = 3824", fontsize=7, color=GRAY)
    ax.text(1100, 0.70, "R = 0.67", fontsize=7, color=GRAY)
    save(fig, "ch15_nr_basegraph")


# ============================================================================ 2nd-edition concept figures
# Light computations only (seconds each); none of them touches the Monte Carlo cache except
# fig_anatomy, which re-reads one cached curve.
SW, SH = 3.0, 2.4          # narrow single-panel figures for side-by-side pairs


def fig_timeline():
    """Seventy years of coding in one strip."""
    ev = [(1948, "Shannon:|capacity", ACCENT, 0.55), (1960, "Gallager's|LDPC thesis", GREEN, -0.55),
          (1967, "Viterbi|algorithm", NAVY, 0.55), (1974, "BCJR|algorithm", NAVY, -0.55),
          (1981, "Tanner|graphs", GREEN, 0.55), (1993, "turbo codes|(Berrou et al.)", ORANGE, -0.55),
          (1996, "MacKay & Neal|rediscover LDPC", GREEN, 1.15), (2001, "density evolution,|EXIT charts", PURPLE, -1.15),
          (2005, "DVB-S2|LDPC", GREEN, 0.55), (2009, "Arıkan:|polar codes", "#2E86C1", -0.55),
          (2011, "Tal & Vardy:|list decoding", "#2E86C1", 1.15), (2016, "3GPP: LDPC + polar|for 5G", ACCENT, -1.15)]
    ev = [(y, f"{y}|{l}".replace("|", chr(10)), c, h) for y, l, c, h in ev]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.axhline(0, color=NAVY, lw=2.0, zorder=1)
    for x0, x1, lab, c in [(1948, 1993, "the long chase: 2-3 dB short of the limit", GRAY),
                           (1993, 2018, "the iterative era: within ~1 dB", ACCENT)]:
        ax.add_patch(plt.Rectangle((x0, -0.08), x1 - x0, 0.16, color=c, alpha=0.18, lw=0))
        ax.text((x0 + x1) / 2, 1.62, lab, ha="center", va="center", fontsize=7.2, color=c, style="italic")
        ax.annotate("", xy=(x0, 1.45), xytext=(x1, 1.45), arrowprops=dict(arrowstyle="<->", color=c, lw=0.8))
    for yr, lab, c, h in ev:
        s = np.sign(h)
        ax.plot([yr, yr], [0.08 * s, h - 0.36 * s], color=c, lw=0.9)
        ax.plot(yr, 0, "o", color=c, ms=4.5, zorder=3)
        ax.text(yr, h, lab, ha="center", va="center", fontsize=6.4, color=c)
    ax.set_xlim(1942, 2022); ax.set_ylim(-1.55, 1.8)
    ax.axis("off")
    save(fig, "ch15_timeline")


def fig_gap_chase():
    """Required Eb/N0 at BER 1e-5 for landmark schemes (values of Table 15.1)."""
    pts = [(1948, 9.6, "uncoded"), (1950, 9.2, "Hamming\n(7,4)"), (1972, 4.4, "K = 7\nViterbi"),
           (1977, 2.5, "RS + conv.\n(Voyager)"), (1993, 0.7, "turbo"), (2001, 0.23, "irregular\nLDPC")]
    fig, ax = plt.subplots(figsize=(SW, SH))
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.step(x + [2010], y + [y[-1]], where="post", color=NAVY, lw=1.6)
    ax.plot(x, y, "o", color=NAVY, ms=4)
    for (xx, yy, lab), dx, dy in zip(pts, [-1, 3, 2, 2, -2, 1], [-1.0, 0.9, 1.0, 1.0, 1.2, 1.5]):
        ax.text(xx + dx, yy + dy, lab, fontsize=6.5, color=NAVY, va="center", ha="left" if dx > 0 else "center")
    ax.axhline(0.19, color=ACCENT, ls="--", lw=1.0)
    ax.text(1950, -0.9, "Shannon limit, rate 1/2 (0.19 dB)", fontsize=6.5, color=ACCENT)
    ax.set_xlim(1945, 2010); ax.set_ylim(-1.5, 11)
    ax.set_xlabel("year"); ax.set_ylabel(r"$E_b/N_0$ for BER $10^{-5}$ (dB)")
    save(fig, "ch15_gap_chase")


def fig_llr():
    """(a) LLR as a confidence dial; (b) channel LLR densities obey consistency."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    a = ax[0]
    L = np.linspace(-8, 8, 400)
    a.plot(L, 1 / (1 + np.exp(-L)), color=NAVY, lw=1.8)
    for Lv, txt in [(0, "L = 0: a coin toss"), (4.6, "L = 4.6: 99 % sure it is 0"), (-2.2, "L = -2.2: 90 % sure it is 1")]:
        p = 1 / (1 + np.exp(-Lv))
        a.plot(Lv, p, "o", color=ACCENT, ms=4)
        a.annotate(txt, (Lv, p), xytext=(Lv + (0.7 if Lv < 3 else -7.5), p + (0.13 if Lv <= 0 else -0.14)),
                   fontsize=6.8, color=ACCENT, arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))
    a.set_xlabel("LLR $L$"); a.set_ylabel("$P(c=0)$")
    a.set_title("(a) sign = decision, size = confidence", fontsize=8.5)
    a = ax[1]
    r = np.random.default_rng(1)
    for eb, c in [(-1.0, ORANGE), (2.0, GREEN), (5.0, NAVY)]:
        Lc = awgn_llr(np.zeros(200000, int), eb, 0.5, r)
        mu = Lc.mean(); var = Lc.var()
        a.hist(Lc, bins=150, density=True, histtype="stepfilled", alpha=0.25, color=c)
        a.hist(Lc, bins=150, density=True, histtype="step", color=c,
               label=f"{eb:+.0f} dB: mean {mu:.1f}, var {var:.1f}")
    a.axvline(0, color=GRAY, lw=0.8)
    a.text(-9, 0.12, "wrong\nsign", fontsize=7, color=GRAY)
    a.set_xlim(-12, 35); a.set_xlabel("channel LLR given $x=+1$"); a.set_ylabel("density")
    a.set_title("(b) variance = twice the mean", fontsize=8.5)
    a.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch15_llr")


def fig_boxplus():
    """Box-plus: the XOR is as weak as its weakest input."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    L2 = np.linspace(0, 8, 300)
    for L1, c in [(0.5, ORANGE), (1.5, GREEN), (3.0, PURPLE), (6.0, NAVY)]:
        bp = 2 * np.arctanh(np.tanh(L1 / 2) * np.tanh(L2 / 2))
        ax.plot(L2, bp, color=c, lw=1.6, label=f"$L_1$ = {L1}")
        ax.plot(L2, np.minimum(L1, L2), color=c, lw=0.9, ls="--")
    ax.set_xlabel("$L_2$"); ax.set_ylabel(r"$L_1 \boxplus L_2$")
    ax.text(3.6, 2.15, "dashed: min-sum\n(always too confident)", fontsize=6.6, color=GRAY)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_xlim(0, 8); ax.set_ylim(0, 6.5)
    save(fig, "ch15_boxplus")


def fig_maxstar():
    """The Jacobian-log correction and an 8-entry lookup table."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    d = np.linspace(0, 5, 400)
    ax.plot(d, np.log1p(np.exp(-d)), color=NAVY, lw=1.8, label=r"$\ln(1+e^{-|a-b|})$")
    edges = np.arange(0, 4.01, 0.5)
    lut = np.log1p(np.exp(-(edges[:-1] + 0.25)))
    ax.step(np.r_[edges[:-1], 5], np.r_[lut, 0], where="post", color=ACCENT, lw=1.2, label="8-entry table")
    ax.axhline(np.log(2), color=GRAY, ls=":", lw=0.8)
    ax.text(2.6, 0.62, r"at most $\ln 2$ = 0.69", fontsize=6.8, color=GRAY)
    ax.set_xlabel("$|a-b|$"); ax.set_ylabel("correction")
    ax.legend(fontsize=7, loc="center right"); ax.set_xlim(0, 5); ax.set_ylim(0, 0.75)
    save(fig, "ch15_maxstar")


def _bcjr_ab(code, Lu, Lp):
    """Single-block BCJR returning normalised state probabilities (alpha, beta) and APP LLRs."""
    T, S = len(Lu), code.S
    xs = np.array([1.0, -1.0]); xp = 1.0 - 2.0 * code.parity
    g = 0.5 * (Lu[:, None, None] * xs[None, None, :] + Lp[:, None, None] * xp[None, :, :])
    A = np.full((T + 1, S), -1e30); A[0, 0] = 0
    ps, pu, ns = code.prev_s, code.prev_u, code.next_state
    for k in range(T):
        c0 = A[k, ps[:, 0]] + g[k, ps[:, 0], pu[:, 0]]
        c1 = A[k, ps[:, 1]] + g[k, ps[:, 1], pu[:, 1]]
        an = np.logaddexp(c0, c1); A[k + 1] = an - an.max()
    Bm = np.full((T + 1, S), -1e30); Bm[T, 0] = 0
    Lo = np.empty(T)
    for k in range(T - 1, -1, -1):
        m0 = A[k] + g[k, :, 0] + Bm[k + 1, ns[:, 0]]
        m1 = A[k] + g[k, :, 1] + Bm[k + 1, ns[:, 1]]
        Lo[k] = np.logaddexp.reduce(m0) - np.logaddexp.reduce(m1)
        bn = np.logaddexp(g[k, :, 0] + Bm[k + 1, ns[:, 0]], g[k, :, 1] + Bm[k + 1, ns[:, 1]])
        Bm[k] = bn - bn.max()
    pa = np.exp(A - A.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True)
    pb = np.exp(Bm - Bm.max(1, keepdims=True)); pb /= pb.sum(1, keepdims=True)
    return pa, pb, Lo


def fig_bcjr_heat():
    """Forward and backward state probabilities of a real BCJR run, and its output LLRs."""
    rsc = tb.RSC()
    K = 40
    r = np.random.default_rng(12)
    u = r.integers(0, 2, K)
    s, p = rsc.encode(u[None])
    s, p = s[0], p[0]
    eb, R = 1.0, 0.5
    Ls = awgn_llr(s, eb, R, r); Lp = awgn_llr(p, eb, R, r)
    pa, pb, Lo = _bcjr_ab(rsc, Ls, Lp)
    T = len(Ls)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 4.3), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.25]))
    for a, P, ttl in [(ax[0], pa[1:], r"forward $\alpha_k(s)$: what the past says about the state"),
                      (ax[1], pb[1:], r"backward $\beta_k(s)$: what the future says about the state")]:
        a.imshow(P.T, aspect="auto", cmap="Blues", origin="lower", extent=(-0.5, T - 0.5, -0.5, 7.5),
                 vmin=0, vmax=1, interpolation="nearest")
        a.set_ylabel("state"); a.set_yticks([0, 7]); a.grid(False)
        a.set_title(ttl, fontsize=8.2, loc="left")
    a = ax[2]
    k = np.arange(T)
    xk = 1 - 2 * s
    a.bar(k - 0.2, Ls * xk, width=0.4, color=GRAY, label="channel LLR alone")
    a.bar(k + 0.2, Lo * xk, width=0.4, color=NAVY, label="BCJR output")
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("LLR $\\times$ true sign"); a.set_xlabel("time $k$ (last 3 steps: tail)")
    a.legend(fontsize=6.8, loc="upper left", ncol=2)
    a.set_title("below zero = wrong decision", fontsize=8.2, loc="left")
    nwc = int(np.sum(Ls * xk < 0)); nwo = int(np.sum(Lo * xk < 0))
    print(f"  bcjr_heat: channel wrong {nwc}, BCJR wrong {nwo}")
    fig.tight_layout(h_pad=0.4)
    save(fig, "ch15_bcjr_heat")


def _turbo_loop(tc, Lch, iters, mode="extrinsic", scale=1.0):
    """Turbo decoding returning APP LLRs (natural order) after each iteration.
    mode='echo' feeds back the full APP instead of the extrinsic part (the classic bug)."""
    c = tc.unflatten(np.atleast_2d(Lch))
    K, pi = tc.K, tc.pi
    inv = np.argsort(pi)
    Ls = c["sys"]; La = np.zeros_like(Ls)
    out = []
    for it in range(iters):
        L1 = tb.bcjr(tc.rsc, np.concatenate([Ls + La, c["t1s"]], 1), np.concatenate([c["p1"], c["t1p"]], 1))[:, :K]
        E1 = (L1 - Ls - La) if mode == "extrinsic" else (L1 - Ls)
        La2 = scale * E1[:, pi]
        L2 = tb.bcjr(tc.rsc, np.concatenate([Ls[:, pi] + La2, c["t2s"]], 1), np.concatenate([c["p2"], c["t2p"]], 1))[:, :K]
        E2 = (L2 - Ls[:, pi] - La2) if mode == "extrinsic" else (L2 - Ls[:, pi])
        La = scale * E2[:, inv]
        out.append(L2[:, inv])
    return out


def fig_turbo_inside():
    """(pair) double counting vs extrinsic; LLR clouds separating over iterations."""
    K = 1024
    tc = tb.TurboCode(K, tb.qpp_interleaver(K, *QPP[K]))
    r = np.random.default_rng(31)
    B = 120
    u = r.integers(0, 2, (B, K))
    x = tc.flatten(tc.encode(u))
    eb = 0.8
    Lch = awgn_llr(x, eb, tc.rate, r)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for mode, c, lab in [("extrinsic", NAVY, "pass extrinsic only"), ("echo", ACCENT, "pass full APP (echo)")]:
        outs = _turbo_loop(tc, Lch, 10, mode)
        ber = [np.mean((o < 0) != u) for o in outs]
        ax.semilogy(np.arange(1, 11), np.maximum(ber, 1e-6), "o-", ms=3, color=c, label=lab)
        print(f"  {mode}: {np.round(ber, 5)}")
    ax.set_xlabel("iteration"); ax.set_ylabel("bit error rate")
    ax.set_title(f"K = 1024, rate 1/3, {eb} dB", fontsize=8.2)
    ax.legend(fontsize=6.8, loc="lower left"); ax.set_ylim(1e-6, 0.3)
    save(fig, "ch15_doublecount")
    # LLR clouds
    K = 6144
    tc = tb.TurboCode(K, tb.qpp_interleaver(K, *QPP[K]))
    B = 6
    u = r.integers(0, 2, (B, K))
    x = tc.flatten(tc.encode(u))
    Lch = awgn_llr(x, 0.4, tc.rate, r)
    outs = _turbo_loop(tc, Lch, 8)
    fig, ax = plt.subplots(figsize=(SW, SH))
    xk = (1 - 2 * u).ravel()
    for it, c in [(0, ORANGE), (1, GREEN), (3, PURPLE), (7, NAVY)]:
        v = np.clip(outs[it].ravel() * xk, -20, 159)
        ax.hist(v, bins=np.linspace(-20, 160, 120), density=True, histtype="step", color=c, lw=1.3,
                label=f"after {it + 1} iteration" + ("s" if it else ""))
        print(f"  it {it+1}: wrong {np.mean(v < 0):.2e}, median {np.median(v):.1f}")
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("APP LLR $\\times$ true sign"); ax.set_ylabel("density")
    ax.set_title("K = 6144, 0.4 dB", fontsize=8.2)
    ax.legend(fontsize=6.6, loc="upper right"); ax.set_xlim(-20, 160)
    save(fig, "ch15_llr_clouds")


def fig_rsc_impulse():
    """Feed-forward vs recursive encoder responses to weight-1 and weight-2 inputs."""
    T = 30
    ff = _octal_taps_local(0o15)
    rsc = tb.RSC()
    def ffenc(u):
        return np.array([sum(ff[i] * (u[k - i] if k - i >= 0 else 0) for i in range(4)) % 2 for k in range(T)])
    def rscenc(u):
        _, p = rsc.encode(np.array(u)[None], terminate=False)
        return p[0]
    u1 = np.zeros(T, int); u1[3] = 1
    u2 = u1.copy(); u2[10] = 1
    rows = [("feed-forward encoder, a single 1", u1, ffenc(u1), GRAY),
            ("recursive encoder, a single 1: parity never stops", u1, rscenc(u1), ACCENT),
            ("recursive encoder, two 1s seven apart: back to zero", u2, rscenc(u2), NAVY)]
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.2), sharex=True)
    for a, (ttl, u, p, c) in zip(ax, rows):
        k = np.arange(T)
        a.bar(k[u == 1], 0.45, bottom=0.55, width=0.7, color=ORANGE)
        a.bar(k[p == 1], 0.45, bottom=0.0, width=0.7, color=c)
        a.set_yticks([0.22, 0.77]); a.set_yticklabels(["parity", "input"], fontsize=7)
        a.set_ylim(-0.05, 1.05); a.grid(False)
        a.set_title(f"{ttl}   (parity weight {int(p.sum())}{'+' if (c == ACCENT) else ''})", fontsize=8, loc="left")
    ax[-1].set_xlabel("time $k$")
    fig.tight_layout(h_pad=0.3)
    save(fig, "ch15_rsc_impulse")


def _octal_taps_local(g, m=3):
    return np.array([(g >> (m - i)) & 1 for i in range(m + 1)], dtype=int)


def fig_interleavers():
    """Identity, random and QPP permutations of K = 256 positions."""
    K = 256
    perms = [("no interleaver", np.arange(K), GRAY), ("random", np.random.default_rng(5).permutation(K), ACCENT),
             ("QPP (LTE), $f_1$=15, $f_2$=32", tb.qpp_interleaver(K, 15, 32), NAVY)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    for a, (ttl, p, c) in zip(ax, perms):
        a.plot(np.arange(K), p, ".", ms=1.6, color=c)
        a.set_title(ttl, fontsize=8); a.set_xlabel("input position $i$")
        a.set_xlim(0, K); a.set_ylim(0, K); a.set_aspect("equal")
    ax[0].set_ylabel(r"output position $\pi(i)$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch15_interleavers")


def fig_anatomy():
    """Annotated anatomy of an iterative-code error curve (cached random-interleaver run)."""
    K = 1024
    tc = tb.TurboCode(K, np.random.default_rng(5).permutation(K))
    eb = np.arange(0.0, 2.01, 0.25)
    ber, _ = simulate_turbo(tc, eb, iters=8, max_frames=1200, min_berr=200, min_ferr=40, seed=21)
    b = ber[-1]; ok = b > 0
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.semilogy(eb[ok], b[ok], "o-", ms=3, color=NAVY)
    ebf = np.linspace(0, 2.5, 100)
    ax.semilogy(ebf, cl.ber_bpsk(ebf), color=GRAY, lw=0.9)
    ax.text(1.55, 4e-2, "uncoded", fontsize=6.5, color=GRAY)
    ax.axvspan(0.0, 0.25, color=ORANGE, alpha=0.12)
    ax.text(0.125, 3e-6, "decoder lost", fontsize=6.3, color=ORANGE, rotation=90, ha="center")
    ax.axvspan(0.25, 1.0, color=ACCENT, alpha=0.10)
    ax.text(0.62, 2e-7, "the cliff\n(waterfall)", fontsize=6.5, color=ACCENT, ha="center")
    ax.axvspan(1.0, 2.5, color=PURPLE, alpha=0.08)
    ax.text(1.75, 2e-7, "the stubborn few\n(error floor)", fontsize=6.5, color=PURPLE, ha="center")
    ax.set_xlim(0, 2.5); ax.set_ylim(1e-7, 0.2)
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    save(fig, "ch15_anatomy")


# ---------------------------------------------------------------- small Tanner graph used by two figures
_HSMALL = np.array([[1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])


def _draw_tanner(a, H, vcol, vtxt=None, ccol=None, ctxt=None, ehl=None, title=None):
    m, n = H.shape
    vx = np.linspace(0, 1, n); cx = np.linspace(0.1, 0.9, m)
    for i in range(m):
        for j in range(n):
            if H[i, j]:
                hl = ehl is not None and (i, j) in ehl
                a.plot([cx[i], vx[j]], [1, 0], color=ACCENT if hl else "#B0BEC5", lw=1.6 if hl else 0.6, zorder=1)
    for j in range(n):
        a.scatter(vx[j], 0, s=150, color=vcol[j], edgecolors=NAVY, linewidths=0.6, zorder=3)
        if vtxt is not None:
            a.text(vx[j], -0.2, vtxt[j], ha="center", va="top", fontsize=5.8)
    for i in range(m):
        a.scatter(cx[i], 1, s=110, marker="s", color=(ccol[i] if ccol is not None else "#FBE3D6"),
                  edgecolors=ACCENT, linewidths=0.6, zorder=3)
        if ctxt is not None:
            a.text(cx[i], 1.17, ctxt[i], ha="center", va="bottom", fontsize=6)
    a.set_xlim(-0.07, 1.07); a.set_ylim(-0.42, 1.38); a.axis("off")
    if title:
        a.set_title(title, fontsize=7.8)


def _llr_colour(L):
    """green for confident-and-right (all-zero codeword => positive is right), red for wrong."""
    t = np.tanh(np.abs(L) / 4)
    base = np.array([0.12, 0.48, 0.31]) if L > 0 else np.array([0.75, 0.22, 0.17])
    return tuple(1 - t * (1 - base))


def fig_gossip():
    """Belief propagation on a small graph, iteration by iteration (all-zero codeword)."""
    H = _HSMALL
    m, n = H.shape
    L0 = np.array([2.4, 1.9, -0.8, 2.2, 1.5, 2.8, 1.7, -0.6, 2.3, 1.6])
    r = np.zeros((m, n))
    snaps = [L0.copy()]
    for it in range(3):
        q = np.where(H, (L0 + r.sum(0))[None, :] - r, 0)
        t = np.where(H, np.tanh(np.clip(q, -30, 30) / 2), 1.0)
        for i in range(m):
            for j in range(n):
                if H[i, j]:
                    pr = np.prod(np.delete(t[i], j))
                    r[i, j] = 2 * np.arctanh(np.clip(pr, -0.999999, 0.999999))
        snaps.append(L0 + r.sum(0))
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.3))
    ax = ax.ravel()
    syn = lambda L: (H @ (L < 0).astype(int)) % 2
    for k, (a, L) in enumerate(zip(ax, snaps)):
        s = syn(L)
        _draw_tanner(a, H, [_llr_colour(v) for v in L], [f"{v:+.1f}" for v in L],
                     ccol=["#F5B7B1" if x else "#D5F5E3" for x in s],
                     title=("channel only" if k == 0 else f"after {k} iteration" + ("s" if k > 1 else "")) +
                     f": {int(np.sum(L < 0))} wrong")
        print("  gossip", k, np.round(L, 2))
    fig.tight_layout(w_pad=0.2)
    save(fig, "ch15_gossip")


def _peel(H, erased):
    steps, hls = [erased.copy()], []
    e = erased.copy()
    while e.any():
        ready = np.where(H[:, e].sum(1) == 1)[0]
        if len(ready) == 0:
            break
        hl, newe = set(), e.copy()
        for i in ready:
            newe[np.where((H[i] == 1) & e)[0][0]] = False
            for jj in np.where(H[i])[0]:
                hl.add((i, jj))
        hls.append(hl); e = newe; steps.append(e.copy())
    return steps, hls


def fig_peeling():
    """Peeling decoder on the BEC: checks with one unknown neighbour resolve it; a stopping set."""
    H = _HSMALL
    m, n = H.shape
    er = np.zeros(n, bool); er[[0, 4, 9]] = True
    steps, hls = _peel(H, er)
    er2 = np.zeros(n, bool); er2[[1, 4, 7, 9]] = True
    st2, _ = _peel(H, er2)
    stuck = st2[-1]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.3))
    ax = ax.ravel()
    titles = ["(a) 3 erased; the yellow checks see just one '?'", "(b) after round 1: one '?' left, one check ready",
              "(c) after round 2: every bit recovered"]
    for k in range(3):
        e = steps[k]
        ccol = None
        if k < len(hls):
            act = {i for (i, j) in hls[k]}
            ccol = ["#F9E79F" if i in act else "#FBE3D6" for i in range(m)]
        _draw_tanner(ax[k], H, ["#ECEFF1" if x else "#D5F5E3" for x in e], ["?" if x else "ok" for x in e],
                     ccol=ccol, ehl=hls[k] if k < len(hls) else None, title=titles[k])
    sh = {(i, j) for i in range(m) for j in range(n) if H[i, j] and stuck[j]}
    _draw_tanner(ax[3], H, ["#F5B7B1" if x else "#D5F5E3" for x in stuck], ["?" if x else "ok" for x in stuck],
                 ehl=sh, title=f"(d) a stopping set: {int(stuck.sum())} bits no check can resolve")
    print("  stopping set", np.where(stuck)[0])
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "ch15_peeling")


def fig_degrees():
    """Edge vs node perspective of the worked-example ensemble."""
    deg = np.array([2, 3, 8]); lam = np.array([0.3, 0.3, 0.4])
    node = (lam / deg) / np.sum(lam / deg)
    fig, ax = plt.subplots(figsize=(SW, SH))
    xx = np.arange(3)
    ax.bar(xx - 0.18, lam, width=0.36, color=NAVY, label=r"fraction of edges $\lambda_i$")
    ax.bar(xx + 0.18, node, width=0.36, color=ORANGE, label="fraction of nodes")
    for i in range(3):
        ax.text(xx[i] - 0.18, lam[i] + 0.01, f"{lam[i]:.0%}", ha="center", fontsize=6.8, color=NAVY)
        ax.text(xx[i] + 0.18, node[i] + 0.01, f"{node[i]:.0%}", ha="center", fontsize=6.8, color=ORANGE)
    ax.set_xticks(xx); ax.set_xticklabels([f"degree {d}" for d in deg])
    ax.set_ylim(0, 0.62); ax.set_ylabel("fraction")
    ax.legend(fontsize=6.8, loc="upper right")
    save(fig, "ch15_degrees")


def fig_spy():
    """A random sparse H vs a quasi-cyclic H of the same size."""
    r = np.random.default_rng(2)
    Hr = cl.LDPCCode._peg(192, 96, 3, r)
    S = qc_girth8_shifts(3, 6, 32, np.random.default_rng(4))
    Hq = qc_expand(S, 32)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.95))
    for a, H, ttl in [(ax[0], Hr, "PEG (unstructured): 576 ones scattered"),
                      (ax[1], Hq, "quasi-cyclic, Z = 32: 18 shifted identities")]:
        a.spy(H, markersize=1.1, color=NAVY)
        a.set_title(ttl, fontsize=8); a.set_xticks([]); a.set_yticks([])
    for k in range(1, 6):
        ax[1].axvline(k * 32 - 0.5, color=ACCENT, lw=0.4)
    for k in range(1, 3):
        ax[1].axhline(k * 32 - 0.5, color=ACCENT, lw=0.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15_spy")


def fig_flash():
    """TLC NAND: eight threshold-voltage states, hard read references and extra soft reads."""
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    v = np.linspace(-1, 8.2, 1200)
    mus = np.arange(8) * 1.0 + 0.2
    sig = [0.32] + [0.2] * 7
    gray = ["111", "110", "100", "000", "010", "011", "001", "101"]
    for k in range(8):
        y = np.exp(-(v - mus[k]) ** 2 / (2 * sig[k] ** 2))
        ax.fill_between(v, y, color=CYCLE[k % 6], alpha=0.25)
        ax.plot(v, y, color=CYCLE[k % 6], lw=1.0)
        ax.text(mus[k], 1.05, gray[k], ha="center", fontsize=7, family="monospace")
    for k in range(7):
        ref = (mus[k] + mus[k + 1]) / 2
        ax.axvline(ref, color="k", lw=0.9)
        if k == 3:
            for d in (-0.18, 0.18):
                ax.axvline(ref + d, color=ACCENT, lw=0.8, ls="--")
    ax.annotate("extra soft reads\n(refine the LLR)", xy=(3.88, 0.55), xytext=(4.9, 0.72), fontsize=7,
                color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax.set_yticks([]); ax.set_ylim(0, 1.2); ax.set_xlim(-0.8, 7.9)
    ax.set_xlabel("cell threshold voltage (arbitrary units)")
    ax.grid(False)
    save(fig, "ch15_flash")


def fig_polar_step():
    """One polarization step on the BEC: capacity is conserved but split."""
    I = np.linspace(0, 1, 200)
    e = 1 - I
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.plot(I, 1 - e ** 2, color=GREEN, lw=1.8, label=r"$I(W^+)$: the lucky bit")
    ax.plot(I, 1 - (2 * e - e ** 2), color=ACCENT, lw=1.8, label=r"$I(W^-)$: the unlucky bit")
    ax.plot(I, I, color=GRAY, ls=":", lw=1.0, label="no transform")
    ax.plot([0.5, 0.5], [0.25, 0.75], color=NAVY, lw=0.8)
    ax.plot(0.5, 0.75, "o", color=GREEN, ms=4); ax.plot(0.5, 0.25, "o", color=ACCENT, ms=4)
    ax.text(0.53, 0.47, "BEC(0.5):\n0.25 + 0.75\n= 2 x 0.5", fontsize=6.5, color=NAVY)
    ax.set_xlabel("capacity of the channel $I(W)$"); ax.set_ylabel("capacity after one step")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "ch15_polar_step")


def _bec_z(N, eps):
    z = np.array([eps])
    while len(z) < N:
        z = np.concatenate([2 * z - z ** 2, z ** 2])
    return z


def _bec_z_ordered(N, eps):
    """Erasure probabilities in commlib / F^{(x)n} order (index bits: MSB = first split)."""
    if N == 1:
        return np.array([eps])
    return np.concatenate([_bec_z_ordered(N // 2, 2 * eps - eps ** 2), _bec_z_ordered(N // 2, eps ** 2)])


def fig_polar_class():
    """N = 16 bit-channels of BEC(0.5): teach only the brilliant ones."""
    N, K = 16, 8
    z = _bec_z_ordered(N, 0.5)
    info = np.argsort(z)[:K]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    cap = 1 - z
    cols = [NAVY if i in info else "#BDC3C7" for i in range(N)]
    ax.bar(np.arange(N), cap, color=cols, width=0.75)
    for i in range(N):
        ax.text(i, cap[i] + 0.02, f"{cap[i]:.2f}", ha="center", fontsize=5.8, rotation=90)
    ax.axhline(0.5, color=ACCENT, ls=":", lw=0.9)
    ax.text(8, 0.54, "original channel", fontsize=6.5, color=ACCENT, ha="center")
    ax.set_xticks(np.arange(N)); ax.set_xticklabels([f"$u_{{{i}}}$" for i in range(N)], fontsize=7)
    ax.set_ylabel("capacity of bit-channel"); ax.set_ylim(0, 1.18)
    ax.bar([0], [0], color=NAVY, label="information (the 8 best)")
    ax.bar([0], [0], color="#BDC3C7", label="frozen to 0")
    ax.legend(fontsize=6.8, loc="upper left", ncol=2)
    print("  info set", sorted(info.tolist()))
    save(fig, "ch15_polar_class")


def _polar_enc(u):
    u = np.array(u, int) % 2
    N = len(u)
    if N == 1:
        return u
    h = N // 2
    return np.concatenate([_polar_enc((u[:h] + u[h:]) % 2), _polar_enc(u[h:])])


def _polar_llr(L, prefix, i):
    N = len(L)
    if N == 1:
        return L[0]
    h = N // 2
    a, b = L[:h], L[h:]
    if i < h:
        f = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        return _polar_llr(f, prefix[:i], i)
    s = _polar_enc(prefix[:h])
    return _polar_llr(b + (1 - 2 * s) * a, prefix[h:i], i - h)


def fig_scl_tree():
    """SC list decoding (L = 4) of an N = 16 polar code: a shortlist of suspects."""
    N, K, Lmax = 16, 8, 4
    z = _bec_z_ordered(N, 0.5)
    info = set(np.argsort(z)[:K].tolist())
    best = None
    for seed in range(400):
        r = np.random.default_rng(seed)
        u = np.zeros(N, int)
        for i in sorted(info):
            u[i] = r.integers(0, 2)
        x = _polar_enc(u)
        L = awgn_llr(x, 1.5, K / N, r)
        # SC
        pre = []
        scm, scpm = [], 0.0
        for i in range(N):
            l = _polar_llr(L, np.array(pre, int), i)
            pre.append(0 if i not in info else int(l < 0))
            if i not in info and l < 0:
                scpm += abs(l)
            if i in info:
                scm.append(scpm)
        sc_ok = np.array_equal(pre, u)
        # SCL with history
        paths = [([], 0.0, 0)]          # (prefix, metric, id)
        nid = 1
        hist = []                       # (i, parent_id, id, bit, metric, survived)
        for i in range(N):
            new = []
            for pre_, pm, pid in paths:
                l = _polar_llr(L, np.array(pre_, int), i)
                if i not in info:
                    new.append((pre_ + [0], pm + (abs(l) if l < 0 else 0), pid, 0))
                else:
                    for bit in (0, 1):
                        pen = abs(l) if (bit == 1) != (l < 0) else 0
                        new.append((pre_ + [bit], pm + pen, pid, bit))
            new.sort(key=lambda t: t[1])
            keep = new[:Lmax]
            paths2 = []
            for k_, (p_, pm, pid, bit) in enumerate(new):
                if i in info:
                    hist.append((i, pid, nid, bit, pm, k_ < Lmax, p_))
                if k_ < Lmax:
                    paths2.append((p_, pm, nid if i in info else pid))
                if i in info:
                    nid += 1
            paths = paths2
        win_ok = np.array_equal(paths[0][0], u)
        inlist = any(np.array_equal(p[0], u) for p in paths)
        if (not sc_ok) and win_ok:
            best = (seed, u, hist, paths, scm, pre)
            break
    seed, u, hist, paths, scm, scdec = best
    print("  scl seed", seed)
    infol = sorted(info)
    col = {i: k + 1 for k, i in enumerate(infol)}
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    pos = {0: (0, 0.0)}
    # place nodes: x = column, y = metric (log-ish)
    for (i, pid, myid, bit, pm, surv, p_) in hist:
        pos[myid] = (col[i], pm)
    truth_ids = set()
    for (i, pid, myid, bit, pm, surv, p_) in hist:
        if np.array_equal(p_, u[:i + 1]):
            truth_ids.add(myid)
    for (i, pid, myid, bit, pm, surv, p_) in hist:
        x0, y0 = pos.get(pid, (0, 0)); x1, y1 = pos[myid]
        tr = myid in truth_ids
        ax.plot([x0, x1], [y0, y1], color=GREEN if tr else (NAVY if surv else "#D5D8DC"),
                lw=2.0 if tr else (0.9 if surv else 0.6), zorder=2 if tr else 1)
        ax.plot(x1, y1, "o" if surv else "x", color=GREEN if tr else (NAVY if surv else "#ABB2B9"),
                ms=4 if surv else 3.5, zorder=3)
    xs = np.arange(1, K + 1)
    scm = np.array(scm)
    tm = [h[4] for h in hist if h[2] in truth_ids][-1]
    ax.plot(np.r_[0, xs], np.r_[0, scm], "--", color=ACCENT, lw=1.5, zorder=4)
    first_bad = next(k for k, i in enumerate(infol) if scdec[i] != u[i])
    ax.annotate(f"SC's greedy path: wrong at $u_{{{infol[first_bad]}}}$ by a whisker; the frozen bit after it\n"
                f"charges a penalty, but SC can never go back (final metric {scm[-1]:.2f} vs {tm:.2f})",
                xy=(first_bad + 2, scm[first_bad + 1]), xytext=(1.3, 5.2), fontsize=6.6, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax.set_ylim(4.9, 12.5); ax.set_xlim(0.7, K + 0.3)
    ax.text(K + 0.2, 12.0, "pruned paths run off the top", fontsize=6.3, color=GRAY, ha="right")
    print("  SC metrics", np.round(scm, 2), "first wrong info index", infol[first_bad])
    ax.set_xticks(range(0, K + 1)); ax.set_xticklabels(["start"] + [f"$u_{{{i}}}$" for i in infol], fontsize=7)
    ax.set_ylabel("path metric (penalty)")
    ax.set_xlabel("information bits, decided in order")
    ax.plot([], [], color=GREEN, lw=2, label="the true message")
    ax.plot([], [], "o-", color=NAVY, ms=3, lw=0.9, label="kept on the list (L = 4)")
    ax.plot([], [], "x", color="#ABB2B9", label="pruned")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("N = 16, K = 8 at 1.5 dB: SC alone fails here, the list keeps the right path", fontsize=8)
    save(fig, "ch15_scl_tree")


def fig_reliability():
    """GA reliability order vs the simple polarization-weight rule (N = 256)."""
    N = 256
    mu = ga_construction(N, 1.0, 0.5)
    ga_rank = np.argsort(np.argsort(mu))
    n = int(np.log2(N))
    # commlib order: index bit (n-1-j) set means the 'plus' branch at level j
    pw = np.array([sum(((i >> (n - 1 - j)) & 1) * 2 ** ((n - 1 - j) / 4) for j in range(n)) for i in range(N)])
    pw_rank = np.argsort(np.argsort(pw))
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.plot(pw_rank, ga_rank, ".", ms=2.5, color=NAVY)
    ax.plot([0, N], [0, N], color=GRAY, lw=0.6, ls=":")
    ax.axhline(N / 2, color=ACCENT, lw=0.6); ax.axvline(N / 2, color=ACCENT, lw=0.6)
    agree = np.mean((ga_rank >= N / 2) == (pw_rank >= N / 2))
    ax.text(5, 220, f"same K = 128 set:\n{agree:.0%} agreement", fontsize=6.8, color=ACCENT)
    ax.set_xlabel("rank by polarization weight"); ax.set_ylabel("rank by GA at 1 dB")
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    print("  pw/ga agreement", agree)
    save(fig, "ch15_reliability")


def fig_5g_codes():
    """Which 5G NR code protects which payload (TS 38.212)."""
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    bars = [(1, 2.95, "rep./\nsimplex", GRAY, 3), (3, 11.9, "Reed-\nMuller", PURPLE, 3),
            (12, 1706, "polar (+ PC bits or CRC-11)", "#2E86C1", 3),
            (1, 140, "polar + CRC-24 (downlink control, DCI)", NAVY, 2),
            (32, 33, "PBCH: 32 bits, polar", NAVY, 1),
            (24, 1.3e6, "LDPC BG2 / BG1 (data, PDSCH/PUSCH), segmented into code blocks of up to 8448 bits", GREEN, 0)]
    for lo, hi, lab, c, row in bars:
        ax.add_patch(plt.Rectangle((lo, row - 0.35), hi - lo, 0.7, color=c, alpha=0.75, lw=0.5, ec="white"))
        inside = hi / lo > 2
        xm = np.sqrt(lo * hi) if inside else hi * 1.15
        ax.text(xm, row, lab, ha="center" if inside else "left", va="center",
                fontsize=5.6 if hi / lo < 5 else 6.3, color="white" if inside else c, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.8, 2e6); ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["data", "broadcast", "downlink\ncontrol", "uplink\ncontrol"], fontsize=7)
    ax.set_xlabel("payload (bits)")
    ax.grid(axis="y", visible=False)
    save(fig, "ch15_5g_codes")


def fig_rv_ring():
    """The NR circular buffer (BG1) and the four redundancy-version starting points."""
    fig, ax = plt.subplots(figsize=(SW, SH + 0.3), subplot_kw=dict(projection="polar"))
    tot = 66.0
    th = lambda f: np.pi / 2 - 2 * np.pi * f
    ax.bar(th(np.array([10 / tot])), 0.4, width=2 * np.pi * 20 / tot, bottom=0.8, color=NAVY, alpha=0.8)
    ax.bar(th(np.array([(20 + 23) / tot])), 0.4, width=2 * np.pi * 46 / tot, bottom=0.8, color=ORANGE, alpha=0.6)
    for rv, f in [(0, 0), (1, 17 / 66), (2, 33 / 66), (3, 56 / 66)]:
        ax.plot([th(f), th(f)], [0.7, 1.32], color=ACCENT, lw=1.3)
        ax.text(th(f), 1.58, f"RV{rv}", ha="center", va="center", fontsize=7.5, color=ACCENT, fontweight="bold")
    fs = np.linspace(0, 0.47, 50)
    ax.plot(th(fs), np.full(50, 0.6), color=GREEN, lw=1.6)
    ax.annotate("", xy=(th(0.47), 0.6), xytext=(th(0.45), 0.6), arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.text(0, 0, "1st transmission\nat rate 1/2", ha="center", va="center", fontsize=6.3, color=GREEN)
    ax.text(th(9 / 66), 1.0, "systematic", ha="center", va="center", fontsize=5.8, color="white",
            rotation=np.degrees(th(9 / 66)) - 90)
    ax.text(th(45 / 66), 0.98, "parity", ha="center", va="center", fontsize=6.5, color="k")
    ax.set_ylim(0, 1.72); ax.axis("off")
    save(fig, "ch15_rv_ring")


def fig_optical_ncg():
    """Net coding gain of optical FEC generations at 1e-15 (values quoted in the text)."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    labs = ["RS(255,239)\nG.709", "G.975.1\nconcatenated", "staircase\nG.709.2"]
    vals = [6.0, 8.5, 9.4]
    err = [0, 0.5, 0]
    ax.bar(range(3), vals, color=[GRAY, ORANGE, NAVY], width=0.6)
    ax.errorbar([1], [8.5], yerr=[0.5], color="k", capsize=3, lw=0.8)
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + 0.6, ("8-9" if i == 1 else f"{v_:.1f}") + " dB", ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("net coding gain at $10^{-15}$ (dB)"); ax.set_ylim(0, 11.5)
    ax.set_title("all at 6.7 % overhead, hard decisions", fontsize=7.8)
    save(fig, "ch15_optical_ncg")


def fig_grand():
    """Hard-decision GRAND on a random (32, 26) code: guesses needed per block."""
    r = np.random.default_rng(3)
    n, k = 32, 26
    P = r.integers(0, 2, (k, n - k))
    Hm = np.concatenate([P.T, np.eye(n - k, dtype=int)], 1)
    from itertools import combinations
    pats = [()]
    for w in (1, 2, 3):
        pats += list(combinations(range(n), w))
    synd_of = lambda e: tuple((Hm[:, list(e)].sum(1) % 2) if e else np.zeros(n - k, int))
    S = np.array([synd_of(e) for e in pats])
    guesses = {}
    for p, c in [(0.01, GREEN), (0.03, ORANGE)]:
        g = []
        for _ in range(3000):
            e = r.random(n) < p
            s = (Hm @ e.astype(int)) % 2
            hit = np.where(np.all(S == s, 1))[0]
            g.append(hit[0] + 1 if len(hit) else len(pats) + 1)
        guesses[p] = (np.array(g), c)
    fig, ax = plt.subplots(figsize=(SW, SH))
    bins = np.logspace(0, np.log10(len(pats) + 2), 30)
    for p, (g, c) in guesses.items():
        ax.hist(g, bins=bins, color=c, alpha=0.55, label=f"BSC p = {p}: mean {g.mean():.0f} guesses")
        print(f"  grand p={p}: mean {g.mean():.1f}, median {np.median(g)}")
    ax.set_xscale("log"); ax.set_xlabel("guesses until a codeword appears")
    ax.set_ylabel("blocks"); ax.legend(fontsize=6.4, loc="upper right")
    ax.axvline(1 + n, color=GRAY, lw=0.6, ls=":"); ax.axvline(1 + n + n * (n - 1) / 2, color=GRAY, lw=0.6, ls=":")
    ax.text(1 + n, ax.get_ylim()[1] * 0.55, " all 1-bit\n patterns", fontsize=6, color=GRAY)
    ax.text(1 + n + n * (n - 1) / 2, ax.get_ylim()[1] * 0.4, " all 2-bit\n patterns", fontsize=6, color=GRAY)
    save(fig, "ch15_grand")


def fig_deployments():
    """Where the three families went, by first standard (approximate years)."""
    rows = [("UMTS / HSPA (turbo)", 1999, ORANGE), ("CCSDS deep space (turbo)", 1999, ORANGE),
            ("WiMAX 802.16e (turbo, LDPC)", 2005, ORANGE), ("DVB-S2 (LDPC + BCH)", 2005, GREEN),
            ("10GBASE-T (LDPC)", 2006, GREEN), ("LTE (turbo)", 2008, ORANGE),
            ("Wi-Fi 802.11n (LDPC)", 2009, GREEN), ("SSD controllers (LDPC)", 2012, GREEN),
            ("5G NR data (LDPC)", 2018, GREEN), ("5G NR control (polar)", 2018, "#2E86C1")]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    for i, (lab, y0, c) in enumerate(rows[::-1]):
        ax.barh(i, 2026 - y0, left=y0, color=c, alpha=0.75, height=0.62)
        ax.text(y0 - 0.4, i, lab, ha="right", va="center", fontsize=6.8)
    ax.set_yticks([]); ax.set_xlim(1983, 2026); ax.set_xlabel("year of first standard (approximate)")
    for lab, c in [("turbo", ORANGE), ("LDPC", GREEN), ("polar", "#2E86C1")]:
        ax.barh([-5], [0], color=c, label=lab)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", visible=False)
    save(fig, "ch15_deployments")


def fig_thresholds():
    """How far from capacity: BP thresholds of LDPC ensembles quoted in the text (rate 1/2, BI-AWGN)."""
    labs = ["(3,6) regular", "practical\nirregular", "optimised,\nmax degree 100", "Chung et al.,\nmax degree 8000"]
    gaps = [1.10 - 0.19, 0.35, 0.06, 0.0045]
    fig, ax = plt.subplots(figsize=(SW, SH))
    cols = [GRAY, ORANGE, GREEN, NAVY]
    ax.barh(range(4), gaps, color=cols, height=0.6)
    ax.errorbar([0.35], [1], xerr=[0.15], color="k", capsize=3, lw=0.8)
    for i, g in enumerate(gaps):
        ax.text(g + 0.03 + (0.2 if i == 1 else 0), i, ("0.2-0.5" if i == 1 else f"{g:g}") + " dB", va="center", fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("BP threshold minus capacity limit (dB)"); ax.set_xlim(0, 1.2)
    ax.grid(axis="y", visible=False)
    save(fig, "ch15_thresholds")


def fig_qpp40():
    """The K = 40 LTE QPP interleaver drawn as a wiring diagram."""
    K = 40
    p = tb.qpp_interleaver(K, 3, 10)
    fig, ax = plt.subplots(figsize=(W1, 1.25))
    for i in range(K):
        c = ACCENT if i < 6 else "#B0BEC5"
        ax.plot([i, p[i]], [1, 0], color=c, lw=1.1 if i < 6 else 0.6, zorder=2 if i < 6 else 1)
    ax.scatter(range(K), np.ones(K), s=9, color=NAVY, zorder=3)
    ax.scatter(range(K), np.zeros(K), s=9, color=NAVY, zorder=3)
    ax.text(-1.2, 1, "$i$", ha="right", va="center", fontsize=8)
    ax.text(-1.2, 0, r"$\pi(i)$", ha="right", va="center", fontsize=8)
    ax.text(41, 0.5, "first six inputs (red):\n0, 1, 2, 3, 4, 5 go to\n0, 13, 6, 19, 12, 25", fontsize=6.8,
            va="center", color=ACCENT)
    ax.set_xlim(-3, 50); ax.set_ylim(-0.15, 1.15); ax.axis("off")
    save(fig, "ch15_qpp40")


def fig_witnesses():
    """Independent evidence adds in the LLR domain."""
    fig, ax = plt.subplots(figsize=(SW, 1.7))
    vals = [np.log(3), np.log(3), 2 * np.log(3)]
    labs = ["witness 1\n(3 : 1)", "witness 2\n(3 : 1)", "together\n(9 : 1)"]
    ax.barh([2, 1, 0], vals, color=[ORANGE, GREEN, NAVY], height=0.6)
    for y, v in zip([2, 1, 0], vals):
        ax.text(v + 0.05, y, f"L = {v:.2f}", va="center", fontsize=7)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlim(0, 3.0); ax.set_xlabel("LLR (evidence that the bit is 0)", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "ch15_witnesses")


CONCEPTS = [fig_qpp40, fig_witnesses, fig_thresholds, fig_timeline, fig_gap_chase, fig_llr, fig_boxplus, fig_maxstar, fig_bcjr_heat, fig_turbo_inside,
            fig_rsc_impulse, fig_interleavers, fig_anatomy, fig_gossip, fig_peeling, fig_degrees, fig_spy,
            fig_flash, fig_polar_step, fig_polar_class, fig_scl_tree, fig_reliability, fig_5g_codes,
            fig_rv_ring, fig_optical_ncg, fig_grand, fig_deployments]


if __name__ == "__main__":
    todo = sys.argv[1:]
    if todo == ["concepts"]:
        for f in CONCEPTS:
            t0 = time.time(); f(); print(f"[{f.__name__}] {time.time() - t0:.0f}s")
        sys.exit(0)
    fns = [checknode, polarization, harq, nr_basegraph, de_bec, de_awgn, spatial_coupling,
           exit_chart, turbo_iters, turbo_length, turbo_floor, ldpc_decoders, polar_decoders,
           short_compare, gap_length] + CONCEPTS
    for f in fns:
        if not todo or f.__name__ in todo:
            t0 = time.time()
            f()
            print(f"[{f.__name__}] {time.time() - t0:.0f}s")
